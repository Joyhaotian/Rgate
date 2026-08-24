#!/usr/bin/env python3
"""Replay nuScenes detection metrics from scene-local event traces.

The technical-pilot scope is fixed to the first eight registered scene draws
and is never scientific evidence.  The same exact core can later process the
registered 1000 draws once all five-seed event bundles are independently
closed against the official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


PLAN_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_plan_v1"
EVENT_SCHEMA = "nuscenes_scene_metric_event_cache_v1"
EVENT_CHECK_SCHEMA = "nuscenes_scene_metric_event_cache_independent_check_v1"
EVENT_CHECK_STATUS = "passed_independent_raw_event_and_official_metric_closure"
REPORT_SCHEMA = "nuscenes_paired_scene_bootstrap_report_v1"
METRICS = ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err")


class BootstrapError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(BootstrapError("non-finite JSON")))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_record(path: Path, expected_sha256: str = "") -> Dict[str, Any]:
    logical = path.absolute()
    require(logical.is_file(), "missing input: %s" % logical)
    before = logical.stat()
    digest = sha(logical)
    after = logical.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "input changed while hashing: %s" % logical,
    )
    if expected_sha256:
        require(digest == expected_sha256, "input SHA mismatch: %s" % logical)
    return {"path": str(logical), "size_bytes": before.st_size, "sha256": digest}


def expected_manifest_arm_label(alias: str) -> str:
    seed0_prefix = "seed_00_"
    if alias.startswith(seed0_prefix) and alias[len(seed0_prefix):] in ("E3", "E4", "E4NR"):
        return alias[len(seed0_prefix):]
    return alias


def parse_labeled_paths(values: Sequence[str], description: str) -> Sequence[Tuple[str, Path]]:
    rows = []
    seen = set()
    for raw in values:
        alias, separator, path = raw.partition("=")
        require(bool(separator) and bool(alias) and bool(path) and alias not in seen, "invalid/duplicate %s argument" % description)
        seen.add(alias)
        rows.append((alias, Path(path).absolute()))
    return rows


def bind_event_inputs(manifest_values: Sequence[str], receipt_values: Sequence[str]) -> Sequence[Tuple[str, Path, Path]]:
    manifests = parse_labeled_paths(manifest_values, "event manifest")
    receipts = parse_labeled_paths(receipt_values, "event-check receipt")
    require([label for label, _path in receipts] == [label for label, _path in manifests], "event manifest/receipt label order mismatch")
    return [
        (label, manifest_path, receipt_path)
        for (label, manifest_path), (_receipt_label, receipt_path) in zip(manifests, receipts)
    ]


def validate_event_check_receipt(alias: str, manifest_path: Path, manifest: Mapping[str, Any], receipt_path: Path) -> Dict[str, Any]:
    expected_arm = expected_manifest_arm_label(alias)
    require(manifest.get("arm_label") == expected_arm, "event alias/manifest arm-label mismatch: %s" % alias)
    receipt_record = stable_record(receipt_path)
    receipt = load_json(receipt_path)
    require(
        receipt.get("schema_version") == EVENT_CHECK_SCHEMA
        and receipt.get("status") == EVENT_CHECK_STATUS
        and receipt.get("passed") is True,
        "event-check receipt did not pass: %s" % alias,
    )
    require(receipt.get("arm_label") == expected_arm, "event alias/receipt arm-label mismatch: %s" % alias)
    require(receipt.get("event_manifest") == stable_record(manifest_path), "event-check receipt/manifest identity mismatch: %s" % alias)
    return receipt_record


def array_sha(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def threshold_key(value: float) -> str:
    return ("%.1f" % value).replace(".", "p")


def no_predictions() -> Dict[str, np.ndarray]:
    return {
        "recall": np.linspace(0, 1, 101),
        "precision": np.zeros(101),
        "confidence": np.zeros(101),
        "trans_err": np.ones(101),
        "vel_err": np.ones(101),
        "scale_err": np.ones(101),
        "orient_err": np.ones(101),
        "attr_err": np.ones(101),
    }


def cummean(values: np.ndarray) -> np.ndarray:
    if int(np.isnan(values).sum()) == len(values):
        return np.ones(len(values))
    sums = np.nancumsum(values.astype(float))
    counts = np.cumsum(~np.isnan(values))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def metric_data(
    arrays: Mapping[str, np.ndarray],
    class_name: str,
    threshold: float,
    draw: np.ndarray,
    scene_count: int,
) -> Dict[str, np.ndarray]:
    gt = arrays["gt__%s" % class_name]
    npos = int(np.asarray(gt, dtype=np.int64)[draw].sum())
    if npos == 0:
        return no_predictions()
    event_scene = arrays["scene__%s" % class_name]
    score = arrays["score__%s" % class_name]
    tp_all = arrays["tp__%s__%s" % (class_name, threshold_key(threshold))]
    # Expand clones around the locked original prediction ordinal.  Grouping
    # events by the sampled scene order would silently renumber equal-score
    # predictions and therefore change the devkit's reverse (score, index)
    # tie rule.  Each original event is instead repeated by the multiplicity
    # of its scene in the draw, with clone occurrences kept contiguous.
    multiplicity = np.bincount(np.asarray(draw, dtype=np.int64), minlength=scene_count)
    selected = np.repeat(
        np.arange(len(event_scene), dtype=np.int64),
        multiplicity[np.asarray(event_scene, dtype=np.int64)],
    )
    if len(selected) == 0:
        return no_predictions()
    selected_scores = score[selected].astype(np.float64, copy=False)
    synthetic_index = np.arange(len(selected), dtype=np.int64)
    # Equivalent to Python sorted((score, index))[::-1].  np.lexsort uses the
    # final key as primary, so sorting (-index, -score) gives score desc then
    # synthetic index desc, including exact score ties.
    order = np.lexsort((-synthetic_index, -selected_scores))
    selected = selected[order]
    sorted_scores = selected_scores[order]
    tp_flags = tp_all[selected].astype(np.int64, copy=False)
    if int(tp_flags.sum()) == 0:
        return no_predictions()
    fp_flags = 1 - tp_flags
    tp_cum = np.cumsum(tp_flags).astype(float)
    fp_cum = np.cumsum(fp_flags).astype(float)
    precision = tp_cum / (fp_cum + tp_cum)
    recall = tp_cum / float(npos)
    recall_grid = np.linspace(0, 1, 101)
    precision_grid = np.interp(recall_grid, recall, precision, right=0)
    confidence_grid = np.interp(recall_grid, recall, sorted_scores, right=0)
    result = {"recall": recall_grid, "precision": precision_grid, "confidence": confidence_grid}
    match_mask = tp_flags.astype(bool)
    match_confidence = sorted_scores[match_mask]
    for metric in METRICS:
        key = "err__%s__%s__%s" % (class_name, threshold_key(threshold), metric)
        if key in arrays:
            match_values = arrays[key][selected][match_mask].astype(np.float64, copy=False)
            means = cummean(match_values)
            result[metric] = np.interp(confidence_grid[::-1], match_confidence[::-1], means[::-1])[::-1]
        else:
            result[metric] = np.ones(101)
    return result


def calc_ap(data: Mapping[str, np.ndarray], min_recall: float, min_precision: float) -> float:
    precision = np.copy(data["precision"])[round(100 * min_recall) + 1:]
    precision -= min_precision
    precision[precision < 0] = 0
    return float(np.mean(precision)) / (1.0 - min_precision)


def calc_tp(data: Mapping[str, np.ndarray], min_recall: float, metric: str) -> float:
    first = round(100 * min_recall) + 1
    nonzero = np.nonzero(data["confidence"])[0]
    last = int(nonzero[-1]) if len(nonzero) else 0
    return 1.0 if last < first else float(np.mean(data[metric][first:last + 1]))


def summarize(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any], draw: np.ndarray) -> Dict[str, Any]:
    classes = list(manifest["classes"])
    thresholds = [float(value) for value in manifest["distance_thresholds"]]
    min_recall, min_precision = float(manifest["min_recall"]), float(manifest["min_precision"])
    tp_threshold = float(manifest["tp_distance_threshold"])
    label_aps, label_tp = {}, {}
    for class_name in classes:
        label_aps[class_name] = {}
        data_by_threshold = {}
        for threshold in thresholds:
            data = metric_data(arrays, class_name, threshold, draw, int(manifest["scene_count"]))
            data_by_threshold[threshold] = data
            label_aps[class_name][str(threshold)] = calc_ap(data, min_recall, min_precision)
        label_tp[class_name] = {}
        tp_data = data_by_threshold[tp_threshold]
        for metric in METRICS:
            if class_name == "traffic_cone" and metric in ("attr_err", "vel_err", "orient_err"):
                value = float("nan")
            elif class_name == "barrier" and metric in ("attr_err", "vel_err"):
                value = float("nan")
            else:
                value = calc_tp(tp_data, min_recall, metric)
            label_tp[class_name][metric] = value
    mean_dist = {name: float(np.mean(list(values.values()))) for name, values in label_aps.items()}
    mean_ap = float(np.mean(list(mean_dist.values())))
    tp_errors = {}
    for metric in METRICS:
        tp_errors[metric] = float(np.nanmean([label_tp[name][metric] for name in classes]))
    tp_scores = {metric: max(0.0, 1.0 - value) for metric, value in tp_errors.items()}
    nd_score = float((5.0 * mean_ap + sum(tp_scores.values())) / 10.0)
    serializable_label_tp = {
        class_name: {
            metric: (None if math.isnan(value) else value)
            for metric, value in values.items()
        }
        for class_name, values in label_tp.items()
    }
    return {
        "mean_ap": mean_ap,
        "nd_score": nd_score,
        "mAVE": tp_errors["vel_err"],
        "tp_errors": tp_errors,
        "label_aps": label_aps,
        "label_tp_errors": serializable_label_tp,
    }


def load_bundle(manifest_path: Path) -> Tuple[Mapping[str, Any], Dict[str, np.ndarray]]:
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == EVENT_SCHEMA and manifest.get("status") == "passed_official_event_trace_closure", "event manifest schema/status mismatch")
    npz_spec = manifest["npz"]
    npz_path = (manifest_path.parent / npz_spec["path"]).absolute()
    require(npz_path.is_file() and npz_path.stat().st_size == npz_spec["size_bytes"] and sha(npz_path) == npz_spec["sha256"], "event NPZ identity mismatch")
    with np.load(str(npz_path), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    require(set(arrays) == set(manifest["arrays"]), "event array key set mismatch")
    for key, spec in manifest["arrays"].items():
        array = arrays[key]
        require(str(array.dtype) == spec["dtype"] and list(array.shape) == spec["shape"] and array_sha(array) == spec["sha256"], "event array identity mismatch: %s" % key)
    return manifest, arrays


def close(reference: float, observed: float, tolerance: float = 1e-12) -> bool:
    return abs(float(reference) - float(observed)) <= tolerance


def run(plan: Mapping[str, Any], draw_bytes: bytes, bundles: Mapping[str, Tuple[Mapping[str, Any], Mapping[str, np.ndarray], Mapping[str, Any]]], scope: str) -> Dict[str, Any]:
    scene_count = int(plan["scene_bootstrap"]["scene_count"])
    all_draws = np.frombuffer(draw_bytes, dtype="<u2").reshape(-1, scene_count)
    replicates = 8 if scope == "technical_pilot_B8" else int(plan["scene_bootstrap"]["registered_replicates"])
    require(scope in ("technical_pilot_B8", "registered_B1000"), "unknown scope")
    require(len(all_draws) >= replicates, "draw matrix too short")
    expected_draw = plan["scene_bootstrap"]["draw_generator"]
    require(len(draw_bytes) == int(expected_draw["draw_matrix_size_bytes"]), "draw matrix size differs from plan")
    require(hashlib.sha256(draw_bytes).hexdigest() == expected_draw["expected_draw_matrix_sha256"], "draw matrix bytes differ from plan")
    require(all_draws[0, :12].tolist() == expected_draw["first_replicate_first_12_indices"], "draw prefix differs from plan")
    scene_hashes = {manifest.get("scene_tokens_sha256") for manifest, _arrays, _receipt in bundles.values()}
    require(len(scene_hashes) == 1 and None not in scene_hashes, "event bundles do not share one canonical scene-token order")
    outputs = {}
    for label, (manifest, arrays, receipt_record) in bundles.items():
        require(manifest["scene_count"] == scene_count, "event scene extent mismatch")
        full = summarize(arrays, manifest, np.arange(scene_count, dtype=np.uint16))
        official = manifest["official_summary"]
        closure = {
            "mean_ap_abs_diff": abs(full["mean_ap"] - official["mean_ap"]),
            "nd_score_abs_diff": abs(full["nd_score"] - official["nd_score"]),
            "mAVE_abs_diff": abs(full["mAVE"] - official["mAVE"]),
        }
        require(max(closure.values()) <= 1e-12 and manifest["metric_details_max_abs_diff"] <= 1e-12, "all-ones official closure failed: %s" % label)
        replicate_rows = []
        for index in range(replicates):
            values = summarize(arrays, manifest, all_draws[index])
            replicate_rows.append({"replicate": index, "mean_ap": values["mean_ap"], "nd_score": values["nd_score"], "mAVE": values["mAVE"]})
        manifest_path = Path(manifest["self_path"])
        outputs[label] = {
            "event_manifest": stable_record(manifest_path),
            "event_check_receipt": dict(receipt_record),
            "all_ones": full,
            "official_closure": closure,
            "replicates": replicate_rows,
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed_technical_event_replay" if scope == "technical_pilot_B8" else "passed_registered_bootstrap_replay",
        "scope": scope,
        "scientific_evidence": scope == "registered_B1000",
        "replicate_count": replicates,
        "scene_count": scene_count,
        "scene_tokens_sha256": next(iter(scene_hashes)),
        "arms": outputs,
        "per_scene_AP_average_used": False,
        "official_evaluator_calls": 0,
        "official_algorithm_equivalent_internal_replay": True,
        "draw_prefix_sha256": hashlib.sha256(draw_bytes[: replicates * scene_count * 2]).hexdigest(),
    }


def publish(path: Path, report: Mapping[str, Any]) -> None:
    require(path.parent.is_dir() and not os.path.lexists(str(path)), "output contract failed")
    data = canonical(report)
    fd, name = tempfile.mkstemp(prefix=".%s.tmp." % path.name, dir=str(path.parent))
    temporary, owned = Path(name), False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(str(temporary), str(path)); owned = True
    except Exception:
        if owned and path.exists():
            path.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-bootstrap-runner-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--draw-matrix", required=True)
    parser.add_argument("--expected-draw-sha256", required=True)
    parser.add_argument("--event-manifest", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--event-check-receipt", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--scope", choices=("technical_pilot_B8", "registered_B1000"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        plan_path, draw_path = Path(args.plan).absolute(), Path(args.draw_matrix).absolute()
        plan_record = stable_record(plan_path, args.expected_plan_sha256)
        draw_record = stable_record(draw_path, args.expected_draw_sha256)
        plan = load_json(plan_path)
        require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
        report = None
        for label, manifest_path, receipt_path in bind_event_inputs(args.event_manifest, args.event_check_receipt):
            manifest, arrays = load_bundle(manifest_path)
            receipt_record = validate_event_check_receipt(label, manifest_path, manifest, receipt_path)
            manifest = dict(manifest); manifest["self_path"] = str(manifest_path)
            partial = run(plan, draw_path.read_bytes(), {label: (manifest, arrays, receipt_record)}, args.scope)
            if report is None:
                report = partial
            else:
                for key, value in partial.items():
                    if key != "arms":
                        require(report[key] == value, "cross-arm report metadata drift: %s" % key)
                report["arms"].update(partial["arms"])
            del arrays
        require(report is not None, "no event bundles supplied")
        report["inputs"] = {"plan": plan_record, "draw_matrix": draw_record}
        report["implementation"] = {"bootstrap_runner": stable_record(Path(__file__).absolute(), args.expected_bootstrap_runner_sha256)}
        publish(Path(args.output).absolute(), report)
    except (BootstrapError, OSError, ValueError) as exc:
        print("FAILED: %s" % exc, file=os.sys.stderr); return 1
    print(str(Path(args.output).absolute())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
