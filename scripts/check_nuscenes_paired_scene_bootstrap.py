#!/usr/bin/env python3
"""Independently verify a paired scene-bootstrap event-replay report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


PLAN_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_plan_v1"
EVENT_SCHEMA = "nuscenes_scene_metric_event_cache_v1"
EVENT_CHECK_SCHEMA = "nuscenes_scene_metric_event_cache_independent_check_v1"
EVENT_CHECK_STATUS = "passed_independent_raw_event_and_official_metric_closure"
REPORT_SCHEMA = "nuscenes_paired_scene_bootstrap_report_v1"
CHECK_SCHEMA = "nuscenes_paired_scene_bootstrap_independent_check_v1"
ERRORS = ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err")


class CheckError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise CheckError("non-finite JSON constant: %s" % value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_record(path: Path, expected_sha256: str = "") -> Dict[str, Any]:
    logical = path.absolute()
    require(logical.is_file(), "missing input: %s" % logical)
    before = logical.stat()
    digest = sha256_file(logical)
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def load_checker_core(path: Path, expected_sha256: str):
    record = stable_record(path, expected_sha256)
    spec = importlib.util.spec_from_file_location("rgate_independent_event_core", str(path))
    require(spec is not None and spec.loader is not None, "cannot import independent event checker core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return record, module


def load_bundle(manifest_path: Path) -> Tuple[Mapping[str, Any], Dict[str, np.ndarray]]:
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == EVENT_SCHEMA, "event manifest schema mismatch")
    require(manifest.get("status") == "passed_official_event_trace_closure", "event manifest status mismatch")
    npz_spec = manifest["npz"]
    npz_path = (manifest_path.parent / npz_spec["path"]).absolute()
    stable_record(npz_path, npz_spec["sha256"])
    require(npz_path.stat().st_size == npz_spec["size_bytes"], "event NPZ size mismatch")
    with np.load(str(npz_path), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    require(set(arrays) == set(manifest["arrays"]), "event array key set mismatch")
    for key, expected in manifest["arrays"].items():
        value = np.ascontiguousarray(arrays[key])
        observed = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        require(observed == expected, "event array identity mismatch: %s" % key)
    return manifest, arrays


def calc_ap(data: Mapping[str, np.ndarray], min_recall: float, min_precision: float) -> float:
    precision = np.copy(data["precision"])[round(100 * min_recall) + 1 :]
    precision -= min_precision
    precision[precision < 0] = 0
    return float(np.mean(precision)) / (1.0 - min_precision)


def calc_tp(data: Mapping[str, np.ndarray], min_recall: float, metric: str) -> float:
    first = round(100 * min_recall) + 1
    nonzero = np.nonzero(data["confidence"])[0]
    last = int(nonzero[-1]) if len(nonzero) else 0
    return 1.0 if last < first else float(np.mean(data[metric][first : last + 1]))


def summarize(core: Any, arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any], draw: np.ndarray) -> Dict[str, Any]:
    classes = list(manifest["classes"])
    thresholds = [float(value) for value in manifest["distance_thresholds"]]
    min_recall = float(manifest["min_recall"])
    min_precision = float(manifest["min_precision"])
    tp_threshold = float(manifest["tp_distance_threshold"])
    label_aps = {}
    label_tp = {}
    for class_name in classes:
        scores = arrays["score__%s" % class_name]
        event_scenes = arrays["scene__%s" % class_name]
        gt_counts = arrays["gt__%s" % class_name]
        label_aps[class_name] = {}
        details = {}
        for threshold in thresholds:
            suffix = core.threshold_key(threshold)
            tp = arrays["tp__%s__%s" % (class_name, suffix)]
            errors = {
                name: arrays["err__%s__%s__%s" % (class_name, suffix, name)]
                for name in ERRORS
            }
            data = core.draw_metric_data(scores, tp, errors, event_scenes, gt_counts, draw, int(manifest["scene_count"]))
            details[threshold] = data
            label_aps[class_name][str(threshold)] = calc_ap(data, min_recall, min_precision)
        label_tp[class_name] = {}
        for metric in ERRORS:
            if class_name == "traffic_cone" and metric in ("attr_err", "vel_err", "orient_err"):
                value = float("nan")
            elif class_name == "barrier" and metric in ("attr_err", "vel_err"):
                value = float("nan")
            else:
                value = calc_tp(details[tp_threshold], min_recall, metric)
            label_tp[class_name][metric] = value
    mean_dist = {name: float(np.mean(list(values.values()))) for name, values in label_aps.items()}
    mean_ap = float(np.mean(list(mean_dist.values())))
    tp_errors = {
        metric: float(np.nanmean([label_tp[name][metric] for name in classes]))
        for metric in ERRORS
    }
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


def build_expected(
    core: Any,
    plan: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    draw_bytes: bytes,
    draw_record: Mapping[str, Any],
    bundles: Mapping[str, Tuple[Path, Mapping[str, Any], Mapping[str, np.ndarray], Mapping[str, Any]]],
    scope: str,
) -> Mapping[str, Any]:
    scene_count = int(plan["scene_bootstrap"]["scene_count"])
    matrix = np.frombuffer(draw_bytes, dtype="<u2").reshape(-1, scene_count)
    replicates = 8 if scope == "technical_pilot_B8" else int(plan["scene_bootstrap"]["registered_replicates"])
    require(scope in ("technical_pilot_B8", "registered_B1000"), "invalid scope")
    require(len(matrix) >= replicates, "draw matrix too short")
    contract = plan["scene_bootstrap"]["draw_generator"]
    require(len(draw_bytes) == int(contract["draw_matrix_size_bytes"]), "draw matrix size mismatch")
    require(hashlib.sha256(draw_bytes).hexdigest() == contract["expected_draw_matrix_sha256"], "draw matrix plan identity mismatch")
    require(matrix[0, :12].tolist() == contract["first_replicate_first_12_indices"], "draw prefix mismatch")
    scene_hashes = {manifest.get("scene_tokens_sha256") for _path, manifest, _arrays, _receipt in bundles.values()}
    require(len(scene_hashes) == 1 and None not in scene_hashes, "event bundles do not share one canonical scene-token order")
    arms = {}
    for label, (manifest_path, manifest, arrays, receipt_record) in bundles.items():
        full = summarize(core, arrays, manifest, np.arange(scene_count, dtype=np.uint16))
        official = manifest["official_summary"]
        closure = {
            "mean_ap_abs_diff": abs(full["mean_ap"] - official["mean_ap"]),
            "nd_score_abs_diff": abs(full["nd_score"] - official["nd_score"]),
            "mAVE_abs_diff": abs(full["mAVE"] - official["mAVE"]),
        }
        require(max(closure.values()) <= 1e-12, "all-ones closure failed: %s" % label)
        rows = []
        for replicate in range(replicates):
            values = summarize(core, arrays, manifest, matrix[replicate])
            rows.append({"replicate": replicate, "mean_ap": values["mean_ap"], "nd_score": values["nd_score"], "mAVE": values["mAVE"]})
        arms[label] = {
            "event_manifest": stable_record(manifest_path),
            "event_check_receipt": dict(receipt_record),
            "all_ones": full,
            "official_closure": closure,
            "replicates": rows,
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed_technical_event_replay" if scope == "technical_pilot_B8" else "passed_registered_bootstrap_replay",
        "scope": scope,
        "scientific_evidence": scope == "registered_B1000",
        "replicate_count": replicates,
        "scene_count": scene_count,
        "scene_tokens_sha256": next(iter(scene_hashes)),
        "arms": arms,
        "per_scene_AP_average_used": False,
        "official_evaluator_calls": 0,
        "official_algorithm_equivalent_internal_replay": True,
        "draw_prefix_sha256": hashlib.sha256(draw_bytes[: replicates * scene_count * 2]).hexdigest(),
        "inputs": {"plan": dict(plan_record), "draw_matrix": dict(draw_record)},
    }


def publish(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.absolute()
    require(target.parent.is_dir() and not os.path.lexists(str(target)), "checker output contract failed")
    descriptor, raw_temporary = tempfile.mkstemp(prefix=".%s.tmp." % target.name, dir=str(target.parent))
    temporary = Path(raw_temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(target))
        linked = True
    except Exception:
        if linked and target.exists():
            target.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-runner", required=True)
    parser.add_argument("--expected-bootstrap-runner-sha256", required=True)
    parser.add_argument("--expected-bootstrap-checker-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--draw-matrix", required=True)
    parser.add_argument("--expected-draw-sha256", required=True)
    parser.add_argument("--event-checker-core", required=True)
    parser.add_argument("--expected-event-checker-core-sha256", required=True)
    parser.add_argument("--event-manifest", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--event-check-receipt", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--scope", choices=("technical_pilot_B8", "registered_B1000"), required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        plan_path = Path(args.plan).absolute()
        draw_path = Path(args.draw_matrix).absolute()
        report_path = Path(args.report).absolute()
        plan_record = stable_record(plan_path, args.expected_plan_sha256)
        draw_record = stable_record(draw_path, args.expected_draw_sha256)
        report_record = stable_record(report_path, args.expected_report_sha256)
        core_record, core = load_checker_core(Path(args.event_checker_core).absolute(), args.expected_event_checker_core_sha256)
        runner_record = stable_record(Path(args.bootstrap_runner).absolute(), args.expected_bootstrap_runner_sha256)
        checker_record = stable_record(Path(__file__).absolute(), args.expected_bootstrap_checker_sha256)
        plan = load_json(plan_path)
        require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
        expected = None
        labels = []
        for label, manifest_path, receipt_path in bind_event_inputs(args.event_manifest, args.event_check_receipt):
            labels.append(label)
            manifest, arrays = load_bundle(manifest_path)
            receipt_record = validate_event_check_receipt(label, manifest_path, manifest, receipt_path)
            partial = build_expected(core, plan, plan_record, draw_path.read_bytes(), draw_record, {label: (manifest_path, manifest, arrays, receipt_record)}, args.scope)
            if expected is None:
                expected = partial
            else:
                for key, value in partial.items():
                    if key != "arms":
                        require(expected[key] == value, "cross-arm report metadata drift: %s" % key)
                expected["arms"].update(partial["arms"])
            del arrays
        require(expected is not None, "no event bundles supplied")
        expected["implementation"] = {"bootstrap_runner": runner_record}
        observed = load_json(report_path)
        require(canonical_bytes(observed) == canonical_bytes(expected), "bootstrap report differs from independent replay")
        payload = {
            "schema_version": CHECK_SCHEMA,
            "status": "passed_independent_paired_scene_bootstrap_replay",
            "passed": True,
            "scope": args.scope,
            "scientific_evidence": args.scope == "registered_B1000",
            "report": report_record,
            "event_checker_core": core_record,
            "implementation": {"bootstrap_runner": runner_record, "bootstrap_checker": checker_record},
            "arm_labels": sorted(labels),
            "event_check_receipts": {
                label: expected["arms"][label]["event_check_receipt"]
                for label in sorted(labels)
            },
            "all_event_check_receipts_passed": True,
            "replicate_count": expected["replicate_count"],
            "scene_count": expected["scene_count"],
            "all_report_values_exact": True,
            "per_scene_AP_average_used": False,
            "official_evaluator_calls": 0,
        }
        publish(Path(args.output), payload)
    except (CheckError, KeyError, OSError, ValueError) as exc:
        print("FAILED: %s" % exc, file=os.sys.stderr)
        return 1
    print(str(Path(args.output).absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
