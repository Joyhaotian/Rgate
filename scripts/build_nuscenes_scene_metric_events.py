#!/usr/bin/env python3
"""Build scene-local nuScenes metric events with exact official closure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PLAN_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_plan_v1"
IDENTITY_ERRATUM_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_identity_erratum_v1"
EVENT_SCHEMA = "nuscenes_scene_metric_event_cache_v1"
TABLE_LOCK_SCHEMA = "nuscenes_table_identity_lock_v1"
CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
ERRORS = ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err")
NUSCENES_TABLES = ("category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor", "ego_pose", "log", "scene", "sample", "sample_data", "sample_annotation", "map")


class EventError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EventError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(EventError("non-finite JSON")))


def load_official_metrics_json(path: Path) -> Any:
    def official_constant(value: str) -> float:
        require(value == "NaN", "unexpected official metric constant: %s" % value)
        return float("nan")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=official_constant)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, expected_sha: str) -> Dict[str, Any]:
    logical = path.absolute()
    require(logical.is_file(), "missing input: %s" % logical)
    before = logical.stat(); digest = sha(logical); after = logical.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed while hashing")
    require(digest == expected_sha, "input SHA mismatch: %s" % logical)
    return {"path": str(logical), "size_bytes": before.st_size, "sha256": digest}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_identity_erratum(
    erratum_path: Path,
    expected_sha256: str,
    plan_path: Path,
    plan_record: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, str]]:
    erratum_record = record(erratum_path, expected_sha256)
    payload = load_json(erratum_path)
    require(
        set(payload) == {"schema_version", "status", "scope", "original_plan", "correction", "supporting_evidence", "invariants"},
        "identity erratum top-level schema mismatch",
    )
    require(payload["schema_version"] == IDENTITY_ERRATUM_SCHEMA, "identity erratum schema mismatch")
    require(payload["status"] == "frozen_narrow_identity_transcription_correction_before_event_cache_execution", "identity erratum status mismatch")
    require(payload["scope"] == "seed0_E4_prediction_sha256_only", "identity erratum scope mismatch")
    original = payload["original_plan"]
    require(set(original) == {"relative_path", "schema_version", "size_bytes", "sha256"}, "identity erratum original-plan schema mismatch")
    require(original["relative_path"] == "configs/rgate_rq2_multiseed_scene_bootstrap_plan_v1.json", "identity erratum plan path mismatch")
    require(original["schema_version"] == PLAN_SCHEMA, "identity erratum plan schema mismatch")
    require(original["size_bytes"] == plan_record["size_bytes"] and original["sha256"] == plan_record["sha256"], "identity erratum does not bind the supplied plan")
    require(plan_path.name == "rgate_rq2_multiseed_scene_bootstrap_plan_v1.json", "unexpected plan filename for identity erratum")

    correction = payload["correction"]
    require(
        set(correction) == {
            "json_pointer", "arm_label", "field_semantics", "malformed_value", "malformed_character_count",
            "corrected_value", "corrected_character_count", "correction_kind", "appended_character",
        },
        "identity erratum correction schema mismatch",
    )
    malformed = plan["frozen_inputs_and_reuse"]["seed0_artifacts"]["prediction_sha256"]["E4"]
    corrected = correction["corrected_value"]
    require(correction["json_pointer"] == "/frozen_inputs_and_reuse/seed0_artifacts/prediction_sha256/E4", "identity erratum JSON pointer mismatch")
    require(correction["arm_label"] == "E4" and correction["field_semantics"] == "sha256_of_seed0_E4_prediction_json", "identity erratum field semantics mismatch")
    require(correction["malformed_value"] == malformed and correction["malformed_character_count"] == len(malformed) == 63, "identity erratum malformed value mismatch")
    require(correction["corrected_character_count"] == len(corrected) == 64, "identity erratum corrected length mismatch")
    require(correction["correction_kind"] == "append_one_omitted_trailing_hex_character" and correction["appended_character"] == "d", "identity erratum correction kind mismatch")
    require(corrected == malformed + correction["appended_character"], "identity erratum is not a one-character append")
    require(all(character in "0123456789abcdef" for character in corrected), "identity erratum corrected SHA is not lowercase hexadecimal")

    invariants = payload["invariants"]
    require(
        invariants == {
            "original_plan_remains_immutable": True,
            "all_other_plan_fields_unchanged": True,
            "random_seeds_changed": False,
            "bootstrap_draws_changed": False,
            "scientific_hypotheses_or_gates_changed": False,
            "models_predictions_or_metrics_changed": False,
            "event_cache_retry_requires_original_plan_and_this_erratum": True,
            "failed_precompute_attempt_generated_event_arrays": False,
            "failed_precompute_attempt_generated_scientific_results": False,
        },
        "identity erratum invariants mismatch",
    )
    project_root = plan_path.parent.parent.absolute()
    evidence_records = []
    evidence_rows = payload["supporting_evidence"]
    require(isinstance(evidence_rows, list) and len(evidence_rows) == 4, "identity erratum supporting-evidence extent mismatch")
    seen_paths = set()
    for expected in evidence_rows:
        require(set(expected) == {"relative_path", "size_bytes", "sha256", "must_contain_corrected_value"}, "identity erratum evidence schema mismatch")
        relative = Path(expected["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "identity erratum evidence path is not project-relative")
        require(str(relative) not in seen_paths, "duplicate identity erratum evidence path")
        seen_paths.add(str(relative))
        evidence_path = (project_root / relative).absolute()
        observed = record(evidence_path, expected["sha256"])
        require(observed["size_bytes"] == expected["size_bytes"], "identity erratum evidence size mismatch")
        require(expected["must_contain_corrected_value"] is True, "identity erratum evidence literal requirement missing")
        require(corrected.encode("ascii") in evidence_path.read_bytes(), "identity erratum evidence does not contain corrected SHA")
        evidence_records.append(observed)
    effective = dict(plan["frozen_inputs_and_reuse"]["seed0_artifacts"]["prediction_sha256"])
    effective["E4"] = corrected
    return erratum_record, evidence_records, effective


def array_record(array: np.ndarray) -> Dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def threshold_key(value: float) -> str:
    return ("%.1f" % value).replace(".", "p")


def load_bootstrap_module(path: Path):
    spec = importlib.util.spec_from_file_location("rgate_bootstrap_core", str(path))
    require(spec is not None and spec.loader is not None, "cannot import bootstrap core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_table_lock(lock_path: Path, expected_sha256: str, nusc_root: Path, version: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    lock_record = record(lock_path, expected_sha256)
    payload = load_json(lock_path)
    require(payload.get("schema_version") == TABLE_LOCK_SCHEMA and payload.get("status") == "frozen_before_event_execution", "nuScenes table-lock schema/status mismatch")
    require(Path(payload.get("dataroot", "")).absolute() == nusc_root.absolute(), "nuScenes dataroot differs from table lock")
    require(payload.get("version") == version, "nuScenes version differs from table lock")
    require(set(payload.get("tables", {})) == set(NUSCENES_TABLES), "nuScenes table-lock key set mismatch")
    table_records = {}
    for name in NUSCENES_TABLES:
        path = (nusc_root / version / (name + ".json")).absolute()
        expected = payload["tables"][name]
        observed = record(path, expected["sha256"])
        require(observed == expected, "nuScenes table identity mismatch: %s" % name)
        table_records[name] = observed
    return lock_record, table_records


def require_within(path: Path, root: Path, message: str) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        raise EventError(message)


def mount_identity(path: Path) -> Dict[str, str]:
    completed = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE,FSTYPE", "-T", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    require(completed.returncode == 0, "findmnt failed for event output: %s" % completed.stderr.strip())
    values = completed.stdout.strip().split()
    require(len(values) == 2, "unexpected findmnt output for event root")
    return {"source": values[0], "fstype": values[1]}


def scene_layout(nusc: Any, sample_tokens: Sequence[str], split_scene_names: Sequence[str]) -> tuple[list[str], Dict[str, int]]:
    """Map arbitrary result order onto the canonical official split order."""
    require(len(split_scene_names) == 150 and len(set(split_scene_names)) == 150, "validation split scene-name extent mismatch")
    scene_by_name = {str(row["name"]): str(row["token"]) for row in nusc.scene}
    require(all(name in scene_by_name for name in split_scene_names), "validation split scene missing from nuScenes table")
    scenes = [scene_by_name[name] for name in split_scene_names]
    scene_index = {token: index for index, token in enumerate(scenes)}
    token_scene = {}
    for token in sample_tokens:
        scene = str(nusc.get("sample", token)["scene_token"])
        require(scene in scene_index, "prediction sample belongs outside registered validation scenes")
        token_scene[token] = scene_index[scene]
    require(len(sample_tokens) == 6019 and len(set(sample_tokens)) == 6019, "validation sample extent/uniqueness mismatch")
    require(set(token_scene.values()) == set(range(150)), "prediction result does not cover every validation scene")
    return scenes, token_scene


def match_events(gt_boxes: Any, predictions: Sequence[Any], class_name: str, threshold: float, distance: Any, utils: Mapping[str, Any]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    scores = [float(box.detection_score) for box in predictions]
    order = [index for score, index in sorted((score, index) for index, score in enumerate(scores))][::-1]
    taken = set()
    tp = np.zeros(len(predictions), dtype=np.uint8)
    errors = {name: np.full(len(predictions), np.nan, dtype=np.float64) for name in ERRORS}
    for index in order:
        prediction = predictions[index]
        minimum, matched = np.inf, None
        for gt_index, gt in enumerate(gt_boxes[prediction.sample_token]):
            if gt.detection_name == class_name and (prediction.sample_token, gt_index) not in taken:
                candidate = distance(gt, prediction)
                if candidate < minimum:
                    minimum, matched = candidate, gt_index
        if minimum < threshold:
            require(matched is not None, "matched GT index missing")
            taken.add((prediction.sample_token, matched)); tp[index] = 1
            gt = gt_boxes[prediction.sample_token][matched]
            errors["trans_err"][index] = utils["center_distance"](gt, prediction)
            errors["vel_err"][index] = utils["velocity_l2"](gt, prediction)
            errors["scale_err"][index] = 1.0 - utils["scale_iou"](gt, prediction)
            period = np.pi if class_name == "barrier" else 2.0 * np.pi
            errors["orient_err"][index] = utils["yaw_diff"](gt, prediction, period=period)
            errors["attr_err"][index] = 1.0 - utils["attr_acc"](gt, prediction)
    return tp, errors


def build(args: argparse.Namespace, staging: Path) -> Mapping[str, Any]:
    plan_path, result_path = Path(args.plan).absolute(), Path(args.result_json).absolute()
    metrics_summary_path, metrics_details_path = Path(args.metrics_summary).absolute(), Path(args.metrics_details).absolute()
    bootstrap_path = Path(args.bootstrap_core).absolute()
    nusc_root = Path(args.nusc_root).absolute()
    table_lock_record, table_records = validate_table_lock(Path(args.nusc_table_lock).absolute(), args.expected_nusc_table_lock_sha256, nusc_root, args.version)
    plan_record = record(plan_path, args.expected_plan_sha256)
    plan = load_json(plan_path)
    require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
    erratum_record, erratum_evidence, effective_seed0 = validate_identity_erratum(
        Path(args.identity_erratum).absolute(), args.expected_identity_erratum_sha256, plan_path, plan_record, plan
    )
    inputs = {
        "event_builder": record(Path(__file__).absolute(), args.expected_event_builder_sha256),
        "plan": plan_record,
        "identity_erratum": erratum_record,
        "identity_erratum_supporting_evidence": erratum_evidence,
        "result": record(result_path, args.expected_result_sha256),
        "metrics_summary": record(metrics_summary_path, args.expected_metrics_summary_sha256),
        "metrics_details": record(metrics_details_path, args.expected_metrics_details_sha256),
        "bootstrap_core": record(bootstrap_path, args.expected_bootstrap_core_sha256),
        "nusc_table_lock": table_lock_record,
        "nusc_tables": table_records,
    }
    registered_seed0 = plan["frozen_inputs_and_reuse"]["seed0_artifacts"]["prediction_sha256"]
    if args.arm_label in effective_seed0:
        require(effective_seed0[args.arm_label] == args.expected_result_sha256, "seed0 arm result identity mismatch after registered erratum")

    # Imports are deliberately local: unit tests and static validation do not
    # require nuScenes, while a real event build must use the locked devkit.
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
    from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.utils.splits import create_splits_scenes

    config = config_factory("detection_cvpr_2019")
    require(tuple(config.class_names) == CLASSES and tuple(float(v) for v in config.dist_ths) == THRESHOLDS, "devkit detection config drift")
    require(float(config.dist_th_tp) == 2.0 and float(config.min_recall) == 0.1 and float(config.min_precision) == 0.1, "devkit metric constants drift")
    nusc = NuScenes(version=args.version, dataroot=str(nusc_root), verbose=False)
    pred_boxes, _meta = load_prediction(str(result_path), config.max_boxes_per_sample, DetectionBox, verbose=False)
    gt_boxes = load_gt(nusc, args.eval_set, DetectionBox, verbose=False)
    require(set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), "prediction/GT token mismatch")
    pred_boxes, gt_boxes = add_center_dist(nusc, pred_boxes), add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, config.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, config.class_range, verbose=False)
    sample_tokens = list(pred_boxes.sample_tokens)
    split_scene_names = list(create_splits_scenes()[args.eval_set])
    scenes, token_scene = scene_layout(nusc, sample_tokens, split_scene_names)
    arrays: Dict[str, np.ndarray] = {}
    utilities = {"center_distance": center_distance, "velocity_l2": velocity_l2, "scale_iou": scale_iou, "yaw_diff": yaw_diff, "attr_acc": attr_acc}
    all_gt = gt_boxes.all
    all_pred = pred_boxes.all
    class_counts = {}
    for class_name in CLASSES:
        predictions = [box for box in all_pred if box.detection_name == class_name]
        scene_values = np.array([token_scene[box.sample_token] for box in predictions], dtype=np.uint16)
        scores = np.array([float(box.detection_score) for box in predictions], dtype=np.float64)
        arrays["scene__%s" % class_name] = scene_values
        arrays["score__%s" % class_name] = scores
        arrays["ordinal__%s" % class_name] = np.arange(len(predictions), dtype=np.uint32)
        gt_counts = np.zeros(150, dtype=np.uint32)
        for box in all_gt:
            if box.detection_name == class_name:
                gt_counts[token_scene[box.sample_token]] += 1
        arrays["gt__%s" % class_name] = gt_counts
        class_counts[class_name] = {"predictions": len(predictions), "ground_truth": int(gt_counts.sum())}
        for threshold in THRESHOLDS:
            tp, errors = match_events(gt_boxes, predictions, class_name, threshold, config.dist_fcn_callable, utilities)
            key = threshold_key(threshold)
            arrays["tp__%s__%s" % (class_name, key)] = tp
            for error_name, values in errors.items():
                arrays["err__%s__%s__%s" % (class_name, key, error_name)] = values

    npz_path = staging / "events.npz"
    np.savez_compressed(str(npz_path), **arrays)
    official_summary = load_official_metrics_json(metrics_summary_path)
    official_details = load_official_metrics_json(metrics_details_path)
    bootstrap = load_bootstrap_module(bootstrap_path)
    bootstrap_manifest = {
        "classes": list(CLASSES), "distance_thresholds": list(THRESHOLDS),
        "min_recall": 0.1, "min_precision": 0.1, "tp_distance_threshold": 2.0,
        "scene_count": 150,
    }
    full = bootstrap.summarize(arrays, bootstrap_manifest, np.arange(150, dtype=np.uint16))
    summary_diffs = {
        "mean_ap": abs(full["mean_ap"] - float(official_summary["mean_ap"])),
        "nd_score": abs(full["nd_score"] - float(official_summary["nd_score"])),
        "mAVE": abs(full["mAVE"] - float(official_summary["tp_errors"]["vel_err"])),
    }
    max_detail_diff = 0.0
    for class_name in CLASSES:
        for threshold in THRESHOLDS:
            reconstructed = bootstrap.metric_data(arrays, class_name, threshold, np.arange(150, dtype=np.uint16), 150)
            registered = official_details["%s:%.1f" % (class_name, threshold)]
            for field in ("recall", "precision", "confidence") + ERRORS:
                left, right = reconstructed[field], np.asarray(registered[field], dtype=np.float64)
                require(left.shape == right.shape and np.array_equal(np.isnan(left), np.isnan(right)), "metric-details shape/NaN drift")
                finite = ~np.isnan(left)
                difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
                max_detail_diff = max(max_detail_diff, difference)
    require(max(summary_diffs.values()) <= 1e-12 and max_detail_diff <= 1e-12, "official event closure exceeds 1e-12")
    arrays_record = {key: array_record(value) for key, value in sorted(arrays.items())}
    scene_sha = hashlib.sha256(canonical(scenes)).hexdigest()
    scene_name_sha = hashlib.sha256(canonical(split_scene_names)).hexdigest()
    return {
        "schema_version": EVENT_SCHEMA,
        "status": "passed_official_event_trace_closure",
        "arm_label": args.arm_label,
        "seed0_prediction_identity_resolution": {
            "plan_value": registered_seed0.get(args.arm_label),
            "effective_value": effective_seed0.get(args.arm_label),
            "identity_erratum_applied": args.arm_label == "E4",
        },
        "inputs": inputs,
        "devkit": {"version": "1.1.11", "config": "detection_cvpr_2019", "eval_set": args.eval_set},
        "classes": list(CLASSES),
        "distance_thresholds": list(THRESHOLDS),
        "min_recall": 0.1,
        "min_precision": 0.1,
        "tp_distance_threshold": 2.0,
        "scene_count": 150,
        "sample_count": 6019,
        "scene_tokens": scenes,
        "scene_tokens_sha256": scene_sha,
        "scene_names": split_scene_names,
        "scene_names_sha256": scene_name_sha,
        "scene_order_policy": "nuscenes_devkit_create_splits_scenes_eval_set_order_mapped_name_to_locked_scene_token",
        "sample_order_policy": "result_json_original_sample_and_box_order_without_scene_contiguity_requirement",
        "tie_policy": "python_sorted_score_index_reverse",
        "class_counts": class_counts,
        "npz": {"path": "events.npz", "size_bytes": npz_path.stat().st_size, "sha256": sha(npz_path)},
        "arrays": arrays_record,
        "official_summary": {"mean_ap": float(official_summary["mean_ap"]), "nd_score": float(official_summary["nd_score"]), "mAVE": float(official_summary["tp_errors"]["vel_err"])},
        "all_ones_reconstruction": {"mean_ap": full["mean_ap"], "nd_score": full["nd_score"], "mAVE": full["mAVE"]},
        "summary_abs_diffs": summary_diffs,
        "metric_details_max_abs_diff": max_detail_diff,
        "per_scene_AP_materialized": False,
        "validation_artifacts_modified": 0,
        "official_evaluator_calls": 0,
    }


def publish(output_dir: Path, manifest: Mapping[str, Any], staging: Path) -> None:
    target = output_dir.absolute()
    require(target.parent.is_dir() and not os.path.lexists(str(target)), "output contract failed")
    manifest_path = staging / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical(manifest)); handle.flush(); os.fsync(handle.fileno())
    require(set(path.name for path in staging.iterdir()) == {"events.npz", "manifest.json"}, "staged file set mismatch")
    os.rename(str(staging), str(target))


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-event-builder-sha256", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--identity-erratum", required=True); parser.add_argument("--expected-identity-erratum-sha256", required=True)
    parser.add_argument("--bootstrap-core", required=True); parser.add_argument("--expected-bootstrap-core-sha256", required=True)
    parser.add_argument("--result-json", required=True); parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--metrics-summary", required=True); parser.add_argument("--expected-metrics-summary-sha256", required=True)
    parser.add_argument("--metrics-details", required=True); parser.add_argument("--expected-metrics-details-sha256", required=True)
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--nusc-root", required=True); parser.add_argument("--version", default="v1.0-trainval"); parser.add_argument("--eval-set", default="val")
    parser.add_argument("--nusc-table-lock", required=True); parser.add_argument("--expected-nusc-table-lock-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-output-root", required=True)
    parser.add_argument("--expected-output-mount-source", required=True)
    parser.add_argument("--expected-output-mount-fstype", required=True)
    parser.add_argument("--minimum-output-free-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).absolute()
    allowed_root = Path(args.allowed_output_root).absolute()
    require(allowed_root.is_dir(), "allowed output root missing")
    require_within(output, allowed_root, "event output escapes allowed root")
    require(output.parent.is_dir() and not os.path.lexists(str(output)), "output must be fresh")
    mount = mount_identity(allowed_root)
    require(mount == {"source": args.expected_output_mount_source, "fstype": args.expected_output_mount_fstype}, "event output mount identity mismatch")
    free_before = shutil.disk_usage(allowed_root).free
    require(type(args.minimum_output_free_bytes) is int and args.minimum_output_free_bytes >= 2 * 1024 ** 3, "event minimum free-space gate too small")
    require(free_before >= args.minimum_output_free_bytes, "insufficient event output free space")
    staging = Path(tempfile.mkdtemp(prefix=".%s.staging." % output.name, dir=str(output.parent)))
    try:
        manifest = dict(build(args, staging))
        free_before_publish = shutil.disk_usage(allowed_root).free
        require(free_before_publish >= args.minimum_output_free_bytes, "event output free-space gate failed before publish")
        manifest["storage"] = {
            "allowed_output_root": str(allowed_root),
            "mount": mount,
            "minimum_free_bytes": args.minimum_output_free_bytes,
            "free_bytes_before_build": free_before,
            "free_bytes_before_publish": free_before_publish,
        }
        publish(output, manifest, staging)
    except Exception as exc:
        if staging.exists(): shutil.rmtree(staging)
        print("FAILED: %s" % exc, file=os.sys.stderr); return 1
    print(str(output / "manifest.json")); return 0


if __name__ == "__main__":
    raise SystemExit(main())
