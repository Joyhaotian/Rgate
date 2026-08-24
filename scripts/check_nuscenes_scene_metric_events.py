#!/usr/bin/env python3
"""Independently verify a scene-local nuScenes metric-event bundle.

This checker deliberately does not import the event builder or bootstrap
runner.  It reloads the locked prediction and nuScenes tables, repeats the
sample-local greedy matching, and compares every persisted event array before
reconstructing all forty official metric-detail entries.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


PLAN_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_plan_v1"
IDENTITY_ERRATUM_SCHEMA = "rgate_rq2_multiseed_scene_bootstrap_identity_erratum_v1"
DIRECT_REFERENCE_SCHEMA = "rgate_rq2_multiseed_direct_reference_draws_v1"
EVENT_SCHEMA = "nuscenes_scene_metric_event_cache_v1"
CHECK_SCHEMA = "nuscenes_scene_metric_event_cache_independent_check_v1"
TABLE_LOCK_SCHEMA = "nuscenes_table_identity_lock_v1"
CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
ERRORS = ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err")
NUSCENES_TABLES = ("category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor", "ego_pose", "log", "scene", "sample", "sample_data", "sample_annotation", "map")
DIRECT_REFERENCE_RULE = {
    "domain_separator": "rgate-rq2-multiseed-direct-reference-draws-v1",
    "preimage_encoding": "ASCII domain|plan_sha256=<lowerhex>|draw_sha256=<lowerhex>|ordinal=<base10>|counter=<base10>",
    "hash": "sha256",
    "digest_prefix": "first_8_bytes_interpreted_as_unsigned_big_endian_uint64",
    "candidate_formula": "1 + (digest_prefix_uint64 mod 999)",
    "candidate_index_min_inclusive": 1,
    "candidate_index_max_inclusive": 999,
    "selection_count": 2,
    "collision_policy": "For each ordinal in ascending order, start counter at 0 and increment until the candidate index is distinct from every earlier accepted index.",
}


class CheckError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def reject_constant(value: str) -> None:
    raise CheckError("non-finite JSON constant: %s" % value)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def load_official_metrics_json(path: Path) -> Any:
    def official_constant(value: str) -> float:
        require(value == "NaN", "unexpected official metric constant: %s" % value)
        return float("nan")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=official_constant)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_record(path: Path, expected_sha256: str = "") -> Dict[str, Any]:
    logical = path.absolute()
    require(logical.is_file(), "missing file: %s" % logical)
    before = logical.stat()
    digest = file_sha256(logical)
    after = logical.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    require(identity_before == identity_after, "file changed while hashing: %s" % logical)
    if expected_sha256:
        require(digest == expected_sha256, "SHA-256 mismatch: %s" % logical)
    return {"path": str(logical), "size_bytes": before.st_size, "sha256": digest}


def derive_direct_reference_selections(plan_sha256: str, draw_sha256: str) -> list[Dict[str, Any]]:
    require(len(plan_sha256) == len(draw_sha256) == 64, "direct-reference input SHA length mismatch")
    require(all(character in "0123456789abcdef" for character in plan_sha256 + draw_sha256), "direct-reference input SHA format mismatch")
    selections = []
    accepted = set()
    domain = DIRECT_REFERENCE_RULE["domain_separator"]
    for ordinal in range(DIRECT_REFERENCE_RULE["selection_count"]):
        counter = 0
        while True:
            preimage = "%s|plan_sha256=%s|draw_sha256=%s|ordinal=%d|counter=%d" % (
                domain,
                plan_sha256,
                draw_sha256,
                ordinal,
                counter,
            )
            digest = hashlib.sha256(preimage.encode("ascii")).hexdigest()
            prefix = int(digest[:16], 16)
            draw_index = 1 + prefix % 999
            if draw_index not in accepted:
                accepted.add(draw_index)
                selections.append(
                    {
                        "ordinal": ordinal,
                        "counter": counter,
                        "preimage": preimage,
                        "sha256": digest,
                        "digest_prefix_uint64_be_decimal": str(prefix),
                        "draw_index": draw_index,
                    }
                )
                break
            counter += 1
            require(counter < 10000, "direct-reference collision resolution exceeded bound")
    return selections


def validate_direct_reference_config(
    config_path: Path,
    expected_sha256: str,
    plan_record: Mapping[str, Any],
    draw_record: Mapping[str, Any],
) -> Tuple[Dict[str, Any], list[int]]:
    config_record = stable_record(config_path, expected_sha256)
    payload = load_json(config_path)
    require(
        set(payload)
        == {
            "schema_version",
            "status",
            "frozen_at_utc",
            "purpose",
            "plan",
            "draw_matrix",
            "selection_rule",
            "selections",
            "direct_reference_draw_indices",
            "invariants",
        },
        "direct-reference config schema mismatch",
    )
    require(payload["schema_version"] == DIRECT_REFERENCE_SCHEMA, "direct-reference config version mismatch")
    require(payload["status"] == "frozen_before_event_execution_technical_addendum", "direct-reference config status mismatch")
    require(isinstance(payload["frozen_at_utc"], str) and payload["frozen_at_utc"].endswith("Z"), "direct-reference freeze time mismatch")
    require(isinstance(payload["purpose"], str) and bool(payload["purpose"]), "direct-reference purpose missing")
    require(payload["plan"] == dict(plan_record), "direct-reference plan identity mismatch")
    require(payload["draw_matrix"] == dict(draw_record), "direct-reference draw identity mismatch")
    require(payload["selection_rule"] == DIRECT_REFERENCE_RULE, "direct-reference selection rule mismatch")
    expected_selections = derive_direct_reference_selections(plan_record["sha256"], draw_record["sha256"])
    require(payload["selections"] == expected_selections, "direct-reference deterministic selections mismatch")
    indices = [0] + [int(row["draw_index"]) for row in expected_selections]
    require(payload["direct_reference_draw_indices"] == indices, "direct-reference index vector mismatch")
    require(len(indices) == len(set(indices)) == 3 and indices[0] == 0 and all(1 <= value <= 999 for value in indices[1:]), "direct-reference index extent mismatch")
    require(
        payload["invariants"]
        == {
            "event_execution_started_at_freeze": False,
            "draw_zero_retained": True,
            "additional_indices_nonzero_and_distinct": True,
            "registered_draw_matrix_changed": False,
            "scientific_replicates_changed": False,
            "models_predictions_metrics_hypotheses_or_gates_changed": False,
            "technical_checker_coverage_only": True,
        },
        "direct-reference invariants mismatch",
    )
    return config_record, indices


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_identity_erratum(
    erratum_path: Path,
    expected_sha256: str,
    plan_path: Path,
    plan_record: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, str]]:
    erratum_record = stable_record(erratum_path, expected_sha256)
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

    require(
        payload["invariants"] == {
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
    evidence_rows = payload["supporting_evidence"]
    require(isinstance(evidence_rows, list) and len(evidence_rows) == 4, "identity erratum supporting-evidence extent mismatch")
    evidence_records = []
    seen_paths = set()
    for expected in evidence_rows:
        require(set(expected) == {"relative_path", "size_bytes", "sha256", "must_contain_corrected_value"}, "identity erratum evidence schema mismatch")
        relative = Path(expected["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "identity erratum evidence path is not project-relative")
        require(str(relative) not in seen_paths, "duplicate identity erratum evidence path")
        seen_paths.add(str(relative))
        evidence_path = (project_root / relative).absolute()
        observed = stable_record(evidence_path, expected["sha256"])
        require(observed["size_bytes"] == expected["size_bytes"], "identity erratum evidence size mismatch")
        require(expected["must_contain_corrected_value"] is True, "identity erratum evidence literal requirement missing")
        require(corrected.encode("ascii") in evidence_path.read_bytes(), "identity erratum evidence does not contain corrected SHA")
        evidence_records.append(observed)
    effective = dict(plan["frozen_inputs_and_reuse"]["seed0_artifacts"]["prediction_sha256"])
    effective["E4"] = corrected
    return erratum_record, evidence_records, effective


def array_record(value: np.ndarray) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


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


def cumulative_mean(values: np.ndarray) -> np.ndarray:
    if int(np.isnan(values).sum()) == len(values):
        return np.ones(len(values))
    sums = np.nancumsum(values.astype(float))
    counts = np.cumsum(~np.isnan(values))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def event_metric_data(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    errors: Mapping[str, np.ndarray],
    positives: int,
) -> Dict[str, np.ndarray]:
    if positives == 0 or int(tp_flags.sum()) == 0:
        return no_predictions()
    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((-indices, -scores.astype(np.float64, copy=False)))
    ordered_scores = scores[order].astype(np.float64, copy=False)
    ordered_tp = tp_flags[order].astype(np.int64, copy=False)
    ordered_fp = 1 - ordered_tp
    tp_cumulative = np.cumsum(ordered_tp).astype(float)
    fp_cumulative = np.cumsum(ordered_fp).astype(float)
    recall = tp_cumulative / float(positives)
    precision = tp_cumulative / (tp_cumulative + fp_cumulative)
    recall_grid = np.linspace(0, 1, 101)
    precision_grid = np.interp(recall_grid, recall, precision, right=0)
    confidence_grid = np.interp(recall_grid, recall, ordered_scores, right=0)
    result = {"recall": recall_grid, "precision": precision_grid, "confidence": confidence_grid}
    matched = ordered_tp.astype(bool)
    matched_confidence = ordered_scores[matched]
    for name in ERRORS:
        values = errors[name][order][matched].astype(np.float64, copy=False)
        means = cumulative_mean(values)
        result[name] = np.interp(confidence_grid[::-1], matched_confidence[::-1], means[::-1])[::-1]
    return result


def draw_metric_data(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    errors: Mapping[str, np.ndarray],
    event_scenes: np.ndarray,
    gt_counts: np.ndarray,
    draw: np.ndarray,
    scene_count: int,
) -> Dict[str, np.ndarray]:
    positives = int(np.asarray(gt_counts, dtype=np.int64)[draw].sum())
    # Independently preserve the event archive's original prediction ordinal.
    # Scene-order concatenation changes synthetic indices for exact score ties
    # and no longer reproduces Python sorted((score, index))[::-1].
    multiplicity = np.bincount(np.asarray(draw, dtype=np.int64), minlength=scene_count)
    selected = np.repeat(
        np.arange(len(event_scenes), dtype=np.int64),
        multiplicity[np.asarray(event_scenes, dtype=np.int64)],
    )
    if len(selected) == 0:
        return no_predictions()
    return event_metric_data(
        scores[selected],
        tp_flags[selected],
        {name: values[selected] for name, values in errors.items()},
        positives,
    )


def direct_clone_boxes(
    eval_boxes_type: Any,
    source_boxes: Any,
    scene_samples: Sequence[Sequence[str]],
    draw: np.ndarray,
    class_name: str,
) -> Any:
    cloned = eval_boxes_type()
    for clone_ordinal, scene_value in enumerate(draw):
        scene_index = int(scene_value)
        require(0 <= scene_index < len(scene_samples), "direct-clone scene index out of range")
        for sample_ordinal, original_token in enumerate(scene_samples[scene_index]):
            clone_token = "rgate-direct-clone-v1|%d|%d|%s" % (clone_ordinal, sample_ordinal, original_token)
            boxes = []
            for original in source_boxes[original_token]:
                if original.detection_name != class_name:
                    continue
                duplicated = copy.copy(original)
                duplicated.sample_token = clone_token
                boxes.append(duplicated)
            cloned.add_boxes(clone_token, boxes)
    return cloned


class OriginalOrdinalCloneBoxes:
    """Minimal EvalBoxes-compatible view with an explicit global box order."""

    def __init__(self, boxes: Mapping[str, Sequence[Any]], ordered: Sequence[Any]):
        self.boxes = {token: list(values) for token, values in boxes.items()}
        self._ordered = list(ordered)

    def __getitem__(self, sample_token: str) -> Sequence[Any]:
        return self.boxes.get(sample_token, [])

    @property
    def sample_tokens(self) -> Sequence[str]:
        return list(self.boxes)

    @property
    def all(self) -> Sequence[Any]:
        return list(self._ordered)


def direct_clone_prediction_boxes(
    source_boxes: Any,
    scene_samples: Sequence[Sequence[str]],
    token_scene: Mapping[str, int],
    draw: np.ndarray,
    class_name: str,
) -> OriginalOrdinalCloneBoxes:
    """Clone predictions while preserving the source's global ordinal.

    The stock EvalBoxes container flattens boxes by its sample-token insertion
    order.  A draw-order construction would therefore obscure the registered
    original-ordinal tie policy.  This independent audit view retains normal
    sample-token lookup for greedy matching while exposing the exact event
    expansion used by the replay engine through ``all``.
    """

    sample_ordinals = {}
    for scene_index, samples in enumerate(scene_samples):
        for sample_ordinal, token in enumerate(samples):
            sample_ordinals[token] = (scene_index, sample_ordinal)
    occurrences = [[] for _ in scene_samples]
    for clone_ordinal, scene_value in enumerate(draw):
        scene_index = int(scene_value)
        require(0 <= scene_index < len(scene_samples), "direct-clone scene index out of range")
        occurrences[scene_index].append(clone_ordinal)

    mapping = {}
    ordered = []
    for original in source_boxes.all:
        if original.detection_name != class_name:
            continue
        original_token = original.sample_token
        require(original_token in sample_ordinals, "direct-clone prediction token missing from scene layout")
        scene_index, sample_ordinal = sample_ordinals[original_token]
        require(token_scene.get(original_token) == scene_index, "direct-clone token/scene mapping mismatch")
        for clone_ordinal in occurrences[scene_index]:
            clone_token = "rgate-direct-clone-v1|%d|%d|%s" % (clone_ordinal, sample_ordinal, original_token)
            duplicated = copy.copy(original)
            duplicated.sample_token = clone_token
            mapping.setdefault(clone_token, []).append(duplicated)
            ordered.append(duplicated)
    return OriginalOrdinalCloneBoxes(mapping, ordered)


def validate_table_lock(lock_path: Path, expected_sha256: str, nusc_root: Path, version: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lock_record = stable_record(lock_path, expected_sha256)
    payload = load_json(lock_path)
    require(payload.get("schema_version") == TABLE_LOCK_SCHEMA and payload.get("status") == "frozen_before_event_execution", "nuScenes table-lock schema/status mismatch")
    require(Path(payload.get("dataroot", "")).absolute() == nusc_root.absolute(), "nuScenes dataroot differs from table lock")
    require(payload.get("version") == version, "nuScenes version differs from table lock")
    require(set(payload.get("tables", {})) == set(NUSCENES_TABLES), "nuScenes table-lock key set mismatch")
    records = {}
    for name in NUSCENES_TABLES:
        path = (nusc_root / version / (name + ".json")).absolute()
        observed = stable_record(path, payload["tables"][name]["sha256"])
        require(observed == payload["tables"][name], "nuScenes table identity mismatch: %s" % name)
        records[name] = observed
    return lock_record, records


def require_within(path: Path, root: Path, message: str) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        raise CheckError(message)


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


def independent_match(
    gt_boxes: Any,
    predictions: Sequence[Any],
    class_name: str,
    threshold: float,
    distance: Any,
    utilities: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    scores = [float(box.detection_score) for box in predictions]
    order = [index for _score, index in sorted((score, index) for index, score in enumerate(scores))][::-1]
    taken = set()
    tp = np.zeros(len(predictions), dtype=np.uint8)
    errors = {name: np.full(len(predictions), np.nan, dtype=np.float64) for name in ERRORS}
    for index in order:
        prediction = predictions[index]
        minimum = np.inf
        matched = None
        for gt_index, gt in enumerate(gt_boxes[prediction.sample_token]):
            key = (prediction.sample_token, gt_index)
            if gt.detection_name == class_name and key not in taken:
                candidate = distance(gt, prediction)
                if candidate < minimum:
                    minimum = candidate
                    matched = gt_index
        if minimum < threshold:
            require(matched is not None, "matched GT index missing")
            taken.add((prediction.sample_token, matched))
            tp[index] = 1
            gt = gt_boxes[prediction.sample_token][matched]
            errors["trans_err"][index] = utilities["center_distance"](gt, prediction)
            errors["vel_err"][index] = utilities["velocity_l2"](gt, prediction)
            errors["scale_err"][index] = 1.0 - utilities["scale_iou"](gt, prediction)
            period = np.pi if class_name == "barrier" else 2.0 * np.pi
            errors["orient_err"][index] = utilities["yaw_diff"](gt, prediction, period=period)
            errors["attr_err"][index] = 1.0 - utilities["attr_acc"](gt, prediction)
    return tp, errors


def compare_array(archive: Any, specs: Mapping[str, Any], key: str, expected: np.ndarray) -> None:
    require(key in archive.files and key in specs, "missing event array: %s" % key)
    observed = np.asarray(archive[key])
    expected_record = array_record(expected)
    require(array_record(observed) == expected_record, "event array differs from independent replay: %s" % key)
    require(specs[key] == expected_record, "manifest array identity mismatch: %s" % key)


def check(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest_path = Path(args.manifest).absolute()
    manifest_record = stable_record(manifest_path, args.expected_manifest_sha256)
    manifest = load_json(manifest_path)
    allowed_root = Path(args.allowed_event_root).absolute()
    require(allowed_root.is_dir(), "allowed event root missing")
    require_within(manifest_path, allowed_root, "event manifest escapes allowed root")
    live_mount = mount_identity(allowed_root)
    expected_mount = {"source": args.expected_event_mount_source, "fstype": args.expected_event_mount_fstype}
    require(live_mount == expected_mount, "event root mount identity mismatch")
    require(manifest.get("schema_version") == EVENT_SCHEMA, "event schema mismatch")
    require(manifest.get("status") == "passed_official_event_trace_closure", "event status mismatch")
    require(manifest.get("arm_label") == args.arm_label, "arm label mismatch")
    require(manifest.get("classes") == list(CLASSES), "class order mismatch")
    require(manifest.get("distance_thresholds") == list(THRESHOLDS), "distance thresholds mismatch")
    require(manifest.get("scene_count") == 150 and manifest.get("sample_count") == 6019, "scene/sample extent mismatch")
    require(manifest.get("per_scene_AP_materialized") is False, "per-scene AP materialization forbidden")
    require(manifest.get("official_evaluator_calls") == 0 and manifest.get("validation_artifacts_modified") == 0, "event builder safety mismatch")
    storage = manifest.get("storage")
    require(isinstance(storage, dict) and set(storage) == {"allowed_output_root", "mount", "minimum_free_bytes", "free_bytes_before_build", "free_bytes_before_publish"}, "event storage witness mismatch")
    require(Path(storage["allowed_output_root"]).absolute() == allowed_root and storage["mount"] == expected_mount, "event storage root/mount mismatch")
    require(type(storage["minimum_free_bytes"]) is int and storage["minimum_free_bytes"] >= 2 * 1024 ** 3, "event storage minimum gate mismatch")
    require(type(storage["free_bytes_before_build"]) is int and type(storage["free_bytes_before_publish"]) is int, "event storage free-byte types mismatch")
    require(storage["free_bytes_before_build"] >= storage["minimum_free_bytes"] and storage["free_bytes_before_publish"] >= storage["minimum_free_bytes"], "event storage free-space gate failed")

    plan_path = Path(args.plan).absolute()
    result_path = Path(args.result_json).absolute()
    summary_path = Path(args.metrics_summary).absolute()
    details_path = Path(args.metrics_details).absolute()
    bootstrap_path = Path(args.bootstrap_core).absolute()
    builder_path = Path(args.event_builder).absolute()
    nusc_root = Path(args.nusc_root).absolute()
    table_lock_record, table_records = validate_table_lock(Path(args.nusc_table_lock).absolute(), args.expected_nusc_table_lock_sha256, nusc_root, args.version)
    plan_record = stable_record(plan_path, args.expected_plan_sha256)
    plan = load_json(plan_path)
    require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
    erratum_record, erratum_evidence, effective_seed0 = validate_identity_erratum(
        Path(args.identity_erratum).absolute(), args.expected_identity_erratum_sha256, plan_path, plan_record, plan
    )
    input_records = {
        "event_builder": stable_record(builder_path, args.expected_event_builder_sha256),
        "plan": plan_record,
        "identity_erratum": erratum_record,
        "identity_erratum_supporting_evidence": erratum_evidence,
        "result": stable_record(result_path, args.expected_result_sha256),
        "metrics_summary": stable_record(summary_path, args.expected_metrics_summary_sha256),
        "metrics_details": stable_record(details_path, args.expected_metrics_details_sha256),
        "bootstrap_core": stable_record(bootstrap_path, args.expected_bootstrap_core_sha256),
        "nusc_table_lock": table_lock_record,
        "nusc_tables": table_records,
    }
    for name, expected in input_records.items():
        observed = manifest["inputs"][name]
        require(observed == expected, "manifest input identity mismatch: %s" % name)
    seed0 = plan["frozen_inputs_and_reuse"]["seed0_artifacts"]["prediction_sha256"]
    if args.arm_label in effective_seed0:
        require(effective_seed0[args.arm_label] == args.expected_result_sha256, "plan+erratum arm identity mismatch")
    require(
        manifest.get("seed0_prediction_identity_resolution") == {
            "plan_value": seed0.get(args.arm_label),
            "effective_value": effective_seed0.get(args.arm_label),
            "identity_erratum_applied": args.arm_label == "E4",
        },
        "manifest seed0 identity resolution mismatch",
    )
    draw_path = Path(args.draw_matrix).absolute()
    draw_record = stable_record(draw_path, args.expected_draw_sha256)
    draw_contract = plan["scene_bootstrap"]["draw_generator"]
    require(draw_record["sha256"] == draw_contract["expected_draw_matrix_sha256"], "plan draw-matrix SHA mismatch")
    require(draw_record["size_bytes"] == draw_contract["draw_matrix_size_bytes"], "plan draw-matrix size mismatch")
    draw_values = np.frombuffer(draw_path.read_bytes(), dtype="<u2")
    require(draw_values.size == 1000 * 150, "draw matrix extent mismatch")
    direct_reference_config_record, direct_reference_indices = validate_direct_reference_config(
        Path(args.direct_reference_config).absolute(),
        args.expected_direct_reference_config_sha256,
        plan_record,
        draw_record,
    )
    draw_matrix = draw_values.reshape(1000, 150)

    npz_spec = manifest["npz"]
    npz_path = (manifest_path.parent / npz_spec["path"]).absolute()
    npz_record = stable_record(npz_path, npz_spec["sha256"])
    require(npz_record["size_bytes"] == npz_spec["size_bytes"], "event NPZ size mismatch")

    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
    from nuscenes.eval.common.utils import attr_acc, center_distance, scale_iou, velocity_l2, yaw_diff
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.utils.splits import create_splits_scenes

    config = config_factory("detection_cvpr_2019")
    require(tuple(config.class_names) == CLASSES, "devkit class order drift")
    require(tuple(float(value) for value in config.dist_ths) == THRESHOLDS, "devkit thresholds drift")
    require(float(config.dist_th_tp) == 2.0 and float(config.min_recall) == 0.1 and float(config.min_precision) == 0.1, "devkit metric constants drift")
    nusc = NuScenes(version=args.version, dataroot=str(nusc_root), verbose=False)
    pred_boxes, _meta = load_prediction(str(result_path), config.max_boxes_per_sample, DetectionBox, verbose=False)
    gt_boxes = load_gt(nusc, args.eval_set, DetectionBox, verbose=False)
    require(set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), "prediction/GT token mismatch")
    pred_boxes, gt_boxes = add_center_dist(nusc, pred_boxes), add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, config.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, config.class_range, verbose=False)
    sample_tokens = list(pred_boxes.sample_tokens)
    require(len(sample_tokens) == 6019, "prediction sample count mismatch")

    scene_names = list(create_splits_scenes()[args.eval_set])
    require(len(scene_names) == 150 and len(set(scene_names)) == 150, "validation split scene-name extent mismatch")
    scene_by_name = {str(row["name"]): str(row["token"]) for row in nusc.scene}
    require(all(name in scene_by_name for name in scene_names), "validation split scene missing from nuScenes table")
    scenes = [scene_by_name[name] for name in scene_names]
    scene_index = {token: index for index, token in enumerate(scenes)}
    token_scene = {}
    scene_samples = [[] for _ in scenes]
    for token in sample_tokens:
        scene = str(nusc.get("sample", token)["scene_token"])
        require(scene in scene_index, "prediction sample belongs outside registered validation scenes")
        token_scene[token] = scene_index[scene]
        scene_samples[scene_index[scene]].append(token)
    require(len(sample_tokens) == len(set(sample_tokens)) == 6019, "prediction sample extent/uniqueness mismatch")
    require(all(scene_samples), "prediction result does not cover every validation scene")
    require(manifest["scene_tokens"] == scenes, "scene-token order mismatch")
    require(manifest["scene_tokens_sha256"] == hashlib.sha256(canonical_bytes(scenes)).hexdigest(), "scene-token hash mismatch")
    require(manifest.get("scene_names") == scene_names, "scene-name order mismatch")
    require(manifest.get("scene_names_sha256") == hashlib.sha256(canonical_bytes(scene_names)).hexdigest(), "scene-name hash mismatch")
    require(manifest.get("scene_order_policy") == "nuscenes_devkit_create_splits_scenes_eval_set_order_mapped_name_to_locked_scene_token", "scene-order policy mismatch")
    require(manifest.get("sample_order_policy") == "result_json_original_sample_and_box_order_without_scene_contiguity_requirement", "sample-order policy mismatch")

    utilities = {
        "center_distance": center_distance,
        "velocity_l2": velocity_l2,
        "scale_iou": scale_iou,
        "yaw_diff": yaw_diff,
        "attr_acc": attr_acc,
    }
    official_details = load_official_metrics_json(details_path)
    official_summary = load_official_metrics_json(summary_path)
    expected_array_keys = set()
    maximum_detail_difference = 0.0
    direct_clone_maximum_differences = {index: 0.0 for index in direct_reference_indices}
    class_counts = {}
    all_predictions = pred_boxes.all
    all_ground_truth = gt_boxes.all
    with np.load(str(npz_path), allow_pickle=False) as archive:
        for class_name in CLASSES:
            predictions = [box for box in all_predictions if box.detection_name == class_name]
            scores = np.asarray([float(box.detection_score) for box in predictions], dtype=np.float64)
            event_scenes = np.asarray([token_scene[box.sample_token] for box in predictions], dtype=np.uint16)
            ordinals = np.arange(len(predictions), dtype=np.uint32)
            gt_counts = np.zeros(150, dtype=np.uint32)
            for box in all_ground_truth:
                if box.detection_name == class_name:
                    gt_counts[token_scene[box.sample_token]] += 1
            base_arrays = {
                "scene__%s" % class_name: event_scenes,
                "score__%s" % class_name: scores,
                "ordinal__%s" % class_name: ordinals,
                "gt__%s" % class_name: gt_counts,
            }
            for key, value in base_arrays.items():
                compare_array(archive, manifest["arrays"], key, value)
                expected_array_keys.add(key)
            class_counts[class_name] = {"predictions": len(predictions), "ground_truth": int(gt_counts.sum())}
            for threshold in THRESHOLDS:
                tp, errors = independent_match(gt_boxes, predictions, class_name, threshold, config.dist_fcn_callable, utilities)
                suffix = threshold_key(threshold)
                tp_key = "tp__%s__%s" % (class_name, suffix)
                compare_array(archive, manifest["arrays"], tp_key, tp)
                expected_array_keys.add(tp_key)
                for error_name, values in errors.items():
                    key = "err__%s__%s__%s" % (class_name, suffix, error_name)
                    compare_array(archive, manifest["arrays"], key, values)
                    expected_array_keys.add(key)
                reconstructed = event_metric_data(scores, tp, errors, int(gt_counts.sum()))
                registered = official_details["%s:%.1f" % (class_name, threshold)]
                for field in ("recall", "precision", "confidence") + ERRORS:
                    left = reconstructed[field]
                    right = np.asarray(registered[field], dtype=np.float64)
                    require(left.shape == right.shape and np.array_equal(np.isnan(left), np.isnan(right)), "metric-details shape/NaN mismatch")
                    finite = ~np.isnan(left)
                    difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
                    maximum_detail_difference = max(maximum_detail_difference, difference)
            for draw_index in direct_reference_indices:
                direct_draw = draw_matrix[draw_index]
                direct_gt = direct_clone_boxes(EvalBoxes, gt_boxes, scene_samples, direct_draw, class_name)
                direct_predictions = direct_clone_prediction_boxes(
                    pred_boxes,
                    scene_samples,
                    token_scene,
                    direct_draw,
                    class_name,
                )
                for threshold in THRESHOLDS:
                    suffix = threshold_key(threshold)
                    tp = np.asarray(archive["tp__%s__%s" % (class_name, suffix)])
                    errors = {
                        error_name: np.asarray(archive["err__%s__%s__%s" % (class_name, suffix, error_name)])
                        for error_name in ERRORS
                    }
                    replayed_draw = draw_metric_data(scores, tp, errors, event_scenes, gt_counts, direct_draw, 150)
                    direct = accumulate(direct_gt, direct_predictions, class_name, config.dist_fcn_callable, threshold, verbose=False).serialize()
                    for field in ("recall", "precision", "confidence") + ERRORS:
                        left = replayed_draw[field]
                        right = np.asarray(direct[field], dtype=np.float64)
                        require(left.shape == right.shape and np.array_equal(np.isnan(left), np.isnan(right)), "direct-clone shape/NaN mismatch at draw %d" % draw_index)
                        finite = ~np.isnan(left)
                        difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
                        direct_clone_maximum_differences[draw_index] = max(direct_clone_maximum_differences[draw_index], difference)
        require(set(archive.files) == expected_array_keys, "unexpected/missing event arrays in NPZ")
    require(set(manifest["arrays"]) == expected_array_keys, "manifest event-array key set mismatch")
    require(manifest["class_counts"] == class_counts, "class count summary mismatch")
    require(maximum_detail_difference <= 1e-12, "independent metric-details closure exceeds 1e-12")
    require(max(direct_clone_maximum_differences.values()) <= 1e-12, "direct-reference clone closure exceeds 1e-12")
    official_values = {
        "mean_ap": float(official_summary["mean_ap"]),
        "nd_score": float(official_summary["nd_score"]),
        "mAVE": float(official_summary["tp_errors"]["vel_err"]),
    }
    require(manifest["official_summary"] == official_values, "official summary witness mismatch")
    require(max(float(value) for value in manifest["summary_abs_diffs"].values()) <= 1e-12, "builder summary closure failed")
    require(float(manifest["metric_details_max_abs_diff"]) <= 1e-12, "builder detail closure failed")
    checker_record = stable_record(Path(__file__).absolute(), args.expected_event_checker_sha256)
    return {
        "schema_version": CHECK_SCHEMA,
        "status": "passed_independent_raw_event_and_official_metric_closure",
        "passed": True,
        "arm_label": args.arm_label,
        "event_manifest": manifest_record,
        "event_npz": npz_record,
        "implementation": {"event_builder": input_records["event_builder"], "event_checker": checker_record},
        "inputs": input_records,
        "draw_matrix": draw_record,
        "direct_reference_config": direct_reference_config_record,
        "direct_reference_draw_indices": direct_reference_indices,
        "direct_reference_draw_closure": [
            {
                "draw_index": draw_index,
                "metric_details_max_abs_diff": direct_clone_maximum_differences[draw_index],
                "passed_at_tolerance_1e_12": direct_clone_maximum_differences[draw_index] <= 1e-12,
            }
            for draw_index in direct_reference_indices
        ],
        "scene_count": 150,
        "sample_count": 6019,
        "class_count": len(CLASSES),
        "distance_threshold_count": len(THRESHOLDS),
        "event_array_count": len(expected_array_keys),
        "independent_metric_details_max_abs_diff": maximum_detail_difference,
        "direct_reference_metric_details_max_abs_diff": max(direct_clone_maximum_differences.values()),
        "all_raw_event_arrays_byte_exact": True,
        "per_scene_AP_average_used": False,
        "official_evaluator_calls": 0,
        "validation_artifacts_modified": 0,
    }


def publish_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.absolute()
    require(target.parent.is_dir() and not os.path.lexists(str(target)), "checker output contract failed")
    data = canonical_bytes(payload)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=".%s.tmp." % target.name, dir=str(target.parent))
    temporary = Path(raw_temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
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
    parser.add_argument("--event-builder", required=True)
    parser.add_argument("--expected-event-builder-sha256", required=True)
    parser.add_argument("--expected-event-checker-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--identity-erratum", required=True)
    parser.add_argument("--expected-identity-erratum-sha256", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--metrics-summary", required=True)
    parser.add_argument("--expected-metrics-summary-sha256", required=True)
    parser.add_argument("--metrics-details", required=True)
    parser.add_argument("--expected-metrics-details-sha256", required=True)
    parser.add_argument("--draw-matrix", required=True)
    parser.add_argument("--expected-draw-sha256", required=True)
    parser.add_argument("--direct-reference-config", required=True)
    parser.add_argument("--expected-direct-reference-config-sha256", required=True)
    parser.add_argument("--bootstrap-core", required=True)
    parser.add_argument("--expected-bootstrap-core-sha256", required=True)
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--nusc-root", required=True)
    parser.add_argument("--nusc-table-lock", required=True)
    parser.add_argument("--expected-nusc-table-lock-sha256", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--allowed-event-root", required=True)
    parser.add_argument("--expected-event-mount-source", required=True)
    parser.add_argument("--expected-event-mount-fstype", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        payload = check(args)
        publish_exclusive(Path(args.output), payload)
    except (CheckError, KeyError, OSError, ValueError) as exc:
        print("FAILED: %s" % exc, file=os.sys.stderr)
        return 1
    print(str(Path(args.output).absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
