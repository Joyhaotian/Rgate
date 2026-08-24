#!/usr/bin/env python3
"""Normalize one registered learned JSON artifact into a new external stage.

The source is authenticated against ARTIFACT_MANIFEST.json before JSON parsing.
Only the metadata pointers named by the v2 contract are rewritten.  The tool
then proves canonical inference-payload equality and exercises the bundled
inference implementation on a deterministic synthetic fixture.  It never
edits the manifest or the candidate's models directory.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_APPLY_SOURCE_PATH = SCRIPT_DIR / "apply_bge_af_arbiter.py"
_APPLY_BYTES_BEFORE_IMPORT = _APPLY_SOURCE_PATH.read_bytes()

from apply_bge_af_arbiter import (  # noqa: E402
    CALIBRATION_TABLE_SCHEMA,
    LINEAR_MODEL_SCHEMA,
    LIGHTGBM_MODEL_SCHEMA,
    MLP_MODEL_SCHEMA,
    calibrate_score,
    calibration_params_for_row,
    model_score,
    normalize_range_edges,
    range_bin_for_value,
    row_calibration_key,
)

_APPLY_BYTES_AFTER_IMPORT = _APPLY_SOURCE_PATH.read_bytes()
if _APPLY_BYTES_BEFORE_IMPORT != _APPLY_BYTES_AFTER_IMPORT:
    raise RuntimeError("inference implementation changed while it was imported")
_LOADED_APPLY_BYTES = _APPLY_BYTES_AFTER_IMPORT
_LOADED_NORMALIZER_BYTES = Path(__file__).resolve().read_bytes()


MANIFEST_SCHEMA = "rgate_release_artifact_manifest_v2"
CONTRACT_SCHEMA = "rgate_release_metadata_normalization_v1"
RECEIPT_SCHEMA = "rgate_release_normalization_receipt_v1"
MODEL_FIXTURE_SCHEMA = "rgate_model_equivalence_fixture_v1"
CALIBRATION_FIXTURE_SCHEMA = "rgate_calibration_equivalence_fixture_v1"
CANONICALIZATION = "utf8_json_sorted_keys_compact_no_nan_v1"
EXPECTED_POINTERS = {
    "model": ["/training/cache_jsonl"],
    "calibration": ["/cache_jsonl", "/model"],
}
EXPECTED_PUBLIC_PROOFS = [
    "public_artifact_identity",
    "source_inference_parameter_sha256",
    "public_inference_parameter_sha256",
    "fixture_equivalence_receipt_sha256",
]
CACHE_PUBLIC_VALUE = "artifacts/cache/fullval_cache.jsonl"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_ID_RE = re.compile(r"[a-z0-9_]+\Z")
MODEL_PATH_RE = re.compile(
    r"models/seed_[0-9]{2}/(?:no_radar_|radar_)(?:model|calibration)\.json\Z"
)
WINDOWS_ABSOLUTE_RE = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
POSIX_ABSOLUTE_RE = re.compile(r"(?:^|[^A-Za-z0-9._/~-])/{1,2}[^/\s]")
TILDE_PATH_RE = re.compile(r"(?:^|[^A-Za-z0-9._~-])~/")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ENVIRONMENT_ROOTS = (
    "/" + "ho" + "me" + "/",
    "/" + "m" + "nt" + "/",
    "/" + "U" + "sers" + "/",
)


class NormalizationError(ValueError):
    """A closed-gate normalization failure safe to show without source data."""


class DuplicateKeyError(ValueError):
    pass


def _no_duplicate_object(items: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise DuplicateKeyError("duplicate JSON key")
        result[key] = value
    return result


def load_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NormalizationError("invalid %s JSON: %s" % (label, exc)) from None


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NormalizationError("value is not canonical finite JSON: %s" % exc) from None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if WINDOWS_ABSOLUTE_RE.search(value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and "." not in path.parts and ".." not in path.parts


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _reject_symlink_components(path: Path, include_leaf: bool, label: str) -> None:
    """Reject symlinks in the caller-supplied lexical path."""

    absolute = path.absolute()
    components = list(reversed(absolute.parents))
    if include_leaf:
        components.append(absolute)
    for component in components:
        if component == Path(component.anchor):
            continue
        if component.is_symlink():
            raise NormalizationError("%s path contains a symbolic-link component" % label)


def _read_regular_authenticated(path: Path, expected_size: int, expected_sha: str) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        raise NormalizationError("source is unavailable or not a direct regular file") from None
    try:
        before = os.fstat(descriptor)
        if before.st_mode & 0o170000 != 0o100000:
            raise NormalizationError("source is not a regular file")
        if before.st_size != expected_size:
            raise NormalizationError("source size does not match the private registry")
        chunks: List[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_fields = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_fields != after_fields or len(payload) != expected_size:
        raise NormalizationError("source changed while it was authenticated")
    if sha256_bytes(payload) != expected_sha:
        raise NormalizationError("source SHA-256 does not match the private registry")
    return payload


def _bound_implementation_identities() -> Dict[str, Dict[str, Any]]:
    sources = (
        ("inference", _APPLY_SOURCE_PATH, "scripts/apply_bge_af_arbiter.py", _LOADED_APPLY_BYTES),
        (
            "normalizer",
            Path(__file__).resolve(),
            "tools/normalize_learned_artifact.py",
            _LOADED_NORMALIZER_BYTES,
        ),
    )
    result = {}
    for key, path, relative, loaded_bytes in sources:
        loaded_sha = sha256_bytes(loaded_bytes)
        try:
            observed = _read_regular_authenticated(path, len(loaded_bytes), loaded_sha)
        except NormalizationError:
            raise NormalizationError("implementation bytes differ from the loaded snapshot") from None
        if observed != loaded_bytes:
            raise NormalizationError("implementation bytes differ from the loaded snapshot")
        result[key] = {
            "relative_path": relative,
            "size_bytes": len(loaded_bytes),
            "sha256": loaded_sha,
        }
    return result


def _pointer_parts(pointer: str) -> List[str]:
    if not pointer.startswith("/") or pointer == "/":
        raise NormalizationError("invalid JSON pointer in normalization contract")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _get_pointer(value: Any, pointer: str) -> Any:
    current = value
    for part in _pointer_parts(pointer):
        if not isinstance(current, dict) or part not in current:
            raise NormalizationError("required normalization pointer is absent: %s" % pointer)
        current = current[part]
    return current


def _set_pointer(value: Any, pointer: str, replacement: str) -> None:
    parts = _pointer_parts(pointer)
    current = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise NormalizationError("required normalization pointer is absent: %s" % pointer)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise NormalizationError("required normalization pointer is absent: %s" % pointer)
    if not isinstance(current[parts[-1]], str):
        raise NormalizationError("normalization source metadata must be a string: %s" % pointer)
    if current[parts[-1]] == replacement:
        raise NormalizationError("normalization pointer was already public: %s" % pointer)
    current[parts[-1]] = replacement


def _delete_pointer(value: Any, pointer: str) -> None:
    parts = _pointer_parts(pointer)
    current = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise NormalizationError("required normalization pointer is absent: %s" % pointer)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise NormalizationError("required normalization pointer is absent: %s" % pointer)
    del current[parts[-1]]


def _escape_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_differences(left: Any, right: Any, pointer: str = "") -> List[str]:
    if type(left) is not type(right):
        return [pointer]
    if isinstance(left, dict):
        if set(left) != set(right):
            changed = []
            for key in sorted(set(left) | set(right)):
                child = pointer + "/" + _escape_pointer_part(key)
                if key not in left or key not in right:
                    changed.append(child)
                else:
                    changed.extend(_json_differences(left[key], right[key], child))
            return changed
        changed = []
        for key in sorted(left):
            child = pointer + "/" + _escape_pointer_part(key)
            changed.extend(_json_differences(left[key], right[key], child))
        return changed
    if isinstance(left, list):
        if len(left) != len(right):
            return [pointer]
        changed = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            changed.extend(_json_differences(left_item, right_item, pointer + "/" + str(index)))
        return changed
    return [] if left == right else [pointer]


def _inference_payload_sha(value: Dict[str, Any], pointers: List[str]) -> str:
    payload = copy.deepcopy(value)
    for pointer in pointers:
        _delete_pointer(payload, pointer)
    return sha256_bytes(canonical_json_bytes(payload))


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _verify_public_strings(value: Any) -> None:
    for text in _walk_strings(value):
        lowered = text.lower()
        if (
            text.startswith("/")
            or "\\" in text
            or POSIX_ABSOLUTE_RE.search(text)
            or TILDE_PATH_RE.search(text)
            or WINDOWS_ABSOLUTE_RE.search(text)
            or "file://" in lowered
        ):
            raise NormalizationError("public artifact retains an absolute path string")
        if any(root.lower() in lowered for root in ENVIRONMENT_ROOTS):
            raise NormalizationError("public artifact retains an environment-root path string")
        if EMAIL_RE.search(text):
            raise NormalizationError("public artifact retains an email-like string")


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise NormalizationError("invalid numeric model field: %s" % label)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise NormalizationError("invalid numeric model field: %s" % label) from None
    if not math.isfinite(number):
        raise NormalizationError("non-finite numeric model field: %s" % label)
    return number


def _validate_model(model: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]], int]:
    schema = model.get("schema_version")
    if schema == LIGHTGBM_MODEL_SCHEMA:
        raise NormalizationError("LightGBM artifacts are outside the standard-library proof path")
    if schema not in (LINEAR_MODEL_SCHEMA, MLP_MODEL_SCHEMA):
        raise NormalizationError("unsupported model schema for equivalence proof")
    numeric = model.get("numeric_features")
    if (
        not isinstance(numeric, list)
        or any(not isinstance(name, str) or not name for name in numeric)
        or len(set(numeric)) != len(numeric)
    ):
        raise NormalizationError("invalid or duplicate numeric feature names")
    stats = model.get("stats")
    if not isinstance(stats, dict) or set(numeric) - set(stats):
        raise NormalizationError("model statistics do not cover numeric features")
    for name in numeric:
        block = stats[name]
        if not isinstance(block, dict):
            raise NormalizationError("invalid statistics block")
        _require_finite_number(block.get("mean"), "stats.mean")
        _require_finite_number(block.get("std"), "stats.std")
    levels = model.get("categorical_levels")
    if not isinstance(levels, dict):
        raise NormalizationError("invalid categorical levels")
    clean_levels: Dict[str, List[str]] = {}
    for key, values in levels.items():
        if not isinstance(key, str) or not key or not isinstance(values, list):
            raise NormalizationError("invalid categorical level entry")
        if any(not isinstance(item, str) for item in values) or len(set(values)) != len(values):
            raise NormalizationError("invalid or duplicate categorical levels")
        clean_levels[key] = list(values)
    input_dimension = 1 + len(numeric) + sum(len(clean_levels[key]) for key in sorted(clean_levels))
    if schema == LINEAR_MODEL_SCHEMA:
        weights = model.get("weights")
        if not isinstance(weights, list) or len(weights) != input_dimension:
            raise NormalizationError("linear weight dimension mismatch")
        for value in weights:
            _require_finite_number(value, "weights")
    else:
        first = model.get("input_hidden_weights")
        hidden_bias = model.get("hidden_bias")
        output = model.get("output_weights")
        if not isinstance(first, list) or not first:
            raise NormalizationError("invalid MLP input weights")
        hidden_dimension = len(first)
        if (
            not isinstance(hidden_bias, list)
            or not isinstance(output, list)
            or len(hidden_bias) != hidden_dimension
            or len(output) != hidden_dimension
        ):
            raise NormalizationError("MLP hidden dimension mismatch")
        if model.get("hidden_dim") is not None and model.get("hidden_dim") != hidden_dimension:
            raise NormalizationError("MLP declared hidden dimension mismatch")
        for row in first:
            if not isinstance(row, list) or len(row) != input_dimension:
                raise NormalizationError("MLP input dimension mismatch")
            for value in row:
                _require_finite_number(value, "input_hidden_weights")
        for value in hidden_bias:
            _require_finite_number(value, "hidden_bias")
        for value in output:
            _require_finite_number(value, "output_weights")
        _require_finite_number(model.get("output_bias"), "output_bias")
        if model.get("activation") not in (None, "tanh"):
            raise NormalizationError("unsupported MLP activation")
    return list(numeric), clean_levels, input_dimension


def _model_fixture(model: Dict[str, Any]) -> Dict[str, Any]:
    numeric, levels, _ = _validate_model(model)
    rows = []
    for row_index in range(4):
        features = {}
        for index, name in enumerate(numeric):
            multiplier = float(index + 1)
            candidates = (0.0, 0.125 * multiplier, -0.25 * multiplier, 0.5 - 0.0625 * multiplier)
            features[name] = candidates[row_index]
        categorical = {}
        for key in sorted(levels):
            choices = levels[key]
            if row_index == 0 or not choices:
                categorical[key] = ""
            elif row_index == 1:
                categorical[key] = choices[0]
            elif row_index == 2:
                categorical[key] = choices[-1]
            else:
                categorical[key] = "__synthetic_unseen__"
        rows.append({"features": features, "categorical": categorical})
    return {"schema_version": MODEL_FIXTURE_SCHEMA, "rows": rows}


def _model_outputs(model: Dict[str, Any], fixture: Dict[str, Any]) -> bytes:
    outputs = []
    for index, row in enumerate(fixture["rows"]):
        try:
            score = float(model_score(row, model))
        except (KeyError, IndexError, TypeError, ValueError, SystemExit) as exc:
            raise NormalizationError("bundled model inference rejected the synthetic fixture: %s" % exc) from None
        if not math.isfinite(score):
            raise NormalizationError("bundled model inference produced a non-finite score")
        outputs.append({"row_index": index, "score_float_hex": score.hex()})
    return canonical_json_bytes(outputs)


def _range_representatives(edges: List[float]) -> Dict[str, float]:
    candidates = [edges[0] - 1.0]
    candidates.extend((lower + upper) / 2.0 for lower, upper in zip(edges, edges[1:]))
    candidates.append(edges[-1] + 1.0)
    result = {}
    for value in candidates:
        result[range_bin_for_value(value, edges)] = value
    return result


def _calibration_fixture(table: Dict[str, Any]) -> Dict[str, Any]:
    if table.get("schema_version") != CALIBRATION_TABLE_SCHEMA:
        raise NormalizationError("unsupported calibration table schema")
    bins = table.get("bins")
    grouping = table.get("grouping")
    global_block = table.get("global")
    if not isinstance(bins, list) or not isinstance(grouping, dict) or not isinstance(global_block, dict):
        raise NormalizationError("calibration table structure is incomplete")
    raw_edges = grouping.get("range_edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise NormalizationError("calibration range edges are absent")
    edges = normalize_range_edges(raw_edges)
    if len(edges) != len(raw_edges):
        raise NormalizationError("calibration range edges are invalid or duplicate")
    representatives = _range_representatives(edges)
    rows = []
    keys = set()
    for item in bins:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise NormalizationError("invalid calibration bin")
        key = item["key"]
        if key in keys:
            raise NormalizationError("duplicate calibration bin key")
        keys.add(key)
        parts = key.split("|")
        if (
            len(parts) != 3
            or not parts[0].startswith("class=")
            or not parts[1].startswith("range=")
            or not parts[2].startswith("source=")
        ):
            raise NormalizationError("calibration bin key is not reproducible")
        class_name = parts[0][len("class=") :]
        range_name = parts[1][len("range=") :]
        source_name = parts[2][len("source=") :]
        if not class_name or not source_name or range_name not in representatives:
            raise NormalizationError("calibration bin key is outside the fixed fixture grammar")
        if item.get("class_name") != class_name or item.get("range_bin") != range_name:
            raise NormalizationError("calibration bin metadata disagrees with its key")
        if item.get("source_signature") != source_name:
            raise NormalizationError("calibration source metadata disagrees with its key")
        row = {
            "class_name": class_name,
            "features": {"ego_range_xy": representatives[range_name]},
            "categorical": {"source_signature": source_name},
        }
        if row_calibration_key(row, edges) != key:
            raise NormalizationError("calibration bin cannot be reached through the bundled path")
        rows.append(row)
    suffix = 0
    fallback_range = range_bin_for_value(edges[-1] + 1.0, edges)
    while True:
        fallback = {
            "class_name": "__synthetic_unseen_%d__" % suffix,
            "features": {"ego_range_xy": edges[-1] + 1.0},
            "categorical": {"source_signature": "__synthetic_unseen__"},
        }
        if row_calibration_key(fallback, edges) not in keys:
            break
        suffix += 1
    rows.append(fallback)
    return {
        "schema_version": CALIBRATION_FIXTURE_SCHEMA,
        "expected_bin_keys": sorted(keys),
        "fallback_range_bin": fallback_range,
        "probe_scores": [0.03125, 0.25, 0.5, 0.8125, 0.96875],
        "rows": rows,
    }


def _prepare_calibration(table: Dict[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(table)
    if "_bin_by_key" in prepared:
        raise NormalizationError("calibration table contains a reserved runtime key")
    bins = prepared.get("bins")
    assert isinstance(bins, list)
    prepared["_bin_by_key"] = {str(item["key"]): item for item in bins}
    return prepared


def _calibration_outputs(table: Dict[str, Any], fixture: Dict[str, Any]) -> bytes:
    prepared = _prepare_calibration(table)
    records = []
    observed_bin_keys = set()
    for row_index, row in enumerate(fixture["rows"]):
        try:
            temperature, power, cap, source = calibration_params_for_row(
                row=row,
                table=prepared,
                default_temperature=1.0,
                default_power=1.0,
                default_cap=1.0,
            )
        except (KeyError, IndexError, TypeError, ValueError, SystemExit) as exc:
            raise NormalizationError("bundled calibration path rejected the synthetic fixture: %s" % exc) from None
        if source == "bin":
            edges = normalize_range_edges(prepared["grouping"]["range_edges"])
            observed_bin_keys.add(row_calibration_key(row, edges))
        parameters = (temperature, power, cap)
        if any(not math.isfinite(float(value)) for value in parameters):
            raise NormalizationError("calibration path produced non-finite parameters")
        for score in fixture["probe_scores"]:
            output = calibrate_score(
                score=score,
                temperature=temperature,
                power=power,
                cap=cap,
            )
            if not math.isfinite(output):
                raise NormalizationError("calibration path produced a non-finite score")
            records.append(
                {
                    "row_index": row_index,
                    "probe_float_hex": float(score).hex(),
                    "temperature_float_hex": float(temperature).hex(),
                    "power_float_hex": float(power).hex(),
                    "cap_float_hex": float(cap).hex(),
                    "source": source,
                    "output_float_hex": float(output).hex(),
                }
            )
    if observed_bin_keys != set(fixture["expected_bin_keys"]):
        raise NormalizationError("fixed fixture did not exercise every calibration bin")
    return canonical_json_bytes(records)


def _equivalence_proof(
    kind: str, source: Dict[str, Any], public: Dict[str, Any]
) -> Dict[str, Any]:
    if kind == "model":
        source_fixture = _model_fixture(source)
        public_fixture = _model_fixture(public)
        method = "apply_bge_af_arbiter.model_score_exact_float_hex_v1"
        source_outputs = _model_outputs(source, source_fixture)
        public_outputs = _model_outputs(public, public_fixture)
        probe_count = len(source_fixture["rows"])
    else:
        source_fixture = _calibration_fixture(source)
        public_fixture = _calibration_fixture(public)
        method = "apply_bge_af_arbiter.calibration_path_exact_float_hex_v1"
        source_outputs = _calibration_outputs(source, source_fixture)
        public_outputs = _calibration_outputs(public, public_fixture)
        probe_count = len(source_fixture["rows"]) * len(source_fixture["probe_scores"])
    source_fixture_bytes = canonical_json_bytes(source_fixture)
    public_fixture_bytes = canonical_json_bytes(public_fixture)
    if source_fixture_bytes != public_fixture_bytes:
        raise NormalizationError("source/public synthetic fixtures differ")
    if source_outputs != public_outputs:
        raise NormalizationError("source/public fixed-fixture outputs differ")
    return {
        "schema_version": (
            MODEL_FIXTURE_SCHEMA if kind == "model" else CALIBRATION_FIXTURE_SCHEMA
        ),
        "method": method,
        "comparison": "exact_canonical_float_hex",
        "probe_count": probe_count,
        "fixture_sha256": sha256_bytes(source_fixture_bytes),
        "source_output_sha256": sha256_bytes(source_outputs),
        "public_output_sha256": sha256_bytes(public_outputs),
    }


def _load_manifest_registration(
    manifest_path: Path, artifact_id: str
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[str]]:
    try:
        payload = manifest_path.read_bytes()
    except OSError:
        raise NormalizationError("artifact manifest is unavailable") from None
    manifest = load_json_bytes(payload, "artifact manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise NormalizationError("unexpected artifact manifest schema")
    contract = manifest.get("normalization_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != CONTRACT_SCHEMA:
        raise NormalizationError("unexpected normalization contract schema")
    if contract.get("allowed_changed_json_pointers") != EXPECTED_POINTERS:
        raise NormalizationError("normalization pointer allowlist differs from the audited contract")
    if contract.get("required_public_proofs") != EXPECTED_PUBLIC_PROOFS:
        raise NormalizationError("public proof list differs from the audited contract")
    artifacts = manifest.get("artifacts")
    public_artifacts = manifest.get("public_artifacts")
    if not isinstance(artifacts, list) or not isinstance(public_artifacts, list):
        raise NormalizationError("artifact registries must be lists")
    private_ids = [item.get("id") for item in artifacts if isinstance(item, dict)]
    private_paths = [item.get("relative_path") for item in artifacts if isinstance(item, dict)]
    public_ids = [item.get("id") for item in public_artifacts if isinstance(item, dict)]
    if (
        len(private_ids) != len(artifacts)
        or any(not isinstance(value, str) for value in private_ids)
        or any(not isinstance(value, str) for value in private_paths)
        or any(not isinstance(value, str) for value in public_ids)
        or len(private_ids) != len(set(private_ids))
        or len(private_paths) != len(set(private_paths))
        or len(public_ids) != len(public_artifacts)
        or len(public_ids) != len(set(public_ids))
    ):
        raise NormalizationError("artifact registries contain invalid or duplicate entries")
    if any(isinstance(item, dict) and item.get("id") == artifact_id for item in public_artifacts):
        raise NormalizationError("artifact already has a public registration")
    matches = [item for item in artifacts if isinstance(item, dict) and item.get("id") == artifact_id]
    if len(matches) != 1:
        raise NormalizationError("artifact id must resolve to exactly one private registry entry")
    item = matches[0]
    relative = item.get("relative_path")
    if (
        not ARTIFACT_ID_RE.fullmatch(artifact_id)
        or not _safe_relative(relative)
        or not MODEL_PATH_RE.fullmatch(relative)
    ):
        raise NormalizationError("artifact id or registered path is outside the model layout")
    if item.get("required") is not True or item.get("availability") != "missing_pending_verified_copy":
        raise NormalizationError("private registry state is not eligible for normalization")
    expected_size = item.get("size_bytes")
    expected_sha = item.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 0 or not _valid_sha256(expected_sha):
        raise NormalizationError("private registry identity is invalid")
    kind = "calibration" if relative.endswith("_calibration.json") else "model"
    pointers = EXPECTED_POINTERS[kind]
    return manifest, item, kind, list(pointers)


def _replacement_values(
    manifest: Dict[str, Any], relative: str, kind: str
) -> Dict[str, str]:
    if kind == "model":
        return {"/training/cache_jsonl": CACHE_PUBLIC_VALUE}
    model_relative = relative[: -len("_calibration.json")] + "_model.json"
    registered_paths = {
        item.get("relative_path")
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    if model_relative not in registered_paths:
        raise NormalizationError("calibration's paired model is absent from the private registry")
    replacements = {
        "/cache_jsonl": CACHE_PUBLIC_VALUE,
        "/model": model_relative,
    }
    if any(not _safe_relative(value) for value in replacements.values()):
        raise NormalizationError("derived public metadata is not relative")
    return replacements


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _directory_fd_identity(descriptor: int) -> Tuple[int, int]:
    status = os.fstat(descriptor)
    if status.st_mode & 0o170000 != 0o040000:
        raise NormalizationError("anchored transaction object is not a directory")
    return status.st_dev, status.st_ino


def _verify_path_matches_directory_fd(path: Path, descriptor: int, label: str) -> None:
    try:
        status = os.lstat(str(path))
    except OSError:
        raise NormalizationError("%s directory path disappeared during the transaction" % label) from None
    if status.st_mode & 0o170000 != 0o040000:
        raise NormalizationError("%s directory path changed type during the transaction" % label)
    if (status.st_dev, status.st_ino) != _directory_fd_identity(descriptor):
        raise NormalizationError("%s directory identity changed during the transaction" % label)


def _verify_name_matches_directory_fd(
    parent_descriptor: int, name: str, descriptor: int, label: str
) -> None:
    try:
        status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        raise NormalizationError("%s directory name disappeared during the transaction" % label) from None
    if status.st_mode & 0o170000 != 0o040000:
        raise NormalizationError("%s directory name changed type during the transaction" % label)
    if (status.st_dev, status.st_ino) != _directory_fd_identity(descriptor):
        raise NormalizationError("%s directory identity changed during the transaction" % label)


def _directory_fd_is_within(child_descriptor: int, ancestor_descriptor: int) -> bool:
    ancestor_identity = _directory_fd_identity(ancestor_descriptor)
    current = os.dup(child_descriptor)
    try:
        for _ in range(4096):
            current_identity = _directory_fd_identity(current)
            if current_identity == ancestor_identity:
                return True
            try:
                parent = os.open("..", _directory_flags(), dir_fd=current)
            except OSError:
                raise NormalizationError("directory ancestry could not be anchored") from None
            parent_identity = _directory_fd_identity(parent)
            os.close(current)
            current = parent
            if parent_identity == current_identity:
                return False
    finally:
        os.close(current)
    raise NormalizationError("directory ancestry exceeds the fail-closed traversal bound")


def _new_transaction_directory(parent_descriptor: int) -> Tuple[str, int]:
    for _ in range(128):
        name = ".rgate-normalize-" + os.urandom(16).hex()
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
        except OSError:
            raise NormalizationError("new transaction directory could not be anchored") from None
        _verify_name_matches_directory_fd(parent_descriptor, name, descriptor, "transaction")
        return name, descriptor
    raise NormalizationError("could not allocate a unique transaction directory")


def _open_new_directory_at(parent_descriptor: int, name: str) -> int:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise NormalizationError("unsafe internal staging component")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
    except OSError:
        raise NormalizationError("internal staging directory creation failed") from None
    _verify_name_matches_directory_fd(parent_descriptor, name, descriptor, "internal staging")
    os.fsync(parent_descriptor)
    return descriptor


def _write_exclusive_at(root_descriptor: int, relative: str, payload: bytes) -> None:
    path = PurePosixPath(relative)
    if not _safe_relative(path.as_posix()) or len(path.parts) < 2:
        raise NormalizationError("unsafe internal staging path")
    current = os.dup(root_descriptor)
    try:
        for part in path.parts[:-1]:
            child = _open_new_directory_at(current, part)
            os.close(current)
            current = child
        file_name = path.parts[-1]
        if not file_name or file_name in (".", "..") or "/" in file_name or "\\" in file_name:
            raise NormalizationError("unsafe internal staging filename")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(file_name, flags, 0o600, dir_fd=current)
        except OSError:
            raise NormalizationError("internal staging file creation failed") from None
        try:
            status = os.fstat(descriptor)
            if status.st_mode & 0o170000 != 0o100000:
                raise NormalizationError("internal staging output is not a regular file")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise NormalizationError("internal staging write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(current)
    finally:
        os.close(current)


def _read_expected_file_at(parent_descriptor: int, name: str, expected: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        raise NormalizationError("staged output could not be reopened") from None
    try:
        before = os.fstat(descriptor)
        if before.st_mode & 0o170000 != 0o100000 or before.st_size != len(expected):
            raise NormalizationError("staged output identity changed before publication")
        chunks = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        observed = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity or observed != expected:
        raise NormalizationError("staged output bytes changed before publication")


def _verify_transaction_tree(
    root_descriptor: int, expected_payloads: Dict[str, bytes], prefix: str = ""
) -> None:
    allowed_directories = set()
    for relative in expected_payloads:
        parts = PurePosixPath(relative).parts
        for depth in range(1, len(parts)):
            allowed_directories.add("/".join(parts[:depth]))

    seen = set()

    def walk(descriptor: int, current_prefix: str) -> None:
        try:
            names = os.listdir(descriptor)
        except OSError:
            raise NormalizationError("transaction tree could not be enumerated") from None
        for name in names:
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise NormalizationError("transaction tree contains an unsafe name")
            relative = current_prefix + "/" + name if current_prefix else name
            try:
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                raise NormalizationError("transaction tree changed during enumeration") from None
            kind = status.st_mode & 0o170000
            if kind == 0o040000:
                if relative not in allowed_directories:
                    raise NormalizationError("transaction tree contains an unexpected directory")
                try:
                    child = os.open(name, _directory_flags(), dir_fd=descriptor)
                except OSError:
                    raise NormalizationError("transaction directory could not be anchored") from None
                try:
                    if (status.st_dev, status.st_ino) != _directory_fd_identity(child):
                        raise NormalizationError("transaction directory changed during enumeration")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif kind == 0o100000:
                if relative not in expected_payloads:
                    raise NormalizationError("transaction tree contains an unexpected file")
                _read_expected_file_at(descriptor, name, expected_payloads[relative])
                seen.add(relative)
            else:
                raise NormalizationError("transaction tree contains a non-regular object")

    walk(root_descriptor, prefix)
    if seen != set(expected_payloads):
        raise NormalizationError("transaction tree is incomplete")


def _rename_noreplace_at(parent_descriptor: int, source_name: str, destination_name: str) -> None:
    """Atomically publish within an anchored parent without replacement."""

    if any(
        not name or name in (".", "..") or "/" in name or "\\" in name
        for name in (source_name, destination_name)
    ):
        raise NormalizationError("unsafe atomic publication name")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        raise NormalizationError("atomic no-replace directory publication is unavailable") from None
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise NormalizationError("staging destination appeared during the transaction")
    raise NormalizationError("atomic no-replace directory publication failed")


def _atomic_stage(
    staging_path: Path,
    relative: str,
    public_payload: bytes,
    receipt_name: str,
    receipt_payload: bytes,
    parent_descriptor: Optional[int] = None,
    candidate_root_descriptor: Optional[int] = None,
) -> None:
    parent = staging_path.parent
    destination_name = staging_path.name
    if parent_descriptor is None:
        try:
            owned_parent_descriptor = os.open(str(parent), _directory_flags())
        except OSError:
            raise NormalizationError("staging parent could not be anchored") from None
    else:
        owned_parent_descriptor = os.dup(parent_descriptor)
    transaction_descriptor = None
    try:
        _verify_path_matches_directory_fd(parent, owned_parent_descriptor, "staging parent")
        if candidate_root_descriptor is not None and _directory_fd_is_within(
            owned_parent_descriptor, candidate_root_descriptor
        ):
            raise NormalizationError("staging parent moved inside the candidate")
        try:
            os.stat(destination_name, dir_fd=owned_parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise NormalizationError("staging destination must not already exist")
        transaction_name, transaction_descriptor = _new_transaction_directory(owned_parent_descriptor)
        receipt_relative = "normalization_receipts/" + receipt_name
        expected_payloads = {
            relative: public_payload,
            receipt_relative: receipt_payload,
        }
        _write_exclusive_at(transaction_descriptor, relative, public_payload)
        _write_exclusive_at(transaction_descriptor, receipt_relative, receipt_payload)
        os.fsync(transaction_descriptor)
        _verify_transaction_tree(transaction_descriptor, expected_payloads)
        _verify_name_matches_directory_fd(
            owned_parent_descriptor, transaction_name, transaction_descriptor, "transaction"
        )
        _verify_path_matches_directory_fd(parent, owned_parent_descriptor, "staging parent")
        if candidate_root_descriptor is not None and _directory_fd_is_within(
            owned_parent_descriptor, candidate_root_descriptor
        ):
            raise NormalizationError("staging parent moved inside the candidate")
        _rename_noreplace_at(owned_parent_descriptor, transaction_name, destination_name)
        _verify_name_matches_directory_fd(
            owned_parent_descriptor, destination_name, transaction_descriptor, "published stage"
        )
        _verify_transaction_tree(transaction_descriptor, expected_payloads)
        _verify_path_matches_directory_fd(parent, owned_parent_descriptor, "staging parent")
        if candidate_root_descriptor is not None and _directory_fd_is_within(
            owned_parent_descriptor, candidate_root_descriptor
        ):
            raise NormalizationError("staging parent moved inside the candidate")
        try:
            os.fsync(owned_parent_descriptor)
        except OSError:
            # The complete directory remains the atomic visibility boundary.
            pass
    finally:
        if transaction_descriptor is not None:
            os.close(transaction_descriptor)
        os.close(owned_parent_descriptor)


def _normalize_artifact_with_held_parent(
    *,
    manifest_path: Path,
    artifact_id: str,
    source_path: Path,
    staging_dir: Path,
    bundle_root: Optional[Path] = None,
    held_parent_descriptor: int,
    held_root_descriptor: int,
) -> Dict[str, Any]:
    """Run normalization while the externally checked stage parent is held."""

    root = (bundle_root or manifest_path.parent).resolve(strict=True)
    _reject_symlink_components(manifest_path, include_leaf=True, label="manifest")
    _reject_symlink_components(source_path, include_leaf=True, label="source")
    _reject_symlink_components(staging_dir.parent, include_leaf=True, label="staging")
    manifest_real = manifest_path.resolve(strict=True)
    if manifest_real.parent != root:
        raise NormalizationError("artifact manifest must be at the candidate root")
    if staging_dir.name in ("", ".", ".."):
        raise NormalizationError("invalid staging destination")
    try:
        staging_parent = staging_dir.parent.resolve(strict=True)
    except OSError:
        raise NormalizationError("staging parent must already exist") from None
    staging_real = staging_parent / staging_dir.name
    source_parent = source_path.parent.resolve(strict=True)
    source_real = source_parent / source_path.name
    if _is_within(staging_real, root):
        raise NormalizationError("staging destination must be outside the candidate")
    if _is_within(source_real, root):
        raise NormalizationError("private source must be outside the candidate")
    if os.path.lexists(str(staging_real)):
        raise NormalizationError("staging destination must not already exist")

    implementation_identities = _bound_implementation_identities()
    manifest, item, kind, pointers = _load_manifest_registration(manifest_real, artifact_id)
    source_payload = _read_regular_authenticated(
        source_real,
        int(item["size_bytes"]),
        str(item["sha256"]),
    )
    source = load_json_bytes(source_payload, "private artifact")
    if not isinstance(source, dict):
        raise NormalizationError("private artifact root must be an object")
    public = copy.deepcopy(source)
    replacements = _replacement_values(manifest, str(item["relative_path"]), kind)
    if list(replacements) != pointers:
        raise NormalizationError("replacement order differs from the audited pointer contract")
    for pointer in pointers:
        _get_pointer(source, pointer)
        _set_pointer(public, pointer, replacements[pointer])
    differences = _json_differences(source, public)
    if differences != pointers:
        raise NormalizationError("source/public differences exceed the audited pointer contract")
    _verify_public_strings(public)

    source_parameter_sha = _inference_payload_sha(source, pointers)
    public_parameter_sha = _inference_payload_sha(public, pointers)
    if source_parameter_sha != public_parameter_sha:
        raise NormalizationError("source/public canonical inference payloads differ")
    fixture = _equivalence_proof(kind, source, public)
    if _bound_implementation_identities() != implementation_identities:
        raise NormalizationError("implementation identity changed during the equivalence proof")
    public_payload = canonical_json_bytes(public)
    public_identity = {
        "size_bytes": len(public_payload),
        "sha256": sha256_bytes(public_payload),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": artifact_id,
        "intended_public_relative_path": item["relative_path"],
        "canonicalization": CANONICALIZATION,
        "source_artifact_identity": {
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        },
        "public_artifact_identity": public_identity,
        "normalization": {
            "schema_version": CONTRACT_SCHEMA,
            "changed_json_pointers": pointers,
            "source_inference_parameter_sha256": source_parameter_sha,
            "public_inference_parameter_sha256": public_parameter_sha,
        },
        "fixture_equivalence": fixture,
        "implementation_identities": implementation_identities,
    }
    receipt_payload = canonical_json_bytes(receipt)
    receipt_identity = {
        "size_bytes": len(receipt_payload),
        "sha256": sha256_bytes(receipt_payload),
    }
    receipt_name = artifact_id + ".normalization_receipt.json"
    if _bound_implementation_identities() != implementation_identities:
        raise NormalizationError("implementation identity changed before staging")
    _atomic_stage(
        staging_real,
        str(item["relative_path"]),
        public_payload,
        receipt_name,
        receipt_payload,
        parent_descriptor=held_parent_descriptor,
        candidate_root_descriptor=held_root_descriptor,
    )
    return {
        "artifact_id": artifact_id,
        "public_artifact_identity": public_identity,
        "normalization_receipt_identity": receipt_identity,
        "changed_json_pointers": pointers,
        "source_inference_parameter_sha256": source_parameter_sha,
        "public_inference_parameter_sha256": public_parameter_sha,
        "fixture_sha256": fixture["fixture_sha256"],
    }


def normalize_artifact(
    *,
    manifest_path: Path,
    artifact_id: str,
    source_path: Path,
    staging_dir: Path,
    bundle_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create one complete external stage, anchored before the boundary check."""

    root = (bundle_root or manifest_path.parent).resolve(strict=True)
    _reject_symlink_components(manifest_path, include_leaf=True, label="manifest")
    _reject_symlink_components(source_path, include_leaf=True, label="source")
    _reject_symlink_components(staging_dir.parent, include_leaf=True, label="staging")
    try:
        staging_parent = staging_dir.parent.resolve(strict=True)
        root_descriptor = os.open(str(root), _directory_flags())
    except OSError:
        raise NormalizationError("candidate or staging parent could not be anchored") from None
    parent_descriptor = None
    try:
        try:
            parent_descriptor = os.open(str(staging_parent), _directory_flags())
        except OSError:
            raise NormalizationError("staging parent could not be anchored") from None
        _verify_path_matches_directory_fd(root, root_descriptor, "candidate root")
        _verify_path_matches_directory_fd(staging_parent, parent_descriptor, "staging parent")
        if _directory_fd_is_within(parent_descriptor, root_descriptor):
            raise NormalizationError("staging destination must be outside the candidate")
        return _normalize_artifact_with_held_parent(
            manifest_path=manifest_path,
            artifact_id=artifact_id,
            source_path=source_path,
            staging_dir=staging_dir,
            bundle_root=bundle_root,
            held_parent_descriptor=parent_descriptor,
            held_root_descriptor=root_descriptor,
        )
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        os.close(root_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "ARTIFACT_MANIFEST.json",
        help="v2 registry at the candidate root",
    )
    args = parser.parse_args()
    try:
        result = normalize_artifact(
            manifest_path=args.manifest,
            artifact_id=args.artifact_id,
            source_path=args.source_json,
            staging_dir=args.staging_dir,
            bundle_root=args.manifest.parent,
        )
    except (NormalizationError, OSError) as exc:
        print("FAIL: %s" % exc, file=sys.stderr)
        return 1
    print(canonical_json_bytes({"status": "pass", **result}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
