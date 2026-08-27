#!/usr/bin/env python3
"""Assemble twenty audited normalization stages into a public companion clone.

The command is intentionally all-or-nothing.  It authenticates the source
candidate, validates the complete one-stage-per-artifact set, builds every
output byte in a private sibling transaction directory, and publishes with
Linux renameat2(RENAME_NOREPLACE).  It never edits the source candidate or a
stage and it never replaces an existing output path.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


MANIFEST_SCHEMA = "rgate_release_artifact_manifest_v2"
PRIVATE_SOURCE_RELEASE_STATUS = "prepublication_candidate"
PRIVATE_SOURCE_LICENSE_STATUS = "not_selected"
PRIVATE_SOURCE_IDENTITY_STATUS = "anonymous_candidate"
PUBLIC_RELEASE_STATUS = "public_research_code_companion"
PUBLIC_LICENSE_STATUS = "no_project_license_granted"
PUBLIC_IDENTITY_STATUS = "public_identity_disclosed"
CONTRACT_SCHEMA = "rgate_release_metadata_normalization_v1"
RECEIPT_SCHEMA = "rgate_release_normalization_receipt_v1"
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
EXPECTED_IMPLEMENTATIONS = {
    "inference": "scripts/apply_bge_af_arbiter.py",
    "normalizer": "tools/normalize_learned_artifact.py",
}
MODEL_FIXTURE_SCHEMA = "rgate_model_equivalence_fixture_v1"
CALIBRATION_FIXTURE_SCHEMA = "rgate_calibration_equivalence_fixture_v1"
CACHE_PUBLIC_VALUE = "artifacts/cache/fullval_cache.jsonl"
PUBLIC_ENTRY_KEYS = {
    "id",
    "relative_path",
    "availability",
    "public_artifact_identity",
    "normalization",
    "normalization_receipt",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_ID_RE = re.compile(
    r"seed_(?:00|01|02|03|04)_(?:no_radar_|radar_)(?:model|calibration)\Z"
)
MODEL_PATH_RE = re.compile(
    r"models/seed_(?:00|01|02|03|04)/(?:no_radar_|radar_)(?:model|calibration)\.json\Z"
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
RUNTIME_TOP_LEVEL = {".git", ".venv", "data", "outputs"}
RUNTIME_PREFIXES = {
    ("artifacts", "cache"),
    ("artifacts", "expert_results"),
    ("artifacts", "metadata"),
}
TRANSIENT_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}


class AssemblyError(ValueError):
    """A fail-closed assembly error safe to display without private data."""


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
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AssemblyError("invalid %s JSON: %s" % (label, exc)) from None


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
        raise AssemblyError("value is not canonical finite JSON: %s" % exc) from None


def pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AssemblyError("value is not finite JSON: %s" % exc) from None


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
    absolute = path.absolute()
    components = list(reversed(absolute.parents))
    if include_leaf:
        components.append(absolute)
    for component in components:
        if component == Path(component.anchor):
            continue
        if component.is_symlink():
            raise AssemblyError("%s path contains a symbolic-link component" % label)


def _expected_layout() -> Dict[str, str]:
    result = {}
    for seed in range(5):
        for modality in ("radar", "no_radar"):
            for kind in ("model", "calibration"):
                artifact_id = "seed_%02d_%s_%s" % (seed, modality, kind)
                relative = "models/seed_%02d/%s_%s.json" % (seed, modality, kind)
                result[artifact_id] = relative
    return result


EXPECTED_LAYOUT = _expected_layout()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _verify_public_strings(
    value: Any, label: str, allowed_path_tokens: Optional[Set[str]] = None
) -> None:
    allowed = set() if allowed_path_tokens is None else allowed_path_tokens
    for candidate in _walk_strings(value):
        if candidate in allowed:
            continue
        lowered = candidate.lower()
        if (
            candidate.startswith("/")
            or "\\" in candidate
            or POSIX_ABSOLUTE_RE.search(candidate)
            or TILDE_PATH_RE.search(candidate)
            or WINDOWS_ABSOLUTE_RE.search(candidate)
            or "file://" in lowered
            or any(root.lower() in lowered for root in ENVIRONMENT_ROOTS)
            or EMAIL_RE.search(candidate)
        ):
            raise AssemblyError("%s contains a sensitive absolute-path or identity string" % label)


def _pointer_parts(pointer: str) -> List[str]:
    if not pointer.startswith("/") or pointer == "/":
        raise AssemblyError("invalid JSON pointer in normalization contract")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _get_pointer(value: Any, pointer: str) -> Any:
    current = value
    for part in _pointer_parts(pointer):
        if not isinstance(current, dict) or part not in current:
            raise AssemblyError("public artifact lacks normalization pointer: %s" % pointer)
        current = current[part]
    return current


def _delete_pointer(value: Any, pointer: str) -> None:
    parts = _pointer_parts(pointer)
    current = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise AssemblyError("public artifact lacks normalization pointer: %s" % pointer)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise AssemblyError("public artifact lacks normalization pointer: %s" % pointer)
    del current[parts[-1]]


def _parameter_sha(value: Dict[str, Any], pointers: List[str]) -> str:
    payload = copy.deepcopy(value)
    for pointer in pointers:
        _delete_pointer(payload, pointer)
    return sha256_bytes(canonical_json_bytes(payload))


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        raise AssemblyError("%s is unavailable or not a direct regular file" % label) from None
    try:
        before = os.fstat(descriptor)
        if before.st_mode & 0o170000 != 0o100000:
            raise AssemblyError("%s is not a regular file" % label)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise AssemblyError("%s changed while it was read" % label)
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise AssemblyError("%s size changed while it was read" % label)
    return payload


def _runtime_excluded(parts: Tuple[str, ...]) -> bool:
    return bool(
        (parts and parts[0] in RUNTIME_TOP_LEVEL)
        or tuple(parts[:2]) in RUNTIME_PREFIXES
        or any(part in TRANSIENT_DIR_NAMES for part in parts)
    )


def _snapshot_source_candidate(root: Path) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    for current_text, directory_names, file_names in os.walk(str(root), followlinks=False):
        current = Path(current_text)
        current_relative = current.relative_to(root)
        kept_directories = []
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                raise AssemblyError("source candidate contains a symbolic link")
            parts = tuple((current_relative / name).parts)
            if not _runtime_excluded(parts):
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            parts = PurePosixPath(relative).parts
            if _runtime_excluded(tuple(parts)):
                continue
            if path.is_symlink():
                raise AssemblyError("source candidate contains a symbolic link")
            if not _safe_relative(relative):
                raise AssemblyError("source candidate contains an unsafe path")
            files[relative] = _read_regular(path, "source candidate file")
    checksum_payload = files.get("SHA256SUMS")
    if checksum_payload is None:
        raise AssemblyError("source candidate lacks SHA256SUMS")
    expected: Dict[str, str] = {}
    try:
        lines = checksum_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise AssemblyError("source candidate SHA256SUMS is not UTF-8") from None
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise AssemblyError("source candidate SHA256SUMS has an invalid line")
        digest, relative = match.groups()
        if not _safe_relative(relative) or relative in expected or relative == "SHA256SUMS":
            raise AssemblyError("source candidate SHA256SUMS has an unsafe or duplicate path")
        expected[relative] = digest
    actual = set(files) - {"SHA256SUMS"}
    if set(expected) != actual:
        raise AssemblyError("source candidate SHA256SUMS file set is incomplete")
    for relative, digest in expected.items():
        if sha256_bytes(files[relative]) != digest:
            raise AssemblyError("source candidate SHA256SUMS authentication failed")
    return files


def _validate_source_manifest(files: Dict[str, bytes]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    payload = files.get("ARTIFACT_MANIFEST.json")
    if payload is None:
        raise AssemblyError("source candidate lacks ARTIFACT_MANIFEST.json")
    manifest = load_json_bytes(payload, "source artifact manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise AssemblyError("unexpected source artifact manifest schema")
    if manifest.get("release_status") != PRIVATE_SOURCE_RELEASE_STATUS:
        raise AssemblyError("source candidate release state is not prepublication")
    if manifest.get("license_status") != PRIVATE_SOURCE_LICENSE_STATUS:
        raise AssemblyError("source candidate license state is not prepublication")
    if manifest.get("double_blind_status") != PRIVATE_SOURCE_IDENTITY_STATUS:
        raise AssemblyError("source candidate identity state is not anonymous")
    contract = manifest.get("normalization_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != CONTRACT_SCHEMA:
        raise AssemblyError("unexpected normalization contract schema")
    if contract.get("allowed_changed_json_pointers") != EXPECTED_POINTERS:
        raise AssemblyError("normalization pointer allowlist differs from the audited contract")
    if contract.get("required_public_proofs") != EXPECTED_PUBLIC_PROOFS:
        raise AssemblyError("normalization proof list differs from the audited contract")
    if manifest.get("public_artifacts") != []:
        raise AssemblyError("source candidate already contains public artifact registrations")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 20:
        raise AssemblyError("source manifest must contain exactly twenty private registrations")
    by_id: Dict[str, Dict[str, Any]] = {}
    paths: Set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise AssemblyError("private artifact registration is not an object")
        artifact_id = item.get("id")
        relative = item.get("relative_path")
        if (
            not isinstance(artifact_id, str)
            or ARTIFACT_ID_RE.fullmatch(artifact_id) is None
            or artifact_id in by_id
            or artifact_id not in EXPECTED_LAYOUT
            or relative != EXPECTED_LAYOUT[artifact_id]
            or not isinstance(relative, str)
            or MODEL_PATH_RE.fullmatch(relative) is None
            or relative in paths
        ):
            raise AssemblyError("private artifact id/path layout is invalid or duplicate")
        if item.get("required") is not True or item.get("availability") != "missing_pending_verified_copy":
            raise AssemblyError("private artifact source-provenance state is not eligible")
        if (
            not isinstance(item.get("size_bytes"), int)
            or isinstance(item.get("size_bytes"), bool)
            or item["size_bytes"] <= 0
            or not _valid_sha256(item.get("sha256"))
        ):
            raise AssemblyError("private artifact source identity is invalid")
        if relative in files:
            raise AssemblyError("source candidate already contains a model at a public path")
        by_id[artifact_id] = item
        paths.add(relative)
    if set(by_id) != set(EXPECTED_LAYOUT):
        raise AssemblyError("private artifact registrations do not cover the exact twenty-item layout")
    if any(relative.startswith("normalization_receipts/") for relative in files):
        raise AssemblyError("source candidate already contains normalization receipts")
    return manifest, by_id


def _stage_snapshot(stage: Path, expected_files: Set[str]) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    directories: Set[str] = set()
    for current_text, directory_names, file_names in os.walk(str(stage), followlinks=False):
        current = Path(current_text)
        current_relative = current.relative_to(stage)
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                raise AssemblyError("normalization stage contains a symbolic link")
            relative = (current_relative / name).as_posix()
            if not _safe_relative(relative):
                raise AssemblyError("normalization stage contains an unsafe directory")
            directories.add(relative)
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(stage).as_posix()
            if path.is_symlink():
                raise AssemblyError("normalization stage contains a symbolic link")
            if not _safe_relative(relative) or relative in files:
                raise AssemblyError("normalization stage contains an unsafe or duplicate file")
            files[relative] = _read_regular(path, "normalization stage file")
    expected_directories: Set[str] = set()
    for relative in expected_files:
        parts = PurePosixPath(relative).parts
        for depth in range(1, len(parts)):
            expected_directories.add(PurePosixPath(*parts[:depth]).as_posix())
    if set(files) != expected_files or directories != expected_directories:
        raise AssemblyError("normalization stage tree is incomplete or contains extra entries")
    return files


def _required_object_keys(value: Any, keys: Set[str], label: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AssemblyError("%s fields differ from the audited schema" % label)
    return value


def _candidate_implementation_identities(source_files: Dict[str, bytes]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for key, relative in EXPECTED_IMPLEMENTATIONS.items():
        payload = source_files.get(relative)
        if payload is None:
            raise AssemblyError("source candidate lacks a receipt-bound implementation")
        result[key] = {
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    return result


def _expected_pointer_values(relative: str, kind: str) -> Dict[str, str]:
    if kind == "model":
        return {"/training/cache_jsonl": CACHE_PUBLIC_VALUE}
    model_relative = relative[: -len("_calibration.json")] + "_model.json"
    return {"/cache_jsonl": CACHE_PUBLIC_VALUE, "/model": model_relative}


def _validate_stage(
    *,
    artifact_id: str,
    registration: Dict[str, Any],
    stage: Path,
    implementation_identities: Dict[str, Dict[str, Any]],
) -> Tuple[bytes, str, bytes, str, Dict[str, Any]]:
    relative = str(registration["relative_path"])
    receipt_relative = "normalization_receipts/%s.normalization_receipt.json" % artifact_id
    stage_files = _stage_snapshot(stage, {relative, receipt_relative})
    public_payload = stage_files[relative]
    receipt_payload = stage_files[receipt_relative]
    public = load_json_bytes(public_payload, "public learned artifact")
    receipt = load_json_bytes(receipt_payload, "normalization receipt")
    if not isinstance(public, dict) or canonical_json_bytes(public) != public_payload:
        raise AssemblyError("public learned artifact is not a canonical JSON object")
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != receipt_payload:
        raise AssemblyError("normalization receipt is not a canonical JSON object")
    _verify_public_strings(public, "public learned artifact")
    _verify_public_strings(
        receipt,
        "normalization receipt",
        allowed_path_tokens={pointer for values in EXPECTED_POINTERS.values() for pointer in values},
    )

    kind = "calibration" if relative.endswith("_calibration.json") else "model"
    pointers = list(EXPECTED_POINTERS[kind])
    for pointer, expected_value in _expected_pointer_values(relative, kind).items():
        if _get_pointer(public, pointer) != expected_value:
            raise AssemblyError("public normalization pointer has an unexpected value")
    public_identity = {"size_bytes": len(public_payload), "sha256": sha256_bytes(public_payload)}

    receipt_keys = {
        "schema_version",
        "artifact_id",
        "intended_public_relative_path",
        "canonicalization",
        "source_artifact_identity",
        "public_artifact_identity",
        "normalization",
        "fixture_equivalence",
        "implementation_identities",
    }
    _required_object_keys(receipt, receipt_keys, "normalization receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise AssemblyError("normalization receipt schema is unexpected")
    if receipt.get("artifact_id") != artifact_id:
        raise AssemblyError("normalization receipt artifact id does not match its stage")
    if receipt.get("intended_public_relative_path") != relative:
        raise AssemblyError("normalization receipt public path does not match the registry")
    if receipt.get("canonicalization") != CANONICALIZATION:
        raise AssemblyError("normalization receipt canonicalization is unexpected")
    source_identity = _required_object_keys(
        receipt.get("source_artifact_identity"), {"size_bytes", "sha256"}, "source identity"
    )
    expected_source_identity = {
        "size_bytes": registration["size_bytes"],
        "sha256": registration["sha256"],
    }
    if source_identity != expected_source_identity:
        raise AssemblyError("normalization receipt source identity does not match the private registry")
    observed_public_identity = _required_object_keys(
        receipt.get("public_artifact_identity"), {"size_bytes", "sha256"}, "public identity"
    )
    if observed_public_identity != public_identity:
        raise AssemblyError("normalization receipt public identity does not match the public JSON")

    normalization = _required_object_keys(
        receipt.get("normalization"),
        {
            "schema_version",
            "changed_json_pointers",
            "source_inference_parameter_sha256",
            "public_inference_parameter_sha256",
        },
        "normalization proof",
    )
    if normalization.get("schema_version") != CONTRACT_SCHEMA:
        raise AssemblyError("normalization proof schema is unexpected")
    if normalization.get("changed_json_pointers") != pointers:
        raise AssemblyError("normalization changed-pointer set is incomplete or reordered")
    source_parameter_sha = normalization.get("source_inference_parameter_sha256")
    public_parameter_sha = normalization.get("public_inference_parameter_sha256")
    if not _valid_sha256(source_parameter_sha) or not _valid_sha256(public_parameter_sha):
        raise AssemblyError("normalization parameter identity is invalid")
    if source_parameter_sha != public_parameter_sha:
        raise AssemblyError("source/public canonical inference parameters differ")
    if _parameter_sha(public, pointers) != public_parameter_sha:
        raise AssemblyError("public JSON inference parameters do not match the receipt")

    fixture = _required_object_keys(
        receipt.get("fixture_equivalence"),
        {
            "schema_version",
            "method",
            "comparison",
            "probe_count",
            "fixture_sha256",
            "source_output_sha256",
            "public_output_sha256",
        },
        "fixture equivalence proof",
    )
    expected_fixture_schema = MODEL_FIXTURE_SCHEMA if kind == "model" else CALIBRATION_FIXTURE_SCHEMA
    expected_method = (
        "apply_bge_af_arbiter.model_score_exact_float_hex_v1"
        if kind == "model"
        else "apply_bge_af_arbiter.calibration_path_exact_float_hex_v1"
    )
    if fixture.get("schema_version") != expected_fixture_schema or fixture.get("method") != expected_method:
        raise AssemblyError("fixture equivalence method or schema is unexpected")
    if fixture.get("comparison") != "exact_canonical_float_hex":
        raise AssemblyError("fixture equivalence comparison is unexpected")
    probe_count = fixture.get("probe_count")
    if (
        not isinstance(probe_count, int)
        or isinstance(probe_count, bool)
        or (kind == "model" and probe_count != 4)
        or (kind == "calibration" and (probe_count < 5 or probe_count % 5 != 0))
    ):
        raise AssemblyError("fixture equivalence probe count is invalid")
    fixture_sha = fixture.get("fixture_sha256")
    source_output_sha = fixture.get("source_output_sha256")
    public_output_sha = fixture.get("public_output_sha256")
    if not all(_valid_sha256(value) for value in (fixture_sha, source_output_sha, public_output_sha)):
        raise AssemblyError("fixture equivalence hashes are invalid")
    if source_output_sha != public_output_sha:
        raise AssemblyError("source/public fixed-fixture outputs differ")
    if receipt.get("implementation_identities") != implementation_identities:
        raise AssemblyError("receipt-bound implementation identities do not match the source candidate")

    receipt_identity = {"size_bytes": len(receipt_payload), "sha256": sha256_bytes(receipt_payload)}
    public_entry = {
        "id": artifact_id,
        "relative_path": relative,
        "availability": "present_verified",
        "public_artifact_identity": public_identity,
        "normalization": {
            "schema_version": CONTRACT_SCHEMA,
            "changed_json_pointers": pointers,
            "source_inference_parameter_sha256": source_parameter_sha,
            "public_inference_parameter_sha256": public_parameter_sha,
            "fixture_equivalence_receipt_sha256": receipt_identity["sha256"],
        },
        "normalization_receipt": {
            "relative_path": receipt_relative,
            "size_bytes": receipt_identity["size_bytes"],
            "sha256": receipt_identity["sha256"],
        },
    }
    if set(public_entry) != PUBLIC_ENTRY_KEYS:
        raise AssertionError("internal public registration schema mismatch")
    return public_payload, relative, receipt_payload, receipt_relative, public_entry


def _read_stages(
    stages_root: Path,
    registrations: Dict[str, Dict[str, Any]],
    implementation_identities: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, bytes], List[Dict[str, Any]]]:
    observed: Dict[str, Path] = {}
    try:
        entries = list(os.scandir(str(stages_root)))
    except OSError:
        raise AssemblyError("stages root is unavailable") from None
    for entry in entries:
        if entry.is_symlink():
            raise AssemblyError("stages root contains a symbolic link")
        if not entry.is_dir(follow_symlinks=False):
            raise AssemblyError("stages root contains a non-directory entry")
        if entry.name in observed:
            raise AssemblyError("stages root contains duplicate stage names")
        observed[entry.name] = Path(entry.path)
    if set(observed) != set(registrations):
        missing = sorted(set(registrations) - set(observed))
        extra = sorted(set(observed) - set(registrations))
        raise AssemblyError("stages root must contain exactly twenty registered stages: missing=%r extra=%r" % (missing, extra))
    additions: Dict[str, bytes] = {}
    public_entries = []
    seen_public_paths: Set[str] = set()
    seen_receipt_paths: Set[str] = set()
    for artifact_id in sorted(registrations):
        public_payload, public_relative, receipt_payload, receipt_relative, public_entry = _validate_stage(
            artifact_id=artifact_id,
            registration=registrations[artifact_id],
            stage=observed[artifact_id],
            implementation_identities=implementation_identities,
        )
        if public_relative in seen_public_paths or receipt_relative in seen_receipt_paths:
            raise AssemblyError("stages resolve to duplicate public or receipt paths")
        seen_public_paths.add(public_relative)
        seen_receipt_paths.add(receipt_relative)
        additions[public_relative] = public_payload
        additions[receipt_relative] = receipt_payload
        public_entries.append(public_entry)
    if len(public_entries) != 20 or len(additions) != 40:
        raise AssemblyError("validated stage set does not yield exactly twenty artifacts and receipts")
    return additions, public_entries


def _models_readme(
    registrations: Dict[str, Dict[str, Any]], public_entries: List[Dict[str, Any]]
) -> bytes:
    public_by_id = {item["id"]: item for item in public_entries}
    lines = [
        "# Learned artifacts",
        "",
        "This public research-code companion contains all twenty public-normalized learned JSON",
        "artifacts and their canonical normalization receipts. The original private",
        "scientific-source bytes are not included. Their immutable provenance",
        "identities remain in `ARTIFACT_MANIFEST.json` with",
        "`availability: missing_pending_verified_copy`; that field describes the",
        "non-public source, not the normalized file beside this document.",
        "",
        "Each `public_artifacts` registration records the normalized byte identity,",
        "the exact allowed metadata pointers, equal source/public canonical",
        "inference-parameter hashes, and the SHA-256 of its receipt. The receipt files",
        "are under `normalization_receipts/` and are covered by `SHA256SUMS`.",
        "",
        "| Public file | Public bytes | Public SHA-256 | Scientific-source SHA-256 | State |",
        "|---|---:|---|---|---|",
    ]
    for artifact_id in sorted(registrations):
        registration = registrations[artifact_id]
        public = public_by_id[artifact_id]
        identity = public["public_artifact_identity"]
        relative = str(registration["relative_path"])
        model_relative = relative[len("models/") :] if relative.startswith("models/") else relative
        lines.append(
            "| `%s` | %d | `%s` | `%s` | present, normalized, verified |"
            % (model_relative, identity["size_bytes"], identity["sha256"], registration["sha256"])
        )
    lines.extend(
        [
            "",
            "Run `python3 -B verify_bundle.py` from the candidate root. Missing, extra,",
            "misregistered, non-canonical, path-bearing, or hash-inconsistent artifacts",
            "and receipts remain hard failures.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _release_status() -> bytes:
    return (
        "# Release status\n\n"
        "Status: **PUBLIC_RESEARCH_CODE_COMPANION**\n\n"
        "Scope: publicly readable R-GATE scripts, an example configuration template,\n"
        "public-normalized learned artifacts and integrity checks accompanying the\n"
        "dissertation.\n\n"
        "This repository is not a self-contained end-to-end reproducibility package.\n"
        "It does not include nuScenes data or annotations, cached expert predictions,\n"
        "upstream detector source trees or checkpoints, or the original locked run\n"
        "plans and full experiment inputs.\n\n"
        "The twenty learned JSON artifacts are present only in public-normalized form.\n"
        "Their byte identities, allowed metadata transformations and fixed-fixture\n"
        "equivalence receipts are registered in `ARTIFACT_MANIFEST.json` and checked by\n"
        "`python3 -B verify_bundle.py`.\n\n"
        "License status: no project license is granted by this repository. Public\n"
        "readability does not grant rights to reuse, redistribute or commercially use\n"
        "the project materials. The applicable boundaries for third-party software,\n"
        "models and data are recorded in `NOTICE`.\n"
    ).encode("utf-8")


def _release_ready_root_readme(payload: bytes) -> bytes:
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        raise AssemblyError("source README is not UTF-8") from None
    return payload


def _checksum_payload(files: Dict[str, bytes]) -> bytes:
    lines = ["%s  %s" % (sha256_bytes(files[relative]), relative) for relative in sorted(files)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _prepare_output_payloads(
    source_files: Dict[str, bytes],
    manifest: Dict[str, Any],
    registrations: Dict[str, Dict[str, Any]],
    additions: Dict[str, bytes],
    public_entries: List[Dict[str, Any]],
) -> Dict[str, bytes]:
    output = {relative: payload for relative, payload in source_files.items() if relative != "SHA256SUMS"}
    collisions = set(output) & set(additions)
    if collisions:
        raise AssemblyError("normalized stages collide with source candidate files")
    output.update(additions)
    output_manifest = copy.deepcopy(manifest)
    output_manifest["public_artifacts"] = public_entries
    output_manifest["release_status"] = PUBLIC_RELEASE_STATUS
    output_manifest["license_status"] = PUBLIC_LICENSE_STATUS
    output_manifest["double_blind_status"] = PUBLIC_IDENTITY_STATUS
    # The scientific-source registrations and their missing-source state remain
    # unchanged while the assembled directory becomes a public companion.
    if output_manifest.get("artifacts") != manifest.get("artifacts"):
        raise AssertionError("private source registrations changed internally")
    output["ARTIFACT_MANIFEST.json"] = pretty_json_bytes(output_manifest)
    output["models/README.md"] = _models_readme(registrations, public_entries)
    output["RELEASE_STATUS.md"] = _release_status()
    source_readme = source_files.get("README.md")
    if source_readme is None:
        raise AssemblyError("source candidate lacks README.md")
    output["README.md"] = _release_ready_root_readme(source_readme)
    output["SHA256SUMS"] = _checksum_payload(output)
    return output


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _directory_identity(descriptor: int) -> Tuple[int, int]:
    status = os.fstat(descriptor)
    if status.st_mode & 0o170000 != 0o040000:
        raise AssemblyError("anchored output object is not a directory")
    return status.st_dev, status.st_ino


def _verify_name_is_directory(parent_descriptor: int, name: str, descriptor: int, label: str) -> None:
    try:
        status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        raise AssemblyError("%s disappeared during assembly" % label) from None
    if status.st_mode & 0o170000 != 0o040000 or (status.st_dev, status.st_ino) != _directory_identity(descriptor):
        raise AssemblyError("%s identity changed during assembly" % label)


def _open_or_create_directory_at(parent_descriptor: int, name: str) -> int:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise AssemblyError("unsafe output directory component")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    except OSError:
        raise AssemblyError("output directory creation failed") from None
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
    except OSError:
        raise AssemblyError("output directory could not be anchored") from None
    _verify_name_is_directory(parent_descriptor, name, descriptor, "output directory")
    return descriptor


def _write_exclusive_at(root_descriptor: int, relative: str, payload: bytes) -> None:
    if not _safe_relative(relative):
        raise AssemblyError("unsafe output file path")
    parts = PurePosixPath(relative).parts
    current = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            child = _open_or_create_directory_at(current, part)
            os.close(current)
            current = child
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(parts[-1], flags, 0o600, dir_fd=current)
        except OSError:
            raise AssemblyError("exclusive output file creation failed") from None
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise AssemblyError("output write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(current)
    finally:
        os.close(current)


def _read_file_at(parent_descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        raise AssemblyError("assembled output file could not be reopened") from None
    try:
        status = os.fstat(descriptor)
        if status.st_mode & 0o170000 != 0o100000:
            raise AssemblyError("assembled output contains a non-regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_tree_at(root_descriptor: int, expected: Dict[str, bytes]) -> None:
    expected_directories: Set[str] = set()
    for relative in expected:
        parts = PurePosixPath(relative).parts
        for depth in range(1, len(parts)):
            expected_directories.add(PurePosixPath(*parts[:depth]).as_posix())
    observed: Set[str] = set()

    def walk(descriptor: int, prefix: str) -> None:
        try:
            names = os.listdir(descriptor)
        except OSError:
            raise AssemblyError("assembled output tree cannot be enumerated") from None
        for name in names:
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise AssemblyError("assembled output tree contains an unsafe name")
            relative = prefix + "/" + name if prefix else name
            try:
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                raise AssemblyError("assembled output tree changed during verification") from None
            kind = status.st_mode & 0o170000
            if kind == 0o040000:
                if relative not in expected_directories:
                    raise AssemblyError("assembled output tree contains an unexpected directory")
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
                try:
                    if (status.st_dev, status.st_ino) != _directory_identity(child):
                        raise AssemblyError("assembled output directory changed during verification")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif kind == 0o100000:
                if relative not in expected:
                    raise AssemblyError("assembled output tree contains an unexpected file")
                if _read_file_at(descriptor, name) != expected[relative]:
                    raise AssemblyError("assembled output bytes differ from the verified payload")
                observed.add(relative)
            else:
                raise AssemblyError("assembled output tree contains a non-regular object")

    walk(root_descriptor, "")
    if observed != set(expected):
        raise AssemblyError("assembled output tree is incomplete")


def _rename_noreplace_at(parent_descriptor: int, source_name: str, destination_name: str) -> None:
    if any(not value or value in (".", "..") or "/" in value or "\\" in value for value in (source_name, destination_name)):
        raise AssemblyError("unsafe atomic publication name")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        raise AssemblyError("atomic no-replace publication is unavailable") from None
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
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
        raise AssemblyError("output candidate appeared during publication")
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
        raise AssemblyError("filesystem does not support atomic no-replace publication")
    raise AssemblyError("atomic no-replace publication failed")


def _publish_payloads(output: Path, payloads: Dict[str, bytes]) -> None:
    parent = output.parent
    name = output.name
    try:
        parent_descriptor = os.open(str(parent), _directory_flags())
    except OSError:
        raise AssemblyError("output parent could not be anchored") from None
    transaction_descriptor: Optional[int] = None
    try:
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AssemblyError("output candidate must not already exist")
        transaction_name = ".rgate-assemble-" + os.urandom(16).hex()
        try:
            os.mkdir(transaction_name, 0o700, dir_fd=parent_descriptor)
            transaction_descriptor = os.open(transaction_name, _directory_flags(), dir_fd=parent_descriptor)
        except OSError:
            raise AssemblyError("assembly transaction directory could not be created") from None
        _verify_name_is_directory(parent_descriptor, transaction_name, transaction_descriptor, "assembly transaction")
        for relative in sorted(payloads):
            _write_exclusive_at(transaction_descriptor, relative, payloads[relative])
        os.fsync(transaction_descriptor)
        _verify_tree_at(transaction_descriptor, payloads)
        _verify_name_is_directory(parent_descriptor, transaction_name, transaction_descriptor, "assembly transaction")
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AssemblyError("output candidate appeared before publication")
        _rename_noreplace_at(parent_descriptor, transaction_name, name)
        _verify_name_is_directory(parent_descriptor, name, transaction_descriptor, "published candidate")
        _verify_tree_at(transaction_descriptor, payloads)
        try:
            os.fsync(parent_descriptor)
        except OSError:
            # Some mounted Windows filesystems do not implement directory fsync.
            # RENAME_NOREPLACE remains the required no-overwrite visibility gate.
            pass
    finally:
        if transaction_descriptor is not None:
            os.close(transaction_descriptor)
        os.close(parent_descriptor)


def assemble_candidate(*, source_candidate: Path, stages_root: Path, output_candidate: Path) -> Dict[str, Any]:
    """Validate and atomically publish a fresh normalized candidate clone."""

    _reject_symlink_components(source_candidate, include_leaf=True, label="source candidate")
    _reject_symlink_components(stages_root, include_leaf=True, label="stages root")
    _reject_symlink_components(output_candidate.parent, include_leaf=True, label="output parent")
    if output_candidate.name in ("", ".", "..") or "/" in output_candidate.name or "\\" in output_candidate.name:
        raise AssemblyError("output candidate name is unsafe")
    try:
        source = source_candidate.resolve(strict=True)
        stages = stages_root.resolve(strict=True)
        output_parent = output_candidate.parent.resolve(strict=True)
    except OSError:
        raise AssemblyError("source candidate, stages root, and output parent must already exist") from None
    if not source.is_dir() or not stages.is_dir() or not output_parent.is_dir():
        raise AssemblyError("source candidate, stages root, and output parent must be directories")
    output = output_parent / output_candidate.name
    if os.path.lexists(str(output)):
        raise AssemblyError("output candidate must not already exist")
    if (
        _is_within(source, stages)
        or _is_within(stages, source)
        or _is_within(output, source)
        or _is_within(output, stages)
        or _is_within(source, output)
        or _is_within(stages, output)
    ):
        raise AssemblyError("source, stages, and output trees must be disjoint")

    source_files = _snapshot_source_candidate(source)
    manifest, registrations = _validate_source_manifest(source_files)
    implementation_identities = _candidate_implementation_identities(source_files)
    additions, public_entries = _read_stages(stages, registrations, implementation_identities)
    payloads = _prepare_output_payloads(
        source_files,
        manifest,
        registrations,
        additions,
        public_entries,
    )
    # Re-authenticate both immutable input snapshots before making any output
    # visible. Any concurrent mutation turns the operation into a hard failure.
    if _snapshot_source_candidate(source) != source_files:
        raise AssemblyError("source candidate changed during assembly")
    second_additions, second_public_entries = _read_stages(stages, registrations, implementation_identities)
    if second_additions != additions or second_public_entries != public_entries:
        raise AssemblyError("normalization stages changed during assembly")
    _publish_payloads(output, payloads)
    return {
        "artifact_count": len(public_entries),
        "receipt_count": len(public_entries),
        "candidate_file_count": len(payloads),
        "sha256sums_sha256": sha256_bytes(payloads["SHA256SUMS"]),
        "release_status": PUBLIC_RELEASE_STATUS,
        "license_status": PUBLIC_LICENSE_STATUS,
        "double_blind_status": PUBLIC_IDENTITY_STATUS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate", required=True, type=Path)
    parser.add_argument("--stages-root", required=True, type=Path)
    parser.add_argument("--output-candidate", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = assemble_candidate(
            source_candidate=args.source_candidate,
            stages_root=args.stages_root,
            output_candidate=args.output_candidate,
        )
    except (AssemblyError, OSError) as exc:
        print("FAIL: %s" % exc, file=sys.stderr)
        return 1
    print(canonical_json_bytes({"status": "pass", **result}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
