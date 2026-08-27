#!/usr/bin/env python3
"""Fail-closed integrity and publication-scope checks for this repository."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent
HASH_FILE = ROOT / "SHA256SUMS"
TEXT_SUFFIXES = {"", ".md", ".txt", ".json", ".yml", ".yaml", ".py", ".gitignore"}
PAPER_SUFFIXES = {".tex", ".bib", ".aux", ".bbl", ".blg", ".sty", ".cls"}
OPTIONAL_EXTERNAL_MODULES = {"lightgbm", "numpy", "nuscenes"}
NORMALIZATION_RECEIPT_SCHEMA = "rgate_release_normalization_receipt_v1"
NORMALIZATION_CANONICALIZATION = "utf8_json_sorted_keys_compact_no_nan_v1"
MODEL_FIXTURE_SCHEMA = "rgate_model_equivalence_fixture_v1"
CALIBRATION_FIXTURE_SCHEMA = "rgate_calibration_equivalence_fixture_v1"
PUBLIC_ARTIFACT_ENTRY_KEYS = {
    "id",
    "relative_path",
    "availability",
    "public_artifact_identity",
    "normalization",
    "normalization_receipt",
}
# Python 3.8 does not expose sys.stdlib_module_names. This frozen set covers
# every standard-library top-level import in the bundled scientific sources;
# newer interpreters additionally contribute their own complete set below.
PYTHON38_BUNDLED_STDLIB_MODULES = {
    "__future__",
    "argparse",
    "ast",
    "collections",
    "contextlib",
    "copy",
    "ctypes",
    "dataclasses",
    "datetime",
    "gzip",
    "hashlib",
    "importlib",
    "json",
    "math",
    "mmap",
    "os",
    "errno",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "statistics",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "types",
    "typing",
}
RUNTIME_TOP_LEVEL = {".git", ".venv", "data", "outputs"}
RUNTIME_PREFIXES = {
    ("artifacts", "cache"),
    ("artifacts", "expert_results"),
    ("artifacts", "metadata"),
}
TRANSIENT_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
PUBLIC_RELEASE_STATUS = "public_research_code_companion"
PUBLIC_LICENSE_STATUS = "no_project_license_granted"
PUBLIC_IDENTITY_STATUS = "public_identity_disclosed"
SENSITIVE_ROOT_RE = re.compile(
    r"/(?:(?:" + "ho" + "me" + r")|(?:" + "m" + "nt" + r")|(?:" + "U" + "sers" + r"))(?:/|\b)",
    re.IGNORECASE,
)
WINDOWS_ABSOLUTE_RE = re.compile(r"\b[A-Za-z]:[\\/]")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SECRET_RE = re.compile(
    r"(?:BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|AKIA[0-9A-Z]{16}|"
    r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*[\"'][^\"']+[\"'])",
    re.IGNORECASE,
)


class DuplicateKeyError(ValueError):
    pass


def object_pairs_no_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise DuplicateKeyError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs_no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite JSON value: %s" % value)
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> bool:
    if not value or "\\" in value or WINDOWS_ABSOLUTE_RE.search(value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def regular_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        parts = PurePosixPath(relative).parts
        if (parts and parts[0] in RUNTIME_TOP_LEVEL) or tuple(parts[:2]) in RUNTIME_PREFIXES or any(
            part in TRANSIENT_DIR_NAMES for part in parts
        ):
            continue
        if path.is_symlink():
            raise ValueError("symbolic link is not allowed: %s" % relative)
        if path.is_file():
            files[relative] = path
    return files


def verify_sha256sums(files: dict[str, Path], errors: list[str]) -> None:
    if not HASH_FILE.is_file():
        errors.append("missing SHA256SUMS")
        return
    expected: dict[str, str] = {}
    for line_number, line in enumerate(HASH_FILE.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if not match:
            errors.append("invalid SHA256SUMS line %d" % line_number)
            continue
        digest, relative = match.groups()
        if not safe_relative(relative) or relative in expected:
            errors.append("unsafe or duplicate SHA256SUMS path: %s" % relative)
            continue
        expected[relative] = digest
    actual_names = set(files) - {"SHA256SUMS"}
    if set(expected) != actual_names:
        missing = sorted(actual_names - set(expected))
        extra = sorted(set(expected) - actual_names)
        errors.append("SHA256SUMS file-set mismatch: missing=%r extra=%r" % (missing, extra))
    for relative in sorted(set(expected) & actual_names):
        observed = sha256_file(files[relative])
        if observed != expected[relative]:
            errors.append("SHA mismatch: %s" % relative)


def verify_source_manifest(errors: list[str]) -> None:
    path = ROOT / "SOURCE_MANIFEST.json"
    try:
        data = load_json(path)
    except (OSError, ValueError) as exc:
        errors.append("invalid source manifest: %s" % exc)
        return
    if data.get("schema_version") != "rgate_release_source_manifest_v1":
        errors.append("unexpected source manifest schema")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("source manifest has no entries")
        return
    seen: set[str] = set()
    for entry in entries:
        relative = entry.get("bundled_path") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not safe_relative(relative) or relative in seen:
            errors.append("invalid or duplicate bundled source path: %r" % relative)
            continue
        seen.add(relative)
        bundled = ROOT / relative
        if not bundled.is_file() or bundled.is_symlink():
            errors.append("missing bundled source: %s" % relative)
            continue
        if bundled.stat().st_size != entry.get("bundled_size_bytes"):
            errors.append("bundled source size mismatch: %s" % relative)
        if sha256_file(bundled) != entry.get("bundled_sha256"):
            errors.append("bundled source SHA mismatch: %s" % relative)
        source_relative = entry.get("source_repo_path")
        if not isinstance(source_relative, str) or not safe_relative(source_relative):
            errors.append("invalid source repository path for %s" % relative)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _normalization_pointers(relative: str, contract: dict[str, Any]) -> list[str]:
    pointer_map = contract.get("allowed_changed_json_pointers")
    if not isinstance(pointer_map, dict):
        return []
    kind = "calibration" if "calibration" in PurePosixPath(relative).name else "model"
    pointers = pointer_map.get(kind)
    return list(pointers) if isinstance(pointers, list) else []


def _remove_json_pointer(value: Any, pointer: str) -> bool:
    if not pointer.startswith("/"):
        return False
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    parent = value
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return False
        parent = parent[part]
    if not isinstance(parent, dict) or not parts or parts[-1] not in parent:
        return False
    del parent[parts[-1]]
    return True


def _public_parameter_sha256(path: Path, pointers: list[str]) -> str:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError("model artifact root must be an object")
    for pointer in pointers:
        if not _remove_json_pointer(value, pointer):
            raise ValueError("normalization pointer is absent: %s" % pointer)
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _verify_normalization_receipt(
    *,
    artifact_id: str,
    relative: str,
    source_registration: dict[str, Any],
    public_identity: dict[str, Any],
    normalization: dict[str, Any],
    public_entry: dict[str, Any],
    allowed_pointers: list[str],
    errors: list[str],
) -> Optional[str]:
    expected_relative = (
        "normalization_receipts/%s.normalization_receipt.json" % artifact_id
    )
    receipt_identity = public_entry.get("normalization_receipt")
    if not isinstance(receipt_identity, dict) or set(receipt_identity) != {
        "relative_path",
        "size_bytes",
        "sha256",
    }:
        errors.append("missing normalization receipt registration: %s" % relative)
        return None
    receipt_relative = receipt_identity.get("relative_path")
    receipt_size = receipt_identity.get("size_bytes")
    receipt_sha = receipt_identity.get("sha256")
    if (
        receipt_relative != expected_relative
        or not safe_relative(str(receipt_relative))
        or not isinstance(receipt_size, int)
        or isinstance(receipt_size, bool)
        or receipt_size <= 0
        or not _valid_sha256(receipt_sha)
    ):
        errors.append("invalid normalization receipt identity: %s" % relative)
        return None
    if normalization.get("fixture_equivalence_receipt_sha256") != receipt_sha:
        errors.append("normalization receipt SHA is not bound to the public proof: %s" % relative)
    path = ROOT / receipt_relative
    if not path.is_file() or path.is_symlink():
        errors.append("registered normalization receipt is not a regular file: %s" % relative)
        return receipt_relative
    try:
        payload = path.read_bytes()
        receipt = load_json(path)
    except (OSError, ValueError) as exc:
        errors.append("invalid normalization receipt %s: %s" % (relative, exc))
        return receipt_relative
    if len(payload) != receipt_size or hashlib.sha256(payload).hexdigest() != receipt_sha:
        errors.append("normalization receipt byte identity mismatch: %s" % relative)
    try:
        if _canonical_json_bytes(receipt) != payload:
            errors.append("normalization receipt is not canonical JSON: %s" % relative)
    except (TypeError, ValueError):
        errors.append("normalization receipt is not finite canonical JSON: %s" % relative)
    expected_receipt_keys = {
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
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys:
        errors.append("normalization receipt fields differ from the audited schema: %s" % relative)
        return receipt_relative
    if receipt.get("schema_version") != NORMALIZATION_RECEIPT_SCHEMA:
        errors.append("unexpected normalization receipt schema: %s" % relative)
    if receipt.get("artifact_id") != artifact_id:
        errors.append("normalization receipt artifact id mismatch: %s" % relative)
    if receipt.get("intended_public_relative_path") != relative:
        errors.append("normalization receipt public path mismatch: %s" % relative)
    if receipt.get("canonicalization") != NORMALIZATION_CANONICALIZATION:
        errors.append("normalization receipt canonicalization mismatch: %s" % relative)
    expected_source_identity = {
        "size_bytes": source_registration.get("size_bytes"),
        "sha256": source_registration.get("sha256"),
    }
    if receipt.get("source_artifact_identity") != expected_source_identity:
        errors.append("normalization receipt source identity mismatch: %s" % relative)
    if receipt.get("public_artifact_identity") != public_identity:
        errors.append("normalization receipt public identity mismatch: %s" % relative)
    receipt_normalization = receipt.get("normalization")
    expected_receipt_normalization = {
        "schema_version": normalization.get("schema_version"),
        "changed_json_pointers": normalization.get("changed_json_pointers"),
        "source_inference_parameter_sha256": normalization.get(
            "source_inference_parameter_sha256"
        ),
        "public_inference_parameter_sha256": normalization.get(
            "public_inference_parameter_sha256"
        ),
    }
    if receipt_normalization != expected_receipt_normalization:
        errors.append("normalization receipt parameter proof mismatch: %s" % relative)
    fixture = receipt.get("fixture_equivalence")
    expected_kind = "calibration" if relative.endswith("_calibration.json") else "model"
    expected_fixture_schema = (
        CALIBRATION_FIXTURE_SCHEMA if expected_kind == "calibration" else MODEL_FIXTURE_SCHEMA
    )
    expected_method = (
        "apply_bge_af_arbiter.calibration_path_exact_float_hex_v1"
        if expected_kind == "calibration"
        else "apply_bge_af_arbiter.model_score_exact_float_hex_v1"
    )
    expected_fixture_keys = {
        "schema_version",
        "method",
        "comparison",
        "probe_count",
        "fixture_sha256",
        "source_output_sha256",
        "public_output_sha256",
    }
    if not isinstance(fixture, dict) or set(fixture) != expected_fixture_keys:
        errors.append("normalization receipt fixture fields are invalid: %s" % relative)
    else:
        if fixture.get("schema_version") != expected_fixture_schema or fixture.get("method") != expected_method:
            errors.append("normalization receipt fixture method mismatch: %s" % relative)
        if fixture.get("comparison") != "exact_canonical_float_hex":
            errors.append("normalization receipt fixture comparison mismatch: %s" % relative)
        probe_count = fixture.get("probe_count")
        if (
            not isinstance(probe_count, int)
            or isinstance(probe_count, bool)
            or (expected_kind == "model" and probe_count != 4)
            or (
                expected_kind == "calibration"
                and (probe_count < 5 or probe_count % 5 != 0)
            )
        ):
            errors.append("normalization receipt fixture probe count mismatch: %s" % relative)
        fixture_hashes = (
            fixture.get("fixture_sha256"),
            fixture.get("source_output_sha256"),
            fixture.get("public_output_sha256"),
        )
        if not all(_valid_sha256(value) for value in fixture_hashes):
            errors.append("normalization receipt fixture hashes are invalid: %s" % relative)
        elif fixture_hashes[1] != fixture_hashes[2]:
            errors.append("normalization receipt fixture outputs differ: %s" % relative)
    implementation_identities = receipt.get("implementation_identities")
    expected_implementations = {
        "inference": "scripts/apply_bge_af_arbiter.py",
        "normalizer": "tools/normalize_learned_artifact.py",
    }
    if not isinstance(implementation_identities, dict) or set(implementation_identities) != set(
        expected_implementations
    ):
        errors.append("normalization receipt implementation set mismatch: %s" % relative)
    else:
        for key, implementation_relative in expected_implementations.items():
            implementation = implementation_identities.get(key)
            implementation_path = ROOT / implementation_relative
            if not isinstance(implementation, dict) or set(implementation) != {
                "relative_path",
                "size_bytes",
                "sha256",
            }:
                errors.append("normalization receipt implementation identity invalid: %s" % relative)
                continue
            if implementation.get("relative_path") != implementation_relative:
                errors.append("normalization receipt implementation path mismatch: %s" % relative)
                continue
            try:
                expected_size = implementation_path.stat().st_size
                expected_sha = sha256_file(implementation_path)
            except OSError:
                errors.append("receipt-bound implementation is unavailable: %s" % relative)
                continue
            if implementation.get("size_bytes") != expected_size or implementation.get("sha256") != expected_sha:
                errors.append("receipt-bound implementation identity mismatch: %s" % relative)
    # The public verifier recomputes the parameter SHA separately. Keep the
    # explicit pointer list argument in this receipt check to fail if a caller
    # accidentally validates against a different contract.
    if normalization.get("changed_json_pointers") != allowed_pointers:
        errors.append("receipt validation pointer contract mismatch: %s" % relative)
    return receipt_relative


def verify_artifacts(
    allow_missing: bool, errors: list[str], warnings: list[str]
) -> None:
    try:
        data = load_json(ROOT / "ARTIFACT_MANIFEST.json")
    except (OSError, ValueError) as exc:
        errors.append("invalid artifact manifest: %s" % exc)
        return
    if data.get("schema_version") != "rgate_release_artifact_manifest_v2":
        errors.append("unexpected artifact manifest schema")
    if data.get("release_status") != PUBLIC_RELEASE_STATUS:
        errors.append("unexpected public release status")
    if data.get("license_status") != PUBLIC_LICENSE_STATUS:
        errors.append("unexpected project-license status")
    if data.get("double_blind_status") != PUBLIC_IDENTITY_STATUS:
        errors.append("unexpected public-identity status")
    contract = data.get("normalization_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != "rgate_release_metadata_normalization_v1":
        errors.append("invalid normalization contract")
        contract = {}
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 20:
        errors.append("artifact manifest must contain exactly twenty entries")
        return
    public_entries = data.get("public_artifacts")
    if not isinstance(public_entries, list):
        errors.append("public_artifacts must be a list")
        public_entries = []
    public_by_id: dict[str, dict[str, Any]] = {}
    for public in public_entries:
        public_id = public.get("id") if isinstance(public, dict) else None
        if not isinstance(public_id, str) or public_id in public_by_id:
            errors.append("invalid or duplicate public artifact id: %r" % public_id)
            continue
        public_by_id[public_id] = public

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    registered_public_paths: set[str] = set()
    registered_receipt_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            errors.append("non-object artifact entry")
            continue
        artifact_id = item.get("id")
        relative = item.get("relative_path")
        if not isinstance(artifact_id, str) or artifact_id in seen_ids:
            errors.append("invalid or duplicate artifact id: %r" % artifact_id)
        else:
            seen_ids.add(artifact_id)
        if not isinstance(relative, str) or not safe_relative(relative) or relative in seen_paths:
            errors.append("invalid or duplicate artifact path: %r" % relative)
            continue
        seen_paths.add(relative)
        if not relative.startswith("models/seed_") or not relative.endswith(".json"):
            errors.append("artifact path outside registered model layout: %s" % relative)
        if item.get("required") is not True or item.get("availability") != "missing_pending_verified_copy":
            errors.append("scientific-source state contract mismatch: %s" % relative)
        expected_size, expected_sha = item.get("size_bytes"), item.get("sha256")
        if not isinstance(expected_size, int) or expected_size <= 0:
            errors.append("invalid artifact size: %s" % relative)
        if not _valid_sha256(expected_sha):
            errors.append("invalid artifact SHA: %s" % relative)
        target = ROOT / relative
        public = public_by_id.pop(artifact_id, None)
        if public is None:
            if target.exists() or target.is_symlink():
                errors.append(
                    "unregistered file at public artifact path; private source copies are forbidden: %s"
                    % relative
                )
                continue
            message = "missing required learned artifact: %s" % relative
            (warnings if allow_missing else errors).append(message)
            continue
        if set(public) != PUBLIC_ARTIFACT_ENTRY_KEYS:
            errors.append("public artifact registration fields are invalid: %s" % relative)
        if public.get("relative_path") != relative or public.get("availability") != "present_verified":
            errors.append("public artifact state/path mismatch: %s" % relative)
            continue
        if not target.is_file() or target.is_symlink():
            errors.append("registered public artifact is not a regular file: %s" % relative)
            continue
        public_identity = public.get("public_artifact_identity")
        if not isinstance(public_identity, dict):
            errors.append("missing public artifact identity: %s" % relative)
            continue
        public_size = public_identity.get("size_bytes")
        public_sha = public_identity.get("sha256")
        if not isinstance(public_size, int) or public_size <= 0 or not _valid_sha256(public_sha):
            errors.append("invalid public artifact identity: %s" % relative)
            continue
        if target.stat().st_size != public_size or sha256_file(target) != public_sha:
            errors.append("public artifact identity mismatch: %s" % relative)
        else:
            try:
                public_value = load_json(target)
                if _canonical_json_bytes(public_value) != target.read_bytes():
                    errors.append("public artifact is not canonical JSON: %s" % relative)
            except (OSError, TypeError, ValueError) as exc:
                errors.append("cannot verify canonical public artifact %s: %s" % (relative, exc))
        registered_public_paths.add(relative)
        normalization = public.get("normalization")
        allowed = _normalization_pointers(relative, contract)
        if not isinstance(normalization, dict) or normalization.get("schema_version") != contract.get("schema_version"):
            errors.append("missing public normalization proof: %s" % relative)
            continue
        if normalization.get("changed_json_pointers") != allowed:
            errors.append("normalization pointer set mismatch: %s" % relative)
        source_parameter_sha = normalization.get("source_inference_parameter_sha256")
        public_parameter_sha = normalization.get("public_inference_parameter_sha256")
        fixture_sha = normalization.get("fixture_equivalence_receipt_sha256")
        if not (_valid_sha256(source_parameter_sha) and _valid_sha256(public_parameter_sha)):
            errors.append("invalid parameter-equivalence hashes: %s" % relative)
        elif source_parameter_sha != public_parameter_sha:
            errors.append("source/public parameter payload mismatch: %s" % relative)
        try:
            observed_parameter_sha = _public_parameter_sha256(target, allowed)
        except (OSError, ValueError) as exc:
            errors.append("cannot verify public parameter payload %s: %s" % (relative, exc))
        else:
            if observed_parameter_sha != public_parameter_sha:
                errors.append("public parameter payload SHA mismatch: %s" % relative)
        try:
            public_value_for_metadata = load_json(target)
            if relative.endswith("_calibration.json"):
                expected_values = {
                    "/cache_jsonl": "artifacts/cache/fullval_cache.jsonl",
                    "/model": relative[: -len("_calibration.json")] + "_model.json",
                }
            else:
                expected_values = {
                    "/training/cache_jsonl": "artifacts/cache/fullval_cache.jsonl"
                }
            for pointer, expected_value in expected_values.items():
                parts = [
                    part.replace("~1", "/").replace("~0", "~")
                    for part in pointer[1:].split("/")
                ]
                observed_value: Any = public_value_for_metadata
                for part in parts:
                    if not isinstance(observed_value, dict) or part not in observed_value:
                        raise ValueError("normalization pointer is absent: %s" % pointer)
                    observed_value = observed_value[part]
                if observed_value != expected_value:
                    errors.append("public normalization metadata value mismatch: %s" % relative)
        except (OSError, ValueError) as exc:
            errors.append("cannot verify public normalization metadata %s: %s" % (relative, exc))
        if not _valid_sha256(fixture_sha):
            errors.append("missing fixed-fixture equivalence receipt identity: %s" % relative)
        receipt_relative = _verify_normalization_receipt(
            artifact_id=artifact_id,
            relative=relative,
            source_registration=item,
            public_identity=public_identity,
            normalization=normalization,
            public_entry=public,
            allowed_pointers=allowed,
            errors=errors,
        )
        if receipt_relative is not None:
            if receipt_relative in registered_receipt_paths:
                errors.append("duplicate normalization receipt path: %s" % receipt_relative)
            registered_receipt_paths.add(receipt_relative)
    if public_by_id:
        errors.append("public artifacts have no scientific-source registry entry: %r" % sorted(public_by_id))
    model_root = ROOT / "models"
    if model_root.is_dir():
        for path in model_root.rglob("*.json"):
            relative = path.relative_to(ROOT).as_posix()
            if relative not in registered_public_paths:
                errors.append("unregistered learned JSON in models tree: %s" % relative)
    receipt_root = ROOT / "normalization_receipts"
    if receipt_root.is_dir():
        for path in receipt_root.rglob("*"):
            if path.is_file() or path.is_symlink():
                relative = path.relative_to(ROOT).as_posix()
                if relative not in registered_receipt_paths:
                    errors.append("unregistered normalization receipt: %s" % relative)


def verify_json_and_relative_paths(errors: list[str]) -> None:
    loaded: dict[str, Any] = {}
    for relative in ("ARTIFACT_MANIFEST.json", "SOURCE_MANIFEST.json", "configs/repro.example.json"):
        try:
            loaded[relative] = load_json(ROOT / relative)
        except (OSError, ValueError) as exc:
            errors.append("invalid JSON document %s: %s" % (relative, exc))
    config = loaded.get("configs/repro.example.json")
    if not isinstance(config, dict):
        return
    paths = config.get("paths")
    if not isinstance(paths, dict) or not paths:
        errors.append("example config paths must be a non-empty object")
    else:
        for key, value in paths.items():
            if not isinstance(value, str) or not safe_relative(value):
                errors.append("non-relative configured path at paths.%s" % key)
    experts = config.get("expert_results_in_fixed_order")
    if not isinstance(experts, list) or len(experts) != 4:
        errors.append("example config must contain exactly four expert result paths")
    else:
        for index, value in enumerate(experts):
            if not isinstance(value, str) or not safe_relative(value):
                errors.append("non-relative expert result path at index %d" % index)


def verify_release_state_documents(errors: list[str]) -> None:
    required_fragments = {
        "README.md": ("research-code companion", "not a self-contained end-to-end reproducibility package"),
        "RELEASE_STATUS.md": ("status: **public_research_code_companion**", "no project license"),
        "NOTICE": ("public research-code archive", "not sublicense the dataset"),
    }
    for relative, fragments in required_fragments.items():
        try:
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
        except OSError as exc:
            errors.append("cannot read release-state document %s: %s" % (relative, exc))
            continue
        for fragment in fragments:
            if fragment not in text:
                errors.append("release-state document contract mismatch: %s" % relative)
    if (ROOT / "LICENSE_PENDING.md").exists():
        errors.append("obsolete pending-license document is present")
    try:
        ignore_lines = {
            line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
    except OSError as exc:
        errors.append("cannot read .gitignore: %s" % exc)
        return
    forbidden_model_patterns = {"models/*.json", "models/**/*.json"}
    if ignore_lines & forbidden_model_patterns:
        errors.append("public normalized model JSON files must not be ignored")


def verify_publication_safety(files: dict[str, Path], errors: list[str]) -> None:
    for relative, path in files.items():
        if path.suffix.lower() in PAPER_SUFFIXES:
            errors.append("paper-source file is excluded from this candidate: %s" % relative)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "NOTICE":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append("unexpected non-UTF-8 candidate file: %s" % relative)
            continue
        if SENSITIVE_ROOT_RE.search(text) or WINDOWS_ABSOLUTE_RE.search(text):
            errors.append("environment-specific absolute path found: %s" % relative)
        if EMAIL_RE.search(text):
            errors.append("email-like identifier found: %s" % relative)
        if SECRET_RE.search(text):
            errors.append("secret-like token found: %s" % relative)


def verify_import_and_syntax_closure(errors: list[str]) -> None:
    python_files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    local_modules = {path.stem for path in python_files}
    standard = (
        set(getattr(sys, "stdlib_module_names", ()))
        | set(sys.builtin_module_names)
        | PYTHON38_BUNDLED_STDLIB_MODULES
    )
    for path in python_files + [ROOT / "verify_bundle.py", ROOT / "tests/synthetic_smoke.py"]:
        relative = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, relative, "exec")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append("Python syntax failure in %s: %s" % (relative, exc))
            continue
        if not relative.startswith(("scripts/", "tools/")):
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        unresolved = imports - standard - local_modules - OPTIONAL_EXTERNAL_MODULES
        if unresolved:
            errors.append("unclosed imports in %s: %r" % (relative, sorted(unresolved)))


def run_smoke(errors: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "tests.synthetic_smoke"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        errors.append("synthetic smoke failed:\n%s\n%s" % (result.stdout, result.stderr))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Treat absent registered model JSON files as warnings; wrong files still fail.",
    )
    parser.add_argument("--run-smoke", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        files = regular_files()
    except ValueError as exc:
        print("FAIL: %s" % exc)
        return 1
    verify_sha256sums(files, errors)
    verify_source_manifest(errors)
    verify_artifacts(args.allow_missing_models, errors, warnings)
    verify_json_and_relative_paths(errors)
    verify_release_state_documents(errors)
    verify_publication_safety(files, errors)
    verify_import_and_syntax_closure(errors)
    if args.run_smoke:
        run_smoke(errors)
    for message in warnings:
        print("WARN: %s" % message)
    for message in errors:
        print("FAIL: %s" % message)
    if errors:
        print("FAILED: %d error(s), %d warning(s)" % (len(errors), len(warnings)))
        return 1
    print("PASS: candidate structure and integrity checks; %d warning(s)" % len(warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
