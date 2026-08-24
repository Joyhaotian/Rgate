#!/usr/bin/env python3
"""Materialize the locked RQ2/RQ3 comparison stages in one cache pass.

This runner intentionally does not evaluate or train anything.  It consumes the
promoted, grouped BGE-AF cluster cache and writes five standard nuScenes result
JSONs:

* a genuinely symmetric four-expert, raw-result WBF baseline;
* the frozen R-GATE clusters without candidate-only retrieval;
* learned radar scoring before calibration;
* learned radar scoring after calibration; and
* the calibrated learned no-radar control.

The historical full-val cache contains cluster membership and non-radar
features, but its radar features are placeholders and it lacks ego-relative
range.  Per-sample radar evidence and ``ego_range_xy`` are therefore rebuilt
from the locked nuScenes inputs in memory.  The cache, its cluster boxes, and
all other features are read-only.

Large outputs are written into a new staging directory and become visible only
after the complete directory is atomically renamed. Full runs are restricted
to the live mounted storage root supplied with ``--storage-mount``; a bounded
smoke can opt out explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import re
from types import SimpleNamespace
import tempfile
from typing import Any, BinaryIO, Iterator

from apply_bge_af_arbiter import (
    load_calibration_table,
    load_meta,
    load_model,
    row_calibration_key,
    score_row,
    sorted_sample_rows,
)
from build_bge_af_arbiter_cache import (
    ResultJsonIndex,
    SCHEMA_VERSION as CACHE_ROW_SCHEMA,
    _ego_xy_from_info,
    _load_infos_by_token,
    _radar_evidence_features,
    _radar_points_global_for_info,
    _radars_from_nuscenes_tables,
    _safe_float,
)
from fuse_nuscenes_expert_results import (
    DEFAULT_CLUSTER_THRESHOLDS,
    quat_from_yaw,
    yaw_from_quat,
)


SUMMARY_SCHEMA = "rgate_rq2_stage_materialization_summary_v1"
EXECUTION_CONFIG_SCHEMA = "rgate_rq2_stage_execution_config_v1"
OUTPUT_FILENAMES = {
    "naive_equal_wbf": "naive_equal_wbf_results_nusc.json",
    "rgate_rescore_only": "rgate_rescore_only_results_nusc.json",
    "learned_radar_uncalibrated": "learned_radar_uncalibrated_results_nusc.json",
    "learned_radar_calibrated": "learned_radar_calibrated_results_nusc.json",
    "learned_no_radar_calibrated": "learned_no_radar_calibrated_results_nusc.json",
}
CACHE_STAGE_NAMES = (
    "rgate_rescore_only",
    "learned_radar_uncalibrated",
    "learned_radar_calibrated",
    "learned_no_radar_calibrated",
)
LOCKED_SCORE_BLEND = 0.1
LOCKED_MAX_PER_SAMPLE = 500
LOCKED_RADAR_MAX_SWEEPS = 1
LOCKED_RADAR_BOX_MARGIN = 1.0
LOCKED_RADAR_SEARCH_RADIUS = 4.0
LOCKED_NAIVE_DEFAULT_RADIUS = 0.7
HASH_CHUNK_BYTES = 8 * 1024 * 1024
TOKEN_FIELD_RE = re.compile(rb'"token"\s*:\s*"([0-9a-f]+)"')
SAMPLE_TOKEN_FIELD_RE = re.compile(rb'"sample_token"\s*:\s*"([0-9a-f]+)"')


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if hash_file else None,
    }


def maybe_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False}
    identity = file_identity(path)
    identity["exists"] = True
    return identity


def _require_identity_match(
    *, label: str, actual_path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise SystemExit(f"execution config is missing identity: {label}")
    configured_path = Path(str(expected.get("path", "")))
    if configured_path.resolve() != actual_path.resolve():
        raise SystemExit(
            f"execution config path drift for {label}: "
            f"configured={configured_path} cli={actual_path}"
        )
    actual = file_identity(actual_path)
    for key in ("size_bytes", "sha256"):
        if expected.get(key) != actual[key]:
            raise SystemExit(
                f"execution config identity drift for {label}.{key}: "
                f"configured={expected.get(key)!r} actual={actual[key]!r}"
            )
    return actual


def _same_path(left: Any, right: Any) -> bool:
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def _mount_source(path: Path) -> str:
    target = str(path.resolve())
    candidates: list[tuple[int, str]] = []
    escapes = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    with Path("/proc/self/mountinfo").open("r", encoding="utf-8") as fh:
        for line in fh:
            left, separator, right = line.rstrip("\n").partition(" - ")
            if not separator:
                continue
            fields = left.split()
            after = right.split()
            if len(fields) < 5 or len(after) < 2:
                continue
            mount_point = fields[4]
            source = after[1]
            for encoded, decoded in escapes.items():
                mount_point = mount_point.replace(encoded, decoded)
                source = source.replace(encoded, decoded)
            mount_path = Path(mount_point)
            if is_relative_to(Path(target), mount_path):
                candidates.append((len(mount_path.parts), source))
    if not candidates:
        raise SystemExit(f"could not resolve mount source for {path}")
    return max(candidates)[1]


def _validate_config_against_plan(
    config: dict[str, Any], plan_payload: dict[str, Any]
) -> None:
    """Prove that the execution lock is an implementation of the preregistration."""

    inputs = config["inputs"]
    registered = plan_payload.get("registered_inputs")
    if not isinstance(registered, dict):
        raise SystemExit("preregistered plan has no registered_inputs")

    def require_registered(
        config_key: str,
        plan_key: str,
        *,
        plan_path_key: str = "path",
        require_size: bool = False,
    ) -> None:
        actual = inputs.get(config_key)
        expected = registered.get(plan_key)
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise SystemExit(f"execution config/plan lacks {config_key}/{plan_key}")
        if not _same_path(actual.get("path"), expected.get(plan_path_key)):
            raise SystemExit(f"execution config {config_key} path is not preregistered")
        if actual.get("sha256") != expected.get("sha256"):
            raise SystemExit(f"execution config {config_key} SHA is not preregistered")
        if require_size and actual.get("size_bytes") != expected.get("size_bytes"):
            raise SystemExit(f"execution config {config_key} size is not preregistered")

    require_registered("cache_jsonl", "merged_cluster_cache", require_size=True)
    require_registered(
        "sample_info_pkl",
        "val_infos_for_radar_repair",
        plan_path_key="resolved_path",
    )
    # The preregistration calls the serialized model records ``*_mlp`` while
    # the execution CLI uses the generic ``*_model`` names.  Keep this mapping
    # explicit: accepting a same-named fallback would weaken the frozen-plan
    # closure and was the reason the first bounded smoke correctly stopped
    # before opening any large input.
    for config_key, plan_key in (
        ("radar_model", "radar_mlp"),
        ("radar_calibration", "radar_calibration"),
        ("no_radar_model", "no_radar_mlp"),
        ("no_radar_calibration", "no_radar_calibration"),
    ):
        require_registered(config_key, plan_key)

    configured_experts = inputs.get("naive_experts")
    registered_experts = registered.get("expert_pool")
    if not isinstance(configured_experts, list) or not isinstance(registered_experts, list):
        raise SystemExit("execution config/plan lacks expert pool")
    if len(configured_experts) != 4 or len(registered_experts) != 4:
        raise SystemExit("execution config/plan expert pool must contain four entries")
    for ordinal, (actual, expected) in enumerate(
        zip(configured_experts, registered_experts)
    ):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise SystemExit("malformed execution config/plan expert entry")
        if (
            actual.get("name") != expected.get("name")
            or int(actual.get("fixed_ordinal", -1)) != ordinal
            or not _same_path(actual.get("path"), expected.get("path"))
            or actual.get("sha256") != expected.get("sha256")
            or actual.get("size_bytes") != expected.get("size_bytes")
        ):
            raise SystemExit(f"execution config expert {ordinal} is not preregistered")

    configured_nusc = inputs.get("nuscenes")
    if not isinstance(configured_nusc, dict) or not _same_path(
        configured_nusc.get("root"), registered.get("nuscenes_root")
    ):
        raise SystemExit("execution config nuScenes root is not preregistered")

    scopes = config["scopes"]
    full_scope = scopes.get("full")
    storage = plan_payload.get("storage_and_safety")
    if not isinstance(full_scope, dict) or not isinstance(storage, dict):
        raise SystemExit("execution config/plan lacks full storage contract")
    if not _same_path(full_scope.get("output_dir"), storage.get("large_output_root")):
        raise SystemExit("execution config full output is not preregistered")
    if int(full_scope.get("minimum_free_bytes", -1)) < int(
        storage.get("minimum_D_free_bytes_before_start", -1)
    ):
        raise SystemExit("execution config weakens the preregistered free-space floor")


def _load_and_validate_execution_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_sha = sha256_file(args.execution_config_json)
    if config_sha != args.expected_execution_config_sha256:
        raise SystemExit(
            "execution config SHA mismatch: "
            f"expected={args.expected_execution_config_sha256} actual={config_sha}"
        )
    config = json.loads(args.execution_config_json.read_text(encoding="utf-8"))
    if config.get("schema_version") != EXECUTION_CONFIG_SCHEMA:
        raise SystemExit(
            f"unsupported execution config schema: {config.get('schema_version')}"
        )
    if config.get("status") != "frozen_for_execution":
        raise SystemExit(
            f"execution config is not frozen_for_execution: {config.get('status')}"
        )

    plan = config.get("plan")
    if not isinstance(plan, dict):
        raise SystemExit("execution config is missing plan identity")
    plan_path = Path(str(plan.get("path", "")))
    plan_identity = _require_identity_match(
        label="plan", actual_path=plan_path, expected=plan
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan_payload.get("schema_version") != "rgate_rq2_cpu_controls_plan_v1":
        raise SystemExit("execution config plan has the wrong schema")
    if plan_payload.get("status") != "preregistered_before_full_run":
        raise SystemExit("execution config plan is not preregistered_before_full_run")

    inputs = config.get("inputs")
    implementation = config.get("implementation")
    scopes = config.get("scopes")
    counts = config.get("registered_counts")
    if not all(isinstance(value, dict) for value in (inputs, implementation, scopes, counts)):
        raise SystemExit("execution config is missing inputs/implementation/scopes/counts")
    assert isinstance(inputs, dict) and isinstance(implementation, dict)
    assert isinstance(scopes, dict) and isinstance(counts, dict)
    if int(counts.get("samples", -1)) != 6019 or int(counts.get("cache_rows", -1)) != 4732657:
        raise SystemExit("execution config must lock 6019 samples and 4732657 cache rows")
    _validate_config_against_plan(config, plan_payload)

    cli_inputs = {
        "cache_jsonl": args.cache_jsonl,
        "cache_summary_json": args.cache_summary_json,
        "sample_info_pkl": args.sample_info_pkl,
        "radar_model": args.radar_model,
        "radar_calibration": args.radar_calibration,
        "no_radar_model": args.no_radar_model,
        "no_radar_calibration": args.no_radar_calibration,
        "meta_from_result_json": args.meta_from_result_json,
    }
    validated_inputs: dict[str, Any] = {}
    for label, path in cli_inputs.items():
        if path is None:
            raise SystemExit(f"--{label.replace('_', '-')} is required by execution config")
        validated_inputs[label] = _require_identity_match(
            label=label, actual_path=path, expected=inputs.get(label)
        )

    configured_experts = inputs.get("naive_experts")
    if not isinstance(configured_experts, list) or len(configured_experts) != 4:
        raise SystemExit("execution config must contain exactly four naive_experts")
    if len(args.naive_expert) != 4:
        raise SystemExit("CLI must contain exactly four naive experts")
    validated_experts: list[dict[str, Any]] = []
    for ordinal, ((name, path), expected) in enumerate(
        zip(args.naive_expert, configured_experts)
    ):
        if not isinstance(expected, dict):
            raise SystemExit(f"malformed configured naive expert {ordinal}")
        if expected.get("name") != name or int(expected.get("fixed_ordinal", -1)) != ordinal:
            raise SystemExit(f"naive expert name/order drift at ordinal {ordinal}")
        validated_experts.append(
            {
                "name": name,
                "fixed_ordinal": ordinal,
                **_require_identity_match(
                    label=f"naive_experts[{ordinal}]", actual_path=path, expected=expected
                ),
            }
        )

    script_dir = Path(__file__).resolve().parent
    implementation_paths = {
        "materializer": Path(__file__).resolve(),
        "independent_checker": script_dir / "check_rgate_rq2_stage_results.py",
        "apply_bge_af_arbiter": script_dir / "apply_bge_af_arbiter.py",
        "build_bge_af_arbiter_cache": script_dir / "build_bge_af_arbiter_cache.py",
        "fuse_nuscenes_expert_results": script_dir / "fuse_nuscenes_expert_results.py",
    }
    validated_implementation = {
        label: _require_identity_match(
            label=f"implementation.{label}",
            actual_path=path,
            expected=implementation.get(label),
        )
        for label, path in implementation_paths.items()
    }

    configured_nusc = inputs.get("nuscenes")
    if not isinstance(configured_nusc, dict):
        raise SystemExit("execution config is missing inputs.nuscenes")
    if Path(str(configured_nusc.get("root", ""))).resolve() != args.nuscenes_root.resolve():
        raise SystemExit("nuScenes root drift from execution config")
    if configured_nusc.get("version") != args.nuscenes_version:
        raise SystemExit("nuScenes version drift from execution config")
    configured_tables = configured_nusc.get("tables")
    if not isinstance(configured_tables, dict):
        raise SystemExit("execution config is missing nuScenes table identities")
    table_root = args.nuscenes_root / args.nuscenes_version
    validated_tables = {
        name: _require_identity_match(
            label=f"nuscenes.tables.{name}",
            actual_path=table_root / f"{name}.json",
            expected=configured_tables.get(name),
        )
        for name in ("sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor")
    }

    if args.max_samples:
        scope_name = "bounded_smoke"
        scope = scopes.get(scope_name)
        if not isinstance(scope, dict):
            raise SystemExit("execution config does not authorize bounded_smoke")
        maximum = int(scope.get("max_samples_max", 0))
        if maximum <= 0 or args.max_samples > maximum:
            raise SystemExit(
                f"smoke max_samples exceeds execution config: {args.max_samples} > {maximum}"
            )
        smoke_root = Path(str(scope.get("output_root", ""))).resolve()
        if not is_relative_to(args.out_dir.resolve(), smoke_root):
            raise SystemExit(f"smoke output is outside configured root: {smoke_root}")
        if bool(scope.get("allow_non_mount")) != bool(args.allow_non_mount_smoke):
            raise SystemExit("--allow-non-mount-smoke differs from execution config")
    else:
        scope_name = "full"
        scope = scopes.get(scope_name)
        if not isinstance(scope, dict):
            raise SystemExit("execution config does not authorize full scope")
        if Path(str(scope.get("output_dir", ""))).resolve() != args.out_dir.resolve():
            raise SystemExit("full output directory differs from execution config")
        if Path(str(scope.get("storage_mount", ""))).resolve() != args.storage_mount.resolve():
            raise SystemExit("full storage mount differs from execution config")
        expected_source = str(scope.get("expected_mount_source", ""))
        if not expected_source:
            raise SystemExit("execution config full scope lacks expected_mount_source")
        actual_source = _mount_source(args.storage_mount)
        if actual_source != expected_source:
            raise SystemExit(
                f"storage mount source drift: expected={expected_source!r} actual={actual_source!r}"
            )
        if args.allow_non_mount_smoke:
            raise SystemExit("full scope may not bypass the storage mount")
        required = int(scope.get("minimum_free_bytes", -1))
        configured_by_cli = int(args.min_free_gib * (1024**3))
        if required < 0 or configured_by_cli < required:
            raise SystemExit("--min-free-gib is below the execution config minimum")

    return config, {
        "execution_config": file_identity(args.execution_config_json),
        "plan": plan_identity,
        "validated_inputs": validated_inputs,
        "validated_experts": validated_experts,
        "validated_implementation": validated_implementation,
        "validated_nuscenes_tables": validated_tables,
        "scope": scope_name,
    }


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass
class StorageGuard:
    mount: Path
    out_dir: Path
    allow_non_mount_smoke: bool
    max_samples: int
    min_free_gib: float
    recheck_every: int
    initial_device: int | None = None
    initial_free_bytes: int = 0
    required_free_bytes: int = 0

    def preflight(self, *, estimated_output_bytes: int) -> dict[str, Any]:
        if self.min_free_gib < 0:
            raise SystemExit("--min-free-gib must be >= 0")
        if self.recheck_every <= 0:
            raise SystemExit("--mount-recheck-every must be > 0")
        self.mount = self.mount.resolve()
        self.out_dir = self.out_dir.resolve()
        bypass = self.allow_non_mount_smoke
        if bypass and self.max_samples <= 0:
            raise SystemExit("--allow-non-mount-smoke requires --max-samples > 0")
        if not bypass:
            if not self.mount.is_dir() or not os.path.ismount(self.mount):
                raise SystemExit(f"storage mount is not live: {self.mount}")
            if not is_relative_to(self.out_dir, self.mount):
                raise SystemExit(
                    f"--out-dir must be below live storage mount {self.mount}: {self.out_dir}"
                )
            self.initial_device = self.mount.stat().st_dev
        else:
            self.initial_device = None

        probe = self.mount if not bypass else self.out_dir.parent
        probe.mkdir(parents=True, exist_ok=True)
        # Close the mount-check -> mkdir race: if D disappeared in that window,
        # mkdir may have created a same-named directory on the WSL root device.
        if not bypass:
            if not self.mount.is_dir() or not os.path.ismount(self.mount):
                raise SystemExit(f"storage mount disappeared during preflight: {self.mount}")
            if self.mount.stat().st_dev != self.initial_device:
                raise SystemExit(f"storage mount device changed during preflight: {self.mount}")
        usage = shutil.disk_usage(probe)
        self.initial_free_bytes = usage.free
        reserve = int(self.min_free_gib * (1024**3))
        self.required_free_bytes = max(reserve, estimated_output_bytes)
        if usage.free < self.required_free_bytes:
            raise SystemExit(
                "insufficient output space: "
                f"free={usage.free} required={self.required_free_bytes} at {probe}"
            )
        return {
            "storage_mount": str(self.mount),
            "out_dir": str(self.out_dir),
            "mount_guard_bypassed_for_smoke": bypass,
            "mount_device": self.initial_device,
            "initial_free_bytes": self.initial_free_bytes,
            "required_free_bytes": self.required_free_bytes,
            "estimated_output_bytes": estimated_output_bytes,
            "min_free_gib": self.min_free_gib,
            "recheck_every_samples": self.recheck_every,
        }

    def recheck(self) -> None:
        if self.allow_non_mount_smoke:
            return
        if not self.mount.is_dir() or not os.path.ismount(self.mount):
            raise RuntimeError(f"storage mount disappeared during materialization: {self.mount}")
        if self.mount.stat().st_dev != self.initial_device:
            raise RuntimeError(f"storage mount device changed during materialization: {self.mount}")


class ResultJsonWriter:
    """Compact result writer with an online byte identity."""

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.fh: BinaryIO = path.open("xb")
        self.digest = hashlib.sha256()
        self.first = True
        self.sample_count = 0
        self.box_count = 0
        self.closed = False
        self._write(b'{"meta":')
        self._write(canonical_json_bytes(meta))
        self._write(b',"results":{')

    def _write(self, value: bytes) -> None:
        self.fh.write(value)
        self.digest.update(value)

    def write_sample(self, token: str, rows: list[dict[str, Any]]) -> None:
        if self.closed:
            raise RuntimeError("result writer is already closed")
        if not self.first:
            self._write(b",")
        self.first = False
        self._write(canonical_json_bytes(str(token)))
        self._write(b":")
        self._write(canonical_json_bytes(rows))
        self.sample_count += 1
        self.box_count += len(rows)

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self._write(b"}}")
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()
            self.closed = True
        return {
            "filename": self.path.name,
            "sample_count": self.sample_count,
            "output_box_count": self.box_count,
            "size_bytes": self.path.stat().st_size,
            "sha256": self.digest.hexdigest(),
            "standard_nuscenes_result_json": True,
            "top_k_per_sample": LOCKED_MAX_PER_SAMPLE,
        }

    def abort(self) -> None:
        if not self.closed:
            self.fh.close()
            self.closed = True


@dataclass(frozen=True)
class NaiveExpertSpec:
    name: str
    path: Path
    ordinal: int


def parse_naive_expert(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=artifacts/results_nusc.json")
    name, path_text = value.split("=", 1)
    name = name.strip()
    path = Path(path_text)
    if not name:
        raise argparse.ArgumentTypeError("naive expert name must be non-empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing naive expert result: {path}")
    return name, path


@dataclass(frozen=True)
class NaiveCandidate:
    source: str
    source_ordinal: int
    input_row_ordinal: int
    score: float
    row: dict[str, Any]

    @property
    def xy(self) -> tuple[float, float]:
        translation = self.row["translation"]
        return float(translation[0]), float(translation[1])

    @property
    def deterministic_key(self) -> tuple[float, int, int]:
        return (-self.score, self.source_ordinal, self.input_row_ordinal)


@dataclass
class NaiveCluster:
    creation_ordinal: int
    members: list[NaiveCandidate] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    @property
    def center_xy(self) -> tuple[float, float]:
        if not self.members:
            raise RuntimeError("empty naive cluster")
        return (
            sum(item.xy[0] for item in self.members) / len(self.members),
            sum(item.xy[1] for item in self.members) / len(self.members),
        )

    def append(self, item: NaiveCandidate) -> None:
        if item.source in self.sources:
            raise RuntimeError("naive cluster may contain at most one box per source")
        self.members.append(item)
        self.sources.add(item.source)


def _mean_vector(values: list[list[float]], width: int) -> list[float]:
    if not values:
        return [0.0] * width
    return [sum(float(value[idx]) for value in values) / len(values) for idx in range(width)]


def _validated_raw_candidate(
    *, source: str, source_ordinal: int, row_ordinal: int, row: dict[str, Any]
) -> NaiveCandidate:
    required = ("translation", "size", "rotation", "detection_name")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"raw expert row missing {missing}: {source}[{row_ordinal}]")
    if len(row["translation"]) < 3 or len(row["size"]) < 3 or len(row["rotation"]) < 4:
        raise ValueError(f"malformed raw expert geometry: {source}[{row_ordinal}]")
    score = _safe_float(row.get("detection_score"), float("nan"))
    if not math.isfinite(score):
        raise ValueError(f"non-finite raw expert score: {source}[{row_ordinal}]")
    return NaiveCandidate(
        source=source,
        source_ordinal=source_ordinal,
        input_row_ordinal=row_ordinal,
        score=max(0.0, score),
        row=row,
    )


def _naive_cluster_to_box(sample_token: str, class_name: str, cluster: NaiveCluster) -> dict[str, Any]:
    members = cluster.members
    if not members:
        raise RuntimeError("cannot fuse an empty naive cluster")
    translations = [[float(value) for value in item.row["translation"][:3]] for item in members]
    sizes = [[float(value) for value in item.row["size"][:3]] for item in members]
    velocities: list[list[float]] = []
    for item in members:
        velocity = [float(value) for value in item.row.get("velocity", [0.0, 0.0])[:2]]
        while len(velocity) < 2:
            velocity.append(0.0)
        velocities.append(velocity)
    yaws = [yaw_from_quat([float(value) for value in item.row["rotation"][:4]]) for item in members]
    yaw = math.atan2(
        sum(math.sin(value) for value in yaws),
        sum(math.cos(value) for value in yaws),
    )
    best = min(members, key=lambda item: item.deterministic_key)
    return {
        "sample_token": sample_token,
        "translation": _mean_vector(translations, 3),
        "size": _mean_vector(sizes, 3),
        "rotation": quat_from_yaw(yaw),
        "velocity": _mean_vector(velocities, 2),
        "detection_name": class_name,
        "detection_score": sum(item.score for item in members) / len(members),
        "attribute_name": str(best.row.get("attribute_name", "")),
    }


def naive_equal_wbf_sample(
    *,
    sample_token: str,
    rows_by_expert: list[tuple[NaiveExpertSpec, list[dict[str, Any]]]],
    max_per_sample: int = LOCKED_MAX_PER_SAMPLE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse raw experts symmetrically without inheriting R-GATE clusters/trust."""

    by_class: dict[str, list[NaiveCandidate]] = defaultdict(list)
    input_counts: dict[str, int] = {}
    for spec, rows in rows_by_expert:
        input_counts[spec.name] = len(rows)
        for row_ordinal, row in enumerate(rows):
            candidate = _validated_raw_candidate(
                source=spec.name,
                source_ordinal=spec.ordinal,
                row_ordinal=row_ordinal,
                row=row,
            )
            by_class[str(row["detection_name"])].append(candidate)

    fused_with_order: list[tuple[int, dict[str, Any], int]] = []
    next_cluster_ordinal = 0
    support_hist: Counter[int] = Counter()
    for class_name in sorted(by_class):
        threshold = DEFAULT_CLUSTER_THRESHOLDS.get(class_name, LOCKED_NAIVE_DEFAULT_RADIUS)
        clusters: list[NaiveCluster] = []
        for candidate in sorted(by_class[class_name], key=lambda item: item.deterministic_key):
            legal: list[tuple[float, int, NaiveCluster]] = []
            for cluster in clusters:
                if candidate.source in cluster.sources:
                    continue
                cx, cy = cluster.center_xy
                distance = math.hypot(candidate.xy[0] - cx, candidate.xy[1] - cy)
                if distance <= threshold:
                    legal.append((distance, cluster.creation_ordinal, cluster))
            if legal:
                _, _, selected = min(legal, key=lambda item: (item[0], item[1]))
            else:
                selected = NaiveCluster(creation_ordinal=next_cluster_ordinal)
                next_cluster_ordinal += 1
                clusters.append(selected)
            selected.append(candidate)

        for cluster in clusters:
            support_hist[len(cluster.sources)] += 1
            fused_with_order.append(
                (
                    cluster.creation_ordinal,
                    _naive_cluster_to_box(sample_token, class_name, cluster),
                    len(cluster.sources),
                )
            )

    fused_with_order.sort(
        key=lambda item: (-float(item[1]["detection_score"]), item[0])
    )
    rows = [row for _, row, _ in fused_with_order[:max_per_sample]]
    return rows, {
        "input_box_counts": input_counts,
        "cluster_count": len(fused_with_order),
        "output_box_count": len(rows),
        "support_histogram": dict(sorted(support_hist.items())),
        "truncated": len(fused_with_order) > max_per_sample,
    }


def _plain_xy(value: Any) -> tuple[float, float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return _safe_float(value[0]), _safe_float(value[1])


def ego_xy_for_sample(info: dict[str, Any], radar_args: SimpleNamespace) -> tuple[tuple[float, float], str]:
    nusc = getattr(radar_args, "_rgate_nuscenes", None)
    info_translation = _plain_xy(info.get("ego2global_translation"))
    if info_translation is None:
        info_translation = _ego_xy_from_info(info)
    if nusc is not None:
        token = str(info.get("sample_token") or info.get("token") or "")
        sample = nusc.get("sample", token)
        data = sample.get("data", {}) if isinstance(sample, dict) else {}
        sample_data_token = data.get("LIDAR_TOP") or next(iter(data.values()), None)
        if not sample_data_token:
            raise ValueError(f"nuScenes sample has no sample_data: {token}")
        sample_data = nusc.get("sample_data", sample_data_token)
        pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        table_translation = (
            _plain_xy(pose.get("translation")) if isinstance(pose, dict) else None
        )
        if table_translation is None:
            raise ValueError(f"nuScenes ego pose has no translation: {token}")
        if info_translation is not None:
            closure_error = math.hypot(
                table_translation[0] - info_translation[0],
                table_translation[1] - info_translation[1],
            )
            if closure_error > 1e-3:
                raise ValueError(
                    f"info/table ego pose does not close for {token}: {closure_error:.6g} m"
                )
        return table_translation, "nuscenes_tables.ego_pose"
    if info_translation is not None:
        return info_translation, "embedded_info_smoke_pose"
    raise ValueError("sample has no ego pose in info and nuScenes tables were not initialized")


def radar_points_for_sample(
    *,
    info: dict[str, Any],
    radar_args: SimpleNamespace,
    allow_embedded_smoke: bool,
) -> tuple[Any, dict[str, int], str]:
    """Resolve formal radar sweeps from tables, with a synthetic-smoke escape."""

    if allow_embedded_smoke and isinstance(info.get("radars"), dict):
        points, mapping = _radar_points_global_for_info(info, radar_args)
        return points, mapping, "embedded_info_smoke"
    table_radars = _radars_from_nuscenes_tables(info, radar_args)
    resolved_info = dict(info)
    resolved_info["radars"] = table_radars
    points, mapping = _radar_points_global_for_info(resolved_info, radar_args)
    return points, mapping, "nuscenes_tables"


def _nonradar_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in features.items()
        if key != "ego_range_xy" and not key.startswith("radar_")
    }


def enrich_cache_row(
    *,
    row: dict[str, Any],
    radar_points: Any,
    radar_mapping: dict[str, int],
    ego_xy: tuple[float, float],
    radar_args: SimpleNamespace,
) -> dict[str, Any]:
    """Return a copy with only radar fields and ego-relative range replaced."""

    if row.get("schema_version") != CACHE_ROW_SCHEMA:
        raise ValueError(f"unsupported cache row schema: {row.get('schema_version')}")
    output_box = row.get("output_box")
    if not isinstance(output_box, dict):
        raise ValueError("cache row has no output_box")
    old_features = row.get("features")
    if not isinstance(old_features, dict):
        raise ValueError("cache row has no features dict")
    features = dict(old_features)
    radar_features = _radar_evidence_features(
        fused_row=output_box,
        radar_points=radar_points,
        radar_mapping=radar_mapping,
        args=radar_args,
    )
    for key, value in radar_features.items():
        if not key.startswith("radar_"):
            raise RuntimeError(f"unexpected rebuilt radar key: {key}")
        features[key] = value
    translation = output_box.get("translation")
    if not isinstance(translation, list) or len(translation) < 2:
        raise ValueError("cache output_box has no XY translation")
    features["ego_range_xy"] = math.hypot(
        _safe_float(translation[0]) - ego_xy[0],
        _safe_float(translation[1]) - ego_xy[1],
    )
    if _nonradar_features(features) != _nonradar_features(old_features):
        raise RuntimeError("non-radar feature drift while enriching cache row")
    enriched = dict(row)
    enriched["features"] = features
    return enriched


def cache_cluster_identity_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_token": row.get("sample_token"),
        "cluster_id": row.get("cluster_id"),
        "class_name": row.get("class_name"),
        "sources": row.get("sources"),
        "groups": row.get("groups"),
        "source_signature": row.get("source_signature"),
        "group_signature": row.get("group_signature"),
        "base_score": row.get("base_score"),
        "output_box": row.get("output_box"),
        "cluster_boxes": row.get("cluster_boxes"),
    }


def cache_has_primary_box(row: dict[str, Any]) -> bool:
    boxes = row.get("cluster_boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("cache row lacks locked cluster_boxes")
    return any(
        isinstance(box, dict) and str(box.get("mode")) != "candidate_only"
        for box in boxes
    )


def learned_box_for_row(
    *,
    row: dict[str, Any],
    row_idx: int,
    model: dict[str, Any],
    calibration_table: dict[str, Any] | None,
) -> tuple[tuple[int, dict[str, Any]], dict[str, Any]]:
    """Apply the locked blend by delegating to the established row function."""

    (
        sample_token,
        item,
        base_score,
        raw_arbiter,
        calibrated_arbiter,
        score,
        cap_hit,
        calibration_source,
    ) = score_row(
        row=row,
        row_idx=row_idx,
        model=model,
        calibration_table=calibration_table,
        mode="blend",
        blend=LOCKED_SCORE_BLEND,
        score_temperature=1.0,
        score_power=1.0,
        score_cap=1.0,
        args=SimpleNamespace(),
    )
    if sample_token != str(row.get("sample_token", "")):
        raise RuntimeError("learned scorer changed sample token")
    return item, {
        "base_score": base_score,
        "raw_arbiter_score": raw_arbiter,
        "calibrated_arbiter_score": calibrated_arbiter,
        "output_score": score,
        "cap_hit": cap_hit,
        "calibration_source": calibration_source,
    }


@dataclass
class CacheScan:
    byte_digest: Any = field(default_factory=hashlib.sha256)
    cluster_digest: Any = field(default_factory=hashlib.sha256)
    processed_bytes: int = 0
    processed_rows: int = 0
    processed_samples: int = 0
    reached_eof: bool = False


def iter_grouped_cache(
    path: Path, *, max_samples: int, scan: CacheScan
) -> Iterator[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    current_token: str | None = None
    current_rows: list[tuple[int, dict[str, Any]]] = []
    closed_tokens: set[str] = set()
    accepted_row_idx = 0
    with path.open("rb") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                scan.byte_digest.update(raw_line)
                scan.processed_bytes += len(raw_line)
                continue
            row = json.loads(stripped)
            token = str(row.get("sample_token", ""))
            if not token:
                raise ValueError("cache row has an empty sample_token")
            if current_token is None:
                if max_samples and scan.processed_samples >= max_samples:
                    return
                current_token = token
            elif token != current_token:
                closed_tokens.add(current_token)
                yield current_token, current_rows
                scan.processed_samples += 1
                current_rows = []
                if max_samples and scan.processed_samples >= max_samples:
                    return
                if token in closed_tokens:
                    raise ValueError(f"cache rows are not grouped; token reappeared: {token}")
                current_token = token
            scan.byte_digest.update(raw_line)
            scan.processed_bytes += len(raw_line)
            scan.cluster_digest.update(canonical_json_bytes(cache_cluster_identity_payload(row)))
            scan.cluster_digest.update(b"\n")
            current_rows.append((accepted_row_idx, row))
            accepted_row_idx += 1
            scan.processed_rows += 1
        if current_token is not None:
            yield current_token, current_rows
            scan.processed_samples += 1
        scan.reached_eof = True


def _zero_radar_cache_contract(summary_path: Path, cache_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "bge_af_arbiter_cache_summary_v1":
        raise SystemExit(f"unsupported cache summary schema: {payload.get('schema_version')}")
    fields = ("radar_sample_count", "radar_point_count", "radar_rows_with_points")
    values = {key: int(payload.get(key, -1)) for key in fields}
    if any(value != 0 for value in values.values()):
        raise SystemExit(f"source cache is not the locked zero-radar cache: {values}")
    declared_out = payload.get("out")
    if declared_out and Path(str(declared_out)).resolve() != cache_path.resolve():
        raise SystemExit(
            f"cache summary points to a different cache: {declared_out} != {cache_path}"
        )
    return payload, {
        "validated": True,
        "meaning": "no source-cache row had real radar support; placeholder radar defaults are replaced",
        "declared_counts": values,
        "cache_summary": file_identity(summary_path),
    }


def _validate_expert_lock(
    cache_summary: dict[str, Any], specs: list[NaiveExpertSpec]
) -> None:
    declared = cache_summary.get("experts")
    if not isinstance(declared, list) or len(declared) != 4:
        raise SystemExit("cache summary must declare exactly four experts")
    by_name = {str(item.get("name")): item for item in declared if isinstance(item, dict)}
    if set(by_name) != {spec.name for spec in specs}:
        raise SystemExit(
            "naive experts differ from cache expert pool: "
            f"cache={sorted(by_name)} cli={sorted(spec.name for spec in specs)}"
        )
    declared_order = [
        str(item.get("name")) for item in declared if isinstance(item, dict)
    ]
    cli_order = [spec.name for spec in specs]
    if declared_order != cli_order:
        raise SystemExit(
            "naive expert fixed ordinal/order differs from cache manifest: "
            f"cache={declared_order} cli={cli_order}"
        )
    for spec in specs:
        declared_path = Path(str(by_name[spec.name].get("path", "")))
        if declared_path.resolve() != spec.path.resolve():
            raise SystemExit(
                f"expert path drift for {spec.name}: {declared_path} != {spec.path}"
            )


def _validate_models(radar_model: dict[str, Any], no_radar_model: dict[str, Any]) -> None:
    radar_names = {str(value) for value in radar_model.get("numeric_features", [])}
    no_radar_names = {str(value) for value in no_radar_model.get("numeric_features", [])}
    if not any(name.startswith("radar_") for name in radar_names):
        raise SystemExit("--radar-model has no radar_* numeric feature")
    leaked = sorted(name for name in no_radar_names if name.startswith("radar_"))
    if leaked:
        raise SystemExit(f"--no-radar-model contains radar features: {leaked}")


def _table_identities(nuscenes_root: Path, version: str) -> dict[str, Any]:
    table_root = nuscenes_root / version
    return {
        name: maybe_file_identity(table_root / f"{name}.json")
        for name in ("sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor")
    }


def _load_info_payload(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    infos = _load_infos_by_token(path)
    # Re-open only to recover the authoritative list order.  The full payload is
    # small relative to the cache and this avoids depending on dict sorting.
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        items = payload.get("infos") or payload.get("data_list") or list(payload.values())
    else:
        items = payload
    if not isinstance(items, list):
        raise SystemExit(f"sample info is not list-like: {path}")
    order: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("token") or item.get("sample_token")
        if token is not None:
            order.append(str(token))
    if len(order) != len(set(order)) or set(order) != set(infos):
        raise SystemExit("sample info token order/identity is malformed")
    return infos, order


def _iter_pretty_table_objects(path: Path) -> Iterator[bytes]:
    """Yield objects from the locked pretty-printed nuScenes table format.

    The official JSON tables are top-level arrays with one object at a time and
    no nested JSON objects.  This line-streamed reader keeps only one record in
    memory.  It deliberately rejects an unexpected compact/nested-object
    layout instead of silently falling back to loading a multi-gigabyte table.
    """

    current: list[bytes] | None = None
    depth = 0
    in_string = False
    structural = re.compile(rb'\\.|["{}]')
    with path.open("rb") as fh:
        first = fh.readline().strip()
        if first != b"[":
            raise ValueError(f"unsupported nuScenes table layout (expected '['): {path}")
        for line in fh:
            stripped = line.lstrip()
            if current is None:
                if stripped.startswith(b"]"):
                    return
                if not stripped.startswith(b"{"):
                    if stripped.strip():
                        raise ValueError(f"unexpected nuScenes table record start: {path}")
                    continue
                current = [line]
            else:
                current.append(line)
            for match in structural.finditer(line):
                token = match.group(0)
                if token.startswith(b"\\"):
                    continue
                if token == b'"':
                    in_string = not in_string
                elif not in_string and token == b"{":
                    depth += 1
                elif not in_string and token == b"}":
                    depth -= 1
                    if depth < 0:
                        raise ValueError(f"unbalanced nuScenes table object: {path}")
            if current is not None and depth == 0:
                if in_string:
                    raise ValueError(f"unterminated string in nuScenes table: {path}")
                yield b"".join(current).rstrip().rstrip(b",")
                current = None
    if current is not None or depth != 0 or in_string:
        raise ValueError(f"truncated nuScenes table: {path}")


def _load_filtered_table(
    path: Path,
    *,
    field_re: re.Pattern[bytes],
    wanted: set[str],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    wanted_bytes = {value.encode("ascii") for value in wanted}
    for raw in _iter_pretty_table_objects(path):
        match = field_re.search(raw)
        if match is None or match.group(1) not in wanted_bytes:
            continue
        row = json.loads(raw)
        token = str(row.get("token", ""))
        if token in selected:
            raise ValueError(f"duplicate nuScenes token in {path}: {token}")
        selected[token] = row
    return selected


class MinimalNuScenesTables:
    """RAM-bounded read-only subset implementing the ``NuScenes.get`` API."""

    REQUIRED_CHANNELS = {
        "LIDAR_TOP",
        "RADAR_FRONT",
        "RADAR_FRONT_LEFT",
        "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT",
        "RADAR_BACK_RIGHT",
    }

    def __init__(self, *, root: Path, version: str, sample_tokens: set[str]):
        table_root = root / version
        sample_path = table_root / "sample.json"
        sample_data_path = table_root / "sample_data.json"
        calib_path = table_root / "calibrated_sensor.json"
        ego_path = table_root / "ego_pose.json"
        sensor_path = table_root / "sensor.json"

        samples = _load_filtered_table(
            sample_path, field_re=TOKEN_FIELD_RE, wanted=sample_tokens
        )
        if set(samples) != sample_tokens:
            raise ValueError(
                "nuScenes sample table coverage drift: "
                f"missing={sorted(sample_tokens - set(samples))[:5]}"
            )

        # sample_data is the only very large table needed here.  Select key
        # frames belonging to validation samples before decoding downstream
        # calibration/pose rows.
        sample_data: dict[str, dict[str, Any]] = {}
        wanted_sample_bytes = {value.encode("ascii") for value in sample_tokens}
        for raw in _iter_pretty_table_objects(sample_data_path):
            match = SAMPLE_TOKEN_FIELD_RE.search(raw)
            if match is None or match.group(1) not in wanted_sample_bytes:
                continue
            row = json.loads(raw)
            if not bool(row.get("is_key_frame")):
                continue
            token = str(row.get("token", ""))
            if not token or token in sample_data:
                raise ValueError(f"malformed/duplicate selected sample_data token: {token}")
            sample_data[token] = row

        calib_tokens = {
            str(row.get("calibrated_sensor_token", "")) for row in sample_data.values()
        }
        pose_tokens = {str(row.get("ego_pose_token", "")) for row in sample_data.values()}
        calibrated = _load_filtered_table(
            calib_path, field_re=TOKEN_FIELD_RE, wanted=calib_tokens
        )
        poses = _load_filtered_table(ego_path, field_re=TOKEN_FIELD_RE, wanted=pose_tokens)
        sensor_tokens = {
            str(row.get("sensor_token", "")) for row in calibrated.values()
        }
        sensors = _load_filtered_table(
            sensor_path, field_re=TOKEN_FIELD_RE, wanted=sensor_tokens
        )
        if set(calibrated) != calib_tokens or set(poses) != pose_tokens or set(sensors) != sensor_tokens:
            raise ValueError("nuScenes selected table foreign-key closure failed")

        for row in sample_data.values():
            calib = calibrated[str(row["calibrated_sensor_token"])]
            sensor = sensors[str(calib["sensor_token"])]
            channel = str(sensor.get("channel", ""))
            if channel not in self.REQUIRED_CHANNELS:
                continue
            sample_token = str(row.get("sample_token", ""))
            data = samples[sample_token].setdefault("data", {})
            if channel in data:
                raise ValueError(
                    f"multiple key-frame sample_data rows for {sample_token}/{channel}"
                )
            data[channel] = str(row["token"])
        for token, sample in samples.items():
            missing = self.REQUIRED_CHANNELS - set(sample.get("data", {}))
            if missing:
                raise ValueError(f"nuScenes sample {token} lacks channels: {sorted(missing)}")

        self.tables = {
            "sample": samples,
            "sample_data": sample_data,
            "calibrated_sensor": calibrated,
            "ego_pose": poses,
            "sensor": sensors,
        }

    def get(self, table: str, token: str) -> dict[str, Any]:
        try:
            return self.tables[table][str(token)]
        except KeyError as exc:
            raise KeyError(f"minimal nuScenes table miss: {table}/{token}") from exc


def _estimated_output_bytes(cache_size: int, sample_count: int, max_samples: int) -> int:
    fraction = 1.0
    if max_samples and sample_count:
        fraction = min(1.0, max_samples / sample_count)
    # The promoted cache includes feature and cluster telemetry and is much
    # larger than a result JSON.  Ten percent per arm is a conservative bound
    # observed for this schema; add 1 GiB working/free-space margin.
    working_margin = 16 * 1024**2 if max_samples else 1024**3
    return int(cache_size * 0.10 * len(OUTPUT_FILENAMES) * fraction + working_margin)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-jsonl", required=True, type=Path)
    parser.add_argument("--cache-summary-json", required=True, type=Path)
    parser.add_argument("--execution-config-json", required=True, type=Path)
    parser.add_argument("--expected-execution-config-sha256", required=True)
    parser.add_argument("--sample-info-pkl", required=True, type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument(
        "--naive-expert",
        action="append",
        type=parse_naive_expert,
        required=True,
        help="Locked expert as NAME=artifacts/expert_results/result.json; repeat exactly four times.",
    )
    parser.add_argument("--radar-model", required=True, type=Path)
    parser.add_argument("--radar-calibration", required=True, type=Path)
    parser.add_argument("--no-radar-model", required=True, type=Path)
    parser.add_argument("--no-radar-calibration", required=True, type=Path)
    parser.add_argument("--meta-from-result-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--storage-mount", type=Path, default=Path("outputs"))
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--mount-recheck-every", type=int, default=50)
    parser.add_argument(
        "--allow-non-mount-smoke",
        action="store_true",
        help="Tests only: allow a non-mounted output root when --max-samples is bounded.",
    )
    args = parser.parse_args(argv)
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")
    for path_name in (
        "cache_jsonl",
        "cache_summary_json",
        "execution_config_json",
        "sample_info_pkl",
        "radar_model",
        "radar_calibration",
        "no_radar_model",
        "no_radar_calibration",
    ):
        path = getattr(args, path_name)
        if not path.is_file():
            raise SystemExit(f"missing --{path_name.replace('_', '-')}: {path}")
    if not args.nuscenes_root.is_dir():
        raise SystemExit(f"missing --nuscenes-root: {args.nuscenes_root}")
    if not args.meta_from_result_json.is_file():
        raise SystemExit(f"missing --meta-from-result-json: {args.meta_from_result_json}")
    if len(args.expected_execution_config_sha256) != 64 or any(
        char not in "0123456789abcdef"
        for char in args.expected_execution_config_sha256
    ):
        raise SystemExit("--expected-execution-config-sha256 must be 64 lowercase hex chars")
    if len(args.naive_expert) != 4:
        raise SystemExit(f"exactly four --naive-expert inputs are required, got {len(args.naive_expert)}")
    names = [name for name, _ in args.naive_expert]
    if len(names) != len(set(names)):
        raise SystemExit("--naive-expert names must be unique")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing --out-dir: {args.out_dir}")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution_config, execution_lock = _load_and_validate_execution_config(args)
    cache_summary, zero_radar_contract = _zero_radar_cache_contract(
        args.cache_summary_json, args.cache_jsonl
    )
    expert_specs = [
        NaiveExpertSpec(name=name, path=path, ordinal=index)
        for index, (name, path) in enumerate(args.naive_expert)
    ]
    _validate_expert_lock(cache_summary, expert_specs)

    infos, info_order = _load_info_payload(args.sample_info_pkl)
    declared_samples = int(cache_summary.get("sample_count", 0))
    if declared_samples != len(infos):
        raise SystemExit(
            f"cache/info sample-count drift: cache={declared_samples} info={len(infos)}"
        )

    cache_stat = args.cache_jsonl.stat()
    storage = StorageGuard(
        mount=args.storage_mount,
        out_dir=args.out_dir,
        allow_non_mount_smoke=args.allow_non_mount_smoke,
        max_samples=args.max_samples,
        min_free_gib=args.min_free_gib,
        recheck_every=args.mount_recheck_every,
    )
    storage_summary = storage.preflight(
        estimated_output_bytes=_estimated_output_bytes(
            cache_stat.st_size, len(infos), args.max_samples
        )
    )

    radar_model = load_model(args.radar_model, "blend")
    no_radar_model = load_model(args.no_radar_model, "blend")
    assert radar_model is not None and no_radar_model is not None
    _validate_models(radar_model, no_radar_model)
    radar_calibration = load_calibration_table(args.radar_calibration)
    no_radar_calibration = load_calibration_table(args.no_radar_calibration)
    assert radar_calibration is not None and no_radar_calibration is not None

    meta_path = args.meta_from_result_json
    meta = load_meta(meta_path)
    if not isinstance(meta, dict):
        raise SystemExit(f"could not load nuScenes meta from {meta_path}")

    storage.recheck()
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    storage.recheck()
    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.out_dir.name}.tmp.", dir=args.out_dir.parent)
    )
    writers: dict[str, ResultJsonWriter] = {}
    scan = CacheScan()
    committed = False
    try:
        storage.recheck()
        if not args.allow_non_mount_smoke:
            if not is_relative_to(stage.resolve(), storage.mount):
                raise SystemExit(f"staging directory escaped storage mount: {stage}")
            if stage.stat().st_dev != storage.initial_device:
                raise SystemExit(f"staging directory is on wrong device: {stage}")
        for stage_name, filename in OUTPUT_FILENAMES.items():
            writers[stage_name] = ResultJsonWriter(stage / filename, meta)

        naive_totals = {
            "input_box_counts": Counter(),
            "cluster_count": 0,
            "support_histogram": Counter(),
            "truncated_sample_count": 0,
        }
        stage_candidate_counts: Counter[str] = Counter()
        stage_truncated_samples: Counter[str] = Counter()
        calibration_sources: dict[str, Counter[str]] = {
            "learned_radar_uncalibrated": Counter(),
            "learned_radar_calibrated": Counter(),
            "learned_no_radar_calibrated": Counter(),
        }
        observed_cache_positive_radar_rows = 0
        observed_cache_ego_range_rows = 0
        enriched_rows_with_radar = 0
        enriched_radar_point_sum_over_rows = 0
        radar_sample_count = 0
        radar_point_count = 0
        ego_pose_sources: Counter[str] = Counter()
        radar_sweep_sources: Counter[str] = Counter()
        nonradar_feature_check_count = 0
        sample_tokens: list[str] = []

        radar_args = SimpleNamespace(
            radar_from_nuscenes_tables=True,
            nuscenes_root=args.nuscenes_root,
            nuscenes_version=args.nuscenes_version,
            radar_channels="RADAR_FRONT,RADAR_FRONT_LEFT,RADAR_FRONT_RIGHT,RADAR_BACK_LEFT,RADAR_BACK_RIGHT",
            radar_max_sweeps=LOCKED_RADAR_MAX_SWEEPS,
            radar_box_margin=LOCKED_RADAR_BOX_MARGIN,
            radar_search_radius=LOCKED_RADAR_SEARCH_RADIUS,
        )
        if not args.allow_non_mount_smoke:
            radar_args._rgate_nuscenes = MinimalNuScenesTables(
                root=args.nuscenes_root,
                version=args.nuscenes_version,
                sample_tokens=set(infos),
            )

        with ExitStack() as stack:
            expert_indexes: list[tuple[NaiveExpertSpec, ResultJsonIndex]] = []
            info_tokens = set(infos)
            for spec in expert_specs:
                index = stack.enter_context(ResultJsonIndex(spec.path))
                if index.tokens() != info_tokens:
                    missing = sorted(info_tokens - index.tokens())[:5]
                    extra = sorted(index.tokens() - info_tokens)[:5]
                    raise SystemExit(
                        f"raw expert coverage drift for {spec.name}: "
                        f"missing={missing} extra={extra}"
                    )
                expert_indexes.append((spec, index))

            for sample_number, (sample_token, indexed_rows) in enumerate(
                iter_grouped_cache(
                    args.cache_jsonl, max_samples=args.max_samples, scan=scan
                ),
                start=1,
            ):
                if sample_token not in infos:
                    raise ValueError(f"cache token missing from sample infos: {sample_token}")
                sample_tokens.append(sample_token)
                info = infos[sample_token]
                radar_points, radar_mapping, sweep_source = radar_points_for_sample(
                    info=info,
                    radar_args=radar_args,
                    allow_embedded_smoke=bool(args.allow_non_mount_smoke),
                )
                ego_xy, pose_source = ego_xy_for_sample(info, radar_args)
                ego_pose_sources[pose_source] += 1
                radar_sweep_sources[sweep_source] += 1
                point_count = int(radar_points.shape[0])
                radar_sample_count += int(point_count > 0)
                radar_point_count += point_count

                naive_rows, naive_sample_summary = naive_equal_wbf_sample(
                    sample_token=sample_token,
                    rows_by_expert=[
                        (spec, index.rows(sample_token))
                        for spec, index in expert_indexes
                    ],
                )
                writers["naive_equal_wbf"].write_sample(sample_token, naive_rows)
                for name, count in naive_sample_summary["input_box_counts"].items():
                    naive_totals["input_box_counts"][name] += int(count)
                naive_totals["cluster_count"] += int(naive_sample_summary["cluster_count"])
                for support, count in naive_sample_summary["support_histogram"].items():
                    naive_totals["support_histogram"][int(support)] += int(count)
                naive_totals["truncated_sample_count"] += int(naive_sample_summary["truncated"])

                per_stage: dict[str, list[tuple[int, dict[str, Any]]]] = {
                    "rgate_rescore_only": [],
                    "learned_radar_uncalibrated": [],
                    "learned_radar_calibrated": [],
                    "learned_no_radar_calibrated": [],
                }
                for row_idx, row in indexed_rows:
                    old_features = row.get("features")
                    if not isinstance(old_features, dict):
                        raise ValueError("cache row has no features dict")
                    observed_cache_positive_radar_rows += int(
                        _safe_float(old_features.get("radar_point_count")) > 0
                    )
                    observed_cache_ego_range_rows += int("ego_range_xy" in old_features)
                    enriched = enrich_cache_row(
                        row=row,
                        radar_points=radar_points,
                        radar_mapping=radar_mapping,
                        ego_xy=ego_xy,
                        radar_args=radar_args,
                    )
                    nonradar_feature_check_count += 1
                    new_point_count = int(
                        _safe_float(enriched["features"].get("radar_point_count"))
                    )
                    enriched_rows_with_radar += int(new_point_count > 0)
                    enriched_radar_point_sum_over_rows += new_point_count

                    full_mode_count = _safe_float(
                        old_features.get("full_mode_count"), float("nan")
                    )
                    if not math.isfinite(full_mode_count):
                        raise ValueError("cache row lacks finite features.full_mode_count")
                    cluster_full_count = sum(
                        1
                        for box in row.get("cluster_boxes", [])
                        if isinstance(box, dict) and str(box.get("mode")) == "full"
                    )
                    if abs(full_mode_count - cluster_full_count) > 1e-9:
                        raise RuntimeError(
                            "features.full_mode_count does not close against cluster_boxes"
                        )
                    if full_mode_count > 0:
                        base_box = dict(row["output_box"])
                        base_box["sample_token"] = sample_token
                        base_box["detection_score"] = max(
                            0.0,
                            min(
                                1.0,
                                _safe_float(
                                    row.get("base_score", base_box.get("detection_score"))
                                ),
                            ),
                        )
                        per_stage["rgate_rescore_only"].append((row_idx, base_box))

                    learned_specs = (
                        ("learned_radar_uncalibrated", radar_model, None),
                        ("learned_radar_calibrated", radar_model, radar_calibration),
                        (
                            "learned_no_radar_calibrated",
                            no_radar_model,
                            no_radar_calibration,
                        ),
                    )
                    for stage_name, model, calibration in learned_specs:
                        item, telemetry = learned_box_for_row(
                            row=enriched,
                            row_idx=row_idx,
                            model=model,
                            calibration_table=calibration,
                        )
                        per_stage[stage_name].append(item)
                        calibration_sources[stage_name][
                            str(telemetry["calibration_source"])
                        ] += 1

                    # The calibrated key must be based on the newly reconstructed
                    # ego-relative range, never the legacy global ``range_xy``.
                    for table in (radar_calibration, no_radar_calibration):
                        grouping = table.get("grouping", {})
                        key = row_calibration_key(
                            enriched,
                            grouping.get("range_edges", [0.0, 20.0, 40.0, 60.0]),
                        )
                        expected_key = row_calibration_key(
                            {
                                **enriched,
                                "features": {
                                    **enriched["features"],
                                    "range_xy": 1e30,
                                    "mean_input_range_xy": 1e30,
                                },
                            },
                            grouping.get("range_edges", [0.0, 20.0, 40.0, 60.0]),
                        )
                        if key != expected_key:
                            raise RuntimeError("calibration key did not prioritize ego_range_xy")

                for stage_name, token_rows in per_stage.items():
                    stage_candidate_counts[stage_name] += len(token_rows)
                    selected = sorted_sample_rows(token_rows, LOCKED_MAX_PER_SAMPLE)
                    stage_truncated_samples[stage_name] += int(
                        len(token_rows) > LOCKED_MAX_PER_SAMPLE
                    )
                    writers[stage_name].write_sample(sample_token, selected)

                if sample_number % storage.recheck_every == 0:
                    storage.recheck()

        if observed_cache_positive_radar_rows:
            raise RuntimeError(
                "source cache contradicts its zero-radar summary: "
                f"positive rows={observed_cache_positive_radar_rows}"
            )
        if observed_cache_ego_range_rows:
            raise RuntimeError(
                "source cache unexpectedly already contains ego_range_xy: "
                f"rows={observed_cache_ego_range_rows}"
            )
        if scan.reached_eof:
            if scan.processed_rows != int(cache_summary.get("row_count", -1)):
                raise RuntimeError(
                    "cache row-count drift: "
                    f"observed={scan.processed_rows} declared={cache_summary.get('row_count')}"
                )
            if set(sample_tokens) != set(infos):
                missing = sorted(set(infos) - set(sample_tokens))[:5]
                extra = sorted(set(sample_tokens) - set(infos))[:5]
                raise RuntimeError(
                    f"cache/info coverage drift: missing={missing} extra={extra}"
                )

        storage.recheck()
        output_identities = {
            name: writer.close() for name, writer in writers.items()
        }
        output_identities["naive_equal_wbf"]["candidate_box_count"] = int(
            naive_totals["cluster_count"]
        )
        output_identities["naive_equal_wbf"]["truncated_sample_count"] = int(
            naive_totals["truncated_sample_count"]
        )
        for stage_name in CACHE_STAGE_NAMES:
            output_identities[stage_name]["candidate_box_count"] = int(
                stage_candidate_counts[stage_name]
            )
            output_identities[stage_name]["truncated_sample_count"] = int(
                stage_truncated_samples[stage_name]
            )

        script_dir = Path(__file__).resolve().parent
        input_identities = {
            "cache_jsonl": {
                "path": str(args.cache_jsonl.resolve()),
                "size_bytes": cache_stat.st_size,
                "mtime_ns": cache_stat.st_mtime_ns,
                "sha256": scan.byte_digest.hexdigest() if scan.reached_eof else None,
                "processed_prefix_sha256": scan.byte_digest.hexdigest(),
                "processed_prefix_bytes": scan.processed_bytes,
                "identity_scope": "full_file" if scan.reached_eof else "processed_prefix",
            },
            "sample_info_pkl": file_identity(args.sample_info_pkl),
            "radar_model": file_identity(args.radar_model),
            "radar_calibration": file_identity(args.radar_calibration),
            "no_radar_model": file_identity(args.no_radar_model),
            "no_radar_calibration": file_identity(args.no_radar_calibration),
            "meta_from_result_json": file_identity(meta_path),
            "naive_experts": {
                spec.name: {
                    **file_identity(spec.path),
                    "fixed_ordinal": spec.ordinal,
                }
                for spec in expert_specs
            },
            "implementation": {
                "materializer": file_identity(Path(__file__).resolve()),
                "apply_bge_af_arbiter": file_identity(
                    script_dir / "apply_bge_af_arbiter.py"
                ),
                "build_bge_af_arbiter_cache": file_identity(
                    script_dir / "build_bge_af_arbiter_cache.py"
                ),
                "fuse_nuscenes_expert_results": file_identity(
                    script_dir / "fuse_nuscenes_expert_results.py"
                ),
            },
            "nuscenes": {
                "root": str(args.nuscenes_root.resolve()),
                "version": args.nuscenes_version,
                "tables": _table_identities(args.nuscenes_root, args.nuscenes_version),
            },
        }

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "materialized_not_evaluated",
            "scope": "full" if args.max_samples == 0 else "bounded_smoke",
            "evaluation_performed": False,
            "training_performed": False,
            "atomic_directory_commit": True,
            "overwrite_allowed": False,
            "output_dir": str(args.out_dir.resolve()),
            "execution_lock": execution_lock,
            "locked_protocol": {
                "score_blend": LOCKED_SCORE_BLEND,
                "max_per_sample": LOCKED_MAX_PER_SAMPLE,
                "radar_max_sweeps": LOCKED_RADAR_MAX_SWEEPS,
                "radar_box_margin": LOCKED_RADAR_BOX_MARGIN,
                "radar_search_radius": LOCKED_RADAR_SEARCH_RADIUS,
                "naive": {
                    "candidate_source": "four original result JSONs (not R-GATE cache clusters)",
                    "expert_weighting": "equal",
                    "box_mode": "all full",
                    "cluster_order": "raw score desc, source fixed ordinal, input row ordinal",
                    "cluster_constraint": "at most one box per source; try nearest legal cluster then create",
                    "cluster_center": "unweighted mean XY of current members",
                    "class_radii_m": DEFAULT_CLUSTER_THRESHOLDS,
                    "score": "arithmetic mean of the present sources' raw scores",
                    "geometry": "unweighted mean translation/size/velocity and circular-mean yaw",
                },
                "rgate_rescore_only": {
                    "cluster_source": "cache output_box/base_score",
                    "retrieval": "exclude clusters whose boxes are all candidate_only",
                },
                "learned_radar_uncalibrated": {
                    "model": "radar",
                    "calibration_table": None,
                    "temperature": 1.0,
                    "power": 1.0,
                    "cap": 1.0,
                    "score_blend": LOCKED_SCORE_BLEND,
                },
                "learned_radar_calibrated": {
                    "model": "radar",
                    "calibration_table": "locked radar calibration",
                    "range_key": "rebuilt ego_range_xy",
                    "score_blend": LOCKED_SCORE_BLEND,
                },
                "learned_no_radar_calibrated": {
                    "model": "no-radar",
                    "calibration_table": "locked no-radar calibration",
                    "range_key": "rebuilt ego_range_xy",
                    "score_blend": LOCKED_SCORE_BLEND,
                },
            },
            "source_cache_zero_radar_contract": {
                **zero_radar_contract,
                "observed_positive_radar_rows": observed_cache_positive_radar_rows,
                "observed_preexisting_ego_range_rows": observed_cache_ego_range_rows,
            },
            "read_only_cache_invariants": {
                "source_cache_written": False,
                "cluster_boxes_unchanged": True,
                "output_box_geometry_unchanged_for_rgate_and_learned_stages": True,
                "nonradar_features_unchanged": True,
                "nonradar_feature_rows_checked": nonradar_feature_check_count,
                "only_in_memory_replacements": ["features.ego_range_xy", "features.radar_*"],
                "processed_cluster_identity_sha256": scan.cluster_digest.hexdigest(),
            },
            "coverage": {
                "declared_full_sample_count": len(info_order),
                "processed_sample_count": len(sample_tokens),
                "processed_row_count": scan.processed_rows,
                "cache_reached_eof": scan.reached_eof,
                "max_samples": args.max_samples,
                "sample_token_sha256": hashlib.sha256(
                    ("\n".join(sample_tokens) + ("\n" if sample_tokens else "")).encode("utf-8")
                ).hexdigest(),
                "raw_expert_full_coverage_validated": True,
            },
            "rebuilt_context": {
                "nuscenes_table_backend": (
                    "minimal_read_only_streamed_subset"
                    if not args.allow_non_mount_smoke
                    else "embedded_info_smoke"
                ),
                "ego_pose_sources": dict(sorted(ego_pose_sources.items())),
                "radar_sweep_sources": dict(sorted(radar_sweep_sources.items())),
                "formal_sweep_source": "nuscenes_tables",
                "info_table_pose_closure_tolerance_m": 1e-3,
                "radar_sample_count": radar_sample_count,
                "radar_point_count": radar_point_count,
                "rows_with_positive_radar_point_count": enriched_rows_with_radar,
                "radar_point_sum_over_rows": enriched_radar_point_sum_over_rows,
                "calibration_range_key_priority": "ego_range_xy before legacy global range_xy",
            },
            "naive_telemetry": {
                "input_box_counts": dict(sorted(naive_totals["input_box_counts"].items())),
                "cluster_count": int(naive_totals["cluster_count"]),
                "support_histogram": {
                    str(key): value
                    for key, value in sorted(naive_totals["support_histogram"].items())
                },
                "truncated_sample_count": int(naive_totals["truncated_sample_count"]),
            },
            "calibration_match_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in calibration_sources.items()
            },
            "outputs": output_identities,
            "input_identities": input_identities,
            "storage_guard": storage_summary,
            "notes": [
                "No official nuScenes evaluation is run by this materializer.",
                "All five arms use the same 500-box/sample cap.",
                "The fifth arm repairs the no-radar paired control's historical global-range calibration bug.",
            ],
        }
        summary_path = stage / "materialization_summary.json"
        summary_path.write_bytes(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        with summary_path.open("rb") as fh:
            os.fsync(fh.fileno())
        storage.recheck()
        if args.out_dir.exists():
            raise RuntimeError(f"output directory appeared during run: {args.out_dir}")
        stage.rename(args.out_dir)
        committed = True
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return summary
    finally:
        for writer in writers.values():
            writer.abort()
        if not committed and stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
