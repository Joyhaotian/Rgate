#!/usr/bin/env python3
"""Build a per-cluster feature cache for BGE-AF evidence arbitration.

The cache is the bridge between rule-based result-level expert fusion and the
V1 BGE-AF learned arbiter.  It reuses the same expert argument format as
``fuse_nuscenes_expert_results.py`` and emits one JSONL row per fused cluster.

Optional ``--target-result-json`` attaches a soft distillation target by matching
each fused cluster to the closest same-class target box.  ``--target-from-info-gt``
instead uses ``gt_boxes`` / ``gt_names`` from ``--sample-info-pkl`` and transforms
GT centers from LiDAR frame to global frame before matching.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import math
import mmap
import pickle
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional radar-evidence dependency.
    np = None  # type: ignore

from check_nuscenes_result_json_scope_stream import find_value_end, parse_json_string, skip_ws
from fuse_nuscenes_expert_results import (
    DEFAULT_CLUSTER_THRESHOLDS,
    Box,
    center_distance,
    cluster_boxes,
    fuse_cluster,
    load_boxes,
    parse_class_threshold,
    parse_expert,
    yaw_from_quat,
)


QUALITY_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
SCHEMA_VERSION = "bge_af_arbiter_cache_row_v1"
RADAR_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "dyn_prop",
    "id",
    "rcs",
    "vx",
    "vy",
    "vx_comp",
    "vy_comp",
    "is_quality_valid",
    "ambig_state",
    "x_rms",
    "y_rms",
    "invalid_state",
    "pdh0",
    "vx_rms",
    "vy_rms",
    "time_lag",
]
NUSCENES_RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("--include-radar-evidence requires numpy")
    return np


class ResultJsonIndex:
    """Byte-range index over top-level nuScenes result arrays."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")
        self.mm = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.ranges: dict[str, tuple[int, int]] = {}
        self._index_results()

    def close(self) -> None:
        self.mm.close()
        self.handle.close()

    def __enter__(self) -> "ResultJsonIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _index_results(self) -> None:
        key_idx = self.mm.find(b'"results"')
        if key_idx < 0:
            raise ValueError(f"{self.path} has no top-level-looking results key")
        colon = self.mm.find(b":", key_idx)
        if colon < 0:
            raise ValueError(f"{self.path}: malformed results key")
        idx = skip_ws(self.mm, colon + 1)
        if self.mm[idx] != 0x7B:
            raise ValueError(f"{self.path}: results value is not an object")
        idx += 1
        while True:
            idx = skip_ws(self.mm, idx)
            if self.mm[idx] == 0x7D:
                return
            token, idx = parse_json_string(self.mm, idx)
            idx = skip_ws(self.mm, idx)
            if self.mm[idx] != 0x3A:
                raise ValueError(f"{self.path}: expected ':' after sample token {token!r}")
            idx = skip_ws(self.mm, idx + 1)
            if self.mm[idx] != 0x5B:
                raise ValueError(f"{self.path}: results[{token!r}] is not an array")
            end = find_value_end(self.mm, idx)
            self.ranges[str(token)] = (idx, end)
            idx = skip_ws(self.mm, end)
            if self.mm[idx] == 0x2C:
                idx += 1
                continue
            if self.mm[idx] == 0x7D:
                return
            raise ValueError(f"{self.path}: expected ',' or '}}' after sample {token!r}")

    def tokens(self) -> set[str]:
        return set(self.ranges)

    def rows(self, token: str) -> list[dict[str, Any]]:
        span = self.ranges.get(token)
        if span is None:
            return []
        start, end = span
        rows = json.loads(self.mm[start:end])
        if not isinstance(rows, list):
            raise ValueError(f"{self.path}: results[{token!r}] decoded as non-list")
        return [row for row in rows if isinstance(row, dict)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_result_rows(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path} has no results dict")
    out: dict[str, list[dict[str, Any]]] = {}
    for token, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"{path}: results[{token!r}] is not a list")
        out[str(token)] = [row for row in rows if isinstance(row, dict)]
    return out


def _boxes_from_rows(expert: Any, sample_token: str, rows: list[dict[str, Any]]) -> list[Box]:
    boxes: list[Box] = []
    for row in rows:
        name = str(row["detection_name"])
        score = float(row.get("detection_score", 0.0))
        translation = [float(v) for v in row["translation"]]
        size = [float(v) for v in row["size"]]
        rotation = [float(v) for v in row["rotation"]]
        velocity = [float(v) for v in row.get("velocity", [0.0, 0.0])[:2]]
        while len(velocity) < 2:
            velocity.append(0.0)
        boxes.append(
            Box(
                sample_token=sample_token,
                detection_name=name,
                score=score,
                translation=translation,
                size=size,
                rotation=rotation,
                velocity=velocity,
                attribute_name=str(row.get("attribute_name", "")),
                source=expert.name,
                group=expert.group,
                mode=expert.mode,
                score_weight=expert.score_weight,
                geometry_weight=expert.geometry_weight,
            )
        )
    return boxes


def _load_sample_tokens_from_info(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        if "infos" in payload:
            infos = payload["infos"]
        elif "data_list" in payload:
            infos = payload["data_list"]
        else:
            infos = list(payload.values())
    else:
        infos = payload
    tokens: set[str] = set()
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain a list-like infos payload")
    for item in infos:
        if not isinstance(item, dict):
            continue
        token = item.get("token") or item.get("sample_token")
        if token is not None:
            tokens.add(str(token))
    if not tokens:
        raise ValueError(f"{path} did not yield any sample tokens")
    return tokens


def _load_infos_by_token(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        infos = payload.get("infos") or payload.get("data_list") or list(payload.values())
    else:
        infos = payload
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain a list-like infos payload")
    out: dict[str, dict[str, Any]] = {}
    for item in infos:
        if not isinstance(item, dict):
            continue
        token = item.get("token") or item.get("sample_token")
        if token is not None:
            out[str(token)] = item
    return out


def _ego_xy_from_info(info: dict[str, Any]) -> tuple[float, float] | None:
    translation = info.get("ego2global_translation")
    if isinstance(translation, list) and len(translation) >= 2:
        return _safe_float(translation[0]), _safe_float(translation[1])
    radars = info.get("radars")
    if isinstance(radars, dict):
        for sweeps in radars.values():
            if not isinstance(sweeps, list):
                continue
            for sweep in sweeps:
                if not isinstance(sweep, dict):
                    continue
                translation = sweep.get("ego2global_translation")
                if isinstance(translation, list) and len(translation) >= 2:
                    return _safe_float(translation[0]), _safe_float(translation[1])
    return None


def _load_sample_tokens_from_file(path: Path | None) -> set[str]:
    return set(_load_ordered_sample_tokens_from_file(path))


def _load_ordered_sample_tokens_from_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    tokens = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not tokens:
        raise ValueError(f"{path} did not yield any sample tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{path} contains duplicate sample tokens")
    return tokens


def _ordered_selected_tokens(
    *,
    all_tokens: set[str],
    requested_tokens: set[str],
    ordered_file_tokens: list[str],
    max_samples: int,
) -> list[str]:
    selected = set(all_tokens)
    if requested_tokens:
        selected &= requested_tokens
    if ordered_file_tokens:
        ordered = [token for token in ordered_file_tokens if token in selected]
        ordered.extend(sorted(selected - set(ordered_file_tokens)))
    else:
        ordered = sorted(selected)
    if max_samples:
        ordered = ordered[:max_samples]
    return ordered


def _row_xy(row: dict[str, Any]) -> tuple[float, float]:
    translation = row.get("translation", [0.0, 0.0, 0.0])
    return _safe_float(translation[0]), _safe_float(translation[1])


def _target_quality(
    *,
    fused_row: dict[str, Any],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    class_name = str(fused_row.get("detection_name", ""))
    center = _row_xy(fused_row)
    best: dict[str, Any] | None = None
    best_dist = float("inf")
    for target in target_rows:
        if str(target.get("detection_name", "")) != class_name:
            continue
        dist = center_distance(center, _row_xy(target))
        if dist < best_dist:
            best = target
            best_dist = dist
    if best is None:
        return {
            "quality": None,
            "matched": False,
            "distance": None,
            "score": None,
        }
    proximity = sum(float(best_dist < threshold) for threshold in QUALITY_THRESHOLDS)
    proximity /= len(QUALITY_THRESHOLDS)
    target_score = min(max(_safe_float(best.get("detection_score", 1.0), 1.0), 0.0), 1.0)
    return {
        "quality": float(proximity * target_score),
        "matched": True,
        "distance": float(best_dist),
        "score": float(target_score),
    }


def _quat_rotate_point(q: Any, point: Any) -> list[float]:
    w, x, y, z = [_safe_float(value) for value in list(q)[:4]]
    px, py, pz = [_safe_float(value) for value in list(point)[:3]]
    return [
        (1.0 - 2.0 * (y * y + z * z)) * px + (2.0 * (x * y - z * w)) * py + (2.0 * (x * z + y * w)) * pz,
        (2.0 * (x * y + z * w)) * px + (1.0 - 2.0 * (x * x + z * z)) * py + (2.0 * (y * z - x * w)) * pz,
        (2.0 * (x * z - y * w)) * px + (2.0 * (y * z + x * w)) * py + (1.0 - 2.0 * (x * x + y * y)) * pz,
    ]


def _transform_point_quat(point: Any, rotation: Any, translation: Any) -> list[float]:
    rotated = _quat_rotate_point(rotation, point)
    trans = list(translation)[:3]
    while len(trans) < 3:
        trans.append(0.0)
    return [rotated[idx] + _safe_float(trans[idx]) for idx in range(3)]


def _to_plain_sequence(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _target_rows_from_info_gt(info: dict[str, Any]) -> list[dict[str, Any]]:
    gt_boxes = _to_plain_sequence(info.get("gt_boxes"))
    gt_names = _to_plain_sequence(info.get("gt_names"))
    valid_flags = _to_plain_sequence(info.get("valid_flag"))
    gt_velocity = _to_plain_sequence(info.get("gt_velocity"))
    lidar_rot = info.get("lidar2ego_rotation", [1.0, 0.0, 0.0, 0.0])
    lidar_trans = info.get("lidar2ego_translation", [0.0, 0.0, 0.0])
    ego_rot = info.get("ego2global_rotation", [1.0, 0.0, 0.0, 0.0])
    ego_trans = info.get("ego2global_translation", [0.0, 0.0, 0.0])

    rows: list[dict[str, Any]] = []
    for idx, box in enumerate(gt_boxes):
        if idx >= len(gt_names):
            continue
        if valid_flags and idx < len(valid_flags) and not bool(valid_flags[idx]):
            continue
        values = _to_plain_sequence(box)
        if len(values) < 3:
            continue
        center_ego = _transform_point_quat(values[:3], lidar_rot, lidar_trans)
        center_global = _transform_point_quat(center_ego, ego_rot, ego_trans)
        velocity = [0.0, 0.0]
        if idx < len(gt_velocity):
            velocity_values = _to_plain_sequence(gt_velocity[idx])
            if len(velocity_values) >= 2:
                velocity = [_safe_float(velocity_values[0]), _safe_float(velocity_values[1])]
        rows.append(
            {
                "detection_name": str(gt_names[idx]),
                "translation": [float(value) for value in center_global],
                "detection_score": 1.0,
                "velocity": velocity,
            }
        )
    return rows


def _quat_to_rot(q: Any) -> np.ndarray:
    npx = _require_numpy()
    w, x, y, z = [float(v) for v in q]
    return npx.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=npx.float64,
    )


def _load_radar_points(path: Path) -> tuple[np.ndarray, list[str]]:
    npx = _require_numpy()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = npx.load(path).astype(npx.float32, copy=False)
        names = RADAR_FEATURE_NAMES[: points.shape[1]]
        if len(names) < points.shape[1]:
            names = [*names, *[f"f{i}" for i in range(len(names), points.shape[1])]]
        return points, names
    if suffix == ".npz":
        payload = npx.load(path, allow_pickle=True)
        points = npx.asarray(payload["points"], dtype=npx.float32)
        if "feature_names" in payload:
            names = [str(value) for value in payload["feature_names"].tolist()]
        else:
            names = RADAR_FEATURE_NAMES[: points.shape[1]]
        return points, names
    if suffix == ".pcd":
        try:
            from nuscenes.utils.data_classes import RadarPointCloud  # type: ignore
        except Exception as exc:  # pragma: no cover - real nuScenes path.
            raise RuntimeError("Reading .pcd radar files requires nuscenes-devkit") from exc
        cloud = RadarPointCloud.from_file(str(path))
        points = npx.asarray(cloud.points.T, dtype=npx.float32)
        return points, RADAR_FEATURE_NAMES[: points.shape[1]]
    raise ValueError(f"unsupported radar file type: {path}")


def _radar_name_index(names: list[str]) -> dict[str, int]:
    aliases = {"velocity": "vx_comp", "radial_velocity": "vx_comp", "doppler": "vx_comp"}
    out: dict[str, int] = {}
    for idx, name in enumerate(names):
        key = aliases.get(str(name).strip().lower(), str(name).strip().lower())
        if key not in out:
            out[key] = idx
    return out


def _radars_from_nuscenes_tables(info: dict[str, Any], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    token = info.get("sample_token") or info.get("token")
    if token is None:
        return {}
    nusc = getattr(args, "_rgate_nuscenes", None)
    if nusc is None:
        try:
            from nuscenes.nuscenes import NuScenes  # type: ignore
        except Exception as exc:  # pragma: no cover - real nuScenes path.
            raise RuntimeError("--radar-from-nuscenes-tables requires nuscenes-devkit") from exc
        nusc = NuScenes(version=args.nuscenes_version, dataroot=str(args.nuscenes_root), verbose=False)
        setattr(args, "_rgate_nuscenes", nusc)

    sample = nusc.get("sample", str(token))
    sample_data = sample.get("data", {}) if isinstance(sample, dict) else {}
    channels = [channel.strip() for channel in str(args.radar_channels).split(",") if channel.strip()]
    radars: dict[str, list[dict[str, Any]]] = {}
    for channel in channels:
        sd_token = sample_data.get(channel)
        sweeps: list[dict[str, Any]] = []
        seen: set[str] = set()
        while sd_token and len(sweeps) < args.radar_max_sweeps and str(sd_token) not in seen:
            seen.add(str(sd_token))
            sd = nusc.get("sample_data", sd_token)
            filename = sd.get("filename") if isinstance(sd, dict) else None
            if filename:
                path = Path(str(filename))
                if not path.is_absolute():
                    path = Path(args.nuscenes_root) / path
                calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                sweeps.append(
                    {
                        "data_path": str(path),
                        "sensor2ego_translation": calib["translation"],
                        "sensor2ego_rotation": calib["rotation"],
                        "ego2global_translation": pose["translation"],
                        "ego2global_rotation": pose["rotation"],
                        "timestamp": sd.get("timestamp"),
                    }
                )
            sd_token = sd.get("prev") if isinstance(sd, dict) else ""
        if sweeps:
            radars[channel] = sweeps
    return radars


def _radar_points_global_for_info(info: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, int]]:
    npx = _require_numpy()
    radars = info.get("radars")
    if not isinstance(radars, dict) and getattr(args, "radar_from_nuscenes_tables", False):
        radars = _radars_from_nuscenes_tables(info, args)
    if not isinstance(radars, dict):
        return npx.zeros((0, 8), dtype=npx.float32), {}
    chunks: list[np.ndarray] = []
    merged_mapping: dict[str, int] | None = None
    for channel in sorted(radars):
        sweeps = radars.get(channel)
        if not isinstance(sweeps, list):
            continue
        for sweep in sweeps[: args.radar_max_sweeps]:
            if not isinstance(sweep, dict) or not sweep.get("data_path"):
                continue
            path = Path(str(sweep["data_path"]))
            if not path.is_file():
                continue
            points, names = _load_radar_points(path)
            if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
                continue
            mapping = _radar_name_index(names)
            sensor_rot = _quat_to_rot(sweep["sensor2ego_rotation"])
            sensor_trans = npx.asarray(sweep["sensor2ego_translation"], dtype=npx.float64)
            ego_rot = _quat_to_rot(sweep["ego2global_rotation"])
            ego_trans = npx.asarray(sweep["ego2global_translation"], dtype=npx.float64)
            xyz_sensor = points[:, :3].astype(npx.float64, copy=False)
            xyz_ego = (sensor_rot @ xyz_sensor.T).T + sensor_trans
            xyz_global = (ego_rot @ xyz_ego.T).T + ego_trans

            vx_idx = mapping.get("vx_comp", mapping.get("vx"))
            vy_idx = mapping.get("vy_comp", mapping.get("vy"))
            if vx_idx is not None and vy_idx is not None and vx_idx < points.shape[1] and vy_idx < points.shape[1]:
                velocity_sensor = npx.zeros((points.shape[0], 3), dtype=npx.float64)
                velocity_sensor[:, 0] = points[:, vx_idx]
                velocity_sensor[:, 1] = points[:, vy_idx]
                velocity_global = (ego_rot @ (sensor_rot @ velocity_sensor.T)).T[:, :2]
            else:
                velocity_global = npx.zeros((points.shape[0], 2), dtype=npx.float64)

            rcs_idx = mapping.get("rcs")
            rcs = points[:, rcs_idx] if rcs_idx is not None and rcs_idx < points.shape[1] else npx.zeros((points.shape[0],), dtype=npx.float32)
            time_lag_idx = mapping.get("time_lag")
            if time_lag_idx is not None and time_lag_idx < points.shape[1]:
                time_lag = points[:, time_lag_idx]
            else:
                sample_ts = info.get("timestamp")
                sweep_ts = sweep.get("timestamp")
                if sample_ts is not None and sweep_ts is not None:
                    lag_seconds = (_safe_float(sample_ts) - _safe_float(sweep_ts)) / 1e6
                else:
                    lag_seconds = 0.0
                time_lag = npx.full((points.shape[0],), lag_seconds, dtype=npx.float32)
            chunk = npx.column_stack(
                [
                    xyz_global[:, 0],
                    xyz_global[:, 1],
                    xyz_global[:, 2],
                    rcs.astype(npx.float64, copy=False),
                    velocity_global[:, 0],
                    velocity_global[:, 1],
                    time_lag.astype(npx.float64, copy=False),
                ]
            )
            chunks.append(chunk.astype(npx.float32, copy=False))
            if merged_mapping is None:
                merged_mapping = {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "rcs": 3,
                    "vx": 4,
                    "vy": 5,
                    "time_lag": 6,
                }
    if not chunks:
        return npx.zeros((0, 7), dtype=npx.float32), {"x": 0, "y": 1, "z": 2, "rcs": 3, "vx": 4, "vy": 5, "time_lag": 6}
    return npx.concatenate(chunks, axis=0), merged_mapping or {"x": 0, "y": 1, "z": 2, "rcs": 3, "vx": 4, "vy": 5, "time_lag": 6}


def _radar_evidence_features(
    *,
    fused_row: dict[str, Any],
    radar_points: np.ndarray,
    radar_mapping: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, float]:
    npx = _require_numpy()
    if radar_points.size == 0:
        return {
            "radar_point_count": 0.0,
            "radar_point_density": 0.0,
            "radar_min_center_dist": args.radar_search_radius,
            "radar_mean_center_dist": args.radar_search_radius,
            "radar_rcs_mean": 0.0,
            "radar_rcs_max": 0.0,
            "radar_vx_mean": 0.0,
            "radar_vy_mean": 0.0,
            "radar_speed_mean": 0.0,
            "radar_velocity_delta_mean": 0.0,
            "radar_velocity_support_count": 0.0,
            "radar_time_lag_mean": 0.0,
            "radar_time_lag_min": 0.0,
            "radar_time_lag_max": 0.0,
            "radar_time_lag_std": 0.0,
        }
    center = npx.asarray(fused_row.get("translation", [0.0, 0.0, 0.0])[:2], dtype=npx.float64)
    size = fused_row.get("size", [0.0, 0.0, 0.0])
    width = max(_safe_float(size[0], 0.0), 0.0)
    length = max(_safe_float(size[1], 0.0), 0.0)
    yaw = yaw_from_quat([float(v) for v in fused_row.get("rotation", [1.0, 0.0, 0.0, 0.0])])
    xy = radar_points[:, [radar_mapping["x"], radar_mapping["y"]]].astype(npx.float64, copy=False)
    rel = xy - center.reshape(1, 2)
    dist = npx.hypot(rel[:, 0], rel[:, 1])
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = rel[:, 0] * cos_yaw - rel[:, 1] * sin_yaw
    local_y = rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw
    in_box = (
        (npx.abs(local_x) <= width * 0.5 + args.radar_box_margin)
        & (npx.abs(local_y) <= length * 0.5 + args.radar_box_margin)
        & (dist <= args.radar_search_radius)
    )
    selected = radar_points[in_box]
    selected_dist = dist[in_box]
    count = int(selected.shape[0])
    area = max((width + 2.0 * args.radar_box_margin) * (length + 2.0 * args.radar_box_margin), 1e-6)
    if count == 0:
        return {
            "radar_point_count": 0.0,
            "radar_point_density": 0.0,
            "radar_min_center_dist": args.radar_search_radius,
            "radar_mean_center_dist": args.radar_search_radius,
            "radar_rcs_mean": 0.0,
            "radar_rcs_max": 0.0,
            "radar_vx_mean": 0.0,
            "radar_vy_mean": 0.0,
            "radar_speed_mean": 0.0,
            "radar_velocity_delta_mean": 0.0,
            "radar_velocity_support_count": 0.0,
            "radar_time_lag_mean": 0.0,
            "radar_time_lag_min": 0.0,
            "radar_time_lag_max": 0.0,
            "radar_time_lag_std": 0.0,
        }
    rcs = selected[:, radar_mapping["rcs"]].astype(npx.float64, copy=False)
    vx = selected[:, radar_mapping["vx"]].astype(npx.float64, copy=False)
    vy = selected[:, radar_mapping["vy"]].astype(npx.float64, copy=False)
    box_vel = fused_row.get("velocity", [0.0, 0.0])
    box_vx = _safe_float(box_vel[0] if isinstance(box_vel, list) and len(box_vel) > 0 else 0.0)
    box_vy = _safe_float(box_vel[1] if isinstance(box_vel, list) and len(box_vel) > 1 else 0.0)
    velocity_delta = npx.hypot(vx - box_vx, vy - box_vy)
    radar_speed = npx.hypot(vx, vy)
    finite_delta = velocity_delta[npx.isfinite(velocity_delta)]
    finite_vx = vx[npx.isfinite(vx)]
    finite_vy = vy[npx.isfinite(vy)]
    finite_speed = radar_speed[npx.isfinite(radar_speed)]
    time_lag_idx = radar_mapping.get("time_lag")
    if time_lag_idx is not None and time_lag_idx < selected.shape[1]:
        time_lag = selected[:, time_lag_idx].astype(npx.float64, copy=False)
        finite_time_lag = time_lag[npx.isfinite(time_lag)]
    else:
        finite_time_lag = npx.zeros((0,), dtype=npx.float64)
    return {
        "radar_point_count": float(count),
        "radar_point_density": float(count / area),
        "radar_min_center_dist": float(npx.min(selected_dist)),
        "radar_mean_center_dist": float(npx.mean(selected_dist)),
        "radar_rcs_mean": float(npx.mean(npx.nan_to_num(rcs, nan=0.0))),
        "radar_rcs_max": float(npx.max(npx.nan_to_num(rcs, nan=0.0))),
        "radar_vx_mean": float(npx.mean(finite_vx)) if finite_vx.size else 0.0,
        "radar_vy_mean": float(npx.mean(finite_vy)) if finite_vy.size else 0.0,
        "radar_speed_mean": float(npx.mean(finite_speed)) if finite_speed.size else 0.0,
        "radar_velocity_delta_mean": float(npx.mean(finite_delta)) if finite_delta.size else 0.0,
        "radar_velocity_support_count": float(finite_delta.size),
        "radar_time_lag_mean": float(npx.mean(finite_time_lag)) if finite_time_lag.size else 0.0,
        "radar_time_lag_min": float(npx.min(finite_time_lag)) if finite_time_lag.size else 0.0,
        "radar_time_lag_max": float(npx.max(finite_time_lag)) if finite_time_lag.size else 0.0,
        "radar_time_lag_std": float(npx.std(finite_time_lag)) if finite_time_lag.size else 0.0,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _cluster_features(
    *,
    cluster: list[Box],
    fused_row: dict[str, Any],
    source_names: list[str],
) -> dict[str, float]:
    scores = [float(box.score) for box in cluster]
    weighted_scores = [float(box.weighted_score) for box in cluster]
    geom_weights = [float(box.weighted_geometry_score) for box in cluster]
    fused_xy = _row_xy(fused_row)
    distances = [center_distance(box.xy, fused_xy) for box in cluster]
    ranges = [math.hypot(float(box.translation[0]), float(box.translation[1])) for box in cluster]
    yaws = [yaw_from_quat(box.rotation) for box in cluster]
    source_counts = Counter(box.source for box in cluster)
    group_counts = Counter(box.group for box in cluster)
    candidate_count = sum(1 for box in cluster if box.mode == "candidate_only")
    full_count = sum(1 for box in cluster if box.mode == "full")
    no_velocity_count = sum(1 for box in cluster if box.mode == "no_velocity")

    features: dict[str, float] = {
        "cluster_size": float(len(cluster)),
        "source_count": float(len(source_counts)),
        "group_count": float(len(group_counts)),
        "candidate_only_count": float(candidate_count),
        "full_mode_count": float(full_count),
        "no_velocity_mode_count": float(no_velocity_count),
        "candidate_fraction": candidate_count / max(len(cluster), 1),
        "base_score": _safe_float(fused_row.get("detection_score")),
        "max_raw_score": max(scores) if scores else 0.0,
        "mean_raw_score": mean(scores) if scores else 0.0,
        "sum_raw_score": sum(scores),
        "max_weighted_score": max(weighted_scores) if weighted_scores else 0.0,
        "sum_weighted_score": sum(weighted_scores),
        "sum_geometry_weighted_score": sum(geom_weights),
        "center_mean_dist": mean(distances) if distances else 0.0,
        "center_max_dist": max(distances) if distances else 0.0,
        "center_p90_dist": _percentile(distances, 0.9) or 0.0,
        "range_xy": math.hypot(fused_xy[0], fused_xy[1]),
        "mean_input_range_xy": mean(ranges) if ranges else 0.0,
        "yaw_sin_mean": mean([math.sin(yaw) for yaw in yaws]) if yaws else 0.0,
        "yaw_cos_mean": mean([math.cos(yaw) for yaw in yaws]) if yaws else 1.0,
        "velocity_norm": math.hypot(
            _safe_float(fused_row.get("velocity", [0.0, 0.0])[0]),
            _safe_float(fused_row.get("velocity", [0.0, 0.0])[1]),
        ),
        "size_x": _safe_float(fused_row.get("size", [0.0, 0.0, 0.0])[0]),
        "size_y": _safe_float(fused_row.get("size", [0.0, 0.0, 0.0])[1]),
        "size_z": _safe_float(fused_row.get("size", [0.0, 0.0, 0.0])[2]),
    }

    for source in source_names:
        source_scores = [box.score for box in cluster if box.source == source]
        features[f"src_present:{source}"] = 1.0 if source_scores else 0.0
        features[f"src_count:{source}"] = float(len(source_scores))
        features[f"src_max_score:{source}"] = max(source_scores) if source_scores else 0.0
        features[f"src_mean_score:{source}"] = mean(source_scores) if source_scores else 0.0

    # A compact disagreement proxy: higher values mean score support is spread
    # across more sources instead of concentrated in one detector.
    total_by_source = [sum(box.score for box in cluster if box.source == source) for source in source_names]
    total_score = sum(total_by_source)
    if total_score > 0:
        features["source_score_entropy"] = -sum(
            (value / total_score) * math.log(max(value / total_score, 1e-12))
            for value in total_by_source
            if value > 0
        )
    else:
        features["source_score_entropy"] = 0.0

    return features


def _compact_cluster_boxes(cluster: list[Box]) -> list[dict[str, Any]]:
    return [
        {
            "source": box.source,
            "group": box.group,
            "mode": box.mode,
            "score": float(box.score),
            "score_weight": float(box.score_weight),
            "geometry_weight": float(box.geometry_weight),
            "translation": [float(value) for value in box.translation],
            "size": [float(value) for value in box.size],
            "rotation": [float(value) for value in box.rotation],
            "velocity": [float(value) for value in box.velocity],
            "attribute_name": box.attribute_name,
        }
        for box in cluster
    ]


def _sample_cache_rows(
    *,
    sample_token: str,
    sample_boxes_by_expert: list[tuple[Any, list[Box]]],
    target_rows: list[dict[str, Any]],
    source_names: list[str],
    thresholds: dict[str, float],
    radar_points: np.ndarray | None,
    radar_mapping: dict[str, int] | None,
    ego_xy: tuple[float, float] | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter[int], Counter[str], int, float]:
    by_class: dict[str, list[Box]] = defaultdict(list)
    for _, boxes in sample_boxes_by_expert:
        for box in boxes:
            by_class[box.detection_name].append(box)

    rows: list[dict[str, Any]] = []
    support_hist: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()
    target_count = 0
    target_quality_sum = 0.0
    cluster_idx = 0
    for class_name, class_boxes in sorted(by_class.items()):
        threshold = thresholds.get(class_name, args.default_threshold)
        for cluster in cluster_boxes(class_boxes, threshold=threshold):
            fused_row = fuse_cluster(
                cluster,
                gamma=args.gamma,
                tau=args.tau,
                score_power=args.score_power,
                score_cap=None if args.no_score_cap else args.score_cap,
                score_mode=args.score_mode,
                support_mode=args.support_mode,
                single_candidate_score_weight=args.single_candidate_score_weight,
            )
            target = _target_quality(
                fused_row=fused_row,
                target_rows=target_rows,
            )
            if target["quality"] is not None:
                target_count += 1
                target_quality_sum += float(target["quality"])
            sources = sorted({box.source for box in cluster})
            groups = sorted({box.group for box in cluster})
            support_hist[len(groups if args.support_mode == "group" else sources)] += 1
            class_counts[class_name] += 1
            features = _cluster_features(
                cluster=cluster,
                fused_row=fused_row,
                source_names=source_names,
            )
            if ego_xy is not None:
                translation = fused_row.get("translation", [0.0, 0.0, 0.0])
                if isinstance(translation, list) and len(translation) >= 2:
                    features["ego_range_xy"] = math.hypot(
                        _safe_float(translation[0]) - ego_xy[0],
                        _safe_float(translation[1]) - ego_xy[1],
                    )
            if args.include_radar_evidence:
                npx = _require_numpy()
                features.update(
                    _radar_evidence_features(
                        fused_row=fused_row,
                        radar_points=radar_points if radar_points is not None else npx.zeros((0, 7), dtype=npx.float32),
                        radar_mapping=radar_mapping or {"x": 0, "y": 1, "z": 2, "rcs": 3, "vx": 4, "vy": 5, "time_lag": 6},
                        args=args,
                    )
                )
            row = {
                "schema_version": SCHEMA_VERSION,
                "sample_token": sample_token,
                "cluster_id": f"{sample_token}:{cluster_idx:06d}",
                "class_name": class_name,
                "sources": sources,
                "groups": groups,
                "source_signature": "+".join(sources),
                "group_signature": "+".join(groups),
                "base_score": _safe_float(fused_row.get("detection_score")),
                "output_box": fused_row,
                "features": features,
                "categorical": {
                    "class_name": class_name,
                    "source_signature": "+".join(sources),
                    "group_signature": "+".join(groups),
                },
                "target": target,
            }
            if args.include_cluster_boxes:
                row["cluster_boxes"] = _compact_cluster_boxes(cluster)
            rows.append(row)
            cluster_idx += 1
    return rows, support_hist, class_counts, target_count, target_quality_sum


def _summary_payload(
    *,
    args: argparse.Namespace,
    loaded_experts: list[tuple[Any, Any]],
    target_count: int,
    target_quality_sum: float,
    requested_tokens: set[str],
    missing_requested_tokens: list[str],
    sample_count: int,
    row_count: int,
    class_counts: Counter[str],
    support_hist: Counter[int],
    thresholds: dict[str, float],
    stream_build: bool,
    radar_sample_count: int,
    radar_point_count: int,
    radar_rows_with_points: int,
) -> dict[str, Any]:
    return {
        "schema_version": "bge_af_arbiter_cache_summary_v1",
        "out": str(args.out),
        "stream_build": stream_build,
        "experts": [
            {
                "name": expert.name,
                "score_weight": expert.score_weight,
                "geometry_weight": expert.geometry_weight,
                "group": expert.group,
                "mode": expert.mode,
                "path": str(expert.path),
            }
            for expert, _ in loaded_experts
        ],
        "target_result_json": str(args.target_result_json) if args.target_result_json else None,
        "target_from_info_gt": bool(args.target_from_info_gt),
        "sample_info_pkl": str(args.sample_info_pkl) if args.sample_info_pkl else None,
        "sample_token_file": str(args.sample_token_file) if args.sample_token_file else None,
        "requested_sample_count": len(requested_tokens) if requested_tokens else None,
        "missing_requested_sample_count": len(missing_requested_tokens),
        "missing_requested_sample_examples": missing_requested_tokens[:10],
        "sample_count": sample_count,
        "row_count": row_count,
        "target_labeled_row_count": target_count,
        "mean_target_quality": target_quality_sum / max(target_count, 1),
        "class_counts": dict(sorted(class_counts.items())),
        "support_histogram": {str(key): value for key, value in sorted(support_hist.items())},
        "cluster_thresholds": thresholds,
        "gamma": args.gamma,
        "tau": args.tau,
        "score_power": args.score_power,
        "score_mode": args.score_mode,
        "support_mode": args.support_mode,
        "score_cap": None if args.no_score_cap else args.score_cap,
        "single_candidate_score_weight": args.single_candidate_score_weight,
        "include_cluster_boxes": bool(args.include_cluster_boxes),
        "include_radar_evidence": bool(args.include_radar_evidence),
        "radar_from_nuscenes_tables": bool(args.radar_from_nuscenes_tables),
        "nuscenes_root": str(args.nuscenes_root) if args.nuscenes_root else None,
        "nuscenes_version": args.nuscenes_version,
        "radar_channels": args.radar_channels,
        "radar_max_sweeps": args.radar_max_sweeps,
        "radar_box_margin": args.radar_box_margin,
        "radar_search_radius": args.radar_search_radius,
        "radar_sample_count": radar_sample_count,
        "radar_point_count": radar_point_count,
        "radar_rows_with_points": radar_rows_with_points,
    }


def build_cache_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = dict(DEFAULT_CLUSTER_THRESHOLDS)
    thresholds.update(dict(args.class_threshold))

    loaded = []
    all_tokens: set[str] = set()
    for expert in args.expert:
        by_sample, _ = load_boxes(expert)
        loaded.append((expert, by_sample))
        all_tokens.update(by_sample)
    source_names = sorted(expert.name for expert, _ in loaded)

    ordered_file_tokens = _load_ordered_sample_tokens_from_file(args.sample_token_file)
    requested_tokens: set[str] = set()
    requested_tokens.update(_load_sample_tokens_from_info(args.sample_info_pkl))
    requested_tokens.update(ordered_file_tokens)
    missing_requested_tokens: list[str] = []
    if requested_tokens:
        missing_requested_tokens = sorted(requested_tokens - all_tokens)
    ordered_tokens = _ordered_selected_tokens(
        all_tokens=all_tokens,
        requested_tokens=requested_tokens,
        ordered_file_tokens=ordered_file_tokens,
        max_samples=args.max_samples,
    )
    all_tokens = set(ordered_tokens)

    info_by_token = _load_infos_by_token(args.sample_info_pkl) if args.sample_info_pkl else {}
    target_by_sample = _load_result_rows(args.target_result_json)
    if args.target_from_info_gt:
        target_by_sample = {
            sample_token: _target_rows_from_info_gt(info)
            for sample_token, info in info_by_token.items()
        }
    rows: list[dict[str, Any]] = []
    support_hist: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()
    target_count = 0
    target_quality_sum = 0.0
    radar_sample_count = 0
    radar_point_count = 0
    radar_rows_with_points = 0

    for sample_token in ordered_tokens:
        radar_points = None
        radar_mapping = None
        info = info_by_token.get(sample_token, {})
        if args.include_radar_evidence:
            radar_points, radar_mapping = _radar_points_global_for_info(info, args)
            radar_sample_count += int(radar_points.shape[0] > 0)
            radar_point_count += int(radar_points.shape[0])
        sample_rows, sample_support_hist, sample_class_counts, sample_target_count, sample_target_quality_sum = _sample_cache_rows(
            sample_token=sample_token,
            sample_boxes_by_expert=[(expert, sample_boxes.get(sample_token, [])) for expert, sample_boxes in loaded],
            target_rows=target_by_sample.get(sample_token, []),
            source_names=source_names,
            thresholds=thresholds,
            radar_points=radar_points,
            radar_mapping=radar_mapping,
            ego_xy=_ego_xy_from_info(info),
            args=args,
        )
        if args.include_radar_evidence:
            radar_rows_with_points += sum(
                1 for row in sample_rows if _safe_float(row.get("features", {}).get("radar_point_count")) > 0
            )
        rows.extend(sample_rows)
        support_hist.update(sample_support_hist)
        class_counts.update(sample_class_counts)
        target_count += sample_target_count
        target_quality_sum += sample_target_quality_sum

    summary = _summary_payload(
        args=args,
        loaded_experts=loaded,
        target_count=target_count,
        target_quality_sum=target_quality_sum,
        requested_tokens=requested_tokens,
        missing_requested_tokens=missing_requested_tokens,
        sample_count=len(all_tokens),
        row_count=len(rows),
        class_counts=class_counts,
        support_hist=support_hist,
        thresholds=thresholds,
        stream_build=False,
        radar_sample_count=radar_sample_count,
        radar_point_count=radar_point_count,
        radar_rows_with_points=radar_rows_with_points,
    )
    return rows, summary


def build_cache_rows_streaming(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = dict(DEFAULT_CLUSTER_THRESHOLDS)
    thresholds.update(dict(args.class_threshold))

    ordered_file_tokens = _load_ordered_sample_tokens_from_file(args.sample_token_file)
    requested_tokens: set[str] = set()
    requested_tokens.update(_load_sample_tokens_from_info(args.sample_info_pkl))
    requested_tokens.update(ordered_file_tokens)

    with ExitStack() as stack:
        loaded: list[tuple[Any, ResultJsonIndex]] = []
        all_tokens: set[str] = set()
        for expert in args.expert:
            index = stack.enter_context(ResultJsonIndex(expert.path))
            loaded.append((expert, index))
            all_tokens.update(index.tokens())
        source_names = sorted(expert.name for expert, _ in loaded)

        missing_requested_tokens: list[str] = []
        if requested_tokens:
            missing_requested_tokens = sorted(requested_tokens - all_tokens)
        ordered_tokens = _ordered_selected_tokens(
            all_tokens=all_tokens,
            requested_tokens=requested_tokens,
            ordered_file_tokens=ordered_file_tokens,
            max_samples=args.max_samples,
        )
        all_tokens = set(ordered_tokens)

        info_by_token = _load_infos_by_token(args.sample_info_pkl) if args.sample_info_pkl else {}
        target_index = stack.enter_context(ResultJsonIndex(args.target_result_json)) if args.target_result_json else None
        target_gt_by_sample: dict[str, list[dict[str, Any]]] = {}
        if args.target_from_info_gt:
            target_gt_by_sample = {
                sample_token: _target_rows_from_info_gt(info)
                for sample_token, info in info_by_token.items()
            }

        support_hist: Counter[int] = Counter()
        class_counts: Counter[str] = Counter()
        target_count = 0
        target_quality_sum = 0.0
        row_count = 0
        radar_sample_count = 0
        radar_point_count = 0
        radar_rows_with_points = 0

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for sample_token in ordered_tokens:
                radar_points = None
                radar_mapping = None
                info = info_by_token.get(sample_token, {})
                if args.include_radar_evidence:
                    radar_points, radar_mapping = _radar_points_global_for_info(info, args)
                    radar_sample_count += int(radar_points.shape[0] > 0)
                    radar_point_count += int(radar_points.shape[0])
                sample_rows, sample_support_hist, sample_class_counts, sample_target_count, sample_target_quality_sum = _sample_cache_rows(
                    sample_token=sample_token,
                    sample_boxes_by_expert=[
                        (expert, _boxes_from_rows(expert, sample_token, index.rows(sample_token)))
                        for expert, index in loaded
                    ],
                    target_rows=target_gt_by_sample.get(sample_token, [])
                    if args.target_from_info_gt
                    else target_index.rows(sample_token)
                    if target_index
                    else [],
                    source_names=source_names,
                    thresholds=thresholds,
                    radar_points=radar_points,
                    radar_mapping=radar_mapping,
                    ego_xy=_ego_xy_from_info(info),
                    args=args,
                )
                for row in sample_rows:
                    fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                row_count += len(sample_rows)
                if args.include_radar_evidence:
                    radar_rows_with_points += sum(
                        1 for row in sample_rows if _safe_float(row.get("features", {}).get("radar_point_count")) > 0
                    )
                support_hist.update(sample_support_hist)
                class_counts.update(sample_class_counts)
                target_count += sample_target_count
                target_quality_sum += sample_target_quality_sum

        return _summary_payload(
            args=args,
            loaded_experts=loaded,
            target_count=target_count,
            target_quality_sum=target_quality_sum,
            requested_tokens=requested_tokens,
            missing_requested_tokens=missing_requested_tokens,
            sample_count=len(all_tokens),
            row_count=row_count,
            class_counts=class_counts,
            support_hist=support_hist,
            thresholds=thresholds,
            stream_build=True,
            radar_sample_count=radar_sample_count,
            radar_point_count=radar_point_count,
            radar_rows_with_points=radar_rows_with_points,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", action="append", type=parse_expert, required=True)
    parser.add_argument("--target-result-json", type=Path)
    parser.add_argument(
        "--target-from-info-gt",
        action="store_true",
        help="Use gt_boxes/gt_names from --sample-info-pkl as target labels after LiDAR-to-global transform.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=("weighted_max", "max", "mean"), default="weighted_max")
    parser.add_argument("--support-mode", choices=("source", "group"), default="source")
    parser.add_argument("--score-cap", type=float, default=1.0)
    parser.add_argument("--no-score-cap", action="store_true")
    parser.add_argument("--single-candidate-score-weight", type=float, default=1.0)
    parser.add_argument("--default-threshold", type=float, default=0.7)
    parser.add_argument("--class-threshold", action="append", type=parse_class_threshold, default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-info-pkl", type=Path)
    parser.add_argument("--sample-token-file", type=Path)
    parser.add_argument(
        "--stream-build",
        action="store_true",
        help="Build cache by mmap-indexing expert result JSONs and writing rows per sample. Lower memory for full-val.",
    )
    parser.add_argument(
        "--include-cluster-boxes",
        action="store_true",
        help="Store compact source boxes per cluster for default-off center refinement experiments.",
    )
    parser.add_argument(
        "--include-radar-evidence",
        action="store_true",
        help="Attach per-cluster radar evidence features from --sample-info-pkl radars.",
    )
    parser.add_argument(
        "--radar-from-nuscenes-tables",
        action="store_true",
        help="If --sample-info-pkl has no radars field, resolve radar sweeps from nuScenes raw tables by sample_token.",
    )
    parser.add_argument("--nuscenes-root", type=Path)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--radar-channels", default=",".join(NUSCENES_RADAR_CHANNELS))
    parser.add_argument("--radar-max-sweeps", type=int, default=1)
    parser.add_argument("--radar-box-margin", type=float, default=1.0)
    parser.add_argument("--radar-search-radius", type=float, default=4.0)
    args = parser.parse_args()

    if args.tau <= 0:
        raise SystemExit("--tau must be > 0")
    if args.gamma < 0:
        raise SystemExit("--gamma must be >= 0")
    if args.score_power < 0:
        raise SystemExit("--score-power must be >= 0")
    if args.single_candidate_score_weight < 0:
        raise SystemExit("--single-candidate-score-weight must be >= 0")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")
    if args.include_radar_evidence and args.sample_info_pkl is None:
        raise SystemExit("--include-radar-evidence requires --sample-info-pkl")
    if args.radar_max_sweeps <= 0:
        raise SystemExit("--radar-max-sweeps must be > 0")
    if args.radar_box_margin < 0:
        raise SystemExit("--radar-box-margin must be >= 0")
    if args.radar_search_radius <= 0:
        raise SystemExit("--radar-search-radius must be > 0")
    if args.target_result_json is not None and not args.target_result_json.is_file():
        raise SystemExit(f"missing target result json: {args.target_result_json}")
    if args.sample_info_pkl is not None and not args.sample_info_pkl.is_file():
        raise SystemExit(f"missing sample info pkl: {args.sample_info_pkl}")
    if args.radar_from_nuscenes_tables and args.nuscenes_root is None:
        raise SystemExit("--radar-from-nuscenes-tables requires --nuscenes-root")
    if args.radar_from_nuscenes_tables and not args.nuscenes_root.is_dir():
        raise SystemExit(f"missing nuScenes root: {args.nuscenes_root}")
    if args.sample_token_file is not None and not args.sample_token_file.is_file():
        raise SystemExit(f"missing sample token file: {args.sample_token_file}")
    if args.target_result_json is not None and args.target_from_info_gt:
        raise SystemExit("--target-result-json and --target-from-info-gt are mutually exclusive")
    if args.target_from_info_gt and args.sample_info_pkl is None:
        raise SystemExit("--target-from-info-gt requires --sample-info-pkl")

    if args.stream_build:
        summary = build_cache_rows_streaming(args)
    else:
        rows, summary = build_cache_rows(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
