#!/usr/bin/env python3
"""Fuse nuScenes detection result_json files with reliability-aware WBF."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CLUSTER_THRESHOLDS = {
    "car": 0.7,
    "truck": 1.0,
    "bus": 1.0,
    "trailer": 1.0,
    "construction_vehicle": 1.0,
    "pedestrian": 0.4,
    "motorcycle": 0.5,
    "bicycle": 0.5,
    "traffic_cone": 0.25,
    "barrier": 0.4,
}


@dataclass(frozen=True)
class Expert:
    name: str
    score_weight: float
    geometry_weight: float
    group: str
    mode: str
    path: Path


@dataclass
class Box:
    sample_token: str
    detection_name: str
    score: float
    translation: list[float]
    size: list[float]
    rotation: list[float]
    velocity: list[float]
    attribute_name: str
    source: str
    group: str
    mode: str
    score_weight: float
    geometry_weight: float

    @property
    def xy(self) -> tuple[float, float]:
        return float(self.translation[0]), float(self.translation[1])

    @property
    def weighted_score(self) -> float:
        return max(0.0, float(self.score)) * max(0.0, float(self.score_weight))

    @property
    def weighted_geometry_score(self) -> float:
        if self.mode == "candidate_only":
            return 0.0
        return max(0.0, float(self.score)) * max(0.0, float(self.geometry_weight))


def parse_expert(value: str) -> Expert:
    legacy_parts = value.split(":", 2)
    extended_parts = value.split(":", 5)
    if len(extended_parts) == 6 and extended_parts[4] in {"full", "no_velocity", "candidate_only"}:
        name, score_weight_text, geometry_weight_text, group, mode, path_text = extended_parts
    elif len(legacy_parts) == 3:
        name, weight_text, path_text = legacy_parts
        score_weight_text = weight_text
        geometry_weight_text = weight_text
        group = name
        mode = "full"
    else:
        raise argparse.ArgumentTypeError(
            "expected NAME:WEIGHT:artifacts/results_nusc.json or "
            "NAME:SCORE_WEIGHT:GEOMETRY_WEIGHT:GROUP:MODE:artifacts/results_nusc.json"
        )
    if not name:
        raise argparse.ArgumentTypeError("expert name must be non-empty")
    if not group:
        raise argparse.ArgumentTypeError("expert group must be non-empty")
    if mode not in {"full", "no_velocity", "candidate_only"}:
        raise argparse.ArgumentTypeError(
            "expert mode must be full, no_velocity, or candidate_only"
        )
    try:
        score_weight = float(score_weight_text)
        geometry_weight = float(geometry_weight_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid expert weight") from exc
    path = Path(path_text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing result json: {path}")
    if score_weight < 0 or geometry_weight < 0:
        raise argparse.ArgumentTypeError("expert weight must be >= 0")
    return Expert(
        name=name,
        score_weight=score_weight,
        geometry_weight=geometry_weight,
        group=group,
        mode=mode,
        path=path,
    )


def parse_class_threshold(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected CLASS=THRESHOLD")
    name, threshold_text = value.split("=", 1)
    try:
        threshold = float(threshold_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid threshold: {threshold_text}") from exc
    if threshold <= 0:
        raise argparse.ArgumentTypeError("threshold must be > 0")
    return name, threshold


def yaw_from_quat(rotation: list[float]) -> float:
    w, x, y, z = [float(v) for v in rotation]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_boxes(expert: Expert) -> tuple[dict[str, list[Box]], dict[str, Any]]:
    data = json.loads(expert.path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{expert.path} has no results dict")
    by_sample: dict[str, list[Box]] = defaultdict(list)
    for sample_token, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"{expert.path}: results[{sample_token!r}] is not a list")
        by_sample.setdefault(sample_token, [])
        for row in rows:
            name = str(row["detection_name"])
            score = float(row.get("detection_score", 0.0))
            translation = [float(v) for v in row["translation"]]
            size = [float(v) for v in row["size"]]
            rotation = [float(v) for v in row["rotation"]]
            velocity = [float(v) for v in row.get("velocity", [0.0, 0.0])[:2]]
            while len(velocity) < 2:
                velocity.append(0.0)
            by_sample[sample_token].append(
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
    return by_sample, data.get("meta", {})


def load_sample_token_order(path: Path) -> list[str]:
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


def weighted_mean(values: list[list[float]], weights: list[float]) -> list[float]:
    denom = sum(weights)
    if denom <= 0:
        denom = float(len(values))
        weights = [1.0] * len(values)
    width = len(values[0])
    return [
        sum(float(value[i]) * weight for value, weight in zip(values, weights)) / denom
        for i in range(width)
    ]


def fuse_cluster(
    boxes: list[Box],
    *,
    gamma: float,
    tau: float,
    score_power: float,
    score_cap: float | None,
    score_mode: str,
    support_mode: str,
    single_candidate_score_weight: float = 1.0,
) -> dict[str, Any]:
    score_weights = [
        max(0.0, box.score_weight) * max(1e-6, box.score) ** score_power
        for box in boxes
    ]
    geometry_boxes = [
        box for box in boxes if box.mode != "candidate_only" and box.geometry_weight > 0
    ]
    geometry_weights = [
        max(0.0, box.geometry_weight) * max(1e-6, box.score) ** score_power
        for box in geometry_boxes
    ]
    if not geometry_boxes:
        geometry_boxes = boxes
        geometry_weights = score_weights

    translation = weighted_mean([box.translation for box in geometry_boxes], geometry_weights)
    size = weighted_mean([box.size for box in geometry_boxes], geometry_weights)

    velocity_boxes = [
        box for box in geometry_boxes if box.mode == "full" and box.geometry_weight > 0
    ]
    velocity_weights = [
        max(0.0, box.geometry_weight) * max(1e-6, box.score) ** score_power
        for box in velocity_boxes
    ]
    if not velocity_boxes:
        velocity_boxes = geometry_boxes
        velocity_weights = geometry_weights
    velocity = weighted_mean([box.velocity for box in velocity_boxes], velocity_weights)

    yaw_sin = 0.0
    yaw_cos = 0.0
    for box, weight in zip(geometry_boxes, geometry_weights):
        yaw = yaw_from_quat(box.rotation)
        yaw_sin += math.sin(yaw) * weight
        yaw_cos += math.cos(yaw) * weight
    yaw = math.atan2(yaw_sin, yaw_cos)

    center_var = 0.0
    denom = sum(score_weights) or float(len(boxes))
    for box, weight in zip(boxes, score_weights):
        dist = center_distance(box.xy, (translation[0], translation[1]))
        center_var += weight * dist * dist
    center_var /= denom

    if support_mode == "group":
        support_count = len({box.group for box in boxes})
    else:
        support_count = len({box.source for box in boxes})
    if score_mode == "max":
        base_score = max(box.score for box in boxes)
    elif score_mode == "mean":
        base_score = sum(box.score * weight for box, weight in zip(boxes, score_weights)) / denom
    else:
        base_score = max(box.weighted_score for box in boxes)
    support_bonus = 1.0 + gamma * math.log1p(support_count)
    agreement_bonus = math.exp(-center_var / max(tau, 1e-6))
    score = base_score * support_bonus * agreement_bonus
    if support_count == 1 and all(box.mode == "candidate_only" for box in boxes):
        score *= single_candidate_score_weight
    if score_cap is not None:
        score = min(score, score_cap)
    score = max(0.0, float(score))

    best_attr = max(boxes, key=lambda box: box.weighted_score).attribute_name
    return {
        "sample_token": boxes[0].sample_token,
        "translation": translation,
        "size": size,
        "rotation": quat_from_yaw(yaw),
        "velocity": velocity,
        "detection_name": boxes[0].detection_name,
        "detection_score": score,
        "attribute_name": best_attr,
    }


def cluster_boxes(
    boxes: list[Box],
    *,
    threshold: float,
) -> list[list[Box]]:
    def cluster_center(cluster: list[Box]) -> tuple[float, float]:
        geometry_items = [
            item for item in cluster if item.mode != "candidate_only" and item.weighted_geometry_score > 0
        ]
        if geometry_items:
            weights = [max(1e-6, item.weighted_geometry_score) for item in geometry_items]
            points = [[item.translation[0], item.translation[1]] for item in geometry_items]
        else:
            weights = [max(1e-6, item.weighted_score) for item in cluster]
            points = [[item.translation[0], item.translation[1]] for item in cluster]
        center = weighted_mean(points, weights)
        return float(center[0]), float(center[1])

    clusters: list[list[Box]] = []
    centers: list[tuple[float, float]] = []
    for box in sorted(boxes, key=lambda item: item.weighted_score, reverse=True):
        best_idx = -1
        best_dist = float("inf")
        for idx, center in enumerate(centers):
            dist = center_distance(box.xy, center)
            if dist <= threshold and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx < 0:
            clusters.append([box])
            centers.append(box.xy)
        else:
            clusters[best_idx].append(box)
            centers[best_idx] = cluster_center(clusters[best_idx])
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert",
        action="append",
        type=parse_expert,
        required=True,
        help=(
            "NAME:WEIGHT:artifacts/results_nusc.json or "
            "NAME:SCORE_WEIGHT:GEOMETRY_WEIGHT:GROUP:MODE:artifacts/results_nusc.json. "
            "MODE is full, no_velocity, or candidate_only. May be repeated."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--sample-token-file",
        type=Path,
        help="Optional exact output token order; every expert must have this coverage.",
    )
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=("weighted_max", "max", "mean"), default="weighted_max")
    parser.add_argument("--support-mode", choices=("source", "group"), default="source")
    parser.add_argument("--score-cap", type=float, default=1.0)
    parser.add_argument("--no-score-cap", action="store_true")
    parser.add_argument(
        "--single-candidate-score-weight",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier for single-support clusters made only from candidate_only boxes. "
            "Use this for unmatched candidate retrieval alpha; primary full boxes are unaffected."
        ),
    )
    parser.add_argument("--max-per-sample", type=int, default=500)
    parser.add_argument("--default-threshold", type=float, default=0.7)
    parser.add_argument(
        "--class-threshold",
        action="append",
        type=parse_class_threshold,
        default=[],
        help="Override clustering threshold, e.g. car=0.7. May be repeated.",
    )
    args = parser.parse_args()
    if args.gamma < 0:
        raise SystemExit("--gamma must be >= 0")
    if args.tau <= 0:
        raise SystemExit("--tau must be > 0")
    if args.score_power < 0:
        raise SystemExit("--score-power must be >= 0")
    if args.single_candidate_score_weight < 0:
        raise SystemExit("--single-candidate-score-weight must be >= 0")
    if args.max_per_sample <= 0:
        raise SystemExit("--max-per-sample must be > 0")

    thresholds = dict(DEFAULT_CLUSTER_THRESHOLDS)
    thresholds.update(dict(args.class_threshold))

    loaded: list[tuple[Expert, dict[str, list[Box]], dict[str, Any]]] = []
    all_tokens: set[str] = set()
    meta: dict[str, Any] = {}
    for expert in args.expert:
        by_sample, expert_meta = load_boxes(expert)
        loaded.append((expert, by_sample, expert_meta))
        all_tokens.update(by_sample)
        if not meta and isinstance(expert_meta, dict):
            meta = dict(expert_meta)

    if args.sample_token_file is not None:
        if not args.sample_token_file.is_file():
            raise SystemExit(f"missing sample token file: {args.sample_token_file}")
        ordered_tokens = load_sample_token_order(args.sample_token_file)
        expected_token_set = set(ordered_tokens)
        for expert, by_sample, _expert_meta in loaded:
            missing = expected_token_set - set(by_sample)
            extra = set(by_sample) - expected_token_set
            if missing or extra:
                raise SystemExit(
                    f"expert token coverage differs from --sample-token-file for {expert.name}: "
                    f"missing={len(missing)} extra={len(extra)}"
                )
        if all_tokens != expected_token_set:
            raise SystemExit("combined expert token coverage differs from --sample-token-file")
    else:
        ordered_tokens = sorted(all_tokens)

    fused_results: dict[str, list[dict[str, Any]]] = {}
    support_hist: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()
    single_candidate_scaled_count = 0
    input_counts = {expert.name: sum(len(rows) for rows in by_sample.values()) for expert, by_sample, _ in loaded}

    for sample_token in ordered_tokens:
        by_class: dict[str, list[Box]] = defaultdict(list)
        for _, sample_boxes, _ in loaded:
            for box in sample_boxes.get(sample_token, []):
                by_class[box.detection_name].append(box)

        fused_rows: list[dict[str, Any]] = []
        for class_name, class_boxes in by_class.items():
            threshold = thresholds.get(class_name, args.default_threshold)
            for cluster in cluster_boxes(class_boxes, threshold=threshold):
                if args.support_mode == "group":
                    support_count = len({box.group for box in cluster})
                else:
                    support_count = len({box.source for box in cluster})
                support_hist[support_count] += 1
                if support_count == 1 and all(box.mode == "candidate_only" for box in cluster):
                    single_candidate_scaled_count += 1
                class_counts[class_name] += 1
                fused_rows.append(
                    fuse_cluster(
                        cluster,
                        gamma=args.gamma,
                        tau=args.tau,
                        score_power=args.score_power,
                        score_cap=None if args.no_score_cap else float(args.score_cap),
                        score_mode=args.score_mode,
                        support_mode=args.support_mode,
                        single_candidate_score_weight=args.single_candidate_score_weight,
                    )
                )

        fused_rows.sort(key=lambda row: float(row["detection_score"]), reverse=True)
        fused_results[sample_token] = fused_rows[: args.max_per_sample]

    output = {
        "meta": meta,
        "results": fused_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    summary = {
        "out": str(args.out),
        "experts": [
            {
                "name": expert.name,
                "score_weight": expert.score_weight,
                "geometry_weight": expert.geometry_weight,
                "group": expert.group,
                "mode": expert.mode,
                "path": str(expert.path),
            }
            for expert in args.expert
        ],
        "gamma": args.gamma,
        "tau": args.tau,
        "score_power": args.score_power,
        "score_mode": args.score_mode,
        "support_mode": args.support_mode,
        "score_cap": None if args.no_score_cap else args.score_cap,
        "single_candidate_score_weight": args.single_candidate_score_weight,
        "single_candidate_scaled_count": single_candidate_scaled_count,
        "max_per_sample": args.max_per_sample,
        "sample_token_file": (
            str(args.sample_token_file) if args.sample_token_file is not None else None
        ),
        "ordered_sample_tokens": list(fused_results) == ordered_tokens,
        "cluster_thresholds": thresholds,
        "sample_count": len(fused_results),
        "input_box_counts": input_counts,
        "output_box_count": sum(len(rows) for rows in fused_results.values()),
        "support_histogram": {str(key): value for key, value in sorted(support_hist.items())},
        "class_counts": dict(sorted(class_counts.items())),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
