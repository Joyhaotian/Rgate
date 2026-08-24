#!/usr/bin/env python3
"""Fit per class/range/source score calibration for the BGE-AF arbiter."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from apply_bge_af_arbiter import (
    CALIBRATION_TABLE_SCHEMA,
    calibrate_score,
    calibration_key,
    clamp01,
    load_model,
    model_score,
    normalize_range_edges,
    range_bin_for_value,
    row_class_name,
    row_range_xy,
    row_source_signature,
    _safe_float,
)


def parse_float_list(text: str, *, name: str) -> list[float]:
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise SystemExit(f"--{name} contains a non-float value: {raw!r}") from exc
        if not math.isfinite(value):
            raise SystemExit(f"--{name} contains a non-finite value: {raw!r}")
        values.append(value)
    if not values:
        raise SystemExit(f"--{name} did not contain any values")
    return values


def row_target_quality(row: dict[str, Any]) -> float | None:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    quality = target.get("quality")
    if quality is None:
        return None
    return clamp01(_safe_float(quality))


def iter_labeled_rows(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema_version") != "bge_af_arbiter_cache_row_v1":
                continue
            target = row_target_quality(row)
            if target is not None:
                yield row, target


def weighted_bce(probs: list[float], targets: list[float]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for prob, target in zip(probs, targets):
        prob = min(max(prob, 1e-7), 1.0 - 1e-7)
        target = clamp01(target)
        total += -(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob))
    return total / len(probs)


def brier(probs: list[float], targets: list[float]) -> float:
    if not probs:
        return 0.0
    return sum((prob - target) ** 2 for prob, target in zip(probs, targets)) / len(probs)


def evaluate_params(
    items: list[tuple[float, float]],
    *,
    temperature: float,
    power: float,
    cap: float,
) -> dict[str, float]:
    probs = [
        calibrate_score(score=score, temperature=temperature, power=power, cap=cap)
        for score, _target in items
    ]
    targets = [target for _score, target in items]
    return {
        "bce": weighted_bce(probs, targets),
        "brier": brier(probs, targets),
        "mean_score": sum(probs) / len(probs) if probs else 0.0,
        "mean_target": sum(targets) / len(targets) if targets else 0.0,
    }


def fit_grid(
    items: list[tuple[float, float]],
    *,
    temperatures: list[float],
    powers: list[float],
    caps: list[float],
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for temperature in temperatures:
        if temperature <= 0:
            continue
        for power in powers:
            if power <= 0:
                continue
            for cap in caps:
                if not 0 <= cap <= 1:
                    continue
                metrics = evaluate_params(
                    items,
                    temperature=temperature,
                    power=power,
                    cap=cap,
                )
                candidate: dict[str, float | int] = {
                    "temperature": float(temperature),
                    "power": float(power),
                    "cap": float(cap),
                    "count": len(items),
                    **metrics,
                }
                if best is None or float(candidate["bce"]) < float(best["bce"]):
                    best = candidate
    if best is None:
        raise SystemExit("calibration grid has no valid parameter candidates")
    return best


def raw_score_for_row(row: dict[str, Any], *, mode: str, model: dict[str, Any] | None) -> float:
    if mode == "noop":
        output_box = row.get("output_box") if isinstance(row.get("output_box"), dict) else {}
        return clamp01(_safe_float(row.get("base_score", output_box.get("detection_score"))))
    if model is None:
        raise SystemExit("--model is required for --mode model")
    return clamp01(model_score(row, model))


def build_table(args: argparse.Namespace) -> dict[str, Any]:
    range_edges = normalize_range_edges(parse_float_list(args.range_edges, name="range-edges"))
    temperatures = parse_float_list(args.temperature_grid, name="temperature-grid")
    powers = parse_float_list(args.power_grid, name="power-grid")
    caps = parse_float_list(args.cap_grid, name="cap-grid")
    model = load_model(args.model, args.mode)

    all_items: list[tuple[float, float]] = []
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    group_meta: dict[str, dict[str, str]] = {}
    row_count = 0
    for row, target in iter_labeled_rows(args.cache_jsonl):
        row_count += 1
        raw_score = raw_score_for_row(row, mode=args.mode, model=model)
        item = (raw_score, target)
        class_name = row_class_name(row)
        range_bin = range_bin_for_value(row_range_xy(row), range_edges)
        source_signature = row_source_signature(row)
        key = calibration_key(
            class_name=class_name,
            range_bin=range_bin,
            source_signature=source_signature,
        )
        all_items.append(item)
        groups[key].append(item)
        group_meta[key] = {
            "class_name": class_name,
            "range_bin": range_bin,
            "source_signature": source_signature,
        }

    if not all_items:
        raise SystemExit(f"{args.cache_jsonl} has no labeled cache rows")

    global_fit = fit_grid(
        all_items,
        temperatures=temperatures,
        powers=powers,
        caps=caps,
    )
    bins: list[dict[str, Any]] = []
    skipped_small_bins = 0
    for key in sorted(groups):
        items = groups[key]
        if len(items) < args.min_bin_rows:
            skipped_small_bins += 1
            continue
        fitted = fit_grid(
            items,
            temperatures=temperatures,
            powers=powers,
            caps=caps,
        )
        global_on_bin = evaluate_params(
            items,
            temperature=float(global_fit["temperature"]),
            power=float(global_fit["power"]),
            cap=float(global_fit["cap"]),
        )
        bins.append(
            {
                "key": key,
                **group_meta[key],
                **fitted,
                "global_bce_on_bin": global_on_bin["bce"],
                "delta_bce_vs_global": global_on_bin["bce"] - float(fitted["bce"]),
            }
        )

    return {
        "schema_version": CALIBRATION_TABLE_SCHEMA,
        "cache_jsonl": str(args.cache_jsonl),
        "model": str(args.model) if args.model else None,
        "mode": args.mode,
        "grouping": {
            "class_key": "class_name",
            "range_edges": range_edges,
            "range_feature_priority": ["ego_range_xy", "range_xy", "mean_input_range_xy", "output_box.translation"],
            "source_key": "source_signature",
            "min_bin_rows": args.min_bin_rows,
        },
        "grid": {
            "temperatures": temperatures,
            "powers": powers,
            "caps": caps,
        },
        "row_count": row_count,
        "labeled_row_count": len(all_items),
        "group_count": len(groups),
        "bin_count": len(bins),
        "skipped_small_bin_count": skipped_small_bins,
        "global": global_fit,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-jsonl", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mode", choices=("model", "noop"), default="model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--range-edges", default="0,20,40,60")
    parser.add_argument("--min-bin-rows", type=int, default=20)
    parser.add_argument("--temperature-grid", default="0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--power-grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--cap-grid", default="0.90,0.95,0.98,1.0")
    args = parser.parse_args()

    if args.min_bin_rows <= 0:
        raise SystemExit("--min-bin-rows must be > 0")
    if args.mode == "model" and args.model is None:
        raise SystemExit("--model is required for --mode model")

    table = build_table(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(table, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
