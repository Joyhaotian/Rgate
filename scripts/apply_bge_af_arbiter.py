#!/usr/bin/env python3
"""Apply a BGE-AF arbiter model to a cached cluster table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


LINEAR_MODEL_SCHEMA = "bge_af_linear_arbiter_v1"
MLP_MODEL_SCHEMA = "bge_af_mlp_arbiter_v1"
LIGHTGBM_MODEL_SCHEMA = "bge_af_lightgbm_arbiter_v1"
CALIBRATION_TABLE_SCHEMA = "bge_af_score_calibration_table_v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def calibrate_score_before_cap(*, score: float, temperature: float, power: float) -> float:
    out = clamp01(score)
    if temperature != 1.0:
        out = sigmoid(logit(out) / temperature)
    if power != 1.0:
        out = out ** power
    return clamp01(out)


def calibrate_score(*, score: float, temperature: float, power: float, cap: float) -> float:
    """Apply monotonic score calibration to the learned arbiter probability."""

    return min(calibrate_score_before_cap(score=score, temperature=temperature, power=power), cap)


def _format_range_edge(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


def normalize_range_edges(values: Any) -> list[float]:
    if not isinstance(values, list):
        return [0.0, 20.0, 40.0, 60.0]
    edges = sorted({_safe_float(value) for value in values if math.isfinite(_safe_float(value))})
    return edges if edges else [0.0, 20.0, 40.0, 60.0]


def range_bin_for_value(value: float, edges: list[float]) -> str:
    edges = normalize_range_edges(edges)
    if value < edges[0]:
        return f"lt_{_format_range_edge(edges[0])}"
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper:
            return f"{_format_range_edge(lower)}_{_format_range_edge(upper)}"
    return f"{_format_range_edge(edges[-1])}_inf"


def row_range_xy(row: dict[str, Any]) -> float:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    for key in ("ego_range_xy", "range_xy", "mean_input_range_xy"):
        value = _safe_float(features.get(key), float("nan"))
        if math.isfinite(value):
            return max(0.0, value)
    output_box = row.get("output_box") if isinstance(row.get("output_box"), dict) else {}
    translation = output_box.get("translation")
    if isinstance(translation, list) and len(translation) >= 2:
        return math.hypot(_safe_float(translation[0]), _safe_float(translation[1]))
    return 0.0


def row_class_name(row: dict[str, Any]) -> str:
    output_box = row.get("output_box") if isinstance(row.get("output_box"), dict) else {}
    return str(row.get("class_name") or output_box.get("detection_name") or "unknown")


def row_source_signature(row: dict[str, Any]) -> str:
    categorical = row.get("categorical") if isinstance(row.get("categorical"), dict) else {}
    sources = row.get("sources")
    if row.get("source_signature"):
        return str(row["source_signature"])
    if categorical.get("source_signature"):
        return str(categorical["source_signature"])
    if isinstance(sources, list) and sources:
        return "+".join(str(source) for source in sources)
    return "unknown"


def calibration_key(*, class_name: str, range_bin: str, source_signature: str) -> str:
    return f"class={class_name}|range={range_bin}|source={source_signature}"


def row_calibration_key(row: dict[str, Any], range_edges: list[float]) -> str:
    return calibration_key(
        class_name=row_class_name(row),
        range_bin=range_bin_for_value(row_range_xy(row), range_edges),
        source_signature=row_source_signature(row),
    )


def load_calibration_table(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("schema_version") != CALIBRATION_TABLE_SCHEMA:
        raise SystemExit(f"unsupported calibration table schema: {table.get('schema_version')}")
    bins = table.get("bins")
    if not isinstance(bins, list):
        raise SystemExit("calibration table missing bins list")
    table["_bin_by_key"] = {
        str(item.get("key")): item
        for item in bins
        if isinstance(item, dict) and item.get("key") is not None
    }
    return table


def _calibration_value(block: dict[str, Any], name: str, default: float) -> float:
    value = _safe_float(block.get(name), default)
    if name == "cap":
        return clamp01(value)
    if value <= 0:
        return default
    return value


def calibration_params_for_row(
    *,
    row: dict[str, Any],
    table: dict[str, Any] | None,
    default_temperature: float,
    default_power: float,
    default_cap: float,
) -> tuple[float, float, float, str]:
    if table is None:
        return default_temperature, default_power, default_cap, "cli"
    grouping = table.get("grouping") if isinstance(table.get("grouping"), dict) else {}
    range_edges = normalize_range_edges(grouping.get("range_edges"))
    key = row_calibration_key(row, range_edges)
    bin_by_key = table.get("_bin_by_key") if isinstance(table.get("_bin_by_key"), dict) else {}
    block = bin_by_key.get(key)
    source = "bin"
    if not isinstance(block, dict):
        block = table.get("global") if isinstance(table.get("global"), dict) else {}
        source = "global"
    return (
        _calibration_value(block, "temperature", default_temperature),
        _calibration_value(block, "power", default_power),
        _calibration_value(block, "cap", default_cap),
        source,
    )


def load_cache(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema_version") == "bge_af_arbiter_cache_row_v1":
                rows.append(row)
    return rows


def iter_cache(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema_version") == "bge_af_arbiter_cache_row_v1":
                yield row


def load_sample_token_order(path: Path | None) -> list[str]:
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


def load_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    decoder = json.JSONDecoder()
    prefix = ""
    with path.open("r", encoding="utf-8") as fh:
        for _ in range(32):
            chunk = fh.read(65536)
            if not chunk:
                break
            prefix += chunk
            key_pos = prefix.find('"meta"')
            if key_pos < 0:
                continue
            colon = prefix.find(":", key_pos)
            if colon < 0:
                continue
            try:
                meta, _ = decoder.raw_decode(prefix[colon + 1 :].lstrip())
            except json.JSONDecodeError:
                continue
            return dict(meta) if isinstance(meta, dict) else {}
    return {}


class ScoreStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": self.count,
            "min": self.min_value,
            "mean": self.total / self.count,
            "max": self.max_value,
        }


def load_model(path: Path | None, mode: str) -> dict[str, Any] | None:
    if mode in {"noop", "radar_rule"}:
        return None
    if path is None:
        raise SystemExit("--model is required unless --mode noop or --mode radar_rule")
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") not in {LINEAR_MODEL_SCHEMA, MLP_MODEL_SCHEMA, LIGHTGBM_MODEL_SCHEMA}:
        raise SystemExit(f"unsupported model schema: {model.get('schema_version')}")
    if model.get("schema_version") == LIGHTGBM_MODEL_SCHEMA:
        try:
            import lightgbm as lgb  # type: ignore[import-not-found]
        except Exception as exc:
            raise SystemExit("lightgbm is required to apply a LightGBM arbiter") from exc
        model["_lightgbm_booster"] = lgb.Booster(model_str=str(model.get("booster", "")))
    return model


def encode(row: dict[str, Any], model: dict[str, Any]) -> list[float]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    categorical = row.get("categorical") if isinstance(row.get("categorical"), dict) else {}
    values = [1.0]
    stats = model["stats"]
    for name in model["numeric_features"]:
        stat = stats[name]
        raw = _safe_float(features.get(name, 0.0))
        values.append((raw - _safe_float(stat.get("mean"))) / max(_safe_float(stat.get("std"), 1.0), 1e-6))
    levels = model["categorical_levels"]
    for key in sorted(levels):
        value = str(categorical.get(key, ""))
        values.extend(1.0 if value == level else 0.0 for level in levels[key])
    return values


def model_score(row: dict[str, Any], model: dict[str, Any]) -> float:
    x = encode(row, model)
    if model.get("schema_version") == LINEAR_MODEL_SCHEMA:
        weights = [_safe_float(value) for value in model["weights"]]
        return sigmoid(sum(weight * value for weight, value in zip(weights, x)))
    if model.get("schema_version") == MLP_MODEL_SCHEMA:
        w1 = model["input_hidden_weights"]
        b1 = model["hidden_bias"]
        w2 = [_safe_float(value) for value in model["output_weights"]]
        b2 = _safe_float(model.get("output_bias"))
        hidden = []
        for j in range(len(w2)):
            hidden.append(
                math.tanh(
                    _safe_float(b1[j])
                    + sum(_safe_float(w1[j][i]) * x[i] for i in range(len(x)))
                )
            )
        return sigmoid(b2 + sum(weight * value for weight, value in zip(w2, hidden)))
    if model.get("schema_version") == LIGHTGBM_MODEL_SCHEMA:
        booster = model.get("_lightgbm_booster")
        if booster is None:
            raise SystemExit("LightGBM booster was not initialized")
        return clamp01(float(booster.predict([x])[0]))
    raise SystemExit(f"unsupported model schema: {model.get('schema_version')}")


def resolve_blend(*, mode: str, score_blend: float | None, model: dict[str, Any] | None) -> float:
    blend = score_blend
    if blend is None:
        blend = 0.0 if mode == "noop" else 1.0 if mode == "radar_rule" else _safe_float(model.get("score_blend_default", 1.0), 1.0)
    if mode == "model":
        blend = 1.0
    return blend


def radar_rule_score(row: dict[str, Any], base_score: float, args: argparse.Namespace) -> tuple[float, float]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    point_count = max(0.0, _safe_float(features.get("radar_point_count")))
    point_signal = min(math.log1p(point_count) / math.log1p(max(args.radar_rule_support_saturation, 1.0)), 1.0)
    velocity_delta = max(0.0, _safe_float(features.get("radar_velocity_delta_mean")))
    velocity_signal = point_signal * math.exp(-velocity_delta / max(args.radar_rule_velocity_tau, 1e-6))
    min_dist = max(0.0, _safe_float(features.get("radar_min_center_dist")))
    distance_signal = point_signal * math.exp(-min_dist / max(args.radar_rule_distance_tau, 1e-6))
    bonus = (
        args.radar_rule_support_scale * point_signal
        + args.radar_rule_velocity_scale * velocity_signal
        + args.radar_rule_distance_scale * distance_signal
    )
    if point_count <= 0:
        bonus -= args.radar_rule_no_support_penalty
    bonus = min(max(bonus, -args.radar_rule_max_penalty), args.radar_rule_max_bonus)
    return clamp01(base_score * (1.0 + bonus)), bonus


def score_row(
    *,
    row: dict[str, Any],
    row_idx: int,
    model: dict[str, Any] | None,
    calibration_table: dict[str, Any] | None,
    mode: str,
    blend: float,
    score_temperature: float,
    score_power: float,
    score_cap: float,
    args: argparse.Namespace,
) -> tuple[str, tuple[int, dict[str, Any]], float, float, float, float, bool, str]:
    sample_token = str(row.get("sample_token", ""))
    output_box = dict(row.get("output_box") or {})
    base_score = clamp01(_safe_float(row.get("base_score", output_box.get("detection_score"))))
    cap_hit = False
    calibration_source = "inactive"
    if mode == "noop":
        raw_arbiter = base_score
        calibrated_arbiter = base_score
    elif mode == "radar_rule":
        calibrated_arbiter, _bonus = radar_rule_score(row, base_score, args)
        raw_arbiter = calibrated_arbiter
    else:
        assert model is not None
        raw_arbiter = model_score(row, model)
        temperature, power, cap, calibration_source = calibration_params_for_row(
            row=row,
            table=calibration_table,
            default_temperature=score_temperature,
            default_power=score_power,
            default_cap=score_cap,
        )
        before_cap = calibrate_score_before_cap(
            score=raw_arbiter,
            temperature=temperature,
            power=power,
        )
        cap_hit = before_cap > cap
        calibrated_arbiter = min(before_cap, cap)
    score = base_score if mode == "noop" else (1.0 - blend) * base_score + blend * calibrated_arbiter
    score = clamp01(score)
    output_box["sample_token"] = sample_token
    output_box["detection_score"] = score
    return sample_token, (row_idx, output_box), base_score, raw_arbiter, calibrated_arbiter, score, cap_hit, calibration_source


def _xy_from_translation(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return _safe_float(value[0]), _safe_float(value[1])


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def refine_output_box_center(
    *,
    row: dict[str, Any],
    output_box: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    if args.center_refine_mode == "off":
        return 0.0
    cluster_boxes = row.get("cluster_boxes")
    if not isinstance(cluster_boxes, list) or len(cluster_boxes) < 2:
        return 0.0
    groups = {
        str(box.get("group", ""))
        for box in cluster_boxes
        if isinstance(box, dict) and str(box.get("group", ""))
    }
    if len(groups) < args.center_refine_min_support_groups:
        return 0.0
    translation = output_box.get("translation")
    base_xy = _xy_from_translation(translation)
    if base_xy is None:
        return 0.0

    points: list[tuple[float, float]] = []
    weights: list[float] = []
    candidate_used = False
    for box in cluster_boxes:
        if not isinstance(box, dict):
            continue
        xy = _xy_from_translation(box.get("translation"))
        if xy is None:
            continue
        mode = str(box.get("mode", ""))
        score = max(1e-6, _safe_float(box.get("score")))
        if mode == "candidate_only":
            if _xy_distance(xy, base_xy) > args.center_refine_candidate_max_dist:
                continue
            weight = args.center_refine_candidate_weight * max(0.0, _safe_float(box.get("score_weight"))) * score
            candidate_used = True
        else:
            weight = max(0.0, _safe_float(box.get("geometry_weight"))) * score
        if weight <= 0:
            continue
        points.append(xy)
        weights.append(weight)

    if not candidate_used or len(points) < 2:
        return 0.0
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    new_xy = (
        sum(point[0] * weight for point, weight in zip(points, weights)) / denom,
        sum(point[1] * weight for point, weight in zip(points, weights)) / denom,
    )
    shift = _xy_distance(base_xy, new_xy)
    if shift <= 0 or shift > args.center_refine_max_shift:
        return 0.0
    assert isinstance(translation, list)
    output_box["translation"] = [*translation]
    output_box["translation"][0] = new_xy[0]
    output_box["translation"][1] = new_xy[1]
    return shift


def sorted_sample_rows(token_rows: list[tuple[int, dict[str, Any]]], max_per_sample: int) -> list[dict[str, Any]]:
    token_rows.sort(key=lambda item: (-float(item[1].get("detection_score", 0.0)), item[0]))
    return [box for _, box in token_rows[:max_per_sample]]


def make_summary(
    *,
    args: argparse.Namespace,
    blend: float,
    sample_count: int,
    input_row_count: int,
    output_box_count: int,
    cap_hit_count: int,
    base_stats: ScoreStats,
    raw_arbiter_stats: ScoreStats,
    arbiter_stats: ScoreStats,
    output_stats: ScoreStats,
    stream_output: bool,
    center_refine_update_count: int,
    center_refine_shift_stats: ScoreStats,
    calibration_match_counts: dict[str, int],
) -> dict[str, Any]:
    table_active = args.mode not in {"noop", "radar_rule"} and args.calibration_table is not None
    return {
        "schema_version": "bge_af_arbiter_apply_summary_v1",
        "cache_jsonl": str(args.cache_jsonl),
        "model": str(args.model) if args.model else None,
        "out": str(args.out),
        "mode": args.mode,
        "score_blend": blend,
        "score_temperature": args.score_temperature,
        "score_power": args.score_power,
        "score_cap": args.score_cap,
        "calibration_table": str(args.calibration_table) if args.calibration_table else None,
        "calibration_table_active": table_active,
        "calibration_active": args.mode not in {"noop", "radar_rule"}
        and (
            table_active
            or args.score_temperature != 1.0
            or args.score_power != 1.0
            or args.score_cap < 1.0
        ),
        "calibration_match_counts": dict(sorted(calibration_match_counts.items())),
        "radar_rule": {
            "active": args.mode == "radar_rule",
            "support_scale": getattr(args, "radar_rule_support_scale", None),
            "velocity_scale": getattr(args, "radar_rule_velocity_scale", None),
            "distance_scale": getattr(args, "radar_rule_distance_scale", None),
            "no_support_penalty": getattr(args, "radar_rule_no_support_penalty", None),
            "max_bonus": getattr(args, "radar_rule_max_bonus", None),
            "max_penalty": getattr(args, "radar_rule_max_penalty", None),
            "support_saturation": getattr(args, "radar_rule_support_saturation", None),
            "velocity_tau": getattr(args, "radar_rule_velocity_tau", None),
            "distance_tau": getattr(args, "radar_rule_distance_tau", None),
        },
        "cap_hit_count": cap_hit_count,
        "cap_hit_rate": cap_hit_count / max(input_row_count, 1),
        "tie_break": "score_desc_then_cache_row_order",
        "stream_output": stream_output,
        "sample_count": sample_count,
        "input_row_count": input_row_count,
        "output_box_count": output_box_count,
        "base_score_stats": base_stats.as_dict(),
        "raw_arbiter_score_stats": raw_arbiter_stats.as_dict(),
        "arbiter_score_stats": arbiter_stats.as_dict(),
        "output_score_stats": output_stats.as_dict(),
        "center_refine_mode": args.center_refine_mode,
        "center_refine_update_count": center_refine_update_count,
        "center_refine_update_rate": center_refine_update_count / max(input_row_count, 1),
        "center_refine_shift_stats": center_refine_shift_stats.as_dict(),
    }


def apply_in_memory(
    args: argparse.Namespace,
    model: dict[str, Any] | None,
    calibration_table: dict[str, Any] | None,
    blend: float,
) -> dict[str, Any]:
    rows = load_cache(args.cache_jsonl)
    by_sample: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    base_stats = ScoreStats()
    raw_arbiter_stats = ScoreStats()
    arbiter_stats = ScoreStats()
    output_stats = ScoreStats()
    center_refine_shift_stats = ScoreStats()
    calibration_match_counts: dict[str, int] = defaultdict(int)
    cap_hit_count = 0
    center_refine_update_count = 0
    for row_idx, row in enumerate(rows):
        sample_token, item, base_score, raw_arbiter, calibrated_arbiter, score, cap_hit, calibration_source = score_row(
            row=row,
            row_idx=row_idx,
            model=model,
            calibration_table=calibration_table,
            mode=args.mode,
            blend=blend,
            score_temperature=args.score_temperature,
            score_power=args.score_power,
            score_cap=args.score_cap,
            args=args,
        )
        shift = refine_output_box_center(row=row, output_box=item[1], args=args)
        if shift > 0:
            center_refine_shift_stats.add(shift)
            center_refine_update_count += 1
        by_sample[sample_token].append(item)
        base_stats.add(base_score)
        raw_arbiter_stats.add(raw_arbiter)
        arbiter_stats.add(calibrated_arbiter)
        output_stats.add(score)
        cap_hit_count += int(cap_hit)
        calibration_match_counts[calibration_source] += 1

    ordered_tokens = load_sample_token_order(args.sample_token_file)
    if ordered_tokens:
        extra_tokens = set(by_sample) - set(ordered_tokens)
        if extra_tokens:
            raise SystemExit(
                "cache contains tokens outside --sample-token-file: "
                + str(len(extra_tokens))
            )
    else:
        ordered_tokens = sorted(by_sample)
    results: dict[str, list[dict[str, Any]]] = {}
    output_box_count = 0
    for token in ordered_tokens:
        token_rows = by_sample.get(token, [])
        sample_rows = sorted_sample_rows(token_rows, args.max_per_sample)
        results[token] = sample_rows
        output_box_count += len(sample_rows)

    output = {
        "meta": load_meta(args.meta_from_result_json),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    return make_summary(
        args=args,
        blend=blend,
        sample_count=len(results),
        input_row_count=len(rows),
        output_box_count=output_box_count,
        cap_hit_count=cap_hit_count,
        base_stats=base_stats,
        raw_arbiter_stats=raw_arbiter_stats,
        arbiter_stats=arbiter_stats,
        output_stats=output_stats,
        stream_output=False,
        center_refine_update_count=center_refine_update_count,
        center_refine_shift_stats=center_refine_shift_stats,
        calibration_match_counts=calibration_match_counts,
    )


def write_sample_entry(fh, *, first_sample: bool, token: str, rows: list[dict[str, Any]]) -> None:
    if not first_sample:
        fh.write(",")
    fh.write(json.dumps(str(token), ensure_ascii=False))
    fh.write(":")
    fh.write(json.dumps(rows, ensure_ascii=False))


def apply_streaming(
    args: argparse.Namespace,
    model: dict[str, Any] | None,
    calibration_table: dict[str, Any] | None,
    blend: float,
) -> dict[str, Any]:
    base_stats = ScoreStats()
    raw_arbiter_stats = ScoreStats()
    arbiter_stats = ScoreStats()
    output_stats = ScoreStats()
    center_refine_shift_stats = ScoreStats()
    calibration_match_counts: dict[str, int] = defaultdict(int)
    cap_hit_count = 0
    center_refine_update_count = 0
    input_row_count = 0
    output_box_count = 0
    sample_count = 0
    current_token: str | None = None
    current_rows: list[tuple[int, dict[str, Any]]] = []
    closed_tokens: set[str] = set()
    ordered_tokens = load_sample_token_order(args.sample_token_file)
    token_positions = {token: index for index, token in enumerate(ordered_tokens)}
    next_expected_index = 0

    def flush_current(fh, first: bool) -> bool:
        nonlocal current_token, current_rows, output_box_count, sample_count
        if current_token is None:
            return first
        sample_rows = sorted_sample_rows(current_rows, args.max_per_sample)
        write_sample_entry(fh, first_sample=first, token=current_token, rows=sample_rows)
        output_box_count += len(sample_rows)
        sample_count += 1
        closed_tokens.add(current_token)
        current_token = None
        current_rows = []
        return False

    def begin_expected_token(fh, first: bool, token: str) -> bool:
        nonlocal next_expected_index, sample_count
        if not ordered_tokens:
            return first
        position = token_positions.get(token)
        if position is None:
            raise SystemExit(
                "cache contains a token outside --sample-token-file: " + token
            )
        if position < next_expected_index:
            raise SystemExit(
                "cache token order differs from --sample-token-file at " + token
            )
        while next_expected_index < position:
            write_sample_entry(
                fh,
                first_sample=first,
                token=ordered_tokens[next_expected_index],
                rows=[],
            )
            first = False
            sample_count += 1
            next_expected_index += 1
        next_expected_index = position + 1
        return first

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write('{"meta":')
        fh.write(json.dumps(load_meta(args.meta_from_result_json), ensure_ascii=False))
        fh.write(',"results":{')
        first_sample = True
        for row_idx, row in enumerate(iter_cache(args.cache_jsonl)):
            sample_token, item, base_score, raw_arbiter, calibrated_arbiter, score, cap_hit, calibration_source = score_row(
                row=row,
                row_idx=row_idx,
                model=model,
                calibration_table=calibration_table,
                mode=args.mode,
                blend=blend,
                score_temperature=args.score_temperature,
                score_power=args.score_power,
                score_cap=args.score_cap,
                args=args,
            )
            if current_token is None:
                if sample_token in closed_tokens:
                    raise SystemExit(
                        "--stream-output requires cache rows grouped by sample_token; "
                        f"sample {sample_token!r} reappeared"
                    )
                first_sample = begin_expected_token(fh, first_sample, sample_token)
                current_token = sample_token
            elif sample_token != current_token:
                first_sample = flush_current(fh, first_sample)
                if sample_token in closed_tokens:
                    raise SystemExit(
                        "--stream-output requires cache rows grouped by sample_token; "
                        f"sample {sample_token!r} reappeared"
                    )
                first_sample = begin_expected_token(fh, first_sample, sample_token)
                current_token = sample_token
            shift = refine_output_box_center(row=row, output_box=item[1], args=args)
            if shift > 0:
                center_refine_shift_stats.add(shift)
                center_refine_update_count += 1
            current_rows.append(item)
            base_stats.add(base_score)
            raw_arbiter_stats.add(raw_arbiter)
            arbiter_stats.add(calibrated_arbiter)
            output_stats.add(score)
            cap_hit_count += int(cap_hit)
            calibration_match_counts[calibration_source] += 1
            input_row_count += 1
        first_sample = flush_current(fh, first_sample)
        while next_expected_index < len(ordered_tokens):
            write_sample_entry(
                fh,
                first_sample=first_sample,
                token=ordered_tokens[next_expected_index],
                rows=[],
            )
            first_sample = False
            sample_count += 1
            next_expected_index += 1
        fh.write("}}")

    return make_summary(
        args=args,
        blend=blend,
        sample_count=sample_count,
        input_row_count=input_row_count,
        output_box_count=output_box_count,
        cap_hit_count=cap_hit_count,
        base_stats=base_stats,
        raw_arbiter_stats=raw_arbiter_stats,
        arbiter_stats=arbiter_stats,
        output_stats=output_stats,
        stream_output=True,
        center_refine_update_count=center_refine_update_count,
        center_refine_shift_stats=center_refine_shift_stats,
        calibration_match_counts=calibration_match_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-jsonl", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--meta-from-result-json", type=Path)
    parser.add_argument(
        "--sample-token-file",
        type=Path,
        help="Optional exact output token order; missing cache groups become empty arrays.",
    )
    parser.add_argument("--mode", choices=("noop", "model", "blend", "radar_rule"), default="model")
    parser.add_argument("--score-blend", type=float, default=None)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--score-cap", type=float, default=1.0)
    parser.add_argument(
        "--calibration-table",
        type=Path,
        help="Optional per class/range/source calibration table from fit_bge_af_score_calibration.py.",
    )
    parser.add_argument("--max-per-sample", type=int, default=500)
    parser.add_argument("--center-refine-mode", choices=("off", "candidate_compact"), default="off")
    parser.add_argument("--center-refine-min-support-groups", type=int, default=2)
    parser.add_argument("--center-refine-candidate-max-dist", type=float, default=0.75)
    parser.add_argument("--center-refine-candidate-weight", type=float, default=0.25)
    parser.add_argument("--center-refine-max-shift", type=float, default=0.5)
    parser.add_argument("--radar-rule-support-scale", type=float, default=0.03)
    parser.add_argument("--radar-rule-velocity-scale", type=float, default=0.02)
    parser.add_argument("--radar-rule-distance-scale", type=float, default=0.01)
    parser.add_argument("--radar-rule-no-support-penalty", type=float, default=0.01)
    parser.add_argument("--radar-rule-max-bonus", type=float, default=0.06)
    parser.add_argument("--radar-rule-max-penalty", type=float, default=0.02)
    parser.add_argument("--radar-rule-support-saturation", type=float, default=3.0)
    parser.add_argument("--radar-rule-velocity-tau", type=float, default=1.0)
    parser.add_argument("--radar-rule-distance-tau", type=float, default=2.0)
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream result JSON writing for large grouped caches. Requires cache rows grouped by sample_token.",
    )
    args = parser.parse_args()

    if args.max_per_sample <= 0:
        raise SystemExit("--max-per-sample must be > 0")
    if args.score_blend is not None and not 0 <= args.score_blend <= 1:
        raise SystemExit("--score-blend must be in [0, 1]")
    if args.score_temperature <= 0:
        raise SystemExit("--score-temperature must be > 0")
    if args.score_power <= 0:
        raise SystemExit("--score-power must be > 0")
    if not 0 <= args.score_cap <= 1:
        raise SystemExit("--score-cap must be in [0, 1]")
    if args.center_refine_min_support_groups <= 0:
        raise SystemExit("--center-refine-min-support-groups must be > 0")
    if args.center_refine_candidate_max_dist <= 0:
        raise SystemExit("--center-refine-candidate-max-dist must be > 0")
    if args.center_refine_candidate_weight < 0:
        raise SystemExit("--center-refine-candidate-weight must be >= 0")
    if args.center_refine_max_shift <= 0:
        raise SystemExit("--center-refine-max-shift must be > 0")
    if args.radar_rule_support_scale < 0:
        raise SystemExit("--radar-rule-support-scale must be >= 0")
    if args.radar_rule_velocity_scale < 0:
        raise SystemExit("--radar-rule-velocity-scale must be >= 0")
    if args.radar_rule_distance_scale < 0:
        raise SystemExit("--radar-rule-distance-scale must be >= 0")
    if args.radar_rule_no_support_penalty < 0:
        raise SystemExit("--radar-rule-no-support-penalty must be >= 0")
    if args.radar_rule_max_bonus < 0:
        raise SystemExit("--radar-rule-max-bonus must be >= 0")
    if args.radar_rule_max_penalty < 0:
        raise SystemExit("--radar-rule-max-penalty must be >= 0")
    if args.radar_rule_support_saturation <= 0:
        raise SystemExit("--radar-rule-support-saturation must be > 0")
    if args.radar_rule_velocity_tau <= 0:
        raise SystemExit("--radar-rule-velocity-tau must be > 0")
    if args.radar_rule_distance_tau <= 0:
        raise SystemExit("--radar-rule-distance-tau must be > 0")
    if args.sample_token_file is not None and not args.sample_token_file.is_file():
        raise SystemExit(f"missing sample token file: {args.sample_token_file}")

    model = load_model(args.model, args.mode)
    calibration_table = load_calibration_table(args.calibration_table)
    blend = resolve_blend(mode=args.mode, score_blend=args.score_blend, model=model)
    summary = (
        apply_streaming(args, model, calibration_table, blend)
        if args.stream_output
        else apply_in_memory(args, model, calibration_table, blend)
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
