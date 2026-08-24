#!/usr/bin/env python3
"""Train a lightweight BGE-AF score arbiter from a JSONL feature cache.

The linear and MLP models intentionally use only the Python standard library.
The LightGBM branch is optional and runs only when the runtime provides the
`lightgbm` package.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


EPS = 1e-9
MODEL_SCHEMA = "bge_af_linear_arbiter_v1"
MLP_MODEL_SCHEMA = "bge_af_mlp_arbiter_v1"
LIGHTGBM_MODEL_SCHEMA = "bge_af_lightgbm_arbiter_v1"


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


def stable_seed_int(seed: str) -> int:
    return int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12], 16)


def stable_holdout(token: str, fraction: float, seed: str) -> bool:
    if fraction <= 0:
        return False
    digest = hashlib.sha1(f"{seed}:{token}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return value < fraction


def load_rows(path: Path, *, target_threshold: float, holdout_fraction: float, seed: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema_version") != "bge_af_arbiter_cache_row_v1":
                continue
            target = row.get("target") or {}
            quality = target.get("quality")
            if quality is None:
                continue
            quality = min(max(_safe_float(quality), 0.0), 1.0)
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"{path}:{line_no}: missing features dict")
            categorical = row.get("categorical")
            if not isinstance(categorical, dict):
                categorical = {}
            token = str(row.get("sample_token", ""))
            rows.append(
                {
                    "sample_token": token,
                    "class_name": str(row.get("class_name", "")),
                    "features": {str(key): _safe_float(value) for key, value in features.items()},
                    "categorical": {str(key): str(value) for key, value in categorical.items()},
                    "target": quality,
                    "binary_target": 1 if quality >= target_threshold else 0,
                    "base_score": min(max(_safe_float(row.get("base_score")), 0.0), 1.0),
                    "split": "holdout" if stable_holdout(token, holdout_fraction, seed) else "train",
                }
            )
    return rows


def collect_schema(train_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, float]], dict[str, list[str]]]:
    numeric_names = sorted({name for row in train_rows for name in row["features"]})
    stats: dict[str, dict[str, float]] = {}
    for name in numeric_names:
        values = [row["features"].get(name, 0.0) for row in train_rows]
        avg = mean(values) if values else 0.0
        var = mean([(value - avg) ** 2 for value in values]) if values else 0.0
        std = math.sqrt(var)
        if std < 1e-6:
            std = 1.0
        stats[name] = {"mean": avg, "std": std}
    categorical_names = sorted({name for row in train_rows for name in row["categorical"]})
    levels = {
        name: sorted({str(row["categorical"].get(name, "")) for row in train_rows})
        for name in categorical_names
    }
    return numeric_names, stats, levels


def feature_names(numeric_names: list[str], levels: dict[str, list[str]]) -> list[str]:
    names = ["bias"]
    names.extend(f"num:{name}" for name in numeric_names)
    for key in sorted(levels):
        names.extend(f"cat:{key}={value}" for value in levels[key])
    return names


def encode(row: dict[str, Any], numeric_names: list[str], stats: dict[str, dict[str, float]], levels: dict[str, list[str]]) -> list[float]:
    values = [1.0]
    for name in numeric_names:
        stat = stats[name]
        values.append((row["features"].get(name, 0.0) - stat["mean"]) / stat["std"])
    for key in sorted(levels):
        value = str(row["categorical"].get(key, ""))
        values.extend(1.0 if value == level else 0.0 for level in levels[key])
    return values


def make_matrix(rows: list[dict[str, Any]], numeric_names: list[str], stats: dict[str, dict[str, float]], levels: dict[str, list[str]]) -> tuple[list[list[float]], list[float]]:
    return [encode(row, numeric_names, stats, levels) for row in rows], [float(row["target"]) for row in rows]


def weighted_bce(probs: list[float], targets: list[float]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for prob, target in zip(probs, targets):
        prob = min(max(prob, 1e-7), 1.0 - 1e-7)
        target = min(max(target, 0.0), 1.0)
        total += -(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob))
    return total / len(probs)


def train_linear_sigmoid(
    features: list[list[float]],
    targets: list[float],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[list[float], list[dict[str, float]]]:
    if not features:
        raise ValueError("No training rows with target quality")
    dim = len(features[0])
    weights = [0.0] * dim
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        grad = [0.0] * dim
        probs: list[float] = []
        for x, target in zip(features, targets):
            prob = sigmoid(sum(w * v for w, v in zip(weights, x)))
            probs.append(prob)
            diff = prob - target
            for idx, value in enumerate(x):
                grad[idx] += diff * value
        for idx in range(dim):
            penalty = 0.0 if idx == 0 else l2 * weights[idx]
            weights[idx] -= lr * (grad[idx] / len(features) + penalty)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append({"epoch": epoch, "bce": weighted_bce(probs, targets)})
    return weights, history


def train_mlp_sigmoid(
    features: list[list[float]],
    targets: list[float],
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: str,
    init_scale: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if not features:
        raise ValueError("No training rows with target quality")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0")
    dim = len(features[0])
    rng = random.Random(stable_seed_int(seed))
    w1 = [
        [rng.uniform(-init_scale, init_scale) for _ in range(dim)]
        for _ in range(hidden_dim)
    ]
    b1 = [0.0] * hidden_dim
    w2 = [rng.uniform(-init_scale, init_scale) for _ in range(hidden_dim)]
    b2 = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        gw1 = [[0.0] * dim for _ in range(hidden_dim)]
        gb1 = [0.0] * hidden_dim
        gw2 = [0.0] * hidden_dim
        gb2 = 0.0
        probs: list[float] = []
        for x, target in zip(features, targets):
            hidden_raw = [
                b1[j] + sum(w1[j][i] * x[i] for i in range(dim))
                for j in range(hidden_dim)
            ]
            hidden = [math.tanh(value) for value in hidden_raw]
            logit = b2 + sum(w2[j] * hidden[j] for j in range(hidden_dim))
            prob = sigmoid(logit)
            probs.append(prob)
            diff = prob - target
            for j in range(hidden_dim):
                gw2[j] += diff * hidden[j]
            gb2 += diff
            for j in range(hidden_dim):
                d_hidden_raw = diff * w2[j] * (1.0 - hidden[j] ** 2)
                gb1[j] += d_hidden_raw
                for i in range(dim):
                    gw1[j][i] += d_hidden_raw * x[i]
        denom = len(features)
        for j in range(hidden_dim):
            w2[j] -= lr * (gw2[j] / denom + l2 * w2[j])
            for i in range(dim):
                w1[j][i] -= lr * (gw1[j][i] / denom + l2 * w1[j][i])
            b1[j] -= lr * (gb1[j] / denom)
        b2 -= lr * (gb2 / denom)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append({"epoch": epoch, "bce": weighted_bce(probs, targets)})

    return {
        "input_hidden_weights": w1,
        "hidden_bias": b1,
        "output_weights": w2,
        "output_bias": b2,
    }, history


def predict(features: list[list[float]], weights: list[float]) -> list[float]:
    return [sigmoid(sum(w * v for w, v in zip(weights, x))) for x in features]


def predict_mlp(features: list[list[float]], params: dict[str, Any]) -> list[float]:
    w1 = params["input_hidden_weights"]
    b1 = params["hidden_bias"]
    w2 = params["output_weights"]
    b2 = float(params["output_bias"])
    out: list[float] = []
    for x in features:
        hidden = [
            math.tanh(float(b1[j]) + sum(float(w1[j][i]) * x[i] for i in range(len(x))))
            for j in range(len(w2))
        ]
        out.append(sigmoid(b2 + sum(float(w2[j]) * hidden[j] for j in range(len(w2)))))
    return out


def train_lightgbm_regressor(
    features: list[list[float]],
    targets: list[float],
    *,
    estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    l2: float,
    seed: str,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if not features:
        raise ValueError("No training rows with target quality")
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "lightgbm is not installed; install it on the remote environment or use --model-kind linear/mlp"
        ) from exc
    regressor = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_lambda=l2,
        random_state=stable_seed_int(seed) % (2**31 - 1),
        verbosity=-1,
    )
    regressor.fit(features, targets)
    train_scores = [clamp01(value) for value in regressor.predict(features)]
    return {
        "booster": regressor.booster_.model_to_string(),
        "params": {
            "estimators": estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "l2": l2,
            "seed": seed,
        },
    }, [{"epoch": estimators, "bce": weighted_bce(train_scores, targets)}]


def predict_lightgbm(features: list[list[float]], params: dict[str, Any]) -> list[float]:
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("lightgbm is required to apply a LightGBM arbiter") from exc
    booster = lgb.Booster(model_str=str(params["booster"]))
    return [clamp01(value) for value in booster.predict(features)]


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = sum(1 for label in labels if label == 1)
    neg = sum(1 for label in labels if label == 0)
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and ranked[j][0] == ranked[idx][0]:
            j += 1
        avg_rank = (rank + rank + (j - idx) - 1) * 0.5
        for _, label in ranked[idx:j]:
            if label == 1:
                rank_sum += avg_rank
        rank += j - idx
        idx = j
    return (rank_sum - pos * (pos + 1) * 0.5) / (pos * neg)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ece(probs: list[float], targets: list[float], bins: int = 10) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            indices = [i for i, prob in enumerate(probs) if lo <= prob <= hi]
        else:
            indices = [i for i, prob in enumerate(probs) if lo <= prob < hi]
        if indices:
            total += len(indices) / len(probs) * abs(
                mean([probs[i] for i in indices]) - mean([targets[i] for i in indices])
            )
    return total


def brier(probs: list[float], targets: list[float]) -> float:
    if not probs:
        return 0.0
    return mean([(prob - target) ** 2 for prob, target in zip(probs, targets)])


def metric_bundle(rows: list[dict[str, Any]], scores: list[float], *, name: str) -> dict[str, Any]:
    targets = [float(row["target"]) for row in rows]
    labels = [int(row["binary_target"]) for row in rows]
    return {
        "name": name,
        "count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "mean_score": mean(scores) if scores else None,
        "mean_target": mean(targets) if targets else None,
        "auc": auc(scores, labels),
        "pearson": pearson(scores, targets),
        "bce": weighted_bce(scores, targets),
        "brier": brier(scores, targets),
        "ece": ece(scores, targets),
    }


def class_metrics(rows: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_class[str(row["class_name"])].append(idx)
    out = []
    for class_name, indices in sorted(by_class.items()):
        out.append(metric_bundle([rows[i] for i in indices], [scores[i] for i in indices], name=class_name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-jsonl", required=True, type=Path)
    parser.add_argument("--out-model", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="bge-af-v1")
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--model-kind", choices=("linear", "mlp", "lightgbm"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--mlp-init-scale", type=float, default=0.05)
    parser.add_argument("--lightgbm-estimators", type=int, default=80)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=15)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=20)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.lr <= 0:
        raise SystemExit("--lr must be > 0")
    if args.l2 < 0:
        raise SystemExit("--l2 must be >= 0")
    if not 0 <= args.holdout_fraction < 1:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    if args.hidden_dim <= 0:
        raise SystemExit("--hidden-dim must be > 0")
    if args.mlp_init_scale <= 0:
        raise SystemExit("--mlp-init-scale must be > 0")
    if args.lightgbm_estimators <= 0:
        raise SystemExit("--lightgbm-estimators must be > 0")
    if args.lightgbm_learning_rate <= 0:
        raise SystemExit("--lightgbm-learning-rate must be > 0")
    if args.lightgbm_num_leaves <= 1:
        raise SystemExit("--lightgbm-num-leaves must be > 1")
    if args.lightgbm_min_child_samples <= 0:
        raise SystemExit("--lightgbm-min-child-samples must be > 0")

    rows = load_rows(
        args.cache_jsonl,
        target_threshold=args.target_threshold,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    if not train_rows:
        raise SystemExit("cache has no train rows with target. Did you pass --target-result-json when building it?")

    numeric_names, stats, levels = collect_schema(train_rows)
    x_train, y_train = make_matrix(train_rows, numeric_names, stats, levels)
    x_all, _ = make_matrix(rows, numeric_names, stats, levels)
    x_holdout, _ = make_matrix(holdout_rows, numeric_names, stats, levels) if holdout_rows else ([], [])
    feature_name_list = feature_names(numeric_names, levels)
    if args.model_kind == "linear":
        weights, history = train_linear_sigmoid(x_train, y_train, epochs=args.epochs, lr=args.lr, l2=args.l2)
        train_scores = predict(x_train, weights)
        all_scores = predict(x_all, weights)
        holdout_scores = predict(x_holdout, weights) if holdout_rows else []
        model = {
            "schema_version": MODEL_SCHEMA,
            "numeric_features": numeric_names,
            "categorical_levels": levels,
            "stats": stats,
            "feature_names": feature_name_list,
            "weights": weights,
            "target_threshold": args.target_threshold,
            "score_blend_default": 1.0,
            "training": {
                "cache_jsonl": str(args.cache_jsonl),
                "model_kind": args.model_kind,
                "epochs": args.epochs,
                "lr": args.lr,
                "l2": args.l2,
                "holdout_fraction": args.holdout_fraction,
                "seed": args.seed,
            },
        }
    elif args.model_kind == "mlp":
        mlp_params, history = train_mlp_sigmoid(
            x_train,
            y_train,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
            init_scale=args.mlp_init_scale,
        )
        train_scores = predict_mlp(x_train, mlp_params)
        all_scores = predict_mlp(x_all, mlp_params)
        holdout_scores = predict_mlp(x_holdout, mlp_params) if holdout_rows else []
        model = {
            "schema_version": MLP_MODEL_SCHEMA,
            "numeric_features": numeric_names,
            "categorical_levels": levels,
            "stats": stats,
            "feature_names": feature_name_list,
            "hidden_dim": args.hidden_dim,
            "activation": "tanh",
            **mlp_params,
            "target_threshold": args.target_threshold,
            "score_blend_default": 1.0,
            "training": {
                "cache_jsonl": str(args.cache_jsonl),
                "model_kind": args.model_kind,
                "epochs": args.epochs,
                "lr": args.lr,
                "l2": args.l2,
                "holdout_fraction": args.holdout_fraction,
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "mlp_init_scale": args.mlp_init_scale,
            },
        }
    else:
        try:
            lightgbm_params, history = train_lightgbm_regressor(
                x_train,
                y_train,
                estimators=args.lightgbm_estimators,
                learning_rate=args.lightgbm_learning_rate,
                num_leaves=args.lightgbm_num_leaves,
                min_child_samples=args.lightgbm_min_child_samples,
                l2=args.l2,
                seed=args.seed,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        train_scores = predict_lightgbm(x_train, lightgbm_params)
        all_scores = predict_lightgbm(x_all, lightgbm_params)
        holdout_scores = predict_lightgbm(x_holdout, lightgbm_params) if holdout_rows else []
        model = {
            "schema_version": LIGHTGBM_MODEL_SCHEMA,
            "numeric_features": numeric_names,
            "categorical_levels": levels,
            "stats": stats,
            "feature_names": feature_name_list,
            "booster": lightgbm_params["booster"],
            "target_threshold": args.target_threshold,
            "score_blend_default": 1.0,
            "training": {
                "cache_jsonl": str(args.cache_jsonl),
                "model_kind": args.model_kind,
                "epochs": args.epochs,
                "lr": args.lr,
                "l2": args.l2,
                "holdout_fraction": args.holdout_fraction,
                "seed": args.seed,
                **lightgbm_params["params"],
            },
        }
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_model.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")

    base_scores = [float(row["base_score"]) for row in rows]
    summary = {
        "schema_version": "bge_af_arbiter_training_summary_v1",
        "cache_jsonl": str(args.cache_jsonl),
        "out_model": str(args.out_model),
        "model_kind": args.model_kind,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "feature_count": len(feature_name_list),
        "history": history,
        "metrics": {
            "base_all": metric_bundle(rows, base_scores, name="base_all"),
            "model_all": metric_bundle(rows, all_scores, name="model_all"),
            "model_train": metric_bundle(train_rows, train_scores, name="model_train"),
            "model_holdout": metric_bundle(holdout_rows, holdout_scores, name="model_holdout") if holdout_rows else None,
        },
        "class_metrics": class_metrics(rows, all_scores),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
