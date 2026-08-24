#!/usr/bin/env python3
"""Preflight or run the official nuScenes detection evaluator.

The project already exports nuScenes-style detection JSON. This tool is the
next gate: it checks that the official devkit and local tables are available,
validates candidate submissions, and can run ``NuScenesEval`` when requested.
It can also ingest an existing official ``metrics_summary.json`` produced on a
machine with the devkit installed.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_NUSC_ROOT = "data/nuscenes_raw"
DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_EVAL_SET = "val"
DEFAULT_CONFIG = "detection_cvpr_2019"
DEFAULT_REQUIRED_MODULES = ("nuscenes",)
REQUIRED_TABLES = (
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
)
REQUIRED_BOX_FIELDS = (
    "translation",
    "size",
    "rotation",
    "velocity",
    "detection_name",
    "detection_score",
    "attribute_name",
)


def _fail(message: str, code: int = 1) -> int:
    print(f"[nuscenes_official_detection_eval] {message}", file=sys.stderr)
    return code


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path_str: str | None, root: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _slug(label: str) -> str:
    out = []
    for char in label.strip().lower():
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "candidate"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _try_load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        return _load_json(path), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_labeled_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"{option} must use label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"{option} label is empty")
    if not path:
        raise ValueError(f"{option} path is empty")
    return label, Path(path)


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_vector(value: Any, length: int, *, positive: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    for item in value:
        if not _is_number(item):
            return False
        if positive and float(item) <= 0.0:
            return False
    return True


def _field_rate(passed: int, total: int) -> float | None:
    if total <= 0:
        return None
    return passed / float(total)


def _module_checks(required_modules: Sequence[str]) -> dict[str, Any]:
    missing = []
    import_failed: dict[str, str] = {}
    importable = []
    for raw_name in required_modules:
        name = str(raw_name).strip()
        if not name:
            continue
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - native dependency failures are environment-specific.
            import_failed[name] = str(exc)
        else:
            importable.append(name)
    issues = [f"python module missing: {name}" for name in missing]
    issues.extend(f"python module import failed: {name}: {error}" for name, error in import_failed.items())
    return {
        "ready": not issues,
        "status": "ready" if not issues else "not_ready",
        "required_modules": list(required_modules),
        "importable_modules": importable,
        "missing_modules": missing,
        "failed_modules": import_failed,
        "issues": issues,
    }


def _table_checks(version_dir: Path) -> dict[str, Any]:
    rows = []
    issues = []
    for name in REQUIRED_TABLES:
        path = version_dir / name
        row = {
            "table": name,
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        if not row["exists"]:
            issues.append(f"nuScenes table missing: {path}")
        rows.append(row)
    return {
        "ready": not issues,
        "status": "ready" if not issues else "not_ready",
        "version_dir": str(version_dir),
        "rows": rows,
        "issues": issues,
    }


def _load_sample_tokens(version_dir: Path) -> tuple[set[str], str | None]:
    sample_path = version_dir / "sample.json"
    payload, error = _try_load_json(sample_path)
    if error is not None:
        return set(), error
    if not isinstance(payload, list):
        return set(), "sample.json is not a JSON list"
    tokens = {str(row.get("token")) for row in payload if isinstance(row, dict) and row.get("token")}
    return tokens, None


def _submission_summary(path: Path, known_sample_tokens: set[str]) -> dict[str, Any]:
    payload, error = _try_load_json(path)
    if error is not None:
        return {
            "path": str(path),
            "ready": False,
            "status": "missing_or_unreadable",
            "issues": [f"submission unreadable: {error}"],
            "sample_count": 0,
            "box_count": 0,
        }
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "ready": False,
            "status": "invalid_schema",
            "issues": ["submission is not a JSON object"],
            "sample_count": 0,
            "box_count": 0,
        }
    results = payload.get("results")
    meta = payload.get("meta")
    issues = []
    if not isinstance(results, dict):
        issues.append("results is not an object")
        results = {}
    if not isinstance(meta, dict):
        issues.append("meta is not an object")

    field_pass = {field: 0 for field in REQUIRED_BOX_FIELDS}
    box_count = 0
    invalid_box_count = 0
    unknown_sample_tokens = []
    sample_count = 0
    for sample_token, boxes in results.items():
        sample_count += 1
        if known_sample_tokens and str(sample_token) not in known_sample_tokens:
            unknown_sample_tokens.append(str(sample_token))
        if not isinstance(boxes, list):
            issues.append(f"results[{sample_token!r}] is not a list")
            continue
        for box in boxes:
            box_count += 1
            if not isinstance(box, dict):
                invalid_box_count += 1
                continue
            if _finite_vector(box.get("translation"), 3):
                field_pass["translation"] += 1
            if _finite_vector(box.get("size"), 3, positive=True):
                field_pass["size"] += 1
            if _finite_vector(box.get("rotation"), 4):
                field_pass["rotation"] += 1
            if _finite_vector(box.get("velocity"), 2):
                field_pass["velocity"] += 1
            if isinstance(box.get("detection_name"), str) and box.get("detection_name").strip():
                field_pass["detection_name"] += 1
            if _is_number(box.get("detection_score")):
                field_pass["detection_score"] += 1
            if isinstance(box.get("attribute_name"), str):
                field_pass["attribute_name"] += 1

    if box_count == 0:
        issues.append("submission contains no boxes")
    if invalid_box_count:
        issues.append(f"invalid non-object boxes: {invalid_box_count}")
    if unknown_sample_tokens:
        issues.append(f"unknown sample tokens: {len(unknown_sample_tokens)}")
    field_rates = {field: _field_rate(count, box_count) for field, count in field_pass.items()}
    missing_field_names = [field for field, rate in field_rates.items() if rate != 1.0]
    if missing_field_names:
        issues.append("incomplete box fields: " + ", ".join(missing_field_names))

    ready = not issues
    return {
        "path": str(path),
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "issues": issues,
        "sample_count": sample_count,
        "box_count": box_count,
        "unknown_sample_token_count": len(unknown_sample_tokens),
        "unknown_sample_token_examples": unknown_sample_tokens[:5],
        "field_rates": field_rates,
        "meta": meta if isinstance(meta, dict) else {},
    }


def _metrics_summary(path: Path) -> dict[str, Any]:
    payload, error = _try_load_json(path)
    if error is not None:
        return {
            "path": str(path),
            "ready": False,
            "status": "missing_or_unreadable",
            "issues": [f"metrics_summary unreadable: {error}"],
        }
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "ready": False,
            "status": "invalid_schema",
            "issues": ["metrics_summary is not a JSON object"],
        }
    mean_ap = payload.get("mean_ap")
    nd_score = payload.get("nd_score")
    issues = []
    if not _is_number(mean_ap):
        issues.append("mean_ap missing or non-finite")
    if not _is_number(nd_score):
        issues.append("nd_score missing or non-finite")
    label_aps = payload.get("label_aps")
    tp_errors = payload.get("tp_errors")
    label_tp_errors = payload.get("label_tp_errors")
    if not isinstance(label_aps, dict):
        issues.append("label_aps missing")
    if not isinstance(tp_errors, dict):
        issues.append("tp_errors missing")
    if not isinstance(label_tp_errors, dict):
        issues.append("label_tp_errors missing")
    return {
        "path": str(path),
        "ready": not issues,
        "status": "official_metrics_ready" if not issues else "not_ready",
        "issues": issues,
        "mean_ap": float(mean_ap) if _is_number(mean_ap) else None,
        "nd_score": float(nd_score) if _is_number(nd_score) else None,
        "tp_errors": tp_errors if isinstance(tp_errors, dict) else {},
        "label_count": len(label_aps) if isinstance(label_aps, dict) else 0,
    }


def _run_official_eval(
    *,
    nusc_root: Path,
    version: str,
    eval_set: str,
    config_name: str,
    submission_path: Path,
    output_dir: Path,
    render_curves: bool,
) -> dict[str, Any]:
    try:
        from nuscenes import NuScenes  # type: ignore
        from nuscenes.eval.detection.config import config_factory  # type: ignore
        from nuscenes.eval.detection.evaluate import NuScenesEval  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional external devkit.
        return {
            "ready": False,
            "status": "official_eval_import_failed",
            "issues": [str(exc)],
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        nusc = NuScenes(version=version, dataroot=str(nusc_root), verbose=False)
        config = config_factory(config_name)
        evaluator = NuScenesEval(
            nusc,
            config=config,
            result_path=str(submission_path),
            eval_set=eval_set,
            output_dir=str(output_dir),
            verbose=True,
        )
        try:
            evaluator.main(plot_examples=0, render_curves=render_curves)
        except TypeError:
            evaluator.main(render_curves=render_curves)
    except Exception as exc:  # pragma: no cover - official evaluator runtime depends on dataset/devkit.
        return {
            "ready": False,
            "status": "official_eval_failed",
            "issues": [str(exc)],
            "output_dir": str(output_dir),
        }
    metrics_path = output_dir / "metrics_summary.json"
    metrics = _metrics_summary(metrics_path)
    return {
        **metrics,
        "output_dir": str(output_dir),
        "status": "official_eval_ready" if metrics["ready"] else "official_eval_metrics_missing",
    }


def _best_row(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any] | None:
    best = None
    for row in rows:
        value = row.get(key)
        if not _is_number(value):
            continue
        if best is None or float(value) > float(best.get(key, -1.0)):
            best = row
    return best


def build_report(
    *,
    nusc_root: Path,
    version: str,
    eval_set: str,
    config_name: str,
    candidates: Sequence[tuple[str, Path]],
    output_dir: Path,
    execute: bool = False,
    metrics_summaries: dict[str, Path] | None = None,
    required_modules: Sequence[str] = DEFAULT_REQUIRED_MODULES,
    render_curves: bool = False,
) -> dict[str, Any]:
    version_dir = nusc_root / version
    modules = _module_checks(required_modules)
    tables = _table_checks(version_dir)
    sample_tokens, sample_error = _load_sample_tokens(version_dir)
    table_issues = list(tables["issues"])
    if sample_error is not None:
        table_issues.append(f"sample token load failed: {sample_error}")
    if table_issues:
        tables = {**tables, "ready": False, "status": "not_ready", "issues": table_issues}
    metrics_summaries = metrics_summaries or {}

    rows = []
    for label, submission_path in candidates:
        summary = _submission_summary(submission_path, sample_tokens)
        output_path = output_dir / _slug(label)
        metrics_path = metrics_summaries.get(label)
        metrics = _metrics_summary(metrics_path) if metrics_path is not None else None
        official_eval = metrics
        if execute and summary["ready"] and modules["ready"] and tables["ready"]:
            official_eval = _run_official_eval(
                nusc_root=nusc_root,
                version=version,
                eval_set=eval_set,
                config_name=config_name,
                submission_path=submission_path,
                output_dir=output_path,
                render_curves=render_curves,
            )
        row = {
            "label": label,
            "submission": str(submission_path),
            "submission_ready": summary["ready"],
            "submission_status": summary["status"],
            "submission_issues": summary["issues"],
            "sample_count": summary["sample_count"],
            "box_count": summary["box_count"],
            "field_rates": summary.get("field_rates", {}),
            "output_dir": str(output_path),
            "metrics_summary": official_eval.get("path") if isinstance(official_eval, dict) else None,
            "official_eval_ready": bool(isinstance(official_eval, dict) and official_eval.get("ready") is True),
            "official_eval_status": official_eval.get("status") if isinstance(official_eval, dict) else "not_run",
            "official_eval_issues": official_eval.get("issues", []) if isinstance(official_eval, dict) else [],
            "mean_ap": official_eval.get("mean_ap") if isinstance(official_eval, dict) else None,
            "nd_score": official_eval.get("nd_score") if isinstance(official_eval, dict) else None,
        }
        rows.append(row)

    candidate_issues = []
    for row in rows:
        if not row["submission_ready"]:
            candidate_issues.append(f"{row['label']}: submission not ready")
    preflight_ready = bool(rows) and modules["ready"] and tables["ready"] and not candidate_issues
    official_ready = bool(rows) and all(row["official_eval_ready"] for row in rows)
    if official_ready:
        status = "official_eval_ready"
    elif execute and preflight_ready:
        status = "official_eval_failed"
    elif preflight_ready:
        status = "ready_to_execute"
    else:
        status = "preflight_not_ready"

    issues = []
    issues.extend(modules["issues"])
    issues.extend(tables["issues"])
    issues.extend(candidate_issues)
    if not rows:
        issues.append("at least one candidate is required")
    for row in rows:
        issues.extend(f"{row['label']}: {issue}" for issue in row["official_eval_issues"])

    best_nd = _best_row(rows, "nd_score")
    best_map = _best_row(rows, "mean_ap")
    return {
        "schema_version": "nuscenes_official_detection_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_nds_ready": official_ready,
        "preflight_ready": preflight_ready,
        "execute": bool(execute),
        "nusc_root": str(nusc_root),
        "version": version,
        "eval_set": eval_set,
        "config_name": config_name,
        "output_dir": str(output_dir),
        "modules": modules,
        "tables": tables,
        "sample_token_count": len(sample_tokens),
        "rows": rows,
        "best_nd_score": {
            "label": best_nd.get("label"),
            "value": best_nd.get("nd_score"),
        }
        if best_nd
        else None,
        "best_mean_ap": {
            "label": best_map.get("label"),
            "value": best_map.get("mean_ap"),
        }
        if best_map
        else None,
        "issues": issues,
        "claim_boundary": (
            "Only status=official_eval_ready with metrics_summary.json from the official "
            "nuScenes detection evaluator supports official NDS/mAP claims. A preflight "
            "or ready_to_execute status is not an NDS result."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# nuScenes Official Detection Eval",
        "",
        f"- Status: **{report['status']}**",
        f"- Official NDS ready: **{_fmt(report['official_nds_ready'])}**",
        f"- Eval set: `{report['eval_set']}`",
        f"- Config: `{report['config_name']}`",
        f"- nuScenes root: `{report['nusc_root']}`",
        f"- Sample tokens loaded: `{report['sample_token_count']}`",
        "",
        "## Preconditions",
        "",
        f"- Python modules: `{report['modules']['status']}`",
        f"- nuScenes tables: `{report['tables']['status']}`",
    ]
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| Candidate | Submission Ready | Boxes | Samples | Official Eval | mAP | NDS | Metrics | Output |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {label} | {submission_ready} | {boxes} | {samples} | `{eval_status}` | {map} | {nds} | `{metrics}` | `{output}` |".format(
                label=row["label"],
                submission_ready=_fmt(row["submission_ready"]),
                boxes=_fmt(row["box_count"]),
                samples=_fmt(row["sample_count"]),
                eval_status=row["official_eval_status"],
                map=_fmt(row["mean_ap"]),
                nds=_fmt(row["nd_score"]),
                metrics=row["metrics_summary"] or "-",
                output=row["output_dir"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- A preflight-ready artifact is not an official NDS/mAP result.",
            "- Official claims require `metrics_summary.json` from the nuScenes detection evaluator.",
            "- This tool does not prove CenterFusion parity unless the official scores are compared against a reference-compatible baseline or threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nusc-root", default=DEFAULT_NUSC_ROOT)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", action="append", default=[], help="Label=submission.json")
    parser.add_argument("--metrics-summary", action="append", default=[], help="Label=metrics_summary.json")
    parser.add_argument("--output-dir", default="runs/centerfusion_3d/nuscenes_official_detection_eval")
    parser.add_argument("--out", default="runs/centerfusion_3d/nuscenes_official_detection_eval.json")
    parser.add_argument("--md", default="runs/centerfusion_3d/nuscenes_official_detection_eval.md")
    parser.add_argument("--execute", action="store_true", help="Run NuScenesEval when preflight is ready")
    parser.add_argument("--render-curves", action="store_true")
    args = parser.parse_args(argv)

    root = _repo_root()
    try:
        candidates = [_parse_labeled_path(value, "--candidate") for value in args.candidate]
        metrics = dict(_parse_labeled_path(value, "--metrics-summary") for value in args.metrics_summary)
    except ValueError as exc:
        return _fail(str(exc), 2)
    if not candidates:
        return _fail("at least one --candidate is required", 2)

    nusc_root = _resolve(args.nusc_root, root)
    output_dir = _resolve(args.output_dir, root)
    out_path = _resolve(args.out, root)
    md_path = _resolve(args.md, root)
    assert nusc_root is not None
    assert output_dir is not None
    assert out_path is not None
    assert md_path is not None
    resolved_candidates = [(label, _resolve(str(path), root) or path) for label, path in candidates]
    resolved_metrics = {label: _resolve(str(path), root) or path for label, path in metrics.items()}
    report = build_report(
        nusc_root=nusc_root,
        version=args.version,
        eval_set=args.eval_set,
        config_name=args.config_name,
        candidates=resolved_candidates,
        output_dir=output_dir,
        execute=args.execute,
        metrics_summaries=resolved_metrics,
        render_curves=args.render_curves,
    )
    _write_json(out_path, report)
    _write_text(md_path, render_markdown(report))
    print(f"[nuscenes_official_detection_eval] Wrote {out_path}")
    print(f"[nuscenes_official_detection_eval] Wrote {md_path}")
    if report["status"] in {"official_eval_ready", "ready_to_execute"}:
        return 0
    if report["status"] == "official_eval_failed":
        return 5
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
