#!/usr/bin/env python3
"""Publish the registered five-seed R-GATE bootstrap analysis.

This program deliberately performs no model inference and no metric evaluation.  It
consumes only a completed B1000 paired-scene replay and its independent terminal
verification, applies the already-frozen analysis contract, and publishes a small
JSON/Markdown evidence bundle atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Sequence, Tuple


CONFIG_SCHEMA = "rgate_rq2_multiseed_registered_analysis_execution_v1"
CONTRACT_SCHEMA = "rgate_rq2_multiseed_analysis_contract_v1"
BOOTSTRAP_SCHEMA = "nuscenes_paired_scene_bootstrap_report_v1"
BOOTSTRAP_CHECK_SCHEMA = "nuscenes_paired_scene_bootstrap_independent_check_v1"
REPORT_SCHEMA = "rgate_rq2_multiseed_registered_analysis_v1"
STATUS = "passed_registered_multiseed_analysis"
METRICS = ("mean_ap", "nd_score", "mAVE")


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise AnalysisError("non-finite JSON constant: %s" % value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_chain(path: Path) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        if os.path.lexists(str(current)):
            require(not current.is_symlink(), "symlink path component rejected: %s" % current)


def stable_record(path: Path, expected: Mapping[str, Any] = None, expected_sha: str = "") -> Dict[str, Any]:
    logical = path.absolute()
    reject_symlink_chain(logical)
    require(logical.is_file(), "missing input: %s" % logical)
    before = logical.stat()
    digest = sha256_file(logical)
    after = logical.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "input changed while hashing: %s" % logical,
    )
    record = {"path": str(logical), "size_bytes": int(before.st_size), "sha256": digest}
    if expected is not None:
        require(set(expected) == {"path", "size_bytes", "sha256"}, "invalid identity record")
        require(record == dict(expected), "input identity mismatch: %s" % logical)
    if expected_sha:
        require(digest == expected_sha, "input SHA mismatch: %s" % logical)
    return record


def exact_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    require(type(value) is dict and set(value) == set(keys), "%s key set mismatch" % label)


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool), "%s must be numeric" % label)
    result = float(value)
    require(math.isfinite(result), "%s must be finite" % label)
    return result


def type7_quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "cannot take an empty quantile")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def metric_summary(full_deltas: Sequence[float], replicate_deltas: Sequence[float], contract: Mapping[str, Any]) -> Dict[str, Any]:
    require(len(full_deltas) == 5, "exactly five seed deltas required")
    require(len(replicate_deltas) == int(contract["registered_replicates"]), "replicate extent mismatch")
    full = [finite_number(value, "full-data delta") for value in full_deltas]
    replicates = [finite_number(value, "replicate delta") for value in replicate_deltas]
    mean = sum(full) / len(full)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in full) / (len(full) - 1))
    critical = finite_number(contract["intervals"]["seed_descriptor"]["two_sided_95_t_critical"], "t critical")
    half_width = critical * sample_sd / math.sqrt(len(full))
    lower_probability = finite_number(contract["intervals"]["scene_percentile_CI"]["lower_probability"], "lower probability")
    upper_probability = finite_number(contract["intervals"]["scene_percentile_CI"]["upper_probability"], "upper probability")
    return {
        "per_seed_full_data_deltas": full,
        "seed_summary": {
            "mean": mean,
            "sample_sd": sample_sd,
            "minimum": min(full),
            "maximum": max(full),
            "positive_seed_count": sum(value > 0.0 for value in full),
            "student_t_df": 4,
            "student_t_95_interval": {"lower": mean - half_width, "upper": mean + half_width},
        },
        "paired_scene_replicate_mean_deltas": replicates,
        "paired_scene_percentile_95_interval": {
            "lower": type7_quantile(replicates, lower_probability),
            "upper": type7_quantile(replicates, upper_probability),
            "quantile_method": "linear",
            "replicate_count": len(replicates),
        },
    }


def arm_metric(report: Mapping[str, Any], label: str, metric: str, replicate: int = None) -> float:
    require(label in report["arms"], "missing required arm: %s" % label)
    arm = report["arms"][label]
    if replicate is None:
        return finite_number(arm["all_ones"][metric], "%s full %s" % (label, metric))
    rows = arm["replicates"]
    require(len(rows) == int(report["replicate_count"]), "arm replicate extent mismatch: %s" % label)
    row = rows[replicate]
    require(type(row) is dict and row.get("replicate") == replicate, "replicate order mismatch: %s" % label)
    return finite_number(row[metric], "%s replicate %d %s" % (label, replicate, metric))


def contrast(report: Mapping[str, Any], contract: Mapping[str, Any], left_template: str, right_template: str) -> Dict[str, Any]:
    seed_indices = list(contract["seed_indices"])
    replicate_count = int(contract["registered_replicates"])
    result = {"left": left_template, "right": right_template, "metrics": {}}
    for metric in METRICS:
        full_deltas = []
        for seed in seed_indices:
            left = left_template.format(index=seed)
            right = right_template.format(index=seed)
            full_deltas.append(arm_metric(report, left, metric) - arm_metric(report, right, metric))
        replicate_deltas = []
        for replicate in range(replicate_count):
            values = []
            for seed in seed_indices:
                left = left_template.format(index=seed)
                right = right_template.format(index=seed)
                values.append(arm_metric(report, left, metric, replicate) - arm_metric(report, right, metric, replicate))
            replicate_deltas.append(sum(values) / len(values))
        result["metrics"][metric] = metric_summary(full_deltas, replicate_deltas, contract)
    return result


def apply_primary_gate(result: Mapping[str, Any], gate: Mapping[str, Any]) -> Dict[str, Any]:
    map_stats = result["metrics"]["mean_ap"]
    nds_stats = result["metrics"]["nd_score"]
    checks = {
        "mAP_mean_min": map_stats["seed_summary"]["mean"] >= finite_number(gate["mAP_mean_min"], "mAP mean gate"),
        "mAP_scene_CI_lower_gt": map_stats["paired_scene_percentile_95_interval"]["lower"] > finite_number(gate["mAP_scene_CI_lower_gt"], "mAP CI gate"),
        "mAP_positive_seed_count_min": map_stats["seed_summary"]["positive_seed_count"] >= int(gate["mAP_positive_seed_count_min"]),
        "NDS_mean_min": nds_stats["seed_summary"]["mean"] >= finite_number(gate["NDS_mean_min"], "NDS mean gate"),
        "NDS_scene_CI_lower_gt": nds_stats["paired_scene_percentile_95_interval"]["lower"] > finite_number(gate["NDS_scene_CI_lower_gt"], "NDS CI gate"),
    }
    return {"checks": checks, "passed": all(checks.values())}


def build_payload(config_record: Mapping[str, Any], config: Mapping[str, Any], contract_record: Mapping[str, Any], contract: Mapping[str, Any], report_record: Mapping[str, Any], report: Mapping[str, Any], verification_record: Mapping[str, Any], runner_record: Mapping[str, Any], checker_record: Mapping[str, Any]) -> Dict[str, Any]:
    labels = list(contract["required_event_labels_in_exact_order"])
    # JSON objects are serialized canonically with sorted keys; the registered
    # order lives in the explicit label vector rather than object insertion order.
    require(set(report["arms"]) == set(labels) and len(report["arms"]) == len(labels), "bootstrap arm set differs from registered extent")
    require(report.get("schema_version") == BOOTSTRAP_SCHEMA and report.get("status") == "passed_registered_bootstrap_replay", "bootstrap report schema/status mismatch")
    require(report.get("scope") == "registered_B1000" and report.get("scientific_evidence") is True, "bootstrap report is not registered scientific evidence")
    require(report.get("replicate_count") == contract["registered_replicates"] and report.get("scene_count") == contract["scene_count"], "bootstrap extent mismatch")

    gates = {item["id"]: item for item in contract["hypotheses_in_fixed_sequence"]}
    core = contrast(report, contract, "seed_{index:02d}_E4", "N")
    core["gate"] = apply_primary_gate(core, gates["core_E4_minus_N"])
    learned = contrast(report, contract, "seed_{index:02d}_E3", "E2")
    learned["gate"] = apply_primary_gate(learned, gates["learned_E3_minus_E2"])
    learned["gate"]["claim_active"] = bool(core["gate"]["passed"])
    learned["gate"]["claim_passed"] = bool(core["gate"]["passed"] and learned["gate"]["passed"])

    calibration = contrast(report, contract, "seed_{index:02d}_E4", "seed_{index:02d}_E3")
    calibration_contract = contract["mandatory_non_primary_audits"]["calibration_E4_minus_E3"]
    calibration_checks = {
        "mAP_scene_CI_upper_lt": calibration["metrics"]["mean_ap"]["paired_scene_percentile_95_interval"]["upper"] < finite_number(calibration_contract["mAP_scene_CI_upper_lt"], "calibration mAP bound"),
        "NDS_scene_CI_lower_gt": calibration["metrics"]["nd_score"]["paired_scene_percentile_95_interval"]["lower"] > finite_number(calibration_contract["NDS_scene_CI_lower_gt"], "calibration NDS guardrail"),
    }
    calibration["audit"] = {
        "checks": calibration_checks,
        "passed_non_materiality_bound": all(calibration_checks.values()),
        "language": calibration_contract["language_if_pass"] if all(calibration_checks.values()) else calibration_contract["language_if_fail"],
    }

    radar = contrast(report, contract, "seed_{index:02d}_E4", "seed_{index:02d}_E4NR")
    radar_contract = contract["mandatory_non_primary_audits"]["radar_E4_minus_E4NR"]
    radar["audit"] = {
        "formal_positive_gate_in_original_RQ2": False,
        "mAP_practical_bound_for_context_only": radar_contract["mAP_practical_bound_for_context_only"],
        "language": radar_contract["default_language"],
    }

    if not core["gate"]["passed"]:
        terminal = contract["terminal_states"]["core_fail"]
    elif not learned["gate"]["passed"]:
        terminal = contract["terminal_states"]["core_pass_learned_fail"]
    else:
        terminal = contract["terminal_states"]["core_and_learned_pass"]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": STATUS,
        "passed": True,
        "scientific_terminal_state": terminal,
        "identities": {
            "execution_config": dict(config_record),
            "analysis_contract": dict(contract_record),
            "bootstrap_report": dict(report_record),
            "bootstrap_verification": dict(verification_record),
            "analysis_runner": dict(runner_record),
            "analysis_checker": dict(checker_record),
        },
        "registered_extent": {
            "arm_labels": labels,
            "seed_indices": list(contract["seed_indices"]),
            "scene_count": int(contract["scene_count"]),
            "replicate_count": int(contract["registered_replicates"]),
            "metric_names": list(contract["metric_names"]),
        },
        "fixed_sequence": {"core_E4_minus_N": core, "learned_E3_minus_E2": learned},
        "mandatory_non_primary_audits": {"calibration_E4_minus_E3": calibration, "radar_E4_minus_E4NR": radar},
        "claim_boundaries": dict(contract["claim_boundaries"]),
        "safety": {
            "model_inference_calls": 0,
            "official_evaluator_calls": 0,
            "training": False,
            "optimizer_steps": 0,
            "per_scene_AP_average_used": False,
            "automatic_next_stage": False,
        },
    }


def markdown(payload: Mapping[str, Any]) -> str:
    def row(name: str, result: Mapping[str, Any]) -> str:
        metric = result["metrics"]["mean_ap"]
        mean = metric["seed_summary"]["mean"]
        interval = metric["paired_scene_percentile_95_interval"]
        return "| %s | %.6f | [%.6f, %.6f] |\n" % (name, mean, interval["lower"], interval["upper"])
    lines = [
        "# Registered five-seed R-GATE analysis\n\n",
        "Terminal state: `%s`\n\n" % payload["scientific_terminal_state"],
        "| Contrast | Mean full-data mAP delta | Paired-scene 95% interval |\n",
        "|---|---:|---:|\n",
        row("E4 - N", payload["fixed_sequence"]["core_E4_minus_N"]),
        row("E3 - E2", payload["fixed_sequence"]["learned_E3_minus_E2"]),
        row("E4 - E3", payload["mandatory_non_primary_audits"]["calibration_E4_minus_E3"]),
        row("E4 - E4NR", payload["mandatory_non_primary_audits"]["radar_E4_minus_E4NR"]),
        "\nThis is a data-informed prospective five-seed extension on the previously seen validation set. ",
        "The scene intervals are exact official-algorithm replays, not per-scene AP averages and not test-set evidence.\n",
    ]
    return "".join(lines)


def load_context(config_path: Path, expected_config_sha: str) -> Tuple[Dict[str, Any], Mapping[str, Any], Dict[str, Any], Mapping[str, Any], Dict[str, Any], Mapping[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config_record = stable_record(config_path, expected_sha=expected_config_sha)
    config = load_json(config_path)
    exact_keys(config, ("schema_version", "status", "analysis_contract", "bootstrap_report", "bootstrap_verification", "implementation", "output_dir", "checker_output_path"), "execution config")
    require(config["schema_version"] == CONFIG_SCHEMA and config["status"] == "frozen_evidence_identity_only_after_complete_B1000", "execution config schema/status mismatch")
    contract_path = Path(config["analysis_contract"]["path"])
    report_path = Path(config["bootstrap_report"]["path"])
    verification_path = Path(config["bootstrap_verification"]["path"])
    contract_record = stable_record(contract_path, config["analysis_contract"])
    report_record = stable_record(report_path, config["bootstrap_report"])
    verification_record = stable_record(verification_path, config["bootstrap_verification"])
    contract = load_json(contract_path)
    report = load_json(report_path)
    verification = load_json(verification_path)
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "analysis contract schema mismatch")
    require(verification.get("schema_version") == BOOTSTRAP_CHECK_SCHEMA and verification.get("status") == "passed_independent_paired_scene_bootstrap_replay" and verification.get("passed") is True, "bootstrap verification mismatch")
    require(verification.get("scope") == "registered_B1000" and verification.get("scientific_evidence") is True, "bootstrap verification is not scientific B1000")
    require(verification.get("report") == report_record, "bootstrap verification does not bind report")
    exact_keys(config["implementation"], ("analysis_runner", "analysis_checker"), "implementation")
    runner_record = stable_record(Path(config["implementation"]["analysis_runner"]["path"]), config["implementation"]["analysis_runner"])
    checker_record = stable_record(Path(config["implementation"]["analysis_checker"]["path"]), config["implementation"]["analysis_checker"])
    require(runner_record["path"] == str(Path(__file__).absolute()), "execution config binds a different analysis runner")
    return config_record, config, contract_record, contract, report_record, report, verification_record, runner_record, checker_record


def publish_directory(output_dir: Path, payload: Mapping[str, Any]) -> None:
    target = output_dir.absolute()
    reject_symlink_chain(target.parent)
    require(target.parent.is_dir() and not os.path.lexists(str(target)), "analysis output directory must be fresh")
    temporary = Path(tempfile.mkdtemp(prefix=".%s.tmp." % target.name, dir=str(target.parent)))
    owned_target = False
    linked_md = False
    linked_json = False
    try:
        json_path = temporary / "registered_analysis.json"
        md_path = temporary / "registered_analysis.md"
        with json_path.open("xb") as handle:
            handle.write(canonical_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
        with md_path.open("xb") as handle:
            handle.write(markdown(payload).encode("utf-8")); handle.flush(); os.fsync(handle.fileno())
        # mkdir is the exclusive ownership claim.  Markdown is linked first and
        # the JSON evidence is the terminal-last commit marker; a checker never
        # accepts a directory without both exact files.
        os.mkdir(str(target)); owned_target = True
        os.link(str(md_path), str(target / "registered_analysis.md")); linked_md = True
        os.link(str(json_path), str(target / "registered_analysis.json")); linked_json = True
        directory_fd = os.open(str(target), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        parent_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if owned_target:
            if linked_json and (target / "registered_analysis.json").exists():
                (target / "registered_analysis.json").unlink()
            if linked_md and (target / "registered_analysis.md").exists():
                (target / "registered_analysis.md").unlink()
            try:
                target.rmdir()
            except OSError:
                pass
        raise
    finally:
        for item in temporary.iterdir():
            item.unlink()
        temporary.rmdir()


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--expected-execution-config-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enable-registered-multiseed-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(args.enable_registered_multiseed_analysis, "explicit analysis enable flag is required")
        context = load_context(Path(args.execution_config).absolute(), args.expected_execution_config_sha256)
        config_record, config, contract_record, contract, report_record, report, verification_record, runner_record, checker_record = context
        output_dir = Path(args.output_dir).absolute()
        require(str(output_dir) == config["output_dir"], "analysis output path drift")
        payload = build_payload(config_record, config, contract_record, contract, report_record, report, verification_record, runner_record, checker_record)
        publish_directory(output_dir, payload)
    except (AnalysisError, KeyError, OSError, ValueError) as exc:
        print("FAILED: %s" % exc, file=os.sys.stderr)
        return 1
    print(str(Path(args.output_dir).absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
