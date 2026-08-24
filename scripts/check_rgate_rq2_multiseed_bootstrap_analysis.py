#!/usr/bin/env python3
"""Independently recompute and verify the registered five-seed analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Sequence


CONFIG_SCHEMA = "rgate_rq2_multiseed_registered_analysis_execution_v1"
CONTRACT_SCHEMA = "rgate_rq2_multiseed_analysis_contract_v1"
BOOTSTRAP_SCHEMA = "nuscenes_paired_scene_bootstrap_report_v1"
BOOTSTRAP_CHECK_SCHEMA = "nuscenes_paired_scene_bootstrap_independent_check_v1"
ANALYSIS_SCHEMA = "rgate_rq2_multiseed_registered_analysis_v1"
CHECK_SCHEMA = "rgate_rq2_multiseed_registered_analysis_independent_check_v1"
METRICS = ("mean_ap", "nd_score", "mAVE")


class CheckError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise CheckError("non-finite JSON constant: %s" % value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_symlinks(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
        current = current / component
        if os.path.lexists(str(current)):
            require(not current.is_symlink(), "symlink path component rejected: %s" % current)


def record(path: Path, expected: Mapping[str, Any] = None, expected_sha: str = "") -> Dict[str, Any]:
    logical = path.absolute()
    reject_symlinks(logical)
    require(logical.is_file(), "missing input: %s" % logical)
    before = logical.stat()
    digest = file_sha256(logical)
    after = logical.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed while hashing")
    result = {"path": str(logical), "size_bytes": int(before.st_size), "sha256": digest}
    if expected is not None:
        require(set(expected) == {"path", "size_bytes", "sha256"} and result == dict(expected), "identity mismatch: %s" % logical)
    if expected_sha:
        require(digest == expected_sha, "SHA mismatch: %s" % logical)
    return result


def finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool), "%s must be numeric" % label)
    result = float(value)
    require(math.isfinite(result), "%s must be finite" % label)
    return result


def quantile_linear(values: Sequence[float], probability: float) -> float:
    ordered = sorted(finite(value, "replicate delta") for value in values)
    require(bool(ordered), "empty replicate vector")
    location = (len(ordered) - 1) * probability
    low = int(math.floor(location))
    high = int(math.ceil(location))
    weight = location - low
    return ordered[low] + weight * (ordered[high] - ordered[low])


def arm_value(report: Mapping[str, Any], label: str, metric: str, replicate: int = None) -> float:
    require(label in report["arms"], "missing arm: %s" % label)
    arm = report["arms"][label]
    if replicate is None:
        return finite(arm["all_ones"][metric], "%s %s" % (label, metric))
    rows = arm["replicates"]
    require(len(rows) == report["replicate_count"], "replicate extent mismatch")
    require(rows[replicate].get("replicate") == replicate, "replicate index mismatch")
    return finite(rows[replicate][metric], "%s replicate metric" % label)


def summarize_metric(full_values: Sequence[float], scene_values: Sequence[float], contract: Mapping[str, Any]) -> Dict[str, Any]:
    require(len(full_values) == 5 and len(scene_values) == contract["registered_replicates"], "statistic extent mismatch")
    full = [finite(item, "full delta") for item in full_values]
    scenes = [finite(item, "scene delta") for item in scene_values]
    mean = sum(full) / 5.0
    sd = math.sqrt(sum((item - mean) ** 2 for item in full) / 4.0)
    critical = finite(contract["intervals"]["seed_descriptor"]["two_sided_95_t_critical"], "t critical")
    half = critical * sd / math.sqrt(5.0)
    lower_p = finite(contract["intervals"]["scene_percentile_CI"]["lower_probability"], "lower probability")
    upper_p = finite(contract["intervals"]["scene_percentile_CI"]["upper_probability"], "upper probability")
    return {
        "per_seed_full_data_deltas": full,
        "seed_summary": {
            "mean": mean,
            "sample_sd": sd,
            "minimum": min(full),
            "maximum": max(full),
            "positive_seed_count": sum(item > 0.0 for item in full),
            "student_t_df": 4,
            "student_t_95_interval": {"lower": mean - half, "upper": mean + half},
        },
        "paired_scene_replicate_mean_deltas": scenes,
        "paired_scene_percentile_95_interval": {
            "lower": quantile_linear(scenes, lower_p),
            "upper": quantile_linear(scenes, upper_p),
            "quantile_method": "linear",
            "replicate_count": len(scenes),
        },
    }


def recompute_contrast(report: Mapping[str, Any], contract: Mapping[str, Any], left_pattern: str, right_pattern: str) -> Dict[str, Any]:
    seeds = list(contract["seed_indices"])
    result = {"left": left_pattern, "right": right_pattern, "metrics": {}}
    for metric in METRICS:
        full = []
        for seed in seeds:
            left = left_pattern.format(index=seed)
            right = right_pattern.format(index=seed)
            full.append(arm_value(report, left, metric) - arm_value(report, right, metric))
        scenes = []
        for replicate in range(contract["registered_replicates"]):
            seed_values = []
            for seed in seeds:
                left = left_pattern.format(index=seed)
                right = right_pattern.format(index=seed)
                seed_values.append(arm_value(report, left, metric, replicate) - arm_value(report, right, metric, replicate))
            scenes.append(sum(seed_values) / 5.0)
        result["metrics"][metric] = summarize_metric(full, scenes, contract)
    return result


def primary_gate(result: Mapping[str, Any], gate: Mapping[str, Any]) -> Dict[str, Any]:
    map_result = result["metrics"]["mean_ap"]
    nds_result = result["metrics"]["nd_score"]
    checks = {
        "mAP_mean_min": map_result["seed_summary"]["mean"] >= finite(gate["mAP_mean_min"], "mAP threshold"),
        "mAP_scene_CI_lower_gt": map_result["paired_scene_percentile_95_interval"]["lower"] > finite(gate["mAP_scene_CI_lower_gt"], "mAP CI threshold"),
        "mAP_positive_seed_count_min": map_result["seed_summary"]["positive_seed_count"] >= int(gate["mAP_positive_seed_count_min"]),
        "NDS_mean_min": nds_result["seed_summary"]["mean"] >= finite(gate["NDS_mean_min"], "NDS threshold"),
        "NDS_scene_CI_lower_gt": nds_result["paired_scene_percentile_95_interval"]["lower"] > finite(gate["NDS_scene_CI_lower_gt"], "NDS CI threshold"),
    }
    return {"checks": checks, "passed": all(checks.values())}


def recompute_payload(config_rec: Mapping[str, Any], contract_rec: Mapping[str, Any], contract: Mapping[str, Any], report_rec: Mapping[str, Any], report: Mapping[str, Any], verification_rec: Mapping[str, Any], runner_rec: Mapping[str, Any], checker_rec: Mapping[str, Any]) -> Dict[str, Any]:
    labels = list(contract["required_event_labels_in_exact_order"])
    require(set(report["arms"]) == set(labels) and len(report["arms"]) == len(labels), "registered arm set mismatch")
    hypotheses = {row["id"]: row for row in contract["hypotheses_in_fixed_sequence"]}
    core = recompute_contrast(report, contract, "seed_{index:02d}_E4", "N")
    core["gate"] = primary_gate(core, hypotheses["core_E4_minus_N"])
    learned = recompute_contrast(report, contract, "seed_{index:02d}_E3", "E2")
    learned["gate"] = primary_gate(learned, hypotheses["learned_E3_minus_E2"])
    learned["gate"]["claim_active"] = bool(core["gate"]["passed"])
    learned["gate"]["claim_passed"] = bool(core["gate"]["passed"] and learned["gate"]["passed"])

    calibration = recompute_contrast(report, contract, "seed_{index:02d}_E4", "seed_{index:02d}_E3")
    calibration_spec = contract["mandatory_non_primary_audits"]["calibration_E4_minus_E3"]
    calibration_checks = {
        "mAP_scene_CI_upper_lt": calibration["metrics"]["mean_ap"]["paired_scene_percentile_95_interval"]["upper"] < finite(calibration_spec["mAP_scene_CI_upper_lt"], "calibration bound"),
        "NDS_scene_CI_lower_gt": calibration["metrics"]["nd_score"]["paired_scene_percentile_95_interval"]["lower"] > finite(calibration_spec["NDS_scene_CI_lower_gt"], "calibration guardrail"),
    }
    calibration_passed = all(calibration_checks.values())
    calibration["audit"] = {
        "checks": calibration_checks,
        "passed_non_materiality_bound": calibration_passed,
        "language": calibration_spec["language_if_pass"] if calibration_passed else calibration_spec["language_if_fail"],
    }
    radar = recompute_contrast(report, contract, "seed_{index:02d}_E4", "seed_{index:02d}_E4NR")
    radar_spec = contract["mandatory_non_primary_audits"]["radar_E4_minus_E4NR"]
    radar["audit"] = {
        "formal_positive_gate_in_original_RQ2": False,
        "mAP_practical_bound_for_context_only": radar_spec["mAP_practical_bound_for_context_only"],
        "language": radar_spec["default_language"],
    }
    if not core["gate"]["passed"]:
        terminal = contract["terminal_states"]["core_fail"]
    elif not learned["gate"]["passed"]:
        terminal = contract["terminal_states"]["core_pass_learned_fail"]
    else:
        terminal = contract["terminal_states"]["core_and_learned_pass"]
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "passed_registered_multiseed_analysis",
        "passed": True,
        "scientific_terminal_state": terminal,
        "identities": {
            "execution_config": dict(config_rec),
            "analysis_contract": dict(contract_rec),
            "bootstrap_report": dict(report_rec),
            "bootstrap_verification": dict(verification_rec),
            "analysis_runner": dict(runner_rec),
            "analysis_checker": dict(checker_rec),
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


def expected_markdown(payload: Mapping[str, Any]) -> str:
    def row(name: str, value: Mapping[str, Any]) -> str:
        stats = value["metrics"]["mean_ap"]
        ci = stats["paired_scene_percentile_95_interval"]
        return "| %s | %.6f | [%.6f, %.6f] |\n" % (name, stats["seed_summary"]["mean"], ci["lower"], ci["upper"])
    return "".join([
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
    ])


def publish(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.absolute()
    reject_symlinks(target.parent)
    require(target.parent.is_dir() and not os.path.lexists(str(target)), "checker output must be fresh")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s.tmp." % target.name, dir=str(target.parent))
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
        os.link(str(temporary), str(target)); linked = True
        parent_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if linked and target.exists():
            target.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--expected-execution-config-sha256", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--expected-analysis-json-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.execution_config).absolute()
        config_rec = record(config_path, expected_sha=args.expected_execution_config_sha256)
        config = load_json(config_path)
        require(set(config) == {"schema_version", "status", "analysis_contract", "bootstrap_report", "bootstrap_verification", "implementation", "output_dir", "checker_output_path"}, "config key set mismatch")
        require(config["schema_version"] == CONFIG_SCHEMA and config["status"] == "frozen_evidence_identity_only_after_complete_B1000", "config schema/status mismatch")
        contract_rec = record(Path(config["analysis_contract"]["path"]), config["analysis_contract"])
        report_rec = record(Path(config["bootstrap_report"]["path"]), config["bootstrap_report"])
        verification_rec = record(Path(config["bootstrap_verification"]["path"]), config["bootstrap_verification"])
        runner_rec = record(Path(config["implementation"]["analysis_runner"]["path"]), config["implementation"]["analysis_runner"])
        checker_rec = record(Path(config["implementation"]["analysis_checker"]["path"]), config["implementation"]["analysis_checker"])
        require(checker_rec["path"] == str(Path(__file__).absolute()), "config binds another checker")
        contract, report, verification = load_json(Path(contract_rec["path"])), load_json(Path(report_rec["path"])), load_json(Path(verification_rec["path"]))
        require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema mismatch")
        require(report.get("schema_version") == BOOTSTRAP_SCHEMA and report.get("status") == "passed_registered_bootstrap_replay" and report.get("scope") == "registered_B1000" and report.get("scientific_evidence") is True, "bootstrap report mismatch")
        require(verification.get("schema_version") == BOOTSTRAP_CHECK_SCHEMA and verification.get("status") == "passed_independent_paired_scene_bootstrap_replay" and verification.get("passed") is True, "bootstrap verification mismatch")
        require(verification.get("report") == report_rec and verification.get("replicate_count") == contract["registered_replicates"], "bootstrap verification identity/extent mismatch")
        analysis_dir = Path(args.analysis_dir).absolute()
        require(str(analysis_dir) == config["output_dir"] and analysis_dir.is_dir() and not analysis_dir.is_symlink(), "analysis directory mismatch")
        require({path.name for path in analysis_dir.iterdir()} == {"registered_analysis.json", "registered_analysis.md"}, "unexpected analysis files")
        analysis_path = analysis_dir / "registered_analysis.json"
        analysis_rec = record(analysis_path, expected_sha=args.expected_analysis_json_sha256)
        observed = load_json(analysis_path)
        expected = recompute_payload(config_rec, contract_rec, contract, report_rec, report, verification_rec, runner_rec, checker_rec)
        require(canonical_bytes(observed) == canonical_bytes(expected), "analysis JSON differs from independent recomputation")
        require((analysis_dir / "registered_analysis.md").read_text(encoding="utf-8") == expected_markdown(expected), "analysis Markdown differs from independent rendering")
        output = Path(args.output).absolute()
        require(str(output) == config["checker_output_path"], "checker output path drift")
        payload = {
            "schema_version": CHECK_SCHEMA,
            "status": "passed_independent_registered_multiseed_analysis",
            "passed": True,
            "scientific_terminal_state": expected["scientific_terminal_state"],
            "analysis": analysis_rec,
            "execution_config": config_rec,
            "analysis_contract": contract_rec,
            "bootstrap_report": report_rec,
            "bootstrap_verification": verification_rec,
            "implementation": {"analysis_runner": runner_rec, "analysis_checker": checker_rec},
            "all_registered_statistics_exact": True,
            "all_gate_booleans_exact": True,
            "all_17_event_arms_present": True,
            "all_1000_replicates_present": True,
            "per_scene_AP_average_used": False,
            "official_evaluator_calls": 0,
            "automatic_next_stage": False,
        }
        publish(output, payload)
    except (CheckError, KeyError, OSError, ValueError) as exc:
        print("FAILED: %s" % exc, file=os.sys.stderr)
        return 1
    print(str(Path(args.output).absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
