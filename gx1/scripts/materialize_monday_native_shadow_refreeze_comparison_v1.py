#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

TOP_LEVEL_SUMMARY = "truth_monday_native_shadow_refreeze_comparison_v1.json"
COMPARATOR_PREFIX = "MONDAY_TOP_PRE_RL_BASELINE_COMPARATOR_V1_"
BENCHMARK_SNAPSHOT_PREFIX = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_"

CONTRACT = "contract_v1.json"
ACTIVE_BENCHMARK_MATRIX = "active_vs_benchmark_matrix_v1.csv"
REFREEZE_READINESS_MATRIX = "refreeze_readiness_matrix_v1.csv"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_prefixed_dir(reports_root: Path, prefix: str) -> Path:
    candidates = [path for path in reports_root.glob(f"{prefix}*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Missing required directory with prefix {prefix} under {reports_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / f"MONDAY_NATIVE_SHADOW_REFREEZE_COMPARISON_V1_{_utc_stamp()}"


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_policy_logging_artifact(reports_root: Path) -> Path | None:
    candidates = sorted(reports_root.glob("ALL_TRADE_REVIEW_LEDGER_*/shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet"))
    return candidates[-1] if candidates else None


def _step_status_counts(rebuild_summary: dict[str, Any]) -> tuple[int, int]:
    steps = rebuild_summary.get("steps", [])
    ok = sum(1 for step in steps if str(step.get("status")) == "ok")
    blocked = sum(1 for step in steps if str(step.get("status")) != "ok")
    return int(ok), int(blocked)


def _headline_matrix(
    *,
    monday_foundation: dict[str, Any],
    benchmark_foundation: dict[str, Any],
    entry_summary: dict[str, Any],
    management_summary: dict[str, Any],
    sequence_summary: dict[str, Any],
    bandit_summary: dict[str, Any],
    exit_local_summary: dict[str, Any],
    benchmark_r6_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    benchmark_r6_selected = benchmark_r6_summary.get("selected_candidate_v1", {})
    benchmark_batch05 = benchmark_r6_summary.get("batch05_v1", {})
    return [
        {
            "metric_name_v1": "trade_count",
            "active_value_v1": monday_foundation.get("trade_count"),
            "benchmark_value_v1": benchmark_foundation.get("trade_count"),
            "comparison_v1": "LOWER_IS_ACTIVE_WEAKER" if int(monday_foundation.get("trade_count", 0)) < int(benchmark_foundation.get("trade_count", 0)) else "MATCH_OR_HIGHER",
            "notes_v1": "Raw closed-trade count.",
        },
        {
            "metric_name_v1": "avg_pnl_bps",
            "active_value_v1": monday_foundation.get("avg_pnl_bps"),
            "benchmark_value_v1": benchmark_foundation.get("avg_pnl_bps"),
            "comparison_v1": "ACTIVE_WEAKER" if float(monday_foundation.get("avg_pnl_bps", 0.0)) < float(benchmark_foundation.get("avg_pnl_bps", 0.0)) else "ACTIVE_STRONGER_OR_EQUAL",
            "notes_v1": "Raw pre-RL truth foundation.",
        },
        {
            "metric_name_v1": "profit_factor",
            "active_value_v1": monday_foundation.get("profit_factor"),
            "benchmark_value_v1": benchmark_foundation.get("profit_factor"),
            "comparison_v1": "ACTIVE_WEAKER" if float(monday_foundation.get("profit_factor", 0.0)) < float(benchmark_foundation.get("profit_factor", 0.0)) else "ACTIVE_STRONGER_OR_EQUAL",
            "notes_v1": "Raw pre-RL truth foundation.",
        },
        {
            "metric_name_v1": "max_drawdown_bps",
            "active_value_v1": monday_foundation.get("max_drawdown_bps"),
            "benchmark_value_v1": benchmark_foundation.get("max_drawdown_bps"),
            "comparison_v1": "ACTIVE_WORSE" if float(monday_foundation.get("max_drawdown_bps", 0.0)) < float(benchmark_foundation.get("max_drawdown_bps", 0.0)) else "ACTIVE_BETTER_OR_EQUAL",
            "notes_v1": "More negative is worse.",
        },
        {
            "metric_name_v1": "entry_observed_direct_rows",
            "active_value_v1": entry_summary.get("observed_direct_entry_rows_v1"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday entry observability substrate rows.",
        },
        {
            "metric_name_v1": "management_rows",
            "active_value_v1": management_summary.get("management_rows_v1"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday management RL readiness substrate rows.",
        },
        {
            "metric_name_v1": "management_observation_feature_count",
            "active_value_v1": management_summary.get("observation_feature_count_v1"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday management observation vector width.",
        },
        {
            "metric_name_v1": "strict_sequence_row_count",
            "active_value_v1": sequence_summary.get("strict_sequence_row_count_v1"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday strict sequence substrate rows.",
        },
        {
            "metric_name_v1": "bandit_dm_candidate_rows",
            "active_value_v1": bandit_summary.get("MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday bandit-safe management rows.",
        },
        {
            "metric_name_v1": "exit_local_status",
            "active_value_v1": exit_local_summary.get("MANAGEMENT_EXIT_LOCAL_BASELINE_STATUS"),
            "benchmark_value_v1": None,
            "comparison_v1": "ACTIVE_ONLY",
            "notes_v1": "Monday offline exit-local trainer status.",
        },
        {
            "metric_name_v1": "benchmark_r6_bad_blocks",
            "active_value_v1": None,
            "benchmark_value_v1": benchmark_r6_selected.get("should_not_take_block_count_v1"),
            "comparison_v1": "BENCHMARK_ONLY",
            "notes_v1": "Locked Wednesday R6 shadow-freeze reference.",
        },
        {
            "metric_name_v1": "benchmark_r6_tail_help",
            "active_value_v1": None,
            "benchmark_value_v1": benchmark_r6_selected.get("tail_10_50_help_count_v1"),
            "comparison_v1": "BENCHMARK_ONLY",
            "notes_v1": "Locked Wednesday R6 tail-control reference.",
        },
        {
            "metric_name_v1": "benchmark_r6_batch05_precision",
            "active_value_v1": None,
            "benchmark_value_v1": benchmark_batch05.get("precision_v1"),
            "comparison_v1": "BENCHMARK_ONLY",
            "notes_v1": "Locked Wednesday R6 thin-margin monitor reference.",
        },
    ]


def _readiness_rows(
    *,
    reports_root: Path,
    ledger_dir: Path,
    rebuild_summary: dict[str, Any],
    entry_summary: dict[str, Any],
    management_summary: dict[str, Any],
    bandit_summary: dict[str, Any],
    benchmark_snapshot_dir: Path,
    benchmark_r6_dir: Path,
) -> list[dict[str, Any]]:
    ok_steps, blocked_steps = _step_status_counts(rebuild_summary)
    policy_logging_artifact = _find_policy_logging_artifact(reports_root)
    current_refreeze_dirs = [
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FULLCOVERAGE_POLICY_RECALIBRATION_AND_SHADOW_REPLAY_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_LOSO_BATCH04_ROBUSTNESS_RETRAIN_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1",
        "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1",
    ]
    current_refreeze_present = all((reports_root / name).exists() for name in current_refreeze_dirs)
    return [
        {
            "check_name_v1": "MONDAY_ROOT_POINTER_MATCHES_REPORTS_ROOT",
            "status_v1": "PASS" if _resolve_reports_root(str(reports_root)) == reports_root else "FAIL",
            "details_v1": str(reports_root),
        },
        {
            "check_name_v1": "MONDAY_REBUILD_GREEN",
            "status_v1": "PASS" if blocked_steps == 0 and ok_steps > 0 else "FAIL",
            "details_v1": json.dumps({"ok_steps_v1": ok_steps, "blocked_steps_v1": blocked_steps}, ensure_ascii=True),
        },
        {
            "check_name_v1": "MONDAY_LEDGER_PRESENT",
            "status_v1": "PASS" if ledger_dir.exists() else "FAIL",
            "details_v1": str(ledger_dir),
        },
        {
            "check_name_v1": "MONDAY_ENTRY_OBSERVABILITY_PRESENT",
            "status_v1": "PASS" if int(entry_summary.get("observed_direct_entry_rows_v1", 0)) > 0 else "FAIL",
            "details_v1": json.dumps(entry_summary.get("logged_action_counts_v1", {}), ensure_ascii=True),
        },
        {
            "check_name_v1": "MONDAY_MANAGEMENT_RL_READINESS_PRESENT",
            "status_v1": "PASS" if int(management_summary.get("management_rows_v1", 0)) > 0 else "FAIL",
            "details_v1": json.dumps(
                {
                    "management_rows_v1": management_summary.get("management_rows_v1"),
                    "feature_count_v1": management_summary.get("observation_feature_count_v1"),
                },
                ensure_ascii=True,
            ),
        },
        {
            "check_name_v1": "MONDAY_BANDIT_SUBSTRATE_PRESENT",
            "status_v1": "PASS" if int(bandit_summary.get("MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1", 0)) > 0 else "FAIL",
            "details_v1": json.dumps(
                {
                    "dm_candidates_v1": bandit_summary.get("MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1"),
                    "trainer_recommendation_v1": bandit_summary.get("MANAGEMENT_BANDIT_TRAINER_RECOMMENDATION"),
                },
                ensure_ascii=True,
            ),
        },
        {
            "check_name_v1": "BENCHMARK_SNAPSHOT_LOCKED",
            "status_v1": "PASS" if benchmark_snapshot_dir.exists() else "FAIL",
            "details_v1": str(benchmark_snapshot_dir),
        },
        {
            "check_name_v1": "BENCHMARK_R6_FREEZE_LOCKED",
            "status_v1": "PASS" if benchmark_r6_dir.exists() else "FAIL",
            "details_v1": str(benchmark_r6_dir),
        },
        {
            "check_name_v1": "MONDAY_POLICY_LOGGING_ARTIFACT_PRESENT",
            "status_v1": "PASS" if policy_logging_artifact is not None else "FAIL",
            "details_v1": str(policy_logging_artifact) if policy_logging_artifact is not None else "NOT_FOUND",
        },
        {
            "check_name_v1": "MONDAY_CURRENT_REFREEZE_CHAIN_PRESENT",
            "status_v1": "PASS" if current_refreeze_present else "FAIL",
            "details_v1": json.dumps(
                {
                    "required_dirs_v1": current_refreeze_dirs,
                    "missing_dirs_v1": [name for name in current_refreeze_dirs if not (reports_root / name).exists()],
                },
                ensure_ascii=True,
            ),
        },
    ]


def _decision(readiness_rows: list[dict[str, Any]], monday_foundation: dict[str, Any], benchmark_foundation: dict[str, Any]) -> tuple[str, str]:
    failed = {row["check_name_v1"] for row in readiness_rows if row["status_v1"] != "PASS"}
    active_pf = float(monday_foundation.get("profit_factor", 0.0) or 0.0)
    benchmark_pf = float(benchmark_foundation.get("profit_factor", 0.0) or 0.0)
    if "MONDAY_REBUILD_GREEN" in failed or "MONDAY_LEDGER_PRESENT" in failed:
        return "MONDAY_ROOT_NOT_READY", "Active Monday canonical root is not rebuild-green."
    if "MONDAY_POLICY_LOGGING_ARTIFACT_PRESENT" in failed:
        return "MONDAY_COMPARE_READY_REFREEZE_CHAIN_BLOCKED_BY_POLICY_LOGGING", "Benchmark comparison is valid, but Monday root still lacks the materialized management policy-log artifact."
    if "MONDAY_CURRENT_REFREEZE_CHAIN_PRESENT" in failed:
        return "MONDAY_COMPARE_READY_REBUILD_SHADOW_CHAIN_NEXT", "Benchmark comparison is valid, but the current Monday-native R4/R5/R5.2/R6 chain is not materialized yet."
    if active_pf < benchmark_pf:
        return "MONDAY_REFREEZE_REQUIRED_TO_CHALLENGE_BENCHMARK", "Monday raw truth remains weaker than the locked benchmark; rebuild the shadow chain before any promotion."
    return "MONDAY_REFREEZE_CHAIN_CAN_CHALLENGE_BENCHMARK", "Monday root has the required shadow chain prerequisites and can be compared apples-to-apples."


def _hard_status(readiness_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    readiness_map = {str(row["check_name_v1"]): str(row["status_v1"]) for row in readiness_rows}
    bevist = [
        "Monday canonical rebuild is green and points to a live ALL_TRADE_REVIEW_LEDGER.",
        "Benchmark-only Wednesday snapshot is still available locally for pre-RL/R6 comparison.",
        "Monday entry and management RL substrate summaries are materialized on the new ledger.",
    ]
    indikert = [
        "Monday raw truth is still weaker than the locked benchmark on raw trade foundation metrics.",
        "A fresh Monday-native R5.2/R6-style shadow chain should be rebuilt before any apples-to-apples replacement claim.",
    ]
    ikke_etablert = [
        "A new Monday-native R5.2 freeze.",
        "A new Monday-native R6 freeze beating the locked benchmark.",
    ]
    if readiness_map.get("MONDAY_POLICY_LOGGING_ARTIFACT_PRESENT") == "PASS":
        bevist.append("Monday management policy logging artifact is now materialized on the active ledger.")
    else:
        ikke_etablert.append("A complete Monday policy-logging artifact chain suitable for full path-dynamics shadow-freeze parity.")
    if readiness_map.get("MONDAY_CURRENT_REFREEZE_CHAIN_PRESENT") == "FAIL":
        ikke_etablert.append("A complete Monday-native R4/R5/R5.2/R6 refreeze chain on the active root.")
    return {
        "BEVIST": bevist,
        "INDIKERT": indikert,
        "IKKE_ETABLERT": ikke_etablert,
    }


def _report(
    *,
    summary: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MONDAY_NATIVE_SHADOW_REFREEZE_COMPARISON_V1",
        "",
        f"- Decision: `{summary['decision_v1']}`",
        f"- Active ledger: `{summary['active_ledger_dir_v1']}`",
        f"- Benchmark snapshot: `{summary['benchmark_snapshot_dir_v1']}`",
        f"- Benchmark R6 freeze: `{summary['benchmark_r6_freeze_id_v1']}`",
        "",
        "## Headline",
    ]
    for row in matrix_rows:
        lines.append(
            f"- `{row['metric_name_v1']}`: active=`{row['active_value_v1']}` benchmark=`{row['benchmark_value_v1']}` note=`{row['comparison_v1']}`"
        )
    lines.extend(["", "## Refreeze Readiness"])
    for row in readiness_rows:
        lines.append(f"- `{row['check_name_v1']}`: `{row['status_v1']}`")
    lines.extend(["", "## Hard Status"])
    for bucket, items in summary["hard_status_division_v1"].items():
        lines.append(f"### {bucket}")
        for item in items:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def materialize(
    reports_root: str | Path | None = None,
    *,
    extension_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = _resolve_reports_root(str(reports_root) if reports_root is not None else None)
    out_dir = Path(extension_dir).expanduser().resolve() if extension_dir is not None else _default_extension_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparator_dir = _latest_prefixed_dir(root, COMPARATOR_PREFIX)
    benchmark_snapshot_dir = _latest_prefixed_dir(root, BENCHMARK_SNAPSHOT_PREFIX)
    benchmark_r6_dir = benchmark_snapshot_dir / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
    if not benchmark_r6_dir.exists():
        raise FileNotFoundError(f"Benchmark R6 freeze snapshot missing: {benchmark_r6_dir}")

    rebuild_summary = _load_json(root / "truth_downstream_canonical_rebuild_v1.json")
    ledger_dir = Path(str(rebuild_summary["ledger_dir"])).expanduser().resolve()
    comparator_summary = _load_json(comparator_dir / "summary_v1.json")
    benchmark_snapshot_summary = _load_json(benchmark_snapshot_dir / "summary_v1.json")
    benchmark_r6_summary = _load_json(benchmark_r6_dir / "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json")

    entry_summary = _load_json(ledger_dir / "shadow_meta_all_trade_review_entry_rl_observability_summary_v1.json")
    management_summary = _load_json(ledger_dir / "shadow_meta_all_trade_review_management_rl_readiness_summary_v1.json")
    sequence_summary = _load_json(ledger_dir / "shadow_meta_all_trade_review_management_rl_sequence_summary_v1.json")
    bandit_summary = _load_json(ledger_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json")
    exit_local_summary = _load_json(ledger_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json")

    monday_foundation = dict(comparator_summary.get("monday_trade_foundation_v1", {}))
    benchmark_foundation = dict(comparator_summary.get("benchmark_trade_foundation_v1", {}))
    matrix_rows = _headline_matrix(
        monday_foundation=monday_foundation,
        benchmark_foundation=benchmark_foundation,
        entry_summary=entry_summary,
        management_summary=management_summary,
        sequence_summary=sequence_summary,
        bandit_summary=bandit_summary,
        exit_local_summary=exit_local_summary,
        benchmark_r6_summary=benchmark_r6_summary,
    )
    readiness_rows = _readiness_rows(
        reports_root=root,
        ledger_dir=ledger_dir,
        rebuild_summary=rebuild_summary,
        entry_summary=entry_summary,
        management_summary=management_summary,
        bandit_summary=bandit_summary,
        benchmark_snapshot_dir=benchmark_snapshot_dir,
        benchmark_r6_dir=benchmark_r6_dir,
    )
    decision, decision_reason = _decision(readiness_rows, monday_foundation, benchmark_foundation)

    consistency_rows = [
        {
            "check_name_v1": "ACTIVE_ROOT_POINTER_IS_MONDAY_ROOT",
            "status_v1": "PASS" if root == _resolve_reports_root(str(root)) else "FAIL",
            "observed_v1": str(root),
            "expected_v1": str(root),
        },
        {
            "check_name_v1": "BENCHMARK_SNAPSHOT_EXISTS",
            "status_v1": "PASS" if benchmark_snapshot_summary.get("copied_count_v1", 0) > 0 else "FAIL",
            "observed_v1": benchmark_snapshot_summary.get("copied_count_v1"),
            "expected_v1": ">0",
        },
        {
            "check_name_v1": "R6_FREEZE_ID_LOCKED",
            "status_v1": "PASS" if benchmark_r6_summary.get("freeze_id_v1") == "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1" else "FAIL",
            "observed_v1": benchmark_r6_summary.get("freeze_id_v1"),
            "expected_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        },
    ]
    failed_consistency = sum(1 for row in consistency_rows if row["status_v1"] != "PASS")

    hard_status = _hard_status(readiness_rows)

    summary = {
        "layer_name_v1": "MONDAY_NATIVE_SHADOW_REFREEZE_COMPARISON_V1",
        "built_at_utc_v1": _utc_now().isoformat(),
        "reports_root_v1": str(root),
        "extension_dir_v1": str(out_dir),
        "active_ledger_dir_v1": str(ledger_dir),
        "benchmark_snapshot_dir_v1": str(benchmark_snapshot_dir),
        "benchmark_r6_dir_v1": str(benchmark_r6_dir),
        "benchmark_r6_freeze_id_v1": benchmark_r6_summary.get("freeze_id_v1"),
        "comparator_dir_v1": str(comparator_dir),
        "decision_v1": decision,
        "decision_reason_v1": decision_reason,
        "monday_trade_foundation_v1": monday_foundation,
        "benchmark_trade_foundation_v1": benchmark_foundation,
        "entry_observability_v1": {
            "observed_direct_entry_rows_v1": entry_summary.get("observed_direct_entry_rows_v1"),
            "logged_action_counts_v1": entry_summary.get("logged_action_counts_v1"),
            "opportunity_rich_zero_trade_run_count_v1": entry_summary.get("opportunity_rich_zero_trade_run_count_v1"),
        },
        "management_runtime_v1": {
            "management_rows_v1": management_summary.get("management_rows_v1"),
            "observation_feature_count_v1": management_summary.get("observation_feature_count_v1"),
            "strict_sequence_row_count_v1": sequence_summary.get("strict_sequence_row_count_v1"),
            "bandit_dm_candidate_row_count_v1": bandit_summary.get("MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1"),
            "exit_local_status_v1": exit_local_summary.get("MANAGEMENT_EXIT_LOCAL_BASELINE_STATUS"),
        },
        "failed_consistency_count_v1": int(failed_consistency),
        "hard_status_division_v1": hard_status,
    }

    matrix_path = out_dir / ACTIVE_BENCHMARK_MATRIX
    readiness_path = out_dir / REFREEZE_READINESS_MATRIX
    consistency_path = out_dir / CONSISTENCY_AUDIT
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)
    with readiness_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(readiness_rows[0].keys()))
        writer.writeheader()
        writer.writerows(readiness_rows)
    with consistency_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(consistency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consistency_rows)

    _write_json(
        out_dir / CONTRACT,
        {
            "layer_name_v1": "MONDAY_NATIVE_SHADOW_REFREEZE_COMPARISON_CONTRACT_V1",
            "not_live_gate_v1": True,
            "purpose_v1": "Compare active Monday canonical substrate against locked pre-RL and R6 benchmark references without reviving retired Wednesday roots.",
            "active_root_v1": str(root),
            "benchmark_snapshot_v1": str(benchmark_snapshot_dir),
            "benchmark_r6_freeze_v1": str(benchmark_r6_dir),
        },
    )
    _write_json(out_dir / SUMMARY, summary)
    _write_json(out_dir / STATUS, {"decision_v1": decision, "not_live_gate_v1": True, "failed_consistency_count_v1": int(failed_consistency)})
    _write_json(
        out_dir / MANIFEST,
        {
            "layer_name_v1": "MONDAY_NATIVE_SHADOW_REFREEZE_COMPARISON_MANIFEST_V1",
            "artifacts_v1": [
                CONTRACT,
                ACTIVE_BENCHMARK_MATRIX,
                REFREEZE_READINESS_MATRIX,
                SUMMARY,
                REPORT,
                MANIFEST,
                STATUS,
                CONSISTENCY_AUDIT,
            ],
        },
    )
    (out_dir / REPORT).write_text(_report(summary=summary, matrix_rows=matrix_rows, readiness_rows=readiness_rows), encoding="utf-8")
    _write_json(root / TOP_LEVEL_SUMMARY, summary)
    return {"extension_dir": str(out_dir), "summary": summary, "status": {"decision_v1": decision, "not_live_gate_v1": True}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args(argv)
    result = materialize(args.reports_root, extension_dir=args.extension_dir)
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
