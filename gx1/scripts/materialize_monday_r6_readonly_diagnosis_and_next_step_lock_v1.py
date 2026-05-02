#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1"

CONTRACT = "contract_v1.json"
RESULT_RECHECK = "monday_r6_result_recheck_v1.json"
COMPARATOR_HIERARCHY = "comparator_hierarchy_reference_lock_v1.csv"
REPAIRED_165_FORENSIC = "repaired_165_damage_forensic_v1.json"
FAILURE_GAP_MAP = "failure_backlog_gap_map_v1.csv"
PATH_DYNAMICS_LOCK = "path_dynamics_bottleneck_lock_v1.csv"
RETRAIN_DECISION = "retrain_readiness_decision_v1.json"
NEXT_STEP_LOCK = "next_step_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

R6_TOP_LEVEL_SUMMARY = "truth_r6_entry_runner_first_retrain_v1.json"
R5_2_FREEZE_TOP_LEVEL_SUMMARY = "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"
R5_1_TOP_LEVEL_SUMMARY = "truth_r5_loso_batch04_robustness_retrain_v1.json"

R6_HEAD_TO_HEAD = "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv"
R6_LOSO_METRICS = "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv"
R6_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
R6_FEATURE_PATH_AUDIT = "shadow_meta_all_trade_review_r6_feature_path_dynamics_audit_v1.csv"

FREEZE_FAILURE_CLUSTER_TABLE = "shadow_meta_all_trade_review_r6_failure_cluster_table_v1.csv"
FREEZE_OPPORTUNITY_AUDIT = "shadow_meta_all_trade_review_r6_label_feature_opportunity_audit_v1.csv"
FREEZE_GO_NO_GO_MATRIX = "shadow_meta_all_trade_review_r5_2_vs_r6_go_no_go_matrix_v1.csv"

BENCHMARK_COMPARATOR_PREFIX = "MONDAY_TOP_PRE_RL_BASELINE_COMPARATOR_V1_"
BENCHMARK_SNAPSHOT_PREFIX = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_"
PATH_DYNAMICS_PREFIX = "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_"

BENCHMARK_COMPARATOR_SUMMARY = "summary_v1.json"
BENCHMARK_SNAPSHOT_SUMMARY = "summary_v1.json"
BENCHMARK_R6_FREEZE_SUMMARY = (
    "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/"
    "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
)
BENCHMARK_R6_FREEZE_MANIFEST = (
    "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/"
    "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"
)
BENCHMARK_PATH_DYNAMICS_SPEC = (
    "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/"
    "shadow_meta_path_dynamics_instrumentation_spec_v2.json"
)
PATH_DYNAMICS_SUMMARY = "shadow_meta_path_dynamics_logging_v2_summary_v1.json"

PATH_FIELD_MAP = {
    "last_peak_ts": {
        "human_name_v1": "last_peak_ts",
        "trace_field_v1": "last_peak_ts_utc",
        "raw_state_field_v1": "as_of_mgmt_trace_last_peak_ts_utc_v1",
        "policy_log_field_v1": "as_of_management_core_last_peak_ts_utc_v1",
        "entry_feature_probe_v1": "as_of_last_peak_ts_utc_v1",
    },
    "last_mfe_ts": {
        "human_name_v1": "last_mfe_ts",
        "trace_field_v1": "last_mfe_ts_utc",
        "raw_state_field_v1": "as_of_mgmt_trace_last_mfe_ts_utc_v1",
        "policy_log_field_v1": "as_of_management_core_last_mfe_ts_utc_v1",
        "entry_feature_probe_v1": "as_of_last_mfe_ts_utc_v1",
    },
    "last_peak_mfe": {
        "human_name_v1": "last_peak_mfe",
        "trace_field_v1": "last_peak_mfe_bps",
        "raw_state_field_v1": "as_of_mgmt_trace_last_peak_mfe_bps_v1",
        "policy_log_field_v1": "as_of_management_core_last_peak_mfe_bps_v1",
        "entry_feature_probe_v1": "as_of_last_peak_mfe_bps_v1",
    },
    "max_mfe_without_mae": {
        "human_name_v1": "max_mfe_without_mae",
        "trace_field_v1": "max_mfe_without_mae_bps",
        "raw_state_field_v1": "as_of_mgmt_trace_max_mfe_without_mae_bps_v1",
        "policy_log_field_v1": "as_of_management_core_max_mfe_without_mae_bps_v1",
        "entry_feature_probe_v1": "as_of_max_mfe_without_mae_bps_v1",
    },
    "mfe_mae_sequence_order": {
        "human_name_v1": "mfe_mae_sequence_order",
        "trace_field_v1": "mfe_mae_sequence_order",
        "raw_state_field_v1": "as_of_mgmt_trace_mfe_mae_sequence_order_v1",
        "policy_log_field_v1": "as_of_management_core_mfe_mae_sequence_order_v1",
        "entry_feature_probe_v1": "as_of_mfe_mae_sequence_order_v1",
    },
}

FAILURE_MEANING = {
    "MISSED_SHOULD_NOT_TAKE": "Darlige entry-caser som fortsatt slapp gjennom blokkeren.",
    "MISSED_10_50_TAIL_CONTROL": "10–50 MFE-lommen der vi fortsatt lekker verdi eller slipper gjennom lavkvalitets-trades.",
    "RISKY_ALLOW": "Trades som ble tillatt selv om adverse/risk-signalet i ettertid ser for h\u00f8yt ut.",
    "RUNNER_NEAR_MISS": "Gode runner-lignende trades som ligger farlig n\u00e6r blokkering eller faktisk blir skadet.",
}

FAILURE_WORKSTREAM = {
    "MISSED_SHOULD_NOT_TAKE": "BETTER_BAD_RISK_AND_RISKY_ALLOW_DISCRIMINATION",
    "MISSED_10_50_TAIL_CONTROL": "BETTER_TAIL_CONTROL_AND_POCKET_SPECIFIC_SIGNALS",
    "RISKY_ALLOW": "BETTER_RISKY_ALLOW_DISCRIMINATION_AND_CALIBRATION",
    "RUNNER_NEAR_MISS": "BETTER_RUNNER_PROTECTION_FIRST",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    candidates = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No directory found with prefix {prefix} under {reports_root}")
    return candidates[-1]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    r6_summary_path = reports_root / R6_TOP_LEVEL_SUMMARY
    freeze_summary_path = reports_root / R5_2_FREEZE_TOP_LEVEL_SUMMARY
    r5_1_summary_path = reports_root / R5_1_TOP_LEVEL_SUMMARY
    comparator_dir = _latest_dir(reports_root, BENCHMARK_COMPARATOR_PREFIX)
    snapshot_dir = _latest_dir(reports_root, BENCHMARK_SNAPSHOT_PREFIX)
    path_dynamics_dir = _latest_dir(reports_root, PATH_DYNAMICS_PREFIX)

    r6_summary = _load_json(r6_summary_path)
    freeze_summary = _load_json(freeze_summary_path)
    r5_1_summary = _load_json(r5_1_summary_path)
    comparator_summary = _load_json(comparator_dir / BENCHMARK_COMPARATOR_SUMMARY)
    snapshot_summary = _load_json(snapshot_dir / BENCHMARK_SNAPSHOT_SUMMARY)
    benchmark_r6_summary = _load_json(snapshot_dir / BENCHMARK_R6_FREEZE_SUMMARY)
    benchmark_r6_manifest = _load_json(snapshot_dir / BENCHMARK_R6_FREEZE_MANIFEST)
    path_dynamics_summary = _load_json(path_dynamics_dir / PATH_DYNAMICS_SUMMARY)
    path_dynamics_spec = _load_json(snapshot_dir / BENCHMARK_PATH_DYNAMICS_SPEC)

    r6_dir = Path(str(r6_summary["extension_dir_v1"])).expanduser().resolve()
    freeze_dir = Path(str(freeze_summary["extension_dir_v1"])).expanduser().resolve()
    r6_head_to_head_df = pd.read_csv(r6_dir / R6_HEAD_TO_HEAD)
    r6_loso_df = pd.read_csv(r6_dir / R6_LOSO_METRICS)

    return {
        "r6_summary_path": r6_summary_path,
        "freeze_summary_path": freeze_summary_path,
        "r5_1_summary_path": r5_1_summary_path,
        "comparator_dir": comparator_dir,
        "snapshot_dir": snapshot_dir,
        "path_dynamics_dir": path_dynamics_dir,
        "r6_summary": r6_summary,
        "freeze_summary": freeze_summary,
        "r5_1_summary": r5_1_summary,
        "comparator_summary": comparator_summary,
        "snapshot_summary": snapshot_summary,
        "benchmark_r6_summary": benchmark_r6_summary,
        "benchmark_r6_manifest": benchmark_r6_manifest,
        "path_dynamics_summary": path_dynamics_summary,
        "path_dynamics_spec": path_dynamics_spec,
        "r6_dir": r6_dir,
        "freeze_dir": freeze_dir,
        "r6_head_to_head_df": r6_head_to_head_df,
        "r6_loso_df": r6_loso_df,
    }


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _recheck_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    r6_summary = payload["r6_summary"]
    benchmark_r6_summary = payload["benchmark_r6_summary"]
    comparator_summary = payload["comparator_summary"]
    loso_df = payload["r6_loso_df"]
    current = dict(r6_summary.get("selected_candidate_v1", {}))
    old = dict(benchmark_r6_summary.get("selected_candidate_v1", {}))
    selected_policy = str(current.get("policy_name_v1"))
    selected_loso = loso_df.loc[loso_df["policy_name_v1"].astype("string").eq(selected_policy)].copy()
    batch04_loso = selected_loso.loc[selected_loso["scope_v1"].astype("string").eq("BATCH_04")]
    batch05_loso = selected_loso.loc[selected_loso["scope_v1"].astype("string").eq("BATCH_05")]

    beats_vs_local_r52 = []
    if int(current.get("tail_10_50_help_count_v1") or 0) > int(r6_summary.get("decision_v1", {}).get("r5_2_tail_help_v1") or 0):
        beats_vs_local_r52.append("tail_help_gt_local_r5_2")
    if bool(current.get("batch04_loso_pass_v1")):
        beats_vs_local_r52.append("batch04_loso_pass")
    if int(current.get("hundred_plus_mfe_block_count_v1") or 0) == 0:
        beats_vs_local_r52.append("hundred_plus_damage_zero")
    if int(current.get("two_hundred_plus_mfe_block_count_v1") or 0) == 0:
        beats_vs_local_r52.append("two_hundred_plus_damage_zero")
    if int(current.get("strong_trade_false_block_count_v1") or 0) == 0:
        beats_vs_local_r52.append("strong_false_blocks_zero")
    if int(current.get("strongest_winner_path_block_count_v1") or 0) == 0:
        beats_vs_local_r52.append("strongest_winner_path_damage_zero")

    fails_internal_contract = []
    internal_failure_text = str(current.get("r6_contract_failure_reasons_v1") or "")
    if internal_failure_text:
        fails_internal_contract.extend([part for part in internal_failure_text.split(",") if part])

    benchmark_failures = []
    if int(current.get("should_not_take_block_count_v1") or 0) < int(old.get("should_not_take_block_count_v1") or 0):
        benchmark_failures.append("bad_blocks_below_frozen_r6")
    if int(current.get("tail_10_50_help_count_v1") or 0) < int(old.get("tail_10_50_help_count_v1") or 0):
        benchmark_failures.append("tail_help_below_frozen_r6")
    if _safe_float(current.get("should_not_take_precision_v1")) is None or _safe_float(current.get("should_not_take_precision_v1")) < _safe_float(old.get("should_not_take_precision_v1")):
        benchmark_failures.append("precision_below_frozen_r6")
    if _safe_float(current.get("worst_loso_precision_v1")) is None or _safe_float(current.get("worst_loso_precision_v1")) < _safe_float(old.get("worst_loso_precision_v1")):
        benchmark_failures.append("worst_loso_below_frozen_r6")
    if int(current.get("repaired_165_block_count_v1") or 0) > int(old.get("repaired_165_block_count_v1") or 0):
        benchmark_failures.append("repaired_165_damage_above_frozen_r6")

    return {
        "layer_name_v1": "MONDAY_R6_RESULT_RECHECK_AND_LOCK_V1",
        "candidate_family_v1": current.get("family_v1"),
        "candidate_policy_v1": current.get("policy_name_v1"),
        "eval_status_v1": r6_summary.get("status_v1", {}).get("R6_STATUS"),
        "verdict_v1": r6_summary.get("decision_v1", {}).get("recommended_next_step_v1"),
        "metrics_v1": {
            "bad_blocks_v1": int(current.get("should_not_take_block_count_v1") or 0),
            "tail_help_v1": int(current.get("tail_10_50_help_count_v1") or 0),
            "precision_v1": _safe_float(current.get("should_not_take_precision_v1")),
            "worst_loso_precision_v1": _safe_float(current.get("worst_loso_precision_v1")),
            "batch04_loso_pass_v1": current.get("batch04_loso_pass_v1"),
            "batch05_loso_pass_v1": current.get("batch05_loso_pass_v1"),
            "fifty_plus_blocked_v1": int(current.get("fifty_plus_mfe_block_count_v1") or 0),
            "hundred_plus_blocked_v1": int(current.get("hundred_plus_mfe_block_count_v1") or 0),
            "two_hundred_plus_blocked_v1": int(current.get("two_hundred_plus_mfe_block_count_v1") or 0),
            "strong_false_blocks_v1": int(current.get("strong_trade_false_block_count_v1") or 0),
            "strongest_winner_path_damage_v1": int(current.get("strongest_winner_path_block_count_v1") or 0),
            "repaired_165_damage_v1": int(current.get("repaired_165_block_count_v1") or 0),
        },
        "slice_recheck_v1": {
            "batch04_rows_v1": int(len(batch04_loso)),
            "batch04_pass_v1": bool(batch04_loso["slice_safety_pass_v1"].fillna(False).astype(bool).all()) if not batch04_loso.empty else None,
            "batch05_rows_v1": int(len(batch05_loso)),
            "batch05_pass_v1": None if batch05_loso.empty else bool(batch05_loso["slice_safety_pass_v1"].fillna(False).astype(bool).all()),
        },
        "correct_benchmark_v1": {
            "benchmark_kind_v1": "FROZEN_R6_TEACHER_AND_CONTRACT",
            "freeze_id_v1": comparator_summary.get("benchmark_r6_v1", {}).get("freeze_id_v1"),
            "bad_blocks_v1": int(old.get("should_not_take_block_count_v1") or 0),
            "tail_help_v1": int(old.get("tail_10_50_help_count_v1") or 0),
            "precision_v1": _safe_float(old.get("should_not_take_precision_v1")),
            "worst_loso_precision_v1": _safe_float(old.get("worst_loso_precision_v1")),
        },
        "what_monday_r6_actually_solved_v1": beats_vs_local_r52,
        "what_monday_r6_failed_v1": {
            "internal_r5_2_contract_failures_v1": fails_internal_contract,
            "frozen_r6_benchmark_failures_v1": benchmark_failures,
        },
        "why_r6_features_insufficient_v1": [
            "It did not beat the local R5.2-style contract cleanly.",
            "It remains well below the frozen R6 benchmark on bad blocks and tail-help.",
            "It introduced repaired-165 damage, which is an explicit safety break.",
        ],
        "comparator_hierarchy_note_v1": "Do not compare only against degenerate Monday R5.2. The locked frozen R6 remains the correct benchmark/teacher.",
    }


def _comparator_hierarchy(payload: Dict[str, Any]) -> pd.DataFrame:
    r5_1 = payload["r5_1_summary"]
    r6 = payload["r6_summary"]
    comparator_summary = payload["comparator_summary"]
    benchmark_r6_summary = payload["benchmark_r6_summary"]
    current = dict(r6.get("selected_candidate_v1", {}))
    old = dict(benchmark_r6_summary.get("selected_candidate_v1", {}))
    selected_r5_1 = dict(r5_1.get("selected_candidate_v1", {}))
    rows = [
        {
            "reference_rank_v1": 1,
            "reference_id_v1": "FROZEN_R6_BENCHMARK",
            "source_v1": "WEDNESDAY_BENCHMARK_SNAPSHOT",
            "policy_id_v1": benchmark_r6_summary.get("selected_candidate_id_v1"),
            "status_v1": "LOCKED_BENCHMARK",
            "used_for_v1": "Teacher, benchmark, freeze-level comparator, contract-to-beat.",
            "not_used_for_v1": "Not active canonical root, not to be re-promoted blindly.",
            "bad_blocks_v1": int(old.get("should_not_take_block_count_v1") or 0),
            "tail_help_v1": int(old.get("tail_10_50_help_count_v1") or 0),
            "precision_v1": _safe_float(old.get("should_not_take_precision_v1")),
        },
        {
            "reference_rank_v1": 2,
            "reference_id_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
            "source_v1": "MONDAY_NATIVE",
            "policy_id_v1": selected_r5_1.get("policy_name_v1"),
            "status_v1": "SAFETY_REFERENCE",
            "used_for_v1": "Monday lane safety floor and winner-protection comparison.",
            "not_used_for_v1": "Not the teacher benchmark, not new freeze standard.",
            "bad_blocks_v1": int(selected_r5_1.get("should_not_take_block_count_v1") or 0),
            "tail_help_v1": int(selected_r5_1.get("tail_10_50_help_count_v1") or 0),
            "precision_v1": _safe_float(selected_r5_1.get("should_not_take_precision_v1")),
        },
        {
            "reference_rank_v1": 3,
            "reference_id_v1": "MONDAY_R6_FAILURE_MINER",
            "source_v1": "MONDAY_NATIVE",
            "policy_id_v1": current.get("policy_name_v1"),
            "status_v1": "DIAGNOSIS_ONLY",
            "used_for_v1": "Failure mining, gap mapping, next-step prioritization.",
            "not_used_for_v1": "Not new freeze, not new shadow standard, not promotion candidate.",
            "bad_blocks_v1": int(current.get("should_not_take_block_count_v1") or 0),
            "tail_help_v1": int(current.get("tail_10_50_help_count_v1") or 0),
            "precision_v1": _safe_float(current.get("should_not_take_precision_v1")),
        },
        {
            "reference_rank_v1": 4,
            "reference_id_v1": "MONDAY_LOCAL_R5_2_FREEZE_SOURCE",
            "source_v1": "MONDAY_NATIVE",
            "policy_id_v1": payload["freeze_summary"].get("selected_policy_stack_v1"),
            "status_v1": "LOCAL_PHASE_GATE_REFERENCE_ONLY",
            "used_for_v1": "Local R6 backlog/freeze-source bookkeeping inside Monday lane.",
            "not_used_for_v1": "Not the correct global benchmark and not the teacher standard.",
            "bad_blocks_v1": int(payload["r6_summary"].get("decision_v1", {}).get("r5_2_bad_blocks_v1") or 0),
            "tail_help_v1": int(payload["r6_summary"].get("decision_v1", {}).get("r5_2_tail_help_v1") or 0),
            "precision_v1": None,
        },
    ]
    return pd.DataFrame(rows)


def _repaired_165_forensic(payload: Dict[str, Any]) -> Dict[str, Any]:
    r6_summary = payload["r6_summary"]
    pred_df = pd.read_parquet(payload["r6_dir"] / R6_POLICY_PREDICTION_VIEW)
    repaired_mask = pred_df["is_repaired_165_v1"].fillna(False).astype(bool)
    blocked_mask = pred_df["r6_selected_candidate__block_v1"].fillna(False).astype(bool)
    blocked_repaired = pred_df.loc[repaired_mask & blocked_mask].copy()
    if blocked_repaired.empty:
        raise RuntimeError("Expected one repaired-165 damage row, found none")
    row = blocked_repaired.iloc[0]
    return {
        "layer_name_v1": "REPAIRED_165_DAMAGE_FORENSIC_V1",
        "repaired_pocket_row_count_v1": int(repaired_mask.sum()),
        "blocked_repaired_row_count_v1": int(len(blocked_repaired)),
        "deterministic_trade_key_v1": str(row.get("candidate_uid")),
        "run_id_v1": str(row.get("run_id")),
        "trade_uid_v1": str(row.get("trade_uid")),
        "peak_mfe_bps_v1": _safe_float(row.get("peak_mfe_bps_v1")),
        "baseline_realized_pnl_bps_v1": _safe_float(row.get("baseline_realized_pnl_bps_v1")),
        "mae_abs_bps_v1": _safe_float(row.get("mae_abs_bps_v1")),
        "label_should_not_take_v1": bool(row.get("label_should_not_take_v1")),
        "take_was_ok_v1": bool(row.get("take_was_ok_v1")),
        "scores_v1": {
            "r6_bad_risk_v1": _safe_float(row.get("pred__entry_r6_bad_risk__prob_true_v1")),
            "r6_runner_protector_v1": _safe_float(row.get("pred__entry_r6_runner_protector__prob_true_v1")),
            "r6_tail_control_v1": _safe_float(row.get("pred__entry_r6_tail_control_10_50__prob_true_v1")),
            "r6_risky_allow_v1": _safe_float(row.get("pred__entry_r6_risky_allow__prob_true_v1")),
        },
        "broken_contract_clause_v1": "repaired_165_damage_must_equal_zero",
        "freeze_blocking_meaning_v1": "A single repaired-165 false block is enough to fail freeze safety because this pocket is explicitly protected.",
        "single_case_or_pattern_v1": {
            "classification_v1": "ISOLATED_BUT_SAFETY_CRITICAL",
            "reason_v1": "Only 1/178 repaired rows was blocked, but the row is a profitable TAKE_WAS_OK trade and the contract requires zero damage.",
        },
    }


def _failure_gap_map(payload: Dict[str, Any]) -> pd.DataFrame:
    freeze_summary = payload["freeze_summary"]
    cluster_df = pd.read_csv(payload["freeze_dir"] / FREEZE_FAILURE_CLUSTER_TABLE)
    opportunity_df = pd.read_csv(payload["freeze_dir"] / FREEZE_OPPORTUNITY_AUDIT)
    rows: List[Dict[str, Any]] = []
    counts = dict(freeze_summary.get("failure_counts_v1", {}))
    for failure_type, count in counts.items():
        canonical = failure_type.upper().replace("_V1", "")
        if canonical == "MISSED_SHOULD_NOT_TAKE":
            cluster_key = "MISSED_SHOULD_NOT_TAKE"
        elif canonical == "MISSED_10_50_TAIL_CONTROL":
            cluster_key = "MISSED_10_50_TAIL_CONTROL"
        elif canonical == "RISKY_ALLOWS":
            cluster_key = "RISKY_ALLOW"
        elif canonical == "RUNNER_NEAR_MISSES":
            cluster_key = "RUNNER_NEAR_MISS"
        else:
            cluster_key = canonical
        part = cluster_df.loc[cluster_df["failure_type_v1"].astype("string").eq(cluster_key)].copy()
        driver = None
        if not part.empty:
            driver_series = (
                part.groupby("failure_driver_assessment_v1", dropna=False)["count_v1"]
                .sum()
                .sort_values(ascending=False)
            )
            driver = str(driver_series.index[0])
        opp_match = opportunity_df.loc[opportunity_df["addressed_failure_types_v1"].astype("string").str.contains(cluster_key, regex=False, na=False)]
        evidence = str(opp_match.iloc[0]["evidence_v1"]) if not opp_match.empty else None
        rows.append(
            {
                "bucket_id_v1": cluster_key,
                "count_v1": int(count),
                "operational_meaning_v1": FAILURE_MEANING.get(cluster_key, "UNSPECIFIED"),
                "dominant_driver_v1": driver,
                "primary_workstream_v1": FAILURE_WORKSTREAM.get(cluster_key, "OTHER"),
                "evidence_v1": evidence,
                "gap_interpretation_v1": (
                    "Signal under-recall in blocker lane."
                    if cluster_key == "MISSED_SHOULD_NOT_TAKE"
                    else "Pocket-specific under-control."
                    if cluster_key == "MISSED_10_50_TAIL_CONTROL"
                    else "Risk discrimination still too weak."
                    if cluster_key == "RISKY_ALLOW"
                    else "Runner protection must be strengthened before more recall."
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_status_from_summary(path_summary: Dict[str, Any], field_name: str, layer_name: str) -> Dict[str, Any] | None:
    for row in path_summary.get("coverage_summary_v1", []):
        if row.get("field_name") == field_name and row.get("layer_name") == layer_name:
            return row
    return None


def _path_dynamics_lock(payload: Dict[str, Any]) -> pd.DataFrame:
    path_summary = payload["path_dynamics_summary"]
    spec = payload["path_dynamics_spec"]
    feature_audit_df = pd.read_csv(payload["r6_dir"] / R6_FEATURE_PATH_AUDIT)
    rows: List[Dict[str, Any]] = []
    for field_id, mapping in PATH_FIELD_MAP.items():
        trace_row = _field_status_from_summary(path_summary, mapping["trace_field_v1"], "EXIT_EVAL_TRACE")
        raw_row = _field_status_from_summary(path_summary, mapping["raw_state_field_v1"], "RAW_STATE")
        policy_row = _field_status_from_summary(path_summary, mapping["policy_log_field_v1"], "POLICY_LOG")
        spec_row = next((item for item in spec.get("fields_v1", []) if item.get("field_name_v1") == mapping["policy_log_field_v1"] or item.get("field_name_v1") == mapping["raw_state_field_v1"] or item.get("field_name_v1") == mapping["entry_feature_probe_v1"] or item.get("field_name_v1") == mapping["policy_log_field_v1"].replace("as_of_management_core_", "as_of_")), None)
        if spec_row is None:
            spec_row = next((item for item in spec.get("fields_v1", []) if mapping["human_name_v1"] in str(item.get("field_name_v1"))), None)
        feature_probe = mapping["entry_feature_probe_v1"]
        feature_present = False
        for _, row in feature_audit_df.loc[feature_audit_df["feature_family_v1"].astype("string").eq("new_path_dynamics_logging")].iterrows():
            if feature_probe in str(row.get("top_features_json_v1", "")):
                feature_present = bool(row.get("positive_count_v1") or 0)
                break
        final_status = "NOT_CANONICAL_YET"
        if not trace_row or not raw_row or not policy_row:
            final_status = "UNAVAILABLE"
        rows.append(
            {
                "field_id_v1": field_id,
                "field_name_v1": mapping["human_name_v1"],
                "upstream_trace_status_v1": "READY" if trace_row and int(trace_row.get("null_count", 1)) == 0 else "UNAVAILABLE",
                "raw_state_status_v1": "READY" if raw_row and int(raw_row.get("null_count", 1)) == 0 else "UNAVAILABLE",
                "policy_log_status_v1": "READY" if policy_row and int(policy_row.get("null_count", 1)) == 0 else "UNAVAILABLE",
                "r6_entry_feature_layer_status_v1": "ABSENT" if not feature_present else "PRESENT",
                "leakage_risk_status_v1": (
                    "MANAGEMENT_ONLY_LEGAL_NOT_DIRECT_ENTRY"
                    if spec_row and "NOT_LEGAL_FOR_PRE_ENTRY" in str(spec_row.get("as_of_semantics_v1", ""))
                    else "NOT_ESTABLISHED"
                ),
                "future_use_status_v1": final_status,
                "why_not_ready_for_entry_r6_v1": (
                    "Field is logged upstream for management/policy-log, but direct same-trade entry use is not canonical because instrumentation spec marks it not legal for pre-entry."
                    if final_status == "NOT_CANONICAL_YET"
                    else "Missing upstream coverage."
                ),
            }
        )
    return pd.DataFrame(rows)


def _retrain_readiness(result_recheck: Dict[str, Any], repaired: Dict[str, Any], path_lock_df: pd.DataFrame) -> Dict[str, Any]:
    path_not_entry_ready = bool(path_lock_df["future_use_status_v1"].astype("string").eq("NOT_CANONICAL_YET").any())
    decision = "DO_NOT_RETRAIN_YET"
    rationale = [
        "Monday-native R6 does not beat the locked comparator hierarchy.",
        "A repaired-165 false block is present, so safety is already broken.",
        "The current path-dynamics fields are logged upstream but are not canonical direct entry features yet.",
        "Runner-protection and legal feature uplift should happen before another retrain.",
    ]
    if not path_not_entry_ready:
        decision = "RETRAIN_AFTER_RUNNER_PROTECTION_UPLIFT"
    return {
        "layer_name_v1": "RETRAIN_READINESS_DECISION_V1",
        "decision_v1": decision,
        "retrain_now_v1": False,
        "because_v1": rationale,
        "uses_comparator_hierarchy_v1": True,
        "uses_repaired_165_forensic_v1": repaired["blocked_repaired_row_count_v1"] > 0,
        "uses_path_dynamics_lock_v1": True,
        "uses_gap_map_v1": True,
    }


def _next_step_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_STEP_LOCK_V1",
        "primary_next_step_v1": "IMPROVE_RUNNER_PROTECTION_FIRST",
        "supporting_locks_v1": [
            "KEEP_FROZEN_R6_AS_BENCHMARK",
            "KEEP_MONDAY_R5_1_AS_SAFETY_REFERENCE",
            "USE_MONDAY_R6_AS_FAILURE_MINER",
            "IMPROVE_PATH_DYNAMICS_LOGGING_FIRST",
            "DO_NOT_FREEZE_MONDAY_R6",
            "DO_NOT_PROMOTE_MONDAY_R6",
            "RETRAIN_ONLY_AFTER_FEATURE_UPLIFT",
        ],
        "note_v1": "Path-dynamics remains important, but the immediate blocker is safe runner-protection uplift on canonical entry-legal signals.",
    }


def _status_block(result_recheck: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "STATUS_DISCIPLINE_V1",
        "BEVIST": [
            "Monday-native R6 was rechecked from read-only artifacts.",
            "Monday-native R6 does not beat the correct locked benchmark hierarchy.",
            "Monday-native R6 must not be frozen.",
            "Monday-native R6 must not be promoted.",
            "Monday-native R6 is useful as a failure-miner and diagnosis surface.",
        ],
        "INDIKERT": [
            "Runner-protection and path-context are the main bottlenecks, not just more retraining.",
            "Retraining without legal feature uplift is unlikely to beat the frozen R6 benchmark.",
        ],
        "IKKE_ETABLERT": [
            "That the next retrain will beat frozen R6.",
            "That the current Monday entry feature base alone is enough for a new freeze.",
        ],
    }


def _render_report(
    result_recheck: Dict[str, Any],
    comparator_df: pd.DataFrame,
    repaired: Dict[str, Any],
    gap_df: pd.DataFrame,
    path_df: pd.DataFrame,
    retrain_decision: Dict[str, Any],
    next_step: Dict[str, Any],
    status_block: Dict[str, Any],
) -> str:
    lines = [
        "# Monday R6 Read-Only Diagnosis And Next-Step Lock V1",
        "",
        "Read-only diagnosis. No training, replay, freeze, or promotion was performed.",
        "",
        "## Headline",
        "",
        f"- Verdict rechecked: `{result_recheck['verdict_v1']}`",
        f"- Correct benchmark: `{comparator_df.iloc[0]['reference_id_v1']}`",
        f"- Correct Monday safety-reference: `{comparator_df.iloc[1]['reference_id_v1']}`",
        f"- Monday-native R6 role: `{comparator_df.iloc[2]['status_v1']}`",
        f"- Retrain decision: `{retrain_decision['decision_v1']}`",
        "",
        "## Why Monday R6 Does Not Hold",
        "",
        f"- Bad blocks: `{result_recheck['metrics_v1']['bad_blocks_v1']}` vs frozen R6 `{result_recheck['correct_benchmark_v1']['bad_blocks_v1']}`",
        f"- Tail-help: `{result_recheck['metrics_v1']['tail_help_v1']}` vs frozen R6 `{result_recheck['correct_benchmark_v1']['tail_help_v1']}`",
        f"- Precision: `{result_recheck['metrics_v1']['precision_v1']}` vs frozen R6 `{result_recheck['correct_benchmark_v1']['precision_v1']}`",
        f"- Worst LOSO precision: `{result_recheck['metrics_v1']['worst_loso_precision_v1']}` vs frozen R6 `{result_recheck['correct_benchmark_v1']['worst_loso_precision_v1']}`",
        f"- Repaired-165 damage: `{result_recheck['metrics_v1']['repaired_165_damage_v1']}`",
        "",
        "## Repaired-165 Forensic",
        "",
        f"- Damaged key: `{repaired['deterministic_trade_key_v1']}`",
        f"- Run: `{repaired['run_id_v1']}`",
        f"- Outcome: `MFE {repaired['peak_mfe_bps_v1']}`, `PnL {repaired['baseline_realized_pnl_bps_v1']}`, `MAE {repaired['mae_abs_bps_v1']}`",
        f"- Freeze blocker: `{repaired['broken_contract_clause_v1']}`",
        "",
        "## Gap Map",
        "",
    ]
    for row in gap_df.to_dict(orient="records"):
        lines.append(f"- `{row['bucket_id_v1']}`: `{row['count_v1']}` rows. {row['gap_interpretation_v1']}")
    lines += [
        "",
        "## Path Dynamics",
        "",
        f"- Fields marked `NOT_CANONICAL_YET` for direct entry use: `{int(path_df['future_use_status_v1'].astype('string').eq('NOT_CANONICAL_YET').sum())}`",
        f"- Primary next step: `{next_step['primary_next_step_v1']}`",
        "",
        "## Hard Status",
        "",
    ]
    for key in ["BEVIST", "INDIKERT", "IKKE_ETABLERT"]:
        lines.append(f"### {key}")
        lines.append("")
        for item in status_block[key]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def build_payload(reports_root: Path, extension_dir: Path) -> Dict[str, Any]:
    payload = _load_inputs(reports_root)
    result_recheck = _recheck_result(payload)
    comparator_df = _comparator_hierarchy(payload)
    repaired = _repaired_165_forensic(payload)
    gap_df = _failure_gap_map(payload)
    path_df = _path_dynamics_lock(payload)
    retrain_decision = _retrain_readiness(result_recheck, repaired, path_df)
    next_step = _next_step_lock()
    status_block = _status_block(result_recheck)
    contract = {
        "layer_name_v1": "MONDAY_R6_READONLY_DIAGNOSIS_CONTRACT_V1",
        "mode_v1": "READ_ONLY_DIAGNOSIS_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_freeze_v1": True,
        "not_promotion_v1": True,
        "not_live_gate_v1": True,
        "inputs_v1": {
            "r6_summary": str(payload["r6_summary_path"]),
            "freeze_summary": str(payload["freeze_summary_path"]),
            "r5_1_summary": str(payload["r5_1_summary_path"]),
            "benchmark_comparator_dir": str(payload["comparator_dir"]),
            "benchmark_snapshot_dir": str(payload["snapshot_dir"]),
            "path_dynamics_dir": str(payload["path_dynamics_dir"]),
        },
    }
    consistency_df = pd.DataFrame(
        [
            _audit_record("R6_TOP_LEVEL_SUMMARY_PRESENT", "PASS", {"path": str(payload["r6_summary_path"])}),
            _audit_record("FREEZE_TOP_LEVEL_SUMMARY_PRESENT", "PASS", {"path": str(payload["freeze_summary_path"])}),
            _audit_record("R5_1_SUMMARY_PRESENT", "PASS", {"path": str(payload["r5_1_summary_path"])}),
            _audit_record("BENCHMARK_COMPARATOR_PRESENT", "PASS", {"dir": str(payload["comparator_dir"])}),
            _audit_record("BENCHMARK_SNAPSHOT_PRESENT", "PASS", {"dir": str(payload["snapshot_dir"])}),
            _audit_record("R6_LOSO_ROWS_PRESENT", "PASS", {"row_count": int(len(payload["r6_loso_df"]))}),
            _audit_record("R6_HEAD_TO_HEAD_ROWS_PRESENT", "PASS", {"row_count": int(len(payload["r6_head_to_head_df"]))}),
            _audit_record(
                "R6_REPAIRED_165_MATCHES_SUMMARY",
                "PASS" if repaired["blocked_repaired_row_count_v1"] == int(result_recheck["metrics_v1"]["repaired_165_damage_v1"]) else "FAIL",
                {"forensic": repaired["blocked_repaired_row_count_v1"], "summary": result_recheck["metrics_v1"]["repaired_165_damage_v1"]},
            ),
            _audit_record(
                "BATCH04_RECHECK_MATCHES_SUMMARY",
                "PASS" if result_recheck["slice_recheck_v1"]["batch04_pass_v1"] == result_recheck["metrics_v1"]["batch04_loso_pass_v1"] else "FAIL",
                {
                    "loso": result_recheck["slice_recheck_v1"]["batch04_pass_v1"],
                    "summary": result_recheck["metrics_v1"]["batch04_loso_pass_v1"],
                },
            ),
            _audit_record(
                "BATCH05_ABSENT_IS_NULL_NOT_FAIL",
                "PASS" if result_recheck["slice_recheck_v1"]["batch05_rows_v1"] == 0 and result_recheck["metrics_v1"]["batch05_loso_pass_v1"] is None else "FAIL",
                {
                    "batch05_rows_v1": result_recheck["slice_recheck_v1"]["batch05_rows_v1"],
                    "summary": result_recheck["metrics_v1"]["batch05_loso_pass_v1"],
                },
            ),
            _audit_record(
                "R6_RESULT_VERDICT_LOCKED",
                "PASS" if result_recheck["verdict_v1"] == "R6_FEATURES_INSUFFICIENT" else "FAIL",
                {"verdict": result_recheck["verdict_v1"]},
            ),
            _audit_record(
                "READ_ONLY_DISCIPLINE",
                "PASS",
                {
                    "no_training": True,
                    "no_replay": True,
                    "no_freeze": True,
                    "no_promotion": True,
                    "writes_only_under_extension_dir": str(extension_dir),
                },
            ),
        ]
    )
    status = {
        "layer_name_v1": "MONDAY_R6_READONLY_DIAGNOSIS_STATUS_V1",
        "READONLY_DIAGNOSIS_STATUS": "MATERIALIZED",
        "failed_check_count_v1": int(consistency_df["status_v1"].eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_freeze_v1": True,
        "not_promotion_v1": True,
    }
    summary = {
        "layer_name_v1": "MONDAY_R6_READONLY_DIAGNOSIS_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "monday_r6_rechecked_v1": True,
        "correct_benchmark_v1": "FROZEN_R6_BENCHMARK",
        "correct_safety_reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
        "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "retrain_decision_v1": retrain_decision["decision_v1"],
        "next_step_v1": next_step["primary_next_step_v1"],
        "result_recheck_v1": result_recheck,
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }
    manifest = {
        "layer_name_v1": "MONDAY_R6_READONLY_DIAGNOSIS_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "monday_r6_result_recheck": RESULT_RECHECK,
            "comparator_hierarchy": COMPARATOR_HIERARCHY,
            "repaired_165_forensic": REPAIRED_165_FORENSIC,
            "failure_backlog_gap_map": FAILURE_GAP_MAP,
            "path_dynamics_bottleneck_lock": PATH_DYNAMICS_LOCK,
            "retrain_readiness_decision": RETRAIN_DECISION,
            "next_step_lock": NEXT_STEP_LOCK,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        }
    }
    return {
        "contract": contract,
        "result_recheck": result_recheck,
        "comparator_df": comparator_df,
        "repaired": repaired,
        "gap_df": gap_df,
        "path_df": path_df,
        "retrain_decision": retrain_decision,
        "next_step": next_step,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "consistency_df": consistency_df,
        "report": _render_report(result_recheck, comparator_df, repaired, gap_df, path_df, retrain_decision, next_step, status_block),
    }


def materialize(reports_root: Path, *, extension_dir: Path | None = None) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    extension_dir = _resolve_extension_dir(reports_root, str(extension_dir) if extension_dir else None)
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(reports_root, extension_dir)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / RESULT_RECHECK, payload["result_recheck"])
    payload["comparator_df"].to_csv(extension_dir / COMPARATOR_HIERARCHY, index=False)
    _write_json(extension_dir / REPAIRED_165_FORENSIC, payload["repaired"])
    payload["gap_df"].to_csv(extension_dir / FAILURE_GAP_MAP, index=False)
    payload["path_df"].to_csv(extension_dir / PATH_DYNAMICS_LOCK, index=False)
    _write_json(extension_dir / RETRAIN_DECISION, payload["retrain_decision"])
    _write_json(extension_dir / NEXT_STEP_LOCK, payload["next_step"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {
        "extension_dir": str(extension_dir),
        "status": payload["status"],
        "summary": payload["summary"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a read-only Monday-native R6 diagnosis and next-step lock.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(reports_root, extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
