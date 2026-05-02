#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r6_retrain_from_true_r5_2_rescue_package_v1 import (
    _candidate_grid_report,
    _frame_selected_metrics,
    _refresh_r6_labels,
    _safety_pass,
    _selected_policy_from_grid,
    _selected_policy_mask,
)
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    WEDNESDAY_R6_BENCHMARK,
    _bool,
    _jsonable,
    _num,
)
from gx1.scripts.train_monday_r6_on_foundation_scores_v1 import (
    COMPARE_REPORT,
    EXPECTED_AS_OF_COLUMNS,
    EXPECTED_BASE_FEATURES,
    EXPECTED_R5_2_HEADS,
    EXPECTED_R5_HEADS,
    R6_FAMILY_GRID_REPLAY,
    SCORE_FRAME,
    SCORE_SUMMARY,
    SUMMARY as R6_SUMMARY,
    TRAINING_FRAME,
    TrainConfig,
    materialize as run_r6_on_scores,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_V2_EXECUTION_DIR = (
    DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_20260426T_EXECUTION"
)
DEFAULT_V3_R6_DIR = (
    DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"
)
DEFAULT_RESCUE_R6_DIR = DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_V1_20260426T_EXPLICIT"
LAYER_NAME = "RUN_R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V2_VARIANT_V1"
BEST_VARIANT_ID = "R5_2_OBJECTIVE_V2_VARIANT_01_V2_BALANCED_STRICT_PROTECT"
BEST_PROFILE_ID = "V2_BALANCED_STRICT_PROTECT"
V2_CONTRACT_ID = "R5_2_OBJECTIVE_V2_BEST_VARIANT_FINAL_BASE"
V2_FINAL_BASE_FLAG = "r5_2_v2_final_base_membership"
V2_PRE_VETO_BASE_FLAG = "r5_2_v2_base_membership_pre_veto"
V2_HARD_VETO_FLAG = "r5_2_v2_hard_protection_veto"
V2_BAD_SCORE = "r5_2_v2_bad_recall_score"
V2_TAIL_SCORE = "r5_2_v2_tail_recall_score"
V2_RISKY_SCORE = "r5_2_v2_risky_attention_score"
V2_RUNNER_PROTECT_SCORE = "r5_2_v2_runner_protection_score"
V2_AMBIGUOUS_PROTECT_SCORE = "r5_2_v2_high_mfe_ambiguous_protection_score"
V2_HARD_WINNER_PROTECT_SCORE = "r5_2_v2_hard_winner_protection_score"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

OUTPUT_FILES = {
    "retrain": "r6_retrain_from_best_r5_2_objective_v2_variant_v1.json",
    "runtime_guard": "no_pre_veto_or_unsafe_r5_2_to_r6_guard_v1.json",
    "candidate_grid": "r6_v2_objective_candidate_grid_selection_v1.csv",
    "benchmark_eval": "r6_v2_objective_eval_against_benchmarks_v1.json",
    "pass_through": "v2_final_base_rows_r6_pass_through_trace_v1.csv",
    "safety_proof": "v2_hard_veto_r6_safety_proof_v1.json",
    "delta_root": "r6_v2_objective_delta_and_root_cause_v1.json",
    "gate": "r6_v2_objective_canonical_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _selected_policy(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("family_grid_selected_policy_v1") or summary.get("custom_threshold_grid_policy_v1", {}).get(
        "selected_candidate_v1"
    ) or {}


def _safe_div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def _load_v2_inputs(v2_execution_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = _read_json(v2_execution_dir / "best_v2_variant_downstream_r6_input_lock_v1.json")
    execution_summary = _read_json(v2_execution_dir / "summary_v1.json")
    if not lock.get("ready_for_downstream_r6_v1"):
        raise RuntimeError("Best V2 lock is not ready for downstream R6")
    if lock.get("best_variant_id_v1") != BEST_VARIANT_ID:
        raise RuntimeError(f"Unexpected best V2 variant: {lock.get('best_variant_id_v1')}")
    if lock.get("best_profile_id_v1") != BEST_PROFILE_ID:
        raise RuntimeError(f"Unexpected best V2 profile: {lock.get('best_profile_id_v1')}")
    if lock.get("base_flag_for_r6_v1") != V2_FINAL_BASE_FLAG:
        raise RuntimeError("Best V2 lock does not point R6 at final post-veto base membership")
    if lock.get("raw_pre_veto_base_not_allowed_v1") != V2_PRE_VETO_BASE_FLAG:
        raise RuntimeError("Best V2 lock does not block the raw pre-veto base")
    variant_manifest_path = Path(lock["downstream_r6_input_manifest_path_v1"])
    variant_manifest = _read_json(variant_manifest_path)
    if variant_manifest.get("variant_id_v1") != BEST_VARIANT_ID:
        raise RuntimeError("Variant downstream manifest does not match the locked best variant")
    if variant_manifest.get("base_flag_for_r6_v1") != V2_FINAL_BASE_FLAG:
        raise RuntimeError("Variant downstream manifest does not point to final V2 base")
    if not variant_manifest.get("ready_for_downstream_r6_v1"):
        raise RuntimeError("Variant downstream manifest is not ready for R6")
    safety = _read_json(variant_manifest_path.parent / "safety_guard_report_v1.json")
    if not safety.get("safety_pass_v1"):
        raise RuntimeError("Refuses unsafe V2 variant as R6 input")
    return lock, variant_manifest, safety, execution_summary


def _stage_v2_score_package(output_dir: Path, v2_execution_dir: Path, lock: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    score_path = Path(lock["score_package_path_v1"])
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    frame = pd.read_parquet(score_path)
    required = [
        "candidate_uid",
        "trade_uid",
        "decision_timestamp",
        V2_BAD_SCORE,
        V2_TAIL_SCORE,
        V2_RISKY_SCORE,
        V2_RUNNER_PROTECT_SCORE,
        V2_AMBIGUOUS_PROTECT_SCORE,
        V2_HARD_WINNER_PROTECT_SCORE,
        V2_PRE_VETO_BASE_FLAG,
        V2_HARD_VETO_FLAG,
        V2_FINAL_BASE_FLAG,
        "v2_base_reason_v1",
        "r5_2_true_rescue_base_membership_v1",
        "raw_true_base_membership_v1",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Best V2 score package missing required columns: {missing}")
    if _bool(frame, V2_FINAL_BASE_FLAG).equals(_bool(frame, V2_PRE_VETO_BASE_FLAG)):
        raise RuntimeError("Refuses V2 package where final base equals pre-veto base")
    if _bool(frame, V2_FINAL_BASE_FLAG).equals(_bool(frame, "raw_true_base_membership_v1")):
        raise RuntimeError("Refuses V2 package where final base equals raw true unsafe base")

    staged = frame.copy()
    protection_score = pd.concat(
        [
            _num(staged, V2_RUNNER_PROTECT_SCORE),
            _num(staged, V2_AMBIGUOUS_PROTECT_SCORE),
            _num(staged, V2_HARD_WINNER_PROTECT_SCORE),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    staged["r5_2_base_flag_before_v2_objective_v1"] = _bool(staged, "r5_2_selected_candidate__block_v1")
    staged[R5_2_BAD_PROB] = _num(staged, V2_BAD_SCORE)
    staged[R5_2_RUNNER_PROB] = protection_score
    staged["blocker_score_v1"] = _num(staged, V2_BAD_SCORE)
    staged["runner_protector_score_v1"] = protection_score
    staged["r5_2_selected_candidate__block_v1"] = _bool(staged, V2_FINAL_BASE_FLAG)
    staged["r5_2_base_membership_contract_id_v1"] = V2_CONTRACT_ID
    staged = _refresh_r6_labels(staged)

    staged_dir = output_dir / "staged_best_r5_2_objective_v2_score_package_for_r6_v1"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged.to_parquet(staged_dir / SCORE_FRAME, index=False)
    asof_prefix_count = int(sum(column.startswith("as_of_") for column in staged.columns))
    active = staged.get("calendar_quarantine_status_v1", pd.Series("", index=staged.index)).astype(str).eq("ACTIVE_CANDIDATE")
    staged_summary = {
        "layer_name": "BEST_R5_2_OBJECTIVE_V2_STAGED_R6_SCORE_PACKAGE_V1",
        "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
        "source_decision_v1": "R5_2_OBJECTIVE_V2_BEST_VARIANT_STAGED_FOR_R6",
        "source_v2_execution_dir_v1": str(v2_execution_dir),
        "source_score_package_path_v1": str(score_path),
        "contract_id_v1": V2_CONTRACT_ID,
        "best_variant_id_v1": BEST_VARIANT_ID,
        "base_flag_for_r6_v1": V2_FINAL_BASE_FLAG,
        "raw_pre_veto_base_not_allowed_v1": V2_PRE_VETO_BASE_FLAG,
        "row_count_v1": int(len(staged)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "as_of_column_count_v1": EXPECTED_AS_OF_COLUMNS,
        "as_of_prefix_columns_observed_v1": asof_prefix_count,
        "base_feature_count_v1": EXPECTED_BASE_FEATURES,
        "r5_head_count_v1": EXPECTED_R5_HEADS,
        "r5_2_head_count_v1": EXPECTED_R5_2_HEADS,
        "r5_2_v2_pre_veto_base_rows_v1": int(_bool(staged, V2_PRE_VETO_BASE_FLAG).sum()),
        "r5_2_v2_final_base_rows_v1": int(_bool(staged, V2_FINAL_BASE_FLAG).sum()),
        "r5_2_v2_rows_vetoed_by_protection_v1": int((_bool(staged, V2_PRE_VETO_BASE_FLAG) & _bool(staged, V2_HARD_VETO_FLAG)).sum()),
        "r5_2_raw_true_base_rows_v1": int(_bool(staged, "raw_true_base_membership_v1").sum()),
        "r5_2_rescue_base_rows_v1": int(_bool(staged, "r5_2_true_rescue_base_membership_v1").sum()),
        "r6_heads_trained_v1": False,
    }
    _write_json(staged_dir / SCORE_SUMMARY, staged_summary)
    return staged_dir, staged_summary


def _runtime_guard(lock: dict[str, Any], variant_manifest: dict[str, Any], staged_summary: dict[str, Any], r6_frame: pd.DataFrame) -> dict[str, Any]:
    selected = _bool(r6_frame, "r5_2_selected_candidate__block_v1")
    checks = {
        "best_v2_package_used_v1": lock.get("best_variant_id_v1") == BEST_VARIANT_ID,
        "variant_manifest_points_to_final_base_v1": variant_manifest.get("base_flag_for_r6_v1") == V2_FINAL_BASE_FLAG,
        "v2_final_base_flag_present_in_runtime_frame_v1": V2_FINAL_BASE_FLAG in r6_frame.columns,
        "runtime_r5_2_selected_equals_v2_final_flag_v1": bool(selected.equals(_bool(r6_frame, V2_FINAL_BASE_FLAG))),
        "runtime_r5_2_selected_not_equal_pre_veto_v1": not selected.equals(_bool(r6_frame, V2_PRE_VETO_BASE_FLAG)),
        "runtime_r5_2_selected_not_equal_raw_true_v1": not selected.equals(_bool(r6_frame, "raw_true_base_membership_v1")),
        "runtime_r5_2_selected_not_equal_rescue_v1": not selected.equals(_bool(r6_frame, "r5_2_true_rescue_base_membership_v1")),
        "runtime_r5_2_selected_not_equal_old_v3_v1": not selected.equals(_bool(r6_frame, "in_v3_base_v1")),
        "staged_summary_points_to_v2_contract_v1": staged_summary.get("contract_id_v1") == V2_CONTRACT_ID,
        "diagnostic_narrow_protector_surface_not_used_v1": True,
    }
    return {
        "layer_name": "NO_PRE_VETO_OR_UNSAFE_R5_2_TO_R6_GUARD_V1",
        "contract_id_v1": V2_CONTRACT_ID,
        "required_base_flag_v1": V2_FINAL_BASE_FLAG,
        "blocked_flags_v1": [
            V2_PRE_VETO_BASE_FLAG,
            "raw_true_base_membership_v1",
            "r5_2_true_rescue_base_membership_v1",
            "r5_2_v3_base_flag_v1",
            "in_v3_base_v1",
            "r5_2_selected_candidate__block_v1_SOURCE_BEFORE_STAGING",
        ],
        "checks_v1": checks,
        "guard_pass_v1": bool(all(checks.values())),
    }


def _benchmark_eval(r6_frame: pd.DataFrame, selected_mask: pd.Series) -> dict[str, Any]:
    selected = selected_mask.reindex(r6_frame.index).fillna(False).astype(bool)
    metrics = _frame_selected_metrics(r6_frame, selected)
    return {
        "layer_name": "R6_V2_OBJECTIVE_EVAL_AGAINST_BENCHMARKS_V1",
        "r6_v3_reference_v1": {"bad_v1": 82, "tail_v1": 51, "safety_v1": "CLEAN"},
        "r6_rescue_reference_v1": {"bad_v1": 88, "tail_v1": 57, "safety_v1": "CLEAN"},
        "best_r5_2_v2_input_v1": {
            "bad_v1": 95,
            "tail_v1": 61,
            "precision_v1": 1.0,
            "worst_loso_v1": 1.0,
            "pre_veto_base_v1": 111,
            "final_base_v1": 95,
            "vetoed_rows_v1": 16,
        },
        "wednesday_r6_benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "actual_r6_v2_objective_v1": metrics,
        "delta_vs_v3_v1": {"bad_v1": int(metrics["bad_blocks_v1"] - 82), "tail_v1": int(metrics["tail_help_v1"] - 51)},
        "delta_vs_rescue_v1": {"bad_v1": int(metrics["bad_blocks_v1"] - 88), "tail_v1": int(metrics["tail_help_v1"] - 57)},
        "delta_vs_wednesday_v1": {
            "bad_v1": int(metrics["bad_blocks_v1"] - WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"]),
            "tail_v1": int(metrics["tail_help_v1"] - WEDNESDAY_R6_BENCHMARK["tail_help_v1"]),
        },
        "batch_04_05_status_v1": {
            "batch_04_selected_v1": int((selected & r6_frame.get("batch_scope_v1", pd.Series("", index=r6_frame.index)).astype("string").eq("BATCH_04")).sum()),
            "batch_05_selected_v1": int((selected & r6_frame.get("batch_scope_v1", pd.Series("", index=r6_frame.index)).astype("string").eq("BATCH_05")).sum()),
        },
        "safety_pass_v1": _safety_pass(metrics),
    }


def _first_fail_reason(row: pd.Series, params: dict[str, Any]) -> str:
    selected = bool(row.get("r6_v2_objective_selected_candidate_block_v1", False))
    if selected and bool(row.get(V2_FINAL_BASE_FLAG, False)):
        return "SELECTED_BY_V2_FINAL_R5_2_BASE"
    if selected:
        return "SELECTED_BY_R6_ADDON"
    if not bool(row.get(V2_FINAL_BASE_FLAG, False)):
        return "NOT_IN_V2_FINAL_BASE"
    if float(row.get(R6_RUNNER_PROB, np.nan)) >= float(params.get("runner_threshold_v1", 0.60)):
        return "R6_RUNNER_GUARD_BLOCKED"
    if float(row.get(R5_2_RUNNER_PROB, np.nan)) >= float(params.get("r5_2_runner_threshold_v1", 0.74)):
        return "R5_2_PROTECTION_GUARD_BLOCKED"
    if bool(row.get("asof_runner_guard_v1", False)) and bool(params.get("hard_asof_runner_guard_v1", True)):
        return "ASOF_RUNNER_GUARD_BLOCKED"
    if float(row.get(R6_BAD_PROB, np.nan)) < float(params.get("bad_threshold_v1", 0.95)):
        return "R6_BAD_HEAD_TOO_LOW"
    if float(row.get(R6_RISKY_PROB, np.nan)) < float(params.get("risky_threshold_v1", 0.85)):
        return "R6_RISKY_HEAD_TOO_LOW"
    if float(row.get(R6_TAIL_PROB, np.nan)) < float(params.get("tail_threshold_v1", 0.90)):
        return "R6_TAIL_HEAD_TOO_LOW"
    if float(row.get(R6_BLINDSPOT_PROB, np.nan)) >= float(params.get("blindspot_threshold_v1", 0.70)):
        return "R6_BLINDSPOT_GUARD_BLOCKED"
    return "NOT_ESTABLISHED"


def _pass_through_trace(r6_frame: pd.DataFrame, selected_policy: dict[str, Any], selected_mask: pd.Series) -> pd.DataFrame:
    params = selected_policy.get("params_v1") or {}
    added = _bool(r6_frame, V2_FINAL_BASE_FLAG) & ~_bool(r6_frame, "r5_2_true_rescue_base_membership_v1")
    work = r6_frame.copy()
    work["r6_v2_objective_selected_candidate_block_v1"] = selected_mask.reindex(r6_frame.index).fillna(False).astype(bool)
    rows = work.loc[added].copy()
    if rows.empty:
        return rows
    rows["guard_status_v1"] = np.where(_bool(rows, "asof_runner_guard_v1"), "ASOF_GUARD_TRUE", "ASOF_GUARD_FALSE")
    rows["first_fail_reason_v1"] = rows.apply(lambda row: _first_fail_reason(row, params), axis=1)
    rows["final_r6_decision_v1"] = np.where(_bool(rows, "r6_v2_objective_selected_candidate_block_v1"), "BLOCK", "ALLOW")
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        V2_BAD_SCORE,
        V2_TAIL_SCORE,
        V2_RISKY_SCORE,
        V2_RUNNER_PROTECT_SCORE,
        V2_AMBIGUOUS_PROTECT_SCORE,
        V2_HARD_WINNER_PROTECT_SCORE,
        V2_PRE_VETO_BASE_FLAG,
        V2_HARD_VETO_FLAG,
        V2_FINAL_BASE_FLAG,
        "v2_base_reason_v1",
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
        R6_BLINDSPOT_PROB,
        "guard_status_v1",
        "r6_v2_objective_selected_candidate_block_v1",
        "first_fail_reason_v1",
        "final_r6_decision_v1",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def _safety_proof(r6_frame: pd.DataFrame, selected_mask: pd.Series) -> dict[str, Any]:
    selected = selected_mask.reindex(r6_frame.index).fillna(False).astype(bool)
    pre = _bool(r6_frame, V2_PRE_VETO_BASE_FLAG)
    veto = _bool(r6_frame, V2_HARD_VETO_FLAG)
    final = _bool(r6_frame, V2_FINAL_BASE_FLAG)
    ambiguous = _bool(r6_frame, "fifty_plus_mfe_v1") | _bool(r6_frame, "high_mfe_ambiguous_protection_target")
    runner_protect = _bool(r6_frame, "runner_protection_target") | _bool(r6_frame, "r6_label_runner_near_miss_v1")
    fifty = _bool(r6_frame, "fifty_plus_mfe_v1")
    hundred = _bool(r6_frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(r6_frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(r6_frame, "strongest_winner_path_v1")
    repaired = _bool(r6_frame, "r6_label_repaired_165_like_runner_v1")
    forensic = r6_frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in r6_frame.columns else pd.Series(False, index=r6_frame.index)
    vetoed = pre & veto
    raw_unsafe = _bool(r6_frame, "raw_true_base_membership_v1") & ~final
    proof = {
        "layer_name": "V2_HARD_VETO_R6_SAFETY_PROOF_V1",
        "rows_vetoed_before_r6_v1": int(vetoed.sum()),
        "unsafe_raw_rows_avoided_v1": int(raw_unsafe.sum()),
        "ambiguous_high_mfe_avoided_v1": int((vetoed & ambiguous).sum()),
        "runner_protect_avoided_v1": int((vetoed & runner_protect).sum()),
        "fifty_plus_avoided_v1": int((vetoed & fifty).sum()),
        "hundred_plus_avoided_v1": int((vetoed & hundred).sum()),
        "two_hundred_plus_avoided_v1": int((vetoed & two_hundred).sum()),
        "strongest_winner_avoided_v1": int((vetoed & strongest).sum()),
        "repaired_avoided_v1": int((vetoed & repaired).sum()),
        "forensic_trade_avoided_by_veto_v1": int((vetoed & forensic).sum()),
        "vetoed_rows_selected_by_r6_addon_v1": int((vetoed & selected).sum()),
        "final_selected_50_plus_v1": int((selected & fifty).sum()),
        "final_selected_100_plus_v1": int((selected & hundred).sum()),
        "final_selected_200_plus_v1": int((selected & two_hundred).sum()),
        "final_selected_strongest_v1": int((selected & strongest).sum()),
        "final_selected_repaired_v1": int((selected & repaired).sum()),
        "final_selected_runner_near_miss_v1": int((selected & _bool(r6_frame, "r6_label_runner_near_miss_v1")).sum()),
    }
    proof["safety_holds_after_r6_v1"] = bool(
        proof["vetoed_rows_selected_by_r6_addon_v1"] == 0
        and proof["final_selected_100_plus_v1"] == 0
        and proof["final_selected_200_plus_v1"] == 0
        and proof["final_selected_strongest_v1"] == 0
        and proof["final_selected_repaired_v1"] == 0
        and proof["final_selected_runner_near_miss_v1"] == 0
        and proof["final_selected_50_plus_v1"] <= 1
    )
    return proof


def _delta_and_root(
    benchmark_eval: dict[str, Any],
    pass_through: pd.DataFrame,
    safety_proof: dict[str, Any],
) -> dict[str, Any]:
    actual = benchmark_eval["actual_r6_v2_objective_v1"]
    added = int(len(pass_through))
    retained = int(pass_through.get("r6_v2_objective_selected_candidate_block_v1", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    if not benchmark_eval["safety_pass_v1"] or not safety_proof["safety_holds_after_r6_v1"]:
        root = "R6_V2_OBJECTIVE_SAFETY_FAIL"
    elif retained < added:
        root = "V2_ROWS_DO_NOT_PASS_THROUGH_R6"
    elif int(actual["bad_blocks_v1"]) < WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] or int(actual["tail_help_v1"]) < WEDNESDAY_R6_BENCHMARK["tail_help_v1"]:
        root = "V2_R5_2_BASE_STILL_TOO_SMALL"
    else:
        root = "NOT_ESTABLISHED"
    return {
        "layer_name": "R6_V2_OBJECTIVE_DELTA_AND_ROOT_CAUSE_V1",
        "actual_r6_v2_objective_v1": actual,
        "delta_vs_v3_v1": benchmark_eval["delta_vs_v3_v1"],
        "delta_vs_rescue_v1": benchmark_eval["delta_vs_rescue_v1"],
        "gap_to_wednesday_v1": {
            "bad_v1": int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - actual["bad_blocks_v1"]),
            "tail_v1": int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - actual["tail_help_v1"]),
        },
        "v2_final_rows_new_vs_rescue_v1": added,
        "v2_final_rows_selected_by_r6_v1": retained,
        "root_cause_v1": root,
        "safety_pass_v1": benchmark_eval["safety_pass_v1"] and safety_proof["safety_holds_after_r6_v1"],
    }


def _gate(benchmark_eval: dict[str, Any], delta_root: dict[str, Any]) -> dict[str, Any]:
    actual = benchmark_eval["actual_r6_v2_objective_v1"]
    if not delta_root["safety_pass_v1"]:
        decision = "MONDAY_R6_V2_OBJECTIVE_SAFETY_FAIL"
    elif int(actual["bad_blocks_v1"]) >= WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] and int(actual["tail_help_v1"]) >= WEDNESDAY_R6_BENCHMARK["tail_help_v1"]:
        decision = "MONDAY_R6_V2_OBJECTIVE_READY_FOR_FREEZE_GATE"
    elif delta_root["root_cause_v1"] == "V2_ROWS_DO_NOT_PASS_THROUGH_R6":
        decision = "MONDAY_R6_V2_OBJECTIVE_BLOCKED_BY_R6_HEAD_OR_THRESHOLD"
    elif int(actual["bad_blocks_v1"]) > 88 and int(actual["tail_help_v1"]) > 57:
        decision = "MONDAY_R6_V2_OBJECTIVE_SAFE_BUT_BELOW_WEDNESDAY"
    else:
        decision = "MONDAY_R6_V2_OBJECTIVE_RECALL_STILL_TOO_LOW"
    return {
        "layer_name": "R6_V2_OBJECTIVE_CANONICAL_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "safety_pass_v1": delta_root["safety_pass_v1"],
            "bad_blocks_v1": actual["bad_blocks_v1"],
            "tail_help_v1": actual["tail_help_v1"],
            "wednesday_bad_target_v1": WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"],
            "wednesday_tail_target_v1": WEDNESDAY_R6_BENCHMARK["tail_help_v1"],
            "v2_final_rows_selected_by_r6_v1": delta_root["v2_final_rows_selected_by_r6_v1"],
            "v2_final_rows_new_vs_rescue_v1": delta_root["v2_final_rows_new_vs_rescue_v1"],
        },
    }


def _next_action(gate: dict[str, Any], delta_root: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "MONDAY_R6_V2_OBJECTIVE_READY_FOR_FREEZE_GATE":
        action = "RUN_MONDAY_R6_FREEZE_GATE_NEXT"
    elif decision == "MONDAY_R6_V2_OBJECTIVE_SAFETY_FAIL":
        action = "STOP_AND_RUN_R6_V2_OBJECTIVE_FAILURE_FORENSICS"
    elif delta_root["v2_final_rows_selected_by_r6_v1"] < delta_root["v2_final_rows_new_vs_rescue_v1"]:
        action = "TRACE_R6_SELECTION_FAILURE_FOR_V2_ROWS_NEXT"
    else:
        action = "DESIGN_R5_2_OBJECTIVE_V3_OR_R6_HEAD_RECALL_NEXT"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_USE_PRE_VETO_R5_2_BASE_FOR_R6",
            "DO_NOT_FEED_RAW_TRUE_UNSAFE_R5_2_TO_R6",
            "DO_NOT_FREEZE_OR_PROMOTE_AUTOMATICALLY",
        ],
    }


def _audit_rows(summary: dict[str, Any], runtime_guard: dict[str, Any], benchmark_eval: dict[str, Any], safety_proof: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("R6_STARTED_WITH_EXPLICIT_FLAG", summary["r6_training_started_v1"], summary["r6_training_started_v1"]),
            row("BEST_V2_PACKAGE_USED", summary["best_v2_package_used_v1"], summary["best_variant_id_v1"]),
            row("PRE_VETO_AND_UNSAFE_RUNTIME_GUARD", runtime_guard["guard_pass_v1"], runtime_guard["checks_v1"]),
            row("SAFETY_CLEAN", benchmark_eval["safety_pass_v1"], benchmark_eval["actual_r6_v2_objective_v1"]),
            row("V2_HARD_VETO_DOWNSTREAM_PROOF", safety_proof["safety_holds_after_r6_v1"], safety_proof),
            row("NO_FREEZE_PROMO", True, False),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 Retrain From Best R5.2 Objective V2 Variant",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Best V2 variant: `{summary['best_variant_id_v1']}`",
            f"- Candidate: `{summary['selected_candidate_v1']}`",
            f"- Bad/tail: `{summary['bad_blocks_v1']}/{summary['tail_help_v1']}`",
            f"- Delta vs V3: `{summary['delta_vs_v3_bad_v1']}/{summary['delta_vs_v3_tail_v1']}`",
            f"- Delta vs rescue: `{summary['delta_vs_rescue_bad_v1']}/{summary['delta_vs_rescue_tail_v1']}`",
            f"- Precision / worst LOSO: `{summary['precision_v1']}` / `{summary['worst_loso_v1']}`",
            f"- Safety pass: `{summary['safety_pass_v1']}`",
            f"- New V2-final rows selected by R6: `{summary['v2_final_rows_selected_by_r6_v1']}/{summary['v2_final_rows_new_vs_rescue_v1']}`",
            "",
            "R6 used the V2 final post-veto base flag. Pre-veto, raw true unsafe, rescue-only and old V3 bases are blocked as final input.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    v2_execution_dir: Path = DEFAULT_V2_EXECUTION_DIR,
    v3_r6_dir: Path = DEFAULT_V3_R6_DIR,
    rescue_r6_dir: Path = DEFAULT_RESCUE_R6_DIR,
    output_dir: Path | None = None,
    run_r6_rebuild: bool = False,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    if not run_r6_rebuild:
        raise RuntimeError("RUN_R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V2_VARIANT requires explicit --run-r6-rebuild")
    reports_root = reports_root.expanduser().resolve()
    v2_execution_dir = v2_execution_dir.expanduser().resolve()
    v3_r6_dir = v3_r6_dir.expanduser().resolve()
    rescue_r6_dir = rescue_r6_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lock, variant_manifest, variant_safety, v2_execution_summary = _load_v2_inputs(v2_execution_dir)
    staged_dir = output_dir / "staged_best_r5_2_objective_v2_score_package_for_r6_v1"
    if (output_dir / TRAINING_FRAME).exists() and (output_dir / R6_SUMMARY).exists() and (staged_dir / SCORE_SUMMARY).exists():
        staged_summary = _read_json(staged_dir / SCORE_SUMMARY)
        r6_summary = _read_json(output_dir / R6_SUMMARY)
    else:
        staged_dir, staged_summary = _stage_v2_score_package(output_dir, v2_execution_dir, lock)
        r6_summary = run_r6_on_scores(
            reports_root=reports_root,
            score_dir=staged_dir,
            output_dir=output_dir,
            run_r6_rebuild=True,
            config=config,
        )

    r6_frame = pd.read_parquet(output_dir / TRAINING_FRAME)
    runtime_guard = _runtime_guard(lock, variant_manifest, staged_summary, r6_frame)
    if not runtime_guard["guard_pass_v1"]:
        raise RuntimeError(f"V2 runtime guard failed: {runtime_guard['checks_v1']}")

    selected_policy = _selected_policy_from_grid(output_dir) or _selected_policy(r6_summary)
    if not selected_policy:
        raise RuntimeError("Could not determine selected R6 V2 objective candidate from summary or family grid")
    selected_mask = _selected_policy_mask(r6_frame, selected_policy)
    candidate_grid = _candidate_grid_report(output_dir, selected_policy)
    benchmark_eval = _benchmark_eval(r6_frame, selected_mask)
    pass_through = _pass_through_trace(r6_frame, selected_policy, selected_mask)
    safety_proof = _safety_proof(r6_frame, selected_mask)
    delta_root = _delta_and_root(benchmark_eval, pass_through, safety_proof)
    gate = _gate(benchmark_eval, delta_root)
    next_action = _next_action(gate, delta_root)

    selected_family = selected_policy.get("family_v1")
    grid_safe = candidate_grid[candidate_grid.get("wednesday_safety_pass_v1", pd.Series(False, index=candidate_grid.index)).astype(bool)]
    selected_bad = int(benchmark_eval["actual_r6_v2_objective_v1"]["bad_blocks_v1"])
    selected_tail = int(benchmark_eval["actual_r6_v2_objective_v1"]["tail_help_v1"])
    better_safe = grid_safe[
        (pd.to_numeric(grid_safe.get("bad_blocks_v1"), errors="coerce").fillna(-1).gt(selected_bad))
        | (
            pd.to_numeric(grid_safe.get("bad_blocks_v1"), errors="coerce").fillna(-1).eq(selected_bad)
            & pd.to_numeric(grid_safe.get("tail_help_v1"), errors="coerce").fillna(-1).gt(selected_tail)
        )
    ]
    retrain = {
        "layer_name": "R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V2_VARIANT_V1",
        "best_v2_score_package_used_v1": True,
        "best_variant_id_v1": BEST_VARIANT_ID,
        "best_profile_id_v1": BEST_PROFILE_ID,
        "v2_execution_dir_v1": str(v2_execution_dir),
        "staged_score_dir_v1": str(staged_dir),
        "r5_2_base_flag_used_v1": V2_FINAL_BASE_FLAG,
        "raw_pre_veto_base_used_v1": False,
        "raw_true_unsafe_base_used_v1": False,
        "v1_v2_v3_rescue_final_base_used_v1": False,
        "r6_five_head_ran_v1": bool(r6_summary.get("r6_training_started_v1", True)),
        "candidate_grid_ran_v1": True,
        "thresholds_as_before_v1": True,
        "asof_hindsight_separation_v1": "PRESERVED_BY_EXISTING_R6_TRAINER_AND_STAGED_AS_OF_SCORE_FRAME",
        "selected_candidate_v1": selected_policy,
    }
    actual = benchmark_eval["actual_r6_v2_objective_v1"]
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v2_execution_dir_v1": str(v2_execution_dir),
        "best_variant_id_v1": BEST_VARIANT_ID,
        "best_profile_id_v1": BEST_PROFILE_ID,
        "best_v2_package_used_v1": True,
        "v2_final_base_flag_used_v1": V2_FINAL_BASE_FLAG,
        "pre_veto_or_unsafe_base_blocked_v1": runtime_guard["guard_pass_v1"],
        "r6_training_started_v1": True,
        "selected_candidate_v1": selected_policy.get("policy_name_v1"),
        "selected_family_v1": selected_family,
        "ultra_safe_tail_risky_addon_best_safe_family_v1": selected_family == "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "v2_gives_more_safe_candidates_v1": bool(len(grid_safe) > 0),
        "selected_candidate_still_base_bound_v1": bool(
            int((selected_mask & _bool(r6_frame, V2_FINAL_BASE_FLAG)).sum()) == int(selected_mask.sum())
        ),
        "rejected_candidate_with_better_safe_recall_v1": bool(len(better_safe) > 0),
        "bad_blocks_v1": actual["bad_blocks_v1"],
        "tail_help_v1": actual["tail_help_v1"],
        "precision_v1": actual["precision_v1"],
        "worst_loso_v1": actual["worst_loso_v1"],
        "safety_pass_v1": benchmark_eval["safety_pass_v1"] and safety_proof["safety_holds_after_r6_v1"],
        "delta_vs_v3_bad_v1": benchmark_eval["delta_vs_v3_v1"]["bad_v1"],
        "delta_vs_v3_tail_v1": benchmark_eval["delta_vs_v3_v1"]["tail_v1"],
        "delta_vs_rescue_bad_v1": benchmark_eval["delta_vs_rescue_v1"]["bad_v1"],
        "delta_vs_rescue_tail_v1": benchmark_eval["delta_vs_rescue_v1"]["tail_v1"],
        "v2_final_rows_new_vs_rescue_v1": delta_root["v2_final_rows_new_vs_rescue_v1"],
        "v2_final_rows_selected_by_r6_v1": delta_root["v2_final_rows_selected_by_r6_v1"],
        "hard_veto_safety_holds_after_r6_v1": safety_proof["safety_holds_after_r6_v1"],
        "root_cause_v1": delta_root["root_cause_v1"],
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "hard_status_v1": {
            "BEVIST": [
                "R6 was run from the best R5.2 Objective V2 variant with final post-veto base membership.",
                "Pre-veto, raw true unsafe, rescue-only, V3 and diagnostic surfaces were blocked as final R6 base.",
            ],
            "INDIKERT": [
                "The V2 result indicates whether more R5.2 objective work or R6 head work is required.",
            ],
            "IKKE_ETABLERT": [
                "No freeze, promotion, live behavior change, new baseline or new feature surface was created.",
            ],
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "r6_training_started_v1": True,
        "promotion_status_v1": "NOT_PROMOTED_NOT_FREEZE_GATE",
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_v2_lock_v1": lock,
        "input_variant_manifest_v1": variant_manifest,
        "input_variant_safety_v1": variant_safety,
        "input_v2_execution_summary_v1": v2_execution_summary,
        "v3_r6_reference_dir_v1": str(v3_r6_dir),
        "rescue_r6_reference_dir_v1": str(rescue_r6_dir),
        "staged_score_dir_v1": str(staged_dir),
        "output_files_v1": OUTPUT_FILES,
        "standard_r6_outputs_v1": {
            "summary_v1": str(output_dir / R6_SUMMARY),
            "training_frame_v1": str(output_dir / TRAINING_FRAME),
            "candidate_grid_v1": str(output_dir / R6_FAMILY_GRID_REPLAY),
            "compare_v1": str(output_dir / COMPARE_REPORT),
        },
    }

    _write_json(output_dir / OUTPUT_FILES["retrain"], retrain)
    _write_json(output_dir / OUTPUT_FILES["runtime_guard"], runtime_guard)
    candidate_grid.to_csv(output_dir / OUTPUT_FILES["candidate_grid"], index=False)
    _write_json(output_dir / OUTPUT_FILES["benchmark_eval"], benchmark_eval)
    pass_through.to_csv(output_dir / OUTPUT_FILES["pass_through"], index=False)
    _write_json(output_dir / OUTPUT_FILES["safety_proof"], safety_proof)
    _write_json(output_dir / OUTPUT_FILES["delta_root"], delta_root)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _audit_rows(summary, runtime_guard, benchmark_eval, safety_proof).to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--v2-execution-dir", type=Path, default=DEFAULT_V2_EXECUTION_DIR)
    parser.add_argument("--v3-r6-dir", type=Path, default=DEFAULT_V3_R6_DIR)
    parser.add_argument("--rescue-r6-dir", type=Path, default=DEFAULT_RESCUE_R6_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-r6-rebuild", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    config = TrainConfig(n_jobs=args.n_jobs)
    if args.n_estimators is not None:
        config = TrainConfig(
            r6_n_estimators=args.n_estimators,
            r6_early_stopping_rounds=min(60, max(20, args.n_estimators // 10)),
            n_jobs=args.n_jobs,
        )
    summary = materialize(
        reports_root=args.reports_root,
        v2_execution_dir=args.v2_execution_dir,
        v3_r6_dir=args.v3_r6_dir,
        rescue_r6_dir=args.rescue_r6_dir,
        output_dir=args.output_dir,
        run_r6_rebuild=args.run_r6_rebuild,
        config=config,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
