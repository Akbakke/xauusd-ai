#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_safe_true_r5_2_rescue_base_rule_v1 import (
    CONTRACT_ID,
    RESCUE_BASE_FLAG_COL,
)
from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import (
    BASE_FLAG_COL as RAW_TRUE_BASE_FLAG_COL,
    BAD_SCORE_COL as TRUE_R5_2_BAD_SCORE_COL,
    RISKY_SCORE_COL as TRUE_R5_2_RISKY_SCORE_COL,
    RUNNER_SCORE_COL as TRUE_R5_2_RUNNER_SCORE_COL,
    TAIL_SCORE_COL as TRUE_R5_2_TAIL_SCORE_COL,
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
    _asof_runner_guard,
    _bool,
    _jsonable,
    _num,
)
from gx1.scripts.train_monday_r6_on_foundation_scores_v1 import (
    COMPARE_REPORT,
    R6_FAMILY_GRID_REPLAY,
    SCORE_FRAME,
    SCORE_SUMMARY,
    SUMMARY as R6_SUMMARY,
    TRAINING_FRAME,
    TrainConfig,
    materialize as run_r6_on_scores,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_RESCUE_DIR = DEFAULT_REPORTS_ROOT / "SAFE_TRUE_R5_2_RESCUE_BASE_RULE_V1_20260426T_LOCK"
DEFAULT_V3_R6_DIR = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"
LAYER_NAME = "RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_V1"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

OUTPUT_FILES = {
    "retrain": "r6_retrain_from_true_r5_2_rescue_package_v1.json",
    "runtime_guard": "no_raw_true_r5_2_to_r6_runtime_guard_v1.json",
    "candidate_grid": "r6_rescue_candidate_grid_selection_v1.csv",
    "benchmark_eval": "r6_rescue_eval_against_benchmarks_v1.json",
    "pass_through": "rescued_rows_r6_pass_through_trace_v1.csv",
    "delta": "r6_rescue_delta_vs_v3_and_raw_true_v1.json",
    "forensics": "r6_rescue_failure_or_success_forensics_v1.json",
    "gate": "r6_rescue_canonical_gate_v1.json",
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


def _safe_div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def _worst_loso(frame: pd.DataFrame, selected: pd.Series) -> float | None:
    selected = selected.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    values: list[float] = []
    for _, group in frame.assign(_selected=selected, _should=should).groupby(frame["run_id"].astype("string"), dropna=False):
        block_count = int(group["_selected"].sum())
        if block_count:
            values.append(float((group["_selected"] & group["_should"]).sum() / block_count))
    return min(values) if values else None


def _frame_selected_metrics(frame: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    selected = selected.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    fifty = _bool(frame, "fifty_plus_mfe_v1")
    hundred = _bool(frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(frame, "strongest_winner_path_v1")
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1")
    runner_near = _bool(frame, "r6_label_runner_near_miss_v1")
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index)
    block = int(selected.sum())
    bad = int((selected & should).sum())
    return {
        "block_count_v1": block,
        "bad_blocks_v1": bad,
        "tail_help_v1": int((selected & tail).sum()),
        "precision_v1": _safe_div(bad, block),
        "worst_loso_v1": _worst_loso(frame, selected),
        "false_take_ok_blocks_v1": int((selected & take_ok).sum()),
        "repaired_damage_v1": int((selected & repaired).sum()),
        "forensic_trade_blocked_v1": int((selected & forensic).sum()),
        "forensic_trade_status_v1": "UNBLOCKED" if int((selected & forensic).sum()) == 0 else "BLOCKED",
        "fifty_plus_mfe_blocked_v1": int((selected & fifty).sum()),
        "hundred_plus_mfe_blocked_v1": int((selected & hundred).sum()),
        "two_hundred_plus_mfe_blocked_v1": int((selected & two_hundred).sum()),
        "strongest_winner_damage_v1": int((selected & strongest).sum()),
        "runner_near_miss_blocked_v1": int((selected & runner_near).sum()),
    }


def _safety_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["repaired_damage_v1"] == 0
        and metrics["forensic_trade_blocked_v1"] == 0
        and metrics["fifty_plus_mfe_blocked_v1"] <= 1
        and metrics["hundred_plus_mfe_blocked_v1"] == 0
        and metrics["two_hundred_plus_mfe_blocked_v1"] == 0
        and metrics["strongest_winner_damage_v1"] == 0
        and metrics["runner_near_miss_blocked_v1"] == 0
        and (metrics["precision_v1"] or 0.0) >= WEDNESDAY_R6_BENCHMARK["precision_v1"]
        and (metrics["worst_loso_v1"] or 0.0) >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]
    )


def _load_rescue_inputs(rescue_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(rescue_dir / "manifest_v1.json")
    r6_manifest = _read_json(rescue_dir / "true_r5_2_rescue_downstream_r6_input_manifest_v1.json")
    audit = _read_json(rescue_dir / "rescue_rule_application_audit_v1.json")
    if r6_manifest.get("contract_id_v1") != CONTRACT_ID:
        raise RuntimeError(f"Rescue contract mismatch: {r6_manifest.get('contract_id_v1')}")
    if r6_manifest.get("base_flag_for_r6_v1") != RESCUE_BASE_FLAG_COL:
        raise RuntimeError("Rescue manifest does not point R6 at the rescue base flag")
    if not audit.get("safety_pass_v1"):
        raise RuntimeError("Refuses unsafe rescue package")
    return manifest, r6_manifest, audit


def _refresh_r6_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    selected = _bool(out, "r5_2_selected_candidate__block_v1")
    take_ok = _bool(out, "take_was_ok_v1")
    fifty = _bool(out, "fifty_plus_mfe_v1")
    hundred = _bool(out, "hundred_plus_mfe_v1")
    two_hundred = _bool(out, "two_hundred_plus_mfe_v1")
    strong = _bool(out, "label_strong_trade_candidate_v1")
    should = _bool(out, "label_should_not_take_v1")
    mae = _num(out, "mae_abs_bps_v1")
    pnl = _num(out, "baseline_realized_pnl_bps_v1")
    mfe = _num(out, "peak_mfe_bps_v1")
    giveback = _num(out, "giveback_bps_v1")

    out["r6_label_runner_50_mfe_v1"] = take_ok & fifty
    out["r6_label_runner_100_mfe_v1"] = take_ok & hundred
    out["r6_label_runner_200_mfe_v1"] = take_ok & two_hundred
    out["r6_label_strong_low_mae_runner_v1"] = take_ok & strong & mae.le(25.0)
    out["r6_label_high_mfe_low_giveback_v1"] = take_ok & fifty & (giveback.le(25.0) | giveback.le(mfe * 0.25))
    out["r6_label_runner_near_miss_v1"] = take_ok & fifty & (_num(out, R5_2_BAD_PROB).ge(0.50) | _num(out, R5_2_RUNNER_PROB).lt(0.60) | selected)
    out["r6_label_runner_protect_v1"] = (
        _bool(out, "r6_label_runner_50_mfe_v1")
        | _bool(out, "r6_label_runner_100_mfe_v1")
        | _bool(out, "r6_label_runner_200_mfe_v1")
        | _bool(out, "r6_label_strong_low_mae_runner_v1")
        | _bool(out, "r6_label_high_mfe_low_giveback_v1")
        | _bool(out, "r6_label_runner_near_miss_v1")
        | _bool(out, "r6_label_repaired_165_like_runner_v1")
    )
    out["r6_label_missed_should_not_take_v1"] = should & ~selected
    out["r6_label_risky_allow_v1"] = _bool(out, "r6_label_missed_should_not_take_v1") & (
        mae.ge(40.0) | pnl.le(-25.0) | _num(out, R5_2_BAD_PROB).ge(0.60)
    )
    out["r6_label_high_mae_low_mfe_v1"] = should & mae.ge(40.0) & mfe.lt(50.0)
    out["r6_label_low_mfe_low_value_v1"] = should & mfe.lt(10.0) & pnl.le(0.0)
    out["r6_label_early_adverse_excursion_v1"] = _bool(out, "r6_label_high_mae_low_mfe_v1")
    out["r6_label_bad_trade_overlap_extreme_vol_v1"] = (
        should
        & out.get("as_of_session_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("OVERLAP")
        & out.get("as_of_candidate_vol_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("EXTREME")
    )
    if "batch_scope_v1" not in out.columns:
        out["batch_scope_v1"] = "NOT_ESTABLISHED"
    out["r6_label_batch04_blindspot_v1"] = _bool(out, "r6_label_missed_should_not_take_v1") & out["batch_scope_v1"].astype("string").eq("BATCH_04")
    out["r6_label_trend_neutral_extreme_vol_risk_v1"] = (
        should
        & out.get("as_of_candidate_trend_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("TREND_NEUTRAL")
        & out.get("as_of_candidate_vol_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("EXTREME")
    )
    out["r6_label_bad_risk_v1"] = should
    out["r6_label_tail_control_10_50_v1"] = _bool(out, "tail_10_50_mfe_v1")
    return out


def _stage_rescue_score_package(output_dir: Path, rescue_dir: Path, manifest: dict[str, Any], r6_manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_score_path = Path(manifest["input_paths_v1"]["score_path_v1"])
    source_score_dir = source_score_path.parent
    source_summary = _read_json(source_score_dir / SCORE_SUMMARY)
    foundation = pd.read_parquet(source_score_path)
    rescue_scores = pd.read_parquet(r6_manifest["score_package_path_v1"])
    required = [
        "candidate_uid",
        TRUE_R5_2_BAD_SCORE_COL,
        TRUE_R5_2_TAIL_SCORE_COL,
        TRUE_R5_2_RISKY_SCORE_COL,
        TRUE_R5_2_RUNNER_SCORE_COL,
        RESCUE_BASE_FLAG_COL,
        "in_v3_base_v1",
        "raw_true_base_membership_v1",
        "added_by_true_rescue_rule_v1",
    ]
    missing = [column for column in required if column not in rescue_scores.columns]
    if missing:
        raise RuntimeError(f"Rescue score package missing required columns: {missing}")
    if len(foundation) != len(rescue_scores):
        raise RuntimeError(f"Rescue/foundation row mismatch: {len(rescue_scores)} vs {len(foundation)}")
    merged = foundation.merge(rescue_scores[required], on="candidate_uid", how="left", validate="one_to_one")
    if int(merged[RESCUE_BASE_FLAG_COL].isna().sum()) != 0:
        raise RuntimeError("Rescue base flag missing after key alignment")

    merged["r5_2_v3_base_flag_before_rescue_v1"] = _bool(merged, "r5_2_selected_candidate__block_v1")
    merged["r5_2_raw_true_base_membership_v1"] = _bool(merged, "raw_true_base_membership_v1")
    merged[R5_2_BAD_PROB] = _num(merged, TRUE_R5_2_BAD_SCORE_COL)
    merged[R5_2_RUNNER_PROB] = _num(merged, TRUE_R5_2_RUNNER_SCORE_COL)
    merged["blocker_score_v1"] = _num(merged, TRUE_R5_2_BAD_SCORE_COL)
    merged["runner_protector_score_v1"] = _num(merged, TRUE_R5_2_RUNNER_SCORE_COL)
    merged["r5_2_selected_candidate__block_v1"] = _bool(merged, RESCUE_BASE_FLAG_COL)
    merged["r5_2_base_membership_contract_id_v1"] = CONTRACT_ID
    merged = _refresh_r6_labels(merged)

    staged_dir = output_dir / "staged_true_r5_2_rescue_score_package_for_r6_v1"
    staged_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(staged_dir / SCORE_FRAME, index=False)
    staged_summary = {
        **source_summary,
        "layer_name": "TRUE_R5_2_RESCUE_STAGED_R6_SCORE_PACKAGE_V1",
        "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
        "source_score_dir_v1": str(source_score_dir),
        "source_rescue_dir_v1": str(rescue_dir),
        "contract_id_v1": CONTRACT_ID,
        "base_flag_for_r6_v1": RESCUE_BASE_FLAG_COL,
        "raw_true_base_flag_blocked_v1": RAW_TRUE_BASE_FLAG_COL,
        "r5_2_rescue_base_rows_v1": int(_bool(merged, RESCUE_BASE_FLAG_COL).sum()),
        "r5_2_v3_base_rows_before_rescue_v1": int(_bool(merged, "in_v3_base_v1").sum()),
        "r5_2_raw_true_base_rows_v1": int(_bool(merged, "raw_true_base_membership_v1").sum()),
        "r6_heads_trained_v1": False,
    }
    _write_json(staged_dir / SCORE_SUMMARY, staged_summary)
    return staged_dir, staged_summary


def _selected_policy(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("family_grid_selected_policy_v1") or summary.get("custom_threshold_grid_policy_v1", {}).get("selected_candidate_v1") or {}


def _selected_policy_from_grid(output_dir: Path) -> dict[str, Any]:
    grid_path = output_dir / R6_FAMILY_GRID_REPLAY
    if not grid_path.exists():
        return {}
    grid = pd.read_csv(grid_path)
    if grid.empty or "wednesday_safety_pass_v1" not in grid.columns:
        return {}
    safe = grid[grid["wednesday_safety_pass_v1"].astype(bool)].copy()
    if safe.empty:
        return {}
    row = safe.sort_values(
        ["bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False],
        na_position="last",
    ).iloc[0].to_dict()
    params = {
        "bad_threshold_v1": float(row["bad_threshold_v1"]),
        "runner_threshold_v1": float(row["runner_threshold_v1"]),
        "tail_threshold_v1": float(row["tail_threshold_v1"]),
        "risky_threshold_v1": float(row["risky_threshold_v1"]),
        "blindspot_threshold_v1": float(row["blindspot_threshold_v1"]),
        "r5_2_runner_threshold_v1": float(row["r5_2_runner_threshold_v1"]),
        "use_r5_2_base_v1": bool(row["use_r5_2_base_v1"]),
        "hard_asof_runner_guard_v1": bool(row["hard_asof_runner_guard_v1"]),
    }
    metrics = {
        "row_count_v1": int(row["row_count_v1"]),
        "block_count_v1": int(row["block_count_v1"]),
        "bad_blocks_v1": int(row["bad_blocks_v1"]),
        "tail_help_v1": int(row["tail_help_v1"]),
        "precision_v1": None if pd.isna(row["precision_v1"]) else float(row["precision_v1"]),
        "false_take_ok_blocks_v1": int(row["false_take_ok_blocks_v1"]),
        "fifty_plus_mfe_blocked_v1": int(row["fifty_plus_mfe_blocked_v1"]),
        "hundred_plus_mfe_blocked_v1": int(row["hundred_plus_mfe_blocked_v1"]),
        "two_hundred_plus_mfe_blocked_v1": int(row["two_hundred_plus_mfe_blocked_v1"]),
        "strongest_winner_damage_v1": int(row["strongest_winner_damage_v1"]),
        "repaired_165_damage_v1": int(row["repaired_165_damage_v1"]),
        "quarantine_blocks_v1": int(row["quarantine_blocks_v1"]),
        "runner_near_miss_blocked_v1": int(row["runner_near_miss_blocked_v1"]),
    }
    return {
        "policy_name_v1": str(row["policy_name_v1"]),
        "policy_source_v1": "R6_FAMILY_GRID_SAFE_CANDIDATE",
        "family_v1": str(row["family_v1"]),
        "params_v1": params,
        "metrics_v1": metrics,
        "candidate_worst_loso_v1": None if pd.isna(row["worst_loso_v1"]) else float(row["worst_loso_v1"]),
    }


def _summary_selected_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_policy(summary)
    return selected.get("metrics_v1") or selected.get("compare_v1", {}).get("candidate_metrics_v1") or {}


def _selected_params(summary: dict[str, Any]) -> dict[str, Any]:
    return _selected_policy(summary).get("params_v1") or {}


def _first_fail_reason(row: pd.Series, params: dict[str, Any]) -> str:
    selected = bool(row.get("r6_rescue_selected_candidate_block_v1", row.get("selected_candidate_block_v1", False)))
    if selected and bool(row.get(RESCUE_BASE_FLAG_COL, False)):
        return "SELECTED_BY_TRUE_R5_2_RESCUE_BASE"
    if selected:
        return "SELECTED_BY_R6_ADDON"
    if float(row.get(R6_RUNNER_PROB, np.nan)) >= float(params.get("runner_threshold_v1", params.get("r6_runner_threshold_v1", 0.60))):
        return "R6_RUNNER_GUARD_BLOCKED"
    if float(row.get(R5_2_RUNNER_PROB, np.nan)) >= float(params.get("r5_2_runner_threshold_v1", params.get("r5_2_protect_threshold_v1", 0.74))):
        return "R5_2_RUNNER_GUARD_BLOCKED"
    if bool(row.get("asof_runner_guard_v1", False)) and bool(params.get("hard_asof_runner_guard_v1", True)):
        return "ASOF_RUNNER_GUARD_BLOCKED"
    if float(row.get(R6_BAD_PROB, np.nan)) < float(params.get("bad_threshold_v1", params.get("r6_bad_threshold_v1", 0.95))):
        return "R6_BAD_HEAD_TOO_LOW"
    if float(row.get(R6_RISKY_PROB, np.nan)) < float(params.get("risky_threshold_v1", params.get("r6_risky_threshold_v1", 0.85))):
        return "R6_RISKY_HEAD_TOO_LOW"
    if float(row.get(R6_TAIL_PROB, np.nan)) < float(params.get("tail_threshold_v1", params.get("r6_tail_threshold_v1", 0.90))):
        return "R6_TAIL_HEAD_TOO_LOW"
    if float(row.get(R6_BLINDSPOT_PROB, np.nan)) >= float(params.get("blindspot_threshold_v1", 0.70)):
        return "R6_BLINDSPOT_GUARD_BLOCKED"
    return "NOT_ESTABLISHED"


def _selected_policy_mask(frame: pd.DataFrame, selected_policy: dict[str, Any]) -> pd.Series:
    params = selected_policy.get("params_v1") or {}
    if "bad_threshold_v1" not in params:
        return _bool(frame, "selected_candidate_block_v1")
    r5_2_base = _bool(frame, "r5_2_selected_candidate__block_v1") if bool(params.get("use_r5_2_base_v1", True)) else pd.Series(False, index=frame.index)
    protect = (
        _num(frame, R6_RUNNER_PROB).ge(float(params["runner_threshold_v1"])).fillna(False)
        | _num(frame, R5_2_RUNNER_PROB).ge(float(params["r5_2_runner_threshold_v1"])).fillna(False)
    )
    if bool(params.get("hard_asof_runner_guard_v1", True)):
        protect = protect | _asof_runner_guard(frame, params)
    blind_ok = _num(frame, R6_BLINDSPOT_PROB).lt(float(params["blindspot_threshold_v1"])).fillna(True)
    addon = (
        _num(frame, R6_BAD_PROB).ge(float(params["bad_threshold_v1"])).fillna(False)
        & _num(frame, R6_RISKY_PROB).ge(float(params["risky_threshold_v1"])).fillna(False)
        & _num(frame, R6_TAIL_PROB).ge(float(params["tail_threshold_v1"])).fillna(False)
        & blind_ok
        & ~protect
    )
    return (r5_2_base | addon).fillna(False).astype(bool)


def _pass_through_trace(frame: pd.DataFrame, params: dict[str, Any], selected_mask: pd.Series) -> pd.DataFrame:
    added = _bool(frame, "added_by_true_rescue_rule_v1")
    work = frame.copy()
    work["r6_rescue_selected_candidate_block_v1"] = selected_mask.reindex(frame.index).fillna(False).astype(bool)
    rows = work.loc[added].copy()
    if rows.empty:
        return rows
    rows["rescue_base_reason_v1"] = "ADDED_BY_TRUE_R5_2_RESCUE_RULE"
    rows["guard_status_v1"] = np.where(_bool(rows, "asof_runner_guard_v1"), "ASOF_GUARD_TRUE", "ASOF_GUARD_FALSE")
    rows["first_fail_reason_v1"] = rows.apply(lambda row: _first_fail_reason(row, params), axis=1)
    rows["final_r6_decision_v1"] = np.where(_bool(rows, "r6_rescue_selected_candidate_block_v1"), "BLOCK", "ALLOW")
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        TRUE_R5_2_BAD_SCORE_COL,
        TRUE_R5_2_TAIL_SCORE_COL,
        TRUE_R5_2_RISKY_SCORE_COL,
        TRUE_R5_2_RUNNER_SCORE_COL,
        RESCUE_BASE_FLAG_COL,
        "rescue_base_reason_v1",
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
        R6_BLINDSPOT_PROB,
        "guard_status_v1",
        "r6_rescue_selected_candidate_block_v1",
        "first_fail_reason_v1",
        "final_r6_decision_v1",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def _candidate_grid_report(output_dir: Path, selected_policy: dict[str, Any]) -> pd.DataFrame:
    grid = pd.read_csv(output_dir / R6_FAMILY_GRID_REPLAY)
    selected_name = str(selected_policy.get("policy_name_v1") or "")
    grid["selected_candidate_v1"] = grid["policy_name_v1"].astype(str).eq(selected_name)
    grid["rejection_reason_v1"] = np.select(
        [
            grid["selected_candidate_v1"],
            ~grid.get("wednesday_safety_pass_v1", pd.Series(False, index=grid.index)).astype(bool),
            grid.get("wednesday_safety_pass_v1", pd.Series(False, index=grid.index)).astype(bool),
        ],
        ["SELECTED", "REJECTED_SAFETY", "REJECTED_LOWER_SAFE_RECALL_OR_SELECTION_SCORE"],
        default="NOT_ESTABLISHED",
    )
    return grid


def _runtime_guard(r6_manifest: dict[str, Any], staged_summary: dict[str, Any], r6_frame: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "rescued_package_used_v1": r6_manifest.get("contract_id_v1") == CONTRACT_ID,
        "rescue_base_flag_present_in_runtime_frame_v1": RESCUE_BASE_FLAG_COL in r6_frame.columns,
        "runtime_r5_2_selected_equals_rescue_flag_v1": bool(_bool(r6_frame, "r5_2_selected_candidate__block_v1").equals(_bool(r6_frame, RESCUE_BASE_FLAG_COL))),
        "raw_true_base_not_used_as_final_base_v1": not _bool(r6_frame, "r5_2_selected_candidate__block_v1").equals(_bool(r6_frame, "raw_true_base_membership_v1")),
        "v3_base_not_used_as_final_base_v1": not _bool(r6_frame, "r5_2_selected_candidate__block_v1").equals(_bool(r6_frame, "in_v3_base_v1")),
        "staged_summary_points_to_rescue_contract_v1": staged_summary.get("contract_id_v1") == CONTRACT_ID,
    }
    return {
        "layer_name": "NO_RAW_TRUE_R5_2_TO_R6_RUNTIME_GUARD_V1",
        "contract_id_v1": CONTRACT_ID,
        "checks_v1": checks,
        "guard_pass_v1": bool(all(checks.values())),
        "blocked_action_v1": [
            "DO_NOT_USE_RAW_TRUE_R5_2_BASE_FOR_R6",
            "DO_NOT_USE_V1_V2_V3_BASE_AS_FINAL_RESCUE_R6_BASE",
        ],
    }


def _benchmark_eval(r6_summary: dict[str, Any], r6_frame: pd.DataFrame, selected_mask: pd.Series) -> dict[str, Any]:
    selected = selected_mask.reindex(r6_frame.index).fillna(False).astype(bool)
    metrics = _frame_selected_metrics(r6_frame, selected)
    return {
        "layer_name": "R6_RESCUE_EVAL_AGAINST_BENCHMARKS_V1",
        "r6_v3_reference_v1": {"bad_v1": 82, "tail_v1": 51, "safety_v1": "CLEAN"},
        "r5_2_rescue_expected_v1": {"bad_v1": 88, "tail_v1": 57, "precision_v1": 1.0, "worst_loso_v1": 1.0},
        "wednesday_r6_benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "actual_r6_rescue_v1": metrics,
        "batch_04_05_status_v1": {
            "batch_04_selected_v1": int((selected & r6_frame.get("batch_scope_v1", pd.Series("", index=r6_frame.index)).astype("string").eq("BATCH_04")).sum()),
            "batch_05_selected_v1": int((selected & r6_frame.get("batch_scope_v1", pd.Series("", index=r6_frame.index)).astype("string").eq("BATCH_05")).sum()),
        },
        "safety_pass_v1": _safety_pass(metrics),
    }


def _delta_report(v3_dir: Path, r6_frame: pd.DataFrame, rescue_audit: dict[str, Any], benchmark_eval: dict[str, Any], selected_mask: pd.Series) -> dict[str, Any]:
    v3_summary = _read_json(v3_dir / R6_SUMMARY)
    v3_metrics = _summary_selected_metrics(v3_summary) or {"bad_blocks_v1": 82, "tail_help_v1": 51, "precision_v1": 1.0}
    actual = benchmark_eval["actual_r6_rescue_v1"]
    added = _bool(r6_frame, "added_by_true_rescue_rule_v1")
    selected = selected_mask.reindex(r6_frame.index).fillna(False).astype(bool)
    raw_unsafe = _bool(r6_frame, "raw_true_base_membership_v1") & ~_bool(r6_frame, RESCUE_BASE_FLAG_COL)
    return {
        "layer_name": "R6_RESCUE_DELTA_VS_V3_AND_RAW_TRUE_V1",
        "v3_r6_v1": v3_metrics,
        "rescued_r5_2_to_r6_v1": actual,
        "raw_true_hard_negative_reference_v1": rescue_audit.get("raw_true_v1"),
        "delta_rescue_minus_v3_v1": {
            "bad_v1": int(actual["bad_blocks_v1"] - int(v3_metrics.get("bad_blocks_v1") or 82)),
            "tail_v1": int(actual["tail_help_v1"] - int(v3_metrics.get("tail_help_v1") or 51)),
            "precision_v1": None if actual["precision_v1"] is None else float(actual["precision_v1"] - float(v3_metrics.get("precision_v1") or 0.0)),
            "safety_delta_v1": int(
                actual["repaired_damage_v1"]
                + max(0, actual["fifty_plus_mfe_blocked_v1"] - 0)
                + actual["hundred_plus_mfe_blocked_v1"]
                + actual["two_hundred_plus_mfe_blocked_v1"]
                + actual["strongest_winner_damage_v1"]
            ),
        },
        "rescued_rows_retained_by_r6_v1": int((added & selected).sum()),
        "rescued_rows_added_v1": int(added.sum()),
        "unsafe_raw_true_rows_avoided_v1": int(raw_unsafe.sum()),
        "inherits_raw_true_safety_fail_v1": not benchmark_eval["safety_pass_v1"],
    }


def _forensics(benchmark_eval: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    actual = benchmark_eval["actual_r6_rescue_v1"]
    safety = benchmark_eval["safety_pass_v1"]
    if not safety:
        root = "R6_RESCUE_SAFETY_FAIL"
    elif int(actual["bad_blocks_v1"]) <= 100 and int(actual["tail_help_v1"]) <= 70:
        root = "TRUE_R5_2_RESCUE_ONLY_TINY_UPLIFT"
    elif delta["rescued_rows_retained_by_r6_v1"] < delta["rescued_rows_added_v1"]:
        root = "R6_HEAD_SCORES_TOO_WEAK"
    elif int(actual["bad_blocks_v1"]) < WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"]:
        root = "R5_2_BASE_STILL_TOO_SMALL"
    else:
        root = "NOT_ESTABLISHED"
    return {
        "layer_name": "R6_RESCUE_FAILURE_OR_SUCCESS_FORENSICS_V1",
        "root_cause_v1": root,
        "safety_pass_v1": safety,
        "gap_to_wednesday_v1": {
            "bad_v1": int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - actual["bad_blocks_v1"]),
            "tail_v1": int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - actual["tail_help_v1"]),
        },
        "rescued_rows_pass_through_v1": {
            "added_v1": delta["rescued_rows_added_v1"],
            "retained_by_r6_v1": delta["rescued_rows_retained_by_r6_v1"],
        },
    }


def _gate(benchmark_eval: dict[str, Any], forensics: dict[str, Any]) -> dict[str, Any]:
    actual = benchmark_eval["actual_r6_rescue_v1"]
    if not benchmark_eval["safety_pass_v1"]:
        decision = "MONDAY_R6_RESCUE_SAFETY_FAIL"
    elif int(actual["bad_blocks_v1"]) >= WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] and int(actual["tail_help_v1"]) >= WEDNESDAY_R6_BENCHMARK["tail_help_v1"]:
        decision = "MONDAY_R6_RESCUE_CANDIDATE_READY_FOR_FREEZE_GATE"
    elif forensics["root_cause_v1"] == "TRUE_R5_2_RESCUE_ONLY_TINY_UPLIFT":
        decision = "MONDAY_R6_RESCUE_RECALL_STILL_TOO_LOW"
    else:
        decision = "MONDAY_R6_RESCUE_SAFE_BUT_BELOW_WEDNESDAY"
    return {
        "layer_name": "R6_RESCUE_CANONICAL_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "safety_pass_v1": benchmark_eval["safety_pass_v1"],
            "bad_blocks_v1": actual["bad_blocks_v1"],
            "tail_help_v1": actual["tail_help_v1"],
            "wednesday_bad_target_v1": WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"],
            "wednesday_tail_target_v1": WEDNESDAY_R6_BENCHMARK["tail_help_v1"],
        },
    }


def _next_action(gate: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "MONDAY_R6_RESCUE_CANDIDATE_READY_FOR_FREEZE_GATE":
        action = "RUN_MONDAY_R6_FREEZE_GATE_NEXT"
    elif decision == "MONDAY_R6_RESCUE_SAFETY_FAIL":
        action = "STOP_AND_RUN_R6_RESCUE_FAILURE_FORENSICS"
    elif delta["rescued_rows_retained_by_r6_v1"] < delta["rescued_rows_added_v1"]:
        action = "TRACE_R6_SELECTION_FAILURE_FOR_RESCUE_ROWS_NEXT"
    else:
        action = "INVESTIGATE_R5_2_OBJECTIVE_V2_OR_R6_HEAD_RECALL_NEXT"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_FEED_RAW_TRUE_R5_2_TO_R6",
            "DO_NOT_FREEZE_OR_PROMOTE_AUTOMATICALLY",
        ],
    }


def _audit_rows(summary: dict[str, Any], runtime_guard: dict[str, Any], benchmark_eval: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("R6_STARTED_WITH_EXPLICIT_FLAG", summary["r6_training_started_v1"], summary["r6_training_started_v1"]),
            row("RESCUE_PACKAGE_USED", summary["rescued_package_used_v1"], summary["rescue_dir_v1"]),
            row("RAW_TRUE_RUNTIME_GUARD", runtime_guard["guard_pass_v1"], runtime_guard["checks_v1"]),
            row("SAFETY_CLEAN", benchmark_eval["safety_pass_v1"], benchmark_eval["actual_r6_rescue_v1"]),
            row("NO_FREEZE_PROMO", True, False),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 Retrain From True R5.2 Rescue Package",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Candidate: `{summary['selected_candidate_v1']}`",
            f"- Bad/tail: `{summary['bad_blocks_v1']}/{summary['tail_help_v1']}`",
            f"- Precision / worst LOSO: `{summary['precision_v1']}` / `{summary['worst_loso_v1']}`",
            f"- Safety pass: `{summary['safety_pass_v1']}`",
            f"- Rescue rows retained by R6: `{summary['rescued_rows_retained_by_r6_v1']}/{summary['rescued_rows_added_v1']}`",
            "",
            "Raw true R5.2 base is blocked; R6 used the rescued base flag.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    rescue_dir: Path = DEFAULT_RESCUE_DIR,
    v3_r6_dir: Path = DEFAULT_V3_R6_DIR,
    output_dir: Path | None = None,
    run_r6_rebuild: bool = False,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    if not run_r6_rebuild:
        raise RuntimeError("RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE requires explicit --run-r6-rebuild")
    reports_root = reports_root.expanduser().resolve()
    rescue_dir = rescue_dir.expanduser().resolve()
    v3_r6_dir = v3_r6_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rescue_manifest, rescue_r6_manifest, rescue_audit = _load_rescue_inputs(rescue_dir)
    staged_dir = output_dir / "staged_true_r5_2_rescue_score_package_for_r6_v1"
    if (output_dir / TRAINING_FRAME).exists() and (output_dir / R6_SUMMARY).exists() and (staged_dir / SCORE_SUMMARY).exists():
        staged_summary = _read_json(staged_dir / SCORE_SUMMARY)
        r6_summary = _read_json(output_dir / R6_SUMMARY)
    else:
        staged_dir, staged_summary = _stage_rescue_score_package(output_dir, rescue_dir, rescue_manifest, rescue_r6_manifest)
        r6_summary = run_r6_on_scores(
            reports_root=reports_root,
            score_dir=staged_dir,
            output_dir=output_dir,
            run_r6_rebuild=True,
            config=config,
        )
    r6_frame = pd.read_parquet(output_dir / TRAINING_FRAME)
    runtime_guard = _runtime_guard(rescue_r6_manifest, staged_summary, r6_frame)
    if not runtime_guard["guard_pass_v1"]:
        raise RuntimeError(f"Raw true runtime guard failed: {runtime_guard['checks_v1']}")

    selected_policy = _selected_policy(r6_summary) or _selected_policy_from_grid(output_dir)
    if not selected_policy:
        raise RuntimeError("Could not determine selected R6 rescue candidate from summary or family grid")
    params = _selected_params(r6_summary)
    selected_mask = _selected_policy_mask(r6_frame, selected_policy)
    candidate_grid = _candidate_grid_report(output_dir, selected_policy)
    benchmark_eval = _benchmark_eval(r6_summary, r6_frame, selected_mask)
    pass_through = _pass_through_trace(r6_frame, params, selected_mask)
    delta = _delta_report(v3_r6_dir, r6_frame, rescue_audit, benchmark_eval, selected_mask)
    forensics = _forensics(benchmark_eval, delta)
    gate = _gate(benchmark_eval, forensics)
    next_action = _next_action(gate, delta)

    retrain = {
        "layer_name": "R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_V1",
        "rescued_package_used_v1": True,
        "contract_id_v1": CONTRACT_ID,
        "rescue_dir_v1": str(rescue_dir),
        "staged_score_dir_v1": str(staged_dir),
        "r5_2_base_flag_used_v1": RESCUE_BASE_FLAG_COL,
        "raw_true_base_used_v1": False,
        "v1_v2_v3_final_base_used_v1": False,
        "r6_five_head_ran_v1": bool(r6_summary.get("r6_training_started_v1")),
        "candidate_grid_ran_v1": True,
        "thresholds_as_before_v1": True,
        "asof_hindsight_separation_v1": "PRESERVED_BY_EXISTING_R6_TRAINER_AND_STAGED_AS_OF_SCORE_FRAME",
        "selected_candidate_v1": selected_policy,
    }
    actual = benchmark_eval["actual_r6_rescue_v1"]
    selected_family = selected_policy.get("family_v1")
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "rescue_dir_v1": str(rescue_dir),
        "contract_id_v1": CONTRACT_ID,
        "rescued_package_used_v1": True,
        "raw_true_base_blocked_v1": runtime_guard["guard_pass_v1"],
        "r6_training_started_v1": True,
        "selected_candidate_v1": selected_policy.get("policy_name_v1"),
        "selected_family_v1": selected_family,
        "ultra_safe_tail_risky_addon_best_safe_family_v1": selected_family == "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "bad_blocks_v1": actual["bad_blocks_v1"],
        "tail_help_v1": actual["tail_help_v1"],
        "precision_v1": actual["precision_v1"],
        "worst_loso_v1": actual["worst_loso_v1"],
        "safety_pass_v1": benchmark_eval["safety_pass_v1"],
        "rescued_rows_added_v1": delta["rescued_rows_added_v1"],
        "rescued_rows_retained_by_r6_v1": delta["rescued_rows_retained_by_r6_v1"],
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "hard_status_v1": {
            "BEVIST": [
                "R6 was run from the rescued true R5.2 package with the rescue base flag.",
                "Raw true R5.2 base was blocked at runtime.",
            ],
            "INDIKERT": [
                "The R6 result determines whether the rescued package is worth further objective/head work.",
            ],
            "IKKE_ETABLERT": [
                "No freeze or promotion was run.",
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
        "input_rescue_manifest_v1": rescue_r6_manifest,
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
    _write_json(output_dir / OUTPUT_FILES["delta"], delta)
    _write_json(output_dir / OUTPUT_FILES["forensics"], forensics)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _audit_rows(summary, runtime_guard, benchmark_eval).to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--rescue-dir", type=Path, default=DEFAULT_RESCUE_DIR)
    parser.add_argument("--v3-r6-dir", type=Path, default=DEFAULT_V3_R6_DIR)
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
        rescue_dir=args.rescue_dir,
        v3_r6_dir=args.v3_r6_dir,
        output_dir=args.output_dir,
        run_r6_rebuild=args.run_r6_rebuild,
        config=config,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
