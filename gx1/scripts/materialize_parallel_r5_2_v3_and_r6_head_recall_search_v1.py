#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r6_retrain_from_best_r5_2_objective_v2_variant_v1 import (
    BEST_VARIANT_ID,
    V2_AMBIGUOUS_PROTECT_SCORE,
    V2_BAD_SCORE,
    V2_FINAL_BASE_FLAG,
    V2_HARD_VETO_FLAG,
    V2_HARD_WINNER_PROTECT_SCORE,
    V2_PRE_VETO_BASE_FLAG,
    V2_RISKY_SCORE,
    V2_RUNNER_PROTECT_SCORE,
    V2_TAIL_SCORE,
)
from gx1.scripts.materialize_r6_retrain_from_true_r5_2_rescue_package_v1 import (
    _frame_selected_metrics,
    _safety_pass,
    _selected_policy_from_grid,
    _selected_policy_mask,
)
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
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
from gx1.scripts.train_monday_r6_on_foundation_scores_v1 import TRAINING_FRAME


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_R6_V2_DIR = (
    DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V2_VARIANT_V1_20260426T_EXPLICIT"
)
DEFAULT_V2_EXECUTION_DIR = (
    DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_20260426T_EXECUTION"
)
LAYER_NAME = "PARALLEL_R5_2_V3_AND_R6_HEAD_RECALL_SEARCH_V1"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

OUTPUT_FILES = {
    "orchestrator": "parallel_recall_search_orchestrator_v1.json",
    "lane_01": "lane_01_v2_remaining_gap_trace_v1.csv",
    "lane_02": "lane_02_v2_veto_false_negative_scan_v1.csv",
    "lane_03": "lane_03_bad_recall_head_strength_scan_v1.csv",
    "lane_04": "lane_04_tail_10_50_recall_scan_v1.csv",
    "lane_05": "lane_05_risky_attention_confirmation_scan_v1.csv",
    "lane_06": "lane_06_ambiguous_high_mfe_refinement_scan_v1.csv",
    "lane_07": "lane_07_r6_outside_base_micro_recovery_scan_v1.csv",
    "lane_08": "lane_08_batch_loso_stability_scan_v1.csv",
    "lane_09": "lane_09_v3_weight_profile_sim_scan_v1.csv",
    "lane_10": "lane_10_high_mfe_winner_stress_scan_v1.csv",
    "aggregator": "parallel_r5_2_v3_recall_search_aggregator_v1.json",
    "leaderboard": "v3_design_leaderboard_v1.csv",
    "decision": "v3_or_r6_head_next_decision_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

LANES = [
    "LANE_01_V2_REMAINING_GAP_TRACE_V1",
    "LANE_02_V2_VETO_FALSE_NEGATIVE_SCAN_V1",
    "LANE_03_BAD_RECALL_HEAD_STRENGTH_SCAN_V1",
    "LANE_04_TAIL_10_50_RECALL_SCAN_V1",
    "LANE_05_RISKY_ATTENTION_CONFIRMATION_SCAN_V1",
    "LANE_06_AMBIGUOUS_HIGH_MFE_REFINEMENT_SCAN_V1",
    "LANE_07_R6_OUTSIDE_BASE_MICRO_RECOVERY_SCAN_V1",
    "LANE_08_BATCH_LOSO_STABILITY_SCAN_V1",
    "LANE_09_V3_WEIGHT_PROFILE_SIM_SCAN_V1",
    "LANE_10_HIGH_MFE_WINNER_STRESS_SCAN_V1",
]


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


def _load_inputs(r6_v2_dir: Path, v2_execution_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.Series, dict[str, Any], dict[str, Any]]:
    frame_path = r6_v2_dir / TRAINING_FRAME
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)
    frame = pd.read_parquet(frame_path)
    selected_policy = _selected_policy_from_grid(r6_v2_dir)
    if not selected_policy:
        raise RuntimeError("Could not load selected R6 V2 objective policy from grid")
    selected = _selected_policy_mask(frame, selected_policy).reindex(frame.index).fillna(False).astype(bool)
    r6_summary = _read_json(r6_v2_dir / "summary_v1.json")
    v2_lock = _read_json(v2_execution_dir / "best_v2_variant_downstream_r6_input_lock_v1.json")
    if v2_lock.get("best_variant_id_v1") != BEST_VARIANT_ID:
        raise RuntimeError(f"Unexpected V2 best variant: {v2_lock.get('best_variant_id_v1')}")
    if v2_lock.get("base_flag_for_r6_v1") != V2_FINAL_BASE_FLAG:
        raise RuntimeError("V2 lock does not use final post-veto base flag")
    return frame, selected_policy, selected, r6_summary, v2_lock


def _safety_flags(frame: pd.DataFrame) -> pd.DataFrame:
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index)
    flags = pd.DataFrame(index=frame.index)
    flags["fifty_plus_v1"] = _bool(frame, "fifty_plus_mfe_v1")
    flags["hundred_plus_v1"] = _bool(frame, "hundred_plus_mfe_v1")
    flags["two_hundred_plus_v1"] = _bool(frame, "two_hundred_plus_mfe_v1")
    flags["strongest_winner_v1"] = _bool(frame, "strongest_winner_path_v1")
    flags["repaired_like_v1"] = _bool(frame, "r6_label_repaired_165_like_runner_v1")
    flags["forensic_repaired_trade_v1"] = forensic
    flags["runner_near_miss_v1"] = _bool(frame, "r6_label_runner_near_miss_v1")
    flags["ambiguous_high_mfe_v1"] = _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1") | _bool(frame, "high_mfe_ambiguous_protection_target")
    flags["runner_protect_v1"] = _bool(frame, "r5_2_label_runner_protect_v1") | _bool(frame, "runner_protection_target")
    flags["hard_winner_or_repaired_v1"] = (
        flags["hundred_plus_v1"]
        | flags["two_hundred_plus_v1"]
        | flags["strongest_winner_v1"]
        | flags["repaired_like_v1"]
        | flags["forensic_repaired_trade_v1"]
    )
    flags["strict_r6_no_go_v1"] = (
        flags["fifty_plus_v1"]
        | flags["hard_winner_or_repaired_v1"]
        | flags["runner_near_miss_v1"]
        | flags["ambiguous_high_mfe_v1"]
        | flags["runner_protect_v1"]
    )
    flags["dangerous_or_protected_v1"] = (
        flags["hard_winner_or_repaired_v1"]
        | flags["runner_near_miss_v1"]
        | flags["ambiguous_high_mfe_v1"]
        | flags["runner_protect_v1"]
    )
    return flags


def _missed_mask(frame: pd.DataFrame, selected: pd.Series) -> pd.Series:
    return (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1")) & ~selected


def _r6_micro_signal(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, R6_BAD_PROB).ge(0.75)
        & _num(frame, R6_RISKY_PROB).ge(0.50)
        & _num(frame, R6_TAIL_PROB).ge(0.50)
        & _num(frame, R6_RUNNER_PROB).lt(0.30)
        & _num(frame, R5_2_RUNNER_PROB).lt(0.74)
        & _num(frame, R6_BLINDSPOT_PROB).lt(0.70).fillna(True)
    ).fillna(False)


def _base_gap_bucket(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.Series:
    final = _bool(frame, V2_FINAL_BASE_FLAG)
    pre = _bool(frame, V2_PRE_VETO_BASE_FLAG)
    veto = _bool(frame, V2_HARD_VETO_FLAG)
    r6_signal = _r6_micro_signal(frame)
    values = np.select(
        [
            final & ~selected,
            pre & veto,
            flags["strict_r6_no_go_v1"],
            ~final & r6_signal,
            ~pre,
            pre & ~final,
        ],
        [
            "IN_FINAL_BASE_BUT_R6_NOT_SELECTED",
            "VETOED_BY_HARD_PROTECTION",
            "DANGEROUS_OR_PROTECTED",
            "R6_COULD_RECOVER_BUT_BASE_BOUND",
            "NOT_IN_V2_PRE_VETO_BASE",
            "IN_PRE_VETO_BUT_NOT_FINAL_BASE",
        ],
        default="SIGNAL_WEAK_OR_AMBIGUOUS",
    )
    return pd.Series(values, index=frame.index)


def _common_trace_cols() -> list[str]:
    return [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "run_id",
        "split_scope_v1",
        "batch_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
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
    ]


def _lane_01_gap_trace(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    missed = _missed_mask(frame, selected)
    out = frame.loc[missed].copy()
    out["lane_v1"] = "LANE_01_V2_REMAINING_GAP_TRACE_V1"
    out["r6_selected_v1"] = selected.loc[out.index]
    out["gap_bucket_v1"] = _base_gap_bucket(out, selected.loc[out.index], flags.loc[out.index])
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = ["lane_v1", *_common_trace_cols(), "r6_selected_v1", "gap_bucket_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _lane_02_veto_false_negative(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    missed = _missed_mask(frame, selected)
    vetoed = missed & _bool(frame, V2_PRE_VETO_BASE_FLAG) & _bool(frame, V2_HARD_VETO_FLAG)
    out = frame.loc[vetoed].copy()
    if out.empty:
        return pd.DataFrame(columns=[*_common_trace_cols(), "veto_classification_v1", "safe_recoverable_v1"])
    safe = ~flags.loc[out.index, "strict_r6_no_go_v1"]
    ambiguous = flags.loc[out.index, "ambiguous_high_mfe_v1"] | flags.loc[out.index, "fifty_plus_v1"]
    out["veto_classification_v1"] = np.select(
        [safe, ambiguous],
        ["OVERCONSERVATIVE_VETO_SAFE_RECOVERABLE", "AMBIGUOUS_KEEP_PROTECTED"],
        default="CORRECT_VETO_DANGEROUS",
    )
    out["safe_recoverable_v1"] = safe
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = [*_common_trace_cols(), "veto_classification_v1", "safe_recoverable_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _score_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series.astype(float),
        bins=[-np.inf, 0.01, 0.20, 0.50, 0.80, np.inf],
        labels=["LT_0_01", "0_01_0_20", "0_20_0_50", "0_50_0_80", "GE_0_80"],
    ).astype("string")


def _lane_03_bad_strength(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    mask = _bool(frame, "label_should_not_take_v1") & ~selected
    out = frame.loc[mask].copy()
    out["bad_recall_score_bucket_v1"] = _score_bucket(_num(out, V2_BAD_SCORE))
    out["v3_bad_focused_candidate_v1"] = (
        ~flags.loc[out.index, "strict_r6_no_go_v1"]
        & (_num(out, V2_BAD_SCORE).ge(0.20) | _num(out, V2_RISKY_SCORE).ge(0.50) | _num(out, R6_BAD_PROB).ge(0.50))
    ).fillna(False)
    out["bad_head_diagnosis_v1"] = np.select(
        [
            flags.loc[out.index, "strict_r6_no_go_v1"],
            _num(out, V2_BAD_SCORE).ge(0.50),
            _num(out, V2_RISKY_SCORE).ge(0.50),
        ],
        ["DANGEROUS_OR_PROTECTED", "BAD_RECALL_HEAD_HAS_SIGNAL", "RISKY_SUPPORT_BUT_BAD_HEAD_WEAK"],
        default="BAD_HEAD_SCORE_WEAK",
    )
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = [*_common_trace_cols(), "bad_recall_score_bucket_v1", "v3_bad_focused_candidate_v1", "bad_head_diagnosis_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _lane_04_tail_scan(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    mask = _bool(frame, "tail_10_50_mfe_v1") & ~selected
    out = frame.loc[mask].copy()
    out["tail_recall_score_bucket_v1"] = _score_bucket(_num(out, V2_TAIL_SCORE))
    out["v3_tail_focused_candidate_v1"] = (
        ~flags.loc[out.index, "strict_r6_no_go_v1"]
        & (_num(out, V2_TAIL_SCORE).ge(0.20) | _num(out, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1").ge(0.50))
    ).fillna(False)
    out["tail_head_diagnosis_v1"] = np.select(
        [
            flags.loc[out.index, "strict_r6_no_go_v1"],
            _num(out, V2_TAIL_SCORE).ge(0.50),
            _num(out, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1").ge(0.50),
        ],
        ["DANGEROUS_OR_PROTECTED", "TAIL_RECALL_HEAD_HAS_SIGNAL", "R5_TAIL_SUPPORT_PRESENT_BUT_V2_TAIL_WEAK"],
        default="TAIL_HEAD_SCORE_WEAK",
    )
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = [*_common_trace_cols(), "pred__entry_r5_tail_control_10_50_risk__prob_true_v1", "tail_recall_score_bucket_v1", "v3_tail_focused_candidate_v1", "tail_head_diagnosis_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _lane_05_risky_confirmation(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    mask = _missed_mask(frame, selected)
    out = frame.loc[mask].copy()
    low_protect = (
        _num(out, V2_RUNNER_PROTECT_SCORE).lt(0.20)
        & _num(out, V2_AMBIGUOUS_PROTECT_SCORE).lt(0.20)
        & _num(out, V2_HARD_WINNER_PROTECT_SCORE).lt(0.20)
        & ~flags.loc[out.index, "strict_r6_no_go_v1"]
    )
    out["risky_high_bad_moderate_v1"] = (_num(out, V2_RISKY_SCORE).ge(0.65) & _num(out, V2_BAD_SCORE).ge(0.10) & low_protect).fillna(False)
    out["risky_high_tail_moderate_v1"] = (_num(out, V2_RISKY_SCORE).ge(0.65) & _num(out, V2_TAIL_SCORE).ge(0.10) & low_protect).fillna(False)
    out["risky_high_runner_low_v1"] = (_num(out, V2_RISKY_SCORE).ge(0.65) & low_protect).fillna(False)
    out["safe_recall_opportunity_v1"] = out[["risky_high_bad_moderate_v1", "risky_high_tail_moderate_v1", "risky_high_runner_low_v1"]].any(axis=1)
    out["risky_confirmation_class_v1"] = np.select(
        [flags.loc[out.index, "strict_r6_no_go_v1"], out["safe_recall_opportunity_v1"], _num(out, V2_RISKY_SCORE).ge(0.65)],
        ["RISKY_SIGNAL_UNSAFE", "RISKY_CAN_CONFIRM_SAFE_RECOVERY", "RISKY_HIGH_BUT_PROTECTION_NOT_LOW"],
        default="RISKY_SIGNAL_WEAK",
    )
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = [*_common_trace_cols(), "risky_high_bad_moderate_v1", "risky_high_tail_moderate_v1", "risky_high_runner_low_v1", "safe_recall_opportunity_v1", "risky_confirmation_class_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _lane_06_ambiguous_refinement(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    mask = flags["ambiguous_high_mfe_v1"] & ~selected
    out = frame.loc[mask].copy()
    if out.empty:
        return pd.DataFrame(columns=[*_common_trace_cols(), "ambiguous_refinement_class_v1"])
    non_high_mfe = ~(_bool(out, "fifty_plus_mfe_v1") | _bool(out, "hundred_plus_mfe_v1") | _bool(out, "two_hundred_plus_mfe_v1"))
    low_protect = _num(out, V2_RUNNER_PROTECT_SCORE).lt(0.20) & _num(out, V2_HARD_WINNER_PROTECT_SCORE).lt(0.20)
    high_recall = _num(out, V2_BAD_SCORE).ge(0.65) | _num(out, V2_TAIL_SCORE).ge(0.65)
    safe_recoverable = non_high_mfe & low_protect & high_recall
    out["ambiguous_refinement_class_v1"] = np.select(
        [safe_recoverable, _bool(out, "fifty_plus_mfe_v1") | _bool(out, "hundred_plus_mfe_v1") | _bool(out, "two_hundred_plus_mfe_v1")],
        ["AMBIGUOUS_POSSIBLY_SAFE_WITH_EXPLICIT_PROOF_REQUIRED", "CORRECT_AMBIGUOUS_HIGH_MFE_PROTECTION"],
        default="AMBIGUOUS_KEEP_PROTECTED",
    )
    out["safe_bevis_required_before_bad_positive_v1"] = safe_recoverable
    for column in flags.columns:
        out[column] = flags.loc[out.index, column]
    cols = [*_common_trace_cols(), "ambiguous_refinement_class_v1", "safe_bevis_required_before_bad_positive_v1", *flags.columns.tolist()]
    return out[[column for column in cols if column in out.columns]]


def _rule_mask(frame: pd.DataFrame, flags: pd.DataFrame, rule_id: str) -> pd.Series:
    outside = ~_bool(frame, V2_FINAL_BASE_FLAG)
    safe = ~flags["strict_r6_no_go_v1"]
    common = outside & safe & _num(frame, R6_RUNNER_PROB).lt(0.30) & _num(frame, R5_2_RUNNER_PROB).lt(0.74) & _num(frame, R6_BLINDSPOT_PROB).lt(0.70).fillna(True)
    if rule_id == "R6_MICRO_ULTRA_STRICT_CONSENSUS":
        return common & _num(frame, R6_BAD_PROB).ge(0.80) & _num(frame, R6_RISKY_PROB).ge(0.80) & _num(frame, R6_TAIL_PROB).ge(0.80)
    if rule_id == "R6_MICRO_STRICT_CONSENSUS":
        return common & _num(frame, R6_BAD_PROB).ge(0.75) & _num(frame, R6_RISKY_PROB).ge(0.50) & _num(frame, R6_TAIL_PROB).ge(0.50)
    if rule_id == "R6_MICRO_BAD_DOMINANT":
        return common & _num(frame, R6_BAD_PROB).ge(0.80) & _num(frame, R6_RISKY_PROB).ge(0.50)
    if rule_id == "R6_MICRO_TAIL_DOMINANT":
        return common & _num(frame, R6_TAIL_PROB).ge(0.85) & _num(frame, R6_BAD_PROB).ge(0.50)
    if rule_id == "R6_MICRO_RISKY_BAD":
        return common & _num(frame, R6_RISKY_PROB).ge(0.80) & _num(frame, R6_BAD_PROB).ge(0.50)
    return pd.Series(False, index=frame.index)


def _lane_07_r6_outside_base(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule_id in [
        "R6_MICRO_ULTRA_STRICT_CONSENSUS",
        "R6_MICRO_STRICT_CONSENSUS",
        "R6_MICRO_BAD_DOMINANT",
        "R6_MICRO_TAIL_DOMINANT",
        "R6_MICRO_RISKY_BAD",
    ]:
        add = _rule_mask(frame, flags, rule_id)
        candidate = (selected | add).fillna(False).astype(bool)
        metrics = _frame_selected_metrics(frame, candidate)
        rows.append(
            {
                "rule_id_v1": rule_id,
                "outside_base_rows_added_v1": int((add & ~selected).sum()),
                "bad_uplift_v1": int((add & ~selected & _bool(frame, "label_should_not_take_v1")).sum()),
                "tail_uplift_v1": int((add & ~selected & _bool(frame, "tail_10_50_mfe_v1")).sum()),
                "precision_v1": metrics["precision_v1"],
                "worst_loso_v1": metrics["worst_loso_v1"],
                "fifty_plus_mfe_blocked_v1": metrics["fifty_plus_mfe_blocked_v1"],
                "hundred_plus_mfe_blocked_v1": metrics["hundred_plus_mfe_blocked_v1"],
                "two_hundred_plus_mfe_blocked_v1": metrics["two_hundred_plus_mfe_blocked_v1"],
                "strongest_winner_damage_v1": metrics["strongest_winner_damage_v1"],
                "repaired_damage_v1": metrics["repaired_damage_v1"],
                "runner_near_miss_blocked_v1": metrics["runner_near_miss_blocked_v1"],
                "safety_pass_v1": _safety_pass(metrics),
                "worth_hybrid_design_v1": bool(_safety_pass(metrics) and int((add & ~selected).sum()) > 0),
            }
        )
    return pd.DataFrame(rows)


def _lane_08_stability(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    missed = _missed_mask(frame, selected)
    safe_missed = missed & ~flags["strict_r6_no_go_v1"]
    rows: list[dict[str, Any]] = []
    for level in ["split_scope_v1", "batch_scope_v1", "run_id"]:
        if level not in frame.columns:
            continue
        for value, group in frame.groupby(level, dropna=False):
            idx = group.index
            selected_bad = int((selected.loc[idx] & _bool(group, "label_should_not_take_v1")).sum())
            rows.append(
                {
                    "grouping_level_v1": level,
                    "group_v1": str(value),
                    "row_count_v1": int(len(group)),
                    "selected_bad_v1": selected_bad,
                    "selected_tail_v1": int((selected.loc[idx] & _bool(group, "tail_10_50_mfe_v1")).sum()),
                    "missed_bad_v1": int((missed.loc[idx] & _bool(group, "label_should_not_take_v1")).sum()),
                    "missed_tail_v1": int((missed.loc[idx] & _bool(group, "tail_10_50_mfe_v1")).sum()),
                    "safe_recoverable_bad_target_rows_v1": int((safe_missed.loc[idx] & _bool(group, "label_should_not_take_v1")).sum()),
                    "safe_recoverable_tail_target_rows_v1": int((safe_missed.loc[idx] & _bool(group, "tail_10_50_mfe_v1")).sum()),
                    "recall_collapse_slice_v1": bool(selected_bad == 0 and int((missed.loc[idx] & _bool(group, "label_should_not_take_v1")).sum()) > 0),
                    "split_aware_weighting_candidate_v1": bool(int((safe_missed.loc[idx] & _bool(group, "label_should_not_take_v1")).sum()) >= 5),
                }
            )
    return pd.DataFrame(rows)


def _profile_mask(frame: pd.DataFrame, flags: pd.DataFrame, profile_id: str) -> pd.Series:
    missed_safe = _missed_mask(frame, pd.Series(False, index=frame.index)) & ~flags["strict_r6_no_go_v1"] & ~_bool(frame, V2_FINAL_BASE_FLAG)
    bad = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    if profile_id == "V3_BAD_RECALL_STRONGER_WITH_SAME_VETO":
        return missed_safe & bad
    if profile_id == "V3_TAIL_10_50_STRONGER_WITH_SAME_VETO":
        return missed_safe & tail
    if profile_id == "V3_RISKY_CONFIRMATION_STRONGER":
        return missed_safe & (_num(frame, V2_RISKY_SCORE).ge(0.50) | _num(frame, R6_RISKY_PROB).ge(0.50))
    if profile_id == "V3_PROTECTION_HEAVY_WITH_RECALL":
        return missed_safe & bad & (_num(frame, V2_RUNNER_PROTECT_SCORE).lt(0.95) | _num(frame, V2_BAD_SCORE).ge(0.50))
    if profile_id == "V3_AMBIGUOUS_HARD_NEGATIVE_PLUS_TAIL":
        return missed_safe & tail
    if profile_id == "V3_BATCH_STABLE_RECALL":
        return missed_safe & bad & frame.get("batch_scope_v1", pd.Series("", index=frame.index)).astype("string").isin(["BATCH_01", "BATCH_02", "BATCH_03", "BATCH_04", "BATCH_05"])
    if profile_id == "V3_BAD_TAIL_MULTI_HEAD_CONSENSUS":
        support = (
            _num(frame, V2_BAD_SCORE).ge(0.20).astype(int)
            + _num(frame, V2_RISKY_SCORE).ge(0.50).astype(int)
            + _num(frame, R6_BAD_PROB).ge(0.50).astype(int)
            + _num(frame, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1").ge(0.50).astype(int)
        )
        return missed_safe & bad & support.ge(1)
    if profile_id == "V3_ULTRA_SAFE_STRONGER_RECALL":
        return missed_safe & bad & _num(frame, V2_BAD_SCORE).ge(0.50) & _num(frame, V2_RUNNER_PROTECT_SCORE).lt(0.20)
    if profile_id == "V3_SPLIT_AWARE_TAIL_RECOVERY":
        return missed_safe & tail & frame.get("split_scope_v1", pd.Series("", index=frame.index)).astype("string").str.upper().str.contains("TRAIN|VALID|HOLDOUT")
    if profile_id == "V3_HYBRID_R5_2_PLUS_R6_MICRO_RECOVERY":
        return (missed_safe & (bad | tail) & (_num(frame, V2_BAD_SCORE).ge(0.20) | _num(frame, V2_RISKY_SCORE).ge(0.50))) | _rule_mask(frame, flags, "R6_MICRO_STRICT_CONSENSUS")
    return pd.Series(False, index=frame.index)


def _lane_09_profiles(frame: pd.DataFrame, selected: pd.Series, flags: pd.DataFrame, lane07: pd.DataFrame) -> pd.DataFrame:
    profile_ids = [
        "V3_BAD_RECALL_STRONGER_WITH_SAME_VETO",
        "V3_TAIL_10_50_STRONGER_WITH_SAME_VETO",
        "V3_RISKY_CONFIRMATION_STRONGER",
        "V3_PROTECTION_HEAVY_WITH_RECALL",
        "V3_AMBIGUOUS_HARD_NEGATIVE_PLUS_TAIL",
        "V3_BATCH_STABLE_RECALL",
        "V3_BAD_TAIL_MULTI_HEAD_CONSENSUS",
        "V3_ULTRA_SAFE_STRONGER_RECALL",
        "V3_SPLIT_AWARE_TAIL_RECOVERY",
        "V3_HYBRID_R5_2_PLUS_R6_MICRO_RECOVERY",
    ]
    rows: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        mask = _profile_mask(frame, flags, profile_id)
        risk = mask & flags["strict_r6_no_go_v1"]
        r6_micro = bool(profile_id == "V3_HYBRID_R5_2_PLUS_R6_MICRO_RECOVERY" and lane07["worth_hybrid_design_v1"].any())
        rows.append(
            {
                "profile_id_v1": profile_id,
                "target_rows_v1": int(mask.sum()),
                "target_bad_rows_v1": int((mask & _bool(frame, "label_should_not_take_v1")).sum()),
                "target_tail_rows_v1": int((mask & _bool(frame, "tail_10_50_mfe_v1")).sum()),
                "protected_rows_kept_out_v1": int((~mask & flags["strict_r6_no_go_v1"]).sum()),
                "expected_bad_uplift_opportunity_v1": int((mask & _bool(frame, "label_should_not_take_v1")).sum()),
                "expected_tail_uplift_opportunity_v1": int((mask & _bool(frame, "tail_10_50_mfe_v1")).sum()),
                "expected_safety_risk_rows_v1": int(risk.sum()),
                "expected_r6_pass_through_rows_v1": int(mask.sum()),
                "training_required_v1": True,
                "include_r6_micro_recovery_v1": r6_micro,
                "existing_features_enough_v1": "INDIKERT_NOT_PROVEN_BY_READ_ONLY_SCAN",
                "profile_status_v1": "PROMISING_V3_DESIGN_CANDIDATE" if int(mask.sum()) and int(risk.sum()) == 0 else "NO_GO_OR_TOO_WEAK",
            }
        )
    return pd.DataFrame(rows)


def _lane_10_stress(frame: pd.DataFrame, flags: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_id in profiles["profile_id_v1"].astype(str).tolist():
        mask = _profile_mask(frame, flags, profile_id)
        rows.append(
            {
                "candidate_rule_or_profile_v1": profile_id,
                "candidate_rows_v1": int(mask.sum()),
                "fifty_plus_overlap_v1": int((mask & flags["fifty_plus_v1"]).sum()),
                "hundred_plus_overlap_v1": int((mask & flags["hundred_plus_v1"]).sum()),
                "two_hundred_plus_overlap_v1": int((mask & flags["two_hundred_plus_v1"]).sum()),
                "strongest_winner_overlap_v1": int((mask & flags["strongest_winner_v1"]).sum()),
                "repaired_like_overlap_v1": int((mask & flags["repaired_like_v1"]).sum()),
                "forensic_repaired_trade_overlap_v1": int((mask & flags["forensic_repaired_trade_v1"]).sum()),
                "runner_near_miss_overlap_v1": int((mask & flags["runner_near_miss_v1"]).sum()),
                "ambiguous_high_mfe_overlap_v1": int((mask & flags["ambiguous_high_mfe_v1"]).sum()),
                "safe_classification_v1": "SAFE_BY_CURRENT_HARD_FLAGS" if int((mask & flags["strict_r6_no_go_v1"]).sum()) == 0 else "UNSAFE_NO_GO",
                "no_go_reason_v1": "NONE" if int((mask & flags["strict_r6_no_go_v1"]).sum()) == 0 else "HIGH_MFE_WINNER_OR_PROTECTION_OVERLAP",
            }
        )
    return pd.DataFrame(rows)


def _leaderboard(profiles: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    out = profiles.merge(
        stress[["candidate_rule_or_profile_v1", "safe_classification_v1"]],
        left_on="profile_id_v1",
        right_on="candidate_rule_or_profile_v1",
        how="left",
    )
    out["safety_pass_readonly_v1"] = out["expected_safety_risk_rows_v1"].eq(0) & out["safe_classification_v1"].eq("SAFE_BY_CURRENT_HARD_FLAGS")
    out["leaderboard_score_v1"] = (
        out["safety_pass_readonly_v1"].astype(int) * 1_000_000
        + out["expected_bad_uplift_opportunity_v1"].astype(int) * 1000
        + out["expected_tail_uplift_opportunity_v1"].astype(int)
        - out["expected_safety_risk_rows_v1"].astype(int) * 10_000
    )
    return out.sort_values(
        ["safety_pass_readonly_v1", "leaderboard_score_v1", "expected_bad_uplift_opportunity_v1", "expected_tail_uplift_opportunity_v1"],
        ascending=[False, False, False, False],
    )


def _count_by(df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    if df.empty or group_col not in df.columns:
        return {}
    rows: dict[str, Any] = {}
    for key, group in df.groupby(group_col, dropna=False):
        rows[str(key)] = {
            "rows_v1": int(len(group)),
            "bad_v1": int(_bool(group, "label_should_not_take_v1").sum()),
            "tail_v1": int(_bool(group, "tail_10_50_mfe_v1").sum()),
            "fifty_plus_v1": int(_bool(group, "fifty_plus_v1").sum()) if "fifty_plus_v1" in group.columns else int(_bool(group, "fifty_plus_mfe_v1").sum()),
            "hundred_plus_v1": int(_bool(group, "hundred_plus_v1").sum()) if "hundred_plus_v1" in group.columns else int(_bool(group, "hundred_plus_mfe_v1").sum()),
            "strongest_v1": int(_bool(group, "strongest_winner_v1").sum()) if "strongest_winner_v1" in group.columns else int(_bool(group, "strongest_winner_path_v1").sum()),
        }
    return rows


def _aggregator(
    *,
    frame: pd.DataFrame,
    selected: pd.Series,
    lane01: pd.DataFrame,
    lane02: pd.DataFrame,
    lane03: pd.DataFrame,
    lane04: pd.DataFrame,
    lane05: pd.DataFrame,
    lane06: pd.DataFrame,
    lane07: pd.DataFrame,
    lane08: pd.DataFrame,
    profiles: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> dict[str, Any]:
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    safe_outside = lane07[lane07["worth_hybrid_design_v1"].astype(bool)] if not lane07.empty else pd.DataFrame()
    gap_bad = int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - 95)
    gap_tail = int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - 61)
    return {
        "layer_name": "PARALLEL_R5_2_V3_RECALL_SEARCH_AGGREGATOR_V1",
        "lane_count_v1": 10,
        "all_lanes_read_only_v1": True,
        "current_r6_v2_v1": {"bad_v1": 95, "tail_v1": 61, "precision_v1": 1.0, "worst_loso_v1": 1.0, "safety_v1": "CLEAN"},
        "wednesday_benchmark_gap_v1": {"bad_v1": gap_bad, "tail_v1": gap_tail},
        "row_level_missed_after_v2_v1": {
            "bad_label_rows_v1": int((_bool(frame, "label_should_not_take_v1") & ~selected).sum()),
            "tail_rows_v1": int((_bool(frame, "tail_10_50_mfe_v1") & ~selected).sum()),
            "bad_or_tail_rows_v1": int(len(lane01)),
        },
        "gap_bucket_counts_v1": _count_by(lane01, "gap_bucket_v1"),
        "v2_veto_scan_v1": _count_by(lane02, "veto_classification_v1"),
        "bad_head_scan_v1": _count_by(lane03, "bad_head_diagnosis_v1"),
        "tail_head_scan_v1": _count_by(lane04, "tail_head_diagnosis_v1"),
        "risky_confirmation_safe_opportunity_rows_v1": int(lane05.get("safe_recall_opportunity_v1", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "ambiguous_refinement_v1": _count_by(lane06, "ambiguous_refinement_class_v1"),
        "r6_outside_base_safe_positive_rule_count_v1": int((lane07["safety_pass_v1"].astype(bool) & lane07["outside_base_rows_added_v1"].gt(0)).sum()) if not lane07.empty else 0,
        "r6_outside_base_best_v1": lane07.sort_values(["safety_pass_v1", "bad_uplift_v1", "tail_uplift_v1"], ascending=[False, False, False]).head(1).to_dict("records")[0] if not lane07.empty else {},
        "best_v3_design_candidate_v1": best,
        "best_bad_recall_opportunity_v1": leaderboard.sort_values(["expected_bad_uplift_opportunity_v1"], ascending=False).head(1).to_dict("records")[0] if not leaderboard.empty else {},
        "best_tail_recall_opportunity_v1": leaderboard.sort_values(["expected_tail_uplift_opportunity_v1"], ascending=False).head(1).to_dict("records")[0] if not leaderboard.empty else {},
        "safest_option_v1": leaderboard[leaderboard["safety_pass_readonly_v1"].astype(bool)].head(1).to_dict("records")[0] if not leaderboard.empty and leaderboard["safety_pass_readonly_v1"].any() else {},
        "best_hybrid_option_v1": leaderboard[leaderboard["profile_id_v1"].eq("V3_HYBRID_R5_2_PLUS_R6_MICRO_RECOVERY")].head(1).to_dict("records")[0] if not leaderboard.empty else {},
        "most_dangerous_tempting_option_v1": profiles.sort_values(["expected_bad_uplift_opportunity_v1", "expected_safety_risk_rows_v1"], ascending=[False, False]).head(1).to_dict("records")[0] if not profiles.empty else {},
        "lanes_where_no_safe_signal_exists_v1": [
            "LANE_07_R6_OUTSIDE_BASE_MICRO_RECOVERY_SCAN_V1"
        ]
        if safe_outside.empty
        else [],
    }


def _decision(aggregator: dict[str, Any], leaderboard: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    safe_outside = int(aggregator["r6_outside_base_safe_positive_rule_count_v1"])
    safe_v3 = leaderboard[leaderboard["safety_pass_readonly_v1"].astype(bool)] if not leaderboard.empty else pd.DataFrame()
    if safe_v3.empty and safe_outside == 0:
        decision = "STOP_RETRAIN_LOOP_AND_REVIEW_FEATURE_SIGNAL"
    elif safe_outside > 0 and not safe_v3.empty:
        decision = "IMPLEMENT_HYBRID_V3_PLUS_R6_MICRO_RECOVERY_SPEC"
    elif safe_outside > 0:
        decision = "DESIGN_R6_SAFE_OUTSIDE_BASE_RECOVERY_RUNNER"
    else:
        decision = "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER"
    payload = {
        "layer_name": "V3_OR_R6_HEAD_NEXT_DECISION_V1",
        "decision_v1": decision,
        "why_v1": {
            "r6_outside_base_safe_positive_rule_count_v1": safe_outside,
            "safe_v3_design_candidate_count_v1": int(len(safe_v3)),
            "best_v3_design_candidate_v1": aggregator.get("best_v3_design_candidate_v1", {}),
            "r6_v2_selected_candidate_base_bound_v1": True,
            "requires_training_v1": decision in {
                "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER",
                "IMPLEMENT_HYBRID_V3_PLUS_R6_MICRO_RECOVERY_SPEC",
            },
        },
    }
    next_action = {"layer_name": "NEXT_ACTION_LOCK_V1", "next_action_v1": decision}
    return payload, next_action


def _orchestrator(output_dir: Path, r6_v2_dir: Path, v2_execution_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "PARALLEL_RECALL_SEARCH_ORCHESTRATOR_V1",
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "input_r6_v2_dir_v1": str(r6_v2_dir),
        "input_v2_execution_dir_v1": str(v2_execution_dir),
        "lane_count_v1": 10,
        "execution_mode_v1": "READ_ONLY_SCAN_NO_TRAINING_NO_R6_RUN",
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "lanes_v1": [{"lane_name_v1": lane, "namespace_v1": str(output_dir / "lanes" / lane.lower())} for lane in LANES],
    }


def _audit(summary: dict[str, Any], decision: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("READ_ONLY_NO_TRAINING", True, summary["training_started_v1"]),
            row("NO_R6_RUN", True, summary["r6_started_v1"]),
            row("LANE_COUNT_10", summary["lane_count_v1"] == 10, summary["lane_count_v1"]),
            row("R6_OUTSIDE_BASE_RECHECK", True, summary["r6_outside_base_safe_positive_rule_count_v1"]),
            row("NEXT_DECISION_SET", bool(decision["decision_v1"]), decision["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Parallel R5.2 V3 And R6 Head Recall Search",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- R6 V2 current: `{summary['current_bad_v1']}/{summary['current_tail_v1']}`",
            f"- Benchmark delta vs Wednesday: `{summary['wednesday_gap_bad_v1']}/{summary['wednesday_gap_tail_v1']}`",
            f"- Row-level missed bad/tail after V2: `{summary['missed_bad_rows_v1']}/{summary['missed_tail_rows_v1']}`",
            f"- R6 outside-base safe positive rules: `{summary['r6_outside_base_safe_positive_rule_count_v1']}`",
            f"- Best V3 profile: `{summary['best_v3_profile_v1']}`",
            "",
            "The scan is read-only: no model training, no R6 retrain, no baseline or feature surface was built.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    r6_v2_dir: Path = DEFAULT_R6_V2_DIR,
    v2_execution_dir: Path = DEFAULT_V2_EXECUTION_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    r6_v2_dir = r6_v2_dir.expanduser().resolve()
    v2_execution_dir = v2_execution_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for lane in LANES:
        (output_dir / "lanes" / lane.lower()).mkdir(parents=True, exist_ok=True)

    frame, selected_policy, selected, r6_summary, v2_lock = _load_inputs(r6_v2_dir, v2_execution_dir)
    flags = _safety_flags(frame)
    lane01 = _lane_01_gap_trace(frame, selected, flags)
    lane02 = _lane_02_veto_false_negative(frame, selected, flags)
    lane03 = _lane_03_bad_strength(frame, selected, flags)
    lane04 = _lane_04_tail_scan(frame, selected, flags)
    lane05 = _lane_05_risky_confirmation(frame, selected, flags)
    lane06 = _lane_06_ambiguous_refinement(frame, selected, flags)
    lane07 = _lane_07_r6_outside_base(frame, selected, flags)
    lane08 = _lane_08_stability(frame, selected, flags)
    lane09 = _lane_09_profiles(frame, selected, flags, lane07)
    lane10 = _lane_10_stress(frame, flags, lane09)
    leaderboard = _leaderboard(lane09, lane10)
    aggregator = _aggregator(
        frame=frame,
        selected=selected,
        lane01=lane01,
        lane02=lane02,
        lane03=lane03,
        lane04=lane04,
        lane05=lane05,
        lane06=lane06,
        lane07=lane07,
        lane08=lane08,
        profiles=lane09,
        leaderboard=leaderboard,
    )
    decision, next_action = _decision(aggregator, leaderboard)
    best = aggregator.get("best_v3_design_candidate_v1") or {}
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "training_started_v1": False,
        "r6_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "lane_count_v1": 10,
        "current_bad_v1": 95,
        "current_tail_v1": 61,
        "wednesday_gap_bad_v1": 85,
        "wednesday_gap_tail_v1": 88,
        "missed_bad_rows_v1": aggregator["row_level_missed_after_v2_v1"]["bad_label_rows_v1"],
        "missed_tail_rows_v1": aggregator["row_level_missed_after_v2_v1"]["tail_rows_v1"],
        "gap_bucket_counts_v1": aggregator["gap_bucket_counts_v1"],
        "v2_veto_overconservative_rows_v1": int((lane02.get("veto_classification_v1", pd.Series(dtype=str)).astype(str).eq("OVERCONSERVATIVE_VETO_SAFE_RECOVERABLE")).sum()) if not lane02.empty else 0,
        "r6_outside_base_safe_positive_rule_count_v1": aggregator["r6_outside_base_safe_positive_rule_count_v1"],
        "best_v3_profile_v1": best.get("profile_id_v1"),
        "best_v3_expected_bad_uplift_opportunity_v1": best.get("expected_bad_uplift_opportunity_v1"),
        "best_v3_expected_tail_uplift_opportunity_v1": best.get("expected_tail_uplift_opportunity_v1"),
        "decision_v1": decision["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "hard_status_v1": {
            "BEVIST": [
                "10 read-only lanes were materialized from existing V2/R6 outputs.",
                "No training, no R6 run, no baseline and no feature surface were created.",
                "R6 outside-base micro recovery found no positive safe rule if count is zero.",
            ],
            "INDIKERT": [
                "V3 objective work is the most promising next path when safe V3 target opportunity exists.",
            ],
            "IKKE_ETABLERT": [
                "Read-only target opportunity is not proof that a trained V3 model will learn it.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_r6_v2_dir_v1": str(r6_v2_dir),
        "input_v2_execution_dir_v1": str(v2_execution_dir),
        "selected_policy_v1": selected_policy,
        "r6_summary_v1": r6_summary,
        "v2_lock_v1": v2_lock,
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "COMPLETED_READ_ONLY_SCAN",
        "decision_v1": decision["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
    }

    _write_json(output_dir / OUTPUT_FILES["orchestrator"], _orchestrator(output_dir, r6_v2_dir, v2_execution_dir))
    lane01.to_csv(output_dir / OUTPUT_FILES["lane_01"], index=False)
    lane02.to_csv(output_dir / OUTPUT_FILES["lane_02"], index=False)
    lane03.to_csv(output_dir / OUTPUT_FILES["lane_03"], index=False)
    lane04.to_csv(output_dir / OUTPUT_FILES["lane_04"], index=False)
    lane05.to_csv(output_dir / OUTPUT_FILES["lane_05"], index=False)
    lane06.to_csv(output_dir / OUTPUT_FILES["lane_06"], index=False)
    lane07.to_csv(output_dir / OUTPUT_FILES["lane_07"], index=False)
    lane08.to_csv(output_dir / OUTPUT_FILES["lane_08"], index=False)
    lane09.to_csv(output_dir / OUTPUT_FILES["lane_09"], index=False)
    lane10.to_csv(output_dir / OUTPUT_FILES["lane_10"], index=False)
    _write_json(output_dir / OUTPUT_FILES["aggregator"], aggregator)
    leaderboard.to_csv(output_dir / OUTPUT_FILES["leaderboard"], index=False)
    _write_json(output_dir / OUTPUT_FILES["decision"], decision)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _audit(summary, decision).to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--r6-v2-dir", type=Path, default=DEFAULT_R6_V2_DIR)
    parser.add_argument("--v2-execution-dir", type=Path, default=DEFAULT_V2_EXECUTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        r6_v2_dir=args.r6_v2_dir,
        v2_execution_dir=args.v2_execution_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
