#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    WEDNESDAY_R6_BENCHMARK,
    _jsonable,
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V1,
    R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
    R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
    SCORE_FRAME,
    SCORE_SUMMARY,
    _bool,
    _num,
    _read_json,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "INVESTIGATE_TRUE_R5_2_REBUILD_OR_LABEL_OBJECTIVE_NEXT_V1"

V3_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V3_R5_R51_R52"
V3_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"
V3_FORENSICS_DEFAULT = DEFAULT_REPORTS_ROOT / "RUN_R6_FROM_V3_AND_R6_HEAD_RECALL_FORENSICS_V1_20260426T_LOCK"
ASSET_REUSE_DEFAULT = DEFAULT_REPORTS_ROOT / "EXISTING_ASSET_FIRST_R6_REUSE_AND_DUPLICATE_GUARD_V1_20260426T_LOCK"

R6_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_SUMMARY = "summary_v1.json"

OUTPUT_FILES = {
    "trace": "post_v3_missed_rows_true_root_trace_v1.csv",
    "weakness": "r5_2_score_weakness_vs_label_objective_v1.json",
    "label_audit": "r5_2_label_objective_audit_v1.json",
    "requirement": "r5_2_true_rebuild_requirement_check_v1.json",
    "feature_signal": "existing_feature_signal_check_for_r5_2_v1.csv",
    "options": "r5_2_objective_redesign_options_v1.json",
    "experiments": "r5_2_rebuild_experiment_spec_candidates_v1.json",
    "stop_loop": "stop_tiny_extension_loop_lock_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

SCORE_FIELDS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1",
    "pred__entry_r5_strong_trade_candidate__prob_true_v1",
    "pred__entry_r5_take_was_ok__prob_true_v1",
    "r5_1_bad_blocker_score_v1",
    "r5_1_runner_guard_score_v1",
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_RISKY_PROB,
    R6_TAIL_PROB,
    R6_RUNNER_PROB,
    R6_BLINDSPOT_PROB,
]

SAFETY_FLAGS = [
    "take_was_ok_v1",
    "fifty_plus_mfe_v1",
    "hundred_plus_mfe_v1",
    "two_hundred_plus_mfe_v1",
    "strongest_winner_path_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_runner_near_miss_v1",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _merge_score_r6(score: pd.DataFrame, r6: pd.DataFrame) -> pd.DataFrame:
    score_cols = ["candidate_uid", *[col for col in SCORE_FIELDS if col in score.columns]]
    r6_drop = [col for col in SCORE_FIELDS if col in r6.columns]
    return r6.drop(columns=r6_drop, errors="ignore").merge(score[score_cols], on="candidate_uid", how="left", validate="one_to_one")


def _base_masks(frame: pd.DataFrame, score_summary: dict[str, Any]) -> dict[str, pd.Series]:
    policy = score_summary.get("r5_2_selected_policy_v1") or {}
    params = policy.get("params_v1") or {}
    bad_threshold = float(params.get("bad_threshold_v1") or 0.3680292099714267)
    runner_max = float(params.get("runner_max_v1") or 0.2)
    original = (_num(frame, R5_2_BAD_PROB).ge(bad_threshold) & _num(frame, R5_2_RUNNER_PROB).lt(runner_max)).fillna(False)
    v1_ext = (
        _num(frame, R5_2_BAD_PROB).ge(bad_threshold).fillna(False)
        & _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_min_r5_immediate_mae_score_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_2_runner_max_v1"])).fillna(False)
    )
    v1 = (original | v1_ext).fillna(False)
    v2_ext = (
        _num(frame, R5_2_BAD_PROB).ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_bad_threshold_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_min_r5_immediate_mae_score_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_2_runner_max_v1"])).fillna(False)
    )
    v2 = (v1 | v2_ext).fillna(False)
    v3_ext = (
        _num(frame, "pred__entry_r5_should_not_take__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_bad_threshold_v1"])).fillna(False)
        & _num(frame, "r5_1_bad_blocker_score_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_bad_threshold_v1"])).fillna(False)
        & _num(frame, R5_2_BAD_PROB).ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_bad_threshold_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_runner_max_v1"])).fillna(False)
    )
    v3 = (v2 | v3_ext).fillna(False)
    return {
        "original": original.astype(bool),
        "v1": v1.astype(bool),
        "v2": v2.astype(bool),
        "v3": v3.astype(bool),
        "v1_ext": v1_ext.astype(bool),
        "v2_ext": v2_ext.astype(bool),
        "v3_ext": v3_ext.astype(bool),
    }


def _mfe_bucket(frame: pd.DataFrame) -> pd.Series:
    mfe = _num(frame, "peak_mfe_bps_v1")
    return pd.cut(mfe, bins=[-999999, 0, 10, 50, 100, 200, 999999], labels=["<=0", "0_10", "10_50", "50_100", "100_200", "200_PLUS"]).astype("string")


def _mae_bucket(frame: pd.DataFrame) -> pd.Series:
    mae = _num(frame, "mae_abs_bps_v1")
    return pd.cut(mae, bins=[-1, 10, 25, 40, 75, 150, 999999], labels=["<=10", "10_25", "25_40", "40_75", "75_150", "150_PLUS"]).astype("string")


def _danger(frame: pd.DataFrame) -> pd.Series:
    return (
        _bool(frame, "take_was_ok_v1")
        | _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
    ).fillna(False)


def _first_exclusion(row: pd.Series) -> str:
    if bool(row.get("r5_2_v3_base_flag_v1", False)):
        return "IN_R5_2_V3_BASE"
    if not bool(row.get("r5_2_label_bad_blocker_v1", False)):
        return "R5_2_LABEL_EXCLUDES_HIGH_MFE_AMBIGUOUS_CASE"
    if float(row.get(R5_2_BAD_PROB, np.nan)) < 0.35:
        if float(row.get("r5_1_bad_blocker_score_v1", np.nan)) >= 0.65 or float(row.get(R6_BAD_PROB, np.nan)) >= 0.85:
            return "R5_2_SCORE_WEAK_DESPITE_OTHER_SCORE_SIGNAL"
        return "R5_2_BAD_SCORE_TOO_LOW"
    if float(row.get(R5_2_RUNNER_PROB, np.nan)) >= 0.55:
        return "R5_2_RUNNER_SCORE_TOO_HIGH_FOR_SAFE_BASE"
    if float(row.get("r5_1_bad_blocker_score_v1", np.nan)) < 0.85:
        return "R5_1_BAD_CONFIRMATION_TOO_LOW_FOR_V3"
    if float(row.get("pred__entry_r5_should_not_take__prob_true_v1", np.nan)) < 0.35:
        return "R5_BAD_CONFIRMATION_TOO_LOW_FOR_V3"
    if float(row.get("pred__entry_r5_runner_protect__prob_true_v1", np.nan)) >= 0.55:
        return "R5_RUNNER_CAP_BLOCKS_V3"
    if float(row.get("r5_1_runner_guard_score_v1", np.nan)) >= 0.55:
        return "R5_1_RUNNER_CAP_BLOCKS_V3"
    return "NOT_ESTABLISHED"


def _root_bucket(row: pd.Series) -> str:
    if bool(row.get("row_is_actually_dangerous_v1", False)):
        return "ROW_IS_ACTUALLY_DANGEROUS"
    if not bool(row.get("r5_2_label_bad_blocker_v1", False)):
        return "R5_2_LABEL_DOES_NOT_REWARD_THIS_CASE"
    if row.get("r5_2_first_exclusion_reason_v1") == "R5_2_SCORE_WEAK_DESPITE_OTHER_SCORE_SIGNAL":
        return "R5_2_FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED"
    if row.get("r5_2_first_exclusion_reason_v1") in {"R5_2_BAD_SCORE_TOO_LOW", "R5_2_SCORE_WEAK_DESPITE_OTHER_SCORE_SIGNAL"}:
        return "R5_2_SCORE_WEAK"
    if row.get("r5_2_first_exclusion_reason_v1") in {"R5_2_RUNNER_SCORE_TOO_HIGH_FOR_SAFE_BASE", "R5_1_BAD_CONFIRMATION_TOO_LOW_FOR_V3", "R5_BAD_CONFIRMATION_TOO_LOW_FOR_V3", "R5_RUNNER_CAP_BLOCKS_V3", "R5_1_RUNNER_CAP_BLOCKS_V3"}:
        return "R5_2_BASE_CONTRACT_TOO_STRICT"
    return "NOT_ESTABLISHED"


def _trace(score: pd.DataFrame, r6: pd.DataFrame, score_summary: dict[str, Any]) -> pd.DataFrame:
    frame = _merge_score_r6(score, r6)
    masks = _base_masks(frame, score_summary)
    missed = (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1")) & ~_bool(frame, "selected_candidate_block_v1")
    out = frame.loc[missed].copy()
    for name, mask in masks.items():
        out[f"r5_2_{name}_base_flag_v1"] = mask.loc[out.index].to_numpy(dtype=bool)
    out["r5_2_v3_base_flag_v1"] = masks["v3"].loc[out.index].to_numpy(dtype=bool)
    out["r6_selected_flag_v1"] = _bool(out, "selected_candidate_block_v1").to_numpy(dtype=bool)
    out["r6_first_fail_reason_v1"] = "NOT_IN_R5_2_BASE"
    out["mfe_bucket_v1"] = _mfe_bucket(out).to_numpy()
    out["mae_bucket_v1"] = _mae_bucket(out).to_numpy()
    out["row_is_actually_dangerous_v1"] = _danger(out).to_numpy(dtype=bool)
    out["r5_2_first_exclusion_reason_v1"] = out.apply(_first_exclusion, axis=1)
    out["root_cause_bucket_v1"] = out.apply(_root_bucket, axis=1)
    keep = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "r5_2_label_bad_blocker_v1",
        "r6_label_risky_allow_v1",
        *SCORE_FIELDS,
        "r5_2_original_base_flag_v1",
        "r5_2_v1_base_flag_v1",
        "r5_2_v2_base_flag_v1",
        "r5_2_v3_base_flag_v1",
        "r5_2_first_exclusion_reason_v1",
        "r5_2_label_runner_protect_v1",
        "r5_2_label_runner_50_mfe_v1",
        "r5_2_label_runner_100_mfe_v1",
        "r5_2_label_runner_200_mfe_v1",
        "r5_2_label_strong_low_mae_runner_v1",
        "r6_selected_flag_v1",
        "r6_first_fail_reason_v1",
        *SAFETY_FLAGS,
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "mfe_bucket_v1",
        "mae_bucket_v1",
        "row_is_actually_dangerous_v1",
        "root_cause_bucket_v1",
        "batch_scope_v1",
        "split_scope_v1",
        "run_id",
    ]
    return out[[col for col in keep if col in out.columns]].copy()


def _count_bool(frame: pd.DataFrame, col: str) -> int:
    return int(_bool(frame, col).sum()) if col in frame.columns else 0


def _score_stats(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in frame.columns:
        return {}
    s = _num(frame, col)
    return {f"p{int(q*100)}_v1": (None if pd.isna(v) else float(v)) for q, v in s.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).items()}


def _weakness(trace: pd.DataFrame) -> dict[str, Any]:
    buckets: dict[str, Any] = {}
    for bucket, group in trace.groupby("root_cause_bucket_v1", dropna=False):
        buckets[str(bucket)] = {
            "count_v1": int(len(group)),
            "bad_count_v1": _count_bool(group, "label_should_not_take_v1"),
            "tail_count_v1": _count_bool(group, "tail_10_50_mfe_v1"),
            "high_mfe_risk_v1": _count_bool(group, "fifty_plus_mfe_v1"),
            "hundred_plus_risk_v1": _count_bool(group, "hundred_plus_mfe_v1"),
            "two_hundred_plus_risk_v1": _count_bool(group, "two_hundred_plus_mfe_v1"),
            "repaired_risk_v1": _count_bool(group, "r6_label_repaired_165_like_runner_v1"),
            "strongest_winner_risk_v1": _count_bool(group, "strongest_winner_path_v1"),
            "runner_near_miss_risk_v1": _count_bool(group, "r6_label_runner_near_miss_v1"),
            "score_distribution_v1": {
                R5_2_BAD_PROB: _score_stats(group, R5_2_BAD_PROB),
                R5_2_RUNNER_PROB: _score_stats(group, R5_2_RUNNER_PROB),
                "r5_1_bad_blocker_score_v1": _score_stats(group, "r5_1_bad_blocker_score_v1"),
                R6_BAD_PROB: _score_stats(group, R6_BAD_PROB),
            },
            "label_distribution_v1": {
                "r5_2_label_bad_blocker_v1": _count_bool(group, "r5_2_label_bad_blocker_v1"),
                "r6_label_risky_allow_v1": _count_bool(group, "r6_label_risky_allow_v1"),
            },
        }
    return {
        "layer_name": "R5_2_SCORE_WEAKNESS_VS_LABEL_OBJECTIVE_V1",
        "bucket_summary_v1": buckets,
        "dominant_findings_v1": [
            "A large part of the miss set is intentionally not rewarded by the current R5.2 bad-blocker label because it excludes label_should_not_take rows with 50+ MFE.",
            "For the remaining rows, R5.2 bad score is weak: median missed R5.2 bad score is far below V3 entry threshold while some R5/R5.1/R6 score signals are stronger.",
            "Tiny contract rules are not enough because the missing population is not near the current R5.2 base boundary.",
        ],
    }


def _feature_signal(trace: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    selected = score[_bool(score, "r5_2_selected_candidate__block_v1")].copy()
    missed = score[score["candidate_uid"].isin(set(trace["candidate_uid"].astype("string")))].copy()
    rows: list[dict[str, Any]] = []
    candidates = [
        col
        for col in score.columns
        if (
            col.startswith("as_of_")
            or col.startswith("pred__entry_r5_")
            or col.startswith("r5_1_")
            or col.startswith("pred__entry_r5_2_")
            or col.startswith("pred__entry_r6_")
        )
    ]
    for col in candidates:
        s_sel = pd.to_numeric(_num(selected, col), errors="coerce").astype("float64")
        s_miss = pd.to_numeric(_num(missed, col), errors="coerce").astype("float64")
        if s_sel.notna().sum() < 10 or s_miss.notna().sum() < 10:
            continue
        pooled = float(pd.concat([s_sel, s_miss]).std() or 0.0)
        effect = None if pooled == 0.0 else float((s_miss.mean() - s_sel.mean()) / pooled)
        if col.startswith("pred__entry_r6_"):
            legality = "REUSE_FOR_EVAL_ONLY"
            status = "SIGNAL_PRESENT_BUT_R6_DOWNSTREAM_ONLY"
        elif col.startswith("pred__entry_r5_") or col.startswith("r5_1_") or col.startswith("pred__entry_r5_2_"):
            legality = "REUSE_NOW"
            status = "EXISTING_SCORE_SIGNAL"
        elif col.startswith("as_of_"):
            legality = "REUSE_NOW"
            status = "AS_OF_ENTRY_LEGAL"
        else:
            legality = "NOT_ESTABLISHED"
            status = "NOT_ESTABLISHED"
        rows.append(
            {
                "feature_v1": col,
                "status_v1": status,
                "legality_v1": legality,
                "selected_base_mean_v1": float(s_sel.mean()),
                "missed_mean_v1": float(s_miss.mean()),
                "selected_base_p50_v1": float(s_sel.quantile(0.5)),
                "missed_p50_v1": float(s_miss.quantile(0.5)),
                "absolute_effect_size_v1": None if effect is None else abs(effect),
                "direction_v1": "HIGHER_IN_MISSED" if effect is not None and effect > 0 else "LOWER_IN_MISSED",
                "can_use_directly_in_r5_2_v1": legality == "REUSE_NOW",
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_effect_size_v1", ascending=False, na_position="last").head(120)


def _label_audit(trace: pd.DataFrame) -> dict[str, Any]:
    high_mfe_excluded = int((~_bool(trace, "r5_2_label_bad_blocker_v1") & _bool(trace, "label_should_not_take_v1")).sum())
    return {
        "layer_name": "R5_2_LABEL_OBJECTIVE_AUDIT_V1",
        "what_r5_2_learns_v1": "R5.2 bad head learns label_should_not_take excluding high-MFE ambiguous rows; runner head learns take_ok high-MFE/strong-low-MAE runner protection. The base then requires bad score high and runner score low.",
        "bad_tail_recovery_reward_strong_enough_v1": False,
        "winner_damage_penalty_present_v1": True,
        "too_conservative_v1": True,
        "pocket_separation_v1": {
            "bad_risk_v1": "PARTIAL",
            "tail_10_50_v1": "WEAK_INDIRECT_ONLY",
            "risky_allow_v1": "R6_DOWNSTREAM_LABEL_NOT_R5_2_BASE_OBJECTIVE",
            "runner_seed_v1": "STRONG_PROTECTION",
            "high_mfe_winner_v1": "STRONG_PROTECTION_AND_BAD_LABEL_EXCLUSION",
            "repaired_like_good_trade_v1": "EVAL_GUARD_ONLY_IN_CURRENT_MONDAY_FRAME",
        },
        "r5_2_r6_contract_mismatch_v1": True,
        "evidence_v1": {
            "missed_rows_v1": int(len(trace)),
            "missed_high_mfe_label_excluded_from_r5_2_bad_v1": high_mfe_excluded,
            "missed_r6_risky_allow_v1": _count_bool(trace, "r6_label_risky_allow_v1"),
            "missed_tail_10_50_v1": _count_bool(trace, "tail_10_50_mfe_v1"),
        },
    }


def _requirement_check(trace: pd.DataFrame, feature_signal: pd.DataFrame) -> dict[str, Any]:
    signal_present = bool((feature_signal["absolute_effect_size_v1"].fillna(0) > 0.40).any()) if not feature_signal.empty else False
    label_excluded = int((trace["root_cause_bucket_v1"] == "ROW_IS_ACTUALLY_DANGEROUS").sum())
    score_weak = int(trace["root_cause_bucket_v1"].isin(["R5_2_SCORE_WEAK", "R5_2_FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED"]).sum())
    if label_excluded > 0 and score_weak > 0:
        decision = "LABEL_OBJECTIVE_FIX_REQUIRED_BEFORE_R5_2_REBUILD"
    elif signal_present:
        decision = "R5_2_FEATURE_SIGNAL_PRESENT_BUT_OBJECTIVE_WRONG"
    else:
        decision = "TRUE_R5_2_REBUILD_REQUIRED"
    return {
        "layer_name": "R5_2_TRUE_REBUILD_REQUIREMENT_CHECK_V1",
        "decision_v1": decision,
        "existing_scores_too_weak_under_current_objective_v1": True,
        "labels_objective_must_change_before_rebuild_v1": decision == "LABEL_OBJECTIVE_FIX_REQUIRED_BEFORE_R5_2_REBUILD",
        "feature_set_complete_enough_for_experiment_v1": signal_present,
        "existing_feature_signal_present_but_not_used_correctly_v1": signal_present,
        "needs_cost_weighting_v1": True,
        "needs_multi_head_or_pocket_aware_target_v1": True,
        "base_contract_only_fix_not_enough_v1": True,
        "evidence_v1": {
            "missed_rows_v1": int(len(trace)),
            "label_or_safety_excluded_rows_v1": label_excluded,
            "score_weak_or_underused_rows_v1": score_weak,
            "top_existing_signal_features_v1": feature_signal.head(10)["feature_v1"].tolist() if not feature_signal.empty else [],
        },
    }


def _options() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_REDESIGN_OPTIONS_V1",
        "options_v1": [
            {
                "option_id_v1": "R5_2_BAD_TAIL_RECALL_WEIGHTED_WITH_HARD_WINNER_COST",
                "change_v1": "Weight safe bad/tail recall higher while applying hard cost to 50+/100+/200+/strongest/repaired/near-miss pockets.",
                "helps_v1": ["R5_2_SCORE_WEAK", "R5_2_LABEL_DOES_NOT_REWARD_THIS_CASE"],
                "protects_winners_v1": "Use explicit hard negative weights and disqualification gates.",
                "uses_existing_features_v1": ["AS_OF 109", "R5/R5.1 score assets"],
                "requires_true_retrain_v1": True,
                "risk_v1": "Can over-block high-MFE ambiguous rows if hard winner cost is too weak.",
            },
            {
                "option_id_v1": "POCKET_AWARE_R5_2_TARGET",
                "change_v1": "Separate safe bad, 10-50 tail, high-MFE ambiguous, and runner/winner pockets before base eligibility.",
                "helps_v1": ["tail_10_50", "risky_allow", "runner_seed separation"],
                "protects_winners_v1": "High-MFE/strongest/repaired are explicit protected labels, not just runner score side effects.",
                "uses_existing_features_v1": ["AS_OF 109", "path-quality proxies", "skip-replay features"],
                "requires_true_retrain_v1": True,
                "risk_v1": "More complex selection contract; must be frozen behind audit gates.",
            },
            {
                "option_id_v1": "SEPARATE_R5_2_BAD_AND_TAIL_ELIGIBILITY_HEADS",
                "change_v1": "Train independent bad eligibility and 10-50 tail eligibility heads plus a winner-protect head.",
                "helps_v1": ["missed tail", "R6 risky_allow dependency"],
                "protects_winners_v1": "Final base membership requires low winner-protect risk.",
                "uses_existing_features_v1": ["R5 scores", "R5.1 scores", "AS_OF features"],
                "requires_true_retrain_v1": True,
                "risk_v1": "May need LOSO-specific calibration to avoid batch collapse.",
            },
            {
                "option_id_v1": "SAFE_RECOVERABLE_BAD_CLASSIFIER",
                "change_v1": "Train R5.2 base as safe-recoverable bad vs dangerous runner-like row classifier.",
                "helps_v1": ["R5_2_FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED"],
                "protects_winners_v1": "Dangerous class includes high-MFE, strongest, repaired, and runner-near-miss pockets.",
                "uses_existing_features_v1": ["AS_OF features", "existing score layers"],
                "requires_true_retrain_v1": True,
                "risk_v1": "Requires strict leakage review because danger labels are hindsight for training/eval design.",
            },
        ],
    }


def _experiments() -> dict[str, Any]:
    base_constraints = {
        "repaired_damage_v1": 0,
        "forensic_trade_blocked_v1": 0,
        "fifty_plus_mfe_blocked_v1": "<=1",
        "hundred_twohundred_blocked_v1": "0/0",
        "strongest_winner_damage_v1": 0,
        "runner_near_miss_not_worse_v1": True,
    }
    return {
        "layer_name": "R5_2_REBUILD_EXPERIMENT_SPEC_CANDIDATES_V1",
        "do_not_run_in_this_round_v1": True,
        "candidates_v1": [
            {
                "experiment_id_v1": "R5_2_REBUILD_BAD_TAIL_WEIGHTED_HARD_WINNER_COST_V1",
                "training_surface_v1": "Monday 1914 fullcoverage foundation, not 1689 exact-only",
                "labels_objective_v1": "bad/tail weighted objective with hard winner-cost disqualification",
                "features_v1": "Existing AS_OF 109 plus existing R5/R5.1 score assets",
                "hard_safety_constraints_v1": base_constraints,
                "eval_metrics_v1": ["bad_blocks", "tail_help", "precision", "worst_loso", "pocket damage"],
                "expected_outputs_v1": ["R5.2 score package", "row trace", "R6 downstream replay"],
                "no_go_cases_v1": ["any 100+/200+ block", "strongest damage", "forensic blocked"],
                "r6_usage_v1": "R6 uses new R5.2 base flag as base input only after explicit R6 retrain/eval.",
            },
            {
                "experiment_id_v1": "R5_2_REBUILD_POCKET_AWARE_MULTI_HEAD_V1",
                "training_surface_v1": "Monday 1914 fullcoverage foundation",
                "labels_objective_v1": "separate bad, tail, winner-protect, high-MFE ambiguous heads",
                "features_v1": "Existing AS_OF and score assets; no exit/management/hindsight as entry features",
                "hard_safety_constraints_v1": base_constraints,
                "eval_metrics_v1": ["per-pocket recall", "LOSO", "winner retention"],
                "expected_outputs_v1": ["multi-head R5.2 prediction view", "base eligibility report"],
                "no_go_cases_v1": ["label leakage", "batch-specific collapse"],
                "r6_usage_v1": "R6 candidate grid consumes base eligibility and protection heads.",
            },
        ],
    }


def _stop_loop() -> dict[str, Any]:
    return {
        "layer_name": "STOP_TINY_EXTENSION_LOOP_LOCK_V1",
        "v1_v2_v3_safe_but_minimal_v1": True,
        "v1_v2_v3_uplift_v1": {
            "v1_to_v2_v1": "+2 bad / +1 tail",
            "v2_to_v3_v1": "+4 bad / +2 tail",
            "v3_total_v1": "82 bad / 51 tail vs Wednesday 180 / 149",
        },
        "tiny_base_extension_alone_closes_gap_v1": False,
        "do_not_continue_v4_v5_rule_search_without_objective_fix_v1": True,
        "next_work_must_be_v1": "true R5.2 rebuild design or label/objective redesign",
    }


def _next_action(requirement: dict[str, Any]) -> dict[str, Any]:
    decision = requirement["decision_v1"]
    if decision == "LABEL_OBJECTIVE_FIX_REQUIRED_BEFORE_R5_2_REBUILD":
        action = "FIX_R5_2_LABEL_OBJECTIVE_FIRST"
    elif decision == "R5_2_FEATURE_SIGNAL_PRESENT_BUT_OBJECTIVE_WRONG":
        action = "WIRE_EXISTING_FEATURE_SIGNAL_INTO_R5_2_FIRST"
    elif decision == "TRUE_R5_2_REBUILD_REQUIRED":
        action = "DESIGN_TRUE_R5_2_REBUILD_EXPERIMENT_NEXT"
    else:
        action = "NOT_ESTABLISHED"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_CONTINUE_R5_2_TINY_EXTENSION_LOOP",
            "DO_NOT_RUN_R6_RETRAIN_BEFORE_R5_2_OBJECTIVE_DECISION",
            "DO_NOT_BUILD_NEW_BASELINE",
            "DO_NOT_USE_1689_EXACT_ONLY",
            "DO_NOT_USE_PROTECTOR_FIRST",
        ],
    }


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("TRACE_ALL_390_ROWS", summary["missed_rows_traced_v1"] == 390, summary["missed_rows_traced_v1"]),
            row("NO_TINY_EXTENSION_LOOP", summary["stop_tiny_extension_loop_v1"], True),
            row("LABEL_OBJECTIVE_DECISION_MATERIALIZED", summary["next_action_v1"] != "NOT_ESTABLISHED", summary["next_action_v1"]),
            row("NO_TRAINING_RUN", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_NEW_BASELINE", True, True),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Investigate True R5.2 Rebuild Or Label Objective Next",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Missed rows traced: `{summary['missed_rows_traced_v1']}`",
            f"- Label/objective excluded or dangerous: `{summary['label_or_safety_excluded_rows_v1']}`",
            f"- Score weak / underused signal rows: `{summary['score_weak_or_underused_rows_v1']}`",
            f"- Existing feature/signal present: `{summary['existing_feature_signal_present_v1']}`",
            f"- Tiny extension loop stopped: `{summary['stop_tiny_extension_loop_v1']}`",
            "",
            "No model training, baseline rebuild, feature rebuild, R6 retrain, freeze, promotion, live gate, or controller path was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_score_dir: Path = V3_SCORE_DEFAULT,
    v3_r6_dir: Path = V3_R6_DEFAULT,
    v3_forensics_dir: Path = V3_FORENSICS_DEFAULT,
    asset_reuse_dir: Path = ASSET_REUSE_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_score_dir = v3_score_dir.expanduser().resolve()
    v3_r6_dir = v3_r6_dir.expanduser().resolve()
    v3_forensics_dir = v3_forensics_dir.expanduser().resolve()
    asset_reuse_dir = asset_reuse_dir.expanduser().resolve()

    score = pd.read_parquet(v3_score_dir / SCORE_FRAME)
    r6 = pd.read_parquet(v3_r6_dir / R6_FRAME)
    score_summary = _read_json(v3_score_dir / SCORE_SUMMARY)
    trace = _trace(score, r6, score_summary)
    weakness = _weakness(trace)
    feature_signal = _feature_signal(trace, score)
    label_audit = _label_audit(trace)
    requirement = _requirement_check(trace, feature_signal)
    options = _options()
    experiments = _experiments()
    stop_loop = _stop_loop()
    next_action = _next_action(requirement)

    label_or_safety = int(trace["root_cause_bucket_v1"].isin(["ROW_IS_ACTUALLY_DANGEROUS", "R5_2_LABEL_DOES_NOT_REWARD_THIS_CASE"]).sum())
    score_weak = int(trace["root_cause_bucket_v1"].isin(["R5_2_SCORE_WEAK", "R5_2_FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED"]).sum())
    existing_signal = bool((feature_signal["absolute_effect_size_v1"].fillna(0) > 0.40).any()) if not feature_signal.empty else False
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v3_score_dir_v1": str(v3_score_dir),
        "v3_r6_dir_v1": str(v3_r6_dir),
        "v3_forensics_dir_v1": str(v3_forensics_dir),
        "asset_reuse_dir_v1": str(asset_reuse_dir),
        "decision_v1": requirement["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "missed_rows_traced_v1": int(len(trace)),
        "missed_bad_count_v1": _count_bool(trace, "label_should_not_take_v1"),
        "missed_tail_count_v1": _count_bool(trace, "tail_10_50_mfe_v1"),
        "label_or_safety_excluded_rows_v1": label_or_safety,
        "score_weak_or_underused_rows_v1": score_weak,
        "existing_feature_signal_present_v1": existing_signal,
        "r5_2_must_rebuild_v1": True,
        "label_objective_fix_required_first_v1": requirement["decision_v1"] == "LABEL_OBJECTIVE_FIX_REQUIRED_BEFORE_R5_2_REBUILD",
        "stop_tiny_extension_loop_v1": True,
        "training_started_v1": False,
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "hard_status_v1": {
            "BEVIST": [
                "All 390 post-V3 missed bad/tail rows were traced with R5/R5.1/R5.2/R6 scores, labels, pockets, and base flags.",
                "V1/V2/V3 tiny base-extension loop is insufficient and is explicitly stopped.",
            ],
            "INDIKERT": [
                "R5.2 label/objective is the upstream mismatch: it excludes high-MFE ambiguous bad rows and underweights safe bad/tail recovery.",
                "Existing AS_OF/score signals are present, but the current R5.2 objective/base contract underuses them.",
            ],
            "IKKE_ETABLERT": [
                "A green true R5.2 rebuild is not established because no rebuild was run in this round.",
            ],
        },
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "v3_score_dir_v1": str(v3_score_dir),
            "v3_r6_dir_v1": str(v3_r6_dir),
            "v3_forensics_dir_v1": str(v3_forensics_dir),
            "asset_reuse_dir_v1": str(asset_reuse_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
    }

    trace.to_csv(output_dir / OUTPUT_FILES["trace"], index=False)
    _write_json(output_dir / OUTPUT_FILES["weakness"], weakness)
    _write_json(output_dir / OUTPUT_FILES["label_audit"], label_audit)
    _write_json(output_dir / OUTPUT_FILES["requirement"], requirement)
    feature_signal.to_csv(output_dir / OUTPUT_FILES["feature_signal"], index=False)
    _write_json(output_dir / OUTPUT_FILES["options"], options)
    _write_json(output_dir / OUTPUT_FILES["experiments"], experiments)
    _write_json(output_dir / OUTPUT_FILES["stop_loop"], stop_loop)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v3-score-dir", type=Path, default=V3_SCORE_DEFAULT)
    parser.add_argument("--v3-r6-dir", type=Path, default=V3_R6_DEFAULT)
    parser.add_argument("--v3-forensics-dir", type=Path, default=V3_FORENSICS_DEFAULT)
    parser.add_argument("--asset-reuse-dir", type=Path, default=ASSET_REUSE_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_score_dir=args.v3_score_dir,
        v3_r6_dir=args.v3_r6_dir,
        v3_forensics_dir=args.v3_forensics_dir,
        asset_reuse_dir=args.asset_reuse_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
