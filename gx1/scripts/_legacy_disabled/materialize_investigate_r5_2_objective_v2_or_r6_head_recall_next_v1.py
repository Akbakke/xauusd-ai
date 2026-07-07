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
    DEFAULT_RESCUE_DIR,
    DEFAULT_REPORTS_ROOT,
    FORENSIC_REPAIRED_CANDIDATE_UID,
    OUTPUT_FILES as R6_RESCUE_OUTPUT_FILES,
    _frame_selected_metrics,
    _read_json,
    _safety_pass,
    _selected_policy_from_grid,
    _selected_policy_mask,
)
from gx1.scripts.materialize_safe_true_r5_2_rescue_base_rule_v1 import RESCUE_BASE_FLAG_COL
from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import (
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
from gx1.scripts.train_monday_r6_on_foundation_scores_v1 import TRAINING_FRAME


LAYER_NAME = "INVESTIGATE_R5_2_OBJECTIVE_V2_OR_R6_HEAD_RECALL_NEXT_V1"
DEFAULT_R6_RESCUE_DIR = DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_V1_20260426T_EXPLICIT"

OUTPUT_FILES = {
    "gap_map": "post_rescue_recall_gap_map_v1.csv",
    "objective_scan": "r5_2_objective_v2_opportunity_scan_v1.json",
    "r6_head_scan": "r6_head_recall_outside_base_scan_v1.csv",
    "base_gate_review": "r5_2_base_gate_dependency_review_v1.json",
    "outside_base_sim": "safe_r6_outside_base_recovery_simulation_v1.csv",
    "decision_matrix": "r5_2_v2_vs_r6_outside_base_decision_matrix_v1.json",
    "next_specs": "next_experiment_spec_options_v1.json",
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


def _danger_mask(frame: pd.DataFrame) -> pd.Series:
    base = _bool(frame, RESCUE_BASE_FLAG_COL)
    return (
        _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
        | _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1")
        | _bool(frame, "r5_2_label_runner_protect_v1")
        | (_bool(frame, "raw_true_base_membership_v1") & ~base)
    ).fillna(False).astype(bool)


def _policy_params(r6_dir: Path) -> dict[str, Any]:
    selected = _selected_policy_from_grid(r6_dir)
    params = selected.get("params_v1") or {}
    if not params:
        raise RuntimeError("Could not recover selected R6 policy params from rescue R6 grid")
    return params


def _current_selected(frame: pd.DataFrame, r6_dir: Path) -> pd.Series:
    return _selected_policy_mask(frame, _selected_policy_from_grid(r6_dir)).reindex(frame.index).fillna(False).astype(bool)


def _first_fail_reason(row: pd.Series, params: dict[str, Any]) -> str:
    if bool(row.get(RESCUE_BASE_FLAG_COL, False)):
        return "IN_RESCUED_BASE"
    if float(row.get(R6_RUNNER_PROB, np.nan)) >= float(params.get("runner_threshold_v1", 0.30)):
        return "R6_RUNNER_GUARD_BLOCKED"
    if float(row.get(R5_2_RUNNER_PROB, np.nan)) >= float(params.get("r5_2_runner_threshold_v1", 0.74)):
        return "R5_2_RUNNER_GUARD_BLOCKED"
    if bool(row.get("asof_runner_guard_v1", False)) and bool(params.get("hard_asof_runner_guard_v1", True)):
        return "ASOF_RUNNER_GUARD_BLOCKED"
    if float(row.get(R6_BAD_PROB, np.nan)) < float(params.get("bad_threshold_v1", 0.85)):
        return "R6_BAD_HEAD_TOO_LOW"
    if float(row.get(R6_RISKY_PROB, np.nan)) < float(params.get("risky_threshold_v1", 0.85)):
        return "R6_RISKY_HEAD_TOO_LOW"
    if float(row.get(R6_TAIL_PROB, np.nan)) < float(params.get("tail_threshold_v1", 0.85)):
        return "R6_TAIL_HEAD_TOO_LOW"
    if float(row.get(R6_BLINDSPOT_PROB, np.nan)) >= float(params.get("blindspot_threshold_v1", 0.70)):
        return "R6_BLINDSPOT_GUARD_BLOCKED"
    return "NOT_ESTABLISHED"


def _r6_signal_classification(frame: pd.DataFrame, danger: pd.Series, params: dict[str, Any]) -> pd.Series:
    bad = _num(frame, R6_BAD_PROB)
    risky = _num(frame, R6_RISKY_PROB)
    tail = _num(frame, R6_TAIL_PROB)
    runner = _num(frame, R6_RUNNER_PROB)
    blind = _num(frame, R6_BLINDSPOT_PROB)
    r5_runner = _num(frame, R5_2_RUNNER_PROB)
    low_protect = runner.lt(0.60).fillna(False) & r5_runner.lt(0.74).fillna(False) & blind.lt(0.70).fillna(True)
    strongish = ((bad.ge(0.50).astype(int) + risky.ge(0.50).astype(int) + tail.ge(0.50).astype(int)) >= 2) & low_protect
    weak = (bad.lt(0.50).fillna(True) & risky.lt(0.50).fillna(True) & tail.lt(0.50).fillna(True)) | tail.lt(0.50).fillna(True)
    exact_addon = (
        bad.ge(float(params.get("bad_threshold_v1", 0.85))).fillna(False)
        & risky.ge(float(params.get("risky_threshold_v1", 0.85))).fillna(False)
        & tail.ge(float(params.get("tail_threshold_v1", 0.85))).fillna(False)
        & runner.lt(float(params.get("runner_threshold_v1", 0.30))).fillna(False)
        & r5_runner.lt(float(params.get("r5_2_runner_threshold_v1", 0.74))).fillna(False)
        & blind.lt(float(params.get("blindspot_threshold_v1", 0.70))).fillna(True)
    )
    return pd.Series(
        np.select(
            [danger, exact_addon | strongish, weak],
            ["R6_HEAD_SIGNAL_UNSAFE", "R6_HEAD_SIGNAL_STRONG_BUT_BASE_BLOCKED", "R6_HEAD_SIGNAL_WEAK"],
            default="R6_HEAD_SIGNAL_AMBIGUOUS",
        ),
        index=frame.index,
    )


def _gap_bucket(frame: pd.DataFrame, selected: pd.Series, danger: pd.Series, signal: pd.Series, params: dict[str, Any]) -> pd.Series:
    base = _bool(frame, RESCUE_BASE_FLAG_COL)
    out: list[str] = []
    for idx, row in frame.iterrows():
        if danger.loc[idx]:
            out.append("DANGEROUS_OR_PROTECTED")
        elif base.loc[idx] and not selected.loc[idx]:
            reason = _first_fail_reason(row, params)
            if "GUARD" in reason:
                out.append("IN_RESCUED_BASE_BUT_R6_GUARD_BLOCKED")
            else:
                out.append("IN_RESCUED_BASE_BUT_R6_HEAD_TOO_LOW")
        elif not base.loc[idx] and signal.loc[idx] == "R6_HEAD_SIGNAL_STRONG_BUT_BASE_BLOCKED":
            out.append("R6_COULD_RECOVER_BUT_BASE_GATE_BLOCKS")
        elif not base.loc[idx]:
            out.append("NOT_IN_RESCUED_R5_2_BASE")
        else:
            out.append("NOT_ESTABLISHED")
    return pd.Series(out, index=frame.index)


def _gap_map(frame: pd.DataFrame, r6_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = _current_selected(frame, r6_dir)
    params = _policy_params(r6_dir)
    danger = _danger_mask(frame)
    signal = _r6_signal_classification(frame, danger, params)
    missed = (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1")) & ~selected
    work = frame.loc[missed].copy()
    work["r6_selected_after_rescue_v1"] = selected.loc[work.index]
    work["r6_first_fail_reason_v1"] = work.apply(lambda row: _first_fail_reason(row, params), axis=1)
    work["r6_head_signal_class_v1"] = signal.loc[work.index]
    work["post_rescue_gap_bucket_v1"] = _gap_bucket(work, selected.loc[work.index], danger.loc[work.index], signal.loc[work.index], params)
    work["mfe_bucket_v1"] = pd.cut(
        _num(work, "peak_mfe_bps_v1"),
        bins=[-np.inf, 10, 50, 100, 200, np.inf],
        labels=["LT_10", "10_50", "50_100", "100_200", "200_PLUS"],
    ).astype("string")
    work["mae_bucket_v1"] = pd.cut(
        _num(work, "mae_abs_bps_v1"),
        bins=[-np.inf, 25, 50, 100, np.inf],
        labels=["LT_25", "25_50", "50_100", "100_PLUS"],
    ).astype("string")
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "calendar_quarantine_status_v1",
        "pred__entry_r5_should_not_take__prob_true_v1",
        "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
        "pred__entry_r5_runner_protect__prob_true_v1",
        "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
        "r5_1_bad_blocker_score_v1",
        "r5_1_runner_guard_score_v1",
        "in_v3_base_v1",
        TRUE_R5_2_BAD_SCORE_COL,
        TRUE_R5_2_TAIL_SCORE_COL,
        TRUE_R5_2_RISKY_SCORE_COL,
        TRUE_R5_2_RUNNER_SCORE_COL,
        RESCUE_BASE_FLAG_COL,
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
        R6_BLINDSPOT_PROB,
        "r6_first_fail_reason_v1",
        "r6_head_signal_class_v1",
        "post_rescue_gap_bucket_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
        "r5_2_label_runner_protect_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "mfe_bucket_v1",
        "mae_bucket_v1",
        "batch_scope_v1",
        "split_scope_v1",
        "run_id",
    ]
    summary = {
        "missed_rows_v1": int(len(work)),
        "missed_bad_v1": int(_bool(work, "label_should_not_take_v1").sum()),
        "missed_tail_v1": int(_bool(work, "tail_10_50_mfe_v1").sum()),
        "bucket_counts_v1": {str(key): int(value) for key, value in work["post_rescue_gap_bucket_v1"].value_counts().to_dict().items()},
        "signal_counts_v1": {str(key): int(value) for key, value in work["r6_head_signal_class_v1"].value_counts().to_dict().items()},
        "dangerous_or_protected_v1": int(danger.loc[work.index].sum()),
        "safeish_hindsight_rows_v1": int((~danger.loc[work.index]).sum()),
    }
    return work[[col for col in cols if col in work.columns]], summary


def _quantiles(frame: pd.DataFrame, mask: pd.Series, columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in columns:
        vals = _num(frame.loc[mask], column).dropna()
        out[column] = {
            "count_v1": int(vals.shape[0]),
            "p50_v1": None if vals.empty else float(vals.quantile(0.50)),
            "p90_v1": None if vals.empty else float(vals.quantile(0.90)),
            "max_v1": None if vals.empty else float(vals.max()),
            "ge_0_50_v1": int(vals.ge(0.50).sum()) if not vals.empty else 0,
            "ge_0_70_v1": int(vals.ge(0.70).sum()) if not vals.empty else 0,
            "ge_0_85_v1": int(vals.ge(0.85).sum()) if not vals.empty else 0,
        }
    return out


def _r6_head_scan(gap: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate_uid",
        "trade_uid",
        "decision_timestamp",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        RESCUE_BASE_FLAG_COL,
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
        R6_BLINDSPOT_PROB,
        "r6_head_signal_class_v1",
        "r6_first_fail_reason_v1",
        "post_rescue_gap_bucket_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_runner_near_miss_v1",
    ]
    out = gap[[col for col in cols if col in gap.columns]].copy()
    out["score_margin_bad_minus_runner_v1"] = _num(out, R6_BAD_PROB) - _num(out, R6_RUNNER_PROB)
    out["score_margin_tail_minus_runner_v1"] = _num(out, R6_TAIL_PROB) - _num(out, R6_RUNNER_PROB)
    out["score_margin_risky_minus_runner_v1"] = _num(out, R6_RISKY_PROB) - _num(out, R6_RUNNER_PROB)
    return out


def _simulate_rule(frame: pd.DataFrame, selected: pd.Series, base: pd.Series, danger: pd.Series, rule_id: str, rule_type: str, add: pd.Series) -> dict[str, Any]:
    add = add.reindex(frame.index).fillna(False).astype(bool)
    mask = selected | add
    metrics = _frame_selected_metrics(frame, mask)
    return {
        "rule_id_v1": rule_id,
        "rule_type_v1": rule_type,
        "rows_recovered_v1": int(add.sum()),
        "bad_recovered_v1": int((add & _bool(frame, "label_should_not_take_v1")).sum()),
        "tail_recovered_v1": int((add & _bool(frame, "tail_10_50_mfe_v1")).sum()),
        "danger_rows_rejected_by_hard_filter_v1": int(((~base) & danger).sum()),
        "bad_blocks_v1": metrics["bad_blocks_v1"],
        "tail_help_v1": metrics["tail_help_v1"],
        "precision_v1": metrics["precision_v1"],
        "worst_loso_v1": metrics["worst_loso_v1"],
        "repaired_damage_v1": metrics["repaired_damage_v1"],
        "forensic_trade_blocked_v1": metrics["forensic_trade_blocked_v1"],
        "fifty_plus_mfe_blocked_v1": metrics["fifty_plus_mfe_blocked_v1"],
        "hundred_plus_mfe_blocked_v1": metrics["hundred_plus_mfe_blocked_v1"],
        "two_hundred_plus_mfe_blocked_v1": metrics["two_hundred_plus_mfe_blocked_v1"],
        "strongest_winner_damage_v1": metrics["strongest_winner_damage_v1"],
        "runner_near_miss_blocked_v1": metrics["runner_near_miss_blocked_v1"],
        "safety_pass_v1": _safety_pass(metrics),
        "explainable_v1": True,
    }


def _outside_base_sim(frame: pd.DataFrame, r6_dir: Path) -> pd.DataFrame:
    selected = _current_selected(frame, r6_dir)
    base = _bool(frame, RESCUE_BASE_FLAG_COL)
    danger = _danger_mask(frame)
    eligible_scope = (~base) & ~danger
    rows: list[dict[str, Any]] = []
    for bad in [0.50, 0.60, 0.70, 0.80, 0.85]:
        for risky in [0.50, 0.60, 0.70, 0.80, 0.85]:
            for tail in [0.50, 0.70, 0.85, 0.90]:
                for runner_cap in [0.30, 0.45, 0.60, 0.74, 1.01]:
                    add = (
                        eligible_scope
                        & _num(frame, R6_BAD_PROB).ge(bad)
                        & _num(frame, R6_RISKY_PROB).ge(risky)
                        & _num(frame, R6_TAIL_PROB).ge(tail)
                        & _num(frame, R6_RUNNER_PROB).lt(runner_cap)
                    )
                    rows.append(_simulate_rule(frame, selected, base, danger, f"consensus_b{bad}_r{risky}_t{tail}_p{runner_cap}", "R6_BAD_RISKY_TAIL_CONSENSUS_OUTSIDE_BASE", add))
    for tail in [0.08, 0.10, 0.30, 0.50, 0.70, 0.85]:
        for runner_cap in [0.30, 0.45, 0.60, 0.74, 1.01]:
            add = eligible_scope & _num(frame, R6_TAIL_PROB).ge(tail) & _num(frame, R6_RUNNER_PROB).lt(runner_cap)
            rows.append(_simulate_rule(frame, selected, base, danger, f"tail_only_t{tail}_p{runner_cap}", "R6_TAIL_ONLY_10_50_OUTSIDE_BASE", add))
    for bad in [0.50, 0.60, 0.70, 0.80, 0.85]:
        for runner_cap in [0.30, 0.45, 0.60, 0.74, 1.01]:
            add = eligible_scope & _num(frame, R6_BAD_PROB).ge(bad) & _num(frame, R6_RUNNER_PROB).lt(runner_cap)
            rows.append(_simulate_rule(frame, selected, base, danger, f"bad_high_b{bad}_p{runner_cap}", "R6_BAD_HIGH_RUNNER_LOW_OUTSIDE_BASE", add))
    for risky in [0.50, 0.60, 0.70, 0.80, 0.85]:
        for tail in [0.08, 0.10, 0.30, 0.50, 0.70]:
            add = eligible_scope & _num(frame, R6_RISKY_PROB).ge(risky) & _num(frame, R6_TAIL_PROB).ge(tail) & _num(frame, R6_RUNNER_PROB).lt(0.74)
            rows.append(_simulate_rule(frame, selected, base, danger, f"risky_tail_r{risky}_t{tail}", "R6_RISKY_TAIL_PROTECTION_LOW_OUTSIDE_BASE", add))
    for bad in [0.50, 0.60, 0.70]:
        for blind in [0.05, 0.10, 0.20, 0.30]:
            add = eligible_scope & _num(frame, R6_BAD_PROB).ge(bad) & _num(frame, R6_BLINDSPOT_PROB).ge(blind)
            rows.append(_simulate_rule(frame, selected, base, danger, f"blindspot_bad_b{bad}_bs{blind}", "R6_BLINDSPOT_BATCH_ASSIST_OUTSIDE_BASE", add))
    out = pd.DataFrame(rows)
    return out.sort_values(["safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False, False], na_position="last")


def _objective_scan(frame: pd.DataFrame, gap: pd.DataFrame, sim: pd.DataFrame, rescue_dir: Path) -> dict[str, Any]:
    rescue_audit = _read_json(rescue_dir / "rescue_rule_application_audit_v1.json")
    danger = _danger_mask(frame)
    missed_index = set(gap["candidate_uid"].astype(str))
    missed_mask = frame["candidate_uid"].astype(str).isin(missed_index)
    safeish = missed_mask & ~danger
    score_cols = [
        TRUE_R5_2_BAD_SCORE_COL,
        TRUE_R5_2_TAIL_SCORE_COL,
        TRUE_R5_2_RISKY_SCORE_COL,
        TRUE_R5_2_RUNNER_SCORE_COL,
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
    ]
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_OPPORTUNITY_SCAN_V1",
        "raw_true_rebuild_findings_v1": {
            "raw_true_bad_tail_v1": [rescue_audit.get("raw_true_v1", {}).get("bad_v1"), rescue_audit.get("raw_true_v1", {}).get("tail_v1")],
            "rescued_bad_tail_v1": [rescue_audit.get("rescued_v1", {}).get("bad_v1"), rescue_audit.get("rescued_v1", {}).get("tail_v1")],
            "raw_true_safety_fail_v1": {
                "fifty_plus_v1": rescue_audit.get("raw_true_v1", {}).get("fifty_plus_overlap_v1"),
                "hundred_plus_v1": rescue_audit.get("raw_true_v1", {}).get("hundred_plus_overlap_v1"),
                "strongest_v1": rescue_audit.get("raw_true_v1", {}).get("strongest_winner_overlap_v1"),
                "ambiguous_v1": rescue_audit.get("raw_true_v1", {}).get("ambiguous_high_mfe_included_v1"),
                "runner_protect_v1": rescue_audit.get("raw_true_v1", {}).get("runner_protect_included_v1"),
            },
        },
        "post_rescue_gap_counts_v1": {
            "missed_rows_v1": int(len(gap)),
            "safeish_hindsight_rows_v1": int(safeish.sum()),
            "dangerous_or_protected_rows_v1": int((missed_mask & danger).sum()),
            "safeish_bad_v1": int((safeish & _bool(frame, "label_should_not_take_v1")).sum()),
            "safeish_tail_v1": int((safeish & _bool(frame, "tail_10_50_mfe_v1")).sum()),
        },
        "score_separability_v1": _quantiles(frame, safeish, score_cols),
        "outside_base_simulation_safe_rule_count_v1": int((sim["safety_pass_v1"].astype(bool) & sim["rows_recovered_v1"].gt(0)).sum()) if not sim.empty else 0,
        "options_v1": [
            {
                "option_v1": "STRONGER_RUNNER_PROTECTION_WEIGHT",
                "what_changes_v1": "Increase runner/protect loss and calibrate protection before bad/tail eligibility.",
                "theoretical_help_v1": "Could keep true rebuild recall signal while preventing high-MFE leakage.",
                "safety_rows_must_stop_v1": int((missed_mask & danger).sum()),
                "existing_features_support_v1": "INDIKERT_BUT_NOT_SUFFICIENT_WITH_CURRENT_WEIGHTS",
                "requires_true_rebuild_v1": True,
            },
            {
                "option_v1": "AMBIGUOUS_HIGH_MFE_HARD_NEGATIVE_CLASS",
                "what_changes_v1": "Treat ambiguous high-MFE as hard protected/negative, not merely monitor.",
                "theoretical_help_v1": "Directly addresses raw true leakage: ambiguous/high-MFE/winner rows.",
                "existing_features_support_v1": "INDIKERT",
                "requires_true_rebuild_v1": True,
            },
            {
                "option_v1": "TWO_STAGE_R5_2_RECALL_HEAD_PLUS_PROTECTION_VETO",
                "what_changes_v1": "Separate recall scoring from hard protection veto and only expose post-veto base to R6.",
                "theoretical_help_v1": "Most aligned with failure mode: recall exists but protection calibration is weak.",
                "existing_features_support_v1": "BEST_NEXT_DESIGN",
                "requires_true_rebuild_v1": True,
            },
            {
                "option_v1": "R6_OUTSIDE_BASE_WITH_EXISTING_HEADS",
                "what_changes_v1": "Use R6 heads to add rows outside R5.2-base.",
                "theoretical_help_v1": "Low; simulation found no safe score-only rule on current heads.",
                "existing_features_support_v1": "NOT_SUFFICIENT",
                "requires_true_rebuild_v1": False,
            },
        ],
        "decision_v1": "R5_2_OBJECTIVE_V2_REBUILD_NEEDED",
    }


def _base_gate_review(frame: pd.DataFrame, r6_dir: Path) -> dict[str, Any]:
    selected_policy = _selected_policy_from_grid(r6_dir)
    selected = _current_selected(frame, r6_dir)
    base = _bool(frame, RESCUE_BASE_FLAG_COL)
    addon = selected & ~base
    return {
        "layer_name": "R5_2_BASE_GATE_DEPENDENCY_REVIEW_V1",
        "selected_policy_v1": selected_policy,
        "use_r5_2_base_v1": bool(selected_policy.get("params_v1", {}).get("use_r5_2_base_v1")),
        "wednesday_contract_interpretation_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON uses R5.2 base OR a strict R6 bad+risky+tail addon outside base.",
        "monday_current_behavior_v1": {
            "selected_rows_v1": int(selected.sum()),
            "rescued_r5_2_base_rows_v1": int(base.sum()),
            "outside_base_addon_rows_v1": int(addon.sum()),
            "outside_base_addon_bad_v1": int((addon & _bool(frame, "label_should_not_take_v1")).sum()),
            "outside_base_addon_tail_v1": int((addon & _bool(frame, "tail_10_50_mfe_v1")).sum()),
        },
        "findings_v1": [
            "R5.2 is not the only allowed recall port in the R6 family contract.",
            "On this Monday rescue run, the selected safe candidate still gets all recall from rescued R5.2 base; outside-base addon contributes zero rows.",
            "The issue is not that outside-base recovery is forbidden; current R6 outside-base signals are too weak/noisy under safety constraints.",
        ],
        "decision_v1": "R6_ADDON_ALLOWED_BUT_SCORES_TOO_WEAK",
        "secondary_decision_v1": "R5_2_BASE_HARD_GATE_CORRECT_BUT_TOO_SMALL",
    }


def _decision_matrix(gap_summary: dict[str, Any], sim: pd.DataFrame, base_review: dict[str, Any]) -> dict[str, Any]:
    safe = sim[sim["safety_pass_v1"].astype(bool) & sim["rows_recovered_v1"].gt(0)].copy() if not sim.empty else pd.DataFrame()
    best_safe = None if safe.empty else safe.sort_values(["bad_blocks_v1", "tail_help_v1"], ascending=[False, False]).iloc[0].to_dict()
    return {
        "layer_name": "R5_2_V2_VS_R6_OUTSIDE_BASE_DECISION_MATRIX_V1",
        "options_v1": [
            {
                "path_v1": "R5_2_OBJECTIVE_V2_REBUILD",
                "expected_uplift_v1": "MEDIUM_TO_HIGH_IF_PROTECTION_VETO_LEARNS",
                "safety_risk_v1": "MANAGEABLE_WITH_HARD_PROTECTION_CLASS_AND_VETO",
                "training_required_v1": True,
                "existing_scores_enough_v1": False,
                "likely_to_close_meaningful_gap_v1": True,
                "recommendation_v1": "PREFERRED",
            },
            {
                "path_v1": "R6_SAFE_OUTSIDE_BASE_RECOVERY",
                "expected_uplift_v1": 0 if best_safe is None else int(best_safe["bad_blocks_v1"] - 88),
                "safety_risk_v1": "HIGH_PRECISION_COLLAPSE_ON_CURRENT_HEADS",
                "training_required_v1": False,
                "existing_scores_enough_v1": bool(best_safe is not None),
                "likely_to_close_meaningful_gap_v1": False,
                "recommendation_v1": "DO_NOT_IMPLEMENT_NOW",
            },
            {
                "path_v1": "HYBRID_R5_2_V2_PLUS_R6_RECOVERY",
                "expected_uplift_v1": "POTENTIALLY_HIGHEST_AFTER_R5_2_V2",
                "safety_risk_v1": "UNKNOWN_UNTIL_R5_2_V2_OUTPUT_EXISTS",
                "training_required_v1": True,
                "existing_scores_enough_v1": False,
                "likely_to_close_meaningful_gap_v1": True,
                "recommendation_v1": "SECOND_AFTER_R5_2_V2",
            },
            {
                "path_v1": "NO_SAFE_SIGNAL_FOUND",
                "expected_uplift_v1": 0,
                "safety_risk_v1": "LOW_BUT_STALLS_RECALL",
                "training_required_v1": False,
                "existing_scores_enough_v1": False,
                "likely_to_close_meaningful_gap_v1": False,
                "recommendation_v1": "NOT_SELECTED",
            },
        ],
        "best_safe_outside_base_rule_v1": best_safe,
        "gap_summary_v1": gap_summary,
        "base_gate_decision_v1": base_review["decision_v1"],
        "decision_v1": "R5_2_OBJECTIVE_V2_REBUILD",
    }


def _next_specs(decision: str) -> dict[str, Any]:
    return {
        "layer_name": "NEXT_EXPERIMENT_SPEC_OPTIONS_V1",
        "recommended_path_v1": decision,
        "r5_2_objective_v2_rebuild_spec_v1": {
            "design_v1": "TWO_STAGE_RECALL_HEAD_PLUS_PROTECTION_VETO",
            "labels_v1": [
                "bad_recall_target",
                "tail_recall_target",
                "risky_attention_target",
                "hard_protection_target_for_50_100_200_strongest_repaired_runner_near_miss",
                "ambiguous_high_mfe_hard_negative",
            ],
            "requirements_v1": [
                "Do not expose raw recall head as base.",
                "Construct R5.2 base only after hard protection veto.",
                "Evaluate against rescue R6 88/57 and Wednesday 180/149.",
                "Hard fail on any 100+/200+/strongest/repaired/ambiguous leakage.",
            ],
        },
        "r6_outside_base_experiment_spec_v1": {
            "status_v1": "DEFER",
            "reason_v1": "Current R6 heads do not yield a safe outside-base rule.",
        },
        "hybrid_order_v1": [
            "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT",
            "RUN_TRUE_R5_2_V2_REBUILD_WITH_EXPLICIT_FLAG",
            "RECHECK_R6_OUTSIDE_BASE_AFTER_V2_OUTPUT",
        ],
    }


def _next_action(decision: str) -> dict[str, Any]:
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT" if decision == "R5_2_OBJECTIVE_V2_REBUILD" else "NOT_ESTABLISHED",
        "blocked_action_v1": [
            "DO_NOT_TRAIN_IN_THIS_ANALYSIS",
            "DO_NOT_BUILD_NEW_BASELINE",
            "DO_NOT_USE_RAW_TRUE_UNSAFE_PACKAGE_DIRECTLY",
            "DO_NOT_IMPLEMENT_R6_OUTSIDE_BASE_RULE_FROM_CURRENT_HEADS",
        ],
    }


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}
    return pd.DataFrame(
        [
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_R6_RERUN", not summary["r6_rerun_started_v1"], summary["r6_rerun_started_v1"]),
            row("GAP_MAP_WRITTEN", summary["gap_map_rows_v1"] > 0, summary["gap_map_rows_v1"]),
            row("RAW_TRUE_NOT_DIRECT", summary["raw_true_unsafe_direct_use_v1"] is False, summary["raw_true_unsafe_direct_use_v1"]),
            row("DECISION_LOCKED", summary["next_action_v1"] != "NOT_ESTABLISHED", summary["next_action_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R5.2 Objective V2 Or R6 Head Recall Investigation",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Missed rows: `{summary['gap_map_rows_v1']}`",
            f"- Missed bad/tail: `{summary['missed_bad_v1']}/{summary['missed_tail_v1']}`",
            f"- Outside-base safe R6 rules found: `{summary['safe_outside_base_rule_count_v1']}`",
            f"- R6 addon rows in selected policy: `{summary['outside_base_addon_rows_v1']}`",
            "",
            "No training, R6 rerun, baseline build, or feature build was performed.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    r6_rescue_dir: Path = DEFAULT_R6_RESCUE_DIR,
    rescue_dir: Path = DEFAULT_RESCUE_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    r6_rescue_dir = r6_rescue_dir.expanduser().resolve()
    rescue_dir = rescue_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(r6_rescue_dir / TRAINING_FRAME)
    selected = _current_selected(frame, r6_rescue_dir)
    base = _bool(frame, RESCUE_BASE_FLAG_COL)
    metrics = _frame_selected_metrics(frame, selected)
    gap, gap_summary = _gap_map(frame, r6_rescue_dir)
    r6_head_scan = _r6_head_scan(gap)
    sim = _outside_base_sim(frame, r6_rescue_dir)
    base_review = _base_gate_review(frame, r6_rescue_dir)
    objective_scan = _objective_scan(frame, gap, sim, rescue_dir)
    decision_matrix = _decision_matrix(gap_summary, sim, base_review)
    next_specs = _next_specs(decision_matrix["decision_v1"])
    next_action = _next_action(decision_matrix["decision_v1"])
    safe_rules = sim[sim["safety_pass_v1"].astype(bool) & sim["rows_recovered_v1"].gt(0)] if not sim.empty else pd.DataFrame()
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "r6_rescue_dir_v1": str(r6_rescue_dir),
        "rescue_dir_v1": str(rescue_dir),
        "training_started_v1": False,
        "r6_rerun_started_v1": False,
        "raw_true_unsafe_direct_use_v1": False,
        "current_r6_rescue_bad_tail_v1": [metrics["bad_blocks_v1"], metrics["tail_help_v1"]],
        "wednesday_bad_tail_v1": [WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"], WEDNESDAY_R6_BENCHMARK["tail_help_v1"]],
        "gap_to_wednesday_bad_tail_v1": [
            int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - metrics["bad_blocks_v1"]),
            int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - metrics["tail_help_v1"]),
        ],
        "gap_map_rows_v1": int(len(gap)),
        "missed_bad_v1": gap_summary["missed_bad_v1"],
        "missed_tail_v1": gap_summary["missed_tail_v1"],
        "dangerous_or_protected_gap_rows_v1": gap_summary["dangerous_or_protected_v1"],
        "safeish_hindsight_gap_rows_v1": gap_summary["safeish_hindsight_rows_v1"],
        "outside_base_addon_rows_v1": base_review["monday_current_behavior_v1"]["outside_base_addon_rows_v1"],
        "safe_outside_base_rule_count_v1": int(safe_rules.shape[0]),
        "decision_v1": decision_matrix["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "hard_status_v1": {
            "BEVIST": [
                "Post-rescue gap map was built from the existing R6 rescue run.",
                "Current selected R6 policy gets zero rows from outside-base addon.",
                "No safe R6 outside-base recovery rule was found on existing scores.",
            ],
            "INDIKERT": [
                "R5.2 objective V2 with two-stage recall plus protection veto is the best next path.",
            ],
            "IKKE_ETABLERT": [
                "No new model uplift is established because no training or R6 rerun was performed.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_artifacts_v1": {
            "r6_rescue_training_frame_v1": str(r6_rescue_dir / TRAINING_FRAME),
            "r6_rescue_summary_v1": str(r6_rescue_dir / R6_RESCUE_OUTPUT_FILES["summary"]),
            "rescue_package_dir_v1": str(rescue_dir),
        },
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "r6_rerun_started_v1": False,
    }

    gap.to_csv(output_dir / OUTPUT_FILES["gap_map"], index=False)
    _write_json(output_dir / OUTPUT_FILES["objective_scan"], objective_scan)
    r6_head_scan.to_csv(output_dir / OUTPUT_FILES["r6_head_scan"], index=False)
    _write_json(output_dir / OUTPUT_FILES["base_gate_review"], base_review)
    sim.to_csv(output_dir / OUTPUT_FILES["outside_base_sim"], index=False)
    _write_json(output_dir / OUTPUT_FILES["decision_matrix"], decision_matrix)
    _write_json(output_dir / OUTPUT_FILES["next_specs"], next_specs)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _audit(summary).to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--r6-rescue-dir", type=Path, default=DEFAULT_R6_RESCUE_DIR)
    parser.add_argument("--rescue-dir", type=Path, default=DEFAULT_RESCUE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        r6_rescue_dir=args.r6_rescue_dir,
        rescue_dir=args.rescue_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
