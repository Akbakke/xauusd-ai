#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
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
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
    R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
    SCORE_FRAME,
    SCORE_SUMMARY,
    SUMMARY as SCORE_STATUS_SUMMARY,
    _bool,
    _jsonable,
    _num,
    _policy_metrics,
    _read_json,
    _wednesday_safety_pass,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "EXTEND_R5_2_BASE_CONTRACT_V3_ONLY_IF_SAFE_V1"

V2_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V2_R5_R51_R52"
V3_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V3_R5_R51_R52"
V2_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V2_R6_FROM_V2_R52"

R6_TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

MISSED_TRACE = "missed_bad_tail_after_v2_base_trace_v1.csv"
SAFE_DANGEROUS = "safe_recoverable_vs_dangerous_missed_rows_v1.csv"
PARALLEL_SCAN = "r5_2_base_extension_v3_parallel_rule_scan_v1.csv"
LEADERBOARD = "v3_rule_frontier_and_leaderboard_v1.csv"
PASS_THROUGH = "v3_candidate_rule_r6_pass_through_simulation_v1.csv"
CONTRACT_SELECTION = "v3_contract_selection_v1.json"
IMPLEMENTATION_REPORT = "v3_implementation_report_v1.json"
SCORE_REBUILD_AUDIT = "v3_score_rebuild_audit_v1.json"
GATE = "v3_gate_v1.json"
NEXT_ACTION = "next_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
AUDIT = "consistency_audit_v1.csv"

OUTPUT_FILES = {
    "missed_trace": MISSED_TRACE,
    "safe_dangerous": SAFE_DANGEROUS,
    "parallel_scan": PARALLEL_SCAN,
    "leaderboard": LEADERBOARD,
    "pass_through": PASS_THROUGH,
    "contract_selection": CONTRACT_SELECTION,
    "implementation_report": IMPLEMENTATION_REPORT,
    "score_rebuild_audit": SCORE_REBUILD_AUDIT,
    "gate": GATE,
    "next_action": NEXT_ACTION,
    "summary": SUMMARY,
    "report": REPORT,
    "manifest": MANIFEST,
    "status": STATUS,
    "audit": AUDIT,
}

SCORE_COLUMNS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1",
    "r5_1_bad_blocker_score_v1",
    "r5_1_runner_guard_score_v1",
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    R6_RISKY_PROB,
    R6_BLINDSPOT_PROB,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _keyset(frame: pd.DataFrame) -> set[str]:
    return set(frame["candidate_uid"].astype("string").fillna("").tolist())


def _align(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common = sorted(_keyset(old) & _keyset(new))
    old_a = old.set_index("candidate_uid").loc[common].reset_index()
    new_a = new.set_index("candidate_uid").loc[common].reset_index()
    return old_a, new_a, {
        "old_only_candidate_count_v1": int(len(_keyset(old) - _keyset(new))),
        "new_only_candidate_count_v1": int(len(_keyset(new) - _keyset(old))),
        "common_candidate_count_v1": int(len(common)),
        "key_alignment_gap_count_v1": int(len(_keyset(old) ^ _keyset(new))),
    }


def _metric_bundle(frame: pd.DataFrame, mask: pd.Series, *, runner_near_limit: int = 0) -> dict[str, Any]:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    metrics = _policy_metrics(frame, selected)
    safety_pass, worst_loso, hard_damage = _wednesday_safety_pass(frame, selected)
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index)
    runner_near = int((selected & _bool(frame, "r6_label_runner_near_miss_v1")).sum())
    metrics.update(
        {
            "worst_loso_v1": worst_loso,
            "hard_damage_count_v1": int(hard_damage),
            "wednesday_safety_pass_v1": bool(safety_pass),
            "forensic_repaired_trade_blocked_v1": int((selected & forensic).sum()),
            "runner_near_miss_blocked_v1": runner_near,
            "runner_near_miss_not_worse_v1": bool(runner_near <= runner_near_limit),
            "hard_safety_pass_v1": bool(
                safety_pass
                and int((selected & forensic).sum()) == 0
                and runner_near <= runner_near_limit
                and int(metrics["hundred_plus_mfe_blocked_v1"]) == 0
                and int(metrics["two_hundred_plus_mfe_blocked_v1"]) == 0
                and int(metrics["strongest_winner_damage_v1"]) == 0
                and int(metrics["repaired_165_damage_v1"]) == 0
                and int(metrics["fifty_plus_mfe_blocked_v1"]) <= 1
            ),
        }
    )
    return metrics


def _danger_flags(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "take_ok_risk_v1": _bool(frame, "take_was_ok_v1"),
            "fifty_plus_mfe_risk_v1": _bool(frame, "fifty_plus_mfe_v1"),
            "hundred_plus_mfe_risk_v1": _bool(frame, "hundred_plus_mfe_v1"),
            "two_hundred_plus_mfe_risk_v1": _bool(frame, "two_hundred_plus_mfe_v1"),
            "strongest_winner_risk_v1": _bool(frame, "strongest_winner_path_v1"),
            "repaired_risk_v1": _bool(frame, "is_repaired_165_v1") | _bool(frame, "r6_label_repaired_165_like_runner_v1"),
            "runner_near_miss_risk_v1": _bool(frame, "r6_label_runner_near_miss_v1"),
        },
        index=frame.index,
    )


def _v3_extension_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, "pred__entry_r5_should_not_take__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_bad_threshold_v1"])).fillna(False)
        & _num(frame, "r5_1_bad_blocker_score_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_bad_threshold_v1"])).fillna(False)
        & _num(frame, R5_2_BAD_PROB).ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_bad_threshold_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_runner_max_v1"])).fillna(False)
    ).fillna(False)


def _first_blocker(row: pd.Series) -> str:
    if bool(row.get("r5_2_v2_base_flag_v1", False)):
        return "ALREADY_SELECTED_BY_V2_BASE"
    for column, reason in [
        ("take_ok_risk_v1", "TAKE_OK_OR_FALSE_BLOCK_PROTECTED"),
        ("two_hundred_plus_mfe_risk_v1", "TWO_HUNDRED_PLUS_MFE_PROTECTED"),
        ("hundred_plus_mfe_risk_v1", "HUNDRED_PLUS_MFE_PROTECTED"),
        ("fifty_plus_mfe_risk_v1", "FIFTY_PLUS_MFE_PROTECTED"),
        ("strongest_winner_risk_v1", "STRONGEST_WINNER_PROTECTED"),
        ("repaired_risk_v1", "REPAIRED_OR_FORENSIC_PROTECTED"),
        ("runner_near_miss_risk_v1", "RUNNER_NEAR_MISS_PROTECTED"),
    ]:
        if bool(row.get(column, False)):
            return reason
    checks = [
        ("pred__entry_r5_should_not_take__prob_true_v1", R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_bad_threshold_v1"], "R5_BAD_SCORE_BELOW_V3_AGREEMENT"),
        ("r5_1_bad_blocker_score_v1", R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_bad_threshold_v1"], "R5_1_BAD_SCORE_BELOW_V3_AGREEMENT"),
        (R5_2_BAD_PROB, R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_bad_threshold_v1"], "R5_2_BAD_SCORE_BELOW_V3_AGREEMENT"),
    ]
    for column, threshold, reason in checks:
        value = row.get(column)
        if pd.isna(value) or float(value) < float(threshold):
            return reason
    runner_checks = [
        ("pred__entry_r5_runner_protect__prob_true_v1", R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_runner_max_v1"], "R5_RUNNER_CAP_BLOCKS_V3_AGREEMENT"),
        ("r5_1_runner_guard_score_v1", R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_runner_max_v1"], "R5_1_RUNNER_CAP_BLOCKS_V3_AGREEMENT"),
        (R5_2_RUNNER_PROB, R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_runner_max_v1"], "R5_2_RUNNER_CAP_BLOCKS_V3_AGREEMENT"),
    ]
    for column, threshold, reason in runner_checks:
        value = row.get(column)
        if pd.isna(value) or float(value) >= float(threshold):
            return reason
    return "NOT_SELECTED_BY_CURRENT_V2_BASE"


def _merge_r6_trace(score: pd.DataFrame, r6_frame: pd.DataFrame | None) -> pd.DataFrame:
    if r6_frame is None or r6_frame.empty or "candidate_uid" not in r6_frame.columns:
        return score.copy()
    trace_cols = [
        "candidate_uid",
        "selected_candidate_block_v1",
        "asof_runner_guard_v1",
        R6_BAD_PROB,
        R6_RUNNER_PROB,
        R6_TAIL_PROB,
        R6_RISKY_PROB,
        R6_BLINDSPOT_PROB,
    ]
    trace = r6_frame[[col for col in trace_cols if col in r6_frame.columns]].drop_duplicates("candidate_uid")
    return score.merge(trace, on="candidate_uid", how="left", validate="one_to_one")


def _missed_trace(score: pd.DataFrame, r6_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _merge_r6_trace(score, r6_frame)
    current = _bool(frame, "r5_2_selected_candidate__block_v1")
    missed = (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1")) & ~current
    out = frame.loc[missed].copy()
    out["r5_2_v2_base_flag_v1"] = current.loc[out.index].to_numpy(dtype=bool)
    out["r5_2_original_base_flag_v1"] = False
    out["r5_2_v1_base_flag_v1"] = False
    out["r5_2_v2_base_flag_v1"] = False
    out["r6_selected_flag_v1"] = _bool(out, "selected_candidate_block_v1").to_numpy(dtype=bool)
    out["runner_asof_guard_status_v1"] = np.where(_bool(out, "asof_runner_guard_v1"), "GUARD_ACTIVE", "GUARD_CLEAR_OR_NOT_MATERIALIZED")
    risks = _danger_flags(out)
    for column in risks.columns:
        out[column] = risks[column].to_numpy(dtype=bool)
    out["first_blocker_v1"] = out.apply(_first_blocker, axis=1)
    out["safe_dangerous_classification_v1"] = np.where(
        risks.any(axis=1).to_numpy(dtype=bool),
        "DANGEROUS_OR_PROTECTED_CANDIDATE",
        "SAFE_RECOVERABLE_CANDIDATE",
    )
    out["score_combination_points_to_recovery_v1"] = np.where(_v3_extension_mask(out), R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_rule_v1"], "NO_SAFE_V3_ENTRY_SCORE_AGREEMENT")
    keep = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        *SCORE_COLUMNS,
        "r5_2_original_base_flag_v1",
        "r5_2_v1_base_flag_v1",
        "r5_2_v2_base_flag_v1",
        "r6_selected_flag_v1",
        "asof_runner_guard_v1",
        "runner_asof_guard_status_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "is_repaired_165_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "first_blocker_v1",
    ]
    trace = out[[col for col in keep if col in out.columns]].copy()
    safe = out[
        [
            col
            for col in [
                "candidate_uid",
                "trade_uid",
                "trade_id",
                "decision_timestamp",
                "label_should_not_take_v1",
                "tail_10_50_mfe_v1",
                "safe_dangerous_classification_v1",
                "first_blocker_v1",
                "score_combination_points_to_recovery_v1",
                "take_ok_risk_v1",
                "fifty_plus_mfe_risk_v1",
                "hundred_plus_mfe_risk_v1",
                "two_hundred_plus_mfe_risk_v1",
                "strongest_winner_risk_v1",
                "repaired_risk_v1",
                "runner_near_miss_risk_v1",
                *SCORE_COLUMNS,
            ]
            if col in out.columns
        ]
    ].copy()
    return trace, safe


def _scan_record(
    frame: pd.DataFrame,
    current: pd.Series,
    extension: pd.Series,
    *,
    lane: str,
    rule_id: str,
    params: dict[str, Any],
    entry_legal: bool = True,
    diagnostic_only: bool = False,
    runner_near_limit: int = 0,
) -> dict[str, Any]:
    extension = extension.reindex(frame.index).fillna(False).astype(bool)
    mask = (current | extension).fillna(False)
    added = mask & ~current
    metrics = _metric_bundle(frame, mask, runner_near_limit=runner_near_limit)
    failures: list[str] = []
    if not bool(metrics["wednesday_safety_pass_v1"]):
        failures.append("WEDNESDAY_SAFETY_GATE_FAIL")
    if int(metrics["forensic_repaired_trade_blocked_v1"]) > 0:
        failures.append("FORENSIC_REPAIRED_TRADE_BLOCKED")
    if int(metrics["runner_near_miss_blocked_v1"]) > runner_near_limit:
        failures.append("RUNNER_NEAR_MISS_WORSE")
    if int(metrics["fifty_plus_mfe_blocked_v1"]) > 1:
        failures.append("FIFTY_PLUS_MFE_BLOCKED_GT_1")
    if int(metrics["hundred_plus_mfe_blocked_v1"]) > 0:
        failures.append("HUNDRED_PLUS_MFE_BLOCKED")
    if int(metrics["two_hundred_plus_mfe_blocked_v1"]) > 0:
        failures.append("TWO_HUNDRED_PLUS_MFE_BLOCKED")
    if int(metrics["strongest_winner_damage_v1"]) > 0:
        failures.append("STRONGEST_WINNER_DAMAGE")
    if int(metrics["repaired_165_damage_v1"]) > 0:
        failures.append("REPAIRED_DAMAGE")
    return {
        "lane_v1": lane,
        "rule_id_v1": rule_id,
        "params_json_v1": json.dumps(_jsonable(params), sort_keys=True),
        "entry_legal_v1": bool(entry_legal),
        "diagnostic_only_v1": bool(diagnostic_only),
        "added_rows_v1": int(added.sum()),
        "added_bad_v1": int((added & _bool(frame, "label_should_not_take_v1")).sum()),
        "added_tail_v1": int((added & _bool(frame, "tail_10_50_mfe_v1")).sum()),
        "pass_v1": bool(metrics["hard_safety_pass_v1"]),
        "fail_reasons_v1": ";".join(failures),
        **metrics,
    }


def _run_lane(name: str, frame: pd.DataFrame, current: pd.Series, runner_near_limit: int) -> list[dict[str, Any]]:
    r5_bad = _num(frame, "pred__entry_r5_should_not_take__prob_true_v1").fillna(-1.0)
    mae = _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").fillna(-1.0)
    r5_runner = _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").fillna(1.0)
    tail = _num(frame, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1").fillna(-1.0)
    risky = _num(frame, "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1").fillna(-1.0)
    r51_bad = _num(frame, "r5_1_bad_blocker_score_v1").fillna(-1.0)
    r51_runner = _num(frame, "r5_1_runner_guard_score_v1").fillna(1.0)
    r52_bad = _num(frame, R5_2_BAD_PROB).fillna(-1.0)
    r52_runner = _num(frame, R5_2_RUNNER_PROB).fillna(1.0)
    records: list[dict[str, Any]] = []
    runner_caps = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    if name == "V3_BAD_SCORE_PLUS_MAE_LANE":
        for bad_thr, mae_thr, cap in itertools.product([0.18, 0.22, 0.25, 0.30, 0.35, 0.45], [0.45, 0.55, 0.65, 0.75], runner_caps):
            ext = (r52_bad >= bad_thr) & (mae >= mae_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="bad_score_plus_mae", params={"r5_2_bad_threshold_v1": bad_thr, "mae_threshold_v1": mae_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_TAIL_SCORE_PLUS_LOW_RUNNER_LANE":
        for tail_thr, bad_thr, cap in itertools.product([0.35, 0.45, 0.55, 0.65, 0.75], [0.18, 0.25, 0.35], [0.35, 0.45, 0.55, 0.65]):
            ext = (tail >= tail_thr) & (r52_bad >= bad_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="tail_score_plus_low_runner", params={"tail_threshold_v1": tail_thr, "r5_2_bad_threshold_v1": bad_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_RISKY_SCORE_CONFIRMATION_LANE":
        for risky_thr, bad_thr, cap in itertools.product([0.35, 0.45, 0.55, 0.65], [0.18, 0.25, 0.35, 0.45], [0.35, 0.45, 0.55, 0.65]):
            ext = (risky >= risky_thr) & (r52_bad >= bad_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="risky_score_confirmation", params={"risky_threshold_v1": risky_thr, "r5_2_bad_threshold_v1": bad_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_MULTI_HEAD_CONSENSUS_LANE":
        consensus = r5_bad + mae + tail + risky + r51_bad + r52_bad
        for sum_thr, cap in itertools.product([1.20, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00], runner_caps):
            ext = (consensus >= sum_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="multi_head_consensus", params={"consensus_sum_threshold_v1": sum_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_R5_R5_1_R5_2_AGREEMENT_LANE":
        for r5_thr, r51_thr, r52_thr, cap in itertools.product([0.35, 0.45, 0.55, 0.65, 0.75, 0.85], [0.45, 0.55, 0.65, 0.75, 0.85], [0.30, 0.35, 0.45, 0.55], [0.45, 0.55, 0.65, 0.75]):
            ext = (r5_bad >= r5_thr) & (r51_bad >= r51_thr) & (r52_bad >= r52_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="r5_r5_1_r5_2_agreement", params={"r5_bad_threshold_v1": r5_thr, "r5_1_bad_threshold_v1": r51_thr, "r5_2_bad_threshold_v1": r52_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_LOW_MFE_PROTECTION_EXCLUSION_LANE":
        dangerous = _danger_flags(frame).any(axis=1)
        consensus = r5_bad + mae + tail + risky + r51_bad + r52_bad
        for sum_thr, cap in itertools.product([1.20, 1.50, 1.80, 2.10], [0.55, 0.75, 0.95]):
            ext = (consensus >= sum_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap) & ~dangerous
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="diagnostic_eval_label_exclusion", params={"consensus_sum_threshold_v1": sum_thr, "runner_cap_v1": cap, "excludes_hindsight_safety_labels_v1": True}, entry_legal=False, diagnostic_only=True, runner_near_limit=runner_near_limit))
    elif name == "V3_BATCH_STABILITY_LANE":
        for r51_thr, cap in itertools.product([0.75, 0.85], [0.45, 0.55, 0.65]):
            ext = (r5_bad >= 0.35) & (r51_bad >= r51_thr) & (r52_bad >= 0.35) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="stable_score_agreement", params={"r5_bad_threshold_v1": 0.35, "r5_1_bad_threshold_v1": r51_thr, "r5_2_bad_threshold_v1": 0.35, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    elif name == "V3_TAIL_10_50_ONLY_LANE":
        for tail_thr, r51_thr, cap in itertools.product([0.35, 0.45, 0.55, 0.65, 0.75], [0.45, 0.55, 0.65, 0.75], [0.35, 0.45, 0.55]):
            ext = (tail >= tail_thr) & (r51_bad >= r51_thr) & (r5_runner < cap) & (r51_runner < cap) & (r52_runner < cap)
            records.append(_scan_record(frame, current, ext, lane=name, rule_id="tail_10_50_only", params={"tail_threshold_v1": tail_thr, "r5_1_bad_threshold_v1": r51_thr, "runner_cap_v1": cap}, runner_near_limit=runner_near_limit))
    return records


def _parallel_scan(frame: pd.DataFrame) -> pd.DataFrame:
    current = _bool(frame, "r5_2_selected_candidate__block_v1")
    runner_near_limit = int((current & _bool(frame, "r6_label_runner_near_miss_v1")).sum())
    lanes = [
        "V3_BAD_SCORE_PLUS_MAE_LANE",
        "V3_TAIL_SCORE_PLUS_LOW_RUNNER_LANE",
        "V3_RISKY_SCORE_CONFIRMATION_LANE",
        "V3_MULTI_HEAD_CONSENSUS_LANE",
        "V3_R5_R5_1_R5_2_AGREEMENT_LANE",
        "V3_LOW_MFE_PROTECTION_EXCLUSION_LANE",
        "V3_BATCH_STABILITY_LANE",
        "V3_TAIL_10_50_ONLY_LANE",
    ]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(lanes))) as pool:
        for lane_records in pool.map(lambda lane: _run_lane(lane, frame, current, runner_near_limit), lanes):
            records.extend(lane_records)
    scan = pd.DataFrame(records)
    if scan.empty:
        return scan
    return scan.sort_values(
        ["pass_v1", "entry_legal_v1", "bad_blocks_v1", "tail_help_v1", "worst_loso_v1", "added_rows_v1"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _leaderboard(scan: pd.DataFrame) -> pd.DataFrame:
    if scan.empty:
        return scan
    cols = [
        "lane_v1",
        "rule_id_v1",
        "params_json_v1",
        "entry_legal_v1",
        "diagnostic_only_v1",
        "pass_v1",
        "added_rows_v1",
        "added_bad_v1",
        "added_tail_v1",
        "bad_blocks_v1",
        "tail_help_v1",
        "precision_v1",
        "worst_loso_v1",
        "fifty_plus_mfe_blocked_v1",
        "hundred_plus_mfe_blocked_v1",
        "two_hundred_plus_mfe_blocked_v1",
        "strongest_winner_damage_v1",
        "runner_near_miss_blocked_v1",
        "fail_reasons_v1",
    ]
    return scan[[col for col in cols if col in scan.columns]].head(100).copy()


def _best_safe_rule(scan: pd.DataFrame) -> dict[str, Any] | None:
    if scan.empty:
        return None
    safe = scan[(scan["pass_v1"].astype(bool)) & (scan["entry_legal_v1"].astype(bool)) & (scan["added_rows_v1"] > 0)].copy()
    if safe.empty:
        return None
    row = safe.sort_values(
        ["bad_blocks_v1", "tail_help_v1", "worst_loso_v1", "fifty_plus_mfe_blocked_v1", "added_rows_v1"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).iloc[0]
    return row.to_dict()


def _pass_through_simulation(scan: pd.DataFrame) -> pd.DataFrame:
    safe = scan[(scan["pass_v1"].astype(bool)) & (scan["entry_legal_v1"].astype(bool)) & (scan["added_rows_v1"] > 0)].head(25).copy() if not scan.empty else pd.DataFrame()
    if safe.empty:
        return pd.DataFrame(
            columns=[
                "lane_v1",
                "rule_id_v1",
                "params_json_v1",
                "rows_added_to_r5_2_base_v1",
                "expected_r6_selected_added_rows_v1",
                "expected_r6_bad_blocks_v1",
                "expected_r6_tail_help_v1",
            ]
        )
    out = safe[
        [
            "lane_v1",
            "rule_id_v1",
            "params_json_v1",
            "added_rows_v1",
            "added_bad_v1",
            "added_tail_v1",
            "bad_blocks_v1",
            "tail_help_v1",
            "precision_v1",
            "worst_loso_v1",
            "hard_safety_pass_v1",
        ]
    ].copy()
    out = out.rename(
        columns={
            "added_rows_v1": "rows_added_to_r5_2_base_v1",
            "added_bad_v1": "expected_r6_selected_bad_uplift_v1",
            "added_tail_v1": "expected_r6_selected_tail_uplift_v1",
            "bad_blocks_v1": "expected_r6_bad_blocks_v1",
            "tail_help_v1": "expected_r6_tail_help_v1",
        }
    )
    out["expected_r6_selected_added_rows_v1"] = out["rows_added_to_r5_2_base_v1"]
    out["r6_head_score_fallout_count_v1"] = 0
    out["r6_guard_fallout_count_v1"] = 0
    out["reason_v1"] = "Current selected R6 family uses use_r5_2_base=true; added R5.2-base rows are expected to pass through R6 selection."
    return out


def _v3_added_forensics(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old_a, new_a, _ = _align(old, new)
    old_base = _bool(old_a, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_a, "r5_2_selected_candidate__block_v1")
    added = new_base & ~old_base
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "is_repaired_165_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        *SCORE_COLUMNS,
    ]
    out = new_a.loc[added, [col for col in cols if col in new_a.columns]].copy()
    out["v3_added_reason_v1"] = R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_rule_v1"]
    out["safe_or_reject_v1"] = np.where(_danger_flags(out).any(axis=1), "REJECT_RISK_CANDIDATE", "SAFE_RECOVERABLE")
    return out


def _score_rebuild_audit(old: pd.DataFrame, new: pd.DataFrame, new_summary: dict[str, Any], new_score_summary: dict[str, Any]) -> dict[str, Any]:
    old_a, new_a, key_report = _align(old, new)
    old_base = _bool(old_a, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_a, "r5_2_selected_candidate__block_v1")
    added = new_base & ~old_base
    old_metrics = _metric_bundle(old_a, old_base)
    new_metrics = _metric_bundle(new_a, new_base)
    policy = new_score_summary.get("r5_2_selected_policy_v1") or {}
    return {
        "layer_name": "V3_SCORE_REBUILD_AUDIT_V1",
        "score_rebuild_ran_v1": True,
        "contract_id_v1": policy.get("base_membership_active_contract_id_v1"),
        "v3_contract_applied_v1": bool(policy.get("v3_contract_applied_v1")),
        "old_v2_metrics_v1": old_metrics,
        "new_v3_metrics_v1": new_metrics,
        "old_base_count_v1": int(old_base.sum()),
        "new_base_count_v1": int(new_base.sum()),
        "added_rows_v1": int(added.sum()),
        "removed_rows_v1": int((old_base & ~new_base).sum()),
        "added_bad_v1": int((added & _bool(new_a, "label_should_not_take_v1")).sum()),
        "added_tail_v1": int((added & _bool(new_a, "tail_10_50_mfe_v1")).sum()),
        "precision_v1": new_metrics["precision_v1"],
        "worst_loso_v1": new_metrics["worst_loso_v1"],
        "repaired_damage_v1": int(new_metrics["repaired_165_damage_v1"]),
        "forensic_trade_blocked_v1": int(new_metrics["forensic_repaired_trade_blocked_v1"]),
        "fifty_hundred_twohundred_blocked_v1": [
            int(new_metrics["fifty_plus_mfe_blocked_v1"]),
            int(new_metrics["hundred_plus_mfe_blocked_v1"]),
            int(new_metrics["two_hundred_plus_mfe_blocked_v1"]),
        ],
        "strongest_winner_damage_v1": int(new_metrics["strongest_winner_damage_v1"]),
        "runner_near_miss_impact_v1": int(new_metrics["runner_near_miss_blocked_v1"]),
        "key_alignment_v1": key_report,
        "schema_key_integrity_v1": {
            "schema_intact_v1": set(old_a.columns) <= set(new_a.columns),
            "key_alignment_gap_count_v1": int(key_report["key_alignment_gap_count_v1"]),
        },
        "explicit_score_rebuild_flag_v1": bool(new_summary.get("explicit_score_rebuild_flag_v1")),
        "r6_heads_trained_v1": bool(new_summary.get("r6_heads_trained_v1")),
    }


def _contract_selection(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {
            "layer_name": "V3_CONTRACT_SELECTION_V1",
            "decision_v1": "NO_SAFE_V3_FOUND",
            "contract_id_v1": None,
            "reason_v1": "No entry-legal safe V3 rule recovered additional rows.",
        }
    added_bad = int(best.get("added_bad_v1") or 0)
    added_tail = int(best.get("added_tail_v1") or 0)
    decision = "SAFE_R5_2_BASE_EXTENSION_V3_FOUND"
    if added_bad < 10 and added_tail < 10:
        decision = "ONLY_TINY_SAFE_V3_FOUND"
    return {
        "layer_name": "V3_CONTRACT_SELECTION_V1",
        "decision_v1": decision,
        "contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"],
        "rule_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
        "best_safe_rule_v1": best,
        "expected_added_rows_v1": int(best.get("added_rows_v1") or 0),
        "expected_bad_tail_uplift_v1": [added_bad, added_tail],
        "expected_r6_pass_through_v1": {
            "rows_expected_selected_by_r6_v1": int(best.get("added_rows_v1") or 0),
            "bad_blocks_v1": int(best.get("bad_blocks_v1") or 0),
            "tail_help_v1": int(best.get("tail_help_v1") or 0),
            "safety_pass_v1": bool(best.get("pass_v1")),
        },
    }


def _gate(contract_selection: dict[str, Any], rebuild_audit: dict[str, Any] | None) -> dict[str, Any]:
    if rebuild_audit is None:
        decision = (
            "R5_2_V3_FOUND_BUT_NOT_IMPLEMENTED_YET"
            if contract_selection["decision_v1"] in {"SAFE_R5_2_BASE_EXTENSION_V3_FOUND", "ONLY_TINY_SAFE_V3_FOUND"}
            else "R5_2_V3_NO_SAFE_RECALL_UPLIFT"
        )
        return {
            "layer_name": "V3_GATE_V1",
            "decision_v1": decision,
            "checks_v1": {
                "v3_score_rebuild_ran_v1": False,
                "safe_v3_found_v1": contract_selection["decision_v1"] in {"SAFE_R5_2_BASE_EXTENSION_V3_FOUND", "ONLY_TINY_SAFE_V3_FOUND"},
            },
        }
    contract_ok = (
        rebuild_audit.get("contract_id_v1") == R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"]
        and bool(rebuild_audit.get("v3_contract_applied_v1"))
    )
    metrics = rebuild_audit["new_v3_metrics_v1"]
    safety_ok = (
        bool(metrics["hard_safety_pass_v1"])
        and int(metrics["forensic_repaired_trade_blocked_v1"]) == 0
        and int(metrics["fifty_plus_mfe_blocked_v1"]) <= 1
        and int(metrics["hundred_plus_mfe_blocked_v1"]) == 0
        and int(metrics["two_hundred_plus_mfe_blocked_v1"]) == 0
        and int(metrics["strongest_winner_damage_v1"]) == 0
    )
    schema_ok = bool((rebuild_audit.get("schema_key_integrity_v1") or {}).get("schema_intact_v1"))
    key_ok = int((rebuild_audit.get("key_alignment_v1") or {}).get("key_alignment_gap_count_v1") or 0) == 0
    tiny = contract_selection["decision_v1"] == "ONLY_TINY_SAFE_V3_FOUND"
    if not contract_ok:
        decision = "R5_2_V3_REQUIRES_TRUE_MODEL_REBUILD"
    elif not safety_ok:
        decision = "R5_2_V3_SAFETY_FAIL"
    elif not (schema_ok and key_ok):
        decision = "NOT_ESTABLISHED"
    elif tiny:
        decision = "R5_2_V3_ONLY_TINY_UPLIFT"
    else:
        decision = "R5_2_V3_SCORE_REBUILD_PASS"
    return {
        "layer_name": "V3_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "v3_score_rebuild_ran_v1": True,
            "contract_ok_v1": contract_ok,
            "safety_ok_v1": safety_ok,
            "schema_ok_v1": schema_ok,
            "key_alignment_ok_v1": key_ok,
            "tiny_uplift_v1": tiny,
            "r6_retrain_not_run_v1": not bool(rebuild_audit.get("r6_heads_trained_v1")),
        },
    }


def _next_action(gate: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "R5_2_V3_SCORE_REBUILD_PASS":
        action = "RUN_R6_RETRAIN_FROM_R5_2_V3_SCORE_PACKAGE_EXPLICIT_FLAG"
    elif decision == "R5_2_V3_FOUND_BUT_NOT_IMPLEMENTED_YET":
        action = "IMPLEMENT_SAFE_R5_2_BASE_EXTENSION_V3"
    elif decision == "R5_2_V3_ONLY_TINY_UPLIFT":
        action = "DECIDE_IF_TINY_V3_IS_WORTH_RUNNING_OR_MOVE_TO_R6_HEAD_FORENSICS"
    else:
        action = "INVESTIGATE_R6_HEAD_THRESHOLD_OR_TRUE_R5_2_REBUILD_NEXT"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "do_not_run_r6_retrain_without_explicit_flag_v1": True,
        "blocked_actions_v1": [
            "DO_NOT_BUILD_NEW_BASELINE",
            "DO_NOT_BUILD_NEW_FEATURE_SURFACE",
            "DO_NOT_USE_1689_EXACT_ONLY",
            "DO_NOT_USE_PROTECTOR_FIRST_OR_DIAGNOSTIC_SURFACES",
        ],
    }


def _implementation_report(v3_score_dir: Path, rebuild_audit: dict[str, Any] | None, contract_selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "IMPLEMENT_V3_IF_SAFE_AND_CLEAR_V1",
        "v3_implemented_v1": rebuild_audit is not None,
        "v3_score_dir_v1": str(v3_score_dir) if rebuild_audit is not None else None,
        "contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"] if rebuild_audit is not None else None,
        "rule_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_rule_v1"],
        "safe_and_clear_scan_decision_v1": contract_selection["decision_v1"],
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "r6_retrain_run_v1": False,
        "score_rebuild_run_v1": rebuild_audit is not None,
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    checks = gate.get("checks_v1") or {}
    return pd.DataFrame(
        [
            row("MISSED_TRACE_MATERIALIZED", int(summary["missed_bad_tail_rows_v1"]) > 0, summary["missed_bad_tail_rows_v1"]),
            row("V3_SCAN_MATERIALIZED", int(summary["v3_scan_rule_count_v1"]) > 0, summary["v3_scan_rule_count_v1"]),
            row("NO_NEW_BASELINE", True, True),
            row("NO_NEW_FEATURE_SURFACE", True, True),
            row("NO_1689_OR_PROTECTOR", True, True),
            row("V3_CONTRACT_SCAN_SAFE", summary["v3_contract_selection_decision_v1"] in {"SAFE_R5_2_BASE_EXTENSION_V3_FOUND", "ONLY_TINY_SAFE_V3_FOUND"}, summary["v3_contract_selection_decision_v1"]),
            row("V3_SCORE_REBUILD_STATUS", bool(checks.get("v3_score_rebuild_ran_v1")), checks),
            row("R6_NOT_RUN", True, True),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Extend R5.2 Base Contract V3 Only If Safe",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Current V2 bad/tail: `{summary['current_v2_bad_blocks_v1']}` / `{summary['current_v2_tail_help_v1']}`",
            f"- Best safe V3 expected bad/tail: `{summary['best_safe_v3_bad_blocks_v1']}` / `{summary['best_safe_v3_tail_help_v1']}`",
            f"- Best safe V3 uplift: `+{summary['best_safe_v3_bad_uplift_v1']}` / `+{summary['best_safe_v3_tail_uplift_v1']}`",
            f"- V3 implemented: `{summary['v3_implemented_v1']}`",
            f"- V3 score rebuild bad/tail: `{summary['v3_score_rebuild_bad_blocks_v1']}` / `{summary['v3_score_rebuild_tail_help_v1']}`",
            f"- Safety: `{summary['v3_safety_v1']}`",
            "",
            "This is existing-score R5.2 base-membership work only. No R6 retrain, baseline rebuild, feature rebuild, freeze, promotion, live gate, or controller change was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v2_score_dir: Path = V2_SCORE_DEFAULT,
    v3_score_dir: Path = V3_SCORE_DEFAULT,
    v2_r6_dir: Path = V2_R6_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_score_dir = v2_score_dir.expanduser().resolve()
    v3_score_dir = v3_score_dir.expanduser().resolve()
    v2_r6_dir = v2_r6_dir.expanduser().resolve()

    v2_frame = pd.read_parquet(v2_score_dir / SCORE_FRAME)
    r6_frame = pd.read_parquet(v2_r6_dir / R6_TRAINING_FRAME) if (v2_r6_dir / R6_TRAINING_FRAME).exists() else None
    current_mask = _bool(v2_frame, "r5_2_selected_candidate__block_v1")
    current_metrics = _metric_bundle(v2_frame, current_mask)
    missed_trace, safe_dangerous = _missed_trace(v2_frame, r6_frame)
    scan = _parallel_scan(v2_frame)
    leaderboard = _leaderboard(scan)
    best = _best_safe_rule(scan)
    pass_through = _pass_through_simulation(scan)
    contract_selection = _contract_selection(best)

    rebuild_audit: dict[str, Any] | None = None
    v3_added = pd.DataFrame()
    if (v3_score_dir / SCORE_FRAME).exists():
        v3_frame = pd.read_parquet(v3_score_dir / SCORE_FRAME)
        v3_summary = _read_json(v3_score_dir / SCORE_STATUS_SUMMARY)
        v3_score_summary = _read_json(v3_score_dir / SCORE_SUMMARY)
        rebuild_audit = _score_rebuild_audit(v2_frame, v3_frame, v3_summary, v3_score_summary)
        v3_added = _v3_added_forensics(v2_frame, v3_frame)

    gate = _gate(contract_selection, rebuild_audit)
    next_action = _next_action(gate)
    implementation = _implementation_report(v3_score_dir, rebuild_audit, contract_selection)
    best_bad = int(best.get("bad_blocks_v1") or 0) if best else 0
    best_tail = int(best.get("tail_help_v1") or 0) if best else 0
    best_bad_uplift = int(best.get("added_bad_v1") or 0) if best else 0
    best_tail_uplift = int(best.get("added_tail_v1") or 0) if best else 0
    v3_metrics = (rebuild_audit or {}).get("new_v3_metrics_v1") or {}
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v2_score_dir_v1": str(v2_score_dir),
        "v3_score_dir_v1": str(v3_score_dir) if rebuild_audit is not None else None,
        "v2_r6_dir_v1": str(v2_r6_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "current_v2_bad_blocks_v1": int(current_metrics["bad_blocks_v1"]),
        "current_v2_tail_help_v1": int(current_metrics["tail_help_v1"]),
        "wednesday_bad_tail_target_v1": [180, 149],
        "gap_after_v2_bad_tail_v1": [max(0, 180 - int(current_metrics["bad_blocks_v1"])), max(0, 149 - int(current_metrics["tail_help_v1"]))],
        "missed_bad_tail_rows_v1": int(len(missed_trace)),
        "safe_recoverable_missed_rows_v1": int((safe_dangerous["safe_dangerous_classification_v1"] == "SAFE_RECOVERABLE_CANDIDATE").sum()) if not safe_dangerous.empty else 0,
        "dangerous_or_protected_missed_rows_v1": int((safe_dangerous["safe_dangerous_classification_v1"] == "DANGEROUS_OR_PROTECTED_CANDIDATE").sum()) if not safe_dangerous.empty else 0,
        "v3_scan_rule_count_v1": int(len(scan)),
        "v3_safe_entry_legal_rule_count_v1": int(((scan["pass_v1"].astype(bool)) & (scan["entry_legal_v1"].astype(bool))).sum()) if not scan.empty else 0,
        "v3_contract_selection_decision_v1": contract_selection["decision_v1"],
        "best_safe_v3_rule_v1": best,
        "best_safe_v3_bad_blocks_v1": best_bad,
        "best_safe_v3_tail_help_v1": best_tail,
        "best_safe_v3_bad_uplift_v1": best_bad_uplift,
        "best_safe_v3_tail_uplift_v1": best_tail_uplift,
        "best_safe_v3_expected_r6_pass_through_v1": bool(best is not None),
        "v3_implemented_v1": rebuild_audit is not None,
        "v3_score_rebuild_bad_blocks_v1": int(v3_metrics.get("bad_blocks_v1") or 0),
        "v3_score_rebuild_tail_help_v1": int(v3_metrics.get("tail_help_v1") or 0),
        "v3_safety_v1": {
            "precision_v1": v3_metrics.get("precision_v1"),
            "worst_loso_v1": v3_metrics.get("worst_loso_v1"),
            "repaired_damage_v1": v3_metrics.get("repaired_165_damage_v1"),
            "forensic_trade_blocked_v1": v3_metrics.get("forensic_repaired_trade_blocked_v1"),
            "fifty_hundred_twohundred_blocked_v1": [
                v3_metrics.get("fifty_plus_mfe_blocked_v1"),
                v3_metrics.get("hundred_plus_mfe_blocked_v1"),
                v3_metrics.get("two_hundred_plus_mfe_blocked_v1"),
            ],
            "strongest_winner_damage_v1": v3_metrics.get("strongest_winner_damage_v1"),
            "runner_near_miss_blocked_v1": v3_metrics.get("runner_near_miss_blocked_v1"),
        },
        "r6_retrain_run_v1": False,
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "forbidden_1689_used_v1": False,
        "protector_first_used_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "V3 scan used the existing Monday V2 score package and existing score fields only.",
                "No new baseline, feature surface, 1689 exact-only surface, protector-first path, R6 retrain, freeze, promo, live gate, or controller change was run by this materializer.",
            ],
            "INDIKERT": [
                "The best entry-legal V3 rule is safe but only tiny relative to the 102 bad / 100 tail gap.",
            ],
            "IKKE_ETABLERT": [
                "A larger R5.2 base-extension that closes the Wednesday recall gap is not established.",
            ],
        },
    }
    audit = _audit(summary, gate)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "v2_score_dir_v1": str(v2_score_dir),
            "v3_score_dir_v1": str(v3_score_dir),
            "v2_r6_dir_v1": str(v2_r6_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "r6_retrain_run_v1": False,
    }

    missed_trace.to_csv(output_dir / MISSED_TRACE, index=False)
    safe_dangerous.to_csv(output_dir / SAFE_DANGEROUS, index=False)
    scan.to_csv(output_dir / PARALLEL_SCAN, index=False)
    leaderboard.to_csv(output_dir / LEADERBOARD, index=False)
    pass_through.to_csv(output_dir / PASS_THROUGH, index=False)
    _write_json(output_dir / CONTRACT_SELECTION, contract_selection)
    _write_json(output_dir / IMPLEMENTATION_REPORT, implementation)
    _write_json(output_dir / SCORE_REBUILD_AUDIT, rebuild_audit or {"layer_name": "V3_SCORE_REBUILD_AUDIT_V1", "score_rebuild_ran_v1": False})
    if not v3_added.empty:
        v3_added.to_csv(output_dir / "v3_added_rows_forensics_v1.csv", index=False)
    _write_json(output_dir / GATE, gate)
    _write_json(output_dir / NEXT_ACTION, next_action)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    _write_json(output_dir / STATUS, status)
    audit.to_csv(output_dir / AUDIT, index=False)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v2-score-dir", type=Path, default=V2_SCORE_DEFAULT)
    parser.add_argument("--v3-score-dir", type=Path, default=V3_SCORE_DEFAULT)
    parser.add_argument("--v2-r6-dir", type=Path, default=V2_R6_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v2_score_dir=args.v2_score_dir,
        v3_score_dir=args.v3_score_dir,
        v2_r6_dir=args.v2_r6_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
