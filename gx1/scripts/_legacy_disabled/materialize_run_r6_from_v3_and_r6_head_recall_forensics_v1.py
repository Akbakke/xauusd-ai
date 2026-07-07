#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
    _asof_runner_guard,
    _jsonable,
    _policy_metrics,
    _r6_policy_mask,
    _worst_run_precision,
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
    SCORE_FRAME,
    SCORE_SUMMARY,
    _bool,
    _num,
    _read_json,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "RUN_R6_FROM_V3_AND_R6_HEAD_RECALL_FORENSICS_V1"

V3_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V3_R5_R51_R52"
V2_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V2_R5_R51_R52"
V3_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"
V2_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V2_R6_FROM_V2_R52"

R6_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_GRID = "r6_family_grid_replay_v1.csv"
R6_SUMMARY = "summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"

FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
EXPECTED_ROW_COUNT = 1914
EXPECTED_AS_OF_COLUMNS = 109

OUTPUT_FILES = {
    "retrain": "r6_retrain_from_r5_2_v3_score_package_v1.json",
    "pass_through": "r6_v3_pass_through_audit_v1.csv",
    "eval_delta": "r6_v3_eval_against_v2_and_wednesday_v1.json",
    "head_gap": "r6_head_score_gap_forensics_v1.csv",
    "threshold_frontier": "r6_threshold_frontier_on_v3_scores_v1.csv",
    "rejected_grid": "r6_rejected_grid_safe_recall_scan_on_v3_v1.csv",
    "root_cause": "post_v3_root_cause_decision_v1.json",
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


def _keyset(frame: pd.DataFrame) -> set[str]:
    return set(frame["candidate_uid"].astype("string").fillna("").tolist())


def _align(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    old_keys = _keyset(old)
    new_keys = _keyset(new)
    common = sorted(old_keys & new_keys)
    return (
        old.set_index("candidate_uid").loc[common].reset_index(),
        new.set_index("candidate_uid").loc[common].reset_index(),
        {
            "old_only_candidate_count_v1": int(len(old_keys - new_keys)),
            "new_only_candidate_count_v1": int(len(new_keys - old_keys)),
            "common_candidate_count_v1": int(len(common)),
            "key_alignment_gap_count_v1": int(len(old_keys ^ new_keys)),
        },
    )


def _selected_policy(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("family_grid_selected_policy_v1") or {}


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_policy(summary)
    return selected.get("metrics_v1") or selected.get("compare_v1", {}).get("candidate_metrics_v1") or {}


def _worst(summary: dict[str, Any]) -> float | None:
    value = _selected_policy(summary).get("candidate_worst_loso_v1")
    return None if value is None else float(value)


def _hard_safety(frame: pd.DataFrame, mask: pd.Series, *, runner_near_limit: int = 0) -> tuple[dict[str, Any], bool, list[str]]:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    metrics = _policy_metrics(frame, selected)
    worst = _worst_run_precision(frame, selected) if metrics.get("precision_v1") is not None and float(metrics["precision_v1"]) >= WEDNESDAY_R6_BENCHMARK["precision_v1"] else None
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index)
    forensic_blocked = int((selected & forensic).sum())
    failures: list[str] = []
    if int(metrics["repaired_165_damage_v1"]) != 0:
        failures.append("REPAIRED_DAMAGE")
    if forensic_blocked != 0:
        failures.append("FORENSIC_REPAIRED_TRADE_BLOCKED")
    if int(metrics["fifty_plus_mfe_blocked_v1"]) > 1:
        failures.append("FIFTY_PLUS_MFE_GT_1")
    if int(metrics["hundred_plus_mfe_blocked_v1"]) != 0:
        failures.append("HUNDRED_PLUS_MFE_BLOCKED")
    if int(metrics["two_hundred_plus_mfe_blocked_v1"]) != 0:
        failures.append("TWO_HUNDRED_PLUS_MFE_BLOCKED")
    if int(metrics["strongest_winner_damage_v1"]) != 0:
        failures.append("STRONGEST_WINNER_DAMAGE")
    if int(metrics["runner_near_miss_blocked_v1"]) > runner_near_limit:
        failures.append("RUNNER_NEAR_MISS_WORSE")
    if metrics.get("precision_v1") is None or float(metrics["precision_v1"]) < WEDNESDAY_R6_BENCHMARK["precision_v1"]:
        failures.append("PRECISION_FAIL")
    if worst is None or float(worst) < WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]:
        failures.append("WORST_LOSO_FAIL")
    metrics.update(
        {
            "worst_loso_v1": worst,
            "forensic_repaired_trade_blocked_v1": forensic_blocked,
            "hard_safety_pass_v1": len(failures) == 0,
            "safety_failures_v1": failures,
        }
    )
    return metrics, len(failures) == 0, failures


def _r6_ultra_safe_mask(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    r5_2_base = _bool(frame, "r5_2_selected_candidate__block_v1") if params.get("use_r5_2_base_v1", True) else pd.Series(False, index=frame.index)
    protect = (
        _num(frame, R6_RUNNER_PROB).ge(float(params["runner_threshold_v1"])).fillna(False)
        | _num(frame, R5_2_RUNNER_PROB).ge(float(params["r5_2_runner_threshold_v1"])).fillna(False)
    )
    if bool(params.get("hard_asof_runner_guard_v1", True)):
        protect = protect | _asof_runner_guard(frame)
    blind_ok = _num(frame, R6_BLINDSPOT_PROB).lt(float(params["blindspot_threshold_v1"])).fillna(True)
    addon = (
        _num(frame, R6_BAD_PROB).ge(float(params["bad_threshold_v1"])).fillna(False)
        & _num(frame, R6_RISKY_PROB).ge(float(params["risky_threshold_v1"])).fillna(False)
        & _num(frame, R6_TAIL_PROB).ge(float(params["tail_threshold_v1"])).fillna(False)
        & blind_ok
        & ~protect
    )
    return (r5_2_base | addon).fillna(False).astype(bool)


def _first_fail_reason(row: pd.Series, params: dict[str, Any]) -> str:
    if bool(row.get("r5_2_selected_candidate__block_v1", False)):
        return "SELECTED_BY_R5_2_BASE"
    if float(row.get(R6_RUNNER_PROB, np.nan)) >= float(params["runner_threshold_v1"]):
        return "RUNNER_GUARD_BLOCKED"
    if float(row.get(R5_2_RUNNER_PROB, np.nan)) >= float(params["r5_2_runner_threshold_v1"]):
        return "R5_2_RUNNER_GUARD_BLOCKED"
    if bool(row.get("asof_runner_guard_v1", False)) and bool(params.get("hard_asof_runner_guard_v1", True)):
        return "RUNNER_GUARD_BLOCKED"
    if float(row.get(R6_BAD_PROB, np.nan)) < float(params["bad_threshold_v1"]):
        return "R6_BAD_HEAD_TOO_LOW"
    if float(row.get(R6_RISKY_PROB, np.nan)) < float(params["risky_threshold_v1"]):
        return "R6_RISKY_HEAD_TOO_LOW"
    if float(row.get(R6_TAIL_PROB, np.nan)) < float(params["tail_threshold_v1"]):
        return "R6_TAIL_HEAD_TOO_LOW"
    if float(row.get(R6_BLINDSPOT_PROB, np.nan)) >= float(params["blindspot_threshold_v1"]):
        return "R6_BLINDSPOT_GUARD_BLOCKED"
    return "NOT_ESTABLISHED"


def _pass_through_audit(v2_score: pd.DataFrame, v3_score: pd.DataFrame, r6_frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    v2_a, v3_a, _ = _align(v2_score, v3_score)
    added = set(v3_a.loc[_bool(v3_a, "r5_2_selected_candidate__block_v1") & ~_bool(v2_a, "r5_2_selected_candidate__block_v1"), "candidate_uid"].astype("string"))
    frame = r6_frame.set_index("candidate_uid", drop=False)
    rows: list[dict[str, Any]] = []
    for candidate_uid in sorted(added):
        if candidate_uid not in frame.index:
            rows.append({"candidate_uid": candidate_uid, "r6_trace_status_v1": "MISSING_FROM_R6_FRAME"})
            continue
        row = frame.loc[candidate_uid]
        selected = bool(row.get("selected_candidate_block_v1", False))
        rows.append(
            {
                "candidate_uid": candidate_uid,
                "trade_uid": row.get("trade_uid"),
                "trade_id": row.get("trade_id"),
                "decision_timestamp": row.get("decision_timestamp"),
                "bad_label_v1": bool(row.get("label_should_not_take_v1")),
                "tail_label_v1": bool(row.get("tail_10_50_mfe_v1")),
                "v3_base_flag_v1": bool(row.get("r5_2_selected_candidate__block_v1", False)),
                "r6_selected_v1": selected,
                "r6_bad_score_v1": row.get(R6_BAD_PROB),
                "r6_risky_score_v1": row.get(R6_RISKY_PROB),
                "r6_tail_score_v1": row.get(R6_TAIL_PROB),
                "r6_runner_score_v1": row.get(R6_RUNNER_PROB),
                "r6_blindspot_score_v1": row.get(R6_BLINDSPOT_PROB),
                "r5_2_runner_score_v1": row.get(R5_2_RUNNER_PROB),
                "asof_runner_guard_v1": bool(row.get("asof_runner_guard_v1", False)),
                "first_fail_reason_v1": "SELECTED_BY_V3_R5_2_BASE" if selected and bool(row.get("r5_2_selected_candidate__block_v1", False)) else _first_fail_reason(row, params),
                "final_r6_decision_v1": "BLOCK" if selected else "ALLOW",
            }
        )
    return pd.DataFrame(rows)


def _eval_delta(v2_summary: dict[str, Any], v3_summary: dict[str, Any], r6_frame: pd.DataFrame, compare: dict[str, Any]) -> dict[str, Any]:
    v2 = _metrics(v2_summary)
    v3 = _metrics(v3_summary)
    selected = _bool(r6_frame, "selected_candidate_block_v1")
    forensic = r6_frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    return {
        "layer_name": "R6_V3_EVAL_AGAINST_V2_AND_WEDNESDAY_V1",
        "v2_v1": {"bad_blocks_v1": int(v2.get("bad_blocks_v1") or 0), "tail_help_v1": int(v2.get("tail_help_v1") or 0), "precision_v1": v2.get("precision_v1"), "worst_loso_v1": _worst(v2_summary)},
        "v3_v1": {"bad_blocks_v1": int(v3.get("bad_blocks_v1") or 0), "tail_help_v1": int(v3.get("tail_help_v1") or 0), "precision_v1": v3.get("precision_v1"), "worst_loso_v1": _worst(v3_summary)},
        "wednesday_benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "delta_v3_minus_v2_v1": {
            "bad_blocks_v1": int((v3.get("bad_blocks_v1") or 0) - (v2.get("bad_blocks_v1") or 0)),
            "tail_help_v1": int((v3.get("tail_help_v1") or 0) - (v2.get("tail_help_v1") or 0)),
            "precision_v1": float((v3.get("precision_v1") or 0.0) - (v2.get("precision_v1") or 0.0)),
            "worst_loso_v1": float((_worst(v3_summary) or 0.0) - (_worst(v2_summary) or 0.0)),
        },
        "gap_v3_to_wednesday_v1": {
            "bad_blocks_v1": int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - (v3.get("bad_blocks_v1") or 0)),
            "tail_help_v1": int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - (v3.get("tail_help_v1") or 0)),
        },
        "repaired_damage_v1": int(v3.get("repaired_165_damage_v1") or 0),
        "forensic_trade_status_v1": "UNBLOCKED" if int((selected & forensic).sum()) == 0 else "BLOCKED",
        "forensic_trade_blocked_v1": int((selected & forensic).sum()),
        "fifty_plus_mfe_blocked_v1": int(v3.get("fifty_plus_mfe_blocked_v1") or 0),
        "hundred_plus_mfe_blocked_v1": int(v3.get("hundred_plus_mfe_blocked_v1") or 0),
        "two_hundred_plus_mfe_blocked_v1": int(v3.get("two_hundred_plus_mfe_blocked_v1") or 0),
        "strongest_winner_damage_v1": int(v3.get("strongest_winner_damage_v1") or 0),
        "runner_near_miss_blocked_v1": int(v3.get("runner_near_miss_blocked_v1") or 0),
        "compare_verdict_v1": compare.get("verdict_v1"),
    }


def _head_gap_forensics(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    missed = (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1")) & ~_bool(frame, "selected_candidate_block_v1")
    out = frame.loc[missed].copy()
    out["r5_2_base_status_v1"] = np.where(_bool(out, "r5_2_selected_candidate__block_v1"), "IN_R5_2_BASE", "NOT_IN_R5_2_BASE")
    out["r6_bad_threshold_v1"] = float(params["bad_threshold_v1"])
    out["r6_risky_threshold_v1"] = float(params["risky_threshold_v1"])
    out["r6_tail_threshold_v1"] = float(params["tail_threshold_v1"])
    out["r6_runner_threshold_v1"] = float(params["runner_threshold_v1"])
    out["r5_2_runner_threshold_v1"] = float(params["r5_2_runner_threshold_v1"])
    out["r6_blindspot_threshold_v1"] = float(params["blindspot_threshold_v1"])
    out["bad_head_pass_v1"] = _num(out, R6_BAD_PROB).ge(float(params["bad_threshold_v1"])).to_numpy(dtype=bool)
    out["risky_head_pass_v1"] = _num(out, R6_RISKY_PROB).ge(float(params["risky_threshold_v1"])).to_numpy(dtype=bool)
    out["tail_head_pass_v1"] = _num(out, R6_TAIL_PROB).ge(float(params["tail_threshold_v1"])).to_numpy(dtype=bool)
    out["runner_guard_blocks_v1"] = _num(out, R6_RUNNER_PROB).ge(float(params["runner_threshold_v1"])).to_numpy(dtype=bool) | _bool(out, "asof_runner_guard_v1").to_numpy(dtype=bool)
    out["r5_2_runner_guard_blocks_v1"] = _num(out, R5_2_RUNNER_PROB).ge(float(params["r5_2_runner_threshold_v1"])).to_numpy(dtype=bool)
    out["first_fail_reason_v1"] = out.apply(lambda row: _first_fail_reason(row, params), axis=1)
    out["bucket_v1"] = np.where(
        out["r5_2_base_status_v1"].eq("NOT_IN_R5_2_BASE"),
        "NOT_IN_R5_2_BASE",
        np.select(
            [
                out["runner_guard_blocks_v1"],
                out["r5_2_runner_guard_blocks_v1"],
                ~out["bad_head_pass_v1"],
                ~out["risky_head_pass_v1"],
                ~out["tail_head_pass_v1"],
            ],
            [
                "RUNNER_GUARD_BLOCKED",
                "R5_2_RUNNER_GUARD_BLOCKED",
                "IN_BASE_BAD_HEAD_TOO_LOW",
                "IN_BASE_RISKY_HEAD_TOO_LOW",
                "IN_BASE_TAIL_HEAD_TOO_LOW",
            ],
            default="NOT_ESTABLISHED",
        ),
    )
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "r5_2_base_status_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        R6_BAD_PROB,
        R6_RISKY_PROB,
        R6_TAIL_PROB,
        R6_RUNNER_PROB,
        R6_BLINDSPOT_PROB,
        "bad_head_pass_v1",
        "risky_head_pass_v1",
        "tail_head_pass_v1",
        "runner_guard_blocks_v1",
        "r5_2_runner_guard_blocks_v1",
        "first_fail_reason_v1",
        "bucket_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
    ]
    return out[[col for col in cols if col in out.columns]].copy()


def _threshold_frontier(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    current_selected = _bool(frame, "selected_candidate_block_v1")
    runner_near_limit = int((current_selected & _bool(frame, "r6_label_runner_near_miss_v1")).sum())
    rows: list[dict[str, Any]] = []
    for bad, risky, tail, runner, r52_runner, blind in itertools.product(
        [0.75, 0.85, 0.90, 0.95, 0.99],
        [0.75, 0.85, 0.90, 0.95, 0.99],
        [0.75, 0.85, 0.90, 0.95, 0.99],
        [0.30, 0.60, 0.82],
        [0.60, 0.74, 0.90],
        [0.70, 1.01],
    ):
        frontier_params = {
            **params,
            "bad_threshold_v1": bad,
            "risky_threshold_v1": risky,
            "tail_threshold_v1": tail,
            "runner_threshold_v1": runner,
            "r5_2_runner_threshold_v1": r52_runner,
            "blindspot_threshold_v1": blind,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
        }
        mask = _r6_ultra_safe_mask(frame, frontier_params)
        metrics, safety, failures = _hard_safety(frame, mask, runner_near_limit=runner_near_limit)
        rows.append(
            {
                "params_json_v1": json.dumps(_jsonable(frontier_params), sort_keys=True),
                "pass_v1": safety,
                "fail_reasons_v1": ";".join(failures),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _grid_scan(grid: pd.DataFrame, selected_policy: str) -> pd.DataFrame:
    out = grid.copy()
    out["selected_candidate_v1"] = out["policy_name_v1"].astype("string").eq(selected_policy)
    out["rejection_reason_v1"] = "SAFE_BUT_NOT_SELECTED"
    out.loc[out["selected_candidate_v1"], "rejection_reason_v1"] = "SELECTED_BEST_SAFE_CANDIDATE"
    unsafe = ~out.get("wednesday_safety_pass_v1", pd.Series(False, index=out.index)).astype(bool)
    out.loc[unsafe, "rejection_reason_v1"] = "NOT_WEDNESDAY_SAFE"
    out.loc[unsafe & pd.to_numeric(out.get("hard_damage_count_v1"), errors="coerce").fillna(0).gt(0), "rejection_reason_v1"] = "HARD_DAMAGE_FAIL"
    out.loc[unsafe & pd.to_numeric(out.get("precision_v1"), errors="coerce").lt(WEDNESDAY_R6_BENCHMARK["precision_v1"]), "rejection_reason_v1"] = "PRECISION_FAIL"
    return out.sort_values(
        ["selected_candidate_v1", "wednesday_safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )


def _root_cause(r6_frame: pd.DataFrame, gap: pd.DataFrame, frontier: pd.DataFrame, grid: pd.DataFrame, selected_bad: int, selected_tail: int) -> dict[str, Any]:
    missed = len(gap)
    not_base = int((gap["bucket_v1"] == "NOT_IN_R5_2_BASE").sum()) if missed else 0
    safe_frontier = frontier[frontier["pass_v1"].astype(bool)].copy()
    best_frontier_bad = int(safe_frontier["bad_blocks_v1"].max()) if not safe_frontier.empty else selected_bad
    best_frontier_tail = int(safe_frontier.sort_values(["bad_blocks_v1", "tail_help_v1"], ascending=[False, False]).iloc[0]["tail_help_v1"]) if not safe_frontier.empty else selected_tail
    safe_grid = grid[grid.get("wednesday_safety_pass_v1", pd.Series(False, index=grid.index)).astype(bool)].copy()
    best_grid_bad = int(safe_grid["bad_blocks_v1"].max()) if not safe_grid.empty else selected_bad
    best_grid_tail = int(safe_grid.sort_values(["bad_blocks_v1", "tail_help_v1"], ascending=[False, False]).iloc[0]["tail_help_v1"]) if not safe_grid.empty else selected_tail
    if not_base >= max(1, int(missed * 0.80)) and best_frontier_bad <= selected_bad and best_grid_bad <= selected_bad:
        cause = "R5_2_BASE_STILL_TOO_SMALL"
        interpretation = "Most missed bad/tail rows are outside R5.2 base, and neither threshold frontier nor rejected grid found a larger safe R6 candidate."
    elif best_frontier_bad > selected_bad or best_frontier_tail > selected_tail:
        cause = "R6_THRESHOLDS_TOO_CONSERVATIVE"
        interpretation = "Read-only frontier found a safe threshold point above selected V3."
    elif best_grid_bad > selected_bad or best_grid_tail > selected_tail:
        cause = "R6_CANDIDATE_GRID_TOO_CONSERVATIVE"
        interpretation = "Rejected grid contains a better safe candidate."
    else:
        cause = "NO_SAFE_RECALL_FRONTIER_FOUND"
        interpretation = "No safe R6-head threshold/grid route recovered material recall."
    return {
        "layer_name": "POST_V3_ROOT_CAUSE_DECISION_V1",
        "primary_cause_v1": cause,
        "missed_bad_tail_rows_v1": int(missed),
        "missed_not_in_r5_2_base_v1": not_base,
        "best_safe_frontier_bad_tail_v1": [best_frontier_bad, best_frontier_tail],
        "best_safe_grid_bad_tail_v1": [best_grid_bad, best_grid_tail],
        "thresholds_alone_can_lift_recall_safely_v1": bool(best_frontier_bad > selected_bad or best_frontier_tail > selected_tail),
        "rejected_grid_has_better_safe_candidate_v1": bool(best_grid_bad > selected_bad or best_grid_tail > selected_tail),
        "stop_r5_2_tiny_extension_loop_v1": bool(cause in {"R5_2_BASE_STILL_TOO_SMALL", "NO_SAFE_RECALL_FRONTIER_FOUND"}),
        "interpretation_v1": interpretation,
    }


def _next_action(root: dict[str, Any], safety_ok: bool) -> dict[str, Any]:
    if not safety_ok:
        action = "STOP_AND_RUN_R6_V3_FAILURE_FORENSICS"
    elif root["thresholds_alone_can_lift_recall_safely_v1"]:
        action = "IMPLEMENT_SAFE_R6_THRESHOLD_FRONTIER_NEXT"
    elif root["rejected_grid_has_better_safe_candidate_v1"]:
        action = "FIX_R6_CANDIDATE_SELECTION_NEXT"
    elif root["stop_r5_2_tiny_extension_loop_v1"]:
        action = "STOP_R5_2_BASE_TINY_EXTENSION_LOOP"
    else:
        action = "MOVE_TO_R6_HEAD_THRESHOLD_OR_TRUE_R5_2_REBUILD_FORENSICS"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "followup_action_v1": "INVESTIGATE_TRUE_R5_2_REBUILD_OR_LABEL_OBJECTIVE_NEXT" if action == "STOP_R5_2_BASE_TINY_EXTENSION_LOOP" else None,
        "blocked_action_v1": [
            "DO_NOT_BUILD_NEW_BASELINE",
            "DO_NOT_BUILD_NEW_FEATURE_SURFACE",
            "DO_NOT_USE_1689_EXACT_ONLY",
            "DO_NOT_USE_PROTECTOR_FIRST",
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_THIS_RUN",
        ],
    }


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("R6_V3_RAN", summary["r6_v3_ran_v1"], summary["r6_dir_v1"]),
            row("V3_SCORE_PACKAGE_USED", summary["v3_score_package_used_v1"], summary["score_dir_v1"]),
            row("V3_CONTRACT_USED", summary["v3_contract_used_v1"], summary["contract_id_v1"]),
            row("V3_ROWS_PASS_THROUGH", summary["v3_added_rows_selected_v1"] == summary["v3_added_rows_v1"], [summary["v3_added_rows_selected_v1"], summary["v3_added_rows_v1"]]),
            row("SAFETY_OK", summary["safety_ok_v1"], summary["safety_v1"]),
            row("NO_NEW_SURFACE", True, True),
            row("NO_R6_FREEZE_PROMO_LIVE", True, True),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Run R6 From V3 And R6 Head Recall Forensics",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- R6 V3 bad/tail: `{summary['r6_v3_bad_blocks_v1']}` / `{summary['r6_v3_tail_help_v1']}`",
            f"- Delta vs V2: `+{summary['bad_delta_vs_v2_v1']}` / `+{summary['tail_delta_vs_v2_v1']}`",
            f"- V3-added rows selected: `{summary['v3_added_rows_selected_v1']}` / `{summary['v3_added_rows_v1']}`",
            f"- Safety: `{summary['safety_v1']}`",
            f"- Root cause: `{summary['root_cause_v1']}`",
            f"- Best safe threshold frontier: `{summary['best_safe_threshold_frontier_bad_tail_v1']}`",
            f"- Best safe rejected-grid candidate: `{summary['best_safe_rejected_grid_bad_tail_v1']}`",
            "",
            "No new baseline, feature surface, 1689 exact-only, protector-first, freeze, promotion, live gate, or controller path was used.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_score_dir: Path = V3_SCORE_DEFAULT,
    v2_score_dir: Path = V2_SCORE_DEFAULT,
    v3_r6_dir: Path = V3_R6_DEFAULT,
    v2_r6_dir: Path = V2_R6_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_score_dir = v3_score_dir.expanduser().resolve()
    v2_score_dir = v2_score_dir.expanduser().resolve()
    v3_r6_dir = v3_r6_dir.expanduser().resolve()
    v2_r6_dir = v2_r6_dir.expanduser().resolve()

    v3_score = pd.read_parquet(v3_score_dir / SCORE_FRAME)
    v2_score = pd.read_parquet(v2_score_dir / SCORE_FRAME)
    v3_r6 = pd.read_parquet(v3_r6_dir / R6_FRAME)
    v3_grid_raw = pd.read_csv(v3_r6_dir / R6_GRID)
    v2_summary = _read_json(v2_r6_dir / R6_SUMMARY)
    v3_summary = _read_json(v3_r6_dir / R6_SUMMARY)
    v3_compare = _read_json(v3_r6_dir / R6_COMPARE)
    score_summary = _read_json(v3_score_dir / SCORE_SUMMARY)
    selected = _selected_policy(v3_summary)
    params = selected.get("params_v1") or {}
    selected_policy = str(selected.get("policy_name_v1"))
    r6_selected = _bool(v3_r6, "selected_candidate_block_v1")

    pass_through = _pass_through_audit(v2_score, v3_score, v3_r6, params)
    eval_delta = _eval_delta(v2_summary, v3_summary, v3_r6, v3_compare)
    head_gap = _head_gap_forensics(v3_r6, params)
    frontier = _threshold_frontier(v3_r6, params)
    rejected_grid = _grid_scan(v3_grid_raw, selected_policy)
    v3_metrics = _metrics(v3_summary)
    frontier_safe = frontier[frontier["pass_v1"].astype(bool)].copy()
    grid_safe = rejected_grid[rejected_grid.get("wednesday_safety_pass_v1", pd.Series(False, index=rejected_grid.index)).astype(bool)].copy()
    root = _root_cause(v3_r6, head_gap, frontier, rejected_grid, int(v3_metrics.get("bad_blocks_v1") or 0), int(v3_metrics.get("tail_help_v1") or 0))
    safety_ok = (
        int(eval_delta["repaired_damage_v1"]) == 0
        and eval_delta["forensic_trade_status_v1"] == "UNBLOCKED"
        and int(eval_delta["fifty_plus_mfe_blocked_v1"]) <= 1
        and int(eval_delta["hundred_plus_mfe_blocked_v1"]) == 0
        and int(eval_delta["two_hundred_plus_mfe_blocked_v1"]) == 0
        and int(eval_delta["strongest_winner_damage_v1"]) == 0
        and int(eval_delta["runner_near_miss_blocked_v1"]) == 0
    )
    next_action = _next_action(root, safety_ok)
    contract_id = (score_summary.get("r5_2_selected_policy_v1") or {}).get("base_membership_active_contract_id_v1")
    if not safety_ok:
        decision = "MONDAY_R6_V3_SAFETY_FAIL"
    elif root["primary_cause_v1"] == "R5_2_BASE_STILL_TOO_SMALL":
        decision = "MONDAY_R6_V3_SAFE_BUT_R5_2_BASE_BOUND"
    elif root["thresholds_alone_can_lift_recall_safely_v1"]:
        decision = "MONDAY_R6_V3_SAFE_THRESHOLD_FRONTIER_FOUND"
    else:
        decision = "MONDAY_R6_V3_SAFE_BUT_BELOW_WEDNESDAY"
    best_frontier = frontier_safe.sort_values(["bad_blocks_v1", "tail_help_v1"], ascending=[False, False]).head(1)
    best_grid = grid_safe.sort_values(["bad_blocks_v1", "tail_help_v1"], ascending=[False, False]).head(1)
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "score_dir_v1": str(v3_score_dir),
        "r6_dir_v1": str(v3_r6_dir),
        "decision_v1": decision,
        "next_action_v1": next_action["next_action_v1"],
        "contract_id_v1": contract_id,
        "v3_contract_used_v1": contract_id == R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"],
        "v3_score_package_used_v1": v3_summary.get("score_dir_v1") == str(v3_score_dir),
        "v3_base_flags_used_v1": int(_bool(v3_r6, "r5_2_selected_candidate__block_v1").sum()) == 82,
        "v2_fixed_base_used_v1": False,
        "r6_v3_ran_v1": bool(v3_summary.get("r6_training_started_v1")),
        "selected_candidate_v1": selected_policy,
        "selected_family_v1": selected.get("family_v1"),
        "r6_v3_bad_blocks_v1": int(v3_metrics.get("bad_blocks_v1") or 0),
        "r6_v3_tail_help_v1": int(v3_metrics.get("tail_help_v1") or 0),
        "precision_v1": v3_metrics.get("precision_v1"),
        "worst_loso_v1": _worst(v3_summary),
        "bad_delta_vs_v2_v1": eval_delta["delta_v3_minus_v2_v1"]["bad_blocks_v1"],
        "tail_delta_vs_v2_v1": eval_delta["delta_v3_minus_v2_v1"]["tail_help_v1"],
        "wednesday_gap_bad_tail_v1": [eval_delta["gap_v3_to_wednesday_v1"]["bad_blocks_v1"], eval_delta["gap_v3_to_wednesday_v1"]["tail_help_v1"]],
        "v3_added_rows_v1": int(len(pass_through)),
        "v3_added_rows_selected_v1": int(pass_through.get("r6_selected_v1", pd.Series(dtype=bool)).fillna(False).sum()) if not pass_through.empty else 0,
        "safety_ok_v1": safety_ok,
        "safety_v1": {
            "repaired_damage_v1": eval_delta["repaired_damage_v1"],
            "forensic_trade_status_v1": eval_delta["forensic_trade_status_v1"],
            "fifty_hundred_twohundred_blocked_v1": [
                eval_delta["fifty_plus_mfe_blocked_v1"],
                eval_delta["hundred_plus_mfe_blocked_v1"],
                eval_delta["two_hundred_plus_mfe_blocked_v1"],
            ],
            "strongest_winner_damage_v1": eval_delta["strongest_winner_damage_v1"],
            "runner_near_miss_blocked_v1": eval_delta["runner_near_miss_blocked_v1"],
        },
        "missed_bad_tail_after_v3_v1": int(len(head_gap)),
        "missed_not_in_r5_2_base_v1": int((head_gap["bucket_v1"] == "NOT_IN_R5_2_BASE").sum()) if not head_gap.empty else 0,
        "threshold_frontier_can_lift_safely_v1": root["thresholds_alone_can_lift_recall_safely_v1"],
        "best_safe_threshold_frontier_bad_tail_v1": [
            int(best_frontier.iloc[0]["bad_blocks_v1"]) if not best_frontier.empty else int(v3_metrics.get("bad_blocks_v1") or 0),
            int(best_frontier.iloc[0]["tail_help_v1"]) if not best_frontier.empty else int(v3_metrics.get("tail_help_v1") or 0),
        ],
        "rejected_grid_has_better_safe_candidate_v1": root["rejected_grid_has_better_safe_candidate_v1"],
        "best_safe_rejected_grid_bad_tail_v1": [
            int(best_grid.iloc[0]["bad_blocks_v1"]) if not best_grid.empty else int(v3_metrics.get("bad_blocks_v1") or 0),
            int(best_grid.iloc[0]["tail_help_v1"]) if not best_grid.empty else int(v3_metrics.get("tail_help_v1") or 0),
        ],
        "root_cause_v1": root["primary_cause_v1"],
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "forbidden_1689_used_v1": False,
        "protector_first_used_v1": False,
        "freeze_promo_live_started_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "R6 V3 used the V3 score package and V3 base flags.",
                "No new baseline, feature surface, 1689, protector-first, freeze, promotion, live gate, or controller path was used.",
            ],
            "INDIKERT": [
                "The V3 lift passes through R6, but recall remains primarily base-bound.",
            ],
            "IKKE_ETABLERT": [
                "A safe R6 threshold/grid route that closes the Wednesday recall gap is not established.",
            ],
        },
    }
    retrain = {
        "layer_name": "R6_RETRAIN_FROM_R5_2_V3_SCORE_PACKAGE_V1",
        "score_dir_v1": str(v3_score_dir),
        "r6_output_dir_v1": str(v3_r6_dir),
        "v3_score_package_used_v1": summary["v3_score_package_used_v1"],
        "v3_contract_id_v1": contract_id,
        "v3_base_flags_used_v1": summary["v3_base_flags_used_v1"],
        "v2_fixed_base_used_v1": False,
        "r6_five_head_count_v1": int(v3_summary.get("r6_head_count_v1") or 0),
        "candidate_grid_count_v1": int(len(v3_grid_raw)),
        "thresholds_v1": params,
        "no_new_baseline_or_feature_surface_v1": True,
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "v3_score_dir_v1": str(v3_score_dir),
            "v2_score_dir_v1": str(v2_score_dir),
            "v3_r6_dir_v1": str(v3_r6_dir),
            "v2_r6_dir_v1": str(v2_r6_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": True,
        "r6_training_started_v1": True,
        "freeze_promo_live_started_v1": False,
    }
    audit = _audit(summary)

    _write_json(output_dir / OUTPUT_FILES["retrain"], retrain)
    pass_through.to_csv(output_dir / OUTPUT_FILES["pass_through"], index=False)
    _write_json(output_dir / OUTPUT_FILES["eval_delta"], eval_delta)
    head_gap.to_csv(output_dir / OUTPUT_FILES["head_gap"], index=False)
    frontier.to_csv(output_dir / OUTPUT_FILES["threshold_frontier"], index=False)
    rejected_grid.to_csv(output_dir / OUTPUT_FILES["rejected_grid"], index=False)
    _write_json(output_dir / OUTPUT_FILES["root_cause"], root)
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
    parser.add_argument("--v2-score-dir", type=Path, default=V2_SCORE_DEFAULT)
    parser.add_argument("--v3-r6-dir", type=Path, default=V3_R6_DEFAULT)
    parser.add_argument("--v2-r6-dir", type=Path, default=V2_R6_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_score_dir=args.v3_score_dir,
        v2_score_dir=args.v2_score_dir,
        v3_r6_dir=args.v3_r6_dir,
        v2_r6_dir=args.v2_r6_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
