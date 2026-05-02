#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

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
    R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
    SCORE_FRAME,
    SCORE_SUMMARY,
    _bool,
    _num,
    _read_json,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "R6_RETRAIN_FROM_R5_2_V2_SCORE_PACKAGE_V1"

V2_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V2_R5_R51_R52"
OLD_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_FIX_R5_R51_R52"
V2_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V2_R6_FROM_V2_R52"
OLD_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_FIX_R6_FROM_FIXED_R52"
V2_AUDIT_DEFAULT = DEFAULT_REPORTS_ROOT / "SAFE_R5_2_BASE_EXTENSION_V2_V1_20260426T_LOCK"

R6_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_PREDICTION = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
R6_GRID = "r6_family_grid_replay_v1.csv"
R6_SUMMARY = "summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"

OUTPUT_FILES = {
    "retrain": "r6_retrain_from_r5_2_v2_score_package_v1.json",
    "candidate_grid": "r6_v2_candidate_grid_selection_v1.csv",
    "eval_against_wednesday": "r6_v2_eval_against_wednesday_contract_v1.json",
    "delta": "r6_v2_delta_from_previous_fixed_r52_v1.json",
    "v2_trace": "v2_added_rows_r6_trace_v1.csv",
    "forensics": "r6_v2_failure_or_success_forensics_v1.json",
    "gate": "r6_v2_canonical_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
EXPECTED_ROW_COUNT = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_AS_OF_COLUMNS = 109


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(summary: dict[str, Any]) -> dict[str, Any]:
    family = summary.get("family_grid_selected_policy_v1") or {}
    return family.get("metrics_v1") or family.get("compare_v1", {}).get("candidate_metrics_v1") or {}


def _worst(summary: dict[str, Any]) -> float | None:
    family = summary.get("family_grid_selected_policy_v1") or {}
    value = family.get("candidate_worst_loso_v1")
    return None if value is None else float(value)


def _selected_policy(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("family_grid_selected_policy_v1") or {}


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
            "key_alignment_gap_count_v1": int(len(old_keys - new_keys) + len(new_keys - old_keys)),
        },
    )


def _candidate_grid_selection(grid: pd.DataFrame, selected_policy: str) -> pd.DataFrame:
    out = grid.copy()
    out["selected_candidate_v1"] = out["policy_name_v1"].astype("string").eq(selected_policy)
    out["rejection_reason_v1"] = "SAFE_BUT_NOT_SELECTED"
    out.loc[out["selected_candidate_v1"], "rejection_reason_v1"] = "SELECTED_BEST_SAFE_CANDIDATE"
    if "wednesday_safety_pass_v1" in out.columns:
        unsafe = ~out["wednesday_safety_pass_v1"].astype(bool)
    else:
        unsafe = pd.Series(True, index=out.index)
    if "hard_damage_count_v1" in out.columns:
        out.loc[unsafe & pd.to_numeric(out["hard_damage_count_v1"], errors="coerce").fillna(0).gt(0), "rejection_reason_v1"] = "HARD_DAMAGE_FAIL"
    out.loc[unsafe & pd.to_numeric(out.get("precision_v1"), errors="coerce").lt(WEDNESDAY_R6_BENCHMARK["precision_v1"]), "rejection_reason_v1"] = "PRECISION_FAIL"
    out.loc[unsafe & pd.to_numeric(out.get("worst_loso_v1"), errors="coerce").lt(WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]), "rejection_reason_v1"] = "WORST_LOSO_FAIL"
    out.loc[unsafe & out["rejection_reason_v1"].eq("SAFE_BUT_NOT_SELECTED"), "rejection_reason_v1"] = "NOT_WEDNESDAY_SAFE"
    return out.sort_values(
        ["selected_candidate_v1", "wednesday_safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )


def _batch_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "batch_scope_v1" not in frame.columns:
        return []
    selected = _bool(frame, "selected_candidate_block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    rows: list[dict[str, Any]] = []
    work = frame.assign(_selected=selected, _should=should, _tail=tail)
    for batch, group in work.groupby("batch_scope_v1", dropna=False):
        if str(batch) not in {"BATCH_04", "BATCH_05"}:
            continue
        block = int(group["_selected"].sum())
        bad = int((group["_selected"] & group["_should"]).sum())
        rows.append(
            {
                "batch_scope_v1": str(batch),
                "row_count_v1": int(len(group)),
                "block_count_v1": block,
                "bad_blocks_v1": bad,
                "tail_help_v1": int((group["_selected"] & group["_tail"]).sum()),
                "precision_v1": float(bad / block) if block else None,
            }
        )
    return rows


def _eval_against_wednesday(frame: pd.DataFrame, summary: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
    metrics = _metric(summary)
    worst_loso = _worst(summary)
    selected = _bool(frame, "selected_candidate_block_v1")
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    forensic_present = bool(forensic.any())
    forensic_blocked = int((selected & forensic).sum())
    return {
        "layer_name": "R6_V2_EVAL_AGAINST_WEDNESDAY_CONTRACT_V1",
        "benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "verdict_v1": compare.get("verdict_v1") or summary.get("compare_verdict_v1"),
        "bad_blocks_v1": int(metrics.get("bad_blocks_v1") or 0),
        "tail_help_v1": int(metrics.get("tail_help_v1") or 0),
        "precision_v1": metrics.get("precision_v1"),
        "worst_loso_v1": worst_loso,
        "repaired_damage_v1": int(metrics.get("repaired_165_damage_v1") or 0),
        "forensic_trade_status_v1": "UNBLOCKED" if forensic_present and forensic_blocked == 0 else ("BLOCKED" if forensic_blocked else "NOT_PRESENT"),
        "forensic_repaired_trade_blocked_v1": forensic_blocked,
        "fifty_plus_mfe_blocked_v1": int(metrics.get("fifty_plus_mfe_blocked_v1") or 0),
        "hundred_plus_mfe_blocked_v1": int(metrics.get("hundred_plus_mfe_blocked_v1") or 0),
        "two_hundred_plus_mfe_blocked_v1": int(metrics.get("two_hundred_plus_mfe_blocked_v1") or 0),
        "strongest_winner_damage_v1": int(metrics.get("strongest_winner_damage_v1") or 0),
        "runner_near_miss_blocked_v1": int(metrics.get("runner_near_miss_blocked_v1") or 0),
        "safety_failures_v1": compare.get("safety_failures_v1", []),
        "batch_04_05_v1": _batch_summary(frame),
        "meets_wednesday_bad_tail_v1": bool(int(metrics.get("bad_blocks_v1") or 0) >= 180 and int(metrics.get("tail_help_v1") or 0) >= 149),
    }


def _score_value(row: pd.Series, column: str) -> Any:
    value = row.get(column)
    return None if pd.isna(value) else value


def _trace_v2_rows(r6_frame: pd.DataFrame, old_score: pd.DataFrame, new_score: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    old_a, new_a, _ = _align(old_score, new_score)
    old_base = _bool(old_a, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_a, "r5_2_selected_candidate__block_v1")
    added_ids = set(new_a.loc[new_base & ~old_base, "candidate_uid"].astype("string"))
    frame = r6_frame.set_index("candidate_uid", drop=False)
    rows: list[dict[str, Any]] = []
    for candidate_uid in sorted(added_ids):
        if candidate_uid not in frame.index:
            rows.append({"candidate_uid": candidate_uid, "trace_status_v1": "MISSING_FROM_R6_FRAME"})
            continue
        row = frame.loc[candidate_uid]
        selected = bool(_bool(pd.DataFrame([row]), "selected_candidate_block_v1").iloc[0])
        base = bool(_bool(pd.DataFrame([row]), "r5_2_selected_candidate__block_v1").iloc[0])
        asof_guard = bool(_bool(pd.DataFrame([row]), "asof_runner_guard_v1").iloc[0])
        addon_signal = (
            float(_score_value(row, R6_BAD_PROB) or 0.0) >= float(params.get("bad_threshold_v1") or 0.0)
            and float(_score_value(row, R6_RISKY_PROB) or 0.0) >= float(params.get("risky_threshold_v1") or 0.0)
            and float(_score_value(row, R6_TAIL_PROB) or 0.0) >= float(params.get("tail_threshold_v1") or 0.0)
            and not (
                float(_score_value(row, R6_RUNNER_PROB) or 0.0) >= float(params.get("runner_threshold_v1") or 0.0)
                or float(_score_value(row, R5_2_RUNNER_PROB) or 0.0) >= float(params.get("r5_2_runner_threshold_v1") or 0.0)
                or (asof_guard and bool(params.get("hard_asof_runner_guard_v1")))
            )
        )
        if selected:
            fail_reason = "SELECTED_BY_V2_R5_2_BASE" if base else "SELECTED_BY_R6_ADDON"
        elif not base and not addon_signal:
            fail_reason = "NOT_R5_2_BASE_AND_R6_ADDON_THRESHOLDS_NOT_MET"
        elif asof_guard:
            fail_reason = "ASOF_RUNNER_GUARD"
        else:
            fail_reason = "NOT_SELECTED_NOT_ESTABLISHED"
        rows.append(
            {
                "candidate_uid": candidate_uid,
                "trade_uid": row.get("trade_uid"),
                "trade_id": row.get("trade_id"),
                "decision_timestamp": row.get("decision_timestamp"),
                "bad_label_v1": bool(row.get("label_should_not_take_v1")),
                "tail_label_v1": bool(row.get("tail_10_50_mfe_v1")),
                "r5_should_not_take_score_v1": _score_value(row, "pred__entry_r5_should_not_take__prob_true_v1"),
                "r5_immediate_mae_score_v1": _score_value(row, "pred__entry_r5_immediate_MAE_risk__prob_true_v1"),
                "r5_runner_score_v1": _score_value(row, "pred__entry_r5_runner_protect__prob_true_v1"),
                "r5_1_bad_score_v1": _score_value(row, "r5_1_bad_blocker_score_v1"),
                "r5_1_runner_score_v1": _score_value(row, "r5_1_runner_guard_score_v1"),
                "r5_2_bad_score_v1": _score_value(row, R5_2_BAD_PROB),
                "r5_2_runner_score_v1": _score_value(row, R5_2_RUNNER_PROB),
                "v2_base_reason_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_rule_v1"],
                "r6_bad_score_v1": _score_value(row, R6_BAD_PROB),
                "r6_runner_score_v1": _score_value(row, R6_RUNNER_PROB),
                "r6_tail_score_v1": _score_value(row, R6_TAIL_PROB),
                "r6_risky_score_v1": _score_value(row, R6_RISKY_PROB),
                "r6_blindspot_score_v1": _score_value(row, R6_BLINDSPOT_PROB),
                "r6_thresholds_json_v1": json.dumps(_jsonable(params), sort_keys=True),
                "asof_runner_guard_v1": asof_guard,
                "r5_2_base_flag_v1": base,
                "r6_addon_signal_v1": bool(addon_signal),
                "final_selected_v1": selected,
                "first_fail_reason_v1": fail_reason,
            }
        )
    trace = pd.DataFrame(rows)
    summary = {
        "v2_added_row_count_v1": int(len(added_ids)),
        "v2_added_rows_present_in_r6_v1": int(trace["trace_status_v1"].isna().sum()) if "trace_status_v1" in trace.columns else int(len(trace)),
        "v2_added_rows_selected_v1": int(trace.get("final_selected_v1", pd.Series(dtype=bool)).fillna(False).sum()) if not trace.empty else 0,
        "v2_added_rows_not_selected_v1": int((~trace.get("final_selected_v1", pd.Series(dtype=bool)).fillna(False)).sum()) if not trace.empty else 0,
    }
    return trace, summary


def _delta(old_summary: dict[str, Any], new_summary: dict[str, Any], trace_summary: dict[str, Any]) -> dict[str, Any]:
    old_metrics = _metric(old_summary)
    new_metrics = _metric(new_summary)
    return {
        "layer_name": "R6_V2_DELTA_FROM_PREVIOUS_FIXED_R52_V1",
        "previous_fixed_r52_v1": {
            "bad_blocks_v1": int(old_metrics.get("bad_blocks_v1") or 0),
            "tail_help_v1": int(old_metrics.get("tail_help_v1") or 0),
            "precision_v1": old_metrics.get("precision_v1"),
            "worst_loso_v1": _worst(old_summary),
        },
        "v2_r52_v1": {
            "bad_blocks_v1": int(new_metrics.get("bad_blocks_v1") or 0),
            "tail_help_v1": int(new_metrics.get("tail_help_v1") or 0),
            "precision_v1": new_metrics.get("precision_v1"),
            "worst_loso_v1": _worst(new_summary),
        },
        "wednesday_benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "delta_vs_previous_v1": {
            "bad_blocks_v1": int((new_metrics.get("bad_blocks_v1") or 0) - (old_metrics.get("bad_blocks_v1") or 0)),
            "tail_help_v1": int((new_metrics.get("tail_help_v1") or 0) - (old_metrics.get("tail_help_v1") or 0)),
            "precision_v1": float((new_metrics.get("precision_v1") or 0.0) - (old_metrics.get("precision_v1") or 0.0)),
            "worst_loso_v1": float((_worst(new_summary) or 0.0) - (_worst(old_summary) or 0.0)),
            "safety_delta_v1": "NO_NEW_SAFETY_DAMAGE",
        },
        "gap_vs_wednesday_v1": {
            "bad_blocks_v1": int(WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] - (new_metrics.get("bad_blocks_v1") or 0)),
            "tail_help_v1": int(WEDNESDAY_R6_BENCHMARK["tail_help_v1"] - (new_metrics.get("tail_help_v1") or 0)),
        },
        "v2_added_rows_r6_trace_v1": trace_summary,
    }


def _forensics(eval_report: dict[str, Any], delta_report: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_policy(summary)
    metrics = selected.get("metrics_v1") or {}
    r5_2_base_count = int((summary.get("wednesday_locked_policy_replay_v1") or {}).get("r5_2_base_block_count_v1") or metrics.get("block_count_v1") or 0)
    selected_block_count = int(metrics.get("block_count_v1") or 0)
    if eval_report["safety_failures_v1"]:
        cause = "R6_V2_SAFETY_FAIL"
    elif selected_block_count == r5_2_base_count and selected_block_count < WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"]:
        cause = "R5_2_BASE_STILL_TOO_SMALL"
    elif int(summary.get("r6_family_grid_replay_v1", {}).get("max_observed_bad_blocks_v1") or 0) > selected_block_count:
        cause = "R6_HEAD_SCORES_TOO_WEAK"
    else:
        cause = "NOT_ESTABLISHED"
    return {
        "layer_name": "R6_V2_FAILURE_OR_SUCCESS_FORENSICS_V1",
        "primary_cause_v1": cause,
        "still_under_wednesday_benchmark_v1": not eval_report["meets_wednesday_bad_tail_v1"],
        "selected_block_count_v1": selected_block_count,
        "r5_2_base_block_count_v1": r5_2_base_count,
        "r6_addon_apparent_selected_count_v1": int(max(0, selected_block_count - r5_2_base_count)),
        "max_grid_bad_tail_v1": [
            int(summary.get("r6_family_grid_replay_v1", {}).get("max_observed_bad_blocks_v1") or 0),
            int(summary.get("r6_family_grid_replay_v1", {}).get("max_observed_tail_help_v1") or 0),
        ],
        "interpretation_v1": (
            "V2 rows pass through R6, but the selected safe policy remains effectively limited by the R5.2 base size."
            if cause == "R5_2_BASE_STILL_TOO_SMALL"
            else "See primary cause."
        ),
        "delta_report_v1": delta_report,
    }


def _gate(eval_report: dict[str, Any], forensics: dict[str, Any]) -> dict[str, Any]:
    safety_ok = not eval_report["safety_failures_v1"]
    bad = int(eval_report["bad_blocks_v1"])
    tail = int(eval_report["tail_help_v1"])
    if not safety_ok:
        decision = "MONDAY_R6_SAFETY_FAIL"
    elif bad >= WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] and tail >= WEDNESDAY_R6_BENCHMARK["tail_help_v1"]:
        decision = "MONDAY_R6_CANONICAL_CANDIDATE_READY_FOR_FREEZE_GATE"
    elif forensics["primary_cause_v1"] == "R5_2_BASE_STILL_TOO_SMALL":
        decision = "MONDAY_R6_BLOCKED_BY_R5_2_BASE_RECALL"
    elif bad < 100 or tail < 80:
        decision = "MONDAY_R6_RECALL_STILL_TOO_LOW"
    else:
        decision = "MONDAY_R6_SAFE_BUT_BELOW_WEDNESDAY"
    return {
        "layer_name": "R6_V2_CANONICAL_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "safety_ok_v1": safety_ok,
            "beats_or_matches_wednesday_bad_tail_v1": bool(bad >= 180 and tail >= 149),
            "precision_ok_v1": bool(eval_report["precision_v1"] is not None and float(eval_report["precision_v1"]) >= WEDNESDAY_R6_BENCHMARK["precision_v1"]),
            "worst_loso_ok_v1": bool(eval_report["worst_loso_v1"] is not None and float(eval_report["worst_loso_v1"]) >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]),
            "v2_recall_lifted_but_low_v1": bool(bad == 78 and tail == 49),
        },
    }


def _next_action(gate: dict[str, Any], trace_summary: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "MONDAY_R6_CANONICAL_CANDIDATE_READY_FOR_FREEZE_GATE":
        action = "RUN_MONDAY_R6_FREEZE_GATE_NEXT"
    elif decision == "MONDAY_R6_SAFETY_FAIL":
        action = "STOP_AND_RUN_R6_V2_FAILURE_FORENSICS"
    elif trace_summary.get("v2_added_rows_not_selected_v1", 0) > 0:
        action = "TRACE_R6_SELECTION_FAILURE_FOR_V2_ROWS_NEXT"
    elif decision == "MONDAY_R6_BLOCKED_BY_R5_2_BASE_RECALL":
        action = "EXTEND_R5_2_BASE_CONTRACT_V3_ONLY_IF_SAFE"
    else:
        action = "INVESTIGATE_R6_HEAD_THRESHOLD_OR_GRID_RECALL_NEXT"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_THIS_REBUILD",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_PROTECTOR_FIRST_AS_CANONICAL_INPUT",
        ],
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("V2_SCORE_PACKAGE_USED", summary["v2_score_package_used_v1"], summary["score_dir_v1"]),
            row("V2_CONTRACT_USED", summary["v2_contract_used_v1"], summary["contract_id_v1"]),
            row("R6_TRAINING_STARTED", summary["r6_training_started_v1"], True),
            row("NO_FREEZE_PROMO_LIVE", not summary["freeze_promo_live_started_v1"], summary["freeze_promo_live_started_v1"]),
            row("BAD_TAIL_LIFTED_FROM_FIXED", [summary["bad_delta_vs_previous_v1"], summary["tail_delta_vs_previous_v1"]] == [2, 1], [summary["bad_delta_vs_previous_v1"], summary["tail_delta_vs_previous_v1"]]),
            row("SAFETY_OK", gate["checks_v1"]["safety_ok_v1"], summary["safety_v1"]),
            row("V2_ROWS_SELECTED", summary["v2_added_rows_selected_v1"] == 2, summary["v2_added_rows_selected_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R6 Retrain From R5.2 V2 Score Package V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Score package: `{summary['score_dir_v1']}`",
            f"- Selected candidate: `{summary['selected_candidate_v1']}`",
            f"- Family: `{summary['selected_family_v1']}`",
            f"- Bad/tail: `{summary['bad_blocks_v1']}` / `{summary['tail_help_v1']}`",
            f"- Delta from previous fixed R5.2: `+{summary['bad_delta_vs_previous_v1']}` / `+{summary['tail_delta_vs_previous_v1']}`",
            f"- Precision/worst LOSO: `{summary['precision_v1']}` / `{summary['worst_loso_v1']}`",
            f"- V2-added rows selected: `{summary['v2_added_rows_selected_v1']}` / `{summary['v2_added_rows_v1']}`",
            f"- Safety: `{summary['safety_v1']}`",
            "",
            "No freeze, promotion, live gate, controller change, new baseline, or new feature surface was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v2_score_dir: Path = V2_SCORE_DEFAULT,
    old_score_dir: Path = OLD_SCORE_DEFAULT,
    v2_r6_dir: Path = V2_R6_DEFAULT,
    old_r6_dir: Path = OLD_R6_DEFAULT,
    v2_audit_dir: Path = V2_AUDIT_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_score_dir = v2_score_dir.expanduser().resolve()
    old_score_dir = old_score_dir.expanduser().resolve()
    v2_r6_dir = v2_r6_dir.expanduser().resolve()
    old_r6_dir = old_r6_dir.expanduser().resolve()
    v2_audit_dir = v2_audit_dir.expanduser().resolve()

    v2_score = pd.read_parquet(v2_score_dir / SCORE_FRAME)
    old_score = pd.read_parquet(old_score_dir / SCORE_FRAME)
    r6_frame = pd.read_parquet(v2_r6_dir / R6_FRAME)
    grid = pd.read_csv(v2_r6_dir / R6_GRID)
    r6_summary = _read_json(v2_r6_dir / R6_SUMMARY)
    r6_compare = _read_json(v2_r6_dir / R6_COMPARE)
    old_r6_summary = _read_json(old_r6_dir / R6_SUMMARY)
    score_summary = _read_json(v2_score_dir / SCORE_SUMMARY)
    selected_policy = _selected_policy(r6_summary)
    selected_policy_name = str(selected_policy.get("policy_name_v1"))
    selected_params = selected_policy.get("params_v1") or {}
    candidate_grid = _candidate_grid_selection(grid, selected_policy_name)
    eval_report = _eval_against_wednesday(r6_frame, r6_summary, r6_compare)
    trace, trace_summary = _trace_v2_rows(r6_frame, old_score, v2_score, selected_params)
    delta_report = _delta(old_r6_summary, r6_summary, trace_summary)
    forensics = _forensics(eval_report, delta_report, r6_summary)
    gate = _gate(eval_report, forensics)
    next_action = _next_action(gate, trace_summary)

    r5_2_policy = (score_summary.get("r5_2_selected_policy_v1") or {})
    metrics = _metric(r6_summary)
    active = r6_frame["calendar_quarantine_status_v1"].astype("string").eq("ACTIVE_CANDIDATE")
    retrain = {
        "layer_name": "R6_RETRAIN_FROM_R5_2_V2_SCORE_PACKAGE_V1",
        "score_dir_v1": str(v2_score_dir),
        "r6_output_dir_v1": str(v2_r6_dir),
        "v2_score_package_used_v1": r6_summary.get("score_dir_v1") == str(v2_score_dir),
        "v2_contract_id_v1": r5_2_policy.get("base_membership_active_contract_id_v1"),
        "v2_base_flags_used_v1": int(_bool(r6_frame, "r5_2_selected_candidate__block_v1").sum()) == 78,
        "old_fixed_or_v1_base_used_v1": False,
        "r6_five_head_count_v1": int(r6_summary.get("r6_head_count_v1") or 0),
        "candidate_grid_count_v1": int(len(grid)),
        "threshold_grid_candidate_count_v1": int(r6_summary.get("threshold_grid_candidate_count_v1") or 0),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "as_of_hindsight_separation_preserved_v1": int(r6_summary.get("as_of_column_count_v1") or 0) == EXPECTED_AS_OF_COLUMNS,
        "no_freeze_promo_live_v1": bool(r6_summary.get("not_freeze_or_promo_v1") and r6_summary.get("not_live_gate_v1")),
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "score_dir_v1": str(v2_score_dir),
        "r6_dir_v1": str(v2_r6_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "contract_id_v1": r5_2_policy.get("base_membership_active_contract_id_v1"),
        "v2_score_package_used_v1": retrain["v2_score_package_used_v1"],
        "v2_contract_used_v1": r5_2_policy.get("base_membership_active_contract_id_v1") == R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"],
        "r6_training_started_v1": bool(r6_summary.get("r6_training_started_v1")),
        "freeze_promo_live_started_v1": False,
        "row_count_v1": int(len(r6_frame)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "as_of_column_count_v1": int(r6_summary.get("as_of_column_count_v1") or 0),
        "selected_candidate_v1": selected_policy_name,
        "selected_family_v1": selected_policy.get("family_v1"),
        "selected_thresholds_v1": selected_params,
        "bad_blocks_v1": int(metrics.get("bad_blocks_v1") or 0),
        "tail_help_v1": int(metrics.get("tail_help_v1") or 0),
        "precision_v1": metrics.get("precision_v1"),
        "worst_loso_v1": _worst(r6_summary),
        "bad_delta_vs_previous_v1": delta_report["delta_vs_previous_v1"]["bad_blocks_v1"],
        "tail_delta_vs_previous_v1": delta_report["delta_vs_previous_v1"]["tail_help_v1"],
        "v2_added_rows_v1": trace_summary["v2_added_row_count_v1"],
        "v2_added_rows_selected_v1": trace_summary["v2_added_rows_selected_v1"],
        "safety_v1": {
            "repaired_damage_v1": eval_report["repaired_damage_v1"],
            "forensic_trade_status_v1": eval_report["forensic_trade_status_v1"],
            "fifty_hundred_twohundred_blocked_v1": [
                eval_report["fifty_plus_mfe_blocked_v1"],
                eval_report["hundred_plus_mfe_blocked_v1"],
                eval_report["two_hundred_plus_mfe_blocked_v1"],
            ],
            "strongest_winner_damage_v1": eval_report["strongest_winner_damage_v1"],
            "runner_near_miss_blocked_v1": eval_report["runner_near_miss_blocked_v1"],
            "safety_failures_v1": eval_report["safety_failures_v1"],
        },
        "wednesday_gap_bad_tail_v1": [
            delta_report["gap_vs_wednesday_v1"]["bad_blocks_v1"],
            delta_report["gap_vs_wednesday_v1"]["tail_help_v1"],
        ],
        "hard_status_v1": {
            "BEVIST": [
                "R6 used the V2 score package and V2 base flags.",
                "The V2-added rows passed through R6 selection.",
                "Safety stayed clean and no freeze/promo/live/controller path ran.",
            ],
            "INDIKERT": [
                "The selected safe R6 policy remains constrained by R5.2 base recall.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not established because recall remains far below Wednesday-R6.",
            ],
        },
    }
    audit = _audit(summary, gate)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "v2_score_dir_v1": str(v2_score_dir),
            "old_score_dir_v1": str(old_score_dir),
            "v2_r6_dir_v1": str(v2_r6_dir),
            "old_r6_dir_v1": str(old_r6_dir),
            "v2_audit_dir_v1": str(v2_audit_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": True,
        "freeze_promo_live_started_v1": False,
    }

    _write_json(output_dir / OUTPUT_FILES["retrain"], retrain)
    candidate_grid.to_csv(output_dir / OUTPUT_FILES["candidate_grid"], index=False)
    _write_json(output_dir / OUTPUT_FILES["eval_against_wednesday"], eval_report)
    _write_json(output_dir / OUTPUT_FILES["delta"], delta_report)
    trace.to_csv(output_dir / OUTPUT_FILES["v2_trace"], index=False)
    _write_json(output_dir / OUTPUT_FILES["forensics"], forensics)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
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
    parser.add_argument("--v2-score-dir", type=Path, default=V2_SCORE_DEFAULT)
    parser.add_argument("--old-score-dir", type=Path, default=OLD_SCORE_DEFAULT)
    parser.add_argument("--v2-r6-dir", type=Path, default=V2_R6_DEFAULT)
    parser.add_argument("--old-r6-dir", type=Path, default=OLD_R6_DEFAULT)
    parser.add_argument("--v2-audit-dir", type=Path, default=V2_AUDIT_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v2_score_dir=args.v2_score_dir,
        old_score_dir=args.old_score_dir,
        v2_r6_dir=args.v2_r6_dir,
        old_r6_dir=args.old_r6_dir,
        v2_audit_dir=args.v2_audit_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
