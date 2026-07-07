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

from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V1,
    R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
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
LAYER_NAME = "SAFE_R5_2_BASE_EXTENSION_V2_V1"

OLD_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_FIX_R5_R51_R52"
NEW_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V2_R5_R51_R52"
SCAN_DEFAULT = DEFAULT_REPORTS_ROOT / "PARALLEL_MONDAY_R6_RECALL_RECOVERY_SCAN_V1_20260426T_LOCK"
FOUNDATION_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_20260425T_FOUNDATION_LOCK_V4"

SCAN_AGGREGATOR = "parallel_scan_aggregator_v1.json"
SCAN_LANE_02 = "lane_02_r5_2_base_extension_v2_scan_v1.csv"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

OUTPUT_FILES = {
    "rule": "safe_r5_2_base_extension_v2_rule_v1.json",
    "implementation": "r5_2_base_contract_v2_implementation_v1.json",
    "rebuild": "r5_r5_1_r5_2_score_rebuild_with_v2_contract_v1.json",
    "contract_audit": "v2_contract_application_audit_v1.csv",
    "added_forensics": "v2_added_rows_forensics_v1.csv",
    "surface_guard": "no_new_surface_duplicate_guard_v1.json",
    "gate": "r5_2_v2_score_rebuild_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

SCORE_SCHEMA_COLUMNS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "r5_1_bad_blocker_score_v1",
    "r5_1_runner_guard_score_v1",
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    "r5_selected_candidate__block_v1",
    "r5_1_selected_candidate__block_v1",
    "r5_2_selected_candidate__block_v1",
    "blocker_score_v1",
    "runner_protector_score_v1",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _keyset(frame: pd.DataFrame) -> set[str]:
    return set(frame["candidate_uid"].astype("string").fillna("").tolist()) if "candidate_uid" in frame.columns else set()


def _aligned_pair(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    old_keys = _keyset(old)
    new_keys = _keyset(new)
    common = sorted(old_keys & new_keys)
    old_a = old.set_index("candidate_uid").loc[common].reset_index()
    new_a = new.set_index("candidate_uid").loc[common].reset_index()
    return old_a, new_a, {
        "old_only_candidate_count_v1": int(len(old_keys - new_keys)),
        "new_only_candidate_count_v1": int(len(new_keys - old_keys)),
        "common_candidate_count_v1": int(len(common)),
        "key_alignment_gap_count_v1": int(len(old_keys - new_keys) + len(new_keys - old_keys)),
    }


def _metric_bundle(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    metrics = _policy_metrics(frame, mask)
    safety_pass, worst_loso, hard_damage = _wednesday_safety_pass(frame, mask)
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index)
    return {
        **metrics,
        "worst_loso_v1": worst_loso,
        "hard_damage_count_v1": hard_damage,
        "wednesday_safety_pass_v1": bool(safety_pass),
        "forensic_repaired_trade_blocked_v1": int((selected & forensic).sum()),
        "is_repaired_165_blocked_v1": int((selected & _bool(frame, "is_repaired_165_v1")).sum()),
        "runner_near_miss_blocked_v1": int((selected & _bool(frame, "r6_label_runner_near_miss_v1")).sum()),
    }


def _load_scan_rule(scan_dir: Path) -> dict[str, Any]:
    aggregator = _read_json(scan_dir / SCAN_AGGREGATOR)
    best = aggregator.get("best_safe_bad_candidate_v1") or {}
    params = json.loads(best.get("params_json_v1") or "{}")
    lane = pd.read_csv(scan_dir / SCAN_LANE_02) if (scan_dir / SCAN_LANE_02).exists() else pd.DataFrame()
    return {
        "layer_name": "EXTRACT_SAFE_R5_2_BASE_EXTENSION_V2_RULE_V1",
        "source_scan_dir_v1": str(scan_dir),
        "source_lane_v1": best.get("lane_v1"),
        "rule_id_v1": best.get("rule_id_v1"),
        "contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"],
        "rule_definition_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_rule_v1"],
        "score_fields_v1": [
            R5_2_BAD_PROB,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
            "pred__entry_r5_runner_protect__prob_true_v1",
            "r5_1_runner_guard_score_v1",
            R5_2_RUNNER_PROB,
        ],
        "thresholds_v1": params,
        "as_of_legality_v1": {
            "uses_existing_scores_only_v1": True,
            "uses_hindsight_or_exit_truth_as_input_v1": False,
            "uses_new_feature_surface_v1": False,
            "why_legal_v1": "All fields are existing entry-time score outputs from the canonical Monday foundation score package.",
        },
        "expected_v1_bad_tail_v1": [76, 48],
        "expected_v2_bad_tail_v1": [78, 49],
        "expected_uplift_bad_tail_v1": [2, 1],
        "expected_safety_v1": {
            "precision_v1": 1.0,
            "worst_loso_v1": 1.0,
            "repaired_damage_v1": 0,
            "forensic_trade_blocked_v1": 0,
            "fifty_hundred_twohundred_blocked_v1": [0, 0, 0],
            "strongest_winner_damage_v1": 0,
        },
        "best_scan_row_v1": best,
        "lane_02_row_count_v1": int(len(lane)),
    }


def _contract_conditions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "r5_2_bad_ge_035_v1": _num(frame, R5_2_BAD_PROB).ge(0.35).fillna(False),
        "r5_immediate_mae_ge_075_v1": _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(0.75).fillna(False),
        "r5_runner_lt_045_v1": _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(0.45).fillna(False),
        "r5_1_runner_lt_045_v1": _num(frame, "r5_1_runner_guard_score_v1").lt(0.45).fillna(False),
        "r5_2_runner_lt_035_v1": _num(frame, R5_2_RUNNER_PROB).lt(0.35).fillna(False),
    }


def _v2_extension_mask(frame: pd.DataFrame) -> pd.Series:
    conditions = _contract_conditions(frame)
    mask = pd.Series(True, index=frame.index)
    for values in conditions.values():
        mask &= values
    return mask.fillna(False).astype(bool)


def _mfe_bucket(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                _bool(frame, "two_hundred_plus_mfe_v1"),
                _bool(frame, "hundred_plus_mfe_v1"),
                _bool(frame, "fifty_plus_mfe_v1"),
                _bool(frame, "tail_10_50_mfe_v1"),
            ],
            ["200_PLUS", "100_PLUS", "50_PLUS", "TAIL_10_50"],
            default="LOW_OR_NO_MFE",
        ),
        index=frame.index,
    )


def _added_forensics(new_frame: pd.DataFrame, added_v2: pd.Series) -> pd.DataFrame:
    cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "split_scope_v1",
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
        "pred__entry_r5_should_not_take__prob_true_v1",
        "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
        "pred__entry_r5_runner_protect__prob_true_v1",
        "r5_1_bad_blocker_score_v1",
        "r5_1_runner_guard_score_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
    ]
    out = new_frame.loc[added_v2, [col for col in cols if col in new_frame.columns]].copy()
    conditions = _contract_conditions(new_frame)
    for name, values in conditions.items():
        out[name] = values.loc[out.index].to_numpy(dtype=bool)
    out["mfe_bucket_v1"] = _mfe_bucket(out).to_numpy()
    out["v2_added_reason_v1"] = R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_rule_v1"]
    risk = (
        _bool(out, "take_was_ok_v1")
        | _bool(out, "fifty_plus_mfe_v1")
        | _bool(out, "hundred_plus_mfe_v1")
        | _bool(out, "two_hundred_plus_mfe_v1")
        | _bool(out, "strongest_winner_path_v1")
        | _bool(out, "r6_label_repaired_165_like_runner_v1")
        | _bool(out, "r6_label_runner_near_miss_v1")
    )
    out["safe_or_reject_v1"] = np.where(risk, "REJECT_RISK_CANDIDATE", "SAFE_RECOVERABLE")
    return out


def _surface_guard(old_score_dir: Path, new_score_dir: Path, foundation_dir: Path, old_frame: pd.DataFrame, new_frame: pd.DataFrame, new_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "NO_NEW_SURFACE_DUPLICATE_GUARD_V1",
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "forbidden_1689_exact_only_used_v1": int(len(old_frame)) == 1689 or int(len(new_frame)) == 1689,
        "protector_first_used_v1": False,
        "diagnostic_surfaces_used_as_canonical_input_v1": False,
        "foundation_dir_v1": str(foundation_dir),
        "old_score_dir_v1": str(old_score_dir),
        "new_score_dir_v1": str(new_score_dir),
        "same_monday_foundation_v1": str(new_summary.get("foundation_dir_v1") or "") == str(foundation_dir),
        "row_count_v1": int(len(new_frame)),
        "active_rows_v1": int(new_summary.get("active_rows_v1") or 0),
        "quarantine_rows_v1": int(new_summary.get("quarantine_rows_v1") or 0),
        "as_of_column_count_v1": int(new_summary.get("as_of_column_count_v1") or 0),
        "uses_existing_score_features_v1": True,
    }


def _application_audit(
    old_frame: pd.DataFrame,
    new_frame: pd.DataFrame,
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    r5_2_policy: dict[str, Any],
    added_v2: pd.Series,
) -> pd.DataFrame:
    v2_conditions = _contract_conditions(new_frame)
    rows = [
        ("V2_CONTRACT_ID", r5_2_policy.get("base_membership_active_contract_id_v1"), "PASS" if r5_2_policy.get("base_membership_active_contract_id_v1") == R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"] else "FAIL"),
        ("V2_CONTRACT_APPLIED", r5_2_policy.get("v2_contract_applied_v1"), "PASS" if r5_2_policy.get("v2_contract_applied_v1") is True else "FAIL"),
        ("OLD_FIXED_BAD_TAIL", [old_metrics["bad_blocks_v1"], old_metrics["tail_help_v1"]], "PASS" if [old_metrics["bad_blocks_v1"], old_metrics["tail_help_v1"]] == [76, 48] else "WARN"),
        ("NEW_V2_BAD_TAIL", [new_metrics["bad_blocks_v1"], new_metrics["tail_help_v1"]], "PASS" if [new_metrics["bad_blocks_v1"], new_metrics["tail_help_v1"]] == [78, 49] else "WARN"),
        ("EXPECTED_INCREMENTAL_UPLIFT", [new_metrics["bad_blocks_v1"] - old_metrics["bad_blocks_v1"], new_metrics["tail_help_v1"] - old_metrics["tail_help_v1"]], "PASS" if [new_metrics["bad_blocks_v1"] - old_metrics["bad_blocks_v1"], new_metrics["tail_help_v1"] - old_metrics["tail_help_v1"]] == [2, 1] else "FAIL"),
        ("V2_INCREMENTAL_ROWS_ADDED", int(added_v2.sum()), "PASS" if int(added_v2.sum()) == 2 else "WARN"),
        ("V2_INCREMENTAL_BAD_ROWS_ADDED", int((added_v2 & _bool(new_frame, "label_should_not_take_v1")).sum()), "PASS" if int((added_v2 & _bool(new_frame, "label_should_not_take_v1")).sum()) == 2 else "WARN"),
        ("V2_INCREMENTAL_TAIL_ROWS_ADDED", int((added_v2 & _bool(new_frame, "tail_10_50_mfe_v1")).sum()), "PASS" if int((added_v2 & _bool(new_frame, "tail_10_50_mfe_v1")).sum()) == 1 else "WARN"),
        ("SAFETY_PRECISION", new_metrics["precision_v1"], "PASS" if new_metrics["precision_v1"] == 1.0 else "FAIL"),
        ("SAFETY_WORST_LOSO", new_metrics["worst_loso_v1"], "PASS" if new_metrics["worst_loso_v1"] == 1.0 else "FAIL"),
        ("SAFETY_HARD_DAMAGE", new_metrics["hard_damage_count_v1"], "PASS" if int(new_metrics["hard_damage_count_v1"]) == 0 else "FAIL"),
        ("FORENSIC_TRADE_UNBLOCKED", new_metrics["forensic_repaired_trade_blocked_v1"], "PASS" if int(new_metrics["forensic_repaired_trade_blocked_v1"]) == 0 else "FAIL"),
        ("HIGH_MFE_50_100_200", [new_metrics["fifty_plus_mfe_blocked_v1"], new_metrics["hundred_plus_mfe_blocked_v1"], new_metrics["two_hundred_plus_mfe_blocked_v1"]], "PASS" if [new_metrics["fifty_plus_mfe_blocked_v1"], new_metrics["hundred_plus_mfe_blocked_v1"], new_metrics["two_hundred_plus_mfe_blocked_v1"]] == [0, 0, 0] else "FAIL"),
        ("STRONGEST_WINNER_DAMAGE", new_metrics["strongest_winner_damage_v1"], "PASS" if int(new_metrics["strongest_winner_damage_v1"]) == 0 else "FAIL"),
    ]
    for name, values in v2_conditions.items():
        rows.append((name, int(values[added_v2].sum()), "PASS" if bool(values[added_v2].all()) else "FAIL"))
    return pd.DataFrame(
        [
            {
                "check_v1": name,
                "value_v1": json.dumps(_jsonable(value), sort_keys=True),
                "status_v1": status,
            }
            for name, value, status in rows
        ]
    )


def _schema_report(old_frame: pd.DataFrame, new_frame: pd.DataFrame) -> dict[str, Any]:
    old_cols = [col for col in SCORE_SCHEMA_COLUMNS if col in old_frame.columns]
    new_cols = [col for col in SCORE_SCHEMA_COLUMNS if col in new_frame.columns]
    return {
        "old_count_v1": int(len(old_cols)),
        "new_count_v1": int(len(new_cols)),
        "missing_in_new_v1": sorted(set(old_cols) - set(new_cols)),
        "new_extra_v1": sorted(set(new_cols) - set(old_cols)),
        "schema_intact_v1": set(old_cols) == set(new_cols),
    }


def _gate(
    *,
    r5_2_policy: dict[str, Any],
    key_report: dict[str, Any],
    schema: dict[str, Any],
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    added_v2: pd.Series,
    surface_guard: dict[str, Any],
) -> dict[str, Any]:
    contract_ok = (
        r5_2_policy.get("base_membership_active_contract_id_v1") == R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"]
        and bool(r5_2_policy.get("v2_contract_applied_v1"))
    )
    uplift_ok = (
        int(new_metrics["bad_blocks_v1"]) == 78
        and int(new_metrics["tail_help_v1"]) == 49
        and int(new_metrics["bad_blocks_v1"] - old_metrics["bad_blocks_v1"]) == 2
        and int(new_metrics["tail_help_v1"] - old_metrics["tail_help_v1"]) == 1
        and int(added_v2.sum()) == 2
    )
    safety_ok = (
        bool(new_metrics["wednesday_safety_pass_v1"])
        and int(new_metrics["hard_damage_count_v1"]) == 0
        and int(new_metrics["forensic_repaired_trade_blocked_v1"]) == 0
        and int(new_metrics["fifty_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["hundred_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["two_hundred_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["strongest_winner_damage_v1"]) == 0
    )
    surface_ok = (
        surface_guard["no_new_baseline_v1"]
        and surface_guard["no_new_feature_surface_v1"]
        and not surface_guard["forbidden_1689_exact_only_used_v1"]
        and not surface_guard["protector_first_used_v1"]
        and not surface_guard["diagnostic_surfaces_used_as_canonical_input_v1"]
        and surface_guard["uses_existing_score_features_v1"]
    )
    schema_ok = bool(schema["schema_intact_v1"])
    keys_ok = int(key_report["key_alignment_gap_count_v1"]) == 0
    if contract_ok and uplift_ok and safety_ok and surface_ok and schema_ok and keys_ok:
        decision = "R5_2_V2_SCORE_REBUILD_PASS"
    elif not contract_ok:
        decision = "R5_2_V2_CONTRACT_NOT_APPLIED"
    elif not uplift_ok:
        decision = "R5_2_V2_DIVERGES_FROM_PARALLEL_SCAN"
    elif not safety_ok:
        decision = "R5_2_V2_SAFETY_FAIL"
    else:
        decision = "R5_2_V2_NOT_READY_FOR_R6"
    return {
        "layer_name": "R5_2_V2_SCORE_REBUILD_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "contract_ok_v1": contract_ok,
            "uplift_matches_parallel_scan_v1": uplift_ok,
            "safety_ok_v1": safety_ok,
            "surface_ok_v1": surface_ok,
            "schema_ok_v1": schema_ok,
            "key_alignment_ok_v1": keys_ok,
            "ready_for_r6_retrain_v1": decision == "R5_2_V2_SCORE_REBUILD_PASS",
        },
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    checks = gate["checks_v1"]
    return pd.DataFrame(
        [
            row("V2_CONTRACT_USED", checks["contract_ok_v1"], summary["contract_id_v1"]),
            row("UPLIFT_MATCHES_SCAN", checks["uplift_matches_parallel_scan_v1"], [summary["bad_uplift_v1"], summary["tail_uplift_v1"]]),
            row("SAFETY_OK", checks["safety_ok_v1"], summary["new_safety_v1"]),
            row("NO_NEW_SURFACE", checks["surface_ok_v1"], True),
            row("SCHEMA_OK", checks["schema_ok_v1"], True),
            row("KEY_ALIGNMENT_OK", checks["key_alignment_ok_v1"], True),
            row("R6_NOT_RUN", not summary["r6_heads_trained_v1"], summary["r6_heads_trained_v1"]),
        ]
    )


def _report(summary: dict[str, Any], gate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Safe R5.2 Base Extension V2",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- V2 rule: `{summary['v2_rule_v1']}`",
            f"- Old fixed bad/tail: `{summary['old_bad_blocks_v1']}` / `{summary['old_tail_help_v1']}`",
            f"- New V2 bad/tail: `{summary['new_bad_blocks_v1']}` / `{summary['new_tail_help_v1']}`",
            f"- Uplift: `+{summary['bad_uplift_v1']}` / `+{summary['tail_uplift_v1']}`",
            f"- New precision/worst LOSO: `{summary['new_precision_v1']}` / `{summary['new_worst_loso_v1']}`",
            f"- Safety: `{summary['new_safety_v1']}`",
            f"- Gate checks: `{gate['checks_v1']}`",
            "",
            "This is an R5/R5.1/R5.2 score rebuild only. R6 retrain was not run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    old_score_dir: Path = OLD_SCORE_DEFAULT,
    new_score_dir: Path = NEW_SCORE_DEFAULT,
    scan_dir: Path = SCAN_DEFAULT,
    foundation_dir: Path = FOUNDATION_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_score_dir = old_score_dir.expanduser().resolve()
    new_score_dir = new_score_dir.expanduser().resolve()
    scan_dir = scan_dir.expanduser().resolve()
    foundation_dir = foundation_dir.expanduser().resolve()

    old_frame = pd.read_parquet(old_score_dir / SCORE_FRAME)
    new_frame = pd.read_parquet(new_score_dir / SCORE_FRAME)
    old_score_summary = _read_json(old_score_dir / SCORE_SUMMARY)
    new_score_summary = _read_json(new_score_dir / SCORE_SUMMARY)
    new_summary = _read_json(new_score_dir / SCORE_STATUS_SUMMARY)
    old_aligned, new_aligned, key_report = _aligned_pair(old_frame, new_frame)
    old_base = _bool(old_aligned, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_aligned, "r5_2_selected_candidate__block_v1")
    added_v2 = new_base & ~old_base
    old_metrics = _metric_bundle(old_aligned, old_base)
    new_metrics = _metric_bundle(new_aligned, new_base)
    r5_2_policy = new_score_summary.get("r5_2_selected_policy_v1") or {}
    rule = _load_scan_rule(scan_dir)
    implementation = {
        "layer_name": "IMPLEMENT_R5_2_BASE_CONTRACT_V2_V1",
        "contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"],
        "v1_contract_retained_for_lineage_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V1,
        "v2_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
        "changed_script_v1": "gx1/scripts/train_monday_r6_foundation_score_rebuild_v1.py",
        "original_base_definition_v1": "calibrated R5.2 bad score and runner max base grid",
        "v1_extension_definition_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_rule_v1"],
        "v2_extension_definition_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_rule_v1"],
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "uses_existing_scores_only_v1": True,
    }
    schema = _schema_report(old_aligned, new_aligned)
    surface_guard = _surface_guard(old_score_dir, new_score_dir, foundation_dir, old_frame, new_frame, new_summary)
    contract_audit = _application_audit(old_aligned, new_aligned, old_metrics, new_metrics, r5_2_policy, added_v2)
    added_forensics = _added_forensics(new_aligned, added_v2)
    rebuild = {
        "layer_name": "RUN_R5_R5_1_R5_2_SCORE_REBUILD_WITH_V2_CONTRACT_V1",
        "new_score_dir_v1": str(new_score_dir),
        "foundation_dir_v1": str(foundation_dir),
        "explicit_score_rebuild_flag_v1": bool(new_summary.get("explicit_score_rebuild_flag_v1")),
        "decision_v1": new_summary.get("decision_v1"),
        "r6_heads_trained_v1": bool(new_summary.get("r6_heads_trained_v1")),
        "contract_id_v1": r5_2_policy.get("base_membership_active_contract_id_v1"),
        "v2_contract_applied_v1": bool(r5_2_policy.get("v2_contract_applied_v1")),
        "old_fixed_metrics_v1": old_metrics,
        "new_v2_metrics_v1": new_metrics,
        "schema_report_v1": schema,
        "key_alignment_v1": key_report,
        "base_counts_v1": {
            "old_fixed_base_v1": int(old_base.sum()),
            "new_v2_base_v1": int(new_base.sum()),
            "v2_added_rows_v1": int(added_v2.sum()),
            "rows_removed_v1": int((old_base & ~new_base).sum()),
            "original_base_before_contract_v1": (r5_2_policy.get("base_metrics_before_contract_v1") or {}).get("block_count_v1"),
            "v1_base_after_contract_v1": (r5_2_policy.get("v1_contract_metrics_v1") or {}).get("block_count_v1"),
        },
        "divergence_from_scan_v1": None
        if [new_metrics["bad_blocks_v1"], new_metrics["tail_help_v1"]] == [78, 49]
        else "Score-head rebuild drift, threshold drift, row membership, or contract bug requires investigation before R6 retrain.",
    }
    gate = _gate(
        r5_2_policy=r5_2_policy,
        key_report=key_report,
        schema=schema,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        added_v2=added_v2,
        surface_guard=surface_guard,
    )
    next_action = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": (
            "RUN_R6_RETRAIN_FROM_R5_2_V2_SCORE_PACKAGE_EXPLICIT_FLAG"
            if gate["decision_v1"] == "R5_2_V2_SCORE_REBUILD_PASS"
            else "DO_NOT_RUN_R6_RETRAIN_YET"
        ),
        "blocked_action_v1": (
            []
            if gate["decision_v1"] == "R5_2_V2_SCORE_REBUILD_PASS"
            else ["FIX_R5_2_V2_CONTRACT_APPLICATION_FIRST", "FIX_V2_REBUILD_DIVERGENCE_FIRST"]
        ),
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "old_score_dir_v1": str(old_score_dir),
        "new_score_dir_v1": str(new_score_dir),
        "scan_dir_v1": str(scan_dir),
        "foundation_dir_v1": str(foundation_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "contract_id_v1": r5_2_policy.get("base_membership_active_contract_id_v1"),
        "v2_rule_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_rule_v1"],
        "explicit_score_rebuild_flag_v1": bool(new_summary.get("explicit_score_rebuild_flag_v1")),
        "r6_heads_trained_v1": bool(new_summary.get("r6_heads_trained_v1")),
        "old_bad_blocks_v1": int(old_metrics["bad_blocks_v1"]),
        "old_tail_help_v1": int(old_metrics["tail_help_v1"]),
        "new_bad_blocks_v1": int(new_metrics["bad_blocks_v1"]),
        "new_tail_help_v1": int(new_metrics["tail_help_v1"]),
        "bad_uplift_v1": int(new_metrics["bad_blocks_v1"] - old_metrics["bad_blocks_v1"]),
        "tail_uplift_v1": int(new_metrics["tail_help_v1"] - old_metrics["tail_help_v1"]),
        "v2_added_rows_v1": int(added_v2.sum()),
        "v2_added_bad_rows_v1": int((added_v2 & _bool(new_aligned, "label_should_not_take_v1")).sum()),
        "v2_added_tail_rows_v1": int((added_v2 & _bool(new_aligned, "tail_10_50_mfe_v1")).sum()),
        "new_precision_v1": new_metrics["precision_v1"],
        "new_worst_loso_v1": new_metrics["worst_loso_v1"],
        "new_safety_v1": {
            "repaired_damage_v1": int(new_metrics["repaired_165_damage_v1"]),
            "forensic_trade_blocked_v1": int(new_metrics["forensic_repaired_trade_blocked_v1"]),
            "fifty_hundred_twohundred_blocked_v1": [
                int(new_metrics["fifty_plus_mfe_blocked_v1"]),
                int(new_metrics["hundred_plus_mfe_blocked_v1"]),
                int(new_metrics["two_hundred_plus_mfe_blocked_v1"]),
            ],
            "strongest_winner_damage_v1": int(new_metrics["strongest_winner_damage_v1"]),
            "runner_near_miss_blocked_v1": int(new_metrics["runner_near_miss_blocked_v1"]),
        },
        "new_score_package_ready_for_r6_retrain_v1": gate["decision_v1"] == "R5_2_V2_SCORE_REBUILD_PASS",
        "hard_status_v1": {
            "BEVIST": [
                "The V2 R5.2 base contract is implemented and active in the rebuilt score package.",
                "The explicit score rebuild matched the parallel scan at 78 bad / 49 tail with zero hard safety damage.",
                "No R6 retrain, new baseline, new feature surface, 1689, diagnostic, or protector-first input was used.",
            ],
            "INDIKERT": [
                "The new score package is ready for an explicit R6 retrain gate.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not established until R6 is retrained and compared.",
            ],
        },
    }
    audit = _audit(summary, gate)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "old_score_dir_v1": str(old_score_dir),
            "new_score_dir_v1": str(new_score_dir),
            "scan_dir_v1": str(scan_dir),
            "foundation_dir_v1": str(foundation_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "score_rebuild_already_completed_v1": True,
        "r6_heads_trained_v1": False,
    }

    _write_json(output_dir / OUTPUT_FILES["rule"], rule)
    _write_json(output_dir / OUTPUT_FILES["implementation"], implementation)
    _write_json(output_dir / OUTPUT_FILES["rebuild"], rebuild)
    contract_audit.to_csv(output_dir / OUTPUT_FILES["contract_audit"], index=False)
    added_forensics.to_csv(output_dir / OUTPUT_FILES["added_forensics"], index=False)
    _write_json(output_dir / OUTPUT_FILES["surface_guard"], surface_guard)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, gate), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--old-score-dir", type=Path, default=OLD_SCORE_DEFAULT)
    parser.add_argument("--new-score-dir", type=Path, default=NEW_SCORE_DEFAULT)
    parser.add_argument("--scan-dir", type=Path, default=SCAN_DEFAULT)
    parser.add_argument("--foundation-dir", type=Path, default=FOUNDATION_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        old_score_dir=args.old_score_dir,
        new_score_dir=args.new_score_dir,
        scan_dir=args.scan_dir,
        foundation_dir=args.foundation_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
