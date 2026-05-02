#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_parallel_true_r5_2_rebuild_failure_rescue_scan_v1 import (
    TRUE_R5_2_DEFAULT,
    _load_frame,
    _masks,
    _metrics,
    _read_json,
    _safety_pass,
)
from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import (
    BASE_FLAG_COL as RAW_TRUE_BASE_FLAG_COL,
    BAD_SCORE_COL,
    RISKY_SCORE_COL,
    RUNNER_SCORE_COL,
    TAIL_SCORE_COL,
)
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import _bool, _jsonable, _num


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_RESCUE_SCAN_DIR = DEFAULT_REPORTS_ROOT / "PARALLEL_TRUE_R5_2_REBUILD_FAILURE_RESCUE_SCAN_V1_20260426T_LOCK"
LAYER_NAME = "SAFE_TRUE_R5_2_RESCUE_BASE_RULE_V1"
CONTRACT_ID = "TRUE_R5_2_RESCUE_BASE_MEMBERSHIP_SAFE_V1"
RESCUE_BASE_FLAG_COL = "r5_2_true_rescue_base_membership_v1"
RULE_ID = "v3_union_plus_true_scores__b0.85_t0.7_r0.65_p0.2_c1_m0.0_hm0_amb0_run0"

OUTPUT_FILES = {
    "rule": "safe_true_r5_2_rescue_rule_v1.json",
    "base_membership": "true_r5_2_rescue_base_membership_v1.parquet",
    "score_package": "true_r5_2_rescue_score_package_v1.parquet",
    "prediction_view": "true_r5_2_rescue_prediction_view_v1.parquet",
    "r6_manifest": "true_r5_2_rescue_downstream_r6_input_manifest_v1.json",
    "audit": "rescue_rule_application_audit_v1.json",
    "forensics": "rescued_and_rejected_rows_forensics_v1.csv",
    "raw_guard": "no_raw_true_package_to_r6_guard_v1.json",
    "gate": "true_r5_2_rescue_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "consistency_audit": "consistency_audit_v1.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_locked_rule(scan_dir: Path) -> dict[str, Any]:
    decision = _read_json(scan_dir / "rescue_or_retrain_decision_v1.json")
    rule = decision.get("best_safe_rescue_rule_v1") or {}
    if rule.get("rule_id_v1") != RULE_ID:
        raise RuntimeError(f"Safe rescue rule mismatch. Expected {RULE_ID}, got {rule.get('rule_id_v1')}")
    required = {
        "base_source_v1": "v3_union_plus_true_scores",
        "bad_threshold_v1": 0.85,
        "tail_threshold_v1": 0.70,
        "risky_threshold_v1": 0.65,
        "runner_cap_v1": 0.20,
        "consensus_min_v1": 1,
        "margin_min_v1": 0.0,
        "safety_pass_v1": True,
    }
    for key, expected in required.items():
        actual = rule.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"Safe rescue rule {key} mismatch. Expected {expected}, got {actual}")
        elif actual != expected:
            raise RuntimeError(f"Safe rescue rule {key} mismatch. Expected {expected}, got {actual}")
    return rule


def _rule_payload(rule: dict[str, Any], scan_dir: Path, true_r5_2_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "EXTRACT_SAFE_RESCUE_RULE_V1",
        "contract_id_v1": CONTRACT_ID,
        "rule_id_v1": RULE_ID,
        "source_scan_dir_v1": str(scan_dir),
        "true_r5_2_dir_v1": str(true_r5_2_dir),
        "contract_v1": {
            "start_from_v3_safe_base_v1": True,
            "add_true_score_rows_when_v1": {
                BAD_SCORE_COL: ">= 0.85",
                TAIL_SCORE_COL: ">= 0.70",
                RISKY_SCORE_COL: ">= 0.65",
            },
            "required_runner_protection_v1": {RUNNER_SCORE_COL: "< 0.20"},
            "hard_safety_zero_overlap_v1": [
                "ambiguous_high_mfe",
                "runner_protect_bucket",
                "high_mfe_unsafe",
                "repaired_like",
                "strongest_winner",
                "100_plus_mfe",
                "200_plus_mfe",
                "runner_near_miss",
            ],
            "do_not_use_raw_true_base_membership_directly_v1": True,
            "new_base_flag_v1": RESCUE_BASE_FLAG_COL,
        },
        "expected_from_scan_v1": {
            "bad_v1": int(rule["bad_v1"]),
            "tail_v1": int(rule["tail_v1"]),
            "bad_delta_vs_v3_v1": int(rule["bad_delta_vs_v3_v1"]),
            "tail_delta_vs_v3_v1": int(rule["tail_delta_vs_v3_v1"]),
            "precision_v1": float(rule["precision_v1"]),
            "worst_loso_v1": float(rule["worst_loso_v1"]),
            "repaired_like_overlap_v1": int(rule["repaired_like_overlap_v1"]),
            "fifty_plus_overlap_v1": int(rule["fifty_plus_overlap_v1"]),
            "hundred_plus_overlap_v1": int(rule["hundred_plus_overlap_v1"]),
            "two_hundred_plus_overlap_v1": int(rule["two_hundred_plus_overlap_v1"]),
            "strongest_winner_overlap_v1": int(rule["strongest_winner_overlap_v1"]),
            "runner_near_miss_overlap_v1": int(rule["runner_near_miss_overlap_v1"]),
            "ambiguous_high_mfe_included_v1": int(rule["ambiguous_high_mfe_included_v1"]),
            "runner_protect_included_v1": int(rule["runner_protect_included_v1"]),
        },
    }


def _build_rescue_membership(frame: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    bad = _num(frame, BAD_SCORE_COL)
    tail = _num(frame, TAIL_SCORE_COL)
    risky = _num(frame, RISKY_SCORE_COL)
    runner = _num(frame, RUNNER_SCORE_COL)
    score_trigger = (bad >= 0.85) | (tail >= 0.70) | (risky >= 0.65)
    runner_ok = runner < 0.20
    hard_unsafe = (
        pd.Series(masks["ambiguous"], index=frame.index)
        | pd.Series(masks["runner_bucket"], index=frame.index)
        | pd.Series(masks["fifty"], index=frame.index)
        | pd.Series(masks["hundred"], index=frame.index)
        | pd.Series(masks["two_hundred"], index=frame.index)
        | pd.Series(masks["strongest"], index=frame.index)
        | pd.Series(masks["runner_near"], index=frame.index)
        | pd.Series(masks["repaired"], index=frame.index)
    )
    in_v3 = pd.Series(masks["v3_base"], index=frame.index)
    raw_true = pd.Series(masks["current_base"], index=frame.index)
    added = (~in_v3) & score_trigger & runner_ok & ~hard_unsafe
    rescued = in_v3 | added
    rejected_raw = raw_true & ~rescued

    out = frame.copy()
    out["contract_id_v1"] = CONTRACT_ID
    out["in_v3_base_v1"] = in_v3
    out["raw_true_base_membership_v1"] = raw_true
    out["true_rescue_score_trigger_v1"] = score_trigger
    out["added_by_true_rescue_rule_v1"] = added
    out["rejected_by_runner_protection_v1"] = ((~in_v3) & score_trigger & ~runner_ok) | (rejected_raw & (runner >= 0.20))
    out["rejected_by_ambiguous_high_mfe_v1"] = ((~in_v3) & score_trigger & runner_ok & pd.Series(masks["ambiguous"], index=frame.index)) | (rejected_raw & pd.Series(masks["ambiguous"], index=frame.index))
    out["rejected_by_high_mfe_safety_v1"] = (
        ((~in_v3) & score_trigger & runner_ok & (pd.Series(masks["fifty"], index=frame.index) | pd.Series(masks["hundred"], index=frame.index) | pd.Series(masks["two_hundred"], index=frame.index) | pd.Series(masks["runner_near"], index=frame.index)))
        | (rejected_raw & (pd.Series(masks["fifty"], index=frame.index) | pd.Series(masks["hundred"], index=frame.index) | pd.Series(masks["two_hundred"], index=frame.index) | pd.Series(masks["runner_near"], index=frame.index)))
    )
    out["rejected_by_strongest_or_repaired_safety_v1"] = (
        ((~in_v3) & score_trigger & runner_ok & (pd.Series(masks["strongest"], index=frame.index) | pd.Series(masks["repaired"], index=frame.index)))
        | (rejected_raw & (pd.Series(masks["strongest"], index=frame.index) | pd.Series(masks["repaired"], index=frame.index)))
    )
    out["rejected_by_runner_protect_bucket_v1"] = ((~in_v3) & score_trigger & runner_ok & pd.Series(masks["runner_bucket"], index=frame.index)) | (rejected_raw & pd.Series(masks["runner_bucket"], index=frame.index))
    out[RESCUE_BASE_FLAG_COL] = rescued
    out["raw_true_rejected_by_rescue_rule_v1"] = rejected_raw
    return out


def _selected_metrics_from_frame(frame: pd.DataFrame, selected_col: str) -> dict[str, Any]:
    masks = _masks(frame)
    return _metrics(frame, _bool(frame, selected_col).to_numpy(dtype=bool), masks)


def _application_audit(rescued: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    v3 = _selected_metrics_from_frame(rescued, "in_v3_base_v1")
    raw = _selected_metrics_from_frame(rescued, "raw_true_base_membership_v1")
    rescue = _selected_metrics_from_frame(rescued, RESCUE_BASE_FLAG_COL)
    added = _bool(rescued, "added_by_true_rescue_rule_v1")
    rejected = _bool(rescued, "raw_true_rejected_by_rescue_rule_v1")
    hard_fail_reasons = []
    if rescue["repaired_like_overlap_v1"] > 0:
        hard_fail_reasons.append("REPAIRED_OVERLAP")
    if rescue["fifty_plus_overlap_v1"] > 1:
        hard_fail_reasons.append("FIFTY_PLUS_GT_1")
    if rescue["hundred_plus_overlap_v1"] > 0 or rescue["two_hundred_plus_overlap_v1"] > 0:
        hard_fail_reasons.append("100_200_PLUS_OVERLAP")
    if rescue["strongest_winner_overlap_v1"] > 0:
        hard_fail_reasons.append("STRONGEST_WINNER_OVERLAP")
    if rescue["runner_near_miss_overlap_v1"] > 0:
        hard_fail_reasons.append("RUNNER_NEAR_MISS_OVERLAP")
    if rescue["ambiguous_high_mfe_included_v1"] > 0:
        hard_fail_reasons.append("AMBIGUOUS_HIGH_MFE_INCLUDED")
    if rescue["runner_protect_included_v1"] > 0:
        hard_fail_reasons.append("RUNNER_PROTECT_INCLUDED")
    if (rescue["precision_v1"] or 0.0) < float(rule["precision_v1"]):
        hard_fail_reasons.append("PRECISION_BELOW_SCAN")
    if (rescue["worst_loso_v1"] or 0.0) < float(rule["worst_loso_v1"]):
        hard_fail_reasons.append("WORST_LOSO_BELOW_SCAN")
    return {
        "layer_name": "RESCUE_RULE_APPLICATION_AUDIT_V1",
        "contract_id_v1": CONTRACT_ID,
        "rule_id_v1": RULE_ID,
        "v3_baseline_v1": v3,
        "raw_true_v1": raw,
        "rescued_v1": rescue,
        "added_rows_v1": int(added.sum()),
        "rejected_raw_true_rows_v1": int(rejected.sum()),
        "rejected_unsafe_rows_v1": int((rejected & (_bool(rescued, "fifty_plus_mfe_v1") | _bool(rescued, "hundred_plus_mfe_v1") | _bool(rescued, "two_hundred_plus_mfe_v1") | _bool(rescued, "strongest_winner_path_v1") | _bool(rescued, "r6_label_runner_near_miss_v1") | rescued["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD") | rescued["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET"))).sum()),
        "forensic_repaired_trade_status_v1": "UNBLOCKED" if rescue["repaired_like_overlap_v1"] == 0 else "BLOCKED",
        "matches_scan_expected_v1": bool(
            rescue["bad_v1"] == int(rule["bad_v1"])
            and rescue["tail_v1"] == int(rule["tail_v1"])
            and abs(float(rescue["precision_v1"] or 0.0) - float(rule["precision_v1"])) < 1e-12
            and abs(float(rescue["worst_loso_v1"] or 0.0) - float(rule["worst_loso_v1"])) < 1e-12
        ),
        "safety_pass_v1": _safety_pass(rescue),
        "hard_fail_reasons_v1": hard_fail_reasons,
    }


def _reason(row: pd.Series) -> str:
    if bool(row.get("added_by_true_rescue_rule_v1", False)):
        return "ADDED_BY_TRUE_RESCUE_RULE"
    if bool(row.get("rejected_by_runner_protection_v1", False)):
        return "REJECTED_BY_RUNNER_PROTECTION"
    if bool(row.get("rejected_by_ambiguous_high_mfe_v1", False)):
        return "REJECTED_BY_AMBIGUOUS_HIGH_MFE"
    if bool(row.get("rejected_by_high_mfe_safety_v1", False)):
        return "REJECTED_BY_HIGH_MFE_SAFETY"
    if bool(row.get("rejected_by_strongest_or_repaired_safety_v1", False)):
        return "REJECTED_BY_STRONGEST_OR_REPAIRED_SAFETY"
    if bool(row.get("rejected_by_runner_protect_bucket_v1", False)):
        return "REJECTED_BY_RUNNER_PROTECT_BUCKET"
    if bool(row.get("raw_true_rejected_by_rescue_rule_v1", False)):
        return "REJECTED_BY_RESCUE_RULE"
    return "NOT_ESTABLISHED"


def _forensics(rescued: pd.DataFrame) -> pd.DataFrame:
    raw_rejected = _bool(rescued, "raw_true_rejected_by_rescue_rule_v1")
    added = _bool(rescued, "added_by_true_rescue_rule_v1")
    raw_damage = _bool(rescued, "raw_true_base_membership_v1") & (
        _bool(rescued, "fifty_plus_mfe_v1")
        | _bool(rescued, "hundred_plus_mfe_v1")
        | _bool(rescued, "two_hundred_plus_mfe_v1")
        | _bool(rescued, "strongest_winner_path_v1")
        | _bool(rescued, "r6_label_repaired_165_like_runner_v1")
        | _bool(rescued, "r6_label_runner_near_miss_v1")
        | rescued["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
        | rescued["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")
    )
    rows = rescued.loc[added | raw_rejected | raw_damage].copy()
    if rows.empty:
        return rows
    rows["forensic_row_type_v1"] = np.select(
        [added.loc[rows.index], raw_damage.loc[rows.index], raw_rejected.loc[rows.index]],
        ["RESCUED_ADDED_ROW", "RAW_TRUE_SAFETY_DAMAGE_ROW", "RAW_TRUE_REJECTED_ROW"],
        default="NOT_ESTABLISHED",
    )
    rows["reason_added_or_rejected_v1"] = rows.apply(_reason, axis=1)
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "run_id",
        "forensic_row_type_v1",
        "reason_added_or_rejected_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "new_r5_2_label_bucket_v1",
        "in_v3_base_v1",
        "raw_true_base_membership_v1",
        RESCUE_BASE_FLAG_COL,
        "added_by_true_rescue_rule_v1",
        "raw_true_rejected_by_rescue_rule_v1",
        BAD_SCORE_COL,
        TAIL_SCORE_COL,
        RISKY_SCORE_COL,
        RUNNER_SCORE_COL,
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "rejected_by_runner_protection_v1",
        "rejected_by_ambiguous_high_mfe_v1",
        "rejected_by_high_mfe_safety_v1",
        "rejected_by_strongest_or_repaired_safety_v1",
        "rejected_by_runner_protect_bucket_v1",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def _write_packages(output_dir: Path, rescued: pd.DataFrame) -> dict[str, str]:
    key_cols = [col for col in ["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id"] if col in rescued.columns]
    score_cols = [BAD_SCORE_COL, TAIL_SCORE_COL, RISKY_SCORE_COL, RUNNER_SCORE_COL]
    flag_cols = [
        "in_v3_base_v1",
        "raw_true_base_membership_v1",
        "true_rescue_score_trigger_v1",
        "added_by_true_rescue_rule_v1",
        "rejected_by_runner_protection_v1",
        "rejected_by_ambiguous_high_mfe_v1",
        "rejected_by_high_mfe_safety_v1",
        "rejected_by_strongest_or_repaired_safety_v1",
        "rejected_by_runner_protect_bucket_v1",
        RESCUE_BASE_FLAG_COL,
    ]
    prediction_cols = key_cols + [col for col in score_cols + flag_cols if col in rescued.columns]
    base_cols = ["candidate_uid", "trade_uid", "decision_timestamp", "contract_id_v1"] + [col for col in flag_cols if col in rescued.columns]
    rescued[prediction_cols].to_parquet(output_dir / OUTPUT_FILES["prediction_view"], index=False)
    rescued[prediction_cols].to_parquet(output_dir / OUTPUT_FILES["score_package"], index=False)
    rescued[[col for col in base_cols if col in rescued.columns]].to_parquet(output_dir / OUTPUT_FILES["base_membership"], index=False)
    return {
        "prediction_view_v1": str(output_dir / OUTPUT_FILES["prediction_view"]),
        "score_package_v1": str(output_dir / OUTPUT_FILES["score_package"]),
        "base_membership_v1": str(output_dir / OUTPUT_FILES["base_membership"]),
    }


def _r6_manifest(output_dir: Path, package_paths: dict[str, str], true_r5_2_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "MATERIALIZE_TRUE_R5_2_RESCUE_SCORE_PACKAGE_V1",
        "contract_id_v1": CONTRACT_ID,
        "score_package_path_v1": package_paths["score_package_v1"],
        "prediction_view_path_v1": package_paths["prediction_view_v1"],
        "base_membership_path_v1": package_paths["base_membership_v1"],
        "score_columns_for_r6_v1": [BAD_SCORE_COL, TAIL_SCORE_COL, RISKY_SCORE_COL, RUNNER_SCORE_COL],
        "base_flag_for_r6_v1": RESCUE_BASE_FLAG_COL,
        "raw_true_base_flag_not_allowed_v1": RAW_TRUE_BASE_FLAG_COL,
        "raw_true_package_dir_blocked_v1": str(true_r5_2_dir),
        "old_flags_not_allowed_as_final_base_v1": [
            "r5_2_original_base_flag_v1",
            "r5_2_v1_base_flag_v1",
            "r5_2_v2_base_flag_v1",
            "r5_2_v3_base_flag_v1",
            RAW_TRUE_BASE_FLAG_COL,
        ],
        "r6_retrain_started_v1": False,
        "ready_for_future_explicit_r6_retrain_v1": True,
        "manifest_path_v1": str(output_dir / OUTPUT_FILES["r6_manifest"]),
    }


def _raw_true_guard(r6_manifest: dict[str, Any], audit: dict[str, Any], rescued: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "downstream_manifest_uses_rescue_base_flag_v1": r6_manifest.get("base_flag_for_r6_v1") == RESCUE_BASE_FLAG_COL,
        "downstream_manifest_does_not_use_raw_true_base_flag_v1": r6_manifest.get("base_flag_for_r6_v1") != RAW_TRUE_BASE_FLAG_COL,
        "rescue_base_flag_present_v1": RESCUE_BASE_FLAG_COL in rescued.columns,
        "raw_true_metrics_not_used_as_pass_v1": (not _safety_pass(audit["raw_true_v1"])) and _safety_pass(audit["rescued_v1"]),
        "raw_true_direct_r6_blocked_v1": True,
    }
    return {
        "layer_name": "NO_RAW_TRUE_PACKAGE_TO_R6_GUARD_V1",
        "contract_id_v1": CONTRACT_ID,
        "checks_v1": checks,
        "guard_pass_v1": bool(all(checks.values())),
        "blocked_actions_v1": [
            "DO_NOT_FEED_RAW_TRUE_R5_2_TO_R6",
            "DO_NOT_USE_R5_2_REBUILT_BASE_MEMBERSHIP_V1_AS_R6_BASE",
        ],
    }


def _gate(audit: dict[str, Any], raw_guard: dict[str, Any], package_paths: dict[str, str]) -> dict[str, Any]:
    package_ready = all(Path(path).exists() for path in package_paths.values())
    if not package_ready:
        decision = "TRUE_R5_2_RESCUE_PACKAGE_NOT_READY_FOR_R6"
    elif not audit["safety_pass_v1"]:
        decision = "TRUE_R5_2_RESCUE_SAFETY_FAIL"
    elif not audit["matches_scan_expected_v1"]:
        decision = "TRUE_R5_2_RESCUE_DIVERGES_FROM_SCAN"
    elif not raw_guard["guard_pass_v1"]:
        decision = "TRUE_R5_2_RESCUE_PACKAGE_NOT_READY_FOR_R6"
    else:
        decision = "TRUE_R5_2_RESCUE_BASE_RULE_PASS"
    return {
        "layer_name": "TRUE_R5_2_RESCUE_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "rescue_rule_implemented_v1": True,
            "rescued_package_written_v1": package_ready,
            "downstream_r6_manifest_written_v1": True,
            "matches_scan_expected_v1": audit["matches_scan_expected_v1"],
            "safety_pass_v1": audit["safety_pass_v1"],
            "raw_true_package_blocked_v1": raw_guard["guard_pass_v1"],
        },
    }


def _next_action(gate: dict[str, Any]) -> dict[str, Any]:
    if gate["decision_v1"] == "TRUE_R5_2_RESCUE_BASE_RULE_PASS":
        action = "RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_EXPLICIT_FLAG"
    elif gate["decision_v1"] == "TRUE_R5_2_RESCUE_DIVERGES_FROM_SCAN":
        action = "FIX_TRUE_R5_2_RESCUE_RULE_APPLICATION_FIRST"
    else:
        action = "DO_NOT_RUN_R6_RETRAIN_YET"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_FEED_RAW_TRUE_R5_2_TO_R6",
            "DO_NOT_RUN_R6_RETRAIN_WITHOUT_EXPLICIT_FLAG",
        ],
    }


def _consistency_audit(summary: dict[str, Any], audit: dict[str, Any], raw_guard: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_R6", not summary["r6_started_v1"], summary["r6_started_v1"]),
            row("MATCHES_SCAN_EXPECTED", audit["matches_scan_expected_v1"], audit["rescued_v1"]),
            row("SAFETY_PASS", audit["safety_pass_v1"], audit["hard_fail_reasons_v1"]),
            row("RAW_TRUE_GUARD", raw_guard["guard_pass_v1"], raw_guard["checks_v1"]),
            row("PACKAGE_READY", summary["rescued_package_ready_for_r6_v1"], summary["rescued_package_ready_for_r6_v1"]),
        ]
    )


def _report(summary: dict[str, Any], audit: dict[str, Any]) -> str:
    rescue = audit["rescued_v1"]
    return "\n".join(
        [
            "# Safe True R5.2 Rescue Base Rule",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rescued bad/tail: `{rescue['bad_v1']}/{rescue['tail_v1']}`",
            f"- Precision / worst LOSO: `{rescue['precision_v1']}` / `{rescue['worst_loso_v1']}`",
            f"- Safety pass: `{audit['safety_pass_v1']}`",
            f"- Added rows: `{audit['added_rows_v1']}`",
            f"- Rejected raw true rows: `{audit['rejected_raw_true_rows_v1']}`",
            "",
            "Raw true base membership is blocked as direct R6 input.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    true_r5_2_dir: Path = TRUE_R5_2_DEFAULT,
    rescue_scan_dir: Path = DEFAULT_RESCUE_SCAN_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    true_r5_2_dir = true_r5_2_dir.expanduser().resolve()
    rescue_scan_dir = rescue_scan_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_rule = _load_locked_rule(rescue_scan_dir)
    rule_payload = _rule_payload(locked_rule, rescue_scan_dir, true_r5_2_dir)
    frame, input_paths = _load_frame(true_r5_2_dir)
    masks = _masks(frame)
    rescued = _build_rescue_membership(frame, masks)
    audit = _application_audit(rescued, locked_rule)
    package_paths = _write_packages(output_dir, rescued)
    r6_manifest = _r6_manifest(output_dir, package_paths, true_r5_2_dir)
    raw_guard = _raw_true_guard(r6_manifest, audit, rescued)
    gate = _gate(audit, raw_guard, package_paths)
    next_action = _next_action(gate)
    forensics = _forensics(rescued)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "true_r5_2_dir_v1": str(true_r5_2_dir),
        "rescue_scan_dir_v1": str(rescue_scan_dir),
        "contract_id_v1": CONTRACT_ID,
        "rule_id_v1": RULE_ID,
        "training_started_v1": False,
        "r6_started_v1": False,
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "rescued_bad_v1": audit["rescued_v1"]["bad_v1"],
        "rescued_tail_v1": audit["rescued_v1"]["tail_v1"],
        "uplift_vs_v3_bad_v1": int(audit["rescued_v1"]["bad_v1"] - audit["v3_baseline_v1"]["bad_v1"]),
        "uplift_vs_v3_tail_v1": int(audit["rescued_v1"]["tail_v1"] - audit["v3_baseline_v1"]["tail_v1"]),
        "precision_v1": audit["rescued_v1"]["precision_v1"],
        "worst_loso_v1": audit["rescued_v1"]["worst_loso_v1"],
        "safety_pass_v1": audit["safety_pass_v1"],
        "added_rows_v1": audit["added_rows_v1"],
        "rejected_raw_true_rows_v1": audit["rejected_raw_true_rows_v1"],
        "raw_true_blocked_from_r6_v1": raw_guard["guard_pass_v1"],
        "rescued_package_ready_for_r6_v1": gate["decision_v1"] == "TRUE_R5_2_RESCUE_BASE_RULE_PASS",
        "hard_status_v1": {
            "BEVIST": [
                "Safe true R5.2 rescue rule was applied without training or R6 execution.",
                "Raw true base membership is blocked as direct R6 input.",
            ],
            "INDIKERT": [
                "The rescued package is suitable for a future explicit R6 retrain if the gate is PASS.",
            ],
            "IKKE_ETABLERT": [
                "No R6 uplift is established because R6 was not run.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_paths_v1": input_paths,
        "output_files_v1": OUTPUT_FILES,
        "package_paths_v1": package_paths,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "training_started_v1": False,
        "r6_started_v1": False,
    }

    _write_json(output_dir / OUTPUT_FILES["rule"], rule_payload)
    _write_json(output_dir / OUTPUT_FILES["r6_manifest"], r6_manifest)
    _write_json(output_dir / OUTPUT_FILES["audit"], audit)
    forensics.to_csv(output_dir / OUTPUT_FILES["forensics"], index=False)
    _write_json(output_dir / OUTPUT_FILES["raw_guard"], raw_guard)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _consistency_audit(summary, audit, raw_guard).to_csv(output_dir / OUTPUT_FILES["consistency_audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, audit), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--true-r5-2-dir", type=Path, default=TRUE_R5_2_DEFAULT)
    parser.add_argument("--rescue-scan-dir", type=Path, default=DEFAULT_RESCUE_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        true_r5_2_dir=args.true_r5_2_dir,
        rescue_scan_dir=args.rescue_scan_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
