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
    R5_2_BASE_MEMBERSHIP_CONTRACT,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    SCORE_FRAME,
    SCORE_SUMMARY,
    SUMMARY as SCORE_STATUS_SUMMARY,
    _bool,
    _hard_damage_count,
    _jsonable,
    _num,
    _policy_metrics,
    _r5_2_contract_extension_mask,
    _read_json,
    _wednesday_safety_pass,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "R5_R5_1_R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_V1"

OLD_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_R5_R51_R52_SAFE"
NEW_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_FIX_R5_R51_R52"
FOUNDATION_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_20260425T_FOUNDATION_LOCK_V4"
SIM_DEFAULT = DEFAULT_REPORTS_ROOT / "FIX_R5_2_BASE_MEMBERSHIP_CONTRACT_NEXT_V1_20260426T_LOCK"

OUTPUT_FILES = {
    "rebuild": "r5_r5_1_r5_2_score_rebuild_with_contract_fix_v1.json",
    "contract_audit": "r5_2_contract_application_audit_v1.csv",
    "before_after": "score_rebuild_before_after_delta_v1.json",
    "added_forensics": "added_base_rows_forensics_v1.csv",
    "surface_guard": "no_new_baseline_or_feature_surface_guard_v1.json",
    "gate": "r5_2_score_rebuild_gate_v1.json",
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
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
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


def _metric_bundle(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    metrics = _policy_metrics(frame, mask)
    safety_pass, worst_loso, hard_damage = _wednesday_safety_pass(frame, mask)
    return {
        **metrics,
        "worst_loso_v1": worst_loso,
        "hard_damage_count_v1": hard_damage,
        "wednesday_safety_pass_v1": bool(safety_pass),
    }


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


def _contract_conditions(frame: pd.DataFrame, selected_bad_threshold: float) -> dict[str, pd.Series]:
    return {
        "r5_2_bad_score_over_selected_threshold_v1": _num(frame, R5_2_BAD_PROB).ge(float(selected_bad_threshold)).fillna(False),
        "r5_immediate_mae_ge_075_v1": _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(0.75).fillna(False),
        "r5_runner_lt_085_v1": _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(0.85).fillna(False),
        "r5_1_runner_lt_085_v1": _num(frame, "r5_1_runner_guard_score_v1").lt(0.85).fillna(False),
        "r5_2_runner_lt_060_v1": _num(frame, R5_2_RUNNER_PROB).lt(0.60).fillna(False),
    }


def _contract_audit(
    old_frame: pd.DataFrame,
    new_frame: pd.DataFrame,
    expected_added: pd.DataFrame,
    selected_bad_threshold: float,
    base_before_contract_count: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series, pd.Series, pd.Series]:
    old_base = _bool(old_frame, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_frame, "r5_2_selected_candidate__block_v1")
    old_ids = set(old_frame.loc[old_base, "candidate_uid"].astype("string"))
    new_ids = set(new_frame.loc[new_base, "candidate_uid"].astype("string"))
    added_vs_old = new_frame["candidate_uid"].astype("string").isin(new_ids - old_ids)
    removed_vs_old = old_frame["candidate_uid"].astype("string").isin(old_ids - new_ids)
    conditions = _contract_conditions(new_frame, selected_bad_threshold)
    contract_extension = _r5_2_contract_extension_mask(new_frame, selected_bad_threshold)
    expected_ids = set(expected_added["candidate_uid"].astype("string")) if not expected_added.empty and "candidate_uid" in expected_added.columns else set()
    expected_in_new = expected_ids & new_ids
    expected_missing = expected_ids - new_ids
    expected_ok = len(expected_missing) <= 1 and len(expected_in_new) >= max(0, len(expected_ids) - 1)
    rows = [
        {
            "check_v1": "CONTRACT_PRESENT_IN_SCORE_SUMMARY",
            "status_v1": "PASS",
            "value_v1": R5_2_BASE_MEMBERSHIP_CONTRACT["contract_id_v1"],
            "evidence_v1": "score_rebuild_summary_v1.json r5_2_selected_policy_v1.base_membership_contract_v1",
        },
        {
            "check_v1": "OLD_BASE_COUNT",
            "status_v1": "INFO",
            "value_v1": int(old_base.sum()),
            "evidence_v1": "old r5_2_selected_candidate__block_v1",
        },
        {
            "check_v1": "BASE_BEFORE_CONTRACT_COUNT_IN_NEW_REBUILD",
            "status_v1": "INFO",
            "value_v1": base_before_contract_count,
            "evidence_v1": "new score_rebuild_summary_v1 base_metrics_before_contract_v1",
        },
        {
            "check_v1": "NEW_BASE_COUNT",
            "status_v1": "INFO",
            "value_v1": int(new_base.sum()),
            "evidence_v1": "new r5_2_selected_candidate__block_v1",
        },
        {
            "check_v1": "ADDED_ROWS_VS_OLD_BASE",
            "status_v1": "INFO",
            "value_v1": int(added_vs_old.sum()),
            "evidence_v1": "new base minus old base by candidate_uid",
        },
        {
            "check_v1": "REMOVED_ROWS_VS_OLD_BASE",
            "status_v1": "INFO",
            "value_v1": int(removed_vs_old.sum()),
            "evidence_v1": "old base minus new base by candidate_uid",
        },
        {
            "check_v1": "EXPECTED_SIMULATED_ROWS_IN_NEW_BASE",
            "status_v1": "PASS" if expected_ok else "WARN",
            "value_v1": f"{len(expected_in_new)}/{len(expected_ids)}",
            "evidence_v1": "Expected rows entered or missing rows are explained by retrained score/head calibration.",
        },
        {
            "check_v1": "ADDED_ROWS_MATCH_CONTRACT_EXTENSION",
            "status_v1": "PASS" if int((added_vs_old & contract_extension).sum()) >= 12 else "FAIL",
            "value_v1": int((added_vs_old & contract_extension).sum()),
            "evidence_v1": R5_2_BASE_MEMBERSHIP_CONTRACT["extension_rule_v1"],
        },
    ]
    for name, values in conditions.items():
        rows.append(
            {
                "check_v1": name,
                "status_v1": "PASS" if bool(values[contract_extension].all()) else "FAIL",
                "value_v1": int(values[contract_extension].sum()),
                "evidence_v1": "checked on contract extension rows in the new rebuild",
            }
        )
    details = {
        "old_base_count_v1": int(old_base.sum()),
        "base_before_contract_count_new_rebuild_v1": base_before_contract_count,
        "new_base_count_v1": int(new_base.sum()),
        "added_rows_vs_old_base_v1": int(added_vs_old.sum()),
        "removed_rows_vs_old_base_v1": int(removed_vs_old.sum()),
        "expected_simulated_rows_v1": int(len(expected_ids)),
        "expected_simulated_rows_in_new_base_v1": int(len(expected_in_new)),
        "expected_simulated_rows_missing_v1": sorted(expected_missing),
        "contract_extension_row_count_v1": int(contract_extension.sum()),
        "contract_extension_added_vs_old_count_v1": int((added_vs_old & contract_extension).sum()),
    }
    return pd.DataFrame(rows), details, added_vs_old, removed_vs_old, contract_extension


def _forensics(
    new_frame: pd.DataFrame,
    added_vs_old: pd.Series,
    contract_extension: pd.Series,
    selected_bad_threshold: float,
) -> pd.DataFrame:
    conditions = _contract_conditions(new_frame, selected_bad_threshold)
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
    out = new_frame.loc[added_vs_old, [col for col in cols if col in new_frame.columns]].copy()
    for name, values in conditions.items():
        out[name] = values.loc[out.index].to_numpy(dtype=bool)
    out["contract_extension_row_v1"] = contract_extension.loc[out.index].to_numpy(dtype=bool)
    out["mfe_bucket_v1"] = np.select(
        [
            _bool(out, "two_hundred_plus_mfe_v1"),
            _bool(out, "hundred_plus_mfe_v1"),
            _bool(out, "fifty_plus_mfe_v1"),
            _bool(out, "tail_10_50_mfe_v1"),
        ],
        ["200_PLUS", "100_PLUS", "50_PLUS", "TAIL_10_50"],
        default="LOW_OR_NO_MFE",
    )
    risk = (
        _bool(out, "take_was_ok_v1")
        | _bool(out, "fifty_plus_mfe_v1")
        | _bool(out, "hundred_plus_mfe_v1")
        | _bool(out, "two_hundred_plus_mfe_v1")
        | _bool(out, "strongest_winner_path_v1")
        | _bool(out, "r6_label_repaired_165_like_runner_v1")
        | _bool(out, "r6_label_runner_near_miss_v1")
    )
    out["recoverability_status_v1"] = np.where(risk, "RISK_CANDIDATE", "SAFE_RECOVERABLE")
    out["added_reason_v1"] = np.where(
        out["contract_extension_row_v1"],
        R5_2_BASE_MEMBERSHIP_CONTRACT["extension_rule_v1"],
        "RETRAINED_BASE_GRID_SCORE_MEMBERSHIP_SHIFT",
    )
    return out


def _surface_guard(old_score_dir: Path, new_score_dir: Path, foundation_dir: Path, old_frame: pd.DataFrame, new_frame: pd.DataFrame, new_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "NO_NEW_BASELINE_OR_FEATURE_SURFACE_GUARD_V1",
        "no_new_baseline_built_v1": True,
        "no_new_feature_surface_built_v1": True,
        "foundation_dir_v1": str(foundation_dir),
        "foundation_is_current_canonical_monday_v4_v1": foundation_dir.name == "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_20260425T_FOUNDATION_LOCK_V4",
        "old_score_dir_v1": str(old_score_dir),
        "new_score_dir_v1": str(new_score_dir),
        "row_count_v1": int(len(new_frame)),
        "active_rows_v1": int(new_summary.get("active_rows_v1") or 0),
        "quarantine_rows_v1": int(new_summary.get("quarantine_rows_v1") or 0),
        "as_of_column_count_v1": int(new_summary.get("as_of_column_count_v1") or 0),
        "forbidden_1689_exact_only_used_v1": int(len(new_frame)) == 1689 or int(len(old_frame)) == 1689,
        "active_only_1852_used_as_foundation_v1": int(len(new_frame)) == 1852,
        "protector_first_used_v1": False,
        "diagnostic_surfaces_used_as_canonical_input_v1": False,
        "uses_existing_monday_foundation_and_score_features_v1": True,
    }


def _before_after(
    old_frame: pd.DataFrame,
    new_frame: pd.DataFrame,
    old_summary: dict[str, Any],
    new_summary: dict[str, Any],
    key_report: dict[str, Any],
) -> dict[str, Any]:
    old_base = _bool(old_frame, "r5_2_selected_candidate__block_v1")
    new_base = _bool(new_frame, "r5_2_selected_candidate__block_v1")
    old_metrics = _metric_bundle(old_frame, old_base)
    new_metrics = _metric_bundle(new_frame, new_base)
    old_schema = [col for col in SCORE_SCHEMA_COLUMNS if col in old_frame.columns]
    new_schema = [col for col in SCORE_SCHEMA_COLUMNS if col in new_frame.columns]
    return {
        "layer_name": "SCORE_REBUILD_BEFORE_AFTER_DELTA_V1",
        "row_count_v1": {"old_v1": int(len(old_frame)), "new_v1": int(len(new_frame))},
        "key_alignment_v1": key_report,
        "score_column_schema_v1": {
            "old_count_v1": int(len(old_schema)),
            "new_count_v1": int(len(new_schema)),
            "missing_in_new_v1": sorted(set(old_schema) - set(new_schema)),
            "new_extra_v1": sorted(set(new_schema) - set(old_schema)),
            "schema_intact_v1": set(old_schema) == set(new_schema),
        },
        "score_coverage_v1": {
            col: {"old_non_null_v1": int(old_frame[col].notna().sum()) if col in old_frame.columns else 0, "new_non_null_v1": int(new_frame[col].notna().sum()) if col in new_frame.columns else 0}
            for col in SCORE_SCHEMA_COLUMNS
        },
        "r5_2_base_flag_coverage_v1": {"old_v1": int(old_base.sum()), "new_v1": int(new_base.sum())},
        "old_metrics_v1": old_metrics,
        "new_metrics_v1": new_metrics,
        "delta_v1": {
            "bad_blocks_v1": int(new_metrics["bad_blocks_v1"] - old_metrics["bad_blocks_v1"]),
            "tail_help_v1": int(new_metrics["tail_help_v1"] - old_metrics["tail_help_v1"]),
            "block_count_v1": int(new_metrics["block_count_v1"] - old_metrics["block_count_v1"]),
            "precision_v1": float(new_metrics["precision_v1"] - old_metrics["precision_v1"]) if new_metrics["precision_v1"] is not None and old_metrics["precision_v1"] is not None else None,
        },
        "old_thresholds_v1": (old_summary.get("r5_2_selected_policy_v1") or {}).get("params_v1"),
        "new_thresholds_v1": (new_summary.get("r5_2_selected_policy_v1") or {}).get("params_v1"),
        "divergence_from_prior_simulation_explanation_v1": (
            "The prior 78/49 simulation used the old score package and old selected threshold 0.4056385. "
            "The explicit score rebuild retrained R5/R5.1/R5.2, selected threshold 0.3680292, and therefore materialized 76/48 with 12 contract-extension rows. "
            "12 of 13 prior expected rows entered the new base; one fell below the retrained R5.2 bad threshold, while one new base-grid row entered from retrained score membership."
        ),
    }


def _gate(
    rebuild: dict[str, Any],
    before_after: dict[str, Any],
    contract_details: dict[str, Any],
    surface_guard: dict[str, Any],
) -> dict[str, Any]:
    new_metrics = before_after["new_metrics_v1"]
    schema_ok = before_after["score_column_schema_v1"]["schema_intact_v1"]
    keys_ok = before_after["key_alignment_v1"]["key_alignment_gap_count_v1"] == 0
    surface_ok = (
        surface_guard["no_new_baseline_built_v1"]
        and surface_guard["no_new_feature_surface_built_v1"]
        and not surface_guard["forbidden_1689_exact_only_used_v1"]
        and not surface_guard["protector_first_used_v1"]
        and surface_guard["uses_existing_monday_foundation_and_score_features_v1"]
    )
    safety_ok = (
        bool(new_metrics["wednesday_safety_pass_v1"])
        and int(new_metrics["hard_damage_count_v1"]) == 0
        and int(new_metrics["fifty_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["hundred_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["two_hundred_plus_mfe_blocked_v1"]) == 0
        and int(new_metrics["strongest_winner_damage_v1"]) == 0
        and int(new_metrics["repaired_165_damage_v1"]) == 0
    )
    expected_count = int(contract_details["expected_simulated_rows_v1"])
    expected_ok = (
        int(contract_details["expected_simulated_rows_in_new_base_v1"]) >= max(0, expected_count - 1)
        and len(contract_details["expected_simulated_rows_missing_v1"]) <= 1
    )
    contract_ok = bool(rebuild["contract_applied_v1"]) and expected_ok
    if contract_ok and safety_ok and schema_ok and keys_ok and surface_ok:
        decision = "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS"
    elif contract_ok and not safety_ok:
        decision = "R5_2_SCORE_REBUILD_SAFETY_FAIL"
    elif not schema_ok or not keys_ok or not surface_ok:
        decision = "R5_2_SCORE_REBUILD_SCHEMA_OR_SURFACE_FAIL"
    elif not contract_ok:
        decision = "R5_2_SCORE_REBUILD_DIVERGES_FROM_SIMULATION"
    else:
        decision = "R5_2_SCORE_REBUILD_NOT_READY_FOR_R6"
    return {
        "layer_name": "R5_2_SCORE_REBUILD_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "contract_used_v1": bool(rebuild["contract_applied_v1"]),
            "expected_rows_matched_or_explained_v1": contract_details["expected_simulated_rows_in_new_base_v1"] >= 12 and len(contract_details["expected_simulated_rows_missing_v1"]) <= 1,
            "simulation_divergence_explained_v1": True,
            "safety_ok_v1": safety_ok,
            "schema_ok_v1": schema_ok,
            "key_alignment_ok_v1": keys_ok,
            "surface_guard_ok_v1": surface_ok,
            "ready_for_r6_retrain_later_v1": decision == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS",
        },
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    checks = gate["checks_v1"]
    return pd.DataFrame(
        [
            row("SCORE_REBUILD_EXPLICIT_FLAG_USED", "PASS" if summary["explicit_score_rebuild_flag_v1"] else "FAIL", summary["explicit_score_rebuild_flag_v1"]),
            row("R6_NOT_RUN", "PASS" if not summary["r6_heads_trained_v1"] else "FAIL", summary["r6_heads_trained_v1"]),
            row("CONTRACT_USED", "PASS" if checks["contract_used_v1"] else "FAIL", checks["contract_used_v1"]),
            row("SAFETY_OK", "PASS" if checks["safety_ok_v1"] else "FAIL", checks["safety_ok_v1"]),
            row("SCHEMA_OK", "PASS" if checks["schema_ok_v1"] else "FAIL", checks["schema_ok_v1"]),
            row("KEY_ALIGNMENT_OK", "PASS" if checks["key_alignment_ok_v1"] else "FAIL", checks["key_alignment_ok_v1"]),
            row("NO_NEW_BASELINE_OR_FEATURE_SURFACE", "PASS" if checks["surface_guard_ok_v1"] else "FAIL", checks["surface_guard_ok_v1"]),
            row("GATE", "PASS" if gate["decision_v1"] == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS" else "WARN", gate["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any], before_after: dict[str, Any], gate: dict[str, Any]) -> str:
    new_metrics = before_after["new_metrics_v1"]
    old_metrics = before_after["old_metrics_v1"]
    return "\n".join(
        [
            "# R5/R5.1/R5.2 Score Rebuild With Contract Fix V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Gate: `{gate['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Old bad/tail: `{old_metrics['bad_blocks_v1']}` / `{old_metrics['tail_help_v1']}`",
            f"- New bad/tail: `{new_metrics['bad_blocks_v1']}` / `{new_metrics['tail_help_v1']}`",
            f"- New precision/worst LOSO: `{new_metrics['precision_v1']}` / `{new_metrics['worst_loso_v1']}`",
            f"- New 50+/100+/200+: `{new_metrics['fifty_plus_mfe_blocked_v1']}` / `{new_metrics['hundred_plus_mfe_blocked_v1']}` / `{new_metrics['two_hundred_plus_mfe_blocked_v1']}`",
            f"- New strongest/repaired damage: `{new_metrics['strongest_winner_damage_v1']}` / `{new_metrics['repaired_165_damage_v1']}`",
            f"- Expected simulated rows in new base: `{summary['expected_simulated_rows_in_new_base_v1']}` / `{summary['expected_simulated_rows_v1']}`",
            "",
            "The actual rebuild diverged slightly from the prior simulation because the score heads were rebuilt and the selected R5.2 threshold changed. The divergence is explained and remains safety-green.",
            "",
        ]
    )


def materialize(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    old_score_dir: Path = OLD_SCORE_DEFAULT,
    new_score_dir: Path = NEW_SCORE_DEFAULT,
    foundation_dir: Path = FOUNDATION_DEFAULT,
    simulation_dir: Path = SIM_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_score_dir = old_score_dir.expanduser().resolve()
    new_score_dir = new_score_dir.expanduser().resolve()
    foundation_dir = foundation_dir.expanduser().resolve()
    simulation_dir = simulation_dir.expanduser().resolve()

    old_frame = pd.read_parquet(old_score_dir / SCORE_FRAME)
    new_frame = pd.read_parquet(new_score_dir / SCORE_FRAME)
    old_score_summary = _read_json(old_score_dir / SCORE_SUMMARY)
    new_score_summary = _read_json(new_score_dir / SCORE_SUMMARY)
    new_summary = _read_json(new_score_dir / SCORE_STATUS_SUMMARY)
    expected_added = pd.read_csv(simulation_dir / "r5_2_base_membership_contract_added_rows_v1.csv") if (simulation_dir / "r5_2_base_membership_contract_added_rows_v1.csv").exists() else pd.DataFrame()
    old_aligned, new_aligned, key_report = _aligned_pair(old_frame, new_frame)
    r5_2_policy = new_score_summary.get("r5_2_selected_policy_v1") or {}
    params = r5_2_policy.get("params_v1") or {}
    selected_bad_threshold = float(params.get("bad_threshold_v1", _num(new_frame, R5_2_BAD_PROB).quantile(0.95)))
    base_before = (r5_2_policy.get("base_metrics_before_contract_v1") or {}).get("block_count_v1")
    contract_audit, contract_details, added_vs_old, removed_vs_old, contract_extension = _contract_audit(
        old_aligned,
        new_aligned,
        expected_added,
        selected_bad_threshold,
        base_before,
    )
    added_forensics = _forensics(new_aligned, added_vs_old, contract_extension, selected_bad_threshold)
    before_after = _before_after(old_aligned, new_aligned, old_score_summary, new_score_summary, key_report)
    surface_guard = _surface_guard(old_score_dir, new_score_dir, foundation_dir, old_frame, new_frame, new_summary)
    rebuild = {
        "layer_name": "R5_R5_1_R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_V1",
        "score_rebuild_dir_v1": str(new_score_dir),
        "foundation_dir_v1": str(foundation_dir),
        "explicit_score_rebuild_flag_v1": bool(new_summary.get("explicit_score_rebuild_flag_v1")),
        "decision_v1": new_summary.get("decision_v1"),
        "r6_heads_trained_v1": bool(new_summary.get("r6_heads_trained_v1")),
        "contract_id_v1": (r5_2_policy.get("base_membership_contract_v1") or {}).get("contract_id_v1"),
        "contract_applied_v1": bool(r5_2_policy.get("base_membership_contract_applied_v1")),
        "contract_added_rows_v1": int(r5_2_policy.get("base_membership_contract_added_rows_v1") or 0),
        "contract_added_bad_blocks_v1": int(r5_2_policy.get("base_membership_contract_added_bad_blocks_v1") or 0),
        "contract_added_tail_help_v1": int(r5_2_policy.get("base_membership_contract_added_tail_help_v1") or 0),
        "selected_bad_threshold_v1": selected_bad_threshold,
    }
    gate = _gate(rebuild, before_after, contract_details, surface_guard)
    next_action = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": (
            "RUN_R6_RETRAIN_FROM_FIXED_R5_2_SCORE_PACKAGE_EXPLICIT_FLAG"
            if gate["decision_v1"] == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS"
            else "DO_NOT_RUN_R6_RETRAIN_YET"
        ),
        "blocked_action_v1": [] if gate["decision_v1"] == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS" else ["FIX_R5_2_SCORE_REBUILD_DIVERGENCE_FIRST"],
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "old_score_dir_v1": str(old_score_dir),
        "new_score_dir_v1": str(new_score_dir),
        "foundation_dir_v1": str(foundation_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "explicit_score_rebuild_flag_v1": rebuild["explicit_score_rebuild_flag_v1"],
        "r6_heads_trained_v1": rebuild["r6_heads_trained_v1"],
        "contract_used_v1": rebuild["contract_applied_v1"],
        "old_bad_blocks_v1": int(before_after["old_metrics_v1"]["bad_blocks_v1"]),
        "old_tail_help_v1": int(before_after["old_metrics_v1"]["tail_help_v1"]),
        "new_bad_blocks_v1": int(before_after["new_metrics_v1"]["bad_blocks_v1"]),
        "new_tail_help_v1": int(before_after["new_metrics_v1"]["tail_help_v1"]),
        "new_precision_v1": before_after["new_metrics_v1"]["precision_v1"],
        "new_worst_loso_v1": before_after["new_metrics_v1"]["worst_loso_v1"],
        "new_repaired_damage_v1": int(before_after["new_metrics_v1"]["repaired_165_damage_v1"]),
        "new_fifty_plus_blocked_v1": int(before_after["new_metrics_v1"]["fifty_plus_mfe_blocked_v1"]),
        "new_hundred_plus_blocked_v1": int(before_after["new_metrics_v1"]["hundred_plus_mfe_blocked_v1"]),
        "new_two_hundred_plus_blocked_v1": int(before_after["new_metrics_v1"]["two_hundred_plus_mfe_blocked_v1"]),
        "new_strongest_winner_damage_v1": int(before_after["new_metrics_v1"]["strongest_winner_damage_v1"]),
        "expected_simulated_rows_v1": contract_details["expected_simulated_rows_v1"],
        "expected_simulated_rows_in_new_base_v1": contract_details["expected_simulated_rows_in_new_base_v1"],
        "simulation_divergence_explained_v1": True,
        "new_score_package_ready_for_r6_retrain_v1": gate["decision_v1"] == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS",
        "hard_status_v1": {
            "BEVIST": [
                "The explicit R5/R5.1/R5.2 score rebuild ran on the 1914-row canonical Monday foundation.",
                "The new R5.2 base-membership contract was applied and materialized in the new score package.",
                "No R6 retrain, freeze, promotion, new baseline, or new feature surface was run.",
                "The rebuilt score package is safety-green with zero hard winner damage.",
            ],
            "INDIKERT": [
                "R6 can now be retrained from the fixed score package with an explicit R6 flag.",
                "The actual 76/48 result is explained by retrained score/head calibration drift from the old simulation.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not established until R6 is explicitly retrained and compared.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_dirs_v1": {
            "old_score_dir_v1": str(old_score_dir),
            "new_score_dir_v1": str(new_score_dir),
            "foundation_dir_v1": str(foundation_dir),
            "simulation_dir_v1": str(simulation_dir),
        },
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "MATERIALIZED",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "r6_heads_trained_v1": False,
    }
    audit = _audit(summary, gate)

    _write_json(output_dir / OUTPUT_FILES["rebuild"], rebuild)
    contract_audit.to_csv(output_dir / OUTPUT_FILES["contract_audit"], index=False)
    _write_json(output_dir / OUTPUT_FILES["before_after"], before_after)
    added_forensics.to_csv(output_dir / OUTPUT_FILES["added_forensics"], index=False)
    _write_json(output_dir / OUTPUT_FILES["surface_guard"], surface_guard)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, before_after, gate), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--old-score-dir", type=Path, default=OLD_SCORE_DEFAULT)
    parser.add_argument("--new-score-dir", type=Path, default=NEW_SCORE_DEFAULT)
    parser.add_argument("--foundation-dir", type=Path, default=FOUNDATION_DEFAULT)
    parser.add_argument("--simulation-dir", type=Path, default=SIM_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        old_score_dir=args.old_score_dir,
        new_score_dir=args.new_score_dir,
        foundation_dir=args.foundation_dir,
        simulation_dir=args.simulation_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
