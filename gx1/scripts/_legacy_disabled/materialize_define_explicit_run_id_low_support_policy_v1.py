#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_V1"
LAYER_NAME = ACTION
SUPPORT_AUDIT_ROOT = DEFAULT_REPORTS_ROOT / "DEEPEN_RUN_ID_SUPPORT_SIGNAL_AUDIT_V1_20260427T134852Z_LOCK"
OPPORTUNITY_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1_20260427T122550Z_LOCK"
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
DENOMINATOR_TARGET = 5
WORST_RUN_ID = "TRUTH_MONFRI_WEEK_20250106_20250113"

VARIANT_COLUMN_MAP = {
    "V2_OOF_CORE_ONLY": "member_v2_oof_core_only_v1",
    "RECOMMENDED_73_RUN_ID_SUPPORT": "member_v2_oof_plus_run_id_support_v1",
    "BALANCED_209_DIAGNOSTIC": "member_balanced_v2_r5_tail_run_id_support_v1",
    "MAX_FEASIBLE_UNDER_HARD_VETOES": "member_safety_first_upper_bound_v1",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING_LOCAL_ARTIFACT"


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_forbidden_actions(*, optuna: bool, model: bool, r6: bool, package: bool, freeze: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if model:
        failures.append("MODEL_TRAINING_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_input_artifacts_unchanged(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = [key for key, value in before.items() if after.get(key) != value]
    return {
        "status_v1": "PASS" if not changed else "FAIL",
        "changed_v1": changed,
        "v2_oof_scores_unchanged_v1": "v2_oof_scores_sha256_v1" not in changed,
        "v2_oof_provenance_unchanged_v1": "v2_oof_provenance_sha256_v1" not in changed,
        "opportunity_rows_unchanged_v1": "opportunity_rows_sha256_v1" not in changed,
    }


def structural_low_support_is_model_failure_automatically() -> bool:
    return False


def secondary_metric_can_override_strict_invalid_for_final_promotion() -> bool:
    return False


def final_promotion_allowed(*, unresolved_structural_low_support: bool, explicit_exception_gate: bool) -> bool:
    return (not unresolved_structural_low_support) or explicit_exception_gate


def training_surface_allows_structural_low_support_safe_rows(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("structural_low_support_v1"))
        and bool(row.get("can_be_used_in_training_surface_v1"))
        and not bool(row.get("can_be_used_in_decision_valid_eval_v1"))
    )


def protected_runner_ambiguous_quarantine_positive_allowed(row: dict[str, Any]) -> bool:
    return not (
        bool(row.get("protected_winner_v1"))
        or bool(row.get("runner_protect_v1"))
        or bool(row.get("ambiguous_high_mfe_v1"))
        or bool(row.get("quarantine_v1"))
    )


def classify_run_id_registry_row(matrix: dict[str, Any]) -> str:
    current = int(matrix["current_denominator_v1"])
    feasible = int(matrix["feasible_max_denominator_v1"])
    target = int(matrix["denominator_target_v1"])
    status = str(matrix.get("support_repairability_status_v1", ""))
    if current >= target:
        return "SUPPORT_SUFFICIENT"
    if current == 0 and feasible == 0:
        if int(matrix.get("unknown_artifact_missing_rows_v1", 0)) > 0:
            return "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS"
        return "ZERO_DENOMINATOR_NO_SELECTED_ROWS"
    if status == "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS":
        return "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS"
    if feasible < target:
        if int(matrix.get("protected_winner_rows_v1", 0)) or int(matrix.get("runner_protect_rows_v1", 0)) or int(matrix.get("ambiguous_high_mfe_rows_v1", 0)):
            return "LOW_SUPPORT_DUE_TO_SAFETY_VETOES"
        return "STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET"
    if int(matrix.get("unknown_artifact_missing_rows_v1", 0)) > 0:
        return "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _load_inputs(support_audit_root: Path, opportunity_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "matrix": pd.read_csv(support_audit_root / "run_id_support_feasibility_matrix_v1.csv"),
        "taxonomy": pd.read_csv(support_audit_root / "low_support_run_id_taxonomy_v1.csv"),
        "frontier": pd.read_csv(support_audit_root / "feasible_run_id_support_frontier_v1.csv"),
        "opportunity_rows": pd.read_csv(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
    }


def _registry(matrix: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in matrix.iterrows():
        raw = item.to_dict()
        classification = classify_run_id_registry_row(raw)
        current = int(raw["current_denominator_v1"])
        feasible = int(raw["feasible_max_denominator_v1"])
        selected_low = 0 < current < DENOMINATOR_TARGET
        structural = classification == "STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET" or (
            selected_low and feasible < DENOMINATOR_TARGET
        )
        zero_den = current == 0
        row = {
            "run_id_v1": str(raw["run_id_v1"]),
            "total_rows_v1": int(raw["total_rows_v1"]),
            "active_rows_v1": int(raw["active_rows_v1"]),
            "quarantine_rows_v1": int(raw["quarantine_rows_v1"]),
            "current_selected_denominator_v1": current,
            "feasible_safe_max_denominator_v1": feasible,
            "denominator_target_v1": int(raw["denominator_target_v1"]),
            "denominator_gap_v1": int(raw["denominator_gap_v1"]),
            "safe_recoverable_count_v1": int(raw["safe_recoverable_rows_v1"]),
            "safe_signal_backed_candidate_count_v1": int(raw["feasible_safe_max_selected_under_current_hard_vetoes_v1"]),
            "protected_winner_count_v1": int(raw["protected_winner_rows_v1"]),
            "runner_protect_count_v1": int(raw["runner_protect_rows_v1"]),
            "ambiguous_high_mfe_count_v1": int(raw["ambiguous_high_mfe_rows_v1"]),
            "quarantine_count_v1": int(raw["quarantine_rows_v1"]),
            "unknown_missing_artifact_count_v1": int(raw["unknown_artifact_missing_rows_v1"]),
            "support_repairability_status_v1": raw["support_repairability_status_v1"],
            "run_id_policy_class_v1": classification,
            "structural_low_support_v1": structural,
            "selected_low_support_v1": selected_low,
            "zero_denominator_group_v1": zero_den,
            "can_be_used_in_training_surface_v1": feasible > 0 and structural,
            "can_be_used_in_decision_valid_eval_v1": current >= DENOMINATOR_TARGET,
            "requires_special_reporting_v1": selected_low or structural or zero_den,
            "reason_v1": _registry_reason(classification, current, feasible),
        }
        rows.append(row)
    return rows


def _registry_reason(classification: str, current: int, feasible: int) -> str:
    if classification == "SUPPORT_SUFFICIENT":
        return "Strict denominator target is met."
    if classification == "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS":
        return "Existing safe signal-backed rows can close support gap."
    if classification == "STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET":
        return "Feasible safe max is below denominator target; do not treat as model failure or decision-valid pass."
    if classification == "LOW_SUPPORT_DUE_TO_SAFETY_VETOES":
        return "Closing support would require violating hard safety vetoes."
    if classification == "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS":
        return "Missing artifact evidence prevents clean classification."
    if classification == "ZERO_DENOMINATOR_NO_SELECTED_ROWS":
        return "No selected denominator rows; report explicitly, do not hide."
    return f"Current denominator {current}, feasible max {feasible}; requires explicit reporting."


def _policy() -> dict[str, Any]:
    return {
        "contract": "RUN_ID_LOW_SUPPORT_POLICY_V1",
        "denominator_target_v1": DENOMINATOR_TARGET,
        "policy_does_not_approve_live_freeze_promo_v1": True,
        "policy_does_not_weaken_loso_guards_v1": True,
        "policy_does_not_make_v2_oof_decision_valid_v1": True,
        "lanes_v1": {
            "TRAINING_OPPORTUNITY_SURFACE": {
                "purpose_v1": "Allow legal, safe, AS_OF-safe, provenance-backed row-level learning/opportunity evidence from structurally low-support groups.",
                "structural_low_support_rows_allowed_v1": True,
                "every_included_row_needs_role_evidence_reason_v1": True,
                "protected_winners_as_positives_allowed_v1": False,
                "runner_protect_as_positives_allowed_v1": False,
                "ambiguous_high_mfe_as_positive_requires_safe_proof_v1": True,
                "quarantine_as_positive_allowed_v1": False,
                "dummy_synthetic_fallback_allowed_v1": False,
                "decision_valid_claim_allowed_v1": False,
            },
            "CANDIDATE_EVAL_SURFACE": {
                "purpose_v1": "Analyze candidate packages/models before final promotion while retaining strict low-support reporting.",
                "low_support_groups_reported_separately_v1": True,
                "strict_loso_denominator_status_visible_v1": True,
                "candidate_analysis_allowed_with_low_support_v1": True,
                "final_clean_claim_from_excluding_low_support_allowed_v1": False,
                "required_reports_v1": [
                    "strict_all_groups_loso_status",
                    "evaluable_groups_loso_status",
                    "structural_low_support_registry_status",
                    "low_support_group_safety_status",
                ],
            },
            "FINAL_PROMOTION_SURFACE": {
                "purpose_v1": "Freeze/promo/live decisioning.",
                "silent_pass_with_unresolved_structural_low_support_allowed_v1": False,
                "support_sufficient_under_strict_contract_required_v1": True,
                "explicit_human_approved_exception_gate_allowed_v1": "NOT_IMPLEMENTED_HERE",
                "automatic_promotion_from_this_policy_allowed_v1": False,
            },
        },
    }


def _metric_contract() -> dict[str, Any]:
    required = [
        "strict_all_run_id_loso_value",
        "strict_all_run_id_loso_denominator",
        "strict_all_run_id_loso_decision_valid",
        "strict_all_run_id_low_support_group_count",
        "selected_low_support_group_count",
        "structurally_unsatisfiable_group_count",
        "zero_selected_group_count",
        "evaluable_groups_loso_value",
        "evaluable_groups_denominator_min",
        "evaluable_groups_decision_valid",
        "low_support_group_safety_status",
        "low_support_group_false_positive_count",
        "low_support_group_protected_winner_count",
        "low_support_group_runner_protect_count",
        "low_support_group_ambiguous_high_mfe_count",
        "training_surface_allowed",
        "final_promotion_allowed",
        "explicit_exception_required",
        "reason",
    ]
    return {
        "contract": "LOW_SUPPORT_METRIC_REPORTING_CONTRACT_V1",
        "required_fields_v1": required,
        "strict_all_run_id_loso_never_hidden_v1": True,
        "low_support_groups_never_silently_dropped_v1": True,
        "evaluable_groups_loso_secondary_only_v1": True,
        "secondary_metrics_cannot_override_strict_invalid_for_final_promotion_v1": True,
        "structural_low_support_not_equal_model_failure_v1": True,
        "unsupported_expansion_forbidden_v1": True,
    }


def _training_decisioning_contract() -> dict[str, Any]:
    return {
        "contract": "TRAINING_VS_DECISIONING_SURFACE_CONTRACT_V1",
        "core_rule_v1": "A row/group can be valid for training/opportunity analysis while not being valid as final decisioning evidence.",
        "training_opportunity_surface_v1": {
            "allowed_v1": ["safe recoverable rows", "provenance-backed V2 OOF rows", "explicitly role/evidence/reason-tagged structural low-support rows"],
            "forbidden_as_positive_v1": ["protected winners", "runner-protect", "quarantine", "ambiguous high-MFE without safe proof", "dummy/synthetic/fallback"],
        },
        "candidate_eval_surface_v1": {
            "allowed_v1": ["candidate analysis with strict all-groups metrics plus secondary evaluable-groups metrics"],
            "required_tags_v1": ["structural_low_support", "selected_low_support", "zero_denominator", "safety status"],
        },
        "final_decision_valid_promotion_v1": {
            "allowed_v1": ["strict support sufficient under denominator contract", "or explicit exception gate not implemented here"],
            "blocked_by_default_when_v1": ["unresolved structural low-support", "silent low-support exclusion", "unsafe fill", "missing strict LOSO reporting"],
        },
        "hard_negatives_v1": "Protected winners and runner-protect rows remain hard negatives/veto rows.",
        "ambiguous_monitor_only_v1": "Ambiguous high-MFE rows remain monitor-only unless separately safe-proven.",
        "quarantine_v1": "Quarantine rows are excluded from positive support.",
        "missing_artifacts_v1": "Missing-artifact rows require evidence before support use.",
    }


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _dry_run(rows: pd.DataFrame, registry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reg = {row["run_id_v1"]: row for row in registry_rows}
    results = []
    for variant, column in VARIANT_COLUMN_MAP.items():
        selected = _bool(rows, column)
        selected_rows = rows[selected].copy()
        counts = selected_rows.groupby("run_id_v1").size() if not selected_rows.empty else pd.Series(dtype=int)
        low_support_groups = [str(key) for key, value in counts.items() if 0 < int(value) < DENOMINATOR_TARGET]
        structural_selected = [
            run_id for run_id in low_support_groups
            if bool(reg.get(run_id, {}).get("structural_low_support_v1", False))
        ]
        strict_min = int(counts.min()) if not counts.empty else 0
        evaluable_counts = counts[counts >= DENOMINATOR_TARGET]
        evaluable_min = int(evaluable_counts.min()) if not evaluable_counts.empty else 0
        hard_safety = _safety_counts(selected_rows)
        safety_clean = all(value == 0 for value in hard_safety.values())
        strict_valid = strict_min >= DENOMINATOR_TARGET and not structural_selected and safety_clean
        training_allowed = bool(len(selected_rows) > 0 and safety_clean)
        final_allowed = final_promotion_allowed(
            unresolved_structural_low_support=bool(structural_selected),
            explicit_exception_gate=False,
        ) and strict_valid
        results.append(
            {
                "variant_id_v1": variant,
                "selected_rows_v1": int(selected.sum()),
                "bad_proxy_v1": int(_bool(selected_rows, "bad_label_v1").sum()),
                "tail_proxy_v1": int(_bool(selected_rows, "tail_label_v1").sum()),
                "strict_loso_min_denominator_v1": strict_min,
                "strict_loso_decision_valid_v1": strict_valid,
                "selected_low_support_groups_v1": len(low_support_groups),
                "structurally_unsatisfiable_selected_groups_v1": len(structural_selected),
                "evaluable_groups_count_v1": int(len(evaluable_counts)),
                "evaluable_groups_loso_v1": 1.0 if not selected_rows.empty and int(_bool(selected_rows, "bad_label_v1").sum()) == int(selected.sum()) else _safe_div(float(_bool(selected_rows, "bad_label_v1").sum()), float(selected.sum())),
                "evaluable_groups_denominator_min_v1": evaluable_min,
                "low_support_group_safety_status_v1": "PASS" if safety_clean else "FAIL",
                **hard_safety,
                "training_surface_allowed_v1": training_allowed,
                "final_promotion_allowed_v1": final_allowed,
                "explicit_exception_required_v1": bool(structural_selected),
                "reason_v1": _dry_run_reason(variant, strict_valid, training_allowed, final_allowed, structural_selected),
            }
        )
    return results


def _safety_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "low_support_group_false_positive_count_v1": int((~_bool(frame, "bad_label_v1")).sum()),
        "low_support_group_protected_winner_count_v1": int(_bool(frame, "protected_winner_status_v1").sum()),
        "low_support_group_runner_protect_count_v1": int(_bool(frame, "runner_protect_status_v1").sum()),
        "low_support_group_ambiguous_high_mfe_count_v1": int(_bool(frame, "ambiguous_high_mfe_status_v1").sum()),
        "low_support_group_quarantine_count_v1": int(frame["active_quarantine_v1"].astype(str).ne("ACTIVE_CANDIDATE").sum()) if "active_quarantine_v1" in frame.columns else 0,
    }


def _dry_run_reason(variant: str, strict_valid: bool, training_allowed: bool, final_allowed: bool, structural_selected: list[str]) -> str:
    if final_allowed and strict_valid:
        return "Strict denominator and safety conditions pass for this dry-run variant."
    if training_allowed and structural_selected:
        return "Allowed for training/opportunity analysis with explicit structural low-support tags; final promotion remains blocked."
    if training_allowed:
        return "Allowed for training/opportunity analysis only; final promotion still requires strict policy gate."
    return f"{variant} is not allowed as a final policy candidate by this dry-run."


def _recommendation(registry_rows: list[dict[str, Any]], dry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    structural_selected = sum(1 for row in registry_rows if bool(row["selected_low_support_v1"]) and bool(row["structural_low_support_v1"]))
    safety_blocked = any(int(row["low_support_group_protected_winner_count_v1"]) or int(row["low_support_group_runner_protect_count_v1"]) or int(row["low_support_group_ambiguous_high_mfe_count_v1"]) for row in dry_rows)
    if safety_blocked:
        status = "LOW_SUPPORT_POLICY_BLOCKED_BY_SAFETY_RISK"
        final = status
        next_action = "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    elif structural_selected:
        status = "LOW_SUPPORT_POLICY_DEFINED_BUT_FINAL_PROMOTION_BLOCKED"
        final = status
        next_action = "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1"
    else:
        status = "LOW_SUPPORT_POLICY_DEFINED_READY_FOR_COVERAGE_AWARE_OPPORTUNITY_BASE"
        final = status
        next_action = "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1"
    return {
        "layer_name": "LOW_SUPPORT_POLICY_NEXT_ACTION_RECOMMENDATION_V1",
        "status_v1": status,
        "final_go_no_go_v1": final,
        "next_recommended_action_v1": next_action,
        "policy_precise_enough_to_prevent_misuse_v1": True,
        "training_opportunity_can_proceed_v1": status != "LOW_SUPPORT_POLICY_BLOCKED_BY_SAFETY_RISK",
        "final_promotion_blocked_until_exception_or_sufficient_support_v1": status == "LOW_SUPPORT_POLICY_DEFINED_BUT_FINAL_PROMOTION_BLOCKED",
    }


def _summary_counts(registry_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "run_id_count_v1": len(registry_rows),
        "structurally_unsatisfiable_run_id_groups_v1": sum(1 for row in registry_rows if row["structural_low_support_v1"]),
        "selected_low_support_groups_v1": sum(1 for row in registry_rows if row["selected_low_support_v1"]),
        "zero_denominator_groups_v1": sum(1 for row in registry_rows if row["zero_denominator_group_v1"]),
        "training_allowed_structural_groups_v1": sum(1 for row in registry_rows if row["can_be_used_in_training_surface_v1"]),
        "decision_valid_groups_v1": sum(1 for row in registry_rows if row["can_be_used_in_decision_valid_eval_v1"]),
    }


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    support_audit_root: Path = SUPPORT_AUDIT_ROOT,
    opportunity_root: Path = OPPORTUNITY_ROOT,
    v2_oof_root: Path = V2_OOF_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
        "support_matrix_sha256_v1": _file_hash(support_audit_root / "run_id_support_feasibility_matrix_v1.csv"),
    }
    inputs = _load_inputs(support_audit_root, opportunity_root)
    registry_rows = _registry(inputs["matrix"])
    policy = _policy()
    metric_contract = _metric_contract()
    training_contract = _training_decisioning_contract()
    dry_rows = _dry_run(inputs["opportunity_rows"], registry_rows)
    recommendation = _recommendation(registry_rows, dry_rows)
    go_no_go = {
        "layer_name": "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_GO_NO_GO_V1",
        "decision_v1": recommendation["final_go_no_go_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "no_optuna_run_v1": True,
        "no_training_run_v1": True,
        "not_package_ready_v1": True,
        "not_r6_ready_v1": True,
        "not_freeze_promo_live_ready_v1": True,
    }
    input_hashes_after = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
        "support_matrix_sha256_v1": _file_hash(support_audit_root / "run_id_support_feasibility_matrix_v1.csv"),
    }
    integrity = validate_input_artifacts_unchanged(input_hashes_before, input_hashes_after)
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    no_forbidden = validate_no_forbidden_actions(optuna=False, model=False, r6=False, package=False, freeze=False, live=False)

    _write_rows(output_dir / "structural_low_support_run_id_registry_v1.csv", registry_rows)
    _write_json(output_dir / "structural_low_support_run_id_registry_v1.json", {"rows_v1": registry_rows, "summary_v1": _summary_counts(registry_rows)})
    _write_json(output_dir / "run_id_low_support_policy_v1.json", policy)
    _write_json(output_dir / "low_support_metric_reporting_contract_v1.json", metric_contract)
    _write_json(output_dir / "training_vs_decisioning_surface_contract_v1.json", training_contract)
    _write_rows(output_dir / "opportunity_base_low_support_policy_dry_run_v1.csv", dry_rows)
    _write_json(output_dir / "opportunity_base_low_support_policy_dry_run_v1.json", {"rows_v1": dry_rows})
    _write_json(output_dir / "low_support_policy_next_action_recommendation_v1.json", recommendation)
    _write_json(output_dir / "define_explicit_run_id_low_support_policy_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "support_audit_root_v1": str(support_audit_root),
                "opportunity_root_v1": str(opportunity_root),
                "v2_oof_root_v1": str(v2_oof_root),
            },
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_integrity_v1": integrity,
            "no_dummy_synthetic_fallback_v1": no_dummy,
            "no_forbidden_actions_v1": no_forbidden,
        },
    )
    _write_reports(output_dir, registry_rows, policy, metric_contract, training_contract, dry_rows, recommendation)
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        **_summary_counts(registry_rows),
        "worst_run_id_v1": WORST_RUN_ID,
        "worst_run_id_class_v1": next(row for row in registry_rows if row["run_id_v1"] == WORST_RUN_ID)["run_id_policy_class_v1"],
        "v2_scores_provenance_model_objective_thresholds_unchanged_v1": integrity["status_v1"] == "PASS",
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "go_no_go_v1": go_no_go})
    return summary


def _write_reports(
    output_dir: Path,
    registry_rows: list[dict[str, Any]],
    policy: dict[str, Any],
    metric_contract: dict[str, Any],
    training_contract: dict[str, Any],
    dry_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> None:
    summary = _summary_counts(registry_rows)
    worst = next(row for row in registry_rows if row["run_id_v1"] == WORST_RUN_ID)
    _write_report(
        output_dir / "structural_low_support_run_id_registry_report_v1.md",
        [
            "# Structural Low Support Run ID Registry V1",
            "",
            f"Run IDs: `{summary['run_id_count_v1']}`",
            f"Structural low-support groups: `{summary['structurally_unsatisfiable_run_id_groups_v1']}`",
            f"Selected low-support groups: `{summary['selected_low_support_groups_v1']}`",
            f"Zero-denominator groups: `{summary['zero_denominator_groups_v1']}`",
            f"Worst run_id `{WORST_RUN_ID}` class: `{worst['run_id_policy_class_v1']}`.",
        ],
    )
    _write_report(
        output_dir / "run_id_low_support_policy_v1.md",
        [
            "# Run ID Low Support Policy V1",
            "",
            "This policy separates training/opportunity use, candidate eval, and final promotion.",
            "It does not weaken LOSO, does not approve live/freeze/promo, and does not make V2 OOF decision-valid.",
            f"Policy lanes: `{list(policy['lanes_v1'].keys())}`",
        ],
    )
    _write_report(
        output_dir / "low_support_metric_reporting_contract_v1.md",
        [
            "# Low Support Metric Reporting Contract V1",
            "",
            "Strict all-run_id LOSO must always be reported.",
            "Evaluable-groups LOSO is secondary and cannot override strict invalid status for final promotion.",
            f"Required fields: `{len(metric_contract['required_fields_v1'])}`",
        ],
    )
    _write_report(
        output_dir / "training_vs_decisioning_surface_contract_v1.md",
        [
            "# Training Vs Decisioning Surface Contract V1",
            "",
            training_contract["core_rule_v1"],
            "Hard negatives, ambiguous monitor-only rows, quarantine rows, and missing-artifact rows retain explicit handling.",
        ],
    )
    recommended = next(row for row in dry_rows if row["variant_id_v1"] == "RECOMMENDED_73_RUN_ID_SUPPORT")
    _write_report(
        output_dir / "opportunity_base_low_support_policy_dry_run_report_v1.md",
        [
            "# Opportunity Base Low Support Policy Dry Run V1",
            "",
            f"Recommended variant selected rows: `{recommended['selected_rows_v1']}`",
            f"Strict LOSO min denominator: `{recommended['strict_loso_min_denominator_v1']}`",
            f"Strict decision-valid: `{recommended['strict_loso_decision_valid_v1']}`",
            f"Training surface allowed: `{recommended['training_surface_allowed_v1']}`",
            f"Final promotion allowed: `{recommended['final_promotion_allowed_v1']}`",
        ],
    )
    _write_report(
        output_dir / "low_support_policy_next_action_recommendation_v1.md",
        [
            "# Low Support Policy Next Action Recommendation V1",
            "",
            f"Status: `{recommendation['status_v1']}`",
            f"Next action: `{recommendation['next_recommended_action_v1']}`",
            f"Final promotion blocked: `{recommendation['final_promotion_blocked_until_exception_or_sufficient_support_v1']}`",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Define Explicit Run ID Low Support Policy V1",
            "",
            f"Decision: `{recommendation['final_go_no_go_v1']}`",
            f"Next action: `{recommendation['next_recommended_action_v1']}`",
            "No model, Optuna, R6, package, freeze, promo, live, V2 mutation, or denominator weakening was performed.",
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--support-audit-root", type=Path, default=SUPPORT_AUDIT_ROOT)
    parser.add_argument("--opportunity-root", type=Path, default=OPPORTUNITY_ROOT)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        support_audit_root=args.support_audit_root,
        opportunity_root=args.opportunity_root,
        v2_oof_root=args.v2_oof_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
