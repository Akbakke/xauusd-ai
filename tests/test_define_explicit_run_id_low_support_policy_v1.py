from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_define_explicit_run_id_low_support_policy_v1 as policy


def _matrix_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id_v1": policy.WORST_RUN_ID,
        "total_rows_v1": 10,
        "active_rows_v1": 10,
        "quarantine_rows_v1": 0,
        "safe_recoverable_rows_v1": 2,
        "protected_winner_rows_v1": 0,
        "runner_protect_rows_v1": 0,
        "ambiguous_high_mfe_rows_v1": 0,
        "unknown_artifact_missing_rows_v1": 0,
        "feasible_safe_max_selected_under_current_hard_vetoes_v1": 2,
        "current_denominator_v1": 2,
        "feasible_max_denominator_v1": 2,
        "denominator_target_v1": policy.DENOMINATOR_TARGET,
        "denominator_gap_v1": 3,
        "support_repairability_status_v1": "STRUCTURALLY_UNSATISFIABLE_FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR",
    }
    row.update(overrides)
    return row


def _toy_opportunity_rows() -> pd.DataFrame:
    rows = []
    for idx in range(2):
        rows.append(
            {
                "candidate_uid_v1": f"c{idx}",
                "run_id_v1": policy.WORST_RUN_ID,
                "active_quarantine_v1": "ACTIVE_CANDIDATE",
                "bad_label_v1": True,
                "tail_label_v1": True,
                "protected_winner_status_v1": False,
                "runner_protect_status_v1": False,
                "ambiguous_high_mfe_status_v1": False,
                "member_v2_oof_core_only_v1": True,
                "member_v2_oof_plus_run_id_support_v1": True,
                "member_balanced_v2_r5_tail_run_id_support_v1": True,
                "member_safety_first_upper_bound_v1": True,
            }
        )
    for idx in range(5):
        rows.append(
            {
                "candidate_uid_v1": f"s{idx}",
                "run_id_v1": "SUPPORTED_RUN",
                "active_quarantine_v1": "ACTIVE_CANDIDATE",
                "bad_label_v1": True,
                "tail_label_v1": False,
                "protected_winner_status_v1": False,
                "runner_protect_status_v1": False,
                "ambiguous_high_mfe_status_v1": False,
                "member_v2_oof_core_only_v1": False,
                "member_v2_oof_plus_run_id_support_v1": True,
                "member_balanced_v2_r5_tail_run_id_support_v1": True,
                "member_safety_first_upper_bound_v1": True,
            }
        )
    return pd.DataFrame(rows)


def test_structural_low_support_is_not_model_failure_automatically() -> None:
    assert policy.structural_low_support_is_model_failure_automatically() is False


def test_structural_low_support_is_not_final_decision_valid_pass() -> None:
    assert policy.final_promotion_allowed(unresolved_structural_low_support=True, explicit_exception_gate=False) is False


def test_low_support_groups_cannot_be_silently_dropped_from_strict_reporting() -> None:
    contract = policy._metric_contract()

    assert contract["strict_all_run_id_loso_never_hidden_v1"] is True
    assert contract["low_support_groups_never_silently_dropped_v1"] is True


def test_secondary_metrics_cannot_override_strict_invalid_for_final_promotion() -> None:
    assert policy.secondary_metric_can_override_strict_invalid_for_final_promotion() is False
    assert policy._metric_contract()["secondary_metrics_cannot_override_strict_invalid_for_final_promotion_v1"] is True


def test_training_surface_can_include_structural_low_support_safe_rows_with_tags() -> None:
    row = {
        "structural_low_support_v1": True,
        "can_be_used_in_training_surface_v1": True,
        "can_be_used_in_decision_valid_eval_v1": False,
    }

    assert policy.training_surface_allows_structural_low_support_safe_rows(row) is True


def test_final_promotion_requires_exception_gate_when_structural_low_support_unresolved() -> None:
    assert policy.final_promotion_allowed(unresolved_structural_low_support=True, explicit_exception_gate=False) is False
    assert policy.final_promotion_allowed(unresolved_structural_low_support=True, explicit_exception_gate=True) is True


def test_worst_run_id_appears_in_registry() -> None:
    registry = policy._registry(pd.DataFrame([_matrix_row()]))

    assert registry[0]["run_id_v1"] == policy.WORST_RUN_ID
    assert registry[0]["run_id_policy_class_v1"] == "STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET"


def test_feasible_safe_max_below_target_classifies_as_structural_low_support() -> None:
    assert policy.classify_run_id_registry_row(_matrix_row()) == "STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET"


def test_protected_winners_cannot_repair_low_support() -> None:
    assert policy.protected_runner_ambiguous_quarantine_positive_allowed({"protected_winner_v1": True}) is False


def test_runner_protect_rows_cannot_repair_low_support() -> None:
    assert policy.protected_runner_ambiguous_quarantine_positive_allowed({"runner_protect_v1": True}) is False


def test_ambiguous_high_mfe_rows_cannot_be_positive_without_safe_proof() -> None:
    assert policy.protected_runner_ambiguous_quarantine_positive_allowed({"ambiguous_high_mfe_v1": True}) is False


def test_quarantine_rows_cannot_be_positive_support() -> None:
    assert policy.protected_runner_ambiguous_quarantine_positive_allowed({"quarantine_v1": True}) is False


def test_zero_denominator_groups_are_reported_not_hidden() -> None:
    registry = policy._registry(
        pd.DataFrame(
            [
                _matrix_row(
                    run_id_v1="ZERO_RUN",
                    current_denominator_v1=0,
                    feasible_max_denominator_v1=0,
                    denominator_gap_v1=policy.DENOMINATOR_TARGET,
                    feasible_safe_max_selected_under_current_hard_vetoes_v1=0,
                )
            ]
        )
    )

    assert registry[0]["run_id_policy_class_v1"] == "ZERO_DENOMINATOR_NO_SELECTED_ROWS"
    assert registry[0]["zero_denominator_group_v1"] is True
    assert registry[0]["requires_special_reporting_v1"] is True


def test_dry_run_does_not_modify_original_artifact_hashes() -> None:
    before = {
        "v2_oof_scores_sha256_v1": "scores",
        "v2_oof_provenance_sha256_v1": "provenance",
        "opportunity_rows_sha256_v1": "rows",
    }
    after = dict(before)

    assert policy.validate_input_artifacts_unchanged(before, after)["status_v1"] == "PASS"


def test_policy_does_not_change_v2_scores_provenance_model_or_thresholds() -> None:
    before = {"v2_oof_scores_sha256_v1": "a", "v2_oof_provenance_sha256_v1": "b"}
    changed = {"v2_oof_scores_sha256_v1": "z", "v2_oof_provenance_sha256_v1": "b"}

    result = policy.validate_input_artifacts_unchanged(before, changed)

    assert result["status_v1"] == "FAIL"
    assert result["v2_oof_scores_unchanged_v1"] is False


def test_no_forbidden_optuna_model_r6_package_freeze_or_live() -> None:
    clean = policy.validate_no_forbidden_actions(optuna=False, model=False, r6=False, package=False, freeze=False, live=False)
    blocked = policy.validate_no_forbidden_actions(optuna=True, model=True, r6=True, package=True, freeze=True, live=True)

    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert blocked["failures_v1"] == [
        "OPTUNA_FORBIDDEN",
        "MODEL_TRAINING_FORBIDDEN",
        "R6_FORBIDDEN",
        "PACKAGE_BUILD_FORBIDDEN",
        "FREEZE_PROMO_FORBIDDEN",
        "LIVE_FORBIDDEN",
    ]


def test_no_dummy_synthetic_or_fallback() -> None:
    result = policy.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=True, fallback=True)

    assert result["status_v1"] == "FAIL"
    assert result["failures_v1"] == ["DUMMY_INPUT_FORBIDDEN", "SYNTHETIC_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN"):
        policy.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_dry_run_keeps_recommended_variant_training_only_when_structural_low_support_selected() -> None:
    registry = policy._registry(
        pd.DataFrame(
            [
                _matrix_row(),
                _matrix_row(
                    run_id_v1="SUPPORTED_RUN",
                    total_rows_v1=5,
                    active_rows_v1=5,
                    safe_recoverable_rows_v1=5,
                    feasible_safe_max_selected_under_current_hard_vetoes_v1=5,
                    current_denominator_v1=5,
                    feasible_max_denominator_v1=5,
                    denominator_gap_v1=0,
                    support_repairability_status_v1="SUPPORT_ALREADY_SUFFICIENT",
                ),
            ]
        )
    )
    dry = policy._dry_run(_toy_opportunity_rows(), registry)
    recommended = next(row for row in dry if row["variant_id_v1"] == "RECOMMENDED_73_RUN_ID_SUPPORT")

    assert recommended["training_surface_allowed_v1"] is True
    assert recommended["strict_loso_decision_valid_v1"] is False
    assert recommended["final_promotion_allowed_v1"] is False
    assert recommended["explicit_exception_required_v1"] is True


def test_recommendation_blocks_final_promotion_when_selected_structural_low_support_exists() -> None:
    registry = policy._registry(pd.DataFrame([_matrix_row()]))
    dry = policy._dry_run(_toy_opportunity_rows(), registry)
    recommendation = policy._recommendation(registry, dry)

    assert recommendation["status_v1"] == "LOW_SUPPORT_POLICY_DEFINED_BUT_FINAL_PROMOTION_BLOCKED"
    assert recommendation["next_recommended_action_v1"] == "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1"
