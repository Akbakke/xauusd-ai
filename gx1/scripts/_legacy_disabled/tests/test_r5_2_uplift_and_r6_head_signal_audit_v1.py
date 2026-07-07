from __future__ import annotations

import pytest

from gx1.scripts import materialize_r5_2_uplift_and_r6_head_signal_audit_v1 as audit


def test_audit_must_include_required_fixed_comparisons() -> None:
    assert audit.validate_required_comparisons_present(audit.FIXED_COMPARISONS) is True
    incomplete = dict(audit.FIXED_COMPARISONS)
    incomplete.pop("wednesday_180_149")
    with pytest.raises(RuntimeError, match="REQUIRED_FIXED_COMPARISONS_MISSING"):
        audit.validate_required_comparisons_present(incomplete)


def test_uplift_attribution_selected_rows_have_explicit_classes() -> None:
    row = {"v2_oof_captured_v1": True}
    assert audit._contribution_class(row, selected=True, coverage_proxy=True) == "RETAINED_FROM_V2_OOF"
    row = {"opportunity_role_v1": "COVERAGE_EXPANSION_TAIL", "source_evidence_v1": "R5_TAIL_SCORE:STRONG"}
    assert audit._contribution_class(row, selected=True, coverage_proxy=True) == "GAINED_FROM_TAIL_SIGNAL"


def test_gap_analysis_must_not_treat_coverage_proxy_as_final_candidate() -> None:
    payload = {"final_promotion_allowed_v1": False, "model_trained_v1": False, "r6_ready_v1": False}
    assert audit.coverage_proxy_is_not_final_candidate(payload) is True
    with pytest.raises(RuntimeError, match="COVERAGE_PROXY_CANNOT_BE_FINAL_CANDIDATE"):
        audit.coverage_proxy_is_not_final_candidate({"final_promotion_allowed_v1": True})


def test_tail_gap_rows_cannot_be_recommended_if_safety_blocked() -> None:
    with pytest.raises(RuntimeError, match="TAIL_GAP_SAFETY_BLOCKED"):
        audit.validate_tail_gap_recommendation(
            {
                "recommended_next_use_v1": "TAIL_REPAIR_CANDIDATE",
                "fifty_plus_mfe_risk_v1": True,
            }
        )


def test_r6_head_audit_must_include_all_five_heads() -> None:
    assert audit.R6_HEADS == [
        "bad_risk",
        "runner_protector",
        "tail_control_10_50",
        "risky_allow",
        "batch04_blindspot",
    ]


def test_failed_r6_expansion_candidates_cannot_be_promoted() -> None:
    assert audit.failed_expansion_can_be_promoted({"candidate_constraint_pass_v1": False, "safety_clean_v1": False}) is False
    assert audit.failed_expansion_can_be_promoted({"candidate_constraint_pass_v1": True, "safety_clean_v1": True}) is True


def test_safe_subset_mining_does_not_change_thresholds_or_selected_candidate() -> None:
    no_forbidden = audit.validate_no_forbidden_actions(
        optuna=False,
        model=False,
        package=False,
        r6_rerun=False,
        freeze=False,
        live=False,
    )
    assert no_forbidden["status_v1"] == "PASS"


def test_anti_overfit_audit_fails_if_in_sample_detected() -> None:
    result = audit.validate_anti_overfit_audit(
        {
            "no_in_sample_decisioning_v1": False,
            "oof_provenance_pass_v1": True,
            "train_validation_overlap_zero_v1": True,
            "fixed_controls_included_v1": True,
            "no_large_sweep_v1": True,
            "no_optuna_v1": True,
            "strict_loso_visible_v1": True,
            "low_support_visible_v1": True,
            "selected_candidate_safety_clean_v1": True,
            "no_dummy_synthetic_fallback_v1": True,
            "no_new_feature_surface_v1": True,
        }
    )
    assert result["status_v1"] == "FAIL"
    assert "no_in_sample_decisioning_v1" in result["failures_v1"]


def test_anti_overfit_audit_fails_if_oof_provenance_missing() -> None:
    result = audit.validate_anti_overfit_audit(
        {
            "no_in_sample_decisioning_v1": True,
            "oof_provenance_pass_v1": False,
            "train_validation_overlap_zero_v1": True,
            "fixed_controls_included_v1": True,
            "no_large_sweep_v1": True,
            "no_optuna_v1": True,
            "strict_loso_visible_v1": True,
            "low_support_visible_v1": True,
            "selected_candidate_safety_clean_v1": True,
            "no_dummy_synthetic_fallback_v1": True,
            "no_new_feature_surface_v1": True,
        }
    )
    assert result["status_v1"] == "FAIL"
    assert "oof_provenance_pass_v1" in result["failures_v1"]


def test_low_support_and_strict_loso_must_remain_visible() -> None:
    result = audit.validate_anti_overfit_audit(
        {
            "no_in_sample_decisioning_v1": True,
            "oof_provenance_pass_v1": True,
            "train_validation_overlap_zero_v1": True,
            "fixed_controls_included_v1": True,
            "no_large_sweep_v1": True,
            "no_optuna_v1": True,
            "strict_loso_visible_v1": False,
            "low_support_visible_v1": False,
            "selected_candidate_safety_clean_v1": True,
            "no_dummy_synthetic_fallback_v1": True,
            "no_new_feature_surface_v1": True,
        }
    )
    assert result["status_v1"] == "FAIL"
    assert "strict_loso_visible_v1" in result["failures_v1"]
    assert "low_support_visible_v1" in result["failures_v1"]


def test_no_optuna_model_package_r6_rerun_freeze_live_in_audit() -> None:
    blocked = audit.validate_no_forbidden_actions(
        optuna=True,
        model=True,
        package=True,
        r6_rerun=True,
        freeze=True,
        live=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_no_dummy_synthetic_fallback() -> None:
    clean = audit.validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    blocked = audit.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=True, fallback=True)
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"


def test_no_implicit_latest_glob_artifact_selection() -> None:
    assert audit.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        audit.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_recommendation_must_not_suggest_blind_sweep() -> None:
    assert audit.validate_recommendation_not_blind_sweep({"next_recommended_action_v1": "BUILD_TAIL_SPECIFIC_R5_2_R6_REPAIR_V1"}) is True
    with pytest.raises(RuntimeError, match="BLIND_SWEEP"):
        audit.validate_recommendation_not_blind_sweep({"next_recommended_action_v1": "RUN_MORE_OPTUNA_BLIND_SWEEP"})
