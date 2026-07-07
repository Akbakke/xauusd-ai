from __future__ import annotations

import pytest

from gx1.scripts import materialize_stability_recheck_best_lane_185_139_before_r6_v1 as recheck


def _repro(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "selected_lane_id_v1": "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY",
        "selected_rows_v1": 185,
        "bad_count_v1": 185,
        "tail_count_v1": 139,
        "precision_v1": 1.0,
        "precision_denominator_v1": 185,
        "strict_loso_denominator_v1": 2,
        "selected_low_support_group_count_v1": 9,
        "structural_low_support_selected_group_count_v1": 7,
        "added_rows_count_v1": 45,
        "added_bad_rows_v1": 45,
        "added_tail_rows_v1": 45,
        "safety_clean_v1": True,
    }
    payload.update(overrides)
    return payload


def test_recheck_must_reproduce_185_139_and_delta_exactly() -> None:
    assert recheck.validate_reproducibility(_repro()) is True
    with pytest.raises(RuntimeError, match="BEST_LANE_REPRODUCIBILITY_FAILURE"):
        recheck.validate_reproducibility(_repro(bad_count_v1=184))
    with pytest.raises(RuntimeError, match="BEST_LANE_REPRODUCIBILITY_FAILURE"):
        recheck.validate_reproducibility(_repro(added_tail_rows_v1=44))


def test_recheck_preserves_safety_strict_loso_and_low_support_visibility() -> None:
    with pytest.raises(RuntimeError, match="BEST_LANE_REPRODUCIBILITY_FAILURE"):
        recheck.validate_reproducibility(_repro(safety_clean_v1=False))
    with pytest.raises(RuntimeError, match="BEST_LANE_REPRODUCIBILITY_FAILURE"):
        recheck.validate_reproducibility(_repro(strict_loso_denominator_v1=5))
    with pytest.raises(RuntimeError, match="BEST_LANE_REPRODUCIBILITY_FAILURE"):
        recheck.validate_reproducibility(_repro(selected_low_support_group_count_v1=0))


def test_added_rows_require_evidence_and_reason() -> None:
    rows = [{"row_id_v1": "a", "signal_evidence_v1": "R5_1_BAD_SCORE:SUPPORT"}]
    assert recheck.validate_added_rows_have_evidence(rows) is True
    with pytest.raises(RuntimeError, match="ADDED_ROWS_REQUIRE_EVIDENCE"):
        recheck.validate_added_rows_have_evidence([{"row_id_v1": "b", "signal_evidence_v1": ""}])


def test_label_or_membership_selection_is_flagged_not_causal() -> None:
    classification = recheck.classify_added_row_selection(
        source_lane_logic="BASE_PLUS_SAFETY_CLEAR_TAIL_GAP_ROWS",
        signal_evidence="R5_1_BAD_SCORE:SUPPORT",
        selected_from_coverage_proxy_membership=True,
        selected_from_tail_gap_membership=True,
        final_bad_label_available_in_source=True,
        final_tail_label_available_in_source=True,
        post_outcome_safety_used=True,
        as_of_score_only=False,
    )
    assert classification == "MEMBERSHIP_ONLY_NOT_CAUSALLY_SCORABLE"


def test_as_of_score_only_selection_can_be_classified_causal() -> None:
    classification = recheck.classify_added_row_selection(
        source_lane_logic="EXPLICIT_AS_OF_SCORE_RULE",
        signal_evidence="R5_TAIL_SCORE:SUPPORT",
        selected_from_coverage_proxy_membership=False,
        selected_from_tail_gap_membership=False,
        final_bad_label_available_in_source=False,
        final_tail_label_available_in_source=False,
        post_outcome_safety_used=False,
        as_of_score_only=True,
    )
    assert classification == "CAUSAL_AS_OF_SIGNAL_SELECTION"


def test_membership_only_candidate_cannot_be_marked_directly_r6_ready() -> None:
    with pytest.raises(RuntimeError, match="MEMBERSHIP_ONLY_CANDIDATE"):
        recheck.validate_adapter_not_direct_r6_ready(
            {
                "r6_directly_compatible_v1": True,
                "adapter_would_require_final_labels_or_hindsight_v1": False,
            }
        )


def test_r6_adapter_cannot_require_final_labels_or_hindsight() -> None:
    with pytest.raises(RuntimeError, match="R6_ADAPTER_CANNOT_REQUIRE_FINAL_LABELS"):
        recheck.validate_adapter_not_direct_r6_ready(
            {
                "r6_directly_compatible_v1": False,
                "adapter_would_require_final_labels_or_hindsight_v1": True,
            }
        )


def test_gain_concentration_in_low_support_groups_is_flaggable() -> None:
    assert recheck.concentration_flag(20 / 45) is True
    assert recheck.concentration_flag(10 / 45) is False


def test_anti_overfit_fails_on_hidden_label_or_hindsight_dependency() -> None:
    with pytest.raises(RuntimeError, match="ANTI_OVERFIT_AUDIT_FAILS"):
        recheck.validate_anti_overfit_no_hidden_oracle(
            {
                "hidden_label_or_hindsight_selection_detected_v1": True,
                "status_v1": "BEST_LANE_BLOCKED_BY_ORACLE_OR_HINDSIGHT_DEPENDENCY",
            }
        )


def test_membership_dependency_cannot_claim_causal_adapter_pass() -> None:
    with pytest.raises(RuntimeError, match="MEMBERSHIP_ONLY_DEPENDENCY"):
        recheck.validate_anti_overfit_no_hidden_oracle(
            {
                "hidden_label_or_hindsight_selection_detected_v1": False,
                "membership_only_dependency_visible_v1": True,
                "status_v1": "BEST_LANE_STABILITY_RECHECK_PASS_CAUSAL_ADAPTER_FEASIBLE",
            }
        )


def test_no_r6_model_training_package_freeze_promo_live() -> None:
    clean = recheck.validate_no_forbidden_actions(
        optuna=False,
        r6=False,
        training=False,
        package_build=False,
        adapter_build=False,
        freeze=False,
        promo=False,
        live=False,
    )
    blocked = recheck.validate_no_forbidden_actions(
        optuna=True,
        r6=True,
        training=True,
        package_build=True,
        adapter_build=True,
        freeze=True,
        promo=True,
        live=True,
    )
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "MODEL_TRAINING_FORBIDDEN" in blocked["failures_v1"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    assert recheck.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        recheck.validate_explicit_artifact_selection("LATEST")
