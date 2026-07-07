from __future__ import annotations

import pytest

from gx1.scripts import materialize_revalidate_v2_baseline_under_current_guards_v1 as revalidate


def test_v2_cannot_be_decision_valid_without_oof_provenance() -> None:
    result = revalidate.classify_v2_decision_validity(
        precision_valid=True,
        worst_loso_valid=True,
        oof_provenance_valid=False,
        no_in_sample_decisioning=True,
        safety_clean=True,
        row_level_selection_exists=True,
    )

    assert result["decision_valid_v1"] is False
    assert result["status_v1"] == "V2_COLLAPSES_DUE_TO_MISSING_PROVENANCE"
    assert "MISSING_OOF_PROVENANCE" in result["invalid_reasons_v1"]


def test_v2_cannot_be_decision_valid_with_invalid_worst_loso_denominator() -> None:
    result = revalidate.classify_v2_decision_validity(
        precision_valid=True,
        worst_loso_valid=False,
        oof_provenance_valid=True,
        no_in_sample_decisioning=True,
        safety_clean=True,
        row_level_selection_exists=True,
    )

    assert result["decision_valid_v1"] is False
    assert "WORST_LOSO_DENOMINATOR_INVALID" in result["invalid_reasons_v1"]


def test_precision_denominator_can_pass_while_worst_loso_denominator_fails() -> None:
    precision = revalidate.metric_ratio("precision", numerator=95, denominator=95)
    decision = revalidate.classify_v2_decision_validity(
        precision_valid=precision["precision_decision_valid_v1"],
        worst_loso_valid=False,
        oof_provenance_valid=True,
        no_in_sample_decisioning=True,
        safety_clean=True,
        row_level_selection_exists=True,
    )

    assert precision["precision_decision_valid_v1"] is True
    assert decision["decision_valid_v1"] is False


def test_v2_fixed_control_can_exist_without_bypassing_guards() -> None:
    contract = revalidate._fixed_control_contract(
        {
            "worst_loso_denominator_v1": 2,
            "oof_provenance_status_v1": "MISSING_OOF_PROVENANCE_FILES",
            "selected_training_overlap_v1": 94,
        }
    )

    assert contract["bypass_guards_allowed_v1"] is False
    assert contract["v2_role_v1"].startswith("CURRENT_BEST_HISTORICAL_SAFE_MONDAY_COMPARATOR")


def test_future_search_without_v2_fixed_control_is_coverage_failure() -> None:
    result = revalidate.assess_search_space_coverage(
        has_v2_fixed_control=False,
        can_evaluate_v2=True,
        can_reproduce_v2=True,
    )

    assert result["status_v1"] == "SEARCH_SPACE_COVERAGE_FAILURE"
    assert result["model_limit_claim_allowed_v1"] is False


def test_v2_row_level_reconstruction_cannot_be_faked_if_missing() -> None:
    assert revalidate.reconstruction_status(False) == "V2_ROW_LEVEL_SELECTION_MISSING_LOCAL"


@pytest.mark.parametrize(
    ("source", "config", "model", "rows", "writes_oof", "avoids_in_sample", "expected"),
    [
        (False, True, True, True, True, True, "V2_REPLAY_NOT_POSSIBLE_LOCAL"),
        (True, False, True, True, True, True, "V2_REPLAY_REQUIRES_MISSING_CONFIG"),
        (True, True, False, True, True, True, "V2_REPLAY_REQUIRES_MISSING_MODEL_ARTIFACT"),
        (True, True, True, True, False, False, "V2_REPLAY_REQUIRES_SOURCE_PATCH_ONLY"),
        (True, True, True, True, True, False, "V2_REPLAY_UNSAFE_OR_IN_SAMPLE_ONLY"),
    ],
)
def test_v2_replay_feasibility_distinguishes_missing_inputs_and_patch_only(
    source: bool,
    config: bool,
    model: bool,
    rows: bool,
    writes_oof: bool,
    avoids_in_sample: bool,
    expected: str,
) -> None:
    assert (
        revalidate.classify_replay_feasibility(
            source_logic_exists=source,
            config_exists=config,
            model_artifacts_exist=model,
            row_level_outputs_exist=rows,
            current_runner_writes_oof_provenance=writes_oof,
            current_runner_avoids_in_sample_decisioning=avoids_in_sample,
        )
        == expected
    )


def test_existing_legal_learning_foundation_cannot_use_dummy_or_synthetic_labels() -> None:
    result = revalidate.validate_learning_labels(dummy_label_used=True, synthetic_label_used=True)

    assert result["valid_v1"] is False
    assert set(result["failures_v1"]) == {"DUMMY_LABEL_FORBIDDEN", "SYNTHETIC_LABEL_FORBIDDEN"}


def test_ambiguous_high_mfe_rows_are_monitor_only_not_rewarded() -> None:
    use = revalidate.recommended_learning_use(
        safe_recoverable=True,
        bad=True,
        tail=False,
        quarantine=False,
        protected_winner=False,
        runner_protect=False,
        ambiguous_high_mfe=True,
    )

    assert use == "AMBIGUOUS_MONITOR_ONLY"


def test_protected_winners_become_hard_negative_or_veto_rows() -> None:
    use = revalidate.recommended_learning_use(
        safe_recoverable=True,
        bad=True,
        tail=True,
        quarantine=False,
        protected_winner=True,
        runner_protect=False,
        ambiguous_high_mfe=False,
    )

    assert use == "HARD_NEGATIVE_PROTECTED_WINNER"


def test_safe_recoverable_rows_are_not_all_positive_without_class_reason() -> None:
    use = revalidate.recommended_learning_use(
        safe_recoverable=True,
        bad=False,
        tail=False,
        quarantine=False,
        protected_winner=False,
        runner_protect=False,
        ambiguous_high_mfe=False,
    )

    assert use == "UNKNOWN_REQUIRES_ARTIFACT"


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN"):
        revalidate.validate_explicit_selection_policy("LATEST_GLOB")


def test_old_invalid_v3_artifacts_cannot_be_selected() -> None:
    status = revalidate.selected_v3_artifact_status(
        selected_for_decisioning=True,
        decision_valid_status="INVALID_FOR_OPTUNA_DECISIONING",
    )

    assert status == "BLOCK_SELECTED_INVALID_V3"


def test_optuna_56_55_cannot_replace_v2_baseline() -> None:
    assert revalidate.optuna_result_can_replace_v2("SAFE_BUT_NOT_BETTER_THAN_V2", 56, 55) is False
