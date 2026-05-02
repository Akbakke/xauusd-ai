from __future__ import annotations

import pytest

from gx1.scripts import materialize_find_back_to_wednesday_r6_skeleton_and_rebuild_monday_foundation_v1 as skeleton


def test_search_without_v2_fixed_control_is_coverage_failure() -> None:
    result = skeleton.assess_search_space_coverage(
        has_v2_fixed_control=False,
        can_evaluate_current_best_baseline=True,
        can_reproduce_current_best_baseline=True,
    )

    assert result["status_v1"] == "SEARCH_SPACE_COVERAGE_FAILURE"
    assert result["model_limit_claim_allowed_v1"] is False


def test_search_that_cannot_reproduce_baseline_cannot_claim_model_failure() -> None:
    result = skeleton.assess_search_space_coverage(
        has_v2_fixed_control=True,
        can_evaluate_current_best_baseline=True,
        can_reproduce_current_best_baseline=False,
    )

    assert result["status_v1"] == "SEARCH_SPACE_COVERAGE_FAILURE"
    assert result["model_limit_claim_allowed_v1"] is False


def test_old_invalid_v3_artifact_cannot_be_selected() -> None:
    status = skeleton.selected_v3_artifact_status(
        selected_for_decisioning=True,
        decision_valid_status="INVALID_FOR_OPTUNA_DECISIONING",
    )

    assert status == "BLOCK_SELECTED_INVALID_V3"


def test_historical_invalid_v3_artifact_does_not_block_when_not_selected() -> None:
    status = skeleton.selected_v3_artifact_status(
        selected_for_decisioning=False,
        decision_valid_status="INVALID_FOR_OPTUNA_DECISIONING",
    )

    assert status == "HISTORY_ONLY_NOT_BLOCKER"


def test_wednesday_contract_missing_artifact_does_not_invent_hash() -> None:
    gap = skeleton.missing_wednesday_artifact_gap("models", "/missing/models")

    assert gap["status_v1"] == "MISSING_LOCAL_ARTIFACT"
    assert gap["hash_v1"] == "MISSING_LOCAL_ARTIFACT"
    assert gap["invented_v1"] is False


def test_wednesday_threshold_diagnostic_control_is_representable_as_config() -> None:
    control = skeleton.wednesday_threshold_diagnostic_control()

    assert control["exact_model_required_v1"] is False
    assert control["thresholds_v1"]["bad_threshold_v1"] == 0.95
    assert control["thresholds_v1"]["guard_v1"] == "hard_asof_runner_guard"


@pytest.mark.parametrize(
    ("metric_denominator_valid", "safety_clean", "provenance_valid", "artifacts_missing", "expected"),
    [
        (True, True, True, False, "V2_DECISION_VALID_UNDER_CURRENT_GUARDS"),
        (True, True, False, False, "V2_HISTORICAL_ONLY_NOT_PROVENANCE_VALID"),
        (False, True, True, False, "V2_COLLAPSES_UNDER_CURRENT_GUARDS"),
        (True, False, True, False, "V2_COLLAPSES_UNDER_CURRENT_GUARDS"),
        (True, True, True, True, "V2_REQUIRES_MISSING_ARTIFACT"),
    ],
)
def test_v2_baseline_reconciliation_status_enum(
    metric_denominator_valid: bool,
    safety_clean: bool,
    provenance_valid: bool,
    artifacts_missing: bool,
    expected: str,
) -> None:
    assert (
        skeleton.classify_v2_reconciliation(
            metric_denominator_valid=metric_denominator_valid,
            safety_clean=safety_clean,
            provenance_valid=provenance_valid,
            artifacts_missing=artifacts_missing,
        )
        == expected
    )


def test_optuna_56_55_safe_but_not_better_cannot_become_new_baseline() -> None:
    assert skeleton.optuna_result_can_be_new_baseline("SAFE_BUT_NOT_BETTER_THAN_V2", 56, 55) is False


def test_metric_denominator_invalidity_blocks_decision_valid() -> None:
    metric = skeleton._metric_ratio("precision", numerator=0, denominator=0)

    assert metric["precision_denominator_status_v1"] == "EMPTY_DENOMINATOR"
    assert metric["precision_decision_valid_v1"] is False


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN"):
        skeleton.validate_explicit_selection_policy("LATEST_FOLDER_WINS")


def test_no_dummy_synthetic_or_degraded_fallback() -> None:
    attestation = skeleton.decision_input_attestation(
        dummy_input_used=True,
        synthetic_input_used=True,
        degraded_fallback_used=True,
        in_sample_decisioning_used=False,
    )

    assert attestation["decision_valid_v1"] is False
    assert attestation["status_v1"] == "BLOCKED"
    assert set(attestation["failures_v1"]) == {
        "DUMMY_INPUT_FORBIDDEN",
        "SYNTHETIC_INPUT_FORBIDDEN",
        "DEGRADED_FALLBACK_FORBIDDEN",
    }


def test_in_sample_score_cannot_be_decision_valid() -> None:
    attestation = skeleton.decision_input_attestation(
        dummy_input_used=False,
        synthetic_input_used=False,
        degraded_fallback_used=False,
        in_sample_decisioning_used=True,
    )

    assert attestation["decision_valid_v1"] is False
    assert attestation["failures_v1"] == ["IN_SAMPLE_DECISIONING_FORBIDDEN"]
