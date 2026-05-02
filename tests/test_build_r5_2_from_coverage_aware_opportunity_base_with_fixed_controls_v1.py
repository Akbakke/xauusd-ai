from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_build_r5_2_from_coverage_aware_opportunity_base_with_fixed_controls_v1 as r5


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_uid_v1": "c1",
        "trade_uid_v1": "t1",
        "trade_id_v1": "trade-1",
        "decision_timestamp_v1": "2026-04-20T00:00:00Z",
        "run_id_v1": "run-a",
        "active_quarantine_v1": "ACTIVE_CANDIDATE",
        "opportunity_role_v1": "COVERAGE_EXPANSION_STRONG_BAD",
        "training_weight_tier_v1": "COVERAGE_MEDIUM_WEIGHT",
        "evaluation_role_v1": "TRAINING_OPPORTUNITY_ONLY",
        "run_id_policy_class_v1": "SUPPORT_SUFFICIENT",
        "structural_low_support_v1": False,
        "zero_denominator_group_v1": False,
        "training_opportunity_allowed_v1": True,
        "final_promotion_evidence_allowed_v1": True,
        "bad_label_v1": True,
        "tail_label_v1": False,
        "safe_recoverable_v1": True,
        "v2_oof_captured_v1": False,
        "historical_v2_captured_v1": False,
        "optuna_captured_v1": False,
        "v3_captured_v1": False,
        "r5_bad_score_signal_bucket_v1": "STRONG",
        "r5_1_bad_score_signal_bucket_v1": "NONE",
        "r5_tail_score_signal_bucket_v1": "NONE",
        "v2_like_bad_tail_signal_bucket_v1": "NONE",
        "v3_oof_signal_bucket_v1": "NONE",
        "protected_winner_status_v1": False,
        "runner_protect_status_v1": False,
        "ambiguous_high_mfe_status_v1": False,
        "fifty_plus_mfe_risk_v1": False,
        "hundred_plus_mfe_risk_v1": False,
        "two_hundred_plus_mfe_risk_v1": False,
        "existing_legal_signal_evidence_count_v1": 1,
        "source_evidence_v1": "R5_BAD_SCORE:STRONG",
        "coverage_reason_v1": "test",
    }
    row.update(overrides)
    return row


def _target_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["target_class_v1"] = frame.apply(r5.classify_target_class, axis=1)
    for key in ["bad_target_v1", "tail_target_v1", "hard_negative_v1", "monitor_only_v1", "exclude_v1"]:
        frame[key] = frame["target_class_v1"].map(lambda target_class: r5._target_class_flags(str(target_class))[key])
    return frame


def test_r5_2_training_target_table_cannot_include_quarantine_positives() -> None:
    target = _target_frame([_row(active_quarantine_v1="QUARANTINE", opportunity_role_v1="QUARANTINE_EXCLUDE")])

    assert target.loc[0, "target_class_v1"] == "EXCLUDE_QUARANTINE"
    assert bool(target.loc[0, "bad_target_v1"]) is False
    assert r5.validate_training_target_table(target)["status_v1"] == "PASS"


def test_protected_winners_are_hard_negatives_veto() -> None:
    row = _row(opportunity_role_v1="HARD_NEGATIVE_PROTECTED_WINNER", protected_winner_status_v1=True)

    assert r5.classify_target_class(row) == "HARD_NEGATIVE_PROTECTED_WINNER"
    assert r5.row_can_be_training_positive(row) is False


def test_runner_protect_rows_are_hard_negatives_veto() -> None:
    row = _row(opportunity_role_v1="HARD_NEGATIVE_RUNNER_PROTECT", runner_protect_status_v1=True)

    assert r5.classify_target_class(row) == "HARD_NEGATIVE_RUNNER_PROTECT"
    assert r5.row_can_be_training_positive(row) is False


def test_ambiguous_high_mfe_cannot_be_positive_without_safe_proof() -> None:
    row = _row(opportunity_role_v1="AMBIGUOUS_MONITOR_ONLY", ambiguous_high_mfe_status_v1=True)

    assert r5.classify_target_class(row) == "MONITOR_ONLY_AMBIGUOUS"
    assert r5.row_can_be_training_positive(row) is False


def test_structural_low_support_positives_are_training_only_not_final_promotion() -> None:
    allowed = r5.candidate_final_promotion_allowed(
        structural_low_support_selected=True,
        strict_loso_decision_valid=True,
        explicit_exception_gate=False,
    )

    assert allowed is False


def test_low_support_groups_remain_in_strict_loso_reporting() -> None:
    scores = pd.DataFrame(
        [
            {"run_id_v1": "run-a", "bad_label_v1": True},
            {"run_id_v1": "run-a", "bad_label_v1": True},
            {"run_id_v1": "run-b", "bad_label_v1": True},
            {"run_id_v1": "run-b", "bad_label_v1": False},
            {"run_id_v1": "run-b", "bad_label_v1": True},
        ]
    )
    selected = pd.Series([True, True, True, False, False])

    rows, summary = r5._loso_rows(scores, selected)

    assert any(row["run_id_v1"] == "run-a" and row["selected_denominator_v1"] == 2 for row in rows)
    assert summary["selected_low_support_group_count_v1"] == 2
    assert summary["strict_all_run_id_decision_valid_v1"] is False


def test_r5_2_oof_cannot_mark_row_decision_valid_when_row_was_in_training_membership() -> None:
    scores = pd.DataFrame(
        [
            {"candidate_uid_v1": "c1", "was_row_in_train_for_scoring_model_v1": True},
            {"candidate_uid_v1": "c2", "was_row_in_train_for_scoring_model_v1": False},
        ]
    )

    result = r5.validate_no_in_sample_scoring(scores)

    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False
    assert result["in_sample_scored_count_v1"] == 1


def test_train_validation_overlap_blocks_decision_valid() -> None:
    membership = pd.DataFrame([{"candidate_uid_v1": "c1", "is_train_v1": True, "is_validation_v1": True}])

    result = r5.validate_no_train_validation_overlap(membership)

    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False


def test_oof_provenance_required_for_every_scored_row() -> None:
    scores = pd.DataFrame([{"candidate_uid_v1": "c1"}, {"candidate_uid_v1": "c2"}])
    provenance = pd.DataFrame(
        [
            {"candidate_uid_v1": "c1", "scorefield_v1": "r5_2_coverage_bad_score_v1", "provenance_valid_v1": True},
            {"candidate_uid_v1": "c1", "scorefield_v1": "r5_2_coverage_tail_score_v1", "provenance_valid_v1": True},
            {"candidate_uid_v1": "c1", "scorefield_v1": "r5_2_coverage_hard_veto_score_v1", "provenance_valid_v1": True},
        ]
    )

    result = r5.validate_oof_provenance_complete(scores, provenance)

    assert result["status_v1"] == "FAIL"
    assert result["missing_provenance_rows_v1"] == 3


def test_no_dummy_synthetic_fallback() -> None:
    result = r5.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=True, fallback=False)

    assert result["status_v1"] == "FAIL"
    assert set(result["failures_v1"]) == {"DUMMY_INPUT_FORBIDDEN", "SYNTHETIC_INPUT_FORBIDDEN"}


def test_no_forbidden_id_leakage_features() -> None:
    result = r5.validate_no_forbidden_features(["as_of_signal_v1", "candidate_uid"])

    assert result["status_v1"] == "FAIL"


def test_no_hindsight_leakage_features() -> None:
    result = r5.validate_no_hindsight_features(["as_of_signal_v1", "future_return_v1"])

    assert result["status_v1"] == "FAIL"


def test_no_implicit_latest_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN"):
        r5.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_v2_oof_scores_provenance_model_thresholds_unchanged_contract() -> None:
    assert r5.historical_v2_role() == "BLUEPRINT_COMPARATOR_ONLY_NOT_DECISION_VALID"


def test_threshold_candidate_cannot_pass_with_safety_violation() -> None:
    row = {
        "fifty_plus_mfe_overlap_v1": 0,
        "hundred_plus_mfe_overlap_v1": 1,
        "two_hundred_plus_overlap_v1": 0,
        "two_hundred_plus_mfe_overlap_v1": 0,
        "strongest_winner_overlap_v1": 0,
        "protected_winner_selected_v1": 0,
        "runner_protect_leakage_v1": 0,
        "ambiguous_high_mfe_leakage_v1": 0,
        "quarantine_selected_v1": 0,
    }

    assert r5.threshold_candidate_passes_safety(row) is False


def test_candidate_cannot_claim_final_promotion_if_structural_low_support_remains() -> None:
    assert (
        r5.candidate_final_promotion_allowed(
            structural_low_support_selected=True,
            strict_loso_decision_valid=False,
            explicit_exception_gate=False,
        )
        is False
    )


def test_historical_v2_remains_comparator_only() -> None:
    assert r5.historical_v2_role() == "BLUEPRINT_COMPARATOR_ONLY_NOT_DECISION_VALID"


def test_optuna_and_v3_cannot_become_baseline() -> None:
    assert r5.weak_control_can_be_baseline("optuna") is False
    assert r5.weak_control_can_be_baseline("v3") is False


def test_candidate_comparison_includes_v2_oof_69_53() -> None:
    comparison = r5._fixed_control_comparison({"bad_count_v1": 80, "tail_count_v1": 60})
    v2 = next(row for row in comparison if row["control_v1"] == "v2_oof")

    assert v2["bad_v1"] == 69
    assert v2["tail_v1"] == 53


def test_no_r6_package_freeze_live_in_this_task() -> None:
    result = r5.validate_no_forbidden_actions(optuna=False, r6=False, package=False, freeze=False, live=False)

    assert result["status_v1"] == "PASS"
