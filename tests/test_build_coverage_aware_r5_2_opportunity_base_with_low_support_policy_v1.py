from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_build_coverage_aware_r5_2_opportunity_base_with_low_support_policy_v1 as cov


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_uid_v1": "c",
        "trade_uid_v1": "t",
        "trade_id_v1": "trade",
        "decision_timestamp_v1": "2025-01-01T00:00:00Z",
        "run_id_v1": "SUPPORTED_RUN",
        "active_quarantine_v1": "ACTIVE_CANDIDATE",
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
        "provenance_status_v1": "LOCAL_OPPORTUNITY_BASE",
        "existing_legal_signal_evidence_count_v1": 1,
        "member_v2_oof_core_only_v1": False,
        "member_v2_oof_plus_run_id_support_v1": False,
    }
    row.update(overrides)
    return row


def _registry_row(run_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id_v1": run_id,
        "run_id_policy_class_v1": "SUPPORT_SUFFICIENT",
        "structural_low_support_v1": False,
        "selected_low_support_v1": False,
        "zero_denominator_group_v1": False,
        "can_be_used_in_training_surface_v1": False,
        "can_be_used_in_decision_valid_eval_v1": True,
        "feasible_safe_max_denominator_v1": 5,
    }
    row.update(overrides)
    return row


def _toy_rows_and_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for idx in range(2):
        rows.append(
            _row(
                candidate_uid_v1=f"w{idx}",
                run_id_v1=cov.WORST_RUN_ID,
                tail_label_v1=True,
                v2_oof_captured_v1=True,
                historical_v2_captured_v1=True,
                r5_tail_score_signal_bucket_v1="STRONG",
                existing_legal_signal_evidence_count_v1=4,
                member_v2_oof_core_only_v1=True,
                member_v2_oof_plus_run_id_support_v1=True,
            )
        )
    for idx in range(5):
        rows.append(
            _row(
                candidate_uid_v1=f"s{idx}",
                run_id_v1="SUPPORTED_RUN",
                v2_oof_captured_v1=True,
                historical_v2_captured_v1=True,
                member_v2_oof_core_only_v1=True,
                member_v2_oof_plus_run_id_support_v1=True,
            )
        )
    rows.append(_row(candidate_uid_v1="tail", run_id_v1="REPAIR_RUN", tail_label_v1=True, r5_tail_score_signal_bucket_v1="STRONG"))
    rows.append(_row(candidate_uid_v1="bad", run_id_v1="REPAIR_RUN", r5_bad_score_signal_bucket_v1="STRONG"))
    registry = pd.DataFrame(
        [
            _registry_row(
                cov.WORST_RUN_ID,
                run_id_policy_class_v1="STRUCTURAL_LOW_SUPPORT_FEASIBLE_MAX_BELOW_TARGET",
                structural_low_support_v1=True,
                selected_low_support_v1=True,
                can_be_used_in_training_surface_v1=True,
                can_be_used_in_decision_valid_eval_v1=False,
                feasible_safe_max_denominator_v1=2,
            ),
            _registry_row("SUPPORTED_RUN"),
            _registry_row(
                "REPAIR_RUN",
                run_id_policy_class_v1="SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS",
                selected_low_support_v1=True,
                can_be_used_in_decision_valid_eval_v1=False,
                feasible_safe_max_denominator_v1=4,
            ),
        ]
    )
    return pd.DataFrame(rows), registry


def test_coverage_base_cannot_include_quarantine_positives() -> None:
    row = _row(active_quarantine_v1="QUARANTINE", existing_legal_signal_evidence_count_v1=10)

    assert cov.classify_opportunity_role(row) == "QUARANTINE_EXCLUDE"
    assert cov.row_can_be_positive(row) is False


def test_protected_winners_are_hard_negative_veto() -> None:
    row = _row(protected_winner_status_v1=True)

    assert cov.classify_opportunity_role(row) == "HARD_NEGATIVE_PROTECTED_WINNER"
    assert cov.row_can_be_positive(row) is False


def test_runner_protect_rows_are_hard_negative_veto() -> None:
    row = _row(runner_protect_status_v1=True)

    assert cov.classify_opportunity_role(row) == "HARD_NEGATIVE_RUNNER_PROTECT"
    assert cov.row_can_be_positive(row) is False


def test_ambiguous_high_mfe_cannot_be_positive_without_safe_proof() -> None:
    row = _row(ambiguous_high_mfe_status_v1=True)

    assert cov.classify_opportunity_role(row) == "AMBIGUOUS_MONITOR_ONLY"
    assert cov.row_can_be_positive(row) is False


def test_structural_low_support_rows_are_training_only_not_final_evidence() -> None:
    rows, registry = _toy_rows_and_registry()
    coverage = cov._coverage_rows(rows, registry)
    worst = coverage[coverage["run_id_v1"].eq(cov.WORST_RUN_ID)].iloc[0]

    assert bool(worst["training_opportunity_allowed_v1"]) is True
    assert bool(worst["final_promotion_evidence_allowed_v1"]) is False
    assert worst["evaluation_role_v1"] == "TRAINING_ONLY_LOW_SUPPORT"


def test_low_support_groups_remain_in_strict_loso_reporting() -> None:
    rows, registry = _toy_rows_and_registry()
    coverage = cov._coverage_rows(rows, registry)
    memberships = cov._memberships(coverage)
    variants = cov._variant_summary(coverage, memberships)
    core = next(row for row in variants if row["variant_id_v1"] == "V2_OOF_CORE_69")

    assert core["strict_all_run_id_min_denominator_v1"] == 2
    assert core["selected_low_support_groups_v1"] >= 1


def test_coverage_aware_variant_cannot_claim_final_decision_valid_with_structural_low_support() -> None:
    rows, registry = _toy_rows_and_registry()
    coverage = cov._coverage_rows(rows, registry)
    variants = cov._variant_summary(coverage, cov._memberships(coverage))
    run_balanced = next(row for row in variants if row["variant_id_v1"] == "COVERAGE_AWARE_RUN_ID_BALANCED")

    assert run_balanced["strict_all_run_id_decision_valid_v1"] is False
    assert run_balanced["final_promotion_allowed_v1"] is False


def test_max_policy_allowed_diagnostic_cannot_be_final_recommendation() -> None:
    assert cov.max_policy_allowed_can_be_final_recommendation("MAX_POLICY_ALLOWED_DIAGNOSTIC") is False


def test_every_positive_row_requires_signal_evidence_and_reason() -> None:
    row = _row(existing_legal_signal_evidence_count_v1=1, r5_bad_score_signal_bucket_v1="STRONG")

    assert cov.positive_row_has_evidence(row) is True
    assert cov.row_can_be_positive(row) is True
    assert cov.classify_opportunity_role(row) == "COVERAGE_EXPANSION_STRONG_BAD"


def test_safe_recoverable_alone_is_not_positive() -> None:
    row = _row(
        bad_label_v1=True,
        safe_recoverable_v1=True,
        r5_bad_score_signal_bucket_v1="NONE",
        r5_1_bad_score_signal_bucket_v1="NONE",
        r5_tail_score_signal_bucket_v1="NONE",
        v2_like_bad_tail_signal_bucket_v1="NONE",
        existing_legal_signal_evidence_count_v1=0,
    )

    assert cov.positive_row_has_evidence(row) is False
    assert cov.row_can_be_positive(row) is False


def test_training_weight_tiers_do_not_imply_model_training_occurred() -> None:
    rows, registry = _toy_rows_and_registry()
    coverage = cov._coverage_rows(rows, registry)
    variants = cov._variant_summary(coverage, cov._memberships(coverage))

    assert set(coverage["training_weight_tier_v1"]) <= {
        "CORE_HIGH_WEIGHT",
        "TAIL_HIGH_WEIGHT",
        "COVERAGE_MEDIUM_WEIGHT",
        "LOW_SUPPORT_LOW_WEIGHT",
        "HARD_NEGATIVE_HIGH_WEIGHT",
        "MONITOR_ZERO_WEIGHT",
        "EXCLUDE_ZERO_WEIGHT",
        "UNKNOWN_ZERO_WEIGHT",
    }
    assert all(row["model_trained_v1"] is False for row in variants)


def test_hard_negative_veto_rows_cannot_be_selected_as_positive_memberships() -> None:
    rows, registry = _toy_rows_and_registry()
    rows = pd.concat([rows, pd.DataFrame([_row(candidate_uid_v1="protected", protected_winner_status_v1=True)])], ignore_index=True)
    coverage = cov._coverage_rows(rows, registry)
    memberships = cov._memberships(coverage)
    protected_idx = coverage[coverage["candidate_uid_v1"].eq("protected")].index[0]

    assert not any(bool(membership.loc[protected_idx]) for membership in memberships.values())


def test_v2_oof_scores_provenance_model_and_thresholds_unchanged_validation() -> None:
    before = {
        "v2_oof_scores_sha256_v1": "score",
        "v2_oof_provenance_sha256_v1": "prov",
        "opportunity_rows_sha256_v1": "rows",
        "low_support_registry_sha256_v1": "reg",
    }

    assert cov.validate_input_artifacts_unchanged(before, dict(before))["status_v1"] == "PASS"


def test_no_optuna_model_r6_package_freeze_or_live() -> None:
    blocked = cov.validate_no_forbidden_actions(optuna=True, model=True, r6=True, package=True, freeze=True, live=True)

    assert blocked["status_v1"] == "FAIL"
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]
    assert "MODEL_TRAINING_FORBIDDEN" in blocked["failures_v1"]


def test_no_dummy_synthetic_or_fallback() -> None:
    result = cov.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=True, fallback=True)

    assert result["status_v1"] == "FAIL"
    assert result["failures_v1"] == ["DUMMY_INPUT_FORBIDDEN", "SYNTHETIC_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN"):
        cov.validate_explicit_artifact_selection("LATEST_GLOB")


def test_recommendation_includes_fixed_controls_for_next_rebuild() -> None:
    rows, registry = _toy_rows_and_registry()
    coverage = cov._coverage_rows(rows, registry)
    memberships = cov._memberships(coverage)
    recommendation = cov._recommendation(cov._variant_summary(coverage, memberships), cov._addition_plan(coverage, memberships))

    controls = recommendation["fixed_controls_v1"]["fixed_controls_required_for_next_r5_2_rebuild_v1"]
    assert "V2 OOF 69/53 as provenance-valid signal control" in controls
    assert "strict LOSO all-run_id reporting" in controls


def test_final_promotion_remains_blocked_when_structural_low_support_remains() -> None:
    assert (
        cov.final_promotion_allowed(
            structural_low_support_selected=True,
            strict_loso_decision_valid=True,
            explicit_exception_gate=False,
        )
        is False
    )
