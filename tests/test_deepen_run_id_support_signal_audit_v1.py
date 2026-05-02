from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_deepen_run_id_support_signal_audit_v1 as audit


def _candidate(**overrides: object) -> pd.Series:
    row = {
        "active_quarantine_v1": "ACTIVE_CANDIDATE",
        "safe_recoverable_v1": True,
        "protected_winner_status_v1": False,
        "runner_protect_status_v1": False,
        "ambiguous_high_mfe_status_v1": False,
        "fifty_plus_mfe_risk_v1": False,
        "hundred_plus_mfe_risk_v1": False,
        "two_hundred_plus_mfe_risk_v1": False,
        "existing_legal_signal_evidence_count_v1": 1,
    }
    row.update(overrides)
    return pd.Series(row)


def _toy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_uid_v1": "c1",
                "trade_uid_v1": "t1",
                "run_id_v1": audit.WORST_RUN_ID,
                "active_quarantine_v1": "ACTIVE_CANDIDATE",
                "bad_label_v1": True,
                "tail_label_v1": True,
                "safe_recoverable_v1": True,
                "v2_oof_captured_v1": True,
                "historical_v2_captured_v1": True,
                "optuna_captured_v1": False,
                "v3_captured_v1": False,
                "r5_bad_score_signal_bucket_v1": "STRONG",
                "r5_1_bad_score_signal_bucket_v1": "SUPPORT",
                "r5_tail_score_signal_bucket_v1": "STRONG",
                "v2_like_bad_tail_signal_bucket_v1": "STRONG",
                "protected_winner_status_v1": False,
                "runner_protect_status_v1": False,
                "ambiguous_high_mfe_status_v1": False,
                "fifty_plus_mfe_risk_v1": False,
                "hundred_plus_mfe_risk_v1": False,
                "two_hundred_plus_mfe_risk_v1": False,
                "existing_legal_signal_evidence_count_v1": 4,
                "recommended_opportunity_role_v1": "CORE_OOF_V2_TAIL_POSITIVE",
                audit.V2_CORE_COL: True,
                audit.RECOMMENDED_COL: True,
                audit.BALANCED_COL: True,
                audit.UPPER_BOUND_COL: True,
            },
            {
                "candidate_uid_v1": "c2",
                "trade_uid_v1": "t2",
                "run_id_v1": audit.WORST_RUN_ID,
                "active_quarantine_v1": "ACTIVE_CANDIDATE",
                "bad_label_v1": True,
                "tail_label_v1": True,
                "safe_recoverable_v1": True,
                "v2_oof_captured_v1": True,
                "historical_v2_captured_v1": True,
                "optuna_captured_v1": False,
                "v3_captured_v1": False,
                "r5_bad_score_signal_bucket_v1": "STRONG",
                "r5_1_bad_score_signal_bucket_v1": "SUPPORT",
                "r5_tail_score_signal_bucket_v1": "STRONG",
                "v2_like_bad_tail_signal_bucket_v1": "STRONG",
                "protected_winner_status_v1": False,
                "runner_protect_status_v1": False,
                "ambiguous_high_mfe_status_v1": False,
                "fifty_plus_mfe_risk_v1": False,
                "hundred_plus_mfe_risk_v1": False,
                "two_hundred_plus_mfe_risk_v1": False,
                "existing_legal_signal_evidence_count_v1": 4,
                "recommended_opportunity_role_v1": "CORE_OOF_V2_TAIL_POSITIVE",
                audit.V2_CORE_COL: True,
                audit.RECOMMENDED_COL: True,
                audit.BALANCED_COL: True,
                audit.UPPER_BOUND_COL: True,
            },
        ]
    )


def test_feasible_safe_max_below_target_is_structural_not_model_failure() -> None:
    status = audit.classify_support_repairability(
        current_denominator=2,
        feasible_safe_max=2,
        denominator_target=5,
        additional_safe_candidates=0,
        tail_candidates=0,
        risky_signal_candidates=0,
        protected_winners=0,
        runner_protect=0,
        ambiguous_high_mfe=0,
        quarantine=0,
        missing_artifacts=0,
    )

    assert status == "STRUCTURALLY_UNSATISFIABLE_FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR"
    assert status != "TRUE_MODEL_UNDER_SELECTION"


def test_low_support_groups_cannot_be_silently_dropped() -> None:
    with pytest.raises(RuntimeError, match="LOW_SUPPORT_GROUPS_CANNOT_BE_SILENTLY_DROPPED"):
        audit.validate_low_support_groups_not_silently_dropped(dropped=True, explicitly_reported=False)


def test_denominator_guard_cannot_be_weakened() -> None:
    with pytest.raises(RuntimeError, match="DENOMINATOR_GUARD_WEAKENING_FORBIDDEN"):
        audit.validate_denominator_guard_not_weakened(2)


def test_protected_winners_cannot_repair_support() -> None:
    with pytest.raises(RuntimeError, match="PROTECTED_WINNER_CANNOT_REPAIR_SUPPORT"):
        audit.validate_added_support_candidate(_candidate(protected_winner_status_v1=True))


def test_runner_protect_rows_cannot_repair_support() -> None:
    with pytest.raises(RuntimeError, match="RUNNER_PROTECT_CANNOT_REPAIR_SUPPORT"):
        audit.validate_added_support_candidate(_candidate(runner_protect_status_v1=True))


def test_ambiguous_high_mfe_rows_cannot_repair_support_without_safe_proof() -> None:
    with pytest.raises(RuntimeError, match="AMBIGUOUS_HIGH_MFE_CANNOT_REPAIR_SUPPORT_WITHOUT_SAFE_PROOF"):
        audit.validate_added_support_candidate(_candidate(ambiguous_high_mfe_status_v1=True))


def test_quarantine_rows_cannot_repair_support() -> None:
    with pytest.raises(RuntimeError, match="QUARANTINE_CANNOT_REPAIR_SUPPORT"):
        audit.validate_added_support_candidate(_candidate(active_quarantine_v1="QUARANTINE"))


def test_balanced_broad_expansion_is_flagged_when_it_creates_low_support_groups() -> None:
    summary = audit._balanced_summary(
        [
            {
                "balanced_denominator_v1": 1,
                "v2_core_denominator_v1": 0,
                "balanced_worsened_low_support_v1": True,
                "created_new_low_support_selected_group_v1": True,
                "added_to_already_supported_group_v1": False,
            }
        ]
    )

    assert summary["groups_worsened_v1"] == 1
    assert summary["new_low_support_groups_created_v1"] == 1


def test_max_feasible_under_hard_vetoes_cannot_include_unsafe_rows() -> None:
    rows = pd.DataFrame({"active_quarantine_v1": ["ACTIVE_CANDIDATE"], "protected_winner_status_v1": [True]})
    with pytest.raises(RuntimeError, match="MAX_FEASIBLE_UNDER_HARD_VETOES_CANNOT_INCLUDE_UNSAFE_ROWS"):
        audit.validate_frontier_has_no_unsafe_rows(rows, pd.Series([True]))


def test_every_added_support_candidate_requires_signal_evidence() -> None:
    with pytest.raises(RuntimeError, match="SUPPORT_CANDIDATE_MUST_HAVE_SIGNAL_EVIDENCE"):
        audit.validate_added_support_candidate(_candidate(existing_legal_signal_evidence_count_v1=0))


def test_worst_run_id_must_appear_in_feasibility_matrix() -> None:
    matrix = audit._run_id_feasibility_matrix(_toy_rows())

    assert any(row["run_id_v1"] == audit.WORST_RUN_ID for row in matrix)
    worst = next(row for row in matrix if row["run_id_v1"] == audit.WORST_RUN_ID)
    assert worst["feasible_max_denominator_v1"] == 2


def test_v2_oof_scores_and_provenance_remain_unchanged() -> None:
    before = {"v2_oof_scores_sha256_v1": "score", "v2_oof_provenance_sha256_v1": "prov"}
    after = {"v2_oof_scores_sha256_v1": "score", "v2_oof_provenance_sha256_v1": "prov"}

    assert audit.validate_input_artifacts_unchanged(before, after)["status_v1"] == "PASS"


def test_no_optuna_model_r6_package_freeze_or_live() -> None:
    result = audit.validate_no_forbidden_actions(optuna=False, model=False, r6=False, package=False, freeze=False, live=False)
    assert result["status_v1"] == "PASS"
    failed = audit.validate_no_forbidden_actions(optuna=True, model=True, r6=False, package=False, freeze=False, live=False)
    assert failed["status_v1"] == "FAIL"
    assert failed["failures_v1"] == ["OPTUNA_FORBIDDEN", "MODEL_TRAINING_FORBIDDEN"]


def test_no_dummy_synthetic_or_fallback() -> None:
    result = audit.validate_no_dummy_synthetic_fallback(dummy=False, synthetic=True, fallback=True)
    assert result["status_v1"] == "FAIL"
    assert result["failures_v1"] == ["SYNTHETIC_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN"):
        audit.validate_explicit_artifact_selection("LATEST_GLOB")
