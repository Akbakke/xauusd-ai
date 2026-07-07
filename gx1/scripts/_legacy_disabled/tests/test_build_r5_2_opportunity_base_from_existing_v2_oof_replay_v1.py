from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_build_r5_2_opportunity_base_from_existing_v2_oof_replay_v1 as opp


def _role(**overrides: object) -> str:
    args = {
        "active": True,
        "quarantine": False,
        "protected_winner": False,
        "runner_protect": False,
        "ambiguous_high_mfe": False,
        "high_mfe_unsafe": False,
        "bad_label": True,
        "tail_label": False,
        "safe_recoverable": True,
        "v2_oof_captured": False,
        "historical_v2_captured": False,
        "optuna_captured": False,
        "v3_captured": False,
        "r5_bad_bucket": "NONE",
        "r5_1_bad_bucket": "NONE",
        "r5_tail_bucket": "NONE",
        "run_id_low_support": False,
    }
    args.update(overrides)
    return opp.classify_opportunity_role(**args)


def _variant_row(name: str, *, total: int, worst: int, low: int, safety: int = 0) -> dict[str, object]:
    return {
        "variant_id_v1": name,
        "total_selected_rows_v1": total,
        "worst_run_id_support_denominator_v1": worst,
        "run_id_groups_below_denominator_threshold_v1": low,
        "fifty_plus_overlap_v1": 0,
        "hundred_plus_overlap_v1": 0,
        "two_hundred_plus_overlap_v1": 0,
        "strongest_winner_overlap_v1": safety,
        "runner_protect_leakage_v1": 0,
        "ambiguous_high_mfe_leakage_v1": 0,
        "quarantine_selected_v1": 0,
    }


def test_quarantine_rows_cannot_be_positive() -> None:
    assert _role(active=False, quarantine=True, r5_bad_bucket="STRONG") == "QUARANTINE_EXCLUDE"


def test_protected_winners_are_hard_negatives() -> None:
    assert _role(protected_winner=True, r5_bad_bucket="STRONG") == "HARD_NEGATIVE_PROTECTED_WINNER"


def test_runner_protect_rows_are_hard_negatives() -> None:
    assert _role(runner_protect=True, r5_bad_bucket="STRONG") == "HARD_NEGATIVE_RUNNER_PROTECT"


def test_ambiguous_high_mfe_rows_are_monitor_only() -> None:
    assert _role(ambiguous_high_mfe=True, r5_bad_bucket="STRONG") == "AMBIGUOUS_MONITOR_ONLY"


def test_safe_recoverable_rows_are_not_automatic_positives_without_signal() -> None:
    assert _role(safe_recoverable=True, bad_label=False, tail_label=False) == "UNKNOWN_REQUIRES_ARTIFACT"


def test_v2_oof_scores_and_provenance_are_not_modified() -> None:
    before = {"v2_oof_scores_sha256_v1": "score", "v2_oof_provenance_sha256_v1": "prov"}
    after = {"v2_oof_scores_sha256_v1": "score", "v2_oof_provenance_sha256_v1": "prov"}
    changed = {"v2_oof_scores_sha256_v1": "changed", "v2_oof_provenance_sha256_v1": "prov"}

    assert opp.validate_input_artifacts_unchanged(before, after)["status_v1"] == "PASS"
    failed = opp.validate_input_artifacts_unchanged(before, changed)
    assert failed["status_v1"] == "FAIL"
    assert failed["v2_oof_scores_unchanged_v1"] is False


def test_loso_denominator_guard_is_not_weakened() -> None:
    with pytest.raises(RuntimeError, match="LOSO_DENOMINATOR_GUARD_WEAKENING_FORBIDDEN"):
        opp.validate_loso_guard_not_weakened(2)
    assert opp.validate_loso_guard_not_weakened(opp.MIN_RUN_ID_SUPPORT) is True


def test_candidate_variants_are_membership_sets_not_trained_models() -> None:
    assert opp.validate_variant_is_membership_set({"model_trained_v1": False, "package_built_v1": False, "r6_ready_v1": False}) is True
    with pytest.raises(RuntimeError, match="OPPORTUNITY_VARIANT_MUST_BE_MEMBERSHIP_SET_NOT_MODEL"):
        opp.validate_variant_is_membership_set({"model_trained_v1": True})


def test_historical_v2_remains_historical_only() -> None:
    assert opp.historical_v2_status() == "HISTORICAL_ONLY_NOT_DECISION_VALID"


def test_optuna_and_v3_weak_candidates_cannot_be_baseline() -> None:
    assert opp.optuna_v3_can_be_baseline("SAFE_BUT_NOT_BETTER_THAN_V2") is False
    assert opp.optuna_v3_can_be_baseline("WEAK_CONTROL") is False


def test_no_dummy_synthetic_or_fallback_allowed() -> None:
    result = opp.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=False, fallback=True)
    assert result["status_v1"] == "FAIL"
    assert result["failures_v1"] == ["DUMMY_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN"):
        opp.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_run_id_support_analysis_includes_known_low_support_group() -> None:
    rows = pd.DataFrame(
        {
            "run_id": [opp.WORST_RUN_ID, opp.WORST_RUN_ID, "OTHER_RUN"],
            "active_v1": [True, True, True],
            "safe_recoverable_v1": [True, True, True],
            "bad_label_v1": [True, True, True],
            "tail_label_v1": [True, True, False],
            "protected_winner_v1": [False, False, False],
            "runner_protect_v1": [False, False, False],
            "ambiguous_high_mfe_v1": [False, False, False],
            "hard_safety_veto_v1": [False, False, False],
            "r5_bad_score_signal_bucket_v1": ["NONE", "NONE", "STRONG"],
            "r5_1_bad_score_signal_bucket_v1": ["NONE", "NONE", "NONE"],
            "r5_tail_score_signal_bucket_v1": ["NONE", "NONE", "NONE"],
        }
    )
    memberships = {
        "V2_OOF_CORE_ONLY": pd.Series([True, True, False]),
        "BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT": pd.Series([True, True, True]),
    }

    records = opp._run_id_support(rows, memberships)

    assert any(row["run_id_v1"] == opp.WORST_RUN_ID for row in records)


def test_recommendation_cannot_be_r6_or_live_ready() -> None:
    with pytest.raises(RuntimeError, match="OPPORTUNITY_BASE_RECOMMENDATION_CANNOT_BE_R6_OR_LIVE_READY"):
        opp.recommendation_not_r6_ready("OPPORTUNITY_BASE_R6_READY")


def test_every_included_positive_row_must_have_evidence() -> None:
    no_evidence = pd.Series(
        {
            "safe_recoverable_v1": True,
            "bad_label_v1": False,
            "tail_label_v1": False,
            "r5_bad_score_signal_bucket_v1": "NONE",
            "r5_1_bad_score_signal_bucket_v1": "NONE",
            "r5_tail_score_signal_bucket_v1": "NONE",
            "v2_oof_captured_v1": False,
            "historical_v2_captured_v1": False,
        }
    )
    signal_evidence = no_evidence.copy()
    signal_evidence["r5_bad_score_signal_bucket_v1"] = "SUPPORT"

    assert opp.row_has_positive_evidence(no_evidence) is False
    assert opp.row_has_positive_evidence(signal_evidence) is True


def test_broad_variant_cannot_claim_ready_when_run_id_support_worsens() -> None:
    variants = [
        _variant_row("V2_OOF_CORE_ONLY", total=69, worst=2, low=7),
        _variant_row("V2_OOF_PLUS_RUN_ID_SUPPORT", total=73, worst=2, low=6),
        _variant_row("BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT", total=209, worst=1, low=11),
    ]

    recommendation = opp._recommendation(variants, [{"run_id_v1": opp.WORST_RUN_ID}])

    assert recommendation["status_v1"] == "OPPORTUNITY_BASE_SIGNAL_PRESENT_BUT_RUN_ID_SUPPORT_WEAK"
    assert recommendation["recommended_variant_v1"] == "V2_OOF_PLUS_RUN_ID_SUPPORT"
    assert recommendation["broader_balanced_run_id_support_improved_v1"] is False
