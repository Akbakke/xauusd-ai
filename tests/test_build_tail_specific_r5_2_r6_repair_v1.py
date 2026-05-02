from __future__ import annotations

import pytest

from gx1.scripts import materialize_build_tail_specific_r5_2_r6_repair_v1 as tail


def _tail_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_uid_v1": "c1",
        "recommended_role_v1": "TAIL_REPAIR_PRIMARY_CANDIDATE",
        "active_quarantine_v1": "ACTIVE_CANDIDATE",
        "r5_tail_score_evidence_v1": True,
        "tail_control_10_50_evidence_v1": False,
        "v2_oof_tail_evidence_v1": False,
        "protected_winner_status_v1": False,
        "runner_protect_status_v1": False,
        "ambiguous_high_mfe_status_v1": False,
        "fifty_plus_mfe_risk_v1": False,
        "hundred_plus_mfe_risk_v1": False,
        "two_hundred_plus_mfe_risk_v1": False,
        "provenance_status_v1": "PASS",
    }
    row.update(overrides)
    return row


def test_tail_repair_cannot_use_protected_winners_as_positives() -> None:
    with pytest.raises(RuntimeError, match="SAFETY_CLEARANCE"):
        tail.validate_tail_repair_positive(_tail_row(protected_winner_status_v1=True))


def test_tail_repair_cannot_use_runner_protect_rows_as_positives() -> None:
    with pytest.raises(RuntimeError, match="SAFETY_CLEARANCE"):
        tail.validate_tail_repair_positive(_tail_row(runner_protect_status_v1=True))


def test_tail_repair_cannot_use_ambiguous_high_mfe_rows_as_positives_without_safe_proof() -> None:
    with pytest.raises(RuntimeError, match="SAFETY_CLEARANCE"):
        tail.validate_tail_repair_positive(_tail_row(ambiguous_high_mfe_status_v1=True))


def test_tail_repair_cannot_use_quarantine_rows_as_positives() -> None:
    with pytest.raises(RuntimeError, match="SAFETY_CLEARANCE"):
        tail.validate_tail_repair_positive(_tail_row(active_quarantine_v1="QUARANTINE"))


def test_tail_repair_positive_requires_tail_evidence_and_safety_clearance() -> None:
    assert tail.validate_tail_repair_positive(_tail_row()) is True
    with pytest.raises(RuntimeError, match="TAIL_EVIDENCE"):
        tail.validate_tail_repair_positive(
            _tail_row(
                r5_tail_score_evidence_v1=False,
                tail_control_10_50_evidence_v1=False,
                v2_oof_tail_evidence_v1=False,
            )
        )


def test_tail_repair_variants_are_deterministic_small_set() -> None:
    variants = [{"variant_id_v1": variant_id} for variant_id in tail.TAIL_REPAIR_VARIANT_IDS]
    assert tail.validate_variant_grid(variants) is True
    with pytest.raises(RuntimeError, match="TAIL_REPAIR_VARIANTS"):
        tail.validate_variant_grid([{"variant_id_v1": "BASE_R5_2_130_86_CONTROL"}])


def test_tail_repair_max_safe_variant_cannot_be_final_by_default() -> None:
    assert tail.max_safe_variant_can_be_final("TAIL_REPAIR_MAX_SAFE_DIAGNOSTIC") is False
    assert tail.max_safe_variant_can_be_final("TAIL_TARGET_WEIGHT_REPAIR_BALANCED") is True


def test_no_optuna_broad_sweep_freeze_promo_live() -> None:
    clean = tail.validate_no_forbidden_actions(optuna=False, broad_sweep=False, freeze=False, promo=False, live=False)
    blocked = tail.validate_no_forbidden_actions(optuna=True, broad_sweep=True, freeze=True, promo=True, live=True)

    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]
    assert "BROAD_SWEEP_FORBIDDEN" in blocked["failures_v1"]


def test_no_dummy_synthetic_fallback() -> None:
    clean = tail.validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    blocked = tail.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=True, fallback=True)

    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert set(blocked["failures_v1"]) == {"DUMMY_INPUT_FORBIDDEN", "SYNTHETIC_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"}


def test_no_implicit_latest_glob_artifact_selection() -> None:
    assert tail.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        tail.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_fixed_controls_include_r5_2_and_wednesday() -> None:
    controls = [
        {"control_v1": "r5_2_package"},
        {"control_v1": "wednesday_benchmark"},
    ]

    assert tail.validate_fixed_controls(controls) is True
    with pytest.raises(RuntimeError, match="TAIL_REPAIR_FIXED_CONTROLS_MISSING"):
        tail.validate_fixed_controls([{"control_v1": "r5_2_package"}])


def test_candidate_selection_requires_safety_and_precision_validity() -> None:
    assert tail.candidate_can_be_selected({"safety_clean_v1": True, "precision_decision_valid_v1": True}) is True
    assert tail.candidate_can_be_selected({"safety_clean_v1": False, "precision_decision_valid_v1": True}) is False
    assert tail.candidate_can_be_selected({"safety_clean_v1": True, "precision_decision_valid_v1": False}) is False


def test_input_artifact_hash_mismatch_causes_failure() -> None:
    result = tail.validate_input_artifacts_unchanged({"r5": "abc"}, {"r5": "def"})

    assert result["status_v1"] == "FAIL"
    assert result["changed_v1"] == ["r5"]


def test_best_path_final_promotion_remains_false_when_candidate_improves() -> None:
    result = tail._best_path(
        {
            "trained_v1": True,
            "best_candidate": {
                "safety_clean_v1": True,
                "precision_decision_valid_v1": True,
                "bad_count_v1": 130,
                "tail_count_v1": 90,
            },
        }
    )

    assert result["status_v1"] == "TAIL_REPAIR_CANDIDATE_BEATS_130_86_SAFELY_FINAL_PROMOTION_BLOCKED"
    assert result["final_promotion_allowed_v1"] is False


def test_low_support_and_strict_loso_remain_visible_in_anti_overfit_audit() -> None:
    result = tail._anti_overfit(
        {
            "trained_v1": True,
            "best_candidate": {
                "safety_clean_v1": True,
                "final_promotion_allowed_v1": False,
                "oof_provenance_status_v1": "PASS",
                "in_sample_scored_count_v1": 0,
            },
        },
        {"status_v1": "PASS"},
        {"status_v1": "PASS"},
    )

    assert result["status_v1"] == "TAIL_REPAIR_STABLE_TRACK_PASS"
    assert result["strict_loso_visible_v1"] is True
    assert result["low_support_visible_v1"] is True

