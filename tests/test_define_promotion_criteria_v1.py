"""Tests for materialize_define_promotion_criteria_v1."""
from __future__ import annotations

import pytest

from gx1.scripts import materialize_define_promotion_criteria_v1 as gate


def test_promotion_criteria_has_six_criteria() -> None:
    assert len(gate.PROMOTION_CRITERIA_V1["criteria_v1"]) == 6
    ids = {c["criterion_id_v1"] for c in gate.PROMOTION_CRITERIA_V1["criteria_v1"]}
    assert ids == {
        "CROSS_FOLD_STABILITY",
        "MIN_MEAN_LIFT_BPS",
        "MAX_SINGLE_FOLD_LOSS_BPS",
        "BEAT_TRAIL_STOP_RULE",
        "DETERMINISTIC_REPRODUCIBLE",
        "NO_FORBIDDEN_LEAK",
    }


def test_paper_trading_blocked_in_contract() -> None:
    block = gate.PROMOTION_CRITERIA_V1["downstream_block_v1"]
    assert block["paper_trading_allowed_v1"] is False
    assert block["live_trading_allowed_v1"] is False
    assert block["adapter_build_allowed_v1"] is False


def test_evaluate_passes_when_all_criteria_met() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="ideal",
        per_fold_lifts_bps=[300.0, 350.0, 400.0],
        per_fold_pnl_bps=[1500.0, 1600.0, 1700.0],
        per_fold_trail_stop_pnl_bps=[1000.0, 1100.0, 1200.0],
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    assert out["overall_pass_v1"] is True
    assert out["n_criteria_passed_v1"] == 6


def test_evaluate_fails_on_cross_fold_instability() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="unstable",
        per_fold_lifts_bps=[1000.0, -500.0, -200.0],
        per_fold_pnl_bps=[2000.0, 100.0, 300.0],
        per_fold_trail_stop_pnl_bps=[1000.0, 1000.0, 1000.0],
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    assert out["overall_pass_v1"] is False
    cross_fold = next(c for c in out["breakdown_v1"] if c["criterion_id_v1"] == "CROSS_FOLD_STABILITY")
    assert cross_fold["passed_v1"] is False


def test_evaluate_fails_on_low_mean_lift() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="weak",
        per_fold_lifts_bps=[50.0, 80.0, 100.0],  # mean 76.7 < 200
        per_fold_pnl_bps=[100.0, 200.0, 300.0],
        per_fold_trail_stop_pnl_bps=[50.0, 100.0, 150.0],
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    weak = next(c for c in out["breakdown_v1"] if c["criterion_id_v1"] == "MIN_MEAN_LIFT_BPS")
    assert weak["passed_v1"] is False


def test_evaluate_fails_on_catastrophic_single_fold_loss() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="catastrophic",
        per_fold_lifts_bps=[500.0, 600.0, -300.0],  # min < -200
        per_fold_pnl_bps=[1000.0, 1100.0, 200.0],
        per_fold_trail_stop_pnl_bps=[800.0, 900.0, 100.0],
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    cat = next(
        c for c in out["breakdown_v1"]
        if c["criterion_id_v1"] == "MAX_SINGLE_FOLD_LOSS_BPS"
    )
    assert cat["passed_v1"] is False


def test_evaluate_fails_on_loss_to_trail_stop() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="weak_vs_rule",
        per_fold_lifts_bps=[300.0, 350.0, 400.0],
        per_fold_pnl_bps=[500.0, 600.0, 700.0],  # mean 600
        per_fold_trail_stop_pnl_bps=[1000.0, 1100.0, 1200.0],  # mean 1100
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    rule = next(
        c for c in out["breakdown_v1"]
        if c["criterion_id_v1"] == "BEAT_TRAIL_STOP_RULE"
    )
    assert rule["passed_v1"] is False


def test_evaluate_fails_on_leakage() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="leaky",
        per_fold_lifts_bps=[300.0, 350.0, 400.0],
        per_fold_pnl_bps=[2000.0, 2100.0, 2200.0],
        per_fold_trail_stop_pnl_bps=[1000.0, 1100.0, 1200.0],
        no_shortcut_audit_passed=False,
        deterministic_reproducible=True,
    )
    leak = next(c for c in out["breakdown_v1"] if c["criterion_id_v1"] == "NO_FORBIDDEN_LEAK")
    assert leak["passed_v1"] is False
    assert out["overall_pass_v1"] is False


def test_evaluate_fails_on_non_deterministic() -> None:
    out = gate.evaluate_candidate_against_criteria(
        candidate_id="non_repro",
        per_fold_lifts_bps=[300.0, 350.0, 400.0],
        per_fold_pnl_bps=[2000.0, 2100.0, 2200.0],
        per_fold_trail_stop_pnl_bps=[1000.0, 1100.0, 1200.0],
        no_shortcut_audit_passed=True,
        deterministic_reproducible=False,
    )
    repro = next(c for c in out["breakdown_v1"] if c["criterion_id_v1"] == "DETERMINISTIC_REPRODUCIBLE")
    assert repro["passed_v1"] is False
    assert out["overall_pass_v1"] is False


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("DEFINE_PROMOTION_CRITERIA_LOCKED_V1", "TRAIN_NOW")


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))
