"""Tests for materialize_build_hybrid_trail_stop_plus_small_adjustment_learner_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_build_hybrid_trail_stop_plus_small_adjustment_learner_v1 as gate,
)


def _per_bar(uid, pnls, mfes):
    return pd.DataFrame(
        {
            "candidate_uid_v1": [uid] * len(pnls),
            "primary_split_v1": ["test"] * len(pnls),
            "bars_held_v1": list(range(len(pnls))),
            "running_pnl_at_close_bps_v1": pnls,
            "running_mfe_bps_v1": mfes,
        }
    )


def test_compute_trail_stop_per_trade_never_fires_when_mfe_too_low() -> None:
    df = _per_bar("X", [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    out = gate._compute_trail_stop_per_trade(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["firing_status_v1"] == "NEVER_FIRED"
    assert row["trail_stop_pnl_v1"] == 3.0
    assert row["realized_pnl_v1"] == 3.0
    assert row["would_delay_help_v1"] == 0


def test_compute_trail_stop_per_trade_fires_when_giveback_threshold() -> None:
    # MFE 100 at bar 1, PNL drops to 70 at bar 2 -> giveback 30/100 = 0.30 >= 0.25
    df = _per_bar("X", [0.0, 100.0, 70.0, 50.0], [0.0, 100.0, 100.0, 100.0])
    out = gate._compute_trail_stop_per_trade(df)
    row = out.iloc[0]
    assert row["firing_status_v1"] == "FIRED"
    assert row["fire_bar_index_v1"] == 2
    assert row["trail_stop_pnl_v1"] == 70.0


def test_compute_trail_stop_label_would_delay_help_when_post_fire_higher() -> None:
    # PNL recovers to 150 after firing at 70 -> delaying would help (delta > 5)
    df = _per_bar("X", [0.0, 100.0, 70.0, 150.0, 80.0], [0.0, 100.0, 100.0, 150.0, 150.0])
    out = gate._compute_trail_stop_per_trade(df)
    assert out.iloc[0]["would_delay_help_v1"] == 1


def test_compute_trail_stop_label_would_delay_not_help_when_post_fire_lower() -> None:
    df = _per_bar("X", [0.0, 100.0, 70.0, 50.0, 60.0], [0.0, 100.0, 100.0, 100.0, 100.0])
    out = gate._compute_trail_stop_per_trade(df)
    assert out.iloc[0]["would_delay_help_v1"] == 0


def _hybrid_input_df():
    return pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C", "D"],
            "firing_status_v1": ["FIRED", "FIRED", "NEVER_FIRED", "FIRED"],
            "trail_stop_pnl_v1": [50.0, 100.0, 0.0, -30.0],
            "realized_pnl_v1": [80.0, 90.0, 20.0, -100.0],
            "would_delay_help_v1": [1, 0, 0, 0],
            "p_delay_v1": [0.7, 0.3, 0.6, 0.8],
        }
    )


def test_evaluate_hybrid_at_threshold_applies_decision_correctly() -> None:
    df = _hybrid_input_df()
    out = gate._evaluate_hybrid_at_threshold(df, threshold=0.5)
    # A: FIRED, p=0.7 >= 0.5 -> delay -> realized 80
    # B: FIRED, p=0.3 < 0.5 -> fire -> trail_stop 100
    # C: NEVER_FIRED -> realized 20
    # D: FIRED, p=0.8 >= 0.5 -> delay -> realized -100
    expected_pnl = 80.0 + 100.0 + 20.0 + (-100.0)
    assert out["hybrid_total_pnl_v1"] == pytest.approx(expected_pnl)
    assert out["delay_count_v1"] == 2
    assert out["fire_count_v1"] == 1
    assert out["never_fired_count_v1"] == 1


def test_evaluate_hybrid_at_threshold_higher_threshold_fires_more() -> None:
    df = _hybrid_input_df()
    out_low = gate._evaluate_hybrid_at_threshold(df, threshold=0.5)
    out_high = gate._evaluate_hybrid_at_threshold(df, threshold=0.75)
    # At threshold 0.75: A (p=0.7) doesn't delay -> fires; B doesn't delay -> fires;
    # D (p=0.8) still delays.
    assert out_high["delay_count_v1"] == 1
    assert out_high["fire_count_v1"] == 2
    assert out_low["delay_count_v1"] >= out_high["delay_count_v1"]


def test_evaluate_hybrid_confusion_matrix_only_for_fired_trades() -> None:
    df = _hybrid_input_df()
    out = gate._evaluate_hybrid_at_threshold(df, threshold=0.5)
    # Among fired (A, B, D):
    #  A: pred=1 (delay), label=1 -> TP
    #  B: pred=0, label=0 -> TN
    #  D: pred=1 (delay), label=0 -> FP
    assert out["tp_v1"] == 1
    assert out["fp_v1"] == 1
    assert out["tn_v1"] == 1
    assert out["fn_v1"] == 0


def test_label_threshold_constant() -> None:
    assert gate.LABEL_DELAY_PNL_DELTA_THRESHOLD_BPS == 5.0


def test_delay_probability_grid_covers_balanced_range() -> None:
    assert 0.50 in gate.DELAY_PROBABILITY_GRID
    assert min(gate.DELAY_PROBABILITY_GRID) <= 0.30
    assert max(gate.DELAY_PROBABILITY_GRID) >= 0.70


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "BUILD_HYBRID_TRAIL_STOP_PASS_MEETS_PROMOTION_CRITERIA", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def _mk_promotion_eval(
    overall_pass, n_pass, mean_lift_ts, n_folds_beating_ts, lifts_ts
):
    return {
        "overall_pass_v1": overall_pass,
        "n_criteria_passed_v1": n_pass,
        "n_criteria_total_v1": 6,
        "per_fold_lifts_vs_trail_stop_v1": lifts_ts,
        "per_fold_lifts_vs_realized_v1": [200.0] * len(lifts_ts),
        "per_fold_pnl_v1": [800.0] * len(lifts_ts),
        "per_fold_trail_stop_pnl_v1": [600.0] * len(lifts_ts),
    }


def test_go_no_go_pass_when_all_promotion_criteria_met() -> None:
    promotion = _mk_promotion_eval(
        overall_pass=True, n_pass=6, mean_lift_ts=300.0,
        n_folds_beating_ts=3, lifts_ts=[400.0, 200.0, 300.0]
    )
    status, action, _, _ = gate._go_no_go(promotion)
    assert status == "BUILD_HYBRID_TRAIL_STOP_PASS_MEETS_PROMOTION_CRITERIA"
    assert action == "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1"


def test_go_no_go_pass_when_beats_trail_stop_but_fails_other_criteria() -> None:
    promotion = _mk_promotion_eval(
        overall_pass=False, n_pass=4, mean_lift_ts=150.0,
        n_folds_beating_ts=3, lifts_ts=[100.0, 200.0, 150.0]
    )
    status, action, _, _ = gate._go_no_go(promotion)
    assert status == "BUILD_HYBRID_TRAIL_STOP_PASS_BEATS_TRAIL_STOP_BUT_FAILS_OTHER_CRITERIA"
    assert action == "BUILD_REGIME_CONDITIONED_HYBRID_TRAIL_STOP_V2"


def test_go_no_go_partial_when_ties_trail_stop() -> None:
    promotion = _mk_promotion_eval(
        overall_pass=False, n_pass=2, mean_lift_ts=10.0,
        n_folds_beating_ts=2, lifts_ts=[50.0, -30.0, 10.0]
    )
    status, _, _, _ = gate._go_no_go(promotion)
    assert status == "BUILD_HYBRID_TRAIL_STOP_PARTIAL_TIES_TRAIL_STOP"


def test_go_no_go_partial_when_degrades_vs_trail_stop() -> None:
    promotion = _mk_promotion_eval(
        overall_pass=False, n_pass=2, mean_lift_ts=-356.0,
        n_folds_beating_ts=2, lifts_ts=[786.0, 50.0, -1904.0]
    )
    status, action, _, _ = gate._go_no_go(promotion)
    assert status == "BUILD_HYBRID_TRAIL_STOP_PARTIAL_DEGRADES_VS_TRAIL_STOP"
    assert action == "REPAIR_HYBRID_TRAIL_STOP_BEFORE_FURTHER_WORK_V1"
