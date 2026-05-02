"""Tests for Phase 2 gates: head-to-head, rolling-window, regime ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_run_live_system_vs_research_candidates_head_to_head_v1 as h2h_gate,
)
from gx1.scripts import (
    materialize_build_rolling_window_retrained_skip_v1 as rolling_gate,
)
from gx1.scripts import (
    materialize_build_regime_detector_plus_policy_ensemble_v1 as regime_gate,
)


# ---------------------------------------------------------------------------
# Head-to-head tests
# ---------------------------------------------------------------------------


def test_h2h_per_policy_metrics_basic_stats() -> None:
    df = pd.DataFrame(
        {
            "REALIZED_LIVE_SYSTEM": [100.0, -50.0, 200.0, -10.0, 30.0],
        }
    )
    out = h2h_gate._per_policy_metrics(df, "REALIZED_LIVE_SYSTEM")
    assert out["trade_count_v1"] == 5
    assert out["total_pnl_bps_v1"] == 270.0
    assert out["mean_pnl_bps_v1"] == pytest.approx(54.0)
    assert out["win_rate_v1"] == pytest.approx(0.6)
    assert out["best_trade_v1"] == 200.0
    assert out["worst_trade_v1"] == -50.0


def test_h2h_pairwise_correlation_identifies_perfect_correlation() -> None:
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],  # 2 * A
            "C": [-1.0, -2.0, -3.0, -4.0, -5.0],  # -A
        }
    )
    matrix = h2h_gate._pairwise_correlation_matrix(df, ["A", "B", "C"])
    assert matrix["A"]["B"] == pytest.approx(1.0)
    assert matrix["A"]["C"] == pytest.approx(-1.0)
    assert matrix["A"]["A"] == pytest.approx(1.0)


def test_h2h_diversification_score_higher_for_anticorrelated() -> None:
    matrix = {
        "ref": {"ref": 1.0, "X": 0.5, "Y": -0.3},
        "X": {"ref": 0.5, "X": 1.0, "Y": 0.0},
        "Y": {"ref": -0.3, "X": 0.0, "Y": 1.0},
    }
    out = h2h_gate._diversification_score(matrix, reference="ref")
    assert out["ref"] == 0.0
    assert out["X"] == pytest.approx(0.5)
    assert out["Y"] == pytest.approx(1.3)


def test_h2h_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        h2h_gate.validate_final_status(
            "MADE_UP", "BUILD_ROLLING_WINDOW_RETRAINED_SKIP_V1"
        )


def test_h2h_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    h2h_gate.validate_no_deprecated_revival(Path(h2h_gate.__file__))


# ---------------------------------------------------------------------------
# Rolling-window tests
# ---------------------------------------------------------------------------


def test_rolling_compute_steps_correct_window_layout() -> None:
    n = 1000
    steps = rolling_gate._compute_steps(n)
    # First step: train_start=1, train_end depends on TRAIN_FRACTION_WITHIN_WINDOW=0.85,
    # WINDOW_SIZE=800 -> train_end = 680, val_end = 800, test_end = min(850, 1000) = 850
    first = steps[0]
    assert first["train_start_v1"] == 1
    assert first["train_end_v1"] == 680  # 0.85 * 800
    assert first["val_end_v1"] == 800
    assert first["test_end_v1"] == 850
    # Subsequent step starts at test_end + 1.
    assert steps[1]["train_start_v1"] == 51  # test_end of step 1 was 850; window 800 -> start = 51
    assert steps[1]["train_end_v1"] == 730


def test_rolling_step_uid_to_split_assigns_correctly() -> None:
    uids = [f"U_{i:04d}" for i in range(1, 21)]  # 1..20
    step = {"train_start_v1": 5, "train_end_v1": 10, "val_end_v1": 14, "test_end_v1": 18}
    out = rolling_gate._step_uid_to_split(uids, step)
    assert out["U_0001"] == "hold_out"
    assert out["U_0004"] == "hold_out"
    assert out["U_0005"] == "train"
    assert out["U_0010"] == "train"
    assert out["U_0011"] == "val"
    assert out["U_0014"] == "val"
    assert out["U_0015"] == "test"
    assert out["U_0018"] == "test"
    assert out["U_0019"] == "hold_out"


def test_rolling_window_size_constants() -> None:
    assert rolling_gate.WINDOW_SIZE_TRADES == 800
    assert rolling_gate.STEP_SIZE_TRADES == 50
    assert 0.5 < rolling_gate.TRAIN_FRACTION_WITHIN_WINDOW < 1.0


def test_rolling_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        rolling_gate.validate_final_status(
            "MADE_UP", "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1"
        )


def test_rolling_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    rolling_gate.validate_no_deprecated_revival(Path(rolling_gate.__file__))


# ---------------------------------------------------------------------------
# Regime ensemble tests
# ---------------------------------------------------------------------------


def test_regime_label_threshold() -> None:
    assert regime_gate.REGIME_LOSS_LABEL_THRESHOLD_BPS == -50.0


def test_regime_add_regime_label_marks_losing_trades() -> None:
    df = pd.DataFrame({"pnl_bps": [-100.0, -25.0, 50.0, -60.0]})
    out = regime_gate._add_regime_label(df)
    assert out["regime_loss_v1"].tolist() == [1, 0, 0, 1]


def test_regime_evaluate_ensemble_routes_correctly() -> None:
    per_trade = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C", "D"],
            "regime_loss_v1": [1, 0, 1, 0],
            "pnl_bps": [-100.0, 50.0, -80.0, 30.0],
        }
    )
    p_loss = np.array([0.7, 0.3, 0.6, 0.2])
    realized_per_uid = {"A": -100.0, "B": 50.0, "C": -80.0, "D": 30.0}
    combined_per_uid = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    out = regime_gate._evaluate_ensemble_at_threshold(
        per_trade, p_loss, realized_per_uid, combined_per_uid, threshold=0.5
    )
    # Routed to combined (p_loss >= 0.5): A and C -> use combined PNL = 0
    # Routed to realized: B (50) and D (30)
    assert out["n_routed_combined_v1"] == 2
    assert out["n_routed_realized_v1"] == 2
    assert out["ensemble_total_pnl_v1"] == pytest.approx(0.0 + 50.0 + 0.0 + 30.0)
    assert out["realized_total_pnl_v1"] == pytest.approx(-100 + 50 - 80 + 30)
    # confusion: A pred=1 label=1 -> TP; B pred=0 label=0 -> TN; C pred=1 label=1 -> TP; D pred=0 label=0 -> TN
    assert out["tp_v1"] == 2
    assert out["fp_v1"] == 0
    assert out["tn_v1"] == 2
    assert out["fn_v1"] == 0


def test_regime_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        regime_gate.validate_final_status(
            "MADE_UP", "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1"
        )


def test_regime_validate_final_status_rejects_unknown_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        regime_gate.validate_final_status(
            "REGIME_ENSEMBLE_PASS_MEETS_PROMOTION_CRITERIA", "TRAIN_NOW"
        )


def test_regime_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    regime_gate.validate_no_deprecated_revival(Path(regime_gate.__file__))


def test_regime_compute_realized_per_uid_uses_last_bar() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "A", "A", "B", "B"],
            "bars_held_v1": [0, 1, 2, 0, 1],
            "running_pnl_at_close_bps_v1": [10.0, 20.0, 30.0, -5.0, -10.0],
        }
    )
    out = regime_gate._compute_realized_per_uid(df)
    assert out["A"] == 30.0
    assert out["B"] == -10.0


def test_regime_threshold_grid_balanced() -> None:
    assert 0.50 in regime_gate.REGIME_THRESHOLD_GRID
    assert min(regime_gate.REGIME_THRESHOLD_GRID) <= 0.30
    assert max(regime_gate.REGIME_THRESHOLD_GRID) >= 0.70
