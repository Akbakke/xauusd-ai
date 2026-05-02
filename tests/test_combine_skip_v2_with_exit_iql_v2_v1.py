"""Tests for materialize_combine_skip_v2_with_exit_iql_v2_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_combine_skip_v2_with_exit_iql_v2_v1 as gate


def test_per_trade_realized_pnl_returns_last_bar_per_trade() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "A", "A", "B", "B"],
            "bars_held_v1": [0, 1, 2, 0, 1],
            "running_pnl_at_close_bps_v1": [10.0, 20.0, 30.0, -5.0, -10.0],
        }
    )
    out = gate._per_trade_realized_pnl(df)
    assert out["A"] == 30.0
    assert out["B"] == -10.0


def _mk_variant_metric(reward, n_skip, n_take, no_skip_real, no_skip_iql, skip_real, skip_iql):
    return {
        "reward_variant_v1": reward,
        "split_v1": "test",
        "trade_count_total_v1": n_skip + n_take,
        "trade_count_taken_v1": n_take,
        "trade_count_skipped_v1": n_skip,
        "pnl_no_skip_realized_v1": no_skip_real,
        "pnl_no_skip_iql_v1": no_skip_iql,
        "pnl_skip_then_realized_v1": skip_real,
        "pnl_skip_then_iql_v1": skip_iql,
        "lift_skip_only_v1": skip_real - no_skip_real,
        "lift_iql_only_v1": no_skip_iql - no_skip_real,
        "lift_combined_v1": skip_iql - no_skip_real,
        "lift_combined_minus_sum_of_components_v1": (
            (skip_iql - no_skip_real)
            - ((skip_real - no_skip_real) + (no_skip_iql - no_skip_real))
        ),
    }


def _baseline(realized, trail_stop):
    return {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": realized},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": trail_stop},
        ]
    }


def _test_eval(per_variant):
    return {
        "split_v1": "test",
        "tuned_threshold_v1": 0.5,
        "trade_count_total_v1": 258,
        "trade_count_taken_v1": 162,
        "trade_count_skipped_v1": 96,
        "pnl_no_skip_realized_v1": -355.0,
        "pnl_skip_then_realized_v1": 1842.0,
        "per_variant_v1": per_variant,
    }


def test_go_no_go_pass_superadditive_when_combined_beats_components_with_positive_interaction() -> None:
    # Combined 3000, skip-only 1500, iql-only 800 -> combined - sum-of-lifts = 3000 - 355 - (1500+355) - (800+355) ...
    # actually let's pick numbers that give clear superadditive.
    pv = [
        _mk_variant_metric(
            "X", n_skip=96, n_take=162, no_skip_real=-355.0,
            no_skip_iql=500.0, skip_real=1500.0, skip_iql=3000.0
        )
    ]
    status, action, _, headline = gate._go_no_go(_test_eval(pv), _baseline(-355.0, 1052.0))
    # Interaction = (3000 - (-355)) - ((1500-(-355)) + (500-(-355))) = 3355 - (1855 + 855) = 645 -> superadditive
    assert headline["additivity_classification_v1"] == "SUPERADDITIVE"
    assert status == "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SUPERADDITIVE_LIFT"
    assert action == "WALK_FORWARD_VALIDATION_V1"


def test_go_no_go_subadditive_when_combined_loses_to_skip_only() -> None:
    pv = [
        _mk_variant_metric(
            "GIVEBACK", n_skip=96, n_take=162, no_skip_real=-355.0,
            no_skip_iql=509.0, skip_real=1842.0, skip_iql=643.0
        )
    ]
    status, action, _, headline = gate._go_no_go(_test_eval(pv), _baseline(-355.0, 1052.0))
    # combined (643) < best component (1842) -> subadditive
    assert status == "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PARTIAL_SUBADDITIVE_LIFT"
    assert action == "REPAIR_COMBINED_STACK_BEFORE_PROMOTION_V1"
    assert headline["additivity_classification_v1"] == "SUBADDITIVE"
    assert headline["pnl_skip_then_iql_v1"] == 643.0
    assert headline["pnl_skip_then_realized_v1"] == 1842.0


def test_go_no_go_skip_dominates_when_combined_close_to_skip_only() -> None:
    pv = [
        _mk_variant_metric(
            "X", n_skip=96, n_take=162, no_skip_real=-355.0,
            no_skip_iql=200.0, skip_real=1842.0, skip_iql=1850.0
        )
    ]
    status, action, _, headline = gate._go_no_go(_test_eval(pv), _baseline(-355.0, 1052.0))
    assert status == "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SKIP_DOMINATES"
    assert action == "WALK_FORWARD_VALIDATION_V1"


def test_go_no_go_additive_when_combined_beats_components_with_neutral_interaction() -> None:
    # Choose numbers where interaction is between 0 and 50.
    # combined - sum-of-lifts where combined > best component by > 50 but interaction < 50.
    pv = [
        _mk_variant_metric(
            "X", n_skip=96, n_take=162, no_skip_real=-355.0,
            no_skip_iql=500.0, skip_real=1500.0, skip_iql=2400.0
        )
    ]
    # Interaction = (2400-(-355)) - ((1500-(-355)) + (500-(-355))) = 2755 - 2710 = 45 -> additive (between -50 and 50)
    status, action, _, headline = gate._go_no_go(_test_eval(pv), _baseline(-355.0, 1052.0))
    assert headline["additivity_classification_v1"] == "ADDITIVE"
    assert status == "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_ADDITIVE_LIFT"


def test_go_no_go_raises_on_empty_per_variant() -> None:
    with pytest.raises(RuntimeError, match="COMBINED_TEST_RESULTS_MISSING"):
        gate._go_no_go(_test_eval([]), _baseline(0.0, 0.0))


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "WALK_FORWARD_VALIDATION_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_ADDITIVE_LIFT", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_evaluate_combined_partitions_trades_correctly() -> None:
    """Synthetic test: 4 trades, threshold 0.5, p_skip = [0.7, 0.3, 0.6, 0.2].
    Should skip 2 trades (rows 0 and 2) and take 2 (rows 1 and 3)."""
    per_trade_split = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C", "D"],
            "primary_split_v1": ["test", "test", "test", "test"],
            "pnl_bps": [-100.0, 50.0, -80.0, 30.0],
        }
    )
    p_skip_full = np.array([0.7, 0.3, 0.6, 0.2])
    per_bar_full = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C", "D"],
            "bars_held_v1": [0, 0, 0, 0],
            "primary_split_v1": ["test", "test", "test", "test"],
            "running_pnl_at_close_bps_v1": [-100.0, 50.0, -80.0, 30.0],
        }
    )
    X_full = np.zeros((4, 2))
    # Synthetic IQL coefs that always exit at bar 0 -> realized = running_pnl_at_close
    models = {
        "STUB": {
            "coef_hold": np.array([0.0, 0.0]),
            "coef_exit_now": np.array([1.0, 0.0]),  # always > coef_hold
        }
    }
    out = gate._evaluate_combined(
        per_trade_split, p_skip_full, 0.5, per_bar_full, X_full, models, split="test"
    )
    assert out["trade_count_skipped_v1"] == 2
    assert out["trade_count_taken_v1"] == 2
    assert out["pnl_no_skip_realized_v1"] == pytest.approx(-100.0)
    # Taken: B (50.0) + D (30.0) = 80.0
    assert out["pnl_skip_then_realized_v1"] == pytest.approx(80.0)
