"""Tests for materialize_joint_policy_validation_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_joint_policy_validation_v1 as joint_gate
from gx1.scripts import entry_iql_gpu_core_v1 as entry_iql


def test_action_constant():
    assert joint_gate.ACTION == "JOINT_POLICY_VALIDATION_V1"


def test_locked_entry_iql_combo_matches_fold_3_winner():
    """The locked combo should match what FOLD_3 picked as best non-degenerate."""
    assert joint_gate.LOCKED_ENTRY_IQL_VARIANT == "PRIORITY_WEIGHTED_REWARD"
    assert joint_gate.LOCKED_ENTRY_IQL_TAU == 0.7
    assert joint_gate.LOCKED_ENTRY_IQL_BETA == 1.0


def test_validate_final_status_known_values():
    assert joint_gate.validate_final_status(
        "JOINT_PASS_BEATS_REALIZED_AND_TRAIL_STOP",
        "DEPLOY_JOINT_POLICY_TO_LIVE_PAPER_TRADING_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        joint_gate.validate_final_status("BOGUS", "RETRAIN_XGB_TRANSFORMERS_FULL_2020_2026_V1")


def test_skip_v2_target_uses_should_skip_when_present():
    df = pd.DataFrame({
        "hindsight_should_skip_trade_v1": [True, False, True, False],
        "realized_pnl_bps": [-50.0, 30.0, -10.0, 20.0],
    })
    y = joint_gate._build_skip_v2_target(df)
    assert list(y) == [1, 0, 1, 0]


def test_skip_v2_target_falls_back_to_pnl_sign():
    df = pd.DataFrame({"realized_pnl_bps": [-50.0, 30.0, -10.0, 20.0]})
    y = joint_gate._build_skip_v2_target(df)
    assert list(y) == [1, 0, 1, 0]


def test_evaluate_realized_pnl_skip_zero_take_actual():
    df = pd.DataFrame({
        "action_label_v1": ["TAKE_NOW", "SKIP", "WAIT", "TAKE_NOW"],
        "realized_pnl_bps": [10.0, -5.0, 3.0, -7.0],
    })
    out = joint_gate._evaluate_realized(df)
    # SKIP/WAIT contribute 0, only TAKE_NOW counts: 10 + (-7) = 3
    assert out["total_pnl_bps_v1"] == 3.0
    assert out["n_taken_v1"] == 2
    assert out["n_skipped_v1"] == 1
    assert out["n_waited_v1"] == 1


def test_evaluate_skip_v2_at_threshold():
    df = pd.DataFrame({"realized_pnl_bps": [10.0, -5.0, 30.0, -20.0]})
    p_skip = np.array([0.1, 0.7, 0.4, 0.9])
    out = joint_gate._evaluate_skip_v2(df, p_skip, threshold=0.5)
    # rows 1, 3 skipped (p>=0.5); rows 0, 2 taken
    assert out["total_pnl_bps_v1"] == 10.0 + 30.0
    assert out["n_skipped_v1"] == 2
    assert out["n_taken_v1"] == 2


def test_evaluate_entry_iql_actions_to_pnl():
    df = pd.DataFrame({
        "realized_pnl_bps": [10.0, 5.0, 30.0],
        "hindsight_policy_counterfactual_value_bps_v1": [0.0, 25.0, 0.0],
    })
    actions = np.array([
        entry_iql.ACTION_TAKE_NOW_ID,
        entry_iql.ACTION_WAIT_ID,
        entry_iql.ACTION_SKIP_ID,
    ])
    out = joint_gate._evaluate_entry_iql(df, actions)
    # row 0 TAKE → +10; row 1 WAIT → +25; row 2 SKIP → 0
    assert out["total_pnl_bps_v1"] == 10.0 + 25.0 + 0.0
    assert out["n_taken_v1"] == 1
    assert out["n_waited_v1"] == 1
    assert out["n_skipped_v1"] == 1


def test_hindsight_optimal_picks_max_per_row():
    df = pd.DataFrame({
        "realized_pnl_bps": [10.0, -5.0, 0.0],
        "hindsight_policy_counterfactual_value_bps_v1": [3.0, 2.0, 7.0],
    })
    out = joint_gate._hindsight_optimal(df)
    # row 0: max(0, 10, 3) = 10; row 1: max(0, -5, 2) = 2; row 2: max(0, 0, 7) = 7
    assert out["total_pnl_bps_v1"] == 10.0 + 2.0 + 7.0


def test_train_skip_v2_returns_classifier():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = joint_gate._train_skip_v2_logistic(X, y)
    p = joint_gate._predict_skip_v2(clf, X)
    assert p.shape == (200,)
    assert (p >= 0).all() and (p <= 1).all()


def test_go_no_go_pass_when_joint_beats_realized_and_trail_stop():
    per_fold = [
        {
            "fold_id_v1": "FOLD_1",
            "realized_test_v1": {"total_pnl_bps_v1": 100.0},
            "skip_v2_test_v1": {"total_pnl_bps_v1": 5000.0},
            "skip_v2_best_threshold_v1": 0.5,
            "entry_iql_test_v1": {"total_pnl_bps_v1": 4000.0, "n_skipped_v1": 0, "n_taken_v1": 0, "n_waited_v1": 0},
            "hindsight_test_v1": {"total_pnl_bps_v1": 8000.0},
        }
    ]
    status, action, _, headline = joint_gate._go_no_go(per_fold)
    # 5000 > 100 + 200 AND 5000 > 1052 → PASS
    assert status == "JOINT_PASS_BEATS_REALIZED_AND_TRAIL_STOP"
    assert headline["best_policy_v1"] == "SKIP_V2_THEN_V3"


def test_go_no_go_partial_when_ties_realized():
    per_fold = [
        {
            "fold_id_v1": "FOLD_1",
            "realized_test_v1": {"total_pnl_bps_v1": 1000.0},
            "skip_v2_test_v1": {"total_pnl_bps_v1": 1100.0},
            "skip_v2_best_threshold_v1": 0.5,
            "entry_iql_test_v1": {"total_pnl_bps_v1": 1050.0, "n_skipped_v1": 0, "n_taken_v1": 0, "n_waited_v1": 0},
            "hindsight_test_v1": {"total_pnl_bps_v1": 5000.0},
        }
    ]
    status, action, _, _ = joint_gate._go_no_go(per_fold)
    assert status == "JOINT_PARTIAL_TIES_REALIZED"
