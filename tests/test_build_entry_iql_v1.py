"""Tests for materialize_build_entry_iql_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_build_entry_iql_v1 as entry_gate
from gx1.scripts import entry_iql_gpu_core_v1 as entry_iql


def test_action_constant():
    assert entry_gate.ACTION == "BUILD_ENTRY_IQL_V1"


def test_reward_variants_complete():
    ids = {v["reward_id_v1"] for v in entry_gate.REWARD_VARIANTS_V1}
    assert ids == {
        "REALIZED_PNL_REWARD",
        "PRIORITY_WEIGHTED_REWARD",
        "TEACHER_AGREE_REWARD",
        "AUDIT_SHOULD_SKIP_REWARD",
    }


def test_audit_should_skip_reward_rewards_skip_when_should_skip_true():
    df = pd.DataFrame({
        "hindsight_should_skip_trade_v1": [True, True, False],
        "hindsight_skip_trade_avoided_loss_bps_v1": [50.0, 30.0, 0.0],
        "hindsight_teacher_priority_bps_v1": [50.0, 30.0, 100.0],
        "realized_pnl_bps": [-50.0, -30.0, 100.0],
        "hindsight_policy_counterfactual_value_bps_v1": [0.0, 0.0, 50.0],
    })
    actions = np.array([0, 1, 1])  # SKIP, TAKE_NOW, TAKE_NOW
    gt = np.array([0, 1, 1])
    r = entry_gate._build_audit_should_skip_reward(df, actions, gt)
    assert r[0] == 50.0  # should_skip=True, took SKIP → +avoided_loss
    assert r[1] == -30.0  # should_skip=True, took TAKE_NOW → -avoided_loss
    assert r[2] == 100.0  # should_skip=False, took TAKE_NOW → realized_pnl


def test_class_balanced_weights_invert_class_frequency():
    actions = np.array([0, 0, 0, 0, 0, 0, 1, 2, 2])  # 6 SKIP, 1 TAKE, 2 WAIT
    w = entry_gate._build_class_balanced_weights(actions)
    # Expected raw inv_freq: 1/6, 1/1, 1/2 = 0.167, 1.0, 0.5
    # Sum: 1.667. Normalize so mean=1 across 3 classes: scale=3/1.667=1.8
    # Per-class normalized: 0.3, 1.8, 0.9
    assert w[0] < w[6]  # SKIP weight < TAKE weight
    assert w[6] > w[7]  # TAKE weight > WAIT weight
    # Each sample with same class has same weight
    assert w[0] == w[1] == w[5]
    assert w[7] == w[8]


def test_validate_final_status_rejects_unknown():
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        entry_gate.validate_final_status("BOGUS", "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V1")


def test_validate_final_status_accepts_pass():
    assert entry_gate.validate_final_status(
        "ENTRY_IQL_PASS_BEATS_REALIZED",
        "BUILD_ENTRY_TIMING_IQL_WITH_WAIT_N_BARS_V1",
    )


def test_extract_action_id_for_known_strings():
    assert entry_gate._extract_action_id("SKIP") == 0
    assert entry_gate._extract_action_id("SKIP_ENTRY") == 0
    assert entry_gate._extract_action_id("TAKE_NOW") == 1
    assert entry_gate._extract_action_id("TAKE_ENTRY") == 1
    assert entry_gate._extract_action_id("WAIT") == 2
    assert entry_gate._extract_action_id("WAIT_ENTRY") == 2


def test_extract_action_id_for_nan_returns_none():
    assert entry_gate._extract_action_id(None) is None
    assert entry_gate._extract_action_id(np.nan) is None


def test_candidate_feature_columns_excludes_leak():
    df = pd.DataFrame({
        "as_of_entry_replay_micro_momentum_3_v1": [1.0, 2.0, 3.0],
        "as_of_entry_candidate_p_long_v1": [0.5, 0.6, 0.7],
        "hindsight_entry_skip_v1": [True, False, True],
        "skipability_label_v1": ["SKIP", "TAKE_NOW", "WAIT"],
        "candidate_uid": ["A", "B", "C"],
        "realized_pnl_bps": [10.0, -5.0, 2.0],
        "as_of_candidate_vol_regime_v1": [1, 2, 1],
    })
    cols = entry_gate._candidate_feature_columns(df)
    assert "as_of_entry_replay_micro_momentum_3_v1" in cols
    assert "as_of_entry_candidate_p_long_v1" in cols
    assert "as_of_candidate_vol_regime_v1" in cols
    # Leaks excluded:
    assert "hindsight_entry_skip_v1" not in cols
    assert "skipability_label_v1" not in cols
    assert "candidate_uid" not in cols
    assert "realized_pnl_bps" not in cols


def test_realized_pnl_reward_skip_zero_take_pnl_wait_cf():
    df = pd.DataFrame({
        "realized_pnl_bps": [50.0, -20.0, 30.0],
        "hindsight_policy_counterfactual_value_bps_v1": [10.0, 25.0, -5.0],
    })
    actions_taken = np.array([0, 1, 2])
    gt = np.array([0, 1, 2])
    r = entry_gate._build_realized_pnl_reward(df, actions_taken, gt)
    assert r[0] == 0.0  # SKIP → 0
    assert r[1] == -20.0  # TAKE_NOW → realized
    assert r[2] == -5.0  # WAIT → cf value


def test_priority_weighted_reward_signed_by_agreement():
    df = pd.DataFrame({
        "hindsight_teacher_priority_bps_v1": [50.0, -30.0, 20.0],
    })
    actions_taken = np.array([1, 0, 1])
    gt = np.array([1, 1, 2])
    r = entry_gate._build_priority_weighted_reward(df, actions_taken, gt)
    assert r[0] == 50.0  # match → +abs
    assert r[1] == -30.0  # mismatch → -abs
    assert r[2] == -20.0  # mismatch → -abs


def test_teacher_agree_reward_pm_one():
    df = pd.DataFrame()
    actions = np.array([0, 1, 2, 1])
    gt = np.array([0, 1, 0, 2])
    r = entry_gate._build_teacher_agree_reward(df, actions, gt)
    assert list(r) == [1.0, 1.0, -1.0, -1.0]


def test_walk_forward_splits_three_folds_with_increasing_train_size():
    df = pd.DataFrame({"x": np.arange(1000)})
    folds = entry_gate._build_walk_forward_splits(df)
    assert len(folds) == 3
    train_sizes = [len(f["train_idx_v1"]) for f in folds]
    assert train_sizes[0] < train_sizes[1] < train_sizes[2]
    # Sum of train+val+test ≤ total
    for f in folds:
        s = len(f["train_idx_v1"]) + len(f["val_idx_v1"]) + len(f["test_idx_v1"])
        assert s <= len(df)


def test_evaluate_policy_pnl_skip_yields_zero():
    df = pd.DataFrame({
        "realized_pnl_bps": [100.0, -50.0, 30.0],
        "hindsight_policy_counterfactual_value_bps_v1": [10.0, 5.0, -3.0],
    })
    actions = np.array([0, 0, 0])  # all SKIP
    out = entry_gate._evaluate_policy_pnl(df, actions)
    assert out["total_pnl_bps_v1"] == 0.0
    assert out["action_dist_v1"]["n_skip_v1"] == 3


def test_evaluate_policy_pnl_take_yields_realized():
    df = pd.DataFrame({
        "realized_pnl_bps": [100.0, -50.0, 30.0],
        "hindsight_policy_counterfactual_value_bps_v1": [0.0, 0.0, 0.0],
    })
    actions = np.array([1, 1, 1])
    out = entry_gate._evaluate_policy_pnl(df, actions)
    assert out["total_pnl_bps_v1"] == 80.0  # 100 - 50 + 30


def test_hindsight_optimal_actions_picks_best_per_row():
    df = pd.DataFrame({
        "realized_pnl_bps": [100.0, -50.0, 5.0],
        "hindsight_policy_counterfactual_value_bps_v1": [50.0, 200.0, 0.0],
    })
    actions = entry_gate._hindsight_optimal_actions(df)
    # row 0: realized=100 > cf=50 → TAKE_NOW (1)
    # row 1: realized=-50 < cf=200 → WAIT (2)
    # row 2: realized=5 > 0 but cf=0 not > realized → TAKE_NOW (1)
    assert actions[0] == entry_iql.ACTION_TAKE_NOW_ID
    assert actions[1] == entry_iql.ACTION_WAIT_ID
    assert actions[2] == entry_iql.ACTION_TAKE_NOW_ID


def test_go_no_go_pass_when_iql_beats_realized_significantly():
    per_fold = [
        {
            "iql_test_metric_v1": {
                "total_pnl_bps_v1": 1000.0,
                "action_dist_v1": {"n_skip_v1": 30, "n_take_now_v1": 40, "n_wait_v1": 30, "n_total_v1": 100},
            },
            "test_baselines_v1": {
                "realized_v1": {"total_pnl_bps_v1": 100.0, "action_dist_v1": {}},
                "skipability_teacher_v1": {"total_pnl_bps_v1": 500.0, "action_dist_v1": {}},
                "hindsight_optimal_v1": {"total_pnl_bps_v1": 1500.0, "action_dist_v1": {}},
                "always_take_v1": {"total_pnl_bps_v1": 50.0, "action_dist_v1": {}},
                "random_uniform_v1": {"total_pnl_bps_v1": 0.0, "action_dist_v1": {}},
            },
            "best_combo_v1": {"variant": "REALIZED_PNL_REWARD", "tau": 0.7, "beta": 3.0},
        }
    ]
    status, action, _, headline = entry_gate._go_no_go(per_fold)
    assert status == "ENTRY_IQL_PASS_BEATS_REALIZED"
    assert headline["iql_minus_realized_v1"] == 900.0


def test_go_no_go_partial_when_action_dist_degenerate():
    per_fold = [
        {
            "iql_test_metric_v1": {
                "total_pnl_bps_v1": 1000.0,
                "action_dist_v1": {"n_skip_v1": 0, "n_take_now_v1": 100, "n_wait_v1": 0, "n_total_v1": 100},
            },
            "test_baselines_v1": {
                "realized_v1": {"total_pnl_bps_v1": 100.0, "action_dist_v1": {}},
                "skipability_teacher_v1": {"total_pnl_bps_v1": 500.0, "action_dist_v1": {}},
                "hindsight_optimal_v1": {"total_pnl_bps_v1": 1500.0, "action_dist_v1": {}},
                "always_take_v1": {"total_pnl_bps_v1": 50.0, "action_dist_v1": {}},
                "random_uniform_v1": {"total_pnl_bps_v1": 0.0, "action_dist_v1": {}},
            },
            "best_combo_v1": {"variant": "REALIZED_PNL_REWARD", "tau": 0.7, "beta": 3.0},
        }
    ]
    status, action, _, headline = entry_gate._go_no_go(per_fold)
    assert status == "ENTRY_IQL_PARTIAL_DEGENERATE_ACTION_DIST"
    assert headline["min_action_frac_v1"] == 0.0


def test_go_no_go_partial_when_ties_realized():
    per_fold = [
        {
            "iql_test_metric_v1": {
                "total_pnl_bps_v1": 150.0,
                "action_dist_v1": {"n_skip_v1": 30, "n_take_now_v1": 40, "n_wait_v1": 30, "n_total_v1": 100},
            },
            "test_baselines_v1": {
                "realized_v1": {"total_pnl_bps_v1": 100.0, "action_dist_v1": {}},
                "skipability_teacher_v1": {"total_pnl_bps_v1": 200.0, "action_dist_v1": {}},
                "hindsight_optimal_v1": {"total_pnl_bps_v1": 500.0, "action_dist_v1": {}},
                "always_take_v1": {"total_pnl_bps_v1": 50.0, "action_dist_v1": {}},
                "random_uniform_v1": {"total_pnl_bps_v1": 0.0, "action_dist_v1": {}},
            },
            "best_combo_v1": {"variant": "TEACHER_AGREE_REWARD", "tau": 0.7, "beta": 3.0},
        }
    ]
    status, action, _, _ = entry_gate._go_no_go(per_fold)
    assert status == "ENTRY_IQL_PARTIAL_TIES_REALIZED"
