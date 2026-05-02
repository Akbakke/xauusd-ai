"""Tests for joint_entry_exit_iql_v1 — combined skip+exit policy evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import joint_entry_exit_iql_v1 as joint


def _toy_per_bar():
    """3 trades, varying length and pnl trajectories."""
    return pd.DataFrame({
        "candidate_uid_v1": [
            "A", "A", "A",        # 3 bars, peak at bar 1
            "B", "B",             # 2 bars, peak at bar 1
            "C", "C", "C", "C",   # 4 bars, peak at bar 2
        ],
        "bars_held_v1": [0, 1, 2, 0, 1, 0, 1, 2, 3],
        "running_pnl_at_close_bps_v1": [
            10.0, 20.0, 5.0,
            -5.0, 15.0,
            8.0, 25.0, 30.0, 22.0,
        ],
    })


def _toy_entry_features():
    return pd.DataFrame({
        "candidate_uid_v1": ["A", "B", "C"],
        "f1": [0.1, 0.5, 0.9],
        "f2": [1.0, 2.0, 3.0],
    })


def test_skip_all_yields_zero_joint_pnl():
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    skip_all = lambda X: np.ones(X.shape[0])
    exit_never = lambda X: np.zeros(X.shape[0])
    res = joint.evaluate_joint_policy(
        per_bar, entry, exit_X, skip_all, exit_never,
        skip_threshold=0.5, exit_threshold=0.5,
    )
    assert res.n_taken_v1 == 0
    assert res.n_skipped_v1 == 3
    assert res.skip_rate_v1 == 1.0
    assert res.joint_pnl_total_bps_v1 == 0.0
    assert all(s == "SKIP" for s in res.per_candidate_status_v1)


def test_take_all_with_exit_at_first_bar_uses_first_pnl():
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    take_all = lambda X: np.zeros(X.shape[0])
    exit_immediate = lambda X: np.ones(X.shape[0])
    res = joint.evaluate_joint_policy(
        per_bar, entry, exit_X, take_all, exit_immediate,
        skip_threshold=0.5, exit_threshold=0.5,
    )
    # Exit at first bar → A=10, B=-5, C=8
    assert res.n_taken_v1 == 3
    assert res.joint_pnl_total_bps_v1 == 10.0 - 5.0 + 8.0


def test_take_all_exit_never_falls_through_to_last_bar():
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    take_all = lambda X: np.zeros(X.shape[0])
    exit_never = lambda X: np.zeros(X.shape[0])
    res = joint.evaluate_joint_policy(
        per_bar, entry, exit_X, take_all, exit_never,
        skip_threshold=0.5, exit_threshold=0.5,
    )
    # Last bar → A=5, B=15, C=22
    assert res.joint_pnl_total_bps_v1 == 5.0 + 15.0 + 22.0


def test_skip_filters_out_negative_trade():
    """Skip trade B (the only one with negative running pnl at first bar)."""
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    # Skip only B (uid 'B' is index 1, f1=0.5).
    def skip_b_only(X):
        return np.array([0.0, 1.0, 0.0])  # B is skipped, A & C taken
    exit_immediate = lambda X: np.ones(X.shape[0])
    res = joint.evaluate_joint_policy(
        per_bar, entry, exit_X, skip_b_only, exit_immediate,
        skip_threshold=0.5, exit_threshold=0.5,
    )
    assert res.n_taken_v1 == 2
    assert res.n_skipped_v1 == 1
    # A=10, C=8; B skipped; B's realized_pnl_at_trade_end was 15 → skip floor = 15
    assert res.joint_pnl_total_bps_v1 == 10.0 + 8.0
    assert res.skip_floor_total_bps_v1 == 15.0


def test_realized_pnl_total_unaffected_by_policy_choices():
    """realized_pnl_total reflects what the trade ended at, regardless of skip/exit."""
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    res = joint.evaluate_joint_policy(
        per_bar, entry, exit_X,
        skip_policy=lambda X: np.zeros(X.shape[0]),
        exit_policy=lambda X: np.zeros(X.shape[0]),
    )
    # Realized = last bar of each trade: A=5, B=15, C=22.
    assert res.realized_pnl_total_bps_v1 == 5.0 + 15.0 + 22.0


def test_evaluate_empty_per_bar_returns_zero_result():
    res = joint.evaluate_joint_policy(
        pd.DataFrame(columns=["candidate_uid_v1", "bars_held_v1", "running_pnl_at_close_bps_v1"]),
        pd.DataFrame(columns=["candidate_uid_v1", "f1"]),
        np.zeros((0, 1), dtype=np.float32),
        skip_policy=lambda X: np.zeros(X.shape[0]),
        exit_policy=lambda X: np.zeros(X.shape[0]),
    )
    assert res.n_candidates_v1 == 0
    assert res.joint_pnl_total_bps_v1 == 0.0


def test_skip_policy_shape_mismatch_raises():
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    bad_skip = lambda X: np.array([0.5])  # wrong shape
    with pytest.raises(ValueError, match="skip_policy must return shape"):
        joint.evaluate_joint_policy(
            per_bar, entry, exit_X, bad_skip,
            exit_policy=lambda X: np.zeros(X.shape[0]),
        )


def test_exit_policy_shape_mismatch_raises():
    per_bar = _toy_per_bar()
    entry = _toy_entry_features()
    exit_X = np.zeros((len(per_bar), 3), dtype=np.float32)
    bad_exit = lambda X: np.array([0.5])  # wrong shape
    with pytest.raises(ValueError, match="exit_policy must return shape"):
        joint.evaluate_joint_policy(
            per_bar, entry, exit_X,
            skip_policy=lambda X: np.zeros(X.shape[0]),
            exit_policy=bad_exit,
        )


def test_diagnose_trade_timing_categorizes_correctly():
    per_bar = _toy_per_bar()
    # A peaks at bar 1 (pnl=20). Choose bar 0 → TOO_EARLY.
    # B peaks at bar 1 (pnl=15). Choose bar 1 → ON_TIME.
    # C peaks at bar 2 (pnl=30). Choose bar 3 → TOO_LATE.
    chosen = {"A": 0, "B": 1, "C": 3}
    out = joint.diagnose_trade_timing(per_bar, chosen)
    out_by_uid = out.set_index("candidate_uid_v1")
    assert out_by_uid.loc["A", "timing_label_v1"] == "TOO_EARLY"
    assert out_by_uid.loc["B", "timing_label_v1"] == "ON_TIME"
    assert out_by_uid.loc["C", "timing_label_v1"] == "TOO_LATE"
    assert out_by_uid.loc["A", "regret_bps_v1"] == 10.0  # 20 - 10
    assert out_by_uid.loc["B", "regret_bps_v1"] == 0.0
    assert out_by_uid.loc["C", "regret_bps_v1"] == 8.0   # 30 - 22


def test_first_exit_index_threshold_logic():
    p = np.array([0.1, 0.4, 0.6, 0.3])
    assert joint._first_exit_index(p, threshold=0.5) == 2
    assert joint._first_exit_index(p, threshold=0.7) == -1
    assert joint._first_exit_index(p, threshold=0.0) == 0
