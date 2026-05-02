"""Tests for materialize_build_advanced_offline_rl_gpu_v1."""
from __future__ import annotations

import pytest

from gx1.scripts import materialize_build_advanced_offline_rl_gpu_v1 as adv_gate


def test_action_constant_is_unique():
    assert adv_gate.ACTION == "BUILD_ADVANCED_OFFLINE_RL_GPU_V1"


def test_includes_three_algorithms():
    assert set(adv_gate.ALLOWED_ALGORITHMS) == {"IQL_GPU", "CQL_GPU", "DIST_IQL_GPU"}


def test_combo_grid_for_iql_has_no_alpha():
    grid = adv_gate._algorithm_combo_grid("IQL_GPU")
    assert all(c["cql_alpha"] == 0.0 for c in grid)
    assert len(grid) == len(adv_gate.TAU_GRID) * len(adv_gate.BETA_GRID)


def test_combo_grid_for_cql_includes_alpha_sweep():
    grid = adv_gate._algorithm_combo_grid("CQL_GPU")
    expected = (
        len(adv_gate.TAU_GRID)
        * len(adv_gate.BETA_GRID)
        * len(adv_gate.CQL_ALPHA_GRID)
    )
    assert len(grid) == expected
    assert all(c["cql_alpha"] > 0 for c in grid)


def test_combo_grid_for_distq_no_alpha():
    grid = adv_gate._algorithm_combo_grid("DIST_IQL_GPU")
    assert all(c["cql_alpha"] == 0.0 for c in grid)


def test_combo_grid_unknown_algorithm_safe_default():
    """Unknown algorithm names should be handled gracefully — return base grid
    or raise. We expect the gate to never call this with unknown names."""
    grid = adv_gate._algorithm_combo_grid("UNKNOWN")
    assert all(c["cql_alpha"] == 0.0 for c in grid)


def test_validate_final_status_rejects_v1_constants():
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        adv_gate.validate_final_status(
            "TRUE_IQL_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
        )


def test_validate_final_status_accepts_valid_pair():
    assert adv_gate.validate_final_status(
        "ADV_RL_PASS_BEST_BEATS_PROMOTION_CRITERIA",
        "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
    )


def test_train_one_unknown_algorithm_raises():
    import numpy as np
    X = np.zeros((4, 3), dtype=np.float32)
    a = np.array([0, 1, 0, 1], dtype=np.int64)
    r = np.zeros(4, dtype=np.float32)
    next_idx = np.array([1, 2, 3, -1], dtype=np.int64)
    done = np.array([False, False, False, True], dtype=bool)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        adv_gate._train_one(
            "BOGUS", X, a, r, next_idx, done, tau=0.7, cql_alpha=0.0
        )


def test_per_alg_per_fold_pnl_handles_missing_test_metric():
    fold_results = [{
        "fold_id_v1": "FOLD_1",
        "best_per_algorithm_v1": {
            "IQL_GPU": {"best_combo_v1": None, "test_at_locked_v1": None},
            "CQL_GPU": {"best_combo_v1": {}, "test_at_locked_v1": {"total_realized_pnl_bps_v1": 100.0}},
            "DIST_IQL_GPU": {"best_combo_v1": None, "test_at_locked_v1": None},
        },
    }]
    out = adv_gate._per_alg_per_fold_pnl(fold_results, "IQL_GPU")
    assert out == [0.0]
    out = adv_gate._per_alg_per_fold_pnl(fold_results, "CQL_GPU")
    assert out == [100.0]


def test_go_no_go_picks_best_algorithm():
    fold_results = [{
        "fold_id_v1": "FOLD_1",
        "best_per_algorithm_v1": {
            "IQL_GPU": {"test_at_locked_v1": {"total_realized_pnl_bps_v1": 100.0}},
            "CQL_GPU": {"test_at_locked_v1": {"total_realized_pnl_bps_v1": 300.0}},
            "DIST_IQL_GPU": {"test_at_locked_v1": {"total_realized_pnl_bps_v1": 200.0}},
        },
    }]
    status, action, _, headline = adv_gate._go_no_go(
        fold_results, realized_per_fold=[50.0], trail_stop_per_fold=[150.0],
    )
    assert headline["best_algorithm_v1"] == "CQL_GPU"
    assert headline["best_total_pnl_v1"] == 300.0
    assert status == "ADV_RL_PASS_BEST_BEATS_PROMOTION_CRITERIA"
    assert action == "BUILD_JOINT_ENTRY_EXIT_IQL_V1"
