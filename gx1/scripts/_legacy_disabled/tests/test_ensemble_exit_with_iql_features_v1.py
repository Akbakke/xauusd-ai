"""Tests for materialize_ensemble_exit_with_iql_features_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.scripts import (
    materialize_ensemble_exit_with_iql_features_v1 as ens_gate,
)


def test_action_constant():
    assert ens_gate.ACTION == "ENSEMBLE_EXIT_WITH_IQL_FEATURES_V1"


def test_meta_feature_cols_includes_v3_and_iql():
    cols = ens_gate.META_FEATURE_COLS
    assert "v3_p_exit_v1" in cols
    assert "iql_p_exit_v1" in cols
    assert "iql_advantage_v1" in cols


def test_validate_final_status_accepts_known():
    assert ens_gate.validate_final_status(
        "ENSEMBLE_PASS_BEATS_V3_ALONE",
        "DISTILL_V3_AS_TARGET_FROM_IQL_LABELS_V1",
    )


def test_validate_final_status_rejects_unknown():
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        ens_gate.validate_final_status("BOGUS", "FULL_BUDGET_IQL_RETRAIN_V1")


def test_better_than_realized_label_correctness():
    """Each bar where running_pnl > final_pnl gets label 1."""
    df = pd.DataFrame({
        "candidate_uid_v1": ["A", "A", "A", "B", "B"],
        "bars_held_v1": [0, 1, 2, 0, 1],
        # A's final pnl = 5; bars 0 (10) > 5 → 1, bar 1 (20) > 5 → 1, bar 2 (5) == 5 → 0
        # B's final pnl = -2; bar 0 (3) > -2 → 1, bar 1 (-2) == -2 → 0
        "running_pnl_at_close_bps_v1": [10.0, 20.0, 5.0, 3.0, -2.0],
    })
    label = ens_gate._build_better_than_realized_label(df)
    assert list(label) == [1, 1, 0, 1, 0]


def test_train_meta_policy_runs_on_cpu_when_no_cuda():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.int64)
    # Force training to run by calling helper; smoke test only
    model = ens_gate._train_meta_policy_torch(X, y)
    p = ens_gate._predict_meta(model, X)
    assert p.shape == (100,)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_exit_index_threshold_first_trigger_returned():
    df = pd.DataFrame({
        "candidate_uid_v1": ["X", "X", "X", "Y", "Y"],
        "bars_held_v1": [0, 1, 2, 0, 1],
        "running_pnl_at_close_bps_v1": [0, 0, 0, 0, 0],
        "exit_label_det_v1": ["KEEP", "KEEP", "EXIT", "KEEP", "EXIT"],
        "ts_v1": pd.to_datetime(["2025-01-01"] * 5, utc=True),
    }).reset_index(drop=True)
    p_exit = np.array([0.1, 0.4, 0.6, 0.3, 0.8])
    out = ens_gate._exit_index_threshold(df, p_exit, threshold=0.5)
    # X: first idx where p>=0.5 is idx 2 (within full df it's df row 2)
    # Y: first idx where p>=0.5 is idx 4 (within full df it's df row 4)
    assert out["X"] == 2
    assert out["Y"] == 4


def test_go_no_go_pass_when_meta_beats_v3():
    per_fold_results = [
        {
            "fold_id_v1": "FOLD_1",
            "test": {
                "meta_best_total_pnl_v1": 500.0,
                "v3_alone_best_total_pnl_v1": 200.0,
                "iql_alone_best_total_pnl_v1": 100.0,
                "meta_minus_v3_v1": 300.0,
            },
            "val": {},
        }
    ]
    status, action, _, headline = ens_gate._go_no_go(per_fold_results)
    assert status == "ENSEMBLE_PASS_BEATS_V3_ALONE"
    assert headline["delta_meta_minus_v3_v1"] == 300.0


def test_go_no_go_partial_when_ties():
    per_fold_results = [
        {
            "fold_id_v1": "FOLD_1",
            "test": {
                "meta_best_total_pnl_v1": 220.0,
                "v3_alone_best_total_pnl_v1": 200.0,
                "iql_alone_best_total_pnl_v1": 100.0,
                "meta_minus_v3_v1": 20.0,
            },
            "val": {},
        }
    ]
    status, action, _, _ = ens_gate._go_no_go(per_fold_results)
    assert status == "ENSEMBLE_PARTIAL_TIES_V3_ALONE"
    assert action == "FULL_BUDGET_IQL_RETRAIN_V1"


def test_go_no_go_partial_when_degrades():
    per_fold_results = [
        {
            "fold_id_v1": "FOLD_1",
            "test": {
                "meta_best_total_pnl_v1": -100.0,
                "v3_alone_best_total_pnl_v1": 500.0,
                "iql_alone_best_total_pnl_v1": -200.0,
                "meta_minus_v3_v1": -600.0,
            },
            "val": {},
        }
    ]
    status, action, _, _ = ens_gate._go_no_go(per_fold_results)
    assert status == "ENSEMBLE_PARTIAL_DEGRADES_VS_V3_ALONE"
