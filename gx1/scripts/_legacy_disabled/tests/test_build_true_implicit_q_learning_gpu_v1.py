"""Tests for materialize_build_true_implicit_q_learning_gpu_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_build_true_implicit_q_learning_gpu_v1 as gpu_gate,
)
from gx1.scripts import materialize_build_true_implicit_q_learning_v1 as v1_gate


def test_action_constant_is_unique_to_gpu():
    assert gpu_gate.ACTION == "BUILD_TRUE_IMPLICIT_Q_LEARNING_GPU_V1"
    assert gpu_gate.ACTION != v1_gate.ACTION


def test_hyperparameters_match_v1_for_comparability():
    assert gpu_gate.TAU_GRID == v1_gate.TAU_GRID
    assert gpu_gate.BETA_GRID == v1_gate.BETA_GRID
    assert gpu_gate.GAMMA_LOCKED == v1_gate.GAMMA_LOCKED
    assert gpu_gate.K_VQ_ITERATIONS == v1_gate.K_VQ_ITERATIONS
    assert gpu_gate.ADVANTAGE_CLIP == v1_gate.ADVANTAGE_CLIP


def test_gpu_specific_knobs_have_sensible_values():
    assert gpu_gate.INNER_EPOCHS >= 5
    assert gpu_gate.HIDDEN_DIM > 0
    assert gpu_gate.N_HIDDEN >= 1
    assert 0 < gpu_gate.LR < 1.0
    assert 0 <= gpu_gate.WEIGHT_DECAY < 1.0
    assert gpu_gate.BATCH_SIZE > 0


def test_validate_final_status_rejects_v1_constants():
    """V1 status constants must NOT be allowed in GPU gate."""
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gpu_gate.validate_final_status(
            "TRUE_IQL_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
        )


def test_validate_final_status_rejects_unknown_action():
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gpu_gate.validate_final_status(
            "TRUE_IQL_GPU_PASS_MEETS_PROMOTION_CRITERIA",
            "TRAIN_NOW",
        )


def test_validate_final_status_accepts_valid_pair():
    assert gpu_gate.validate_final_status(
        "TRUE_IQL_GPU_PASS_MEETS_PROMOTION_CRITERIA",
        "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
    )


def test_evaluate_fold_gpu_handles_empty_variants(monkeypatch):
    """If reward_col missing for a variant, that combo is skipped (not error)."""
    df = pd.DataFrame({
        "primary_split_v1": ["train", "train", "val", "test"],
        "candidate_uid_v1": ["A", "A", "B", "C"],
        "bars_held_v1": [0, 1, 0, 0],
        "action_id_v1": [0, 1, 1, 1],
        "running_pnl_at_close_bps_v1": [0.0, 5.0, 3.0, -2.0],
    })
    X = np.eye(4, dtype=np.float32)
    out = gpu_gate._evaluate_fold_gpu(df, X, fold_id="FOLD_TEST")
    assert "all_evaluations_v1" in out
    assert isinstance(out["all_evaluations_v1"], list)


def test_no_deprecated_revival_passes_on_self():
    """The GPU gate must not import quarantined modules."""
    from pathlib import Path
    v1_gate.validate_no_deprecated_revival(Path(gpu_gate.__file__))


def test_go_no_go_returns_valid_status_when_overall_pass():
    promotion = {
        "n_criteria_passed_v1": 6,
        "n_criteria_total_v1": 6,
        "overall_pass_v1": True,
        "per_fold_pnl_v1": [100.0, 200.0, 300.0],
        "per_fold_realized_v1": [50.0, 50.0, 50.0],
        "per_fold_lifts_vs_realized_v1": [50.0, 150.0, 250.0],
    }
    per_fold = [
        {"fold_id_v1": "FOLD_1", "best_variant_v1": "X", "best_tau_v1": 0.7,
         "best_beta_v1": 3.0, "test_at_locked_v1": {}}
    ]
    status, action, _, headline = gpu_gate._go_no_go_gpu(per_fold, promotion)
    assert status in gpu_gate.ALLOWED_FINAL_STATUSES
    assert action in gpu_gate.ALLOWED_NEXT_ACTIONS
    assert headline["promotion_pass_v1"] is True
    assert headline["gpu_total_v1"] == 600.0
