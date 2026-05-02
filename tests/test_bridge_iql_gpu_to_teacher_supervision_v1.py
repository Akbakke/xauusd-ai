"""Tests for materialize_bridge_iql_gpu_to_teacher_supervision_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.scripts import (
    materialize_bridge_iql_gpu_to_teacher_supervision_v1 as bridge_gate,
)
from gx1.scripts import true_iql_gpu_core_v1 as iql_core


def test_action_constant_unique():
    assert bridge_gate.ACTION == "BRIDGE_IQL_GPU_TO_TEACHER_SUPERVISION_V1"


def test_locked_hyperparameters_match_advanced_rl_best():
    """The locked hyperparams should reflect advanced-RL gate's empirical best."""
    assert bridge_gate.LOCKED_TAU == 0.7
    assert bridge_gate.LOCKED_BETA == 3.0
    assert bridge_gate.LOCKED_VARIANT == "GIVEBACK_PENALTY_REWARD"


def test_validate_final_status_rejects_unknown():
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        bridge_gate.validate_final_status("MADE_UP", "DISTILL_V3_EXIT_TRANSFORMER_FROM_IQL_LABELS_V1")


def test_validate_next_action_rejects_unknown():
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        bridge_gate.validate_final_status("BRIDGE_PASS_LABELS_WRITTEN", "TRAIN_NOW")


def test_validate_final_status_accepts_known_pair():
    assert bridge_gate.validate_final_status(
        "BRIDGE_PASS_LABELS_WRITTEN",
        "DISTILL_V3_EXIT_TRANSFORMER_FROM_IQL_LABELS_V1",
    )


def test_attach_iql_recommendations_adds_expected_columns():
    """Smoke test: attach_iql_recommendations adds the right columns."""
    df = pd.DataFrame({
        "candidate_uid": ["A", "B", "C", "D"],
        "hindsight_decision_anchor_timestamp_utc_v1": pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"], utc=True
        ),
        "hindsight_policy_action_v1": ["HOLD", "EXIT_NOW", "HOLD", "EXIT_NOW"],
        "hindsight_policy_action_domain_v1": ["MANAGEMENT"] * 4,
        "used_for_training": [True, True, False, False],
        "used_for_validation": [False, False, True, False],
        "used_for_holdout": [False, False, False, True],
    })
    state_dim = 5
    X = np.random.default_rng(7).normal(size=(4, state_dim)).astype(np.float32)
    valid = np.array([True, True, False, True], dtype=bool)
    # Build a tiny model
    a = np.array([0, 1, 0, 1], dtype=np.int64)
    r = np.zeros(4, dtype=np.float32)
    next_idx = np.array([1, 2, 3, -1], dtype=np.int64)
    done = np.array([False, False, False, True], dtype=bool)
    model = iql_core.train_true_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, k_iterations=1, inner_epochs=3, prefer_cuda=False,
    )
    out = bridge_gate._attach_iql_recommendations(df, model, X, valid)
    for col in [
        "iql_gpu_p_exit_v1", "iql_gpu_q_hold_v1", "iql_gpu_q_exit_v1",
        "iql_gpu_advantage_v1", "iql_gpu_recommended_action_v1",
        "iql_gpu_state_resolved_v1", "iql_gpu_agreement_with_teacher_v1",
    ]:
        assert col in out.columns, f"missing column: {col}"
    # Unresolved rows must have NaN p_exit and recommendation NOT_ESTABLISHED
    assert np.isnan(out.iloc[2]["iql_gpu_p_exit_v1"])
    assert out.iloc[2]["iql_gpu_recommended_action_v1"] == "NOT_ESTABLISHED"
    assert out.iloc[2]["iql_gpu_state_resolved_v1"] == False  # noqa: E712
    # Resolved rows must have a defined recommendation
    for i in [0, 1, 3]:
        assert out.iloc[i]["iql_gpu_recommended_action_v1"] in {"HOLD", "EXIT_NOW"}


def test_recommendation_threshold_at_half():
    """p_exit >= 0.5 → EXIT_NOW; < 0.5 → HOLD."""
    df = pd.DataFrame({
        "candidate_uid": ["X"],
        "hindsight_decision_anchor_timestamp_utc_v1": pd.to_datetime(["2025-01-01"], utc=True),
        "hindsight_policy_action_v1": ["HOLD"],
        "hindsight_policy_action_domain_v1": ["MANAGEMENT"],
        "used_for_training": [False],
        "used_for_validation": [False],
        "used_for_holdout": [True],
    })
    X = np.zeros((1, 3), dtype=np.float32)
    valid = np.array([True], dtype=bool)
    # Make a trivial model that won't have meaningful Q values
    a = np.array([0], dtype=np.int64)
    r = np.array([0.0], dtype=np.float32)
    next_idx = np.array([-1], dtype=np.int64)
    done = np.array([True], dtype=bool)
    model = iql_core.train_true_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, k_iterations=1, inner_epochs=2, prefer_cuda=False,
    )
    out = bridge_gate._attach_iql_recommendations(df, model, X, valid)
    p = out.iloc[0]["iql_gpu_p_exit_v1"]
    assert 0.0 <= p <= 1.0
    expected_action = "EXIT_NOW" if p >= 0.5 else "HOLD"
    assert out.iloc[0]["iql_gpu_recommended_action_v1"] == expected_action


def test_agreement_diagnostics_computes_per_split():
    df = pd.DataFrame({
        "hindsight_policy_action_v1": ["HOLD", "EXIT_NOW", "HOLD", "EXIT_NOW"],
        "iql_gpu_recommended_action_v1": ["HOLD", "EXIT_NOW", "EXIT_NOW", "EXIT_NOW"],
        "iql_gpu_agreement_with_teacher_v1": [True, True, False, True],
        "iql_gpu_state_resolved_v1": [True, True, True, True],
        "hindsight_policy_action_domain_v1": ["MANAGEMENT"] * 4,
        "used_for_training": [True, False, False, False],
        "used_for_validation": [False, True, False, False],
        "used_for_holdout": [False, False, True, True],
    })
    diag = bridge_gate._agreement_diagnostics(df)
    assert diag["train"]["n_management_total_v1"] == 1
    assert diag["train"]["agreement_rate_v1"] == 1.0
    assert diag["holdout"]["n_management_total_v1"] == 2
    assert diag["holdout"]["n_agree_v1"] == 1
    assert diag["holdout"]["agreement_rate_v1"] == 0.5


def test_go_no_go_pass_when_n_mgmt_positive():
    diag = {
        "train": {"agreement_rate_v1": 0.7},
        "val": {"agreement_rate_v1": 0.65},
        "holdout": {"agreement_rate_v1": 0.6},
    }
    status, action, msg = bridge_gate._go_no_go(diag, n_mgmt=100)
    assert status == "BRIDGE_PASS_LABELS_WRITTEN"
    assert action == "DISTILL_V3_EXIT_TRANSFORMER_FROM_IQL_LABELS_V1"


def test_go_no_go_partial_when_no_management_rows():
    diag = {
        "train": {"agreement_rate_v1": 0.0},
        "val": {"agreement_rate_v1": 0.0},
        "holdout": {"agreement_rate_v1": 0.0},
    }
    status, action, msg = bridge_gate._go_no_go(diag, n_mgmt=0)
    assert status == "BRIDGE_PARTIAL_NO_MANAGEMENT_ROWS"
    assert action == "REPAIR_BRIDGE_BEFORE_FURTHER_WORK_V1"
