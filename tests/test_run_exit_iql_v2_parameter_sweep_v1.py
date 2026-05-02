"""Tests for materialize_run_exit_iql_v2_parameter_sweep_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_run_exit_iql_v2_parameter_sweep_v1 as gate
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)


# ---------------------------------------------------------------------------
# Sweep grid shape
# ---------------------------------------------------------------------------


def test_sweep_grid_has_exactly_ten_configs() -> None:
    assert len(gate.SWEEP_CONFIGS) == 10


def test_every_config_has_required_keys() -> None:
    required = {
        "config_id_v1",
        "reward_id_v1",
        "ridge_lambda_v1",
        "state_subset_v1",
    }
    for c in gate.SWEEP_CONFIGS:
        assert required <= set(c.keys()), f"missing keys on {c}"


def test_config_ids_are_unique() -> None:
    ids = [c["config_id_v1"] for c in gate.SWEEP_CONFIGS]
    assert len(set(ids)) == len(ids)


def test_validate_sweep_grid_passes_on_built_in_configs() -> None:
    audit = gate.validate_sweep_grid(gate.SWEEP_CONFIGS)
    assert audit["status_v1"] == "PASS"
    assert audit["config_count_v1"] == 10


def test_validate_sweep_grid_rejects_wrong_count() -> None:
    bad = list(gate.SWEEP_CONFIGS)[:5]
    with pytest.raises(RuntimeError, match="SWEEP_CONFIG_COUNT_MISMATCH"):
        gate.validate_sweep_grid(bad)


def test_validate_sweep_grid_rejects_duplicate_id() -> None:
    bad = list(gate.SWEEP_CONFIGS)
    bad.append({**bad[0]})  # duplicate of first config
    bad = bad[:10]  # keep count at 10 by trimming
    bad[1] = {**bad[0]}  # force duplicate id
    with pytest.raises(RuntimeError, match="DUPLICATE_CONFIG_ID_V1"):
        gate.validate_sweep_grid(bad)


def test_validate_sweep_grid_rejects_unknown_reward() -> None:
    bad = [
        {**c, "reward_id_v1": "MADE_UP_REWARD"} if i == 0 else c
        for i, c in enumerate(gate.SWEEP_CONFIGS)
    ]
    with pytest.raises(RuntimeError, match="UNKNOWN_REWARD_ID_V1"):
        gate.validate_sweep_grid(bad)


def test_validate_sweep_grid_rejects_unknown_state_subset() -> None:
    bad = [
        {**c, "state_subset_v1": "XYZ"} if i == 0 else c
        for i, c in enumerate(gate.SWEEP_CONFIGS)
    ]
    with pytest.raises(RuntimeError, match="UNKNOWN_STATE_SUBSET_V1"):
        gate.validate_sweep_grid(bad)


def test_validate_sweep_grid_rejects_non_positive_lambda() -> None:
    bad = [
        {**c, "ridge_lambda_v1": 0.0} if i == 0 else c
        for i, c in enumerate(gate.SWEEP_CONFIGS)
    ]
    with pytest.raises(RuntimeError, match="NON_POSITIVE_RIDGE_LAMBDA"):
        gate.validate_sweep_grid(bad)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def _toy_state_full() -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "intercept",
        "running_pnl_at_close_bps_v1__z",
        "pnl_velocity_v2__z",
        "mfe_decay_rate_3bars_v2__z",
        "p_long_entry_v1__pass_or_sentinel",
        "margin_entry_v1__pass_or_sentinel",
        "atr_bps_now_v1__z",
    ]
    X = np.zeros((4, len(feature_names)))
    return X, feature_names


def test_state_matrix_for_subset_full_returns_unchanged() -> None:
    X, names = _toy_state_full()
    X_sub, names_sub = gate._state_matrix_for_subset(X, names, "FULL")
    assert names_sub == names
    assert X_sub.shape == X.shape


def test_state_matrix_for_subset_no_derivatives_drops_v2_derivatives() -> None:
    X, names = _toy_state_full()
    X_sub, names_sub = gate._state_matrix_for_subset(X, names, "NO_DERIVATIVES")
    assert "pnl_velocity_v2__z" not in names_sub
    assert "mfe_decay_rate_3bars_v2__z" not in names_sub
    assert "p_long_entry_v1__pass_or_sentinel" in names_sub
    assert X_sub.shape[1] == X.shape[1] - 2


def test_state_matrix_for_subset_no_recovery_drops_recovery_fields() -> None:
    X, names = _toy_state_full()
    X_sub, names_sub = gate._state_matrix_for_subset(X, names, "NO_RECOVERY")
    assert "p_long_entry_v1__pass_or_sentinel" not in names_sub
    assert "margin_entry_v1__pass_or_sentinel" not in names_sub
    assert "pnl_velocity_v2__z" in names_sub
    assert X_sub.shape[1] == X.shape[1] - 2


def test_state_matrix_for_subset_rejects_unknown_subset() -> None:
    X, names = _toy_state_full()
    with pytest.raises(RuntimeError, match="UNKNOWN_STATE_SUBSET"):
        gate._state_matrix_for_subset(X, names, "MADE_UP")


# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------


def _mk_metric(reward, lam, subset, pnl) -> dict:
    return {
        "split_v1": "test",
        "reward_variant_v1": reward,
        "ridge_lambda_v1": lam,
        "state_subset_v1": subset,
        "total_realized_pnl_bps_v1": pnl,
    }


def test_ridge_lambda_sensitivity_groups_by_reward_and_subset() -> None:
    rows = [
        _mk_metric("R1", 1e-3, "FULL", 100.0),
        _mk_metric("R1", 1e-2, "FULL", 110.0),
        _mk_metric("R1", 1e-1, "FULL", 90.0),
        _mk_metric("R2", 1e-3, "FULL", 50.0),  # only one lambda, should be omitted
    ]
    out = gate._ridge_lambda_sensitivity(rows)
    assert len(out) == 1
    assert out[0]["reward_variant_v1"] == "R1"
    assert out[0]["best_lambda_v1"] == 1e-2
    assert out[0]["spread_test_pnl_v1"] == 20.0


def test_state_subset_ablation_groups_by_reward_and_lambda() -> None:
    rows = [
        _mk_metric("R1", 1e-3, "FULL", 500.0),
        _mk_metric("R1", 1e-3, "NO_DERIVATIVES", 460.0),
        _mk_metric("R1", 1e-3, "NO_RECOVERY", 380.0),
    ]
    out = gate._state_subset_ablation(rows)
    assert len(out) == 1
    assert out[0]["best_subset_v1"] == "FULL"
    assert out[0]["spread_test_pnl_v1"] == 120.0


def test_reward_variant_sensitivity_groups_by_lambda_and_subset() -> None:
    rows = [
        _mk_metric("R_GIVEBACK", 1e-3, "FULL", 500.0),
        _mk_metric("R_MAE", 1e-3, "FULL", 380.0),
        _mk_metric("R_REALIZED", 1e-3, "FULL", 170.0),
    ]
    out = gate._reward_variant_sensitivity(rows)
    assert len(out) == 1
    assert out[0]["best_reward_v1"] == "R_GIVEBACK"
    assert out[0]["spread_test_pnl_v1"] == 330.0


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _baseline(realized: float, trail_stop: float) -> dict:
    return {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": realized},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": trail_stop},
        ]
    }


def test_go_no_go_pass_when_best_beats_trail_stop() -> None:
    metrics = [
        {**_mk_metric("R", 1e-3, "FULL", 1100.0), "config_id_v1": "C01", "mean_bars_to_exit_v1": 1.0}
    ]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, action, _, headline = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0)
    assert status == "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_TRAIL_STOP"
    assert action == "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1"
    assert headline["best_test_pnl_v1"] == 1100.0


def test_go_no_go_partial_when_best_beats_realized_only() -> None:
    metrics = [
        {**_mk_metric("R", 1e-3, "FULL", 509.0), "config_id_v1": "C01", "mean_bars_to_exit_v1": 0.5}
    ]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, action, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0)
    assert status == "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_REALIZED_NOT_TRAIL_STOP"
    assert action == "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"


def test_go_no_go_partial_ties() -> None:
    metrics = [
        {**_mk_metric("R", 1e-3, "FULL", -380.0), "config_id_v1": "C01", "mean_bars_to_exit_v1": 0.5}
    ]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, _, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0)
    assert status == "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_TIES_REALIZED"


def test_go_no_go_partial_underperforms() -> None:
    metrics = [
        {**_mk_metric("R", 1e-3, "FULL", -800.0), "config_id_v1": "C01", "mean_bars_to_exit_v1": 0.5}
    ]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, _, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0)
    assert status == "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_UNDERPERFORMS_REALIZED"


def test_go_no_go_raises_on_empty_metrics() -> None:
    with pytest.raises(RuntimeError, match="SWEEP_TEST_METRICS_MISSING"):
        gate._go_no_go([], _baseline(0.0, 0.0), None)


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_TRAIL_STOP", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


# ---------------------------------------------------------------------------
# Cross-module re-use sanity
# ---------------------------------------------------------------------------


def test_sweep_uses_same_reward_variants_as_v2_train_gate() -> None:
    sweep_rewards = {c["reward_id_v1"] for c in gate.SWEEP_CONFIGS}
    v2_rewards = {v["reward_id_v1"] for v in v2_train_gate.REWARD_VARIANTS_V2}
    assert sweep_rewards <= v2_rewards


def test_state_subset_codes_are_known() -> None:
    subsets = {c["state_subset_v1"] for c in gate.SWEEP_CONFIGS}
    assert subsets <= gate.ALLOWED_STATE_SUBSETS
