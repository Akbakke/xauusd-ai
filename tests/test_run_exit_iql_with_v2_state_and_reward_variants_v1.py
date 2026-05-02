"""Tests for materialize_run_exit_iql_with_v2_state_and_reward_variants_v1.

Pure function tests: state-matrix construction, derivative computation,
ridge fit, target builder, no-shortcut audit, and go-no-go branches.
The full training pipeline is integration-tested by the actual gate run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as gate,
)


def test_reward_variants_count_is_five() -> None:
    assert len(gate.REWARD_VARIANTS_V2) == 5
    ids = [v["reward_id_v1"] for v in gate.REWARD_VARIANTS_V2]
    assert "REALIZED_PNL_REWARD" in ids
    assert "MFE_CAPTURE_REWARD" in ids
    assert "MAE_PENALTY_REWARD" in ids
    assert "GIVEBACK_PENALTY_REWARD" in ids
    assert "TRANSPARENT_COMBINED_REWARD" in ids


def test_ridge_fit_recovers_least_squares_solution() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    true_beta = np.array([1.0, -0.5, 0.25, 0.0, 2.0])
    y = X @ true_beta + rng.normal(scale=0.01, size=100)
    beta_hat = gate._ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta_hat, true_beta, atol=0.05)


def test_compute_derivatives_handles_per_trade_isolation() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A"] * 6 + ["B"] * 4,
            "bars_held_v1": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
            "running_pnl_at_close_bps_v1": [
                10.0,
                12.0,
                15.0,
                14.0,
                13.0,
                10.0,
                5.0,
                8.0,
                7.0,
                6.0,
            ],
            "running_mfe_bps_v1": [
                10.0,
                12.0,
                15.0,
                15.0,
                15.0,
                15.0,
                5.0,
                8.0,
                8.0,
                8.0,
            ],
        }
    )
    out = gate._compute_derivatives(df)
    assert out["pnl_velocity_v2"].iloc[0] == 0.0
    assert out["pnl_velocity_v2"].iloc[1] == 2.0
    assert out["pnl_velocity_v2"].iloc[6] == 0.0
    assert out["pnl_acceleration_v2"].iloc[0] == 0.0
    assert out["pnl_acceleration_v2"].iloc[1] == 0.0
    # 5-bar slope at bar 4 of trade A: (13 - 10) / 4 = 0.75
    assert pytest.approx(out["rolling_slope_pnl_5bars_v2"].iloc[4]) == 0.75
    # MFE decay rate at bar 5 of trade A: 0 (mfe never decreases)
    assert out["mfe_decay_rate_3bars_v2"].iloc[5] == 0.0
    # If mfe decreased: synthetic test
    df2 = pd.DataFrame(
        {
            "candidate_uid_v1": ["A"] * 5,
            "bars_held_v1": [0, 1, 2, 3, 4],
            "running_pnl_at_close_bps_v1": [0.0] * 5,
            "running_mfe_bps_v1": [10.0, 9.0, 8.0, 7.0, 6.0],
        }
    )
    out2 = gate._compute_derivatives(df2)
    assert out2["mfe_decay_rate_3bars_v2"].iloc[3] == pytest.approx(-1.0)  # (7-10)/3


def test_compute_targets_for_variant_pulls_terminal_pnl_for_hold() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A"] * 4 + ["B"] * 3,
            "bars_held_v1": [0, 1, 2, 3, 0, 1, 2],
            "reward_realized_pnl_reward_v1": [1.0, 2.0, 3.0, 5.0, -1.0, -2.0, -3.0],
        }
    )
    out = gate._compute_targets_for_variant(df, "reward_realized_pnl_reward_v1")
    # All rows of trade A have target_hold = 5 (terminal)
    assert (out[out["candidate_uid_v1"] == "A"]["__target_hold_v1"] == 5.0).all()
    assert (out[out["candidate_uid_v1"] == "B"]["__target_hold_v1"] == -3.0).all()
    # target_exit_now = the row's reward
    assert out["__target_exit_now_v1"].iloc[0] == 1.0
    assert out["__target_exit_now_v1"].iloc[3] == 5.0


def test_compute_targets_raises_on_missing_reward_column() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A"],
            "bars_held_v1": [0],
        }
    )
    with pytest.raises(RuntimeError, match="REWARD_COLUMN_MISSING_IN_AUGMENTED"):
        gate._compute_targets_for_variant(df, "nonexistent_column")


def test_zscore_handles_nan_via_median_imputation() -> None:
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    cfg = {"transform": "z", "mean": 2.75, "std": 1.5, "median": 2.5}
    out = gate._zscore(s, cfg)
    assert np.isfinite(out).all()
    # NaN row gets median 2.5 -> z = (2.5 - 2.75) / 1.5
    assert pytest.approx(out[3]) == (2.5 - 2.75) / 1.5


def test_passthrough_zero_one_with_sentinel_marks_nan() -> None:
    s = pd.Series([0.5, 0.7, np.nan, 1.2, -0.1])
    out = gate._passthrough_zero_one_with_sentinel(s)
    assert out[0] == 0.5
    assert out[1] == 0.7
    assert out[2] == gate.RECOVERY_SENTINEL_VALUE
    assert out[3] == 1.0  # clipped
    assert out[4] == 0.0  # clipped


def test_onehot_yields_correct_indicators() -> None:
    s = pd.Series(["EU", "US", "ASIA", "OVERLAP", "UNKNOWN"])
    out = gate._onehot(s, ["ASIA", "EU", "OVERLAP", "US"])
    assert out.shape == (5, 4)
    np.testing.assert_array_equal(out[0], [0, 1, 0, 0])  # EU
    np.testing.assert_array_equal(out[1], [0, 0, 0, 1])  # US
    np.testing.assert_array_equal(out[2], [1, 0, 0, 0])  # ASIA
    np.testing.assert_array_equal(out[3], [0, 0, 1, 0])  # OVERLAP
    np.testing.assert_array_equal(out[4], [0, 0, 0, 0])  # UNKNOWN -> all zero


def test_no_shortcut_audit_passes_on_used_columns() -> None:
    feature_names = ["intercept", "running_pnl_at_close_bps_v1__z", "session_id_v1__EU"]
    raw_columns_used = {"running_pnl_at_close_bps_v1", "session_id_v1"}
    audit = gate.audit_no_shortcut_at_training_time(feature_names, raw_columns_used)
    assert audit["status_v1"] == "PASS"


def test_no_shortcut_audit_rejects_forbidden_raw_column() -> None:
    feature_names = ["intercept"]
    raw_columns_used = {"pnl_bps", "running_pnl_at_close_bps_v1"}
    with pytest.raises(RuntimeError, match="TRAINING_USES_FORBIDDEN_FIELDS"):
        gate.audit_no_shortcut_at_training_time(feature_names, raw_columns_used)


def test_no_shortcut_audit_rejects_audit_token_in_state() -> None:
    feature_names = ["intercept", "audit_delay_better_v2__pass"]
    raw_columns_used: set[str] = set()
    with pytest.raises(RuntimeError, match="FORBIDDEN_TOKEN_IN_FEATURE_COLUMN"):
        gate.audit_no_shortcut_at_training_time(feature_names, raw_columns_used)


def test_no_shortcut_audit_rejects_post_exit_in_feature_column() -> None:
    feature_names = ["intercept", "post_exit_mfe_bps__z"]
    with pytest.raises(RuntimeError, match="FORBIDDEN_TOKEN_IN_FEATURE_COLUMN"):
        gate.audit_no_shortcut_at_training_time(feature_names, set())


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_TRAIL_STOP", "TRAIN_NOW"
        )


def _make_iql_results(test_pnl_per_variant: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    for variant, pnl in test_pnl_per_variant.items():
        rows.append(
            {
                "split_v1": "test",
                "reward_variant_v1": variant,
                "total_realized_pnl_bps_v1": pnl,
                "mean_bars_to_exit_v1": 1.5,
            }
        )
    return rows


def _baseline_per_split(realized_pnl: float, trail_stop_pnl: float) -> dict:
    return {
        "test": [
            {
                "policy_id_v1": "REALIZED_EXIT_BASELINE",
                "total_realized_pnl_bps_v1": realized_pnl,
            },
            {
                "policy_id_v1": "TRAIL_STOP_25_PCT_DD",
                "total_realized_pnl_bps_v1": trail_stop_pnl,
            },
        ]
    }


def test_go_no_go_pass_when_best_variant_beats_trail_stop() -> None:
    iql = _make_iql_results({"V1": 1100.0, "V2": 800.0})
    base = _baseline_per_split(realized_pnl=-355.0, trail_stop_pnl=1052.0)
    status, action, _, headline = gate._go_no_go(iql, base, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_TRAIL_STOP"
    assert action == "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_DEEPER_SWEEP_V1"
    assert headline["best_variant_v1"] == "V1"


def test_go_no_go_partial_when_best_beats_realized_only() -> None:
    iql = _make_iql_results({"V1": 500.0, "V2": 300.0})
    base = _baseline_per_split(realized_pnl=-355.0, trail_stop_pnl=1052.0)
    status, action, _, _ = gate._go_no_go(iql, base, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP"
    assert action == "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"


def test_go_no_go_partial_ties_realized() -> None:
    # "TIES" applies when best <= realized AND |best - realized| <= 50.
    iql = _make_iql_results({"V1": -380.0})
    base = _baseline_per_split(realized_pnl=-355.0, trail_stop_pnl=1052.0)
    status, action, _, _ = gate._go_no_go(iql, base, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_TIES_REALIZED"
    assert action == "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"


def test_go_no_go_partial_underperforms_realized() -> None:
    iql = _make_iql_results({"V1": -800.0})
    base = _baseline_per_split(realized_pnl=-355.0, trail_stop_pnl=1052.0)
    status, action, _, _ = gate._go_no_go(iql, base, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED"
    assert action == "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1"


def test_go_no_go_raises_on_empty_test_results() -> None:
    iql: list[dict] = []
    base = _baseline_per_split(realized_pnl=0.0, trail_stop_pnl=0.0)
    with pytest.raises(RuntimeError, match="IQL_TEST_RESULTS_MISSING"):
        gate._go_no_go(iql, base, v1_test_total_pnl=None)


def test_audit_split_isolation_passes_when_each_trade_in_one_split() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "A", "B", "B", "C"],
            "primary_split_v1": ["train", "train", "val", "val", "test"],
        }
    )
    audit = gate.audit_split_isolation(df)
    assert audit["status_v1"] == "PASS"


def test_audit_split_isolation_raises_when_trade_spans_multiple_splits() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "A", "B"],
            "primary_split_v1": ["train", "val", "test"],
        }
    )
    with pytest.raises(RuntimeError, match="SPLIT_ISOLATION_VIOLATION"):
        gate.audit_split_isolation(df)


def test_audit_policy_safety_at_inference_passes_for_in_range_indices() -> None:
    per_bar = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "A", "A", "B", "B"],
            "bars_held_v1": [0, 1, 2, 0, 1],
        }
    )
    exit_indices = pd.Series({"A": 1, "B": 0})
    audit = gate.audit_policy_safety_at_inference(
        per_bar, exit_indices, variant_id="TEST"
    )
    assert audit["status_v1"] == "PASS"


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_validate_no_deprecated_revival_detects_quarantine_import(tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""x"""\nimport gx1.quarantine.something\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
