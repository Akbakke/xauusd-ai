"""Tests for materialize_run_exit_iql_v3_with_per_bar_xgb_transformer_signal_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_run_exit_iql_v3_with_per_bar_xgb_transformer_signal_v1 as gate,
)


def test_v3_reuses_v2_reward_variants() -> None:
    from gx1.scripts import (
        materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
    )

    assert gate.REWARD_VARIANTS_V3 == list(v2_train_gate.REWARD_VARIANTS_V2)


def test_per_bar_xgb_fields_match_replay_output_columns() -> None:
    from gx1.scripts import (
        materialize_run_per_bar_xgb_replay_for_transformer_signal_family_v1 as replay_gate,
    )

    assert set(gate.PER_BAR_XGB_FIELDS) == set(replay_gate.PER_BAR_XGB_OUTPUT_COLUMNS)


def test_per_bar_xgb_passthrough_handles_nan_via_sentinel() -> None:
    s = pd.Series([0.5, 0.7, np.nan, 0.3, 1.5])
    out = gate._per_bar_xgb_passthrough_or_sentinel(s)
    assert out[0] == 0.5
    assert out[1] == 0.7
    assert out[2] == gate.PER_BAR_XGB_SENTINEL_VALUE
    assert out[3] == 0.3
    # 1.5 is within [-2, 5] generous clip (entropy may exceed 1).
    assert out[4] == 1.5


def test_per_bar_xgb_passthrough_clips_extreme_outliers() -> None:
    s = pd.Series([10.0, -10.0])
    out = gate._per_bar_xgb_passthrough_or_sentinel(s)
    assert out[0] == 5.0  # clip upper
    assert out[1] == -2.0  # clip lower


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_TRAIL_STOP", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def _mk_test_metric(reward, pnl, bars=0.5):
    return {
        "split_v1": "test",
        "reward_variant_v1": reward,
        "total_realized_pnl_bps_v1": pnl,
        "mean_bars_to_exit_v1": bars,
    }


def _baseline(realized, trail_stop):
    return {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": realized},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": trail_stop},
        ]
    }


def test_go_no_go_pass_when_best_beats_trail_stop() -> None:
    metrics = [_mk_test_metric("GIVEBACK", 1100.0)]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, action, _, headline = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_TRAIL_STOP"
    assert action == "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1"


def test_go_no_go_pass_lift_v2_when_above_realized_and_lifts_v2() -> None:
    metrics = [_mk_test_metric("GIVEBACK", 700.0)]  # > V2 baseline 509 by 191
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, action, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_LIFTS_V2"


def test_go_no_go_pass_ties_v2_when_above_realized_and_close_to_v2() -> None:
    metrics = [_mk_test_metric("GIVEBACK", 510.0)]  # ~= V2 baseline 509
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, action, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_TIES_V2"


def test_go_no_go_partial_when_ties_realized() -> None:
    metrics = [_mk_test_metric("X", -380.0)]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, _, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_TIES_REALIZED"


def test_go_no_go_partial_when_underperforms_realized() -> None:
    metrics = [_mk_test_metric("X", -800.0)]
    base = _baseline(realized=-355.0, trail_stop=1052.0)
    status, _, _, _ = gate._go_no_go(metrics, base, v2_baseline_test_pnl=509.0, v1_test_total_pnl=250.0)
    assert status == "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED"


def test_go_no_go_raises_on_empty_metrics() -> None:
    with pytest.raises(RuntimeError, match="V3_TEST_RESULTS_MISSING"):
        gate._go_no_go([], _baseline(0.0, 0.0), None, None)


def test_audit_no_shortcut_v3_inherits_v2_logic() -> None:
    audit = gate.audit_no_shortcut_at_training_time_v3(
        ["intercept", "running_pnl_at_close_bps_v1__z", "per_bar_xgb_p_long_v2__pass_or_sentinel"],
        {"running_pnl_at_close_bps_v1", "per_bar_xgb_p_long_v2"},
    )
    assert audit["status_v1"] == "PASS"
    assert audit["audit_id_v1"] == "TRAINING_NO_SHORTCUT_AUDIT_V3"


def test_audit_no_shortcut_v3_rejects_forbidden_pattern() -> None:
    with pytest.raises(RuntimeError):
        gate.audit_no_shortcut_at_training_time_v3(
            ["intercept", "post_exit_drift_v1"], {"post_exit_drift_v1"}
        )
