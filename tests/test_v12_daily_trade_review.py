from __future__ import annotations

import math

import pytest

from tests.model_native_sizing_support import (
    unverified_learned_sizing_authority,
)
from tests.model_native_offline_rl_support import offline_rl_evidence
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)
from gx1.execution.v12_daily_trade_review import (
    INDEX_CSV,
    ModelNativeTradeReviewError,
    trade_summary_row,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _model_evidence() -> dict:
    direction_logits = [2.0, 0.2, -1.0]
    direction_probs = _softmax(direction_logits)
    public_logits = [2.0, -1.0]
    public_probs = _softmax(public_logits)
    side_logits = [1.0, -0.5]
    side_probs = _softmax(side_logits)
    side_bad_path_logits = [-1.2, 0.7]
    side_validity_logits = [1.4, -0.3]
    mtf_logits = [0.8, -0.2, -0.6]
    rail_logits = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    tf_agreement_logit = 0.4
    position_size_logit = -0.2
    path_quality_log_var = math.log(0.25)
    return {
        "decision_ts": "2026-07-08T17:55:00+00:00",
        "runtime_evidence_schema_version": MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
        "session_id": 2,
        "session": "OVERLAP",
        "entry_vol_regime_id": 2,
        "entry_vol_regime": "MEDIUM",
        "entry_atr_bucket": 2,
        "entry_spread_bucket": 1,
        "entry_h4_trend_sign_cat": 2,
        "entry_trend_regime_id": 1,
        "entry_trend_regime": "TREND_NEUTRAL",
        "decision_available_ts": "2026-07-08T18:00:00+00:00",
        "entry_signal_latency_sec": 0.0,
        "context_cutoff_ts": "2026-07-08T17:55:00+00:00",
        "context_age_m5_bars": 0,
        "raw_direction_logits": [2.09, 0.33, -1.1],
        "direction_logits": direction_logits,
        "direction_probs": direction_probs,
        "model_direction_index": 0,
        "model_direction": "LONG",
        "selected_side": 0,
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": public_probs,
        "public_trade_flat_decision_index": 0,
        "public_trade_flat_decision": "TRADE",
        "p_trade": public_probs[0],
        "p_flat_hier": public_probs[1],
        "p_long_given_trade": side_probs[0],
        "p_short_given_trade": side_probs[1],
        "side_logits": side_logits,
        "side_probs": side_probs,
        "model_native_logits": [0.5, -0.25, 0.1],
        "path_quality_raw": 1.25,
        "path_quality": 1.25,
        "path_quality_pred": 1.25,
        "path_quality_log_var": path_quality_log_var,
        "path_quality_std": math.exp(0.5 * path_quality_log_var),
        "mfe_first_n": 6.5,
        "mfe_first_n_pred": 6.5,
        "tradable_logit": _logit(0.82),
        "tradable_prob": 0.82,
        "trade_logit": 0.7,
        "bad_path_logit_raw": _logit(0.11),
        "bad_path_logit": _logit(0.11),
        "bad_path_prob": 0.11,
        "clean_edge_logit": _logit(0.76),
        "clean_edge_prob": 0.76,
        "survival_logit": _logit(0.68),
        "survival_prob": 0.68,
        "dip_pred": [0.0] * 18,
        "forecast_pred": [0.0] * 4,
        "timing_pred": [0.0] * 12,
        "tail_risk_pred": [0.0] * 6,
        "vol_forecast_pred": [0.0] * 3,
        **offline_rl_evidence(),
        "atr_bps": 12.0,
        "tf_agreement_logit": tf_agreement_logit,
        "tf_agreement_pred": _sigmoid(tf_agreement_logit),
        "position_size_logit": position_size_logit,
        "position_size_pred": _sigmoid(position_size_logit),
        "sizing_authority_contract": unverified_learned_sizing_authority(),
        "side_utility": [2.4, -0.8],
        "side_bad_path_logit": side_bad_path_logits,
        "long_bad_path_prob": _sigmoid(side_bad_path_logits[0]),
        "short_bad_path_prob": _sigmoid(side_bad_path_logits[1]),
        "side_validity_logit": side_validity_logits,
        "long_validity_prob": _sigmoid(side_validity_logits[0]),
        "short_validity_prob": _sigmoid(side_validity_logits[1]),
        "side_mae": [-3.2, -8.1],
        "mtf_dir_logits": mtf_logits,
        "mtf_dir_probs": _softmax(mtf_logits),
        "mtf_trend_evidence": 0.69,
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_gate": [0.125] * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "trendline_rail_logits": rail_logits,
        "trendline_rail_probs": [_sigmoid(value) for value in rail_logits],
        "geometry_channel_edge_pressure": 0.42,
        "geometry_rising_support_rail_long_pressure": 0.81,
        "geometry_rising_support_rail_short_trap_pressure": 0.07,
        "geometry_falling_resistance_rail_short_pressure": 0.02,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.03,
        "calibration_version": "direction-cal-v1",
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": 1.1,
        "direction_calibration_bias": [0.1, -0.1, 0.0],
        "path_calibration_enabled": True,
        "path_calibration": {
            "enabled": True,
            "version": "path-cal-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
    }


def _trade() -> dict:
    evidence = _model_evidence()
    return {
        "trade_key": "TRADE:T-MODEL-NATIVE",
        "trade_id": "T-MODEL-NATIVE",
        "entry_snapshot": {
            "trade_id": "T-MODEL-NATIVE",
            "entry_time": "2026-07-08T18:00:00+00:00",
            "instrument": "XAU_USD",
            "side": "long",
            "entry_price": 2360.12,
            "entry_bid": 2360.0,
            "entry_ask": 2360.12,
            "entry_spread_bps": 0.51,
            "atr_bps": 12.0,
            "session": "OVERLAP",
            "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
            "model_evidence": evidence,
            "execution_checks": [
                "fresh_quote",
                "spread_within_execution_cap",
                "model_native_sizing_authority",
            ],
            "units": 2,
            "applied_size_multiplier": 1.0,
        },
        "exit_summary": {
            "exit_time": "2026-07-08T18:20:00+00:00",
            "exit_price": 2365.0,
            "exit_reason": "EXIT_IQL",
            "realized_pnl_bps": 20.7,
            "max_mfe_bps": 22.0,
            "max_mae_bps": -2.0,
            "intratrade_drawdown_bps": 1.3,
        },
        "v12_bar_decisions": [
            {
                "timestamp": "2026-07-08T18:05:00+00:00",
                "bars_in_trade": 1,
                "bid": 2362.0,
                "current_pnl_bps": 7.9,
                "cum_mfe_bps": 8.1,
                "cum_mae_bps": -0.5,
                "v3_should_exit_prob": 0.2,
                "v3_consecutive_exits": 0,
                "iql_action": "HOLD",
                "iql_decision_source": "EXIT_IQL",
            },
            {
                "timestamp": "2026-07-08T18:20:00+00:00",
                "bars_in_trade": 4,
                "bid": 2365.0,
                "current_pnl_bps": 20.7,
                "cum_mfe_bps": 22.0,
                "cum_mae_bps": -0.5,
                "v3_should_exit_prob": 0.8,
                "v3_consecutive_exits": 1,
                "iql_action": "EXIT_NOW",
                "iql_decision_source": "EXIT_IQL",
            },
        ],
    }


def test_daily_trade_review_blocks_unadopted_capital_sizing_record(tmp_path) -> None:
    trade = _trade()

    with pytest.raises(ModelNativeTradeReviewError, match="sizing authority"):
        trade_summary_row(trade)
    assert not (tmp_path / "trade.md").exists()


def test_daily_trade_review_never_falls_back_to_retired_entry_score() -> None:
    trade = _trade()
    trade["entry_snapshot"].pop("model_evidence")
    trade["entry_snapshot"]["entry_score"] = {
        "v10_p_long": 0.99,
        "smart_p_long": 0.99,
        "q_take_long": 999.0,
    }

    with pytest.raises(
        ModelNativeTradeReviewError,
        match="entry_snapshot.model_evidence",
    ):
        trade_summary_row(trade)


def test_daily_trade_review_fails_closed_on_missing_specialist_evidence() -> None:
    trade = _trade()
    trade["entry_snapshot"]["model_evidence"].pop("specialist_gate")

    with pytest.raises(ModelNativeTradeReviewError, match="specialist_gate"):
        trade_summary_row(trade)


def test_daily_trade_review_uses_versioned_model_native_index_path() -> None:
    assert INDEX_CSV.name == "trade_journal_index_model_native_v1.csv"


def test_daily_trade_review_requires_complete_executable_timing() -> None:
    trade = _trade()
    for key in (
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    ):
        trade["entry_snapshot"]["model_evidence"].pop(key)

    with pytest.raises(ModelNativeTradeReviewError, match="timing"):
        trade_summary_row(trade)


@pytest.mark.parametrize(
    ("entry_key", "value", "match"),
    [
        ("entry_time", "2026-07-08T18:01:00+00:00", "entry_time"),
        ("session", "US", "entry_snapshot.session"),
        ("model_policy", "manual_override", "entry_snapshot.model_policy"),
    ],
)
def test_daily_trade_review_rejects_entry_wrapper_snapshot_mismatch(
    entry_key: str,
    value: object,
    match: str,
) -> None:
    trade = _trade()
    trade["entry_snapshot"][entry_key] = value

    with pytest.raises(ModelNativeTradeReviewError, match=match):
        trade_summary_row(trade)
