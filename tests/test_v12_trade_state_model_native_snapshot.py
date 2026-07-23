from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.model_native_sizing_support import unverified_learned_sizing_authority
from tests.model_native_offline_rl_support import offline_rl_evidence
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)
from gx1.execution.v12_trade_state import (
    M1_RETURNS_WINDOW_MAXLEN,
    PERSISTED_TRADE_STATE_SCHEMA_VERSION,
    TRAJECTORY_HISTORY_MAXLEN,
    TradeState,
    require_model_native_entry_snapshot,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.trade_overlay import compute_m1_micro_feature_arrays


def _softmax(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    exp = np.exp(array - array.max())
    return (exp / exp.sum()).tolist()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def _snapshot() -> dict:
    direction_logits = [5.0, 1.0, 0.0]
    public_logits = [5.0, 0.0]
    side_logits = [1.0, -0.5]
    side_bad_logits = [-1.2, 0.7]
    side_validity_logits = [1.4, -0.3]
    mtf_logits = [0.8, -0.2, -0.6]
    rail_logits = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    tf_logit = 0.5
    size_logit = 0.25
    return {
        "decision_ts": "2026-07-16T11:55:00+00:00",
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
        "decision_available_ts": "2026-07-16T12:00:00+00:00",
        "entry_signal_latency_sec": 0.0,
        "context_cutoff_ts": "2026-07-16T11:55:00+00:00",
        "context_age_m5_bars": 0,
        "raw_direction_logits": [5.39, 1.21, 0.0],
        "direction_logits": direction_logits,
        "direction_probs": _softmax(direction_logits),
        "model_direction_index": 0,
        "model_direction": "LONG",
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": _softmax(public_logits),
        "public_trade_flat_decision_index": 0,
        "public_trade_flat_decision": "TRADE",
        "selected_side": 0,
        "model_native_logits": [0.5, -0.25, 0.1],
        "path_quality_raw": 1.5,
        "path_quality": 1.5,
        "path_quality_pred": 1.5,
        "mfe_first_n": 12.0,
        "mfe_first_n_pred": 12.0,
        "tradable_logit": 1.0,
        "tradable_prob": _sigmoid(1.0),
        "trade_logit": 0.7,
        "bad_path_logit_raw": -1.0,
        "bad_path_logit": -1.0,
        "bad_path_prob": _sigmoid(-1.0),
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
        "p_trade": _softmax(public_logits)[0],
        "p_flat_hier": _softmax(public_logits)[1],
        "atr_bps": 9.0,
        "tf_agreement_logit": tf_logit,
        "tf_agreement_pred": _sigmoid(tf_logit),
        "path_quality_log_var": 0.0,
        "path_quality_std": 1.0,
        "position_size_logit": size_logit,
        "position_size_pred": _sigmoid(size_logit),
        "sizing_authority_contract": unverified_learned_sizing_authority(),
        "p_long_given_trade": _softmax(side_logits)[0],
        "p_short_given_trade": _softmax(side_logits)[1],
        "side_logits": side_logits,
        "side_probs": _softmax(side_logits),
        "side_utility": [2.4, -0.8],
        "side_bad_path_logit": side_bad_logits,
        "long_bad_path_prob": _sigmoid(side_bad_logits[0]),
        "short_bad_path_prob": _sigmoid(side_bad_logits[1]),
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


def _open(snapshot: dict | None = None) -> TradeState:
    return TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
        side="long",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=_snapshot() if snapshot is None else snapshot,
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )


def _valid_closed_m1_bar(
    timestamp: str = "2026-07-16T12:00:00Z",
) -> dict[str, object]:
    return {
        "m1_bar_ts": pd.Timestamp(timestamp),
        "bid": 3301.0,
        "ask": 3301.2,
        "m1_close": 3301.1,
        "bid_high": 3301.5,
        "bid_low": 3300.7,
        "ask_high": 3301.7,
        "ask_low": 3300.9,
    }


def test_trade_state_keeps_learned_sizing_as_evidence_with_fixed_hold_sentinel() -> None:
    trade = _open()

    features = trade.build_v10_entry_snapshot_features()

    assert features["v10_position_size_at_entry_v1"] == pytest.approx(
        _snapshot()["position_size_pred"]
    )
    assert features["v10_hold_horizon_at_entry_v1"] == -1.0
    assert trade.sizing_execution_evidence["mode"] == (
        "unit_normalized_research_only"
    )
    assert trade.sizing_execution_evidence["executable_order_authority"] is False


def test_executable_trade_state_has_no_default_units_or_sizing_fallback() -> None:
    with pytest.raises(TypeError, match="units"):
        TradeState.open(
            entry_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_snapshot(),
        )


def test_trade_state_rejects_entry_time_not_derived_from_snapshot_timing() -> None:
    with pytest.raises(RuntimeError, match="model-derived minute"):
        TradeState.open_unit_normalized_research(
            entry_ts=pd.Timestamp("2026-07-16T12:01:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_snapshot(),
            normalization_contract="unit_normalized_direction_exit_research_v1",
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "direction_logits",
        "path_quality",
        "tf_agreement_logit",
        "path_quality_log_var",
        "position_size_logit",
        "decision_available_ts",
        "model_policy",
        "session_id",
    ],
)
def test_trade_state_rejects_missing_model_native_entry_evidence(missing_key: str) -> None:
    snapshot = _snapshot()
    del snapshot[missing_key]

    with pytest.raises(RuntimeError, match=missing_key):
        _open(snapshot)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("direction_probs", [0.1, 0.8, 0.1]),
        ("public_trade_flat_decision_logits", [4.0, 0.0]),
        ("tf_agreement_pred", 0.01),
        ("path_quality_std", 2.0),
        ("position_size_pred", 0.01),
    ],
)
def test_trade_state_rejects_inconsistent_model_native_entry_evidence(
    key: str,
    value: object,
) -> None:
    snapshot = _snapshot()
    snapshot[key] = value

    with pytest.raises(RuntimeError, match=key):
        require_model_native_entry_snapshot(snapshot)


@pytest.mark.parametrize("key", ["hold_horizon_pred", "hold_horizon_bars_pred"])
def test_trade_state_rejects_blocked_hold_horizon_pass_through(key: str) -> None:
    snapshot = _snapshot()
    snapshot[key] = 96

    with pytest.raises(RuntimeError, match="retired fields"):
        _open(snapshot)


def test_trade_state_load_fails_closed_on_stale_snapshot(tmp_path: Path) -> None:
    trade = _open()
    payload = trade.to_dict()
    del payload["v10_snapshot"]["position_size_logit"]
    path = tmp_path / "open_trade_stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to load trade state"):
        TradeState.load(path)


def test_trade_state_update_bar_records_exact_intrabar_state() -> None:
    trade = _open()

    trade.update_bar(**_valid_closed_m1_bar())

    assert trade.bars_in_trade == 1
    assert list(trade.peak_history) == pytest.approx(
        [(3301.5 - trade.entry_ask) / trade.entry_ask * 10_000.0]
    )
    assert list(trade.trough_history) == pytest.approx(
        [(3300.7 - trade.entry_ask) / trade.entry_ask * 10_000.0]
    )
    assert list(trade.atr_bps_history) == pytest.approx(
        [(3301.7 - 3300.7) / 3301.1 * 10_000.0]
    )


def test_trade_state_update_bar_requires_all_bid_ask_ohlc_before_mutation() -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    del bar["ask_low"]

    with pytest.raises(TypeError, match="ask_low"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.peak_history


def test_trade_state_rejects_duplicate_or_gapped_m1_before_mutation() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    before = trade.to_dict()

    with pytest.raises(ValueError, match="cadence gap/duplicate"):
        trade.update_bar(**_valid_closed_m1_bar())
    assert trade.to_dict() == before

    with pytest.raises(ValueError, match="cadence gap/duplicate"):
        trade.update_bar(
            **_valid_closed_m1_bar("2026-07-16T12:02:00Z")
        )
    assert trade.to_dict() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid", np.nan),
        ("ask", np.inf),
        ("m1_close", 0.0),
        ("bid_high", -1.0),
        ("bid_low", np.nan),
        ("ask_high", np.inf),
        ("ask_low", 0.0),
    ],
)
def test_trade_state_update_bar_rejects_invalid_price_values_without_mutation(
    field: str,
    value: float,
) -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    bar[field] = value

    with pytest.raises(ValueError, match="non-finite/non-positive"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.atr_bps_history


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ask", 3301.0),
        ("bid_low", 3301.1),
        ("bid_high", 3300.9),
        ("ask_low", 3301.3),
        ("ask_high", 3301.1),
        ("ask_low", 3300.7),
        ("ask_high", 3301.5),
        ("m1_close", 3301.0),
    ],
)
def test_trade_state_update_bar_rejects_invalid_geometry_without_mutation(
    field: str,
    value: float,
) -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    bar[field] = value

    with pytest.raises(ValueError, match="OHLC geometry invalid"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.atr_bps_history


def test_trade_state_from_dict_rejects_missing_intrabar_history() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    payload = trade.to_dict()
    del payload["peak_history"]

    with pytest.raises(ValueError, match="missing exact intrabar histories"):
        TradeState.from_dict(payload)


def test_trade_state_from_dict_rejects_synthetic_or_misaligned_intrabar_history() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    payload = trade.to_dict()
    payload["atr_bps_history"] = [0.0]

    with pytest.raises(ValueError, match="intrabar histories are invalid"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["trough_history"] = []
    with pytest.raises(ValueError, match="not aligned"):
        TradeState.from_dict(payload)


def test_trade_state_zero_bar_round_trip_uses_exact_versioned_schema() -> None:
    trade = _open()

    payload = trade.to_dict()
    restored = TradeState.from_dict(json.loads(json.dumps(payload)))

    assert payload["schema_version"] == PERSISTED_TRADE_STATE_SCHEMA_VERSION
    assert restored.bars_in_trade == 0
    assert restored.entry_ts == trade.entry_ts
    assert restored.to_dict() == payload


def test_trade_state_from_dict_requires_every_serialized_field() -> None:
    payload = _open().to_dict()

    for field_name in payload:
        incomplete = json.loads(json.dumps(payload))
        del incomplete[field_name]
        with pytest.raises(ValueError):
            TradeState.from_dict(incomplete)


def test_trade_state_rejects_unexpected_or_retired_persisted_fields() -> None:
    payload = _open().to_dict()
    payload["legacy_entry_filter"] = 0.0

    with pytest.raises(ValueError, match="unexpected=.*legacy_entry_filter"):
        TradeState.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("entry_ts", "2026-07-16T14:00:00+02:00", "timezone-aware UTC"),
        ("entry_ts", "2026-07-16T12:00:00", "timezone-aware UTC"),
        ("side", "flat", "side must be"),
        ("entry_bid", 0.0, "entry_bid must be positive"),
        ("entry_ask", np.inf, "entry_ask must be finite"),
        ("current_bid", np.nan, "current_bid must be finite"),
        ("current_ask", -1.0, "current_ask must be positive"),
        ("units", 0, "units must be positive"),
        ("units", 1.0, "units must be an exact integer"),
    ],
)
def test_trade_state_rejects_invalid_identity_price_and_unit_fields(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _open().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)


def test_trade_state_rejects_side_model_direction_and_spread_mismatch() -> None:
    payload = _open().to_dict()
    payload["side"] = "short"
    with pytest.raises(ValueError, match="frozen model direction"):
        TradeState.from_dict(payload)

    payload = _open().to_dict()
    payload["entry_spread_bps"] += 0.01
    with pytest.raises(ValueError, match="does not match entry prices"):
        TradeState.from_dict(payload)


def test_trade_state_to_dict_rejects_invalid_in_memory_state() -> None:
    trade = _open()
    trade.units = 0

    with pytest.raises(ValueError, match="units must be positive"):
        trade.to_dict()


def test_trade_state_binds_bar_count_to_all_persisted_history_lengths() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    payload = trade.to_dict()
    payload["bars_in_trade"] = 2
    payload["last_processed_m1_ts"] = "2026-07-16T12:01:00+00:00"

    with pytest.raises(ValueError, match="m1_returns_window length"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["peak_history"].append(payload["peak_history"][-1])
    with pytest.raises(ValueError, match="not aligned"):
        TradeState.from_dict(payload)


def test_trade_state_accepts_exact_deque_maxlen_truncation() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    payload = trade.to_dict()
    payload["bars_in_trade"] = TRAJECTORY_HISTORY_MAXLEN + 1
    payload["last_processed_m1_ts"] = (
        pd.Timestamp("2026-07-16T12:00:00Z")
        + pd.Timedelta(minutes=TRAJECTORY_HISTORY_MAXLEN)
    ).isoformat()
    payload["m1_returns_window"] = (
        payload["m1_returns_window"] * M1_RETURNS_WINDOW_MAXLEN
    )
    for field_name in (
        "pnl_history",
        "peak_history",
        "trough_history",
        "atr_bps_history",
    ):
        payload[field_name] = payload[field_name] * TRAJECTORY_HISTORY_MAXLEN

    restored = TradeState.from_dict(payload)

    assert len(restored.m1_returns_window) == M1_RETURNS_WINDOW_MAXLEN
    assert len(restored.pnl_history) == TRAJECTORY_HISTORY_MAXLEN
    assert restored.bars_in_trade == TRAJECTORY_HISTORY_MAXLEN + 1


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("current_pnl_bps", np.inf, "current_pnl_bps must be finite"),
        ("cum_mfe_bps", np.nan, "cum_mfe_bps must be finite"),
        ("cum_mae_bps", np.inf, "cum_mae_bps must be finite"),
        ("last_atr_bps", -0.1, "last_atr_bps must be nonnegative"),
        ("bars_since_mfe_peak", -1, "bars_since_mfe_peak must be nonnegative"),
        ("v3_last_prob", np.nan, "v3_last_prob must be finite"),
        ("v3_signal_acceleration", 1.1, "V3 acceleration"),
        ("v3_total_exit_decisions", -1, "v3_total_exit_decisions must be nonnegative"),
    ],
)
def test_trade_state_rejects_invalid_running_and_v3_metrics(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _open().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)


def test_trade_state_rejects_inconsistent_v3_counter_state() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    trade.update_v3({"v3_v8_should_exit_prob": 0.8})
    payload = trade.to_dict()

    payload["v3_total_exit_decisions"] = 0
    with pytest.raises(ValueError, match="V3 counters are inconsistent"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["v3_consecutive_exits"] = 0
    with pytest.raises(ValueError, match="probability/consecutive counter mismatch"):
        TradeState.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("m1_returns_window", [np.nan], "m1_returns_window is invalid"),
        ("pnl_history", [np.inf], "pnl_history is invalid"),
        ("atr_bps_history", [0.0], "intrabar histories are invalid"),
    ],
)
def test_trade_state_rejects_nonfinite_or_nonpositive_history(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    payload = trade.to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)


def test_trade_state_rejects_intrabar_order_or_close_outside_excursion() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())

    payload = trade.to_dict()
    payload["peak_history"][0] = payload["trough_history"][0] - 0.1
    with pytest.raises(ValueError, match="intrabar histories are invalid"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["pnl_history"][0] = payload["peak_history"][0] + 0.1
    with pytest.raises(ValueError, match="intrabar histories are invalid"):
        TradeState.from_dict(payload)


def test_trade_state_m1_micro_features_match_shared_training_owner() -> None:
    trade = _open()
    closes = 2000.0 * np.exp(
        np.sin(np.arange(90, dtype=np.float64) / 7.0) * 0.001
    )
    trade.m1_returns_window.extend(closes)
    expected = compute_m1_micro_feature_arrays(
        np.asarray(trade.m1_returns_window, dtype=np.float64)
    )

    actual = trade.build_trade_state_features()

    assert actual["m1_last_5bar_return_bps_v1"] == pytest.approx(expected[0][-1])
    assert actual["m1_last_15bar_return_bps_v1"] == pytest.approx(expected[1][-1])
    assert actual["m1_last_60bar_return_bps_v1"] == pytest.approx(expected[2][-1])
    assert actual["m1_realized_vol_15bar_bps_v1"] == pytest.approx(expected[3][-1])
    assert actual["m1_realized_vol_60bar_bps_v1"] == pytest.approx(expected[4][-1])


def test_trade_state_persists_strategy_f_deferral_and_transactional_commit() -> None:
    trade = _open()
    staged = trade.clone_for_exit_decision()
    staged.update_bar(**_valid_closed_m1_bar())
    staged.update_v3({"v3_v8_should_exit_prob": 0.8})
    staged.sf_defer_state_v1["sf_first_veto_bar"] = 1
    assert staged.build_v3_overlay()["bars_held"].tolist() == [1.0]

    assert trade.bars_in_trade == 0
    assert trade.sf_defer_state_v1 == {}

    trade.commit_complete_exit_bar(staged)
    restored = TradeState.from_dict(trade.to_dict())

    assert restored.last_processed_m1_ts == pd.Timestamp(
        "2026-07-16T12:00:00Z"
    )
    assert restored.sf_defer_state_v1 == {"sf_first_veto_bar": 1}
    assert restored.v3_last_prob == pytest.approx(0.8)


def test_trade_state_load_all_requires_filename_identity_and_sorts(tmp_path: Path) -> None:
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    later_snapshot = _snapshot()
    later_snapshot.update(
        {
            "decision_ts": "2026-07-16T12:00:00+00:00",
            "decision_available_ts": "2026-07-16T12:05:00+00:00",
            "context_cutoff_ts": "2026-07-16T12:00:00+00:00",
        }
    )
    later = TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-07-16T12:05:00Z"),
        side="long",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=later_snapshot,
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    later.trade_id = "later"
    earlier = _open()
    earlier.trade_id = "earlier"
    later.save(state_dir)
    earlier.save(state_dir)

    restored = TradeState.load_all(state_dir)

    assert [trade.trade_id for trade in restored] == ["earlier", "later"]

    bad_path = state_dir / "open_trade_wrong.json"
    bad_payload = _open().to_dict()
    bad_payload["trade_id"] = "actual"
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="filename/identity mismatch"):
        TradeState.load_all(state_dir)
