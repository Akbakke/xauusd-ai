from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CAT_FIELDS, MODEL_NATIVE_CTX_CONT_FIELDS
from gx1.features.entry_model_native_feature_layers_v1 import (
    build_candlestick_derived_layer,
    build_chart_layer,
    build_deep_interaction_layer,
    build_price_derived_layer,
)


def _valid_inputs(tmp_path: Path, *, rows: int = 240):
    names = [f"snap.{name}" for name in MODEL_NATIVE_BASE_FIELDS]
    names.extend(f"ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS)
    names.extend(f"ctx_cat.{name}" for name in MODEL_NATIVE_CTX_CAT_FIELDS)
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(len(names), dtype=np.float64)[None, :]
    matrix = (0.5 + 0.4 * np.sin(row * 0.071 + column * 0.137)).astype(np.float32)
    # EMA200 plus its two causal spread derivatives need 202 closed source
    # bars before the first emitted sample.  Warmup belongs to the source
    # history, not to zero-filled model rows.
    warmup_rows = 202
    source_rows = warmup_rows + rows
    times = pd.date_range("2026-01-01", periods=source_rows, freq="5min", tz="UTC")
    index = np.arange(source_rows, dtype=np.float64)
    mid = 2500.0 + index * 0.1 + np.sin(index * 0.11)
    open_ = mid - 0.05
    close = mid + 0.05 * np.sin(index * 0.17)
    source = pd.DataFrame(
        {
            "time": times,
            "mid": mid,
            "atr": 2.0 + 0.1 * np.cos(index * 0.09),
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
        }
    )
    source_path = tmp_path / "canonical_source.parquet"
    source.to_parquet(source_path, index=False)
    samples = pd.DataFrame({"time": times[warmup_rows:]})
    return matrix, names, samples, source, source_path


def _sha256_matrix(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes(order="C")).hexdigest()


def _sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def test_valid_full_contract_has_stable_names_order_and_bits(tmp_path: Path) -> None:
    matrix, names, samples, _source, source_path = _valid_inputs(tmp_path)
    chart_x, chart_names = build_chart_layer(matrix, names)
    price_x, price_names = build_price_derived_layer(samples, source_path)
    candle_x, candle_names = build_candlestick_derived_layer(samples, source_path)
    deep_input = np.concatenate([matrix, chart_x, price_x, candle_x], axis=1)
    deep_names_in = names + chart_names + price_names + candle_names
    deep_x, deep_names = build_deep_interaction_layer(deep_input, deep_names_in, samples)

    expected = {
        "chart": (
            chart_x,
            chart_names,
            (240, 242),
            "8a05e6d39902a9c1a8277744711dd1f1aab7806128531dc8f183df4f6032f1fc",
            "df40b572938f61d81be0cac5dad4df44e6f48b1233a0d224cc734c14e1ff01d9",
        ),
        "price": (
            price_x,
            price_names,
            (240, 11),
            "4a3ae77465c1b599d7746c8b29b2bab6af3caa995a9e90c8769be98d449c83a2",
            "c28fc8b5163b7ccbddf9a80f5445c4149dc5a672b0818635ea0f200b0e147b61",
        ),
        "candle": (
            candle_x,
            candle_names,
            (240, 60),
            "e4df04d431e6eac26e6068af305ce0bf21872c4b117030f88ddf5a65c69389ff",
            "102894513328840980d120ff830b1f3c76fb4617557619285107a5eb87134d47",
        ),
        "deep": (
            deep_x,
            deep_names,
            (240, 315),
            "aad1fe1e9b28169958ab862357082fb81c092735c412285d2e757e2b6e818b58",
            "8492ae7579b24364d21edbe08678177ea4610edb8e1a3cecf97d96625c4f82a8",
        ),
    }
    for values, feature_names, shape, value_hash, name_hash in expected.values():
        assert values.shape == shape
        assert len(feature_names) == len(set(feature_names)) == shape[1]
        assert np.isfinite(values).all()
        assert _sha256_matrix(values) == value_hash
        assert _sha256_names(feature_names) == name_hash


def test_chart_layer_rejects_missing_duplicate_and_nonfinite_sources(tmp_path: Path) -> None:
    matrix, names, _samples, _source, _source_path = _valid_inputs(tmp_path)
    missing_index = names.index("ctx_cont._v1h1_ema_diff")
    with pytest.raises(RuntimeError, match="CHART_LAYER_SOURCE_FIELDS_MISSING"):
        build_chart_layer(np.delete(matrix, missing_index, axis=1), names[:missing_index] + names[missing_index + 1 :])

    bad = matrix.copy()
    bad[3, missing_index] = np.nan
    with pytest.raises(RuntimeError, match="CHART_LAYER_SOURCE_NONFINITE"):
        build_chart_layer(bad, names)

    with pytest.raises(RuntimeError, match="CHART_LAYER_DUPLICATE_FEATURE_NAMES"):
        build_chart_layer(
            np.concatenate([matrix, matrix[:, :1]], axis=1),
            names + [names[0]],
        )


def test_price_layer_rejects_missing_or_invalid_source_evidence(tmp_path: Path) -> None:
    _matrix, _names, samples, source, _source_path = _valid_inputs(tmp_path)

    missing_atr = tmp_path / "missing_atr.parquet"
    source.drop(columns=["atr"]).to_parquet(missing_atr, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_ATR_MISSING"):
        build_price_derived_layer(samples, missing_atr)

    nonfinite = source.copy()
    nonfinite.loc[10, "atr"] = np.nan
    nonfinite_path = tmp_path / "nonfinite_atr.parquet"
    nonfinite.to_parquet(nonfinite_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_NONFINITE"):
        build_price_derived_layer(samples, nonfinite_path)

    duplicate = pd.concat([source, source.iloc[[4]]], ignore_index=True)
    duplicate_path = tmp_path / "duplicate_time.parquet"
    duplicate.to_parquet(duplicate_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_TIME_DUPLICATE"):
        build_price_derived_layer(samples, duplicate_path)

    cold_samples = pd.DataFrame({"time": source["time"].iloc[:200]})
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_LOCAL_EMA_WARMUP_INCOMPLETE"):
        build_price_derived_layer(cold_samples, tmp_path / "canonical_source.parquet")

    bad_samples = pd.DataFrame({"time": ["not-a-timestamp"]})
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SAMPLE_TIME_INVALID"):
        build_price_derived_layer(bad_samples, tmp_path / "canonical_source.parquet")

    gap_samples = pd.concat(
        [samples, pd.DataFrame({"time": [pd.Timestamp("2030-01-01", tz="UTC")]})],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_ROW_GAP"):
        build_price_derived_layer(gap_samples, tmp_path / "canonical_source.parquet")


def test_price_layer_uses_exact_close_and_has_no_mid_fallback(tmp_path: Path) -> None:
    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)
    expected, expected_names = build_price_derived_layer(samples, source_path)

    changed_mid = source.copy()
    changed_mid["mid"] = changed_mid["mid"] * 7.0 + 123.0
    changed_mid_path = tmp_path / "changed_mid.parquet"
    changed_mid.to_parquet(changed_mid_path, index=False)
    observed, observed_names = build_price_derived_layer(samples, changed_mid_path)
    assert observed_names == expected_names
    np.testing.assert_array_equal(observed, expected)

    mid_only_path = tmp_path / "mid_only.parquet"
    source.drop(columns=["close"]).to_parquet(mid_only_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_PRICE_MISSING"):
        build_price_derived_layer(samples, mid_only_path)


def test_chart_layer_bos_and_sweep_pressures_are_directionally_symmetric(
    tmp_path: Path,
) -> None:
    _matrix, names, _samples, _source, _source_path = _valid_inputs(tmp_path, rows=2)
    matrix = np.zeros((2, len(names)), dtype=np.float32)
    index = {name: column for column, name in enumerate(names)}

    for name in (
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.m15_trend_sign_canon_v2",
    ):
        matrix[:, index[name]] = np.asarray([1.0, -1.0], dtype=np.float32)
    matrix[:, index["ctx_cont.H1_range_compression_ratio"]] = 1.0
    matrix[:, index["ctx_cont.M15_range_compression_ratio"]] = 1.0
    matrix[:, index["ctx_cont.smc_bos_pressure_last12"]] = [0.8, -0.8]
    matrix[:, index["ctx_cont.smc_bos_pressure_last48"]] = [0.4, -0.4]
    matrix[:, index["ctx_cont.smc_sweep_bull_pressure_last12"]] = [0.8, -0.8]
    matrix[:, index["ctx_cont.smc_sweep_bull_pressure_last48"]] = [0.4, -0.4]

    values, output_names = build_chart_layer(matrix, names)
    output = {name: values[:, column] for column, name in enumerate(output_names)}
    assert output["chart.hh_breakout_proxy"][0] == pytest.approx(
        output["chart.ll_breakdown_proxy"][1]
    )
    assert output["chart.false_breakout_low_reject"][0] == pytest.approx(
        output["chart.false_breakout_high_reject"][1]
    )


def test_deep_layer_inverts_normalized_d1_regime_age_without_compression(
    tmp_path: Path,
) -> None:
    matrix, names, samples, _source, source_path = _valid_inputs(tmp_path, rows=2)
    matrix.fill(0.0)
    source_index = {name: column for column, name in enumerate(names)}
    matrix[:, source_index["ctx_cont.H1_range_compression_ratio"]] = 1.0
    matrix[:, source_index["ctx_cont.M15_range_compression_ratio"]] = 1.0
    matrix[:, source_index["ctx_cont.bars_since_d1_regime_change_v3"]] = [0.0, 1.0]
    matrix[:, source_index["ctx_cont.d1_regime_changed_flag_v3"]] = 0.0

    chart_x, chart_names = build_chart_layer(matrix, names)
    price_x, price_names = build_price_derived_layer(samples, source_path)
    candle_x, candle_names = build_candlestick_derived_layer(samples, source_path)
    deep_x, deep_names = build_deep_interaction_layer(
        np.concatenate([matrix, chart_x, price_x, candle_x], axis=1),
        names + chart_names + price_names + candle_names,
        samples,
    )
    fresh = deep_x[:, deep_names.index("chart.fresh_d1_regime_change_pressure")]
    np.testing.assert_array_equal(fresh, np.asarray([1.0, 0.0], dtype=np.float32))


def test_candlestick_and_deep_layers_reject_bad_geometry_and_row_mismatch(tmp_path: Path) -> None:
    matrix, names, samples, source, source_path = _valid_inputs(tmp_path)
    invalid_ohlc = source.copy()
    invalid_ohlc.loc[8, "high"] = invalid_ohlc.loc[8, "low"] - 1.0
    invalid_path = tmp_path / "invalid_ohlc.parquet"
    invalid_ohlc.to_parquet(invalid_path, index=False)
    with pytest.raises(RuntimeError, match="CANDLESTICK_DERIVED_SOURCE_OHLC_GEOMETRY_INVALID"):
        build_candlestick_derived_layer(samples, invalid_path)

    chart_x, chart_names = build_chart_layer(matrix, names)
    price_x, price_names = build_price_derived_layer(samples, source_path)
    deep_input = np.concatenate([matrix, chart_x, price_x], axis=1)
    deep_names = names + chart_names + price_names
    with pytest.raises(RuntimeError, match="DEEP_INTERACTION_ROW_MISMATCH"):
        build_deep_interaction_layer(deep_input, deep_names, samples.iloc[:-1])

    missing_index = deep_names.index("chart.local_ema50_200_spread_atr")
    with pytest.raises(RuntimeError, match="DEEP_INTERACTION_SOURCE_FIELDS_MISSING"):
        build_deep_interaction_layer(
            np.delete(deep_input, missing_index, axis=1),
            deep_names[:missing_index] + deep_names[missing_index + 1 :],
            samples,
        )
