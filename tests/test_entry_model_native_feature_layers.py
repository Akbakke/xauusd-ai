from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CAT_NAMES_V3, ORDERED_CTX_CONT_NAMES_V3
from gx1.features.entry_model_native_feature_layers_v1 import (
    build_candlestick_derived_layer,
    build_chart_layer,
    build_deep_interaction_layer,
    build_price_derived_layer,
)


def _valid_inputs(tmp_path: Path, *, rows: int = 240):
    names = [f"snap.{name}" for name in MODEL_NATIVE_BASE_FIELDS]
    names.extend(f"ctx_cont.{name}" for name in ORDERED_CTX_CONT_NAMES_V3)
    names.extend(f"ctx_cat.{name}" for name in ORDERED_CTX_CAT_NAMES_V3)
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(len(names), dtype=np.float64)[None, :]
    matrix = (0.5 + 0.4 * np.sin(row * 0.071 + column * 0.137)).astype(np.float32)
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    index = np.arange(rows, dtype=np.float64)
    mid = 2500.0 + index * 0.1 + np.sin(index * 0.11)
    open_ = mid - 0.05
    close = mid + 0.05 * np.sin(index * 0.17)
    source = pd.DataFrame(
        {
            "time": times,
            "mid": mid,
            "_v1_atr14": 2.0 + 0.1 * np.cos(index * 0.09),
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
        }
    )
    source_path = tmp_path / "canonical_source.parquet"
    source.to_parquet(source_path, index=False)
    return matrix, names, pd.DataFrame({"time": times}), source, source_path


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
            "8958e1eb2cd189a1c8a41a2f12cb741dd577aab6959dfff42631777435814600",
            "df40b572938f61d81be0cac5dad4df44e6f48b1233a0d224cc734c14e1ff01d9",
        ),
        "price": (
            price_x,
            price_names,
            (240, 11),
            "9c05ab71393cdc2cb59ad528c8e37d8f0f7383dd894b8b78fc236b2bb3e8f102",
            "cbc76f9975d8087be90ab336ee5fc3cfc2e5bba0fdc42bf64bcd5dec5fcd5f1a",
        ),
        "candle": (
            candle_x,
            candle_names,
            (240, 60),
            "6db5ab8703f9a50cbd2dc41534ab1b691c2e968b24ed04c289cfd92914ae854b",
            "102894513328840980d120ff830b1f3c76fb4617557619285107a5eb87134d47",
        ),
        "deep": (
            deep_x,
            deep_names,
            (240, 315),
            "41f82cbc9f0d0c6ebb046c459de0806f92c384a1f2171ac099a13d399b7c85d6",
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
    source.drop(columns=["_v1_atr14"]).to_parquet(missing_atr, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_ATR_MISSING"):
        build_price_derived_layer(samples, missing_atr)

    nonfinite = source.copy()
    nonfinite.loc[10, "_v1_atr14"] = np.nan
    nonfinite_path = tmp_path / "nonfinite_atr.parquet"
    nonfinite.to_parquet(nonfinite_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_NONFINITE"):
        build_price_derived_layer(samples, nonfinite_path)

    duplicate = pd.concat([source, source.iloc[[4]]], ignore_index=True)
    duplicate_path = tmp_path / "duplicate_time.parquet"
    duplicate.to_parquet(duplicate_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_TIME_DUPLICATE"):
        build_price_derived_layer(samples, duplicate_path)

    bad_samples = pd.DataFrame({"time": ["not-a-timestamp"]})
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SAMPLE_TIME_INVALID"):
        build_price_derived_layer(bad_samples, tmp_path / "canonical_source.parquet")

    gap_samples = pd.concat(
        [samples, pd.DataFrame({"time": [pd.Timestamp("2030-01-01", tz="UTC")]})],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_ROW_GAP"):
        build_price_derived_layer(gap_samples, tmp_path / "canonical_source.parquet")


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

    missing_index = deep_names.index("chart.m5_ema50_200_spread_atr")
    with pytest.raises(RuntimeError, match="DEEP_INTERACTION_SOURCE_FIELDS_MISSING"):
        build_deep_interaction_layer(
            np.delete(deep_input, missing_index, axis=1),
            deep_names[:missing_index] + deep_names[missing_index + 1 :],
            samples,
        )
