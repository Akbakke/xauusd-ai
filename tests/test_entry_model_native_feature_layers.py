from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CAT_FIELDS, MODEL_NATIVE_CTX_CONT_FIELDS
from gx1.features.entry_chart_geometry_v1 import CHART_GEOMETRY_FEATURE_NAMES
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    build_candlestick_derived_layer,
    build_chart_layer,
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

    # The chart layer is a pure dispatcher of its two registered children;
    # the per-column value hashes below are bit-identical to the pre-removal
    # emissions of the same columns.
    assert chart_names == [
        *FOUNDATION_STRUCTURE_FEATURE_NAMES,
        *CHART_GEOMETRY_FEATURE_NAMES,
    ]

    expected = {
        "chart": (
            chart_x,
            chart_names,
            # V30 package 7 (2026-08-13): 57 foundation + 15 chart geometry.
            # The chart-geometry layer dropped 43 NAME-ONLY / duplicate columns,
            # so BOTH hashes move; the surviving 15 columns are bit-identical to
            # their pre-removal emissions (verified column-by-column below).
            (240, 72),
            "bb4c08c942e525c0d6c19e42a13158dfc786c90575fa4112e573be37420b6261",
            "7a19f36fda7af8e706bfddc854d092b12b9b850ce68b5ece5d3e38c6337c5843",
        ),
        "price": (
            price_x,
            price_names,
            # V30 (2026-08-13): package 1 added chart.local_kama_efficiency_30,
            # package 2 the three GAP-2/3 local age fields, and package 3 the
            # four price-vs-EMA cross events (19 fields); all hashes
            # re-measured on the unchanged source fixture.
            (240, 19),
            "31eebaba49421ad2657675810185b989e4aea96b26c5a5be2b2ad7e92dacb3f5",
            "7f0aaa17bd5628053736526dcb74ef2064f110d6a4566a49393f2d3d14118c84",
        ),
        "candle": (
            candle_x,
            candle_names,
            # V30 package 7 (2026-08-13): 60 -> 53 columns (six aggregate votes
            # + the affine duplicate close_pressure_signed); both hashes move.
            (240, 53),
            "685415499b56b26b158d49cad2beb45fc80cc5e0902b21e1f5a0b7e982928474",
            "abdc8e32c37e0308fbdfcf2468bb075cf45790f95d9228d9e2367c4000aaed9a",
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


def test_candlestick_layer_rejects_bad_ohlc_geometry(tmp_path: Path) -> None:
    _matrix, _names, samples, source, _source_path = _valid_inputs(tmp_path)
    invalid_ohlc = source.copy()
    invalid_ohlc.loc[8, "high"] = invalid_ohlc.loc[8, "low"] - 1.0
    invalid_path = tmp_path / "invalid_ohlc.parquet"
    invalid_ohlc.to_parquet(invalid_path, index=False)
    with pytest.raises(RuntimeError, match="CANDLESTICK_DERIVED_SOURCE_OHLC_GEOMETRY_INVALID"):
        build_candlestick_derived_layer(samples, invalid_path)
