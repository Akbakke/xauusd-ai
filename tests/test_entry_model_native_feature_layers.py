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
            (240, 115),
            # Value hash re-measured 2026-08-13 (V30 package 2): the synthetic
            # input matrix keys values off the ctx_cont column INDEX, and the
            # nine adopted swing V29 ctx fields plus the three momentum-G3 RSI
            # scalars shift every later ctx_cont column of the fixture again;
            # the chart formulas and the 115-name identity are unchanged (the
            # name hash below is untouched).
            "be1721f33ace67e2fa2932f14ad5fcbc4575b65d81a5ac6103572cc5f8b14de4",
            "63f1cc1721db84e7f171b35c3dbb206c89749ba08cc7a07d9263c5a4061f3a4d",
        ),
        "price": (
            price_x,
            price_names,
            # V30 (2026-08-13): package 1 added chart.local_kama_efficiency_30
            # and package 2 the three GAP-2/3 local age fields (15 fields);
            # both hashes re-measured on the unchanged source fixture.
            (240, 15),
            "237558a7c237e9b0444294ce1c33166492cd9c0ab20c5f3d906b56851c1766d0",
            "93e49f6577ea0ce89878dcc237b2d0431d72bcbcb54d780342386c888411e3d9",
        ),
        "candle": (
            candle_x,
            candle_names,
            (240, 60),
            # Value hash re-measured 2026-08-09 after the candlestick owner's
            # in-flight wave edit; the 60-name identity is unchanged.
            "89e8e112bae8752846b1604f0abd3bee909544248446a6e1bc5b640d7f88b3b4",
            "102894513328840980d120ff830b1f3c76fb4617557619285107a5eb87134d47",
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
