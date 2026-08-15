from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.basic_v1 import compute_plus5_features
from gx1.features.htf_features import _atr as htf_atr
from gx1.features.technical_indicators_v1 import (
    EMA50_200_SPREAD_ATR_BLOCK_COLUMNS,
    TECHNICAL_INDICATOR_FORMULA_SHA256,
    TECHNICAL_INDICATOR_PRIMITIVES_VERSION,
    classic_ema,
    ema50_200_spread_atr_block,
    wilder_atr,
    wilder_atr14_positive,
    wilder_rsi,
)


def test_technical_formula_identity_is_exact_and_current() -> None:
    assert TECHNICAL_INDICATOR_PRIMITIVES_VERSION == (
        "technical_indicator_primitives_v2_classic_seed_raw_20260814"
    )
    assert TECHNICAL_INDICATOR_FORMULA_SHA256 == (
        "e08d52d42a01b850fca755bf7996a7701609b776a21f12428abb08450f2ab1c1"
    )


def test_wilder_atr_has_sma_seed_then_recursive_updates() -> None:
    index = pd.RangeIndex(4)
    high = pd.Series([11.0, 13.0, 15.0, 18.0], index=index)
    low = pd.Series([9.0, 10.0, 11.0, 13.0], index=index)
    close = pd.Series([10.0, 12.0, 14.0, 17.0], index=index)

    observed = wilder_atr(high, low, close, 2).to_numpy()

    # TR = [2, 3, 4, 5], seed=(2+3)/2=2.5, then Wilder recursion.
    np.testing.assert_allclose(
        observed,
        np.asarray([np.nan, 2.5, 3.25, 4.125]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_local_and_mtf_atr_routes_are_formula_identical() -> None:
    rows = 40
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(2000.0 + np.sin(np.arange(rows) / 3.0), index=index)
    high = close + 1.0 + 0.1 * np.cos(np.arange(rows))
    low = close - 0.8 - 0.1 * np.sin(np.arange(rows))
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(rows, 100.0),
        },
        index=index,
    )

    local = compute_plus5_features(frame)["atr"].to_numpy(dtype=np.float64)
    mtf = htf_atr(high, low, close, 14).to_numpy(dtype=np.float64)

    np.testing.assert_array_equal(local, mtf.astype(np.float32))


def test_wilder_atr_fails_closed_on_bad_period_and_geometry() -> None:
    high = pd.Series([2.0, 2.0])
    low = pd.Series([1.0, 3.0])
    close = pd.Series([1.5, 2.5])
    with pytest.raises(RuntimeError, match="WILDER_ATR_PERIOD_INVALID"):
        wilder_atr(high, low, close, 0)
    with pytest.raises(RuntimeError, match="WILDER_ATR_SOURCE_GEOMETRY_INVALID"):
        wilder_atr(high, low, close, 2)


def test_wilder_rsi_known_vector_and_flat_series() -> None:
    # Published-style canonical Wilder vector used independently by the HTF
    # owner test; first RSI after 14 changes is 70.464135...
    close = pd.Series(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
            46.21,
        ]
    )
    observed = wilder_rsi(close, 14)
    assert observed.iloc[:14].isna().all()
    np.testing.assert_allclose(
        observed.iloc[14:21].to_numpy(dtype=np.float64),
        np.asarray(
            [
                70.464135,
                66.249619,
                66.480942,
                69.346853,
                66.294713,
                57.915021,
                62.880718,
            ]
        ),
        rtol=0.0,
        atol=1e-6,
    )
    flat = wilder_rsi(pd.Series(np.full(30, 100.0)), 14)
    assert (flat.iloc[14:] == 50.0).all()


def test_ema50_200_spread_block_has_one_float64_formula_contract() -> None:
    rows = 260
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(
        2000.0 + np.arange(rows) * 0.04 + np.sin(np.arange(rows) / 7.0),
        index=index,
    )
    high = close + 0.7
    low = close - 0.5
    observed = ema50_200_spread_atr_block(high, low, close)
    assert tuple(observed.columns) == EMA50_200_SPREAD_ATR_BLOCK_COLUMNS
    assert all(dtype == np.dtype(np.float64) for dtype in observed.dtypes)

    atr14 = wilder_atr(high, low, close, 14)
    atr_positive = atr14.where(atr14 > 0.0)
    ema50 = classic_ema(close, 50)
    ema200 = classic_ema(close, 200)
    spread = ema50 - ema200
    expected = spread / atr_positive
    np.testing.assert_array_equal(
        observed["spread_atr"].to_numpy(dtype=np.float64),
        expected.to_numpy(dtype=np.float64),
    )


def test_classic_ema_has_sma_seed_and_matches_basic_owner() -> None:
    source = pd.Series(np.arange(1.0, 9.0))
    observed = classic_ema(source, 3)
    expected = np.asarray([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(observed.to_numpy(), expected)

    from gx1.features.basic_v1 import _ema as basic_ema

    np.testing.assert_array_equal(
        observed.to_numpy(dtype=np.float64),
        basic_ema(source, 3).to_numpy(dtype=np.float64),
    )


def test_zero_atr_is_unavailable_and_spread_is_not_saturated() -> None:
    flat = pd.Series(np.full(40, 2000.0))
    atr14, positive = wilder_atr14_positive(flat, flat, flat)
    assert (atr14.iloc[13:] == 0.0).all()
    assert positive.isna().all()

    rows = 500
    close = pd.Series(1000.0 + np.arange(rows, dtype=np.float64))
    high = close + 0.01
    low = close - 0.01
    raw = ema50_200_spread_atr_block(high, low, close)["spread_atr"]
    assert float(raw.iloc[-1]) > 30.0


def test_local_price_owner_has_no_clip_floor_or_epsilon_route() -> None:
    import inspect

    from gx1.features.entry_model_native_feature_layers_v1 import (
        build_price_derived_layer,
    )
    from gx1.features import technical_indicators_v1 as technical

    local_source = inspect.getsource(build_price_derived_layer)
    shared_source = inspect.getsource(technical.ema50_200_spread_atr_block)
    assert "_clip" not in local_source
    assert ".clip(" not in local_source
    assert "1e-12" not in local_source
    assert "atr_floor" not in shared_source
    assert ".clip(" not in shared_source
    assert ".ewm(" not in shared_source
