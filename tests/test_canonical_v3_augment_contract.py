from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.htf_features import (
    MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4,
    model_native_mtf_owner_marker_v4,
)
from gx1.scripts.materialize_canonical_v3_augment import (
    add_cross_tf_momentum,
)


def _v4_projected_frame(
    close: np.ndarray,
    h1_atr: np.ndarray,
    *,
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    freq = "min" if decision_bar_duration == pd.Timedelta(minutes=1) else "5min"
    frame = pd.DataFrame(
        {
            name: np.ones(len(close), dtype=np.float64)
            for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq=freq, tz="UTC"),
    )
    frame["close"] = close
    frame["_v1h1_atr_bps"] = h1_atr
    frame.attrs["model_native_mtf_owner_v4"] = model_native_mtf_owner_marker_v4(
        decision_bar_duration=decision_bar_duration
    )
    return frame


def test_cross_tf_momentum_treats_zero_h1_atr_as_unavailable_not_tiny() -> None:
    close = np.arange(20, dtype=np.float64) + 2_000.0
    h1_atr = np.zeros(20, dtype=np.float64)
    h1_atr[13:] = 2.0

    out = add_cross_tf_momentum(
        _v4_projected_frame(
            close,
            h1_atr,
            decision_bar_duration=pd.Timedelta(minutes=5),
        ),
        decision_bar_duration=pd.Timedelta(minutes=5),
    )["m5h1_momentum"].to_numpy(dtype=np.float64)

    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:13], np.zeros(13))
    # 2026-08-09 unit repair: numerator is now bps of price, ATR input is bps.
    expected = (12.0 / close[13:] * 10000.0) / 2.0
    np.testing.assert_allclose(out[13:], expected, rtol=1e-6)


def test_cross_tf_momentum_carries_causal_nan_warmup_prefix() -> None:
    # Causal HTF construction emits a NaN warmup prefix (not neutral zero).
    close = np.arange(20, dtype=np.float64) + 2_000.0
    h1_atr = np.full(20, np.nan, dtype=np.float64)
    h1_atr[13:] = 2.0

    out = add_cross_tf_momentum(
        _v4_projected_frame(
            close,
            h1_atr,
            decision_bar_duration=pd.Timedelta(minutes=5),
        ),
        decision_bar_duration=pd.Timedelta(minutes=5),
    )["m5h1_momentum"].to_numpy(dtype=np.float64)

    assert np.isnan(out[:13]).all()
    assert np.isfinite(out[13:]).all()
    # 2026-08-09 unit repair: bps numerator over bps ATR (see producer comment).
    expected = (12.0 / close[13:] * 10000.0) / 2.0
    np.testing.assert_allclose(out[13:], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "column,value,error",
    (
        ("_v1h1_atr_bps", -1.0, "H1 ATR must be non-negative"),
        ("_v1h1_atr_bps", np.nan, "non-finite values after causal warmup"),
        ("close", np.nan, "close must be finite"),
    ),
)
def test_cross_tf_momentum_fails_closed_on_invalid_source(
    column: str,
    value: float,
    error: str,
) -> None:
    frame = _v4_projected_frame(
        np.arange(20, dtype=np.float64) + 2_000.0,
        np.ones(20, dtype=np.float64),
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    frame.iloc[15, frame.columns.get_loc(column)] = value

    with pytest.raises(RuntimeError, match=error):
        add_cross_tf_momentum(
            frame,
            decision_bar_duration=pd.Timedelta(minutes=5),
        )


def test_cross_tf_momentum_routes_one_hour_as_12_m5_or_60_m1_rows() -> None:
    m5_close = np.arange(70, dtype=np.float64) + 2_000.0
    m1_close = np.arange(70, dtype=np.float64) + 2_000.0
    atr = np.full(70, 2.0, dtype=np.float64)

    m5 = add_cross_tf_momentum(
        _v4_projected_frame(
            m5_close,
            atr,
            decision_bar_duration=pd.Timedelta(minutes=5),
        ),
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    m1 = add_cross_tf_momentum(
        _v4_projected_frame(
            m1_close,
            atr,
            decision_bar_duration=pd.Timedelta(minutes=1),
        ),
        decision_bar_duration=pd.Timedelta(minutes=1),
    )

    assert m5.attrs["m5h1_momentum_owner"]["horizon_rows"] == 12
    assert m1.attrs["m5h1_momentum_owner"]["horizon_rows"] == 60
    # 2026-08-09 unit repair: expected = (delta/close*1e4) / atr_bps.
    m5_expected = (12.0 / m5_close[-1] * 10000.0) / 2.0
    m1_expected = (60.0 / m1_close[-1] * 10000.0) / 2.0
    np.testing.assert_allclose(
        float(m5["m5h1_momentum"].iloc[-1]), m5_expected, rtol=1e-6
    )
    np.testing.assert_allclose(
        float(m1["m5h1_momentum"].iloc[-1]), m1_expected, rtol=1e-6
    )
