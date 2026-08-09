"""Value contracts for basic_v1 candle-shape conventions and ATR-relative trend.

Covers the 2026-08 forensic-audit repairs:
- ``_v1_body_share_1`` uses the file's stated zero-range convention
  (0/eps = 0.0 honest defined value) instead of fabricating a 0.5 half body.
- ``_v1_clv`` keeps its deliberately different neutral-0.5 convention for the
  undefined close location of a zero-range bar.
- ``_v1_ema_diff`` is an ATR-multiple (era-stable), not a USD spread that
  tracked gold's price level across the tape.
"""
import numpy as np
import pandas as pd

from gx1.features.basic_v1 import build_basic_v1


def _market_frame(periods: int = 5000) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="5min")
    phase = np.arange(periods, dtype=np.float64) * 2.0 * np.pi / (288.0 * 4.0)
    close = (
        2_000.0
        + np.linspace(0.0, 10.0, periods)
        + 8.0 * np.sin(phase)
        + 1.5 * np.sin(phase * 0.31)
    )
    half_range = 0.4 + 0.25 * (1.0 + np.sin(phase * 0.37))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": np.linspace(100.0, 300.0, periods),
            "spread_pct": np.full(periods, 1e-4, dtype=np.float64),
        },
        index=index,
    )


def test_body_share_1_zero_range_bar_is_honest_zero_and_clv_stays_neutral() -> None:
    frame = _market_frame()
    row = 3000
    price = float(frame["close"].iloc[row])
    frame.iloc[row, frame.columns.get_indexer(["open", "high", "low", "close"])] = price

    out, _ = build_basic_v1(frame)

    # Shift(1): the zero-range bar is read on the next row.
    body_share = out["_v1_body_share_1"].to_numpy(dtype=np.float64)
    clv = out["_v1_clv"].to_numpy(dtype=np.float64)
    # Zero body over zero range: 0.0 is the honest defined share, not a
    # fabricated half body.
    assert body_share[row + 1] == 0.0
    # Close location inside a zero-width range is undefined; 0.5 is the true
    # neutral midpoint of the [0, 1] domain and must stay.
    assert clv[row + 1] == 0.5
    # Row-0 seed follows the candle-shape convention (no prior bar -> 0.0).
    assert body_share[0] == 0.0
    assert np.isfinite(body_share).all()


def test_ema_diff_is_atr_relative_and_price_level_invariant() -> None:
    base = _market_frame()
    # 2.0 is a power of two: every linear price operation scales exactly in
    # binary floating point, so the ATR-multiple must be (near-)identical.
    scaled = base.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 2.0

    out_base, _ = build_basic_v1(base)
    out_scaled, _ = build_basic_v1(scaled)

    ema_base = out_base["_v1_ema_diff"].to_numpy(dtype=np.float64)
    ema_scaled = out_scaled["_v1_ema_diff"].to_numpy(dtype=np.float64)

    # Causal warmup: the ATR(14) rolling warmup propagates as a contiguous
    # NaN prefix (exact length is owned by the rolling implementation),
    # followed by an all-finite tail.
    first_finite = int(np.flatnonzero(np.isfinite(ema_base))[0])
    assert first_finite > 0
    assert np.isnan(ema_base[:first_finite]).all()
    assert np.isfinite(ema_base[first_finite:]).all()
    # Non-degenerate: the trend spread is actually live on this tape.
    assert np.nanmax(np.abs(ema_base)) > 0.0
    # Era-stability: doubling the price level must not change the feature.
    # A USD-scaled spread would double instead (the refuted date proxy).
    np.testing.assert_allclose(ema_scaled, ema_base, rtol=1e-9, equal_nan=True)
