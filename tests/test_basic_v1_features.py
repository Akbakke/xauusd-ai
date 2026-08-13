"""Value contracts for basic_v1 candle-shape conventions and ATR-relative trend.

Covers the 2026-08 forensic-audit repairs:
- ``_v1_body_share_1`` uses the file's stated zero-range convention
  (0/eps = 0.0 honest defined value) instead of fabricating a 0.5 half body.
- ``_v1_clv`` keeps its deliberately different neutral-0.5 convention for the
  undefined close location of a zero-range bar.
- ``_v1_ema_diff`` is an ATR-multiple (era-stable), not a USD spread that
  tracked gold's price level across the tape.

And the V30 (2026-08-13) noise-amplifier repairs (formula-based expectations,
the d71a8e57 repair-wave precedent):
- ``_v1_kama_slope_30`` / ``_v1_tema_slope_20`` are k-bar ATR-multiple
  changes (k=5 / k=3), not 5th/3rd-order finite differences: on an
  (asymptotically) linear tape the k-bar change equals k x the per-bar step
  while any order>=2 finite difference is identically 0 — an algebraic
  discriminator between the repair and the bug.
- ``_v1_bb_bandwidth_delta_10`` is the plain 3-bar change of the
  dimensionless bandwidth.
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


def _linear_frame(periods: int = 5000, step: float = 0.5) -> pd.DataFrame:
    """Strictly linear close with constant true range (ATR14 == 4.0 exactly).

    high - low = 4.0 dominates |high - prev_close| = step + 2.0 and
    |low - prev_close| = 2.0 - step, so TR is the constant 4.0 on every bar
    and the rolling ATR14 is exactly 4.0 wherever it is defined.
    """
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="5min")
    close = 2_000.0 + step * np.arange(periods, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.full(periods, 100.0),
            "spread_pct": np.full(periods, 1e-4, dtype=np.float64),
        },
        index=index,
    )


def test_kama_tema_slopes_are_kbar_atr_multiple_changes() -> None:
    """Algebraic proof of the V30 repair, independent of the implementation.

    On a strictly linear tape both smoothers converge to lagged linear
    trajectories with the SAME per-bar step d (TEMA is a linear filter; KAMA
    has efficiency ratio exactly 1 on a monotone tape, so its smoothing
    constant is fixed and its transient decays geometrically).  In the
    converged tail the k-bar change is exactly k*d, so the ATR-multiple
    slope is k*d/ATR — while the retired order-k finite difference of a
    linear sequence is identically 0.  The equality below therefore both
    pins the repaired formula and refutes the noise-amplifier form.
    """
    step = 0.5
    out, _ = build_basic_v1(_linear_frame(step=step))
    tema_slope = out["_v1_tema_slope_20"].to_numpy(dtype=np.float64)
    kama_slope = out["_v1_kama_slope_30"].to_numpy(dtype=np.float64)

    # Honest causal warmup: one contiguous NaN prefix, then all-finite.
    for values in (tema_slope, kama_slope):
        first_finite = int(np.flatnonzero(np.isfinite(values))[0])
        assert first_finite > 0
        assert np.isnan(values[:first_finite]).all()
        assert np.isfinite(values[first_finite:]).all()

    tail = slice(-500, None)
    atr = 4.0  # exact by construction (constant TR tape)
    np.testing.assert_allclose(
        tema_slope[tail], np.full(500, 3.0 * step / atr), rtol=1e-9
    )
    np.testing.assert_allclose(
        kama_slope[tail], np.full(500, 5.0 * step / atr), rtol=1e-9
    )


def test_bb_bandwidth_delta_10_is_plain_3bar_change() -> None:
    """Formula-based expectation: shift-by-1 of bw[t] - bw[t-3] where bw is
    the dimensionless 10-bar Bollinger bandwidth (4*std10 / (mean10+eps),
    min_periods=5, ddof=0).  The rolling mean/std come from the module's own
    timed_rolling owner (the property under test is the plain 3-bar-change
    construction, not the rolling-std algorithm; the retired
    ``np.diff(n=3)`` 3rd-order form fails this identity)."""
    from gx1.features.rolling_timer import timed_rolling

    frame = _market_frame()
    out, _ = build_basic_v1(frame)
    got = out["_v1_bb_bandwidth_delta_10"].to_numpy(dtype=np.float64)

    close = frame["close"]
    mean10 = timed_rolling(close, 10, "mean", min_periods=5)
    std10 = timed_rolling(close, 10, "std", min_periods=5, ddof=0)
    # Exact production algebra ((m+2s) - (m-2s), not the algebraically equal
    # 4s) so the identity is bit-tight.
    bw = ((mean10 + 2.0 * std10) - (mean10 - 2.0 * std10)) / (mean10 + 1e-12)
    expected = (bw - bw.shift(3)).shift(1).to_numpy(dtype=np.float64)

    np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)
    # Honest causal warmup prefix, then all-finite (no nan_to_num masking).
    first_finite = int(np.flatnonzero(np.isfinite(got))[0])
    assert first_finite > 0
    assert np.isnan(got[:first_finite]).all()
    assert np.isfinite(got[first_finite:]).all()
