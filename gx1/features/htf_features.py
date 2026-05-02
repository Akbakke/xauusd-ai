"""On-the-fly higher-timeframe (HTF) feature computation from M5 OHLC candles.

Reproduces the same five context features that
`gx1/scripts/add_ctx_cont_columns_to_prebuilt.py` computes offline:

  - D1_dist_from_ema200_atr  (continuous)
  - H1_range_compression_ratio (continuous)
  - D1_atr_percentile_252 (continuous)
  - M15_range_compression_ratio (continuous)
  - H4_trend_sign_cat (categorical 0/1/2)

These are the 4 multi-timeframe ctx_cont features and the 1 ctx_cat feature
that were previously hard-coded to constants in
`gx1/execution/entry_context_features.py`. The hard-coding was harmless in
backtest because `oanda_demo_runner.py` overwrites with prebuilt values
before the values reach XGB - but the constants were dead code that would
become an active drift bug the moment we ran without prebuilt features
(e.g., real OANDA live trading).

Public API:

  - `compute_htf_features(m5_candles, current_ts) -> HTFFeatureResult`

Returns whatever subset of features could be computed given the available M5
warmup, with explicit `None` for those that lacked sufficient history. Callers
must handle `None` (typically by deferring to a prebuilt overwrite or by
fail-closed at the tensor-build step).

Determinism: same M5 input -> same HTF output, identical bit-for-bit to the
offline `add_ctx_cont_columns_to_prebuilt.py` computation, modulo the
`_align_last_closed` semantics ("last closed HTF bar at or before t - shift").

The functions never write to disk and never modify their inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Warmup requirements (mirrors the offline builder's hard-fail thresholds)
# ---------------------------------------------------------------------------

D1_EMA200_MIN_BARS = 220  # offline builder: len(df_d1) < 220 -> raise
H1_ATR100_MIN_BARS = 120  # offline builder: len(df_h1) < 120 -> raise
M15_ATR100_MIN_BARS = 200  # offline builder: len(df_m15) < 200 -> raise
H4_EMA50_MIN_BARS = 80  # offline builder: len(df_h4) < 80 -> raise
# D1 ATR14-percentile-252 needs 14 ATR14 warmup bars + 252 rolling window
# bars before producing the first non-NaN percentile, so min ~266 D1 bars.
# Using 270 for a small safety margin.
D1_PCTL252_MIN_BARS = 270

ATR_EPS = 1e-12


@dataclass
class HTFFeatureResult:
    d1_dist_from_ema200_atr: Optional[float]
    h1_range_compression_ratio: Optional[float]
    d1_atr_percentile_252: Optional[float]
    m15_range_compression_ratio: Optional[float]
    h4_trend_sign_cat: Optional[int]
    insufficient_warmup_for_v1: list[str]


# ---------------------------------------------------------------------------
# Helpers (deterministic, stateless; mirror add_ctx_cont_columns_to_prebuilt.py)
# ---------------------------------------------------------------------------


def _last_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": _last_valid}
    return df.resample(rule).agg(agg).dropna(how="all")


def _last_closed_at_or_before(
    series_htf: pd.Series, target_ts: pd.Timestamp, shift: pd.Timedelta
) -> Optional[float]:
    """Return the HTF series value at the last closed HTF bar whose timestamp
    is <= (target_ts - shift). Returns None if no such bar exists."""
    if series_htf is None or len(series_htf) == 0:
        return None
    cutoff = target_ts - shift
    candidates = series_htf.dropna()
    if candidates.empty:
        return None
    eligible = candidates[candidates.index <= cutoff]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def _last_closed_int_at_or_before(
    series_htf: pd.Series, target_ts: pd.Timestamp, shift: pd.Timedelta
) -> Optional[int]:
    val = _last_closed_at_or_before(series_htf, target_ts, shift)
    if val is None:
        return None
    return int(val)


def _validate_m5_input(m5_candles: pd.DataFrame) -> None:
    if not isinstance(m5_candles, pd.DataFrame):
        raise TypeError(
            f"HTF_INPUT_FAIL: m5_candles must be DataFrame, got {type(m5_candles).__name__}"
        )
    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in m5_candles.columns]
    if missing:
        raise RuntimeError(
            f"HTF_INPUT_FAIL: m5_candles missing required columns: {missing}"
        )
    if not isinstance(m5_candles.index, pd.DatetimeIndex):
        raise RuntimeError(
            "HTF_INPUT_FAIL: m5_candles index must be DatetimeIndex"
        )


# ---------------------------------------------------------------------------
# Per-feature computations
# ---------------------------------------------------------------------------


def compute_d1_dist_from_ema200_atr(
    m5_candles: pd.DataFrame, current_ts: pd.Timestamp
) -> Optional[float]:
    """Distance from D1 mid to D1 EMA200, normalized by D1 ATR14.

    Returns None if fewer than 220 D1 bars are available (insufficient EMA200 warmup).
    """
    _validate_m5_input(m5_candles)
    df_d1 = _resample_ohlc(m5_candles, "1D")
    if len(df_d1) < D1_EMA200_MIN_BARS:
        return None
    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = _ema(d1_mid, 200)
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
    d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)
    return _last_closed_at_or_before(d1_dist, current_ts, pd.Timedelta(days=1))


def compute_h1_range_compression_ratio(
    m5_candles: pd.DataFrame, current_ts: pd.Timestamp
) -> Optional[float]:
    """H1 ATR14 / H1 ATR100. Returns None if fewer than 120 H1 bars."""
    _validate_m5_input(m5_candles)
    df_h1 = _resample_ohlc(m5_candles, "1h")
    if len(df_h1) < H1_ATR100_MIN_BARS:
        return None
    h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
    h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
    h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)
    return _last_closed_at_or_before(h1_comp, current_ts, pd.Timedelta(hours=1))


def compute_d1_atr_percentile_252(
    m5_candles: pd.DataFrame, current_ts: pd.Timestamp
) -> Optional[float]:
    """Percentile rank of latest D1 ATR14 within rolling-252-day window.

    Returns None if fewer than 252 D1 bars available.
    """
    _validate_m5_input(m5_candles)
    df_d1 = _resample_ohlc(m5_candles, "1D")
    if len(df_d1) < D1_PCTL252_MIN_BARS:
        return None
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()

    def _pctl_last(window: np.ndarray) -> float:
        w = np.asarray(window, dtype=float)
        if not np.isfinite(w).all():
            return float("nan")
        last = w[-1]
        return float((w <= last).mean())

    atr_pctl252 = d1_atr14.rolling(252, min_periods=252).apply(_pctl_last, raw=True)
    atr_pctl252 = atr_pctl252.ffill()
    return _last_closed_at_or_before(atr_pctl252, current_ts, pd.Timedelta(days=1))


def compute_m15_range_compression_ratio(
    m5_candles: pd.DataFrame, current_ts: pd.Timestamp
) -> Optional[float]:
    """M15 ATR14 / M15 ATR100. Returns None if fewer than 200 M15 bars."""
    _validate_m5_input(m5_candles)
    df_m15 = _resample_ohlc(m5_candles, "15min")
    if len(df_m15) < M15_ATR100_MIN_BARS:
        return None
    m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
    m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill()
    m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
    return _last_closed_at_or_before(m15_comp, current_ts, pd.Timedelta(minutes=15))


def compute_h4_trend_sign_cat(
    m5_candles: pd.DataFrame, current_ts: pd.Timestamp
) -> Optional[int]:
    """sign(H4 mid - H4 EMA50) mapped to {0, 1, 2} for {-1, 0, +1}.

    Returns None if fewer than 80 H4 bars (insufficient EMA50 warmup).
    """
    _validate_m5_input(m5_candles)
    df_h4 = _resample_ohlc(m5_candles, "4h")
    if len(df_h4) < H4_EMA50_MIN_BARS:
        return None
    h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
    h4_ema50 = _ema(h4_mid, 50)
    diff = (h4_mid - h4_ema50)
    sign = np.sign(diff.fillna(0.0).to_numpy()).astype(np.int64)
    sign_cat = (sign + 1).astype(np.int64)
    series = pd.Series(sign_cat, index=df_h4.index, dtype="int64")
    return _last_closed_int_at_or_before(series, current_ts, pd.Timedelta(hours=4))


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def compute_htf_features(
    m5_candles: pd.DataFrame, current_ts: Optional[pd.Timestamp] = None
) -> HTFFeatureResult:
    """Compute all five HTF features at `current_ts` from `m5_candles`.

    Args:
        m5_candles: DataFrame indexed by DatetimeIndex with columns
            ``open``, ``high``, ``low``, ``close`` (M5 OHLC, UTC).
        current_ts: Decision-time timestamp. If None, uses the last index
            value of `m5_candles`.

    Returns:
        HTFFeatureResult with each field set to a float (or int for
        h4_trend_sign_cat) when warmup is satisfied, or None otherwise.
        ``insufficient_warmup_for_v1`` lists the names that returned None.
    """
    _validate_m5_input(m5_candles)
    if current_ts is None:
        if m5_candles.empty:
            raise RuntimeError(
                "HTF_INPUT_FAIL: cannot derive current_ts from empty m5_candles"
            )
        current_ts = pd.Timestamp(m5_candles.index[-1])
    current_ts = pd.Timestamp(current_ts)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("UTC")

    d1_dist = compute_d1_dist_from_ema200_atr(m5_candles, current_ts)
    h1_comp = compute_h1_range_compression_ratio(m5_candles, current_ts)
    d1_pctl = compute_d1_atr_percentile_252(m5_candles, current_ts)
    m15_comp = compute_m15_range_compression_ratio(m5_candles, current_ts)
    h4_trend = compute_h4_trend_sign_cat(m5_candles, current_ts)

    insufficient = []
    if d1_dist is None:
        insufficient.append("D1_dist_from_ema200_atr")
    if h1_comp is None:
        insufficient.append("H1_range_compression_ratio")
    if d1_pctl is None:
        insufficient.append("D1_atr_percentile_252")
    if m15_comp is None:
        insufficient.append("M15_range_compression_ratio")
    if h4_trend is None:
        insufficient.append("H4_trend_sign_cat")

    return HTFFeatureResult(
        d1_dist_from_ema200_atr=d1_dist,
        h1_range_compression_ratio=h1_comp,
        d1_atr_percentile_252=d1_pctl,
        m15_range_compression_ratio=m15_comp,
        h4_trend_sign_cat=h4_trend,
        insufficient_warmup_for_v1=insufficient,
    )
