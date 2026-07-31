"""On-the-fly higher-timeframe (HTF) feature computation from M5 OHLC candles.

Reproduces the same five context features that
`gx1/scripts/add_ctx_cont_columns_to_prebuilt.py` computes offline:

  - D1_dist_from_ema200_atr  (continuous)
  - H1_range_compression_ratio (continuous)
  - D1_atr_percentile_252 (continuous)
  - M15_range_compression_ratio (continuous)
  - H4_trend_sign_cat (categorical 0/1/2)

These are the 4 multi-timeframe ctx_cont features and the 1 ctx_cat feature
that an older, now-retired context helper replaced with constants. Active
model-native state construction requires the observed values and fails closed
when they are unavailable.

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

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

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


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate the exact observed OHLCV source needed by the V2 model path."""
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise RuntimeError(
            f"HTF_V2_VOLUME_SOURCE_MISSING: exact OHLCV source required; missing={missing}"
        )
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": _last_valid,
        "volume": "sum",
    }
    return df.loc[:, list(required)].resample(rule).agg(agg).dropna(how="all")


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


def _validate_m5_input(
    m5_candles: pd.DataFrame,
    *,
    require_volume: bool = False,
) -> None:
    if not isinstance(m5_candles, pd.DataFrame):
        raise TypeError(
            f"HTF_INPUT_FAIL: m5_candles must be DataFrame, got {type(m5_candles).__name__}"
        )
    if m5_candles.empty:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles must be non-empty")
    required_cols = ["open", "high", "low", "close"]
    if require_volume:
        required_cols.append("volume")
    missing = [c for c in required_cols if c not in m5_candles.columns]
    if missing:
        raise RuntimeError(
            f"HTF_INPUT_FAIL: m5_candles missing required columns: {missing}"
        )
    if not isinstance(m5_candles.index, pd.DatetimeIndex):
        raise RuntimeError(
            "HTF_INPUT_FAIL: m5_candles index must be DatetimeIndex"
        )
    if m5_candles.index.tz is None:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be timezone-aware UTC")
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in m5_candles.index[:1]):
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be UTC")
    if (
        m5_candles.index.hasnans
        or not m5_candles.index.is_unique
        or not m5_candles.index.is_monotonic_increasing
    ):
        raise RuntimeError(
            "HTF_INPUT_FAIL: timestamps must be finite, unique and chronological"
        )
    numeric = m5_candles.loc[:, required_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("HTF_INPUT_FAIL: exact OHLCV sources must be finite")
    open_values = numeric["open"].to_numpy(dtype=np.float64)
    high_values = numeric["high"].to_numpy(dtype=np.float64)
    low_values = numeric["low"].to_numpy(dtype=np.float64)
    close_values = numeric["close"].to_numpy(dtype=np.float64)
    if (
        np.any(open_values <= 0.0)
        or np.any(high_values <= 0.0)
        or np.any(low_values <= 0.0)
        or np.any(close_values <= 0.0)
        or np.any(high_values < close_values)
        or np.any(low_values > close_values)
        or np.any(high_values < low_values)
        or (require_volume and np.any(high_values < open_values))
        or (require_volume and np.any(low_values > open_values))
    ):
        raise RuntimeError("HTF_INPUT_FAIL: OHLC geometry is invalid")
    if require_volume and np.any(numeric["volume"].to_numpy(dtype=np.float64) <= 0.0):
        raise RuntimeError(
            "HTF_V2_VOLUME_SOURCE_INVALID: observed volume must be finite and positive"
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
# Vectorized full-tape HTF (one truth for offline and immutable live snapshots)
# ---------------------------------------------------------------------------

HTF_TAPE_COLUMNS = (
    "D1_dist_from_ema200_atr",
    "D1_atr_percentile_252",
    "H1_range_compression_ratio",
    "M15_range_compression_ratio",
    "H4_trend_sign_cat",
)


def _align_last_closed_tape(
    target_index: pd.DatetimeIndex, htf_series: pd.Series, shift: pd.Timedelta
) -> pd.Series:
    """Align at the M5 decision time, five minutes after its start label."""

    shifted = htf_series.copy()
    shifted.index = shifted.index + shift
    decision_index = target_index + pd.Timedelta(minutes=5)
    aligned = shifted.reindex(decision_index, method="ffill")
    aligned.index = target_index
    return aligned


def build_htf_tape(m5_candles: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 HTF features for EVERY M5 bar (vectorized full-tape), no lookahead.

    ONE TRUTH for the offline ctx builder (add_ctx_cont_columns_to_prebuilt) and the
    immutable live snapshot owner, so serving uses the same fresh HTF the training
    distribution was built with — preventing forward-fill/frozen-HTF drift (e.g. the
    H4 trend sign says down-when-down instead of a stale value). Strict warmup floors match
    the offline builder; FAIL-LOUD (raise) if warmup is unmet — never emits an unconverged
    value. Returns a DataFrame indexed like ``m5_candles`` with ``HTF_TAPE_COLUMNS``.
    """
    _validate_m5_input(m5_candles)
    m5 = m5_candles
    if not isinstance(m5.index, pd.DatetimeIndex):
        if "time" in m5.columns:
            m5 = m5.set_index(pd.DatetimeIndex(pd.to_datetime(m5["time"], utc=True)))
        else:
            raise RuntimeError("[BUILD_HTF_TAPE] m5_candles needs a DatetimeIndex or a 'time' column")

    df_d1 = _resample_ohlc(m5, "1D")
    df_h1 = _resample_ohlc(m5, "1h")
    df_m15 = _resample_ohlc(m5, "15min")
    df_h4 = _resample_ohlc(m5, "4h")
    # Strict warmup floors (match the offline builder: no unconverged HTF on short tapes).
    if len(df_d1) < D1_PCTL252_MIN_BARS:
        raise RuntimeError(
            f"[BUILD_HTF_TAPE] insufficient D1 bars ({len(df_d1)} < {D1_PCTL252_MIN_BARS}) "
            "for ATR14 252-day percentile warmup"
        )
    if len(df_h1) < H1_ATR100_MIN_BARS:
        raise RuntimeError(f"[BUILD_HTF_TAPE] insufficient H1 bars ({len(df_h1)} < {H1_ATR100_MIN_BARS})")
    if len(df_m15) < M15_ATR100_MIN_BARS:
        raise RuntimeError(f"[BUILD_HTF_TAPE] insufficient M15 bars ({len(df_m15)} < {M15_ATR100_MIN_BARS})")
    if len(df_h4) < H4_EMA50_MIN_BARS:
        raise RuntimeError(f"[BUILD_HTF_TAPE] insufficient H4 bars ({len(df_h4)} < {H4_EMA50_MIN_BARS})")

    # D1: dist-from-EMA200 (ATR units) + ATR14 252-day percentile rank
    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = _ema(d1_mid, 200)
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
    d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)

    def _pctl_last(window: np.ndarray) -> float:
        w = np.asarray(window, dtype=float)
        if not np.isfinite(w).all():
            return float("nan")
        return float((w <= w[-1]).mean())

    d1_atr_pctl252 = d1_atr14.rolling(252, min_periods=252).apply(_pctl_last, raw=True).ffill()

    # H1 / M15 range compression (ATR14 / ATR100)
    h1_comp = (
        _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
        / np.maximum(_atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill(), ATR_EPS)
    )
    m15_comp = (
        _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
        / np.maximum(_atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill(), ATR_EPS)
    )

    # H4 trend sign cat {0,1,2} = sign(H4 mid - H4 EMA50) + 1
    h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
    h4_sign = np.sign((h4_mid - _ema(h4_mid, 50)).fillna(0.0).to_numpy()).astype(np.int64)
    h4_cat = pd.Series((h4_sign + 1).astype(np.int64), index=df_h4.index)

    idx = m5.index
    out = pd.DataFrame(index=idx)
    out["D1_dist_from_ema200_atr"] = _align_last_closed_tape(idx, d1_dist, pd.Timedelta(days=1)).to_numpy(dtype=float)
    out["D1_atr_percentile_252"] = _align_last_closed_tape(idx, d1_atr_pctl252, pd.Timedelta(days=1)).to_numpy(dtype=float)
    out["H1_range_compression_ratio"] = _align_last_closed_tape(idx, h1_comp, pd.Timedelta(hours=1)).to_numpy(dtype=float)
    out["M15_range_compression_ratio"] = _align_last_closed_tape(idx, m15_comp, pd.Timedelta(minutes=15)).to_numpy(dtype=float)
    out["H4_trend_sign_cat"] = _align_last_closed_tape(idx, h4_cat, pd.Timedelta(hours=4)).to_numpy()
    return out


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


# ---------------------------------------------------------------------------
# Multi-TF per-bar features (V12.2)
#
# Used by V10 v3 / V3 v8 multi-TF mode (enable_multi_tf=True).
# Produces per-bar time-series feature tables for H1/H4/D1 so transformer
# encoders can attend across each timeframe's recent history.
# Reuses existing _resample_ohlc / _atr / _ema for consistency with v3 scalar
# HTF features — same math, just per-bar instead of scalar lookup.
# ---------------------------------------------------------------------------

# Per-bar feature contract — order must stay stable (state_dict keys depend on it).
# V12.2 v2: dropped raw OHLC (4150 ≈ XAUUSD price dominates everything else).
# Replaced with scale-invariant relatives. All features now in roughly [-5, 5] range
# after winsorizing — works much better with transformer input.
MULTI_TF_PER_BAR_FEATURES = (
    # Price-relative features (all scale-invariant)
    "close_open_pct",        # (close-open)/open  — bar direction strength
    "high_low_atr",          # (high-low)/atr14   — bar range vs typical
    "close_open_atr",        # (close-open)/atr14 — bar direction in ATR units
    # Returns (clipped to ±500 bps = ±5% per bar)
    "ret_1", "ret_3", "ret_5", "ret_10",
    # Volatility
    "atr_bps_14",            # ATR in bps of close (already scale-invariant)
    # Momentum (RSI normalized)
    "rsi14_centered",        # (rsi14 - 50) / 50  → [-1, 1]
    "mom_1_atr", "mom_5_atr", "mom_20_atr",  # already ATR-normalized, but clipped
    # Position / shape
    "range_pos_20",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    # Trend
    "ema20_dist_atr",        # clipped
)
MULTI_TF_FEATURE_COUNT = len(MULTI_TF_PER_BAR_FEATURES)   # = 17

# ─────────────────────────────────────────────────────────────────────────────
# V2 (2026-05-22): 25 features per TF. Adds full EMA stack + VWAP + BB + regime
# to address user-observed gap (live trade went LONG while H4/D1 trend negative).
# V1 above is preserved for backward-compat with currently-deployed V10/V3 bundles.
# Both can co-exist until V2 retraining is cement + deployed.
# ─────────────────────────────────────────────────────────────────────────────
MULTI_TF_PER_BAR_FEATURES_V2 = (
    # KEPT from V1 (9 features) — proven signal from feature_importance analysis
    "atr_bps_14", "rsi14_centered", "mom_5_atr", "mom_20_atr",
    "close_open_atr", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "ema20_dist_atr",
    # NEW: full EMA stack (3 distances + 3 slopes)
    "ema50_dist_atr", "ema100_dist_atr", "ema200_dist_atr",
    "ema20_slope_atr", "ema50_slope_atr", "ema200_slope_atr",
    # NEW: EMA-regime (2)
    "ema_stack_aligned_v2",   # int {-1,0,+1}: bear/range/bull
    "regime_class_id",        # int {0..4}: range/up_low/up_high/down_low/down_high
    # NEW: VWAP family (4)
    "vwap_session_dist_atr",  # dist to session VWAP
    "vwap20_dist_atr",        # rolling 20-bar VWAP
    "vwap96_dist_atr",        # rolling 96-bar VWAP
    "vwap_session_slope_atr", # session-VWAP velocity
    # NEW: Bollinger + trend strength (4)
    "bb_position",            # (close − bb_lower) / (bb_upper − bb_lower) ∈ [0,1]
    "bb_width_atr",           # (bb_upper − bb_lower) / atr14
    "adx_centered",           # (adx − 25) / 25
    "trend_age_bars_norm",    # bars since last EMA stack flip, normalized log
)
MULTI_TF_FEATURE_COUNT_V2 = len(MULTI_TF_PER_BAR_FEATURES_V2)   # = 25
MULTI_TF_FEATURE_NAMES_SHA256_V2 = hashlib.sha256(
    json.dumps(
        list(MULTI_TF_PER_BAR_FEATURES_V2),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
HTF_V2_CACHE_SCHEMA_VERSION = "htf_v2_disk_cache_manifest_v2"
HTF_V2_CACHE_BUILDER_VERSION = (
    "prebuild_multi_tf_cache_v2_causal_sha256_bound_no_fallback_20260723"
)
HTF_V2_MATRIX_CONTRACT = "HTF_V2_CAUSAL_MATRIX_V1"

# ── V3 per-bar contract: the same 25, plus the two families that are pure price
# geometry and therefore mean the same thing at every resolution ──────────────
#
# The 513 signal fields are M5-only because their builders sit on 199 upstream
# source fields, 194 of them derived, with dependencies between the families -
# reproducing them per timeframe means rebuilding the whole context pipeline.
# Two owners do NOT have that dependency: the candlestick family needs exactly
# ["open", "high", "low", "close", "time"], and swing structure is a pure
# function of (high, low, close). Both were run unchanged on resampled bars on
# 2026-07-28: every value finite at all five resolutions, and the non-zero share
# holds instead of collapsing - candles 0.361 at M5 against 0.379 at D1, swing
# structure 0.981 against 0.968. Sparsity was the reason to fear higher
# timeframes; measured, it is not present in these two families.
#
# V2 is left exactly as it is. Its cache, hashes and every artifact built on it
# stay valid; V3 is a second contract, not a redefinition.
def _candlestick_v3_names() -> tuple[str, ...]:
    """The candlestick family's own declared names, prefixed for this surface.

    Imported from the owner rather than duplicated: one truth for what the
    family emits, and a name change there cannot silently desync this contract.
    """
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )
    return tuple(
        f"mtf_{name.split('.', 1)[1] if '.' in name else name}"
        for name in CANDLESTICK_PATTERN_FEATURE_NAMES
    )


MULTI_TF_PER_BAR_CANDLESTICK_V3 = _candlestick_v3_names()
MULTI_TF_PER_BAR_SWING_V3 = (
    "swing_bars_since_swing_high",
    "swing_bars_since_swing_low",
    "swing_dist_last_swing_high_atr",
    "swing_dist_last_swing_low_atr",
    "swing_retracement_from_last_impulse",
)
MULTI_TF_PER_BAR_FEATURES_V3 = (
    MULTI_TF_PER_BAR_FEATURES_V2
    + MULTI_TF_PER_BAR_CANDLESTICK_V3
    + MULTI_TF_PER_BAR_SWING_V3
)
MULTI_TF_FEATURE_COUNT_V3 = len(MULTI_TF_PER_BAR_FEATURES_V3)
MULTI_TF_FEATURE_NAMES_SHA256_V3 = hashlib.sha256(
    json.dumps(
        list(MULTI_TF_PER_BAR_FEATURES_V3),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
HTF_V3_MATRIX_CONTRACT = "HTF_V3_CAUSAL_MATRIX_V1"
HTF_V3_CACHE_SCHEMA_VERSION = "htf_v3_disk_cache_manifest_v1"
HTF_V3_CACHE_BUILDER_VERSION = (
    "prebuild_multi_tf_cache_v3_causal_sha256_bound_20260728"
)

# V4 is the first per-resolution surface with all eight Entry specialist
# families.  V2/V3 remain immutable historical contracts.  The two old
# ``vwap_session_*`` names are deliberately corrected here: on D1 their owner
# computes a declared five-bar local cycle rather than a daily session, so V4
# names the shared semantic honestly instead of carrying a proxy label.
from gx1.features.smc_v1 import (  # noqa: E402
    SMC_MTF_FEATURE_NAMES_V1,
    SMC_MTF_GEOMETRY_FEATURE_NAMES_V1,
)

MULTI_TF_PER_BAR_FEATURES_V4_BASE = tuple(
    "vwap_local_cycle_dist_atr"
    if name == "vwap_session_dist_atr"
    else "vwap_local_cycle_slope_atr"
    if name == "vwap_session_slope_atr"
    else name
    for name in MULTI_TF_PER_BAR_FEATURES_V3
)
MULTI_TF_PER_BAR_FEATURES_V4 = (
    MULTI_TF_PER_BAR_FEATURES_V4_BASE
    + SMC_MTF_FEATURE_NAMES_V1
    + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
)
MULTI_TF_FEATURE_COUNT_V4 = len(MULTI_TF_PER_BAR_FEATURES_V4)
MULTI_TF_FEATURE_NAMES_SHA256_V4 = hashlib.sha256(
    json.dumps(
        list(MULTI_TF_PER_BAR_FEATURES_V4),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
HTF_V4_MATRIX_CONTRACT = "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V2"
HTF_V4_CACHE_SCHEMA_VERSION = "htf_v4_disk_cache_manifest_v3"
HTF_V4_CACHE_BUILDER_VERSION = (
    "prebuild_multi_tf_cache_v4_closed_resample_smc_pivot_envelope_20260729"
)
HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION = (
    "htf_v4_full_input_liveness_v2"
)

MULTI_TF_RESAMPLE_RULES = {
    # Resample cadence only. Entry window lengths are explicit recipe inputs
    # and must form a strictly increasing wall-clock coverage pyramid.
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}
MULTI_TF_TIMEFRAMES = tuple(MULTI_TF_RESAMPLE_RULES)
MULTI_TF_TIMEFRAMES_LOWER = tuple(
    timeframe.lower() for timeframe in MULTI_TF_TIMEFRAMES
)
MULTI_TF_TIMEFRAMES_LOWER_M5_LAST = (
    *MULTI_TF_TIMEFRAMES_LOWER[1:],
    MULTI_TF_TIMEFRAMES_LOWER[0],
)
MULTI_TF_BARS_IN_M5 = {
    timeframe.lower(): int(
        pd.Timedelta(rule) / pd.Timedelta(MULTI_TF_RESAMPLE_RULES["M5"])
    )
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items()
}

# Pandas-Timedelta shift per TF: ensures we use only CLOSED bars at-or-before t
MULTI_TF_SHIFT = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
MULTI_TF_PYRAMID_SCHEMA_VERSION = "entry_multi_tf_causal_resolution_pyramid_v1"


def multi_tf_last_closed_label(
    decision_bar_start: pd.Timestamp | str,
    timeframe: str,
) -> pd.Timestamp:
    """Return the exact opening label of the last closed bar for one TF.

    ``decision_bar_start`` is the opening timestamp of an observed M5 candle.
    Its information becomes available five minutes later.  HTF resample labels
    are bar-opening timestamps, so the availability cutoff must be shifted by
    the full HTF duration and then floored to that timeframe's UTC grid.
    """
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(
            f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}"
        )
    timestamp = pd.Timestamp(decision_bar_start)
    if timestamp.tz is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_DECISION_TIMESTAMP_INVALID: timezone-aware UTC required"
        )
    return (
        timestamp
        + MULTI_TF_SHIFT["M5"]
        - MULTI_TF_SHIFT[timeframe]
    ).floor(MULTI_TF_RESAMPLE_RULES[timeframe])


def build_multi_tf_v4_closed_timestamp_indices(
    m5_index: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    """Derive the one admissible closed-bar timestamp axis for every V4 TF."""
    if not isinstance(m5_index, pd.DatetimeIndex) or len(m5_index) == 0:
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: non-empty "
            "DatetimeIndex required"
        )
    m5_index = m5_index.as_unit("ns")
    if (
        m5_index.tz is None
        or m5_index.hasnans
        or not m5_index.is_unique
        or not m5_index.is_monotonic_increasing
        or m5_index[0].utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: exact chronological "
            "unique UTC timestamps required"
        )
    if not m5_index.floor(MULTI_TF_RESAMPLE_RULES["M5"]).equals(m5_index):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: source timestamps "
            "must lie on the exact M5 UTC grid"
        )

    expected: dict[str, pd.DatetimeIndex] = {}
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items():
        labels = m5_index.floor(rule).drop_duplicates()
        last_closed = multi_tf_last_closed_label(m5_index[-1], timeframe)
        labels = labels[labels <= last_closed]
        if len(labels) and m5_index[0] > labels[0]:
            labels = labels[1:]
        if len(labels) == 0:
            raise RuntimeError(
                f"HTF_V4_NO_COMPLETE_RESAMPLED_BARS: {timeframe}"
            )
        expected[timeframe] = labels
    return expected


def require_multi_tf_resolution_pyramid(
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Validate explicit windows as strictly increasing wall-clock coverage."""
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(per_tf_seq_lens, dict) or tuple(per_tf_seq_lens) != expected_tfs:
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 declaration required"
        )
    lengths: dict[str, int] = {}
    coverage_seconds: dict[str, int] = {}
    for tf in expected_tfs:
        value = per_tf_seq_lens[tf]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"MULTI_TF_RESOLUTION_PYRAMID_LENGTH_INVALID: {tf}={value!r}"
            )
        lengths[tf] = int(value)
        coverage_seconds[tf] = int(
            value * MULTI_TF_SHIFT[tf].total_seconds()
        )
    spans = tuple(coverage_seconds.values())
    if any(left >= right for left, right in zip(spans, spans[1:])):
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_COVERAGE_INVALID: progressively "
            f"coarser timeframes must cover strictly older history; {coverage_seconds}"
        )
    payload: dict[str, object] = {
        "schema_version": MULTI_TF_PYRAMID_SCHEMA_VERSION,
        "timeframe_order": list(expected_tfs),
        "per_tf_seq_lens": lengths,
        "coverage_seconds": coverage_seconds,
        "strictly_increasing_wall_clock_coverage": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def build_multi_tf_v4_liveness_contract(
    features: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Prove every V4 field is finite, variable and non-duplicated on every TF."""

    if tuple(features) != tuple(MULTI_TF_RESAMPLE_RULES):
        raise RuntimeError(
            "HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 required"
        )
    failures: list[str] = []
    timeframe_rows: dict[str, object] = {}
    for tf_name in MULTI_TF_RESAMPLE_RULES:
        frame = features[tf_name]
        if (
            not isinstance(frame, pd.DataFrame)
            or tuple(frame.columns) != MULTI_TF_PER_BAR_FEATURES_V4
            or frame.attrs.get("htf_feature_contract")
            != HTF_V4_MATRIX_CONTRACT
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_SURFACE_INVALID: {tf_name}"
            )
        values = np.asarray(frame.attrs.get("feats_np"))
        warmup_rows = frame.attrs.get("causal_warmup_rows")
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape
            != (len(frame), MULTI_TF_FEATURE_COUNT_V4)
            or isinstance(warmup_rows, bool)
            or not isinstance(warmup_rows, (int, np.integer))
            or not 0 <= int(warmup_rows) < len(frame)
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_ARRAY_INVALID: {tf_name}"
            )
        warmup = int(warmup_rows)
        live = values[warmup:].astype(np.float64, copy=False)
        if not np.isfinite(live).all():
            failures.append(f"{tf_name}:nonfinite_post_warmup")
        field_stats: dict[str, object] = {}
        column_hash_owner: dict[str, str] = {}
        duplicate_pairs: list[list[str]] = []
        constant_fields: list[str] = []
        for index, feature_name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4):
            column = live[:, index]
            unique_count = int(np.unique(column).size)
            standard_deviation = float(np.std(column, dtype=np.float64))
            nonzero_fraction = float(np.mean(np.abs(column) > 1e-12))
            digest = hashlib.sha256(
                np.ascontiguousarray(column).view(np.uint8)
            ).hexdigest()
            if unique_count <= 1 or standard_deviation <= 0.0:
                constant_fields.append(feature_name)
            prior = column_hash_owner.get(digest)
            if prior is not None:
                duplicate_pairs.append([prior, feature_name])
            else:
                column_hash_owner[digest] = feature_name
            field_stats[feature_name] = {
                "unique_count": unique_count,
                "mean": float(np.mean(column, dtype=np.float64)),
                "std": standard_deviation,
                "minimum": float(np.min(column)),
                "maximum": float(np.max(column)),
                "nonzero_fraction": nonzero_fraction,
                "values_sha256": digest,
            }
        if constant_fields:
            failures.append(
                f"{tf_name}:constant_fields={constant_fields}"
            )
        if duplicate_pairs:
            failures.append(
                f"{tf_name}:exact_duplicate_fields={duplicate_pairs}"
            )
        timeframe_rows[tf_name] = {
            "rows": int(len(frame)),
            "warmup_rows": warmup,
            "live_rows": int(len(live)),
            "constant_fields": constant_fields,
            "exact_duplicate_pairs": duplicate_pairs,
            "fields": field_stats,
        }
    payload: dict[str, object] = {
        "schema_version": HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
        "matrix_contract": HTF_V4_MATRIX_CONTRACT,
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "timeframe_order": list(MULTI_TF_RESAMPLE_RULES),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "timeframes": timeframe_rows,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def require_multi_tf_v4_liveness_contract(
    value: object,
) -> dict[str, object]:
    """Validate the exact immutable V4 per-field/per-timeframe proof."""

    if not isinstance(value, dict):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_MISSING")
    expected_keys = {
        "schema_version",
        "matrix_contract",
        "feature_names_sha256",
        "timeframe_order",
        "decision",
        "failures",
        "timeframes",
        "contract_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_KEYS_INVALID")
    identity_payload = {
        key: item for key, item in value.items() if key != "contract_sha256"
    }
    expected_sha = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if value.get("contract_sha256") != expected_sha:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_IDENTITY_INVALID")
    if (
        value.get("schema_version")
        != HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION
        or value.get("matrix_contract") != HTF_V4_MATRIX_CONTRACT
        or value.get("feature_names_sha256")
        != MULTI_TF_FEATURE_NAMES_SHA256_V4
        or value.get("timeframe_order") != list(MULTI_TF_RESAMPLE_RULES)
        or value.get("decision") != "PASS"
        or value.get("failures") != []
    ):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_DECISION_INVALID")
    timeframes = value.get("timeframes")
    if not isinstance(timeframes, dict) or set(timeframes) != set(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID")
    expected_tf_keys = {
        "rows",
        "warmup_rows",
        "live_rows",
        "constant_fields",
        "exact_duplicate_pairs",
        "fields",
    }
    expected_stat_keys = {
        "unique_count",
        "mean",
        "std",
        "minimum",
        "maximum",
        "nonzero_fraction",
        "values_sha256",
    }
    for tf_name, row in timeframes.items():
        if not isinstance(row, dict) or set(row) != expected_tf_keys:
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_KEYS_INVALID: {tf_name}")
        rows = row.get("rows")
        warmup = row.get("warmup_rows")
        live_rows = row.get("live_rows")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or not 0 <= warmup < rows
            or live_rows != rows - warmup
            or row.get("constant_fields") != []
            or row.get("exact_duplicate_pairs") != []
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_DECISION_INVALID: {tf_name}")
        fields = row.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(
            MULTI_TF_PER_BAR_FEATURES_V4
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_FIELDS_INVALID: {tf_name}")
        for field_name, stats in fields.items():
            if not isinstance(stats, dict) or set(stats) != expected_stat_keys:
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_KEYS_INVALID: {tf_name}:{field_name}"
                )
            unique_count = stats.get("unique_count")
            numeric = [
                stats.get(name)
                for name in (
                    "mean",
                    "std",
                    "minimum",
                    "maximum",
                    "nonzero_fraction",
                )
            ]
            if (
                isinstance(unique_count, bool)
                or not isinstance(unique_count, int)
                or unique_count <= 1
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in numeric
                )
                or float(stats["std"]) <= 0.0
                or not 0.0 < float(stats["nonzero_fraction"]) <= 1.0
                or float(stats["minimum"]) > float(stats["maximum"])
                or not isinstance(stats.get("values_sha256"), str)
                or len(stats["values_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in stats["values_sha256"]
                )
            ):
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_INVALID: {tf_name}:{field_name}"
                )
    return value


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder-style RSI on close series. Returns Series indexed like close."""
    diff = close.diff()
    gain = diff.where(diff > 0, 0.0)
    loss = -diff.where(diff < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_per_bar_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Compute V12.2 v2 multi-TF per-bar features — all scale-invariant + clipped.

    Input: DataFrame with columns [open, high, low, close], DatetimeIndex.
    Output: DataFrame with MULTI_TF_PER_BAR_FEATURES columns.

    V12.2 v2 fixes vs v1:
    - Dropped raw OHLC (4150 dominated all other features)
    - Added scale-invariant alternatives: close_open_pct, high_low_atr
    - Centered RSI to [-1, 1] range
    - Winsorize all features to ±5 std to prevent outliers from poisoning training
      (some features had max=89,887 due to division-by-near-zero in low-vol periods)
    - Use realistic ATR floor (0.5% of close) instead of 1e-12
    """
    if not all(c in ohlc.columns for c in ("open", "high", "low", "close")):
        raise RuntimeError(f"compute_per_bar_features: missing OHLC cols in {list(ohlc.columns)}")
    df = ohlc[["open", "high", "low", "close"]].astype(np.float64).copy()
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    c = df["close"]
    o = df["open"]
    h = df["high"]
    low = df["low"]

    # Realistic ATR floor: 1 bps of close (= 0.01% of price)
    # Prevents division-by-near-zero outliers in low-volatility periods.
    atr14 = _atr(h, low, c, 14)
    atr_floor = np.maximum(c * 1e-4, 1e-3)   # at least 0.01% of close, min 0.001
    atr_safe = np.maximum(atr14, atr_floor)

    # Price-relative scale-invariant features
    out["close_open_pct"] = ((c - o) / np.maximum(o, 1e-6)).fillna(0.0)        # ≈ [-0.05, 0.05]
    out["high_low_atr"] = ((h - low) / atr_safe).fillna(0.0)                    # ≈ [0, 5]
    out["close_open_atr"] = ((c - o) / atr_safe).fillna(0.0)                    # ≈ [-3, 3]

    # Close-to-close returns (bps) — winsorize to ±500 bps (5%)
    for k in (1, 3, 5, 10):
        ret = ((c - c.shift(k)) / np.maximum(c.shift(k), 1e-6) * 1e4)
        out[f"ret_{k}"] = ret.clip(-500.0, 500.0).fillna(0.0)

    # ATR in bps
    out["atr_bps_14"] = (atr14 / np.maximum(c, 1e-6) * 1e4).clip(0, 500).fillna(0.0)

    # RSI centered to [-1, 1] (was [0, 100])
    rsi = _rsi(c, 14)
    out["rsi14_centered"] = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0).fillna(0.0)

    # Momentum in ATR units — winsorize to ±10
    for k in (1, 5, 20):
        delta = c - c.shift(k)
        out[f"mom_{k}_atr"] = (delta / atr_safe).clip(-10.0, 10.0).fillna(0.0)

    # Position in last 20-bar range
    rolling_high = h.rolling(20, min_periods=1).max()
    rolling_low = low.rolling(20, min_periods=1).min()
    span = np.maximum(rolling_high - rolling_low, atr_floor)
    out["range_pos_20"] = ((c - rolling_low) / span).clip(0.0, 1.0)

    # Body / wick fractions
    bar_range = np.maximum(h - low, atr_floor)
    body = (c - o).abs()
    upper_wick = h - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - low
    out["body_pct"] = (body / bar_range).clip(0.0, 1.0)
    out["upper_wick_pct"] = (upper_wick / bar_range).clip(0.0, 1.0)
    out["lower_wick_pct"] = (lower_wick / bar_range).clip(0.0, 1.0)

    # EMA20 distance in ATR units — winsorize to ±10
    ema20 = _ema(c, 20)
    out["ema20_dist_atr"] = ((c - ema20) / atr_safe).clip(-10.0, 10.0).fillna(0.0)

    return out[list(MULTI_TF_PER_BAR_FEATURES)]


# ─────────────────────────────────────────────────────────────────────────────
# V2 helpers + compute_per_bar_features_v2
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling N-bar VWAP from observed volume only."""
    if volume.isna().any() or (~np.isfinite(volume.to_numpy(dtype=np.float64))).any():
        raise RuntimeError("HTF_V2_VOLUME_SOURCE_INVALID: rolling VWAP volume is non-finite")
    if (volume <= 0.0).any():
        raise RuntimeError("HTF_V2_VOLUME_SOURCE_INVALID: rolling VWAP volume must be positive")
    pv = close * volume
    pv_sum = pv.rolling(window, min_periods=1).sum()
    v_sum = volume.rolling(window, min_periods=1).sum()
    return pv_sum / v_sum


def _session_vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP reset at each calendar day's midnight UTC."""
    if volume.isna().any() or (~np.isfinite(volume.to_numpy(dtype=np.float64))).any():
        raise RuntimeError("HTF_V2_VOLUME_SOURCE_INVALID: session VWAP volume is non-finite")
    if (volume <= 0.0).any():
        raise RuntimeError("HTF_V2_VOLUME_SOURCE_INVALID: session VWAP volume must be positive")
    pv = close * volume
    # Group by date — cumulative within day
    grp = close.index.normalize()
    pv_cs = pv.groupby(grp).cumsum()
    v_cs = volume.groupby(grp).cumsum()
    return pv_cs / v_cs


def _adx14(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
) -> pd.Series:
    """Welles Wilder's ADX with explicit causal warmup."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / np.maximum(plus_di + minus_di, 1e-12)
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean()
    adx.iloc[: 2 * n - 1] = np.nan
    return adx


def _regime_class(stack_aligned: pd.Series, ema200_slope: pd.Series, atr_safe: pd.Series) -> pd.Series:
    """Combine EMA-stack alignment + EMA200 slope into 5-class regime enum.

    0 = range (stack=0)
    1 = uptrend_low (stack=+1, slope <= +0.3 ATR)
    2 = uptrend_high (stack=+1, slope > +0.3 ATR)
    3 = downtrend_low (stack=-1, slope >= -0.3 ATR)
    4 = downtrend_high (stack=-1, slope < -0.3 ATR)
    """
    slope_atr = ema200_slope / atr_safe
    valid = stack_aligned.notna() & slope_atr.notna()
    out = pd.Series(np.nan, index=stack_aligned.index, dtype=np.float64)
    out.loc[valid] = 0.0
    up = valid & (stack_aligned == 1)
    down = valid & (stack_aligned == -1)
    out.loc[up] = np.where(slope_atr.loc[up] > 0.3, 2.0, 1.0)
    out.loc[down] = np.where(slope_atr.loc[down] < -0.3, 4.0, 3.0)
    return out


def _trend_age_bars(stack_aligned: pd.Series) -> pd.Series:
    """Number of consecutive bars since the EMA stack last changed sign."""
    # Convert to int sign sequence; count runs
    chg = (stack_aligned != stack_aligned.shift(1)).cumsum()
    return stack_aligned.groupby(chg).cumcount().astype(float)


def validate_causal_feature_matrix(
    values,
    *,
    expected_width: int,
    context: str,
) -> int:
    """Validate an exact feature matrix and return its warmup-prefix length.

    A model feature may be unavailable only in one chronological prefix. Once a
    complete row exists, every later row must be finite. Numeric sentinels are
    deliberately not introduced here.
    """
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[1] != int(expected_width) or arr.shape[0] == 0:
        raise RuntimeError(
            f"[{context}] feature matrix must have non-zero shape (N, {expected_width}); "
            f"observed={arr.shape}"
        )
    try:
        numeric = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[{context}] feature matrix must be numeric") from exc
    if np.isinf(numeric).any():
        raise RuntimeError(f"[{context}] feature matrix contains infinity")
    complete = np.isfinite(numeric).all(axis=1)
    if not complete.any():
        return int(len(numeric))
    first_complete = int(np.argmax(complete))
    if not complete[first_complete:].all():
        raise RuntimeError(
            f"[{context}] non-finite feature rows are not one causal warmup prefix"
        )
    return first_complete


def compute_per_bar_features_v2(ohlcv: pd.DataFrame, *,
                                feature_set: tuple = MULTI_TF_PER_BAR_FEATURES_V2) -> pd.DataFrame:
    """Compute the exact causal V2 per-bar feature contract from OHLCV.

    Initial indicator warmup remains NaN. Consumers must request a fully
    observed historical window; they may not replace missing history with a
    neutral numeric value.
    """
    requested = tuple(feature_set)
    if requested != MULTI_TF_PER_BAR_FEATURES_V2:
        raise RuntimeError(
            "HTF_V2_FEATURE_CONTRACT_MISMATCH: feature_set must be the exact V2 contract"
        )
    _validate_m5_input(ohlcv, require_volume=True)
    df = ohlcv[["open", "high", "low", "close", "volume"]].astype(np.float64).copy()
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    c = df["close"]
    o = df["open"]
    h = df["high"]
    low = df["low"]
    v = df["volume"]

    atr14 = _atr(h, low, c, 14)
    atr_floor = np.maximum(c * 1e-4, 1e-3)
    atr_safe = np.maximum(atr14, atr_floor)

    # ─── KEPT (9) ────────────────────────────────────────────────────
    out["atr_bps_14"] = (atr14 / c * 1e4).clip(0, 500)
    rsi = _rsi(c, 14)
    rsi.iloc[:14] = np.nan
    out["rsi14_centered"] = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    for k in (5, 20):
        delta = c - c.shift(k)
        out[f"mom_{k}_atr"] = (delta / atr_safe).clip(-10.0, 10.0)
    out["close_open_atr"] = ((c - o) / atr_safe).clip(-10.0, 10.0)
    bar_range = np.maximum(h - low, atr_floor)
    body = (c - o).abs()
    upper_wick = h - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - low
    out["body_pct"] = (body / bar_range).clip(0.0, 1.0)
    out["upper_wick_pct"] = (upper_wick / bar_range).clip(0.0, 1.0)
    out["lower_wick_pct"] = (lower_wick / bar_range).clip(0.0, 1.0)
    ema20 = _ema(c, 20)
    out["ema20_dist_atr"] = ((c - ema20) / atr_safe).clip(-10.0, 10.0)

    # ─── NEW: EMA stack (3 dist + 3 slopes) ──────────────────────────
    ema50 = _ema(c, 50)
    ema100 = _ema(c, 100)
    ema200 = _ema(c, 200)
    out["ema50_dist_atr"] = ((c - ema50) / atr_safe).clip(-15.0, 15.0)
    out["ema100_dist_atr"] = ((c - ema100) / atr_safe).clip(-20.0, 20.0)
    out["ema200_dist_atr"] = ((c - ema200) / atr_safe).clip(-30.0, 30.0)
    out["ema20_slope_atr"] = ((ema20 - ema20.shift(5)) / atr_safe).clip(-5.0, 5.0)
    out["ema50_slope_atr"] = ((ema50 - ema50.shift(5)) / atr_safe).clip(-5.0, 5.0)
    out["ema200_slope_atr"] = ((ema200 - ema200.shift(5)) / atr_safe).clip(-5.0, 5.0)
    # ─── NEW: EMA-stack regime (2) ───────────────────────────────────
    # 2026-05-24: now uses ffill'd EMAs above + checks full stack (50<100<200) for
    # stricter alignment. Previously bear/bull only checked 20<50<200, missing 100.
    bull = (ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)
    bear = (ema20 < ema50) & (ema50 < ema100) & (ema100 < ema200)
    stack = pd.Series(0, index=c.index)
    stack[bull] = 1
    stack[bear] = -1
    out["ema_stack_aligned_v2"] = stack.astype(float)
    out["regime_class_id"] = _regime_class(stack, ema200 - ema200.shift(5), atr_safe)

    # ─── NEW: VWAP family (4) ────────────────────────────────────────
    # 2026-05-24 FIX: session_vwap on D1 is degenerate (one bar per day → VWAP=close
    # → distance always 0). For TFs where bars span >= 1 day, use a 5-bar (weekly-ish)
    # rolling VWAP as a "session" proxy. Detect via index frequency.
    if len(c) >= 2:
        median_delta_hours = float(
            (c.index.to_series().diff().median() or pd.Timedelta(0)).total_seconds() / 3600.0
        )
    else:
        median_delta_hours = 0.0
    if median_delta_hours >= 23.0:  # D1 or coarser
        vwap_sess = _rolling_vwap(c, v, 5)  # 5-day VWAP as "session" proxy
    else:
        vwap_sess = _session_vwap(c, v)
    out["vwap_session_dist_atr"] = ((c - vwap_sess) / atr_safe).clip(-15.0, 15.0)
    vwap20 = _rolling_vwap(c, v, 20)
    out["vwap20_dist_atr"] = ((c - vwap20) / atr_safe).clip(-10.0, 10.0)
    vwap96 = _rolling_vwap(c, v, 96)
    out["vwap96_dist_atr"] = ((c - vwap96) / atr_safe).clip(-15.0, 15.0)
    out["vwap_session_slope_atr"] = ((vwap_sess - vwap_sess.shift(5)) / atr_safe).clip(-5.0, 5.0)

    # ─── NEW: Bollinger + trend strength (4) ─────────────────────────
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = (bb_upper - bb_lower)
    out["bb_position"] = ((c - bb_lower) / np.maximum(bb_width, atr_floor)).clip(0.0, 1.0)
    out["bb_width_atr"] = (bb_width / atr_safe).clip(0.0, 20.0)
    adx = _adx14(h, low, c, 14)
    out["adx_centered"] = ((adx - 25.0) / 25.0).clip(-1.0, 3.0)
    # log1p(bars_since_flip) / log1p(500) — normalized 0..1, saturates at 500 bars
    age = _trend_age_bars(stack).clip(upper=500.0)
    out["trend_age_bars_norm"] = np.log1p(age) / np.log1p(500.0)

    result = out[list(requested)].astype(np.float32)
    validate_causal_feature_matrix(
        result.to_numpy(copy=False),
        expected_width=len(requested),
        context="HTF_V2_CAUSAL_FEATURES",
    )
    return result


def compute_per_bar_features_v3(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """V2's 25 features plus candlestick patterns and swing structure.

    Both additions are pure functions of the bars themselves, so they carry the
    same meaning at every resolution and need none of the derived context the
    513-field specialist families depend on. Column order is exactly
    ``MULTI_TF_PER_BAR_FEATURES_V3``, whose first 25 entries are V2 unchanged.
    """
    from gx1.features.entry_candlestick_patterns_v1 import (
        build_entry_candlestick_pattern_layer,
    )
    from gx1.features.swing_structure_v1 import compute_swing_structure_features

    base = compute_per_bar_features_v2(ohlcv)

    frame = ohlcv[["open", "high", "low", "close"]].copy()
    frame.index.name = "time"
    candle_arr, candle_names = build_entry_candlestick_pattern_layer(
        frame.reset_index()
    )
    candle_arr = np.asarray(candle_arr, dtype=np.float64)
    if candle_arr.shape[0] != len(base):
        raise RuntimeError(
            "HTF_V3_CANDLE_ROW_MISMATCH: "
            f"candles={candle_arr.shape[0]} bars={len(base)}"
        )
    if len(candle_names) != len(MULTI_TF_PER_BAR_CANDLESTICK_V3):
        raise RuntimeError(
            "HTF_V3_CANDLE_WIDTH_MISMATCH: "
            f"got={len(candle_names)} expected={len(MULTI_TF_PER_BAR_CANDLESTICK_V3)}"
        )

    swing = compute_swing_structure_features(
        ohlcv["high"].to_numpy(dtype=np.float64),
        ohlcv["low"].to_numpy(dtype=np.float64),
        ohlcv["close"].to_numpy(dtype=np.float64),
    )

    out = base.copy()
    for column, values in zip(MULTI_TF_PER_BAR_CANDLESTICK_V3, candle_arr.T, strict=True):
        out[column] = values
    for column in MULTI_TF_PER_BAR_SWING_V3:
        key = column[len("swing_"):]
        if key not in swing:
            raise RuntimeError(f"HTF_V3_SWING_FIELD_MISSING: {key}")
        out[column] = np.asarray(swing[key], dtype=np.float64)

    out = out[list(MULTI_TF_PER_BAR_FEATURES_V3)]
    if tuple(out.columns) != MULTI_TF_PER_BAR_FEATURES_V3:
        raise RuntimeError("HTF_V3_COLUMN_ORDER_INVALID")
    return out


def compute_per_bar_features_v4(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return the exact all-eight-family causal surface for one resolution."""
    from gx1.features.smc_v1 import compute_smc_mtf_primitives_v1

    v3 = compute_per_bar_features_v3(ohlcv)
    base = v3.rename(
        columns={
            "vwap_session_dist_atr": "vwap_local_cycle_dist_atr",
            "vwap_session_slope_atr": "vwap_local_cycle_slope_atr",
        }
    )
    smc_source = ohlcv[["high", "low", "close"]].astype(np.float64).copy()
    smc_source["atr"] = _atr(
        smc_source["high"],
        smc_source["low"],
        smc_source["close"],
        14,
    )
    primitives = compute_smc_mtf_primitives_v1(smc_source)
    if not primitives.index.equals(base.index):
        raise RuntimeError("HTF_V4_SMC_ROW_AXIS_MISMATCH")

    out = pd.concat((base, primitives), axis=1)
    if tuple(out.columns) != MULTI_TF_PER_BAR_FEATURES_V4:
        raise RuntimeError(
            "HTF_V4_COLUMN_ORDER_INVALID: "
            f"observed={tuple(out.columns)}"
        )
    validate_causal_feature_matrix(
        out.to_numpy(dtype=np.float64, copy=False),
        expected_width=MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_CAUSAL_FEATURES",
    )
    return out.astype(np.float32)


def build_multi_tf_per_bar_features_v2(m5_df: pd.DataFrame) -> dict:
    """Build the exact causal V2 feature tables from observed M5 OHLCV.

    Resamples M5 → M5/M15/H1/H4/D1, computes V2 25-feature set per TF.
    Result attaches .attrs["ts_int64"] and .attrs["feats_np"] for fast slicing
    (same fast-path API as V1).
    """
    _validate_m5_input(m5_df, require_volume=True)
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlcv(m5_df, rule)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        feats = compute_per_bar_features_v2(resampled)
        ts_int64 = feats.index.asi8.astype(np.int64, copy=True)
        feats_np = feats.to_numpy(dtype=np.float32, copy=True)
        warmup_rows = validate_causal_feature_matrix(
            feats_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V2,
            context=f"HTF_V2_{tf_name}",
        )
        feats.attrs["ts_int64"] = ts_int64
        feats.attrs["feats_np"] = feats_np
        feats.attrs["causal_warmup_rows"] = warmup_rows
        feats.attrs["htf_feature_contract"] = HTF_V2_MATRIX_CONTRACT
        result[tf_name] = feats
    return result


def build_multi_tf_per_bar_features_v3(m5_df: pd.DataFrame) -> dict:
    """Build the exact causal V3 feature tables from observed M5 OHLCV.

    Resamples M5 → M5/M15/H1/H4/D1, computes V2 25-feature set per TF.
    Result attaches .attrs["ts_int64"] and .attrs["feats_np"] for fast slicing
    (same fast-path API as V1).
    """
    _validate_m5_input(m5_df, require_volume=True)
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlcv(m5_df, rule)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        feats = compute_per_bar_features_v3(resampled)
        ts_int64 = feats.index.asi8.astype(np.int64, copy=True)
        feats_np = feats.to_numpy(dtype=np.float32, copy=True)
        warmup_rows = validate_causal_feature_matrix(
            feats_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V3,
            context=f"HTF_V3_{tf_name}",
        )
        feats.attrs["ts_int64"] = ts_int64
        feats.attrs["feats_np"] = feats_np
        feats.attrs["causal_warmup_rows"] = warmup_rows
        feats.attrs["htf_feature_contract"] = HTF_V3_MATRIX_CONTRACT
        result[tf_name] = feats
    return result


def build_multi_tf_per_bar_features_v4(m5_df: pd.DataFrame) -> dict:
    """Build all eight causal specialist families at every declared timeframe."""
    _validate_m5_input(m5_df, require_volume=True)
    source = m5_df.copy(deep=False)
    source.index = source.index.as_unit("ns")
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(source.index)
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlcv(source, rule)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        expected_index = expected_indices[tf_name]
        if not resampled.index.is_unique or not expected_index.isin(
            resampled.index
        ).all():
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        resampled = resampled.loc[expected_index]
        if not resampled.index.equals(expected_index):
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        feats = compute_per_bar_features_v4(resampled)
        ts_int64 = feats.index.asi8.astype(np.int64, copy=True)
        feats_np = feats.to_numpy(dtype=np.float32, copy=True)
        warmup_rows = validate_causal_feature_matrix(
            feats_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"HTF_V4_{tf_name}",
        )
        feats.attrs["ts_int64"] = ts_int64
        feats.attrs["feats_np"] = feats_np
        feats.attrs["causal_warmup_rows"] = warmup_rows
        feats.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        result[tf_name] = feats
    return result


def attach_v2_mtf_per_bar_scalars(
    m5_df: "pd.DataFrame",
    target_ts_ns,
    per_tf_map,
    tfs=("m15", "h1", "h4", "d1"),
    skip=frozenset(),
) -> dict:
    """V2 (2026-06-04) ONE-TRUTH per-bar V2 multi-TF scalar projection.

    Shared by the live serve loader (v12_state_from_prebuilt._augment_cv3_with_v2_mtf_scalars) AND the
    V3 exit builder so train==serve by construction (was duplicated -> the build join drifted to a stale
    vintage matching serve only 88-95%). build_multi_tf_per_bar_features_v2(m5_df) -> for each TF uses
    ``M5 label + 5 minutes - TF duration`` (only-closed-bars searchsorted) onto target_ts_ns ->
    {tf_lower}_{live_frag}_v2 arrays.
    per_tf_map = [(live_frag, src_col), ...]. Returns dict[col ->
    np.ndarray(len(target_ts_ns), float64)]. Unavailable causal warmup remains
    NaN and must be trimmed by the owning state contract.
    """
    tf_feats = build_multi_tf_per_bar_features_v2(m5_df)
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    if target_ts_ns.ndim != 1 or len(target_ts_ns) == 0 or np.any(np.diff(target_ts_ns) <= 0):
        raise RuntimeError(
            "HTF_V2_PROJECTION_TARGET_INVALID: target timestamps must be non-empty, "
            "unique and chronological"
        )
    requested_tfs = tuple(str(name).lower() for name in tfs)
    unknown_tfs = [name for name in requested_tfs if name.upper() not in MULTI_TF_SHIFT]
    if unknown_tfs or len(set(requested_tfs)) != len(requested_tfs):
        raise RuntimeError(f"HTF_V2_PROJECTION_TF_INVALID: tfs={requested_tfs}")
    projection = tuple((str(live_frag), str(src_col)) for live_frag, src_col in per_tf_map)
    if not projection or len(set(projection)) != len(projection):
        raise RuntimeError("HTF_V2_PROJECTION_MAP_INVALID: projection map must be non-empty and unique")
    out: dict = {}
    for tf_lower in requested_tfs:
        tf_key = tf_lower.upper()
        tf_df = tf_feats[tf_key]
        tf_ts_ns = np.asarray(tf_df.attrs.get("ts_int64"), dtype=np.int64)
        values_np = np.asarray(tf_df.attrs.get("feats_np"))
        if tf_ts_ns.shape != (len(tf_df),) or np.any(np.diff(tf_ts_ns) <= 0):
            raise RuntimeError(f"HTF_V2_PROJECTION_SOURCE_INVALID: malformed {tf_key} timestamps")
        validate_causal_feature_matrix(
            values_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V2,
            context=f"HTF_V2_PROJECTION_{tf_key}",
        )
        decision_close_ns = target_ts_ns + int(MULTI_TF_SHIFT["M5"].value)
        cutoffs = decision_close_ns - int(MULTI_TF_SHIFT[tf_key].value)
        right = np.searchsorted(tf_ts_ns, cutoffs, side="right") - 1
        valid_mask = right >= 0
        safe_idx = np.clip(right, 0, len(tf_ts_ns) - 1)
        for live_frag, src_col in projection:
            if (tf_lower, live_frag) in skip:
                continue
            if src_col not in tf_df.columns:
                raise RuntimeError(
                    f"HTF_V2_PROJECTION_SOURCE_MISSING: {tf_key}.{src_col}"
                )
            values = tf_df[src_col].to_numpy(dtype=np.float64, copy=False)
            projected = np.full(len(target_ts_ns), np.nan, dtype=np.float64)
            projected[valid_mask] = values[safe_idx[valid_mask]]
            out[f"{tf_lower}_{live_frag}_v2"] = projected
    if not out:
        raise RuntimeError("HTF_V2_PROJECTION_EMPTY: projection produced no features")
    validate_causal_feature_matrix(
        np.column_stack(list(out.values())),
        expected_width=len(out),
        context="HTF_V2_PROJECTION",
    )
    return out


# ── REGIME_V4 per-TF V2 multi-TF scalar projection — ONE TRUTH ──────────────────
# The (live-fragment, source-col) projection + TF set + skip that produce the per-TF
# `{tf}_*_v2` scalars REGIME_V4 needs (R1 regime_class_id / R2 trend_age_bars_norm /
# R3 ema_stack_aligned, plus the mom/rsi/atr_bps/slope/lower_wick context). This is the
# 5-TF/9-feature REGIME version (m5 ADDED 2026-06-05, user vedtak: regime ALL-5) — NOT the
# older 4-TF/8-feature projection (retired; its module v12_live_features is deleted).
# The live serve loader (v12_state_from_prebuilt._V2_MTF_PER_TF imports THIS) AND the
# immutable snapshot path (BASE28 context recompute) and the offline build use
# these so admitted live regime columns are train==serve by construction.
REGIME_V4_V2_MTF_PER_TF = (
    ("ema20_slope_atr", "ema20_slope_atr"),
    ("ema_stack_aligned", "ema_stack_aligned_v2"),
    ("regime_class_id", "regime_class_id"),
    ("trend_age_bars_norm", "trend_age_bars_norm"),
    ("mom_5_atr", "mom_5_atr"),
    ("mom_20_atr", "mom_20_atr"),
    ("rsi14_centered", "rsi14_centered"),
    ("atr_bps_14", "atr_bps_14"),
    ("lower_wick_pct", "lower_wick_pct"),
)
REGIME_V4_V2_MTF_TFS = MULTI_TF_TIMEFRAMES_LOWER_M5_LAST
REGIME_V4_V2_MTF_SKIP = frozenset({("d1", "lower_wick_pct")})


def attach_default_regime_v4_v2_scalars(cv3: "pd.DataFrame") -> "pd.DataFrame":
    """Attach the per-TF V2 multi-TF scalars REGIME_V4 needs (its R1/R2/R3 inputs +
    context) to a cv3 frame IN PLACE, using the canonical REGIME_V4_V2_MTF_* constants.

    ONE TRUTH shared by the live serve loader, the build (add_ctx_cont), and the
    immutable snapshot owner — so admitted live regime columns are computed
    the same way as offline training context (no train≠serve or carry-forward freeze).
    Existing derived columns are overwritten from the exact source so stale or
    externally injected values cannot pass through this owner.
    """
    _validate_m5_input(cv3, require_volume=True)
    m5_df = cv3[["open", "high", "low", "close", "volume"]].astype(np.float64).copy()
    ts_ns = cv3.index.asi8.astype(np.int64, copy=False)
    for _col, _vals in attach_v2_mtf_per_bar_scalars(
        m5_df, ts_ns, REGIME_V4_V2_MTF_PER_TF, REGIME_V4_V2_MTF_TFS, REGIME_V4_V2_MTF_SKIP
    ).items():
        cv3[_col] = _vals
    return cv3


_HTF_V2_CACHE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "cache_identity_sha256",
        "feature_count",
        "feature_names",
        "shift_contract",
        "builder_version",
        "m5_prebuilt_source",
        "m5_prebuilt_source_sha256",
        "tfs",
    }
)
_HTF_V4_CACHE_MANIFEST_KEYS = (
    _HTF_V2_CACHE_MANIFEST_KEYS | {"full_input_liveness"}
)
_HTF_V2_CACHE_TF_KEYS = frozenset(
    {
        "n_bars",
        "feature_count",
        "feats_npy",
        "feats_npy_sha256",
        "feats_npy_size_bytes",
        "ts_npy",
        "ts_npy_sha256",
        "ts_npy_size_bytes",
        "first_ts_ns",
        "last_ts_ns",
        "causal_warmup_rows",
    }
)


class MultiTFV2DiskCache(dict):
    """Verified TF mapping with one content-bound disk-cache identity."""

    def __init__(
        self,
        *,
        cache_identity_sha256: str,
        manifest_sha256: str,
        m5_prebuilt_source: str,
        m5_prebuilt_source_sha256: str,
    ) -> None:
        super().__init__()
        self.cache_identity_sha256 = cache_identity_sha256
        self.manifest_sha256 = manifest_sha256
        self.m5_prebuilt_source = m5_prebuilt_source
        self.m5_prebuilt_source_sha256 = m5_prebuilt_source_sha256


def compute_htf_v2_cache_identity(manifest: dict) -> str:
    """Return the canonical identity for a manifest and all declared arrays."""

    identity_payload = dict(manifest)
    identity_payload.pop("cache_identity_sha256", None)
    try:
        encoded = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("HTF_V2_CACHE_MANIFEST_INVALID: non-canonical value") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _cache_path_has_symlink_component(path: Path) -> bool:
    absolute = path if path.is_absolute() else Path.cwd() / path
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _read_cache_file_bytes(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
    label: str,
) -> bytes:
    """Read one regular cache file once and verify those exact bytes.

    ``dir_fd`` pins the already-opened cache directory. ``O_NOFOLLOW`` prevents
    a manifest-named symlink from being resolved between inventory validation
    and open. The returned bytes are also the bytes passed to ``numpy.load``.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(f"HTF_V2_CACHE_FILE_INVALID: {label}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"HTF_V2_CACHE_FILE_INVALID: {label} is not regular")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed_size += len(chunk)
    finally:
        os.close(fd)
    if expected_size_bytes is not None and observed_size != expected_size_bytes:
        raise RuntimeError(
            f"HTF_V2_CACHE_SIZE_MISMATCH: {label} "
            f"observed={observed_size} expected={expected_size_bytes}"
        )
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"HTF_V2_CACHE_SHA256_MISMATCH: {label} "
            f"observed={observed_sha256} expected={expected_sha256}"
        )
    return b"".join(chunks)


def _exact_cache_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(
            f"HTF_V2_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError(
            f"HTF_V2_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    return value


def _exact_cache_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(
            f"HTF_V2_CACHE_CONTRACT_MISMATCH: {label} must be an exact integer"
        )
    observed = int(value)
    if observed < minimum:
        raise RuntimeError(
            f"HTF_V2_CACHE_CONTRACT_MISMATCH: {label}={observed} < {minimum}"
        )
    return observed


def _load_verified_cache_npy(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> np.ndarray:
    payload = _read_cache_file_bytes(
        directory_fd,
        name,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        label=label,
    )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"HTF_V2_CACHE_NPY_INVALID: {label}") from exc
    if not isinstance(loaded, np.ndarray):
        raise RuntimeError(f"HTF_V2_CACHE_NPY_INVALID: {label} is not an ndarray")
    return loaded


def load_multi_tf_v2_cache(cache_dir) -> dict:
    """Load one verified V2/V3/V4 cache through the retained cache owner.

    Returns the same dict shape as build_multi_tf_per_bar_features_v2(): one
    DataFrame per TF (M5/M15/H1/H4/D1) with .attrs["ts_int64"] and
    .attrs["feats_np"] populated for get_last_n_at_or_before fast-path.

    Saves the ~84s rebuild cost on every trainer launch.
    """
    supplied = Path(cache_dir).expanduser()
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if _cache_path_has_symlink_component(absolute):
        raise RuntimeError(
            f"HTF_V2_CACHE_PATH_INVALID: cache path traverses a symlink: {absolute}"
        )
    try:
        resolved_cache_dir = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"HTF_V2_CACHE_PATH_INVALID: {absolute}") from exc
    if not resolved_cache_dir.is_dir():
        raise RuntimeError(f"HTF_V2_CACHE_PATH_INVALID: {resolved_cache_dir}")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        directory_fd = os.open(resolved_cache_dir, directory_flags)
    except OSError as exc:
        raise RuntimeError(
            f"HTF_V2_CACHE_PATH_INVALID: {resolved_cache_dir}"
        ) from exc
    try:
        initial_inventory = set(os.listdir(directory_fd))
        if "manifest.json" not in initial_inventory:
            raise RuntimeError(
                f"HTF_V2_CACHE_MANIFEST_MISSING: {resolved_cache_dir / 'manifest.json'}"
            )
        manifest_bytes = _read_cache_file_bytes(
            directory_fd,
            "manifest.json",
            expected_sha256=None,
            expected_size_bytes=None,
            label="manifest.json",
        )
        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"HTF_V2_CACHE_MANIFEST_INVALID: {resolved_cache_dir / 'manifest.json'}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("HTF_V2_CACHE_MANIFEST_INVALID: root must be an object")
        schema_version = manifest.get("schema_version")
        expected_manifest_keys = (
            _HTF_V4_CACHE_MANIFEST_KEYS
            if schema_version == HTF_V4_CACHE_SCHEMA_VERSION
            else _HTF_V2_CACHE_MANIFEST_KEYS
        )
        if set(manifest) != expected_manifest_keys:
            raise RuntimeError(
                "HTF_V2_CACHE_CONTRACT_MISMATCH: manifest exact keys differ "
                f"missing={sorted(expected_manifest_keys - set(manifest))} "
                f"unexpected={sorted(set(manifest) - expected_manifest_keys)}"
            )
        expected_shift = {tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()}
        declared_cache_contracts = {
            HTF_V2_CACHE_SCHEMA_VERSION: (
                HTF_V2_MATRIX_CONTRACT,
                MULTI_TF_FEATURE_COUNT_V2,
                MULTI_TF_PER_BAR_FEATURES_V2,
                HTF_V2_CACHE_BUILDER_VERSION,
            ),
            HTF_V3_CACHE_SCHEMA_VERSION: (
                HTF_V3_MATRIX_CONTRACT,
                MULTI_TF_FEATURE_COUNT_V3,
                MULTI_TF_PER_BAR_FEATURES_V3,
                HTF_V3_CACHE_BUILDER_VERSION,
            ),
            HTF_V4_CACHE_SCHEMA_VERSION: (
                HTF_V4_MATRIX_CONTRACT,
                MULTI_TF_FEATURE_COUNT_V4,
                MULTI_TF_PER_BAR_FEATURES_V4,
                HTF_V4_CACHE_BUILDER_VERSION,
            ),
        }
        if schema_version not in declared_cache_contracts:
            raise RuntimeError(
                "HTF_V2_CACHE_CONTRACT_MISMATCH: unknown schema_version "
                f"{schema_version!r}"
            )
        matrix_contract, feature_width, feature_names, builder_version = (
            declared_cache_contracts[schema_version]
        )
        contracts = {
            "schema_version": schema_version,
            "builder_version": builder_version,
            "feature_count": feature_width,
            "feature_names": list(feature_names),
            "shift_contract": expected_shift,
        }
        for name, expected in contracts.items():
            if manifest.get(name) != expected:
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {name} observed={manifest.get(name)!r} "
                    f"expected={expected!r}"
                )
        source_path = Path(str(manifest.get("m5_prebuilt_source") or "")).expanduser()
        if not source_path.is_absolute():
            raise RuntimeError(
                "HTF_V2_CACHE_CONTRACT_MISMATCH: m5_prebuilt_source must be absolute"
            )
        m5_prebuilt_source_sha256 = _exact_cache_sha256(
            manifest["m5_prebuilt_source_sha256"],
            label="m5_prebuilt_source_sha256",
        )
        cache_identity_sha256 = _exact_cache_sha256(
            manifest["cache_identity_sha256"],
            label="cache_identity_sha256",
        )
        computed_cache_identity = compute_htf_v2_cache_identity(manifest)
        if cache_identity_sha256 != computed_cache_identity:
            raise RuntimeError(
                "HTF_V2_CACHE_IDENTITY_MISMATCH: "
                f"observed={cache_identity_sha256} expected={computed_cache_identity}"
            )
        tf_manifest = manifest.get("tfs")
        if not isinstance(tf_manifest, dict) or tuple(tf_manifest) != tuple(
            MULTI_TF_RESAMPLE_RULES
        ):
            raise RuntimeError(
                "HTF_V2_CACHE_CONTRACT_MISMATCH: ordered exact "
                "M5/M15/H1/H4/D1 entries required"
            )
        declared_inventory = {"manifest.json"}
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            if not isinstance(info, dict) or set(info) != _HTF_V2_CACHE_TF_KEYS:
                observed_keys = set(info) if isinstance(info, dict) else set()
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name} exact keys differ "
                    f"missing={sorted(_HTF_V2_CACHE_TF_KEYS - observed_keys)} "
                    f"unexpected={sorted(observed_keys - _HTF_V2_CACHE_TF_KEYS)}"
                )
            feats_name = str(info["feats_npy"])
            ts_name = str(info["ts_npy"])
            expected_names = (f"{tf_name}_feats.npy", f"{tf_name}_ts.npy")
            if (feats_name, ts_name) != expected_names:
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name} filenames "
                    f"observed={(feats_name, ts_name)!r} expected={expected_names!r}"
                )
            declared_inventory.update((feats_name, ts_name))
        if initial_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V2_CACHE_INVENTORY_MISMATCH: "
                f"missing={sorted(declared_inventory - initial_inventory)} "
                f"unexpected={sorted(initial_inventory - declared_inventory)}"
            )

        out = MultiTFV2DiskCache(
            cache_identity_sha256=cache_identity_sha256,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            m5_prebuilt_source=str(source_path),
            m5_prebuilt_source_sha256=m5_prebuilt_source_sha256,
        )
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            n_bars = _exact_cache_int(
                info["n_bars"], label=f"{tf_name}.n_bars", minimum=1
            )
            feature_count = _exact_cache_int(
                info["feature_count"],
                label=f"{tf_name}.feature_count",
                minimum=1,
            )
            if feature_count != feature_width:
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name}.feature_count "
                    f"observed={feature_count} expected={feature_width}"
                )
            feats_size = _exact_cache_int(
                info["feats_npy_size_bytes"],
                label=f"{tf_name}.feats_npy_size_bytes",
                minimum=1,
            )
            ts_size = _exact_cache_int(
                info["ts_npy_size_bytes"],
                label=f"{tf_name}.ts_npy_size_bytes",
                minimum=1,
            )
            feats_np = _load_verified_cache_npy(
                directory_fd,
                str(info["feats_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["feats_npy_sha256"],
                    label=f"{tf_name}.feats_npy_sha256",
                ),
                expected_size_bytes=feats_size,
                label=f"{tf_name}.feats_npy",
            )
            ts_int64 = _load_verified_cache_npy(
                directory_fd,
                str(info["ts_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["ts_npy_sha256"],
                    label=f"{tf_name}.ts_npy_sha256",
                ),
                expected_size_bytes=ts_size,
                label=f"{tf_name}.ts_npy",
            )
            if (
                feats_np.dtype != np.dtype(np.float32)
                or ts_int64.dtype != np.dtype(np.int64)
            ):
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name} requires "
                    "float32 features/int64 timestamps"
                )
            if feats_np.shape != (n_bars, feature_width):
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name} feature shape "
                    f"observed={feats_np.shape} "
                    f"expected={(n_bars, feature_width)}"
                )
            if ts_int64.shape != (n_bars,) or np.any(np.diff(ts_int64) <= 0):
                raise RuntimeError(
                    f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name} timestamps invalid"
                )
            warmup_rows = validate_causal_feature_matrix(
                feats_np,
                expected_width=feature_width,
                context=f"HTF_V2_CACHE_{tf_name}",
            )
            if warmup_rows == len(feats_np):
                raise RuntimeError(
                    f"HTF_V2_CACHE_WARMUP_INCOMPLETE: {tf_name} has no complete row"
                )
            expected_meta = {
                "n_bars": n_bars,
                "feature_count": feature_width,
                "first_ts_ns": int(ts_int64[0]),
                "last_ts_ns": int(ts_int64[-1]),
                "causal_warmup_rows": warmup_rows,
            }
            for name, expected in expected_meta.items():
                observed = _exact_cache_int(
                    info[name],
                    label=f"{tf_name}.{name}",
                    minimum=0,
                )
                if observed != expected:
                    raise RuntimeError(
                        f"HTF_V2_CACHE_CONTRACT_MISMATCH: {tf_name}.{name} "
                        f"observed={observed!r} expected={expected!r}"
                    )
            # Reconstruct minimal DataFrame (only index + attrs matter for fast-path).
            idx = pd.DatetimeIndex(ts_int64.astype("datetime64[ns]"), tz="UTC")
            df = pd.DataFrame(
                np.empty((len(idx), feats_np.shape[1]), dtype=np.float32),
                index=idx,
                columns=feature_names,
            )
            df.attrs["ts_int64"] = np.ascontiguousarray(ts_int64)
            df.attrs["feats_np"] = np.ascontiguousarray(feats_np)
            df.attrs["causal_warmup_rows"] = warmup_rows
            df.attrs["htf_feature_contract"] = matrix_contract
            out[tf_name] = df
        if schema_version == HTF_V4_CACHE_SCHEMA_VERSION:
            try:
                require_multi_tf_v4_liveness_contract(
                    manifest.get("full_input_liveness")
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
                ) from exc
            observed_liveness = build_multi_tf_v4_liveness_contract(out)
            if (
                observed_liveness.get("decision") != "PASS"
                or manifest.get("full_input_liveness")
                != observed_liveness
            ):
                raise RuntimeError(
                    "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
                )
        final_inventory = set(os.listdir(directory_fd))
        if final_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V2_CACHE_INVENTORY_CHANGED_DURING_LOAD: "
                f"missing={sorted(declared_inventory - final_inventory)} "
                f"unexpected={sorted(final_inventory - declared_inventory)}"
            )
        return out
    finally:
        os.close(directory_fd)


# Contract-neutral public name for active Entry callers.  The historical V2
# function name remains as a compatibility surface for immutable V2/V3
# research readers, while active model-native paths validate and require V4.
load_multi_tf_cache = load_multi_tf_v2_cache


def build_multi_tf_per_bar_features(m5_df: pd.DataFrame) -> dict:
    """Resample M5 → H1/H4/D1 and compute per-bar features for each TF.

    Input: M5 DataFrame with DatetimeIndex (UTC) and [open, high, low, close].
    Output: {"H1": DataFrame, "H4": DataFrame, "D1": DataFrame} — each indexed
    by that TF's bar-close timestamp, columns from MULTI_TF_PER_BAR_FEATURES.

    V12.2 perf: each DataFrame has `.attrs["ts_int64"]` (sorted timestamps as
    int64 ns) and `.attrs["feats_np"]` ((N, 19) float32 array) attached. These
    let get_last_n_at_or_before() use O(log N) searchsorted instead of
    O(N) pandas .loc — ~100× per-slice speedup, critical for training where
    we slice 60k samples × 5 TFs × N epochs.
    """
    _validate_m5_input(m5_df)
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlc(m5_df, rule)
        # Drop rows with any NaN OHLC (gaps from weekends/holidays don't have full bars)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        feats = compute_per_bar_features(resampled)
        # Pre-compute fast-path arrays (V12.2 perf optimization)
        feats.attrs["ts_int64"] = feats.index.values.astype("datetime64[ns]").astype(np.int64)
        feats.attrs["feats_np"] = feats.fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        result[tf_name] = feats
    return result


def get_last_n_at_or_before(
    feats: pd.DataFrame, target_ts: pd.Timestamp, n: int, tf_shift: pd.Timedelta,
) -> np.ndarray:
    """Slice the last `n` per-bar feature rows whose close-time is <= (target_ts - tf_shift).

    Returns an exact finite ``(n, n_features)`` float32 array. Missing history,
    indicator warmup, malformed cache metadata, and non-finite evidence are hard
    errors; this owner never pads or substitutes a neutral value.

    `tf_shift` enforces the "only closed bars" invariant: e.g. for H1, target=12:35
    means we use H1 bars closing at-or-before 11:35 (the 11:00 H1 bar, since
    12:00 H1 bar hasn't closed yet at 12:35).

    V12.2 fast path: when `feats.attrs["ts_int64"]` and `feats.attrs["feats_np"]`
    are present (set by build_multi_tf_per_bar_features), we use numpy
    searchsorted on int64 timestamps — ~100× faster than pandas .loc.
    """
    if not isinstance(feats, pd.DataFrame) or feats.empty:
        raise RuntimeError("HTF_WINDOW_SOURCE_MISSING: exact non-empty feature table required")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise RuntimeError(f"HTF_WINDOW_LENGTH_INVALID: n={n!r}")
    n = int(n)
    if not isinstance(tf_shift, pd.Timedelta) or tf_shift <= pd.Timedelta(0):
        raise RuntimeError(f"HTF_WINDOW_SHIFT_INVALID: tf_shift={tf_shift!r}")
    target = pd.Timestamp(target_ts)
    if target.tzinfo is None or target.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("HTF_WINDOW_TARGET_INVALID: target_ts must be timezone-aware UTC")
    declared_contract = feats.attrs.get("htf_feature_contract")
    exact_contracts = {
        HTF_V2_MATRIX_CONTRACT: MULTI_TF_PER_BAR_FEATURES_V2,
        HTF_V3_MATRIX_CONTRACT: MULTI_TF_PER_BAR_FEATURES_V3,
        HTF_V4_MATRIX_CONTRACT: MULTI_TF_PER_BAR_FEATURES_V4,
    }
    if (
        declared_contract not in exact_contracts
        or tuple(feats.columns) != tuple(exact_contracts[declared_contract])
    ):
        raise RuntimeError(
            "HTF_WINDOW_SOURCE_CONTRACT_MISSING: refusing unknown or "
            "order-mismatched feature table"
        )

    ts_int64 = np.asarray(feats.attrs.get("ts_int64"))
    feats_np = np.asarray(feats.attrs.get("feats_np"))
    width = int(feats.shape[1])
    if (
        ts_int64.dtype != np.dtype(np.int64)
        or ts_int64.shape != (len(feats),)
        or feats_np.dtype != np.dtype(np.float32)
        or feats_np.shape != (len(feats), width)
    ):
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: malformed exact cache arrays")
    warmup_rows = feats.attrs.get("causal_warmup_rows")
    if (
        isinstance(warmup_rows, bool)
        or not isinstance(warmup_rows, (int, np.integer))
        or not 0 <= int(warmup_rows) <= len(feats)
    ):
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: causal warmup metadata missing")

    cutoff_ns = int(target.value) - int(tf_shift.value)
    right = int(np.searchsorted(ts_int64, cutoff_ns, side="right"))
    if right < n:
        raise RuntimeError(
            f"HTF_WINDOW_HISTORY_INSUFFICIENT: need={n} closed_rows={right} target={target.isoformat()}"
        )
    left = right - n
    if left < int(warmup_rows):
        raise RuntimeError(
            f"HTF_WINDOW_WARMUP_INCOMPLETE: first_row={left} warmup_rows={int(warmup_rows)}"
        )
    tail = np.asarray(feats_np[left:right], dtype=np.float32)
    if tail.shape != (n, width) or not np.isfinite(tail).all():
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: selected feature evidence is non-finite")
    return np.ascontiguousarray(tail)


def require_multi_tf_decision_window_coverage(
    features: dict[str, pd.DataFrame],
    *,
    per_tf_seq_lens: dict[str, int],
    decision_times_by_split: dict[str, object],
) -> dict[str, object]:
    """Prove the exact TF pyramid is sliceable at every split boundary.

    This pre-training check calls the same closed-bar slicer as the dataset and
    includes each resolution's real causal warmup. It therefore cannot confuse
    the 96-row M5 signal sequence with the independently declared MTF windows.
    """

    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(features, dict) or tuple(features) != expected_tfs:
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_FEATURE_SET_INVALID: exact ordered "
            "M5/M15/H1/H4/D1 cache required"
        )
    if (
        not isinstance(decision_times_by_split, dict)
        or tuple(decision_times_by_split) != ("train", "val", "test")
    ):
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_SPLIT_SET_INVALID: exact ordered "
            "train/val/test decision times required"
        )

    split_bounds: dict[str, dict[str, object]] = {}
    boundary_times: list[tuple[str, str, pd.Timestamp]] = []
    for split, raw_times in decision_times_by_split.items():
        try:
            times = pd.DatetimeIndex(
                pd.to_datetime(raw_times, utc=True, errors="raise")
            )
        except Exception as exc:
            raise RuntimeError(
                f"MULTI_TF_DECISION_COVERAGE_TIME_INVALID: {split}"
            ) from exc
        if (
            times.empty
            or times.hasnans
            or not times.is_monotonic_increasing
            or not times.is_unique
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TIME_INVALID: "
                f"{split} must be non-empty, unique and chronological"
            )
        first = pd.Timestamp(times[0])
        last = pd.Timestamp(times[-1])
        split_bounds[split] = {
            "rows": int(len(times)),
            "first_utc": first.isoformat(),
            "last_utc": last.isoformat(),
        }
        boundary_times.extend(
            ((split, "first", first), (split, "last", last))
        )

    target_availability_shift = pd.Timedelta(minutes=5)
    per_tf: dict[str, object] = {}
    for tf in expected_tfs:
        frame = features[tf]
        n = int(per_tf_seq_lens[tf])
        boundary_rows: dict[str, object] = {}
        for split, edge, target in boundary_times:
            availability = target + target_availability_shift
            try:
                window = get_last_n_at_or_before(
                    frame,
                    availability,
                    n=n,
                    tf_shift=MULTI_TF_SHIFT[tf],
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_UNAVAILABLE: "
                    f"{split}.{edge}/{tf} target={target.isoformat()} "
                    f"seq_len={n}: {exc}"
                ) from exc
            boundary_rows[f"{split}_{edge}"] = {
                "target_utc": target.isoformat(),
                "window_sha256": hashlib.sha256(
                    np.ascontiguousarray(window, dtype="<f4").tobytes()
                ).hexdigest(),
            }
        per_tf[tf] = {
            "seq_len": n,
            "coverage_seconds": pyramid["coverage_seconds"][tf],
            "causal_warmup_rows": int(frame.attrs["causal_warmup_rows"]),
            "boundaries": boundary_rows,
        }

    payload: dict[str, object] = {
        "schema_version": "entry_multi_tf_decision_window_coverage_v1",
        "target_availability_shift_minutes": 5,
        "resolution_pyramid": pyramid,
        "split_bounds": split_bounds,
        "per_tf": per_tf,
        "all_split_boundaries_sliceable": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return require_multi_tf_decision_window_coverage_metadata(
        payload,
        per_tf_seq_lens=per_tf_seq_lens,
    )


def require_multi_tf_decision_window_coverage_metadata(
    value: Mapping[str, object],
    *,
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Strictly validate the immutable split-boundary coverage proof."""

    expected_keys = {
        "schema_version",
        "target_availability_shift_minutes",
        "resolution_pyramid",
        "split_bounds",
        "per_tf",
        "all_split_boundaries_sliceable",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_KEYS_INVALID")
    payload = dict(value)
    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    if (
        payload["schema_version"]
        != "entry_multi_tf_decision_window_coverage_v1"
        or payload["target_availability_shift_minutes"] != 5
        or payload["resolution_pyramid"] != pyramid
        or payload["all_split_boundaries_sliceable"] is not True
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_INVALID")
    split_bounds = payload["split_bounds"]
    if not isinstance(split_bounds, dict) or tuple(split_bounds) != (
        "train",
        "val",
        "test",
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID")
    parsed_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, raw in split_bounds.items():
        if not isinstance(raw, dict) or set(raw) != {
            "rows",
            "first_utc",
            "last_utc",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
            )
        rows = raw["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
            )
        first = pd.Timestamp(raw["first_utc"])
        last = pd.Timestamp(raw["last_utc"])
        if (
            first.tzinfo is None
            or last.tzinfo is None
            or first.utcoffset() != pd.Timedelta(0)
            or last.utcoffset() != pd.Timedelta(0)
            or first > last
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
            )
        parsed_bounds[split] = (first, last)
    per_tf = payload["per_tf"]
    if not isinstance(per_tf, dict) or tuple(per_tf) != tuple(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID")
    expected_boundaries = tuple(
        f"{split}_{edge}"
        for split in ("train", "val", "test")
        for edge in ("first", "last")
    )
    for tf, raw in per_tf.items():
        if not isinstance(raw, dict) or set(raw) != {
            "seq_len",
            "coverage_seconds",
            "causal_warmup_rows",
            "boundaries",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        warmup = raw["causal_warmup_rows"]
        if (
            raw["seq_len"] != per_tf_seq_lens[tf]
            or raw["coverage_seconds"] != pyramid["coverage_seconds"][tf]
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or warmup < 0
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        boundaries = raw["boundaries"]
        if not isinstance(boundaries, dict) or tuple(boundaries) != (
            expected_boundaries
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
            )
        for boundary, row in boundaries.items():
            if not isinstance(row, dict) or set(row) != {
                "target_utc",
                "window_sha256",
            }:
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                )
            split, edge = boundary.rsplit("_", 1)
            expected_target = parsed_bounds[split][0 if edge == "first" else 1]
            if (
                pd.Timestamp(row["target_utc"]) != expected_target
                or not isinstance(row["window_sha256"], str)
                or len(row["window_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in row["window_sha256"]
                )
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                )
    observed_hash = payload.pop("contract_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_HASH_INVALID")
    payload["contract_sha256"] = observed_hash
    return payload
