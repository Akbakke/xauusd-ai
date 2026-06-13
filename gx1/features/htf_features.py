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


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """V2-extension: also aggregates volume (sum) — needed for VWAP per TF."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": _last_valid, "volume": "sum"}
    cols = [c for c in agg.keys() if c in df.columns]
    return df[cols].resample(rule).agg({c: agg[c] for c in cols}).dropna(how="all")


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
# Vectorized full-tape HTF (ONE TRUTH for the offline ctx builder + the live daemon)
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
    """Vectorized no-lookahead align: each M5 bar gets the value of the last HTF bar
    that CLOSED at or before it (an HTF bar opened at T closes at T+shift). Mirrors
    add_ctx_cont_columns_to_prebuilt._align_last_closed and v12_ctx_augment_live._align_last_closed."""
    shifted = htf_series.copy()
    shifted.index = shifted.index + shift
    return shifted.reindex(target_index, method="ffill")


def build_htf_tape(m5_candles: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 HTF features for EVERY M5 bar (vectorized full-tape), no lookahead.

    ONE TRUTH for the offline ctx builder (add_ctx_cont_columns_to_prebuilt) AND the live
    canonical_incremental daemon, so live serves the SAME fresh HTF the training
    distribution was built with — fixing the daemon forward-fill / frozen-HTF bug (e.g. the
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

# ─────────────────────────────────────────────────────────────────────────────
# V3 (2026-06-10): V2 + ema20_50_diff_atr — the LEARNED EMA20×EMA50 cross/distance
# (user ask: "ema20 som krysser ema50"). Signed gap = which EMA is on top (cross STATE),
# |mag| = separation, zero-crossing = the cross EVENT; flips EARLIER than ema_stack_aligned_v2
# (which needs the full 20<50<100<200 stack) → earlier dip/recovery signal. DEFINED ONLY here +
# co-exists with V2 (compute_per_bar_features_v2(..., feature_set=V3) returns 26). NOT yet wired
# into any trainer/cache (V2 stays mandatory) — the revival retrain flips it (25→26 cascade:
# MTF cache → V10 → candidates → V3; warm-start the per-TF *_proj). Pure-additive until then.
# ─────────────────────────────────────────────────────────────────────────────
MULTI_TF_PER_BAR_FEATURES_V3 = MULTI_TF_PER_BAR_FEATURES_V2 + ("ema20_50_diff_atr",)
MULTI_TF_FEATURE_COUNT_V3 = len(MULTI_TF_PER_BAR_FEATURES_V3)   # = 26

MULTI_TF_RESAMPLE_RULES = {
    # V10 (entry) base is M5 → uses M15+H1+H4+D1.
    # V3 (exit) base is M1 → uses M5+M15+H1+H4+D1 (M5 added below).
    "M5": "5min",     # 96 bars = 8h — micro momentum (V3 only — V10 has M5 as base)
    "M15": "15min",   # 96 bars = 24h — intraday momentum
    "H1": "1h",       # 96 bars = 4 days
    "H4": "4h",       # 96 bars = 16 days
    "D1": "1D",       # 96 bars = ~3 months
}

# Pandas-Timedelta shift per TF: ensures we use only CLOSED bars at-or-before t
MULTI_TF_SHIFT = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}


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
    l = df["low"]

    # Realistic ATR floor: 1 bps of close (= 0.01% of price)
    # Prevents division-by-near-zero outliers in low-volatility periods.
    atr14 = _atr(h, l, c, 14)
    atr_floor = np.maximum(c * 1e-4, 1e-3)   # at least 0.01% of close, min 0.001
    atr_safe = np.maximum(atr14, atr_floor)

    # Price-relative scale-invariant features
    out["close_open_pct"] = ((c - o) / np.maximum(o, 1e-6)).fillna(0.0)        # ≈ [-0.05, 0.05]
    out["high_low_atr"] = ((h - l) / atr_safe).fillna(0.0)                      # ≈ [0, 5]
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
    rolling_low = l.rolling(20, min_periods=1).min()
    span = np.maximum(rolling_high - rolling_low, atr_floor)
    out["range_pos_20"] = ((c - rolling_low) / span).clip(0.0, 1.0)

    # Body / wick fractions
    bar_range = np.maximum(h - l, atr_floor)
    body = (c - o).abs()
    upper_wick = h - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - l
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
    """Rolling N-bar VWAP. Uses 1.0 volume when zero/missing (degenerate guard)."""
    v = volume.fillna(0).where(volume > 0, 1.0)
    pv = close * v
    pv_sum = pv.rolling(window, min_periods=1).sum()
    v_sum = v.rolling(window, min_periods=1).sum()
    return pv_sum / np.maximum(v_sum, 1e-12)


def _session_vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP reset at each calendar day's midnight UTC."""
    v = volume.fillna(0).where(volume > 0, 1.0)
    pv = close * v
    # Group by date — cumulative within day
    grp = close.index.normalize()
    pv_cs = pv.groupby(grp).cumsum()
    v_cs = v.groupby(grp).cumsum()
    return pv_cs / np.maximum(v_cs, 1e-12)


def _adx14(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    """Welles Wilder's ADX. Returns 0..100 series indexed like c."""
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm = pd.Series(plus_dm, index=c.index)
    minus_dm = pd.Series(minus_dm, index=c.index)
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / np.maximum(plus_di + minus_di, 1e-12)
    return dx.ewm(alpha=1.0/n, adjust=False).mean().fillna(0.0)


def _regime_class(stack_aligned: pd.Series, ema200_slope: pd.Series, atr_safe: pd.Series) -> pd.Series:
    """Combine EMA-stack alignment + EMA200 slope into 5-class regime enum.

    0 = range (stack=0)
    1 = uptrend_low (stack=+1, slope <= +0.3 ATR)
    2 = uptrend_high (stack=+1, slope > +0.3 ATR)
    3 = downtrend_low (stack=-1, slope >= -0.3 ATR)
    4 = downtrend_high (stack=-1, slope < -0.3 ATR)
    """
    slope_atr = ema200_slope / np.maximum(atr_safe, 1e-9)
    out = pd.Series(0, index=stack_aligned.index, dtype=int)
    out[stack_aligned == 1] = np.where(slope_atr[stack_aligned == 1] > 0.3, 2, 1)
    out[stack_aligned == -1] = np.where(slope_atr[stack_aligned == -1] < -0.3, 4, 3)
    return out


def _trend_age_bars(stack_aligned: pd.Series) -> pd.Series:
    """Number of consecutive bars since the EMA stack last changed sign."""
    # Convert to int sign sequence; count runs
    chg = (stack_aligned != stack_aligned.shift(1)).cumsum()
    return stack_aligned.groupby(chg).cumcount().astype(float)


def compute_per_bar_features_v2(ohlcv: pd.DataFrame, *,
                                feature_set: tuple = MULTI_TF_PER_BAR_FEATURES_V2) -> pd.DataFrame:
    """V2 multi-TF per-bar features — 25 cols. Requires OHLCV (volume needed for VWAP).

    Backward compat: if volume missing, falls back to volume=1 (tick-equal-weight VWAP).
    """
    if not all(c in ohlcv.columns for c in ("open", "high", "low", "close")):
        raise RuntimeError(f"compute_per_bar_features_v2: missing OHLC cols")
    df = ohlcv[["open", "high", "low", "close"]].astype(np.float64).copy()
    if "volume" in ohlcv.columns:
        df["volume"] = pd.to_numeric(ohlcv["volume"], errors="coerce").fillna(0).astype(np.float64)
    else:
        df["volume"] = 1.0   # tick-equal-weight fallback
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    c = df["close"]; o = df["open"]; h = df["high"]; l = df["low"]; v = df["volume"]

    atr14 = _atr(h, l, c, 14)
    atr_floor = np.maximum(c * 1e-4, 1e-3)
    atr_safe = np.maximum(atr14, atr_floor)

    # ─── KEPT (9) ────────────────────────────────────────────────────
    out["atr_bps_14"] = (atr14 / np.maximum(c, 1e-6) * 1e4).clip(0, 500).fillna(0.0)
    rsi = _rsi(c, 14)
    out["rsi14_centered"] = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0).fillna(0.0)
    # 2026-05-24 FIX: ffill close before shift so D1/H4 with sparse warmup don't
    # produce zeros from NaN propagation. Previously D1 mom_20 was DEAD because
    # shift(20) on sparse D1 series → NaN → fillna(0) → constant zero column.
    c_ffilled = c.ffill()
    for k in (5, 20):
        delta = c_ffilled - c_ffilled.shift(k)
        out[f"mom_{k}_atr"] = (delta / atr_safe).clip(-10.0, 10.0).fillna(0.0)
    out["close_open_atr"] = ((c - o) / atr_safe).fillna(0.0).clip(-10.0, 10.0)
    bar_range = np.maximum(h - l, atr_floor)
    body = (c - o).abs()
    upper_wick = h - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - l
    out["body_pct"] = (body / bar_range).clip(0.0, 1.0)
    out["upper_wick_pct"] = (upper_wick / bar_range).clip(0.0, 1.0)
    out["lower_wick_pct"] = (lower_wick / bar_range).clip(0.0, 1.0)
    # 2026-05-24 PM: ffill ema20 too (used in stack check line 603). Audit caught
    # that only ema50/100/200 were ffill'd in Bug 1 fix, leaving ema20 NaN-vulnerable
    # in early warmup → stack-check evaluates NaN>x as False → bull=0 silent.
    ema20 = _ema(c, 20).ffill()
    out["ema20_dist_atr"] = ((c - ema20) / atr_safe).clip(-10.0, 10.0).fillna(0.0)

    # ─── NEW: EMA stack (3 dist + 3 slopes) ──────────────────────────
    # 2026-05-24 FIX: ffill EMA series before stack-check. Without ffill, NaN in
    # any EMA (early warmup, especially on resampled D1/H4 with <200 bars history)
    # silently dropped stack_aligned to 0 → dead feature on D1/H4/M15.
    ema50 = _ema(c, 50).ffill()
    ema100 = _ema(c, 100).ffill()
    ema200 = _ema(c, 200).ffill()
    out["ema50_dist_atr"] = ((c - ema50) / atr_safe).clip(-15.0, 15.0).fillna(0.0)
    out["ema100_dist_atr"] = ((c - ema100) / atr_safe).clip(-20.0, 20.0).fillna(0.0)
    out["ema200_dist_atr"] = ((c - ema200) / atr_safe).clip(-30.0, 30.0).fillna(0.0)
    out["ema20_slope_atr"] = ((ema20 - ema20.shift(5)) / atr_safe).clip(-5.0, 5.0).fillna(0.0)
    out["ema50_slope_atr"] = ((ema50 - ema50.shift(5)) / atr_safe).clip(-5.0, 5.0).fillna(0.0)
    out["ema200_slope_atr"] = ((ema200 - ema200.shift(5)) / atr_safe).clip(-5.0, 5.0).fillna(0.0)
    # V3 (additive): EMA20×EMA50 cross/distance in ATR units. Always computed (cheap — both EMAs are
    # already above); only RETURNED when feature_set=V3 (else dropped by the V2 column filter below).
    out["ema20_50_diff_atr"] = ((ema20 - ema50) / atr_safe).clip(-15.0, 15.0).fillna(0.0)

    # ─── NEW: EMA-stack regime (2) ───────────────────────────────────
    # 2026-05-24: now uses ffill'd EMAs above + checks full stack (50<100<200) for
    # stricter alignment. Previously bear/bull only checked 20<50<200, missing 100.
    bull = (ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)
    bear = (ema20 < ema50) & (ema50 < ema100) & (ema100 < ema200)
    stack = pd.Series(0, index=c.index)
    stack[bull] = 1
    stack[bear] = -1
    out["ema_stack_aligned_v2"] = stack.astype(float)
    out["regime_class_id"] = _regime_class(stack, ema200 - ema200.shift(5), atr_safe).astype(float)

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
    out["vwap_session_dist_atr"] = ((c - vwap_sess) / atr_safe).clip(-15.0, 15.0).fillna(0.0)
    vwap20 = _rolling_vwap(c, v, 20)
    out["vwap20_dist_atr"] = ((c - vwap20) / atr_safe).clip(-10.0, 10.0).fillna(0.0)
    vwap96 = _rolling_vwap(c, v, 96)
    out["vwap96_dist_atr"] = ((c - vwap96) / atr_safe).clip(-15.0, 15.0).fillna(0.0)
    out["vwap_session_slope_atr"] = ((vwap_sess - vwap_sess.shift(5)) / atr_safe).clip(-5.0, 5.0).fillna(0.0)

    # ─── NEW: Bollinger + trend strength (4) ─────────────────────────
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std().fillna(0.0)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = (bb_upper - bb_lower)
    out["bb_position"] = ((c - bb_lower) / np.maximum(bb_width, atr_floor)).clip(0.0, 1.0).fillna(0.5)
    out["bb_width_atr"] = (bb_width / atr_safe).clip(0.0, 20.0).fillna(0.0)
    adx = _adx14(h, l, c, 14)
    out["adx_centered"] = ((adx - 25.0) / 25.0).clip(-1.0, 3.0).fillna(0.0)
    # log1p(bars_since_flip) / log1p(500) — normalized 0..1, saturates at 500 bars
    age = _trend_age_bars(stack).clip(upper=500.0)
    out["trend_age_bars_norm"] = (np.log1p(age) / np.log1p(500.0)).fillna(0.0)

    return out[list(feature_set)].astype(np.float32)


def build_multi_tf_per_bar_features_v2(m5_df: pd.DataFrame) -> dict:
    """V2 multi-TF builder. Requires OHLCV (will tick-equal-weight if volume missing).

    Resamples M5 → M5/M15/H1/H4/D1, computes V2 25-feature set per TF.
    Result attaches .attrs["ts_int64"] and .attrs["feats_np"] for fast slicing
    (same fast-path API as V1).
    """
    _validate_m5_input(m5_df)
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlcv(m5_df, rule)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        feats = compute_per_bar_features_v2(resampled)
        feats.attrs["ts_int64"] = feats.index.values.astype("datetime64[ns]").astype(np.int64)
        feats.attrs["feats_np"] = feats.fillna(0.0).to_numpy(dtype=np.float32, copy=True)
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
    vintage matching serve only 88-95%). build_multi_tf_per_bar_features_v2(m5_df) -> for each TF asof-shift
    via MULTI_TF_SHIFT (only-closed-bars searchsorted) onto target_ts_ns -> {tf_lower}_{live_frag}_v2 arrays.
    per_tf_map = [(live_frag, src_col), ...]. Returns dict[col -> np.ndarray(len(target_ts_ns), float64)];
    0.0 where no closed TF bar exists yet.
    """
    tf_feats = build_multi_tf_per_bar_features_v2(m5_df)
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    out: dict = {}
    for tf_lower in tfs:
        tf_key = tf_lower.upper()
        tf_df = tf_feats.get(tf_key)
        if tf_df is None or len(tf_df) == 0:
            continue
        tf_ts_ns = tf_df.index.values.astype("datetime64[ns]").astype(np.int64)
        cutoffs = target_ts_ns - int(MULTI_TF_SHIFT[tf_key].value)
        right = np.searchsorted(tf_ts_ns, cutoffs, side="right") - 1
        valid_mask = right >= 0
        safe_idx = np.clip(right, 0, len(tf_ts_ns) - 1)
        for live_frag, src_col in per_tf_map:
            if (tf_lower, live_frag) in skip or src_col not in tf_df.columns:
                continue
            values = tf_df[src_col].to_numpy(dtype=np.float64)
            out[f"{tf_lower}_{live_frag}_v2"] = np.where(valid_mask, values[safe_idx], 0.0)
    return out


# ── REGIME_V4 per-TF V2 multi-TF scalar projection — ONE TRUTH ──────────────────
# The (live-fragment, source-col) projection + TF set + skip that produce the per-TF
# `{tf}_*_v2` scalars REGIME_V4 needs (R1 regime_class_id / R2 trend_age_bars_norm /
# R3 ema_stack_aligned, plus the mom/rsi/atr_bps/slope/lower_wick context). This is the
# 5-TF/9-feature REGIME version (m5 ADDED 2026-06-05, user vedtak: regime ALL-5) — NOT the
# older 4-TF/8-feature XGB-v7 projection in v12_live_features.V2_MTF_PER_TF_FEATURES.
# The live serve loader (v12_state_from_prebuilt._V2_MTF_PER_TF imports THIS) AND the
# canonical_incremental daemon (BASE34 ctx recompute) AND the build (add_ctx_cont) all use
# these so the daemon-maintained BASE34 regime cols are train==serve by construction.
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
REGIME_V4_V2_MTF_TFS = ("m15", "h1", "h4", "d1", "m5")
REGIME_V4_V2_MTF_SKIP = frozenset({("d1", "lower_wick_pct")})


def attach_default_regime_v4_v2_scalars(cv3: "pd.DataFrame") -> "pd.DataFrame":
    """Attach the per-TF V2 multi-TF scalars REGIME_V4 needs (its R1/R2/R3 inputs +
    context) to a cv3 frame IN PLACE, using the canonical REGIME_V4_V2_MTF_* constants.

    ONE TRUTH shared by the live serve loader, the build (add_ctx_cont), and the
    canonical_incremental daemon — so the daemon-maintained BASE34 regime columns are
    computed the SAME way the training BASE34 was (no train≠serve, no carry-forward freeze).
    Idempotent: returns unchanged if the scalars are already present. Fail-soft on missing
    OHLC (returns unchanged) so a degenerate frame never crashes the live append; the
    REGIME_V4 block downstream is itself fail-closed if its inputs are still absent.
    """
    probe = f"{REGIME_V4_V2_MTF_TFS[0]}_{REGIME_V4_V2_MTF_PER_TF[0][0]}_v2"
    if probe in cv3.columns:
        return cv3
    ohlc = ["open", "high", "low", "close"]
    if any(c not in cv3.columns for c in ohlc):
        return cv3
    m5_df = cv3[ohlc].astype(np.float64).copy()
    m5_df["volume"] = (
        cv3["volume"].astype(np.float64) if "volume" in cv3.columns else 1.0
    )
    ts_ns = cv3.index.values.astype("datetime64[ns]").astype(np.int64)
    for _col, _vals in attach_v2_mtf_per_bar_scalars(
        m5_df, ts_ns, REGIME_V4_V2_MTF_PER_TF, REGIME_V4_V2_MTF_TFS, REGIME_V4_V2_MTF_SKIP
    ).items():
        cv3[_col] = _vals
    return cv3


def load_multi_tf_v2_cache(cache_dir) -> dict:
    """Load a pre-built V2 multi-TF cache (see scripts/prebuild_multi_tf_cache_v2.py).

    Returns the same dict shape as build_multi_tf_per_bar_features_v2(): one
    DataFrame per TF (M5/M15/H1/H4/D1) with .attrs["ts_int64"] and
    .attrs["feats_np"] populated for get_last_n_at_or_before fast-path.

    Saves the ~84s rebuild cost on every trainer launch.
    """
    import json
    from pathlib import Path as _P
    cache_dir = _P(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    out = {}
    for tf_name, info in manifest["tfs"].items():
        feats_np = np.load(cache_dir / info["feats_npy"], mmap_mode="r")
        ts_int64 = np.load(cache_dir / info["ts_npy"], mmap_mode="r")
        # Reconstruct minimal DataFrame (only index + attrs matter for fast-path).
        idx = pd.DatetimeIndex(ts_int64.astype("datetime64[ns]"), tz="UTC")
        # NOTE: keep DataFrame columns minimal — fast-path uses attrs only.
        df = pd.DataFrame(np.empty((len(idx), feats_np.shape[1]), dtype=np.float32), index=idx)
        df.attrs["ts_int64"] = np.ascontiguousarray(ts_int64).astype(np.int64, copy=False)
        df.attrs["feats_np"] = np.ascontiguousarray(feats_np).astype(np.float32, copy=False)
        out[tf_name] = df
    return out


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

    Returns: (n, n_features) float32 array. If fewer than n bars are available
    (warmup not satisfied), the result is left-padded with zeros.

    `tf_shift` enforces the "only closed bars" invariant: e.g. for H1, target=12:35
    means we use H1 bars closing at-or-before 11:35 (the 11:00 H1 bar, since
    12:00 H1 bar hasn't closed yet at 12:35).

    V12.2 fast path: when `feats.attrs["ts_int64"]` and `feats.attrs["feats_np"]`
    are present (set by build_multi_tf_per_bar_features), we use numpy
    searchsorted on int64 timestamps — ~100× faster than pandas .loc.
    """
    # V2: infer pad dim from feats so this works for both V1 (17) and V2 (25).
    pad_dim = MULTI_TF_FEATURE_COUNT
    if feats is not None and len(feats) > 0:
        pad_dim = int(feats.shape[1])

    if feats is None or len(feats) == 0:
        return np.zeros((n, pad_dim), dtype=np.float32)

    # Fast path: pre-computed int64 timestamps + numpy feature array
    ts_int64 = feats.attrs.get("ts_int64")
    feats_np = feats.attrs.get("feats_np")
    if ts_int64 is not None and feats_np is not None:
        target_ns = pd.Timestamp(target_ts).value
        shift_ns = int(tf_shift.value)
        cutoff_ns = target_ns - shift_ns
        right = np.searchsorted(ts_int64, cutoff_ns, side="right")
        if right == 0:
            return np.zeros((n, pad_dim), dtype=np.float32)
        left = max(0, right - n)
        tail = feats_np[left:right]
        if tail.shape[0] < n:
            pad = np.zeros((n - tail.shape[0], pad_dim), dtype=np.float32)
            tail = np.concatenate([pad, tail], axis=0)
        return tail

    # Fallback: pandas .loc (slower) — for old caches without attrs
    cutoff = pd.Timestamp(target_ts) - tf_shift
    eligible = feats.loc[feats.index <= cutoff]
    if eligible.empty:
        return np.zeros((n, pad_dim), dtype=np.float32)
    tail = eligible.tail(n).fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    if tail.shape[0] < n:
        pad = np.zeros((n - tail.shape[0], pad_dim), dtype=np.float32)
        tail = np.concatenate([pad, tail], axis=0)
    return tail
