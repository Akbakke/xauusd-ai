#!/usr/bin/env python3
"""V12 live ctx-augmentation — adds the 32 features that XGB v5 / V10 v3
need on top of canonical_v3.

Background: XGB v5 was trained against a prebuilt parquet that contained
canonical_v3 columns + ~32 augmented ctx-cont / ctx-cat features computed
by `add_ctx_cont_columns_to_prebuilt.py`. Today the canonical_v3 prebuilt
on disk has been regenerated without those augmentations, so any live
inference must compute them from scratch.

Features added (32 total):

  Spread / ATR derivations (2):
    - atr_bps                              # canonical_v2 atr / mid * 1e4
    - spread_bps                           # (ask - bid) / bid * 1e4

  Session features (5) — see gx1.time.session_detector for the SSoT:
    - session_id                           # 0/1/2/3 = ASIA/EU/OVERLAP/US
    - minutes_since_session_open
    - minutes_to_next_session_boundary
    - session_change_flag                  # 1 if session changed vs prev bar
    - session_tradable                     # 1 if session_id != 0 (not ASIA)

  Session flags (3) — for backward-compat with _v1_is_* shifted columns
  that basic_v1.py only computes when a 'ts' column is present:
    - is_ASIA
    - _v1_is_EU                            # is_EU shifted by 1 bar
    - _v1_is_US                            # is_US shifted by 1 bar

  Session-interaction (3) — products with _v1_is_US:
    - _v1_int_ema_us                       # _v1_ema_diff * _v1_is_US
    - _v1_int_range_us                     # _v1_range_z * _v1_is_US
    - _v1_int_slope_h1_us                  # _v1h1_slope3 * _v1_is_US

  Regime / bucket categoricals (4):
    - trend_regime_id                      # 0/1/2 from price_vs_ema50_atr
    - vol_regime_id                        # 0..4 percentile rank of atr_bps
    - atr_bucket                           # = vol_regime_id
    - spread_bucket                        # 0..4 percentile rank of spread_bps

  H4 trend sign (1):
    - H4_trend_sign_cat                    # 0/1/2 from sign(H4 mid - H4 EMA50)

  HTF derivations (4):
    - D1_dist_from_ema200_atr              # (D1 mid - D1 EMA200) / D1 ATR14
    - H1_range_compression_ratio           # H1 ATR14 / H1 ATR100
    - D1_atr_percentile_252                # rolling 252-day percentile of D1 ATR14
    - M15_range_compression_ratio          # M15 ATR14 / M15 ATR100

  Microstructure on M5 close (5):
    - micro_momentum_3                     # close - close.shift(3)
    - micro_momentum_5                     # close - close.shift(5)
    - micro_acceleration                   # diff of diff
    - wick_ratio                           # (high - close) / range
    - distance_ema_fast                    # close - EMA5

  Swing structure (5):
    - dist_last_swing_high_atr             # (close - last pivot-high) / ATR14
    - dist_last_swing_low_atr              # (close - last pivot-low)  / ATR14
    - bars_since_swing_high
    - bars_since_swing_low
    - retracement_from_last_impulse        # 0..1 retracement

Caveat: regime/bucket percentile ranks are computed over the live
lookback window (default 45 days) rather than the full training
distribution. Tree models (XGB) are tolerant of this; the resulting
bucket assignments are within ±1 of the training-distribution buckets
in normal vol regimes.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from gx1.time.session_detector import (
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
)

LOG = logging.getLogger("v12_ctx_augment_live")

ATR_EPS = 1e-9
SWING_ATR_PERIOD = 14


# ── HTF resampling helpers ────────────────────────────────────────────────


def _resample_ohlc(df_m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M5 OHLC to a higher timeframe (1H/4H/15min/1D).

    Input df_m5 must be DatetimeIndex'd with columns open/high/low/close.
    Returns DataFrame with same columns, indexed at the start of each HTF bar.
    """
    out = pd.DataFrame({
        "open":  df_m5["open"].resample(rule).first(),
        "high":  df_m5["high"].resample(rule).max(),
        "low":   df_m5["low"].resample(rule).min(),
        "close": df_m5["close"].resample(rule).last(),
    }).dropna()
    return out


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=max(2, n // 2)).mean()


def _align_last_closed(target_idx: pd.DatetimeIndex,
                        htf_series: pd.Series,
                        shift: pd.Timedelta) -> pd.Series:
    """For each M5 timestamp, return the value of the last fully-closed HTF
    bar (no lookahead). The HTF bar at time T closes at T + shift.
    """
    shifted = htf_series.copy()
    shifted.index = shifted.index + shift
    aligned = shifted.reindex(target_idx, method="ffill")
    return aligned


def _rank_bucket_0_4(x: np.ndarray, fallback: int) -> np.ndarray:
    """0..4 bucket via percentile rank across the input array."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    s = pd.Series(x)
    q = s.rank(pct=True, method="average").to_numpy(dtype=float)
    if not np.isfinite(q).any():
        return np.full(len(x), int(fallback), dtype=np.int64)
    b = np.clip(q * 5.0, 0.0, 4.99).astype(np.int64)
    b = np.where(np.isfinite(b), b, int(fallback)).astype(np.int64)
    return b


# ── per-feature-group computations ────────────────────────────────────────


def _add_session_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3: adds session_id, is_ASIA, _v1_is_EU, _v1_is_US,
    minutes_since/to, session_change_flag, session_tradable."""
    idx = cv3.index
    cv3["session_id"] = get_session_id_vectorized(idx).astype(np.int64)
    cv3["is_ASIA"] = (cv3["session_id"] == 0).astype(np.int64)
    cv3["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(idx).astype(np.float32)
    cv3["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(idx).astype(np.float32)
    sess_tag = get_session_vectorized(idx)
    cv3["session_change_flag"] = (sess_tag != sess_tag.shift(1)).fillna(False).astype(np.int64)
    cv3["session_tradable"] = (cv3["session_id"] != 0).astype(np.int64)
    # _v1_is_EU / _v1_is_US: shifted by 1 bar (no-lookahead). Same convention
    # as basic_v1.py:984/988.
    is_eu = (sess_tag == "EU").astype(np.float64)
    is_us = (sess_tag == "US").astype(np.float64)
    is_eu_shifted = np.roll(is_eu.to_numpy(), 1); is_eu_shifted[0] = 0.0
    is_us_shifted = np.roll(is_us.to_numpy(), 1); is_us_shifted[0] = 0.0
    cv3["_v1_is_EU"] = is_eu_shifted
    cv3["_v1_is_US"] = is_us_shifted


def _add_session_interactions(cv3: pd.DataFrame) -> None:
    """Mutates cv3: _v1_int_ema_us, _v1_int_range_us, _v1_int_slope_h1_us.

    Requires _v1_is_US already set, plus _v1_ema_diff / _v1_range_z /
    _v1h1_slope3 from canonical_v2.
    """
    is_us = cv3["_v1_is_US"].to_numpy(dtype=np.float64)
    if "_v1_ema_diff" in cv3.columns:
        cv3["_v1_int_ema_us"] = cv3["_v1_ema_diff"].to_numpy(dtype=np.float64) * is_us
    if "_v1_range_z" in cv3.columns:
        cv3["_v1_int_range_us"] = cv3["_v1_range_z"].to_numpy(dtype=np.float64) * is_us
    if "_v1h1_slope3" in cv3.columns:
        cv3["_v1_int_slope_h1_us"] = cv3["_v1h1_slope3"].to_numpy(dtype=np.float64) * is_us
    # XGB base80-contract aliases: canonical_v3 renamed 4 features but the XGB
    # base80 contract still expects the legacy names. They are EXACT duplicates
    # (see materialize_canonical_v3_augment.py duplicate-pair list) — alias them
    # so the serving augmenter feeds XGB the same inputs as training. One truth:
    # raw canonical_v3 (e.g. CANONICAL_V3_FULL) lacks these; FULL_PLUS_CTX baked
    # them in. Guarded so pre-baked frames are untouched.
    for _dst, _src in (
        ("_v1_body_tr", "_v1_body_share_1"),
        ("_v1_int_clv_atr", "_v1_clv"),
        ("_v1_int_r5_atr", "_v1_r5"),
        ("_v1_int_slope_h4_atr", "_v1h4_slope5"),
    ):
        if _dst not in cv3.columns and _src in cv3.columns:
            cv3[_dst] = cv3[_src].to_numpy(dtype=np.float64)


def _add_spread_atr_bps(cv3: pd.DataFrame) -> None:
    """Mutates cv3: atr_bps, spread_bps."""
    # atr_bps = canonical_v2 atr / mid * 1e4. canonical_v3 has _v1_atr14 (not 'atr', which was pruned).
    atr_col = "_v1_atr14" if "_v1_atr14" in cv3.columns else "atr"
    if atr_col in cv3.columns and "close" in cv3.columns:
        atr = cv3[atr_col].astype(float).to_numpy()
        mid = cv3["close"].astype(float).to_numpy()
        cv3["atr_bps"] = (atr / np.maximum(mid, ATR_EPS)) * 1e4
    else:
        cv3["atr_bps"] = 0.0
    # spread_bps = (ask - bid) / bid * 1e4
    if "bid_close" in cv3.columns and "ask_close" in cv3.columns:
        bid = cv3["bid_close"].astype(float).to_numpy()
        ask = cv3["ask_close"].astype(float).to_numpy()
        cv3["spread_bps"] = (ask - bid) / np.maximum(bid, ATR_EPS) * 1e4
    else:
        cv3["spread_bps"] = 0.0


def _add_htf_features(cv3: pd.DataFrame, df_m5: pd.DataFrame) -> None:
    """Mutates cv3: D1_dist_from_ema200_atr, H1_range_compression_ratio,
    D1_atr_percentile_252, M15_range_compression_ratio, H4_trend_sign_cat."""
    # Ensure df_m5 is DatetimeIndex'd
    m5 = df_m5.copy()
    if "time" in m5.columns and not isinstance(m5.index, pd.DatetimeIndex):
        m5["time"] = pd.to_datetime(m5["time"], utc=True)
        m5 = m5.set_index("time")

    # D1 features
    df_d1 = _resample_ohlc(m5, "1D")
    if len(df_d1) >= 14:
        d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
        d1_ema200 = _ema(d1_mid, 200)
        d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
        d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)
        cv3["D1_dist_from_ema200_atr"] = _align_last_closed(
            cv3.index, d1_dist, pd.Timedelta(days=1)
        ).fillna(0.0).to_numpy(dtype=float)
        # D1_atr_percentile_252
        if len(df_d1) >= 30:  # need at least some history; 252 is target
            window = min(252, len(df_d1))
            def _pctl_last(arr):
                a = np.asarray(arr, dtype=float)
                if not np.isfinite(a).all():
                    return float("nan")
                return float((a <= a[-1]).mean())
            atr_pctl = d1_atr14.rolling(window, min_periods=max(2, window // 4)).apply(_pctl_last, raw=True).ffill()
            cv3["D1_atr_percentile_252"] = _align_last_closed(
                cv3.index, atr_pctl, pd.Timedelta(days=1)
            ).fillna(0.5).to_numpy(dtype=float)
        else:
            cv3["D1_atr_percentile_252"] = 0.5
    else:
        cv3["D1_dist_from_ema200_atr"] = 0.0
        cv3["D1_atr_percentile_252"] = 0.5

    # H1 features
    df_h1 = _resample_ohlc(m5, "1H")
    if len(df_h1) >= 14:
        h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
        h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
        h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)
        cv3["H1_range_compression_ratio"] = _align_last_closed(
            cv3.index, h1_comp, pd.Timedelta(hours=1)
        ).fillna(1.0).to_numpy(dtype=float)
    else:
        cv3["H1_range_compression_ratio"] = 1.0

    # M15 features
    df_m15 = _resample_ohlc(m5, "15min")
    if len(df_m15) >= 14:
        m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
        m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill()
        m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
        cv3["M15_range_compression_ratio"] = _align_last_closed(
            cv3.index, m15_comp, pd.Timedelta(minutes=15)
        ).fillna(1.0).to_numpy(dtype=float)
    else:
        cv3["M15_range_compression_ratio"] = 1.0

    # H4 trend sign categorical
    df_h4 = _resample_ohlc(m5, "4H")
    if len(df_h4) >= 50:
        h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
        h4_ema50 = _ema(h4_mid, 50)
        diff = (h4_mid - h4_ema50).to_numpy(dtype=float)
        sign = np.sign(np.where(np.isfinite(diff), diff, 0.0)).astype(np.int64)
        sign_cat = (sign + 1).astype(np.int64)  # {-1,0,+1} → {0,1,2}
        sign_series = pd.Series(sign_cat, index=df_h4.index, dtype="int64")
        h4_aligned = _align_last_closed(cv3.index, sign_series, pd.Timedelta(hours=4))
        cv3["H4_trend_sign_cat"] = h4_aligned.fillna(1).astype(np.int64).to_numpy()
    else:
        cv3["H4_trend_sign_cat"] = 1


def _add_micro_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3: micro_momentum_3/5, micro_acceleration, wick_ratio,
    distance_ema_fast. Computed on M5 close/high/low (already in cv3)."""
    eps = 1e-9
    close = cv3["close"].astype(float)
    high = cv3["high"].astype(float)
    low = cv3["low"].astype(float)
    cv3["micro_momentum_3"] = (close - close.shift(3)).fillna(0.0).astype(np.float32)
    cv3["micro_momentum_5"] = (close - close.shift(5)).fillna(0.0).astype(np.float32)
    cv3["micro_acceleration"] = (
        (close - close.shift(1)) - (close.shift(1) - close.shift(2))
    ).fillna(0.0).astype(np.float32)
    cv3["wick_ratio"] = ((high - close) / (high - low + eps)).astype(np.float32)
    ema_fast = close.ewm(span=5, adjust=False).mean()
    cv3["distance_ema_fast"] = (close - ema_fast).astype(np.float32)


def _add_swing_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3: dist_last_swing_high/low_atr, bars_since_swing_high/low,
    retracement_from_last_impulse. Swing pivots are 5-bar lookahead-symmetric
    (i.e. high[i] is a pivot-high iff it exceeds high[i±1] and high[i±2])."""
    eps = 1e-9
    close = cv3["close"].astype(float)
    high = cv3["high"].astype(float)
    low = cv3["low"].astype(float)

    # ATR for the period
    prev_close = close.shift(1).fillna(close)
    tr = np.maximum(
        (high - low).abs(),
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr = tr.rolling(window=SWING_ATR_PERIOD, min_periods=1).mean()
    atr_safe = atr.clip(lower=eps).to_numpy()

    # Pivot detection (looks 2 bars forward + 2 bars back → uses future data
    # within the lookback slice, NOT a leak for live since we only emit features
    # for fully-closed M5 bars that already have 2 future bars).
    h_arr = high.to_numpy()
    l_arr = low.to_numpy()
    n = len(h_arr)
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(2, n - 2):
        if (h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i-2] and
                h_arr[i] > h_arr[i+1] and h_arr[i] > h_arr[i+2]):
            pivot_high[i] = True
        if (l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i-2] and
                l_arr[i] < l_arr[i+1] and l_arr[i] < l_arr[i+2]):
            pivot_low[i] = True

    last_high_vals = np.empty(n, dtype=np.float64)
    last_low_vals = np.empty(n, dtype=np.float64)
    last_high_idx = np.empty(n, dtype=np.int64)
    last_low_idx = np.empty(n, dtype=np.int64)
    last_high = float(h_arr[0]); last_low = float(l_arr[0])
    last_hi_i = 0; last_lo_i = 0
    for i in range(n):
        if pivot_high[i]:
            last_high = float(h_arr[i]); last_hi_i = i
        if pivot_low[i]:
            last_low = float(l_arr[i]); last_lo_i = i
        last_high_vals[i] = last_high
        last_low_vals[i] = last_low
        last_high_idx[i] = last_hi_i
        last_low_idx[i] = last_lo_i

    idx_arr = np.arange(n, dtype=np.int64)
    cv3["bars_since_swing_high"] = (idx_arr - last_high_idx).astype(np.float32)
    cv3["bars_since_swing_low"] = (idx_arr - last_low_idx).astype(np.float32)
    cv3["dist_last_swing_high_atr"] = ((close.to_numpy() - last_high_vals) / atr_safe).astype(np.float32)
    cv3["dist_last_swing_low_atr"] = ((close.to_numpy() - last_low_vals) / atr_safe).astype(np.float32)

    denom = np.maximum((last_high_vals - last_low_vals), eps)
    retracement = np.zeros(n, dtype=np.float64)
    up_mask = last_high_idx > last_low_idx
    down_mask = last_low_idx > last_high_idx
    retracement[up_mask] = (last_high_vals[up_mask] - close.to_numpy()[up_mask]) / denom[up_mask]
    retracement[down_mask] = (close.to_numpy()[down_mask] - last_low_vals[down_mask]) / denom[down_mask]
    retracement = np.clip(retracement, 0.0, 1.0)
    cv3["retracement_from_last_impulse"] = retracement.astype(np.float32)


def _add_regime_categoricals(cv3: pd.DataFrame) -> None:
    """Mutates cv3: trend_regime_id, vol_regime_id, atr_bucket, spread_bucket."""
    # trend_regime_id: 3-bin trend regime. MIRRORS add_ctx_cont_columns_to_prebuilt.py (one-truth).
    # 2026-06-03 (BIG-8): old price_vs_ema50_atr basis was DEGENERATE (constant=1). When
    # GX1_TREND_REGIME_FROM_D1=1, bucket by the TRUE D1 trend signal D1_dist_from_ema200_atr
    # (computed above in this same function). Default OFF = cement-compatible (cement V10's
    # ctx_cat embedding was trained on the old values); the regime-robust retrain enables it.
    import os as _os
    if _os.environ.get("GX1_TREND_REGIME_FROM_D1", "0") == "1" and "D1_dist_from_ema200_atr" in cv3.columns:
        d = cv3["D1_dist_from_ema200_atr"].astype(float).to_numpy()
        d = np.where(np.isfinite(d), d, 0.0)
        cv3["trend_regime_id"] = np.where(d < -1.0, 0, np.where(d <= 1.0, 1, 2)).astype(np.int64)
    elif "price_vs_ema50_atr" in cv3.columns:
        p = cv3["price_vs_ema50_atr"].astype(float).to_numpy()
        p = np.where(np.isfinite(p), p, 0.0)
        cv3["trend_regime_id"] = np.where(p < -0.5, 0, np.where(p <= 0.5, 1, 2)).astype(np.int64)
    else:
        cv3["trend_regime_id"] = 1
    # vol_regime_id / atr_bucket: percentile rank of atr_bps (0..4)
    if "atr_bps" in cv3.columns:
        vol = _rank_bucket_0_4(cv3["atr_bps"].to_numpy(dtype=float), fallback=2)
    else:
        vol = np.full(len(cv3), 2, dtype=np.int64)
    cv3["vol_regime_id"] = vol.astype(np.int64)
    cv3["atr_bucket"] = vol.astype(np.int64)
    # spread_bucket
    if "spread_bps" in cv3.columns:
        sp = _rank_bucket_0_4(cv3["spread_bps"].to_numpy(dtype=float), fallback=0)
    else:
        sp = np.zeros(len(cv3), dtype=np.int64)
    cv3["spread_bucket"] = sp.astype(np.int64)


# ── public API ────────────────────────────────────────────────────────────


def augment_canonical_v3(cv3: pd.DataFrame, df_m5: pd.DataFrame) -> pd.DataFrame:
    """Add the 32 ctx-cont / ctx-cat / session / interaction / swing features
    on top of a canonical_v3 DataFrame.

    Args:
        cv3: canonical_v3 (DatetimeIndex'd, output of LiveCanonicalV3Builder).
             Must already contain canonical_v2 features like _v1_atr14,
             _v1_ema_diff, _v1_range_z, _v1h1_slope3, plus M5 OHLC columns.
        df_m5: raw M5 OHLC tape covering the same time range (DatetimeIndex or
               'time' column). Needed for HTF resampling (1H/4H/1D/M15).

    Returns:
        DataFrame with cv3 + 32 added columns. Same index as cv3.
        Mutation note: cv3 is copied first; input is not modified.
    """
    if cv3.empty:
        return cv3
    out = cv3.copy()
    _add_spread_atr_bps(out)
    _add_session_features(out)
    _add_session_interactions(out)
    _add_htf_features(out, df_m5)
    _add_micro_features(out)
    _add_swing_features(out)
    _add_regime_categoricals(out)
    # Volume / order-flow per-bar features — SAME helper the V10 builder uses, so
    # the seq's vol_z_20/vol_ratio_5_20/vol_pct_96/signed_vol_z_20 are identical
    # train↔serve. Computed on the full `out` frame (full history) so trailing
    # windows match training (no window-edge skew). Fail-closed-neutral if no vol.
    from gx1.features.volume_features import add_volume_features
    add_volume_features(out)
    # REGIME_V4 (2026-06-03): multi-TF regime CONDITIONING + 'regime is shifting' CHANGE-
    # DETECTION features. ONE-TRUTH: identical gx1.features.regime_v4_features helper as the
    # build-side (add_ctx_cont_columns_to_prebuilt.py) — cannot drift. Default OFF = bit-parity
    # (inert until the Phase-C contract bump + retrain). Sources are already on `out`: per-TF
    # regime/trend-age/ema-stack from v12_state_from_prebuilt._V2_MTF_PER_TF, D1_dist from
    # _add_htf_features above. `out` is full-history time-ordered (same as the volume helper).
    import os as _os
    if _os.environ.get("GX1_REGIME_V4", "0") == "1":
        from gx1.features.regime_v4_features import add_regime_v4_features
        add_regime_v4_features(out)
    return out
