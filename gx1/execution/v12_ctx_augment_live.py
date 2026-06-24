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

from gx1.features.swing_structure_v1 import compute_swing_structure_features
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
    # A1 2026-06-04: STRICT min_periods=n (matches gx1.features.htf_features._atr).
    # The loose max(2, n//2) emitted an unconverged ATR on short serve/rescore windows;
    # used ONLY by _add_htf_features (HTF-block-local helper), so this is contained.
    return tr.rolling(window=n, min_periods=n).mean()


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
    """0..4 bucket via percentile rank across the input array.

    WARNING — FRAME-DEPENDENT: this ranks RELATIVE to whatever array it is handed (no fixed bin
    edges), so the SAME atr_bps lands in different buckets depending on the window. The BUILD ranks
    over full history; the daemon ranked over a trailing 420d window -> ~31% of live appends got a
    wrong-by-one vol bucket the IQL one-hots act on (2026-06-13 audit). Prefer _digitize_bucket_0_4
    against FROZEN full-history edges (frame-invariant); this rank path is the fail-soft fallback.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    s = pd.Series(x)
    q = s.rank(pct=True, method="average").to_numpy(dtype=float)
    if not np.isfinite(q).any():
        return np.full(len(x), int(fallback), dtype=np.int64)
    b = np.clip(q * 5.0, 0.0, 4.99).astype(np.int64)
    b = np.where(np.isfinite(b), b, int(fallback)).astype(np.int64)
    return b


# Frozen full-history bucket EDGES (2026-06-13 audit fix): digitizing atr_bps/spread_bps against
# fixed quantile edges makes vol_regime_id/atr_bucket/spread_bucket FRAME-INVARIANT, so the daemon
# (any window), the entry serve (augment_canonical_v3), the exit (base34), and the build all produce
# the IDENTICAL bucket = training. Edges live next to the cv3 prebuilt; regenerated at each cement/
# cutover (gx1.scripts.write_regime_bucket_edges). Verified: digitize == full-history rank at 100%.
REGIME_BUCKET_EDGES_PATH = (
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/regime_bucket_edges_v1.json"
)
_REGIME_EDGES_CACHE: dict | None = None
_REGIME_EDGES_MTIME: float | None = None


def _load_regime_bucket_edges() -> dict | None:
    """Load the frozen bucket edges JSON (cached, mtime-invalidated). None if absent."""
    global _REGIME_EDGES_CACHE, _REGIME_EDGES_MTIME
    import os as _os
    import json as _json
    try:
        mt = _os.stat(REGIME_BUCKET_EDGES_PATH).st_mtime
    except OSError:
        return None
    if _REGIME_EDGES_CACHE is None or mt != _REGIME_EDGES_MTIME:
        with open(REGIME_BUCKET_EDGES_PATH) as _fh:
            _REGIME_EDGES_CACHE = _json.load(_fh)
        _REGIME_EDGES_MTIME = mt
    return _REGIME_EDGES_CACHE


def _digitize_bucket_0_4(x: np.ndarray, edges, fallback: int) -> np.ndarray:
    """0..4 bucket by digitizing against FROZEN edges [q20,q40,q60,q80] — frame-invariant."""
    x = np.asarray(x, dtype=float)
    e = np.asarray(edges, dtype=float)
    b = np.clip(np.digitize(x, e, right=False), 0, 4).astype(np.int64)
    return np.where(np.isfinite(x), b, int(fallback)).astype(np.int64)


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
    # A1 FIX (2026-06-04): if cv3 already carries these HTF features, KEEP them.
    # The prebuilt (add_ctx_cont_columns_to_prebuilt.py) computes them over the FULL
    # history (D1 EMA200 / rolling(252,252)). Recomputing here over the short serve/rescore
    # window (~45d) badly skews the long-lookback D1 feats — measured |Δ| median 0.41 on the
    # 0..1 D1_atr_percentile_252 vs training. So when the caller passes the prebuilt (which
    # has the correct full-history values), do NOT clobber them with a short-window recompute.
    _htf_cols = (
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
    )
    if all(c in cv3.columns for c in _htf_cols):
        return
    # Ensure df_m5 is DatetimeIndex'd
    m5 = df_m5.copy()
    if "time" in m5.columns and not isinstance(m5.index, pd.DatetimeIndex):
        m5["time"] = pd.to_datetime(m5["time"], utc=True)
        m5 = m5.set_index("time")

    # D1 features — A1 2026-06-04: STRICT warmup floors matching gx1.features.htf_features
    # (D1_EMA200_MIN_BARS=220, D1_PCTL252_MIN_BARS=270). A too-short serve/rescore window
    # now fails-closed to the cement-neutral default instead of writing an unconverged
    # (loose) value (was 29% of bars off by >0.40 on D1_atr_percentile_252). Normal live is
    # unaffected: the preserve-guard above keeps the prebuilt's full-history HTF when present.
    df_d1 = _resample_ohlc(m5, "1D")
    if len(df_d1) >= 220:  # EMA200 converged
        d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
        d1_ema200 = _ema(d1_mid, 200)
        d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
        d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)
        cv3["D1_dist_from_ema200_atr"] = _align_last_closed(
            cv3.index, d1_dist, pd.Timedelta(days=1)
        ).fillna(0.0).to_numpy(dtype=float)
        # D1_atr_percentile_252 — only with the FULL 252 window (strict min_periods=252)
        if len(df_d1) >= 270:
            def _pctl_last(arr):
                a = np.asarray(arr, dtype=float)
                if not np.isfinite(a).all():
                    return float("nan")
                return float((a <= a[-1]).mean())
            atr_pctl = d1_atr14.rolling(252, min_periods=252).apply(_pctl_last, raw=True).ffill()
            cv3["D1_atr_percentile_252"] = _align_last_closed(
                cv3.index, atr_pctl, pd.Timedelta(days=1)
            ).fillna(0.5).to_numpy(dtype=float)
        else:
            cv3["D1_atr_percentile_252"] = 0.5
    else:
        cv3["D1_dist_from_ema200_atr"] = 0.0
        cv3["D1_atr_percentile_252"] = 0.5

    # H1 features (H1_ATR100_MIN_BARS=120 — ATR100 converged)
    df_h1 = _resample_ohlc(m5, "1H")
    if len(df_h1) >= 120:
        h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
        h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
        h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)
        cv3["H1_range_compression_ratio"] = _align_last_closed(
            cv3.index, h1_comp, pd.Timedelta(hours=1)
        ).fillna(1.0).to_numpy(dtype=float)
    else:
        cv3["H1_range_compression_ratio"] = 1.0

    # M15 features (M15_ATR100_MIN_BARS=200)
    df_m15 = _resample_ohlc(m5, "15min")
    if len(df_m15) >= 200:
        m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
        m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill()
        m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
        cv3["M15_range_compression_ratio"] = _align_last_closed(
            cv3.index, m15_comp, pd.Timedelta(minutes=15)
        ).fillna(1.0).to_numpy(dtype=float)
    else:
        cv3["M15_range_compression_ratio"] = 1.0

    # H4 trend sign categorical (H4_EMA50_MIN_BARS=80 — EMA50 converged)
    df_h4 = _resample_ohlc(m5, "4H")
    if len(df_h4) >= 80:
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
    """Mutates cv3 with the 5 swing-structure ctx features (dist_last_swing_high/low_atr,
    bars_since_swing_high/low, retracement_from_last_impulse). Delegates to the ONE-TRUTH
    helper gx1.features.swing_structure_v1 (lookahead-safe confirmation lag) — do NOT
    re-implement the math here (2026-06-24 unification; live decision bar stays causal)."""
    feats = compute_swing_structure_features(
        cv3["high"].to_numpy(dtype=np.float64),
        cv3["low"].to_numpy(dtype=np.float64),
        cv3["close"].to_numpy(dtype=np.float64),
        lookback=2,
        atr_period=SWING_ATR_PERIOD,
    )
    for _name, _arr in feats.items():
        cv3[_name] = _arr


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
    # vol_regime_id / atr_bucket / spread_bucket: bucket atr_bps/spread_bps into 0..4.
    # 2026-06-13 audit FIX: digitize against FROZEN full-history edges (frame-invariant, so daemon /
    # entry-serve / exit / build all agree = training) instead of a frame-relative percentile rank
    # (which made the daemon's 420d window disagree with full-history training on ~31% of appends).
    # Fall back to the rank path + WARN if the edges file is missing (degraded, but never crash live).
    _edges = _load_regime_bucket_edges()
    if "atr_bps" in cv3.columns:
        _av = cv3["atr_bps"].to_numpy(dtype=float)
        if _edges and _edges.get("atr_bps_edges"):
            vol = _digitize_bucket_0_4(_av, _edges["atr_bps_edges"], fallback=2)
        else:
            LOG.warning("[REGIME] frozen atr_bps bucket edges absent (%s) — falling back to "
                        "FRAME-RELATIVE rank (train≠serve risk). Regenerate via "
                        "gx1.scripts.write_regime_bucket_edges.", REGIME_BUCKET_EDGES_PATH)
            vol = _rank_bucket_0_4(_av, fallback=2)
    else:
        vol = np.full(len(cv3), 2, dtype=np.int64)
    cv3["vol_regime_id"] = vol.astype(np.int64)
    cv3["atr_bucket"] = vol.astype(np.int64)
    # spread_bucket (spread_bps is ~constant on XAU — frame-invariant either way, but digitize for parity)
    if "spread_bps" in cv3.columns:
        _sv = cv3["spread_bps"].to_numpy(dtype=float)
        if _edges and _edges.get("spread_bps_edges"):
            sp = _digitize_bucket_0_4(_sv, _edges["spread_bps_edges"], fallback=0)
        else:
            sp = _rank_bucket_0_4(_sv, fallback=0)
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
