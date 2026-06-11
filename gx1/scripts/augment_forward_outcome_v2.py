#!/usr/bin/env python3
"""V2 Phase B5 — augment forward-outcome parquets with 125 per-TF + 28 group-A V2 scalars.

C+prune strategy: embed ALL 25 V2-features per TF (5 TFs × 25 = 125), plus 28
group-A features. Permutation importance later (Phase C4b/C5b) prunes weak ones.

OPTIMIZED (2026-05-22): builds all caches ONCE per session, per-candidate compute
is O(log N) lookups. Target throughput: 50-200 cand/s (vs 1 cand/s naive).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.features.htf_features import (
    build_multi_tf_per_bar_features_v2, MULTI_TF_PER_BAR_FEATURES_V2,
)
# Round-number proximity: import the ONE-TRUTH compute (group_a_features) — no duplicated math.
from gx1.features.group_a_features import (
    round_number_levels as _round_number_levels, ROUND_FEATURE_NAMES as _ROUND_FEATURE_NAMES,
)
_AUG_ROUND_ON = os.environ.get("GX1_ROUND_NUMBER", "0") == "1"

TF_NAMES = ("M5", "M15", "H1", "H4", "D1")

def _pertf_name(tf: str, feat: str) -> str:
    base = f"{tf.lower()}_{feat}"
    return base if feat.endswith("_v2") else base + "_v2"

PER_TF_FEATURE_NAMES = tuple(
    _pertf_name(tf, feat) for tf in TF_NAMES for feat in MULTI_TF_PER_BAR_FEATURES_V2
)
# 5 × 25 = 125

GROUP_A_FEATURE_NAMES = (
    "is_asia_eu_overlap", "is_eu_us_overlap", "is_eu_only", "is_us_only",
    "atr_ratio_m5_h4", "atr_ratio_m15_d1", "atr_ratio_h1_d1", "atr_ratio_m5_m15",
    "vol_pct_m5_1yr", "vol_pct_h1_1yr",
    "dist_to_R1_atr", "dist_to_R2_atr", "dist_to_S1_atr", "dist_to_S2_atr",
    # 2026-05-24 Bug 2 fix: added M15+D1 (was only M5/H1/H4)
    "dist_to_m5_hi_atr",  "dist_to_m5_lo_atr",
    "dist_to_m15_hi_atr", "dist_to_m15_lo_atr",
    "dist_to_h1_hi_atr",  "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr",  "dist_to_h4_lo_atr",
    "dist_to_d1_hi_atr",  "dist_to_d1_lo_atr",
    "long_win_rate_last10",  "long_mean_pnl_last10",
    "long_n_consec_losses",  "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
)
if _AUG_ROUND_ON:  # +5 ONLY under GX1_ROUND_NUMBER → default-OFF keeps the Group-A name set byte-identical;
    # the zero-fill (augment_candidate m5_idx<0), new_cols dict (augment_week), and attach_group_a copy-set all
    # iterate GROUP_A_FEATURE_NAMES, so appending here wires every code path in one place.
    GROUP_A_FEATURE_NAMES = GROUP_A_FEATURE_NAMES + _ROUND_FEATURE_NAMES

# FVG (fair-value-gap) 3-bar imbalance proximity per-TF M5/M15/H1 (2026-06-11, env-gated GX1_FVG_FEATURES,
# default OFF). EMPTY tuple when OFF → the zero-fill / new_cols / cols_to_overwrite paths add nothing →
# byte-identical to cement. (H4/D1 gaps too sparse for shuffled-liveness → M5/M15/H1 only.)
_FVG_ON = os.environ.get("GX1_FVG_FEATURES", "0") == "1"
FVG_FEATURE_NAMES = tuple(
    f"{tf}_{c}" for tf in ("m5", "m15", "h1")
    for c in ("dist_to_unfilled_fvg_atr", "fvg_active")
) if _FVG_ON else ()

# 2026-05-24 GROUP_S: SMC features from canonical_v3 (joined by decision_ts).
# These ARE in canonical_v3 prebuilt with real signal (smc_choch 421 non-zero,
# smc_bos_down 76K, smc_sweep_up/down 25K each) but were NEVER joined into
# forward_outcome → all zero downstream. Fix: load canonical_v3 once, asof-join
# by time to add SMC cols (with _canon_v1 suffix to match downstream expectations).
# 2026-05-24 PM: 5-TF dip + structure features (computed in augment_candidate).
# Persisted into fwd parquet so both Entry-IQL and Exit-IQL state builders can
# read them via column lookup (no need to recompute at training time).
DIP_STRUCT_FEATURE_NAMES = tuple(
    [f"dip_proximity_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"dip_confirmed_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_continuation_up_{tf}_v3"      for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_pullback_in_uptrend_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_continuation_down_{tf}_v3"    for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_bounce_in_downtrend_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_pullback_depth_{tf}_v3"       for tf in ("m5","m15","h1","h4","d1")]
  + ["struct_all_tf_pullback_v3", "struct_tf_agree_count_v3", "struct_dip_x_uptrend_v3"]
)  # 5×7 + 3 = 38 cols (smc_swing_x_dip computed downstream after SMC join)

GROUP_S_SMC_FEATURE_NAMES = (
    "smc_swing_state_canon_v1",
    "smc_bos_up_canon_v1",
    "smc_bos_down_canon_v1",
    "smc_choch_canon_v1",
    "smc_sweep_up_canon_v1",
    "smc_sweep_down_canon_v1",
    "smc_sweep_size_atr_canon_v1",
    "smc_bars_since_sweep_canon_v1",
    "smc_premium_discount_canon_v1",
    "smc_premium_state_canon_v1",
)
ASIA_HOURS = set(list(range(22, 24)) + list(range(0, 9)))
EU_HOURS   = set(range(7, 17))
US_HOURS   = set(range(13, 22))

# --forward-outcome-dir is REQUIRED (no silent stale default; rule 8). The old hardcoded literal
# (CANDIDATE_FORWARD_OUTCOME_V3PLUS_..._20260521 LOCK) is superseded 2x (v3+ -> COSTFIX -> fase2b)
# and quarantined — pass the current forward-outcome dir explicitly so an augment can't read a stale set.
M5_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")


@dataclass
class AugmentContext:
    """Pre-built caches for fast per-candidate compute."""
    m5_close: np.ndarray            # close prices
    m5_high: np.ndarray
    m5_low: np.ndarray
    m5_ts_ns: np.ndarray            # int64 ns
    # Per-TF cache (from V2 multi-TF builder): {tf: feats_df_with_attrs}
    multi_tf: dict
    # Resampled H1/H4 OHLC for liquidity zones
    h1_ts_ns: np.ndarray; h1_high: np.ndarray; h1_low: np.ndarray
    h4_ts_ns: np.ndarray; h4_high: np.ndarray; h4_low: np.ndarray
    # 2026-05-24 Bug 2 fix: M15 + D1 for liquidity zones
    m15_ts_ns: np.ndarray; m15_high: np.ndarray; m15_low: np.ndarray
    d1_ts_ns: np.ndarray;  d1_high: np.ndarray;  d1_low: np.ndarray
    # Pre-computed ATR percentile arrays — at each M5 bar, current ATR vs trailing-1yr
    m5_atr_pct_1yr: np.ndarray
    h1_atr_pct_1yr: np.ndarray      # per-H1-bar
    h1_atr_pct_ts_ns: np.ndarray
    # D1 pivot levels per M5 (R1/R2/S1/S2 from prior D1 OHLC)
    # store as per-day arrays: lookup via date
    daily_pivot_by_date: dict       # date_str → {R1, R2, S1, S2}
    # Trade history from journal — DataFrame with (close_ts, side, pnl_bps)
    trade_history: pd.DataFrame


def _build_resampled_ohlc_array(df: pd.DataFrame, rule: str) -> tuple:
    """Return (ts_int64, high, low) for resampled bars."""
    resamp = df.resample(rule).agg({"high": "max", "low": "min"}).dropna()
    ts_ns = resamp.index.values.astype("datetime64[ns]").astype(np.int64)
    return ts_ns, resamp["high"].to_numpy(np.float64), resamp["low"].to_numpy(np.float64)


def _build_atr_percentile_array(df: pd.DataFrame, ts_ns: np.ndarray, window_days: int = 365) -> np.ndarray:
    """For each row, percentile-rank of current ATR vs last `window_days` ATRs.

    Uses Wilder ATR-14. Returns array of percentiles in [0, 1].
    """
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().fillna(method="bfill").to_numpy(np.float64)
    # Rolling 1yr percentile via numpy: for each i, percentile rank of atr[i] within
    # atr[i-WIN:i+1]. WIN = bars in 1yr (M5: 365*24*12 = 105k, H1: 365*24 = 8.7k).
    n_per_day = int(round(86400 / (ts_ns[1] - ts_ns[0]) * 1e9)) if len(ts_ns) > 1 else 288
    window = window_days * n_per_day
    pct = np.full(len(atr), 0.5, dtype=np.float32)
    # Simple loop — for full M5 (455K) this is 455K iters with each doing a sort on N=105K.
    # Use approximation: compute percentile from sample of last 1000 values per stride.
    # For speed: compute on STRIDED sample then ffill.
    stride = max(1, len(atr) // 50000)   # ~50K computations max
    for i in range(window, len(atr), stride):
        recent = atr[max(0, i - window):i + 1]
        if len(recent) >= 10:
            pct[i] = float((recent < atr[i]).mean())
    # Forward-fill between strided computations
    pct = pd.Series(pct).fillna(method="ffill").fillna(0.5).to_numpy(dtype=np.float32)
    return pct


def _build_daily_pivots(df: pd.DataFrame) -> dict[str, dict]:
    """Compute classic pivots R1/R2/S1/S2 per day, keyed by date_str (YYYY-MM-DD)."""
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    out = {}
    for ts, row in daily.iterrows():
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        pp = (h + l + c) / 3.0
        out[ts.strftime("%Y-%m-%d")] = {
            "R1": 2 * pp - l,  "R2": pp + (h - l),
            "S1": 2 * pp - h,  "S2": pp - (h - l),
        }
    return out


def _build_trade_history(journal_dir: Path, suffix: str = "live_v12_4") -> pd.DataFrame:
    """One-shot parse of all live journals → DataFrame (close_ts, side, pnl_bps)."""
    if not journal_dir.exists():
        return pd.DataFrame(columns=["close_ts", "side", "pnl_bps"])
    files = sorted(journal_dir.glob(f"v12_paper_journal_*_{suffix}.jsonl"))
    rows = []
    last_snap: dict[str, dict] = {}
    prev_ids: set[str] = set()
    for fp in files:
        try:
            with fp.open() as f:
                for line in f:
                    try: e = json.loads(line)
                    except json.JSONDecodeError: continue
                    otrs = e.get("open_trade_records") or []
                    cur_ids = {str(r["trade_id"]) for r in otrs if r.get("trade_id")}
                    for r in otrs:
                        tid = str(r.get("trade_id") or "")
                        if tid:
                            last_snap[tid] = {
                                "side": r.get("side"),
                                "pnl_bps": r.get("pnl_bps"),
                                "close_ts": e.get("logged_at_utc") or e.get("ts_utc"),
                            }
                    for tid in (prev_ids - cur_ids):
                        snap = last_snap.pop(tid, None)
                        if snap and snap.get("pnl_bps") is not None and snap.get("close_ts"):
                            rows.append(snap)
                    prev_ids = cur_ids
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["close_ts", "side", "pnl_bps"])
    df = pd.DataFrame(rows)
    df["close_ts"] = pd.to_datetime(df["close_ts"], utc=True)
    df["pnl_bps"] = pd.to_numeric(df["pnl_bps"], errors="coerce")
    return df.dropna(subset=["close_ts", "pnl_bps"]).sort_values("close_ts").reset_index(drop=True)


def _assert_multi_tf_cache_fresh(m5_df: pd.DataFrame, multi_tf: dict) -> None:
    """Fail-closed (rule 4): a multi-TF V2 cache ending long before the M5 build data
    means the build SILENTLY used a STALE cache — the 2026-06-05 audit footgun, where
    builders loaded the 05-22 cache via the GX1_V10_MULTI_TF_V2_CACHE_DIR env-default
    while building on fresh cv3 (silent train/serve feature skew). Raise if the cache's
    newest bar lags the M5 cutoff by > GX1_MTF_CACHE_MAX_LAG_DAYS (default 2). Bypass
    only for an explicit OOT/replay build via GX1_MTF_CACHE_ALLOW_STALE=1 (logged loud).
    Covers ALL build paths because every one funnels through build_context().
    """
    if not multi_tf:
        return
    try:
        m5_last = int(pd.Timestamp(m5_df.index[-1]).value)
        cache_last = max(
            int(np.asarray(df.attrs["ts_int64"]).max())
            for df in multi_tf.values()
            if getattr(df, "attrs", None) and len(df.attrs.get("ts_int64", []))
        )
    except (ValueError, KeyError, IndexError):
        return  # can't determine cutoff — don't block
    lag_days = (m5_last - cache_last) / 86_400e9
    print(f"[MTF_CACHE] m5_cutoff={pd.Timestamp(m5_last)} cache_cutoff={pd.Timestamp(cache_last)} "
          f"lag={lag_days:.1f}d", flush=True)
    if os.environ.get("GX1_MTF_CACHE_ALLOW_STALE", "0").strip().lower() in ("1", "true", "yes", "on"):
        print("[MTF_CACHE_STALE] freshness check BYPASSED via GX1_MTF_CACHE_ALLOW_STALE=1", flush=True)
        return
    max_lag = float(os.environ.get("GX1_MTF_CACHE_MAX_LAG_DAYS", "2"))
    if lag_days > max_lag:
        raise RuntimeError(
            f"[MTF_CACHE_STALE] multi-TF V2 cache ends {lag_days:.1f} days before the M5 build data "
            f"(cache={pd.Timestamp(cache_last)}, data={pd.Timestamp(m5_last)}); refusing to build on a "
            f"stale cache. Regenerate prebuild_multi_tf_cache_v2.py for this cutoff and set "
            f"GX1_V10_MULTI_TF_V2_CACHE_DIR, or set GX1_MTF_CACHE_ALLOW_STALE=1 for an explicit OOT replay."
        )


def build_context(m5_df: pd.DataFrame, multi_tf: dict, journal_dir: Path) -> AugmentContext:
    """Pre-compute all caches ONCE. Heavy upfront cost, fast per-candidate after."""
    _assert_multi_tf_cache_fresh(m5_df, multi_tf)  # fail-closed on stale multi-TF cache (rule 4)
    ts_ns = m5_df.index.values.astype("datetime64[ns]").astype(np.int64)
    h1_ts, h1_hi, h1_lo = _build_resampled_ohlc_array(m5_df, "1h")
    h4_ts, h4_hi, h4_lo = _build_resampled_ohlc_array(m5_df, "4h")
    # 2026-05-24 Bug 2 fix: M15 + D1 resampled OHLC for liquidity zones
    m15_ts, m15_hi, m15_lo = _build_resampled_ohlc_array(m5_df, "15min")
    d1_ts, d1_hi, d1_lo = _build_resampled_ohlc_array(m5_df, "1D")
    # ATR percentile arrays
    m5_atr_pct = _build_atr_percentile_array(m5_df, ts_ns)
    # H1 ATR percentile
    h1_df = m5_df.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    h1_ts_pct = h1_df.index.values.astype("datetime64[ns]").astype(np.int64)
    h1_atr_pct = _build_atr_percentile_array(h1_df, h1_ts_pct)
    # Daily pivots
    daily_pivots = _build_daily_pivots(m5_df)
    # Trade history (journal)
    trade_hist = _build_trade_history(journal_dir)

    return AugmentContext(
        m5_close=m5_df["close"].to_numpy(np.float64),
        m5_high=m5_df["high"].to_numpy(np.float64),
        m5_low=m5_df["low"].to_numpy(np.float64),
        m5_ts_ns=ts_ns,
        multi_tf=multi_tf,
        h1_ts_ns=h1_ts, h1_high=h1_hi, h1_low=h1_lo,
        h4_ts_ns=h4_ts, h4_high=h4_hi, h4_low=h4_lo,
        m15_ts_ns=m15_ts, m15_high=m15_hi, m15_low=m15_lo,
        d1_ts_ns=d1_ts, d1_high=d1_hi, d1_low=d1_lo,
        m5_atr_pct_1yr=m5_atr_pct,
        h1_atr_pct_1yr=h1_atr_pct,
        h1_atr_pct_ts_ns=h1_ts_pct,
        daily_pivot_by_date=daily_pivots,
        trade_history=trade_hist,
    )


def _session_overlap(ts: pd.Timestamp) -> dict[str, float]:
    h = ts.hour
    asia, eu, us = h in ASIA_HOURS, h in EU_HOURS, h in US_HOURS
    return {
        "is_asia_eu_overlap": float(asia and eu),
        "is_eu_us_overlap":   float(eu and us),
        "is_eu_only":         float(eu and not us and not asia),
        "is_us_only":         float(us and not eu and not asia),
    }


def _per_tf_all(ctx: AugmentContext, ts_ns: int) -> dict[str, float]:
    """O(log N) lookup of all 25 features per TF using V2 cache's int64 arrays."""
    out: dict[str, float] = {}
    for tf in TF_NAMES:
        feats = ctx.multi_tf.get(tf)
        if feats is None or len(feats) == 0:
            for feat in MULTI_TF_PER_BAR_FEATURES_V2:
                out[_pertf_name(tf, feat)] = 0.0
            continue
        ts_arr = feats.attrs.get("ts_int64")
        feats_arr = feats.attrs.get("feats_np")
        if ts_arr is None or feats_arr is None:
            for feat in MULTI_TF_PER_BAR_FEATURES_V2:
                out[_pertf_name(tf, feat)] = 0.0
            continue
        # searchsorted: index of FIRST ts > ts_ns; -1 = last bar at or before
        right = np.searchsorted(ts_arr, ts_ns, side="right")
        if right == 0:
            for feat in MULTI_TF_PER_BAR_FEATURES_V2:
                out[_pertf_name(tf, feat)] = 0.0
            continue
        row = feats_arr[right - 1]
        for j, feat in enumerate(MULTI_TF_PER_BAR_FEATURES_V2):
            out[_pertf_name(tf, feat)] = float(row[j])
    return out


def _vol_term(ctx: AugmentContext, ts_ns: int) -> dict[str, float]:
    """ATR ratios across TFs at ts_ns — uses V2 cache."""
    def _last_atr(tf):
        feats = ctx.multi_tf.get(tf)
        if feats is None: return 0.0
        ts_arr = feats.attrs["ts_int64"]
        right = np.searchsorted(ts_arr, ts_ns, side="right")
        if right == 0: return 0.0
        feats_np = feats.attrs["feats_np"]
        # atr_bps_14 is feature index 0 in MULTI_TF_PER_BAR_FEATURES_V2
        return float(feats_np[right - 1, 0])
    a_m5 = _last_atr("M5"); a_m15 = _last_atr("M15")
    a_h1 = _last_atr("H1"); a_h4 = _last_atr("H4"); a_d1 = _last_atr("D1")
    return {
        "atr_ratio_m5_h4":  min(50.0, a_m5 / max(a_h4, 1e-3)),
        "atr_ratio_m15_d1": min(50.0, a_m15 / max(a_d1, 1e-3)),
        "atr_ratio_h1_d1":  min(50.0, a_h1 / max(a_d1, 1e-3)),
        "atr_ratio_m5_m15": min(50.0, a_m5 / max(a_m15, 1e-3)),
    }


def _vol_pct(ctx: AugmentContext, ts_ns: int) -> dict[str, float]:
    """Lookup pre-computed M5 / H1 ATR percentile at ts_ns."""
    m5_idx = np.searchsorted(ctx.m5_ts_ns, ts_ns, side="right") - 1
    m5_pct = float(ctx.m5_atr_pct_1yr[m5_idx]) if m5_idx >= 0 else 0.5
    h1_idx = np.searchsorted(ctx.h1_atr_pct_ts_ns, ts_ns, side="right") - 1
    h1_pct = float(ctx.h1_atr_pct_1yr[h1_idx]) if h1_idx >= 0 else 0.5
    return {"vol_pct_m5_1yr": m5_pct, "vol_pct_h1_1yr": h1_pct}


def _pivots(ctx: AugmentContext, ts: pd.Timestamp, current_atr: float, current_price: float) -> dict[str, float]:
    """Lookup prior-day pivots — O(1) dict lookup."""
    prior_date = (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    p = ctx.daily_pivot_by_date.get(prior_date)
    if p is None:
        # try 2 days back for weekends
        prior_date = (ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        p = ctx.daily_pivot_by_date.get(prior_date)
    if p is None:
        return {"dist_to_R1_atr": 0.0, "dist_to_R2_atr": 0.0,
                "dist_to_S1_atr": 0.0, "dist_to_S2_atr": 0.0}
    atr_safe = max(current_atr, 1e-3)
    return {
        "dist_to_R1_atr": (current_price - p["R1"]) / atr_safe,
        "dist_to_R2_atr": (current_price - p["R2"]) / atr_safe,
        "dist_to_S1_atr": (current_price - p["S1"]) / atr_safe,
        "dist_to_S2_atr": (current_price - p["S2"]) / atr_safe,
    }


def _liquidity_zones(ctx: AugmentContext, ts_ns: int, current_price: float, current_atr: float) -> dict[str, float]:
    """Distance to nearest unswept high/low per TF — uses pre-resampled arrays."""
    atr_safe = max(current_atr, 1e-3)
    out: dict[str, float] = {}
    for tf_name, ts_arr, hi_arr, lo_arr, lookback in (
        ("m5",  ctx.m5_ts_ns,  ctx.m5_high,  ctx.m5_low,  240),
        ("m15", ctx.m15_ts_ns, ctx.m15_high, ctx.m15_low, 192),
        ("h1",  ctx.h1_ts_ns,  ctx.h1_high,  ctx.h1_low,  168),
        ("h4",  ctx.h4_ts_ns,  ctx.h4_high,  ctx.h4_low,  168),
        ("d1",  ctx.d1_ts_ns,  ctx.d1_high,  ctx.d1_low,  60),
    ):
        right = np.searchsorted(ts_arr, ts_ns, side="right")
        if right == 0:
            out[f"dist_to_{tf_name}_hi_atr"] = 0.0
            out[f"dist_to_{tf_name}_lo_atr"] = 0.0
            continue
        left = max(0, right - lookback)
        window_hi = hi_arr[left:right]
        window_lo = lo_arr[left:right]
        # Nearest UNSWEPT high above current
        highs_above = window_hi[window_hi > current_price]
        out[f"dist_to_{tf_name}_hi_atr"] = (
            float((highs_above.min() - current_price) / atr_safe) if len(highs_above) > 0 else 0.0
        )
        lows_below = window_lo[window_lo < current_price]
        out[f"dist_to_{tf_name}_lo_atr"] = (
            float((current_price - lows_below.max()) / atr_safe) if len(lows_below) > 0 else 0.0
        )
    return out


def _per_side_perf(ctx: AugmentContext, ts: pd.Timestamp, lookback_n: int = 10) -> dict[str, float]:
    """Lookup last-N closures before ts from pre-built trade history."""
    default = {
        "long_win_rate_last10": 0.5, "long_mean_pnl_last10": 0.0,
        "long_n_consec_losses": 0.0, "long_time_since_last_close_min": 240.0,
        "short_win_rate_last10": 0.5, "short_mean_pnl_last10": 0.0,
        "short_n_consec_losses": 0.0, "short_time_since_last_close_min": 240.0,
    }
    if len(ctx.trade_history) == 0:
        return default
    th = ctx.trade_history
    eligible = th[th["close_ts"] <= ts]
    if eligible.empty:
        return default
    out = dict(default)
    for side in ("long", "short"):
        side_df = eligible[eligible["side"] == side].tail(lookback_n)
        if side_df.empty: continue
        pnls = side_df["pnl_bps"].to_numpy()
        out[f"{side}_win_rate_last10"] = float((pnls > 0).mean())
        out[f"{side}_mean_pnl_last10"] = float(pnls.mean())
        # consec losses from end backwards
        n_consec = 0
        for p in reversed(pnls.tolist()):
            if p > 0: break
            n_consec += 1
        out[f"{side}_n_consec_losses"] = float(n_consec)
        mins = max(0.0, min(1440.0, (ts - side_df["close_ts"].iloc[-1]).total_seconds() / 60.0))
        out[f"{side}_time_since_last_close_min"] = float(mins)
    return out


def _dip_struct_5tf(per_tf_out: dict[str, float], liq_out: dict[str, float]) -> dict[str, float]:
    """5-TF dip + structure features computed from per-TF V2 + liquidity zones.
    Persisted into fwd dataset so both Entry-IQL and Exit-IQL state builders can
    read them without recomputing.
    """
    out: dict[str, float] = {}
    # DIP per TF: proximity-to-low × sigmoid(ema20-slope)
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        dist_lo = float(liq_out.get(f"dist_to_{tf}_lo_atr", 2.0))
        slope = float(per_tf_out.get(f"{tf}_ema20_slope_atr_v2", 0.0))
        dip_prox = max(0.0, min(1.0, 1.0 - dist_lo / 2.0))
        recovery = 1.0 / (1.0 + math.exp(-slope * 5.0))
        out[f"dip_proximity_{tf}_v3"] = dip_prox
        out[f"dip_confirmed_{tf}_v3"] = dip_prox * recovery
    # STRUCTURE per TF: HH/HL/LH/LL via mom_5 + mom_20 signs
    pback = {}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        m5_mom = float(per_tf_out.get(f"{tf}_mom_5_atr_v2", 0.0))
        m20_mom = float(per_tf_out.get(f"{tf}_mom_20_atr_v2", 0.0))
        out[f"struct_continuation_up_{tf}_v3"] = float((m20_mom > 0) and (m5_mom > 0))
        out[f"struct_pullback_in_uptrend_{tf}_v3"] = float((m20_mom > 0) and (m5_mom < 0))
        out[f"struct_continuation_down_{tf}_v3"] = float((m20_mom < 0) and (m5_mom < 0))
        out[f"struct_bounce_in_downtrend_{tf}_v3"] = float((m20_mom < 0) and (m5_mom > 0))
        depth = -m5_mom / max(abs(m20_mom), 1e-6) if m20_mom > 1e-6 else 0.0
        out[f"struct_pullback_depth_{tf}_v3"] = max(-2.0, min(2.0, depth))
        pback[tf] = out[f"struct_pullback_in_uptrend_{tf}_v3"]
    # Multi-TF combo: strict AND across all 5 TFs
    out["struct_all_tf_pullback_v3"] = pback["m5"] * pback["m15"] * pback["h1"] * pback["h4"] * pback["d1"]
    out["struct_tf_agree_count_v3"] = (pback["m5"] + pback["m15"] + pback["h1"] + pback["h4"] + pback["d1"]) / 5.0
    avg_dip = (out["dip_confirmed_m5_v3"] + out["dip_confirmed_m15_v3"] + out["dip_confirmed_h1_v3"]
               + out["dip_confirmed_h4_v3"] + out["dip_confirmed_d1_v3"]) / 5.0
    out["struct_dip_x_uptrend_v3"] = avg_dip * pback["m5"]
    # struct_smc_swing_x_dip_v3 computed downstream in augment_week (needs SMC join)
    return out


def _fvg_5tf(ctx: "AugmentContext", ts_ns: int, current_price: float, current_atr: float) -> dict[str, float]:
    """Nearest UNFILLED 3-bar fair-value-gap per TF — signed ATR-distance + decayed activity.
    Bullish FVG: high[k-2] < low[k] (gap band); bearish: low[k-2] > high[k]. A gap is FILLED once a later
    bar trades back through it. Mirrors _liquidity_zones' per-TF resampled hi/lo arrays. M5/M15/H1 only."""
    import math as _math
    atr_safe = max(current_atr, 1e-3)
    CLIP, TAU = 3.0, 1.0
    M5_NS = 300_000_000_000  # decision moment = close of the M5 bar labeled ts (= ts + 5min)
    out: dict[str, float] = {}
    for tf, ts_arr, hi_arr, lo_arr, lb, period_ns in (
        ("m5", ctx.m5_ts_ns, ctx.m5_high, ctx.m5_low, 240, M5_NS),
        ("m15", ctx.m15_ts_ns, ctx.m15_high, ctx.m15_low, 192, 900_000_000_000),
        ("h1", ctx.h1_ts_ns, ctx.h1_high, ctx.h1_low, 168, 3_600_000_000_000),
    ):
        # 2026-06-11 LOOK-AHEAD FIX: only bars whose period has COMPLETED by the decision moment
        # (close of the M5 bar labeled ts, i.e. ts+5min) may enter the window. The old side="right"
        # cut at ts included the FORMING M15/H1 bar, whose full-period high/low (resampled from the
        # complete tape) leaks intra-period FUTURE data → build≠serve at the same ts. The serve-time
        # contract for FVG is therefore COMPLETED TF bars only. M5 is unchanged (label<=ts ⇔
        # completed-by-ts+5min on the native M5 grid).
        right = int(np.searchsorted(ts_arr, ts_ns + M5_NS - period_ns, side="right"))
        if right < 3:
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = CLIP
            out[f"{tf}_fvg_active"] = float(_math.exp(-CLIP / TAU))
            continue
        left = max(0, right - lb)
        hi = hi_arr[left:right]; lo = lo_arr[left:right]
        n = len(hi)
        bull = hi[:-2] < lo[2:]            # gap band (hi[k-2], lo[k])
        bear = lo[:-2] > hi[2:]            # gap band (hi[k], lo[k-2])
        suf_minlow = np.minimum.accumulate(lo[::-1])[::-1]   # suf_minlow[j] = min(lo[j:])
        suf_maxhi = np.maximum.accumulate(hi[::-1])[::-1]
        best_signed = None; best_abs = CLIP * atr_safe + 1.0
        for j in np.flatnonzero(bull):        # local formation idx = j+2
            f = j + 2
            gap_lo, gap_hi = hi[j], lo[f]     # gap_lo < gap_hi by construction
            if f + 1 < n and suf_minlow[f + 1] <= gap_lo:
                continue                      # filled (later low traded back down through the gap)
            if current_price >= gap_hi:
                signed = current_price - gap_hi
            elif current_price <= gap_lo:
                signed = -(gap_lo - current_price)
            else:
                signed = 0.0                  # inside the gap
            if abs(signed) < best_abs:
                best_abs = abs(signed); best_signed = signed
        for j in np.flatnonzero(bear):
            f = j + 2
            gap_lo, gap_hi = hi[f], lo[j]
            if f + 1 < n and suf_maxhi[f + 1] >= gap_hi:
                continue
            if current_price <= gap_lo:
                signed = -(gap_lo - current_price)
            elif current_price >= gap_hi:
                signed = current_price - gap_hi
            else:
                signed = 0.0
            if abs(signed) < best_abs:
                best_abs = abs(signed); best_signed = signed
        if best_signed is None:
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = CLIP
            out[f"{tf}_fvg_active"] = float(_math.exp(-CLIP / TAU))
        else:
            d = max(-CLIP, min(CLIP, float(best_signed) / atr_safe))
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = float(d)
            out[f"{tf}_fvg_active"] = float(_math.exp(-abs(d) / TAU))
    return out


def augment_candidate(ctx: AugmentContext, ts: pd.Timestamp) -> dict[str, float]:
    """Fast per-candidate compute: O(log N) lookups for all V2 + dip/struct features."""
    ts_ns = ts.value if hasattr(ts, "value") else pd.Timestamp(ts).value
    m5_idx = np.searchsorted(ctx.m5_ts_ns, ts_ns, side="right") - 1
    if m5_idx < 0:
        return {**{k: 0.0 for k in PER_TF_FEATURE_NAMES}, **{k: 0.0 for k in GROUP_A_FEATURE_NAMES},
                **{k: 0.0 for k in DIP_STRUCT_FEATURE_NAMES}, **{k: 0.0 for k in FVG_FEATURE_NAMES}}
    current_price = float(ctx.m5_close[m5_idx])
    m5_feats = ctx.multi_tf.get("M5")
    if m5_feats is not None and m5_feats.attrs.get("feats_np") is not None:
        m5_ts_arr = m5_feats.attrs["ts_int64"]
        right = np.searchsorted(m5_ts_arr, ts_ns, side="right")
        if right > 0:
            current_atr = float(m5_feats.attrs["feats_np"][right - 1, 0] / 1e4 * current_price)
        else:
            current_atr = 1.5
    else:
        current_atr = 1.5

    out: dict[str, float] = {}
    per_tf = _per_tf_all(ctx, ts_ns)
    liq = _liquidity_zones(ctx, ts_ns, current_price, current_atr)
    out.update(per_tf)                                         # 125
    out.update(_session_overlap(ts))                           # 4
    out.update(_vol_term(ctx, ts_ns))                          # 4
    out.update(_vol_pct(ctx, ts_ns))                           # 2
    out.update(_pivots(ctx, ts, current_atr, current_price))   # 4
    if _AUG_ROUND_ON:
        out.update(_round_number_levels(current_price, current_atr))   # 5 (env-gated)
    out.update(liq)                                            # 10 (5 TFs × hi/lo)
    out.update(_per_side_perf(ctx, ts))                        # 8
    out.update(_dip_struct_5tf(per_tf, liq))                   # 38 (dip + struct)
    if _FVG_ON:
        out.update(_fvg_5tf(ctx, ts_ns, current_price, current_atr))   # 6 (3 TFs × {dist, active})
    return out


DEFAULT_MULTI_TF_V2_CACHE_DIR = "/home/andre2/GX1_DATA/data/data/prebuilt/MULTI_TF_V2_CACHE"


def attach_group_a_dip_struct_ctx_columns(
    df: pd.DataFrame,
    *,
    cache_dir: str | None = None,
    multi_tf: dict | None = None,
    journal_label: str = "parity",
    smc_col_candidates: tuple[str, ...] = ("smc_swing_state", "smc_swing_state_canon_v1"),
) -> pd.DataFrame:
    """Add the 24 GROUP-A parity + 36 dip/struct ctx_cont columns to ``df`` in place.

    ONE TRUTH for the V10 entry builder, the V3 exit builder, and the
    inference-batch candidate generator (serving). All three compute these
    ctx_cont features identically from :func:`augment_candidate`, so the model
    sees the same values at train and inference time (no train/serve skew).

    Requirements:
      - ``df`` has lowercase ``high``/``low``/``close`` columns.
      - ``df`` has a ``time`` column OR a tz-aware ``DatetimeIndex``.

    Idempotent: returns immediately if all 60 columns are already present.
    Mirrors the cemented-V10 builder block exactly (including the
    ``struct_smc_swing_x_dip_v3`` SMC×dip-proximity derivation).
    """
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_GROUP_A_PARITY as _GROUP_A,
        ORDERED_CTX_CONT_DIP_STRUCT as _DIP_STRUCT,
    )
    from gx1.features.htf_features import load_multi_tf_v2_cache

    need = list(_GROUP_A) + list(_DIP_STRUCT)
    if all(c in df.columns for c in need):
        return df

    for c in ("high", "low", "close"):
        if c not in df.columns:
            raise RuntimeError(
                f"[CTX_CONT_PARITY] df missing required OHLC column '{c}'"
            )
    if "time" in df.columns:
        ts_index = pd.to_datetime(df["time"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_index = pd.to_datetime(df.index, utc=True)
    else:
        raise RuntimeError("[CTX_CONT_PARITY] df needs a 'time' column or DatetimeIndex")

    m5 = df[["high", "low", "close"]].copy()
    m5.index = ts_index.to_numpy()
    m5 = m5.sort_index()
    # Multi-TF source: either an explicitly-provided in-memory bundle (LIVE serve passes
    # the mtf it just built from this SAME cv3 → staleness structurally impossible, no
    # on-disk dependency) OR the on-disk V2 cache (BUILD pipeline: pinned cv3 + a freshly
    # regenerated workspace cache). Both come from build_multi_tf_per_bar_features_v2() on
    # the same bars and the features are causal/asof, so for any decision ts the values are
    # bit-identical → train==serve preserved either way. (rule: make running-stale
    # IMPOSSIBLE for live, not "remember to refresh the cache".)
    if multi_tf is None:
        cache = cache_dir or os.environ.get(
            "GX1_V10_MULTI_TF_V2_CACHE_DIR", DEFAULT_MULTI_TF_V2_CACHE_DIR
        )
        multi_tf = load_multi_tf_v2_cache(cache)
    ctx = build_context(
        m5, multi_tf,
        journal_dir=Path(f"/nonexistent_{journal_label}_journal"),
    )

    dip_from_aug = [f for f in _DIP_STRUCT if f != "struct_smc_swing_x_dip_v3"]
    extract = list(_GROUP_A) + dip_from_aug + ["dip_proximity_m5_v3"]
    cols = {k: np.zeros(len(df), dtype=np.float32) for k in extract}
    for i, ts in enumerate(pd.DatetimeIndex(ts_index)):
        feat = augment_candidate(ctx, ts)
        for k in extract:
            cols[k][i] = float(feat.get(k, 0.0))
    for k in (list(_GROUP_A) + dip_from_aug):
        df[k] = cols[k]

    # struct_smc_swing_x_dip_v3 = clip(smc_swing_state / max|·|, -1, 1) × dip_proximity_m5_v3
    smc_col = next((c for c in smc_col_candidates if c in df.columns), None)
    if smc_col is not None:
        sw = pd.to_numeric(df[smc_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        max_abs = float(np.abs(sw).max()) if len(sw) else 1.0
        sw_norm = np.clip(sw / max(max_abs, 1.0), -1.0, 1.0)
        df["struct_smc_swing_x_dip_v3"] = (sw_norm * cols["dip_proximity_m5_v3"]).astype(np.float32)
    else:
        df["struct_smc_swing_x_dip_v3"] = np.zeros(len(df), dtype=np.float32)
    return df


def augment_week(week_pq: Path, out_pq: Path, ctx: AugmentContext,
                 smc_cache: pd.DataFrame | None = None) -> dict:
    df = pd.read_parquet(week_pq)
    n = len(df)
    if n == 0:
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)
        return {"week": week_pq.stem, "n": 0, "skipped": True}
    if "decision_ts_utc" not in df.columns:
        raise RuntimeError(f"{week_pq} missing decision_ts_utc")
    # 2026-05-24: drop existing V2/group-A/SMC cols if input was pre-augmented
    # (prevents duplicate cols when re-running on V10V2_AUGMENTED with bug-fixes).
    cols_to_overwrite = (set(PER_TF_FEATURE_NAMES) | set(GROUP_A_FEATURE_NAMES)
                         | set(GROUP_S_SMC_FEATURE_NAMES) | set(DIP_STRUCT_FEATURE_NAMES)
                         | set(FVG_FEATURE_NAMES))   # FVG_FEATURE_NAMES = () when GX1_FVG_FEATURES off
    existing_to_drop = [c for c in df.columns if c in cols_to_overwrite]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
    ts_arr = pd.to_datetime(df["decision_ts_utc"], utc=True)
    new_cols: dict[str, list[float]] = {
        name: [] for name in (PER_TF_FEATURE_NAMES + GROUP_A_FEATURE_NAMES + DIP_STRUCT_FEATURE_NAMES
                              + FVG_FEATURE_NAMES)
    }
    t0 = time.time()
    for ts in ts_arr:
        d = augment_candidate(ctx, ts)
        for k, v in d.items():
            new_cols[k].append(v)
    # Build new cols as a DataFrame, then concat once (avoids fragmentation warning).
    new_df = pd.DataFrame({k: np.asarray(vals, dtype=np.float32) for k, vals in new_cols.items()},
                          index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # 2026-05-24 BUG-3 FIX: join SMC features from canonical_v3 by decision_ts_utc.
    # smc_cache is the canonical_v3 SMC slice (time + smc_* cols), loaded once
    # in main() and passed in. We asof-join by time (canonical M5 bars).
    if smc_cache is not None and len(smc_cache) > 0:
        # Build asof-join: each candidate ts → nearest preceding M5 bar in canonical
        cand_ts = pd.to_datetime(df["decision_ts_utc"], utc=True).sort_values()
        cand_idx = pd.to_datetime(df["decision_ts_utc"], utc=True).reset_index(drop=True)
        smc_sorted = smc_cache.sort_values("time").reset_index(drop=True)
        merged = pd.merge_asof(
            cand_idx.to_frame("decision_ts_utc").sort_values("decision_ts_utc"),
            smc_sorted, left_on="decision_ts_utc", right_on="time",
            direction="backward", tolerance=pd.Timedelta("5min"),
        )
        # Reorder back to original df order
        merged = merged.set_index("decision_ts_utc").reindex(cand_idx).reset_index(drop=True)
        for src_col, dst_col in zip(
            ["smc_swing_state","smc_bos_up","smc_bos_down","smc_choch",
             "smc_sweep_up","smc_sweep_down","smc_sweep_size_atr",
             "smc_bars_since_sweep","smc_premium_discount","smc_premium_state"],
            GROUP_S_SMC_FEATURE_NAMES,
        ):
            if src_col in merged.columns:
                df[dst_col] = merged[src_col].fillna(0.0).to_numpy(dtype=np.float32)
            else:
                df[dst_col] = np.zeros(n, dtype=np.float32)
    else:
        # No SMC cache → fill zeros (graceful fallback for non-canonical-v3 sources)
        for dst_col in GROUP_S_SMC_FEATURE_NAMES:
            df[dst_col] = np.zeros(n, dtype=np.float32)

    # 2026-05-24 PM: struct_smc_swing_x_dip_v3 = SMC swing state × M5 dip proximity.
    # Computed here (after SMC join) since needs both signals.
    if "smc_swing_state_canon_v1" in df.columns and "dip_proximity_m5_v3" in df.columns:
        sw = pd.to_numeric(df["smc_swing_state_canon_v1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        max_abs = float(np.abs(sw).max()) if len(sw) else 1.0
        sw_norm = np.clip(sw / max(max_abs, 1.0), -1.0, 1.0)
        dp = pd.to_numeric(df["dip_proximity_m5_v3"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        df["struct_smc_swing_x_dip_v3"] = (sw_norm * dp).astype(np.float32)
    else:
        df["struct_smc_swing_x_dip_v3"] = np.zeros(n, dtype=np.float32)

    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_pq, index=False)
    elapsed = time.time() - t0
    return {"week": week_pq.stem, "n": n, "elapsed_sec": elapsed,
            "rate_per_sec": n / elapsed if elapsed > 0 else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-outcome-dir", type=Path, required=True,
                    help="explicit forward-outcome dir (no silent stale default; pass the current set)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--m5-prebuilt", type=Path, default=M5_PREBUILT)
    ap.add_argument("--n-weeks-test", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.forward_outcome_dir.with_name(
        args.forward_outcome_dir.name + "_AUGMENTED_V2"
    )
    out_per_week = out_dir / "per_week"
    out_per_week.mkdir(parents=True, exist_ok=True)
    print(f"[AUG_V2] source: {args.forward_outcome_dir}")
    print(f"[AUG_V2] output: {out_dir}")
    print(f"[AUG_V2] new cols per row: {len(PER_TF_FEATURE_NAMES) + len(GROUP_A_FEATURE_NAMES)} "
          f"(125 per-TF + 28 group-A)")

    print(f"[AUG_V2] loading M5 prebuilt + building V2 multi-TF cache...")
    t0 = time.time()
    m5_df = pd.read_parquet(args.m5_prebuilt, columns=["time", "open", "high", "low", "close", "volume"])
    m5_df["time"] = pd.to_datetime(m5_df["time"], utc=True)
    m5_df = m5_df.set_index("time").sort_index()
    multi_tf = build_multi_tf_per_bar_features_v2(m5_df)
    print(f"[AUG_V2]   M5={len(m5_df):,} bars  multi-TF in {time.time()-t0:.1f}s")

    print(f"[AUG_V2] building augment context (one-shot pre-compute)...")
    t1 = time.time()
    ctx = build_context(m5_df, multi_tf, JOURNAL_DIR)
    print(f"[AUG_V2]   context built in {time.time()-t1:.1f}s "
          f"(trade_history rows: {len(ctx.trade_history)})")

    # 2026-05-24 BUG-3 FIX: load SMC features from canonical_v3 prebuilt
    print(f"[AUG_V2] loading SMC features from canonical_v3 prebuilt...")
    t_smc = time.time()
    smc_cache = pd.DataFrame()
    try:
        smc_cols = ["time"] + [
            "smc_swing_state","smc_bos_up","smc_bos_down","smc_choch",
            "smc_sweep_up","smc_sweep_down","smc_sweep_size_atr",
            "smc_bars_since_sweep","smc_premium_discount","smc_premium_state",
        ]
        smc_cache = pd.read_parquet(args.m5_prebuilt, columns=smc_cols)
        smc_cache["time"] = pd.to_datetime(smc_cache["time"], utc=True)
        print(f"[AUG_V2]   SMC cache loaded: {len(smc_cache):,} rows × {len(smc_cols)-1} cols "
              f"in {time.time()-t_smc:.1f}s")
    except Exception as exc:
        print(f"[AUG_V2]   WARN: SMC load failed ({exc}) → SMC features will be zero")

    week_files = sorted((args.forward_outcome_dir / "per_week").glob("forward_outcomes_*.parquet"))
    if args.n_weeks_test > 0:
        week_files = week_files[:args.n_weeks_test]
        print(f"[AUG_V2] SMOKE TEST: first {args.n_weeks_test} weeks")
    print(f"[AUG_V2] processing {len(week_files)} weekly parquets...")

    total_n = 0; total_t = 0.0
    week_rows: dict[str, int] = {}
    skipped_existing: list[str] = []
    errors: list[str] = []
    for i, wp in enumerate(week_files):
        out_pq = out_per_week / wp.name
        if out_pq.exists():
            skipped_existing.append(wp.name)
            continue
        try:
            s = augment_week(wp, out_pq, ctx, smc_cache=smc_cache)
            week_rows[wp.name] = int(s["n"])
            total_n += s["n"]; total_t += s.get("elapsed_sec", 0)
            if (i+1) % 25 == 0 or i+1 == len(week_files):
                rate = total_n / max(total_t, 1e-6)
                print(f"  [{i+1}/{len(week_files)}] {wp.stem}  n={s['n']:>4}  "
                      f"({rate:.0f} cand/s, {total_n:,} done)", flush=True)
        except Exception as exc:
            errors.append(f"{wp.name}: {exc}")
            print(f"  [{i+1}/{len(week_files)}] {wp.stem}  ERROR: {exc}", flush=True)
        gc.collect()

    print(f"\n[AUG_V2] DONE — {total_n:,} candidates in {total_t/60:.1f} min "
          f"({total_n/max(total_t,1e-6):.0f} cand/s)")
    print(f"[AUG_V2] output → {out_dir}")
    # 2026-06-11: run-manifest (rule 4 — the step1feats build had NO manifest and had to be
    # verified forensically after a reboot wiped the only log). Records source, env-gates,
    # commit, per-week rowcounts and errors; status != DONE means PARTIAL — do not consume.
    import json as _json, subprocess as _sp
    try:
        _commit = _sp.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        _commit = "unknown"
    manifest = {
        "schema": "AUG_V2_MANIFEST_V1",
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "commit": _commit,
        "source_forward_outcome_dir": str(args.forward_outcome_dir),
        "m5_prebuilt": str(args.m5_prebuilt),
        "env_gates": {k: os.environ.get(k, "0") for k in
                      ("GX1_ROUND_NUMBER", "GX1_FVG_FEATURES", "GX1_SMC_SWEEP_RECLAIM")},
        "n_week_files_seen": len(week_files),
        "n_built": len(week_rows),
        "n_skipped_existing": len(skipped_existing),
        "rows_built_total": total_n,
        "per_week_rows": week_rows,
        "errors": errors,
        "status": "DONE" if not errors else "FAILED_PARTIAL",
    }
    (out_dir / "manifest_v1.json").write_text(_json.dumps(manifest, indent=2))
    print(f"[AUG_V2] manifest → {out_dir / 'manifest_v1.json'}  status={manifest['status']}")
    if errors:
        print(f"[AUG_V2] FAIL-LOUD: {len(errors)} week(s) errored — output is PARTIAL, do not consume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
