"""Group A — internal-data features added in V2 rebuild (2026-05-22).

Six function families, all computable from local data (no external feeds):

  A1 - per_side_recent_performance(journal_path)        → 8 features
  A2 - vol_term_structure(multi_tf_cache, target_ts)    → 4 features
  A3 - realized_vol_percentile(m5_df, target_ts)        → 2 features
  A4 - session_overlap_markers(target_ts)               → 4 features
  A5 - liquidity_zones(m5_df, target_ts, atr)           → 6 features
  A6 - daily_pivot_levels(m5_df, target_ts, atr)        → 4 features

Total: 28 features per candidate/bar.

Used by Entry-IQL state-builder and Exit-IQL bar_state-builder to give the
models direct access to context the V10/V3 transformer outputs alone don't
capture.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# A4 — Session overlap markers (cheapest — purely time-based)
# ─────────────────────────────────────────────────────────────────────────────

# UTC hour ranges per standard FX session
ASIA_HOURS = set(list(range(22, 24)) + list(range(0, 9)))   # 22:00–09:00 UTC
EU_HOURS   = set(range(7, 17))                              # 07:00–17:00 UTC
US_HOURS   = set(range(13, 22))                             # 13:00–22:00 UTC


def session_overlap_markers(ts: pd.Timestamp) -> dict[str, float]:
    """Return 4 mutually-non-exclusive session-overlap booleans (as 0/1 floats)."""
    h = ts.hour
    asia = h in ASIA_HOURS
    eu = h in EU_HOURS
    us = h in US_HOURS
    return {
        "is_asia_eu_overlap": float(asia and eu),         # ~07-08 UTC
        "is_eu_us_overlap":   float(eu and us),           # ~13-17 UTC
        "is_eu_only":         float(eu and not us and not asia),
        "is_us_only":         float(us and not eu and not asia),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Vol term structure (ATR ratios across TFs)
# ─────────────────────────────────────────────────────────────────────────────

def vol_term_structure(multi_tf_cache: dict[str, pd.DataFrame],
                       target_ts: pd.Timestamp) -> dict[str, float]:
    """ATR-based vol term structure — ratios indicate squeeze/expansion regime.

    multi_tf_cache: dict {TF_name: DataFrame with 'atr_bps_14' col + DatetimeIndex}
    Returns 4 ratios.
    """
    def _last_atr(tf: str) -> float:
        feats = multi_tf_cache.get(tf)
        if feats is None or len(feats) == 0 or "atr_bps_14" not in feats.columns:
            return 0.0
        eligible = feats.loc[feats.index <= target_ts]
        if eligible.empty:
            return 0.0
        return float(eligible["atr_bps_14"].iloc[-1])

    a_m5  = _last_atr("M5")
    a_m15 = _last_atr("M15")
    a_h1  = _last_atr("H1")
    a_h4  = _last_atr("H4")
    a_d1  = _last_atr("D1")

    def _ratio(num, den):
        # Clip to [0, 50] to avoid blow-ups when den is tiny (early bars).
        # ratio > 50 = degenerate; gives IQL stable input.
        return float(min(50.0, num / max(den, 1e-3)))

    return {
        "atr_ratio_m5_h4":  _ratio(a_m5, a_h4),   # squeeze: <0.5, expansion: >2
        "atr_ratio_m15_d1": _ratio(a_m15, a_d1),
        "atr_ratio_h1_d1":  _ratio(a_h1, a_d1),
        "atr_ratio_m5_m15": _ratio(a_m5, a_m15),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Realized vol percentile (current vs 1-yr distribution)
# ─────────────────────────────────────────────────────────────────────────────

def realized_vol_percentile(m5_df: pd.DataFrame,
                             target_ts: pd.Timestamp,
                             lookback_days: int = 365) -> dict[str, float]:
    """Current ATR vs trailing-1-year percentile.

    m5_df: DataFrame indexed by time with 'close' + 'high' + 'low' columns
    Returns 2 percentile values in [0, 1].
    """
    eligible = m5_df.loc[m5_df.index <= target_ts]
    if len(eligible) < 100:
        return {"vol_pct_m5_1yr": 0.5, "vol_pct_h1_1yr": 0.5}

    lookback_start = target_ts - pd.Timedelta(days=lookback_days)
    window = eligible.loc[eligible.index >= lookback_start]
    if len(window) < 100:
        window = eligible.tail(100)

    # Wilder ATR-14 on M5
    h, l, c = window["high"], window["low"], window["close"]
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_m5_series = tr.ewm(alpha=1/14, adjust=False).mean().dropna()
    cur_atr_m5 = float(atr_m5_series.iloc[-1]) if len(atr_m5_series) > 0 else 0.0
    if len(atr_m5_series) >= 10:
        pct_m5 = float((atr_m5_series < cur_atr_m5).mean())
    else:
        pct_m5 = 0.5

    # H1 by resample
    h1 = window.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(h1) >= 14:
        tr_h1 = pd.concat([
            (h1["high"] - h1["low"]).abs(),
            (h1["high"] - h1["close"].shift(1)).abs(),
            (h1["low"] - h1["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_h1_series = tr_h1.ewm(alpha=1/14, adjust=False).mean().dropna()
        cur_atr_h1 = float(atr_h1_series.iloc[-1]) if len(atr_h1_series) > 0 else 0.0
        pct_h1 = float((atr_h1_series < cur_atr_h1).mean()) if len(atr_h1_series) >= 10 else 0.5
    else:
        pct_h1 = 0.5

    return {"vol_pct_m5_1yr": pct_m5, "vol_pct_h1_1yr": pct_h1}


# ─────────────────────────────────────────────────────────────────────────────
# A6 — Daily pivot levels (R1/R2/S1/S2 relative to current price in ATR units)
# ─────────────────────────────────────────────────────────────────────────────

def daily_pivot_levels(m5_df: pd.DataFrame,
                        target_ts: pd.Timestamp,
                        current_atr: float) -> dict[str, float]:
    """Standard pivot R1/R2/S1/S2 based on prior D1 OHLC. Returns dists in ATR units."""
    # Get prior trading day's OHLC
    target_day = pd.Timestamp(target_ts).normalize()
    prior_day_end = target_day
    prior_day_start = target_day - pd.Timedelta(days=3)   # weekend-safe lookback
    prior_window = m5_df.loc[(m5_df.index >= prior_day_start) & (m5_df.index < prior_day_end)]
    if prior_window.empty:
        return {"dist_to_R1_atr": 0.0, "dist_to_R2_atr": 0.0,
                "dist_to_S1_atr": 0.0, "dist_to_S2_atr": 0.0}

    high = float(prior_window["high"].max())
    low = float(prior_window["low"].min())
    close = float(prior_window["close"].iloc[-1])
    current_price = float(m5_df.loc[m5_df.index <= target_ts, "close"].iloc[-1])

    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)

    atr_safe = max(current_atr, 1e-3)
    return {
        "dist_to_R1_atr": (current_price - r1) / atr_safe,
        "dist_to_R2_atr": (current_price - r2) / atr_safe,
        "dist_to_S1_atr": (current_price - s1) / atr_safe,
        "dist_to_S2_atr": (current_price - s2) / atr_safe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A5 — Liquidity zones (distance to nearest unswept high/low per TF)
# ─────────────────────────────────────────────────────────────────────────────

def liquidity_zones(m5_df: pd.DataFrame,
                     target_ts: pd.Timestamp,
                     current_atr: float,
                     lookback_bars_per_tf: dict[str, int] | None = None) -> dict[str, float]:
    """Distance to nearest unswept high/low per TF, in ATR units.

    "Unswept" = the high/low has not been broken by subsequent bars.
    We compute over rolling N-bar windows per TF.

    lookback_bars_per_tf default:
      M5: 240 (20 hours)
      H1: 168 (1 week)
      H4: 168 (4 weeks)
    """
    # 2026-05-24 FIX: add M15 + D1. Previously only M5/H1/H4 → Entry/Exit IQL
    # could not see M15 or D1 swing highs/lows, blocking "wait when next M15
    # turns down" behavior.
    if lookback_bars_per_tf is None:
        lookback_bars_per_tf = {"M5": 240, "M15": 192, "H1": 168, "H4": 168, "D1": 60}

    eligible = m5_df.loc[m5_df.index <= target_ts]
    if eligible.empty:
        return {f"dist_to_{tf}_{side}_atr": 0.0
                for tf in ("m5", "m15", "h1", "h4", "d1")
                for side in ("hi", "lo")}

    current_price = float(eligible["close"].iloc[-1])
    atr_safe = max(current_atr, 1e-3)

    out: dict[str, float] = {}
    for tf_name, lookback_bars in lookback_bars_per_tf.items():
        if tf_name == "M5":
            window = eligible.tail(lookback_bars)
        elif tf_name == "M15":
            tail = eligible.tail(lookback_bars * 3)    # 3 M5 / 1 M15
            window = tail.resample("15min").agg({"high": "max", "low": "min"}).dropna().tail(lookback_bars)
        elif tf_name == "H1":
            tail = eligible.tail(lookback_bars * 12)   # 12 M5 / 1 H1
            window = tail.resample("1h").agg({"high": "max", "low": "min"}).dropna().tail(lookback_bars)
        elif tf_name == "H4":
            tail = eligible.tail(lookback_bars * 48)   # 48 M5 / 1 H4
            window = tail.resample("4h").agg({"high": "max", "low": "min"}).dropna().tail(lookback_bars)
        elif tf_name == "D1":
            tail = eligible.tail(lookback_bars * 288)  # 288 M5 / 1 D1
            window = tail.resample("1D").agg({"high": "max", "low": "min"}).dropna().tail(lookback_bars)
        else:
            continue
        if window.empty:
            out[f"dist_to_{tf_name.lower()}_hi_atr"] = 0.0
            out[f"dist_to_{tf_name.lower()}_lo_atr"] = 0.0
            continue
        # Nearest UNSWEPT high above current: highest swing-high in window > current_price
        highs_above = window.loc[window["high"] > current_price, "high"]
        if len(highs_above) > 0:
            nearest_hi = float(highs_above.min())
            out[f"dist_to_{tf_name.lower()}_hi_atr"] = (nearest_hi - current_price) / atr_safe
        else:
            out[f"dist_to_{tf_name.lower()}_hi_atr"] = 0.0
        lows_below = window.loc[window["low"] < current_price, "low"]
        if len(lows_below) > 0:
            nearest_lo = float(lows_below.max())
            out[f"dist_to_{tf_name.lower()}_lo_atr"] = (current_price - nearest_lo) / atr_safe
        else:
            out[f"dist_to_{tf_name.lower()}_lo_atr"] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Per-side recent trade performance (reads paper-runner journal)
# ─────────────────────────────────────────────────────────────────────────────

def per_side_recent_performance(journal_dir: Path,
                                  target_ts: pd.Timestamp,
                                  lookback_n: int = 10,
                                  suffix: str = "live_v12_4") -> dict[str, float]:
    """Per-side stats over last N closed trades before target_ts.

    Reads v12_paper_journal_*_<suffix>.jsonl files, detects closures via
    open_trade_records delta tracking. Returns 8 features.

    Note: this is slow if called per-candidate. For backfill during dataset
    rebuild, call ONCE and cache per timestamp range. For live runtime,
    maintain a rolling cache that's updated when trades close.
    """
    default = {
        "long_win_rate_last10":   0.5, "long_mean_pnl_last10":   0.0,
        "long_n_consec_losses":   0.0, "long_time_since_last_close_min": 240.0,
        "short_win_rate_last10":  0.5, "short_mean_pnl_last10":  0.0,
        "short_n_consec_losses":  0.0, "short_time_since_last_close_min": 240.0,
    }

    if not journal_dir.exists():
        return default

    # Read most-recent 7 journal files (1 week)
    files = sorted(journal_dir.glob(f"v12_paper_journal_*_{suffix}.jsonl"))[-7:]
    if not files:
        return default

    closed = []   # list of {ts, side, pnl_bps}
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
                        if tid in last_snap and last_snap[tid].get("pnl_bps") is not None:
                            closed.append(last_snap[tid])
                            last_snap.pop(tid, None)
                    prev_ids = cur_ids
        except Exception:
            continue

    if not closed:
        return default

    target = pd.Timestamp(target_ts)
    out = dict(default)
    for side in ("long", "short"):
        side_trades = [c for c in closed if c.get("side") == side]
        side_trades = [c for c in side_trades if c.get("close_ts")]
        if not side_trades:
            continue
        # Filter to closures before target_ts
        side_trades_dated = []
        for c in side_trades:
            try:
                ct = pd.to_datetime(c["close_ts"], utc=True)
                if ct <= target:
                    side_trades_dated.append((ct, float(c["pnl_bps"])))
            except Exception:
                continue
        if not side_trades_dated:
            continue
        side_trades_dated.sort(key=lambda x: x[0])
        last_n = side_trades_dated[-lookback_n:]
        pnls = np.array([p for _, p in last_n])
        wins = (pnls > 0).mean()
        mean_pnl = pnls.mean()
        # consecutive losses streak going backwards
        n_consec = 0
        for _, p in reversed(last_n):
            if p > 0: break
            n_consec += 1
        last_close_ts = last_n[-1][0]
        mins_since = max(0.0, min(1440.0, (target - last_close_ts).total_seconds() / 60.0))
        out[f"{side}_win_rate_last10"] = float(wins)
        out[f"{side}_mean_pnl_last10"] = float(mean_pnl)
        out[f"{side}_n_consec_losses"] = float(n_consec)
        out[f"{side}_time_since_last_close_min"] = float(mins_since)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Unified accessor — call once per candidate/bar to get all 28 features
# ─────────────────────────────────────────────────────────────────────────────

GROUP_A_FEATURE_NAMES = (
    # A4 — overlap (4)
    "is_asia_eu_overlap", "is_eu_us_overlap", "is_eu_only", "is_us_only",
    # A2 — vol term (4)
    "atr_ratio_m5_h4", "atr_ratio_m15_d1", "atr_ratio_h1_d1", "atr_ratio_m5_m15",
    # A3 — vol percentile (2)
    "vol_pct_m5_1yr", "vol_pct_h1_1yr",
    # A6 — pivots (4)
    "dist_to_R1_atr", "dist_to_R2_atr", "dist_to_S1_atr", "dist_to_S2_atr",
    # A5 — liquidity zones (6)
    "dist_to_m5_hi_atr", "dist_to_m5_lo_atr",
    "dist_to_h1_hi_atr", "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr", "dist_to_h4_lo_atr",
    # A1 — per-side perf (8)
    "long_win_rate_last10",  "long_mean_pnl_last10",
    "long_n_consec_losses",  "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
)
GROUP_A_FEATURE_COUNT = len(GROUP_A_FEATURE_NAMES)   # = 28


def compute_group_a_features(
    target_ts: pd.Timestamp,
    *,
    m5_df: pd.DataFrame,
    multi_tf_cache: dict[str, pd.DataFrame],
    current_atr: float,
    journal_dir: Path | None = None,
) -> dict[str, float]:
    """Unified accessor: compute all 28 group-A features for one candidate/bar.

    Returns dict keyed by GROUP_A_FEATURE_NAMES. Missing dependencies default
    to neutral values (0.5 for percentile, 0 elsewhere).
    """
    out: dict[str, float] = {}
    out.update(session_overlap_markers(target_ts))
    out.update(vol_term_structure(multi_tf_cache, target_ts))
    out.update(realized_vol_percentile(m5_df, target_ts))
    out.update(daily_pivot_levels(m5_df, target_ts, current_atr))
    out.update(liquidity_zones(m5_df, target_ts, current_atr))
    if journal_dir is not None:
        out.update(per_side_recent_performance(journal_dir, target_ts))
    else:
        # Default neutral values
        for k in (
            "long_win_rate_last10", "long_mean_pnl_last10",
            "long_n_consec_losses", "long_time_since_last_close_min",
            "short_win_rate_last10", "short_mean_pnl_last10",
            "short_n_consec_losses", "short_time_since_last_close_min",
        ):
            out.setdefault(k, 0.5 if "win_rate" in k else (240.0 if "time_since" in k else 0.0))
    # Order by GROUP_A_FEATURE_NAMES
    return {k: float(out.get(k, 0.0)) for k in GROUP_A_FEATURE_NAMES}
