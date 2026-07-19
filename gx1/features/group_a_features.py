"""Group A — internal-data features added in V2 rebuild (2026-05-22).

Six function families, all computable from local data (no external feeds):

  A1 - per_side_recent_performance(journal_path)        → 8 features
  A2 - vol_term_structure(multi_tf_cache, target_ts)    → 4 features
  A3 - realized_vol_percentile(m5_df, target_ts)        → 2 features
  A4 - session_overlap_markers(target_ts)               → 4 features
  A5 - liquidity_zones(m5_df, target_ts, atr)           → 6 features
  A6 - daily_pivot_levels(m5_df, target_ts, atr)        → 4 features

Total: 28 features per candidate/bar.

Used by the model-native Entry feature stack and the Exit-IQL bar-state
builder to expose context the transformer outputs alone do not capture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.htf_features import MULTI_TF_SHIFT

# Round-number / psychological $-level proximity (2026-06-11, env-gated GX1_ROUND_NUMBER, default OFF =
# contract byte-unchanged). The chain is otherwise ENTIRELY ATR-scale-invariant → structurally blind to
# absolute price. This is the deliberate exception: mod on ABSOLUTE price, THEN ATR-normalize the proximity
# (scale-invariant, no 2024-25 price-zone overfit). Gated everywhere so default-OFF keeps GROUP_A count = 28.
_ROUND_NUMBER_ON = os.environ.get("GX1_ROUND_NUMBER", "0") == "1"
ROUND_GRIDS = (100.0, 50.0, 25.0, 10.0)
ROUND_MAGNET_W = {100.0: 1.0, 50.0: 0.6, 25.0: 0.35, 10.0: 0.2}
ROUND_MAGNET_SCALE = 0.5  # ATR units
ROUND_FEATURE_NAMES = (
    "dist_to_round_100_atr", "dist_to_round_50_atr",
    "dist_to_round_25_atr", "dist_to_round_10_atr", "round_magnet_score",
)


def _require_utc_ts(value: object, *, context: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None or ts.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"[{context}] timestamp must be finite UTC: {value!r}")
    return ts.tz_convert("UTC")


def _require_market_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"[{context}] market frame must be a non-empty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise RuntimeError(f"[{context}] market frame needs a timezone-aware UTC DatetimeIndex")
    if pd.Timestamp(frame.index[0]).utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"[{context}] market frame index must be UTC")
    if frame.index.hasnans or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise RuntimeError(f"[{context}] timestamps must be finite, unique and chronological")
    missing = [name for name in ("high", "low", "close") if name not in frame.columns]
    if missing:
        raise RuntimeError(f"[{context}] exact OHLC sources missing: {missing}")
    numeric = frame[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"[{context}] OHLC sources must be finite")
    high = numeric["high"].to_numpy(dtype=np.float64)
    low = numeric["low"].to_numpy(dtype=np.float64)
    close = numeric["close"].to_numpy(dtype=np.float64)
    if np.any(low <= 0.0) or np.any(high < low) or np.any(close < low) or np.any(close > high):
        raise RuntimeError(f"[{context}] OHLC geometry is invalid")
    return numeric


def _require_positive_atr(value: object, *, context: str) -> float:
    atr = float(value)
    if not np.isfinite(atr) or atr <= 0.0:
        raise RuntimeError(f"[{context}] current_atr must be finite and positive")
    return atr


def _closed_tf_cutoff(target_ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    return target_ts + pd.Timedelta(minutes=5) - MULTI_TF_SHIFT[tf]


def round_number_levels(price: float, current_atr: float) -> dict[str, float]:
    """Signed ATR-distance to the nearest $100/$50/$25/$10 level + a decayed magnet score.
    `price` is the absolute close; the mod is on absolute price, the proximity is ATR-normalized."""
    price = float(price)
    atr_safe = _require_positive_atr(current_atr, context="GROUP_A_ROUND")
    if not np.isfinite(price) or price <= 0.0:
        raise RuntimeError("[GROUP_A_ROUND] price must be finite and positive")
    out: dict[str, float] = {}
    magnet = 0.0
    for g in ROUND_GRIDS:
        nearest = round(price / g) * g
        d_atr = (price - nearest) / atr_safe          # signed, ATR-normalized
        out[f"dist_to_round_{int(g)}_atr"] = float(d_atr)
        magnet += ROUND_MAGNET_W[g] * float(np.exp(-abs(d_atr) / ROUND_MAGNET_SCALE))
    out["round_magnet_score"] = float(magnet)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# A4 — Session overlap markers (cheapest — purely time-based)
# ─────────────────────────────────────────────────────────────────────────────

# UTC hour ranges per standard FX session
ASIA_HOURS = set(list(range(22, 24)) + list(range(0, 9)))   # 22:00–09:00 UTC
EU_HOURS   = set(range(7, 17))                              # 07:00–17:00 UTC
US_HOURS   = set(range(13, 22))                             # 13:00–22:00 UTC


def session_overlap_markers(ts: pd.Timestamp) -> dict[str, float]:
    """Return 4 mutually-non-exclusive session-overlap booleans (as 0/1 floats)."""
    ts = _require_utc_ts(ts, context="GROUP_A_SESSION")
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
    target_ts = _require_utc_ts(target_ts, context="GROUP_A_VOL_TERM")
    expected = {"M5", "M15", "H1", "H4", "D1"}
    if not isinstance(multi_tf_cache, dict) or set(multi_tf_cache) != expected:
        raise RuntimeError(
            f"[GROUP_A_VOL_TERM] exact TF cache required: expected={sorted(expected)} "
            f"observed={sorted(map(str, multi_tf_cache)) if isinstance(multi_tf_cache, dict) else None}"
        )

    def _last_atr(tf: str) -> float:
        feats = multi_tf_cache[tf]
        if not isinstance(feats, pd.DataFrame) or feats.empty or "atr_bps_14" not in feats.columns:
            raise RuntimeError(f"[GROUP_A_VOL_TERM] {tf}.atr_bps_14 source missing")
        if not isinstance(feats.index, pd.DatetimeIndex) or feats.index.tz is None:
            raise RuntimeError(f"[GROUP_A_VOL_TERM] {tf} needs a UTC DatetimeIndex")
        if feats.index.hasnans or not feats.index.is_monotonic_increasing or not feats.index.is_unique:
            raise RuntimeError(f"[GROUP_A_VOL_TERM] {tf} timestamps are invalid")
        eligible = feats.loc[feats.index <= _closed_tf_cutoff(target_ts, tf)]
        if eligible.empty:
            raise RuntimeError(f"[GROUP_A_VOL_TERM_WARMUP] no closed {tf} bar")
        value = float(pd.to_numeric(eligible["atr_bps_14"], errors="coerce").iloc[-1])
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"[GROUP_A_VOL_TERM] {tf}.atr_bps_14 must be finite and positive")
        return value

    a_m5  = _last_atr("M5")
    a_m15 = _last_atr("M15")
    a_h1  = _last_atr("H1")
    a_h4  = _last_atr("H4")
    a_d1  = _last_atr("D1")

    def _ratio(num, den):
        return float(min(50.0, num / den))

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
    target_ts = _require_utc_ts(target_ts, context="GROUP_A_VOL_PERCENTILE")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days <= 0:
        raise RuntimeError("[GROUP_A_VOL_PERCENTILE] lookback_days must be a positive integer")
    numeric = _require_market_frame(m5_df, context="GROUP_A_VOL_PERCENTILE")
    if target_ts not in numeric.index:
        raise RuntimeError("[GROUP_A_VOL_PERCENTILE] target must be an exact M5 row")
    eligible = numeric.loc[:target_ts]
    lookback_start = target_ts - pd.Timedelta(days=lookback_days)

    # Wilder ATR is computed over the complete causal prefix; only the rank
    # reference is clipped to the requested trailing calendar window. This
    # avoids resetting the smoother at each split/window boundary.
    high, low, close = eligible["high"], eligible["low"], eligible["close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_m5_series = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_m5_rank = atr_m5_series.loc[atr_m5_series.index >= lookback_start]
    if atr_m5_rank.empty or not np.isfinite(atr_m5_rank.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("[GROUP_A_VOL_PERCENTILE_WARMUP] M5 ATR rank history unavailable")
    cur_atr_m5 = float(atr_m5_rank.iloc[-1])
    pct_m5 = float((atr_m5_rank < cur_atr_m5).mean())

    # H1 uses complete bars only. The resampled row containing target_ts is
    # excluded until its full one-hour period has closed.
    h1 = eligible.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    h1 = h1.loc[h1.index <= _closed_tf_cutoff(target_ts, "H1")]
    if h1.empty:
        raise RuntimeError("[GROUP_A_VOL_PERCENTILE_WARMUP] no closed H1 bar")
    tr_h1 = pd.concat([
        (h1["high"] - h1["low"]).abs(),
        (h1["high"] - h1["close"].shift(1)).abs(),
        (h1["low"] - h1["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_h1_series = tr_h1.ewm(alpha=1 / 14, adjust=False).mean()
    atr_h1_rank = atr_h1_series.loc[atr_h1_series.index >= lookback_start]
    if atr_h1_rank.empty or not np.isfinite(atr_h1_rank.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("[GROUP_A_VOL_PERCENTILE_WARMUP] H1 ATR rank history unavailable")
    cur_atr_h1 = float(atr_h1_rank.iloc[-1])
    pct_h1 = float((atr_h1_rank < cur_atr_h1).mean())

    return {"vol_pct_m5_1yr": pct_m5, "vol_pct_h1_1yr": pct_h1}


# ─────────────────────────────────────────────────────────────────────────────
# A6 — Daily pivot levels (R1/R2/S1/S2 relative to current price in ATR units)
# ─────────────────────────────────────────────────────────────────────────────

def daily_pivot_levels(m5_df: pd.DataFrame,
                        target_ts: pd.Timestamp,
                        current_atr: float) -> dict[str, float]:
    """Standard pivot R1/R2/S1/S2 based on prior D1 OHLC. Returns dists in ATR units."""
    target_ts = _require_utc_ts(target_ts, context="GROUP_A_PIVOT")
    atr = _require_positive_atr(current_atr, context="GROUP_A_PIVOT")
    numeric = _require_market_frame(m5_df, context="GROUP_A_PIVOT")
    if target_ts not in numeric.index:
        raise RuntimeError("[GROUP_A_PIVOT] target must be an exact M5 row")
    eligible = numeric.loc[:target_ts]
    target_day = target_ts.normalize()
    daily = eligible.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    prior_days = daily.loc[daily.index < target_day]
    if prior_days.empty:
        raise RuntimeError("[GROUP_A_PIVOT_WARMUP] no completed prior trading day")
    prior = prior_days.iloc[-1]
    high = float(prior["high"])
    low = float(prior["low"])
    close = float(prior["close"])
    current_price = float(eligible["close"].iloc[-1])

    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)

    return {
        "dist_to_R1_atr": (current_price - r1) / atr,
        "dist_to_R2_atr": (current_price - r2) / atr,
        "dist_to_S1_atr": (current_price - s1) / atr,
        "dist_to_S2_atr": (current_price - s2) / atr,
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
    target_ts = _require_utc_ts(target_ts, context="GROUP_A_LIQUIDITY")
    atr = _require_positive_atr(current_atr, context="GROUP_A_LIQUIDITY")
    numeric = _require_market_frame(m5_df, context="GROUP_A_LIQUIDITY")
    if target_ts not in numeric.index:
        raise RuntimeError("[GROUP_A_LIQUIDITY] target must be an exact M5 row")
    expected_tf = {"M5", "M15", "H1", "H4", "D1"}
    if lookback_bars_per_tf is None:
        lookback_bars_per_tf = {"M5": 240, "M15": 192, "H1": 168, "H4": 168, "D1": 60}
    if set(lookback_bars_per_tf) != expected_tf or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in lookback_bars_per_tf.values()
    ):
        raise RuntimeError("[GROUP_A_LIQUIDITY] exact positive lookback contract required for all five TFs")
    eligible = numeric.loc[:target_ts]
    current_price = float(eligible["close"].iloc[-1])

    out: dict[str, float] = {}
    for tf_name, lookback_bars in lookback_bars_per_tf.items():
        if tf_name == "M5":
            window = eligible.tail(lookback_bars)
        elif tf_name == "M15":
            window = eligible.resample("15min").agg({"high": "max", "low": "min"}).dropna()
        elif tf_name == "H1":
            window = eligible.resample("1h").agg({"high": "max", "low": "min"}).dropna()
        elif tf_name == "H4":
            window = eligible.resample("4h").agg({"high": "max", "low": "min"}).dropna()
        else:
            window = eligible.resample("1D").agg({"high": "max", "low": "min"}).dropna()
        if tf_name != "M5":
            window = window.loc[window.index <= _closed_tf_cutoff(target_ts, tf_name)]
        window = window.tail(lookback_bars)
        if len(window) != lookback_bars:
            raise RuntimeError(
                f"[GROUP_A_LIQUIDITY_WARMUP] {tf_name} requires {lookback_bars} closed rows; "
                f"observed={len(window)}"
            )
        # Positive means an unswept level exists beyond price. When the whole
        # window has been swept, preserve the nearest observed level with a
        # negative sign instead of substituting a zero-distance sentinel.
        highs_above = window.loc[window["high"] > current_price, "high"]
        if len(highs_above) > 0:
            nearest_hi = float(highs_above.min())
        else:
            nearest_hi = float(window["high"].max())
        out[f"dist_to_{tf_name.lower()}_hi_atr"] = (nearest_hi - current_price) / atr
        lows_below = window.loc[window["low"] < current_price, "low"]
        if len(lows_below) > 0:
            nearest_lo = float(lows_below.max())
        else:
            nearest_lo = float(window["low"].min())
        out[f"dist_to_{tf_name.lower()}_lo_atr"] = (current_price - nearest_lo) / atr
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
    target = _require_utc_ts(target_ts, context="GROUP_A_PERF")
    if isinstance(lookback_n, bool) or not isinstance(lookback_n, int) or lookback_n <= 0:
        raise RuntimeError("[GROUP_A_PERF] lookback_n must be a positive integer")
    journal_dir = Path(journal_dir)
    if not journal_dir.is_dir():
        raise RuntimeError(f"[GROUP_A_PERF] journal directory missing: {journal_dir}")

    # Read most-recent 7 journal files (1 week)
    files = sorted(journal_dir.glob(f"v12_paper_journal_*_{suffix}.jsonl"))[-7:]
    if not files:
        raise RuntimeError("[GROUP_A_PERF] no exact journal files found")

    closed = []   # list of {ts, side, pnl_bps}
    last_snap: dict[str, dict] = {}
    prev_ids: set[str] = set()
    for fp in files:
        with fp.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"[GROUP_A_PERF] invalid JSON: {fp}:{line_no}") from exc
                if "open_trade_records" not in event or not isinstance(event["open_trade_records"], list):
                    raise RuntimeError(f"[GROUP_A_PERF] missing open_trade_records: {fp}:{line_no}")
                otrs = event["open_trade_records"]
                cur_ids = {str(record["trade_id"]) for record in otrs if record.get("trade_id")}
                for record in otrs:
                    tid = str(record.get("trade_id") or "")
                    if tid:
                        last_snap[tid] = {
                            "side": record.get("side"),
                            "pnl_bps": record.get("pnl_bps"),
                            "close_ts": event.get("logged_at_utc") or event.get("ts_utc"),
                        }
                for tid in (prev_ids - cur_ids):
                    if tid in last_snap and last_snap[tid].get("pnl_bps") is not None:
                        closed.append(last_snap.pop(tid))
                prev_ids = cur_ids

    if not closed:
        raise RuntimeError("[GROUP_A_PERF_WARMUP] no closed trades in journal history")

    out: dict[str, float] = {}
    for side in ("long", "short"):
        side_trades = [c for c in closed if c.get("side") == side]
        side_trades = [c for c in side_trades if c.get("close_ts")]
        if not side_trades:
            raise RuntimeError(f"[GROUP_A_PERF_WARMUP] no closed {side} trades")
        # Filter to closures before target_ts
        side_trades_dated = []
        for c in side_trades:
            ct = _require_utc_ts(c["close_ts"], context="GROUP_A_PERF")
            pnl = float(c["pnl_bps"])
            if not np.isfinite(pnl):
                raise RuntimeError("[GROUP_A_PERF] pnl_bps must be finite")
            if ct <= target:
                side_trades_dated.append((ct, pnl))
        if not side_trades_dated:
            raise RuntimeError(f"[GROUP_A_PERF_WARMUP] no closed {side} trades before target")
        side_trades_dated.sort(key=lambda x: x[0])
        last_n = side_trades_dated[-lookback_n:]
        pnls = np.array([p for _, p in last_n])
        wins = (pnls > 0).mean()
        mean_pnl = pnls.mean()
        # consecutive losses streak going backwards
        n_consec = 0
        for _, p in reversed(last_n):
            if p > 0:
                break
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
    # A5 — liquidity zones (10) — 2026-05-26 one-truth fix: added M15+D1 to match
    # liquidity_zones() (which computes all 5 TFs) + Entry/Exit-IQL V2_GROUP_A_COLS.
    # Previously stale at 6 (m5/h1/h4) → compute_group_a_features silently dropped m15/d1.
    "dist_to_m5_hi_atr", "dist_to_m5_lo_atr",
    "dist_to_m15_hi_atr", "dist_to_m15_lo_atr",
    "dist_to_h1_hi_atr", "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr", "dist_to_h4_lo_atr",
    "dist_to_d1_hi_atr", "dist_to_d1_lo_atr",
    # A1 — per-side perf (8)
    "long_win_rate_last10",  "long_mean_pnl_last10",
    "long_n_consec_losses",  "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
)
if _ROUND_NUMBER_ON:  # +5 ONLY under the flag → default-OFF keeps the base Group-A contract byte-identical
    GROUP_A_FEATURE_NAMES = GROUP_A_FEATURE_NAMES + ROUND_FEATURE_NAMES
GROUP_A_FEATURE_COUNT = len(GROUP_A_FEATURE_NAMES)   # = 32 (OFF) / 37 (GX1_ROUND_NUMBER=1)


def compute_group_a_features(
    target_ts: pd.Timestamp,
    *,
    m5_df: pd.DataFrame,
    multi_tf_cache: dict[str, pd.DataFrame],
    current_atr: float,
    journal_dir: Path,
) -> dict[str, float]:
    """Unified accessor: compute all 28 group-A features for one candidate/bar.

    Returns the exact ordered contract. Missing, stale, non-finite or
    under-warmed sources raise; no neutral dependency substitutions exist.
    """
    out: dict[str, float] = {}
    out.update(session_overlap_markers(target_ts))
    out.update(vol_term_structure(multi_tf_cache, target_ts))
    out.update(realized_vol_percentile(m5_df, target_ts))
    out.update(daily_pivot_levels(m5_df, target_ts, current_atr))
    out.update(liquidity_zones(m5_df, target_ts, current_atr))
    if _ROUND_NUMBER_ON:
        _elig = m5_df.loc[m5_df.index <= target_ts]
        if _elig.empty:
            raise RuntimeError("[GROUP_A] no exact current close for round-number features")
        out.update(round_number_levels(float(_elig["close"].iloc[-1]), current_atr))
    out.update(per_side_recent_performance(journal_dir, target_ts))
    missing = [name for name in GROUP_A_FEATURE_NAMES if name not in out]
    if missing:
        raise RuntimeError(f"[GROUP_A] output contract incomplete: {missing}")
    ordered = {name: float(out[name]) for name in GROUP_A_FEATURE_NAMES}
    if not np.isfinite(np.fromiter(ordered.values(), dtype=np.float64)).all():
        raise RuntimeError("[GROUP_A] output contract contains non-finite values")
    return ordered
