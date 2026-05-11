#!/usr/bin/env python3
"""V12 live feature builder — computes a curated 26-feature snapshot per M1
for counterfactual replay enrichment.

NOT the full 201-feature V12 contract — that comes in Phase B when we wire
the V12 stack. This module is Phase A: minimal analytical context to make
counterfactual journals interpretable ("which regime + ATR + session was
this missed-opportunity in?").

Feature groups:
  - Time/session (6): hour_utc, day_of_week, minute_of_5min, session,
                      minutes_to_session_change, is_news_hour
  - Price (4):        m1_mid, m5_close, range_1h_bps, range_24h_bps
  - Volatility (5):   atr14_m5_bps, atr14_m5_pct_2w, rv_30m_bps,
                      rv_2h_bps, spread_z_24h
  - Trend/EMA (5):    ema9, ema21, ema55, ema200, ema_stack_aligned
  - Microstructure (4): m1_body_pct, m1_upper_wick_pct,
                        m1_lower_wick_pct, m1_return_bps
  - Regime (1):       vol_regime ("low"/"normal"/"high")
  - Spread context (1): spread_bps (already known by caller; pass through)

Performance target: < 200ms per call. Achieved via M5-bucket caching —
M5-derived features (ATR, EMA, M5 close, regime) are computed once per
5-minute bucket and re-used for all M1 ticks in the same bucket.

Usage:
    builder = LiveFeatureBuilder(
        collector_dir=Path("/home/andre2/GX1_DATA/reports/v12_live_data"),
        canonical_m1_dir=Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL"),
    )
    feats = builder.compute(now_ts=current_minute, bid=bid, ask=ask, spread_bps=spread_bps)
    journal_event["features"] = feats
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("v12_live_features")

# Default lookback to keep enough history for ATR14 on M5 + EMA200 + 2w percentile.
# 2 weeks of M5 = 14d * 288 bars = 4032 bars. EMA200 needs ~400 to stabilize.
DEFAULT_LOOKBACK_DAYS = 14

# Session boundaries (UTC). XAUUSD is most liquid during EU+US overlap.
SESSION_BOUNDS = {
    "asia": (22, 7),   # 22:00-07:00 (wraps midnight)
    "eu":   (7, 13),
    "us":   (13, 21),
    "off":  (21, 22),  # 1h pre-Sydney
}

# News-impact hours (high typical vol due to scheduled releases):
#   06-08 UTC: EU open (PMI, ECB, etc.)
#   12-14 UTC: US morning data (CPI, NFP — usually 12:30 UTC)
#   17-18 UTC: Fed (decision releases)
NEWS_HOURS_UTC = {6, 7, 12, 13, 17}


def _utc_hour_to_session(hour: int) -> str:
    if 22 <= hour or hour < 7: return "asia"
    if 7 <= hour < 13:         return "eu"
    if 13 <= hour < 21:        return "us"
    return "off"


def _minutes_to_session_change(hour: int, minute: int) -> int:
    """Minutes remaining until the current UTC session ends."""
    cur = _utc_hour_to_session(hour)
    if cur == "asia":
        target_h = 7
    elif cur == "eu":
        target_h = 13
    elif cur == "us":
        target_h = 21
    else:  # off
        target_h = 22
    # Wrap-around handling for asia (22-07)
    if cur == "asia" and hour >= 22:
        target_h = 7 + 24
        cur_h = hour
    else:
        cur_h = hour
    return (target_h - cur_h) * 60 - minute


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR on OHLC."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=max(2, period // 2)).mean()


def _aggregate_m1_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """M1 → M5 OHLC. Drops incomplete final bucket (< 5 M1 bars).

    Required columns in m1: time, open, high, low, close, (volume optional).
    """
    df = m1.copy()
    df["m5_bucket"] = df["time"].dt.floor("5min")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    grouped = df.groupby("m5_bucket").agg(agg).reset_index()
    grouped = grouped.rename(columns={"m5_bucket": "time"})
    # Drop incomplete final bucket (< 5 M1 bars)
    counts = df.groupby("m5_bucket").size()
    full = counts[counts >= 5].index
    return grouped[grouped["time"].isin(full)].reset_index(drop=True)


@dataclass
class _M5Cache:
    """Cached M5-derived features keyed by latest M5 bucket timestamp."""
    last_bucket: pd.Timestamp | None = None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveFeatureBuilder:
    collector_dir: Path
    canonical_m1_dir: Path
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    _m5_cache: _M5Cache = field(default_factory=_M5Cache)

    def _load_m1_window(self, now_ts: pd.Timestamp) -> pd.DataFrame:
        """Union of collector parquets + canonical M1 tape covering [now-lookback, now]."""
        start_ts = now_ts - pd.Timedelta(days=self.lookback_days)
        parts: list[pd.DataFrame] = []

        # Collector (recent)
        for fp in sorted(self.collector_dir.glob("xauusd_m1_*.parquet")):
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            sub = df[(df["time"] >= start_ts) & (df["time"] <= now_ts)]
            if len(sub) > 0:
                parts.append(sub)

        # Canonical M1 tape (historical, last few years)
        for yr in range(start_ts.year, now_ts.year + 1):
            fp = self.canonical_m1_dir / f"year={yr}" / "part-000.parquet"
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            sub = df[(df["time"] >= start_ts) & (df["time"] <= now_ts)]
            if len(sub) > 0:
                parts.append(sub)

        if not parts:
            return pd.DataFrame()
        return (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True))

    def _compute_m5_features(self, now_ts: pd.Timestamp) -> dict[str, Any]:
        """Recompute M5-derived features. Cached per M5 bucket."""
        # Current 5-min bucket (e.g. 14:33 → 14:30)
        cur_bucket = now_ts.floor("5min")
        if self._m5_cache.last_bucket == cur_bucket and self._m5_cache.features:
            return self._m5_cache.features

        m1 = self._load_m1_window(now_ts)
        if m1.empty:
            return {}
        m5 = _aggregate_m1_to_m5(m1)
        if len(m5) < 50:   # need enough for ATR + EMA
            return {}

        # ATR14
        m5["atr14"] = _compute_atr(m5, period=14)
        last = m5.iloc[-1]
        cur_price = float(last["close"])
        atr14_bps = float(last["atr14"]) / cur_price * 10000.0 if cur_price > 0 else 0.0

        # ATR percentile over the loaded window (~2 weeks of M5)
        atr_pct_2w = float((m5["atr14"] <= last["atr14"]).mean()) if len(m5) >= 100 else 0.5

        # EMAs (price-based)
        ema9   = m5["close"].ewm(span=9,   adjust=False).mean().iloc[-1]
        ema21  = m5["close"].ewm(span=21,  adjust=False).mean().iloc[-1]
        ema55  = m5["close"].ewm(span=55,  adjust=False).mean().iloc[-1]
        ema200 = m5["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        # Stack aligned: +1 if 9>21>55>200 (bullish), -1 if reverse, 0 mixed
        if ema9 > ema21 > ema55 > ema200:
            stack = 1
        elif ema9 < ema21 < ema55 < ema200:
            stack = -1
        else:
            stack = 0
        dist_ema200_bps = (cur_price - ema200) / cur_price * 10000.0 if cur_price > 0 else 0.0

        # Range features
        bars_1h = m5.tail(12)   # 12 M5 = 1h
        bars_24h = m5.tail(288)
        range_1h_bps = (bars_1h["high"].max() - bars_1h["low"].min()) / cur_price * 10000.0
        range_24h_bps = (bars_24h["high"].max() - bars_24h["low"].min()) / cur_price * 10000.0

        # Vol regime — tercile split on the ATR pct
        if atr_pct_2w < 0.33:
            regime = "low"
        elif atr_pct_2w > 0.67:
            regime = "high"
        else:
            regime = "normal"

        # Realized vol (std of log-returns scaled to bps)
        m1_rets_bps = (np.log(m1["close"]).diff() * 10000.0).dropna()
        rv_30m_bps = float(m1_rets_bps.tail(30).std()) if len(m1_rets_bps) >= 5 else 0.0
        rv_2h_bps  = float(m1_rets_bps.tail(120).std()) if len(m1_rets_bps) >= 5 else 0.0

        feats = {
            "m5_close": cur_price,
            "atr14_m5_bps": atr14_bps,
            "atr14_m5_pct_2w": atr_pct_2w,
            "ema9_m5": float(ema9),
            "ema21_m5": float(ema21),
            "ema55_m5": float(ema55),
            "ema200_m5": float(ema200),
            "ema_stack_aligned": stack,
            "dist_to_ema200_bps": dist_ema200_bps,
            "range_1h_bps": float(range_1h_bps),
            "range_24h_bps": float(range_24h_bps),
            "rv_30m_bps": rv_30m_bps,
            "rv_2h_bps": rv_2h_bps,
            "vol_regime": regime,
            "_m1_recent_count": int(len(m1)),
        }
        self._m5_cache.last_bucket = cur_bucket
        self._m5_cache.features = feats
        # Also cache last 100 spread observations for spread_z
        recent_spreads = ((m1["ask_close"] - m1["bid_close"]) / m1["bid_close"] * 10000.0).tail(1440)  # 24h
        self._m5_cache.features["_spread_mean_24h"] = float(recent_spreads.mean()) if len(recent_spreads) else 0.0
        self._m5_cache.features["_spread_std_24h"] = float(recent_spreads.std()) if len(recent_spreads) > 1 else 1.0
        return feats

    def compute(self, now_ts: pd.Timestamp, *, bid: float, ask: float,
                 spread_bps: float) -> dict[str, Any]:
        """Build the 26-feature snapshot for the current minute.

        Args:
            now_ts: M1-aligned UTC timestamp (e.g. 2026-05-11 05:23:00+00:00).
            bid, ask, spread_bps: live OANDA quote at this moment.

        Returns dict of features. Empty dict if M1 history insufficient.
        """
        m5f = self._compute_m5_features(now_ts)
        if not m5f:
            return {}

        # Spread Z-score vs 24h
        sp_mean = m5f.get("_spread_mean_24h", spread_bps)
        sp_std = m5f.get("_spread_std_24h", 1.0) or 1.0
        spread_z = (spread_bps - sp_mean) / sp_std

        # Time/session
        h, m = now_ts.hour, now_ts.minute
        session = _utc_hour_to_session(h)
        is_news = h in NEWS_HOURS_UTC

        # Microstructure — needs the latest M1 bar (≤ now_ts)
        m1 = self._load_m1_window(now_ts)
        if not m1.empty:
            last_m1 = m1.iloc[-1]
            o, hi, lo, cl = float(last_m1["open"]), float(last_m1["high"]), float(last_m1["low"]), float(last_m1["close"])
            rng = max(hi - lo, 1e-9)
            body_pct = abs(cl - o) / rng
            upper_wick_pct = (hi - max(o, cl)) / rng
            lower_wick_pct = (min(o, cl) - lo) / rng
            ret_bps = (cl - float(m1["close"].iloc[-2])) / float(m1["close"].iloc[-2]) * 10000.0 if len(m1) >= 2 else 0.0
        else:
            body_pct = upper_wick_pct = lower_wick_pct = ret_bps = 0.0

        return {
            # time/session
            "hour_utc": h,
            "day_of_week": int(now_ts.dayofweek),
            "minute_of_5min": m % 5,
            "session": session,
            "minutes_to_session_change": _minutes_to_session_change(h, m),
            "is_news_hour": bool(is_news),
            # price (current quote already known to caller, but pass m5_close from cache)
            "m1_mid": (bid + ask) / 2,
            "m5_close": m5f["m5_close"],
            "range_1h_bps": m5f["range_1h_bps"],
            "range_24h_bps": m5f["range_24h_bps"],
            # volatility
            "atr14_m5_bps": m5f["atr14_m5_bps"],
            "atr14_m5_pct_2w": m5f["atr14_m5_pct_2w"],
            "rv_30m_bps": m5f["rv_30m_bps"],
            "rv_2h_bps": m5f["rv_2h_bps"],
            "spread_z_24h": float(spread_z),
            # trend
            "ema9_m5": m5f["ema9_m5"],
            "ema21_m5": m5f["ema21_m5"],
            "ema55_m5": m5f["ema55_m5"],
            "ema200_m5": m5f["ema200_m5"],
            "ema_stack_aligned": m5f["ema_stack_aligned"],
            "dist_to_ema200_bps": m5f["dist_to_ema200_bps"],
            # microstructure
            "m1_body_pct": float(body_pct),
            "m1_upper_wick_pct": float(upper_wick_pct),
            "m1_lower_wick_pct": float(lower_wick_pct),
            "m1_return_bps": float(ret_bps),
            # regime
            "vol_regime": m5f["vol_regime"],
        }
