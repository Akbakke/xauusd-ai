#!/usr/bin/env python3
"""Build the canonical model-agnostic M5 feature parquet.

Mission
-------
Produce one canonical feature parquet covering the declared source interval.
Output is keyed by `time` and feeds the current model-native dataset/runtime
owners; it contains no decision policy.

Feature families
----------------
1. basic_v1 (~60 features): all `_v1_*` features, `_v1h1_*` H1 multi-TF,
   `_v1h4_*` H4 multi-TF — the canonical V10 input.
2. m5_phase (5 features): one-hot of (minute_of_hour // 12) ∈ {0..4}.
3. High-level basics (~15 features): atr, atr50, atr_z, ret_1/5/20, roc20/100,
   rvol_20/60, std50, vol_ratio, body_pct, wick_asym, ema20_slope, ema100_slope,
   pos_vs_ema200, mid, range. These match V10 manifest's "high-level" group.

Output
------
  canonical_features_v1.parquet  — single file, 449k rows × ~80 cols, indexed by `time`

Joinability
-----------
Downstream causal builders load this file and use `np.searchsorted` on the
time-array for O(log N) lookup at any
candidate or held-trade-bar timestamp.

This is RESEARCH-ONLY. No runtime promotion. Production V10/V3 features
remain canonical-runtime; this is a derivative offline view to feed IQL.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.features.basic_v1 import (  # noqa: E402
    PLUS5_FEATURES,
    build_basic_v1,
    compute_plus5_features,
)
ACTION = "BUILD_CANONICAL_FEATURES_V1"
DEFAULT_M5_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"
)


def load_m5_tape(m5_root: Path = DEFAULT_M5_TAPE_ROOT) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year_dir in sorted(m5_root.glob("year=*")):
        for parquet in sorted(year_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(parquet))
    if not parts:
        raise RuntimeError(f"[{ACTION}] no M5 parquets under {m5_root}")
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def add_m5_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 one-hot phase features = (minute_of_hour // 12)."""
    minute = df["time"].dt.minute
    phase = (minute // 12).clip(0, 4).astype(int)
    for i in range(5):
        df[f"m5_phase_{i}"] = (phase == i).astype(np.float32)
    return df


def add_high_level_basics(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic high-level feature columns matching V10 manifest names.

    Uses bid+ask mid for OHLC-derived calcs; bid_close for close.
    """
    mid = (df["bid_close"] + df["ask_close"]) / 2.0
    df["mid"] = mid.astype(np.float32)
    df["range"] = ((df["ask_high"] - df["bid_low"]) / np.maximum(mid, 1e-9) * 10000.0).astype(np.float32)

    # body_pct, wick_asym
    body_abs = (df["close"] - df["open"]).abs()
    bar_range = (df["high"] - df["low"]).clip(lower=1e-9)
    df["body_pct"] = (body_abs / bar_range).clip(lower=0.0, upper=1.0).astype(np.float32)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    wick_total = (upper_wick + lower_wick).clip(lower=1e-9)
    df["wick_asym"] = ((upper_wick - lower_wick) / wick_total).astype(np.float32)

    # ATR families (in price units, then as bps)
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df["atr50"] = tr.rolling(50, min_periods=1).mean().astype(np.float32)
    atr50_mean = df["atr50"].rolling(50, min_periods=10).mean()
    atr50_std = df["atr50"].rolling(50, min_periods=10).std().clip(lower=1e-9)
    df["atr_z"] = ((df["atr50"] - atr50_mean) / atr50_std).astype(np.float32)

    # Returns
    close = df["close"]
    df["ret_1"] = (close.pct_change(1) * 10000.0).astype(np.float32)
    df["ret_5"] = (close.pct_change(5) * 10000.0).astype(np.float32)
    df["ret_20"] = (close.pct_change(20) * 10000.0).astype(np.float32)
    df["roc100"] = (close.pct_change(100) * 10000.0).astype(np.float32)

    # Realized vol (bps)
    def rvol_window(window: int) -> pd.Series:
        return (
            close.pct_change()
            .rolling(window, min_periods=max(2, window // 4))
            .std()
            * 10000.0
            * np.sqrt(window)
        )
    df["rvol_20"] = rvol_window(20).astype(np.float32)
    df["rvol_60"] = rvol_window(60).astype(np.float32)

    # EMA slopes
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["ema20_slope"] = (ema20 - ema20.shift(5)).astype(np.float32)
    df["ema100_slope"] = (ema100 - ema100.shift(20)).astype(np.float32)
    df["pos_vs_ema200"] = ((close - ema200) / np.maximum(ema200, 1e-9) * 10000.0).astype(np.float32)

    # vol_ratio: rvol_20 / rvol_60
    df["vol_ratio"] = (df["rvol_20"] / df["rvol_60"].clip(lower=1e-6)).astype(np.float32)

    plus5 = compute_plus5_features(df)
    for feature in PLUS5_FEATURES:
        df[feature] = plus5[feature]
    return df


def build_basic_v1_chunked(
    df: pd.DataFrame,
    chunk_size: int = 100_000,
    *,
    decision_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Run the local-only basic_v1 owner over the full tape."""

    # basic_v1 expects DataFrame with DatetimeIndex
    work = df.set_index("time").copy()

    print(f"[{ACTION}] running basic_v1 on {len(work):,} rows...", flush=True)
    t0 = _time.time()
    if not isinstance(decision_bar_duration, pd.Timedelta) or decision_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[BUILD_CANONICAL_FEATURES_V1] decision bar duration must be positive")
    result = build_basic_v1(
        work,
        decision_delay_seconds=int(decision_bar_duration.total_seconds()),
    )
    if isinstance(result, tuple):
        result = result[0]
    elapsed = _time.time() - t0
    print(f"[{ACTION}] basic_v1 done in {elapsed:.1f}s, output shape: {result.shape}", flush=True)
    result = result.reset_index().rename(columns={"index": "time"})
    if "time" not in result.columns:
        result["time"] = work.index.values
    return result


def build_canonical_features(
    m5: pd.DataFrame,
    *,
    decision_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Apply basic_v1 + high-level basics + m5_phase. Returns full feature parquet."""
    feats = build_basic_v1_chunked(
        m5,
        decision_bar_duration=decision_bar_duration,
    )

    # Add high-level basics from raw bars
    feats = add_high_level_basics(feats)

    # Add m5_phase one-hot
    feats = add_m5_phase(feats)

    return feats
