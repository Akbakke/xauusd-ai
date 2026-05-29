#!/usr/bin/env python3
"""Augment canonical_features_v3.parquet with 5 features that were 0-filled
in both training and live (improvement 2026-05-21 user-requested).

5 features computed:
  - _v1_vwap_drift48     = (close - VWAP_48_M5) / VWAP_48_M5  ∈ ~[-0.01, 0.01]
  - _v1h1_vwap_drift     = (close - VWAP_24_H1) / VWAP_24_H1  (24 H1 bars ≈ 1 day)
  - atr                  = M5 14-period ATR (raw, in price units)
  - std50                = 50-period std of M5 returns
  - roc20                = 20-period rate of change of close

Input:  /home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V3/
            canonical_features_v3.parquet (113 cols)
Output: /home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V3_PLUS5/
            canonical_features_v3_plus5.parquet (118 cols)

Forward-outcome builder + Entry-IQL trainer should be re-run pointing at the
new file. Models WILL need retraining since these features were 0 in prior
training and now carry real values.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V3/canonical_features_v3.parquet")
DST_DIR = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V3_PLUS5")
DST = DST_DIR / "canonical_features_v3_plus5.parquet"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 missing features to canonical_v3 dataframe in-place."""
    # Ensure sorted by time
    df = df.sort_values("time").reset_index(drop=True)
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    volume = df["volume"].fillna(0).astype(np.float64).replace(0, 1.0)   # guard against 0-vol bars
    n = len(df)
    print(f"[AUG5] computing 5 features on {n:,} M5 rows", flush=True)

    # 1. _v1_vwap_drift48: M5 48-period VWAP drift
    pv = close * volume
    pv_48 = pv.rolling(48, min_periods=1).sum()
    v_48 = volume.rolling(48, min_periods=1).sum()
    vwap48 = pv_48 / v_48.replace(0, 1.0)
    df["_v1_vwap_drift48"] = ((close - vwap48) / vwap48.replace(0, 1.0)).astype(np.float32)

    # 2. _v1h1_vwap_drift: H1 VWAP drift (12 M5 bars/H1, 24 H1 bars = 288 M5)
    pv_h1 = pv.rolling(288, min_periods=12).sum()
    v_h1 = volume.rolling(288, min_periods=12).sum()
    vwap_h1 = pv_h1 / v_h1.replace(0, 1.0)
    df["_v1h1_vwap_drift"] = ((close - vwap_h1) / vwap_h1.replace(0, 1.0)).astype(np.float32)

    # 3. atr (raw) — Wilder M5 14-period ATR
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # EWMA approximation of Wilder: alpha = 1/14
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean().astype(np.float32)

    # 4. std50 — 50-period std of close-to-close returns
    rets = close.pct_change().fillna(0.0)
    df["std50"] = rets.rolling(50, min_periods=2).std().fillna(0.0).astype(np.float32)

    # 5. roc20 — 20-period rate of change
    roc20 = close.pct_change(20).fillna(0.0)
    df["roc20"] = roc20.astype(np.float32)

    # Sanity stats
    for c in ("_v1_vwap_drift48", "_v1h1_vwap_drift", "atr", "std50", "roc20"):
        nz = (df[c] != 0).sum()
        print(f"[AUG5]   {c}: nonzero={nz:,}  mean={df[c].mean():.6f}  std={df[c].std():.6f}",
              flush=True)

    return df


def main() -> int:
    print(f"[AUG5] loading source: {SRC}", flush=True)
    df = pd.read_parquet(SRC)
    print(f"[AUG5]   rows={len(df):,}  cols_in={len(df.columns)}", flush=True)
    df = compute_features(df)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"[AUG5] wrote {len(df):,} rows × {len(df.columns)} cols -> {DST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
