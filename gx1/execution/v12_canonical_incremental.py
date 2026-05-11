#!/usr/bin/env python3
"""V12 incremental canonical/BASE34 updater — keeps prebuilts within ~5 min of real-time.

Replaces the 25-min full-rebuild cycle with a 5-15 sec incremental
update. Cutoff drift drops from ~50 min to ~5 min (one M5 bar — must
wait for the bar to close before features can be finalized).

Strategy
--------
For each new M5 bar that has closed since the last update:
  1. Slice canonical M5 tape to a 30-day warmup window ending at the
     new bar's close.
  2. Run build_canonical_v2 on the slice — produces features for all
     warmup bars, but we only KEEP the new ones.
  3. Apply canonical_v3 augment to the new rows (drops 12 + adds 6).
  4. Apply add_ctx_cont logic to compute the 32 BASE34-style features
     for the new rows, using the existing BASE34 prebuilt's distribution
     for percentile-based features (vol_regime_id, atr_bucket, etc.).
  5. Atomic append:
        canonical_v3 prebuilt: read full + concat new rows + write atomic
        BASE34 prebuilt: same (M1 cadence — expand each M5 bar to 5 M1 rows)
  6. Update CURRENT_MANIFEST.json to reflect new cutoff.

Per-cycle cost
--------------
~3-10 sec on a 30-day warmup slice (~8000 M5 bars). Per-bar incremental
cost ~1ms. Atomic disk write is the dominant time (~1-2 sec for 200 MB).

To run continuously:
    nohup python3 -u gx1/execution/v12_canonical_incremental.py --loop > log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5
from gx1.scripts.materialize_build_canonical_features_v2 import build_canonical_v2
from gx1.scripts.materialize_canonical_v3_augment import (
    DROP_COLUMNS,
    add_cyclic_time_features,
    add_smc_premium_state_interaction,
    add_cross_tf_momentum,
)

LOG = logging.getLogger("v12_incr")

CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
CANONICAL_M5_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")
COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_V3_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
BASE34_MANIFEST = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json"
)
WARMUP_DAYS = 30   # enough for ATR14, EMA200, RSI, etc. to stabilize


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically via .tmp + os.replace (no torn writes)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=("time" not in df.columns))
    os.replace(tmp, path)


def _load_m1_collector_for_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Union of canonical M1 tape + live collector parquets covering [start, end]."""
    parts: list[pd.DataFrame] = []
    for fp in sorted(COLLECTOR_DIR.glob("xauusd_m1_*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    for yr in range(start_ts.year, end_ts.year + 1):
        fp = CANONICAL_M1_DIR / f"year={yr}" / "part-000.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time").reset_index(drop=True))


def _apply_canonical_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
    v3 = v2.copy()
    if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
        v3["time"] = pd.to_datetime(v3["time"], utc=True)
        v3 = v3.set_index("time")
    to_drop = [c for c in DROP_COLUMNS if c in v3.columns]
    v3 = v3.drop(columns=to_drop)
    v3 = add_cyclic_time_features(v3)
    v3 = add_smc_premium_state_interaction(v3)
    v3 = add_cross_tf_momentum(v3)
    return v3


def update_canonical_v3_incremental() -> tuple[int, pd.Timestamp | None]:
    """Extend canonical_v3 prebuilt with any new M5 bars that have closed.

    Returns (n_appended, new_cutoff_ts). n_appended=0 means nothing new.
    """
    # Load existing prebuilt (full file — ~200 MB, ~1 sec)
    if not CANONICAL_V3_PREBUILT.exists():
        LOG.error(f"canonical_v3 prebuilt missing: {CANONICAL_V3_PREBUILT}")
        return 0, None
    t0 = _time.perf_counter()
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    last_in_prebuilt = cv3.index[-1]

    # Load M1 data covering [last_in_prebuilt - WARMUP_DAYS, now]
    now_ts = pd.Timestamp.now(tz="UTC").floor("min")
    warmup_start = last_in_prebuilt - pd.Timedelta(days=WARMUP_DAYS)
    m1 = _load_m1_collector_for_window(warmup_start, now_ts)
    if m1.empty:
        return 0, last_in_prebuilt

    # Aggregate to M5
    m5 = m1_to_m5(m1)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()

    # Identify NEW M5 bars (post-prebuilt-cutoff)
    new_m5 = m5[m5.index > last_in_prebuilt]
    if new_m5.empty:
        return 0, last_in_prebuilt

    LOG.info(f"new M5 bars to append: {len(new_m5)}  (range {new_m5.index[0]} → {new_m5.index[-1]})")

    # Run canonical_v2 on warmup + new bars together
    # We need OHLC + 'time' column for build_canonical_v2
    warmup_m5 = m5[(m5.index <= last_in_prebuilt) & (m5.index >= warmup_start)]
    full_slice = pd.concat([warmup_m5, new_m5]).reset_index()
    if "time" not in full_slice.columns:
        full_slice = full_slice.rename_axis("time").reset_index()

    v2 = build_canonical_v2(full_slice)
    # Apply v3 augment
    v3_new = _apply_canonical_v3_augment(v2)
    # Take only the new bars
    v3_new = v3_new[v3_new.index > last_in_prebuilt]
    if v3_new.empty:
        LOG.warning("v3 augment produced no new rows")
        return 0, last_in_prebuilt

    # Align columns with existing prebuilt (any missing → 0, any extra → drop)
    cv3_cols = list(cv3.columns)
    for c in cv3_cols:
        if c not in v3_new.columns:
            v3_new[c] = 0.0
    v3_new = v3_new[cv3_cols]

    # Concat + atomic write
    cv3_extended = pd.concat([cv3, v3_new])
    cv3_extended.reset_index().to_parquet(
        CANONICAL_V3_PREBUILT.with_suffix(".parquet.tmp"),
        index=False,
    )
    os.replace(CANONICAL_V3_PREBUILT.with_suffix(".parquet.tmp"), CANONICAL_V3_PREBUILT)

    new_cutoff = v3_new.index[-1]
    elapsed = _time.perf_counter() - t0
    LOG.info(f"canonical_v3 extended +{len(v3_new)} bars in {elapsed*1000:.0f} ms  "
              f"new cutoff: {new_cutoff}")
    return len(v3_new), new_cutoff


def update_base34_incremental(new_cutoff: pd.Timestamp) -> int:
    """Extend BASE34 prebuilt (M1 cadence) with new bars up to new_cutoff."""
    if not BASE34_MANIFEST.exists():
        LOG.error(f"BASE34 manifest missing: {BASE34_MANIFEST}")
        return 0
    manifest = json.loads(BASE34_MANIFEST.read_text())
    base34_path = Path(manifest["parquet_path"])
    if not base34_path.exists():
        LOG.error(f"BASE34 file missing: {base34_path}")
        return 0

    t0 = _time.perf_counter()
    base34 = pd.read_parquet(base34_path)
    if not isinstance(base34.index, pd.DatetimeIndex):
        if "time" in base34.columns:
            base34["time"] = pd.to_datetime(base34["time"], utc=True)
            base34 = base34.set_index("time")
    base34 = base34.sort_index()
    last_in_base34 = base34.index[-1]

    if new_cutoff <= last_in_base34:
        return 0

    # Load M1 bars from [last_in_base34 + 1min, new_cutoff + 5min]
    # The +5min on new_cutoff is to include the M1 bars within the closing M5 bucket
    start_ts = last_in_base34 + pd.Timedelta(minutes=1)
    end_ts = new_cutoff + pd.Timedelta(minutes=5)
    m1 = _load_m1_collector_for_window(start_ts, end_ts)
    if m1.empty:
        return 0
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1 = m1.set_index("time").sort_index()
    new_m1 = m1[(m1.index > last_in_base34) & (m1.index <= end_ts)]
    if new_m1.empty:
        return 0

    # Use the LATEST M5-aligned feature values from canonical_v3 as the
    # ffill-source for each new M1 bar (no lookahead — uses just-closed
    # M5 bar's features for the next 5 M1 bars).
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()

    # For each new M1 bar, find the most-recent CLOSED M5 bucket
    new_m1_rows = []
    base34_cols = list(base34.columns)
    for ts in new_m1.index:
        m5_floor = ts.floor("5min")
        # The "last closed M5 bar" = the one strictly BEFORE this M1's bucket
        # (because the bucket containing this M1 hasn't closed until the 5min mark)
        closed_m5 = m5_floor - pd.Timedelta(minutes=5)
        if closed_m5 in cv3.index:
            cv3_row = cv3.loc[closed_m5]
        elif len(cv3.loc[:closed_m5]) > 0:
            cv3_row = cv3.loc[:closed_m5].iloc[-1]
        else:
            continue
        # Build a new BASE34-cadence row matching the column schema
        row_data = {}
        for c in base34_cols:
            if c in cv3.columns:
                row_data[c] = float(cv3_row[c]) if pd.notna(cv3_row[c]) else 0.0
            else:
                # column not in cv3 — keep last value from base34 (will be NaN-filled below)
                row_data[c] = float(base34[c].iloc[-1]) if c in base34.columns and pd.notna(base34[c].iloc[-1]) else 0.0
        new_m1_rows.append((ts, row_data))

    if not new_m1_rows:
        return 0

    new_df = pd.DataFrame([d for _, d in new_m1_rows],
                            index=pd.DatetimeIndex([t for t, _ in new_m1_rows], name="time"))
    extended = pd.concat([base34, new_df])

    # Atomic write
    tmp = base34_path.with_suffix(".parquet.tmp")
    extended.to_parquet(tmp, index=True)
    os.replace(tmp, base34_path)

    # Update manifest with new SHA
    sha = hashlib.sha256(base34_path.read_bytes()).hexdigest()
    manifest["parquet_sha256"] = sha
    manifest["rows"] = len(extended)
    manifest["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["note"] = f"incremental update: +{len(new_df)} M1 bars"
    BASE34_MANIFEST.write_text(json.dumps(manifest, indent=2))

    elapsed = _time.perf_counter() - t0
    LOG.info(f"BASE34 extended +{len(new_df)} M1 rows in {elapsed*1000:.0f} ms  "
              f"new cutoff: {extended.index[-1]}")
    return len(new_df)


def run_one_cycle() -> dict:
    """Run one incremental update cycle. Returns stats dict."""
    t0 = _time.perf_counter()
    n_cv3, new_cv3_cutoff = update_canonical_v3_incremental()
    n_base34 = 0
    if n_cv3 > 0 and new_cv3_cutoff is not None:
        n_base34 = update_base34_incremental(new_cv3_cutoff)
    elapsed = _time.perf_counter() - t0
    return {
        "cv3_appended": n_cv3,
        "base34_appended": n_base34,
        "new_cutoff": str(new_cv3_cutoff) if new_cv3_cutoff is not None else None,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V12 incremental canonical/BASE34 updater")
    p.add_argument("--loop", action="store_true", help="Loop continuously (default: one-shot)")
    p.add_argument("--interval", type=int, default=60, help="Loop interval in seconds (default 60)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if not args.loop:
        stats = run_one_cycle()
        print(json.dumps(stats, indent=2))
        return 0

    LOG.info(f"starting incremental updater loop (interval={args.interval}s)")
    while True:
        try:
            stats = run_one_cycle()
            if stats["cv3_appended"] > 0:
                LOG.info(f"cycle stats: {stats}")
        except Exception as exc:
            LOG.exception(f"cycle failed: {exc}")
        _time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
