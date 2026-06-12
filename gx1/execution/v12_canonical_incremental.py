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

# PLUS5: 5 features the v3 augment originally dropped as "duplicates". Re-added
# 2026-05-21 because PLUS5 Entry-IQL ensemble was trained on them with real values
# (mean test reward 95K vs 94K without). Logic mirrors
# augment_canonical_v3_with_missing_features.py:compute_features.
PLUS5_FEATURES = ("atr", "std50", "roc20", "_v1_vwap_drift48", "_v1h1_vwap_drift")


def _compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 PLUS5 features on an OHLCV DataFrame in place and return it.

    Expects columns: open, high, low, close, volume. DataFrame may be time-indexed
    or have a 'time' column; output preserves whichever input had.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(np.float64)
    high = pd.to_numeric(out["high"], errors="coerce").astype(np.float64)
    low = pd.to_numeric(out["low"], errors="coerce").astype(np.float64)
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(np.float64).replace(0, 1.0)
    pv = close * volume

    # 1. _v1_vwap_drift48: M5 48-period VWAP drift
    pv_48 = pv.rolling(48, min_periods=1).sum()
    v_48 = volume.rolling(48, min_periods=1).sum()
    vwap48 = pv_48 / v_48.replace(0, 1.0)
    out["_v1_vwap_drift48"] = ((close - vwap48) / vwap48.replace(0, 1.0)).astype(np.float32)

    # 2. _v1h1_vwap_drift: H1 VWAP drift (24 H1 bars ≈ 288 M5 bars)
    pv_h1 = pv.rolling(288, min_periods=12).sum()
    v_h1 = volume.rolling(288, min_periods=12).sum()
    vwap_h1 = pv_h1 / v_h1.replace(0, 1.0)
    out["_v1h1_vwap_drift"] = ((close - vwap_h1) / vwap_h1.replace(0, 1.0)).astype(np.float32)

    # 3. atr: Wilder M5 14-period ATR (EWMA alpha=1/14)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1/14, adjust=False).mean().astype(np.float32)

    # 4. std50: rolling std of M5 close-to-close returns
    rets = close.pct_change().fillna(0.0)
    out["std50"] = rets.rolling(50, min_periods=2).std().fillna(0.0).astype(np.float32)

    # 5. roc20: 20-period rate of change of close
    out["roc20"] = close.pct_change(20).fillna(0.0).astype(np.float32)

    return out


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
    # PLUS5: compute the 5 features on the full warmup slice (needs OHLCV history)
    # and merge into v3_new by index. Uses m5 which has OHLCV pre-augment.
    plus5_df = _compute_plus5_features(m5[["open", "high", "low", "close", "volume"]])
    for c in PLUS5_FEATURES:
        v3_new[c] = plus5_df[c].reindex(v3_new.index).astype(np.float32).fillna(0.0)
    # Phase 0a/C3 (2026-06-04): recompute the 5 HTF cols FRESH via the ONE-TRUTH
    # build_htf_tape (same math as the offline ctx builder) instead of letting the
    # BASE34 append forward-fill a FROZEN value (the H4-sign freeze: stale '2 bull' vs
    # true '0 bear', stuck since the last full build). Compute over the FULL tape
    # (existing cv3 OHLC + new bars) so the D1 270-bar percentile warmup is satisfied,
    # then assign to v3_new by index. These 5 cols PERSIST only once cv3's schema carries
    # them (one-shot backfill); pre-backfill the column-alignment below silently drops
    # them (safe no-op). A compute hiccup must NOT break the live append (that would
    # stale the whole pipeline) -> log loud + fall back to the prior forward-fill.
    try:
        from gx1.features.htf_features import build_htf_tape, HTF_TAPE_COLUMNS
        _ohlc = ["open", "high", "low", "close"]
        if all(c in cv3.columns for c in _ohlc):
            _full_ohlc = pd.concat([cv3[_ohlc], new_m5[_ohlc]]).sort_index()
            _full_ohlc = _full_ohlc[~_full_ohlc.index.duplicated(keep="last")]
            _htf = build_htf_tape(_full_ohlc)
            for c in HTF_TAPE_COLUMNS:
                v3_new[c] = _htf[c].reindex(v3_new.index)
        else:
            LOG.error("[C3_HTF] cv3 lacks OHLC — HTF recompute skipped (stale forward-fill remains)")
    except Exception as _htf_err:  # never crash the daemon append on an HTF hiccup
        LOG.error(f"[C3_HTF] HTF recompute FAILED ({_htf_err}); HTF left to forward-fill fallback")
    # Take only the new bars
    v3_new = v3_new[v3_new.index > last_in_prebuilt]
    if v3_new.empty:
        LOG.warning("v3 augment produced no new rows")
        return 0, last_in_prebuilt

    # Align columns with existing prebuilt (any missing → 0, any extra → drop).
    # PLUS5 cols are added to v3_new above; they will survive the alignment if
    # cv3 has them (after one-shot backfill).
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

    # 2026-06-11 FREEZE FIX: the 37 BASE34-only columns (32 ctx-augment features + session/regime/
    # swing/micro flags + is_model_bar) used to be COPY-FORWARDED from the last base34 row on every
    # append → ALL of them froze from 2026-05-25 18:25 (session pinned US, atr_bps 5.348, vol MEDIUM,
    # trend NEUTRAL — journal-confirmed all the way into the live entry/exit state vectors). They are
    # now RECOMPUTED per cycle via the ONE-TRUTH live augmenter (v12_ctx_augment_live.
    # augment_canonical_v3 — the docstring's step-4 intent, never implemented), on a trailing cv3
    # window long enough for the percentile features (D1_atr_percentile_252 needs ~1y of D1).
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    AUG_WINDOW_DAYS = 420
    cv3_win = cv3.loc[cv3.index >= (new_cutoff - pd.Timedelta(days=AUG_WINDOW_DAYS))]
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in cv3_win.columns]
    cv3_aug = augment_canonical_v3(cv3_win, cv3_win[_m5_cols].copy())
    _carry_warned: set = set()

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
            elif c == "is_model_bar":
                # marker for source M5 model-bar timestamps (extension builder: index.isin(model bars))
                row_data[c] = float(ts in cv3.index)
            elif c in cv3_aug.columns:
                # recomputed ctx value at the SAME last-closed M5 bar the cv3 features come from
                _v = cv3_aug.at[cv3_row.name, c] if cv3_row.name in cv3_aug.index else np.nan
                row_data[c] = float(_v) if pd.notna(_v) else 0.0
            else:
                # genuinely underivable column — carry last value, but LOUDLY (rule 9:
                # never a silent freeze again)
                if c not in _carry_warned:
                    LOG.warning(f"[BASE34] column '{c}' not in cv3 and not produced by the ctx "
                                f"augmenter — carrying last value (FROZEN). Fix the wiring.")
                    _carry_warned.add(c)
                row_data[c] = float(base34[c].iloc[-1]) if c in base34.columns and pd.notna(base34[c].iloc[-1]) else 0.0
        # V3-producer (2026-06-05): carry the RAW M1 OHLCV onto each base34 row (M1-NATIVE, NOT M5-ffilled).
        # base34 today holds only M5-derived features ffilled onto M1; the exit V3 transformer needs raw M1
        # volume/close (V3-consume: M1-native vol features, build_window) + high/low (V4: intrabar-high MFE).
        # Additive cols (downstream consumers read by name, ignore extras); the full-history rebuild adds them
        # to ALL rows so the interim NaN on pre-existing rows resolves at rebuild.
        for _oc in ("open", "high", "low", "close", "volume"):
            if _oc in new_m1.columns:
                _ov = new_m1.loc[ts, _oc]
                row_data[_oc] = float(_ov) if pd.notna(_ov) else 0.0
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


def backfill_base34_ctx(since_ts: pd.Timestamp) -> dict:
    """One-shot repair of the 2026-05-25 FREEZE: recompute the 37 BASE34-only ctx columns for all
    rows after `since_ts` via the same ONE-TRUTH augmenter the (fixed) incremental path uses.
    Run with the gx1-canonical-incremental daemon STOPPED (this rewrites the same parquet)."""
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    manifest = json.loads(BASE34_MANIFEST.read_text())
    base34_path = Path(manifest["parquet_path"])
    base34 = pd.read_parquet(base34_path)
    base34.index = pd.to_datetime(base34.index, utc=True)
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    win = cv3.loc[cv3.index >= (since_ts - pd.Timedelta(days=420))]
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in win.columns]
    cv3_aug = augment_canonical_v3(win, win[_m5_cols].copy())
    target = [c for c in base34.columns if c not in cv3.columns and c != "is_model_bar"
              and c in cv3_aug.columns]
    mask = base34.index > since_ts
    n_rows = int(mask.sum())
    # map each M1 row to its last CLOSED M5 bar (same semantics as the append path);
    # int64-ns on both sides (tz-aware vs naive .values would raise in searchsorted)
    closed_ns = (base34.index[mask].floor("5min") - pd.Timedelta(minutes=5)).asi8
    aug_idx = np.searchsorted(cv3_aug.index.asi8, closed_ns, side="right") - 1
    valid = aug_idx >= 0
    before = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    for c in target:
        vals = cv3_aug[c].to_numpy()[aug_idx]
        vals = np.where(valid, vals, np.nan)
        base34.loc[mask, c] = pd.Series(vals, index=base34.index[mask]).fillna(0.0).astype(np.float32)
    if "is_model_bar" in base34.columns:
        base34.loc[mask, "is_model_bar"] = base34.index[mask].isin(cv3.index)
    after = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    tmp = base34_path.with_suffix(".parquet.tmp")
    base34.to_parquet(tmp, index=True)
    os.replace(tmp, base34_path)
    manifest["parquet_sha256"] = hashlib.sha256(base34_path.read_bytes()).hexdigest()
    manifest["rows"] = len(base34)
    manifest["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["note"] = (manifest.get("note", "") +
                        f" | BACKFILL {datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}: recomputed "
                        f"{len(target)} frozen ctx cols for {n_rows} rows since {since_ts} (freeze fix).")
    BASE34_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return {"rows_backfilled": n_rows, "cols": len(target),
            "nunique_before_sample": before, "nunique_after_sample": after}


def main() -> int:
    p = argparse.ArgumentParser(description="V12 incremental canonical/BASE34 updater")
    p.add_argument("--loop", action="store_true", help="Loop continuously (default: one-shot)")
    p.add_argument("--interval", type=int, default=60, help="Loop interval in seconds (default 60)")
    p.add_argument("--backfill-base34-since", type=str, default=None,
                   help="One-shot: recompute the frozen BASE34 ctx cols for rows after this UTC ts "
                        "(freeze fix repair). Stop the daemon first.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if args.backfill_base34_since:
        stats = backfill_base34_ctx(pd.Timestamp(args.backfill_base34_since, tz="UTC"))
        print(json.dumps(stats, indent=2))
        return 0

    if not args.loop:
        stats = run_one_cycle()
        print(json.dumps(stats, indent=2))
        return 0

    LOG.info(f"starting incremental updater loop (interval={args.interval}s)")
    _cycles = 0
    while True:
        try:
            stats = run_one_cycle()
            if stats["cv3_appended"] > 0:
                LOG.info(f"cycle stats: {stats}")
        except Exception as exc:
            LOG.exception(f"cycle failed: {exc}")
        # Rule-9 LIVE-TAIL self-check (user vedtak 2026-06-11): every ~1h, scan the prebuilt
        # tails this daemon maintains for the freeze signature (was-varying → now-constant).
        # ERROR-loud, never fatal (killing the appender would stop data collection too) —
        # the launch_live_practice.sh preflight is the hard gate.
        _cycles += 1
        if _cycles % max(1, 3600 // max(args.interval, 1)) == 0:
            try:
                from gx1.audit.feature_liveness import check_live_prebuilt_tail, check_live_continuity
                _rep = check_live_prebuilt_tail()
                if not _rep["ok"]:
                    LOG.error(f"[RULE9-LIVE-TAIL] FREEZE SIGNATURE: {_rep['frozen']} — fix the append wiring NOW")
                else:
                    LOG.info(f"[RULE9-LIVE-TAIL] ok (stale_min={_rep['stale_minutes']})")
                _crep = check_live_continuity()
                if not _crep["ok"]:
                    LOG.error(f"[RULE9-CONTINUITY] FERSKE HULL: {_crep['fresh_gaps']}")
                else:
                    LOG.info(f"[RULE9-CONTINUITY] ok (freshness={_crep['freshness_min']})")
            except Exception as exc:
                LOG.error(f"[RULE9-LIVE-TAIL] self-check failed to run: {exc}")
        _time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
