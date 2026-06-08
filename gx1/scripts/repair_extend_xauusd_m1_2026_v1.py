#!/usr/bin/env python3
"""Targeted repair + extend of the XAUUSD M1 canonical tape for 2026.

PROBLEM: year=2026 partition has an x10-DEFLATION corruption in April 2026
(~56% of April bars divided by 10 -> median close ~482 vs real ~4800), mixed
with clean bars -> fake +/-9000 bps jumps ("impossible trades" the honest gate
band-aids away by skipping April). It also stops at 2026-06-03 while the live
feed has data to ~today.

WHY A NEW THIN DRIVER (rule 7 — named the alternatives):
  - materialize_backfill_xauusd_m1_repair_v1 (_fetch_year_day_by_day) does the
    robust day-by-day OANDA fetch, but its persist path runs the FULL-YEAR
    validator which REJECTS a partial year (2026 = Jan..Jun ~150k < count bound).
  - v12_backfill_to_present only EXTENDS (doesn't repair April).
  Neither fits "repair a corrupt partial year AND extend". This driver REUSES the
  one-truth fetch helper + mirrors the partition format; the only new logic is
  partial-year x10-sanity + backup + full-replace.

ACTION: re-fetch 2026 FRESH from OANDA (Jan 1 -> today; clean + extends to present),
validate invariants + x10-sanity, back up the corrupt partition, REPLACE it.

Research-only data ingest. No model training. No runtime modification.
vedtak data_repair_extend_exit_retrain_20260608.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.scripts.materialize_backfill_xauusd_m1_repair_v1 import (  # noqa: E402
    _client_from_env,
    _fetch_year_day_by_day,
)

TAPE = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
PART = TAPE / "year=2026" / "part-000.parquet"
YEAR = 2026


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"], utc=True)).drop(columns=["time"])
        else:
            raise SystemExit("fetched df has neither DatetimeIndex nor 'time' column")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df


def main() -> int:
    print(f"[REPAIR2026] re-fetching {YEAR} M1 day-by-day from OANDA (clean + extend to today)...", flush=True)
    client = _client_from_env()
    df_new, audit = _fetch_year_day_by_day(client, YEAR)
    if df_new is None or len(df_new) == 0:
        raise SystemExit("[REPAIR2026] fetch returned no rows — ABORT (OANDA/creds?)")
    df_new = _norm_index(df_new)
    ok_days = sum(1 for a in audit if a.get("status_v1") == "OK")
    print(f"[REPAIR2026] fetched rows={len(df_new):,}  ok_days={ok_days}  "
          f"range {df_new.index.min()} -> {df_new.index.max()}", flush=True)

    # ---- existing (corrupt) partition for comparison ----
    old = pd.read_parquet(PART)
    if not isinstance(old.index, pd.DatetimeIndex):
        old.index = pd.to_datetime(old["time"], utc=True) if "time" in old.columns else pd.to_datetime(old.index, utc=True)
    pc = "close"
    def april(d):
        m = (d.index >= pd.Timestamp("2026-04-01", tz="UTC")) & (d.index < pd.Timestamp("2026-05-01", tz="UTC"))
        return d.loc[m, pc].astype(float)
    a_old, a_new = april(old), april(df_new)
    print(f"[REPAIR2026] APRIL close  OLD: median={a_old.median():.1f} <1000={int((a_old<1000).sum())}  "
          f"NEW: median={a_new.median():.1f} <1000={int((a_new<1000).sum())}", flush=True)

    # ---- VALIDATE fresh data (invariants + x10-sanity) ----
    fails = []
    cols = set(df_new.columns)
    need = {"open", "high", "low", "close"}
    if not need.issubset({c.lower() for c in cols}):
        fails.append(f"missing OHLC cols (have {sorted(cols)[:12]})")
    if df_new.index.duplicated().any():
        fails.append("DUPLICATE_TS")
    for c in ("bid_close", "ask_close"):
        if c in cols and (df_new[c] <= 0).any():
            fails.append(f"NEGATIVE_{c}")
    if "bid_close" in cols and "ask_close" in cols and not bool((df_new["bid_close"] <= df_new["ask_close"] + 1e-9).all()):
        fails.append("BID_GT_ASK")
    # x10-sanity: April must NOT be deflated
    if len(a_new) == 0:
        fails.append("NO_APRIL_ROWS_IN_FRESH")
    elif a_new.median() < 4000:
        fails.append(f"APRIL_STILL_DEFLATED (median {a_new.median():.1f})")
    elif int((a_new < 1000).sum()) > 0:
        fails.append(f"APRIL_HAS_{int((a_new<1000).sum())}_SUB1000_BARS")
    # extension sanity
    if df_new.index.max() < pd.Timestamp("2026-06-04", tz="UTC"):
        fails.append(f"NOT_EXTENDED (max {df_new.index.max()})")
    # don't lose data: fresh should cover >= existing span minus a small margin
    if len(df_new) < len(old) * 0.95:
        fails.append(f"FRESH_TOO_SHORT ({len(df_new)} vs old {len(old)})")
    if fails:
        raise SystemExit(f"[REPAIR2026] VALIDATION FAILED — NOT writing. fails={fails}")
    print("[REPAIR2026] validation PASS (invariants + April un-deflated + extended).", flush=True)

    # ---- backup + replace (rule 5: backup before overwrite) ----
    bak = PART.with_suffix(f".parquet.x10corrupt_{_stamp()}.bak")
    shutil.copy2(PART, bak)
    print(f"[REPAIR2026] backed up corrupt partition -> {bak.name}", flush=True)
    # Mirror the existing partition's column order when fresh covers them all;
    # otherwise keep fresh's own columns. Never carry a stray 'time' column.
    old_cols = [c for c in old.columns if c != "time"]
    if set(old_cols).issubset(set(df_new.columns)):
        out_cols = old_cols + [c for c in df_new.columns if c not in old_cols and c != "time"]
    else:
        out_cols = [c for c in df_new.columns if c != "time"]
    missing = [c for c in old_cols if c not in df_new.columns]
    if missing:
        raise SystemExit(f"[REPAIR2026] fresh data missing columns the tape has: {missing} — ABORT")
    df_out = df_new[out_cols]
    df_out.index.name = "time"
    tmp = PART.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp, index=True)
    tmp.replace(PART)
    print(f"[REPAIR2026] WROTE fresh partition rows={len(df_out):,} cols={list(df_out.columns)[:14]}", flush=True)

    # ---- verify on disk ----
    chk = pd.read_parquet(PART)
    if not isinstance(chk.index, pd.DatetimeIndex):
        chk.index = pd.to_datetime(chk.index, utc=True)
    ac = april(chk)
    print(f"[REPAIR2026] VERIFY on-disk: rows={len(chk):,} range {chk.index.min()} -> {chk.index.max()}  "
          f"APRIL median={ac.median():.1f} <1000={int((ac<1000).sum())}", flush=True)
    print("[REPAIR2026] DONE — April repaired + tape extended. Next: extend canonical_v3 features.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
