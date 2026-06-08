#!/usr/bin/env python3
"""Repair + extend a XAUUSD canonical tape (M1 or M5) for a corrupt/partial year.

PROBLEM: the 2026 canonical tapes (xauusd_m1_bid_ask__CANONICAL AND
xauusd_m5_bid_ask__CANONICAL) carry an x10-DEFLATION in April 2026 (median close
~482 vs real ~4800; M1: 15,777 bad bars, M5: 3,156) mixed with clean bars -> fake
+/-9000 bps jumps the honest gate band-aids by skipping April. The tapes also stop
short of the live feed.

WHY THIS GENERIC DRIVER (rule 7 — named the alternatives):
  - materialize_backfill_xauusd_m1_repair_v1._fetch_year_day_by_day is M1-hardcoded
    AND iterates to Dec 31 (future days -> OANDA 400 storm); its persist path runs the
    FULL-YEAR validator that rejects a partial year.
  - v12_backfill_to_present only extends (no repair).
  Neither fits "repair a corrupt PARTIAL current year for an arbitrary granularity".
  This driver: own day-by-day fetch CAPPED AT TODAY (no future-day storm), any
  granularity, partial-year x10-sanity + invariants + backup + full-replace.

Research-only data ingest. No model training. No runtime modification.
vedtak data_repair_extend_exit_retrain_20260608.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.execution.oanda_client import OandaAPIError  # noqa: E402
from gx1.scripts.materialize_backfill_xauusd_m1_repair_v1 import _client_from_env  # noqa: E402

INSTRUMENT = "XAU_USD"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _fetch_day_by_day(client, granularity: str, year: int):
    """Day-by-day fetch Jan 1 -> min(year-end, today). Empty/future days tolerated."""
    parts, ok_days, empty_days, err_days = [], 0, 0, 0
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    # cap at start of TODAY+1 so we never request future bars (no 400 storm)
    now = datetime.now(timezone.utc)
    end = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), now + timedelta(days=1))
    cur = start
    while cur < end:
        nxt = cur + timedelta(days=1)
        try:
            df = client.get_candles(INSTRUMENT, granularity, from_ts=pd.Timestamp(cur),
                                    to_ts=pd.Timestamp(nxt), include_mid=True, exclude_incomplete=True)
            if df is not None and len(df) > 0:
                parts.append(df); ok_days += 1
            else:
                empty_days += 1
        except OandaAPIError as exc:
            if "No candles returned" in str(exc):
                empty_days += 1
            else:
                err_days += 1
        time.sleep(0.1)  # ~10 req/s
        cur = nxt
    if not parts:
        return None, (ok_days, empty_days, err_days)
    df = pd.concat(parts, ignore_index=False)
    return df, (ok_days, empty_days, err_days)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"], utc=True)).drop(columns=["time"])
        else:
            raise SystemExit("fetched df has neither DatetimeIndex nor 'time' column")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df


def _april(d: pd.DataFrame):
    m = (d.index >= pd.Timestamp("2026-04-01", tz="UTC")) & (d.index < pd.Timestamp("2026-05-01", tz="UTC"))
    return d.loc[m, "close"].astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--granularity", required=True, choices=["M1", "M5"])
    ap.add_argument("--tape", required=True, help="canonical tape root (has year=YYYY/part-000.parquet)")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    part = Path(args.tape) / f"year={args.year}" / "part-000.parquet"
    if not part.exists():
        raise SystemExit(f"partition not found: {part}")
    tag = f"{args.granularity} {args.tape.split('/')[-1]} year={args.year}"

    print(f"[REPAIR] {tag}: re-fetching from OANDA (capped at today)...", flush=True)
    client = _client_from_env()
    df_new, (ok, empty, err) = _fetch_day_by_day(client, args.granularity, args.year)
    if df_new is None or len(df_new) == 0:
        raise SystemExit(f"[REPAIR] {tag}: fetch returned no rows — ABORT")
    df_new = _norm(df_new)
    print(f"[REPAIR] {tag}: fetched rows={len(df_new):,} ok_days={ok} empty={empty} err={err} "
          f"range {df_new.index.min()} -> {df_new.index.max()}", flush=True)

    old = pd.read_parquet(part)
    if not isinstance(old.index, pd.DatetimeIndex):
        old.index = pd.to_datetime(old["time"], utc=True) if "time" in old.columns else pd.to_datetime(old.index, utc=True)
    a_old, a_new = _april(old), _april(df_new)
    print(f"[REPAIR] {tag}: APRIL close OLD median={a_old.median():.1f} <1000={int((a_old<1000).sum())}  "
          f"NEW median={a_new.median():.1f} <1000={int((a_new<1000).sum())}", flush=True)

    fails = []
    cols = set(df_new.columns)
    if not {"open", "high", "low", "close"}.issubset({c.lower() for c in cols}):
        fails.append(f"missing OHLC (have {sorted(cols)[:12]})")
    if df_new.index.duplicated().any():
        fails.append("DUP_TS")
    for c in ("bid_close", "ask_close", "close"):
        if c in cols and (df_new[c] <= 0).any():
            fails.append(f"NEG_{c}")
    if "bid_close" in cols and "ask_close" in cols and not bool((df_new["bid_close"] <= df_new["ask_close"] + 1e-9).all()):
        fails.append("BID_GT_ASK")
    if len(a_new) == 0:
        fails.append("NO_APRIL_ROWS")
    elif a_new.median() < 4000:
        fails.append(f"APRIL_STILL_DEFLATED({a_new.median():.1f})")
    elif int((a_new < 1000).sum()) > 0:
        fails.append(f"APRIL_{int((a_new<1000).sum())}_SUB1000")
    if df_new.index.max() < pd.Timestamp("2026-06-04", tz="UTC"):
        fails.append(f"NOT_EXTENDED({df_new.index.max()})")
    if len(df_new) < len(old) * 0.95:
        fails.append(f"TOO_SHORT({len(df_new)} vs {len(old)})")
    missing = [c for c in old.columns if c != "time" and c not in df_new.columns]
    if missing:
        fails.append(f"MISSING_COLS{missing}")
    if fails:
        raise SystemExit(f"[REPAIR] {tag}: VALIDATION FAILED — NOT writing. {fails}")
    print(f"[REPAIR] {tag}: validation PASS.", flush=True)

    bak = part.with_suffix(f".parquet.x10corrupt_{_stamp()}.bak")
    shutil.copy2(part, bak)
    old_cols = [c for c in old.columns if c != "time"]
    out_cols = old_cols + [c for c in df_new.columns if c not in old_cols and c != "time"]
    df_out = df_new[out_cols]
    df_out.index.name = "time"
    tmp = part.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp, index=True)
    tmp.replace(part)
    print(f"[REPAIR] {tag}: backed up -> {bak.name}; WROTE rows={len(df_out):,}", flush=True)

    chk = pd.read_parquet(part)
    if not isinstance(chk.index, pd.DatetimeIndex):
        chk.index = pd.to_datetime(chk.index, utc=True)
    ac = _april(chk)
    print(f"[REPAIR] {tag}: VERIFY rows={len(chk):,} range {chk.index.min()} -> {chk.index.max()} "
          f"APRIL median={ac.median():.1f} <1000={int((ac<1000).sum())}", flush=True)
    print(f"[REPAIR] {tag}: DONE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
