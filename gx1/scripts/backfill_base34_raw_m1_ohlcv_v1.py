#!/usr/bin/env python3
"""Backfill raw M1 OHLCV (open/high/low/close/volume) onto the FULL M1-expanded base34.

WHY (R12 + V4): the V3 exit transformer's M1-native volume features (idx 91-94, served
by v12_v3_live.py:307-318) and V4's intrabar-high MFE require base34 to carry raw M1
volume/close (+ high/low). The incremental daemon (v12_canonical_incremental.py:352-356)
emits these on NEW rows only; pre-existing base34 rows lack them, so the serve M1-native
branch never fires and falls back to M5-ffill (train != serve skew). No full-rebuild
builder emits volume, so this ONE-SHOT, additive, reversible backfill closes the gap.

Extends nothing (the existing materialize_backfill_xauusd_m1_* scripts repair the M1 TAPE,
not base34; the daemon's emit lives in PROTECTED core). It IMPORTS the daemon's one-truth
raw-M1 source (_load_m1_collector_for_window) so the bytes are identical to live — only the
full-history index-join is new here.

RUN ORDER (critical — Fase-2B): run this as the LAST step that touches base34 — i.e. AFTER
base34 is (re)built, BEFORE the V3 build. If a base34 rebuild runs AGAIN later, RE-RUN this
(the full-rebuild path does NOT emit volume, so a rebuild reverts base34 to volume-less).
Idempotent + re-runnable; a fresh content-addressed .bak is kept before each write.

Usage:
  python -m gx1.scripts.backfill_base34_raw_m1_ohlcv_v1 --dry-run
  python -m gx1.scripts.backfill_base34_raw_m1_ohlcv_v1 --write [--min-coverage 0.99]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.execution.v12_canonical_incremental import (
    BASE34_MANIFEST,
    _load_m1_collector_for_window,
)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _resolve_base34_path() -> Path:
    manifest = json.loads(BASE34_MANIFEST.read_text())
    return Path(manifest["parquet_path"])


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "time" in df.columns:
        out = df.set_index(pd.to_datetime(df["time"], utc=True))
        return out.sort_index()
    raise SystemExit("[BACKFILL_FAIL] frame has neither a DatetimeIndex nor a 'time' column")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill raw M1 OHLCV onto the M1-expanded base34")
    ap.add_argument("--base34", type=str, default=None,
                    help="explicit base34 parquet (default: resolve from BASE34 manifest)")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="fail-closed if < this fraction of base34 rows get ALL 5 OHLCV (no silent partial fill)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true", help="report coverage, write NOTHING")
    grp.add_argument("--write", action="store_true", help="write back (atomic; fresh content-addressed .bak)")
    args = ap.parse_args()

    base34_path = Path(args.base34) if args.base34 else _resolve_base34_path()
    if not base34_path.exists():
        raise SystemExit(f"[BACKFILL_FAIL] base34 not found: {base34_path}")
    print(f"[backfill] base34: {base34_path}", flush=True)

    b = _to_dt_index(pd.read_parquet(base34_path))
    n = len(b)
    if n == 0:
        raise SystemExit("[BACKFILL_FAIL] base34 is empty")
    # Index hygiene — the NaN-fill below is POSITIONAL (numpy), but a duplicate index would
    # still make the join ambiguous and signals a corrupt base34. Fail-closed (review blocker).
    if b.index.has_duplicates:
        ndup = int(b.index.duplicated().sum())
        raise SystemExit(f"[BACKFILL_FAIL] base34 index has {ndup} duplicate timestamps — refusing "
                         f"(corrupt base34; dedup it first).")
    if not b.index.is_monotonic_increasing:
        raise SystemExit("[BACKFILL_FAIL] base34 index not monotonic after sort — corrupt; aborting.")
    orig_cols = list(b.columns)
    print(f"[backfill] base34 rows={n:,}  span {b.index[0]} -> {b.index[-1]}  cols={len(orig_cols)}", flush=True)

    missing = [c for c in OHLCV_COLS if c not in b.columns]
    incomplete = [c for c in OHLCV_COLS if c in b.columns and bool(b[c].isna().any())]
    print(f"[backfill] OHLCV missing-cols={missing}  present-but-NaN-cols={incomplete}", flush=True)
    if not missing and not incomplete:
        print("[backfill] base34 ALREADY carries all 5 OHLCV with no NaN — nothing to do.", flush=True)
        return 0

    # One-truth raw M1 source (same loader the live daemon uses) over base34's full span.
    print("[backfill] loading raw M1 OHLCV from the daemon's one-truth source ...", flush=True)
    m1 = _load_m1_collector_for_window(b.index[0], b.index[-1])
    if m1 is None or m1.empty:
        raise SystemExit("[BACKFILL_FAIL] raw M1 source returned empty for the base34 span")
    m1 = _to_dt_index(m1)
    m1 = m1[~m1.index.duplicated(keep="last")]
    src_missing = [c for c in OHLCV_COLS if c not in m1.columns]
    if src_missing:
        raise SystemExit(f"[BACKFILL_FAIL] raw M1 source missing columns {src_missing}")
    print(f"[backfill] M1 source rows={len(m1):,}  span {m1.index[0]} -> {m1.index[-1]}", flush=True)

    # Exact timestamp-index alignment onto base34's M1 index (no resample, no ffill).
    aligned = m1[OHLCV_COLS].reindex(b.index)
    covered = aligned.notna().all(axis=1)
    coverage = float(covered.mean())
    print(f"[backfill] coverage: {int(covered.sum()):,}/{n:,} base34 rows get all 5 OHLCV "
          f"({coverage:.4%})", flush=True)
    if (~covered).any():
        gaps = b.index[~covered]
        print(f"[backfill] {len(gaps):,} uncovered rows (expected: market-closed gaps). "
              f"sample={[str(t) for t in gaps[:3]]}", flush=True)

    if coverage < args.min_coverage:
        raise SystemExit(
            f"[BACKFILL_FAIL] coverage {coverage:.4%} < --min-coverage {args.min_coverage:.4%}. "
            f"Refusing to write a partially-filled volume column (silent train/serve skew). "
            f"Check the M1 source span vs base34, or lower --min-coverage deliberately."
        )

    if args.dry_run:
        print("[backfill] DRY-RUN ok — nothing written. Re-run with --write to apply.", flush=True)
        return 0

    # --write: additive fill via POSITIONAL numpy (no .loc label-alignment fragility).
    # New column -> set fully. Existing column -> fill ONLY its NaN cells (preserve daemon
    # values). Uncovered (gap) cells stay NaN -> compute_volume_features is NaN-safe (0.0).
    for c in OHLCV_COLS:
        src = aligned[c].to_numpy(dtype=np.float64)
        if c not in b.columns:
            b[c] = src
        else:
            cur = b[c].to_numpy(dtype=np.float64, copy=True)
            nan_mask = np.isnan(cur)
            cur[nan_mask] = src[nan_mask]
            b[c] = cur

    # Integrity: we ONLY added OHLCV cols, dropped/renamed nothing (review: rebuild-impact).
    added = [c for c in b.columns if c not in orig_cols]
    dropped = [c for c in orig_cols if c not in b.columns]
    if dropped or any(c not in OHLCV_COLS for c in added):
        raise SystemExit(f"[BACKFILL_FAIL] schema changed unexpectedly: added={added} dropped={dropped}")
    print(f"[backfill] schema: +{added} (additive, by name); {len(orig_cols)}->{len(b.columns)} cols", flush=True)

    # Fresh content-addressed backup BEFORE each write (review: a stale .bak from a prior run
    # could lose daemon-appended rows on restore). Keyed by the current file mtime so each
    # distinct pre-state gets its own recovery point.
    mt = int(base34_path.stat().st_mtime)
    bak = base34_path.with_suffix(f".parquet.prebackfill_{mt}.bak")
    if not bak.exists():
        shutil.copy2(base34_path, bak)
        print(f"[backfill] backup -> {bak}", flush=True)

    tmp = base34_path.with_suffix(".parquet.tmp")
    b.to_parquet(tmp, index=True)
    os.replace(tmp, base34_path)
    sha = hashlib.sha256(base34_path.read_bytes()).hexdigest()
    print(f"[backfill] WROTE {base34_path}  rows={len(b):,}  cols={len(b.columns)}  sha256={sha[:16]}...",
          flush=True)

    # Refresh the BASE34 manifest sha/rows ATOMICALLY (review: was a non-atomic best-effort write).
    try:
        man = json.loads(BASE34_MANIFEST.read_text())
    except Exception as exc:  # only the READ is best-effort; the parquet is the source of truth
        print(f"[backfill] WARN manifest read failed, skipping refresh: {exc}", flush=True)
        man = None
    if man is not None and Path(man.get("parquet_path", "")) == base34_path:
        man["parquet_sha256"] = sha
        man["rows"] = int(len(b))
        tag = "+raw M1 OHLCV backfill (R12/V4)"
        if tag not in man.get("note", ""):
            man["note"] = (man.get("note", "") + " | " + tag).strip(" |")
        mtmp = BASE34_MANIFEST.with_suffix(".json.tmp")
        mtmp.write_text(json.dumps(man, indent=2))
        os.replace(mtmp, BASE34_MANIFEST)  # atomic
        print(f"[backfill] refreshed manifest {BASE34_MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
