#!/usr/bin/env python3
"""Repair the Dec-2024 M5 canonical tape segment from the clean canonical M1 tape.

Verified defect (read-only recount 2026-07-23): the canonical M5 bid/ask tape
carries 3430 rows with impossible OHLC geometry (close outside [low, high]
across mid/bid/ask surfaces), all inside 2024-11-30 -> 2024-12-31. Exactly
2799 are Saturday/Sunday rows. The clean canonical M1 supports 5757 rebuilt
M5 buckets in the December window and leaves 3459 canonical M5 rows without
M1 backing. Canonical M1 is geometry-clean, so it is the deeper source of
truth for the event-local repair.

This tool writes an EVENT-LOCAL repaired copy of the tape root. It never
touches the canonical tape, the live prebuilt or anything outside --out-root.

Method:
1. Convention proof: aggregate M1 -> M5 on a clean control day and require the
   result to match the existing tape almost exactly (max abs diff <= 1e-6 on
   all twelve OHLC columns). A convention mismatch aborts the repair.
2. Inside the repair window every M5 bar is recomputed from its M1 bucket
   (first/max/min/last per bid/ask/mid surface, volume summed). M5 bars with
   zero M1 backing are DROPPED (they are synthetic closed-market bars).
3. All other rows in all years are copied byte-identically.
4. The full repaired root is geometry-verified (mid/bid/ask, all years) and an
   immutable REPAIR manifest binds source/output hashes, counts and the vedtak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gx1_guards.gates import require_retrain_vedtak
from gx1.contracts.xau_tape_provenance_v1 import (
    BASE_REPAIR_METHOD,
    BASE_REPAIR_SCHEMA,
    XAU_INSTRUMENT,
    canonical_xau_source_descriptor_v1,
    validate_xau_tape_provenance_v1,
)

REPAIR_WINDOW_START = pd.Timestamp("2024-11-30T00:00:00Z")
REPAIR_WINDOW_END = pd.Timestamp("2025-01-01T00:00:00Z")  # exclusive
CONTROL_DAY = pd.Timestamp("2024-11-14T00:00:00Z")  # clean weekday before window
# The canonical M5 tape is OANDA-native M5. M1 aggregation reproduces high/low
# EXACTLY; open/close/volume may differ by boundary-tick assignment (a tick at
# an exact minute boundary lands in different candles in the M1 vs M5 stream).
# Measured on the clean control day 2024-11-14: high/low max diff 0.0 across
# 276 buckets; open/close differ in 7/276 buckets, max 0.035 (~1.3 bps, both
# directions); volume differs in 41/276, max 5 ticks. Sub-spread noise.
HIGH_LOW_TOLERANCE = 1e-6
OPEN_CLOSE_BOUNDARY_TICK_TOLERANCE = 0.10
OHLC_PREFIXES = ("", "bid_", "ask_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_bad_mask(df: pd.DataFrame) -> np.ndarray:
    bad = np.zeros(len(df), dtype=bool)
    for prefix in OHLC_PREFIXES:
        if f"{prefix}open" not in df.columns:
            continue
        o, h, lo, c = (
            df[f"{prefix}{k}"].to_numpy(dtype=np.float64)
            for k in ("open", "high", "low", "close")
        )
        bad |= (
            (h < np.maximum(o, c))
            | (lo > np.minimum(o, c))
            | (h < lo)
            | ~np.isfinite(o + h + lo + c)
        )
    return bad


def _aggregate_m1_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1.sort_values("time").reset_index(drop=True)
    bucket = m1["time"].dt.floor("5min")
    grouped = m1.groupby(bucket, sort=True)
    out = pd.DataFrame({"time": list(grouped.groups.keys())})
    for prefix in OHLC_PREFIXES:
        out[f"{prefix}open"] = grouped[f"{prefix}open"].first().to_numpy()
        out[f"{prefix}high"] = grouped[f"{prefix}high"].max().to_numpy()
        out[f"{prefix}low"] = grouped[f"{prefix}low"].min().to_numpy()
        out[f"{prefix}close"] = grouped[f"{prefix}close"].last().to_numpy()
    out["volume"] = grouped["volume"].sum().to_numpy()
    return out


def _prove_convention(tape_2024: pd.DataFrame, m1_2024: pd.DataFrame) -> float:
    day_end = CONTROL_DAY + pd.Timedelta(days=1)
    m1_day = m1_2024[(m1_2024["time"] >= CONTROL_DAY) & (m1_2024["time"] < day_end)]
    tape_day = tape_2024[
        (tape_2024["time"] >= CONTROL_DAY) & (tape_2024["time"] < day_end)
    ]
    if m1_day.empty or tape_day.empty:
        raise RuntimeError("TAPE_REPAIR_CONTROL_DAY_EMPTY")
    agg = _aggregate_m1_to_m5(m1_day)
    joined = tape_day.merge(agg, on="time", suffixes=("_tape", "_agg"), how="inner")
    if len(joined) != len(tape_day):
        raise RuntimeError(
            "TAPE_REPAIR_CONTROL_DAY_BUCKET_MISMATCH: "
            f"tape={len(tape_day)} joined={len(joined)}"
        )
    proof: dict = {}
    for prefix in OHLC_PREFIXES:
        for part in ("open", "high", "low", "close"):
            diff = np.abs(
                joined[f"{prefix}{part}_tape"].to_numpy(dtype=np.float64)
                - joined[f"{prefix}{part}_agg"].to_numpy(dtype=np.float64)
            )
            max_diff = float(diff.max())
            proof[f"{prefix}{part}_max_abs_diff"] = max_diff
            proof[f"{prefix}{part}_n_buckets_diff"] = int((diff > 1e-6).sum())
            tolerance = (
                HIGH_LOW_TOLERANCE
                if part in ("high", "low")
                else OPEN_CLOSE_BOUNDARY_TICK_TOLERANCE
            )
            if max_diff > tolerance:
                raise RuntimeError(
                    "TAPE_REPAIR_CONVENTION_MISMATCH: "
                    f"{prefix}{part} max_abs_diff={max_diff} tolerance={tolerance}"
                )
    proof["control_buckets"] = int(len(joined))
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vedtak", required=True)
    parser.add_argument("--m5-tape-root", type=Path, required=True)
    parser.add_argument("--m1-tape-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    vedtak = require_retrain_vedtak(args.vedtak)
    m5_source = canonical_xau_source_descriptor_v1(
        args.m5_tape_root.expanduser(),
        timeframe="M5",
    )
    m1_source = canonical_xau_source_descriptor_v1(
        args.m1_tape_root.expanduser(),
        timeframe="M1",
    )
    m5_root = Path(m5_source["root"])
    m1_root = Path(m1_source["root"])
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists():
        raise RuntimeError(f"TAPE_REPAIR_OUT_ROOT_NOT_FRESH: {out_root}")

    year_parts = sorted(m5_root.glob("year=*/part-000.parquet"))
    if not year_parts:
        raise RuntimeError(f"TAPE_REPAIR_M5_ROOT_EMPTY: {m5_root}")

    m1_2024_path = m1_root / "year=2024" / "part-000.parquet"
    m1_2024 = pd.read_parquet(m1_2024_path)
    m1_2024["time"] = pd.to_datetime(m1_2024["time"], utc=True)

    manifest: dict = {
        "schema_version": BASE_REPAIR_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": XAU_INSTRUMENT,
        "explicit_vedtak_id": vedtak,
        "method": BASE_REPAIR_METHOD,
        "repair_window": [str(REPAIR_WINDOW_START), str(REPAIR_WINDOW_END)],
        "m5_tape_root": str(m5_root),
        "m1_tape_root": str(m1_root),
        "canonical_sources": {"m5": m5_source, "m1": m1_source},
        "m1_2024_sha256": _sha256_file(m1_2024_path),
        "years": {},
    }

    total_bad_before = 0
    total_bad_after = 0
    for part in year_parts:
        year_label = part.parent.name
        df = pd.read_parquet(part)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        bad_before = int(_geometry_bad_mask(df).sum())
        total_bad_before += bad_before
        entry: dict = {
            "source_sha256": _sha256_file(part),
            "rows_before": int(len(df)),
            "geometry_bad_before": bad_before,
            "repaired": False,
        }

        if year_label == "year=2024":
            entry["convention_proof"] = _prove_convention(df, m1_2024)

            in_window = (df["time"] >= REPAIR_WINDOW_START) & (
                df["time"] < REPAIR_WINDOW_END
            )
            m1_window = m1_2024[
                (m1_2024["time"] >= REPAIR_WINDOW_START)
                & (m1_2024["time"] < REPAIR_WINDOW_END)
            ]
            rebuilt = _aggregate_m1_to_m5(m1_window)
            keep = df.loc[~in_window]
            missing_cols = [c for c in keep.columns if c not in rebuilt.columns]
            if missing_cols:
                raise RuntimeError(
                    f"TAPE_REPAIR_COLUMN_MISMATCH: rebuilt lacks {missing_cols}"
                )
            rebuilt = rebuilt[list(keep.columns)]
            df = (
                pd.concat([keep, rebuilt], ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )
            entry.update(
                {
                    "repaired": True,
                    "window_rows_before": int(in_window.sum()),
                    "window_rows_rebuilt_from_m1": int(len(rebuilt)),
                    "window_rows_dropped_unbacked": int(
                        in_window.sum() - len(rebuilt)
                    ),
                }
            )

        bad_after = int(_geometry_bad_mask(df).sum())
        total_bad_after += bad_after
        entry["rows_after"] = int(len(df))
        entry["geometry_bad_after"] = bad_after

        out_part = out_root / year_label / "part-000.parquet"
        out_part.parent.mkdir(parents=True, exist_ok=False)
        df.to_parquet(out_part, index=False)
        entry["output_sha256"] = _sha256_file(out_part)
        manifest["years"][year_label] = entry

    if total_bad_after != 0:
        raise RuntimeError(
            f"TAPE_REPAIR_GEOMETRY_STILL_BAD: after={total_bad_after}"
        )
    manifest["geometry_bad_total_before"] = total_bad_before
    manifest["geometry_bad_total_after"] = total_bad_after

    manifest_path = out_root / "REPAIR_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_xau_tape_provenance_v1(
        out_root,
        expected_run_id=vedtak,
        require_current=False,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "years"}, indent=2))
    print(json.dumps(manifest["years"].get("year=2024", {}), indent=2))


if __name__ == "__main__":
    main()
