#!/usr/bin/env python3
"""
Build canonical XAUUSD M5 bid/ask tape (per-year partitions) from existing OANDA years.

Input (must already exist):
  /home/andre2/GX1_DATA/data/oanda/years/{YEAR}/xauusd_m5_{YEAR}_bid_ask.parquet  for YEAR in 2020..2025

Output (written deterministically):
  /home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL/year=YYYY/part-000.parquet
  /home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL/MANIFEST.json

Rules:
- Required columns (all years): time (datetime64[ns, UTC]), open, high, low, close, bid_open/high/low/close, ask_open/high/low/close.
- Include volume only if present in all years; otherwise omit.
- Optional passthrough columns: intersection of remaining columns present in all years (stable order).
- Validation per year: UTC time, sorted, no duplicates, 5m spacing (gaps allowed), no NaN in required price cols.
- Global: no duplicate timestamps across years.

Usage:
  python gx1/scripts/build_canonical_market_tape.py \
    --years 2020 2021 2022 2023 2024 2025 \
    --source-root /home/andre2/GX1_DATA/data/oanda/years \
    --out-dir /home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_BASE = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]


def _ensure_time_column(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "time" in df.columns:
        ts = df["time"]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, name="time")
        df = df.reset_index(drop=True)
        df.insert(0, "time", ts)
    else:
        raise RuntimeError(f"[{path}] MISSING_TIME_COLUMN_AND_INDEX_NOT_DATETIME")

    if not pd.api.types.is_datetime64tz_dtype(ts.dtype):
        raise RuntimeError(f"[{path}] TIME_NOT_TZ_AWARE")
    if str(ts.dt.tz) != "UTC":
        raise RuntimeError(f"[{path}] TIME_NOT_UTC: {ts.dt.tz}")
    df["time"] = ts
    return df


def _validate_time(df: pd.DataFrame, path: Path) -> None:
    ts = df["time"]
    if ts.isnull().any():
        raise RuntimeError(f"[{path}] TIME_CONTAINS_NAN")
    if not ts.is_monotonic_increasing:
        raise RuntimeError(f"[{path}] TIME_NOT_SORTED")
    if ts.duplicated().any():
        raise RuntimeError(f"[{path}] DUPLICATE_TIMESTAMPS")
    diffs = ts.diff().dropna()
    if (diffs.dt.total_seconds() < 0).any():
        raise RuntimeError(f"[{path}] NEGATIVE_TIME_STEP")
    # Check 5m spacing where present (allow gaps for weekends/holidays)
    bad_step = diffs[(diffs.dt.total_seconds() % 300) != 0]
    if not bad_step.empty:
        raise RuntimeError(f"[{path}] NON_5M_SPACING_FOUND: examples={bad_step.head().tolist()}")


def _validate_required(df: pd.DataFrame, required_cols: List[str], path: Path) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{path}] MISSING_REQUIRED_COLS: {missing}")
    nan_cols = [c for c in required_cols if df[c].isnull().any()]
    if nan_cols:
        raise RuntimeError(f"[{path}] NAN_IN_REQUIRED_COLS: {nan_cols}")


def _load_year(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"INPUT_NOT_FOUND: {path}")
    df = pd.read_parquet(path)
    # Drop stray index artifact if present
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    df = _ensure_time_column(df, path)
    _validate_time(df, path)
    return df


def build_tape(years: List[int], source_root: Path, out_root: Path) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    per_year: Dict[int, pd.DataFrame] = {}
    row_counts: Dict[int, int] = {}
    all_columns_per_year: Dict[int, List[str]] = {}

    # Load + validate each year
    for y in years:
        src = source_root / str(y) / f"xauusd_m5_{y}_bid_ask.parquet"
        df = _load_year(src)
        per_year[y] = df
        row_counts[y] = len(df)
        all_columns_per_year[y] = list(df.columns)

    # Determine required/optional columns
    required_cols = list(REQUIRED_BASE)
    volume_present_all = all("volume" in cols for cols in all_columns_per_year.values())
    if volume_present_all:
        required_cols.append("volume")

    # Optional = intersection of remaining columns across all years (exclude required)
    common_cols = set(all_columns_per_year[years[0]])
    for cols in all_columns_per_year.values():
        common_cols &= set(cols)
    optional_cols = [c for c in all_columns_per_year[years[0]] if c in common_cols and c not in required_cols]

    # Validate required columns across years and write
    seen_times: set[pd.Timestamp] = set()
    for y in years:
        df = per_year[y]
        _validate_required(df, required_cols, source_root / str(y))

        # Check global duplicate timestamps
        times = df["time"].to_list()
        overlap = [t for t in times if t in seen_times]
        if overlap:
            raise RuntimeError(f"[GLOBAL] DUPLICATE_TIMESTAMPS_BETWEEN_YEARS: year={y} samples={overlap[:5]}")
        seen_times.update(times)

        # Build output frame
        cols_out = required_cols + optional_cols
        df_out = df.loc[:, cols_out].copy()
        df_out = df_out.sort_values("time")

        out_dir = out_root / f"year={y}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-000.parquet"
        df_out.to_parquet(out_path, index=False)

    manifest = {
        "instrument": "xauusd",
        "timeframe": "m5",
        "years": years,
        "schema_required_cols": required_cols,
        "schema_optional_cols": optional_cols,
        "row_counts": row_counts,
        "source_root": str(source_root),
        "out_root": str(out_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical XAUUSD M5 bid/ask tape (per-year partitions).")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/oanda/years"),
        help="Root containing per-year xauusd_m5_{YEAR}_bid_ask.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"),
        help="Canonical output directory",
    )
    args = parser.parse_args()

    manifest = build_tape(args.years, args.source_root, args.out_dir)
    print("Wrote MANIFEST:", args.out_dir / "MANIFEST.json")
    for y in args.years:
        print(f"  year={y} rows={manifest['row_counts'][y]} -> {args.out_dir}/year={y}/part-000.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
