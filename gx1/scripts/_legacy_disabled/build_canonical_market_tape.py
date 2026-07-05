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
from gx1.utils.granularity import granularity_to_minutes, granularity_to_pandas_freq


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


def _validate_time(df: pd.DataFrame, path: Path, granularity: str) -> None:
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
    step_minutes = granularity_to_minutes(granularity)
    # Check spacing where present (allow gaps for weekends/holidays)
    bad_step = diffs[(diffs.dt.total_seconds() % (60 * step_minutes)) != 0]
    if not bad_step.empty:
        raise RuntimeError(f"[{path}] NON_5M_SPACING_FOUND: examples={bad_step.head().tolist()}")


def _validate_required(df: pd.DataFrame, required_cols: List[str], path: Path) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{path}] MISSING_REQUIRED_COLS: {missing}")
    nan_cols = [c for c in required_cols if df[c].isnull().any()]
    if nan_cols:
        raise RuntimeError(f"[{path}] NAN_IN_REQUIRED_COLS: {nan_cols}")


def _load_year(path: Path, granularity: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"INPUT_NOT_FOUND: {path}")
    df = pd.read_parquet(path)
    # Drop stray index artifact if present
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    df = _ensure_time_column(df, path)
    _validate_time(df, path, granularity)
    return df


def _downsample_to_freq(df: pd.DataFrame, source_granularity: str, out_granularity: str) -> pd.DataFrame:
    """
    Downsample by taking every Nth source row when out_granularity is an integer multiple
    of source_granularity. This preserves the bar sequence semantics the runtime already uses.
    Falls back to a regular time-based resample only when the ratio is not an integer.
    """
    src_min = granularity_to_minutes(source_granularity)
    out_min = granularity_to_minutes(out_granularity)
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")

    if out_min > src_min and out_min % src_min == 0:
        step = out_min // src_min
        out = out.iloc[::step].copy()
        return out.reset_index(drop=True)

    # Conservative fallback: true OHLC resample
    freq = granularity_to_pandas_freq(out_granularity)
    out = out.set_index("time")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "bid_open": "first",
        "bid_high": "max",
        "bid_low": "min",
        "bid_close": "last",
        "ask_open": "first",
        "ask_high": "max",
        "ask_low": "min",
        "ask_close": "last",
    }
    if "volume" in out.columns:
        agg["volume"] = "sum"
    extra_cols = [c for c in out.columns if c not in agg]
    for c in extra_cols:
        agg[c] = "last"

    resampled = out.resample(freq, closed="left", label="left").agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled.reset_index()
    return resampled


def _write_year_partitions(df: pd.DataFrame, out_root: Path, required_cols: List[str], optional_cols: List[str]) -> Dict[int, int]:
    row_counts: Dict[int, int] = {}
    df = df.sort_values("time").copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    for year, dfi in df.groupby(df["time"].dt.year):
        dfi = dfi.loc[:, required_cols + optional_cols].copy()
        out_dir = out_root / f"year={int(year)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-000.parquet"
        dfi.to_parquet(out_path, index=False)
        row_counts[int(year)] = len(dfi)
    return row_counts


def build_tape(
    years: List[int],
    source_root: Path,
    out_root: Path,
    source_granularity: str = "M5",
    out_granularity: str = "M5",
    source_file: Path | None = None,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    per_year: Dict[int, pd.DataFrame] = {}
    row_counts: Dict[int, int] = {}
    all_columns_per_year: Dict[int, List[str]] = {}

    if source_file is not None:
        src = Path(source_file)
        df = _load_year(src, source_granularity)
        if source_granularity != out_granularity:
            df = _downsample_to_freq(df, source_granularity, out_granularity)
        df = _ensure_time_column(df, src)
        all_columns_per_year[int(df["time"].dt.year.min())] = list(df.columns)
        row_counts = _write_year_partitions(
            df,
            out_root,
            required_cols=REQUIRED_BASE + (["volume"] if "volume" in df.columns else []),
            optional_cols=[c for c in df.columns if c not in REQUIRED_BASE and c != "volume"],
        )
        manifest = {
            "instrument": "xauusd",
            "timeframe": out_granularity.lower(),
            "years": sorted(row_counts.keys()),
            "schema_required_cols": REQUIRED_BASE + (["volume"] if "volume" in df.columns else []),
            "schema_optional_cols": [c for c in df.columns if c not in REQUIRED_BASE and c != "volume"],
            "row_counts": row_counts,
            "source_file": str(src),
            "source_granularity": source_granularity,
            "out_granularity": out_granularity,
            "out_root": str(out_root),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = out_root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # Load + validate each year
    for y in years:
        src = source_root / str(y) / f"xauusd_{source_granularity.lower()}_{y}_bid_ask.parquet"
        df = _load_year(src, source_granularity)
        if source_granularity != out_granularity:
            df = _downsample_to_freq(df, source_granularity, out_granularity)
            df = _ensure_time_column(df, src)
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
        "timeframe": out_granularity.lower(),
        "years": years,
        "schema_required_cols": required_cols,
        "schema_optional_cols": optional_cols,
        "row_counts": row_counts,
        "source_root": str(source_root),
        "source_granularity": source_granularity,
        "out_granularity": out_granularity,
        "out_root": str(out_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical XAUUSD bid/ask tape (per-year partitions).")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/oanda/years"),
        help="Root containing per-year xauusd_{GRANULARITY}_{YEAR}_bid_ask.parquet",
    )
    parser.add_argument("--source-file", type=Path, default=None, help="Optional single parquet source file to partition")
    parser.add_argument("--source-granularity", default="M5", help="Granularity of source parquet(s)")
    parser.add_argument("--out-granularity", default="M5", help="Granularity to write in canonical output")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"),
        help="Canonical output directory",
    )
    args = parser.parse_args()

    manifest = build_tape(
        args.years,
        args.source_root,
        args.out_dir,
        source_granularity=args.source_granularity,
        out_granularity=args.out_granularity,
        source_file=args.source_file,
    )
    print("Wrote MANIFEST:", args.out_dir / "MANIFEST.json")
    years_to_print = sorted(int(y) for y in manifest["row_counts"].keys())
    for y in years_to_print:
        print(f"  year={y} rows={manifest['row_counts'][y]} -> {args.out_dir}/year={y}/part-000.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
