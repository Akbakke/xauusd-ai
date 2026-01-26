#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Parquet Dataset by Year

Mechanically splits a parquet file by year based on timestamp index.
No resampling, no filtering, no "smartness" - pure timestamp-based split.

Usage:
    python3 gx1/scripts/split_parquet_by_year.py \
        --input data/oanda/XAUUSD_M5_2020_2024_bidask.parquet \
        --output-dir data/oanda/years

Dependencies (explicit install line):
  pip install pandas pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def generate_manifest(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate manifest JSON file for dataset.
    
    Args:
        df: DataFrame with candles
        output_path: Path to parquet file
    
    Returns:
        Path to manifest file
    """
    manifest_path = output_path.parent / f"MANIFEST_{output_path.stem}.json"
    
    # Compute SHA256
    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    sha256 = sha256_hash.hexdigest()
    
    manifest = {
        "instrument": "XAU_USD",
        "granularity": "M5",
        "prices": "MBA",
        "time_range_start": df.index.min().isoformat(),
        "time_range_end": df.index.max().isoformat(),
        "row_count": len(df),
        "sha256": sha256,
        "schema": {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index_type": "DatetimeIndex",
            "index_tz": str(df.index.tz) if df.index.tz else None,
        },
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def split_by_year(input_path: Path, output_dir: Path) -> dict:
    """
    Split parquet file by year.
    
    Args:
        input_path: Path to input parquet file
        output_dir: Directory for output files
    
    Returns:
        dict with year -> (output_path, row_count, sha256)
    """
    log.info("=" * 60)
    log.info("Split Parquet by Year")
    log.info("=" * 60)
    log.info(f"Input: {input_path}")
    log.info(f"Output directory: {output_dir}")
    log.info("")
    
    # Load input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    log.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    log.info(f"Loaded {len(df):,} rows")
    log.info(f"Time range: {df.index.min()} to {df.index.max()}")
    log.info("")
    
    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex, got {type(df.index)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by year
    results = {}
    years = sorted(df.index.year.unique())
    
    log.info(f"Splitting into {len(years)} years: {years}")
    log.info("")
    
    for year in years:
        log.info(f"Processing year {year}...")
        
        # Filter by year (mechanical, no resampling)
        year_df = df[df.index.year == year].copy()
        
        if year_df.empty:
            log.warning(f"  No data for year {year}, skipping")
            continue
        
        # Sort by index (should already be sorted, but ensure)
        year_df = year_df.sort_index()
        
        # Output path
        output_path = output_dir / f"{year}.parquet"
        
        # Save
        log.info(f"  Writing {len(year_df):,} rows to {output_path}...")
        year_df.to_parquet(output_path, index=True)
        
        # Generate manifest
        manifest_path = generate_manifest(year_df, output_path)
        
        # Load manifest to get SHA256
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        results[year] = {
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "row_count": len(year_df),
            "sha256": manifest["sha256"],
            "time_range_start": manifest["time_range_start"],
            "time_range_end": manifest["time_range_end"],
        }
        
        log.info(f"  ✅ Year {year}: {len(year_df):,} rows, SHA256={manifest['sha256'][:16]}...")
        log.info(f"     Range: {year_df.index.min()} to {year_df.index.max()}")
        log.info("")
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split parquet dataset by year (mechanical, no resampling)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/oanda/years",
        help="Output directory for year files (default: data/oanda/years)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    try:
        results = split_by_year(input_path, output_dir)
    except Exception as e:
        log.error(f"Split failed: {e}", exc_info=True)
        return 1
    
    # Summary
    log.info("=" * 60)
    log.info("✅ SPLIT COMPLETE")
    log.info("=" * 60)
    log.info(f"Total years: {len(results)}")
    log.info("")
    log.info("Year breakdown:")
    for year in sorted(results.keys()):
        info = results[year]
        log.info(
            f"  {year}: {info['row_count']:,} rows, "
            f"SHA256={info['sha256'][:16]}..., "
            f"Range: {info['time_range_start'][:10]} to {info['time_range_end'][:10]}"
        )
    log.info("")
    log.info(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
