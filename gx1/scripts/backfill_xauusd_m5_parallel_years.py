#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel OANDA XAUUSD M5 Bid/Ask Backfill by Year (2020-2024)

Runs 4 parallel backfill processes, one per year (2020, 2021, 2022, 2023, 2024).
Each process fetches its own year independently, then all are merged.

Usage:
    python3 gx1/scripts/backfill_xauusd_m5_parallel_years.py \
        --years 2020 2021 2022 2023 2024 \
        --output-dir data/oanda/years

Dependencies (explicit install line):
  pip install pandas pyarrow requests
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def backfill_year(year: int, output_dir: Path, checkpoint_dir: Path) -> dict:
    """
    Backfill a single year.
    
    Args:
        year: Year to backfill (e.g., 2020)
        output_dir: Directory for output files
        checkpoint_dir: Directory for checkpoint files
    
    Returns:
        dict with success, year, output_path, error
    """
    log.info(f"[YEAR {year}] Starting backfill...")
    
    # Calculate year boundaries
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z"
    output_path = output_dir / f"{year}.parquet"
    checkpoint_path = checkpoint_dir / f"{year}_checkpoint.json"
    
    # Run backfill script (use absolute path)
    script_path = Path(__file__).parent.parent.parent / "gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--start", start,
        "--end", end,
        "--out", str(output_path),
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent.parent.parent),  # Run from workspace root
        )
        
        log.info(f"[YEAR {year}] ✅ Backfill complete")
        
        # Load manifest to get row count and SHA256
        manifest_path = output_path.parent / f"MANIFEST_{output_path.stem}.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            row_count = manifest.get("row_count", 0)
            sha256 = manifest.get("sha256", "")
        else:
            row_count = 0
            sha256 = ""
        
        return {
            "success": True,
            "year": year,
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "row_count": row_count,
            "sha256": sha256,
            "error": None,
        }
    except subprocess.CalledProcessError as e:
        # Get full error message
        error_output = e.stderr if e.stderr else e.stdout if e.stdout else str(e)
        error_msg = f"Backfill failed (exit code {e.returncode}): {error_output[-1000:]}"
        log.error(f"[YEAR {year}] ❌ {error_msg}")
        return {
            "success": False,
            "year": year,
            "output_path": str(output_path),
            "manifest_path": "",
            "row_count": 0,
            "sha256": "",
            "error": error_msg,
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log.error(f"[YEAR {year}] ❌ {error_msg}")
        return {
            "success": False,
            "year": year,
            "output_path": str(output_path),
            "manifest_path": "",
            "row_count": 0,
            "sha256": "",
            "error": error_msg,
        }


def merge_year_files(year_files: List[Path], output_path: Path) -> dict:
    """
    Merge year files into single dataset.
    
    Args:
        year_files: List of parquet file paths (one per year)
        output_path: Output path for merged dataset
    
    Returns:
        dict with row_count, sha256, time_range
    """
    import hashlib
    import pandas as pd
    from datetime import datetime, timezone
    
    log.info("=" * 60)
    log.info("Merging year files...")
    log.info("=" * 60)
    
    all_dfs = []
    for year_file in sorted(year_files):
        if not year_file.exists():
            log.warning(f"  Skipping missing file: {year_file}")
            continue
        log.info(f"  Loading {year_file.name}...")
        df = pd.read_parquet(year_file)
        log.info(f"    {len(df):,} rows, range: {df.index.min()} to {df.index.max()}")
        all_dfs.append(df)
    
    if not all_dfs:
        raise RuntimeError("No year files to merge")
    
    log.info("  Concatenating...")
    merged_df = pd.concat(all_dfs, axis=0)
    merged_df = merged_df.sort_index()
    merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
    
    log.info(f"  Writing {len(merged_df):,} rows to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=True)
    
    # Generate manifest
    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    sha256 = sha256_hash.hexdigest()
    
    manifest_path = output_path.parent / f"MANIFEST_{output_path.stem}.json"
    manifest = {
        "instrument": "XAU_USD",
        "granularity": "M5",
        "prices": "MBA",
        "time_range_start": merged_df.index.min().isoformat(),
        "time_range_end": merged_df.index.max().isoformat(),
        "row_count": len(merged_df),
        "sha256": sha256,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_years": sorted([int(f.stem) for f in year_files if f.exists()]),
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    log.info(f"  ✅ Manifest saved: {manifest_path}")
    
    return {
        "row_count": len(merged_df),
        "sha256": sha256,
        "time_range_start": merged_df.index.min().isoformat(),
        "time_range_end": merged_df.index.max().isoformat(),
        "manifest_path": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel OANDA XAUUSD M5 backfill by year (2020-2024)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="Years to backfill (default: 2020 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/oanda/years",
        help="Output directory for year files (default: data/oanda/years)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/oanda/checkpoints",
        help="Checkpoint directory (default: data/oanda/checkpoints)",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default="data/oanda/XAUUSD_M5_2020_2024_bidask.parquet",
        help="Output path for merged dataset (default: data/oanda/XAUUSD_M5_2020_2024_bidask.parquet)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging year files (keep separate)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("Parallel OANDA XAUUSD M5 Backfill by Year")
    log.info("=" * 60)
    log.info(f"Years: {args.years}")
    log.info(f"Max workers: {args.max_workers}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info("")
    
    # Run parallel backfills
    start_time = time.time()
    
    with multiprocessing.Pool(processes=min(args.max_workers, len(args.years))) as pool:
        results = pool.starmap(
            backfill_year,
            [(year, output_dir, checkpoint_dir) for year in args.years],
        )
    
    elapsed_time = time.time() - start_time
    
    # Report results
    log.info("")
    log.info("=" * 60)
    log.info("Backfill Results")
    log.info("=" * 60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    log.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    log.info(f"Successful: {len(successful)}/{len(results)}")
    log.info(f"Failed: {len(failed)}/{len(results)}")
    log.info("")
    
    if successful:
        log.info("Successful years:")
        total_rows = 0
        for r in sorted(successful, key=lambda x: x["year"]):
            sha256_display = r['sha256'][:16] + "..." if r['sha256'] else 'N/A'
            log.info(
                f"  {r['year']}: {r['row_count']:,} rows, "
                f"SHA256={sha256_display}"
            )
            total_rows += r["row_count"]
        log.info(f"  Total: {total_rows:,} rows")
        log.info("")
    
    if failed:
        log.error("Failed years:")
        for r in sorted(failed, key=lambda x: x["year"]):
            log.error(f"  {r['year']}: {r['error']}")
        log.error("")
        return 1
    
    # Merge if requested
    if not args.no_merge:
        log.info("")
        year_files = [output_dir / f"{r['year']}.parquet" for r in successful]
        merge_output_path = Path(args.merge_output)
        
        try:
            merge_result = merge_year_files(year_files, merge_output_path)
            log.info("")
            log.info("=" * 60)
            log.info("✅ MERGE COMPLETE")
            log.info("=" * 60)
            log.info(f"Merged dataset: {merge_output_path}")
            log.info(f"Total rows: {merge_result['row_count']:,}")
            log.info(f"SHA256: {merge_result['sha256'][:16]}...")
            log.info(f"Time range: {merge_result['time_range_start'][:10]} to {merge_result['time_range_end'][:10]}")
            log.info(f"Manifest: {merge_result['manifest_path']}")
        except Exception as e:
            log.error(f"Merge failed: {e}", exc_info=True)
            return 1
    else:
        log.info("Skipping merge (--no-merge specified)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
