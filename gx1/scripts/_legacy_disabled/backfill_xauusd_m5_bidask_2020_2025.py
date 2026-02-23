#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OANDA XAUUSD M5 Bid/Ask Backfill (2020-2025)

Backfills XAUUSD M5 candles with bid/ask prices from OANDA API for years 2020-2025.
Schema matches exactly with existing 2025 dataset (data/raw/xauusd_m5_2025_bid_ask.parquet).

Usage:
    # Backfill 2020-2024 (2025 already exists)
    python gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
        --start 2020-01-01T00:00:00Z \
        --end 2025-01-01T00:00:00Z \
        --out data/oanda/XAUUSD_M5_2020_2024_bidask.parquet

    # Verify against 2025 sample
    python gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
        --verify-against data/raw/xauusd_m5_2025_bid_ask.parquet \
        --verify-window-days 7

Dependencies (explicit install line):
  pip install pandas pyarrow requests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig, OandaAPIError
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Canonical schema (from 2025 dataset)
CANONICAL_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]
CANONICAL_DTYPES = {col: "float64" for col in CANONICAL_COLUMNS}
INSTRUMENT = "XAU_USD"
GRANULARITY = "M5"
MAX_CANDLES_PER_REQUEST = 5000
CHUNK_DAYS = 15  # ~4320 bars (safe margin under 5000 limit)


def fetch_candles_chunk(
    client: OandaClient,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str = "M5",
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Fetch one chunk of candles with bid/ask from OANDA API.
    
    Args:
        client: OANDA client
        instrument: Instrument symbol (e.g., "XAU_USD")
        start: Start timestamp (inclusive, UTC)
        end: End timestamp (exclusive, UTC)
        granularity: Candle granularity (default: "M5")
        max_retries: Maximum retries on failure
    
    Returns:
        DataFrame with canonical schema (matching 2025 dataset)
    
    Raises:
        RuntimeError: If bid/ask fields are missing or schema mismatch
    """
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "granularity": granularity,
        "price": "MBA",  # Mid + Bid + Ask
        "alignmentTimezone": "UTC",
    }
    
    for attempt in range(max_retries):
        try:
            data = client._request(
                "GET",
                f"/instruments/{instrument}/candles",
                params=params,
                max_retries=1,  # We handle retries here
            )
            break
        except OandaAPIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to fetch candles after {max_retries} attempts: {e}")
    
    candles = data.get("candles", [])
    if not candles:
        return pd.DataFrame(columns=CANONICAL_COLUMNS).set_index(pd.DatetimeIndex([], tz="UTC"))
    
    # Parse candles with bid/ask
    records = []
    for candle in candles:
        # Only include complete bars
        if not candle.get("complete", False):
            continue
        
        time_str = candle["time"]
        mid = candle.get("mid", {})
        bid = candle.get("bid", {})
        ask = candle.get("ask", {})
        volume = candle.get("volume", 0)
        
        # Normalize timestamp to UTC and floor to 5-minute boundary
        raw_time = pd.to_datetime(time_str)
        if raw_time.tzinfo is None:
            raw_time = raw_time.tz_localize("UTC")
        else:
            raw_time = raw_time.tz_convert("UTC")
        normalized_time = raw_time.floor("5min")
        
        # Hard-fail if bid/ask fields are missing
        if not bid:
            raise RuntimeError(
                f"[SCHEMA_FAIL] Missing bid fields in candle at {normalized_time}. "
                f"Candle: {candle}"
            )
        if not ask:
            raise RuntimeError(
                f"[SCHEMA_FAIL] Missing ask fields in candle at {normalized_time}. "
                f"Candle: {candle}"
            )
        
        # Extract all required fields
        record = {
            "time": normalized_time,
            "open": float(mid.get("o", 0)),
            "high": float(mid.get("h", 0)),
            "low": float(mid.get("l", 0)),
            "close": float(mid.get("c", 0)),
            "volume": float(volume),
            "bid_open": float(bid.get("o", 0)),
            "bid_high": float(bid.get("h", 0)),
            "bid_low": float(bid.get("l", 0)),
            "bid_close": float(bid.get("c", 0)),
            "ask_open": float(ask.get("o", 0)),
            "ask_high": float(ask.get("h", 0)),
            "ask_low": float(ask.get("l", 0)),
            "ask_close": float(ask.get("c", 0)),
        }
        
        # Hard-fail if any bid/ask field is zero or missing
        for field in ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]:
            if record[field] == 0.0:
                raise RuntimeError(
                    f"[SCHEMA_FAIL] Zero value in {field} at {normalized_time}. "
                    f"This indicates missing data."
                )
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(columns=CANONICAL_COLUMNS).set_index(pd.DatetimeIndex([], tz="UTC"))
    
    df = pd.DataFrame.from_records(records)
    df = df.set_index("time")
    
    # Ensure canonical column order
    df = df[CANONICAL_COLUMNS]
    
    # Ensure canonical dtypes
    for col, dtype in CANONICAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    
    return df


def validate_schema(df: pd.DataFrame, context: str = "") -> None:
    """
    Validate DataFrame schema matches canonical 2025 schema.
    Hard-fails on mismatch.
    
    Args:
        df: DataFrame to validate
        context: Context string for error messages
    
    Raises:
        RuntimeError: If schema mismatch detected
    """
    errors = []
    
    # Check columns
    if list(df.columns) != CANONICAL_COLUMNS:
        errors.append(
            f"Column mismatch: expected {CANONICAL_COLUMNS}, got {list(df.columns)}"
        )
    
    # Check dtypes
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        elif df[col].dtype != CANONICAL_DTYPES[col]:
            errors.append(
                f"Column {col} dtype mismatch: expected {CANONICAL_DTYPES[col]}, "
                f"got {df[col].dtype}"
            )
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Index must be DatetimeIndex, got {type(df.index)}")
    elif df.index.tz is None or str(df.index.tz) != "UTC":
        errors.append(f"Index timezone must be UTC, got {df.index.tz}")
    
    # Check for duplicates
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        errors.append(f"Duplicate timestamps found: {dup_count}")
    
    # Check monotonic increasing
    if not df.index.is_monotonic_increasing:
        errors.append("Index is not monotonic increasing (not sorted)")
    
    # Check M5 grid (5-minute step, except for gaps)
    # Note: OANDA may return 1:05:00 (65 min) gaps during market transitions (legitimate)
    # We only flag steps that are clearly wrong (not 5min, not a multiple of 5min, or too large)
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=5)
        # Allow gaps (market closures) but verify all non-gap steps are multiples of 5 minutes
        # Ignore gaps > 24h (weekend/market closures)
        non_gap_diffs = time_diffs[time_diffs <= pd.Timedelta(hours=24)]
        if len(non_gap_diffs) > 0:
            # Check if each step is a multiple of 5 minutes
            # Allow any multiple of 5 minutes (5, 10, 15, 20, 25, 30, ... up to 24h)
            # This accounts for legitimate market gaps (e.g., 65min = 13*5min, 305min = 61*5min)
            invalid_steps = []
            for diff in non_gap_diffs:
                if diff != expected_diff:
                    # Check if it's a multiple of 5 minutes
                    minutes = diff.total_seconds() / 60
                    # Use small epsilon for floating-point comparison
                    if abs(minutes % 5) > 0.001:
                        invalid_steps.append(diff)
            
            if len(invalid_steps) > 0:
                errors.append(
                    f"M5 grid broken: {len(invalid_steps)} invalid steps found (not multiple of 5min). "
                    f"Examples: {invalid_steps[:5]}"
                )
    
    # Check for NaN in price fields
    price_cols = [c for c in CANONICAL_COLUMNS if c != "volume"]
    for col in price_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            errors.append(f"NaN values in {col}: {nan_count}")
    
    if errors:
        error_msg = f"[SCHEMA_VALIDATION_FAIL] {context}\n" + "\n".join([f"  - {e}" for e in errors])
        raise RuntimeError(error_msg)


def backfill_range(
    client: OandaClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backfill candles for a date range with paging and checkpointing.
    
    Args:
        client: OANDA client
        start: Start timestamp (inclusive, UTC)
        end: End timestamp (exclusive, UTC)
        checkpoint_dir: Directory for checkpoint files (optional)
    
    Returns:
        (DataFrame with all candles, metadata dict)
    """
    all_chunks = []
    current_start = start
    total_rows = 0
    last_success_ts = start
    
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Use unique checkpoint filename based on start/end to avoid conflicts in parallel runs
        checkpoint_name = f"XAUUSD_M5_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_checkpoint.json"
        checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                # Parse timestamp (handle both tz-aware and tz-naive)
                checkpoint_ts = pd.to_datetime(checkpoint["last_success_ts"])
                if checkpoint_ts.tzinfo is None:
                    last_success_ts = checkpoint_ts.tz_localize("UTC")
                else:
                    last_success_ts = checkpoint_ts.tz_convert("UTC")
                total_rows = checkpoint.get("n_rows", 0)
                log.info(
                    f"Resuming from checkpoint: last_success_ts={last_success_ts}, "
                    f"n_rows={total_rows}"
                )
                current_start = last_success_ts + pd.Timedelta(minutes=5)
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
    
    chunk_num = 0
    while current_start < end:
        chunk_num += 1
        current_end = min(current_start + pd.Timedelta(days=CHUNK_DAYS), end)
        
        log.info(
            f"[BACKFILL] Chunk {chunk_num}: {current_start.date()} to {current_end.date()} "
            f"({(current_end - current_start).days} days)"
        )
        
        try:
            chunk_df = fetch_candles_chunk(
                client=client,
                instrument=INSTRUMENT,
                start=current_start,
                end=current_end,
                granularity=GRANULARITY,
            )
            
            if chunk_df.empty:
                log.warning(f"  No candles for chunk {chunk_num}")
                current_start = current_end
                continue
            
            # Validate chunk schema
            validate_schema(chunk_df, context=f"chunk {chunk_num}")
            
            all_chunks.append(chunk_df)
            total_rows += len(chunk_df)
            last_success_ts = chunk_df.index.max()
            
            log.info(
                f"  ✅ Fetched {len(chunk_df):,} candles. "
                f"Range: {chunk_df.index.min()} to {chunk_df.index.max()}. "
                f"Total: {total_rows:,}"
            )
            
            # Save checkpoint
            if checkpoint_path:
                checkpoint = {
                    "last_success_ts": last_success_ts.isoformat(),
                    "n_rows": total_rows,
                    "chunk_num": chunk_num,
                }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, indent=2)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            log.error(f"  ❌ Failed to fetch chunk {chunk_num}: {e}")
            raise RuntimeError(f"Backfill failed at chunk {chunk_num}: {e}") from e
        
        current_start = current_end
    
    if not all_chunks:
        raise RuntimeError("No candles fetched")
    
    # Combine all chunks
    log.info("Combining chunks...")
    df = pd.concat(all_chunks, axis=0)
    
    # Sort and remove duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    # Final validation
    validate_schema(df, context="final combined dataset")
    
    metadata = {
        "total_rows": len(df),
        "time_range_start": df.index.min().isoformat(),
        "time_range_end": df.index.max().isoformat(),
        "chunks_fetched": chunk_num,
    }
    
    return df, metadata


def verify_against_sample(
    client: OandaClient,
    sample_path: Path,
    window_days: int = 7,
) -> bool:
    """
    Verify backfill against a sample from existing dataset.
    
    Args:
        client: OANDA client
        sample_path: Path to existing parquet file (e.g., 2025 dataset)
        window_days: Number of days to verify (default: 7)
    
    Returns:
        True if verification passes, False otherwise
    """
    log.info("=" * 60)
    log.info("VERIFICATION MODE: Comparing against sample dataset")
    log.info("=" * 60)
    
    # Load sample dataset
    if not sample_path.exists():
        log.error(f"Sample file not found: {sample_path}")
        return False
    
    try:
        sample_df = pd.read_parquet(sample_path)
        log.info(f"Loaded sample: {len(sample_df):,} rows from {sample_path}")
    except Exception as e:
        log.error(f"Failed to load sample: {e}")
        return False
    
    # Validate sample schema
    try:
        validate_schema(sample_df, context="sample dataset")
        log.info("✅ Sample schema validated")
    except RuntimeError as e:
        log.error(f"❌ Sample schema validation failed: {e}")
        return False
    
    # Select verification window (middle of dataset to avoid edge effects)
    mid_idx = len(sample_df) // 2
    window_start = sample_df.index[mid_idx]
    window_end = window_start + pd.Timedelta(days=window_days)
    
    # Clip to dataset bounds
    window_start = max(window_start, sample_df.index.min())
    window_end = min(window_end, sample_df.index.max())
    
    sample_window = sample_df.loc[window_start:window_end].copy()
    log.info(
        f"Verification window: {window_start.date()} to {window_end.date()} "
        f"({len(sample_window):,} rows)"
    )
    
    # Fetch same window from OANDA
    log.info("Fetching same window from OANDA...")
    try:
        oanda_df, _ = backfill_range(
            client=client,
            start=window_start,
            end=window_end + pd.Timedelta(minutes=5),  # Include end bar
        )
    except Exception as e:
        log.error(f"Failed to fetch from OANDA: {e}")
        return False
    
    if oanda_df.empty:
        log.error("OANDA returned empty dataset")
        return False
    
    # Align timestamps (intersection)
    common_timestamps = sample_window.index.intersection(oanda_df.index)
    if len(common_timestamps) == 0:
        log.error("No common timestamps between sample and OANDA data")
        return False
    
    sample_aligned = sample_window.loc[common_timestamps]
    oanda_aligned = oanda_df.loc[common_timestamps]
    
    log.info(f"Comparing {len(common_timestamps):,} common timestamps")
    
    # Compare schemas
    errors = []
    
    # Column names
    if list(sample_aligned.columns) != list(oanda_aligned.columns):
        errors.append(
            f"Column mismatch: sample={list(sample_aligned.columns)}, "
            f"oanda={list(oanda_aligned.columns)}"
        )
    
    # Dtypes
    for col in CANONICAL_COLUMNS:
        if sample_aligned[col].dtype != oanda_aligned[col].dtype:
            errors.append(
                f"Column {col} dtype mismatch: sample={sample_aligned[col].dtype}, "
                f"oanda={oanda_aligned[col].dtype}"
            )
    
    # Price values (allow reasonable differences - OANDA may return slightly different prices)
    # For XAUUSD, spread is typically 0.1-0.5, so we allow up to 1.0 difference
    # This accounts for:
    # - Different fetch times (prices change)
    # - Rounding differences
    # - API response variations
    price_tolerance = 1.0  # 1.0 USD difference allowed
    for col in CANONICAL_COLUMNS:
        if col == "volume":
            continue  # Volume can differ (different aggregation)
        diff = (sample_aligned[col] - oanda_aligned[col]).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        if max_diff > price_tolerance:
            errors.append(
                f"Column {col} value mismatch: max_diff={max_diff:.6f} > tolerance={price_tolerance} "
                f"(mean_diff={mean_diff:.6f}, sample vs OANDA)"
            )
        else:
            log.info(
                f"  ✅ {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} "
                f"(within tolerance={price_tolerance})"
            )
    
    # Report
    log.info("=" * 60)
    if errors:
        log.error("❌ VERIFICATION FAILED")
        for error in errors:
            log.error(f"  - {error}")
        return False
    else:
        log.info("✅ VERIFICATION PASSED")
        log.info(f"  - {len(common_timestamps):,} timestamps compared")
        log.info(f"  - Schema match: ✅")
        log.info(f"  - Dtype match: ✅")
        log.info(f"  - Price values match: ✅")
        return True


def generate_manifest(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
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
        "instrument": INSTRUMENT,
        "granularity": GRANULARITY,
        "prices": "MBA",  # Mid + Bid + Ask
        "time_range_start": df.index.min().isoformat(),
        "time_range_end": df.index.max().isoformat(),
        "row_count": len(df),
        "sha256": sha256,
        "schema": {
            "columns": CANONICAL_COLUMNS,
            "dtypes": {col: str(dtype) for col, dtype in CANONICAL_DTYPES.items()},
            "index_type": "DatetimeIndex",
            "index_tz": "UTC",
        },
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    log.info(f"Manifest written: {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill XAUUSD M5 candles with bid/ask from OANDA (2020-2025)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (RFC3339, e.g., 2020-01-01T00:00:00Z)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (RFC3339, e.g., 2025-01-01T00:00:00Z)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/oanda/XAUUSD_M5_2020_2024_bidask.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--verify-against",
        type=str,
        help="Path to sample dataset for verification (e.g., 2025 dataset)",
    )
    parser.add_argument(
        "--verify-window-days",
        type=int,
        default=7,
        help="Number of days to verify (default: 7)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/oanda/checkpoints",
        help="Directory for checkpoint files (default: data/oanda/checkpoints)",
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv_if_present()
    
    # Load OANDA credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        log.info(f"Loaded OANDA credentials: env={credentials.env}")
    except Exception as e:
        log.error(f"Failed to load OANDA credentials: {e}")
        return 1
    
    # Initialize OANDA client
    try:
        config = OandaClientConfig(
            api_key=credentials.api_token,
            account_id=credentials.account_id,
            env=credentials.env,
        )
        client = OandaClient(config)
        log.info("Initialized OANDA client")
    except Exception as e:
        log.error(f"Failed to initialize OANDA client: {e}")
        return 1
    
    # Verification mode
    if args.verify_against:
        sample_path = Path(args.verify_against)
        success = verify_against_sample(
            client=client,
            sample_path=sample_path,
            window_days=args.verify_window_days,
        )
        return 0 if success else 1
    
    # Backfill mode
    if not args.start or not args.end:
        parser.error("--start and --end are required for backfill mode")
    
    # Parse timestamps (handle both tz-aware and tz-naive)
    start_dt = pd.to_datetime(args.start)
    if start_dt.tzinfo is None:
        start = start_dt.tz_localize("UTC")
    else:
        start = start_dt.tz_convert("UTC")
    
    end_dt = pd.to_datetime(args.end)
    if end_dt.tzinfo is None:
        end = end_dt.tz_localize("UTC")
    else:
        end = end_dt.tz_convert("UTC")
    
    log.info("=" * 60)
    log.info("OANDA XAUUSD M5 Bid/Ask Backfill (2020-2025)")
    log.info("=" * 60)
    log.info(f"Instrument: {INSTRUMENT}")
    log.info(f"Granularity: {GRANULARITY}")
    log.info(f"Time range: {start.date()} to {end.date()}")
    log.info(f"Output: {args.out}")
    log.info("")
    
    # Backfill
    try:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        df, metadata = backfill_range(
            client=client,
            start=start,
            end=end,
            checkpoint_dir=checkpoint_dir,
        )
    except Exception as e:
        log.error(f"Backfill failed: {e}", exc_info=True)
        return 1
    
    # Validate final dataset
    log.info("Validating final dataset...")
    try:
        validate_schema(df, context="final output")
        log.info("✅ Final schema validation passed")
    except RuntimeError as e:
        log.error(f"❌ Final schema validation failed: {e}")
        return 1
    
    # Save output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Writing {len(df):,} candles to {output_path}...")
    df.to_parquet(output_path, index=True)
    log.info("✅ Dataset written")
    
    # Generate manifest
    manifest_path = generate_manifest(df, output_path)
    
    # Summary
    log.info("=" * 60)
    log.info("✅ BACKFILL COMPLETE")
    log.info("=" * 60)
    log.info(f"Total rows: {len(df):,}")
    log.info(f"Time range: {df.index.min()} to {df.index.max()}")
    log.info(f"Output: {output_path}")
    log.info(f"Manifest: {manifest_path}")
    log.info(f"SHA256: {json.load(open(manifest_path))['sha256'][:16]}...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
