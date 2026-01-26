# OANDA Backfill Commands — 2020-2025

**Date:** 2026-01-16  
**Status:** READY

## Quick Start

### 1. Test Backfill (2 weeks in 2020)

```bash
python3 gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
    --start 2020-01-01T00:00:00Z \
    --end 2020-01-15T00:00:00Z \
    --out data/oanda/XAUUSD_M5_TEST_2020_2weeks.parquet
```

**Expected output:**
- `data/oanda/XAUUSD_M5_TEST_2020_2weeks.parquet` (~2,440 rows)
- `data/oanda/MANIFEST_XAUUSD_M5_TEST_2020_2weeks.json`

### 2. Verify Against 2025 Sample

```bash
python3 gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
    --verify-against data/raw/xauusd_m5_2025_bid_ask.parquet \
    --verify-window-days 7
```

**Expected output:**
- ✅ VERIFICATION PASSED
- Schema match: ✅
- Dtype match: ✅
- Price values match: ✅ (within tolerance)

### 3. Full Backfill 2020-2024

**Note:** 2025 dataset already exists at `data/raw/xauusd_m5_2025_bid_ask.parquet`

```bash
python3 gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
    --start 2020-01-01T00:00:00Z \
    --end 2025-01-01T00:00:00Z \
    --out data/oanda/XAUUSD_M5_2020_2024_bidask.parquet \
    --checkpoint-dir data/oanda/checkpoints
```

**Expected output:**
- `data/oanda/XAUUSD_M5_2020_2024_bidask.parquet` (~1,050,000 rows estimated)
- `data/oanda/MANIFEST_XAUUSD_M5_2020_2024_bidask.json`
- `data/oanda/checkpoints/XAUUSD_M5_2020_2025_checkpoint.json` (updated during backfill)

**Estimated time:** ~2-4 hours (depending on API rate limits)

### 4. Merge 2020-2024 with 2025

After backfill completes, merge datasets:

```bash
python3 << PY
import pandas as pd
from pathlib import Path

# Load datasets
df_2020_2024 = pd.read_parquet('data/oanda/XAUUSD_M5_2020_2024_bidask.parquet')
df_2025 = pd.read_parquet('data/raw/xauusd_m5_2025_bid_ask.parquet')

# Merge (2020-2024 + 2025)
df_merged = pd.concat([df_2020_2024, df_2025], axis=0)
df_merged = df_merged.sort_index()
df_merged = df_merged[~df_merged.index.duplicated(keep='last')]

# Validate
print(f"Merged dataset: {len(df_merged):,} rows")
print(f"Time range: {df_merged.index.min()} to {df_merged.index.max()}")
print(f"2020-2024: {len(df_2020_2024):,} rows")
print(f"2025: {len(df_2025):,} rows")

# Save merged
output_path = Path('data/raw/xauusd_m5_2020_2025_bid_ask.parquet')
df_merged.to_parquet(output_path, index=True)
print(f"✅ Merged dataset saved: {output_path}")

# Generate manifest
import json
import hashlib
from datetime import datetime, timezone

sha256_hash = hashlib.sha256()
with open(output_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
sha256 = sha256_hash.hexdigest()

manifest = {
    "instrument": "XAU_USD",
    "granularity": "M5",
    "prices": "MBA",
    "time_range_start": df_merged.index.min().isoformat(),
    "time_range_end": df_merged.index.max().isoformat(),
    "row_count": len(df_merged),
    "sha256": sha256,
    "generated": datetime.now(timezone.utc).isoformat(),
}

manifest_path = output_path.parent / f"MANIFEST_{output_path.stem}.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Manifest saved: {manifest_path}")
PY
```

## Environment Variables

Script uses existing OANDA credentials from environment:
- `OANDA_API_TOKEN` or `OANDA_API_KEY`
- `OANDA_ACCOUNT_ID`
- `OANDA_ENV` (default: "practice")

## Checkpointing

If backfill is interrupted, it will resume from last checkpoint:
- Checkpoint file: `data/oanda/checkpoints/XAUUSD_M5_2020_2025_checkpoint.json`
- Contains: `last_success_ts`, `n_rows`, `chunk_num`
- Resume: Just run the same command again (automatically resumes)

## Validation

All datasets are validated with hard-fail on:
- Missing bid/ask fields
- Schema mismatch (columns, dtypes, index)
- Duplicate timestamps
- Non-monotonic index
- M5 grid violations (non-multiple-of-5min steps)
- NaN values in price fields

## Output Files

**Backfill output:**
- `data/oanda/XAUUSD_M5_2020_2024_bidask.parquet` - Main dataset
- `data/oanda/MANIFEST_XAUUSD_M5_2020_2024_bidask.json` - Metadata manifest

**Merged output (after merge):**
- `data/raw/xauusd_m5_2020_2025_bid_ask.parquet` - Complete 2020-2025 dataset
- `data/raw/MANIFEST_xauusd_m5_2020_2025_bid_ask.json` - Merged manifest

## Troubleshooting

**Rate limit errors (429):**
- Script automatically retries with exponential backoff
- If persistent, reduce `CHUNK_DAYS` in script (line 50)

**Schema validation failures:**
- Check that OANDA API returns complete bars (`complete=True`)
- Verify instrument name is exactly `XAU_USD`
- Check that `price="MBA"` is used in API request

**Missing data:**
- Some time periods may have no data (market closures)
- Script logs warnings for empty chunks
- Gaps > 24h are considered legitimate market closures
