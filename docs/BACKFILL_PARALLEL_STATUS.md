# Parallel Backfill 2020-2024 Status

**Started:** 2026-01-16  
**Status:** RUNNING (4 parallel processes)

## Command

```bash
python3 gx1/scripts/backfill_xauusd_m5_parallel_years.py \
    --years 2020 2021 2022 2023 2024 \
    --max-workers 4
```

## How It Works

1. **4 Parallel Processes**: One per year (2020, 2021, 2022, 2023, 2024)
2. **Independent Checkpoints**: Each year has its own checkpoint file
3. **Automatic Merge**: After all years complete, automatically merges into single file

## Output Files

**Per Year:**
- `data/oanda/years/2020.parquet`
- `data/oanda/years/2021.parquet`
- `data/oanda/years/2022.parquet`
- `data/oanda/years/2023.parquet`
- `data/oanda/years/2024.parquet`

**Each with:**
- `MANIFEST_<year>.json` (row_count, sha256, time_range)

**Merged (after completion):**
- `data/oanda/XAUUSD_M5_2020_2024_bidask.parquet`
- `MANIFEST_XAUUSD_M5_2020_2024_bidask.json`

## Checkpoints

Each year has its own checkpoint:
- `data/oanda/checkpoints/2020_checkpoint.json`
- `data/oanda/checkpoints/2021_checkpoint.json`
- `data/oanda/checkpoints/2022_checkpoint.json`
- `data/oanda/checkpoints/2023_checkpoint.json`
- `data/oanda/checkpoints/2024_checkpoint.json`

## Speed Improvement

**Expected:** ~4x faster than sequential (4 parallel processes)

**Estimated time:**
- Sequential: ~2-4 hours
- Parallel: ~30-60 minutes (depending on rate limits)

## Monitoring

Check progress:
```bash
# View log
tail -f /tmp/backfill_parallel.log

# Check running processes
ps aux | grep backfill_xauusd_m5_bidask_2020_2025.py

# Check year files
ls -lh data/oanda/years/*.parquet
```

## Resume

If interrupted, just run the same command again. Each year will resume from its own checkpoint.
