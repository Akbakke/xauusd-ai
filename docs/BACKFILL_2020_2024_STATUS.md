# Backfill 2020-2024 Status

**Started:** 2026-01-16  
**Status:** IN PROGRESS

## Command

```bash
python3 gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
    --start 2020-01-01T00:00:00Z \
    --end 2025-01-01T00:00:00Z \
    --out data/oanda/XAUUSD_M5_2020_2024_bidask.parquet \
    --checkpoint-dir data/oanda/checkpoints
```

## Checkpoint Status

Checkpoint file: `data/oanda/checkpoints/XAUUSD_M5_2020_2025_checkpoint.json`

**Last checkpoint:**
- `last_success_ts`: [will be updated]
- `n_rows`: [will be updated]
- `chunk_num`: [will be updated]

## Expected Output

**File:** `data/oanda/XAUUSD_M5_2020_2024_bidask.parquet`

**Estimated:**
- Total rows: ~1,050,000 (5 years × ~210,000 rows/year)
- Time range: 2020-01-01 to 2024-12-31
- File size: ~150-200 MB

**Manifest:** `data/oanda/MANIFEST_XAUUSD_M5_2020_2024_bidask.json`
- Contains: row_count, sha256, time_range, schema

## Next Steps (After Completion)

### Step B: Split by Year

```bash
python3 gx1/scripts/split_parquet_by_year.py \
    --input data/oanda/XAUUSD_M5_2020_2024_bidask.parquet \
    --output-dir data/oanda/years
```

**Output files:**
- `data/oanda/years/2020.parquet`
- `data/oanda/years/2021.parquet`
- `data/oanda/years/2022.parquet`
- `data/oanda/years/2023.parquet`
- `data/oanda/years/2024.parquet`

**Each with:**
- `MANIFEST_<year>.json` (row_count, sha256, time_range)

## Important Notes

1. **Do not modify** `XAUUSD_M5_2020_2024_bidask.parquet` after completion
2. **Note** row count and SHA256 from manifest
3. **Resume** is automatic if interrupted (uses checkpoint)
4. **No parallelization** - checkpoint is there for a reason

## Future Phases (Overview, Not Action Now)

### Phase 1: Prebuilt per Year
- Build prebuilt features for each year (2020-2025)
- Same feature schema, same fingerprint-rails

### Phase 2: Replay per Year
- FULLYEAR replay per year
- Same metrics as 2025:
  - PnL, DD, P1/P5/P50
  - Per-session breakdown
  - Block rates
  - Kill-chain

### Phase 3: Multi-Year Aggregator
- One report showing:
  - Best year / worst year
  - DD per year
  - Regime signatures
  - Whether edge degrades gradually or abruptly

## Mental Lock

**Same model + same policy (Trial 160) on all years.**

This means:
- No retraining
- No Optuna
- No adjustments

This is a **stress test**, not an optimization.

If Trial 160:
- **Survives 2020-2022** → Extremely strong signal
- **Collapses in one year** → Equally valuable (we see where and why)
