# SNIPER Resume After Sleep

## Overview

After Mac sleep/wake, run a single command to:
1. Backfill missing M5 candles from OANDA
2. (Optional) Backfill trades for today into trade journal
3. Start SNIPER live

## Usage

### Simple (recommended)

```bash
./scripts/resume_sniper_after_sleep.sh
```

### Skip trade backfill

```bash
./scripts/resume_sniper_after_sleep.sh --skip-trade-backfill
```

### Python directly

```bash
python gx1/scripts/resume_sniper_after_sleep.py
```

## What it does

### Step A: Candle Backfill
- Loads existing candle file: `data/raw/xauusd_m5_2025_bid_ask.parquet`
- Determines last timestamp
- Fetches missing M5 candles from OANDA
- Merges with existing data
- **Idempotent**: Safe to run multiple times (only fetches new candles)

### Step B: Trade Backfill (optional)
- Finds latest SNIPER run directory
- Fetches today's trades from OANDA
- Updates trade journal
- **Non-blocking**: If it fails, SNIPER still starts

### Step C: Start SNIPER
- Launches `scripts/run_live_demo_sniper.sh`
- Streams output live to terminal
- Exits with SNIPER's exit code

## Files

- `gx1/scripts/backfill_xauusd_m5_from_oanda.py` - Idempotent candle backfill
- `gx1/scripts/resume_sniper_after_sleep.py` - Main orchestrator
- `scripts/resume_sniper_after_sleep.sh` - Shell wrapper

## Requirements

- `.env` file with OANDA credentials
- Existing candle file (or will create new one)
- SNIPER run directory (for trade backfill)

## Notes

- Candle backfill failure → script exits (won't start SNIPER without data)
- Trade backfill failure → warning logged, SNIPER still starts
- All operations are idempotent (safe to run multiple times)
