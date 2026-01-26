# SNIPER Auto-Backfill Integration

## Overview

Every time you run `scripts/run_live_demo_sniper.sh`, it now automatically:

1. **Backfills missing XAU_USD M5 candles** from OANDA
2. **Optionally backfills today's trades** from OANDA into trade journal
3. **Starts SNIPER live** as normal

## What Changed

### Modified Files

- `scripts/run_live_demo_sniper.sh` - Added auto-backfill steps before SNIPER start
- `gx1/scripts/backfill_xauusd_m5_from_oanda.py` - Enhanced logging with `[BACKFILL_CANDLES]` prefix

### Flow

```
1. Load .env
2. Validate OANDA credentials
3. Create run directory
4. [NEW] Step 1: Candle backfill (required - exits if fails)
5. [NEW] Step 2: Trade backfill for today (best-effort, non-fatal)
6. Preflight sanity checks
7. Start SNIPER (unchanged)
```

## Behavior

### Candle Backfill

- **Idempotent**: Safe to run multiple times
- **Smart detection**: Only fetches candles if `last_ts < now_utc - 1 bar`
- **Fatal on error**: Script exits if backfill fails (SNIPER won't start without data)
- **Logging**: Uses `[BACKFILL_CANDLES]` prefix for easy filtering

Example logs:
```
[BACKFILL_CANDLES] No new candles to backfill (last_ts=2025-12-26 21:55:00+00:00, now=2025-12-27 19:55:00+00:00)
[BACKFILL_CANDLES] Added 5 candles from 2025-12-26 22:00:00+00:00 to 2025-12-26 22:20:00+00:00. Total now: 70,094 bars.
```

### Trade Backfill

- **Best-effort**: Non-fatal if it fails
- **Today's trades only**: Uses UTC date range for current day
- **Targets current run**: Backfills into the run directory being created
- **Graceful failure**: Logs warning and continues if no trades or errors

## Usage

No changes needed - just run SNIPER as before:

```bash
./scripts/run_live_demo_sniper.sh
```

The backfill happens automatically before SNIPER starts.

## Testing

To verify backfill works:

1. **Test candle backfill**:
   ```bash
   python3 gx1/scripts/backfill_xauusd_m5_from_oanda.py
   ```
   Should show `[BACKFILL_CANDLES]` messages.

2. **Test full flow** (without starting SNIPER):
   ```bash
   # Check script syntax
   bash -n scripts/run_live_demo_sniper.sh
   ```

3. **Run SNIPER** and watch logs:
   ```bash
   ./scripts/run_live_demo_sniper.sh
   ```
   You should see:
   - `[RUN_LIVE_SNIPER] Step 1: Backfilling missing XAU_USD M5 candles...`
   - `[BACKFILL_CANDLES]` messages
   - `[RUN_LIVE_SNIPER] Step 2: Backfilling today's trades...`
   - Then normal SNIPER startup logs

## Notes

- **No runtime changes**: SNIPER trading logic is unchanged
- **Idempotent**: Safe to run multiple times (won't duplicate data)
- **Error handling**: Candle backfill failure stops script; trade backfill failure is logged but continues
- **Environment**: Uses existing `.env` file and credential loading
