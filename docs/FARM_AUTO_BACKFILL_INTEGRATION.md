# FARM Auto-Backfill Integration

## Overview

FARM now has the same auto-backfill robustness as SNIPER. Every time you run `scripts/run_live_demo_farm.sh`, it automatically:

1. **Backfills missing XAU_USD M5 candles** from OANDA
2. (Optional hook for future trade backfill)
3. **Starts FARM live** as normal

## What Changed

### Modified Files

- `scripts/run_live_demo_farm.sh` - Added auto-backfill step before FARM start

### Shared Backfill Script

Both SNIPER and FARM use the same shared backfill script:
- `gx1/scripts/backfill_xauusd_m5_from_oanda.py`

This ensures consistency and avoids code duplication.

## Flow

```
1. Load .env
2. Validate OANDA credentials
3. Create run directory
4. [NEW] Step 1: Candle backfill (required - exits if fails)
5. [NEW] Optional hook for trade backfill (commented out for future)
6. Preflight sanity checks
7. Start FARM (unchanged)
```

## Behavior

### Candle Backfill

- **Idempotent**: Safe to run multiple times
- **Smart detection**: Only fetches candles if `last_ts < now_utc - 1 bar`
- **Fatal on error**: Script exits if backfill fails (FARM won't start without data)
- **Logging**: Uses `[BACKFILL_CANDLES]` prefix for easy filtering

Example logs:
```
[RUN_LIVE_FARM] Step 1: Backfilling missing XAU_USD M5 candles...
[BACKFILL_CANDLES] No new candles to backfill (last_ts=2025-12-26 21:55:00+00:00, now=2025-12-27 20:00:00+00:00)
```

### Trade Backfill

- **Optional hook**: Commented out in script, ready for future implementation
- **Same pattern as SNIPER**: Can be enabled when needed

## Usage

No changes needed - just run FARM as before:

```bash
./scripts/run_live_demo_farm.sh
```

The backfill happens automatically before FARM starts.

## Testing

To verify backfill works:

1. **Test candle backfill**:
   ```bash
   python3 gx1/scripts/backfill_xauusd_m5_from_oanda.py
   ```
   Should show `[BACKFILL_CANDLES]` messages.

2. **Test full flow** (without starting FARM):
   ```bash
   # Check script syntax
   bash -n scripts/run_live_demo_farm.sh
   ```

3. **Run FARM** and watch logs:
   ```bash
   ./scripts/run_live_demo_farm.sh
   ```
   You should see:
   - `[RUN_LIVE_FARM] Step 1: Backfilling missing XAU_USD M5 candles...`
   - `[BACKFILL_CANDLES]` messages
   - Then normal FARM startup logs

## Notes

- **No runtime changes**: FARM trading logic is unchanged
- **Shared backfill**: Same script used by both SNIPER and FARM
- **Idempotent**: Safe to run multiple times (won't duplicate data)
- **Error handling**: Candle backfill failure stops script
- **Environment**: Uses existing `.env` file and credential loading
