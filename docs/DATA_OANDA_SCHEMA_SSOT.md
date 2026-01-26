# OANDA Data Schema — Single Source of Truth

**Date:** 2026-01-16  
**Status:** CANONICAL

## 2025 Dataset (Source of Truth)

**File:** `data/raw/xauusd_m5_2025_bid_ask.parquet`

### Schema

**Index:**
- Type: `pandas.DatetimeIndex`
- Timezone: UTC
- Format: RFC3339 (ISO 8601 with timezone)
- Example: `2025-01-01 23:00:00+00:00`

**Columns (13 total, all float64):**
1. `open` - Mid price open
2. `high` - Mid price high
3. `low` - Mid price low
4. `close` - Mid price close
5. `volume` - Volume (float64)
6. `bid_open` - Bid price open
7. `bid_high` - Bid price high
8. `bid_low` - Bid price low
9. `bid_close` - Bid price close
10. `ask_open` - Ask price open
11. `ask_high` - Ask price high
12. `ask_low` - Ask price low
13. `ask_close` - Ask price close

**Shape:** (70217, 13) for 2025

**Time Range:** 2025-01-01 23:00:00 UTC to 2025-12-31 (approx)

**Granularity:** M5 (5-minute bars)

**Timestamp Alignment:**
- All timestamps are on 5-minute boundaries (00:00, 00:05, 00:10, ...)
- Timestamps are UTC
- Index is monotonic increasing (sorted ascending)
- No duplicate timestamps

**Data Quality:**
- All price fields are float64 (no NaN in 2025 dataset)
- Volume is float64 (can be 0.0 for some bars)
- Only complete bars included (complete=True from OANDA API)

## OANDA API Configuration

**Instrument:** `XAU_USD`

**Granularity:** `M5`

**Price Type:** `MBA` (Mid + Bid + Ask)

**API Endpoint:** `/instruments/{instrument}/candles`

**Parameters:**
- `from`: RFC3339 timestamp (inclusive)
- `to`: RFC3339 timestamp (exclusive, half-open interval)
- `granularity`: "M5"
- `price`: "MBA"
- `alignmentTimezone`: "UTC"

**Response Format:**
```json
{
  "candles": [
    {
      "time": "2025-01-01T23:00:00.000000000Z",
      "complete": true,
      "volume": 5,
      "mid": {"o": 2625.07, "h": 2625.2, "l": 2625.07, "c": 2625.2},
      "bid": {"o": 2624.39, "h": 2624.58, "l": 2624.39, "c": 2624.58},
      "ask": {"o": 2625.75, "h": 2625.86, "l": 2625.75, "c": 2625.82}
    }
  ]
}
```

**Rate Limits:**
- Max candles per request: 5000 (OANDA limit)
- Recommended chunk size: ~15 days (288 bars/day × 15 = 4320 bars)
- Backoff on 429 (rate limit) and 5xx errors

## Data Processing Rules

1. **Complete Bars Only:** Only include candles where `complete=True`
2. **Timestamp Normalization:** Floor to 5-minute boundary (e.g., 23:03:45 → 23:00:00)
3. **Duplicate Handling:** Keep last occurrence if duplicate timestamps found
4. **Missing Fields:** Hard-fail if any bid/ask field is missing (no fallback to mid)
5. **Monotonic Index:** Sort by timestamp ascending, verify monotonic increasing
6. **M5 Grid Validation:** Verify 5-minute step between consecutive bars (except market gaps)

## Source Script

**2025 Dataset Created By:**
- Script: `_archive_artifacts/20260107_224017/scripts/fetch_oanda_history_2025.py`
- Method: Direct `_request` to OANDA API with `price="MBA"`
- Chunking: 15-day chunks
- Filter: `complete=True` only

## Validation Rules

All new datasets (2020-2024) must match this schema exactly:
- Same column names (case-sensitive)
- Same column order
- Same dtypes (float64 for all)
- Same index type (DatetimeIndex, UTC)
- Same timestamp format (RFC3339, UTC)
- No additional columns
- No missing bid/ask fields
