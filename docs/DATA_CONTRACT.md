# Data Contract

**Version:** 1.0  
**Last Updated:** 2025-01-21  
**Purpose:** Single Source of Truth for data paths, formats, slicing rules, and invariants.

## Canonical Data Roots

### 1. Candles Data (OHLC)

**Primary Path Pattern:**
```
{data_root}/oanda/years/{YEAR}/xauusd_m5_{YEAR}_bid_ask.parquet
```

**Alternative Path Patterns:**
```
{data_root}/oanda/years/{YEAR}.parquet
{data_root}/oanda/years/xauusd_m5_{YEAR}_bid_ask.parquet
```

**After DEL 4 (GX1_DATA split):**
- Default `data_root`: `../GX1_DATA/data` (via `GX1_DATA_DATA` env var)
- Can be overridden with `--data-root` argument

**Format:**
- **File Type:** Parquet
- **Index:** `pd.DatetimeIndex` (required)
- **Timezone:** UTC (required)
- **Columns:**
  - `open`, `high`, `low`, `close` (float32/float64)
  - `bid_open`, `bid_high`, `bid_low`, `bid_close` (float32/float64)
  - `ask_open`, `ask_high`, `ask_low`, `ask_close` (float32/float64)
  - `volume` (float32/float64, optional)

**Validation:**
- Hard-fail if index is not `DatetimeIndex`
- Hard-fail if index timezone is not UTC
- Hard-fail if required OHLC columns are missing

### Reserved Candle Columns (Case-Insensitive)

**Purpose:** Prevent collisions between candle data and prebuilt features.

**Reserved Columns:**
- `open`, `high`, `low`, `close`, `volume`
- `bid_open`, `bid_high`, `bid_low`, `bid_close`
- `ask_open`, `ask_high`, `ask_low`, `ask_close`

**Rules:**
1. **Prebuilt Features:** Must NOT contain any reserved columns (case-insensitive match)
2. **CLOSE Special Case:** `CLOSE` feature is dropped from prebuilt schema and aliased from `candles.close` in transformer input assembly
3. **Case Collisions:** Case-insensitive collisions (e.g., `close` vs `CLOSE`) are fatal and must be fixed at source

**Validation:**
- Prebuilt builder (`build_fullyear_features_parquet.py`) hard-fails if reserved columns are found
- Transformer input assembly aliases `CLOSE` from `candles.close` when needed
- Manifest generator (`generate_feature_manifest.py`) hard-fails if `CLOSE` is found in prebuilt schema
- Prebuilt loader (`prebuilt_features_loader.py`) hard-fails if `CLOSE` is found in loaded schema

### 2. Prebuilt Features

**Primary Path Pattern:**
```
{data_root}/features/xauusd_m5_{YEAR}_features_v10_ctx.parquet
```

**Alternative Path Pattern:**
```
{data_root}/prebuilt/TRIAL160/{YEAR}/xauusd_m5_{YEAR}_features_v10_ctx.parquet
```

**After DEL 4 (GX1_DATA split):**
- Default `data_root`: `../GX1_DATA/data` (via `GX1_DATA_DATA` env var)
- Can be overridden with `--prebuilt-parquet` argument (absolute path)

**Format:**
- **File Type:** Parquet
- **Index:** `pd.DatetimeIndex` (required)
- **Timezone:** UTC (required)
- **Index Alignment:** Must match candles DatetimeIndex exactly (row-by-row)
- **Columns:** Feature columns (see `FEATURE_MANIFEST.md`)

**Validation:**
- Hard-fail if index is not `DatetimeIndex`
- Hard-fail if index timezone is not UTC
- Hard-fail if index does not match candles index (for same year)
- Hard-fail if required features are missing
- **Hard-fail if schema contains reserved candle columns** (see "Reserved Candle Columns" section above)

## Timestamp Requirements

### DatetimeIndex Format

**Type:** `pd.DatetimeIndex`  
**Timezone:** UTC (required)  
**Validation:** Hard-fail if index is not DatetimeIndex or timezone is not UTC

**Example:**
```python
import pandas as pd

# CORRECT
df.index = pd.DatetimeIndex(df.index, tz='UTC')

# WRONG (will hard-fail)
df.index = pd.RangeIndex(0, len(df))  # Not DatetimeIndex
df.index = pd.DatetimeIndex(df.index)  # No timezone
```

### Timestamp Extraction

**Prebuilt Metadata:**
- Use `pyarrow.parquet.ParquetFile` metadata for efficient extraction
- Extract `min` and `max` timestamps from metadata (no full load)
- Convert to `pd.Timestamp` with UTC timezone

**Candles Metadata:**
- Load DataFrame and use `df.index[0]` and `df.index[-1]`
- Ensure timezone-aware before extraction

**Conversion:**
```python
import pandas as pd
import pyarrow.parquet as pq

# Prebuilt (efficient)
parquet_file = pq.ParquetFile(prebuilt_path)
schema = parquet_file.schema
# Extract min/max from metadata (if available) or load first/last row

# Candles (after load)
first_ts = pd.to_datetime(df.index[0], utc=True)
last_ts = pd.to_datetime(df.index[-1], utc=True)
```

## Slicing Rules

### Date Range Slicing

**Format:** `--smoke-date-range "START..END"` (inclusive)

**Example:**
```
--smoke-date-range "2025-01-01..2025-03-31"
```

**Rules:**
1. **Inclusive:** Both `START` and `END` are included in the slice
2. **Timezone:** Must be UTC (or will be converted to UTC)
3. **Format:** ISO8601 (e.g., `2025-01-01T00:00:00+00:00` or `2025-01-01`)
4. **Implementation:** `df.loc[start_ts:end_ts]` (pandas inclusive slicing)

**Timezone Handling:**
```python
# CORRECT: Timezone-aware slicing
if df.index.tz is not None:
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize('UTC')
    elif start_ts.tz != df.index.tz:
        start_ts = start_ts.tz_convert(df.index.tz)
df_subset = df.loc[start_ts:end_ts]
```

**Location:** `gx1/scripts/replay_eval_gated_parallel.py:split_data_into_chunks()`

### Bars Count Slicing

**Format:** `--smoke-bars N` (integer)

**Example:**
```
--smoke-bars 17077
```

**Rules:**
1. **Deterministic:** Always takes first N bars
2. **Implementation:** `df.iloc[:N]` or `df.head(N)`
3. **Order:** Must be sorted by index (DatetimeIndex)

**Location:** `gx1/scripts/run_depth_ladder_eval_multiyear.py:compute_data_universe_fingerprint()`

### Chunk Slicing (Parallel Replay)

**Format:** `(chunk_start, chunk_end, chunk_idx)` tuples

**Rules:**
1. **No Overlap:** Chunks do not overlap
2. **Coverage:** All chunks together cover the entire dataset
3. **Timezone-Aware:** Same timezone handling as date range slicing

**Location:** `gx1/scripts/replay_eval_gated_parallel.py:split_data_into_chunks()`

## Bar Counter Definitions

### Counter Semantics

| Counter | Definition | Location | Notes |
|---------|------------|----------|-------|
| `candles_total` | Total candles in dataset/subset | `compute_data_universe_fingerprint()` | From `len(df_candles)` |
| `candles_seen` | Total bars iterated in loop | `oanda_demo_runner.py` | Incremented for every `df.iterrows()` |
| `candles_iterated` | Total bars in subset (from chunk_footer) | `replay_eval_gated_parallel.py` | Should equal `candles_seen` if loop completes |
| `candles_processed` | Bars that passed warmup (reached model call) | `oanda_demo_runner.py` | `perf_n_bars_processed` |
| `warmup_skipped` | Bars skipped due to HTF warmup or min_bars_for_features | `oanda_demo_runner.py` | Incremented before `perf_n_bars_processed++` |
| `pregate_skipped` | Bars skipped by pregate (eligibility checks) | `oanda_demo_runner.py` | Incremented after warmup, before entry stage |
| `decision_eligible` | Bars that reached `evaluate_entry()` call | `oanda_demo_runner.py` | After pregate, before model |

### Counter Relationships

```
candles_seen = candles_iterated (if loop completes)
candles_seen = candles_processed + warmup_skipped + pregate_skipped
candles_processed = decision_eligible (if no early exits)
```

### Invariants

1. **Bars Skip Invariant:**
   ```
   bars_skipped == warmup_bars + eligibility_blocks
   ```
   Where:
   - `bars_skipped = bars_seen - bars_processed`
   - `warmup_bars = bars_skipped_warmup`
   - `eligibility_blocks = bars_skipped_pregate`
   
   **Location:** `gx1/scripts/replay_eval_gated_parallel.py:process_chunk()` (chunk_footer generation)
   
   **Hard-Fail:** Raises `RuntimeError` if invariant fails

2. **Completion Invariant:**
   ```
   completed == true IFF bars_iterated == bars_total_in_subset
   ```
   **Location:** `gx1/scripts/run_depth_ladder_eval_multiyear.py:run_smoke_eval()`
   
   **Hard-Fail:** Marks `completed=false` if invariant fails

3. **Warmup Invariant:**
   ```
   warmup_skipped >= 0
   warmup_skipped <= candles_total (sanity check)
   ```

4. **Pregate Invariant:**
   ```
   pregate_skipped >= 0
   pregate_skipped <= candles_processed (sanity check)
   ```

## Replay Invariants (FATAL)

### PREBUILT Mode Invariants

1. **Feature Build Disabled:**
   ```
   GX1_FEATURE_BUILD_DISABLED == "1"
   feature_build_call_count == 0
   ```
   **Hard-Fail:** Raises `RuntimeError` if violated

2. **No Forbidden Imports:**
   ```
   Forbidden modules NOT in sys.modules:
   - gx1.features.basic_v1
   - gx1.execution.live_features
   - gx1.features.runtime_v10_ctx
   - gx1.features.runtime_sniper_core
   ```
   **Hard-Fail:** Raises `RuntimeError` if violated
   **Preflight:** `gx1/scripts/preflight_prebuilt_import_check.py` (runs before workers)

3. **Prebuilt Used:**
   ```
   prebuilt_used == true
   prebuilt_bypass_count > 0 (if bars_processed > 0)
   ```
   **Hard-Fail:** Raises `RuntimeError` if violated

### Temperature Scaling Invariants

1. **Temperature Scaling Enabled:**
   ```
   temperature_scaling_effective_enabled == true
   ```
   **Hard-Fail:** Raises `RuntimeError` if false (in Depth Ladder eval)

### Data Universe Invariants

1. **Universe Match (Depth Ladder):**
   ```
   data_universe_fingerprint matches between baseline and L+1:
   - candles_first_ts
   - candles_last_ts
   - candles_rowcount_loaded
   - prebuilt_first_ts
   - prebuilt_last_ts
   - prebuilt_rowcount
   - policy_id (or hash)
   - replay_mode
   - temperature_scaling_effective_enabled
   ```
   **Hard-Fail:** Raises `RuntimeError` with "TRADE_UNIVERSE_MISMATCH" if violated

2. **Subset Range Match:**
   ```
   subset_first_ts_actual >= subset_first_ts_expected (warmup may skip early bars)
   subset_last_ts_actual == subset_last_ts_expected (if completed == true)
   candles_processed_actual == expected_candles_in_subset (if completed == true)
   ```
   **Hard-Fail:** Raises `RuntimeError` if violated (when `completed == true`)

## Environment Variables

### Data Path Control (After DEL 4)

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `GX1_DATA_DATA` | Data root directory | No | `../GX1_DATA/data` |
| `GX1_REPORTS_ROOT` | Reports root directory | No | `../GX1_DATA/reports` |
| `GX1_MODELS_ROOT` | Models root directory | No | `../GX1_DATA/models` |
| `GX1_REPLAY_USE_PREBUILT_FEATURES` | Enable PREBUILT mode | Yes (PREBUILT) | `"1"` |
| `GX1_REPLAY_PREBUILT_FEATURES_PATH` | Explicit prebuilt parquet path | Yes (PREBUILT) | None |
| `GX1_FEATURE_BUILD_DISABLED` | Disable feature building | Yes (PREBUILT) | `"1"` |
| `GX1_ALLOW_PARALLEL_REPLAY` | Allow parallel chunk processing | Optional | `"1"` |
| `GX1_ABORT_AFTER_N_BARS_PER_CHUNK` | Fast abort for testing | Optional | None |
| `GX1_STOP_REQUESTED` | Graceful shutdown flag | Optional | `"0"` |

**Note:** See `GX1_PATHS.md` for detailed path configuration.

### Depth Ladder Control

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `GX1_DEPTH_LADDER_MODE` | Enable depth ladder mode | Yes (Depth Ladder) | `"1"` |
| `GX1_DEPTH_LADDER_VARIANT` | Arm variant (baseline/lplus1) | Yes (Depth Ladder) | None |
| `GX1_DEPTH_LADDER_NUM_LAYERS` | Number of transformer layers | Yes (L+1 training) | `"4"` |

## Data Universe Fingerprint

### Purpose

Deterministic metadata to ensure baseline and L+1 evaluations operate on identical data universes.

### Components

1. **Candles Stats:**
   - `candles_rowcount_loaded` - Total rows in candles parquet
   - `candles_first_ts` - First timestamp in candles parquet
   - `candles_last_ts` - Last timestamp in candles parquet

2. **Prebuilt Stats:**
   - `prebuilt_rowcount` - Total rows in prebuilt parquet
   - `prebuilt_first_ts` - First timestamp in prebuilt parquet
   - `prebuilt_last_ts` - Last timestamp in prebuilt parquet

3. **Subset Stats (if applicable):**
   - `expected_candles_in_subset` - Expected candles in subset (from slicing)
   - `expected_prebuilt_rows_in_subset` - Expected prebuilt rows in subset
   - `subset_first_ts_expected` - Expected first timestamp in subset
   - `subset_last_ts_expected` - Expected last timestamp in subset

4. **Configuration:**
   - `data_root_resolved` - Resolved absolute path to data root
   - `prebuilt_parquet_resolved` - Resolved absolute path to prebuilt parquet
   - `policy_path_resolved` - Resolved absolute path to policy file
   - `policy_id` - Policy ID (or hash of policy file)
   - `replay_mode` - Must be `"PREBUILT"` (hard-fail if not)
   - `temperature_scaling_effective_enabled` - Temperature scaling status

**Location:** `gx1/scripts/run_depth_ladder_eval_multiyear.py:compute_data_universe_fingerprint()`

## References

- **Main Replay Runner:** `gx1/execution/oanda_demo_runner.py:_run_replay_impl()`
- **Parallel Orchestrator:** `gx1/scripts/replay_eval_gated_parallel.py:main()`
- **Depth Ladder Orchestrator:** `gx1/scripts/run_depth_ladder_eval_multiyear.py:main()`
- **Prebuilt Loader:** `gx1/execution/prebuilt_features_loader.py`
- **Data Map:** `reports/repo_audit/DATA_MAP.md`
