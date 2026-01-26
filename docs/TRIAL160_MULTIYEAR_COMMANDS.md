# Trial 160 Multi-Year Backtest Commands

**Date:** 2026-01-16  
**Status:** READY

## Overview

Backtest Trial 160 on 2020-2025 with same rails as 2025 (SSoT, PREBUILT, invariants), running 4 years in parallel for ~4× throughput.

## Architecture

- **Year = Isolated Job**: Each worker runs one year from start→finish (prebuilt + replay + report)
- **No Shared State**: Each year has its own output directory and invariants
- **Fail-Fast**: If one worker fails → hard-fail entire batch (but keep logs)

## Commands

### 1. Run Multi-Year Parallel Backtest

```bash
python3 gx1/scripts/run_trial160_multiyear_parallel.py \
    --years 2020,2021,2022,2023,2024,2025 \
    --max-workers 4
```

**What it does:**
- Runs 4 parallel workers (one per year)
- Each worker:
  1. Runs doctor check
  2. Builds prebuilt features for the year
  3. Runs replay (PREBUILT only)
  4. Generates reports
  5. Writes RUN_IDENTITY

**Output:**
- Per year: `reports/replay_eval/TRIAL160_YEARLY/{year}/`
- Status: `reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/MULTIYEAR_PARALLEL_STATUS.json`

**Expected time:** ~4× faster than sequential (4 parallel processes)

### 2. Run Single Year (Debug)

```bash
python3 gx1/scripts/run_trial160_year_job.py \
    --year 2020 \
    --data data/oanda/years/2020.parquet \
    --prebuilt-out data/prebuilt/TRIAL160/2020/ \
    --report-out reports/replay_eval/TRIAL160_YEARLY/2020/ \
    --policy policies/sniper_trial160_prod.json \
    --bundle models/entry_v10_ctx
```

**What it does:**
- Runs complete job for a single year
- Useful for debugging or testing

### 3. Aggregate Results

After all years complete, generate summary report:

```bash
python3 gx1/scripts/aggregate_trial160_multiyear.py \
    --years 2020,2021,2022,2023,2024,2025 \
    --report-base reports/replay_eval/TRIAL160_YEARLY
```

**Output:**
- `reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/MULTIYEAR_SUMMARY.md`
- `reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/MULTIYEAR_METRICS.json`

## Output Structure

```
reports/replay_eval/TRIAL160_YEARLY/
├── 2020/
│   ├── RUN_IDENTITY.json
│   ├── FULLYEAR_TRIAL160_REPORT_2020.md
│   ├── FULLYEAR_TRIAL160_METRICS_2020.json
│   ├── chunk_*/ (replay artifacts)
│   └── ...
├── 2021/
│   └── ...
├── ...
└── 2025/
    └── ...

reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/
├── MULTIYEAR_SUMMARY.md
├── MULTIYEAR_METRICS.json
└── MULTIYEAR_PARALLEL_STATUS.json
```

## Invariants (Enforced Per Worker)

For each year job (PREBUILT replay):
- ✅ RUN_IDENTITY must be written before trading
- ✅ `replay_mode == PREBUILT`
- ✅ `feature_build_call_count == 0`
- ✅ Schema validation PASS
- ✅ KeyErrors == 0 (hard-fail)
- ✅ Lookup invariant: `lookup_hits == lookup_attempts - eligibility_blocks`
- ✅ `policy_id` and `policy_sha256` must match `trial160_prod_v1`
- ✅ `bundle_sha256` must match expected
- ✅ No warnings that hide mismatch: mismatch = FATAL

## Lock Strategy

- **Per-Output-Dir Lock**: Each year has its own lock (doesn't block parallel execution)
- **Within-Job Lock**: Same year cannot run twice simultaneously
- **Global Lock**: Not used (parallel execution allowed)

## Error Handling

If one job fails:
- Stop new jobs from starting
- Terminate remaining workers gracefully
- Write `MULTIYEAR_PARALLEL_STATUS.json` with error diagnostics and log paths
- Hard-fail entire batch (no partial results)

## Mental Lock

**Same model + same policy (Trial 160) on all years.**

This means:
- ❌ No retraining
- ❌ No Optuna
- ❌ No adjustments

This is a **stress test**, not an optimization.

If Trial 160:
- **Survives 2020-2022** → Extremely strong signal
- **Collapses in one year** → Equally valuable (we see where and why)
