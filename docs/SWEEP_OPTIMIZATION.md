# Sweep Optimization - Minimal Output + Early Stopping

## Overview

This document describes the sweep optimization features implemented to reduce runtime, I/O, and instability in large multiyear sweeps without losing decision-making capability.

**This is a pure ORCHESTRATOR + OUTPUT optimization.**
**NO changes to model, features, gates, exits, or thresholds.**

---

## DEL 1: Output Mode (Minimal vs Full)

### Purpose
Reduce I/O and file count dramatically for sweeps while preserving all decision-making metrics.

### Implementation

**Environment Variable:**
- `GX1_OUTPUT_MODE = "minimal" | "full"`
- Default for sweep scripts: `"minimal"` (set automatically in orchestrator)
- Default for single runs/debug: `"full"` (backward compatible)

**MINIMAL Mode:**
- **WRITES:**
  - `RUN_IDENTITY.json` (always required)
  - `FULLYEAR_TRIAL160_METRICS_*.json` (PnL, trades, MaxDD, tails, per-session summary)
  - `chunk_footer.json` (always required)
  - `orchestrator_summary.json` (always required)
- **SKIPS:**
  - `raw_signals_*.parquet`
  - `policy_decisions_*.parquet`
  - `trade_outcomes_*.parquet`
  - `attribution_*.json`
  - Per-bar dumps

**FULL Mode:**
- Existing behavior (all artifacts written)
- No changes

**Logging:**
- `output_mode` field in `RUN_IDENTITY.json`
- Clear log messages indicating which mode is active

### File Reduction
- **MINIMAL mode**: ~90% reduction in file count
- All trading truth preserved in metrics JSON
- No loss of PnL, trades, MaxDD, or session breakdown

---

## DEL 2: Partial Aggregator

### Purpose
Allow aggregation of incomplete runs (e.g., only 2020-2021 complete) without waiting for full multiyear completion.

### Implementation

**Enhanced `aggregate_overlap_sanity_pack.py`:**

1. **Partial Run Detection:**
   - Automatically detects if not all years are available
   - Calculates `coverage_years` and `coverage_ratio`

2. **Per-Variant Coverage:**
   - Each variant reports which years are included
   - Metrics computed only over available years

3. **Output Files:**
   - **Partial runs**: `PARTIAL_LEADERBOARD.md` and `PARTIAL_METRICS.json`
   - **Full runs**: `OVERLAP_SANITY_SUMMARY_2020_2025.md` and `OVERLAP_SANITY_METRICS_2020_2025.json`

4. **Clear Marking:**
   - Summary shows `coverage_years` and `coverage_ratio`
   - Per-variant breakdown shows coverage info
   - Warning if partial run

### Usage
```bash
# Works with incomplete runs
python3 gx1/scripts/aggregate_overlap_sanity_pack.py \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK
```

---

## DEL 3: Early Stopping After 2020

### Purpose
Stop poor-performing variants after 2020, saving compute for remaining years (2021-2025).

### Implementation

**Command-Line Arguments:**
- `--top-k-after-year 2020`: Enable early stopping after this year
- `--top-k 10`: Number of top variants to keep (default: 10)

**Process:**

1. **After ALL variants complete for early_stop_year (e.g., 2020):**
   - Load metrics for all variants
   - Rank variants by:
     - Total PnL (2020)
     - OVERLAP delta vs baseline
     - MaxDD (2020)

2. **Keep TOP_K variants:**
   - Always includes BASELINE (reference)
   - TOP_K best-performing variants

3. **Stop remaining variants:**
   - Cancel futures for dropped variants (years > early_stop_year)
   - Mark as `stopped_by_early_filter=true` in results
   - Log clearly which variants were dropped

4. **Continue only TOP_K:**
   - Remaining years (2021-2025) run only for kept variants
   - Dramatic reduction in total tasks

**Logging:**
- `early_filter_year`, `top_k`, `variants_dropped`, `variants_kept` in `ORCHESTRATOR_SUMMARY.json`
- Full traceability of early stopping decision

### Example
```bash
# Keep top 10 variants after 2020
python3 gx1/scripts/run_overlap_sanity_pack_multiyear.py \
  --policy trial160_prod_v1 \
  --prebuilt-parquet <PATH> \
  --data-root <PATH> \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK \
  --workers 6 \
  --top-k-after-year 2020 \
  --top-k 10
```

**Task Reduction:**
- Without early stopping: 354 tasks (59 variants × 6 years)
- With early stopping (TOP_10 after 2020): ~119 tasks (59 × 2020 + 10 × 5 remaining years)
- **~66% reduction in tasks**

---

## DEL 4: Full Artifacts Only for Finalists (Future)

**Status:** Not yet implemented

**Planned:**
- After full sweep / early-stopped sweep, identify TOP_FINAL (e.g., top 3-5)
- Re-run ONLY these variants with `GX1_OUTPUT_MODE=full`
- Full multiyear (2020-2025) with all artifacts
- Used for trade-level analysis, churn/exit reason, sanity before CANARY

---

## Backward Compatibility

- **Default behavior unchanged:** `GX1_OUTPUT_MODE` defaults to `"full"` if not set
- **Single runs unaffected:** All existing scripts work as before
- **Sweep scripts:** Automatically use `"minimal"` mode (can be overridden)

---

## Files Modified

1. `gx1/runtime/run_identity.py`
   - Added `output_mode` field
   - Reads `GX1_OUTPUT_MODE` env var

2. `gx1/scripts/replay_eval_gated.py`
   - Respects `output_mode` in `flush_replay_eval_collectors()`
   - Skips raw_signals, policy_decisions, trade_outcomes in minimal mode

3. `gx1/scripts/replay_eval_gated_parallel.py`
   - Respects `output_mode` in `merge_artifacts()`
   - Skips parquet merging in minimal mode

4. `gx1/scripts/run_overlap_sanity_pack_multiyear.py`
   - Sets `GX1_OUTPUT_MODE="minimal"` by default
   - Implements early stopping logic
   - Logs early stopping status in summary

5. `gx1/scripts/aggregate_overlap_sanity_pack.py`
   - Supports partial runs
   - Generates `PARTIAL_LEADERBOARD.md` and `PARTIAL_METRICS.json` for incomplete runs

---

## Usage Examples

### Minimal Output Sweep (Default)
```bash
python3 gx1/scripts/run_overlap_sanity_pack_multiyear.py \
  --policy trial160_prod_v1 \
  --prebuilt-parquet <PATH> \
  --data-root <PATH> \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK \
  --workers 6
# Automatically uses GX1_OUTPUT_MODE=minimal
```

### Early Stopping Sweep
```bash
python3 gx1/scripts/run_overlap_sanity_pack_multiyear.py \
  --policy trial160_prod_v1 \
  --prebuilt-parquet <PATH> \
  --data-root <PATH> \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK \
  --workers 6 \
  --top-k-after-year 2020 \
  --top-k 10
```

### Full Output (Override)
```bash
GX1_OUTPUT_MODE=full python3 gx1/scripts/run_overlap_sanity_pack_multiyear.py \
  --policy trial160_prod_v1 \
  --prebuilt-parquet <PATH> \
  --data-root <PATH> \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK \
  --workers 6
```

### Partial Aggregation
```bash
# Works even if only some years are complete
python3 gx1/scripts/aggregate_overlap_sanity_pack.py \
  --out-root reports/replay_eval/OVERLAP_SANITY_PACK
```

---

## Benefits

1. **Runtime Reduction:**
   - Early stopping: ~66% reduction in tasks (TOP_10 after 2020)
   - Minimal output: Faster I/O, less disk usage

2. **I/O Reduction:**
   - Minimal mode: ~90% reduction in file count
   - Faster writes, less disk space

3. **Stability:**
   - Fewer tasks = fewer failure points
   - Early stopping focuses compute on promising variants

4. **Decision-Making:**
   - All metrics preserved (PnL, trades, MaxDD, session breakdown)
   - No loss of information needed for variant selection

---

## Safety

- **Deterministic:** Early stopping uses deterministic ranking (no randomness)
- **Traceable:** All decisions logged in `ORCHESTRATOR_SUMMARY.json`
- **Reversible:** Can always re-run with full output if needed
- **Backward Compatible:** Default behavior unchanged
