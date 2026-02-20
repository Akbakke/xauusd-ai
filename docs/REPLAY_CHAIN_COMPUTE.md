# CHAIN_COMPUTE Replay Mode

## Overview

CHAIN_COMPUTE is a replay mode that maintains **100% live-faithful trade-state continuity** while parallelizing compute (features + model inference) for high CPU utilization.

**Key Principle**: Parallelize work, not time. Trade-state and exit-state are continuous (like live), but compute is parallelized across blocks.

## Architecture

### Master Process (State Owner)

The master process owns **ALL** trade-state:
- `open_trades`: Dictionary of currently open trades
- `exit_manager`: Exit policy evaluation and trailing stop state
- `trade_journal`: Entry/EXIT event logging

The master processes blocks in **strict chronological order** (bar-by-bar), ensuring:
- Trades can open in bar `t` and close at bar `t+k` even if across block boundaries
- Exit decisions are made deterministically on every bar
- State is never duplicated or diverged

### Compute Pool Workers

Compute workers **DO NOT** own trade-state. They:
1. Receive a block of timestamps/bars (already joined raw+prebuilt)
2. Run model inference in batches (vectorized)
3. Return per-bar compute outputs:
   - Model scores (`p_long`, `p_short`, or `p_hat`)
   - Optional gating inputs (regime/session info if used)
   - Telemetry (min/max/mean scores per block)

### Blocked Pipeline

1. **Block Size**: Configurable (default: 1024 bars)
2. **Prefetch Queue**: Master scheduler holds `K` blocks "in flight" (default: K=2..4)
3. **Strict Ordering**: Master processes blocks in strict sequence index
4. **Buffering**: If a block result arrives out of order, buffer until needed

## Mode Configuration

### Environment Variables

- `GX1_REPLAY_MODE=TRUTH_CHAIN_COMPUTE` (or `GX1_CHAIN_COMPUTE=1`)
- `GX1_REPLAY_ACCOUNTING_CLOSE_AT_END=1` (optional, default: False in chain mode)

### RUN_IDENTITY Fields

```json
{
  "replay_mode": "TRUTH_CHAIN_COMPUTE",
  "compute_pool_workers": 8,
  "block_size_bars": 1024,
  "prefetch_blocks": 2,
  "accounting_close_enabled": false
}
```

## Data Handling

### Preferred Approach (Minimal Overhead)

1. Master loader joins full DataFrame for window (or memmap)
2. Master slices per block and sends only necessary arrays to workers:
   - Timestamps: `[B]`
   - Features: `[B, feature_dim]` (snapshot) + `[B, seq_len, seq_dim]` if needed
3. Workers receive arrays and run inference

### Alternative (Higher Overhead)

- Worker loader does join for its block (more IO/overhead)

**Goal**: Keep IO low, CPU high. Since prebuilt is already fast, overhead is in Python/process orchestration and model inference.

## Model Inference Batching

1. Worker receives arrays shaped `[B, feature_dim]` (snapshot) + `[B, seq_len, seq_dim]` if needed
2. Run model inference in batches inside the worker (vectorized), not bar-by-bar
3. Return:
   - Scores per bar
   - Any needed outputs for master (e.g., uncertainty score)

## Exit Correctness (Live-Faithful)

1. Master runs `exit_manager` on every bar (like live)
2. Trades can open in bar `t` and close at bar `t+k` even if across "blocks"
3. `EXIT_COVERAGE` counters show natural exits without accounting close

## Journaling

1. Keep `TRADE_JOURNAL_EVENTS` (append-only):
   - `ENTRY` event logged at open
   - `EXIT` event logged at close
2. In chain mode, we expect both `ENTRY` and `EXIT` events over long windows
3. `YEAR_METRICS` should use `TRADE_JOURNAL_EVENTS` with `closed_coverage` likely >0

## Safety / Invariants (TRUTH-only)

### Ordering Invariant

- Bar timestamps processed strictly increasing; `FATAL` if not

### Compute Result Integrity

- `block_result.block_id` matches expected, `ts` range matches slice
- No `NaN`/`Inf` in scores; `FATAL`

### State Continuity

- `open_trades_end_of_block` carries to next block naturally (no snapshot needed; same process)

### Performance Counters

- `bars/sec`
- `model_infer_ms` per block
- `master_state_ms` per block
- CPU utilization target: raise from ~10–15% to much higher

## Usage

```bash
export GX1_RUN_MODE=TRUTH
export GX1_REPLAY_MODE=TRUTH_CHAIN_COMPUTE
export GX1_GATED_FUSION_ENABLED=1
export GX1_CANONICAL_BUNDLE_DIR=/path/to/bundle
export CANONICAL_PYTHON=/path/to/python

python3 gx1/scripts/replay_eval_chain_compute.py \
  --output-dir /path/to/output \
  --data /path/to/raw.parquet \
  --prebuilt-parquet /path/to/prebuilt.parquet \
  --policy /path/to/policy.yaml \
  --bundle-dir /path/to/bundle \
  --compute-workers 8 \
  --block-size-bars 1024 \
  --prefetch-blocks 2 \
  --start-ts 2025-01-01T00:00:00Z \
  --end-ts 2025-12-31T23:59:59Z
```

## Proof Plan

### Phase 1: Mini Proof (2–7 days)

- `chain_compute` with `compute_workers=8`, `block_size=1024`, `prefetch=2`

**PASS Criteria**:
- `RUN_COMPLETED.json` exists
- Both `ENTRY` and (at least some) `EXIT` events logged
- `YEAR_METRICS` populated (`closed_trades > 0`)
- Determinism check: run twice => identical counts and metrics

### Phase 2: FULLYEAR 2025

- `compute_workers` sweep: 4, 8, 12, 16
- Choose best throughput config

## Deliverables

- `docs/REPLAY_CHAIN_COMPUTE.md` (this file)
- `gx1/scripts/replay_eval_chain_compute.py` (canonical script)
- `RESULT_CARD.md` for proof run + fullyear run

## Implementation Status

- [x] Core architecture (master/worker separation)
- [x] Block pipeline with prefetch
- [ ] Model inference integration
- [ ] GX1DemoRunner integration
- [ ] Exit correctness
- [ ] Journaling
- [ ] Safety invariants
- [ ] Performance counters
- [ ] Phase 1 proof run
- [ ] Phase 2 FULLYEAR run
