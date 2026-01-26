# Performance Summary Merge - Test Plan

## Overview

This test plan verifies that the performance summary system for parallel replay chunks works correctly:
- Each chunk always writes its summary (even on crash)
- Global merge aggregates correctly and fails fast on missing chunks or incomplete chunks
- Invariants hold across all scenarios (machine-verifiable)

## Implementation Status

✅ **Implemented:**
- Per-chunk summary writing in `run_mini_replay_perf.py` (finally block)
- Global merge script `scripts/merge_perf_summaries.py` (fail-fast on missing or incomplete chunks)
- Crash injection mechanism in `gx1/execution/oanda_demo_runner.py` (env vars)
- Test scripts: `test_perf_merge_smoke.sh`, `test_perf_merge_missing_chunk.sh`, `test_perf_merge_crash_mid_chunk.sh`
- Invariant assertion helper: `scripts/assert_perf_invariants.py`

## Schema: Status Field

**Chunk JSON (`REPLAY_PERF_SUMMARY.json`):**
- `status`: `"complete"` | `"incomplete"` (string, required)
- `completed`: `true` | `false` (boolean, legacy compatibility)
- `early_stop_reason`: `string | null`

**Merge behavior:**
- Merge **fails fast** if any chunk has `status != "complete"`
- Error message identifies chunk_id and early_stop_reason

## Test Scenarios

### 1. Normal Case (Smoke Test)

**Description:** All chunks complete successfully and write summaries. Merge succeeds.

**Setup:**
- Mini dataset (few days, ~1000-2000 bars)
- 2-3 chunks
- All chunks complete normally

**Expected Results:**
- All chunk directories contain `REPLAY_PERF_SUMMARY.json` and `.md`
- Global merge succeeds (exit code 0)
- Global `REPLAY_PERF_SUMMARY.json` and `.md` exist in root output dir
- All invariants hold (see below)

**Test Script:** `scripts/test_perf_merge_smoke.sh`

---

### 2. Missing Summary Case

**Description:** One chunk summary file is deleted before merge. Merge must fail fast.

**Setup:**
- Run smoke test first (creates all summaries)
- Delete one chunk's `REPLAY_PERF_SUMMARY.json`

**Expected Results:**
- Merge script exits with non-zero code (U)
- Error message clearly identifies which chunk(s) are missing (V)
- No global summary files created (or existing ones not overwritten)

**Test Script:** `scripts/test_perf_merge_missing_chunk.sh`

---

### 3. Crash Mid-Chunk Case

**Description:** One chunk crashes (controlled exception) mid-execution. Summary should still be written in finally block with `status="incomplete"`, and merge should fail fast.

**Setup:**
- Mini dataset with 2-3 chunks
- Set `GX1_TEST_INDUCE_CRASH_CHUNK_ID=<chunk_id>` (e.g., "1")
- Set `GX1_TEST_CRASH_AFTER_N_BARS=<n>` (e.g., "50")
- Run replay

**Expected Behavior:**
- Crashed chunk: `REPLAY_PERF_SUMMARY.json` exists (written in finally) (W)
- Crashed chunk: `status == "incomplete"` (W)
- Crashed chunk: `early_stop_reason` contains "TEST_INDUCED_CRASH" (X)
- Crashed chunk: `bars_processed < bars_total`
- Other chunks: complete normally
- Merge: **Fails fast** (exit code != 0) (Y)
- Merge output: Lists incomplete chunk_id and early_stop_reason (Z)

**Expected Results:**
- Global summary is **NOT created** (merge fails)
- Merge error message identifies chunk_id and reason

**Test Script:** `scripts/test_perf_merge_crash_mid_chunk.sh`

---

## Invariants (Machine-Verifiable)

### Per-Chunk Invariants (A-I)

For each `chunk_i/REPLAY_PERF_SUMMARY.json`:

- **A.** `chunk_id` in JSON must match directory name `chunk_(\d+)`
- **B.** `window_start` and `window_end` must exist and be parseable; `window_start < window_end`
- **C.** `duration_sec >= 0`, `feat_time_sec >= 0` (from `runner_perf_metrics`)
- **D.** `bars_total > 0`
- **E.** `0 <= bars_processed <= bars_total`
- **F.** `trades_total >= 0`
- **G.** `feature_top_blocks` is list; each element has:
  - `name` (non-empty string)
  - `total_sec >= 0`
  - `count >= 0` (integer)
- **H.** `top_pandas_ops` is list; each element has:
  - `name` (non-empty string)
  - `total_sec >= 0`
  - `count >= 0` (integer)
- **I.** `status` exists and is `"complete"` in normal smoke-run

### Global Merge Invariants (J-R)

For `OUTDIR/REPLAY_PERF_SUMMARY.json`:

- **J.** `bars_total == SUM(chunk.bars_total)`
- **K.** `bars_processed == SUM(chunk.bars_processed)`
- **L.** `trades_total == SUM(chunk.trades_total)`
- **M.** `feat_time_sec == SUM(chunk.feat_time_sec)` (tolerance 1e-6)
- **N.** `duration_sec == MAX(chunk.duration_sec)` (parallel wall-clock, tolerance 1.0s)
- **O.** Feature blocks merged by name:
  - For each block-name in merged: `total_sec == SUM(per-chunk total_sec)`, `count == SUM(per-chunk count)`
- **P.** Merged `feature_top_blocks` sorted desc on `total_sec`, limited to top 15
- **Q.** Merged `top_pandas_ops` sorted desc on `total_sec`, limited to top 10
- **R.** `chunk_completion` table lists ALL chunk_ids and status per chunk

### Scenario-Specific Invariants

**Smoke Test (S, T):**
- **S.** All chunks: `status == "complete"` and `early_stop_reason` is null/empty
- **T.** Merge exit code == 0

**Missing Chunk Summary (U, V):**
- **U.** After deletion of one chunk summary: merge exit code != 0
- **V.** stderr/stdout must contain exact file name + chunk_id

**Crash Mid-Chunk (W, X, Y, Z):**
- **W.** The crashing chunk: chunk summary FILE MUST EXIST (finally), but `status == "incomplete"`
- **X.** `early_stop_reason` must contain "TEST_INDUCED_CRASH"
- **Y.** Merge exit code != 0
- **Z.** Merge output must list early_stop_reasons and identify chunk_id as incomplete

---

## Validation Tool

**Helper script:** `scripts/assert_perf_invariants.py`

**Usage:**
```bash
# Validate chunk summary
python3 scripts/assert_perf_invariants.py chunk <path_to_chunk_REPLAY_PERF_SUMMARY.json>

# Validate merged summary (requires chunk directories for comparison)
python3 scripts/assert_perf_invariants.py merged <path_to_merged_REPLAY_PERF_SUMMARY.json> <chunk_dir1> <chunk_dir2> ...
```

**Output:**
- `PASS: All invariants satisfied` (exit 0)
- `FAIL: Invariant violations:` with specific invariant IDs (A, B, C, ...) (exit 1)

---

## Implementation Requirements

### Crash Injection Mechanism

**Environment Variables:**
- `GX1_TEST_INDUCE_CRASH_CHUNK_ID`: Chunk ID (as string, e.g., "1") to crash. If unset, no crash.
- `GX1_TEST_CRASH_AFTER_N_BARS`: Number of bars to process before crashing. If unset, crash immediately.

**Behavior:**
- Only affects the specified chunk (matched by `GX1_CHUNK_ID` env var)
- Raises `RuntimeError("TEST_INDUCED_CRASH: ...")` after N bars processed
- Must not affect normal runs (when env vars not set)

**Location:** In `gx1/execution/oanda_demo_runner.py`, in the replay loop (`_run_replay_impl`, around line 7571 after `perf_n_bars_processed` increment).

---

## Running Tests

**Exact commands:**
```bash
# 1. Smoke test (normal case)
bash scripts/test_perf_merge_smoke.sh

# 2. Missing chunk test
bash scripts/test_perf_merge_missing_chunk.sh

# 3. Crash mid-chunk test
bash scripts/test_perf_merge_crash_mid_chunk.sh
```

All tests use `assert_perf_invariants.py` for machine-verifiable checks. Tests fail fast on any invariant violation.
