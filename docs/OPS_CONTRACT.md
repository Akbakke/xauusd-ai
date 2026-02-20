# GX1 Operations Contract: No Silent Death

## Overview

For TRUTH full-year runs and other critical replays, **watchdog + monitor are required** to ensure runs do not die silently.

## Components

### 1. Run Watchdog (`gx1/utils/watchdog.py`)

**Purpose:** Master-side watchdog that monitors run progress and detects stalls.

**Behavior:**
- Writes `HEARTBEAT.json` every 5 seconds (configurable via `GX1_WATCHDOG_HEARTBEAT_INTERVAL_SEC`)
- Detects progress via filesystem events:
  - New `chunk_footer.json` files
  - New `WORKER_BOOT.json` files
  - New/updated `RAW_PREBUILT_JOIN.json` files
  - New lines in `CHUNK_EXIT_STATUS.jsonl`
- Writes `RUN_STALL_FATAL.json` if no progress detected for `stall_timeout_seconds` (default: 120s)
- Hard-fails (exit code 2) when stall is detected

**Configuration:**
- `GX1_WATCHDOG_STALL_TIMEOUT_SEC`: Stall timeout in seconds (default: 120)
- `GX1_WATCHDOG_PROGRESS_WINDOW_SEC`: Progress window for detection (default: 30)
- `GX1_WATCHDOG_HEARTBEAT_INTERVAL_SEC`: Heartbeat write interval (default: 5.0)

**Integration:**
- Automatically started in `replay_eval_gated_parallel.py` after gates pass, before chunk planning
- Runs in a separate daemon thread
- Stops gracefully when run completes

### 2. Run Monitor (`gx1/scripts/monitor_run.py`)

**Purpose:** CLI tool for monitoring run status from outside the replay process.

**Usage:**
```bash
# Continuous monitoring
python3 gx1/scripts/monitor_run.py --run-dir /path/to/output --interval 5

# One-shot status check
python3 gx1/scripts/monitor_run.py --run-dir /path/to/output --once

# JSON output
python3 gx1/scripts/monitor_run.py --run-dir /path/to/output --json

# With stall warning threshold
python3 gx1/scripts/monitor_run.py --run-dir /path/to/output --stall-timeout 120
```

**Exit Codes:**
- `0`: Run completed successfully (`RUN_COMPLETED.json` exists)
- `2`: Run failed or fatal (`RUN_FAILED.json`, `RUN_STALL_FATAL.json`, or `*_FATAL.json` exists)
- `1`: Run still running or unknown status (for `--once` mode)

**Output:**
- Human-readable status lines with:
  - Timestamp
  - Run directory name
  - Stage (planning, submitting, running, completed)
  - Chunk counts (planned, submitted, completed, failed)
  - Heartbeat age (seconds since last update)
  - Progress age (seconds since last progress)
  - Stall warnings (if `--stall-timeout` is set)

### 3. Completion Markers

**RUN_COMPLETED.json:**
- Written when all chunks complete successfully
- Contains: `status`, `run_id`, `timestamp`, `chunks_submitted`, `chunks_completed`, `total_time_sec`

**RUN_FAILED.json:**
- Written when run fails (incomplete chunks, exceptions, etc.)
- Contains: `status`, `reason`, `run_id`, `timestamp`, `chunks_submitted`, `chunks_completed`

**RUN_STALL_FATAL.json:**
- Written by watchdog when stall is detected
- Contains: `status`, `now_utc`, `last_progress_utc`, `seconds_since_progress`, `stage`, `counts`, `chunk_artifact_snapshot`, `ps_snapshot`

**HEARTBEAT.json:**
- Written by watchdog every 5 seconds
- Contains: `timestamp_utc`, `stage`, `chunks_planned`, `chunks_submitted`, `chunks_completed`, `chunks_failed`, `last_progress_ts_utc`, `last_completed_chunk_id`, `active_children_pids`

## Required Workflow for TRUTH Full-Year Runs

### Before Starting Run

1. **Verify canonical environment:**
   ```bash
   source /home/andre2/venvs/gx1/bin/activate
   python3 gx1/tools/gx1_doctor.py
   ```

2. **Set required environment variables:**
   ```bash
   export GX1_RUN_MODE=TRUTH
   export GX1_CANONICAL_BUNDLE_DIR=/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91
   export GX1_GATED_FUSION_ENABLED=1
   export GX1_XGB_INPUT_FINGERPRINT=1
   ```

3. **Start replay:**
   ```bash
   gx1/scripts/run_replay_canonical.sh \
       --policy /path/to/policy.yaml \
       --data /path/to/raw.parquet \
       --prebuilt-parquet /path/to/prebuilt.parquet \
       --workers 19 \
       --chunk-local-padding-days 7 \
       --start-ts 2025-01-01T00:00:00Z \
       --end-ts 2025-12-31T23:59:59Z \
       --output-dir /path/to/output
   ```

### During Run

4. **Monitor in separate terminal:**
   ```bash
   python3 gx1/scripts/monitor_run.py \
       --run-dir /path/to/output \
       --interval 5 \
       --stall-timeout 120 \
       2>&1 | tee /path/to/output/MONITOR.log
   ```

5. **Watch for:**
   - Heartbeat updates (should change every ~5 seconds)
   - Progress indicators (chunks completing)
   - Stall warnings (if `--stall-timeout` is set)

### After Run Completes

6. **Verify completion:**
   ```bash
   python3 gx1/scripts/monitor_run.py --run-dir /path/to/output --once
   ```

7. **Check for fatal capsules:**
   ```bash
   ls -la /path/to/output/*_FATAL.json
   ```

8. **Review artifacts:**
   - `RUN_COMPLETED.json` or `RUN_FAILED.json`
   - `HEARTBEAT.json` (final state)
   - `WATCHDOG_STATE.json` (watchdog configuration)
   - `MONITOR.log` (monitor output)

## Failure Modes

### Silent Death (Prevented by Watchdog)

**Symptom:** Run stops without writing completion markers.

**Detection:**
- Watchdog detects no progress for `stall_timeout_seconds`
- Writes `RUN_STALL_FATAL.json`
- Hard-fails (exit code 2)

**Investigation:**
1. Check `RUN_STALL_FATAL.json` for:
   - `last_progress_utc`: When did progress last occur?
   - `chunk_artifact_snapshot`: Which chunks have artifacts?
   - `ps_snapshot`: Are processes still running?
2. Check `HEARTBEAT.json` for final state
3. Check worker logs in `chunk_*/logs/`
4. Check system logs for OOM killer, signals, etc.

### Master Crash (Detected by Monitor)

**Symptom:** Master process dies unexpectedly.

**Detection:**
- Monitor sees heartbeat age increasing beyond threshold
- No new completion markers appear
- Monitor reports "NO_HEARTBEAT_YET" or stale heartbeat

**Investigation:**
1. Check for `MASTER_UNCAUGHT_FATAL.json` in output directory
2. Check `/tmp/gx1_master_uncaught_*.json` (fallback location)
3. Check system logs (`dmesg`, `journalctl`)
4. Check for OOM killer messages

### Worker Stall (Detected by Watchdog)

**Symptom:** Workers stop making progress.

**Detection:**
- Watchdog detects no new `chunk_footer.json` or `RAW_PREBUILT_JOIN.json` files
- Progress age exceeds `stall_timeout_seconds`
- Watchdog writes `RUN_STALL_FATAL.json`

**Investigation:**
1. Check `RUN_STALL_FATAL.json` for which chunks are stalled
2. Check worker logs in `chunk_*/logs/worker_stderr.log`
3. Check for deadlock indicators (CPU usage, thread dumps)
4. Check memory usage (OOM near-miss)

## Testing

### Mini-Run Test (G1)

**Purpose:** Verify watchdog and monitor work correctly on a short run.

**Command:**
```bash
gx1/scripts/test_mini_run.sh
```

**Expected:**
- `HEARTBEAT.json` updates every ~5 seconds
- At least 19 `WORKER_BOOT.json` files appear
- `RUN_COMPLETED.json` written at end
- Monitor reports correct status

**Proof:** `MINI_PROOF.md` in output directory

### Intentional Stall Test (G2)

**Purpose:** Verify watchdog detects stalls and writes `RUN_STALL_FATAL.json`.

**Command:**
```bash
gx1/scripts/test_stall.sh
```

**Expected:**
- Workers boot successfully
- Workers are killed to simulate stall
- Watchdog detects stall within timeout
- `RUN_STALL_FATAL.json` written
- Master exits (not hanging)

**Proof:** `STALL_PROOF.md` in output directory

## Best Practices

1. **Always use canonical runner:** `gx1/scripts/run_replay_canonical.sh`
2. **Always monitor long runs:** Use `monitor_run.py` in separate terminal
3. **Set appropriate timeouts:** Adjust `GX1_WATCHDOG_STALL_TIMEOUT_SEC` based on expected chunk duration
4. **Check completion markers:** Always verify `RUN_COMPLETED.json` or `RUN_FAILED.json` exists
5. **Review fatal capsules:** If any `*_FATAL.json` exists, investigate root cause
6. **Keep monitor logs:** Save `MONITOR.log` for post-run analysis

## Summary

**No silent death:** Watchdog + monitor ensure that runs either:
- Complete successfully (`RUN_COMPLETED.json`)
- Fail explicitly (`RUN_FAILED.json` or `*_FATAL.json`)
- Are killed by watchdog if they stall (`RUN_STALL_FATAL.json`)

**Required for:** TRUTH full-year runs, any run > 1 day, any run with > 10 workers.
