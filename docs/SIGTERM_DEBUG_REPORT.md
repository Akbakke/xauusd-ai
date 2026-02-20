# SIGTERM Debug Report: FULLYEAR 2025 TRUTH 1W1C

## Root Cause Hypothesis

**Conclusion: External SIGTERM (most likely from Cursor/IDE run timeout or wrapper)**

### Evidence

1. **No internal SIGTERM sender**: Grep shows no `os.kill(..., SIGTERM)` or `signal.raise_signal(SIGTERM)` to self.
2. **Watchdog did NOT kill**: No `RUN_STALL_FATAL.json` in output dir. Watchdog uses `os._exit(2)` (exit code 2), not SIGTERM.
3. **Consistent bar count (~70117/70217)**: Suggests a time-based external killer – FULLYEAR takes ~10–15 min to process 70k bars; timeout fires at roughly the same point.
4. **Cursor run_command timeout**: The `run_terminal_cmd` / `run_command` in Cursor has a timeout. When it fires, the system typically sends SIGTERM (graceful) before SIGKILL.

### Hypothesis Tree (narrowed)

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| A) Parent (Cursor/IDE) timeout | Timeout param in run; consistent ~same bars | **Most likely** |
| B) Our code sends SIGTERM | No os.kill/raise_signal to self | Ruled out |
| C) OS OOM killer | Would see dmesg/OOM; exit code usually 137 | Unlikely |
| D) Watchdog stall kill | Would see RUN_STALL_FATAL; uses os._exit(2) | Ruled out |

---

## Changes Implemented

### 1. `SIGNAL_EVENT.json` (chunk_failure.py + replay_chunk.py)

When SIGTERM is received and `status="stopped"`:
- Writes `chunk_output_dir/SIGNAL_EVENT.json` with:
  - `pid`, `ppid`, `vmrss_kb`, `vmhwm_kb` (from `/proc/self/status`)
  - `bars_processed`, `total_bars`, `last_ts`, `wall_clock_sec`
  - `env_hints` (GX1_STOP_REQUESTED, GX1_RUN_MODE)

### 2. `BAR_PROGRESS_HEARTBEAT.json` (oanda_demo_runner.py)

Every 1000 bars (CHECKPOINT_EVERY_BARS):
- Writes `chunk_dir/BAR_PROGRESS_HEARTBEAT.json` with:
  - `bars_processed`, `last_ts`, `wall_clock_sec`, `vmrss_kb`, `vmhwm_kb`

### 3. Watchdog progress (watchdog.py)

- Added `BAR_PROGRESS_HEARTBEAT.json` to progress detection: if mtime < 30s → progress.
- Prevents false stall detection during long in-process bar loops (1W1C).

### 4. run_fullyear env ordering (run_fullyear_2025_truth_proof.py)

- `GX1_WATCHDOG_STALL_TIMEOUT_SEC` is now set **before** `replay_main()` (was dead code after).

---

## How to Run FULLYEAR (avoid external SIGTERM)

**Recommended: Run outside Cursor/IDE**

```bash
cd /home/andre2/src/GX1_ENGINE

export GX1_DATA=/home/andre2/GX1_DATA
export GX1_CANONICAL_TRUTH_FILE=/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json
export GX1_CANONICAL_POLICY_PATH=/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml
export GX1_REQUIRED_VENV=/home/andre2/venvs/gx1/bin/python

# Option A: nohup (survives terminal close)
nohup /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_fullyear_2025_truth_proof \
  --workers=1 --chunks=1 > fullyear.log 2>&1 &

# Option B: tmux/screen
tmux new -s fullyear
# ... set env vars, then:
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_fullyear_2025_truth_proof --workers=1 --chunks=1

# Option C: Plain terminal (no IDE)
# Run in a separate terminal window, no timeout.
```

**If using Cursor**: Increase or disable the run timeout for this command (if configurable).

---

## Re-Run Checklist

After running FULLYEAR, verify:

1. **RUN_COMPLETED.json** exists in output dir  
2. **MERGE_PROOF_*.json** exists  
3. **chunk_0/chunk_footer.json**: `status == "ok"`  
4. **chunk_0/chunk_footer.json**: `bars_processed == total_bars` (70217)  
5. **No SIGNAL_EVENT.json** (if present, SIGTERM was received)  
6. **No RUN_FAILED.json** (or if present, reason is not SIGTERM/INCOMPLETE_CHUNKS)

```bash
RUN_DIR=/home/andre2/GX1_DATA/reports/fullyear_truth_proof/FULLYEAR_2025_PROOF_<timestamp>

# Quick checks
test -f "$RUN_DIR/RUN_COMPLETED.json" && echo "OK: RUN_COMPLETED" || echo "FAIL: RUN_COMPLETED missing"
test -f "$RUN_DIR/chunk_0/chunk_footer.json" && grep -q '"status": "ok"' "$RUN_DIR/chunk_0/chunk_footer.json" && echo "OK: footer status=ok" || echo "FAIL: footer status"
grep '"bars_processed"' "$RUN_DIR/chunk_0/chunk_footer.json" | grep -q 70217 && echo "OK: bars complete" || echo "FAIL: bars incomplete"
test ! -f "$RUN_DIR/chunk_0/SIGNAL_EVENT.json" && echo "OK: no SIGTERM" || echo "INFO: SIGTERM received (see SIGNAL_EVENT.json)"
```
