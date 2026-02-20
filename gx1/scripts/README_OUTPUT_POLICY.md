# OUTPUT_POLICY - Output Management for GX1 Replay Runs

> **Shell note**: Disse kommandoene er for Bash/zsh (WSL). I PowerShell må du kjøre dem inne i wsl.

## Setup (One-time)

For å unngå å skrive `GX1_DATA_DIR` og full python-path i hver kommando, legg til dette i `~/.bashrc` (eller `~/.zshrc` hvis du bruker zsh):

```bash
# GX1 aliases and environment
export GX1_DATA_DIR="$HOME/GX1_DATA"
alias gx1py="$HOME/venvs/gx1/bin/python"
```

Etter dette kan du bruke kortform-eksempler (se nedenfor).

## Overview

This document describes the OUTPUT_MODE policy that prevents `GX1_DATA/reports/` from exploding with millions of files.

## OUTPUT_MODE Values

- **MINIMAL** (default): Only essential files (RUN_IDENTITY, RUN_COMPLETED, metrics)
- **DEBUG**: MINIMAL + telemetry + error logs
- **TRUTH**: Full artifacts (no per-bar/per-tick logs)

## Automatic Cleanup

Cleanup runs automatically after `RUN_COMPLETED.json` is written. It deletes:
- `chunk_*/` directories
- `logs/` directory
- `*_raw_*.parquet` files
- `*_decisions_*.parquet` files
- Additional files based on OUTPUT_MODE (see below)

**Safety**: Cleanup only runs if output_dir is under `GX1_DATA/reports/`. It never touches `GX1_DATA/data/`, `models/`, or `quarantine/`.

## What Gets Deleted Per Mode

### MINIMAL (default)
- All chunk directories
- All logs
- All raw/decisions parquet files
- ENTRY_FEATURES_TELEMETRY.json
- All log files

### DEBUG
- All chunk directories
- All logs
- All raw/decisions parquet files
- **Keeps**: ENTRY_FEATURES_TELEMETRY.json, error logs

### TRUTH
- All chunk directories
- All logs
- All raw/decisions parquet files
- **Keeps**: All merged artifacts, telemetry, logs

## Hard-Fail Tripwire: Reports Explosion Prevention

To prevent reports from exploding with millions of files (like the 1.8M files issue), a hard-fail tripwire is enforced:

- **MINIMAL**: Maximum 2,000 files in output directory
- **DEBUG**: Maximum 20,000 files
- **TRUTH**: No limit (full artifacts allowed)

If a run produces too many files, it fails early with a clear error message asking you to explicitly set `OUTPUT_MODE=DEBUG` or `OUTPUT_MODE=TRUTH` if you need more files.

This tripwire runs automatically during cleanup, so you'll never accidentally create 1.8M files again.

## Future-Proof: Capsule Detection

If `RUN_COMPLETED.json` exists but warning/fatal capsules are found (`*_FAIL*.json`, `MASTER_FATAL.json`, etc.), cleanup automatically upgrades to DEBUG mode to preserve debugging information.

## Manual Cleanup

### Clean a specific run directory:

**Full form:**
```bash
GX1_DATA_DIR=/home/andre2/GX1_DATA /home/andre2/venvs/gx1/bin/python /home/andre2/src/GX1_ENGINE/gx1/scripts/reports_cleanup.py --output-dir /path/to/run/output --dry-run
```

**Short form (after setup):**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/reports_cleanup.py --output-dir /path/to/run/output --dry-run
```

### Garbage Collect Old Reports:

**Full form:**
```bash
GX1_DATA_DIR=/home/andre2/GX1_DATA /home/andre2/venvs/gx1/bin/python /home/andre2/src/GX1_ENGINE/gx1/scripts/reports_gc.py --days 7 --dry-run
```

**Short form (after setup):**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/reports_gc.py --days 7 --dry-run
```

## Verification

### Quick sanity check:
```bash
OUT="/home/andre2/GX1_DATA/reports/<run_dir>"

# A) Check chunk directories are gone
find "$OUT" -maxdepth 2 -type d -name "chunk_*" -print
# Expected: no output

# B) Check RUN_IDENTITY has output_mode
/home/andre2/venvs/gx1/bin/python -c "import json; print(json.load(open('$OUT/RUN_IDENTITY.json')).get('output_mode', 'MISSING'))"
# Expected: MINIMAL (or DEBUG/TRUTH if set)
```

### Full verification script:

**Full form:**
```bash
GX1_DATA_DIR=/home/andre2/GX1_DATA /home/andre2/venvs/gx1/bin/python /home/andre2/src/GX1_ENGINE/gx1/scripts/verify_cleanup.py --output-dir /path/to/run/output
```

**Short form (after setup):**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/verify_cleanup.py --output-dir /path/to/run/output
```

## Recommended GC Policy (Future)

When setting up cron/Task Scheduler:
- Daily: `--days 14` + `--keep-last 20` per category
- TRUTH exception: Keep TRUTH runs for 90 days or "keep forever" if tagged `truth_lock=true`

## Run Index Ledger

All runs are automatically indexed in `$GX1_DATA/reports/_index.jsonl` (append-only JSON Lines format).

### View Recent Runs

**CLI tool:**
```bash
# Show last 20 runs
gx1py -m gx1.tools.gx1_runs --last 20

# Filter by kind
gx1py -m gx1.tools.gx1_runs --last 10 --kind replay_eval

# Filter by status
gx1py -m gx1.tools.gx1_runs --last 10 --status failed

# Show runs from last 24 hours
gx1py -m gx1.tools.gx1_runs --since-hours 24

# Raw JSON output
gx1py -m gx1.tools.gx1_runs --last 5 --json
```

### Verify Index Integrity

**Verification script:**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/verify_run_index.py
```

Checks:
- No duplicate `entry_id` in last 10k lines
- Last 5 runs have valid `output_dir` paths under `$GX1_DATA/reports/`

### Demo: 1-Minute Test

**1. Run a successful smoke test:**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/run_smoke_replay_us_disabled_proof.py \
  --output-dir "$GX1_DATA_DIR/reports/us_disabled_proof/SMOKE_TEST_$(date +%Y%m%d_%H%M%S)" \
  --days 2
```

**2. Run a FAIL case (doctor failure):**
```bash
# Temporarily set invalid GX1_DATA_DIR to trigger doctor failure
GX1_DATA_DIR=/nonexistent/path gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/run_smoke_replay_us_disabled_proof.py \
  --output-dir "$HOME/GX1_DATA/reports/us_disabled_proof/DOCTOR_FAIL_$(date +%Y%m%d_%H%M%S)" \
  --days 2
```

**3. View recent runs:**
```bash
gx1py -m gx1.tools.gx1_runs --last 10
```

**4. Verify index:**
```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/verify_run_index.py
```

**Expected output:**
- 2 new index entries (1 COMPLETED, 1 FAILED/DOCTOR_FATAL)
- `verify_run_index.py` returns exit code 0
- No duplicate `entry_id` entries
- All paths valid

### Deduplication

The index uses `entry_id` (SHA256 of `run_id|event|git_head_sha|canonical_output_dir`) to prevent duplicate entries. The `entry_id` is based on:
- `run_id`: Unique identifier for the run
- `event`: Terminal event (RUN_COMPLETED, RUN_FAILED, MASTER_FATAL, DOCTOR_FATAL) - this is the stable identifier, not `status`
- `git_head_sha`: Git commit SHA (if available)
- `canonical_output_dir`: Resolved, normalized output directory path

**Deduplication Strategy:**
- **Fullscan mode**: If `_index.jsonl` is <= 20MB, the entire file is scanned line-by-line to build a set of all `entry_id` values. This prevents duplicates even if the original entry is beyond the `last_n` window.
- **Last-N fallback**: If the file is > 20MB, only the last 2000 lines are checked (configurable via `last_n` parameter).

This ensures that `_index.jsonl` has at most one entry per `run_id` + `event` combination, even when:
- Multiple exception handlers write completion contracts
- Watchdog threads write completion contracts
- Cleanup code paths append entries
- Retries attempt to write the same terminal event multiple times

**Debug Observability:**
When using `append_run_index_dedup()` with `debug_stats={}`, the following information is populated:
- `dedup_hit`: `true` if duplicate was found and prevented, `false` otherwise
- `dedup_mode`: `"fullscan"` or `"last_n"` (or `"none"` if file was empty)
- `scanned_lines`: Number of lines scanned during deduplication check
- `invalid_lines_ignored`: Number of malformed/partial lines encountered and ignored

### Run Index Integrity Guarantees

**PASS-run behavior:**
- A successful run produces exactly **one** `RUN_COMPLETED` entry in the run index
- The entry has `event=RUN_COMPLETED` and `status=COMPLETED`
- No duplicate entries can be created, even if multiple code paths attempt to write completion

**FAIL-run behavior:**
- A failed run produces exactly **one** terminal event entry (MASTER_FATAL, DOCTOR_FATAL, or RUN_FAILED)
- The entry has `event` set to the terminal event type and `status=FAILED`
- No duplicate entries can be created, even if retries or multiple exception handlers attempt to write the same terminal event

**Terminal Event Uniqueness:**
- Each `run_id` can have **at most one** terminal event (RUN_COMPLETED, RUN_FAILED, MASTER_FATAL, DOCTOR_FATAL)
- This is enforced by:
  1. Deduplication at write time (prevents duplicate `entry_id`)
  2. Verification checks (detects any violations in last 10k lines)

**Timestamp Sanity:**
- Timestamps (`ts_utc`) should not go backwards more than 24 hours relative to the maximum timestamp seen in the verification window (last 2000 lines)
- This detects clock bugs or timestamp generation issues

### Verification

**Quick verification:**
```bash
# Verify run index integrity
gx1py -m gx1.tools.gx1_runs verify
```

**Expected output:**
- Exit code `0` on clean ledger
- Exit code `2` on detected violations (duplicate terminal events, timestamp issues, etc.)

**Verification checks:**
1. No duplicate `entry_id` in last 10k lines
2. Last 5 runs have valid `output_dir` paths under `$GX1_DATA/reports/`
3. Terminal event uniqueness: Each `run_id` has at most one terminal event in last 10k lines
4. Timestamp sanity: No timestamps going backwards >24h in last 2000 lines

### Intentional Fail Run (Testing)

To test run-index behavior under failure conditions:

```bash
gx1py /home/andre2/src/GX1_ENGINE/gx1/scripts/run_intentional_fail_run.py \
  --output-dir "$GX1_DATA_DIR/reports/test_kind/INTENTIONAL_FAIL_$(date +%Y%m%d_%H%M%S)"
```

**Expected behavior:**
1. Runs `gx1 doctor` preflight (as normal)
2. Writes `RUN_IDENTITY.json`
3. Triggers intentional fatal error (creates `MASTER_FATAL.json`)
4. Appends exactly **one** entry to run index with `event=MASTER_FATAL`
5. Returns exit code `2` (failure)

This proves that:
- FAIL-runs produce exactly one terminal event entry
- Deduplication prevents duplicate terminal events
- Run index remains consistent under failure conditions