#!/bin/bash
#
# High-Throughput Baseline Evaluation Runner (FAIL-FAST + MAXCPU + MASTER DEATH FORENSICS)
#
# Maximizes CPU utilization and GUARANTEES master death is observable.
# CRITICAL: If master dies, RUN_FAILED.json is ALWAYS written with forensics.
#
# Usage:
#   ./run_baseline_eval_maxcpu.sh [--days DAYS] [--workers WORKERS] [--debug-master-death]
#

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

# ============================================================================
# MASTER DEATH FORENSICS: These variables are set early and used by trap
# ============================================================================
OUTPUT_DIR=""
REPLAY_PID=""
START_TS_EPOCH=""
MASTER_CMDLINE=""
WRAPPER_LOG=""
WORKERS=""
DAYS=""

# ============================================================================
# SIGNAL NAME DECODER
# ============================================================================
decode_signal() {
    local exitcode=$1
    if [[ ${exitcode} -ge 128 ]]; then
        local sig=$((exitcode - 128))
        case ${sig} in
            1) echo "SIGHUP" ;;
            2) echo "SIGINT" ;;
            3) echo "SIGQUIT" ;;
            4) echo "SIGILL" ;;
            6) echo "SIGABRT" ;;
            7) echo "SIGBUS" ;;
            8) echo "SIGFPE" ;;
            9) echo "SIGKILL" ;;
            11) echo "SIGSEGV" ;;
            13) echo "SIGPIPE" ;;
            14) echo "SIGALRM" ;;
            15) echo "SIGTERM" ;;
            *) echo "SIG${sig}" ;;
        esac
    else
        echo ""
    fi
}

# ============================================================================
# SIGNAL HANDLER: Writes completion contract and cleans up on SIGINT/SIGTERM
# ============================================================================
write_contract_and_cleanup() {
    local sig="$1"
    
    # Prevent recursive calls
    if [[ -n "${_IN_CLEANUP:-}" ]]; then
        echo "[SIGNAL ${sig}] Already in cleanup, exiting immediately"
        exit 143
    fi
    _IN_CLEANUP=1
    
    # Check OUTPUT_DIR is set
    if [[ -z "${OUTPUT_DIR}" ]] || [[ ! -d "${OUTPUT_DIR}" ]]; then
        echo "[SIGNAL ${sig}] ERROR: OUTPUT_DIR not set, cannot write contract" >&2
        exit 143
    fi
    
    # Idempotent: Don't overwrite if already written (unless it's a wrapper signal)
    if [[ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ]]; then
        echo "[SIGNAL ${sig}] RUN_COMPLETED.json exists, skipping cleanup"
        return 0
    fi
    
    # If RUN_FAILED.json exists, write RUN_FAILED_WRAPPER.json instead (wrapper signal takes precedence)
    local contract_file="${OUTPUT_DIR}/RUN_FAILED.json"
    if [[ -f "${contract_file}" ]]; then
        contract_file="${OUTPUT_DIR}/RUN_FAILED_WRAPPER.json"
    fi
    
    echo ""
    echo "[SIGNAL ${sig}] Wrapper received ${sig}, writing completion contract and cleaning up..."
    echo "[SIGNAL ${sig}] This ensures we always have a completion contract even if wrapper dies"
    
    local end_ts_epoch=$(date +%s)
    local duration=$((end_ts_epoch - START_TS_EPOCH))
    
    # Count chunks
    local chunks_completed=0
    local chunks_total=0
    local completed_list=""
    local incomplete_list=""
    
    for chunk_dir in "${OUTPUT_DIR}"/chunk_*/; do
        if [[ -d "${chunk_dir}" ]]; then
            chunks_total=$((chunks_total + 1))
            chunk_name=$(basename "${chunk_dir}")
            chunk_idx=${chunk_name#chunk_}
            if [[ -f "${chunk_dir}/chunk_footer.json" ]]; then
                chunks_completed=$((chunks_completed + 1))
                completed_list="${completed_list}${chunk_idx},"
            else
                incomplete_list="${incomplete_list}${chunk_idx},"
            fi
        fi
    done
    
    # Run scan_incomplete_chunks.py for summary
    local scan_summary=""
    local scan_table_path=""
    local WORKSPACE_ROOT
    WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if [[ -f "${WORKSPACE_ROOT}/gx1/scripts/scan_incomplete_chunks.py" ]]; then
        echo "[SIGNAL ${sig}] Running chunk status scan..."
        if /home/andre2/venvs/gx1/bin/python "${WORKSPACE_ROOT}/gx1/scripts/scan_incomplete_chunks.py" --run-dir "${OUTPUT_DIR}" > "${OUTPUT_DIR}/CHUNK_STATUS_TABLE.txt" 2>&1; then
            scan_table_path="${OUTPUT_DIR}/CHUNK_STATUS_TABLE.json"
            if [[ -f "${scan_table_path}" ]]; then
                scan_summary=$(/home/andre2/venvs/gx1/bin/python -c "import json; d=json.load(open('${scan_table_path}')); print(f\"Complete: {d.get('summary', {}).get('complete', 0)}, Incomplete: {d.get('summary', {}).get('incomplete', 0)}, Failed: {d.get('summary', {}).get('failed', 0)}\")" 2>/dev/null || echo "")
            fi
        fi
    fi
    
    # Write contract atomically (tmp + mv)
    local tmp_file="${contract_file}.tmp.$$"
    cat > "${tmp_file}" << CONTRACT_EOF
{
  "status": "FAILED",
  "reason": "WRAPPER_SIGNAL",
  "signal": "${sig}",
  "written_by": "wrapper_signal_handler",
  "timestamp": "$(date -Iseconds)",
  "wrapper_pid": $$,
  "replay_pid": ${REPLAY_PID:-0},
  "start_ts": "$(date -Iseconds -d @${START_TS_EPOCH} 2>/dev/null || echo "")",
  "end_ts": "$(date -Iseconds -d @${end_ts_epoch})",
  "duration_seconds": ${duration},
  "cmdline": "${MASTER_CMDLINE}",
  "workers": ${WORKERS:-0},
  "days": ${DAYS:-0},
  "chunks_submitted": ${chunks_total},
  "chunks_completed": ${chunks_completed},
  "chunks_incomplete": $((chunks_total - chunks_completed)),
  "completed_chunks": [${completed_list%,}],
  "incomplete_chunks": [${incomplete_list%,}],
  "scan_summary": "${scan_summary}",
  "scan_table_path": "${scan_table_path}"
}
CONTRACT_EOF
    
    # Atomic move
    mv "${tmp_file}" "${contract_file}" 2>/dev/null || true
    echo "[SIGNAL ${sig}] ✅ Wrote ${contract_file}"
    
    # Terminate replay-child and workers controllably
    if [[ -n "${REPLAY_PID}" ]] && ps -p "${REPLAY_PID}" > /dev/null 2>&1; then
        echo "[SIGNAL ${sig}] Terminating replay (PID=${REPLAY_PID})..."
        kill -TERM "${REPLAY_PID}" 2>/dev/null || true
        sleep 3
        if ps -p "${REPLAY_PID}" > /dev/null 2>&1; then
            echo "[SIGNAL ${sig}] Replay still running, sending SIGKILL..."
            kill -KILL "${REPLAY_PID}" 2>/dev/null || true
        fi
        # Kill children (best effort)
        pkill -P "${REPLAY_PID}" 2>/dev/null || true
        sleep 1
    fi
    
    echo "[SIGNAL ${sig}] Cleanup complete"
}

# ============================================================================
# MASTER DEATH HANDLER: ALWAYS writes RUN_FAILED.json with forensics
# ============================================================================
write_death_forensics() {
    local exitcode=$1
    local signal_name=$2
    local reason=$3
    
    if [[ -z "${OUTPUT_DIR}" ]] || [[ ! -d "${OUTPUT_DIR}" ]]; then
        echo "[FORENSICS] Cannot write RUN_FAILED.json: OUTPUT_DIR not set or missing" >&2
        return 1
    fi
    
    # Skip if RUN_COMPLETED.json already exists (successful run)
    if [[ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ]]; then
        echo "[FORENSICS] RUN_COMPLETED.json exists, skipping death forensics"
        return 0
    fi
    
    # Skip if RUN_FAILED.json already exists (Python wrote it)
    if [[ -f "${OUTPUT_DIR}/RUN_FAILED.json" ]]; then
        echo "[FORENSICS] RUN_FAILED.json already exists (written by Python)"
        return 0
    fi
    
    local end_ts_epoch=$(date +%s)
    local duration=$((end_ts_epoch - START_TS_EPOCH))
    
    # Count completed chunks
    local chunks_completed=0
    local chunks_incomplete=0
    local completed_list=""
    local incomplete_list=""
    
    for chunk_dir in "${OUTPUT_DIR}"/chunk_*/; do
        if [[ -d "${chunk_dir}" ]]; then
            chunk_name=$(basename "${chunk_dir}")
            chunk_idx=${chunk_name#chunk_}
            if [[ -f "${chunk_dir}/chunk_footer.json" ]]; then
                chunks_completed=$((chunks_completed + 1))
                completed_list="${completed_list}${chunk_idx},"
            else
                chunks_incomplete=$((chunks_incomplete + 1))
                incomplete_list="${incomplete_list}${chunk_idx},"
            fi
        fi
    done
    
    # Get last 200 lines of replay.log and wrapper.log
    local log_tail=""
    if [[ -f "${OUTPUT_DIR}/replay.log" ]]; then
        log_tail=$(tail -200 "${OUTPUT_DIR}/replay.log" 2>/dev/null | head -200 || echo "")
    fi
    # Also include wrapper.log tail if available
    if [[ -f "${WRAPPER_LOG}" ]]; then
        local wrapper_tail=$(tail -50 "${WRAPPER_LOG}" 2>/dev/null || echo "")
        log_tail="${log_tail}

--- WRAPPER LOG TAIL ---
${wrapper_tail}"
    fi
    
    # Get faulthandler dump if exists
    local fault_dump=""
    if [[ -f "${OUTPUT_DIR}/MASTER_FAULT_DUMP.txt" ]]; then
        fault_dump=$(cat "${OUTPUT_DIR}/MASTER_FAULT_DUMP.txt" 2>/dev/null || echo "")
    fi
    
    # Get strace tail if exists
    local strace_tail=""
    if [[ -d "${OUTPUT_DIR}/strace" ]]; then
        local master_strace=$(ls -t "${OUTPUT_DIR}/strace/trace."* 2>/dev/null | head -1)
        if [[ -n "${master_strace}" ]] && [[ -f "${master_strace}" ]]; then
            strace_tail=$(tail -100 "${master_strace}" 2>/dev/null || echo "")
        fi
    fi
    
    # Get memory log if exists
    local memory_log=""
    if [[ -f "${OUTPUT_DIR}/MASTER_MEMORY_LOG.jsonl" ]]; then
        memory_log=$(tail -50 "${OUTPUT_DIR}/MASTER_MEMORY_LOG.jsonl" 2>/dev/null || echo "")
    fi
    
    # Check for coredump
    local coredump_path=""
    local core_files=$(find /tmp -maxdepth 1 -name "core.python*" -newer "${OUTPUT_DIR}/replay.log" 2>/dev/null | head -1)
    if [[ -n "${core_files}" ]]; then
        coredump_path="${core_files}"
    fi
    
    # Write RUN_FAILED.json with full forensics
    cat > "${OUTPUT_DIR}/RUN_FAILED.json" << FORENSICS_EOF
{
  "status": "FAILED",
  "reason": "${reason}",
  "written_by": "wrapper_forensics",
  "timestamp": "$(date -Iseconds)",
  "master_pid": ${REPLAY_PID:-0},
  "exit_code": ${exitcode},
  "signal_name": "${signal_name}",
  "signal_number": $((exitcode >= 128 ? exitcode - 128 : 0)),
  "duration_seconds": ${duration},
  "start_ts": "$(date -Iseconds -d @${START_TS_EPOCH})",
  "end_ts": "$(date -Iseconds -d @${end_ts_epoch})",
  "cmdline": "${MASTER_CMDLINE}",
  "chunks_submitted": $((chunks_completed + chunks_incomplete)),
  "chunks_completed": ${chunks_completed},
  "chunks_incomplete": ${chunks_incomplete},
  "completed_chunks": [${completed_list%,}],
  "incomplete_chunks": [${incomplete_list%,}],
  "coredump_path": "${coredump_path}",
  "forensics": {
    "log_tail_lines": 200,
    "has_fault_dump": $([ -n "${fault_dump}" ] && echo "true" || echo "false"),
    "has_strace": $([ -n "${strace_tail}" ] && echo "true" || echo "false"),
    "has_memory_log": $([ -n "${memory_log}" ] && echo "true" || echo "false"),
    "has_coredump": $([ -n "${coredump_path}" ] && echo "true" || echo "false")
  }
}
FORENSICS_EOF

    # Also write log tail to separate file for easy reading
    if [[ -n "${log_tail}" ]]; then
        echo "${log_tail}" > "${OUTPUT_DIR}/DEATH_LOG_TAIL.txt"
    fi
    
    echo "[FORENSICS] ✅ Wrote RUN_FAILED.json with exit_code=${exitcode}, signal=${signal_name}, chunks=${chunks_completed}/$((chunks_completed + chunks_incomplete))"
    
    # Run scan_incomplete_chunks.py if available
    local WORKSPACE_ROOT
    WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if [[ -f "${WORKSPACE_ROOT}/gx1/scripts/scan_incomplete_chunks.py" ]]; then
        echo "[FORENSICS] Running chunk status scan..."
        /home/andre2/venvs/gx1/bin/python "${WORKSPACE_ROOT}/gx1/scripts/scan_incomplete_chunks.py" --run-dir "${OUTPUT_DIR}" > "${OUTPUT_DIR}/CHUNK_STATUS_TABLE.txt" 2>&1 || true
    fi
}

# ============================================================================
# ERR TRAP: Fail-fast diagnostics (wrapper errors only)
# ============================================================================
trap 'echo "[FATAL] wrapper failed at line $LINENO (exit=$?)" >&2; exit 1' ERR

# ============================================================================
# SCRIPT SETUP
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# DETECT CPU CORES
# ============================================================================
NPROC="$(/usr/bin/nproc 2>/dev/null || nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo "")"
if [[ -z "${NPROC}" || "${NPROC}" -lt 2 ]]; then
    echo "[FATAL] nproc returned '${NPROC}'. Something is wrong with CPU detection." >&2
    exit 3
fi
echo "[PREFLIGHT] NPROC=${NPROC}"

# Default arguments
DAYS=2
WORKERS=""
START_TS=""
DEBUG_MASTER_DEATH=0
POLICY_YAML="${WORKSPACE_ROOT}/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
BUNDLE_DIR="/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION"
PREBUILT_PARQUET="/home/andre2/GX1_DATA/data/data/entry_v10/entry_v10_1_dataset_seq90.parquet"
DATA_PATH="/home/andre2/GX1_DATA/data/data/entry_v9/full_2025.parquet"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --policy-yaml)
            POLICY_YAML="$2"
            shift 2
            ;;
        --bundle-dir)
            BUNDLE_DIR="$2"
            shift 2
            ;;
        --prebuilt-parquet)
            PREBUILT_PARQUET="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --start-ts)
            START_TS="$2"
            shift 2
            ;;
        --debug-master-death)
            DEBUG_MASTER_DEATH=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find Python executable
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_CMD="${VIRTUAL_ENV}/bin/python"
elif [[ -f "/home/andre2/venvs/gx1/bin/python" ]]; then
    PYTHON_CMD="/home/andre2/venvs/gx1/bin/python"
else
    PYTHON_CMD="/home/andre2/venvs/gx1/bin/python"
fi

# Auto-detect prebuilt start time
if [[ -z "$START_TS" ]] && [[ -f "${PREBUILT_PARQUET}" ]]; then
    echo "[PREFLIGHT] Detecting prebuilt start time..."
    PREBUILT_START=$("${PYTHON_CMD}" -c "
import pandas as pd
df = pd.read_parquet('${PREBUILT_PARQUET}')
if 'ts' in df.columns:
    ts_col = pd.to_datetime(df['ts'])
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize('UTC')
    print(ts_col.min())
elif isinstance(df.index, pd.DatetimeIndex):
    ts_idx = df.index
    if ts_idx.tz is None:
        ts_idx = ts_idx.tz_localize('UTC')
    print(ts_idx.min())
else:
    print('')
" 2>/dev/null || echo "")
    if [[ -n "$PREBUILT_START" ]] && [[ "$PREBUILT_START" != "0" ]] && [[ "$PREBUILT_START" != "" ]]; then
        START_TS="$PREBUILT_START"
        echo "[PREFLIGHT] Using prebuilt start time: ${START_TS}"
    fi
fi

# Auto-detect workers
if [[ -z "$WORKERS" ]]; then
    WORKERS=$((NPROC - 1))
fi

# Validate workers
if [[ "${WORKERS}" -lt 2 ]]; then
    echo "[FATAL] WORKERS=${WORKERS} is too low (minimum 2)." >&2
    exit 3
fi

if [[ "${WORKERS}" != "19" ]]; then
    echo "[PREFLIGHT] ⚠️  WARNING: WORKERS=${WORKERS} (not 19, but continuing)"
fi

echo "=================================================================================="
echo "HIGH-THROUGHPUT BASELINE EVALUATION RUNNER (MASTER DEATH FORENSICS ENABLED)"
echo "=================================================================================="
echo "Date: $(date -Iseconds)"
echo "CPU cores (NPROC): ${NPROC}"
echo "Workers: ${WORKERS}"
echo "Days: ${DAYS}"
echo "Debug master death: ${DEBUG_MASTER_DEATH}"
echo ""

# ============================================================================
# SET CPU THREAD ENVIRONMENT VARIABLES
# ============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYARROW_NUM_THREADS="${NPROC}"
export ARROW_NUM_THREADS="${NPROC}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "CPU Thread Environment Variables:"
echo "  OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS: 1"
echo "  PYARROW/ARROW: ${NPROC}"
echo "  PYTHONFAULTHANDLER: 1"
echo ""

# ============================================================================
# SET OUTPUT DIRECTORY (PRE-CREATE FOR FORENSICS + LOGGING)
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GX1_DATA_ROOT="${GX1_DATA_ROOT:-/home/andre2/GX1_DATA}"
OUTPUT_DIR="${GX1_DATA_ROOT}/reports/transformer_baseline_eval/BASELINE_MAXCPU_${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/strace"

# Wrapper log file (SIGPIPE-safe, always written internally)
WRAPPER_LOG="${OUTPUT_DIR}/wrapper.log"
START_TS_EPOCH=$(date +%s)

# ============================================================================
# SETUP SIGPIPE-SAFE LOGGING (WRAPPER IS RESPONSIBLE FOR OWN LOGGING)
# ============================================================================
# CRITICAL: Ignore SIGPIPE so wrapper doesn't die if stdout/stderr disappear
# This ensures wrapper survives even if stdout/stderr are closed
trap '' PIPE

# Redirect all output to wrapper.log (100% SIGPIPE-safe)
# Wrapper writes ONLY to its own log file - no pipe/tee dependencies
exec >>"${WRAPPER_LOG}" 2>&1

# Print header to wrapper.log
echo "=================================================================================="
echo "WRAPPER LOG START"
echo "=================================================================================="
echo "Timestamp: $(date -Iseconds)"
echo "Wrapper PID: $$"
echo "Output directory: ${OUTPUT_DIR}"
echo "Wrapper log: ${WRAPPER_LOG}"
echo ""

# Print one-line status to console if /dev/tty is available (non-blocking)
# Suppress error if /dev/tty is not available (e.g., in non-interactive contexts)
if [[ -c /dev/tty ]] 2>/dev/null; then
    echo "Running. Logs: ${WRAPPER_LOG}" > /dev/tty 2>/dev/null || true
fi

echo "Output directory: ${OUTPUT_DIR}"
echo "Wrapper log: ${WRAPPER_LOG}"
echo ""

# ============================================================================
# VERIFY INPUTS
# ============================================================================
if [[ ! -f "${POLICY_YAML}" ]]; then
    echo "❌ ERROR: Policy YAML not found: ${POLICY_YAML}"
    exit 1
fi

if [[ ! -d "${BUNDLE_DIR}" ]]; then
    echo "❌ ERROR: Bundle directory not found: ${BUNDLE_DIR}"
    exit 1
fi

if [[ ! -f "${PREBUILT_PARQUET}" ]]; then
    echo "❌ ERROR: Prebuilt parquet not found: ${PREBUILT_PARQUET}"
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "❌ ERROR: Data path not found: ${DATA_PATH}"
    exit 1
fi

# ============================================================================
# KILL ZOMBIES AND VERIFY NO ORPHANS
# ============================================================================
# ============================================================================
# DEL A: HARD CLEANUP FØR HVER RUN (OBLIGATORISK)
# ============================================================================
# CRITICAL: Kill all stale multiprocessing processes to prevent IPC deadlock
echo "[PREFLIGHT] Hard cleanup: Killing all stale multiprocessing processes..."
# First try graceful termination
pkill -u "$USER" -f "multiprocessing.spawn" 2>/dev/null || true
pkill -u "$USER" -f "spawn_main" 2>/dev/null || true
pkill -u "$USER" -f "resource_tracker" 2>/dev/null || true
# Canonical TRUTH: run_truth_e2e_sanity only (replay_eval_gated_parallel removed, ghost purge)
pkill -u "$USER" -f "run_truth_e2e_sanity" 2>/dev/null || true
sleep 2  # Give processes time to die

# Force kill any remaining processes
pkill -9 -u "$USER" -f "spawn_main" 2>/dev/null || true
pkill -9 -u "$USER" -f "resource_tracker" 2>/dev/null || true
pkill -9 -u "$USER" -f "multiprocessing.spawn" 2>/dev/null || true
sleep 1  # Give processes time to die

# HARD-FAIL if any stale processes still exist
if pgrep -u "$USER" -f "spawn_main" >/dev/null; then
    echo "[FATAL] Stale spawn_main processes detected after cleanup. Cannot start replay safely." >&2
    echo "[FATAL] PIDs:" >&2
    pgrep -u "$USER" -f "spawn_main" | head -10 >&2
    exit 1
fi

if pgrep -u "$USER" -f "resource_tracker" >/dev/null; then
    echo "[FATAL] Stale resource_tracker processes detected after cleanup. Cannot start replay safely." >&2
    echo "[FATAL] PIDs:" >&2
    pgrep -u "$USER" -f "resource_tracker" | head -10 >&2
    exit 1
fi

echo "[PREFLIGHT] ✅ Multiprocessing environment clean"

# Check for orphaned replay processes
# CRITICAL: Write orphan PIDs to temp file to avoid grep matching tee output
echo "[CLEANUP] Checking for orphaned replay processes..."
ORPHAN_TMP="/tmp/gx1_orphan_pids_$$.txt"
ps -eo pid,cmd --no-headers 2>/dev/null | grep -E "python.*run_truth_e2e_sanity" | grep -v grep | awk '{print $1}' > "${ORPHAN_TMP}" 2>/dev/null || true
if [[ -s "${ORPHAN_TMP}" ]]; then
    ORPHAN_COUNT=$(wc -l < "${ORPHAN_TMP}")
    echo "[FORENSICS] ⚠️  Found ${ORPHAN_COUNT} orphaned replay processes"
    while read -r pid; do
        if [[ -n "${pid}" ]]; then
            echo "[FORENSICS] Killing orphan PID ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done < "${ORPHAN_TMP}"
    sleep 2
    while read -r pid; do
        if [[ -n "${pid}" ]]; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done < "${ORPHAN_TMP}"
    echo "[FORENSICS] Orphan cleanup complete"
fi
rm -f "${ORPHAN_TMP}"

sleep 1
echo "Zombie cleanup complete."
echo ""

# ============================================================================
# GLOBAL LOCK
# ============================================================================
LOCK_FILE="/tmp/gx1_baseline_eval_maxcpu.lock"

if [[ -f "${LOCK_FILE}" ]]; then
    LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
    if [[ -n "${LOCK_PID}" ]] && ps -p "${LOCK_PID}" > /dev/null 2>&1; then
        echo "❌ ERROR: Another baseline eval is running (PID: ${LOCK_PID})"
        exit 1
    else
        echo "⚠️  Removing stale lock file..."
        rm -f "${LOCK_FILE}"
    fi
fi

echo $$ > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT
echo "✅ Global lock acquired (PID: $$)"
echo ""

# ============================================================================
# SET TRUTH MODE ENVIRONMENT
# ============================================================================
export GX1_RUN_MODE="TRUTH"
export GX1_REPLAY_USE_PREBUILT_FEATURES="1"
export GX1_REQUIRE_ENTRY_TELEMETRY="1"
export GX1_GATED_FUSION_ENABLED="1"
export GX1_FAULTHANDLER_OUTPUT="${OUTPUT_DIR}/MASTER_FAULT_DUMP.txt"

# ============================================================================
# BUILD REPLAY COMMAND
# ============================================================================
cd "${WORKSPACE_ROOT}"

# Canonical TRUTH entrypoint is run_truth_e2e_sanity (docs/GHOST_PURGE_PLAN.md).
# Legacy script below references removed replay_eval_gated_parallel.py; use run_truth_e2e_sanity for new runs.
REPLAY_CMD=(
    "${PYTHON_CMD}"
    -m gx1.scripts.run_truth_e2e_sanity
    --start-ts "${START_TS:-2025-01-01}" --end-ts "${END_TS:-2025-12-31}"
    --run-dir "${OUTPUT_DIR}"
)

if [[ -n "$START_TS" ]]; then
    REPLAY_CMD+=(--start-ts "${START_TS}")
fi
if [[ -n "$DAYS" ]]; then
    REPLAY_CMD+=(--days "${DAYS}")
fi

MASTER_CMDLINE="${REPLAY_CMD[*]}"

echo "Command: ${MASTER_CMDLINE}"
echo ""

# ============================================================================
# SET SIGNAL TRAPS (MUST BE SET BEFORE REPLAY STARTS)
# ============================================================================
# CRITICAL: Set traps for SIGINT/SIGTERM to ensure completion contract is ALWAYS written
# These traps will fire if wrapper receives Ctrl+C, SSH drop, systemd stop, etc.
trap 'write_contract_and_cleanup SIGINT; exit 130' INT
trap 'write_contract_and_cleanup SIGTERM; exit 143' TERM

# Notify that traps are active
if [[ -c /dev/tty ]] 2>/dev/null; then
    echo "✅ Signal traps active (SIGINT/SIGTERM). Logs: ${WRAPPER_LOG}" > /dev/tty 2>/dev/null || true
fi
echo "[WRAPPER] Signal traps active: SIGINT (exit 130), SIGTERM (exit 143)"
echo "[WRAPPER] Completion contract will be written on any signal"
echo ""

# ============================================================================
# START REPLAY AS CHILD PROCESS (NOT IN BACKGROUND - WRAPPER WAITS)
# ============================================================================
echo "Starting replay with master death supervision..."
echo ""

# CRITICAL: Start replay as foreground child (not background with &)
# Wrapper will wait for it and write completion contract
if [[ ${DEBUG_MASTER_DEATH} -eq 1 ]]; then
    echo "[DEBUG_MASTER_DEATH] Starting with strace..."
    strace -ff -tt -T -s 256 -o "${OUTPUT_DIR}/strace/trace" \
        "${REPLAY_CMD[@]}" > "${OUTPUT_DIR}/replay.log" 2>&1 &
    REPLAY_PID=$!
else
    "${REPLAY_CMD[@]}" > "${OUTPUT_DIR}/replay.log" 2>&1 &
    REPLAY_PID=$!
fi

echo "Replay started with PID=${REPLAY_PID}"

# Wait for MASTER_START signature
echo "Waiting for MASTER_START signature..."
MASTER_START_FOUND=0
for i in {1..15}; do
    if grep -q "\[MASTER_START\]" "${OUTPUT_DIR}/replay.log" 2>/dev/null; then
        MASTER_START_FOUND=1
        echo "✅ MASTER_START signature found"
        break
    fi
    sleep 1
done

if [[ ${MASTER_START_FOUND} -eq 0 ]]; then
    echo "❌ FATAL: MASTER_START signature not found after 15 seconds"
    write_death_forensics 1 "" "MASTER_START_TIMEOUT"
    exit 1
fi

# ============================================================================
# WAIT FOR REPLAY TO COMPLETE WITH HEARTBEAT MONITORING
# ============================================================================
echo ""
echo "Waiting for replay to complete (PID=${REPLAY_PID})..."
echo "[HEARTBEAT] Monitoring replay for stuck detection (CPU + log activity)"
echo ""

# Heartbeat monitoring: Detect if replay hangs (low CPU OR no progress)
HEARTBEAT_INTERVAL=10  # Check every 10 seconds
HEARTBEAT_TIMEOUT=20   # 20 seconds of no activity = stuck
LAST_LOG_MTIME=$(stat -c %Y "${OUTPUT_DIR}/replay.log" 2>/dev/null || echo "0")
LAST_HEARTBEAT_CHECK=$(date +%s)
STUCK_START_TIME=""
LAST_CHUNKS_COMPLETE=0
LAST_CHUNKS_CHECK_TIME=$(date +%s)

# CRITICAL: Wait for replay process with heartbeat monitoring
# Wrapper MUST survive even if stdout/stderr disappear (SIGPIPE handled)
set +e  # Temporarily disable errexit for wait

# Start background heartbeat monitor
(
    while ps -p ${REPLAY_PID} > /dev/null 2>&1; do
        sleep ${HEARTBEAT_INTERVAL}
        
        # Check if replay process still exists
        if ! ps -p ${REPLAY_PID} > /dev/null 2>&1; then
            break
        fi
        
        # Check CPU usage
        CPU_USAGE=$(ps -p ${REPLAY_PID} -o %cpu --no-headers 2>/dev/null | tr -d ' ' || echo "0")
        CPU_USAGE_INT=${CPU_USAGE%.*}  # Remove decimal
        
        # Check log file modification time
        CURRENT_LOG_MTIME=$(stat -c %Y "${OUTPUT_DIR}/replay.log" 2>/dev/null || echo "0")
        LOG_AGE=$(( $(date +%s) - CURRENT_LOG_MTIME ))
        
        # Check chunks progress
        CURRENT_CHUNKS_COMPLETE=$(ls "${OUTPUT_DIR}"/chunk_*/chunk_footer.json 2>/dev/null | wc -l || echo "0")
        CURRENT_TIME=$(date +%s)
        CHUNKS_STAGNANT=0
        if [[ ${CURRENT_CHUNKS_COMPLETE} -eq ${LAST_CHUNKS_COMPLETE} ]]; then
            CHUNKS_STAGNANT=$((CURRENT_TIME - LAST_CHUNKS_CHECK_TIME))
        else
            LAST_CHUNKS_COMPLETE=${CURRENT_CHUNKS_COMPLETE}
            LAST_CHUNKS_CHECK_TIME=${CURRENT_TIME}
        fi
        
        # Check if stuck: (CPU < 1% AND log age > timeout) OR (chunks stagnant > timeout)
        # CRITICAL: Chunks stagnant should trigger timeout regardless of CPU
        IS_STUCK=0
        if [[ ${CHUNKS_STAGNANT} -gt ${HEARTBEAT_TIMEOUT} ]] && [[ ${CURRENT_CHUNKS_COMPLETE} -lt 12 ]]; then
            IS_STUCK=1
        elif [[ ${CPU_USAGE_INT} -lt 1 ]] && [[ ${LOG_AGE} -gt ${HEARTBEAT_TIMEOUT} ]]; then
            IS_STUCK=1
        fi
        
        if [[ ${IS_STUCK} -eq 1 ]]; then
            if [[ -z "${STUCK_START_TIME}" ]]; then
                STUCK_START_TIME=$(date +%s)
                if [[ ${CHUNKS_STAGNANT} -gt ${HEARTBEAT_TIMEOUT} ]]; then
                    echo "[HEARTBEAT] ⚠️  WARNING: Replay appears stuck (chunks=${CURRENT_CHUNKS_COMPLETE}/12 stagnant for ${CHUNKS_STAGNANT}s, CPU=${CPU_USAGE}%)"
                else
                    echo "[HEARTBEAT] ⚠️  WARNING: Replay appears stuck (CPU=${CPU_USAGE}%, log age=${LOG_AGE}s)"
                fi
            else
                STUCK_DURATION=$(( $(date +%s) - STUCK_START_TIME ))
                if [[ ${STUCK_DURATION} -gt ${HEARTBEAT_TIMEOUT} ]]; then
                    echo "[HEARTBEAT] ❌ FATAL: Replay stuck for ${STUCK_DURATION}s (CPU=${CPU_USAGE}%, log age=${LOG_AGE}s, chunks=${CURRENT_CHUNKS_COMPLETE}/12 stagnant ${CHUNKS_STAGNANT}s)"
                    echo "[HEARTBEAT] Triggering hang state dump (SIGUSR2)..."
                    
                    # Trigger hang state dump before terminating
                    kill -USR2 ${REPLAY_PID} 2>/dev/null || true
                    sleep 2  # Give replay time to write hang dump
                    
                    echo "[HEARTBEAT] Writing completion contract and terminating replay..."
                    
                    # Write completion contract
                    write_death_forensics 1 "STUCK" "REPLAY_HUNG_NO_ACTIVITY"
                    
                    # Terminate replay
                    kill -TERM ${REPLAY_PID} 2>/dev/null || true
                    sleep 3
                    kill -KILL ${REPLAY_PID} 2>/dev/null || true
                    
                    exit 1
                fi
            fi
            # Log heartbeat status
            echo "[HEARTBEAT] Replay alive: CPU=${CPU_USAGE}%, log age=${LOG_AGE}s, chunks=${CURRENT_CHUNKS_COMPLETE}/12"
        else
            # Not stuck - reset stuck timer
            if [[ -n "${STUCK_START_TIME}" ]]; then
                echo "[HEARTBEAT] ✅ Replay activity detected (CPU=${CPU_USAGE}%, log age=${LOG_AGE}s, chunks=${CURRENT_CHUNKS_COMPLETE}/12)"
                STUCK_START_TIME=""
            fi
            # Log heartbeat status
            echo "[HEARTBEAT] Replay alive: CPU=${CPU_USAGE}%, log age=${LOG_AGE}s, chunks=${CURRENT_CHUNKS_COMPLETE}/12"
        fi
    done
) &
HEARTBEAT_PID=$!

# Wait for replay process
wait ${REPLAY_PID}
REPLAY_EXIT_CODE=$?

# Kill heartbeat monitor
kill ${HEARTBEAT_PID} 2>/dev/null || true
wait ${HEARTBEAT_PID} 2>/dev/null || true

set -e  # Re-enable errexit

END_TS_EPOCH=$(date +%s)
DURATION=$((END_TS_EPOCH - START_TS_EPOCH))

echo ""
echo "=================================================================================="
echo "REPLAY FINISHED"
echo "=================================================================================="
echo "Exit code: ${REPLAY_EXIT_CODE}"
echo "Duration: ${DURATION} seconds"

# Decode signal if exit code >= 128
SIGNAL_NAME=$(decode_signal ${REPLAY_EXIT_CODE})
if [[ -n "${SIGNAL_NAME}" ]]; then
    echo "Signal: ${SIGNAL_NAME} (${REPLAY_EXIT_CODE} - 128 = $((REPLAY_EXIT_CODE - 128)))"
fi

# ============================================================================
# WRITE COMPLETION CONTRACT (WRAPPER IS ALWAYS RESPONSIBLE)
# ============================================================================
echo ""
echo "Writing completion contract..."

# Check if Python wrote completion contract
if [[ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ]]; then
    echo "✅ RUN_COMPLETED.json found (written by Python)"
    CHUNKS_COMPLETED=$("${PYTHON_CMD}" -c "import json; d=json.load(open('${OUTPUT_DIR}/RUN_COMPLETED.json')); print(d.get('chunks_completed', 0))" 2>/dev/null || echo "0")
    CHUNKS_SUBMITTED=$("${PYTHON_CMD}" -c "import json; d=json.load(open('${OUTPUT_DIR}/RUN_COMPLETED.json')); print(d.get('chunks_submitted', 0))" 2>/dev/null || echo "0")
    echo "  Chunks completed: ${CHUNKS_COMPLETED}/${CHUNKS_SUBMITTED}"
    
    # Wrapper confirms Python's completion contract
    echo "✅ Completion contract confirmed: RUN_COMPLETED.json"
elif [[ -f "${OUTPUT_DIR}/RUN_FAILED.json" ]]; then
    echo "⚠️  RUN_FAILED.json found (written by Python)"
    REASON=$("${PYTHON_CMD}" -c "import json; d=json.load(open('${OUTPUT_DIR}/RUN_FAILED.json')); print(d.get('reason', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    echo "  Reason: ${REASON}"
    
    # Wrapper confirms Python's failure contract
    echo "✅ Completion contract confirmed: RUN_FAILED.json"
else
    # NO completion contract from Python - wrapper MUST write it
    echo "❌ NO completion contract found - master died without writing"
    echo ""
    echo "[FORENSICS] Writing completion contract from wrapper..."
    
    if [[ ${REPLAY_EXIT_CODE} -eq 0 ]]; then
        DEATH_REASON="SILENT_EXIT_NO_CONTRACT"
    elif [[ -n "${SIGNAL_NAME}" ]]; then
        DEATH_REASON="KILLED_BY_${SIGNAL_NAME}"
    else
        DEATH_REASON="EXIT_CODE_${REPLAY_EXIT_CODE}"
    fi
    
    write_death_forensics ${REPLAY_EXIT_CODE} "${SIGNAL_NAME}" "${DEATH_REASON}"
    echo "✅ Completion contract written: RUN_FAILED.json"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=================================================================================="
echo "BASELINE EVALUATION COMPLETE"
echo "=================================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Wrapper log: ${WRAPPER_LOG}"
echo "Replay exit code: ${REPLAY_EXIT_CODE}"
echo "Duration: ${DURATION} seconds"
echo ""

# List forensics files
echo "Forensics files:"
for f in RUN_COMPLETED.json RUN_FAILED.json MASTER_FAULT_DUMP.txt DEATH_LOG_TAIL.txt CHUNK_STATUS_TABLE.txt MASTER_MEMORY_LOG.jsonl wrapper.log; do
    if [[ -f "${OUTPUT_DIR}/${f}" ]]; then
        echo "  ✅ ${f}"
    fi
done

if [[ -d "${OUTPUT_DIR}/strace" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}/strace" 2>/dev/null)" ]]; then
    echo "  ✅ strace/ ($(ls "${OUTPUT_DIR}/strace" | wc -l) files)"
fi

echo ""
echo "Wrapper log location: ${WRAPPER_LOG}"
echo ""

# Print final status to console if /dev/tty is available (non-blocking)
if [[ -c /dev/tty ]] 2>/dev/null; then
    if [[ ${REPLAY_EXIT_CODE} -eq 0 ]]; then
        echo "✅ Run complete. Logs: ${WRAPPER_LOG}" > /dev/tty 2>/dev/null || true
    else
        echo "❌ Run failed (exit=${REPLAY_EXIT_CODE}). Logs: ${WRAPPER_LOG}" > /dev/tty 2>/dev/null || true
    fi
fi

# Exit with replay exit code
exit ${REPLAY_EXIT_CODE}
