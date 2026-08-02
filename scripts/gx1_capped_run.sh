#!/usr/bin/env bash
# gx1_capped_run.sh — run a heavy job under a HARD cgroup/resource ceiling so that an OOM
# kills the JOB (cgroup OOM-killer), NEVER freezes/crashes the whole machine.
#
# BIRTHED 2026-06-17: the FULL phase6 entry-gate (peaks ~56G on a 58G WSL cap) OOM-crashed the
# PC mid-run (froze → hard reboot → live runner left down → open paper trades unmanaged ~3h).
# CLAUDE.md/AGENTS.md make RAM-headroom a HARD ceiling ("an OOM CRASHED the PC"); this enforces it
# by construction instead of "remember to watch RAM". Use for EVERY heavy job (gate / build / replay).
#
# Usage:  scripts/gx1_capped_run.sh [--mem 14G] [--swap 1G] -- <command ...>
#   --mem   MemoryMax (hard) + MemoryHigh for the job's cgroup scope. This machine's
#           immutable safety ceiling is 14G; larger requests are rejected before launch.
#   --swap  MemorySwapMax. The immutable safety ceiling is 1G; swap storms are forbidden.
# The runner also requires >=16G currently available RAM, serializes heavy jobs, limits the
# job to two CPU cores, and constrains common numerical libraries to one thread. If the
# systemd cgroup cannot be created, the job fails closed and is never launched.
set -euo pipefail
MEM=14G ; SWAP=1G

SAFE_JOB_MEMORY_KIB=$((14 * 1024 * 1024))
SAFE_JOB_SWAP_KIB=$((1 * 1024 * 1024))
MIN_HOST_MEMORY_KIB=$((42 * 1024 * 1024))
MIN_AVAILABLE_MEMORY_KIB=$((16 * 1024 * 1024))

size_to_kib() {
  local value="$1" number unit multiplier
  if [[ ! "$value" =~ ^([1-9][0-9]*)(K|M|G|T)$ ]]; then
    echo "FATAL: size must be a positive integer K/M/G/T value: $value" >&2
    exit 2
  fi
  number="${BASH_REMATCH[1]}"
  unit="${BASH_REMATCH[2]}"
  case "$unit" in
    K) multiplier=1 ;;
    M) multiplier=1024 ;;
    G) multiplier=$((1024 * 1024)) ;;
    T) multiplier=$((1024 * 1024 * 1024)) ;;
  esac
  echo $((number * multiplier))
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem)  MEM="$2";  shift 2 ;;
    --swap) SWAP="$2"; shift 2 ;;
    --)     shift; break ;;
    *) echo "FATAL: unknown arg '$1' (put the command after '--')"; exit 2 ;;
  esac
done
[[ $# -ge 1 ]] || { echo "FATAL: no command given after '--'"; exit 2; }

requested_mem_kib=$(size_to_kib "$MEM")
requested_swap_kib=$(size_to_kib "$SWAP")
host_total_kib=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)
host_available_kib=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
if [[ -z "$host_total_kib" || -z "$host_available_kib" ]]; then
  echo "FATAL: host memory state is unavailable; refusing heavy job" >&2
  exit 75
fi
if (( host_total_kib < MIN_HOST_MEMORY_KIB )); then
  echo "FATAL: host has less than the GX1 safety floor (${MIN_HOST_MEMORY_KIB}KiB)" >&2
  exit 75
fi
if (( host_available_kib < MIN_AVAILABLE_MEMORY_KIB )); then
  echo "FATAL: host currently has less than 16G available RAM; refusing heavy job" >&2
  exit 75
fi
if (( requested_mem_kib > SAFE_JOB_MEMORY_KIB )); then
  echo "FATAL: requested MemoryMax exceeds GX1 safety ceiling (14G)" >&2
  exit 75
fi
if (( requested_swap_kib > SAFE_JOB_SWAP_KIB )); then
  echo "FATAL: requested MemorySwapMax exceeds GX1 safety ceiling (1G)" >&2
  exit 75
fi

LOCK_PATH=/run/user/$(id -u)/gx1-heavy-job.lock
[[ -d ${LOCK_PATH%/*} ]] || { echo "FATAL: heavy-job lock directory missing: ${LOCK_PATH%/*}"; exit 2; }
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  echo "FATAL: another GX1 heavy job owns the exclusive lock: $LOCK_PATH" >&2
  exit 75
fi
echo "[capped_run] MemoryMax=$MEM MemoryHigh=$MEM MemorySwapMax=$SWAP CPUQuota=200% TasksMax=256"
echo "[capped_run] cmd: $*"
systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP" \
  -p CPUQuota=200% -p TasksMax=256 -p KillMode=control-group \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=VECLIB_MAXIMUM_THREADS=1 \
  --setenv=BLIS_NUM_THREADS=1 \
  --setenv=ARROW_NUM_THREADS=1 \
  --setenv=POLARS_MAX_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 \
  -- "$@"
exit_code=$?
flock -u 9
exit "$exit_code"
