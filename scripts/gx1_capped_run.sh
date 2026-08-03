#!/usr/bin/env bash
# Run one heavy GX1 job under a verified hard cgroup ceiling. A job that reaches
# its cap fails inside its own cgroup instead of consuming the workstation.
# GX1_RULES.md and AGENTS.md require this wrapper for every heavy operation.
#
# Usage:  scripts/gx1_capped_run.sh [--mem 10G] [--swap 512M] -- <command ...>
#   --mem   MemoryMax (hard) + MemoryHigh for the job's cgroup scope. This machine's
#           immutable safety ceiling is 10G; larger requests are rejected before launch.
#   --swap  MemorySwapMax. The immutable safety ceiling is 512M; swap storms are forbidden.
# The runner also requires >=20G currently available RAM, serializes heavy jobs, binds the
# job to two CPU cores, lowers its CPU/I/O priority, and constrains common numerical
# libraries to one thread. The scope self-check proves the memory/pids limits before the
# target command is entered. If the cgroup cannot be created or verified, the job fails
# closed and is never launched.
set -euo pipefail
MEM=10G ; SWAP=512M

SAFE_JOB_MEMORY_KIB=$((10 * 1024 * 1024))
SAFE_JOB_SWAP_KIB=$((512 * 1024))
MIN_HOST_MEMORY_KIB=$((30 * 1024 * 1024))
MIN_AVAILABLE_MEMORY_KIB=$((20 * 1024 * 1024))
WSL_GUARD_MIN_REQUEST_KIB=$((4 * 1024 * 1024))
WSL_CONFIG_TOLERANCE_KIB=$((1024 * 1024))
TASKS_MAX=64
CPU_AFFINITY=0-1

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
  echo "FATAL: host currently has less than 20G available RAM; refusing heavy job" >&2
  exit 75
fi
if (( requested_mem_kib > SAFE_JOB_MEMORY_KIB )); then
  echo "FATAL: requested MemoryMax exceeds GX1 safety ceiling (10G)" >&2
  exit 75
fi
if (( requested_swap_kib > SAFE_JOB_SWAP_KIB )); then
  echo "FATAL: requested MemorySwapMax exceeds GX1 safety ceiling (512M)" >&2
  exit 75
fi
if (( requested_mem_kib > WSL_GUARD_MIN_REQUEST_KIB )) \
  && grep -qi microsoft /proc/version; then
  WSL_CONFIG_PATH=/mnt/c/Users/Andre/.wslconfig
  [[ -r "$WSL_CONFIG_PATH" ]] || {
    echo "FATAL: WSL safety config is unavailable; refusing model/dataset job" >&2
    exit 75
  }
  wsl_memory_raw=$(awk -F= '$1 == "memory" {gsub(/[[:space:]]/, "", $2); print toupper($2); exit}' "$WSL_CONFIG_PATH")
  wsl_swap_raw=$(awk -F= '$1 == "swap" {gsub(/[[:space:]]/, "", $2); print toupper($2); exit}' "$WSL_CONFIG_PATH")
  [[ "$wsl_memory_raw" =~ ^([1-9][0-9]*)(MB|GB)$ ]] || {
    echo "FATAL: WSL memory= setting is invalid; refusing model/dataset job" >&2
    exit 75
  }
  wsl_memory_number=${BASH_REMATCH[1]}
  wsl_memory_unit=${BASH_REMATCH[2]}
  [[ "$wsl_swap_raw" =~ ^([1-9][0-9]*)(MB|GB)$ ]] || {
    echo "FATAL: WSL swap= setting is invalid; refusing model/dataset job" >&2
    exit 75
  }
  wsl_swap_number=${BASH_REMATCH[1]}
  wsl_swap_unit=${BASH_REMATCH[2]}
  if [[ "$wsl_memory_unit" == "GB" ]]; then
    wsl_memory_kib=$((wsl_memory_number * 1024 * 1024))
  else
    wsl_memory_kib=$((wsl_memory_number * 1024))
  fi
  if [[ "$wsl_swap_unit" == "GB" ]]; then
    wsl_swap_kib=$((wsl_swap_number * 1024 * 1024))
  else
    wsl_swap_kib=$((wsl_swap_number * 1024))
  fi
  host_swap_kib=$(awk '/^SwapTotal:/ {print $2; exit}' /proc/meminfo)
  if (( host_total_kib > wsl_memory_kib + WSL_CONFIG_TOLERANCE_KIB )); then
    echo "FATAL: active WSL MemTotal exceeds configured memory cap; restart WSL before a model/dataset job" >&2
    exit 75
  fi
  if (( host_swap_kib > wsl_swap_kib + WSL_CONFIG_TOLERANCE_KIB )); then
    echo "FATAL: active WSL SwapTotal exceeds configured swap cap; restart WSL before a model/dataset job" >&2
    exit 75
  fi
fi
for helper in /usr/bin/taskset /usr/bin/ionice /usr/bin/nice /bin/bash; do
  [[ -x "$helper" ]] || { echo "FATAL: required capacity helper is missing: $helper" >&2; exit 75; }
done

if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "$XDG_RUNTIME_DIR" && -w "$XDG_RUNTIME_DIR" ]]; then
  LOCK_PATH="$XDG_RUNTIME_DIR/gx1-heavy-job.lock"
else
  LOCK_PATH="/tmp/gx1-heavy-job-$(id -u).lock"
fi
[[ -d ${LOCK_PATH%/*} && -w ${LOCK_PATH%/*} ]] || { echo "FATAL: heavy-job lock directory is unavailable: ${LOCK_PATH%/*}"; exit 2; }
exec 9>>"$LOCK_PATH"
if ! flock -n 9; then
  echo "FATAL: another GX1 heavy job owns the exclusive lock: $LOCK_PATH" >&2
  exit 75
fi
echo "[capped_run] MemoryMax=$MEM MemoryHigh=$MEM MemorySwapMax=$SWAP CPUAffinity=$CPU_AFFINITY TasksMax=$TASKS_MAX"
echo "[capped_run] cmd: $*"

# systemd can accept CPUQuota/IOWeight properties even when the delegated cgroup
# controllers are absent. Do not describe those properties as protection. The
# affinity and priority wrappers below remain effective at process level, while
# the scope guard proves the hard memory/pids controls that prevent a job from
# consuming the whole WSL VM.
SCOPE_GUARD='
set -euo pipefail
cg_rel=$(awk -F: '\''$1 == "0" {print $3}'\'' /proc/self/cgroup)
cg_dir="/sys/fs/cgroup${cg_rel}"
[[ -d "$cg_dir" ]] || { echo "FATAL: scope cgroup path is unavailable: $cg_dir" >&2; exit 75; }
[[ "$(cat "$cg_dir/memory.max")" == "$GX1_EXPECTED_MEMORY_BYTES" ]] || { echo "FATAL: memory.max scope proof failed" >&2; exit 75; }
[[ "$(cat "$cg_dir/memory.high")" == "$GX1_EXPECTED_MEMORY_BYTES" ]] || { echo "FATAL: memory.high scope proof failed" >&2; exit 75; }
[[ "$(cat "$cg_dir/memory.swap.max")" == "$GX1_EXPECTED_SWAP_BYTES" ]] || { echo "FATAL: memory.swap.max scope proof failed" >&2; exit 75; }
[[ "$(cat "$cg_dir/pids.max")" == "$GX1_EXPECTED_TASKS" ]] || { echo "FATAL: pids.max scope proof failed" >&2; exit 75; }
echo "[capped_run_scope_verified] memory.max=$GX1_EXPECTED_MEMORY_BYTES memory.high=$GX1_EXPECTED_MEMORY_BYTES memory.swap.max=$GX1_EXPECTED_SWAP_BYTES pids.max=$GX1_EXPECTED_TASKS"
verified_cpu_affinity="$GX1_CPU_AFFINITY"
unset GX1_EXPECTED_MEMORY_BYTES GX1_EXPECTED_SWAP_BYTES GX1_EXPECTED_TASKS GX1_CPU_AFFINITY
exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 "$@"
'
systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP" \
  -p TasksMax="$TASKS_MAX" -p KillMode=control-group \
  --setenv=GX1_EXPECTED_MEMORY_BYTES="$((requested_mem_kib * 1024))" \
  --setenv=GX1_EXPECTED_SWAP_BYTES="$((requested_swap_kib * 1024))" \
  --setenv=GX1_EXPECTED_TASKS="$TASKS_MAX" \
  --setenv=GX1_CPU_AFFINITY="$CPU_AFFINITY" \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=VECLIB_MAXIMUM_THREADS=1 \
  --setenv=BLIS_NUM_THREADS=1 \
  --setenv=ARROW_NUM_THREADS=1 \
  --setenv=POLARS_MAX_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 \
  -- /bin/bash -c "$SCOPE_GUARD" gx1-capped-scope "$@"
exit_code=$?
flock -u 9
exit "$exit_code"
