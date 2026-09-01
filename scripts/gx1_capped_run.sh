#!/usr/bin/env bash
# Run one heavy GX1 job under a verified hard cgroup ceiling. A job that reaches
# its cap fails inside its own cgroup instead of consuming the workstation.
# GX1_RULES.md and AGENTS.md require this wrapper for every heavy operation.
#
# Usage: scripts/gx1_capped_run.sh --class audit|producer|trainer [--mem 4G] [--swap 512M]
#          [--attended-smoke|--cuda-producer] -- <command ...>
#   --class audit is capped at 4G and cannot launch the trainer.
#   --class producer is for the heavy offline dataset producers (feature lanes,
#           model source, ranker, dataset rebuild) and may request at most 20G.
#           It cannot launch the trainer.
#   --class trainer is reserved for the one canonical trainer module and may
#           request at most 20G.
#   --mem   MemoryMax (hard) + MemoryHigh for the job's cgroup scope. This machine's
#           immutable safety ceiling is 20G (the same figure already used below as
#           MIN_AVAILABLE_MEMORY_KIB, the pre-launch available-RAM gate); larger
#           requests are rejected before launch. Raised from 10G on 2026-08-09 after
#           real batch=640 candidate-training measurement showed a 640-row batch's
#           pre-step host RSS baseline alone is ~10.1G, leaving no headroom under the
#           old ceiling; host has 31G total, so 20G leaves 11G for everything else.
#   --swap  MemorySwapMax. The immutable safety ceiling is 512M; swap storms are forbidden.
# The runner also requires >=20G currently available RAM, serializes heavy jobs, binds the
# job to an explicit CPU set, lowers its CPU/I/O priority, and constrains common
# numerical libraries. Trainer jobs and the one allow-listed CUDA inference
# producer additionally pass through the fail-closed wall-clock/GPU guard below;
# an unavailable signed host sensor, excessive configured power limit, thermal
# breach, or wall-clock expiry terminates the whole process group.
# `--attended-smoke` is a deliberately narrower operator-present exception for
# one CUDA smoke only. It neither creates candidate authority nor relaxes CPU,
# memory, pids or actual-power protection; see the fixed policy below.
# The scope self-check proves the memory/pids limits before the target command is
# entered. If any protection cannot be created or verified, the job fails closed.
set -euo pipefail
JOB_CLASS="" ; MEM=4G ; SWAP=512M ; ATTENDED_SMOKE=false

CANONICAL_TRAINER_MODULE=gx1.models.entry_v10.entry_v10_ctx_train_v3
ATTENDED_HARDWARE_SMOKE_MODULE=gx1.scripts.attended_model_native_hardware_smoke_v1
# CUDA producer routes are intentionally enumerated rather than accepting an
# arbitrary module.  Both are read-only, TRAIN/VAL-only evidence producers;
# neither has an execution, promotion or TEST surface.
CUDA_PRODUCER_MODULE=gx1.scripts.evaluate_entry_candidate_selective_edge_v1
TECHNICAL_VALIDATION_PRODUCER_MODULE=gx1.scripts.validate_entry_model_native_technical_checkpoint_v1
RUNNER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$RUNNER_PATH")/.." && pwd -P)"
CANONICAL_TRAINER_PYTHON="$REPO_ROOT/.venv/bin/python"
GPU_GUARD_PATH="$REPO_ROOT/scripts/gx1_guarded_trainer_exec.sh"

# Crash-response safety freeze (2026-08-23). These are source-bound constants,
# not caller-controlled defaults. V9 follows the V8 physical-host incident:
# canonical CUDA may run only with the signed Windows memory-junction bridge,
# 160 W physical limit, a separate actual-draw stop, and a narrow CPU set. Any
# later change requires a reviewed commit and a fresh source-bound recipe.
TRAINER_EXECUTION_MODE=canonical
TRAINER_MAX_WALL_SECONDS=7200
TRAINER_MODEL_MAX_WALL_SECONDS=7200
TRAINER_ATTENDED_STAGE_REQUIRED=false
TRAINER_GPU_INDEX=0
TRAINER_GPU_MAX_CORE_TEMP_C=65
TRAINER_GPU_MAX_MEMORY_TEMP_C=80
TRAINER_GPU_MAX_POWER_LIMIT_W=160
TRAINER_GPU_MAX_POWER_DRAW_W=170
TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
TRAINER_HOST_TELEMETRY_QUERY_PATH="$REPO_ROOT/scripts/gx1_host_telemetry_bridge_query.sh"
TRAINER_HOST_TELEMETRY_URL='http://172.30.224.1:38128/gx1/v1/telemetry/'
TRAINER_HOST_TELEMETRY_CERT_PATH='/mnt/c/ProgramData/GX1/HostTelemetryBridgeV4/GX1HostTelemetryBridgePublic.pem'
TRAINER_HOST_TELEMETRY_CERT_SHA256='25c9260c2168db53cf58c5f963f2008d5163d80aa69699c5726e0680ed74eb6e'
TRAINER_HOST_TELEMETRY_GPU_UUID='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS=2
TRAINER_DEVICE=
TRAINER_OUT_BUNDLE_DIR=
CUDA_PRODUCER_GUARD=false

SAFE_JOB_MEMORY_KIB=$((20 * 1024 * 1024))
SAFE_AUDIT_MEMORY_KIB=$((4 * 1024 * 1024))
SAFE_JOB_SWAP_KIB=$((512 * 1024))
MIN_HOST_MEMORY_KIB=$((30 * 1024 * 1024))
MIN_AVAILABLE_MEMORY_KIB=$((20 * 1024 * 1024))
WSL_GUARD_MIN_REQUEST_KIB=$((4 * 1024 * 1024))
WSL_CONFIG_TOLERANCE_KIB=$((1024 * 1024))
TASKS_MAX=64
# The canonical CUDA trainer needs room for PyTorch/CUDA worker threads as
# well as the one-second safety watchdog. With sixteen numerical workers, a
# 64-task cgroup can exhaust before the first batch, abort the trainer, and
# prevent the watchdog from spawning its next telemetry probe. Keep the
# tighter audit cap, while retaining a finite trainer-specific ceiling.
TRAINER_TASKS_MAX=128
# V8's 0-15 affinity left only three WSL vCPUs for host/desktop activity during
# the physical host hang. V9 permanently reserves 11 of WSL's 19 vCPUs and
# limits the candidate to 0-7 / eight numerical workers. DataLoader workers
# remain disabled by the model's fixed low-memory contract.
CPU_AFFINITY=0-7
# Audit and producer processes retain one numerical worker so the hard
# TasksMax=64 boundary continues to leave room for their own Python/Arrow
# helper threads. The canonical trainer receives the bounded eight-thread
# allowance and bounded 128-task cgroup only after its class has been
# validated below.
NUMERICAL_THREAD_COUNT=1

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

is_direct_python() {
  [[ -x "$1" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" ]]
}

validate_target_command() {
  local executable_basename="${1##*/}" target_arg module
  local trainer_reference=false hardware_smoke_reference=false
  local trainer_flag_count=0 trainer_device_count=0 hardware_smoke_flag_count=0
  local profile_count=0 execution_tier_count=0 out_bundle_dir_count=0 out_dir_count=0
  local profile_value= execution_tier_value=
  local -a target_args=("$@")

  case "$executable_basename" in
    env|bash|sh|dash|zsh|ksh)
      echo "FATAL: env and shell wrappers are forbidden as capped targets" >&2
      exit 75
      ;;
  esac

  for target_arg in "$@"; do
    if [[ "$target_arg" == *"$CANONICAL_TRAINER_MODULE"* ]]; then
      trainer_reference=true
    fi
    if [[ "$target_arg" == *"$ATTENDED_HARDWARE_SMOKE_MODULE"* ]]; then
      hardware_smoke_reference=true
    fi
    if [[ "$target_arg" == "--train" ]]; then
      trainer_flag_count=$((trainer_flag_count + 1))
    fi
  done

  if [[ "$JOB_CLASS" == "audit" || "$JOB_CLASS" == "producer" ]]; then
    if [[ "$trainer_reference" == true ]]; then
      echo "FATAL: canonical trainer requires --class trainer" >&2
      exit 75
    fi
    if [[ "$hardware_smoke_reference" == true ]]; then
      echo "FATAL: attended hardware smoke requires --class trainer" >&2
      exit 75
    fi
    if [[ "$CUDA_PRODUCER_GUARD" == true ]]; then
      [[ "$JOB_CLASS" == producer ]] || {
        echo "FATAL: --cuda-producer requires --class producer" >&2
        exit 75
      }
      if ! is_direct_python "$1" || [[ "${2:-}" != "-m" ]] \
        || { [[ "${3:-}" != "$CUDA_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TECHNICAL_VALIDATION_PRODUCER_MODULE" ]]; }; then
        echo "FATAL: --cuda-producer is reserved for an allow-listed TRAIN/VAL-only evidence producer" >&2
        exit 75
      fi
      for ((target_index = 0; target_index < ${#target_args[@]}; target_index++)); do
        case "${target_args[$target_index]}" in
          --device)
            trainer_device_count=$((trainer_device_count + 1))
            (( target_index + 1 < ${#target_args[@]} )) || {
              echo "FATAL: CUDA producer requires a value after --device" >&2
              exit 75
            }
            TRAINER_DEVICE="${target_args[$((target_index + 1))]}"
            ;;
          --device=*)
            trainer_device_count=$((trainer_device_count + 1))
            TRAINER_DEVICE="${target_args[$target_index]#--device=}"
            ;;
          --out-dir)
            out_dir_count=$((out_dir_count + 1))
            (( target_index + 1 < ${#target_args[@]} )) || {
              echo "FATAL: CUDA producer requires a value after --out-dir" >&2
              exit 75
            }
            TRAINER_OUT_BUNDLE_DIR="${target_args[$((target_index + 1))]}"
            ;;
          --out-dir=*)
            out_dir_count=$((out_dir_count + 1))
            TRAINER_OUT_BUNDLE_DIR="${target_args[$target_index]#--out-dir=}"
            ;;
        esac
      done
      if (( trainer_device_count != 1 )) || [[ "$TRAINER_DEVICE" != cuda ]]; then
        echo "FATAL: --cuda-producer requires exactly one --device cuda" >&2
        exit 75
      fi
      if (( out_dir_count != 1 )) || [[ "$TRAINER_OUT_BUNDLE_DIR" != /* ]]; then
        echo "FATAL: --cuda-producer requires one absolute --out-dir" >&2
        exit 75
      fi
      TRAINER_EXECUTION_MODE=cuda_producer
      # Full bounded VAL inference needs more than the trainer's short smoke
      # window, but is still terminated after one hour if it stalls.
      TRAINER_MAX_WALL_SECONDS=3600
      TRAINER_MODEL_MAX_WALL_SECONDS=3600
    fi
    return
  fi

  if ! is_direct_python "$1" \
    || [[ "${2:-}" != "-m" ]]; then
    echo "FATAL: trainer class is reserved for the canonical trainer module as a direct target" >&2
    exit 75
  fi
  module="${3:-}"
  [[ "$module" == "$CANONICAL_TRAINER_MODULE" \
    || "$module" == "$ATTENDED_HARDWARE_SMOKE_MODULE" ]] || {
    echo "FATAL: trainer class permits only the canonical trainer or attended hardware smoke module" >&2
    exit 75
  }
  if [[ "$module" == "$CANONICAL_TRAINER_MODULE" ]] \
    && (( trainer_flag_count != 1 )); then
    echo "FATAL: trainer class requires the canonical --train mode exactly once" >&2
    exit 75
  fi
  for ((target_index = 0; target_index < ${#target_args[@]}; target_index++)); do
    if [[ "${target_args[$target_index]}" == "--device" ]]; then
      trainer_device_count=$((trainer_device_count + 1))
      (( target_index + 1 < ${#target_args[@]} )) || {
        echo "FATAL: trainer class requires a value after --device" >&2
        exit 75
      }
      TRAINER_DEVICE="${target_args[$((target_index + 1))]}"
    fi
    if [[ "${target_args[$target_index]}" == "--attended-hardware-smoke" ]]; then
      hardware_smoke_flag_count=$((hardware_smoke_flag_count + 1))
    fi
    case "${target_args[$target_index]}" in
      --profile)
        profile_count=$((profile_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || {
          echo "FATAL: attended smoke requires a value after --profile" >&2
          exit 75
        }
        profile_value="${target_args[$((target_index + 1))]}"
        ;;
      --execution-tier)
        execution_tier_count=$((execution_tier_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || {
          echo "FATAL: attended smoke requires a value after --execution-tier" >&2
          exit 75
        }
        execution_tier_value="${target_args[$((target_index + 1))]}"
        ;;
      --out_bundle_dir)
        out_bundle_dir_count=$((out_bundle_dir_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || {
          echo "FATAL: trainer class requires a value after --out_bundle_dir" >&2
          exit 75
        }
        TRAINER_OUT_BUNDLE_DIR="${target_args[$((target_index + 1))]}"
        ;;
    esac
  done
  if (( trainer_device_count != 1 )) \
    || [[ "$TRAINER_DEVICE" != cpu && "$TRAINER_DEVICE" != cuda ]]; then
    echo "FATAL: trainer class requires exactly one canonical --device cpu|cuda" >&2
    exit 75
  fi

  if [[ "$module" == "$CANONICAL_TRAINER_MODULE" ]]; then
    if (( out_bundle_dir_count != 1 )) || [[ "$TRAINER_OUT_BUNDLE_DIR" != /* ]]; then
      echo "FATAL: trainer class requires one absolute --out_bundle_dir" >&2
      exit 75
    fi
    if [[ "$ATTENDED_SMOKE" == true ]]; then
      if [[ $profile_count -ne 1 || "$profile_value" != smoke \
        || $execution_tier_count -ne 1 ]]; then
        echo "FATAL: attended smoke is reserved for one smoke command with one attended execution tier" >&2
        exit 75
      fi
      case "$TRAINER_DEVICE:$execution_tier_value" in
        cuda:attended_only|cpu:attended_cpu_only) ;;
        *)
          echo "FATAL: attended CUDA requires --execution-tier attended_only; attended CPU requires attended_cpu_only" >&2
          exit 75
          ;;
      esac
    fi
    if [[ "$ATTENDED_SMOKE" == true ]]; then
      # Only the exact canonical trainer can advance from the complete
      # data-preflight into the separately bounded model phase. The hardware
      # smoke intentionally remains a standalone telemetry diagnostic.
      TRAINER_ATTENDED_STAGE_REQUIRED=true
    fi
    return
  fi

  # This is intentionally a separate target, not a shortcut into canonical
  # training.  It carries no --train flag, accepts CUDA only, and is admitted
  # only through the same attended guard owned by this source file.
  if [[ "$ATTENDED_SMOKE" != true || "$TRAINER_DEVICE" != cuda \
    || $trainer_flag_count -ne 0 || $hardware_smoke_flag_count -ne 1 ]]; then
    echo "FATAL: attended hardware smoke requires --attended-smoke, CUDA, no --train, and one --attended-hardware-smoke marker" >&2
    exit 75
  fi
  if (( profile_count != 0 || execution_tier_count != 0 )); then
    echo "FATAL: attended hardware smoke cannot carry trainer profile or execution-tier arguments" >&2
    exit 75
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --class)
      [[ $# -ge 2 ]] || { echo "FATAL: --class requires a value" >&2; exit 2; }
      JOB_CLASS="$2"; shift 2
      ;;
    --mem)
      [[ $# -ge 2 ]] || { echo "FATAL: --mem requires a value" >&2; exit 2; }
      MEM="$2"; shift 2
      ;;
    --swap)
      [[ $# -ge 2 ]] || { echo "FATAL: --swap requires a value" >&2; exit 2; }
      SWAP="$2"; shift 2
      ;;
    --attended-smoke)
      ATTENDED_SMOKE=true; shift
      ;;
    --cuda-producer)
      CUDA_PRODUCER_GUARD=true; shift
      ;;
    --research-smoke)
      echo "FATAL: --research-smoke is disabled after the WSL/GPU reset; use only the bounded attended hardware diagnostic" >&2
      exit 75
      ;;
    --)     shift; break ;;
    *) echo "FATAL: unknown arg '$1' (put the command after '--')"; exit 2 ;;
  esac
done
[[ $# -ge 1 ]] || { echo "FATAL: no command given after '--'"; exit 2; }
[[ "$JOB_CLASS" == "audit" || "$JOB_CLASS" == "producer" || "$JOB_CLASS" == "trainer" ]] || {
  echo "FATAL: --class must be exactly audit, producer or trainer" >&2
  exit 2
}

requested_mem_kib=$(size_to_kib "$MEM")
requested_swap_kib=$(size_to_kib "$SWAP")
if (( requested_mem_kib > SAFE_JOB_MEMORY_KIB )); then
  echo "FATAL: requested MemoryMax exceeds GX1 safety ceiling (20G)" >&2
  exit 75
fi
if [[ "$JOB_CLASS" == "audit" ]] && (( requested_mem_kib > SAFE_AUDIT_MEMORY_KIB )); then
  echo "FATAL: audit jobs may request at most 4G" >&2
  exit 75
fi
if (( requested_swap_kib > SAFE_JOB_SWAP_KIB )); then
  echo "FATAL: requested MemorySwapMax exceeds GX1 safety ceiling (512M)" >&2
  exit 75
fi
validate_target_command "$@"

if [[ "$JOB_CLASS" == trainer ]]; then
  NUMERICAL_THREAD_COUNT=8
  TASKS_MAX="$TRAINER_TASKS_MAX"
fi

if [[ "$ATTENDED_SMOKE" == true ]]; then
  [[ "$JOB_CLASS" == trainer ]] || {
    echo "FATAL: --attended-smoke requires --class trainer" >&2
    exit 75
  }
  # This is an operator-present diagnostic exception, not a second hardware
  # policy. It retains the exact V9 signed-memory telemetry and all physical,
  # cgroup, CPU-affinity, thermal, actual-power and residency stops. The former
  # 24-hour research route held nearly all VRAM under WSL and is disabled.
  if [[ "$TRAINER_DEVICE" == cpu ]]; then
    TRAINER_EXECUTION_MODE=attended_cpu_smoke
  else
    TRAINER_EXECUTION_MODE=attended_smoke
  fi
  # A real V40 attended run proved that the one former five-minute envelope
  # can expire while correctly re-hashing the immutable 6.62 GB TRAIN source,
  # before model construction. Only the exact canonical data-smoke target has
  # the source-bound 600+300 staged envelope. The independent no-data hardware diagnostic remains a single five-minute run; it must not gain time merely
  # because it shares the attended telemetry exception.
  if [[ "$TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then
    TRAINER_MAX_WALL_SECONDS=600
    TRAINER_MODEL_MAX_WALL_SECONDS=300
  else
    TRAINER_MAX_WALL_SECONDS=300
    TRAINER_MODEL_MAX_WALL_SECONDS=300
  fi
fi
if [[ "$CUDA_PRODUCER_GUARD" == true && "$ATTENDED_SMOKE" == true ]]; then
  echo "FATAL: --cuda-producer and --attended-smoke are mutually exclusive" >&2
  exit 75
fi

if [[ -n "${GX1_CAPPED_CLASS:-}" \
  || -n "${GX1_CAPPED_MEMORY_BYTES:-}" \
  || -n "${GX1_CAPPED_SWAP_BYTES:-}" \
  || -n "${GX1_CAPPED_TASKS_MAX:-}" ]]; then
  [[ "${GX1_CAPPED_CLASS:-}" == "$JOB_CLASS" ]] || {
    echo "FATAL: nested capped job class mismatch" >&2
    exit 75
  }
  [[ "${GX1_CAPPED_MEMORY_BYTES:-}" == "$((requested_mem_kib * 1024))" ]] || {
    echo "FATAL: nested capped job memory request differs from parent scope" >&2
    exit 75
  }
  [[ "${GX1_CAPPED_SWAP_BYTES:-}" == "$((requested_swap_kib * 1024))" ]] || {
    echo "FATAL: nested capped job swap request differs from parent scope" >&2
    exit 75
  }
  [[ "${GX1_CAPPED_TASKS_MAX:-}" == "$TASKS_MAX" ]] || {
    echo "FATAL: nested capped job task limit differs from parent scope" >&2
    exit 75
  }
  nested_cgroup_rel=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  nested_cgroup_dir="/sys/fs/cgroup${nested_cgroup_rel}"
  [[ -d "$nested_cgroup_dir" \
    && "$(cat "$nested_cgroup_dir/memory.max")" == "$GX1_CAPPED_MEMORY_BYTES" \
    && "$(cat "$nested_cgroup_dir/memory.high")" == "$GX1_CAPPED_MEMORY_BYTES" \
    && "$(cat "$nested_cgroup_dir/memory.swap.max")" == "$GX1_CAPPED_SWAP_BYTES" \
    && "$(cat "$nested_cgroup_dir/pids.max")" == "$GX1_CAPPED_TASKS_MAX" ]] || {
    echo "FATAL: nested capped job parent scope proof failed" >&2
    exit 75
  }
  if [[ "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ]]; then
    [[ "${GX1_TRAINER_DEVICE:-}" == "$TRAINER_DEVICE" ]] || {
      echo "FATAL: nested guarded CUDA device differs from protected parent scope" >&2
      exit 75
    }
    [[ -x "$GPU_GUARD_PATH" ]] || {
      echo "FATAL: canonical trainer safety guard is unavailable" >&2
      exit 75
    }
    exec "$GPU_GUARD_PATH" "$@"
  fi
  exec "$@"
fi
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
if [[ ( "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ) && ! -x "$GPU_GUARD_PATH" ]]; then
  echo "FATAL: guarded CUDA safety owner is unavailable: $GPU_GUARD_PATH" >&2
  exit 75
fi
if [[ ( "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ) && "$TRAINER_DEVICE" == cuda ]]; then
  [[ -x "$TRAINER_HOST_TELEMETRY_QUERY_PATH" && ! -L "$TRAINER_HOST_TELEMETRY_QUERY_PATH" ]] || {
    echo "FATAL: signed host telemetry query is unavailable: $TRAINER_HOST_TELEMETRY_QUERY_PATH" >&2
    exit 75
  }
  [[ -f "$TRAINER_HOST_TELEMETRY_CERT_PATH" && ! -L "$TRAINER_HOST_TELEMETRY_CERT_PATH" ]] || {
    echo "FATAL: signed host telemetry certificate is unavailable: $TRAINER_HOST_TELEMETRY_CERT_PATH" >&2
    exit 75
  }
  [[ "$TRAINER_HOST_TELEMETRY_CERT_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$TRAINER_HOST_TELEMETRY_GPU_UUID" =~ ^GPU-[0-9a-fA-F-]{36}$ \
    && "$TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS" =~ ^[1-5]$ ]] || {
    echo "FATAL: signed host telemetry source binding is malformed" >&2
    exit 75
  }
fi

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
echo "[capped_run] Class=$JOB_CLASS MemoryMax=$MEM MemoryHigh=$MEM MemorySwapMax=$SWAP CPUAffinity=$CPU_AFFINITY NumericalThreads=$NUMERICAL_THREAD_COUNT TasksMax=$TASKS_MAX" >&2
echo "[capped_run] cmd: $*" >&2
TRAINER_GUARD_LOG_PATH=
TRAINER_STDIO_LOG_PATH=
if [[ ( "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ) && -n "$TRAINER_OUT_BUNDLE_DIR" ]]; then
  TRAINER_GUARD_LOG_PARENT="${TRAINER_OUT_BUNDLE_DIR%/*}"
  TRAINER_GUARD_LOG_BASENAME="${TRAINER_OUT_BUNDLE_DIR##*/}"
  [[ -d "$TRAINER_GUARD_LOG_PARENT" && ! -L "$TRAINER_GUARD_LOG_PARENT" ]] || {
    echo "FATAL: trainer guard-log parent is unavailable: $TRAINER_GUARD_LOG_PARENT" >&2
    exit 75
  }
  TRAINER_GUARD_LOG_PATH=$(
    /usr/bin/mktemp "${TRAINER_GUARD_LOG_PARENT}/.${TRAINER_GUARD_LOG_BASENAME}.guard.XXXXXXXX.log"
  ) || {
    echo "FATAL: could not create exclusive trainer guard log" >&2
    exit 75
  }
  TRAINER_STDIO_LOG_PATH=$(
    /usr/bin/mktemp "${TRAINER_GUARD_LOG_PARENT}/.${TRAINER_GUARD_LOG_BASENAME}.trainer.XXXXXXXX.log"
  ) || {
    echo "FATAL: could not create exclusive trainer stdio log" >&2
    exit 75
  }
  echo "[capped_run_trainer_guard_log] path=$TRAINER_GUARD_LOG_PATH" >&2
  echo "[capped_run_trainer_stdio_log] path=$TRAINER_STDIO_LOG_PATH" >&2
fi
if [[ "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ]]; then
echo "[capped_run_trainer_safety] execution_mode=$TRAINER_EXECUTION_MODE device=$TRAINER_DEVICE data_preflight_max_wall_seconds=$TRAINER_MAX_WALL_SECONDS model_max_wall_seconds=$TRAINER_MODEL_MAX_WALL_SECONDS attended_stage_required=$TRAINER_ATTENDED_STAGE_REQUIRED gpu_index=$TRAINER_GPU_INDEX max_core_temp_c=$TRAINER_GPU_MAX_CORE_TEMP_C max_memory_temp_c=$TRAINER_GPU_MAX_MEMORY_TEMP_C max_power_limit_w=$TRAINER_GPU_MAX_POWER_LIMIT_W max_power_draw_w=$TRAINER_GPU_MAX_POWER_DRAW_W max_memory_used_mib=$TRAINER_GPU_MAX_MEMORY_USED_MIB monitor_interval_seconds=$TRAINER_GPU_MONITOR_INTERVAL_SECONDS telemetry_owner=signed_windows_bridge telemetry_url=$TRAINER_HOST_TELEMETRY_URL" >&2
fi

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
echo "[capped_run_scope_verified] memory.max=$GX1_EXPECTED_MEMORY_BYTES memory.high=$GX1_EXPECTED_MEMORY_BYTES memory.swap.max=$GX1_EXPECTED_SWAP_BYTES pids.max=$GX1_EXPECTED_TASKS" >&2
verified_cpu_affinity="$GX1_CPU_AFFINITY"
unset GX1_EXPECTED_MEMORY_BYTES GX1_EXPECTED_SWAP_BYTES GX1_EXPECTED_TASKS GX1_CPU_AFFINITY
if [[ "$GX1_CAPPED_CLASS" == trainer || "$GX1_CUDA_PRODUCER_GUARD" == true ]]; then
  [[ -x "$GX1_GPU_GUARD_PATH" ]] || { echo "FATAL: guarded CUDA safety owner unavailable inside scope" >&2; exit 75; }
  exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 "$GX1_GPU_GUARD_PATH" "$@"
fi
exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 "$@"
'
systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP" \
  -p TasksMax="$TASKS_MAX" -p KillMode=control-group \
  --setenv=GX1_EXPECTED_MEMORY_BYTES="$((requested_mem_kib * 1024))" \
  --setenv=GX1_EXPECTED_SWAP_BYTES="$((requested_swap_kib * 1024))" \
  --setenv=GX1_EXPECTED_TASKS="$TASKS_MAX" \
  --setenv=GX1_CPU_AFFINITY="$CPU_AFFINITY" \
  --setenv=GX1_CAPPED_CLASS="$JOB_CLASS" \
  --setenv=GX1_CAPPED_MEMORY_BYTES="$((requested_mem_kib * 1024))" \
  --setenv=GX1_CAPPED_SWAP_BYTES="$((requested_swap_kib * 1024))" \
  --setenv=GX1_CAPPED_TASKS_MAX="$TASKS_MAX" \
  --setenv=GX1_GPU_GUARD_PATH="$GPU_GUARD_PATH" \
  --setenv=GX1_CUDA_PRODUCER_GUARD="$CUDA_PRODUCER_GUARD" \
  --setenv=GX1_TRAINER_GUARD_LOG_PATH="$TRAINER_GUARD_LOG_PATH" \
  --setenv=GX1_TRAINER_STDIO_LOG_PATH="$TRAINER_STDIO_LOG_PATH" \
  --setenv=GX1_TRAINER_DEVICE="$TRAINER_DEVICE" \
  --setenv=GX1_TRAINER_EXECUTION_MODE="$TRAINER_EXECUTION_MODE" \
  --setenv=GX1_TRAINER_MAX_WALL_SECONDS="$TRAINER_MAX_WALL_SECONDS" \
  --setenv=GX1_TRAINER_MODEL_MAX_WALL_SECONDS="$TRAINER_MODEL_MAX_WALL_SECONDS" \
  --setenv=GX1_TRAINER_ATTENDED_STAGE_REQUIRED="$TRAINER_ATTENDED_STAGE_REQUIRED" \
  --setenv=GX1_TRAINER_GPU_INDEX="$TRAINER_GPU_INDEX" \
  --setenv=GX1_TRAINER_GPU_MAX_CORE_TEMP_C="$TRAINER_GPU_MAX_CORE_TEMP_C" \
  --setenv=GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C="$TRAINER_GPU_MAX_MEMORY_TEMP_C" \
  --setenv=GX1_TRAINER_GPU_MAX_POWER_LIMIT_W="$TRAINER_GPU_MAX_POWER_LIMIT_W" \
  --setenv=GX1_TRAINER_GPU_MAX_POWER_DRAW_W="$TRAINER_GPU_MAX_POWER_DRAW_W" \
  --setenv=GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB="$TRAINER_GPU_MAX_MEMORY_USED_MIB" \
  --setenv=GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS="$TRAINER_GPU_MONITOR_INTERVAL_SECONDS" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH="$TRAINER_HOST_TELEMETRY_QUERY_PATH" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_URL="$TRAINER_HOST_TELEMETRY_URL" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_CERT_PATH="$TRAINER_HOST_TELEMETRY_CERT_PATH" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256="$TRAINER_HOST_TELEMETRY_CERT_SHA256" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_GPU_UUID="$TRAINER_HOST_TELEMETRY_GPU_UUID" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS="$TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS" \
  --setenv=OMP_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=MKL_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=OPENBLAS_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=NUMEXPR_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=VECLIB_MAXIMUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=BLIS_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=ARROW_NUM_THREADS=1 \
  --setenv=POLARS_MAX_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 \
  -- /bin/bash -c "$SCOPE_GUARD" gx1-capped-scope "$@"
exit_code=$?
flock -u 9
exit "$exit_code"
