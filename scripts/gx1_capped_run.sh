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
# The runner also requires >=20G currently available RAM, serializes heavy jobs within GX1_CURRENT, binds the
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
H4_LINE_AUDIT_GUARD=false

CANONICAL_TRAINER_MODULE=gx1.models.entry_v10.entry_v10_ctx_train_v3
RANDOM_ACCESS_FIXED_STEP_MODULE=gx1.scripts.run_unified_exit_random_access_fixed_step_v1
RANDOM_ACCESS_VAL_MODULE=gx1.scripts.run_unified_exit_random_access_val_v1
NATIVE_CANDIDATE_WINDOW_MODULE=gx1.scripts.run_unified_exit_native_candidate_window_v1
ATTENDED_HARDWARE_SMOKE_MODULE=gx1.scripts.attended_model_native_hardware_smoke_v1
# CUDA producer routes are intentionally enumerated rather than accepting an
# arbitrary module. These are read-only, TRAIN/VAL-only evidence producers;
# neither has an execution, promotion or TEST surface.
CUDA_PRODUCER_MODULE=gx1.scripts.evaluate_entry_candidate_selective_edge_v1
TECHNICAL_VALIDATION_PRODUCER_MODULE=gx1.scripts.validate_entry_model_native_technical_checkpoint_v1
TA_C171_HEAD_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_c171_short_heads_v1
TA_NON_M5_INITIAL_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_non_m5_initial_v1
TA_TF_ACTIVITY_INITIAL_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_tf_activity_initial_v1
TA_TF_ACTIVITY_INITIAL_INPUTS_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_tf_activity_initial_inputs_v1
TA_TF_ACTIVITY_BOUNDARIES_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_tf_activity_boundaries_v1
TA_CURRENT_UPDATES_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_current_updates_v1
TA_CURRENT_NUMERIC_ROUTES_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_current_numeric_routes_v1
TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE=gx1.audit.score_ta_ema_final_fit_v1
TA_EMA_LOSS_GRADIENT_PROBE_PRODUCER_MODULE=gx1.audit.probe_ta_ema_loss_gradient_v1
TA_M5_PROJECTED_H4_LINE_AUDIT_MODULE=gx1.audit.measure_ta_m5_projected_h4_line_v1
TA_CURRENT_MODES_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_current_modes_v1
TA_CONTEXT_DRIFT_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_context_drift_v1
TA_HEAD_INPUT_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_head_input_v1
TA_FOLD2_FRESH_INITIAL_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1
TA_FOLD2_SHARED_M5_INITIAL_PRODUCER_MODULE=gx1.audit.observe_market_state_ta_fold2_shared_m5_initial_v1
TA_FOLD2_SHARED_M5_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_fold2_shared_m5_fit_v1
TA_INTEGRATED_FOLD2_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_integrated_fold2_fit_v1
TA_APR_JUN_PAIRED_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_apr_jun_paired_fit_v1
TA_M5_DECISION_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_m5_decision_fit_v1
TA_M5_EMA_SF_ONLY_CONTROL_TRAINER_MODULE=gx1.audit.run_ta_m5_ema_sf_only_control_v1
NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE=gx1.audit.qualify_native_m5_full_gc_direction_v1
RUNNER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$RUNNER_PATH")/.." && pwd -P)"
CANONICAL_TRAINER_PYTHON="$REPO_ROOT/.venv/bin/python"
CAPPED_EXECUTION_OWNER="$REPO_ROOT/gx1/contracts/gx1_capped_execution_v1.py"
GPU_GUARD_PATH="$REPO_ROOT/scripts/gx1_guarded_trainer_exec.sh"
POWER_BENCHMARK_OWNER="$REPO_ROOT/gx1/contracts/local_power_benchmark_v1.py"
POWER_BENCHMARK_SCOPE_JSON=
POWER_BENCHMARK_SCOPE_SHA256=

# Crash-response safety freeze (2026-08-23). These are source-bound constants,
# not caller-controlled defaults. V9 follows the V8 physical-host incident:
# canonical CUDA may run only with the signed Windows memory-junction bridge,
# 160 W physical limit, a separate actual-draw stop, and a narrow CPU set. Any
# later change requires a reviewed commit and a fresh source-bound recipe.
TRAINER_EXECUTION_MODE=canonical
# Private four-step native CPU mechanics is a separate exact technical route.
NATIVE_FUTURE_STATE_CPU_MECHANICS=false
NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256=
NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE=
NATIVE_FUTURE_STATE_CUDA_QUALIFICATION=false
NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256=
NATIVE_FUTURE_STATE_CUDA_LANE=
# BEGIN GX1 FROZEN BIAS ROUTE 20261122 STATE
NATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE=gx1.audit.diagnose_native_future_state_bias_v1
NATIVE_FUTURE_STATE_BIAS_FROZEN=false
NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256=
# END GX1 FROZEN BIAS ROUTE 20261122 STATE
# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 STATE
NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE=gx1.audit.diagnose_native_future_state_bias_suffix_v1
NATIVE_FUTURE_STATE_BIAS_SUFFIX=false
NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256=
# END GX1 BIAS SUFFIX ROUTE 20261122 003 STATE
# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 STATE
NATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE=gx1.audit.continue_native_future_state_bias_suffix_v1
NATIVE_FUTURE_STATE_BIAS_CONTINUE=false
NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256=
# END GX1 BIAS CONTINUE ROUTE 20261122 004 STATE
# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 STATE
NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE=gx1.audit.native_provider_hlc_row_membership_qualification_v1
NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=false
NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256=
NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE=
# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 STATE
# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 STATE
NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_MODULE=gx1.audit.diagnose_native_provider_hlc_gradient_profile_v1
NATIVE_PROVIDER_HLC_GRADIENT_PROFILE=false
NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256=
# END GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 STATE
# BEGIN GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 STATE
NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_MODULE=gx1.audit.diagnose_native_provider_hlc_endpoint_cross_v1
NATIVE_PROVIDER_HLC_ENDPOINT_CROSS=false
NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256=
# END GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 STATE
# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 STATE
NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=false
NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256=
NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE=
# END GX1 ALL TF LEARNING ROUTE 20261124 001 STATE
# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 STATE
NATIVE_PROVIDER_HLC_LEARNING_RESUME_MODULE=gx1.audit.native_provider_hlc_learning_v1
NATIVE_PROVIDER_HLC_LEARNING_RESUME=false
NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256=
# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 STATE
# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 STATE
NATIVE_PROVIDER_HLC_LEARNING_MODULE=gx1.audit.native_provider_hlc_learning_v1
NATIVE_PROVIDER_HLC_LEARNING=false
NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256=
# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 STATE
# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 STATE
NATIVE_PROVIDER_HLC_REPLAY_MODULE=gx1.audit.native_provider_hlc_replay_v1
NATIVE_PROVIDER_HLC_REPLAY=false
NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256=
NATIVE_PROVIDER_HLC_REPLAY_LANE=
# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 STATE
# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE
NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=false
NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256=
NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=false
NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256=
NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE=
# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE
# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE
NATIVE_PROVIDER_HLC_INITIAL_MODULE=gx1.audit.native_future_state_provider_hlc_initial_v1
NATIVE_V3_DEVELOPMENT=false
NATIVE_V3_DEVELOPMENT_PLAN_SHA256=
NATIVE_V3_DEVELOPMENT_PHASE=
NATIVE_PROVIDER_HLC_INITIAL=false
NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256=
# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE
# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 STATE
NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE=gx1.audit.capture_native_future_state_full_train_v1
NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE=false
NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256=
# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 STATE
# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE
NATIVE_FUTURE_STATE_THROUGHPUT_MODULE=gx1.audit.measure_native_future_state_throughput_v1
NATIVE_FUTURE_STATE_TRAINING_MODULE=gx1.audit.native_future_state_training_v1
NATIVE_FUTURE_STATE_THROUGHPUT=false
NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256=
NATIVE_FUTURE_STATE_TRAINING=false
NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256=
NATIVE_FUTURE_STATE_TRAINING_LANE=
# END GX1 FUTURE LEAF ROUTES 20261121 STATE
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
# Conservative initial host ceilings, not hardware limits or an optimization claim.
TRAINER_CPU_MAX_TEMP_C=80
TRAINER_CPU_MAX_PACKAGE_POWER_W=125
TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=$((8 * 1024 * 1024 * 1024))
TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=$((8 * 1024 * 1024 * 1024))
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
# Audits keep their existing affinity. The authorized trainer uses all 19
# WSL vCPUs while its numerical library remains bounded to eight threads.
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

require_campaign_plan_environment() {
  local plan_path="${GX1_CAMPAIGN_PLAN_PATH:-}"
  local plan_sha="${GX1_CAMPAIGN_PLAN_SHA256:-}"
  local plan_file_sha="${GX1_CAMPAIGN_PLAN_FILE_SHA256:-}"
  [[ "$plan_path" == /* \
    && "$plan_path" != *$'\n'* \
    && -f "$plan_path" \
    && ! -L "$plan_path" \
    && "$(readlink -f "$plan_path")" == "$plan_path" \
    && "$plan_sha" =~ ^[0-9a-f]{64}$ \
    && "$plan_file_sha" =~ ^[0-9a-f]{64}$ \
    && "$(/usr/bin/sha256sum "$plan_path" | /usr/bin/awk '{print $1}')" == "$plan_file_sha" ]] || {
    echo "FATAL: campaign plan path/file SHA-256 binding invalid" >&2
    exit 75
  }
}

# This selector does not execute a model. Its caller supplies one fixed plan and
# output path; all eleven target arguments and the plan bytes must match.
matches_bound_m5_intervention() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
    && "$4" == --direction-m5-momentum-intervention-plan && "$5" == "$expected_plan" \
    && "$6" == --direction-m5-momentum-intervention-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact current-profile GC freshness scope, with the existing signed guard.
matches_bound_gc_q1_freshness() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
    && "$4" == --direction-gc-q1-freshness-plan && "$5" == "$expected_plan" \
    && "$6" == --direction-gc-q1-freshness-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}


# Exact one-use fold-1 TRAIN diagnostic, preserving the signed CUDA producer guard.
matches_bound_market_state_fold1_paired_diagnostic() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
    && "$4" == --market-state-fold1-paired-diagnostic-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-fold1-paired-diagnostic-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact user-authorized 250 W frozen TA direction-cache scope (08.11).
matches_bound_market_state_ta_direction_cache() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-direction-cache-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-direction-cache-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact read-only capture of the short c171 initial TRAIN head inputs.
matches_bound_market_state_ta_c171_short_heads() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_C171_HEAD_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-c171-short-heads-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-c171-short-heads-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact read-only initial comparison for independent non-M5 side heads.
matches_bound_market_state_ta_non_m5_initial() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_NON_M5_INITIAL_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-non-m5-initial-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-non-m5-initial-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact read-only initial comparison for independent timeframe activity heads.
matches_bound_market_state_ta_tf_activity_initial() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_TF_ACTIVITY_INITIAL_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-tf-activity-initial-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-tf-activity-initial-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}
# Exact read-only full-prefix capture of activity INITIAL head inputs.
matches_bound_market_state_ta_tf_activity_initial_inputs() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_TF_ACTIVITY_INITIAL_INPUTS_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-tf-activity-initial-inputs-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-tf-activity-initial-inputs-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact read-only full-prefix capture of activity paired pre-query and joint inputs.
matches_bound_market_state_ta_tf_activity_boundaries() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_TF_ACTIVITY_BOUNDARIES_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-tf-activity-boundaries-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-tf-activity-boundaries-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact passive current-endpoint gradient observation, including metadata-only preflight.
matches_bound_market_state_ta_current_updates() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_CURRENT_UPDATES_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-current-updates-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-current-updates-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact current TRAIN mode contrast; no optimization, with metadata-only preflight.
matches_bound_market_state_ta_current_modes() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_CURRENT_MODES_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-current-modes-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-current-modes-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}


# Exact passive current-context intervention, never a feature/training promotion.
matches_bound_market_state_ta_context_drift() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_CONTEXT_DRIFT_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-context-drift-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-context-drift-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

matches_bound_market_state_ta_head_input() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_HEAD_INPUT_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-head-input-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-head-input-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Fixed fold 2 TRAIN/later zero-update observation; one plan and one output root.
matches_bound_market_state_ta_fold2_fresh_initial() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_FOLD2_FRESH_INITIAL_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-fold2-fresh-initial-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-fold2-fresh-initial-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One fixed shared-M5-head zero-update baseline; no alternative plan or output root.
matches_bound_market_state_ta_fold2_shared_m5_initial() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_FOLD2_SHARED_M5_INITIAL_PRODUCER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One exact shared-M5 TRAIN pass; same capped trainer and signed GPU guard.
matches_bound_market_state_ta_fold2_shared_m5_fit() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_FOLD2_SHARED_M5_FIT_TRAINER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One exact integrated five-TF TRAIN pass with its fixed initial/final development score.
matches_bound_market_state_ta_integrated_fold2_fit() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_INTEGRATED_FOLD2_FIT_TRAINER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One exact paired-S full TRAIN pass and fixed April-June development score.
matches_bound_market_state_ta_apr_jun_paired_fit() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m && "$3" == "$TA_APR_JUN_PAIRED_FIT_TRAINER_MODULE" && "$4" == --plan && "$5" == "$expected_plan" && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ && "$8" == --device && "$9" == cuda && "${10}" == --out-dir && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One exact direct-cost M5 fit on the reused Fold2 and April-June axes.
matches_bound_market_state_ta_m5_decision_fit() {
  local expected_plan="$1" expected_output="$2"
  if [[ "$expected_plan" == "$REPO_ROOT/configs/research/TA_M5_COLD_JOINT_FIT_20261117_001.json" ]]; then
    [[ "$expected_output" == /home/andre2/GX1_RUNS/TA_M5_COLD_JOINT_FIT_20261117_001 ]] || return 1
  elif [[ "$expected_plan" == "$REPO_ROOT/configs/research/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001.json" ]]; then
    [[ "$expected_output" == /home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001 ]] || return 1
  fi
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ ( "$expected_plan" == "$REPO_ROOT/configs/research/TA_M5_COLD_JOINT_FIT_20261117_001.json" \
      || "$expected_plan" == "$REPO_ROOT/configs/research/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001.json" ) \
      && "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m && "$3" == "$TA_M5_DECISION_FIT_TRAINER_MODULE" && "$4" == --plan && "$5" == "$expected_plan" && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ && "$8" == --device && "$9" == cuda && "${10}" == --out-dir && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One exact same-view S/F-primary control, with full FIT and later score.
matches_bound_ta_m5_ema_sf_only_control() {
  local expected_plan="$REPO_ROOT/configs/research/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001.json"
  local expected_output=/home/andre2/GX1_RUNS/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_M5_EMA_SF_ONLY_CONTROL_TRAINER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Only the two source-bound native M5/fullGC fit plans may use this trainer route.
matches_bound_native_m5_full_gc_fit() {
  local expected_plan expected_output
  case "${5:-}" in
    "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIT_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIT_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIT_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_002.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_002.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_002 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_001.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_001.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_001 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_002.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_002.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_002 ;;
    "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json"
      expected_output=/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003 ;;
    "$REPO_ROOT"/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_WINDOW_*.json|"$REPO_ROOT"/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_CONTINUE_*.json)
      local window_basename="${5##*/}"
      [[ "$window_basename" =~ ^NATIVE_M5_TA_ATR03_FIVE_YEAR_(WINDOW|CONTINUE)_[0-9]{8}_[0-9]{3}\.json$ ]] || return 1
      expected_plan="$REPO_ROOT/configs/research/$window_basename"
      expected_output="/home/andre2/GX1_RUNS/${window_basename%.json}" ;;
    *) return 1 ;;
  esac
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m     && "$3" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE"     && "$4" == --plan && "$5" == "$expected_plan"     && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$     && "$8" == --device && "$9" == cuda && "${10}" == --out-dir     && "${11}" == "$expected_output" ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan"     && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact passive current first-batch numerical route comparison, including metadata-only preflight.
matches_bound_market_state_ta_current_numeric_routes() {
  local expected_plan="$1" expected_output="$2"
  shift 2
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_CURRENT_NUMERIC_ROUTES_PRODUCER_MODULE" \
    && "$4" == --market-state-ta-current-numeric-routes-plan && "$5" == "$expected_plan" \
    && "$6" == --market-state-ta-current-numeric-routes-plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" && "$expected_plan" == /* && "$expected_output" == /* ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One read-only final FIT score for the sealed EMA checkpoint; no training route.
matches_bound_ta_ema_final_fit_score() {
  local expected_plan="$REPO_ROOT/configs/research/TA_M5_EMA_FINAL_FIT_SCORE_20261118_001.json"
  local expected_output=/home/andre2/GX1_RUNS/TA_M5_EMA_FINAL_FIT_SCORE_20261118_001
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# One no-update gradient partition of the frozen EMA model on fixed FIT batches.
matches_bound_ta_ema_loss_gradient_probe() {
  local expected_plan="$REPO_ROOT/configs/research/TA_M5_EMA_LOSS_GRADIENT_PROBE_20261118_001.json"
  local expected_output=/home/andre2/GX1_RUNS/TA_M5_EMA_LOSS_GRADIENT_PROBE_20261118_001
  if [[ $# -eq 12 ]]; then
    [[ "${12}" == --preflight ]] || return 1
    set -- "${@:1:11}"
  fi
  [[ $# -eq 11 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_EMA_LOSS_GRADIENT_PROBE_PRODUCER_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --device && "$9" == cuda && "${10}" == --out-dir \
    && "${11}" == "$expected_output" ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

# Exact one-use CPU audit for M5 price versus a causal projected H4 line.
matches_bound_ta_m5_projected_h4_line() {
  local expected_plan="$REPO_ROOT/configs/research/TA_M5_PROJECTED_H4_LINE_20261118_001.json"
  local expected_output=/home/andre2/GX1_RUNS/TA_M5_PROJECTED_H4_LINE_20261118_001
  if [[ $# -eq 10 ]]; then
    [[ "${10}" == --preflight ]] || return 1
    set -- "${@:1:9}"
  fi
  [[ $# -eq 9 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$TA_M5_PROJECTED_H4_LINE_AUDIT_MODULE" \
    && "$4" == --plan && "$5" == "$expected_plan" \
    && "$6" == --plan-sha256 && "$7" =~ ^[0-9a-f]{64}$ \
    && "$8" == --out-dir && "$9" == "$expected_output" ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$7" ]]
}

matches_bound_native_future_state_cpu_mechanics() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001.json"
  if [[ $# -eq 9 ]]; then
    [[ "${9}" == --preflight ]] || return 1
    set -- "${@:1:8}"
  fi
  [[ $# -eq 8 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE" \
    && "$4" == --native-future-state-cpu-mechanics \
    && "$5" == --plan-sha256 && "$6" =~ ^[0-9a-f]{64}$ \
    && "$7" == --lane && ( "$8" == continuous || "$8" == replay ) ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$6" ]]
}

matches_bound_native_future_state_cuda_qualification() {
  local expected_plan
  if [[ $# -eq 9 ]]; then
    [[ "${9}" == --preflight ]] || return 1
    set -- "${@:1:8}"
  fi
  [[ $# -eq 8 ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE" \
    && "$4" == --native-future-state-cuda-qualification \
    && "$5" == --plan-sha256 && "$6" =~ ^[0-9a-f]{64}$ \
    && "$7" == --lane ]] || return 1
  case "$8" in
    continuous|replay) expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001.json" ;;
    initial) expected_plan="$REPO_ROOT/configs/research/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001.json" ;;
    *) return 1 ;;
  esac
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$6" ]]
}

# BEGIN GX1 FUTURE LEAF ROUTES 20261121 MATCHERS
matches_bound_native_future_state_throughput() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

matches_bound_native_future_state_training() {
  local expected_plan
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_TRAINING_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ && "$6" == --lane ]] || return 1
  case "$7" in
    joint|joint_replay|m5_primary|m5_primary_replay)
      expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001.json" ;;
    main) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json" ;;
    recovery_probe) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001.json" ;;
    main_resume) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001.json" ;;
    *) return 1 ;;
  esac
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_future_state_leaf() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: native future-state leaf requires exact canonical trainer scope" >&2
    exit 75
  }
  local future_leaf_output future_leaf_seconds future_leaf_kind
  if [[ "${3:-}" == "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE" ]] \
    && matches_bound_native_future_state_throughput "$@"; then
    future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003
    future_leaf_seconds=1800
    future_leaf_kind=throughput
  elif [[ "${3:-}" == "$NATIVE_FUTURE_STATE_TRAINING_MODULE" ]] \
    && matches_bound_native_future_state_training "$@"; then
    future_leaf_kind=training
    case "$7" in
      joint|m5_primary) future_leaf_seconds=5400 ;;
      joint_replay|m5_primary_replay|recovery_probe) future_leaf_seconds=1800 ;;
      main|main_resume) future_leaf_seconds=12000 ;;
    esac
    case "$7" in
      main) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001 ;;
      recovery_probe) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001 ;;
      main_resume) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001 ;;
      *) future_leaf_output="/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001/$7" ;;
    esac
  else
    echo "FATAL: native future-state leaf CLI/plan is not exactly bound" >&2
    exit 75
  fi
  [[ ! -e "$future_leaf_output" && ! -L "$future_leaf_output" \
    && -d "${future_leaf_output%/*}" && ! -L "${future_leaf_output%/*}" \
    && "$(readlink -f "${future_leaf_output%/*}")" == "${future_leaf_output%/*}" ]] || {
    echo "FATAL: native future-state leaf output/guard-log parent is not fresh canonical scope" >&2
    exit 75
  }
  if [[ "$future_leaf_kind" == throughput ]]; then
    NATIVE_FUTURE_STATE_THROUGHPUT=true
    NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256="$5"
  else
    NATIVE_FUTURE_STATE_TRAINING=true
    NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256="$5"
    NATIVE_FUTURE_STATE_TRAINING_LANE="$7"
  fi
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$future_leaf_output"
  TRAINER_MAX_WALL_SECONDS="$future_leaf_seconds"
  TRAINER_MODEL_MAX_WALL_SECONDS="$future_leaf_seconds"
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=300
  TRAINER_GPU_MAX_POWER_DRAW_W=310
  TRAINER_GPU_MAX_CORE_TEMP_C=85
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
}
# END GX1 FUTURE LEAF ROUTES 20261121 MATCHERS

# BEGIN GX1 FROZEN BIAS ROUTE 20261122 MATCHERS
matches_bound_native_future_state_bias_frozen() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_future_state_bias_frozen() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: frozen bias diagnosis requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_future_state_bias_frozen "$@" || {
    echo "FATAL: frozen bias diagnosis CLI/plan is not exactly bound" >&2
    exit 75
  }
  local frozen_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001
  [[ ! -e "$frozen_output" && ! -L "$frozen_output" \
    && -d "${frozen_output%/*}" && ! -L "${frozen_output%/*}" \
    && "$(readlink -f "${frozen_output%/*}")" == "${frozen_output%/*}" ]] || {
    echo "FATAL: frozen bias output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_FUTURE_STATE_BIAS_FROZEN=true
  NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$frozen_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3600
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=300
  TRAINER_GPU_MAX_POWER_DRAW_W=310
  TRAINER_GPU_MAX_CORE_TEMP_C=85
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
}
# END GX1 FROZEN BIAS ROUTE 20261122 MATCHERS

# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 MATCHERS
matches_bound_native_future_state_bias_suffix() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_future_state_bias_suffix() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: bias suffix order study requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_future_state_bias_suffix "$@" || {
    echo "FATAL: bias suffix order study CLI/plan is not exactly bound" >&2
    exit 75
  }
  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003
  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \
    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \
    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {
    echo "FATAL: bias suffix output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_FUTURE_STATE_BIAS_SUFFIX=true
  NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$suffix_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
}
# END GX1 BIAS SUFFIX ROUTE 20261122 003 MATCHERS

# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 MATCHERS
matches_bound_native_future_state_bias_suffix_continuation() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_future_state_bias_suffix_continuation() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: bias suffix continuation requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_future_state_bias_suffix_continuation "$@" || {
    echo "FATAL: bias suffix continuation CLI/plan is not exactly bound" >&2
    exit 75
  }
  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004
  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \
    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \
    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {
    echo "FATAL: bias suffix continuation output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_FUTURE_STATE_BIAS_CONTINUE=true
  NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$suffix_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
}
# END GX1 BIAS CONTINUE ROUTE 20261122 004 MATCHERS

# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 MATCHERS
matches_bound_native_future_state_full_train_capture() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_future_state_full_train_capture() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: full TRAIN capture requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_future_state_full_train_capture "$@" || {
    echo "FATAL: full TRAIN capture CLI/plan is not exactly bound" >&2
    exit 75
  }
  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001
  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \
    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \
    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {
    echo "FATAL: full TRAIN capture output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE=true
  NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$suffix_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
}
# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 MATCHERS

# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 MATCHERS
matches_bound_native_provider_hlc_initial() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_initial() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_initial "$@" || {
    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2
    exit 75
  }
  local initial_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002
  [[ ! -e "$initial_output" && ! -L "$initial_output" \
    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \
    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {
    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_INITIAL=true
  NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$initial_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 MATCHERS

# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 MATCHERS
matches_bound_native_provider_hlc_replay() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001.json"
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \
    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_replay() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_replay "$@" || {
    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2
    exit 75
  }
  local replay_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001_$7"
  [[ ! -e "$replay_output" && ! -L "$replay_output" \
    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \
    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {
    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_REPLAY=true
  NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256="$5"
  NATIVE_PROVIDER_HLC_REPLAY_LANE="$7"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$replay_output"
  TRAINER_MAX_WALL_SECONDS=1800
  TRAINER_MODEL_MAX_WALL_SECONDS=1770
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 MATCHERS

# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 MATCHERS
matches_bound_native_provider_hlc_learning() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_LEARNING_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_learning() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC learning requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_learning "$@" || {
    echo "FATAL: provider HLC learning CLI/plan is not exactly bound" >&2
    exit 75
  }
  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001"
  [[ ! -e "$learning_output" && ! -L "$learning_output" \
    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \
    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {
    echo "FATAL: provider HLC learning output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_LEARNING=true
  NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$learning_output"
  TRAINER_MAX_WALL_SECONDS=12000
  TRAINER_MODEL_MAX_WALL_SECONDS=11970
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 MATCHERS

# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 MATCHERS
matches_bound_native_provider_hlc_learning_resume() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_MODULE" \
    && "$4" == --resume-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_learning_resume() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC learning resume requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_learning_resume "$@" || {
    echo "FATAL: provider HLC learning resume CLI/plan is not exactly bound" >&2
    exit 75
  }
  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004"
  [[ ! -e "$learning_output" && ! -L "$learning_output" \
    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \
    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {
    echo "FATAL: provider HLC learning resume output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_LEARNING_RESUME=true
  NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$learning_output"
  TRAINER_MAX_WALL_SECONDS=12000
  TRAINER_MODEL_MAX_WALL_SECONDS=11970
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 MATCHERS

# BEGIN GX1 V3 DEVELOPMENT ROUTE 20261127 001
matches_bound_native_v3_development() {
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" \
    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \
    && "$4" == --v3-development-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \
    && "$6" == --phase ]] || return 1
  case "$7" in cohort|warm|cold|fit) ;; *) return 1 ;; esac
  local plan_suffix="${7^^}"
  if [[ "$7" == warm ]]; then plan_suffix=WARM_002; fi
  if [[ "$7" == fit ]]; then plan_suffix=FIT_003; fi
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_${plan_suffix}.json"
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" \
    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}
configure_native_v3_development() {
  matches_bound_native_v3_development "$@" || { echo 'FATAL: exact V3 development CLI/plan required' >&2; exit 75; }
  [[ "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \
    && "$SWAP" == 512M && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo 'FATAL: V3 development forbids alternate resource modes' >&2; exit 75;
  }
  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_$7"
  if [[ "$7" == fit ]]; then output="${output}_003"; fi
  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \
    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {
    echo 'FATAL: V3 development output must be fresh and canonical' >&2; exit 75;
  }
  NATIVE_V3_DEVELOPMENT=true
  NATIVE_V3_DEVELOPMENT_PLAN_SHA256="$5"
  NATIVE_V3_DEVELOPMENT_PHASE="$7"
  if [[ "$7" == cohort ]]; then
    [[ "$JOB_CLASS" == producer && "$MEM" == 12G ]] || { echo 'FATAL: V3 cohort requires CPU producer12G' >&2; exit 75; }
    return
  fi
  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G ]] || { echo 'FATAL: V3 NN requires canonical trainer20G' >&2; exit 75; }
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$output"
  TRAINER_MAX_WALL_SECONDS=1800
  TRAINER_MODEL_MAX_WALL_SECONDS=1770
  if [[ "$7" == fit ]]; then TRAINER_MAX_WALL_SECONDS=5400; TRAINER_MODEL_MAX_WALL_SECONDS=5370; fi
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 V3 DEVELOPMENT ROUTE 20261127 001

# BEGIN GX1 ALL TF V3 ROUTES 20261125 001 MATCHERS
matches_bound_native_all_tf_provider_hlc_v3_initial_segment() {
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \
    && "$4" == --all-tf-v3-score-segment-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  local base_root=/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001
  local plan_dir="$base_root/PLANS"
  local expected_plan="$plan_dir/$5.json"
  [[ -d "$base_root" && ! -L "$base_root" && "$(readlink -f "$base_root")" == "$base_root" \
    && -d "$plan_dir" && ! -L "$plan_dir" && "$(readlink -f "$plan_dir")" == "$plan_dir" \
    && -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_all_tf_provider_hlc_v3_initial_segment() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_all_tf_provider_hlc_v3_initial_segment "$@" || {
    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2
    exit 75
  }
  local initial_output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001/SEGMENT_$5"
  [[ ! -e "$initial_output" && ! -L "$initial_output" \
    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \
    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {
    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$initial_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}

matches_bound_native_all_tf_provider_hlc_v3_replay() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001.json"
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \
    && "$4" == --all-tf-v3-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \
    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_all_tf_provider_hlc_v3_replay() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_all_tf_provider_hlc_v3_replay "$@" || {
    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2
    exit 75
  }
  local replay_output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001_$7"
  [[ ! -e "$replay_output" && ! -L "$replay_output" \
    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \
    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {
    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=true
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$5"
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$7"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$replay_output"
  TRAINER_MAX_WALL_SECONDS=1800
  TRAINER_MODEL_MAX_WALL_SECONDS=1770
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 ALL TF V3 ROUTES 20261125 001 MATCHERS
# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS
matches_bound_native_provider_hlc_all_tf_initial() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002.json"
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \
    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_all_tf_initial() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_all_tf_initial "$@" || {
    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2
    exit 75
  }
  local initial_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002
  [[ ! -e "$initial_output" && ! -L "$initial_output" \
    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \
    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {
    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$initial_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}

matches_bound_native_provider_hlc_all_tf_replay() {
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001.json"
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \
    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \
    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_all_tf_replay() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_all_tf_replay "$@" || {
    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2
    exit 75
  }
  local replay_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001_$7"
  [[ ! -e "$replay_output" && ! -L "$replay_output" \
    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \
    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {
    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=true
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$5"
  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$7"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$replay_output"
  TRAINER_MAX_WALL_SECONDS=1800
  TRAINER_MODEL_MAX_WALL_SECONDS=1770
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS

# BEGIN GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 MATCHERS
matches_bound_native_provider_hlc_all_tf_initial_segment() {
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \
    && "$4" == --all-tf-score-segment-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  local base_root=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003
  local plan_dir="$base_root/PLANS"
  local expected_plan="$plan_dir/$5.json"
  [[ -d "$base_root" && ! -L "$base_root" && "$(readlink -f "$base_root")" == "$base_root" \
    && -d "$plan_dir" && ! -L "$plan_dir" && "$(readlink -f "$plan_dir")" == "$plan_dir" \
    && -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_all_tf_initial_segment() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2
    exit 75
  }
  matches_bound_native_provider_hlc_all_tf_initial_segment "$@" || {
    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2
    exit 75
  }
  local initial_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003/SEGMENT_$5"
  [[ ! -e "$initial_output" && ! -L "$initial_output" \
    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \
    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {
    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2
    exit 75
  }
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true
  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$initial_output"
  TRAINER_MAX_WALL_SECONDS=3600
  TRAINER_MODEL_MAX_WALL_SECONDS=3570
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}

# END GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 MATCHERS

# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 MATCHERS
matches_bound_native_provider_hlc_all_tf_learning() {
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  local expected_plan
  case "$3:$4" in
    "$NATIVE_PROVIDER_HLC_LEARNING_MODULE:--all-tf-learning-plan-sha256")
      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002.json" ;;
    "$NATIVE_PROVIDER_HLC_LEARNING_MODULE:--all-tf-learning-resume-plan-sha256")
      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002/RECOVERY_PLANS/$5.json" ;;
    gx1.audit.native_provider_hlc_block_order_v1:--block-order-plan-sha256)
      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001.json" ;;
    gx1.audit.native_provider_hlc_block_order_v1:--block-order-resume-plan-sha256)
      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001/RECOVERY_PLANS/$5.json" ;;
    gx1.audit.native_provider_hlc_row_membership_fit_v1:--row-membership-plan-sha256)
      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001.json" ;;
    gx1.audit.native_provider_hlc_row_membership_fit_v1:--row-membership-resume-plan-sha256)
      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001/RECOVERY_PLANS/$5.json" ;;
    *) return 1 ;;
  esac
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_all_tf_learning() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: ALL-TF learning requires exact canonical trainer containment" >&2; exit 75
  }
  matches_bound_native_provider_hlc_all_tf_learning "$@" || {
    echo "FATAL: ALL-TF learning requires its exact canonical command/plan" >&2; exit 75
  }
  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002"
  local learning_mode=main learning_env_name
  if [[ "$3" == gx1.audit.native_provider_hlc_block_order_v1 ]]; then
    learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001"
  fi
  if [[ "$3" == gx1.audit.native_provider_hlc_row_membership_fit_v1 ]]; then
    learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001"
  fi
  if [[ "$4" == --all-tf-learning-resume-plan-sha256 || "$4" == --block-order-resume-plan-sha256 || "$4" == --row-membership-resume-plan-sha256 ]]; then
    learning_output="$learning_output/RECOVERY_$5"
    learning_mode=resume
  fi
  [[ ! -e "$learning_output" && ! -L "$learning_output" \
    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \
    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {
    echo "FATAL: ALL-TF learning requires a fresh output in its canonical parent" >&2; exit 75
  }
  for learning_env_name in ${!GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING@}; do
    case "$learning_env_name" in
      GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING|GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256|GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE) ;;
      *) echo "FATAL: unknown ALL-TF learning environment field" >&2; exit 75 ;;
    esac
  done
  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-false}" == false \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" ]] || {
    echo "FATAL: ALL-TF learning forbids mixed legacy modes/plans/lanes" >&2; exit 75
  }
  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-}" == true \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$5" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$learning_mode" ]] || {
      echo "FATAL: ALL-TF learning mode/plan differs from protected parent" >&2; exit 75
    }
  else
    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" ]] || {
      echo "FATAL: unprotected ALL-TF learning environment pollution" >&2; exit 75
    }
  fi
  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=true
  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256="$5"
  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE="$learning_mode"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$learning_output"
  TRAINER_MAX_WALL_SECONDS=12000
  TRAINER_MODEL_MAX_WALL_SECONDS=11970
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 ALL TF LEARNING ROUTE 20261124 001 MATCHERS

# BEGIN GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 MATCHERS
matches_bound_native_provider_hlc_endpoint_cross() {
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_NONE_ENDPOINT_CROSS_20261124_002.json"
  [[ -f "$expected_plan" && ! -L "$expected_plan" \
    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

configure_native_provider_hlc_endpoint_cross() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: endpoint cross requires exact canonical CUDA containment" >&2; exit 75
  }
  matches_bound_native_provider_hlc_endpoint_cross "$@" || {
    echo "FATAL: endpoint cross command/plan is not exactly bound" >&2; exit 75
  }
  local cross_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_NONE_ENDPOINT_CROSS_20261124_002
  local cross_env_name
  [[ ! -e "$cross_output" && ! -L "$cross_output" \
    && -d "${cross_output%/*}" && ! -L "${cross_output%/*}" \
    && "$(readlink -f "${cross_output%/*}")" == "${cross_output%/*}" ]] || {
    echo "FATAL: endpoint cross requires a fresh canonical output" >&2; exit 75
  }
  for cross_env_name in ${!GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS@}; do
    case "$cross_env_name" in
      GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS|GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256) ;;
      *) echo "FATAL: unknown endpoint cross environment field" >&2; exit 75 ;;
    esac
  done
  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == "false" \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-false}" == "false" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "" ]] || {
    echo "FATAL: endpoint cross forbids mixed legacy modes/plans/lanes" >&2; exit 75
  }
  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS:-}" == true \
      && "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256:-}" == "$5" ]] || {
      echo "FATAL: endpoint cross mode/plan differs from protected parent" >&2; exit 75
    }
  else
    [[ "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256:-}" ]] || {
      echo "FATAL: unprotected endpoint cross environment pollution" >&2; exit 75
    }
  fi
  NATIVE_PROVIDER_HLC_ENDPOINT_CROSS=true
  NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$cross_output"
  TRAINER_MAX_WALL_SECONDS=900
  TRAINER_MODEL_MAX_WALL_SECONDS=870
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 MATCHERS
# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 MATCHERS
matches_bound_native_provider_hlc_gradient_profile() {
  if [[ $# -eq 6 ]]; then
    [[ "$6" == --preflight ]] || return 1
    set -- "${@:1:5}"
  fi
  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_MODULE" \
    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1
  local profile_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_BLOCK_ORDER_GRADIENT_PROFILE_20261125_001.json"
  [[ -f "$profile_plan" && ! -L "$profile_plan" \
    && "$(readlink -f "$profile_plan")" == "$profile_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$profile_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

require_native_provider_hlc_gradient_profile_environment() {
  local profile_nested="$1" profile_env profile_state profile_expected
  # Reuse the declared inactive native states; never accept another route's authority.
  for profile_env in ${!GX1_NATIVE_@}; do
    case "$profile_env" in
      GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE|GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256) continue ;;
      GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256) profile_state=NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256 ;;
      *) profile_state="${profile_env#GX1_}" ;;
    esac
    [[ "$profile_state" != *_MODULE && -v "$profile_state" \
      && ( "${!profile_state}" == false || -z "${!profile_state}" ) \
      && "${!profile_env}" == "${!profile_state}" ]] || {
      echo "FATAL: gradient profile forbids mixed or unknown native environment" >&2; exit 75
    }
  done
  if [[ "$profile_nested" == true ]]; then
    for profile_state in ${!NATIVE_@}; do
      case "$profile_state" in
        *_MODULE|NATIVE_PROVIDER_HLC_GRADIENT_PROFILE|NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256) continue ;;
        NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256) profile_env=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256 ;;
        *) profile_env="GX1_$profile_state" ;;
      esac
      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {
        echo "FATAL: gradient profile requires every inactive protected native state" >&2; exit 75
      }
    done
  fi
}

configure_native_provider_hlc_gradient_profile() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: gradient profile requires exact canonical CUDA containment" >&2; exit 75
  }
  matches_bound_native_provider_hlc_gradient_profile "$@" || {
    echo "FATAL: gradient profile command/plan is not exactly bound" >&2; exit 75
  }
  local profile_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_BLOCK_ORDER_GRADIENT_PROFILE_20261125_001
  [[ ! -e "$profile_output" && ! -L "$profile_output" \
    && -d "${profile_output%/*}" && ! -L "${profile_output%/*}" \
    && "$(readlink -f "${profile_output%/*}")" == "${profile_output%/*}" ]] || {
    echo "FATAL: gradient profile requires a fresh canonical output" >&2; exit 75
  }
  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then
    require_native_provider_hlc_gradient_profile_environment true
    [[ "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE:-}" == true \
      && "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256:-}" == "$5" ]] || {
      echo "FATAL: gradient profile mode/plan differs from protected parent" >&2; exit 75
    }
  else
    require_native_provider_hlc_gradient_profile_environment false
    [[ "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256:-}" ]] || {
      echo "FATAL: unprotected gradient profile environment pollution" >&2; exit 75
    }
  fi
  NATIVE_PROVIDER_HLC_GRADIENT_PROFILE=true
  NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256="$5"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$profile_output"
  TRAINER_MAX_WALL_SECONDS=900
  TRAINER_MODEL_MAX_WALL_SECONDS=870
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 MATCHERS
# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 MATCHERS
matches_bound_native_provider_hlc_row_membership_qualification() {
  if [[ $# -eq 8 ]]; then
    [[ "$8" == --preflight ]] || return 1
    set -- "${@:1:7}"
  fi
  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1
  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \
    && "$3" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE" \
    && "$4" == --membership-qualification-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \
    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1
  local profile_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001.json"
  [[ -f "$profile_plan" && ! -L "$profile_plan" \
    && "$(readlink -f "$profile_plan")" == "$profile_plan" ]] || return 1
  [[ "$(/usr/bin/sha256sum "$profile_plan" | /usr/bin/awk '{print $1}')" == "$5" ]]
}

require_native_provider_hlc_row_membership_qualification_environment() {
  local profile_nested="$1" profile_env profile_state profile_expected
  # Reuse the declared inactive native states; never accept another route's authority.
  for profile_env in ${!GX1_NATIVE_@}; do
    case "$profile_env" in
      GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION|GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256|GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE) continue ;;
      GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256) profile_state=NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256 ;;
      *) profile_state="${profile_env#GX1_}" ;;
    esac
    [[ "$profile_state" != *_MODULE && -v "$profile_state" \
      && ( "${!profile_state}" == false || -z "${!profile_state}" ) \
      && "${!profile_env}" == "${!profile_state}" ]] || {
      echo "FATAL: row membership qualification forbids mixed or unknown native environment" >&2; exit 75
    }
  done
  if [[ "$profile_nested" == true ]]; then
    for profile_state in ${!NATIVE_@}; do
      case "$profile_state" in
        *_MODULE|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE) continue ;;
        NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256) profile_env=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256 ;;
        *) profile_env="GX1_$profile_state" ;;
      esac
      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {
        echo "FATAL: row membership qualification requires every inactive protected native state" >&2; exit 75
      }
    done
  fi
}

configure_native_provider_hlc_row_membership_qualification() {
  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \
    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \
    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {
    echo "FATAL: row membership qualification requires exact canonical CUDA containment" >&2; exit 75
  }
  matches_bound_native_provider_hlc_row_membership_qualification "$@" || {
    echo "FATAL: row membership qualification command/plan is not exactly bound" >&2; exit 75
  }
  local profile_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001_$7"
  [[ ! -e "$profile_output" && ! -L "$profile_output" \
    && -d "${profile_output%/*}" && ! -L "${profile_output%/*}" \
    && "$(readlink -f "${profile_output%/*}")" == "${profile_output%/*}" ]] || {
    echo "FATAL: row membership qualification requires a fresh canonical output" >&2; exit 75
  }
  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then
    require_native_provider_hlc_row_membership_qualification_environment true
    [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-}" == true \
      && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" == "$5" \
      && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" == "$7" ]] || {
      echo "FATAL: row membership qualification mode/plan differs from protected parent" >&2; exit 75
    }
  else
    require_native_provider_hlc_row_membership_qualification_environment false
    [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" ]] || {
      echo "FATAL: unprotected row membership qualification environment pollution" >&2; exit 75
    }
  fi
  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=true
  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256="$5"
  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE="$7"
  TRAINER_DEVICE=cuda
  TRAINER_EXECUTION_MODE=canonical
  TRAINER_OUT_BUNDLE_DIR="$profile_output"
  TRAINER_MAX_WALL_SECONDS=1800
  TRAINER_MODEL_MAX_WALL_SECONDS=1770
  TRAINER_GPU_INDEX=0
  TRAINER_GPU_MAX_POWER_LIMIT_W=250
  TRAINER_GPU_MAX_POWER_DRAW_W=250
  TRAINER_GPU_MAX_CORE_TEMP_C=75
  TRAINER_GPU_MAX_MEMORY_TEMP_C=80
  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288
  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1
  TRAINER_CPU_MAX_TEMP_C=80
  TRAINER_CPU_MAX_PACKAGE_POWER_W=125
  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592
  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592
}
# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 MATCHERS
validate_target_command() {
# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 DISPATCH
  local gradient_profile_arg gradient_profile_reference=false
  for gradient_profile_arg in "$@"; do
    case "$gradient_profile_arg" in
      "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE"|*/gx1/audit/native_provider_hlc_row_membership_qualification_v1.py|gx1/audit/native_provider_hlc_row_membership_qualification_v1.py)
        gradient_profile_reference=true ;;
    esac
  done
  if [[ "$gradient_profile_reference" == true ]]; then
    configure_native_provider_hlc_row_membership_qualification "$@"
    return
  fi
# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 DISPATCH
# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 DISPATCH
  local gradient_profile_arg gradient_profile_reference=false
  for gradient_profile_arg in "$@"; do
    case "$gradient_profile_arg" in
      "$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_MODULE"|*/gx1/audit/diagnose_native_provider_hlc_gradient_profile_v1.py|gx1/audit/diagnose_native_provider_hlc_gradient_profile_v1.py)
        gradient_profile_reference=true ;;
    esac
  done
  if [[ "$gradient_profile_reference" == true ]]; then
    configure_native_provider_hlc_gradient_profile "$@"
    return
  fi
# END GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 DISPATCH
# BEGIN GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 DISPATCH
  local endpoint_cross_arg endpoint_cross_reference=false
  for endpoint_cross_arg in "$@"; do
    case "$endpoint_cross_arg" in
      "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_MODULE"|*/gx1/audit/diagnose_native_provider_hlc_endpoint_cross_v1.py|gx1/audit/diagnose_native_provider_hlc_endpoint_cross_v1.py)
        endpoint_cross_reference=true ;;
    esac
  done
  if [[ "$endpoint_cross_reference" == true ]]; then
    configure_native_provider_hlc_endpoint_cross "$@"
    return
  fi
# END GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 DISPATCH
# BEGIN GX1 V3 DEVELOPMENT DISPATCH 20261127 001
  local development_arg
  for development_arg in "$@"; do
    case "$development_arg" in
      --v3-development-plan-sha256|--v3-development-plan-sha256=*) configure_native_v3_development "$@"; return ;;
    esac
  done
# END GX1 V3 DEVELOPMENT DISPATCH 20261127 001
# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 DISPATCH
  local all_tf_learning_arg all_tf_learning_reference=false
  for all_tf_learning_arg in "$@"; do
    case "$all_tf_learning_arg" in
      --all-tf-learning-plan-sha256|--all-tf-learning-plan-sha256=*|--all-tf-learning-resume-plan-sha256|--all-tf-learning-resume-plan-sha256=*|--block-order-plan-sha256|--block-order-plan-sha256=*|--block-order-resume-plan-sha256|--block-order-resume-plan-sha256=*|gx1.audit.native_provider_hlc_block_order_v1|*/gx1/audit/native_provider_hlc_block_order_v1.py|gx1/audit/native_provider_hlc_block_order_v1.py|--row-membership-plan-sha256|--row-membership-plan-sha256=*|--row-membership-resume-plan-sha256|--row-membership-resume-plan-sha256=*|gx1.audit.native_provider_hlc_row_membership_fit_v1|*/gx1/audit/native_provider_hlc_row_membership_fit_v1.py|gx1/audit/native_provider_hlc_row_membership_fit_v1.py)
        all_tf_learning_reference=true ;;
    esac
  done
  if [[ "$all_tf_learning_reference" == true ]]; then
    configure_native_provider_hlc_all_tf_learning "$@"
    return
  fi
# END GX1 ALL TF LEARNING ROUTE 20261124 001 DISPATCH
# BEGIN GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 DISPATCH
  if [[ "${4:-}" == --all-tf-score-segment-plan-sha256 ]]; then
    configure_native_provider_hlc_all_tf_initial_segment "$@"
    return
  fi
# END GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 DISPATCH
# BEGIN GX1 ALL TF V3 ROUTES 20261125 001 DISPATCH
  if [[ "${4:-}" == --all-tf-v3-score-segment-plan-sha256 ]]; then
    [[ "${3:-}" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" ]] || { echo "FATAL: V3 INITIAL requires exact module" >&2; exit 75; }
    configure_native_all_tf_provider_hlc_v3_initial_segment "$@"; return
  fi
  if [[ "${4:-}" == --all-tf-v3-plan-sha256 ]]; then
    [[ "${3:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" ]] || { echo "FATAL: V3 replay requires exact module" >&2; exit 75; }
    configure_native_all_tf_provider_hlc_v3_replay "$@"; return
  fi
# END GX1 ALL TF V3 ROUTES 20261125 001 DISPATCH
# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH
  if [[ "${4:-}" == --all-tf-plan-sha256 ]]; then
    case "${3:-}" in
      "$NATIVE_PROVIDER_HLC_INITIAL_MODULE")
        configure_native_provider_hlc_all_tf_initial "$@"; return ;;
      "$NATIVE_PROVIDER_HLC_REPLAY_MODULE")
        configure_native_provider_hlc_all_tf_replay "$@"; return ;;
      *)
        echo "FATAL: ALL-TF plan flag requires its exact native module" >&2
        exit 75 ;;
    esac
  fi
# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH
# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 DISPATCH
  local provider_learning_resume_arg provider_learning_resume_reference=false
  for provider_learning_resume_arg in "$@"; do
    case "$provider_learning_resume_arg" in
      --resume-plan-sha256|--resume-plan-sha256=*)
        provider_learning_resume_reference=true ;;
    esac
  done
  if [[ "$provider_learning_resume_reference" == true ]]; then
    configure_native_provider_hlc_learning_resume "$@"
    return
  fi
# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 DISPATCH
# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 DISPATCH
  local provider_learning_arg provider_learning_reference=false
  for provider_learning_arg in "$@"; do
    case "$provider_learning_arg" in
      "$NATIVE_PROVIDER_HLC_LEARNING_MODULE"|*/gx1/audit/native_provider_hlc_learning_v1.py|gx1/audit/native_provider_hlc_learning_v1.py)
        provider_learning_reference=true ;;
    esac
  done
  if [[ "$provider_learning_reference" == true ]]; then
    configure_native_provider_hlc_learning "$@"
    return
  fi
# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 DISPATCH
# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 DISPATCH
  local provider_replay_arg provider_replay_reference=false
  for provider_replay_arg in "$@"; do
    case "$provider_replay_arg" in
      "$NATIVE_PROVIDER_HLC_REPLAY_MODULE"|*/gx1/audit/native_provider_hlc_replay_v1.py|gx1/audit/native_provider_hlc_replay_v1.py)
        provider_replay_reference=true ;;
    esac
  done
  if [[ "$provider_replay_reference" == true ]]; then
    configure_native_provider_hlc_replay "$@"
    return
  fi
# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 DISPATCH
# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 DISPATCH
  local provider_initial_arg provider_initial_reference=false
  for provider_initial_arg in "$@"; do
    case "$provider_initial_arg" in
      "$NATIVE_PROVIDER_HLC_INITIAL_MODULE"|*/gx1/audit/native_future_state_provider_hlc_initial_v1.py|gx1/audit/native_future_state_provider_hlc_initial_v1.py)
        provider_initial_reference=true ;;
    esac
  done
  if [[ "$provider_initial_reference" == true ]]; then
    configure_native_provider_hlc_initial "$@"
    return
  fi
# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 DISPATCH
# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 DISPATCH
  local frozen_bias_arg frozen_bias_reference=false
  for frozen_bias_arg in "$@"; do
    case "$frozen_bias_arg" in
      "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE"|*/gx1/audit/capture_native_future_state_full_train_v1.py|gx1/audit/capture_native_future_state_full_train_v1.py)
        frozen_bias_reference=true ;;
    esac
  done
  if [[ "$frozen_bias_reference" == true ]]; then
    configure_native_future_state_full_train_capture "$@"
    return
  fi
# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 DISPATCH
# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 DISPATCH
  local frozen_bias_arg frozen_bias_reference=false
  for frozen_bias_arg in "$@"; do
    case "$frozen_bias_arg" in
      "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE"|*/gx1/audit/continue_native_future_state_bias_suffix_v1.py|gx1/audit/continue_native_future_state_bias_suffix_v1.py)
        frozen_bias_reference=true ;;
    esac
  done
  if [[ "$frozen_bias_reference" == true ]]; then
    configure_native_future_state_bias_suffix_continuation "$@"
    return
  fi
# END GX1 BIAS CONTINUE ROUTE 20261122 004 DISPATCH
# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 DISPATCH
  local frozen_bias_arg frozen_bias_reference=false
  for frozen_bias_arg in "$@"; do
    case "$frozen_bias_arg" in
      "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE"|*/gx1/audit/diagnose_native_future_state_bias_suffix_v1.py|gx1/audit/diagnose_native_future_state_bias_suffix_v1.py)
        frozen_bias_reference=true ;;
    esac
  done
  if [[ "$frozen_bias_reference" == true ]]; then
    configure_native_future_state_bias_suffix "$@"
    return
  fi
# END GX1 BIAS SUFFIX ROUTE 20261122 003 DISPATCH
# BEGIN GX1 FROZEN BIAS ROUTE 20261122 DISPATCH
  local frozen_bias_arg frozen_bias_reference=false
  for frozen_bias_arg in "$@"; do
    case "$frozen_bias_arg" in
      "$NATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE"|*/gx1/audit/diagnose_native_future_state_bias_v1.py|gx1/audit/diagnose_native_future_state_bias_v1.py)
        frozen_bias_reference=true ;;
    esac
  done
  if [[ "$frozen_bias_reference" == true ]]; then
    configure_native_future_state_bias_frozen "$@"
    return
  fi
# END GX1 FROZEN BIAS ROUTE 20261122 DISPATCH
# BEGIN GX1 FUTURE LEAF ROUTES 20261121 DISPATCH
  local future_leaf_arg future_leaf_reference=false
  for future_leaf_arg in "$@"; do
    case "$future_leaf_arg" in
      "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE"|"$NATIVE_FUTURE_STATE_TRAINING_MODULE"|*/gx1/audit/measure_native_future_state_throughput_v1.py|*/gx1/audit/native_future_state_training_v1.py|gx1/audit/measure_native_future_state_throughput_v1.py|gx1/audit/native_future_state_training_v1.py)
        future_leaf_reference=true ;;
    esac
  done
  if [[ "$future_leaf_reference" == true ]]; then
    configure_native_future_state_leaf "$@"
    return
  fi
# END GX1 FUTURE LEAF ROUTES 20261121 DISPATCH
  local executable_basename="${1##*/}" target_arg module
  local trainer_reference=false hardware_smoke_reference=false h4_line_reference=false
  local native_cpu_mechanics_reference=false native_cuda_qualification_reference=false
  local trainer_flag_count=0 trainer_device_count=0 hardware_smoke_flag_count=0
  local profile_count=0 execution_tier_count=0 out_bundle_dir_count=0 out_dir_count=0
  local checkpoint_dir_count=0 launch_manifest_count=0 stage_count=0 arm_batch_size_count=0
  local progress_path_count=0 train_session_count=0 max_optimizer_steps_count=0
  local profile_value= execution_tier_value= stage_value= arm_batch_size_value=
  local stage_requires_attended=false
  local -a target_args=("$@")

  case "$executable_basename" in
    env|bash|sh|dash|zsh|ksh)
      echo "FATAL: env and shell wrappers are forbidden as capped targets" >&2
      exit 75
      ;;
  esac

  for target_arg in "$@"; do
    if [[ "$target_arg" == *"$CANONICAL_TRAINER_MODULE"*       || "$target_arg" == *"$RANDOM_ACCESS_FIXED_STEP_MODULE"*       || "$target_arg" == *"$RANDOM_ACCESS_VAL_MODULE"*       || "$target_arg" == *"$NATIVE_CANDIDATE_WINDOW_MODULE"* ]]; then
      trainer_reference=true
    fi
    if [[ "$target_arg" == *"$ATTENDED_HARDWARE_SMOKE_MODULE"* ]]; then
      hardware_smoke_reference=true
    fi
    if [[ "$target_arg" == *"$TA_M5_PROJECTED_H4_LINE_AUDIT_MODULE"* ]]; then
      h4_line_reference=true
    fi
    if [[ "$target_arg" == *--native-future-state-cuda-qualification* ]]; then
      native_cuda_qualification_reference=true
    fi
    if [[ "$target_arg" == *--native-future-state-cpu-mechanics* ]]; then
      native_cpu_mechanics_reference=true
    fi
    if [[ "$target_arg" == "--train" ]]; then
      trainer_flag_count=$((trainer_flag_count + 1))
    fi
  done

  if [[ "$native_cuda_qualification_reference" == true ]]; then
    if [[ "$JOB_CLASS" != trainer || "$ATTENDED_SMOKE" != false \
      || "$CUDA_PRODUCER_GUARD" != false || "$MEM" != 20G || "$SWAP" != 512M \
      || -n "$POWER_BENCHMARK_SCOPE_JSON" || -n "$POWER_BENCHMARK_SCOPE_SHA256" ]] \
      || ! matches_bound_native_future_state_cuda_qualification "$@"; then
      echo "FATAL: native future-state CUDA qualification is not exactly bound" >&2
      exit 75
    fi
    local future_cuda_output future_cuda_seconds
    if [[ "$8" == initial ]]; then
      future_cuda_output=/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001
      future_cuda_seconds=5400
    else
      future_cuda_output="/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001/$8"
      future_cuda_seconds=1800
    fi
    [[ ! -e "$future_cuda_output" && ! -L "$future_cuda_output" ]] || {
      echo "FATAL: native future-state CUDA output is not fresh" >&2
      exit 75
    }
    NATIVE_FUTURE_STATE_CUDA_QUALIFICATION=true
    NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256="${6}"
    NATIVE_FUTURE_STATE_CUDA_LANE="${8}"
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="$future_cuda_output"
    TRAINER_MAX_WALL_SECONDS="$future_cuda_seconds"
    TRAINER_MODEL_MAX_WALL_SECONDS="$future_cuda_seconds"
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi

  if [[ "$native_cpu_mechanics_reference" == true ]]; then
    if [[ "$JOB_CLASS" != producer || "$ATTENDED_SMOKE" != false \
      || "$CUDA_PRODUCER_GUARD" != false || "$MEM" != 12G || "$SWAP" != 512M ]] \
      || ! matches_bound_native_future_state_cpu_mechanics "$@"; then
      echo "FATAL: native future-state CPU mechanics is not exactly bound" >&2
      exit 75
    fi
    NATIVE_FUTURE_STATE_CPU_MECHANICS=true
    NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256="${6}"
    NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE="${8}"
    return
  fi

  if [[ "$JOB_CLASS" == "audit" || "$JOB_CLASS" == "producer" ]]; then
    if [[ "$trainer_reference" == true ]]; then
      echo "FATAL: canonical trainer requires --class trainer" >&2
      exit 75
    fi
    if [[ "$hardware_smoke_reference" == true ]]; then
      echo "FATAL: attended hardware smoke requires --class trainer" >&2
      exit 75
    fi
    if [[ "$h4_line_reference" == true ]]; then
      if [[ "$JOB_CLASS" != audit || "$ATTENDED_SMOKE" != false \
        || "$CUDA_PRODUCER_GUARD" != false || "$MEM" != 4G || "$SWAP" != 512M ]] \
        || ! matches_bound_ta_m5_projected_h4_line "$@"; then
        echo "FATAL: projected H4 line audit is not exactly bound" >&2
        exit 75
      fi
      H4_LINE_AUDIT_GUARD=true
    fi
    if [[ "$CUDA_PRODUCER_GUARD" == true ]]; then
      [[ "$JOB_CLASS" == producer ]] || {
        echo "FATAL: --cuda-producer requires --class producer" >&2
        exit 75
      }
      if ! is_direct_python "$1" || [[ "${2:-}" != "-m" ]] \
        || { [[ "${3:-}" != "$CUDA_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TECHNICAL_VALIDATION_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_C171_HEAD_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_NON_M5_INITIAL_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_TF_ACTIVITY_INITIAL_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_TF_ACTIVITY_INITIAL_INPUTS_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_TF_ACTIVITY_BOUNDARIES_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_CURRENT_UPDATES_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_CURRENT_NUMERIC_ROUTES_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_EMA_LOSS_GRADIENT_PROBE_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_CURRENT_MODES_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_CONTEXT_DRIFT_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_HEAD_INPUT_PRODUCER_MODULE" ]] \
          && [[ "${3:-}" != "$TA_FOLD2_FRESH_INITIAL_PRODUCER_MODULE" ]] && [[ "${3:-}" != "$TA_FOLD2_SHARED_M5_INITIAL_PRODUCER_MODULE" ]]; }; then
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
      if [[ "${3:-}" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
        && "${4:-}" == --direction-gc-q1-freshness-plan ]]; then
        local gc_root=/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
        local gc_plan="$gc_root/DIRECTION_GC_Q1_FRESHNESS_PLAN_20261030_001.json"
        local gc_output="$gc_root/DIRECTION_GC_Q1_FRESHNESS_20261030_001"
        if ! matches_bound_gc_q1_freshness "$gc_plan" "$gc_output" "$@"; then
          echo "FATAL: GC Q1 freshness producer 250W scope is not exactly bound" >&2
          exit 75
        fi
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
      fi
      if [[ "${3:-}" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
        && "${4:-}" == --market-state-fold1-paired-diagnostic-plan ]]; then
        local paired_plan="$REPO_ROOT/configs/research/MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_20261107_002.json"
        local paired_output=/home/andre2/GX1_RUNS/MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_20261107_002
        if ! matches_bound_market_state_fold1_paired_diagnostic "$paired_plan" "$paired_output" "$@"; then
          echo "FATAL: fold-1 paired diagnostic 250W scope is not exactly bound" >&2
          exit 75
        fi
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
      fi
      if [[ "${3:-}" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
        && "${4:-}" == --market-state-ta-direction-cache-plan ]]; then
        local ta_cache_root
        case "${5:-}" in
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FINAL_CACHE_20261109_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FINAL_CACHE_20261109_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_REVERSED_REPR_CACHE_20261110_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_REVERSED_REPR_CACHE_20261110_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_REVERSED_FIXED_NATIVE_SCORE_20261110_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_REVERSED_FIXED_NATIVE_SCORE_20261110_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_LOCKED_SHRINK_NATIVE_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_LOCKED_SHRINK_NATIVE_20261111_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_FUSION_OBSERVATION_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_FUSION_OBSERVATION_20261111_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_FUSION_INITIAL_OBSERVATION_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_FUSION_INITIAL_OBSERVATION_20261111_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_MTF_RETAINED_INTERVENTION_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_MTF_RETAINED_INTERVENTION_20261111_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_TEMPORAL_BOUNDARY_OBSERVATION_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_TEMPORAL_BOUNDARY_OBSERVATION_20261111_001 ;;
          /home/andre2/GX1_RUNS/MARKET_STATE_TA_MIXING_GRADIENT_OBSERVATION_20261111_001/PLAN.json)
            ta_cache_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_MIXING_GRADIENT_OBSERVATION_20261111_001 ;;
          *) echo "FATAL: unbound TA direction-cache plan" >&2; exit 75 ;;
        esac
        if ! matches_bound_market_state_ta_direction_cache "$ta_cache_root/PLAN.json" "$ta_cache_root" "$@"; then
          echo "FATAL: TA direction-cache 250W scope is not exactly bound" >&2
          exit 75
        fi
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
      fi
      if [[ "${3:-}" == "$TA_C171_HEAD_PRODUCER_MODULE" ]]; then
        local ta_heads_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_MATCHED_TRAIN_MODE_20261111_001
        if ! matches_bound_market_state_ta_c171_short_heads "$ta_heads_root/PLAN.json" "$ta_heads_root" "$@"; then
          echo "FATAL: c171 initial-head observation is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_NON_M5_INITIAL_PRODUCER_MODULE" ]]; then
        local ta_non_m5_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_INITIAL_PARITY_20261111_001
        if ! matches_bound_market_state_ta_non_m5_initial "$ta_non_m5_root/PLAN.json" "$ta_non_m5_root" "$@"; then
          echo "FATAL: non-M5 initial comparison is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_TF_ACTIVITY_INITIAL_PRODUCER_MODULE" ]]; then
        local ta_tf_activity_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_INITIAL_PARITY_20261112_001
        if ! matches_bound_market_state_ta_tf_activity_initial "$ta_tf_activity_root/PLAN.json" "$ta_tf_activity_root" "$@"; then
          echo "FATAL: timeframe activity initial comparison is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_TF_ACTIVITY_INITIAL_INPUTS_PRODUCER_MODULE" ]]; then
        local ta_tf_activity_inputs_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_INITIAL_INPUTS_20261112_001
        if ! matches_bound_market_state_ta_tf_activity_initial_inputs "$ta_tf_activity_inputs_root/PLAN.json" "$ta_tf_activity_inputs_root" "$@"; then
          echo "FATAL: timeframe activity initial input capture is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_TF_ACTIVITY_BOUNDARIES_PRODUCER_MODULE" ]]; then
        local ta_tf_activity_boundaries_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_BOUNDARIES_20261112_001
        if ! matches_bound_market_state_ta_tf_activity_boundaries "$ta_tf_activity_boundaries_root/PLAN.json" "$ta_tf_activity_boundaries_root" "$@"; then
          echo "FATAL: timeframe activity paired boundary capture is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_CURRENT_UPDATES_PRODUCER_MODULE" ]]; then
        local ta_current_updates_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_UPDATE_OBSERVATION_20261113_003
        if ! matches_bound_market_state_ta_current_updates "$ta_current_updates_root/PLAN.json" "$ta_current_updates_root" "$@"; then
          echo "FATAL: current update observation is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1200
        TRAINER_MODEL_MAX_WALL_SECONDS=1200
      fi
      if [[ "${3:-}" == "$TA_CURRENT_MODES_PRODUCER_MODULE" ]]; then
        local ta_current_modes_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_MODE_OBSERVATION_20261113_001
        if ! matches_bound_market_state_ta_current_modes "$ta_current_modes_root/PLAN.json" "$ta_current_modes_root" "$@"; then
          echo "FATAL: current mode observation is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=900
        TRAINER_MODEL_MAX_WALL_SECONDS=900
      fi
      if [[ "${3:-}" == "$TA_CONTEXT_DRIFT_PRODUCER_MODULE" ]]; then
        local ta_context_drift_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_CONTEXT_DRIFT_OBSERVATION_20261113_001
        if ! matches_bound_market_state_ta_context_drift "$ta_context_drift_root/PLAN.json" "$ta_context_drift_root" "$@"; then
          echo "FATAL: current context drift observation is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=900
        TRAINER_MODEL_MAX_WALL_SECONDS=900
      fi
      if [[ "${3:-}" == "$TA_HEAD_INPUT_PRODUCER_MODULE" ]]; then
        local ta_head_input_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_HEAD_INPUT_OBSERVATION_20261113_001
        local ta_prepool_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_PREPOOL_CAPTURE_20261113_001
        if ! matches_bound_market_state_ta_head_input "$ta_head_input_root/PLAN.json" "$ta_head_input_root" "$@" \
          && ! matches_bound_market_state_ta_head_input "$ta_prepool_root/PLAN.json" "$ta_prepool_root" "$@"; then
          echo "FATAL: current head input observation is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=900
        TRAINER_MODEL_MAX_WALL_SECONDS=900
      fi
      if [[ "${3:-}" == "$TA_FOLD2_FRESH_INITIAL_PRODUCER_MODULE" ]]; then
        local fold2_initial_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_FRESH_INITIAL_20261114_004
        if ! matches_bound_market_state_ta_fold2_fresh_initial "$fold2_initial_root/PLAN.json" "$fold2_initial_root" "$@"; then
          echo "FATAL: fold 2 fresh INITIAL observation is not exactly bound" >&2
          exit 75
        fi
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1800
        TRAINER_MODEL_MAX_WALL_SECONDS=1800
      fi
      if [[ "${3:-}" == "$TA_FOLD2_SHARED_M5_INITIAL_PRODUCER_MODULE" ]]; then
        local shared_m5_initial_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SHARED_M5_INITIAL_20261114_001
        if ! matches_bound_market_state_ta_fold2_shared_m5_initial "$shared_m5_initial_root/PLAN.json" "$shared_m5_initial_root" "$@"; then
          echo "FATAL: shared-M5 initial observation is not exactly bound" >&2
          exit 75
        fi
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1800
        TRAINER_MODEL_MAX_WALL_SECONDS=1800
      fi
      if [[ "${3:-}" == "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE" ]]; then
        if [[ "$MEM" != 20G || "$SWAP" != 512M ]] \
          || ! matches_bound_ta_ema_final_fit_score "$@"; then
          echo "FATAL: final EMA FIT score producer is not exactly bound" >&2
          exit 75
        fi
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        # Reuse the 3600 s CUDA producer cap: the prior 6118 s job had
        # 4037 mixed forwards; this FIT-only replay has 1849, no updates.
      fi
      if [[ "${3:-}" == "$TA_EMA_LOSS_GRADIENT_PROBE_PRODUCER_MODULE" ]]; then
        if [[ "$MEM" != 20G || "$SWAP" != 512M ]] \
          || ! matches_bound_ta_ema_loss_gradient_probe "$@"; then
          echo "FATAL: EMA loss gradient probe is not exactly bound" >&2
          exit 75
        fi
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=1800
        TRAINER_MODEL_MAX_WALL_SECONDS=1800
      fi
      if [[ "${3:-}" == "$TA_CURRENT_NUMERIC_ROUTES_PRODUCER_MODULE" ]]; then
        local ta_current_numeric_routes_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_NUMERIC_ROUTES_20261113_001
        if ! matches_bound_market_state_ta_current_numeric_routes "$ta_current_numeric_routes_root/PLAN.json" "$ta_current_numeric_routes_root" "$@"; then
          echo "FATAL: current numerical route comparison is not exactly bound" >&2
          exit 75
        fi
        # Match the original deterministic_fp32 evaluation's numerical threads.
        # Affinity, TasksMax and every hardware/memory stop remain unchanged.
        NUMERICAL_THREAD_COUNT=8
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
        TRAINER_MAX_WALL_SECONDS=600
        TRAINER_MODEL_MAX_WALL_SECONDS=600
      fi
      if [[ "${3:-}" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \
        && "${4:-}" == --direction-m5-momentum-intervention-plan ]]; then
        # Only the exact one-shot M5 plans below have this configured-limit
        # exception. The 170 W draw stop, 65/80 C temperature stops, signed
        # telemetry, cgroup, lock, GPU memory and timeout remain unchanged.
        local m5_root=/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
        local m5_plan m5_output
        case "${5:-}" in
          "$m5_root/DIRECTION_M5_MOMENTUM_INTERVENTION_PLAN_20261029_003.json")
            m5_plan="$m5_root/DIRECTION_M5_MOMENTUM_INTERVENTION_PLAN_20261029_003.json"
            m5_output="$m5_root/DIRECTION_M5_MOMENTUM_INTERVENTION_20261029_003" ;;
          "$m5_root/DIRECTION_M5_MOMENTUM_TRACE_PLAN_20261029_002.json")
            m5_plan="$m5_root/DIRECTION_M5_MOMENTUM_TRACE_PLAN_20261029_002.json"
            m5_output="$m5_root/DIRECTION_M5_MOMENTUM_TRACE_20261029_002" ;;
          /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/UPSTREAM_GRADIENT_PLAN_20261031_002.json)
            m5_plan=/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/UPSTREAM_GRADIENT_PLAN_20261031_002.json
            m5_output=/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/UPSTREAM_GRADIENT_20261031_002 ;;
          *) echo "FATAL: unknown M5 momentum producer plan" >&2; exit 75 ;;
        esac
        if ! matches_bound_m5_intervention "$m5_plan" "$m5_output" "$@"; then
          echo "FATAL: M5 momentum producer 250W scope is not exactly bound" >&2
          exit 75
        fi
        TRAINER_GPU_MAX_POWER_LIMIT_W=250
      fi
    fi
    return
  fi

  if [[ "${3:-}" == "$TA_FOLD2_SHARED_M5_FIT_TRAINER_MODULE" ]]; then
    local shared_m5_fit_root=/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SHARED_M5_FIT_20261114_001
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]] \
      || ! matches_bound_market_state_ta_fold2_shared_m5_fit \
        "$shared_m5_fit_root/PLAN.json" "$shared_m5_fit_root" "$@"; then
      echo "FATAL: shared-M5 fit is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="$shared_m5_fit_root"
    TRAINER_MAX_WALL_SECONDS=10800
    TRAINER_MODEL_MAX_WALL_SECONDS=10800
    # Same physical research limits as the completed fold-2 plan004.
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if [[ "${3:-}" == "$TA_INTEGRATED_FOLD2_FIT_TRAINER_MODULE" ]]; then
    local integrated_fit_root=/home/andre2/GX1_RUNS/TA_INTEGRATED_FOLD2_FIT_20261116_002
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]] \
      || ! matches_bound_market_state_ta_integrated_fold2_fit \
        "$integrated_fit_root/PLAN.json" "$integrated_fit_root" "$@"; then
      echo "FATAL: integrated five-TF fit is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="$integrated_fit_root"
    TRAINER_MAX_WALL_SECONDS=10800
    TRAINER_MODEL_MAX_WALL_SECONDS=10800
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if [[ "${3:-}" == "$TA_APR_JUN_PAIRED_FIT_TRAINER_MODULE" ]]; then
    local paired_fit_root=/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_FIT_20261117_002
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]] || ! matches_bound_market_state_ta_apr_jun_paired_fit "$REPO_ROOT/configs/research/TA_INTEGRATED_APR_JUN_PAIRED_FIT_20261117_002.json" "$paired_fit_root" "$@"; then
      echo "FATAL: April-June paired TA fit is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="$paired_fit_root"
    TRAINER_MAX_WALL_SECONDS=10800
    TRAINER_MODEL_MAX_WALL_SECONDS=10800
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if [[ "${3:-}" == "$TA_M5_DECISION_FIT_TRAINER_MODULE" ]]; then
    local decision_root=/home/andre2/GX1_RUNS/TA_M5_DECISION_FIT_20261117_004
    local cold_decision_root=/home/andre2/GX1_RUNS/TA_M5_COLD_JOINT_FIT_20261117_001
    local ema_decision_root=/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]]; then
      echo "FATAL: direct-cost M5 decision fit is not exactly bound" >&2
      exit 75
    fi
    if matches_bound_market_state_ta_m5_decision_fit "$REPO_ROOT/configs/research/TA_M5_DECISION_FIT_20261117_004.json" "$decision_root" "$@"; then
      :
    elif matches_bound_market_state_ta_m5_decision_fit "$REPO_ROOT/configs/research/TA_M5_COLD_JOINT_FIT_20261117_001.json" "$cold_decision_root" "$@"; then
      decision_root="$cold_decision_root"
    elif matches_bound_market_state_ta_m5_decision_fit "$REPO_ROOT/configs/research/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001.json" "$ema_decision_root" "$@"; then
      decision_root="$ema_decision_root"
    else
      echo "FATAL: direct-cost M5 decision fit is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="$decision_root"
    TRAINER_MAX_WALL_SECONDS=10800
    TRAINER_MODEL_MAX_WALL_SECONDS=10800
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if [[ "${3:-}" == "$TA_M5_EMA_SF_ONLY_CONTROL_TRAINER_MODULE" ]]; then
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]] \
      || ! matches_bound_ta_m5_ema_sf_only_control "$@"; then
      echo "FATAL: EMA S/F-only control is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR=/home/andre2/GX1_RUNS/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001
    TRAINER_MAX_WALL_SECONDS=10800
    TRAINER_MODEL_MAX_WALL_SECONDS=10800
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if [[ "${3:-}" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE" ]]; then
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false ]]       || ! matches_bound_native_m5_full_gc_fit "$@"; then
      echo "FATAL: native M5/fullGC fit is not exactly bound" >&2
      exit 75
    fi
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="${11}"
    if [[ "$5" =~ /NATIVE_M5_TA_ATR03_FIVE_YEAR_(WINDOW|CONTINUE)_[0-9]{8}_[0-9]{3}\.json$ ]]; then
      local window_wall_seconds
      if ! window_wall_seconds=$(/usr/bin/jq -er 'select(.schema_version == "gx1_native_m5_main_checkpoint_window_plan_v1") | .execution.outer_seconds | select(type == "number" and floor == . and (. == 1800 or . == 10800))' "$5"); then
        echo "FATAL: checkpoint window has no exact supported execution deadline" >&2
        exit 75
      fi
      TRAINER_MAX_WALL_SECONDS="$window_wall_seconds"
      TRAINER_MODEL_MAX_WALL_SECONDS="$window_wall_seconds"
    elif [[ "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001.json" ]]; then
      TRAINER_MAX_WALL_SECONDS=5400
      TRAINER_MODEL_MAX_WALL_SECONDS=5400
    elif [[ "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_002.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json" ]]; then
      TRAINER_MAX_WALL_SECONDS=10800
      TRAINER_MODEL_MAX_WALL_SECONDS=10800
    elif [[ "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001.json" ]]; then
      TRAINER_MAX_WALL_SECONDS=3600
      TRAINER_MODEL_MAX_WALL_SECONDS=3600
    elif [[ "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIT_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_002.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004.json" || "$5" == "$REPO_ROOT/configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003.json" ]]; then
      TRAINER_MAX_WALL_SECONDS=1800
      TRAINER_MODEL_MAX_WALL_SECONDS=1800
    else
      TRAINER_MAX_WALL_SECONDS=10800
      TRAINER_MODEL_MAX_WALL_SECONDS=10800
    fi
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    return
  fi
  if ! is_direct_python "$1" \
    || [[ "${2:-}" != "-m" ]]; then
    echo "FATAL: trainer class is reserved for the canonical trainer module as a direct target" >&2
    exit 75
  fi
  module="${3:-}"
  [[ "$module" == "$NATIVE_CANDIDATE_WINDOW_MODULE" \
    || "$module" == "$ATTENDED_HARDWARE_SMOKE_MODULE" ]] || {
    echo "FATAL: retired training route; use the current native campaign and NEXT_RUN_POLICY.json" >&2
    exit 75
  }
  if [[ "$module" == "$NATIVE_CANDIDATE_WINDOW_MODULE" ]]; then
    require_campaign_plan_environment
    if [[ "$ATTENDED_SMOKE" != false || "$CUDA_PRODUCER_GUARD" != false \
      || ${#target_args[@]} -ne 9 \
      || "${target_args[3]}" != --window-policy \
      || "${target_args[4]}" != /* || ! -f "${target_args[4]}" || -L "${target_args[4]}" \
      || "${target_args[5]}" != --window-policy-file-sha256 \
      || ! "${target_args[6]}" =~ ^[0-9a-f]{64}$ \
      || "${target_args[7]}" != --progress-path || "${target_args[8]}" != /* \
      || ! "${GX1_CAMPAIGN_INVOCATION_SHA256:-}" =~ ^[0-9a-f]{64}$ \
      || "${GX1_CAMPAIGN_GUARD_LOG_PATH:-}" != /* ]]; then
      echo "FATAL: native candidate requires the exact guarded campaign window" >&2
      exit 75
    fi
    [[ "$(readlink -f "${target_args[4]}")" == "${target_args[4]}" \
      && "$(/usr/bin/sha256sum "${target_args[4]}" | /usr/bin/awk '{print $1}')" == "${target_args[6]}" ]] || exit 75
    # The bound native window owns the finite calibration exception; an unbound
    # readiness check still blocks full training until its evidence is present.
    (cd "$REPO_ROOT" && "$CANONICAL_TRAINER_PYTHON" -m scripts.collect_gx1_handover_readonly \
      --require-next-run --native-window-policy "${target_args[4]}" \
      --native-window-policy-file-sha256 "${target_args[6]}") || exit 78
    # Read the hash-bound window; callers cannot supply a free-form guard timeout.
    local native_seconds
    native_seconds=$("$CANONICAL_TRAINER_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); n=p["max_invocation_seconds"]; assert p["schema_version"] in ("gx1_native_candidate_window_policy_v1", "gx1_native_direction_research_window_policy_v1", "gx1_native_direction_prefix_fit_window_policy_v1", "gx1_native_direction_checkpoint_diagnostic_window_policy_v1", "gx1_native_direction_frozen_extraction_window_policy_v1") and type(n) is int and (n == 12000 or n == 3000 and p["schema_version"] == "gx1_native_direction_research_window_policy_v1" and p.get("recipe", {}).get("path") == "/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_PAIRED_BATCH_ORDER_20261110_002/RECIPE.json" or n == 3600 and p["schema_version"] == "gx1_native_direction_research_window_policy_v1" and p.get("recipe", {}).get("path") in ("/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002/RECIPE.json", "/home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_SHORT_LEARNING_20261112_001/RECIPE.json", "/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001/RECIPE.json", "/home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_002/RECIPE.json")); print(n)' "${target_args[4]}") || exit 75
    if [[ "$native_seconds" == 3600 ]]; then TRAINER_MAX_WALL_SECONDS=3900; elif [[ "$native_seconds" == 3000 ]]; then TRAINER_MAX_WALL_SECONDS=3300; else TRAINER_MAX_WALL_SECONDS=$((native_seconds + 1800)); fi
    TRAINER_MODEL_MAX_WALL_SECONDS=$TRAINER_MAX_WALL_SECONDS
    # Operator-authorized native main training; legacy invocations retain their limits.
    TRAINER_GPU_MAX_POWER_LIMIT_W=300
    # Exact matched research scopes retain the user's configured 250 W ceiling.
    # The window and full recipe are revalidated before selecting the tighter
    # bound; all other native/thermal limits and signed telemetry stay intact.
    local research_power_limit
    research_power_limit=$(cd "$REPO_ROOT" && "$CANONICAL_TRAINER_PYTHON" -c '
from pathlib import Path
import hashlib, json, subprocess, sys
repo = Path.cwd()
raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
    raise RuntimeError("native window hash mismatch")
window = json.loads(raw)
if window.get("schema_version") == "gx1_native_direction_prefix_fit_window_policy_v1":
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    window = campaign.read_bound_json(Path(sys.argv[1]), sys.argv[2])
    window = prefix.require_prefix_fit_window_policy(window)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    recipe = prefix.require_prefix_fit_launch_recipe(window["recipe"], source_repo=repo, source_commit=head)
    print(prefix.guarded_prefix_power_limit(recipe))
elif window.get("schema_version") != "gx1_native_direction_research_window_policy_v1":
    print(300)
else:
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    window = campaign.read_bound_json(Path(sys.argv[1]), sys.argv[2])
    window = research.require_research_window_policy(window)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    recipe = research.require_research_launch_recipe(window["recipe"], source_repo=repo, source_commit=head)
    print(research.guarded_research_power_limit(recipe))
' "${target_args[4]}" "${target_args[6]}") || exit 75
    [[ "$research_power_limit" == 250 || "$research_power_limit" == 300 ]] || exit 75
    TRAINER_GPU_MAX_POWER_LIMIT_W="$research_power_limit"
    TRAINER_GPU_MAX_POWER_DRAW_W=310
    TRAINER_GPU_MAX_CORE_TEMP_C=85
    TRAINER_DEVICE=cuda
    TRAINER_OUT_BUNDLE_DIR="${target_args[8]}"
    # C and F claim their fresh output; stdio uses the bound existing guard-log
    # directory so no output/progress parent is created ahead of that claim.
    case "$("$CANONICAL_TRAINER_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["schema_version"])' "${target_args[4]}")" in
      gx1_native_direction_checkpoint_diagnostic_window_policy_v1|gx1_native_direction_frozen_extraction_window_policy_v1)
        TRAINER_OUT_BUNDLE_DIR="$GX1_CAMPAIGN_GUARD_LOG_PATH" ;;
    esac
    return
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
      --checkpoint-dir)
        checkpoint_dir_count=$((checkpoint_dir_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
        TRAINER_OUT_BUNDLE_DIR="${target_args[$((target_index + 1))]}"
        ;;
      --launch-manifest)
        launch_manifest_count=$((launch_manifest_count + 1))
        ;;
      --stage)
        stage_count=$((stage_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
        stage_value="${target_args[$((target_index + 1))]}"
        ;;
      --arm-batch-size)
        arm_batch_size_count=$((arm_batch_size_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
        arm_batch_size_value="${target_args[$((target_index + 1))]}"
        ;;
      --progress-path)
        progress_path_count=$((progress_path_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
        ;;
      --train-session-manifest)
        train_session_count=$((train_session_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
        ;;
      --max-optimizer-steps)
        max_optimizer_steps_count=$((max_optimizer_steps_count + 1))
        (( target_index + 1 < ${#target_args[@]} )) || exit 75
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
    --power-benchmark-scope-json)
      [[ $# -ge 2 && -z "$POWER_BENCHMARK_SCOPE_JSON" && -n "$2" ]] || exit 2
      POWER_BENCHMARK_SCOPE_JSON="$2"; shift 2
      ;;
    --power-benchmark-scope-sha256)
      [[ $# -ge 2 && -z "$POWER_BENCHMARK_SCOPE_SHA256" && -n "$2" ]] || exit 2
      POWER_BENCHMARK_SCOPE_SHA256="$2"; shift 2
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

if [[ -n "$POWER_BENCHMARK_SCOPE_JSON" || -n "$POWER_BENCHMARK_SCOPE_SHA256" ]]; then
  [[ -n "$POWER_BENCHMARK_SCOPE_JSON" && -n "$POWER_BENCHMARK_SCOPE_SHA256" \
    && "$JOB_CLASS" == trainer && "$TRAINER_DEVICE" == cuda \
    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false ]] || {
      echo "FATAL: power comparison requires paired scope arguments and canonical CUDA trainer" >&2
      exit 75
    }
  benchmark_limits=$("$CANONICAL_TRAINER_PYTHON" -I -B "$POWER_BENCHMARK_OWNER" inspect \
    --scope-json "$POWER_BENCHMARK_SCOPE_JSON" --scope-sha256 "$POWER_BENCHMARK_SCOPE_SHA256" \
    -- "$@") || exit 75
  case "$benchmark_limits" in
    '160 170') TRAINER_GPU_MAX_POWER_LIMIT_W=160; TRAINER_GPU_MAX_POWER_DRAW_W=170 ;;
    '200 210') TRAINER_GPU_MAX_POWER_LIMIT_W=200; TRAINER_GPU_MAX_POWER_DRAW_W=210 ;;
    *) echo "FATAL: invalid validated benchmark limits" >&2; exit 75 ;;
  esac
fi

if [[ "$JOB_CLASS" == trainer ]]; then
  CPU_AFFINITY=0-18
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
  fi
  "$CANONICAL_TRAINER_PYTHON" -I -B "$CAPPED_EXECUTION_OWNER" --verify-lock-ancestry || {
    echo "FATAL: nested capped job canonical lock ancestry proof failed" >&2
    exit 75
  }
# BEGIN GX1 FROZEN BIAS ROUTE 20261122 NESTED
  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN" ]] || {
    echo "FATAL: nested frozen bias mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_FUTURE_STATE_BIAS_FROZEN" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 300 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 310 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 85 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 ]] || {
      echo "FATAL: nested frozen bias plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 FROZEN BIAS ROUTE 20261122 NESTED
# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 NESTED
  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" ]] || {
    echo "FATAL: nested bias suffix mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested bias suffix plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 BIAS SUFFIX ROUTE 20261122 003 NESTED
# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 NESTED
  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE" ]] || {
    echo "FATAL: nested bias suffix continuation mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_FUTURE_STATE_BIAS_CONTINUE" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested bias suffix continuation plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 BIAS CONTINUE ROUTE 20261122 004 NESTED
# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-false}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" \
    && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256" \
    && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE" ]] || {
    echo "FATAL: nested row membership qualification mode/plan differs from protected parent" >&2; exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" == true ]]; then
    require_native_provider_hlc_row_membership_qualification_environment true
    [[ "${GX1_CAPPED_CLASS:-}" == trainer \
      && "${GX1_CAPPED_MEMORY_BYTES:-}" == 21474836480 \
      && "${GX1_CAPPED_SWAP_BYTES:-}" == 536870912 \
      && "${GX1_CAPPED_TASKS_MAX:-}" == 128 \
      && "${GX1_CUDA_PRODUCER_GUARD:-}" == false \
      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \
      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" ]] || {
      echo "FATAL: nested row membership qualification containment differs from protected parent" >&2; exit 75
    }
    for profile_state in TRAINER_DEVICE TRAINER_EXECUTION_MODE \
      TRAINER_MAX_WALL_SECONDS TRAINER_MODEL_MAX_WALL_SECONDS TRAINER_ATTENDED_STAGE_REQUIRED \
      TRAINER_GPU_INDEX TRAINER_GPU_MAX_POWER_LIMIT_W TRAINER_GPU_MAX_POWER_DRAW_W \
      TRAINER_GPU_MAX_CORE_TEMP_C TRAINER_GPU_MAX_MEMORY_TEMP_C TRAINER_GPU_MAX_MEMORY_USED_MIB \
      TRAINER_GPU_MONITOR_INTERVAL_SECONDS TRAINER_CPU_MAX_TEMP_C TRAINER_CPU_MAX_PACKAGE_POWER_W \
      TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES; do
      profile_env="GX1_$profile_state"
      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {
        echo "FATAL: nested row membership qualification deadline/hardware differs from protected parent" >&2; exit 75
      }
    done
    for profile_env in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
      [[ "${!profile_env:-}" == 8 ]] || {
        echo "FATAL: nested row membership qualification numerical threads differ from protected parent" >&2; exit 75
      }
    done
    [[ "${ARROW_NUM_THREADS:-}" == 1 && "${POLARS_MAX_THREADS:-}" == 1 ]] || {
      echo "FATAL: nested row membership qualification auxiliary threads differ from protected parent" >&2; exit 75
    }
  fi
# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 NESTED
# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE:-false}" == "$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" \
    && "${GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256" ]] || {
    echo "FATAL: nested gradient profile mode/plan differs from protected parent" >&2; exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" == true ]]; then
    require_native_provider_hlc_gradient_profile_environment true
    [[ "${GX1_CAPPED_CLASS:-}" == trainer \
      && "${GX1_CAPPED_MEMORY_BYTES:-}" == 21474836480 \
      && "${GX1_CAPPED_SWAP_BYTES:-}" == 536870912 \
      && "${GX1_CAPPED_TASKS_MAX:-}" == 128 \
      && "${GX1_CUDA_PRODUCER_GUARD:-}" == false \
      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \
      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" ]] || {
      echo "FATAL: nested gradient profile containment differs from protected parent" >&2; exit 75
    }
    for profile_state in TRAINER_DEVICE TRAINER_EXECUTION_MODE \
      TRAINER_MAX_WALL_SECONDS TRAINER_MODEL_MAX_WALL_SECONDS TRAINER_ATTENDED_STAGE_REQUIRED \
      TRAINER_GPU_INDEX TRAINER_GPU_MAX_POWER_LIMIT_W TRAINER_GPU_MAX_POWER_DRAW_W \
      TRAINER_GPU_MAX_CORE_TEMP_C TRAINER_GPU_MAX_MEMORY_TEMP_C TRAINER_GPU_MAX_MEMORY_USED_MIB \
      TRAINER_GPU_MONITOR_INTERVAL_SECONDS TRAINER_CPU_MAX_TEMP_C TRAINER_CPU_MAX_PACKAGE_POWER_W \
      TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES; do
      profile_env="GX1_$profile_state"
      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {
        echo "FATAL: nested gradient profile deadline/hardware differs from protected parent" >&2; exit 75
      }
    done
    for profile_env in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
      [[ "${!profile_env:-}" == 8 ]] || {
        echo "FATAL: nested gradient profile numerical threads differ from protected parent" >&2; exit 75
      }
    done
    [[ "${ARROW_NUM_THREADS:-}" == 1 && "${POLARS_MAX_THREADS:-}" == 1 ]] || {
      echo "FATAL: nested gradient profile auxiliary threads differ from protected parent" >&2; exit 75
    }
  fi
# END GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 NESTED
# BEGIN GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS:-false}" == "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS" \
    && "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256" ]] || {
    echo "FATAL: nested endpoint cross mode/plan differs from protected parent" >&2; exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS" == true ]]; then
    [[ "${GX1_CAPPED_MEMORY_BYTES:-}" == "21474836480" \
      && "${GX1_CAPPED_SWAP_BYTES:-}" == "536870912" \
      && "${GX1_CAPPED_TASKS_MAX:-}" == "128" \
      && "${OMP_NUM_THREADS:-}" == "8" \
      && "${MKL_NUM_THREADS:-}" == "8" \
      && "${OPENBLAS_NUM_THREADS:-}" == "8" \
      && "${NUMEXPR_NUM_THREADS:-}" == "8" \
      && "${VECLIB_MAXIMUM_THREADS:-}" == "8" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "900" \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "870" \
      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == "false" \
      && "${GX1_TRAINER_GPU_INDEX:-}" == "0" \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == "250" \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == "250" \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == "75" \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == "80" \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == "12288" \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == "1" \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == "80" \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == "125" \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == "8589934592" \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == "8589934592" \
      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \
      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "" \
      && "${GX1_CAPPED_CLASS:-}" == "trainer" \
      && "${GX1_CUDA_PRODUCER_GUARD:-}" == "false" \
      && "${GX1_TRAINER_DEVICE:-}" == "cuda" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == "canonical" \
      && "${BLIS_NUM_THREADS:-}" == "8" \
      && "${ARROW_NUM_THREADS:-}" == "1" \
      && "${POLARS_MAX_THREADS:-}" == "1" \
      && "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS:-}" == "true" \
      && "${GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256" ]] || {
      echo "FATAL: nested endpoint cross resources or legacy modes differ from protected parent" >&2; exit 75
    }
  fi
# END GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 NESTED
# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \
    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" ]] || {
    echo "FATAL: nested ALL-TF learning mode/plan differs from protected parent" >&2; exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" == true ]]; then
    [[ "${GX1_CAPPED_MEMORY_BYTES:-}" == "21474836480" \
      && "${GX1_CAPPED_SWAP_BYTES:-}" == "536870912" \
      && "${GX1_CAPPED_TASKS_MAX:-}" == "128" \
      && "${OMP_NUM_THREADS:-}" == "8" \
      && "${MKL_NUM_THREADS:-}" == "8" \
      && "${OPENBLAS_NUM_THREADS:-}" == "8" \
      && "${NUMEXPR_NUM_THREADS:-}" == "8" \
      && "${VECLIB_MAXIMUM_THREADS:-}" == "8" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-}" == "false" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-}" == "false" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "12000" \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "11970" \
      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == "false" \
      && "${GX1_TRAINER_GPU_INDEX:-}" == "0" \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == "250" \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == "250" \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == "75" \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == "80" \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == "12288" \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == "1" \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == "80" \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == "125" \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == "8589934592" \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == "8589934592" \
      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \
      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-}" == "true" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" \
      && "${GX1_CAPPED_CLASS:-}" == "trainer" \
      && "${GX1_CUDA_PRODUCER_GUARD:-}" == "false" \
      && "${GX1_TRAINER_DEVICE:-}" == "cuda" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == "canonical" \
      && "${BLIS_NUM_THREADS:-}" == "8" \
      && "${ARROW_NUM_THREADS:-}" == "1" \
      && "${POLARS_MAX_THREADS:-}" == "1" ]] || {
      echo "FATAL: nested ALL-TF learning resources or mixed modes differ from protected parent" >&2; exit 75
    }
  fi
# END GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED
# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-false}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME" \
    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" ]] || {
    echo "FATAL: nested provider HLC learning resume mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_LEARNING_RESUME" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == false \
      && -z "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" \
      && -z "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 12000 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 11970 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC learning resume plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 NESTED
# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == "$NATIVE_PROVIDER_HLC_LEARNING" ]] || {
    echo "FATAL: nested provider HLC learning mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_LEARNING" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == false \
      && -z "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" \
      && -z "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 12000 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 11970 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC learning plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 NESTED
# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == "$NATIVE_PROVIDER_HLC_REPLAY" ]] || {
    echo "FATAL: nested provider HLC replay mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_REPLAY" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_LANE" \
      && ( "$NATIVE_PROVIDER_HLC_REPLAY_LANE" == warm || "$NATIVE_PROVIDER_HLC_REPLAY_LANE" == cold ) \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 1800 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 1770 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC replay plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 NESTED
# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED
# BEGIN GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" ]] || {
    echo "FATAL: nested provider HLC INITIAL mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC INITIAL plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED
# BEGIN GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" ]] || {
    echo "FATAL: nested provider HLC replay mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \
      && ( "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == warm || "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == cold ) \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \
      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == false \
      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 1800 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 1770 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC replay plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED
# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED
# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED
  [[ "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == "$NATIVE_PROVIDER_HLC_INITIAL" ]] || {
    echo "FATAL: nested provider HLC INITIAL mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_PROVIDER_HLC_INITIAL" == true ]]; then
    [[ "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \
      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \
      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested provider HLC INITIAL plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED
# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 NESTED
  [[ "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" ]] || {
    echo "FATAL: nested full TRAIN capture mode differs from protected parent" >&2
    exit 75
  }
  if [[ "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \
      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \
      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \
      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \
      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \
      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {
      echo "FATAL: nested full TRAIN capture plan/deadline/hardware differs from protected parent" >&2
      exit 75
    }
  fi
# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 NESTED
# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED
  [[ "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == "$NATIVE_FUTURE_STATE_THROUGHPUT" \
    && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == "$NATIVE_FUTURE_STATE_TRAINING" ]] || {
    echo "FATAL: nested native future-state leaf mode differs from parent" >&2
    exit 75
  }
  if [[ "$NATIVE_FUTURE_STATE_THROUGHPUT" == true || "$NATIVE_FUTURE_STATE_TRAINING" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "$NATIVE_FUTURE_STATE_TRAINING_LANE" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "$TRAINER_MAX_WALL_SECONDS" \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "$TRAINER_MODEL_MAX_WALL_SECONDS" \
      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 300 \
      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 310 \
      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 85 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \
      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 ]] || {
      echo "FATAL: nested native future-state leaf plan/deadline/hardware differs from parent" >&2
      exit 75
    }
  fi
# END GX1 FUTURE LEAF ROUTES 20261121 NESTED
  if [[ "$NATIVE_FUTURE_STATE_CUDA_QUALIFICATION" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-}" == true \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "$NATIVE_FUTURE_STATE_CUDA_LANE" \
      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \
      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "$TRAINER_MAX_WALL_SECONDS" \
      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "$TRAINER_MODEL_MAX_WALL_SECONDS" ]] || {
      echo "FATAL: nested native CUDA qualification differs from protected parent mode" >&2
      exit 75
    }
  fi
  if [[ "$JOB_CLASS" == trainer || "$CUDA_PRODUCER_GUARD" == true ]]; then
    exec "$GPU_GUARD_PATH" "$@"
  fi
  if [[ "$NATIVE_FUTURE_STATE_CPU_MECHANICS" == true ]]; then
    [[ "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-}" == true \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256" \
      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "$NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE" ]] || {
      echo "FATAL: nested native CPU mechanics differs from protected parent mode" >&2
      exit 75
    }
    exec /usr/bin/timeout --signal=TERM --kill-after=10s 900s "$@"
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
  for host_limit in "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" \
    "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES" "$TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES"; do
    [[ "$host_limit" =~ ^[1-9][0-9]*$ ]] || {
      echo "FATAL: protected host resource limit is malformed" >&2
      exit 75
    }
  done
  [[ "$TRAINER_HOST_TELEMETRY_CERT_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$TRAINER_HOST_TELEMETRY_GPU_UUID" =~ ^GPU-[0-9a-fA-F-]{36}$ \
    && "$TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS" =~ ^[1-5]$ ]] || {
    echo "FATAL: signed host telemetry source binding is malformed" >&2
    exit 75
  }
fi

[[ -x "$CANONICAL_TRAINER_PYTHON" && -f "$CAPPED_EXECUTION_OWNER" ]] || {
  echo "FATAL: canonical heavy-job lock owner is unavailable" >&2
  exit 75
}
LOCK_PATH="$("$CANONICAL_TRAINER_PYTHON" -I -B "$CAPPED_EXECUTION_OWNER" --lock-path)" || exit 2
LOCK_UMASK=$(umask)
umask 077
exec 9>>"$LOCK_PATH"
umask "$LOCK_UMASK"
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
  if [[ -n "${GX1_CAMPAIGN_GUARD_LOG_PATH:-}" ]]; then
    [[ "$GX1_CAMPAIGN_GUARD_LOG_PATH" == /*       && -d "${GX1_CAMPAIGN_GUARD_LOG_PATH%/*}"       && ! -L "${GX1_CAMPAIGN_GUARD_LOG_PATH%/*}"       && ! -e "$GX1_CAMPAIGN_GUARD_LOG_PATH"       && ! -L "$GX1_CAMPAIGN_GUARD_LOG_PATH" ]] || {
      echo "FATAL: campaign guard-log path is not a fresh exact regular path" >&2
      exit 75
    }
    if ! (set -o noclobber; umask 077; : > "$GX1_CAMPAIGN_GUARD_LOG_PATH"); then
      echo "FATAL: could not create exclusive campaign guard log" >&2
      exit 75
    fi
    TRAINER_GUARD_LOG_PATH="$GX1_CAMPAIGN_GUARD_LOG_PATH"
  else
    TRAINER_GUARD_LOG_PATH=$(
      /usr/bin/mktemp "${TRAINER_GUARD_LOG_PARENT}/.${TRAINER_GUARD_LOG_BASENAME}.guard.XXXXXXXX.log"
    ) || {
      echo "FATAL: could not create exclusive trainer guard log" >&2
      exit 75
    }
  fi
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
"$1" -I -B "$2" --verify-lock-ancestry || { echo "FATAL: scope canonical lock ancestry proof failed" >&2; exit 75; }
shift 2
echo "[capped_run_scope_verified] memory.max=$GX1_EXPECTED_MEMORY_BYTES memory.high=$GX1_EXPECTED_MEMORY_BYTES memory.swap.max=$GX1_EXPECTED_SWAP_BYTES pids.max=$GX1_EXPECTED_TASKS" >&2
verified_cpu_affinity="$GX1_CPU_AFFINITY"
unset GX1_EXPECTED_MEMORY_BYTES GX1_EXPECTED_SWAP_BYTES GX1_EXPECTED_TASKS GX1_CPU_AFFINITY
if [[ "$GX1_CAPPED_CLASS" == trainer || "$GX1_CUDA_PRODUCER_GUARD" == true ]]; then
  [[ -x "$GX1_GPU_GUARD_PATH" ]] || { echo "FATAL: guarded CUDA safety owner unavailable inside scope" >&2; exit 75; }
  if [[ "$GX1_CAPPED_CLASS" == trainer ]]; then
    exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 2 -n 4 /usr/bin/nice -n 0 "$GX1_GPU_GUARD_PATH" "$@"
  fi
  exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 "$GX1_GPU_GUARD_PATH" "$@"
fi
if [[ "$GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS" == true ]]; then
  exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 \
    /usr/bin/timeout --signal=TERM --kill-after=10s 900s "$@"
fi
if [[ "$GX1_H4_LINE_AUDIT_GUARD" == true ]]; then
  exec /usr/bin/taskset -c "$verified_cpu_affinity" /usr/bin/ionice -c 3 /usr/bin/nice -n 10 \
    /usr/bin/timeout --signal=TERM --kill-after=10s 1800s "$@"
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
  --setenv=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS="$NATIVE_FUTURE_STATE_CPU_MECHANICS" \
  --setenv=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256="$NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE="$NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE" \
  --setenv=GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION="$NATIVE_FUTURE_STATE_CUDA_QUALIFICATION" \
  --setenv=GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256="$NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_CUDA_LANE="$NATIVE_FUTURE_STATE_CUDA_LANE" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN="$NATIVE_FUTURE_STATE_BIAS_FROZEN" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX="$NATIVE_FUTURE_STATE_BIAS_SUFFIX" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE="$NATIVE_FUTURE_STATE_BIAS_CONTINUE" \
  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256" \
  --setenv=GX1_NATIVE_V3_DEVELOPMENT="$NATIVE_V3_DEVELOPMENT" \
  --setenv=GX1_NATIVE_V3_DEVELOPMENT_PLAN_SHA256="$NATIVE_V3_DEVELOPMENT_PLAN_SHA256" \
  --setenv=GX1_NATIVE_V3_DEVELOPMENT_PHASE="$NATIVE_V3_DEVELOPMENT_PHASE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME="$NATIVE_PROVIDER_HLC_LEARNING_RESUME" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256="$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING="$NATIVE_PROVIDER_HLC_LEARNING" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256="$NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY="$NATIVE_PROVIDER_HLC_REPLAY" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256="$NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE="$NATIVE_PROVIDER_HLC_REPLAY_LANE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL="$NATIVE_PROVIDER_HLC_INITIAL" \
  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256="$NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE="$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" \
  --setenv=GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256="$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \
  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256="$NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING="$NATIVE_FUTURE_STATE_TRAINING" \
  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256="$NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256" \
  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING_LANE="$NATIVE_FUTURE_STATE_TRAINING_LANE" \
  --setenv=GX1_POWER_BENCHMARK_SCOPE_JSON="$POWER_BENCHMARK_SCOPE_JSON" \
  --setenv=GX1_POWER_BENCHMARK_SCOPE_SHA256="$POWER_BENCHMARK_SCOPE_SHA256" \
  --setenv=GX1_CAPPED_MEMORY_BYTES="$((requested_mem_kib * 1024))" \
  --setenv=GX1_CAPPED_SWAP_BYTES="$((requested_swap_kib * 1024))" \
  --setenv=GX1_CAPPED_TASKS_MAX="$TASKS_MAX" \
  --setenv=GX1_GPU_GUARD_PATH="$GPU_GUARD_PATH" \
  --setenv=GX1_CUDA_PRODUCER_GUARD="$CUDA_PRODUCER_GUARD" \
  --setenv=GX1_H4_LINE_AUDIT_GUARD="$H4_LINE_AUDIT_GUARD" \
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
  --setenv=GX1_TRAINER_CPU_MAX_TEMP_C="$TRAINER_CPU_MAX_TEMP_C" \
  --setenv=GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W="$TRAINER_CPU_MAX_PACKAGE_POWER_W" \
  --setenv=GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES="$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES" \
  --setenv=GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES="$TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH="$TRAINER_HOST_TELEMETRY_QUERY_PATH" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_URL="$TRAINER_HOST_TELEMETRY_URL" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_CERT_PATH="$TRAINER_HOST_TELEMETRY_CERT_PATH" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256="$TRAINER_HOST_TELEMETRY_CERT_SHA256" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_GPU_UUID="$TRAINER_HOST_TELEMETRY_GPU_UUID" \
  --setenv=GX1_TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS="$TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS" \
  --setenv=GX1_CAMPAIGN_PLAN_SHA256="${GX1_CAMPAIGN_PLAN_SHA256:-}" \
  --setenv=GX1_CAMPAIGN_PLAN_PATH="${GX1_CAMPAIGN_PLAN_PATH:-}" \
  --setenv=GX1_CAMPAIGN_PLAN_FILE_SHA256="${GX1_CAMPAIGN_PLAN_FILE_SHA256:-}" \
  --setenv=GX1_CAMPAIGN_INVOCATION_SHA256="${GX1_CAMPAIGN_INVOCATION_SHA256:-}" \
  --setenv=OMP_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=MKL_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=OPENBLAS_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=NUMEXPR_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=VECLIB_MAXIMUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=BLIS_NUM_THREADS="$NUMERICAL_THREAD_COUNT" \
  --setenv=ARROW_NUM_THREADS=1 \
  --setenv=POLARS_MAX_THREADS=1 \
  --setenv=MALLOC_ARENA_MAX=2 \
  -- /bin/bash -c "$SCOPE_GUARD" gx1-capped-scope "$CANONICAL_TRAINER_PYTHON" "$CAPPED_EXECUTION_OWNER" "$@"
exit_code=$?
flock -u 9
exit "$exit_code"
