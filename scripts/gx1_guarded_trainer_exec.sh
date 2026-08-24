#!/usr/bin/env bash
# Fail-closed wall-clock and GPU safety owner for the canonical trainer.
# This script is entered only by gx1_capped_run.sh after that runner has
# validated the exact trainer target and proved the enclosing cgroup limits.
set -euo pipefail

die() {
  printf 'FATAL: trainer safety guard: %s\n' "$*" >&2
  exit 75
}

require_uint() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] \
    || die "$name must be a positive integer"
}

for variable in \
  GX1_CAPPED_CLASS \
  GX1_CAPPED_MEMORY_BYTES \
  GX1_CAPPED_SWAP_BYTES \
  GX1_CAPPED_TASKS_MAX \
  GX1_TRAINER_DEVICE \
  GX1_TRAINER_EXECUTION_MODE \
  GX1_TRAINER_MAX_WALL_SECONDS \
  GX1_TRAINER_MODEL_MAX_WALL_SECONDS \
  GX1_TRAINER_ATTENDED_STAGE_REQUIRED \
  GX1_TRAINER_GPU_INDEX \
  GX1_TRAINER_GPU_MAX_CORE_TEMP_C \
  GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C \
  GX1_TRAINER_GPU_MAX_POWER_LIMIT_W \
  GX1_TRAINER_GPU_MAX_POWER_DRAW_W \
  GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS \
  GX1_TRAINER_NVIDIA_SMI_PATH; do
  [[ -n "${!variable:-}" ]] || die "missing protected environment: $variable"
done
[[ "$GX1_CAPPED_CLASS" == trainer ]] \
  || die "requires the canonical trainer cgroup"
case "$GX1_TRAINER_DEVICE" in
  cpu|cuda) ;;
  *) die "GX1_TRAINER_DEVICE must be cpu or cuda" ;;
esac
case "$GX1_TRAINER_EXECUTION_MODE" in
  canonical|attended_smoke|research_smoke) ;;
  *) die "GX1_TRAINER_EXECUTION_MODE must be canonical, attended_smoke, or research_smoke" ;;
esac
case "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" in
  true|false) ;;
  *) die "GX1_TRAINER_ATTENDED_STAGE_REQUIRED must be true or false" ;;
esac
if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true \
  && "$GX1_TRAINER_EXECUTION_MODE" != attended_smoke ]]; then
  die "staged attended preflight requires attended_smoke execution mode"
fi
for variable in \
  GX1_CAPPED_MEMORY_BYTES \
  GX1_CAPPED_SWAP_BYTES \
  GX1_CAPPED_TASKS_MAX \
  GX1_TRAINER_MAX_WALL_SECONDS \
  GX1_TRAINER_MODEL_MAX_WALL_SECONDS \
  GX1_TRAINER_GPU_MAX_CORE_TEMP_C \
  GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C \
  GX1_TRAINER_GPU_MAX_POWER_LIMIT_W \
  GX1_TRAINER_GPU_MAX_POWER_DRAW_W \
  GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS; do
  require_uint "$variable" "${!variable}"
done
[[ "$GX1_TRAINER_GPU_INDEX" =~ ^[0-9]+$ ]] \
  || die "GX1_TRAINER_GPU_INDEX must be a non-negative integer"
[[ $# -ge 1 ]] || die "missing canonical trainer command"

cgroup_relative=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
cgroup_directory="/sys/fs/cgroup${cgroup_relative}"
[[ -d "$cgroup_directory" ]] || die "trainer cgroup is unavailable"
[[ "$(<"$cgroup_directory/memory.max")" == "$GX1_CAPPED_MEMORY_BYTES" \
  && "$(<"$cgroup_directory/memory.high")" == "$GX1_CAPPED_MEMORY_BYTES" \
  && "$(<"$cgroup_directory/memory.swap.max")" == "$GX1_CAPPED_SWAP_BYTES" \
  && "$(<"$cgroup_directory/pids.max")" == "$GX1_CAPPED_TASKS_MAX" ]] \
  || die "trainer cgroup proof does not match protected environment"

for helper in /usr/bin/setsid /usr/bin/timeout /bin/kill /bin/date /bin/sleep /usr/bin/mktemp /usr/bin/mkfifo /usr/bin/od /usr/bin/tr /bin/rm /bin/rmdir; do
  [[ -x "$helper" ]] || die "required helper is unavailable: $helper"
done

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

telemetry_values=()
read_gpu_telemetry() {
  local output core_temp memory_temp power_draw power_limit extra memory_observed
  [[ -x "$GX1_TRAINER_NVIDIA_SMI_PATH" ]] || return 1
  output=$(
    /usr/bin/timeout --signal=KILL 5s \
      "$GX1_TRAINER_NVIDIA_SMI_PATH" \
      --id="$GX1_TRAINER_GPU_INDEX" \
      --query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit \
      --format=csv,noheader,nounits 2>/dev/null
  ) || return 1
  IFS=, read -r core_temp memory_temp power_draw power_limit extra <<<"$output"
  [[ -z "${extra:-}" ]] || return 1
  core_temp=$(trim "${core_temp:-}")
  memory_temp=$(trim "${memory_temp:-}")
  power_draw=$(trim "${power_draw:-}")
  power_limit=$(trim "${power_limit:-}")
  [[ "$core_temp" =~ ^[0-9]+([.][0-9]+)?$ \
    && "$power_draw" =~ ^[0-9]+([.][0-9]+)?$ \
    && "$power_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  memory_observed=true
  if [[ "$memory_temp" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    :
  elif { [[ "$GX1_TRAINER_EXECUTION_MODE" == attended_smoke ]] \
    || [[ "$GX1_TRAINER_EXECUTION_MODE" == research_smoke ]]; } \
    && [[ "$memory_temp" == N/A ]]; then
    # WSL's own driver reports literal N/A for this sensor.  This exception is
    # deliberately unavailable to canonical training.  The only two admitted
    # uses are bounded attended smoke and the source-bound research-smoke
    # route, both of which retain the one-second core/actual-draw stops.
    memory_observed=false
  else
    return 1
  fi
  telemetry_values=("$core_temp" "$memory_temp" "$power_draw" "$power_limit" "$memory_observed")
}

float_gt() {
  awk -v left="$1" -v right="$2" 'BEGIN { exit !(left > right) }'
}

assert_safe_telemetry() {
  local phase="$1" core_temp memory_temp power_draw power_limit memory_observed
  read_gpu_telemetry \
    || die "CUDA telemetry unavailable during $phase; refusing unmonitored load"
  core_temp=${telemetry_values[0]}
  memory_temp=${telemetry_values[1]}
  power_draw=${telemetry_values[2]}
  power_limit=${telemetry_values[3]}
  memory_observed=${telemetry_values[4]}
  float_gt "$power_limit" "$GX1_TRAINER_GPU_MAX_POWER_LIMIT_W" \
    && die "configured GPU power limit ${power_limit}W exceeds ${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W}W during $phase"
  float_gt "$core_temp" "$GX1_TRAINER_GPU_MAX_CORE_TEMP_C" \
    && die "GPU core temperature ${core_temp}C exceeds ${GX1_TRAINER_GPU_MAX_CORE_TEMP_C}C during $phase"
  if [[ "$memory_observed" == true ]]; then
    float_gt "$memory_temp" "$GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C" \
      && die "GPU memory temperature ${memory_temp}C exceeds ${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C}C during $phase"
  fi
  # Both independently observed power conditions are mandatory: canonical
  # runs require a 250 W configured ceiling, while attended smoke additionally
  # keeps the same 250 W actual-draw stop despite its observed 390 W setting.
  float_gt "$power_draw" "$GX1_TRAINER_GPU_MAX_POWER_DRAW_W" \
    && die "GPU draw ${power_draw}W exceeds ${GX1_TRAINER_GPU_MAX_POWER_DRAW_W}W during $phase"
  return 0
}

child_pid=
stage_dir=
stage_fifo=
stage_fd=
stage_token=
stage_name=canonical
stage_start_epoch=

cleanup_stage_notification() {
  if [[ -n "$stage_fd" ]]; then
    exec {stage_fd}>&- || true
    stage_fd=
  fi
  if [[ -n "$stage_fifo" ]]; then
    /bin/rm -f -- "$stage_fifo" 2>/dev/null || true
    stage_fifo=
  fi
  if [[ -n "$stage_dir" ]]; then
    /bin/rmdir -- "$stage_dir" 2>/dev/null || true
    stage_dir=
  fi
}

create_stage_notification() {
  stage_dir=$(/usr/bin/mktemp -d /tmp/gx1-attended-stage.XXXXXXXX) \
    || die "could not create private attended-stage directory"
  stage_fifo="$stage_dir/preflight-ready"
  /usr/bin/mkfifo -m 600 "$stage_fifo" \
    || die "could not create private attended-stage pipe"
  stage_token=$(
    /usr/bin/od -An -N32 -tx1 /dev/urandom | /usr/bin/tr -d ' \n'
  ) || die "could not generate attended-stage token"
  [[ "$stage_token" =~ ^[0-9a-f]{64}$ ]] \
    || die "generated attended-stage token is invalid"
  exec {stage_fd}<>"$stage_fifo" \
    || die "could not open private attended-stage pipe"
  export GX1_TRAINER_ATTENDED_STAGE_FIFO="$stage_fifo"
  export GX1_TRAINER_ATTENDED_STAGE_TOKEN="$stage_token"
}

stage_notification_error=
consume_stage_notifications() {
  local message
  while IFS= read -r -t 0.01 -u "$stage_fd" message; do
    if [[ "$stage_name" == data_preflight \
      && "$message" == "gx1_attended_preflight_ready_v1:$stage_token" ]]; then
      stage_name=model_smoke
      stage_start_epoch=$(/bin/date +%s)
      printf '[trainer_safety_stage_transition] from=data_preflight to=model_smoke preflight_elapsed_seconds=%s model_max_wall_seconds=%s\n' \
        "$((stage_start_epoch - start_epoch))" \
        "$GX1_TRAINER_MODEL_MAX_WALL_SECONDS" >&2
      continue
    fi
    stage_notification_error=invalid_attended_stage_notification
    return 1
  done
  return 0
}

terminate_child_group() {
  local reason="$1"
  [[ -n "$child_pid" ]] || return 0
  if /bin/kill -0 "$child_pid" 2>/dev/null; then
    printf '[trainer_safety_stop] reason=%s pid=%s\n' "$reason" "$child_pid" >&2
    /bin/kill -TERM -- "-$child_pid" 2>/dev/null || true
    for _ in {1..10}; do
      /bin/kill -0 "$child_pid" 2>/dev/null || return 0
      /bin/sleep 1
    done
    /bin/kill -KILL -- "-$child_pid" 2>/dev/null || true
  fi
}

trap 'terminate_child_group guard_exit; cleanup_stage_notification' EXIT
trap 'terminate_child_group signal; cleanup_stage_notification; exit 130' INT TERM HUP

if [[ "$GX1_TRAINER_DEVICE" == cuda ]]; then
  assert_safe_telemetry preflight
fi

printf '[trainer_safety_guard] execution_mode=%s device=%s data_preflight_max_wall_seconds=%s model_max_wall_seconds=%s attended_stage_required=%s gpu_index=%s max_core_temp_c=%s max_memory_temp_c=%s max_power_limit_w=%s max_power_draw_w=%s monitor_interval_seconds=%s\n' \
  "$GX1_TRAINER_EXECUTION_MODE" \
  "$GX1_TRAINER_DEVICE" \
  "$GX1_TRAINER_MAX_WALL_SECONDS" \
  "$GX1_TRAINER_MODEL_MAX_WALL_SECONDS" \
  "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" \
  "$GX1_TRAINER_GPU_INDEX" \
  "$GX1_TRAINER_GPU_MAX_CORE_TEMP_C" \
  "$GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C" \
  "$GX1_TRAINER_GPU_MAX_POWER_LIMIT_W" \
  "$GX1_TRAINER_GPU_MAX_POWER_DRAW_W" \
  "$GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS" >&2
if [[ "$GX1_TRAINER_EXECUTION_MODE" == attended_smoke ]]; then
  printf '[trainer_safety_attended_only] WSL VRAM telemetry may be literal N/A; this run has no candidate, TEST, promotion, or live authority\n' >&2
fi
if [[ "$GX1_TRAINER_EXECUTION_MODE" == research_smoke ]]; then
  printf '[trainer_safety_research_smoke] WSL VRAM telemetry may be literal N/A; this run is smoke-only and has no candidate, TEST, promotion, paper, or live authority\n' >&2
fi

start_epoch=$(/bin/date +%s)
stage_start_epoch=$start_epoch
if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then
  stage_name=data_preflight
  create_stage_notification
  printf '[trainer_safety_staged_preflight] data_preflight_max_wall_seconds=%s model_max_wall_seconds=%s transition=guard_verified_private_pipe\n' \
    "$GX1_TRAINER_MAX_WALL_SECONDS" \
    "$GX1_TRAINER_MODEL_MAX_WALL_SECONDS" >&2
else
  unset GX1_TRAINER_ATTENDED_STAGE_FIFO GX1_TRAINER_ATTENDED_STAGE_TOKEN
fi
/usr/bin/setsid "$@" &
child_pid=$!
last_heartbeat_epoch=$start_epoch

while /bin/kill -0 "$child_pid" 2>/dev/null; do
  /bin/sleep "$GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS"
  now_epoch=$(/bin/date +%s)
  stage_elapsed=$((now_epoch - stage_start_epoch))
  if [[ "$stage_name" == model_smoke ]]; then
    stage_limit=$GX1_TRAINER_MODEL_MAX_WALL_SECONDS
  else
    stage_limit=$GX1_TRAINER_MAX_WALL_SECONDS
  fi
  if (( stage_elapsed >= stage_limit )); then
    if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then
      stop_reason="stage_${stage_name}_wall_clock_limit_${stage_limit}s"
    else
      stop_reason="wall_clock_limit_${stage_limit}s"
    fi
    terminate_child_group "$stop_reason"
    wait "$child_pid" 2>/dev/null || true
    child_pid=
    die "wall-clock limit reached during stage=$stage_name"
  fi
  if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then
    if ! consume_stage_notifications; then
      terminate_child_group "$stage_notification_error"
      wait "$child_pid" 2>/dev/null || true
      child_pid=
      die "invalid attended-stage notification"
    fi
  fi
  /bin/kill -0 "$child_pid" 2>/dev/null || break
  if [[ "$GX1_TRAINER_DEVICE" == cuda ]]; then
    if ! read_gpu_telemetry; then
      terminate_child_group telemetry_unavailable
      wait "$child_pid" 2>/dev/null || true
      child_pid=
      die "CUDA telemetry became unavailable"
    fi
    core_temp=${telemetry_values[0]}
    memory_temp=${telemetry_values[1]}
    power_draw=${telemetry_values[2]}
    power_limit=${telemetry_values[3]}
    memory_observed=${telemetry_values[4]}
    breach=
    float_gt "$power_limit" "$GX1_TRAINER_GPU_MAX_POWER_LIMIT_W" && breach=power_limit
    float_gt "$power_draw" "$GX1_TRAINER_GPU_MAX_POWER_DRAW_W" && breach=power_draw
    float_gt "$core_temp" "$GX1_TRAINER_GPU_MAX_CORE_TEMP_C" && breach=core_temperature
    if [[ "$memory_observed" == true ]]; then
      float_gt "$memory_temp" "$GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C" && breach=memory_temperature
    fi
    if [[ -n "$breach" ]]; then
      terminate_child_group "$breach"
      wait "$child_pid" 2>/dev/null || true
      child_pid=
      die "GPU safety threshold breached: $breach"
    fi
  fi
  if (( now_epoch - last_heartbeat_epoch >= 30 )); then
    if [[ "$GX1_TRAINER_DEVICE" == cuda ]]; then
      printf '[trainer_safety_heartbeat] stage=%s stage_elapsed_seconds=%s core_temp_c=%s memory_temp_c=%s memory_observed=%s power_draw_w=%s power_limit_w=%s\n' \
        "$stage_name" "$stage_elapsed" "$core_temp" "$memory_temp" "$memory_observed" "$power_draw" "$power_limit" >&2
    else
      printf '[trainer_safety_heartbeat] stage=%s stage_elapsed_seconds=%s device=cpu\n' "$stage_name" "$stage_elapsed" >&2
    fi
    last_heartbeat_epoch=$now_epoch
  fi
done

if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then
  if ! consume_stage_notifications; then
    terminate_child_group "$stage_notification_error"
    wait "$child_pid" 2>/dev/null || true
    child_pid=
    die "invalid attended-stage notification"
  fi
fi

set +e
wait "$child_pid"
child_status=$?
set -e
child_pid=
if [[ "$GX1_TRAINER_ATTENDED_STAGE_REQUIRED" == true \
  && "$stage_name" == data_preflight ]]; then
  cleanup_stage_notification
  trap - EXIT
  die "attended data preflight exited without its required model-stage marker"
fi
cleanup_stage_notification
trap - EXIT
exit "$child_status"
