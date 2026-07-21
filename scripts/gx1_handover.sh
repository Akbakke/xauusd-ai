#!/usr/bin/env bash
# Compact, read-only takeover viewer for the XAUUSD direction-repair handover.
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
LAUNCH_STATE="$REPO/PROJECT_STATE_xau_direction_launch.json"
PY="$REPO/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/gx1_handover.sh [--check|--verbose]

Default: compact, hash-bound status. --check prints only the deterministic
authority fingerprint and minimal source state. --verbose additionally prints
the full handover document and raw Python process table.
EOF
}

mode=compact
case "${1:-}" in
  "") ;;
  --check) mode=check ;;
  --verbose) mode=verbose ;;
  -h|--help) usage; exit 0 ;;
  *) printf 'FATAL: unsupported argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "FATAL: expected at most one argument" >&2; exit 2; }

sources=(
  "$REPO/AGENTS.md"
  "$REPO/ROADMAP.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/PROJECT_STATE.md"
  "$REPO/DECISION_LOG.md"
  "$REPO/PROJECT_STATE_artifacts.json"
  "$REPO/PROJECT_STATE_entry_iql_delete_incident.json"
  "$LAUNCH_STATE"
)
for source in "${sources[@]}"; do
  [[ -f "$source" ]] || { echo "FATAL: authoritative input missing: $source" >&2; exit 2; }
done
[[ -x "$PY" ]] || { echo "FATAL: repository Python is not executable: $PY" >&2; exit 2; }
cd "$REPO"

if [[ "$mode" == "check" ]]; then
  mapfile -t git_lines < <(git status --short --untracked-files=all)
  "$PY" - "${sources[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


paths = tuple(Path(raw) for raw in sys.argv[1:])
digest = hashlib.sha256()
digest.update(b"gx1-takeover-authority-v1\0")
for index, path in enumerate(paths):
    path_bytes = str(path).encode("utf-8")
    payload = path.read_bytes()
    digest.update(index.to_bytes(4, "big"))
    digest.update(len(path_bytes).to_bytes(8, "big"))
    digest.update(path_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)

try:
    state = json.loads(paths[-1].read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"FATAL: malformed launch authority: {exc}") from exc
if not isinstance(state, dict):
    raise SystemExit("FATAL: malformed launch authority: root must be an object")
for key in ("decision", "updated_utc"):
    if not isinstance(state.get(key), str) or not state[key]:
        raise SystemExit(f"FATAL: malformed launch authority: missing {key}")

print("mode: check")
print(f"authority_fingerprint: {digest.hexdigest()}")
print(f"decision: {state['decision']}")
print(f"updated_utc: {state['updated_utc']}")
PY
  echo "head_commit: $(git rev-parse HEAD)"
  printf 'changed_path_count: %d\n' "${#git_lines[@]}"
  exit 0
fi

echo "# GX1 XAU Direction Repair Takeover (compact)"
echo "mode: $mode"
echo "full_view_command: bash $REPO/scripts/gx1_handover.sh --verbose"
echo
echo "## Goal"
echo "Build the GX1 trading bot for gold/XAUUSD with one learned full-stack path"
echo "that selects LONG/SHORT/FLAT direction. No fallback, live hand-rules, stale"
echo "artifact authority or soft pass-through is allowed; no competing decision path exists."
echo "Near-perfect practical precision is a target, not a current claim."
echo
echo "## Authoritative sources (read in this order)"
for source in "${sources[@]}"; do
  printf '%s  %s\n' "$(sha256sum -- "$source" | cut -d' ' -f1)" "$source"
done
echo "Use this script only: scripts/gx1_handover.sh"
echo
echo "## Launch authority"
"$PY" - "$LAUNCH_STATE" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in (
    "decision", "updated_utc", "latest_terminal_event_id",
    "latest_terminal_event_decision", "required_contract_mode",
    "accepted_bundle_dir", "bundle_metadata_sha256",
):
    value = state.get(key)
    print(f"{key}: {'NONE' if value is None else value}")
print(
    "dimensions: "
    + " ".join(
        f"{label}={state.get(key, 'MISSING')}"
        for label, key in (
            ("signal", "required_signal_dim"),
            ("base", "required_base_signal_dim"),
            ("specialist", "required_selected_feature_count"),
            ("mandatory", "required_mandatory_causal_layer_feature_count"),
            ("train_ranked", "required_train_ranked_remainder_feature_count"),
            ("ctx_cont", "required_ctx_cont_dim"),
            ("ctx_cat", "required_ctx_cat_dim"),
        )
    )
)
blockers = state.get("blockers")
print(f"blocker_count: {len(blockers) if isinstance(blockers, list) else 'INVALID'}")
PY
echo
echo "## Source worktree"
echo "head_commit: $(git rev-parse HEAD)"
mapfile -t git_lines < <(git status --short --untracked-files=all)
printf 'changed_path_count: %d\n' "${#git_lines[@]}"
git_limit=24
for ((i = 0; i < ${#git_lines[@]} && i < git_limit; i++)); do
  printf '%s\n' "${git_lines[$i]}"
done
if (( ${#git_lines[@]} > git_limit )); then
  printf '... %d more; run git status --short explicitly\n' "$(( ${#git_lines[@]} - git_limit ))"
fi
echo
echo "## Host capacity"
echo "gx1_data_available_bytes: $(df -B1 --output=avail /home/andre2/GX1_DATA | awk 'NR == 2 {print $1}')"
echo "memory_available_bytes: $(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)"
echo "swap_free_bytes: $(awk '/^SwapFree:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)"
echo
echo "## Active GX1 process groups"
declare -A process_count=() process_pids=() process_ppids=() process_states=()
declare -a chain_specs=()
while read -r pid ppid state _cpu _mem _elapsed args; do
  executable="${args%% *}"
  executable="${executable##*/}"
  case "$executable" in
    python|python3|python3.*|bash|sh) ;;
    *) continue ;;
  esac
  identity=""
  if [[ "$args" =~ [[:space:]]-m[[:space:]]+(gx1\.[[:alnum:]_.]+) ]]; then
    identity="python -m ${BASH_REMATCH[1]}"
  elif [[ "$args" =~ $REPO/((gx1|scripts)/[^[:space:]]+\.(py|sh)) ]]; then
    identity="${BASH_REMATCH[1]}"
  elif [[ "$args" =~ ((gx1|scripts)/[^[:space:]]+\.(py|sh)) ]]; then
    identity="${BASH_REMATCH[1]}"
  fi
  if [[ "$executable" == "bash" || "$executable" == "sh" ]]; then
    case "$identity" in
      scripts/run_seq513_rebuild_chain_v1.sh|scripts/rebuild_entry_model_native_seq513_dataset.sh) ;;
      *) continue ;;
    esac
  fi
  [[ -n "$identity" && "$identity" != "scripts/gx1_handover.sh" ]] || continue
  process_count["$identity"]=$(( ${process_count["$identity"]:-0} + 1 ))
  process_pids["$identity"]="${process_pids["$identity"]:-},$pid"
  process_ppids["$identity"]="${process_ppids["$identity"]:-},$ppid"
  process_states["$identity"]="${process_states["$identity"]:-},$state"
  if [[ "$identity" == "scripts/run_seq513_rebuild_chain_v1.sh" ]]; then
    read -ra command_parts <<< "$args"
    chain_run_id=""
    chain_event_dir=""
    for ((j = 0; j + 1 < ${#command_parts[@]}; j++)); do
      case "${command_parts[$j]}" in
        --run-id) chain_run_id="${command_parts[$((j + 1))]}" ;;
        --event-root) chain_event_dir="${command_parts[$((j + 1))]}" ;;
      esac
    done
    if [[ -n "$chain_run_id" && -n "$chain_event_dir" ]]; then
      chain_specs+=("$chain_run_id|$chain_event_dir")
    fi
  fi
done < <(ps -ww -eo pid=,ppid=,stat=,%cpu=,%mem=,etime=,args=)

printf 'matched_group_count: %d\n' "${#process_count[@]}"
if (( ${#process_count[@]} == 0 )); then
  echo "active_gx1_processes: NONE"
else
  while IFS= read -r identity; do
    printf -- '- %s: count=%d pids=%s ppids=%s states=%s\n' \
      "$identity" "${process_count[$identity]}" \
      "${process_pids[$identity]#,}" "${process_ppids[$identity]#,}" \
      "${process_states[$identity]#,}"
  done < <(printf '%s\n' "${!process_count[@]}" | sort)
fi

if (( ${#chain_specs[@]} == 0 )); then
  echo "active_seq513_chain: NONE"
else
  for spec in "${chain_specs[@]}"; do
    run_id="${spec%%|*}"
    event_dir="${spec#*|}"
    [[ "$event_dir" == /* ]] || event_dir="$REPO/$event_dir"
    event_dir="$(realpath -m -- "$event_dir")"
    status_path="$event_dir/CHAIN_STATUS.json"
    echo "active_seq513_chain_run_id: $run_id"
    echo "active_seq513_chain_event_dir: $event_dir"
    echo "active_seq513_chain_status_path: $status_path"
    if [[ -f "$status_path" ]]; then
      "$PY" - "$status_path" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("active_seq513_chain_status: " + " ".join(
    f"{key}={status[key]}"
    for key in ("state", "step", "updated_utc", "started_utc")
    if key in status
))
PY
    else
      echo "active_seq513_chain_status: MISSING"
    fi
  done
fi
echo
echo "Compact snapshot complete. Read SYSTEM_MAP.md before any targeted rg."

if [[ "$mode" == "verbose" ]]; then
  echo
  echo "## Raw Python process table (--verbose)"
  ps -ww -C python -C python3 -o pid,ppid,stat,%cpu,%mem,etime,args --sort=-%cpu \
    || echo "(no python/python3 processes matched)"
  echo
  echo "## Full Handover (--verbose)"
  cat -- "$HANDOVER"
fi
