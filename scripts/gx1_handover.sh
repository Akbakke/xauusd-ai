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
  "$REPO/CLAUDE.md"
  "$REPO/DEVELOPMENT_NOTES.md"
  "$REPO/README.md"
  "$REPO/GX1_PATHS.md"
  "$REPO/ROADMAP.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/PROJECT_STATE.md"
  "$REPO/DECISION_LOG.md"
  "$REPO/PIPELINE_AUDIT_XAU_20260723.md"
  "$REPO/docs/CANONICAL_EXIT_STATUS.md"
  "$REPO/docs/DATA_CONTRACT.md"
  "$REPO/docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md"
  "$REPO/docs/FEATURE_MANIFEST.md"
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
import subprocess
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_train_launch_v1 import (
    RECIPE_AUDIT_SCHEMA,
    artifact_binding,
    canonical_json_sha256,
    recipe_source_binding_paths,
)


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
repair = state.get("source_repair_checkpoint")
if (
    not isinstance(repair, dict)
    or repair.get("status") != "CODE_PROVEN_EMPIRICALLY_UNPROVEN"
    or repair.get("fresh_rebuild_started") is not True
    or repair.get("fresh_training_started") is not True
    or repair.get("empirical_direction_edge_proven") is not False
    or repair.get("remaining_source_p0")
    != [
        "produce_immutable_train_only_rank_reference_and_execute_bound_exit_routes",
        "successor_exit_io_contract_replacing_xgb_bridge_with_accepted_entry_outputs",
        "fresh_v3_and_exit_iql_on_accepted_entry_prediction_evidence",
        "execute_canonical_full_test_active_exit_replay_on_accepted_chain",
    ]
):
    raise SystemExit("FATAL: malformed source-repair checkpoint")
verification = repair.get("repository_verification")
if (
    not isinstance(verification, dict)
    or verification.get("tests_collected") != 1912
    or verification.get("tests_passed") != 1907
    or verification.get("tests_skipped") != 5
    or verification.get("tests_failed") != 0
    or verification.get("changed_python_compile") != "PASS"
    or verification.get("git_diff_check") != "PASS"
    or verification.get("json_parse") != "PASS"
    or verification.get("shell_syntax") != "PASS"
    or verification.get("handover_self_check") != "PASS"
    or verification.get("forbidden_instrument_scan") != "PASS"
):
    raise SystemExit("FATAL: malformed repository-verification checkpoint")
if state["decision"] == "BLOCK" and state.get("accepted_via_vedtak") is not None:
    raise SystemExit("FATAL: blocked launch state carries approval authority")

dataset_event_id = state.get("dataset_event_id")
terminal = state.get("accepted_dataset_terminal_evidence")
if dataset_event_id is not None:
    if not isinstance(dataset_event_id, str) or not dataset_event_id:
        raise SystemExit("FATAL: malformed launch authority: invalid dataset_event_id")
    if not isinstance(terminal, dict):
        raise SystemExit("FATAL: dataset event has no terminal evidence object")
    terminal_path = Path(str(terminal.get("path", "")))
    expected_sha256 = terminal.get("sha256")
    if (
        not terminal_path.is_absolute()
        or terminal_path.resolve() != terminal_path
        or not terminal_path.is_file()
        or terminal_path.is_symlink()
    ):
        raise SystemExit("FATAL: dataset terminal evidence path is not an exact regular file")
    observed_sha256 = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit("FATAL: dataset terminal evidence SHA-256 mismatch")
    terminal_state = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal_state.get("entry_run_id") != dataset_event_id:
        raise SystemExit("FATAL: dataset terminal evidence run-id mismatch")
    if terminal_state.get("state") != terminal.get("state"):
        raise SystemExit("FATAL: dataset terminal evidence state mismatch")
    audits = state.get("current_audited_dataset_evidence")
    if not isinstance(audits, dict) or not audits:
        raise SystemExit("FATAL: dataset event has no audited evidence bindings")
    for name, binding in audits.items():
        if not isinstance(binding, dict):
            raise SystemExit(f"FATAL: malformed dataset audit binding: {name}")
        audit_path = Path(str(binding.get("path", "")))
        if (
            not audit_path.is_absolute()
            or audit_path.resolve() != audit_path
            or not audit_path.is_file()
            or audit_path.is_symlink()
        ):
            raise SystemExit(f"FATAL: dataset audit path is not exact: {name}")
        audit_bytes = audit_path.read_bytes()
        if hashlib.sha256(audit_bytes).hexdigest() != binding.get("sha256"):
            raise SystemExit(f"FATAL: dataset audit SHA-256 mismatch: {name}")
        audit = json.loads(audit_bytes)
        if audit.get("decision") != binding.get("decision"):
            raise SystemExit(f"FATAL: dataset audit decision mismatch: {name}")

smoke_launch = state.get("current_smoke_launch_evidence")
recipe_binding = (
    smoke_launch.get("train_recipe_audit")
    if isinstance(smoke_launch, dict)
    else None
)
if not isinstance(recipe_binding, dict):
    raise SystemExit("FATAL: launch authority has no smoke recipe binding")
recipe_path = Path(str(recipe_binding.get("path", "")))
if (
    not recipe_path.is_absolute()
    or recipe_path.resolve() != recipe_path
    or not recipe_path.is_file()
    or recipe_path.is_symlink()
):
    raise SystemExit("FATAL: smoke recipe path is not an exact regular file")
recipe_bytes = recipe_path.read_bytes()
if hashlib.sha256(recipe_bytes).hexdigest() != recipe_binding.get("sha256"):
    raise SystemExit("FATAL: smoke recipe SHA-256 mismatch")
recipe = json.loads(recipe_bytes)
for key in ("decision", "profile", "run_id", "dataset_run_id", "source_commit"):
    if recipe.get(key) != recipe_binding.get(key):
        raise SystemExit(f"FATAL: smoke recipe binding mismatch: {key}")
if recipe.get("dataset_run_id") != dataset_event_id:
    raise SystemExit("FATAL: smoke recipe dataset lineage mismatch")
env_contract = recipe.get("trainer_env_contract")
trainer_env = recipe.get("trainer_env")
if (
    recipe.get("schema_version") != RECIPE_AUDIT_SCHEMA
    or not isinstance(env_contract, dict)
    or not isinstance(trainer_env, dict)
    or len(trainer_env) != recipe_binding.get("trainer_env_count")
    or env_contract.get("count") != recipe_binding.get("trainer_env_count")
    or env_contract.get("sha256") != recipe_binding.get("trainer_env_contract_sha256")
    or recipe.get("source_bindings_sha256") != recipe_binding.get("source_bindings_sha256")
):
    raise SystemExit("FATAL: smoke recipe contract mismatch")
repo = Path.cwd().resolve()
wrapper_path = (
    repo
    / "scripts"
    / (
        "run_entry_model_native_seq513_smoke_train.sh"
        if recipe["profile"] == "smoke"
        else "run_entry_model_native_seq513_candidate_train.sh"
    )
).resolve()
expected_source_paths = recipe_source_binding_paths(
    repo=repo,
    wrapper_path=wrapper_path,
)
source_bindings = recipe.get("source_bindings")
if (
    not isinstance(source_bindings, dict)
    or set(source_bindings) != set(expected_source_paths)
    or recipe.get("source_bindings_sha256")
    != canonical_json_sha256(source_bindings)
):
    raise SystemExit("FATAL: smoke recipe source binding set/hash mismatch")
for name, source_path in expected_source_paths.items():
    if (
        recipe_binding.get("execution_state") == "READY_NOT_STARTED"
        and source_bindings.get(name) != artifact_binding(source_path)
    ):
        raise SystemExit(f"FATAL: ready smoke recipe source binding is stale: {name}")
execution_state = recipe_binding.get("execution_state")
out_bundle_path = Path(str(recipe_binding.get("out_bundle_dir", "")))
if (
    recipe_binding.get("dry_run_decision") != "PASS"
    or recipe_binding.get("out_bundle_present") is not False
    or out_bundle_path.exists()
):
    raise SystemExit("FATAL: smoke recipe output-state mismatch")
if execution_state == "READY_NOT_STARTED":
    if (
        recipe_binding.get("execution_started") is not False
        or recipe_binding.get("execution_completed") not in (None, False)
    ):
        raise SystemExit("FATAL: ready smoke recipe execution-state mismatch")
elif execution_state == "TERMINAL_FAILED":
    failed = state.get("latest_failed_smoke_execution")
    if (
        recipe_binding.get("execution_started") is not True
        or recipe_binding.get("execution_completed") is not True
        or recipe_binding.get("execution_decision") != "BLOCK"
        or not isinstance(failed, dict)
        or failed.get("run_id") != recipe_binding.get("run_id")
        or failed.get("dataset_run_id") != recipe_binding.get("dataset_run_id")
        or failed.get("decision") != "BLOCK"
        or failed.get("failure_code") != recipe_binding.get("execution_failure_code")
        or failed.get("epochs_completed") != recipe_binding.get("epochs_completed")
        or failed.get("bundle_created") is not False
        or failed.get("completed_utc") != recipe_binding.get("execution_completed_utc")
    ):
        raise SystemExit("FATAL: terminal failed smoke recipe evidence mismatch")
else:
    raise SystemExit(f"FATAL: unsupported smoke recipe execution state: {execution_state!r}")
if subprocess.run(
    ["git", "merge-base", "--is-ancestor", recipe["source_commit"], "HEAD"],
    check=False,
).returncode != 0:
    raise SystemExit("FATAL: smoke recipe source commit is not an ancestor")

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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_train_launch_v1 import (
    RECIPE_AUDIT_SCHEMA,
    artifact_binding,
    canonical_json_sha256,
    recipe_source_binding_paths,
)

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in (
    "decision", "updated_utc", "latest_terminal_event_id",
    "latest_terminal_event_decision", "required_contract_mode",
    "accepted_bundle_dir", "bundle_metadata_sha256",
):
    value = state.get(key)
    print(f"{key}: {'NONE' if value is None else value}")
for key in ("dataset_event_id", "dataset_admission_stage", "accepted_dataset_dir"):
    value = state.get(key)
    print(f"{key}: {'NONE' if value is None else value}")

terminal = state.get("accepted_dataset_terminal_evidence")
dataset_event_id = state.get("dataset_event_id")
if dataset_event_id is None:
    if terminal is not None:
        raise SystemExit("FATAL: terminal evidence exists without dataset_event_id")
    print("dataset_terminal_evidence: NONE")
else:
    if not isinstance(terminal, dict):
        raise SystemExit("FATAL: dataset event has no terminal evidence object")
    terminal_path = Path(str(terminal.get("path", "")))
    expected_sha256 = terminal.get("sha256")
    dataset_dir = Path(str(state.get("accepted_dataset_dir", "")))
    if (
        not dataset_dir.is_absolute()
        or dataset_dir.resolve() != dataset_dir
        or not dataset_dir.is_dir()
        or dataset_dir.is_symlink()
    ):
        raise SystemExit("FATAL: accepted dataset is not an exact directory")
    if (
        not terminal_path.is_absolute()
        or terminal_path.resolve() != terminal_path
        or not terminal_path.is_file()
        or terminal_path.is_symlink()
    ):
        raise SystemExit("FATAL: dataset terminal evidence path is not an exact regular file")
    observed_sha256 = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit("FATAL: dataset terminal evidence SHA-256 mismatch")
    terminal_state = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal_state.get("entry_run_id") != dataset_event_id:
        raise SystemExit("FATAL: dataset terminal evidence run-id mismatch")
    if terminal_state.get("state") != terminal.get("state"):
        raise SystemExit("FATAL: dataset terminal evidence state mismatch")
    print(
        "dataset_terminal_evidence: VERIFIED "
        f"state={terminal['state']} sha256={observed_sha256}"
    )
    print(f"dataset_terminal_path: {terminal_path}")
    audits = state.get("current_audited_dataset_evidence")
    if not isinstance(audits, dict) or not audits:
        raise SystemExit("FATAL: dataset event has no audited evidence bindings")
    for name, binding in audits.items():
        if not isinstance(binding, dict):
            raise SystemExit(f"FATAL: malformed dataset audit binding: {name}")
        audit_path = Path(str(binding.get("path", "")))
        if (
            not audit_path.is_absolute()
            or audit_path.resolve() != audit_path
            or not audit_path.is_file()
            or audit_path.is_symlink()
        ):
            raise SystemExit(f"FATAL: dataset audit path is not exact: {name}")
        audit_bytes = audit_path.read_bytes()
        if hashlib.sha256(audit_bytes).hexdigest() != binding.get("sha256"):
            raise SystemExit(f"FATAL: dataset audit SHA-256 mismatch: {name}")
        audit = json.loads(audit_bytes)
        if audit.get("decision") != binding.get("decision"):
            raise SystemExit(f"FATAL: dataset audit decision mismatch: {name}")
    print(f"dataset_audit_evidence: VERIFIED count={len(audits)}")

smoke_launch = state.get("current_smoke_launch_evidence")
recipe_binding = (
    smoke_launch.get("train_recipe_audit")
    if isinstance(smoke_launch, dict)
    else None
)
if not isinstance(recipe_binding, dict):
    raise SystemExit("FATAL: launch authority has no smoke recipe binding")
recipe_path = Path(str(recipe_binding.get("path", "")))
if (
    not recipe_path.is_absolute()
    or recipe_path.resolve() != recipe_path
    or not recipe_path.is_file()
    or recipe_path.is_symlink()
):
    raise SystemExit("FATAL: smoke recipe path is not an exact regular file")
recipe_bytes = recipe_path.read_bytes()
observed_recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
if observed_recipe_sha256 != recipe_binding.get("sha256"):
    raise SystemExit("FATAL: smoke recipe SHA-256 mismatch")
recipe = json.loads(recipe_bytes)
for key in ("decision", "profile", "run_id", "dataset_run_id", "source_commit"):
    if recipe.get(key) != recipe_binding.get(key):
        raise SystemExit(f"FATAL: smoke recipe binding mismatch: {key}")
if recipe.get("dataset_run_id") != dataset_event_id:
    raise SystemExit("FATAL: smoke recipe dataset lineage mismatch")
env_contract = recipe.get("trainer_env_contract")
trainer_env = recipe.get("trainer_env")
if (
    recipe.get("schema_version") != RECIPE_AUDIT_SCHEMA
    or not isinstance(env_contract, dict)
    or not isinstance(trainer_env, dict)
    or len(trainer_env) != recipe_binding.get("trainer_env_count")
    or env_contract.get("count") != recipe_binding.get("trainer_env_count")
    or env_contract.get("sha256") != recipe_binding.get("trainer_env_contract_sha256")
    or recipe.get("source_bindings_sha256") != recipe_binding.get("source_bindings_sha256")
):
    raise SystemExit("FATAL: smoke recipe contract mismatch")
repo = Path.cwd().resolve()
wrapper_path = (
    repo
    / "scripts"
    / (
        "run_entry_model_native_seq513_smoke_train.sh"
        if recipe["profile"] == "smoke"
        else "run_entry_model_native_seq513_candidate_train.sh"
    )
).resolve()
expected_source_paths = recipe_source_binding_paths(
    repo=repo,
    wrapper_path=wrapper_path,
)
source_bindings = recipe.get("source_bindings")
if (
    not isinstance(source_bindings, dict)
    or set(source_bindings) != set(expected_source_paths)
    or recipe.get("source_bindings_sha256")
    != canonical_json_sha256(source_bindings)
):
    raise SystemExit("FATAL: smoke recipe source binding set/hash mismatch")
for name, source_path in expected_source_paths.items():
    if (
        recipe_binding.get("execution_state") == "READY_NOT_STARTED"
        and source_bindings.get(name) != artifact_binding(source_path)
    ):
        raise SystemExit(f"FATAL: ready smoke recipe source binding is stale: {name}")
execution_state = recipe_binding.get("execution_state")
out_bundle_path = Path(str(recipe_binding.get("out_bundle_dir", "")))
if (
    recipe_binding.get("dry_run_decision") != "PASS"
    or recipe_binding.get("out_bundle_present") is not False
    or out_bundle_path.exists()
):
    raise SystemExit("FATAL: smoke recipe output-state mismatch")
if execution_state == "READY_NOT_STARTED":
    if (
        recipe_binding.get("execution_started") is not False
        or recipe_binding.get("execution_completed") not in (None, False)
    ):
        raise SystemExit("FATAL: ready smoke recipe execution-state mismatch")
elif execution_state == "TERMINAL_FAILED":
    failed = state.get("latest_failed_smoke_execution")
    if (
        recipe_binding.get("execution_started") is not True
        or recipe_binding.get("execution_completed") is not True
        or recipe_binding.get("execution_decision") != "BLOCK"
        or not isinstance(failed, dict)
        or failed.get("run_id") != recipe_binding.get("run_id")
        or failed.get("dataset_run_id") != recipe_binding.get("dataset_run_id")
        or failed.get("decision") != "BLOCK"
        or failed.get("failure_code") != recipe_binding.get("execution_failure_code")
        or failed.get("epochs_completed") != recipe_binding.get("epochs_completed")
        or failed.get("bundle_created") is not False
        or failed.get("completed_utc") != recipe_binding.get("execution_completed_utc")
    ):
        raise SystemExit("FATAL: terminal failed smoke recipe evidence mismatch")
else:
    raise SystemExit(f"FATAL: unsupported smoke recipe execution state: {execution_state!r}")
if subprocess.run(
    ["git", "merge-base", "--is-ancestor", recipe["source_commit"], "HEAD"],
    check=False,
).returncode != 0:
    raise SystemExit("FATAL: smoke recipe source commit is not an ancestor")
print(
    "smoke_recipe_evidence: VERIFIED "
    f"decision={recipe['decision']} env_count={len(trainer_env)} "
    f"source_commit={recipe['source_commit']} sha256={observed_recipe_sha256}"
)
print(f"smoke_recipe_dry_run: {recipe_binding['dry_run_decision']}")
print(f"smoke_recipe_execution_state: {execution_state}")
print(f"smoke_recipe_path: {recipe_path}")
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
if isinstance(blockers, list) and blockers:
    print(f"next_blocker: {blockers[0]}")
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
declare -A chain_spec_seen=()
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
      chain_spec="$chain_run_id|$chain_event_dir"
      if [[ -z "${chain_spec_seen[$chain_spec]:-}" ]]; then
        chain_specs+=("$chain_spec")
        chain_spec_seen["$chain_spec"]=1
      fi
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
