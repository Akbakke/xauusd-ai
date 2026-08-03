#!/usr/bin/env bash
# Compact, read-only takeover viewer for the XAUUSD direction-repair handover.
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
LAUNCH_STATE="$REPO/PROJECT_STATE_xau_direction_launch.json"
PY="$REPO/.venv/bin/python"
GX1_DATA_ROOT=/home/andre2/GX1_DATA/data/data/prebuilt
CURRENT_DATASET_DIR="$GX1_DATA_ROOT/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_DATASET_V8"
CURRENT_EXIT_LIFECYCLE="$GX1_DATA_ROOT/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_EXIT_LIFECYCLE_V13/UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
CURRENT_MTF_MANIFEST="$GX1_DATA_ROOT/CANONICAL_V3_BASE28_OFFLINE_20260801_MTF_V4/manifest.json"
CURRENT_RECIPE_AUDIT="$GX1_DATA_ROOT/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_TRAIN_RECIPE_AUDIT_V15_20260803T144245Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260803T144212558258Z.json"
CURRENT_SMOKE_BUNDLE_DIR="$GX1_DATA_ROOT/v10_entry_model_native_seq513_smoke_XAU_SEQ513_OFFLINE_20260801_V3_20260803T085638Z"

usage() {
  cat <<'EOF'
Usage: scripts/gx1_handover.sh [--check|--verbose]

Default: compact, hash-bound status. --check prints only the deterministic
authority fingerprint and minimal source state. The compact view includes the
exact resume boundary and public control routes. --verbose additionally prints
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
  "$REPO/GX1_RULES.md"
  "$REPO/CLAUDE.md"
  "$REPO/DEVELOPMENT_NOTES.md"
  "$REPO/README.md"
  "$REPO/GX1_PATHS.md"
  "$REPO/RISK_OF_WRONG_CODE_2026_05_24.md"
  "$REPO/ROADMAP.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/PROJECT_STATE.md"
  "$REPO/DECISION_LOG.md"
  "$REPO/docs/BACKFILL_2020_2025_COMMANDS.md"
  "$REPO/docs/CANONICAL_EXIT_STATUS.md"
  "$REPO/docs/DATA_CONTRACT.md"
  "$REPO/docs/DATA_OANDA_SCHEMA_SSOT.md"
  "$REPO/docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md"
  "$REPO/docs/FEATURE_MANIFEST.md"
  "$REPO/docs/GIT_WORKTREE_POLICY.md"
  "$REPO/docs/SESSION_CONTEXT_OBSERVABILITY_NOTE.md"
  "$REPO/docs/TRAINING_DETERMINISM_MPS.md"
  "$REPO/PROJECT_STATE_artifacts.json"
  "$REPO/PROJECT_STATE_entry_iql_delete_incident.json"
  "$LAUNCH_STATE"
)
for source in "${sources[@]}"; do
  [[ -f "$source" ]] || { echo "FATAL: authoritative input missing: $source" >&2; exit 2; }
done
[[ -x "$PY" ]] || { echo "FATAL: repository Python is not executable: $PY" >&2; exit 2; }
cd "$REPO"

verify_current_offline_evidence() {
  "$PY" - \
    "$CURRENT_RECIPE_AUDIT" \
    "$CURRENT_DATASET_DIR" \
    "$CURRENT_EXIT_LIFECYCLE" \
    "$CURRENT_MTF_MANIFEST" <<'PY'
import argparse
import hashlib
import json
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_train_launch_v1 import validate_launch
from gx1.features.htf_features import HTF_V4_CACHE_SCHEMA_VERSION


recipe_path, dataset_dir, exit_lifecycle_path, mtf_manifest_path = (
    Path(raw).resolve(strict=True) for raw in sys.argv[1:]
)
recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
bindings = recipe.get("artifact_bindings")
trainer_cli = recipe.get("trainer_cli")
source_bindings = recipe.get("source_bindings")
if (
    not isinstance(bindings, dict)
    or not isinstance(trainer_cli, dict)
    or not isinstance(source_bindings, dict)
):
    raise SystemExit("FATAL: current train recipe has no exact binding/CLI map")

if Path(str(recipe.get("dataset_dir", ""))).resolve() != dataset_dir:
    raise SystemExit("FATAL: current train recipe does not bind the handover dataset")
for key, expected in (
    ("unified_exit_lifecycle_manifest_json", exit_lifecycle_path),
    ("multi_tf_cache_manifest_json", mtf_manifest_path),
):
    binding = bindings.get(key)
    if not isinstance(binding, dict) or Path(str(binding.get("path", ""))).resolve() != expected:
        raise SystemExit(f"FATAL: current train recipe binding mismatch: {key}")

launch_args = dict(trainer_cli)
launch_args.update(
    profile=recipe.get("profile"),
    repo=str(Path.cwd()),
    wrapper_path=str(Path(str(source_bindings["wrapper"]["path"]))),
    run_id=recipe.get("run_id"),
    dataset_dir=str(dataset_dir),
    out_bundle_dir=recipe.get("out_bundle_dir"),
    recipe_audit_json=str(recipe_path),
)
for key, binding in bindings.items():
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise SystemExit(f"FATAL: malformed current train recipe binding: {key}")
    launch_args[key] = binding["path"]
validate_launch(argparse.Namespace(**launch_args))

mtf_manifest = json.loads(mtf_manifest_path.read_text(encoding="utf-8"))
mtf_liveness = mtf_manifest.get("full_input_liveness")
if (
    mtf_manifest.get("schema_version") != HTF_V4_CACHE_SCHEMA_VERSION
    or mtf_manifest.get("feature_count") != 111
    or not isinstance(mtf_liveness, dict)
    or mtf_liveness.get("decision") != "PASS"
):
    raise SystemExit("FATAL: current offline MTF V4 cache is not schema-v3 PASS")

exit_lifecycle = json.loads(exit_lifecycle_path.read_text(encoding="utf-8"))
shared = exit_lifecycle.get("shared_feature_base_contract")
if (
    exit_lifecycle.get("decision") != "PASS"
    or exit_lifecycle.get("action_order") != ["HOLD", "EXIT_NOW"]
    or not isinstance(shared, dict)
    or shared.get("ordered_signal_dim") != 513
    or shared.get("specialist_family_count") != 8
    or shared.get("entry_feature_sequence_bars") != 96
    or shared.get("exit_feature_sequence_bars") != 480
    or shared.get("separate_feature_implementations_forbidden") is not True
):
    raise SystemExit("FATAL: current unified Exit lifecycle contract mismatch")

print(
    "V8_V13_VERIFIED "
    f"mtf={mtf_manifest['schema_version']} "
    f"recipe_sha256={hashlib.sha256(recipe_path.read_bytes()).hexdigest()}"
)
PY
}

offline_evidence_status=$(verify_current_offline_evidence)

worktree_fingerprint() {
  "$PY" - "$REPO" <<'PY'
import hashlib
import os
import subprocess
import sys
from pathlib import Path


repo = Path(sys.argv[1])


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


digest = hashlib.sha256()
digest.update(b"gx1-worktree-identity-v1\0")
for label, payload in (
    (b"head", git_bytes("rev-parse", "HEAD")),
    (
        b"tracked-diff",
        git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--"),
    ),
):
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)

untracked = tuple(
    raw
    for raw in git_bytes(
        "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    if raw
)
for raw_path in untracked:
    path = repo / os.fsdecode(raw_path)
    if path.is_symlink():
        kind = b"symlink"
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
    elif path.is_file():
        kind = b"file"
        payload = path.read_bytes()
    else:
        raise SystemExit(
            "FATAL: unsupported untracked worktree entry: "
            + os.fsdecode(raw_path)
        )
    digest.update(len(raw_path).to_bytes(8, "big"))
    digest.update(raw_path)
    digest.update(len(kind).to_bytes(4, "big"))
    digest.update(kind)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)

print(digest.hexdigest())
PY
}

worktree_sha256=$(worktree_fingerprint)
wsl_vm_status=ACTIVE_OR_UNVERIFIED
if grep -qi microsoft /proc/version \
  && [[ -r /mnt/c/Users/Andre/.wslconfig ]] \
  && (( $(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo) > 34 * 1024 * 1024 )); then
  wsl_vm_status=PENDING_RESTART
fi

if [[ "$mode" == "check" ]]; then
  mapfile -t git_lines < <(git status --short --untracked-files=all)
  "$PY" - "${sources[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from gx1.features.htf_features import HTF_V4_CACHE_SCHEMA_VERSION


paths = tuple(Path(raw) for raw in sys.argv[1:])
digest = hashlib.sha256()
digest.update(b"gx1-takeover-authority-v2\0")
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
    or repair.get("historical_rebuild_execution_started") is not True
    or repair.get("historical_training_execution_started") is not True
    or repair.get("active_v4_rebuild_started") is not False
    or repair.get("active_v4_training_started") is not False
    or repair.get("empirical_direction_edge_proven") is not False
):
    raise SystemExit("FATAL: malformed source-repair checkpoint")
remaining_source_p0 = repair.get("remaining_source_p0")
if (
    not isinstance(remaining_source_p0, list)
    or not remaining_source_p0
    or any(not isinstance(item, str) or not item for item in remaining_source_p0)
    or len(set(remaining_source_p0)) != len(remaining_source_p0)
):
    raise SystemExit("FATAL: malformed remaining source-P0 register")
verification = repair.get("repository_verification")
if (
    not isinstance(verification, dict)
    or verification.get("current_tree_complete_verification")
    not in {"PENDING_FINAL_ROOT_VERIFICATION", "PASS"}
    or not isinstance(verification.get("tests_collected"), int)
    or not isinstance(verification.get("tests_passed"), int)
    or not isinstance(verification.get("tests_skipped"), int)
    or verification.get("tests_failed") != 0
    or verification["tests_collected"]
    != verification["tests_passed"] + verification["tests_skipped"]
    or verification.get("changed_python_compile") != "PASS"
    or verification.get("git_diff_check") != "PASS"
    or verification.get("json_parse") != "PASS"
    or verification.get("shell_syntax") != "PASS"
    or verification.get("handover_self_check") != "PASS"
    or verification.get("forbidden_instrument_scan") != "PASS"
):
    raise SystemExit("FATAL: malformed repository-verification checkpoint")

feature_checkpoint = state.get("current_feature_stack_checkpoint")
if (
    not isinstance(feature_checkpoint, dict)
    or feature_checkpoint.get("status")
    != "V4_ARCHITECTURE_PROVEN_HISTORICAL_CACHE_STALE_REBUILD_REQUIRED"
    or feature_checkpoint.get("signal_fields_per_bar") != 513
    or feature_checkpoint.get("continuous_context_fields") != 142
    or feature_checkpoint.get("categorical_context_fields") != 5
    or feature_checkpoint.get("multi_timeframe_order")
    != ["M5", "M15", "H1", "H4", "D1"]
    or feature_checkpoint.get("multi_timeframe_fields_per_bar") != 111
    or feature_checkpoint.get("multi_timeframe_total_cells_per_decision_step")
    != 555
    or feature_checkpoint.get("specialist_family_count") != 8
    or feature_checkpoint.get("family_timeframe_route_count") != 40
):
    raise SystemExit("FATAL: malformed current feature-stack checkpoint")
cache_binding = feature_checkpoint.get("cache_manifest")
if not isinstance(cache_binding, dict):
    raise SystemExit("FATAL: current feature stack has no cache manifest")
cache_manifest_path = Path(str(cache_binding.get("path", "")))
if (
    not cache_manifest_path.is_absolute()
    or cache_manifest_path.resolve() != cache_manifest_path
    or not cache_manifest_path.is_file()
    or cache_manifest_path.is_symlink()
):
    raise SystemExit("FATAL: V4 cache manifest path is not exact")
cache_manifest_bytes = cache_manifest_path.read_bytes()
if hashlib.sha256(cache_manifest_bytes).hexdigest() != cache_binding.get("sha256"):
    raise SystemExit("FATAL: V4 cache manifest SHA-256 mismatch")
cache_manifest = json.loads(cache_manifest_bytes)
cache_liveness = cache_manifest.get("full_input_liveness")
if (
    cache_binding.get("decision") != "HISTORICAL_PASS_ACTIVE_CONTRACT_BLOCK"
    or cache_binding.get("observed_schema_version")
    != cache_manifest.get("schema_version")
    or cache_binding.get("required_schema_version")
    != HTF_V4_CACHE_SCHEMA_VERSION
    or cache_manifest.get("schema_version") == HTF_V4_CACHE_SCHEMA_VERSION
    or cache_manifest.get("feature_count") != 111
    or cache_manifest.get("cache_identity_sha256")
    != cache_binding.get("cache_identity_sha256")
    or not isinstance(cache_liveness, dict)
    or cache_liveness.get("decision") != "PASS"
    or cache_liveness.get("contract_sha256")
    != cache_binding.get("liveness_contract_sha256")
):
    raise SystemExit("FATAL: V4 cache manifest contract mismatch")
if state["decision"] == "BLOCK" and state.get("accepted_via_vedtak") is not None:
    raise SystemExit("FATAL: blocked launch state carries approval authority")

dataset_event_id = state.get("dataset_event_id")
terminal = state.get("accepted_dataset_terminal_evidence")
if dataset_event_id is None:
    if (
        state.get("dataset_admission_stage")
        != "NO_ADMITTED_UNIFIED_DATASET"
        or state.get("accepted_dataset_dir") is not None
        or terminal is not None
        or state.get("current_audited_dataset_evidence") != {}
        or state.get("current_smoke_launch_evidence") is not None
    ):
        raise SystemExit(
            "FATAL: no-dataset authority carries stale current dataset/smoke evidence"
        )
else:
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

print("mode: check")
print(f"authority_fingerprint: {digest.hexdigest()}")
print(f"decision: {state['decision']}")
print(f"updated_utc: {state['updated_utc']}")
observed_cache_tag = str(cache_manifest["schema_version"]).removeprefix(
    "htf_v4_disk_cache_manifest_"
)
required_cache_tag = str(HTF_V4_CACHE_SCHEMA_VERSION).removeprefix(
    "htf_v4_disk_cache_manifest_"
)
print(
    f"v4: OFFLINE_{required_cache_tag}_PASS "
    f"LAUNCH_{observed_cache_tag}_BLOCK 5x111"
)
PY
  echo "head_commit: $(git rev-parse HEAD)"
  printf 'changed_path_count: %d\n' "${#git_lines[@]}"
  printf 'worktree_fingerprint: %s\n' "$worktree_sha256"
  echo "capacity: job=10G/512M cpu=0-1"
  echo "wsl_vm_cap: $wsl_vm_status"
  exit 0
fi

echo "# GX1 XAU Direction Repair Takeover (compact)"
echo "mode: $mode"
echo "full_view_command: bash $REPO/scripts/gx1_handover.sh --verbose"
echo
echo "## Goal"
echo "Build the GX1 trading bot for gold/XAUUSD as one immutable learned bundle"
echo "that selects one unique LONG/SHORT/FLAT Entry and HOLD/EXIT_NOW Exit argmax"
echo "and learned size through one shared encoder."
echo "An exact top-logit tie is unavailable evidence and fails closed, never by array order."
echo "No fallback, live hand-rule, stale artifact authority or soft pass-through exists;"
echo "there is no competing decision path. Entry and Exit train together in one candidate."
echo "Near-perfect practical precision is a target, not a current claim."
echo
echo "## Canonical takeover order"
echo "  1. GX1_RULES.md (binding scope freeze)"
echo "  2. AGENTS.md"
echo "  3. SYSTEM_MAP.md"
echo "  4. HANDOVER_XAU_DIRECTION_REPAIR_20260714.md (reference under scope freeze)"
echo "  5. PROJECT_STATE_xau_direction_launch.json"
echo "  6. relevant code contracts/tests"
echo
echo "## Authority fingerprint inventory (complete; not a reading order)"
for source in "${sources[@]}"; do
  printf '%s  %s\n' "$(sha256sum -- "$source" | cut -d' ' -f1)" "$source"
done
echo "takeover_entrypoint: scripts/entry_next_edge_control.sh handover"
echo "handover_owner: scripts/gx1_handover.sh"
echo
echo "## Current offline evidence anchors"
echo "dataset_dir: $CURRENT_DATASET_DIR"
echo "exit_lifecycle_manifest: $CURRENT_EXIT_LIFECYCLE"
echo "mtf_v4_manifest: $CURRENT_MTF_MANIFEST"
echo "train_recipe_audit: $CURRENT_RECIPE_AUDIT"
echo "offline_evidence_contract: $offline_evidence_status"
for anchor in "$CURRENT_DATASET_DIR" "$CURRENT_EXIT_LIFECYCLE" "$CURRENT_MTF_MANIFEST" "$CURRENT_RECIPE_AUDIT"; do
  if [[ -e "$anchor" ]]; then
    echo "anchor_state: PRESENT $anchor"
  else
    echo "anchor_state: MISSING $anchor"
  fi
done
if pgrep -f 'gx1.models.entry_v10.entry_v10_ctx_train_v3 --train' >/dev/null 2>&1; then
  echo "current_smoke_execution: RUNNING"
else
  echo "current_smoke_execution: NOT_RUNNING"
fi
if [[ -d "$CURRENT_SMOKE_BUNDLE_DIR" ]]; then
  echo "current_smoke_bundle: PRESENT $CURRENT_SMOKE_BUNDLE_DIR"
else
  echo "current_smoke_bundle: NOT_PRODUCED $CURRENT_SMOKE_BUNDLE_DIR"
fi
echo
echo "## Launch authority (historical checkpoint; BLOCK is binding)"
"$PY" - "$LAUNCH_STATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from gx1.features.htf_features import HTF_V4_CACHE_SCHEMA_VERSION

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("checkpoint_scope: HISTORICAL_LAUNCH_STATE_BLOCK_ONLY")
for key in (
    "decision", "updated_utc", "required_contract_mode",
    "accepted_bundle_dir", "bundle_metadata_sha256",
):
    value = state.get(key)
    print(f"{key}: {'NONE' if value is None else value}")

repair = state.get("source_repair_checkpoint")
if (
    not isinstance(repair, dict)
    or repair.get("active_v4_rebuild_started") is not False
    or repair.get("active_v4_training_started") is not False
):
    raise SystemExit("FATAL: active V4 execution state is not fail-closed")
print("active_v4_rebuild_started: false")
print("active_v4_training_started: false")
verification = repair.get("repository_verification")
if not isinstance(verification, dict):
    raise SystemExit("FATAL: repository verification state missing")
current_verification = verification.get("current_tree_complete_verification")
if current_verification not in {"PENDING_FINAL_ROOT_VERIFICATION", "PASS"}:
    raise SystemExit("FATAL: current-tree verification state invalid")
print(f"current_tree_complete_verification: {current_verification}")

feature_checkpoint = state.get("current_feature_stack_checkpoint")
if (
    not isinstance(feature_checkpoint, dict)
    or feature_checkpoint.get("status")
    != "V4_ARCHITECTURE_PROVEN_HISTORICAL_CACHE_STALE_REBUILD_REQUIRED"
    or feature_checkpoint.get("signal_fields_per_bar") != 513
    or feature_checkpoint.get("continuous_context_fields") != 142
    or feature_checkpoint.get("categorical_context_fields") != 5
    or feature_checkpoint.get("multi_timeframe_order")
    != ["M5", "M15", "H1", "H4", "D1"]
    or feature_checkpoint.get("multi_timeframe_fields_per_bar") != 111
    or feature_checkpoint.get("multi_timeframe_total_cells_per_decision_step")
    != 555
    or feature_checkpoint.get("specialist_family_count") != 8
    or feature_checkpoint.get("family_timeframe_route_count") != 40
):
    raise SystemExit("FATAL: malformed current feature-stack checkpoint")
cache_binding = feature_checkpoint.get("cache_manifest")
if not isinstance(cache_binding, dict):
    raise SystemExit("FATAL: current feature stack has no cache manifest")
cache_manifest_path = Path(str(cache_binding.get("path", "")))
if (
    not cache_manifest_path.is_absolute()
    or cache_manifest_path.resolve() != cache_manifest_path
    or not cache_manifest_path.is_file()
    or cache_manifest_path.is_symlink()
):
    raise SystemExit("FATAL: V4 cache manifest path is not exact")
cache_manifest_bytes = cache_manifest_path.read_bytes()
if hashlib.sha256(cache_manifest_bytes).hexdigest() != cache_binding.get("sha256"):
    raise SystemExit("FATAL: V4 cache manifest SHA-256 mismatch")
cache_manifest = json.loads(cache_manifest_bytes)
cache_liveness = cache_manifest.get("full_input_liveness")
if (
    cache_binding.get("decision") != "HISTORICAL_PASS_ACTIVE_CONTRACT_BLOCK"
    or cache_binding.get("observed_schema_version")
    != cache_manifest.get("schema_version")
    or cache_binding.get("required_schema_version")
    != HTF_V4_CACHE_SCHEMA_VERSION
    or cache_manifest.get("schema_version") == HTF_V4_CACHE_SCHEMA_VERSION
    or cache_manifest.get("feature_count") != 111
    or cache_manifest.get("cache_identity_sha256")
    != cache_binding.get("cache_identity_sha256")
    or not isinstance(cache_liveness, dict)
    or cache_liveness.get("decision") != "PASS"
    or cache_liveness.get("contract_sha256")
    != cache_binding.get("liveness_contract_sha256")
):
    raise SystemExit("FATAL: V4 cache manifest contract mismatch")
print(
    "v4_architecture: VERIFIED "
    "timeframes=5 families=8 fields_per_tf=111 routes=40 cells=555"
)
print(
    "launch_checkpoint_v4_cache: BLOCK "
    f"observed={cache_manifest.get('schema_version')} "
    f"required={HTF_V4_CACHE_SCHEMA_VERSION} "
    f"historical_identity={cache_binding['cache_identity_sha256']}"
)
for key in ("dataset_event_id", "dataset_admission_stage", "accepted_dataset_dir"):
    value = state.get(key)
    print(f"historical_checkpoint_{key}: {'NONE' if value is None else value}")

terminal = state.get("accepted_dataset_terminal_evidence")
dataset_event_id = state.get("dataset_event_id")
if dataset_event_id is None:
    if (
        state.get("dataset_admission_stage")
        != "NO_ADMITTED_UNIFIED_DATASET"
        or state.get("accepted_dataset_dir") is not None
        or terminal is not None
        or state.get("current_audited_dataset_evidence") != {}
        or state.get("current_smoke_launch_evidence") is not None
    ):
        raise SystemExit(
            "FATAL: no-dataset authority carries stale current dataset/smoke evidence"
        )
    print("dataset_terminal_evidence: NONE")
    print("current_smoke_launch_evidence: NONE")
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
    print(
        "current_smoke_launch_evidence: "
        + ("BOUND" if state.get("current_smoke_launch_evidence") else "NONE")
    )
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
print(
    "historical_checkpoint_blocker_count: "
    f"{len(blockers) if isinstance(blockers, list) else 'INVALID'}"
)
if isinstance(blockers, list) and blockers:
    print(f"historical_checkpoint_first_blocker: {blockers[0]}")
remaining = repair.get("remaining_source_p0")
if not isinstance(remaining, list) or not remaining:
    raise SystemExit("FATAL: no explicit remaining source-P0 register")
print("historical_checkpoint_remaining_source_p0:")
for index, item in enumerate(remaining, start=1):
    print(f"  {index}. {item}")
PY
echo
echo "## Source worktree"
echo "head_commit: $(git rev-parse HEAD)"
mapfile -t git_lines < <(git status --short --untracked-files=all)
printf 'changed_path_count: %d\n' "${#git_lines[@]}"
printf 'worktree_fingerprint: %s\n' "$worktree_sha256"
git_limit=24
for ((i = 0; i < ${#git_lines[@]} && i < git_limit; i++)); do
  printf '%s\n' "${git_lines[$i]}"
done
if (( ${#git_lines[@]} > git_limit )); then
  printf '... %d more; run git status --short explicitly\n' "$(( ${#git_lines[@]} - git_limit ))"
fi
echo
echo "## Resume boundary"
echo "scope: OFFLINE_SHARED_FEATUREBASE_ONLY"
echo "live_operation: FORBIDDEN"
echo "drift_adaptation: FORBIDDEN"
echo "resume_owner: scripts/entry_next_edge_control.sh"
echo "source_contract: CURRENT_COMMITTED_TREE_REQUIRED"
echo "dataset_contract: CURRENT_OFFLINE_V8_V13_VERIFIED_NOT_LAUNCH_ADMITTED"
echo "model_contract: NO_ADMITTED_UNIFIED_BUNDLE"
echo "wsl_vm_cap: $wsl_vm_status"
if (( ${#git_lines[@]} == 0 )); then
  echo "source_identity_gate: READY_CLEAN_WORKTREE"
  if [[ "$wsl_vm_status" == "PENDING_RESTART" ]]; then
    echo "resume_stage: WAIT_FOR_WSL_CAP_THEN_EXACT_V8_V13_SMOKE"
  else
    echo "resume_stage: READY_FOR_EXACT_V8_V13_SMOKE"
  fi
else
  echo "resume_stage: VERIFY_AND_COMMIT_CURRENT_SOURCE_BEFORE_SMOKE"
  echo "source_identity_gate: BLOCK_DIRTY_WORKTREE"
fi
echo "ordered_control_routes:"
echo "  1. require wsl_vm_cap != PENDING_RESTART and revalidate exact V15 recipe"
echo "  2. model-native-smoke-train -> model-native-smoke-bundle-audit"
echo "  3. model-native-candidate-readiness -> model-native-candidate-train"
echo "  4. model-native-selective-edge -> candidate-bound replay/evidence"
echo "data_rebuild_route: FORBIDDEN_UNLESS_CURRENT_V8_V13_VALIDATION_FAILS"
echo "forbidden_routes: live-tail, broker, daemon, polling, promotion, drift-adaptation"
echo "exact_route_help: bash scripts/entry_next_edge_control.sh --help"
echo
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
