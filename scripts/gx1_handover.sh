#!/usr/bin/env bash
# Read-only, fail-closed takeover status for the GX1 gold/XAUUSD project.
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
LAUNCH_STATE="$REPO/PROJECT_STATE_xau_direction_launch.json"
PY="$REPO/.venv/bin/python"
CURRENT_PAIR_MANIFEST=/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_MODEL_NATIVE_20260809_PAIR_MANIFEST.json

# Keep the authority fingerprint path-ordered. Historical prose is reference
# only; GX1_RULES.md defines the active scope.
sources=(
  "$REPO/AGENTS.md"
  "$REPO/CLAUDE.md"
  "$REPO/GX1_RULES.md"
  "$REPO/README.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/docs/DATA_CONTRACT.md"
  "$REPO/docs/GIT_WORKTREE_POLICY.md"
  "$REPO/docs/RECIPE_DECISION_DRAFT_20260808.md"
  "$REPO/docs/V29_EVENT_SURFACE_DESIGN_20260811.md"
  "$LAUNCH_STATE"
)

usage() {
  cat <<'EOF'
Usage: scripts/gx1_handover.sh [--check|--verbose]

Default prints compact status. --check prints only deterministic authority and
worktree identity. --verbose appends the exact handover document.
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

for source in "${sources[@]}"; do
  [[ -f "$source" ]] || { echo "FATAL: authority input missing: $source" >&2; exit 2; }
done
[[ -f "$CURRENT_PAIR_MANIFEST" && ! -L "$CURRENT_PAIR_MANIFEST" ]] || {
  echo "FATAL: current pair manifest missing/non-regular: $CURRENT_PAIR_MANIFEST" >&2
  exit 2
}
[[ -x "$PY" ]] || { echo "FATAL: repository Python is not executable: $PY" >&2; exit 2; }
cd "$REPO"

readarray -t identity < <("$PY" - "$REPO" "$LAUNCH_STATE" \
  "$CURRENT_PAIR_MANIFEST" "${sources[@]}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
launch_path = Path(sys.argv[2])
pair_path = Path(sys.argv[3])
authority_paths = tuple(Path(raw) for raw in sys.argv[4:])

state = json.loads(launch_path.read_text(encoding="utf-8"))
if not isinstance(state, dict) or state.get("decision") != "BLOCK":
    raise SystemExit("FATAL: launch authority must remain BLOCK")
if state.get("accepted_bundle_dir") is not None:
    raise SystemExit("FATAL: blocked launch authority carries an accepted bundle")
pair = json.loads(pair_path.read_text(encoding="utf-8"))
pair_id = str(pair.get("pair_generation_id") or "")
artifacts = pair.get("artifacts")
lineage = pair.get("lineage")
native = lineage.get("native_sources") if isinstance(lineage, dict) else None
if (
    len(pair_id) != 64
    or not isinstance(artifacts, dict)
    or not isinstance(native, dict)
):
    raise SystemExit("FATAL: current pair authority is invalid")

authority = hashlib.sha256()
authority.update(b"gx1-takeover-authority-v2\0")
for index, path in enumerate(authority_paths):
    path_bytes = str(path).encode("utf-8")
    payload = path.read_bytes()
    authority.update(index.to_bytes(4, "big"))
    authority.update(len(path_bytes).to_bytes(8, "big"))
    authority.update(path_bytes)
    authority.update(len(payload).to_bytes(8, "big"))
    authority.update(payload)

def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout

worktree = hashlib.sha256()
worktree.update(b"gx1-worktree-identity-v1\0")
for label, payload in (
    (b"head", git_bytes("rev-parse", "HEAD")),
    (b"tracked-diff", git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--")),
):
    worktree.update(len(label).to_bytes(4, "big"))
    worktree.update(label)
    worktree.update(len(payload).to_bytes(8, "big"))
    worktree.update(payload)
for raw in filter(None, git_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
    path = repo / os.fsdecode(raw)
    if path.is_symlink():
        kind = b"symlink"
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
    elif path.is_file():
        kind = b"file"
        payload = path.read_bytes()
    else:
        raise SystemExit(f"FATAL: unsupported untracked entry: {path}")
    for value in (raw, kind, payload):
        worktree.update(len(value).to_bytes(8, "big"))
        worktree.update(value)

status = git_bytes("status", "--porcelain=v1", "-z")
changed = len(tuple(filter(None, status.split(b"\0"))))
print(authority.hexdigest())
print(worktree.hexdigest())
print(changed)
print(state.get("required_contract_mode", "MISSING"))
print(state.get("dataset_event_id") or "NONE")
print(state.get("dataset_admission_stage") or "NONE")
print(pair_id)
print(artifacts["canonical_v3"]["parquet_path"])
print(artifacts["base28"]["parquet_path"])
print(native["m1"]["root"])
print(native["m5"]["root"])
print(lineage["coverage"]["base28_time_max_utc"])
print(lineage["coverage"]["canonical_time_max_utc"])
PY
)

authority_sha256=${identity[0]}
worktree_sha256=${identity[1]}
changed_path_count=${identity[2]}
required_contract_mode=${identity[3]}
dataset_event_id=${identity[4]}
dataset_admission_stage=${identity[5]}
pair_generation_id=${identity[6]}
canonical_v3_path=${identity[7]}
base28_path=${identity[8]}
native_m1_root=${identity[9]}
native_m5_root=${identity[10]}
m1_time_max=${identity[11]}
m5_time_max=${identity[12]}
head_commit=$(git rev-parse HEAD)

if [[ "$mode" == check ]]; then
  echo "mode: check"
  echo "authority_fingerprint: $authority_sha256"
  echo "decision: BLOCK"
  echo "head_commit: $head_commit"
  echo "changed_path_count: $changed_path_count"
  echo "worktree_fingerprint: $worktree_sha256"
  exit 0
fi

echo "# GX1 XAU Direction Repair Takeover (compact)"
echo "mode: $mode"
echo "takeover_entrypoint: scripts/entry_next_edge_control.sh handover"
echo "handover_owner: scripts/gx1_handover.sh"
echo
echo "## Goal"
echo "Build the GX1 trading bot for gold/XAUUSD as one learned Entry/Exit bundle."
echo "Entry selects LONG/SHORT/FLAT direction; Exit selects HOLD/EXIT_NOW."
echo "The active path has no competing direction route, fallback or soft pass-through."
echo "Near-perfect practical precision remains a target, not a proven result."
echo
echo "## Current verdict"
echo "decision: BLOCK"
echo "required_contract_mode: $required_contract_mode"
echo "dataset_event_id: $dataset_event_id"
echo "dataset_admission_stage: $dataset_admission_stage"
echo "accepted_bundle_dir: NONE"
echo "dataset_contract: CURRENT_PAIR_READY_FEATURE_DATASET_REBUILD_PENDING"
echo "train_recipe: NONE_VALID_V18_RETIRED_RUN_ID_COLLISION"
echo "model_contract: NO_ADMITTED_UNIFIED_BUNDLE"
echo "historical_pnl_winrate: UNPROVEN"
echo "source_regression: PASS_2078_TESTS_ZERO_SKIPS_ZERO_WARNINGS"
echo "pair_generation_id: $pair_generation_id"
echo "native_m1_root: $native_m1_root"
echo "native_m5_root: $native_m5_root"
echo "canonical_v3_path: $canonical_v3_path"
echo "base28_path: $base28_path"
echo "source_time_max: M1=$m1_time_max M5=$m5_time_max"
echo
echo "## Fixed architecture"
echo "feature_owners: SAME_8_IMPLEMENTATIONS_NATIVE_M5_AND_M1_NO_VALUE_COPY"
echo "entry: local=M5 sequence=96 signal=592 ctx_cont=142 ctx_cat=5 mtf=M15,H1,H4,D1"
echo "entry_feature_surface: HASH_BOUND_NATIVE_M5_LOADED_ONCE_EXACT_ZERO_COPY_SPLIT_WINDOWS"
echo "exit: local=M1 sequence=480 mtf=M5,M15,H1,H4,D1 same_contract_plus_causal_path shared_encoder=true"
echo "mtf_construction: CLOSED_OHLCV_BEFORE_FEATURES_NO_COMPUTED_M1_RESAMPLING"
echo "direction_authority: UNIQUE_CALIBRATED_MODEL_ARGMAX_OR_FAIL_CLOSED"
echo "exit_authority: UNIQUE_SAME_BUNDLE_MODEL_ARGMAX_OR_FAIL_CLOSED"
echo "execution_path: DETERMINISTIC_FP32 feature_workers=1 dataloader_workers=0"
echo
echo "## Resume boundary"
echo "scope: OFFLINE_SHARED_FEATUREBASE_ONLY"
echo "source_identity_gate: $([[ $changed_path_count == 0 ]] && echo READY_CLEAN_WORKTREE || echo BLOCK_DIRTY_WORKTREE)"
echo "resume_stage: RUN_CURRENT_PAIR_SHARED_FEATURE_DATASET_CHAIN_THEN_BOUNDED_SMOKE"
echo "capacity: audits=4G training_max=20G swap=512M cpu=0-1 one_job_at_a_time"
echo "environment: CPYTHON_3.10.12 PINNED_DIRECT_REQUIREMENTS"
echo "ordered_control_routes:"
echo "  1. run current-pair shared M1/M5 featurebase, lifecycle and dataset chain"
echo "  2. model-native-smoke-train -> model-native-smoke-bundle-audit"
echo "  3. model-native-candidate-readiness -> model-native-candidate-train"
echo "  4. calibration -> untouched TEST selective edge -> same-bundle unified Exit proof"
echo "forbidden_routes: live, paper, broker, daemon, promotion, drift-adaptation"
echo
echo "## Source worktree"
echo "head_commit: $head_commit"
echo "changed_path_count: $changed_path_count"
echo "worktree_fingerprint: $worktree_sha256"
echo "authority_fingerprint: $authority_sha256"
echo "registered_worktrees: $(git worktree list --porcelain | awk '$1 == "worktree" {count++} END {print count+0}')"
echo "active_training_processes: $(pgrep -fc 'gx1.models.entry_v10.entry_v10_ctx_train_v3 --train' || true)"

if [[ "$mode" == verbose ]]; then
  echo
  echo "## Full Handover (--verbose)"
  cat "$HANDOVER"
fi
