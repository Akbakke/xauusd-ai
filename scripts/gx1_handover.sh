#!/usr/bin/env bash
# Read-only, fail-closed takeover status for the GX1 gold/XAUUSD project.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
LAUNCH_STATE="$REPO/PROJECT_STATE_xau_direction_launch.json"
PY="$REPO/.venv/bin/python"

# Keep the authority fingerprint path-ordered. Historical prose is reference
# only; GX1_RULES.md defines the active scope.
sources=(
  "$REPO/AGENTS.md"
  "$REPO/CLAUDE.md"
  "$REPO/GX1_RULES.md"
  "$REPO/README.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/docs/CURRENT_AUDIT_STATUS_20260828.md"
  "$REPO/docs/OFFLINE_CHAMPION_CHALLENGER_V1.md"
  "$REPO/docs/DATA_CONTRACT.md"
  "$REPO/docs/ATTENDED_STAGED_PREFLIGHT_DESIGN_20260823.md"
  "$REPO/docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md"
  "$REPO/docs/FEATURE_VALUE_REVIEW_20260813.md"
  "$REPO/docs/INDICATOR_FIDELITY_AUDIT_20260813.md"
  "$REPO/docs/GIT_WORKTREE_POLICY.md"
  "$REPO/docs/POST_BUILD_INTEGRITY_GATE_20260825.md"
  "$REPO/docs/PREREGISTERED_DIRECTION_TEST_20260820.md"
  "$REPO/docs/RECIPE_DECISION_DRAFT_20260808.md"
  "$REPO/docs/V29_EVENT_SURFACE_DESIGN_20260811.md"
  "$REPO/docs/TRAIN_WINDOW_WIDENING_20260819.md"
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
CURRENT_PAIR_MANIFEST=$("$PY" - "$LAUNCH_STATE" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = state.get("current_pair_manifest")
if not isinstance(path, str) or not path.startswith("/"):
    raise SystemExit("FATAL: launch authority current_pair_manifest is invalid")
print(path)
PY
)
[[ -f "$CURRENT_PAIR_MANIFEST" && ! -L "$CURRENT_PAIR_MANIFEST" ]] || {
  echo "FATAL: current pair manifest missing/non-regular: $CURRENT_PAIR_MANIFEST" >&2
  exit 2
}
[[ -x "$PY" ]] || { echo "FATAL: repository Python is not executable: $PY" >&2; exit 2; }
cd "$REPO"

readarray -t identity < <("$PY" - "$REPO" "$LAUNCH_STATE" \
  "$CURRENT_PAIR_MANIFEST" "${sources[@]}" "$CURRENT_PAIR_MANIFEST" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.current_audited_dataset_evidence_v1 import (
    require_blocked_launch_state_with_current_audited_dataset,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
)

repo = Path(sys.argv[1])
launch_path = Path(sys.argv[2])
pair_path = Path(sys.argv[3])
authority_paths = tuple(Path(raw) for raw in sys.argv[4:])

state = json.loads(launch_path.read_text(encoding="utf-8"))
try:
    audited_dataset = require_blocked_launch_state_with_current_audited_dataset(
        state
    )
except RuntimeError as exc:
    raise SystemExit(f"FATAL: current audited dataset evidence invalid: {exc}") from exc
expected_state = {
    "required_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
    "required_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
    "required_base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
    "required_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    "required_mandatory_causal_layer_feature_count": (
        MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
    ),
    "required_available_candidate_feature_count": (
        MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
    ),
    "required_mandatory_causal_layer_count": len(
        MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    ),
    "required_ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
    "required_ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
}
for key, expected in expected_state.items():
    if state.get(key) != expected:
        raise SystemExit(
            f"FATAL: launch authority {key}={state.get(key)!r} "
            f"does not match source owner {expected!r}"
        )
if len(MULTI_TF_PER_BAR_FEATURES_V4) != MULTI_TF_FEATURE_COUNT_V4:
    raise SystemExit("FATAL: MTF tuple/count owner mismatch")
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
authority.update(b"gx1-takeover-authority-v3\0")
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
print(audited_dataset["status"])
print(audited_dataset["dataset_run_id"])
print(audited_dataset["report_count"])
print(audited_dataset["blocker"])
print(pair_id)
print(artifacts["canonical_v3"]["parquet_path"])
print(artifacts["base28"]["parquet_path"])
print(native["m1"]["root"])
print(native["m5"]["root"])
print(lineage["coverage"]["base28_time_max_utc"])
print(lineage["coverage"]["canonical_time_max_utc"])
print(
    "local=M5 sequence=96 "
    f"signal={MODEL_NATIVE_SIGNAL_DIM} "
    f"ctx_cont={MODEL_NATIVE_CTX_CONT_DIM} "
    f"ctx_cat={MODEL_NATIVE_CTX_CAT_DIM} "
    f"mtf_per_tf={MULTI_TF_FEATURE_COUNT_V4} mtf=M15,H1,H4,D1"
)
print(
    f"signal={MODEL_NATIVE_SIGNAL_SCHEMA_VERSION} "
    f"split={MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION} "
    f"mandatory={MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION} "
    f"matrix={HTF_V4_MATRIX_CONTRACT} "
    f"cache={HTF_V4_CACHE_SCHEMA_VERSION} "
    f"liveness={HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION}"
)
PY
)

authority_sha256=${identity[0]}
worktree_sha256=${identity[1]}
changed_path_count=${identity[2]}
required_contract_mode=${identity[3]}
dataset_event_id=${identity[4]}
dataset_admission_stage=${identity[5]}
audited_dataset_status=${identity[6]}
audited_dataset_run_id=${identity[7]}
audited_dataset_report_count=${identity[8]}
audited_dataset_blocker=${identity[9]}
pair_generation_id=${identity[10]}
canonical_v3_path=${identity[11]}
base28_path=${identity[12]}
native_m1_root=${identity[13]}
native_m5_root=${identity[14]}
m1_time_max=${identity[15]}
m5_time_max=${identity[16]}
entry_contract_summary=${identity[17]}
feature_contract_summary=${identity[18]}
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
echo "current_audited_dataset_status: $audited_dataset_status"
echo "current_audited_dataset_run_id: $audited_dataset_run_id"
echo "current_audited_dataset_report_count: $audited_dataset_report_count"
echo "dataset_contract: HASH_BOUND_AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED"
echo "train_recipe: FROZEN_PRETEST_V4_RESEARCH_RECIPE_PARTIAL_CANDIDATE_SESSION_ONLY"
echo "model_contract: NO_ADMITTED_UNIFIED_BUNDLE"
echo "historical_pnl_winrate: UNPROVEN"
echo "strict_preflight: PASS_V4_TECHNICAL_PIPELINE_ONLY_NO_EXTERNAL_TRAIN_AUTHORITY"
echo "strict_preflight_test_accessed: NO"
echo "technical_checkpoint_bundle_parity: PASS_TECHNICAL_ONLY_NOT_CANDIDATE"
echo "technical_checkpoint_bundle_parity_method: CLEAN_CPU_TO_CLEAN_CPU_EXACT__CUDA_HASH_BOUND_NOT_BITWISE_CLAIMED"
echo "val_decision_journal: PASS_VAL_ONLY_PLUMBING_NOT_EDGE_OR_BACKTEST"
echo "candidate_static_gate_source_policy: EXIT_ONLY_PROVISIONAL_POSITIVE_OPEN__HASH_BOUND_DIRECT_EXIT_INPUT_REQUIRED__ENTRY_STRICT"
echo "candidate_static_gate_runtime_evidence: PARTIAL_TRAIN_AND_FRESH_PROCESS_RESUME_ONLY_NO_VAL"
echo "candidate_session: CHECKPOINT_640__FIRST_WINDOW_576__FRESH_PROCESS_RESUMED_577_TO_640"
echo "candidate_validation: NOT_REACHED__FIRST_VAL_AFTER_31004_TRAIN_BATCHES"
echo "external_full_training: NO_GO_PENDING_EXPLICIT_COST_REVIEW_FROZEN_COMMIT_RECIPE_AND_FULL_CANDIDATE_PLAN"
echo "exit_contract: LOCAL_M1_PLUS_CAUSAL_M5_M15_H1_H4_D1_REQUIRED"
# A restated test count goes stale the moment anyone adds a test — and
# every restated number in this repository has (rule 13/25). State the
# standing requirement, which cannot rot, and date the last verification.
echo "source_regression: RELEVANT_CONTRACT_TESTS_MUST_PASS_BEFORE_EACH_SOURCE_CHANGE"
echo "source_regression_last_verified: focused source-binding/gate/parity regressions must pass on the frozen repair commit; no whole-repository green claim is made here"
echo "pair_generation_id: $pair_generation_id"
echo "native_m1_root: $native_m1_root"
echo "native_m5_root: $native_m5_root"
echo "canonical_v3_path: $canonical_v3_path"
echo "base28_path: $base28_path"
echo "source_time_max: M1=$m1_time_max M5=$m5_time_max"
echo
echo "## Fixed architecture"
echo "feature_owners: SAME_8_IMPLEMENTATIONS_NATIVE_M5_AND_M1_NO_VALUE_COPY"
# Current counts and identities are imported above from the code-owned signal
# and feature owners. The local surface exposes the entire candidate pool in
# owner order; no shell-restated top-k, quota or score cutoff is authoritative.
echo "entry: $entry_contract_summary"
echo "feature_contracts: $feature_contract_summary"
echo "entry_feature_surface: HASH_BOUND_NATIVE_M5_LOADED_ONCE_EXACT_ZERO_COPY_SPLIT_WINDOWS"
echo "exit: local=M1 sequence=480 mtf=M5,M15,H1,H4,D1 same_contract_plus_causal_path shared_encoder=true"
echo "mtf_construction: CLOSED_OHLCV_BEFORE_FEATURES_NO_COMPUTED_M1_RESAMPLING"
echo "direction_authority: UNIQUE_RAW_BPS_ENTRY_Q_ARGMAX_OR_FAIL_CLOSED"
echo "exit_authority: UNIQUE_SAME_BUNDLE_MODEL_ARGMAX_OR_FAIL_CLOSED"
echo "execution_path: DETERMINISTIC_FP32 feature_workers=1 dataloader_workers=0"
echo
echo "## Resume boundary"
echo "scope: OFFLINE_SHARED_FEATUREBASE_ONLY"
echo "source_identity_gate: $([[ $changed_path_count == 0 ]] && echo READY_CLEAN_WORKTREE || echo BLOCK_DIRTY_WORKTREE)"
echo "resume_stage: PRESERVE_FROZEN_CHECKPOINT_640__DECLARE_NEXT_FULL_EPOCH_OR_EXTERNAL_PLAN_EXPLICITLY"
echo "dataset_rebuild: NOT_REQUIRED_FOR_OFFLINE_RESEARCH; PRODUCTION_ECONOMICS_REVIEW_MAY_REQUIRE_A_SUCCESSOR"
echo "production_economics_blocker: $audited_dataset_blocker"
echo "capacity: audits=4G training_max=20G swap=512M cpu=0-1 dataloader_workers=0 one_job_at_a_time"
echo "local_cuda: CHECKPOINT_640_RESUME_ONLY_BEHIND_220W_ACTUAL_70C_12G_GUARD__390W_DRIVER_LIMIT_IS_NOT_AUTHORITY"
echo "cuda_speed: CUDA_ACTIVATION_RETENTION_0_45_ALLOCATOR_FENCE_FP32_ONLY__64_BATCHES_101_889S_TO_86_863S__FULL_TRAIN_EPOCH_APPROX_11_7H"
echo "current_cuda_authority: PARTIAL_SESSION_MECHANICS_PROVED__NO_AUTOMATIC_FULL_EPOCH_OR_VAL_TEST_EXECUTION"
echo "remote_compute: PREPARE_ONLY_UNTIL_EXPLICIT_COST_APPROVAL_FROZEN_COMMIT_AND_V46_HASHES_REQUIRED"
echo "environment: CPYTHON_3.10.12 PINNED_DIRECT_REQUIREMENTS"
echo "ordered_control_routes:"
echo "  1. run this handover and confirm clean source, no competing job, checkpoint-640 pointer and frozen recipe/source identity"
echo "  2. choose an explicitly declared full-epoch execution plan: guarded local resumes or approved external compute; do not change features, targets or guard limits"
echo "  3. reach first complete TRAIN epoch and full VAL before interpreting learning; checkpoint selection and early-stop policy stay frozen"
echo "  4. repeat candidate audit only after a completed full candidate; no partial-session metric is an edge claim"
echo "  5. run preregistered untouched-TEST evaluation only after the candidate/OOS gates, never as a troubleshooting input"
echo "  6. bind immutable broker costs, financing, gap/terminal treatment and portfolio capital before demo, paper, live or production-net claims"
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
