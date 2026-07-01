#!/usr/bin/env bash
set -euo pipefail

# Canonical control surface while Entry foundation-freeze is active.
# Training, promotion, legacy shadow, and live order paths stay blocked until
# feature and target foundation audits pass.

REPO=/home/andre2/src/GX1_ENGINE
PY=$REPO/.venv/bin/python

usage() {
  cat <<'EOF'
Usage:
  scripts/entry_next_edge_control.sh handover
  scripts/entry_next_edge_control.sh readiness-report [--json]
  scripts/entry_next_edge_control.sh verify
  scripts/entry_next_edge_control.sh selftest
  scripts/entry_next_edge_control.sh foundation-guardrails
  scripts/entry_next_edge_control.sh foundation-adoption-candidate --dataset-dir <dir> --feature-audit-json <json> --target-audit-json <json> --specialist-audit-json <json> --smoke-dataset-dir <dir>
  scripts/entry_next_edge_control.sh foundation-activation-plan [--adoption-report <json>]
  scripts/entry_next_edge_control.sh foundation-activation-apply --plan-json <json> [--dry-run|--apply --vedtak <id>]
  scripts/entry_next_edge_control.sh foundation-activation-post-apply --activation-apply-json <json> [--dry-run|--apply --vedtak <id>]
  scripts/entry_next_edge_control.sh worktree-hygiene
  scripts/entry_next_edge_control.sh stage-foundation-cleanup [--dry-run|--apply --vedtak <id>]
  scripts/entry_next_edge_control.sh materialize-smoke
  scripts/entry_next_edge_control.sh materialize-smoke-seq215
  scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>
  scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>
  scripts/entry_next_edge_control.sh train-readiness
  scripts/entry_next_edge_control.sh candidate-readiness
  scripts/entry_next_edge_control.sh candidate-readiness-seq215
  scripts/entry_next_edge_control.sh candidate-readiness-smart
  scripts/entry_next_edge_control.sh replay-readiness
  scripts/entry_next_edge_control.sh replay-readiness-seq215
  scripts/entry_next_edge_control.sh replay-readiness-smart
  scripts/entry_next_edge_control.sh feature-ai-inventory
  scripts/entry_next_edge_control.sh chart-geometry-audit
  scripts/entry_next_edge_control.sh candlestick-audit
  scripts/entry_next_edge_control.sh challenger-extension-manifest
  scripts/entry_next_edge_control.sh challenger-smart-extension-manifest
  scripts/entry_next_edge_control.sh smart-rebuild-preflight --verify-large-input-hashes
  scripts/entry_next_edge_control.sh smart-post-rebuild-readiness
  scripts/entry_next_edge_control.sh smart-post-rebuild-refresh --apply --vedtak <id>
  scripts/entry_next_edge_control.sh smart-smoke-manifest --vedtak <id>
  scripts/entry_next_edge_control.sh smart-smoke-readiness
  scripts/entry_next_edge_control.sh smart-trainability-readiness
  scripts/entry_next_edge_control.sh smart-ablation-replay-plan
  scripts/entry_next_edge_control.sh smart-ablation-replay-matrix
  scripts/entry_next_edge_control.sh smart-smoke-train --vedtak <id> --require-edge-audit
  scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit
  scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit
  scripts/entry_next_edge_control.sh audit-smoke-bundle --bundle-dir <dir>
  scripts/entry_next_edge_control.sh candidate-train --vedtak <id>
  scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id>
  scripts/entry_next_edge_control.sh candidate-train-smart --vedtak <id>
  scripts/entry_next_edge_control.sh selective-edge --bundle-dir <dir> --no-xgb-bundle-dir <dir>
  scripts/entry_next_edge_control.sh replay-evidence --trades-path <csv|parquet>
  scripts/entry_next_edge_control.sh iql-distill --vedtak <id> [--materialize-only|--no-fail-on-not-ready]
  scripts/entry_next_edge_control.sh iql-student-trade-log --vedtak <id>
  scripts/entry_next_edge_control.sh iql-replay-evidence --trades-path <csv|parquet>
  scripts/entry_next_edge_control.sh iql-compare
  scripts/entry_next_edge_control.sh iql-slice-audit
  scripts/entry_next_edge_control.sh entry-exit-materialize
  scripts/entry_next_edge_control.sh entry-exit-handoff
  scripts/entry_next_edge_control.sh entry-exit-reconstruction-audit
  scripts/entry_next_edge_control.sh entry-exit-state-reward-contract
  scripts/entry_next_edge_control.sh entry-exit-split-leakage-audit
  scripts/entry_next_edge_control.sh entry-exit-model-dataset-readiness
  scripts/entry_next_edge_control.sh entry-exit-feature-alignment
  scripts/entry_next_edge_control.sh entry-exit-transformer-architecture-readiness
  scripts/entry_next_edge_control.sh entry-exit-transformer-training-plan-readiness
  scripts/entry_next_edge_control.sh entry-exit-transformer-trainer-wrapper-readiness
  scripts/entry_next_edge_control.sh entry-exit-transformer-pretrain-manifest
  scripts/entry_next_edge_control.sh entry-exit-model-dataset-slice-robustness
  scripts/entry_next_edge_control.sh entry-exit-transformer-train-execution-review
  scripts/entry_next_edge_control.sh entry-exit-transformer-post-train-contract
  scripts/entry_next_edge_control.sh entry-exit-transformer-train-enablement --vedtak <id>
  scripts/entry_next_edge_control.sh entry-exit-transformer-train --vedtak <id> [--dry-run]

Allowed path:
  Entry foundation cleanup -> feature audit -> target audit -> rebuilt dataset -> adoption-candidate proof -> activation-plan review -> optional vedtak-gated activation apply -> vedtak-gated post-apply audit refresh + active verify -> foundation-guardrails -> worktree-hygiene -> optional vedtak-gated stage-foundation-cleanup -> train-readiness -> optional smoke-manifest proof -> vedtak-gated smoke train -> smoke bundle audit -> candidate-readiness -> vedtak-gated candidate train -> selective-edge/no-XGB ablation -> replay-evidence -> replay-readiness -> vedtak-gated IQL distillation contract -> IQL student trade log -> IQL replay evidence -> IQL replay comparison -> IQL slice/tail audit -> Entry-bound Exit per-bar handoff materialization -> Entry-to-Exit handoff readiness -> active Exit per-bar reconstruction audit -> active Exit state/reward contract -> active Exit split/leakage audit -> active Exit model dataset/readiness -> active Entry-to-Exit feature alignment -> active Exit Transformer architecture/readiness -> active Exit Transformer training plan/readiness -> fail-closed active Exit Transformer trainer wrapper readiness -> active Exit Transformer pretrain manifest -> active Exit model dataset slice robustness -> active Exit Transformer train-execution review -> active Exit Transformer post-train audit contract -> explicit active Exit Transformer train-enablement package.

Blocked here:
  generic train, retrain, promote, pin, live, xgb-train, et-train, shadow.
EOF
}

blocked() {
  cat >&2 <<'EOF'
FATAL: blocked by active Entry foundation-freeze.

Current path is foundation seq146 smoke-readiness. Do not run generic train,
promote, pin, start shadow, or place live/practice orders.

Run:
  scripts/entry_next_edge_control.sh handover
  scripts/entry_next_edge_control.sh readiness-report
  scripts/entry_next_edge_control.sh verify
  scripts/entry_next_edge_control.sh selftest
  scripts/entry_next_edge_control.sh foundation-guardrails
  scripts/entry_next_edge_control.sh worktree-hygiene
  scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run
  scripts/entry_next_edge_control.sh train-readiness
EOF
  exit 2
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" = "-h" || "$cmd" = "--help" ]]; then
  usage
  exit 0
fi
shift

cd "$REPO"

case "$cmd" in
  handover)
    exec "$REPO/scripts/gx1_handover.sh" "$@"
    ;;

  readiness-report)
    READINESS_REPORT_JSON=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --json) READINESS_REPORT_JSON=1; shift ;;
        -h|--help)
          echo "Usage: scripts/entry_next_edge_control.sh readiness-report [--json]"
          exit 0
          ;;
        *) echo "FATAL: unknown readiness-report arg: $1" >&2; exit 2 ;;
      esac
    done
    READINESS_REPORT_REFRESH_SKIPPED=0
    if [[ "${GX1_READINESS_REPORT_POLICY_SNAPSHOT:-}" == "20260629_GUARDRAIL_POLICY_ONLY" ]]; then
      READINESS_REPORT_REFRESH_SKIPPED=1
    else
      "$PY" -m gx1.scripts.audit_entry_foundation_worktree_hygiene_v1 --quiet --no-fail-on-dirty
      "$PY" -m gx1.scripts.verify_entry_training_readiness_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 --challenger-seq215 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 --smart-seq520 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 --challenger-seq215 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 --smart-seq520 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 --include-smart-layers --quiet --no-fail-on-audit-fail
      "$PY" -m gx1.scripts.materialize_entry_smart_seq520_rebuild_preflight_v1 --verify-large-input-hashes --quiet --no-fail-on-audit-fail
      "$PY" -m gx1.scripts.audit_entry_smart_dataset_post_rebuild_readiness_v1 --fullscan --verify-source-parquet-hashes --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_smart_seq520_smoke_readiness_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_smart_seq520_trainability_readiness_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.materialize_entry_smart_ablation_replay_plan_gate_v1 --quiet --no-fail-on-not-ready
      "$PY" -m gx1.scripts.verify_entry_smart_ablation_replay_matrix_v1 --quiet --no-fail-on-not-ready
    fi
    "$PY" - "$READINESS_REPORT_JSON" "$READINESS_REPORT_REFRESH_SKIPPED" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

def sha256_file(path):
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def latest_smart_selected_replay_readiness(root):
    candidates = []
    if not root.exists():
        return None
    for path in root.glob("smart_seq520_candidate_*/ENTRY_REPLAY_READINESS_latest.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        decision = str(payload.get("decision") or "")
        ready = int(decision == "READY_FOR_IQL_DISTILLATION_VEDTAK")
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        candidates.append((ready, mtime_ns, path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][2]

def latest_ready_smart_report(root, pattern, ready_decisions):
    candidates = []
    if not root.exists():
        return None
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ready = int(str(payload.get("decision") or "") in set(ready_decisions))
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        candidates.append((ready, mtime_ns, path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][2]

paths = {
    "train-readiness": Path("/home/andre2/GX1_DATA/reports/entry_training_readiness_20260628_v1/ENTRY_TRAINING_READINESS_latest.json"),
    "worktree-hygiene": Path("/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.json"),
    "candidate-readiness": Path("/home/andre2/GX1_DATA/reports/entry_candidate_readiness_20260628_v1/ENTRY_CANDIDATE_READINESS_latest.json"),
    "candidate-readiness-seq215": Path("/home/andre2/GX1_DATA/reports/entry_candidate_readiness_20260628_v1/challenger_seq215_20260630/ENTRY_CANDIDATE_READINESS_latest.json"),
    "candidate-readiness-smart": Path("/home/andre2/GX1_DATA/reports/entry_candidate_readiness_20260628_v1/smart_seq520_candidate/ENTRY_CANDIDATE_READINESS_latest.json"),
    "replay-readiness": Path("/home/andre2/GX1_DATA/reports/entry_replay_readiness_20260628_v1/ENTRY_REPLAY_READINESS_latest.json"),
    "replay-readiness-seq215": Path("/home/andre2/GX1_DATA/reports/entry_replay_readiness_20260628_v1/challenger_seq215_20260630/ENTRY_REPLAY_READINESS_latest.json"),
    "replay-readiness-smart": Path("/home/andre2/GX1_DATA/reports/entry_replay_readiness_20260628_v1/smart_seq520_candidate/ENTRY_REPLAY_READINESS_latest.json"),
    "iql-distillation-contract": Path("/home/andre2/GX1_DATA/reports/entry_iql_distillation_contract_20260628_v1/ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"),
    "iql-student-trade-log": Path("/home/andre2/GX1_DATA/reports/entry_iql_student_trade_log_20260628_v1/ENTRY_IQL_STUDENT_TRADE_LOG_latest.json"),
    "iql-replay-evidence": Path("/home/andre2/GX1_DATA/reports/entry_iql_distillation_replay_20260628_v1/ENTRY_IQL_REPLAY_EVIDENCE_latest.json"),
    "iql-replay-comparison": Path("/home/andre2/GX1_DATA/reports/entry_iql_replay_comparison_20260628_v1/ENTRY_IQL_REPLAY_COMPARISON_latest.json"),
    "iql-replay-slice-audit": Path("/home/andre2/GX1_DATA/reports/entry_iql_replay_slice_audit_20260628_v1/ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json"),
    "entry-exit-per-bar-handoff": Path("/home/andre2/GX1_DATA/reports/entry_exit_per_bar_handoff_20260630_v1/ENTRY_EXIT_PER_BAR_HANDOFF_latest.json"),
    "entry-exit-handoff": Path("/home/andre2/GX1_DATA/reports/entry_exit_handoff_readiness_20260630_v1/ENTRY_EXIT_HANDOFF_READINESS_latest.json"),
    "entry-exit-reconstruction-audit": Path("/home/andre2/GX1_DATA/reports/entry_exit_per_bar_reconstruction_audit_20260630_v1/ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_latest.json"),
    "entry-exit-state-reward-contract": Path("/home/andre2/GX1_DATA/reports/entry_exit_state_reward_contract_20260630_v1/ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.json"),
    "entry-exit-split-leakage-audit": Path("/home/andre2/GX1_DATA/reports/entry_exit_split_leakage_audit_20260630_v1/ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_latest.json"),
    "entry-exit-model-dataset-readiness": Path("/home/andre2/GX1_DATA/reports/entry_exit_model_dataset_readiness_20260630_v1/ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"),
    "entry-exit-feature-alignment": Path("/home/andre2/GX1_DATA/reports/entry_exit_feature_alignment_20260630_v1/ENTRY_EXIT_FEATURE_ALIGNMENT_latest.json"),
    "entry-exit-transformer-architecture-readiness": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_architecture_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_latest.json"),
    "entry-exit-transformer-training-plan-readiness": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"),
    "entry-exit-transformer-trainer-wrapper-readiness": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_trainer_wrapper_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json"),
    "entry-exit-transformer-pretrain-manifest": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_pretrain_manifest_20260630_v1/ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.json"),
    "entry-exit-model-dataset-slice-robustness": Path("/home/andre2/GX1_DATA/reports/entry_exit_model_dataset_slice_robustness_20260630_v1/ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_latest.json"),
    "entry-exit-transformer-train-execution-review": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_train_execution_review_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json"),
    "entry-exit-transformer-post-train-contract": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_post_train_contract_20260630_v1/ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.json"),
    "entry-exit-transformer-train-enablement": Path("/home/andre2/GX1_DATA/reports/entry_exit_transformer_train_enablement_20260701_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_latest.json"),
    "feature-ai-inventory": Path("/home/andre2/GX1_DATA/reports/entry_feature_ai_inventory_20260630_v1/ENTRY_FEATURE_AI_INVENTORY_latest.json"),
    "chart-geometry-audit": Path("/home/andre2/GX1_DATA/reports/entry_chart_geometry_challenger_audit_20260630_v1/ENTRY_CHART_GEOMETRY_CHALLENGER_AUDIT_latest.json"),
    "candlestick-audit": Path("/home/andre2/GX1_DATA/reports/entry_candlestick_pattern_challenger_audit_20260630_v1/ENTRY_CANDLESTICK_PATTERN_CHALLENGER_AUDIT_latest.json"),
    "challenger-extension-manifest": Path("/home/andre2/GX1_DATA/reports/entry_specialist_challenger_extension_manifest_20260630_v1/ENTRY_SPECIALIST_CHALLENGER_EXTENSION_REPORT_latest.json"),
    "challenger-smart-extension-manifest": Path("/home/andre2/GX1_DATA/reports/entry_specialist_challenger_extension_manifest_20260630_v1/ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_REPORT_latest.json"),
    "smart-rebuild-preflight": Path("/home/andre2/GX1_DATA/reports/entry_smart_seq_rebuild_preflight_20260630_v1/ENTRY_SMART_REBUILD_PREFLIGHT_latest.json"),
    "smart-post-rebuild-readiness": Path("/home/andre2/GX1_DATA/reports/entry_smart_dataset_post_rebuild_readiness_20260630_v1/ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json"),
    "smart-smoke-manifest": Path("/home/andre2/GX1_DATA/reports/entry_smart_seq520_smoke_manifest_20260630_v1/ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_latest.json"),
    "smart-smoke-readiness": Path("/home/andre2/GX1_DATA/reports/entry_smart_seq520_smoke_readiness_20260630_v1/ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json"),
    "smart-trainability-readiness": Path("/home/andre2/GX1_DATA/reports/entry_smart_seq520_trainability_readiness_20260630_v1/ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"),
    "smart-ablation-replay-plan": Path("/home/andre2/GX1_DATA/reports/entry_smart_ablation_replay_plan_gate_20260630_v1/ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"),
    "smart-ablation-replay-matrix": Path("/home/andre2/GX1_DATA/reports/entry_smart_ablation_replay_matrix_gate_20260701_v1/ENTRY_SMART_ABLATION_REPLAY_MATRIX_GATE_latest.json"),
}
smart_selected_replay_readiness_path = latest_smart_selected_replay_readiness(
    Path("/home/andre2/GX1_DATA/reports/entry_replay_readiness_20260628_v1")
)
if smart_selected_replay_readiness_path is not None:
    paths["replay-readiness-smart-selected"] = smart_selected_replay_readiness_path
smart_iql_artifacts = {
    "iql-distillation-contract": (
        Path("/home/andre2/GX1_DATA/reports/entry_iql_distillation_contract_20260628_v1"),
        "smart_seq520_candidate_*/ENTRY_IQL_DISTILLATION_CONTRACT_latest.json",
        {"ENTRY_IQL_DISTILLATION_CONTRACT_READY"},
    ),
    "iql-student-trade-log": (
        Path("/home/andre2/GX1_DATA/reports/entry_iql_student_trade_log_20260628_v1"),
        "smart_seq520_candidate_*/ENTRY_IQL_STUDENT_TRADE_LOG_latest.json",
        {"PASS"},
    ),
    "iql-replay-evidence": (
        Path("/home/andre2/GX1_DATA/reports/entry_iql_distillation_replay_20260628_v1"),
        "smart_seq520_candidate_*/ENTRY_IQL_REPLAY_EVIDENCE_latest.json",
        {"PASS"},
    ),
    "iql-replay-comparison": (
        Path("/home/andre2/GX1_DATA/reports/entry_iql_replay_comparison_20260628_v1"),
        "smart_seq520_candidate_*/ENTRY_IQL_REPLAY_COMPARISON_latest.json",
        {"READY_FOR_PROMOTION_REVIEW_VEDTAK"},
    ),
    "iql-replay-slice-audit": (
        Path("/home/andre2/GX1_DATA/reports/entry_iql_replay_slice_audit_20260628_v1"),
        "smart_seq520_candidate_*/ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json",
        {"PASS"},
    ),
}
for key, (root, pattern, ready_decisions) in smart_iql_artifacts.items():
    selected = latest_ready_smart_report(root, pattern, ready_decisions)
    if selected is not None:
        paths[key] = selected
adoption_root = Path("/home/andre2/GX1_DATA/reports/entry_foundation_adoption_candidate_20260629_v1")
adoption_candidates = (
    sorted(
        adoption_root.glob("*/ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if adoption_root.exists()
    else []
)
if adoption_candidates:
    paths["foundation-adoption-candidate"] = adoption_candidates[0]
activation_plan_root = Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_plan_20260629_v1")
activation_plan_candidates = (
    sorted(
        activation_plan_root.glob("*/ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if activation_plan_root.exists()
    else []
)
if not activation_plan_candidates and (activation_plan_root / "ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json").exists():
    activation_plan_candidates = [activation_plan_root / "ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json"]
if activation_plan_candidates:
    paths["foundation-activation-plan"] = activation_plan_candidates[0]
activation_apply_root = Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_apply_20260629_v1")
activation_apply_candidates = (
    sorted(
        activation_apply_root.glob("*/ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if activation_apply_root.exists()
    else []
)
activation_apply_root_latest = activation_apply_root / "ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json"
if activation_apply_root_latest.exists():
    activation_apply_candidates.append(activation_apply_root_latest)
activation_apply_candidates = sorted(
    activation_apply_candidates,
    key=lambda p: p.stat().st_mtime_ns,
    reverse=True,
)
if activation_apply_candidates:
    paths["foundation-activation-apply"] = activation_apply_candidates[0]
activation_post_apply_root = Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_post_apply_20260629_v1")
activation_post_apply_candidates = (
    sorted(
        activation_post_apply_root.glob("*/ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if activation_post_apply_root.exists()
    else []
)
if (
    not activation_post_apply_candidates
    and (activation_post_apply_root / "ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json").exists()
):
    activation_post_apply_candidates = [activation_post_apply_root / "ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json"]
if activation_post_apply_candidates:
    paths["foundation-activation-post-apply"] = activation_post_apply_candidates[0]

json_mode = sys.argv[1] == "1"
refresh_skipped = sys.argv[2] == "1"
reports = {}
for name, path in paths.items():
    if not path.exists():
        if not json_mode:
            print(f"{name}: MISSING {path}")
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    reports[name] = report
train = reports.get("train-readiness") or {}
hygiene = reports.get("worktree-hygiene") or {}
train_activation = (
    train.get("foundation_activation")
    if isinstance(train.get("foundation_activation"), dict)
    else {}
)
train_activation_required_before_smoke = bool(train.get("foundation_activation_required_before_smoke"))
train_activation_apply_required_before_smoke = bool(train.get("foundation_activation_apply_required_before_smoke"))
train_activation_post_apply_required_before_smoke = bool(train.get("foundation_activation_post_apply_required_before_smoke"))
train_activation_apply_argv = (
    train_activation.get("activation_apply_command")
    if isinstance(train_activation.get("activation_apply_command"), list)
    else []
)
allowed_now = [
    "scripts/entry_next_edge_control.sh handover",
    "scripts/entry_next_edge_control.sh readiness-report",
    "scripts/entry_next_edge_control.sh readiness-report --json",
    "scripts/entry_next_edge_control.sh verify --quiet",
    "scripts/entry_next_edge_control.sh selftest --quiet",
    "scripts/entry_next_edge_control.sh foundation-guardrails --quiet",
    "scripts/entry_next_edge_control.sh foundation-activation-plan",
    "scripts/entry_next_edge_control.sh worktree-hygiene --no-fail-on-dirty",
    "scripts/entry_next_edge_control.sh train-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh candidate-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh candidate-readiness-seq215 --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh candidate-readiness-smart --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh replay-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh replay-readiness-seq215 --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh replay-readiness-smart --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh feature-ai-inventory --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh chart-geometry-audit --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh candlestick-audit --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh challenger-extension-manifest --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh challenger-smart-extension-manifest --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh smart-rebuild-preflight --verify-large-input-hashes --quiet --no-fail-on-audit-fail",
    "scripts/entry_next_edge_control.sh smart-post-rebuild-readiness --fullscan --verify-source-parquet-hashes --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh smart-ablation-replay-plan --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh smart-ablation-replay-matrix --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh iql-slice-audit --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-materialize --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-handoff --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-reconstruction-audit --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-state-reward-contract --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-split-leakage-audit --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-model-dataset-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-feature-alignment --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-architecture-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-training-plan-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-trainer-wrapper-readiness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-pretrain-manifest --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-model-dataset-slice-robustness --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-train-execution-review --quiet --no-fail-on-not-ready",
    "scripts/entry_next_edge_control.sh entry-exit-transformer-post-train-contract --quiet --no-fail-on-not-ready",
]
if hygiene.get("foundation_cleanup_stage_ready"):
    allowed_now.append("scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run")
optional_proof_commands = []
if train.get("foundation_contract_ready_for_smoke"):
    optional_proof_commands.append("scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>  # proof only, no trainer start")
optional_proof_commands.append("scripts/entry_next_edge_control.sh entry-exit-transformer-train-enablement --vedtak <id> --quiet --no-fail-on-not-ready  # package proof only, no trainer start")
blocked_now = [
    "scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit  # needs clean git + explicit vedtak",
    "scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit  # needs clean git + explicit SEQ215 vedtak",
    "candidate-train, replay, IQL distillation, promotion, shadow and live remain blocked until their gates pass",
]
stage_cmd = hygiene.get("foundation_cleanup_stage_command") or []
post_stage = hygiene.get("foundation_cleanup_post_stage_verification") or {}
critical_gate_review = (
    hygiene.get("foundation_cleanup_critical_gate_review")
    if isinstance(hygiene.get("foundation_cleanup_critical_gate_review"), dict)
    else {}
)
summaries = {}
for name, path in paths.items():
    report = reports.get(name) or {}
    summaries[name] = {
        "path": str(path),
        "exists": path.exists(),
        "decision": report.get("decision"),
        "failures_count": len(report.get("failures") or []),
        "execution_blockers_count": len(report.get("execution_blockers") or []),
        "next": report.get("next_required_gate")
        or report.get("next_allowed_command")
        or report.get("next_required_action"),
    }

foundation_ready = bool(train.get("foundation_contract_ready_for_smoke"))
real_smoke_train_allowed = bool(train.get("smoke_training_allowed_with_explicit_vedtak"))
smoke_manifest_proof_allowed = foundation_ready
adoption = reports.get("foundation-adoption-candidate") or {}
adoption_candidate_ready = bool(adoption.get("candidate_ready_for_activation"))
adoption_candidate_path = paths.get("foundation-adoption-candidate")
adoption_candidate_artifacts = (
    adoption.get("artifacts") if isinstance(adoption.get("artifacts"), dict) else {}
)
activation_plan = reports.get("foundation-activation-plan") or {}
activation_plan_ready = str(activation_plan.get("decision")) == "READY_FOR_VEDTAK_ACTIVATION"
activation_plan_path = paths.get("foundation-activation-plan")
activation_apply = reports.get("foundation-activation-apply") or {}
activation_apply_ready = str(activation_apply.get("decision")) == "READY_FOR_VEDTAK_APPLY"
activation_apply_path = paths.get("foundation-activation-apply")
activation_apply_applied = (
    str(activation_apply.get("decision")) == "APPLIED_ALIAS_SWITCH"
    and bool(activation_apply.get("mutation_performed"))
)
activation_apply_post_commands = (
    activation_apply.get("post_apply_commands")
    if isinstance(activation_apply.get("post_apply_commands"), list)
    else []
)
activation_post_apply = reports.get("foundation-activation-post-apply") or {}
activation_post_apply_path = paths.get("foundation-activation-post-apply")
activation_post_apply_decision = str(activation_post_apply.get("decision") or "")
activation_post_apply_waiting = activation_post_apply_decision == "WAITING_FOR_ACTIVATION_APPLY"
activation_post_apply_ready = activation_post_apply_decision == "READY_FOR_POST_APPLY_REFRESH"
activation_post_apply_completed = activation_post_apply_decision == "POST_APPLY_REFRESH_COMPLETED"
activation_post_apply_mutations_performed = bool(activation_post_apply.get("post_apply_mutations_performed"))
activation_apply_dry_run_argv = [
    "scripts/entry_next_edge_control.sh",
    "foundation-activation-apply",
    "--plan-json",
    str(activation_plan_path),
    "--dry-run",
] if activation_plan_path else []
activation_apply_argv = train_activation_apply_argv or (
    [
        "scripts/entry_next_edge_control.sh",
        "foundation-activation-apply",
        "--plan-json",
        str(activation_plan_path),
        "--apply",
        "--vedtak",
        "<id>",
    ]
    if activation_plan_path
    else [
        "scripts/entry_next_edge_control.sh",
        "foundation-activation-apply",
        "--plan-json",
        "<activation-plan>",
        "--apply",
        "--vedtak",
        "<id>",
    ]
)
if activation_plan_ready and activation_apply_dry_run_argv:
    allowed_now.append(" ".join(activation_apply_dry_run_argv))
activation_post_apply_dry_run_argv = [
    "scripts/entry_next_edge_control.sh",
    "foundation-activation-post-apply",
    "--activation-apply-json",
    str(activation_apply_path),
    "--dry-run",
] if activation_apply_path else []
activation_post_apply_argv = [
    "scripts/entry_next_edge_control.sh",
    "foundation-activation-post-apply",
    "--activation-apply-json",
    str(activation_apply_path),
    "--apply",
    "--vedtak",
    "<id>",
] if activation_apply_path else [
    "scripts/entry_next_edge_control.sh",
    "foundation-activation-post-apply",
    "--activation-apply-json",
    "<activation-apply-report>",
    "--apply",
    "--vedtak",
    "<id>",
]
activation_apply_allowed_after_vedtak = bool(activation_plan_ready and not activation_apply_applied)
activation_post_apply_allowed_after_vedtak = bool(
    (activation_apply_applied and not activation_post_apply_completed)
    or activation_post_apply_ready
)
if activation_post_apply_dry_run_argv:
    allowed_now.append(" ".join(activation_post_apply_dry_run_argv))
foundation_cleanup_stage_ready = bool(hygiene.get("foundation_cleanup_stage_ready"))
candidate_training_allowed = bool(
    (reports.get("candidate-readiness") or {}).get("candidate_training_allowed_with_explicit_vedtak")
)
candidate_seq215 = reports.get("candidate-readiness-seq215") or {}
candidate_training_seq215_allowed = bool(candidate_seq215.get("candidate_training_allowed_with_explicit_vedtak"))
expected_seq215_specialists = {
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
}
candidate_seq215_required_specialists = {
    str(name)
    for name in candidate_seq215.get("required_specialist_groups", [])
    if str(name)
}
seq215_smoke_contract_preflight_ready = bool(
    str(candidate_seq215.get("contract_mode") or "") == "challenger_seq215"
    and int(candidate_seq215.get("expected_signal_dim") or 0) == 215
    and candidate_seq215_required_specialists == expected_seq215_specialists
)
smoke_manifest_seq215_proof_allowed = bool(
    smoke_manifest_proof_allowed and seq215_smoke_contract_preflight_ready
)
real_smoke_train_seq215_allowed = bool(
    real_smoke_train_allowed and seq215_smoke_contract_preflight_ready
)
smart_post_rebuild = reports.get("smart-post-rebuild-readiness") or {}
smart_post_rebuild_ready = (
    str(smart_post_rebuild.get("decision") or "") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
)
if smoke_manifest_seq215_proof_allowed:
    optional_proof_commands.append("scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>  # proof only, no trainer start")
if smart_post_rebuild_ready:
    optional_proof_commands.append("scripts/entry_next_edge_control.sh smart-post-rebuild-refresh --apply --vedtak <id>  # materialize smart smoke dataset only")
    optional_proof_commands.append("scripts/entry_next_edge_control.sh smart-smoke-manifest --vedtak <id>  # proof only, no trainer start")
iql_distillation_allowed = bool(
    (reports.get("replay-readiness") or {}).get("iql_distillation_allowed_with_explicit_vedtak")
)
iql_distillation_contract_ready = (
    str((reports.get("iql-distillation-contract") or {}).get("decision")) == "ENTRY_IQL_DISTILLATION_CONTRACT_READY"
)
iql_student_trade_log_allowed = bool(iql_distillation_contract_ready)
iql_replay_evidence_ready = str((reports.get("iql-replay-evidence") or {}).get("decision")) == "PASS"
iql_replay_comparison_ready = (
    str((reports.get("iql-replay-comparison") or {}).get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK"
)
iql_replay_slice_audit_ready = str((reports.get("iql-replay-slice-audit") or {}).get("decision")) == "PASS"
entry_exit_handoff = reports.get("entry-exit-handoff") or {}
entry_exit_per_bar = reports.get("entry-exit-per-bar-handoff") or {}
entry_exit_reconstruction = reports.get("entry-exit-reconstruction-audit") or {}
entry_exit_state_reward = reports.get("entry-exit-state-reward-contract") or {}
entry_exit_split_leakage = reports.get("entry-exit-split-leakage-audit") or {}
entry_exit_model_dataset = reports.get("entry-exit-model-dataset-readiness") or {}
entry_exit_feature_alignment = reports.get("entry-exit-feature-alignment") or {}
entry_exit_transformer_architecture = reports.get("entry-exit-transformer-architecture-readiness") or {}
entry_exit_transformer_training_plan = reports.get("entry-exit-transformer-training-plan-readiness") or {}
entry_exit_transformer_trainer_wrapper = reports.get("entry-exit-transformer-trainer-wrapper-readiness") or {}
entry_exit_transformer_pretrain_manifest = reports.get("entry-exit-transformer-pretrain-manifest") or {}
entry_exit_model_dataset_slice_robustness = reports.get("entry-exit-model-dataset-slice-robustness") or {}
entry_exit_transformer_train_execution_review = reports.get("entry-exit-transformer-train-execution-review") or {}
entry_exit_transformer_post_train_contract = reports.get("entry-exit-transformer-post-train-contract") or {}
entry_exit_transformer_train_enablement = reports.get("entry-exit-transformer-train-enablement") or {}
entry_exit_per_bar_decision = str(entry_exit_per_bar.get("decision") or "")
entry_exit_per_bar_ready = entry_exit_per_bar_decision in {"PASS", "PASS_WITH_EXPLICIT_GAP_EXCLUSIONS"}
entry_exit_handoff_entry_ready = bool(entry_exit_handoff.get("entry_evidence_ready"))
entry_exit_handoff_substrate_ready = bool(entry_exit_handoff.get("exit_per_bar_substrate_ready"))
entry_exit_handoff_decision = str(entry_exit_handoff.get("decision") or "")
entry_exit_reconstruction_decision = str(entry_exit_reconstruction.get("decision") or "")
entry_exit_reconstruction_ready = entry_exit_reconstruction_decision == "READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW"
entry_exit_state_reward_decision = str(entry_exit_state_reward.get("decision") or "")
entry_exit_state_reward_ready = entry_exit_state_reward_decision == "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY"
entry_exit_split_leakage_decision = str(entry_exit_split_leakage.get("decision") or "")
entry_exit_split_leakage_ready = entry_exit_split_leakage_decision == "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY"
entry_exit_model_dataset_decision = str(entry_exit_model_dataset.get("decision") or "")
entry_exit_model_dataset_ready = entry_exit_model_dataset_decision == "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
entry_exit_feature_alignment_decision = str(entry_exit_feature_alignment.get("decision") or "")
entry_exit_feature_alignment_ready = entry_exit_feature_alignment_decision == "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
entry_exit_transformer_architecture_decision = str(entry_exit_transformer_architecture.get("decision") or "")
entry_exit_transformer_architecture_ready = entry_exit_transformer_architecture_decision == "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW"
entry_exit_transformer_training_plan_decision = str(entry_exit_transformer_training_plan.get("decision") or "")
entry_exit_transformer_training_plan_ready = entry_exit_transformer_training_plan_decision == "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
entry_exit_transformer_trainer_wrapper_decision = str(entry_exit_transformer_trainer_wrapper.get("decision") or "")
entry_exit_transformer_trainer_wrapper_ready = entry_exit_transformer_trainer_wrapper_decision == "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
entry_exit_transformer_pretrain_manifest_decision = str(entry_exit_transformer_pretrain_manifest.get("decision") or "")
entry_exit_transformer_pretrain_manifest_ready = entry_exit_transformer_pretrain_manifest_decision == "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
entry_exit_model_dataset_slice_robustness_decision = str(entry_exit_model_dataset_slice_robustness.get("decision") or "")
entry_exit_model_dataset_slice_robustness_ready = entry_exit_model_dataset_slice_robustness_decision == "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
entry_exit_transformer_train_execution_review_decision = str(entry_exit_transformer_train_execution_review.get("decision") or "")
entry_exit_transformer_train_execution_review_ready = entry_exit_transformer_train_execution_review_decision == "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
entry_exit_transformer_post_train_contract_decision = str(entry_exit_transformer_post_train_contract.get("decision") or "")
entry_exit_transformer_post_train_contract_ready = entry_exit_transformer_post_train_contract_decision == "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
entry_exit_transformer_train_enablement_decision = str(entry_exit_transformer_train_enablement.get("decision") or "")
entry_exit_transformer_train_enablement_ready = entry_exit_transformer_train_enablement_decision == "ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION"
promotion_review_allowed = bool(
    (reports.get("iql-replay-comparison") or {}).get("promotion_review_allowed_with_explicit_vedtak")
    and iql_replay_slice_audit_ready
)
current_blockers = []
if not real_smoke_train_allowed:
    if foundation_ready:
        current_blockers.append("clean git worktree and explicit smoke-train vedtak")
    else:
        current_blockers.append("foundation contract is not ready for smoke")
if adoption_candidate_ready and not foundation_ready:
    current_blockers.append("explicit vedtak to switch active foundation dataset/audit paths")
if activation_apply_applied and not activation_post_apply_completed:
    current_blockers.append("explicit post-apply vedtak to refresh canonical audits and smoke dataset")
if not candidate_training_allowed:
    current_blockers.append("candidate training requires real smoke bundle edge audit")
if foundation_ready and not seq215_smoke_contract_preflight_ready:
    current_blockers.append("seq215 smoke train requires challenger_seq215 215-dim 8-specialist contract preflight")
if not candidate_training_seq215_allowed:
    current_blockers.append("seq215 candidate training requires real seq215 smoke bundle edge audit")
if not iql_distillation_allowed:
    current_blockers.append("IQL distillation requires candidate train, selective-edge and replay evidence")
if entry_exit_handoff_decision == "READY_FOR_EXIT_PER_BAR_RECONSTRUCTION_REVIEW" and not entry_exit_reconstruction_ready:
    current_blockers.append("active Exit per-bar reconstruction audit required before Exit state/reward contract work")
if entry_exit_reconstruction_ready and not entry_exit_state_reward_ready:
    current_blockers.append("active Exit state/reward contract required before Exit split/leakage work")
if entry_exit_state_reward_ready and not entry_exit_split_leakage_ready:
    current_blockers.append("active Exit split/leakage audit required before Exit model dataset/readiness gates")
if entry_exit_split_leakage_ready and not entry_exit_model_dataset_ready:
    current_blockers.append("active Exit model dataset/readiness required before Exit Transformer architecture/readiness review")
if entry_exit_model_dataset_ready and not entry_exit_feature_alignment_ready:
    current_blockers.append("active Entry-to-Exit feature alignment is missing HH/SMC/trend/momentum/MTF/specialist-gate state before Exit training")
if entry_exit_feature_alignment_ready and not entry_exit_transformer_architecture_ready:
    current_blockers.append("active Exit Transformer architecture/readiness required before Exit training plan review")
if entry_exit_transformer_architecture_ready and not entry_exit_transformer_training_plan_ready:
    current_blockers.append("active Exit Transformer training plan/readiness required before trainer wrapper review")
if entry_exit_transformer_training_plan_ready and not entry_exit_transformer_trainer_wrapper_ready:
    current_blockers.append("fail-closed active Exit Transformer trainer wrapper readiness required before trainer implementation review")
if entry_exit_transformer_trainer_wrapper_ready and not entry_exit_transformer_pretrain_manifest_ready:
    current_blockers.append("active Exit Transformer pretrain manifest required before train-execution review")
if entry_exit_transformer_pretrain_manifest_ready and not entry_exit_model_dataset_slice_robustness_ready:
    current_blockers.append("active Exit model dataset slice robustness disclosure required before train-execution review")
if entry_exit_model_dataset_slice_robustness_ready and not entry_exit_transformer_train_execution_review_ready:
    current_blockers.append("active Exit Transformer train-execution review required before any training enablement package")
if entry_exit_transformer_train_execution_review_ready and not entry_exit_transformer_post_train_contract_ready:
    current_blockers.append("active Exit Transformer post-train audit contract required before any training enablement package")
if entry_exit_transformer_post_train_contract_ready and entry_exit_feature_alignment_ready and not entry_exit_transformer_train_enablement_ready:
    current_blockers.append("explicit Exit Transformer train-execution enablement vedtak package required; training remains closed")
if entry_exit_transformer_train_enablement_ready:
    current_blockers.append("Exit Transformer train package is ready, but shadow/live/promotion remain closed and training still requires the exact capped wrapper command")
if not iql_replay_evidence_ready:
    current_blockers.append("IQL replay evidence requires distillation contract and IQL-student replay trade log")
if not iql_replay_comparison_ready:
    current_blockers.append("promotion review requires candidate-vs-IQL replay comparison PASS")
if iql_replay_comparison_ready and not iql_replay_slice_audit_ready:
    current_blockers.append("promotion review requires IQL slice/tail audit PASS")

try:
    from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import (
        DEFAULT_BASE_SIGNAL_FEATURE_COUNT,
        SMART_LAYER_FEATURES,
    )
    smart_layer_feature_count = sum(
        len(features) for _, features, _, _ in SMART_LAYER_FEATURES.values()
    )
except Exception as exc:
    DEFAULT_BASE_SIGNAL_FEATURE_COUNT = 41
    smart_layer_feature_count = 0
    smart_layer_feature_count_error = str(exc)
else:
    smart_layer_feature_count_error = None
challenger_extension_counts = (
    (reports.get("challenger-extension-manifest") or {}).get("counts")
    if isinstance((reports.get("challenger-extension-manifest") or {}).get("counts"), dict)
    else {}
)
smart_candidate_base_signal_features = int(
    challenger_extension_counts.get("base_signal_features") or DEFAULT_BASE_SIGNAL_FEATURE_COUNT
)
smart_candidate_seq215_extension_features = int(
    challenger_extension_counts.get("combined_selected_features") or 174
)
smart_candidate_expected_signal_dim = (
    smart_candidate_base_signal_features
    + smart_candidate_seq215_extension_features
    + smart_layer_feature_count
)
smart_candidate_manifest_variant = f"smart_seq{smart_candidate_expected_signal_dim}_candidate"
smart_rebuild_preflight = reports.get("smart-rebuild-preflight") or {}
smart_rebuild_preflight_path = paths.get("smart-rebuild-preflight")
smart_post_rebuild = reports.get("smart-post-rebuild-readiness") or {}
smart_smoke_manifest = reports.get("smart-smoke-manifest") or {}
smart_smoke_readiness = reports.get("smart-smoke-readiness") or {}
smart_trainability_readiness = reports.get("smart-trainability-readiness") or {}
smart_ablation_replay_plan = reports.get("smart-ablation-replay-plan") or {}
smart_ablation_replay_matrix = reports.get("smart-ablation-replay-matrix") or {}
smart_replay_default = reports.get("replay-readiness-smart") or {}
smart_replay_selected = reports.get("replay-readiness-smart-selected") or {}
smart_post_rebuild_ready = (
    str(smart_post_rebuild.get("decision") or "") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
)
smart_smoke_manifest_ready = (
    str(smart_smoke_manifest.get("decision") or "") == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
)
smart_smoke_readiness_ready = (
    str(smart_smoke_readiness.get("decision") or "") == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
)
smart_trainability_readiness_ready = (
    str(smart_trainability_readiness.get("decision") or "") == "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW"
)
smart_ablation_replay_plan_ready = (
    str(smart_ablation_replay_plan.get("decision") or "") == "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW"
)
smart_ablation_replay_matrix_ready = (
    str(smart_ablation_replay_matrix.get("decision") or "") == "READY_FOR_SMART_ABLATION_REPLAY_MATRIX_REVIEW"
)
smart_replay_default_ready = (
    str(smart_replay_default.get("decision") or "") == "READY_FOR_IQL_DISTILLATION_VEDTAK"
)
smart_replay_selected_ready = (
    str(smart_replay_selected.get("decision") or "") == "READY_FOR_IQL_DISTILLATION_VEDTAK"
)
smart_rebuild_preflight_ready = (
    str(smart_rebuild_preflight.get("decision") or "") == "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW"
)
candidate_smart = reports.get("candidate-readiness-smart") or {}
candidate_training_smart_allowed = bool(candidate_smart.get("candidate_training_allowed_with_explicit_vedtak"))
real_smoke_train_smart_allowed = bool(smart_smoke_readiness_ready and smart_trainability_readiness_ready)

commands = {
    "handover": {
        "argv": ["scripts/entry_next_edge_control.sh", "handover"],
        "allowed": True,
        "mode": "report",
        "requires_vedtak": False,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Active seq146 session orientation.",
    },
    "readiness_report": {
        "argv": ["scripts/entry_next_edge_control.sh", "readiness-report"],
        "allowed": True,
        "mode": "report",
        "requires_vedtak": False,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Human-readable readiness snapshot.",
    },
    "readiness_report_json": {
        "argv": ["scripts/entry_next_edge_control.sh", "readiness-report", "--json"],
        "allowed": True,
        "mode": "report",
        "requires_vedtak": False,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Machine-readable readiness snapshot.",
    },
    "worktree_hygiene": {
        "argv": ["scripts/entry_next_edge_control.sh", "worktree-hygiene", "--no-fail-on-dirty"],
        "allowed": True,
        "mode": "audit",
        "requires_vedtak": False,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Refresh worktree hygiene report without changing the git index.",
    },
    "stage_foundation_cleanup_dry_run": {
        "argv": ["scripts/entry_next_edge_control.sh", "stage-foundation-cleanup", "--dry-run"],
        "allowed": True,
        "mode": "dry_run",
        "requires_vedtak": False,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Print audited foundation cleanup stage command without changing the git index.",
    },
    "stage_foundation_cleanup_apply": {
        "argv": ["scripts/entry_next_edge_control.sh", "stage-foundation-cleanup", "--apply", "--vedtak", "<id>"],
        "allowed": foundation_cleanup_stage_ready,
        "mode": "git_index_mutation",
        "requires_vedtak": True,
        "requires_clean_git": False,
        "mutates_git_index": True,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Stage audited foundation cleanup pathspec and require post-stage PASS_STAGED.",
    },
    "smoke_manifest": {
        "argv": ["scripts/entry_next_edge_control.sh", "smoke-manifest", "--vedtak", "<id>"],
        "allowed": smoke_manifest_proof_allowed,
        "mode": "proof_only",
        "requires_vedtak": True,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Run pre-train manifest proof and stop before trainer start.",
        "declares_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "num_workers": 0,
        "specialist_contract_mode": "foundation_seq146",
        "expected_signal_dim": 146,
        "required_training_specialist_count": 6,
    },
    "smoke_manifest_seq215": {
        "argv": ["scripts/entry_next_edge_control.sh", "smoke-manifest-seq215", "--vedtak", "<id>"],
        "allowed": smoke_manifest_seq215_proof_allowed,
        "mode": "proof_only",
        "requires_vedtak": True,
        "requires_clean_git": False,
        "mutates_git_index": False,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Run seq215 pre-train manifest proof and stop before trainer start.",
        "requires_seq215_vedtak": True,
        "declares_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "num_workers": 0,
        "specialist_contract_mode": "challenger_seq215",
        "expected_signal_dim": 215,
        "required_training_specialist_count": 8,
        "requires_exact_specialist_contract_proof": True,
    },
    "smoke_train": {
        "argv": [
            "scripts/entry_next_edge_control.sh",
            "smoke-train",
            "--vedtak",
            "<id>",
            "--require-edge-audit",
        ],
        "allowed": real_smoke_train_allowed,
        "mode": "train",
        "requires_vedtak": True,
        "requires_clean_git": True,
        "mutates_git_index": False,
        "starts_trainer": True,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Real smoke train; requires clean git and explicit vedtak.",
        "requires_edge_audit": True,
        "requires_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "num_workers": 0,
        "specialist_contract_mode": "foundation_seq146",
        "expected_signal_dim": 146,
        "required_training_specialist_count": 6,
    },
    "smoke_train_seq215": {
        "argv": [
            "scripts/entry_next_edge_control.sh",
            "smoke-train-seq215",
            "--vedtak",
            "<id>",
            "--require-edge-audit",
        ],
        "allowed": real_smoke_train_seq215_allowed,
        "mode": "train",
        "requires_vedtak": True,
        "requires_clean_git": True,
        "mutates_git_index": False,
        "starts_trainer": True,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Real seq215 smoke train; requires clean git and explicit SEQ215 vedtak.",
        "requires_seq215_vedtak": True,
        "requires_edge_audit": True,
        "requires_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "num_workers": 0,
        "specialist_contract_mode": "challenger_seq215",
        "expected_signal_dim": 215,
        "required_training_specialist_count": 8,
        "requires_exact_specialist_contract_proof": True,
    },
    "smart_smoke_train": {
        "argv": [
            "scripts/entry_next_edge_control.sh",
            "smart-smoke-train",
            "--vedtak",
            "<id>",
            "--require-edge-audit",
        ],
        "allowed": real_smoke_train_smart_allowed,
        "mode": "train",
        "requires_vedtak": True,
        "requires_clean_git": True,
        "mutates_git_index": False,
        "starts_trainer": True,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "description": "Real smart seq520 smoke train; requires smart smoke-readiness, trainability, clean git and explicit SMART/SEQ520 vedtak.",
        "requires_smart_vedtak": True,
        "requires_edge_audit": True,
        "requires_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "num_workers": 0,
        "specialist_contract_mode": "smart_seq520_candidate",
        "expected_signal_dim": 520,
        "required_training_specialist_count": 8,
        "requires_exact_specialist_contract_proof": True,
    },
}

commands.update(
    {
        "verify": {
            "argv": ["scripts/entry_next_edge_control.sh", "verify", "--quiet"],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Active foundation state verifier.",
        },
        "selftest": {
            "argv": ["scripts/entry_next_edge_control.sh", "selftest", "--quiet"],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Foundation verifier self-test.",
        },
        "foundation_guardrails": {
            "argv": ["scripts/entry_next_edge_control.sh", "foundation-guardrails", "--quiet"],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Control-surface guardrails and legacy-block audit.",
        },
        "foundation_activation_plan": {
            "argv": ["scripts/entry_next_edge_control.sh", "foundation-activation-plan"],
            "allowed": True,
            "mode": "report",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Report-only activation plan for a green adoption candidate.",
        },
        "foundation_activation_apply_dry_run": {
            "argv": activation_apply_dry_run_argv or [
                "scripts/entry_next_edge_control.sh",
                "foundation-activation-apply",
                "--plan-json",
                "<activation-plan>",
                "--dry-run",
            ],
            "allowed": bool(activation_plan_ready and activation_apply_dry_run_argv),
            "mode": "dry_run",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "mutates_foundation_paths": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Dry-run activation apply proof; no filesystem mutation.",
        },
        "foundation_activation_apply": {
            "argv": activation_apply_argv,
            "allowed": activation_apply_allowed_after_vedtak,
            "mode": "foundation_path_mutation",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "mutates_foundation_paths": True,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Vedtak-gated canonical foundation dataset alias switch; does not train.",
        },
        "foundation_activation_post_apply_dry_run": {
            "argv": activation_post_apply_dry_run_argv or [
                "scripts/entry_next_edge_control.sh",
                "foundation-activation-post-apply",
                "--activation-apply-json",
                "<activation-apply-report>",
                "--dry-run",
            ],
            "allowed": bool(activation_apply_path),
            "mode": "dry_run",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "mutates_foundation_paths": False,
            "mutates_foundation_audits": False,
            "materializes_smoke_dataset": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Dry-run post-activation audit refresh proof; no filesystem mutation.",
        },
        "foundation_activation_post_apply": {
            "argv": activation_post_apply_argv,
            "allowed": activation_post_apply_allowed_after_vedtak,
            "mode": "audit_refresh_mutation",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "mutates_foundation_paths": False,
            "mutates_foundation_audits": True,
            "materializes_smoke_dataset": True,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Vedtak-gated post-activation refresh of canonical audits and smoke dataset; does not train.",
        },
        "train_readiness_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "train-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Refresh train-readiness without opening trainer start.",
        },
        "candidate_readiness_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "candidate-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Refresh candidate-readiness without opening candidate training.",
        },
        "candidate_readiness_seq215_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "candidate-readiness-seq215",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Refresh seq215 candidate-readiness without opening candidate training.",
        },
        "candidate_readiness_smart_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "candidate-readiness-smart",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "specialist_contract_mode": "smart_seq520_candidate",
            "expected_signal_dim": 520,
            "training_allowed": False,
            "candidate_training_allowed": False,
            "replay_allowed": False,
            "description": "Refresh smart seq520 candidate-readiness without opening candidate training.",
        },
        "replay_readiness_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "replay-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Refresh replay-readiness without opening IQL distillation.",
        },
        "replay_readiness_seq215_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "replay-readiness-seq215",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "specialist_contract_mode": "challenger_seq215",
            "expected_signal_dim": 215,
            "required_training_specialist_count": 8,
            "description": "Refresh seq215 replay-readiness without opening IQL distillation.",
        },
        "replay_readiness_smart_report": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "replay-readiness-smart",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "specialist_contract_mode": "smart_seq520_candidate",
            "expected_signal_dim": 520,
            "training_allowed": False,
            "candidate_training_allowed": False,
            "replay_allowed": False,
            "description": "Refresh smart seq520 replay-readiness without opening IQL distillation.",
        },
        "chart_geometry_audit": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "chart-geometry-audit",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit numeric chart-geometry challenger features and manifest; no training or replay.",
        },
        "candlestick_audit": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "candlestick-audit",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit closed-bar candlestick pattern challenger features and manifest; no training or replay.",
        },
        "challenger_extension_manifest": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "challenger-extension-manifest",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "report",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize combined foundation+chart+candlestick sequence extension manifest for a later gated dataset rebuild.",
        },
        "challenger_smart_extension_manifest": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "challenger-smart-extension-manifest",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "report",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "smart_layer_feature_count": smart_layer_feature_count,
            "smart_layer_feature_count_error": smart_layer_feature_count_error,
            "description": (
                "Materialize report-only smart-layer extension manifest for a later "
                "separately gated dataset rebuild; no training or replay."
            ),
        },
        "smart_rebuild_preflight": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-rebuild-preflight",
                "--verify-large-input-hashes",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "report",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "dataset_rebuild_allowed_without_vedtak": False,
            "verifies_large_input_hashes": True,
            "requires_ram_cap_for_future_rebuild": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "smart_layer_feature_count": smart_layer_feature_count,
            "description": (
                "Preflight the smart capped dataset rebuild contract; "
                "report-only, no dataset rebuild, training or replay."
            ),
        },
        "smart_post_rebuild_readiness": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-post-rebuild-readiness",
                "--fullscan",
                "--verify-source-parquet-hashes",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "specialist_contract_mode": "smart_seq520_candidate",
            "training_allowed": False,
            "description": (
                "Report-only post-rebuild audit for the smart seq/snap dataset; "
                "checks manifests, hashes, finite values and specialist liveness without training."
            ),
        },
        "smart_post_rebuild_refresh": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-post-rebuild-refresh",
                "--apply",
                "--vedtak",
                "<id>",
            ],
            "allowed": smart_post_rebuild_ready,
            "mode": "smoke_dataset_materialization",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "writes_smoke_dataset": True,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "specialist_contract_mode": "smart_seq520_candidate",
            "training_allowed": False,
            "replay_allowed": False,
            "declares_ram_cap": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "memory_cap": "8G",
            "swap_cap": "1G",
            "num_workers": 0,
            "description": (
                "Vedtak-gated smart smoke dataset materialization after post-rebuild readiness; "
                "copies audited smart splits only and never starts the trainer."
            ),
        },
        "smart_smoke_manifest": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-smoke-manifest",
                "--vedtak",
                "<id>",
            ],
            "allowed": smart_post_rebuild_ready,
            "mode": "proof_only",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "specialist_contract_mode": "smart_seq520_candidate",
            "training_allowed": False,
            "replay_allowed": False,
            "declares_ram_cap": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "num_workers": 0,
            "description": (
                "Vedtak-gated report-only smart smoke manifest proof; writes report-dir manifest "
                "and never starts the trainer."
            ),
        },
        "smart_smoke_readiness": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-smoke-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "specialist_contract_mode": "smart_seq520_candidate",
            "training_allowed": False,
            "requires_clean_git_before_future_train": True,
            "requires_ram_cap_for_future_train": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "num_workers": 0,
            "description": (
                "Report-only smart smoke-readiness gate; proves audit/smoke-manifest/RAM-cap contract "
                "before any future vedtak-gated smart smoke trainer can be reviewed."
            ),
        },
        "smart_trainability_readiness": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-trainability-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "implementation_readiness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "specialist_contract_mode": "smart_seq520_candidate",
            "training_allowed": False,
            "candidate_training_allowed": False,
            "replay_allowed": False,
            "description": (
                "Report-only smart trainability gate; proves trainer/wrapper/control/candidate/replay "
                "plumbing before any future smart training vedtak can be reviewed."
            ),
        },
        "smart_ablation_replay_plan": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-ablation-replay-plan",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "evidence_plan",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "training_allowed": False,
            "replay_allowed_by_this_gate": False,
            "description": (
                "Report-only smart ablation/replay evidence plan; requires future explicit replay evidence "
                "but never starts replay from this gate."
            ),
        },
        "smart_ablation_replay_matrix": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "smart-ablation-replay-matrix",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
            "allowed": True,
            "mode": "evidence_matrix_gate",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "manifest_variant": smart_candidate_manifest_variant,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "training_allowed": False,
            "replay_allowed_by_this_gate": False,
            "ready": smart_ablation_replay_matrix_ready,
            "description": (
                "Report-only gate for the complete smart ablation replay matrix. "
                "Requires all 14 arm replay artifacts and never starts replay itself."
            ),
        },
        "feature_ai_inventory": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "feature-ai-inventory",
                "--quiet",
                "--no-fail-on-audit-fail",
            ],
            "allowed": True,
            "mode": "report",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize all active Entry inputs/features and ranked specialist-AI plan.",
        },
        "candidate_train": {
            "argv": ["scripts/entry_next_edge_control.sh", "candidate-train", "--vedtak", "<id>"],
            "allowed": candidate_training_allowed,
            "mode": "train",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Full candidate train after real smoke edge audit.",
            "requires_smoke_bundle_edge_audit": True,
            "requires_ram_cap": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "num_workers": 0,
            "specialist_contract_mode": "foundation_seq146",
            "expected_signal_dim": 146,
            "required_training_specialist_count": 6,
        },
        "candidate_train_seq215": {
            "argv": ["scripts/entry_next_edge_control.sh", "candidate-train-seq215", "--vedtak", "<id>"],
            "allowed": candidate_training_seq215_allowed,
            "mode": "train",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Seq215 candidate train after real seq215 smoke edge audit and seq215 candidate-readiness.",
            "requires_seq215_vedtak": True,
            "requires_candidate_readiness": True,
            "requires_smoke_bundle_edge_audit": True,
            "requires_ram_cap": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "num_workers": 0,
            "specialist_contract_mode": "challenger_seq215",
            "expected_signal_dim": 215,
            "required_training_specialist_count": 8,
            "requires_exact_specialist_contract_proof": True,
        },
        "candidate_train_smart": {
            "argv": ["scripts/entry_next_edge_control.sh", "candidate-train-smart", "--vedtak", "<id>"],
            "allowed": candidate_training_smart_allowed,
            "mode": "train",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Smart seq520 candidate train after smart smoke edge audit and smart candidate-readiness.",
            "requires_smart_vedtak": True,
            "requires_candidate_readiness": True,
            "requires_smoke_bundle_edge_audit": True,
            "requires_ram_cap": True,
            "ram_cap_runner": "scripts/gx1_capped_run.sh",
            "num_workers": 0,
            "specialist_contract_mode": "smart_seq520_candidate",
            "expected_signal_dim": 520,
            "required_training_specialist_count": 8,
            "requires_exact_specialist_contract_proof": True,
        },
        "selective_edge": {
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "selective-edge",
                "--bundle-dir",
                "<candidate>",
                "--no-xgb-bundle-dir",
                "<ablation>",
            ],
            "allowed": False,
            "mode": "evidence_eval",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Post-candidate selective-edge/no-XGB evidence writer.",
        },
        "replay_evidence": {
            "argv": ["scripts/entry_next_edge_control.sh", "replay-evidence", "--trades-path", "<csv|parquet>"],
            "allowed": False,
            "mode": "evidence_materializer",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize candidate replay evidence from an explicit trade log.",
        },
        "iql_distill": {
            "argv": ["scripts/entry_next_edge_control.sh", "iql-distill", "--vedtak", "<id>"],
            "allowed": iql_distillation_allowed,
            "mode": "iql_distillation_contract",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": True,
            "touches_shadow_or_live": False,
            "description": "Vedtak-gated IQL distillation contract after replay-readiness PASS.",
        },
        "iql_student_trade_log": {
            "argv": ["scripts/entry_next_edge_control.sh", "iql-student-trade-log", "--vedtak", "<id>"],
            "allowed": iql_student_trade_log_allowed,
            "mode": "offline_iql_student_trade_log",
            "requires_vedtak": True,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": True,
            "starts_replay": False,
            "starts_iql_distillation": True,
            "touches_shadow_or_live": False,
            "description": "Vedtak-gated offline IQL-student policy fit and explicit trade-log materializer.",
        },
        "iql_replay_evidence": {
            "argv": ["scripts/entry_next_edge_control.sh", "iql-replay-evidence", "--trades-path", "<csv|parquet>"],
            "allowed": False,
            "mode": "evidence_materializer",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize IQL-student replay evidence from an explicit trade log.",
        },
        "iql_compare": {
            "argv": ["scripts/entry_next_edge_control.sh", "iql-compare"],
            "allowed": False,
            "mode": "comparison_gate",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Final IQL-vs-candidate replay comparison gate.",
        },
        "iql_slice_audit": {
            "argv": ["scripts/entry_next_edge_control.sh", "iql-slice-audit"],
            "allowed": True,
            "mode": "slice_tail_audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Report-only IQL-vs-candidate session/regime/side/tail slice audit.",
        },
        "entry_exit_handoff": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-handoff"],
            "allowed": True,
            "mode": "handoff_audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Report-only Entry/IQL evidence to Exit per-bar substrate handoff audit.",
        },
        "entry_exit_materialize": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-materialize"],
            "allowed": True,
            "mode": "data_materializer",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize active Entry-bound per-bar Exit handoff substrate; no training or replay.",
        },
        "entry_exit_reconstruction_audit": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-reconstruction-audit"],
            "allowed": True,
            "mode": "exit_reconstruction_audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Report-only active Exit per-bar reconstruction audit before state/reward contract work.",
        },
        "entry_exit_state_reward_contract": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-state-reward-contract"],
            "allowed": True,
            "mode": "exit_state_reward_contract",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize active Exit HOLD/EXIT_NOW state/reward contract; no training or replay.",
        },
        "entry_exit_split_leakage_audit": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-split-leakage-audit"],
            "allowed": True,
            "mode": "exit_split_leakage_audit",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit active Exit train/val/test split and leakage gates; no training or replay.",
        },
        "entry_exit_model_dataset_readiness": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-model-dataset-readiness"],
            "allowed": True,
            "mode": "exit_model_dataset_readiness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize active Exit model dataset shards/schema/readiness; no training or replay.",
        },
        "entry_exit_feature_alignment": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-feature-alignment"],
            "allowed": True,
            "mode": "entry_exit_feature_alignment",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit Entry-to-Exit market-mechanism feature alignment; no training or replay.",
        },
        "entry_exit_transformer_architecture_readiness": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-architecture-readiness"],
            "allowed": True,
            "mode": "exit_transformer_architecture_readiness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit active Exit Transformer architecture/readiness contract; no training or replay.",
        },
        "entry_exit_transformer_training_plan_readiness": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-training-plan-readiness"],
            "allowed": True,
            "mode": "exit_transformer_training_plan_readiness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize active Exit Transformer training plan/readiness manifest; no training or replay.",
        },
        "entry_exit_transformer_trainer_wrapper_readiness": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-trainer-wrapper-readiness"],
            "allowed": True,
            "mode": "exit_transformer_trainer_wrapper_readiness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit the fail-closed active Exit Transformer trainer wrapper; no training or replay.",
        },
        "entry_exit_transformer_pretrain_manifest": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-pretrain-manifest"],
            "allowed": True,
            "mode": "exit_transformer_pretrain_manifest",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize active Exit Transformer pretrain manifest with finite forward preflight; no training or replay.",
        },
        "entry_exit_model_dataset_slice_robustness": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-model-dataset-slice-robustness"],
            "allowed": True,
            "mode": "exit_model_dataset_slice_robustness",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Audit active Exit session/regime/side slice robustness and weak-slice disclosure; no training or replay.",
        },
        "entry_exit_transformer_train_execution_review": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-train-execution-review"],
            "allowed": True,
            "mode": "exit_transformer_train_execution_review",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Review active Exit Transformer train-execution contract, RAM guard and weak-slice policy; no training or replay.",
        },
        "entry_exit_transformer_post_train_contract": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-post-train-contract"],
            "allowed": True,
            "mode": "exit_transformer_post_train_contract",
            "requires_vedtak": False,
            "requires_clean_git": False,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Lock active Exit Transformer post-train bundle audit contract; no training or replay.",
        },
        "entry_exit_transformer_train_enablement": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-train-enablement", "--vedtak", "<id>"],
            "allowed": False,
            "mode": "exit_transformer_train_enablement",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Materialize explicit active Exit Transformer train-enablement package; dry-run only, no trainer start.",
        },
        "entry_exit_transformer_train": {
            "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-train", "--vedtak", "<id>"],
            "allowed": False,
            "mode": "exit_transformer_train",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "description": "Blocked active Exit Transformer trainer path; train enablement remains closed.",
        },
        "preview_shadow": {
            "argv": ["scripts/entry_next_edge_control.sh", "preview-shadow"],
            "allowed": False,
            "mode": "blocked_shadow_live",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": True,
            "description": "Blocked until promotion review explicitly opens shadow/live paths.",
        },
        "start_shadow": {
            "argv": ["scripts/entry_next_edge_control.sh", "start-shadow"],
            "allowed": False,
            "mode": "blocked_shadow_live",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": True,
            "description": "Blocked until promotion review explicitly opens shadow/live paths.",
        },
        "live": {
            "argv": ["scripts/entry_next_edge_control.sh", "live"],
            "allowed": False,
            "mode": "blocked_shadow_live",
            "requires_vedtak": True,
            "requires_clean_git": True,
            "mutates_git_index": False,
            "starts_trainer": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": True,
            "description": "Blocked order path; no live/practice orders from Entry foundation control.",
        },
    }
)

execution_allowed_now = {
    "handover": True,
    "readiness_report": True,
    "readiness_report_json": True,
    "verify": True,
    "selftest": True,
    "foundation_guardrails": True,
    "foundation_activation_plan": True,
    "foundation_activation_apply_dry_run": bool(activation_plan_ready and activation_apply_dry_run_argv),
    "foundation_activation_apply": False,
    "foundation_activation_post_apply_dry_run": bool(activation_post_apply_dry_run_argv),
    "foundation_activation_post_apply": False,
    "worktree_hygiene": True,
    "train_readiness_report": True,
    "candidate_readiness_report": True,
    "candidate_readiness_seq215_report": True,
    "candidate_readiness_smart_report": True,
    "replay_readiness_report": True,
    "replay_readiness_seq215_report": True,
    "replay_readiness_smart_report": True,
    "feature_ai_inventory": True,
    "chart_geometry_audit": True,
    "candlestick_audit": True,
    "challenger_extension_manifest": True,
    "challenger_smart_extension_manifest": True,
    "smart_rebuild_preflight": True,
    "smart_post_rebuild_readiness": True,
    "smart_post_rebuild_refresh": False,
    "smart_smoke_manifest": False,
    "smart_smoke_readiness": True,
    "smart_trainability_readiness": True,
    "smart_ablation_replay_plan": True,
    "smart_ablation_replay_matrix": True,
    "stage_foundation_cleanup_dry_run": True,
    "stage_foundation_cleanup_apply": False,
    "smoke_manifest": False,
    "smoke_manifest_seq215": False,
    "smoke_train": False,
    "smoke_train_seq215": False,
    "smart_smoke_train": False,
    "candidate_train": False,
    "candidate_train_seq215": False,
    "candidate_train_smart": False,
    "selective_edge": False,
    "replay_evidence": False,
    "iql_distill": False,
    "iql_student_trade_log": False,
    "iql_replay_evidence": False,
    "iql_compare": False,
    "iql_slice_audit": True,
    "entry_exit_materialize": True,
    "entry_exit_handoff": True,
    "entry_exit_reconstruction_audit": True,
    "entry_exit_state_reward_contract": True,
    "entry_exit_split_leakage_audit": True,
    "entry_exit_model_dataset_readiness": True,
    "entry_exit_feature_alignment": True,
    "entry_exit_transformer_architecture_readiness": True,
    "entry_exit_transformer_training_plan_readiness": True,
    "entry_exit_transformer_trainer_wrapper_readiness": True,
    "entry_exit_transformer_pretrain_manifest": True,
    "entry_exit_model_dataset_slice_robustness": True,
    "entry_exit_transformer_train_execution_review": True,
    "entry_exit_transformer_post_train_contract": True,
    "entry_exit_transformer_train_enablement": False,
    "entry_exit_transformer_train": False,
    "preview_shadow": False,
    "start_shadow": False,
    "live": False,
}
allowed_after_explicit_vedtak = {
    "handover": True,
    "readiness_report": True,
    "readiness_report_json": True,
    "verify": True,
    "selftest": True,
    "foundation_guardrails": True,
    "foundation_activation_plan": True,
    "foundation_activation_apply_dry_run": bool(activation_plan_ready and activation_apply_dry_run_argv),
    "foundation_activation_apply": activation_apply_allowed_after_vedtak,
    "foundation_activation_post_apply_dry_run": bool(activation_post_apply_dry_run_argv),
    "foundation_activation_post_apply": activation_post_apply_allowed_after_vedtak,
    "worktree_hygiene": True,
    "train_readiness_report": True,
    "candidate_readiness_report": True,
    "candidate_readiness_seq215_report": True,
    "candidate_readiness_smart_report": True,
    "replay_readiness_report": True,
    "replay_readiness_seq215_report": True,
    "replay_readiness_smart_report": True,
    "feature_ai_inventory": True,
    "chart_geometry_audit": True,
    "candlestick_audit": True,
    "challenger_extension_manifest": True,
    "challenger_smart_extension_manifest": True,
    "smart_rebuild_preflight": True,
    "smart_post_rebuild_readiness": True,
    "smart_post_rebuild_refresh": smart_post_rebuild_ready,
    "smart_smoke_manifest": smart_post_rebuild_ready,
    "smart_smoke_readiness": True,
    "smart_trainability_readiness": True,
    "smart_ablation_replay_plan": True,
    "smart_ablation_replay_matrix": True,
    "stage_foundation_cleanup_dry_run": True,
    "stage_foundation_cleanup_apply": foundation_cleanup_stage_ready,
    "smoke_manifest": smoke_manifest_proof_allowed,
    "smoke_manifest_seq215": smoke_manifest_seq215_proof_allowed,
    "smoke_train": real_smoke_train_allowed,
    "smoke_train_seq215": real_smoke_train_seq215_allowed,
    "smart_smoke_train": real_smoke_train_smart_allowed,
    "candidate_train": candidate_training_allowed,
    "candidate_train_seq215": candidate_training_seq215_allowed,
    "candidate_train_smart": candidate_training_smart_allowed,
    "selective_edge": False,
    "replay_evidence": False,
    "iql_distill": iql_distillation_allowed,
    "iql_student_trade_log": iql_student_trade_log_allowed,
    "iql_replay_evidence": False,
    "iql_compare": False,
    "iql_slice_audit": True,
    "entry_exit_materialize": True,
    "entry_exit_handoff": True,
    "entry_exit_reconstruction_audit": True,
    "entry_exit_state_reward_contract": True,
    "entry_exit_split_leakage_audit": True,
    "entry_exit_model_dataset_readiness": True,
    "entry_exit_feature_alignment": True,
    "entry_exit_transformer_architecture_readiness": True,
    "entry_exit_transformer_training_plan_readiness": True,
    "entry_exit_transformer_trainer_wrapper_readiness": True,
    "entry_exit_transformer_pretrain_manifest": True,
    "entry_exit_model_dataset_slice_robustness": True,
    "entry_exit_transformer_train_execution_review": True,
    "entry_exit_transformer_post_train_contract": True,
    "entry_exit_transformer_train_enablement": bool(entry_exit_transformer_post_train_contract_ready and entry_exit_feature_alignment_ready),
    "entry_exit_transformer_train": False,
    "preview_shadow": False,
    "start_shadow": False,
    "live": False,
}
activation_apply_not_executable_now_reason = (
    "activation apply already completed; canonical foundation alias switch is complete"
    if activation_apply_applied
    else "requires explicit activation vedtak and mutates canonical foundation dataset path"
)
activation_post_apply_not_executable_now_reason = (
    "post-apply refresh already completed; no post-activation audit mutation is currently open"
    if activation_post_apply_completed
    else "requires activation apply APPLIED_ALIAS_SWITCH, mutation_performed=true, and explicit post-apply vedtak"
)

not_executable_now_reason = {
    "stage_foundation_cleanup_apply": "requires explicit staging vedtak and mutates git index",
    "foundation_activation_apply": activation_apply_not_executable_now_reason,
    "foundation_activation_post_apply": activation_post_apply_not_executable_now_reason,
    "smoke_manifest": "requires explicit smoke-manifest vedtak; proof-only no trainer start",
    "smoke_manifest_seq215": "requires explicit SEQ215 smoke-manifest vedtak; proof-only no trainer start",
    "smart_smoke_manifest": (
        "requires smart post-rebuild dataset readiness PASS and explicit smart smoke-manifest vedtak"
        if not smart_post_rebuild_ready
        else "requires explicit smart smoke-manifest vedtak; proof-only no trainer start"
    ),
    "smart_post_rebuild_refresh": (
        "requires smart post-rebuild dataset readiness PASS and explicit smart post-rebuild refresh vedtak"
        if not smart_post_rebuild_ready
        else "requires explicit smart post-rebuild refresh vedtak; materializes smoke dataset only"
    ),
    "smoke_train": (
        "requires clean git worktree and explicit smoke-train vedtak"
        if foundation_ready
        else "foundation contract is not ready for smoke"
    ),
    "smoke_train_seq215": (
        "foundation contract is not ready for smoke"
        if not foundation_ready
        else (
            "seq215 smoke contract preflight is not ready"
            if not seq215_smoke_contract_preflight_ready
            else "requires clean git worktree and explicit SEQ215 smoke-train vedtak"
        )
    ),
    "smart_smoke_train": (
        "smart smoke-readiness and smart trainability are not both ready"
        if not real_smoke_train_smart_allowed
        else "requires clean git worktree and explicit SMART/SEQ520 smoke-train vedtak"
    ),
    "candidate_train": "requires real smoke bundle edge audit, clean git worktree and explicit candidate-train vedtak",
    "candidate_train_seq215": "requires real seq215 smoke bundle edge audit, clean git worktree and explicit seq215 candidate-train vedtak",
    "candidate_train_smart": "requires smart smoke bundle edge audit, clean git worktree and explicit SMART/SEQ520 candidate-train vedtak",
    "selective_edge": "requires actual candidate bundle and no-XGB ablation bundle",
    "replay_evidence": "requires explicit post-candidate replay trade log and candidate/selective-edge evidence",
    "iql_distill": "requires replay-readiness PASS and explicit IQL vedtak",
    "iql_student_trade_log": "requires ready IQL distillation contract and explicit IQL vedtak",
    "iql_replay_evidence": "requires IQL distillation contract and explicit IQL replay trade log",
    "iql_compare": "requires candidate and IQL replay evidence plus preserved distillation identity",
    "entry_exit_transformer_train_enablement": "requires clean git, ready Exit post-train audit contract and explicit Exit Transformer train vedtak; dry-run package only",
    "entry_exit_transformer_train": "requires active Entry-to-Exit feature alignment, post-train audit contract, clean git and explicit Exit train enablement vedtak",
    "preview_shadow": "shadow/live remains blocked until promotion review explicitly opens it",
    "start_shadow": "shadow/live remains blocked until promotion review explicitly opens it",
    "live": "live/practice order path remains blocked during Entry foundation work",
}
for name, command in commands.items():
    command["execution_allowed_now"] = bool(execution_allowed_now.get(name))
    command["allowed_after_explicit_vedtak"] = bool(allowed_after_explicit_vedtak.get(name))
    command["not_executable_now_reason"] = not_executable_now_reason.get(name)

payload = {
    "schema_version": "entry_next_edge_readiness_report_v1",
    "report_only": True,
    "refresh_skipped": refresh_skipped,
    "status_summary": {
        "foundation_contract_ready_for_smoke": foundation_ready,
        "foundation_adoption_candidate_ready": adoption_candidate_ready,
        "foundation_adoption_candidate_report": str(adoption_candidate_path) if adoption_candidate_path else None,
        "foundation_adoption_candidate_dataset_dir": adoption_candidate_artifacts.get("candidate_dataset_dir"),
        "foundation_adoption_candidate_smoke_dataset_dir": adoption_candidate_artifacts.get("candidate_smoke_dataset_dir"),
        "foundation_activation_plan_ready": activation_plan_ready,
        "foundation_activation_plan_report": str(activation_plan_path) if activation_plan_path else None,
        "foundation_activation_plan_strategy": activation_plan.get("recommended_strategy"),
        "foundation_activation_required_before_smoke": train_activation_required_before_smoke,
        "foundation_activation_apply_required_before_smoke": train_activation_apply_required_before_smoke,
        "foundation_activation_post_apply_required_before_smoke": train_activation_post_apply_required_before_smoke,
        "foundation_activation_next_command": train.get("next_allowed_command") if train_activation_required_before_smoke else None,
        "foundation_activation_apply_command": activation_apply_argv,
        "foundation_activation_apply_ready": activation_apply_ready,
        "foundation_activation_apply_report": str(activation_apply_path) if activation_apply_path else None,
        "foundation_activation_apply_mutation_performed": bool(activation_apply.get("mutation_performed")),
        "foundation_activation_apply_post_apply_command_count": len(activation_apply_post_commands),
        "foundation_activation_post_apply_report": str(activation_post_apply_path) if activation_post_apply_path else None,
        "foundation_activation_post_apply_waiting_for_activation": activation_post_apply_waiting,
        "foundation_activation_post_apply_ready": activation_post_apply_ready,
        "foundation_activation_post_apply_completed": activation_post_apply_completed,
        "foundation_activation_post_apply_mutations_performed": activation_post_apply_mutations_performed,
        "foundation_activation_post_apply_next_action": activation_post_apply.get("next_required_action"),
        "foundation_activation_post_apply_dry_run_command": activation_post_apply_dry_run_argv,
        "foundation_activation_post_apply_command": activation_post_apply_argv,
        "activation_allowed_without_vedtak": bool(adoption.get("activation_allowed_without_vedtak")),
        "real_smoke_train_allowed": real_smoke_train_allowed,
        "smoke_manifest_proof_allowed": smoke_manifest_proof_allowed,
        "foundation_cleanup_stage_ready": foundation_cleanup_stage_ready,
        "stage_plan_safe": bool(hygiene.get("stage_plan_safe")),
        "clean_git_resolution_decision": (hygiene.get("clean_git_resolution") or {}).get("decision"),
        "candidate_training_allowed": bool(
            candidate_training_allowed
        ),
        "candidate_training_foundation_seq146_allowed": bool(
            candidate_training_allowed
        ),
        "candidate_training_seq215_allowed": bool(
            candidate_training_seq215_allowed
        ),
        "candidate_training_smart_seq520_allowed": bool(
            candidate_training_smart_allowed
        ),
        "candidate_readiness_seq215_decision": candidate_seq215.get("decision"),
        "candidate_readiness_seq215_next": (
            candidate_seq215.get("next_required_gate")
            or summaries.get("candidate-readiness-seq215", {}).get("next")
        ),
        "candidate_readiness_smart_decision": candidate_smart.get("decision"),
        "candidate_readiness_smart_next": (
            candidate_smart.get("next_required_gate")
            or summaries.get("candidate-readiness-smart", {}).get("next")
        ),
        "seq215_smoke_contract_preflight_ready": seq215_smoke_contract_preflight_ready,
        "seq215_smoke_contract_required_specialists": sorted(candidate_seq215_required_specialists),
        "smoke_manifest_seq215_proof_allowed": smoke_manifest_seq215_proof_allowed,
        "real_smoke_train_seq215_allowed": real_smoke_train_seq215_allowed,
        "smart_candidate_manifest_variant": smart_candidate_manifest_variant,
        "smart_candidate_expected_signal_dim": smart_candidate_expected_signal_dim,
        "smart_candidate_smart_layer_feature_count": smart_layer_feature_count,
        "smart_rebuild_preflight_decision": smart_rebuild_preflight.get("decision"),
        "smart_rebuild_preflight_ready": smart_rebuild_preflight_ready,
        "smart_rebuild_preflight_report": str(smart_rebuild_preflight_path) if smart_rebuild_preflight_path else None,
        "smart_rebuild_preflight_sha256": sha256_file(smart_rebuild_preflight_path),
        "smart_post_rebuild_readiness_decision": smart_post_rebuild.get("decision"),
        "smart_post_rebuild_readiness_ready": smart_post_rebuild_ready,
        "smart_smoke_manifest_decision": smart_smoke_manifest.get("decision"),
        "smart_smoke_manifest_ready": smart_smoke_manifest_ready,
        "smart_smoke_readiness_decision": smart_smoke_readiness.get("decision"),
        "smart_smoke_readiness_ready": smart_smoke_readiness_ready,
        "smart_trainability_readiness_decision": smart_trainability_readiness.get("decision"),
        "smart_trainability_readiness_ready": smart_trainability_readiness_ready,
        "smart_ablation_replay_plan_decision": smart_ablation_replay_plan.get("decision"),
        "smart_ablation_replay_plan_ready": smart_ablation_replay_plan_ready,
        "smart_ablation_replay_matrix_decision": smart_ablation_replay_matrix.get("decision"),
        "smart_ablation_replay_matrix_ready": smart_ablation_replay_matrix_ready,
        "smart_ablation_replay_matrix_failures": int(len(smart_ablation_replay_matrix.get("failures") or [])),
        "smart_ablation_replay_matrix_report": str(paths.get("smart-ablation-replay-matrix"))
        if paths.get("smart-ablation-replay-matrix")
        else None,
        "smart_replay_default_readiness_decision": smart_replay_default.get("decision"),
        "smart_replay_default_readiness_ready": smart_replay_default_ready,
        "smart_replay_default_readiness_report": str(paths.get("replay-readiness-smart")) if paths.get("replay-readiness-smart") else None,
        "smart_selected_replay_readiness_decision": smart_replay_selected.get("decision"),
        "smart_selected_replay_readiness_ready": smart_replay_selected_ready,
        "smart_selected_replay_readiness_report": str(paths.get("replay-readiness-smart-selected")) if paths.get("replay-readiness-smart-selected") else None,
        "smart_selected_replay_dir": smart_replay_selected.get("replay_dir"),
        "smart_selected_iql_distillation_allowed": bool(
            smart_replay_selected.get("iql_distillation_allowed_with_explicit_vedtak")
        ),
        "real_smoke_train_smart_seq520_allowed": real_smoke_train_smart_allowed,
        "iql_distillation_allowed": bool(
            iql_distillation_allowed
        ),
        "iql_replay_evidence_ready": iql_replay_evidence_ready,
        "iql_replay_comparison_ready": iql_replay_comparison_ready,
        "iql_replay_slice_audit_ready": iql_replay_slice_audit_ready,
        "entry_exit_handoff_decision": entry_exit_handoff_decision,
        "entry_exit_handoff_entry_ready": entry_exit_handoff_entry_ready,
        "entry_exit_handoff_substrate_ready": entry_exit_handoff_substrate_ready,
        "entry_exit_per_bar_decision": entry_exit_per_bar_decision,
        "entry_exit_per_bar_ready": entry_exit_per_bar_ready,
        "entry_exit_per_bar_included_trade_count": entry_exit_per_bar.get("included_trade_count"),
        "entry_exit_per_bar_excluded_trade_count": entry_exit_per_bar.get("excluded_trade_count"),
        "entry_exit_per_bar_covered_trade_ratio": entry_exit_per_bar.get("covered_trade_ratio"),
        "entry_exit_reconstruction_decision": entry_exit_reconstruction_decision,
        "entry_exit_reconstruction_ready": entry_exit_reconstruction_ready,
        "entry_exit_reconstruction_dataset_rows": entry_exit_reconstruction.get("dataset_rows"),
        "entry_exit_reconstruction_observed_trade_count": entry_exit_reconstruction.get("observed_trade_count"),
        "entry_exit_state_reward_decision": entry_exit_state_reward_decision,
        "entry_exit_state_reward_ready": entry_exit_state_reward_ready,
        "entry_exit_state_reward_dataset_rows": entry_exit_state_reward.get("dataset_rows"),
        "entry_exit_state_reward_episode_count": entry_exit_state_reward.get("episode_count"),
        "entry_exit_split_leakage_decision": entry_exit_split_leakage_decision,
        "entry_exit_split_leakage_ready": entry_exit_split_leakage_ready,
        "entry_exit_split_leakage_dataset_rows": entry_exit_split_leakage.get("dataset_rows"),
        "entry_exit_split_leakage_episode_count": entry_exit_split_leakage.get("episode_count"),
        "entry_exit_model_dataset_decision": entry_exit_model_dataset_decision,
        "entry_exit_model_dataset_ready": entry_exit_model_dataset_ready,
        "entry_exit_model_dataset_rows": entry_exit_model_dataset.get("dataset_rows"),
        "entry_exit_model_dataset_episode_count": entry_exit_model_dataset.get("episode_count"),
        "entry_exit_feature_alignment_decision": entry_exit_feature_alignment_decision,
        "entry_exit_feature_alignment_ready": entry_exit_feature_alignment_ready,
        "entry_exit_feature_alignment_missing_families": (entry_exit_feature_alignment.get("family_review") or {}).get("missing_families") if isinstance(entry_exit_feature_alignment.get("family_review"), dict) else None,
        "entry_exit_transformer_architecture_decision": entry_exit_transformer_architecture_decision,
        "entry_exit_transformer_architecture_ready": entry_exit_transformer_architecture_ready,
        "entry_exit_transformer_architecture_dataset_rows": entry_exit_transformer_architecture.get("dataset_rows"),
        "entry_exit_transformer_architecture_episode_count": entry_exit_transformer_architecture.get("episode_count"),
        "entry_exit_transformer_training_plan_decision": entry_exit_transformer_training_plan_decision,
        "entry_exit_transformer_training_plan_ready": entry_exit_transformer_training_plan_ready,
        "entry_exit_transformer_training_plan_dataset_rows": entry_exit_transformer_training_plan.get("dataset_rows"),
        "entry_exit_transformer_training_plan_episode_count": entry_exit_transformer_training_plan.get("episode_count"),
        "entry_exit_transformer_trainer_wrapper_decision": entry_exit_transformer_trainer_wrapper_decision,
        "entry_exit_transformer_trainer_wrapper_ready": entry_exit_transformer_trainer_wrapper_ready,
        "entry_exit_transformer_pretrain_manifest_decision": entry_exit_transformer_pretrain_manifest_decision,
        "entry_exit_transformer_pretrain_manifest_ready": entry_exit_transformer_pretrain_manifest_ready,
        "entry_exit_model_dataset_slice_robustness_decision": entry_exit_model_dataset_slice_robustness_decision,
        "entry_exit_model_dataset_slice_robustness_ready": entry_exit_model_dataset_slice_robustness_ready,
        "entry_exit_model_dataset_slice_robustness_weak_slice_count": (entry_exit_model_dataset_slice_robustness.get("slice_review") or {}).get("weak_slice_count") if isinstance(entry_exit_model_dataset_slice_robustness.get("slice_review"), dict) else None,
        "entry_exit_model_dataset_slice_robustness_unsupported_slice_count": (entry_exit_model_dataset_slice_robustness.get("slice_review") or {}).get("unsupported_slice_count") if isinstance(entry_exit_model_dataset_slice_robustness.get("slice_review"), dict) else None,
        "entry_exit_transformer_train_execution_review_decision": entry_exit_transformer_train_execution_review_decision,
        "entry_exit_transformer_train_execution_review_ready": entry_exit_transformer_train_execution_review_ready,
        "entry_exit_transformer_post_train_contract_decision": entry_exit_transformer_post_train_contract_decision,
        "entry_exit_transformer_post_train_contract_ready": entry_exit_transformer_post_train_contract_ready,
        "entry_exit_transformer_post_train_contract_exact_heads": (entry_exit_transformer_post_train_contract.get("post_train_audit_contract") or {}).get("exact_output_heads") if isinstance(entry_exit_transformer_post_train_contract.get("post_train_audit_contract"), dict) else None,
        "entry_exit_transformer_train_enablement_decision": entry_exit_transformer_train_enablement_decision,
        "entry_exit_transformer_train_enablement_ready": entry_exit_transformer_train_enablement_ready,
        "entry_exit_transformer_train_enablement_report": summaries.get("entry-exit-transformer-train-enablement", {}).get("path"),
        "entry_exit_transformer_train_enablement_allows_package_only": bool(entry_exit_transformer_train_enablement.get("exit_training_allowed_with_this_package")),
        "entry_exit_transformer_train_allowed_after_vedtak": False,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "promotion_review_allowed": promotion_review_allowed,
        "promotion_shadow_live_allowed": False,
        "current_blockers": current_blockers,
    },
    "side_effects_started": {
        "dataset_rebuild": False,
        "staging": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    },
    "reports": summaries,
    "commands": commands,
    "allowed_now": allowed_now,
    "blocked_now": blocked_now,
    "optional_proof_commands": optional_proof_commands,
    "worktree_hygiene": {
        "decision": hygiene.get("decision"),
        "dirty_count": hygiene.get("dirty_count"),
        "foundation_cleanup_dirty_count": hygiene.get("foundation_cleanup_dirty_count"),
        "review_before_stage_dirty_count": hygiene.get("review_before_stage_dirty_count"),
        "foundation_cleanup_stage_ready": hygiene.get("foundation_cleanup_stage_ready"),
        "critical_gate_path_count": critical_gate_review.get("critical_gate_path_count"),
        "critical_gate_ok_count": critical_gate_review.get("ok_count"),
        "critical_gate_missing_from_repo": critical_gate_review.get("missing_from_repo") or [],
        "critical_gate_dirty_missing_from_stage": critical_gate_review.get("dirty_missing_from_stage") or [],
        "stage_plan_safe": hygiene.get("stage_plan_safe"),
        "post_stage_decision": post_stage.get("decision"),
        "post_stage_cached_count": post_stage.get("cached_count"),
        "clean_git_resolution": hygiene.get("clean_git_resolution") or {},
        "foundation_stage_paths_txt": hygiene.get("foundation_stage_paths_txt"),
        "review_hold_paths_txt": hygiene.get("review_hold_paths_txt"),
        "canonical_stage_dry_run": "scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run",
        "canonical_stage_apply": "scripts/entry_next_edge_control.sh stage-foundation-cleanup --apply --vedtak <id>",
        "raw_stage_command": " ".join(str(part) for part in stage_cmd) if stage_cmd else "",
    },
}

if json_mode:
    print(json.dumps(payload, indent=2, sort_keys=True))
else:
    for name, path in paths.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        summary = summaries[name]
        report = reports[name]
        print(
            f"{name}: {summary['decision']} "
            f"failures={summary['failures_count']} "
            f"execution_blockers={summary['execution_blockers_count']}"
        )
        if name == "train-readiness" and report.get("foundation_contract_ready_for_smoke"):
            print("  optional proof: scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>")
        if name == "train-readiness":
            print(f"  foundation activation required: {report.get('foundation_activation_required_before_smoke')}")
            print(f"  foundation activation apply required: {report.get('foundation_activation_apply_required_before_smoke')}")
            print(f"  foundation post-apply required: {report.get('foundation_activation_post_apply_required_before_smoke')}")
        if name == "worktree-hygiene":
            wh = payload["worktree_hygiene"]
            print(
                "  dirty/stage/hold: "
                f"{wh['dirty_count']}/"
                f"{wh['foundation_cleanup_dirty_count']}/"
                f"{wh['review_before_stage_dirty_count']}"
            )
            print(
                "  stage-ready/safe: "
                f"{wh['foundation_cleanup_stage_ready']}/"
                f"{wh['stage_plan_safe']}"
            )
            print(
                "  critical gate paths ok: "
                f"{wh['critical_gate_ok_count']}/"
                f"{wh['critical_gate_path_count']} "
                f"missing={len(wh['critical_gate_missing_from_repo'])} "
                f"dirty-missing-stage={len(wh['critical_gate_dirty_missing_from_stage'])}"
            )
            print(f"  post-stage: {wh['post_stage_decision']} cached={wh['post_stage_cached_count']}")
            print(f"  clean-git resolution: {wh['clean_git_resolution'].get('decision')}")
            print(f"  stage paths: {wh['foundation_stage_paths_txt']}")
            print(f"  hold paths: {wh['review_hold_paths_txt']}")
            print(f"  canonical stage dry-run: {wh['canonical_stage_dry_run']}")
            print(f"  canonical stage apply: {wh['canonical_stage_apply']}")
            if wh["raw_stage_command"]:
                print(f"  raw stage command: {wh['raw_stage_command']}")
        if summary["next"]:
            print(f"  next: {summary['next']}")
    print("allowed now:")
    for cmd in allowed_now:
        print(f"  {cmd}")
    if optional_proof_commands:
        print("optional proof commands:")
        for cmd in optional_proof_commands:
            print(f"  {cmd}")
    print("blocked now:")
    for cmd in blocked_now:
        print(f"  {cmd}")
    print("report-only: no training, replay, IQL distillation, staging, shadow, or live path was started")
PY
    ;;

  verify)
    VERIFY_ERR=$(mktemp)
    if "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 "$@" 2>"$VERIFY_ERR"; then
      rm -f "$VERIFY_ERR"
      exit 0
    else
      VERIFY_RC=$?
    fi
    echo "FATAL: foundation verify failed; Entry foundation state is not ready." >&2
    LAST_VERIFY_ERROR=$(grep -E "RuntimeError:|Error:" "$VERIFY_ERR" | tail -n 1 || true)
    if [[ -n "${LAST_VERIFY_ERROR:-}" ]]; then
      echo "foundation verify error: $LAST_VERIFY_ERROR" >&2
    else
      FIRST_VERIFY_ERROR=$(head -n 1 "$VERIFY_ERR" || true)
      if [[ -n "${FIRST_VERIFY_ERROR:-}" ]]; then
        echo "foundation verify error: $FIRST_VERIFY_ERROR" >&2
      fi
    fi
    rm -f "$VERIFY_ERR"
    exit "$VERIFY_RC"
    ;;

  selftest)
    exec "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 --selftest "$@"
    ;;

  foundation-guardrails)
    exec "$PY" -m gx1.scripts.verify_entry_foundation_guardrails_v1 "$@"
    ;;

  foundation-adoption-candidate)
    exec "$PY" -m gx1.scripts.verify_entry_foundation_adoption_candidate_v1 "$@"
    ;;

  foundation-activation-plan)
    exec "$PY" -m gx1.scripts.plan_entry_foundation_activation_v1 "$@"
    ;;

  foundation-activation-apply)
    exec "$PY" -m gx1.scripts.apply_entry_foundation_activation_v1 "$@"
    ;;

  foundation-activation-post-apply)
    exec "$PY" -m gx1.scripts.run_entry_foundation_activation_post_apply_v1 "$@"
    ;;

  worktree-hygiene)
    exec "$PY" -m gx1.scripts.audit_entry_foundation_worktree_hygiene_v1 "$@"
    ;;

  stage-foundation-cleanup)
    exec "$REPO/scripts/stage_entry_foundation_cleanup.sh" "$@"
    ;;

  materialize-smoke)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --vedtak MATERIALIZE_ONLY --materialize-only "$@"
    ;;

  materialize-smoke-seq215)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 --vedtak MATERIALIZE_ONLY --materialize-only "$@"
    ;;

  smoke-manifest)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --manifest-only "$@"
    ;;

  smoke-manifest-seq215)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 --manifest-only "$@"
    ;;

  train-readiness)
    exec "$PY" -m gx1.scripts.verify_entry_training_readiness_v1 "$@"
    ;;

  candidate-readiness)
    exec "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 "$@"
    ;;

  candidate-readiness-seq215)
    exec "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 --challenger-seq215 "$@"
    ;;

  candidate-readiness-smart)
    exec "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 --smart-seq520 "$@"
    ;;

  replay-readiness)
    exec "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 "$@"
    ;;

  replay-readiness-seq215)
    exec "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 --challenger-seq215 "$@"
    ;;

  replay-readiness-smart)
    exec "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 --smart-seq520 "$@"
    ;;

  chart-geometry-audit)
    exec "$PY" -m gx1.scripts.audit_entry_chart_geometry_challenger_v1 "$@"
    ;;

  feature-ai-inventory)
    exec "$PY" -m gx1.scripts.materialize_entry_feature_ai_inventory_v1 "$@"
    ;;

  candlestick-audit)
    exec "$PY" -m gx1.scripts.audit_entry_candlestick_pattern_challenger_v1 "$@"
    ;;

  challenger-extension-manifest)
    exec "$PY" -m gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 "$@"
    ;;

  challenger-smart-extension-manifest)
    exec "$PY" -m gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 --include-smart-layers "$@"
    ;;

  smart-rebuild-preflight)
    exec "$PY" -m gx1.scripts.materialize_entry_smart_seq520_rebuild_preflight_v1 --verify-large-input-hashes "$@"
    ;;

  smart-post-rebuild-readiness)
    exec "$PY" -m gx1.scripts.audit_entry_smart_dataset_post_rebuild_readiness_v1 "$@"
    ;;

  smart-post-rebuild-refresh)
    APPLY=0
    VEDTAK_ID=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --apply)
          APPLY=1
          shift
          ;;
        --vedtak)
          VEDTAK_ID="${2:-}"
          shift 2
          ;;
        -h|--help)
          echo "Usage: scripts/entry_next_edge_control.sh smart-post-rebuild-refresh --apply --vedtak <id>"
          exit 0
          ;;
        *)
          echo "FATAL: unknown smart-post-rebuild-refresh arg: $1" >&2
          exit 2
          ;;
      esac
    done
    if [[ "$APPLY" != "1" ]]; then
      echo "FATAL: smart-post-rebuild-refresh requires --apply" >&2
      exit 2
    fi
    if [[ -z "$VEDTAK_ID" || "$VEDTAK_ID" == *"<"* || "$VEDTAK_ID" == *">"* ]]; then
      echo "FATAL: smart-post-rebuild-refresh requires explicit --vedtak <id>" >&2
      exit 2
    fi
    "$PY" - <<'PY'
import json
import sys
from pathlib import Path

path = Path("/home/andre2/GX1_DATA/reports/entry_smart_dataset_post_rebuild_readiness_20260630_v1/ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json")
if not path.exists():
    print(f"FATAL: missing smart post-rebuild readiness report: {path}", file=sys.stderr)
    raise SystemExit(2)
report = json.loads(path.read_text(encoding="utf-8"))
decision = str(report.get("decision") or "")
if decision != "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW":
    print(f"FATAL: smart-post-rebuild-refresh blocked by decision={decision}", file=sys.stderr)
    raise SystemExit(2)
PY
    exec "$REPO/scripts/gx1_capped_run.sh" --mem 8G --swap 1G -- "$PY" -m gx1.scripts.materialize_entry_foundation_smoke_dataset_v1 \
      --source-dir /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_candidate_20260630 \
      --out-dir /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_seq520_smoke_20260630 \
      --stem v10_smart_seq520_smoke__HOLD_03B \
      --feature-audit-json /home/andre2/GX1_DATA/reports/entry_feature_foundation_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json \
      --target-audit-json /home/andre2/GX1_DATA/reports/entry_target_foundation_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_TARGET_FOUNDATION_AUDIT_latest.json \
      --specialist-audit-json /home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json \
      --schema-version entry_smart_seq520_smoke_dataset_v1 \
      --split-schema-version entry_smart_seq520_smoke_split_manifest_v1 \
      --manifest-variant smart_seq520_candidate \
      --expected-seq-snap-width 520 \
      --batch-size 256 \
      --extreme-snap-feature session_regime.session_trend_structure_liquidity_long_score \
      --extreme-snap-rows 64 \
      --quiet
    ;;

  smart-smoke-manifest)
    exec "$PY" -m gx1.scripts.materialize_entry_smart_seq520_smoke_manifest_v1 "$@"
    ;;

  smart-smoke-readiness)
    exec "$PY" -m gx1.scripts.verify_entry_smart_seq520_smoke_readiness_v1 "$@"
    ;;

  smart-trainability-readiness)
    exec "$PY" -m gx1.scripts.verify_entry_smart_seq520_trainability_readiness_v1 "$@"
    ;;

  smart-ablation-replay-plan)
    exec "$PY" -m gx1.scripts.materialize_entry_smart_ablation_replay_plan_gate_v1 "$@"
    ;;

  smart-ablation-replay-matrix)
    exec "$PY" -m gx1.scripts.verify_entry_smart_ablation_replay_matrix_v1 "$@"
    ;;

  candidate-train)
    exec "$REPO/scripts/run_entry_foundation_seq146_candidate_train.sh" "$@"
    ;;

  candidate-train-seq215)
    exec "$REPO/scripts/run_entry_foundation_seq146_candidate_train.sh" --challenger-seq215 "$@"
    ;;

  candidate-train-smart)
    exec "$REPO/scripts/run_entry_foundation_seq146_candidate_train.sh" --smart-seq520 "$@"
    ;;

  iql-distill)
    exec "$REPO/scripts/run_entry_foundation_iql_distill.sh" "$@"
    ;;

  iql-student-trade-log)
    exec "$PY" -m gx1.scripts.materialize_entry_iql_student_trade_log_v1 "$@"
    ;;

  iql-replay-evidence)
    exec "$PY" -m gx1.scripts.materialize_entry_iql_replay_evidence_v1 "$@"
    ;;

  iql-compare)
    exec "$PY" -m gx1.scripts.verify_entry_iql_replay_comparison_v1 "$@"
    ;;

  iql-slice-audit)
    exec "$PY" -m gx1.scripts.audit_entry_iql_replay_slices_v1 "$@"
    ;;

  entry-exit-handoff)
    exec "$PY" -m gx1.scripts.audit_entry_exit_handoff_readiness_v1 "$@"
    ;;

  entry-exit-materialize)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_per_bar_handoff_v1 "$@"
    ;;

  entry-exit-reconstruction-audit)
    exec "$PY" -m gx1.scripts.audit_entry_exit_per_bar_reconstruction_v1 "$@"
    ;;

  entry-exit-state-reward-contract)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_state_reward_contract_v1 "$@"
    ;;

  entry-exit-split-leakage-audit)
    exec "$PY" -m gx1.scripts.audit_entry_exit_split_leakage_v1 "$@"
    ;;

  entry-exit-model-dataset-readiness)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_model_dataset_readiness_v1 "$@"
    ;;

  entry-exit-feature-alignment)
    exec "$PY" -m gx1.scripts.audit_entry_exit_feature_alignment_v1 "$@"
    ;;

  entry-exit-transformer-architecture-readiness)
    exec "$PY" -m gx1.scripts.audit_entry_exit_transformer_architecture_readiness_v1 "$@"
    ;;

  entry-exit-transformer-training-plan-readiness)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_transformer_training_plan_readiness_v1 "$@"
    ;;

  entry-exit-transformer-trainer-wrapper-readiness)
    exec "$PY" -m gx1.scripts.audit_entry_exit_transformer_trainer_wrapper_readiness_v1 "$@"
    ;;

  entry-exit-transformer-pretrain-manifest)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_transformer_pretrain_manifest_v1 "$@"
    ;;

  entry-exit-model-dataset-slice-robustness)
    exec "$PY" -m gx1.scripts.audit_entry_exit_model_dataset_slice_robustness_v1 "$@"
    ;;

  entry-exit-transformer-train-execution-review)
    exec "$PY" -m gx1.scripts.audit_entry_exit_transformer_train_execution_review_v1 "$@"
    ;;

  entry-exit-transformer-post-train-contract)
    exec "$PY" -m gx1.scripts.audit_entry_exit_transformer_post_train_contract_v1 "$@"
    ;;

  entry-exit-transformer-train-enablement)
    exec "$PY" -m gx1.scripts.materialize_entry_exit_transformer_train_enablement_package_v1 "$@"
    ;;

  entry-exit-transformer-train)
    exec "$REPO/scripts/run_entry_exit_transformer_train.sh" "$@"
    ;;

  smoke-train)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" "$@"
    ;;

  smoke-train-seq215)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 "$@"
    ;;

  smart-smoke-train)
    exec "$REPO/scripts/run_entry_foundation_seq146_smoke_train.sh" --smart-seq520 "$@"
    ;;

  audit-smoke-bundle)
    exec "$PY" -m gx1.scripts.audit_entry_foundation_smoke_bundle_v1 "$@"
    ;;

  selective-edge)
    exec "$PY" -m gx1.scripts.evaluate_entry_candidate_selective_edge_v1 "$@"
    ;;

  candidate-replay-trade-log)
    exec "$PY" -m gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 "$@"
    ;;

  replay-evidence)
    exec "$PY" -m gx1.scripts.materialize_entry_candidate_replay_evidence_v1 "$@"
    ;;

  preview-shadow)
    blocked
    ;;

  start-shadow)
    blocked
    ;;

  verify-shadow)
    blocked
    ;;

  train|retrain|promote|pin|live|start-live|xgb|xgb-train|et|et-train|entry-train|shadow)
    blocked
    ;;

  *)
    echo "FATAL: unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
