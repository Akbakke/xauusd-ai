#!/usr/bin/env bash
set -euo pipefail

# Single control surface for immutable model-native Entry seq513 work.
# Direction launch, promotion, shadow, live trading, and mutable Exit evidence
# remain unavailable here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PY="$REPO/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/entry_next_edge_control.sh COMMAND [explicit arguments]

Model-native seq513 evidence:
  handover
  model-native-state
  model-native-state-selftest
  model-native-abstention-probe
  model-native-rebuild-preflight
  model-native-adoption-candidate
  model-native-smoke-manifest
  model-native-smoke-readiness
  model-native-trainability-readiness
  model-native-candidate-readiness
  model-native-selective-edge
  model-native-replay-trade-log
  model-native-replay-evidence
  model-native-replay-readiness

Explicit, vedtak-gated execution:
  model-native-rebuild --vedtak <id> <all other explicit rebuild arguments>
  model-native-smoke-train --vedtak <id> <all other explicit arguments> (--dry-run|--execute)
  model-native-candidate-train --vedtak <id> <all other explicit arguments> (--dry-run|--execute)

Every evidence input and output directory must be explicit. Mutable mirrors,
soft failure flags, feature-mask ablations, alternate contract modes, and
secondary direction paths are rejected. Entry direction launch, promotion,
shadow, and live trading are not exposed. Exit evidence remains unavailable
until its producers publish immutable, explicitly bound events.
EOF
}

die() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 2
}

require_flag() {
  local route="$1"
  local required="$2"
  shift 2
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$required" || "$arg" == "$required="* ]]; then
      return 0
    fi
  done
  die "$route requires explicit $required"
}

reject_non_authoritative_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      *"_latest.json"*|*"_latest.md"*|*"/latest/"*)
        die "mutable latest input is forbidden: $arg"
        ;;
      --no-fail-on-*|--no-require-*|--allow-non-2026)
        die "soft pass-through is forbidden: $arg"
        ;;
      --feature-mask-json|--feature-mask-json=*|--contract-mode|--contract-mode=*)
        die "alternate or ablation contract input is forbidden: $arg"
        ;;
    esac
  done
}

reject_flags() {
  local route="$1"
  shift
  local forbidden arg
  for forbidden in "$@"; do
    for arg in "${COMMAND_ARGS[@]}"; do
      if [[ "$arg" == "$forbidden" || "$arg" == "$forbidden="* ]]; then
        die "$route fixes $forbidden in the exact evidence contract"
      fi
    done
  done
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
shift
COMMAND_ARGS=("$@")

[[ -x "$PY" ]] || die "repository Python is not executable: $PY"
cd "$REPO"

case "$cmd" in
  handover)
    [[ $# -eq 0 ]] || die "handover accepts no arguments"
    exec "$REPO/scripts/gx1_handover.sh"
    ;;

  model-native-state)
    reject_non_authoritative_args "$@"
    for flag in \
      --rebuild-preflight-json \
      --smoke-manifest-json \
      --smoke-readiness-json \
      --trainability-readiness-json \
      --candidate-readiness-json \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 "$@"
    ;;

  model-native-state-selftest)
    reject_non_authoritative_args "$@"
    exec "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 --selftest "$@"
    ;;

  model-native-abstention-probe)
    reject_non_authoritative_args "$@"
    for flag in \
      --train-smoke-manifest \
      --train-smoke-manifest-sha256 \
      --val-smoke-manifest \
      --val-smoke-manifest-sha256 \
      --test-smoke-manifest \
      --test-smoke-manifest-sha256 \
      --artifact-registry-json \
      --artifact-registry-sha256 \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_model_native_abstention_probe_v1 "$@"
    ;;

  model-native-rebuild-preflight)
    reject_non_authoritative_args "$@"
    for flag in \
      --vedtak \
      --source-parquet \
      --canonical-v2-parquet \
      --signal-manifest \
      --feature-ranking-json \
      --rank-reference-npz \
      --mtf-cache-dir \
      --tape-root \
      --output \
      --audit-out-dir \
      --history-start \
      --train-start \
      --train-end \
      --val-start \
      --val-end \
      --test-start \
      --test-end \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_rebuild_preflight_v1 "$@"
    ;;

  model-native-rebuild)
    reject_non_authoritative_args "$@"
    exec "$REPO/scripts/rebuild_entry_model_native_seq513_dataset.sh" "$@"
    ;;

  model-native-adoption-candidate)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --feature-audit-json \
      --target-audit-json \
      --specialist-audit-json \
      --smoke-manifest-json \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_foundation_adoption_candidate_v1 "$@"
    ;;

  model-native-smoke-manifest)
    reject_non_authoritative_args "$@"
    for flag in \
      --smart-smoke-dataset-dir \
      --post-rebuild-readiness-json \
      --smart-specialist-audit-json \
      --train-parquet \
      --train-parquet-sha256 \
      --train-manifest-json \
      --train-manifest-sha256 \
      --val-parquet \
      --val-parquet-sha256 \
      --val-manifest-json \
      --val-manifest-sha256 \
      --test-parquet \
      --test-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --out-dir \
      --vedtak \
      --memory-cap \
      --swap-cap \
      --sample-rows \
      --batch-size; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_smoke_manifest_v1 "$@"
    ;;

  model-native-smoke-readiness)
    reject_non_authoritative_args "$@"
    for flag in \
      --smart-post-rebuild-readiness-json \
      --full-input-liveness-json \
      --model-native-rebuild-preflight-json \
      --smart-dataset-dir \
      --smart-smoke-dataset-dir \
      --smart-feature-audit-json \
      --smart-target-audit-json \
      --smart-specialist-audit-json \
      --smoke-manifest-event-json \
      --repo-dir \
      --out-dir \
      --memory-cap \
      --swap-cap; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_model_native_seq513_smoke_readiness_v1 "$@"
    ;;

  model-native-trainability-readiness)
    reject_non_authoritative_args "$@"
    for flag in \
      --smart-post-rebuild-readiness-json \
      --smart-smoke-readiness-json \
      --full-input-liveness-json \
      --control-script \
      --trainer-source \
      --smoke-wrapper \
      --candidate-wrapper \
      --candidate-readiness-script \
      --selective-edge-script \
      --replay-evidence-script \
      --replay-readiness-script \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_model_native_seq513_trainability_readiness_v1 "$@"
    ;;

  model-native-smoke-train)
    reject_non_authoritative_args "$@"
    exec "$REPO/scripts/run_entry_model_native_seq513_smoke_train.sh" "$@"
    ;;

  model-native-candidate-readiness)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --edge-test-scope --min-active-specialists
    for flag in \
      --smoke-bundle-audit-json \
      --specialist-audit-json \
      --trainability-readiness-json \
      --expected-smoke-dataset-dir \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 \
      --edge-test-scope strict --min-active-specialists 8 "$@"
    ;;

  model-native-candidate-train)
    reject_non_authoritative_args "$@"
    exec "$REPO/scripts/run_entry_model_native_seq513_candidate_train.sh" "$@"
    ;;

  model-native-selective-edge)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --splits --top-fracs --model-name --selection-score-mode
    for flag in \
      --bundle-dir \
      --dataset-dir \
      --device \
      --batch-size \
      --stream-chunk-rows \
      --m5-prebuilt-path \
      --multi-tf-cache-dir \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.evaluate_entry_candidate_selective_edge_v1 "$@"
    ;;

  model-native-replay-trade-log)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --exit-mode \
      --take-profit-bps \
      --stop-loss-bps \
      --same-bar-policy \
      --mfe-protect-activation-bps \
      --mfe-protect-breakeven-offset-bps \
      --mfe-protect-trailing-capture-ratio \
      --mfe-protect-trailing-floor-bps \
      --cooldown-bars \
      --max-trades-per-day \
      --daily-loss-limit-bps \
      --fail-on-audit-fail
    for flag in \
      --model-native-state-json \
      --candidate-readiness-json \
      --selective-edge-predictions \
      --selective-edge-report-json \
      --dataset-dir \
      --source-parquet \
      --out-dir \
      --model-name \
      --cost-stress-bps \
      --policy-id \
      --slippage-bps; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 "$@"
    ;;

  model-native-replay-evidence)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --ablation-id \
      --require-year \
      --allow-non-2026 \
      --require-model-native-trade-fields \
      --no-require-model-native-trade-fields \
      --require-identity-artifacts \
      --no-require-identity-artifacts
    for flag in \
      --trades-path \
      --trade-log-manifest-json \
      --out-dir \
      --candidate-bundle-audit-json \
      --selective-edge-report-json \
      --policy-id; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_candidate_replay_evidence_v1 "$@"
    ;;

  model-native-replay-readiness)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --model-name \
      --min-top5-mean-pnl-bps \
      --min-top10-mean-pnl-bps \
      --min-top-direction-precision \
      --min-direction-slice-precision \
      --min-direction-slice-n \
      --min-replay-net-sum-bps \
      --min-profit-factor \
      --max-abs-drawdown-bps
    for flag in \
      --candidate-readiness-json \
      --candidate-bundle-audit-json \
      --selective-edge-report-json \
      --selective-edge-metrics-csv \
      --replay-evidence-json \
      --pretrain-audit-json \
      --expected-dataset-dir \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_entry_replay_readiness_v1 "$@"
    ;;

  train|retrain|promote|pin|shadow|live|start-live|start-shadow|preview-shadow|verify-shadow)
    die "$cmd is not exposed by the model-native evidence control surface"
    ;;

  *)
    printf 'FATAL: unknown command: %s\n' "$cmd" >&2
    usage >&2
    exit 2
    ;;
esac
