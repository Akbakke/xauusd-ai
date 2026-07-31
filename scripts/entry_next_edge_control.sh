#!/usr/bin/env bash
set -euo pipefail

# Single control surface for immutable model-native unified Entry/Exit work.
# Direction launch is exposed only through the exact transactional finalizer.
# Direct promotion, shadow/live start and caller-authored Exit evidence remain
# unavailable here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PY="$REPO/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/entry_next_edge_control.sh COMMAND [explicit arguments]

Model-native seq513 evidence:
  handover [--check|--verbose]
  model-native-state
  model-native-state-selftest
  model-native-native-m1-source --publication-mode bootstrap|successor --vedtak <id> [--start-utc <M1 UTC>] --end-utc <exclusive M1 UTC> --out-root <new-dir> [--parent-root <immutable-dir> --expected-parent-manifest-sha256 <sha256>]
  model-native-native-m5-source --publication-mode bootstrap|successor --vedtak <id> [--start-utc <M5 UTC>] --end-utc <exclusive M5 UTC> --out-root <new-dir> [--parent-root <immutable-dir> --expected-parent-manifest-sha256 <sha256>]
  model-native-canonical-pair --publication-mode bootstrap|successor --native-m1-root <immutable-dir> --native-m5-root <immutable-dir> --vedtak <id> --checkpoint-dir <new-dir> --pair-manifest <json> --generation-root <dir> [--expected-pair-generation-id <sha256> --expected-manifest-sha256 <sha256>] [--workers <n>]
  model-native-live-tail-pair --native-m1-root <immutable-successor-dir> --native-m5-root <immutable-successor-dir> --vedtak <id> --checkpoint-dir <new-dir> --pair-manifest <json> --generation-root <dir> --expected-pair-generation-id <sha256> --expected-manifest-sha256 <sha256> --live-tail-publication-event-root <dir> [--previous-live-tail-publication-json <event> --previous-live-tail-publication-sha256 <sha256>] [--workers <n>]
  model-native-live-tail-admission --pair-manifest <json> --generation-root <dir> --live-tail-admission-event-root <dir> --parent-live-tail-publication-json <event> --parent-live-tail-publication-sha256 <sha256> --child-live-tail-publication-json <event> --child-live-tail-publication-sha256 <sha256>
  model-native-mtf-v4-cache --m5-prebuilt <immutable-parquet> --expected-source-sha256 <sha256> --out-dir <new-event-local-dir>
  model-native-rebuild-preflight \
    --run-id <id> \
    --source-parquet <immutable-parquet> \
    --canonical-v2-parquet <immutable-parquet> \
    --signal-manifest <json> \
    --feature-ranking-json <json> \
    --rank-reference-npz <npz> \
    --mtf-cache-dir <schema-v3-dir> \
    --tape-root <dir> \
    --m1-lifecycle-pair-manifest-json <json> \
    --m1-lifecycle-pair-generation-root <dir> \
    --exit-lifecycle-dir <new-dir> \
    --exit-target-lookahead-m1-steps <n> \
    --early-move-threshold-bps <bps> \
    --output <new-parquet> \
    --audit-out-dir <new-dir> \
    --history-start <UTC> \
    --train-start <UTC> --train-end <UTC> \
    --val-start <UTC> --val-end <UTC> \
    --test-start <UTC> --test-end <UTC> \
    --out-dir <new-dir>
  model-native-post-rebuild-readiness
  model-native-foundation-feature-audit
  model-native-foundation-target-audit
  model-native-specialist-feature-audit
  model-native-adoption-candidate
  model-native-smoke-manifest
  model-native-smoke-readiness
  model-native-trainability-readiness
  model-native-train-recipe-audit
  model-native-smoke-bundle-audit
  model-native-candidate-readiness
  model-native-selective-edge
  model-native-replay-trade-log
  model-native-replay-evidence
  model-native-replay-readiness
  model-native-sizing-capture-instrument
  model-native-sizing-fit-calibration
  model-native-sizing-bind-bundle
  model-native-sizing-materialize-test-oos
  model-native-sizing-finalize-test-proof
  model-native-sizing-produce-unified-joint-proof
  model-native-sizing-adopt
  model-native-sizing-runtime-parity
  model-native-serve-parity
  model-native-direction-pocket-audit
  model-native-adaptation-drift
  model-native-adaptation-shadow
  model-native-adaptation-lifecycle
  model-native-finalize-launch --accepted-bundle-dir <dir> --sizing-adoption-json <event> --joint-exit-proof-json <event> --sizing-runtime-parity-json <event> --serve-parity-json <event> --direction-pocket-json <event> --adaptation-lifecycle-json <event> --live-tail-admission-json <event> --launch-vedtak-json <canonical-immutable-event> --transaction-id <id> --max-trades <n>

Immutable run-lineage execution (evidence gates remain authoritative):
  model-native-rebuild --run-id <id> <all other explicit rebuild arguments> --early-move-threshold-bps <bps>
  model-native-smoke-train --run-id <id> <all other explicit arguments> (--dry-run|--execute)
  model-native-candidate-train --run-id <id> <all other explicit arguments> (--dry-run|--execute)

Every evidence input and output directory must be explicit. Mutable mirrors,
soft failure flags, feature-mask ablations, alternate contract modes, and
secondary direction paths are rejected. Entry launch is available only through
the complete transactional finalizer; direct promotion, shadow and live start
remain unavailable. Unified Exit evidence is admitted only through the
same-candidate, full-TEST producer route above.
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
    if [[ $# -gt 1 ]]; then
      die "handover accepts at most one of --check or --verbose"
    fi
    case "${1:-}" in
      ""|--check|--verbose) ;;
      *) die "handover accepts only --check or --verbose" ;;
    esac
    exec "$REPO/scripts/gx1_handover.sh" "$@"
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

  model-native-native-m1-source|model-native-native-m5-source)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --timeframe \
      --instrument \
      --granularity \
      --chunk-days \
      --repair-mode \
      --raw-in \
      --raw-out \
      --dukascopy-enabled \
      --dukascopy-disabled
    for flag in \
      --publication-mode \
      --vedtak \
      --end-utc \
      --out-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    publication_mode=""
    previous=""
    for arg in "$@"; do
      if [[ "$previous" == "--publication-mode" ]]; then
        publication_mode="$arg"
        previous=""
      elif [[ "$arg" == --publication-mode=* ]]; then
        publication_mode="${arg#--publication-mode=}"
      else
        previous="$arg"
      fi
    done
    case "$publication_mode" in
      bootstrap)
        require_flag "$cmd" --start-utc "$@"
        ;;
      successor)
        require_flag "$cmd" --parent-root "$@"
        require_flag "$cmd" --expected-parent-manifest-sha256 "$@"
        ;;
      *)
        die "$cmd requires --publication-mode bootstrap or successor"
        ;;
    esac
    if [[ "$cmd" == "model-native-native-m1-source" ]]; then
      exec "$PY" -m gx1.scripts.backfill_xauusd_m5_from_oanda \
        --timeframe M1 "$@"
    fi
    exec "$PY" -m gx1.scripts.backfill_xauusd_m5_from_oanda \
      --timeframe M5 "$@"
    ;;

  model-native-canonical-pair|model-native-live-tail-pair)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --loop \
      --canonical-parquet \
      --base28-parquet \
      --raw-m1-parquet \
      --raw-m5-parquet
    if [[ "$cmd" == "model-native-live-tail-pair" ]]; then
      reject_flags "$cmd" --publication-mode
    else
      reject_flags "$cmd" \
        --live-tail-publication-event-root \
        --previous-live-tail-publication-json \
        --previous-live-tail-publication-sha256
    fi
    for flag in \
      --publication-mode \
      --native-m1-root \
      --native-m5-root \
      --vedtak \
      --checkpoint-dir \
      --pair-manifest \
      --generation-root; do
      if [[ "$cmd" == "model-native-live-tail-pair" && "$flag" == "--publication-mode" ]]; then
        continue
      fi
      require_flag "$cmd" "$flag" "$@"
    done
    if [[ "$cmd" == "model-native-live-tail-pair" ]]; then
      for flag in \
        --expected-pair-generation-id \
        --expected-manifest-sha256 \
        --live-tail-publication-event-root; do
        require_flag "$cmd" "$flag" "$@"
      done
      exec "$REPO/scripts/gx1_capped_run.sh" --mem 30G --swap 2G -- \
        "$PY" -m gx1.execution.v12_canonical_incremental \
        --publication-mode successor "$@"
    fi
    publication_mode=""
    for ((index = 0; index < ${#COMMAND_ARGS[@]}; index++)); do
      arg="${COMMAND_ARGS[$index]}"
      if [[ "$arg" == "--publication-mode" ]]; then
        ((index + 1 < ${#COMMAND_ARGS[@]})) || die "$cmd requires a publication mode value"
        publication_mode="${COMMAND_ARGS[$((index + 1))]}"
      elif [[ "$arg" == --publication-mode=* ]]; then
        publication_mode="${arg#--publication-mode=}"
      fi
    done
    case "$publication_mode" in
      bootstrap)
        reject_flags "$cmd bootstrap" \
          --expected-pair-generation-id \
          --expected-manifest-sha256
        ;;
      successor)
        require_flag "$cmd successor" --expected-pair-generation-id "$@"
        require_flag "$cmd successor" --expected-manifest-sha256 "$@"
        ;;
      *)
        die "$cmd requires --publication-mode bootstrap or successor"
        ;;
    esac
    exec "$REPO/scripts/gx1_capped_run.sh" --mem 30G --swap 2G -- \
      "$PY" -m gx1.execution.v12_canonical_incremental "$@"
    ;;

  model-native-live-tail-admission)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --publication-mode \
      --native-m1-root \
      --native-m5-root \
      --vedtak \
      --checkpoint-dir \
      --expected-pair-generation-id \
      --expected-manifest-sha256 \
      --live-tail-publication-event-root \
      --previous-live-tail-publication-json \
      --previous-live-tail-publication-sha256 \
      --workers
    for flag in \
      --pair-manifest \
      --generation-root \
      --live-tail-admission-event-root \
      --parent-live-tail-publication-json \
      --parent-live-tail-publication-sha256 \
      --child-live-tail-publication-json \
      --child-live-tail-publication-sha256; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.execution.v12_canonical_incremental \
      --publication-mode live-tail-admission "$@"
    ;;

  model-native-mtf-v4-cache)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --contract
    for flag in \
      --m5-prebuilt \
      --expected-source-sha256 \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/gx1_capped_run.sh" --mem 30G --swap 2G -- \
      "$PY" -m gx1.scripts.prebuild_multi_tf_cache_v2 \
      --contract v4 "$@"
    ;;

  model-native-rebuild-preflight)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --source-parquet \
      --canonical-v2-parquet \
      --signal-manifest \
      --feature-ranking-json \
      --rank-reference-npz \
      --mtf-cache-dir \
      --tape-root \
      --m1-lifecycle-pair-manifest-json \
      --m1-lifecycle-pair-generation-root \
      --exit-lifecycle-dir \
      --exit-target-lookahead-m1-steps \
      --early-move-threshold-bps \
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
    for flag in \
      --run-id \
      --source-parquet \
      --canonical-v2-parquet \
      --signal-manifest \
      --feature-ranking-json \
      --rank-reference-npz \
      --mtf-cache-dir \
      --tape-root \
      --m1-lifecycle-pair-manifest-json \
      --m1-lifecycle-pair-generation-root \
      --exit-lifecycle-dir \
      --exit-target-lookahead-m1-steps \
      --early-move-threshold-bps \
      --output \
      --audit-out-dir \
      --history-start \
      --train-start \
      --train-end \
      --val-start \
      --val-end \
      --test-start \
      --test-end \
      --existing-rank-reference; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/rebuild_entry_model_native_seq513_dataset.sh" "$@"
    ;;

  model-native-post-rebuild-readiness)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --event-root \
      --repo-dir \
      --chain-terminal-json \
      --rebuild-preflight-json \
      --full-input-liveness-json \
      --pretrain-audit-json \
      --dataset-dir \
      --smoke-dataset-dir \
      --train-manifest-json \
      --train-manifest-sha256 \
      --train-parquet \
      --train-parquet-sha256 \
      --val-manifest-json \
      --val-manifest-sha256 \
      --val-parquet \
      --val-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --test-parquet \
      --test-parquet-sha256 \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_post_rebuild_readiness_v1 "$@"
    ;;

  model-native-foundation-feature-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --train-manifest-json \
      --train-manifest-sha256 \
      --train-parquet-sha256 \
      --val-manifest-json \
      --val-manifest-sha256 \
      --val-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --test-parquet-sha256 \
      --seq-structure-manifest \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.audit_entry_foundation_features_v1 "$@"
    ;;

  model-native-foundation-target-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --train-manifest-json \
      --train-manifest-sha256 \
      --train-parquet-sha256 \
      --val-manifest-json \
      --val-manifest-sha256 \
      --val-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --test-parquet-sha256 \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.audit_entry_foundation_targets_v1 "$@"
    ;;

  model-native-specialist-feature-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --train-manifest-json \
      --train-manifest-sha256 \
      --train-parquet-sha256 \
      --val-manifest-json \
      --val-manifest-sha256 \
      --val-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --test-parquet-sha256 \
      --seq-structure-manifest \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.audit_entry_specialist_feature_groups_v1 "$@"
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
      --run-id \
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

  model-native-train-recipe-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --profile \
      --repo \
      --wrapper-path \
      --run-id \
      --dataset-dir \
      --out-bundle-dir \
      --device \
      --seed \
      --epochs \
      --batch-size \
      --learning-rate \
      --early-stop-patience \
      --early-stop-min-delta \
      --grad-clip-norm \
      --weight-decay \
      --multi-tf-scale \
      --specialist-fusion-scale \
      --subsample-rows \
      --memory-cap \
      --swap-cap \
      --gx1-data-root \
      --train-manifest-json \
      --val-manifest-json \
      --test-manifest-json \
      --train-parquet \
      --val-parquet \
      --test-parquet \
      --unified-exit-lifecycle-manifest-json \
      --m5-prebuilt-path \
      --multi-tf-cache-manifest-json \
      --post-rebuild-readiness-json \
      --full-input-liveness-audit-json \
      --feature-audit-json \
      --target-audit-json \
      --specialist-audit-json \
      --pretrain-audit-json \
      --trainability-readiness-json \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_train_recipe_audit_v1 "$@"
    ;;

  model-native-smoke-train)
    reject_non_authoritative_args "$@"
    exec "$REPO/scripts/run_entry_model_native_seq513_smoke_train.sh" "$@"
    ;;

  model-native-smoke-bundle-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --bundle-dir \
      --dataset-dir \
      --val-manifest-json \
      --test-manifest-json \
      --predictions-parquet \
      --prediction-report-json \
      --target-audit-json \
      --specialist-audit-json \
      --pretrain-audit-json \
      --out-dir \
      --device; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.audit_entry_foundation_smoke_bundle_v1 "$@"
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
      --val-manifest-json \
      --val-manifest-sha256 \
      --val-parquet \
      --val-parquet-sha256 \
      --test-manifest-json \
      --test-manifest-sha256 \
      --test-parquet \
      --test-parquet-sha256 \
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

  model-native-sizing-capture-instrument)
    reject_non_authoritative_args "$@"
    require_flag "$cmd" --authority-root "$@"
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      capture-instrument "$@"
    ;;

  model-native-sizing-fit-calibration)
    reject_non_authoritative_args "$@"
    for flag in \
      --predictions \
      --prediction-report \
      --bundle-dir \
      --dataset-dir \
      --dataset-manifest \
      --instrument-evidence \
      --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      fit-calibration "$@"
    ;;

  model-native-sizing-bind-bundle)
    reject_non_authoritative_args "$@"
    for flag in --source-bundle-dir --out-bundle-dir --calibration; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      bind-bundle "$@"
    ;;

  model-native-sizing-materialize-test-oos)
    reject_non_authoritative_args "$@"
    for flag in \
      --calibration \
      --test-predictions \
      --test-prediction-report \
      --bundle-dir \
      --dataset-dir \
      --source-tape \
      --model-head-serve-parity \
      --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      materialize-test-oos "$@"
    ;;

  model-native-sizing-finalize-test-proof)
    reject_non_authoritative_args "$@"
    for flag in --calibration --oos-source --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      finalize-test-proof "$@"
    ;;

  model-native-sizing-produce-unified-joint-proof)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --artifact-registry \
      --replay-rows \
      --exit-trace-rows
    for flag in \
      --calibration \
      --proof \
      --source-tape \
      --prebuilt-pair-manifest \
      --prebuilt-generation-root \
      --train-rank-reference-npz \
      --train-rank-reference-sha256 \
      --authority-root \
      --device; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/gx1_capped_run.sh" --mem 30G --swap 2G -- \
      "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      produce-unified-joint-exit-proof "$@"
    ;;

  model-native-sizing-adopt)
    reject_non_authoritative_args "$@"
    for flag in \
      --bundle-dir \
      --calibration \
      --proof \
      --joint-exit-proof \
      --authority-root \
      --run-id; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      adopt "$@"
    ;;

  model-native-sizing-runtime-parity)
    reject_non_authoritative_args "$@"
    for flag in --adoption --observations --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      finalize-runtime-parity "$@"
    ;;

  model-native-serve-parity)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --pair-manifest-path \
      --pair-generation-root \
      --pinned-predictions \
      --prediction-report-json \
      --bundle-dir \
      --max-trades \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.verify_model_native_serve_parity_v1 "$@"
    ;;

  model-native-direction-pocket-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --dataset-parquet \
      --predictions-parquet \
      --prediction-report-json \
      --bundle-dir \
      --bundle-metadata-json \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.audit_model_native_direction_pockets_v1 "$@"
    ;;

  model-native-adaptation-drift)
    reject_non_authoritative_args "$@"
    for flag in \
      --bundle-dir \
      --reference-rows \
      --observation-rows \
      --output-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_adaptation_drift_v1 "$@"
    ;;

  model-native-adaptation-shadow)
    reject_non_authoritative_args "$@"
    for flag in \
      --incumbent-bundle-dir \
      --candidate-bundle-dir \
      --paired-rows \
      --output-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_adaptation_shadow_v1 "$@"
    ;;

  model-native-adaptation-lifecycle)
    reject_non_authoritative_args "$@"
    for flag in \
      --transition \
      --incumbent-bundle-dir \
      --drift-evidence \
      --replay-readiness \
      --output-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_adaptation_lifecycle_v1 "$@"
    ;;

  model-native-finalize-launch)
    reject_non_authoritative_args "$@"
    for flag in \
      --accepted-bundle-dir \
      --sizing-adoption-json \
      --joint-exit-proof-json \
      --sizing-runtime-parity-json \
      --serve-parity-json \
      --direction-pocket-json \
      --adaptation-lifecycle-json \
      --live-tail-admission-json \
      --launch-vedtak-json \
      --transaction-id \
      --max-trades; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$PY" -m gx1.scripts.finalize_entry_model_native_launch_v1 "$@"
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
