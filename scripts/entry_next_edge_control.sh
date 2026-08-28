#!/usr/bin/env bash
set -euo pipefail

# Single control surface for immutable model-native unified Entry/Exit work.
# Direction launch is exposed only through the exact transactional finalizer.
# Direct promotion, shadow/live start and caller-authored Exit evidence remain
# unavailable here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PY="$REPO/.venv/bin/python"
AUDIT_CAP=("$REPO/scripts/gx1_capped_run.sh" --class audit --mem 4G --swap 512M --)
# Heavy offline producers over the complete population. The pre-wave control
# surface gave these 10G; collapsing them into the 4G audit cap is what got the
# feature ranker and the M1 enriched lane cgroup-killed.
PRODUCER_CAP=("$REPO/scripts/gx1_capped_run.sh" --class producer --mem 10G --swap 512M --)

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
  model-native-fit-volatility-squeeze-artifacts --m1-source <immutable-parquet> --m1-source-sha256 <sha256> --m5-source <immutable-parquet> --m5-source-sha256 <sha256> --tape-manifest <json> --tape-manifest-sha256 <sha256> --pair-manifest <json> --pair-manifest-sha256 <sha256> --pair-generation-id <sha256> --train-window-start <UTC> --train-window-end <UTC> --output-dir <existing-empty-dir>
  model-native-m1-enriched-frame --native-m1-root <immutable-dir> --pair-manifest <json> --expected-pair-manifest-sha256 <sha256> --multi-tf-cache-dir <immutable-dir> --output-parquet <new-parquet> --manifest-path <new-json> --checkpoint-dir <new-dir> --dataset-run-id <id> --pair-generation-id <sha256> --registry-fit-train-start <UTC> --registry-fit-inner-end <UTC> --registry-fit-train-end <UTC> --registry-fit-tape-manifest <json> --expected-registry-fit-tape-manifest-sha256 <sha256> --volatility-squeeze-manifest <json> --expected-volatility-squeeze-manifest-sha256 <sha256> [--workers 1 --checkpoint-chunk-rows <n>]
  model-native-m5-enriched-frame --native-m5-root <immutable-dir> --pair-manifest <json> --expected-pair-manifest-sha256 <sha256> --multi-tf-cache-dir <new-dir> --output-parquet <new-parquet> --manifest-path <new-json> --checkpoint-dir <new-dir> --dataset-run-id <id> --pair-generation-id <sha256> --registry-fit-train-start <UTC> --registry-fit-inner-end <UTC> --registry-fit-train-end <UTC> --registry-fit-tape-manifest <json> --expected-registry-fit-tape-manifest-sha256 <sha256> --volatility-squeeze-manifest <json> --expected-volatility-squeeze-manifest-sha256 <sha256> [--workers 1 --checkpoint-chunk-rows <n>]
  model-native-m5-source-frame --enriched-parquet <immutable-parquet> --multi-tf-cache-dir <immutable-v4-dir> --native-m5-root <immutable-dir> --pair-manifest <json> --output-parquet <new-parquet> --dataset-run-id <id> --pair-generation-id <sha256>
  model-native-current-source-cascade-proof --run-id <id> --source-parquet <immutable-parquet> --canonical-v2-parquet <immutable-parquet> --mtf-cache-dir <immutable-dir> --pair-manifest <json> --required-history-start <UTC> --out <new-json>
  model-native-m1-feature-base --source-parquet <immutable-parquet> --alignment-parquet <pair-bound-m1-parquet> --seq-structure-manifest <json> --output-parquet <new-parquet> --dataset-run-id <id> --pair-generation-id <sha256> --v29-registry-constants-json <m1-enriched-manifest-json> --volatility-squeeze-manifest <json> --expected-volatility-squeeze-manifest-sha256 <sha256>
  model-native-m5-feature-base --source-parquet <immutable-parquet> --seq-structure-manifest <json> --output-parquet <new-parquet> --dataset-run-id <id> --pair-generation-id <sha256> --v29-registry-constants-json <v4-cache-manifest-json> --volatility-squeeze-manifest <json> --expected-volatility-squeeze-manifest-sha256 <sha256>
  model-native-cross-surface-overlap --run-id <id> --signal-manifest <json> --m1-feature-base-parquet <immutable-parquet> --m5-feature-base-parquet <immutable-parquet> --mtf-cache-dir <immutable-dir> --out-json <new-json>
  model-native-feature-surface-liveness --run-id <id> --signal-manifest <json> --m1-feature-base-parquet <immutable-parquet> --m5-feature-base-parquet <immutable-parquet> --cross-surface-overlap-json <immutable-json> --out-json <new-json>
  model-native-rebuild-preflight \
    --run-id <id> \
    --source-parquet <immutable-parquet> \
    --canonical-v2-parquet <immutable-parquet> \
    --signal-manifest <json> \
    --feature-ranking-json <json> \
    --mtf-cache-dir <schema-v3-dir> \
    --tape-root <dir> \
    --m1-lifecycle-pair-manifest-json <json> \
    --m1-lifecycle-pair-generation-root <dir> \
    --m1-feature-base-parquet <immutable-parquet> \
    --m5-feature-base-parquet <immutable-parquet> \
    --exit-lifecycle-dir <new-dir> \
    # Exit target policy is fit on exact TRAIN native M1 and hash-bound. \
    # Entry direction/early-move policy is fit on exact TRAIN native M5. \
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
  model-native-execution-causality-audit --dataset-dir <dataset-dir> --signal-manifest <json> --train-manifest <json> --val-manifest <json> --train-lifecycle-manifest <json> --val-lifecycle-manifest <json> --output <new-json>
  model-native-sequence-roll-audit --parquet <split-parquet> --manifest-json <split-manifest> --out-json <new-json>
  model-native-sequence-integrity-audit --parquet <split-parquet> --manifest-json <split-manifest> --out-json <new-json>
  model-native-sequence-source-reconstruction-audit --parquet <split-parquet> --manifest-json <split-manifest> --out-json <new-json>
  model-native-train-recipe-audit
  model-native-smoke-bundle-audit
  model-native-candidate-readiness
  model-native-selective-edge
  model-native-seed-stability
  model-native-sizing-fit-calibration
  model-native-sizing-bind-bundle
  model-native-sizing-materialize-test-oos
  model-native-sizing-finalize-test-proof
  model-native-trade-path-metrics
  model-native-serve-parity
  model-native-direction-pocket-audit

Immutable run-lineage execution (evidence gates remain authoritative):
  model-native-smoke-train --run-id <id> <all other explicit arguments> \
    --train-sequence-source-audit-json <immutable-json> \
    --val-sequence-source-audit-json <immutable-json> (--dry-run|--execute)
  model-native-attended-smoke-train --run-id <id> <all other explicit arguments> \
    --train-sequence-source-audit-json <immutable-json> \
    --val-sequence-source-audit-json <immutable-json> (--dry-run|--execute)
  model-native-attended-cpu-smoke-train --run-id <id> <all other explicit arguments> \
    --train-sequence-source-audit-json <immutable-json> \
    --val-sequence-source-audit-json <immutable-json> (--dry-run|--execute)
  model-native-attended-hardware-smoke --specialist-audit-json <immutable-json> (--dry-run|--execute)
  model-native-candidate-train --run-id <id> <all other explicit arguments> \
    --train-sequence-source-audit-json <immutable-json> \
    --val-sequence-source-audit-json <immutable-json> (--dry-run|--execute)

Every evidence input and output directory must be explicit. Mutable mirrors,
soft failure flags, feature-mask ablations, alternate contract modes, and
secondary direction paths are rejected. Entry/Exit launch, promotion, shadow
and live operation are outside this checkout. The retired fixed-proxy
joint-Exit route is not exposed. A future production economic contract needs
immutable broker costs, financing and shared-portfolio replay before it can
become an authority.
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
  local arg count=0
  for arg in "$@"; do
    if [[ "$arg" == "$required" || "$arg" == "$required="* ]]; then
      count=$((count + 1))
    fi
  done
  [[ $count -eq 1 ]] \
    || die "$route requires exactly one explicit $required (observed=$count)"
}

exact_flag_value() {
  local route="$1"
  local required="$2"
  shift 2
  local arg value= count=0
  local -a values=()
  while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ "$arg" == "$required" ]]; then
      [[ $# -ge 2 ]] || die "$route $required requires a value"
      values+=("$2")
      count=$((count + 1))
      shift 2
      continue
    fi
    if [[ "$arg" == "$required="* ]]; then
      values+=("${arg#"$required="}")
      count=$((count + 1))
    fi
    shift
  done
  [[ $count -eq 1 && -n "${values[0]:-}" ]] \
    || die "$route requires one non-empty $required value"
  printf '%s\n' "${values[0]}"
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
      --live-tail-*|--previous-live-tail-*|--parent-live-tail-*|--child-live-tail-*)
        die "GX1_OFFLINE_SCOPE_FORBIDDEN: live-tail inputs are disabled"
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

case "$cmd" in
  model-native-live-tail-*|model-native-finalize-launch|model-native-adaptation-*)
    die "GX1_OFFLINE_SCOPE_FORBIDDEN: live, launch and drift/adaptation routes are disabled"
    ;;
esac

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
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 "$@"
    ;;

  model-native-state-selftest)
    reject_non_authoritative_args "$@"
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 --selftest "$@"
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
      exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.backfill_xauusd_m5_from_oanda \
        --timeframe M1 "$@"
    fi
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.backfill_xauusd_m5_from_oanda \
      --timeframe M5 "$@"
    ;;

  model-native-canonical-pair)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" \
      --loop \
      --canonical-parquet \
      --base28-parquet \
      --raw-m1-parquet \
      --raw-m5-parquet
    for flag in \
      --publication-mode \
      --native-m1-root \
      --native-m5-root \
      --vedtak \
      --checkpoint-dir \
      --pair-manifest \
      --generation-root; do
      require_flag "$cmd" "$flag" "$@"
    done
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
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.execution.v12_canonical_incremental "$@"
    ;;

  model-native-fit-volatility-squeeze-artifacts)
    reject_non_authoritative_args "$@"
    for flag in \
      --m1-source \
      --m1-source-sha256 \
      --m5-source \
      --m5-source-sha256 \
      --tape-manifest \
      --tape-manifest-sha256 \
      --pair-manifest \
      --pair-manifest-sha256 \
      --pair-generation-id \
      --train-window-start \
      --train-window-end \
      --output-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.fit_volatility_squeeze_artifacts_v1 "$@"
    ;;

  model-native-m1-enriched-frame)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --timeframe --native-root --native-m5-root
    for flag in \
      --native-m1-root \
      --pair-manifest \
      --expected-pair-manifest-sha256 \
      --multi-tf-cache-dir \
      --output-parquet \
      --manifest-path \
      --checkpoint-dir \
      --dataset-run-id \
      --pair-generation-id \
      --registry-fit-train-start \
      --registry-fit-inner-end \
      --registry-fit-train-end \
      --registry-fit-tape-manifest \
      --expected-registry-fit-tape-manifest-sha256 \
      --volatility-squeeze-manifest \
      --expected-volatility-squeeze-manifest-sha256; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.build_entry_exit_m1_enriched_frame_v1 "$@"
    ;;

  model-native-m5-enriched-frame)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --timeframe --native-m1-root --native-root
    for flag in \
      --native-m5-root \
      --pair-manifest \
      --expected-pair-manifest-sha256 \
      --multi-tf-cache-dir \
      --output-parquet \
      --manifest-path \
      --checkpoint-dir \
      --dataset-run-id \
      --pair-generation-id \
      --registry-fit-train-start \
      --registry-fit-inner-end \
      --registry-fit-train-end \
      --registry-fit-tape-manifest \
      --expected-registry-fit-tape-manifest-sha256 \
      --volatility-squeeze-manifest \
      --expected-volatility-squeeze-manifest-sha256; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.build_entry_exit_m1_enriched_frame_v1 "$@"
    ;;

  model-native-m5-source-frame)
    reject_non_authoritative_args "$@"
    for flag in \
      --enriched-parquet \
      --multi-tf-cache-dir \
      --native-m5-root \
      --pair-manifest \
      --output-parquet \
      --dataset-run-id \
      --pair-generation-id; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.materialize_entry_model_native_m5_source_v1 "$@"
    ;;

  model-native-current-source-cascade-proof)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --source-parquet \
      --canonical-v2-parquet \
      --mtf-cache-dir \
      --pair-manifest \
      --required-history-start \
      --out; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.materialize_current_pair_source_cascade_proof_v1 "$@"
    ;;

  model-native-m1-feature-base)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --timeframe
    for flag in \
      --source-parquet \
      --alignment-parquet \
      --seq-structure-manifest \
      --output-parquet \
      --dataset-run-id \
      --pair-generation-id \
      --v29-registry-constants-json \
      --volatility-squeeze-manifest \
      --expected-volatility-squeeze-manifest-sha256; do
      require_flag "$cmd" "$flag" "$@"
    done
    # The complete 2.3m-row M1 inline signal extension needs more than the
    # canonical M5/MTF builders, but remains hard-capped below host memory.
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.materialize_entry_exit_m1_feature_base_v1 "$@"
    ;;

  model-native-m5-feature-base)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --timeframe --alignment-parquet
    for flag in \
      --source-parquet \
      --seq-structure-manifest \
      --output-parquet \
      --dataset-run-id \
      --pair-generation-id \
      --v29-registry-constants-json \
      --volatility-squeeze-manifest \
      --expected-volatility-squeeze-manifest-sha256; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" \
      "$PY" -m gx1.scripts.materialize_entry_exit_m1_feature_base_v1 "$@"
    ;;

  model-native-rebuild-preflight)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --source-parquet \
      --canonical-v2-parquet \
      --signal-manifest \
      --feature-ranking-json \
      --mtf-cache-dir \
      --tape-root \
      --m1-lifecycle-pair-manifest-json \
      --m1-lifecycle-pair-generation-root \
      --m1-feature-base-parquet \
      --m5-feature-base-parquet \
      --exit-lifecycle-dir \
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
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_rebuild_preflight_v1 "$@"
    ;;

  model-native-post-rebuild-readiness)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --event-root \
      --repo-dir \
      --chain-terminal-json \
      --test-seal-json \
      --test-seal-sha256 \
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
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_post_rebuild_readiness_v1 "$@"
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
      --seq-structure-manifest \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_foundation_features_v1 "$@"
    ;;

  model-native-cross-surface-overlap)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --signal-manifest \
      --m1-feature-base-parquet \
      --m5-feature-base-parquet \
      --mtf-cache-dir \
      --out-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_cross_surface_feature_overlap_v1 "$@"
    ;;

  model-native-feature-surface-liveness)
    reject_non_authoritative_args "$@"
    for flag in \
      --run-id \
      --signal-manifest \
      --m1-feature-base-parquet \
      --m5-feature-base-parquet \
      --cross-surface-overlap-json \
      --out-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_feature_surface_liveness_v1 "$@"
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
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_foundation_targets_v1 "$@"
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
      --seq-structure-manifest \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_specialist_feature_groups_v1 "$@"
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
    # This is a report-only hash/provenance gate: it never materializes data
    # or starts training, so keep it within the narrow audit safety envelope.
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_foundation_adoption_candidate_v1 "$@"
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
      --out-dir \
      --run-id \
      --memory-cap \
      --swap-cap \
      --sample-rows \
      --batch-size; do
      require_flag "$cmd" "$flag" "$@"
    done
    # The smoke manifest is immutable evidence only, not a dataset producer.
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_smoke_manifest_v1 "$@"
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
    # Readiness verification has no model or dataset side effect.
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_model_native_seq513_smoke_readiness_v1 "$@"
    ;;

  model-native-trainability-readiness)
    reject_non_authoritative_args "$@"
    for flag in \
      --smart-post-rebuild-readiness-json \
      --smart-smoke-readiness-json \
      --full-input-liveness-json \
      --control-script \
      --trainer-source \
      --train-wrapper \
      --candidate-readiness-script \
      --selective-edge-script \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    # Source/control inspection is report-only and must not reserve producer memory.
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_model_native_seq513_trainability_readiness_v1 "$@"
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
      --train-parquet \
      --val-parquet \
      --unified-exit-lifecycle-manifest-json \
      --m5-prebuilt-path \
      --multi-tf-cache-manifest-json \
      --post-rebuild-readiness-json \
      --prefreeze-test-seal-json \
      --prefreeze-test-seal-sha256 \
      --full-input-liveness-audit-json \
      --feature-audit-json \
      --target-audit-json \
      --specialist-audit-json \
      --pretrain-audit-json \
      --execution-causality-audit-json \
      --train-sequence-integrity-audit-json \
      --val-sequence-integrity-audit-json \
      --train-sequence-source-audit-json \
      --val-sequence-source-audit-json \
      --trainability-readiness-json \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    recipe_profile=
    for ((recipe_index=0; recipe_index < ${#COMMAND_ARGS[@]}; recipe_index++)); do
      recipe_arg="${COMMAND_ARGS[recipe_index]}"
      if [[ "$recipe_arg" == "--profile" ]]; then
        ((recipe_index + 1 < ${#COMMAND_ARGS[@]})) \
          || die "$cmd --profile requires a value"
        recipe_profile="${COMMAND_ARGS[recipe_index + 1]}"
      elif [[ "$recipe_arg" == --profile=* ]]; then
        recipe_profile="${recipe_arg#--profile=}"
      fi
    done
    if [[ "$recipe_profile" == "smoke" ]]; then
      require_flag "$cmd" --smoke-manifest-json "$@"
      require_flag "$cmd" --smoke-readiness-json "$@"
    elif [[ "$recipe_profile" == "candidate" ]]; then
      require_flag "$cmd" --candidate-readiness-json "$@"
      require_flag "$cmd" --smoke-bundle-audit-json "$@"
    else
      die "$cmd --profile must be exactly smoke or candidate"
    fi
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_train_recipe_audit_v1 "$@"
    ;;

  model-native-execution-causality-audit)
    # Manifest-only.  This runs before any trainer/GPU allocation and returns
    # BLOCK when one active auxiliary still uses a same-close proxy.
    reject_non_authoritative_args "$@"
    for flag in \
      --dataset-dir \
      --signal-manifest \
      --train-manifest \
      --val-manifest \
      --train-lifecycle-manifest \
      --val-lifecycle-manifest \
      --output; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_execution_causality_v1 "$@"
    ;;

  model-native-smoke-train)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --attended-smoke --research-smoke
    for flag in \
      --train-sequence-source-audit-json \
      --val-sequence-source-audit-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/run_entry_model_native_seq513_train.sh" --profile smoke "$@"
    ;;

  model-native-attended-smoke-train)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --research-smoke
    for flag in \
      --train-sequence-source-audit-json \
      --val-sequence-source-audit-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/run_entry_model_native_seq513_train.sh" --profile smoke --attended-smoke "$@"
    ;;

  model-native-attended-cpu-smoke-train)
    # CPU-only recovery lane after a CUDA safety stop. It uses the same full
    # data proofs and staged time/cgroup limits, but has no CUDA allocation.
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --research-smoke --attended-smoke
    for flag in \
      --train-sequence-source-audit-json \
      --val-sequence-source-audit-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/run_entry_model_native_seq513_train.sh" --profile smoke --attended-cpu-smoke "$@"
    ;;

  model-native-attended-hardware-smoke)
    # A diagnostic-only CUDA architecture step.  It has no dataset, bundle or
    # candidate output and must remain distinct from model-native smoke train.
    reject_non_authoritative_args "$@"
    HARDWARE_AUDIT_JSON= HARDWARE_RUN_MODE=
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --specialist-audit-json)
          [[ -z "$HARDWARE_AUDIT_JSON" && $# -ge 2 ]] \
            || die "$cmd requires one --specialist-audit-json PATH"
          HARDWARE_AUDIT_JSON="$2"
          shift 2
          ;;
        --dry-run|--execute)
          [[ -z "$HARDWARE_RUN_MODE" ]] || die "$cmd requires exactly one run mode"
          HARDWARE_RUN_MODE="${1#--}"
          shift
          ;;
        *) die "$cmd accepts only --specialist-audit-json and --dry-run|--execute" ;;
      esac
    done
    [[ -n "$HARDWARE_AUDIT_JSON" && -n "$HARDWARE_RUN_MODE" ]] \
      || die "$cmd requires --specialist-audit-json and one run mode"
    [[ -f "$HARDWARE_AUDIT_JSON" && ! -L "$HARDWARE_AUDIT_JSON" ]] \
      || die "$cmd specialist audit must be an existing regular non-symlink file"
    HARDWARE_CMD=(
      "$REPO/scripts/gx1_capped_run.sh" --class trainer --mem 4G --swap 512M --attended-smoke --
      "$PY" -m gx1.scripts.attended_model_native_hardware_smoke_v1
      --attended-hardware-smoke --device cuda
      --specialist-audit-json "$HARDWARE_AUDIT_JSON"
    )
    if [[ "$HARDWARE_RUN_MODE" == dry-run ]]; then
      printf 'Validated attended hardware smoke (authority=none):'
      printf ' %q' "${HARDWARE_CMD[@]}"
      printf '\n'
      exit 0
    fi
    [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] \
      || die "$cmd --execute requires a clean worktree"
    exec "${HARDWARE_CMD[@]}"
    ;;

  model-native-sequence-roll-audit)
    # Full byte-level sequence/snapshot identity proof.  It is an audit only:
    # no target/model fitting, dataset production or downstream authority.
    reject_non_authoritative_args "$@"
    for flag in --parquet --manifest-json --out-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_sequence_roll_v1 "$@"
    ;;

  model-native-sequence-integrity-audit)
    # Full byte-level physical event-chain proof. Unlike the attended-only
    # reconstruction proof, this accepts causally filtered output rows while
    # proving their exact sequence overlap before any trainer/GPU allocation.
    reject_non_authoritative_args "$@"
    for flag in --parquet --manifest-json --out-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_sequence_integrity_v1 "$@"
    ;;

  model-native-sequence-source-reconstruction-audit)
    # Full byte-level proof that a causally filtered split can recover every
    # stored window from its immutable M5 feature surface.  This is a
    # data-storage audit only and does not allocate a model or GPU.
    reject_non_authoritative_args "$@"
    for flag in --parquet --manifest-json --out-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${AUDIT_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_sequence_source_reconstruction_v1 "$@"
    ;;

  model-native-smoke-bundle-audit)
    reject_non_authoritative_args "$@"
    for flag in \
      --bundle-dir \
      --dataset-dir \
      --val-manifest-json \
      --predictions-parquet \
      --prediction-report-json \
      --target-audit-json \
      --specialist-audit-json \
      --pretrain-audit-json \
      --out-dir \
      --device; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_entry_foundation_smoke_bundle_v1 "$@"
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
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_candidate_readiness_v1 \
      --edge-test-scope strict --min-active-specialists 8 "$@"
    ;;

  model-native-candidate-train)
    reject_non_authoritative_args "$@"
    for flag in \
      --train-sequence-source-audit-json \
      --val-sequence-source-audit-json; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "$REPO/scripts/run_entry_model_native_seq513_train.sh" --profile candidate "$@"
    ;;

  model-native-selective-edge)
    reject_non_authoritative_args "$@"
    reject_flags "$cmd" --top-fracs --model-name --selection-score-mode
    for flag in \
      --bundle-dir \
      --dataset-dir \
      --splits \
      --evidence-stage \
      --device \
      --batch-size \
      --stream-chunk-rows \
      --m5-prebuilt-path \
      --multi-tf-cache-dir \
      --out-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    selective_edge_device=$(exact_flag_value "$cmd" --device "$@")
    if [[ "$selective_edge_device" == cuda ]]; then
      # Prediction evidence is required before the smoke-bundle audit. It is
      # nevertheless a long CUDA process, so route the only accepted CUDA
      # evaluator through the same one-second 220 W / thermal / VRAM guard as
      # canonical training. CPU evaluation remains the ordinary producer path.
      exec "$REPO/scripts/gx1_capped_run.sh" --class producer --mem 10G --swap 512M --cuda-producer -- \
        "$PY" -m gx1.scripts.evaluate_entry_candidate_selective_edge_v1 "$@"
    fi
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.evaluate_entry_candidate_selective_edge_v1 "$@"
    ;;

  model-native-seed-stability)
    reject_non_authoritative_args "$@"
    require_flag "$cmd" --out-dir "$@"
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.verify_entry_candidate_seed_stability_v1 "$@"
    ;;

  model-native-sizing-fit-calibration)
    reject_non_authoritative_args "$@"
    for flag in \
      --predictions \
      --predictions-sha256 \
      --prediction-report \
      --bundle-dir \
      --dataset-dir \
      --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      fit-calibration "$@"
    ;;

  model-native-sizing-bind-bundle)
    reject_non_authoritative_args "$@"
    for flag in --source-bundle-dir --out-bundle-dir --calibration; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      bind-bundle "$@"
    ;;

  model-native-sizing-materialize-test-oos)
    reject_non_authoritative_args "$@"
    for flag in \
      --calibration \
      --test-predictions \
      --test-predictions-sha256 \
      --test-prediction-report \
      --bundle-dir \
      --dataset-dir \
      --source-tape \
      --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      materialize-test-oos "$@"
    ;;

  model-native-sizing-finalize-test-proof)
    reject_non_authoritative_args "$@"
    for flag in --calibration --oos-source --authority-root; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.finalize_entry_model_native_sizing_v1 \
      finalize-test-proof "$@"
    ;;

  model-native-trade-path-metrics)
    # Exact unified-Exit rows can yield research-only MAE/MFE/path diagnostics.
    # This route cannot create a candidate, net-PnL, demo or live authority.
    reject_non_authoritative_args "$@"
    for flag in \
      --replay-rows \
      --exit-trace-rows \
      --candidate-bundle-sha256 \
      --output-dir; do
      require_flag "$cmd" "$flag" "$@"
    done
    exec "${PRODUCER_CAP[@]}" "$PY" -m \
      gx1.scripts.materialize_entry_model_native_trade_path_metrics_v1 "$@"
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
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.verify_model_native_serve_parity_v1 "$@"
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
    exec "${PRODUCER_CAP[@]}" "$PY" -m gx1.scripts.audit_model_native_direction_pockets_v1 "$@"
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
