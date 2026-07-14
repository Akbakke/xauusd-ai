#!/usr/bin/env bash
set -euo pipefail

# Vedtak-gated candidate train launcher for the audited Entry foundation seq146
# dataset. This wrapper is fail-closed behind candidate-readiness and never
# promotes, pins, starts shadow, or touches live/practice order paths.

REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python

FOUNDATION_DATASET=$DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral_20260629_directional_smc
FOUNDATION_STEM=v10_foundation_seq146__HOLD_03B
M5_PREBUILT=$DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet
SPECIALIST_AUDIT=$DATA/reports/entry_specialist_feature_group_audit_20260628_v1/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json
SMOKE_BUNDLE_AUDIT=$DATA/reports/entry_foundation_smoke_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json
CANDIDATE_AUDIT_OUT=$DATA/reports/entry_candidate_bundle_audit_20260628_v1
CANDIDATE_READINESS_DIR=$DATA/reports/entry_candidate_readiness_20260628_v1
CANDIDATE_READINESS_JSON=$CANDIDATE_READINESS_DIR/ENTRY_CANDIDATE_READINESS_latest.json
CANDIDATE_TRAIN_MANIFEST_DIR=$DATA/reports/entry_foundation_candidate_train_manifests_20260628_v1
XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_DIR=$DATA/reports/xau_direction_repair_pretrain_audit_20260713_v1

VEDTAK="${ENTRY_FOUNDATION_CANDIDATE_VEDTAK:-}"
RUN_FLAVOR=foundation_seq146
SPECIALIST_CONTRACT_MODE=foundation_seq146
EXPECTED_SIGNAL_DIM=146
CANDIDATE_READINESS_EDGE_SCOPE=strict
CANDIDATE_READINESS_NEXT_CMD="scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit"
CANDIDATE_READINESS_RERUN_CMD="scripts/entry_next_edge_control.sh candidate-readiness"
DEVICE=auto
EPOCHS=8
BATCH_SIZE=96
SEED="${ENTRY_FOUNDATION_CANDIDATE_SEED:-1337}"
SUBSAMPLE_ROWS="${ENTRY_FOUNDATION_CANDIDATE_SUBSAMPLE_ROWS:-0}"
DRY_RUN=0
OUT_BUNDLE_OVERRIDE=""
SMOKE_BUNDLE_AUDIT_ARG="$SMOKE_BUNDLE_AUDIT"
AUDIT_AFTER=1
AUDIT_DEVICE=cpu
AUDIT_BATCH_SIZE=256
CANDIDATE_MEM_CAP="${ENTRY_FOUNDATION_CANDIDATE_MEM_CAP:-32G}"
CANDIDATE_SWAP_CAP="${ENTRY_FOUNDATION_CANDIDATE_SWAP_CAP:-2G}"
CANDIDATE_CAPPED_RUNNER=scripts/gx1_capped_run.sh
CANDIDATE_BAD_PATH_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_WEIGHT:-1.00}"
CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT:-2.00}"
CANDIDATE_BAD_PATH_QUALITY_RANK_MARGIN="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_MARGIN:-0.25}"
CANDIDATE_BAD_PATH_QUALITY_RANK_QUANTILE="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_QUANTILE:-0.25}"
CANDIDATE_BAD_PATH_PROB_PENALTY="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY:-0.24}"
CANDIDATE_PATH_QUALITY_RANK_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_WEIGHT:-2.00}"
CANDIDATE_PATH_QUALITY_RANK_MARGIN="${ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_MARGIN:-0.25}"
CANDIDATE_PATH_QUALITY_RANK_QUANTILE="${ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_QUANTILE:-0.25}"
CANDIDATE_PRED_BALANCE_ALPHA="${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA:-0.05}"
CANDIDATE_PRED_BALANCE_TARGET="${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET:-label}"
CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS="${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS:-1.0,1.0,1.0}"
CANDIDATE_DIRECTION_CE_SCALE="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE:-1.30}"
CANDIDATE_TAIL_DIRECTION_CE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_CE_WEIGHT:-0.35}"
CANDIDATE_TAIL_DIRECTION_QUALITY_QUANTILE="${ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_QUALITY_QUANTILE:-0.70}"
CANDIDATE_TAIL_DIRECTION_MIN_BATCH="${ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_MIN_BATCH:-8}"
CANDIDATE_CKPT_MONITOR="${ENTRY_FOUNDATION_CANDIDATE_CKPT_MONITOR:-dir_acc}"
CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT:-0.0}"
CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL="${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL:-0.0}"
CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE="${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE:-0.0}"
CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT:-0.0}"
CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION:-0.0}"
CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR:-0.0}"
CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE:-1.0}"
CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT:-0.0}"
CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION:-0.0}"
CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR:-0.0}"
CANDIDATE_DIRECTION_SLICE_MIN_LABEL_RATE="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_LABEL_RATE:-0.10}"
CANDIDATE_DIRECTION_SLICE_MIN_ROWS="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_ROWS:-8}"
CANDIDATE_DIRECTION_SLICE_CTX_CAT_INDICES="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_CTX_CAT_INDICES:-0,1,2,3,4}"
CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT:-0.0}"
CANDIDATE_DIRECTION_SLICE_RECALL_PROB_FLOOR="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_PROB_FLOOR:-0.30}"
CANDIDATE_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE:-0.10}"
CANDIDATE_DIRECTION_SLICE_RECALL_MIN_ROWS="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_MIN_ROWS:-8}"
CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT:-0.0}"
CANDIDATE_DIRECTION_VS_FLAT_MARGIN="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN:-0.0}"
CANDIDATE_HIER_LEGACY_CE_MULT="${ENTRY_FOUNDATION_CANDIDATE_HIER_LEGACY_CE_MULT:-0.35}"
CANDIDATE_HIER_TRADE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_WEIGHT:-0.85}"
CANDIDATE_HIER_SIDE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_WEIGHT:-0.85}"
CANDIDATE_HIER_UTILITY_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_UTILITY_WEIGHT:-0.20}"
CANDIDATE_HIER_BAD_PATH_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_BAD_PATH_WEIGHT:-0.35}"
CANDIDATE_HIER_MAE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_MAE_WEIGHT:-0.15}"
CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT:-0.0}"
CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS="${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS:-10.0}"
CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP="${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP:-20.0}"
CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT:-0.0}"
CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT:-0.0}"
CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS="${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS:-10.0}"
CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT:-$CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT}"
CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT:-$CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT}"
CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT:-0.0}"
CANDIDATE_TRENDLINE_RAIL_MARGIN="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_MARGIN:-0.50}"
CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS="${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS:-5.0}"
CANDIDATE_ENABLE_XAU_DIRECTION_REPAIR_HEADS=0
CANDIDATE_ANCHOR_GATE_INIT="${ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT:-1.0}"
CANDIDATE_SYMMETRIC_NEGATIVES="${ENTRY_FOUNDATION_CANDIDATE_SYMMETRIC_NEGATIVES:-1}"
CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR="${ENTRY_FOUNDATION_CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR:-1.0}"
CANDIDATE_SPECIALIST_GATE_ENTROPY_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_ENTROPY_WEIGHT:-0.05}"
CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT:-0.25}"
CANDIDATE_SPECIALIST_GATE_MIN_MEAN="${ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_MIN_MEAN:-0.02}"
CANDIDATE_SPECIALIST_FUSION_SCALE="${ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_FUSION_SCALE:-0.25}"
CANDIDATE_LR="${ENTRY_FOUNDATION_CANDIDATE_LR:-3e-4}"
CANDIDATE_WEIGHT_DECAY="${ENTRY_FOUNDATION_CANDIDATE_WEIGHT_DECAY:-1e-5}"
CANDIDATE_GRAD_CLIP_NORM="${ENTRY_FOUNDATION_CANDIDATE_GRAD_CLIP_NORM:-1.0}"
CANDIDATE_MULTI_TF_SCALE="${ENTRY_FOUNDATION_CANDIDATE_MULTI_TF_SCALE:-0.5}"
CANDIDATE_MTF_DIR_SCALE_INIT="${ENTRY_FOUNDATION_CANDIDATE_MTF_DIR_SCALE_INIT:-0.2}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_entry_foundation_seq146_candidate_train.sh --vedtak <id> [options]

Options:
  --vedtak <id>        Required explicit user decision id for candidate train.
  --device <auto|cpu|cuda>
  --epochs <n>         Default: 8
  --batch-size <n>     Default: 96
  --seed <n>           Default: 1337
  --subsample-rows <n> Default: 0 (all rows). Intended for short sweep trials.
  --out-bundle-dir <path>
                       Explicit output bundle dir. Intended for sweep trials.
  --smoke-bundle-audit-json <path>
                       Edge-required smoke bundle audit to satisfy candidate-readiness.
  --challenger-seq215  Use the audited seq215 challenger dataset and 8-specialist
                       challenger_seq215 contract. Still requires candidate-readiness,
                       explicit vedtak, clean git and post-candidate audit.
  --smart-seq520       Use the audited smart seq520 candidate dataset and
                       smart_seq520_candidate contract. Still requires smart
                       trainability, candidate-readiness, explicit SMART vedtak,
                       clean git and post-candidate audit.
  --skip-candidate-audit
                       Do not run strict post-candidate bundle/head-contract audit.
  --audit-device <auto|cpu|cuda>
                       Default: cpu
  --audit-batch-size <n>
                       Default: 256
  --dry-run            Print the train command after candidate-readiness passes.

Resource caps:
  ENTRY_FOUNDATION_CANDIDATE_MEM_CAP   Default: 32G
  ENTRY_FOUNDATION_CANDIDATE_SWAP_CAP  Default: 2G

Candidate edge recipe env:
  ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_WEIGHT                 Default: 1.00
  ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT    Default: 2.00
  ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_MARGIN    Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_QUANTILE  Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_WEIGHT        Default: 2.00
  ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_MARGIN        Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_PATH_QUALITY_RANK_QUANTILE      Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA              Default: 0.05
  ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET             Default: label
  ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS      Default: 1.0,1.0,1.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE              Default: 1.30
  ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_CE_WEIGHT        Default: 0.35
  ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_QUALITY_QUANTILE Default: 0.70
  ENTRY_FOUNDATION_CANDIDATE_TAIL_DIRECTION_MIN_BATCH        Default: 8
  ENTRY_FOUNDATION_CANDIDATE_CKPT_MONITOR                    Default: dir_acc
  ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE Default: 1.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_LEGACY_CE_MULT             Default: 0.35
  ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT        Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS Default: 10.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP Default: 20.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT      Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT  Default: 0.0
  ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS  Default: 10.0
  ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT                Default: 1.0
  ENTRY_FOUNDATION_CANDIDATE_SYMMETRIC_NEGATIVES             Default: 1
  ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_ENTROPY_WEIGHT  Default: 0.05
  ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT  Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_MIN_MEAN        Default: 0.02
  ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_FUSION_SCALE         Default: 0.25
  ENTRY_FOUNDATION_CANDIDATE_LR                              Default: 3e-4
  ENTRY_FOUNDATION_CANDIDATE_WEIGHT_DECAY                    Default: 1e-5
  ENTRY_FOUNDATION_CANDIDATE_GRAD_CLIP_NORM                  Default: 1.0
  ENTRY_FOUNDATION_CANDIDATE_MULTI_TF_SCALE                  Default: 0.5
  ENTRY_FOUNDATION_CANDIDATE_MTF_DIR_SCALE_INIT              Default: 0.2

This is a full candidate-training path, not promotion. It will not run until
candidate-readiness proves a real smoke-train bundle with edge diagnostics.
After training it strict-loads the candidate bundle and verifies the active
head contract before replay/selective-edge gates. It writes a pre-train
candidate manifest before trainer start and passes that manifest into the
post-candidate audit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vedtak) VEDTAK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --subsample-rows) SUBSAMPLE_ROWS="$2"; shift 2 ;;
    --out-bundle-dir) OUT_BUNDLE_OVERRIDE="$2"; shift 2 ;;
    --smoke-bundle-audit-json) SMOKE_BUNDLE_AUDIT_ARG="$2"; shift 2 ;;
    --challenger-seq215)
      RUN_FLAVOR=challenger_seq215
      FOUNDATION_DATASET=$DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_challenger_seq215_neutral_20260630
      FOUNDATION_STEM=v10_challenger_seq215__HOLD_03B
      SPECIALIST_AUDIT=$DATA/reports/entry_specialist_feature_group_audit_20260628_v1/challenger_seq215_20260630_contract8/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json
      SMOKE_BUNDLE_AUDIT_ARG=$DATA/reports/entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json
      CANDIDATE_AUDIT_OUT=$DATA/reports/entry_candidate_bundle_audit_20260628_v1/challenger_seq215_20260630
      SPECIALIST_CONTRACT_MODE=challenger_seq215
      EXPECTED_SIGNAL_DIM=215
      CANDIDATE_READINESS_NEXT_CMD="scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit"
      CANDIDATE_READINESS_RERUN_CMD="scripts/entry_next_edge_control.sh candidate-readiness-seq215"
      CANDIDATE_READINESS_DIR=$DATA/reports/entry_candidate_readiness_20260628_v1/challenger_seq215_20260630
      CANDIDATE_READINESS_JSON=$CANDIDATE_READINESS_DIR/ENTRY_CANDIDATE_READINESS_latest.json
      shift
      ;;
    --smart-seq520)
      RUN_FLAVOR=smart_seq520
      CANDIDATE_READINESS_EDGE_SCOPE=smoke
      # XAU-only direction repair candidate: use the fresh stage-5 rebuild
      # dataset. Older utilityrepair_v2 artifacts are explicitly audit-blocked
      # because they predate scalar bad-path/size-target repair and rail
      # feature gates.
      FOUNDATION_DATASET=$DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair
      FOUNDATION_STEM=v10_6yr_dataset__HOLD_03B
      SPECIALIST_AUDIT=$DATA/reports/entry_specialist_feature_group_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json
      SMOKE_BUNDLE_AUDIT_ARG=$DATA/reports/entry_foundation_smoke_bundle_audit_20260628_v1/smart_seq520_candidate/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json
      CANDIDATE_AUDIT_OUT=$DATA/reports/entry_candidate_bundle_audit_20260628_v1/smart_seq520_candidate
      SPECIALIST_CONTRACT_MODE=smart_seq520_candidate
      EXPECTED_SIGNAL_DIM=520
      CANDIDATE_ENABLE_XAU_DIRECTION_REPAIR_HEADS=1
      CANDIDATE_READINESS_NEXT_CMD="scripts/entry_next_edge_control.sh smart-smoke-train --vedtak <id> --require-edge-audit"
      CANDIDATE_READINESS_RERUN_CMD="scripts/entry_next_edge_control.sh candidate-readiness-smart"
      CANDIDATE_READINESS_DIR=$DATA/reports/entry_candidate_readiness_20260628_v1/smart_seq520_candidate
      CANDIDATE_READINESS_JSON=$CANDIDATE_READINESS_DIR/ENTRY_CANDIDATE_READINESS_latest.json
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA+x}" ]]; then
        CANDIDATE_PRED_BALANCE_ALPHA=0.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE+x}" ]]; then
        CANDIDATE_DIRECTION_CE_SCALE=4.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS+x}" ]]; then
        CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS=1.0,1.0,4.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT+x}" ]]; then
        CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL+x}" ]]; then
        CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE+x}" ]]; then
        CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT+x}" ]]; then
        CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION+x}" ]]; then
        CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION=0.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR+x}" ]]; then
        CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR=0.05
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE+x}" ]]; then
        CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.15
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT+x}" ]]; then
        CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION+x}" ]]; then
        CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR+x}" ]]; then
        CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT+x}" ]]; then
        CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT+x}" ]]; then
        CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN+x}" ]]; then
        CANDIDATE_DIRECTION_VS_FLAT_MARGIN=0.10
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_LEGACY_CE_MULT+x}" ]]; then
        CANDIDATE_HIER_LEGACY_CE_MULT=1.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_TRADE_WEIGHT=2.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_SIDE_WEIGHT=1.75
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_UTILITY_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_UTILITY_WEIGHT=1.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_BAD_PATH_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_BAD_PATH_WEIGHT=1.25
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_MAE_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_MAE_WEIGHT=0.35
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT=1.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS+x}" ]]; then
        CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS=15.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP+x}" ]]; then
        CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP=8.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT=5.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT+x}" ]]; then
        CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT=3.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS+x}" ]]; then
        CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS=30.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT=1.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=1.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT=1.50
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT=1.75
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=5.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT=4.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT=3.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT=5.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_MARGIN+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_MARGIN=1.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS+x}" ]]; then
        CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS=30.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR+x}" ]]; then
        CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR=1.00
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT+x}" ]]; then
        CANDIDATE_ANCHOR_GATE_INIT=0.0
      fi
      if [[ -z "${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY+x}" ]]; then
        CANDIDATE_BAD_PATH_PROB_PENALTY=0.0
      fi
      shift
      ;;
    --skip-candidate-audit) AUDIT_AFTER=0; shift ;;
    --audit-device) AUDIT_DEVICE="$2"; shift 2 ;;
    --audit-batch-size) AUDIT_BATCH_SIZE="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR_OVERRIDE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Optional explicit candidate dataset dir (same *_HOLD_03B_{train,val,test}.parquet
# stem). Used for the 2026-in-train re-split so the candidate learns the current
# regime. Provenance is recorded (not pinned) in the pre-train manifest.
# Vedtak SMART_SEQ520_candidate_train_20260703.
if [[ -n "${DATASET_DIR_OVERRIDE:-}" ]]; then
  if [[ ! -f "$DATASET_DIR_OVERRIDE/${FOUNDATION_STEM}_train.parquet" ]]; then
    echo "FATAL: --dataset-dir override missing ${FOUNDATION_STEM}_train.parquet in $DATASET_DIR_OVERRIDE" >&2
    exit 2
  fi
  echo "NOTE: dataset dir overridden -> $DATASET_DIR_OVERRIDE" >&2
  FOUNDATION_DATASET="$DATASET_DIR_OVERRIDE"
fi
if [[ "$RUN_FLAVOR" = "smart_seq520" ]]; then
  DATASET_LC=$(printf '%s' "$FOUNDATION_DATASET" | tr '[:upper:]' '[:lower:]')
  case "$DATASET_LC" in
    *utilityrepair*|*20260710*|*smart_candidate_20260630*|*julyext*)
      echo "FATAL: smart XAU direction repair refuses known stale dataset path: $FOUNDATION_DATASET" >&2
      echo "Use the fresh stage-5 v10_dataset_6yr_smartctx_xau_direction_repair dataset." >&2
      exit 2
      ;;
  esac
  case "$DATASET_LC" in
    *xau*) ;;
    *)
      echo "FATAL: smart XAU direction repair dataset path must be XAU-specific: $FOUNDATION_DATASET" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "${VEDTAK:-}" ]]; then
  echo "FATAL: --vedtak is required for Entry foundation candidate train." >&2
  echo "This is a full train command, so it must be an explicit user decision." >&2
  exit 2
fi
if [[ "$RUN_FLAVOR" = "challenger_seq215" && "$VEDTAK" != *"SEQ215"* ]]; then
  echo "FATAL: challenger seq215 candidate train requires an explicit SEQ215 vedtak id." >&2
  echo "Use a reviewed id containing SEQ215 only after seq215 smoke and candidate-readiness gates are green." >&2
  exit 2
fi
if [[ "$RUN_FLAVOR" = "smart_seq520" && "$VEDTAK" != *"SMART"* && "$VEDTAK" != *"SEQ520"* ]]; then
  echo "FATAL: smart seq520 candidate train requires an explicit SMART/SEQ520 vedtak id." >&2
  echo "Use a reviewed id only after smart trainability, smoke edge audit and candidate-readiness gates are green." >&2
  exit 2
fi

case "$DEVICE" in
  auto|cpu|cuda) ;;
  *) echo "FATAL: --device must be auto, cpu, or cuda; got $DEVICE" >&2; exit 2 ;;
esac
case "$AUDIT_DEVICE" in
  auto|cpu|cuda) ;;
  *) echo "FATAL: --audit-device must be auto, cpu, or cuda; got $AUDIT_DEVICE" >&2; exit 2 ;;
esac

cd "$REPO"

if [[ "$RUN_FLAVOR" = "smart_seq520" ]]; then
  "$PY" -m gx1.scripts.audit_xau_direction_repair_pretrain_v1 \
    --dataset-dir "$FOUNDATION_DATASET" \
    --stem "$FOUNDATION_STEM" \
    --out-dir "$XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_DIR" \
    --data-splits train,val,test \
    --require-rail-features \
    --fail-on-audit-fail \
    --quiet
  scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet
fi

require_clean_git_for_real_candidate_train() {
  GIT_STATUS_SHORT=$(git status --short)
  if [[ -n "$GIT_STATUS_SHORT" ]]; then
    echo "FATAL: real foundation candidate train requires clean git worktree." >&2
    echo "Use --dry-run for inspection without starting trainer." >&2
    echo "$GIT_STATUS_SHORT" >&2
    exit 2
  fi
}

if ! scripts/entry_next_edge_control.sh candidate-readiness \
  --smoke-bundle-audit-json "$SMOKE_BUNDLE_AUDIT_ARG" \
  --specialist-audit-json "$SPECIALIST_AUDIT" \
  --contract-mode "$SPECIALIST_CONTRACT_MODE" \
  --edge-test-scope "$CANDIDATE_READINESS_EDGE_SCOPE" \
  --out-dir "$CANDIDATE_READINESS_DIR" \
  --quiet
then
  cat >&2 <<EOF
FATAL: candidate-readiness is NOT_READY; candidate train is blocked.

Required next gate:
  $CANDIDATE_READINESS_NEXT_CMD

Then rerun:
  $CANDIDATE_READINESS_RERUN_CMD
EOF
  exit 2
fi

STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
OUT_BUNDLE=$DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_${RUN_FLAVOR}_candidate_$STAMP
if [[ -n "$OUT_BUNDLE_OVERRIDE" ]]; then
  OUT_BUNDLE="$OUT_BUNDLE_OVERRIDE"
fi
CANDIDATE_MANIFEST=$CANDIDATE_TRAIN_MANIFEST_DIR/ENTRY_FOUNDATION_CANDIDATE_TRAIN_RUN_MANIFEST_$STAMP.json

CMD=(
  env
  GX1_DATA="$DATA"
  GX1_REGIME_V4=1
  GX1_TREND_REGIME_FROM_D1=1
  GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH
  GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1
  GX1_PERTF_CLOSED_BAR=1
  ENTRY_AUX_BAD_PATH_WEIGHT="$CANDIDATE_BAD_PATH_WEIGHT"
  ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT="$CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT"
  ENTRY_BAD_PATH_QUALITY_RANK_MARGIN="$CANDIDATE_BAD_PATH_QUALITY_RANK_MARGIN"
  ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE="$CANDIDATE_BAD_PATH_QUALITY_RANK_QUANTILE"
  ENTRY_BAD_PATH_PROB_PENALTY="$CANDIDATE_BAD_PATH_PROB_PENALTY"
  ENTRY_PATH_QUALITY_RANK_WEIGHT="$CANDIDATE_PATH_QUALITY_RANK_WEIGHT"
  ENTRY_PATH_QUALITY_RANK_MARGIN="$CANDIDATE_PATH_QUALITY_RANK_MARGIN"
  ENTRY_PATH_QUALITY_RANK_QUANTILE="$CANDIDATE_PATH_QUALITY_RANK_QUANTILE"
  ENTRY_PRED_BALANCE_ALPHA="$CANDIDATE_PRED_BALANCE_ALPHA"
  ENTRY_PRED_BALANCE_TARGET="$CANDIDATE_PRED_BALANCE_TARGET"
  ENTRY_PRED_BALANCE_CLASS_WEIGHTS="$CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS"
  ENTRY_DIRECTION_CE_SCALE="$CANDIDATE_DIRECTION_CE_SCALE"
  ENTRY_TAIL_DIRECTION_CE_WEIGHT="$CANDIDATE_TAIL_DIRECTION_CE_WEIGHT"
  ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE="$CANDIDATE_TAIL_DIRECTION_QUALITY_QUANTILE"
  ENTRY_TAIL_DIRECTION_MIN_BATCH="$CANDIDATE_TAIL_DIRECTION_MIN_BATCH"
  GX1_V10_CKPT_MONITOR="$CANDIDATE_CKPT_MONITOR"
  ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT="$CANDIDATE_CKPT_CLASS_BALANCE_GUARD_WEIGHT"
  ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL="$CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL"
  ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE="$CANDIDATE_CKPT_CLASS_BALANCE_MIN_PRED_RATE"
  ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT="$CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT"
  ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION="$CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION"
  ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR="$CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR"
  ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE="$CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE"
  ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT="$CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT"
  ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION="$CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION"
  ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR="$CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR"
  ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE="$CANDIDATE_DIRECTION_SLICE_MIN_LABEL_RATE"
  ENTRY_DIRECTION_SLICE_MIN_ROWS="$CANDIDATE_DIRECTION_SLICE_MIN_ROWS"
  ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES="$CANDIDATE_DIRECTION_SLICE_CTX_CAT_INDICES"
  ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT="$CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT"
  ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR="$CANDIDATE_DIRECTION_SLICE_RECALL_PROB_FLOOR"
  ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE="$CANDIDATE_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE"
  ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS="$CANDIDATE_DIRECTION_SLICE_RECALL_MIN_ROWS"
  ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT="$CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT"
  ENTRY_DIRECTION_VS_FLAT_MARGIN="$CANDIDATE_DIRECTION_VS_FLAT_MARGIN"
  ENTRY_HIER_LEGACY_CE_MULT="$CANDIDATE_HIER_LEGACY_CE_MULT"
  ENTRY_HIER_TRADE_WEIGHT="$CANDIDATE_HIER_TRADE_WEIGHT"
  ENTRY_HIER_SIDE_WEIGHT="$CANDIDATE_HIER_SIDE_WEIGHT"
  ENTRY_HIER_UTILITY_WEIGHT="$CANDIDATE_HIER_UTILITY_WEIGHT"
  ENTRY_HIER_BAD_PATH_WEIGHT="$CANDIDATE_HIER_BAD_PATH_WEIGHT"
  ENTRY_HIER_MAE_WEIGHT="$CANDIDATE_HIER_MAE_WEIGHT"
  ENTRY_HIER_SIDE_VALIDITY_WEIGHT="$CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT"
  ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS="$CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS"
  ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP="$CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP"
  ENTRY_HIER_POCKET_ABSTAIN_WEIGHT="$CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT"
  ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT="$CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT"
  ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS="$CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS"
  ENTRY_TRENDLINE_RAIL_AUX_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT"
  ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT"
  ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT"
  ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT"
  ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT"
  ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT"
  ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT"
  ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT="$CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT"
  ENTRY_TRENDLINE_RAIL_MARGIN="$CANDIDATE_TRENDLINE_RAIL_MARGIN"
  ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS="$CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS"
  ENTRY_SYMMETRIC_NEGATIVES="$CANDIDATE_SYMMETRIC_NEGATIVES"
  ENTRY_FLAT_CLASS_WEIGHT_FLOOR="$CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR"
  ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT="$CANDIDATE_SPECIALIST_GATE_ENTROPY_WEIGHT"
  ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT="$CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT"
  ENTRY_SPECIALIST_GATE_MIN_MEAN="$CANDIDATE_SPECIALIST_GATE_MIN_MEAN"
  "$PY" -m gx1.models.entry_v10.entry_v10_ctx_train_v3
  --train
  --vedtak "$VEDTAK"
  --seed "$SEED"
  --device "$DEVICE"
  --dataset_dir "$FOUNDATION_DATASET"
  --dataset_train_parquet "$FOUNDATION_DATASET/${FOUNDATION_STEM}_train.parquet"
  --out_bundle_dir "$OUT_BUNDLE"
  --m5-prebuilt-path "$M5_PREBUILT"
  --epochs "$EPOCHS"
  --lr "$CANDIDATE_LR"
  --batch_size "$BATCH_SIZE"
  --early-stopping-patience 2
  --early-stopping-min-delta 0.0
  --num-workers 0
  --multi-tf-seq-len 96
  --multi-tf-scale "$CANDIDATE_MULTI_TF_SCALE"
  --per-tf-seq-len-h4 48
  --per-tf-seq-len-d1 30
  --grad-accum-steps 1
  --subsample-rows "$SUBSAMPLE_ROWS"
  --grad-clip-norm "$CANDIDATE_GRAD_CLIP_NORM"
  --weight-decay "$CANDIDATE_WEIGHT_DECAY"
  --enable-tf-agreement-head
  --enable-path-quality-variance-head
  --enable-position-size-head
  --enable-dip-head
  --enable-forecast-head
  --enable-timing-head
  --enable-tail-risk-head
  --enable-vol-forecast-head
  --enable-mtf-direction-head
  --mtf-dir-scale-init "$CANDIDATE_MTF_DIR_SCALE_INIT"
  --enable-specialist-fusion
  --specialist-audit-json "$SPECIALIST_AUDIT"
  --specialist-contract-mode "$SPECIALIST_CONTRACT_MODE"
  --specialist-num-layers 1
  --specialist-fusion-scale "$CANDIDATE_SPECIALIST_FUSION_SCALE"
  --enable-tf-input-scale
)

if [[ "$CANDIDATE_ENABLE_XAU_DIRECTION_REPAIR_HEADS" = "1" ]]; then
  CMD+=(
    --enable-xau-direction-repair-heads
    --anchor-gate-init "$CANDIDATE_ANCHOR_GATE_INIT"
  )
fi

# Post-candidate audit edge scope (red-team fix 2026-07-04): the smart flavor
# audits with --edge-test-scope candidate — per-slice BLANKET diagnostics are
# loud advisories there; the HARD per-slice responsibility lives in
# replay-readiness' SELECTED-TAIL slice-precision floors (min 0.50, min n 20),
# which stay untouched and currently fail-close on real weaknesses (e.g. EU).
# foundation/seq215 keep strict. Responsibility chain, ONE truth:
#   blanket slices  -> advisory @ candidate audit (this cmd)
#   selected-tail   -> HARD     @ replay-readiness (verify_entry_replay_readiness_v1)
AUDIT_EDGE_SCOPE=strict
if [[ "$RUN_FLAVOR" = "smart_seq520" ]]; then
  AUDIT_EDGE_SCOPE=candidate
fi
AUDIT_CMD=(
  scripts/entry_next_edge_control.sh
  audit-smoke-bundle
  --bundle-dir "$OUT_BUNDLE"
  --dataset-dir "$FOUNDATION_DATASET"
  --splits val,test
  --device "$AUDIT_DEVICE"
  --batch-size "$AUDIT_BATCH_SIZE"
  --require-edge
  --edge-test-scope "$AUDIT_EDGE_SCOPE"
  --require-head-contract
  --out-dir "$CANDIDATE_AUDIT_OUT"
  --pretrain-manifest-json "$CANDIDATE_MANIFEST"
)

if [[ "$DRY_RUN" = "1" ]]; then
  echo "Pre-candidate train manifest path: $CANDIDATE_MANIFEST"
  echo "Candidate resource cap: mem=$CANDIDATE_MEM_CAP swap=$CANDIDATE_SWAP_CAP runner=$CANDIDATE_CAPPED_RUNNER num_workers=0"
  printf 'Capped candidate train command:'
  printf ' %q' "$CANDIDATE_CAPPED_RUNNER" --mem "$CANDIDATE_MEM_CAP" --swap "$CANDIDATE_SWAP_CAP" -- "${CMD[@]}"
  printf '\n'
  printf 'Candidate train command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  if [[ "$AUDIT_AFTER" = "1" ]]; then
    printf 'Post-candidate audit command:'
    printf ' %q' "${AUDIT_CMD[@]}"
    printf '\n'
  else
    echo "Post-candidate audit command: skipped by --skip-candidate-audit"
  fi
  echo "Post-candidate gates still required: strict load, 2026 offline replay, session/side/regime slices, explicit promotion review."
  exit 0
fi

require_clean_git_for_real_candidate_train

mkdir -p "$CANDIDATE_TRAIN_MANIFEST_DIR"
"$PY" - "$CANDIDATE_MANIFEST" "$VEDTAK" "$OUT_BUNDLE" "$FOUNDATION_DATASET" "$SMOKE_BUNDLE_AUDIT_ARG" \
  "$SPECIALIST_AUDIT" "$CANDIDATE_READINESS_JSON" "$AUDIT_AFTER" "$DEVICE" "$EPOCHS" "$BATCH_SIZE" \
  "$CANDIDATE_MEM_CAP" "$CANDIDATE_SWAP_CAP" \
  "$RUN_FLAVOR" "$SPECIALIST_CONTRACT_MODE" "$EXPECTED_SIGNAL_DIM" \
  "${CMD[@]}" __AUDIT_CMD__ "${AUDIT_CMD[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
repo = Path("/home/andre2/src/GX1_ENGINE")
sep = "__AUDIT_CMD__"
sep_idx = sys.argv.index(sep)
train_cmd = sys.argv[17:sep_idx]
audit_cmd = sys.argv[sep_idx + 1:]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    return (proc.stdout if proc.returncode == 0 else proc.stderr).strip()


def gate_decision(report: dict, name: str) -> str:
    for gate in report.get("gates") or []:
        if isinstance(gate, dict) and str(gate.get("name")) == name:
            return str(gate.get("decision") or "")
    return ""


def command_env_value(command: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for item in command:
        if isinstance(item, str) and item.startswith(prefix):
            return item[len(prefix):]
    return None


def command_arg_value(command: list[str], flag: str) -> str | None:
    for idx, item in enumerate(command):
        if item == flag and idx + 1 < len(command):
            return command[idx + 1]
    return None


def report_json_path(report: dict, fallback: str | Path) -> str:
    raw = str(report.get("json_path") or "").strip()
    path = Path(raw).expanduser() if raw else Path(fallback).expanduser()
    return str(path.resolve())


smoke_audit = read_json(sys.argv[5])
specialist_audit = read_json(sys.argv[6])
candidate_readiness = read_json(sys.argv[7])
smoke_audit_json_path = report_json_path(smoke_audit, sys.argv[5])
specialist_audit_json_path = report_json_path(specialist_audit, sys.argv[6])
candidate_readiness_json_path = report_json_path(candidate_readiness, sys.argv[7])
smoke_pretrain = (
    smoke_audit.get("pretrain_manifest_contract")
    if isinstance(smoke_audit.get("pretrain_manifest_contract"), dict)
    else {}
)
head_contract = smoke_audit.get("head_contract") if isinstance(smoke_audit.get("head_contract"), dict) else {}
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract

trainer_indices, trainer_meta = _load_specialist_fusion_contract(
    Path(sys.argv[6]),
    expected_signal_dim=int(sys.argv[16]),
    contract_mode=sys.argv[15],
)

payload = {
    "schema_version": "entry_foundation_candidate_train_run_manifest_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_kind": f"vedtak_gated_{sys.argv[14]}_candidate_train",
    "manifest_variant": sys.argv[15],
    "candidate_variant": sys.argv[15],
    "specialist_contract_mode": sys.argv[15],
    "expected_signal_dim": int(sys.argv[16]),
    "vedtak": sys.argv[2],
    "out_bundle_dir": sys.argv[3],
    "promotion_shadow_live_allowed": False,
    "trainer_started_by_manifest_writer": False,
    "inputs": {
        "candidate_dataset_dir": sys.argv[4],
        "smoke_bundle_audit_json": smoke_audit_json_path,
        "specialist_audit_json": specialist_audit_json_path,
        "candidate_readiness_json": candidate_readiness_json_path,
    },
    "artifact_sha256": {
        "smoke_bundle_audit": sha256_file(smoke_audit_json_path),
        "specialist_audit": sha256_file(specialist_audit_json_path),
        "candidate_readiness": sha256_file(candidate_readiness_json_path),
    },
    "preflight_contracts": {
        "candidate_readiness": {
            "json_path": candidate_readiness.get("json_path"),
            "created_utc": candidate_readiness.get("created_utc"),
            "decision": candidate_readiness.get("decision"),
            "candidate_training_allowed_with_explicit_vedtak": candidate_readiness.get(
                "candidate_training_allowed_with_explicit_vedtak"
            ),
            "artifact_provenance_decision": gate_decision(candidate_readiness, "artifact_provenance"),
            "artifact_fingerprints": candidate_readiness.get("artifact_fingerprints") or {},
            "failures": candidate_readiness.get("failures") or [],
        },
        "smoke_edge_audit": {
            "decision": smoke_audit.get("decision"),
            "bundle_dir": smoke_audit.get("bundle_dir"),
            "required_training_specialists": smoke_audit.get("required_training_specialists") or [],
            "specialist_groups": (smoke_audit.get("bundle_summary") or {}).get("specialist_groups") or [],
            "pretrain_manifest_contract_decision": smoke_pretrain.get("decision"),
            "feature_objective_coverage_all_present": smoke_pretrain.get("feature_objective_coverage_all_present"),
            "feature_objective_liveness_all_live": smoke_pretrain.get("feature_objective_liveness_all_live"),
            "feature_source_field_liveness_all_live": smoke_pretrain.get("feature_source_field_liveness_all_live"),
            "specialist_active_heads_match_target": smoke_pretrain.get("specialist_active_heads_match_target"),
            "specialist_blocked_heads_match_target": smoke_pretrain.get("specialist_blocked_heads_match_target"),
            "specialist_objective_routing_all_present_and_expected": smoke_pretrain.get(
                "specialist_objective_routing_all_present_and_expected"
            ),
            "specialist_model_contract_valid": smoke_pretrain.get("specialist_model_contract_valid"),
            "specialist_model_contract_set_exact": smoke_pretrain.get("specialist_model_contract_set_exact"),
            "specialist_model_contract_owned_objectives_match": smoke_pretrain.get(
                "specialist_model_contract_owned_objectives_match"
            ),
            "smoke_dataset_audit_provenance_all_artifacts_present": smoke_pretrain.get(
                "smoke_dataset_audit_provenance_all_artifacts_present"
            ),
            "smoke_dataset_audit_provenance_all_artifact_hashes_present": smoke_pretrain.get(
                "smoke_dataset_audit_provenance_all_artifact_hashes_present"
            ),
            "worktree_critical_gate_review_ok": smoke_pretrain.get("worktree_critical_gate_review_ok"),
        },
        "target_foundation": {
            "decision": head_contract.get("decision"),
            "active_training_heads": head_contract.get("active_training_heads") or [],
            "blocked_heads": head_contract.get("blocked_heads") or [],
        },
        "specialist_contract": {
            "decision": specialist_audit.get("decision"),
            "required_training_specialists": specialist_audit.get("required_training_specialists") or [],
            "trainable_specialists": trainer_meta.get("trainable_specialists") or sorted(trainer_indices),
            "trainer_loader_group_feature_counts": trainer_meta.get("group_feature_counts") or {},
            "excluded_specialist_groups": trainer_meta.get("excluded_specialist_groups") or {},
            "specialist_model_contract_valid": specialist_audit.get("specialist_model_contract_valid"),
            "specialist_model_contract_failures": specialist_audit.get("specialist_model_contract_failures") or [],
            "specialist_model_contract": specialist_audit.get("specialist_model_contract") or {},
            "architecture_active_heads": (
                (
                    (specialist_audit.get("architecture_contract") or {}).get("recommended_fusion") or {}
                ).get("active_heads")
                or (
                    (specialist_audit.get("architecture_contract") or {}).get("recommended_fusion") or {}
                ).get("heads")
                or []
            ),
            "architecture_blocked_heads": (
                (
                    (specialist_audit.get("architecture_contract") or {}).get("recommended_fusion") or {}
                ).get("blocked_heads")
                or []
            ),
            "foundation_objective_routing_all_present_and_expected": specialist_audit.get(
                "foundation_objective_routing_all_present_and_expected"
            ),
            "specialist_input_liveness_all_live": specialist_audit.get("specialist_input_liveness_all_live"),
            "specialist_input_liveness": [
                {
                    "split": row.get("split"),
                    "specialist": row.get("specialist"),
                    "live_feature_count": row.get("live_feature_count"),
                    "feature_count": row.get("feature_count"),
                    "min_required_live_feature_count": row.get("min_required_live_feature_count"),
                    "nonfinite_count": row.get("nonfinite_count"),
                    "mean_active_rate": row.get("mean_active_rate"),
                }
                for row in specialist_audit.get("specialist_input_liveness", [])
                if isinstance(row, dict)
            ],
        },
    },
    "options": {
        "audit_after": sys.argv[8] == "1",
        "device": sys.argv[9],
        "epochs": int(sys.argv[10]),
        "batch_size": int(sys.argv[11]),
        "memory_cap": sys.argv[12],
        "swap_cap": sys.argv[13],
        "cgroup_runner": "scripts/gx1_capped_run.sh",
        "uses_gx1_capped_run": True,
        "num_workers": int(command_arg_value(train_cmd, "--num-workers") or -1),
    },
    "commands": {
        "train": train_cmd,
        "post_candidate_audit": audit_cmd,
    },
    "candidate_recipe_env": {
        name: command_env_value(train_cmd, name)
        for name in [
            "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES",
            "ENTRY_AUX_BAD_PATH_WEIGHT",
            "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
            "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN",
            "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE",
            "ENTRY_PATH_QUALITY_RANK_WEIGHT",
            "ENTRY_PATH_QUALITY_RANK_MARGIN",
            "ENTRY_PATH_QUALITY_RANK_QUANTILE",
            "ENTRY_PRED_BALANCE_ALPHA",
            "ENTRY_PRED_BALANCE_TARGET",
            "ENTRY_PRED_BALANCE_CLASS_WEIGHTS",
            "ENTRY_DIRECTION_CE_SCALE",
            "ENTRY_TAIL_DIRECTION_CE_WEIGHT",
            "ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE",
            "ENTRY_TAIL_DIRECTION_MIN_BATCH",
            "ENTRY_BAD_PATH_PROB_PENALTY",
            "GX1_V10_CKPT_MONITOR",
            "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT",
            "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL",
            "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE",
            "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT",
            "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION",
            "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR",
            "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE",
            "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT",
            "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION",
            "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR",
            "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE",
            "ENTRY_DIRECTION_SLICE_MIN_ROWS",
            "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES",
            "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT",
            "ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR",
            "ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE",
            "ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS",
            "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT",
            "ENTRY_DIRECTION_VS_FLAT_MARGIN",
            "ENTRY_HIER_LEGACY_CE_MULT",
            "ENTRY_HIER_TRADE_WEIGHT",
            "ENTRY_HIER_SIDE_WEIGHT",
            "ENTRY_HIER_UTILITY_WEIGHT",
            "ENTRY_HIER_BAD_PATH_WEIGHT",
            "ENTRY_HIER_MAE_WEIGHT",
            "ENTRY_HIER_SIDE_VALIDITY_WEIGHT",
            "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS",
            "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP",
            "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT",
            "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT",
            "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS",
            "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_MARGIN",
            "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS",
            "ENTRY_SYMMETRIC_NEGATIVES",
            "ENTRY_FLAT_CLASS_WEIGHT_FLOOR",
            "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT",
            "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT",
            "ENTRY_SPECIALIST_GATE_MIN_MEAN",
        ]
    },
    "git": {
        "repo": str(repo),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": git("status", "--short").splitlines(),
    },
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
latest = manifest_path.parent / "ENTRY_FOUNDATION_CANDIDATE_TRAIN_RUN_MANIFEST_latest.json"
latest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Pre-candidate train manifest written: {manifest_path}")
PY

"$CANDIDATE_CAPPED_RUNNER" --mem "$CANDIDATE_MEM_CAP" --swap "$CANDIDATE_SWAP_CAP" -- "${CMD[@]}"
echo "Candidate bundle written: $OUT_BUNDLE"
if [[ "$AUDIT_AFTER" = "1" ]]; then
  "${AUDIT_CMD[@]}"
else
  echo "Candidate bundle audit skipped by --skip-candidate-audit: $OUT_BUNDLE"
fi
echo "Post-candidate gates still required: strict load, 2026 offline replay, session/side/regime slices, explicit promotion review."
