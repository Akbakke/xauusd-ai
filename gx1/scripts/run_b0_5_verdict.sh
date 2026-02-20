#!/bin/bash
# B0.5 Verdict Runner
# Runs RUN A (workers=1) and RUN B (workers=7) and produces verdict

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="/home/andre2/GX1_DATA/reports/replay_eval"
OUTPUT_DIR_A="${BASE_DIR}/B0_5_RUN_A_WORKERS_1_${TIMESTAMP}"
OUTPUT_DIR_B="${BASE_DIR}/B0_5_RUN_B_WORKERS_7_${TIMESTAMP}"

# Environment
export GX1_RUN_MODE=TRUTH
export GX1_CANONICAL_BUNDLE_DIR=/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_GATED_FUSION_ENABLED=1
export GX1_XGB_INPUT_FINGERPRINT=1
export GX1_XGB_INPUT_FINGERPRINT_SAMPLE_N=10
export GX1_XGB_INPUT_FINGERPRINT_MAX_PER_SESSION=500
export GX1_TRUTH_TELEMETRY=1
export GX1_TRUTH_MODE=1

# Common args
START_TS="2025-01-03T00:00:00Z"
END_TS="2025-01-10T00:00:00Z"
POLICY="/home/andre2/src/GX1_ENGINE/gx1/configs/entry_configs/ENTRY_V10_CTX_SNIPER_REPLAY.yaml"
DATA="/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet"
BUNDLE_DIR="/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"
CHUNK_PADDING=7

cd /home/andre2/src/GX1_ENGINE

echo "=== B0.5 RUN A: workers=1 ==="
echo "Output dir: $OUTPUT_DIR_A"
echo "Starting..."

# Clean up any existing dir
rm -rf "$OUTPUT_DIR_A"

gx1/scripts/run_replay_canonical.sh \
  --start-ts "$START_TS" \
  --end-ts "$END_TS" \
  --workers 1 \
  --chunk-local-padding-days "$CHUNK_PADDING" \
  --output-dir "$OUTPUT_DIR_A" \
  --policy "$POLICY" \
  --data "$DATA" \
  --bundle-dir "$BUNDLE_DIR" \
  2>&1 | tee "${OUTPUT_DIR_A}/run.log" || true

echo ""
echo "=== B0.5 RUN B: workers=7 ==="
echo "Output dir: $OUTPUT_DIR_B"
echo "Starting..."

# Clean up any existing dir
rm -rf "$OUTPUT_DIR_B"

gx1/scripts/run_replay_canonical.sh \
  --start-ts "$START_TS" \
  --end-ts "$END_TS" \
  --workers 7 \
  --chunk-local-padding-days "$CHUNK_PADDING" \
  --output-dir "$OUTPUT_DIR_B" \
  --policy "$POLICY" \
  --data "$DATA" \
  --bundle-dir "$BUNDLE_DIR" \
  2>&1 | tee "${OUTPUT_DIR_B}/run.log" || true

echo ""
echo "=== Both runs completed ==="
echo "RUN A: $OUTPUT_DIR_A"
echo "RUN B: $OUTPUT_DIR_B"
