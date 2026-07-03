#!/usr/bin/env bash
# ENTRY_V10.1 Offline Pipeline Runner
#
# Runs the complete ENTRY_V10.1 offline training pipeline:
# 1. Build V9 entry dataset (2025 data)
# 2. Train XGBoost models
# 3. Run XGBoost inference
# 4. Build V10.1 dataset (seq_len=90)
# 5. Train V10.1 Transformer (deeper variant)
# 6. Evaluate V10.1
# 7. Run threshold/label quality analysis
#
# Usage: bash scripts/run_entry_v10_1_offline_pipeline.sh
# (Run from repo root)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=========================================="
echo "ENTRY_V10.1 Offline Pipeline"
echo "=========================================="
echo ""
echo "V10.1 improvements:"
echo "  - Longer sequences: seq_len=90 (30 → 90 bars, ~7.5 hours on M5)"
echo "  - Deeper Transformer: num_layers=6, d_model=256, dim_feedforward=1024"
echo "  - Adaptive threshold engine"
echo "  - NO dummy/fallback values - HARD FEIL if data missing"
echo ""

# Step 1: Build V9 entry dataset (FULLYEAR 2025 data - same source as P4.1 replays)
echo "=========================================="
echo ">>> STEP 1/7: Building V9 entry dataset (FULLYEAR 2025)"
echo "=========================================="
echo "Data source: Same as P4.1 replays"
echo "Date range: 2025-01-01 to 2025-12-31"
echo ""
python -m gx1.rl.entry_v9.build_entry_v9_full_dataset \
    --start-date 2025-01-01 \
    --end-date 2025-12-31 \
    --output-parquet data/entry_v9/full_2025.parquet
echo ""
echo ">>> STEP 1/7 DONE: V9 dataset built"
echo ""

# Step 2: Train XGBoost models
echo ">>> STEP 2/7: Training XGBoost snapshot models"
python -m gx1.rl.entry_v10.train_entry_xgb_v10 \
    --dataset data/entry_v9/full_2025.parquet \
    --model-out-dir models/entry_v10
echo ">>> STEP 2/7 DONE"
echo ""

# Step 3: Run XGBoost inference
echo ">>> STEP 3/7: Running XGBoost inference over historical data"
python -m gx1.rl.entry_v10.run_xgb_inference_v10 \
    --input-parquet data/entry_v9/full_2025.parquet \
    --output-parquet data/entry_v10/xgb_annotated_2025.parquet \
    --xgb-model-dir models/entry_v10 \
    --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json
echo ">>> STEP 3/7 DONE"
echo ""

# Step 4: Build V10.1 dataset with seq_len=90
echo "=========================================="
echo ">>> STEP 4/7: Building V10.1 dataset (seq_len=90)"
echo "=========================================="
echo "Configuration:"
echo "  - Input: data/entry_v10/xgb_annotated_2025.parquet"
echo "  - Output: data/entry_v10/entry_v10_1_dataset_seq90.parquet"
echo "  - Sequence length: 90 bars (~7.5 hours on M5)"
echo "  - No dummy padding: Rows with insufficient history are dropped"
echo ""
python -m gx1.rl.entry_v10.build_entry_v10_dataset \
    --input-parquet data/entry_v10/xgb_annotated_2025.parquet \
    --output-parquet data/entry_v10/entry_v10_1_dataset_seq90.parquet \
    --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --lookback 90
echo ""
echo ">>> STEP 4/7 DONE: V10.1 dataset built (seq_len=90)"
echo ""

# Step 5: Split dataset and train V10.1 Transformer
echo "=========================================="
echo ">>> STEP 5/7: Splitting dataset and training V10.1 Transformer"
echo "=========================================="
echo "Configuration:"
echo "  - Variant: v10_1 (deeper model)"
echo "  - Sequence length: 90"
echo "  - Architecture: num_layers=6, d_model=256, dim_feedforward=1024"
echo ""
echo "[5a] Splitting dataset (train/val split)..."
python -c "
from gx1.rl.entry_v10.dataset_v10 import train_val_split
from pathlib import Path

train_path, val_path = train_val_split(
    Path('data/entry_v10/entry_v10_1_dataset_seq90.parquet'),
    val_frac=0.2,
    by_date=True,
    output_dir=Path('data/entry_v10')
)
print(f'✅ Train split: {train_path}')
print(f'✅ Val split: {val_path}')
"

echo ""
echo "[5b] Training V10.1 Transformer (variant=v10_1)..."
python -m gx1.rl.entry_v10.train_entry_transformer_v10 \
    --train-parquet data/entry_v10/entry_v10_1_dataset_seq90_train.parquet \
    --val-parquet data/entry_v10/entry_v10_1_dataset_seq90_val.parquet \
    --model-out models/entry_v10/entry_v10_1_transformer.pt \
    --meta-out models/entry_v10/entry_v10_1_transformer_meta.json \
    --epochs 30 \
    --early-stopping-patience 7 \
    --batch-size 256 \
    --lr 1e-4 \
    --device cpu \
    --seq-len 90 \
    --variant v10_1
echo ""
echo ">>> STEP 5/7 DONE: V10.1 Transformer trained (variant=v10_1)"
echo ""

# Step 6: Evaluate V10.1 (including per-session/per-regime metrics)
echo "=========================================="
echo ">>> STEP 6/7: Evaluating V10.1 (variant=v10_1, seq_len=90)"
echo "=========================================="
echo "Configuration:"
echo "  - Model: models/entry_v10/entry_v10_1_transformer.pt"
echo "  - Variant: v10_1"
echo "  - Sequence length: 90"
echo "  - Output report: reports/rl/entry_v10/ENTRY_V10_1_EVAL_2025.md"
echo "  - Includes: Global metrics, session breakdown, regime breakdown"
echo ""
python -m gx1.rl.entry_v10.evaluate_entry_v10_offline \
    --test-parquet data/entry_v10/entry_v10_1_dataset_seq90_val.parquet \
    --v10-model models/entry_v10/entry_v10_1_transformer.pt \
    --v10-meta models/entry_v10/entry_v10_1_transformer_meta.json \
    --output-report reports/rl/entry_v10/ENTRY_V10_1_EVAL_2025.md \
    --device cpu \
    --seq-len 90 \
    --session-breakdown \
    --regime-breakdown
echo ""
echo ">>> STEP 6/7 DONE: V10.1 evaluation complete"
echo ""

# Step 7: Run threshold/label quality analysis
echo ">>> STEP 7/7: Running threshold/label quality analysis"
echo "Note: This step requires replay trade log for full analysis."
echo "      If trade log is not available, threshold computation will still run."
echo ""

# Compute thresholds (always run)
python -m gx1.rl.entry_v10.thresholds_v10 \
    --dataset data/entry_v10/entry_v10_1_dataset_seq90.parquet \
    --model models/entry_v10/entry_v10_1_transformer.pt \
    --meta models/entry_v10/entry_v10_1_transformer_meta.json \
    --output-json reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.json \
    --output-md reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.md \
    --per-regime \
    --per-session \
    --device cpu

# Label quality analysis (requires trade log - skip if not available)
if [ -f "runs/replay_shadow/SNIPER_P4_1_V10_HYBRID/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv" ]; then
    python -m gx1.tools.debug.analyze_entry_v10_label_quality \
        --v10-dataset data/entry_v10/entry_v10_1_dataset_seq90.parquet \
        --v10-model models/entry_v10/entry_v10_1_transformer.pt \
        --v10-meta models/entry_v10/entry_v10_1_transformer_meta.json \
        --replay-trade-log runs/replay_shadow/SNIPER_P4_1_V10_HYBRID/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_LABEL_QUALITY.md
else
    echo "  (Skipping label quality analysis - trade log not found)"
    echo "  To run this step, ensure replay trade log exists at:"
    echo "  runs/replay_shadow/SNIPER_P4_1_V10_HYBRID/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
fi

echo ">>> STEP 7/7 DONE"
echo ""

echo "=========================================="
echo "ENTRY_V10.1 Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated artifacts:"
echo "  - V9 dataset: data/entry_v9/full_2025.parquet"
echo "  - XGBoost models: models/entry_v10/xgb_entry_*_v10.joblib"
echo "  - XGB annotated: data/entry_v10/xgb_annotated_2025.parquet"
echo "  - V10.1 dataset: data/entry_v10/entry_v10_1_dataset_seq90.parquet"
echo "  - V10.1 Transformer: models/entry_v10/entry_v10_1_transformer.pt"
echo "  - V10.1 Metadata: models/entry_v10/entry_v10_1_transformer_meta.json"
echo "  - Evaluation report: reports/rl/entry_v10/ENTRY_V10_1_EVAL.md"
echo "  - Thresholds (JSON): reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.json"
echo "  - Thresholds (MD): reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.md"
echo ""
echo "V10.1 Configuration:"
echo "  - Sequence length: 90 bars (~7.5 hours on M5)"
echo "  - Transformer: num_layers=6, d_model=256, dim_feedforward=1024"
echo "  - Variant: v10_1"
echo ""

