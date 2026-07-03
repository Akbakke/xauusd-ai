#!/usr/bin/env bash
# ENTRY_V10 Offline Pipeline Runner
#
# Runs the complete ENTRY_V10 offline training pipeline:
# 1. Build V9 entry dataset
# 2. Train XGBoost models
# 3. Run XGBoost inference
# 4. Build V10 dataset
# 5. Train V10 Transformer
# 6. Evaluate V10 vs baseline
#
# Usage: bash scripts/run_entry_v10_offline_pipeline.sh
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
echo "ENTRY_V10 Offline Pipeline"
echo "=========================================="
echo ""

# Step 1: Build V9 entry dataset
echo ">>> STEP 1/6: Building V9 entry dataset"
python -m gx1.rl.entry_v9.build_entry_v9_full_dataset \
    --start-date 2020-01-01 \
    --end-date 2025-12-31 \
    --output-parquet data/entry_v9/full_2020_2025.parquet
echo ">>> STEP 1/6 DONE"
echo ""

# Step 2: Train XGBoost models
echo ">>> STEP 2/6: Training XGBoost snapshot models"
python -m gx1.rl.entry_v10.train_entry_xgb_v10 \
    --dataset data/entry_v9/full_2020_2025.parquet \
    --model-out-dir models/entry_v10
echo ">>> STEP 2/6 DONE"
echo ""

# Step 3: Run XGBoost inference
echo ">>> STEP 3/6: Running XGBoost inference over historical data"
python -m gx1.rl.entry_v10.run_xgb_inference_v10 \
    --input-parquet data/entry_v9/full_2020_2025.parquet \
    --output-parquet data/entry_v10/xgb_annotated.parquet \
    --xgb-model-dir models/entry_v10 \
    --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json
echo ">>> STEP 3/6 DONE"
echo ""

# Step 4: Build V10 dataset
echo ">>> STEP 4/6: Building V10 dataset with sequences"
python -m gx1.rl.entry_v10.build_entry_v10_dataset \
    --input-parquet data/entry_v10/xgb_annotated.parquet \
    --output-parquet data/entry_v10/entry_v10_dataset.parquet \
    --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --lookback 30
echo ">>> STEP 4/6 DONE"
echo ""

# Step 5: Split dataset and train Transformer
echo ">>> STEP 5/6: Splitting dataset and training V10 Transformer"
python -c "
from gx1.rl.entry_v10.dataset_v10 import train_val_split
from pathlib import Path

train_path, val_path = train_val_split(
    Path('data/entry_v10/entry_v10_dataset.parquet'),
    val_frac=0.2,
    by_date=True,
    output_dir=Path('data/entry_v10')
)
print(f'Train: {train_path}')
print(f'Val: {val_path}')
"

python -m gx1.rl.entry_v10.train_entry_transformer_v10 \
    --train-parquet data/entry_v10/entry_v10_train.parquet \
    --val-parquet data/entry_v10/entry_v10_val.parquet \
    --model-out models/entry_v10/entry_v10_transformer.pt \
    --meta-out models/entry_v10/entry_v10_transformer_meta.json \
    --epochs 50 \
    --batch-size 1024 \
    --lr 1e-4 \
    --device cpu \
    --seq-len 30
echo ">>> STEP 5/6 DONE"
echo ""

# Step 6: Evaluate V10
echo ">>> STEP 6/6: Evaluating V10 vs baseline"
python -m gx1.rl.entry_v10.evaluate_entry_v10_offline \
    --test-parquet data/entry_v10/entry_v10_val.parquet \
    --v10-model models/entry_v10/entry_v10_transformer.pt \
    --v10-meta models/entry_v10/entry_v10_transformer_meta.json \
    --output-report reports/rl/entry_v10/ENTRY_V10_VS_V9_EVAL.md \
    --device cpu \
    --seq-len 30
echo ">>> STEP 6/6 DONE"
echo ""

echo "=========================================="
echo "ENTRY_V10 Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated artifacts:"
echo "  - V9 dataset: data/entry_v9/full_2020_2025.parquet"
echo "  - XGBoost models: models/entry_v10/xgb_entry_*_v10.joblib"
echo "  - XGB annotated: data/entry_v10/xgb_annotated.parquet"
echo "  - V10 dataset: data/entry_v10/entry_v10_dataset.parquet"
echo "  - V10 Transformer: models/entry_v10/entry_v10_transformer.pt"
echo "  - Evaluation report: reports/rl/entry_v10/ENTRY_V10_VS_V9_EVAL.md"

