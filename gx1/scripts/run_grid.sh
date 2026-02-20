#!/bin/bash
# V2 Grid Search Script
# Runs train + eval for all (T, M) combinations

set -e

cd /home/andre2/src/GX1_ENGINE
source ~/venvs/gx1/bin/activate
export GX1_DATA_ROOT=/home/andre2/GX1_DATA

YEARS="2020 2021 2022 2023 2024 2025"
SESSIONS="EU US OVERLAP"
SEED=42
REF_YEAR=2025

T_VALUES=(0.3 0.4 0.5 0.6)
M_VALUES=(0.10 0.25 0.40 0.55)

GRID_DIR=$GX1_DATA_ROOT/models/xgb_drift_test/v2_grid

echo "============================================================"
echo "V2 GRID SEARCH"
echo "============================================================"
echo "T values: ${T_VALUES[@]}"
echo "M values: ${M_VALUES[@]}"
echo ""

for T in "${T_VALUES[@]}"; do
    for M in "${M_VALUES[@]}"; do
        CONFIG="T${T}_M${M}"
        MODEL_DIR="${GRID_DIR}/${CONFIG}"
        
        echo ""
        echo "========================================"
        echo "Config: $CONFIG"
        echo "========================================"
        
        # Train
        echo "Training..."
        /home/andre2/venvs/gx1/bin/python gx1/scripts/train_xgb_universal_multihead_v2.py \
            --years $YEARS \
            --sessions $SESSIONS \
            --label-version v2 \
            --threshold-v2 $T \
            --min-margin-v2 $M \
            --seed $SEED \
            --output-dir $MODEL_DIR 2>&1 | grep -E "(LogLoss|Label distribution|LONG:|SHORT:|FLAT:|ERROR|TRAINING COMPLETE)" || true
        
        # Eval
        MODEL_PATH="${MODEL_DIR}/xgb_universal_multihead_v2.joblib"
        EVAL_DIR="${MODEL_DIR}/eval"
        
        if [ -f "$MODEL_PATH" ]; then
            echo "Evaluating..."
            /home/andre2/venvs/gx1/bin/python gx1/scripts/eval_xgb_multihead_v2_multiyear.py \
                --years $YEARS \
                --reference-year $REF_YEAR \
                --model-path $MODEL_PATH \
                --output-dir $EVAL_DIR 2>&1 | grep -E "(Max KS|Max PSI|VERDICT|by head)" || true
        else
            echo "ERROR: Model not found at $MODEL_PATH"
        fi
    done
done

echo ""
echo "============================================================"
echo "GRID SEARCH COMPLETE"
echo "============================================================"

# Collect results
echo ""
echo "Summary of all configs:"
echo "Config          MaxKS     MaxPSI    US_KS     Verdict"
echo "------------------------------------------------------"

for T in "${T_VALUES[@]}"; do
    for M in "${M_VALUES[@]}"; do
        CONFIG="T${T}_M${M}"
        SUMMARY="${GRID_DIR}/${CONFIG}/eval/EVAL_SUMMARY.json"
        
        if [ -f "$SUMMARY" ]; then
            MAX_KS=$(/home/andre2/venvs/gx1/bin/python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('max_ks', 999):.4f}\")")
            MAX_PSI=$(/home/andre2/venvs/gx1/bin/python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('max_psi', 999):.4f}\")")
            US_KS=$(/home/andre2/venvs/gx1/bin/python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('max_ks_by_head', {}).get('US', 999):.4f}\")")
            VERDICT=$(/home/andre2/venvs/gx1/bin/python -c "import json; d=json.load(open('$SUMMARY')); print(d.get('verdict', 'N/A'))")
            printf "%-15s %-9s %-9s %-9s %-10s\n" "$CONFIG" "$MAX_KS" "$MAX_PSI" "$US_KS" "$VERDICT"
        else
            printf "%-15s %-9s %-9s %-9s %-10s\n" "$CONFIG" "N/A" "N/A" "N/A" "ERROR"
        fi
    done
done
