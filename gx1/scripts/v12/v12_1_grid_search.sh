#!/usr/bin/env bash
# V12.1 grid search on 6-yr data — 4 configs (DEFAULT excluded, already have 6-yr result).
# Each config: train ~50 min + eval ~5 min = ~55 min. Total: ~3.5 hours.
#
# Reads V12 V3TRACKED (full 6-yr), trains R_V12_1 with config-specific reward params.
set +e   # don't abort on single config error — keep grid going

REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/tmp/v12_cascade_logs/v12_1_grid
mkdir -p "$LOG_DIR"
V3T=/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_EXIT_IQL_PER_BAR_DATASET_V12_20260508T093249Z_LOCK_V3TRACKED_20260508T225545Z_LOCK
ENTRY_DEC=/tmp/v12_entry_iql_decisions.parquet
SAMPLE_N=2000000

# Configs (DEFAULT skipped — already trained as V12.1 baseline)
CONFIGS=(
  "NO_GIVEBACK   0.05  0.5  0.0  0.0  0.8  0.10"
  "NO_TRAIL      0.05  0.5  0.7  0.5  0.8  0.0"
  "AGGRESSIVE    0.02  0.8  0.7  0.5  0.8  0.10"
  "NO_MFE_FLOOR  0.05  0.0  0.7  0.5  0.8  0.10"
)

echo "============================================================"
echo "V12.1 GRID SEARCH on 6-yr data — 4 ablation configs"
echo "  Sample: $SAMPLE_N rows per config (~50 min train + ~5 min eval)"
echo "  DEFAULT not re-run (have +50.93 / 90.2% wr from earlier 6-yr run)"
echo "============================================================"

for line in "${CONFIGS[@]}"; do
    set -- $line
    LABEL=$1; CAP=$2; MFE=$3; GBF=$4; GBP=$5; TCF=$6; TBF=$7
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    LOCK="${V3T}_GRID_6YR_${LABEL}_${TS}_LOCK"
    echo
    echo "[GRID $(date -u +%H:%M:%SZ)] [$LABEL] cap=$CAP mfe=$MFE gbf=$GBF gbp=$GBP tcf=$TCF tbf=$TBF"

    PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/materialize_build_exit_iql_v5_v12_1_m1.py \
        --per-bar-dir "$V3T" --out-root "$LOCK" \
        --budget medium --sample-n-rows $SAMPLE_N --variants R_V12_1 \
        --capital-cost-bps $CAP --mfe-floor-weight $MFE \
        --giveback-frac $GBF --giveback-penalty $GBP \
        --trail-capture-frac $TCF --trail-bonus-frac $TBF \
        > "$LOG_DIR/train_${LABEL}_${TS}.log" 2>&1
    TRAIN_RC=$?

    N_CKPT=$(find "$LOCK/trained_models_v1" -name "R_V12_1_FOLD_*.pt" 2>/dev/null | wc -l)
    echo "  ckpts: $N_CKPT  rc=$TRAIN_RC"
    if [[ "$N_CKPT" -lt 1 ]]; then
        echo "  ❌ training failed for $LABEL"
        continue
    fi

    TS6=$(date -u +%Y%m%dT%H%M%SZ)
    P6LOCK="${LOCK}_VALIDATED_${TS6}_LOCK"
    echo "[GRID $(date -u +%H:%M:%SZ)] [$LABEL] Phase 6 eval"
    PYTHONPATH=$REPO python3 -u /tmp/v12_phase6_joint_validation.py \
        --v3tracked-lock "$V3T" --exit-iql-v5-lock "$LOCK" \
        --entry-iql-decisions "$ENTRY_DEC" \
        --variant R_V12_1 --fold-id FOLD_1 \
        --out-root "$P6LOCK" \
        --max-candidates 2000 --v3-override-threshold 0.95 \
        --skip-v12-on \
        > "$LOG_DIR/eval_${LABEL}_${TS6}.log" 2>&1
    EVAL_RC=$?

    if [[ -f "$P6LOCK/summary_v1.json" ]]; then
        PYTHONPATH=$REPO python3 -c "
import json
s = json.load(open('$P6LOCK/summary_v1.json'))
cfg = s['configs'][0]
label = '$LABEL'
print('  RESULT [' + label + ']: pnl=' + str(round(cfg['joint_mean_pnl_bps'], 2)) + '  wr=' + str(round(cfg['win_rate_take']*100, 1)) + '%  iql_active=' + str(round(cfg['exit_iql_active_frac']*100, 1)) + '%  bars_held=' + str(round(cfg['mean_bars_held_take'])) + '  forced=' + str(round(cfg['forced_terminal_frac']*100, 1)) + '%  median=' + str(round(cfg.get('median_pnl_take', 0), 1)))
" 2>&1
    else
        echo "  ❌ Phase 6 produced no summary (rc=$EVAL_RC)"
    fi
done

echo
echo "============================================================"
echo "V12.1 GRID DONE — $(date -u +%H:%M:%SZ)"
echo "============================================================"
echo "All RESULT lines (DEFAULT from earlier: pnl=+50.93 wr=90.2% iql=93.8%):"
grep "RESULT" "$LOG_DIR"/../grid_search_v2.log 2>/dev/null
echo
echo "Pick best label, then run /tmp/v12_1_final_6yr.sh <BEST>"
