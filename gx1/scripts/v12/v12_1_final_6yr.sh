#!/usr/bin/env bash
# V12.1 FINAL: train best grid-config on full 6-yr data + Phase 6 eval.
#
# Usage:
#   /tmp/v12_1_final_6yr.sh DEFAULT
#   /tmp/v12_1_final_6yr.sh AGGRESSIVE
#   /tmp/v12_1_final_6yr.sh NO_GIVEBACK
set -euo pipefail

LABEL="${1:?Usage: $0 <CONFIG_LABEL>  (one of: DEFAULT, NO_GIVEBACK, NO_TRAIL, AGGRESSIVE, NO_MFE_FLOOR)}"

# Config table — must match v12_1_grid_search.sh
case "$LABEL" in
    DEFAULT)      CAP=0.05; MFE=0.5; GBF=0.7; GBP=0.5; TCF=0.8; TBF=0.10 ;;
    NO_GIVEBACK)  CAP=0.05; MFE=0.5; GBF=0.0; GBP=0.0; TCF=0.8; TBF=0.10 ;;
    NO_TRAIL)     CAP=0.05; MFE=0.5; GBF=0.7; GBP=0.5; TCF=0.8; TBF=0.0  ;;
    AGGRESSIVE)   CAP=0.02; MFE=0.8; GBF=0.7; GBP=0.5; TCF=0.8; TBF=0.10 ;;
    NO_MFE_FLOOR) CAP=0.05; MFE=0.0; GBF=0.7; GBP=0.5; TCF=0.8; TBF=0.10 ;;
    *) echo "Unknown LABEL: $LABEL" >&2; exit 1 ;;
esac

REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/tmp/v12_cascade_logs
V3T=/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_EXIT_IQL_PER_BAR_DATASET_V12_20260508T093249Z_LOCK_V3TRACKED_20260508T225545Z_LOCK
ENTRY_DEC=/tmp/v12_entry_iql_decisions.parquet

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOCK="${V3T}_FINAL_${LABEL}_${TS}_LOCK"

echo "============================================================"
echo "V12.1 FINAL — $LABEL on full 6-yr data"
echo "  cap=$CAP mfe=$MFE gbf=$GBF gbp=$GBP tcf=$TCF tbf=$TBF"
echo "  out: $LOCK"
echo "============================================================"

PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/materialize_build_exit_iql_v5_v12_1_m1.py \
    --per-bar-dir "$V3T" --out-root "$LOCK" \
    --budget medium --sample-n-rows 2000000 --variants R_V12_1 \
    --capital-cost-bps $CAP --mfe-floor-weight $MFE \
    --giveback-frac $GBF --giveback-penalty $GBP \
    --trail-capture-frac $TCF --trail-bonus-frac $TBF \
    > "$LOG_DIR/v12_1_final_${LABEL}_${TS}.log" 2>&1

N_CKPT=$(find "$LOCK/trained_models_v1" -name "R_V12_1_FOLD_*.pt" 2>/dev/null | wc -l)
echo "  ckpts: $N_CKPT"
[[ "$N_CKPT" -lt 1 ]] && { echo "  Training failed" >&2; exit 2; }

TS6=$(date -u +%Y%m%dT%H%M%SZ)
P6LOCK="${LOCK}_VALIDATED_${TS6}_LOCK"
echo "Phase 6 eval (full 6-yr, V12_OFF + V12_ON, 2k cands)"
PYTHONPATH=$REPO python3 -u /tmp/v12_phase6_joint_validation.py \
    --v3tracked-lock "$V3T" --exit-iql-v5-lock "$LOCK" \
    --entry-iql-decisions "$ENTRY_DEC" \
    --variant R_V12_1 --fold-id FOLD_1 \
    --out-root "$P6LOCK" \
    --max-candidates 2000 --v3-override-threshold 0.95 \
    > "$LOG_DIR/v12_1_final_eval_${LABEL}_${TS6}.log" 2>&1

PYTHONPATH=$REPO python3 -c "
import json
s = json.load(open('$P6LOCK/summary_v1.json'))
print('=== V12.1 FINAL [$LABEL] (6-yr) ===')
for cfg in s['configs']:
    print(f\"\\n[{cfg['config']}]  pnl={cfg['joint_mean_pnl_bps']:+.2f}  wr={cfg['win_rate_take']*100:.1f}%  iql_active={cfg['exit_iql_active_frac']*100:.1f}%  bars_held={cfg['mean_bars_held_take']:.0f}\")
"
