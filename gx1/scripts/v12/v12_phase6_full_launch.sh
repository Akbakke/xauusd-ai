#!/usr/bin/env bash
# V12 Phase 6 FULL launcher — runs joint validation on full candidate pool.
#
# Used after Phase 5 chainer's smoke-mode (10k cands) is done. Full mode
# evaluates all ~50k candidates → solid V12_OFF vs V12_ON statistics.
#
# Usage:
#   /tmp/v12_phase6_full_launch.sh                                           (auto-detects newest P5)
#   /tmp/v12_phase6_full_launch.sh <P5_LOCK>                                  (explicit)
#   /tmp/v12_phase6_full_launch.sh <P5_LOCK> <V3TRACKED_LOCK> <DECISIONS>     (override all 3)
set -euo pipefail

REPORTS=/home/andre2/GX1_DATA/reports/truth_e2e_sanity
REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/tmp/v12_cascade_logs
mkdir -p "$LOG_DIR"

# EXIT operating-point pin FROM THE CONTRACT (vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707):
# phase6 replays the one-truth exit overlays (strategy_f_decision / let_winners_run_hold / hard-stop
# knobs) whose constants are env-read at import — without this pin a replay silently used the code
# DEFAULTS (hard-stop OFF, LWR OFF) instead of the live policy, so replay evidence could be measured
# on a different exit policy than live. Research ablations: override AFTER this line, deliberately.
EXIT_ENV_PIN=$(bash "$REPO/scripts/gx1_exit_env_pin.sh")
eval "$EXIT_ENV_PIN"
echo "[exit-env pin] $(echo "$EXIT_ENV_PIN" | sed 's/^export //' | tr '\n' ' ')"

# Default: auto-detect newest V12.2 trained Phase 5 LOCK.
# Glob is V12_2_* (NOT V12_*) so we never silently fall back to a V12.1 build
# if a V12.2 dir is removed/touched. V12.1 datasets were retired 2026-05-16.
P5_LOCK="${1:-$(ls -td $REPORTS/BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_*_TRAINED_*_LOCK 2>/dev/null | head -1)}"
V3TRACKED_LOCK="${2:-$(ls -td $REPORTS/BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_*_V3TRACKED_*_LOCK 2>/dev/null | head -1)}"
DECISIONS="${3:-/tmp/v12_entry_iql_decisions.parquet}"

[[ -d "$P5_LOCK" ]] || { echo "ERR: P5_LOCK not found: $P5_LOCK" >&2; exit 1; }
[[ -d "$V3TRACKED_LOCK" ]] || { echo "ERR: V3TRACKED_LOCK not found: $V3TRACKED_LOCK" >&2; exit 1; }
[[ -e "$DECISIONS" ]] || { echo "ERR: decisions parquet missing: $DECISIONS" >&2; exit 1; }

# Verify Phase 5 produced R_V12 checkpoints
N_CKPT=$(find "$P5_LOCK/trained_models_v1" -name "R_V12_FOLD_*.pt" 2>/dev/null | wc -l)
[[ "$N_CKPT" -ge 1 ]] || { echo "ERR: $P5_LOCK has 0 R_V12 checkpoints" >&2; exit 2; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
P6LOCK="${V3TRACKED_LOCK}_VALIDATED_FULL_${TS}_LOCK"

echo "============================================================"
echo "V12 Phase 6 FULL evaluation"
echo "  P5_LOCK         = $P5_LOCK ($N_CKPT R_V12 checkpoints)"
echo "  V3TRACKED_LOCK  = $V3TRACKED_LOCK"
echo "  DECISIONS       = $DECISIONS"
echo "  P6LOCK output   = $P6LOCK"
echo "  log             = $LOG_DIR/phase6_full_${TS}.log"
echo "============================================================"

PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/v12/v12_phase6_joint_validation.py \
    --v3tracked-lock "$V3TRACKED_LOCK" \
    --exit-iql-v5-lock "$P5_LOCK" \
    --entry-iql-decisions "$DECISIONS" \
    --variant R_V12 --fold-id FOLD_1 \
    --out-root "$P6LOCK" \
    --v3-override-threshold 0.95 \
    > "$LOG_DIR/phase6_full_${TS}.log" 2>&1

echo "[FULL $(date -u +%H:%M:%SZ)] Done. Output: $P6LOCK"
echo
echo "=== Phase 6 full results ==="
if [[ -f "$P6LOCK/summary_v1.json" ]]; then
    PYTHONPATH=$REPO python3 -c "
import json
s = json.load(open('$P6LOCK/summary_v1.json'))
for cfg in s.get('configs', []):
    print(f\"\\n[{cfg['config']}]  n_total={cfg['n_total']}  n_take={cfg['n_take']}  n_skip={cfg['n_skip']}\")
    print(f\"  joint_mean_pnl_bps     = {cfg['joint_mean_pnl_bps']:+.2f}\")
    print(f\"  mean_pnl_per_take      = {cfg['mean_pnl_per_take']:+.2f}\")
    print(f\"  win_rate_take          = {cfg['win_rate_take']*100:.1f}%\")
    print(f\"  loss_rate_take         = {cfg['loss_rate_take']*100:.1f}%\")
    print(f\"  exit_iql_active_frac   = {cfg['exit_iql_active_frac']*100:.1f}%\")
    print(f\"  v3_override_frac       = {cfg.get('v3_override_frac', 0)*100:.1f}%\")
    print(f\"  forced_terminal_frac   = {cfg['forced_terminal_frac']*100:.1f}%\")
    print(f\"  mean_bars_held_take    = {cfg['mean_bars_held_take']:.1f}\")
"
fi
