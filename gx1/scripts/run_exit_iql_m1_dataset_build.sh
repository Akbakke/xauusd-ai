#!/usr/bin/env bash
# Build EXIT_IQL_PER_BAR_DATASET_V2_M1 in parallel.
# Same structure as run_forward_outcome_rebuild.sh: split 276 weeks into N
# disjoint groups, run parallel processes writing to the same out-root, then
# a final summary-only pass to consolidate the manifest.

set -uo pipefail

N_PAR="${1:-4}"
OUT_ROOT=/home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_PER_BAR_DATASET_V2_M1
WEEKS_FILE=/tmp/v6_weeks_with_candidates.txt
PYTHON=/home/andre2/venvs/gx1/bin/python3
export GX1_DATA=/home/andre2/GX1_DATA
export PYTHONPATH=/home/andre2/src/GX1_ENGINE:${PYTHONPATH:-}

if [[ ! -f "$WEEKS_FILE" ]]; then
    echo "FAIL: $WEEKS_FILE not found" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT/per_week"
echo "[m1-build] out_root: $OUT_ROOT"
echo "[m1-build] N_PAR: $N_PAR"
echo "[m1-build] total weeks: $(wc -l <"$WEEKS_FILE")"

PIDS=()
for g in $(seq 0 $((N_PAR - 1))); do
    LOG=/tmp/exit_iql_m1_g${g}.log
    WEEK_ARGS=$(awk -v g=$g -v n=$N_PAR 'NR % n == g {printf "--week %s ", $0}' "$WEEKS_FILE")
    echo "[m1-build] group $g: launching ($(echo "$WEEK_ARGS" | tr ' ' '\n' | grep -c '^--week') weeks) -> $LOG"
    nohup "$PYTHON" -u -m gx1.scripts.materialize_build_exit_iql_per_bar_dataset_v2_m1 \
        --out-root "$OUT_ROOT" $WEEK_ARGS > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[m1-build] waiting on PIDs: ${PIDS[*]}"
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[m1-build] PID $pid exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "[m1-build] FAIL: $FAILED of $N_PAR groups failed"
    exit 1
fi

echo "[m1-build] all parallel groups done — running final summary pass"
"$PYTHON" -u -m gx1.scripts.materialize_build_exit_iql_per_bar_dataset_v2_m1 \
    --out-root "$OUT_ROOT" > /tmp/exit_iql_m1_summary.log 2>&1

PARQUET_COUNT=$(find "$OUT_ROOT/per_week" -name "*.parquet" | wc -l)
TOTAL_ROWS=$($PYTHON -c "
import json
m = json.load(open('$OUT_ROOT/manifest_v1.json'))
stats = m.get('per_week_stats_v1', [])
print(sum(s.get('output_rows', 0) for s in stats))
" 2>/dev/null || echo "unknown")
echo "[m1-build] DONE — $PARQUET_COUNT per-week parquets, $TOTAL_ROWS total rows"
echo "[m1-build] manifest: $OUT_ROOT/manifest_v1.json"
