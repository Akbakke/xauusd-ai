#!/usr/bin/env bash
# Rebuild CANDIDATE_FORWARD_OUTCOME_RUN/v1_full in parallel.
#
# Splits the 276 weeks into N groups (default 4), runs each in a separate
# python process writing disjoint per-week parquets to the same out-root,
# then runs a final summary-only pass that consolidates the global manifest.
#
# Usage:
#   run_forward_outcome_rebuild.sh [N_PARALLEL]
#
# Default N_PARALLEL=4. Logs to /tmp/forward_outcome_rebuild_*.log.

set -uo pipefail

N_PAR="${1:-4}"
OUT_ROOT=/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANDIDATE_FORWARD_OUTCOME_RUN/v1_full
WEEKS_FILE=/tmp/v6_weeks_with_candidates.txt
PYTHON=/home/andre2/venvs/gx1/bin/python3
export GX1_DATA=/home/andre2/GX1_DATA
export PYTHONPATH=/home/andre2/src/GX1_ENGINE:${PYTHONPATH:-}

if [[ ! -f "$WEEKS_FILE" ]]; then
    echo "FAIL: $WEEKS_FILE not found — regenerate with the find/awk pipeline" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT/per_week"
echo "[rebuild] out_root: $OUT_ROOT"
echo "[rebuild] N_PAR: $N_PAR"
echo "[rebuild] total weeks: $(wc -l <"$WEEKS_FILE")"

# Build N parallel jobs, each handling weeks where (NR % N) == group_id.
PIDS=()
for g in $(seq 0 $((N_PAR - 1))); do
    LOG=/tmp/forward_outcome_rebuild_g${g}.log
    WEEK_ARGS=$(awk -v g=$g -v n=$N_PAR 'NR % n == g {printf "--week %s ", $0}' "$WEEKS_FILE")
    echo "[rebuild] group $g: launching ($(echo "$WEEK_ARGS" | tr ' ' '\n' | grep -c '^--week') weeks) -> $LOG"
    nohup "$PYTHON" -u -m gx1.scripts.materialize_build_candidate_forward_outcome_dataset_v1 \
        --out-root "$OUT_ROOT" --no-skip-existing $WEEK_ARGS > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[rebuild] waiting on PIDs: ${PIDS[*]}"
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[rebuild] PID $pid exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "[rebuild] FAIL: $FAILED of $N_PAR groups failed — check /tmp/forward_outcome_rebuild_g*.log"
    exit 1
fi

echo "[rebuild] all parallel groups done — running final summary pass"
"$PYTHON" -u -m gx1.scripts.materialize_build_candidate_forward_outcome_dataset_v1 \
    --out-root "$OUT_ROOT" > /tmp/forward_outcome_rebuild_summary.log 2>&1

PARQUET_COUNT=$(find "$OUT_ROOT/per_week" -name "*.parquet" | wc -l)
TOTAL_ROWS=$($PYTHON -c "
import json
m = json.load(open('$OUT_ROOT/manifest_v1.json'))
stats = m.get('per_week_stats_v1', [])
print(sum(s.get('output_rows', 0) for s in stats))
" 2>/dev/null || echo "unknown")
echo "[rebuild] DONE — $PARQUET_COUNT per-week parquets, $TOTAL_ROWS total rows"
echo "[rebuild] manifest: $OUT_ROOT/manifest_v1.json"
