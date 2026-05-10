#!/usr/bin/env bash
# V12 daily counterfactual auto-runner.
#
# Finds all paper-runner journals older than 25 hours and not yet replayed;
# runs counterfactual analysis on each, writes report + flags high-value
# missed opportunities.
#
# Designed to run as cron job (daily) or background daemon. Idempotent —
# replays already-processed journals are skipped via marker files.
#
# Usage:
#   /home/andre2/src/GX1_ENGINE/gx1/execution/v12_daily_counterfactual.sh
#
# Cron setup (daily at 08:00 UTC):
#   0 8 * * * /home/andre2/src/GX1_ENGINE/gx1/execution/v12_daily_counterfactual.sh
#
# Or run as continuous daemon (loops every hour):
#   nohup .../v12_daily_counterfactual.sh --daemon &
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PAPER_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs
CF_DIR="$PAPER_DIR/counterfactual_reports"
MARKER_DIR="$PAPER_DIR/.replayed_markers"
LOG_DIR=/home/andre2/GX1_DATA/reports/v12_live_data/logs
mkdir -p "$CF_DIR" "$MARKER_DIR" "$LOG_DIR"

DAEMON_MODE=false
[[ "${1:-}" == "--daemon" ]] && DAEMON_MODE=true

run_pending_replays() {
    NOW_EPOCH=$(date -u +%s)
    AGE_THRESHOLD_SEC=$((25 * 3600))   # 25h — slightly past K=1440 (24h)
    PENDING_COUNT=0
    PROCESSED_COUNT=0

    for journal in "$PAPER_DIR"/v12_paper_journal_*.jsonl; do
        [[ -f "$journal" ]] || continue
        BASE=$(basename "$journal" .jsonl)
        # Parse YYYYMMDD + optional suffix
        if [[ "$BASE" =~ ^v12_paper_journal_([0-9]{8})(_(.+))?$ ]]; then
            DATE="${BASH_REMATCH[1]}"
            SUFFIX="${BASH_REMATCH[3]}"
        else
            continue
        fi
        MARKER="$MARKER_DIR/$BASE.replayed"
        if [[ -f "$MARKER" ]]; then
            continue   # already processed
        fi
        # Check age — last modification of journal must be > 25h ago
        # (so all K=1440 forward-bars are available)
        FILE_MTIME=$(stat -c %Y "$journal")
        AGE=$((NOW_EPOCH - FILE_MTIME))
        if [[ "$AGE" -lt "$AGE_THRESHOLD_SEC" ]]; then
            PENDING_COUNT=$((PENDING_COUNT + 1))
            continue   # too fresh — wait
        fi
        # Run replay
        echo "[$(date -u +%H:%M:%SZ)] replaying $BASE (age $((AGE/3600))h)"
        SUFFIX_ARG=""
        [[ -n "$SUFFIX" ]] && SUFFIX_ARG="--journal-suffix $SUFFIX"
        if PYTHONPATH=$REPO python3 -u $REPO/gx1/execution/v12_counterfactual_replay.py \
            --journal-date "$DATE" $SUFFIX_ARG \
            --out-dir "$CF_DIR" \
            > "$LOG_DIR/cf_replay_${BASE}.log" 2>&1; then
            touch "$MARKER"
            PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
            echo "  ✅ done. report: $CF_DIR/counterfactual_summary_${DATE}${SUFFIX:+_$SUFFIX}.json"
            # Print summary inline
            SUMMARY="$CF_DIR/counterfactual_summary_${DATE}${SUFFIX:+_$SUFFIX}.json"
            if [[ -f "$SUMMARY" ]]; then
                echo "  Summary:"
                python3 -c "
import json
s = json.load(open('$SUMMARY'))
print(f\"    total_events={s.get('total_events',0)}\")
print(f\"    decisions={s.get('decisions',{})}\")
print(f\"    missed_opportunities={s.get('missed_opportunity_count',0)}  (mean={s.get('missed_opportunity_mean_bps',0):.1f} bps)\")
print(f\"    high_value_50plus={s.get('high_value_missed_50plus',0)}  100plus={s.get('high_value_missed_100plus',0)}\")
"
            fi
        else
            echo "  ❌ replay FAILED — see $LOG_DIR/cf_replay_${BASE}.log"
        fi
    done
    echo "[$(date -u +%H:%M:%SZ)] processed=$PROCESSED_COUNT pending=$PENDING_COUNT"
}

if $DAEMON_MODE; then
    echo "Running V12 counterfactual daemon (loops every hour)"
    while true; do
        run_pending_replays
        sleep 3600
    done
else
    run_pending_replays
fi
