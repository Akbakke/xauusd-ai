#!/usr/bin/env bash
# V12 counterfactual auto-runner — TWO tracks running in the same daemon.
#
# Track A: forward-outcome (existing, hourly, 25h-aged journals)
#   "What would price have done if we had taken?" — produces missed_opportunity
#   reports in counterfactual_reports/.
#
# Track B: variant-shadow (added 2026-05-31, every 10 min, TODAY+yesterday)
#   Reward-family take-rate comparison: "What would LAM30/LAM20/HYBRID have decided
#   on the same polls?" NOTE (2026-06-10): the LIVE entry DECISION policy is the
#   conviction-gate + skip-ASIA serve-overlay on the LAM50-REWARD bundle (NOT raw LAM50
#   argmax) — these shadow variants compare reward-family take-rates, not the live operating point.
#   Produces take-rate-per-variant reports in variant_shadow_reports/.
#   No fwd window needed (variants only choose actions), so we run fresh every
#   cycle as the live journal grows.
#
# Both tracks are idempotent. Track A uses marker files; Track B overwrites
# the per-day report each cycle.
#
# Usage:
#   /home/andre2/src/GX1_ENGINE/gx1/execution/v12_daily_counterfactual.sh
#   nohup .../v12_daily_counterfactual.sh --daemon &
set -euo pipefail

COUNTERFACTUAL_ACK_VAR=GX1_ALLOW_LEGACY_V12_COUNTERFACTUAL_DAEMON
COUNTERFACTUAL_ACK_VALUE=20260627_ALLOW_LEGACY_V12_COUNTERFACTUAL_DAEMON
if [[ "${GX1_ALLOW_LEGACY_V12_COUNTERFACTUAL_DAEMON:-}" != "$COUNTERFACTUAL_ACK_VALUE" ]]; then
    cat >&2 <<EOF
[ABORT] Active Entry next-edge plan blocks legacy V12 counterfactual daemon.
        Current path is Entry foundation seq146 cleanup/audit/smoke-readiness.
        Use:
          scripts/entry_next_edge_control.sh verify
          scripts/entry_next_edge_control.sh selftest
          scripts/entry_next_edge_control.sh foundation-guardrails
          scripts/entry_next_edge_control.sh worktree-hygiene
          scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run
          scripts/entry_next_edge_control.sh materialize-smoke
          scripts/entry_next_edge_control.sh train-readiness
        Override only with:
          $COUNTERFACTUAL_ACK_VAR=$COUNTERFACTUAL_ACK_VALUE bash gx1/execution/v12_daily_counterfactual.sh
EOF
    exit 2
fi

REPO=/home/andre2/src/GX1_ENGINE
PY="${GX1_PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "FATAL: repo Python venv missing: $PY" >&2
    exit 2
fi
PAPER_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs
CF_DIR="$PAPER_DIR/counterfactual_reports"
VARIANT_DIR="$PAPER_DIR/variant_shadow_reports"
MARKER_DIR="$PAPER_DIR/.replayed_markers"
LOG_DIR=/home/andre2/GX1_DATA/reports/v12_live_data/logs
mkdir -p "$CF_DIR" "$VARIANT_DIR" "$MARKER_DIR" "$LOG_DIR"

DAEMON_MODE=false
[[ "${1:-}" == "--daemon" ]] && DAEMON_MODE=true

# Track B (variant shadow) tunables — env-overridable.
# Default "auto" (2026-06-12): enumerate every checkpoint in the ACTIVE bundle
# (contract-resolved) instead of a hardcoded variant list. The old hardcoded
# default (LAM50/30/20/10/HYBRID) silently no-op'ed after the volbal bundle
# flip 2026-06-11 — per_variant={} while the daemon counted success.
SHADOW_VARIANTS="${GX1_SHADOW_VARIANTS:-auto}"
# Optional CANDIDATE bundle dir: when set, Track B scores THAT bundle's
# checkpoints over the live journal (shadow a PENDING candidate before promote).
SHADOW_BUNDLE_DIR="${GX1_SHADOW_BUNDLE_DIR:-}"
SHADOW_LOOKBACK_DAYS="${GX1_SHADOW_LOOKBACK_DAYS:-2}"
SHADOW_INTERVAL_SEC="${GX1_SHADOW_INTERVAL_SEC:-600}"     # 10 min
FORWARD_INTERVAL_SEC="${GX1_FORWARD_INTERVAL_SEC:-3600}"  # 1h

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
        if PYTHONPATH=$REPO "$PY" -u $REPO/gx1/execution/v12_counterfactual_replay.py \
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
                "$PY" -c "
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

run_variant_shadow() {
    # Replay TODAY + last N days through alternate Entry-IQL variants.
    # No fwd window required → can run on a still-growing journal. We just
    # overwrite the day's report each cycle (cheap, fresh, idempotent).
    local TODAY_EPOCH=$(date -u +%s)
    local PROC=0
    for d in $(seq 0 "$SHADOW_LOOKBACK_DAYS"); do
        local DAY_EPOCH=$((TODAY_EPOCH - d * 86400))
        local DATE
        DATE=$(date -u -d "@$DAY_EPOCH" +%Y%m%d)
        # Match every journal for this date (any suffix)
        for journal in "$PAPER_DIR"/v12_paper_journal_${DATE}*.jsonl; do
            [[ -f "$journal" ]] || continue
            local BASE
            BASE=$(basename "$journal" .jsonl)
            if [[ "$BASE" =~ ^v12_paper_journal_([0-9]{8})(_(.+))?$ ]]; then
                local SUFFIX="${BASH_REMATCH[3]}"
            else
                continue
            fi
            local SUFFIX_ARG=""
            [[ -n "$SUFFIX" ]] && SUFFIX_ARG="--journal-suffix $SUFFIX"
            local BUNDLE_ARG=""
            [[ -n "$SHADOW_BUNDLE_DIR" ]] && BUNDLE_ARG="--bundle-dir $SHADOW_BUNDLE_DIR"
            if PYTHONPATH=$REPO "$PY" -u $REPO/gx1/execution/v12_counterfactual_replay.py \
                --journal-date "$DATE" $SUFFIX_ARG \
                --mode variants --variants "$SHADOW_VARIANTS" $BUNDLE_ARG \
                --out-dir "$VARIANT_DIR" \
                > "$LOG_DIR/variant_shadow_${BASE}.log" 2>&1; then
                PROC=$((PROC + 1))
            else
                # fail-loud: a degraded/failed shadow must be visible in the daemon
                # log, not silently absorbed (the 2026-06-11→12 no-op lesson).
                echo "[$(date -u +%H:%M:%SZ)] [variant-shadow] ❌ FAILED/DEGRADED for $BASE — see $LOG_DIR/variant_shadow_${BASE}.log"
            fi
        done
    done
    echo "[$(date -u +%H:%M:%SZ)] [variant-shadow] replayed $PROC journal-day(s); reports in $VARIANT_DIR"
}

if $DAEMON_MODE; then
    echo "Running V12 counterfactual daemon — Track A (fwd-outcome, every ${FORWARD_INTERVAL_SEC}s) + Track B (variant-shadow, every ${SHADOW_INTERVAL_SEC}s)"
    LAST_FORWARD=0
    while true; do
        NOW=$(date -u +%s)
        # Track B: variant shadow — every loop tick
        run_variant_shadow || true
        # Track A: forward-outcome — only when interval elapsed
        if (( NOW - LAST_FORWARD >= FORWARD_INTERVAL_SEC )); then
            run_pending_replays || true
            LAST_FORWARD=$NOW
        fi
        sleep "$SHADOW_INTERVAL_SEC"
    done
else
    run_pending_replays
    run_variant_shadow
fi
