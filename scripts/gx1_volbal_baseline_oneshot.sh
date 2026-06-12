#!/usr/bin/env bash
# ONE-SHOT (2026-06-12): generate the FULL volbal baseline CSV in the ASIA window.
#
# WHY a window: the FULL phase6 gate peaks ~32 GB anon-RSS (OOM-killed pid 208985
# at 10:36Z with the live runner resident). ASIA is the skip-ASIA no-take window,
# so stopping the paper runner here costs zero trades. Daemons (collector,
# canonical_incremental, cf-daemon) keep running — only the runner is paused.
#
# GUARANTEES: the runner relaunch runs via trap on EXIT — gate success, gate
# failure or OOM all end in `launch_live_practice.sh` (idempotent + rule-9
# three-leg preflight). Scheduled via transient systemd timer (fires ONCE).
set -uo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PAPER_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs
GATES_DIR=$PAPER_DIR/nightly_learning/candidate_gates
VOLBAL=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/entry_iql_volbal_20260611
LOG=$PAPER_DIR/logs/volbal_baseline_oneshot.log
exec >> "$LOG" 2>&1

log() { echo "[$(date -u +%FT%TZ)] [baseline-oneshot] $*"; }

relaunch() {
    log "relaunching live stack (trap)…"
    bash "$REPO/scripts/launch_live_practice.sh" || log "RELAUNCH FAILED — manual attention needed"
    log "relaunch done."
}
trap relaunch EXIT

# Already staged (e.g. manual run succeeded earlier)? Then do nothing.
if [[ -s "$GATES_DIR/baselines/volbal_per_candidate_V12_OFF.csv" ]]; then
    log "baseline already staged — no-op."
    trap - EXIT
    exit 0
fi

log "stopping paper runner for the ASIA-window baseline run…"
RPID=$(cat "$PAPER_DIR/paper_runner.pid" 2>/dev/null || true)
if [[ -n "$RPID" ]] && kill -0 "$RPID" 2>/dev/null; then
    kill "$RPID"
    for _ in $(seq 1 30); do kill -0 "$RPID" 2>/dev/null || break; sleep 2; done
    rm -f "$PAPER_DIR/paper_runner.pid"
    log "runner stopped (pid $RPID)."
else
    log "runner not running — proceeding."
fi

AVAIL=$(free -g | awk 'NR==2{print $7}')
log "RAM available: ${AVAIL} GB (need ~34)"
if (( AVAIL < 34 )); then
    log "insufficient RAM even without the runner — aborting; baseline stays pending."
    exit 1
fi

log "running FULL gate on ACTIVE volbal bundle…"
if bash "$REPO/scripts/gx1_candidate_gate.sh" "$VOLBAL"; then
    log "gate run completed (verdict written)."
else
    log "gate exited nonzero (expected for a baseline run if posthoc self-compare fails) — checking for CSV anyway."
fi
CSV=$(find "$GATES_DIR/runs" -path "*entry_iql_volbal*" -name per_candidate_V12_OFF.csv 2>/dev/null | sort | tail -1)
if [[ -f "$CSV" ]]; then
    mkdir -p "$GATES_DIR/baselines"
    cp "$CSV" "$GATES_DIR/baselines/volbal_per_candidate_V12_OFF.csv"
    log "BASELINE STAGED: $(wc -l < "$GATES_DIR/baselines/volbal_per_candidate_V12_OFF.csv") rows → baselines/volbal_per_candidate_V12_OFF.csv"
else
    log "NO CSV produced — baseline still pending; investigate phase6 log under candidate_gates/runs/."
fi
# trap relaunches the runner on exit
