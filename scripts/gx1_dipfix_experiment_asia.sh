#!/usr/bin/env bash
# ONE-SHOT (2026-06-15): DIPFIX serve-time-selection A/B/C gate experiment in the ASIA window.
#
# WHAT: runs scripts/gx1_candidate_gate.sh on the ACTIVE volbal Entry-IQL bundle THREE times —
#   arm A = BASELINE         (GX1_ENTRY_DIPFIX unset = current op; byte-identical to the cement chain)
#   arm B = suppress_short   (GX1_ENTRY_DIPFIX=1 GX1_ENTRY_DIPFIX_MODE=suppress_short)
#   arm C = both             (GX1_ENTRY_DIPFIX=1 GX1_ENTRY_DIPFIX_MODE=both)
# — and stages the 3 per-candidate CSVs side by side under candidate_gates/dipfix_experiment/ so the
# arms can be graded against each other (posthoc skip-ASIA + per-year floor) through the FULL exit+cap-3
# chain, OOT. The DIPFIX overlay is the one-truth gx1.execution.v12_entry_iql_live.apply_dipfix_overlay
# (default-OFF, byte-identical when unset — verified 2026-06-15 on 1416 real candidates), wired into the
# gate inference (v12_phase1_entry_iql_inference) so the gate evaluates EXACTLY what live serve applies.
#
# WHY a window (mirrors scripts/gx1_volbal_baseline_oneshot.sh EXACTLY for safety): the FULL phase6 gate
# peaks ~32-43 GB anon-RSS and OOM-killed the box with the live runner resident (2026-06-12). ASIA is the
# skip-ASIA no-take window, so stopping the paper runner here costs zero trades. The data daemons
# (collector, canonical_incremental, cf-daemon) keep running — only the runner is paused. GX1_REPLAY_WORKERS=4
# caps the COW write-amplification (12 workers peaked 42.8 GB → OOM). The runner relaunch runs via trap on
# EXIT — gate success, gate failure or OOM all end in launch_live_practice.sh (idempotent + rule-9 preflight).
#
# RUN MANUALLY in the ASIA window (DO NOT schedule from here): bash scripts/gx1_dipfix_experiment_asia.sh
set -uo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PY=$REPO/.venv/bin/python
PAPER_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs
GATES_DIR=$PAPER_DIR/nightly_learning/candidate_gates
OUT_DIR=$GATES_DIR/dipfix_experiment
LOG=$PAPER_DIR/logs/dipfix_experiment_asia.log
exec >> "$LOG" 2>&1

log() { echo "[$(date -u +%FT%TZ)] [dipfix-asia] $*"; }

relaunch() {
    log "relaunching live stack (trap)…"
    bash "$REPO/scripts/launch_live_practice.sh" || log "RELAUNCH FAILED — manual attention needed"
    log "relaunch done."
}
trap relaunch EXIT

# Resolve the ACTIVE volbal Entry-IQL bundle via the contract (rule 8 — never a hardcoded vintage).
VOLBAL=$(PYTHONPATH=$REPO $PY -c "from gx1_guards.artifacts import load_decision_entry; print(load_decision_entry('entry_iql')['path'])")
if [[ -z "$VOLBAL" || ! -d "$VOLBAL" ]]; then
    log "FATAL: could not resolve ACTIVE entry_iql bundle via contract ('$VOLBAL') — aborting."
    exit 1
fi
log "ACTIVE entry_iql (volbal) bundle = $VOLBAL"

# Idempotent: if all 3 arm CSVs are already staged, do nothing.
if [[ -s "$OUT_DIR/arm_A_baseline_per_candidate.csv" \
   && -s "$OUT_DIR/arm_B_suppress_short_per_candidate.csv" \
   && -s "$OUT_DIR/arm_C_both_per_candidate.csv" ]]; then
    log "all 3 arm CSVs already staged — no-op."
    trap - EXIT
    exit 0
fi

log "stopping paper runner for the ASIA-window dipfix experiment…"
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
log "RAM available: ${AVAIL} GB (need >=34)"
if (( AVAIL < 34 )); then
    log "insufficient RAM even without the runner — aborting; experiment stays pending."
    exit 1
fi

# 12 replay workers peaked 42.8 GB → OOM on this 47 GB box; 4 workers fits (slower replay leg).
export GX1_REPLAY_WORKERS=4
mkdir -p "$OUT_DIR"

# Each arm = one FULL gate run on the SAME volbal bundle, differing ONLY by the DIPFIX env. The gate
# writes phase6/per_candidate_V12_OFF.csv inside its own timestamped out-dir under candidate_gates/runs/;
# we copy that arm's CSV to a stable arm-named path. The gate's posthoc compares each arm vs the staged
# volbal baseline (baselines/volbal_per_candidate_V12_OFF.csv) — arm A reproduces that baseline (sanity).
run_arm() {
    local arm_name="$1" dest_csv="$2"; shift 2   # remaining args = env assignments for this arm
    if [[ -s "$dest_csv" ]]; then
        log "arm '$arm_name' already staged ($dest_csv) — skipping."
        return 0
    fi
    log "=== ARM '$arm_name' : env [$*] : FULL gate on $VOLBAL (GX1_REPLAY_WORKERS=4) ==="
    if env "$@" bash "$REPO/scripts/gx1_candidate_gate.sh" "$VOLBAL"; then
        log "arm '$arm_name' gate completed (verdict written)."
    else
        log "arm '$arm_name' gate exited nonzero (a self-compare FAIL is expected for the baseline arm) — checking for CSV anyway."
    fi
    # Newest per_candidate_V12_OFF.csv for THIS volbal bundle under the gate's runs/ tree.
    local csv
    csv=$(find "$GATES_DIR/runs" -path "*$(basename "$VOLBAL")*" -name per_candidate_V12_OFF.csv -newermt "-2 hours" 2>/dev/null | sort | tail -1)
    if [[ -f "$csv" ]]; then
        cp "$csv" "$dest_csv"
        log "ARM '$arm_name' STAGED: $(wc -l < "$dest_csv") rows → $dest_csv (src: $csv)"
    else
        log "arm '$arm_name': NO CSV produced — investigate phase6 log under candidate_gates/runs/."
    fi
}

# Arm A — BASELINE: DIPFIX explicitly unset (current op). GX1_ENTRY_DIPFIX is read at import in both the
# live wrapper and the gate inference, so NOT exporting it = the default-OFF byte-identical path.
run_arm "A_baseline"       "$OUT_DIR/arm_A_baseline_per_candidate.csv"
# Arm B — suppress_short.
run_arm "B_suppress_short" "$OUT_DIR/arm_B_suppress_short_per_candidate.csv" GX1_ENTRY_DIPFIX=1 GX1_ENTRY_DIPFIX_MODE=suppress_short
# Arm C — both (suppress_short + high-p_long SKIP→LONG revival).
run_arm "C_both"           "$OUT_DIR/arm_C_both_per_candidate.csv"           GX1_ENTRY_DIPFIX=1 GX1_ENTRY_DIPFIX_MODE=both

log "DIPFIX experiment done. Staged arms under $OUT_DIR/ — grade A vs B vs C (posthoc skip-ASIA + per-year)."
# trap relaunches the runner on exit
