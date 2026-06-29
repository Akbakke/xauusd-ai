#!/usr/bin/env bash
# Cleanly stop the COSTFIX live-practice stack (collectors + paper runner).
#
# Use this before code edits that affect the live runtime, before reboots,
# or whenever the stack should be quiet. Pairs with launch_live_practice.sh.
set -euo pipefail

PAPER_RUNS=/home/andre2/GX1_DATA/reports/v12_paper_runs

is_entry_next_edge_shadow_runner_args() {
    local args=$1
    [[ "$args" == *"gx1.execution.v12_paper_runner"* ]] \
        && [[ "$args" == *"--shadow-only"* ]] \
        && [[ "$args" == *"noxgb_shadow"* ]]
}

is_entry_next_edge_shadow_runner() {
    local pid=$1
    local args
    args=$(ps -p "$pid" -o args= 2>/dev/null || true)
    is_entry_next_edge_shadow_runner_args "$args"
}

note_legacy_entry_next_edge_shadow() {
    local label=$1
    local pid=$2
    if is_entry_next_edge_shadow_runner "$pid"; then
        echo "  - $label  PID=$pid  STOP legacy pre-foundation no-XGB shadow runner"
    fi
}

if [[ "${1:-}" == "--verify-entry-next-edge-guards" ]]; then
    is_entry_next_edge_shadow_runner_args \
        "/home/andre2/src/GX1_ENGINE/.venv/bin/python -m gx1.execution.v12_paper_runner --dry-run --shadow-only --journal-suffix noxgb_shadow" \
        || { echo "FAIL: legacy no-XGB shadow runner was not recognized"; exit 1; }
    if is_entry_next_edge_shadow_runner_args \
        "/home/andre2/src/GX1_ENGINE/.venv/bin/python -m gx1.execution.v12_paper_runner --dry-run --journal-suffix legacy_live"; then
        echo "FAIL: legacy paper runner was incorrectly recognized as Entry next-edge shadow"
        exit 1
    fi
    echo "entry_foundation_stop_guard=PASS"
    exit 0
fi

stop_pid() {
    local label=$1
    local pf=$2
    if [[ ! -f "$pf" ]]; then
        echo "  - $label  (no pid file)"
        return
    fi
    local pid
    pid=$(cat "$pf" 2>/dev/null || true)
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "  - $label  PID=$pid  ALREADY STOPPED"
        rm -f "$pf"
        return
    fi
    note_legacy_entry_next_edge_shadow "$label" "$pid"
    echo "  stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "    SIGTERM didn't take, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
    echo "  ✓ $label stopped"
}

echo "Stopping live-practice stack..."
stop_pid paper_runner            "$PAPER_RUNS/paper_runner.pid"
# Reap ANY orphan paper_runner not tracked by the pid-file (2026-06-13 audit: prior relaunches
# accrued 9 concurrent runners; stop_pid only killed the recorded pid, orphans survived).
for _p in $(pgrep -f "gx1.execution.v12_paper_runner" 2>/dev/null || true); do
    note_legacy_entry_next_edge_shadow "orphan paper_runner" "$_p"
    echo "  reaping orphan paper_runner PID $_p"
    kill "$_p" 2>/dev/null || true
done
stop_pid canonical_incremental   "$PAPER_RUNS/canonical_incremental.pid"
stop_pid oanda_data_collector    "$PAPER_RUNS/collector.pid"
stop_pid counterfactual_daemon   "$PAPER_RUNS/counterfactual_daemon.pid"
echo "done."
