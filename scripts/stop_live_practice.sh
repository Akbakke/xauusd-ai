#!/usr/bin/env bash
# Cleanly stop the COSTFIX live-practice stack (collectors + paper runner).
#
# Use this before code edits that affect the live runtime, before reboots,
# or whenever the stack should be quiet. Pairs with launch_live_practice.sh.
set -euo pipefail

PAPER_RUNS=/home/andre2/GX1_DATA/reports/v12_paper_runs

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
    echo "  reaping orphan paper_runner PID $_p"
    kill "$_p" 2>/dev/null || true
done
stop_pid canonical_incremental   "$PAPER_RUNS/canonical_incremental.pid"
stop_pid oanda_data_collector    "$PAPER_RUNS/collector.pid"
stop_pid counterfactual_daemon   "$PAPER_RUNS/counterfactual_daemon.pid"
echo "done."
