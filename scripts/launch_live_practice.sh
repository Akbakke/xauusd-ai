#!/usr/bin/env bash
# Launch the full COSTFIX live-practice stack on OANDA practice account.
#
# Idempotent — safe to re-run; will SKIP processes that are already alive
# (verified via PID files + kill -0). Starts missing processes only.
#
# Components:
#   1. v12_oanda_data_collector       — pulls M1 OHLC ticks from OANDA every 60s
#                                        → /home/andre2/GX1_DATA/reports/v12_live_data_strict_m1_v1/xauusd_m1_<DATE>.parquet
#   2. immutable snapshot publication  — the repository successor/admission
#                                        owner is short-lived and invoked
#                                        separately through the control surface
#   3. v12_paper_runner               — exact full-stack model-native XAU
#                                        LONG/SHORT/FLAT Entry plus HOLD/EXIT_NOW
#                                        from the same admitted model bundle
#
# Usage:
#   bash scripts/launch_live_practice.sh           # idempotent, won't restart what's already up
#   bash scripts/launch_live_practice.sh --force   # kill any running, then relaunch all;
#                                                  # parity-bound source bytes are mandatory
#
# Logs land under /tmp/gx1_live_practice/<component>.log
# PID files persist under /home/andre2/GX1_DATA/reports/v12_paper_runs/
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PY="$REPO/.venv/bin/python"
DATA_DIR=/home/andre2/GX1_DATA
PAPER_RUNS=$DATA_DIR/reports/v12_paper_runs
LOG_DIR=/tmp/gx1_live_practice

cd "$REPO"
FORCE=0
case "${1-}" in
    "") ;;
    --force) FORCE=1 ;;
    *) echo "FATAL: unknown argument: ${1}" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "FATAL: unexpected extra arguments" >&2; exit 2; }

# ── MODEL-NATIVE XAU DIRECTION LAUNCH GATE ─────────────────────────────────
# The legacy multi-model Entry chain is RETIRED (bundles physically gone
# 2026-07-07); an entry policy can be served only when the newest XAU direction
# launch state admits the exact hashed model-native bundle served
# by gx1/execution/v12_smart_entry_live (operating point read from the contract —
# ONE truth, no env pins here). Launch is fail-closed on FOUR requirements:
#   1. the immutable newest TRAIN==SERVE parity event PASS for the exact
#      launch-bound model-native bundle and prediction-evidence SHA
#   2. the immutable newest directional live-like pocket event PASS for the
#      same launch-bound bundle and prediction-evidence SHA
#   3. a newest immutable one-time launch approval cross-bound inside the
#      launch state to the complete evidence set and bundle commit
#   4. an exact rule-free operating point: selection_score plus execution
#      max_trades; edge/utility thresholds and session allowlists are rejected
# (This replaces the 20260627 legacy-ack block, which guarded — and referenced —
# the retired chain.)
echo "[preflight] model-native train==serve parity-gate check…"
SMART_GATE_OUTPUT=$(PYTHONPATH=$REPO "$PY" - <<'PYEOF'
# ONE truth: the same assert the runner's own guard calls — launcher and
# runner-direct cannot diverge (gx1/execution/v12_smart_entry_live.py).
from gx1.execution.v12_smart_entry_live import assert_smart_serving_gate
from gx1_guards.artifacts import load_decision_entry
from gx1.models.entry_v10.direction_decision_contract import (
    require_model_direction_operating_point,
)
rep = assert_smart_serving_gate()
print(f"[smart-gate] OK: parity PASS ({rep.get('n_bars')} bars, created {rep.get('created_utc')})")
entry = load_decision_entry("v10_entry")
op = require_model_direction_operating_point(
    entry.get("operating_point"),
    context="live practice launcher v10_entry",
)
print(f"CONTRACT_MAX_TRADES={int(op['max_trades'])}")
PYEOF
) || { echo "FATAL: model-native serve gate BLOCKED — rerun parity and directional pocket audit for the exact contract-bound bundle before relaunch." >&2; exit 2; }
printf '%s\n' "$SMART_GATE_OUTPUT"
MAX_TRADES=$(printf '%s\n' "$SMART_GATE_OUTPUT" | sed -n 's/^CONTRACT_MAX_TRADES=//p')
[[ "$MAX_TRADES" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: exact contract max_trades was not emitted by the smart gate" >&2
    exit 2
}

# No external directory is touched and no credentials are loaded before the
# immutable byte-exact source identity and complete launch authority both pass.
# The source manifest excludes only the two transaction-bound authority JSON
# files that finalization must replace; every other tracked source byte and any
# untracked runtime source/config file remain fail-closed.
mkdir -p "$LOG_DIR" "$PAPER_RUNS/open_trades"
if [[ -f .env ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi
if [[ "${OANDA_ENV:-practice}" != "practice" ]]; then
    echo "FATAL: launch_live_practice.sh requires OANDA_ENV=practice exactly" >&2
    exit 2
fi

# ── Collector poll cadence (live SLA) ───────────────────────────────────────
# Tighten the OANDA M1 poll to 15s so a newly-closed bar reaches disk within ~15s
# (default code constant is 60s). The live source is the systemd unit (drop-in
# gx1-collector.service.d/poll15.conf sets the same value); this export keeps a
# manually launched collector on the same observation cadence.
export GX1_COLLECTOR_POLL_SECONDS=${GX1_COLLECTOR_POLL_SECONDS:-15}

# ── ENTRY operating point (513 genuine signal fields; model-native) ──────────
# Entry has no env-pinned direction operating point. The exact
# model-direction argmax contract is read in-process and rejects edge/utility
# thresholds and session allowlists. Trend/session/structure/liquidity/volatility/
# momentum/price-action/path/utility evidence belongs inside the learned model;
# retired pre-unified pins (GX1_CONVICTION_*, GX1_SIZING_*, GX1_SKIP_ASIA,
# GX1_ENTRY_DIPFIX) are rejected. Current launch requires the separate learned
# sizing contract with fresh OOS, calibration and train==serve proof; historical
# fixed_1x is not a live fallback.

# IN-PROCESS SHADOW: disabled. Challenger comparison is an offline,
# immutable identical-path lifecycle followed by zero-order shadow evidence;
# the launcher owns no alternate model path.

# Live-tail freshness is intentionally not a process-wide preflight.  The
# runner validates the newest immutable PASS admission against the exact pair
# used for inference before every new Entry and again immediately before an
# order. Missing/stale evidence blocks new exposure but does not prevent the
# same admitted model bundle from managing an already-open trade through Exit.

# is_alive <pidfile> → echoes the alive pid or empty
is_alive() {
    local pf=$1
    [[ ! -f "$pf" ]] && return 1
    local pid
    pid=$(cat "$pf" 2>/dev/null || true)
    [[ -z "$pid" ]] && return 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
        return 0
    fi
    return 1
}

# systemd_active <unit> → 0 if the user systemd unit owns the daemon.
# 2026-06-03 audit: the data daemons run under systemd --user (gx1-collector /
# gx1-canonical-incremental). This launcher checks only stale PID files, so without
# this guard it would DOUBLE-SPAWN a second collector+canonical writing the SAME
# canonical_v3 parquet concurrently and unlocked -> torn parquet / silent row-loss.
systemd_active() {
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
        systemctl --user is-active --quiet "$1" 2>/dev/null
}

stop_if_running() {
    local pf=$1
    local pid
    if pid=$(is_alive "$pf"); then
        echo "  stopping $pf (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
}

# 2. Immutable canonical snapshot publisher ---------------------------------
# Publication is snapshot-driven through model-native-live-tail-pair followed
# by model-native-live-tail-admission. It is not a daemon. Missing or expired
# admission is an Entry-level fail-closed condition, not a reason to suppress
# Exit recovery inside an already admitted runner. Obsolete daemon ownership
# is reported here; its output cannot authorize Entry.
CANON_PID_FILE="$PAPER_RUNS/canonical_incremental.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$CANON_PID_FILE"; fi
if systemd_active gx1-canonical-incremental.service; then
    echo "WARN: retired gx1-canonical-incremental.service is active; it cannot authorize new Entry. Stop it and use immutable successor publication." >&2
elif pid=$(is_alive "$CANON_PID_FILE"); then
    echo "WARN: retired canonical_incremental PID $pid is active; it cannot authorize new Entry. Stop it and use immutable successor publication." >&2
else
    echo "[2/3] immutable publisher is snapshot-driven (no daemon expected); the runner revalidates a fresh admission before every new Entry"
fi

# 1. OANDA data collector ---------------------------------------------------
COLL_PID_FILE="$PAPER_RUNS/collector.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$COLL_PID_FILE"; fi
if systemd_active gx1-collector.service; then
    echo "[1/3] oanda_data_collector owned by systemd (gx1-collector.service ACTIVE) — skip (avoid double-spawn / torn parquet)"
elif pid=$(is_alive "$COLL_PID_FILE"); then
    echo "[1/3] oanda_data_collector already RUNNING (PID $pid) — skip"
else
    echo "[1/3] launching v12_oanda_data_collector..."
    nohup "$PY" -m gx1.execution.v12_oanda_data_collector \
        > "$LOG_DIR/oanda_data_collector.log" 2>&1 &
    echo $! > "$COLL_PID_FILE"
    echo "    PID=$(cat $COLL_PID_FILE), log=$LOG_DIR/oanda_data_collector.log"
fi

# Give the collectors a head-start so the runner sees fresh data on first poll.
sleep 3

# 3. Paper runner (exact model direction + model-native spread evidence) ------
RUNNER_PID_FILE="$PAPER_RUNS/paper_runner.pid"
# max_trades came directly from the exact launch authority above. Ambient
# overrides and launcher defaults are forbidden.
SUFFIX=${GX1_PAPER_SUFFIX:-xau_model_direction_argmax_latency90}

# Orphan-reaper (2026-06-13 audit): the spawn gate below only kill -0's the SINGLE pid in the
# pid-file, so every relaunch that found that pid dead spawned a fresh runner ON TOP of still-alive
# orphans from prior runs — 9 concurrent runners accrued, double-journaling under one suffix and
# multi-submitting against one OANDA account (and pre-fix-code orphans = NO-OLD-CODE violation).
# Reap any v12_paper_runner that is NOT the (alive) pid-file pid before deciding to spawn.
_keep=$(is_alive "$RUNNER_PID_FILE" || true)
for _p in $(pgrep -f "gx1.execution.v12_paper_runner" 2>/dev/null || true); do
    if [[ "$_p" != "$_keep" ]]; then
        echo "[3/3] reaping ORPHAN paper_runner PID $_p (not pid-file-tracked '${_keep:-none}') — prevents double-journal / multi-submit / stale code"
        kill "$_p" 2>/dev/null || true
    fi
done

if [[ $FORCE -eq 1 ]]; then stop_if_running "$RUNNER_PID_FILE"; fi
if pid=$(is_alive "$RUNNER_PID_FILE"); then
    echo "[3/3] paper_runner already RUNNING (PID $pid) — skip"
else
    echo "[3/3] launching v12_paper_runner with proof-bound learned sizing --max-trades $MAX_TRADES ..."
    nohup "$PY" -m gx1.execution.v12_paper_runner \
        --max-trades "$MAX_TRADES" \
        --journal-suffix "$SUFFIX" \
        > "$LOG_DIR/paper_runner.log" 2>&1 &
    echo $! > "$RUNNER_PID_FILE"
    echo "    PID=$(cat $RUNNER_PID_FILE), log=$LOG_DIR/paper_runner.log"
fi

echo ""
echo "=== Live practice stack status ==="
for label in "oanda_data_collector:$COLL_PID_FILE" \
             "paper_runner:$RUNNER_PID_FILE"; do
    name=${label%%:*}
    pf=${label##*:}
    if pid=$(is_alive "$pf"); then
        echo "  ✓ $name  PID=$pid  log=$LOG_DIR/${name}.log"
    else
        echo "  ✗ $name  NOT RUNNING (check $LOG_DIR/${name}.log)"
    fi
done

echo ""
echo "Tail live logs:"
echo "  tail -f $LOG_DIR/paper_runner.log              # runner decisions + trades"
echo "  tail -f $LOG_DIR/oanda_data_collector.log      # M1 observation polling"
echo "  canonical pair: repository successor/admission owner; stale/missing authority blocks new Entry"
echo ""
echo "Stop everything:"
echo "  bash scripts/stop_live_practice.sh"
