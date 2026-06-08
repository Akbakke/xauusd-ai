#!/usr/bin/env bash
# Launch the full COSTFIX live-practice stack on OANDA practice account.
#
# Idempotent — safe to re-run; will SKIP processes that are already alive
# (verified via PID files + kill -0). Starts missing processes only.
#
# Components:
#   1. v12_oanda_data_collector       — pulls M1 OHLC ticks from OANDA every 60s
#                                        → /home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_<DATE>.parquet
#   2. v12_canonical_incremental loop — appends new M1 → canonical_v3 + BASE34 prebuilts
#                                        (paper runner's PrebuiltStateLoader auto-detects + re-augments)
#   3. v12_paper_runner               — XGB → V10 → Entry-IQL → V3 → Exit-IQL → OANDA orders
#                                        env GX1_PURE_PHASE6=1 = no live-only wrappers (1:1 Phase 6 OOT)
#
# Usage:
#   bash scripts/launch_live_practice.sh           # idempotent, won't restart what's already up
#   bash scripts/launch_live_practice.sh --force   # kill any running, then relaunch all
#
# Logs land under /tmp/gx1_live_practice/<component>.log
# PID files persist under /home/andre2/GX1_DATA/reports/v12_paper_runs/
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
DATA_DIR=/home/andre2/GX1_DATA
PAPER_RUNS=$DATA_DIR/reports/v12_paper_runs
LOG_DIR=/tmp/gx1_live_practice
mkdir -p "$LOG_DIR" "$PAPER_RUNS/open_trades"

cd "$REPO"
# Load OANDA credentials into env (the collectors + runner all need OANDA_API_TOKEN + OANDA_ACCOUNT_ID).
if [[ -f .env ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# ── Regime-flag pins (2026-06-04, audit R2/R3) ──────────────────────────────
# LIVE serves the COSTFIX cement (V10 ctx_cont=105, trend_regime price-basis, EXIT_IO_V7).
# The build/contract defaults were flipped ON (GX1_REGIME_V4 -> ctx_cont 121) for the
# upcoming regime retrain, so live MUST pin them OFF here until the 121-dim regime bundle is
# cemented + promoted — otherwise the V10/V3 loaders fail-closed on the 121-vs-105 dim
# mismatch (v12_v10_live.py) on relaunch. Flip BOTH to 1 in lockstep with promoting the
# regime cement (and the serve mirrors P3-P5). Flags are EXPLICIT here — never rely on the
# code defaults for live (build==serve flag parity).
export GX1_REGIME_V4=0
export GX1_TREND_REGIME_FROM_D1=0

# ── Strategy-F overlay pins (2026-06-04, audit MISS-7) ──────────────────────
# The Exit-IQL Strategy-F overlay constants (v12_exit_iql_live.py:110-123) and the distilled-exit
# swap default to the CEMENTED values in code; pin them EXPLICITLY here so live's deployed policy is
# visible and cannot silently drift from the Phase-6-validated config (Phase-6 OOT == live). Change a
# value ONLY together with a re-validated Phase-6 ablation. These ARE the cemented defaults.
export GX1_STRATEGY_F_ENABLED=1
export GX1_MFE_GIVEBACK_PCT=0.30
export GX1_MFE_GIVEBACK_MIN_MFE_BPS=30.0
export GX1_BREAKEVEN_RATIO=0.30
export GX1_BREAKEVEN_MIN_MFE=10.0
export GX1_STRONG_HOLD_QADV=-200.0
export GX1_HOLD_HORIZON_OVERRUN_MULT=1.5
export GX1_HOLD_HORIZON_MIN_FLOOR_BARS=60
export GX1_USE_DISTILLED_EXIT=0

FORCE=0
[[ "${1-}" == "--force" ]] && FORCE=1

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

# 1. OANDA data collector ---------------------------------------------------
COLL_PID_FILE="$PAPER_RUNS/collector.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$COLL_PID_FILE"; fi
if systemd_active gx1-collector.service; then
    echo "[1/4] oanda_data_collector owned by systemd (gx1-collector.service ACTIVE) — skip (avoid double-spawn / torn parquet)"
elif pid=$(is_alive "$COLL_PID_FILE"); then
    echo "[1/4] oanda_data_collector already RUNNING (PID $pid) — skip"
else
    echo "[1/4] launching v12_oanda_data_collector..."
    nohup python3 -m gx1.execution.v12_oanda_data_collector \
        > "$LOG_DIR/oanda_data_collector.log" 2>&1 &
    echo $! > "$COLL_PID_FILE"
    echo "    PID=$(cat $COLL_PID_FILE), log=$LOG_DIR/oanda_data_collector.log"
fi

# 2. Canonical incremental updater (loop @60s) ------------------------------
CANON_PID_FILE="$PAPER_RUNS/canonical_incremental.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$CANON_PID_FILE"; fi
if systemd_active gx1-canonical-incremental.service; then
    echo "[2/4] canonical_incremental owned by systemd (gx1-canonical-incremental.service ACTIVE) — skip (avoid double-spawn / torn parquet)"
elif pid=$(is_alive "$CANON_PID_FILE"); then
    echo "[2/4] canonical_incremental already RUNNING (PID $pid) — skip"
else
    echo "[2/4] launching v12_canonical_incremental --loop --interval 60..."
    nohup python3 -m gx1.execution.v12_canonical_incremental --loop --interval 60 \
        > "$LOG_DIR/canonical_incremental.log" 2>&1 &
    echo $! > "$CANON_PID_FILE"
    echo "    PID=$(cat $CANON_PID_FILE), log=$LOG_DIR/canonical_incremental.log"
fi

# Give the collectors a head-start so the runner sees fresh data on first poll.
sleep 3

# 3. Paper runner (PURE_PHASE6 = Phase 6 OOT 1:1) ---------------------------
RUNNER_PID_FILE="$PAPER_RUNS/paper_runner.pid"
UNITS=${GX1_PAPER_UNITS:-10}
MAX_TRADES=${GX1_PAPER_MAX_TRADES:-100}
MAX_SPREAD=${GX1_PAPER_MAX_SPREAD_BPS:-9999}
SUFFIX=${GX1_PAPER_SUFFIX:-fase2b_regime_v4_pure_phase6}

if [[ $FORCE -eq 1 ]]; then stop_if_running "$RUNNER_PID_FILE"; fi
if pid=$(is_alive "$RUNNER_PID_FILE"); then
    echo "[3/4] paper_runner already RUNNING (PID $pid) — skip"
else
    echo "[3/4] launching v12_paper_runner --units $UNITS --max-trades $MAX_TRADES ..."
    GX1_PURE_PHASE6=1 GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1 GX1_EXIT_AUGMENT_64=1 \
    nohup python3 -m gx1.execution.v12_paper_runner \
        --units "$UNITS" \
        --max-trades "$MAX_TRADES" \
        --max-spread-bps "$MAX_SPREAD" \
        --journal-suffix "$SUFFIX" \
        > "$LOG_DIR/paper_runner.log" 2>&1 &
    echo $! > "$RUNNER_PID_FILE"
    echo "    PID=$(cat $RUNNER_PID_FILE), log=$LOG_DIR/paper_runner.log"
fi

# 4. Daily counterfactual daemon (auto-runs "skulle/skulle ikke" analysis on
#    finished journals — replays + tags missed_opportunity + false_take per day)
CF_PID_FILE="$PAPER_RUNS/counterfactual_daemon.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$CF_PID_FILE"; fi
if pid=$(is_alive "$CF_PID_FILE"); then
    echo "[4/4] counterfactual_daemon already RUNNING (PID $pid) — skip"
else
    echo "[4/4] launching v12_daily_counterfactual --daemon..."
    nohup bash "$REPO/gx1/execution/v12_daily_counterfactual.sh" --daemon \
        > "$LOG_DIR/counterfactual_daemon.log" 2>&1 &
    echo $! > "$CF_PID_FILE"
    echo "    PID=$(cat $CF_PID_FILE), log=$LOG_DIR/counterfactual_daemon.log"
fi

echo ""
echo "=== Live practice stack status ==="
for label in "oanda_data_collector:$COLL_PID_FILE" \
             "canonical_incremental:$CANON_PID_FILE" \
             "paper_runner:$RUNNER_PID_FILE" \
             "counterfactual_daemon:$CF_PID_FILE"; do
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
echo "  tail -f $LOG_DIR/oanda_data_collector.log      # M1 tick fetch every 60s"
echo "  tail -f $LOG_DIR/canonical_incremental.log     # cv3 advance every 60s"
echo ""
echo "Stop everything:"
echo "  bash scripts/stop_live_practice.sh"
