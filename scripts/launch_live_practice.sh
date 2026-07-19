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
#   3. v12_paper_runner               — exact 513-signal model-native XAU entry
#                                        → LONG/SHORT/FLAT model argmax; the separate
#                                        Exit stack owns exits; immutable learned sizing
#                                        and execution safety cannot rewrite direction
#
# Usage:
#   bash scripts/launch_live_practice.sh           # idempotent, won't restart what's already up
#   bash scripts/launch_live_practice.sh --force   # kill any running, then relaunch all;
#                                                  # a clean worktree is still mandatory
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

# ── Rule 2: git-clean before the live launch (2026-06-13 audit gap) ──────────
# Must run before any Python preflight, so a dirty tree cannot execute live code
# and then abort later.
if [[ -n "$(git -C "$REPO" status --short)" ]]; then
    echo "FATAL: git tree is dirty — live source identity must be clean and cannot be overridden:"
    git -C "$REPO" status --short
    exit 1
fi

# No external directory is touched and no credentials are loaded before the
# exact source-identity gate above passes.
mkdir -p "$LOG_DIR" "$PAPER_RUNS/open_trades"
if [[ -f .env ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# ── MODEL-NATIVE XAU DIRECTION LAUNCH GATE ─────────────────────────────────
# The legacy XGB->V10->Entry-IQL entry chain is RETIRED (bundles physically gone
# 2026-07-07); an entry policy can be served only when the newest XAU direction
# launch state admits the exact hashed model-native bundle served
# by gx1/execution/v12_smart_entry_live (operating point read from the contract —
# ONE truth, no env pins here). Launch is fail-closed on FOUR requirements:
#   1. the immutable newest TRAIN==SERVE parity event PASS for the exact
#      launch-bound model-native bundle and prediction-evidence SHA
#   2. the immutable newest directional live-like pocket event PASS for the
#      same launch-bound bundle and prediction-evidence SHA
#   3. an explicit user LAUNCH VEDTAK id in GX1_SMART_LAUNCH_VEDTAK
#   4. an exact rule-free operating point: selection_score plus execution
#      max_trades; edge/utility thresholds and session allowlists are rejected
# (This replaces the 20260627 legacy-ack block, which guarded — and referenced —
# the retired chain.)
if [[ -z "${GX1_SMART_LAUNCH_VEDTAK:-}" ]]; then
    echo "[ABORT] smart-serving launch requires an explicit user vedtak:" >&2
    echo "        GX1_SMART_LAUNCH_VEDTAK=<vedtak-id> bash scripts/launch_live_practice.sh" >&2
    echo "        (demo/paper launch opens only after parity-gate PASS + preflight + vedtak)" >&2
    exit 2
fi
echo "[preflight] model-native train==serve parity-gate check…"
PYTHONPATH=$REPO "$PY" - <<'PYEOF' || { echo "FATAL: model-native serve gate BLOCKED — rerun parity and directional pocket audit for the exact contract-bound bundle before relaunch." >&2; exit 2; }
# ONE truth: the same assert the runner's own guard calls — launcher and
# runner-direct cannot diverge (gx1/execution/v12_smart_entry_live.py).
from gx1.execution.v12_smart_entry_live import assert_smart_serving_gate
rep = assert_smart_serving_gate()
print(f"[smart-gate] OK: parity PASS ({rep.get('n_bars')} bars, created {rep.get('created_utc')})")
PYEOF

# ── Collector poll cadence (live SLA) ───────────────────────────────────────
# Tighten the OANDA M1 poll to 15s so a newly-closed bar reaches disk within ~15s
# (default code constant is 60s). The live source is the systemd unit (drop-in
# gx1-collector.service.d/poll15.conf sets the same value); this export only keeps
# the nohup FALLBACK collector in parity if systemd is ever not owning it.
export GX1_COLLECTOR_POLL_SECONDS=${GX1_COLLECTOR_POLL_SECONDS:-15}

# ── Regime-flag pins (historical fase2b-era pin; Entry surface superseded) ──────────────
# NOTE: the ctx_cont=123 V10 entry surface from the original 2026-06 pin is RETIRED —
# the active model-native Entry contract is 142 continuous + 5 categorical and Entry
# launch is BLOCK, so this script cannot open Entry until a bundle is admitted. The
# pins stay explicit for the separately retained Exit chain (EXIT_IO_V8=173) — never
# rely on code defaults for live (build==serve flag parity).
export GX1_REGIME_V4=1
export GX1_TREND_REGIME_FROM_D1=1

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
export GX1_STRONG_HOLD_QADV=-66.5
# Deferral-Exit-IQL serving knob (vedtak EXIT_IQL_DEFERRAL_PROMOTION_20260707):
# the Q-net may DEFER a Strategy-F exit at most this many M1 bars; the -80 hard
# stop and giveback hard-release remain rule-based floors (unlearnable).
export GX1_STRATEGY_F_DEFER_CAP_BARS=240
export GX1_HOLD_HORIZON_OVERRUN_MULT=1.5
export GX1_HOLD_HORIZON_MIN_FLOOR_BARS=60
export GX1_USE_DISTILLED_EXIT=0
# AUG64 exit features (EX2/EX3 2026-06-04 cement). Was inline-only on the paper-runner line;
# promoted to a top-level export 2026-07-07 (vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707)
# so the contract launch-assert below can verify it mechanically. Value UNCHANGED (=1 since cement).
export GX1_EXIT_AUGMENT_64=1
# ── HARD MAE-STOP risk overlay (2026-06-17, user vedtak: stop 'tåle 500 i minus i 8t for 16 i pluss') ──
# The learned Exit-IQL holds through deep adverse excursion to scratch a win (95% win-rate but worst
# single trade −416 bps MAE = the bulk of the 564 bps cap-3 account-DD). This caps EVERY trade: when the
# live unrealized PnL hits −80 bps, force EXIT_NOW. Validated (OOT hard-stop sim, 17132 trades): −80 caps
# the −416 tail at −80 for −1.7% total PnL (this is a deliberate risk-for-PnL trade the user chose). TUNE:
# raise to 120 for less cost (−0.8%) / lower to 50 for tighter risk (−4.9%); 0 = OFF (revert to pure IQL exit).
export GX1_EXIT_HARD_STOP_BPS=80
# ── LET-WINNERS-RUN overlay (2026-06-18, user vedtak: arm if green — confirmed on 2 OOT periods) ──
# The self-diagnosis found held_too_short = the dominant live leak (508 bps): the Exit-IQL takes profit AT
# the in-trade peak (giveback ~0) and the price keeps running. LWR suppresses that profit-EXIT_NOW while
# in-profit (pnl≥15) AND near-peak (giveback < FRAC), so the winner rides until a real trailing giveback
# (Strategy-F 30%) / hard-stop. OOT exit-replay gate (cap-3, LWR OFF vs ON, identical env): block[4]
# (2025-11→2026-05) +16.6% PnL / +16.4% DD; period2 (2024-06→2025-06) +32.7% PnL / +0.0% DD — robustly
# +PnL across regimes, worst-trade UNCHANGED both periods, DD increase regime-specific (flat in calm). ONE
# TRUTH (live make_exit_decision + phase6 gate). REVERSIBLE: GX1_EXIT_LET_WINNERS_RUN=0. TUNE: FRAC tighter
# (0.20) = less captured continuation + less DD.
export GX1_EXIT_LET_WINNERS_RUN=1
export GX1_LWR_GIVEBACK_FRAC=0.30
export GX1_LWR_MIN_PNL_BPS=15.0

# ── ENTRY operating point (513 genuine signal fields; model-native) ──────────
# Entry has no env-pinned direction operating point. The exact
# model-direction argmax contract is read in-process and rejects edge/utility
# thresholds and session allowlists. Trend/session/structure/liquidity/volatility/
# momentum/price-action/path/utility evidence belongs inside the learned model;
# the retired entry_iql-era pins (GX1_CONVICTION_*, GX1_SIZING_*, GX1_SKIP_ASIA,
# GX1_ENTRY_DIPFIX) are rejected. Current launch requires the separate learned
# sizing contract with fresh OOS, calibration and train==serve proof; historical
# fixed_1x is not a live fallback.

# ── CONTRACT OPERATING-POINT LAUNCH-ASSERT (vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707) ──
# For EVERY var in the contract's exit_iql.operating_point.live_env (dict), verify this launcher
# actually exported it with the contract value — a contract-named-but-never-exported Exit var
# is caught MECHANICALLY here instead of silently riding on a code
# default. ONE compare truth: gx1.execution.v12_exit_iql_live.exit_env_contract_diff (same
# normalize as the runner's own fail-closed startup assert). The entry-side live_env leg was
# REMOVED with the retired entry_iql chain (serving wave 2026-07-08): the smart entry consumes
# its operating point in-process from v10_entry.operating_point (asserted in the smart-gate above).
echo "[preflight] contract operating-point launch-assert (exit live_env)…"
PYTHONPATH=$REPO "$PY" - <<'PYEOF' || { echo "FATAL: launcher exports do not match the contract live_env — fix the export blocks above (or the contract, via explicit vedtak) before launching."; exit 1; }
import json, sys
from gx1.execution.v12_exit_iql_live import exit_env_contract_diff

contract = json.loads(open("/home/andre2/src/GX1_ENGINE/PROJECT_STATE_artifacts.json").read())
problems: list[str] = []

exit_le = (((contract.get("active") or {}).get("exit_iql") or {}).get("operating_point") or {}).get("live_env") or {}
if not exit_le:
    problems.append("exit_iql.operating_point.live_env missing/empty — exit policy UNPINNED (fail-closed)")
else:
    problems += [f"exit_iql: {d}" for d in exit_env_contract_diff(exit_le)]

if problems:
    print("[launch-assert] CONTRACT/LAUNCHER ENV MISMATCH:", *("  " + p for p in problems), sep="\n")
    sys.exit(1)
print(f"[launch-assert] OK: {len(exit_le)} exit contract live_env vars exported by this launcher")
PYEOF

# IN-PROCESS SHADOW: DISABLED with the retired Entry-IQL chain (serving wave
# 2026-07-08). The shadow config (GX1_DATA/config/shadow_bundle_dir.txt) named
# Entry-IQL candidate bundles; the smart chain v1 has no entry-IQL layer, so a
# shadow export would only produce fail-safe load errors in the runner log.
# Re-enable (new adapter class) when a smart-chain shadow candidate exists.

# ── Rule-9 LIVE-TAIL preflight (user vedtak 2026-06-11) ─────────────────────
# Freeze-signature scan of the live cv3+BASE34 prebuilt tails BEFORE launching anything:
# a was-varying column that is now constant on the recent tail = the 2026-05-25 BASE34
# copy-forward freeze class (lived 17 days while every training-side audit was green).
# Live must NEVER serve frozen context — hard fail here. The same call runs the
# CONTINUITY guard (grid gaps vs weekend/pause/holidays/KNOWN_DATA_GAPS; a fresh
# UNKNOWN gap <48h = hard fail) — gaps in history BLOCK every (re)start.
echo "[preflight] rule-9 live-tail freeze-signature + continuity scan…"
/home/andre2/src/GX1_ENGINE/.venv/bin/python -m gx1.audit.feature_liveness --live-tail --strict \
  || { echo "FATAL: rule-9 LIVE-TAIL/CONTINUITY check failed — frozen context or a fresh unknown gap in the live prebuilts. Fix the append wiring / backfill the gap (see gx1.audit.feature_liveness) before launching."; exit 1; }

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

# 2. Canonical incremental updater (loop @60s) ------------------------------
CANON_PID_FILE="$PAPER_RUNS/canonical_incremental.pid"
if [[ $FORCE -eq 1 ]]; then stop_if_running "$CANON_PID_FILE"; fi
if systemd_active gx1-canonical-incremental.service; then
    echo "[2/3] canonical_incremental owned by systemd (gx1-canonical-incremental.service ACTIVE) — skip (avoid double-spawn / torn parquet)"
elif pid=$(is_alive "$CANON_PID_FILE"); then
    echo "[2/3] canonical_incremental already RUNNING (PID $pid) — skip"
else
    echo "[2/3] launching v12_canonical_incremental --loop --interval 60..."
    nohup "$PY" -m gx1.execution.v12_canonical_incremental --loop --interval 60 \
        > "$LOG_DIR/canonical_incremental.log" 2>&1 &
    echo $! > "$CANON_PID_FILE"
    echo "    PID=$(cat $CANON_PID_FILE), log=$LOG_DIR/canonical_incremental.log"
fi

# Give the collectors a head-start so the runner sees fresh data on first poll.
sleep 3

# 3. Paper runner (exact model direction + execution safety) -----------------
RUNNER_PID_FILE="$PAPER_RUNS/paper_runner.pid"
# max_trades is an execution exposure cap verified against the exact model-
# direction operating-point contract; it never selects direction.
MAX_TRADES=${GX1_PAPER_MAX_TRADES:-3}
MAX_SPREAD=${GX1_PAPER_MAX_SPREAD_BPS:-7}
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
    GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1 GX1_EXIT_AUGMENT_64=1 \
    nohup "$PY" -m gx1.execution.v12_paper_runner \
        --max-trades "$MAX_TRADES" \
        --max-spread-bps "$MAX_SPREAD" \
        --journal-suffix "$SUFFIX" \
        > "$LOG_DIR/paper_runner.log" 2>&1 &
    echo $! > "$RUNNER_PID_FILE"
    echo "    PID=$(cat $RUNNER_PID_FILE), log=$LOG_DIR/paper_runner.log"
fi

echo ""
echo "=== Live practice stack status ==="
for label in "oanda_data_collector:$COLL_PID_FILE" \
             "canonical_incremental:$CANON_PID_FILE" \
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
echo "  tail -f $LOG_DIR/oanda_data_collector.log      # M1 tick fetch every 60s"
echo "  tail -f $LOG_DIR/canonical_incremental.log     # cv3 advance every 60s"
echo ""
echo "Stop everything:"
echo "  bash scripts/stop_live_practice.sh"
