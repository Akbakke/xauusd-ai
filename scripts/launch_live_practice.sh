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

# ── Collector poll cadence (live SLA) ───────────────────────────────────────
# Tighten the OANDA M1 poll to 15s so a newly-closed bar reaches disk within ~15s
# (default code constant is 60s). The live source is the systemd unit (drop-in
# gx1-collector.service.d/poll15.conf sets the same value); this export only keeps
# the nohup FALLBACK collector in parity if systemd is ever not owning it.
export GX1_COLLECTOR_POLL_SECONDS=${GX1_COLLECTOR_POLL_SECONDS:-15}

# ── Regime-flag pins (2026-06-08 fase2b cement; top-level pin 2026-06-11) ────────────────
# LIVE serves the fase2b/CLEAN cement (V10 regime-v4 ctx_cont=123, EXIT_IO_V8=173). Pinned ON
# at top level so EVERY component launched from this script inherits the cemented flags (the
# old =0 COSTFIX-era pins survived here while the paper-runner line overrode them inline —
# stale narrative + a footgun for any future component that didn't override). Flags are
# EXPLICIT here — never rely on code defaults for live (build==serve flag parity).
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
export GX1_STRONG_HOLD_QADV=-200.0
export GX1_HOLD_HORIZON_OVERRUN_MULT=1.5
export GX1_HOLD_HORIZON_MIN_FLOOR_BARS=60
export GX1_USE_DISTILLED_EXIT=0

# ── ENTRY-SELECTION pins (2026-06-11, CEMENTED — vedtak 'thr −67 + conviction-sizing, låser denne') ──
# Entry-selection #2 = WIDER conviction-gate + CONVICTION-SIZING + skip-ASIA (PROJECT_STATE_artifacts.json
# entry_iql.operating_point). OPEN when raw_adv = best-TAKE-Q − SKIP-Q (un-clipped) >= −67.0
# (band [−67,−34.2) replayed through the FULL exit chain 2026-06-11: 18.2 bps/take @ 0.915 win,
# FLAT across sub-bands — the old top-20% gate was cutting into positive EV). SIZING: linear
# conviction multiplier 1.0→2.0 over [−67,−34.2] (band trades at 0.5–1.0× the old-gate exposure;
# UNITS dropped 10→5 so old-gate trades keep EXACTLY their prior 10-unit exposure at 2.0×).
# Locked numbers (skipASIA, size-weighted, scaled to full take-set): total +36.0% vs the −34.2
# gate, 2026 size-weighted win 0.847 (vs 0.853 — ACCEPTED by vedtak), union cap-3 DD −315 (vs
# −172). test==serve: size_units 500/500 exact; gate formula parity 200/200 (e44fd7dc).
# SUPERSEDES the 2026-06-10 thr −34.2 operating point. Bundle UNCHANGED — serve-time overlay,
# fully reversible (thr back to −34.2 + GX1_SIZING_MODE=off). Pinned EXPLICITLY.
# 2026-06-11 v2 (vedtak 'Go — lås both-sizing'): conviction × inverse-ATR vol-sizing. Replayed
# (exact live formula): total +25.8% vs the −34.2 gate, 2026 size-w win 0.8505 (BACK ABOVE the
# 0.85 floor), cap-3 DD −118.7 (vs −314.6 conviction-only, −172.3 old gate). 2026-vol (atr p50
# 13.4 vs 8.3 hist) auto-downsizes the tail regime. test==serve size_units 500/500 exact.
# 2026-06-11 v3 (vedtak 'Flip!'): BUNDLE flip -> entry_iql_volbal_20260611 (LAM50_SYM, vol-regime-
# balansert). OP rekalibrert til NY Q-skala: thr -37.71 = top-35%, sizing-ankre 65./80. pctil
# (LO -37.71 / HI -13.99). Evidens: 24.43 bps/take @ 0.9466 win, 2026-win 0.8543 PASS, DD -113.
# Kontrakten (entry_iql.operating_point) er sannhetskilden; kommentarene over beskriver v1/v2-historikken.
export GX1_CONVICTION_GATE=1
export GX1_SKIP_ASIA=1
export GX1_CONVICTION_THR=-37.71
export GX1_SIZING_MODE=both
export GX1_SIZING_MAX_MULT=2.0
export GX1_SIZING_MIN_MULT=0.5
export GX1_SIZING_CONV_LO=-37.71
export GX1_SIZING_CONV_HI=-13.99
export GX1_SIZING_ATR_REF_BPS=14.0
export GX1_SIZING_ATR_FLOOR_BPS=14.0

# IN-PROCESS SHADOW (2026-06-12, ladder wave): if a candidate bundle is named in
# the config file, the runner loads it as a SHADOW adapter — scores every poll,
# journals shadow_q/action/agreement, affects NOTHING. The nightly refit updates
# this file to its newest PENDING candidate; the runner picks it up at restart.
SHADOW_CFG=/home/andre2/GX1_DATA/config/shadow_bundle_dir.txt
if [[ -s "$SHADOW_CFG" ]]; then
    export GX1_SHADOW_BUNDLE_DIR="$(head -1 "$SHADOW_CFG" | tr -d '[:space:]')"
    echo "[launch] in-process SHADOW armed: $GX1_SHADOW_BUNDLE_DIR"
fi

FORCE=0
[[ "${1-}" == "--force" ]] && FORCE=1

# ── Rule 2: git-clean before the live launch (2026-06-13 audit gap) ──────────
# CLAUDE.md rule 2 says git-clean before ANY run, explicitly incl. live launch —
# but this launcher had no git check, so the highest-stakes run (the live XGB→V10
# →IQL→V3→Exit decision stack) could start on a dirty tree (we don't know what
# we're running). Mirror fase2b_rebuild.sh / gx1_candidate_gate.sh. --force skips
# (deliberate operator override only).
if [[ "$FORCE" != "1" ]] && [[ -n "$(git -C /home/andre2/src/GX1_ENGINE status --short)" ]]; then
    echo "FATAL: git tree is dirty — rule 2 (git-clean before any run, incl. live launch). Commit/stash, or pass --force to override deliberately:"
    git -C /home/andre2/src/GX1_ENGINE status --short
    exit 1
fi

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

# ── Rule-9 DRIFT preflight (user-direktiv 2026-06-12: «sjekk drift + hull hver
# gang vi starter på nytt») ──────────────────────────────────────────────────
# KS distribution-drift: last 7 days of live entry-states vs the ACTIVE bundle's
# training reference. ADVISORY by design (rule 3): market drift flags a retrain
# VEDTAK, it must never block a launch — a bot that refuses to start in a new
# regime is wrong; the hard catastrophe floors above (freeze/gaps) still block.
# Structural reference problems (dim/name mismatch) surface here loudly too.
echo "[preflight] rule-9 distribution-drift scan (advisory)…"
DRIFT_REF=$(PYTHONPATH=/home/andre2/src/GX1_ENGINE /home/andre2/src/GX1_ENGINE/.venv/bin/python -c "
from gx1_guards.artifacts import load_decision_artifact
from pathlib import Path
p = Path(load_decision_artifact('entry_iql')) / 'drift_reference_v1.parquet'
print(p if p.is_file() else '')" 2>/dev/null || true)
if [[ -n "$DRIFT_REF" ]]; then
    PYTHONPATH=/home/andre2/src/GX1_ENGINE /home/andre2/src/GX1_ENGINE/.venv/bin/python \
        -m gx1.audit.feature_liveness --distribution-drift \
        --drift-reference "$DRIFT_REF" --journal-days 7 \
        || echo "[preflight] ⚠ DRIFT-ALERT (advisory) — review the lines above; persistent drift = consider a retrain vedtak (rule 3: never auto)"
else
    echo "[preflight] ⚠ drift scan SKIPPED — ACTIVE entry bundle has no drift_reference_v1.parquet (generate via feature_liveness --write-drift-reference at next cement)"
fi

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
UNITS=${GX1_PAPER_UNITS:-5}   # 2026-06-11: 10→5 with SIZING_MAX_MULT=2.0 — old-gate trades stay at 10 units (5×2.0)
# max_trades=3 is the DD-VALIDATED portfolio cap (entry_iql.operating_point); the conviction-gate's
# −201 DD was measured at this cap. Default was 100 (unbounded-risk footgun); pinned to 3 for live.
MAX_TRADES=${GX1_PAPER_MAX_TRADES:-3}
MAX_SPREAD=${GX1_PAPER_MAX_SPREAD_BPS:-9999}
SUFFIX=${GX1_PAPER_SUFFIX:-conviction67sized_skipasia_pure_phase6}

# Orphan-reaper (2026-06-13 audit): the spawn gate below only kill -0's the SINGLE pid in the
# pid-file, so every relaunch that found that pid dead spawned a fresh runner ON TOP of still-alive
# orphans from prior runs — 9 concurrent runners accrued, double-journaling under one suffix and
# multi-submitting against one OANDA account (and pre-fix-code orphans = NO-OLD-CODE violation).
# Reap any v12_paper_runner that is NOT the (alive) pid-file pid before deciding to spawn.
_keep=$(is_alive "$RUNNER_PID_FILE" || true)
for _p in $(pgrep -f "gx1.execution.v12_paper_runner" 2>/dev/null || true); do
    if [[ "$_p" != "$_keep" ]]; then
        echo "[3/4] reaping ORPHAN paper_runner PID $_p (not pid-file-tracked '${_keep:-none}') — prevents double-journal / multi-submit / stale code"
        kill "$_p" 2>/dev/null || true
    fi
done

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
