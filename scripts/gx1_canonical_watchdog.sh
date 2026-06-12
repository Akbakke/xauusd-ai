#!/usr/bin/env bash
# Canonical-incremental HANG WATCHDOG (2026-06-12, after hang #3 in 24h).
#
# The daemon has hung silently three times (S-state, main thread in do_select,
# 60 workers in futex_wait): twice at the 21-22Z pause boundary, once mid
# catch-up — the in-loop self-check theory died with hang #3 (it was already
# moved out). Until the root cause is pinned, this watchdog makes the hang a
# self-healing blip: on staleness it first SIGUSR1s the process (faulthandler
# dumps ALL thread stacks into the daemon log — the next hang self-documents),
# then restarts the unit. The runner is fail-closed against stale prebuilts
# (PREBUILT_STALE guard), so the only cost of a hang is decision downtime.
#
# Scheduling: gx1-canonical-watchdog.timer (systemd --user, every 5 min).
set -uo pipefail

LOG=/home/andre2/GX1_DATA/reports/v12_paper_runs/logs/canonical_incremental.log
WLOG=/home/andre2/GX1_DATA/reports/v12_paper_runs/logs/canonical_watchdog.log
STALE_SEC=600

ts() { date -u +%FT%TZ; }

# Daily 21:00-22:05Z pause: dataless window, silence is structural — never
# judge staleness here (the 2026-06-12 lesson; reopen is checked from 22:05).
HM=$(date -u +%H%M)
if [ "$HM" -ge 2100 ] && [ "$HM" -le 2205 ]; then exit 0; fi
# Weekend: market closed Fri 21:00Z → Sun 22:05Z; no data, no staleness signal.
DOW=$(date -u +%u)
if [ "$DOW" -ge 6 ] || { [ "$DOW" -eq 5 ] && [ "$HM" -ge 2100 ]; } || { [ "$DOW" -eq 7 ] && [ "$HM" -le 2205 ]; }; then exit 0; fi

AGE=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
if [ "$AGE" -le "$STALE_SEC" ]; then exit 0; fi

PID=$(systemctl --user show gx1-canonical-incremental -p MainPID --value)
echo "[$(ts)] HANG SUSPECTED: log stale ${AGE}s (pid=$PID) — dumping stacks + restarting" >> "$WLOG"
if [ -n "$PID" ] && [ "$PID" != "0" ]; then
    # faulthandler.register(SIGUSR1) in the daemon → full thread stacks into its log
    kill -USR1 "$PID" 2>/dev/null && sleep 5
fi
systemctl --user restart gx1-canonical-incremental
echo "[$(ts)] restarted gx1-canonical-incremental (stacks, if any, are in canonical_incremental.log above this restart)" >> "$WLOG"
