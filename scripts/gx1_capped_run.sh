#!/usr/bin/env bash
# gx1_capped_run.sh — run a heavy job under a HARD cgroup memory ceiling so that an OOM kills
# the JOB (cgroup OOM-killer), NEVER freezes/crashes the whole machine.
#
# BIRTHED 2026-06-17: the FULL phase6 entry-gate (peaks ~56G on a 58G WSL cap) OOM-crashed the
# PC mid-run (froze → hard reboot → live runner left down → open paper trades unmanaged ~3h).
# CLAUDE.md/AGENTS.md make RAM-headroom a HARD ceiling ("an OOM CRASHED the PC"); this enforces it
# by construction instead of "remember to watch RAM". Use for EVERY heavy job (gate / build / replay).
#
# Usage:  scripts/gx1_capped_run.sh [--mem 42G] [--swap 2G] -- <command ...>
#   --mem   MemoryMax (hard) + MemoryHigh (throttle) for the job's cgroup scope. Default 42G:
#           leaves ~14G headroom on the 58G WSL cap for the OS + systemd data daemons + buff/cache.
#   --swap  MemorySwapMax. Default 2G (small — swap-thrash is what froze the box; cap it tight so the
#           cgroup OOM-killer fires fast instead of the machine swap-storming into a freeze).
# If the job exceeds --mem, the kernel kills THAT scope (non-zero exit) and the PC survives.
set -euo pipefail
MEM=42G ; SWAP=2G
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem)  MEM="$2";  shift 2 ;;
    --swap) SWAP="$2"; shift 2 ;;
    --)     shift; break ;;
    *) echo "FATAL: unknown arg '$1' (put the command after '--')"; exit 2 ;;
  esac
done
[[ $# -ge 1 ]] || { echo "FATAL: no command given after '--'"; exit 2; }
echo "[capped_run] MemoryMax=$MEM MemoryHigh=$MEM MemorySwapMax=$SWAP"
echo "[capped_run] cmd: $*"
# A transient service keeps this wrapper outside the job cgroup. If the kernel
# kills the capped job, the caller can publish a durable terminal RED status.
exec systemd-run --user --wait --collect --quiet \
  -p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP" \
  -- "$@"
