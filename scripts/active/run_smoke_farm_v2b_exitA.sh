#!/usr/bin/env bash
set -euo pipefail

export M5_DATA="gx1/tests/data/xauusd_m5_smoke_2025_06_01_03.parquet"

CFG="gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml"
START="2025-06-01"
END="2025-06-03"
WORKERS=1

bash scripts/active/run_replay.sh "$CFG" "$START" "$END" "$WORKERS"

TRADE_LOG=$(ls gx1/wf_runs/*/trades.csv 2>/dev/null | tail -n1 || true)

if [ -z "$TRADE_LOG" ] || [ ! -f "$TRADE_LOG" ]; then
  # Fallback: use merged trade log produced by replay script
  TRADE_LOG=$(ls gx1/wf_runs/*/trade_log*merged.csv 2>/dev/null | tail -n1 || true)
fi

if [ -z "$TRADE_LOG" ] || [ ! -f "$TRADE_LOG" ]; then
  echo "[SMOKE] ERROR: no trade log" >&2
  exit 1
fi

NUM_TRADES=$(awk 'NR>1' "$TRADE_LOG" | wc -l | tr -d ' ')
echo "[SMOKE] trades: $NUM_TRADES"

if [ "$NUM_TRADES" -eq 0 ]; then
  echo "[SMOKE] ERROR: 0 trades" >&2
  exit 1
fi
