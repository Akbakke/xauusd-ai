#!/bin/bash

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PY="${GX1_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "FATAL: repo Python venv missing: $PY" >&2
    exit 2
fi
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <policy_yaml> <start_date> <end_date> [n_workers] [output_dir]"
  exit 1
fi

POLICY_PATH="$1"
START_DATE="$2"
END_DATE="$3"
N_WORKERS="${4:-6}"
OUTPUT_DIR="${5:-gx1/wf_runs/$(basename "${POLICY_PATH%.*}")}"

export START_DATE END_DATE

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Resolve M5_DATA path: canonical tape only (no raw lane)
if [[ -n "${M5_DATA:-}" ]]; then
    RESOLVED_M5_DATA="$M5_DATA"
    if [[ "$RESOLVED_M5_DATA" == *"/data/data/raw/"* ]] || [[ "$RESOLVED_M5_DATA" == *"xauusd_m5_2025_bid_ask.parquet"* ]]; then
        echo "ERROR: M5_DATA points to legacy raw lane: $RESOLVED_M5_DATA" >&2
        exit 1
    fi
    echo "[DATA] Using M5_DATA from environment: $RESOLVED_M5_DATA"
else
    TAPE_ROOT="${GX1_CANONICAL_TAPE_ROOT:-/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL}"
    TAPE_YEAR="$("$PY" - <<'PY'
import os
from datetime import datetime, timezone

start_ts = os.environ.get("START_DATE", "").strip()
end_ts = os.environ.get("END_DATE", "").strip()
if not start_ts or not end_ts:
    raise SystemExit("ERROR: START_DATE/END_DATE must be set to resolve canonical tape year")

def parse_year(ts: str) -> int:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).year

ys = parse_year(start_ts)
ye = parse_year(end_ts)
if ys != ye:
    raise SystemExit(f"ERROR: start/end span multiple years (start_year={ys}, end_year={ye})")
print(ys)
PY
)"
    RESOLVED_M5_DATA="$TAPE_ROOT/year=$TAPE_YEAR/part-000.parquet"
    echo "[DATA] Resolved canonical tape: $RESOLVED_M5_DATA"
fi

# Verify file exists
if [[ ! -f "$RESOLVED_M5_DATA" ]]; then
    echo "ERROR: M5_DATA file not found: $RESOLVED_M5_DATA"
    exit 1
fi
REPLAY_INSTRUMENT="${GX1_REPLAY_INSTRUMENT:-XAUUSD}"
if [[ "${REPLAY_INSTRUMENT^^}" == "XAUUSD" ]]; then
    _m5_lower="${RESOLVED_M5_DATA,,}"
    if [[ "$_m5_lower" != *"xau"* ]]; then
        echo "ERROR: XAUUSD replay requires XAU price data, got M5_DATA=$RESOLVED_M5_DATA" >&2
        exit 1
    fi
fi

LIMITED_DATA="$OUTPUT_DIR/price_data_filtered.parquet"
DEFAULT_CANONICAL_TRUTH_FILE="$PROJECT_ROOT/gx1/configs/canonical_truth_signal_only.json"
if [[ -z "${GX1_CANONICAL_TRUTH_FILE:-}" && -f "$DEFAULT_CANONICAL_TRUTH_FILE" ]]; then
    export GX1_CANONICAL_TRUTH_FILE="$DEFAULT_CANONICAL_TRUTH_FILE"
fi
if [[ -z "${GX1_CANONICAL_BUNDLE_DIR:-}" && -n "${GX1_CANONICAL_TRUTH_FILE:-}" && -f "$GX1_CANONICAL_TRUTH_FILE" ]]; then
    export GX1_CANONICAL_BUNDLE_DIR="$("$PY" - <<'PY'
import json
import os
from pathlib import Path

truth_file = Path(os.environ["GX1_CANONICAL_TRUTH_FILE"])
with truth_file.open("r", encoding="utf-8") as fh:
    truth = json.load(fh)
bundle = str(truth.get("canonical_xgb_bundle_dir") or "").strip()
if not bundle:
    raise SystemExit("ERROR: canonical_xgb_bundle_dir missing from truth file")
print(bundle)
PY
)"
fi
if [[ -z "${GX1_BUNDLE_DIR:-}" && -n "${GX1_CANONICAL_TRUTH_FILE:-}" && -f "$GX1_CANONICAL_TRUTH_FILE" ]]; then
    export GX1_BUNDLE_DIR="$("$PY" - <<'PY'
import json
import os
from pathlib import Path

truth_file = Path(os.environ["GX1_CANONICAL_TRUTH_FILE"])
with truth_file.open("r", encoding="utf-8") as fh:
    truth = json.load(fh)
bundle = str(truth.get("canonical_transformer_bundle_dir") or "").strip()
if not bundle:
    raise SystemExit("ERROR: canonical_transformer_bundle_dir missing from truth file")
print(bundle)
PY
)"
fi

export M5_DATA="$RESOLVED_M5_DATA" START_DATE END_DATE LIMITED_DATA
export GX1_DATA="${GX1_DATA:-/home/andre2/GX1_DATA}"
export GX1_DISABLE_DOTENV="${GX1_DISABLE_DOTENV:-1}"
export GX1_REPLAY_INCREMENTAL_FEATURES="${GX1_REPLAY_INCREMENTAL_FEATURES:-1}"
export GX1_REPLAY_NO_CSV="${GX1_REPLAY_NO_CSV:-1}"
export GX1_FEATURE_USE_NP_ROLLING="${GX1_FEATURE_USE_NP_ROLLING:-1}"

"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
import os
from datetime import timedelta

input_path = Path(os.environ["M5_DATA"])
output_path = Path(os.environ["LIMITED_DATA"])
start_date = pd.to_datetime(os.environ["START_DATE"], utc=True)
end_date = pd.to_datetime(os.environ["END_DATE"], utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

print(f"[DATA] Using M5 dataset from {input_path}")

if input_path.suffix.lower() == ".parquet":
    df = pd.read_parquet(input_path)
elif input_path.suffix.lower() == ".csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError(f"Unsupported price data format: {input_path}")

if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
elif "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
else:
    df.index = pd.to_datetime(df.index, utc=True)

filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()
filtered.to_parquet(output_path)
print(f"Filtered {len(filtered):,} bars → {output_path}")
PY

POLICY_BASENAME="$(basename "$POLICY_PATH")"
TRUTH_FILE=""
USE_TRUTH_REPLAY=0
case "$POLICY_BASENAME" in
  PHASE5_EXIT_VERIFICATION_V10_CTX__R5C.yaml)
    TRUTH_FILE="$PROJECT_ROOT/gx1/configs/phase5_exit_verification_r5c.json"
    USE_TRUTH_REPLAY=1
    ;;
  PHASE5_EXIT_VERIFICATION_V10_CTX__R5D.yaml)
    TRUTH_FILE="$PROJECT_ROOT/gx1/configs/phase5_exit_verification_r5d.json"
    USE_TRUTH_REPLAY=1
    ;;
esac

if [[ "$USE_TRUTH_REPLAY" == "1" ]]; then
    echo "[REPLAY] Using truth-grade prebuilt replay lane: $TRUTH_FILE"
    export GX1_CANONICAL_TRUTH_FILE="$TRUTH_FILE"
    export GX1_REPLAY_USE_PREBUILT_FEATURES=1
    export GX1_FEATURE_BUILD_DISABLED=1
    "$PY" -m gx1.scripts.run_truth_e2e_sanity \
      --truth-file "$TRUTH_FILE" \
      --start-ts "$START_DATE" \
      --end-ts "$END_DATE" \
      --run-dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/replay.log"
else
    "$PY" scripts/active/replay_entry_exit_parallel.py \
      --price-data "$LIMITED_DATA" \
      --base-policy "$POLICY_PATH" \
      --n-workers "$N_WORKERS" \
      --output "$OUTPUT_DIR/results.json" 2>&1 | tee "$LOG_DIR/replay.log"
fi
