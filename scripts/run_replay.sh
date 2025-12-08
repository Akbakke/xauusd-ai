#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
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

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Resolve M5_DATA path: check env var first, then search for bid/ask datasets
if [[ -n "${M5_DATA:-}" ]]; then
    # Use explicit M5_DATA if set
    RESOLVED_M5_DATA="$M5_DATA"
    echo "[DATA] Using M5_DATA from environment: $RESOLVED_M5_DATA"
else
    # Search for M5 bid/ask datasets
    echo "[DATA] M5_DATA not set, searching for M5 bid/ask datasets..."
    
    # Search patterns (in order of preference)
    SEARCH_PATHS=(
        "data/raw/xauusd*m5*bid*ask*.parquet"
        "data/raw/*xauusd*m5*.parquet"
        "data/raw/*m5*bid*ask*.parquet"
        "data/xauusd*m5*bid*ask*.parquet"
        "data/*xauusd*m5*.parquet"
        "gx1/data/xauusd*m5*bid*ask*.parquet"
        "gx1/data/*xauusd*m5*.parquet"
        "gx1/wf_runs/*/test_data*2025*.parquet"
        "gx1/wf_runs/*/*bid*ask*.parquet"
        "gx1/wf_runs/TEST_REPLAY/*.parquet"
    )
    
    RESOLVED_M5_DATA=""
    for pattern in "${SEARCH_PATHS[@]}"; do
        # Use find to handle glob patterns safely
        found=$(find . -path "./$pattern" -type f 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            # Prefer files with "2025" or longer names (more recent/comprehensive)
            if [[ "$found" == *"2025"* ]] || [[ -z "$RESOLVED_M5_DATA" ]]; then
                RESOLVED_M5_DATA="$found"
            fi
        fi
    done
    
    if [[ -z "$RESOLVED_M5_DATA" ]]; then
        echo "ERROR: No M5 bid/ask dataset found."
        echo "  Set M5_DATA environment variable, e.g.:"
        echo "    export M5_DATA='data/raw/xauusd_m5_2025_bid_ask.parquet'"
        echo "  Or place a file matching one of these patterns:"
        printf "    - %s\n" "${SEARCH_PATHS[@]}"
        exit 1
    fi
    
    echo "[DATA] Found M5 dataset: $RESOLVED_M5_DATA"
fi

# Verify file exists
if [[ ! -f "$RESOLVED_M5_DATA" ]]; then
    echo "ERROR: M5_DATA file not found: $RESOLVED_M5_DATA"
    exit 1
fi

LIMITED_DATA="$OUTPUT_DIR/price_data_filtered.parquet"
export M5_DATA="$RESOLVED_M5_DATA" START_DATE END_DATE LIMITED_DATA

python3 - <<'PY'
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

python scripts/active/replay_entry_exit_parallel.py \
  --price-data "$LIMITED_DATA" \
  --base-policy "$POLICY_PATH" \
  --n-workers "$N_WORKERS" \
  --output "$OUTPUT_DIR/results.json" 2>&1 | tee "$LOG_DIR/replay.log"
