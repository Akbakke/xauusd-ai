#!/usr/bin/env bash
set -euo pipefail

PY="/home/andre2/venvs/gx1/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[RUN_FAIL] canonical python not found/executable: $PY" >&2
  exit 1
fi

if [[ -z "${GX1_DATA:-}" ]]; then
  echo "[RUN_FAIL] GX1_DATA env var is required" >&2
  exit 2
fi

if [[ "${1:-}" != "--inputs_capsule" ]]; then
  echo "[RUN_FAIL] Usage: $0 --inputs_capsule /ABS/PATH/TO/PRUNE_AB_FULL2025_QUARTERS_INPUTS.json" >&2
  exit 2
fi
INPUTS_CAPSULE="${2:-}"
if [[ -z "$INPUTS_CAPSULE" ]]; then
  echo "[RUN_FAIL] --inputs_capsule requires a value" >&2
  exit 2
fi
if [[ "$INPUTS_CAPSULE" != /* ]]; then
  echo "[RUN_FAIL] inputs_capsule must be an absolute path: $INPUTS_CAPSULE" >&2
  exit 2
fi
if [[ ! -f "$INPUTS_CAPSULE" ]]; then
  echo "[RUN_FAIL] inputs_capsule not found: $INPUTS_CAPSULE" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="$GX1_DATA/reports/prune_ab_v13_refined3_full2025_quarters/$RUN_TS"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/RUN.log"

function write_fatal() {
  local exit_code="$1"
  local step="${2:-UNKNOWN}"
  local tail
  tail="$(tail -n 200 "$LOG_PATH" 2>/dev/null || true)"
  cat >"$RUN_DIR/RUN_PRUNE_AB_FULL2025_QUARTERS_FATAL.json" <<EOF
{
  "status": "FAIL",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "GX1_DATA": "$GX1_DATA",
  "sys_executable": "$PY",
  "step": "$step",
  "exit_code": $exit_code,
  "inputs_capsule": "$INPUTS_CAPSULE",
  "log_tail": $("$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<<"$tail")
}
EOF
}

STEP="INIT"
trap 'ec=$?; write_fatal "$ec" "$STEP"; exit "$ec"' ERR

echo "[RUN] run_dir=$RUN_DIR GX1_DATA=$GX1_DATA root=$ROOT_DIR inputs_capsule=$INPUTS_CAPSULE" | tee -a "$LOG_PATH"

STEP="COPY_INPUTS"
cp -f "$INPUTS_CAPSULE" "$RUN_DIR/INPUTS_CAPSULE_USED.json"

STEP="EVAL_QUARTERS"
OUT_EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$OUT_EVAL_DIR"

"$PY" "$ROOT_DIR/gx1/scripts/eval_prune_ab_full2025_quarters.py" \
  --inputs_capsule "$INPUTS_CAPSULE" \
  --out_dir "$OUT_EVAL_DIR" \
  --strict 1 \
  | tee -a "$LOG_PATH"

STEP="WRITE_COMPLETED"
cat >"$RUN_DIR/RUN_PRUNE_AB_FULL2025_QUARTERS_COMPLETED.json" <<EOF
{
  "status": "OK",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "GX1_DATA": "$GX1_DATA",
  "sys_executable": "$PY",
  "run_dir": "$RUN_DIR",
  "inputs_capsule": "$INPUTS_CAPSULE",
  "paths": {
    "eval_dir": "$OUT_EVAL_DIR",
    "summary_md": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_QUARTERS_SUMMARY.md",
    "summary_json": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_QUARTERS_SUMMARY.json",
    "table_csv": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_QUARTERS_TABLE.csv",
    "log": "$LOG_PATH"
  }
}
EOF

echo "[RUN_OK] wrote $RUN_DIR/RUN_PRUNE_AB_FULL2025_QUARTERS_COMPLETED.json" | tee -a "$LOG_PATH"

