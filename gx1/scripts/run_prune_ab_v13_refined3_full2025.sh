#!/usr/bin/env bash
set -euo pipefail

PY="/home/andre2/venvs/gx1/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[RUN_FAIL] canonical python not found/executable: $PY" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GX1_DATA="${GX1_DATA:-/home/andre2/GX1_DATA}"
export GX1_DATA

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="$GX1_DATA/reports/prune_ab_v13_refined3_full2025/$RUN_TS"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/RUN.log"

# Hard requirement: no auto-search. You MUST provide an explicit inputs json path.
# Recommended: point to a prior PRUNE_AB run completion capsule:
#   /home/andre2/GX1_DATA/reports/prune_ab_v13_refined3/<ts>/RUN_PRUNE_AB_COMPLETED.json
INPUTS_CAPSULE="${PRUNE_AB_COMPLETED_JSON:-}"
if [[ -z "${1:-}" ]]; then
  true
else
  if [[ "$1" == "--inputs_capsule" ]]; then
    INPUTS_CAPSULE="${2:-}"
  fi
fi
if [[ -z "$INPUTS_CAPSULE" ]]; then
  echo "[RUN_FAIL] Provide --inputs_capsule /ABS/PATH/TO/PRUNE_AB_FULL2025_INPUTS.json (explicit; no auto-search)." | tee -a "$LOG_PATH" >&2
  exit 2
fi
if [[ ! -f "$INPUTS_CAPSULE" ]]; then
  echo "[RUN_FAIL] inputs_capsule not found: $INPUTS_CAPSULE" | tee -a "$LOG_PATH" >&2
  exit 2
fi

function write_fatal() {
  local exit_code="$1"
  local step="${2:-UNKNOWN}"
  local tail
  tail="$(tail -n 200 "$LOG_PATH" 2>/dev/null || true)"
  cat >"$RUN_DIR/RUN_PRUNE_AB_FULL2025_FATAL.json" <<EOF
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

STEP="EVAL_FULL2025"
OUT_EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$OUT_EVAL_DIR"

"$PY" "$ROOT_DIR/gx1/scripts/eval_prune_ab_full2025.py" \
  --strict 1 \
  --inputs_capsule "$INPUTS_CAPSULE" \
  --out_dir "$OUT_EVAL_DIR" \
  --base28_schema "/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260210_140328.schema_manifest.json" \
  --base28_parquet "/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260210_140328.parquet" \
  | tee -a "$LOG_PATH"

STEP="WRITE_COMPLETED"
cat >"$RUN_DIR/RUN_PRUNE_AB_FULL2025_COMPLETED.json" <<EOF
{
  "status": "OK",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "GX1_DATA": "$GX1_DATA",
  "sys_executable": "$PY",
  "run_dir": "$RUN_DIR",
  "inputs_capsule": "$INPUTS_CAPSULE",
  "paths": {
    "eval_dir": "$OUT_EVAL_DIR",
    "summary_md": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_SUMMARY.md",
    "summary_json": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_SUMMARY.json",
    "calibration_curves_json": "$OUT_EVAL_DIR/PRUNE_AB_FULL2025_CALIBRATION_CURVES.json",
    "log": "$LOG_PATH"
  }
}
EOF

echo "[RUN_OK] wrote $RUN_DIR/RUN_PRUNE_AB_FULL2025_COMPLETED.json" | tee -a "$LOG_PATH"

