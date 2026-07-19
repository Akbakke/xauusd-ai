#!/usr/bin/env bash
# One-shot fail-closed chain driver for the seq513 rebuild vedtak:
#   [wait for ranking] -> signal manifest -> rebuild preflight -> dataset rebuild
# Stops BEFORE the smoke gate. First red stops the chain, writes CHAIN_STATUS.json
# and pings Telegram (via the existing notifier's send()). Resume-safe: steps whose
# terminal artifact already exists are skipped. No new authority: every step calls
# the existing vedtak-gated producers/wrappers with explicit immutable arguments.
set -euo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
VEDTAK=${1:?usage: $0 VEDTAK_ID EVENT_ROOT}
EVENT=${2:?usage: $0 VEDTAK_ID EVENT_ROOT}

SRC="$EVENT/FULL_PLUS_CTX_v3src.parquet"
CV2="$EVENT/canonical_features_v2.parquet"
MTF="$EVENT/MULTI_TF_V2_CACHE"
TAPE="$EVENT/m5_tape_repaired_dec2024"
RANK_NPZ="$EVENT/model_native_train_rank_reference_v3.npz"
OUTPUT="$EVENT/dataset/v10_seq513_dataset__HOLD_03B.parquet"
AUDIT="$EVENT/audit"
PRE_OUT="$EVENT/preflight"
HISTORY_START=2020-11-13T00:00:00Z
TRAIN_START=2020-11-13T00:00:00Z
TRAIN_END=2026-03-31T23:59:59Z
VAL_START=2026-04-01T00:00:00Z
VAL_END=2026-04-30T23:59:59Z
TEST_START=2026-05-01T00:00:00Z
TEST_END=2026-06-14T23:59:59Z

LOG="$EVENT/CHAIN_LOG_$(date -u +%Y%m%dT%H%M%SZ).txt"
STATUS="$EVENT/CHAIN_STATUS.json"

tg() {
  "$PY" - "$1" <<'PYEOF' || true
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "gx1_tg", "/home/andre2/src/GX1_ENGINE/scripts/gx1_telegram_notifier.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.send(sys.argv[1])
PYEOF
}

write_status() {
  "$PY" - "$1" "$2" "$STATUS" <<'PYEOF'
import json, sys
from datetime import datetime, timezone
step, state, path = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {"step": step, "state": state,
           "updated_utc": datetime.now(timezone.utc).isoformat()}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
PYEOF
}

fail() {
  local step=$1
  write_status "$step" RED
  tg "🔴 GX1 seq513-kjede STOPPET rødt på steg: $step. Logg: $LOG"
  echo "[chain] RED at $step — see $LOG" >&2
  exit 2
}

echo "[chain] vedtak=$VEDTAK event=$EVENT log=$LOG"

# ── Step 1: wait for the ranking artifact (ranker runs separately) ──────────
write_status wait-ranking RUNNING
RANKING=""
for _ in $(seq 1 480); do  # up to 8h, 60s poll
  RANKING=$(ls -1 "$EVENT"/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_*.json 2>/dev/null | sort | tail -1 || true)
  [[ -n $RANKING ]] && break
  if ! pgrep -f '^/home/andre2/src/GX1_ENGINE/\.venv/bin/python -m gx1\.scripts\.materialize_entry_model_native_train_feature_ranker_v1' >/dev/null; then
    # ranker not running and no artifact -> red
    sleep 5
    RANKING=$(ls -1 "$EVENT"/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_*.json 2>/dev/null | sort | tail -1 || true)
    [[ -n $RANKING ]] && break
    fail wait-ranking
  fi
  sleep 60
done
[[ -n $RANKING ]] || fail wait-ranking
echo "[chain] ranking: $RANKING"

# ── Step 2: signal manifest ────────────────────────────────────────────────
MANIFEST=$(ls -1 "$EVENT"/ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_*.json 2>/dev/null | sort | tail -1 || true)
if [[ -z $MANIFEST ]]; then
  write_status signal-manifest RUNNING
  STAMP=$(date -u +%Y%m%dT%H%M%S%6NZ)
  MANIFEST="$EVENT/ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_${STAMP}.json"
  (cd "$ENG" && "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 \
    --feature-ranking-json "$RANKING" --out "$MANIFEST" --vedtak "$VEDTAK") \
    >>"$LOG" 2>&1 || fail signal-manifest
fi
echo "[chain] manifest: $MANIFEST"

# ── Step 3: rebuild preflight ──────────────────────────────────────────────
PRE_DONE=$(ls -1 "$PRE_OUT"/*.json 2>/dev/null | head -1 || true)
if [[ -z $PRE_DONE ]]; then
  write_status rebuild-preflight RUNNING
  (cd "$ENG" && bash scripts/entry_next_edge_control.sh model-native-rebuild-preflight \
    --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
    --signal-manifest "$MANIFEST" --rank-reference-npz "$RANK_NPZ" \
    --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
    --output "$OUTPUT" --audit-out-dir "$AUDIT" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --val-start "$VAL_START" --val-end "$VAL_END" \
    --test-start "$TEST_START" --test-end "$TEST_END" \
    --out-dir "$PRE_OUT") >>"$LOG" 2>&1 || fail rebuild-preflight
fi
echo "[chain] preflight OK"

# ── Step 4: dataset rebuild (multi-hour; wrapper is capped internally) ─────
if [[ ! -e "$EVENT/dataset/DATASET_BUILD_PROOF.json" ]]; then
  write_status dataset-rebuild RUNNING
  tg "⚙️ GX1 seq513: preflight grønn — dataset-rebuild startet (fler-timers jobb)."
  (cd "$ENG" && bash scripts/rebuild_entry_model_native_seq513_dataset.sh \
    --vedtak "$VEDTAK" \
    --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
    --signal-manifest "$MANIFEST" --rank-reference-npz "$RANK_NPZ" \
    --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
    --output "$OUTPUT" --audit-out-dir "$AUDIT" \
    --history-start "$HISTORY_START" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --val-start "$VAL_START" --val-end "$VAL_END" \
    --test-start "$TEST_START" --test-end "$TEST_END") \
    >>"$LOG" 2>&1 || fail dataset-rebuild
fi

write_status chain-complete GREEN
tg "✅ GX1 seq513-kjeden er GRØNN t.o.m. dataset-rebuild. Neste: smoke-gate (manuelt vedtakssteg)."
echo "[chain] GREEN — stopped at the smoke gate as designed"
