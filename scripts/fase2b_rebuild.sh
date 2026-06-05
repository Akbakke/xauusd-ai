#!/usr/bin/env bash
# Fase-2B rebuild orchestrator — vedtak fase2b_regime_v4_rebuild_20260605.
# Encodes the EXACT order/inputs/guards from FASE2B_REBUILD_ORDER.md (read that first).
# Steps 1-3 are VERIFIED + idempotent (skip if output exists). Steps 4-7 are printed as a
# guided checklist (heavy/unverified-by-script — run + verify each deliberately).
#
# Usage: bash scripts/fase2b_rebuild.sh            # runs verified steps 1-3, prints 4-7
#        bash scripts/fase2b_rebuild.sh --force     # re-run steps 1-3 even if outputs exist
set -euo pipefail

export GX1_REGIME_V4=1
export GX1_TREND_REGIME_FROM_D1=1
VEDTAK=fase2b_regime_v4_rebuild_20260605
ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
WS=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605
TAPE=/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL
PIN=/home/andre2/GX1_DATA/data/data/prebuilt/_PINNED_FASE2B_20260605
FORCE=${1:-}
mkdir -p "$WS"
cd "$ENG"

# git clean is required before any run (rule 2)
[ -z "$(git status --short)" ] || { echo "[ABORT] git tree dirty — commit/stash first (rule 2)"; exit 1; }

have() { [ -f "$1" ] && [ "$FORCE" != "--force" ]; }

echo "=================== STEP 1a: canonical_features_v2 (clean M5 tape) ==================="
if have "$WS/canonical_features_v2.parquet"; then echo "  [skip] exists"; else
  $PY -m gx1.scripts.materialize_build_canonical_features_v2 --out-path "$WS/canonical_features_v2.parquet"
fi

echo "=================== STEP 1b: cv3 augment + glitch-guard + re-pin ==================="
if have "$WS/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"; then echo "  [skip] exists"; else
  $PY -m gx1.scripts.materialize_canonical_v3_augment \
    --input "$WS/canonical_features_v2.parquet" --output-dir "$WS/cv3"
fi
# glitch-guard (x10 must be FIXED) + re-pin
$PY - <<PYEOF
import pandas as pd
from gx1.io.price_glitch_guard import detect_price_scale_glitch, assert_no_price_scale_glitch
d=pd.read_parquet("$WS/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet")
dd=d.reset_index() if "time" not in d.columns else d
assert len(detect_price_scale_glitch(dd))==0, "[GLITCH] x10 NOT fixed in rebuilt cv3"
assert_no_price_scale_glitch(dd)
print("  [glitch-guard] PASS (cv3 clean)")
PYEOF
mkdir -p "$PIN"
if [ ! -f "$PIN/xauusd_m5_CANONICAL_V3_2020_2026.parquet" ] || [ "$FORCE" = "--force" ]; then
  cp "$WS/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet" "$PIN/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
  echo "  [re-pin] clean cv3 -> $PIN"
fi

echo "=================== STEP 2: FULL_PLUS_CTX (add_ctx_cont on v2 trimmed to model range) ==================="
if have "$WS/FULL_PLUS_CTX.parquet"; then echo "  [skip] exists"; else
  # trim v2 to model range (>=2020-11-09; first ~10mo = HTF warmup via full-tape raw_m5)
  $PY - <<PYEOF
import pandas as pd
d=pd.read_parquet("$WS/canonical_features_v2.parquet")
t=pd.to_datetime(d["time"],utc=True) if "time" in d.columns else pd.to_datetime(d.index,utc=True)
d[t>=pd.Timestamp("2020-11-09",tz="UTC")].to_parquet("$WS/canonical_features_v2_modelrange.parquet",index=False)
PYEOF
  $PY -m gx1.scripts.add_ctx_cont_columns_to_prebuilt \
    --prebuilt_parquet "$WS/canonical_features_v2_modelrange.parquet" \
    --output_parquet "$WS/FULL_PLUS_CTX.parquet" \
    --ctx-cont-dim 16 --ctx-cat-dim 5 \
    --tape-root "$TAPE" --raw_m5_parquet $TAPE/year=*/part-000.parquet
fi
echo "  FULL_PLUS_CTX ready (V10 builder computes the remaining ctx_cont; fail-closes if any missing)"

cat <<'CHECKLIST'

=================== STEPS 4-7 (guided — run + verify each; see FASE2B_REBUILD_ORDER.md) ===================
4. MTF_V2_CACHE:   python -m gx1.scripts.prebuild_multi_tf_cache_v2 ...        # verify last_ts==cutoff
5. BASE28 (fresh) -> base34 (CTX16CAT6) -> backfill:
     python -m gx1.scripts.backfill_base34_raw_m1_ohlcv_v1 --dry-run          # then --write (R12 volume)
6. BUILDS (explicit args; --canonical_v2_parquet=$WS/FULL_PLUS_CTX.parquet):
     build_entry_v10_ctx_training_dataset_v3 -> materialize_build_v3_training_dataset_v2 (EXIT_IO_V8)
     -> materialize_inference_batch_candidates_v3_v1 -> materialize_build_exit_iql_per_bar_dataset_v2_m1
7. RETRAIN (--vedtak fase2b_regime_v4_rebuild_20260605 on EACH; all gated):
     XGB -> V10 -> Entry-IQL -> V3 -> Exit-IQL -> gates (R13 parity + short-in-uptrend + -2000)
     -> cement on PASS -> REMOVE the REGIME_V4 flag + OFF/105/6 path (no off-switch).
CHECKLIST
echo "[fase2b_rebuild] steps 1-3 done/verified. Continue with 4-7 deliberately."
