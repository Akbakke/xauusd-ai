#!/usr/bin/env bash
# V10 EYES 6-year rebuild — vedtak entry_v10_6yr_retrain_20260626 (user: "Kjør HELE V10-rebuild NÅ uansett").
# Goal: retrain the SACRED V10 entry transformer on the FULL 6yr with the DOWN-REGIME GUARANTEED IN TRAIN
# (the cemented V10 may have held it out; dataset deleted so unverifiable). Mirrors scripts/fase2b_rebuild.sh
# EXACTLY for feature/contract parity (train==serve), but writes to a FRESH dir and NEVER re-pins (live untouched).
#
# train==serve target (cemented v10_bundle_clean/bundle_metadata.json, git f87cde48):
#   seq_input_dim=41 snap_input_dim=41 seq_len=96 ctx_cont_dim=123 ctx_cat_dim=5 num_classes=3
#   model_class=EntryV10CtxHybridTransformer  signal_bridge_id=XGB_SIGNAL_BRIDGE_V1  ctx_tag=CTX6CAT5
#   enable_regime_film=False enable_mtf_direction_head=False enable_pos_enc=True
#   epochs=10 batch_size=512 lr=3e-4 seed=1337 early_stopping_patience=10 min_delta=1e-4 ckpt_monitor=dir_acc
# Each heavy stage runs under gx1_capped_run.sh (cgroup OOM kills the JOB not the box; live=PID2157 stays up).
set -euo pipefail

export GX1_DATA=/home/andre2/GX1_DATA        # required by the V10 dataset builder (canonical tape lane resolver)
export GX1_REGIME_V4=1
export GX1_TREND_REGIME_FROM_D1=1
export GX1_V10_CKPT_MONITOR=dir_acc          # cement selected ckpt on dir_acc (default is val_loss) — train==serve
VEDTAK=entry_v10_6yr_retrain_20260626

# rule 2: never run from a dirty tree (mirror fase2b_rebuild.sh:24)
[ -z "$(git status --short)" ] || { echo "[ABORT] git tree dirty — commit/stash first (rule 2)"; exit 1; }
ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
WS=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605
# Default to a fresh bodyfix workspace. The original v10_6yr_rebuild_20260626
# contains a pre-fix train parquet with corrupt body_pct snap/seq_last outliers;
# do not let skip-if-exists reuse it for retrain. Override with REBUILD_DIR only
# for deliberate inspection/resume of a known-clean workspace.
REBUILD=${REBUILD_DIR:-$WS/v10_6yr_rebuild_20260626_bodyfix}
TAPE=/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL
BASE28=/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json
XGB=$WS/xgb_v7
CAP="$ENG/scripts/gx1_capped_run.sh --mem 22G --swap 2G --"
cd "$ENG"
mkdir -p "$REBUILD"

TRAIN_VARIANT=${TRAIN_VARIANT:-baseline}
case "$TRAIN_VARIANT" in
  baseline)
    OUT_BUNDLE_DIR=${OUT_BUNDLE_DIR:-$REBUILD/v10_bundle_6yr_baseline_clean}
    ;;
  symmetric_negatives)
    export GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1
    export ENTRY_SYMMETRIC_NEGATIVES=1
    OUT_BUNDLE_DIR=${OUT_BUNDLE_DIR:-$REBUILD/v10_bundle_6yr_symneg}
    ;;
  *)
    echo "[ABORT] unknown TRAIN_VARIANT=$TRAIN_VARIANT (expected baseline|symmetric_negatives)" >&2
    exit 1
    ;;
esac

# Down-regime IN train; small honest OOT tail held out.
TRAIN_START=2020-11-09T00:00:00Z ; TRAIN_END=2026-04-30T23:59:59Z
VAL_START=2026-05-01T00:00:00Z   ; VAL_END=2026-05-20T23:59:59Z
TEST_START=2026-05-21T00:00:00Z  ; TEST_END=2026-06-14T23:59:59Z

have() { [ -e "$1" ]; }
echo "===== V10 6yr rebuild ($VEDTAK) -> $REBUILD ====="

# ---- STAGE 1: canonical_features_v2 (clean M5 feature tape) ----
if have "$REBUILD/canonical_features_v2.parquet"; then echo "[1] skip canonical_features_v2"; else
  $CAP $PY -m gx1.scripts.materialize_build_canonical_features_v2 \
    --out-path "$REBUILD/canonical_features_v2.parquet"
fi

# ---- STAGE 2: cv3 augment + glitch-guard (x10 price-scale must be FIXED) ----
if have "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"; then echo "[2] skip cv3"; else
  $CAP $PY -m gx1.scripts.materialize_canonical_v3_augment \
    --input "$REBUILD/canonical_features_v2.parquet" --output-dir "$REBUILD/cv3"
fi
$PY - <<PYEOF
import glob,pandas as pd
from gx1.io.price_glitch_guard import detect_price_scale_glitch, assert_no_price_scale_glitch
f=sorted(glob.glob("$REBUILD/cv3/*CANONICAL_V3*.parquet"))[0]
d=pd.read_parquet(f); dd=d.reset_index() if "time" not in d.columns else d
assert len(detect_price_scale_glitch(dd))==0, "[GLITCH] x10 NOT fixed"
assert_no_price_scale_glitch(dd); print("  [glitch-guard] PASS",f)
PYEOF

# ---- STAGE 3: FULL_PLUS_CTX (ctx_cont on CV3 trimmed to model range >=2020-11-09) ----
# NOTE: source from cv3, NOT the pre-augment canonical_features_v2 — the SMC/cyclic/m5h1 source features
# the V10 builder's canonical_v3 contract requires are ADDED in the cv3 augment (fase2b recipe is stale here).
if have "$REBUILD/FULL_PLUS_CTX_v3src.parquet"; then echo "[3] skip FULL_PLUS_CTX_v3src"; else
  $PY - <<PYEOF
import glob,pandas as pd
def load_t(f):
    d=pd.read_parquet(f)
    d.index=pd.to_datetime(d["time"],utc=True) if "time" in d.columns else pd.to_datetime(d.index,utc=True)
    return d
v2=load_t("$REBUILD/canonical_features_v2.parquet")
cv3=load_t(sorted(glob.glob("$REBUILD/cv3/*CANONICAL_V3*.parquet"))[0])
# cv3 is AUTHORITATIVE (canonical_v3 augment); restore the columns the augment dropped (incl real 'atr')
# from v2 by time so add_ctx_cont + the V10 builder have the full source the cement had. cv3 wins on shared cols.
extra=[c for c in v2.columns if c not in cv3.columns and c!="time"]
for c in extra: cv3[c]=v2[c].reindex(cv3.index)
cv3=cv3[cv3.index>=pd.Timestamp("2020-11-09",tz="UTC")].copy()
if "time" not in cv3.columns: cv3["time"]=cv3.index
print(f"[STAGE3] cv3_modelrange rows={len(cv3)} cols={cv3.shape[1]} restored_from_v2={extra}")
assert cv3["atr"].notna().all(), "atr still missing/NaN after restore"
cv3.reset_index(drop=True).to_parquet("$REBUILD/cv3_modelrange.parquet",index=False)
PYEOF
  $CAP $PY -m gx1.scripts.add_ctx_cont_columns_to_prebuilt \
    --prebuilt_parquet "$REBUILD/cv3_modelrange.parquet" \
    --output_parquet "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --ctx-cont-dim 16 --ctx-cat-dim 5 \
    --tape-root "$TAPE" --raw_m5_parquet $TAPE/year=*/part-000.parquet
fi

# ---- STAGE 4: MTF-v2 cache — REBUILD FRESH from new cv3 (existing caches end 06-08/05-25, stale vs build
#      data 2026-06-26; the fail-closed freshness guard (max_lag 2d) would crash the dataset build). ----
export GX1_V10_MULTI_TF_V2_CACHE_DIR="$REBUILD/MULTI_TF_V2_CACHE"
if have "$GX1_V10_MULTI_TF_V2_CACHE_DIR/manifest.json"; then echo "[4] skip MTF cache (fresh exists)"; else
  $CAP $PY -m gx1.scripts.prebuild_multi_tf_cache_v2 \
    --m5-prebuilt "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet" \
    --out-dir "$GX1_V10_MULTI_TF_V2_CACHE_DIR"
fi
# do NOT set GX1_MTF_CACHE_ALLOW_STALE — that would forward-fill (freeze) the recent down-regime ctx_cont.

# ---- STAGE 5: V10 ctx_v3 6yr dataset (time_split; DOWN-REGIME IN TRAIN) ----
if ls "$REBUILD/v10_dataset_6yr/"*train*.parquet >/dev/null 2>&1; then echo "[5] skip v10 dataset (HOLD_03B exists)"; else
  mkdir -p "$REBUILD/v10_dataset_6yr"
  # CEMENT-FAITHFUL (proven from v10_dataset_m5.log): build via --source-parquet-override on the
  # cv3-derived FULL_PLUS_CTX (M5-cadence; is_model_bar filter skipped; is_ASIA auto-derived), NOT
  # --base28_manifest (that M1 base34 path is the EXIT serve lane, missing 15 base80 feats).
  $CAP $PY -m gx1.scripts.build_entry_v10_ctx_training_dataset_v3 \
    --source-parquet-override "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --xgb-feature-contract-path gx1/xgb/contracts/xgb_input_features_base80_v1.json \
    --xgb-sanitizer-config-path gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json \
    --xgb_bundle "$XGB" \
    --canonical_v2_parquet "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --output "$REBUILD/v10_dataset_6yr/v10_6yr_dataset.parquet" \
    --start 2020-11-09 --end 2026-06-14 \
    --hold-bars 3 \
    --time_split \
    --train_start $TRAIN_START --train_end $TRAIN_END \
    --val_start $VAL_START --val_end $VAL_END \
    --test_start $TEST_START --test_end $TEST_END \
    --seq_len 96
fi

# ---- GATE: verify contract dims BEFORE the SACRED GPU train (train==serve) ----
$PY - <<PYEOF
import glob
import numpy as np
import pyarrow.parquet as pq
from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
f=sorted(glob.glob("$REBUILD/v10_dataset_6yr/*train*.parquet"))[0]
pf=pq.ParquetFile(f)
body_idx=ORDERED_SEQ_FIELDS_V3.index("body_pct")
max_body=0.0
bad_rows=0
for batch in pf.iter_batches(batch_size=8192, columns=["snap"]):
    snap=_stack_list_column(batch.to_pandas()["snap"], np.float32)
    vals=snap[:, body_idx]
    max_body=max(max_body, float(np.nanmax(vals)))
    bad_rows += int(((vals < -1e-6) | (vals > 1.000001) | ~np.isfinite(vals)).sum())
if bad_rows:
    raise RuntimeError(f"[BODY_PCT_GATE_FAIL] train snap body_pct out of [0,1]: bad_rows={bad_rows} max={max_body}")
print("[GATE] v10 train dataset:",f,"rows=",pf.metadata.num_rows,"body_pct_max=",max_body)
# NOTE(audit): seq/ctx dims are also asserted by EntryV10CtxDataset/trainer before training.
PYEOF

# ---- STAGE 6: train V10 transformer (SACRED) — GUARDED: only runs with RUN_TRAIN=1 ----
if [ "${RUN_TRAIN:-0}" = "1" ]; then
  $ENG/scripts/gx1_capped_run.sh --mem 30G --swap 2G -- \
  $PY -m gx1.models.entry_v10.entry_v10_ctx_train_v3 --train \
    --vedtak "$VEDTAK" \
    --dataset_dir "$REBUILD/v10_dataset_6yr" \
    --m5-prebuilt-path "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet" \
    --out_bundle_dir "$OUT_BUNDLE_DIR" \
    --seq_len 96 --epochs 10 --batch_size 512 --lr 3e-4 --seed 1337 \
    --early-stopping-patience 10 --early-stopping-min-delta 1e-4 \
    --num-workers 8
  # AUDIT-RESOLVED: train_recipe weights are trainer DEFAULTS (entry_v10_ctx_train_v3.py:234-354), parity holds.
  # AUDIT-FIX: --vedtak (rule 3 fail-closed) + --m5-prebuilt-path (required) + GX1_V10_CKPT_MONITOR=dir_acc added.
else
  echo "[6] SACRED train GUARDED (set RUN_TRAIN=1 after dim-gate verified)"
fi
echo "===== data-prep + dataset stages complete; verify gate then RUN_TRAIN=1 ====="
