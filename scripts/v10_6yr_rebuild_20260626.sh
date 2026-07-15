#!/usr/bin/env bash
# V10 EYES 6-year rebuild — vedtak entry_v10_6yr_retrain_20260626 (user: "Kjør HELE V10-rebuild NÅ uansett").
# Goal: retrain the SACRED V10 entry transformer on the FULL 6yr with the DOWN-REGIME GUARANTEED IN TRAIN
# (the cemented V10 may have held it out; dataset deleted so unverifiable). Mirrors scripts/fase2b_rebuild.sh
# EXACTLY for feature/contract parity (train==serve), but writes to a FRESH dir and NEVER re-pins (live untouched).
#
# train==serve target (cemented v10_bundle_clean/bundle_metadata.json, git f87cde48):
#   seq_input_dim=41 snap_input_dim=41 seq_len=96 ctx_cont_dim=142 ctx_cat_dim=5 num_classes=3
#   model_class=EntryV10CtxHybridTransformer  signal_bridge_id=XGB_SIGNAL_BRIDGE_V1  ctx_tag=CTX6CAT5
#   enable_regime_film=False enable_mtf_direction_head=False enable_pos_enc=True
#   epochs=10 batch_size=512 lr=3e-4 seed=1337 early_stopping_patience=10 min_delta=1e-4 ckpt_monitor=dir_acc
# Each heavy stage runs under gx1_capped_run.sh (cgroup OOM kills the JOB not the box; live=PID2157 stays up).
set -euo pipefail

export GX1_DATA=/home/andre2/GX1_DATA        # required by the V10 dataset builder (canonical tape lane resolver)
export GX1_REGIME_V4=1
export GX1_TREND_REGIME_FROM_D1=1
export GX1_PERTF_CLOSED_BAR=${GX1_PERTF_CLOSED_BAR:-1}
export GX1_V10_CKPT_MONITOR=dir_acc          # cement selected ckpt on dir_acc (default is val_loss) — train==serve
VEDTAK=entry_v10_6yr_retrain_20260626
ENG=/home/andre2/src/GX1_ENGINE

# rule 2: never run from a dirty tree (mirror fase2b_rebuild.sh:24)
[ -z "$(git -C "$ENG" status --short)" ] || { echo "[ABORT] git tree dirty — commit/stash first (rule 2)"; exit 1; }
PY=$ENG/.venv/bin/python
WS=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605
# Default to a fresh bodyfix+spreadfix workspace. The original v10_6yr_rebuild_20260626
# contains a pre-fix train parquet with corrupt body_pct snap/seq_last outliers, and the
# first bodyfix workspace baked spread_bps=0.0 despite bid/ask columns being present.
# Do not let skip-if-exists reuse either stale dataset for retrain. Override with
# REBUILD_DIR only for deliberate inspection/resume of a known-clean workspace.
REBUILD=${REBUILD_DIR:-$WS/v10_6yr_rebuild_20260626_spreadfix}
TAPE=/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL
BASE28=/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json
XGB=${XGB_BUNDLE_DIR:-/home/andre2/GX1_DATA/models/xgb_v7_base80_20260526_cpu_PROMOTED_20260708}
CAP="$ENG/scripts/gx1_capped_run.sh --mem 22G --swap 2G --"
SMART_SEQ_STRUCTURE_MANIFEST=${SMART_SEQ_STRUCTURE_MANIFEST:-/home/andre2/GX1_DATA/reports/entry_specialist_challenger_extension_manifest_20260630_v1/ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_MANIFEST_latest.json}
cd "$ENG"
mkdir -p "$REBUILD"

case "${GX1_MTF_CACHE_ALLOW_STALE:-0}" in
  1|true|TRUE|yes|YES)
    echo "[ABORT] GX1_MTF_CACHE_ALLOW_STALE must stay off for XAU direction-repair rebuild" >&2
    exit 1
    ;;
esac
[ -f "$SMART_SEQ_STRUCTURE_MANIFEST" ] || {
  echo "[ABORT] missing smart seq520 structure manifest: $SMART_SEQ_STRUCTURE_MANIFEST" >&2
  exit 1
}

TRAIN_VARIANT=${TRAIN_VARIANT:-xau_direction_repair}
case "$TRAIN_VARIANT" in
  xau_direction_repair)
    OUT_BUNDLE_DIR=${OUT_BUNDLE_DIR:-$REBUILD/v10_bundle_6yr_xau_direction_repair_smartctx}
    export GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=${GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES:-1}
    ;;
  baseline)
    OUT_BUNDLE_DIR=${OUT_BUNDLE_DIR:-$REBUILD/v10_bundle_6yr_baseline_smartctx}
    ;;
  symmetric_negatives)
    export GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1
    export ENTRY_SYMMETRIC_NEGATIVES=1
    OUT_BUNDLE_DIR=${OUT_BUNDLE_DIR:-$REBUILD/v10_bundle_6yr_symneg_smartctx}
    ;;
  *)
    echo "[ABORT] unknown TRAIN_VARIANT=$TRAIN_VARIANT (expected xau_direction_repair|baseline|symmetric_negatives)" >&2
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
$PY - <<PYEOF
import json
from pathlib import Path
summary_path = Path("$REBUILD/canonical_features_v2_summary.json")
if not summary_path.exists():
    raise RuntimeError(
        "[CANONICAL_V2_GATE_FAIL] missing no-lookahead summary; rebuild stage 1 in a fresh REBUILD_DIR"
    )
summary = json.loads(summary_path.read_text(encoding="utf-8"))
contract = summary.get("htf_alignment_contract_v1") if isinstance(summary.get("htf_alignment_contract_v1"), dict) else {}
if summary.get("canonical_v2_builder_version") != "canonical_features_v2_no_lookahead_close_time_20260713":
    raise RuntimeError(
        "[CANONICAL_V2_GATE_FAIL] stale canonical_v2 builder version; rebuild stage 1 "
        f"observed={summary.get('canonical_v2_builder_version')}"
    )
if contract.get("no_lookahead") is not True or contract.get("d1_feature_time") != "bar_close_time" or contract.get("m15_feature_time") != "bar_close_time":
    raise RuntimeError(f"[CANONICAL_V2_GATE_FAIL] HTF alignment contract is not no-lookahead: {contract}")
print("[GATE] canonical_features_v2 no-lookahead PASS:", summary_path)
PYEOF

# ---- STAGE 2: cv3 augment + glitch-guard (x10 price-scale must be FIXED) ----
if have "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"; then echo "[2] skip cv3"; else
  $CAP $PY -m gx1.scripts.materialize_canonical_v3_augment \
    --input "$REBUILD/canonical_features_v2.parquet" --output-dir "$REBUILD/cv3"
fi
$PY - <<PYEOF
import hashlib
import json
from pathlib import Path
cv2_path = Path("$REBUILD/canonical_features_v2.parquet").resolve()
manifest_path = Path("$REBUILD/cv3/CURRENT_MANIFEST.json")
if not manifest_path.exists():
    raise RuntimeError("[CANONICAL_V3_GATE_FAIL] missing CURRENT_MANIFEST.json")
h = hashlib.sha256()
with cv2_path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
cv2_sha = h.hexdigest()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if str(Path(str(manifest.get("source_v2_parquet") or "")).resolve()) != str(cv2_path):
    raise RuntimeError(
        "[CANONICAL_V3_GATE_FAIL] source_v2_parquet mismatch: "
        f"observed={manifest.get('source_v2_parquet')} expected={cv2_path}"
    )
if manifest.get("source_v2_parquet_sha256") != cv2_sha:
    raise RuntimeError(
        "[CANONICAL_V3_GATE_FAIL] source_v2 sha mismatch; rebuild stage 2 from current no-lookahead v2"
    )
if manifest.get("source_v2_no_lookahead") is not True:
    raise RuntimeError("[CANONICAL_V3_GATE_FAIL] source_v2_no_lookahead not proven true")
print("[GATE] canonical_v3 source provenance PASS:", manifest_path)
PYEOF
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
  $PY - <<PYEOF
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

cv2_path = Path("$REBUILD/canonical_features_v2.parquet").resolve()
cv3_path = Path("$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet").resolve()
full_path = Path("$REBUILD/FULL_PLUS_CTX_v3src.parquet").resolve()
cv2_summary = json.loads(Path("$REBUILD/canonical_features_v2_summary.json").read_text(encoding="utf-8"))
cv3_manifest = json.loads(Path("$REBUILD/cv3/CURRENT_MANIFEST.json").read_text(encoding="utf-8"))
proof = {
    "schema_version": "full_plus_ctx_v3src_xau_direction_repair_proof_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "canonical_v2_parquet": str(cv2_path),
    "canonical_v2_sha256": sha(cv2_path),
    "canonical_v2_builder_version": cv2_summary.get("canonical_v2_builder_version"),
    "canonical_v2_htf_alignment_contract": cv2_summary.get("htf_alignment_contract_v1"),
    "canonical_v3_parquet": str(cv3_path),
    "canonical_v3_sha256": sha(cv3_path),
    "canonical_v3_source_v2_sha256": cv3_manifest.get("source_v2_parquet_sha256"),
    "canonical_v3_source_v2_no_lookahead": cv3_manifest.get("source_v2_no_lookahead"),
    "full_plus_ctx_parquet": str(full_path),
    "full_plus_ctx_sha256": sha(full_path),
}
Path("$REBUILD/FULL_PLUS_CTX_v3src.proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("[GATE] wrote FULL_PLUS_CTX proof:", "$REBUILD/FULL_PLUS_CTX_v3src.proof.json")
PYEOF
fi

$PY - <<PYEOF
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

p="$REBUILD/FULL_PLUS_CTX_v3src.parquet"
proof_path=Path("$REBUILD/FULL_PLUS_CTX_v3src.proof.json")
if not proof_path.exists():
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] missing proof json; rebuild stage 3 in a fresh REBUILD_DIR")
proof=json.loads(proof_path.read_text(encoding="utf-8"))
cv2=Path("$REBUILD/canonical_features_v2.parquet").resolve()
cv3=Path("$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet").resolve()
if proof.get("canonical_v2_sha256") != sha(cv2):
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] canonical_v2 sha mismatch in proof")
if proof.get("canonical_v3_sha256") != sha(cv3):
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] canonical_v3 sha mismatch in proof")
if proof.get("full_plus_ctx_sha256") != sha(Path(p)):
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] FULL_PLUS_CTX sha mismatch in proof")
if (proof.get("canonical_v2_htf_alignment_contract") or {}).get("no_lookahead") is not True:
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] canonical_v2 no-lookahead not proven in proof")
if proof.get("canonical_v3_source_v2_no_lookahead") is not True:
    raise RuntimeError("[FULL_PLUS_CTX_GATE_FAIL] canonical_v3 source no-lookahead not proven in proof")
d=pd.read_parquet(p, columns=["bid_close","ask_close","spread_bps"])
spread=d["spread_bps"].to_numpy(dtype=float)
bid=d["bid_close"].to_numpy(dtype=float)
ask=d["ask_close"].to_numpy(dtype=float)
derived=(ask-bid)/np.maximum(bid,1e-9)*1e4
derived=np.maximum(np.where(np.isfinite(derived), derived, 0.0), 0.0)
if float(np.nanstd(spread)) <= 1e-9 and float(np.nanpercentile(derived,95)) > 0.0:
    raise RuntimeError(
        "[SPREAD_BPS_GATE_FAIL] FULL_PLUS_CTX has constant spread_bps despite bid/ask signal; "
        "use a fresh REBUILD_DIR or rebuild stage 3"
    )
print("[GATE] FULL_PLUS_CTX spread_bps:", p, "mean=", float(np.nanmean(spread)),
      "p95=", float(np.nanpercentile(spread,95)), "std=", float(np.nanstd(spread)))
PYEOF

# ---- STAGE 4: MTF-v2 cache — REBUILD FRESH from new cv3 (existing caches end 06-08/05-25, stale vs build
#      data 2026-06-26; the fail-closed freshness guard (max_lag 2d) would crash the dataset build). ----
export GX1_V10_MULTI_TF_V2_CACHE_DIR="$REBUILD/MULTI_TF_V2_CACHE"
if have "$GX1_V10_MULTI_TF_V2_CACHE_DIR/manifest.json"; then echo "[4] skip MTF cache (fresh exists)"; else
  $CAP $PY -m gx1.scripts.prebuild_multi_tf_cache_v2 \
    --m5-prebuilt "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet" \
    --out-dir "$GX1_V10_MULTI_TF_V2_CACHE_DIR"
fi
# do NOT set GX1_MTF_CACHE_ALLOW_STALE — that would forward-fill (freeze) the recent down-regime ctx_cont.
$PY - <<PYEOF
import hashlib
import json
from pathlib import Path
import pandas as pd
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2, MULTI_TF_SHIFT
manifest_path = Path("$GX1_V10_MULTI_TF_V2_CACHE_DIR") / "manifest.json"
expected_source_path = Path("$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet").resolve()
expected_source = str(expected_source_path)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
if not manifest_path.exists():
    raise RuntimeError(f"[MTF_CACHE_GATE_FAIL] missing manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
expected_features = list(MULTI_TF_PER_BAR_FEATURES_V2)
observed_features = [str(x) for x in (manifest.get("feature_names") or [])]
if observed_features != expected_features:
    raise RuntimeError(
        "[MTF_CACHE_GATE_FAIL] feature_names mismatch: "
        f"observed={observed_features or '<missing>'} expected={expected_features}"
    )
expected_shift = {tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()}
observed_shift = manifest.get("shift_contract") if isinstance(manifest.get("shift_contract"), dict) else {}
if observed_shift != expected_shift:
    raise RuntimeError(
        "[MTF_CACHE_GATE_FAIL] shift_contract mismatch: "
        f"observed={observed_shift or '<missing>'} expected={expected_shift}"
    )
observed_source = str(Path(str(manifest.get("m5_prebuilt_source") or "")).resolve())
if observed_source != expected_source:
    raise RuntimeError(
        "[MTF_CACHE_GATE_FAIL] m5_prebuilt_source mismatch: "
        f"observed={observed_source} expected={expected_source}"
    )
expected_sha = sha(expected_source_path)
observed_sha = str(manifest.get("m5_prebuilt_source_sha256") or "").strip()
if observed_sha != expected_sha:
    raise RuntimeError(
        "[MTF_CACHE_GATE_FAIL] m5_prebuilt_source_sha256 mismatch: "
        f"observed={observed_sha or '<missing>'} expected={expected_sha}"
    )
tfs = manifest.get("tfs") if isinstance(manifest.get("tfs"), dict) else {}
last_ts = [int(row.get("last_ts_ns") or 0) for row in tfs.values() if isinstance(row, dict)]
test_end_ns = int(pd.Timestamp("$TEST_END").timestamp() * 1_000_000_000)
if not last_ts or max(last_ts) < test_end_ns:
    raise RuntimeError(
        "[MTF_CACHE_GATE_FAIL] cache does not cover TEST_END: "
        f"max_last={max(last_ts) if last_ts else None} test_end_ns={test_end_ns}"
    )
print("[GATE] MTF cache fresh:", manifest_path, "max_last_ns=", max(last_ts), "test_end_ns=", test_end_ns)
PYEOF

DATASET_DIR=${DATASET_DIR:-$REBUILD/v10_dataset_6yr_smartctx_xau_direction_repair}
SMART520_RANK_REFERENCE_NPZ=${SMART520_RANK_REFERENCE_NPZ:-$REBUILD/smart520_rank_reference_xau_direction_repair.npz}

# ---- STAGE 5: V10 ctx_v3 6yr dataset (time_split; DOWN-REGIME IN TRAIN) ----
export GX1_ENTRY_DIRECTION_TARGET_MODE=${GX1_ENTRY_DIRECTION_TARGET_MODE:-path_utility_v2}
export GX1_ENTRY_DIRECTION_UTILITY_MFE_WEIGHT=${GX1_ENTRY_DIRECTION_UTILITY_MFE_WEIGHT:-0.35}
export GX1_ENTRY_DIRECTION_UTILITY_MAE_WEIGHT=${GX1_ENTRY_DIRECTION_UTILITY_MAE_WEIGHT:-1.15}
export GX1_ENTRY_DIRECTION_UTILITY_PATH_WEIGHT=${GX1_ENTRY_DIRECTION_UTILITY_PATH_WEIGHT:-0.25}
export GX1_ENTRY_DIRECTION_UTILITY_MIN_BPS=${GX1_ENTRY_DIRECTION_UTILITY_MIN_BPS:-15.0}
export GX1_ENTRY_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS=${GX1_ENTRY_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS:-4.0}
export GX1_SMART_SELECTION_SCORE=${GX1_SMART_SELECTION_SCORE:-expected_utility}
export GX1_ENTRY_EXPECTED_UTILITY_THRESHOLD_BPS=${GX1_ENTRY_EXPECTED_UTILITY_THRESHOLD_BPS:-0.0}
export ENTRY_PRED_BALANCE_ALPHA=${ENTRY_PRED_BALANCE_ALPHA:-0.50}
export ENTRY_PRED_BALANCE_TARGET=${ENTRY_PRED_BALANCE_TARGET:-label}
export ENTRY_PRED_BALANCE_CLASS_WEIGHTS=${ENTRY_PRED_BALANCE_CLASS_WEIGHTS:-1.0,1.0,4.0}
export ENTRY_DIRECTION_CE_SCALE=${ENTRY_DIRECTION_CE_SCALE:-2.00}
export GX1_V10_CKPT_MONITOR=${GX1_V10_CKPT_MONITOR:-dir_acc}
export ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT=${ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT:-0.50}
export ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=${ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL:-0.35}
export ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE=${ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE:-0.05}
export ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=${ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT:-2.50}
export ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION=${ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION:-0.50}
export ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR=${ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR:-0.05}
export ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=${ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE:-0.20}
export ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT=${ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT:-4.00}
export ENTRY_DIRECTION_VS_FLAT_MARGIN=${ENTRY_DIRECTION_VS_FLAT_MARGIN:-0.10}
export ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=${ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT:-4.00}
export ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=${ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS:-15.0}
export ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=${ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN:-0.10}
export ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT:-6.00}
export ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS:-15.0}
export ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN:-0.10}
export ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT=${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT:-8.00}
export ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS=${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS:-15.0}
export ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS=${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS:-0.0}
export ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH=${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH:-0.50}
export ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN=${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN:-0.10}
export ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT=${ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT:-8.00}
export ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS=${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS:-15.0}
export ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS=${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS:-0.0}
export ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH=${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH:-0.50}
export ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP=${ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP:-4.0}
export ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION=${ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION:-1}
export ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP=${ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP:-0.18}
export ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=${ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL:-1}
export ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=${ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT:-8.00}
export ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=${ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:-0.10}
export ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=${ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS:-8}
export ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=${ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION:-0.50}
export ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=${ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR:-0.10}
export ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=${ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN:-0.10}
export ENTRY_HIER_LEGACY_CE_MULT=${ENTRY_HIER_LEGACY_CE_MULT:-1.00}
export ENTRY_HIER_TRADE_WEIGHT=${ENTRY_HIER_TRADE_WEIGHT:-2.00}
export ENTRY_HIER_SIDE_WEIGHT=${ENTRY_HIER_SIDE_WEIGHT:-1.75}
export ENTRY_HIER_UTILITY_WEIGHT=${ENTRY_HIER_UTILITY_WEIGHT:-1.00}
export ENTRY_HIER_BAD_PATH_WEIGHT=${ENTRY_HIER_BAD_PATH_WEIGHT:-1.25}
export ENTRY_HIER_MAE_WEIGHT=${ENTRY_HIER_MAE_WEIGHT:-0.35}
export ENTRY_HIER_SIDE_VALIDITY_WEIGHT=${ENTRY_HIER_SIDE_VALIDITY_WEIGHT:-1.50}
export ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS=${ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS:-15.0}
export ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP=${ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP:-8.0}
export ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT=${ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT:-4.00}
export ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE=${ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE:-0.02}
export ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=${ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:-0.10}
export ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT=${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT:-4.00}
export ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE=${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE:-0.02}
export ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE=${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE:-0.10}
export ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS=${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS:-8}
export ENTRY_HIER_SLICE_SIDE_CE_WEIGHT=${ENTRY_HIER_SLICE_SIDE_CE_WEIGHT:-4.00}
export ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT=${ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT:-3.00}
export ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN=${ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN:-0.10}
export ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE=${ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE:-0.10}
export ENTRY_HIER_SLICE_SIDE_MIN_ROWS=${ENTRY_HIER_SLICE_SIDE_MIN_ROWS:-8}
export ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT=${ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT:-4.00}
export ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE=${ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE:-0.02}
export ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=${ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:-0.10}
export ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT=${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT:-4.00}
export ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE=${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE:-0.02}
export ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE=${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE:-0.10}
export ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS=${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS:-8}
export ENTRY_HIER_POCKET_ABSTAIN_WEIGHT=${ENTRY_HIER_POCKET_ABSTAIN_WEIGHT:-5.00}
export ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT=${ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT:-3.00}
export ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS=${ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS:-30.0}
export ENTRY_TRENDLINE_RAIL_AUX_WEIGHT=${ENTRY_TRENDLINE_RAIL_AUX_WEIGHT:-1.00}
export ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=${ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT:-1.50}
export ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT=${ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT:-1.50}
export ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT=${ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT:-1.75}
export ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=${ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT:-5.00}
export ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT=${ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT:-4.00}
export ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT=${ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT:-3.00}
export ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT=${ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT:-5.00}
export ENTRY_TRENDLINE_RAIL_MARGIN=${ENTRY_TRENDLINE_RAIL_MARGIN:-1.00}
export ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS=${ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS:-30.0}
export GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=${GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH:-20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH}
echo "[5] direction target mode: $GX1_ENTRY_DIRECTION_TARGET_MODE"
if have "$SMART520_RANK_REFERENCE_NPZ"; then
  echo "[5a] skip smart520 rank reference: $SMART520_RANK_REFERENCE_NPZ"
else
  $PY -m gx1.scripts.materialize_smart520_rank_reference_v1 \
    --source-parquet "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --out "$SMART520_RANK_REFERENCE_NPZ" \
    --model-range-start "$TRAIN_START" \
    --reference-end "$TEST_END"
fi
if ls "$DATASET_DIR/"*train*.parquet >/dev/null 2>&1; then echo "[5] skip v10 dataset (smartctx exists)"; else
  mkdir -p "$DATASET_DIR"
  # XAU direction-repair entry dataset: build from the cv3-derived FULL_PLUS_CTX
  # but neutralize the XGB bridge so the new model learns side/flat from smart
  # sequence, MTF and geometry evidence instead of inheriting an XGB direction anchor.
  $CAP $PY -m gx1.scripts.build_entry_v10_ctx_training_dataset_v3 \
    --source-parquet-override "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --xgb-feature-contract-path gx1/xgb/contracts/xgb_input_features_base80_v1.json \
    --xgb-sanitizer-config-path gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json \
    --xgb_bundle "$XGB" \
    --neutral-xgb-bridge \
    --canonical_v2_parquet "$REBUILD/FULL_PLUS_CTX_v3src.parquet" \
    --seq-structure-manifest "$SMART_SEQ_STRUCTURE_MANIFEST" \
    --seq-structure-compute-inline \
    --output "$DATASET_DIR/v10_6yr_dataset.parquet" \
    --start 2020-11-09 --end 2026-06-14 \
    --hold-bars 3 \
    --allow-missing-hold-map \
    --time_split \
    --train_start $TRAIN_START --train_end $TRAIN_END \
    --val_start $VAL_START --val_end $VAL_END \
    --test_start $TEST_START --test_end $TEST_END \
    --smart520-rank-reference-npz "$SMART520_RANK_REFERENCE_NPZ" \
    --seq_len 96
fi

# Direction-repair heads must never train from a stale dataset that lacks the
# pocket labels used to learn "support continuation -> SHORT bad path" and the
# inverse resistance/long trap. This is a training/audit gate, not a live rule.
$PY - <<PYEOF
import glob
import json
import os
import hashlib
import pyarrow.parquet as pq

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

required = {
    "y_trade",
    "y_side",
    "y_side_mask",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
    "y_rising_channel_support_touch",
    "y_falling_channel_resistance_touch",
    "y_support_retest_continuation",
    "y_resistance_retest_continuation",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
    "y_mtf_conflict_m5_vs_higher_side",
    "y_long_high_mae_low_mfe_early_failure",
    "y_short_high_mae_low_mfe_early_failure",
}
required_state_contract = {
    "schema_version",
    "frame_anchor_utc",
    "model_range_start_utc",
    "rank_reference_end_utc",
    "rank_reference_npz",
    "rank_reference_npz_sha256",
}
files_by_split = {}
for split in ("train", "val", "test"):
    matches = sorted(glob.glob("$DATASET_DIR/" + f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] expected exactly one "
            + split
            + " parquet under $DATASET_DIR, got "
            + repr(matches)
        )
    files_by_split[split] = matches[0]
proof_path = "$DATASET_DIR/DATASET_BUILD_PROOF.json"
with open(proof_path, "r", encoding="utf-8") as fh:
    proof = json.load(fh)
if proof.get("neutral_xgb_bridge") is not True:
    raise RuntimeError(
        "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] neutral_xgb_bridge is not true in "
        + proof_path
    )
xgb_bridge_source = str(proof.get("xgb_bridge_source") or "")
if xgb_bridge_source != "neutral_uniform_proba":
    raise RuntimeError(
        "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] xgb_bridge_source is not neutral_uniform_proba in "
        + proof_path
        + f": {xgb_bridge_source!r}"
    )
tape_root = str(proof.get("tape_root") or "").lower()
if "xauusd" not in tape_root:
    raise RuntimeError(
        "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] tape_root is not XAUUSD-only in "
        + proof_path
        + f": {tape_root!r}"
    )
expected_rank_ref = os.path.realpath("$SMART520_RANK_REFERENCE_NPZ")
if not os.path.isfile(expected_rank_ref):
    raise RuntimeError("[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] expected smart520 rank reference missing: " + expected_rank_ref)
expected_rank_ref_sha = _sha256(expected_rank_ref)
for split, parquet_path in files_by_split.items():
    cols = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] "
            + split
            + " stale dataset lacks repair labels: "
            + ",".join(missing)
            + " ; use a fresh DATASET_DIR or rebuild stage 5"
        )
    manifest_path = parquet_path.replace(".parquet", ".manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        split_manifest = json.load(fh)
    extra = split_manifest.get("extra") if isinstance(split_manifest.get("extra"), dict) else {}
    state_contract = extra.get("smart520_state_contract") if isinstance(extra.get("smart520_state_contract"), dict) else {}
    missing_state = sorted(required_state_contract - set(state_contract))
    if missing_state:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] "
            + split
            + " missing smart520_state_contract fields: "
            + ",".join(missing_state)
        )
    rank_ref = os.path.realpath(str(state_contract.get("rank_reference_npz") or ""))
    if rank_ref != expected_rank_ref:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] "
            + split
            + " smart520 rank reference mismatch: "
            + rank_ref
            + " != "
            + expected_rank_ref
        )
    if not os.path.isfile(rank_ref):
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] smart520 rank reference missing: "
            + rank_ref
        )
    observed_sha = str(state_contract.get("rank_reference_npz_sha256") or "").strip().lower()
    if observed_sha != expected_rank_ref_sha:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_DATASET_GATE_FAIL] "
            + split
            + " smart520 rank reference sha mismatch: "
            + observed_sha
            + " != "
            + expected_rank_ref_sha
        )
print("[GATE] XAU direction-repair dataset labels PASS:", files_by_split["train"])
PYEOF

$PY -m gx1.scripts.audit_xau_direction_repair_pretrain_v1 \
  --dataset-dir "$DATASET_DIR" \
  --stem "v10_6yr_dataset__HOLD_03B" \
  --out-dir "$GX1_DATA/reports/xau_direction_repair_pretrain_audit_20260713_v1" \
  --data-splits train,val,test \
  --require-rail-features \
  --fail-on-audit-fail \
  --quiet

# ---- GATE: verify contract dims BEFORE the SACRED GPU train (train==serve) ----
$PY - <<PYEOF
import glob
import numpy as np
import pyarrow.parquet as pq
from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
f=sorted(glob.glob("$DATASET_DIR/*train*.parquet"))[0]
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
    --dataset_dir "$DATASET_DIR" \
    --m5-prebuilt-path "$REBUILD/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet" \
    --out_bundle_dir "$OUT_BUNDLE_DIR" \
    --seq_len 96 --epochs 10 --batch_size 512 --lr 3e-4 --seed 1337 \
    --early-stopping-patience 10 --early-stopping-min-delta 1e-4 \
    --enable-xau-direction-repair-heads \
    --anchor-gate-init 0.0 \
    --num-workers 8
  # AUDIT-FIX: --vedtak (rule 3 fail-closed) + --m5-prebuilt-path (required)
  # + neutral bridge + explicit XAU direction-repair recipe/anchor-gate.
else
  echo "[6] SACRED train GUARDED (set RUN_TRAIN=1 after dim-gate verified)"
fi
echo "===== data-prep + dataset stages complete; verify gate then RUN_TRAIN=1 ====="
