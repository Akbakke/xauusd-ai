#!/usr/bin/env python3
"""Inference-batch candidate generator for V10 v2 + XGB v4.

Bypasses the full replay/policy/runtime stack and directly runs V10 v2 on every
M5 bar in canonical_v2 prebuilt to emit candidate parquet rows compatible with
the V3 dataset builder.

Pipeline
--------
1. Load canonical_v2 prebuilt (314k M5 rows × 136 cols, time-indexed).
2. Run XGB v4 per session → 7-dim signal_bridge_v3 per bar.
3. Build (n_m5, 37) per-bar SEQ matrix: 7 XGB-bridge + 30 PER_BAR_PRICE_STATE.
4. Build (n_m5, 43) ctx_cont and (n_m5, 6) ctx_cat matrices.
5. For each M5 bar with ≥96 prior bars: forward V10 v2 in batches.
6. Apply entry threshold (margin > thresh AND directional > flat).
7. Emit per-(M5 bar) candidate row.
8. Split rows by truth-week (Mon→Mon) and write per-week
   `shadow_meta_candidates_{run_id}_MERGED.parquet`
   into `/home/andre2/GX1_DATA/reports/truth_e2e_sanity/<run_id>/`.

Designed for V3 dataset builder consumption — only the columns V3 actually
reads are populated; the rest are placeholders (null/0/empty).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force v3 contract — this script materializes v3 inference batches and the
# v3 stack (V10 v3 + signal_bridge_v3) is the active baseline for wave 2.
os.environ.setdefault("GX1_SIGNAL_BRIDGE_VERSION", "3")

from gx1.contracts.signal_bridge_v3 import (
    PER_BAR_PRICE_STATE_FIELDS_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_BRIDGE_FIELDS_V3,
    SEQ_SIGNAL_DIM_V3,
    CTX_CONT_DIM_V3,
    CTX_CAT_DIM_V3,
    DEFAULT_SEQ_LEN_V3,
)
from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from gx1.time.session_detector import (
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
)
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


ACTION = "INFERENCE_BATCH_CANDIDATES_V3"

DEFAULT_PREBUILT_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_FULL_PLUS_CTX_2020_2026.parquet"
)
DEFAULT_XGB_BUNDLE = Path(
    # V12-cascade bundle. ONE truth = PROJECT_STATE_artifacts.json (resolve via
    # gx1_guards.load_decision_artifact). CURRENT_BUNDLES.md is SUPERSEDED. (the 081604Z_1000est
    # retrain was deleted 2026-05-11 — it was never used by V12 cascade).
    "/home/andre2/GX1_DATA/models/models/xgb_v7_base80_20260526T052210Z"
)
# V12.2 (2026-05-15): default updated to V10 v_FIXED multi-TF.
# v3+ chain (2026-05-19): default now V10 V3PLUS_v2 (4 aux heads + multi-TF cement).
# Script HARD-REQUIRES multi-TF V10; non-multi-TF V10 v3 baseline is deprecated.
DEFAULT_V10_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/models/entry_v10_ctx/ENTRY_V10_V3PLUS_v2_20260518T135516Z"
)
DEFAULT_XGB_FEATURE_CONTRACT = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base80_v1.json"
DEFAULT_XGB_SANITIZER_CONFIG = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base80_v1.json"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
# Multi-TF V2 cache (5 TFs incl M5 × 25 feats) — same one-truth source as rescore
# + the V10/V3 training builders. The V2-multi-TF V10 was trained on this.
DEFAULT_V2_CACHE_DIR = Path("/home/andre2/GX1_DATA/data/data/prebuilt/MULTI_TF_V2_CACHE")

DEFAULT_BATCH_SIZE = 256
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MIN_DIRECTIONAL_PROB = 0.42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truth_week_run_id(ts: pd.Timestamp) -> str:
    """Map a timestamp to the TRUTH_MONFRI_WEEK_<MonStart>_<NextMon> run_id.

    Truth weeks span Monday 00:00 → next Monday 00:00 (UTC).
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    weekday = ts.weekday()  # Monday=0
    monday = (ts - timedelta(days=weekday)).normalize()
    next_monday = monday + timedelta(days=7)
    return f"TRUTH_MONFRI_WEEK_{monday.strftime('%Y%m%d')}_{next_monday.strftime('%Y%m%d')}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_xgb_inference(
    cv2: pd.DataFrame, *,
    xgb_bundle: Path,
    feature_contract: Path,
    sanitizer_config: Path,
) -> np.ndarray:
    """Return (n_m5, 7) signal_bridge_v3 per-bar XGB outputs."""
    print(f"[{ACTION}] XGB inference on {len(cv2):,} M5 rows...", flush=True)
    contract = json.loads(feature_contract.read_text())
    xgb_features = list(contract["features"])
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config))

    work = cv2.copy()
    if "session_id" not in work.columns:
        ts = pd.to_datetime(work.index if isinstance(work.index, pd.DatetimeIndex) else work["time"], utc=True)
        sess = get_session_vectorized(ts)
        sess_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
        work["session_id"] = sess.map(sess_map).fillna(0).astype(np.int32)
    if "is_ASIA" not in work.columns:
        work["is_ASIA"] = (work["session_id"].astype(int) == 0).astype(np.int8)
    if "minutes_since_session_open" not in work.columns:
        ts = pd.to_datetime(work.index if isinstance(work.index, pd.DatetimeIndex) else work["time"], utc=True)
        work["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    if "minutes_to_next_session_boundary" not in work.columns:
        ts = pd.to_datetime(work.index if isinstance(work.index, pd.DatetimeIndex) else work["time"], utc=True)
        work["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    if "session_change_flag" not in work.columns:
        sid = work["session_id"].astype(np.int32)
        work["session_change_flag"] = (sid.diff().fillna(0) != 0).astype(np.int8)
    if "session_tradable" not in work.columns:
        work["session_tradable"] = (work["session_id"].astype(int) != 0).astype(np.int8)

    missing = [c for c in xgb_features if c not in work.columns]
    if missing:
        raise RuntimeError(f"[{ACTION}] XGB features missing in canonical_v2: {missing}")

    df_features = work[xgb_features].copy()
    x_array, _ = sanitizer.sanitize(df_features, feature_list=xgb_features, allow_nan_fill=True, nan_fill_value=0.0)

    model = XGBMultiheadModel.load(str(xgb_bundle / "xgb_universal_multihead_v2.joblib"))
    df_san = pd.DataFrame(x_array, columns=xgb_features, index=df_features.index)

    sess = work["session_id"].astype(int).to_numpy()
    sess_name_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    bridge = np.zeros((len(work), 7), dtype=np.float32)
    for sid_int in (0, 1, 2, 3):
        idx = np.where(sess == sid_int)[0]
        if idx.size == 0:
            continue
        probs = model.predict_proba(df_san.iloc[idx], session=sess_name_map[sid_int], feature_list=xgb_features)
        if hasattr(probs, "p_long"):
            pl, ps, pf = np.asarray(probs.p_long), np.asarray(probs.p_short), np.asarray(probs.p_flat)
        else:
            pl, ps, pf = np.asarray(probs["p_long"]), np.asarray(probs["p_short"]), np.asarray(probs["p_flat"])
        b = proba_to_signal_bridge_v1(np.column_stack([pl, ps, pf])).astype(np.float32)
        bridge[idx] = b
    print(f"[{ACTION}] XGB-bridge done. shape={bridge.shape}", flush=True)
    return bridge


def build_v10_input_matrices(
    cv2: pd.DataFrame, bridge: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (n_m5, 41) seq_per_bar, (n_m5, 69) ctx_cont, (n_m5, 6) ctx_cat."""
    n = len(cv2)
    # per-bar SEQ = [bridge (7), price_state (34: 30 + 4 volume)]
    per_bar = np.zeros((n, SEQ_SIGNAL_DIM_V3), dtype=np.float32)
    per_bar[:, 0:7] = bridge
    # Volume/order-flow features are DERIVED (not in canonical_v2) — compute with
    # the SAME one-truth helper as the V10 builder + live serving.
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES, add_volume_features
    if any(c not in cv2.columns for c in VOLUME_FEATURE_NAMES):
        add_volume_features(cv2)
    missing_pb = [c for c in PER_BAR_PRICE_STATE_FIELDS_V3 if c not in cv2.columns]
    if missing_pb:
        raise RuntimeError(f"[{ACTION}] PER_BAR_PRICE_STATE missing in prebuilt: {missing_pb}")
    for j, fname in enumerate(PER_BAR_PRICE_STATE_FIELDS_V3):
        per_bar[:, 7 + j] = pd.to_numeric(cv2[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    # ctx_cont (43) — derive is_ASIA from session_id if missing
    ctx_cont = np.zeros((n, CTX_CONT_DIM_V3), dtype=np.float32)
    for j, fname in enumerate(ORDERED_CTX_CONT_NAMES_V3):
        if fname in cv2.columns:
            ctx_cont[:, j] = pd.to_numeric(cv2[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        elif fname == "is_ASIA" and "session_id" in cv2.columns:
            ctx_cont[:, j] = (cv2["session_id"].astype(int) == 0).astype(np.float32).to_numpy()
        else:
            raise RuntimeError(f"[{ACTION}] CTX_CONT column missing in prebuilt: {fname}")

    # ctx_cat (6) — int64 categorical
    ctx_cat = np.zeros((n, CTX_CAT_DIM_V3), dtype=np.int64)
    for j, fname in enumerate(ORDERED_CTX_CAT_NAMES_V3):
        if fname in cv2.columns:
            ctx_cat[:, j] = pd.to_numeric(cv2[fname], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        else:
            raise RuntimeError(f"[{ACTION}] CTX_CAT column missing in prebuilt: {fname}")
    return per_bar, ctx_cont, ctx_cat


def run_v10_inference(
    *, model: EntryV10CtxHybridTransformer,
    per_bar: np.ndarray, ctx_cont: np.ndarray, ctx_cat: np.ndarray,
    seq_len: int, batch_size: int, device: torch.device,
    decision_ts_ns: Optional[np.ndarray] = None,
    multi_tf_feats: Optional[Dict[str, "pd.DataFrame"]] = None,
    multi_tf_shift: Optional[Dict[str, "pd.Timedelta"]] = None,
    per_tf_lens: Optional[Dict[str, int]] = None,
    v2_mode: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run V10 v2 in batches over all decision moments where idx >= seq_len-1.

    V12.2: when the model is multi-TF, also slice M15/H1/H4/D1 windows per
    decision bar (using `decision_ts_ns` + `multi_tf_feats`) and pass them as
    seq_m15/seq_h1/seq_h4/seq_d1 to the model.forward — without this, the
    multi-TF residual branch is silently bypassed.
    """
    n = per_bar.shape[0]
    decision_indices = np.arange(seq_len - 1, n)  # decision possible at idx >= seq_len-1
    n_dec = len(decision_indices)

    enable_mtf = bool(getattr(getattr(model, "cfg", None), "enable_multi_tf", False))
    tf_names: Tuple[str, ...] = ()
    if enable_mtf:
        if (multi_tf_feats is None or multi_tf_shift is None or decision_ts_ns is None
                or per_tf_lens is None):
            raise RuntimeError(
                "[run_v10_inference] V10 model is multi-TF but multi_tf_feats / "
                "multi_tf_shift / decision_ts_ns / per_tf_lens were not provided."
            )
        from gx1.features.htf_features import get_last_n_at_or_before  # noqa: F401
        # V2-multi-TF V10 adds M5 as the 5th branch (gated by v2_mode).
        tf_names = (("M5", "M15", "H1", "H4", "D1") if v2_mode
                    else ("M15", "H1", "H4", "D1"))
        print(f"[{ACTION}] multi-TF inference: TFs={tf_names} seq_lens={per_tf_lens} v2_mode={v2_mode}",
              flush=True)

    print(f"[{ACTION}] V10 inference on {n_dec:,} decision moments (batch={batch_size})...", flush=True)

    out_keys = ("direction_logits", "anchor_logits", "delta_logits", "path_quality",
                "mfe_first_n", "tradable_logit", "bad_path_logit", "clean_edge_logit", "survival_logit",
                # V10 v3+ aux heads — present only when bundle has them; soft-handled below
                "tf_agreement_logit", "path_quality_log_var", "position_size_logit", "hold_horizon_logit",
                # 2026-05-27: COSTFIX V10 new heads (dip/forecast/timing/tail/vol). Emitted
                # base-named for the Entry-IQL path (candidate-gen -> fwd-outcome carry -> state).
                "dip_pred", "forecast_pred", "timing_pred", "tail_risk_pred", "vol_forecast_pred")
    # Per-key width for the NaN-fill of absent heads (keeps concat shape-consistent;
    # scalars default to width 1). Bundles lacking a head emit all-NaN of this width.
    _OUT_KEY_WIDTH = {"dip_pred": 18, "forecast_pred": 4, "timing_pred": 12,
                      "tail_risk_pred": 6, "vol_forecast_pred": 3}
    out_buffers: Dict[str, List[np.ndarray]] = {k: [] for k in out_keys}

    model.eval()
    with torch.no_grad():
        t0 = _time.time()
        for batch_start in range(0, n_dec, batch_size):
            batch_idx = decision_indices[batch_start:batch_start + batch_size]
            B = len(batch_idx)
            seq_x = np.zeros((B, seq_len, SEQ_SIGNAL_DIM_V3), dtype=np.float32)
            for bi, di in enumerate(batch_idx):
                seq_x[bi] = per_bar[di - seq_len + 1: di + 1]
            snap_x = per_bar[batch_idx]  # (B, 37)
            ctx_cont_x = ctx_cont[batch_idx]  # (B, 43)
            ctx_cat_x = ctx_cat[batch_idx]  # (B, 6)

            seq_t = torch.from_numpy(seq_x).to(device)
            snap_t = torch.from_numpy(snap_x).to(device)
            cont_t = torch.from_numpy(ctx_cont_x).to(device)
            cat_t = torch.from_numpy(ctx_cat_x).to(device)

            mtf_kwargs: Dict[str, torch.Tensor] = {}
            if enable_mtf:
                from gx1.features.htf_features import get_last_n_at_or_before
                batch_ts_ns = decision_ts_ns[batch_idx]
                for tf in tf_names:
                    feats_df = multi_tf_feats[tf]
                    shift_td = multi_tf_shift[tf]
                    n_tf = int(per_tf_lens[tf])
                    stacked = np.empty(
                        (B, n_tf, feats_df.shape[1]),
                        dtype=np.float32,
                    )
                    for bi, ts_ns in enumerate(batch_ts_ns):
                        ts = pd.Timestamp(int(ts_ns), unit="ns", tz="UTC")
                        stacked[bi] = get_last_n_at_or_before(
                            feats_df, ts, n=n_tf, tf_shift=shift_td,
                        )
                    mtf_kwargs[f"seq_{tf.lower()}"] = torch.from_numpy(stacked).to(
                        device, non_blocking=True,
                    )

            out = model(
                seq_x=seq_t, snap_x=snap_t, ctx_cont=cont_t, ctx_cat=cat_t,
                **mtf_kwargs,
            )
            for k in out_keys:
                if k in out:
                    out_buffers[k].append(out[k].detach().cpu().numpy())
                else:
                    # Aux head absent in this bundle — fill width-aware NaN sentinel.
                    out_buffers[k].append(np.full((B, _OUT_KEY_WIDTH.get(k, 1)), np.nan, dtype=np.float32))

            if (batch_start // batch_size) % 200 == 0:
                elapsed = _time.time() - t0
                rate = (batch_start + B) / max(elapsed, 1e-6)
                eta = (n_dec - batch_start - B) / max(rate, 1e-6)
                print(f"[{ACTION}]   {batch_start + B:,}/{n_dec:,} ({rate:.0f} bars/s, ETA {eta:.0f}s)", flush=True)

    return {k: np.concatenate(out_buffers[k], axis=0) for k in out_keys}, decision_indices


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Candidate emission
# ---------------------------------------------------------------------------


CANDIDATE_COLS = [
    "side", "session", "weekday_utc", "hour_utc", "atr_bps", "entry_spread_bps",
    "p_long", "p_short", "p_flat", "p_hat", "margin", "uncertainty_score",
    "tradable_prob", "mfe_first_n_pred", "path_quality_pred",
    # NEW: V10 v2 aux outputs needed for entry-IQL BIDIR retrain (Phase 6)
    "bad_path_prob", "direction_logit_long", "direction_logit_short", "direction_logit_flat",
    # V10 v3+ aux heads (Targets 1-4) — needed for Exit-IQL state vector.
    # NaN-filled for legacy bundles without these heads.
    "tf_agreement_pred", "path_quality_log_var", "path_quality_std",
    "position_size_pred", "hold_horizon_pred",
    # 2026-05-27: COSTFIX V10 new heads (base-named, emitted for Entry-IQL path).
    *[f"v10_dip_{_i}" for _i in range(18)],
    *[f"v10_forecast_{_i}" for _i in range(4)],
    *[f"v10_timing_{_i}" for _i in range(12)],
    *[f"v10_tail_risk_{_i}" for _i in range(6)],
    *[f"v10_vol_forecast_{_i}" for _i in range(3)],
    "vol_regime", "trend_regime",
    "run_id", "candidate_uid", "trade_uid", "trade_id",
    "decision_ts_utc", "source_eval_log", "source_eval_log_row",
    "decision", "accepted", "decision_reason", "policy_lane", "policy_hash",
    "entry_bundle_sha256", "exit_bundle_sha256",
    "open_ts_utc", "close_ts_utc",
    "mfe_threshold_bps", "positive_exit", "cata", "never_mfe",
    "good_mfe_then_rot", "trainable_mask_v1", "meta_allow_label_v1",
    "pnl_bps", "mfe_bps", "mae_bps", "bars_in_trade", "exit_reason",
    "good_trade_mfe20_mae5_v1", "mfe_mae_ratio_v1",
]


# H6 (2026-06-03 gap-register): Entry-IQL was REGIME-BLIND — vol_regime/trend_regime were
# constant placeholders, so the one-hot state columns carried zero signal. These map the real
# (now-3-class BIG-8) trend_regime_id + the vol_regime_id quintile to the ONE_HOT categories
# the IQL state expects (materialize_build_entry_iql_v2.ONE_HOT_COLS). Mirror in
# v12_entry_iql_live.py. Gated by GX1_REGIME_V4 (default OFF = cement placeholders, since the
# cement Entry-IQL was trained on the constants — feeding it real regime would be train/serve
# skew; the regime-aware Entry-IQL retrain sets the flag ON).
_TREND_REGIME_ID_TO_LABEL = {0: "TREND_DOWN", 1: "TREND_NEUTRAL", 2: "TREND_UP"}
_VOL_REGIME_ID_TO_LABEL = {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "EXTREME"}


def emit_candidates(
    *, cv2: pd.DataFrame, decision_indices: np.ndarray,
    v10_out: Dict[str, np.ndarray],
    min_margin: float, min_directional_prob: float,
    entry_bundle_sha: str,
) -> pd.DataFrame:
    """Apply entry thresholds and emit candidate rows (no explicit per-trade simulation)."""
    direction = softmax(v10_out["direction_logits"], axis=-1)  # (N, 3) — order: long/short/flat
    p_long = direction[:, 0]
    p_short = direction[:, 1]
    p_flat = direction[:, 2]
    p_hat = direction.max(axis=-1)
    sorted_dir = np.sort(direction, axis=-1)
    margin = sorted_dir[:, -1] - sorted_dir[:, -2]
    uncertainty = 1.0 - p_hat
    tradable_prob = 1.0 / (1.0 + np.exp(-v10_out["tradable_logit"][:, 0]))  # sigmoid
    path_quality = v10_out["path_quality"][:, 0]
    mfe_first_n = v10_out["mfe_first_n"][:, 0]
    # NEW: bad_path_prob (sigmoid of logit) — used as hard penalty in entry-IQL reward
    bad_path_prob = 1.0 / (1.0 + np.exp(-v10_out["bad_path_logit"][:, 0]))
    # NEW: raw direction logits — entry-IQL state will use these instead of softmaxed probs
    direction_logits_raw = v10_out["direction_logits"]  # (N, 3)

    # Entry filter: directional > flat AND max(p_long, p_short) >= threshold AND margin >= min
    directional_mask = (np.maximum(p_long, p_short) >= min_directional_prob) & \
                       ((p_long + p_short) > p_flat) & (margin >= min_margin)
    n_cand = int(directional_mask.sum())
    print(f"[{ACTION}] candidate filter: {n_cand:,}/{len(decision_indices):,} "
          f"({n_cand / max(len(decision_indices),1) * 100:.1f}%) pass entry threshold", flush=True)

    # Build rows
    sel = np.where(directional_mask)[0]
    sel_global_idx = decision_indices[sel]  # M5 indices in cv2

    # Slice cv2 metadata for these rows
    ts_index = cv2.index[sel_global_idx]
    side_arr = np.where(p_long[sel] >= p_short[sel], "long", "short")
    sess_id = cv2["session_id"].iloc[sel_global_idx].to_numpy(dtype=np.int32)
    sess_name_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    sess_str = pd.Series(sess_id).map(sess_name_map).fillna("UNKNOWN").to_list()

    n = len(sel)
    rows: Dict[str, list] = {}
    rows["side"] = side_arr.tolist()
    rows["session"] = sess_str
    rows["weekday_utc"] = [t.weekday() for t in ts_index]
    rows["hour_utc"] = [t.hour for t in ts_index]
    rows["atr_bps"] = cv2["atr_bps"].iloc[sel_global_idx].to_numpy().tolist() if "atr_bps" in cv2.columns else [None] * n
    rows["entry_spread_bps"] = cv2["spread_bps"].iloc[sel_global_idx].to_numpy().tolist() if "spread_bps" in cv2.columns else [None] * n
    rows["p_long"] = p_long[sel].tolist()
    rows["p_short"] = p_short[sel].tolist()
    rows["p_flat"] = p_flat[sel].tolist()
    rows["p_hat"] = p_hat[sel].tolist()
    rows["margin"] = margin[sel].tolist()
    rows["uncertainty_score"] = uncertainty[sel].tolist()
    rows["tradable_prob"] = tradable_prob[sel].tolist()
    rows["mfe_first_n_pred"] = mfe_first_n[sel].tolist()
    rows["path_quality_pred"] = path_quality[sel].tolist()
    # NEW V10 v2 aux outputs for entry-IQL Phase 6
    rows["bad_path_prob"] = bad_path_prob[sel].tolist()
    rows["direction_logit_long"] = direction_logits_raw[sel, 0].tolist()
    rows["direction_logit_short"] = direction_logits_raw[sel, 1].tolist()
    rows["direction_logit_flat"] = direction_logits_raw[sel, 2].tolist()
    # V10 v3+ aux heads — only meaningful when bundle has them. NaN-sentinel
    # for legacy bundles. Stored raw for Exit-IQL to consume as features.
    def _sigmoid_clip(arr):
        a = np.clip(arr, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-a))
    if "tf_agreement_logit" in v10_out and not np.all(np.isnan(v10_out["tf_agreement_logit"])):
        rows["tf_agreement_pred"] = _sigmoid_clip(v10_out["tf_agreement_logit"][sel, 0]).tolist()
    else:
        rows["tf_agreement_pred"] = [float("nan")] * n
    if "path_quality_log_var" in v10_out and not np.all(np.isnan(v10_out["path_quality_log_var"])):
        rows["path_quality_log_var"] = v10_out["path_quality_log_var"][sel, 0].tolist()
        rows["path_quality_std"] = np.exp(np.clip(v10_out["path_quality_log_var"][sel, 0], -5.0, 5.0) / 2.0).tolist()
    else:
        rows["path_quality_log_var"] = [float("nan")] * n
        rows["path_quality_std"] = [float("nan")] * n
    if "position_size_logit" in v10_out and not np.all(np.isnan(v10_out["position_size_logit"])):
        rows["position_size_pred"] = _sigmoid_clip(v10_out["position_size_logit"][sel, 0]).tolist()
    else:
        rows["position_size_pred"] = [float("nan")] * n
    if "hold_horizon_logit" in v10_out and not np.all(np.isnan(v10_out["hold_horizon_logit"])):
        rows["hold_horizon_pred"] = _sigmoid_clip(v10_out["hold_horizon_logit"][sel, 0]).tolist()
    else:
        rows["hold_horizon_pred"] = [float("nan")] * n
    # 2026-05-27: COSTFIX V10 new heads — base-named (no _v2) for the Entry-IQL path
    # (candidate parquet -> forward-outcome carry-forward -> Entry-IQL state). Index-
    # consistent: v10_out["<head>_pred"][:, i] -> column v10_<base>_{i}. NaN if absent.
    _NEW_HEAD_EMIT = {"dip_pred": ("v10_dip", 18), "forecast_pred": ("v10_forecast", 4),
                      "timing_pred": ("v10_timing", 12), "tail_risk_pred": ("v10_tail_risk", 6),
                      "vol_forecast_pred": ("v10_vol_forecast", 3)}
    for _key, (_base, _w) in _NEW_HEAD_EMIT.items():
        _arr = v10_out.get(_key)
        _ok = _arr is not None and _arr.ndim == 2 and _arr.shape[1] >= _w and not np.all(np.isnan(_arr))
        for _i in range(_w):
            rows[f"{_base}_{_i}"] = (_arr[sel, _i].tolist() if _ok else [float("nan")] * n)
    # H6: real regime into the Entry-IQL state when GX1_REGIME_V4=1 (else cement placeholders).
    if os.environ.get("GX1_REGIME_V4", "0") == "1" and \
            "trend_regime_id" in cv2.columns and "vol_regime_id" in cv2.columns:
        _tr = cv2["trend_regime_id"].iloc[sel_global_idx].to_numpy(dtype=np.int64)
        _vr = cv2["vol_regime_id"].iloc[sel_global_idx].to_numpy(dtype=np.int64)
        rows["vol_regime"] = [_VOL_REGIME_ID_TO_LABEL.get(int(v), "MEDIUM") for v in _vr]
        rows["trend_regime"] = [_TREND_REGIME_ID_TO_LABEL.get(int(t), "TREND_NEUTRAL") for t in _tr]
    else:
        rows["vol_regime"] = ["MEDIUM"] * n  # placeholder (cement) — regime-blind
        rows["trend_regime"] = ["TREND_NEUTRAL"] * n  # placeholder (cement) — regime-blind
    rows["decision_ts_utc"] = [t.isoformat() for t in ts_index]
    rows["source_eval_log"] = [""] * n
    rows["source_eval_log_row"] = [0] * n
    rows["decision"] = ["DIRECTIONAL"] * n
    rows["accepted"] = [True] * n
    rows["decision_reason"] = ["v2_inference_batch"] * n
    rows["policy_lane"] = ["v2_inference"] * n
    rows["policy_hash"] = [""] * n
    rows["entry_bundle_sha256"] = [entry_bundle_sha] * n
    rows["exit_bundle_sha256"] = [""] * n
    rows["open_ts_utc"] = [None] * n
    rows["close_ts_utc"] = [None] * n
    rows["mfe_threshold_bps"] = [1.0] * n
    rows["positive_exit"] = [None] * n
    rows["cata"] = [None] * n
    rows["never_mfe"] = [None] * n
    rows["good_mfe_then_rot"] = [None] * n
    rows["trainable_mask_v1"] = [True] * n
    rows["meta_allow_label_v1"] = [None] * n
    rows["pnl_bps"] = [float("nan")] * n
    rows["mfe_bps"] = [float("nan")] * n
    rows["mae_bps"] = [float("nan")] * n
    rows["bars_in_trade"] = [float("nan")] * n
    rows["exit_reason"] = [None] * n
    rows["good_trade_mfe20_mae5_v1"] = [None] * n
    rows["mfe_mae_ratio_v1"] = [float("nan")] * n

    # Build candidate_uid + trade_uid + run_id from week mapping
    rows["run_id"] = [_truth_week_run_id(t) for t in ts_index]
    rows["candidate_uid"] = [f"{r}:{i}:cand:v2_inf:{ts_index[i].strftime('%Y%m%dT%H%M%S')}"
                             for i, r in enumerate(rows["run_id"])]
    rows["trade_uid"] = rows["candidate_uid"]
    rows["trade_id"] = rows["candidate_uid"]

    df = pd.DataFrame({c: rows[c] for c in CANDIDATE_COLS})
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--prebuilt", type=str, default=str(DEFAULT_PREBUILT_PARQUET))
    parser.add_argument("--xgb-bundle", type=str, default=str(DEFAULT_XGB_BUNDLE))
    parser.add_argument("--v10-bundle", type=str, default=str(DEFAULT_V10_BUNDLE))
    parser.add_argument("--xgb-feature-contract", type=str, default=str(DEFAULT_XGB_FEATURE_CONTRACT))
    parser.add_argument("--xgb-sanitizer-config", type=str, default=str(DEFAULT_XGB_SANITIZER_CONFIG))
    parser.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--v2-cache-dir", type=str, default=str(DEFAULT_V2_CACHE_DIR),
                        help="Multi-TF V2 cache dir (5 TFs incl M5 × 25 feats). One-truth "
                             "source the V2-multi-TF V10 was trained on.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--min-directional-prob", type=float, default=DEFAULT_MIN_DIRECTIONAL_PROB)
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="If set, only process the first N rows of canonical_v2 (smoke testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run-no-write", action="store_true",
                        help="Skip per-week parquet writes; print counts only")
    # V12.2: required when V10 bundle is multi-TF (auto-detected from bundle_metadata.json).
    parser.add_argument("--m5-prebuilt-path", type=str, default=None,
                        help="V12.2: M5 prebuilt parquet for multi-TF feature extraction. "
                             "If unset and V10 bundle is multi-TF, defaults to --prebuilt.")
    parser.add_argument("--multi-tf-seq-len", type=int, default=96,
                        help="V12.2: window length per multi-TF stream (M15/H1/H4/D1).")
    args = parser.parse_args()

    print(f"[{ACTION}] device={args.device}", flush=True)

    # ---- Load canonical_v2 ----
    print(f"[{ACTION}] loading prebuilt: {args.prebuilt}", flush=True)
    cv2 = pd.read_parquet(args.prebuilt)
    if "time" in cv2.columns and not isinstance(cv2.index, pd.DatetimeIndex):
        cv2["time"] = pd.to_datetime(cv2["time"], utc=True)
        cv2 = cv2.set_index("time")
    cv2 = cv2.sort_index()
    if args.limit_rows is not None:
        cv2 = cv2.iloc[:int(args.limit_rows)]
    print(f"[{ACTION}] cv2 shape={cv2.shape} ts=[{cv2.index[0]} → {cv2.index[-1]}]", flush=True)

    # ---- canonical_v3 augmentation ----
    # The canonical_v3 prebuilt was regenerated 2026-05-11 without ctx_cont /
    # session / interaction features. Run the live augmenter inline so XGB +
    # V10 can find atr_bps, D1_dist_from_ema200_atr, _v1_is_US, etc.
    pre_aug_cols = set(cv2.columns)
    cv2 = augment_canonical_v3(cv2, cv2)
    added_cols = sorted(set(cv2.columns) - pre_aug_cols)
    print(f"[{ACTION}] canonical_v3 augment: added {len(added_cols)} cols "
          f"(now {len(cv2.columns)} total)", flush=True)

    # ---- ctx_cont parity: 24 group-A + 36 dip/struct (2026-05-26) ----
    # SERVING PARITY: the V10 v3+ contract (ctx_cont=105) requires the group-A
    # parity + dip/struct features that augment_canonical_v3 does NOT produce.
    # Compute them via the SAME one-truth helper the V10/V3 training builders use
    # so candidate generation feeds the model identical ctx_cont at train+serve.
    from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
    n_before = len(cv2.columns)
    attach_group_a_dip_struct_ctx_columns(cv2, journal_label="inference_batch")
    print(f"[{ACTION}] ctx_cont parity: added {len(cv2.columns) - n_before} group-A+dip/struct cols "
          f"(now {len(cv2.columns)} total)", flush=True)

    # ---- XGB inference ----
    bridge = run_xgb_inference(
        cv2,
        xgb_bundle=Path(args.xgb_bundle),
        feature_contract=Path(args.xgb_feature_contract),
        sanitizer_config=Path(args.xgb_sanitizer_config),
    )

    # ---- Build V10 inputs ----
    per_bar, ctx_cont, ctx_cat = build_v10_input_matrices(cv2, bridge)
    print(f"[{ACTION}] inputs: per_bar={per_bar.shape} ctx_cont={ctx_cont.shape} ctx_cat={ctx_cat.shape}",
          flush=True)

    # ---- Load V10 via the one-truth bundle loader ----
    # 2026-05-26: was hand-rolling model construction (4-TF, V1 multi-TF, no M5
    # branch, no dip/forecast/timing/tail/vol heads) → stale vs the V2-multi-TF
    # V10 (e.g. ENTRY_V10_RELABEL15H24_SYMW). Reuse load_entry_v10_ctx_bundle (the
    # SAME loader rescore uses) so construction tracks the bundle's metadata +
    # state_dict exactly (M5 branch, cross-TF gates, all heads). One truth.
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    device = torch.device(args.device)
    bundle = load_entry_v10_ctx_bundle(
        bundle_dir=Path(args.v10_bundle), device=args.device, xgb_models=None,
    )
    model = bundle.transformer_model
    v10_meta = bundle.metadata
    mtf_cfg = v10_meta.get("multi_tf", {}) or {}
    enable_mtf = bool(mtf_cfg.get("enabled", False))
    # multi-TF is REQUIRED (feedback_multi_tf_always_mandatory). Reject non-multi-TF.
    if not enable_mtf:
        raise RuntimeError(
            f"[{ACTION}] V10 bundle {Path(args.v10_bundle).name} is NOT multi-TF "
            f"(multi_tf.enabled={enable_mtf}). Script hard-requires multi-TF V10."
        )
    v2_mode = bool(mtf_cfg.get("v2_mode", False))
    print(f"[{ACTION}] V10 loaded (one-truth bundle loader); seq_len={v10_meta['seq_len']} "
          f"v2_mode={v2_mode} m5_seq_dim={mtf_cfg.get('m5_seq_dim')} on {device}", flush=True)

    # ---- Multi-TF features: load the V2 cache (5 TFs incl M5, 25 feats) ----
    # MUST match training: the V2-multi-TF V10 was trained on load_multi_tf_v2_cache
    # output, NOT the V1 build_multi_tf_per_bar_features (17 feats, no M5 branch).
    mtf_feats = None
    mtf_shift = None
    per_tf_lens: Optional[Dict[str, int]] = None
    if enable_mtf:
        from gx1.features.htf_features import load_multi_tf_v2_cache, MULTI_TF_SHIFT
        v2_cache_dir = Path(args.v2_cache_dir)
        print(f"[{ACTION}] loading multi-TF V2 cache: {v2_cache_dir}", flush=True)
        mtf_feats = load_multi_tf_v2_cache(v2_cache_dir)
        mtf_shift = MULTI_TF_SHIFT
        per_tf_lens = {
            "M5": int(mtf_cfg.get("m5_seq_len", 96)),
            "M15": int(mtf_cfg.get("m15_seq_len", 96)),
            "H1": int(mtf_cfg.get("h1_seq_len", 96)),
            "H4": int(mtf_cfg.get("h4_seq_len", 96)),
            "D1": int(mtf_cfg.get("d1_seq_len", 96)),
        }
        for tf, df in mtf_feats.items():
            print(f"[{ACTION}]   {tf}: {len(df):,} bars × {df.shape[1]} feats", flush=True)

    # ---- decision timestamps (int64 ns) for multi-TF slicing ----
    decision_ts_ns = cv2.index.values.astype("datetime64[ns]").astype(np.int64)

    # ---- V10 inference ----
    v10_out, decision_indices = run_v10_inference(
        model=model, per_bar=per_bar, ctx_cont=ctx_cont, ctx_cat=ctx_cat,
        seq_len=int(v10_meta["seq_len"]), batch_size=int(args.batch_size), device=device,
        decision_ts_ns=decision_ts_ns,
        multi_tf_feats=mtf_feats,
        multi_tf_shift=mtf_shift,
        per_tf_lens=per_tf_lens,
        v2_mode=v2_mode,
    )

    # ---- Emit candidates ----
    entry_bundle_sha = _file_sha256(Path(args.v10_bundle) / "model_state_dict.pt")
    candidates_df = emit_candidates(
        cv2=cv2, decision_indices=decision_indices, v10_out=v10_out,
        min_margin=float(args.min_margin),
        min_directional_prob=float(args.min_directional_prob),
        entry_bundle_sha=entry_bundle_sha,
    )

    # ---- Group by run_id and write per-week parquets ----
    grouped = candidates_df.groupby("run_id")
    print(f"[{ACTION}] candidates total={len(candidates_df):,} weeks={len(grouped)}", flush=True)
    n_written = 0
    if not args.dry_run_no_write:
        reports_root = Path(args.reports_root)
        for run_id, group in grouped:
            week_dir = reports_root / run_id
            week_dir.mkdir(parents=True, exist_ok=True)
            out_path = week_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
            group.to_parquet(out_path, index=False)
            n_written += 1
        print(f"[{ACTION}] wrote {n_written} per-week parquets under {reports_root}", flush=True)

    # ---- Summary ----
    summary = {
        "action_v1": ACTION,
        "built_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "n_m5_bars": int(len(cv2)),
        "n_decision_moments": int(len(decision_indices)),
        "n_candidates_total": int(len(candidates_df)),
        "n_weeks_written": int(n_written),
        "min_margin": float(args.min_margin),
        "min_directional_prob": float(args.min_directional_prob),
        "v10_bundle": str(args.v10_bundle),
        "xgb_bundle": str(args.xgb_bundle),
        "prebuilt_parquet": str(args.prebuilt),
        "side_distribution": {
            "long": int((candidates_df["side"] == "long").sum()),
            "short": int((candidates_df["side"] == "short").sum()),
        },
    }
    summary_path = Path(args.reports_root) / f"{ACTION}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[{ACTION}] summary at {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
