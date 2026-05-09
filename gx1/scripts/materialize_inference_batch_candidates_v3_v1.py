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
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_XGB_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v5__BIDIR_RSI_SMC_PRUNED_CANONICAL_V3_20260505T081604Z_1000est"
)
DEFAULT_V10_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/models/entry_v10_ctx/ENTRY_V10_CTX__RETRAIN_2026Q2_BIDIR_SMC_CANONICAL_V3_6YR_BS512_20260506T120938Z"
)
DEFAULT_XGB_FEATURE_CONTRACT = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_canonical_v3_v1.json"
DEFAULT_XGB_SANITIZER_CONFIG = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_canonical_v3_v1.json"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")

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
    """Build (n_m5, 37) seq_per_bar, (n_m5, 43) ctx_cont, (n_m5, 6) ctx_cat."""
    n = len(cv2)
    # per-bar SEQ = [bridge (7), price_state (30)]
    per_bar = np.zeros((n, SEQ_SIGNAL_DIM_V3), dtype=np.float32)
    per_bar[:, 0:7] = bridge
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
) -> Dict[str, np.ndarray]:
    """Run V10 v2 in batches over all decision moments where idx >= seq_len-1."""
    n = per_bar.shape[0]
    decision_indices = np.arange(seq_len - 1, n)  # decision possible at idx >= seq_len-1
    n_dec = len(decision_indices)
    print(f"[{ACTION}] V10 inference on {n_dec:,} decision moments (batch={batch_size})...", flush=True)

    out_keys = ("direction_logits", "anchor_logits", "delta_logits", "path_quality",
                "mfe_first_n", "tradable_logit", "bad_path_logit", "clean_edge_logit", "survival_logit")
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
            out = model(seq_x=seq_t, snap_x=snap_t, ctx_cont=cont_t, ctx_cat=cat_t)
            for k in out_keys:
                out_buffers[k].append(out[k].detach().cpu().numpy())

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
    rows["vol_regime"] = ["MEDIUM"] * n  # placeholder; V3 builder doesn't read
    rows["trend_regime"] = ["TREND_NEUTRAL"] * n  # placeholder
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--min-directional-prob", type=float, default=DEFAULT_MIN_DIRECTIONAL_PROB)
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="If set, only process the first N rows of canonical_v2 (smoke testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run-no-write", action="store_true",
                        help="Skip per-week parquet writes; print counts only")
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

    # ---- Load V10 v2 ----
    v10_meta = json.loads((Path(args.v10_bundle) / "bundle_metadata.json").read_text())
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=v10_meta["seq_input_dim"],
        snap_input_dim=v10_meta["snap_input_dim"],
        ctx_cont_dim=v10_meta["ctx_cont_dim"],
        ctx_cat_dim=v10_meta["ctx_cat_dim"],
        seq_len=v10_meta["seq_len"],
    )
    state = torch.load(Path(args.v10_bundle) / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device)
    print(f"[{ACTION}] V10 v2 loaded; seq_len={v10_meta['seq_len']} on {device}", flush=True)

    # ---- V10 inference ----
    v10_out, decision_indices = run_v10_inference(
        model=model, per_bar=per_bar, ctx_cont=ctx_cont, ctx_cat=ctx_cat,
        seq_len=int(v10_meta["seq_len"]), batch_size=int(args.batch_size), device=device,
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
