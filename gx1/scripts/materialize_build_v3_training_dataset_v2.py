#!/usr/bin/env python3
"""Build V3 (exit transformer) training dataset — JSONL format for train_exit_transformer_v0_sharded.py.

Mission
-------
Generate per-(trade, bar) records suitable for V3 retraining on the 2026-Q2 stack:
  - exit_io_v5 contract (89 features per M1 bar = V3 prefix 58 + V2 ext 22 + SMC 9)
  - M1L512 window (512 M1 bars lookback per inference moment)
  - Bidirectional trades (long + short candidates from new V10 + XGB v4)
  - Hindsight teacher fields (final_pnl, final_mfe, etc.) for label engineering

Pipeline
--------
1. PRE-COMPUTE 89-feature exit_io_v5 vectors over full M1 tape (~2.2M bars):
     - 7 XGB-bridge fields per M1 bar (run XGB v4 on each M1 bar's canonical features)
     - 5 entry-snapshot per bar (filled with zeros for pre-trade bars; per-bar values for in-trade)
     - 12 trade-state per bar (zeros for pre-trade; per-bar trade evolution for in-trade)
     - 2 giveback per bar (same)
     - 11 ctx_cont per bar (lazy-join from canonical_v2)
     - 5 swing per bar (lazy-join)
     - 5 session per bar (computed from timestamp)
     - 6 ctx_cat per bar (lazy-join)
     - 5 m5_phase per bar (computed from timestamp)
     - 22 V2 extension per bar (H1/H4/D1/M15 from canonical_v2)
     - 9 SMC per bar (from canonical_v2)
     ─ note: trade-state and entry-snapshot are pre-trade-zero; populated per (trade, bar) at emit time

2. FOR EACH CANDIDATE (entry decision from frozen-V10-replay corpus):
     a. Determine side (long/short)
     b. Walk M1 bars from candidate.decision_ts forward, capped at max_bars_per_trade
     c. Per emit_stride bars within trade:
        - Build 512-bar io_features window ending at current M1 bar:
          - Pre-trade portion: pre-computed vectors with trade-state=0
          - In-trade portion: pre-computed vectors with trade-state filled from current trade
        - Compute scalars at this bar (pnl, mfe, mae, dd_from_mfe, giveback, bars_held, etc.)
        - Compute teacher hindsight (final_pnl, final_mfe, final_mae, duration)
        - Emit JSONL record

3. STREAM-WRITE JSONL chunks per week to bound RAM usage

Output
------
Single jsonl file: /home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3.jsonl
Estimated 50-150k records, ~10-30 GB.

Research-only. No runtime promotion until 2026-Q2 V3 retrain validates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.signal_bridge_v2 import (
    ORDERED_BRIDGE_FIELDS_V2,
    ORDERED_CTX_CONT_NAMES_V2,
    ORDERED_CTX_CAT_NAMES_V2,
)
from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION,
    compute_m5_phase_onehot,
)
from gx1.exits.contracts.exit_io_v1_ctx36_features import EXIT_IO_V1_CTX36_FEATURES
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe
from gx1.scripts import materialize_build_exit_iql_per_bar_dataset_v1 as v1_pipe
from gx1.time.session_detector import (
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
)
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


ACTION = "BUILD_V3_TRAINING_DATASET_V1"

DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_M1_TAPE_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
DEFAULT_M5_TAPE_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")
DEFAULT_CANONICAL_V2_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_XGB_BUNDLE = Path(
    # V12-cascade authoritative bundle per CURRENT_BUNDLES.md (the 081604Z_1000est
    # retrain was deleted 2026-05-11 — it was never used by V12 cascade).
    "/home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v5__BIDIR_RSI_SMC_PRUNED_CANONICAL_V3_20260505T060428Z"
)
DEFAULT_XGB_FEATURE_CONTRACT = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_canonical_v3_v1.json"
DEFAULT_XGB_SANITIZER_CONFIG = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_canonical_v3_v1.json"
DEFAULT_OUT_PATH = Path("/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3")

WINDOW_LEN = 512  # M1L512 — V5 default
DEFAULT_MAX_BARS_PER_TRADE = 240  # 4h M1 — most trades close within this
DEFAULT_EMIT_STRIDE = 5  # emit one record every N M1 bars during trade (cuts dataset size by N×)
DEFAULT_DIRECTIONAL_THRESHOLD = 0.05

# Index of trade-state features in V5 contract (these are the columns we OVERWRITE per bar
# during in-trade portion of io_features; pre-trade bars keep these at 0).
TRADE_STATE_FEATURE_NAMES = [
    "p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry",  # entry-snapshot (5)
    "pnl_bps_now", "mfe_bps", "mae_bps", "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps", "bars_held", "time_since_mfe_bars",
    "mfe_decay_rate", "pnl_velocity", "pnl_acceleration",
    "rolling_slope_since_entry", "atr_bps_now",  # trade-state (12)
    "giveback_ratio", "giveback_acceleration",  # giveback (2)
]
TRADE_STATE_FEATURE_INDICES = [
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES.index(n) for n in TRADE_STATE_FEATURE_NAMES
]


# ---------------------------------------------------------------------------
# Pre-compute 89-feature vectors per M1 bar (no trade-state — those filled per-bar at emit)
# ---------------------------------------------------------------------------


def precompute_m1_feature_vectors(
    *,
    m1_df: pd.DataFrame,
    canonical_v2_df: pd.DataFrame,
    xgb_bundle_path: Path,
    xgb_feature_contract_path: Path,
    xgb_sanitizer_config_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_m1, 89) array of exit_io_v5 vectors with trade-state-features ZEROED.

    Returns (m1_time_ns, feature_matrix). Trade-state features are ZERO at pre-trade context;
    caller fills them per (trade, bar) at emit time.
    """
    print(f"[{ACTION}] precompute: M1 bars={len(m1_df):,}, output_dim={EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT}",
          flush=True)
    n = len(m1_df)
    feat_mat = np.zeros((n, EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT), dtype=np.float32)
    m1_time_ns = m1_df["time"].astype("int64").to_numpy()

    # ---- 1) XGB inference per M1 bar ----
    # Each M1 bar gets an XGB-derived 7-dim signal-bridge vector.
    # Strategy: run XGB on canonical_v2 M5 rows, then for each M1 bar pick the M5 row at same minute boundary.
    # XGB takes M5-cadence features so we can't directly run per M1 — instead lazy-join M5 results to M1.
    print(f"[{ACTION}] running XGB inference on canonical_v2 M5 rows...", flush=True)
    contract = json.loads(xgb_feature_contract_path.read_text())
    xgb_features = list(contract["features"])
    sanitizer = XGBInputSanitizer.from_config(str(xgb_sanitizer_config_path))

    # Add trainer-derived columns to canonical_v2 if missing (is_ASIA, session_id, minutes_*)
    cv2 = canonical_v2_df.copy()
    if "session_id" not in cv2.columns:
        ts = pd.to_datetime(cv2["time"], utc=True, errors="coerce")
        sess = get_session_vectorized(ts)
        sess_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
        cv2["session_id"] = sess.map(sess_map).fillna(0).astype(np.int32)
    if "is_ASIA" not in cv2.columns:
        cv2["is_ASIA"] = (cv2["session_id"].astype(int) == 0).astype(np.int8)
    if "minutes_since_session_open" not in cv2.columns:
        ts = pd.to_datetime(cv2["time"], utc=True, errors="coerce")
        cv2["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    if "minutes_to_next_session_boundary" not in cv2.columns:
        ts = pd.to_datetime(cv2["time"], utc=True, errors="coerce")
        cv2["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    if "session_change_flag" not in cv2.columns:
        sid = cv2["session_id"].astype(np.int32)
        cv2["session_change_flag"] = (sid.diff().fillna(0) != 0).astype(np.int8)
    if "session_tradable" not in cv2.columns:
        cv2["session_tradable"] = (cv2["session_id"].astype(int) != 0).astype(np.int8)

    missing = [c for c in xgb_features if c not in cv2.columns]
    if missing:
        raise RuntimeError(f"XGB feature missing in canonical_v2: {missing}")

    df_features = cv2[xgb_features].copy()
    print(f"[{ACTION}] sanitizing {len(df_features):,} rows × {len(xgb_features)} features...", flush=True)
    x_array, _stats = sanitizer.sanitize(df_features, feature_list=xgb_features, allow_nan_fill=True, nan_fill_value=0.0)

    # XGB inference per session
    model_path = Path(xgb_bundle_path) / "xgb_universal_multihead_v2.joblib"
    model = XGBMultiheadModel.load(str(model_path))
    df_features_sanitized = pd.DataFrame(x_array, columns=xgb_features, index=df_features.index)
    sess = cv2["session_id"].astype(int).to_numpy()
    sess_name_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}

    bridge_m5 = np.zeros((len(cv2), 7), dtype=np.float32)
    for sid_int in (0, 1, 2, 3):
        idx = np.where(sess == sid_int)[0]
        if idx.size == 0:
            continue
        sname = sess_name_map[sid_int]
        probs = model.predict_proba(df_features_sanitized.iloc[idx], session=sname, feature_list=xgb_features)
        if hasattr(probs, "p_long"):
            pl, ps, pf = np.asarray(probs.p_long), np.asarray(probs.p_short), np.asarray(probs.p_flat)
        else:
            pl, ps, pf = np.asarray(probs["p_long"]), np.asarray(probs["p_short"]), np.asarray(probs["p_flat"])
        bridge_input = np.column_stack([pl, ps, pf])
        bridge = proba_to_signal_bridge_v1(bridge_input)
        bridge_m5[idx] = bridge.astype(np.float32)

    # Map M5 → M1 via searchsorted (each M1 bar inherits the latest M5 XGB output ≤ its timestamp)
    m5_time_ns = pd.to_datetime(cv2["time"], utc=True).astype("int64").to_numpy()
    m5_sort_idx = np.argsort(m5_time_ns)
    m5_time_sorted = m5_time_ns[m5_sort_idx]
    bridge_m5_sorted = bridge_m5[m5_sort_idx]
    pos = np.searchsorted(m5_time_sorted, m1_time_ns, side="right") - 1
    pos = np.clip(pos, 0, len(m5_time_sorted) - 1)
    bridge_m1 = bridge_m5_sorted[pos]
    # Set first 7 features (XGB-bridge per V5)
    feat_mat[:, 0:7] = bridge_m1
    print(f"[{ACTION}] XGB-bridge populated for {n:,} M1 bars", flush=True)

    # ---- 2) Lazy-join canonical_v2 features for ctx_cont (11 V1 + 22 V2 ext) + ctx_cat (6) + swing (5) + m5_phase (5) + SMC (9) ----
    # Indices 12-49 + 50-79 in V5 feature list — anything that comes from canonical_v2 columns.
    # Build mapping: for each V5 feature name, if present in cv2, copy via M5→M1 alignment.
    print(f"[{ACTION}] lazy-joining canonical_v2 features to M1 grid...", flush=True)
    for v5_idx, fname in enumerate(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES):
        if v5_idx < 7:
            continue  # XGB-bridge already filled
        if fname in TRADE_STATE_FEATURE_NAMES:
            continue  # filled per-bar at emit time
        if fname in cv2.columns:
            col = pd.to_numeric(cv2[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            sorted_col = col[m5_sort_idx]
            feat_mat[:, v5_idx] = sorted_col[pos]
            continue
        # m5_phase one-hot: derive from M1 timestamp directly
        if fname.startswith("m5_phase_"):
            ph_idx = int(fname.split("_")[-1])
            ts = pd.to_datetime(m1_time_ns, utc=True)
            minute = pd.Series(ts).dt.minute.to_numpy()
            phase = (minute % 5).astype(np.int32)
            feat_mat[:, v5_idx] = (phase == ph_idx).astype(np.float32)
            continue
    print(f"[{ACTION}] precompute done. shape={feat_mat.shape}", flush=True)
    return m1_time_ns, feat_mat


# ---------------------------------------------------------------------------
# Per-trade record emission
# ---------------------------------------------------------------------------


def emit_v3_records_for_candidate(
    *,
    candidate_row: dict[str, Any],
    side: str,
    m1_time_ns: np.ndarray,
    m1_arrays: dict[str, np.ndarray],
    m1_feature_mat: np.ndarray,
    window_len: int,
    max_bars_per_trade: int,
    emit_stride: int,
) -> tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[str]]:
    """Emit thin V3 records + per-trade overlay array.

    Returns (records, overlay_array, trade_uid):
      - records: list of thin dicts (no io_features baked in — references via m1_idx_now + overlay_id)
      - overlay_array: (n_avail, 19) float32 — trade-state overlay for this trade
      - trade_uid: stable string ID for this trade
    """
    decision_ts_str = candidate_row.get("decision_ts_utc")
    if decision_ts_str is None:
        return [], None, None
    ts_ns = fwd_pipe._safe_decision_ts_to_int64_ns(decision_ts_str)
    if ts_ns is None:
        return [], None, None
    n_m1 = len(m1_time_ns)
    idx_start = int(np.searchsorted(m1_time_ns, ts_ns, side="right"))
    if idx_start >= n_m1:
        return [], None, None
    s_t, e_t, _status = fwd_pipe.slice_forward_window(
        m1_time_ns, n_m1, idx_start, max_bars_per_trade + 1, fwd_pipe.MAX_INTRA_GAP_MINUTES * 60 * 1_000_000_000,
    )
    if e_t - s_t < 2:
        return [], None, None

    bid_open = m1_arrays["bid_open"][s_t:e_t]
    bid_high = m1_arrays["bid_high"][s_t:e_t]
    bid_low = m1_arrays["bid_low"][s_t:e_t]
    bid_close = m1_arrays["bid_close"][s_t:e_t]
    ask_open = m1_arrays["ask_open"][s_t:e_t]
    ask_high = m1_arrays["ask_high"][s_t:e_t]
    ask_low = m1_arrays["ask_low"][s_t:e_t]
    ask_close = m1_arrays["ask_close"][s_t:e_t]
    bar_times_ns = m1_time_ns[s_t:e_t]

    cur_pnl, peak, trough, atr_bps_now, bar_ret = v1_pipe.compute_per_bar_signals(
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close, side,
    )
    n_avail = len(cur_pnl)
    cum_peak = np.maximum.accumulate(peak)
    cum_trough = np.minimum.accumulate(trough)

    arg_peak = np.zeros(n_avail, dtype=np.int32)
    running_max = -np.inf
    running_max_idx = 0
    for i in range(n_avail):
        if peak[i] >= running_max:
            running_max = float(peak[i])
            running_max_idx = i
        arg_peak[i] = running_max_idx

    pnl_vel = np.zeros(n_avail, dtype=np.float32)
    pnl_acc = np.zeros(n_avail, dtype=np.float32)
    if n_avail >= 2:
        pnl_vel[1:] = cur_pnl[1:] - cur_pnl[:-1]
    if n_avail >= 3:
        pnl_acc[2:] = pnl_vel[2:] - pnl_vel[1:-1]
    mfe_decay = np.zeros(n_avail, dtype=np.float32)
    if n_avail > 4:
        mfe_decay[4:] = cum_peak[4:] - cum_peak[:-4]
    pos_peak = np.maximum(cum_peak, 1e-6)
    giveback = np.clip((1.0 - cur_pnl / pos_peak), -10.0, 10.0).astype(np.float32)
    giveback_acc = np.zeros(n_avail, dtype=np.float32)
    if n_avail >= 3:
        gv_vel = np.zeros(n_avail, dtype=np.float32)
        gv_vel[1:] = giveback[1:] - giveback[:-1]
        giveback_acc[2:] = gv_vel[2:] - gv_vel[1:-1]
    rolling_slope = np.zeros(n_avail, dtype=np.float32)

    p_long_entry = float(candidate_row.get("p_long") or 0.0)
    p_hat_entry = float(candidate_row.get("p_hat") or 0.0)
    uncertainty_entry = float(candidate_row.get("uncertainty_score") or 0.0)
    p_short_entry = float(candidate_row.get("p_short") or 0.0)
    p_flat_entry = float(candidate_row.get("p_flat") or 0.0)
    margin_entry = float(candidate_row.get("margin") or 0.0)
    entropy_entry = v1_pipe._compute_entropy_at_entry(p_long_entry, p_short_entry, p_flat_entry)

    teacher_final_pnl_bps = float(cur_pnl[-1])
    teacher_final_mfe_bps = float(cum_peak.max() if n_avail > 0 else 0.0)
    teacher_final_mae_bps = float(cum_trough.min() if n_avail > 0 else 0.0)
    teacher_duration_bars = int(n_avail)

    # Build per-trade overlay (n_avail, 19) — cols 0-4 entry-snapshot (broadcast), 5-18 per-bar trade-state.
    overlay = np.zeros((n_avail, 19), dtype=np.float32)
    overlay[:, 0] = p_long_entry
    overlay[:, 1] = p_hat_entry
    overlay[:, 2] = uncertainty_entry
    overlay[:, 3] = entropy_entry
    overlay[:, 4] = margin_entry
    overlay[:, 5] = cur_pnl
    overlay[:, 6] = cum_peak
    overlay[:, 7] = cum_trough
    overlay[:, 8] = cum_peak - cur_pnl
    overlay[:, 9] = cum_peak - cur_pnl
    overlay[:, 10] = np.arange(n_avail, dtype=np.float32)
    overlay[:, 11] = np.arange(n_avail, dtype=np.float32) - arg_peak.astype(np.float32)
    overlay[:, 12] = mfe_decay
    overlay[:, 13] = pnl_vel
    overlay[:, 14] = pnl_acc
    overlay[:, 15] = rolling_slope
    overlay[:, 16] = atr_bps_now
    overlay[:, 17] = giveback
    overlay[:, 18] = giveback_acc

    trade_uid = f"{candidate_row.get('run_id', 'unknown')}:{candidate_row.get('candidate_uid', 'unknown')}:{side}"
    records: List[Dict[str, Any]] = []

    for t in range(1, n_avail, emit_stride):
        bar_ts_ns = int(bar_times_ns[t])
        m1_idx_now = int(np.searchsorted(m1_time_ns, bar_ts_ns, side="right")) - 1
        if m1_idx_now < window_len - 1:
            continue
        win_start = m1_idx_now - window_len + 1
        in_trade_start_in_win = max(0, s_t - win_start)
        in_trade_end_in_win = min(window_len, s_t + t - win_start + 1)
        n_in_trade_bars = max(0, in_trade_end_in_win - in_trade_start_in_win)
        # Map: overlay row range = [overlay_start_row, overlay_start_row + n_in_trade_bars)
        overlay_start_row = max(0, win_start - s_t)

        scalars = {
            "pnl_bps_now": float(cur_pnl[t]),
            "mfe_bps": float(cum_peak[t]),
            "mae_bps": float(cum_trough[t]),
            "dd_from_mfe_bps": float(cum_peak[t] - cur_pnl[t]),
            "distance_from_peak_mfe_bps": float(cum_peak[t] - cur_pnl[t]),
            "giveback_ratio": float(giveback[t]),
            "bars_held": float(t),
            "time_since_mfe_bars": float(t - arg_peak[t]),
            "atr_bps_now": float(atr_bps_now[t]),
            "rolling_slope_since_entry": float(rolling_slope[t]),
        }

        rec = {
            "ts": pd.Timestamp(bar_ts_ns, tz="UTC").isoformat(),
            "run_id": str(candidate_row.get("run_id", "")),
            "trade_uid": trade_uid,
            "trade_id": str(candidate_row.get("candidate_uid", "")),
            "side": side,
            "m1_idx_now": int(m1_idx_now),
            "in_trade_start_in_win": int(in_trade_start_in_win),
            "n_in_trade_bars": int(n_in_trade_bars),
            "overlay_start_row": int(overlay_start_row),
            "scalars": scalars,
            "teacher_final_pnl_bps": teacher_final_pnl_bps,
            "teacher_final_mfe_bps": teacher_final_mfe_bps,
            "teacher_final_mae_bps": teacher_final_mae_bps,
            "teacher_duration_bars": teacher_duration_bars,
        }
        records.append(rec)

    return records, overlay, trade_uid


# ---------------------------------------------------------------------------
# Per-week processing + streaming write
# ---------------------------------------------------------------------------


def process_week(
    *,
    week_dir: Path,
    m1_time_ns: np.ndarray,
    m1_arrays: dict[str, np.ndarray],
    m1_feature_mat: np.ndarray,
    out_fh,
    overlay_bin_fh,
    overlay_index_rows: List[Dict[str, Any]],
    overlay_offset_state: dict,
    window_len: int,
    max_bars_per_trade: int,
    emit_stride: int,
    directional_threshold: float,
) -> dict[str, Any]:
    rid = week_dir.name
    cand_path = week_dir / f"shadow_meta_candidates_{rid}_MERGED.parquet"
    stats: Dict[str, Any] = {"run_id": rid, "input_rows": 0, "output_records": 0, "status": "UNKNOWN"}
    if not cand_path.exists():
        stats["status"] = "INPUT_MISSING"
        return stats
    cand = pd.read_parquet(cand_path)
    stats["input_rows"] = len(cand)
    if len(cand) == 0:
        stats["status"] = "INPUT_EMPTY"
        return stats

    cand_dict = {
        col: cand[col].to_list() if col in cand.columns else [None] * len(cand)
        for col in (fwd_pipe.CANDIDATE_IDENTITY_COLS + fwd_pipe.CANDIDATE_FEATURE_COLS)
    }

    n_emitted = 0
    for row_i in range(len(cand)):
        candidate_row = {col: cand_dict[col][row_i] for col in cand_dict}
        side = v1_pipe.derive_side(candidate_row, threshold=directional_threshold)
        if side is None:
            side = "long" if (candidate_row.get("p_long") or 0) >= (candidate_row.get("p_short") or 0) else "short"
        recs, overlay, trade_uid = emit_v3_records_for_candidate(
            candidate_row=candidate_row,
            side=side,
            m1_time_ns=m1_time_ns,
            m1_arrays=m1_arrays,
            m1_feature_mat=m1_feature_mat,
            window_len=window_len,
            max_bars_per_trade=max_bars_per_trade,
            emit_stride=emit_stride,
        )
        if not recs or overlay is None or trade_uid is None:
            continue
        # Stream overlay rows to disk (raw float32 binary); track row offset/length per trade
        overlay_f32 = np.ascontiguousarray(overlay, dtype=np.float32)
        overlay_bin_fh.write(overlay_f32.tobytes())
        overlay_index_rows.append({
            "trade_uid": trade_uid,
            "overlay_offset": int(overlay_offset_state["rows"]),
            "overlay_length": int(overlay_f32.shape[0]),
        })
        overlay_offset_state["rows"] += int(overlay_f32.shape[0])
        for rec in recs:
            out_fh.write(json.dumps(rec) + "\n")
        n_emitted += len(recs)

    stats["output_records"] = n_emitted
    stats["status"] = "OK" if n_emitted > 0 else "EMPTY_OUTPUT"
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH),
                        help="Output DIRECTORY (not single file). Will contain matrix.npy + overlays.npy + records.jsonl + manifest.json")
    parser.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--m1-tape-root", type=str, default=str(DEFAULT_M1_TAPE_ROOT))
    parser.add_argument("--canonical-v2", type=str, default=str(DEFAULT_CANONICAL_V2_PATH))
    parser.add_argument("--xgb-bundle", type=str, default=str(DEFAULT_XGB_BUNDLE))
    parser.add_argument("--xgb-feature-contract", type=str, default=str(DEFAULT_XGB_FEATURE_CONTRACT))
    parser.add_argument("--xgb-sanitizer-config", type=str, default=str(DEFAULT_XGB_SANITIZER_CONFIG))
    parser.add_argument("--week", action="append", default=None)
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--max-bars-per-trade", type=int, default=DEFAULT_MAX_BARS_PER_TRADE)
    parser.add_argument("--emit-stride", type=int, default=DEFAULT_EMIT_STRIDE)
    parser.add_argument("--directional-threshold", type=float, default=DEFAULT_DIRECTIONAL_THRESHOLD)
    args = parser.parse_args()

    out_dir = Path(args.out_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ACTION}] loading M1 tape from {args.m1_tape_root}", flush=True)
    m1_df = fwd_pipe.load_m5_tape(Path(args.m1_tape_root))
    print(f"[{ACTION}] M1 bars loaded: {len(m1_df):,}", flush=True)

    print(f"[{ACTION}] loading canonical_v2 from {args.canonical_v2}", flush=True)
    cv2_df = pd.read_parquet(args.canonical_v2)
    cv2_df["time"] = pd.to_datetime(cv2_df["time"], utc=True)
    print(f"[{ACTION}] canonical_v2: {len(cv2_df):,} M5 rows × {len(cv2_df.columns)} cols", flush=True)

    m1_time_ns, m1_feature_mat = precompute_m1_feature_vectors(
        m1_df=m1_df,
        canonical_v2_df=cv2_df,
        xgb_bundle_path=Path(args.xgb_bundle),
        xgb_feature_contract_path=Path(args.xgb_feature_contract),
        xgb_sanitizer_config_path=Path(args.xgb_sanitizer_config),
    )

    # ---- Save the precomputed matrix ONCE ----
    matrix_path = out_dir / "m1_feature_matrix.npy"
    time_path = out_dir / "m1_time_ns.npy"
    print(f"[{ACTION}] saving M1 feature matrix → {matrix_path} ({m1_feature_mat.nbytes / 1e6:.1f} MB)", flush=True)
    np.save(matrix_path, m1_feature_mat)
    np.save(time_path, m1_time_ns)
    # Free the matrix copy that was returned (we still need it for processing — keep ref)

    m1_arrays = {
        col: m1_df[col].astype(np.float64).to_numpy()
        for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close")
    }

    week_dirs = sorted(
        d for d in Path(args.reports_root).iterdir()
        if d.is_dir() and fwd_pipe._is_truth_monfri_week(d.name)
    )
    if args.week is not None:
        only = set(args.week)
        week_dirs = [d for d in week_dirs if d.name in only]
    print(f"[{ACTION}] weeks to process: {len(week_dirs)}", flush=True)

    overlay_index_rows: List[Dict[str, Any]] = []
    overlay_offset_state: Dict[str, int] = {"rows": 0}
    per_week_stats: List[Dict[str, Any]] = []
    total_records = 0
    records_path = out_dir / "records.jsonl"
    overlays_bin_path = out_dir / "trade_state_overlays.f32"
    print(f"[{ACTION}] streaming records → {records_path}", flush=True)
    print(f"[{ACTION}] streaming overlays → {overlays_bin_path} (raw float32, shape (N,19))", flush=True)
    with records_path.open("w", encoding="utf-8") as out_fh, \
         overlays_bin_path.open("wb") as overlay_bin_fh:
        for w_idx, week_dir in enumerate(week_dirs, start=1):
            t0 = _time.time()
            stats = process_week(
                week_dir=week_dir,
                m1_time_ns=m1_time_ns,
                m1_arrays=m1_arrays,
                m1_feature_mat=m1_feature_mat,
                out_fh=out_fh,
                overlay_bin_fh=overlay_bin_fh,
                overlay_index_rows=overlay_index_rows,
                overlay_offset_state=overlay_offset_state,
                window_len=int(args.window_len),
                max_bars_per_trade=int(args.max_bars_per_trade),
                emit_stride=int(args.emit_stride),
                directional_threshold=float(args.directional_threshold),
            )
            per_week_stats.append(stats)
            total_records += int(stats.get("output_records", 0))
            elapsed = _time.time() - t0
            if w_idx % 5 == 0 or w_idx == len(week_dirs):
                print(
                    f"[{ACTION}] [{w_idx}/{len(week_dirs)}] {week_dir.name}: "
                    f"input={stats['input_rows']} -> records={stats['output_records']} "
                    f"status={stats['status']} trades={len(overlay_index_rows)} "
                    f"overlay_rows={overlay_offset_state['rows']} ({elapsed:.1f}s)",
                    flush=True,
                )
            # Flush periodically so partial state survives an early kill
            if w_idx % 20 == 0:
                out_fh.flush()
                overlay_bin_fh.flush()

    overlay_index_path = out_dir / "overlay_index.parquet"
    pd.DataFrame(overlay_index_rows).to_parquet(overlay_index_path, index=False)
    print(f"[{ACTION}] saved overlay_index → {overlay_index_path} "
          f"({len(overlay_index_rows)} trades, {overlay_offset_state['rows']} rows)", flush=True)

    manifest = {
        "action_v1": ACTION,
        "out_dir_v1": str(out_dir),
        "built_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "io_version": EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION,
        "feature_names_hash": EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH,
        "input_dim": int(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT),
        "window_len": int(args.window_len),
        "max_bars_per_trade": int(args.max_bars_per_trade),
        "emit_stride": int(args.emit_stride),
        "n_m1_bars": int(len(m1_time_ns)),
        "n_trades": int(len(overlay_index_rows)),
        "n_records": int(total_records),
        "trade_state_feature_indices": [int(i) for i in TRADE_STATE_FEATURE_INDICES],
        "trade_state_feature_names": list(TRADE_STATE_FEATURE_NAMES),
        "files": {
            "m1_feature_matrix": "m1_feature_matrix.npy",
            "m1_time_ns": "m1_time_ns.npy",
            "trade_state_overlays": "trade_state_overlays.f32",
            "trade_state_overlays_dtype": "float32",
            "trade_state_overlays_cols": 19,
            "trade_state_overlays_total_rows": int(overlay_offset_state["rows"]),
            "overlay_index": "overlay_index.parquet",
            "records": "records.jsonl",
        },
        "weeks_processed": len(per_week_stats),
        "weeks_ok": sum(1 for s in per_week_stats if s.get("status") == "OK"),
        "weeks_input_missing": sum(1 for s in per_week_stats if s.get("status") == "INPUT_MISSING"),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[{ACTION}] manifest at {manifest_path}", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
