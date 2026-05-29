#!/usr/bin/env python3
"""Build PER-M1-BAR exit-IQL dataset across all 2020-2026 hypothetical trades.

Mission
-------
M1-cadence successor to ``materialize_build_exit_iql_per_bar_dataset_v1.py``.

The M5 v1 builder samples one row every 4 M5 bars (= every 20 minutes), which
is structurally weaker than V3's per-M1 cadence: V3 inferences fire every
minute (5 inferences per M5 bar) using a 512 M1-bar window, but the V1
exit-IQL only decides every 20 minutes. An override head that decides 20x
slower than the model it overrides cannot meaningfully overrule V3.

This v2-M1 builder rebuilds the per-bar dataset at M1 cadence with full
V3 parity:
  - Walks the M1 canonical tape (xauusd_m1_bid_ask__CANONICAL).
  - Emits one row per held M1 bar (stride=1 by default — "eyes on").
  - K_HORIZONS rescaled to M1 units (same wall-clock horizons as v1).
  - Adds the 5 m5_phase features so the IQL distinguishes fresh-XGB-bar
    M1s (phase 0) from carried M1s (phase 1..4) — same signal V3 sees.
  - Adds M1 micro-features (recent 5/15/60 M1 returns + realized vol)
    so the IQL has explicit per-minute texture beyond the V10/V3 chunk_0
    features.

K_HORIZONS_EXIT_M1 = [1, 4, 12, 48, 144, 240] M1-bars (2026-05-24 SCALP)
                   ≈ [1min, 4min, 12min, 48min, 2.4h, 4h]
                   K is the LOOKAHEAD HORIZON for the Q-value's hold-max-pnl
                   estimate, NOT the exit time. Exit is governed by Strategi F
                   (profit-lock + giveback + IQL signal). K-set spans:
                     - K=1,4: immediate-next-bar decision gradient
                     - K=12,48: short-term planning window
                     - K=144,240: realistic-scalp-life lookahead
                   Previous swing-K [5, 20, 60, 120, 240, 480] caused trainer
                   K-mismatch (5/6 K-slots degenerate) and IQL_EXIT firing 2/14581.

Scaling architecture (15 GB host RAM, ~440M rows full build)
------------------------------------------------------------
Two design decisions make stride=1 max_bars=480 tractable on a 15 GB host:

  1. STREAMING WRITE. Process candidates in fixed-size batches; for each
     batch, build a small DataFrame and append to a single per-week parquet
     via pyarrow.parquet.ParquetWriter. Peak RAM ~= one batch's worth of
     rows (~700 MB for 50 candidates × 480 bars), not a full week.

  2. LAZY JOIN. Do NOT denormalize chunk_0 / canonical V10 features into
     the per-M1-bar dataset. Those features are M5-resolution (constant
     within each M5 bar), so denormalizing across 5 M1 bars would 5x the
     row width for zero information gain. Instead we emit only:
       - identity (run_id, candidate_uid, decision_ts_utc, side, bar_idx)
       - bar_ts_ns_v1 (int64) — anchor for downstream asof-join
       - per-bar trade-state (12 fields) + derivatives (6 fields)
       - M1 micro features (5 fields) + m5_phase one-hot (5 fields)
       - entry snapshot (5 fields, constant within trade)
       - candidate context (~13 fields, constant within trade)
       - rewards (EXIT_NOW + 6 K-horizons + quality flags)
     The trainer joins chunk_0 (per-week) and canonical (global) at load
     time using merge_asof on bar_ts_ns_v1 → floor_to_M5(bar_ts_ns_v1).
     This brings disk size for full M1 stride=1 from ~352 GB → ~70 GB.

Outputs
-------
Per-week parquet under per_week/ + manifest. Lazy-join schema (~50 cols).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    compute_m5_phase_index,
)
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe
from gx1.scripts import materialize_build_exit_iql_per_bar_dataset_v1 as v1_pipe


ACTION = "BUILD_EXIT_IQL_PER_BAR_DATASET_V2_M1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_M1_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL"
)
DEFAULT_OUT_ROOT = DEFAULT_REPORTS_ROOT / "EXIT_IQL_PER_BAR_DATASET_V2_M1"

# Wall-clock horizons identical to v1 (M5 [1,4,12,24,48,96] = 5/20min/1/2/4/8h).
# Expressed in M1 bars: 1×5, 4×5, 12×5, 24×5, 48×5, 96×5.
K_HORIZONS_EXIT_M1 = [1, 4, 12, 48, 144, 240]  # SCALP lookahead horizons
N_K_EXIT_M1 = len(K_HORIZONS_EXIT_M1)
MAX_K_EXIT_M1 = max(K_HORIZONS_EXIT_M1)

DEFAULT_BAR_STRIDE = 1               # every M1 bar — full "eyes on" cadence
DEFAULT_MAX_BARS_PER_TRADE = 480     # 8h in M1 units, matches v1's 8h cap (96 M5)
DEFAULT_DIRECTIONAL_THRESHOLD = 0.05

# Reuse v1's candidate identity / context column lists (entry-side context
# is unchanged; only the per-bar walk is M1).
CANDIDATE_CONTEXT_COLS = list(fwd_pipe.CANDIDATE_FEATURE_COLS)
CANDIDATE_IDENTITY_COLS = list(fwd_pipe.CANDIDATE_IDENTITY_COLS)
CHUNK0_FEATURE_COLS = list(fwd_pipe.CHUNK0_FEATURE_COLS)
CHUNK0_SUFFIX = fwd_pipe.CHUNK0_SUFFIX
CANONICAL_FEATURES_SUFFIX = fwd_pipe.CANONICAL_FEATURES_SUFFIX
DEFAULT_CANONICAL_FEATURES_PATH = fwd_pipe.DEFAULT_CANONICAL_FEATURES_PATH

# Trade-state derivatives — same names as v1, but recomputed at M1 cadence.
# v1 also exports these columns; downstream training code reads by name so
# semantics carry over (Δ per bar = Δ per minute now instead of per 5 min).
TRADE_STATE_DERIVATIVE_COLS = list(v1_pipe.TRADE_STATE_DERIVATIVE_COLS)
ENTRY_SNAPSHOT_COLS = list(v1_pipe.ENTRY_SNAPSHOT_COLS)

# M5 tape gap policy: 180 minutes. M1 tape has same weekend gap structure;
# reuse v1's max_gap_ns directly (it's expressed in ns, cadence-agnostic).
MAX_INTRA_GAP_NS = fwd_pipe.MAX_INTRA_GAP_MINUTES * 60 * 1_000_000_000

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json


# ---------------------------------------------------------------------------
# Per-bar state + reward — M1 variant
# ---------------------------------------------------------------------------


def _compute_m1_micro_features(
    bar_close_mid: np.ndarray,  # mid close per M1 bar
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cheap M1-resolution micro features that V10/V3 chunk_0 cannot see.

    Returns arrays aligned to bar_close_mid:
      r5, r15, r60, vol15, vol60
    where r{N} is N-bar log-return in bps and vol{N} is realized vol of
    bar-to-bar returns over last N M1 bars (in bps).
    """
    n = len(bar_close_mid)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, empty

    safe = np.maximum(bar_close_mid, 1e-6)
    log_close = np.log(safe)

    def _rolling_return_bps(lag: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if n > lag:
            out[lag:] = ((log_close[lag:] - log_close[:-lag]) * 10000.0).astype(np.float32)
        return out

    r5 = _rolling_return_bps(5)
    r15 = _rolling_return_bps(15)
    r60 = _rolling_return_bps(60)

    bar_log_ret = np.zeros(n, dtype=np.float64)
    if n > 1:
        bar_log_ret[1:] = log_close[1:] - log_close[:-1]
    bar_log_ret_bps = bar_log_ret * 10000.0

    def _rolling_vol_bps(window: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if n < 2:
            return out
        sq = bar_log_ret_bps * bar_log_ret_bps
        cum = np.concatenate(([0.0], np.cumsum(sq)))
        for i in range(n):
            lo = max(0, i - window + 1)
            count = i - lo + 1
            if count < 2:
                continue
            mean_sq = (cum[i + 1] - cum[lo]) / count
            out[i] = float(np.sqrt(max(mean_sq, 0.0)))
        return out

    vol15 = _rolling_vol_bps(15)
    vol60 = _rolling_vol_bps(60)
    return r5, r15, r60, vol15, vol60


def emit_per_bar_rows_m1(
    candidate_row: dict[str, Any],
    bid_open: np.ndarray, bid_high: np.ndarray, bid_low: np.ndarray, bid_close: np.ndarray,
    ask_open: np.ndarray, ask_high: np.ndarray, ask_low: np.ndarray, ask_close: np.ndarray,
    bar_times_ns: np.ndarray,            # per-M1-bar timestamp (anchor for downstream lazy-join)
    *,
    side: str,
    bar_stride: int,
    max_bars_per_trade: int,
    k_horizons: list[int],
) -> list[dict[str, Any]]:
    """M1 variant of v1.emit_per_bar_rows. One row per sampled M1 bar in trade.

    Lazy-join schema: chunk_0 / canonical V10 features are NOT denormalized
    into the per-bar rows here. Each row carries bar_ts_ns_v1 so the trainer
    can asof-join the M5-resolution context features at load time.

    Differences from v1:
      - K_HORIZONS in M1 units (forward window 480+ M1 bars).
      - bar_ts_ns_v1 added per row.
      - 5 m5_phase one-hot features (m5_phase_{0..4}_v1).
      - 5 M1 micro features (m1_last_*_return_bps_v1, m1_realized_vol_*_bps_v1).
      - chunk_0 / canonical denorm REMOVED (lazy-join at training time).
    """
    n_avail = len(bid_open)
    if n_avail < 2:
        return []

    cur_pnl, peak, trough, atr_bps, bar_ret = v1_pipe.compute_per_bar_signals(
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close,
        side,
    )

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

    entry_snapshot = {
        "p_long_entry_v1": candidate_row.get("p_long"),
        "p_hat_entry_v1": candidate_row.get("p_hat"),
        "uncertainty_entry_v1": candidate_row.get("uncertainty_score"),
        "entropy_entry_v1": v1_pipe._compute_entropy_at_entry(
            candidate_row.get("p_long"),
            candidate_row.get("p_short"),
            candidate_row.get("p_flat"),
        ),
        "margin_entry_v1": candidate_row.get("margin"),
    }

    n_full = len(cur_pnl)
    pnl_vel = np.zeros(n_full, dtype=np.float32)
    pnl_acc = np.zeros(n_full, dtype=np.float32)
    if n_full >= 2:
        pnl_vel[1:] = cur_pnl[1:] - cur_pnl[:-1]
    if n_full >= 3:
        pnl_acc[2:] = pnl_vel[2:] - pnl_vel[1:-1]
    mfe_decay = np.zeros(n_full, dtype=np.float32)
    if n_full > 4:
        mfe_decay[4:] = cum_peak[4:] - cum_peak[:-4]
    pos_peak = np.maximum(cum_peak, 1e-6)
    giveback = (1.0 - cur_pnl / pos_peak).astype(np.float32)
    giveback = np.clip(giveback, -10.0, 10.0)
    giveback_acc = np.zeros(n_full, dtype=np.float32)
    if n_full >= 3:
        gv_vel = np.zeros(n_full, dtype=np.float32)
        gv_vel[1:] = giveback[1:] - giveback[:-1]
        giveback_acc[2:] = gv_vel[2:] - gv_vel[1:-1]
    rolling_slope = np.zeros(n_full, dtype=np.float32)
    if n_full >= 3:
        idx = np.arange(n_full, dtype=np.float64)
        y64 = cur_pnl.astype(np.float64)
        cum_y = np.cumsum(y64)
        cum_xy = np.cumsum(idx * y64)
        for i in range(2, n_full):
            n = i + 1
            sum_x = i * n / 2.0
            sum_x2 = i * n * (2 * i + 1) / 6.0
            sum_y = cum_y[i]
            sum_xy = cum_xy[i]
            denom = n * sum_x2 - sum_x * sum_x
            if abs(denom) > 1e-9:
                rolling_slope[i] = float((n * sum_xy - sum_x * sum_y) / denom)

    # M1 micro features — computed over the in-trade slice.
    # (Strictly we'd want them over a pre-trade lookback too; chunk_0 already
    # encodes the longer-horizon context, so these per-bar microfeatures
    # capture intra-trade tape behavior the M5 chunk_0 cannot.)
    bar_close_mid = ((ask_close + bid_close) / 2.0).astype(np.float64)
    m1_r5, m1_r15, m1_r60, m1_vol15, m1_vol60 = _compute_m1_micro_features(bar_close_mid)

    rows: list[dict[str, Any]] = []
    last_t = min(n_avail, max_bars_per_trade + 1)
    # Skip t=0 (entry M1 bar — no exit decision yet); start from t=1.
    for t in range(1, last_t, bar_stride):
        remaining_peak = peak[t + 1: t + 1 + max(k_horizons)]
        n_remaining = len(remaining_peak)
        if n_remaining == 0:
            hold_rewards = {f"hold_max_pnl_K{K}_v1": float(cur_pnl[t]) for K in k_horizons}
            forced_terminal = True
        else:
            hold_rewards = {}
            for K in k_horizons:
                K_eff = min(K, n_remaining)
                hold_rewards[f"hold_max_pnl_K{K}_v1"] = float(np.max(remaining_peak[:K_eff]))
            forced_terminal = (n_remaining < min(k_horizons))

        bar_ts_ns = int(bar_times_ns[t]) if t < len(bar_times_ns) else 0

        # m5_phase one-hot from M1 timestamp
        if bar_ts_ns > 0:
            ts_val = pd.Timestamp(bar_ts_ns, tz="UTC")
            phase_idx = compute_m5_phase_index(ts_val)
        else:
            phase_idx = 0
        m5_phase = {f"m5_phase_{i}_v1": (1.0 if i == phase_idx else 0.0) for i in range(5)}

        row = {
            # Identity
            "run_id": candidate_row.get("run_id"),
            "candidate_uid": candidate_row.get("candidate_uid"),
            "decision_ts_utc": candidate_row.get("decision_ts_utc"),
            "side_v1": side,
            "bar_idx_v1": int(t),
            # Anchor timestamp — trainer uses this for asof-join to chunk_0 + canonical.
            "bar_ts_ns_v1": int(bar_ts_ns),
            # Per-bar trade state
            "bars_in_trade_v1": int(t),
            "current_unrealized_pnl_bps_v1": float(cur_pnl[t]),
            "current_mfe_bps_v1": float(cum_peak[t]),
            "current_mae_bps_v1": float(cum_trough[t]),
            "bars_since_mfe_peak_v1": int(t - arg_peak[t]),
            "pnl_drawdown_from_peak_v1": float(cum_peak[t] - cur_pnl[t]),
            "current_atr_bps_v1": float(atr_bps[t]),
            "bar_return_bps_v1": float(bar_ret[t]),
            # EXIT_NOW reward (independent of K)
            "exit_now_reward_bps_v1": float(cur_pnl[t]),
            # HOLD rewards per K (M1-units)
            **hold_rewards,
            # Quality
            "forced_terminal_v1": bool(forced_terminal),
            "forward_bars_remaining_v1": int(n_remaining),
            # Cadence marker so downstream code can detect M1 vs M5 datasets
            "cadence_v1": "M1",
        }

        for col in CANDIDATE_CONTEXT_COLS:
            if col in candidate_row:
                row[col] = candidate_row[col]

        row["pnl_velocity_v1"] = float(pnl_vel[t]) if t < len(pnl_vel) else 0.0
        row["pnl_acceleration_v1"] = float(pnl_acc[t]) if t < len(pnl_acc) else 0.0
        row["mfe_decay_rate_v1"] = float(mfe_decay[t]) if t < len(mfe_decay) else 0.0
        row["giveback_ratio_v1"] = float(giveback[t]) if t < len(giveback) else 0.0
        row["giveback_acceleration_v1"] = float(giveback_acc[t]) if t < len(giveback_acc) else 0.0
        row["rolling_slope_since_entry_v1"] = float(rolling_slope[t]) if t < len(rolling_slope) else 0.0

        row.update(m5_phase)
        row["m1_last_5bar_return_bps_v1"] = float(m1_r5[t]) if t < len(m1_r5) else 0.0
        row["m1_last_15bar_return_bps_v1"] = float(m1_r15[t]) if t < len(m1_r15) else 0.0
        row["m1_last_60bar_return_bps_v1"] = float(m1_r60[t]) if t < len(m1_r60) else 0.0
        row["m1_realized_vol_15bar_bps_v1"] = float(m1_vol15[t]) if t < len(m1_vol15) else 0.0
        row["m1_realized_vol_60bar_bps_v1"] = float(m1_vol60[t]) if t < len(m1_vol60) else 0.0

        row.update(entry_snapshot)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-week processing
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


DEFAULT_CANDIDATES_PER_BATCH = 50  # streaming-write batch size — peak RAM ~700 MB


def _rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema | None) -> pa.Table:
    """Convert list-of-dicts to pa.Table, coercing to a locked schema if given."""
    df = pd.DataFrame(rows)
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    # Align columns to schema order; missing cols filled with nulls.
    for col in schema.names:
        if col not in df.columns:
            df[col] = None
    df = df[list(schema.names)]
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)


def process_week_m1(
    week_dir: Path,
    m1_time_ns: np.ndarray,
    m1_arrays: dict[str, np.ndarray],
    *,
    out_path: Path,
    bar_stride: int,
    max_bars_per_trade: int,
    directional_threshold: float,
    directional_only: bool,
    max_gap_ns: int,
    candidates_per_batch: int = DEFAULT_CANDIDATES_PER_BATCH,
    schema_template: pa.Schema | None = None,
) -> tuple[dict[str, Any], pa.Schema | None]:
    """Stream per-M1-bar rows for one week to ``out_path``. Returns (stats, schema).

    Memory bound: at most ``candidates_per_batch`` candidates' worth of rows
    held in memory at once. Schema is locked from the first written batch and
    reused for subsequent batches/weeks for cross-week parquet compatibility.
    """
    rid = week_dir.name
    cand_path = week_dir / f"shadow_meta_candidates_{rid}_MERGED.parquet"
    stats: dict[str, Any] = {
        "run_id": rid,
        "input_path": str(cand_path),
        "output_path": str(out_path),
        "input_sha256": None,
        "input_rows": 0,
        "output_rows": 0,
        "skipped_no_decision_ts": 0,
        "skipped_no_directional_side": 0,
        "skipped_decision_past_tape_end": 0,
        "process_status": "UNKNOWN",
        "cadence_v1": "M1",
    }
    if not cand_path.exists():
        stats["process_status"] = "INPUT_MISSING"
        return stats, schema_template
    stats["input_sha256"] = _file_sha256(cand_path)
    cand = pd.read_parquet(cand_path)
    stats["input_rows"] = len(cand)
    if len(cand) == 0:
        stats["process_status"] = "INPUT_EMPTY"
        return stats, schema_template

    n_m1 = len(m1_time_ns)
    cand_dict = {col: cand[col].to_list() if col in cand.columns else [None] * len(cand)
                 for col in CANDIDATE_IDENTITY_COLS + CANDIDATE_CONTEXT_COLS}

    writer: pq.ParquetWriter | None = None
    schema = schema_template
    pending: list[dict[str, Any]] = []
    pending_candidates = 0

    def _flush() -> None:
        nonlocal writer, schema, pending, pending_candidates
        if not pending:
            pending_candidates = 0
            return
        table = _rows_to_table(pending, schema)
        if schema is None:
            schema = table.schema
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
        writer.write_table(table)
        stats["output_rows"] += len(pending)
        pending = []
        pending_candidates = 0

    try:
        for row_i in range(len(cand)):
            decision_ts_str = cand_dict["decision_ts_utc"][row_i]
            ts_ns = fwd_pipe._safe_decision_ts_to_int64_ns(decision_ts_str)
            if ts_ns is None:
                stats["skipped_no_decision_ts"] += 1
                continue
            candidate_row = {col: cand_dict[col][row_i] for col in cand_dict}
            side = v1_pipe.derive_side(candidate_row, threshold=directional_threshold)
            if side is None:
                if directional_only:
                    stats["skipped_no_directional_side"] += 1
                    continue
                side = "long" if (candidate_row.get("p_long") or 0) >= (candidate_row.get("p_short") or 0) else "short"

            idx_start = int(np.searchsorted(m1_time_ns, ts_ns, side="right"))
            if idx_start >= n_m1:
                stats["skipped_decision_past_tape_end"] += 1
                continue
            s_t, e_t, _ = fwd_pipe.slice_forward_window(
                m1_time_ns, n_m1, idx_start, max_bars_per_trade + max(K_HORIZONS_EXIT_M1) + 1, max_gap_ns,
            )
            bar_times_ns = m1_time_ns[s_t:e_t]
            rows = emit_per_bar_rows_m1(
                candidate_row,
                m1_arrays["bid_open"][s_t:e_t],
                m1_arrays["bid_high"][s_t:e_t],
                m1_arrays["bid_low"][s_t:e_t],
                m1_arrays["bid_close"][s_t:e_t],
                m1_arrays["ask_open"][s_t:e_t],
                m1_arrays["ask_high"][s_t:e_t],
                m1_arrays["ask_low"][s_t:e_t],
                m1_arrays["ask_close"][s_t:e_t],
                bar_times_ns,
                side=side,
                bar_stride=bar_stride,
                max_bars_per_trade=max_bars_per_trade,
                k_horizons=K_HORIZONS_EXIT_M1,
            )
            pending.extend(rows)
            pending_candidates += 1
            if pending_candidates >= candidates_per_batch:
                _flush()

        _flush()
    finally:
        if writer is not None:
            writer.close()

    stats["process_status"] = "OK" if stats["output_rows"] > 0 else "EMPTY_OUTPUT"
    return stats, schema


def write_artifacts(
    out_root: Path | None = None,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    m1_tape_root: Path = DEFAULT_M1_TAPE_ROOT,
    canonical_features_path: Path | None = None,
    only_weeks: list[str] | None = None,
    skip_existing: bool = True,
    bar_stride: int = DEFAULT_BAR_STRIDE,
    max_bars_per_trade: int = DEFAULT_MAX_BARS_PER_TRADE,
    directional_threshold: float = DEFAULT_DIRECTIONAL_THRESHOLD,
    directional_only: bool = False,
    candidates_per_batch: int = DEFAULT_CANDIDATES_PER_BATCH,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (reports_root / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)
    per_week_dir = artifact_root / "per_week"
    per_week_dir.mkdir(exist_ok=True)

    print(f"[{ACTION}] loading M1 tape from {m1_tape_root}", flush=True)
    m1 = fwd_pipe.load_m5_tape(m1_tape_root)  # generic year-partition loader
    print(f"[{ACTION}] M1 bars loaded: {len(m1):,}  range={m1['time'].min()} -> {m1['time'].max()}", flush=True)

    # NOTE: chunk_0 / canonical features are NOT loaded here. The trainer joins
    # them lazily at training time (see header). Path recorded for reference.
    canonical_path = canonical_features_path or DEFAULT_CANONICAL_FEATURES_PATH
    print(f"[{ACTION}] canonical features path (lazy-join target): {canonical_path}", flush=True)

    m1_time_ns = m1["time"].astype("int64").to_numpy()
    m1_arrays = {
        col: m1[col].astype(np.float64).to_numpy()
        for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close")
    }
    # Free the pandas DataFrame — the numpy arrays carry the data we need.
    del m1

    week_dirs = sorted(
        d for d in reports_root.iterdir()
        if d.is_dir() and fwd_pipe._is_truth_monfri_week(d.name)
    )
    if only_weeks is not None:
        only_set = set(only_weeks)
        week_dirs = [d for d in week_dirs if d.name in only_set]
    print(f"[{ACTION}] weeks to process: {len(week_dirs)}", flush=True)
    print(f"[{ACTION}] params: bar_stride={bar_stride} max_bars_per_trade={max_bars_per_trade} "
          f"directional_threshold={directional_threshold} directional_only={directional_only} "
          f"candidates_per_batch={candidates_per_batch} k_horizons_m1={K_HORIZONS_EXIT_M1}", flush=True)

    per_week_stats: list[dict[str, Any]] = []
    total_rows = 0
    schema_template: pa.Schema | None = None
    for w_idx, week_dir in enumerate(week_dirs, start=1):
        out_path = per_week_dir / f"exit_per_bar_m1_{week_dir.name}.parquet"
        stats_path = per_week_dir / f"exit_per_bar_m1_{week_dir.name}_stats.json"
        if skip_existing and out_path.exists() and stats_path.exists():
            try:
                cached = json.loads(stats_path.read_text())
                per_week_stats.append(cached)
                total_rows += int(cached.get("output_rows", 0))
                continue
            except (json.JSONDecodeError, OSError):
                pass

        stats, schema_template = process_week_m1(
            week_dir, m1_time_ns, m1_arrays,
            out_path=out_path,
            bar_stride=bar_stride,
            max_bars_per_trade=max_bars_per_trade,
            directional_threshold=directional_threshold,
            directional_only=directional_only,
            max_gap_ns=MAX_INTRA_GAP_NS,
            candidates_per_batch=candidates_per_batch,
            schema_template=schema_template,
        )
        per_week_stats.append(stats)
        total_rows += int(stats.get("output_rows", 0))
        stats_path.write_text(json.dumps(_jsonable(stats), indent=2, sort_keys=True))
        if w_idx % 5 == 0 or w_idx == len(week_dirs):
            print(f"[{ACTION}] [{w_idx}/{len(week_dirs)}] {week_dir.name}: input={stats['input_rows']} -> output={stats['output_rows']} status={stats['process_status']}", flush=True)

    summary = {
        "layer_name": "EXIT_IQL_PER_BAR_DATASET_SUMMARY_V2_M1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "cadence_v1": "M1",
        "m1_tape_root_v1": str(m1_tape_root),
        "m1_bars_loaded_v1": int(len(m1_time_ns)),
        "canonical_features_path_v1": str(canonical_path),
        "k_horizons_exit_m1_v1": K_HORIZONS_EXIT_M1,
        "bar_stride_v1": bar_stride,
        "max_bars_per_trade_v1": max_bars_per_trade,
        "candidates_per_batch_v1": candidates_per_batch,
        "directional_threshold_v1": directional_threshold,
        "directional_only_v1": directional_only,
        "lazy_join_v1": True,
        "anchor_column_v1": "bar_ts_ns_v1",
        "weeks_processed_v1": len(per_week_stats),
        "weeks_ok_v1": sum(1 for s in per_week_stats if s.get("process_status") == "OK"),
        "weeks_input_missing_v1": sum(1 for s in per_week_stats if s.get("process_status") == "INPUT_MISSING"),
        "total_input_rows_v1": int(sum(int(s.get("input_rows", 0)) for s in per_week_stats)),
        "total_output_rows_v1": int(total_rows),
        "research_only_v1": True,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "manifest_v1.json",
        {
            "layer_id_v1": ACTION,
            "built_at_utc_v1": summary["built_at_utc_v1"],
            "output_dir_v1": str(artifact_root),
            "per_week_dir_v1": str(per_week_dir),
            "summary_path_v1": str(artifact_root / "summary_v1.json"),
            "per_week_stats_v1": per_week_stats,
        },
    )
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--m1-root", type=str, default=str(DEFAULT_M1_TAPE_ROOT))
    parser.add_argument("--canonical-features", type=str, default=str(DEFAULT_CANONICAL_FEATURES_PATH))
    parser.add_argument("--week", action="append", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--bar-stride", type=int, default=DEFAULT_BAR_STRIDE)
    parser.add_argument("--max-bars-per-trade", type=int, default=DEFAULT_MAX_BARS_PER_TRADE)
    parser.add_argument("--directional-threshold", type=float, default=DEFAULT_DIRECTIONAL_THRESHOLD)
    parser.add_argument("--directional-only", action="store_true")
    parser.add_argument("--candidates-per-batch", type=int, default=DEFAULT_CANDIDATES_PER_BATCH)
    parser.add_argument("--built-at-utc", type=str, default=None)
    parser.add_argument("--vedtak", type=str, default=None,
                        help="REQUIRED retrain vedtak (gx1_guards gate). Short reason string.")
    args = parser.parse_args()

    # Retrain-vedtak gate (no auto-retrains).
    try:
        from gx1_guards.gates import require_retrain_vedtak, GateError
        try:
            require_retrain_vedtak(args.vedtak)
        except GateError as e:
            parser.error(str(e))
    except ImportError:
        if not args.vedtak:
            parser.error("--vedtak is required (gx1_guards unavailable; pass --vedtak anyway).")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(
        out_root=out_root,
        reports_root=Path(args.reports_root).expanduser().resolve(),
        m1_tape_root=Path(args.m1_root).expanduser().resolve(),
        canonical_features_path=Path(args.canonical_features).expanduser().resolve(),
        only_weeks=args.week,
        skip_existing=not args.no_skip_existing,
        bar_stride=args.bar_stride,
        max_bars_per_trade=args.max_bars_per_trade,
        directional_threshold=args.directional_threshold,
        directional_only=args.directional_only,
        candidates_per_batch=args.candidates_per_batch,
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
