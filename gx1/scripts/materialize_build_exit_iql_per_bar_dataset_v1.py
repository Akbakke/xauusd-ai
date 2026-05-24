#!/usr/bin/env python3
"""Build per-bar exit-IQL dataset across all 2020-2026 hypothetical trades.

Mission
-------
Mirror of `materialize_build_candidate_forward_outcome_dataset_v1.py` but for
PER-BAR exit decisions inside a held trade. For each candidate × side, simulate
the hypothetical trade's evolution bar-by-bar over the forward M5 window, and
emit one training row per bar.

Each row captures the per-bar state at time t and the counterfactual rewards
for both possible actions:
  - EXIT_NOW: realized pnl if we close position at bar t (signed, can be negative)
  - HOLD:     max-future-pnl over remaining K bars (best outcome by holding)

This gives Exit-IQL a fully-observable counterfactual signal:
  "If you exit now, you get X bps. If you hold, the best you can get in the
   next K bars is Y bps. Choose the action with higher Q-value."

The trained Exit-IQL becomes the "smart RL head" that supervises and overrides
the Exit-Transformer (V3) — same pattern as Entry-IQL guides V10.

Action space (2)
----------------
  HOLD, EXIT_NOW

State per bar (per (candidate, side, bar_t))
--------------------------------------------
Trade-state features (relative, not absolute):
  bars_in_trade_v1                  : t (how long we've been in trade)
  current_unrealized_pnl_bps_v1     : (current_close - entry_fill) signed bps
  current_mfe_bps_v1                : peak unrealized over [0..t]
  current_mae_bps_v1                : trough unrealized over [0..t]
  bars_since_mfe_peak_v1            : t - argmax(unrealized[0..t])
  pnl_drawdown_from_peak_v1         : current_mfe - current_pnl
  current_atr_bps_v1                : volatility now (M5 bar high-low/close in bps)
  bar_return_bps_v1                 : last bar's return
  forward_max_pnl_remaining_K_v1    : max(pnl[t+1..t+K]) per K — used as HOLD reward

Plus all candidate-level state (for context — same as entry-IQL).

Counterfactual rewards (per K-horizon)
--------------------------------------
EXIT_NOW: signed pnl at bar t (independent of K)
HOLD @ K: max future pnl over bars [t+1..t+K] minus exit-cost-adjustment

The HOLD reward depends on K because the agent's bootstrap horizon matters:
  - HOLD @ K=1 says "what's the immediate next-bar best?"
  - HOLD @ K=48 says "what's the best over the next 4 hours?"

K_HORIZONS = [1, 4, 12, 24, 48, 96] M5-bars (5min, 20min, 1h, 2h, 4h, 8h
horizons — exit decisions are typically shorter-horizon than entry decisions).

Tractability scoping
--------------------
Full universe: 844k candidates × 2 sides × ~192 bars = ~324M rows. Too big.

Default strategy:
  - Per-trade subsample: keep every 4th bar  (downsample 192→48 bars)
  - Per-side: derive ONE side per candidate from p_long vs p_short
  - Result: ~844k × 48 = ~40M rows. Still large but tractable on disk (~10GB).

Configurable scope:
  --bar-stride N   : keep every N-th bar (default 4)
  --max-bars-per-trade B : truncate at B bars (default 96 — covers swing horizon)
  --directional-only : only candidates where p_max_dir > 0.05 (much smaller dataset)

Outputs
-------
Per-week parquet under per_week/ + manifest. Aggregable into single dataset.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe


ACTION = "BUILD_EXIT_IQL_PER_BAR_DATASET_V1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_M5_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"
)
DEFAULT_OUT_ROOT = DEFAULT_REPORTS_ROOT / "EXIT_IQL_PER_BAR_DATASET_V1"

K_HORIZONS_EXIT = [1, 4, 12, 48, 144, 240]  # 2026-05-24 SCALP — matches v2_m1 dataset builder
N_K_EXIT = len(K_HORIZONS_EXIT)
MAX_K_EXIT = max(K_HORIZONS_EXIT)
DEFAULT_BAR_STRIDE = 4              # keep every Nth bar to bound dataset size
DEFAULT_MAX_BARS_PER_TRADE = 96     # truncate hypothetical trade at this bar
DEFAULT_DIRECTIONAL_THRESHOLD = 0.05  # candidate considered directional if max(p_long, p_short) > this

# Carry-over candidate context columns (same as entry-IQL).
CANDIDATE_CONTEXT_COLS = list(fwd_pipe.CANDIDATE_FEATURE_COLS)
CANDIDATE_IDENTITY_COLS = list(fwd_pipe.CANDIDATE_IDENTITY_COLS)

# Per-bar V10/V3 feature columns from chunk_0_data.parquet — joined per held bar.
CHUNK0_FEATURE_COLS = list(fwd_pipe.CHUNK0_FEATURE_COLS)
CHUNK0_SUFFIX = fwd_pipe.CHUNK0_SUFFIX
CANONICAL_FEATURES_SUFFIX = fwd_pipe.CANONICAL_FEATURES_SUFFIX
DEFAULT_CANONICAL_FEATURES_PATH = fwd_pipe.DEFAULT_CANONICAL_FEATURES_PATH

# Trade-state-derivative features (V3 uses these per held bar)
TRADE_STATE_DERIVATIVE_COLS = [
    "pnl_velocity_v1",            # Δpnl per bar
    "pnl_acceleration_v1",        # Δ²pnl per bar
    "mfe_decay_rate_v1",          # peak-decay over last 4 bars
    "giveback_ratio_v1",          # 1 - current_pnl / max(peak_mfe, 1)
    "giveback_acceleration_v1",
    "rolling_slope_since_entry_v1",
]
# Entry-snapshot fields carried through the trade (from candidate row)
ENTRY_SNAPSHOT_COLS = [
    "p_long_entry_v1",
    "p_hat_entry_v1",
    "uncertainty_entry_v1",
    "entropy_entry_v1",
    "margin_entry_v1",
]

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json


# ---------------------------------------------------------------------------
# Per-bar state + reward computation
# ---------------------------------------------------------------------------


def _compute_entropy_at_entry(p_long: Any, p_short: Any, p_flat: Any) -> float:
    """Shannon entropy of (p_long, p_short, p_flat). Returns 0.0 on missing."""
    s = 0.0
    for p in (p_long, p_short, p_flat):
        try:
            pf = float(p) if p is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        if pf > 1e-12 and np.isfinite(pf):
            s -= pf * float(np.log(pf))
    return float(s)


def derive_side(candidate: dict[str, Any], threshold: float) -> str | None:
    """Return 'long' or 'short' or None.

    'long' if p_long > p_short and max(p_long, p_short) > threshold;
    'short' if p_short > p_long and max(p_long, p_short) > threshold;
    None otherwise (directional confidence too low — skip the candidate).
    """
    pl = candidate.get("p_long")
    ps = candidate.get("p_short")
    try:
        pl_f = float(pl) if pl is not None else 0.0
        ps_f = float(ps) if ps is not None else 0.0
    except (TypeError, ValueError):
        return None
    if max(pl_f, ps_f) < threshold:
        return None
    return "long" if pl_f >= ps_f else "short"


def compute_per_bar_signals(
    bid_open: np.ndarray, bid_high: np.ndarray, bid_low: np.ndarray, bid_close: np.ndarray,
    ask_open: np.ndarray, ask_high: np.ndarray, ask_low: np.ndarray, ask_close: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-bar signals: (current_pnl_close, peak_high, trough_low, atr_bps, bar_return_bps).

    All bps relative to entry fill (long: ask_open[0]; short: bid_open[0]).
    current_pnl_close: pnl if we closed at bar's close-side
    peak_high: bar's best favorable price
    trough_low: bar's worst adverse price
    atr_bps: bar's high-low range in bps (volatility proxy)
    bar_return_bps: close-to-close bps (or current vs entry for first bar)
    """
    n = len(bid_open)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, empty

    if side == "long":
        entry = float(ask_open[0])
        if entry <= 0:
            return tuple(np.zeros(n, dtype=np.float32) for _ in range(5))  # type: ignore
        current_pnl_close = (bid_close - entry) / entry * 10000.0
        peak_high = (bid_high - entry) / entry * 10000.0
        trough_low = (bid_low - entry) / entry * 10000.0
    elif side == "short":
        entry = float(bid_open[0])
        if entry <= 0:
            return tuple(np.zeros(n, dtype=np.float32) for _ in range(5))  # type: ignore
        current_pnl_close = (entry - ask_close) / entry * 10000.0
        peak_high = (entry - ask_low) / entry * 10000.0
        trough_low = (entry - ask_high) / entry * 10000.0
    else:
        raise ValueError(f"unknown side: {side}")

    # Per-bar ATR proxy: bar's high-low range in bps relative to bar's close
    bar_close_for_atr = (ask_close + bid_close) / 2.0  # mid
    atr_bps = (ask_high - bid_low) / np.maximum(bar_close_for_atr, 1e-6) * 10000.0
    # Bar return: close-to-close bps (using mid)
    bar_return_bps = np.zeros(n, dtype=np.float64)
    if n > 1:
        bar_return_bps[1:] = (bar_close_for_atr[1:] - bar_close_for_atr[:-1]) / np.maximum(bar_close_for_atr[:-1], 1e-6) * 10000.0

    return (current_pnl_close.astype(np.float32),
            peak_high.astype(np.float32),
            trough_low.astype(np.float32),
            atr_bps.astype(np.float32),
            bar_return_bps.astype(np.float32))


def emit_per_bar_rows(
    candidate_row: dict[str, Any],
    bid_open: np.ndarray, bid_high: np.ndarray, bid_low: np.ndarray, bid_close: np.ndarray,
    ask_open: np.ndarray, ask_high: np.ndarray, ask_low: np.ndarray, ask_close: np.ndarray,
    bar_times_ns: np.ndarray,            # per-bar timestamp for feature lookup
    chunk0: pd.DataFrame | None,         # V3 ctx_cont/swing/regime per M5 bar
    canonical: pd.DataFrame | None,      # canonical V10 features per M5 bar
    *,
    side: str,
    bar_stride: int,
    max_bars_per_trade: int,
    k_horizons: list[int],
) -> list[dict[str, Any]]:
    """Emit one row per (sampled bar) within the hypothetical trade lifetime.

    Each row: candidate context + per-bar trade state + EXIT_NOW reward + HOLD-per-K rewards.
    """
    n_avail = len(bid_open)
    if n_avail < 2:
        return []
    cur_pnl, peak, trough, atr_bps, bar_ret = compute_per_bar_signals(
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close,
        side,
    )

    # Cumulative running peak/trough up to bar t (for trade-state)
    cum_peak = np.maximum.accumulate(peak)    # current MFE so far
    cum_trough = np.minimum.accumulate(trough)  # current MAE so far (negative)
    # bars_since_mfe_peak_v1: t - argmax(peak[0..t])
    arg_peak = np.zeros(n_avail, dtype=np.int32)
    running_max = -np.inf
    running_max_idx = 0
    for i in range(n_avail):
        if peak[i] >= running_max:
            running_max = float(peak[i])
            running_max_idx = i
        arg_peak[i] = running_max_idx

    # Carry entry_snapshot V3-features from candidate row (constant within trade)
    entry_snapshot = {
        "p_long_entry_v1": candidate_row.get("p_long"),
        "p_hat_entry_v1": candidate_row.get("p_hat"),
        "uncertainty_entry_v1": candidate_row.get("uncertainty_score"),
        "entropy_entry_v1": _compute_entropy_at_entry(
            candidate_row.get("p_long"),
            candidate_row.get("p_short"),
            candidate_row.get("p_flat"),
        ),
        "margin_entry_v1": candidate_row.get("margin"),
    }

    # Pre-compute trade-state derivatives over the entire path
    # (closed-form so we can index them by bar idx in the loop)
    n_full = len(cur_pnl)
    pnl_vel = np.zeros(n_full, dtype=np.float32)
    pnl_acc = np.zeros(n_full, dtype=np.float32)
    if n_full >= 2:
        pnl_vel[1:] = cur_pnl[1:] - cur_pnl[:-1]
    if n_full >= 3:
        pnl_acc[2:] = pnl_vel[2:] - pnl_vel[1:-1]
    # MFE-decay rate: change in cum_peak over last 4 bars (negative = peak decayed)
    mfe_decay = np.zeros(n_full, dtype=np.float32)
    if n_full > 4:
        mfe_decay[4:] = cum_peak[4:] - cum_peak[:-4]
    # Giveback ratio: how much of peak MFE is "given back" at this bar
    giveback = np.zeros(n_full, dtype=np.float32)
    pos_peak = np.maximum(cum_peak, 1e-6)
    giveback = (1.0 - cur_pnl / pos_peak).astype(np.float32)
    giveback = np.clip(giveback, -10.0, 10.0)
    giveback_acc = np.zeros(n_full, dtype=np.float32)
    if n_full >= 3:
        gv_vel = np.zeros(n_full, dtype=np.float32)
        gv_vel[1:] = giveback[1:] - giveback[:-1]
        giveback_acc[2:] = gv_vel[2:] - gv_vel[1:-1]
    # Rolling slope since entry: closed-form OLS slope over [0..t] in O(n).
    # slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
    # with x=0..t, n=t+1: Σx = t(t+1)/2, Σx² = t(t+1)(2t+1)/6
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

    rows: list[dict[str, Any]] = []
    last_t = min(n_avail, max_bars_per_trade + 1)  # +1 because t=0 is entry bar
    # Skip t=0 (entry bar — no exit decision yet); start from t=1
    for t in range(1, last_t, bar_stride):
        # Forward window for HOLD reward computation
        # At bar t, "remaining" = bars [t+1..]. Compute max-future-pnl per K.
        remaining_close = cur_pnl[t + 1: t + 1 + max(k_horizons)]
        remaining_peak = peak[t + 1: t + 1 + max(k_horizons)]
        n_remaining = len(remaining_peak)
        if n_remaining == 0:
            # No future bars — HOLD reward = current EXIT (degenerate)
            hold_rewards = {f"hold_max_pnl_K{K}_v1": float(cur_pnl[t]) for K in k_horizons}
            forced_terminal = True
        else:
            hold_rewards = {}
            for K in k_horizons:
                K_eff = min(K, n_remaining)
                # Max favourable price excursion in next K bars (using peak — what
                # an exit policy could realise if it timed perfectly).
                hold_rewards[f"hold_max_pnl_K{K}_v1"] = float(np.max(remaining_peak[:K_eff]))
            forced_terminal = (n_remaining < min(k_horizons))

        row = {
            # Identity
            "run_id": candidate_row.get("run_id"),
            "candidate_uid": candidate_row.get("candidate_uid"),
            "decision_ts_utc": candidate_row.get("decision_ts_utc"),
            "side_v1": side,
            "bar_idx_v1": int(t),
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
            # HOLD rewards per K
            **hold_rewards,
            # Quality
            "forced_terminal_v1": bool(forced_terminal),
            "forward_bars_remaining_v1": int(n_remaining),
        }
        # Carry candidate context (regime, session, p_long, etc.)
        for col in CANDIDATE_CONTEXT_COLS:
            if col in candidate_row:
                row[col] = candidate_row[col]
        # CANONICAL V10/V3 features at THIS bar
        bar_ts_ns = int(bar_times_ns[t]) if t < len(bar_times_ns) else 0
        chunk0_feats = fwd_pipe._chunk0_feature_at(chunk0, bar_ts_ns)
        row.update(chunk0_feats)
        row["_chunk0_features_complete_v1"] = bool(chunk0 is not None)
        canon_feats = fwd_pipe._canonical_feature_at(canonical, bar_ts_ns)
        row.update(canon_feats)
        row["_canonical_features_complete_v1"] = bool(canonical is not None)
        # Trade-state derivatives at this bar (V3 uses these per held bar)
        row["pnl_velocity_v1"] = float(pnl_vel[t]) if t < len(pnl_vel) else 0.0
        row["pnl_acceleration_v1"] = float(pnl_acc[t]) if t < len(pnl_acc) else 0.0
        row["mfe_decay_rate_v1"] = float(mfe_decay[t]) if t < len(mfe_decay) else 0.0
        row["giveback_ratio_v1"] = float(giveback[t]) if t < len(giveback) else 0.0
        row["giveback_acceleration_v1"] = float(giveback_acc[t]) if t < len(giveback_acc) else 0.0
        row["rolling_slope_since_entry_v1"] = float(rolling_slope[t]) if t < len(rolling_slope) else 0.0
        # Entry snapshot (constant within trade)
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


def process_week(
    week_dir: Path,
    m5_time_ns: np.ndarray,
    m5_arrays: dict[str, np.ndarray],
    canonical: pd.DataFrame | None,
    *,
    bar_stride: int,
    max_bars_per_trade: int,
    directional_threshold: float,
    directional_only: bool,
    max_gap_ns: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rid = week_dir.name
    cand_path = week_dir / f"shadow_meta_candidates_{rid}_MERGED.parquet"
    stats: dict[str, Any] = {
        "run_id": rid,
        "input_path": str(cand_path),
        "input_sha256": None,
        "input_rows": 0,
        "output_rows": 0,
        "skipped_no_decision_ts": 0,
        "skipped_no_directional_side": 0,
        "skipped_decision_past_tape_end": 0,
        "chunk0_features_loaded": False,
        "chunk0_features_count": 0,
        "process_status": "UNKNOWN",
    }
    if not cand_path.exists():
        stats["process_status"] = "INPUT_MISSING"
        return pd.DataFrame(), stats
    stats["input_sha256"] = _file_sha256(cand_path)
    cand = pd.read_parquet(cand_path)
    stats["input_rows"] = len(cand)
    if len(cand) == 0:
        stats["process_status"] = "INPUT_EMPTY"
        return pd.DataFrame(), stats

    # CANONICAL chunk_0 features (V10/V3 input per M5 bar)
    chunk0 = fwd_pipe._load_chunk0_features(week_dir)
    if chunk0 is not None:
        stats["chunk0_features_loaded"] = True
        stats["chunk0_features_count"] = len([c for c in chunk0.columns if c not in ("time", "_time_ns")])

    n_m5 = len(m5_time_ns)
    out_rows: list[dict[str, Any]] = []

    cand_dict = {col: cand[col].to_list() if col in cand.columns else [None] * len(cand)
                 for col in CANDIDATE_IDENTITY_COLS + CANDIDATE_CONTEXT_COLS}

    for row_i in range(len(cand)):
        decision_ts_str = cand_dict["decision_ts_utc"][row_i]
        ts_ns = fwd_pipe._safe_decision_ts_to_int64_ns(decision_ts_str)
        if ts_ns is None:
            stats["skipped_no_decision_ts"] += 1
            continue
        candidate_row = {col: cand_dict[col][row_i] for col in cand_dict}
        side = derive_side(candidate_row, threshold=directional_threshold)
        if side is None:
            if directional_only:
                stats["skipped_no_directional_side"] += 1
                continue
            # Not directional but still want to emit — pick higher of p_long/p_short anyway
            side = "long" if (candidate_row.get("p_long") or 0) >= (candidate_row.get("p_short") or 0) else "short"

        idx_start = int(np.searchsorted(m5_time_ns, ts_ns, side="right"))
        if idx_start >= n_m5:
            stats["skipped_decision_past_tape_end"] += 1
            continue
        s_t, e_t, _ = fwd_pipe.slice_forward_window(
            m5_time_ns, n_m5, idx_start, max_bars_per_trade + max(K_HORIZONS_EXIT) + 1, max_gap_ns,
        )
        bar_times_ns = m5_time_ns[s_t:e_t]
        rows = emit_per_bar_rows(
            candidate_row,
            m5_arrays["bid_open"][s_t:e_t],
            m5_arrays["bid_high"][s_t:e_t],
            m5_arrays["bid_low"][s_t:e_t],
            m5_arrays["bid_close"][s_t:e_t],
            m5_arrays["ask_open"][s_t:e_t],
            m5_arrays["ask_high"][s_t:e_t],
            m5_arrays["ask_low"][s_t:e_t],
            m5_arrays["ask_close"][s_t:e_t],
            bar_times_ns,
            chunk0,
            canonical,
            side=side,
            bar_stride=bar_stride,
            max_bars_per_trade=max_bars_per_trade,
            k_horizons=K_HORIZONS_EXIT,
        )
        out_rows.extend(rows)

    out_df = pd.DataFrame(out_rows)
    stats["output_rows"] = len(out_df)
    stats["process_status"] = "OK" if len(out_df) > 0 else "EMPTY_OUTPUT"
    return out_df, stats


def _is_truth_monfri_week(name: str) -> bool:
    return fwd_pipe._is_truth_monfri_week(name)


def write_artifacts(
    out_root: Path | None = None,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    m5_tape_root: Path = DEFAULT_M5_TAPE_ROOT,
    canonical_features_path: Path | None = None,
    only_weeks: list[str] | None = None,
    skip_existing: bool = True,
    bar_stride: int = DEFAULT_BAR_STRIDE,
    max_bars_per_trade: int = DEFAULT_MAX_BARS_PER_TRADE,
    directional_threshold: float = DEFAULT_DIRECTIONAL_THRESHOLD,
    directional_only: bool = False,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (reports_root / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)
    per_week_dir = artifact_root / "per_week"
    per_week_dir.mkdir(exist_ok=True)

    print(f"[{ACTION}] loading M5 tape from {m5_tape_root}", flush=True)
    m5 = fwd_pipe.load_m5_tape(m5_tape_root)
    print(f"[{ACTION}] M5 bars loaded: {len(m5):,}  range={m5['time'].min()} -> {m5['time'].max()}", flush=True)

    canonical_path = canonical_features_path or DEFAULT_CANONICAL_FEATURES_PATH
    print(f"[{ACTION}] loading canonical features from {canonical_path}", flush=True)
    canonical = fwd_pipe._load_canonical_features(canonical_path)
    if canonical is not None:
        n_canon_feats = len([c for c in canonical.columns if c != "_time_ns"])
        print(f"[{ACTION}] canonical features loaded: {len(canonical):,} rows × {n_canon_feats} features", flush=True)

    m5_time_ns = m5["time"].astype("int64").to_numpy()
    m5_arrays = {
        col: m5[col].astype(np.float64).to_numpy()
        for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close")
    }
    max_gap_ns = fwd_pipe.MAX_INTRA_GAP_MINUTES * 60 * 1_000_000_000

    week_dirs = sorted(
        d for d in reports_root.iterdir()
        if d.is_dir() and _is_truth_monfri_week(d.name)
    )
    if only_weeks is not None:
        only_set = set(only_weeks)
        week_dirs = [d for d in week_dirs if d.name in only_set]
    print(f"[{ACTION}] weeks to process: {len(week_dirs)}", flush=True)
    print(f"[{ACTION}] params: bar_stride={bar_stride} max_bars_per_trade={max_bars_per_trade} "
          f"directional_threshold={directional_threshold} directional_only={directional_only}", flush=True)

    per_week_stats: list[dict[str, Any]] = []
    total_rows = 0
    for w_idx, week_dir in enumerate(week_dirs, start=1):
        out_path = per_week_dir / f"exit_per_bar_{week_dir.name}.parquet"
        stats_path = per_week_dir / f"exit_per_bar_{week_dir.name}_stats.json"
        if skip_existing and out_path.exists() and stats_path.exists():
            try:
                cached = json.loads(stats_path.read_text())
                per_week_stats.append(cached)
                total_rows += int(cached.get("output_rows", 0))
                continue
            except (json.JSONDecodeError, OSError):
                pass

        df, stats = process_week(
            week_dir, m5_time_ns, m5_arrays, canonical,
            bar_stride=bar_stride,
            max_bars_per_trade=max_bars_per_trade,
            directional_threshold=directional_threshold,
            directional_only=directional_only,
            max_gap_ns=max_gap_ns,
        )
        per_week_stats.append(stats)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            total_rows += len(df)
        stats_path.write_text(json.dumps(_jsonable(stats), indent=2, sort_keys=True))
        if w_idx % 10 == 0 or w_idx == len(week_dirs):
            print(f"[{ACTION}] [{w_idx}/{len(week_dirs)}] {week_dir.name}: input={stats['input_rows']} -> output={stats['output_rows']} status={stats['process_status']}", flush=True)

    summary = {
        "layer_name": "EXIT_IQL_PER_BAR_DATASET_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "m5_tape_root_v1": str(m5_tape_root),
        "m5_bars_loaded_v1": int(len(m5)),
        "k_horizons_exit_v1": K_HORIZONS_EXIT,
        "bar_stride_v1": bar_stride,
        "max_bars_per_trade_v1": max_bars_per_trade,
        "directional_threshold_v1": directional_threshold,
        "directional_only_v1": directional_only,
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
    parser.add_argument("--m5-root", type=str, default=str(DEFAULT_M5_TAPE_ROOT))
    parser.add_argument("--canonical-features", type=str, default=str(DEFAULT_CANONICAL_FEATURES_PATH))
    parser.add_argument("--week", action="append", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--bar-stride", type=int, default=DEFAULT_BAR_STRIDE)
    parser.add_argument("--max-bars-per-trade", type=int, default=DEFAULT_MAX_BARS_PER_TRADE)
    parser.add_argument("--directional-threshold", type=float, default=DEFAULT_DIRECTIONAL_THRESHOLD)
    parser.add_argument("--directional-only", action="store_true")
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(
        out_root=out_root,
        reports_root=Path(args.reports_root).expanduser().resolve(),
        m5_tape_root=Path(args.m5_root).expanduser().resolve(),
        canonical_features_path=Path(args.canonical_features).expanduser().resolve(),
        only_weeks=args.week,
        skip_existing=not args.no_skip_existing,
        bar_stride=args.bar_stride,
        max_bars_per_trade=args.max_bars_per_trade,
        directional_threshold=args.directional_threshold,
        directional_only=args.directional_only,
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
