#!/usr/bin/env python3
"""Build candidate-level forward-outcome dataset across all 282 weekly XAUUSD replays.

Mission
-------
For every XGB+V10 candidate (decision moment) across 2020-2026, compute what
the market actually did over the following K M5-bars — for BOTH long and short
directions, and for both immediate-entry and delayed-entry (WAIT) scenarios.

The output is the training substrate for entry-IQL (the RL "smart head" that
supervises and overrides the entry transformer). Entry-IQL learns: given
candidate state s and action a in {SKIP, TAKE_LONG_NOW, TAKE_SHORT_NOW,
WAIT_LONG, WAIT_SHORT}, what reward (MFE-based) is observable from the
forward path? Because forward bars are known offline, every candidate has
fully-observable counterfactual rewards for every action.

Scope
-----
Universe = ALL 844k+ candidates from 282 weekly TRUTH_MONFRI_WEEK runs, NOT
filtered to V10-approved trades. The IQL must be able to learn that V10's
flat decisions were sometimes wrong.

Forward windows
---------------
K_HORIZONS = [12, 24, 48, 96, 144, 192] M5-bars ≈ [1h, 2h, 4h, 8h, 12h, 16h].
Both sniper/scalp and swing horizons captured simultaneously.

Per (action, side, K), we compute:
  mfe_bps_at_K           — peak favorable excursion (spread-aware)
  mae_bps_at_K           — peak adverse excursion (magnitude, positive)
  mae_before_mfe_bps_at_K — worst drawdown BEFORE mfe-peak (realiserbarhet)
  time_to_mfe_at_K       — bar-idx where MFE peaked
  terminal_pnl_at_K      — pnl if held flat to bar K
  forward_bars_used_at_K — actual bars used (may be < K if truncated)

Spread / fill conventions
-------------------------
LONG: enter on next bar's ask_open; during trade, peak unrealized = bid_high,
trough unrealized = bid_low (we'd close on bid).
SHORT: enter on next bar's bid_open; during trade, peak unrealized = -ask_low,
trough unrealized = -ask_high.

Truncation rules
----------------
Forward window TRUNCATES at the first M5 gap > MAX_INTRA_GAP_MINUTES (60),
which catches Friday-flat (weekend) and overnight Asian-open gaps. Each row
gets a `forward_window_status_v1` flag describing what happened.

Robustness
----------
- Per-week incremental output: each week → its own parquet. Pipeline can be
  resumed; weeks with up-to-date input hashes are skipped.
- Manifest tracks per-week input SHA-256 + row counts + status.
- Deterministic: sorted iteration, no randomness.
- Schema-versioned: all output columns end in _v1.
- M5 tape loaded once into memory (48MB, all 2020-2026).

This is RESEARCH-ONLY. Output drives entry-IQL training but is not promoted to
runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
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


ACTION = "BUILD_CANDIDATE_FORWARD_OUTCOME_DATASET_V1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_M5_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"
)
DEFAULT_OUT_ROOT = DEFAULT_REPORTS_ROOT / "ALL_TRADE_REVIEW_LEDGER_20260411_CANDIDATE_FORWARD_OUTCOME_V1"
DEFAULT_CANONICAL_FEATURES_PATH = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V3/canonical_features_v3.parquet"
)
# Suffix kept as `_canon_v1` for downstream-builder backward compatibility
# (entry-IQL/exit-IQL state column lists hardcode this suffix). The V3 parquet
# is canonical_v2 minus 11 pruned + 5 new (cyclic + smc_premium_state); same
# column names for shared features, so the suffix is just a string convention.
CANONICAL_FEATURES_SUFFIX = "_canon_v1"

K_HORIZONS = [12, 24, 48, 96, 144, 192]
MAX_K = max(K_HORIZONS)
WAIT_BARS = 6  # WAIT action delays entry by this many M5-bars (=30min)
MAX_INTRA_GAP_MINUTES = 180  # truncate window at first gap > this. XAUUSD has
                              # daily ~1h closes (60min gaps) — must NOT truncate
                              # those. Weekends are ≥48h so 180min cleanly splits
                              # daily-close (allowed) vs weekend/holiday (truncate).

# Candidate feature columns kept for downstream IQL training.
# Source: shadow_meta_candidates_*_MERGED.parquet schema.
CANDIDATE_FEATURE_COLS = [
    "side",            # may be NA for non-accepted; IQL must handle
    "session",
    "weekday_utc",
    "hour_utc",
    "atr_bps",
    "entry_spread_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
    "vol_regime",
    "trend_regime",
    "decision_reason",  # flat_dominant / flat_veto / pre_quality / ...
    "accepted",
    "policy_lane",
    # 2026-Q2 V10 v2 BIDIR additions (entry-IQL state expects these)
    "bad_path_prob",
    "direction_logit_long",
    "direction_logit_short",
    "direction_logit_flat",
]

# CANONICAL feature columns from each week's replay/chunk_0/chunk_0_data.parquet.
# This is what V10/V3 transformers actually consume in the live runtime.
# Joining these by time gives the IQL the SAME feature vector V10 sees at
# entry-decision time, eliminating the V10-output-only feature bottleneck.
# 55 features = 28 BASE28 _v1_* + 11 ctx_cont + 6 ctx_cat + 5 swing + 5 session.
CHUNK0_FEATURE_COLS = [
    # 28 BASE28 _v1_* features
    "_v1_atr14", "_v1_atr_regime_id", "_v1_atr_z_10_100",
    "_v1_bb_bandwidth_delta_10", "_v1_bb_squeeze_20_2",
    "_v1_body_share_1", "_v1_body_tr",
    "_v1_close_ema_slope_3", "_v1_clv", "_v1_comp3_ratio",
    "_v1_cost_bps_dyn", "_v1_cost_bps_est",
    "_v1_ema_diff",
    "_v1_int_clv_atr", "_v1_int_ema_us", "_v1_int_r5_atr",
    "_v1_int_range_us", "_v1_int_slope_h1_us", "_v1_int_slope_h4_atr",
    "_v1_int_vwap_h1",
    "_v1_is_EU", "_v1_is_US",
    "_v1_kama_slope_30", "_v1_kurt_r", "_v1_lower_tr",
    "_v1_pk_sigma20", "_v1_r1", "_v1_r12",
    # 11 ctx_cont (multi-TF + microstructure)
    "atr_bps", "spread_bps",
    "D1_dist_from_ema200_atr", "H1_range_compression_ratio",
    "D1_atr_percentile_252", "M15_range_compression_ratio",
    "micro_momentum_3", "micro_momentum_5", "micro_acceleration",
    "wick_ratio", "distance_ema_fast",
    # 5 swing
    "dist_last_swing_high_atr", "dist_last_swing_low_atr",
    "bars_since_swing_high", "bars_since_swing_low",
    "retracement_from_last_impulse",
    # 6 ctx_cat (regime IDs)
    "trend_regime_id", "vol_regime_id",
    "atr_bucket", "spread_bucket", "H4_trend_sign_cat",
    "session_id",
    # 5 session
    "is_ASIA",
    "minutes_since_session_open", "minutes_to_next_session_boundary",
    "session_change_flag", "session_tradable",
]
# Some chunk_0 features overlap by name with CANDIDATE_FEATURE_COLS (atr_bps,
# entry_spread_bps via spread_bps, session_id) — we suffix them on join to
# disambiguate.
CHUNK0_SUFFIX = "_chunk0_v1"

CANDIDATE_IDENTITY_COLS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_ts_utc",
]

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report


# ---------------------------------------------------------------------------
# M5 tape
# ---------------------------------------------------------------------------


def load_m5_tape(m5_root: Path = DEFAULT_M5_TAPE_ROOT) -> pd.DataFrame:
    """Load all year=YYYY/*.parquet files into a single time-sorted frame."""
    parts: list[pd.DataFrame] = []
    for year_dir in sorted(m5_root.glob("year=*")):
        for parquet in sorted(year_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(parquet))
    if not parts:
        raise RuntimeError(f"[{ACTION}] no M5 parquets found under {m5_root}")
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    # Validate required columns
    required = ["time", "bid_open", "bid_high", "bid_low", "bid_close",
                "ask_open", "ask_high", "ask_low", "ask_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{ACTION}] M5 tape missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Forward-window slicing with gap-truncation
# ---------------------------------------------------------------------------


def slice_forward_window(
    m5_time: np.ndarray,  # int64 nanosec, monotonic
    m5_len: int,
    idx_start: int,
    K: int,
    max_gap_ns: int,
) -> tuple[int, int, str]:
    """Return (slice_start, slice_end_exclusive, status).

    status: "FULL", "TRUNCATED_GAP_FRIDAY", "TRUNCATED_GAP_OTHER",
            "TRUNCATED_END_OF_TAPE", "EMPTY".
    """
    if idx_start >= m5_len:
        return idx_start, idx_start, "EMPTY"
    end = min(idx_start + K, m5_len)
    if end <= idx_start + 1:
        return idx_start, end, "TRUNCATED_END_OF_TAPE" if end < idx_start + K else "FULL"
    # Detect first gap > max_gap_ns within [idx_start+1 .. end-1]
    gaps = m5_time[idx_start + 1 : end] - m5_time[idx_start : end - 1]
    big = np.where(gaps > max_gap_ns)[0]
    if len(big) > 0:
        # First big gap is between bar (idx_start + big[0]) and (idx_start + big[0] + 1)
        # We keep up to and including bar (idx_start + big[0])
        gap_at_idx = idx_start + int(big[0])
        new_end = gap_at_idx + 1  # exclusive
        # Classify gap: Friday flat = the bar BEFORE the gap is on Friday >= 20:00 UTC
        bar_before_gap_ts = pd.Timestamp(m5_time[gap_at_idx], tz="UTC")
        if bar_before_gap_ts.weekday() == 4 and bar_before_gap_ts.hour >= 19:
            status = "TRUNCATED_GAP_FRIDAY"
        else:
            status = "TRUNCATED_GAP_OTHER"
        return idx_start, new_end, status
    if end < idx_start + K:
        return idx_start, end, "TRUNCATED_END_OF_TAPE"
    return idx_start, end, "FULL"


# ---------------------------------------------------------------------------
# Forward outcomes (per side, per K)
# ---------------------------------------------------------------------------


def compute_side_metrics(
    bid_open: np.ndarray,
    bid_high: np.ndarray,
    bid_low: np.ndarray,
    bid_close: np.ndarray,
    ask_open: np.ndarray,
    ask_high: np.ndarray,
    ask_low: np.ndarray,
    ask_close: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (peak_bps_per_bar, trough_bps_per_bar, terminal_bps_per_bar).

    All bps are computed against the entry fill (long: ask_open[0]; short:
    bid_open[0]). LONG peak/trough/terminal use bid (close-side); SHORT use
    ask (close-side). All values signed: peak is best favorable excursion at
    each bar (cumulative max of running highs); trough is worst adverse
    (cumulative min of running lows).

    NOTE: peak/trough are PER-BAR (not cumulative). Caller computes cumulative
    max/min over [0..K] to derive mfe_at_K / mae_at_K.
    """
    n = len(bid_open)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty

    if side == "long":
        entry = float(ask_open[0])
        if entry <= 0:
            return (np.zeros(n, dtype=np.float32),
                    np.zeros(n, dtype=np.float32),
                    np.zeros(n, dtype=np.float32))
        # During trade, "best price you could close at" is bid_high; "worst" is bid_low
        peak = (bid_high - entry) / entry * 10000.0
        trough = (bid_low - entry) / entry * 10000.0
        terminal = (bid_close - entry) / entry * 10000.0
    elif side == "short":
        entry = float(bid_open[0])
        if entry <= 0:
            return (np.zeros(n, dtype=np.float32),
                    np.zeros(n, dtype=np.float32),
                    np.zeros(n, dtype=np.float32))
        # SHORT: profit when ask drops. peak = entry - ask_low (best); trough = entry - ask_high (worst)
        peak = (entry - ask_low) / entry * 10000.0
        trough = (entry - ask_high) / entry * 10000.0
        terminal = (entry - ask_close) / entry * 10000.0
    else:
        raise ValueError(f"unknown side: {side}")

    return peak.astype(np.float32), trough.astype(np.float32), terminal.astype(np.float32)


def compute_outcomes_for_window(
    bid_open: np.ndarray, bid_high: np.ndarray, bid_low: np.ndarray, bid_close: np.ndarray,
    ask_open: np.ndarray, ask_high: np.ndarray, ask_low: np.ndarray, ask_close: np.ndarray,
    side: str,
) -> dict[str, float]:
    """For a given forward-bar window and side, return per-K metrics dict."""
    out: dict[str, float] = {}
    n_avail = len(bid_open)
    if n_avail == 0:
        for K in K_HORIZONS:
            out[f"mfe_bps_at_K{K}"] = float("nan")
            out[f"mae_bps_at_K{K}"] = float("nan")
            out[f"mae_before_mfe_bps_at_K{K}"] = float("nan")
            out[f"time_to_mfe_at_K{K}"] = -1
            out[f"terminal_pnl_at_K{K}"] = float("nan")
            out[f"forward_bars_used_at_K{K}"] = 0
        return out

    peak, trough, terminal = compute_side_metrics(
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close,
        side,
    )

    # Cumulative max over peak; cumulative min over trough; argmax of cum_max.
    cum_max_peak = np.maximum.accumulate(peak)
    cum_min_trough = np.minimum.accumulate(trough)

    for K in K_HORIZONS:
        K_eff = min(K, n_avail)
        if K_eff == 0:
            out[f"mfe_bps_at_K{K}"] = float("nan")
            out[f"mae_bps_at_K{K}"] = float("nan")
            out[f"mae_before_mfe_bps_at_K{K}"] = float("nan")
            out[f"time_to_mfe_at_K{K}"] = -1
            out[f"terminal_pnl_at_K{K}"] = float("nan")
            out[f"forward_bars_used_at_K{K}"] = 0
            continue

        peak_K = peak[:K_eff]
        trough_K = trough[:K_eff]
        argmax_idx = int(np.argmax(peak_K))
        mfe = float(peak_K[argmax_idx])
        mae = float(-np.min(trough_K))  # magnitude positive
        # MAE before MFE peak: trough indices [0..argmax_idx]
        mae_before = float(-np.min(trough_K[: argmax_idx + 1]))
        if mae_before < 0:
            mae_before = 0.0  # zero-clamp; means "no drawdown before peak"
        terminal_K = float(terminal[K_eff - 1])

        out[f"mfe_bps_at_K{K}"] = mfe
        out[f"mae_bps_at_K{K}"] = mae if mae > 0 else 0.0
        out[f"mae_before_mfe_bps_at_K{K}"] = mae_before
        out[f"time_to_mfe_at_K{K}"] = argmax_idx
        out[f"terminal_pnl_at_K{K}"] = terminal_K
        out[f"forward_bars_used_at_K{K}"] = K_eff
    return out


# ---------------------------------------------------------------------------
# Per-week processing
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_decision_ts_to_int64_ns(ts_str: Any) -> int | None:
    if ts_str is None:
        return None
    try:
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return int(ts.value)  # int64 ns since epoch
    except (ValueError, TypeError):
        return None


def _load_canonical_features(canonical_path: Path) -> pd.DataFrame | None:
    """Load canonical_features_v1.parquet (84 V10-aligned features over 2020-2026).

    Returns DataFrame with `_time_ns` int64 index column for fast searchsorted.
    Drops the 13 OHLCV/bid/ask columns (those come from m5 tape directly).
    """
    if not canonical_path.exists():
        return None
    df = pd.read_parquet(canonical_path)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["_time_ns"] = df["time"].astype("int64")
    drop_cols = ["open", "high", "low", "close", "volume",
                 "bid_open", "bid_high", "bid_low", "bid_close",
                 "ask_open", "ask_high", "ask_low", "ask_close",
                 "time"]
    feat_cols = [c for c in df.columns if c not in drop_cols and c != "_time_ns"]
    return df[["_time_ns"] + feat_cols].sort_values("_time_ns").reset_index(drop=True)


def _canonical_feature_at(canonical: pd.DataFrame | None, ts_ns: int) -> dict[str, Any]:
    if canonical is None or len(canonical) == 0:
        feat_cols = [c for c in (canonical.columns if canonical is not None else []) if c != "_time_ns"]
        return {f"{c}{CANONICAL_FEATURES_SUFFIX}": np.nan for c in feat_cols}
    arr = canonical["_time_ns"].to_numpy()
    idx = int(np.searchsorted(arr, ts_ns, side="right")) - 1
    if idx < 0:
        feat_cols = [c for c in canonical.columns if c != "_time_ns"]
        return {f"{c}{CANONICAL_FEATURES_SUFFIX}": np.nan for c in feat_cols}
    row = canonical.iloc[idx]
    return {f"{c}{CANONICAL_FEATURES_SUFFIX}": row[c] for c in canonical.columns if c != "_time_ns"}


def _compute_entropy_bps(p_long: float, p_short: float, p_flat: float) -> float:
    """Compute Shannon entropy of (p_long, p_short, p_flat) in nats."""
    s = 0.0
    for p in (p_long, p_short, p_flat):
        if p is None or not np.isfinite(p) or p <= 1e-12:
            continue
        s -= p * float(np.log(p))
    return s


def _load_chunk0_features(week_dir: Path) -> pd.DataFrame | None:
    """Load this week's replay/chunk_0/chunk_0_data.parquet (V10/V3 input).

    Returns DataFrame indexed by `time` (UTC ns int64) with only the 55 V10/V3
    feature columns. Returns None if the file doesn't exist or required cols
    are missing — the join then falls back to candidate-only state.
    """
    chunk_path = week_dir / "replay" / "chunk_0" / "chunk_0_data.parquet"
    if not chunk_path.exists():
        return None
    try:
        df = pd.read_parquet(chunk_path)
    except (OSError, ValueError):
        return None
    if "time" not in df.columns:
        return None
    keep = [c for c in CHUNK0_FEATURE_COLS if c in df.columns]
    if not keep:
        return None
    out = df[["time"] + keep].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).copy()
    out["_time_ns"] = out["time"].astype("int64")
    out = out.sort_values("_time_ns", kind="mergesort").reset_index(drop=True)
    return out


def _chunk0_feature_at(chunk0: pd.DataFrame | None, ts_ns: int) -> dict[str, Any]:
    """Return dict of suffixed chunk_0 features at the bar containing `ts_ns`.

    Lookup uses searchsorted-right to find the bar the candidate falls IN
    (i.e., the most recent M5 bar at or before the decision time). All output
    keys are suffixed with CHUNK0_SUFFIX to avoid collision with candidate cols.
    """
    if chunk0 is None or len(chunk0) == 0:
        return {f"{c}{CHUNK0_SUFFIX}": np.nan for c in CHUNK0_FEATURE_COLS}
    arr = chunk0["_time_ns"].to_numpy()
    # rightmost bar with time <= ts_ns
    idx = int(np.searchsorted(arr, ts_ns, side="right")) - 1
    if idx < 0:
        return {f"{c}{CHUNK0_SUFFIX}": np.nan for c in CHUNK0_FEATURE_COLS}
    row = chunk0.iloc[idx]
    out: dict[str, Any] = {}
    for c in CHUNK0_FEATURE_COLS:
        out[f"{c}{CHUNK0_SUFFIX}"] = row.get(c, np.nan)
    return out


def process_week(
    week_dir: Path,
    m5: pd.DataFrame,
    m5_time_ns: np.ndarray,
    m5_arrays: dict[str, np.ndarray],
    canonical: pd.DataFrame | None,
    *,
    max_gap_ns: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process one TRUTH_MONFRI_WEEK_* directory. Return (rows_df, stats_dict)."""
    rid = week_dir.name
    cand_path = week_dir / f"shadow_meta_candidates_{rid}_MERGED.parquet"
    stats: dict[str, Any] = {
        "run_id": rid,
        "input_path": str(cand_path),
        "input_sha256": None,
        "input_rows": 0,
        "output_rows": 0,
        "skipped_no_decision_ts": 0,
        "skipped_decision_past_tape_end": 0,
        "skipped_no_chunk0_features": 0,
        "chunk0_features_loaded": False,
        "chunk0_features_count": 0,
        "status_counts_take_now": {},
        "status_counts_wait": {},
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

    # CANONICAL: load this week's chunk_0_data.parquet for V10/V3 features.
    chunk0 = _load_chunk0_features(week_dir)
    if chunk0 is not None:
        stats["chunk0_features_loaded"] = True
        stats["chunk0_features_count"] = len(chunk0.columns) - 2  # exclude time + _time_ns

    n_m5 = len(m5_time_ns)
    out_rows: list[dict[str, Any]] = []
    status_take = {}
    status_wait = {}

    # Pre-extract numpy arrays for speed
    cand_dict = {col: cand[col].to_list() if col in cand.columns else [None] * len(cand)
                 for col in CANDIDATE_IDENTITY_COLS + CANDIDATE_FEATURE_COLS}

    for row_i in range(len(cand)):
        decision_ts_str = cand_dict["decision_ts_utc"][row_i]
        ts_ns = _safe_decision_ts_to_int64_ns(decision_ts_str)
        if ts_ns is None:
            stats["skipped_no_decision_ts"] += 1
            continue

        # Find first M5 bar with time STRICTLY GREATER than decision_ts (next bar boundary)
        idx_start = int(np.searchsorted(m5_time_ns, ts_ns, side="right"))
        if idx_start >= n_m5:
            stats["skipped_decision_past_tape_end"] += 1
            continue

        # Take_now window
        s_t, e_t, status_t = slice_forward_window(m5_time_ns, n_m5, idx_start, MAX_K, max_gap_ns)
        # Wait window
        wait_idx_start = idx_start + WAIT_BARS
        s_w, e_w, status_w = slice_forward_window(m5_time_ns, n_m5, wait_idx_start, MAX_K, max_gap_ns)

        status_take[status_t] = status_take.get(status_t, 0) + 1
        status_wait[status_w] = status_wait.get(status_w, 0) + 1

        row_out: dict[str, Any] = {}
        # Identity + candidate features (carry forward)
        for col in CANDIDATE_IDENTITY_COLS + CANDIDATE_FEATURE_COLS:
            row_out[col] = cand_dict[col][row_i]

        # CANONICAL chunk_0 features at decision_ts (V10/V3 feature parity)
        chunk0_feats = _chunk0_feature_at(chunk0, ts_ns)
        if chunk0 is None:
            stats["skipped_no_chunk0_features"] += 1
        row_out.update(chunk0_feats)
        row_out["_chunk0_features_complete_v1"] = bool(chunk0 is not None)

        # Canonical features (basic_v1 + H1 + H4 + m5_phase + high-level)
        canon_feats = _canonical_feature_at(canonical, ts_ns)
        row_out.update(canon_feats)
        row_out["_canonical_features_complete_v1"] = bool(canonical is not None)

        # Derived: entropy of (p_long, p_short, p_flat) — V10/V3 use this
        try:
            row_out["entropy_v1"] = _compute_entropy_bps(
                float(row_out.get("p_long") or 0.0),
                float(row_out.get("p_short") or 0.0),
                float(row_out.get("p_flat") or 0.0),
            )
        except (TypeError, ValueError):
            row_out["entropy_v1"] = np.nan

        # Take_now: long + short
        for side in ("long", "short"):
            metrics = compute_outcomes_for_window(
                m5_arrays["bid_open"][s_t:e_t],
                m5_arrays["bid_high"][s_t:e_t],
                m5_arrays["bid_low"][s_t:e_t],
                m5_arrays["bid_close"][s_t:e_t],
                m5_arrays["ask_open"][s_t:e_t],
                m5_arrays["ask_high"][s_t:e_t],
                m5_arrays["ask_low"][s_t:e_t],
                m5_arrays["ask_close"][s_t:e_t],
                side,
            )
            for k, v in metrics.items():
                row_out[f"take_now_{side}_{k}_v1"] = v
        row_out["take_now_window_status_v1"] = status_t
        row_out["take_now_idx_start_v1"] = idx_start

        # Wait: long + short
        for side in ("long", "short"):
            metrics = compute_outcomes_for_window(
                m5_arrays["bid_open"][s_w:e_w],
                m5_arrays["bid_high"][s_w:e_w],
                m5_arrays["bid_low"][s_w:e_w],
                m5_arrays["bid_close"][s_w:e_w],
                m5_arrays["ask_open"][s_w:e_w],
                m5_arrays["ask_high"][s_w:e_w],
                m5_arrays["ask_low"][s_w:e_w],
                m5_arrays["ask_close"][s_w:e_w],
                side,
            )
            for k, v in metrics.items():
                row_out[f"wait_{side}_{k}_v1"] = v
        row_out["wait_window_status_v1"] = status_w
        row_out["wait_idx_start_v1"] = wait_idx_start
        row_out["wait_bars_offset_v1"] = WAIT_BARS

        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    stats["output_rows"] = len(out_df)
    stats["status_counts_take_now"] = status_take
    stats["status_counts_wait"] = status_wait
    stats["process_status"] = "OK" if len(out_df) > 0 else "EMPTY_OUTPUT"
    return out_df, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _is_truth_monfri_week(name: str) -> bool:
    if not name.startswith("TRUTH_MONFRI_WEEK_"):
        return False
    suffix = name[len("TRUTH_MONFRI_WEEK_"):]
    parts = suffix.split("_")
    if len(parts) != 2:
        return False
    return parts[0].isdigit() and parts[1].isdigit() and len(parts[0]) == 8 and len(parts[1]) == 8


def write_artifacts(
    out_root: Path | None = None,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    m5_tape_root: Path = DEFAULT_M5_TAPE_ROOT,
    canonical_features_path: Path | None = None,
    only_weeks: list[str] | None = None,
    skip_existing: bool = True,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (reports_root / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)
    per_week_dir = artifact_root / "per_week"
    per_week_dir.mkdir(exist_ok=True)

    print(f"[{ACTION}] loading M5 tape from {m5_tape_root}", flush=True)
    m5 = load_m5_tape(m5_tape_root)
    print(f"[{ACTION}] M5 bars loaded: {len(m5):,}  range={m5['time'].min()} -> {m5['time'].max()}", flush=True)

    canonical_path = canonical_features_path or DEFAULT_CANONICAL_FEATURES_PATH
    print(f"[{ACTION}] loading canonical features from {canonical_path}", flush=True)
    canonical = _load_canonical_features(canonical_path)
    if canonical is not None:
        n_canon_feats = len([c for c in canonical.columns if c != "_time_ns"])
        print(f"[{ACTION}] canonical features loaded: {len(canonical):,} rows × {n_canon_feats} features", flush=True)
    else:
        print(f"[{ACTION}] WARNING: canonical features not loaded — IQL state will be missing _v1_* / H1 / H4 / m5_phase / high-level", flush=True)

    m5_time_ns = m5["time"].astype("int64").to_numpy()
    m5_arrays = {
        col: m5[col].astype(np.float64).to_numpy()
        for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close")
    }
    max_gap_ns = MAX_INTRA_GAP_MINUTES * 60 * 1_000_000_000

    week_dirs = sorted(
        d for d in reports_root.iterdir()
        if d.is_dir() and _is_truth_monfri_week(d.name)
    )
    if only_weeks is not None:
        only_set = set(only_weeks)
        week_dirs = [d for d in week_dirs if d.name in only_set]
    print(f"[{ACTION}] weeks to process: {len(week_dirs)}", flush=True)

    per_week_stats: list[dict[str, Any]] = []
    total_rows = 0
    for w_idx, week_dir in enumerate(week_dirs, start=1):
        out_path = per_week_dir / f"forward_outcomes_{week_dir.name}.parquet"
        stats_path = per_week_dir / f"forward_outcomes_{week_dir.name}_stats.json"
        if skip_existing and out_path.exists() and stats_path.exists():
            try:
                cached_stats = json.loads(stats_path.read_text())
                per_week_stats.append(cached_stats)
                total_rows += int(cached_stats.get("output_rows", 0))
                if w_idx % 25 == 0 or w_idx == len(week_dirs):
                    print(f"[{ACTION}] [{w_idx}/{len(week_dirs)}] cached: {week_dir.name} rows={cached_stats.get('output_rows')}", flush=True)
                continue
            except (json.JSONDecodeError, OSError):
                pass  # fallthrough to recompute

        df, stats = process_week(week_dir, m5, m5_time_ns, m5_arrays, canonical, max_gap_ns=max_gap_ns)
        per_week_stats.append(stats)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            total_rows += len(df)
        stats_path.write_text(json.dumps(_jsonable(stats), indent=2, sort_keys=True))
        if w_idx % 10 == 0 or w_idx == len(week_dirs):
            print(f"[{ACTION}] [{w_idx}/{len(week_dirs)}] {week_dir.name}: input={stats['input_rows']} -> output={stats['output_rows']} status={stats['process_status']}", flush=True)

    # Aggregate manifest
    summary = {
        "layer_name": "CANDIDATE_FORWARD_OUTCOME_DATASET_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "m5_tape_root_v1": str(m5_tape_root),
        "m5_bars_loaded_v1": int(len(m5)),
        "m5_date_range_v1": [str(m5["time"].min()), str(m5["time"].max())],
        "k_horizons_v1": K_HORIZONS,
        "wait_bars_v1": WAIT_BARS,
        "max_intra_gap_minutes_v1": MAX_INTRA_GAP_MINUTES,
        "candidate_feature_cols_v1": CANDIDATE_FEATURE_COLS,
        "candidate_identity_cols_v1": CANDIDATE_IDENTITY_COLS,
        "weeks_processed_v1": len(per_week_stats),
        "weeks_ok_v1": sum(1 for s in per_week_stats if s.get("process_status") == "OK"),
        "weeks_input_missing_v1": sum(1 for s in per_week_stats if s.get("process_status") == "INPUT_MISSING"),
        "weeks_input_empty_v1": sum(1 for s in per_week_stats if s.get("process_status") == "INPUT_EMPTY"),
        "total_input_rows_v1": int(sum(int(s.get("input_rows", 0)) for s in per_week_stats)),
        "total_output_rows_v1": int(total_rows),
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
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
    parser.add_argument("--week", action="append", default=None,
                        help="Process only these specific TRUTH_MONFRI_WEEK_* names. Repeatable.")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Force reprocess even if per-week parquet already exists.")
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
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
