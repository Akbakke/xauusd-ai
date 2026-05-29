"""V12 Phase 2: Build EXIT_IQL_PER_BAR_DATASET_V12 from forward-outcome candidates.

Pipeline
--------
For each TAKE_LONG_NOW / TAKE_SHORT_NOW candidate (filtered via Phase 1
Entry-IQL decisions), perform M1 forward rollout and emit one row per held
M1 bar. Output is a drop-in replacement for V8AUG with V12-specific changes.

Reuses primitives from `materialize_build_exit_iql_per_bar_dataset_v2_m1.py`:
  - `compute_per_bar_signals` (per-bar PnL/peak/trough/atr_bps/bar_return)
  - `_compute_m1_micro_features` (5 m1 micro features)
  - `slice_forward_window` (M1 walk with weekend-gap policy)
  - `compute_m5_phase_index`
  - candidate context column lists from fwd_pipe / v1_pipe

V12-specific changes vs V8AUG:
  - Source: forward-outcome candidate parquet (50,824 cands) ⋈ Entry-IQL filter
  - K_HORIZONS:           [12, 60, 240, 480, 1440]   (M1 minutes, 12min→24h)
  - max_bars_per_trade:   1440                       (24h cap, runaway-vern)
  - V10 entry-snapshot    (4 extra cols, constant per trade)
  - Reward redesign       (capital cost + never-fire penalty)
  - never-fire flag       (no exit decision could fire by max_bars=1440)

Output (per-week parquet): identity + bar-state + V8AUG-style hold rewards +
V12 reward redesign columns + V10-snapshot. V3 v8 per-bar inference is added
in Phase 3 (separate augment via score_v3_v8_on_per_bar_v1.py).

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        /tmp/v12_phase2_build_per_bar_dataset.py \\
        --decisions-dir <ENTRY_IQL_INFERENCE_FOR_V12_*>/per_week \\
        --out-root <V12_PER_BAR_*>/ \\
        [--max-weeks N for smoke test]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe  # noqa: E402
from gx1.scripts import materialize_build_exit_iql_per_bar_dataset_v1 as v1_pipe  # noqa: E402
from gx1.scripts import materialize_build_exit_iql_per_bar_dataset_v2_m1 as v8_pipe  # noqa: E402
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import compute_m5_phase_index  # noqa: E402

# ─── V12 design constants ────────────────────────────────────────────────
V12_K_HORIZONS = [12, 60, 240, 480, 1440]            # M1-bars: 12min, 1h, 4h, 8h, 24h
V12_MAX_BARS_PER_TRADE = 1440                         # 24h cap (runaway-vern, NOT a TP)
V12_HOLD_CAPITAL_COST_BPS = 0.1                       # per-bar cost while holding
V12_NEVER_FIRE_PENALTY_BPS = -1000.0                  # if no EXIT before K=1440
V12_BAR_STRIDE = 1                                    # every M1 bar
ACTION = "BUILD_EXIT_IQL_PER_BAR_DATASET_V12"

DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_FORWARD_OUTCOME_DIR = (
    DEFAULT_REPORTS_ROOT / "CANDIDATE_FORWARD_OUTCOME_RUN" / "v1_full" / "per_week"
)
DEFAULT_M1_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL"
)
MAX_INTRA_GAP_NS = fwd_pipe.MAX_INTRA_GAP_MINUTES * 60 * 1_000_000_000

CANDIDATE_CONTEXT_COLS = list(fwd_pipe.CANDIDATE_FEATURE_COLS)
CANDIDATE_IDENTITY_COLS = list(fwd_pipe.CANDIDATE_IDENTITY_COLS)


def emit_per_bar_rows_v12(
    candidate_row: dict[str, Any],
    bid_open: np.ndarray, bid_high: np.ndarray, bid_low: np.ndarray, bid_close: np.ndarray,
    ask_open: np.ndarray, ask_high: np.ndarray, ask_low: np.ndarray, ask_close: np.ndarray,
    bar_times_ns: np.ndarray,
    *,
    side: str,
    bar_stride: int,
    max_bars_per_trade: int,
    k_horizons: list[int],
) -> list[dict[str, Any]]:
    """V12 variant of v8_pipe.emit_per_bar_rows_m1.

    All V8AUG fields preserved; ADDS:
      - v10_p_long_at_entry, v10_p_short_at_entry, v10_path_quality_at_entry,
        v10_mfe_pred_at_entry  (constant per trade)
      - r_hold_K{N}_v1   = hold_max_pnl_K{N}_v1 - V12_HOLD_CAPITAL_COST_BPS * bars_in_trade
      - r_exit_now_v1    = exit_now_reward_bps_v1   (no capital cost — terminal action)
      - r_never_fire_v1  = V12_NEVER_FIRE_PENALTY_BPS if forced at K=1440 else 0
      - is_never_fire_v1 = bool flag
    """
    n_avail = len(bid_open)
    if n_avail < 2:
        return []

    # Reuse V8AUG signal computation
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

    # V12 extra: V10 entry-snapshot (constant per trade)
    v10_snapshot = {
        "v10_p_long_at_entry_v1": candidate_row.get("p_long"),
        "v10_p_short_at_entry_v1": candidate_row.get("p_short"),
        "v10_path_quality_at_entry_v1": candidate_row.get("path_quality_pred"),
        "v10_mfe_pred_at_entry_v1": candidate_row.get("mfe_first_n_pred"),
        # V10 v3+ aux heads — included as Exit-IQL state features so the
        # Q-network can learn associations like "low tf_agreement → favor exit"
        # or "trade past predicted hold_horizon → exit even on small MFE".
        # NaN for legacy bundles; gets filled with neutral 0.5 / 0.0 in Exit-IQL adapter.
        "v10_tf_agreement_at_entry_v1": candidate_row.get("tf_agreement_pred"),
        "v10_path_quality_std_at_entry_v1": candidate_row.get("path_quality_std"),
        "v10_position_size_at_entry_v1": candidate_row.get("position_size_pred"),
        "v10_hold_horizon_at_entry_v1": candidate_row.get("hold_horizon_pred"),
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
            n_lin = i + 1
            sum_x = i * n_lin / 2.0
            sum_x2 = i * n_lin * (2 * i + 1) / 6.0
            sum_y = cum_y[i]
            sum_xy = cum_xy[i]
            denom = n_lin * sum_x2 - sum_x * sum_x
            if abs(denom) > 1e-9:
                rolling_slope[i] = float((n_lin * sum_xy - sum_x * sum_y) / denom)

    bar_close_mid = ((ask_close + bid_close) / 2.0).astype(np.float64)
    m1_r5, m1_r15, m1_r60, m1_vol15, m1_vol60 = v8_pipe._compute_m1_micro_features(bar_close_mid)

    rows: list[dict[str, Any]] = []
    last_t = min(n_avail, max_bars_per_trade + 1)
    max_K = max(k_horizons)
    min_K = min(k_horizons)

    for t in range(1, last_t, bar_stride):
        remaining_peak = peak[t + 1: t + 1 + max_K]
        n_remaining = len(remaining_peak)
        if n_remaining == 0:
            hold_rewards = {f"hold_max_pnl_K{K}_v1": float(cur_pnl[t]) for K in k_horizons}
            forced_terminal = True
        else:
            hold_rewards = {}
            for K in k_horizons:
                K_eff = min(K, n_remaining)
                hold_rewards[f"hold_max_pnl_K{K}_v1"] = float(np.max(remaining_peak[:K_eff]))
            forced_terminal = (n_remaining < min_K)

        # Never-fire: no exit decision possible by max(K_HORIZONS) and we hit
        # the trade-cap. This is the V12 trainer signal that the model failed
        # to commit (different from forced_terminal which is data-end).
        is_never_fire = bool(forced_terminal and t >= max_bars_per_trade)

        bar_ts_ns = int(bar_times_ns[t]) if t < len(bar_times_ns) else 0
        if bar_ts_ns > 0:
            ts_val = pd.Timestamp(bar_ts_ns, tz="UTC")
            phase_idx = compute_m5_phase_index(ts_val)
        else:
            phase_idx = 0
        m5_phase = {f"m5_phase_{i}_v1": (1.0 if i == phase_idx else 0.0) for i in range(5)}

        # V12 reward redesign columns
        bars_held = int(t)
        capital_cost = V12_HOLD_CAPITAL_COST_BPS * bars_held
        r_hold = {
            f"r_hold_K{K}_v1": float(hold_rewards[f"hold_max_pnl_K{K}_v1"] - capital_cost)
            for K in k_horizons
        }

        row: dict[str, Any] = {
            # Identity
            "run_id": candidate_row.get("run_id"),
            "candidate_uid": candidate_row.get("candidate_uid"),
            "decision_ts_utc": candidate_row.get("decision_ts_utc"),
            "side_v1": side,
            "bar_idx_v1": int(t),
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
            # EXIT_NOW reward (V8AUG legacy)
            "exit_now_reward_bps_v1": float(cur_pnl[t]),
            # HOLD rewards per K (V8AUG legacy, peak-tracking)
            **hold_rewards,
            # Quality
            "forced_terminal_v1": bool(forced_terminal),
            "forward_bars_remaining_v1": int(n_remaining),
            "cadence_v1": "M1",
            # ── V12 redesign columns ──────────────────────────────────────
            "r_exit_now_v1": float(cur_pnl[t]),
            **r_hold,
            "r_never_fire_v1": float(V12_NEVER_FIRE_PENALTY_BPS) if is_never_fire else 0.0,
            "is_never_fire_v1": is_never_fire,
            "v12_capital_cost_bps_v1": float(capital_cost),
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
        row.update(v10_snapshot)
        rows.append(row)
    return rows


def _rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema | None) -> pa.Table:
    df = pd.DataFrame(rows)
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    for col in schema.names:
        if col not in df.columns:
            df[col] = None
    df = df[list(schema.names)]
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)


def process_week_v12(
    week_id: str,
    fw_path: Path,
    decisions_path: Path,
    m1_time_ns: np.ndarray,
    m1_arrays: dict[str, np.ndarray],
    *,
    out_path: Path,
    schema_template: pa.Schema | None = None,
    candidates_per_batch: int = 25,
) -> tuple[dict[str, Any], pa.Schema | None]:
    """Build V12 per-bar rows for one week. Returns (stats, schema)."""
    stats: dict[str, Any] = {
        "week_id": week_id,
        "forward_outcome_path": str(fw_path),
        "decisions_path": str(decisions_path),
        "output_path": str(out_path),
        "input_candidates": 0,
        "take_candidates": 0,
        "output_rows": 0,
        "skipped_no_decision_ts": 0,
        "skipped_no_directional_side": 0,
        "skipped_decision_past_tape_end": 0,
        # Diagnostic: cases where Entry-IQL chose LONG/SHORT but V10's p_long/p_short
        # disagreed with that direction. By V12 design Entry-IQL is policy (we trust it),
        # so we log but do NOT override. High counts here = Entry-IQL learned a non-trivial
        # override pattern over V10.
        "iql_v10_disagreement_long": 0,
        "iql_v10_disagreement_short": 0,
        "process_status": "UNKNOWN",
        "cadence_v1": "M1",
    }
    if not fw_path.exists() or not decisions_path.exists():
        stats["process_status"] = "INPUT_MISSING"
        return stats, schema_template

    fw = pd.read_parquet(fw_path)
    decisions = pd.read_parquet(decisions_path, columns=["candidate_uid", "action_label_v1"])
    stats["input_candidates"] = int(len(fw))

    # Inner join + filter to TAKE only
    merged = fw.merge(decisions, on="candidate_uid", how="inner")
    take = merged[merged["action_label_v1"].isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW"])].reset_index(drop=True)
    stats["take_candidates"] = int(len(take))
    if len(take) == 0:
        stats["process_status"] = "NO_TAKE_CANDIDATES"
        return stats, schema_template

    n_m1 = len(m1_time_ns)
    cand_dict = {col: take[col].to_list() if col in take.columns else [None] * len(take)
                 for col in CANDIDATE_IDENTITY_COLS + CANDIDATE_CONTEXT_COLS + ["action_label_v1"]}

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
        for row_i in range(len(take)):
            decision_ts_str = cand_dict["decision_ts_utc"][row_i]
            ts_ns = fwd_pipe._safe_decision_ts_to_int64_ns(decision_ts_str)
            if ts_ns is None:
                stats["skipped_no_decision_ts"] += 1
                continue
            candidate_row = {col: cand_dict[col][row_i] for col in cand_dict}

            # Side comes from Entry-IQL action — not derived from p_long/p_short.
            # By V12 design Entry-IQL is the entry policy (trusted), but we log
            # disagreements with V10's direction-belief for observability.
            action_label = candidate_row["action_label_v1"]
            p_long = candidate_row.get("p_long") or 0.0
            p_short = candidate_row.get("p_short") or 0.0
            if action_label == "TAKE_LONG_NOW":
                side = "long"
                if p_long < p_short:
                    stats["iql_v10_disagreement_long"] += 1
            elif action_label == "TAKE_SHORT_NOW":
                side = "short"
                if p_short < p_long:
                    stats["iql_v10_disagreement_short"] += 1
            else:
                stats["skipped_no_directional_side"] += 1
                continue

            idx_start = int(np.searchsorted(m1_time_ns, ts_ns, side="right"))
            if idx_start >= n_m1:
                stats["skipped_decision_past_tape_end"] += 1
                continue
            s_t, e_t, _ = fwd_pipe.slice_forward_window(
                m1_time_ns, n_m1, idx_start,
                V12_MAX_BARS_PER_TRADE + max(V12_K_HORIZONS) + 1,
                MAX_INTRA_GAP_NS,
            )
            bar_times_ns = m1_time_ns[s_t:e_t]
            rows = emit_per_bar_rows_v12(
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
                bar_stride=V12_BAR_STRIDE,
                max_bars_per_trade=V12_MAX_BARS_PER_TRADE,
                k_horizons=V12_K_HORIZONS,
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


def _week_id_from_forward_outcome(fp: Path) -> str:
    """forward_outcomes_TRUTH_MONFRI_WEEK_20200106_20200113 → TRUTH_MONFRI_WEEK_20200106_20200113"""
    name = fp.stem
    if name.startswith("forward_outcomes_"):
        return name[len("forward_outcomes_"):]
    return name


def main() -> int:
    p = argparse.ArgumentParser(description=f"Build {ACTION}.")
    p.add_argument("--decisions-dir", type=str, required=True,
                   help="Path to ENTRY_IQL_INFERENCE_FOR_V12_*/per_week/")
    p.add_argument("--out-root", type=str, default=None)
    p.add_argument("--forward-outcome-dir", type=str, default=str(DEFAULT_FORWARD_OUTCOME_DIR))
    p.add_argument("--m1-tape-root", type=str, default=str(DEFAULT_M1_TAPE_ROOT))
    p.add_argument("--max-weeks", type=int, default=None,
                   help="Smoke-test cap. None = all 276 weeks.")
    p.add_argument("--candidates-per-batch", type=int, default=25)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    decisions_dir = Path(args.decisions_dir).expanduser().resolve()
    fwd_dir = Path(args.forward_outcome_dir).expanduser().resolve()
    m1_root = Path(args.m1_tape_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = (Path(args.out_root).expanduser().resolve()
                if args.out_root
                else DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    out_root.mkdir(parents=True, exist_ok=True)
    per_week_dir = out_root / "per_week"
    per_week_dir.mkdir(exist_ok=True)

    print(f"V12 PHASE 2 — {ACTION}")
    print(f"  decisions_dir   = {decisions_dir}")
    print(f"  fwd_outcome_dir = {fwd_dir}")
    print(f"  m1_tape_root    = {m1_root}")
    print(f"  out_root        = {out_root}")

    print(f"\n[1/3] Loading M1 tape...")
    t0 = time.time()
    m1 = fwd_pipe.load_m5_tape(m1_root)  # generic year-partition loader
    print(f"  M1 bars: {len(m1):,}  range={m1['time'].min()} → {m1['time'].max()}  ({time.time()-t0:.1f}s)")
    m1_time_ns = m1["time"].astype("int64").to_numpy()
    m1_arrays = {
        col: m1[col].astype(np.float64).to_numpy()
        for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                    "ask_open", "ask_high", "ask_low", "ask_close")
    }
    del m1

    fw_files = sorted(fwd_dir.glob("*.parquet"))
    if args.max_weeks:
        fw_files = fw_files[: args.max_weeks]
    print(f"\n[2/3] Processing {len(fw_files)} weeks...")
    t0 = time.time()

    per_week_stats: list[dict[str, Any]] = []
    schema_template: pa.Schema | None = None
    total_take = 0
    total_rows = 0

    for w_idx, fw_path in enumerate(fw_files, start=1):
        week_id = _week_id_from_forward_outcome(fw_path)
        decisions_path = decisions_dir / f"{fw_path.stem}.parquet"  # decisions filename matches fw stem
        # Filename convention: match V8AUG (`exit_per_bar_m1_*.parquet`) so that
        # `score_v3_v8_on_per_bar_v1.py` (Phase 3) can glob this dir directly.
        out_path = per_week_dir / f"exit_per_bar_m1_{week_id}.parquet"
        stats_path = per_week_dir / f"exit_per_bar_m1_{week_id}_stats.json"

        if not args.no_skip_existing and out_path.exists() and stats_path.exists():
            cached = json.loads(stats_path.read_text())
            per_week_stats.append(cached)
            total_take += int(cached.get("take_candidates", 0))
            total_rows += int(cached.get("output_rows", 0))
            continue

        stats, schema_template = process_week_v12(
            week_id, fw_path, decisions_path, m1_time_ns, m1_arrays,
            out_path=out_path,
            schema_template=schema_template,
            candidates_per_batch=args.candidates_per_batch,
        )
        per_week_stats.append(stats)
        total_take += int(stats.get("take_candidates", 0))
        total_rows += int(stats.get("output_rows", 0))
        stats_path.write_text(json.dumps(stats, indent=2, default=str))

        if w_idx <= 3 or w_idx % 10 == 0 or w_idx == len(fw_files):
            elapsed = time.time() - t0
            print(f"  [{w_idx}/{len(fw_files)}] {week_id}: "
                  f"input={stats['input_candidates']:3d} take={stats['take_candidates']:3d} "
                  f"rows={stats['output_rows']:6d} status={stats['process_status']} "
                  f"elapsed={elapsed:.0f}s total_rows={total_rows:,}")

    print(f"\n[3/3] Writing summary + manifest...")
    summary = {
        "layer_name": "EXIT_IQL_PER_BAR_DATASET_V12_SUMMARY",
        "action_v1": ACTION,
        "artifact_root_v1": str(out_root),
        "built_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "cadence_v1": "M1",
        "k_horizons_v1": V12_K_HORIZONS,
        "max_bars_per_trade_v1": V12_MAX_BARS_PER_TRADE,
        "hold_capital_cost_bps_v1": V12_HOLD_CAPITAL_COST_BPS,
        "never_fire_penalty_bps_v1": V12_NEVER_FIRE_PENALTY_BPS,
        "weeks_processed_v1": len(per_week_stats),
        "weeks_ok_v1": sum(1 for s in per_week_stats if s.get("process_status") == "OK"),
        "weeks_input_missing_v1": sum(1 for s in per_week_stats if s.get("process_status") == "INPUT_MISSING"),
        "total_take_candidates_v1": int(total_take),
        "total_output_rows_v1": int(total_rows),
        "research_only_v1": True,
        "decisions_dir_v1": str(decisions_dir),
        "forward_outcome_dir_v1": str(fwd_dir),
        "m1_tape_root_v1": str(m1_root),
    }
    (out_root / "summary_v1.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_root / "manifest_v1.json").write_text(json.dumps({
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(out_root),
        "per_week_dir_v1": str(per_week_dir),
        "summary_path_v1": str(out_root / "summary_v1.json"),
        "per_week_stats_v1": per_week_stats,
    }, indent=2, default=str))

    print(f"\n✅ Done. weeks={summary['weeks_processed_v1']} "
          f"ok={summary['weeks_ok_v1']} take={total_take:,} rows={total_rows:,}")
    print(f"   → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
