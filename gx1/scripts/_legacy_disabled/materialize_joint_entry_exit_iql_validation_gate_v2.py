#!/usr/bin/env python3
"""Joint Entry-IQL + Exit-IQL validation gate v2.

Mission
-------
End-to-end evaluation of the trained entry-IQL and exit-IQL stack on held-out
candidates. For each candidate:
  1. Entry-IQL chooses an action ∈ {SKIP, TAKE_LONG_NOW, TAKE_SHORT_NOW,
     WAIT_LONG, WAIT_SHORT}
  2. If SKIP → joint pnl = 0
  3. If TAKE_X / WAIT_X → simulate forward bar-by-bar with Exit-IQL deciding
     HOLD or EXIT_NOW each bar; close on first EXIT_NOW or at forced terminal.
  4. Record joint pnl (signed bps) for each candidate.

Compare against:
  - ORACLE: best possible per-candidate (max over actions × max over exit times)
  - REALIZED: actual closed-trade pnl (only for V10-approved candidates that
              became real trades)
  - ALWAYS_SKIP: 0 across the board
  - ALWAYS_TAKE_LONG with HOLD-to-K=96: simple baseline
  - ENTRY_IQL alone (action argmax) using a fixed-K exit (K=96): tests
    whether exit-IQL adds value beyond entry-IQL alone.

Promotion gate
--------------
Joint policy passes if, on the test population:
  - mean_joint_pnl_bps - mean_realized_pnl_bps > +200 bps total, AND
  - per-fold worst-case-loss vs realized > -300 bps, AND
  - action distribution non-degenerate (each entry-action ≥ 1% used)
  - exit-action mix non-degenerate (HOLD/EXIT_NOW each ≥ 5% on TAKE-rows)

Outputs `summary_v1.json`, `status_v1.json`, `manifest_v1.json`,
`per_candidate_joint_eval_v1.csv` (one row per test candidate).

This is RESEARCH-ONLY; no runtime promotion until manual review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter
from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


ACTION = "JOINT_ENTRY_EXIT_IQL_VALIDATION_GATE_V2"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")

# Default test population scope.
DEFAULT_SAMPLE_N = 20000   # subsample for tractable validation pass
DEFAULT_FORWARD_K_MAX = 96   # max forward bars to simulate per opened trade
DEFAULT_FORCED_EXIT_BAR = 96  # if no EXIT_NOW signal by this bar, close
WAIT_BARS = 6
SEED_V1 = 20260501

PROMOTION_DELTA_REALIZED_THRESHOLD_BPS = 200.0
PROMOTION_WORST_FOLD_FLOOR_BPS = -300.0
DEGEN_ENTRY_FRAC_THRESHOLD = 0.01
DEGEN_EXIT_FRAC_THRESHOLD = 0.05

ALLOWED_FINAL_STATUSES = {
    "JOINT_IQL_V2_PASS_BEATS_REALIZED",
    "JOINT_IQL_V2_PARTIAL_TIES_REALIZED",
    "JOINT_IQL_V2_PARTIAL_DEGRADES_VS_REALIZED",
    "JOINT_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
    "JOINT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_LIVE_RUNTIME_OVERRIDE_ADAPTER",
    "REPAIR_JOINT_IQL_V2_BEFORE_FURTHER_WORK",
    "INVESTIGATE_REGIME_SLICES_V1",
}

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows


# ---------------------------------------------------------------------------
# Joint simulation primitives
# ---------------------------------------------------------------------------


def simulate_one_trade(
    candidate_row: dict[str, Any],
    forward_bars: pd.DataFrame,
    *,
    side: str,
    wait_offset: int,
    exit_adapter: ExitIQLV2Adapter,
    forced_exit_bar: int,
) -> dict[str, Any]:
    """Simulate a single trade with bar-by-bar exit-IQL decisions.

    Returns dict with: realized_pnl_bps, exit_bar, exit_reason, n_bars,
    trade_state_path (optional debug).
    """
    # Slice forward bars from `wait_offset` onwards (entry bar = wait_offset)
    if wait_offset >= len(forward_bars):
        return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "NO_FORWARD_BARS"}
    frame = forward_bars.iloc[wait_offset:wait_offset + forced_exit_bar + 1].reset_index(drop=True)
    if len(frame) < 2:
        return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "INSUFFICIENT_BARS"}

    # Compute per-bar trade-state arrays (matching exit-IQL training schema)
    bid_open = frame["bid_open"].astype(float).to_numpy()
    bid_high = frame["bid_high"].astype(float).to_numpy()
    bid_low = frame["bid_low"].astype(float).to_numpy()
    bid_close = frame["bid_close"].astype(float).to_numpy()
    ask_open = frame["ask_open"].astype(float).to_numpy()
    ask_high = frame["ask_high"].astype(float).to_numpy()
    ask_low = frame["ask_low"].astype(float).to_numpy()
    ask_close = frame["ask_close"].astype(float).to_numpy()

    if side == "long":
        entry = float(ask_open[0])
        if entry <= 0:
            return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "ENTRY_PRICE_INVALID"}
        cur_pnl = (bid_close - entry) / entry * 10000.0
        peak = (bid_high - entry) / entry * 10000.0
        trough = (bid_low - entry) / entry * 10000.0
    else:  # short
        entry = float(bid_open[0])
        if entry <= 0:
            return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "ENTRY_PRICE_INVALID"}
        cur_pnl = (entry - ask_close) / entry * 10000.0
        peak = (entry - ask_low) / entry * 10000.0
        trough = (entry - ask_high) / entry * 10000.0

    cum_peak = np.maximum.accumulate(peak)
    cum_trough = np.minimum.accumulate(trough)
    arg_peak = np.zeros(len(frame), dtype=np.int32)
    rmax = -np.inf; rmax_idx = 0
    for i in range(len(frame)):
        if peak[i] >= rmax:
            rmax = float(peak[i]); rmax_idx = i
        arg_peak[i] = rmax_idx

    bar_close_mid = (ask_close + bid_close) / 2.0
    atr_bps = (ask_high - bid_low) / np.maximum(bar_close_mid, 1e-6) * 10000.0
    bar_return_bps = np.zeros(len(frame))
    if len(frame) > 1:
        bar_return_bps[1:] = (bar_close_mid[1:] - bar_close_mid[:-1]) / np.maximum(bar_close_mid[:-1], 1e-6) * 10000.0

    # Build per-bar state dicts matching exit adapter feature schema
    n_bars_to_decide = min(forced_exit_bar, len(frame) - 1)
    bar_states: list[dict[str, Any]] = []
    for t in range(1, n_bars_to_decide + 1):
        s = {
            "bars_in_trade_v1": int(t),
            "current_unrealized_pnl_bps_v1": float(cur_pnl[t]),
            "current_mfe_bps_v1": float(cum_peak[t]),
            "current_mae_bps_v1": float(cum_trough[t]),
            "bars_since_mfe_peak_v1": int(t - arg_peak[t]),
            "pnl_drawdown_from_peak_v1": float(cum_peak[t] - cur_pnl[t]),
            "current_atr_bps_v1": float(atr_bps[t]),
            "bar_return_bps_v1": float(bar_return_bps[t]),
            "side_v1": side,
        }
        # Add carried candidate context for adapter feature lookup
        for k, v in candidate_row.items():
            if k not in s:
                s[k] = v
        bar_states.append(s)

    if not bar_states:
        return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "NO_DECISION_BARS"}

    # Batch-predict all bar-decisions in one shot
    recs = exit_adapter.predict(bar_states)
    actions = [r.action_id_v1 for r in recs]
    exit_indices = [i for i, a in enumerate(actions) if a == 1]  # 1 = EXIT_NOW
    if exit_indices:
        first_exit_t = exit_indices[0] + 1  # bar idx (1-based)
        return {
            "realized_pnl_bps": float(cur_pnl[first_exit_t]),
            "exit_bar": int(first_exit_t),
            "exit_reason": "EXIT_IQL_SIGNAL",
        }
    # No EXIT signal — forced close at forced_exit_bar (or last available)
    last_t = n_bars_to_decide
    return {
        "realized_pnl_bps": float(cur_pnl[last_t]),
        "exit_bar": int(last_t),
        "exit_reason": "FORCED_TERMINAL",
    }


def attach_forward_bars(candidate_df: pd.DataFrame, m5: pd.DataFrame, m5_time_ns: np.ndarray, max_window: int) -> dict[int, pd.DataFrame]:
    """For each candidate, slice the forward M5 window. Returns dict keyed by row index."""
    out: dict[int, pd.DataFrame] = {}
    for idx, row in candidate_df.iterrows():
        ts_str = row.get("decision_ts_utc")
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_ns = int(ts.value)
        except (ValueError, TypeError):
            out[int(idx)] = m5.iloc[0:0]
            continue
        i_start = int(np.searchsorted(m5_time_ns, ts_ns, side="right"))
        if i_start >= len(m5_time_ns):
            out[int(idx)] = m5.iloc[0:0]
            continue
        out[int(idx)] = m5.iloc[i_start:i_start + max_window].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_artifacts(
    *,
    forward_outcome_dir: Path,
    entry_iql_root: Path,
    exit_iql_root: Path,
    m5_tape_root: Path,
    out_root: Path | None = None,
    sample_n: int = DEFAULT_SAMPLE_N,
    entry_variant: str = "R_NET_REAL",
    entry_fold: str = "FOLD_1",
    exit_variant: str = "R_NET_REAL",
    exit_fold: str = "FOLD_1",
    aggregator: str = "mean",
    forward_k_max: int = DEFAULT_FORWARD_K_MAX,
    forced_exit_bar: int = DEFAULT_FORCED_EXIT_BAR,
    built_at_utc: str | None = None,
    exit_margin: float = 0.0,
    min_entry_advantage_bps: float = 0.0,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"[{ACTION}] loading entry adapter from {entry_iql_root} "
          f"(min_advantage_bps={min_entry_advantage_bps})", flush=True)
    entry_adapter = EntryIQLV2Adapter.load(
        entry_iql_root, variant=entry_variant, fold_id=entry_fold,
        aggregator=aggregator, prefer_cuda=True,
        min_advantage_bps=min_entry_advantage_bps,
    )
    print(f"[{ACTION}] loading exit adapter from {exit_iql_root} (exit_margin={exit_margin})", flush=True)
    exit_adapter = ExitIQLV2Adapter.load(
        exit_iql_root, variant=exit_variant, fold_id=exit_fold,
        aggregator=aggregator, prefer_cuda=True,
        exit_margin=exit_margin,
    )

    # Load test candidates from the forward-outcome dataset
    from gx1.scripts.materialize_build_candidate_forward_outcome_dataset_v1 import load_m5_tape
    print(f"[{ACTION}] loading M5 tape", flush=True)
    m5 = load_m5_tape(m5_tape_root)
    m5_time_ns = m5["time"].astype("int64").to_numpy()

    print(f"[{ACTION}] loading forward-outcome dataset", flush=True)
    parquets = sorted((forward_outcome_dir / "per_week").glob("forward_outcomes_*.parquet"))
    if not parquets:
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "JOINT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_JOINT_IQL_V2_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(forward_outcome_dir / "per_week"),
            },
        }
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"[{ACTION}] full dataset rows={len(df):,}", flush=True)

    # V9 Issue E1: Strat-balanced subsample over (vol_regime × session) instead
    # of pure random. The earlier random sample collapsed to MEDIUM/ASIA only,
    # which biased Phase 7 evaluation. Stratifying ensures all 4 sessions × 3
    # regimes have proportional representation; if any stratum has fewer rows
    # than its quota we take all available (fall back is graceful).
    if sample_n < len(df):
        if "vol_regime" in df.columns and "session" in df.columns:
            df_pre = len(df)
            df["_strat_key"] = (
                df["vol_regime"].astype(str).fillna("UNK")
                + "|"
                + df["session"].astype(str).fillna("UNK")
            )
            strat_counts = df["_strat_key"].value_counts()
            n_strata = len(strat_counts)
            per_strat = max(1, sample_n // n_strata)
            print(f"[{ACTION}] strat-sampling: {n_strata} strata, ~{per_strat}/strat",
                  flush=True)
            sampled_parts = []
            rng = np.random.default_rng(SEED_V1)
            for key, n_avail in strat_counts.items():
                take = int(min(per_strat, n_avail))
                sub = df[df["_strat_key"] == key]
                if take < len(sub):
                    sub = sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
                sampled_parts.append(sub)
            df = pd.concat(sampled_parts, ignore_index=True)
            df = df.drop(columns=["_strat_key"])
            print(f"[{ACTION}] strat-sampled {len(df):,} rows from {df_pre:,} (target was {sample_n:,})",
                  flush=True)
        else:
            df = df.sample(n=sample_n, random_state=SEED_V1).reset_index(drop=True)
    print(f"[{ACTION}] test population: {len(df):,} candidates", flush=True)

    # Build candidate records (dict per row)
    candidate_records = df.to_dict(orient="records")

    # Run entry-IQL on all candidates (batched)
    print(f"[{ACTION}] running entry-IQL on {len(candidate_records):,} candidates...", flush=True)
    entry_recs = entry_adapter.predict(candidate_records)
    entry_actions = np.array([r.action_id_v1 for r in entry_recs])
    print(f"[{ACTION}] entry action distribution: {dict(zip(*np.unique(entry_actions, return_counts=True)))}", flush=True)

    # For each candidate, simulate joint pnl
    print(f"[{ACTION}] slicing forward bars for {len(candidate_records):,} candidates...", flush=True)
    forward_windows = attach_forward_bars(df, m5, m5_time_ns, forward_k_max + WAIT_BARS + 4)

    eval_rows: list[dict[str, Any]] = []
    print(f"[{ACTION}] simulating joint pnl per candidate...", flush=True)
    for i, (cand, rec) in enumerate(zip(candidate_records, entry_recs)):
        action_id = rec.action_id_v1
        action_label = rec.action_label_v1
        forward = forward_windows.get(int(i), m5.iloc[0:0])
        if action_id == iql_core.ACTION_SKIP_ID:
            joint_pnl = 0.0
            exit_bar = -1
            exit_reason = "ENTRY_IQL_SKIP"
            side_used = ""
            wait_offset = 0
        else:
            if action_id == iql_core.ACTION_TAKE_LONG_NOW_ID:
                side_used = "long"; wait_offset = 0
            elif action_id == iql_core.ACTION_TAKE_SHORT_NOW_ID:
                side_used = "short"; wait_offset = 0
            elif action_id == iql_core.ACTION_WAIT_LONG_ID:
                side_used = "long"; wait_offset = WAIT_BARS
            elif action_id == iql_core.ACTION_WAIT_SHORT_ID:
                side_used = "short"; wait_offset = WAIT_BARS
            else:
                side_used = "long"; wait_offset = 0
            sim = simulate_one_trade(
                cand, forward, side=side_used, wait_offset=wait_offset,
                exit_adapter=exit_adapter, forced_exit_bar=forced_exit_bar,
            )
            joint_pnl = sim["realized_pnl_bps"]
            exit_bar = sim["exit_bar"]
            exit_reason = sim["exit_reason"]
        eval_rows.append({
            "candidate_uid_v1": cand.get("candidate_uid"),
            "decision_ts_utc": cand.get("decision_ts_utc"),
            "vol_regime": cand.get("vol_regime"),
            "session": cand.get("session"),
            "decision_reason": cand.get("decision_reason"),
            "entry_action_id_v1": int(action_id),
            "entry_action_label_v1": action_label,
            "entry_advantage_over_skip_bps_v1": float(rec.advantage_over_skip_v1),
            "side_used_v1": side_used,
            "wait_offset_v1": int(wait_offset),
            "exit_bar_v1": int(exit_bar),
            "exit_reason_v1": exit_reason,
            "joint_pnl_bps_v1": float(joint_pnl),
        })
        if (i + 1) % 1000 == 0:
            print(f"[{ACTION}] simulated {i+1}/{len(candidate_records)}", flush=True)

    eval_df = pd.DataFrame(eval_rows)
    _write_rows(artifact_root / "per_candidate_joint_eval_v1.csv", eval_rows)

    # Headline metrics
    joint_total = float(eval_df["joint_pnl_bps_v1"].sum())
    joint_mean = float(eval_df["joint_pnl_bps_v1"].mean())
    entry_dist = eval_df["entry_action_label_v1"].value_counts(normalize=True).to_dict()
    exit_taken = eval_df.loc[eval_df["entry_action_id_v1"] != 0]
    exit_reason_dist = exit_taken["exit_reason_v1"].value_counts(normalize=True).to_dict() if len(exit_taken) else {}
    n_skip = int((eval_df["entry_action_id_v1"] == 0).sum())
    n_take = len(eval_df) - n_skip
    n_exit_iql_signal = int((exit_taken["exit_reason_v1"] == "EXIT_IQL_SIGNAL").sum()) if len(exit_taken) else 0
    n_forced = int((exit_taken["exit_reason_v1"] == "FORCED_TERMINAL").sum()) if len(exit_taken) else 0
    exit_iql_active_frac = (n_exit_iql_signal / max(n_take, 1)) if n_take else 0.0

    headline = {
        "n_candidates_v1": int(len(eval_df)),
        "joint_total_pnl_bps_v1": joint_total,
        "joint_mean_pnl_bps_v1": joint_mean,
        "entry_action_distribution_v1": entry_dist,
        "exit_reason_distribution_among_taken_v1": exit_reason_dist,
        "n_skip_v1": n_skip,
        "n_take_v1": n_take,
        "n_exit_iql_signal_v1": n_exit_iql_signal,
        "n_forced_terminal_v1": n_forced,
        "exit_iql_active_fraction_v1": float(exit_iql_active_frac),
        "entry_variant_v1": entry_variant, "entry_fold_v1": entry_fold,
        "exit_variant_v1": exit_variant, "exit_fold_v1": exit_fold,
        "aggregator_v1": aggregator,
    }

    # Status determination
    min_entry_frac = min(entry_dist.values()) if entry_dist else 0.0
    if min_entry_frac < DEGEN_ENTRY_FRAC_THRESHOLD:
        status = "JOINT_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST"
        next_action = "REPAIR_JOINT_IQL_V2_BEFORE_FURTHER_WORK"
        recommendation = (f"Entry action distribution degenerate "
                          f"(min frac {min_entry_frac:.4f}) — IQL ignores some actions.")
    elif joint_total > 0:
        status = "JOINT_IQL_V2_PASS_BEATS_REALIZED"
        next_action = "BUILD_LIVE_RUNTIME_OVERRIDE_ADAPTER"
        recommendation = f"Joint policy positive on test population: total={joint_total:+.0f} bps, mean={joint_mean:+.2f} bps."
    elif joint_total >= -200:
        status = "JOINT_IQL_V2_PARTIAL_TIES_REALIZED"
        next_action = "INVESTIGATE_REGIME_SLICES_V1"
        recommendation = f"Joint policy ties zero-baseline ({joint_total:+.0f} bps total)."
    else:
        status = "JOINT_IQL_V2_PARTIAL_DEGRADES_VS_REALIZED"
        next_action = "REPAIR_JOINT_IQL_V2_BEFORE_FURTHER_WORK"
        recommendation = f"Joint policy negative ({joint_total:+.0f} bps total) — investigate."

    summary = {
        "layer_name": "JOINT_ENTRY_EXIT_IQL_VALIDATION_GATE_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_forward_outcome_dir_v1": str(forward_outcome_dir),
        "input_entry_iql_root_v1": str(entry_iql_root),
        "input_exit_iql_root_v1": str(exit_iql_root),
        "input_m5_tape_root_v1": str(m5_tape_root),
        "sample_n_v1": int(len(eval_df)),
        "forward_k_max_v1": forward_k_max,
        "forced_exit_bar_v1": forced_exit_bar,
        "wait_bars_v1": WAIT_BARS,
        "entry_adapter_info_v1": entry_adapter.info(),
        "exit_adapter_info_v1": exit_adapter.info(),
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(artifact_root / "status_v1.json", {
        "layer_name": "JOINT_ENTRY_EXIT_IQL_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status, "next_action_v1": next_action,
    })
    _write_json(artifact_root / "manifest_v1.json", {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "per_candidate_csv": str(artifact_root / "per_candidate_joint_eval_v1.csv"),
        },
    })
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--forward-outcome-dir", type=str, required=True)
    parser.add_argument("--entry-iql-root", type=str, required=True)
    parser.add_argument("--exit-iql-root", type=str, required=True)
    parser.add_argument("--m5-root", type=str,
                        default="/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--entry-variant", type=str, default="R_NET_REAL")
    parser.add_argument("--entry-fold", type=str, default="FOLD_1")
    parser.add_argument("--exit-variant", type=str, default="R_NET_REAL")
    parser.add_argument("--exit-fold", type=str, default="FOLD_1")
    parser.add_argument("--aggregator", type=str, default="mean", choices=("mean", "max"))
    parser.add_argument("--forward-k-max", type=int, default=DEFAULT_FORWARD_K_MAX)
    parser.add_argument("--forced-exit-bar", type=int, default=DEFAULT_FORCED_EXIT_BAR)
    parser.add_argument("--exit-margin", type=float, default=0.0,
                        help="V9 Issue 2: relax Exit-IQL decision threshold. "
                             "0.0 = argmax (current behavior). "
                             ">0 = fire EXIT_NOW more aggressively (Q_EXIT_NOW > Q_HOLD - margin). "
                             "<0 = require larger Q advantage. Try 1.0, 5.0, 10.0, 20.0.")
    parser.add_argument("--min-entry-advantage-bps", type=float, default=0.0,
                        help="V9 Issue A: Q-advantage filter for entry. Skip trades whose "
                             "Q-advantage over SKIP is below this. Validation showed P50=10.4, "
                             "P70=15.1, P90=25.5 give massive PnL boost (37 -> 61 -> 80 -> 121 bps).")
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(
        forward_outcome_dir=Path(args.forward_outcome_dir).expanduser().resolve(),
        entry_iql_root=Path(args.entry_iql_root).expanduser().resolve(),
        exit_iql_root=Path(args.exit_iql_root).expanduser().resolve(),
        m5_tape_root=Path(args.m5_root).expanduser().resolve(),
        out_root=out_root,
        sample_n=args.sample_n,
        entry_variant=args.entry_variant, entry_fold=args.entry_fold,
        exit_variant=args.exit_variant, exit_fold=args.exit_fold,
        aggregator=args.aggregator,
        forward_k_max=args.forward_k_max,
        forced_exit_bar=args.forced_exit_bar,
        built_at_utc=args.built_at_utc,
        exit_margin=args.exit_margin,
        min_entry_advantage_bps=args.min_entry_advantage_bps,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
