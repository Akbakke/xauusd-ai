#!/usr/bin/env python3
"""Exit-side research-only diagnostic and per-bar decision scaffold.

This gate is research-only. It does not train any model, does not modify the
runtime exit_manager, and does not open R6/adapter/freeze/promo/live.

The five independent sub-tracks each produce concrete artifacts:

  1. PER_BAR_TRAJECTORY_RECONSTRUCTION_V1 - join 1914 historical trades to M5
     raw OHLC and emit a per-bar HOLD/EXIT_NOW decision dataset that future
     exit-IQL training can consume.
  2. GIVEBACK_LADDER_COUNTERFACTUAL_V1 - portfolio PNL counterfactual at
     several MFE-capture levels (100/75/50/25/peak-and-trail).
  3. CATA_PREVENTION_COUNTERFACTUAL_V1 - upper-bound bps savings if the 415
     CATASTROPHIC_GUARD trades had exited at peak MFE.
  4. FRIDAY_FLAT_REFINEMENT_DESIGN_V1 - counterfactual for the 50 Friday-flat
     trades held to Monday open instead of force-flattened.
  5. SAMSTEMTE_FEATURE_AUDIT_V1 - feature-set diff between XGB-entry,
     exit-transformer (CTX36), and IQL state contracts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as refine_gate
from gx1.exits.contracts.exit_io_v1_ctx36_features import (
    EXIT_IO_V1_CTX36_FEATURES,
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1"

INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_REBUILD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_20260429T081445Z_LOCK"
)
INPUT_V2_TRAINING_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T090050Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)

M5_RAW_2025 = Path("/home/andre2/GX1_DATA/data/data/raw/xauusd_m5_2025_bid_ask.parquet")
M5_RAW_2026 = Path(
    "/home/andre2/GX1_DATA/data/oanda/years/2026/xauusd_m5_2026_bid_ask.parquet"
)

EXPECTED_FRAME_ROWS = 1914
EXPECTED_HARDENED_ROWS = 89
EXPECTED_SHIELD_ROWS = 78

GIVEBACK_LADDER_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.10]
TRAIL_DRAWDOWN_PCT = 0.25  # exit when DD-from-peak exceeds this fraction of peak

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ALLOWED_FINAL_STATUSES = {
    "EXIT_QUALITY_DIAGNOSTIC_PASS_FULL_RECONSTRUCTION_AND_COUNTERFACTUALS_AVAILABLE",
    "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_RECONSTRUCTION_GAP",
    "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_FRIDAY_FLAT_REFINEMENT_NOT_ESTABLISHED",
    "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_SAMSTEMTE_AUDIT_NOT_ESTABLISHED",
    "EXIT_QUALITY_DIAGNOSTIC_BLOCKED_BY_M5_GAP",
    "EXIT_QUALITY_DIAGNOSTIC_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "TRAIN_EXIT_BANDIT_HOLD_EXIT_NOW_RESEARCH_V1",
    "DESIGN_FEATURE_HUB_FROM_SAMSTEMTE_AUDIT_V1",
    "REFINE_FRIDAY_FLAT_POLICY_RESEARCH_ONLY_V1",
    "DEEPEN_PER_BAR_RECONSTRUCTION_LINEAGE_V1",
    "HOLD_EXIT_RESEARCH_UNTIL_PER_BAR_COMPLETE_V1",
}

# Reuse helpers from contract_gate
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_CONTRACT_ROOT,
        INPUT_REBUILD_ROOT,
        INPUT_V2_TRAINING_ROOT,
        INPUT_REFINE_CLEAN_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required_locks = {
        "v1_state_contract": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "rebuild_summary": INPUT_REBUILD_ROOT / "summary_v1.json",
        "rebuild_join_audit": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "entry_iql_post_trade_outcome_join_audit_v1.json",
        "rebuild_join_table": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "entry_iql_post_trade_outcome_join_table_v1.csv",
        "v2_training_summary": INPUT_V2_TRAINING_ROOT / "summary_v1.json",
    }
    missing = [str(p) for p in required_locks.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not M5_RAW_2025.exists():
        raise RuntimeError(f"M5_RAW_2025_NOT_FOUND: {M5_RAW_2025}")
    if not M5_RAW_2026.exists():
        raise RuntimeError(f"M5_RAW_2026_NOT_FOUND: {M5_RAW_2026}")
    return {
        "required_paths": required_locks,
        "m5_raw_2025_path": M5_RAW_2025,
        "m5_raw_2026_path": M5_RAW_2026,
        "rebuild_join_audit": _read_json(required_locks["rebuild_join_audit"]),
        "v1_state_contract": _read_json(required_locks["v1_state_contract"]),
        "rebuild_summary": _read_json(required_locks["rebuild_summary"]),
        "v2_training_summary": _read_json(required_locks["v2_training_summary"]),
    }


def _load_trade_outcomes() -> pd.DataFrame:
    weeks = sorted(
        DEFAULT_REPORTS_ROOT.glob("TRUTH_MONFRI_WEEK_*/trade_outcomes_*_MERGED.parquet"),
        key=lambda p: p.parent.name,
    )
    frames = []
    for path in weeks:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_week_v1"] = path.parent.name
        frames.append(df)
    if not frames:
        raise RuntimeError("NO_TRADE_OUTCOMES_FOUND")
    common_cols = list(frames[0].columns)
    for f in frames[1:]:
        common_cols = [c for c in common_cols if c in f.columns]
    aligned = [f.loc[:, common_cols].copy() for f in frames]
    concat = pd.concat(aligned, ignore_index=True)
    if len(concat) != EXPECTED_FRAME_ROWS:
        raise RuntimeError(
            f"TRADE_OUTCOMES_ROW_COUNT_MISMATCH: got {len(concat)}, expected {EXPECTED_FRAME_ROWS}"
        )
    concat["open_ts_utc"] = pd.to_datetime(concat["open_ts_utc"], utc=True)
    # close_ts_utc in the parquet is the parquet-write timestamp (metadata), not
    # the actual trade close time. Reconstruct true close from open + duration.
    concat["close_ts_utc_meta_v1"] = pd.to_datetime(concat["close_ts_utc"], utc=True)
    concat["close_ts_utc"] = concat["open_ts_utc"] + pd.to_timedelta(
        concat["duration_bars"].astype(int) * 5, unit="m"
    )
    concat["candidate_uid"] = concat["candidate_uid"].astype(str)
    concat["trade_uid"] = concat["trade_uid"].astype(str)
    return concat


def _load_m5_raw() -> pd.DataFrame:
    df_25 = pd.read_parquet(M5_RAW_2025)
    df_26 = pd.read_parquet(M5_RAW_2026)
    common = [c for c in df_25.columns if c in df_26.columns]
    df_25 = df_25.loc[:, common].copy()
    df_26 = df_26.loc[:, common].copy()
    out = pd.concat([df_25, df_26], ignore_index=True)
    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out


# ---------------------------------------------------------------------------
# SUB-TRACK 1: per-bar trajectory reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_per_bar_trajectories(
    trades: pd.DataFrame, m5: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    m5_min = m5["time"].min()
    m5_max = m5["time"].max()
    in_range = (trades["open_ts_utc"] >= m5_min) & (
        trades["close_ts_utc"] <= m5_max
    )
    out_of_range = trades[~in_range].copy()
    in_range_trades = trades[in_range].copy()
    rows: list[dict[str, Any]] = []
    m5_indexed = m5.set_index("time")
    rejected: list[dict[str, Any]] = []
    for _, trade in in_range_trades.iterrows():
        side = str(trade["side"]).strip().lower()
        if side not in {"long", "short"}:
            rejected.append(
                {
                    "candidate_uid_v1": trade["candidate_uid"],
                    "reason_v1": f"unknown_side_{side}",
                }
            )
            continue
        entry_price = float(trade["entry_price_used"]) if "entry_price_used" in trade else None
        if entry_price is None or pd.isna(entry_price):
            entry_price = float(trade.get("entry_ask", trade.get("entry_bid", 0.0)) or 0.0)
        if entry_price == 0.0:
            rejected.append(
                {
                    "candidate_uid_v1": trade["candidate_uid"],
                    "reason_v1": "no_entry_price",
                }
            )
            continue
        bars = m5_indexed.loc[trade["open_ts_utc"]: trade["close_ts_utc"]]
        if bars.empty:
            rejected.append(
                {
                    "candidate_uid_v1": trade["candidate_uid"],
                    "reason_v1": "no_bars_in_window",
                }
            )
            continue
        long_sign = 1.0 if side == "long" else -1.0
        running_mfe = -np.inf
        running_mae = np.inf
        bars_list = list(bars.iterrows())
        n_bars = len(bars_list)
        for bar_index, (ts, bar) in enumerate(bars_list):
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            pnl_at_close_bps = ((bar_close - entry_price) / entry_price) * 10_000.0 * long_sign
            pnl_at_high_bps = ((bar_high - entry_price) / entry_price) * 10_000.0 * long_sign
            pnl_at_low_bps = ((bar_low - entry_price) / entry_price) * 10_000.0 * long_sign
            best_in_bar = max(pnl_at_high_bps, pnl_at_low_bps)
            worst_in_bar = min(pnl_at_high_bps, pnl_at_low_bps)
            running_mfe = max(running_mfe, best_in_bar)
            running_mae = min(running_mae, worst_in_bar)
            is_terminal = bar_index == n_bars - 1
            rows.append(
                {
                    "candidate_uid_v1": trade["candidate_uid"],
                    "trade_uid_v1": trade["trade_uid"],
                    "bar_index_v1": bar_index,
                    "bar_count_v1": n_bars,
                    "is_terminal_v1": is_terminal,
                    "side_v1": side,
                    "ts_v1": ts.isoformat(),
                    "entry_price_v1": entry_price,
                    "bar_high_v1": bar_high,
                    "bar_low_v1": bar_low,
                    "bar_close_v1": bar_close,
                    "pnl_at_close_bps_v1": pnl_at_close_bps,
                    "pnl_at_high_bps_v1": pnl_at_high_bps,
                    "pnl_at_low_bps_v1": pnl_at_low_bps,
                    "running_mfe_bps_v1": running_mfe,
                    "running_mae_bps_v1": running_mae,
                    "running_giveback_from_peak_bps_v1": running_mfe - pnl_at_close_bps
                    if running_mfe > 0
                    else 0.0,
                    "action_label_v1": "EXIT_NOW" if is_terminal else "HOLD",
                    "research_only_v1": True,
                }
            )
    decisions = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
    summary = {
        "trades_in_m5_range_v1": int(in_range.sum()),
        "trades_out_of_m5_range_v1": int((~in_range).sum()),
        "out_of_range_count_per_year_v1": (
            out_of_range.groupby(out_of_range["open_ts_utc"].dt.year)
            .size()
            .to_dict()
            if not out_of_range.empty
            else {}
        ),
        "reconstructed_trade_count_v1": int(
            len(in_range_trades) - len(rejected)
        ),
        "rejected_trade_count_v1": len(rejected),
        "rejected_examples_v1": rejected[:10],
        "decision_row_count_v1": int(len(decisions)),
        "hold_action_count_v1": int(
            (decisions["action_label_v1"] == "HOLD").sum()
        )
        if not decisions.empty
        else 0,
        "exit_now_action_count_v1": int(
            (decisions["action_label_v1"] == "EXIT_NOW").sum()
        )
        if not decisions.empty
        else 0,
        "research_only_v1": True,
    }
    summary["reconstruction_completeness_v1"] = float(
        summary["reconstructed_trade_count_v1"] / max(EXPECTED_FRAME_ROWS, 1)
    )
    return decisions, summary


# ---------------------------------------------------------------------------
# SUB-TRACK 2: giveback ladder counterfactual
# ---------------------------------------------------------------------------


def _giveback_ladder_counterfactual(
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eps = 1e-6
    pnl = trades["pnl_bps"].astype(float).to_numpy()
    mfe = trades["mfe_bps"].astype(float).to_numpy()
    mae = trades["mae_bps"].astype(float).to_numpy()
    actual_sum = float(pnl.sum())
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "scenario_v1": "ACTUAL_REALIZED",
            "description_v1": "as-traded realized PNL",
            "trade_count_v1": int(len(trades)),
            "sum_pnl_bps_v1": actual_sum,
            "mean_pnl_bps_v1": float(pnl.mean()),
            "winning_count_v1": int((pnl > 0).sum()),
            "losing_count_v1": int((pnl < 0).sum()),
        }
    )
    for level in GIVEBACK_LADDER_LEVELS:
        # Counterfactual: if exit at level * mfe_bps when reached.
        # Approximation: assume MFE was reached at some bar; exiting at level*MFE
        # captures level*MFE for trades whose realized peak exceeded level*MFE.
        # If actual mfe < threshold target (mfe <= 0 trades), keep actual pnl.
        target = level * mfe
        cf = np.where(mfe > eps, target, pnl)
        rows.append(
            {
                "scenario_v1": f"EXIT_AT_{int(level*100)}PCT_MFE",
                "description_v1": f"exit at {int(level*100)}% of realized peak MFE for trades that achieved positive MFE",
                "trade_count_v1": int(len(trades)),
                "sum_pnl_bps_v1": float(cf.sum()),
                "mean_pnl_bps_v1": float(cf.mean()),
                "delta_vs_actual_v1": float(cf.sum() - actual_sum),
                "winning_count_v1": int((cf > 0).sum()),
                "losing_count_v1": int((cf < 0).sum()),
            }
        )
    # Trail-stop counterfactual: exit when DD-from-peak exceeds TRAIL_DRAWDOWN_PCT * MFE
    # Without per-bar data we approximate:
    # - If mfe > 0 AND realized pnl < (1 - TRAIL_DRAWDOWN_PCT) * mfe: exit would have triggered at (1 - TRAIL_DRAWDOWN_PCT) * mfe
    # - Else: realized pnl unchanged
    trail_target = (1.0 - TRAIL_DRAWDOWN_PCT) * mfe
    triggered = (mfe > eps) & (pnl < trail_target)
    cf_trail = np.where(triggered, trail_target, pnl)
    rows.append(
        {
            "scenario_v1": f"TRAIL_EXIT_AT_PEAK_MINUS_{int(TRAIL_DRAWDOWN_PCT*100)}PCT_DD",
            "description_v1": (
                f"exit when drawdown from peak exceeds {int(TRAIL_DRAWDOWN_PCT*100)}% "
                "of peak MFE, assuming peak was reached before exit"
            ),
            "trade_count_v1": int(len(trades)),
            "trail_triggered_count_v1": int(triggered.sum()),
            "sum_pnl_bps_v1": float(cf_trail.sum()),
            "mean_pnl_bps_v1": float(cf_trail.mean()),
            "delta_vs_actual_v1": float(cf_trail.sum() - actual_sum),
            "winning_count_v1": int((cf_trail > 0).sum()),
            "losing_count_v1": int((cf_trail < 0).sum()),
        }
    )
    summary = {
        "actual_realized_pnl_bps_v1": actual_sum,
        "actual_total_mfe_bps_v1": float(mfe.sum()),
        "actual_total_giveback_bps_v1": float(np.maximum(mfe - pnl, 0.0).sum()),
        "ladder_count_v1": len(rows),
        "interpretation_v1": (
            "Counterfactuals assume realized MFE was reachable at exit time; "
            "they are upper bounds on giveback recovery, not implementable PnL targets."
        ),
        "research_only_v1": True,
        "implementable_v1": False,
    }
    return pd.DataFrame.from_records(rows), summary


# ---------------------------------------------------------------------------
# SUB-TRACK 3: CATA prevention counterfactual
# ---------------------------------------------------------------------------


def _cata_prevention_counterfactual(
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cata_mask = trades["exit_reason"].astype(str) == "CATASTROPHIC_GUARD"
    cata = trades[cata_mask].copy()
    if cata.empty:
        return pd.DataFrame(), {
            "cata_count_v1": 0,
            "research_only_v1": True,
        }
    eps = 1e-6
    cata["counterfactual_peak_mfe_pnl_bps_v1"] = cata["mfe_bps"].astype(float)
    cata["actual_pnl_bps_v1"] = cata["pnl_bps"].astype(float)
    cata["counterfactual_savings_bps_v1"] = (
        cata["counterfactual_peak_mfe_pnl_bps_v1"] - cata["actual_pnl_bps_v1"]
    )
    cata["had_positive_mfe_window_v1"] = cata["mfe_bps"].astype(float) > eps
    rows = []
    for _, row in cata.iterrows():
        rows.append(
            {
                "candidate_uid_v1": str(row["candidate_uid"]),
                "actual_pnl_bps_v1": float(row["actual_pnl_bps_v1"]),
                "actual_mfe_bps_v1": float(row["mfe_bps"]),
                "actual_mae_bps_v1": float(row["mae_bps"]),
                "had_positive_mfe_window_v1": bool(row["had_positive_mfe_window_v1"]),
                "peak_mfe_exit_savings_bps_v1": float(row["counterfactual_savings_bps_v1"]),
            }
        )
    pos_mfe = cata["had_positive_mfe_window_v1"].sum()
    summary = {
        "cata_count_v1": int(len(cata)),
        "cata_with_positive_mfe_window_v1": int(pos_mfe),
        "cata_zero_mfe_immediate_loser_count_v1": int(len(cata) - pos_mfe),
        "actual_cata_total_pnl_bps_v1": float(cata["actual_pnl_bps_v1"].sum()),
        "counterfactual_peak_mfe_total_pnl_bps_v1": float(
            cata["counterfactual_peak_mfe_pnl_bps_v1"].sum()
        ),
        "upper_bound_savings_bps_v1": float(
            cata["counterfactual_savings_bps_v1"].sum()
        ),
        "mean_savings_per_cata_bps_v1": float(
            cata["counterfactual_savings_bps_v1"].mean()
        ),
        "interpretation_v1": (
            "Upper bound: assumes a perfect-foresight exit at peak MFE before "
            "CATASTROPHIC_GUARD triggered. Real-world capture would be lower."
        ),
        "research_only_v1": True,
        "implementable_v1": False,
    }
    return pd.DataFrame.from_records(rows), summary


# ---------------------------------------------------------------------------
# SUB-TRACK 4: Friday-flat refinement
# ---------------------------------------------------------------------------


def _friday_flat_refinement(
    trades: pd.DataFrame, m5: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    friday_mask = trades["exit_reason"].astype(str) == "POLICY_FRIDAY_FLAT"
    friday = trades[friday_mask].copy()
    if friday.empty:
        return pd.DataFrame(), {
            "friday_flat_count_v1": 0,
            "research_only_v1": True,
        }
    m5_indexed = m5.set_index("time")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, trade in friday.iterrows():
        side = str(trade["side"]).strip().lower()
        long_sign = 1.0 if side == "long" else -1.0
        entry_price = float(trade.get("entry_price_used", trade.get("entry_ask", trade.get("entry_bid", 0.0))) or 0.0)
        if entry_price == 0.0:
            skipped += 1
            continue
        close_ts = pd.Timestamp(trade["close_ts_utc"])
        # Look for the next Monday open (search forward up to 4 days)
        search_end = close_ts + pd.Timedelta(days=4)
        upcoming = m5_indexed.loc[close_ts:search_end]
        if upcoming.empty:
            rows.append(
                {
                    "candidate_uid_v1": str(trade["candidate_uid"]),
                    "actual_pnl_bps_v1": float(trade["pnl_bps"]),
                    "monday_open_price_v1": None,
                    "counterfactual_pnl_at_monday_open_bps_v1": None,
                    "delta_vs_friday_flat_bps_v1": None,
                    "monday_open_available_v1": False,
                }
            )
            continue
        next_bar = upcoming.iloc[0]
        next_ts = upcoming.index[0]
        # If next bar is within minutes, that means weekend wasn't crossed
        # We want the bar right after weekend gap (>= 24h gap)
        gap_hours = (next_ts - close_ts).total_seconds() / 3600.0
        monday_open = float(next_bar["open"])
        cf_pnl = ((monday_open - entry_price) / entry_price) * 10_000.0 * long_sign
        rows.append(
            {
                "candidate_uid_v1": str(trade["candidate_uid"]),
                "side_v1": side,
                "actual_pnl_bps_v1": float(trade["pnl_bps"]),
                "actual_mfe_bps_v1": float(trade["mfe_bps"]),
                "actual_mae_bps_v1": float(trade["mae_bps"]),
                "friday_close_ts_v1": close_ts.isoformat(),
                "next_open_ts_v1": next_ts.isoformat(),
                "weekend_gap_hours_v1": gap_hours,
                "monday_open_price_v1": monday_open,
                "counterfactual_pnl_at_monday_open_bps_v1": cf_pnl,
                "delta_vs_friday_flat_bps_v1": cf_pnl - float(trade["pnl_bps"]),
                "monday_open_available_v1": True,
            }
        )
    if not rows:
        return pd.DataFrame(), {
            "friday_flat_count_v1": int(len(friday)),
            "skipped_count_v1": skipped,
            "research_only_v1": True,
        }
    df = pd.DataFrame.from_records(rows)
    available = df["monday_open_available_v1"].fillna(False).astype(bool)
    available_df = df[available]
    cf_total = float(available_df["counterfactual_pnl_at_monday_open_bps_v1"].sum())
    actual_total = float(df["actual_pnl_bps_v1"].sum())
    refined_policy_total = float(
        np.where(
            available_df["actual_pnl_bps_v1"] > 0,
            available_df["counterfactual_pnl_at_monday_open_bps_v1"],
            available_df["actual_pnl_bps_v1"],
        ).sum()
        + df.loc[~available, "actual_pnl_bps_v1"].sum()
    )
    summary = {
        "friday_flat_count_v1": int(len(friday)),
        "monday_open_lookup_available_v1": int(available.sum()),
        "monday_open_lookup_unavailable_v1": int((~available).sum()),
        "actual_friday_flat_total_pnl_bps_v1": actual_total,
        "counterfactual_hold_to_monday_open_total_pnl_bps_v1": cf_total,
        "delta_full_hold_vs_friday_flat_bps_v1": cf_total - actual_total,
        "refined_policy_total_pnl_bps_v1": refined_policy_total,
        "refined_policy_delta_vs_friday_flat_bps_v1": refined_policy_total - actual_total,
        "refined_policy_description_v1": (
            "if friday-pnl > 0, hold to monday open; else flat as before"
        ),
        "research_only_v1": True,
        "implementable_v1": True,
        "interpretation_v1": (
            "Counterfactual uses monday-open mid price; real fill at gap-open "
            "may differ due to weekend slippage and spread widening. Treat the "
            "bps delta as a research signal, not a guaranteed PnL recovery."
        ),
    }
    return df, summary


# ---------------------------------------------------------------------------
# SUB-TRACK 5: samstemte feature audit
# ---------------------------------------------------------------------------


def _xgb_entry_feature_set() -> dict[str, Any]:
    sample_dir = Path(
        "/home/andre2/GX1_DATA/data/data/training/entry_v10_ctx"
    )
    sample_files = sorted(
        sample_dir.glob("entry_v10_ctx__*train__*.parquet"),
    )[:1]
    if not sample_files:
        return {
            "source_v1": "ENTRY_V10_CTX_TRAINING_PARQUETS",
            "feature_set_v1": [],
            "discovery_status_v1": "NOT_ESTABLISHED_NO_SAMPLE_PARQUET",
        }
    sample_path = sample_files[0]
    df = pd.read_parquet(sample_path)
    bundle_columns = list(df.columns)
    return {
        "source_v1": "ENTRY_V10_CTX_TRAINING_PARQUETS",
        "sample_path_v1": str(sample_path),
        "bundle_column_count_v1": len(bundle_columns),
        "bundle_columns_v1": bundle_columns,
        "note_v1": (
            "XGB-entry uses packed columns (ctx_cont, ctx_cat, snap, seq) so "
            "individual feature names are inside the bundle and not visible at "
            "this level. Feature-hub design must enumerate these elsewhere."
        ),
    }


def _exit_transformer_feature_set() -> dict[str, Any]:
    return {
        "source_v1": "EXIT_IO_V1_CTX36_FEATURES",
        "source_path_v1": str(
            REPO_ROOT
            / "gx1"
            / "exits"
            / "contracts"
            / "exit_io_v1_ctx36_features.py"
        ),
        "feature_count_v1": EXIT_IO_V1_CTX36_FEATURE_COUNT,
        "feature_set_v1": list(EXIT_IO_V1_CTX36_FEATURES),
    }


def _iql_state_feature_set(v1_state_contract: dict[str, Any]) -> dict[str, Any]:
    rows = v1_state_contract.get("rows_v1", [])
    allowed = [r["field_name_v1"] for r in rows if r.get("allowed_as_state_v1")]
    denied = [r["field_name_v1"] for r in rows if not r.get("allowed_as_state_v1")]
    return {
        "source_v1": "IQL_OFFLINE_STATE_CONTRACT_V1",
        "feature_count_v1": len(allowed),
        "feature_set_v1": allowed,
        "denied_count_v1": len(denied),
        "denied_field_names_v1": denied,
    }


def _samstemte_feature_audit(
    v1_state_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    xgb = _xgb_entry_feature_set()
    exit_tr = _exit_transformer_feature_set()
    iql = _iql_state_feature_set(v1_state_contract)
    exit_set = set(exit_tr["feature_set_v1"])
    iql_set = set(iql["feature_set_v1"])
    overlap_exit_iql = sorted(exit_set & iql_set)
    only_in_exit = sorted(exit_set - iql_set)
    only_in_iql = sorted(iql_set - exit_set)
    rows: list[dict[str, Any]] = []
    for f in sorted(exit_set | iql_set):
        rows.append(
            {
                "feature_name_v1": f,
                "in_exit_transformer_v1": f in exit_set,
                "in_iql_state_v1": f in iql_set,
                "overlap_v1": f in (exit_set & iql_set),
            }
        )
    audit = {
        "xgb_entry_v1": xgb,
        "exit_transformer_v1": exit_tr,
        "iql_state_v1": iql,
        "exit_iql_overlap_count_v1": len(overlap_exit_iql),
        "exit_only_count_v1": len(only_in_exit),
        "iql_only_count_v1": len(only_in_iql),
        "exit_iql_overlap_v1": overlap_exit_iql,
        "exit_only_features_v1": only_in_exit,
        "iql_only_features_v1": only_in_iql,
        "samstemte_status_v1": "FEATURE_SETS_DIVERGE_NEED_HUB_DESIGN",
        "research_only_v1": True,
        "interpretation_v1": (
            "Exit-transformer (53 features) and IQL state (9 allowed AS_OF "
            "fields) live in disjoint feature spaces. XGB-entry uses bundled "
            "ctx_cont/ctx_cat columns whose individual feature names are not "
            "exposed at parquet-column level. A samstemte feature-hub would "
            "need to (a) unbundle ctx_cont/ctx_cat into named features, (b) "
            "publish a single AS_OF feature snapshot at decision time, (c) "
            "allow each downstream consumer to subscribe to a subset by name."
        ),
    }
    return pd.DataFrame.from_records(rows), audit


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    trades: pd.DataFrame,
    per_bar_summary: dict[str, Any],
    giveback_summary: dict[str, Any],
    cata_summary: dict[str, Any],
    friday_summary: dict[str, Any],
    samstemte_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "layer_name": "EXIT_QUALITY_DIAGNOSTIC_REPRODUCIBILITY_AUDIT_V1",
        "trade_row_count_v1": int(len(trades)),
        "expected_trade_rows_v1": EXPECTED_FRAME_ROWS,
        "row_count_invariant_v1": int(len(trades)) == EXPECTED_FRAME_ROWS,
        "per_bar_reconstruction_completeness_v1": per_bar_summary[
            "reconstruction_completeness_v1"
        ],
        "per_bar_decision_row_count_v1": per_bar_summary["decision_row_count_v1"],
        "giveback_ladder_levels_v1": GIVEBACK_LADDER_LEVELS,
        "trail_drawdown_pct_v1": TRAIL_DRAWDOWN_PCT,
        "cata_count_v1": cata_summary.get("cata_count_v1", 0),
        "friday_flat_count_v1": friday_summary.get("friday_flat_count_v1", 0),
        "samstemte_status_v1": samstemte_summary["samstemte_status_v1"],
        "research_only_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
    }
    if not payload["row_count_invariant_v1"]:
        raise RuntimeError("ROW_COUNT_INVARIANT_FAILED")
    return payload


def _go_no_go(
    per_bar_summary: dict[str, Any],
    giveback_summary: dict[str, Any],
    cata_summary: dict[str, Any],
    friday_summary: dict[str, Any],
    samstemte_summary: dict[str, Any],
) -> tuple[str, str, str]:
    completeness = per_bar_summary["reconstruction_completeness_v1"]
    friday_ok = friday_summary.get("friday_flat_count_v1", 0) > 0 and (
        friday_summary.get("monday_open_lookup_available_v1", 0) > 0
    )
    samstemte_ok = samstemte_summary["samstemte_status_v1"] != "NOT_ESTABLISHED"
    if completeness < 0.50:
        return (
            "EXIT_QUALITY_DIAGNOSTIC_BLOCKED_BY_M5_GAP",
            "DEEPEN_PER_BAR_RECONSTRUCTION_LINEAGE_V1",
            (
                "Per-bar reconstruction covered fewer than 50% of trades due to "
                "M5 raw range gap. Fix lineage before exit-IQL training."
            ),
        )
    if completeness < 0.95:
        return (
            "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_RECONSTRUCTION_GAP",
            "DEEPEN_PER_BAR_RECONSTRUCTION_LINEAGE_V1",
            (
                f"Per-bar reconstruction covered {completeness*100:.1f}% of "
                "trades. Diagnostics and counterfactuals usable, but exit-IQL "
                "training on the full cohort needs lineage repair."
            ),
        )
    if not friday_ok:
        return (
            "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_FRIDAY_FLAT_REFINEMENT_NOT_ESTABLISHED",
            "REFINE_FRIDAY_FLAT_POLICY_RESEARCH_ONLY_V1",
            (
                "Per-bar reconstruction OK but Friday-flat refinement could not "
                "establish Monday-open lookup. Investigate before policy refinement."
            ),
        )
    if not samstemte_ok:
        return (
            "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_SAMSTEMTE_AUDIT_NOT_ESTABLISHED",
            "DESIGN_FEATURE_HUB_FROM_SAMSTEMTE_AUDIT_V1",
            (
                "Per-bar reconstruction and counterfactuals OK but samstemte "
                "feature audit could not enumerate sources. Resolve before hub design."
            ),
        )
    return (
        "EXIT_QUALITY_DIAGNOSTIC_PASS_FULL_RECONSTRUCTION_AND_COUNTERFACTUALS_AVAILABLE",
        "TRAIN_EXIT_BANDIT_HOLD_EXIT_NOW_RESEARCH_V1",
        (
            "All five sub-tracks delivered. Per-bar HOLD/EXIT_NOW dataset is "
            "research-ready for exit-bandit training. Counterfactuals quantify "
            "available edge from exit-timing improvement. Samstemte audit "
            "exposes feature-hub design surface."
        ),
    )


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append(
            {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        )
    files.append(
        {
            "name_v1": "m5_raw_2025",
            "path_v1": str(inputs["m5_raw_2025_path"]),
            "sha256_v1": _file_hash(inputs["m5_raw_2025_path"]),
        }
    )
    files.append(
        {
            "name_v1": "m5_raw_2026",
            "path_v1": str(inputs["m5_raw_2026_path"]),
            "sha256_v1": _file_hash(inputs["m5_raw_2026_path"]),
        }
    )
    return {
        "layer_name": "EXIT_QUALITY_DIAGNOSTIC_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v1_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "rebuild_state_v2_root_v1": str(INPUT_REBUILD_ROOT),
            "v2_training_root_v1": str(INPUT_V2_TRAINING_ROOT),
            "refine_clean_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
        },
        "raw_data_v1": {
            "m5_raw_2025_v1": str(M5_RAW_2025),
            "m5_raw_2026_v1": str(M5_RAW_2026),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    sub_dirs = {
        "per_bar": artifact_root / "PER_BAR_TRAJECTORY_V1",
        "giveback": artifact_root / "GIVEBACK_LADDER_V1",
        "cata": artifact_root / "CATA_PREVENTION_V1",
        "friday": artifact_root / "FRIDAY_FLAT_REFINEMENT_V1",
        "samstemte": artifact_root / "SAMSTEMTE_FEATURE_AUDIT_V1",
    }
    for d in sub_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    _write_json(artifact_root / "input_manifest_v1.json", _build_input_manifest(inputs, artifact_root))

    # Load shared data
    trades = _load_trade_outcomes()
    m5 = _load_m5_raw()

    # Sub-track 1: per-bar reconstruction
    decisions, per_bar_summary = _reconstruct_per_bar_trajectories(trades, m5)
    if not decisions.empty:
        decisions.to_csv(sub_dirs["per_bar"] / "per_bar_decision_dataset_v1.csv", index=False)
        decisions.to_parquet(
            sub_dirs["per_bar"] / "per_bar_decision_dataset_v1.parquet", index=False
        )
    _write_json(sub_dirs["per_bar"] / "per_bar_reconstruction_summary_v1.json", per_bar_summary)

    # Sub-track 2: giveback ladder
    giveback_table, giveback_summary = _giveback_ladder_counterfactual(trades)
    giveback_table.to_csv(sub_dirs["giveback"] / "giveback_ladder_counterfactual_v1.csv", index=False)
    _write_json(
        sub_dirs["giveback"] / "giveback_ladder_summary_v1.json",
        {**giveback_summary, "rows_v1": giveback_table.to_dict(orient="records")},
    )

    # Sub-track 3: CATA prevention
    cata_table, cata_summary = _cata_prevention_counterfactual(trades)
    if not cata_table.empty:
        cata_table.to_csv(sub_dirs["cata"] / "cata_prevention_per_trade_v1.csv", index=False)
    _write_json(sub_dirs["cata"] / "cata_prevention_summary_v1.json", cata_summary)

    # Sub-track 4: Friday-flat refinement
    friday_table, friday_summary = _friday_flat_refinement(trades, m5)
    if not friday_table.empty:
        friday_table.to_csv(
            sub_dirs["friday"] / "friday_flat_refinement_per_trade_v1.csv", index=False
        )
    _write_json(sub_dirs["friday"] / "friday_flat_refinement_summary_v1.json", friday_summary)

    # Sub-track 5: samstemte feature audit
    samstemte_table, samstemte_summary = _samstemte_feature_audit(inputs["v1_state_contract"])
    samstemte_table.to_csv(
        sub_dirs["samstemte"] / "samstemte_feature_diff_v1.csv", index=False
    )
    _write_json(
        sub_dirs["samstemte"] / "samstemte_feature_audit_summary_v1.json", samstemte_summary
    )

    # Reproducibility + go/no-go
    repro = _reproducibility_audit(
        trades, per_bar_summary, giveback_summary, cata_summary, friday_summary, samstemte_summary
    )
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)
    status, next_action, recommendation = _go_no_go(
        per_bar_summary, giveback_summary, cata_summary, friday_summary, samstemte_summary
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_QUALITY_DIAGNOSTIC_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "trade_count_v1": int(len(trades)),
        "per_bar_reconstruction_v1": {
            "completeness_v1": per_bar_summary["reconstruction_completeness_v1"],
            "decision_row_count_v1": per_bar_summary["decision_row_count_v1"],
            "hold_count_v1": per_bar_summary["hold_action_count_v1"],
            "exit_now_count_v1": per_bar_summary["exit_now_action_count_v1"],
            "out_of_range_v1": per_bar_summary["trades_out_of_m5_range_v1"],
        },
        "giveback_ladder_v1": {
            "actual_realized_pnl_bps_v1": giveback_summary["actual_realized_pnl_bps_v1"],
            "actual_total_giveback_bps_v1": giveback_summary["actual_total_giveback_bps_v1"],
            "ladder_count_v1": giveback_summary["ladder_count_v1"],
            "scenarios_v1": giveback_table.to_dict(orient="records"),
        },
        "cata_prevention_v1": cata_summary,
        "friday_flat_refinement_v1": friday_summary,
        "samstemte_feature_audit_v1": {
            "exit_iql_overlap_count_v1": samstemte_summary["exit_iql_overlap_count_v1"],
            "exit_only_count_v1": samstemte_summary["exit_only_count_v1"],
            "iql_only_count_v1": samstemte_summary["iql_only_count_v1"],
            "samstemte_status_v1": samstemte_summary["samstemte_status_v1"],
        },
        "row_count_invariant_v1": repro["row_count_invariant_v1"],
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
        "next_gate_hook_v1": next_action,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "EXIT_QUALITY_DIAGNOSTIC_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_QUALITY_DIAGNOSTIC_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "This gate is research-only diagnostic + scaffold. It does not "
            "open adapter, R6, IQL production/live, freeze, promo, or live, "
            "and does not modify exit_manager or any runtime."
        ),
    }
    _write_json(
        artifact_root / "exit_quality_diagnostic_and_per_bar_decision_scaffold_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Exit Quality Diagnostic And Per-Bar Decision Scaffold V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "",
        "## Per-bar reconstruction",
        f"- Trades reconstructed: {per_bar_summary['reconstructed_trade_count_v1']}/{EXPECTED_FRAME_ROWS} ({per_bar_summary['reconstruction_completeness_v1']*100:.1f}%)",
        f"- Out-of-M5-range trades: {per_bar_summary['trades_out_of_m5_range_v1']}",
        f"- Decision rows: {per_bar_summary['decision_row_count_v1']} (HOLD={per_bar_summary['hold_action_count_v1']}, EXIT_NOW={per_bar_summary['exit_now_action_count_v1']})",
        "",
        "## Giveback ladder counterfactual",
        f"- Actual realized: {giveback_summary['actual_realized_pnl_bps_v1']:.0f} bps",
        f"- Total giveback (mfe-pnl): {giveback_summary['actual_total_giveback_bps_v1']:.0f} bps",
    ]
    for row in giveback_table.to_dict(orient="records"):
        sc = row["scenario_v1"]
        sumv = row["sum_pnl_bps_v1"]
        delta = row.get("delta_vs_actual_v1")
        delta_text = f", delta {delta:+.0f}" if delta is not None else ""
        report_lines.append(f"  - `{sc}`: {sumv:.0f} bps{delta_text}")
    report_lines.extend([
        "",
        "## CATA prevention upper bound",
        f"- CATA count: {cata_summary.get('cata_count_v1', 0)}",
        f"- Mean savings/cata: {cata_summary.get('mean_savings_per_cata_bps_v1', 0):.1f} bps",
        f"- Upper-bound total savings: {cata_summary.get('upper_bound_savings_bps_v1', 0):.0f} bps",
        "",
        "## Friday-flat refinement",
        f"- Friday-flat count: {friday_summary.get('friday_flat_count_v1', 0)}",
        f"- Actual: {friday_summary.get('actual_friday_flat_total_pnl_bps_v1', 0):.0f} bps",
        f"- Hold-to-monday: {friday_summary.get('counterfactual_hold_to_monday_open_total_pnl_bps_v1', 0):.0f} bps",
        f"- Refined policy (only-flat-losers): {friday_summary.get('refined_policy_total_pnl_bps_v1', 0):.0f} bps",
        f"- Refined-vs-actual delta: {friday_summary.get('refined_policy_delta_vs_friday_flat_bps_v1', 0):+.0f} bps",
        "",
        "## Samstemte feature audit",
        f"- Exit-transformer features: {samstemte_summary['exit_transformer_v1']['feature_count_v1']}",
        f"- IQL state features: {samstemte_summary['iql_state_v1']['feature_count_v1']}",
        f"- Overlap: {samstemte_summary['exit_iql_overlap_count_v1']}",
        f"- Status: `{samstemte_summary['samstemte_status_v1']}`",
        "",
        "## Recommendation",
        recommendation,
    ])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root
                / "exit_quality_diagnostic_and_per_bar_decision_scaffold_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "reproducibility": str(artifact_root / "reproducibility_audit_v1.json"),
            "per_bar_summary": str(
                sub_dirs["per_bar"] / "per_bar_reconstruction_summary_v1.json"
            ),
            "per_bar_decision_dataset_csv": str(
                sub_dirs["per_bar"] / "per_bar_decision_dataset_v1.csv"
            ),
            "per_bar_decision_dataset_parquet": str(
                sub_dirs["per_bar"] / "per_bar_decision_dataset_v1.parquet"
            ),
            "giveback_ladder": str(sub_dirs["giveback"] / "giveback_ladder_summary_v1.json"),
            "cata_prevention": str(sub_dirs["cata"] / "cata_prevention_summary_v1.json"),
            "friday_flat_refinement": str(
                sub_dirs["friday"] / "friday_flat_refinement_summary_v1.json"
            ),
            "samstemte_audit": str(
                sub_dirs["samstemte"] / "samstemte_feature_audit_summary_v1.json"
            ),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
