#!/usr/bin/env python3
"""Augment the per-bar exit-decision dataset with counterfactual EXIT_NOW samples.

This is gate 3 of 6 in the exit-IQL pre-train dependency graph. It produces
the actual training-ready offline-RL dataset by:

  - Loading the per-bar decision dataset (gate 0: 169260 rows, 1724 trades)
  - Joining the locked state-feature schema (gate 2: 18 HAVE features) from
    per_bar scaffold + EXIT_EVAL_TRACE + BASE34_M5 + TRADE_OUTCOMES
  - Generating two action samples per bar: (HOLD, EXIT_NOW) - so the dataset
    finally has true action support at every bar, not just one logged action
  - Computing the five trainable terminal reward variants per (bar, action)
  - Adding behavior-policy propensity labels distinguishing logged samples
    from counterfactual augmentation
  - Adding next_row pointers for HOLD non-terminal Bellman backup
  - Audit no-shortcut against the 29 forbidden state fields locked in gate 1
  - Audit join coverage per source

This gate does NOT train any model. It produces only the augmented dataset
plus audits. Training remains BLOCKED until the remaining three pre-train
gates pass.
"""
from __future__ import annotations

import argparse
import json
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
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate
from gx1.scripts import materialize_exit_per_bar_state_feature_contract_v1 as state_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_ACTION_SUPPORT_AUGMENT_V1"

INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)
INPUT_STATE_FEATURE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T113745Z_LOCK"
)
INPUT_PER_BAR_SCAFFOLD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T100845Z_LOCK"
)

BASE34_M5_FEATURES_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/"
    "monday_week_prebuilt_extension_20260423_145325/"
    "xauusd_m5_BASE34_20250101_20260420_MODEL_BARS.parquet"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1

REWARD_VARIANT_IDS = [
    "REALIZED_PNL_REWARD",
    "MFE_CAPTURE_REWARD",
    "MAE_PENALTY_REWARD",
    "GIVEBACK_PENALTY_REWARD",
    "TRANSPARENT_COMBINED_REWARD",
]

ALLOWED_FINAL_STATUSES = {
    "EXIT_ACTION_SUPPORT_AUGMENT_LOCKED_DATASET_READY",
    "EXIT_ACTION_SUPPORT_AUGMENT_PARTIAL_JOIN_COVERAGE_GAP",
    "EXIT_ACTION_SUPPORT_AUGMENT_BLOCKED_BY_NO_SHORTCUT_FAIL",
    "EXIT_ACTION_SUPPORT_AUGMENT_BLOCKED_BY_DATASET_INTEGRITY",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1",
    "DEEPEN_PER_BAR_TRACE_JOIN_LINEAGE_V1",
    "HOLD_UNTIL_AUGMENTATION_GAPS_RESOLVED_V1",
}


# Reuse helpers
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


def validate_state_no_shortcut(state_columns: Sequence[str]) -> bool:
    forbidden = set(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)
    hits = sorted(set(state_columns) & forbidden)
    if hits:
        raise RuntimeError(f"FORBIDDEN_STATE_FIELD_IN_AUGMENTED_COLUMNS: {hits}")
    return True


def validate_action_distribution(actions: pd.Series) -> bool:
    counts = actions.value_counts().to_dict()
    if set(counts.keys()) != {ACTION_HOLD_ID, ACTION_EXIT_NOW_ID}:
        raise RuntimeError(f"ACTION_VALUES_NOT_BINARY: {counts}")
    if counts[ACTION_HOLD_ID] != counts[ACTION_EXIT_NOW_ID]:
        raise RuntimeError(
            f"AUGMENTED_ACTION_COUNTS_MISMATCH: HOLD={counts[ACTION_HOLD_ID]} EXIT_NOW={counts[ACTION_EXIT_NOW_ID]}"
        )
    return True


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _per_bar_decision_dataset_path() -> Path:
    return (
        INPUT_PER_BAR_SCAFFOLD_ROOT
        / "PER_BAR_TRAJECTORY_V1"
        / "per_bar_decision_dataset_v1.parquet"
    )


def _load_per_bar() -> pd.DataFrame:
    df = pd.read_parquet(_per_bar_decision_dataset_path())
    df = df.copy()
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["trade_uid_v1"] = df["trade_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    return df


def _load_trade_outcomes() -> pd.DataFrame:
    weeks = sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "TRUTH_MONFRI_WEEK_*/trade_outcomes_*_MERGED.parquet"
        ),
        key=lambda p: p.parent.name,
    )
    frames = []
    for w in weeks:
        df = pd.read_parquet(w)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("NO_TRADE_OUTCOMES_FOUND")
    common = list(frames[0].columns)
    for f in frames[1:]:
        common = [c for c in common if c in f.columns]
    aligned = [f.loc[:, common].copy() for f in frames]
    out = pd.concat(aligned, ignore_index=True)
    out["candidate_uid"] = out["candidate_uid"].astype(str)
    out["trade_id"] = out["trade_id"].astype(str)
    return out


def _load_exit_eval_trace() -> pd.DataFrame:
    paths = sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "TRUTH_MONFRI_WEEK_*/replay/chunk_0/EXIT_EVAL_TRACE.csv"
        ),
        key=lambda p: p.parent.parent.parent.name,
    )
    keep_cols = [
        "trade_id",
        "timestamp",
        "exit_prob",
        "distance_from_peak_mfe_bps",
        "time_since_mfe_bars",
        "giveback_ratio",
    ]
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in keep_cols)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=keep_cols)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["trade_id"] = out["trade_id"].astype(str)
    return out


def _load_base34() -> pd.DataFrame:
    df = pd.read_parquet(BASE34_M5_FEATURES_PATH)
    if "time" not in df.columns:
        df = df.reset_index()
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    keep_cols = [
        "time",
        "atr_bps",
        "session_id",
        "_v1_atr_regime_id",
        "_v1_close_ema_slope_3",
        "_v1_cost_bps_dyn",
        "minutes_since_session_open",
    ]
    available = [c for c in keep_cols if c in df.columns]
    return df.loc[:, available].copy()


# ---------------------------------------------------------------------------
# State vector materialization
# ---------------------------------------------------------------------------


def _build_state_matrix(
    per_bar: pd.DataFrame,
    trade_outcomes: pd.DataFrame,
    exit_trace: pd.DataFrame,
    base34: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols_to_keep = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "bar_index_v1",
        "bar_count_v1",
        "is_terminal_v1",
        "side_v1",
        "ts_v1",
        "entry_price_v1",
        "bar_close_v1",
        "pnl_at_close_bps_v1",
        "running_mfe_bps_v1",
        "running_mae_bps_v1",
        "running_giveback_from_peak_bps_v1",
    ]
    available = [c for c in cols_to_keep if c in per_bar.columns]
    state = per_bar.loc[:, available].copy()
    state.rename(
        columns={
            "pnl_at_close_bps_v1": "running_pnl_at_close_bps_v1",
            "bar_index_v1": "bars_held_v1",
        },
        inplace=True,
    )

    # Join trade_outcomes for trade_id and entry-context fields
    to_subset = trade_outcomes.loc[
        :,
        [
            "candidate_uid",
            "trade_id",
            "session",
            "entry_spread_bps",
        ],
    ].copy()
    to_subset.rename(
        columns={
            "candidate_uid": "candidate_uid_v1",
            "session": "entry_session_v1",
            "entry_spread_bps": "entry_spread_bps_v1",
        },
        inplace=True,
    )
    state = state.merge(to_subset, on="candidate_uid_v1", how="left")
    trade_id_match = state["trade_id"].notna().sum()
    state["trade_id"] = state["trade_id"].fillna("").astype(str)

    # Join EXIT_EVAL_TRACE on (trade_id, timestamp) using merge_asof to handle
    # M1-vs-M5 timestamp resolution mismatch. We use direction='nearest' with
    # tolerance 3 minutes so each per-bar row gets the closest trace entry
    # within the same trade_id, capped at half an M5 interval.
    if not exit_trace.empty:
        trace = exit_trace.loc[
            :,
            [
                "trade_id",
                "timestamp",
                "exit_prob",
                "distance_from_peak_mfe_bps",
                "time_since_mfe_bars",
                "giveback_ratio",
            ],
        ].copy()
        trace.rename(
            columns={
                "timestamp": "ts_v1",
                "exit_prob": "exit_prob_v1",
                "distance_from_peak_mfe_bps": "distance_from_peak_mfe_bps_v1",
                "time_since_mfe_bars": "time_since_mfe_bars_v1",
                "giveback_ratio": "giveback_ratio_v1",
            },
            inplace=True,
        )
        # merge_asof requires both sides sorted by the on-key globally
        state_for_merge = state.sort_values("ts_v1").reset_index(drop=True)
        trace_for_merge = trace.sort_values("ts_v1").reset_index(drop=True)
        state = pd.merge_asof(
            state_for_merge,
            trace_for_merge,
            on="ts_v1",
            by="trade_id",
            direction="nearest",
            tolerance=pd.Timedelta("3min"),
        )
    else:
        for c in [
            "exit_prob_v1",
            "distance_from_peak_mfe_bps_v1",
            "time_since_mfe_bars_v1",
            "giveback_ratio_v1",
        ]:
            state[c] = np.nan

    # Join BASE34 on ts using merge_asof with backward direction so per-bar gets
    # the most recent BASE34 feature snapshot at or before bar t (the realistic
    # AS_OF semantics: features observed at or before decision time). 2026 raw
    # M5 has different minute-alignment than BASE34, so exact-match misses ~27%.
    if not base34.empty:
        b34 = base34.copy()
        b34.rename(columns={"time": "ts_v1"}, inplace=True)
        b34.rename(
            columns={
                "atr_bps": "atr_bps_now_v1",
                "session_id": "session_id_v1",
                "_v1_atr_regime_id": "vol_regime_id_v1",
                "_v1_close_ema_slope_3": "trend_slope_ema3_v1",
                "_v1_cost_bps_dyn": "spread_bps_dyn_v1",
                "minutes_since_session_open": "minutes_since_session_open_v1",
            },
            inplace=True,
        )
        state_sorted = state.sort_values("ts_v1").reset_index(drop=True)
        b34_sorted = b34.sort_values("ts_v1").reset_index(drop=True)
        state = pd.merge_asof(
            state_sorted,
            b34_sorted,
            on="ts_v1",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
    else:
        for c in [
            "atr_bps_now_v1",
            "session_id_v1",
            "vol_regime_id_v1",
            "trend_slope_ema3_v1",
            "spread_bps_dyn_v1",
            "minutes_since_session_open_v1",
        ]:
            state[c] = np.nan

    coverage = {
        "trade_id_match_count_v1": int(trade_id_match),
        "trade_id_match_rate_v1": float(trade_id_match / max(len(state), 1)),
        "exit_prob_present_count_v1": int(state["exit_prob_v1"].notna().sum())
        if "exit_prob_v1" in state.columns
        else 0,
        "exit_prob_present_rate_v1": float(state["exit_prob_v1"].notna().sum() / max(len(state), 1))
        if "exit_prob_v1" in state.columns
        else 0.0,
        "atr_bps_now_present_count_v1": int(state["atr_bps_now_v1"].notna().sum())
        if "atr_bps_now_v1" in state.columns
        else 0,
        "atr_bps_now_present_rate_v1": float(
            state["atr_bps_now_v1"].notna().sum() / max(len(state), 1)
        )
        if "atr_bps_now_v1" in state.columns
        else 0.0,
        "session_id_present_count_v1": int(state["session_id_v1"].notna().sum())
        if "session_id_v1" in state.columns
        else 0,
    }
    return state, coverage


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def _compute_terminal_reward_at_bar(
    pnl_at_close_bps: pd.Series,
    running_mfe_bps: pd.Series,
    running_mae_bps: pd.Series,
    variant: str,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    pnl = pnl_at_close_bps.astype(float).to_numpy()
    mfe = running_mfe_bps.astype(float).to_numpy()
    mae = running_mae_bps.astype(float).to_numpy()
    if variant == "REALIZED_PNL_REWARD":
        return pnl.copy()
    if variant == "MFE_CAPTURE_REWARD":
        return np.clip(pnl / np.maximum(mfe, eps), -2.0, 2.0)
    if variant == "MAE_PENALTY_REWARD":
        return pnl - 0.5 * np.abs(mae)
    if variant == "GIVEBACK_PENALTY_REWARD":
        return -np.maximum(mfe - pnl, 0.0)
    if variant == "TRANSPARENT_COMBINED_REWARD":
        return pnl - 0.25 * np.abs(mae) - 0.25 * np.maximum(mfe - pnl, 0.0)
    raise RuntimeError(f"UNKNOWN_REWARD_VARIANT: {variant}")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _augment_with_action_pairs(state: pd.DataFrame) -> pd.DataFrame:
    # For each row, emit one HOLD sample and one EXIT_NOW sample.
    # State columns are duplicated; action and reward differ.
    n = len(state)
    if n == 0:
        return state.copy()
    state = state.reset_index(drop=True).copy()
    state["row_id_per_bar_v1"] = np.arange(n)

    # Compute terminal rewards once per bar (used by EXIT_NOW always, and by
    # HOLD only when realized exit bar = forced terminal).
    terminal_rewards: dict[str, np.ndarray] = {}
    for variant in REWARD_VARIANT_IDS:
        terminal_rewards[variant] = _compute_terminal_reward_at_bar(
            state["running_pnl_at_close_bps_v1"],
            state["running_mfe_bps_v1"],
            state["running_mae_bps_v1"],
            variant,
        )

    # HOLD samples
    hold = state.copy()
    hold["action_id_v1"] = ACTION_HOLD_ID
    hold["action_label_v1"] = "HOLD"
    is_realized_terminal = state["is_terminal_v1"].astype(bool).to_numpy()
    for variant in REWARD_VARIANT_IDS:
        col = f"reward_{variant.lower()}_v1"
        # HOLD reward: 0 immediate at non-terminal; terminal reward at realized exit (forced terminal hold)
        hold[col] = np.where(is_realized_terminal, terminal_rewards[variant], 0.0)
    hold["is_terminal_for_action_v1"] = is_realized_terminal
    hold["behavior_propensity_v1"] = np.where(
        is_realized_terminal, "FORCED_TERMINAL_HOLD_DATA_LIMIT", "LOGGED_HOLD_PROPENSITY_1"
    )

    # EXIT_NOW samples
    exit_now = state.copy()
    exit_now["action_id_v1"] = ACTION_EXIT_NOW_ID
    exit_now["action_label_v1"] = "EXIT_NOW"
    for variant in REWARD_VARIANT_IDS:
        col = f"reward_{variant.lower()}_v1"
        # EXIT_NOW reward: terminal reward at this bar's close, always
        exit_now[col] = terminal_rewards[variant]
    exit_now["is_terminal_for_action_v1"] = True
    exit_now["behavior_propensity_v1"] = np.where(
        is_realized_terminal,
        "LOGGED_EXIT_NOW_PROPENSITY_1",
        "COUNTERFACTUAL_EXIT_NOW_NO_PROPENSITY",
    )

    augmented = pd.concat([hold, exit_now], ignore_index=True)
    augmented = augmented.sort_values(
        ["candidate_uid_v1", "bars_held_v1", "action_id_v1"]
    ).reset_index(drop=True)
    return augmented


def _add_next_state_pointers(state: pd.DataFrame, augmented: pd.DataFrame) -> pd.DataFrame:
    # For each HOLD sample at non-terminal bar, next_state pointer is the next bar's row_id.
    state = state.copy()
    state["row_id_per_bar_v1"] = np.arange(len(state))
    next_row_id = state.groupby("candidate_uid_v1")["row_id_per_bar_v1"].shift(-1)
    state["next_row_id_per_bar_v1"] = next_row_id
    pointer_map = state.set_index("row_id_per_bar_v1")["next_row_id_per_bar_v1"].to_dict()
    augmented = augmented.copy()
    augmented["next_row_id_per_bar_v1"] = augmented["row_id_per_bar_v1"].map(pointer_map)
    is_hold = augmented["action_id_v1"] == ACTION_HOLD_ID
    is_terminal_action = augmented["is_terminal_for_action_v1"].astype(bool)
    # EXIT_NOW always has no next state. HOLD at terminal also has no next state.
    augmented.loc[~is_hold | is_terminal_action, "next_row_id_per_bar_v1"] = np.nan
    return augmented


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _no_shortcut_audit(augmented: pd.DataFrame) -> dict[str, Any]:
    state_columns = [
        c for c in augmented.columns if c not in {
            "action_id_v1",
            "action_label_v1",
            "behavior_propensity_v1",
            "is_terminal_for_action_v1",
            "row_id_per_bar_v1",
            "next_row_id_per_bar_v1",
            "trade_id",
            "candidate_uid_v1",
            "trade_uid_v1",
            "ts_v1",
            "entry_price_v1",
            "bar_close_v1",
            "is_terminal_v1",
            "bar_count_v1",
        }
        and not c.startswith("reward_")
    ]
    validate_state_no_shortcut(state_columns)
    return {
        "layer_name": "EXIT_ACTION_SUPPORT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "checked_state_columns_v1": state_columns,
        "checked_count_v1": len(state_columns),
        "forbidden_intersection_v1": [],
    }


def _action_balance_audit(augmented: pd.DataFrame) -> dict[str, Any]:
    counts = augmented["action_id_v1"].value_counts().to_dict()
    propensity_counts = (
        augmented["behavior_propensity_v1"].value_counts().to_dict()
    )
    validate_action_distribution(augmented["action_id_v1"])
    return {
        "layer_name": "EXIT_ACTION_SUPPORT_BALANCE_AUDIT_V1",
        "status_v1": "PASS",
        "action_counts_v1": {str(k): int(v) for k, v in counts.items()},
        "propensity_counts_v1": {str(k): int(v) for k, v in propensity_counts.items()},
        "augmented_row_count_v1": int(len(augmented)),
    }


def _reward_distribution_audit(augmented: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for variant in REWARD_VARIANT_IDS:
        col = f"reward_{variant.lower()}_v1"
        for action_label, action_id in [("HOLD", ACTION_HOLD_ID), ("EXIT_NOW", ACTION_EXIT_NOW_ID)]:
            mask = augmented["action_id_v1"] == action_id
            series = augmented.loc[mask, col].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "reward_variant_v1": variant,
                    "action_v1": action_label,
                    "row_count_v1": int(len(series)),
                    "mean_v1": float(series.mean()),
                    "std_v1": float(series.std(ddof=0)),
                    "p5_v1": float(series.quantile(0.05)),
                    "p50_v1": float(series.quantile(0.50)),
                    "p95_v1": float(series.quantile(0.95)),
                    "min_v1": float(series.min()),
                    "max_v1": float(series.max()),
                    "zero_count_v1": int((series == 0.0).sum()),
                }
            )
    return rows


def _terminal_consistency_audit(augmented: pd.DataFrame) -> dict[str, Any]:
    is_hold = augmented["action_id_v1"] == ACTION_HOLD_ID
    is_terminal_for_action = augmented["is_terminal_for_action_v1"].astype(bool)
    has_next = augmented["next_row_id_per_bar_v1"].notna()
    # EXIT_NOW must always be terminal-for-action
    exit_now_non_terminal = int(((~is_hold) & (~is_terminal_for_action)).sum())
    # Terminal-for-action must never have a next pointer
    terminal_with_next = int((is_terminal_for_action & has_next).sum())
    # HOLD at non-terminal-bar must have a next pointer (unless next bar absent due to last bar)
    audit = {
        "layer_name": "EXIT_ACTION_SUPPORT_TERMINAL_CONSISTENCY_AUDIT_V1",
        "status_v1": "PASS",
        "exit_now_non_terminal_count_v1": exit_now_non_terminal,
        "terminal_with_next_pointer_count_v1": terminal_with_next,
    }
    if exit_now_non_terminal > 0 or terminal_with_next > 0:
        audit["status_v1"] = "FAIL"
        raise RuntimeError(f"TERMINAL_CONSISTENCY_FAIL: {audit}")
    return audit


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_MDP_ROOT,
        INPUT_STATE_FEATURE_ROOT,
        INPUT_PER_BAR_SCAFFOLD_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "mdp_summary": INPUT_MDP_ROOT / "summary_v1.json",
        "state_feature_contract": INPUT_STATE_FEATURE_ROOT / "state_feature_contract_v1.json",
        "state_feature_summary": INPUT_STATE_FEATURE_ROOT / "summary_v1.json",
        "per_bar_summary": INPUT_PER_BAR_SCAFFOLD_ROOT / "summary_v1.json",
        "per_bar_decision_dataset": _per_bar_decision_dataset_path(),
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(f"BASE34_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}")
    return {
        "required_paths": required,
        "base34_path": BASE34_M5_FEATURES_PATH,
        "state_feature_contract": _read_json(required["state_feature_contract"]),
        "state_feature_summary": _read_json(required["state_feature_summary"]),
        "mdp_summary": _read_json(required["mdp_summary"]),
        "per_bar_summary": _read_json(required["per_bar_summary"]),
    }


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    files.append(
        {
            "name_v1": "base34_m5_features",
            "path_v1": str(inputs["base34_path"]),
            "sha256_v1": _file_hash(inputs["base34_path"]),
        }
    )
    return {
        "layer_name": "EXIT_ACTION_SUPPORT_AUGMENT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "mdp_root_v1": str(INPUT_MDP_ROOT),
            "state_feature_root_v1": str(INPUT_STATE_FEATURE_ROOT),
            "per_bar_scaffold_root_v1": str(INPUT_PER_BAR_SCAFFOLD_ROOT),
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


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    per_bar_n: int,
    augmented_n: int,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "layer_name": "EXIT_ACTION_SUPPORT_AUGMENT_REPRODUCIBILITY_AUDIT_V1",
        "per_bar_row_count_v1": per_bar_n,
        "augmented_row_count_v1": augmented_n,
        "augmentation_factor_v1": augmented_n / max(per_bar_n, 1),
        "trade_id_match_rate_v1": coverage["trade_id_match_rate_v1"],
        "exit_prob_present_rate_v1": coverage["exit_prob_present_rate_v1"],
        "atr_bps_now_present_rate_v1": coverage["atr_bps_now_present_rate_v1"],
        "research_only_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
    }
    if augmented_n != per_bar_n * 2:
        raise RuntimeError("AUGMENTATION_FACTOR_NOT_2X")
    return payload


def _go_no_go(coverage: dict[str, Any]) -> tuple[str, str, str]:
    trade_match = coverage["trade_id_match_rate_v1"]
    exit_prob = coverage["exit_prob_present_rate_v1"]
    atr = coverage["atr_bps_now_present_rate_v1"]
    if trade_match < 0.95:
        return (
            "EXIT_ACTION_SUPPORT_AUGMENT_PARTIAL_JOIN_COVERAGE_GAP",
            "DEEPEN_PER_BAR_TRACE_JOIN_LINEAGE_V1",
            (
                f"trade_id match rate is {trade_match:.3f} (< 0.95). "
                "Augmented dataset is materialized but state features from "
                "EXIT_EVAL_TRACE may be sparse. Repair lineage before split-and-leakage gate."
            ),
        )
    # BASE34 features should reach >= 0.95 (M5 market features are dense). Below
    # that signals a real lineage gap. exit_prob is sparse by design because the
    # exit transformer doesn't evaluate every bar - it logs only at evaluation
    # events. We accept exit_prob coverage >= 0.30 as 'sparse but expected', and
    # treat NaN exit_prob as missing-data to mask in training.
    if atr < 0.95:
        return (
            "EXIT_ACTION_SUPPORT_AUGMENT_PARTIAL_JOIN_COVERAGE_GAP",
            "DEEPEN_PER_BAR_TRACE_JOIN_LINEAGE_V1",
            (
                f"trade_id match OK ({trade_match:.3f}) but BASE34 market-state "
                f"coverage is {atr:.3f} (< 0.95). Repair source-time joining "
                "before next gate."
            ),
        )
    if exit_prob < 0.30:
        return (
            "EXIT_ACTION_SUPPORT_AUGMENT_PARTIAL_JOIN_COVERAGE_GAP",
            "DEEPEN_PER_BAR_TRACE_JOIN_LINEAGE_V1",
            (
                f"BASE34 coverage OK ({atr:.3f}) but exit_prob coverage is "
                f"{exit_prob:.3f} (< 0.30). Investigate trace-logging cadence "
                "before treating exit_prob as a primary state feature."
            ),
        )
    return (
        "EXIT_ACTION_SUPPORT_AUGMENT_LOCKED_DATASET_READY",
        "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1",
        (
            f"Augmented dataset locked: 2x action samples per bar, trade_id "
            f"match {trade_match:.3f}, BASE34 coverage {atr:.3f}, exit_prob "
            f"coverage {exit_prob:.3f} (sparse-by-design - the exit "
            "transformer logs only at evaluation events, not every bar). "
            "Ready for split-and-leakage audit gate. NaN exit_prob entries "
            "must be masked or imputed at training time."
        ),
    )


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------


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

    # Load all sources
    per_bar = _load_per_bar()
    trade_outcomes = _load_trade_outcomes()
    exit_trace = _load_exit_eval_trace()
    base34 = _load_base34()

    # Build state matrix and augment
    state_matrix, coverage = _build_state_matrix(per_bar, trade_outcomes, exit_trace, base34)
    augmented = _augment_with_action_pairs(state_matrix)
    augmented = _add_next_state_pointers(state_matrix, augmented)

    # Audits
    no_shortcut = _no_shortcut_audit(augmented)
    _write_json(artifact_root / "no_shortcut_audit_v1.json", no_shortcut)
    balance = _action_balance_audit(augmented)
    _write_json(artifact_root / "action_balance_audit_v1.json", balance)
    reward_dist = _reward_distribution_audit(augmented)
    _write_rows(artifact_root / "reward_distribution_audit_v1.csv", reward_dist)
    _write_json(
        artifact_root / "reward_distribution_audit_v1.json",
        {"row_count_v1": len(reward_dist), "rows_v1": reward_dist},
    )
    terminal_consistency = _terminal_consistency_audit(augmented)
    _write_json(
        artifact_root / "terminal_consistency_audit_v1.json", terminal_consistency
    )
    _write_json(artifact_root / "join_coverage_audit_v1.json", coverage)

    # Persist augmented dataset
    augmented_path_parquet = artifact_root / "augmented_per_bar_action_dataset_v1.parquet"
    # Drop intermediate columns the next gate doesn't need (keep stripped state vector + reward + meta)
    augmented_to_save = augmented.copy()
    if "is_terminal_v1" in augmented_to_save.columns:
        # is_terminal_v1 reflects the realized exit bar, used only for diagnostic. Drop from saved
        # dataset to prevent accidental leakage in downstream code (it also overlaps with bar_count_v1
        # signal-flow). Keep is_terminal_for_action_v1 instead which is action-level terminal.
        augmented_to_save = augmented_to_save.drop(columns=["is_terminal_v1"])
    if "bar_count_v1" in augmented_to_save.columns:
        augmented_to_save = augmented_to_save.drop(columns=["bar_count_v1"])
    augmented_to_save.to_parquet(augmented_path_parquet, index=False)

    # Reproducibility + go/no-go
    repro = _reproducibility_audit(
        per_bar_n=int(len(per_bar)),
        augmented_n=int(len(augmented)),
        coverage=coverage,
    )
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)
    status, next_action, recommendation = _go_no_go(coverage)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_ACTION_SUPPORT_AUGMENT_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "per_bar_row_count_v1": int(len(per_bar)),
        "augmented_row_count_v1": int(len(augmented)),
        "augmentation_factor_v1": float(len(augmented) / max(len(per_bar), 1)),
        "action_counts_v1": balance["action_counts_v1"],
        "propensity_counts_v1": balance["propensity_counts_v1"],
        "trade_id_match_rate_v1": coverage["trade_id_match_rate_v1"],
        "exit_prob_present_rate_v1": coverage["exit_prob_present_rate_v1"],
        "atr_bps_now_present_rate_v1": coverage["atr_bps_now_present_rate_v1"],
        "no_shortcut_audit_status_v1": no_shortcut["status_v1"],
        "terminal_consistency_audit_status_v1": terminal_consistency["status_v1"],
        "reward_variant_count_v1": len(REWARD_VARIANT_IDS),
        "augmented_dataset_path_v1": str(augmented_path_parquet),
        "augmented_dataset_columns_v1": list(augmented_to_save.columns),
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "training_blocked_v1": True,
        "next_pre_train_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "EXIT_ACTION_SUPPORT_AUGMENT_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_ACTION_SUPPORT_AUGMENT_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only augmentation. The augmented dataset is locked and "
            "ready for the split-and-leakage gate. Training remains BLOCKED."
        ),
    }
    _write_json(
        artifact_root / "exit_action_support_augment_go_no_go_v1.json", go_no_go
    )

    report_lines = [
        "# Exit Action Support Augmentation V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** until remaining three pre-train gates pass.",
        "",
        "## Augmentation summary",
        f"- Per-bar rows: {len(per_bar)}",
        f"- Augmented rows (HOLD + EXIT_NOW): {len(augmented)} (= 2x)",
        f"- Reward variants computed per (bar, action): {len(REWARD_VARIANT_IDS)}",
        "",
        "## Action balance",
    ]
    for k, v in balance["action_counts_v1"].items():
        report_lines.append(f"- action_id `{k}`: {v}")
    report_lines.extend([
        "",
        "## Propensity labels",
    ])
    for k, v in balance["propensity_counts_v1"].items():
        report_lines.append(f"- `{k}`: {v}")
    report_lines.extend([
        "",
        "## Join coverage",
        f"- trade_id match rate: {coverage['trade_id_match_rate_v1']:.4f}",
        f"- exit_prob coverage: {coverage['exit_prob_present_rate_v1']:.4f}",
        f"- atr_bps_now coverage: {coverage['atr_bps_now_present_rate_v1']:.4f}",
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
                artifact_root / "exit_action_support_augment_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "no_shortcut_audit": str(artifact_root / "no_shortcut_audit_v1.json"),
            "action_balance_audit": str(
                artifact_root / "action_balance_audit_v1.json"
            ),
            "reward_distribution_audit": str(
                artifact_root / "reward_distribution_audit_v1.json"
            ),
            "terminal_consistency_audit": str(
                artifact_root / "terminal_consistency_audit_v1.json"
            ),
            "join_coverage_audit": str(
                artifact_root / "join_coverage_audit_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "augmented_dataset_parquet": str(augmented_path_parquet),
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
        description="Materialize EXIT_ACTION_SUPPORT_AUGMENT_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
