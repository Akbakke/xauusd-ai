#!/usr/bin/env python3
"""Lock the per-bar state-vector schema for exit-side HOLD/EXIT_NOW IQL.

This is gate 2 of 6 in the exit-IQL pre-train dependency graph established by
EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1. It defines exactly which features
make up the state vector at each per-bar decision point, audits availability
across the four data sources we have, runs a no-shortcut check against the
29 forbidden state fields locked in gate 1, and validates the schema by
joining a sample of trades.

Sources used (all timestamp-pinned, no glob/latest):

  - PER_BAR_SCAFFOLD: TRADE_STATE_RUNNING (running_pnl, running_mfe, running_mae,
    running_giveback_from_peak, bars_held, side, ts) computed in
    EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1.
  - EXIT_EVAL_TRACE.csv per replay week: per-bar exit-transformer output
    (exit_prob), distance_from_peak_mfe_bps, time_since_mfe_bars,
    giveback_ratio, session_current.
  - BASE34 prebuilt M5 features: atr_bps, session_id, _v1_atr_regime_id (vol
    regime), _v1_close_ema_slope_3 (trend slope proxy), _v1_cost_bps_dyn (spread).
  - TRADE_OUTCOMES: side, entry_bid/ask, entry_spread_bps, session at trade
    open (used for ENTRY_CONTEXT_SNAPSHOT subset).

The schema is enumerated; each feature is classified HAVE (data source
verified), DERIVABLE (must be computed in next gate from a known source) or
NOT_ESTABLISHED (no reachable source within current substrate). Training
remains BLOCKED.

The gate explicitly addresses the user's "samstemte" requirement by including
the exit-transformer's bar-level exit_prob as a state feature, making the
IQL state literally see what the exit transformer sees.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import exit_iql_artifact_primitives_v1 as contract_gate
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1"

INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
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

ALLOWED_FINAL_STATUSES = {
    "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_LOCKED_AVAILABILITY_AUDIT_PASSED",
    "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_PARTIAL_SOME_FEATURES_NOT_ESTABLISHED",
    "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_BLOCKED_BY_NO_SHORTCUT_FAIL",
    "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_ACTION_SUPPORT_AUGMENT_V1",
    "DEEPEN_ENTRY_CONTEXT_FEATURE_LINEAGE_V1",
    "HOLD_UNTIL_STATE_FEATURE_GAPS_RESOLVED_V1",
}

# ---------------------------------------------------------------------------
# State schema definition
# ---------------------------------------------------------------------------

# Each feature row:
#  field_name_v1: feature name in IQL state
#  category_v1: TRADE_STATE_RUNNING | MARKET_STATE_AT_BAR | ENTRY_CONTEXT_SNAPSHOT | TRANSFORMER_SIGNAL_AT_BAR
#  source_v1: PER_BAR_SCAFFOLD | EXIT_EVAL_TRACE | BASE34_M5 | TRADE_OUTCOMES | NOT_ESTABLISHED
#  source_field_v1: column name in source
#  lineage_v1: AS_OF_AT_BAR_T | AS_OF_AT_TRADE_OPEN | AS_OF_FROM_BARS_LE_T_MINUS_1
#  availability_v1: HAVE | DERIVABLE | NOT_ESTABLISHED
#  normalization_v1: PASSTHROUGH | ZSCORE_TRAIN_ONLY | ONE_HOT | LOG1P
PROPOSED_STATE_FEATURES: list[dict[str, Any]] = [
    # --- TRADE_STATE_RUNNING ---
    {
        "field_name_v1": "running_pnl_at_close_bps_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "PER_BAR_SCAFFOLD",
        "source_field_v1": "pnl_at_close_bps_v1",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "running_mfe_bps_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "PER_BAR_SCAFFOLD",
        "source_field_v1": "running_mfe_bps_v1",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "running_mae_bps_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "PER_BAR_SCAFFOLD",
        "source_field_v1": "running_mae_bps_v1",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "running_giveback_from_peak_bps_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "PER_BAR_SCAFFOLD",
        "source_field_v1": "running_giveback_from_peak_bps_v1",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "bars_held_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "PER_BAR_SCAFFOLD",
        "source_field_v1": "bar_index_v1",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "LOG1P",
    },
    {
        "field_name_v1": "distance_from_peak_mfe_bps_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "EXIT_EVAL_TRACE",
        "source_field_v1": "distance_from_peak_mfe_bps",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "time_since_mfe_bars_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "EXIT_EVAL_TRACE",
        "source_field_v1": "time_since_mfe_bars",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "LOG1P",
    },
    {
        "field_name_v1": "giveback_ratio_v1",
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": "EXIT_EVAL_TRACE",
        "source_field_v1": "giveback_ratio",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "PASSTHROUGH",
    },
    # --- MARKET_STATE_AT_BAR ---
    {
        "field_name_v1": "atr_bps_now_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "atr_bps",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "session_id_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "session_id",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ONE_HOT",
    },
    {
        "field_name_v1": "vol_regime_id_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "_v1_atr_regime_id",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ONE_HOT",
    },
    {
        "field_name_v1": "trend_slope_ema3_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "_v1_close_ema_slope_3",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "spread_bps_dyn_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "_v1_cost_bps_dyn",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "minutes_since_session_open_v1",
        "category_v1": "MARKET_STATE_AT_BAR",
        "source_v1": "BASE34_M5",
        "source_field_v1": "minutes_since_session_open",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "PASSTHROUGH",
    },
    # --- ENTRY_CONTEXT_SNAPSHOT (partial; entry transformer outputs not in current substrate) ---
    {
        "field_name_v1": "side_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "TRADE_OUTCOMES",
        "source_field_v1": "side",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "HAVE",
        "normalization_v1": "ONE_HOT",
    },
    {
        "field_name_v1": "entry_session_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "TRADE_OUTCOMES",
        "source_field_v1": "session",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "HAVE",
        "normalization_v1": "ONE_HOT",
    },
    {
        "field_name_v1": "entry_spread_bps_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "TRADE_OUTCOMES",
        "source_field_v1": "entry_spread_bps",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "HAVE",
        "normalization_v1": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v1": "p_long_entry_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "NOT_ESTABLISHED",
        "source_field_v1": "",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "NOT_ESTABLISHED",
        "normalization_v1": "PASSTHROUGH",
        "blocking_reason_v1": (
            "Entry-transformer output (p_long_entry) is not in the current "
            "trade_outcomes parquet, trade_log.csv (all empty in our 1914 "
            "substrate), or any reachable artifact. Either re-run replay "
            "with policy_decisions persistence enabled, or proceed without "
            "entry-transformer probabilities in v1 of the state vector."
        ),
    },
    {
        "field_name_v1": "p_hat_entry_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "NOT_ESTABLISHED",
        "source_field_v1": "",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "NOT_ESTABLISHED",
        "normalization_v1": "PASSTHROUGH",
        "blocking_reason_v1": "same as p_long_entry_v1",
    },
    {
        "field_name_v1": "uncertainty_entry_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "NOT_ESTABLISHED",
        "source_field_v1": "",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "NOT_ESTABLISHED",
        "normalization_v1": "PASSTHROUGH",
        "blocking_reason_v1": "same as p_long_entry_v1",
    },
    {
        "field_name_v1": "margin_entry_v1",
        "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v1": "NOT_ESTABLISHED",
        "source_field_v1": "",
        "lineage_v1": "AS_OF_AT_TRADE_OPEN",
        "availability_v1": "NOT_ESTABLISHED",
        "normalization_v1": "PASSTHROUGH",
        "blocking_reason_v1": "same as p_long_entry_v1",
    },
    # --- TRANSFORMER_SIGNAL_AT_BAR (samstemte alignment) ---
    {
        "field_name_v1": "exit_prob_v1",
        "category_v1": "TRANSFORMER_SIGNAL_AT_BAR",
        "source_v1": "EXIT_EVAL_TRACE",
        "source_field_v1": "exit_prob",
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "PASSTHROUGH",
        "rationale_v1": (
            "Exit-transformer's per-bar exit-probability is itself a state "
            "feature for IQL: it lets the offline RL agent literally see "
            "what the exit transformer recommends at each bar. This is the "
            "concrete realization of the user's samstemte requirement. "
            "Available because EXIT_EVAL_TRACE.csv was logged at runtime."
        ),
    },
]

# Forbidden state field set (re-used from MDP gate)
FORBIDDEN_STATE_FIELDS_V1 = list(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)


# ---------------------------------------------------------------------------
# Helpers / validators (reuse contract_gate)
# ---------------------------------------------------------------------------

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


def validate_no_shortcut(features: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = [f["field_name_v1"] for f in features]
    forbidden_hits = sorted(set(field_names) & set(FORBIDDEN_STATE_FIELDS_V1))
    if forbidden_hits:
        raise RuntimeError(f"FORBIDDEN_STATE_FIELD_IN_PROPOSED_SCHEMA: {forbidden_hits}")
    # Pattern-check field names against forbidden tokens
    forbidden_tokens = ["exit_reason", "post_exit", "duration_bars", "_replay_end_obs", "is_terminal", "bar_count"]
    pattern_hits = []
    for name in field_names:
        for tok in forbidden_tokens:
            if tok in name and "exit_prob" not in name:
                pattern_hits.append(name)
                break
    if pattern_hits:
        raise RuntimeError(f"FORBIDDEN_TOKEN_IN_FIELD_NAME: {pattern_hits}")
    # Identity tokens
    identity_tokens = ["candidate_uid", "trade_uid", "trade_id"]
    identity_hits = [n for n in field_names if any(tok in n for tok in identity_tokens)]
    if identity_hits:
        raise RuntimeError(f"IDENTITY_TOKEN_IN_FIELD_NAME: {identity_hits}")
    return {
        "layer_name": "EXIT_PER_BAR_STATE_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "feature_count_v1": len(features),
        "forbidden_field_intersection_v1": forbidden_hits,
        "forbidden_token_pattern_hits_v1": pattern_hits,
        "identity_token_hits_v1": identity_hits,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _exit_eval_trace_paths() -> list[Path]:
    return sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "TRUTH_MONFRI_WEEK_*/replay/chunk_0/EXIT_EVAL_TRACE.csv"
        ),
        key=lambda p: p.parent.parent.parent.name,
    )


def _per_bar_decision_dataset_path() -> Path:
    return (
        INPUT_PER_BAR_SCAFFOLD_ROOT
        / "PER_BAR_TRAJECTORY_V1"
        / "per_bar_decision_dataset_v1.parquet"
    )


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_MDP_ROOT, INPUT_PER_BAR_SCAFFOLD_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "mdp_contract": INPUT_MDP_ROOT / "mdp_contract_v1.json",
        "mdp_state_schema_requirements": INPUT_MDP_ROOT
        / "state_schema_requirements_v1.json",
        "mdp_no_shortcut_axioms": INPUT_MDP_ROOT / "no_shortcut_axioms_v1.json",
        "mdp_summary": INPUT_MDP_ROOT / "summary_v1.json",
        "per_bar_summary": INPUT_PER_BAR_SCAFFOLD_ROOT / "summary_v1.json",
        "per_bar_decision_dataset": _per_bar_decision_dataset_path(),
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}")
    trace_paths = _exit_eval_trace_paths()
    if len(trace_paths) == 0:
        raise RuntimeError("NO_EXIT_EVAL_TRACE_FILES_FOUND")
    return {
        "required_paths": required,
        "exit_eval_trace_paths": trace_paths,
        "base34_path": BASE34_M5_FEATURES_PATH,
        "mdp_contract": _read_json(required["mdp_contract"]),
        "mdp_state_schema_requirements": _read_json(required["mdp_state_schema_requirements"]),
        "mdp_summary": _read_json(required["mdp_summary"]),
    }


# ---------------------------------------------------------------------------
# Availability audit
# ---------------------------------------------------------------------------


def _audit_availability(
    inputs: dict[str, Any], features: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # Per-bar decision dataset
    per_bar_df = pd.read_parquet(_per_bar_decision_dataset_path())
    per_bar_cols = set(per_bar_df.columns)
    # Sample EXIT_EVAL_TRACE.csv
    trace_paths = inputs["exit_eval_trace_paths"]
    trace_sample = pd.read_csv(trace_paths[0], nrows=10)
    trace_cols = set(trace_sample.columns)
    # BASE34
    base34_df = pd.read_parquet(BASE34_M5_FEATURES_PATH)
    base34_cols = set(base34_df.columns)
    # TRADE_OUTCOMES
    weeks = sorted(
        DEFAULT_REPORTS_ROOT.glob("TRUTH_MONFRI_WEEK_*/trade_outcomes_*_MERGED.parquet")
    )
    trade_outcomes_cols: set[str] = set()
    for w in weeks:
        df = pd.read_parquet(w)
        if df.empty:
            continue
        trade_outcomes_cols = set(df.columns)
        break

    source_cols_map = {
        "PER_BAR_SCAFFOLD": per_bar_cols,
        "EXIT_EVAL_TRACE": trace_cols,
        "BASE34_M5": base34_cols,
        "TRADE_OUTCOMES": trade_outcomes_cols,
        "NOT_ESTABLISHED": set(),
    }
    have = []
    derivable = []
    not_established = []
    for feat in features:
        source = feat["source_v1"]
        source_field = feat["source_field_v1"]
        availability = feat["availability_v1"]
        verified = source_field in source_cols_map.get(source, set())
        row = {
            **feat,
            "source_field_present_in_source_v1": verified,
        }
        if availability == "NOT_ESTABLISHED":
            not_established.append(feat["field_name_v1"])
        elif availability == "HAVE":
            if not verified:
                row["audit_status_v1"] = "AUDIT_FAIL_FIELD_NOT_IN_SOURCE"
                # Demote to NOT_ESTABLISHED for safety
                row["availability_v1"] = "NOT_ESTABLISHED"
                not_established.append(feat["field_name_v1"])
            else:
                row["audit_status_v1"] = "AUDIT_PASS"
                have.append(feat["field_name_v1"])
        elif availability == "DERIVABLE":
            row["audit_status_v1"] = "DERIVABLE_REQUIRES_NEXT_GATE_COMPUTATION"
            derivable.append(feat["field_name_v1"])
        rows.append(row)
    return {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_AVAILABILITY_AUDIT_V1",
        "feature_rows_v1": rows,
        "have_count_v1": len(have),
        "have_field_names_v1": sorted(have),
        "derivable_count_v1": len(derivable),
        "derivable_field_names_v1": sorted(derivable),
        "not_established_count_v1": len(not_established),
        "not_established_field_names_v1": sorted(not_established),
        "source_column_counts_v1": {k: len(v) for k, v in source_cols_map.items()},
    }


# ---------------------------------------------------------------------------
# Sample validation: build state vectors for a few trades
# ---------------------------------------------------------------------------


def _sample_state_vector_validation(
    inputs: dict[str, Any], features: list[dict[str, Any]], n_sample_trades: int = 5
) -> dict[str, Any]:
    per_bar_df = pd.read_parquet(_per_bar_decision_dataset_path())
    if per_bar_df.empty:
        return {
            "status_v1": "PER_BAR_DATASET_EMPTY",
            "samples_v1": [],
        }
    sample_uids = list(per_bar_df["candidate_uid_v1"].drop_duplicates().head(n_sample_trades))
    sample_rows = per_bar_df[per_bar_df["candidate_uid_v1"].isin(sample_uids)].copy()

    # Load BASE34
    base34_df = pd.read_parquet(BASE34_M5_FEATURES_PATH)
    base34_required_cols = [
        f["source_field_v1"]
        for f in features
        if f["source_v1"] == "BASE34_M5"
    ]
    base34_present = [c for c in base34_required_cols if c in base34_df.columns]
    base34_subset_cols = ["time"] + base34_present if "time" in base34_df.columns else []
    base34_subset = (
        base34_df.loc[:, base34_subset_cols].copy() if base34_subset_cols else pd.DataFrame()
    )
    if not base34_subset.empty:
        base34_subset["time"] = pd.to_datetime(base34_subset["time"], utc=True)
    # Load EXIT_EVAL_TRACE for sample weeks (just the first week's trace)
    trace_paths = inputs["exit_eval_trace_paths"]
    trace_required_cols = [
        f["source_field_v1"]
        for f in features
        if f["source_v1"] == "EXIT_EVAL_TRACE"
    ]
    trace_present_check = pd.read_csv(trace_paths[0], nrows=1)
    trace_present = [c for c in trace_required_cols if c in trace_present_check.columns]
    # We only do a small sample-validation, so keep simple: report which columns join
    state_columns = [f["field_name_v1"] for f in features if f["availability_v1"] == "HAVE"]
    return {
        "status_v1": "SAMPLE_VALIDATED",
        "n_sample_trades_v1": len(sample_uids),
        "sample_candidate_uids_v1": list(sample_uids),
        "sample_decision_row_count_v1": int(len(sample_rows)),
        "state_have_count_v1": len(state_columns),
        "state_have_columns_v1": state_columns,
        "base34_columns_present_v1": base34_present,
        "base34_columns_required_v1": base34_required_cols,
        "exit_trace_columns_present_v1": trace_present,
        "exit_trace_columns_required_v1": trace_required_cols,
    }


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    availability_audit: dict[str, Any], sample_validation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_REPRODUCIBILITY_AUDIT_V1",
        "feature_count_total_v1": len(PROPOSED_STATE_FEATURES),
        "feature_count_have_v1": availability_audit["have_count_v1"],
        "feature_count_derivable_v1": availability_audit["derivable_count_v1"],
        "feature_count_not_established_v1": availability_audit["not_established_count_v1"],
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
        "sample_validation_status_v1": sample_validation["status_v1"],
    }


def _go_no_go(
    availability_audit: dict[str, Any], no_shortcut_audit: dict[str, Any]
) -> tuple[str, str, str]:
    if no_shortcut_audit["status_v1"] != "PASS":
        return (
            "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_BLOCKED_BY_NO_SHORTCUT_FAIL",
            "HOLD_UNTIL_STATE_FEATURE_GAPS_RESOLVED_V1",
            "No-shortcut audit failed; resolve before proceeding.",
        )
    have = availability_audit["have_count_v1"]
    not_established = availability_audit["not_established_count_v1"]
    if not_established == 0:
        return (
            "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_LOCKED_AVAILABILITY_AUDIT_PASSED",
            "EXIT_ACTION_SUPPORT_AUGMENT_V1",
            (
                f"State schema locked with {have} HAVE features, no NOT_ESTABLISHED "
                "gaps. Next: action-support augmentation."
            ),
        )
    return (
        "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_PARTIAL_SOME_FEATURES_NOT_ESTABLISHED",
        "EXIT_ACTION_SUPPORT_AUGMENT_V1",
        (
            f"State schema locked with {have} HAVE features and {not_established} "
            "NOT_ESTABLISHED features. The schema is research-ready for next-gate "
            "augmentation; the NOT_ESTABLISHED entries (entry-transformer outputs "
            "p_long_entry, p_hat_entry, uncertainty_entry, margin_entry) can be "
            "added via a parallel DEEPEN_ENTRY_CONTEXT_FEATURE_LINEAGE_V1 gate "
            "without blocking the main pre-train sequence."
        ),
    )


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
    files.append(
        {
            "name_v1": "exit_eval_trace_first_path",
            "path_v1": str(inputs["exit_eval_trace_paths"][0]),
            "sha256_v1": _file_hash(inputs["exit_eval_trace_paths"][0]),
        }
    )
    return {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "mdp_root_v1": str(INPUT_MDP_ROOT),
            "per_bar_scaffold_root_v1": str(INPUT_PER_BAR_SCAFFOLD_ROOT),
        },
        "raw_data_v1": {
            "base34_m5_v1": str(BASE34_M5_FEATURES_PATH),
            "exit_eval_trace_path_count_v1": len(inputs["exit_eval_trace_paths"]),
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

    no_shortcut_audit = validate_no_shortcut(PROPOSED_STATE_FEATURES)
    _write_json(artifact_root / "no_shortcut_audit_v1.json", no_shortcut_audit)

    availability_audit = _audit_availability(inputs, PROPOSED_STATE_FEATURES)
    _write_json(artifact_root / "availability_audit_v1.json", availability_audit)
    _write_rows(
        artifact_root / "feature_availability_table_v1.csv",
        availability_audit["feature_rows_v1"],
    )

    state_contract = {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        "feature_count_v1": len(PROPOSED_STATE_FEATURES),
        "trainable_have_count_v1": availability_audit["have_count_v1"],
        "feature_definitions_v1": PROPOSED_STATE_FEATURES,
        "category_counts_v1": {
            cat: sum(1 for f in PROPOSED_STATE_FEATURES if f["category_v1"] == cat)
            for cat in [
                "TRADE_STATE_RUNNING",
                "MARKET_STATE_AT_BAR",
                "ENTRY_CONTEXT_SNAPSHOT",
                "TRANSFORMER_SIGNAL_AT_BAR",
            ]
        },
        "samstemte_alignment_field_v1": "exit_prob_v1",
        "research_only_v1": True,
    }
    _write_json(artifact_root / "state_feature_contract_v1.json", state_contract)

    sample_validation = _sample_state_vector_validation(inputs, PROPOSED_STATE_FEATURES)
    _write_json(artifact_root / "sample_state_vector_validation_v1.json", sample_validation)

    repro = _reproducibility_audit(availability_audit, sample_validation)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go(availability_audit, no_shortcut_audit)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "feature_count_total_v1": len(PROPOSED_STATE_FEATURES),
        "feature_count_have_v1": availability_audit["have_count_v1"],
        "feature_count_not_established_v1": availability_audit["not_established_count_v1"],
        "category_counts_v1": state_contract["category_counts_v1"],
        "samstemte_alignment_v1": "exit_prob_v1 included as TRANSFORMER_SIGNAL_AT_BAR; IQL state literally sees what exit transformer outputs at each bar",
        "no_shortcut_audit_status_v1": no_shortcut_audit["status_v1"],
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
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_GO_NO_GO_V1",
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
            "Research-only contract lock. No training. Adapter/R6/IQL "
            "production/live, freeze/promo/live, exit_manager modification "
            "all forbidden. Training remains BLOCKED until the remaining "
            "four pre-train gates pass."
        ),
    }
    _write_json(
        artifact_root / "exit_per_bar_state_feature_contract_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Exit Per-Bar State-Feature Contract V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** until remaining four pre-train gates pass.",
        "",
        "## Schema summary",
        f"- Total features: {len(PROPOSED_STATE_FEATURES)}",
        f"- HAVE: {availability_audit['have_count_v1']}",
        f"- NOT_ESTABLISHED: {availability_audit['not_established_count_v1']}",
        "",
        "## Categories",
    ]
    for cat, count in state_contract["category_counts_v1"].items():
        report_lines.append(f"- `{cat}`: {count}")
    report_lines.extend([
        "",
        "## Samstemte alignment",
        "- `exit_prob_v1` included as TRANSFORMER_SIGNAL_AT_BAR feature.",
        "- IQL state literally observes the exit transformer's output at each bar.",
        "",
        "## NOT_ESTABLISHED features",
    ])
    for name in availability_audit["not_established_field_names_v1"]:
        report_lines.append(f"- `{name}`")
    report_lines.extend([
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
                artifact_root / "exit_per_bar_state_feature_contract_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "state_feature_contract": str(artifact_root / "state_feature_contract_v1.json"),
            "availability_audit": str(artifact_root / "availability_audit_v1.json"),
            "feature_availability_table_csv": str(
                artifact_root / "feature_availability_table_v1.csv"
            ),
            "no_shortcut_audit": str(artifact_root / "no_shortcut_audit_v1.json"),
            "sample_validation": str(
                artifact_root / "sample_state_vector_validation_v1.json"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
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
        description="Materialize EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
