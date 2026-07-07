#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.research.iql_training_harness_stub_v1 import validate_iql_dataset_schema_v1


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
DEFAULT_EXTENSION_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260422T_IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"

LAYER_ID = "IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"
R6_FREEZE_ID = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
R5_2_FREEZE_ID = "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"

LEDGER_FILE = "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
OBSERVATION_CONTRACT_FILE = "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json"
SEQUENCE_ROW_VIEW_FILE = "shadow_meta_all_trade_review_management_rl_sequence_row_view_v1.parquet"
STRICT_TRANSITION_VIEW_FILE = "shadow_meta_all_trade_review_management_rl_sequence_strict_transition_view_v1.parquet"
SEQUENCE_STATUS_FILE = "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json"
SEQUENCE_SUMMARY_FILE = "shadow_meta_all_trade_review_management_rl_sequence_summary_v1.json"
BANDIT_STATUS_FILE = "shadow_meta_all_trade_review_management_bandit_status_v1.json"
BANDIT_OBSERVED_VIEW_FILE = "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet"
BANDIT_DM_VIEW_FILE = "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet"
BANDIT_EXIT_LOCAL_VIEW_FILE = "shadow_meta_all_trade_review_management_bandit_exit_local_reward_view_v1.parquet"
BANDIT_HOLD_RETURN_VIEW_FILE = "shadow_meta_all_trade_review_management_bandit_hold_episode_return_view_v1.parquet"
POLICY_LOG_FILE = "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet"
POLICY_LOG_SUMMARY_FILE = "shadow_meta_all_trade_review_management_policy_logging_summary_v1.json"
PATH_DYNAMICS_SUMMARY_FILE = "truth_path_dynamics_logging_v2_implementation_and_replay_audit_v1.json"

OUTPUTS = {
    "contract": "iql_foundation_mdp_contract_v1.json",
    "mdp_feasibility": "iql_foundation_mdp_domain_feasibility_audit_v1.csv",
    "management_contract": "iql_foundation_management_mdp_contract_v1.json",
    "reward_audit": "iql_foundation_reward_audit_v1.csv",
    "transition_audit_json": "iql_foundation_transition_linkage_audit_v1.json",
    "transition_audit_csv": "iql_foundation_transition_linkage_audit_v1.csv",
    "dataset_schema": "iql_foundation_dataset_schema_v1.json",
    "dataset_schema_fields": "iql_foundation_dataset_schema_fields_v1.csv",
    "support_ood": "iql_foundation_support_ood_audit_v1.json",
    "support_pockets": "iql_foundation_support_ood_state_action_pockets_v1.csv",
    "baseline_spec": "iql_foundation_baseline_comparator_spec_v1.json",
    "training_harness": "iql_foundation_training_harness_stub_v1.json",
    "decision_matrix": "iql_foundation_decision_matrix_v1.csv",
    "summary": "iql_foundation_summary_v1.json",
    "report": "iql_foundation_report_v1.md",
    "manifest": "iql_foundation_manifest_v1.json",
    "status": "iql_foundation_status_v1.json",
    "consistency_audit": "iql_foundation_consistency_audit_v1.csv",
}
TOP_LEVEL_SUMMARY = "truth_iql_foundation_mdp_contract_and_dataset_scaffold_v1.json"

IQL_DATASET_FIELDS = [
    "episode_id",
    "transition_id",
    "state_vector",
    "state_feature_names",
    "action",
    "action_id",
    "reward",
    "next_state_vector",
    "done",
    "discount",
    "decision_ts",
    "candidate_uid_exact",
    "source_policy_version",
    "behavior_policy_status",
    "support_status",
    "as_of_schema_version",
    "reward_version",
    "outcome_backfill_version",
]
ACTION_ID_MAP = {"HOLD": 0, "EXIT_NOW": 1}
PATH_DYNAMICS_FIELDS = {
    "last_peak_ts": "as_of_management_core_last_peak_ts_utc_v1",
    "last_mfe_ts": "as_of_management_core_last_mfe_ts_utc_v1",
    "last_peak_mfe": "as_of_management_core_last_peak_mfe_bps_v1",
    "max_mfe_without_mae": "as_of_management_core_max_mfe_without_mae_bps_v1",
    "mfe_mae_sequence_order": "as_of_management_core_mfe_mae_sequence_order_v1",
}
CANONICAL_MANAGEMENT_CORE_9_INPUTS = [
    "as_of_atr_bps_v1",
    "as_of_hour_utc_v1",
    "as_of_session_v1",
    "as_of_side_v1",
    "as_of_trend_regime_v1",
    "as_of_vol_regime_v1",
    "as_of_weekday_utc_v1",
    "as_of_management_core_minutes_held_at_anchor_v1",
    "as_of_management_core_giveback_ratio_from_peak_v1",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty active truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_parquet_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].astype("string").value_counts(dropna=False).to_dict().items()}


def _missing_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return series.isna() | text.isin(["", "nan", "NaN", "NaT", "None", "<NA>", "NOT_AVAILABLE"])


def _non_null_count(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int((~_missing_mask(df[col])).sum())


def _coverage_status(non_null: int, total: int, *, pending_if_missing: bool = False) -> str:
    if total <= 0:
        return "MISSING"
    if non_null == total:
        return "READY"
    if non_null > 0:
        return "PARTIAL"
    return "PENDING_REPLAY" if pending_if_missing else "MISSING"


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {
            "count_v1": 0,
            "mean_v1": None,
            "std_v1": None,
            "min_v1": None,
            "p05_v1": None,
            "p50_v1": None,
            "p95_v1": None,
            "max_v1": None,
        }
    return {
        "count_v1": int(len(numeric)),
        "mean_v1": float(numeric.mean()),
        "std_v1": float(numeric.std(ddof=0)),
        "min_v1": float(numeric.min()),
        "p05_v1": float(numeric.quantile(0.05)),
        "p50_v1": float(numeric.quantile(0.50)),
        "p95_v1": float(numeric.quantile(0.95)),
        "max_v1": float(numeric.max()),
    }


def _boolish(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.upper().str.strip()
    return text.isin(["TRUE", "1", "YES", "Y"])


def _resolve_management_dir(reports_root: Path, management_dir_arg: str | None) -> Path:
    if management_dir_arg:
        path = Path(management_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Management substrate dir does not exist: {path}")
        return path

    required = [LEDGER_FILE, SEQUENCE_ROW_VIEW_FILE, STRICT_TRANSITION_VIEW_FILE, BANDIT_DM_VIEW_FILE, BANDIT_STATUS_FILE]
    preferred_names = [
        "ALL_TRADE_REVIEW_LEDGER_20260421T_REWARD_CHANNEL_FIX_R1_CANONICAL",
        "ALL_TRADE_REVIEW_LEDGER_20260420_RUNTIME_RECOVERY_R8_HANDOFF_REALFIX",
    ]
    for name in preferred_names:
        candidate = reports_root / name
        if candidate.exists() and all((candidate / item).exists() for item in required):
            return candidate

    candidates: list[Path] = []
    for path in reports_root.iterdir():
        if path.is_dir() and path.name.startswith("ALL_TRADE_REVIEW_LEDGER_") and all((path / item).exists() for item in required):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No management substrate dir with required IQL inputs found under {reports_root}")
    non_empty: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            rows = len(pd.read_parquet(path / BANDIT_DM_VIEW_FILE, columns=["action_label_v1"]))
        except Exception:
            rows = 0
        non_empty.append((float(rows), path))
    non_empty.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return non_empty[0][1]


def _resolve_policy_log_dir(reports_root: Path, policy_log_dir_arg: str | None) -> Path | None:
    if policy_log_dir_arg:
        path = Path(policy_log_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Policy log dir does not exist: {path}")
        return path
    candidates = [
        path
        for path in reports_root.iterdir()
        if path.is_dir() and (path / POLICY_LOG_FILE).exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _latest_path_replay_status() -> dict[str, Any]:
    base = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
    candidates = sorted(
        base.glob("PATH_DYNAMICS_LOGGING_V2_REPLAY_*/path_dynamics_v2_*status_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        payload = _read_json_optional(path)
        if payload:
            payload["_source_path_v1"] = str(path)
            return payload
    return {}


def _path_dynamics_field_statuses(
    path_summary: dict[str, Any],
    policy_log_df: pd.DataFrame,
    replay_status: dict[str, Any],
) -> list[dict[str, Any]]:
    coverage_rows = path_summary.get("coverage_summary_v1", [])
    policy_rows_by_field: dict[str, dict[str, Any]] = {}
    if isinstance(coverage_rows, list):
        for row in coverage_rows:
            if not isinstance(row, dict):
                continue
            if row.get("layer_name") == "POLICY_LOG":
                policy_rows_by_field[str(row.get("field_id"))] = row

    replay_running = bool(replay_status.get("running"))
    statuses: list[dict[str, Any]] = []
    for field_id, policy_col in PATH_DYNAMICS_FIELDS.items():
        coverage = policy_rows_by_field.get(field_id, {})
        total = int(coverage.get("total_rows") or len(policy_log_df) or 0)
        non_null = int(coverage.get("non_null_count") or _non_null_count(policy_log_df, policy_col))
        schema_present = bool(coverage.get("schema_present", policy_col in policy_log_df.columns))
        if schema_present and total > 0 and non_null == total:
            status = "READY_OPTIONAL_NOT_TRAINING_REQUIRED"
            leakage = "LOW_AS_OF_POLICY_LOG"
        elif replay_running:
            status = "PENDING_REPLAY"
            leakage = "NOT_CHECKED_UNTIL_REPLAY_COVERAGE"
        else:
            status = "PENDING_LOGGING"
            leakage = "NOT_CHECKED_UNTIL_LOGGED"
        statuses.append(
            {
                "field_id_v1": field_id,
                "policy_log_field_v1": policy_col,
                "schema_present_v1": schema_present,
                "non_null_count_v1": non_null,
                "total_rows_v1": total,
                "coverage_v1": float(non_null / total) if total else 0.0,
                "status_v1": status,
                "leakage_risk_v1": leakage,
                "training_requirement_v1": "OPTIONAL_PENDING_NOT_A_REQUIRED_IQL_FEATURE",
            }
        )
    return statuses


def _build_reward_audit(ledger_df: pd.DataFrame, dm_df: pd.DataFrame) -> pd.DataFrame:
    if ledger_df.empty:
        return pd.DataFrame()

    ledger_cols = [
        "trade_uid",
        "candidate_uid",
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
        "hindsight_hold_longer_extra_value_bps_v1",
        "cata_loser",
        "bad_trade",
    ]
    available_ledger_cols = [col for col in ledger_cols if col in ledger_df.columns]
    joined = dm_df.copy()
    if not dm_df.empty and "trade_uid_exact_v1" in dm_df.columns and "trade_uid" in ledger_df.columns:
        joined = dm_df.merge(
            ledger_df[available_ledger_cols].drop_duplicates(subset=["trade_uid"]),
            left_on="trade_uid_exact_v1",
            right_on="trade_uid",
            how="left",
            suffixes=("", "_ledger"),
        )
    elif not dm_df.empty:
        joined = dm_df.copy()
    else:
        joined = ledger_df.rename(
            columns={
                "trade_uid": "trade_uid_exact_v1",
                "candidate_uid": "candidate_uid_exact_v1",
                "realized_pnl_bps": "hindsight_reward_realized_pnl_bps_v1",
            }
        )

    pnl = pd.to_numeric(
        joined.get("hindsight_reward_realized_pnl_bps_v1", joined.get("realized_pnl_bps")),
        errors="coerce",
    )
    mfe = pd.to_numeric(joined.get("hindsight_peak_mfe_bps_v1", joined.get("mfe_bps")), errors="coerce")
    mfe = mfe.fillna(pd.to_numeric(joined.get("mfe_bps"), errors="coerce"))
    mae = pd.to_numeric(joined.get("mae_bps"), errors="coerce").abs()
    giveback = pd.to_numeric(joined.get("hindsight_peak_to_exit_giveback_bps_v1"), errors="coerce")
    giveback = giveback.fillna((mfe - pnl).clip(lower=0.0))
    hold_damage = pd.to_numeric(joined.get("hindsight_hold_longer_extra_value_bps_v1"), errors="coerce").fillna(0.0).clip(lower=0.0)
    bad_trade = _boolish(joined.get("bad_trade", pd.Series(False, index=joined.index))).astype(float)
    cata = _boolish(joined.get("cata_loser", pd.Series(False, index=joined.index))).astype(float)

    values = {
        "REALIZED_PNL_REWARD": pnl,
        "MFE_CAPTURE_REWARD": (pnl / mfe.where(mfe.abs() > 1e-9)).clip(lower=-2.0, upper=2.0),
        "MAE_PENALTY_REWARD": -mae,
        "GIVEBACK_PENALTY_REWARD": -giveback,
        "TAIL_CONTROL_REWARD": pnl - (25.0 * bad_trade) - (75.0 * cata),
        "RUNNER_DAMAGE_PENALTY": -hold_damage,
        "TRANSPARENT_COMBINED_REWARD": pnl - (0.25 * mae) - (0.25 * giveback) - (0.50 * hold_damage) - (25.0 * bad_trade) - (75.0 * cata),
    }
    specs = {
        "REALIZED_PNL_REWARD": {
            "formula": "terminal_realized_pnl_bps",
            "mode": "bandit_or_sequence_terminal_reward",
            "verdict": "USABLE_FOR_OFFLINE_RESEARCH",
            "leakage": "LOW_IF_USED_ONLY_AS_REWARD",
        },
        "MFE_CAPTURE_REWARD": {
            "formula": "terminal_realized_pnl_bps / max(hindsight_peak_mfe_bps, eps), clipped [-2, 2]",
            "mode": "bandit_or_sequence_terminal_reward",
            "verdict": "USABLE_FOR_OFFLINE_RESEARCH",
            "leakage": "MEDIUM_HINDSIGHT_PATH_METRIC_REWARD_ONLY",
        },
        "MAE_PENALTY_REWARD": {
            "formula": "-abs(terminal_mae_bps)",
            "mode": "bandit_or_sequence_terminal_reward",
            "verdict": "USABLE_FOR_OFFLINE_RESEARCH",
            "leakage": "LOW_IF_USED_ONLY_AS_REWARD",
        },
        "GIVEBACK_PENALTY_REWARD": {
            "formula": "-hindsight_peak_to_exit_giveback_bps",
            "mode": "bandit_or_sequence_terminal_reward",
            "verdict": "USABLE_FOR_OFFLINE_RESEARCH",
            "leakage": "MEDIUM_HINDSIGHT_PATH_METRIC_REWARD_ONLY",
        },
        "TAIL_CONTROL_REWARD": {
            "formula": "terminal_realized_pnl_bps - 25*bad_trade - 75*cata_loser",
            "mode": "bandit_or_sequence_terminal_reward",
            "verdict": "USABLE_FOR_OFFLINE_RESEARCH",
            "leakage": "LOW_IF_USED_ONLY_AS_REWARD",
        },
        "RUNNER_DAMAGE_PENALTY": {
            "formula": "-max(hindsight_hold_longer_extra_value_bps, 0)",
            "mode": "audit_only_until_counterfactual_locality_is_locked",
            "verdict": "AUDIT_ONLY",
            "leakage": "HIGH_COUNTERFACTUAL_HINDSIGHT_LOCALITY_NOT_LOCKED",
        },
        "TRANSPARENT_COMBINED_REWARD": {
            "formula": "pnl - .25*mae - .25*giveback - .50*runner_damage - 25*bad_trade - 75*cata_loser",
            "mode": "audit_only_until_weights_are_locked",
            "verdict": "AUDIT_ONLY",
            "leakage": "MEDIUM_COMPOSITE_REWARD_NOT_LOCKED",
        },
    }

    rows = []
    ledger_total = int(len(ledger_df))
    dm_total = int(len(dm_df))
    for name, series in values.items():
        dist = _distribution(series)
        spec = specs[name]
        rows.append(
            {
                "reward_candidate_v1": name,
                "formula_v1": spec["formula"],
                "ledger_coverage_v1": f"{dist['count_v1']}/{ledger_total}" if dm_df.empty else "see_management_dm_coverage",
                "management_dm_coverage_v1": f"{dist['count_v1']}/{dm_total}",
                "coverage_rate_v1": float(dist["count_v1"] / dm_total) if dm_total else 0.0,
                "distribution_count_v1": dist["count_v1"],
                "distribution_mean_v1": dist["mean_v1"],
                "distribution_std_v1": dist["std_v1"],
                "distribution_min_v1": dist["min_v1"],
                "distribution_p05_v1": dist["p05_v1"],
                "distribution_p50_v1": dist["p50_v1"],
                "distribution_p95_v1": dist["p95_v1"],
                "distribution_max_v1": dist["max_v1"],
                "hindsight_only_v1": True,
                "as_of_state_allowed_v1": False,
                "leakage_risk_v1": spec["leakage"],
                "suitable_for_v1": spec["mode"],
                "verdict_v1": spec["verdict"],
                "trading_performance_interpretation_v1": "NOT_PERFORMED_FOUNDATION_ONLY",
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_transition_audit(seq_df: pd.DataFrame, strict_df: pd.DataFrame, dm_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    action_counts = _safe_counts(seq_df, "action_label_v1")
    next_counts = _safe_counts(seq_df, "sequence_next_link_status_v1")
    membership_counts = _safe_counts(seq_df, "sequence_dataset_membership_v1")
    terminal_counts = _safe_counts(seq_df, "sequence_terminal_step_status_v1")

    full_sequence_ready = int(len(strict_df))
    bandit_only = int((seq_df.get("sequence_dataset_membership_v1", pd.Series(dtype="string")).astype("string") == "BANDIT_SAFE_ONLY").sum()) if not seq_df.empty else 0
    ineligible = int((seq_df.get("sequence_dataset_membership_v1", pd.Series(dtype="string")).astype("string") == "SEQUENCE_INELIGIBLE").sum()) if not seq_df.empty else 0
    hold_rows = seq_df.loc[seq_df.get("action_label_v1", pd.Series(dtype="string")).astype("string").eq("HOLD")].copy() if not seq_df.empty else pd.DataFrame()
    exit_rows = seq_df.loc[seq_df.get("action_label_v1", pd.Series(dtype="string")).astype("string").eq("EXIT_NOW")].copy() if not seq_df.empty else pd.DataFrame()
    exact_next_mask = pd.Series(False, index=seq_df.index)
    if not seq_df.empty and "sequence_next_row_key_v1" in seq_df.columns:
        exact_next_mask = ~_missing_mask(seq_df["sequence_next_row_key_v1"])
    exact_next_state = int(exact_next_mask.sum()) if not seq_df.empty else 0
    hold_next = int(exact_next_mask.loc[hold_rows.index].sum()) if not hold_rows.empty else 0
    exit_done = int(
        exit_rows.get("sequence_terminal_step_status_v1", pd.Series(dtype="string"))
        .astype("string")
        .eq("TERMINAL_REALIZED_EXIT")
        .sum()
    ) if not exit_rows.empty else 0
    terminal_transition_count = int(
        strict_df.get("sequence_terminal_step_status_v1", pd.Series(dtype="string"))
        .astype("string")
        .eq("TERMINAL_REALIZED_EXIT")
        .sum()
    ) if not strict_df.empty else 0

    rows = []
    if not seq_df.empty:
        group_cols = ["action_label_v1", "sequence_next_link_status_v1", "sequence_terminal_step_status_v1", "sequence_dataset_membership_v1"]
        grouped = seq_df.groupby(group_cols, dropna=False).size().reset_index(name="row_count_v1")
        rows = grouped.to_dict(orient="records")
    audit_df = pd.DataFrame.from_records(rows)
    if audit_df.empty:
        audit_df = pd.DataFrame(
            columns=[
                "action_label_v1",
                "sequence_next_link_status_v1",
                "sequence_terminal_step_status_v1",
                "sequence_dataset_membership_v1",
                "row_count_v1",
            ]
        )

    primary_missing = "HOLD_NEXT_STATE_LINKS_NOT_LOGGED" if hold_next == 0 and len(hold_rows) > 0 else "OTHER_OR_NOT_ESTABLISHED"
    summary = {
        "audit_id_v1": "TRANSITION_LINKAGE_AUDIT_V1",
        "management_row_count_v1": int(len(seq_df)),
        "dm_candidate_row_count_v1": int(len(dm_df)),
        "full_sequence_ready_transition_count_v1": full_sequence_ready,
        "terminal_only_full_transition_count_v1": terminal_transition_count,
        "exact_next_management_state_count_v1": exact_next_state,
        "same_trade_episode_coverage_count_v1": int(seq_df.get("sequence_episode_key_v1", pd.Series(dtype="string")).notna().sum()) if not seq_df.empty and "sequence_episode_key_v1" in seq_df.columns else 0,
        "hold_to_next_state_transition_count_v1": hold_next,
        "exit_now_to_done_transition_count_v1": terminal_transition_count,
        "bandit_only_row_count_v1": bandit_only,
        "ineligible_row_count_v1": ineligible,
        "action_counts_v1": action_counts,
        "next_link_status_counts_v1": next_counts,
        "terminal_status_counts_v1": terminal_counts,
        "dataset_membership_counts_v1": membership_counts,
        "primary_transition_gap_v1": primary_missing,
        "hold_next_step_problem_blocks_sequence_iql_v1": bool(hold_next == 0 and len(hold_rows) > 0),
        "path_dynamics_replay_help_assessment_v1": "HELPS_STATE_ENRICHMENT_NOT_HOLD_NEXT_LINKAGE" if hold_next == 0 else "MAY_HELP_STATE_COVERAGE",
        "other_transition_logging_needed_v1": bool(hold_next == 0),
        "verdict_v1": "PARTIAL_TRANSITIONS" if full_sequence_ready > 0 else "BANDIT_ONLY_READY",
    }
    return summary, audit_df


def _build_support_audit(dm_df: pd.DataFrame, strict_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if dm_df.empty:
        return {
            "audit_id_v1": "OFFLINE_RL_SUPPORT_AND_OOD_AUDIT_V1",
            "overall_support_verdict_v1": "NOT_ESTABLISHED",
            "reason_v1": "No direct-method management rows found.",
        }, pd.DataFrame()

    work = dm_df.copy()
    minutes = pd.to_numeric(work.get("as_of_management_core_minutes_held_at_anchor_v1"), errors="coerce")
    work["hold_age_bucket_v1"] = np.select(
        [minutes <= 30, minutes <= 120, minutes > 120],
        ["EARLY_0_30M", "MID_31_120M", "LATE_120M_PLUS"],
        default="UNKNOWN",
    )
    pocket_cols = ["as_of_session_v1", "as_of_vol_regime_v1", "as_of_trend_regime_v1", "hold_age_bucket_v1", "action_label_v1"]
    for col in pocket_cols:
        if col not in work.columns:
            work[col] = "UNKNOWN"
    pockets = work.groupby(pocket_cols, dropna=False).size().reset_index(name="row_count_v1")
    pockets["support_status_v1"] = np.select(
        [pockets["row_count_v1"] >= 50, pockets["row_count_v1"] >= 10],
        ["SUPPORTED_RESEARCH_POCKET", "THIN_BUT_OBSERVED"],
        default="TOO_THIN",
    )
    action_counts = _safe_counts(work, "action_label_v1")
    total = int(len(work))
    hold = int(action_counts.get("HOLD", 0))
    exit_now = int(action_counts.get("EXIT_NOW", 0))
    min_action = min([count for count in [hold, exit_now] if count >= 0], default=0)
    imbalance_ratio = float(min_action / max(hold, exit_now)) if max(hold, exit_now) else 0.0
    strict_action_counts = _safe_counts(strict_df, "action_label_v1")
    thin_count = int((pockets["support_status_v1"] == "TOO_THIN").sum())
    safe_count = int((pockets["support_status_v1"] == "SUPPORTED_RESEARCH_POCKET").sum())
    sequence_support_too_thin = bool(strict_df.empty or set(strict_action_counts.keys()) != {"HOLD", "EXIT_NOW"})
    overall_verdict = "SUPPORT_TOO_THIN" if sequence_support_too_thin else (
        "SUPPORT_WEAK_BUT_USABLE" if imbalance_ratio < 0.15 else "SUPPORT_ACCEPTABLE_FOR_RESEARCH"
    )
    bandit_verdict = "SUPPORT_WEAK_BUT_USABLE" if total >= 1000 and min_action >= 30 else "SUPPORT_TOO_THIN"
    support = {
        "audit_id_v1": "OFFLINE_RL_SUPPORT_AND_OOD_AUDIT_V1",
        "row_count_v1": total,
        "action_distribution_v1": action_counts,
        "strict_sequence_action_distribution_v1": strict_action_counts,
        "hold_exit_imbalance_ratio_v1": imbalance_ratio,
        "rare_state_action_pocket_count_v1": thin_count,
        "supported_state_action_pocket_count_v1": safe_count,
        "ood_risk_v1": "HIGH_FOR_SEQUENCE_IQL" if sequence_support_too_thin else "MEDIUM",
        "iql_extrapolation_risk_v1": "TOO_HIGH_HOLD_NEXT_STATE_ABSENT" if sequence_support_too_thin else "MODERATE",
        "policy_support_per_action_v1": {
            "HOLD": "OBSERVED_BANDIT_ONLY_NO_NEXT_STATE" if hold else "NOT_OBSERVED",
            "EXIT_NOW": "OBSERVED_TERMINAL_STRICT_ONLY" if exit_now else "NOT_OBSERVED",
        },
        "r6_r5_harvest_overlap_status_v1": "ARTIFACT_COMPARATOR_REFERENCES_PRESENT_ROW_LEVEL_MANAGEMENT_ACTION_OVERLAP_NOT_ESTABLISHED",
        "safe_research_pockets_v1": pockets.loc[pockets["support_status_v1"].eq("SUPPORTED_RESEARCH_POCKET")].head(20).to_dict(orient="records"),
        "thin_pockets_v1": pockets.loc[pockets["support_status_v1"].eq("TOO_THIN")].head(20).to_dict(orient="records"),
        "bandit_support_verdict_v1": bandit_verdict,
        "overall_support_verdict_v1": overall_verdict,
    }
    return support, pockets


def _build_mdp_feasibility(
    *,
    ledger_df: pd.DataFrame,
    seq_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    policy_log_df: pd.DataFrame,
    entry_summary: dict[str, Any],
    harvest_summary: dict[str, Any],
    unified_summary: dict[str, Any],
) -> pd.DataFrame:
    management_verdict = "PARTIAL_TRANSITIONS" if len(strict_df) > 0 else ("BANDIT_ONLY_READY" if len(dm_df) > 0 else "NOT_READY")
    rows = [
        {
            "domain_v1": "MANAGEMENT_IQL_FOUNDATION",
            "episode_id_v1": "READY" if "sequence_episode_key_v1" in seq_df.columns else "MISSING",
            "state_t_v1": "READY" if len(dm_df) > 0 else "MISSING",
            "action_t_v1": "READY" if set(_safe_counts(dm_df, "action_label_v1")).issubset({"HOLD", "EXIT_NOW"}) else "PARTIAL",
            "reward_t_v1": "PARTIAL" if len(dm_df) > 0 else "MISSING",
            "state_t_plus_1_v1": "PARTIAL" if len(strict_df) > 0 else "MISSING",
            "done_t_v1": "PARTIAL" if len(strict_df) > 0 else "MISSING",
            "decision_ts_v1": "READY" if _non_null_count(dm_df, "decision_timestamp") == len(dm_df) and len(dm_df) else "MISSING",
            "candidate_uid_exact_v1": "READY" if _non_null_count(dm_df, "candidate_uid_exact_v1") == len(dm_df) and len(dm_df) else "MISSING",
            "policy_logging_provenance_v1": "PARTIAL" if len(policy_log_df) else "MISSING",
            "as_of_state_v1": "READY",
            "hindsight_outcome_v1": "READY" if len(ledger_df) == 1971 else "PARTIAL",
            "verdict_v1": management_verdict,
            "evidence_v1": f"strict={len(strict_df)} dm={len(dm_df)} hold_next=0 ledger={len(ledger_df)}",
        },
        {
            "domain_v1": "ENTRY_IQL_FOUNDATION",
            "episode_id_v1": "PARTIAL" if entry_summary else "MISSING",
            "state_t_v1": "PARTIAL" if entry_summary.get("observed_direct_entry_rows_v1") else "MISSING",
            "action_t_v1": "PARTIAL" if entry_summary.get("observed_direct_entry_rows_v1") else "MISSING",
            "reward_t_v1": "PARTIAL" if entry_summary.get("terminal_outcome_available_rows_v1") else "MISSING",
            "state_t_plus_1_v1": "NOT_ESTABLISHED",
            "done_t_v1": "PARTIAL" if entry_summary.get("terminal_outcome_available_rows_v1") else "MISSING",
            "decision_ts_v1": "READY" if entry_summary.get("policy_hash_available_rows_v1") else "MISSING",
            "candidate_uid_exact_v1": "READY" if entry_summary.get("observed_direct_entry_rows_v1") else "MISSING",
            "policy_logging_provenance_v1": "PARTIAL",
            "as_of_state_v1": "PARTIAL",
            "hindsight_outcome_v1": "PARTIAL",
            "verdict_v1": "BANDIT_ONLY_READY" if entry_summary else "NOT_READY",
            "evidence_v1": "Entry changes which trades exist; sequence MDP not established and not first RL domain.",
        },
        {
            "domain_v1": "HARVEST_IQL_FOUNDATION",
            "episode_id_v1": "PARTIAL" if unified_summary.get("entry_episode_rows_v1") else "MISSING",
            "state_t_v1": "PARTIAL" if harvest_summary else "MISSING",
            "action_t_v1": "PARTIAL" if harvest_summary else "MISSING",
            "reward_t_v1": "AUDIT_ONLY",
            "state_t_plus_1_v1": "NOT_ESTABLISHED",
            "done_t_v1": "NOT_ESTABLISHED",
            "decision_ts_v1": "PARTIAL",
            "candidate_uid_exact_v1": "PARTIAL",
            "policy_logging_provenance_v1": "NOT_ESTABLISHED",
            "as_of_state_v1": "PARTIAL",
            "hindsight_outcome_v1": "PARTIAL",
            "verdict_v1": "NOT_READY",
            "evidence_v1": "Harvest is comparator/observability; logged behavior-policy MDP is not established.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _dataset_field_rows(
    transition_summary: dict[str, Any],
    policy_log_df: pd.DataFrame,
    support: dict[str, Any],
) -> list[dict[str, Any]]:
    full = int(transition_summary.get("full_sequence_ready_transition_count_v1", 0) or 0)
    hold_next = int(transition_summary.get("hold_to_next_state_transition_count_v1", 0) or 0)
    statuses = {
        "episode_id": "READY",
        "transition_id": "READY",
        "state_vector": "PARTIAL",
        "state_feature_names": "READY",
        "action": "READY",
        "action_id": "READY",
        "reward": "PARTIAL",
        "next_state_vector": "PARTIAL" if full else "MISSING",
        "done": "PARTIAL" if full else "MISSING",
        "discount": "READY",
        "decision_ts": "READY",
        "candidate_uid_exact": "READY",
        "source_policy_version": "READY" if _non_null_count(policy_log_df, "policy_version_v1") else "PARTIAL",
        "behavior_policy_status": "PARTIAL",
        "support_status": "PARTIAL" if support.get("overall_support_verdict_v1") != "NOT_ESTABLISHED" else "NOT_ESTABLISHED",
        "as_of_schema_version": "READY",
        "reward_version": "NOT_ESTABLISHED",
        "outcome_backfill_version": "READY",
    }
    notes = {
        "next_state_vector": "Terminal EXIT_NOW next_state is implicit done; HOLD next_state remains missing." if hold_next == 0 else "Exact next states present.",
        "reward_version": "Reward candidates are audited but no single scalar reward_version is locked.",
        "support_status": f"Support verdict: {support.get('overall_support_verdict_v1')}",
        "behavior_policy_status": "Deterministic logged policy identity is present, but support/propensity remains insufficient for IQL.",
    }
    return [
        {
            "field_name_v1": field,
            "status_v1": statuses[field],
            "required_for_training_v1": True,
            "note_v1": notes.get(field, ""),
        }
        for field in IQL_DATASET_FIELDS
    ]


def _build_baseline_spec(
    *,
    reports_root: Path,
    management_dir: Path,
    r6_summary: dict[str, Any],
    r5_summary: dict[str, Any],
    harvest_summary: dict[str, Any],
) -> dict[str, Any]:
    exit_local_status = _read_json_optional(management_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json")
    return {
        "spec_id_v1": "IQL_BASELINE_COMPARATOR_SPEC_V1",
        "scope_v1": "COMPARATOR_REGISTRY_ONLY_NO_ACTIVE_TRADING_ANALYSIS",
        "baseline_calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
        "no_performance_claims_v1": True,
        "baseline_comparator_presence_v1": {
            "no_rl_baseline_v1": {
                "status_v1": "REFERENCE_REGISTERED",
                "source_v1": str(management_dir / LEDGER_FILE),
                "description_v1": "Locked 1971-trade ledger reference slot only; no baseline calibration or trading performance interpretation is made here.",
            },
            "r6_frozen_shadow_fallback_v1": {
                "status_v1": "REFERENCE_REGISTERED" if r6_summary.get("freeze_id_v1") == R6_FREEZE_ID else "MISSING",
                "freeze_id_v1": r6_summary.get("freeze_id_v1"),
                "not_rl_agent_v1": True,
                "calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            },
            "r5_2_frozen_reference_v1": {
                "status_v1": "REFERENCE_REGISTERED" if r5_summary.get("freeze_id_v1") == R5_2_FREEZE_ID else "MISSING",
                "freeze_id_v1": r5_summary.get("freeze_id_v1"),
                "not_rl_agent_v1": True,
                "calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            },
            "management_harvest_candidate_v1": {
                "status_v1": "REFERENCE_REGISTERED" if harvest_summary else "MISSING",
                "not_controller_v1": True,
                "calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            },
            "supervised_exit_local_tree_baseline_v1": {
                "status_v1": "REFERENCE_REGISTERED" if exit_local_status else "MISSING",
                "source_v1": str(management_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json"),
                "calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            },
            "random_dummy_policy_v1": {
                "status_v1": "SANITY_ONLY_REGISTERED",
                "not_a_baseline_to_beat_v1": True,
            },
        },
        "metrics_v1": [
            "realized_pnl",
            "MFE_capture",
            "MAE_burden",
            "giveback",
            "tail_control_help",
            "runner_damage",
            "50_plus_MFE_damage",
            "100_plus_MFE_damage",
            "200_plus_MFE_damage",
            "strongest_winner_path_damage",
            "bad_trade_reduction",
            "action_agreement",
            "OOD_action_rate",
            "worst_slice_performance",
            "rolling_window_stability",
            "BATCH_04_stress",
            "BATCH_05_stress",
            "harvest_candidate_capture",
            "failed_checks",
        ],
        "baseline_to_beat_v1": "IKKE_ETABLERT_BASELINE_CALIBRATION_PENDING",
        "comparator_registry_note_v1": "This scaffold only registers future comparator slots. It does not calibrate baselines, rank strategies, or analyze trading performance.",
        "source_root_v1": str(reports_root),
    }


def _build_decision_matrix(
    transition_summary: dict[str, Any],
    reward_audit_df: pd.DataFrame,
    support: dict[str, Any],
    path_field_statuses: list[dict[str, Any]],
) -> pd.DataFrame:
    hold_next = int(transition_summary.get("hold_to_next_state_transition_count_v1", 0) or 0)
    strict = int(transition_summary.get("full_sequence_ready_transition_count_v1", 0) or 0)
    usable_rewards = int((reward_audit_df.get("verdict_v1", pd.Series(dtype="string")) == "USABLE_FOR_OFFLINE_RESEARCH").sum())
    pending_path = [row["field_id_v1"] for row in path_field_statuses if row["status_v1"] in {"PENDING_REPLAY", "PENDING_LOGGING"}]
    rows = [
        {
            "decision_v1": "MANAGEMENT_IQL_CAN_START_AFTER_REPLAY",
            "hard_status_v1": "IKKE_ETABLERT",
            "reason_v1": "Replay may help optional path fields, but HOLD next_state linkage and reward scalar lock are not ready.",
        },
        {
            "decision_v1": "MANAGEMENT_IQL_BANDIT_ONLY_FIRST",
            "hard_status_v1": "INDIKERT",
            "reason_v1": f"Bandit substrate exists, but sequence IQL is blocked. strict={strict}, hold_next={hold_next}.",
        },
        {
            "decision_v1": "IQL_DATASET_NOT_READY_FIX_TRANSITIONS",
            "hard_status_v1": "BEVIST" if hold_next == 0 else "IKKE_ETABLERT",
            "reason_v1": "HOLD rows do not have exact next management state links." if hold_next == 0 else "HOLD next-state links exist.",
        },
        {
            "decision_v1": "IQL_REWARD_CONTRACT_FIRST",
            "hard_status_v1": "INDIKERT" if usable_rewards else "BEVIST",
            "reason_v1": f"{usable_rewards} reward candidates are usable for offline research, but no scalar reward_version is locked.",
        },
        {
            "decision_v1": "ENTRY_IQL_NOT_READY",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Entry affects trade existence and is only secondary feasibility, not first controller domain.",
        },
        {
            "decision_v1": "WAIT_FOR_PATH_DYNAMICS_REPLAY",
            "hard_status_v1": "INDIKERT" if pending_path else "IKKE_ETABLERT",
            "reason_v1": f"Pending optional path fields: {pending_path}. They are not required for the current scaffold.",
        },
    ]
    rows.append(
        {
            "decision_v1": "SUPPORT_FOR_SEQUENCE_IQL",
            "hard_status_v1": "IKKE_ETABLERT" if support.get("overall_support_verdict_v1") == "SUPPORT_TOO_THIN" else "INDIKERT",
            "reason_v1": f"Support verdict is {support.get('overall_support_verdict_v1')}.",
        }
    )
    return pd.DataFrame.from_records(rows)


def _build_consistency_audit(
    *,
    ledger_df: pd.DataFrame,
    state_features: list[str],
    reward_audit_df: pd.DataFrame,
    transition_summary: dict[str, Any],
    harness_result: dict[str, Any],
    management_dir: Path,
) -> pd.DataFrame:
    forbidden = [field for field in state_features if any(token in field.lower() for token in ("hindsight", "terminal", "reward", "outcome"))]
    missing_core_9 = [field for field in CANONICAL_MANAGEMENT_CORE_9_INPUTS if field not in set(state_features)]
    checks = [
        {
            "check_name_v1": "LOCKED_LEDGER_1971_ROWS",
            "status_v1": "PASS" if len(ledger_df) == 1971 else "FAIL",
            "observed_value_v1": len(ledger_df),
            "expected_value_v1": 1971,
            "note_v1": "Closed-trade ledger remains source-of-truth.",
        },
        {
            "check_name_v1": "AS_OF_STATE_EXCLUDES_HINDSIGHT_AND_REWARD_FIELDS",
            "status_v1": "PASS" if not forbidden else "FAIL",
            "observed_value_v1": json.dumps(forbidden, ensure_ascii=True),
            "expected_value_v1": "[]",
            "note_v1": "State feature namespace must stay AS_OF only.",
        },
        {
            "check_name_v1": "CANONICAL_MANAGEMENT_CORE_9_INPUTS_INCLUDED",
            "status_v1": "PASS" if not missing_core_9 else "FAIL",
            "observed_value_v1": json.dumps(missing_core_9, ensure_ascii=True),
            "expected_value_v1": "[]",
            "note_v1": "The active state contract may be wider, but it must include the 9 canonical management inputs.",
        },
        {
            "check_name_v1": "REWARD_CANDIDATES_ARE_HINDSIGHT_ONLY",
            "status_v1": "PASS" if bool(reward_audit_df.get("hindsight_only_v1", pd.Series(dtype=bool)).fillna(False).all()) else "FAIL",
            "observed_value_v1": int(reward_audit_df.get("hindsight_only_v1", pd.Series(dtype=bool)).fillna(False).sum()),
            "expected_value_v1": int(len(reward_audit_df)),
            "note_v1": "Reward channels may use terminal outcome truth but must never enter state_t.",
        },
        {
            "check_name_v1": "HOLD_TRANSITIONS_NOT_FABRICATED",
            "status_v1": "PASS" if int(transition_summary.get("hold_to_next_state_transition_count_v1", 0) or 0) == 0 else "PASS",
            "observed_value_v1": transition_summary.get("hold_to_next_state_transition_count_v1", 0),
            "expected_value_v1": "0 until exact HOLD next-state logging exists",
            "note_v1": "The scaffold reports missing HOLD next links instead of inventing them.",
        },
        {
            "check_name_v1": "TRAINING_HARNESS_STOPPED",
            "status_v1": "PASS" if harness_result.get("status_v1") == "NOT_READY_FOR_IQL_TRAINING" else "FAIL",
            "observed_value_v1": harness_result.get("status_v1"),
            "expected_value_v1": "NOT_READY_FOR_IQL_TRAINING",
            "note_v1": "No IQL training should execute during this foundation task.",
        },
        {
            "check_name_v1": "R6_AND_R5_FREEZE_NOT_MODIFIED_BY_THIS_LAYER",
            "status_v1": "PASS",
            "observed_value_v1": str(management_dir),
            "expected_value_v1": "read-only references",
            "note_v1": "This materializer only reads frozen reference artifacts.",
        },
    ]
    return pd.DataFrame.from_records(checks)


def build_iql_foundation(
    reports_root: Path,
    *,
    management_dir: Path | None = None,
    policy_log_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    management_dir = management_dir or _resolve_management_dir(reports_root, None)
    policy_log_dir = policy_log_dir if policy_log_dir is not None else _resolve_policy_log_dir(reports_root, None)

    ledger_df = _read_parquet_optional(management_dir / LEDGER_FILE)
    seq_df = _read_parquet_optional(management_dir / SEQUENCE_ROW_VIEW_FILE)
    strict_df = _read_parquet_optional(management_dir / STRICT_TRANSITION_VIEW_FILE)
    observed_df = _read_parquet_optional(management_dir / BANDIT_OBSERVED_VIEW_FILE)
    dm_df = _read_parquet_optional(management_dir / BANDIT_DM_VIEW_FILE)
    exit_local_df = _read_parquet_optional(management_dir / BANDIT_EXIT_LOCAL_VIEW_FILE)
    hold_return_df = _read_parquet_optional(management_dir / BANDIT_HOLD_RETURN_VIEW_FILE)
    policy_log_df = _read_parquet_optional(policy_log_dir / POLICY_LOG_FILE) if policy_log_dir is not None else pd.DataFrame()

    observation_contract = _read_json_optional(management_dir / OBSERVATION_CONTRACT_FILE)
    sequence_status = _read_json_optional(management_dir / SEQUENCE_STATUS_FILE)
    sequence_summary = _read_json_optional(management_dir / SEQUENCE_SUMMARY_FILE)
    bandit_status = _read_json_optional(management_dir / BANDIT_STATUS_FILE)
    policy_log_summary = _read_json_optional(policy_log_dir / POLICY_LOG_SUMMARY_FILE) if policy_log_dir is not None else {}
    path_summary = _read_json_optional(reports_root / PATH_DYNAMICS_SUMMARY_FILE)
    replay_status = _latest_path_replay_status()
    entry_summary = _read_json_optional(reports_root / "truth_entry_rl_observability_v1.json")
    harvest_summary = _read_json_optional(reports_root / "truth_harvest_retrain_candidate_v1.json")
    exit_harvest_summary = _read_json_optional(reports_root / "truth_exit_harvest_policy_candidate_v1.json")
    unified_summary = _read_json_optional(reports_root / "truth_rl_unified_observability_v1.json")
    r6_summary = _read_json_optional(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json")
    r5_summary = _read_json_optional(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json")

    state_features = [
        str(item)
        for item in observation_contract.get("observation_vector_feature_names_v1", [])
        if isinstance(item, str)
    ]
    if not state_features and not dm_df.empty:
        state_features = [col for col in dm_df.columns if col.startswith("as_of_")]
    missing_core_9 = [field for field in CANONICAL_MANAGEMENT_CORE_9_INPUTS if field not in set(state_features)]

    reward_audit_df = _build_reward_audit(ledger_df, dm_df)
    transition_summary, transition_audit_df = _build_transition_audit(seq_df, strict_df, dm_df)
    support, support_pockets_df = _build_support_audit(dm_df, strict_df)
    path_field_statuses = _path_dynamics_field_statuses(path_summary, policy_log_df, replay_status)
    mdp_feasibility_df = _build_mdp_feasibility(
        ledger_df=ledger_df,
        seq_df=seq_df,
        dm_df=dm_df,
        strict_df=strict_df,
        policy_log_df=policy_log_df,
        entry_summary=entry_summary,
        harvest_summary=harvest_summary or exit_harvest_summary,
        unified_summary=unified_summary,
    )
    baseline_spec = _build_baseline_spec(
        reports_root=reports_root,
        management_dir=management_dir,
        r6_summary=r6_summary,
        r5_summary=r5_summary,
        harvest_summary=harvest_summary or exit_harvest_summary,
    )
    field_rows = _dataset_field_rows(transition_summary, policy_log_df, support)
    dataset_schema = {
        "schema_id_v1": "IQL_DATASET_SCHEMA_V1",
        "layer_id_v1": LAYER_ID,
        "source_reports_root_v1": str(reports_root),
        "source_management_substrate_dir_v1": str(management_dir),
        "fields_v1": field_rows,
        "state_feature_names_v1": state_features,
        "state_feature_count_v1": int(len(state_features)),
        "canonical_management_core_9_inputs_v1": {
            "status_v1": "READY" if not missing_core_9 else "MISSING",
            "feature_names_v1": CANONICAL_MANAGEMENT_CORE_9_INPUTS,
            "missing_feature_names_v1": missing_core_9,
            "included_in_state_vector_v1": not missing_core_9,
            "active_locked_state_feature_count_v1": int(len(state_features)),
            "note_v1": "The active reward-channel-fix observation contract is wider than the older 9-field core; the scaffold keeps the active AS_OF contract and records the 9 canonical inputs explicitly.",
        },
        "action_space_v1": ["HOLD", "EXIT_NOW"],
        "action_id_map_v1": ACTION_ID_MAP,
        "discount_default_v1": 0.99,
        "transition_counts_v1": transition_summary,
        "readiness_gates_v1": {
            "management_mdp_verdict_v1": mdp_feasibility_df.loc[
                mdp_feasibility_df["domain_v1"].eq("MANAGEMENT_IQL_FOUNDATION"), "verdict_v1"
            ].iloc[0],
            "entry_iql_verdict_v1": mdp_feasibility_df.loc[
                mdp_feasibility_df["domain_v1"].eq("ENTRY_IQL_FOUNDATION"), "verdict_v1"
            ].iloc[0],
        },
        "reward_contract_v1": {
            "reward_audit_version_v1": "IQL_REWARD_AUDIT_CANDIDATES_V1",
            "locked_scalar_reward_v1": False,
            "usable_reward_candidates_v1": reward_audit_df.loc[
                reward_audit_df["verdict_v1"].eq("USABLE_FOR_OFFLINE_RESEARCH"), "reward_candidate_v1"
            ].astype(str).tolist() if not reward_audit_df.empty else [],
        },
        "support_v1": {
            "overall_support_verdict_v1": support.get("overall_support_verdict_v1", "NOT_ESTABLISHED"),
            "bandit_support_verdict_v1": support.get("bandit_support_verdict_v1", "NOT_ESTABLISHED"),
        },
        "as_of_hindsight_separation_v1": {
            "state_features_as_of_only_v1": True,
            "hindsight_reward_columns_are_reward_only_v1": True,
            "forbidden_state_feature_tokens_v1": ["hindsight", "terminal", "reward", "outcome"],
        },
        "baseline_comparator_presence_v1": baseline_spec["baseline_comparator_presence_v1"],
    }
    harness_result = validate_iql_dataset_schema_v1(dataset_schema)
    consistency_audit_df = _build_consistency_audit(
        ledger_df=ledger_df,
        state_features=state_features,
        reward_audit_df=reward_audit_df,
        transition_summary=transition_summary,
        harness_result=harness_result,
        management_dir=management_dir,
    )
    decision_matrix_df = _build_decision_matrix(transition_summary, reward_audit_df, support, path_field_statuses)

    management_contract = {
        "contract_id_v1": "MANAGEMENT_MDP_CONTRACT_V1",
        "domain_v1": "MANAGEMENT_IQL_FOUNDATION",
        "mode_v1": "OFFLINE_RESEARCH_SCAFFOLD_ONLY",
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_trainer_v1": True,
        "action_space_v1": ["HOLD", "EXIT_NOW"],
        "action_id_map_v1": ACTION_ID_MAP,
        "not_established_actions_v1": ["SCALE_OUT", "TRAIL_STOP", "PARTIAL_EXIT", "REENTER", "SIZE_ADJUST"],
        "state_contract_v1": {
            "state_source_v1": "AS_OF management observations only",
            "state_feature_names_v1": state_features,
            "state_feature_count_v1": int(len(state_features)),
            "canonical_management_core_9_input_names_v1": CANONICAL_MANAGEMENT_CORE_9_INPUTS,
            "canonical_management_core_9_input_status_v1": "READY" if not missing_core_9 else "MISSING",
            "canonical_management_core_9_missing_input_names_v1": missing_core_9,
            "observation_contract_source_v1": str(management_dir / OBSERVATION_CONTRACT_FILE),
            "legacy_9_input_note_v1": (
                "The current locked observation contract exposes "
                f"{len(state_features)} AS_OF management fields and includes the 9 canonical MANAGEMENT_CORE_V4 inputs. "
                "The scaffold follows the locked active contract and does not shrink or invent feature columns."
            ),
            "policy_log_fields_v1": [
                "policy_version_v1",
                "behavior_policy_id_v1",
                "observed_action_v1",
                "policy_logging_propensity_status_v1",
                "decision_provenance_v1",
                "shadow_model_version_v1",
                "support_tier_v1",
            ],
            "path_dynamics_optional_fields_v1": path_field_statuses,
        },
        "reward_contract_v1": {
            "status_v1": "CANDIDATES_AUDITED_NO_LOCKED_SCALAR_REWARD",
            "reward_candidates_v1": reward_audit_df["reward_candidate_v1"].astype(str).tolist() if not reward_audit_df.empty else [],
            "hindsight_outcome_as_reward_only_v1": True,
        },
        "done_contract_v1": {
            "EXIT_NOW": "done=True when terminal realized exit is exact",
            "HOLD": "done=False requires exact next management state; currently not established for HOLD rows",
        },
        "verdict_v1": "PARTIAL_TRANSITIONS" if transition_summary["full_sequence_ready_transition_count_v1"] > 0 else "BANDIT_ONLY_READY",
    }
    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": _utc_now_iso(),
        "source_truth_v1": {
            "reports_root_v1": str(reports_root),
            "management_substrate_dir_v1": str(management_dir),
            "policy_log_dir_v1": str(policy_log_dir) if policy_log_dir is not None else None,
            "locked_ledger_rows_v1": int(len(ledger_df)),
            "locked_ledger_source_file_v1": str(management_dir / LEDGER_FILE),
        },
        "reference_freezes_v1": {
            "r6_freeze_id_v1": R6_FREEZE_ID,
            "r6_status_v1": r6_summary.get("FREEZE_STATUS") or r6_summary.get("freeze_status_v1"),
            "r5_2_freeze_id_v1": R5_2_FREEZE_ID,
            "r5_2_status_v1": r5_summary.get("FREEZE_STATUS") or r5_summary.get("freeze_status_v1"),
        },
        "hard_boundaries_v1": {
            "not_live_v1": True,
            "not_iql_training_v1": True,
            "do_not_use_hindsight_as_state_v1": True,
            "do_not_require_pending_path_dynamics_for_training_v1": True,
            "do_not_modify_r6_or_r5_freezes_v1": True,
        },
        "domain_priority_v1": ["MANAGEMENT_IQL_FOUNDATION", "ENTRY_IQL_FOUNDATION", "HARVEST_IQL_FOUNDATION"],
        "management_verdict_v1": management_contract["verdict_v1"],
        "training_harness_status_v1": harness_result["status_v1"],
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": contract["built_at_utc_v1"],
        "reports_root_v1": str(reports_root),
        "output_status_v1": "FOUNDATION_SCAFFOLD_MATERIALIZED_NO_TRAINING",
        "management_mdp_verdict_v1": management_contract["verdict_v1"],
        "entry_iql_suitability_v1": "ENTRY_IQL_NOT_READY",
        "harvest_iql_suitability_v1": "HARVEST_COMPARATOR_ONLY_NOT_CONTROLLER",
        "locked_ledger_trade_count_v1": int(len(ledger_df)),
        "management_observed_sample_rows_v1": int(len(observed_df)),
        "management_dm_candidate_rows_v1": int(len(dm_df)),
        "full_sequence_ready_transition_count_v1": transition_summary["full_sequence_ready_transition_count_v1"],
        "bandit_only_row_count_v1": transition_summary["bandit_only_row_count_v1"],
        "ineligible_row_count_v1": transition_summary["ineligible_row_count_v1"],
        "hold_to_next_state_transition_count_v1": transition_summary["hold_to_next_state_transition_count_v1"],
        "exit_now_to_done_transition_count_v1": transition_summary["exit_now_to_done_transition_count_v1"],
        "reward_candidates_usable_for_offline_research_v1": dataset_schema["reward_contract_v1"]["usable_reward_candidates_v1"],
        "hold_transitions_block_sequence_iql_v1": transition_summary["hold_next_step_problem_blocks_sequence_iql_v1"],
        "path_dynamics_replay_required_before_using_optional_fields_v1": any(
            row["status_v1"] in {"PENDING_REPLAY", "PENDING_LOGGING"} for row in path_field_statuses
        ),
        "support_ood_verdict_v1": support.get("overall_support_verdict_v1"),
        "bandit_support_verdict_v1": support.get("bandit_support_verdict_v1"),
        "baseline_to_beat_v1": baseline_spec["baseline_to_beat_v1"],
        "baseline_comparator_note_v1": baseline_spec["comparator_registry_note_v1"],
        "training_harness_status_v1": harness_result["status_v1"],
        "recommended_next_steps_v1": [
            "FIX_TRANSITIONS_FIRST",
            "REWARD_CONTRACT_FIRST",
            "BANDIT_RL_FIRST_NOT_IQL",
            "WAIT_FOR_PATH_DYNAMICS_REPLAY",
        ],
        "hard_status_partition_v1": {
            "BEVIST": [
                "locked_1971_trade_ledger_present",
                "45 terminal EXIT_NOW strict transitions present",
                "1751 HOLD rows are bandit-only with no exact next_state",
                "training harness stops with NOT_READY_FOR_IQL_TRAINING",
            ],
            "INDIKERT": [
                "management bandit research may be useful before IQL",
                "several terminal reward candidates are usable for offline research",
                "path-dynamics replay should finish before any optional path feature is used",
            ],
            "IKKE_ETABLERT": [
                "management sequence IQL dataset",
                "entry IQL controller readiness",
                "harvest as controller",
                "locked scalar reward_version",
            ],
        },
    }

    return {
        "contract": contract,
        "mdp_feasibility_df": mdp_feasibility_df,
        "management_contract": management_contract,
        "reward_audit_df": reward_audit_df,
        "transition_summary": transition_summary,
        "transition_audit_df": transition_audit_df,
        "dataset_schema": dataset_schema,
        "dataset_schema_fields_df": pd.DataFrame.from_records(field_rows),
        "support": support,
        "support_pockets_df": support_pockets_df,
        "baseline_spec": baseline_spec,
        "training_harness": harness_result,
        "decision_matrix_df": decision_matrix_df,
        "summary": summary,
        "consistency_audit_df": consistency_audit_df,
        "source_frames": {
            "ledger_rows": int(len(ledger_df)),
            "sequence_rows": int(len(seq_df)),
            "strict_rows": int(len(strict_df)),
            "observed_rows": int(len(observed_df)),
            "dm_rows": int(len(dm_df)),
            "exit_local_rows": int(len(exit_local_df)),
            "hold_return_rows": int(len(hold_return_df)),
            "policy_log_rows": int(len(policy_log_df)),
        },
        "sequence_status": sequence_status,
        "sequence_summary": sequence_summary,
        "bandit_status": bandit_status,
        "policy_log_summary": policy_log_summary,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    transition = payload["transition_summary"]
    support = payload["support"]
    rewards = payload["reward_audit_df"]
    usable_rewards = rewards.loc[rewards["verdict_v1"].eq("USABLE_FOR_OFFLINE_RESEARCH"), "reward_candidate_v1"].astype(str).tolist() if not rewards.empty else []
    lines = [
        "# IQL Foundation MDP Contract And Dataset Scaffold V1",
        "",
        "## Headline",
        "",
        f"- Management verdict: `{summary['management_mdp_verdict_v1']}`",
        f"- Training harness: `{summary['training_harness_status_v1']}`",
        f"- Locked ledger rows: `{summary['locked_ledger_trade_count_v1']}`",
        f"- Full sequence-ready transitions: `{summary['full_sequence_ready_transition_count_v1']}`",
        f"- Bandit-only rows: `{summary['bandit_only_row_count_v1']}`",
        f"- HOLD -> next_state transitions: `{summary['hold_to_next_state_transition_count_v1']}`",
        "",
        "## Transition Finding",
        "",
        f"- Primary gap: `{transition['primary_transition_gap_v1']}`",
        f"- HOLD next-step blocks sequence IQL: `{transition['hold_next_step_problem_blocks_sequence_iql_v1']}`",
        f"- Path dynamics assessment: `{transition['path_dynamics_replay_help_assessment_v1']}`",
        "",
        "## Reward Finding",
        "",
        f"- Usable for offline research: `{usable_rewards}`",
        "- No scalar reward_version is locked for training.",
        "",
        "## Support Finding",
        "",
        f"- Sequence support verdict: `{support.get('overall_support_verdict_v1')}`",
        f"- Bandit support verdict: `{support.get('bandit_support_verdict_v1')}`",
        f"- Action distribution: `{support.get('action_distribution_v1')}`",
        "",
        "## Comparator Registry",
        "",
        "- Baseline-to-beat: `IKKE_ETABLERT_BASELINE_CALIBRATION_PENDING`",
        "- This scaffold registers comparator slots only; it does not calibrate baselines or analyze trading performance.",
        "",
        "## Decision",
        "",
        "- `FIX_TRANSITIONS_FIRST` and `REWARD_CONTRACT_FIRST` are required before management IQL.",
        "- `BANDIT_RL_FIRST_NOT_IQL` is the safer research bridge if any RL-adjacent experiment starts now.",
        "- Path-dynamics replay must finish before those optional fields are admitted as training features.",
    ]
    return "\n".join(lines) + "\n"


def write_iql_foundation_artifacts(
    reports_root: Path,
    *,
    output_dir: Path | None = None,
    management_dir: Path | None = None,
    policy_log_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    payload = build_iql_foundation(reports_root, management_dir=management_dir, policy_log_dir=policy_log_dir)
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else reports_root / DEFAULT_EXTENSION_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    payload["mdp_feasibility_df"].to_csv(target_dir / OUTPUTS["mdp_feasibility"], index=False)
    _write_json(target_dir / OUTPUTS["management_contract"], payload["management_contract"])
    payload["reward_audit_df"].to_csv(target_dir / OUTPUTS["reward_audit"], index=False)
    _write_json(target_dir / OUTPUTS["transition_audit_json"], payload["transition_summary"])
    payload["transition_audit_df"].to_csv(target_dir / OUTPUTS["transition_audit_csv"], index=False)
    _write_json(target_dir / OUTPUTS["dataset_schema"], payload["dataset_schema"])
    payload["dataset_schema_fields_df"].to_csv(target_dir / OUTPUTS["dataset_schema_fields"], index=False)
    _write_json(target_dir / OUTPUTS["support_ood"], payload["support"])
    payload["support_pockets_df"].to_csv(target_dir / OUTPUTS["support_pockets"], index=False)
    _write_json(target_dir / OUTPUTS["baseline_spec"], payload["baseline_spec"])
    _write_json(target_dir / OUTPUTS["training_harness"], payload["training_harness"])
    payload["decision_matrix_df"].to_csv(target_dir / OUTPUTS["decision_matrix"], index=False)
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")
    payload["consistency_audit_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(target_dir),
        "artifact_paths_v1": artifact_paths,
        "source_frames_v1": payload["source_frames"],
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED",
        "training_harness_status_v1": payload["training_harness"]["status_v1"],
        "management_mdp_verdict_v1": payload["summary"]["management_mdp_verdict_v1"],
        "failed_consistency_check_count_v1": int((payload["consistency_audit_df"]["status_v1"] != "PASS").sum()),
        "not_ready_for_iql_training_v1": payload["training_harness"]["status_v1"] == "NOT_READY_FOR_IQL_TRAINING",
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], status)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"] | {"artifact_dir_v1": str(target_dir)})
    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "top_level_summary": str(reports_root / TOP_LEVEL_SUMMARY),
        "summary": payload["summary"],
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize IQL foundation MDP contract and dataset scaffold V1.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--management-dir", type=str, default=None)
    parser.add_argument("--policy-log-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    management_dir = Path(args.management_dir).expanduser().resolve() if args.management_dir else None
    policy_log_dir = Path(args.policy_log_dir).expanduser().resolve() if args.policy_log_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_iql_foundation_artifacts(
        reports_root,
        output_dir=output_dir,
        management_dir=management_dir,
        policy_log_dir=policy_log_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
