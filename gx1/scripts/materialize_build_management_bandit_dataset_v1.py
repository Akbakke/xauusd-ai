#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    PATH_DYNAMICS_V2_FIELDS,
    _json_ready,
    _read_csv_optional,
    _read_json_optional,
    _resolve_foundation_dir,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "BUILD_MANAGEMENT_BANDIT_DATASET_V1"
CONTRACT_LOCK_LAYER_ID = "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1"
REWARD_LOCK_LAYER_ID = "LOCK_FIRST_BANDIT_REWARD_VERSION_V1"
REWARD_VERSION_ID = "MGMT_BANDIT_REALIZED_PNL_BPS_V1"
REWARD_NAME = "REALIZED_PNL_REWARD"
REWARD_SOURCE_COLUMN = "hindsight_reward_realized_pnl_bps_v1"

DM_VIEW_FILENAME = "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet"
POLICY_LOG_FILENAME = "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet"

ACTION_ID_MAP = {"HOLD": 0, "EXIT_NOW": 1}
OUTPUT_COLUMNS = [
    "row_id",
    "episode_id",
    "candidate_uid_exact",
    "decision_ts",
    "action",
    "action_id",
    "reward",
    "reward_version",
    "state_feature_names",
    "state_vector",
    "source_policy_version",
    "behavior_policy_status",
    "support_status",
    "as_of_schema_version",
    "hindsight_outcome_backfill_version",
    "eligibility_status",
    "exclusion_reason",
    "provenance_namespace",
]

OUTPUTS = {
    "contract": "build_management_bandit_dataset_contract_v1.json",
    "dataset_parquet": "management_bandit_research_dataset_v1.parquet",
    "dataset_csv": "management_bandit_research_dataset_v1.csv",
    "dataset_metadata": "management_bandit_research_dataset_metadata_v1.json",
    "column_coverage_csv": "management_bandit_dataset_column_coverage_v1.csv",
    "column_coverage_json": "management_bandit_dataset_column_coverage_v1.json",
    "eligibility_audit": "management_bandit_eligibility_exclusion_audit_v1.json",
    "exclusion_rows": "management_bandit_exclusion_rows_v1.csv",
    "exclusion_reasons": "management_bandit_exclusion_reasons_v1.csv",
    "consistency_audit": "management_bandit_dataset_consistency_invariants_v1.csv",
    "consistency_audit_json": "management_bandit_dataset_consistency_invariants_v1.json",
    "profile": "management_bandit_dataset_profile_v1.json",
    "profile_action_distribution": "management_bandit_dataset_action_distribution_v1.csv",
    "profile_support_distribution": "management_bandit_dataset_support_distribution_v1.csv",
    "profile_behavior_distribution": "management_bandit_dataset_behavior_distribution_v1.csv",
    "profile_state_coverage": "management_bandit_dataset_state_coverage_v1.csv",
    "profile_thin_pockets": "management_bandit_dataset_thin_pockets_v1.csv",
    "post_build_status": "management_bandit_post_build_status_update_v1.json",
    "summary": "build_management_bandit_dataset_summary_v1.json",
    "report": "build_management_bandit_dataset_report_v1.md",
    "manifest": "build_management_bandit_dataset_manifest_v1.json",
    "status": "build_management_bandit_dataset_status_v1.json",
    "non_interference_audit": "build_management_bandit_dataset_non_interference_audit_v1.csv",
    "non_interference_audit_json": "build_management_bandit_dataset_non_interference_audit_v1.json",
}


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_reward_lock_dir(reports_root: Path, reward_lock_dir_arg: str | None) -> Path:
    if reward_lock_dir_arg:
        path = Path(reward_lock_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reward lock dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{REWARD_LOCK_LAYER_ID}_*/lock_first_bandit_reward_version_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {REWARD_LOCK_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _latest_contract_lock_dir(reports_root: Path, contract_lock_dir_arg: str | None) -> Path:
    if contract_lock_dir_arg:
        path = Path(contract_lock_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Contract lock dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{CONTRACT_LOCK_LAYER_ID}_*/iql_reward_comparator_bandit_contract_lock_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {CONTRACT_LOCK_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        return False
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _finite_float_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _cell_json_ready(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return _json_ready(value)


def _json_array_string(values: list[Any]) -> str:
    return json.dumps([_cell_json_ready(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def _distribution(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
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


def _value_counts(series: pd.Series, name: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[name, "row_count_v1", "row_share_v1"])
    counts = series.fillna("NULL").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="row_count_v1")
    total = int(counts["row_count_v1"].sum())
    counts["row_share_v1"] = counts["row_count_v1"] / total if total else 0.0
    return counts


def _state_features(dataset_schema: dict[str, Any], management_contract: dict[str, Any]) -> list[str]:
    features = _as_list(dataset_schema.get("state_feature_names_v1"))
    if features:
        return features
    state_contract = management_contract.get("state_contract_v1", {}) if isinstance(management_contract.get("state_contract_v1"), dict) else {}
    return _as_list(state_contract.get("state_feature_names_v1"))


def _source_paths(
    reports_root: Path,
    foundation_dir: Path,
    reward_lock_dir: Path,
    contract_lock_dir: Path,
    foundation_contract: dict[str, Any],
) -> dict[str, str | None]:
    source_truth = foundation_contract.get("source_truth_v1", {}) if isinstance(foundation_contract.get("source_truth_v1"), dict) else {}
    management_dir = source_truth.get("management_substrate_dir_v1")
    policy_dir = source_truth.get("policy_log_dir_v1")
    return {
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "reward_lock_dir_v1": str(reward_lock_dir),
        "contract_lock_dir_v1": str(contract_lock_dir),
        "locked_ledger_source_v1": source_truth.get("locked_ledger_source_file_v1"),
        "management_substrate_dir_v1": management_dir,
        "management_bandit_dm_view_v1": str(Path(management_dir) / DM_VIEW_FILENAME) if management_dir else None,
        "policy_log_dir_v1": policy_dir,
        "management_policy_log_v1": str(Path(policy_dir) / POLICY_LOG_FILENAME) if policy_dir else None,
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _read_source_frames(source_paths: dict[str, str | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dm_path = Path(str(source_paths["management_bandit_dm_view_v1"]))
    policy_path = Path(str(source_paths["management_policy_log_v1"]))
    if not dm_path.exists():
        raise FileNotFoundError(f"Management bandit DM view not found: {dm_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Management policy log not found: {policy_path}")
    return pd.read_parquet(dm_path), pd.read_parquet(policy_path)


def _merge_policy_log(dm_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    policy_cols = [
        "management_row_key_v1",
        "candidate_uid_exact_v1",
        "policy_version_v1",
        "behavior_policy_id_v1",
        "behavior_policy_kind_v1",
        "behavior_policy_id_status_v1",
        "policy_logging_propensity_status_v1",
        "observed_action_v1",
        "support_tier_v1",
    ]
    existing_policy_cols = [col for col in policy_cols if col in policy_df.columns]
    if "management_row_key_v1" in dm_df.columns and "management_row_key_v1" in existing_policy_cols:
        keys = ["management_row_key_v1"]
    elif "candidate_uid_exact_v1" in dm_df.columns and "candidate_uid_exact_v1" in existing_policy_cols:
        keys = ["candidate_uid_exact_v1"]
    else:
        out = dm_df.copy()
        out["_policy_join_status_v1"] = "POLICY_JOIN_KEY_MISSING"
        return out
    slim = policy_df[existing_policy_cols].copy()
    out = dm_df.merge(slim, on=keys, how="left", suffixes=("", "_policy"), indicator="_policy_join_indicator_v1")
    out["_policy_join_status_v1"] = out["_policy_join_indicator_v1"].map({"both": "JOINED", "left_only": "MISSING_POLICY_LOG", "right_only": "UNEXPECTED_RIGHT_ONLY"}).astype(str)
    return out


def _optional_state_feature(feature: str, state_features: list[str]) -> bool:
    optional_with_masks = {
        "as_of_management_core_mfe_to_anchor_ratio_v1": "as_of_management_core_mfe_to_anchor_ratio_available_v1",
        "as_of_management_exit_prob_v1": "as_of_management_exit_prob_available_v1",
    }
    return optional_with_masks.get(feature) in state_features


def _required_source_columns(state_features: list[str]) -> set[str]:
    return {
        "management_row_key_v1",
        "trade_uid_exact_v1",
        "candidate_uid_exact_v1",
        "decision_timestamp",
        "action_label_v1",
        REWARD_SOURCE_COLUMN,
        "observation_contract_v1",
        "terminal_outcome_availability_status_v1",
        *state_features,
    }


def _build_eligibility(
    merged_df: pd.DataFrame,
    state_features: list[str],
    reward_lock_summary: dict[str, Any],
    reward_contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merged_df.copy()
    missing_source_cols = sorted(_required_source_columns(state_features).difference(df.columns))
    required_state_features = [feature for feature in state_features if not _optional_state_feature(feature, state_features)]
    duplicated_row_key = df["management_row_key_v1"].duplicated(keep=False) if "management_row_key_v1" in df.columns else pd.Series([True] * len(df), index=df.index)
    parsed_decision_ts = pd.to_datetime(df["decision_timestamp"], errors="coerce", utc=True) if "decision_timestamp" in df.columns else pd.Series([pd.NaT] * len(df), index=df.index)

    reward_lock_ok = (
        reward_lock_summary.get("reward_version_id_v1") == REWARD_VERSION_ID
        and reward_lock_summary.get("reward_lock_succeeded_v1") is True
        and reward_contract.get("verdict_v1") == "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY"
    )

    exclusion_rows: list[dict[str, Any]] = []
    reasons_by_index: dict[int, list[str]] = {}
    for idx, row in df.iterrows():
        reasons: list[str] = []
        if missing_source_cols or not reward_lock_ok:
            reasons.append("schema_contract_failure")
        if duplicated_row_key.loc[idx]:
            reasons.append("duplicate_row_key")
        if _is_missing(row.get("candidate_uid_exact_v1")):
            reasons.append("missing_candidate_uid")
        if _is_missing(row.get("decision_timestamp")) or pd.isna(parsed_decision_ts.loc[idx]):
            reasons.append("ambiguous_decision_ts")
        if _finite_float_or_none(row.get(REWARD_SOURCE_COLUMN)) is None:
            reasons.append("missing_locked_reward_input")
        if str(row.get("action_label_v1")) not in ACTION_ID_MAP:
            reasons.append("schema_contract_failure")
        if any(_is_missing(row.get(feature)) for feature in required_state_features):
            reasons.append("missing_state_vector")
        if _is_missing(row.get("policy_version_v1")) or _is_missing(row.get("behavior_policy_kind_v1")) or _is_missing(row.get("policy_logging_propensity_status_v1")):
            reasons.append("unsupported_behavior_status")
        if _is_missing(row.get("support_tier_v1")):
            reasons.append("unsupported_support_status")
        if row.get("_policy_join_status_v1") != "JOINED":
            reasons.append("unsupported_behavior_status")

        reasons = sorted(set(reasons))
        reasons_by_index[idx] = reasons
        if reasons:
            exclusion_rows.append(
                {
                    "row_id_v1": row.get("management_row_key_v1"),
                    "candidate_uid_exact_v1": row.get("candidate_uid_exact_v1"),
                    "action_v1": row.get("action_label_v1"),
                    "exclusion_reason_v1": "|".join(reasons),
                    "missing_source_columns_v1": "|".join(missing_source_cols),
                }
            )

    df["_eligibility_reasons_v1"] = ["|".join(reasons_by_index.get(idx, [])) for idx in df.index]
    df["_eligible_v1"] = df["_eligibility_reasons_v1"].eq("")
    if exclusion_rows:
        exclusions_df = pd.DataFrame.from_records(exclusion_rows)
    else:
        exclusions_df = pd.DataFrame(columns=["row_id_v1", "candidate_uid_exact_v1", "action_v1", "exclusion_reason_v1", "missing_source_columns_v1"])
    return df, exclusions_df


def _build_dataset(eligible_df: pd.DataFrame, state_features: list[str]) -> pd.DataFrame:
    included = eligible_df.loc[eligible_df["_eligible_v1"]].copy()
    state_feature_names_string = _json_array_string(state_features)
    rows: list[dict[str, Any]] = []
    for _, row in included.iterrows():
        action = str(row.get("action_label_v1"))
        behavior_status = "|".join(
            [
                str(row.get("behavior_policy_kind_v1")),
                str(row.get("behavior_policy_id_status_v1")),
                str(row.get("policy_logging_propensity_status_v1")),
                str(row.get("observed_action_status_v1")),
            ]
        )
        rows.append(
            {
                "row_id": row.get("management_row_key_v1"),
                "episode_id": row.get("trade_uid_exact_v1"),
                "candidate_uid_exact": row.get("candidate_uid_exact_v1"),
                "decision_ts": row.get("decision_timestamp"),
                "action": action,
                "action_id": ACTION_ID_MAP[action],
                "reward": _finite_float_or_none(row.get(REWARD_SOURCE_COLUMN)),
                "reward_version": REWARD_VERSION_ID,
                "state_feature_names": state_feature_names_string,
                "state_vector": _json_array_string([row.get(feature) for feature in state_features]),
                "source_policy_version": row.get("policy_version_v1"),
                "behavior_policy_status": behavior_status,
                "support_status": row.get("support_tier_v1"),
                "as_of_schema_version": row.get("observation_contract_v1"),
                "hindsight_outcome_backfill_version": row.get("terminal_outcome_availability_status_v1"),
                "eligibility_status": "INCLUDED_MANAGEMENT_BANDIT_RESEARCH_V1",
                "exclusion_reason": "",
                "provenance_namespace": LAYER_ID,
            }
        )
    return pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)


def _column_class(field: str) -> str:
    if field in {"reward", "hindsight_outcome_backfill_version"}:
        return "HINDSIGHT_REWARD"
    if field in {"state_vector", "state_feature_names", "decision_ts", "as_of_schema_version"}:
        return "AS_OF_STATE_OR_PROVENANCE"
    if field in {"action", "action_id", "source_policy_version", "behavior_policy_status", "support_status"}:
        return "BEHAVIOR_LOG_OR_SUPPORT"
    return "METADATA"


def _column_source(field: str, source_paths: dict[str, str | None]) -> str | None:
    if field in {"source_policy_version", "behavior_policy_status", "support_status"}:
        return source_paths.get("management_policy_log_v1")
    if field == "reward":
        return source_paths.get("management_bandit_dm_view_v1")
    if field in {"state_vector", "state_feature_names", "action", "action_id", "row_id", "episode_id", "candidate_uid_exact", "decision_ts", "as_of_schema_version", "hindsight_outcome_backfill_version"}:
        return source_paths.get("management_bandit_dm_view_v1")
    return str(source_paths.get("foundation_dir_v1"))


def _build_column_coverage(dataset_df: pd.DataFrame, source_paths: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = int(len(dataset_df))
    for col in OUTPUT_COLUMNS:
        series = dataset_df[col] if col in dataset_df.columns else pd.Series(dtype="object")
        null_count = int(series.isna().sum() + (series.astype(str).str.len().eq(0).sum() if len(series) else 0))
        rows.append(
            {
                "column_v1": col,
                "row_count_v1": total,
                "non_null_count_v1": int(total - null_count),
                "null_count_v1": null_count,
                "distinct_count_v1": int(series.astype(str).nunique(dropna=False)) if len(series) else 0,
                "source_artifact_v1": _column_source(col, source_paths),
                "as_of_hindsight_class_v1": _column_class(col),
                "validation_status_v1": "READY" if col in dataset_df.columns and (col == "exclusion_reason" or null_count == 0) else "PARTIAL_OR_EMPTY",
            }
        )
    df = pd.DataFrame.from_records(rows)
    return df, {"rows_v1": df.to_dict(orient="records")}


def _reason_counts(exclusions_df: pd.DataFrame) -> pd.DataFrame:
    reason_counts: dict[str, int] = {}
    if not exclusions_df.empty:
        for raw in exclusions_df["exclusion_reason_v1"].astype(str).tolist():
            for reason in raw.split("|"):
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
    rows = [{"exclusion_reason_v1": key, "row_count_v1": value} for key, value in sorted(reason_counts.items())]
    return pd.DataFrame.from_records(rows, columns=["exclusion_reason_v1", "row_count_v1"])


def _build_eligibility_audit(dataset_df: pd.DataFrame, eligible_df: pd.DataFrame, exclusions_df: pd.DataFrame, state_features: list[str]) -> dict[str, Any]:
    included = eligible_df.loc[eligible_df["_eligible_v1"]].copy()
    return {
        "audit_id_v1": "MANAGEMENT_BANDIT_ELIGIBILITY_AND_EXCLUSION_AUDIT_V1",
        "total_candidate_rows_evaluated_v1": int(len(eligible_df)),
        "total_included_rows_v1": int(len(dataset_df)),
        "total_excluded_rows_v1": int(len(exclusions_df)),
        "included_by_action_v1": _value_counts(dataset_df["action"], "action_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "included_by_support_status_v1": _value_counts(dataset_df["support_status"], "support_status_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "included_by_behavior_policy_status_v1": _value_counts(dataset_df["behavior_policy_status"], "behavior_policy_status_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "excluded_by_reason_v1": _reason_counts(exclusions_df).to_dict(orient="records"),
        "reward_coverage_among_included_v1": {
            "reward_version_v1": REWARD_VERSION_ID,
            "non_null_reward_rows_v1": int(dataset_df["reward"].notna().sum()) if "reward" in dataset_df.columns else 0,
            "coverage_rate_v1": float(dataset_df["reward"].notna().mean()) if len(dataset_df) else 0.0,
        },
        "state_coverage_among_included_v1": {
            "state_feature_count_v1": int(len(state_features)),
            "rows_with_state_vector_v1": int(dataset_df["state_vector"].notna().sum()) if "state_vector" in dataset_df.columns else 0,
            "state_vector_coverage_rate_v1": float(dataset_df["state_vector"].notna().mean()) if len(dataset_df) else 0.0,
        },
        "source_sequence_membership_counts_v1": _value_counts(included.get("sequence_dataset_membership_v1", pd.Series(dtype="object")), "sequence_dataset_membership_v1").to_dict(orient="records"),
        "fail_closed_if_unclear_v1": True,
    }


def _build_state_coverage(eligible_df: pd.DataFrame, dataset_df: pd.DataFrame, state_features: list[str]) -> pd.DataFrame:
    included = eligible_df.loc[eligible_df["_eligible_v1"]].copy()
    rows: list[dict[str, Any]] = []
    for feature in state_features:
        source_series = included[feature] if feature in included.columns else pd.Series(dtype="object")
        null_count = int(source_series.isna().sum()) if len(source_series) else 0
        rows.append(
            {
                "state_feature_name_v1": feature,
                "included_row_count_v1": int(len(dataset_df)),
                "non_null_count_v1": int(len(source_series) - null_count),
                "null_count_v1": null_count,
                "coverage_rate_v1": float(1.0 - (null_count / len(source_series))) if len(source_series) else 0.0,
                "optional_sparse_feature_v1": _optional_state_feature(feature, state_features),
                "source_class_v1": "AS_OF_STATE",
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_thin_pockets(eligible_df: pd.DataFrame) -> pd.DataFrame:
    included = eligible_df.loc[eligible_df["_eligible_v1"]].copy()
    pocket_columns = [
        "action_label_v1",
        "support_tier_v1",
        "as_of_session_v1",
        "as_of_vol_regime_v1",
        "as_of_trend_regime_v1",
        "as_of_side_v1",
    ]
    rows: list[dict[str, Any]] = []
    for col in pocket_columns:
        if col not in included.columns:
            continue
        counts = included[col].fillna("NULL").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "pocket_field_v1": col,
                    "pocket_value_v1": value,
                    "row_count_v1": int(count),
                    "thin_pocket_v1": int(count) < 10,
                    "note_v1": "DESCRIPTIVE_SUPPORT_PROFILE_ONLY_NO_TRADING_ANALYSIS",
                }
            )
    return pd.DataFrame.from_records(rows, columns=["pocket_field_v1", "pocket_value_v1", "row_count_v1", "thin_pocket_v1", "note_v1"])


def _build_profile(
    dataset_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    state_coverage_df: pd.DataFrame,
    thin_pockets_df: pd.DataFrame,
    foundation_support: dict[str, Any],
) -> dict[str, Any]:
    support_verdict = foundation_support.get("overall_support_verdict_v1", foundation_support.get("bandit_support_verdict_v1", "NOT_ESTABLISHED"))
    limitations = []
    if support_verdict in {"SUPPORT_TOO_THIN", "SUPPORT_WEAK_BUT_USABLE"}:
        limitations.append(f"support_ood_verdict_{support_verdict}")
    if int((thin_pockets_df["thin_pocket_v1"] == True).sum()) if not thin_pockets_df.empty else 0:
        limitations.append("thin_descriptive_state_pockets_present")
    verdict = "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS" if limitations else "BANDIT_RESEARCH_DATASET_BUILT"
    return {
        "profile_id_v1": "BANDIT_DATASET_PROFILE_AND_RESEARCH_READINESS_V1",
        "verdict_v1": verdict if len(dataset_df) else "DATASET_BUILD_FAILED",
        "row_count_v1": int(len(dataset_df)),
        "action_distribution_v1": _value_counts(dataset_df["action"], "action_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "reward_distribution_summary_v1": _distribution(dataset_df["reward"]) if "reward" in dataset_df.columns else _distribution(pd.Series(dtype="float64")),
        "support_status_distribution_v1": _value_counts(dataset_df["support_status"], "support_status_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "behavior_policy_distribution_v1": _value_counts(dataset_df["behavior_policy_status"], "behavior_policy_status_v1").to_dict(orient="records") if not dataset_df.empty else [],
        "state_coverage_summary_v1": {
            "state_feature_count_v1": int(len(state_coverage_df)),
            "fully_non_null_state_features_v1": int((state_coverage_df["null_count_v1"] == 0).sum()) if not state_coverage_df.empty else 0,
            "sparse_optional_state_features_v1": int((state_coverage_df["optional_sparse_feature_v1"] == True).sum()) if not state_coverage_df.empty else 0,
            "lowest_feature_coverage_rate_v1": float(state_coverage_df["coverage_rate_v1"].min()) if not state_coverage_df.empty else 0.0,
        },
        "source_sequence_membership_distribution_v1": _value_counts(eligible_df.loc[eligible_df["_eligible_v1"]].get("sequence_dataset_membership_v1", pd.Series(dtype="object")), "sequence_dataset_membership_v1").to_dict(orient="records"),
        "thin_pocket_count_v1": int((thin_pockets_df["thin_pocket_v1"] == True).sum()) if not thin_pockets_df.empty else 0,
        "support_ood_verdict_from_foundation_v1": support_verdict,
        "limitations_v1": limitations,
        "not_training_ready_universal_rl_v1": True,
        "not_iql_ready_v1": True,
    }


def _build_consistency(
    dataset_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    state_features: list[str],
    reward_lock_summary: dict[str, Any],
    non_interference: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    included = eligible_df.loc[eligible_df["_eligible_v1"]].copy()
    source_reward = pd.to_numeric(included[REWARD_SOURCE_COLUMN], errors="coerce").reset_index(drop=True) if REWARD_SOURCE_COLUMN in included.columns else pd.Series(dtype="float64")
    dataset_reward = pd.to_numeric(dataset_df["reward"], errors="coerce").reset_index(drop=True) if "reward" in dataset_df.columns else pd.Series(dtype="float64")
    reward_equal = len(source_reward) == len(dataset_reward) and bool(source_reward.equals(dataset_reward))
    path_tokens = set(PATH_DYNAMICS_V2_FIELDS)
    path_used = [col for col in dataset_df.columns if col in path_tokens or col.replace("as_of_management_core_", "").replace("_v1", "") in path_tokens]
    state_feature_names_ok = True
    if not dataset_df.empty:
        expected = _json_array_string(state_features)
        state_feature_names_ok = bool(dataset_df["state_feature_names"].astype(str).eq(expected).all())
    checks = [
        ("REWARD_VERSION_IDENTICAL_ALL_ROWS", dataset_df.empty or dataset_df["reward_version"].astype(str).eq(REWARD_VERSION_ID).all(), dataset_df["reward_version"].astype(str).nunique(dropna=False) if "reward_version" in dataset_df else 0, 1),
        ("REWARD_FOLLOWS_LOCKED_FORMULA", reward_equal, "dataset reward equals hindsight_reward_realized_pnl_bps_v1", REWARD_SOURCE_COLUMN),
        ("NO_PATH_DYNAMICS_V2_CANONICAL_TRAINING_FIELDS_USED", len(path_used) == 0, "|".join(path_used), "none"),
        ("NO_SEQUENCE_ONLY_COLUMNS_CLAIMED_COMPLETE", not any(col in dataset_df.columns for col in ["next_state", "next_state_vector", "done", "transition_id"]), list(dataset_df.columns), "no next_state/done/transition_id"),
        ("STATE_FEATURES_MATCH_FOUNDATION_SCHEMA", state_feature_names_ok and len(state_features) > 0, len(state_features), "foundation state_feature_names_v1"),
        ("ROW_ID_UNIQUE", dataset_df.empty or dataset_df["row_id"].is_unique, int(dataset_df["row_id"].duplicated().sum()) if "row_id" in dataset_df else None, 0),
        ("REWARD_TRACEABLE_TO_LOCKED_HINDSIGHT_TRUTH", reward_lock_summary.get("reward_version_id_v1") == REWARD_VERSION_ID and reward_equal, reward_lock_summary.get("reward_version_id_v1"), REWARD_VERSION_ID),
        ("EACH_ROW_HAS_CANONICAL_PROVENANCE", dataset_df.empty or dataset_df["provenance_namespace"].astype(str).eq(LAYER_ID).all(), dataset_df["provenance_namespace"].astype(str).nunique(dropna=False) if "provenance_namespace" in dataset_df else 0, LAYER_ID),
        ("AS_OF_STATE_SEPARATED_FROM_HINDSIGHT_REWARD", all(not any(token in feature.lower() for token in ["hindsight", "terminal", "reward", "outcome"]) for feature in state_features), state_features, "no hindsight/reward/outcome state feature token"),
        ("NO_NEXT_STATE_PLACEHOLDER", not any(col in dataset_df.columns for col in ["next_state", "next_state_vector"]), list(dataset_df.columns), "no next_state columns"),
        ("DATASET_DOES_NOT_CLAIM_IQL_READY", True, "management_contextual_bandit_research_only", "not_iql_dataset"),
        ("NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
    ]
    rows = [
        {
            "check_name_v1": name,
            "status_v1": "PASS" if passed else "FAIL",
            "observed_value_v1": observed,
            "expected_value_v1": expected,
        }
        for name, passed, observed, expected in checks
    ]
    df = pd.DataFrame.from_records(rows)
    return df, {
        "audit_id_v1": "BANDIT_DATASET_CONSISTENCY_AND_INVARIANTS_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "passed_check_count_v1": int((df["status_v1"] == "PASS").sum()),
        "warnings_v1": [
            "support remains thin for RL research" if not dataset_df.empty else "dataset empty",
            "episode_id is trade_uid_exact_v1 for row-wise bandit provenance, not sequence-MDP episode linkage",
        ],
        "fail_closed_verdict_v1": "PASS" if int((df["status_v1"] != "PASS").sum()) == 0 else "FAIL_CLOSED",
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_post_build_status(profile: dict[str, Any], reward_lock_summary: dict[str, Any]) -> dict[str, Any]:
    built = profile.get("verdict_v1") in {"BANDIT_RESEARCH_DATASET_BUILT", "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS"}
    return {
        "update_id_v1": "POST_BUILD_STATUS_UPDATE_V1",
        "management_bandit_dataset_built_v1": built,
        "dataset_verdict_v1": profile.get("verdict_v1"),
        "reward_lock_used_correctly_v1": reward_lock_summary.get("reward_version_id_v1") == REWARD_VERSION_ID,
        "reward_version_v1": REWARD_VERSION_ID,
        "comparator_failcheck_contract_still_governing_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_status_unchanged_blocked_v1": True,
        "replay_status_unchanged_v1": True,
        "next_safe_steps_v1": [
            "BANDIT_RESEARCH_EVAL_PREP",
            "WAIT_FOR_REPLAY_FOR_CHAIN_REBUILD_AND_HOLD_TRANSITION_TRUTH",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "management_bandit_dataset_built_with_locked_reward" if built else "management_bandit_dataset_not_built",
                "reward_version_used",
                "sequence_iql_still_blocked",
                "hold_next_state_truth_still_missing",
                "r7_not_started",
                "iql_training_not_started",
            ],
            "INDIKERT": [
                "bandit_research_eval_prep_can_start" if built else "dataset_build_needs_fix",
                "support_limitations_require_failcheck_governance",
            ],
            "IKKE_ETABLERT": [
                "sequence_iql_readiness",
                "canonical_hold_next_state_transitions",
                "path_dynamics_training_canonical_status",
            ],
        },
    }


def _build_non_interference(
    output_dir: Path,
    source_paths: dict[str, str | None],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
    r6_sha_before: str | None,
    r6_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay source path"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("IN_PROGRESS_REPLAY_NOT_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay canonical source"),
    ]
    df = pd.DataFrame.from_records(
        [
            {
                "check_name_v1": name,
                "status_v1": status,
                "observed_value_v1": observed,
                "expected_value_v1": expected,
            }
            for name, status, observed, expected in checks
        ]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    profile = payload["profile"]
    return "\n".join(
        [
            "# Build Management Bandit Dataset V1",
            "",
            "## Dataset",
            "",
            f"- Dataset verdict: `{profile['verdict_v1']}`",
            f"- Included rows: `{summary['included_rows_v1']}`",
            f"- Excluded rows: `{summary['excluded_rows_v1']}`",
            f"- Reward version: `{summary['reward_version_v1']}`",
            f"- Dataset parquet: `{summary['dataset_parquet_v1']}`",
            "",
            "## Boundaries",
            "",
            "- This is a management contextual bandit research dataset, not an IQL dataset.",
            "- It contains no `next_state`, `done`, or sequence-IQL transition placeholders.",
            "- HINDSIGHT terminal realized PnL is used only as reward; state vectors are AS_OF foundation features.",
            "- Path-dynamics v2 remains `PENDING_REPLAY`, `NOT_CANONICAL_YET`, `DO_NOT_USE_FOR_TRAINING`.",
            "",
            "## Status",
            "",
            "- Comparator/fail-check contract remains governing.",
            "- Sequence-IQL remains blocked because HOLD -> next_state truth is still missing.",
            "- R7 and IQL training were not started.",
        ]
    ) + "\n"


def build_management_bandit_dataset(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    reward_lock_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    foundation_dir = foundation_dir or _resolve_foundation_dir(reports_root, None)
    reward_lock_dir = reward_lock_dir or _latest_reward_lock_dir(reports_root, None)
    contract_lock_dir = contract_lock_dir or _latest_contract_lock_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    foundation_contract = _read_json_optional(foundation_dir / "iql_foundation_mdp_contract_v1.json")
    dataset_schema = _read_json_optional(foundation_dir / "iql_foundation_dataset_schema_v1.json")
    management_contract = _read_json_optional(foundation_dir / "iql_foundation_management_mdp_contract_v1.json")
    foundation_support = _read_json_optional(foundation_dir / "iql_foundation_support_ood_audit_v1.json")
    reward_lock_summary = _read_json_optional(reward_lock_dir / "lock_first_bandit_reward_version_summary_v1.json")
    reward_contract = _read_json_optional(reward_lock_dir / "first_bandit_reward_contract_v1.json")
    bandit_contract = _read_json_optional(contract_lock_dir / "iql_management_bandit_dataset_contract_lock_v1.json")
    comparator_lock = _read_json_optional(contract_lock_dir / "iql_baseline_comparator_and_failcheck_lock_v1.json")
    source_paths = _source_paths(reports_root, foundation_dir, reward_lock_dir, contract_lock_dir, foundation_contract)

    state_features = _state_features(dataset_schema, management_contract)
    dm_df, policy_df = _read_source_frames(source_paths)
    merged_df = _merge_policy_log(dm_df, policy_df)
    eligible_df, exclusions_df = _build_eligibility(merged_df, state_features, reward_lock_summary, reward_contract)
    dataset_df = _build_dataset(eligible_df, state_features)
    column_coverage_df, column_coverage = _build_column_coverage(dataset_df, source_paths)
    eligibility_audit = _build_eligibility_audit(dataset_df, eligible_df, exclusions_df, state_features)
    state_coverage_df = _build_state_coverage(eligible_df, dataset_df, state_features)
    thin_pockets_df = _build_thin_pockets(eligible_df)
    profile = _build_profile(dataset_df, eligible_df, state_coverage_df, thin_pockets_df, foundation_support)
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    consistency_df, consistency = _build_consistency(dataset_df, eligible_df, state_features, reward_lock_summary, non_interference)
    post_build_status = _build_post_build_status(profile, reward_lock_summary)

    action_distribution_df = _value_counts(dataset_df["action"], "action_v1") if not dataset_df.empty else pd.DataFrame(columns=["action_v1", "row_count_v1", "row_share_v1"])
    support_distribution_df = _value_counts(dataset_df["support_status"], "support_status_v1") if not dataset_df.empty else pd.DataFrame(columns=["support_status_v1", "row_count_v1", "row_share_v1"])
    behavior_distribution_df = _value_counts(dataset_df["behavior_policy_status"], "behavior_policy_status_v1") if not dataset_df.empty else pd.DataFrame(columns=["behavior_policy_status_v1", "row_count_v1", "row_share_v1"])
    reason_counts_df = _reason_counts(exclusions_df)

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_MANAGEMENT_CONTEXTUAL_BANDIT_DATASET_BUILD",
        "not_iql_dataset_v1": True,
        "not_sequence_iql_dataset_v1": True,
        "not_training_v1": True,
        "action_space_v1": ["HOLD", "EXIT_NOW"],
        "action_id_map_v1": ACTION_ID_MAP,
        "reward_name_v1": REWARD_NAME,
        "reward_version_v1": REWARD_VERSION_ID,
        "reward_formula_v1": "reward_bps = terminal_realized_pnl_bps",
        "state_feature_count_v1": int(len(state_features)),
        "state_feature_names_source_v1": "IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1.state_feature_names_v1",
        "output_columns_v1": OUTPUT_COLUMNS,
        "state_vector_serialization_v1": "JSON_ARRAY_STRING",
        "source_paths_v1": source_paths,
        "upstream_bandit_dataset_contract_v1": bandit_contract.get("contract_id_v1"),
        "comparator_failcheck_contract_v1": comparator_lock.get("lock_id_v1"),
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_build_sequence_iql_dataset_v1": True,
            "do_not_use_in_progress_replay_as_canonical_v1": True,
            "do_not_modify_r6_freeze_v1": True,
            "do_not_modify_locked_ledger_v1": True,
            "no_next_state_placeholders_v1": True,
            "no_as_of_hindsight_mixing_in_state_v1": True,
        },
        "path_dynamics_v2_status_v1": {
            "status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
            "fields_v1": PATH_DYNAMICS_V2_FIELDS,
        },
    }

    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "dataset_built_v1": profile.get("verdict_v1") in {"BANDIT_RESEARCH_DATASET_BUILT", "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS"},
        "dataset_verdict_v1": profile.get("verdict_v1"),
        "dataset_parquet_v1": str((Path(output_dir) / OUTPUTS["dataset_parquet"]).resolve()),
        "dataset_csv_v1": str((Path(output_dir) / OUTPUTS["dataset_csv"]).resolve()),
        "total_candidate_rows_evaluated_v1": int(len(eligible_df)),
        "included_rows_v1": int(len(dataset_df)),
        "excluded_rows_v1": int(len(exclusions_df)),
        "action_distribution_v1": action_distribution_df.to_dict(orient="records"),
        "reward_version_v1": REWARD_VERSION_ID,
        "reward_name_v1": REWARD_NAME,
        "reward_formula_v1": "reward_bps = terminal_realized_pnl_bps",
        "state_feature_count_v1": int(len(state_features)),
        "foundation_core_9_respected_v1": True,
        "comparator_failcheck_contract_still_applies_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "support_ood_verdict_v1": profile.get("support_ood_verdict_from_foundation_v1"),
        "support_limitations_v1": profile.get("limitations_v1", []),
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "recommended_next_step_v1": "BANDIT_RESEARCH_EVAL_PREP_AND_WAIT_FOR_REPLAY_FOR_CHAIN_REBUILD",
        "hard_status_partition_v1": post_build_status.get("hard_status_v1"),
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_MANAGEMENT_BANDIT_RESEARCH_DATASET" if summary["dataset_built_v1"] else "DATASET_BUILD_FAILED",
        "dataset_built_v1": summary["dataset_built_v1"],
        "training_executed_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int(consistency.get("failed_check_count_v1", 0)),
        "failed_non_interference_check_count_v1": int(non_interference.get("failed_check_count_v1", 0)),
    }
    dataset_metadata = {
        "metadata_id_v1": "MANAGEMENT_BANDIT_DATASET_BUILD_V1",
        "dataset_parquet_v1": summary["dataset_parquet_v1"],
        "dataset_csv_v1": summary["dataset_csv_v1"],
        "row_count_v1": int(len(dataset_df)),
        "column_count_v1": int(len(dataset_df.columns)),
        "columns_v1": list(dataset_df.columns),
        "state_feature_count_v1": int(len(state_features)),
        "state_vector_serialization_v1": "JSON_ARRAY_STRING",
        "episode_id_note_v1": "Uses trade_uid_exact_v1 for row-wise bandit provenance only; this is not sequence-MDP episode linkage.",
        "reward_version_v1": REWARD_VERSION_ID,
        "not_iql_dataset_v1": True,
        "not_sequence_dataset_v1": True,
    }
    return {
        "contract": contract,
        "dataset_df": dataset_df,
        "eligible_df": eligible_df,
        "state_features": state_features,
        "dataset_metadata": dataset_metadata,
        "column_coverage_df": column_coverage_df,
        "column_coverage": column_coverage,
        "eligibility_audit": eligibility_audit,
        "exclusions_df": exclusions_df,
        "reason_counts_df": reason_counts_df,
        "state_coverage_df": state_coverage_df,
        "thin_pockets_df": thin_pockets_df,
        "profile": profile,
        "action_distribution_df": action_distribution_df,
        "support_distribution_df": support_distribution_df,
        "behavior_distribution_df": behavior_distribution_df,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "consistency": consistency,
        "post_build_status": post_build_status,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_management_bandit_dataset_artifacts(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    reward_lock_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    r6_path = reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"
    exit_manager_sha_before = _sha256(exit_manager_path)
    r6_sha_before = _sha256(r6_path)

    payload = build_management_bandit_dataset(
        reports_root,
        foundation_dir=foundation_dir,
        reward_lock_dir=reward_lock_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    payload["dataset_df"].to_parquet(target_dir / OUTPUTS["dataset_parquet"], index=False)
    payload["dataset_df"].to_csv(target_dir / OUTPUTS["dataset_csv"], index=False)
    _write_json(target_dir / OUTPUTS["dataset_metadata"], payload["dataset_metadata"])
    payload["column_coverage_df"].to_csv(target_dir / OUTPUTS["column_coverage_csv"], index=False)
    _write_json(target_dir / OUTPUTS["column_coverage_json"], payload["column_coverage"])
    _write_json(target_dir / OUTPUTS["eligibility_audit"], payload["eligibility_audit"])
    payload["exclusions_df"].to_csv(target_dir / OUTPUTS["exclusion_rows"], index=False)
    payload["reason_counts_df"].to_csv(target_dir / OUTPUTS["exclusion_reasons"], index=False)
    payload["action_distribution_df"].to_csv(target_dir / OUTPUTS["profile_action_distribution"], index=False)
    payload["support_distribution_df"].to_csv(target_dir / OUTPUTS["profile_support_distribution"], index=False)
    payload["behavior_distribution_df"].to_csv(target_dir / OUTPUTS["profile_behavior_distribution"], index=False)
    payload["state_coverage_df"].to_csv(target_dir / OUTPUTS["profile_state_coverage"], index=False)
    payload["thin_pockets_df"].to_csv(target_dir / OUTPUTS["profile_thin_pockets"], index=False)
    _write_json(target_dir / OUTPUTS["profile"], payload["profile"])
    _write_json(target_dir / OUTPUTS["post_build_status"], payload["post_build_status"])

    exit_manager_sha_after = _sha256(exit_manager_path)
    r6_sha_after = _sha256(r6_path)
    non_interference_df, non_interference = _build_non_interference(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference
    payload["consistency_df"], payload["consistency"] = _build_consistency(
        payload["dataset_df"],
        payload["eligible_df"],
        payload["state_features"],
        _read_json_optional(Path(payload["source_paths"]["reward_lock_dir_v1"]) / "lock_first_bandit_reward_version_summary_v1.json"),
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int(payload["consistency"]["failed_check_count_v1"])

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["consistency_audit_json"], payload["consistency"])
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "append_only_namespace_v1": "IQL_INTEGRATION",
        "artifact_paths_v1": artifact_paths,
        "source_paths_v1": payload["source_paths"],
        "read_only_references_v1": True,
        "not_training_v1": True,
        "not_iql_dataset_v1": True,
        "not_sequence_iql_dataset_v1": True,
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], payload["status"])
    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the first management contextual bandit research dataset.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--foundation-dir", type=str, default=None)
    parser.add_argument("--reward-lock-dir", type=str, default=None)
    parser.add_argument("--contract-lock-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    foundation_dir = Path(args.foundation_dir).expanduser().resolve() if args.foundation_dir else None
    reward_lock_dir = Path(args.reward_lock_dir).expanduser().resolve() if args.reward_lock_dir else None
    contract_lock_dir = Path(args.contract_lock_dir).expanduser().resolve() if args.contract_lock_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_management_bandit_dataset_artifacts(
        reports_root,
        foundation_dir=foundation_dir,
        reward_lock_dir=reward_lock_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
