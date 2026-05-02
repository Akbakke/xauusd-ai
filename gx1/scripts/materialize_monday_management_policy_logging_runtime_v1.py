#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.analysis import shadow_meta_v1 as shadow_meta
from gx1.scripts import (
    materialize_monday_native_shadow_refreeze_comparison_v1 as refreeze_compare,
)
from gx1.scripts import (
    materialize_path_dynamics_logging_v2_implementation_and_replay_audit_v1 as path_dynamics_audit,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_PREFIX = "MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_V1_"
TOP_LEVEL_SUMMARY = "truth_monday_management_policy_logging_runtime_v1.json"

CONTRACT = "contract_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

_POLICY_LOGGING_SPEC = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_SPEC_V1
_POLICY_LOGGING_DECISION = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_DECISION_LOG_HARNESS_V1_PARQUET
_POLICY_LOGGING_OUTCOME = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_OUTCOME_BACKFILL_HARNESS_V1_PARQUET
_POLICY_LOGGING_MISSING = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_MISSING_FIELDS_V1
_POLICY_LOGGING_SUMMARY = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_SUMMARY_V1
_POLICY_LOGGING_CONSISTENCY = shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_POLICY_LOGGING_CONSISTENCY_AUDIT_V1_CSV
_JOIN_KEYS = list(shadow_meta._MANAGEMENT_AUDIT_EXTENSION_EXACT_JOIN_KEYS_V1)
_OBS_FIELDS = list(shadow_meta._MANAGEMENT_RL_OBSERVATION_FIELDS_V1)
_LEDGER_NOT_AVAILABLE = shadow_meta._LEDGER_NOT_AVAILABLE
_LEDGER_IKKE_ETABLERT = shadow_meta._LEDGER_IKKE_ETABLERT
_US_CORE_COMPOSITE = shadow_meta._MANAGEMENT_AUDIT_EXTENSION_US_CORE_COMPOSITE_V1
_RANK27_OUTLIER_COMPOSITE = shadow_meta._MANAGEMENT_AUDIT_EXTENSION_RANK27_OUTLIER_COMPOSITE_V1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_ledger_dir(reports_root: Path, ledger_dir_arg: str | None) -> Path:
    if ledger_dir_arg:
        ledger_dir = Path(ledger_dir_arg).expanduser().resolve()
    else:
        rebuild_summary = json.loads((reports_root / "truth_downstream_canonical_rebuild_v1.json").read_text(encoding="utf-8"))
        ledger_dir = Path(str(rebuild_summary["ledger_dir"])).expanduser().resolve()
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Ledger dir does not exist: {ledger_dir}")
    return ledger_dir


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / f"{EXTENSION_PREFIX}{_utc_stamp()}"


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
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_columns(frame: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column_name, default_value in defaults.items():
        if column_name not in out.columns:
            out[column_name] = default_value
    return out


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(_LEDGER_NOT_AVAILABLE)


def _sample_rows(frame: pd.DataFrame, columns: list[str], limit: int = 3) -> list[dict[str, Any]]:
    available = [column_name for column_name in columns if column_name in frame.columns]
    if not available:
        return []
    sample_df = frame[available].head(limit).copy()
    sample_df = sample_df.where(pd.notna(sample_df), None)
    rows: list[dict[str, Any]] = []
    for record in sample_df.to_dict(orient="records"):
        payload: dict[str, Any] = {}
        for key, value in record.items():
            payload[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(payload)
    return rows


def _build_runtime_overlay_view(
    direct_method_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    as_of_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_pocket_df = as_of_df[_JOIN_KEYS + ["as_of_trade_pocket_v1"]].drop_duplicates(subset=_JOIN_KEYS, keep="last")
    overlay_df = direct_method_df[_JOIN_KEYS].drop_duplicates(subset=_JOIN_KEYS, keep="last").merge(
        eligible_df[
            _JOIN_KEYS
            + [
                "as_of_session_v1",
                "as_of_vol_regime_v1",
                "as_of_management_core_minutes_held_at_anchor_v1",
                "as_of_management_core_giveback_ratio_from_peak_v1",
            ]
        ].drop_duplicates(subset=_JOIN_KEYS, keep="last"),
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    overlay_df = overlay_df.merge(
        trade_pocket_df,
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    overlay_df = shadow_meta._attach_management_regime_overlay_tags_v1(overlay_df)
    overlay_df["overlay_tagging_status_v1"] = np.where(
        overlay_df[
            [
                "overlay_session_axis_v1",
                "overlay_trade_pocket_v1",
                "overlay_vol_axis_v1",
                "overlay_hold_age_axis_v1",
                "overlay_giveback_axis_v1",
            ]
        ]
        .astype("string")
        .eq(_LEDGER_NOT_AVAILABLE)
        .any(axis=1),
        "TAGGING_INCOMPLETE",
        "TAGGED_EXACT",
    )
    overlay_df["overlay_pocket_group_v1"] = "OTHER"
    overlay_df.loc[
        overlay_df["overlay_composite_v1"].astype("string").eq(_US_CORE_COMPOSITE),
        "overlay_pocket_group_v1",
    ] = "US_CORE_POCKET"
    overlay_df.loc[
        overlay_df["overlay_composite_v1"].astype("string").eq(_RANK27_OUTLIER_COMPOSITE),
        "overlay_pocket_group_v1",
    ] = "OVERLAP_OUTLIER_COMPOSITE"
    overlay_df["manual_review_candidate_v1"] = False
    overlay_df["manual_review_candidate_status_v1"] = "RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT"
    return overlay_df


def _build_runtime_shadow_attach_view(
    direct_method_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
) -> pd.DataFrame:
    score_cols = [
        "primary_model_name_v1",
        "primary_model_score_v1",
        "primary_model_score_rank_within_split_v1",
    ]
    score_view = direct_method_df[_JOIN_KEYS].drop_duplicates(subset=_JOIN_KEYS, keep="last").merge(
        eligible_df[_JOIN_KEYS + score_cols].drop_duplicates(subset=_JOIN_KEYS, keep="last"),
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    score_view["shadow_model_source_v1"] = _safe_string(score_view["primary_model_name_v1"])
    score_view["shadow_score_source_v1"] = "RUNTIME_EXIT_LOCAL_PRIMARY_MODEL_SCORE"
    score_view["shadow_score_v1"] = pd.to_numeric(score_view["primary_model_score_v1"], errors="coerce")
    score_view["shadow_domain_status_v1"] = np.where(
        score_view["shadow_score_v1"].notna(),
        "RUNTIME_EXIT_LOCAL_SCORE_ATTACHED",
        _LEDGER_IKKE_ETABLERT,
    )
    score_view["shadow_bucket_status_v1"] = np.where(
        pd.to_numeric(score_view["primary_model_score_rank_within_split_v1"], errors="coerce").notna(),
        "RUNTIME_EXIT_LOCAL_SCORE_RANK_WITHIN_SPLIT",
        _LEDGER_IKKE_ETABLERT,
    )
    score_view["shadow_bucket_rank_v1"] = pd.to_numeric(
        score_view["primary_model_score_rank_within_split_v1"], errors="coerce"
    ).astype("Int64")
    score_view["shadow_usage_status_v1"] = "RUNTIME_SCORE_RESEARCH_ONLY_NOT_CONTROLLER"
    score_view["shadow_counterfactual_status_v1"] = "COUNTERFACTUAL_NOT_ESTABLISHED_RUNTIME_ONLY"
    score_view["research_priority_status_v1"] = "RESEARCH_PRIORITY_NOT_ESTABLISHED_RUNTIME_ONLY"
    return score_view[
        _JOIN_KEYS
        + [
            "shadow_model_source_v1",
            "shadow_score_source_v1",
            "shadow_score_v1",
            "shadow_domain_status_v1",
            "shadow_bucket_status_v1",
            "shadow_bucket_rank_v1",
            "shadow_usage_status_v1",
            "shadow_counterfactual_status_v1",
            "research_priority_status_v1",
        ]
    ].copy()


def _build_runtime_only_policy_logging_payload(
    *,
    observed_sample_df: pd.DataFrame,
    direct_method_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    raw_state_df: pd.DataFrame,
    as_of_df: pd.DataFrame,
    closed_trades_df: pd.DataFrame,
    hindsight_review_export_df: pd.DataFrame,
    management_bandit_action_reward_contract_v1: dict[str, Any],
    management_bandit_observed_action_contract_v1: dict[str, Any],
    management_bandit_status_v1: dict[str, Any],
    as_of_supervision_join_coverage_summary: dict[str, Any],
    leakage_guard_summary: dict[str, Any],
    build_id_v1: str,
    build_timestamp_utc_v1: str,
    source_control_date_v1: str,
) -> dict[str, Any]:
    observed_sample_df = observed_sample_df.copy()
    decision_df = direct_method_df.copy()
    eligible_df = eligible_df.copy()
    raw_state_df = raw_state_df.copy()
    as_of_df = as_of_df.copy()
    closed_trades_df = closed_trades_df.copy()
    hindsight_review_export_df = hindsight_review_export_df.copy()

    action_space_v1 = list(management_bandit_action_reward_contract_v1.get("action_space_v1", ["HOLD", "EXIT_NOW"]))
    action_set_json_v1 = json.dumps(action_space_v1, ensure_ascii=True)

    shadow_view = _build_runtime_shadow_attach_view(decision_df, eligible_df)
    overlay_view = _build_runtime_overlay_view(decision_df, eligible_df, as_of_df)

    decision_df, behavior_policy_identity_summary_v1 = shadow_meta._attach_management_behavior_policy_identity_v1(
        decision_df,
        as_of_df,
        closed_trades_df,
    )
    decision_df, deterministic_propensity_summary_v1 = shadow_meta._attach_management_deterministic_propensity_contract_v1(
        decision_df,
        action_space_v1=action_space_v1,
    )
    decision_df = decision_df.merge(
        shadow_view,
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    decision_df = decision_df.merge(
        overlay_view[
            _JOIN_KEYS
            + [
                "overlay_session_axis_v1",
                "overlay_trade_pocket_v1",
                "overlay_vol_axis_v1",
                "overlay_hold_age_axis_v1",
                "overlay_giveback_axis_v1",
                "overlay_composite_v1",
                "overlay_pocket_group_v1",
            ]
        ].drop_duplicates(subset=_JOIN_KEYS, keep="last"),
        on=_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    decision_df["path_dynamics_raw_state_join_mode_v1"] = "NOT_ATTACHED"
    raw_state_join_cols = [
        "as_of_row_uid_v1",
        "run_id",
        "anchor_timestamp_utc",
        "as_of_mgmt_trace_last_peak_ts_utc_v1",
        "as_of_mgmt_trace_last_mfe_ts_utc_v1",
        "as_of_mgmt_trace_peak_price_v1",
        "as_of_mgmt_trace_anchor_price_v1",
        "as_of_mgmt_trace_mfe_bps_at_anchor_v1",
        "as_of_mgmt_trace_last_peak_mfe_bps_v1",
        "as_of_mgmt_trace_max_mfe_without_mae_bps_v1",
        "as_of_mgmt_trace_mfe_mae_sequence_order_v1",
        "as_of_mgmt_trace_last_peak_ts_utc_null_reason_v1",
        "as_of_mgmt_trace_last_mfe_ts_utc_null_reason_v1",
        "as_of_mgmt_trace_last_peak_mfe_bps_null_reason_v1",
        "as_of_mgmt_trace_max_mfe_without_mae_bps_null_reason_v1",
        "as_of_mgmt_trace_mfe_mae_sequence_order_null_reason_v1",
    ]
    raw_state_view = raw_state_df[[
        column_name for column_name in raw_state_join_cols if column_name in raw_state_df.columns
    ]].copy()
    raw_state_view["as_of_row_uid_v1"] = raw_state_view["as_of_row_uid_v1"].astype("string")
    raw_state_view["run_id"] = raw_state_view["run_id"].astype("string")
    raw_state_view = raw_state_view.drop_duplicates(subset=["as_of_row_uid_v1"], keep="last")
    decision_df["as_of_row_uid_v1"] = decision_df["as_of_row_uid_v1"].astype("string")
    decision_df["run_id"] = decision_df["run_id"].astype("string")
    decision_df = decision_df.merge(
        raw_state_view,
        on=["as_of_row_uid_v1", "run_id"],
        how="left",
        validate="one_to_one",
    )
    path_fields = [
        "as_of_mgmt_trace_last_peak_ts_utc_v1",
        "as_of_mgmt_trace_last_mfe_ts_utc_v1",
        "as_of_mgmt_trace_peak_price_v1",
        "as_of_mgmt_trace_anchor_price_v1",
        "as_of_mgmt_trace_mfe_bps_at_anchor_v1",
        "as_of_mgmt_trace_last_peak_mfe_bps_v1",
        "as_of_mgmt_trace_max_mfe_without_mae_bps_v1",
        "as_of_mgmt_trace_mfe_mae_sequence_order_v1",
    ]
    attached_mask = decision_df[path_fields].notna().any(axis=1)
    decision_df.loc[attached_mask, "path_dynamics_raw_state_join_mode_v1"] = "AS_OF_ROW_UID_EXACT"

    decision_df["build_id_v1"] = build_id_v1
    decision_df["build_timestamp_utc_v1"] = build_timestamp_utc_v1
    decision_df["source_control_date_v1"] = shadow_meta._scalar_string(source_control_date_v1)
    decision_df["record_semantic_layer_v1"] = "AS_OF_DECISION_LOG"
    decision_df["decision_domain_v1"] = "MANAGEMENT"
    decision_df["logging_record_timestamp_utc_v1"] = build_timestamp_utc_v1
    decision_df["decision_ts_utc_v1"] = decision_df["decision_timestamp"].map(shadow_meta._scalar_iso_utc).astype("string")
    if "decision_anchor_timestamp_utc_v1" not in decision_df.columns:
        decision_df["decision_anchor_timestamp_utc_v1"] = decision_df.get("anchor_timestamp_utc", _LEDGER_NOT_AVAILABLE)
    decision_df["decision_anchor_timestamp_utc_v1"] = _safe_string(decision_df["decision_anchor_timestamp_utc_v1"])
    decision_df["observed_action_v1"] = decision_df["action_label_v1"].astype("string")
    decision_df["available_action_set_v1"] = action_set_json_v1
    decision_df["available_action_set_status_v1"] = "CONTRACT_LEVEL_ESTABLISHED"
    decision_df["available_action_set_source_v1"] = shadow_meta._scalar_string(
        management_bandit_action_reward_contract_v1.get("layer_name"),
        default="MANAGEMENT_BANDIT_ACTION_REWARD_CONTRACT_V1",
    )
    decision_df["shadow_model_version_v1"] = decision_df["shadow_model_source_v1"].astype("string")
    decision_df["shadow_model_version_status_v1"] = "RUNTIME_EXIT_LOCAL_PRIMARY_MODEL_ATTACHED"
    decision_df["decision_provenance_v1"] = "MONDAY_RUNTIME_ONLY_POLICY_LOGGING"
    decision_df["shadow_provenance_v1"] = "RUNTIME_EXIT_LOCAL_SCORE"
    decision_df["manual_review_provenance_v1"] = "RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT"
    decision_df["review_priority_tier_v1"] = "RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT"
    decision_df["support_tier_v1"] = "RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT"
    decision_df["review_rank_v1"] = pd.Series([pd.NA] * len(decision_df), index=decision_df.index, dtype="Int64")

    rename_map = {
        "as_of_mgmt_trace_last_peak_ts_utc_v1": "as_of_management_core_last_peak_ts_utc_v1",
        "as_of_mgmt_trace_last_mfe_ts_utc_v1": "as_of_management_core_last_mfe_ts_utc_v1",
        "as_of_mgmt_trace_peak_price_v1": "as_of_management_core_peak_price_v1",
        "as_of_mgmt_trace_anchor_price_v1": "as_of_management_core_anchor_price_v1",
        "as_of_mgmt_trace_mfe_bps_at_anchor_v1": "as_of_management_core_mfe_bps_at_anchor_v1",
        "as_of_mgmt_trace_last_peak_mfe_bps_v1": "as_of_management_core_last_peak_mfe_bps_v1",
        "as_of_mgmt_trace_max_mfe_without_mae_bps_v1": "as_of_management_core_max_mfe_without_mae_bps_v1",
        "as_of_mgmt_trace_mfe_mae_sequence_order_v1": "as_of_management_core_mfe_mae_sequence_order_v1",
        "as_of_mgmt_trace_last_peak_ts_utc_null_reason_v1": "as_of_management_core_last_peak_ts_utc_null_reason_v1",
        "as_of_mgmt_trace_last_mfe_ts_utc_null_reason_v1": "as_of_management_core_last_mfe_ts_utc_null_reason_v1",
        "as_of_mgmt_trace_last_peak_mfe_bps_null_reason_v1": "as_of_management_core_last_peak_mfe_bps_null_reason_v1",
        "as_of_mgmt_trace_max_mfe_without_mae_bps_null_reason_v1": "as_of_management_core_max_mfe_without_mae_bps_null_reason_v1",
        "as_of_mgmt_trace_mfe_mae_sequence_order_null_reason_v1": "as_of_management_core_mfe_mae_sequence_order_null_reason_v1",
    }
    decision_df = decision_df.rename(columns=rename_map)

    decision_df = _ensure_columns(
        decision_df,
        {
            "behavior_policy_id_v1": _LEDGER_NOT_AVAILABLE,
            "behavior_policy_id_status_v1": _LEDGER_IKKE_ETABLERT,
            "behavior_policy_kind_v1": _LEDGER_IKKE_ETABLERT,
            "observed_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
            "observed_action_propensity_v1": np.nan,
            "behavior_policy_action_space_v1": action_set_json_v1,
            "per_action_propensity_vector_v1": _LEDGER_NOT_AVAILABLE,
            "propensity_hold_v1": np.nan,
            "propensity_exit_now_v1": np.nan,
            "bandit_action_reward_eligibility_status_v1": _LEDGER_IKKE_ETABLERT,
            "bandit_reward_locality_status_v1": _LEDGER_IKKE_ETABLERT,
            "terminal_outcome_availability_status_v1": _LEDGER_IKKE_ETABLERT,
            "overlay_session_axis_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_trade_pocket_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_vol_axis_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_hold_age_axis_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_giveback_axis_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_composite_v1": _LEDGER_NOT_AVAILABLE,
            "overlay_pocket_group_v1": "OTHER",
            "shadow_score_v1": np.nan,
            "shadow_bucket_status_v1": _LEDGER_IKKE_ETABLERT,
            "shadow_bucket_rank_v1": pd.Series([pd.NA] * len(decision_df), index=decision_df.index, dtype="Int64"),
            "shadow_domain_status_v1": _LEDGER_IKKE_ETABLERT,
            "shadow_score_source_v1": _LEDGER_NOT_AVAILABLE,
            "shadow_usage_status_v1": _LEDGER_IKKE_ETABLERT,
            "shadow_counterfactual_status_v1": _LEDGER_IKKE_ETABLERT,
            "research_priority_status_v1": _LEDGER_IKKE_ETABLERT,
            "as_of_management_core_last_peak_ts_utc_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_last_mfe_ts_utc_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_peak_price_v1": np.nan,
            "as_of_management_core_anchor_price_v1": np.nan,
            "as_of_management_core_mfe_bps_at_anchor_v1": np.nan,
            "as_of_management_core_last_peak_mfe_bps_v1": np.nan,
            "as_of_management_core_max_mfe_without_mae_bps_v1": np.nan,
            "as_of_management_core_mfe_mae_sequence_order_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_last_peak_ts_utc_null_reason_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_last_mfe_ts_utc_null_reason_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_last_peak_mfe_bps_null_reason_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_max_mfe_without_mae_bps_null_reason_v1": _LEDGER_NOT_AVAILABLE,
            "as_of_management_core_mfe_mae_sequence_order_null_reason_v1": _LEDGER_NOT_AVAILABLE,
        },
    )
    for field_name in _OBS_FIELDS:
        if field_name not in decision_df.columns:
            decision_df[field_name] = pd.NA

    decision_columns = [
        "build_id_v1",
        "build_timestamp_utc_v1",
        "logging_record_timestamp_utc_v1",
        "source_control_date_v1",
        "record_semantic_layer_v1",
        "run_id",
        "decision_domain_v1",
        "management_row_key_v1",
        "candidate_uid_exact_v1",
        "trade_uid_exact_v1",
        "trade_id_exact_v1",
        "as_of_row_uid_v1",
        "decision_anchor_type_v1",
        "decision_timestamp",
        "decision_ts_utc_v1",
        "decision_anchor_timestamp_utc_v1",
        "path_dynamics_raw_state_join_mode_v1",
        "observed_action_v1",
        "available_action_set_v1",
        "available_action_set_status_v1",
        "available_action_set_source_v1",
        "policy_version_v1",
        "policy_version_status_v1",
        "behavior_policy_identity_source_v1",
        "behavior_policy_id_v1",
        "behavior_policy_id_status_v1",
        "behavior_policy_kind_v1",
        "as_of_candidate_policy_hash_v1",
        "as_of_candidate_entry_bundle_sha256_v1",
        "as_of_candidate_exit_bundle_sha256_v1",
        "shadow_model_version_v1",
        "shadow_model_version_status_v1",
        "decision_provenance_v1",
        "shadow_provenance_v1",
        "manual_review_provenance_v1",
        "observed_action_status_v1",
        "observed_action_source_v1",
        "observed_action_propensity_status_v1",
        "policy_logging_propensity_status_v1",
        "observed_action_propensity_v1",
        "behavior_policy_action_space_v1",
        "per_action_propensity_vector_v1",
        "propensity_hold_v1",
        "propensity_exit_now_v1",
        "bandit_action_reward_eligibility_status_v1",
        "bandit_reward_locality_status_v1",
        "terminal_outcome_availability_status_v1",
        "route_status_v1",
        "entry_actualization_presence_status_v1",
        "rl_transition_eligibility_status_v1",
        "management_path_relation_v1",
        "sequence_dataset_membership_v1",
        "sequence_next_link_status_v1",
        "sequence_terminal_step_status_v1",
        "split_bucket_v1",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "activation_origin_v1",
        "shadow_score_v1",
        "shadow_bucket_status_v1",
        "shadow_bucket_rank_v1",
        "shadow_domain_status_v1",
        "shadow_score_source_v1",
        "shadow_usage_status_v1",
        "shadow_counterfactual_status_v1",
        "research_priority_status_v1",
        "review_rank_v1",
        "review_priority_tier_v1",
        "support_tier_v1",
        "overlay_session_axis_v1",
        "overlay_trade_pocket_v1",
        "overlay_vol_axis_v1",
        "overlay_hold_age_axis_v1",
        "overlay_giveback_axis_v1",
        "overlay_composite_v1",
        "overlay_pocket_group_v1",
        "as_of_management_core_last_peak_ts_utc_v1",
        "as_of_management_core_last_mfe_ts_utc_v1",
        "as_of_management_core_peak_price_v1",
        "as_of_management_core_anchor_price_v1",
        "as_of_management_core_mfe_bps_at_anchor_v1",
        "as_of_management_core_last_peak_mfe_bps_v1",
        "as_of_management_core_max_mfe_without_mae_bps_v1",
        "as_of_management_core_mfe_mae_sequence_order_v1",
        "as_of_management_core_last_peak_ts_utc_null_reason_v1",
        "as_of_management_core_last_mfe_ts_utc_null_reason_v1",
        "as_of_management_core_last_peak_mfe_bps_null_reason_v1",
        "as_of_management_core_max_mfe_without_mae_bps_null_reason_v1",
        "as_of_management_core_mfe_mae_sequence_order_null_reason_v1",
        *_OBS_FIELDS,
    ]
    management_policy_logging_decision_log_harness_v1_df = decision_df[decision_columns].copy()

    backfill_base_df = direct_method_df[
        _JOIN_KEYS
        + [
            "management_row_key_v1",
            "run_id",
            "as_of_row_uid_v1",
            "decision_timestamp",
            "action_label_v1",
        ]
    ].copy()
    closed_trade_cols = [
        "candidate_uid_exact_v1",
        "trade_uid_exact_v1",
        "trade_id_exact_v1",
        "entry_timestamp",
        "exit_timestamp",
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "holding_time_bars",
        "trade_outcome_class",
        "exit_reason",
        "good_exit",
        "premature_exit",
        "late_exit",
        "hindsight_management_review_v1",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
    ]
    hindsight_cols = [
        "candidate_uid_exact_v1",
        "trade_uid_exact_v1",
        "trade_id_exact_v1",
        "review_exit_bucket_v1",
        "review_entry_bucket_v1",
        "hindsight_rl_review_reason_v1",
        "hindsight_rl_review_domain_support_v1",
        "hindsight_peak_to_worst_after_peak_bps_v1",
    ]
    backfill_df = backfill_base_df.merge(
        closed_trades_df[closed_trade_cols].drop_duplicates(
            subset=["candidate_uid_exact_v1", "trade_uid_exact_v1", "trade_id_exact_v1"],
            keep="last",
        ),
        on=["candidate_uid_exact_v1", "trade_uid_exact_v1", "trade_id_exact_v1"],
        how="left",
        validate="one_to_one",
    )
    backfill_df = backfill_df.merge(
        hindsight_review_export_df[hindsight_cols].drop_duplicates(
            subset=["candidate_uid_exact_v1", "trade_uid_exact_v1", "trade_id_exact_v1"],
            keep="last",
        ),
        on=["candidate_uid_exact_v1", "trade_uid_exact_v1", "trade_id_exact_v1"],
        how="left",
        validate="one_to_one",
    )
    backfill_df["build_id_v1"] = build_id_v1
    backfill_df["build_timestamp_utc_v1"] = build_timestamp_utc_v1
    backfill_df["record_semantic_layer_v1"] = "HINDSIGHT_OUTCOME_BACKFILL"
    backfill_df["decision_domain_v1"] = "MANAGEMENT"
    backfill_df["decision_anchor_type_v1"] = backfill_df["decision_anchor_type_v1"].astype("string")
    backfill_df["decision_ts_utc_v1"] = backfill_df["decision_timestamp"].map(shadow_meta._scalar_iso_utc).astype("string")
    backfill_df["entry_ts_utc_v1"] = backfill_df["entry_timestamp"].map(shadow_meta._scalar_iso_utc).astype("string")
    backfill_df["exit_ts_utc_v1"] = backfill_df["exit_timestamp"].map(shadow_meta._scalar_iso_utc).astype("string")
    backfill_df["observed_action_v1"] = backfill_df["action_label_v1"].astype("string")
    backfill_df["outcome_backfill_source_v1"] = "CLOSED_TRADE_LEDGER_AND_HINDSIGHT_EXPORT"
    backfill_df["outcome_backfill_status_v1"] = np.where(
        pd.to_numeric(backfill_df["realized_pnl_bps"], errors="coerce").notna(),
        "EXACT_TERMINAL_OUTCOME_BACKFILL",
        "OUTCOME_BACKFILL_MISSING",
    )
    outcome_columns = [
        "build_id_v1",
        "build_timestamp_utc_v1",
        "record_semantic_layer_v1",
        "run_id",
        "decision_domain_v1",
        "management_row_key_v1",
        "candidate_uid_exact_v1",
        "trade_uid_exact_v1",
        "trade_id_exact_v1",
        "as_of_row_uid_v1",
        "decision_anchor_type_v1",
        "decision_timestamp",
        "decision_ts_utc_v1",
        "observed_action_v1",
        "entry_timestamp",
        "entry_ts_utc_v1",
        "exit_timestamp",
        "exit_ts_utc_v1",
        "trade_outcome_class",
        "exit_reason",
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "holding_time_bars",
        "good_exit",
        "premature_exit",
        "late_exit",
        "hindsight_management_review_v1",
        "review_exit_bucket_v1",
        "review_entry_bucket_v1",
        "hindsight_rl_review_reason_v1",
        "hindsight_rl_review_domain_support_v1",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
        "hindsight_peak_to_worst_after_peak_bps_v1",
        "outcome_backfill_source_v1",
        "outcome_backfill_status_v1",
    ]
    management_policy_logging_outcome_backfill_harness_v1_df = backfill_df[outcome_columns].copy()

    missing_fields_payload_v1 = {
        "layer_name": "MANAGEMENT_POLICY_LOGGING_MISSING_FIELDS_V1",
        "context_v1": "MONDAY_RUNTIME_ONLY_MANAGEMENT_POLICY_LOGGING",
        "field_status_v1": [
            {
                "field_name_v1": "as_of_management_core_last_peak_ts_utc_v1",
                "status_v1": "BEVIST",
                "reason_v1": "Exact raw-state attach now carries last-peak timestamp at management anchor.",
            },
            {
                "field_name_v1": "as_of_management_core_last_mfe_ts_utc_v1",
                "status_v1": "BEVIST",
                "reason_v1": "Exact raw-state attach now carries last-MFE timestamp at management anchor.",
            },
            {
                "field_name_v1": "as_of_management_core_last_peak_mfe_bps_v1",
                "status_v1": "BEVIST",
                "reason_v1": "Exact raw-state attach now carries last-peak MFE at management anchor.",
            },
            {
                "field_name_v1": "as_of_management_core_max_mfe_without_mae_bps_v1",
                "status_v1": "BEVIST",
                "reason_v1": "Exact raw-state attach now carries max-MFE-without-MAE at management anchor.",
            },
            {
                "field_name_v1": "as_of_management_core_mfe_mae_sequence_order_v1",
                "status_v1": "BEVIST",
                "reason_v1": "Exact raw-state attach now carries MFE/MAE sequence order at management anchor.",
            },
            {
                "field_name_v1": "review_rank_v1",
                "status_v1": "IKKE_ETABLERT",
                "reason_v1": "Legacy Wednesday manual-review packet is retired and not rebuilt on the Monday-native root.",
            },
            {
                "field_name_v1": "review_priority_tier_v1",
                "status_v1": "IKKE_ETABLERT",
                "reason_v1": "Legacy manual-review priority tiers are not rebuilt on the Monday-native root.",
            },
            {
                "field_name_v1": "support_tier_v1",
                "status_v1": "IKKE_ETABLERT",
                "reason_v1": "Legacy manual-review support tiers are not rebuilt on the Monday-native root.",
            },
            {
                "field_name_v1": "policy_version_v1",
                "status_v1": (
                    "BEVIST"
                    if int(behavior_policy_identity_summary_v1["policy_hash_available_rows_v1"]) == int(len(decision_df))
                    else "DELVIS_ETABLERT"
                ),
                "reason_v1": "Behavior-policy identity is attached from exact AS_OF candidate policy hash and lineage where available.",
            },
            {
                "field_name_v1": "observed_action_propensity_v1",
                "status_v1": (
                    "DELVIS_ETABLERT"
                    if int(deterministic_propensity_summary_v1["deterministic_propensity_rows_v1"]) > 0
                    else "IKKE_ETABLERT"
                ),
                "reason_v1": "Observed propensity is exact only for deterministic rows with exact policy hash and exact observed action.",
            },
            {
                "field_name_v1": "exploration_metadata_v1",
                "status_v1": "IKKE_ETABLERT",
                "reason_v1": "No exploration metadata is persisted in the frozen management path.",
            },
        ],
    }

    consistency_rows = [
        {
            "check_name_v1": "DECISION_LOG_COVERS_CURRENT_DM_ELIGIBLE_ROWS",
            "status_v1": "PASS"
            if int(len(management_policy_logging_decision_log_harness_v1_df)) == int(len(direct_method_df)) and int(len(direct_method_df)) > 0
            else "FAIL",
            "observed_value_v1": int(len(management_policy_logging_decision_log_harness_v1_df)),
            "expected_value_v1": int(len(direct_method_df)),
        },
        {
            "check_name_v1": "DECISION_LOG_HAS_NO_HINDSIGHT_COLUMNS",
            "status_v1": "PASS"
            if all("hindsight_" not in column_name for column_name in management_policy_logging_decision_log_harness_v1_df.columns)
            else "FAIL",
            "observed_value_v1": [
                column_name for column_name in management_policy_logging_decision_log_harness_v1_df.columns if "hindsight_" in column_name
            ],
            "expected_value_v1": [],
        },
        {
            "check_name_v1": "OUTCOME_BACKFILL_COVERS_EXACT_SAME_CURRENT_ROWS",
            "status_v1": "PASS"
            if set(management_policy_logging_outcome_backfill_harness_v1_df["management_row_key_v1"].astype("string"))
            == set(management_policy_logging_decision_log_harness_v1_df["management_row_key_v1"].astype("string"))
            else "FAIL",
            "observed_value_v1": int(len(management_policy_logging_outcome_backfill_harness_v1_df)),
            "expected_value_v1": int(len(management_policy_logging_decision_log_harness_v1_df)),
        },
        {
            "check_name_v1": "RUNTIME_SCORE_ATTACH_EXACT_FOR_ALL_DECISION_ROWS",
            "status_v1": "PASS"
            if int(management_policy_logging_decision_log_harness_v1_df["shadow_score_v1"].isna().sum()) == 0
            else "FAIL",
            "observed_value_v1": int(management_policy_logging_decision_log_harness_v1_df["shadow_score_v1"].isna().sum()),
            "expected_value_v1": 0,
        },
        {
            "check_name_v1": "OVERLAY_TAGS_ESTABLISHED_FOR_ALL_DECISION_ROWS",
            "status_v1": "PASS"
            if int(
                management_policy_logging_decision_log_harness_v1_df["overlay_composite_v1"].astype("string").eq(_LEDGER_NOT_AVAILABLE).sum()
            )
            == 0
            else "FAIL",
            "observed_value_v1": int(
                management_policy_logging_decision_log_harness_v1_df["overlay_composite_v1"].astype("string").eq(_LEDGER_NOT_AVAILABLE).sum()
            ),
            "expected_value_v1": 0,
        },
        {
            "check_name_v1": "PATH_DYNAMICS_RAW_STATE_ATTACH_EXACT_FOR_ALL_DECISION_ROWS",
            "status_v1": "PASS"
            if int(
                management_policy_logging_decision_log_harness_v1_df["path_dynamics_raw_state_join_mode_v1"].astype("string").eq("AS_OF_ROW_UID_EXACT").sum()
            )
            == int(len(management_policy_logging_decision_log_harness_v1_df))
            else "FAIL",
            "observed_value_v1": {
                str(key): int(value)
                for key, value in management_policy_logging_decision_log_harness_v1_df["path_dynamics_raw_state_join_mode_v1"]
                .astype("string")
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
            "expected_value_v1": {"AS_OF_ROW_UID_EXACT": int(len(management_policy_logging_decision_log_harness_v1_df))},
        },
        {
            "check_name_v1": "MANUAL_REVIEW_CONTEXT_EXPLICITLY_RETIRED",
            "status_v1": "PASS"
            if int(management_policy_logging_decision_log_harness_v1_df["review_rank_v1"].notna().sum()) == 0
            and int(
                management_policy_logging_decision_log_harness_v1_df["manual_review_provenance_v1"]
                .astype("string")
                .eq("RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT")
                .sum()
            )
            == int(len(management_policy_logging_decision_log_harness_v1_df))
            else "FAIL",
            "observed_value_v1": {
                "review_rank_non_null_v1": int(management_policy_logging_decision_log_harness_v1_df["review_rank_v1"].notna().sum()),
                "retired_provenance_rows_v1": int(
                    management_policy_logging_decision_log_harness_v1_df["manual_review_provenance_v1"]
                    .astype("string")
                    .eq("RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT")
                    .sum()
                ),
            },
            "expected_value_v1": {
                "review_rank_non_null_v1": 0,
                "retired_provenance_rows_v1": int(len(management_policy_logging_decision_log_harness_v1_df)),
            },
        },
        {
            "check_name_v1": "AVAILABLE_ACTION_SET_STAYS_HOLD_EXIT_NOW",
            "status_v1": "PASS" if action_space_v1 == ["HOLD", "EXIT_NOW"] else "FAIL",
            "observed_value_v1": action_space_v1,
            "expected_value_v1": ["HOLD", "EXIT_NOW"],
        },
        {
            "check_name_v1": "PROPENSITY_STATUS_MATCHES_DETERMINISTIC_POLICY_HASH_COVERAGE",
            "status_v1": "PASS"
            if {
                str(key): int(value)
                for key, value in management_policy_logging_decision_log_harness_v1_df["policy_logging_propensity_status_v1"]
                .astype("string")
                .value_counts(dropna=False)
                .to_dict()
                .items()
            }
            == deterministic_propensity_summary_v1["policy_logging_propensity_status_counts_v1"]
            else "FAIL",
            "observed_value_v1": {
                str(key): int(value)
                for key, value in management_policy_logging_decision_log_harness_v1_df["policy_logging_propensity_status_v1"]
                .astype("string")
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
            "expected_value_v1": deterministic_propensity_summary_v1["policy_logging_propensity_status_counts_v1"],
        },
        shadow_meta._build_join_leakage_consistency_row_v1(
            as_of_supervision_join_coverage_summary,
            leakage_guard_summary,
        ),
    ]
    management_policy_logging_consistency_audit_v1_df = pd.DataFrame.from_records(consistency_rows)
    failed_checks = int((management_policy_logging_consistency_audit_v1_df["status_v1"].astype("string") != "PASS").sum())

    management_policy_logging_spec_v1 = {
        "layer_name": "MANAGEMENT_POLICY_LOGGING_SPEC_V1",
        "scope_v1": "MANAGEMENT_ONLY",
        "mode_v1": "APPEND_ONLY|MONDAY_RUNTIME_ONLY_POLICY_LOGGING|AS_OF_HINDSIGHT_SPLIT",
        "build_id_v1": build_id_v1,
        "build_timestamp_utc_v1": build_timestamp_utc_v1,
        "source_control_date_v1": shadow_meta._scalar_string(source_control_date_v1),
        "natural_flow_position_v1": "MATERIALIZED_AFTER_MONDAY_CANONICAL_LEDGER_REBUILD_WITHOUT_RETIRED_WEDNESDAY_MANUAL_REVIEW_PACKET",
        "decision_log_table_v1": _POLICY_LOGGING_DECISION,
        "outcome_backfill_table_v1": _POLICY_LOGGING_OUTCOME,
        "decision_log_primary_key_v1": _JOIN_KEYS,
        "decision_log_semantics_v1": "AS_OF_ONLY_DECISION_LOG",
        "outcome_backfill_semantics_v1": "HINDSIGHT_ONLY_OUTCOME_BACKFILL",
        "action_space_v1": action_space_v1,
        "observation_feature_names_v1": _OBS_FIELDS,
        "path_dynamics_logging_fields_v1": [
            "as_of_management_core_last_peak_ts_utc_v1",
            "as_of_management_core_last_mfe_ts_utc_v1",
            "as_of_management_core_peak_price_v1",
            "as_of_management_core_anchor_price_v1",
            "as_of_management_core_mfe_bps_at_anchor_v1",
            "as_of_management_core_last_peak_mfe_bps_v1",
            "as_of_management_core_max_mfe_without_mae_bps_v1",
            "as_of_management_core_mfe_mae_sequence_order_v1",
        ],
        "decision_log_field_order_v1": decision_columns,
        "outcome_backfill_field_order_v1": outcome_columns,
        "source_artifacts_v1": [
            "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet",
            "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet",
            "shadow_meta_all_trade_review_management_exit_local_all_eligible_scored_view_v1.parquet",
            "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet",
            "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet",
            "shadow_meta_all_trade_review_ledger_closed_trades.parquet",
            "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet",
        ],
        "retired_context_v1": {
            "legacy_manual_review_packet_v1": "NOT_REBUILT_ON_MONDAY_ROOT",
            "legacy_shadow_candidate_review_v1": "NOT_REBUILT_ON_MONDAY_ROOT",
            "legacy_shadow_hold_research_v1": "NOT_REBUILT_ON_MONDAY_ROOT",
        },
        "runtime_score_source_v1": "primary_model_score_v1 from management_exit_local_all_eligible_scored_view_v1",
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
        "not_propensity_estimator_v1": True,
        "as_of_hindsight_physical_separation_v1": True,
        "behavior_policy_version_status_v1": behavior_policy_identity_summary_v1[
            "behavior_policy_identity_attachment_status_v1"
        ],
        "behavior_policy_propensity_enrichment_v1": deterministic_propensity_summary_v1["propensity_readiness_v1"],
        "propensity_status_v1": shadow_meta._scalar_string(management_bandit_status_v1.get("MANAGEMENT_BANDIT_PROPENSITY_STATUS")),
        "observed_action_contract_v1": management_bandit_observed_action_contract_v1,
    }
    management_policy_logging_summary_v1 = {
        "layer_name": "MANAGEMENT_POLICY_LOGGING_SUMMARY_V1",
        "build_id_v1": build_id_v1,
        "build_timestamp_utc_v1": build_timestamp_utc_v1,
        "source_control_date_v1": shadow_meta._scalar_string(source_control_date_v1),
        "instrumentation_context_v1": "MONDAY_RUNTIME_ONLY_POLICY_LOGGING",
        "observed_sample_rows_v1": int(len(observed_sample_df)),
        "decision_log_rows_v1": int(len(management_policy_logging_decision_log_harness_v1_df)),
        "outcome_backfill_rows_v1": int(len(management_policy_logging_outcome_backfill_harness_v1_df)),
        "manual_review_rows_attached_v1": 0,
        "retired_manual_review_context_v1": True,
        "action_space_v1": action_space_v1,
        "observed_action_counts_v1": {
            str(key): int(value)
            for key, value in management_policy_logging_decision_log_harness_v1_df["observed_action_v1"]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "observed_action_status_counts_v1": {
            str(key): int(value)
            for key, value in management_policy_logging_decision_log_harness_v1_df["observed_action_status_v1"]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "propensity_status_counts_v1": {
            str(key): int(value)
            for key, value in management_policy_logging_decision_log_harness_v1_df["policy_logging_propensity_status_v1"]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "policy_version_status_counts_v1": behavior_policy_identity_summary_v1["policy_version_status_counts_v1"],
        "behavior_policy_identity_summary_v1": behavior_policy_identity_summary_v1,
        "deterministic_propensity_summary_v1": deterministic_propensity_summary_v1,
        "path_dynamics_raw_state_join_mode_counts_v1": {
            str(key): int(value)
            for key, value in management_policy_logging_decision_log_harness_v1_df["path_dynamics_raw_state_join_mode_v1"]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "overlay_pocket_group_counts_v1": {
            str(key): int(value)
            for key, value in management_policy_logging_decision_log_harness_v1_df["overlay_pocket_group_v1"]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "decision_log_sample_rows_v1": _sample_rows(
            management_policy_logging_decision_log_harness_v1_df,
            [
                "run_id",
                "candidate_uid_exact_v1",
                "decision_ts_utc_v1",
                "observed_action_v1",
                "shadow_score_v1",
                "shadow_bucket_status_v1",
                "manual_review_provenance_v1",
                "overlay_composite_v1",
            ],
        ),
        "outcome_backfill_sample_rows_v1": _sample_rows(
            management_policy_logging_outcome_backfill_harness_v1_df,
            [
                "run_id",
                "candidate_uid_exact_v1",
                "decision_ts_utc_v1",
                "observed_action_v1",
                "exit_ts_utc_v1",
                "realized_pnl_bps",
                "trade_outcome_class",
                "review_exit_bucket_v1",
            ],
        ),
        "instrumentation_status_v1": "BEVIST" if failed_checks == 0 else "IKKE_ETABLERT",
        "behavior_policy_readiness_v1": behavior_policy_identity_summary_v1["behavior_policy_readiness_v1"],
        "propensity_readiness_v1": deterministic_propensity_summary_v1["propensity_readiness_v1"],
        "note_v1": "Monday-native runtime policy logging keeps AS_OF decision logging physically separate from hindsight outcome backfill without reviving retired Wednesday manual-review/shadow packets.",
    }
    return {
        "management_policy_logging_spec_v1": management_policy_logging_spec_v1,
        "management_policy_logging_decision_log_harness_v1_df": management_policy_logging_decision_log_harness_v1_df,
        "management_policy_logging_outcome_backfill_harness_v1_df": management_policy_logging_outcome_backfill_harness_v1_df,
        "management_policy_logging_missing_fields_v1": missing_fields_payload_v1,
        "management_policy_logging_summary_v1": management_policy_logging_summary_v1,
        "management_policy_logging_consistency_audit_v1_df": management_policy_logging_consistency_audit_v1_df,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_V1",
        "",
        f"- Decision: `{summary['decision_v1']}`",
        f"- Ledger dir: `{summary['ledger_dir_v1']}`",
        f"- Decision-log rows: `{summary['policy_logging_summary_v1']['decision_log_rows_v1']}`",
        f"- Outcome-backfill rows: `{summary['policy_logging_summary_v1']['outcome_backfill_rows_v1']}`",
        f"- Failed consistency checks: `{summary['failed_consistency_count_v1']}`",
        "",
        "## Hard Status",
    ]
    for bucket, items in summary["hard_status_division_v1"].items():
        lines.append(f"### {bucket}")
        for item in items:
            lines.append(f"- {item}")
    if summary.get("comparison_decision_v1") is not None or summary.get("path_dynamics_decision_v1") is not None:
        lines.extend(["", "## Follow-On Reruns"])
        if summary.get("comparison_decision_v1") is not None:
            lines.append(f"- `monday_native_shadow_refreeze_comparison`: `{summary['comparison_decision_v1']}`")
        if summary.get("path_dynamics_decision_v1") is not None:
            lines.append(f"- `path_dynamics_logging_v2`: `{summary['path_dynamics_decision_v1']}`")
    return "\n".join(lines) + "\n"


def materialize(
    reports_root: str | Path | None = None,
    *,
    ledger_dir: str | Path | None = None,
    extension_dir: str | Path | None = None,
    rerun_follow_on: bool = True,
) -> dict[str, Any]:
    root = _resolve_reports_root(str(reports_root) if reports_root is not None else None)
    active_ledger_dir = _resolve_ledger_dir(root, str(ledger_dir) if ledger_dir is not None else None)
    out_dir = Path(extension_dir).expanduser().resolve() if extension_dir is not None else _default_extension_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_summary = _load_json(active_ledger_dir / "shadow_meta_all_trade_review_ledger_build_summary.json")
    artifact_paths = build_summary.get("artifact_paths", {})
    required_keys = {
        "as_of_decision_moment_ledger_path": active_ledger_dir / "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet",
        "closed_trades_path": active_ledger_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet",
        "hindsight_review_export_path": active_ledger_dir / "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet",
        "management_anchor_raw_state_path": active_ledger_dir / "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet",
        "management_bandit_observed_sample_view_path": active_ledger_dir / "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet",
        "management_bandit_direct_method_candidate_view_path": active_ledger_dir / "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet",
        "management_bandit_action_reward_contract_path": active_ledger_dir / "shadow_meta_all_trade_review_management_bandit_action_reward_contract_v1.json",
        "management_bandit_observed_action_contract_path": active_ledger_dir / "shadow_meta_all_trade_review_management_bandit_observed_action_contract_v1.json",
        "management_bandit_status_path": active_ledger_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json",
        "management_exit_local_all_eligible_scored_view_path": active_ledger_dir / "shadow_meta_all_trade_review_management_exit_local_all_eligible_scored_view_v1.parquet",
    }
    resolved_paths: dict[str, Path] = {}
    for key, default_path in required_keys.items():
        raw = artifact_paths.get(key)
        resolved = Path(str(raw)).expanduser().resolve() if raw else default_path
        if not resolved.exists():
            raise FileNotFoundError(f"Missing required artifact for {key}: {resolved}")
        resolved_paths[key] = resolved

    as_of_df = shadow_meta._rename_exact_join_ids_v1(pd.read_parquet(resolved_paths["as_of_decision_moment_ledger_path"]))
    closed_trades_df = shadow_meta._rename_exact_join_ids_v1(pd.read_parquet(resolved_paths["closed_trades_path"]))
    hindsight_review_export_df = shadow_meta._rename_exact_join_ids_v1(pd.read_parquet(resolved_paths["hindsight_review_export_path"]))
    raw_state_df = pd.read_parquet(resolved_paths["management_anchor_raw_state_path"])
    observed_sample_df = pd.read_parquet(resolved_paths["management_bandit_observed_sample_view_path"])
    direct_method_df = pd.read_parquet(resolved_paths["management_bandit_direct_method_candidate_view_path"])
    eligible_df = pd.read_parquet(resolved_paths["management_exit_local_all_eligible_scored_view_path"])

    build_id_v1 = f"MONDAY_RUNTIME_POLICY_LOGGING_{_utc_stamp()}"
    build_timestamp_utc_v1 = _utc_now().isoformat()
    payload = _build_runtime_only_policy_logging_payload(
        observed_sample_df=observed_sample_df,
        direct_method_df=direct_method_df,
        eligible_df=eligible_df,
        raw_state_df=raw_state_df,
        as_of_df=as_of_df,
        closed_trades_df=closed_trades_df,
        hindsight_review_export_df=hindsight_review_export_df,
        management_bandit_action_reward_contract_v1=_load_json(resolved_paths["management_bandit_action_reward_contract_path"]),
        management_bandit_observed_action_contract_v1=_load_json(resolved_paths["management_bandit_observed_action_contract_path"]),
        management_bandit_status_v1=_load_json(resolved_paths["management_bandit_status_path"]),
        as_of_supervision_join_coverage_summary=dict(build_summary.get("as_of_supervision_join_coverage", {})),
        leakage_guard_summary=dict(build_summary.get("leakage_guard", {})),
        build_id_v1=build_id_v1,
        build_timestamp_utc_v1=build_timestamp_utc_v1,
        source_control_date_v1=str(build_summary.get("control_date", "")),
    )

    spec_path = active_ledger_dir / _POLICY_LOGGING_SPEC
    decision_path = active_ledger_dir / _POLICY_LOGGING_DECISION
    outcome_path = active_ledger_dir / _POLICY_LOGGING_OUTCOME
    missing_path = active_ledger_dir / _POLICY_LOGGING_MISSING
    summary_path = active_ledger_dir / _POLICY_LOGGING_SUMMARY
    consistency_path = active_ledger_dir / _POLICY_LOGGING_CONSISTENCY

    _write_json(spec_path, payload["management_policy_logging_spec_v1"])
    payload["management_policy_logging_decision_log_harness_v1_df"].to_parquet(decision_path, index=False)
    payload["management_policy_logging_outcome_backfill_harness_v1_df"].to_parquet(outcome_path, index=False)
    _write_json(missing_path, payload["management_policy_logging_missing_fields_v1"])
    _write_json(summary_path, payload["management_policy_logging_summary_v1"])
    payload["management_policy_logging_consistency_audit_v1_df"].to_csv(consistency_path, index=False)

    failed_consistency = int(
        (payload["management_policy_logging_consistency_audit_v1_df"]["status_v1"].astype("string") != "PASS").sum()
    )
    comparison_decision = None
    comparison_extension_dir = None
    path_dynamics_decision = None
    path_dynamics_extension_dir = None
    if rerun_follow_on:
        comparison_result = refreeze_compare.materialize(
            reports_root=root,
        )
        comparison_decision = comparison_result["summary"]["decision_v1"]
        comparison_extension_dir = comparison_result["extension_dir"]
        benchmark_snapshot_dir = sorted(
            [path for path in root.glob("MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_*") if path.is_dir()],
            key=lambda path: path.name,
        )[-1]
        freeze_dir = benchmark_snapshot_dir / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
        path_out_dir = root / f"PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_{_utc_stamp()}"
        path_result = path_dynamics_audit.materialize(
            reports_root=root,
            freeze_dir=freeze_dir,
            extension_dir=path_out_dir,
        )
        path_dynamics_decision = path_result["summary"]["decision_v1"]
        path_dynamics_extension_dir = path_result["extension_dir"]

    hard_status = {
        "BEVIST": [
            "Monday-native management policy logging artifact is materialized inside the active ledger.",
            "AS_OF decision rows and HINDSIGHT outcome backfill remain physically separated.",
            "Path-dynamics management anchor fields are attached from exact raw-state rows, not synthetic fills.",
        ],
        "INDIKERT": [
            "Legacy Wednesday manual-review/shadow queue context remains retired and is now marked explicitly as not rebuilt.",
            "Monday-native R4/R5/R5.2/R6 refreeze chain still needs a fresh rebuild after policy logging is restored.",
        ],
        "IKKE_ETABLERT": [
            "A Monday-native R6 freeze beating the locked benchmark.",
            "A rebuilt Monday-native manual-review packet replacing the retired Wednesday packet.",
        ],
    }
    decision_v1 = "MONDAY_RUNTIME_POLICY_LOGGING_BUILT"
    if comparison_decision == "MONDAY_COMPARE_READY_REFREEZE_CHAIN_BLOCKED_BY_POLICY_LOGGING":
        decision_v1 = "POLICY_LOGGING_ARTIFACT_BUILT_BUT_COMPARISON_STILL_BLOCKED"

    summary = {
        "layer_name_v1": "MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_V1",
        "built_at_utc_v1": build_timestamp_utc_v1,
        "reports_root_v1": str(root),
        "ledger_dir_v1": str(active_ledger_dir),
        "extension_dir_v1": str(out_dir),
        "policy_logging_summary_v1": payload["management_policy_logging_summary_v1"],
        "policy_logging_artifacts_v1": {
            "spec_path_v1": str(spec_path),
            "decision_log_path_v1": str(decision_path),
            "outcome_backfill_path_v1": str(outcome_path),
            "missing_fields_path_v1": str(missing_path),
            "summary_path_v1": str(summary_path),
            "consistency_audit_path_v1": str(consistency_path),
        },
        "comparison_decision_v1": comparison_decision,
        "comparison_extension_dir_v1": comparison_extension_dir,
        "path_dynamics_decision_v1": path_dynamics_decision,
        "path_dynamics_extension_dir_v1": path_dynamics_extension_dir,
        "failed_consistency_count_v1": failed_consistency,
        "decision_v1": decision_v1,
        "hard_status_division_v1": hard_status,
    }

    consistency_rows = [
        {
            "check_name_v1": "POLICY_LOGGING_DECISION_LOG_WRITTEN",
            "status_v1": "PASS" if decision_path.exists() else "FAIL",
            "observed_v1": str(decision_path),
            "expected_v1": "exists",
        },
        {
            "check_name_v1": "POLICY_LOGGING_OUTCOME_BACKFILL_WRITTEN",
            "status_v1": "PASS" if outcome_path.exists() else "FAIL",
            "observed_v1": str(outcome_path),
            "expected_v1": "exists",
        },
        {
            "check_name_v1": "POLICY_LOGGING_CONSISTENCY_ROWS_PASS",
            "status_v1": "PASS" if failed_consistency == 0 else "FAIL",
            "observed_v1": failed_consistency,
            "expected_v1": 0,
        },
    ]
    if comparison_decision is not None:
        consistency_rows.append(
            {
                "check_name_v1": "FOLLOW_ON_COMPARISON_RERUN_EXECUTED",
                "status_v1": "PASS",
                "observed_v1": comparison_decision,
                "expected_v1": "decision recorded",
            }
        )
    if path_dynamics_decision is not None:
        consistency_rows.append(
            {
                "check_name_v1": "FOLLOW_ON_PATH_DYNAMICS_RERUN_EXECUTED",
                "status_v1": "PASS",
                "observed_v1": path_dynamics_decision,
                "expected_v1": "decision recorded",
            }
        )

    _write_json(
        out_dir / CONTRACT,
        {
            "layer_name_v1": "MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_CONTRACT_V1",
            "scope_v1": "ACTIVE_MONDAY_LEDGER_ONLY",
            "not_live_gate_v1": True,
            "not_controller_v1": True,
            "purpose_v1": "Materialize a Monday-native management policy logging artifact directly from current ledger runtime artifacts without reviving retired Wednesday manual-review/shadow namespaces.",
            "ledger_dir_v1": str(active_ledger_dir),
        },
    )
    _write_json(out_dir / SUMMARY, summary)
    _write_json(out_dir / STATUS, {"decision_v1": decision_v1, "failed_consistency_count_v1": failed_consistency, "not_live_gate_v1": True})
    _write_json(
        out_dir / MANIFEST,
        {
            "layer_name_v1": "MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_MANIFEST_V1",
            "artifacts_v1": [CONTRACT, SUMMARY, REPORT, MANIFEST, STATUS, CONSISTENCY_AUDIT],
            "ledger_written_artifacts_v1": [
                _POLICY_LOGGING_SPEC,
                _POLICY_LOGGING_DECISION,
                _POLICY_LOGGING_OUTCOME,
                _POLICY_LOGGING_MISSING,
                _POLICY_LOGGING_SUMMARY,
                _POLICY_LOGGING_CONSISTENCY,
            ],
        },
    )
    with (out_dir / CONSISTENCY_AUDIT).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(consistency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consistency_rows)
    (out_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    _write_json(root / TOP_LEVEL_SUMMARY, summary)
    return {"extension_dir": str(out_dir), "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--ledger-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--skip-follow-on-reruns", action="store_true")
    args = parser.parse_args(argv)
    materialize(
        reports_root=args.reports_root,
        ledger_dir=args.ledger_dir,
        extension_dir=args.extension_dir,
        rerun_follow_on=not args.skip_follow_on_reruns,
    )


if __name__ == "__main__":
    main()
