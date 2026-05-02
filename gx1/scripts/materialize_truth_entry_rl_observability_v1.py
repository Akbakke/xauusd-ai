#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
ENTRY_RL_EXTENSION_SUFFIX = "ENTRY_RL_OBSERVABILITY_V1"

ENTRY_POLICY_SNAPSHOT_CONTRACT = "shadow_meta_all_trade_review_entry_policy_snapshot_contract_v1.json"
ENTRY_RL_OBSERVABILITY_CONTRACT = "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json"
ENTRY_RL_OBSERVABILITY_VIEW = "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"
ENTRY_RL_OBSERVABILITY_SUMMARY = "shadow_meta_all_trade_review_entry_rl_observability_summary_v1.json"
ENTRY_RL_OBSERVABILITY_STATUS = "shadow_meta_all_trade_review_entry_rl_observability_status_v1.json"
ENTRY_RL_OBSERVABILITY_AUDIT = "shadow_meta_all_trade_review_entry_rl_observability_consistency_audit_v1.csv"
ENTRY_RL_OBSERVABILITY_MANIFEST = "shadow_meta_all_trade_review_entry_rl_observability_manifest_v1.json"
TOP_LEVEL_SUMMARY = "truth_entry_rl_observability_v1.json"

REQUIRED_REVIEW_ARTIFACTS = [
    "shadow_meta_all_trade_review_entry_direct_policy_composite_v1.parquet",
    "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet",
    "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet",
    "shadow_meta_all_trade_review_entry_anchor_raw_state_v1.parquet",
    "shadow_meta_all_trade_review_policy_action_supervision_join_v1.parquet",
    "shadow_meta_all_trade_review_entry_wait_lifecycle_view_v1.parquet",
    "shadow_meta_all_trade_review_entry_actual_take_terminal_outcome_view_v1.parquet",
    "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet",
]

OBSERVATION_FIELDS_V1 = [
    "as_of_hour_utc_v1",
    "as_of_weekday_utc_v1",
    "as_of_session_v1",
    "as_of_side_v1",
    "as_of_atr_bps_v1",
    "as_of_candidate_entry_spread_bps_v1",
    "as_of_candidate_uncertainty_score_v1",
    "as_of_candidate_tradable_prob_v1",
    "as_of_candidate_mfe_first_n_pred_v1",
    "as_of_candidate_trend_regime_v1",
    "as_of_candidate_vol_regime_v1",
    "as_of_entry_candidate_margin_v1",
    "as_of_entry_candidate_path_quality_pred_v1",
    "as_of_skip_xgb_p_flat_v1",
    "as_of_skip_xgb_p_hat_v1",
    "as_of_skip_xgb_p_long_v1",
    "as_of_skip_xgb_p_short_v1",
    "as_of_skip_xgb_pred_side_v1",
    "as_of_skip_xgb_has_ctx_v1",
]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _resolve_review_dir(reports_root: Path, review_dir_arg: str | None) -> Path:
    if review_dir_arg:
        review_dir = Path(review_dir_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir

    rebuild_summary_path = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if rebuild_summary_path.exists():
        rebuild_summary = _load_json(rebuild_summary_path)
        ledger_dir = rebuild_summary.get("ledger_dir")
        if isinstance(ledger_dir, str) and ledger_dir.strip():
            candidate = Path(ledger_dir).expanduser().resolve()
            if candidate.exists() and all((candidate / name).exists() for name in REQUIRED_REVIEW_ARTIFACTS):
                return candidate

    namespace_dirs = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX)],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if all((candidate / name).exists() for name in REQUIRED_REVIEW_ARTIFACTS):
            return candidate
    raise FileNotFoundError("Could not resolve review dir with required entry RL observability artifacts.")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return payload


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()
    }


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "nat", "<na>", "none", "not_available"}


def _parse_support_map(text: Any) -> Dict[str, str]:
    if _is_missing_text(text):
        return {}
    payload: Dict[str, str] = {}
    for part in str(text).split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            payload[key] = value
    return payload


def _truthy_text(value: Any) -> Any:
    if _is_missing_text(value):
        return pd.NA
    text = str(value).strip().upper()
    if text == "TRUE":
        return True
    if text == "FALSE":
        return False
    return pd.NA


def _float_or_na(value: Any) -> float | None:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    if series.isna().iloc[0]:
        return None
    return float(series.iloc[0])


def _reason_family(reason_path: Any, support_map: Dict[str, str]) -> str:
    support_value = support_map.get("entry_reason_family")
    if not _is_missing_text(support_value):
        return str(support_value)
    if _is_missing_text(reason_path):
        return "NOT_AVAILABLE"
    parts = [segment.strip() for segment in str(reason_path).split(" / ")]
    return parts[2] if len(parts) >= 3 else "NOT_AVAILABLE"


def _reason_code(reason_path: Any, support_map: Dict[str, str]) -> str:
    support_value = support_map.get("entry_reason_code")
    if not _is_missing_text(support_value):
        return str(support_value)
    if _is_missing_text(reason_path):
        return "NOT_AVAILABLE"
    parts = [segment.strip() for segment in str(reason_path).split(" / ")]
    return parts[-1] if parts else "NOT_AVAILABLE"


def _one_hot_action_vector(action: str) -> str:
    payload = {
        "SKIP": 1.0 if action == "SKIP" else 0.0,
        "TAKE_NOW": 1.0 if action == "TAKE_NOW" else 0.0,
        "WAIT": 1.0 if action == "WAIT" else 0.0,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _run_id_list(values: Iterable[Any]) -> List[str]:
    run_ids: List[str] = []
    for value in values:
        if isinstance(value, dict):
            run_id = value.get("run_id")
            if not _is_missing_text(run_id):
                run_ids.append(str(run_id))
        elif not _is_missing_text(value):
            run_ids.append(str(value))
    return run_ids


def _safe_bool_rate(series: pd.Series) -> float | None:
    if series.empty:
        return None
    work = series.astype("boolean")
    if work.notna().sum() == 0:
        return None
    return float(work.fillna(False).mean())


def build_entry_rl_observability_payload(
    *,
    direct_df: pd.DataFrame,
    asof_df: pd.DataFrame,
    skip_raw_df: pd.DataFrame,
    entry_anchor_raw_df: pd.DataFrame,
    supervision_df: pd.DataFrame,
    wait_lifecycle_df: pd.DataFrame,
    actual_take_terminal_df: pd.DataFrame,
    hindsight_export_df: pd.DataFrame,
    skipability_pressure_summary: Dict[str, Any],
    market_opportunity_summary: Dict[str, Any],
    review_dir: Path,
) -> Dict[str, Any]:
    if direct_df.empty:
        raise RuntimeError("ENTRY_RL_OBSERVABILITY_V1 requires a non-empty entry direct composite view.")
    if asof_df.empty:
        raise RuntimeError("ENTRY_RL_OBSERVABILITY_V1 requires a non-empty as_of decision ledger.")

    supervision_direct_df = supervision_df.loc[
        supervision_df["hindsight_policy_action_domain_v1"].astype("string").eq("ENTRY")
        & supervision_df["hindsight_policy_action_projection_kind_v1"].astype("string").eq("DIRECT_ENTRY_DECISION")
    ].copy()
    if supervision_direct_df.empty:
        raise RuntimeError("ENTRY_RL_OBSERVABILITY_V1 requires direct ENTRY supervision rows.")

    work = direct_df.copy()
    work["entry_row_key_v1"] = (
        work["run_id"].astype("string")
        + "|"
        + work["candidate_uid"].astype("string")
        + "|"
        + work["decision_timestamp"].astype("string")
        + "|"
        + work["direct_entry_as_of_row_uid_v1"].astype("string")
    )

    asof_lookup = asof_df[
        [
            "as_of_row_uid_v1",
            "candidate_uid",
            "as_of_candidate_decision_v1",
            "as_of_candidate_decision_reason_v1",
            "as_of_candidate_side_v1",
            "as_of_candidate_session_v1",
            "as_of_candidate_atr_bps_v1",
            "as_of_candidate_entry_spread_bps_v1",
            "as_of_candidate_uncertainty_score_v1",
            "as_of_candidate_tradable_prob_v1",
            "as_of_candidate_mfe_first_n_pred_v1",
            "as_of_candidate_vol_regime_v1",
            "as_of_candidate_trend_regime_v1",
            "as_of_candidate_policy_hash_v1",
            "as_of_candidate_entry_bundle_sha256_v1",
            "as_of_candidate_exit_bundle_sha256_v1",
        ]
    ].drop_duplicates(subset=["as_of_row_uid_v1"])
    work = work.merge(
        asof_lookup.rename(columns={"candidate_uid": "candidate_uid_from_asof_v1"}),
        left_on="direct_entry_as_of_row_uid_v1",
        right_on="as_of_row_uid_v1",
        how="left",
        validate="one_to_one",
    )

    skip_lookup = skip_raw_df[
        [
            "as_of_row_uid_v1",
            "skip_raw_xgb_exact_available_v1",
            "as_of_skip_candidate_margin_v1",
            "as_of_skip_candidate_path_quality_pred_v1",
            "as_of_skip_xgb_has_ctx_v1",
            "as_of_skip_xgb_p_flat_v1",
            "as_of_skip_xgb_p_hat_v1",
            "as_of_skip_xgb_p_long_v1",
            "as_of_skip_xgb_p_short_v1",
            "as_of_skip_xgb_pred_side_v1",
        ]
    ].drop_duplicates(subset=["as_of_row_uid_v1"])
    work = work.merge(
        skip_lookup,
        left_on="direct_entry_as_of_row_uid_v1",
        right_on="as_of_row_uid_v1",
        how="left",
        suffixes=("", "__skip"),
        validate="one_to_one",
    )
    skip_helper_cols = [col for col in work.columns if col.endswith("__skip")]
    if skip_helper_cols:
        work = work.drop(columns=skip_helper_cols)

    entry_anchor_lookup = entry_anchor_raw_df[
        [
            "as_of_row_uid_v1",
            "entry_raw_xgb_multi_horizon_exact_available_v1",
            "as_of_entry_candidate_margin_v1",
            "as_of_entry_candidate_path_quality_pred_v1",
        ]
    ].drop_duplicates(subset=["as_of_row_uid_v1"])
    work = work.merge(
        entry_anchor_lookup,
        left_on="direct_entry_as_of_row_uid_v1",
        right_on="as_of_row_uid_v1",
        how="left",
        suffixes=("", "__entry"),
        validate="one_to_one",
    )
    entry_helper_cols = [col for col in work.columns if col.endswith("__entry")]
    if entry_helper_cols:
        work = work.drop(columns=entry_helper_cols)

    supervision_lookup = supervision_direct_df[
        [
            "candidate_uid",
            "as_of_row_uid_v1",
            "hindsight_policy_action_v1",
            "hindsight_policy_action_reason_path_v1",
            "hindsight_policy_counterfactual_value_bps_v1",
            "hindsight_policy_counterfactual_value_source_v1",
            "hindsight_policy_priority_abs_bps_v1",
            "hindsight_policy_action_support_v1",
            "hindsight_policy_action_semantic_contract_v1",
            "hindsight_supervision_join_contract_v1",
        ]
    ].drop_duplicates(subset=["candidate_uid", "as_of_row_uid_v1"])
    work = work.merge(
        supervision_lookup,
        left_on=["candidate_uid", "direct_entry_as_of_row_uid_v1"],
        right_on=["candidate_uid", "as_of_row_uid_v1"],
        how="left",
        suffixes=("", "__supervision"),
        validate="one_to_one",
    )
    supervision_helper_cols = [col for col in work.columns if col.endswith("__supervision")]
    if supervision_helper_cols:
        work = work.drop(columns=supervision_helper_cols)

    wait_lookup = wait_lifecycle_df[
        [
            "candidate_uid",
            "wait_followthrough_status_v1",
            "wait_lifecycle_rollup_status_v1",
            "wait_lifecycle_terminal_status_v1",
            "wait_lifecycle_terminal_reason_v1",
            "confirmation_delay_minutes_v1",
            "has_provable_confirmation_v1",
            "confirmation_transition_allowed_v1",
            "management_transition_allowed_v1",
            "coverage_status_v1",
        ]
    ].drop_duplicates(subset=["candidate_uid"])
    work = work.merge(
        wait_lookup,
        on="candidate_uid",
        how="left",
        suffixes=("", "__wait"),
        validate="one_to_one",
    )
    if "wait_followthrough_status_v1__wait" in work.columns:
        direct_wait_status = work["wait_followthrough_status_v1"].astype("string")
        joined_wait_status = work["wait_followthrough_status_v1__wait"].astype("string")
        work["wait_followthrough_status_v1"] = direct_wait_status.where(
            ~direct_wait_status.isna(),
            joined_wait_status,
        )
        work = work.drop(columns=["wait_followthrough_status_v1__wait"])

    terminal_lookup = actual_take_terminal_df[
        [
            "candidate_uid",
            "route_status_v1",
            "activation_origin_v1",
            "activation_timestamp_utc_v1",
            "management_handoff_status_v1",
            "management_anchor_type_v1",
            "management_action_label_v1",
            "management_projection_kind_v1",
            "closed_exit_reason_v1",
            "realized_pnl_bps",
            "trade_outcome_class",
            "terminal_outcome_available_v1",
        ]
    ].drop_duplicates(subset=["candidate_uid"]).rename(
        columns={
            "route_status_v1": "terminal_route_status_v1",
            "activation_origin_v1": "terminal_activation_origin_v1",
            "activation_timestamp_utc_v1": "terminal_activation_timestamp_utc_v1",
            "management_handoff_status_v1": "terminal_management_handoff_status_v1",
            "management_anchor_type_v1": "terminal_management_anchor_type_v1",
            "management_action_label_v1": "terminal_management_action_label_v1",
            "management_projection_kind_v1": "terminal_management_projection_kind_v1",
            "closed_exit_reason_v1": "terminal_closed_exit_reason_v1",
        }
    )
    work = work.merge(
        terminal_lookup,
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )

    hindsight_lookup = hindsight_export_df[
        [
            "candidate_uid",
            "post_trade_quality_bucket",
            "post_trade_good_trade_flag_v1",
            "post_trade_good_trade_mfe20_mae5_v1",
            "post_trade_bad_trade_flag_v1",
            "review_entry_bucket_v1",
            "review_exit_bucket_v1",
            "review_good_exit_v1",
            "review_premature_exit_v1",
            "review_late_exit_v1",
            "review_entry_good_but_fragile_v1",
            "review_entry_looked_good_but_failed_v1",
            "hindsight_entry_decision_review_v1",
            "hindsight_should_skip_trade_v1",
            "hindsight_take_was_ok_v1",
            "hindsight_entry_review_unresolved_v1",
            "hindsight_management_review_v1",
            "hindsight_should_hold_longer_v1",
            "hindsight_should_exit_earlier_v1",
            "hindsight_managed_ok_v1",
            "hindsight_hold_longer_extra_value_bps_v1",
            "hindsight_exit_earlier_saved_bps_v1",
            "hindsight_skip_trade_avoided_loss_bps_v1",
            "hindsight_peak_mfe_bps_v1",
            "hindsight_peak_to_exit_giveback_bps_v1",
        ]
    ].drop_duplicates(subset=["candidate_uid"])
    work = work.merge(
        hindsight_lookup,
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )

    work["as_of_timestamp_dt_v1"] = pd.to_datetime(work["as_of_timestamp_utc_v1"], utc=True, errors="coerce")
    work["as_of_hour_utc_v1"] = work["as_of_timestamp_dt_v1"].dt.hour
    work["as_of_weekday_utc_v1"] = pd.to_numeric(work["as_of_weekday_utc_v1"], errors="coerce")
    work["as_of_session_v1"] = work["as_of_candidate_session_v1"].astype("string")
    work["as_of_side_v1"] = work["as_of_candidate_side_v1"].astype("string")
    work["as_of_atr_bps_v1"] = pd.to_numeric(work["as_of_candidate_atr_bps_v1"], errors="coerce")
    work["logged_entry_action_v1"] = work["final_direct_entry_action_v1"].astype("string")
    work["entry_action_truth_status_v1"] = "COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL"
    work["policy_snapshot_status_v1"] = "EXACT_CANDIDATE_POLICY_HASH_AND_MODEL_SNAPSHOT"
    work["policy_snapshot_source_v1"] = "AS_OF_DECISION_MOMENT_LEDGER_V1"
    work["policy_version_v1"] = work["as_of_candidate_policy_hash_v1"].astype("string")
    work["behavior_policy_id_v1"] = work["as_of_candidate_policy_hash_v1"].astype("string")
    work["behavior_policy_id_status_v1"] = "POLICY_SNAPSHOT_ONLY_NOT_LOGGED_DIRECT_POLICY"
    work["behavior_policy_kind_v1"] = "VERSIONED_ENTRY_MODEL_SNAPSHOT_ONLY"
    work["entry_action_propensity_status_v1"] = "PROPENSITY_NOT_ESTABLISHED"
    work["observed_action_propensity_v1"] = pd.NA
    work["per_action_propensity_vector_v1"] = pd.NA
    work["logged_action_matches_teacher_v1"] = work["logged_entry_action_v1"].eq(
        work["hindsight_policy_action_v1"].astype("string")
    )
    work["teacher_action_label_v1"] = work["hindsight_policy_action_v1"].astype("string")
    work["teacher_should_skip_entry_v1"] = work["teacher_action_label_v1"].eq("SKIP")
    work["teacher_should_wait_entry_v1"] = work["teacher_action_label_v1"].eq("WAIT")
    work["teacher_take_now_ok_v1"] = work["teacher_action_label_v1"].eq("TAKE_NOW")

    support_maps = [_parse_support_map(value) for value in work["hindsight_policy_action_support_v1"].tolist()]
    work["entry_reason_family_v1"] = [
        _reason_family(reason_path, support_map)
        for reason_path, support_map in zip(work["hindsight_policy_action_reason_path_v1"], support_maps)
    ]
    work["entry_reason_code_v1"] = [
        _reason_code(reason_path, support_map)
        for reason_path, support_map in zip(work["hindsight_policy_action_reason_path_v1"], support_maps)
    ]
    work["support_entry_bucket_v1"] = [support_map.get("entry_bucket", "NOT_AVAILABLE") for support_map in support_maps]
    work["support_trade_bucket_v1"] = [support_map.get("trade_bucket", "NOT_AVAILABLE") for support_map in support_maps]
    work["support_trade_outcome_class_v1"] = [
        support_map.get("trade_outcome_class", "NOT_AVAILABLE") for support_map in support_maps
    ]
    work["support_adverse_first_v1"] = [_truthy_text(support_map.get("adverse_first")) for support_map in support_maps]
    work["support_confirmation_entry_localizable_v1"] = [
        _truthy_text(support_map.get("confirmation_entry_localizable")) for support_map in support_maps
    ]
    work["support_first_meaningful_mfe_bar_index_v1"] = [
        _float_or_na(support_map.get("first_meaningful_mfe_bar_index")) for support_map in support_maps
    ]
    work["support_mae_bps_v1"] = [_float_or_na(support_map.get("mae_bps")) for support_map in support_maps]
    work["support_peak_mfe_bps_v1"] = [_float_or_na(support_map.get("peak_mfe_bps")) for support_map in support_maps]
    work["support_skip_trade_avoided_loss_bps_v1"] = [
        _float_or_na(support_map.get("skip_trade_avoided_loss_bps")) for support_map in support_maps
    ]
    work["support_confirmation_entry_reason_v1"] = [
        support_map.get("confirmation_entry_reason", "NOT_AVAILABLE") for support_map in support_maps
    ]
    work["support_confirmation_entry_ts_v1"] = [
        support_map.get("confirmation_entry_ts", "NOT_AVAILABLE") for support_map in support_maps
    ]

    terminal_available = pd.Series(work["terminal_outcome_available_v1"], index=work.index).astype("boolean")
    work["actualized_take_v1"] = terminal_available.fillna(False).astype(bool)
    work["entry_reward_semantics_status_v1"] = pd.Series("ENTRY_REWARD_SEMANTICS_UNRESOLVED", index=work.index, dtype="string")
    work.loc[
        work["logged_entry_action_v1"].eq("TAKE_NOW") & work["actualized_take_v1"],
        "entry_reward_semantics_status_v1",
    ] = "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE"
    work.loc[
        work["logged_entry_action_v1"].eq("WAIT")
        & work["wait_followthrough_status_v1"].astype("string").eq("WAIT_WITH_PROVABLE_CONFIRMATION")
        & work["actualized_take_v1"],
        "entry_reward_semantics_status_v1",
    ] = "WAIT_THEN_CONFIRMATION_REALIZED_OUTCOME_AVAILABLE"
    work.loc[
        work["logged_entry_action_v1"].eq("WAIT")
        & work["wait_followthrough_status_v1"].astype("string").eq("WAIT_WITHOUT_LOCALIZABLE_CONFIRMATION"),
        "entry_reward_semantics_status_v1",
    ] = "WAIT_WITHOUT_LOCALIZABLE_CONFIRMATION_HINDSIGHT_ONLY"
    work.loc[
        work["logged_entry_action_v1"].eq("WAIT")
        & work["wait_followthrough_status_v1"].astype("string").eq("WAIT_CONFIRMATION_UNJOINABLE"),
        "entry_reward_semantics_status_v1",
    ] = "WAIT_CONFIRMATION_UNJOINABLE_HINDSIGHT_ONLY"
    work.loc[
        work["logged_entry_action_v1"].eq("SKIP")
        & pd.to_numeric(work["support_skip_trade_avoided_loss_bps_v1"], errors="coerce").notna(),
        "entry_reward_semantics_status_v1",
    ] = "SKIP_AVOIDED_LOSS_HINDSIGHT_ONLY"
    work.loc[
        work["logged_entry_action_v1"].eq("SKIP")
        & pd.to_numeric(work["support_skip_trade_avoided_loss_bps_v1"], errors="coerce").isna(),
        "entry_reward_semantics_status_v1",
    ] = "SKIP_HINDSIGHT_ONLY_NO_SCALAR"

    work["entry_review_focus_v1"] = pd.Series("ENTRY_OK_OR_MIXED", index=work.index, dtype="string")
    work.loc[work["entry_reason_family_v1"].eq("BAD_TRADE_QUALITY"), "entry_review_focus_v1"] = "SHOULD_HAVE_SKIPPED"
    work.loc[work["entry_reason_code_v1"].eq("WAIT_TO_REDUCE_MAE"), "entry_review_focus_v1"] = "SHOULD_HAVE_WAITED"
    work.loc[
        work["support_trade_bucket_v1"].isin(["bad_trade", "cata_trade", "overheld_trade"]),
        "entry_review_focus_v1",
    ] = "TOOK_BAD_OR_FRAGILE_TRADE"

    view_columns = [
        "entry_row_key_v1",
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "direct_entry_as_of_row_uid_v1",
        "split_bucket_v1",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "logged_entry_action_v1",
        "entry_action_truth_status_v1",
        "policy_stack_stage_v1",
        "direct_composite_status_v1",
        "policy_snapshot_status_v1",
        "policy_snapshot_source_v1",
        "policy_version_v1",
        "behavior_policy_id_v1",
        "behavior_policy_id_status_v1",
        "behavior_policy_kind_v1",
        "as_of_candidate_entry_bundle_sha256_v1",
        "as_of_candidate_exit_bundle_sha256_v1",
        "entry_action_propensity_status_v1",
        "observed_action_propensity_v1",
        "per_action_propensity_vector_v1",
        "teacher_action_label_v1",
        "logged_action_matches_teacher_v1",
        "teacher_should_skip_entry_v1",
        "teacher_should_wait_entry_v1",
        "teacher_take_now_ok_v1",
        "hindsight_policy_action_reason_path_v1",
        "entry_reason_family_v1",
        "entry_reason_code_v1",
        "hindsight_policy_counterfactual_value_bps_v1",
        "hindsight_policy_counterfactual_value_source_v1",
        "hindsight_policy_priority_abs_bps_v1",
        "support_entry_bucket_v1",
        "support_trade_bucket_v1",
        "support_trade_outcome_class_v1",
        "support_adverse_first_v1",
        "support_first_meaningful_mfe_bar_index_v1",
        "support_mae_bps_v1",
        "support_peak_mfe_bps_v1",
        "support_skip_trade_avoided_loss_bps_v1",
        "support_confirmation_entry_localizable_v1",
        "support_confirmation_entry_reason_v1",
        "support_confirmation_entry_ts_v1",
        "wait_followthrough_status_v1",
        "wait_lifecycle_rollup_status_v1",
        "wait_lifecycle_terminal_status_v1",
        "wait_lifecycle_terminal_reason_v1",
        "confirmation_delay_minutes_v1",
        "has_provable_confirmation_v1",
        "confirmation_transition_allowed_v1",
        "management_transition_allowed_v1",
        "coverage_status_v1",
        "actualized_take_v1",
        "entry_reward_semantics_status_v1",
        "terminal_route_status_v1",
        "terminal_activation_origin_v1",
        "terminal_activation_timestamp_utc_v1",
        "terminal_management_handoff_status_v1",
        "terminal_management_anchor_type_v1",
        "terminal_management_action_label_v1",
        "terminal_management_projection_kind_v1",
        "terminal_closed_exit_reason_v1",
        "realized_pnl_bps",
        "trade_outcome_class",
        "post_trade_quality_bucket",
        "post_trade_good_trade_flag_v1",
        "post_trade_good_trade_mfe20_mae5_v1",
        "post_trade_bad_trade_flag_v1",
        "review_entry_bucket_v1",
        "review_exit_bucket_v1",
        "review_good_exit_v1",
        "review_premature_exit_v1",
        "review_late_exit_v1",
        "review_entry_good_but_fragile_v1",
        "review_entry_looked_good_but_failed_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_should_skip_trade_v1",
        "hindsight_take_was_ok_v1",
        "hindsight_should_hold_longer_v1",
        "hindsight_should_exit_earlier_v1",
        "hindsight_hold_longer_extra_value_bps_v1",
        "hindsight_exit_earlier_saved_bps_v1",
        "hindsight_skip_trade_avoided_loss_bps_v1",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
        "entry_review_focus_v1",
    ] + OBSERVATION_FIELDS_V1
    entry_rl_observability_view_v1_df = work[view_columns].copy()

    policy_hash_available_rows_v1 = int(work["policy_version_v1"].astype("string").notna().sum())
    supervision_available_rows_v1 = int(work["teacher_action_label_v1"].astype("string").notna().sum())
    snapshot_ready = policy_hash_available_rows_v1 == int(len(work))
    direct_actions_match_supervision = int(work["logged_action_matches_teacher_v1"].fillna(False).sum()) == int(len(work))

    consistency_rows = [
        {
            "check_name_v1": "ENTRY_OBSERVABILITY_VIEW_COVERS_DIRECT_COMPOSITE_EXACTLY",
            "status_v1": "PASS"
            if int(len(entry_rl_observability_view_v1_df)) == int(len(direct_df))
            and int(entry_rl_observability_view_v1_df.duplicated(subset=["entry_row_key_v1"]).sum()) == 0
            else "FAIL",
            "observed_value_v1": int(len(entry_rl_observability_view_v1_df)),
            "expected_value_v1": int(len(direct_df)),
            "note_v1": "Every direct entry composition row must appear exactly once in the RL observability view.",
        },
        {
            "check_name_v1": "EXACT_POLICY_HASH_SNAPSHOT_AVAILABLE_FOR_ALL_DIRECT_ROWS",
            "status_v1": "PASS" if snapshot_ready else "FAIL",
            "observed_value_v1": policy_hash_available_rows_v1,
            "expected_value_v1": int(len(work)),
            "note_v1": "Entry observability requires exact candidate policy hash snapshots for the whole direct-entry universe.",
        },
        {
            "check_name_v1": "EXACT_ENTRY_MODEL_SNAPSHOT_FIELDS_AVAILABLE_FOR_ALL_DIRECT_ROWS",
            "status_v1": "PASS"
            if int(work["entry_raw_xgb_multi_horizon_exact_available_v1"].fillna(False).sum()) == int(len(work))
            and int(work["skip_raw_xgb_exact_available_v1"].fillna(False).sum()) == int(len(work))
            else "FAIL",
            "observed_value_v1": json.dumps(
                {
                    "entry_raw_xgb_exact_rows_v1": int(work["entry_raw_xgb_multi_horizon_exact_available_v1"].fillna(False).sum()),
                    "skip_raw_xgb_exact_rows_v1": int(work["skip_raw_xgb_exact_available_v1"].fillna(False).sum()),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                {
                    "entry_raw_xgb_exact_rows_v1": int(len(work)),
                    "skip_raw_xgb_exact_rows_v1": int(len(work)),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "Entry RL observability must see the exact entry model snapshot fields on every direct row.",
        },
        {
            "check_name_v1": "SUPERVISION_JOIN_COVERS_WHOLE_DIRECT_ENTRY_UNIVERSE",
            "status_v1": "PASS" if supervision_available_rows_v1 == int(len(work)) else "FAIL",
            "observed_value_v1": supervision_available_rows_v1,
            "expected_value_v1": int(len(work)),
            "note_v1": "Every direct entry row must have a matching direct-entry supervision row.",
        },
        {
            "check_name_v1": "DIRECT_ACTIONS_STAY_ALIGNED_WITH_EXISTING_SUPERVISION_COMPOSITION",
            "status_v1": "PASS" if direct_actions_match_supervision else "FAIL",
            "observed_value_v1": int(work["logged_action_matches_teacher_v1"].fillna(False).sum()),
            "expected_value_v1": int(len(work)),
            "note_v1": "Current entry observability remains a compositional read-model and must stay aligned with the frozen supervision stack.",
        },
        {
            "check_name_v1": "WAIT_FOLLOWTHROUGH_COUNTS_STAY_FROZEN",
            "status_v1": "PASS"
            if _counts(work.loc[work["logged_entry_action_v1"].eq("WAIT")], "wait_followthrough_status_v1")
            == _counts(wait_lifecycle_df, "wait_followthrough_status_v1")
            else "FAIL",
            "observed_value_v1": json.dumps(
                _counts(work.loc[work["logged_entry_action_v1"].eq("WAIT")], "wait_followthrough_status_v1"),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                _counts(wait_lifecycle_df, "wait_followthrough_status_v1"),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "Wait/confirmation routing must match the frozen lifecycle view exactly.",
        },
        {
            "check_name_v1": "TAKE_NOW_AND_WAIT_WITH_CONFIRMATION_TERMINAL_COUNT_STAYS_FROZEN",
            "status_v1": "PASS"
            if int(work["actualized_take_v1"].sum()) == int(len(actual_take_terminal_df))
            else "FAIL",
            "observed_value_v1": int(work["actualized_take_v1"].sum()),
            "expected_value_v1": int(len(actual_take_terminal_df)),
            "note_v1": "Actualized direct takes plus wait-then-confirmation takes must preserve the exact terminal-outcome count.",
        },
        {
            "check_name_v1": "ENTRY_PROPENSITY_REMAINS_EXPLICITLY_UNESTABLISHED",
            "status_v1": "PASS"
            if int(work["entry_action_propensity_status_v1"].astype("string").eq("PROPENSITY_NOT_ESTABLISHED").sum())
            == int(len(work))
            else "FAIL",
            "observed_value_v1": int(work["entry_action_propensity_status_v1"].astype("string").eq("PROPENSITY_NOT_ESTABLISHED").sum()),
            "expected_value_v1": int(len(work)),
            "note_v1": "This layer must not fabricate logged entry propensities before exact live-gate action logging exists.",
        },
    ]
    entry_rl_observability_consistency_audit_v1_df = pd.DataFrame.from_records(consistency_rows)
    failed_checks = int((entry_rl_observability_consistency_audit_v1_df["status_v1"].astype("string") != "PASS").sum())

    policy_snapshot_contract_v1 = {
        "layer_name": "ENTRY_POLICY_SNAPSHOT_CONTRACT_V1",
        "mode_v1": "EXACT_ENTRY_MODEL_SNAPSHOT_ONLY",
        "policy_identity_fields_v1": [
            "behavior_policy_id_v1",
            "as_of_candidate_entry_bundle_sha256_v1",
            "as_of_candidate_exit_bundle_sha256_v1",
        ],
        "model_snapshot_fields_v1": OBSERVATION_FIELDS_V1,
        "snapshot_source_v1": "AS_OF_DECISION_MOMENT_LEDGER_V1 + ENTRY_RAW_STATE_PARITY_VIEWS",
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    entry_rl_observability_contract_v1 = {
        "layer_name": "ENTRY_RL_OBSERVABILITY_CONTRACT_V1",
        "mode_v1": "ENTRY_COMPOSITION_AND_HINDSIGHT_OBSERVABILITY_ONLY",
        "domain_scope_v1": "ENTRY_ONLY",
        "action_space_v1": ["SKIP", "TAKE_NOW", "WAIT"],
        "observation_feature_names_v1": OBSERVATION_FIELDS_V1,
        "entry_action_truth_status_levels_v1": ["COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL"],
        "reward_semantics_status_levels_v1": [
            "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE",
            "WAIT_THEN_CONFIRMATION_REALIZED_OUTCOME_AVAILABLE",
            "WAIT_WITHOUT_LOCALIZABLE_CONFIRMATION_HINDSIGHT_ONLY",
            "WAIT_CONFIRMATION_UNJOINABLE_HINDSIGHT_ONLY",
            "SKIP_AVOIDED_LOSS_HINDSIGHT_ONLY",
            "SKIP_HINDSIGHT_ONLY_NO_SCALAR",
            "ENTRY_REWARD_SEMANTICS_UNRESOLVED",
        ],
        "not_live_gate": True,
        "not_policy_truth": True,
        "propensity_not_established_v1": True,
        "contract_note_v1": (
            "This layer gives RL and diagnostics one place to read the frozen entry composition, exact model snapshot, wait/confirmation routing, and hindsight trade-quality channels. "
            "It does not claim exact logged live entry-policy truth or propensity truth."
        ),
    }

    skipability_zero = skipability_pressure_summary.get("completed_zero_trade_runs")
    candidate_rich_zero = skipability_pressure_summary.get("candidate_rich_zero_trade_runs")
    opportunity_rich_zero = market_opportunity_summary.get("opportunity_rich_zero_trade_runs_anchor", [])
    opportunity_rich_zero_run_ids = _run_id_list(opportunity_rich_zero)

    entry_rl_observability_summary_v1 = {
        "layer_name": "ENTRY_RL_OBSERVABILITY_SUMMARY_V1",
        "review_dir_v1": str(review_dir),
        "observed_direct_entry_rows_v1": int(len(work)),
        "logged_action_counts_v1": _counts(work, "logged_entry_action_v1"),
        "teacher_action_counts_v1": _counts(work, "teacher_action_label_v1"),
        "reason_family_counts_v1": _counts(work, "entry_reason_family_v1"),
        "reason_code_counts_v1": _counts(work, "entry_reason_code_v1"),
        "entry_review_focus_counts_v1": _counts(work, "entry_review_focus_v1"),
        "policy_hash_available_rows_v1": policy_hash_available_rows_v1,
        "policy_hash_not_available_rows_v1": int(len(work) - policy_hash_available_rows_v1),
        "snapshot_action_match_rows_v1": int(work["logged_action_matches_teacher_v1"].fillna(False).sum()),
        "actualized_take_count_v1": int(work["actualized_take_v1"].sum()),
        "terminal_outcome_available_rows_v1": int(work["actualized_take_v1"].sum()),
        "wait_followthrough_counts_v1": _counts(work.loc[work["logged_entry_action_v1"].eq("WAIT")], "wait_followthrough_status_v1"),
        "reward_semantics_counts_v1": _counts(work, "entry_reward_semantics_status_v1"),
        "direct_take_bad_trade_rate_v1": _safe_bool_rate(
            work.loc[work["logged_entry_action_v1"].eq("TAKE_NOW"), "post_trade_bad_trade_flag_v1"]
        ),
        "direct_take_hold_longer_rate_v1": _safe_bool_rate(
            work.loc[work["logged_entry_action_v1"].eq("TAKE_NOW"), "hindsight_should_hold_longer_v1"]
        ),
        "skip_hindsight_avoided_loss_rows_v1": int(
            pd.to_numeric(work.loc[work["logged_entry_action_v1"].eq("SKIP"), "support_skip_trade_avoided_loss_bps_v1"], errors="coerce").notna().sum()
        ),
        "completed_zero_trade_runs_v1": int(skipability_zero) if skipability_zero is not None else None,
        "candidate_rich_zero_trade_runs_v1": int(candidate_rich_zero) if candidate_rich_zero is not None else None,
        "opportunity_rich_zero_trade_run_count_v1": int(len(opportunity_rich_zero_run_ids)),
        "opportunity_rich_zero_trade_run_ids_v1": opportunity_rich_zero_run_ids,
        "failed_check_count_v1": failed_checks,
    }
    entry_rl_observability_status_v1 = {
        "layer_name": "ENTRY_RL_OBSERVABILITY_STATUS_V1",
        "ENTRY_POLICY_SNAPSHOT_STATUS": (
            "READY_EXACT_CANDIDATE_POLICY_HASH_AND_MODEL_SNAPSHOT"
            if snapshot_ready
            else "POLICY_SNAPSHOT_COVERAGE_NOT_FULLY_ESTABLISHED"
        ),
        "ENTRY_DIRECT_ACTION_STATUS": "HIERARCHICAL_COMPOSITION_READY_NOT_POLICY_TRUTH",
        "ENTRY_PROPENSITY_STATUS": "NOT_ESTABLISHED",
        "ENTRY_RL_OBSERVABILITY_STATUS": (
            "ENTRY_OBSERVABILITY_READY_HINDSIGHT_AND_COMPOSITION_ONLY"
            if failed_checks == 0
            else "ISSUES_FOUND"
        ),
        "ENTRY_WAIT_STATUS": "FOLLOWTHROUGH_COVERAGE_LIMITED_BUT_HONEST",
        "ENTRY_SKIPABILITY_STATUS": "ENTRY_SKIPABILITY_BRANCH_BUILT_BUT_RARE_CLASS",
        "ENTRY_ZERO_TRADE_PRESSURE_STATUS": (
            "OPPORTUNITY_RICH_ZERO_WEEKS_PRESENT"
            if len(opportunity_rich_zero_run_ids) > 0
            else "NO_OPPORTUNITY_RICH_ZERO_WEEKS_DETECTED"
        ),
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest_v1 = {
        "layer_name": "ENTRY_RL_OBSERVABILITY_MANIFEST_V1",
        "mode_v1": "APPEND_ONLY_EXTENSION",
        "review_dir_v1": str(review_dir),
        "artifacts_v1": {
            "entry_policy_snapshot_contract_v1": ENTRY_POLICY_SNAPSHOT_CONTRACT,
            "entry_rl_observability_contract_v1": ENTRY_RL_OBSERVABILITY_CONTRACT,
            "entry_rl_observability_view_v1": ENTRY_RL_OBSERVABILITY_VIEW,
            "entry_rl_observability_summary_v1": ENTRY_RL_OBSERVABILITY_SUMMARY,
            "entry_rl_observability_status_v1": ENTRY_RL_OBSERVABILITY_STATUS,
            "entry_rl_observability_consistency_audit_v1": ENTRY_RL_OBSERVABILITY_AUDIT,
        },
    }
    return {
        "entry_policy_snapshot_contract_v1": policy_snapshot_contract_v1,
        "entry_rl_observability_contract_v1": entry_rl_observability_contract_v1,
        "entry_rl_observability_view_v1_df": entry_rl_observability_view_v1_df,
        "entry_rl_observability_consistency_audit_v1_df": entry_rl_observability_consistency_audit_v1_df,
        "entry_rl_observability_summary_v1": entry_rl_observability_summary_v1,
        "entry_rl_observability_status_v1": entry_rl_observability_status_v1,
        "entry_rl_observability_manifest_v1": manifest_v1,
    }


def materialize_truth_entry_rl_observability(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    extension_dir: Path | None = None,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_review_dir = _resolve_review_dir(reports_root, str(review_dir) if review_dir else None)

    direct_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_entry_direct_policy_composite_v1.parquet")
    asof_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet")
    skip_raw_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet")
    entry_anchor_raw_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_entry_anchor_raw_state_v1.parquet")
    supervision_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_policy_action_supervision_join_v1.parquet")
    wait_lifecycle_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_entry_wait_lifecycle_view_v1.parquet")
    actual_take_terminal_df = pd.read_parquet(
        resolved_review_dir / "shadow_meta_all_trade_review_entry_actual_take_terminal_outcome_view_v1.parquet"
    )
    hindsight_export_df = pd.read_parquet(
        resolved_review_dir / "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet"
    )
    skipability_pressure_summary = _load_json(reports_root / "truth_entry_skipability_pressure_v1.json")
    market_opportunity_summary = _load_json(reports_root / "truth_continuous_market_opportunity_v1.json")

    payload = build_entry_rl_observability_payload(
        direct_df=direct_df,
        asof_df=asof_df,
        skip_raw_df=skip_raw_df,
        entry_anchor_raw_df=entry_anchor_raw_df,
        supervision_df=supervision_df,
        wait_lifecycle_df=wait_lifecycle_df,
        actual_take_terminal_df=actual_take_terminal_df,
        hindsight_export_df=hindsight_export_df,
        skipability_pressure_summary=skipability_pressure_summary,
        market_opportunity_summary=market_opportunity_summary,
        review_dir=resolved_review_dir,
    )

    if extension_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extension_dir = reports_root / f"{LEDGER_NAMESPACE_PREFIX}{stamp}_{ENTRY_RL_EXTENSION_SUFFIX}"
    extension_dir = Path(extension_dir).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=False)

    payload["entry_rl_observability_view_v1_df"].to_parquet(extension_dir / ENTRY_RL_OBSERVABILITY_VIEW, index=False)
    payload["entry_rl_observability_consistency_audit_v1_df"].to_csv(
        extension_dir / ENTRY_RL_OBSERVABILITY_AUDIT,
        index=False,
    )
    (extension_dir / ENTRY_POLICY_SNAPSHOT_CONTRACT).write_text(
        json.dumps(payload["entry_policy_snapshot_contract_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / ENTRY_RL_OBSERVABILITY_CONTRACT).write_text(
        json.dumps(payload["entry_rl_observability_contract_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / ENTRY_RL_OBSERVABILITY_SUMMARY).write_text(
        json.dumps(payload["entry_rl_observability_summary_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / ENTRY_RL_OBSERVABILITY_STATUS).write_text(
        json.dumps(payload["entry_rl_observability_status_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / ENTRY_RL_OBSERVABILITY_MANIFEST).write_text(
        json.dumps(payload["entry_rl_observability_manifest_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    top_level_summary = dict(payload["entry_rl_observability_summary_v1"])
    top_level_summary["extension_dir_v1"] = str(extension_dir)
    top_level_summary["review_dir_v1"] = str(resolved_review_dir)
    top_level_summary["status_v1"] = payload["entry_rl_observability_status_v1"]
    (reports_root / TOP_LEVEL_SUMMARY).write_text(
        json.dumps(top_level_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "extension_dir": extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["entry_rl_observability_summary_v1"],
        "status": payload["entry_rl_observability_status_v1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize entry RL observability extension from the active truth line.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--review-dir", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    args = parser.parse_args()

    result = materialize_truth_entry_rl_observability(
        _resolve_reports_root(args.reports_root),
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
    )
    print(json.dumps(
        {
            "extension_dir": str(result["extension_dir"]),
            "top_level_summary_path": str(result["top_level_summary_path"]),
            "status": result["status"],
            "summary": result["summary"],
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
