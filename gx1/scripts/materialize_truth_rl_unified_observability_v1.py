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
UNIFIED_RL_EXTENSION_SUFFIX = "RL_UNIFIED_OBSERVABILITY_V1"

UNIFIED_RL_CONTRACT = "shadow_meta_all_trade_review_rl_unified_observability_contract_v1.json"
UNIFIED_RL_EPISODE_VIEW = "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet"
UNIFIED_RL_DECISION_EVENT_VIEW = "shadow_meta_all_trade_review_rl_unified_decision_event_view_v1.parquet"
UNIFIED_RL_SUMMARY = "shadow_meta_all_trade_review_rl_unified_observability_summary_v1.json"
UNIFIED_RL_STATUS = "shadow_meta_all_trade_review_rl_unified_observability_status_v1.json"
UNIFIED_RL_AUDIT = "shadow_meta_all_trade_review_rl_unified_observability_consistency_audit_v1.csv"
UNIFIED_RL_MANIFEST = "shadow_meta_all_trade_review_rl_unified_observability_manifest_v1.json"
TOP_LEVEL_SUMMARY = "truth_rl_unified_observability_v1.json"

REQUIRED_REVIEW_ARTIFACTS = [
    "shadow_meta_all_trade_review_ledger_closed_trades.parquet",
    "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet",
    "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json",
    "shadow_meta_all_trade_review_entry_rl_observability_status_v1.json",
    "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json",
    "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet",
    "shadow_meta_all_trade_review_management_rl_transition_eligible_view_v1.parquet",
    "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet",
    "shadow_meta_all_trade_review_management_exit_local_all_eligible_scored_view_v1.parquet",
    "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json",
    "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
    "shadow_meta_all_trade_review_management_bandit_status_v1.json",
    "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json",
]

REQUIRED_MANAGEMENT_POLICY_ARTIFACTS = [
    "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet",
    "shadow_meta_all_trade_review_management_policy_logging_summary_v1.json",
]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return payload


def _resolve_review_dir(reports_root: Path, review_dir_arg: str | None) -> Path:
    if review_dir_arg:
        review_dir = Path(review_dir_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir

    rebuild_summary_path = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if rebuild_summary_path.exists():
        ledger_dir = _load_json(rebuild_summary_path).get("ledger_dir")
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
    raise FileNotFoundError("Could not resolve review dir with required unified RL artifacts.")


def _resolve_management_policy_dir(reports_root: Path, policy_dir_arg: str | None) -> Path:
    if policy_dir_arg:
        policy_dir = Path(policy_dir_arg).expanduser().resolve()
        if not policy_dir.exists():
            raise FileNotFoundError(f"Management policy dir does not exist: {policy_dir}")
        if not all((policy_dir / name).exists() for name in REQUIRED_MANAGEMENT_POLICY_ARTIFACTS):
            raise FileNotFoundError(f"Management policy dir is missing required policy artifacts: {policy_dir}")
        return policy_dir

    namespace_dirs = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and path.name.startswith(LEDGER_NAMESPACE_PREFIX)
            and path.name.endswith("MANAGEMENT_AUDIT_EXTENSION_V1")
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if all((candidate / name).exists() for name in REQUIRED_MANAGEMENT_POLICY_ARTIFACTS):
            return candidate
    raise FileNotFoundError("Could not resolve management policy logging dir with exact policy artifacts.")


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()
    }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "nat", "<na>", "none", "not_available"}


def _json_counts(values: Iterable[Any]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        key = "NA" if _is_missing(value) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return json.dumps(counts, ensure_ascii=True, sort_keys=True)


def _json_list(values: Iterable[Any]) -> str:
    clean = [str(value) for value in values if not _is_missing(value)]
    return json.dumps(clean, ensure_ascii=True, sort_keys=True)


def _first_non_missing(values: Iterable[Any]) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return pd.NA


def _last_non_missing(values: Iterable[Any]) -> Any:
    last: Any = pd.NA
    for value in values:
        if not _is_missing(value):
            last = value
    return last


def _numeric_or_none(series: pd.Series, op: str) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    if op == "max":
        return float(numeric.max())
    if op == "last":
        return float(numeric.iloc[-1])
    if op == "mean":
        return float(numeric.mean())
    raise ValueError(op)


def _selected_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns].copy()


def _derive_split_bucket(frame: pd.DataFrame) -> pd.Series:
    if "split_bucket_v1" in frame.columns:
        existing = frame["split_bucket_v1"].astype("string")
    elif "as_of_split_bucket_v1" in frame.columns:
        existing = frame["as_of_split_bucket_v1"].astype("string")
    else:
        existing = pd.Series(pd.NA, index=frame.index, dtype="string")
    used_for_training = frame["used_for_training"].fillna(False).astype(bool) if "used_for_training" in frame.columns else pd.Series(False, index=frame.index)
    used_for_validation = frame["used_for_validation"].fillna(False).astype(bool) if "used_for_validation" in frame.columns else pd.Series(False, index=frame.index)
    used_for_holdout = frame["used_for_holdout"].fillna(False).astype(bool) if "used_for_holdout" in frame.columns else pd.Series(False, index=frame.index)
    derived = pd.Series(pd.NA, index=frame.index, dtype="string")
    derived.loc[used_for_training] = "TRAIN"
    derived.loc[used_for_validation] = "VALIDATION"
    derived.loc[used_for_holdout] = "HOLDOUT"
    return existing.where(~existing.isna(), derived)


def _candidate_snapshot_lookup(candidate_snapshot_df: pd.DataFrame | None) -> pd.DataFrame:
    if candidate_snapshot_df is None or candidate_snapshot_df.empty:
        return pd.DataFrame()
    snapshot_cols = [
        "candidate_uid",
        "hour_utc",
        "weekday_utc",
        "session",
        "side",
        "atr_bps",
        "entry_spread_bps",
        "uncertainty_score",
        "tradable_prob",
        "mfe_first_n_pred",
        "trend_regime",
        "vol_regime",
        "margin",
        "path_quality_pred",
        "p_flat",
        "p_hat",
        "p_long",
        "p_short",
        "decision",
        "policy_hash",
    ]
    return _selected_columns(candidate_snapshot_df, snapshot_cols).drop_duplicates(subset=["candidate_uid"])


def _populate_entry_obs_from_candidate_snapshot(events: pd.DataFrame, candidate_snapshot_df: pd.DataFrame | None) -> pd.DataFrame:
    snapshot = _candidate_snapshot_lookup(candidate_snapshot_df)
    if snapshot.empty or events.empty:
        return events
    out = events.merge(snapshot, left_on="candidate_uid_v1", right_on="candidate_uid", how="left", validate="many_to_one")
    mapping = {
        "entry_obs__as_of_hour_utc_v1": "hour_utc",
        "entry_obs__as_of_weekday_utc_v1": "weekday_utc",
        "entry_obs__as_of_session_v1": "session",
        "entry_obs__as_of_side_v1": "side",
        "entry_obs__as_of_atr_bps_v1": "atr_bps",
        "entry_obs__as_of_candidate_entry_spread_bps_v1": "entry_spread_bps",
        "entry_obs__as_of_candidate_uncertainty_score_v1": "uncertainty_score",
        "entry_obs__as_of_candidate_tradable_prob_v1": "tradable_prob",
        "entry_obs__as_of_candidate_mfe_first_n_pred_v1": "mfe_first_n_pred",
        "entry_obs__as_of_candidate_trend_regime_v1": "trend_regime",
        "entry_obs__as_of_candidate_vol_regime_v1": "vol_regime",
        "entry_obs__as_of_entry_candidate_margin_v1": "margin",
        "entry_obs__as_of_entry_candidate_path_quality_pred_v1": "path_quality_pred",
        "entry_obs__as_of_skip_candidate_margin_v1": "margin",
        "entry_obs__as_of_skip_candidate_path_quality_pred_v1": "path_quality_pred",
        "entry_obs__as_of_skip_xgb_p_flat_v1": "p_flat",
        "entry_obs__as_of_skip_xgb_p_hat_v1": "p_hat",
        "entry_obs__as_of_skip_xgb_p_long_v1": "p_long",
        "entry_obs__as_of_skip_xgb_p_short_v1": "p_short",
        "entry_obs__as_of_skip_xgb_pred_side_v1": "decision",
    }
    for target, source in mapping.items():
        if target in out.columns and source in out.columns:
            out[target] = out[target].where(~out[target].isna(), out[source])
    if "entry_obs__as_of_skip_xgb_has_ctx_v1" in out.columns:
        has_snapshot = out["candidate_uid"].astype("string").notna()
        out["entry_obs__as_of_skip_xgb_has_ctx_v1"] = out["entry_obs__as_of_skip_xgb_has_ctx_v1"].where(
            ~out["entry_obs__as_of_skip_xgb_has_ctx_v1"].isna(),
            has_snapshot.astype("Int64"),
        )
    return out.drop(columns=[column for column in snapshot.columns if column in out.columns], errors="ignore")


def _load_candidate_snapshots(reports_root: Path, ledger_df: pd.DataFrame) -> pd.DataFrame:
    runs_root = reports_root / "runs"
    if not runs_root.exists() or ledger_df.empty or "run_id" not in ledger_df.columns:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for run_id in sorted(ledger_df["run_id"].astype("string").dropna().unique().tolist()):
        path = runs_root / run_id / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_unified_rl_observability_payload(
    *,
    entry_df: pd.DataFrame,
    management_row_df: pd.DataFrame,
    management_transition_df: pd.DataFrame,
    management_bandit_df: pd.DataFrame,
    management_exit_local_scored_df: pd.DataFrame,
    management_policy_log_df: pd.DataFrame,
    closed_trade_ledger_df: pd.DataFrame | None = None,
    candidate_snapshot_df: pd.DataFrame | None = None,
    entry_contract: Dict[str, Any],
    management_observation_contract: Dict[str, Any],
    entry_status: Dict[str, Any],
    management_readiness_status: Dict[str, Any],
    management_bandit_status: Dict[str, Any],
    management_sequence_status: Dict[str, Any],
    management_policy_summary: Dict[str, Any],
    entry_handoff_summary: Dict[str, Any],
    review_dir: Path,
    management_policy_dir: Path,
) -> Dict[str, Any]:
    if entry_df.empty:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires non-empty entry observability view.")
    if management_row_df.empty:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires non-empty management row semantics view.")
    if management_transition_df.empty:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires non-empty management transition view.")
    if management_policy_log_df.empty:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires non-empty management policy log harness.")

    entry_obs_fields = list(entry_contract.get("observation_feature_names_v1") or [])
    management_obs_fields = list(management_observation_contract.get("observation_vector_feature_names_v1") or [])
    if not entry_obs_fields:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires entry observation fields from contract.")
    if not management_obs_fields:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 requires management observation fields from contract.")

    entry = entry_df.copy()
    entry["candidate_uid"] = entry["candidate_uid"].astype("string")
    entry["unified_episode_key_v1"] = entry["run_id"].astype("string") + "|" + entry["candidate_uid"]
    closed_trade_ledger = closed_trade_ledger_df.copy() if closed_trade_ledger_df is not None else pd.DataFrame()
    if not closed_trade_ledger.empty:
        for required_column in ["run_id", "candidate_uid", "trade_uid", "trade_id"]:
            if required_column not in closed_trade_ledger.columns:
                raise KeyError(f"Closed trade ledger missing required column for unified RL coverage: {required_column}")
        closed_trade_ledger["candidate_uid"] = closed_trade_ledger["candidate_uid"].astype("string")

    scored_cols = [
        "management_row_key_v1",
        "primary_model_name_v1",
        "primary_model_target_v1",
        "primary_model_score_v1",
        "primary_model_score_rank_within_split_v1",
        "train_action_domain_status_v1",
        "scoring_scope_note_v1",
    ]
    scored = _selected_columns(management_exit_local_scored_df, scored_cols).drop_duplicates(
        subset=["management_row_key_v1"]
    )
    policy_cols = [
        "management_row_key_v1",
        "observed_action_v1",
        "policy_version_v1",
        "policy_version_status_v1",
        "behavior_policy_id_v1",
        "behavior_policy_id_status_v1",
        "behavior_policy_kind_v1",
        "observed_action_status_v1",
        "observed_action_source_v1",
        "observed_action_propensity_status_v1",
        "policy_logging_propensity_status_v1",
        "observed_action_propensity_v1",
        "per_action_propensity_vector_v1",
        "propensity_hold_v1",
        "propensity_exit_now_v1",
    ]
    policy_log = _selected_columns(management_policy_log_df, policy_cols).drop_duplicates(
        subset=["management_row_key_v1"]
    )

    management = management_row_df.copy()
    management["candidate_uid_exact_v1"] = management["candidate_uid_exact_v1"].astype("string")
    management = management.merge(scored, on="management_row_key_v1", how="left", validate="one_to_one")
    management = management.merge(policy_log, on="management_row_key_v1", how="left", validate="one_to_one")
    management["unified_episode_key_v1"] = management["run_id"].astype("string") + "|" + management["candidate_uid_exact_v1"]

    transition_keys = set(management_transition_df["management_row_key_v1"].astype("string").tolist())
    management_policy_keys = set(policy_log["management_row_key_v1"].astype("string").tolist())
    policy_covers_transition = transition_keys.issubset(management_policy_keys)

    agg_rows: List[Dict[str, Any]] = []
    for candidate_uid, group in management.sort_values(
        ["candidate_uid_exact_v1", "decision_anchor_timestamp_utc_v1", "management_decision_index_v1"],
        kind="mergesort",
    ).groupby("candidate_uid_exact_v1", dropna=False):
        action_values = group["action_label_v1"].astype("string").tolist()
        transition_mask = group["management_row_key_v1"].astype("string").isin(transition_keys)
        policy_mask = group["management_row_key_v1"].astype("string").isin(management_policy_keys)
        agg_rows.append(
            {
                "candidate_uid": str(candidate_uid),
                "management_run_id_v1": _first_non_missing(group["run_id"].astype("string").tolist()),
                "management_trade_uid_v1": _first_non_missing(group["trade_uid_exact_v1"].astype("string").tolist()),
                "management_trade_id_v1": _first_non_missing(group["trade_id_exact_v1"].astype("string").tolist()),
                "management_entry_actualization_presence_status_v1": _first_non_missing(
                    group["entry_actualization_presence_status_v1"].astype("string").tolist()
                ),
                "management_terminal_outcome_status_v1": _first_non_missing(
                    group["terminal_outcome_availability_status_v1"].astype("string").tolist()
                ),
                "management_row_count_v1": int(len(group)),
                "management_transition_eligible_row_count_v1": int(transition_mask.sum()),
                "management_policy_logged_row_count_v1": int(policy_mask.sum()),
                "management_action_counts_json_v1": _json_counts(action_values),
                "management_first_action_v1": _first_non_missing(action_values),
                "management_last_action_v1": _last_non_missing(action_values),
                "management_first_decision_timestamp_v1": _first_non_missing(
                    group["decision_anchor_timestamp_utc_v1"].astype("string").tolist()
                ),
                "management_last_decision_timestamp_v1": _last_non_missing(
                    group["decision_anchor_timestamp_utc_v1"].astype("string").tolist()
                ),
                "management_has_hold_v1": bool((group["action_label_v1"].astype("string") == "HOLD").any()),
                "management_has_exit_now_v1": bool((group["action_label_v1"].astype("string") == "EXIT_NOW").any()),
                "management_has_better_exit_hindsight_only_v1": bool(
                    (group["management_path_relation_v1"].astype("string") == "BETTER_EXIT_HINDSIGHT_ONLY").any()
                ),
                "management_primary_model_score_max_v1": _numeric_or_none(group["primary_model_score_v1"], "max"),
                "management_primary_model_score_last_v1": _numeric_or_none(group["primary_model_score_v1"], "last"),
                "management_primary_model_score_mean_v1": _numeric_or_none(group["primary_model_score_v1"], "mean"),
                "management_row_keys_json_v1": _json_list(group["management_row_key_v1"].tolist()),
            }
        )
    management_agg = pd.DataFrame.from_records(agg_rows)
    numeric_count_cols = [
        "management_row_count_v1",
        "management_transition_eligible_row_count_v1",
        "management_policy_logged_row_count_v1",
    ]

    episode_view = entry.merge(
        management_agg,
        on="candidate_uid",
        how="outer",
        validate="one_to_one",
        indicator="episode_merge_source_v1",
    )
    episode_view["episode_source_status_v1"] = "ENTRY_DIRECT_ONLY"
    episode_view.loc[
        episode_view["episode_merge_source_v1"].astype("string").eq("both"),
        "episode_source_status_v1",
    ] = "ENTRY_DIRECT_WITH_MANAGEMENT_ROWS"
    episode_view.loc[
        episode_view["episode_merge_source_v1"].astype("string").eq("right_only"),
        "episode_source_status_v1",
    ] = "MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK"
    for column, management_column in [
        ("run_id", "management_run_id_v1"),
        ("trade_uid", "management_trade_uid_v1"),
        ("trade_id", "management_trade_id_v1"),
        ("decision_timestamp", "management_first_decision_timestamp_v1"),
    ]:
        if column in episode_view.columns and management_column in episode_view.columns:
            episode_view[column] = episode_view[column].where(~episode_view[column].isna(), episode_view[management_column])
    episode_view["unified_episode_key_v1"] = episode_view["unified_episode_key_v1"].where(
        ~episode_view["unified_episode_key_v1"].isna(),
        episode_view["run_id"].astype("string") + "|" + episode_view["candidate_uid"].astype("string"),
    )
    if "actualized_take_v1" in episode_view.columns and "management_terminal_outcome_status_v1" in episode_view.columns:
        management_terminal_exact = episode_view["management_terminal_outcome_status_v1"].astype("string").eq(
            "EXACT_TERMINAL_OUTCOME_AVAILABLE"
        )
        episode_view["actualized_take_v1"] = episode_view["actualized_take_v1"].where(
            ~episode_view["actualized_take_v1"].isna(),
            management_terminal_exact,
        )

    if not closed_trade_ledger.empty:
        existing_episode_candidates = set(episode_view["candidate_uid"].astype("string").dropna().tolist())
        ledger_only = closed_trade_ledger.loc[
            ~closed_trade_ledger["candidate_uid"].astype("string").isin(existing_episode_candidates)
        ].copy()
        if not ledger_only.empty:
            ledger_rows = pd.DataFrame(index=ledger_only.index, columns=episode_view.columns)
            for column in [
                "run_id",
                "candidate_uid",
                "trade_uid",
                "trade_id",
                "decision_timestamp",
                "entry_timestamp",
                "exit_timestamp",
                "used_for_training",
                "used_for_validation",
                "used_for_holdout",
                "realized_pnl_bps",
                "trade_outcome_class",
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
            ]:
                if column in ledger_only.columns and column in ledger_rows.columns:
                    ledger_rows[column] = ledger_only[column]
            if "split_bucket_v1" in ledger_rows.columns:
                ledger_rows["split_bucket_v1"] = _derive_split_bucket(ledger_only)
            if "entry_row_key_v1" in ledger_rows.columns:
                ledger_rows["entry_row_key_v1"] = "CLOSED_LEDGER|" + ledger_only["candidate_uid"].astype("string")
            if "direct_entry_as_of_row_uid_v1" in ledger_rows.columns:
                ledger_rows["direct_entry_as_of_row_uid_v1"] = (
                    "CLOSED_LEDGER_CANDIDATE_SNAPSHOT|" + ledger_only["candidate_uid"].astype("string")
                )
            if "logged_entry_action_v1" in ledger_rows.columns:
                ledger_rows["logged_entry_action_v1"] = "TAKE_NOW"
            if "entry_action_truth_status_v1" in ledger_rows.columns:
                ledger_rows["entry_action_truth_status_v1"] = "CLOSED_TRADE_LEDGER_ACTUALIZED_TAKE_NO_POLICY_ROW"
            if "entry_reward_semantics_status_v1" in ledger_rows.columns:
                ledger_rows["entry_reward_semantics_status_v1"] = "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE"
            if "policy_stack_stage_v1" in ledger_rows.columns:
                ledger_rows["policy_stack_stage_v1"] = "CLOSED_TRADE_LEDGER_TERMINAL_ONLY_BACKFILL_V1"
            if "direct_composite_status_v1" in ledger_rows.columns:
                ledger_rows["direct_composite_status_v1"] = "CLOSED_TRADE_LEDGER_ONLY_NO_DIRECT_COMPOSITE"
            if "policy_snapshot_status_v1" in ledger_rows.columns:
                ledger_rows["policy_snapshot_status_v1"] = "POLICY_HASH_FROM_CLOSED_TRADE_LEDGER"
            if "policy_snapshot_source_v1" in ledger_rows.columns:
                ledger_rows["policy_snapshot_source_v1"] = "CLOSED_TRADE_LEDGER_PLUS_CANDIDATE_SNAPSHOT"
            for column, source in [
                ("policy_version_v1", "policy_hash"),
                ("behavior_policy_id_v1", "policy_hash"),
                ("as_of_candidate_entry_bundle_sha256_v1", "entry_bundle_sha256"),
                ("as_of_candidate_exit_bundle_sha256_v1", "exit_bundle_sha256"),
            ]:
                if column in ledger_rows.columns and source in ledger_only.columns:
                    ledger_rows[column] = ledger_only[source]
            if "behavior_policy_id_status_v1" in ledger_rows.columns:
                ledger_rows["behavior_policy_id_status_v1"] = "POLICY_HASH_FROM_CLOSED_TRADE_LEDGER_NOT_LOGGED_ENTRY_POLICY"
            if "behavior_policy_kind_v1" in ledger_rows.columns:
                ledger_rows["behavior_policy_kind_v1"] = "CLOSED_TRADE_LEDGER_POLICY_HASH_ONLY"
            if "entry_action_propensity_status_v1" in ledger_rows.columns:
                ledger_rows["entry_action_propensity_status_v1"] = "PROPENSITY_NOT_ESTABLISHED"
            if "actualized_take_v1" in ledger_rows.columns:
                ledger_rows["actualized_take_v1"] = True
            if "terminal_route_status_v1" in ledger_rows.columns:
                ledger_rows["terminal_route_status_v1"] = "CLOSED_TRADE_LEDGER_ONLY_TERMINAL"
            if "terminal_activation_origin_v1" in ledger_rows.columns:
                ledger_rows["terminal_activation_origin_v1"] = "CLOSED_TRADE_LEDGER"
            if "terminal_activation_timestamp_utc_v1" in ledger_rows.columns and "entry_timestamp" in ledger_only.columns:
                ledger_rows["terminal_activation_timestamp_utc_v1"] = ledger_only["entry_timestamp"]
            if "terminal_closed_exit_reason_v1" in ledger_rows.columns and "exit_reason" in ledger_only.columns:
                ledger_rows["terminal_closed_exit_reason_v1"] = ledger_only["exit_reason"]
            for column in numeric_count_cols:
                if column in ledger_rows.columns:
                    ledger_rows[column] = 0
            for column in [
                "management_has_hold_v1",
                "management_has_exit_now_v1",
                "management_has_better_exit_hindsight_only_v1",
            ]:
                if column in ledger_rows.columns:
                    ledger_rows[column] = False
            ledger_rows["episode_source_status_v1"] = "CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS"
            ledger_rows["entry_to_management_link_status_v1"] = "CLOSED_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS"
            ledger_rows["unified_episode_observability_status_v1"] = "CLOSED_LEDGER_TERMINAL_ONLY_OBSERVABILITY_READY"
            ledger_rows["unified_episode_key_v1"] = (
                ledger_only["run_id"].astype("string") + "|" + ledger_only["candidate_uid"].astype("string")
            )
            episode_view = pd.concat([episode_view, ledger_rows], ignore_index=True)

    for column in numeric_count_cols:
        episode_view[column] = pd.to_numeric(episode_view[column], errors="coerce").fillna(0).astype(int)
    for column in [
        "management_has_hold_v1",
        "management_has_exit_now_v1",
        "management_has_better_exit_hindsight_only_v1",
    ]:
        episode_view[column] = episode_view[column].fillna(False).astype(bool)
    episode_view["entry_to_management_link_status_v1"] = "UNCLASSIFIED"
    actualized = episode_view["actualized_take_v1"].fillna(False).astype(bool)
    has_management = episode_view["management_row_count_v1"].gt(0)
    source_status = episode_view["episode_source_status_v1"].astype("string")
    has_entry_episode = ~source_status.isin(
        [
            "MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK",
            "CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS",
        ]
    )
    episode_view.loc[has_entry_episode & actualized & has_management, "entry_to_management_link_status_v1"] = (
        "ACTUALIZED_ENTRY_WITH_MANAGEMENT_ROWS"
    )
    episode_view.loc[has_entry_episode & actualized & ~has_management, "entry_to_management_link_status_v1"] = (
        "ACTUALIZED_ENTRY_WITHOUT_MANAGEMENT_ROWS_DIAGNOSTIC_ONLY"
    )
    episode_view.loc[has_entry_episode & ~actualized & has_management, "entry_to_management_link_status_v1"] = (
        "NON_ACTUALIZED_ENTRY_WITH_MANAGEMENT_READ_MODEL_ROWS"
    )
    episode_view.loc[has_entry_episode & ~actualized & ~has_management, "entry_to_management_link_status_v1"] = (
        "NON_ACTUALIZED_ENTRY_WITHOUT_MANAGEMENT_ROWS"
    )
    episode_view.loc[
        source_status.eq("MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK"),
        "entry_to_management_link_status_v1",
    ] = "MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK"
    episode_view.loc[
        source_status.eq("CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS"),
        "entry_to_management_link_status_v1",
    ] = "CLOSED_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS"
    episode_view["unified_episode_observability_status_v1"] = "ENTRY_TO_MANAGEMENT_OBSERVABILITY_READY"
    episode_view.loc[
        source_status.eq("MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK"),
        "unified_episode_observability_status_v1",
    ] = "MANAGEMENT_ONLY_OBSERVABILITY_READY_NO_FROZEN_ENTRY_ROUTE"
    episode_view.loc[
        source_status.eq("CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS"),
        "unified_episode_observability_status_v1",
    ] = "CLOSED_LEDGER_TERMINAL_ONLY_OBSERVABILITY_READY"

    event_base_cols = [
        "unified_event_key_v1",
        "unified_episode_key_v1",
        "rl_domain_v1",
        "run_id",
        "candidate_uid_v1",
        "trade_uid_v1",
        "trade_id_v1",
        "event_timestamp_utc_v1",
        "split_bucket_v1",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "action_label_v1",
        "action_truth_status_v1",
        "behavior_policy_id_v1",
        "behavior_policy_id_status_v1",
        "behavior_policy_kind_v1",
        "observed_action_propensity_v1",
        "propensity_status_v1",
        "per_action_propensity_vector_v1",
        "reward_semantics_status_v1",
        "terminal_outcome_status_v1",
        "entry_review_focus_v1",
        "management_path_relation_v1",
        "management_decision_index_v1",
        "source_row_key_v1",
    ]

    entry_events = pd.DataFrame(index=entry.index)
    entry_events["unified_event_key_v1"] = "ENTRY|" + entry["entry_row_key_v1"].astype("string")
    entry_events["unified_episode_key_v1"] = entry["unified_episode_key_v1"]
    entry_events["rl_domain_v1"] = "ENTRY"
    entry_events["run_id"] = entry["run_id"].astype("string")
    entry_events["candidate_uid_v1"] = entry["candidate_uid"].astype("string")
    entry_events["trade_uid_v1"] = entry["trade_uid"].astype("string")
    entry_events["trade_id_v1"] = entry["trade_id"].astype("string")
    entry_events["event_timestamp_utc_v1"] = entry["decision_timestamp"].astype("string")
    for column in ["split_bucket_v1", "used_for_training", "used_for_validation", "used_for_holdout"]:
        entry_events[column] = entry[column] if column in entry.columns else pd.NA
    entry_events["action_label_v1"] = entry["logged_entry_action_v1"].astype("string")
    entry_events["action_truth_status_v1"] = entry["entry_action_truth_status_v1"].astype("string")
    entry_events["behavior_policy_id_v1"] = entry["behavior_policy_id_v1"].astype("string")
    entry_events["behavior_policy_id_status_v1"] = entry["behavior_policy_id_status_v1"].astype("string")
    entry_events["behavior_policy_kind_v1"] = entry["behavior_policy_kind_v1"].astype("string")
    entry_events["observed_action_propensity_v1"] = pd.NA
    entry_events["propensity_status_v1"] = entry["entry_action_propensity_status_v1"].astype("string")
    entry_events["per_action_propensity_vector_v1"] = pd.NA
    entry_events["reward_semantics_status_v1"] = entry["entry_reward_semantics_status_v1"].astype("string")
    entry_terminal_status = pd.Series("ENTRY_TERMINAL_OUTCOME_NOT_AVAILABLE", index=entry.index, dtype="string")
    entry_terminal_status.loc[entry["actualized_take_v1"].fillna(False).astype(bool)] = (
        "ENTRY_TERMINAL_OUTCOME_AVAILABLE"
    )
    entry_events["terminal_outcome_status_v1"] = entry_terminal_status
    entry_events["entry_review_focus_v1"] = entry["entry_review_focus_v1"].astype("string")
    entry_events["management_path_relation_v1"] = pd.NA
    entry_events["management_decision_index_v1"] = pd.NA
    entry_events["source_row_key_v1"] = entry["entry_row_key_v1"].astype("string")
    for field in entry_obs_fields:
        entry_events[f"entry_obs__{field}"] = entry[field] if field in entry.columns else pd.NA
    for field in management_obs_fields:
        entry_events[f"management_obs__{field}"] = pd.NA

    management_events = pd.DataFrame(index=management.index)
    management_events["unified_event_key_v1"] = "MANAGEMENT|" + management["management_row_key_v1"].astype("string")
    management_events["unified_episode_key_v1"] = management["unified_episode_key_v1"]
    management_events["rl_domain_v1"] = "MANAGEMENT"
    management_events["run_id"] = management["run_id"].astype("string")
    management_events["candidate_uid_v1"] = management["candidate_uid_exact_v1"].astype("string")
    management_events["trade_uid_v1"] = management["trade_uid_exact_v1"].astype("string")
    management_events["trade_id_v1"] = management["trade_id_exact_v1"].astype("string")
    management_events["event_timestamp_utc_v1"] = management["decision_anchor_timestamp_utc_v1"].astype("string")
    for column in ["split_bucket_v1", "used_for_training", "used_for_validation", "used_for_holdout"]:
        management_events[column] = management[column] if column in management.columns else pd.NA
    management_events["action_label_v1"] = management["action_label_v1"].astype("string")
    management_events["action_truth_status_v1"] = management["observed_action_status_v1"].astype("string")
    management_events["behavior_policy_id_v1"] = management["behavior_policy_id_v1"].astype("string")
    management_events["behavior_policy_id_status_v1"] = management["behavior_policy_id_status_v1"].astype("string")
    management_events["behavior_policy_kind_v1"] = management["behavior_policy_kind_v1"].astype("string")
    management_events["observed_action_propensity_v1"] = management["observed_action_propensity_v1"]
    management_events["propensity_status_v1"] = management["policy_logging_propensity_status_v1"].astype("string")
    management_events["per_action_propensity_vector_v1"] = management["per_action_propensity_vector_v1"].astype("string")
    management_events["reward_semantics_status_v1"] = management["terminal_outcome_availability_status_v1"].astype("string")
    management_events["terminal_outcome_status_v1"] = management["terminal_outcome_availability_status_v1"].astype("string")
    management_events["entry_review_focus_v1"] = pd.NA
    management_events["management_path_relation_v1"] = management["management_path_relation_v1"].astype("string")
    management_events["management_decision_index_v1"] = management["management_decision_index_v1"]
    management_events["source_row_key_v1"] = management["management_row_key_v1"].astype("string")
    for field in entry_obs_fields:
        management_events[f"entry_obs__{field}"] = pd.NA
    for field in management_obs_fields:
        management_events[f"management_obs__{field}"] = management[field] if field in management.columns else pd.NA

    ledger_only_events = pd.DataFrame(columns=event_base_cols + [f"entry_obs__{field}" for field in entry_obs_fields] + [f"management_obs__{field}" for field in management_obs_fields])
    if not closed_trade_ledger.empty:
        event_candidates = set(entry["candidate_uid"].astype("string").dropna().tolist()) | set(
            management["candidate_uid_exact_v1"].astype("string").dropna().tolist()
        )
        ledger_only = closed_trade_ledger.loc[
            ~closed_trade_ledger["candidate_uid"].astype("string").isin(event_candidates)
        ].copy()
        if not ledger_only.empty:
            ledger_only_events = pd.DataFrame(index=ledger_only.index)
            ledger_only_events["unified_event_key_v1"] = "CLOSED_LEDGER|" + ledger_only["candidate_uid"].astype("string")
            ledger_only_events["unified_episode_key_v1"] = (
                ledger_only["run_id"].astype("string") + "|" + ledger_only["candidate_uid"].astype("string")
            )
            ledger_only_events["rl_domain_v1"] = "CLOSED_TRADE_LEDGER"
            ledger_only_events["run_id"] = ledger_only["run_id"].astype("string")
            ledger_only_events["candidate_uid_v1"] = ledger_only["candidate_uid"].astype("string")
            ledger_only_events["trade_uid_v1"] = ledger_only["trade_uid"].astype("string")
            ledger_only_events["trade_id_v1"] = ledger_only["trade_id"].astype("string")
            ledger_only_events["event_timestamp_utc_v1"] = ledger_only["decision_timestamp"].astype("string")
            ledger_only_events["split_bucket_v1"] = _derive_split_bucket(ledger_only)
            for column in ["used_for_training", "used_for_validation", "used_for_holdout"]:
                ledger_only_events[column] = ledger_only[column] if column in ledger_only.columns else pd.NA
            ledger_only_events["action_label_v1"] = "TAKE_NOW"
            ledger_only_events["action_truth_status_v1"] = "CLOSED_TRADE_LEDGER_ACTUALIZED_TAKE_NO_ENTRY_OR_MANAGEMENT_ROW"
            ledger_only_events["behavior_policy_id_v1"] = (
                ledger_only["policy_hash"].astype("string") if "policy_hash" in ledger_only.columns else pd.NA
            )
            ledger_only_events["behavior_policy_id_status_v1"] = "POLICY_HASH_FROM_CLOSED_TRADE_LEDGER_NOT_LOGGED_ENTRY_POLICY"
            ledger_only_events["behavior_policy_kind_v1"] = "CLOSED_TRADE_LEDGER_POLICY_HASH_ONLY"
            ledger_only_events["observed_action_propensity_v1"] = pd.NA
            ledger_only_events["propensity_status_v1"] = "PROPENSITY_NOT_ESTABLISHED"
            ledger_only_events["per_action_propensity_vector_v1"] = pd.NA
            ledger_only_events["reward_semantics_status_v1"] = "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE"
            ledger_only_events["terminal_outcome_status_v1"] = "EXACT_TERMINAL_OUTCOME_AVAILABLE"
            ledger_only_events["entry_review_focus_v1"] = (
                ledger_only["hindsight_entry_decision_review_v1"].astype("string")
                if "hindsight_entry_decision_review_v1" in ledger_only.columns
                else pd.NA
            )
            ledger_only_events["management_path_relation_v1"] = "NO_MANAGEMENT_ROW"
            ledger_only_events["management_decision_index_v1"] = pd.NA
            ledger_only_events["source_row_key_v1"] = "CLOSED_LEDGER|" + ledger_only["candidate_uid"].astype("string")
            for field in entry_obs_fields:
                ledger_only_events[f"entry_obs__{field}"] = pd.NA
            for field in management_obs_fields:
                ledger_only_events[f"management_obs__{field}"] = pd.NA
            ledger_only_events = _populate_entry_obs_from_candidate_snapshot(ledger_only_events, candidate_snapshot_df)
            ledger_only_events = ledger_only_events[
                event_base_cols
                + [f"entry_obs__{field}" for field in entry_obs_fields]
                + [f"management_obs__{field}" for field in management_obs_fields]
            ].copy()

    decision_event_view = pd.concat([entry_events, management_events, ledger_only_events], ignore_index=True)
    decision_event_view = decision_event_view[
        event_base_cols
        + [f"entry_obs__{field}" for field in entry_obs_fields]
        + [f"management_obs__{field}" for field in management_obs_fields]
    ].copy()
    decision_event_view = decision_event_view.sort_values(
        ["unified_episode_key_v1", "event_timestamp_utc_v1", "rl_domain_v1", "source_row_key_v1"],
        kind="mergesort",
    ).reset_index(drop=True)

    handoff_counts = entry_handoff_summary.get("management_handoff_status_counts_v1", {})
    expected_provable = int(handoff_counts.get("ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD", 0))
    expected_diagnostic = int(handoff_counts.get("ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW", 0))
    source_status = episode_view["episode_source_status_v1"].astype("string")
    has_entry_episode = ~source_status.isin(
        [
            "MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK",
            "CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS",
        ]
    )
    actualized_with_management = int((has_entry_episode & actualized & has_management).sum())
    actualized_without_management = int((has_entry_episode & actualized & ~has_management).sum())
    nonactualized_with_management = int((has_entry_episode & ~actualized & has_management).sum())
    management_only_episode_count = int(source_status.eq("MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK").sum())
    closed_ledger_only_episode_count = int(
        source_status.eq("CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS").sum()
    )
    closed_trade_episode_covered_count = None
    closed_trade_episode_expected_count = None
    if not closed_trade_ledger.empty:
        closed_trade_episode_expected_count = int(closed_trade_ledger["candidate_uid"].astype("string").nunique())
        closed_trade_episode_covered_count = int(
            len(
                set(closed_trade_ledger["candidate_uid"].astype("string").dropna().tolist())
                & set(episode_view["candidate_uid"].astype("string").dropna().tolist())
            )
        )
    management_presence_counts = _counts(management_row_df, "entry_actualization_presence_status_v1")
    expected_nonactualized_with_management = int(management_presence_counts.get("DIRECT_ENTRY_ROUTE_EXACT_NON_ACTUALIZED", 0))

    consistency_rows = [
        {
            "check_name_v1": "UNIFIED_EPISODE_VIEW_RETAINS_ENTRY_EXACTLY",
            "status_v1": "PASS"
            if int(episode_view["episode_source_status_v1"].astype("string").isin(["ENTRY_DIRECT_ONLY", "ENTRY_DIRECT_WITH_MANAGEMENT_ROWS"]).sum()) == int(len(entry_df))
            and int(episode_view.duplicated(subset=["unified_episode_key_v1"]).sum()) == 0
            and episode_view["unified_episode_key_v1"].astype("string").notna().all()
            else "FAIL",
            "observed_value_v1": int(
                episode_view["episode_source_status_v1"].astype("string").isin(["ENTRY_DIRECT_ONLY", "ENTRY_DIRECT_WITH_MANAGEMENT_ROWS"]).sum()
            ),
            "expected_value_v1": int(len(entry_df)),
            "note_v1": "Every direct entry observability row must still appear exactly once; management-only rows are additive.",
        },
        {
            "check_name_v1": "UNIFIED_EVENT_VIEW_COVERS_ENTRY_MANAGEMENT_AND_LEDGER_ONLY_ROWS",
            "status_v1": "PASS"
            if len(decision_event_view) == len(entry_df) + len(management_row_df) + len(ledger_only_events)
            else "FAIL",
            "observed_value_v1": int(len(decision_event_view)),
            "expected_value_v1": int(len(entry_df) + len(management_row_df) + len(ledger_only_events)),
            "note_v1": "The event view stacks all entry, management, and closed-ledger-only diagnostic rows.",
        },
        {
            "check_name_v1": "UNIFIED_EPISODE_VIEW_COVERS_CLOSED_TRADE_LEDGER_WHEN_AVAILABLE",
            "status_v1": "PASS"
            if closed_trade_episode_expected_count is None
            or closed_trade_episode_covered_count == closed_trade_episode_expected_count
            else "FAIL",
            "observed_value_v1": closed_trade_episode_covered_count,
            "expected_value_v1": closed_trade_episode_expected_count,
            "note_v1": "Closed truth trades must be represented in the unified episode universe without fabricating entry rows.",
        },
        {
            "check_name_v1": "ENTRY_EXACT_MODEL_SNAPSHOT_READY_FOR_ALL_ENTRY_ROWS",
            "status_v1": "PASS"
            if int(entry_df["policy_version_v1"].astype("string").notna().sum()) == int(len(entry_df))
            else "FAIL",
            "observed_value_v1": int(entry_df["policy_version_v1"].astype("string").notna().sum()),
            "expected_value_v1": int(len(entry_df)),
            "note_v1": "Entry side must keep exact candidate policy snapshot coverage.",
        },
        {
            "check_name_v1": "MANAGEMENT_POLICY_LOG_COVERS_TRANSITION_ELIGIBLE_ROWS",
            "status_v1": "PASS" if policy_covers_transition else "FAIL",
            "observed_value_v1": int(len(transition_keys & management_policy_keys)),
            "expected_value_v1": int(len(transition_keys)),
            "note_v1": "Management policy log must cover every realized transition-eligible row exactly.",
        },
        {
            "check_name_v1": "ACTUALIZED_ENTRY_TO_MANAGEMENT_COVERAGE_MATCHES_HANDOFF_SUMMARY",
            "status_v1": "PASS"
            if actualized_with_management == expected_provable and actualized_without_management == expected_diagnostic
            else "FAIL",
            "observed_value_v1": json.dumps(
                {
                    "actualized_with_management_rows_v1": actualized_with_management,
                    "actualized_without_management_rows_v1": actualized_without_management,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                {
                    "actualized_with_management_rows_v1": expected_provable,
                    "actualized_without_management_rows_v1": expected_diagnostic,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "The six diagnostic-only actual takes remain explicit; they are not upgraded into fabricated management heads.",
        },
        {
            "check_name_v1": "NON_ACTUALIZED_ENTRY_MANAGEMENT_READ_MODEL_ROWS_MATCH_MANAGEMENT_PRESENCE",
            "status_v1": "PASS" if nonactualized_with_management == expected_nonactualized_with_management else "FAIL",
            "observed_value_v1": nonactualized_with_management,
            "expected_value_v1": expected_nonactualized_with_management,
            "note_v1": "Non-actualized entry rows may still have management read-model rows; that must match management presence accounting exactly.",
        },
        {
            "check_name_v1": "ENTRY_PROPENSITY_REMAINS_UNESTABLISHED_NO_SYNTHETIC_FILL",
            "status_v1": "PASS"
            if int(entry_df["entry_action_propensity_status_v1"].astype("string").eq("PROPENSITY_NOT_ESTABLISHED").sum())
            == int(len(entry_df))
            else "FAIL",
            "observed_value_v1": int(
                entry_df["entry_action_propensity_status_v1"].astype("string").eq("PROPENSITY_NOT_ESTABLISHED").sum()
            ),
            "expected_value_v1": int(len(entry_df)),
            "note_v1": "Unified RL must not fabricate entry propensities before exact live entry action logging exists.",
        },
    ]
    consistency_audit = pd.DataFrame.from_records(consistency_rows)
    failed_checks = int((consistency_audit["status_v1"].astype("string") != "PASS").sum())

    summary = {
        "layer_name": "RL_UNIFIED_OBSERVABILITY_SUMMARY_V1",
        "review_dir_v1": str(review_dir),
        "management_policy_dir_v1": str(management_policy_dir),
        "entry_episode_rows_v1": int(len(episode_view)),
        "decision_event_rows_v1": int(len(decision_event_view)),
        "entry_event_rows_v1": int((decision_event_view["rl_domain_v1"].astype("string") == "ENTRY").sum()),
        "management_event_rows_v1": int((decision_event_view["rl_domain_v1"].astype("string") == "MANAGEMENT").sum()),
        "closed_trade_ledger_event_rows_v1": int(
            (decision_event_view["rl_domain_v1"].astype("string") == "CLOSED_TRADE_LEDGER").sum()
        ),
        "episode_source_counts_v1": _counts(episode_view, "episode_source_status_v1"),
        "entry_direct_episode_rows_v1": int(
            episode_view["episode_source_status_v1"]
            .astype("string")
            .isin(["ENTRY_DIRECT_ONLY", "ENTRY_DIRECT_WITH_MANAGEMENT_ROWS"])
            .sum()
        ),
        "management_only_episode_rows_v1": management_only_episode_count,
        "closed_trade_ledger_only_episode_rows_v1": closed_ledger_only_episode_count,
        "closed_trade_ledger_episode_covered_count_v1": closed_trade_episode_covered_count,
        "closed_trade_ledger_episode_expected_count_v1": closed_trade_episode_expected_count,
        "management_transition_eligible_rows_v1": int(len(management_transition_df)),
        "management_policy_logged_rows_v1": int(len(policy_log)),
        "entry_action_counts_v1": _counts(entry_df, "logged_entry_action_v1"),
        "management_action_counts_v1": _counts(management_row_df, "action_label_v1"),
        "entry_to_management_link_counts_v1": _counts(episode_view, "entry_to_management_link_status_v1"),
        "actualized_entry_with_management_rows_v1": actualized_with_management,
        "actualized_entry_without_management_rows_v1": actualized_without_management,
        "nonactualized_entry_with_management_read_model_rows_v1": nonactualized_with_management,
        "entry_observation_feature_count_v1": int(len(entry_obs_fields)),
        "management_observation_feature_count_v1": int(len(management_obs_fields)),
        "management_policy_readiness_v1": management_policy_summary.get("behavior_policy_readiness_v1")
        or management_policy_summary.get("behavior_policy_identity_summary_v1", {}).get("status_v1"),
        "management_propensity_readiness_v1": management_policy_summary.get("propensity_readiness_v1")
        or next(iter(management_policy_summary.get("propensity_status_counts_v1", {}).keys()), None),
        "failed_check_count_v1": failed_checks,
    }
    status = {
        "layer_name": "RL_UNIFIED_OBSERVABILITY_STATUS_V1",
        "UNIFIED_RL_OBSERVABILITY_STATUS": (
            "READY_ENTRY_AND_MANAGEMENT_OBSERVABILITY" if failed_checks == 0 else "ISSUES_FOUND"
        ),
        "UNIFIED_RL_SCOPE_STATUS": "ENTRY_PLUS_MANAGEMENT",
        "UNIFIED_EPISODE_SCOPE_STATUS": (
            "ENTRY_MANAGEMENT_AND_CLOSED_LEDGER_UNION"
            if closed_trade_episode_expected_count is not None
            else "ENTRY_MANAGEMENT_UNION"
        ),
        "ENTRY_POLICY_SNAPSHOT_STATUS": entry_status.get("ENTRY_POLICY_SNAPSHOT_STATUS"),
        "ENTRY_DIRECT_ACTION_STATUS": entry_status.get("ENTRY_DIRECT_ACTION_STATUS"),
        "ENTRY_PROPENSITY_STATUS": entry_status.get("ENTRY_PROPENSITY_STATUS"),
        "MANAGEMENT_RL_READINESS_STATUS": management_readiness_status.get("MANAGEMENT_RL_READINESS_STATUS"),
        "MANAGEMENT_BANDIT_STATUS": management_bandit_status.get("MANAGEMENT_BANDIT_STATUS"),
        "MANAGEMENT_SEQUENCE_STATUS": management_sequence_status.get("MANAGEMENT_RL_SEQUENCE_STATUS"),
        "MANAGEMENT_PROPENSITY_STATUS": (
            "READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY" if policy_covers_transition else "PROPENSITY_COVERAGE_NOT_FULLY_ESTABLISHED"
        ),
        "FULL_POLICY_READY_STATUS": "PARTIAL_ENTRY_PROPENSITY_NOT_ESTABLISHED",
        "not_trainer": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    contract = {
        "layer_name": "RL_UNIFIED_OBSERVABILITY_CONTRACT_V1",
        "mode_v1": "ENTRY_PLUS_MANAGEMENT_OBSERVABILITY_ONLY",
        "episode_view_v1": UNIFIED_RL_EPISODE_VIEW,
        "decision_event_view_v1": UNIFIED_RL_DECISION_EVENT_VIEW,
        "domains_v1": ["ENTRY", "MANAGEMENT", "CLOSED_TRADE_LEDGER"],
        "entry_action_space_v1": ["SKIP", "TAKE_NOW", "WAIT"],
        "management_action_space_v1": ["HOLD", "EXIT_NOW"],
        "entry_observation_feature_names_v1": entry_obs_fields,
        "management_observation_feature_names_v1": management_obs_fields,
        "join_keys_v1": {
            "entry_to_management_candidate_key_v1": ["run_id", "candidate_uid"],
            "management_policy_key_v1": ["management_row_key_v1"],
        },
        "semantics_v1": {
            "entry_v1": "COMPOSITIONAL_READ_MODEL_WITH_EXACT_MODEL_SNAPSHOT_AND_HINDSIGHT_REWARD_CHANNELS",
            "management_v1": "REALIZED_PATH_OBSERVABILITY_WITH_EXACT_POLICY_LOGGING_WHERE_TRANSITION_ELIGIBLE",
            "closed_trade_ledger_v1": "TERMINAL_DIAGNOSTIC_COVERAGE_FOR_CLOSED_TRADES_WITHOUT_FROZEN_ENTRY_OR_MANAGEMENT_ROWS",
            "propensity_v1": "MANAGEMENT_DETERMINISTIC_LOGGED_PROPENSITY_READY; ENTRY_PROPENSITY_NOT_ESTABLISHED",
        },
        "prohibitions_v1": [
            "Do not train a live controller directly from hindsight-only SKIP avoided-loss rows.",
            "Do not treat ENTRY direct composition as exact logged live behavior policy.",
            "Do not synthesize entry propensities.",
            "Do not upgrade diagnostic-only management handoffs into trainable management heads.",
            "Do not treat CLOSED_TRADE_LEDGER rows as entry-policy truth or management-policy truth.",
        ],
    }
    manifest = {
        "layer_name": "RL_UNIFIED_OBSERVABILITY_MANIFEST_V1",
        "mode_v1": "APPEND_ONLY_EXTENSION",
        "review_dir_v1": str(review_dir),
        "management_policy_dir_v1": str(management_policy_dir),
        "artifacts_v1": {
            "contract_v1": UNIFIED_RL_CONTRACT,
            "episode_view_v1": UNIFIED_RL_EPISODE_VIEW,
            "decision_event_view_v1": UNIFIED_RL_DECISION_EVENT_VIEW,
            "summary_v1": UNIFIED_RL_SUMMARY,
            "status_v1": UNIFIED_RL_STATUS,
            "consistency_audit_v1": UNIFIED_RL_AUDIT,
        },
    }

    return {
        "contract_v1": contract,
        "episode_view_v1_df": episode_view,
        "decision_event_view_v1_df": decision_event_view,
        "summary_v1": summary,
        "status_v1": status,
        "consistency_audit_v1_df": consistency_audit,
        "manifest_v1": manifest,
    }


def materialize_truth_rl_unified_observability(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    management_policy_dir: Path | None = None,
    extension_dir: Path | None = None,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_review_dir = _resolve_review_dir(reports_root, str(review_dir) if review_dir else None)
    resolved_policy_dir = _resolve_management_policy_dir(
        reports_root,
        str(management_policy_dir) if management_policy_dir else None,
    )
    closed_trade_ledger_df = pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet")

    payload = build_unified_rl_observability_payload(
        entry_df=pd.read_parquet(resolved_review_dir / "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"),
        management_row_df=pd.read_parquet(
            resolved_review_dir / "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet"
        ),
        management_transition_df=pd.read_parquet(
            resolved_review_dir / "shadow_meta_all_trade_review_management_rl_transition_eligible_view_v1.parquet"
        ),
        management_bandit_df=pd.read_parquet(
            resolved_review_dir / "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet"
        ),
        management_exit_local_scored_df=pd.read_parquet(
            resolved_review_dir / "shadow_meta_all_trade_review_management_exit_local_all_eligible_scored_view_v1.parquet"
        ),
        management_policy_log_df=pd.read_parquet(
            resolved_policy_dir / "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet"
        ),
        closed_trade_ledger_df=closed_trade_ledger_df,
        candidate_snapshot_df=_load_candidate_snapshots(reports_root, closed_trade_ledger_df),
        entry_contract=_load_json(resolved_review_dir / "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json"),
        management_observation_contract=_load_json(
            resolved_review_dir / "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json"
        ),
        entry_status=_load_json(resolved_review_dir / "shadow_meta_all_trade_review_entry_rl_observability_status_v1.json"),
        management_readiness_status=_load_json(
            resolved_review_dir / "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json"
        ),
        management_bandit_status=_load_json(resolved_review_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json"),
        management_sequence_status=_load_json(
            resolved_review_dir / "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json"
        ),
        management_policy_summary=_load_json(
            resolved_policy_dir / "shadow_meta_all_trade_review_management_policy_logging_summary_v1.json"
        ),
        entry_handoff_summary=_load_json(
            resolved_review_dir / "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json"
        ),
        review_dir=resolved_review_dir,
        management_policy_dir=resolved_policy_dir,
    )
    if int(payload["summary_v1"].get("failed_check_count_v1", -1)) != 0:
        raise RuntimeError("UNIFIED_RL_OBSERVABILITY_V1 consistency checks failed; refusing to materialize.")

    if extension_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extension_dir = reports_root / f"{LEDGER_NAMESPACE_PREFIX}{stamp}_{UNIFIED_RL_EXTENSION_SUFFIX}"
    extension_dir = Path(extension_dir).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=False)

    payload["episode_view_v1_df"].to_parquet(extension_dir / UNIFIED_RL_EPISODE_VIEW, index=False)
    payload["decision_event_view_v1_df"].to_parquet(extension_dir / UNIFIED_RL_DECISION_EVENT_VIEW, index=False)
    payload["consistency_audit_v1_df"].to_csv(extension_dir / UNIFIED_RL_AUDIT, index=False)
    (extension_dir / UNIFIED_RL_CONTRACT).write_text(
        json.dumps(payload["contract_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / UNIFIED_RL_SUMMARY).write_text(
        json.dumps(payload["summary_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / UNIFIED_RL_STATUS).write_text(
        json.dumps(payload["status_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / UNIFIED_RL_MANIFEST).write_text(
        json.dumps(payload["manifest_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    top_level_summary = dict(payload["summary_v1"])
    top_level_summary["extension_dir_v1"] = str(extension_dir)
    top_level_summary["review_dir_v1"] = str(resolved_review_dir)
    top_level_summary["management_policy_dir_v1"] = str(resolved_policy_dir)
    top_level_summary["status_v1"] = payload["status_v1"]
    (reports_root / TOP_LEVEL_SUMMARY).write_text(
        json.dumps(top_level_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "extension_dir": extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["summary_v1"],
        "status": payload["status_v1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize unified entry+management RL observability from truth artifacts.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--review-dir", type=str, default=None)
    parser.add_argument("--management-policy-dir", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    args = parser.parse_args()

    result = materialize_truth_rl_unified_observability(
        _resolve_reports_root(args.reports_root),
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        management_policy_dir=Path(args.management_policy_dir).expanduser().resolve() if args.management_policy_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
    )
    print(
        json.dumps(
            {
                "extension_dir": str(result["extension_dir"]),
                "top_level_summary_path": str(result["top_level_summary_path"]),
                "status": result["status"],
                "summary": result["summary"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
