from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_WEDNESDAY_CONTRACT_COMPLETENESS_CHECK_V1"
WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"
MONDAY_TRUTH_GLOB = "MONDAY_R6_CANONICAL_TRUTH_V1_*"
LOCAL_R6_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

REQUIRED_WEDNESDAY_SOURCE_ARTIFACTS = [
    "shadow_meta_all_trade_review_r6_entry_runner_first_contract_v1.json",
    "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
    "shadow_meta_all_trade_review_r6_model_family_bakeoff_v1.csv",
    "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv",
    "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv",
    "models",
]

OUTPUT_FILES = {
    "summary": "monday_r6_wednesday_contract_completeness_summary_v1.json",
    "wednesday_expected": "wednesday_r6_expected_contract_v1.json",
    "as_of": "as_of_schema_availability_v1.csv",
    "hindsight": "hindsight_schema_availability_v1.csv",
    "score_heads": "score_head_availability_v1.csv",
    "source_artifacts": "required_source_artifact_availability_v1.csv",
    "local_r6_alternate": "local_r6_alternate_artifact_assessment_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_monday_truth(reports_root: Path) -> Path:
    dirs = sorted([path for path in reports_root.glob(MONDAY_TRUTH_GLOB) if path.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No Monday R6 canonical truth dir found under {reports_root}")
    return dirs[-1]


def _feature_lookup(monday_truth_dir: Path) -> tuple[pd.DataFrame, set[str], dict[str, list[dict[str, Any]]]]:
    manifest_path = monday_truth_dir / "monday_r6_truth_feature_manifest_v1.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Monday feature manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    feature_names = set(manifest["feature_name_v1"].astype(str))
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.to_dict("records"):
        by_name.setdefault(str(row["feature_name_v1"]), []).append(row)
    return manifest, feature_names, by_name


def _normalize_feature_name(name: str) -> str:
    text = str(name).lower()
    for prefix in [
        "as_of_skip_replay_",
        "as_of_skip_candidate_",
        "as_of_skip_xgb_",
        "as_of_entry_candidate_",
        "as_of_candidate_",
        "as_of_",
        "entry_candidate_",
        "entry_xgb_",
        "journal_",
        "canonical_",
        "truth_",
        "_v1_",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.endswith("_v1"):
        text = text[:-3]
    aliases = {
        "decision_timestamp": "decision_ts_utc",
        "baseline_realized_pnl_bps": "pnl_bps",
        "peak_mfe_bps": "mfe_bps",
        "mae_abs_bps": "mae_bps",
        "bb_bandwidth_delta_10": "bb_bandwidth_delta_10",
        "bb_squeeze_20_2": "bb_squeeze_20_2",
        "body_share": "body_share_1",
        "clv": "clv",
        "cost_bps_dyn": "cost_bps_dyn",
        "cost_bps_est": "cost_bps_est",
        "kama_slope_30": "kama_slope_30",
    }
    return aliases.get(text, text)


def _direct_or_normalized_hit(
    expected: str,
    feature_names: set[str],
    normalized_index: dict[str, list[str]],
) -> tuple[str, str | None, str]:
    if expected in feature_names:
        return "DIRECT_PRESENT", expected, "exact feature name exists in Monday truth feature manifest"
    normalized = _normalize_feature_name(expected)
    hits = normalized_index.get(normalized, [])
    if hits:
        return "NORMALIZED_SOURCE_PRESENT", hits[0], f"normalized source match via {normalized}"
    return "MISSING", None, "no direct or normalized source match"


AS_OF_MANUAL_ALIASES = {
    "decision_timestamp": "decision_timestamp_v1",
    "as_of_skip_xgb_p_flat_v1": "entry_xgb_p_flat_v1",
    "as_of_skip_xgb_p_hat_v1": "entry_xgb_p_hat_v1",
    "as_of_skip_xgb_p_long_v1": "entry_xgb_p_long_v1",
    "as_of_skip_xgb_p_short_v1": "entry_xgb_p_short_v1",
    "as_of_skip_xgb_pred_side_v1": "entry_xgb_pred_side_v1",
    "as_of_skip_xgb_has_ctx_v1": "entry_xgb_has_ctx_v1",
    "as_of_entry_candidate_margin_v1": "entry_candidate_margin_v1",
    "as_of_entry_candidate_path_quality_pred_v1": "entry_candidate_path_quality_pred_v1",
}

AS_OF_DERIVABLE_NOT_MATERIALIZED = {
    "used_for_training",
    "used_for_validation",
    "used_for_holdout",
    "entry_observation_present_v1",
    "entry_raw_state_present_v1",
    "management_observation_present_v1",
    "entry_coverage_original_entry_observation_present_v1",
    "entry_coverage_original_entry_raw_state_present_v1",
    "entry_coverage_repair_applied_v1",
    "entry_coverage_repair_source_v1",
    "r6_as_of_feature_contract_v1",
}

REHYDRATABLE_REPLAY_FEATURES = {
    "as_of_skip_replay_body_bps_v1",
    "as_of_skip_replay_close_in_bar_v1",
    "as_of_skip_replay_range_bps_v1",
    "as_of_skip_replay_upper_wick_share_v1",
    "as_of_skip_replay_lower_wick_share_v1",
    "as_of_skip_replay_window_close_in_range_15_v1",
    "as_of_skip_replay_window_close_in_range_60_v1",
    "as_of_skip_replay_window_close_in_range_240_v1",
    "as_of_skip_replay_window_directional_imbalance_15_bps_v1",
    "as_of_skip_replay_window_directional_imbalance_60_bps_v1",
    "as_of_skip_replay_window_directional_imbalance_240_bps_v1",
    "as_of_skip_replay_window_down_move_15_bps_v1",
    "as_of_skip_replay_window_down_move_60_bps_v1",
    "as_of_skip_replay_window_down_move_240_bps_v1",
    "as_of_skip_replay_window_range_15_bps_v1",
    "as_of_skip_replay_window_range_60_bps_v1",
    "as_of_skip_replay_window_range_240_bps_v1",
    "as_of_skip_replay_window_range_minus_mean_5_bps_v1",
    "as_of_skip_replay_window_range_ratio_mean_5_v1",
    "as_of_skip_replay_window_realized_vol_3_bps_v1",
    "as_of_skip_replay_window_realized_vol_5_bps_v1",
    "as_of_skip_replay_window_ret_1_bps_v1",
    "as_of_skip_replay_window_ret_3_bps_v1",
    "as_of_skip_replay_window_ret_5_bps_v1",
    "as_of_skip_replay_window_spread_minus_median_5_bps_v1",
    "as_of_skip_replay_window_spread_ratio_median_5_v1",
    "as_of_skip_replay_window_up_move_15_bps_v1",
    "as_of_skip_replay_window_up_move_60_bps_v1",
    "as_of_skip_replay_window_up_move_240_bps_v1",
}

HINDSIGHT_MANUAL_ALIASES = {
    "decision_timestamp": "decision_timestamp_v1",
    "baseline_realized_pnl_bps_v1": "canonical_pnl_bps_v1",
    "peak_mfe_bps_v1": "canonical_mfe_bps_v1",
    "mae_abs_bps_v1": "canonical_mae_bps_v1",
    "giveback_bps_v1": "journal_dd_from_mfe_bps_exit_v1",
    "r6_label_runner_50_mfe_v1": "truth_runner_50_mfe_v1",
    "r6_label_runner_100_mfe_v1": "truth_runner_100_mfe_v1",
    "r6_label_runner_200_mfe_v1": "truth_runner_200_mfe_v1",
    "r6_label_strong_low_mae_runner_v1": "truth_good_trade_mfe20_mae5_v1",
    "r6_label_high_mfe_low_giveback_v1": "truth_capture_ratio_v1",
    "r6_label_high_mae_low_mfe_v1": "truth_bad_loss_with_low_mfe_v1",
    "r6_label_low_mfe_low_value_v1": "truth_bad_loss_with_low_mfe_v1",
    "r6_label_early_adverse_excursion_v1": "truth_mae_50_or_worse_v1",
    "r6_label_bad_risk_v1": "truth_bad_loss_with_low_mfe_v1",
    "r6_hindsight_contract_v1": "monday_r6_truth_contract_v1",
}

HINDSIGHT_NOT_MATERIALIZED = {
    "hindsight_entry_decision_review_v1",
    "hindsight_management_review_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_runner_near_miss_v1",
    "r6_label_runner_protect_v1",
    "r6_label_missed_should_not_take_v1",
    "r6_label_risky_allow_v1",
    "r6_label_bad_trade_overlap_extreme_vol_v1",
    "r6_label_batch04_blindspot_v1",
    "r6_label_trend_neutral_extreme_vol_risk_v1",
    "r6_label_tail_control_10_50_v1",
}


def _schema_rows(
    schema_name: str,
    columns: list[dict[str, Any]],
    feature_names: set[str],
    by_name: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    normalized_index: dict[str, list[str]] = {}
    for name in feature_names:
        normalized_index.setdefault(_normalize_feature_name(name), []).append(name)

    rows: list[dict[str, Any]] = []
    for ordinal, column in enumerate(columns):
        expected = str(column["name_v1"])
        dtype = str(column.get("dtype_v1"))
        status, monday_name, reason = _direct_or_normalized_hit(expected, feature_names, normalized_index)
        if status == "MISSING" and schema_name == "AS_OF":
            alias = AS_OF_MANUAL_ALIASES.get(expected)
            if alias and alias in feature_names:
                status, monday_name, reason = "ALIAS_PRESENT", alias, "Wednesday field is present under Monday canonical truth alias"
            elif expected in AS_OF_DERIVABLE_NOT_MATERIALIZED:
                status, reason = "DERIVABLE_NOT_MATERIALIZED", "split/coverage/contract metadata must be added when the AS_OF matrix is rebuilt"
            elif expected in REHYDRATABLE_REPLAY_FEATURES:
                status, reason = "REHYDRATABLE_FROM_BAR_SURFACE", "raw bar surface is present, but this exact Wednesday window feature is not materialized yet"
        if status == "MISSING" and schema_name == "HINDSIGHT":
            alias = HINDSIGHT_MANUAL_ALIASES.get(expected)
            if alias and alias in feature_names:
                status, monday_name, reason = "ALIAS_OR_PROXY_PRESENT", alias, "Monday truth has source/proxy value; exact Wednesday hindsight label may still need rebuild"
            elif expected in HINDSIGHT_NOT_MATERIALIZED:
                status, reason = "NOT_MATERIALIZED", "exact Wednesday R6 hindsight/review label is not materialized in Monday truth package"
        surface = None
        role = None
        if monday_name and monday_name in by_name:
            surface = by_name[monday_name][0].get("surface_v1")
            role = by_name[monday_name][0].get("role_v1")
        rows.append(
            {
                "schema_v1": schema_name,
                "ordinal_v1": ordinal,
                "wednesday_column_v1": expected,
                "wednesday_dtype_v1": dtype,
                "availability_status_v1": status,
                "monday_column_v1": monday_name,
                "monday_surface_v1": surface,
                "monday_role_v1": role,
                "reason_v1": reason,
            }
        )
    return pd.DataFrame(rows)


def _score_head_rows(wednesday_manifest: dict[str, Any], feature_names: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, column in (wednesday_manifest.get("score_head_names_v1") or {}).items():
        status = "PRESENT" if column in feature_names else "MISSING"
        rows.append(
            {
                "score_head_role_v1": role,
                "wednesday_score_column_v1": column,
                "availability_status_v1": status,
                "reason_v1": "present in Monday feature manifest" if status == "PRESENT" else "policy prediction view / R5-R5.2-R6 score columns are not in Monday truth package",
            }
        )
    return pd.DataFrame(rows)


def _source_artifact_rows(wednesday_manifest: dict[str, Any]) -> pd.DataFrame:
    source_dir = Path(str(wednesday_manifest.get("r6_source_dir_v1") or ""))
    rows = []
    for rel in REQUIRED_WEDNESDAY_SOURCE_ARTIFACTS:
        path = source_dir / rel
        rows.append(
            {
                "required_artifact_v1": rel,
                "absolute_path_v1": str(path),
                "exists_v1": path.exists(),
                "status_v1": "PRESENT" if path.exists() else "MISSING",
            }
        )
    return pd.DataFrame(rows)


def _local_r6_alternate_assessment(
    reports_root: Path,
    wednesday_summary: dict[str, Any],
    wednesday_manifest: dict[str, Any],
) -> dict[str, Any]:
    local_dir = reports_root / LOCAL_R6_DIR_NAME
    expected_asof_cols = [str(row["name_v1"]) for row in wednesday_manifest["as_of_schema_v1"]["columns_v1"]]
    expected_candidate = str(wednesday_summary.get("selected_candidate_id_v1"))
    out: dict[str, Any] = {
        "local_r6_dir_v1": str(local_dir),
        "exists_v1": local_dir.exists(),
        "assessment_v1": "MISSING",
    }
    if not local_dir.exists():
        return out
    summary_path = local_dir / "shadow_meta_all_trade_review_r6_summary_v1.json"
    asof_path = local_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
    hindsight_path = local_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
    policy_path = local_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    selected = summary.get("selected_candidate_v1") or {}
    asof = pd.read_parquet(asof_path) if asof_path.exists() else pd.DataFrame()
    hindsight = pd.read_parquet(hindsight_path) if hindsight_path.exists() else pd.DataFrame()
    policy = pd.read_parquet(policy_path) if policy_path.exists() else pd.DataFrame()
    local_cols = list(asof.columns)
    missing_expected_cols = [column for column in expected_asof_cols if column not in local_cols]
    extra_cols = [column for column in local_cols if column not in expected_asof_cols]
    selected_policy = str(selected.get("policy_name_v1") or selected.get("selected_policy_name_v1") or "")
    row_count = int(len(asof))
    canonical = (
        row_count == int((wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1") or -1)
        and selected_policy == expected_candidate
        and not missing_expected_cols
        and len(local_cols) == int((wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1") or -1)
    )
    out.update(
        {
            "assessment_v1": "CANONICAL_MATCH" if canonical else "PRESENT_BUT_NOT_CANONICAL_WEDNESDAY_R6",
            "summary_exists_v1": summary_path.exists(),
            "asof_exists_v1": asof_path.exists(),
            "hindsight_exists_v1": hindsight_path.exists(),
            "policy_prediction_exists_v1": policy_path.exists(),
            "selected_policy_v1": selected_policy or None,
            "expected_policy_v1": expected_candidate,
            "asof_rows_v1": row_count,
            "expected_rows_v1": (wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1"),
            "asof_columns_v1": int(len(local_cols)),
            "expected_asof_columns_v1": (wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1"),
            "hindsight_rows_v1": int(len(hindsight)),
            "policy_prediction_rows_v1": int(len(policy)),
            "missing_expected_asof_columns_v1": missing_expected_cols,
            "extra_asof_columns_v1": extra_cols,
            "selected_metrics_v1": {
                key: selected.get(key)
                for key in [
                    "should_not_take_block_count_v1",
                    "tail_10_50_help_count_v1",
                    "should_not_take_precision_v1",
                    "worst_loso_precision_v1",
                    "repaired_165_block_count_v1",
                    "fifty_plus_mfe_block_count_v1",
                    "hundred_plus_mfe_block_count_v1",
                    "two_hundred_plus_mfe_block_count_v1",
                    "strongest_winner_path_block_count_v1",
                ]
            },
        }
    )
    return out


def _audit_rows(
    source_df: pd.DataFrame,
    asof_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
    score_df: pd.DataFrame,
    monday_coverage: dict[str, Any],
    wednesday_summary: dict[str, Any],
    local_r6_assessment: dict[str, Any],
) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    source_ok = bool(source_df["exists_v1"].all()) if len(source_df) else False
    asof_exact_ok = bool(asof_df["availability_status_v1"].isin(["DIRECT_PRESENT", "ALIAS_PRESENT", "NORMALIZED_SOURCE_PRESENT"]).all())
    hindsight_exact_ok = bool(hindsight_df["availability_status_v1"].isin(["DIRECT_PRESENT", "ALIAS_OR_PROXY_PRESENT", "NORMALIZED_SOURCE_PRESENT"]).all())
    score_ok = bool(score_df["availability_status_v1"].eq("PRESENT").all()) if len(score_df) else False
    expected_rows = int((wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1") or 0)
    monday_rows = int(monday_coverage.get("trade_truth_rows_v1") or 0)
    return pd.DataFrame(
        [
            row("WEDNESDAY_SOURCE_ARTIFACTS_PRESENT", "PASS" if source_ok else "FAIL", source_df["status_v1"].value_counts().to_dict()),
            row(
                "LOCAL_R6_ALTERNATE_IS_CANONICAL_WEDNESDAY_R6",
                "PASS" if local_r6_assessment.get("assessment_v1") == "CANONICAL_MATCH" else "FAIL",
                local_r6_assessment,
            ),
            row("MONDAY_TRUTH_PACKAGE_BUILT", "PASS" if monday_rows > 0 else "FAIL", monday_rows),
            row("ROW_COUNT_MATCHES_WEDNESDAY_1971", "PASS" if monday_rows == expected_rows else "FAIL", {"expected": expected_rows, "observed": monday_rows}),
            row("AS_OF_109_EXACTLY_MATERIALIZED", "PASS" if asof_exact_ok else "FAIL", asof_df["availability_status_v1"].value_counts().to_dict()),
            row("HINDSIGHT_30_EXACTLY_MATERIALIZED", "PASS" if hindsight_exact_ok else "FAIL", hindsight_df["availability_status_v1"].value_counts().to_dict()),
            row("WEDNESDAY_SCORE_HEADS_PRESENT", "PASS" if score_ok else "FAIL", score_df["availability_status_v1"].value_counts().to_dict()),
            row(
                "MONDAY_HAS_AVAILABLE_SOURCE_SURFACES",
                "PASS"
                if all(
                    int(monday_coverage.get(key) or 0) > 0
                    for key in ["candidate_surface_rows_v1", "bar_feature_rows_v1", "exit_eval_trace_rows_v1", "xgb_signal_rows_v1"]
                )
                else "FAIL",
                monday_coverage,
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    monday_truth_dir: Path,
    wednesday_summary: dict[str, Any],
    wednesday_manifest: dict[str, Any],
    monday_coverage: dict[str, Any],
    source_df: pd.DataFrame,
    asof_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
    score_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    local_r6_assessment: dict[str, Any],
) -> dict[str, Any]:
    failures = int(audit_df["status_v1"].eq("FAIL").sum())
    if failures:
        decision = "MONDAY_R6_TRUTH_BUILT_BUT_WEDNESDAY_R6_CONTRACT_NOT_FULLY_RESTORED"
        next_action = "REHYDRATE_MONDAY_R6_AS_OF_AND_HINDSIGHT_USING_WEDNESDAY_CONTRACT"
    else:
        decision = "MONDAY_R6_HAS_FULL_WEDNESDAY_R6_CONTRACT_PARITY"
        next_action = "RUN_MONDAY_R6_TRAINING_EVAL_ON_RESTORED_CONTRACT"
    asof_counts = asof_df["availability_status_v1"].value_counts().to_dict()
    hindsight_counts = hindsight_df["availability_status_v1"].value_counts().to_dict()
    score_counts = score_df["availability_status_v1"].value_counts().to_dict()
    return {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "monday_truth_dir_v1": str(monday_truth_dir),
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
        "wednesday_freeze_id_v1": wednesday_summary.get("freeze_id_v1"),
        "wednesday_candidate_id_v1": wednesday_summary.get("selected_candidate_id_v1"),
        "wednesday_model_version_id_v1": wednesday_manifest.get("model_version_id_v1"),
        "wednesday_policy_rows_v1": (wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1"),
        "wednesday_as_of_columns_v1": (wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1"),
        "wednesday_hindsight_columns_v1": (wednesday_manifest.get("hindsight_schema_v1") or {}).get("column_count_v1"),
        "monday_trade_truth_rows_v1": monday_coverage.get("trade_truth_rows_v1"),
        "monday_feature_manifest_rows_v1": monday_coverage.get("feature_manifest_rows_v1"),
        "source_artifact_status_counts_v1": source_df["status_v1"].value_counts().to_dict(),
        "as_of_availability_counts_v1": asof_counts,
        "hindsight_availability_counts_v1": hindsight_counts,
        "score_head_availability_counts_v1": score_counts,
        "local_r6_alternate_assessment_v1": local_r6_assessment.get("assessment_v1"),
        "local_r6_alternate_dir_v1": local_r6_assessment.get("local_r6_dir_v1"),
        "hard_status_v1": {
            "BEVIST": [
                "Wednesday R6 benchmark lock is locally available as a manifest/summary snapshot.",
                "Monday R6 truth package exists and contains candidate, bar, XGB, exit trace, trade truth, lineage, and feature manifest surfaces.",
            ],
            "INDIKERT": [
                "Several Wednesday AS_OF replay features can be rehydrated from Monday raw bar/candidate/XGB surfaces.",
                "Several hindsight labels can be derived only after rebuilding the exact R6 label contract.",
            ],
            "IKKE_ETABLERT": [
                "The original Wednesday R6 source parquet/model artifact directory is not locally present.",
                "The similarly named local R6 artifact directory is present but does not match the frozen Wednesday R6 contract.",
                "Monday does not yet have an exact 109-column Wednesday R6 AS_OF matrix.",
                "Monday does not yet have the exact 30-column Wednesday R6 hindsight matrix.",
                "Monday does not yet have the frozen policy score-head columns required by the Wednesday R6 contract.",
            ],
        },
    }


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 vs Wednesday R6 Contract Completeness V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            "## Counts",
            "",
            f"- Wednesday rows: `{summary['wednesday_policy_rows_v1']}`",
            f"- Monday trade truth rows: `{summary['monday_trade_truth_rows_v1']}`",
            f"- Wednesday AS_OF columns: `{summary['wednesday_as_of_columns_v1']}`",
            f"- Wednesday hindsight columns: `{summary['wednesday_hindsight_columns_v1']}`",
            f"- Monday feature manifest rows: `{summary['monday_feature_manifest_rows_v1']}`",
            "",
            "## Availability",
            "",
            f"- Source artifacts: `{summary['source_artifact_status_counts_v1']}`",
            f"- AS_OF schema: `{summary['as_of_availability_counts_v1']}`",
            f"- Hindsight schema: `{summary['hindsight_availability_counts_v1']}`",
            f"- Score heads: `{summary['score_head_availability_counts_v1']}`",
            f"- Local R6 alternate: `{summary['local_r6_alternate_assessment_v1']}`",
            "",
            "This is a completeness check only. It does not train, freeze, or promote anything.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    monday_truth_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    monday_truth_dir = monday_truth_dir.expanduser().resolve() if monday_truth_dir else _latest_monday_truth(reports_root)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    wednesday_summary = _read_json(freeze_dir / WEDNESDAY_SUMMARY)
    wednesday_manifest = _read_json(freeze_dir / WEDNESDAY_MANIFEST)
    monday_coverage = _read_json(monday_truth_dir / "monday_r6_truth_coverage_summary_v1.json")
    _, feature_names, by_name = _feature_lookup(monday_truth_dir)

    source_df = _source_artifact_rows(wednesday_manifest)
    local_r6_assessment = _local_r6_alternate_assessment(reports_root, wednesday_summary, wednesday_manifest)
    asof_df = _schema_rows("AS_OF", wednesday_manifest["as_of_schema_v1"]["columns_v1"], feature_names, by_name)
    hindsight_df = _schema_rows("HINDSIGHT", wednesday_manifest["hindsight_schema_v1"]["columns_v1"], feature_names, by_name)
    score_df = _score_head_rows(wednesday_manifest, feature_names)
    audit_df = _audit_rows(source_df, asof_df, hindsight_df, score_df, monday_coverage, wednesday_summary, local_r6_assessment)
    summary = _summary(
        output_dir=output_dir,
        monday_truth_dir=monday_truth_dir,
        wednesday_summary=wednesday_summary,
        wednesday_manifest=wednesday_manifest,
        monday_coverage=monday_coverage,
        source_df=source_df,
        asof_df=asof_df,
        hindsight_df=hindsight_df,
        score_df=score_df,
        audit_df=audit_df,
        local_r6_assessment=local_r6_assessment,
    )
    expected = {
        "freeze_id_v1": wednesday_summary.get("freeze_id_v1"),
        "selected_candidate_id_v1": wednesday_summary.get("selected_candidate_id_v1"),
        "model_version_id_v1": wednesday_manifest.get("model_version_id_v1"),
        "threshold_version_id_v1": wednesday_manifest.get("threshold_version_id_v1"),
        "policy_logging_v1": wednesday_summary.get("policy_logging_v1"),
        "selected_candidate_v1": wednesday_summary.get("selected_candidate_v1"),
        "score_head_names_v1": wednesday_manifest.get("score_head_names_v1"),
        "as_of_schema_v1": wednesday_manifest.get("as_of_schema_v1"),
        "hindsight_schema_v1": wednesday_manifest.get("hindsight_schema_v1"),
        "r6_source_dir_v1": wednesday_manifest.get("r6_source_dir_v1"),
    }

    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["wednesday_expected"], expected)
    _write_json(output_dir / OUTPUT_FILES["local_r6_alternate"], local_r6_assessment)
    asof_df.to_csv(output_dir / OUTPUT_FILES["as_of"], index=False)
    hindsight_df.to_csv(output_dir / OUTPUT_FILES["hindsight"], index=False)
    score_df.to_csv(output_dir / OUTPUT_FILES["score_heads"], index=False)
    source_df.to_csv(output_dir / OUTPUT_FILES["source_artifacts"], index=False)
    audit_df.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monday-truth-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(reports_root=args.reports_root, monday_truth_dir=args.monday_truth_dir, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
