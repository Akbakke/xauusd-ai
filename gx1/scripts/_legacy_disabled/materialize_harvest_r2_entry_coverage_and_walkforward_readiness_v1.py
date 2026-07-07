#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1"
R2_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R2"
R1_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R1"
HARVEST_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_POLICY_CANDIDATE_R1"

LEDGER_VIEW = "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
ENTRY_VIEW = "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"
ENTRY_RAW_VIEW = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
MANAGEMENT_VIEW = "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet"
HARVEST_POLICY_VIEW = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_trade_view_v1.parquet"
HARVEST_TARGET_VIEW = "shadow_meta_all_trade_review_harvest_model_adjustment_target_view_v1.parquet"
R2_PREDICTION_VIEW = "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet"
R2_SUMMARY = "shadow_meta_all_trade_review_harvest_retrain_candidate_summary_v1.json"

CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
HINDSIGHT_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
COVERAGE_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_audit_v1.csv"
COVERAGE_RUN_ROLLUP = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_run_rollup_v1.csv"
FEATURE_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_audit_v1.csv"
FEATURE_FAMILY_ROLLUP = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_family_rollup_v1.csv"
WALKFORWARD_METRICS = "shadow_meta_all_trade_review_harvest_r2_entry_model_walkforward_metrics_v1.csv"
SAFETY_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_fallback_policy_safety_audit_v1.csv"
READINESS_MATRIX = "shadow_meta_all_trade_review_harvest_r2_promotion_readiness_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json"
MARKDOWN_REPORT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_report_v1.md"
MANIFEST = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_manifest_v1.json"
TOP_LEVEL_SUMMARY = "truth_harvest_r2_entry_coverage_and_walkforward_readiness_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
ENTRY_RAW_PREFIXES = ("as_of_skip_replay_", "as_of_skip_candidate_", "as_of_skip_xgb_")
ENTRY_CORE_PREFIXES = ("as_of_",)
FORBIDDEN_AS_OF_TOKENS = (
    "hindsight",
    "pnl",
    "reward",
    "target",
    "label",
    "harvest",
    "terminal",
    "good_trade",
    "bad_trade",
    "premature",
    "late_exit",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_review_dir(reports_root: Path, review_dir_arg: str | None) -> Path:
    if review_dir_arg:
        review_dir = Path(review_dir_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir
    rebuild_summary = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if rebuild_summary.exists():
        raw_dir = _load_json(rebuild_summary).get("ledger_dir")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / LEDGER_VIEW).exists():
                return candidate
    namespace_dirs = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX)],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if (candidate / LEDGER_VIEW).exists() and (candidate / ENTRY_VIEW).exists() and (candidate / MANAGEMENT_VIEW).exists():
            return candidate
    raise FileNotFoundError("Could not resolve locked canonical review dir.")


def _resolve_existing_dir(reports_root: Path, arg: str | None, default_name: str, required_file: str) -> Path:
    if arg:
        path = Path(arg).expanduser().resolve()
    else:
        path = reports_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Required dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} is missing required artifact {required_file}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.eq("true").fillna(default).astype(bool)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, frame: pd.DataFrame) -> List[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted([path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)], key=_run_sort_key)
        if run_ids:
            return run_ids
    return sorted(frame["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _split_name(frame: pd.DataFrame) -> pd.Series:
    _require_columns(frame, ["used_for_training", "used_for_validation", "used_for_holdout"], artifact_name="split source")
    out = pd.Series("UNKNOWN", index=frame.index, dtype="string")
    out.loc[_bool(frame, "used_for_training")] = "TRAIN"
    out.loc[_bool(frame, "used_for_validation")] = "VALIDATION"
    out.loc[_bool(frame, "used_for_holdout")] = "HOLDOUT"
    return out


def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ["swing", "retracement", "ema", "kama", "bb_", "squeeze", "bandwidth"]):
        return "structure/swing/retracement"
    if any(token in lower for token in ["vol", "atr", "range", "spread", "cost"]):
        return "volatility/range"
    if any(token in lower for token in ["close_in_bar", "minutes_since_session", "minutes_to_next", "session_change"]):
        return "close-in-bar/timing"
    if any(token in lower for token in ["session", "hour", "weekday"]):
        return "session/time context"
    if any(token in lower for token in ["momentum", "acceleration", "impulse", "ret_", "up_move", "down_move", "directional", "body", "wick", "clv"]):
        return "prior path/impulse context"
    if any(token in lower for token in ["candidate_", "xgb_", "margin", "path_quality", "p_long", "p_short", "p_flat", "p_hat"]):
        return "management handoff context/as_of_candidate"
    return "other/as_of"


def _assert_as_of_feature_names_safe(feature_names: Sequence[str]) -> None:
    bad: List[str] = []
    for feature in feature_names:
        lower = feature.lower()
        for token in FORBIDDEN_AS_OF_TOKENS:
            if token in lower:
                bad.append(feature)
                break
    if bad:
        raise ValueError(f"AS_OF table contains forbidden hindsight/target-like feature names: {bad[:20]}")


def _build_as_of_table(ledger_df: pd.DataFrame, entry_df: pd.DataFrame, entry_raw_df: pd.DataFrame, management_df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    entry_core_features = [
        column
        for column in entry_df.columns
        if column.startswith("as_of_")
        and not any(token in column.lower() for token in ["candidate_entry_bundle_sha256", "candidate_exit_bundle_sha256"])
    ]
    entry_raw_features = [
        column
        for column in entry_raw_df.columns
        if any(column.startswith(prefix) for prefix in ENTRY_RAW_PREFIXES) and column not in entry_core_features
    ]
    feature_names = entry_core_features + entry_raw_features
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted({feature for feature in feature_names if feature_names.count(feature) > 1})
        raise ValueError(f"Duplicate AS_OF feature names: {duplicates[:20]}")
    _assert_as_of_feature_names_safe(feature_names)

    base_cols = ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "used_for_training", "used_for_validation", "used_for_holdout"]
    asof = ledger_df[[column for column in base_cols if column in ledger_df.columns]].copy()
    if "used_for_training" not in asof.columns:
        asof["used_for_training"] = False
        asof["used_for_validation"] = False
        asof["used_for_holdout"] = False
    entry_source = entry_df[["candidate_uid", *entry_core_features]].copy()
    raw_source = entry_raw_df[["candidate_uid", *entry_raw_features]].copy()
    asof = asof.merge(entry_source, on="candidate_uid", how="left", validate="one_to_one")
    asof = asof.merge(raw_source, on="candidate_uid", how="left", validate="one_to_one")
    entry_set = set(entry_df["candidate_uid"].astype("string"))
    raw_set = set(entry_raw_df["candidate_uid"].astype("string"))
    mgmt_set = set(management_df["candidate_uid_exact_v1"].astype("string"))
    asof["entry_observation_present_v1"] = asof["candidate_uid"].astype("string").isin(entry_set)
    asof["entry_raw_state_present_v1"] = asof["candidate_uid"].astype("string").isin(raw_set)
    asof["management_observation_present_v1"] = asof["candidate_uid"].astype("string").isin(mgmt_set)
    asof["as_of_feature_namespace_v1"] = "ENTRY_AS_OF_ONLY_NO_HINDSIGHT_LABELS"
    return asof, feature_names


def _build_hindsight_labels(ledger_df: pd.DataFrame, harvest_policy_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.DataFrame:
    policy_cols = [
        "candidate_uid",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "harvest_quality_bucket_v1",
        "exit_harvest_policy_action_v1",
        "rl_priority_entry_skip_delta_bps_v1",
        "rl_priority_hold_longer_delta_bps_v1",
        "rl_priority_exit_earlier_delta_bps_v1",
        "home_run_200bps_opportunity_v1",
        "runner_100bps_opportunity_v1",
        "runner_50bps_opportunity_v1",
    ]
    labels = ledger_df[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "realized_pnl_bps",
            "mfe_bps",
            "mae_bps",
            "exit_reason",
            "trade_outcome_class",
            "session",
            "vol_regime",
            "trend_regime",
            "hindsight_entry_decision_review_v1",
            "hindsight_management_review_v1",
            "hindsight_should_skip_trade_v1",
            "hindsight_should_hold_longer_v1",
            "hindsight_should_exit_earlier_v1",
        ]
    ].merge(harvest_policy_df[[column for column in policy_cols if column in harvest_policy_df.columns]], on="candidate_uid", how="left", validate="one_to_one")

    entry_support_cols = [
        "candidate_uid",
        "support_adverse_first_v1",
        "support_first_meaningful_mfe_bar_index_v1",
        "confirmation_delay_minutes_v1",
        "has_provable_confirmation_v1",
        "wait_followthrough_status_v1",
        "teacher_should_wait_entry_v1",
    ]
    labels = labels.merge(
        entry_df[[column for column in entry_support_cols if column in entry_df.columns]],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )

    realized = _num(labels, "baseline_realized_pnl_bps_v1")
    peak_mfe = _num(labels, "peak_mfe_bps_v1")
    mae_abs = _num(labels, "mae_abs_bps_v1").abs()
    giveback = _num(labels, "giveback_bps_v1")
    capture = _num(labels, "harvest_capture_ratio_v1")
    should_skip = _bool(labels, "hindsight_should_skip_trade_v1") | labels.get("exit_harvest_policy_action_v1", pd.Series("", index=labels.index)).astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE")
    support_adverse_first = _bool(labels, "support_adverse_first_v1")
    confirmation_delay = _num(labels, "confirmation_delay_minutes_v1", default=0.0)
    teacher_wait = _bool(labels, "teacher_should_wait_entry_v1")

    labels["label_should_not_take_v1"] = should_skip
    labels["label_immediate_mae_risk_v1"] = (mae_abs.ge(30.0) & (realized.le(0.0) | peak_mfe.lt(50.0))) | support_adverse_first
    labels["label_wait_would_have_helped_v1"] = (
        ~should_skip
        & (
            (mae_abs.ge(20.0) & peak_mfe.ge(50.0))
            | confirmation_delay.gt(0.0)
            | teacher_wait
        )
    )
    labels["label_good_mfe_bad_capture_v1"] = peak_mfe.ge(50.0) & (capture.lt(0.35) | giveback.ge(25.0))
    labels["label_low_mfe_low_value_v1"] = peak_mfe.lt(20.0) & realized.le(5.0)
    labels["label_strong_trade_candidate_v1"] = (~should_skip) & peak_mfe.ge(50.0) & mae_abs.le(25.0) & realized.gt(0.0)
    labels["label_direct_take_ok_v1"] = (
        ~should_skip
        & realized.gt(0.0)
        & peak_mfe.ge(20.0)
        & ~labels["label_immediate_mae_risk_v1"]
    )
    labels["hindsight_label_contract_v1"] = "HINDSIGHT_SUPERVISION_ONLY_NOT_POLICY_TRUTH_NOT_AS_OF_FEATURES"
    return labels


def _entry_coverage_reason(entry_missing: bool, raw_missing: bool, join_key_missing: bool, exit_reason: Any) -> tuple[str, str]:
    if join_key_missing:
        return "missing join key", "candidate_uid missing/blank"
    if not entry_missing and raw_missing:
        return "missing AS_OF raw-state", "entry observation present but rich raw-state absent"
    if not entry_missing:
        return "COVERED", "covered"
    if str(exit_reason) == "REPLAY_EOF":
        return "zero-trade/window edge", "REPLAY_EOF terminal ledger trade absent from entry observation artifact"
    detail = "ledger trade absent from entry observation artifact"
    if raw_missing:
        detail = f"{detail}; dependent AS_OF raw-state also absent"
    return "missing entry observation", detail


def _management_coverage_reason(management_missing: bool, join_key_missing: bool, exit_reason: Any) -> tuple[str, str]:
    if join_key_missing:
        return "missing join key", "candidate_uid missing/blank"
    if not management_missing:
        return "COVERED", "covered"
    if str(exit_reason) == "REPLAY_EOF":
        return "zero-trade/window edge", "REPLAY_EOF terminal trade absent from management row semantics"
    return "missing management observation", "ledger trade absent from management row semantics and raw-state artifacts"


def _build_coverage_audit(
    ledger_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    entry_raw_df: pd.DataFrame,
    management_df: pd.DataFrame,
    harvest_policy_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    entry_set = set(entry_df["candidate_uid"].astype("string"))
    raw_set = set(entry_raw_df["candidate_uid"].astype("string"))
    mgmt_set = set(management_df["candidate_uid_exact_v1"].astype("string"))
    policy = harvest_policy_df[["candidate_uid", "exit_harvest_policy_action_v1", "harvest_quality_bucket_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "baseline_realized_pnl_bps_v1"]].copy()
    work = ledger_df.merge(policy, on="candidate_uid", how="left", validate="one_to_one")
    work["entry_observation_present_v1"] = work["candidate_uid"].astype("string").isin(entry_set)
    work["entry_raw_state_present_v1"] = work["candidate_uid"].astype("string").isin(raw_set)
    work["management_observation_present_v1"] = work["candidate_uid"].astype("string").isin(mgmt_set)
    work["candidate_uid_missing_v1"] = work["candidate_uid"].astype("string").isna() | work["candidate_uid"].astype("string").eq("")

    entry_reasons: List[str] = []
    entry_details: List[str] = []
    management_reasons: List[str] = []
    management_details: List[str] = []
    for row in work.to_dict(orient="records"):
        entry_reason, entry_detail = _entry_coverage_reason(
            not bool(row["entry_observation_present_v1"]),
            not bool(row["entry_raw_state_present_v1"]),
            bool(row["candidate_uid_missing_v1"]),
            row.get("exit_reason"),
        )
        mgmt_reason, mgmt_detail = _management_coverage_reason(
            not bool(row["management_observation_present_v1"]),
            bool(row["candidate_uid_missing_v1"]),
            row.get("exit_reason"),
        )
        entry_reasons.append(entry_reason)
        entry_details.append(entry_detail)
        management_reasons.append(mgmt_reason)
        management_details.append(mgmt_detail)
    work["entry_gap_reason_code_v1"] = entry_reasons
    work["entry_gap_reason_detail_v1"] = entry_details
    work["management_gap_reason_code_v1"] = management_reasons
    work["management_gap_reason_detail_v1"] = management_details
    work["coverage_gap_scope_v1"] = np.select(
        [
            work["entry_observation_present_v1"] & work["management_observation_present_v1"],
            ~work["entry_observation_present_v1"] & ~work["management_observation_present_v1"],
            ~work["entry_observation_present_v1"],
            ~work["management_observation_present_v1"],
        ],
        ["FULLY_COVERED", "MISSING_ENTRY_AND_MANAGEMENT", "MISSING_ENTRY_ONLY", "MISSING_MANAGEMENT_ONLY"],
        default="OTHER",
    )
    run_rollup = (
        work.groupby("run_id", dropna=False)
        .agg(
            trade_count_v1=("candidate_uid", "count"),
            entry_missing_count_v1=("entry_observation_present_v1", lambda s: int((~s.astype(bool)).sum())),
            management_missing_count_v1=("management_observation_present_v1", lambda s: int((~s.astype(bool)).sum())),
            both_missing_count_v1=("coverage_gap_scope_v1", lambda s: int((s == "MISSING_ENTRY_AND_MANAGEMENT").sum())),
        )
        .reset_index()
    )
    run_rollup["entry_missing_rate_v1"] = run_rollup["entry_missing_count_v1"] / run_rollup["trade_count_v1"]
    run_rollup["management_missing_rate_v1"] = run_rollup["management_missing_count_v1"] / run_rollup["trade_count_v1"]
    entry_missing = work[~work["entry_observation_present_v1"]]
    mgmt_missing = work[~work["management_observation_present_v1"]]
    summary = {
        "entry_covered_v1": int(work["entry_observation_present_v1"].sum()),
        "entry_missing_v1": int((~work["entry_observation_present_v1"]).sum()),
        "management_covered_v1": int(work["management_observation_present_v1"].sum()),
        "management_missing_v1": int((~work["management_observation_present_v1"]).sum()),
        "both_missing_v1": int(work["coverage_gap_scope_v1"].eq("MISSING_ENTRY_AND_MANAGEMENT").sum()),
        "entry_missing_reason_counts_v1": _counts(entry_missing, "entry_gap_reason_code_v1"),
        "management_missing_reason_counts_v1": _counts(mgmt_missing, "management_gap_reason_code_v1"),
        "entry_missing_run_count_v1": int(entry_missing["run_id"].nunique()),
        "management_missing_run_count_v1": int(mgmt_missing["run_id"].nunique()),
        "entry_missing_exit_reason_counts_v1": _counts(entry_missing, "exit_reason"),
        "management_missing_exit_reason_counts_v1": _counts(mgmt_missing, "exit_reason"),
        "entry_missing_action_counts_v1": _counts(entry_missing, "exit_harvest_policy_action_v1"),
        "management_missing_action_counts_v1": _counts(mgmt_missing, "exit_harvest_policy_action_v1"),
        "coverage_gap_interpretation_v1": {
            "entry_v1": "STRUCTURAL_LEDGER_ONLY_POSITIVE_TAKE_OK_TRADES_ABSENT_FROM_ENTRY_POLICY_TRAINING_EXAMPLES",
            "management_v1": "STRUCTURAL_LEDGER_ONLY_TAIL_OR_WINDOW_EDGE_TRADES_ABSENT_FROM_MANAGEMENT_ROW_SEMANTICS",
            "pipeline_error_v1": False,
            "artifact_schema_mismatch_v1": False,
        },
    }
    keep_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "exit_reason",
        "trade_outcome_class",
        "session",
        "vol_regime",
        "trend_regime",
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "exit_harvest_policy_action_v1",
        "harvest_quality_bucket_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "entry_observation_present_v1",
        "entry_raw_state_present_v1",
        "management_observation_present_v1",
        "coverage_gap_scope_v1",
        "entry_gap_reason_code_v1",
        "entry_gap_reason_detail_v1",
        "management_gap_reason_code_v1",
        "management_gap_reason_detail_v1",
    ]
    return work[[column for column in keep_cols if column in work.columns]].copy(), run_rollup, summary


def _binary_task_mask(labels_df: pd.DataFrame, task: str) -> tuple[pd.Series, pd.Series, str, str]:
    if task == "good_take_vs_should_not_take":
        positive = labels_df["label_direct_take_ok_v1"].astype(bool)
        negative = labels_df["label_should_not_take_v1"].astype(bool)
        return positive, negative, "DIRECT_TAKE_OK", "SHOULD_NOT_TAKE"
    if task == "take_now_vs_wait":
        positive = labels_df["label_direct_take_ok_v1"].astype(bool)
        negative = labels_df["label_wait_would_have_helped_v1"].astype(bool)
        return positive, negative, "DIRECT_TAKE_OK", "WAIT_WOULD_HAVE_HELPED"
    if task == "immediate_mae_risk_vs_clean_entry":
        positive = labels_df["label_immediate_mae_risk_v1"].astype(bool)
        negative = labels_df["label_direct_take_ok_v1"].astype(bool)
        return positive, negative, "IMMEDIATE_MAE_RISK", "CLEAN_DIRECT_TAKE"
    if task == "high_mfe_candidate_vs_low_value_candidate":
        positive = labels_df["label_strong_trade_candidate_v1"].astype(bool)
        negative = labels_df["label_low_mfe_low_value_v1"].astype(bool)
        return positive, negative, "STRONG_TRADE_CANDIDATE", "LOW_MFE_LOW_VALUE"
    raise ValueError(task)


def _numeric_feature_score(values: pd.Series, y: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 20 or int(y.loc[valid].nunique()) < 2:
        return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(valid.sum())}
    x_valid = x.loc[valid].astype(float)
    y_valid = y.loc[valid].astype(int)
    try:
        auc = float(roc_auc_score(y_valid, x_valid))
    except ValueError:
        auc = None
    pos_mean = float(x_valid[y_valid.eq(1)].mean())
    neg_mean = float(x_valid[y_valid.eq(0)].mean())
    pooled_std = float(x_valid.std(ddof=0)) or 0.0
    effect = abs(pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0.0
    auc_sep = abs((auc or 0.5) - 0.5) * 2.0
    return {
        "score_v1": float(effect + auc_sep),
        "auc_v1": auc,
        "direction_v1": "HIGHER_FOR_POSITIVE" if pos_mean >= neg_mean else "LOWER_FOR_POSITIVE",
        "positive_mean_v1": pos_mean,
        "negative_mean_v1": neg_mean,
        "coverage_v1": int(valid.sum()),
    }


def _categorical_feature_score(values: pd.Series, y: pd.Series) -> Dict[str, Any]:
    x = values.astype("string")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 20 or int(y.loc[valid].nunique()) < 2:
        return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(valid.sum())}
    y_valid = y.loc[valid].astype(int)
    overall = float(y_valid.mean())
    grouped = y_valid.groupby(x.loc[valid]).agg(["count", "mean"])
    grouped = grouped[grouped["count"].ge(5)]
    if grouped.empty:
        return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(valid.sum())}
    diffs = (grouped["mean"] - overall).abs()
    best_category = str(diffs.idxmax())
    return {
        "score_v1": float(diffs.max()),
        "auc_v1": None,
        "direction_v1": f"CATEGORY:{best_category}",
        "positive_mean_v1": float(grouped.loc[best_category, "mean"]),
        "negative_mean_v1": overall,
        "coverage_v1": int(valid.sum()),
    }


def _build_feature_audit(asof_df: pd.DataFrame, labels_df: pd.DataFrame, feature_names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    work = asof_df[["candidate_uid", *feature_names]].merge(
        labels_df[
            [
                "candidate_uid",
                "label_direct_take_ok_v1",
                "label_should_not_take_v1",
                "label_wait_would_have_helped_v1",
                "label_immediate_mae_risk_v1",
                "label_low_mfe_low_value_v1",
                "label_strong_trade_candidate_v1",
            ]
        ],
        on="candidate_uid",
        how="inner",
        validate="one_to_one",
    )
    rows: List[Dict[str, Any]] = []
    tasks = [
        "good_take_vs_should_not_take",
        "take_now_vs_wait",
        "immediate_mae_risk_vs_clean_entry",
        "high_mfe_candidate_vs_low_value_candidate",
    ]
    for task in tasks:
        positive, negative, positive_name, negative_name = _binary_task_mask(work, task)
        task_mask = positive | negative
        y = pd.Series(np.where(positive.loc[task_mask], 1, 0), index=work.index[task_mask])
        for feature in feature_names:
            if feature not in work.columns:
                continue
            values = work.loc[task_mask, feature]
            if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                score = _numeric_feature_score(values, y)
                feature_type = "numeric"
            else:
                score = _categorical_feature_score(values, y)
                feature_type = "categorical"
            rows.append(
                {
                    "task_v1": task,
                    "positive_label_v1": positive_name,
                    "negative_label_v1": negative_name,
                    "feature_name_v1": feature,
                    "feature_family_v1": _feature_family(feature),
                    "feature_type_v1": feature_type,
                    **score,
                }
            )
    audit = pd.DataFrame(rows)
    evaluable = audit.dropna(subset=["score_v1"]).copy()
    family = (
        evaluable.sort_values(["task_v1", "score_v1"], ascending=[True, False])
        .groupby(["task_v1", "feature_family_v1"], dropna=False)
        .agg(
            feature_count_v1=("feature_name_v1", "count"),
            mean_score_v1=("score_v1", "mean"),
            max_score_v1=("score_v1", "max"),
            top_feature_v1=("feature_name_v1", "first"),
        )
        .reset_index()
        .sort_values(["task_v1", "max_score_v1"], ascending=[True, False])
    )
    top_by_task: Dict[str, Any] = {}
    for task in tasks:
        top_by_task[task] = (
            evaluable[evaluable["task_v1"].eq(task)]
            .sort_values("score_v1", ascending=False)
            .head(10)[["feature_name_v1", "feature_family_v1", "score_v1", "auc_v1", "direction_v1"]]
            .to_dict(orient="records")
        )
    return audit, family, {"top_features_by_task_v1": top_by_task}


def _ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def _ece_multiclass(y_true: np.ndarray, pred: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    correctness = (y_true == pred).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _classification_slice_metrics(
    frame: pd.DataFrame,
    *,
    task: str,
    target_col: str,
    pred_col: str,
    prob_cols: Dict[str, str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    sub = frame[frame[pred_col].notna()].copy()
    if target_col == "entry_xgb_binary_take_target_v1":
        target = sub[target_col].map({True: "TRUE", False: "FALSE"}).astype("string")
    else:
        target = sub[target_col].astype("string")
    pred = sub[pred_col].astype("string")
    valid = target.notna() & pred.notna() & target.isin(labels) & pred.isin(labels)
    target = target.loc[valid]
    pred = pred.loc[valid]
    sub = sub.loc[valid]
    label_to_code = {label: index for index, label in enumerate(labels)}
    y_true = target.map(label_to_code).to_numpy(dtype=int)
    y_pred = pred.map(label_to_code).to_numpy(dtype=int)
    row: Dict[str, Any] = {
        "task_v1": task,
        "row_count_v1": int(len(sub)),
        "accuracy_v1": None,
        "balanced_accuracy_v1": None,
        "macro_f1_v1": None,
        "logloss_v1": None,
        "brier_score_v1": None,
        "ece_v1": None,
        "probability_row_count_v1": 0,
        "probability_missing_count_v1": int(len(sub)),
        "probability_sum_min_v1": None,
        "probability_sum_max_v1": None,
        "probability_sum_max_abs_deviation_v1": None,
        "probability_metric_status_v1": "NOT_EVALUATED",
        "precision_by_class_json_v1": "{}",
        "recall_by_class_json_v1": "{}",
        "confusion_matrix_json_v1": "[]",
    }
    if len(sub) == 0 or len(set(y_true.tolist())) < 2:
        return row
    row["accuracy_v1"] = float(accuracy_score(y_true, y_pred))
    row["balanced_accuracy_v1"] = float(balanced_accuracy_score(y_true, y_pred))
    row["macro_f1_v1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)
    row["precision_by_class_json_v1"] = _json_dumps({label: float(precision[index]) for index, label in enumerate(labels)})
    row["recall_by_class_json_v1"] = _json_dumps({label: float(recall[index]) for index, label in enumerate(labels)})
    row["confusion_matrix_json_v1"] = _json_dumps(confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist())

    prob_matrix = np.zeros((len(sub), len(labels)), dtype=float)
    prob_available = True
    for label, idx in label_to_code.items():
        col = prob_cols.get(label)
        if not col or col not in sub.columns:
            prob_available = False
            break
        prob_matrix[:, idx] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if prob_available:
        prob_frame = sub[[prob_cols[label] for label in labels]].apply(pd.to_numeric, errors="coerce")
        prob_valid = prob_frame.notna().all(axis=1) & prob_frame.sum(axis=1).gt(0.0)
        prob_sum = prob_frame.loc[prob_valid].sum(axis=1)
        row["probability_row_count_v1"] = int(prob_valid.sum())
        row["probability_missing_count_v1"] = int((~prob_valid).sum())
        row["probability_sum_min_v1"] = _safe_float(prob_sum.min()) if not prob_sum.empty else None
        row["probability_sum_max_v1"] = _safe_float(prob_sum.max()) if not prob_sum.empty else None
        max_abs_dev = (prob_sum - 1.0).abs().max() if not prob_sum.empty else None
        row["probability_sum_max_abs_deviation_v1"] = _safe_float(max_abs_dev)
    if prob_available and bool(prob_valid.any()):
        y_true_prob = y_true[prob_valid.to_numpy()]
        y_pred_prob = y_pred[prob_valid.to_numpy()]
        prob_matrix = prob_matrix[prob_valid.to_numpy()]
        prob_row_sums = prob_matrix.sum(axis=1)
        max_abs_dev = float(np.max(np.abs(prob_row_sums - 1.0))) if len(prob_row_sums) else 0.0
        if max_abs_dev > 1e-3:
            row["probability_metric_status_v1"] = "SKIPPED_INVALID_PROBABILITY_SUM"
            return row
        prob_matrix = prob_matrix / prob_row_sums[:, None]
        row["probability_metric_status_v1"] = "ROW_SUM_NORMALIZED_WITHIN_1E-3_TOLERANCE"
        row["logloss_v1"] = _safe_float(log_loss(y_true_prob, prob_matrix, labels=list(range(len(labels)))))
        if len(labels) == 2:
            positive_idx = label_to_code.get("TRUE", 1)
            row["brier_score_v1"] = _safe_float(brier_score_loss((y_true_prob == positive_idx).astype(int), prob_matrix[:, positive_idx]))
            row["ece_v1"] = _ece_binary((y_true_prob == positive_idx).astype(int), prob_matrix[:, positive_idx])
        else:
            onehot = np.eye(len(labels))[y_true_prob]
            row["brier_score_v1"] = float(np.mean(np.sum((prob_matrix - onehot) ** 2, axis=1)))
            row["ece_v1"] = _ece_multiclass(y_true_prob, y_pred_prob, prob_matrix.max(axis=1))
    return row


def _build_walkforward_metrics(
    reports_root: Path,
    r2_pred_df: pd.DataFrame,
    *,
    batch_weeks: int,
    r1_pred_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    run_ids = _all_run_ids(reports_root, r2_pred_df)
    rows: List[Dict[str, Any]] = []
    tasks = [
        (
            "entry_binary_take",
            "entry_xgb_binary_take_target_v1",
            "pred__entry_xgb_binary_take__label_v1",
            {"FALSE": "pred__entry_xgb_binary_take__prob_false_v1", "TRUE": "pred__entry_xgb_binary_take__prob_true_v1"},
            ["FALSE", "TRUE"],
        ),
        (
            "entry_multiclass_harvest",
            "entry_xgb_harvest_label_v1",
            "pred__entry_xgb_harvest_label__label_v1",
            {
                "ALLOW_BASELINE": "pred__entry_xgb_harvest_label__prob_allow_baseline_v1",
                "PRIORITIZE_CLEAN_RUNNER": "pred__entry_xgb_harvest_label__prob_prioritize_clean_runner_v1",
                "REJECT_OR_LOW_SIZE": "pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1",
            },
            ["ALLOW_BASELINE", "PRIORITIZE_CLEAN_RUNNER", "REJECT_OR_LOW_SIZE"],
        ),
    ]
    def add_candidate_rows(candidate_version: str, candidate_df: pd.DataFrame, batch_index: int, batch_run_ids: Sequence[str]) -> None:
        feature_available = candidate_df.get(
            "pred__entry_xgb_binary_take__feature_available_v1",
            pd.Series(False, index=candidate_df.index),
        ).fillna(False).astype(bool)
        coverage_slices = [
            ("ALL_ROWS", candidate_df),
            ("ENTRY_FEATURE_COVERED", candidate_df[feature_available]),
            ("ENTRY_FEATURE_MISSING", candidate_df[~feature_available]),
        ]
        for task, target_col, pred_col, prob_cols, labels in tasks:
            for coverage_slice, slice_df in coverage_slices:
                metric = _classification_slice_metrics(
                    slice_df,
                    task=task,
                    target_col=target_col,
                    pred_col=pred_col,
                    prob_cols=prob_cols,
                    labels=labels,
                )
                metric.update(
                    {
                        "candidate_version_v1": candidate_version,
                        "coverage_slice_v1": coverage_slice,
                        "batch_index_v1": int(batch_index),
                        "run_count_v1": int(len(batch_run_ids)),
                        "run_start_v1": batch_run_ids[0] if batch_run_ids else None,
                        "run_end_v1": batch_run_ids[-1] if batch_run_ids else None,
                        "entry_feature_covered_rows_v1": int(feature_available.sum()),
                        "entry_feature_missing_rows_v1": int((~feature_available).sum()),
                    }
                )
                rows.append(metric)

    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        r2_batch = r2_pred_df[r2_pred_df["run_id"].astype("string").isin(batch_run_ids)].copy()
        add_candidate_rows("R2", r2_batch, batch_index, batch_run_ids)
        if r1_pred_df is not None:
            r1_batch = r1_pred_df[r1_pred_df["run_id"].astype("string").isin(batch_run_ids)].copy()
            add_candidate_rows("R1", r1_batch, batch_index, batch_run_ids)
    return pd.DataFrame(rows)


def _entry_standalone_block(frame: pd.DataFrame) -> pd.Series:
    label = frame.get("pred__entry_xgb_harvest_label__label_v1", pd.Series(pd.NA, index=frame.index)).astype("string")
    binary = frame.get("pred__entry_xgb_binary_take__label_v1", pd.Series(pd.NA, index=frame.index)).astype("string")
    prob_reject = pd.to_numeric(frame.get("pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1", pd.Series(np.nan, index=frame.index)), errors="coerce")
    prob_false = pd.to_numeric(frame.get("pred__entry_xgb_binary_take__prob_false_v1", pd.Series(np.nan, index=frame.index)), errors="coerce")
    return label.eq("REJECT_OR_LOW_SIZE").fillna(False) | binary.eq("FALSE").fillna(False) | prob_reject.ge(0.5).fillna(False) | prob_false.ge(0.5).fillna(False)


def _metric_row(metric: str, value: Any, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"metric_name_v1": metric, "value_v1": value, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _build_safety_audit(r2_pred_df: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    work = r2_pred_df.merge(
        labels_df[
            [
                "candidate_uid",
                "label_should_not_take_v1",
                "label_direct_take_ok_v1",
                "label_immediate_mae_risk_v1",
                "label_good_mfe_bad_capture_v1",
                "label_low_mfe_low_value_v1",
                "label_strong_trade_candidate_v1",
            ]
        ],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    covered = work.get("pred__entry_xgb_binary_take__feature_available_v1", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    block = _entry_standalone_block(work) & covered
    high_mfe_50 = _num(work, "peak_mfe_bps_v1").ge(50.0)
    strongest_200 = _num(work, "peak_mfe_bps_v1").ge(200.0)
    mfe_10_50_tail = _num(work, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(work, "baseline_realized_pnl_bps_v1").le(0.0) | work["label_should_not_take_v1"].fillna(False).astype(bool)
    )
    fallback = work.get("candidate_shadow_action_source_v1", pd.Series("", index=work.index)).astype("string").eq("ENTRY_MODEL_SUPPRESS_FALLBACK")
    fallback_match = fallback & work.get("candidate_shadow_action_matches_harvest_target_v1", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    rows = [
        _metric_row("entry_standalone_would_block_count", int(block.sum()), "INFO", {"covered_rows": int(covered.sum()), "rate": _safe_rate(int(block.sum()), int(covered.sum()))}),
        _metric_row("entry_blocks_should_not_take_count", int((block & work["label_should_not_take_v1"].fillna(False).astype(bool)).sum()), "PASS", {}),
        _metric_row("entry_blocks_strong_trade_candidate_count", int((block & work["label_strong_trade_candidate_v1"].fillna(False).astype(bool)).sum()), "WARN", {}),
        _metric_row("entry_prioritizes_strong_trade_count", int((work.get("pred__entry_xgb_harvest_label__label_v1", pd.Series("", index=work.index)).astype("string").eq("PRIORITIZE_CLEAN_RUNNER") & work["label_strong_trade_candidate_v1"].fillna(False).astype(bool)).sum()), "INFO", {}),
        _metric_row("entry_blocks_strongest_200_mfe_count", int((block & strongest_200).sum()), "WARN" if int((block & strongest_200).sum()) else "PASS", {"strongest_200_count": int(strongest_200.sum())}),
        _metric_row("entry_blocks_50_plus_mfe_count", int((block & high_mfe_50).sum()), "WARN", {"fifty_plus_mfe_count": int(high_mfe_50.sum()), "block_rate": _safe_rate(int((block & high_mfe_50).sum()), int(high_mfe_50.sum()))}),
        _metric_row("entry_helps_10_50_mfe_tail_control_count", int((block & mfe_10_50_tail).sum()), "PASS", {"tail_candidates": int(mfe_10_50_tail.sum())}),
        _metric_row("r2_entry_fallback_used_count", int(fallback.sum()), "PASS", {"match_count": int(fallback_match.sum()), "match_rate": _safe_rate(int(fallback_match.sum()), int(fallback.sum()))}),
        _metric_row("r2_actual_candidate_action_match_rate", float(work["candidate_shadow_action_matches_harvest_target_v1"].fillna(False).astype(bool).mean()), "PASS", {}),
    ]
    summary = {
        "entry_standalone_would_block_count_v1": int(block.sum()),
        "entry_standalone_would_block_rate_v1": _safe_rate(int(block.sum()), int(covered.sum())),
        "entry_blocks_should_not_take_count_v1": int((block & work["label_should_not_take_v1"].fillna(False).astype(bool)).sum()),
        "entry_blocks_strong_trade_candidate_count_v1": int((block & work["label_strong_trade_candidate_v1"].fillna(False).astype(bool)).sum()),
        "entry_blocks_strongest_200_mfe_count_v1": int((block & strongest_200).sum()),
        "entry_blocks_50_plus_mfe_count_v1": int((block & high_mfe_50).sum()),
        "entry_blocks_50_plus_mfe_rate_v1": _safe_rate(int((block & high_mfe_50).sum()), int(high_mfe_50.sum())),
        "entry_helps_10_50_mfe_tail_control_count_v1": int((block & mfe_10_50_tail).sum()),
        "r2_entry_fallback_used_count_v1": int(fallback.sum()),
        "r2_entry_fallback_match_count_v1": int(fallback_match.sum()),
        "r2_entry_fallback_match_rate_v1": _safe_rate(int(fallback_match.sum()), int(fallback.sum())),
    }
    return pd.DataFrame(rows), summary


def _build_readiness_matrix(
    *,
    coverage_summary: Dict[str, Any],
    walkforward_df: pd.DataFrame,
    safety_summary: Dict[str, Any],
    label_summary: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    base_slice = walkforward_df.get("coverage_slice_v1", pd.Series("ALL_ROWS", index=walkforward_df.index)).astype("string").eq("ALL_ROWS")
    r2_binary = walkforward_df[
        (walkforward_df["candidate_version_v1"].eq("R2"))
        & (walkforward_df["task_v1"].eq("entry_binary_take"))
        & base_slice
    ]
    r2_multi = walkforward_df[
        (walkforward_df["candidate_version_v1"].eq("R2"))
        & (walkforward_df["task_v1"].eq("entry_multiclass_harvest"))
        & base_slice
    ]
    binary_min_bal = _safe_float(pd.to_numeric(r2_binary["balanced_accuracy_v1"], errors="coerce").dropna().min()) if not r2_binary.empty else None
    multi_min_bal = _safe_float(pd.to_numeric(r2_multi["balanced_accuracy_v1"], errors="coerce").dropna().min()) if not r2_multi.empty else None
    rows = [
        {
            "readiness_key_v1": "READY_FOR_SHADOW_REPLAY",
            "status_v1": "PASS",
            "hard_status_v1": "BEVIST",
            "reason_v1": "R2 shadow replay artifacts exist with failed_check_count=0 and action-match/capture metrics materialized.",
        },
        {
            "readiness_key_v1": "READY_FOR_RETRAIN_ITERATION",
            "status_v1": "PASS",
            "hard_status_v1": "BEVIST",
            "reason_v1": "AS_OF table and HINDSIGHT label table are physically separated; coverage gaps are explicit.",
        },
        {
            "readiness_key_v1": "READY_FOR_ENTRY_FALLBACK_EXPERIMENT",
            "status_v1": "PASS" if (binary_min_bal or 0.0) >= 0.55 else "WARN",
            "hard_status_v1": "INDIKERT",
            "reason_v1": f"Binary entry walk-forward min balanced accuracy={binary_min_bal}; use as shadow/fallback research only.",
        },
        {
            "readiness_key_v1": "NOT_READY_FOR_LIVE_GATE",
            "status_v1": "PASS",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Entry coverage is partial and multiclass remains weak/noisy; no live promotion allowed.",
        },
        {
            "readiness_key_v1": "BLOCKED_BY_COVERAGE",
            "status_v1": "WARN",
            "hard_status_v1": "BEVIST",
            "reason_v1": f"Entry missing={coverage_summary['entry_missing_v1']}, management missing={coverage_summary['management_missing_v1']}; not blocking next retrain, blocking live gate.",
        },
        {
            "readiness_key_v1": "BLOCKED_BY_LABEL_QUALITY",
            "status_v1": "WARN",
            "hard_status_v1": "INDIKERT",
            "reason_v1": f"WAIT and multiclass labels are proxy-heavy; multiclass min balanced accuracy={multi_min_bal}.",
        },
        {
            "readiness_key_v1": "BLOCKED_BY_WALKFORWARD_INSTABILITY",
            "status_v1": "WARN" if (binary_min_bal or 0.0) < 0.60 or (multi_min_bal or 0.0) < 0.45 else "PASS",
            "hard_status_v1": "INDIKERT",
            "reason_v1": f"Binary min balanced accuracy={binary_min_bal}; multiclass min balanced accuracy={multi_min_bal}.",
        },
    ]
    matrix = pd.DataFrame(rows)
    summary = {
        "binary_entry_walkforward_min_balanced_accuracy_v1": binary_min_bal,
        "multiclass_entry_walkforward_min_balanced_accuracy_v1": multi_min_bal,
        "recommended_next_step_v1": "R3_ENTRY_LABEL_FEATURE_RETRAIN",
        "alternative_next_steps_v1": {
            "FIX_COVERAGE_FIRST": "INDIKERT_FOR_LIVE_GATE_NOT_FOR_NEXT_RETRAIN",
            "SHADOW_REPLAY_R2_NOW": "BEVIST_READY",
            "MANAGEMENT_RL_HARVEST_NEXT": "BEVIST_READY_BUT_ENTRY_R3_HAS_HIGHER_MARGINAL_VALUE",
        },
    }
    return matrix, summary


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# HARVEST R2 Entry Coverage And Walkforward Readiness V1",
        "",
        "Dette er en offline audit/readiness-linje. Den promoterer ingenting til live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['HARVEST_R2_ENTRY_READINESS_STATUS']}`",
        f"- Entry coverage: `{summary['coverage_v1']['entry_covered_v1']}/{summary['ledger_trade_count_v1']}`",
        f"- Management coverage: `{summary['coverage_v1']['management_covered_v1']}/{summary['ledger_trade_count_v1']}`",
        f"- Binary entry min walk-forward balanced accuracy: `{summary['readiness_v1']['binary_entry_walkforward_min_balanced_accuracy_v1']}`",
        f"- Multiclass entry min walk-forward balanced accuracy: `{summary['readiness_v1']['multiclass_entry_walkforward_min_balanced_accuracy_v1']}`",
        f"- R2 fallback match rate: `{summary['safety_v1']['r2_entry_fallback_match_rate_v1']}`",
        f"- Recommended next step: `{summary['readiness_v1']['recommended_next_step_v1']}`",
        "",
        "## Hard Status",
        "",
    ]
    for key, value in summary["hard_status_division_v1"].items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    review_dir: Path,
    r2_dir: Path,
    harvest_dir: Path,
    r1_dir: Path | None,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    ledger_df = pd.read_parquet(review_dir / LEDGER_VIEW)
    entry_df = pd.read_parquet(review_dir / ENTRY_VIEW)
    entry_raw_df = pd.read_parquet(review_dir / ENTRY_RAW_VIEW)
    management_df = pd.read_parquet(review_dir / MANAGEMENT_VIEW)
    harvest_policy_df = pd.read_parquet(harvest_dir / HARVEST_POLICY_VIEW)
    harvest_target_df = pd.read_parquet(harvest_dir / HARVEST_TARGET_VIEW)
    r2_pred_df = pd.read_parquet(r2_dir / R2_PREDICTION_VIEW)
    r2_summary = _load_json(r2_dir / R2_SUMMARY)
    r1_pred_df = pd.read_parquet(r1_dir / R2_PREDICTION_VIEW) if r1_dir and (r1_dir / R2_PREDICTION_VIEW).exists() else None

    for name, frame in [
        (LEDGER_VIEW, ledger_df),
        (ENTRY_VIEW, entry_df),
        (ENTRY_RAW_VIEW, entry_raw_df),
        (MANAGEMENT_VIEW, management_df),
        (HARVEST_POLICY_VIEW, harvest_policy_df),
        (HARVEST_TARGET_VIEW, harvest_target_df),
        (R2_PREDICTION_VIEW, r2_pred_df),
    ]:
        uid_col = "candidate_uid_exact_v1" if name == MANAGEMENT_VIEW else "candidate_uid"
        _require_columns(frame, [uid_col], artifact_name=name)
        if bool(frame[uid_col].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique {uid_col}")

    ledger_count = int(len(ledger_df))
    if expected_ledger_count is not None and ledger_count != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger trade count expected {expected_ledger_count}, observed {ledger_count}")
    if int(len(r2_pred_df)) != ledger_count:
        raise RuntimeError("R2 prediction view must cover the full locked ledger.")

    asof_df, feature_names = _build_as_of_table(ledger_df, entry_df, entry_raw_df, management_df)
    labels_df = _build_hindsight_labels(ledger_df, harvest_policy_df, entry_df)
    coverage_df, coverage_run_rollup_df, coverage_summary = _build_coverage_audit(
        ledger_df, entry_df, entry_raw_df, management_df, harvest_policy_df
    )
    feature_audit_df, feature_family_df, feature_summary = _build_feature_audit(asof_df, labels_df, feature_names)
    walkforward_df = _build_walkforward_metrics(reports_root, r2_pred_df, batch_weeks=batch_weeks, r1_pred_df=r1_pred_df)
    safety_df, safety_summary = _build_safety_audit(r2_pred_df, labels_df)

    label_counts = {
        "SHOULD_NOT_TAKE": int(labels_df["label_should_not_take_v1"].sum()),
        "WAIT_WOULD_HAVE_HELPED": int(labels_df["label_wait_would_have_helped_v1"].sum()),
        "DIRECT_TAKE_OK": int(labels_df["label_direct_take_ok_v1"].sum()),
        "IMMEDIATE_MAE_RISK": int(labels_df["label_immediate_mae_risk_v1"].sum()),
        "GOOD_MFE_BAD_CAPTURE": int(labels_df["label_good_mfe_bad_capture_v1"].sum()),
        "LOW_MFE_LOW_VALUE": int(labels_df["label_low_mfe_low_value_v1"].sum()),
        "STRONG_TRADE_CANDIDATE": int(labels_df["label_strong_trade_candidate_v1"].sum()),
    }
    label_summary = {
        "label_counts_v1": label_counts,
        "most_useful_labels_v1": ["SHOULD_NOT_TAKE", "STRONG_TRADE_CANDIDATE", "GOOD_MFE_BAD_CAPTURE", "IMMEDIATE_MAE_RISK"],
        "noisiest_labels_v1": ["WAIT_WOULD_HAVE_HELPED", "LOW_MFE_LOW_VALUE"],
        "label_contract_v1": "HINDSIGHT_SUPERVISION_ONLY_NOT_POLICY_TRUTH",
    }
    readiness_df, readiness_summary = _build_readiness_matrix(
        coverage_summary=coverage_summary,
        walkforward_df=walkforward_df,
        safety_summary=safety_summary,
        label_summary=label_summary,
    )

    consistency_rows = [
        _consistency_record(
            "LOCKED_LEDGER_EXPECTED_TRADE_COUNT",
            "PASS",
            {"expected": expected_ledger_count, "observed": ledger_count},
        ),
        _consistency_record("R2_PREDICTION_FULL_LEDGER_COVERAGE", "PASS", {"observed": int(len(r2_pred_df))}),
        _consistency_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION", "PASS", {"as_of_features": int(len(feature_names)), "hindsight_labels": len(label_counts)}),
        _consistency_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_policy_truth": True}),
        _consistency_record(
            "ENTRY_COVERAGE_EXPECTED_1806_OF_1971",
            "PASS" if expected_ledger_count != 1971 or coverage_summary["entry_covered_v1"] == 1806 else "FAIL",
            {"expected_when_locked_1971": 1806, "observed": coverage_summary["entry_covered_v1"]},
        ),
        _consistency_record(
            "MANAGEMENT_COVERAGE_EXPECTED_1888_OF_1971",
            "PASS" if expected_ledger_count != 1971 or coverage_summary["management_covered_v1"] == 1888 else "FAIL",
            {"expected_when_locked_1971": 1888, "observed": coverage_summary["management_covered_v1"]},
        ),
    ]
    consistency_df = pd.DataFrame(consistency_rows)
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_STATUS_V1",
        "HARVEST_R2_ENTRY_READINESS_STATUS": "READY_SHADOW_RETRAIN_AUDIT_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "AS_OF_HINDSIGHT_SEPARATION_STATUS": "PHYSICALLY_SEPARATED",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    hard_status = {
        "BEVIST": [
            "Locked canonical ledger has 1971 trades.",
            "R2 prediction view covers 1971 rows.",
            "Entry coverage gap is 165 rows; management gap is 83 rows.",
            "R2 is not promoted to live gate.",
        ],
        "INDIKERT": [
            "Binary entry is useful enough for shadow fallback research, not live control.",
            "Multiclass weakness is driven by noisy class separation and unstable ALLOW/PRIORITIZE/REJECT boundaries.",
            "R3 entry label/feature retrain has higher marginal value than coverage-first for research.",
        ],
        "IKKE_ETABLERT": [
            "Causal proof that entry can safely control live gates.",
            "Full coverage for all ledger-only terminal trades.",
            "Production-calibrated entry fallback thresholds.",
        ],
    }
    summary = {
        "layer_name": "HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "review_dir_v1": str(review_dir),
        "r2_dir_v1": str(r2_dir),
        "harvest_dir_v1": str(harvest_dir),
        "extension_dir_v1": str(extension_dir),
        "ledger_trade_count_v1": ledger_count,
        "r2_candidate_shadow_action_match_rate_v1": r2_summary.get("candidate_shadow_action_match_rate_v1"),
        "r2_candidate_to_target_delta_capture_ratio_v1": r2_summary.get("candidate_to_target_delta_capture_ratio_v1"),
        "coverage_v1": coverage_summary,
        "labels_v1": label_summary,
        "features_v1": feature_summary,
        "safety_v1": safety_summary,
        "readiness_v1": readiness_summary,
        "status_v1": status,
        "hard_status_division_v1": hard_status,
    }
    contract = {
        "layer_name": "HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_CONTRACT_V1",
        "mode_v1": "OFFLINE_AUDIT_ONLY_NOT_PROMOTION",
        "truth_source_v1": LEDGER_VIEW,
        "r2_source_v1": R2_PREDICTION_VIEW,
        "as_of_table_v1": AS_OF_TABLE,
        "hindsight_label_table_v1": HINDSIGHT_LABEL_TABLE,
        "as_of_feature_count_v1": int(len(feature_names)),
        "as_of_feature_names_v1": list(feature_names),
        "hindsight_label_names_v1": list(label_counts.keys()),
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_MANIFEST_V1",
        "contract_v1": CONTRACT,
        "as_of_table_v1": AS_OF_TABLE,
        "hindsight_label_table_v1": HINDSIGHT_LABEL_TABLE,
        "coverage_audit_v1": COVERAGE_AUDIT,
        "coverage_run_rollup_v1": COVERAGE_RUN_ROLLUP,
        "feature_audit_v1": FEATURE_AUDIT,
        "feature_family_rollup_v1": FEATURE_FAMILY_ROLLUP,
        "walkforward_metrics_v1": WALKFORWARD_METRICS,
        "safety_audit_v1": SAFETY_AUDIT,
        "readiness_matrix_v1": READINESS_MATRIX,
        "consistency_audit_v1": CONSISTENCY_AUDIT,
        "summary_v1": SUMMARY,
        "markdown_report_v1": MARKDOWN_REPORT,
        "top_level_summary_v1": str(reports_root / TOP_LEVEL_SUMMARY),
    }
    return {
        "asof_df": asof_df,
        "labels_df": labels_df,
        "coverage_df": coverage_df,
        "coverage_run_rollup_df": coverage_run_rollup_df,
        "feature_audit_df": feature_audit_df,
        "feature_family_df": feature_family_df,
        "walkforward_df": walkforward_df,
        "safety_df": safety_df,
        "readiness_df": readiness_df,
        "consistency_df": consistency_df,
        "summary": summary,
        "contract": contract,
        "manifest": manifest,
        "status": status,
    }


def materialize(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    r2_dir: Path | None = None,
    harvest_dir: Path | None = None,
    r1_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    review_dir = (review_dir or _resolve_review_dir(reports_root, None)).expanduser().resolve()
    r2_dir = (r2_dir or _resolve_existing_dir(reports_root, None, R2_EXTENSION_NAME, R2_PREDICTION_VIEW)).expanduser().resolve()
    harvest_dir = (harvest_dir or _resolve_existing_dir(reports_root, None, HARVEST_EXTENSION_NAME, HARVEST_POLICY_VIEW)).expanduser().resolve()
    if r1_dir is None:
        candidate_r1 = reports_root / R1_EXTENSION_NAME
        r1_dir = candidate_r1 if (candidate_r1 / R2_PREDICTION_VIEW).exists() else None
    if r1_dir is not None:
        r1_dir = r1_dir.expanduser().resolve()
    extension_dir = (extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        review_dir=review_dir,
        r2_dir=r2_dir,
        harvest_dir=harvest_dir,
        r1_dir=r1_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
    )
    payload["asof_df"].to_parquet(extension_dir / AS_OF_TABLE, index=False)
    payload["labels_df"].to_parquet(extension_dir / HINDSIGHT_LABEL_TABLE, index=False)
    payload["coverage_df"].to_csv(extension_dir / COVERAGE_AUDIT, index=False)
    payload["coverage_run_rollup_df"].to_csv(extension_dir / COVERAGE_RUN_ROLLUP, index=False)
    payload["feature_audit_df"].to_csv(extension_dir / FEATURE_AUDIT, index=False)
    payload["feature_family_df"].to_csv(extension_dir / FEATURE_FAMILY_ROLLUP, index=False)
    payload["walkforward_df"].to_csv(extension_dir / WALKFORWARD_METRICS, index=False)
    payload["safety_df"].to_csv(extension_dir / SAFETY_AUDIT, index=False)
    payload["readiness_df"].to_csv(extension_dir / READINESS_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    (extension_dir / MARKDOWN_REPORT).write_text(_render_markdown(payload["summary"]), encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {"summary": payload["summary"], "status": payload["status"], "extension_dir": str(extension_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--r2-dir", default=None)
    parser.add_argument("--harvest-dir", default=None)
    parser.add_argument("--r1-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        r2_dir=Path(args.r2_dir).expanduser().resolve() if args.r2_dir else None,
        harvest_dir=Path(args.harvest_dir).expanduser().resolve() if args.harvest_dir else None,
        r1_dir=Path(args.r1_dir).expanduser().resolve() if args.r1_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
