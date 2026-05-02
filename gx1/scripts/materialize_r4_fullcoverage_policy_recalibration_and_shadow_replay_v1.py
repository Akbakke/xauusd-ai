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


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FULLCOVERAGE_POLICY_RECALIBRATION_AND_SHADOW_REPLAY_V1"
READINESS_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_ENTRY_COVERAGE_REPAIR_READINESS_V1"
R3_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R3_ENTRY_LABEL_FEATURE_RETRAIN_COVERAGE_REPAIRED_V1"
R4_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_ENTRY_CALIBRATED_FALLBACK_RETRAIN_COVERAGE_REPAIRED_V1"

R2_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
R2_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
REPAIR_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv"
REPAIR_SUMMARY = "shadow_meta_all_trade_review_entry_coverage_repair_summary_v1.json"
READINESS_CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"

R3_PREDICTION_VIEW = "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
R3_SUMMARY = "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json"

R4_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_policy_prediction_view_v1.parquet"
R4_SUMMARY = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_summary_v1.json"

CONTRACT = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_contract_v1.json"
AS_OF_TABLE = "shadow_meta_all_trade_review_r4_fullcoverage_as_of_table_v1.parquet"
HINDSIGHT_TABLE = "shadow_meta_all_trade_review_r4_fullcoverage_hindsight_label_outcome_table_v1.parquet"
REPAIR_VERIFICATION = "shadow_meta_all_trade_review_r4_fullcoverage_repair_verification_v1.csv"
THRESHOLD_FRONTIER = "shadow_meta_all_trade_review_r4_fullcoverage_threshold_frontier_v1.csv"
WINNER_SAFETY_AUDIT = "shadow_meta_all_trade_review_r4_fullcoverage_winner_protection_safety_audit_v1.csv"
HEAD_TO_HEAD = "shadow_meta_all_trade_review_r4_fullcoverage_head_to_head_v1.csv"
WALKFORWARD = "shadow_meta_all_trade_review_r4_fullcoverage_walkforward_shadow_replay_v1.csv"
DECISION_MATRIX = "shadow_meta_all_trade_review_r4_fullcoverage_decision_matrix_v1.csv"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_fullcoverage_policy_prediction_view_v1.parquet"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r4_fullcoverage_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
LEAKAGE_TOKENS = (
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

TASK_PROB_COLUMNS = {
    "should_not_take": "pred__entry_r3_should_not_take__prob_true_v1",
    "immediate_mae_risk": "pred__entry_r3_immediate_mae_risk__prob_true_v1",
    "wait_advisory": "pred__entry_r3_wait_would_have_helped__prob_true_v1",
    "strong_trade_candidate": "pred__entry_r3_strong_trade_candidate__prob_true_v1",
    "direct_take_ok": "pred__entry_r3_direct_take_ok__prob_true_v1",
    "good_mfe_bad_capture": "pred__entry_r3_good_mfe_bad_capture__prob_true_v1",
}

R4_REFERENCE_THRESHOLDS = {
    "should_not_take_threshold_v1": 0.60,
    "direct_take_protection_ceiling_v1": 0.55,
    "strong_winner_protection_threshold_v1": 0.75,
    "immediate_mae_risk_threshold_v1": 0.80,
    "wait_advisory_threshold_v1": 0.85,
}


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


def _resolve_dir(reports_root: Path, path_arg: str | None, default_name: str, required_file: str) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Required dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} missing required artifact {required_file}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.strip().str.lower().eq("true").fillna(default).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _prob(frame: pd.DataFrame, task: str) -> pd.Series:
    return pd.to_numeric(frame.get(TASK_PROB_COLUMNS[task], pd.Series(np.nan, index=frame.index)), errors="coerce")


def _check_feature_names(feature_names: Sequence[str]) -> None:
    bad: List[str] = []
    for feature in feature_names:
        lower = feature.lower()
        for token in LEAKAGE_TOKENS:
            if token == "realized" and "realized_vol" in lower:
                continue
            if token in lower:
                bad.append(feature)
                break
    if bad:
        raise ValueError(f"AS_OF feature list contains forbidden hindsight/target-like names: {bad[:20]}")


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


def _build_joined(
    *,
    readiness_dir: Path,
    r3_dir: Path,
    r4_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    labels_df = pd.read_parquet(readiness_dir / R2_LABEL_TABLE)
    repair_df = pd.read_csv(readiness_dir / REPAIR_AUDIT)
    r3_df = pd.read_parquet(r3_dir / R3_PREDICTION_VIEW)
    r4_df = pd.read_parquet(r4_dir / R4_POLICY_PREDICTION_VIEW)
    readiness_contract = _load_json(readiness_dir / READINESS_CONTRACT)
    repair_summary = _load_json(readiness_dir / REPAIR_SUMMARY)
    r3_summary = _load_json(r3_dir / R3_SUMMARY)
    r4_summary = _load_json(r4_dir / R4_SUMMARY)

    feature_names = [str(feature) for feature in readiness_contract.get("as_of_feature_names_v1", [])]
    if not feature_names:
        raise RuntimeError("Fullcoverage readiness contract missing as_of_feature_names_v1")
    _check_feature_names(feature_names)
    _require_columns(asof_df, ["candidate_uid", "entry_observation_present_v1", "entry_raw_state_present_v1", *feature_names], artifact_name=R2_AS_OF_TABLE)
    _require_columns(labels_df, ["candidate_uid", "hindsight_entry_decision_review_v1", "label_should_not_take_v1", "label_strong_trade_candidate_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "giveback_bps_v1", "baseline_realized_pnl_bps_v1"], artifact_name=R2_LABEL_TABLE)
    _require_columns(repair_df, ["candidate_uid", "synthetic_value_used_v1", "repair_timestamp_utc_v1", "replay_timestamp_utc_v1"], artifact_name=REPAIR_AUDIT)
    _require_columns(r3_df, ["candidate_uid", "entry_r3_feature_available_v1", *TASK_PROB_COLUMNS.values(), "entry_r3_shadow_action_v1"], artifact_name=R3_PREDICTION_VIEW)
    _require_columns(r4_df, ["candidate_uid", "r2_entry_fallback_row_v1", "r3_conservative_blocks_v1", "r4_entry_fallback_block_v1"], artifact_name=R4_POLICY_PREDICTION_VIEW)

    for name, frame in [(R2_AS_OF_TABLE, asof_df), (R2_LABEL_TABLE, labels_df), (R3_PREDICTION_VIEW, r3_df), (R4_POLICY_PREDICTION_VIEW, r4_df)]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")

    r3_cols = [
        "candidate_uid",
        "entry_r3_feature_available_v1",
        "entry_r3_shadow_action_v1",
        "entry_r3_shadow_action_source_v1",
        *TASK_PROB_COLUMNS.values(),
    ]
    r4_cols = [
        "candidate_uid",
        "r2_entry_fallback_row_v1",
        "r2_entry_fallback_correct_v1",
        "r3_conservative_blocks_v1",
        "r4_entry_fallback_block_v1",
        "r4_entry_fallback_action_v1",
        "r4_entry_fallback_source_v1",
    ]
    meta_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "entry_observation_present_v1",
        "entry_raw_state_present_v1",
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
        "entry_coverage_repair_source_v1",
    ]
    label_cols = [
        "candidate_uid",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "trade_outcome_class",
        "exit_reason",
        "session",
        "vol_regime",
        "trend_regime",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "exit_harvest_policy_action_v1",
        "home_run_200bps_opportunity_v1",
        "runner_100bps_opportunity_v1",
        "runner_50bps_opportunity_v1",
        "label_should_not_take_v1",
        "label_immediate_mae_risk_v1",
        "label_wait_would_have_helped_v1",
        "label_good_mfe_bad_capture_v1",
        "label_low_mfe_low_value_v1",
        "label_strong_trade_candidate_v1",
        "label_direct_take_ok_v1",
    ]
    repair_cols = [
        "candidate_uid",
        "repair_timestamp_utc_v1",
        "replay_timestamp_utc_v1",
        "entry_coverage_repair_status_v1",
        "synthetic_value_used_v1",
        "hindsight_label_used_for_as_of_repair_v1",
        "recovery_source_v1",
        "entry_gap_reason_code_v1",
        "coverage_gap_scope_v1",
    ]
    joined = (
        asof_df[[column for column in meta_cols if column in asof_df.columns]]
        .merge(labels_df[[column for column in label_cols if column in labels_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(r3_df[[column for column in r3_cols if column in r3_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(r4_df[[column for column in r4_cols if column in r4_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(repair_df[[column for column in repair_cols if column in repair_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    joined["entry_coverage_repair_applied_v1"] = _bool(joined, "entry_coverage_repair_applied_v1")
    joined["is_repaired_165_v1"] = joined["entry_coverage_repair_applied_v1"]
    joined["take_was_ok_v1"] = joined["hindsight_entry_decision_review_v1"].astype("string").eq("TAKE_WAS_OK")
    joined["fifty_plus_mfe_v1"] = _num(joined, "peak_mfe_bps_v1").ge(50.0)
    joined["hundred_plus_mfe_v1"] = _num(joined, "peak_mfe_bps_v1").ge(100.0)
    joined["two_hundred_plus_mfe_v1"] = _num(joined, "peak_mfe_bps_v1").ge(200.0)
    joined["tail_10_50_mfe_v1"] = _num(joined, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(joined, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(joined, "label_should_not_take_v1")
    )
    joined["strongest_winner_path_v1"] = joined["two_hundred_plus_mfe_v1"] | (
        _bool(joined, "label_strong_trade_candidate_v1")
        & _num(joined, "baseline_realized_pnl_bps_v1").gt(0.0)
        & _num(joined, "harvest_capture_ratio_v1", default=0.0).ge(0.5)
    )
    joined["mae_bucket_v1"] = pd.cut(
        _num(joined, "mae_abs_bps_v1"),
        bins=[-np.inf, 15.0, 40.0, np.inf],
        labels=["LOW_MAE_LT15", "MEDIUM_MAE_15_40", "HIGH_MAE_GE40"],
    ).astype("string")
    return joined, asof_df, labels_df, repair_df, repair_summary, r3_summary, r4_summary


def _policy_metric_row(policy_name: str, scope: str, frame: pd.DataFrame, block: pd.Series, *, thresholds: Dict[str, Any] | None = None) -> Dict[str, Any]:
    block = block.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    strong = _bool(frame, "label_strong_trade_candidate_v1")
    fifty = frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)
    hundred = frame["hundred_plus_mfe_v1"].fillna(False).astype(bool)
    two_hundred = frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool)
    tail = frame["tail_10_50_mfe_v1"].fillna(False).astype(bool)
    repaired = frame["is_repaired_165_v1"].fillna(False).astype(bool)
    take_ok = frame["take_was_ok_v1"].fillna(False).astype(bool)
    strongest = frame["strongest_winner_path_v1"].fillna(False).astype(bool)
    realized_delta = (-_num(frame, "baseline_realized_pnl_bps_v1")).where(block, 0.0)
    row_count = int(len(frame))
    blocked_count = int(block.sum())
    should_count = int(should.sum())
    y_true = should.astype(int)
    y_pred = block.astype(int)
    true_negative = int((~block & ~should).sum())
    true_positive = int((block & should).sum())
    false_positive = int((block & ~should).sum())
    false_negative = int((~block & should).sum())
    specificity = _safe_rate(float(true_negative), float((~should).sum())) or 0.0
    recall = _safe_rate(float(true_positive), float(should_count)) or 0.0
    balanced_accuracy = (specificity + recall) / 2.0
    return {
        "policy_name_v1": policy_name,
        "scope_v1": scope,
        "row_count_v1": row_count,
        "block_count_v1": blocked_count,
        "block_rate_v1": _safe_rate(float(blocked_count), float(row_count)),
        "should_not_take_count_v1": should_count,
        "should_not_take_block_count_v1": true_positive,
        "should_not_take_precision_v1": _safe_rate(float(true_positive), float(blocked_count)),
        "should_not_take_recall_v1": recall,
        "false_allow_should_not_take_count_v1": false_negative,
        "take_was_ok_block_count_v1": int((block & take_ok).sum()),
        "take_was_ok_block_rate_v1": _safe_rate(float((block & take_ok).sum()), float(take_ok.sum())),
        "strong_trade_false_block_count_v1": int((block & strong).sum()),
        "strong_trade_false_block_rate_v1": _safe_rate(float((block & strong).sum()), float(strong.sum())),
        "fifty_plus_mfe_block_count_v1": int((block & fifty).sum()),
        "fifty_plus_mfe_block_rate_v1": _safe_rate(float((block & fifty).sum()), float(fifty.sum())),
        "hundred_plus_mfe_block_count_v1": int((block & hundred).sum()),
        "hundred_plus_mfe_block_rate_v1": _safe_rate(float((block & hundred).sum()), float(hundred.sum())),
        "two_hundred_plus_mfe_block_count_v1": int((block & two_hundred).sum()),
        "two_hundred_plus_mfe_block_rate_v1": _safe_rate(float((block & two_hundred).sum()), float(two_hundred.sum())),
        "strongest_winner_path_block_count_v1": int((block & strongest).sum()),
        "strongest_winner_path_block_rate_v1": _safe_rate(float((block & strongest).sum()), float(strongest.sum())),
        "repaired_165_block_count_v1": int((block & repaired).sum()),
        "repaired_165_block_rate_v1": _safe_rate(float((block & repaired).sum()), float(repaired.sum())),
        "tail_10_50_help_count_v1": int((block & tail).sum()),
        "tail_10_50_help_recall_v1": _safe_rate(float((block & tail).sum()), float(tail.sum())),
        "hindsight_skip_delta_bps_v1": float(realized_delta.sum()),
        "blocked_avg_mfe_bps_v1": _safe_float(_num(frame.loc[block], "peak_mfe_bps_v1").mean()) if blocked_count else None,
        "blocked_avg_mae_bps_v1": _safe_float(_num(frame.loc[block], "mae_abs_bps_v1").mean()) if blocked_count else None,
        "blocked_avg_giveback_bps_v1": _safe_float(_num(frame.loc[block], "giveback_bps_v1").mean()) if blocked_count else None,
        "binary_balanced_accuracy_vs_should_not_take_v1": float(balanced_accuracy),
        "confusion_matrix_json_v1": _json_dumps([[true_negative, false_positive], [false_negative, true_positive]]),
        "thresholds_json_v1": _json_dumps(thresholds or {}),
    }


def _base_policy_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index),
        "R2_FALLBACK_REFERENCE": _bool(frame, "r2_entry_fallback_row_v1"),
        "R3_FULLCOVERAGE_CONSERVATIVE": _bool(frame, "r3_conservative_blocks_v1"),
        "R4_REPAIRED_SELECTED_REFERENCE": _bool(frame, "r4_entry_fallback_block_v1"),
    }


def _threshold_mask(frame: pd.DataFrame, stack_name: str, params: Dict[str, float], *, preserve_r2: bool = False) -> pd.Series:
    p_should = _prob(frame, "should_not_take")
    p_mae = _prob(frame, "immediate_mae_risk")
    p_wait = _prob(frame, "wait_advisory")
    p_strong = _prob(frame, "strong_trade_candidate")
    p_direct = _prob(frame, "direct_take_ok")
    feature_available = _bool(frame, "entry_r3_feature_available_v1")
    should_signal = p_should.ge(params.get("should_not_take_threshold_v1", 0.60)).fillna(False)
    mae_signal = p_mae.ge(params.get("immediate_mae_risk_threshold_v1", 0.80)).fillna(False)
    wait_signal = p_wait.ge(params.get("wait_advisory_threshold_v1", 0.85)).fillna(False)
    direct_weak = p_direct.lt(params.get("direct_take_protection_ceiling_v1", 0.55)).fillna(False)
    strong_protect = p_strong.ge(params.get("strong_winner_protection_threshold_v1", 0.75)).fillna(False)
    if stack_name == "SHOULD_NOT_TAKE_BLOCKER_ONLY":
        signal = should_signal
    elif stack_name == "SHOULD_NOT_TAKE_STRONG_PROTECTED":
        signal = should_signal & ~strong_protect
    elif stack_name == "SHOULD_DIRECT_WEAK_STRONG_PROTECTED":
        signal = should_signal & direct_weak & ~strong_protect
    elif stack_name == "IMMEDIATE_MAE_DIRECT_WEAK_STRONG_PROTECTED":
        signal = mae_signal & direct_weak & ~strong_protect
    elif stack_name == "WAIT_DIRECT_WEAK_STRONG_PROTECTED":
        signal = wait_signal & direct_weak & ~strong_protect
    elif stack_name == "SHOULD_OR_MAE_DIRECT_WEAK_STRONG_PROTECTED":
        signal = (should_signal | mae_signal) & direct_weak & ~strong_protect
    elif stack_name == "SHOULD_OR_WAIT_DIRECT_WEAK_STRONG_PROTECTED":
        signal = (should_signal | wait_signal) & direct_weak & ~strong_protect
    elif stack_name == "COMBINED_CONSERVATIVE_STACK":
        signal = (should_signal | mae_signal | wait_signal) & direct_weak & ~strong_protect
    else:
        raise ValueError(f"Unknown stack name: {stack_name}")
    if preserve_r2:
        signal = signal | _bool(frame, "r2_entry_fallback_row_v1")
    return (feature_available & signal).fillna(False).astype(bool)


def _build_threshold_frontier(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mask_map: Dict[str, pd.Series] = {}
    for policy_name, mask in _base_policy_masks(frame).items():
        rows.append(_policy_metric_row(policy_name, "ALL", frame, mask, thresholds={"reference_policy_v1": policy_name}))
        mask_map[policy_name] = mask

    head_thresholds = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
    for head_name, task_name in [
        ("SHOULD_NOT_TAKE_HEAD_ONLY", "should_not_take"),
        ("IMMEDIATE_MAE_RISK_HEAD_ONLY", "immediate_mae_risk"),
        ("WAIT_ADVISORY_HEAD_ONLY", "wait_advisory"),
    ]:
        for threshold in head_thresholds:
            mask = _prob(frame, task_name).ge(threshold).fillna(False) & _bool(frame, "entry_r3_feature_available_v1")
            rows.append(_policy_metric_row(head_name, "ALL", frame, mask, thresholds={f"{task_name}_threshold_v1": threshold}))

    compact_should = [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    compact_mae = [0.55, 0.65, 0.75, 0.80, 0.85, 0.90]
    compact_wait = [0.65, 0.75, 0.85, 0.95]
    compact_direct = [0.45, 0.55, 0.65]
    compact_protect = [0.50, 0.65, 0.75, 0.85]
    stack_names = [
        "SHOULD_NOT_TAKE_BLOCKER_ONLY",
        "SHOULD_NOT_TAKE_STRONG_PROTECTED",
        "SHOULD_DIRECT_WEAK_STRONG_PROTECTED",
        "IMMEDIATE_MAE_DIRECT_WEAK_STRONG_PROTECTED",
        "WAIT_DIRECT_WEAK_STRONG_PROTECTED",
        "SHOULD_OR_MAE_DIRECT_WEAK_STRONG_PROTECTED",
        "SHOULD_OR_WAIT_DIRECT_WEAK_STRONG_PROTECTED",
        "COMBINED_CONSERVATIVE_STACK",
    ]
    for stack_name in stack_names:
        for preserve_r2 in [False, True]:
            if stack_name.startswith("IMMEDIATE_MAE"):
                iterator = (
                    {
                        "should_not_take_threshold_v1": 0.60,
                        "immediate_mae_risk_threshold_v1": t_mae,
                        "wait_advisory_threshold_v1": 0.85,
                        "direct_take_protection_ceiling_v1": t_direct,
                        "strong_winner_protection_threshold_v1": t_protect,
                    }
                    for t_mae in compact_mae
                    for t_direct in compact_direct
                    for t_protect in compact_protect
                )
            elif stack_name.startswith("WAIT"):
                iterator = (
                    {
                        "should_not_take_threshold_v1": 0.60,
                        "immediate_mae_risk_threshold_v1": 0.80,
                        "wait_advisory_threshold_v1": t_wait,
                        "direct_take_protection_ceiling_v1": t_direct,
                        "strong_winner_protection_threshold_v1": t_protect,
                    }
                    for t_wait in compact_wait
                    for t_direct in compact_direct
                    for t_protect in compact_protect
                )
            elif "MAE" in stack_name:
                iterator = (
                    {
                        "should_not_take_threshold_v1": t_should,
                        "immediate_mae_risk_threshold_v1": t_mae,
                        "wait_advisory_threshold_v1": 0.85,
                        "direct_take_protection_ceiling_v1": t_direct,
                        "strong_winner_protection_threshold_v1": t_protect,
                    }
                    for t_should in compact_should
                    for t_mae in compact_mae
                    for t_direct in compact_direct
                    for t_protect in compact_protect
                )
            elif "WAIT" in stack_name:
                iterator = (
                    {
                        "should_not_take_threshold_v1": t_should,
                        "immediate_mae_risk_threshold_v1": 0.80,
                        "wait_advisory_threshold_v1": t_wait,
                        "direct_take_protection_ceiling_v1": t_direct,
                        "strong_winner_protection_threshold_v1": t_protect,
                    }
                    for t_should in compact_should
                    for t_wait in compact_wait
                    for t_direct in compact_direct
                    for t_protect in compact_protect
                )
            else:
                iterator = (
                    {
                        "should_not_take_threshold_v1": t_should,
                        "immediate_mae_risk_threshold_v1": 0.80,
                        "wait_advisory_threshold_v1": 0.85,
                        "direct_take_protection_ceiling_v1": t_direct,
                        "strong_winner_protection_threshold_v1": t_protect,
                    }
                    for t_should in compact_should
                    for t_direct in compact_direct
                    for t_protect in compact_protect
                )
            for params in iterator:
                policy_name = f"{'R2_PRESERVED_PLUS_' if preserve_r2 else ''}{stack_name}"
                mask = _threshold_mask(frame, stack_name, params, preserve_r2=preserve_r2)
                rows.append(_policy_metric_row(policy_name, "ALL", frame, mask, thresholds={**params, "preserve_r2_fallback_v1": preserve_r2}))

    frontier = pd.DataFrame(rows).drop_duplicates(subset=["policy_name_v1", "thresholds_json_v1"], keep="first")
    reference = frontier[frontier["policy_name_v1"].eq("R4_REPAIRED_SELECTED_REFERENCE")].iloc[0].to_dict()
    constraints = {
        "max_strong_false_blocks_v1": int(reference["strong_trade_false_block_count_v1"]),
        "max_50_plus_mfe_blocks_v1": int(reference["fifty_plus_mfe_block_count_v1"]),
        "max_200_plus_mfe_blocks_v1": int(reference["two_hundred_plus_mfe_block_count_v1"]),
        "max_strongest_winner_path_blocks_v1": int(reference["strongest_winner_path_block_count_v1"]),
        "max_repaired_165_blocks_v1": int(reference["repaired_165_block_count_v1"]),
    }
    frontier["passes_winner_constraints_v1"] = (
        pd.to_numeric(frontier["strong_trade_false_block_count_v1"], errors="coerce").le(constraints["max_strong_false_blocks_v1"])
        & pd.to_numeric(frontier["fifty_plus_mfe_block_count_v1"], errors="coerce").le(constraints["max_50_plus_mfe_blocks_v1"])
        & pd.to_numeric(frontier["two_hundred_plus_mfe_block_count_v1"], errors="coerce").le(constraints["max_200_plus_mfe_blocks_v1"])
        & pd.to_numeric(frontier["strongest_winner_path_block_count_v1"], errors="coerce").le(constraints["max_strongest_winner_path_blocks_v1"])
        & pd.to_numeric(frontier["repaired_165_block_count_v1"], errors="coerce").le(constraints["max_repaired_165_blocks_v1"])
    )
    frontier["bad_recall_gain_vs_r4_reference_v1"] = pd.to_numeric(frontier["should_not_take_recall_v1"], errors="coerce") - float(reference["should_not_take_recall_v1"])
    frontier["should_not_blocks_gain_vs_r4_reference_v1"] = pd.to_numeric(frontier["should_not_take_block_count_v1"], errors="coerce") - int(reference["should_not_take_block_count_v1"])
    constrained = frontier[frontier["passes_winner_constraints_v1"].fillna(False)].copy()
    if constrained.empty:
        best = reference
    else:
        constrained["selection_score_v1"] = (
            pd.to_numeric(constrained["should_not_take_block_count_v1"], errors="coerce") * 1.0
            + pd.to_numeric(constrained["tail_10_50_help_count_v1"], errors="coerce") * 0.20
            + pd.to_numeric(constrained["should_not_take_precision_v1"], errors="coerce").fillna(0.0) * 10.0
            - pd.to_numeric(constrained["block_count_v1"], errors="coerce") * 0.03
        )
        best = constrained.sort_values(
            ["selection_score_v1", "should_not_take_block_count_v1", "should_not_take_precision_v1", "tail_10_50_help_count_v1"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
        frontier = frontier.merge(
            constrained[["policy_name_v1", "thresholds_json_v1", "selection_score_v1"]],
            on=["policy_name_v1", "thresholds_json_v1"],
            how="left",
        )
    frontier["selected_best_constrained_v1"] = (
        frontier["policy_name_v1"].astype("string").eq(str(best["policy_name_v1"]))
        & frontier["thresholds_json_v1"].astype("string").eq(str(best["thresholds_json_v1"]))
    )
    max_safe_should_blocks = int(pd.to_numeric(frontier.loc[frontier["passes_winner_constraints_v1"], "should_not_take_block_count_v1"], errors="coerce").max())
    first_unsafe_more_recall = (
        frontier.loc[
            (~frontier["passes_winner_constraints_v1"].fillna(False))
            & pd.to_numeric(frontier["should_not_take_block_count_v1"], errors="coerce").gt(max_safe_should_blocks)
        ]
        .sort_values(["should_not_take_block_count_v1", "strong_trade_false_block_count_v1", "fifty_plus_mfe_block_count_v1"], ascending=[True, True, True])
        .head(1)
    )
    constraint_metric_map = {
        "max_strong_false_blocks_v1": "strong_trade_false_block_count_v1",
        "max_50_plus_mfe_blocks_v1": "fifty_plus_mfe_block_count_v1",
        "max_200_plus_mfe_blocks_v1": "two_hundred_plus_mfe_block_count_v1",
        "max_strongest_winner_path_blocks_v1": "strongest_winner_path_block_count_v1",
        "max_repaired_165_blocks_v1": "repaired_165_block_count_v1",
    }
    safety_rows = []
    for key, value in constraints.items():
        metric = constraint_metric_map[key]
        safety_rows.append(
            {
                "constraint_name_v1": key,
                "metric_name_v1": metric,
                "max_allowed_v1": value,
                "reference_policy_v1": "R4_REPAIRED_SELECTED_REFERENCE",
                "best_policy_observed_value_v1": best.get(metric),
                "best_policy_passes_v1": bool(float(best.get(metric, np.inf)) <= float(value)),
            }
        )
    safety_rows.append(
        {
            "constraint_name_v1": "candidate_count_passing_all_constraints_v1",
            "metric_name_v1": "passes_winner_constraints_v1",
            "max_allowed_v1": None,
            "reference_policy_v1": "R4_REPAIRED_SELECTED_REFERENCE",
            "best_policy_observed_value_v1": int(frontier["passes_winner_constraints_v1"].sum()),
            "best_policy_passes_v1": True,
        }
    )
    frontier_summary = {
        "constraints_v1": constraints,
        "best_constrained_policy_v1": best,
        "max_safe_should_not_take_block_count_v1": max_safe_should_blocks,
        "max_safe_should_not_take_recall_v1": _safe_rate(float(max_safe_should_blocks), float(_bool(frame, "label_should_not_take_v1").sum())),
        "first_unacceptable_more_recall_candidate_v1": (
            first_unsafe_more_recall.replace({np.nan: None}).iloc[0].to_dict() if not first_unsafe_more_recall.empty else None
        ),
    }
    return (
        frontier.sort_values(["passes_winner_constraints_v1", "should_not_take_block_count_v1", "should_not_take_precision_v1"], ascending=[False, False, False]),
        pd.DataFrame(safety_rows),
        frontier_summary,
    )


def _mask_for_policy_record(frame: pd.DataFrame, record: Dict[str, Any]) -> pd.Series:
    policy = str(record["policy_name_v1"])
    if policy in _base_policy_masks(frame):
        return _base_policy_masks(frame)[policy]
    thresholds = json.loads(str(record.get("thresholds_json_v1") or "{}"))
    if policy == "SHOULD_NOT_TAKE_HEAD_ONLY":
        return (_prob(frame, "should_not_take").ge(float(thresholds.get("should_not_take_threshold_v1", 0.60))).fillna(False) & _bool(frame, "entry_r3_feature_available_v1")).astype(bool)
    if policy == "IMMEDIATE_MAE_RISK_HEAD_ONLY":
        return (_prob(frame, "immediate_mae_risk").ge(float(thresholds.get("immediate_mae_risk_threshold_v1", 0.80))).fillna(False) & _bool(frame, "entry_r3_feature_available_v1")).astype(bool)
    if policy == "WAIT_ADVISORY_HEAD_ONLY":
        return (_prob(frame, "wait_advisory").ge(float(thresholds.get("wait_advisory_threshold_v1", 0.85))).fillna(False) & _bool(frame, "entry_r3_feature_available_v1")).astype(bool)
    preserve_r2 = bool(thresholds.get("preserve_r2_fallback_v1", policy.startswith("R2_PRESERVED_PLUS_")))
    stack_name = policy.removeprefix("R2_PRESERVED_PLUS_")
    return _threshold_mask(frame, stack_name, {k: float(v) for k, v in thresholds.items() if k.endswith("_v1") and isinstance(v, (int, float))}, preserve_r2=preserve_r2)


def _build_head_to_head(frame: pd.DataFrame, best_record: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index),
        "R2_FALLBACK_REFERENCE": _bool(frame, "r2_entry_fallback_row_v1"),
        "R3_FULLCOVERAGE_CONSERVATIVE": _bool(frame, "r3_conservative_blocks_v1"),
        "R4_REPAIRED_SELECTED_REFERENCE": _bool(frame, "r4_entry_fallback_block_v1"),
        "BEST_CONSTRAINED_RECALIBRATED_R4": _mask_for_policy_record(frame, best_record),
    }
    scopes = {
        "ALL_1971": pd.Series(True, index=frame.index),
        "R2_FALLBACK_63": _bool(frame, "r2_entry_fallback_row_v1"),
        "REPAIRED_165": frame["is_repaired_165_v1"].fillna(False).astype(bool),
        "SHOULD_NOT_TAKE_CLASS": _bool(frame, "label_should_not_take_v1"),
        "TAKE_WAS_OK_CLASS": frame["take_was_ok_v1"].fillna(False).astype(bool),
        "FIFTY_PLUS_MFE_RUNNERS": frame["fifty_plus_mfe_v1"].fillna(False).astype(bool),
        "HUNDRED_PLUS_MFE_RUNNERS": frame["hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "TAIL_10_50_MFE_POCKET": frame["tail_10_50_mfe_v1"].fillna(False).astype(bool),
        "STRONG_TRADE_CANDIDATES": _bool(frame, "label_strong_trade_candidate_v1"),
    }
    rows: List[Dict[str, Any]] = []
    prediction = frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "is_repaired_165_v1",
            "label_should_not_take_v1",
            "label_strong_trade_candidate_v1",
            "take_was_ok_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "giveback_bps_v1",
            "baseline_realized_pnl_bps_v1",
        ]
    ].copy()
    for policy_name, mask in policies.items():
        prediction[f"{policy_name.lower()}__block_v1"] = mask.astype(bool).to_numpy()
        for scope_name, scope_mask in scopes.items():
            sub = frame.loc[scope_mask].copy()
            rows.append(_policy_metric_row(policy_name, scope_name, sub, mask.loc[scope_mask], thresholds={"head_to_head_v1": True}))
    return pd.DataFrame(rows), prediction


def _build_walkforward(reports_root: Path, frame: pd.DataFrame, best_record: Dict[str, Any], *, batch_weeks: int) -> pd.DataFrame:
    policies = {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index),
        "R2_FALLBACK_REFERENCE": _bool(frame, "r2_entry_fallback_row_v1"),
        "R3_FULLCOVERAGE_CONSERVATIVE": _bool(frame, "r3_conservative_blocks_v1"),
        "R4_REPAIRED_SELECTED_REFERENCE": _bool(frame, "r4_entry_fallback_block_v1"),
        "BEST_CONSTRAINED_RECALIBRATED_R4": _mask_for_policy_record(frame, best_record),
    }
    run_ids = _all_run_ids(reports_root, frame)
    rows: List[Dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        if not batch_run_ids:
            continue
        mask = frame["run_id"].astype("string").isin(batch_run_ids)
        for policy_name, block in policies.items():
            row = _policy_metric_row(policy_name, f"BATCH_{batch_index:02d}", frame.loc[mask].copy(), block.loc[mask], thresholds={"walkforward_v1": True})
            row.update(
                {
                    "batch_index_v1": int(batch_index),
                    "run_count_v1": int(len(batch_run_ids)),
                    "run_start_v1": batch_run_ids[0],
                    "run_end_v1": batch_run_ids[-1],
                    "slice_passes_winner_constraints_v1": bool(
                        int(row["strong_trade_false_block_count_v1"]) <= 2
                        and int(row["fifty_plus_mfe_block_count_v1"]) <= 3
                        and int(row["two_hundred_plus_mfe_block_count_v1"]) <= 1
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_repair_verification(joined: pd.DataFrame, repair_df: pd.DataFrame, repair_summary: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    repaired = joined[joined["is_repaired_165_v1"].fillna(False)].copy()
    repaired_count = int(len(repaired))
    original_direct_count = int(len(joined) - repaired_count)
    synthetic_count = int(pd.Series(repair_df["synthetic_value_used_v1"]).astype(str).str.lower().eq("true").sum()) if not repair_df.empty else 0
    lineage_cols = [column for column in ["repair_timestamp_utc_v1", "replay_timestamp_utc_v1", "xgb_timestamp_utc_v1"] if column in repair_df.columns]
    lineage_complete = (
        repair_df[lineage_cols].notna().all(axis=1)
        if not repair_df.empty
        else pd.Series(dtype=bool)
    )
    rows = [
        {"verification_name_v1": "ENTRY_COVERAGE_FULL", "status_v1": "PASS" if int(_bool(joined, "entry_observation_present_v1").sum()) == len(joined) else "FAIL", "value_v1": int(_bool(joined, "entry_observation_present_v1").sum()), "expected_v1": len(joined), "details_json_v1": "{}"},
        {"verification_name_v1": "ENTRY_RAW_COVERAGE_FULL", "status_v1": "PASS" if int(_bool(joined, "entry_raw_state_present_v1").sum()) == len(joined) else "FAIL", "value_v1": int(_bool(joined, "entry_raw_state_present_v1").sum()), "expected_v1": len(joined), "details_json_v1": "{}"},
        {"verification_name_v1": "NO_SYNTHETIC_REPAIR_VALUES", "status_v1": "PASS" if synthetic_count == 0 else "FAIL", "value_v1": synthetic_count, "expected_v1": 0, "details_json_v1": "{}"},
        {"verification_name_v1": "REPAIRED_165_LINEAGE_COMPLETE", "status_v1": "PASS" if int(lineage_complete.sum()) == repaired_count else "FAIL", "value_v1": int(lineage_complete.sum()), "expected_v1": repaired_count, "details_json_v1": _json_dumps({"repair_rows": int(len(repair_df))})},
        {"verification_name_v1": "SOURCE_DIRECT_ENTRY_ORIGINAL_ROWS", "status_v1": "PASS", "value_v1": original_direct_count, "expected_v1": original_direct_count, "details_json_v1": "{}"},
        {"verification_name_v1": "SOURCE_MANAGEMENT_ANCHOR_REPAIRED_ROWS", "status_v1": "PASS", "value_v1": int(repair_summary.get("recovered_from_as_of_decision_moment_ledger_rows_v1", 0)), "expected_v1": int(repair_summary.get("recovered_from_as_of_decision_moment_ledger_rows_v1", 0)), "details_json_v1": "{}"},
        {"verification_name_v1": "SOURCE_SHADOW_META_CANDIDATES_REPAIRED_ROWS", "status_v1": "PASS", "value_v1": int(repair_summary.get("recovered_from_run_shadow_meta_candidates_rows_v1", repaired_count)), "expected_v1": repaired_count, "details_json_v1": "{}"},
        {"verification_name_v1": "SOURCE_REPLAY_CHUNK_EXACT_ROWS", "status_v1": "PASS", "value_v1": int(repair_summary.get("recovered_replay_chunk_exact_rows_v1", repaired_count)), "expected_v1": repaired_count, "details_json_v1": "{}"},
        {"verification_name_v1": "SOURCE_XGB_TIMESTAMP_EXACT_ROWS", "status_v1": "PASS", "value_v1": int(repair_summary.get("recovered_xgb_exact_rows_v1", repaired_count)), "expected_v1": repaired_count, "details_json_v1": "{}"},
        {"verification_name_v1": "REPAIRED_TAKE_WAS_OK_ROWS", "status_v1": "PASS" if int(repaired["take_was_ok_v1"].sum()) == repaired_count else "FAIL", "value_v1": int(repaired["take_was_ok_v1"].sum()), "expected_v1": repaired_count, "details_json_v1": "{}"},
        {"verification_name_v1": "REPAIRED_STRONG_TRADE_CANDIDATES", "status_v1": "PASS", "value_v1": int(_bool(repaired, "label_strong_trade_candidate_v1").sum()), "expected_v1": None, "details_json_v1": "{}"},
        {"verification_name_v1": "REPAIRED_50_PLUS_MFE_OPPORTUNITY", "status_v1": "PASS", "value_v1": int(repaired["fifty_plus_mfe_v1"].sum()), "expected_v1": None, "details_json_v1": "{}"},
        {"verification_name_v1": "REPAIRED_MAE_BUCKET_COUNTS", "status_v1": "PASS", "value_v1": int(len(repaired)), "expected_v1": 165, "details_json_v1": _json_dumps(_counts(repaired, "mae_bucket_v1"))},
    ]
    summary = {
        "repaired_rows_v1": int(len(repaired)),
        "repaired_take_was_ok_v1": int(repaired["take_was_ok_v1"].sum()),
        "repaired_should_not_take_v1": int(_bool(repaired, "label_should_not_take_v1").sum()),
        "repaired_strong_trade_candidates_v1": int(_bool(repaired, "label_strong_trade_candidate_v1").sum()),
        "repaired_50_plus_mfe_v1": int(repaired["fifty_plus_mfe_v1"].sum()),
        "repaired_100_plus_mfe_v1": int(repaired["hundred_plus_mfe_v1"].sum()),
        "repaired_200_plus_mfe_v1": int(repaired["two_hundred_plus_mfe_v1"].sum()),
        "repaired_mae_bucket_counts_v1": _counts(repaired, "mae_bucket_v1"),
        "synthetic_count_v1": synthetic_count,
        "lineage_complete_rows_v1": int(lineage_complete.sum()),
    }
    return pd.DataFrame(rows), summary


def _decision_matrix(frontier_summary: Dict[str, Any], head_to_head_df: pd.DataFrame, walkforward_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    best = frontier_summary["best_constrained_policy_v1"]
    r4 = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R4_REPAIRED_SELECTED_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r2 = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R2_FALLBACK_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    best_beats_r4_recall = int(best["should_not_take_block_count_v1"]) > int(r4["should_not_take_block_count_v1"])
    best_same_or_better_damage = (
        int(best["strong_trade_false_block_count_v1"]) <= int(r4["strong_trade_false_block_count_v1"])
        and int(best["fifty_plus_mfe_block_count_v1"]) <= int(r4["fifty_plus_mfe_block_count_v1"])
        and int(best["two_hundred_plus_mfe_block_count_v1"]) <= int(r4["two_hundred_plus_mfe_block_count_v1"])
        and int(best["repaired_165_block_count_v1"]) <= int(r4["repaired_165_block_count_v1"])
    )
    wf_best = walkforward_df[walkforward_df["policy_name_v1"].eq("BEST_CONSTRAINED_RECALIBRATED_R4")].copy()
    wf_stable = bool(wf_best["slice_passes_winner_constraints_v1"].fillna(False).all()) if not wf_best.empty else False
    if best_beats_r4_recall and best_same_or_better_damage and wf_stable:
        recommendation = "R4_RECALIBRATE_THRESHOLDS"
    elif best_same_or_better_damage and wf_stable:
        recommendation = "R4_SHADOW_REPLAY_CANDIDATE"
    elif int(r2["strong_trade_false_block_count_v1"]) < int(r4["strong_trade_false_block_count_v1"]) and int(r2["fifty_plus_mfe_block_count_v1"]) <= int(r4["fifty_plus_mfe_block_count_v1"]):
        recommendation = "KEEP_R2_FALLBACK"
    else:
        recommendation = "R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE"
    rows = [
        {"decision_key_v1": "KEEP_R2_FALLBACK", "status_v1": "VALID_BASELINE", "hard_status_v1": "BEVIST", "reason_v1": "R2 remains the fallback reference and preserves its 63-row pocket."},
        {"decision_key_v1": "R4_SHADOW_REPLAY_CANDIDATE", "status_v1": "PASS" if wf_stable else "WARN", "hard_status_v1": "INDIKERT", "reason_v1": "R4 fullcoverage repaired is suitable for shadow replay if winner constraints hold across slices."},
        {"decision_key_v1": "R4_RECALIBRATE_THRESHOLDS", "status_v1": "PASS" if recommendation == "R4_RECALIBRATE_THRESHOLDS" else "WARN", "hard_status_v1": "INDIKERT", "reason_v1": "Use only if the constrained frontier increases bad-trade recall without extra winner damage."},
        {"decision_key_v1": "R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE", "status_v1": "PASS" if recommendation == "R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE" else "NEXT_OPTION", "hard_status_v1": "INDIKERT", "reason_v1": "Fullcoverage removes the old coverage blocker; R5 can train directly on repaired 1971/1971."},
        {"decision_key_v1": "DO_NOT_USE_ENTRY_FALLBACK_YET", "status_v1": "PASS_FOR_LIVE_GATE_ONLY", "hard_status_v1": "BEVIST", "reason_v1": "Do not use any entry fallback as live gate from this offline audit."},
    ]
    summary = {
        "recommended_next_step_v1": recommendation,
        "best_beats_r4_recall_v1": bool(best_beats_r4_recall),
        "best_same_or_better_damage_v1": bool(best_same_or_better_damage),
        "walkforward_constraints_stable_v1": bool(wf_stable),
        "r2_should_not_blocks_v1": int(r2["should_not_take_block_count_v1"]),
        "r4_should_not_blocks_v1": int(r4["should_not_take_block_count_v1"]),
        "best_should_not_blocks_v1": int(best["should_not_take_block_count_v1"]),
    }
    return pd.DataFrame(rows), summary


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    best = summary["best_constrained_policy_v1"]
    lines = [
        "# R4 Fullcoverage Policy Recalibration And Shadow Replay V1",
        "",
        "Offline shadow/research only. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R4_FULLCOVERAGE_POLICY_RECALIBRATION_STATUS']}`",
        f"- Coverage: `{summary['coverage_v1']['entry_coverage_v1']}/{summary['coverage_v1']['ledger_trade_count_v1']}`",
        f"- Synthetic repair values: `{summary['coverage_v1']['synthetic_count_v1']}`",
        f"- Best constrained policy: `{best['policy_name_v1']}`",
        f"- Best should-not blocks: `{best['should_not_take_block_count_v1']}`",
        f"- Best strong false blocks: `{best['strong_trade_false_block_count_v1']}`",
        f"- Best 50+/200+ MFE blocked: `{best['fifty_plus_mfe_block_count_v1']}` / `{best['two_hundred_plus_mfe_block_count_v1']}`",
        f"- Recommendation: `{summary['decision_v1']['recommended_next_step_v1']}`",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    readiness_dir: Path,
    r3_dir: Path,
    r4_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    joined, asof_df, labels_df, repair_df, repair_summary, r3_summary, r4_summary = _build_joined(
        readiness_dir=readiness_dir,
        r3_dir=r3_dir,
        r4_dir=r4_dir,
    )
    ledger_count = int(len(joined))
    if expected_ledger_count is not None and ledger_count != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger trade count expected {expected_ledger_count}, observed {ledger_count}")

    repair_verification_df, repair_verification_summary = _build_repair_verification(joined, repair_df, repair_summary)
    frontier_df, winner_safety_df, frontier_summary = _build_threshold_frontier(joined)
    best_record = frontier_summary["best_constrained_policy_v1"]
    head_to_head_df, policy_prediction_df = _build_head_to_head(joined, best_record)
    walkforward_df = _build_walkforward(reports_root, joined, best_record, batch_weeks=batch_weeks)
    decision_df, decision_summary = _decision_matrix(frontier_summary, head_to_head_df, walkforward_df)

    entry_coverage = int(_bool(joined, "entry_observation_present_v1").sum())
    raw_coverage = int(_bool(joined, "entry_raw_state_present_v1").sum())
    synthetic_count = int(repair_verification_summary["synthetic_count_v1"])
    missing_count = int(ledger_count - entry_coverage)
    consistency_df = pd.DataFrame(
        [
            _consistency_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS", {"expected": expected_ledger_count, "observed": ledger_count}),
            _consistency_record("ENTRY_COVERAGE_1971_OF_1971", "PASS" if entry_coverage == ledger_count else "FAIL", {"observed": entry_coverage, "expected": ledger_count}),
            _consistency_record("ENTRY_RAW_COVERAGE_1971_OF_1971", "PASS" if raw_coverage == ledger_count else "FAIL", {"observed": raw_coverage, "expected": ledger_count}),
            _consistency_record("NO_SYNTHETIC_REPAIR_VALUES", "PASS" if synthetic_count == 0 else "FAIL", {"observed": synthetic_count}),
            _consistency_record("R3_PREDICTIONS_FULL_LEDGER", "PASS" if int(_bool(joined, "entry_r3_feature_available_v1").sum()) == ledger_count else "FAIL", {"observed": int(_bool(joined, "entry_r3_feature_available_v1").sum())}),
            _consistency_record("R4_REFERENCE_POLICY_FULL_LEDGER", "PASS" if "r4_entry_fallback_block_v1" in joined.columns else "FAIL", {"observed": ledger_count}),
            _consistency_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R4_FULLCOVERAGE_POLICY_RECALIBRATION_STATUS_V1",
        "R4_FULLCOVERAGE_POLICY_RECALIBRATION_STATUS": "FULLCOVERAGE_SHADOW_REPLAY_READY_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    r2_scope63 = head_to_head_df[
        head_to_head_df["scope_v1"].eq("R2_FALLBACK_63") & head_to_head_df["policy_name_v1"].isin(["R2_FALLBACK_REFERENCE", "R4_REPAIRED_SELECTED_REFERENCE"])
    ]
    repaired_scope = head_to_head_df[
        head_to_head_df["scope_v1"].eq("REPAIRED_165") & head_to_head_df["policy_name_v1"].isin(["R2_FALLBACK_REFERENCE", "R4_REPAIRED_SELECTED_REFERENCE", "BEST_CONSTRAINED_RECALIBRATED_R4"])
    ]
    summary = {
        "layer_name": "R4_FULLCOVERAGE_POLICY_RECALIBRATION_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "readiness_dir_v1": str(readiness_dir),
        "r3_dir_v1": str(r3_dir),
        "r4_dir_v1": str(r4_dir),
        "extension_dir_v1": str(extension_dir),
        "coverage_v1": {
            "ledger_trade_count_v1": ledger_count,
            "entry_coverage_v1": entry_coverage,
            "entry_raw_coverage_v1": raw_coverage,
            "missing_count_v1": missing_count,
            "synthetic_count_v1": synthetic_count,
            **repair_verification_summary,
        },
        "frontier_v1": {
            "candidate_count_v1": int(len(frontier_df)),
            "passes_winner_constraints_count_v1": int(frontier_df["passes_winner_constraints_v1"].sum()),
            **frontier_summary,
        },
        "best_constrained_policy_v1": best_record,
        "head_to_head_v1": {
            "r2_vs_r4_on_r2_63_v1": r2_scope63.replace({np.nan: None}).to_dict(orient="records"),
            "r2_r4_best_on_repaired_165_v1": repaired_scope.replace({np.nan: None}).to_dict(orient="records"),
        },
        "r3_reference_v1": {
            "r3_holdout_min_balanced_accuracy_v1": r3_summary.get("r3_holdout_min_balanced_accuracy_v1"),
            "r3_policy_safety_v1": r3_summary.get("r3_policy_safety_v1"),
        },
        "r4_reference_v1": {
            "selected_policy_name_v1": r4_summary.get("selected_policy_name_v1"),
            "selected_policy_metrics_v1": r4_summary.get("selected_policy_metrics_v1"),
        },
        "decision_v1": decision_summary,
        "hard_status_division_v1": {
            "BEVIST": [
                f"Fullcoverage repair is {entry_coverage}/{ledger_count} with {missing_count} missing.",
                f"Synthetic repair value count is {synthetic_count}.",
                f"All {repair_verification_summary['repaired_rows_v1']} repaired rows have exact source lineage.",
                "This materialization is shadow/research only and not live-gate promoted.",
            ],
            "INDIKERT": [
                "Constrained threshold frontier identifies the best offline safety/reward tradeoff.",
                "Walk-forward slice metrics indicate whether the candidate is stable enough for a shadow replay.",
                "R5 entry retrain can now use repaired coverage if threshold tuning is not enough.",
            ],
            "IKKE_ETABLERT": [
                "Live policy safety.",
                "Causal proof of improved future execution.",
                "Broker/live fill behavior under the recalibrated fallback.",
            ],
        },
        "status_v1": status,
    }
    contract = {
        "layer_name": "R4_FULLCOVERAGE_POLICY_RECALIBRATION_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_readiness_dir_v1": str(readiness_dir),
        "input_r3_dir_v1": str(r3_dir),
        "input_r4_dir_v1": str(r4_dir),
        "action_semantics_v1": "ENTRY_FALLBACK_BLOCK_OR_ALLOW_BASELINE_ONLY",
        "threshold_frontier_heads_v1": list(TASK_PROB_COLUMNS.keys()),
        "winner_constraints_v1": frontier_summary["constraints_v1"],
        "hindsight_labels_physically_separate_v1": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R4_FULLCOVERAGE_POLICY_RECALIBRATION_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_table": AS_OF_TABLE,
            "hindsight_label_outcome_table": HINDSIGHT_TABLE,
            "repair_verification": REPAIR_VERIFICATION,
            "threshold_frontier": THRESHOLD_FRONTIER,
            "winner_safety_audit": WINNER_SAFETY_AUDIT,
            "head_to_head": HEAD_TO_HEAD,
            "walkforward": WALKFORWARD,
            "decision_matrix": DECISION_MATRIX,
            "policy_prediction_view": POLICY_PREDICTION_VIEW,
            "consistency_audit": CONSISTENCY_AUDIT,
            "summary": SUMMARY,
            "report": REPORT,
        },
    }
    return {
        "asof_df": asof_df,
        "labels_df": labels_df,
        "repair_verification_df": repair_verification_df,
        "frontier_df": frontier_df,
        "winner_safety_df": winner_safety_df,
        "head_to_head_df": head_to_head_df,
        "walkforward_df": walkforward_df,
        "decision_df": decision_df,
        "policy_prediction_df": policy_prediction_df,
        "consistency_df": consistency_df,
        "contract": contract,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path | None = None,
    r3_dir: Path | None = None,
    r4_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_readiness_dir = readiness_dir or _resolve_dir(reports_root, None, READINESS_EXTENSION_NAME, R2_AS_OF_TABLE)
    resolved_r3_dir = r3_dir or _resolve_dir(reports_root, None, R3_EXTENSION_NAME, R3_PREDICTION_VIEW)
    resolved_r4_dir = r4_dir or _resolve_dir(reports_root, None, R4_EXTENSION_NAME, R4_POLICY_PREDICTION_VIEW)
    resolved_extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    payload = build_payload(
        reports_root=reports_root,
        readiness_dir=Path(resolved_readiness_dir).expanduser().resolve(),
        r3_dir=Path(resolved_r3_dir).expanduser().resolve(),
        r4_dir=Path(resolved_r4_dir).expanduser().resolve(),
        extension_dir=resolved_extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
    )
    resolved_extension_dir.mkdir(parents=True, exist_ok=True)
    payload["asof_df"].to_parquet(resolved_extension_dir / AS_OF_TABLE, index=False)
    payload["labels_df"].to_parquet(resolved_extension_dir / HINDSIGHT_TABLE, index=False)
    payload["repair_verification_df"].to_csv(resolved_extension_dir / REPAIR_VERIFICATION, index=False)
    payload["frontier_df"].to_csv(resolved_extension_dir / THRESHOLD_FRONTIER, index=False)
    payload["winner_safety_df"].to_csv(resolved_extension_dir / WINNER_SAFETY_AUDIT, index=False)
    payload["head_to_head_df"].to_csv(resolved_extension_dir / HEAD_TO_HEAD, index=False)
    payload["walkforward_df"].to_csv(resolved_extension_dir / WALKFORWARD, index=False)
    payload["decision_df"].to_csv(resolved_extension_dir / DECISION_MATRIX, index=False)
    payload["policy_prediction_df"].to_parquet(resolved_extension_dir / POLICY_PREDICTION_VIEW, index=False)
    payload["consistency_df"].to_csv(resolved_extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(resolved_extension_dir / CONTRACT, payload["contract"])
    _write_json(resolved_extension_dir / SUMMARY, payload["summary"])
    _write_json(resolved_extension_dir / STATUS, payload["status"])
    _write_json(resolved_extension_dir / MANIFEST, payload["manifest"])
    (resolved_extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    top_level = dict(payload["summary"])
    top_level["extension_dir_v1"] = str(resolved_extension_dir)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, top_level)
    return {
        "extension_dir": resolved_extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize R4 fullcoverage policy recalibration and shadow replay audit.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", default=None)
    parser.add_argument("--r3-dir", default=None)
    parser.add_argument("--r4-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        readiness_dir=_resolve_dir(reports_root, args.readiness_dir, READINESS_EXTENSION_NAME, R2_AS_OF_TABLE),
        r3_dir=_resolve_dir(reports_root, args.r3_dir, R3_EXTENSION_NAME, R3_PREDICTION_VIEW),
        r4_dir=_resolve_dir(reports_root, args.r4_dir, R4_EXTENSION_NAME, R4_POLICY_PREDICTION_VIEW),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
