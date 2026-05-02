#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _bool,
    _json_dumps,
    _load_json,
    _num,
    _policy_metric_row,
    _safe_rate,
    _write_json,
)
from gx1.scripts.materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import (
    EXTENSION_NAME as FREEZE_EXTENSION_NAME,
    FREEZE_MANIFEST as FREEZE_MANIFEST_JSON,
    GO_NO_GO_MATRIX as FREEZE_GO_NO_GO_MATRIX,
    POLICY_LOGGING_LOCK as FREEZE_POLICY_LOGGING_LOCK,
    R6_TRAINING_TARGET_SPEC as FREEZE_R6_TRAINING_TARGET_SPEC,
    SUMMARY as FREEZE_SUMMARY,
)
from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import (
    AS_OF_TABLE as PHASE_AS_OF_TABLE,
    HINDSIGHT_TABLE as PHASE_HINDSIGHT_TABLE,
    SHADOW_REPLAY_BAKEOFF as PHASE_SHADOW_REPLAY_BAKEOFF,
)
from gx1.scripts.train_r3_entry_label_feature_retrain_v1 import _fit_preprocessor, _transform_features
from gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 import (
    BAD_PROB as R5_2_BAD_PROB,
    POLICY_PREDICTION_VIEW as R5_2_POLICY_PREDICTION_VIEW,
    RUNNER_PROB as R5_2_RUNNER_PROB,
)
from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import R5_PROB, _slice_masks


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

CONTRACT = "shadow_meta_all_trade_review_r6_entry_runner_first_contract_v1.json"
AS_OF_FEATURE_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
HINDSIGHT_LABEL_OUTCOME_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
RUNNER_LABEL_AUDIT = "shadow_meta_all_trade_review_r6_runner_protector_label_audit_v1.csv"
BAD_RISK_LABEL_AUDIT = "shadow_meta_all_trade_review_r6_bad_risk_label_audit_v1.csv"
TAIL_CONTROL_AUDIT = "shadow_meta_all_trade_review_r6_tail_control_audit_v1.csv"
FEATURE_PATH_DYNAMICS_AUDIT = "shadow_meta_all_trade_review_r6_feature_path_dynamics_audit_v1.csv"
MODEL_FAMILY_BAKEOFF = "shadow_meta_all_trade_review_r6_model_family_bakeoff_v1.csv"
THRESHOLD_CALIBRATION = "shadow_meta_all_trade_review_r6_threshold_calibration_v1.csv"
WALKFORWARD_METRICS = "shadow_meta_all_trade_review_r6_walkforward_metrics_v1.csv"
LOSO_METRICS = "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv"
ROLLING_WINDOW_METRICS = "shadow_meta_all_trade_review_r6_rolling_window_metrics_v1.csv"
HEAD_TO_HEAD = "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
DECISION_MATRIX = "shadow_meta_all_trade_review_r6_phase_gate_decision_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r6_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r6_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r6_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r6_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r6_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r6_entry_runner_first_retrain_v1.json"

R6_BAD_PROB = "pred__entry_r6_bad_risk__prob_true_v1"
R6_RUNNER_PROB = "pred__entry_r6_runner_protector__prob_true_v1"
R6_TAIL_PROB = "pred__entry_r6_tail_control_10_50__prob_true_v1"
R6_RISKY_PROB = "pred__entry_r6_risky_allow__prob_true_v1"
R6_BLINDSPOT_PROB = "pred__entry_r6_batch04_blindspot__prob_true_v1"

R5_2_BASELINE = {
    "bad_blocks_v1": 106,
    "tail_help_v1": 82,
    "global_precision_v1": 0.9724770642201835,
    "worst_loso_precision_v1": 0.9285714285714286,
    "fifty_plus_mfe_blocked_v1": 1,
    "hundred_plus_mfe_blocked_v1": 0,
    "two_hundred_plus_mfe_blocked_v1": 0,
    "repaired_165_blocked_v1": 0,
    "strong_false_blocks_v1": 0,
    "strongest_winner_path_blocked_v1": 0,
}


@dataclass(frozen=True)
class R6HeadSpec:
    head_id: str
    label_col: str
    output_col: str
    role: str


@dataclass(frozen=True)
class R6Candidate:
    policy_name: str
    family: str
    bad_threshold: float
    runner_threshold: float
    tail_threshold: float
    risky_threshold: float
    blindspot_threshold: float
    r5_2_runner_threshold: float
    use_r5_2_base: bool
    hard_asof_runner_guard: bool


HEAD_SPECS = (
    R6HeadSpec("bad_risk", "r6_label_bad_risk_v1", R6_BAD_PROB, "BAD_RISK_BLOCKER"),
    R6HeadSpec("runner_protector", "r6_label_runner_protect_v1", R6_RUNNER_PROB, "RUNNER_PROTECTOR_FIRST"),
    R6HeadSpec("tail_control_10_50", "r6_label_tail_control_10_50_v1", R6_TAIL_PROB, "TAIL_CONTROL"),
    R6HeadSpec("risky_allow", "r6_label_risky_allow_v1", R6_RISKY_PROB, "RISKY_ALLOW_RECALL"),
    R6HeadSpec("batch04_blindspot", "r6_label_batch04_blindspot_v1", R6_BLINDSPOT_PROB, "BATCH04_BLINDSPOT"),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_freeze_dir(reports_root: Path, freeze_dir_arg: str | None) -> Path:
    path = Path(freeze_dir_arg).expanduser().resolve() if freeze_dir_arg else reports_root / FREEZE_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R5.2 freeze dir does not exist: {path}")
    for artifact in [FREEZE_SUMMARY, FREEZE_MANIFEST_JSON, FREEZE_POLICY_LOGGING_LOCK, FREEZE_R6_TRAINING_TARGET_SPEC]:
        if not (path / artifact).exists():
            raise FileNotFoundError(f"{path} missing required freeze artifact {artifact}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_ready(value: Any) -> Any:
    if value is None:
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
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if "r5_2" in lower or "entry_r5" in lower or "blocker_score" in lower or "runner_protector_score" in lower:
        return "r5_2_score_context"
    if any(token in lower for token in ["swing", "retracement", "ema", "kama", "trend", "impulse", "dist_last"]):
        return "structure_swing_retracement"
    if any(token in lower for token in ["atr", "vol", "range", "bandwidth", "squeeze", "compression"]):
        return "volatility_range"
    if any(token in lower for token in ["close_in_bar", "clv", "body", "wick"]):
        return "close_in_bar_timing"
    if any(token in lower for token in ["session", "hour", "weekday", "minutes"]):
        return "session_time_context"
    if any(token in lower for token in ["window_ret", "up_move", "down_move", "directional_imbalance", "micro_momentum", "micro_acceleration"]):
        return "prior_path_impulse_context"
    if any(token in lower for token in ["spread", "cost"]):
        return "spread_liquidity_cost"
    if any(token in lower for token in ["candidate", "tradable", "uncertainty", "path_quality", "margin", "pred_side", "xgb"]):
        return "entry_model_context"
    if any(token in lower for token in ["repair", "coverage"]):
        return "repaired_165_lineage"
    return "other_as_of"


def _run_sort_key(run_id: Any) -> str:
    text = str(run_id)
    parts = text.split("_")
    return parts[-2] if len(parts) >= 2 and parts[-2].isdigit() else text


def _all_run_ids(reports_root: Path, frame: pd.DataFrame) -> list[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted([path.name for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("E2E_SANITY_ORDERFIX_")], key=_run_sort_key)
        if run_ids:
            return run_ids
    return sorted(frame["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _batch_lookup(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> dict[str, str]:
    lookup: dict[str, str] = {}
    run_ids = _all_run_ids(reports_root, frame)
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        for run_id in run_ids[start : start + batch_weeks]:
            lookup[str(run_id)] = f"BATCH_{batch_index:02d}"
    return lookup


def _load_inputs(
    *,
    reports_root: Path,
    freeze_dir: Path,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path]:
    freeze_summary = _load_json(freeze_dir / FREEZE_SUMMARY)
    freeze_manifest = _load_json(freeze_dir / FREEZE_MANIFEST_JSON)
    target_spec = _load_json(freeze_dir / FREEZE_R6_TRAINING_TARGET_SPEC)
    phase_dir = Path(str(freeze_summary["phase_gate_dir_v1"])).expanduser().resolve()
    r5_2_source_dir = Path(str(freeze_summary["r5_2_source_dir_v1"])).expanduser().resolve()
    asof_df = pd.read_parquet(phase_dir / PHASE_AS_OF_TABLE)
    hindsight_df = pd.read_parquet(phase_dir / PHASE_HINDSIGHT_TABLE)
    policy_lock_df = pd.read_parquet(freeze_dir / FREEZE_POLICY_LOGGING_LOCK)
    r5_2_pred_df = pd.read_parquet(r5_2_source_dir / R5_2_POLICY_PREDICTION_VIEW)
    phase_bakeoff_df = pd.read_csv(phase_dir / PHASE_SHADOW_REPLAY_BAKEOFF)
    go_no_go_df = pd.read_csv(freeze_dir / FREEZE_GO_NO_GO_MATRIX)
    _require_columns(asof_df, ["candidate_uid", "run_id", "used_for_training", "used_for_validation", "used_for_holdout"], artifact_name=PHASE_AS_OF_TABLE)
    _require_columns(hindsight_df, ["candidate_uid", "baseline_realized_pnl_bps_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "giveback_bps_v1"], artifact_name=PHASE_HINDSIGHT_TABLE)
    _require_columns(policy_lock_df, ["candidate_uid", "blocker_score_v1", "runner_protector_score_v1", "r5_2_selected_candidate__block_v1"], artifact_name=FREEZE_POLICY_LOGGING_LOCK)
    _require_columns(
        r5_2_pred_df,
        [
            "candidate_uid",
            "r2_fallback_reference__block_v1",
            "r4_current_reference__block_v1",
            "r5_current_reference__block_v1",
            "r5_1_selected_reference__block_v1",
            "r5_2_selected_candidate__block_v1",
            "label_should_not_take_v1",
            "take_was_ok_v1",
            "label_strong_trade_candidate_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            R5_2_BAD_PROB,
            R5_2_RUNNER_PROB,
        ],
        artifact_name=R5_2_POLICY_PREDICTION_VIEW,
    )
    for name, frame in [(PHASE_AS_OF_TABLE, asof_df), (PHASE_HINDSIGHT_TABLE, hindsight_df), (FREEZE_POLICY_LOGGING_LOCK, policy_lock_df), (R5_2_POLICY_PREDICTION_VIEW, r5_2_pred_df)]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Expected locked ledger {expected_ledger_count}, observed {len(asof_df)}")
    if freeze_manifest.get("freeze_status_v1") != "FROZEN_SHADOW_FALLBACK_CANDIDATE_NOT_LIVE_GATE":
        raise RuntimeError("R6 requires frozen R5.2 shadow fallback benchmark")
    return asof_df, hindsight_df, policy_lock_df, r5_2_pred_df, phase_bakeoff_df, go_no_go_df, freeze_summary, freeze_manifest, target_spec, phase_dir, r5_2_source_dir


def _prepare_frame(
    *,
    reports_root: Path,
    asof_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
    policy_lock_df: pd.DataFrame,
    r5_2_pred_df: pd.DataFrame,
    batch_weeks: int,
) -> pd.DataFrame:
    pred_cols = [
        "candidate_uid",
        "r2_fallback_reference__block_v1",
        "r4_current_reference__block_v1",
        "r5_current_reference__block_v1",
        "r5_1_selected_reference__block_v1",
        "r5_2_selected_candidate__block_v1",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "r5_2_batch04_hard_negative_runner_v1",
        "r5_2_hard_negative_like_asof_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
    ]
    pred_cols += [column for column in r5_2_pred_df.columns if column.startswith("pred__entry_r5_")]
    pred_cols = list(dict.fromkeys(pred_cols))
    lock_cols = ["candidate_uid", "blocker_score_v1", "runner_protector_score_v1", "selected_action_v1", "block_reason_v1", "allow_reason_v1"]
    frame = (
        asof_df.merge(hindsight_df, on=["candidate_uid", "run_id", "trade_uid", "trade_id", "decision_timestamp"], how="inner", validate="one_to_one")
        .merge(r5_2_pred_df[[column for column in pred_cols if column in r5_2_pred_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(policy_lock_df[[column for column in lock_cols if column in policy_lock_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    frame["label_should_not_take_v1"] = _bool(frame, "label_should_not_take_v1")
    frame["take_was_ok_v1"] = _bool(frame, "take_was_ok_v1")
    frame["label_strong_trade_candidate_v1"] = _bool(frame, "label_strong_trade_candidate_v1")
    frame["fifty_plus_mfe_v1"] = _bool(frame, "fifty_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(50.0)
    frame["hundred_plus_mfe_v1"] = _bool(frame, "hundred_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(100.0)
    frame["two_hundred_plus_mfe_v1"] = _bool(frame, "two_hundred_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(200.0)
    frame["is_repaired_165_v1"] = _bool(frame, "entry_coverage_repair_applied_v1") | _bool(frame, "is_repaired_165_v1")
    frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
    )
    frame["strongest_winner_path_v1"] = frame["two_hundred_plus_mfe_v1"] | (
        frame["label_strong_trade_candidate_v1"] & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0) & frame["fifty_plus_mfe_v1"]
    )
    lookup = _batch_lookup(reports_root, frame, batch_weeks=batch_weeks)
    frame["batch_scope_v1"] = frame["run_id"].astype("string").map(lookup).fillna("BATCH_UNKNOWN")
    selected = _bool(frame, "r5_2_selected_candidate__block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    frame["r6_label_runner_50_mfe_v1"] = take_ok & frame["fifty_plus_mfe_v1"]
    frame["r6_label_runner_100_mfe_v1"] = take_ok & frame["hundred_plus_mfe_v1"]
    frame["r6_label_runner_200_mfe_v1"] = take_ok & frame["two_hundred_plus_mfe_v1"]
    frame["r6_label_repaired_165_like_runner_v1"] = take_ok & frame["is_repaired_165_v1"]
    frame["r6_label_strong_low_mae_runner_v1"] = take_ok & frame["label_strong_trade_candidate_v1"] & _num(frame, "mae_abs_bps_v1").le(25.0)
    frame["r6_label_high_mfe_low_giveback_v1"] = take_ok & frame["fifty_plus_mfe_v1"] & (
        _num(frame, "giveback_bps_v1").le(25.0) | _num(frame, "giveback_bps_v1").le(_num(frame, "peak_mfe_bps_v1") * 0.25)
    )
    frame["r6_label_runner_near_miss_v1"] = take_ok & frame["fifty_plus_mfe_v1"] & (
        pd.to_numeric(frame[R5_2_BAD_PROB], errors="coerce").ge(0.50).fillna(False)
        | pd.to_numeric(frame[R5_2_RUNNER_PROB], errors="coerce").lt(0.60).fillna(False)
        | selected
    )
    frame["r6_label_runner_protect_v1"] = (
        frame["r6_label_runner_50_mfe_v1"]
        | frame["r6_label_runner_100_mfe_v1"]
        | frame["r6_label_runner_200_mfe_v1"]
        | frame["r6_label_repaired_165_like_runner_v1"]
        | frame["r6_label_strong_low_mae_runner_v1"]
        | frame["r6_label_high_mfe_low_giveback_v1"]
        | frame["r6_label_runner_near_miss_v1"]
    )
    frame["r6_label_missed_should_not_take_v1"] = should & ~selected
    frame["r6_label_risky_allow_v1"] = frame["r6_label_missed_should_not_take_v1"] & (
        _num(frame, "mae_abs_bps_v1").ge(40.0)
        | _num(frame, "baseline_realized_pnl_bps_v1").le(-25.0)
        | pd.to_numeric(frame[R5_2_BAD_PROB], errors="coerce").ge(0.60).fillna(False)
    )
    frame["r6_label_high_mae_low_mfe_v1"] = should & _num(frame, "mae_abs_bps_v1").ge(40.0) & _num(frame, "peak_mfe_bps_v1").lt(50.0)
    frame["r6_label_low_mfe_low_value_v1"] = should & _num(frame, "peak_mfe_bps_v1").lt(10.0) & _num(frame, "baseline_realized_pnl_bps_v1").le(0.0)
    frame["r6_label_early_adverse_excursion_v1"] = should & _num(frame, "mae_abs_bps_v1").ge(40.0) & _num(frame, "peak_mfe_bps_v1").lt(50.0)
    frame["r6_label_bad_trade_overlap_extreme_vol_v1"] = should & frame.get("as_of_session_v1", pd.Series("", index=frame.index)).astype("string").str.upper().eq("OVERLAP") & frame.get("as_of_candidate_vol_regime_v1", pd.Series("", index=frame.index)).astype("string").str.upper().eq("EXTREME")
    frame["r6_label_batch04_blindspot_v1"] = frame["r6_label_missed_should_not_take_v1"] & frame["batch_scope_v1"].astype("string").eq("BATCH_04")
    frame["r6_label_trend_neutral_extreme_vol_risk_v1"] = should & frame.get("as_of_candidate_trend_regime_v1", pd.Series("", index=frame.index)).astype("string").str.upper().eq("TREND_NEUTRAL") & frame.get("as_of_candidate_vol_regime_v1", pd.Series("", index=frame.index)).astype("string").str.upper().eq("EXTREME")
    frame["r6_label_bad_risk_v1"] = should
    frame["r6_label_tail_control_10_50_v1"] = frame["tail_10_50_mfe_v1"]
    return frame


def _feature_names(frame: pd.DataFrame) -> list[str]:
    id_like = {
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "r5_2_as_of_feature_contract_v1",
        "r5_2_shadow_phase_gate_as_of_contract_v1",
        "as_of_feature_namespace_v1",
    }
    asof = [column for column in frame.columns if column.startswith("as_of_") and column not in id_like]
    score = [
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "blocker_score_v1",
        "runner_protector_score_v1",
        *[column for column in frame.columns if column.startswith("pred__entry_r5_")],
    ]
    lineage = [
        "entry_observation_present_v1",
        "entry_raw_state_present_v1",
        "management_observation_present_v1",
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
        "entry_coverage_repair_source_v1",
    ]
    names = []
    for column in asof + score + lineage:
        if column in frame.columns and column not in names:
            names.append(column)
    return names


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    valid = np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {"row_count_v1": 0}
    y_pred = (y_prob >= threshold).astype(int)
    row: Dict[str, Any] = {
        "row_count_v1": int(len(y_true)),
        "positive_count_v1": int(y_true.sum()),
        "pred_positive_count_v1": int(y_pred.sum()),
        "confusion_matrix_json_v1": _json_dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()),
        "balanced_accuracy_v1": None,
        "precision_true_v1": None,
        "recall_true_v1": None,
        "roc_auc_v1": None,
        "brier_score_v1": None,
    }
    if len(set(y_true.tolist())) >= 2:
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        row.update(
            {
                "balanced_accuracy_v1": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_true_v1": float(precision[1]),
                "recall_true_v1": float(recall[1]),
                "roc_auc_v1": float(roc_auc_score(y_true, y_prob)),
                "brier_score_v1": float(brier_score_loss(y_true, y_prob)),
            }
        )
    return row


def _sample_weights(frame: pd.DataFrame, label_col: str) -> np.ndarray:
    y = _bool(frame, label_col).astype(int).to_numpy(dtype=int)
    weights = compute_sample_weight("balanced", y).astype(float)
    if label_col == "r6_label_runner_protect_v1":
        weights[_bool(frame, "r6_label_runner_near_miss_v1").to_numpy(dtype=bool)] *= 8.0
        weights[_bool(frame, "r6_label_runner_200_mfe_v1").to_numpy(dtype=bool)] *= 10.0
        weights[_bool(frame, "r6_label_runner_100_mfe_v1").to_numpy(dtype=bool)] *= 6.0
        weights[_bool(frame, "r6_label_repaired_165_like_runner_v1").to_numpy(dtype=bool)] *= 5.0
    elif label_col in {"r6_label_bad_risk_v1", "r6_label_risky_allow_v1", "r6_label_batch04_blindspot_v1"}:
        weights[_bool(frame, "r6_label_missed_should_not_take_v1").to_numpy(dtype=bool)] *= 2.5
        weights[_bool(frame, "r6_label_risky_allow_v1").to_numpy(dtype=bool)] *= 4.0
        weights[_bool(frame, "r6_label_batch04_blindspot_v1").to_numpy(dtype=bool)] *= 5.0
        protected_negative = (
            _bool(frame, "r6_label_runner_200_mfe_v1")
            | _bool(frame, "r6_label_runner_100_mfe_v1")
            | _bool(frame, "r6_label_repaired_165_like_runner_v1")
            | _bool(frame, "r6_label_runner_near_miss_v1")
        ).to_numpy(dtype=bool)
        weights[protected_negative & (y == 0)] *= 5.0
    elif label_col == "r6_label_tail_control_10_50_v1":
        weights[_bool(frame, "r6_label_tail_control_10_50_v1").to_numpy(dtype=bool)] *= 4.0
        weights[_bool(frame, "r6_label_runner_50_mfe_v1").to_numpy(dtype=bool) & (y == 0)] *= 4.0
    return weights


def _train_head(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    spec: R6HeadSpec,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path | None,
    model_tag: str,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.Series, pd.DataFrame]:
    y_all = _bool(frame, spec.label_col).astype(int)
    train_mask = train_mask.reindex(frame.index).fillna(False).astype(bool)
    validation_mask = validation_mask.reindex(frame.index).fillna(False).astype(bool)
    if int(train_mask.sum()) < 20 or len(set(y_all.loc[train_mask].tolist())) < 2:
        constant = float(y_all.loc[train_mask].mean()) if int(train_mask.sum()) else float(y_all.mean())
        probs = pd.Series(constant, index=frame.index, dtype="float64")
        metrics = _classification_metrics(y_all.to_numpy(dtype=int), probs.to_numpy(dtype=float))
        metrics.update({"model_tag_v1": model_tag, "head_id_v1": spec.head_id, "label_col_v1": spec.label_col, "output_col_v1": spec.output_col, "split_v1": "ALL", "constant_model_v1": True})
        return probs, pd.DataFrame([metrics])
    if int(validation_mask.sum()) == 0 or len(set(y_all.loc[validation_mask].tolist())) < 2:
        validation_mask = train_mask
    preprocessor = _fit_preprocessor(frame.loc[train_mask, feature_names], feature_names)
    x_train = _transform_features(preprocessor, frame.loc[train_mask, feature_names])
    x_val = _transform_features(preprocessor, frame.loc[validation_mask, feature_names])
    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_val = y_all.loc[validation_mask].to_numpy(dtype=int)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        reg_alpha=0.5,
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=_sample_weights(frame.loc[train_mask].copy(), spec.label_col), eval_set=[(x_val, y_val)], verbose=False)
    x_all = _transform_features(preprocessor, frame[feature_names])
    probs = pd.Series(model.predict_proba(x_all)[:, 1], index=frame.index, dtype="float64")
    rows: list[dict[str, Any]] = []
    for split_name, mask in {"TRAIN": train_mask, "VALIDATION": validation_mask, "HOLDOUT_OR_OTHER": ~(train_mask | validation_mask), "ALL": pd.Series(True, index=frame.index)}.items():
        if int(mask.sum()) == 0:
            continue
        metrics = _classification_metrics(y_all.loc[mask].to_numpy(dtype=int), probs.loc[mask].to_numpy(dtype=float))
        metrics.update({"model_tag_v1": model_tag, "head_id_v1": spec.head_id, "label_col_v1": spec.label_col, "output_col_v1": spec.output_col, "split_v1": split_name, "constant_model_v1": False})
        rows.append(metrics)
    if output_dir is not None:
        model_dir = output_dir / "models" / model_tag / spec.head_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
        _write_json(
            model_dir / "metadata.json",
            {
                "model_tag_v1": model_tag,
                "head_id_v1": spec.head_id,
                "label_col_v1": spec.label_col,
                "output_col_v1": spec.output_col,
                "feature_count_v1": int(len(feature_names)),
                "train_rows_v1": int(train_mask.sum()),
                "validation_rows_v1": int(validation_mask.sum()),
                "best_iteration_v1": getattr(model, "best_iteration", None),
                "best_score_v1": _safe_float(getattr(model, "best_score", None)),
                "not_live_gate": True,
            },
        )
    return probs, pd.DataFrame(rows)


def _train_heads(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path | None,
    model_tag: str,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = frame[["candidate_uid"]].copy()
    metric_frames: list[pd.DataFrame] = []
    for idx, spec in enumerate(HEAD_SPECS):
        probs, metrics = _train_head(
            frame=frame,
            feature_names=feature_names,
            spec=spec,
            train_mask=train_mask,
            validation_mask=validation_mask,
            output_dir=output_dir,
            model_tag=model_tag,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + idx * 17,
            n_jobs=n_jobs,
        )
        pred[spec.output_col] = probs.to_numpy(dtype=float)
        metric_frames.append(metrics)
    return pred, pd.concat(metric_frames, ignore_index=True)


def _asof_runner_guard(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, "as_of_candidate_tradable_prob_v1").ge(0.94)
        & _num(frame, "as_of_entry_candidate_path_quality_pred_v1").ge(0.70)
        & _num(frame, "as_of_candidate_mfe_first_n_pred_v1").ge(1.75)
        & _num(frame, "as_of_skip_candidate_p_flat_v1").le(0.50)
    )


def _candidate_thresholds(candidate: R6Candidate) -> Dict[str, Any]:
    return {
        "family_v1": candidate.family,
        "bad_threshold_v1": candidate.bad_threshold,
        "runner_threshold_v1": candidate.runner_threshold,
        "tail_threshold_v1": candidate.tail_threshold,
        "risky_threshold_v1": candidate.risky_threshold,
        "blindspot_threshold_v1": candidate.blindspot_threshold,
        "r5_2_runner_threshold_v1": candidate.r5_2_runner_threshold,
        "use_r5_2_base_v1": candidate.use_r5_2_base,
        "hard_asof_runner_guard_v1": candidate.hard_asof_runner_guard,
    }


def _candidate_grid(compact: bool) -> list[R6Candidate]:
    families = [
        "R6_RUNNER_FIRST_TWO_HEAD",
        "R6_THREE_HEAD_BLOCK_RUNNER_TAIL",
        "R6_R5_2_DISTILLED_PLUS_RISKY_ALLOW",
        "R6_BATCH04_AWARE_ROBUST",
        "R6_CONSERVATIVE_HIGH_PRECISION",
        "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
    ]
    out: list[R6Candidate] = []
    idx = 0
    for family in families:
        if family == "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON":
            bad_grid = [0.85, 0.90] if compact else [0.85, 0.90, 0.95, 0.99]
            runner_grid = [0.50, 0.70] if compact else [0.30, 0.50, 0.60, 0.70, 0.74, 0.82, 0.90]
            tail_grid = [0.90] if compact else [0.85, 0.90, 0.95, 0.99]
            risky_grid = [0.90] if compact else [0.85, 0.90, 0.95, 0.99]
            blind_grid = [0.70]
        else:
            bad_grid = [0.55, 0.65, 0.75] if compact else [0.45, 0.55, 0.65, 0.75, 0.85]
            runner_grid = [0.70, 0.82] if compact else [0.62, 0.70, 0.74, 0.82, 0.90]
            tail_grid = [0.65, 0.80] if compact else [0.50, 0.65, 0.75, 0.85]
            risky_grid = [0.65] if compact else [0.50, 0.65, 0.80]
            blind_grid = [0.70] if compact else [0.55, 0.70, 0.85]
        for bad_t in bad_grid:
            for runner_t in runner_grid:
                for tail_t in tail_grid:
                    for risky_t in risky_grid:
                        for blind_t in blind_grid:
                            idx += 1
                            out.append(
                                R6Candidate(
                                    policy_name=f"R6_CANDIDATE_{idx:05d}_{family}",
                                    family=family,
                                    bad_threshold=bad_t,
                                    runner_threshold=runner_t,
                                    tail_threshold=tail_t,
                                    risky_threshold=risky_t,
                                    blindspot_threshold=blind_t,
                                    r5_2_runner_threshold=0.74,
                                    use_r5_2_base=family
                                    in {
                                        "R6_R5_2_DISTILLED_PLUS_RISKY_ALLOW",
                                        "R6_BATCH04_AWARE_ROBUST",
                                        "R6_CONSERVATIVE_HIGH_PRECISION",
                                        "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                                    },
                                    hard_asof_runner_guard=family
                                    in {
                                        "R6_BATCH04_AWARE_ROBUST",
                                        "R6_CONSERVATIVE_HIGH_PRECISION",
                                        "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                                    },
                                )
                            )
    return out


def _policy_mask(frame: pd.DataFrame, candidate: R6Candidate) -> pd.Series:
    bad = pd.to_numeric(frame[R6_BAD_PROB], errors="coerce")
    runner = pd.to_numeric(frame[R6_RUNNER_PROB], errors="coerce")
    tail = pd.to_numeric(frame[R6_TAIL_PROB], errors="coerce")
    risky = pd.to_numeric(frame[R6_RISKY_PROB], errors="coerce")
    blind = pd.to_numeric(frame[R6_BLINDSPOT_PROB], errors="coerce")
    r5_2_runner = pd.to_numeric(frame[R5_2_RUNNER_PROB], errors="coerce")
    r5_2_base = _bool(frame, "r5_2_selected_candidate__block_v1")
    protect = runner.ge(candidate.runner_threshold).fillna(False) | r5_2_runner.ge(candidate.r5_2_runner_threshold).fillna(False)
    if candidate.hard_asof_runner_guard:
        protect = protect | _asof_runner_guard(frame)
    bad_signal = bad.ge(candidate.bad_threshold).fillna(False)
    tail_signal = tail.ge(candidate.tail_threshold).fillna(False)
    risky_signal = risky.ge(candidate.risky_threshold).fillna(False)
    blind_signal = blind.ge(candidate.blindspot_threshold).fillna(False)
    signal = bad_signal
    if candidate.family == "R6_RUNNER_FIRST_TWO_HEAD":
        signal = bad_signal
    elif candidate.family == "R6_THREE_HEAD_BLOCK_RUNNER_TAIL":
        signal = bad_signal | tail_signal
    elif candidate.family == "R6_R5_2_DISTILLED_PLUS_RISKY_ALLOW":
        signal = r5_2_base | risky_signal
    elif candidate.family == "R6_BATCH04_AWARE_ROBUST":
        signal = r5_2_base | bad_signal | risky_signal | blind_signal
    elif candidate.family == "R6_CONSERVATIVE_HIGH_PRECISION":
        signal = r5_2_base | (bad.ge(max(candidate.bad_threshold, 0.80)).fillna(False) & risky.ge(max(candidate.risky_threshold, 0.75)).fillna(False))
    elif candidate.family == "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON":
        addon = bad_signal & risky_signal & tail_signal & ~protect
        return (r5_2_base | addon).fillna(False).astype(bool)
    else:
        raise ValueError(candidate.family)
    return (signal & ~protect).fillna(False).astype(bool)


def _reference_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index, dtype=bool),
        "R2_FALLBACK_REFERENCE": _bool(frame, "r2_fallback_reference__block_v1"),
        "R4_CURRENT_REFERENCE": _bool(frame, "r4_current_reference__block_v1"),
        "R5_CURRENT_REFERENCE": _bool(frame, "r5_current_reference__block_v1"),
        "R5_1_SAFETY_REFERENCE": _bool(frame, "r5_1_selected_reference__block_v1"),
        "R5_2_FROZEN_REFERENCE": _bool(frame, "r5_2_selected_candidate__block_v1"),
    }


def _global_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > R5_2_BASELINE["fifty_plus_mfe_blocked_v1"]:
        failures.append("fifty_plus_mfe_block_count_v1>R5.2")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) != 0:
        failures.append("strong_trade_false_block_count_v1!=0")
    if int(metric.get("strongest_winner_path_block_count_v1") or 0) != 0:
        failures.append("strongest_winner_path_block_count_v1!=0")
    if precision is None or precision < R5_2_BASELINE["global_precision_v1"]:
        failures.append("precision<R5.2")
    return not failures, ",".join(failures)


def _beats_r5_2(metric: Dict[str, Any], *, worst_loso_precision: float | None, batch04_pass: bool, batch05_pass: bool | None) -> tuple[bool, str]:
    safe, safe_reasons = _global_safety(metric)
    failures = [reason for reason in safe_reasons.split(",") if reason]
    if not safe:
        pass
    if int(metric.get("should_not_take_block_count_v1") or 0) <= R5_2_BASELINE["bad_blocks_v1"]:
        failures.append("bad_blocks<=R5.2")
    if int(metric.get("tail_10_50_help_count_v1") or 0) <= R5_2_BASELINE["tail_help_v1"]:
        failures.append("tail_help<=R5.2")
    if worst_loso_precision is None or worst_loso_precision < R5_2_BASELINE["worst_loso_precision_v1"]:
        failures.append("worst_loso_precision<R5.2")
    if not batch04_pass:
        failures.append("BATCH_04_LOSO_FAIL")
    if batch05_pass is False:
        failures.append("BATCH_05_LOSO_FAIL")
    return not failures, ",".join(failures)


def _slice_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    block_count = int(metric.get("block_count_v1") or 0)
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 1:
        failures.append("strong_trade_false_block_count_v1>1")
    if block_count > 0 and (precision is None or precision < 0.85):
        failures.append("precision<0.85")
    return not failures, ",".join(failures)


def _train_fold_predictions(
    *,
    reports_root: Path,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], pd.DataFrame]:
    fold_frames: dict[str, pd.DataFrame] = {}
    slice_infos: list[dict[str, Any]] = []
    metrics: list[pd.DataFrame] = []
    raw = frame.drop(columns=[spec.output_col for spec in HEAD_SPECS], errors="ignore")
    for slice_info in _slice_masks(reports_root, frame, batch_weeks=batch_weeks):
        scope = str(slice_info["scope_v1"])
        holdout = slice_info["mask_v1"].reindex(frame.index).fillna(False).astype(bool)
        train_all = ~holdout
        train_idx = frame.index[train_all].tolist()
        cut = int(len(train_idx) * 0.8) if len(train_idx) >= 50 else len(train_idx)
        inner_train = pd.Series(False, index=frame.index)
        inner_val = pd.Series(False, index=frame.index)
        inner_train.loc[train_idx[:cut]] = True
        inner_val.loc[train_idx[cut:]] = True
        if int(inner_val.sum()) == 0:
            inner_val = inner_train.copy()
        pred, metric_df = _train_heads(
            frame=raw,
            feature_names=feature_names,
            train_mask=inner_train,
            validation_mask=inner_val,
            output_dir=None,
            model_tag=f"r6_loso_{scope.lower()}",
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + int(slice_info["batch_index_v1"]) * 100,
            n_jobs=n_jobs,
        )
        fold = raw.merge(pred, on="candidate_uid", how="left", validate="one_to_one")
        fold_frames[scope] = fold
        metric_df["holdout_scope_v1"] = scope
        metrics.append(metric_df)
        slice_infos.append(slice_info)
    return fold_frames, slice_infos, pd.concat(metrics, ignore_index=True)


def _evaluate_candidates(base: pd.DataFrame, fold_frames: dict[str, pd.DataFrame], slice_infos: Sequence[dict[str, Any]], *, compact: bool) -> tuple[pd.DataFrame, pd.DataFrame, R6Candidate, Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for candidate in _candidate_grid(compact):
        mask = _policy_mask(base, candidate)
        global_metric = _policy_metric_row(candidate.policy_name, "ALL_1971", base, mask, thresholds=_candidate_thresholds(candidate))
        global_safe, global_fail = _global_safety(global_metric)
        slice_metrics: list[dict[str, Any]] = []
        for info in slice_infos:
            scope = str(info["scope_v1"])
            holdout = info["mask_v1"].reindex(base.index).fillna(False).astype(bool)
            fold = fold_frames[scope]
            fold_mask = _policy_mask(fold, candidate)
            metric = _policy_metric_row(candidate.policy_name, scope, fold.loc[holdout].copy(), fold_mask.loc[holdout], thresholds=_candidate_thresholds(candidate))
            spass, sfail = _slice_safety(metric)
            metric.update(
                {
                    "family_v1": candidate.family,
                    "slice_safety_pass_v1": spass,
                    "slice_safety_failure_reasons_v1": sfail,
                    "run_count_v1": int(info["run_count_v1"]),
                    "run_start_v1": info["run_start_v1"],
                    "run_end_v1": info["run_end_v1"],
                }
            )
            slice_metrics.append(metric)
        precisions = [
            _safe_float(item.get("should_not_take_precision_v1"))
            for item in slice_metrics
            if int(item.get("block_count_v1") or 0) > 0 and _safe_float(item.get("should_not_take_precision_v1")) is not None
        ]
        worst_precision = min(precisions) if precisions else 1.0
        batch04 = next((item for item in slice_metrics if item["scope_v1"] == "BATCH_04"), None)
        batch05 = next((item for item in slice_metrics if item["scope_v1"] == "BATCH_05"), None)
        batch04_pass = bool((batch04 or {}).get("slice_safety_pass_v1", False))
        batch05_pass = None if batch05 is None else bool(batch05.get("slice_safety_pass_v1", False))
        beats, beat_reasons = _beats_r5_2(global_metric, worst_loso_precision=worst_precision, batch04_pass=batch04_pass, batch05_pass=batch05_pass)
        loso_all_pass = all(bool(item["slice_safety_pass_v1"]) for item in slice_metrics)
        score = (
            float(global_metric["should_not_take_block_count_v1"]) * 2.0
            + float(global_metric["tail_10_50_help_count_v1"]) * 0.8
            + float(worst_precision) * 20.0
            - float(global_metric["take_was_ok_block_count_v1"]) * 10.0
            - float(global_metric["fifty_plus_mfe_block_count_v1"]) * 20.0
            - float(global_metric["hundred_plus_mfe_block_count_v1"]) * 40.0
            - float(global_metric["two_hundred_plus_mfe_block_count_v1"]) * 80.0
        )
        if not (global_safe and loso_all_pass):
            score -= 1000.0
        if beats:
            score += 500.0
        row = dict(global_metric)
        row.update(
            {
                "family_v1": candidate.family,
                "candidate_type_v1": "R6_MODEL_FAMILY",
                "global_safety_pass_v1": global_safe,
                "global_safety_failure_reasons_v1": global_fail,
                "loso_all_slices_pass_v1": loso_all_pass,
                "worst_loso_precision_v1": worst_precision,
                "batch04_loso_pass_v1": batch04_pass,
                "batch05_loso_pass_v1": batch05_pass,
                "batch04_precision_v1": (batch04 or {}).get("should_not_take_precision_v1"),
                "batch05_precision_v1": (batch05 or {}).get("should_not_take_precision_v1"),
                "r6_beats_r5_2_contract_v1": beats,
                "r6_contract_failure_reasons_v1": beat_reasons,
                "selection_score_v1": score,
                "thresholds_json_v1": _json_dumps(_candidate_thresholds(candidate)),
            }
        )
        rows.append(row)
        loso_rows.extend(slice_metrics)
    calibration = pd.DataFrame(rows)
    viable = calibration[calibration["r6_beats_r5_2_contract_v1"].fillna(False)].copy()
    if viable.empty:
        safe = calibration[calibration["global_safety_pass_v1"].fillna(False) & calibration["loso_all_slices_pass_v1"].fillna(False)].copy()
        selected_row = (safe if not safe.empty else calibration).sort_values(["selection_score_v1", "should_not_take_block_count_v1"], ascending=[False, False]).iloc[0].to_dict()
    else:
        selected_row = viable.sort_values(["selection_score_v1", "should_not_take_block_count_v1"], ascending=[False, False]).iloc[0].to_dict()
    thresholds = json.loads(str(selected_row["thresholds_json_v1"]))
    selected = R6Candidate(
        policy_name=str(selected_row["policy_name_v1"]),
        family=str(thresholds["family_v1"]),
        bad_threshold=float(thresholds["bad_threshold_v1"]),
        runner_threshold=float(thresholds["runner_threshold_v1"]),
        tail_threshold=float(thresholds["tail_threshold_v1"]),
        risky_threshold=float(thresholds["risky_threshold_v1"]),
        blindspot_threshold=float(thresholds["blindspot_threshold_v1"]),
        r5_2_runner_threshold=float(thresholds["r5_2_runner_threshold_v1"]),
        use_r5_2_base=bool(thresholds["use_r5_2_base_v1"]),
        hard_asof_runner_guard=bool(thresholds["hard_asof_runner_guard_v1"]),
    )
    calibration["selected_r6_candidate_v1"] = calibration["policy_name_v1"].astype("string").eq(selected.policy_name)
    return calibration, pd.DataFrame(loso_rows), selected, selected_row


def _rolling_scope_masks(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> list[dict[str, Any]]:
    run_ids = _all_run_ids(reports_root, frame)
    window = min(max(batch_weeks, 1), max(len(run_ids), 1))
    step = max(1, window // 3)
    scopes: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(run_ids), step), start=1):
        batch = run_ids[start : start + window]
        if not batch:
            continue
        scopes.append(
            {
                "scope_v1": f"ROLLING_{window:02d}W_{idx:02d}",
                "mask_v1": frame["run_id"].astype("string").isin(batch),
                "run_count_v1": len(batch),
                "run_start_v1": batch[0],
                "run_end_v1": batch[-1],
            }
        )
        if start + window >= len(run_ids):
            break
    return scopes


def _reference_and_selected_metrics(reports_root: Path, frame: pd.DataFrame, selected: R6Candidate, *, batch_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_mask = _policy_mask(frame, selected)
    policies = _reference_masks(frame)
    policies["R6_SELECTED_CANDIDATE"] = selected_mask
    head_rows: list[dict[str, Any]] = []
    walk_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    scopes = {
        "ALL_1971": pd.Series(True, index=frame.index),
        "REPAIRED_165": _bool(frame, "is_repaired_165_v1"),
        "FIFTY_PLUS_MFE_RUNNERS": _bool(frame, "fifty_plus_mfe_v1"),
        "HUNDRED_PLUS_MFE_RUNNERS": _bool(frame, "hundred_plus_mfe_v1"),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": _bool(frame, "two_hundred_plus_mfe_v1"),
        "STRONGEST_WINNER_PATH": _bool(frame, "strongest_winner_path_v1"),
        "TAIL_10_50_MFE_POCKET": _bool(frame, "tail_10_50_mfe_v1"),
    }
    for policy_name, mask in policies.items():
        for scope, scope_mask in scopes.items():
            head_rows.append(_policy_metric_row(policy_name, scope, frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={"selected_r6_thresholds_v1": _candidate_thresholds(selected) if policy_name == "R6_SELECTED_CANDIDATE" else {}}))
    for info in _slice_masks(reports_root, frame, batch_weeks=batch_weeks):
        scope_mask = info["mask_v1"].reindex(frame.index).fillna(False).astype(bool)
        for policy_name, mask in policies.items():
            row = _policy_metric_row(policy_name, str(info["scope_v1"]), frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={})
            row.update({key: value for key, value in info.items() if key != "mask_v1"})
            walk_rows.append(row)
    for info in _rolling_scope_masks(reports_root, frame, batch_weeks=batch_weeks):
        scope_mask = info["mask_v1"].reindex(frame.index).fillna(False).astype(bool)
        for policy_name, mask in policies.items():
            row = _policy_metric_row(policy_name, str(info["scope_v1"]), frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={})
            row.update({key: value for key, value in info.items() if key != "mask_v1"})
            rolling_rows.append(row)
    prediction_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "is_repaired_165_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "baseline_realized_pnl_bps_v1",
        *[spec.output_col for spec in HEAD_SPECS],
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
    ]
    prediction_df = frame[[column for column in prediction_cols if column in frame.columns]].copy()
    for policy_name, mask in policies.items():
        prediction_df[f"{policy_name.lower()}__block_v1"] = mask.to_numpy(dtype=bool)
    return pd.DataFrame(head_rows), pd.DataFrame(walk_rows), pd.DataFrame(rolling_rows), prediction_df


def _label_audit(frame: pd.DataFrame, labels: Sequence[str], audit_family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    splits = {"ALL": pd.Series(True, index=frame.index), "TRAIN": _bool(frame, "used_for_training"), "VALIDATION": _bool(frame, "used_for_validation"), "HOLDOUT": _bool(frame, "used_for_holdout")}
    near = _bool(frame, "r6_label_runner_near_miss_v1")
    for label in labels:
        series = _bool(frame, label)
        for split_name, mask in splits.items():
            rows.append(
                {
                    "audit_family_v1": audit_family,
                    "label_name_v1": label,
                    "split_v1": split_name,
                    "row_count_v1": int(mask.sum()),
                    "positive_count_v1": int(series.loc[mask].sum()),
                    "positive_rate_v1": _safe_rate(float(series.loc[mask].sum()), float(mask.sum())),
                    "runner_near_miss_positive_count_v1": int((series & near & mask).sum()),
                    "safe_for_training_v1": bool(len(set(series.loc[mask].astype(int).tolist())) == 2) if int(mask.sum()) else False,
                    "hindsight_supervision_only_v1": True,
                }
            )
    return pd.DataFrame(rows)


def _effect_score(feature: pd.Series, positive: pd.Series, negative: pd.Series) -> float | None:
    pos = feature.loc[positive]
    neg = feature.loc[negative]
    if len(pos) < 3 or len(neg) < 3:
        return None
    if pd.api.types.is_numeric_dtype(feature) or pd.api.types.is_bool_dtype(feature):
        pos_n = pd.to_numeric(pos, errors="coerce").dropna()
        neg_n = pd.to_numeric(neg, errors="coerce").dropna()
        if len(pos_n) < 3 or len(neg_n) < 3:
            return None
        denom = float(np.nanstd(pd.concat([pos_n, neg_n]).to_numpy(dtype=float)))
        return 0.0 if denom == 0.0 or not math.isfinite(denom) else abs(float(pos_n.mean()) - float(neg_n.mean())) / denom
    pos_dist = pos.astype("string").fillna("__NA__").value_counts(normalize=True)
    neg_dist = neg.astype("string").fillna("__NA__").value_counts(normalize=True)
    keys = set(pos_dist.index).union(set(neg_dist.index))
    return 0.5 * sum(abs(float(pos_dist.get(key, 0.0)) - float(neg_dist.get(key, 0.0))) for key in keys)


def _feature_path_audit(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    contrasts = {
        "MISSED_SHOULD_NOT_TAKE_VS_CLEAN_TAKE": (_bool(frame, "r6_label_missed_should_not_take_v1"), _bool(frame, "take_was_ok_v1")),
        "RISKY_ALLOW_VS_CLEAN_TAKE": (_bool(frame, "r6_label_risky_allow_v1"), _bool(frame, "take_was_ok_v1")),
        "TAIL_10_50_VS_50_PLUS_RUNNER": (_bool(frame, "r6_label_tail_control_10_50_v1"), _bool(frame, "r6_label_runner_50_mfe_v1")),
        "RUNNER_NEAR_MISS_VS_SHOULD_NOT_TAKE": (_bool(frame, "r6_label_runner_near_miss_v1"), _bool(frame, "label_should_not_take_v1")),
    }
    rows: list[dict[str, Any]] = []
    for contrast, (positive, negative) in contrasts.items():
        for family in sorted({_feature_family(feature) for feature in feature_names}):
            family_features = [feature for feature in feature_names if _feature_family(feature) == family]
            scored: list[tuple[str, float]] = []
            for feature in family_features:
                score = _effect_score(frame[feature], positive, negative)
                if score is not None and math.isfinite(score):
                    scored.append((feature, float(score)))
            scored = sorted(scored, key=lambda item: item[1], reverse=True)
            rows.append(
                {
                    "contrast_name_v1": contrast,
                    "feature_family_v1": family,
                    "feature_count_v1": int(len(family_features)),
                    "positive_count_v1": int(positive.sum()),
                    "negative_count_v1": int(negative.sum()),
                    "mean_top5_effect_score_v1": _safe_float(np.mean([score for _, score in scored[:5]])) if scored else None,
                    "max_effect_score_v1": scored[0][1] if scored else None,
                    "top_features_json_v1": _json_dumps([{"feature_v1": feature, "score_v1": score} for feature, score in scored[:10]]),
                    "path_dynamics_status_v1": "AVAILABLE_EXISTING_AS_OF" if family in {"prior_path_impulse_context", "structure_swing_retracement", "volatility_range"} else "AS_OF_AVAILABLE",
                }
            )
    expected_new = [
        "as_of_last_peak_ts_utc_v1",
        "as_of_last_mfe_ts_utc_v1",
        "as_of_last_peak_mfe_bps_v1",
        "as_of_max_mfe_without_mae_bps_v1",
        "as_of_mfe_mae_sequence_order_v1",
    ]
    for feature in expected_new:
        rows.append(
            {
                "contrast_name_v1": "PATH_DYNAMICS_LOGGING_COVERAGE",
                "feature_family_v1": "new_path_dynamics_logging",
                "feature_count_v1": 1,
                "positive_count_v1": int(feature in frame.columns),
                "negative_count_v1": int(feature not in frame.columns),
                "mean_top5_effect_score_v1": None,
                "max_effect_score_v1": None,
                "top_features_json_v1": _json_dumps([{"feature_v1": feature, "present_v1": feature in frame.columns}]),
                "path_dynamics_status_v1": "AVAILABLE" if feature in frame.columns else "LOGGING_BLOCKED",
            }
        )
    return pd.DataFrame(rows).sort_values(["contrast_name_v1", "mean_top5_effect_score_v1"], ascending=[True, False])


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in HEAD_SPECS:
        label = _bool(frame, spec.label_col)
        score = pd.to_numeric(frame[spec.output_col], errors="coerce")
        valid = score.notna()
        if int(valid.sum()) == 0:
            continue
        for bin_name, part in frame.loc[valid].groupby(pd.cut(score.loc[valid], bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], labels=["0_0.2", "0.2_0.4", "0.4_0.6", "0.6_0.8", "0.8_1.0"]), observed=False):
            if part.empty:
                continue
            part_label = label.loc[part.index]
            rows.append(
                {
                    "head_id_v1": spec.head_id,
                    "score_column_v1": spec.output_col,
                    "label_column_v1": spec.label_col,
                    "bin_v1": str(bin_name),
                    "row_count_v1": int(len(part)),
                    "mean_score_v1": _safe_float(pd.to_numeric(part[spec.output_col], errors="coerce").mean()),
                    "observed_positive_rate_v1": _safe_rate(float(part_label.sum()), float(len(part))),
                    "brier_score_v1": _safe_float(((pd.to_numeric(part[spec.output_col], errors="coerce").fillna(0.0) - part_label.astype(float)) ** 2).mean()),
                }
            )
    return pd.DataFrame(rows)


def _tail_audit(frame: pd.DataFrame, selected_mask: pd.Series) -> pd.DataFrame:
    scopes = {
        "ALL_TAIL_10_50": _bool(frame, "tail_10_50_mfe_v1"),
        "MISSED_TAIL_10_50_FROM_R5_2": _bool(frame, "tail_10_50_mfe_v1") & ~_bool(frame, "r5_2_selected_candidate__block_v1"),
        "FIFTY_PLUS_RUNNERS": _bool(frame, "fifty_plus_mfe_v1"),
        "REPAIRED_165": _bool(frame, "is_repaired_165_v1"),
    }
    rows: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        part = frame.loc[mask].copy()
        rows.append(
            {
                "scope_v1": scope,
                "row_count_v1": int(len(part)),
                "r6_selected_block_count_v1": int(selected_mask.loc[mask].sum()),
                "r5_2_block_count_v1": int(_bool(frame, "r5_2_selected_candidate__block_v1").loc[mask].sum()),
                "avg_tail_score_v1": _safe_float(pd.to_numeric(part.get(R6_TAIL_PROB, pd.Series(dtype=float)), errors="coerce").mean()) if not part.empty else None,
                "runner_damage_count_v1": int((selected_mask.loc[mask] & _bool(part, "take_was_ok_v1")).sum()) if not part.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _decision(global_row: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
    beats = bool(global_row.get("r6_beats_r5_2_contract_v1", False))
    if beats:
        decision = "R6_SHADOW_CANDIDATE_BEATS_R5_2"
    elif bool(global_row.get("global_safety_pass_v1", False)) and int(global_row.get("tail_10_50_help_count_v1") or 0) > R5_2_BASELINE["tail_help_v1"]:
        decision = "R6_IMPROVES_LABELS_BUT_NOT_READY"
    elif int(global_row.get("should_not_take_block_count_v1") or 0) <= R5_2_BASELINE["bad_blocks_v1"]:
        decision = "R6_FEATURES_INSUFFICIENT"
    else:
        decision = "KEEP_R5_2_FROZEN_REFERENCE"
    rows = [
        {"decision_key_v1": "R6_SHADOW_CANDIDATE_BEATS_R5_2", "status_v1": "PASS" if decision == "R6_SHADOW_CANDIDATE_BEATS_R5_2" else "NOT_PRIMARY", "reason_v1": "Requires all R5.2 go/no-go constraints plus bad/tail improvement."},
        {"decision_key_v1": "R6_IMPROVES_LABELS_BUT_NOT_READY", "status_v1": "PASS" if decision == "R6_IMPROVES_LABELS_BUT_NOT_READY" else "NOT_PRIMARY", "reason_v1": "Some R6 label scores improve, but full contract not beaten."},
        {"decision_key_v1": "R6_FEATURES_INSUFFICIENT", "status_v1": "PASS" if decision == "R6_FEATURES_INSUFFICIENT" else "NOT_PRIMARY", "reason_v1": "Existing AS_OF features do not add enough safe recall."},
        {"decision_key_v1": "KEEP_R5_2_FROZEN_REFERENCE", "status_v1": "PASS" if decision == "KEEP_R5_2_FROZEN_REFERENCE" else "REFERENCE", "reason_v1": "Keep R5.2 when R6 fails strict freeze contract."},
        {"decision_key_v1": "IMPROVE_PATH_DYNAMICS_LOGGING_FIRST", "status_v1": "REFERENCE", "reason_v1": "Use when path-dynamics audit is LOGGING_BLOCKED for important fields."},
    ]
    return pd.DataFrame(rows), decision


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# R6 Entry Runner First Retrain V1",
        "",
        "Shadow/research only. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Decision: `{summary['decision_v1']['recommended_next_step_v1']}`",
        f"- Selected family: `{summary['selected_candidate_v1']['family_v1']}`",
        f"- R6 bad blocks: `{summary['selected_candidate_v1']['should_not_take_block_count_v1']}` vs R5.2 `106`",
        f"- R6 tail help: `{summary['selected_candidate_v1']['tail_10_50_help_count_v1']}` vs R5.2 `82`",
        f"- R6 precision: `{summary['selected_candidate_v1']['should_not_take_precision_v1']}` vs R5.2 `0.9724770642201835`",
        f"- R6 beats contract: `{summary['selected_candidate_v1']['r6_beats_r5_2_contract_v1']}`",
        "",
        "## Guardrails",
        "",
        "- Runner-protection is evaluated before blocker expansion.",
        "- AS_OF features and HINDSIGHT labels are materialized separately.",
        "- No live promotion.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    freeze_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    compact_grid: bool,
) -> Dict[str, Any]:
    asof_df, hindsight_df, policy_lock_df, r5_2_pred_df, phase_bakeoff_df, go_no_go_df, freeze_summary, freeze_manifest, target_spec, phase_dir, r5_2_source_dir = _load_inputs(
        reports_root=reports_root,
        freeze_dir=freeze_dir,
        expected_ledger_count=expected_ledger_count,
    )
    base = _prepare_frame(
        reports_root=reports_root,
        asof_df=asof_df,
        hindsight_df=hindsight_df,
        policy_lock_df=policy_lock_df,
        r5_2_pred_df=r5_2_pred_df,
        batch_weeks=batch_weeks,
    )
    feature_names = _feature_names(base)
    train_mask = _bool(base, "used_for_training")
    validation_mask = _bool(base, "used_for_validation")
    global_pred, model_metrics = _train_heads(
        frame=base,
        feature_names=feature_names,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=extension_dir,
        model_tag="global_r6_runner_first",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    scored = base.merge(global_pred, on="candidate_uid", how="left", validate="one_to_one")
    fold_frames, slice_infos, fold_model_metrics = _train_fold_predictions(
        reports_root=reports_root,
        frame=scored,
        feature_names=feature_names,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    calibration_df, loso_all_df, selected, selected_row = _evaluate_candidates(scored, fold_frames, slice_infos, compact=compact_grid)
    selected_mask = _policy_mask(scored, selected)
    head_to_head_df, walkforward_df, rolling_df, prediction_view_df = _reference_and_selected_metrics(reports_root, scored, selected, batch_weeks=batch_weeks)
    selected_loso_df = loso_all_df[loso_all_df["policy_name_v1"].astype("string").eq(selected.policy_name)].copy()
    runner_audit_df = _label_audit(
        scored,
        [
            "r6_label_runner_50_mfe_v1",
            "r6_label_runner_100_mfe_v1",
            "r6_label_runner_200_mfe_v1",
            "r6_label_repaired_165_like_runner_v1",
            "r6_label_strong_low_mae_runner_v1",
            "r6_label_high_mfe_low_giveback_v1",
            "r6_label_runner_near_miss_v1",
            "r6_label_runner_protect_v1",
        ],
        "R6_RUNNER_PROTECTOR_FIRST_RETRAIN_V1",
    )
    bad_audit_df = _label_audit(
        scored,
        [
            "r6_label_missed_should_not_take_v1",
            "r6_label_risky_allow_v1",
            "r6_label_high_mae_low_mfe_v1",
            "r6_label_low_mfe_low_value_v1",
            "r6_label_early_adverse_excursion_v1",
            "r6_label_bad_trade_overlap_extreme_vol_v1",
            "r6_label_batch04_blindspot_v1",
            "r6_label_trend_neutral_extreme_vol_risk_v1",
            "r6_label_bad_risk_v1",
        ],
        "R6_BAD_RISK_LABEL_REDESIGN_V1",
    )
    tail_audit_df = _tail_audit(scored, selected_mask)
    feature_audit_df = _feature_path_audit(scored, feature_names)
    threshold_df = _calibration_table(scored)
    decision_df, decision = _decision(selected_row)
    selected_row = dict(selected_row)
    selected_row["selected_policy_name_v1"] = selected.policy_name
    selected_row["selected_thresholds_v1"] = _candidate_thresholds(selected)
    failed_checks = 0
    consistency_df = pd.DataFrame(
        [
            _audit_record("FREEZE_INPUT_PRESENT", "PASS", {"freeze_dir": str(freeze_dir), "freeze_id": freeze_manifest.get("freeze_id_v1")}),
            _audit_record("FULL_COVERAGE", "PASS" if expected_ledger_count is None or len(scored) == expected_ledger_count else "FAIL", {"observed": len(scored), "expected": expected_ledger_count}),
            _audit_record("NO_SYNTHETIC_INPUT", "PASS" if int(freeze_summary.get("coverage_v1", {}).get("synthetic_count_v1", 0)) == 0 else "FAIL", {"coverage": freeze_summary.get("coverage_v1", {})}),
            _audit_record("AS_OF_HINDSIGHT_SEPARATED", "PASS", {"as_of_table": AS_OF_FEATURE_TABLE, "hindsight_table": HINDSIGHT_LABEL_OUTCOME_TABLE}),
            _audit_record("MODEL_HEADS_TRAINED", "PASS" if len(model_metrics["head_id_v1"].astype("string").unique()) == len(HEAD_SPECS) else "FAIL", {"heads": model_metrics["head_id_v1"].astype("string").unique().tolist()}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_controller": True, "not_live_gate": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R6_ENTRY_RUNNER_FIRST_STATUS_V1",
        "R6_STATUS": "TRAINED_SHADOW_RESEARCH_NOT_PROMOTED" if failed_checks == 0 else "ISSUES_FOUND_NOT_PROMOTED",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R6_ENTRY_RUNNER_FIRST_RETRAIN_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "freeze_dir_v1": str(freeze_dir),
        "phase_gate_dir_v1": str(phase_dir),
        "r5_2_source_dir_v1": str(r5_2_source_dir),
        "coverage_v1": {
            "ledger_trade_count_v1": int(len(scored)),
            "entry_coverage_v1": int(len(scored)),
            "synthetic_count_v1": int(freeze_summary.get("coverage_v1", {}).get("synthetic_count_v1", 0)),
        },
        "feature_count_v1": int(len(feature_names)),
        "head_count_v1": int(len(HEAD_SPECS)),
        "candidate_count_v1": int(len(calibration_df)),
        "selected_candidate_v1": selected_row,
        "decision_v1": {
            "recommended_next_step_v1": decision,
            "r6_beats_r5_2_contract_v1": bool(selected_row.get("r6_beats_r5_2_contract_v1", False)),
            "winning_family_v1": selected.family,
            "r5_2_bad_blocks_v1": R5_2_BASELINE["bad_blocks_v1"],
            "r5_2_tail_help_v1": R5_2_BASELINE["tail_help_v1"],
        },
        "path_dynamics_v1": {
            "logging_blocked_count_v1": int(feature_audit_df["path_dynamics_status_v1"].astype("string").eq("LOGGING_BLOCKED").sum()),
            "existing_path_family_rows_v1": int(feature_audit_df["feature_family_v1"].astype("string").isin(["prior_path_impulse_context", "structure_swing_retracement", "volatility_range"]).sum()),
        },
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R6 trained {len(HEAD_SPECS)} heads on {len(feature_names)} AS_OF/meta features.",
                f"Selected R6 family: {selected.family}.",
                f"R6 bad blocks={selected_row.get('should_not_take_block_count_v1')} and tail_help={selected_row.get('tail_10_50_help_count_v1')}.",
                f"R6 beats R5.2 contract={selected_row.get('r6_beats_r5_2_contract_v1')}.",
                "No live promotion was materialized.",
            ],
            "INDIKERT": [
                "Runner-first constraints reduce the chance of expanding recall into runner damage.",
                "Existing AS_OF path/range/retracement families provide some signal, but missing new path-dynamics fields limit R6.",
            ],
            "IKKE_ETABLERT": [
                "Live fallback safety.",
                "Whether future unseen regimes improve beyond locked canonical replay.",
                "Counterfactual fill quality for newly blocked trades.",
            ],
        },
    }
    contract = {
        "layer_name": "R6_ENTRY_RUNNER_FIRST_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "freeze_benchmark_v1": {
            "freeze_id_v1": freeze_manifest.get("freeze_id_v1"),
            "model_id_v1": freeze_manifest.get("model_version_id_v1"),
            "threshold_version_id_v1": freeze_manifest.get("threshold_version_id_v1"),
            "selected_policy_stack_v1": freeze_manifest.get("selected_policy_stack_v1"),
        },
        "feature_names_v1": feature_names,
        "head_specs_v1": [spec.__dict__ for spec in HEAD_SPECS],
        "r6_contract_to_beat_v1": R5_2_BASELINE,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R6_ENTRY_RUNNER_FIRST_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_feature_table": AS_OF_FEATURE_TABLE,
            "hindsight_label_outcome_table": HINDSIGHT_LABEL_OUTCOME_TABLE,
            "runner_label_audit": RUNNER_LABEL_AUDIT,
            "bad_risk_label_audit": BAD_RISK_LABEL_AUDIT,
            "tail_control_audit": TAIL_CONTROL_AUDIT,
            "feature_path_dynamics_audit": FEATURE_PATH_DYNAMICS_AUDIT,
            "model_family_bakeoff": MODEL_FAMILY_BAKEOFF,
            "threshold_calibration": THRESHOLD_CALIBRATION,
            "walkforward_metrics": WALKFORWARD_METRICS,
            "loso_metrics": LOSO_METRICS,
            "rolling_window_metrics": ROLLING_WINDOW_METRICS,
            "head_to_head": HEAD_TO_HEAD,
            "policy_prediction_view": POLICY_PREDICTION_VIEW,
            "decision_matrix": DECISION_MATRIX,
            "summary": SUMMARY,
            "status": STATUS,
            "report": REPORT,
            "consistency_audit": CONSISTENCY_AUDIT,
            "models_dir": "models",
        },
    }
    asof_identity_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
    ]
    asof_out = scored[[column for column in [*asof_identity_cols, *feature_names] if column in scored.columns]].copy()
    asof_out["r6_as_of_feature_contract_v1"] = "AS_OF_ONLY_WITH_FROZEN_R5_2_SCORE_META_FEATURES"
    hindsight_cols = [
        "candidate_uid",
        "run_id",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        *[column for column in scored.columns if column.startswith("r6_label_")],
    ]
    hindsight_out = scored[[column for column in hindsight_cols if column in scored.columns]].copy()
    hindsight_out["r6_hindsight_contract_v1"] = "HINDSIGHT_SUPERVISION_ONLY_NOT_AS_OF_FEATURES_NOT_POLICY_TRUTH"
    return {
        "contract": _json_ready(contract),
        "asof_df": asof_out,
        "hindsight_df": hindsight_out,
        "runner_label_audit_df": runner_audit_df,
        "bad_label_audit_df": bad_audit_df,
        "tail_audit_df": tail_audit_df,
        "feature_audit_df": feature_audit_df,
        "model_bakeoff_df": calibration_df,
        "threshold_calibration_df": threshold_df,
        "walkforward_df": walkforward_df,
        "loso_df": selected_loso_df,
        "rolling_df": rolling_df,
        "head_to_head_df": head_to_head_df,
        "policy_prediction_df": prediction_view_df,
        "decision_df": decision_df,
        "summary": _json_ready(summary),
        "status": _json_ready(status),
        "manifest": _json_ready(manifest),
        "consistency_df": consistency_df,
        "report": _render_report(summary),
        "model_metrics_df": pd.concat([model_metrics, fold_model_metrics], ignore_index=True),
    }


def materialize(
    reports_root: Path,
    *,
    freeze_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
    n_estimators: int = 800,
    early_stopping_rounds: int = 60,
    learning_rate: float = 0.025,
    max_depth: int = 3,
    seed: int = 20260422,
    n_jobs: int = 4,
    compact_grid: bool = False,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_freeze_dir = _resolve_freeze_dir(reports_root, str(freeze_dir) if freeze_dir else None)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        freeze_dir=resolved_freeze_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
        compact_grid=compact_grid,
    )
    _write_json(extension_dir / CONTRACT, payload["contract"])
    payload["asof_df"].to_parquet(extension_dir / AS_OF_FEATURE_TABLE, index=False)
    payload["hindsight_df"].to_parquet(extension_dir / HINDSIGHT_LABEL_OUTCOME_TABLE, index=False)
    payload["runner_label_audit_df"].to_csv(extension_dir / RUNNER_LABEL_AUDIT, index=False)
    payload["bad_label_audit_df"].to_csv(extension_dir / BAD_RISK_LABEL_AUDIT, index=False)
    payload["tail_audit_df"].to_csv(extension_dir / TAIL_CONTROL_AUDIT, index=False)
    payload["feature_audit_df"].to_csv(extension_dir / FEATURE_PATH_DYNAMICS_AUDIT, index=False)
    payload["model_bakeoff_df"].to_csv(extension_dir / MODEL_FAMILY_BAKEOFF, index=False)
    payload["threshold_calibration_df"].to_csv(extension_dir / THRESHOLD_CALIBRATION, index=False)
    payload["walkforward_df"].to_csv(extension_dir / WALKFORWARD_METRICS, index=False)
    payload["loso_df"].to_csv(extension_dir / LOSO_METRICS, index=False)
    payload["rolling_df"].to_csv(extension_dir / ROLLING_WINDOW_METRICS, index=False)
    payload["head_to_head_df"].to_csv(extension_dir / HEAD_TO_HEAD, index=False)
    payload["policy_prediction_df"].to_parquet(extension_dir / POLICY_PREDICTION_VIEW, index=False)
    payload["decision_df"].to_csv(extension_dir / DECISION_MATRIX, index=False)
    payload["model_metrics_df"].to_csv(extension_dir / "shadow_meta_all_trade_review_r6_model_head_metrics_v1.csv", index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / SUMMARY, payload["summary"])
    _write_json(extension_dir / STATUS, payload["status"])
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {
        "extension_dir": str(extension_dir),
        "top_level_summary_path": str(reports_root / TOP_LEVEL_SUMMARY),
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train R6 entry runner-first shadow research candidate.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--freeze-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--compact-grid", action="store_true")
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        freeze_dir=Path(args.freeze_dir).expanduser().resolve() if args.freeze_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        compact_grid=args.compact_grid,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
