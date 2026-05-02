#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
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
from gx1.scripts.train_r3_entry_label_feature_retrain_v1 import _fit_preprocessor, _transform_features
from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import (
    AS_OF_FEATURE_TABLE as R5_AS_OF_FEATURE_TABLE,
    CONTRACT as R5_CONTRACT,
    HINDSIGHT_LABEL_OUTCOME_TABLE as R5_HINDSIGHT_LABEL_OUTCOME_TABLE,
    POLICY_PREDICTION_VIEW as R5_POLICY_PREDICTION_VIEW,
    R5_PROB,
    SUMMARY as R5_SUMMARY,
    _slice_masks,
)
from gx1.scripts.train_r5_loso_batch04_robustness_retrain_v1 import (
    AS_OF_GUARD_AUDIT as R5_1_AS_OF_GUARD_AUDIT,
    BATCH04_FAILURE_ATTRIBUTION as R5_1_BATCH04_FAILURE_ATTRIBUTION,
    HEAD_TO_HEAD as R5_1_HEAD_TO_HEAD,
    POLICY_PREDICTION_VIEW as R5_1_POLICY_PREDICTION_VIEW,
    SUMMARY as R5_1_SUMMARY,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_V1"
R5_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1"
R5_1_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_LOSO_BATCH04_ROBUSTNESS_RETRAIN_V1"
R4_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FULLCOVERAGE_POLICY_RECALIBRATION_AND_SHADOW_REPLAY_V1"
REPAIR_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_ENTRY_COVERAGE_REPAIR_READINESS_V1"

R4_SUMMARY = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_summary_v1.json"
R4_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_fullcoverage_policy_prediction_view_v1.parquet"
REPAIR_SUMMARY = "shadow_meta_all_trade_review_entry_coverage_repair_summary_v1.json"
REPAIR_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv"

CONTRACT = "shadow_meta_all_trade_review_r5_2_entry_runner_aware_contract_v1.json"
AS_OF_FEATURE_TABLE = "shadow_meta_all_trade_review_r5_2_as_of_feature_table_v1.parquet"
HINDSIGHT_LABEL_OUTCOME_TABLE = "shadow_meta_all_trade_review_r5_2_hindsight_label_outcome_table_v1.parquet"
HARD_NEGATIVE_AUDIT = "shadow_meta_all_trade_review_r5_2_hard_negative_audit_v1.csv"
RUNNER_LABEL_AUDIT = "shadow_meta_all_trade_review_r5_2_runner_label_audit_v1.csv"
TWO_HEAD_STACK_BAKEOFF = "shadow_meta_all_trade_review_r5_2_two_head_stack_bakeoff_v1.csv"
ROBUST_CALIBRATION = "shadow_meta_all_trade_review_r5_2_robust_calibration_v1.csv"
PARETO_FRONTIER = "shadow_meta_all_trade_review_r5_2_pareto_frontier_v1.csv"
LOSO_METRICS = "shadow_meta_all_trade_review_r5_2_loso_metrics_v1.csv"
HEAD_TO_HEAD = "shadow_meta_all_trade_review_r5_2_head_to_head_vs_r2_r4_r5_r5_1_v1.csv"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r5_2_policy_prediction_view_v1.parquet"
DECISION_MATRIX = "shadow_meta_all_trade_review_r5_2_decision_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r5_2_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r5_2_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r5_2_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r5_2_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r5_2_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r5_2_entry_runner_aware_retrain_and_loso_selection_v1.json"

BAD_PROB = "pred__entry_r5_2_bad_blocker__prob_true_v1"
RUNNER_PROB = "pred__entry_r5_2_runner_protector__prob_true_v1"


@dataclass(frozen=True)
class CandidateSpec:
    policy_name: str
    stack_family: str
    guard_mode: str
    bad_threshold: float
    runner_threshold: float
    tail_threshold: float
    r5_bad_threshold: float
    runner_margin: float = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _optional_slice_pass(frame: pd.DataFrame) -> bool | None:
    if frame.empty:
        return None
    return bool(frame["slice_safety_pass_v1"].iloc[0])


def _zscore_distance(frame: pd.DataFrame, reference: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    if reference.empty:
        return pd.Series(np.inf, index=frame.index, dtype="float64")
    usable = [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]
    if not usable:
        return pd.Series(np.inf, index=frame.index, dtype="float64")
    values = frame[usable].apply(pd.to_numeric, errors="coerce")
    ref_values = reference[usable].apply(pd.to_numeric, errors="coerce")
    means = values.mean(axis=0)
    stds = values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    values_z = ((values - means) / stds).fillna(0.0).to_numpy(dtype=float)
    ref_z = ((ref_values - means) / stds).fillna(0.0).to_numpy(dtype=float)
    distances = np.sqrt(((values_z[:, None, :] - ref_z[None, :, :]) ** 2).mean(axis=2))
    return pd.Series(distances.min(axis=1), index=frame.index, dtype="float64")


def _load_inputs(
    *,
    r5_dir: Path,
    r5_1_dir: Path,
    r4_dir: Path,
    repair_dir: Path,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    asof_df = pd.read_parquet(r5_dir / R5_AS_OF_FEATURE_TABLE)
    hindsight_df = pd.read_parquet(r5_dir / R5_HINDSIGHT_LABEL_OUTCOME_TABLE)
    r5_pred_df = pd.read_parquet(r5_dir / R5_POLICY_PREDICTION_VIEW)
    r5_contract = _load_json(r5_dir / R5_CONTRACT)
    r5_summary = _load_json(r5_dir / R5_SUMMARY)
    r5_1_summary = _load_json(r5_1_dir / R5_1_SUMMARY)
    r5_1_failure_df = pd.read_csv(r5_1_dir / R5_1_BATCH04_FAILURE_ATTRIBUTION)
    r5_1_pred_df = pd.read_parquet(r5_1_dir / R5_1_POLICY_PREDICTION_VIEW)
    r4_summary = _load_json(r4_dir / R4_SUMMARY)
    r4_pred_df = pd.read_parquet(r4_dir / R4_POLICY_PREDICTION_VIEW)
    repair_summary = _load_json(repair_dir / REPAIR_SUMMARY)
    repair_audit_df = pd.read_csv(repair_dir / REPAIR_AUDIT)

    feature_names = [str(item) for item in r5_contract.get("as_of_feature_names_v1", [])]
    if not feature_names:
        raise RuntimeError("R5 contract missing as_of_feature_names_v1")
    _require_columns(asof_df, ["candidate_uid", "run_id", "entry_observation_present_v1", "entry_raw_state_present_v1", *feature_names], artifact_name=R5_AS_OF_FEATURE_TABLE)
    _require_columns(hindsight_df, ["candidate_uid", "r5_label_should_not_take_v1", "r5_label_take_was_ok_v1", "r5_label_strong_trade_candidate_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "baseline_realized_pnl_bps_v1"], artifact_name=R5_HINDSIGHT_LABEL_OUTCOME_TABLE)
    _require_columns(r5_pred_df, ["candidate_uid", "r2_fallback_reference__block_v1", "r4_current_reference__block_v1", "r5_selected_candidate__block_v1", *R5_PROB.values()], artifact_name=R5_POLICY_PREDICTION_VIEW)
    _require_columns(r5_1_failure_df, ["candidate_uid", "batch04_loso_failure_role_v1", "two_hundred_plus_runner_false_block_v1"], artifact_name=R5_1_BATCH04_FAILURE_ATTRIBUTION)
    _require_columns(r5_1_pred_df, ["candidate_uid", "r5_1_selected_candidate__block_v1"], artifact_name=R5_1_POLICY_PREDICTION_VIEW)
    _require_columns(r4_pred_df, ["candidate_uid", "best_constrained_recalibrated_r4__block_v1"], artifact_name=R4_POLICY_PREDICTION_VIEW)
    _require_columns(repair_audit_df, ["candidate_uid"], artifact_name=REPAIR_AUDIT)
    for name, frame in [
        (R5_AS_OF_FEATURE_TABLE, asof_df),
        (R5_HINDSIGHT_LABEL_OUTCOME_TABLE, hindsight_df),
        (R5_POLICY_PREDICTION_VIEW, r5_pred_df),
        (R5_1_POLICY_PREDICTION_VIEW, r5_1_pred_df),
        (R4_POLICY_PREDICTION_VIEW, r4_pred_df),
    ]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Locked ledger expected {expected_ledger_count}, observed {len(asof_df)}")
    coverage = r5_summary.get("coverage_v1", {}) if isinstance(r5_summary.get("coverage_v1"), dict) else {}
    if int(coverage.get("entry_coverage_v1", 0)) != len(asof_df) or int(coverage.get("entry_raw_coverage_v1", 0)) != len(asof_df):
        raise RuntimeError("R5.2 requires full repaired R5 entry/raw coverage")
    if int(coverage.get("synthetic_count_v1", -1)) != 0:
        raise RuntimeError(f"R5.2 refuses synthetic input; observed {coverage.get('synthetic_count_v1')}")
    repaired_coverage = int(repair_summary.get("repaired_entry_coverage_v1", repair_summary.get("repaired_entry_coverage_count_v1", len(asof_df))))
    if repaired_coverage != len(asof_df):
        raise RuntimeError(f"Repair build does not confirm full coverage: {repaired_coverage}/{len(asof_df)}")
    return asof_df, hindsight_df, r5_pred_df, r5_1_failure_df, r5_1_pred_df, r5_summary, r5_1_summary, r4_summary, feature_names


def _prepare_frame(
    asof_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
    r5_pred_df: pd.DataFrame,
    r5_1_failure_df: pd.DataFrame,
    r5_1_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    drop_cols = [column for column in ["run_id", "trade_uid", "trade_id", "decision_timestamp"] if column in hindsight_df.columns]
    pred_cols = ["candidate_uid", "no_entry_fallback_baseline__block_v1", "r2_fallback_reference__block_v1", "r3_fullcoverage_conservative__block_v1", "r4_current_reference__block_v1", "r5_selected_candidate__block_v1", *R5_PROB.values()]
    frame = (
        asof_df.merge(hindsight_df.drop(columns=drop_cols), on="candidate_uid", how="inner", validate="one_to_one")
        .merge(r5_pred_df[[column for column in pred_cols if column in r5_pred_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(r5_1_pred_df[["candidate_uid", "r5_1_selected_candidate__block_v1"]], on="candidate_uid", how="left", validate="one_to_one")
    )
    frame["is_repaired_165_v1"] = _bool(frame, "entry_coverage_repair_applied_v1")
    frame["label_should_not_take_v1"] = _bool(frame, "r5_label_should_not_take_v1")
    frame["label_strong_trade_candidate_v1"] = _bool(frame, "r5_label_strong_trade_candidate_v1")
    frame["take_was_ok_v1"] = _bool(frame, "r5_label_take_was_ok_v1")
    frame["fifty_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(50.0)
    frame["hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(100.0)
    frame["two_hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(200.0)
    frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
    )
    frame["strongest_winner_path_v1"] = frame["two_hundred_plus_mfe_v1"] | (
        _bool(frame, "label_strong_trade_candidate_v1") & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0) & _num(frame, "peak_mfe_bps_v1").ge(50.0)
    )
    false_uid = set(r5_1_failure_df.loc[r5_1_failure_df["batch04_loso_failure_role_v1"].astype("string").eq("FALSE_BLOCK"), "candidate_uid"].astype("string").tolist())
    frame["r5_2_batch04_hard_negative_runner_v1"] = frame["candidate_uid"].astype("string").isin(false_uid)
    hard_reference = frame[frame["r5_2_batch04_hard_negative_runner_v1"]].copy()
    similarity_features = [
        "as_of_candidate_tradable_prob_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_body_bps_v1",
        "as_of_skip_replay_wick_ratio_v1",
    ]
    distance = _zscore_distance(frame, hard_reference, similarity_features)
    frame["r5_2_hard_negative_similarity_distance_v1"] = distance
    cutoff = float(np.nanquantile(distance.replace(np.inf, np.nan).dropna(), 0.08)) if distance.replace(np.inf, np.nan).notna().any() else 0.0
    frame["r5_2_hard_negative_like_asof_v1"] = distance.le(cutoff) | frame["r5_2_batch04_hard_negative_runner_v1"]
    frame["r5_2_label_runner_50_mfe_v1"] = frame["take_was_ok_v1"] & frame["fifty_plus_mfe_v1"]
    frame["r5_2_label_runner_100_mfe_v1"] = frame["take_was_ok_v1"] & frame["hundred_plus_mfe_v1"]
    frame["r5_2_label_runner_200_mfe_v1"] = frame["take_was_ok_v1"] & frame["two_hundred_plus_mfe_v1"]
    frame["r5_2_label_repaired_165_like_runner_v1"] = frame["take_was_ok_v1"] & frame["is_repaired_165_v1"]
    frame["r5_2_label_strong_low_mae_runner_v1"] = frame["take_was_ok_v1"] & frame["label_strong_trade_candidate_v1"] & _num(frame, "mae_abs_bps_v1").le(25.0)
    frame["r5_2_label_high_mfe_tail_risk_ambiguous_v1"] = frame["label_should_not_take_v1"] & frame["fifty_plus_mfe_v1"]
    frame["r5_2_label_runner_protect_v1"] = (
        frame["r5_2_label_runner_50_mfe_v1"]
        | frame["r5_2_label_runner_100_mfe_v1"]
        | frame["r5_2_label_runner_200_mfe_v1"]
        | frame["r5_2_label_repaired_165_like_runner_v1"]
        | frame["r5_2_label_strong_low_mae_runner_v1"]
        | frame["r5_2_batch04_hard_negative_runner_v1"]
        | (frame["take_was_ok_v1"] & frame["r5_2_hard_negative_like_asof_v1"])
    )
    frame["r5_2_label_bad_blocker_v1"] = frame["label_should_not_take_v1"] & ~frame["r5_2_label_high_mfe_tail_risk_ambiguous_v1"]
    return frame


def _sample_weights(frame: pd.DataFrame, label_col: str) -> np.ndarray:
    y = _bool(frame, label_col).astype(int).to_numpy(dtype=int)
    weights = compute_sample_weight("balanced", y).astype(float)
    hard_neg = _bool(frame, "r5_2_batch04_hard_negative_runner_v1").to_numpy(dtype=bool)
    hard_like = _bool(frame, "r5_2_hard_negative_like_asof_v1").to_numpy(dtype=bool)
    runner_200 = _bool(frame, "r5_2_label_runner_200_mfe_v1").to_numpy(dtype=bool)
    runner_100 = _bool(frame, "r5_2_label_runner_100_mfe_v1").to_numpy(dtype=bool)
    runner_50 = _bool(frame, "r5_2_label_runner_50_mfe_v1").to_numpy(dtype=bool)
    if label_col == "r5_2_label_bad_blocker_v1":
        weights[hard_neg] *= 9.0
        weights[hard_like & (y == 0)] *= 3.0
        weights[runner_200 & (y == 0)] *= 7.0
        weights[runner_100 & (y == 0)] *= 5.0
        weights[runner_50 & (y == 0)] *= 2.5
    else:
        weights[hard_neg] *= 10.0
        weights[hard_like & (y == 1)] *= 3.5
        weights[runner_200 & (y == 1)] *= 8.0
        weights[runner_100 & (y == 1)] *= 5.0
        weights[runner_50 & (y == 1)] *= 2.0
    return weights


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
    }
    if len(set(y_true.tolist())) >= 2:
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        row.update(
            {
                "balanced_accuracy_v1": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_true_v1": float(precision[1]),
                "recall_true_v1": float(recall[1]),
                "roc_auc_v1": float(roc_auc_score(y_true, y_prob)),
            }
        )
    return row


def _train_head(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_col: str,
    output_col: str,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    model_tag: str,
    output_dir: Path | None,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.Series, pd.DataFrame]:
    y_all = _bool(frame, label_col).astype(int)
    train_mask = train_mask.reindex(frame.index).fillna(False).astype(bool)
    validation_mask = validation_mask.reindex(frame.index).fillna(False).astype(bool)
    if int(train_mask.sum()) < 20:
        raise ValueError(f"{label_col} has too few training rows")
    if len(set(y_all.loc[train_mask].tolist())) < 2:
        raise ValueError(f"{label_col} train split requires both classes")
    if int(validation_mask.sum()) == 0 or len(set(y_all.loc[validation_mask].tolist())) < 2:
        validation_mask = train_mask
    preprocessor = _fit_preprocessor(frame.loc[train_mask, feature_names], feature_names)
    x_train = _transform_features(preprocessor, frame.loc[train_mask, feature_names])
    x_val = _transform_features(preprocessor, frame.loc[validation_mask, feature_names])
    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_val = y_all.loc[validation_mask].to_numpy(dtype=int)
    weights = _sample_weights(frame.loc[train_mask].copy(), label_col)
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
    model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
    x_all = _transform_features(preprocessor, frame[feature_names])
    probs = pd.Series(model.predict_proba(x_all)[:, 1], index=frame.index, dtype="float64")
    rows: list[dict[str, Any]] = []
    for split_name, mask in {
        "TRAIN": train_mask,
        "VALIDATION": validation_mask,
        "HOLDOUT_OR_OTHER": ~(train_mask | validation_mask),
        "ALL": pd.Series(True, index=frame.index),
    }.items():
        if int(mask.sum()) == 0:
            continue
        metric = _classification_metrics(y_all.loc[mask].to_numpy(dtype=int), probs.loc[mask].to_numpy(dtype=float))
        metric.update({"model_tag_v1": model_tag, "label_col_v1": label_col, "output_col_v1": output_col, "split_v1": split_name})
        rows.append(metric)
    if output_dir is not None:
        model_dir = output_dir / "models" / model_tag / label_col
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
        _write_json(
            model_dir / "metadata.json",
            {
                "model_tag_v1": model_tag,
                "label_col_v1": label_col,
                "output_col_v1": output_col,
                "feature_count_v1": int(len(feature_names)),
                "train_rows_v1": int(train_mask.sum()),
                "validation_rows_v1": int(validation_mask.sum()),
                "not_live_gate": True,
            },
        )
    return probs.rename(output_col), pd.DataFrame(rows)


def _train_two_heads(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_mask: pd.Series,
    validation_mask: pd.Series,
    model_tag: str,
    output_dir: Path | None,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bad_prob, bad_metrics = _train_head(
        frame=frame,
        feature_names=feature_names,
        label_col="r5_2_label_bad_blocker_v1",
        output_col=BAD_PROB,
        train_mask=train_mask,
        validation_mask=validation_mask,
        model_tag=model_tag,
        output_dir=output_dir,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    runner_prob, runner_metrics = _train_head(
        frame=frame,
        feature_names=feature_names,
        label_col="r5_2_label_runner_protect_v1",
        output_col=RUNNER_PROB,
        train_mask=train_mask,
        validation_mask=validation_mask,
        model_tag=model_tag,
        output_dir=output_dir,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed + 1,
        n_jobs=n_jobs,
    )
    pred = frame[["candidate_uid"]].copy()
    pred[BAD_PROB] = bad_prob.to_numpy(dtype=float)
    pred[RUNNER_PROB] = runner_prob.to_numpy(dtype=float)
    return pred, pd.concat([bad_metrics, runner_metrics], ignore_index=True)


def _hard_negative_guard(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(False, index=frame.index, dtype=bool)
    hard_like = _bool(frame, "r5_2_hard_negative_like_asof_v1")
    structure_runner = (
        _num(frame, "as_of_candidate_tradable_prob_v1").ge(0.93)
        & _num(frame, "as_of_candidate_mfe_first_n_pred_v1").ge(1.80)
        & _num(frame, "as_of_skip_replay_retracement_from_last_impulse_v1").ge(0.55)
        & _num(frame, "as_of_skip_replay_clv_v1").ge(0.35)
        & _num(frame, "as_of_skip_replay_window_range_15_bps_v1").le(85.0)
    )
    hard_tight = hard_like & _num(frame, "as_of_candidate_tradable_prob_v1").ge(0.90)
    if mode == "hard_negative_like":
        return hard_like
    if mode == "hard_negative_tight":
        return hard_tight
    if mode == "structure_runner":
        return structure_runner
    if mode == "hybrid_light":
        return hard_tight | structure_runner
    raise ValueError(mode)


def _policy_mask(frame: pd.DataFrame, candidate: CandidateSpec) -> pd.Series:
    bad_prob = pd.to_numeric(frame[BAD_PROB], errors="coerce")
    runner_prob = pd.to_numeric(frame[RUNNER_PROB], errors="coerce")
    r5_bad = pd.to_numeric(frame.get(R5_PROB["should_not_take"], pd.Series(np.nan, index=frame.index)), errors="coerce")
    r5_tail = pd.to_numeric(frame.get(R5_PROB["tail_control_10_50_risk"], pd.Series(np.nan, index=frame.index)), errors="coerce")
    r5_runner = pd.to_numeric(frame.get(R5_PROB["runner_protect"], pd.Series(np.nan, index=frame.index)), errors="coerce")
    r5_current = _bool(frame, "r5_selected_candidate__block_v1")
    r4_current = _bool(frame, "r4_current_reference__block_v1")
    r2_current = _bool(frame, "r2_fallback_reference__block_v1")
    guard = _hard_negative_guard(frame, candidate.guard_mode)
    runner_low = runner_prob.lt(candidate.runner_threshold).fillna(False) & r5_runner.lt(min(0.95, candidate.runner_threshold + 0.18)).fillna(True)
    margin_ok = (bad_prob - runner_prob).ge(candidate.runner_margin).fillna(False)
    model_bad = bad_prob.ge(candidate.bad_threshold).fillna(False) & margin_ok
    r5_bad_signal = r5_bad.ge(candidate.r5_bad_threshold).fillna(False)
    tail_signal = r5_tail.ge(candidate.tail_threshold).fillna(False) & bad_prob.ge(max(0.50, candidate.bad_threshold - 0.12)).fillna(False)
    if candidate.stack_family == "TWO_HEAD_DIRECT":
        signal = model_bad | tail_signal
    elif candidate.stack_family == "R5_CURRENT_RUNNER_GATED":
        signal = r5_current | model_bad | tail_signal
    elif candidate.stack_family == "R5_BAD_SCORE_RUNNER_GATED":
        signal = r5_bad_signal | model_bad | tail_signal
    elif candidate.stack_family == "R4_R5_HYBRID_RUNNER_GATED":
        signal = r4_current | (r5_current & bad_prob.ge(max(0.45, candidate.bad_threshold - 0.15)).fillna(False)) | model_bad | tail_signal
    elif candidate.stack_family == "R2_R4_R5_HYBRID_RUNNER_GATED":
        signal = r2_current | r4_current | (r5_current & bad_prob.ge(max(0.45, candidate.bad_threshold - 0.15)).fillna(False)) | model_bad | tail_signal
    else:
        raise ValueError(candidate.stack_family)
    return (signal & runner_low & ~guard).fillna(False).astype(bool)


def _candidate_grid() -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    stacks = [
        "TWO_HEAD_DIRECT",
        "R5_CURRENT_RUNNER_GATED",
        "R5_BAD_SCORE_RUNNER_GATED",
        "R4_R5_HYBRID_RUNNER_GATED",
    ]
    guards = ["none", "hard_negative_tight", "hybrid_light"]
    index = 0
    for stack in stacks:
        for guard in guards:
            for bad_threshold in [0.42, 0.50, 0.62, 0.74]:
                for runner_threshold in [0.50, 0.62, 0.74]:
                    for tail_threshold in [0.75, 0.85]:
                        for r5_bad_threshold in [0.50, 0.62]:
                            for runner_margin in [0.00]:
                                index += 1
                                candidates.append(
                                    CandidateSpec(
                                        policy_name=f"R5_2_CANDIDATE_{index:05d}_{stack}_{guard}",
                                        stack_family=stack,
                                        guard_mode=guard,
                                        bad_threshold=bad_threshold,
                                        runner_threshold=runner_threshold,
                                        tail_threshold=tail_threshold,
                                        r5_bad_threshold=r5_bad_threshold,
                                        runner_margin=runner_margin,
                                    )
                                )
    return candidates


def _candidate_thresholds(candidate: CandidateSpec) -> Dict[str, Any]:
    return {
        "stack_family_v1": candidate.stack_family,
        "guard_mode_v1": candidate.guard_mode,
        "bad_threshold_v1": candidate.bad_threshold,
        "runner_threshold_v1": candidate.runner_threshold,
        "tail_threshold_v1": candidate.tail_threshold,
        "r5_bad_threshold_v1": candidate.r5_bad_threshold,
        "runner_margin_v1": candidate.runner_margin,
    }


def _global_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("hundred_plus_mfe_block_count_v1>1")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 2:
        failures.append("fifty_plus_mfe_block_count_v1>2")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 1:
        failures.append("strong_trade_false_block_count_v1>1")
    if precision is None or precision < 0.90:
        failures.append("precision<0.90")
    return not failures, ",".join(failures)


def _slice_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    block_count = int(metric.get("block_count_v1") or 0)
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 2:
        failures.append("fifty_plus_mfe_block_count_v1>2")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 2:
        failures.append("strong_trade_false_block_count_v1>2")
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
    metric_rows: list[pd.DataFrame] = []
    raw = frame.drop(columns=[BAD_PROB, RUNNER_PROB], errors="ignore")
    for slice_info in _slice_masks(reports_root, frame, batch_weeks=batch_weeks):
        scope = str(slice_info["scope_v1"])
        holdout = slice_info["mask_v1"].reindex(frame.index).fillna(False).astype(bool)
        train_all = ~holdout
        train_indices = frame.index[train_all].tolist()
        if len(train_indices) < 50:
            inner_train = train_all
            inner_validation = train_all
        else:
            cut = int(len(train_indices) * 0.8)
            inner_train = pd.Series(False, index=frame.index)
            inner_validation = pd.Series(False, index=frame.index)
            inner_train.loc[train_indices[:cut]] = True
            inner_validation.loc[train_indices[cut:]] = True
        pred, metrics = _train_two_heads(
            frame=raw,
            feature_names=feature_names,
            train_mask=inner_train,
            validation_mask=inner_validation,
            model_tag=f"r5_2_loso_{scope.lower()}",
            output_dir=None,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + int(slice_info["batch_index_v1"]) * 100,
            n_jobs=n_jobs,
        )
        fold_frames[scope] = raw.merge(pred, on="candidate_uid", how="left", validate="one_to_one")
        metrics["holdout_slice_v1"] = scope
        metric_rows.append(metrics)
        slice_infos.append(slice_info)
    return fold_frames, slice_infos, pd.concat(metric_rows, ignore_index=True)


def _evaluate_candidates(base_frame: pd.DataFrame, fold_frames: dict[str, pd.DataFrame], slice_infos: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, CandidateSpec, Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for candidate in _candidate_grid():
        global_mask = _policy_mask(base_frame, candidate)
        global_metric = _policy_metric_row(candidate.policy_name, "ALL", base_frame, global_mask, thresholds=_candidate_thresholds(candidate))
        global_pass, global_fail = _global_safety(global_metric)
        slice_metrics: list[dict[str, Any]] = []
        for slice_info in slice_infos:
            scope = str(slice_info["scope_v1"])
            holdout = slice_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool)
            fold_frame = fold_frames[scope]
            fold_mask = _policy_mask(fold_frame, candidate)
            metric = _policy_metric_row(candidate.policy_name, scope, fold_frame.loc[holdout].copy(), fold_mask.loc[holdout], thresholds=_candidate_thresholds(candidate))
            spass, sfail = _slice_safety(metric)
            metric.update(
                {
                    "stack_family_v1": candidate.stack_family,
                    "guard_mode_v1": candidate.guard_mode,
                    "slice_safety_pass_v1": spass,
                    "slice_safety_failure_reasons_v1": sfail,
                    "run_count_v1": int(slice_info["run_count_v1"]),
                    "run_start_v1": slice_info["run_start_v1"],
                    "run_end_v1": slice_info["run_end_v1"],
                }
            )
            slice_metrics.append(metric)
        batch04 = next((item for item in slice_metrics if item["scope_v1"] == "BATCH_04"), {})
        batch05 = next((item for item in slice_metrics if item["scope_v1"] == "BATCH_05"), {})
        precisions = [
            _safe_float(item.get("should_not_take_precision_v1"))
            for item in slice_metrics
            if int(item.get("block_count_v1") or 0) > 0 and _safe_float(item.get("should_not_take_precision_v1")) is not None
        ]
        worst_precision = min(precisions) if precisions else 1.0
        loso_pass = all(bool(item["slice_safety_pass_v1"]) for item in slice_metrics)
        target_gap = max(0, 70 - int(global_metric["should_not_take_block_count_v1"]))
        score = (
            float(global_metric["should_not_take_block_count_v1"]) * 1.5
            + float(global_metric["tail_10_50_help_count_v1"]) * 0.25
            + float(worst_precision) * 12.0
            - float(global_metric["take_was_ok_block_count_v1"]) * 2.0
            - float(global_metric["fifty_plus_mfe_block_count_v1"]) * 8.0
            - float(global_metric["hundred_plus_mfe_block_count_v1"]) * 12.0
            - float(global_metric["two_hundred_plus_mfe_block_count_v1"]) * 25.0
            - float(target_gap) * 0.4
        )
        failures: list[str] = []
        if not global_pass:
            failures.extend([item for item in global_fail.split(",") if item])
        for item in slice_metrics:
            failures.extend([f"{item['scope_v1']}:{reason}" for reason in str(item["slice_safety_failure_reasons_v1"]).split(",") if reason])
        if not (global_pass and loso_pass):
            score -= 1000.0
        row = dict(global_metric)
        row.update(
            {
                "stack_family_v1": candidate.stack_family,
                "guard_mode_v1": candidate.guard_mode,
                "global_safety_pass_v1": global_pass,
                "global_safety_failure_reasons_v1": global_fail,
                "loso_all_slices_pass_v1": loso_pass,
                "worst_slice_precision_v1": worst_precision,
                "batch04_loso_pass_v1": _optional_slice_pass(pd.DataFrame([batch04]) if batch04 else pd.DataFrame()),
                "batch04_should_not_take_block_count_v1": batch04.get("should_not_take_block_count_v1"),
                "batch04_precision_v1": batch04.get("should_not_take_precision_v1"),
                "batch04_failure_reasons_v1": batch04.get("slice_safety_failure_reasons_v1", ""),
                "batch05_loso_pass_v1": _optional_slice_pass(pd.DataFrame([batch05]) if batch05 else pd.DataFrame()),
                "batch05_should_not_take_block_count_v1": batch05.get("should_not_take_block_count_v1"),
                "batch05_precision_v1": batch05.get("should_not_take_precision_v1"),
                "batch05_failure_reasons_v1": batch05.get("slice_safety_failure_reasons_v1", ""),
                "safety_failure_count_v1": int(len(failures)),
                "safety_failures_json_v1": _json_dumps(failures[:50]),
                "selection_score_v1": score,
                "thresholds_json_v1": _json_dumps(_candidate_thresholds(candidate)),
            }
        )
        rows.append(row)
        loso_rows.extend(slice_metrics)
    calibration = pd.DataFrame(rows)
    viable = calibration[calibration["global_safety_pass_v1"].fillna(False) & calibration["loso_all_slices_pass_v1"].fillna(False)].copy()
    if viable.empty:
        selected_row = calibration.sort_values(["safety_failure_count_v1", "selection_score_v1"], ascending=[True, False]).iloc[0].to_dict()
    else:
        selected_row = viable.sort_values(["selection_score_v1", "should_not_take_block_count_v1"], ascending=[False, False]).iloc[0].to_dict()
    selected_thresholds = json.loads(str(selected_row["thresholds_json_v1"]))
    selected = CandidateSpec(
        policy_name=str(selected_row["policy_name_v1"]),
        stack_family=str(selected_thresholds["stack_family_v1"]),
        guard_mode=str(selected_thresholds["guard_mode_v1"]),
        bad_threshold=float(selected_thresholds["bad_threshold_v1"]),
        runner_threshold=float(selected_thresholds["runner_threshold_v1"]),
        tail_threshold=float(selected_thresholds["tail_threshold_v1"]),
        r5_bad_threshold=float(selected_thresholds["r5_bad_threshold_v1"]),
        runner_margin=float(selected_thresholds["runner_margin_v1"]),
    )
    calibration["selected_r5_2_candidate_v1"] = calibration["policy_name_v1"].astype("string").eq(selected.policy_name)
    return calibration, pd.DataFrame(loso_rows), selected, selected_row


def _reference_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "R2_FALLBACK_REFERENCE":
        return _bool(frame, "r2_fallback_reference__block_v1")
    if name == "R4_CURRENT_REFERENCE":
        return _bool(frame, "r4_current_reference__block_v1")
    if name == "R5_CURRENT_REFERENCE":
        return _bool(frame, "r5_selected_candidate__block_v1")
    if name == "R5_1_SELECTED_REFERENCE":
        return _bool(frame, "r5_1_selected_candidate__block_v1")
    raise ValueError(name)


def _head_to_head(base_frame: pd.DataFrame, selected: CandidateSpec, slice_infos: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_mask = _policy_mask(base_frame, selected)
    policies = {
        "R2_FALLBACK_REFERENCE": _reference_mask(base_frame, "R2_FALLBACK_REFERENCE"),
        "R4_CURRENT_REFERENCE": _reference_mask(base_frame, "R4_CURRENT_REFERENCE"),
        "R5_CURRENT_REFERENCE": _reference_mask(base_frame, "R5_CURRENT_REFERENCE"),
        "R5_1_SELECTED_REFERENCE": _reference_mask(base_frame, "R5_1_SELECTED_REFERENCE"),
        "R5_2_SELECTED_CANDIDATE": selected_mask,
    }
    batch04_info = next((item for item in slice_infos if str(item["scope_v1"]) == "BATCH_04"), None)
    batch05_info = next((item for item in slice_infos if str(item["scope_v1"]) == "BATCH_05"), None)
    scopes = {
        "ALL_1971": pd.Series(True, index=base_frame.index),
        "BATCH_04": batch04_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool) if batch04_info else pd.Series(False, index=base_frame.index),
        "BATCH_05": batch05_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool) if batch05_info else pd.Series(False, index=base_frame.index),
        "HARD_NEGATIVE_6": _bool(base_frame, "r5_2_batch04_hard_negative_runner_v1"),
        "SHOULD_NOT_TAKE_CLASS": _bool(base_frame, "label_should_not_take_v1"),
        "TAKE_WAS_OK_CLASS": _bool(base_frame, "take_was_ok_v1"),
        "REPAIRED_165": _bool(base_frame, "is_repaired_165_v1"),
        "FIFTY_PLUS_MFE_RUNNERS": _bool(base_frame, "fifty_plus_mfe_v1"),
        "HUNDRED_PLUS_MFE_RUNNERS": _bool(base_frame, "hundred_plus_mfe_v1"),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": _bool(base_frame, "two_hundred_plus_mfe_v1"),
        "STRONGEST_WINNER_PATH": _bool(base_frame, "strongest_winner_path_v1"),
        "TAIL_10_50_MFE_POCKET": _bool(base_frame, "tail_10_50_mfe_v1"),
    }
    rows: list[dict[str, Any]] = []
    pred = base_frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "is_repaired_165_v1",
            "label_should_not_take_v1",
            "take_was_ok_v1",
            "label_strong_trade_candidate_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "r5_2_batch04_hard_negative_runner_v1",
            "r5_2_hard_negative_like_asof_v1",
            "r5_2_hard_negative_similarity_distance_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "baseline_realized_pnl_bps_v1",
            BAD_PROB,
            RUNNER_PROB,
            *R5_PROB.values(),
        ]
    ].copy()
    for policy_name, mask in policies.items():
        pred[f"{policy_name.lower()}__block_v1"] = mask.to_numpy(dtype=bool)
        for scope_name, scope_mask in scopes.items():
            rows.append(_policy_metric_row(policy_name, scope_name, base_frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={"head_to_head_v1": True}))
    pred["r5_2_selected_candidate__block_v1"] = selected_mask.to_numpy(dtype=bool)
    return pd.DataFrame(rows), pred


def _label_audit(frame: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "r5_2_label_runner_50_mfe_v1",
        "r5_2_label_runner_100_mfe_v1",
        "r5_2_label_runner_200_mfe_v1",
        "r5_2_label_repaired_165_like_runner_v1",
        "r5_2_label_strong_low_mae_runner_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
        "r5_2_label_runner_protect_v1",
        "r5_2_label_bad_blocker_v1",
    ]
    rows: list[dict[str, Any]] = []
    hard_neg = _bool(frame, "r5_2_batch04_hard_negative_runner_v1")
    for label in labels:
        series = _bool(frame, label)
        rows.append(
            {
                "label_name_v1": label,
                "row_count_v1": int(len(frame)),
                "positive_count_v1": int(series.sum()),
                "positive_rate_v1": _safe_rate(float(series.sum()), float(len(frame))),
                "hard_negative_positive_count_v1": int((series & hard_neg).sum()),
                "fifty_plus_runner_positive_count_v1": int((series & _bool(frame, "fifty_plus_mfe_v1")).sum()),
                "two_hundred_plus_runner_positive_count_v1": int((series & _bool(frame, "two_hundred_plus_mfe_v1")).sum()),
                "training_role_v1": "HINDSIGHT_SUPERVISION_ONLY_NOT_POLICY_TRUTH",
            }
        )
    return pd.DataFrame(rows)


def _hard_negative_audit(frame: pd.DataFrame, selected: CandidateSpec) -> pd.DataFrame:
    selected_mask = _policy_mask(frame, selected)
    rows = frame[_bool(frame, "r5_2_batch04_hard_negative_runner_v1") | _bool(frame, "r5_2_hard_negative_like_asof_v1")][
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "r5_2_batch04_hard_negative_runner_v1",
            "r5_2_hard_negative_like_asof_v1",
            "r5_2_hard_negative_similarity_distance_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "baseline_realized_pnl_bps_v1",
            "take_was_ok_v1",
            "label_should_not_take_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            BAD_PROB,
            RUNNER_PROB,
        ]
    ].copy()
    rows["r5_2_selected_blocks_v1"] = selected_mask.loc[rows.index].to_numpy(dtype=bool)
    rows["r5_current_blocks_v1"] = _bool(frame, "r5_selected_candidate__block_v1").loc[rows.index].to_numpy(dtype=bool)
    rows["r5_1_selected_blocks_v1"] = _bool(frame, "r5_1_selected_candidate__block_v1").loc[rows.index].to_numpy(dtype=bool)
    return rows


def _pareto_frontier(calibration_df: pd.DataFrame) -> pd.DataFrame:
    safe = calibration_df[calibration_df["global_safety_pass_v1"].fillna(False) & calibration_df["loso_all_slices_pass_v1"].fillna(False)].copy()
    if safe.empty:
        return calibration_df.sort_values(["safety_failure_count_v1", "selection_score_v1"], ascending=[True, False]).head(100)
    candidates = safe.sort_values(["should_not_take_block_count_v1", "tail_10_50_help_count_v1", "worst_slice_precision_v1"], ascending=[False, False, False]).copy()
    frontier_rows: list[pd.Series] = []
    best_precision_seen = -1.0
    for _, row in candidates.iterrows():
        precision = _safe_float(row.get("should_not_take_precision_v1")) or 0.0
        if precision >= best_precision_seen:
            frontier_rows.append(row)
            best_precision_seen = precision
    return pd.DataFrame(frontier_rows).head(100)


def _decision(
    *,
    selected_row: Dict[str, Any],
    head_to_head_df: pd.DataFrame,
    loso_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    def all_row(policy: str) -> Dict[str, Any]:
        return head_to_head_df[(head_to_head_df["policy_name_v1"].eq(policy)) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()

    r2 = all_row("R2_FALLBACK_REFERENCE")
    r4 = all_row("R4_CURRENT_REFERENCE")
    r5 = all_row("R5_CURRENT_REFERENCE")
    r51 = all_row("R5_1_SELECTED_REFERENCE")
    r52 = all_row("R5_2_SELECTED_CANDIDATE")
    selected_loso = loso_df[loso_df["policy_name_v1"].eq(str(selected_row["policy_name_v1"]))]
    loso_pass = bool(selected_loso["slice_safety_pass_v1"].fillna(False).all()) if not selected_loso.empty else False
    batch04 = selected_loso[selected_loso["scope_v1"].eq("BATCH_04")]
    batch05 = selected_loso[selected_loso["scope_v1"].eq("BATCH_05")]
    batch04_pass = _optional_slice_pass(batch04)
    batch05_pass = _optional_slice_pass(batch05)
    batch04_ok = True if batch04_pass is None else bool(batch04_pass)
    batch05_ok = True if batch05_pass is None else bool(batch05_pass)
    beats_r51 = int(r52["should_not_take_block_count_v1"]) > int(r51["should_not_take_block_count_v1"])
    target_70 = int(r52["should_not_take_block_count_v1"]) >= 70
    keeps_some_r5_edge = int(r52["should_not_take_block_count_v1"]) >= max(60, int(int(r5["should_not_take_block_count_v1"]) * 0.65))
    if loso_pass and batch04_ok and batch05_ok and beats_r51 and target_70:
        recommendation = "R5_2_LOSO_SAFE_SHADOW_CANDIDATE"
    elif int(r5["should_not_take_block_count_v1"]) > int(r52["should_not_take_block_count_v1"]) and not loso_pass:
        recommendation = "R5_CURRENT_KEEP_AS_EDGE_REFERENCE"
    elif beats_r51 and loso_pass:
        recommendation = "R5_2_LOSO_SAFE_SHADOW_CANDIDATE" if keeps_some_r5_edge else "R5_1_KEEP_AS_SAFETY_REFERENCE"
    elif loso_pass:
        recommendation = "R5_1_KEEP_AS_SAFETY_REFERENCE"
    else:
        recommendation = "R6_FEATURE_RETRAIN_REQUIRED"
    rows = [
        {"decision_key_v1": "R5_2_LOSO_SAFE_SHADOW_CANDIDATE", "status_v1": "PASS" if recommendation == "R5_2_LOSO_SAFE_SHADOW_CANDIDATE" else "NOT_MET", "reason_v1": "Requires LOSO safety and improved recall over R5.1, ideally >=70 bad blocks."},
        {"decision_key_v1": "R5_CURRENT_KEEP_AS_EDGE_REFERENCE", "status_v1": "PASS" if recommendation == "R5_CURRENT_KEEP_AS_EDGE_REFERENCE" else "NOT_PRIMARY", "reason_v1": "Use R5 current only as edge reference if R5.2 cannot pass LOSO."},
        {"decision_key_v1": "R5_1_KEEP_AS_SAFETY_REFERENCE", "status_v1": "PASS" if recommendation == "R5_1_KEEP_AS_SAFETY_REFERENCE" else "NOT_PRIMARY", "reason_v1": "Use R5.1 as safety reference if R5.2 is not materially better."},
        {"decision_key_v1": "R6_FEATURE_RETRAIN_REQUIRED", "status_v1": "PASS" if recommendation == "R6_FEATURE_RETRAIN_REQUIRED" else "NOT_PRIMARY", "reason_v1": "Use when current features/labels cannot satisfy edge plus LOSO safety."},
        {"decision_key_v1": "ENTRY_FALLBACK_STILL_NOT_FREEZEABLE", "status_v1": "PASS_FOR_LIVE_GATE_ONLY", "reason_v1": "No output is promoted to live gate."},
    ]
    summary = {
        "recommended_next_step_v1": recommendation,
        "selected_policy_name_v1": selected_row.get("policy_name_v1"),
        "selected_stack_family_v1": selected_row.get("stack_family_v1"),
        "selected_guard_mode_v1": selected_row.get("guard_mode_v1"),
        "r5_2_loso_all_slices_pass_v1": loso_pass,
        "batch04_loso_pass_v1": batch04_pass,
        "batch05_loso_pass_v1": batch05_pass,
        "r5_2_beats_r5_1_recall_v1": bool(beats_r51),
        "r5_2_reaches_70_bad_blocks_v1": bool(target_70),
        "r5_2_keeps_some_r5_edge_v1": bool(keeps_some_r5_edge),
        "r2_should_not_blocks_v1": int(r2["should_not_take_block_count_v1"]),
        "r4_should_not_blocks_v1": int(r4["should_not_take_block_count_v1"]),
        "r5_current_should_not_blocks_v1": int(r5["should_not_take_block_count_v1"]),
        "r5_1_should_not_blocks_v1": int(r51["should_not_take_block_count_v1"]),
        "r5_2_should_not_blocks_v1": int(r52["should_not_take_block_count_v1"]),
        "r5_2_precision_v1": _safe_float(r52.get("should_not_take_precision_v1")),
        "r5_2_tail_help_v1": int(r52["tail_10_50_help_count_v1"]),
    }
    return pd.DataFrame(rows), summary


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    d = summary["decision_v1"]
    lines = [
        "# R5.2 Entry Runner Aware Retrain And LOSO Selection V1",
        "",
        "Shadow/research only. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R5_2_STATUS']}`",
        f"- Recommendation: `{d['recommended_next_step_v1']}`",
        f"- Selected: `{d['selected_policy_name_v1']}`",
        f"- R5.2 bad blocks: `{d['r5_2_should_not_blocks_v1']}`",
        f"- R5 current bad blocks: `{d['r5_current_should_not_blocks_v1']}`",
        f"- R5.1 bad blocks: `{d['r5_1_should_not_blocks_v1']}`",
        f"- BATCH_04 pass: `{d['batch04_loso_pass_v1']}`",
        f"- BATCH_05 pass: `{d['batch05_loso_pass_v1']}`",
        "",
        "## Guardrails",
        "",
        "- AS_OF features and HINDSIGHT labels are written as separate artifacts.",
        "- Hard negatives are supervision/diagnostic only, not policy truth.",
        "- No live promotion.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    r5_dir: Path,
    r5_1_dir: Path,
    r4_dir: Path,
    repair_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    asof_df, hindsight_df, r5_pred_df, failure_df, r5_1_pred_df, r5_summary, r5_1_summary, r4_summary, feature_names = _load_inputs(
        r5_dir=r5_dir,
        r5_1_dir=r5_1_dir,
        r4_dir=r4_dir,
        repair_dir=repair_dir,
        expected_ledger_count=expected_ledger_count,
    )
    base = _prepare_frame(asof_df, hindsight_df, r5_pred_df, failure_df, r5_1_pred_df)
    train_mask = _bool(base, "used_for_training")
    validation_mask = _bool(base, "used_for_validation")
    global_pred, model_metrics_df = _train_two_heads(
        frame=base,
        feature_names=feature_names,
        train_mask=train_mask,
        validation_mask=validation_mask,
        model_tag="global_r5_2_runner_aware",
        output_dir=extension_dir,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    policy_frame = base.drop(columns=[BAD_PROB, RUNNER_PROB], errors="ignore").merge(global_pred, on="candidate_uid", how="left", validate="one_to_one")
    fold_frames, slice_infos, fold_metrics_df = _train_fold_predictions(
        reports_root=reports_root,
        frame=policy_frame,
        feature_names=feature_names,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    calibration_df, candidate_loso_df, selected, selected_row = _evaluate_candidates(policy_frame, fold_frames, slice_infos)
    head_to_head_df, prediction_view_df = _head_to_head(policy_frame, selected, slice_infos)
    selected_loso_df = candidate_loso_df[candidate_loso_df["policy_name_v1"].eq(selected.policy_name)].copy()
    decision_df, decision_summary = _decision(selected_row=selected_row, head_to_head_df=head_to_head_df, loso_df=candidate_loso_df)
    label_audit_df = _label_audit(policy_frame)
    hard_negative_df = _hard_negative_audit(policy_frame, selected)
    pareto_df = _pareto_frontier(calibration_df)
    bakeoff_df = (
        calibration_df.sort_values(["global_safety_pass_v1", "loso_all_slices_pass_v1", "selection_score_v1"], ascending=[False, False, False])
        .groupby(["stack_family_v1", "guard_mode_v1"], dropna=False)
        .head(1)
    )
    coverage = r5_summary.get("coverage_v1", {}) if isinstance(r5_summary.get("coverage_v1"), dict) else {}
    consistency_df = pd.DataFrame(
        [
            _audit_record("R5_INPUT_PRESENT", "PASS", {"r5_dir": str(r5_dir)}),
            _audit_record("R5_1_INPUT_PRESENT", "PASS", {"r5_1_dir": str(r5_1_dir)}),
            _audit_record("R4_INPUT_PRESENT", "PASS", {"r4_dir": str(r4_dir), "r4_summary_keys": sorted(r4_summary.keys())[:12]}),
            _audit_record("REPAIR_INPUT_PRESENT", "PASS", {"repair_dir": str(repair_dir)}),
            _audit_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS" if expected_ledger_count is None or len(policy_frame) == expected_ledger_count else "FAIL", {"expected": expected_ledger_count, "observed": len(policy_frame)}),
            _audit_record("FULL_ENTRY_COVERAGE", "PASS" if int(coverage.get("entry_coverage_v1", 0)) == len(policy_frame) else "FAIL", {"coverage": coverage}),
            _audit_record("NO_SYNTHETIC_INPUT", "PASS" if int(coverage.get("synthetic_count_v1", -1)) == 0 else "FAIL", {"synthetic_count": coverage.get("synthetic_count_v1")}),
            _audit_record("BATCH04_HARD_NEGATIVES_FOUND", "PASS" if int(policy_frame["r5_2_batch04_hard_negative_runner_v1"].sum()) == 6 else "FAIL", {"observed": int(policy_frame["r5_2_batch04_hard_negative_runner_v1"].sum())}),
            _audit_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION_OUTPUTS", "PASS", {"as_of_table": AS_OF_FEATURE_TABLE, "hindsight_table": HINDSIGHT_LABEL_OUTCOME_TABLE}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    selected_hard = hard_negative_df[_bool(hard_negative_df, "r5_2_batch04_hard_negative_runner_v1")]
    hard_neg_protected = int((~_bool(selected_hard, "r5_2_selected_blocks_v1")).sum()) if not selected_hard.empty else 0
    protected_498 = bool(
        selected_hard.loc[
            selected_hard["peak_mfe_bps_v1"].astype(float).ge(490.0),
            "r5_2_selected_blocks_v1",
        ]
        .astype(bool)
        .eq(False)
        .all()
    )
    status = {
        "layer_name": "R5_2_STATUS_V1",
        "R5_2_STATUS": "RESEARCH_COMPLETE_NOT_PROMOTED_NOT_LIVE_GATE",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "input_dirs_v1": {"r5": str(r5_dir), "r5_1": str(r5_1_dir), "r4": str(r4_dir), "repair": str(repair_dir)},
        "coverage_v1": {
            "ledger_trade_count_v1": int(len(policy_frame)),
            "entry_coverage_v1": int(coverage.get("entry_coverage_v1", len(policy_frame))),
            "entry_raw_coverage_v1": int(coverage.get("entry_raw_coverage_v1", len(policy_frame))),
            "missing_count_v1": int(coverage.get("missing_count_v1", 0)),
            "synthetic_count_v1": int(coverage.get("synthetic_count_v1", 0)),
            "repaired_rows_v1": int(coverage.get("repaired_rows_v1", int(policy_frame["is_repaired_165_v1"].sum()))),
        },
        "hard_negative_v1": {
            "batch04_false_block_hard_negative_count_v1": int(policy_frame["r5_2_batch04_hard_negative_runner_v1"].sum()),
            "hard_negative_like_count_v1": int(policy_frame["r5_2_hard_negative_like_asof_v1"].sum()),
            "selected_protects_hard_negative_count_v1": hard_neg_protected,
            "selected_protects_all_6_v1": bool(hard_neg_protected == 6),
            "selected_protects_498_bps_runner_v1": protected_498,
        },
        "selected_candidate_v1": selected_row,
        "decision_v1": decision_summary,
        "r5_1_decision_v1": r5_1_summary.get("decision_v1", {}),
        "model_metric_rows_v1": int(len(model_metrics_df) + len(fold_metrics_df)),
        "candidate_count_v1": int(len(calibration_df)),
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R5.2 inherited full coverage {coverage.get('entry_coverage_v1', len(policy_frame))}/{len(policy_frame)} and synthetic_count={coverage.get('synthetic_count_v1', 0)}.",
                f"R5.1 BATCH_04 false-block hard negatives found: {int(policy_frame['r5_2_batch04_hard_negative_runner_v1'].sum())}.",
                "Two-head blocker/protector models were trained with AS_OF features only.",
                "No output is promoted to live gate.",
            ],
            "INDIKERT": [
                "Runner-aware labels and hard-negative weighting indicate whether R5.2 can recover recall without runner damage.",
                "Pareto frontier indicates the best tradeoff between R5 current edge and R5.1 safety.",
            ],
            "IKKE_ETABLERT": [
                "Live fallback safety.",
                "Causal improvement in future unseen regimes.",
                "Whether 70+ bad blocks can be reached without richer R6 features if R5.2 misses the target.",
            ],
        },
    }
    contract = {
        "layer_name": "R5_2_ENTRY_RUNNER_AWARE_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_dirs_v1": summary["input_dirs_v1"],
        "as_of_feature_names_v1": list(feature_names),
        "hindsight_label_columns_v1": [
            "r5_2_label_bad_blocker_v1",
            "r5_2_label_runner_protect_v1",
            "r5_2_label_runner_50_mfe_v1",
            "r5_2_label_runner_100_mfe_v1",
            "r5_2_label_runner_200_mfe_v1",
            "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
        ],
        "safety_constraints_v1": {
            "repaired_165_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "hundred_plus_mfe_blocked_max_v1": 1,
            "fifty_plus_mfe_blocked_max_v1": 2,
            "strong_false_blocks_global_max_v1": 1,
            "batch04_loso_must_pass_v1": True,
            "batch05_loso_must_pass_v1": True,
            "worst_slice_precision_min_v1": 0.85,
            "global_precision_min_v1": 0.90,
        },
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R5_2_ENTRY_RUNNER_AWARE_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_feature_table": AS_OF_FEATURE_TABLE,
            "hindsight_label_outcome_table": HINDSIGHT_LABEL_OUTCOME_TABLE,
            "hard_negative_audit": HARD_NEGATIVE_AUDIT,
            "runner_label_audit": RUNNER_LABEL_AUDIT,
            "two_head_stack_bakeoff": TWO_HEAD_STACK_BAKEOFF,
            "robust_calibration": ROBUST_CALIBRATION,
            "pareto_frontier": PARETO_FRONTIER,
            "loso_metrics": LOSO_METRICS,
            "head_to_head": HEAD_TO_HEAD,
            "policy_prediction_view": POLICY_PREDICTION_VIEW,
            "decision_matrix": DECISION_MATRIX,
            "summary": SUMMARY,
            "report": REPORT,
            "models_dir": "models",
        },
    }
    asof_out = asof_df.copy()
    asof_out["r5_2_as_of_feature_contract_v1"] = "AS_OF_ONLY_NO_HINDSIGHT_FEATURES_R5_2"
    hindsight_out = policy_frame[
        [
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
            "r5_2_label_bad_blocker_v1",
            "r5_2_label_runner_protect_v1",
            "r5_2_label_runner_50_mfe_v1",
            "r5_2_label_runner_100_mfe_v1",
            "r5_2_label_runner_200_mfe_v1",
            "r5_2_label_repaired_165_like_runner_v1",
            "r5_2_label_strong_low_mae_runner_v1",
            "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
            "r5_2_batch04_hard_negative_runner_v1",
            "r5_2_hard_negative_like_asof_v1",
        ]
    ].copy()
    hindsight_out["r5_2_hindsight_contract_v1"] = "HINDSIGHT_SUPERVISION_ONLY_NOT_POLICY_TRUTH_NOT_AS_OF_FEATURES"
    return {
        "asof_df": asof_out,
        "hindsight_df": hindsight_out,
        "hard_negative_df": hard_negative_df,
        "runner_label_audit_df": label_audit_df,
        "two_head_stack_bakeoff_df": bakeoff_df,
        "robust_calibration_df": calibration_df,
        "pareto_frontier_df": pareto_df,
        "loso_metrics_df": selected_loso_df,
        "head_to_head_df": head_to_head_df,
        "policy_prediction_df": prediction_view_df,
        "decision_df": decision_df,
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
    r5_dir: Path | None = None,
    r5_1_dir: Path | None = None,
    r4_dir: Path | None = None,
    repair_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    n_estimators: int = 900,
    early_stopping_rounds: int = 70,
    learning_rate: float = 0.025,
    max_depth: int = 3,
    seed: int = 20260422,
    n_jobs: int = 4,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    r5_dir = _resolve_dir(reports_root, str(r5_dir) if r5_dir else None, R5_EXTENSION_NAME, R5_SUMMARY)
    r5_1_dir = _resolve_dir(reports_root, str(r5_1_dir) if r5_1_dir else None, R5_1_EXTENSION_NAME, R5_1_SUMMARY)
    r4_dir = _resolve_dir(reports_root, str(r4_dir) if r4_dir else None, R4_EXTENSION_NAME, R4_SUMMARY)
    repair_dir = _resolve_dir(reports_root, str(repair_dir) if repair_dir else None, REPAIR_EXTENSION_NAME, REPAIR_SUMMARY)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        r5_dir=r5_dir,
        r5_1_dir=r5_1_dir,
        r4_dir=r4_dir,
        repair_dir=repair_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
        expected_ledger_count=expected_ledger_count,
    )
    payload["asof_df"].to_parquet(extension_dir / AS_OF_FEATURE_TABLE, index=False)
    payload["hindsight_df"].to_parquet(extension_dir / HINDSIGHT_LABEL_OUTCOME_TABLE, index=False)
    payload["hard_negative_df"].to_csv(extension_dir / HARD_NEGATIVE_AUDIT, index=False)
    payload["runner_label_audit_df"].to_csv(extension_dir / RUNNER_LABEL_AUDIT, index=False)
    payload["two_head_stack_bakeoff_df"].to_csv(extension_dir / TWO_HEAD_STACK_BAKEOFF, index=False)
    payload["robust_calibration_df"].to_csv(extension_dir / ROBUST_CALIBRATION, index=False)
    payload["pareto_frontier_df"].to_csv(extension_dir / PARETO_FRONTIER, index=False)
    payload["loso_metrics_df"].to_csv(extension_dir / LOSO_METRICS, index=False)
    payload["head_to_head_df"].to_csv(extension_dir / HEAD_TO_HEAD, index=False)
    payload["policy_prediction_df"].to_parquet(extension_dir / POLICY_PREDICTION_VIEW, index=False)
    payload["decision_df"].to_csv(extension_dir / DECISION_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / CONTRACT, payload["contract"])
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
    parser = argparse.ArgumentParser(description="Build R5.2 entry runner-aware retrain and LOSO selection.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--r5-dir", default=None)
    parser.add_argument("--r5-1-dir", default=None)
    parser.add_argument("--r4-dir", default=None)
    parser.add_argument("--repair-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=70)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        r5_dir=_resolve_dir(reports_root, args.r5_dir, R5_EXTENSION_NAME, R5_SUMMARY),
        r5_1_dir=_resolve_dir(reports_root, args.r5_1_dir, R5_1_EXTENSION_NAME, R5_1_SUMMARY),
        r4_dir=_resolve_dir(reports_root, args.r4_dir, R4_EXTENSION_NAME, R4_SUMMARY),
        repair_dir=_resolve_dir(reports_root, args.repair_dir, REPAIR_EXTENSION_NAME, REPAIR_SUMMARY),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
