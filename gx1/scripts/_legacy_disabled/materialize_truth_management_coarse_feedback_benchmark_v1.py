#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
COARSE_TEACHER_VIEW_FILE = "truth_management_coarse_teacher_v1.parquet"
SUMMARY_FILE = "truth_management_coarse_feedback_benchmark_summary_v1.json"
PREDICTIONS_FILE = "truth_management_coarse_feedback_benchmark_hold_predictions_v1.parquet"
THRESHOLD_SWEEP_FILE = "truth_management_coarse_feedback_benchmark_threshold_sweep_v1.csv"

SPLIT_ORDER = ["TRAIN", "VALIDATION", "HOLDOUT"]
FEATURE_BUNDLES: Dict[str, List[str]] = {
    "P1_MINIMAL_COARSE_RAW_V1": [
        "shadow_score_v1",
        "shadow_score_coarse_band_v1",
        "overlay_session_axis_v1",
        "overlay_hold_age_axis_v1",
        "overlay_giveback_axis_v1",
    ],
    "P1_COARSE_RAW_PLUS_CONTEXT_V1": [
        "shadow_score_v1",
        "shadow_score_coarse_band_v1",
        "overlay_session_axis_v1",
        "overlay_hold_age_axis_v1",
        "overlay_giveback_axis_v1",
        "as_of_management_core_minutes_held_at_anchor_v1",
        "as_of_management_core_giveback_ratio_from_peak_v1",
        "as_of_atr_bps_v1",
    ],
}
REQUIRED_COLUMNS = [
    "management_row_key_v1",
    "split_bucket_v1",
    "observed_action_v1",
    "coarse_teacher_binary_target_v1",
    "coarse_teacher_binary_target_eligible_v1",
    "coarse_teacher_feedback_label_v1",
    "realized_pnl_bps",
    "hold_longer_extra_value_bps_v1",
    "recommended_coarse_grid_name_v1",
    "recommended_coarse_grid_value_v1",
    "recommended_coarse_grid_viable_cell_v1",
    "shadow_score_v1",
    "shadow_bucket_status_v1",
    "shadow_bucket_rank_v1",
    "shadow_score_coarse_band_v1",
    "overlay_session_axis_v1",
    "overlay_hold_age_axis_v1",
    "overlay_giveback_axis_v1",
    "as_of_management_core_minutes_held_at_anchor_v1",
    "as_of_management_core_giveback_ratio_from_peak_v1",
    "as_of_atr_bps_v1",
]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{frame_name} missing required columns: {missing}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _prepare_preprocessor(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[List[str], List[str], ColumnTransformer]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for feature_name in feature_columns:
        if pd.api.types.is_numeric_dtype(df[feature_name]):
            numeric_cols.append(str(feature_name))
        else:
            categorical_cols.append(str(feature_name))
    transformers: List[Tuple[str, Any, Sequence[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return numeric_cols, categorical_cols, ColumnTransformer(transformers=transformers, remainder="drop")


def _classification_metrics(frame: pd.DataFrame, estimator: Any, feature_columns: Sequence[str]) -> Dict[str, Any]:
    if frame.empty:
        return {"rows_v1": 0}
    y_true = frame["coarse_teacher_binary_target_v1"].astype(int).to_numpy(dtype=int)
    X = frame[list(feature_columns)].copy()
    pred_labels = np.asarray(estimator.predict(X), dtype=object)
    pred_labels_int = pd.Series(pred_labels).astype("string").eq("1").astype(int).to_numpy(dtype=int)
    if hasattr(estimator, "predict_proba"):
        raw_proba = np.asarray(estimator.predict_proba(X), dtype=float)
        classes_ = [str(value) for value in getattr(estimator, "classes_", ["0", "1"])]
        positive_index = classes_.index("1")
        positive_proba = raw_proba[:, positive_index]
    else:
        positive_rate = float(pd.Series(y_true).mean()) if len(y_true) else 0.0
        positive_proba = np.full(len(frame), positive_rate, dtype=float)
    try:
        roc_auc = _safe_float(roc_auc_score(y_true, positive_proba))
    except Exception:
        roc_auc = None
    try:
        pr_auc = _safe_float(average_precision_score(y_true, positive_proba))
    except Exception:
        pr_auc = None
    try:
        brier = _safe_float(brier_score_loss(y_true, positive_proba))
    except Exception:
        brier = None
    try:
        ll = _safe_float(log_loss(y_true, np.clip(positive_proba, 1e-9, 1 - 1e-9), labels=[0, 1]))
    except Exception:
        ll = None
    return {
        "rows_v1": int(len(frame)),
        "positive_rate_v1": _safe_float(np.mean(y_true)),
        "predicted_positive_rate_v1": _safe_float(np.mean(pred_labels_int)),
        "roc_auc_v1": roc_auc,
        "pr_auc_v1": pr_auc,
        "brier_score_v1": brier,
        "log_loss_v1": ll,
        "accuracy_v1": _safe_float(accuracy_score(y_true, pred_labels_int)),
        "precision_v1": _safe_float(precision_score(y_true, pred_labels_int, zero_division=0)),
        "recall_v1": _safe_float(recall_score(y_true, pred_labels_int, zero_division=0)),
        "f1_v1": _safe_float(f1_score(y_true, pred_labels_int, zero_division=0)),
        "mean_realized_pnl_bps_v1": _safe_float(pd.to_numeric(frame["realized_pnl_bps"], errors="coerce").mean()),
        "mean_hold_longer_extra_value_bps_v1": _safe_float(
            pd.to_numeric(frame["hold_longer_extra_value_bps_v1"], errors="coerce").mean()
        ),
    }


def _probability_metrics(frame: pd.DataFrame, positive_proba: Sequence[float]) -> Dict[str, Any]:
    if frame.empty:
        return {"rows_v1": 0}
    probs = np.asarray(positive_proba, dtype=float)
    y_true = frame["coarse_teacher_binary_target_v1"].astype(int).to_numpy(dtype=int)
    pred_binary = (probs >= 0.5).astype(int)
    try:
        roc_auc = _safe_float(roc_auc_score(y_true, probs))
    except Exception:
        roc_auc = None
    try:
        pr_auc = _safe_float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = None
    try:
        brier = _safe_float(brier_score_loss(y_true, probs))
    except Exception:
        brier = None
    try:
        ll = _safe_float(log_loss(y_true, np.clip(probs, 1e-9, 1 - 1e-9), labels=[0, 1]))
    except Exception:
        ll = None
    return {
        "rows_v1": int(len(frame)),
        "positive_rate_v1": _safe_float(np.mean(y_true)),
        "predicted_positive_rate_v1": _safe_float(np.mean(pred_binary)),
        "roc_auc_v1": roc_auc,
        "pr_auc_v1": pr_auc,
        "brier_score_v1": brier,
        "log_loss_v1": ll,
        "accuracy_v1": _safe_float(accuracy_score(y_true, pred_binary)),
        "precision_v1": _safe_float(precision_score(y_true, pred_binary, zero_division=0)),
        "recall_v1": _safe_float(recall_score(y_true, pred_binary, zero_division=0)),
        "f1_v1": _safe_float(f1_score(y_true, pred_binary, zero_division=0)),
        "mean_realized_pnl_bps_v1": _safe_float(pd.to_numeric(frame["realized_pnl_bps"], errors="coerce").mean()),
        "mean_hold_longer_extra_value_bps_v1": _safe_float(
            pd.to_numeric(frame["hold_longer_extra_value_bps_v1"], errors="coerce").mean()
        ),
    }


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    estimator: Any,
    feature_columns: Sequence[str],
    model_name: str,
    model_family: str,
    feature_bundle_name: str,
) -> pd.DataFrame:
    out = frame[
        [
            "management_row_key_v1",
            "split_bucket_v1",
            "observed_action_v1",
            "coarse_teacher_feedback_label_v1",
            "coarse_teacher_binary_target_v1",
            "realized_pnl_bps",
            "hold_longer_extra_value_bps_v1",
            "recommended_coarse_grid_name_v1",
            "recommended_coarse_grid_value_v1",
            "recommended_coarse_grid_viable_cell_v1",
        ]
    ].copy()
    X = frame[list(feature_columns)].copy()
    out["model_name_v1"] = str(model_name)
    out["model_family_v1"] = str(model_family)
    out["feature_bundle_name_v1"] = str(feature_bundle_name)
    if hasattr(estimator, "predict_proba"):
        raw_proba = np.asarray(estimator.predict_proba(X), dtype=float)
        classes_ = [str(value) for value in getattr(estimator, "classes_", ["0", "1"])]
        positive_index = classes_.index("1")
        positive_proba = raw_proba[:, positive_index]
    else:
        positive_rate = float(frame["coarse_teacher_binary_target_v1"].astype(int).mean()) if len(frame) else 0.0
        positive_proba = np.full(len(frame), positive_rate, dtype=float)
    out["predicted_positive_prob_v1"] = positive_proba
    out["predicted_positive_label_v1"] = out["predicted_positive_prob_v1"].ge(0.5)
    return out


def build_management_coarse_feedback_benchmark_payload(
    reports_root: Path,
    *,
    teacher_view_path: Path | None = None,
    action_name: str = "HOLD",
    min_train_rows: int = 100,
    min_validation_rows: int = 20,
    min_holdout_rows: int = 20,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    teacher_view_path = (
        Path(teacher_view_path).expanduser().resolve()
        if teacher_view_path is not None
        else (reports_root / COARSE_TEACHER_VIEW_FILE)
    )
    if not teacher_view_path.exists():
        raise FileNotFoundError(f"Coarse teacher view missing: {teacher_view_path}")

    view_df = pd.read_parquet(teacher_view_path)
    if view_df.empty:
        raise RuntimeError("truth_management_coarse_teacher_v1.parquet is empty")
    _require_columns(view_df, REQUIRED_COLUMNS, "teacher_view_df")

    for column in [
        "coarse_teacher_binary_target_v1",
        "realized_pnl_bps",
        "hold_longer_extra_value_bps_v1",
        "shadow_score_v1",
        "as_of_management_core_minutes_held_at_anchor_v1",
        "as_of_management_core_giveback_ratio_from_peak_v1",
        "as_of_atr_bps_v1",
    ]:
        view_df[column] = pd.to_numeric(view_df[column], errors="coerce")
    view_df["coarse_teacher_binary_target_eligible_v1"] = view_df["coarse_teacher_binary_target_eligible_v1"].astype(bool)

    action_df = view_df.loc[
        view_df["observed_action_v1"].astype("string").eq(str(action_name))
        & view_df["coarse_teacher_binary_target_eligible_v1"].astype(bool)
    ].copy()
    if action_df.empty:
        raise RuntimeError(f"No eligible rows for action {action_name}")

    split_counts = {
        split_name: int(action_df["split_bucket_v1"].astype("string").eq(split_name).sum())
        for split_name in SPLIT_ORDER
    }
    if split_counts["TRAIN"] < min_train_rows:
        raise RuntimeError(f"Insufficient TRAIN rows for {action_name}: {split_counts['TRAIN']}")
    if split_counts["VALIDATION"] < min_validation_rows:
        raise RuntimeError(f"Insufficient VALIDATION rows for {action_name}: {split_counts['VALIDATION']}")
    if split_counts["HOLDOUT"] < min_holdout_rows:
        raise RuntimeError(f"Insufficient HOLDOUT rows for {action_name}: {split_counts['HOLDOUT']}")

    train_df = action_df.loc[action_df["split_bucket_v1"].astype("string").eq("TRAIN")].copy()
    if train_df["coarse_teacher_binary_target_v1"].nunique(dropna=True) <= 1:
        raise RuntimeError(f"{action_name} TRAIN target is single-class; cannot train benchmark")

    class_balance_by_split = {
        split_name: {
            str(key): int(value)
            for key, value in (
                action_df.loc[action_df["split_bucket_v1"].astype("string").eq(split_name), "coarse_teacher_binary_target_v1"]
                .astype("Int64")
                .astype("string")
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        }
        for split_name in SPLIT_ORDER
    }
    feedback_action_balance_status_v1 = {
        action: (
            "BALANCED_POSITIVE_AND_NEGATIVE"
            if stats["positive_rows_v1"] > 0 and stats["negative_rows_v1"] > 0
            else (
                "POSITIVE_ONLY"
                if stats["positive_rows_v1"] > 0 and stats["negative_rows_v1"] == 0
                else (
                    "NEGATIVE_ONLY"
                    if stats["negative_rows_v1"] > 0 and stats["positive_rows_v1"] == 0
                    else "NO_ELIGIBLE_ROWS"
                )
            )
        )
        for action, stats in {
            action: {
                "positive_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                "negative_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()),
            }
            for action, part in view_df.loc[view_df["coarse_teacher_binary_target_eligible_v1"].astype(bool)].groupby(
                view_df.loc[view_df["coarse_teacher_binary_target_eligible_v1"].astype(bool), "observed_action_v1"].astype("string"),
                dropna=False,
            )
        }.items()
    }

    train_positive_rate = float(train_df["coarse_teacher_binary_target_v1"].astype(float).mean())
    bucket_status_rate_map = (
        train_df.groupby(train_df["shadow_bucket_status_v1"].astype("string"), dropna=False)["coarse_teacher_binary_target_v1"]
        .mean()
        .to_dict()
    )
    bucket_rank_rate_map = (
        train_df.groupby("shadow_bucket_rank_v1", dropna=False)["coarse_teacher_binary_target_v1"]
        .mean()
        .to_dict()
    )

    def _current_bucket_probs(frame: pd.DataFrame) -> np.ndarray:
        status_series = frame["shadow_bucket_status_v1"].astype("string")
        rank_series = pd.to_numeric(frame["shadow_bucket_rank_v1"], errors="coerce")
        probs = []
        for status_value, rank_value in zip(status_series.tolist(), rank_series.tolist()):
            if status_value in bucket_status_rate_map:
                probs.append(float(bucket_status_rate_map[status_value]))
            elif rank_value in bucket_rank_rate_map:
                probs.append(float(bucket_rank_rate_map[rank_value]))
            else:
                probs.append(float(train_positive_rate))
        return np.asarray(probs, dtype=float)

    current_bucket_baseline_metrics_v1: Dict[str, Any] = {}
    current_bucket_prediction_frames: List[pd.DataFrame] = []
    for split_name in SPLIT_ORDER:
        split_df = action_df.loc[action_df["split_bucket_v1"].astype("string").eq(split_name)].copy()
        split_probs = _current_bucket_probs(split_df)
        current_bucket_baseline_metrics_v1[split_name] = _probability_metrics(split_df, split_probs)
        if split_name in {"VALIDATION", "HOLDOUT"} and not split_df.empty:
            current_bucket_prediction_frames.append(
                pd.DataFrame(
                    {
                        "management_row_key_v1": split_df["management_row_key_v1"].astype("string").tolist(),
                        "split_bucket_v1": split_df["split_bucket_v1"].astype("string").tolist(),
                        "predicted_positive_prob_v1": split_probs.tolist(),
                        "predicted_positive_label_v1": (split_probs >= 0.5).tolist(),
                        "benchmark_name_v1": ["CURRENT_BUCKET_STATUS_BASELINE_V1"] * int(len(split_df)),
                    }
                )
            )

    training_feature_bundle_rows: List[Dict[str, Any]] = []
    predictions_frames: List[pd.DataFrame] = []
    fitted_models: Dict[str, Any] = {}
    selection_rows: List[Dict[str, Any]] = []

    model_specs: List[Tuple[str, str, Any]] = [
        (
            "LOGISTIC_REGRESSION_BASELINE",
            "LOGISTIC_CLASSIFIER",
            LogisticRegression(max_iter=2000, random_state=0),
        ),
        (
            "DECISION_TREE_BASELINE",
            "TREE_CLASSIFIER",
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0),
        ),
    ]

    for bundle_name, feature_columns in FEATURE_BUNDLES.items():
        missing = [column for column in feature_columns if column not in action_df.columns]
        if missing:
            raise RuntimeError(f"{bundle_name} missing feature columns: {missing}")
        usable_feature_columns = [
            column
            for column in feature_columns
            if train_df[column].notna().any() or not pd.api.types.is_numeric_dtype(train_df[column])
        ]
        if usable_feature_columns != feature_columns:
            raise RuntimeError(
                f"{bundle_name} lost training features unexpectedly: "
                f"{sorted(set(feature_columns).difference(set(usable_feature_columns)))}"
            )
        numeric_cols, categorical_cols, preprocessor = _prepare_preprocessor(train_df, usable_feature_columns)
        X_train = train_df[usable_feature_columns].copy()
        y_train = train_df["coarse_teacher_binary_target_v1"].astype("string")

        for model_name, model_family, model in model_specs:
            estimator = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(model))])
            fitted = estimator.fit(X_train, y_train)
            full_model_name = f"{bundle_name}__{model_name}"
            fitted_models[full_model_name] = fitted
            split_metrics: Dict[str, Any] = {}
            for split_name in SPLIT_ORDER:
                split_df = action_df.loc[action_df["split_bucket_v1"].astype("string").eq(split_name)].copy()
                split_metrics[split_name] = _classification_metrics(split_df, fitted, usable_feature_columns)
                if split_name in {"VALIDATION", "HOLDOUT"}:
                    predictions_frames.append(
                        _prediction_frame(
                            split_df,
                            estimator=fitted,
                            feature_columns=usable_feature_columns,
                            model_name=full_model_name,
                            model_family=model_family,
                            feature_bundle_name=bundle_name,
                        )
                    )
            validation_metrics = split_metrics.get("VALIDATION", {})
            holdout_metrics = split_metrics.get("HOLDOUT", {})
            selection_rows.append(
                {
                    "bundle_name_v1": bundle_name,
                    "model_name_v1": full_model_name,
                    "model_family_v1": model_family,
                    "validation_brier_v1": validation_metrics.get("brier_score_v1"),
                    "validation_log_loss_v1": validation_metrics.get("log_loss_v1"),
                    "validation_roc_auc_v1": validation_metrics.get("roc_auc_v1"),
                    "holdout_brier_v1": holdout_metrics.get("brier_score_v1"),
                    "holdout_roc_auc_v1": holdout_metrics.get("roc_auc_v1"),
                }
            )
            training_feature_bundle_rows.append(
                {
                    "bundle_name_v1": bundle_name,
                    "feature_columns_v1": list(usable_feature_columns),
                    "numeric_feature_columns_v1": list(numeric_cols),
                    "categorical_feature_columns_v1": list(categorical_cols),
                    "model_name_v1": full_model_name,
                    "model_family_v1": model_family,
                    "split_metrics_v1": split_metrics,
                }
            )

    def _selection_key(row: Dict[str, Any]) -> Tuple[float, float, float, int, str]:
        validation_brier = row.get("validation_brier_v1")
        validation_log_loss = row.get("validation_log_loss_v1")
        validation_roc_auc = row.get("validation_roc_auc_v1")
        model_name = str(row.get("model_name_v1"))
        return (
            float("inf") if validation_brier is None else float(validation_brier),
            float("inf") if validation_log_loss is None else float(validation_log_loss),
            float("-inf") if validation_roc_auc is None else -float(validation_roc_auc),
            0,
            model_name,
        )

    primary_selection_row = sorted(selection_rows, key=_selection_key)[0]
    primary_model_name = str(primary_selection_row["model_name_v1"])
    primary_model_bundle = str(primary_selection_row["bundle_name_v1"])
    primary_model = fitted_models[primary_model_name]
    primary_bundle_columns = FEATURE_BUNDLES[primary_model_bundle]

    prediction_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    primary_prediction_df = prediction_df.loc[
        prediction_df["model_name_v1"].astype("string").eq(primary_model_name)
    ].copy()

    threshold_rows: List[Dict[str, Any]] = []
    holdout_primary_df = action_df.loc[action_df["split_bucket_v1"].astype("string").eq("HOLDOUT")].copy()
    if not holdout_primary_df.empty:
        holdout_scores = np.asarray(
            primary_model.predict_proba(holdout_primary_df[primary_bundle_columns].copy()),
            dtype=float,
        )[:, 1]
        holdout_out = holdout_primary_df[
            [
                "management_row_key_v1",
                "coarse_teacher_binary_target_v1",
                "realized_pnl_bps",
                "hold_longer_extra_value_bps_v1",
                "recommended_coarse_grid_value_v1",
                "recommended_coarse_grid_viable_cell_v1",
            ]
        ].copy()
        holdout_out["predicted_positive_prob_v1"] = holdout_scores
        for quantile in [0.50, 0.60, 0.70, 0.80, 0.90]:
            threshold = float(pd.Series(holdout_scores).quantile(quantile))
            selected = holdout_out.loc[holdout_out["predicted_positive_prob_v1"].ge(threshold)].copy()
            threshold_rows.append(
                {
                    "model_name_v1": primary_model_name,
                    "coverage_bucket_v1": f"TOP_{int(round((1.0 - quantile) * 100.0))}_PCT_OR_MORE",
                    "probability_threshold_v1": threshold,
                    "coverage_count_v1": int(len(selected)),
                    "coverage_rate_v1": _safe_rate(float(len(selected)), float(len(holdout_out))),
                    "positive_rate_v1": _safe_float(selected["coarse_teacher_binary_target_v1"].astype(float).mean()),
                    "mean_realized_pnl_bps_v1": _safe_float(pd.to_numeric(selected["realized_pnl_bps"], errors="coerce").mean()),
                    "mean_hold_longer_extra_value_bps_v1": _safe_float(
                        pd.to_numeric(selected["hold_longer_extra_value_bps_v1"], errors="coerce").mean()
                    ),
                    "viable_cell_share_v1": _safe_rate(
                        float(selected["recommended_coarse_grid_viable_cell_v1"].astype(bool).sum()),
                        float(len(selected)),
                    ),
                }
            )
    threshold_sweep_df = pd.DataFrame.from_records(threshold_rows)

    exit_df = view_df.loc[
        view_df["observed_action_v1"].astype("string").eq("EXIT_NOW")
        & view_df["coarse_teacher_binary_target_eligible_v1"].astype(bool)
    ].copy()
    exit_positive = int(exit_df["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum())
    exit_negative = int(exit_df["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum())
    if exit_positive > 0 and exit_negative == 0:
        exit_feedback_status_v1 = "POSITIVE_ONLY_NOT_RUN"
    elif exit_positive == 0 and exit_negative > 0:
        exit_feedback_status_v1 = "NEGATIVE_ONLY_NOT_RUN"
    elif exit_positive > 0 and exit_negative > 0:
        exit_feedback_status_v1 = "BALANCED_ELIGIBLE_BUT_HELD_OUT_FROM_P1"
    else:
        exit_feedback_status_v1 = "NO_ELIGIBLE_ROWS"

    primary_metrics_row = next(
        row for row in training_feature_bundle_rows if row["model_name_v1"] == primary_model_name
    )
    primary_validation_metrics = primary_metrics_row["split_metrics_v1"]["VALIDATION"]
    primary_holdout_metrics = primary_metrics_row["split_metrics_v1"]["HOLDOUT"]
    current_bucket_validation_brier_improvement = (
        None
        if current_bucket_baseline_metrics_v1["VALIDATION"].get("brier_score_v1") is None
        or primary_validation_metrics.get("brier_score_v1") is None
        else float(current_bucket_baseline_metrics_v1["VALIDATION"]["brier_score_v1"])
        - float(primary_validation_metrics["brier_score_v1"])
    )
    current_bucket_holdout_brier_improvement = (
        None
        if current_bucket_baseline_metrics_v1["HOLDOUT"].get("brier_score_v1") is None
        or primary_holdout_metrics.get("brier_score_v1") is None
        else float(current_bucket_baseline_metrics_v1["HOLDOUT"]["brier_score_v1"])
        - float(primary_holdout_metrics["brier_score_v1"])
    )
    beats_current_bucket_baseline = bool(
        (current_bucket_validation_brier_improvement or 0.0) > 0.0
        and (current_bucket_holdout_brier_improvement or 0.0) > 0.0
    )
    perfect_metric_flag = bool(
        (primary_validation_metrics.get("accuracy_v1") or 0.0) >= 0.99
        and (primary_holdout_metrics.get("accuracy_v1") or 0.0) >= 0.99
    )

    summary = {
        "reports_root": str(reports_root),
        "teacher_view_path": str(teacher_view_path),
        "training_action_v1": str(action_name),
        "universe_contract_v1": (
            "ACTION_SPECIFIC_ELIGIBLE_ROWS_ONLY|OBSERVED_ACTION_ONLY|NO_SYNTHETIC_COUNTERFACTUALS|"
            "WALK_FORWARD_SPLITS_REUSED"
        ),
        "universe_counts_v1": {
            "eligible_action_rows_v1": int(len(action_df)),
            "split_counts_v1": split_counts,
            "class_balance_by_split_v1": class_balance_by_split,
            "feedback_action_balance_status_v1": feedback_action_balance_status_v1,
        },
        "recommended_grid_name_v1": str(action_df["recommended_coarse_grid_name_v1"].astype("string").mode().iloc[0]),
        "feature_bundles_v1": {
            bundle_name: list(columns) for bundle_name, columns in FEATURE_BUNDLES.items()
        },
        "model_benchmarks_v1": training_feature_bundle_rows,
        "model_selection_table_v1": selection_rows,
        "primary_model_v1": primary_model_name,
        "primary_feature_bundle_v1": primary_model_bundle,
        "primary_validation_metrics_v1": primary_validation_metrics,
        "primary_holdout_metrics_v1": primary_holdout_metrics,
        "current_bucket_baseline_v1": {
            "benchmark_name_v1": "CURRENT_BUCKET_STATUS_BASELINE_V1",
            "status_rate_map_v1": {
                str(key): _safe_float(value) for key, value in bucket_status_rate_map.items()
            },
            "rank_rate_map_v1": {
                str(key): _safe_float(value) for key, value in bucket_rank_rate_map.items()
            },
            "split_metrics_v1": current_bucket_baseline_metrics_v1,
        },
        "current_bucket_validation_brier_improvement_v1": current_bucket_validation_brier_improvement,
        "current_bucket_holdout_brier_improvement_v1": current_bucket_holdout_brier_improvement,
        "beats_current_bucket_baseline_v1": beats_current_bucket_baseline,
        "perfect_metric_flag_v1": perfect_metric_flag,
        "shadow_promotion_guard_v1": (
            "RETRAIN_ALLOWED_BUT_FULL_REPLAY_CONFIRM_REQUIRED"
            if beats_current_bucket_baseline and perfect_metric_flag
            else (
                "RETRAIN_ALLOWED_SHADOW_ONLY"
                if beats_current_bucket_baseline
                else "BENCHMARK_ONLY_NO_RETRAIN"
            )
        ),
        "existing_exit_local_baseline_relevance_v1": (
            "NOT_COMPARABLE_FOR_P1_HOLD_SURFACE_EXIT_ONLY_TRAINING_UNIVERSE"
        ),
        "exit_feedback_status_v1": exit_feedback_status_v1,
        "recommended_next_step_v1": (
            "PROMOTE_TO_SHADOW_MANAGEMENT_RETRAIN_CANDIDATE"
            if beats_current_bucket_baseline
            else "KEEP_AS_BENCHMARK_ONLY"
        ),
        "contract_note_v1": (
            "This benchmark is HOLD-only for P1 because EXIT feedback remains action-observed but one-sided in the current truth surface. "
            "The benchmark does not invent EXIT negatives or synthetic counterfactuals."
        ),
    }
    return {
        "prediction_df": primary_prediction_df.sort_values(
            ["split_bucket_v1", "predicted_positive_prob_v1", "management_row_key_v1"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True),
        "threshold_sweep_df": threshold_sweep_df,
        "current_bucket_prediction_df": (
            pd.concat(current_bucket_prediction_frames, ignore_index=True)
            if current_bucket_prediction_frames
            else pd.DataFrame()
        ),
        "summary": summary,
    }


def write_management_coarse_feedback_benchmark_artifacts(
    reports_root: Path,
    *,
    teacher_view_path: Path | None = None,
    action_name: str = "HOLD",
    min_train_rows: int = 100,
    min_validation_rows: int = 20,
    min_holdout_rows: int = 20,
) -> Dict[str, str]:
    payload = build_management_coarse_feedback_benchmark_payload(
        reports_root=reports_root,
        teacher_view_path=teacher_view_path,
        action_name=action_name,
        min_train_rows=min_train_rows,
        min_validation_rows=min_validation_rows,
        min_holdout_rows=min_holdout_rows,
    )
    reports_root = Path(reports_root).expanduser().resolve()
    predictions_path = reports_root / PREDICTIONS_FILE
    threshold_sweep_path = reports_root / THRESHOLD_SWEEP_FILE
    summary_path = reports_root / SUMMARY_FILE

    payload["prediction_df"].to_parquet(predictions_path, index=False)
    payload["threshold_sweep_df"].to_csv(threshold_sweep_path, index=False)
    _write_json(summary_path, payload["summary"])

    return {
        "predictions_path": str(predictions_path.resolve()),
        "threshold_sweep_path": str(threshold_sweep_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a hard-fail coarse management feedback benchmark from the truth coarse teacher surface."
    )
    parser.add_argument("--reports-root", dest="reports_root", default=None)
    parser.add_argument("--teacher-view-path", dest="teacher_view_path", default=None)
    parser.add_argument("--action-name", dest="action_name", default="HOLD")
    parser.add_argument("--min-train-rows", dest="min_train_rows", type=int, default=100)
    parser.add_argument("--min-validation-rows", dest="min_validation_rows", type=int, default=20)
    parser.add_argument("--min-holdout-rows", dest="min_holdout_rows", type=int, default=20)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    teacher_view_path = (
        Path(args.teacher_view_path).expanduser().resolve()
        if args.teacher_view_path
        else None
    )
    written = write_management_coarse_feedback_benchmark_artifacts(
        reports_root=reports_root,
        teacher_view_path=teacher_view_path,
        action_name=str(args.action_name),
        min_train_rows=max(1, int(args.min_train_rows)),
        min_validation_rows=max(1, int(args.min_validation_rows)),
        min_holdout_rows=max(1, int(args.min_holdout_rows)),
    )
    print(json.dumps(written, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
