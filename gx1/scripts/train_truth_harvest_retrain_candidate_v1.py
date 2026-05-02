#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
HARVEST_RETRAIN_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R2"

ENTRY_VIEW = "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"
ENTRY_CONTRACT = "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json"
ENTRY_SKIPABILITY_RAW_STATE_VIEW = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
MANAGEMENT_VIEW = "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet"
MANAGEMENT_CONTRACT = "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json"
HARVEST_TARGET_VIEW = "shadow_meta_all_trade_review_harvest_model_adjustment_target_view_v1.parquet"
HARVEST_POLICY_VIEW = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_trade_view_v1.parquet"
HARVEST_STATUS = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_status_v1.json"

RETRAIN_PREDICTION_VIEW = "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet"
RETRAIN_REPLAY_15WEEK = "shadow_meta_all_trade_review_harvest_retrain_candidate_shadow_replay_15week_v1.csv"
RETRAIN_METRICS = "shadow_meta_all_trade_review_harvest_retrain_candidate_model_metrics_v1.csv"
RETRAIN_AUDIT = "shadow_meta_all_trade_review_harvest_retrain_candidate_consistency_audit_v1.csv"
RETRAIN_SUMMARY = "shadow_meta_all_trade_review_harvest_retrain_candidate_summary_v1.json"
RETRAIN_STATUS = "shadow_meta_all_trade_review_harvest_retrain_candidate_status_v1.json"
RETRAIN_CONTRACT = "shadow_meta_all_trade_review_harvest_retrain_candidate_contract_v1.json"
RETRAIN_MANIFEST = "shadow_meta_all_trade_review_harvest_retrain_candidate_manifest_v1.json"
RETRAIN_MD = "shadow_meta_all_trade_review_harvest_retrain_candidate_v1.md"
TOP_LEVEL_SUMMARY = "truth_harvest_retrain_candidate_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
MISSING_CATEGORY = "__GX1_MISSING__"
REWARD_CLIP_BPS_V1 = 200.0
LEAKAGE_TOKENS = (
    "hindsight",
    "realized",
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
ENTRY_RICH_RAW_PREFIXES = ("as_of_skip_replay_", "as_of_skip_candidate_", "as_of_skip_xgb_")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    source: str
    kind: str
    target_column: str
    weight_column: str | None
    model_dir_name: str


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="entry_xgb_harvest_label",
        source="entry",
        kind="classification",
        target_column="entry_xgb_harvest_label_v1",
        weight_column="entry_xgb_sample_weight_proposed_v1",
        model_dir_name="entry_xgb_harvest_label_candidate_v1",
    ),
    TaskSpec(
        task_id="entry_xgb_binary_take",
        source="entry",
        kind="classification",
        target_column="entry_xgb_binary_take_target_v1",
        weight_column="entry_xgb_sample_weight_proposed_v1",
        model_dir_name="entry_xgb_binary_take_candidate_v1",
    ),
    TaskSpec(
        task_id="exit_transformer_harvest_supervision",
        source="management",
        kind="classification",
        target_column="exit_transformer_supervision_label_v1",
        weight_column="exit_transformer_sample_weight_proposed_v1",
        model_dir_name="exit_transformer_harvest_teacher_candidate_v1",
    ),
    TaskSpec(
        task_id="management_rl_harvest_action",
        source="management",
        kind="classification",
        target_column="management_rl_harvest_action_label_v1",
        weight_column="exit_transformer_sample_weight_proposed_v1",
        model_dir_name="management_rl_harvest_action_candidate_v1",
    ),
    TaskSpec(
        task_id="management_rl_harvest_reward",
        source="management",
        kind="regression",
        target_column="management_rl_harvest_reward_bps_clipped_200_v1",
        weight_column="exit_transformer_sample_weight_proposed_v1",
        model_dir_name="management_rl_harvest_reward_candidate_v1",
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
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
        payload = _load_json(rebuild_summary)
        raw_dir = payload.get("ledger_dir") or payload.get("review_dir_v1")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / ENTRY_VIEW).exists() and (candidate / MANAGEMENT_VIEW).exists():
                return candidate

    namespace_dirs = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX)],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if all((candidate / name).exists() for name in [ENTRY_VIEW, ENTRY_CONTRACT, MANAGEMENT_VIEW, MANAGEMENT_CONTRACT]):
            return candidate
    raise FileNotFoundError("Could not resolve canonical review dir with entry and management observation views.")


def _resolve_harvest_dir(reports_root: Path, harvest_dir_arg: str | None) -> Path:
    if harvest_dir_arg:
        harvest_dir = Path(harvest_dir_arg).expanduser().resolve()
        if not harvest_dir.exists():
            raise FileNotFoundError(f"Harvest dir does not exist: {harvest_dir}")
        return harvest_dir

    summary_path = reports_root / "truth_exit_harvest_policy_candidate_v1.json"
    if summary_path.exists():
        raw_dir = _load_json(summary_path).get("extension_dir_v1")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / HARVEST_TARGET_VIEW).exists() and (candidate / HARVEST_POLICY_VIEW).exists():
                return candidate

    namespace_dirs = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX) and "HARVEST_POLICY_CANDIDATE" in path.name
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if (candidate / HARVEST_TARGET_VIEW).exists() and (candidate / HARVEST_POLICY_VIEW).exists():
            return candidate
    raise FileNotFoundError("Could not resolve harvest policy candidate dir.")


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / HARVEST_RETRAIN_EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} is missing required columns: {missing}")


def _split_name_frame(frame: pd.DataFrame) -> pd.Series:
    _require_columns(frame, ["used_for_training", "used_for_validation", "used_for_holdout"], artifact_name="split frame")
    train = frame["used_for_training"].fillna(False).astype(bool)
    validation = frame["used_for_validation"].fillna(False).astype(bool)
    holdout = frame["used_for_holdout"].fillna(False).astype(bool)
    split_count = train.astype(int) + validation.astype(int) + holdout.astype(int)
    if bool(split_count.gt(1).any()):
        bad_count = int(split_count.gt(1).sum())
        raise ValueError(f"Expected at most one split flag per row; multi-flag row count={bad_count}")
    out = pd.Series("EXCLUDED_NO_SPLIT", index=frame.index, dtype="string")
    out.loc[train] = "TRAIN"
    out.loc[validation] = "VALIDATION"
    out.loc[holdout] = "HOLDOUT"
    return out


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, policy_df: pd.DataFrame) -> List[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted(
            [path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)],
            key=_run_sort_key,
        )
        if run_ids:
            return run_ids
    return sorted(policy_df["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _load_contract_feature_names(path: Path, key: str) -> List[str]:
    payload = _load_json(path)
    raw_features = payload.get(key)
    if not isinstance(raw_features, list) or not raw_features:
        raise RuntimeError(f"{path} is missing non-empty {key}")
    return [str(feature) for feature in raw_features]


def _check_feature_names_for_leakage(feature_names: Sequence[str], *, source: str) -> List[str]:
    hits = []
    for feature_name in feature_names:
        lower = feature_name.lower()
        for token in LEAKAGE_TOKENS:
            if token == "realized" and "realized_vol" in lower:
                continue
            if token in lower:
                hits.append(feature_name)
                break
    if hits:
        raise ValueError(f"{source} observation feature list contains forbidden leakage-like fields: {hits[:20]}")
    return hits


def _feature_missing_summary(frame: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if frame.empty:
        return summary
    for column in feature_names:
        summary[column] = float(frame[column].isna().mean())
    return summary


def _normalize_class_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "TRUE", False: "FALSE"}).astype("string")
    normalized = series.astype("string").str.strip()
    normalized = normalized.mask(normalized.str.lower().isin(["", "nan", "nat", "<na>", "none"]))
    return normalized


def _numeric_target(series: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.isna().any()):
        raise ValueError(f"Regression target {column} contains null/non-numeric values.")
    return numeric.astype(float)


def _numeric_weight(series: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.isna().any()):
        raise ValueError(f"Sample weight {column} contains null/non-numeric values.")
    if bool(numeric.le(0).any()):
        raise ValueError(f"Sample weight {column} contains non-positive values.")
    return numeric.astype(float)


def _safe_float(value: Any) -> float | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    return as_float


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower() or "label"


def _prepare_feature_join(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    source_uid_column: str,
    feature_names: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    _require_columns(source_df, [source_uid_column, *feature_names], artifact_name=f"{source_name} feature source")
    if bool(source_df[source_uid_column].astype("string").duplicated().any()):
        raise ValueError(f"{source_name} feature source requires unique {source_uid_column}.")
    source = source_df[[source_uid_column, *feature_names]].copy()
    source["candidate_uid"] = source[source_uid_column].astype("string")
    source = source.drop(columns=[source_uid_column]) if source_uid_column != "candidate_uid" else source
    joined = target_df.merge(source, on="candidate_uid", how="left", validate="one_to_one")
    joined[f"{source_name}_feature_available_v1"] = joined[feature_names].notna().any(axis=1)
    return joined


def _build_entry_feature_source(
    *,
    review_dir: Path,
    entry_df: pd.DataFrame,
    base_feature_names: Sequence[str],
    entry_feature_mode: str,
) -> tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if entry_feature_mode == "core":
        return (
            entry_df[["candidate_uid", *base_feature_names]].copy(),
            list(base_feature_names),
            {
                "entry_feature_mode_v1": "core",
                "entry_raw_state_view_v1": None,
                "entry_raw_feature_count_v1": 0,
            },
        )
    if entry_feature_mode != "rich_asof_raw":
        raise ValueError(f"Unknown entry feature mode: {entry_feature_mode}")

    raw_path = review_dir / ENTRY_SKIPABILITY_RAW_STATE_VIEW
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Entry rich AS_OF raw mode requires {ENTRY_SKIPABILITY_RAW_STATE_VIEW}; refusing fallback to core features."
        )
    skip_raw_df = pd.read_parquet(raw_path)
    _require_columns(skip_raw_df, ["candidate_uid"], artifact_name=ENTRY_SKIPABILITY_RAW_STATE_VIEW)
    if bool(skip_raw_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError(f"{ENTRY_SKIPABILITY_RAW_STATE_VIEW} requires unique candidate_uid.")

    raw_feature_names = [
        column
        for column in skip_raw_df.columns
        if column not in base_feature_names and any(column.startswith(prefix) for prefix in ENTRY_RICH_RAW_PREFIXES)
    ]
    if not raw_feature_names:
        raise RuntimeError(f"{ENTRY_SKIPABILITY_RAW_STATE_VIEW} has no rich AS_OF raw entry features.")
    _check_feature_names_for_leakage(raw_feature_names, source="entry_rich_raw")

    base = entry_df[["candidate_uid", *base_feature_names]].copy()
    raw = skip_raw_df[["candidate_uid", *raw_feature_names]].copy()
    source = base.merge(raw, on="candidate_uid", how="left", validate="one_to_one")
    feature_names = list(base_feature_names) + raw_feature_names
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted({feature for feature in feature_names if feature_names.count(feature) > 1})
        raise ValueError(f"Entry rich feature source has duplicate feature names: {duplicates[:20]}")
    return (
        source,
        feature_names,
        {
            "entry_feature_mode_v1": "rich_asof_raw",
            "entry_raw_state_view_v1": ENTRY_SKIPABILITY_RAW_STATE_VIEW,
            "entry_raw_feature_count_v1": int(len(raw_feature_names)),
            "entry_raw_feature_names_v1": raw_feature_names,
            "entry_raw_feature_prefixes_v1": list(ENTRY_RICH_RAW_PREFIXES),
        },
    )


def _fit_preprocessor(train_features: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, Any]:
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for feature_name in feature_names:
        series = train_features[feature_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_features.append(feature_name)
        else:
            categorical_features.append(feature_name)

    encoder = None
    if categorical_features:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(train_features[categorical_features].astype("string").fillna(MISSING_CATEGORY))

    return {
        "feature_names_v1": list(feature_names),
        "numeric_features_v1": numeric_features,
        "categorical_features_v1": categorical_features,
        "categorical_missing_token_v1": MISSING_CATEGORY,
        "one_hot_encoder_v1": encoder,
    }


def _transform_features(preprocessor: Dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    numeric_features = list(preprocessor["numeric_features_v1"])
    categorical_features = list(preprocessor["categorical_features_v1"])
    pieces: List[np.ndarray] = []
    if numeric_features:
        numeric = frame[numeric_features].copy()
        for column in numeric_features:
            if pd.api.types.is_bool_dtype(numeric[column]):
                numeric[column] = numeric[column].astype("float64")
            else:
                numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        pieces.append(numeric.to_numpy(dtype=float))
    if categorical_features:
        encoder = preprocessor["one_hot_encoder_v1"]
        if encoder is None:
            raise RuntimeError("Categorical preprocessor is missing encoder.")
        categorical = frame[categorical_features].astype("string").fillna(MISSING_CATEGORY)
        pieces.append(encoder.transform(categorical))
    if not pieces:
        raise RuntimeError("No transformed features available.")
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces, axis=1)


def _split_masks(frame: pd.DataFrame, available_mask: pd.Series) -> Dict[str, pd.Series]:
    return {
        "TRAIN": frame["used_for_training"].fillna(False).astype(bool) & available_mask,
        "VALIDATION": frame["used_for_validation"].fillna(False).astype(bool) & available_mask,
        "HOLDOUT": frame["used_for_holdout"].fillna(False).astype(bool) & available_mask,
    }


def _classification_metrics(
    *,
    task: TaskSpec,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "task_id_v1": task.task_id,
        "target_column_v1": task.target_column,
        "kind_v1": task.kind,
        "split_v1": split,
        "row_count_v1": int(len(y_true)),
        "class_count_v1": int(len(label_names)),
        "accuracy_v1": None,
        "balanced_accuracy_v1": None,
        "macro_f1_v1": None,
        "logloss_v1": None,
        "mae_v1": None,
        "rmse_v1": None,
    }
    if len(y_true) == 0:
        return record
    record["accuracy_v1"] = float(accuracy_score(y_true, y_pred))
    record["balanced_accuracy_v1"] = float(balanced_accuracy_score(y_true, y_pred))
    record["macro_f1_v1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        record["logloss_v1"] = float(log_loss(y_true, y_prob, labels=list(range(len(label_names)))))
    except ValueError:
        record["logloss_v1"] = None
    return record


def _regression_metrics(*, task: TaskSpec, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "task_id_v1": task.task_id,
        "target_column_v1": task.target_column,
        "kind_v1": task.kind,
        "split_v1": split,
        "row_count_v1": int(len(y_true)),
        "class_count_v1": None,
        "accuracy_v1": None,
        "balanced_accuracy_v1": None,
        "macro_f1_v1": None,
        "logloss_v1": None,
        "mae_v1": None,
        "rmse_v1": None,
    }
    if len(y_true) == 0:
        return record
    record["mae_v1"] = float(mean_absolute_error(y_true, y_pred))
    record["rmse_v1"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return record


def _train_classification_task(
    *,
    task: TaskSpec,
    joined_df: pd.DataFrame,
    feature_names: Sequence[str],
    available_column: str,
    output_dir: Path,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    class_balance_only: bool,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    _require_columns(joined_df, [task.target_column], artifact_name=f"{task.task_id} training frame")
    if task.weight_column:
        _require_columns(joined_df, [task.weight_column], artifact_name=f"{task.task_id} training frame")

    available_mask = joined_df[available_column].fillna(False).astype(bool)
    masks = _split_masks(joined_df, available_mask)
    for split, mask in masks.items():
        if int(mask.sum()) == 0:
            raise ValueError(f"{task.task_id} has zero {split} rows with available features.")

    labels = _normalize_class_labels(joined_df[task.target_column])
    if bool(labels.loc[available_mask].isna().any()):
        raise ValueError(f"{task.task_id} target {task.target_column} contains missing labels in feature-available rows.")
    train_label_values = sorted(labels.loc[masks["TRAIN"]].dropna().unique().tolist())
    if len(train_label_values) < 2:
        raise ValueError(f"{task.task_id} requires at least two train classes.")
    for split in ["VALIDATION", "HOLDOUT"]:
        unseen = sorted(set(labels.loc[masks[split]].dropna().unique().tolist()) - set(train_label_values))
        if unseen:
            raise ValueError(f"{task.task_id} {split} labels are absent from train: {unseen}")

    label_to_code = {label: index for index, label in enumerate(train_label_values)}
    code_to_label = {index: label for label, index in label_to_code.items()}
    y_all = labels.map(label_to_code).astype("Int64")
    train_features = joined_df.loc[masks["TRAIN"], feature_names]
    preprocessor = _fit_preprocessor(train_features, feature_names)
    x_train = _transform_features(preprocessor, joined_df.loc[masks["TRAIN"], feature_names])
    x_validation = _transform_features(preprocessor, joined_df.loc[masks["VALIDATION"], feature_names])
    y_train = y_all.loc[masks["TRAIN"]].to_numpy(dtype=int)
    y_validation = y_all.loc[masks["VALIDATION"]].to_numpy(dtype=int)
    if class_balance_only:
        weights_train = compute_sample_weight("balanced", y_train)
        sample_weight_policy = "CLASS_BALANCED_ONLY_NO_REWARD_WEIGHT"
    elif task.weight_column:
        weights_train = _numeric_weight(joined_df.loc[masks["TRAIN"], task.weight_column], column=task.weight_column).to_numpy(dtype=float)
        sample_weight_policy = f"TARGET_WEIGHT_COLUMN:{task.weight_column}"
    else:
        weights_train = None
        sample_weight_policy = "UNWEIGHTED"

    if len(train_label_values) == 2:
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
            reg_lambda=5.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
    else:
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(train_label_values),
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=3.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
    model.fit(x_train, y_train, sample_weight=weights_train, eval_set=[(x_validation, y_validation)], verbose=False)

    prediction_df = pd.DataFrame({"candidate_uid": joined_df["candidate_uid"].astype("string")})
    prediction_df[f"pred__{task.task_id}__feature_available_v1"] = available_mask.to_numpy()
    prediction_df[f"pred__{task.task_id}__label_v1"] = pd.NA
    prediction_df[f"pred__{task.task_id}__prob_max_v1"] = pd.NA
    for label in train_label_values:
        prediction_df[f"pred__{task.task_id}__prob_{_sanitize_label(label)}_v1"] = pd.NA

    metrics_records: List[Dict[str, Any]] = []
    prediction_codes_by_split: Dict[str, np.ndarray] = {}
    probability_by_split: Dict[str, np.ndarray] = {}
    for split, mask in masks.items():
        x_split = _transform_features(preprocessor, joined_df.loc[mask, feature_names])
        y_true = y_all.loc[mask].to_numpy(dtype=int)
        y_prob = model.predict_proba(x_split)
        y_pred = np.asarray(model.predict(x_split), dtype=int)
        probability_by_split[split] = y_prob
        prediction_codes_by_split[split] = y_pred
        metrics_records.append(
            _classification_metrics(
                task=task,
                split=split,
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                label_names=train_label_values,
            )
        )

        pred_labels = [code_to_label[int(code)] for code in y_pred]
        prediction_df.loc[mask, f"pred__{task.task_id}__label_v1"] = pred_labels
        prediction_df.loc[mask, f"pred__{task.task_id}__prob_max_v1"] = y_prob.max(axis=1)
        for label, code in label_to_code.items():
            prediction_df.loc[mask, f"pred__{task.task_id}__prob_{_sanitize_label(label)}_v1"] = y_prob[:, code]

    model_dir = output_dir / "models" / task.model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
    metadata = {
        "task_id_v1": task.task_id,
        "kind_v1": task.kind,
        "target_column_v1": task.target_column,
        "source_v1": task.source,
        "feature_names_v1": list(feature_names),
        "label_to_code_v1": label_to_code,
        "code_to_label_v1": {str(key): value for key, value in code_to_label.items()},
        "n_estimators_requested_v1": int(n_estimators),
        "early_stopping_rounds_v1": int(early_stopping_rounds),
        "best_iteration_v1": getattr(model, "best_iteration", None),
        "best_score_v1": _safe_float(getattr(model, "best_score", None)),
        "transformed_feature_count_v1": int(x_train.shape[1]),
        "sample_weight_policy_v1": sample_weight_policy,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    _write_json(model_dir / "metadata.json", metadata)
    return prediction_df, metrics_records, metadata


def _train_regression_task(
    *,
    task: TaskSpec,
    joined_df: pd.DataFrame,
    feature_names: Sequence[str],
    available_column: str,
    output_dir: Path,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    _require_columns(joined_df, [task.target_column], artifact_name=f"{task.task_id} training frame")
    if task.weight_column:
        _require_columns(joined_df, [task.weight_column], artifact_name=f"{task.task_id} training frame")

    available_mask = joined_df[available_column].fillna(False).astype(bool)
    masks = _split_masks(joined_df, available_mask)
    for split, mask in masks.items():
        if int(mask.sum()) == 0:
            raise ValueError(f"{task.task_id} has zero {split} rows with available features.")

    y_all = _numeric_target(joined_df[task.target_column], column=task.target_column)
    train_features = joined_df.loc[masks["TRAIN"], feature_names]
    preprocessor = _fit_preprocessor(train_features, feature_names)
    x_train = _transform_features(preprocessor, joined_df.loc[masks["TRAIN"], feature_names])
    x_validation = _transform_features(preprocessor, joined_df.loc[masks["VALIDATION"], feature_names])
    y_train = y_all.loc[masks["TRAIN"]].to_numpy(dtype=float)
    y_validation = y_all.loc[masks["VALIDATION"]].to_numpy(dtype=float)
    weights_train = (
        _numeric_weight(joined_df.loc[masks["TRAIN"], task.weight_column], column=task.weight_column).to_numpy(dtype=float)
        if task.weight_column
        else None
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=weights_train, eval_set=[(x_validation, y_validation)], verbose=False)

    prediction_df = pd.DataFrame({"candidate_uid": joined_df["candidate_uid"].astype("string")})
    prediction_df[f"pred__{task.task_id}__feature_available_v1"] = available_mask.to_numpy()
    prediction_df[f"pred__{task.task_id}__value_bps_v1"] = pd.NA

    metrics_records: List[Dict[str, Any]] = []
    for split, mask in masks.items():
        x_split = _transform_features(preprocessor, joined_df.loc[mask, feature_names])
        y_true = y_all.loc[mask].to_numpy(dtype=float)
        y_pred = model.predict(x_split)
        prediction_df.loc[mask, f"pred__{task.task_id}__value_bps_v1"] = y_pred
        metrics_records.append(_regression_metrics(task=task, split=split, y_true=y_true, y_pred=y_pred))

    model_dir = output_dir / "models" / task.model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
    metadata = {
        "task_id_v1": task.task_id,
        "kind_v1": task.kind,
        "target_column_v1": task.target_column,
        "source_v1": task.source,
        "feature_names_v1": list(feature_names),
        "n_estimators_requested_v1": int(n_estimators),
        "early_stopping_rounds_v1": int(early_stopping_rounds),
        "best_iteration_v1": getattr(model, "best_iteration", None),
        "best_score_v1": _safe_float(getattr(model, "best_score", None)),
        "transformed_feature_count_v1": int(x_train.shape[1]),
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    _write_json(model_dir / "metadata.json", metadata)
    return prediction_df, metrics_records, metadata


def _action_from_exit_label(label: Any) -> str | None:
    text = str(label) if not pd.isna(label) else ""
    if text == "EXIT_EARLIER_DAMAGE_CONTROL":
        return "EXIT_EARLIER_DAMAGE_CONTROL"
    if text == "HOLD_LONGER_OR_RUNNER_TRAIL":
        return "HOLD_LONGER_RUNNER_TRAIL"
    if text == "NO_EXIT_TRAINING_ENTRY_FILTER":
        return "ENTRY_SUPPRESS_OR_DOWNSIZE"
    if text == "KEEP_BASELINE":
        return "KEEP_BASELINE"
    return None


def _predicted_action_delta(row: pd.Series) -> float:
    def _row_float(column: str) -> float:
        value = pd.to_numeric(row.get(column, 0.0), errors="coerce")
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(as_float):
            return 0.0
        return as_float

    action = str(row.get("candidate_shadow_action_v1", "KEEP_BASELINE"))
    if action == "ENTRY_SUPPRESS_OR_DOWNSIZE":
        return _row_float("rl_priority_entry_skip_delta_bps_v1")
    if action == "EXIT_EARLIER_DAMAGE_CONTROL":
        return _row_float("rl_priority_exit_earlier_delta_bps_v1")
    if action in {"HOLD_LONGER_HOME_RUN_RUNNER", "HOLD_LONGER_RUNNER_TRAIL", "DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL"}:
        return _row_float("rl_priority_hold_longer_delta_bps_v1")
    return 0.0


def _build_prediction_view(
    *,
    target_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    prediction_frames: Sequence[pd.DataFrame],
    entry_reject_probability_threshold: float,
) -> pd.DataFrame:
    base_columns = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "harvest_quality_bucket_v1",
        "exit_harvest_policy_action_v1",
        "rl_priority_entry_skip_delta_bps_v1",
        "rl_priority_exit_earlier_delta_bps_v1",
        "rl_priority_hold_longer_delta_bps_v1",
        "management_rl_harvest_reward_bps_raw_v1",
        "management_rl_harvest_reward_bps_clipped_200_v1",
    ]
    _require_columns(policy_df, base_columns, artifact_name="harvest policy view")
    label_columns = [
        "candidate_uid",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "entry_xgb_harvest_label_v1",
        "entry_xgb_binary_take_target_v1",
        "exit_transformer_supervision_label_v1",
        "management_rl_harvest_action_label_v1",
        "harvest_model_update_family_v1",
    ]
    _require_columns(target_df, label_columns, artifact_name="harvest target view")

    out = policy_df[base_columns].merge(target_df[label_columns], on="candidate_uid", how="left", validate="one_to_one")
    for prediction_frame in prediction_frames:
        out = out.merge(prediction_frame, on="candidate_uid", how="left", validate="one_to_one")

    entry_label = out.get("pred__entry_xgb_harvest_label__label_v1", pd.Series(pd.NA, index=out.index)).astype("string")
    entry_reject_prob = pd.to_numeric(
        out.get("pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1", pd.Series(pd.NA, index=out.index)),
        errors="coerce",
    )
    entry_take_label = out.get("pred__entry_xgb_binary_take__label_v1", pd.Series(pd.NA, index=out.index)).astype("string")
    entry_take_prob = pd.to_numeric(
        out.get("pred__entry_xgb_binary_take__prob_true_v1", pd.Series(pd.NA, index=out.index)),
        errors="coerce",
    )

    entry_suppressed = (
        entry_label.eq("REJECT_OR_LOW_SIZE").fillna(False)
        | entry_reject_prob.ge(entry_reject_probability_threshold).fillna(False)
        | entry_take_label.eq("FALSE").fillna(False)
        | entry_take_prob.lt(1.0 - entry_reject_probability_threshold).fillna(False)
    )

    management_action = out.get("pred__management_rl_harvest_action__label_v1", pd.Series(pd.NA, index=out.index)).astype("string")
    exit_action = out.get("pred__exit_transformer_harvest_supervision__label_v1", pd.Series(pd.NA, index=out.index)).map(_action_from_exit_label)

    actions: List[str] = []
    action_sources: List[str] = []
    for idx in out.index:
        mgmt = management_action.loc[idx]
        if not pd.isna(mgmt) and str(mgmt) != "<NA>":
            actions.append(str(mgmt))
            action_sources.append("MANAGEMENT_RL_ACTION_MODEL")
            continue
        if bool(entry_suppressed.loc[idx]):
            actions.append("ENTRY_SUPPRESS_OR_DOWNSIZE")
            action_sources.append("ENTRY_MODEL_SUPPRESS_FALLBACK")
            continue
        mapped_exit = exit_action.loc[idx]
        if mapped_exit:
            actions.append(mapped_exit)
            action_sources.append("EXIT_TEACHER_MODEL_FALLBACK")
            continue
        actions.append("KEEP_BASELINE")
        action_sources.append("NO_MODEL_FEATURES_KEEP_BASELINE")

    out["candidate_shadow_action_v1"] = actions
    out["candidate_shadow_action_source_v1"] = action_sources
    out["candidate_shadow_action_matches_harvest_target_v1"] = out["candidate_shadow_action_v1"].astype("string").eq(
        out["exit_harvest_policy_action_v1"].astype("string")
    )
    out["candidate_shadow_delta_bps_v1"] = out.apply(_predicted_action_delta, axis=1).astype(float)
    out["candidate_shadow_delta_clipped_200_bps_v1"] = out["candidate_shadow_delta_bps_v1"].clip(lower=0.0, upper=REWARD_CLIP_BPS_V1)
    out["candidate_shadow_pnl_bps_v1"] = (
        pd.to_numeric(out["baseline_realized_pnl_bps_v1"], errors="coerce").fillna(0.0) + out["candidate_shadow_delta_bps_v1"]
    )
    return out


def _build_shadow_replay(reports_root: Path, prediction_df: pd.DataFrame, *, batch_weeks: int) -> pd.DataFrame:
    run_ids = _all_run_ids(reports_root, prediction_df)
    rows: List[Dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        batch = prediction_df[prediction_df["run_id"].astype("string").isin(batch_run_ids)].copy()
        baseline_total = _sum_numeric(batch, "baseline_realized_pnl_bps_v1")
        predicted_delta = _sum_numeric(batch, "candidate_shadow_delta_bps_v1")
        target_delta = _sum_numeric(batch, "management_rl_harvest_reward_bps_raw_v1")
        rows.append(
            {
                "batch_index_v1": int(batch_index),
                "run_count_v1": int(len(batch_run_ids)),
                "run_start_v1": batch_run_ids[0] if batch_run_ids else None,
                "run_end_v1": batch_run_ids[-1] if batch_run_ids else None,
                "trade_count_v1": int(len(batch)),
                "zero_trade_run_count_v1": int(sum(1 for run_id in batch_run_ids if run_id not in set(batch["run_id"].astype("string")))),
                "baseline_total_pnl_bps_v1": float(baseline_total),
                "candidate_shadow_delta_bps_v1": float(predicted_delta),
                "candidate_shadow_delta_clipped_200_bps_v1": _sum_numeric(batch, "candidate_shadow_delta_clipped_200_bps_v1"),
                "candidate_shadow_total_pnl_bps_v1": float(baseline_total + predicted_delta),
                "target_harvest_upper_bound_delta_bps_v1": float(target_delta),
                "candidate_to_target_delta_capture_ratio_v1": _safe_rate(predicted_delta, target_delta),
                "action_match_rate_v1": (
                    float(batch["candidate_shadow_action_matches_harvest_target_v1"].mean()) if len(batch) else None
                ),
                "entry_suppress_count_v1": int(batch["candidate_shadow_action_v1"].eq("ENTRY_SUPPRESS_OR_DOWNSIZE").sum()),
                "exit_earlier_count_v1": int(batch["candidate_shadow_action_v1"].eq("EXIT_EARLIER_DAMAGE_CONTROL").sum()),
                "hold_longer_count_v1": int(
                    batch["candidate_shadow_action_v1"].isin(
                        ["HOLD_LONGER_HOME_RUN_RUNNER", "HOLD_LONGER_RUNNER_TRAIL", "DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL"]
                    ).sum()
                ),
                "keep_baseline_count_v1": int(batch["candidate_shadow_action_v1"].eq("KEEP_BASELINE").sum()),
            }
        )
    return pd.DataFrame(rows)


def _audit_record(name: str, status: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "check_name_v1": name,
        "status_v1": status,
        "details_json_v1": json.dumps(details or {}, ensure_ascii=True, sort_keys=True),
    }


def _render_markdown(summary: Dict[str, Any], replay_df: pd.DataFrame) -> str:
    lines = [
        "# Harvest Retrain Candidate V1",
        "",
        "Dette er en offline retrain-kandidat og shadow replay. Den er ikke en live gate, ikke policy truth og ikke auto-promotert.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['HARVEST_RETRAIN_CANDIDATE_STATUS']}`",
        f"- Models trained: `{summary['model_count_v1']}`",
        f"- Baseline PnL bps: `{summary['baseline_total_pnl_bps_v1']:.2f}`",
        f"- Candidate shadow delta bps: `{summary['candidate_shadow_delta_bps_v1']:.2f}`",
        f"- Candidate shadow total bps: `{summary['candidate_shadow_total_pnl_bps_v1']:.2f}`",
        f"- Target harvest upper bound delta bps: `{summary['target_harvest_upper_bound_delta_bps_v1']:.2f}`",
        f"- Candidate-to-target capture ratio: `{summary['candidate_to_target_delta_capture_ratio_v1']}`",
        f"- Action match rate: `{summary['candidate_shadow_action_match_rate_v1']}`",
        "",
        "## 15-Week Replay",
        "",
        "| batch | runs | trades | baseline | candidate delta | candidate total | target delta | action match |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in replay_df.to_dict(orient="records"):
        lines.append(
            "| {batch_index_v1} | {run_count_v1} | {trade_count_v1} | {baseline_total_pnl_bps_v1:.2f} | "
            "{candidate_shadow_delta_bps_v1:.2f} | {candidate_shadow_total_pnl_bps_v1:.2f} | "
            "{target_harvest_upper_bound_delta_bps_v1:.2f} | {action_match_rate_v1} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Uses only canonical AS_OF observation contract features.",
            "- Hindsight/harvest fields are targets only, never model inputs.",
            "- Missing canonical feature rows are kept as coverage gaps, not fabricated.",
            "- Promotion requires separate review and production-trainer integration.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_harvest_retrain_candidate_payload(
    *,
    reports_root: Path,
    review_dir: Path,
    harvest_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    entry_reject_probability_threshold: float,
    entry_feature_mode: str,
) -> Dict[str, Any]:
    target_df = pd.read_parquet(harvest_dir / HARVEST_TARGET_VIEW)
    policy_df = pd.read_parquet(harvest_dir / HARVEST_POLICY_VIEW)
    entry_df = pd.read_parquet(review_dir / ENTRY_VIEW)
    management_df = pd.read_parquet(review_dir / MANAGEMENT_VIEW)
    harvest_status = _load_json(harvest_dir / HARVEST_STATUS)

    required_target_columns = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "entry_xgb_harvest_label_v1",
        "entry_xgb_binary_take_target_v1",
        "entry_xgb_sample_weight_proposed_v1",
        "exit_transformer_supervision_label_v1",
        "exit_transformer_sample_weight_proposed_v1",
        "management_rl_harvest_action_label_v1",
        "management_rl_harvest_reward_bps_clipped_200_v1",
        "harvest_model_update_family_v1",
    ]
    _require_columns(target_df, required_target_columns, artifact_name=HARVEST_TARGET_VIEW)
    if harvest_status.get("EXIT_HARVEST_POLICY_CANDIDATE_STATUS") != "READY_FOR_RETRAIN_TARGET_REVIEW":
        raise RuntimeError("Harvest policy candidate must be READY_FOR_RETRAIN_TARGET_REVIEW before retraining.")
    if target_df.empty:
        raise RuntimeError("Harvest target view is empty.")
    if bool(target_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("Harvest target view requires unique candidate_uid.")
    if bool(policy_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("Harvest policy view requires unique candidate_uid.")
    _split_name_frame(target_df)

    target_candidate_set = set(target_df["candidate_uid"].astype("string"))
    policy_candidate_set = set(policy_df["candidate_uid"].astype("string"))
    if target_candidate_set != policy_candidate_set:
        raise ValueError("Harvest target and policy views must have identical candidate_uid coverage.")

    entry_features = _load_contract_feature_names(review_dir / ENTRY_CONTRACT, "observation_feature_names_v1")
    management_features = _load_contract_feature_names(review_dir / MANAGEMENT_CONTRACT, "observation_vector_feature_names_v1")
    _check_feature_names_for_leakage(entry_features, source="entry")
    _check_feature_names_for_leakage(management_features, source="management")
    entry_feature_source_df, entry_features, entry_feature_source_contract = _build_entry_feature_source(
        review_dir=review_dir,
        entry_df=entry_df,
        base_feature_names=entry_features,
        entry_feature_mode=entry_feature_mode,
    )

    entry_joined = _prepare_feature_join(
        target_df,
        entry_feature_source_df,
        source_uid_column="candidate_uid",
        feature_names=entry_features,
        source_name="entry",
    )
    management_joined = _prepare_feature_join(
        target_df,
        management_df,
        source_uid_column="candidate_uid_exact_v1",
        feature_names=management_features,
        source_name="management",
    )

    extension_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames: List[pd.DataFrame] = []
    metrics_records: List[Dict[str, Any]] = []
    model_metadata: Dict[str, Any] = {}

    for task in TASKS:
        source_joined = entry_joined if task.source == "entry" else management_joined
        feature_names = entry_features if task.source == "entry" else management_features
        available_column = "entry_feature_available_v1" if task.source == "entry" else "management_feature_available_v1"
        if task.kind == "classification":
            prediction_frame, task_metrics, metadata = _train_classification_task(
                task=task,
                joined_df=source_joined,
                feature_names=feature_names,
                available_column=available_column,
                output_dir=extension_dir,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                learning_rate=learning_rate,
                max_depth=max_depth,
                seed=seed,
                n_jobs=n_jobs,
                class_balance_only=task.source == "entry",
            )
        elif task.kind == "regression":
            prediction_frame, task_metrics, metadata = _train_regression_task(
                task=task,
                joined_df=source_joined,
                feature_names=feature_names,
                available_column=available_column,
                output_dir=extension_dir,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                learning_rate=learning_rate,
                max_depth=max_depth,
                seed=seed,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Unknown task kind: {task.kind}")
        prediction_frames.append(prediction_frame)
        metrics_records.extend(task_metrics)
        model_metadata[task.task_id] = metadata

    prediction_df = _build_prediction_view(
        target_df=target_df,
        policy_df=policy_df,
        prediction_frames=prediction_frames,
        entry_reject_probability_threshold=entry_reject_probability_threshold,
    )
    replay_df = _build_shadow_replay(reports_root, prediction_df, batch_weeks=batch_weeks)
    metrics_df = pd.DataFrame(metrics_records)

    baseline_total = _sum_numeric(prediction_df, "baseline_realized_pnl_bps_v1")
    candidate_delta = _sum_numeric(prediction_df, "candidate_shadow_delta_bps_v1")
    target_delta = _sum_numeric(prediction_df, "management_rl_harvest_reward_bps_raw_v1")
    candidate_action_match_rate = (
        float(prediction_df["candidate_shadow_action_matches_harvest_target_v1"].mean()) if len(prediction_df) else None
    )

    entry_coverage = int(entry_joined["entry_feature_available_v1"].sum())
    management_coverage = int(management_joined["management_feature_available_v1"].sum())
    warning_checks = 0
    audit_rows: List[Dict[str, Any]] = [
        _audit_record("HARVEST_STATUS_READY", "PASS", {"status": harvest_status.get("EXIT_HARVEST_POLICY_CANDIDATE_STATUS")}),
        _audit_record("TARGET_POLICY_CANDIDATE_COVERAGE_EXACT", "PASS", {"candidate_count": int(len(target_df))}),
        _audit_record("SPLIT_EXACTLY_ONE_FLAG_PER_ROW", "PASS", _counts(target_df.assign(split_bucket_v1=_split_name_frame(target_df)), "split_bucket_v1")),
        _audit_record("ENTRY_OBSERVATION_FEATURES_LEAKAGE_SCAN", "PASS", {"feature_count": int(len(entry_features))}),
        _audit_record("MANAGEMENT_OBSERVATION_FEATURES_LEAKAGE_SCAN", "PASS", {"feature_count": int(len(management_features))}),
        _audit_record("ENTRY_FEATURE_SOURCE_MODE", "PASS", entry_feature_source_contract),
        _audit_record("MODEL_ARTIFACTS_WRITTEN", "PASS", {"model_count": int(len(TASKS))}),
    ]
    if entry_coverage < len(target_df):
        warning_checks += 1
        audit_rows.append(
            _audit_record(
                "ENTRY_FEATURE_COVERAGE_PARTIAL",
                "WARN",
                {"covered": entry_coverage, "expected": int(len(target_df)), "policy": "missing rows keep explicit coverage gap"},
            )
        )
    else:
        audit_rows.append(_audit_record("ENTRY_FEATURE_COVERAGE_FULL", "PASS", {"covered": entry_coverage}))
    if management_coverage < len(target_df):
        warning_checks += 1
        audit_rows.append(
            _audit_record(
                "MANAGEMENT_FEATURE_COVERAGE_PARTIAL",
                "WARN",
                {"covered": management_coverage, "expected": int(len(target_df)), "policy": "missing rows keep explicit coverage gap"},
            )
        )
    else:
        audit_rows.append(_audit_record("MANAGEMENT_FEATURE_COVERAGE_FULL", "PASS", {"covered": management_coverage}))

    audit_df = pd.DataFrame(audit_rows)
    failed_checks = int(audit_df["status_v1"].eq("FAIL").sum())
    warning_checks += int(audit_df["status_v1"].eq("WARN").sum()) - warning_checks
    status = {
        "layer_name": "HARVEST_RETRAIN_CANDIDATE_STATUS_V1",
        "HARVEST_RETRAIN_CANDIDATE_STATUS": "TRAINED_SHADOW_REPLAY_READY_NOT_PROMOTED" if failed_checks == 0 else "ISSUES_FOUND",
        "MODEL_SCOPE_STATUS": "ENTRY_XGB_EXIT_TEACHER_MANAGEMENT_RL_HARVEST_CANDIDATES",
        "TRAINING_MODE_STATUS": "OFFLINE_TIME_SPLIT_EARLY_STOPPING",
        "REPLAY_MODE_STATUS": "SHADOW_15WEEK_NOT_LIVE_FILL",
        "PROMOTION_STATUS": "NOT_AUTO_PROMOTED",
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }

    holdout_metrics = (
        metrics_df[metrics_df["split_v1"].eq("HOLDOUT")]
        .set_index("task_id_v1")
        .replace({np.nan: None})
        .to_dict(orient="index")
    )
    validation_metrics = (
        metrics_df[metrics_df["split_v1"].eq("VALIDATION")]
        .set_index("task_id_v1")
        .replace({np.nan: None})
        .to_dict(orient="index")
    )

    summary = {
        "layer_name": "HARVEST_RETRAIN_CANDIDATE_SUMMARY_V1",
        "reports_root_v1": str(reports_root),
        "review_dir_v1": str(review_dir),
        "harvest_dir_v1": str(harvest_dir),
        "extension_dir_v1": str(extension_dir),
        "materialized_at_utc_v1": _utc_now_iso(),
        "batch_weeks_v1": int(batch_weeks),
        "candidate_count_v1": int(len(target_df)),
        "entry_feature_coverage_v1": int(entry_coverage),
        "entry_feature_mode_v1": entry_feature_mode,
        "entry_rich_raw_feature_count_v1": int(entry_feature_source_contract.get("entry_raw_feature_count_v1") or 0),
        "management_feature_coverage_v1": int(management_coverage),
        "model_count_v1": int(len(TASKS)),
        "n_estimators_requested_v1": int(n_estimators),
        "early_stopping_rounds_v1": int(early_stopping_rounds),
        "learning_rate_v1": float(learning_rate),
        "max_depth_v1": int(max_depth),
        "entry_reject_probability_threshold_v1": float(entry_reject_probability_threshold),
        "baseline_total_pnl_bps_v1": float(baseline_total),
        "candidate_shadow_delta_bps_v1": float(candidate_delta),
        "candidate_shadow_delta_clipped_200_bps_v1": _sum_numeric(prediction_df, "candidate_shadow_delta_clipped_200_bps_v1"),
        "candidate_shadow_total_pnl_bps_v1": float(baseline_total + candidate_delta),
        "target_harvest_upper_bound_delta_bps_v1": float(target_delta),
        "target_harvest_upper_bound_clipped_200_delta_bps_v1": _sum_numeric(
            prediction_df, "management_rl_harvest_reward_bps_clipped_200_v1"
        ),
        "candidate_to_target_delta_capture_ratio_v1": _safe_rate(candidate_delta, target_delta),
        "candidate_shadow_action_match_rate_v1": candidate_action_match_rate,
        "candidate_shadow_action_counts_v1": _counts(prediction_df, "candidate_shadow_action_v1"),
        "candidate_shadow_action_source_counts_v1": _counts(prediction_df, "candidate_shadow_action_source_v1"),
        "target_action_counts_v1": _counts(prediction_df, "exit_harvest_policy_action_v1"),
        "harvest_model_update_family_counts_v1": _counts(prediction_df, "harvest_model_update_family_v1"),
        "validation_metrics_v1": validation_metrics,
        "holdout_metrics_v1": holdout_metrics,
        "model_metadata_v1": model_metadata,
        "entry_feature_missing_rate_v1": _feature_missing_summary(entry_joined.loc[entry_joined["entry_feature_available_v1"]], entry_features),
        "management_feature_missing_rate_v1": _feature_missing_summary(
            management_joined.loc[management_joined["management_feature_available_v1"]], management_features
        ),
        "failed_check_count_v1": failed_checks,
        "warning_check_count_v1": int(warning_checks),
        "status_v1": status,
    }
    contract = {
        "layer_name": "HARVEST_RETRAIN_CANDIDATE_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RETRAIN_CANDIDATE_NOT_PRODUCTION",
        "input_target_view_v1": HARVEST_TARGET_VIEW,
        "input_policy_view_v1": HARVEST_POLICY_VIEW,
        "entry_observation_contract_v1": ENTRY_CONTRACT,
        "entry_feature_source_contract_v1": entry_feature_source_contract,
        "management_observation_contract_v1": MANAGEMENT_CONTRACT,
        "entry_feature_names_v1": entry_features,
        "management_feature_names_v1": management_features,
        "tasks_v1": [task.__dict__ for task in TASKS],
        "hindsight_fields_allowed_only_as_targets_v1": True,
        "numeric_missing_policy_v1": "XGBOOST_NATIVE_NAN_NO_SYNTHETIC_IMPUTATION",
        "categorical_missing_policy_v1": "EXPLICIT_MISSING_CATEGORY_FOR_ENCODER_ONLY",
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "HARVEST_RETRAIN_CANDIDATE_MANIFEST_V1",
        "summary_v1": RETRAIN_SUMMARY,
        "status_v1": RETRAIN_STATUS,
        "contract_v1": RETRAIN_CONTRACT,
        "prediction_view_v1": RETRAIN_PREDICTION_VIEW,
        "shadow_replay_15week_v1": RETRAIN_REPLAY_15WEEK,
        "metrics_v1": RETRAIN_METRICS,
        "audit_v1": RETRAIN_AUDIT,
        "models_dir_v1": "models",
        "top_level_summary_v1": str(reports_root / TOP_LEVEL_SUMMARY),
    }
    return {
        "summary_v1": summary,
        "status_v1": status,
        "contract_v1": contract,
        "manifest_v1": manifest,
        "prediction_df_v1": prediction_df,
        "replay_df_v1": replay_df,
        "metrics_df_v1": metrics_df,
        "audit_df_v1": audit_df,
    }


def materialize_truth_harvest_retrain_candidate(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    harvest_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    n_estimators: int = 3000,
    early_stopping_rounds: int = 100,
    learning_rate: float = 0.02,
    max_depth: int = 3,
    seed: int = 20260421,
    n_jobs: int = 4,
    entry_reject_probability_threshold: float = 0.5,
    entry_feature_mode: str = "rich_asof_raw",
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    review_dir = (review_dir or _resolve_review_dir(reports_root, None)).expanduser().resolve()
    harvest_dir = (harvest_dir or _resolve_harvest_dir(reports_root, None)).expanduser().resolve()
    extension_dir = (extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    payload = build_harvest_retrain_candidate_payload(
        reports_root=reports_root,
        review_dir=review_dir,
        harvest_dir=harvest_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
        entry_reject_probability_threshold=entry_reject_probability_threshold,
        entry_feature_mode=entry_feature_mode,
    )
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload["prediction_df_v1"].to_parquet(extension_dir / RETRAIN_PREDICTION_VIEW, index=False)
    payload["replay_df_v1"].to_csv(extension_dir / RETRAIN_REPLAY_15WEEK, index=False)
    payload["metrics_df_v1"].to_csv(extension_dir / RETRAIN_METRICS, index=False)
    payload["audit_df_v1"].to_csv(extension_dir / RETRAIN_AUDIT, index=False)
    _write_json(extension_dir / RETRAIN_SUMMARY, payload["summary_v1"])
    _write_json(extension_dir / RETRAIN_STATUS, payload["status_v1"])
    _write_json(extension_dir / RETRAIN_CONTRACT, payload["contract_v1"])
    _write_json(extension_dir / RETRAIN_MANIFEST, payload["manifest_v1"])
    (extension_dir / RETRAIN_MD).write_text(_render_markdown(payload["summary_v1"], payload["replay_df_v1"]), encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary_v1"])
    return {
        "summary": payload["summary_v1"],
        "status": payload["status_v1"],
        "extension_dir": str(extension_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train offline harvest retrain candidates and run 15-week shadow replay.")
    parser.add_argument("--reports-root", default=None, help="Truth root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
    parser.add_argument("--review-dir", default=None, help="Canonical review dir with AS_OF observation views.")
    parser.add_argument("--harvest-dir", default=None, help="Harvest policy candidate dir.")
    parser.add_argument("--extension-dir", default=None, help="Output extension dir.")
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--entry-reject-probability-threshold", type=float, default=0.5)
    parser.add_argument("--entry-feature-mode", choices=["rich_asof_raw", "core"], default="rich_asof_raw")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    review_dir = _resolve_review_dir(reports_root, args.review_dir)
    harvest_dir = _resolve_harvest_dir(reports_root, args.harvest_dir)
    extension_dir = Path(args.extension_dir).expanduser().resolve() if args.extension_dir else _default_extension_dir(reports_root)
    result = materialize_truth_harvest_retrain_candidate(
        reports_root,
        review_dir=review_dir,
        harvest_dir=harvest_dir,
        extension_dir=extension_dir,
        batch_weeks=args.batch_weeks,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        entry_reject_probability_threshold=args.entry_reject_probability_threshold,
        entry_feature_mode=args.entry_feature_mode,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
