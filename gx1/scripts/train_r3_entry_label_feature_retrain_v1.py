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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
R2_READINESS_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1"
R2_RETRAIN_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R2"
R3_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R3_ENTRY_LABEL_FEATURE_RETRAIN_V1"

R2_READINESS_CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
R2_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
R2_HINDSIGHT_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
R2_READINESS_SUMMARY = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json"
R2_PREDICTION_VIEW = "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet"

R3_PREDICTION_VIEW = "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
R3_MODEL_METRICS = "shadow_meta_all_trade_review_r3_entry_label_feature_model_metrics_v1.csv"
R3_WALKFORWARD_METRICS = "shadow_meta_all_trade_review_r3_entry_label_feature_walkforward_metrics_v1.csv"
R3_POLICY_SAFETY = "shadow_meta_all_trade_review_r3_entry_label_feature_policy_safety_v1.csv"
R3_R2_FALLBACK_OVERLAP = "shadow_meta_all_trade_review_r3_entry_label_feature_r2_fallback_overlap_v1.csv"
R3_THRESHOLD_POLICY = "shadow_meta_all_trade_review_r3_entry_label_feature_threshold_policy_v1.json"
R3_CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r3_entry_label_feature_consistency_audit_v1.csv"
R3_CONTRACT = "shadow_meta_all_trade_review_r3_entry_label_feature_contract_v1.json"
R3_STATUS = "shadow_meta_all_trade_review_r3_entry_label_feature_status_v1.json"
R3_SUMMARY = "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json"
R3_MANIFEST = "shadow_meta_all_trade_review_r3_entry_label_feature_manifest_v1.json"
R3_MD = "shadow_meta_all_trade_review_r3_entry_label_feature_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r3_entry_label_feature_retrain_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
MISSING_CATEGORY = "__GX1_MISSING__"
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


@dataclass(frozen=True)
class EntryTaskSpec:
    task_id: str
    target_column: str
    model_dir_name: str
    policy_role: str
    noisy: bool = False


ENTRY_TASKS: tuple[EntryTaskSpec, ...] = (
    EntryTaskSpec(
        task_id="entry_r3_should_not_take",
        target_column="label_should_not_take_v1",
        model_dir_name="entry_r3_should_not_take_v1",
        policy_role="SUPPRESS_BAD_ENTRY",
    ),
    EntryTaskSpec(
        task_id="entry_r3_strong_trade_candidate",
        target_column="label_strong_trade_candidate_v1",
        model_dir_name="entry_r3_strong_trade_candidate_v1",
        policy_role="PROTECT_AND_PRIORITIZE_WINNER_PATH",
    ),
    EntryTaskSpec(
        task_id="entry_r3_immediate_mae_risk",
        target_column="label_immediate_mae_risk_v1",
        model_dir_name="entry_r3_immediate_mae_risk_v1",
        policy_role="TAIL_RISK_WARNING",
    ),
    EntryTaskSpec(
        task_id="entry_r3_good_mfe_bad_capture",
        target_column="label_good_mfe_bad_capture_v1",
        model_dir_name="entry_r3_good_mfe_bad_capture_v1",
        policy_role="RUNNER_MANAGEMENT_HANDOFF",
    ),
    EntryTaskSpec(
        task_id="entry_r3_direct_take_ok",
        target_column="label_direct_take_ok_v1",
        model_dir_name="entry_r3_direct_take_ok_v1",
        policy_role="ALLOW_CLEAN_DIRECT_ENTRY",
    ),
    EntryTaskSpec(
        task_id="entry_r3_wait_would_have_helped",
        target_column="label_wait_would_have_helped_v1",
        model_dir_name="entry_r3_wait_would_have_helped_v1",
        policy_role="WAIT_ADVISORY_ONLY",
        noisy=True,
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


def _resolve_readiness_dir(reports_root: Path, path_arg: str | None) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / R2_READINESS_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R2 readiness dir does not exist: {path}")
    for artifact in [R2_READINESS_CONTRACT, R2_AS_OF_TABLE, R2_HINDSIGHT_LABEL_TABLE, R2_READINESS_SUMMARY]:
        if not (path / artifact).exists():
            raise FileNotFoundError(f"{path} missing required artifact {artifact}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / R3_EXTENSION_NAME


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


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower() or "label"


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


def _check_feature_names_for_leakage(feature_names: Sequence[str]) -> None:
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


def _load_feature_names(contract: Dict[str, Any], asof_df: pd.DataFrame) -> List[str]:
    raw = contract.get("as_of_feature_names_v1")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("R2 readiness contract missing non-empty as_of_feature_names_v1")
    feature_names = [str(feature) for feature in raw]
    _require_columns(asof_df, feature_names, artifact_name=R2_AS_OF_TABLE)
    _check_feature_names_for_leakage(feature_names)
    return feature_names


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    normalized = series.astype("string").str.lower().str.strip()
    return normalized.eq("true").fillna(default).astype(bool)


def _num_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _split_masks(frame: pd.DataFrame, available_mask: pd.Series) -> Dict[str, pd.Series]:
    return {
        "TRAIN": _bool_series(frame, "used_for_training") & available_mask,
        "VALIDATION": _bool_series(frame, "used_for_validation") & available_mask,
        "HOLDOUT": _bool_series(frame, "used_for_holdout") & available_mask,
    }


def _fit_preprocessor(train_features: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, Any]:
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for feature in feature_names:
        series = train_features[feature]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
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
    pieces: List[np.ndarray] = []
    numeric_features = list(preprocessor["numeric_features_v1"])
    categorical_features = list(preprocessor["categorical_features_v1"])
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
        raise RuntimeError("No feature pieces available for transform.")
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "row_count_v1": int(len(y_true)),
        "accuracy_v1": None,
        "balanced_accuracy_v1": None,
        "macro_f1_v1": None,
        "precision_false_v1": None,
        "recall_false_v1": None,
        "precision_true_v1": None,
        "recall_true_v1": None,
        "confusion_matrix_json_v1": "[]",
    }
    if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
        return record
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    record.update(
        {
            "accuracy_v1": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy_v1": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1_v1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_false_v1": float(precision[0]),
            "recall_false_v1": float(recall[0]),
            "precision_true_v1": float(precision[1]),
            "recall_true_v1": float(recall[1]),
            "confusion_matrix_json_v1": _json_dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()),
        }
    )
    return record


def _train_entry_task(
    *,
    task: EntryTaskSpec,
    work_df: pd.DataFrame,
    feature_names: Sequence[str],
    available_mask: pd.Series,
    output_dir: Path,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    _require_columns(work_df, [task.target_column], artifact_name=f"{task.task_id} training frame")
    masks = _split_masks(work_df, available_mask)
    for split, mask in masks.items():
        if int(mask.sum()) == 0:
            raise ValueError(f"{task.task_id} has zero {split} rows with available features.")
        if int(_bool_series(work_df.loc[mask], task.target_column).nunique()) < 2:
            raise ValueError(f"{task.task_id} {split} requires both boolean classes.")

    y_all = _bool_series(work_df, task.target_column).astype(int)
    train_features = work_df.loc[masks["TRAIN"], feature_names]
    preprocessor = _fit_preprocessor(train_features, feature_names)
    x_train = _transform_features(preprocessor, train_features)
    x_validation = _transform_features(preprocessor, work_df.loc[masks["VALIDATION"], feature_names])
    y_train = y_all.loc[masks["TRAIN"]].to_numpy(dtype=int)
    y_validation = y_all.loc[masks["VALIDATION"]].to_numpy(dtype=int)
    weights_train = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=4.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=7.5,
        reg_alpha=0.25,
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=weights_train, eval_set=[(x_validation, y_validation)], verbose=False)

    pred = pd.DataFrame({"candidate_uid": work_df["candidate_uid"].astype("string")})
    pred[f"pred__{task.task_id}__feature_available_v1"] = available_mask.to_numpy(dtype=bool)
    pred[f"pred__{task.task_id}__label_v1"] = pd.NA
    pred[f"pred__{task.task_id}__prob_false_v1"] = pd.NA
    pred[f"pred__{task.task_id}__prob_true_v1"] = pd.NA
    metrics: List[Dict[str, Any]] = []
    for split, mask in masks.items():
        x_split = _transform_features(preprocessor, work_df.loc[mask, feature_names])
        y_true = y_all.loc[mask].to_numpy(dtype=int)
        y_prob = model.predict_proba(x_split)
        y_pred = np.asarray(model.predict(x_split), dtype=int)
        pred.loc[mask, f"pred__{task.task_id}__label_v1"] = np.where(y_pred == 1, "TRUE", "FALSE")
        pred.loc[mask, f"pred__{task.task_id}__prob_false_v1"] = y_prob[:, 0]
        pred.loc[mask, f"pred__{task.task_id}__prob_true_v1"] = y_prob[:, 1]
        metric = _classification_metrics(y_true, y_pred)
        metric.update(
            {
                "task_id_v1": task.task_id,
                "target_column_v1": task.target_column,
                "split_v1": split,
                "policy_role_v1": task.policy_role,
                "label_quality_v1": "NOISY_ADVISORY" if task.noisy else "PRIMARY_HINDSIGHT_SUPERVISION",
            }
        )
        metrics.append(metric)

    model_dir = output_dir / "models" / task.model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
    metadata = {
        "task_id_v1": task.task_id,
        "target_column_v1": task.target_column,
        "policy_role_v1": task.policy_role,
        "label_quality_v1": "NOISY_ADVISORY" if task.noisy else "PRIMARY_HINDSIGHT_SUPERVISION",
        "feature_names_v1": list(feature_names),
        "transformed_feature_count_v1": int(x_train.shape[1]),
        "sample_weight_policy_v1": "CLASS_BALANCED_ONLY_NO_REWARD_WEIGHT",
        "n_estimators_requested_v1": int(n_estimators),
        "early_stopping_rounds_v1": int(early_stopping_rounds),
        "best_iteration_v1": getattr(model, "best_iteration", None),
        "best_score_v1": _safe_float(getattr(model, "best_score", None)),
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    _write_json(model_dir / "metadata.json", metadata)
    return pred, metrics, metadata


def _prob(frame: pd.DataFrame, task_id: str) -> pd.Series:
    return pd.to_numeric(frame.get(f"pred__{task_id}__prob_true_v1", pd.Series(np.nan, index=frame.index)), errors="coerce")


def _policy_block_mask(frame: pd.DataFrame, policy: Dict[str, Any]) -> pd.Series:
    p_should_not = _prob(frame, "entry_r3_should_not_take")
    p_mae = _prob(frame, "entry_r3_immediate_mae_risk")
    p_strong = _prob(frame, "entry_r3_strong_trade_candidate")
    p_direct = _prob(frame, "entry_r3_direct_take_ok")
    feature_available = frame["entry_r3_feature_available_v1"].fillna(False).astype(bool)
    bad_signal = p_should_not.ge(float(policy["should_not_take_threshold_v1"])).fillna(False)
    mae_signal = (
        p_mae.ge(float(policy["immediate_mae_risk_threshold_v1"])).fillna(False)
        & p_direct.lt(float(policy["direct_take_protection_floor_v1"])).fillna(False)
    )
    protected = p_strong.ge(float(policy["strong_trade_protection_threshold_v1"])).fillna(False)
    return feature_available & (bad_signal | mae_signal) & ~protected


def _policy_action_frame(frame: pd.DataFrame, policy: Dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    block = _policy_block_mask(out, policy)
    p_strong = _prob(out, "entry_r3_strong_trade_candidate")
    p_bad_capture = _prob(out, "entry_r3_good_mfe_bad_capture")
    p_wait = _prob(out, "entry_r3_wait_would_have_helped")
    p_should_not = _prob(out, "entry_r3_should_not_take")
    runner = (
        out["entry_r3_feature_available_v1"].fillna(False).astype(bool)
        & ~block
        & p_strong.ge(float(policy["runner_priority_threshold_v1"])).fillna(False)
        & p_bad_capture.ge(float(policy["bad_capture_handoff_threshold_v1"])).fillna(False)
    )
    wait = (
        out["entry_r3_feature_available_v1"].fillna(False).astype(bool)
        & ~block
        & ~runner
        & p_wait.ge(float(policy["wait_advisory_threshold_v1"])).fillna(False)
        & p_should_not.lt(float(policy["wait_should_not_take_ceiling_v1"])).fillna(False)
    )
    action = pd.Series("ENTRY_ALLOW_BASELINE_SHADOW", index=out.index, dtype="string")
    action.loc[block] = "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW"
    action.loc[runner] = "ENTRY_PRIORITIZE_CLEAN_RUNNER_SHADOW"
    action.loc[wait] = "ENTRY_WAIT_FOR_CONFIRMATION_ADVISORY"
    out["entry_r3_shadow_action_v1"] = action
    out["entry_r3_shadow_action_source_v1"] = np.select(
        [block, runner, wait],
        ["R3_SHOULD_NOT_OR_MAE_RISK", "R3_STRONG_AND_BAD_CAPTURE_RUNNER", "R3_WAIT_ADVISORY_NOISY"],
        default="R3_ALLOW_BASELINE",
    )
    return out


def _calibrate_threshold_policy(validation_df: pd.DataFrame) -> Dict[str, Any]:
    defaults = {
        "should_not_take_threshold_v1": 0.65,
        "immediate_mae_risk_threshold_v1": 0.72,
        "strong_trade_protection_threshold_v1": 0.45,
        "direct_take_protection_floor_v1": 0.45,
        "runner_priority_threshold_v1": 0.55,
        "bad_capture_handoff_threshold_v1": 0.45,
        "wait_advisory_threshold_v1": 0.70,
        "wait_should_not_take_ceiling_v1": 0.45,
    }
    if validation_df.empty:
        return {**defaults, "calibration_status_v1": "DEFAULT_NO_VALIDATION_ROWS"}

    best: Dict[str, Any] | None = None
    should = _bool_series(validation_df, "label_should_not_take_v1")
    strong = _bool_series(validation_df, "label_strong_trade_candidate_v1")
    fifty = _num_series(validation_df, "peak_mfe_bps_v1").ge(50.0)
    tail_10_50 = _num_series(validation_df, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num_series(validation_df, "baseline_realized_pnl_bps_v1").le(0.0) | should
    )
    for t_should in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for t_mae in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for t_protect in [0.40, 0.45, 0.50, 0.55, 0.60]:
                candidate = {
                    **defaults,
                    "should_not_take_threshold_v1": t_should,
                    "immediate_mae_risk_threshold_v1": t_mae,
                    "strong_trade_protection_threshold_v1": t_protect,
                }
                block = _policy_block_mask(validation_df, candidate)
                should_recall = _safe_rate(float((block & should).sum()), float(should.sum())) or 0.0
                strong_block_rate = _safe_rate(float((block & strong).sum()), float(strong.sum())) or 0.0
                fifty_block_rate = _safe_rate(float((block & fifty).sum()), float(fifty.sum())) or 0.0
                tail_recall = _safe_rate(float((block & tail_10_50).sum()), float(tail_10_50.sum())) or 0.0
                if strong_block_rate > 0.18 or fifty_block_rate > 0.32:
                    continue
                score = (1.25 * should_recall) + (0.75 * tail_recall) - (2.5 * strong_block_rate) - (0.8 * fifty_block_rate)
                record = {
                    **candidate,
                    "calibration_score_v1": float(score),
                    "validation_should_not_take_recall_v1": float(should_recall),
                    "validation_strong_trade_block_rate_v1": float(strong_block_rate),
                    "validation_50_plus_mfe_block_rate_v1": float(fifty_block_rate),
                    "validation_10_50_tail_recall_v1": float(tail_recall),
                }
                if best is None or record["calibration_score_v1"] > best["calibration_score_v1"]:
                    best = record
    if best is None:
        return {**defaults, "calibration_status_v1": "DEFAULT_CONSTRAINTS_NO_VALID_COMBO"}
    best["calibration_status_v1"] = "VALIDATION_SWEEP_CONSTRAINED_PROTECT_50PLUS"
    return best


def _safety_record(scope: str, metric: str, value: Any, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"scope_v1": scope, "metric_name_v1": metric, "value_v1": value, "details_json_v1": _json_dumps(details)}


def _build_policy_safety(frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scopes = [
        ("ALL", pd.Series(True, index=frame.index)),
        ("TRAIN", _bool_series(frame, "used_for_training")),
        ("VALIDATION", _bool_series(frame, "used_for_validation")),
        ("HOLDOUT", _bool_series(frame, "used_for_holdout")),
    ]
    for scope, mask in scopes:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        block = sub["entry_r3_shadow_action_v1"].eq("ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW")
        runner = sub["entry_r3_shadow_action_v1"].eq("ENTRY_PRIORITIZE_CLEAN_RUNNER_SHADOW")
        wait = sub["entry_r3_shadow_action_v1"].eq("ENTRY_WAIT_FOR_CONFIRMATION_ADVISORY")
        should = _bool_series(sub, "label_should_not_take_v1")
        strong = _bool_series(sub, "label_strong_trade_candidate_v1")
        fifty = _num_series(sub, "peak_mfe_bps_v1").ge(50.0)
        two_hundred = _num_series(sub, "peak_mfe_bps_v1").ge(200.0)
        tail_10_50 = _num_series(sub, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
            _num_series(sub, "baseline_realized_pnl_bps_v1").le(0.0) | should
        )
        rows.extend(
            [
                _safety_record(scope, "r3_entry_block_count", int(block.sum()), {"rate": _safe_rate(float(block.sum()), float(len(sub)))}),
                _safety_record(
                    scope,
                    "r3_blocks_should_not_take_count",
                    int((block & should).sum()),
                    {"recall": _safe_rate(float((block & should).sum()), float(should.sum())), "precision": _safe_rate(float((block & should).sum()), float(block.sum()))},
                ),
                _safety_record(
                    scope,
                    "r3_blocks_strong_trade_candidate_count",
                    int((block & strong).sum()),
                    {"block_rate": _safe_rate(float((block & strong).sum()), float(strong.sum())), "strong_count": int(strong.sum())},
                ),
                _safety_record(
                    scope,
                    "r3_blocks_50_plus_mfe_count",
                    int((block & fifty).sum()),
                    {"block_rate": _safe_rate(float((block & fifty).sum()), float(fifty.sum())), "fifty_plus_count": int(fifty.sum())},
                ),
                _safety_record(
                    scope,
                    "r3_blocks_200_plus_mfe_count",
                    int((block & two_hundred).sum()),
                    {"block_rate": _safe_rate(float((block & two_hundred).sum()), float(two_hundred.sum())), "two_hundred_plus_count": int(two_hundred.sum())},
                ),
                _safety_record(
                    scope,
                    "r3_helps_10_50_mfe_tail_control_count",
                    int((block & tail_10_50).sum()),
                    {"recall": _safe_rate(float((block & tail_10_50).sum()), float(tail_10_50.sum())), "tail_count": int(tail_10_50.sum())},
                ),
                _safety_record(
                    scope,
                    "r3_prioritizes_strong_trade_count",
                    int((runner & strong).sum()),
                    {"precision": _safe_rate(float((runner & strong).sum()), float(runner.sum())), "runner_action_count": int(runner.sum())},
                ),
                _safety_record(scope, "r3_wait_advisory_count", int(wait.sum()), {"rate": _safe_rate(float(wait.sum()), float(len(sub)))}),
            ]
        )
    safety = pd.DataFrame(rows)
    all_rows = safety[safety["scope_v1"].eq("ALL")]
    summary = {
        str(row["metric_name_v1"]) + "_v1": row["value_v1"]
        for row in all_rows.to_dict(orient="records")
    }
    return safety, summary


def _build_r2_fallback_overlap(reports_root: Path, r3_prediction_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    r2_path = reports_root / R2_RETRAIN_EXTENSION_NAME / R2_PREDICTION_VIEW
    empty_summary = {
        "status_v1": "R2_PREDICTION_VIEW_NOT_FOUND",
        "r2_entry_fallback_rows_v1": 0,
        "r2_entry_fallback_match_count_v1": 0,
        "r3_suppresses_r2_fallback_rows_v1": 0,
        "r3_suppresses_r2_fallback_should_not_take_count_v1": 0,
    }
    if not r2_path.exists():
        return pd.DataFrame(), empty_summary
    r2_df = pd.read_parquet(r2_path)
    _require_columns(
        r2_df,
        [
            "candidate_uid",
            "candidate_shadow_action_v1",
            "candidate_shadow_action_source_v1",
            "candidate_shadow_action_matches_harvest_target_v1",
            "exit_harvest_policy_action_v1",
        ],
        artifact_name=R2_PREDICTION_VIEW,
    )
    cols = [
        "candidate_uid",
        "entry_r3_shadow_action_v1",
        "entry_r3_shadow_action_source_v1",
        "label_should_not_take_v1",
        "label_strong_trade_candidate_v1",
        "peak_mfe_bps_v1",
        "baseline_realized_pnl_bps_v1",
    ]
    overlap = r2_df[
        [
            "candidate_uid",
            "candidate_shadow_action_v1",
            "candidate_shadow_action_source_v1",
            "candidate_shadow_action_matches_harvest_target_v1",
            "exit_harvest_policy_action_v1",
        ]
    ].merge(r3_prediction_df[[column for column in cols if column in r3_prediction_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    overlap["r2_entry_fallback_row_v1"] = overlap["candidate_shadow_action_source_v1"].astype("string").eq("ENTRY_MODEL_SUPPRESS_FALLBACK")
    overlap["r3_suppresses_v1"] = overlap["entry_r3_shadow_action_v1"].astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW")
    overlap["r3_action_matches_r2_target_action_v1"] = np.where(
        overlap["r3_suppresses_v1"],
        overlap["exit_harvest_policy_action_v1"].astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE"),
        overlap["exit_harvest_policy_action_v1"].astype("string").eq("KEEP_BASELINE"),
    )
    fallback = overlap[overlap["r2_entry_fallback_row_v1"]].copy()
    should = _bool_series(fallback, "label_should_not_take_v1")
    strong = _bool_series(fallback, "label_strong_trade_candidate_v1")
    r3_suppress = fallback["r3_suppresses_v1"].fillna(False).astype(bool)
    summary = {
        "status_v1": "R2_FALLBACK_OVERLAP_BUILT",
        "r2_entry_fallback_rows_v1": int(len(fallback)),
        "r2_entry_fallback_match_count_v1": int(fallback["candidate_shadow_action_matches_harvest_target_v1"].fillna(False).astype(bool).sum()),
        "r2_entry_fallback_match_rate_v1": _safe_rate(
            float(fallback["candidate_shadow_action_matches_harvest_target_v1"].fillna(False).astype(bool).sum()),
            float(len(fallback)),
        ),
        "r2_entry_fallback_should_not_take_count_v1": int(should.sum()),
        "r2_entry_fallback_strong_trade_count_v1": int(strong.sum()),
        "r3_suppresses_r2_fallback_rows_v1": int(r3_suppress.sum()),
        "r3_suppresses_r2_fallback_should_not_take_count_v1": int((r3_suppress & should).sum()),
        "r3_suppresses_r2_fallback_strong_trade_count_v1": int((r3_suppress & strong).sum()),
        "r3_r2_fallback_action_counts_v1": _counts(fallback, "entry_r3_shadow_action_v1"),
    }
    return overlap, summary


def _walkforward_metrics(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    run_ids = _all_run_ids(reports_root, frame)
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        batch = frame[frame["run_id"].astype("string").isin(batch_run_ids)].copy()
        for task in ENTRY_TASKS:
            pred_col = f"pred__{task.task_id}__label_v1"
            valid = batch[pred_col].notna() & batch["entry_r3_feature_available_v1"].fillna(False).astype(bool)
            y_true = _bool_series(batch.loc[valid], task.target_column).astype(int).to_numpy(dtype=int)
            y_pred = batch.loc[valid, pred_col].astype("string").eq("TRUE").astype(int).to_numpy(dtype=int)
            metric = _classification_metrics(y_true, y_pred)
            metric.update(
                {
                    "batch_index_v1": int(batch_index),
                    "run_count_v1": int(len(batch_run_ids)),
                    "run_start_v1": batch_run_ids[0] if batch_run_ids else None,
                    "run_end_v1": batch_run_ids[-1] if batch_run_ids else None,
                    "task_id_v1": task.task_id,
                    "target_column_v1": task.target_column,
                    "policy_role_v1": task.policy_role,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# R3 Entry Label Feature Retrain V1",
        "",
        "Offline shadow/retrain candidate only. No live gate, no policy truth promotion.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R3_ENTRY_LABEL_FEATURE_RETRAIN_STATUS']}`",
        f"- Ledger rows: `{summary['ledger_trade_count_v1']}`",
        f"- Feature-covered rows: `{summary['entry_feature_coverage_v1']}`",
        f"- AS_OF feature count: `{summary['as_of_feature_count_v1']}`",
        f"- Model count: `{summary['model_count_v1']}`",
        f"- Recommended next step: `{summary['recommended_next_step_v1']}`",
        "",
        "## Guardrails",
        "",
        "- AS_OF features and HINDSIGHT labels remain physically separate inputs.",
        "- WAIT label is trained only as advisory/noisy supervision.",
        "- Threshold policy is validation-calibrated with 50+ MFE protection constraints.",
        "- R3 is not a controller and not a live gate.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    readiness_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    contract_r2 = _load_json(readiness_dir / R2_READINESS_CONTRACT)
    r2_summary = _load_json(readiness_dir / R2_READINESS_SUMMARY)
    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    labels_df = pd.read_parquet(readiness_dir / R2_HINDSIGHT_LABEL_TABLE)
    _require_columns(
        asof_df,
        [
            "candidate_uid",
            "run_id",
            "used_for_training",
            "used_for_validation",
            "used_for_holdout",
            "entry_observation_present_v1",
            "entry_raw_state_present_v1",
        ],
        artifact_name=R2_AS_OF_TABLE,
    )
    _require_columns(labels_df, ["candidate_uid", *[task.target_column for task in ENTRY_TASKS]], artifact_name=R2_HINDSIGHT_LABEL_TABLE)
    if bool(asof_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("AS_OF table requires unique candidate_uid")
    if bool(labels_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("HINDSIGHT label table requires unique candidate_uid")
    if len(asof_df) != len(labels_df):
        raise RuntimeError("AS_OF and HINDSIGHT label row counts must match")
    if expected_ledger_count is not None and int(len(asof_df)) != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger trade count expected {expected_ledger_count}, observed {len(asof_df)}")

    feature_names = _load_feature_names(contract_r2, asof_df)
    label_keep_cols = [
        "candidate_uid",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "exit_harvest_policy_action_v1",
        "trade_outcome_class",
        "exit_reason",
        *[task.target_column for task in ENTRY_TASKS],
    ]
    work = asof_df.merge(labels_df[[column for column in label_keep_cols if column in labels_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
    feature_available = _bool_series(work, "entry_observation_present_v1") & _bool_series(work, "entry_raw_state_present_v1")
    work["entry_r3_feature_available_v1"] = feature_available

    extension_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames: List[pd.DataFrame] = []
    metrics_records: List[Dict[str, Any]] = []
    model_metadata: Dict[str, Any] = {}
    for task in ENTRY_TASKS:
        pred_frame, task_metrics, metadata = _train_entry_task(
            task=task,
            work_df=work,
            feature_names=feature_names,
            available_mask=feature_available,
            output_dir=extension_dir,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed,
            n_jobs=n_jobs,
        )
        prediction_frames.append(pred_frame)
        metrics_records.extend(task_metrics)
        model_metadata[task.task_id] = metadata

    base_cols = [
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
        "management_observation_present_v1",
        "entry_r3_feature_available_v1",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "exit_harvest_policy_action_v1",
        "trade_outcome_class",
        "exit_reason",
        *[task.target_column for task in ENTRY_TASKS],
    ]
    prediction_df = work[[column for column in base_cols if column in work.columns]].copy()
    for pred_frame in prediction_frames:
        prediction_df = prediction_df.merge(pred_frame, on="candidate_uid", how="left", validate="one_to_one")

    validation_mask = _bool_series(prediction_df, "used_for_validation") & prediction_df["entry_r3_feature_available_v1"].astype(bool)
    threshold_policy = _calibrate_threshold_policy(prediction_df.loc[validation_mask].copy())
    prediction_df = _policy_action_frame(prediction_df, threshold_policy)
    metrics_df = pd.DataFrame(metrics_records)
    walkforward_df = _walkforward_metrics(reports_root, prediction_df, batch_weeks=batch_weeks)
    safety_df, safety_summary = _build_policy_safety(prediction_df)
    r2_fallback_overlap_df, r2_fallback_overlap_summary = _build_r2_fallback_overlap(reports_root, prediction_df)

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
    r3_binary_min_holdout = _safe_float(
        pd.to_numeric(metrics_df[metrics_df["split_v1"].eq("HOLDOUT")]["balanced_accuracy_v1"], errors="coerce").dropna().min()
    )
    walkforward_min_by_task = (
        walkforward_df.groupby("task_id_v1")["balanced_accuracy_v1"]
        .min()
        .replace({np.nan: None})
        .to_dict()
        if not walkforward_df.empty
        else {}
    )
    entry_feature_coverage = int(feature_available.sum())
    consistency_rows = [
        _audit_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS", {"expected": expected_ledger_count, "observed": int(len(asof_df))}),
        _audit_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION_INPUTS", "PASS", {"as_of_table": R2_AS_OF_TABLE, "hindsight_table": R2_HINDSIGHT_LABEL_TABLE}),
        _audit_record("AS_OF_FEATURE_LEAKAGE_SCAN", "PASS", {"feature_count": int(len(feature_names))}),
        _audit_record(
            "ENTRY_FEATURE_COVERAGE_WITHIN_LEDGER",
            "PASS" if 0 <= entry_feature_coverage <= int(len(asof_df)) else "FAIL",
            {"observed": entry_feature_coverage, "ledger_trade_count": int(len(asof_df))},
        ),
        _audit_record("MODEL_ARTIFACTS_WRITTEN", "PASS", {"model_count": int(len(ENTRY_TASKS))}),
        _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
    ]
    consistency_df = pd.DataFrame(consistency_rows)
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R3_ENTRY_LABEL_FEATURE_RETRAIN_STATUS_V1",
        "R3_ENTRY_LABEL_FEATURE_RETRAIN_STATUS": "TRAINED_SHADOW_RESEARCH_READY_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "TRAINING_MODE_STATUS": "OFFLINE_TIME_SPLIT_EARLY_STOPPING",
        "POLICY_MODE_STATUS": "CONSERVATIVE_SHADOW_FALLBACK_RESEARCH_ONLY",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R3_ENTRY_LABEL_FEATURE_RETRAIN_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "readiness_dir_v1": str(readiness_dir),
        "extension_dir_v1": str(extension_dir),
        "ledger_trade_count_v1": int(len(asof_df)),
        "entry_feature_coverage_v1": int(feature_available.sum()),
        "entry_feature_missing_v1": int((~feature_available).sum()),
        "as_of_feature_count_v1": int(len(feature_names)),
        "model_count_v1": int(len(ENTRY_TASKS)),
        "n_estimators_requested_v1": int(n_estimators),
        "early_stopping_rounds_v1": int(early_stopping_rounds),
        "learning_rate_v1": float(learning_rate),
        "max_depth_v1": int(max_depth),
        "threshold_policy_v1": threshold_policy,
        "validation_metrics_v1": validation_metrics,
        "holdout_metrics_v1": holdout_metrics,
        "walkforward_min_balanced_accuracy_by_task_v1": {str(k): _safe_float(v) for k, v in walkforward_min_by_task.items()},
        "r3_holdout_min_balanced_accuracy_v1": r3_binary_min_holdout,
        "r3_policy_safety_v1": safety_summary,
        "r2_fallback_overlap_v1": r2_fallback_overlap_summary,
        "r2_reference_v1": {
            "binary_entry_walkforward_min_balanced_accuracy_v1": r2_summary.get("readiness_v1", {}).get("binary_entry_walkforward_min_balanced_accuracy_v1"),
            "multiclass_entry_walkforward_min_balanced_accuracy_v1": r2_summary.get("readiness_v1", {}).get("multiclass_entry_walkforward_min_balanced_accuracy_v1"),
            "entry_blocks_50_plus_mfe_count_v1": r2_summary.get("safety_v1", {}).get("entry_blocks_50_plus_mfe_count_v1"),
            "entry_helps_10_50_mfe_tail_control_count_v1": r2_summary.get("safety_v1", {}).get("entry_helps_10_50_mfe_tail_control_count_v1"),
        },
        "recommended_next_step_v1": "SHADOW_REPLAY_R3_ENTRY_POLICY_THEN_COMPARE_TO_R2",
        "status_v1": status,
    }
    contract = {
        "layer_name": "R3_ENTRY_LABEL_FEATURE_RETRAIN_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_r2_readiness_contract_v1": R2_READINESS_CONTRACT,
        "input_as_of_table_v1": R2_AS_OF_TABLE,
        "input_hindsight_label_table_v1": R2_HINDSIGHT_LABEL_TABLE,
        "as_of_feature_names_v1": list(feature_names),
        "hindsight_target_columns_v1": [task.target_column for task in ENTRY_TASKS],
        "tasks_v1": [task.__dict__ for task in ENTRY_TASKS],
        "label_quality_policy_v1": "PRIMARY_BINARY_HEADS_PLUS_WAIT_ADVISORY_NOISY",
        "sample_weight_policy_v1": "CLASS_BALANCED_ONLY_NO_REWARD_WEIGHT",
        "threshold_policy_v1": threshold_policy,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R3_ENTRY_LABEL_FEATURE_RETRAIN_MANIFEST_V1",
        "prediction_view_v1": R3_PREDICTION_VIEW,
        "model_metrics_v1": R3_MODEL_METRICS,
        "walkforward_metrics_v1": R3_WALKFORWARD_METRICS,
        "policy_safety_v1": R3_POLICY_SAFETY,
        "r2_fallback_overlap_v1": R3_R2_FALLBACK_OVERLAP,
        "threshold_policy_v1": R3_THRESHOLD_POLICY,
        "consistency_audit_v1": R3_CONSISTENCY_AUDIT,
        "contract_v1": R3_CONTRACT,
        "status_v1": R3_STATUS,
        "summary_v1": R3_SUMMARY,
        "report_v1": R3_MD,
        "models_dir_v1": "models",
        "top_level_summary_v1": str(reports_root / TOP_LEVEL_SUMMARY),
    }
    return {
        "prediction_df_v1": prediction_df,
        "metrics_df_v1": metrics_df,
        "walkforward_df_v1": walkforward_df,
        "safety_df_v1": safety_df,
        "r2_fallback_overlap_df_v1": r2_fallback_overlap_df,
        "consistency_df_v1": consistency_df,
        "threshold_policy_v1": threshold_policy,
        "summary_v1": summary,
        "contract_v1": contract,
        "status_v1": status,
        "manifest_v1": manifest,
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    n_estimators: int = 3000,
    early_stopping_rounds: int = 100,
    learning_rate: float = 0.02,
    max_depth: int = 3,
    seed: int = 20260421,
    n_jobs: int = 4,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    readiness_dir = (readiness_dir or _resolve_readiness_dir(reports_root, None)).expanduser().resolve()
    extension_dir = (extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    payload = build_payload(
        reports_root=reports_root,
        readiness_dir=readiness_dir,
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
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload["prediction_df_v1"].to_parquet(extension_dir / R3_PREDICTION_VIEW, index=False)
    payload["metrics_df_v1"].to_csv(extension_dir / R3_MODEL_METRICS, index=False)
    payload["walkforward_df_v1"].to_csv(extension_dir / R3_WALKFORWARD_METRICS, index=False)
    payload["safety_df_v1"].to_csv(extension_dir / R3_POLICY_SAFETY, index=False)
    payload["r2_fallback_overlap_df_v1"].to_csv(extension_dir / R3_R2_FALLBACK_OVERLAP, index=False)
    payload["consistency_df_v1"].to_csv(extension_dir / R3_CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / R3_THRESHOLD_POLICY, payload["threshold_policy_v1"])
    _write_json(extension_dir / R3_CONTRACT, payload["contract_v1"])
    _write_json(extension_dir / R3_STATUS, payload["status_v1"])
    _write_json(extension_dir / R3_SUMMARY, payload["summary_v1"])
    _write_json(extension_dir / R3_MANIFEST, payload["manifest_v1"])
    (extension_dir / R3_MD).write_text(_render_markdown(payload["summary_v1"]), encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary_v1"])
    return {"summary": payload["summary_v1"], "status": payload["status_v1"], "extension_dir": str(extension_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train R3 entry label/feature shadow retrain candidate.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    readiness_dir = _resolve_readiness_dir(reports_root, args.readiness_dir)
    extension_dir = Path(args.extension_dir).expanduser().resolve() if args.extension_dir else _default_extension_dir(reports_root)
    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        extension_dir=extension_dir,
        batch_weeks=args.batch_weeks,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
