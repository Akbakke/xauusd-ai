#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    TrainConfig,
    _bool,
    _fit_preprocessor,
    _jsonable,
    _num,
    _stage_params,
    _transform_features,
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import SCORE_FRAME, SUMMARY as SCORE_SUMMARY


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "TRUE_R5_2_REBUILD_RUNNER_V1"
DEFAULT_SPEC_DIR = DEFAULT_REPORTS_ROOT / "TRUE_R5_2_REBUILD_RUNNER_SPEC_V1_20260426T_LOCK"

SPEC_FILES = {
    "runner_spec": "true_r5_2_rebuild_runner_spec_v1.json",
    "input_contract": "r5_2_rebuild_input_contract_v1.json",
    "loader_spec": "r5_2_rebuild_label_and_weight_loader_spec_v1.json",
    "model_config": "r5_2_rebuild_model_config_lock_v1.json",
    "eval_guards": "r5_2_rebuild_eval_and_safety_guards_v1.json",
    "output_spec": "r5_2_rebuild_output_spec_v1.json",
    "prelaunch": "r5_2_rebuild_prelaunch_checklist_v1.json",
    "abort_rules": "r5_2_rebuild_abort_rules_v1.json",
    "r6_contract": "downstream_r6_consumption_contract_v1.json",
}

DRY_OUTPUT_FILES = {
    "summary": "summary_v1.json",
    "status": "status_v1.json",
    "manifest": "manifest_v1.json",
    "prelaunch_report": "r5_2_rebuild_prelaunch_report_v1.json",
    "feature_matrix_json": "r5_2_feature_matrix_preflight_v1.json",
    "feature_matrix_csv": "r5_2_feature_matrix_preflight_v1.csv",
    "label_weight_manifest": "r5_2_label_weight_manifest_v1.csv",
    "forbidden_feature_scan": "r5_2_forbidden_feature_scan_v1.csv",
    "r6_placeholder": "r5_2_downstream_r6_input_manifest_placeholder_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
}

TRAINING_OUTPUT_FILES = {
    "training_summary": "r5_2_rebuild_training_summary_v1.json",
    "model_manifest": "r5_2_model_manifest_v1.json",
    "config_manifest": "r5_2_config_manifest_v1.json",
    "feature_manifest": "r5_2_feature_manifest_v1.csv",
    "label_weight_manifest": "r5_2_label_weight_manifest_v1.csv",
    "prediction_view": "r5_2_prediction_view_v1.parquet",
    "score_package": "r5_2_score_package_v1.parquet",
    "base_membership": "r5_2_base_membership_v1.parquet",
    "eval_summary": "r5_2_eval_summary_v1.json",
    "pocket_eval": "r5_2_pocket_eval_report_v1.csv",
    "safety_guard": "r5_2_safety_guard_report_v1.json",
    "r6_manifest": "r5_2_downstream_r6_input_manifest_v1.json",
    "base_audit": "r5_2_rebuilt_base_membership_audit_v1.json",
    "compare_to_v3": "r5_2_rebuild_compare_to_v3_v1.json",
    "missed_390_recovery": "r5_2_rebuild_390_missed_rows_recovery_v1.json",
    "downstream_lock": "downstream_r6_input_package_lock_v1.json",
    "gate": "true_r5_2_rebuild_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
}

EXPECTED_FOUNDATION_ROWS = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_ASOF_COLUMNS = 109
EXPECTED_LABEL_ROWS = 1914
EXPECTED_MISSED_BAD_TAIL = 390
EXPECTED_BUCKET_COUNTS = {
    "STRONG_BAD_BLOCK_TARGET": 96,
    "TAIL_CONTROL_TARGET": 147,
    "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD": 130,
    "RUNNER_PROTECT_TARGET": 17,
}
SUPPORTED_BUCKETS = [
    "STRONG_BAD_BLOCK_TARGET",
    "TAIL_CONTROL_TARGET",
    "RISKY_ALLOW_TARGET",
    "RUNNER_PROTECT_TARGET",
    "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
    "IGNORE_OR_MONITOR_ONLY",
]
REQUIRED_KEYS = ["candidate_uid", "trade_uid", "decision_timestamp"]
NEW_HEAD_SPECS = {
    "bad_eligibility": ("bad_eligibility_target_v1", "pred__entry_r5_2_rebuild_bad_eligibility__prob_true_v1"),
    "tail_10_50_eligibility": ("tail_eligibility_target_v1", "pred__entry_r5_2_rebuild_tail_10_50_eligibility__prob_true_v1"),
    "risky_attention": ("risky_attention_target_v1", "pred__entry_r5_2_rebuild_risky_attention__prob_true_v1"),
    "runner_protect": ("runner_protect_target_v1", "pred__entry_r5_2_rebuild_runner_protect__prob_true_v1"),
}
BAD_SCORE_COL = NEW_HEAD_SPECS["bad_eligibility"][1]
TAIL_SCORE_COL = NEW_HEAD_SPECS["tail_10_50_eligibility"][1]
RISKY_SCORE_COL = NEW_HEAD_SPECS["risky_attention"][1]
RUNNER_SCORE_COL = NEW_HEAD_SPECS["runner_protect"][1]
BASE_FLAG_COL = "r5_2_rebuilt_base_membership_v1"
OLD_V3_METRICS = {
    "bad_v1": 82,
    "tail_v1": 51,
    "precision_v1": 1.0,
    "worst_loso_v1": 1.0,
    "repaired_damage_v1": 0,
    "fifty_plus_blocked_v1": 0,
    "hundred_plus_blocked_v1": 0,
    "two_hundred_plus_blocked_v1": 0,
    "strongest_winner_damage_v1": 0,
}
FORBIDDEN_FEATURE_PATTERNS = [
    "hindsight",
    "truth_",
    "exit_",
    "management_",
    "bridge",
    "readiness",
    "protector_first",
    "diagnostic",
    "narrow",
    "exact_only",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_spec_package(spec_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    missing = []
    for key, filename in SPEC_FILES.items():
        path = spec_dir / filename
        if not path.exists():
            missing.append(filename)
        else:
            out[key] = _read_json(path)
    if missing:
        raise FileNotFoundError(f"True R5.2 rebuild spec package missing files: {missing}")
    return out


def _ensure_output_namespace_clean(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output namespace is not clean: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _score_package_dir(spec: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    foundation = spec["input_contract"].get("foundation_v1") or {}
    return Path(foundation["score_package_dir_v1"]).expanduser().resolve()


def _label_table_path(spec: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    label_table = spec["input_contract"].get("label_table_v1") or {}
    return Path(label_table["path_v1"]).expanduser().resolve()


def _normalize_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    key_frame = frame[REQUIRED_KEYS].copy()
    for column in REQUIRED_KEYS:
        key_frame[column] = key_frame[column].astype(str)
    return key_frame


def _validate_score_package(score_dir: Path, score: pd.DataFrame, score_summary: dict[str, Any]) -> dict[str, Any]:
    active = score.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=score.index)).astype("string").eq("ACTIVE_CANDIDATE")
    checks = {
        "foundation_rows_v1": int(len(score)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "asof_columns_v1": int(score_summary.get("as_of_column_count_v1") or 0),
        "score_dir_v1": str(score_dir),
    }
    if checks["foundation_rows_v1"] != EXPECTED_FOUNDATION_ROWS:
        raise RuntimeError(f"Expected foundation rows {EXPECTED_FOUNDATION_ROWS}, observed {checks['foundation_rows_v1']}")
    if checks["active_rows_v1"] != EXPECTED_ACTIVE_ROWS or checks["quarantine_rows_v1"] != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, observed {checks['active_rows_v1']}/{checks['quarantine_rows_v1']}")
    if checks["asof_columns_v1"] != EXPECTED_ASOF_COLUMNS:
        raise RuntimeError(f"Expected AS_OF schema {EXPECTED_ASOF_COLUMNS}, observed {checks['asof_columns_v1']}")
    missing = [column for column in REQUIRED_KEYS if column not in score.columns]
    if missing:
        raise KeyError(f"Foundation score frame missing required key columns: {missing}")
    if score["candidate_uid"].duplicated().any():
        raise RuntimeError("Foundation score frame has duplicate candidate_uid values")
    return checks


def _label_weight_manifest(label_table: pd.DataFrame, loader_spec: dict[str, Any]) -> pd.DataFrame:
    weights = loader_spec.get("locked_weights_v1") or {}
    rows: list[dict[str, Any]] = []
    for bucket in SUPPORTED_BUCKETS:
        bucket_rows = label_table[label_table["new_r5_2_label_bucket_v1"] == bucket]
        if bucket == "STRONG_BAD_BLOCK_TARGET":
            sample_weight = weights.get("bad_target_weight_v1", 3.0)
            protection_weight = 0.0
        elif bucket == "TAIL_CONTROL_TARGET":
            sample_weight = weights.get("tail_target_weight_v1", 2.5)
            protection_weight = 0.0
        elif bucket == "RUNNER_PROTECT_TARGET":
            sample_weight = 0.0
            protection_weight = weights.get("runner_protect_weight_v1", 10.0)
        elif bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD":
            sample_weight = 0.0
            protection_weight = 6.0
        else:
            sample_weight = 1.25 if bucket == "RISKY_ALLOW_TARGET" else 0.25
            protection_weight = 0.0
        rows.append(
            {
                "label_bucket_v1": bucket,
                "row_count_v1": int(len(bucket_rows)),
                "bad_target_rows_v1": int(_bool(bucket_rows, "bad_eligibility_target_v1").sum()),
                "tail_target_rows_v1": int(_bool(bucket_rows, "tail_eligibility_target_v1").sum()),
                "runner_protect_rows_v1": int(_bool(bucket_rows, "runner_protect_target_v1").sum()),
                "sample_weight_v1": float(sample_weight),
                "protection_weight_v1": float(protection_weight),
            }
        )
    return pd.DataFrame(rows)


def _validate_label_table(label_table: pd.DataFrame, loader_spec: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in [*REQUIRED_KEYS, "new_r5_2_label_bucket_v1", "bad_eligibility_target_v1"] if column not in label_table.columns]
    if missing:
        raise KeyError(f"Label table missing required columns: {missing}")
    if int(len(label_table)) != EXPECTED_LABEL_ROWS:
        raise RuntimeError(f"Expected label table rows {EXPECTED_LABEL_ROWS}, observed {len(label_table)}")
    unsupported = sorted(set(label_table["new_r5_2_label_bucket_v1"].astype(str)) - set(SUPPORTED_BUCKETS))
    if unsupported:
        raise RuntimeError(f"Unsupported R5.2 label buckets: {unsupported}")
    missed = label_table[(_bool(label_table, "label_should_not_take_v1") | _bool(label_table, "tail_10_50_mfe_v1")) & ~_bool(label_table, "r5_2_v3_base_flag_v1")]
    missed_counts = {str(key): int(value) for key, value in missed["new_r5_2_label_bucket_v1"].value_counts().to_dict().items()}
    if int(len(missed)) != EXPECTED_MISSED_BAD_TAIL:
        raise RuntimeError(f"Expected {EXPECTED_MISSED_BAD_TAIL} missed bad/tail rows represented, observed {len(missed)}")
    if missed_counts != EXPECTED_BUCKET_COUNTS:
        raise RuntimeError(f"Expected missed bucket counts {EXPECTED_BUCKET_COUNTS}, observed {missed_counts}")
    ambiguous_bad_positive = int(
        (
            label_table["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
            & _bool(label_table, "bad_eligibility_target_v1")
        ).sum()
    )
    runner_bad_positive = int((label_table["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET") & _bool(label_table, "bad_eligibility_target_v1")).sum())
    if ambiguous_bad_positive:
        raise RuntimeError(f"Ambiguous high-MFE rows became bad-positive: {ambiguous_bad_positive}")
    if runner_bad_positive:
        raise RuntimeError(f"Runner-protect rows became bad-positive: {runner_bad_positive}")
    hard_protected_bad_positive = int(
        (
            (
                _bool(label_table, "hundred_plus_mfe_v1")
                | _bool(label_table, "two_hundred_plus_mfe_v1")
                | _bool(label_table, "strongest_winner_path_v1")
                | _bool(label_table, "r6_label_repaired_165_like_runner_v1")
            )
            & _bool(label_table, "bad_eligibility_target_v1")
        ).sum()
    )
    if hard_protected_bad_positive:
        raise RuntimeError(f"Hard protected winner/repaired rows became bad-positive: {hard_protected_bad_positive}")
    weights = loader_spec.get("locked_weights_v1") or {}
    expected_weights = {
        "bad_target_weight_v1": 3.0,
        "tail_target_weight_v1": 2.5,
        "runner_protect_weight_v1": 10.0,
        "hard_protection_weight_v1": 20.0,
    }
    mismatched_weights = {key: weights.get(key) for key, expected in expected_weights.items() if float(weights.get(key) or 0.0) != expected}
    if mismatched_weights:
        raise RuntimeError(f"Label/weight loader has unexpected weights: {mismatched_weights}")
    return {
        "label_table_rows_v1": int(len(label_table)),
        "missed_bad_tail_represented_v1": int(len(missed)),
        "missed_bucket_counts_v1": missed_counts,
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad_positive,
        "runner_protect_bad_positive_count_v1": runner_bad_positive,
        "hard_protected_bad_positive_count_v1": hard_protected_bad_positive,
    }


def _validate_key_alignment(score: pd.DataFrame, label_table: pd.DataFrame) -> dict[str, Any]:
    score_keys = _normalize_key_frame(score)
    label_keys = _normalize_key_frame(label_table)
    score_key_set = set(map(tuple, score_keys.to_numpy()))
    label_key_set = set(map(tuple, label_keys.to_numpy()))
    missing_from_score = label_key_set - score_key_set
    extra_in_score = score_key_set - label_key_set
    if missing_from_score or extra_in_score:
        raise RuntimeError(f"Key alignment mismatch: missing_from_score={len(missing_from_score)} extra_in_score={len(extra_in_score)}")
    return {
        "required_key_columns_v1": REQUIRED_KEYS,
        "label_to_score_missing_keys_v1": int(len(missing_from_score)),
        "score_to_label_extra_keys_v1": int(len(extra_in_score)),
        "aligned_rows_v1": int(len(label_key_set)),
    }


def _allowed_feature_names(score: pd.DataFrame, input_contract: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    families = input_contract.get("required_score_input_families_v1") or {}
    asof_features = [column for column in score.columns if column.startswith("as_of_")]
    r5_signals = [column for column in families.get("r5_signals_v1", []) if column in score.columns]
    r5_1_signals = [column for column in families.get("r5_1_signals_v1", []) if column in score.columns]
    r5_2_inputs = [column for column in families.get("allowed_current_r5_2_inputs_v1", []) if column in score.columns]
    ordered: list[str] = []
    for column in [*asof_features, *r5_signals, *r5_1_signals, *r5_2_inputs]:
        if column not in ordered:
            ordered.append(column)
    return ordered, {
        "AS_OF": asof_features,
        "R5_SIGNALS": r5_signals,
        "R5_1_SIGNALS": r5_1_signals,
        "R5_2_REBUILD_INPUTS": r5_2_inputs,
    }


def _forbidden_feature_scan(feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        lower = feature.lower()
        matches = [pattern for pattern in FORBIDDEN_FEATURE_PATTERNS if pattern in lower]
        rows.append(
            {
                "feature_v1": feature,
                "forbidden_match_v1": "|".join(matches),
                "is_forbidden_v1": bool(matches),
            }
        )
    return pd.DataFrame(rows)


def _feature_matrix_preflight(score: pd.DataFrame, input_contract: dict[str, Any]) -> tuple[list[str], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    feature_names, families = _allowed_feature_names(score, input_contract)
    if not feature_names:
        raise RuntimeError("No legal feature columns available for true R5.2 rebuild")
    forbidden = _forbidden_feature_scan(feature_names)
    forbidden_count = int(forbidden["is_forbidden_v1"].sum()) if not forbidden.empty else 0
    if forbidden_count:
        bad = forbidden[forbidden["is_forbidden_v1"]]["feature_v1"].tolist()
        raise RuntimeError(f"Forbidden features present in R5.2 rebuild feature matrix: {bad}")
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        family = next((name for name, cols in families.items() if feature in cols), "UNKNOWN")
        rows.append(
            {
                "feature_v1": feature,
                "feature_family_v1": family,
                "null_count_v1": int(score[feature].isna().sum()),
                "null_rate_v1": float(score[feature].isna().mean()),
                "coverage_rate_v1": float(1.0 - score[feature].isna().mean()),
                "dtype_v1": str(score[feature].dtype),
            }
        )
    preflight = pd.DataFrame(rows)
    summary = {
        "feature_count_v1": int(len(feature_names)),
        "feature_families_v1": {family: int(len(cols)) for family, cols in families.items()},
        "forbidden_feature_count_v1": forbidden_count,
        "max_null_rate_v1": float(preflight["null_rate_v1"].max()) if not preflight.empty else 0.0,
        "key_alignment_summary_v1": "validated separately before feature matrix build",
    }
    return feature_names, preflight, summary, forbidden


def _merge_training_frame(score: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        "candidate_uid",
        *[col for col in label_table.columns if col not in score.columns or col in {"trade_uid", "decision_timestamp"}],
    ]
    label_cols = list(dict.fromkeys(label_cols))
    merged = score.merge(label_table[label_cols], on="candidate_uid", how="left", validate="one_to_one", suffixes=("", "__label"))
    return merged


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    valid = np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {"row_count_v1": 0}
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "row_count_v1": int(len(y_true)),
        "positive_count_v1": int(y_true.sum()),
        "pred_positive_count_v1": int(y_pred.sum()),
        "confusion_matrix_json_v1": json.dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()),
        "balanced_accuracy_v1": None,
        "precision_true_v1": None,
        "recall_true_v1": None,
        "roc_auc_v1": None,
    }
    if len(set(y_true.tolist())) >= 2:
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        out.update(
            {
                "balanced_accuracy_v1": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_true_v1": float(precision[1]),
                "recall_true_v1": float(recall[1]),
                "roc_auc_v1": float(roc_auc_score(y_true, y_prob)),
            }
        )
    return out


def _head_sample_weights(frame: pd.DataFrame, label_col: str) -> np.ndarray:
    y = _bool(frame, label_col).astype(int).to_numpy(dtype=int)
    weights = compute_sample_weight("balanced", y).astype(float) if len(set(y.tolist())) >= 2 else np.ones(len(frame), dtype=float)
    sample_weight = _num(frame, "sample_weight_v1", 1.0).fillna(1.0).to_numpy(dtype=float)
    protection_weight = _num(frame, "protection_weight_v1", 0.0).fillna(0.0).to_numpy(dtype=float)
    if label_col in {"bad_eligibility_target_v1", "tail_eligibility_target_v1", "risky_attention_target_v1"}:
        weights[y == 1] *= sample_weight[y == 1]
        weights[y == 0] *= np.maximum(1.0, protection_weight[y == 0])
    else:
        weights[y == 1] *= np.maximum(sample_weight[y == 1], protection_weight[y == 1])
    return weights


def _train_weighted_head(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_col: str,
    output_col: str,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path,
    model_tag: str,
    seed: int,
    config: TrainConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    y_all = _bool(frame, label_col).astype(int)
    train_mask = train_mask.reindex(frame.index).fillna(False).astype(bool)
    validation_mask = validation_mask.reindex(frame.index).fillna(False).astype(bool)
    if int(train_mask.sum()) < 20 or len(set(y_all.loc[train_mask].tolist())) < 2:
        constant = float(y_all.loc[train_mask].mean()) if int(train_mask.sum()) else float(y_all.mean())
        probs = pd.Series(constant, index=frame.index, dtype="float64")
        metrics = _classification_metrics(y_all.to_numpy(dtype=int), probs.to_numpy(dtype=float))
        metrics.update({"model_tag_v1": model_tag, "label_col_v1": label_col, "output_col_v1": output_col, "split_v1": "ALL", "constant_model_v1": True})
        return probs.rename(output_col), pd.DataFrame([metrics])
    if int(validation_mask.sum()) == 0 or len(set(y_all.loc[validation_mask].tolist())) < 2:
        validation_mask = train_mask
    preprocessor = _fit_preprocessor(frame.loc[train_mask, feature_names], feature_names)
    x_train = _transform_features(preprocessor, frame.loc[train_mask, feature_names])
    x_val = _transform_features(preprocessor, frame.loc[validation_mask, feature_names])
    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_val = y_all.loc[validation_mask].to_numpy(dtype=int)
    weights = _head_sample_weights(frame.loc[train_mask].copy(), label_col)
    stage_params = _stage_params(config, "r5_2")
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(stage_params["n_estimators"]),
        early_stopping_rounds=int(stage_params["early_stopping_rounds"]),
        learning_rate=float(stage_params["learning_rate"]),
        max_depth=config.max_depth,
        min_child_weight=float(stage_params["min_child_weight"]),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=float(stage_params["reg_lambda"]),
        reg_alpha=float(stage_params["reg_alpha"]),
        tree_method="hist",
        random_state=seed,
        n_jobs=config.n_jobs,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
    probs = pd.Series(model.predict_proba(_transform_features(preprocessor, frame[feature_names]))[:, 1], index=frame.index, dtype="float64")
    rows: list[dict[str, Any]] = []
    for split_name, mask in {
        "TRAIN": train_mask,
        "VALIDATION": validation_mask,
        "HOLDOUT_OR_OTHER": ~(train_mask | validation_mask),
        "ALL": pd.Series(True, index=frame.index),
    }.items():
        if int(mask.sum()) == 0:
            continue
        metrics = _classification_metrics(y_all.loc[mask].to_numpy(dtype=int), probs.loc[mask].to_numpy(dtype=float))
        metrics.update({"model_tag_v1": model_tag, "label_col_v1": label_col, "output_col_v1": output_col, "split_v1": split_name, "constant_model_v1": False})
        rows.append(metrics)
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
            "not_live_gate_v1": True,
        },
    )
    return probs.rename(output_col), pd.DataFrame(rows)


def _execute_training_scaffold(
    *,
    output_dir: Path,
    training_frame: pd.DataFrame,
    feature_names: Sequence[str],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    config = TrainConfig(seed=int(model_config.get("seed_v1") or 20260426))
    train_mask = _bool(training_frame, "used_for_training")
    validation_mask = _bool(training_frame, "used_for_validation")
    pred = training_frame[["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id"]].copy()
    metrics_frames: list[pd.DataFrame] = []
    for idx, (head_name, (label_col, output_col)) in enumerate(NEW_HEAD_SPECS.items()):
        probs, metrics = _train_weighted_head(
            frame=training_frame,
            feature_names=feature_names,
            label_col=label_col,
            output_col=output_col,
            train_mask=train_mask,
            validation_mask=validation_mask,
            output_dir=output_dir,
            model_tag="true_r5_2_rebuild_v1",
            seed=config.seed + idx * 17,
            config=config,
        )
        pred[output_col] = probs.to_numpy(dtype=float)
        metrics["head_name_v1"] = head_name
        metrics_frames.append(metrics)
    bad = _num(pred, BAD_SCORE_COL)
    tail = _num(pred, TAIL_SCORE_COL)
    risky = _num(pred, RISKY_SCORE_COL)
    protect = _num(pred, RUNNER_SCORE_COL)
    pred[BASE_FLAG_COL] = ((bad >= 0.50) | (tail >= 0.50) | (risky >= 0.65)) & (protect < 0.50)
    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    pred.to_parquet(output_dir / TRAINING_OUTPUT_FILES["prediction_view"], index=False)
    pred.to_parquet(output_dir / TRAINING_OUTPUT_FILES["score_package"], index=False)
    pred[["candidate_uid", BASE_FLAG_COL]].to_parquet(output_dir / TRAINING_OUTPUT_FILES["base_membership"], index=False)
    metrics.to_csv(output_dir / "r5_2_model_metrics_v1.csv", index=False)
    return {
        "training_started_v1": True,
        "head_count_v1": int(len(NEW_HEAD_SPECS)),
        "feature_count_v1": int(len(feature_names)),
        "prediction_rows_v1": int(len(pred)),
        "base_membership_rows_v1": int(pred[BASE_FLAG_COL].sum()),
        "config_v1": asdict(config),
    }


def _join_predictions(training_frame: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ["candidate_uid", BAD_SCORE_COL, TAIL_SCORE_COL, RISKY_SCORE_COL, RUNNER_SCORE_COL, BASE_FLAG_COL]
    missing = [column for column in pred_cols if column not in prediction.columns]
    if missing:
        raise RuntimeError(f"Prediction view missing required rebuilt R5.2 columns: {missing}")
    return training_frame.merge(prediction[pred_cols], on="candidate_uid", how="left", validate="one_to_one")


def _safe_div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def _selected_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    fifty = _bool(frame, "fifty_plus_mfe_v1")
    hundred = _bool(frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(frame, "strongest_winner_path_v1")
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1")
    runner_near = _bool(frame, "r6_label_runner_near_miss_v1")
    block = int(selected.sum())
    bad = int((selected & should).sum())
    return {
        "base_rows_v1": block,
        "bad_rows_in_base_v1": bad,
        "tail_rows_in_base_v1": int((selected & tail).sum()),
        "precision_v1": _safe_div(bad, block),
        "false_take_ok_rows_v1": int((selected & take_ok).sum()),
        "fifty_plus_overlap_v1": int((selected & fifty).sum()),
        "hundred_plus_overlap_v1": int((selected & hundred).sum()),
        "two_hundred_plus_overlap_v1": int((selected & two_hundred).sum()),
        "strongest_winner_overlap_v1": int((selected & strongest).sum()),
        "repaired_like_overlap_v1": int((selected & repaired).sum()),
        "runner_near_miss_overlap_v1": int((selected & runner_near).sum()),
    }


def _worst_loso_precision(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    if "run_id" not in frame.columns:
        return None
    values: list[float] = []
    for _, group in frame.groupby(frame["run_id"].astype("string"), dropna=False):
        group_selected = selected.loc[group.index]
        if int(group_selected.sum()) == 0:
            continue
        should = _bool(group, "label_should_not_take_v1")
        values.append(float((group_selected & should).sum() / group_selected.sum()))
    return min(values) if values else None


def _score_recall(frame: pd.DataFrame, target_col: str, score_col: str, threshold: float = 0.5) -> dict[str, Any]:
    target = _bool(frame, target_col)
    selected = _num(frame, score_col).ge(threshold).fillna(False)
    denom = int(target.sum())
    hits = int((target & selected).sum())
    return {
        "target_rows_v1": denom,
        "score_selected_rows_v1": int(selected.sum()),
        "target_hits_v1": hits,
        "recall_v1": _safe_div(hits, denom),
        "threshold_v1": threshold,
    }


def _base_membership_audit(scored: pd.DataFrame) -> dict[str, Any]:
    base = _bool(scored, BASE_FLAG_COL)
    metrics = _selected_metrics(scored, base)
    metrics.update(
        {
            "layer_name": "R5_2_REBUILT_BASE_MEMBERSHIP_AUDIT_V1",
            "worst_loso_v1": _worst_loso_precision(scored, base),
            "strong_bad_rows_recovered_v1": int((base & scored["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET")).sum()),
            "tail_control_rows_recovered_v1": int((base & scored["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET")).sum()),
            "ambiguous_high_mfe_rows_included_v1": int((base & scored["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")).sum()),
            "ambiguous_high_mfe_rows_excluded_v1": int((~base & scored["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")).sum()),
            "runner_protect_rows_included_v1": int((base & scored["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")).sum()),
            "runner_protect_rows_excluded_v1": int((~base & scored["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")).sum()),
        }
    )
    return metrics


def _pocket_eval_report(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = _bool(scored, BASE_FLAG_COL)
    for bucket, group in scored.groupby(scored["new_r5_2_label_bucket_v1"].astype(str), dropna=False):
        group_base = base.loc[group.index]
        rows.append(
            {
                "pocket_v1": str(bucket),
                "row_count_v1": int(len(group)),
                "base_rows_v1": int(group_base.sum()),
                "bad_score_p50_v1": float(_num(group, BAD_SCORE_COL).median()),
                "bad_score_p90_v1": float(_num(group, BAD_SCORE_COL).quantile(0.90)),
                "tail_score_p50_v1": float(_num(group, TAIL_SCORE_COL).median()),
                "tail_score_p90_v1": float(_num(group, TAIL_SCORE_COL).quantile(0.90)),
                "runner_protect_score_p50_v1": float(_num(group, RUNNER_SCORE_COL).median()),
                "runner_protect_score_p90_v1": float(_num(group, RUNNER_SCORE_COL).quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _missed_390_recovery(scored: pd.DataFrame) -> dict[str, Any]:
    missed = scored[(_bool(scored, "label_should_not_take_v1") | _bool(scored, "tail_10_50_mfe_v1")) & ~_bool(scored, "r5_2_v3_base_flag_v1")].copy()
    base = _bool(missed, BASE_FLAG_COL)
    strong = missed["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET")
    tail = missed["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET")
    ambiguous = missed["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
    runner = missed["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")
    low_bad = _num(missed, BAD_SCORE_COL).lt(0.50).fillna(True)
    low_tail = _num(missed, TAIL_SCORE_COL).lt(0.50).fillna(True)
    high_protect = _num(missed, RUNNER_SCORE_COL).ge(0.50).fillna(False)
    return {
        "layer_name": "R5_2_REBUILD_390_MISSED_ROWS_RECOVERY_V1",
        "missed_390_rows_v1": int(len(missed)),
        "strong_bad_targets_v1": int(strong.sum()),
        "strong_bad_high_bad_eligibility_v1": int((strong & _num(missed, BAD_SCORE_COL).ge(0.50).fillna(False)).sum()),
        "tail_control_targets_v1": int(tail.sum()),
        "tail_control_high_tail_eligibility_v1": int((tail & _num(missed, TAIL_SCORE_COL).ge(0.50).fillna(False)).sum()),
        "ambiguous_high_mfe_targets_v1": int(ambiguous.sum()),
        "ambiguous_high_mfe_held_out_v1": int((ambiguous & ~base).sum()),
        "runner_protect_targets_v1": int(runner.sum()),
        "runner_protect_held_out_v1": int((runner & ~base).sum()),
        "now_r5_2_base_eligible_v1": int(base.sum()),
        "still_not_recoverable_v1": int((~base).sum()),
        "first_recovery_blocker_counts_v1": {
            "eligibility_scores_too_low_v1": int((~base & low_bad & low_tail & ~high_protect).sum()),
            "runner_protection_score_high_v1": int((~base & high_protect).sum()),
            "ambiguous_or_runner_protected_v1": int((~base & (ambiguous | runner)).sum()),
        },
    }


def _label_objective_eval(scored: pd.DataFrame, label_weight_manifest: pd.DataFrame) -> dict[str, Any]:
    ambiguous = scored["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
    high_mfe = _bool(scored, "fifty_plus_mfe_v1") | _bool(scored, "hundred_plus_mfe_v1") | _bool(scored, "two_hundred_plus_mfe_v1")
    base = _bool(scored, BASE_FLAG_COL)
    return {
        "layer_name": "R5_2_REBUILD_LABEL_OBJECTIVE_EVAL_V1",
        "strong_bad_target_recall_v1": _score_recall(scored, "bad_eligibility_target_v1", BAD_SCORE_COL),
        "tail_10_50_target_recall_v1": _score_recall(scored, "tail_eligibility_target_v1", TAIL_SCORE_COL),
        "risky_attention_coverage_v1": _score_recall(scored, "risky_attention_target_v1", RISKY_SCORE_COL, threshold=0.65),
        "runner_protect_performance_v1": _score_recall(scored, "runner_protect_target_v1", RUNNER_SCORE_COL),
        "ambiguous_high_mfe_handling_v1": {
            "rows_v1": int(ambiguous.sum()),
            "high_bad_score_rows_v1": int((ambiguous & _num(scored, BAD_SCORE_COL).ge(0.50).fillna(False)).sum()),
            "base_included_rows_v1": int((ambiguous & base).sum()),
            "bad_positive_rows_v1": int((ambiguous & _bool(scored, "bad_eligibility_target_v1")).sum()),
        },
        "high_mfe_false_positive_risk_v1": int((base & high_mfe).sum()),
        "protected_winner_treatment_v1": {
            "runner_protect_in_base_v1": int((base & scored["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")).sum()),
            "hundred_plus_in_base_v1": int((base & _bool(scored, "hundred_plus_mfe_v1")).sum()),
            "two_hundred_plus_in_base_v1": int((base & _bool(scored, "two_hundred_plus_mfe_v1")).sum()),
            "strongest_winner_in_base_v1": int((base & _bool(scored, "strongest_winner_path_v1")).sum()),
            "repaired_like_in_base_v1": int((base & _bool(scored, "r6_label_repaired_165_like_runner_v1")).sum()),
        },
        "class_balance_v1": {str(key): int(value) for key, value in scored["new_r5_2_label_bucket_v1"].value_counts().to_dict().items()},
        "sample_weight_behavior_v1": label_weight_manifest.to_dict(orient="records"),
    }


def _compare_to_v3(base_audit: dict[str, Any]) -> dict[str, Any]:
    precision = base_audit["precision_v1"]
    worst_loso = base_audit["worst_loso_v1"]
    return {
        "layer_name": "R5_2_REBUILD_COMPARE_TO_V3_V1",
        "v3_baseline_v1": OLD_V3_METRICS,
        "rebuilt_v1": {
            "bad_v1": base_audit["bad_rows_in_base_v1"],
            "tail_v1": base_audit["tail_rows_in_base_v1"],
            "precision_v1": precision,
            "worst_loso_v1": worst_loso,
            "repaired_damage_v1": base_audit["repaired_like_overlap_v1"],
            "fifty_plus_blocked_v1": base_audit["fifty_plus_overlap_v1"],
            "hundred_plus_blocked_v1": base_audit["hundred_plus_overlap_v1"],
            "two_hundred_plus_blocked_v1": base_audit["two_hundred_plus_overlap_v1"],
            "strongest_winner_damage_v1": base_audit["strongest_winner_overlap_v1"],
        },
        "delta_v1": {
            "bad_delta_v1": int(base_audit["bad_rows_in_base_v1"] - OLD_V3_METRICS["bad_v1"]),
            "tail_delta_v1": int(base_audit["tail_rows_in_base_v1"] - OLD_V3_METRICS["tail_v1"]),
            "precision_delta_v1": None if precision is None else float(precision - OLD_V3_METRICS["precision_v1"]),
            "loso_delta_v1": None if worst_loso is None else float(worst_loso - OLD_V3_METRICS["worst_loso_v1"]),
            "safety_delta_v1": int(
                base_audit["repaired_like_overlap_v1"]
                + base_audit["hundred_plus_overlap_v1"]
                + base_audit["two_hundred_plus_overlap_v1"]
                + base_audit["strongest_winner_overlap_v1"]
                + max(0, base_audit["fifty_plus_overlap_v1"] - OLD_V3_METRICS["fifty_plus_blocked_v1"])
            ),
        },
        "material_uplift_beyond_tiny_v3_v1": bool(base_audit["bad_rows_in_base_v1"] > OLD_V3_METRICS["bad_v1"] or base_audit["tail_rows_in_base_v1"] > OLD_V3_METRICS["tail_v1"]),
    }


def _safety_guard(scored: pd.DataFrame, base_audit: dict[str, Any], forbidden_count: int, key_alignment: dict[str, Any]) -> dict[str, Any]:
    ambiguous_bad_positive = int((scored["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD") & _bool(scored, "bad_eligibility_target_v1")).sum())
    runner_bad_positive = int((scored["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET") & _bool(scored, "bad_eligibility_target_v1")).sum())
    checks = {
        "ambiguous_high_mfe_bad_positive_zero_v1": ambiguous_bad_positive == 0,
        "runner_protect_bad_positive_zero_v1": runner_bad_positive == 0,
        "repaired_like_protection_pass_v1": int(base_audit["repaired_like_overlap_v1"]) == 0,
        "strongest_winner_protection_pass_v1": int(base_audit["strongest_winner_overlap_v1"]) == 0,
        "hundred_plus_protection_pass_v1": int(base_audit["hundred_plus_overlap_v1"]) == 0,
        "two_hundred_plus_protection_pass_v1": int(base_audit["two_hundred_plus_overlap_v1"]) == 0,
        "fifty_plus_risk_not_exploded_v1": int(base_audit["fifty_plus_overlap_v1"]) <= 1,
        "forbidden_feature_count_zero_v1": int(forbidden_count) == 0,
        "key_schema_no_drift_v1": key_alignment["label_to_score_missing_keys_v1"] == 0 and key_alignment["score_to_label_extra_keys_v1"] == 0,
    }
    return {
        "layer_name": "R5_2_REBUILD_SAFETY_GUARD_EVAL_V1",
        "checks_v1": checks,
        "safety_pass_v1": bool(all(checks.values())),
        "base_audit_v1": base_audit,
    }


def _gate(
    *,
    training_completed: bool,
    outputs_written: bool,
    safety_pass: bool,
    objective_eval: dict[str, Any],
    compare: dict[str, Any],
    recovery: dict[str, Any],
    r6_manifest_ready: bool,
) -> dict[str, Any]:
    strong_recall = objective_eval["strong_bad_target_recall_v1"]["recall_v1"] or 0.0
    tail_recall = objective_eval["tail_10_50_target_recall_v1"]["recall_v1"] or 0.0
    objective_learned = strong_recall >= 0.50 and tail_recall >= 0.50
    material_uplift = bool(compare["material_uplift_beyond_tiny_v3_v1"])
    if not training_completed or not outputs_written:
        decision = "NOT_ESTABLISHED"
    elif not r6_manifest_ready:
        decision = "TRUE_R5_2_REBUILD_DOWNSTREAM_R6_INPUT_NOT_READY"
    elif not safety_pass:
        decision = "TRUE_R5_2_REBUILD_RECALL_IMPROVES_BUT_SAFETY_FAILS" if material_uplift else "TRUE_R5_2_REBUILD_INVALID_FEATURE_OR_SURFACE"
    elif not objective_learned:
        decision = "TRUE_R5_2_REBUILD_OBJECTIVE_NOT_LEARNED"
    elif material_uplift or int(recovery["now_r5_2_base_eligible_v1"]) > 0:
        decision = "TRUE_R5_2_REBUILD_PASS_READY_FOR_R6"
    else:
        decision = "TRUE_R5_2_REBUILD_SAFE_BUT_TOO_WEAK"
    return {
        "layer_name": "TRUE_R5_2_REBUILD_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "training_completed_v1": training_completed,
            "outputs_written_v1": outputs_written,
            "safety_pass_v1": safety_pass,
            "objective_learned_v1": objective_learned,
            "material_uplift_over_v3_v1": material_uplift,
            "recovery_390_reported_v1": bool(recovery),
            "downstream_r6_input_ready_v1": r6_manifest_ready,
        },
    }


def _next_action_from_gate(gate: dict[str, Any]) -> str:
    decision = gate["decision_v1"]
    if decision == "TRUE_R5_2_REBUILD_PASS_READY_FOR_R6":
        return "RUN_R6_RETRAIN_FROM_TRUE_R5_2_REBUILD_SCORE_PACKAGE_EXPLICIT_FLAG"
    if decision == "TRUE_R5_2_REBUILD_SAFE_BUT_TOO_WEAK":
        return "R5_2_REBUILD_SAFE_BUT_NEEDS_OBJECTIVE_OR_FEATURE_REVIEW"
    if decision == "TRUE_R5_2_REBUILD_RECALL_IMPROVES_BUT_SAFETY_FAILS":
        return "STOP_AND_RUN_TRUE_R5_2_REBUILD_FAILURE_FORENSICS"
    if decision == "TRUE_R5_2_REBUILD_OBJECTIVE_NOT_LEARNED":
        return "FIX_R5_2_OBJECTIVE_OR_MODEL_CONFIG_FIRST"
    if decision == "TRUE_R5_2_REBUILD_DOWNSTREAM_R6_INPUT_NOT_READY":
        return "FIX_DOWNSTREAM_R6_INPUT_PACKAGE_FIRST"
    return "NOT_ESTABLISHED"


def _audit_rows(summary: dict[str, Any], validation: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("PRELAUNCH_STATUS", summary["prelaunch_status_v1"] == "PASS", summary["prelaunch_status_v1"]),
            row("NO_TRAINING_WITHOUT_FLAG", (summary["training_started_v1"] is False) or bool(summary["explicit_training_flag_v1"]), summary["training_started_v1"]),
            row("FOUNDATION_ROWS", validation["score"]["foundation_rows_v1"] == EXPECTED_FOUNDATION_ROWS, validation["score"]["foundation_rows_v1"]),
            row("ACTIVE_QUARANTINE", validation["score"]["active_rows_v1"] == EXPECTED_ACTIVE_ROWS and validation["score"]["quarantine_rows_v1"] == EXPECTED_QUARANTINE_ROWS, [validation["score"]["active_rows_v1"], validation["score"]["quarantine_rows_v1"]]),
            row("ASOF_SCHEMA", validation["score"]["asof_columns_v1"] == EXPECTED_ASOF_COLUMNS, validation["score"]["asof_columns_v1"]),
            row("LABEL_ROWS", validation["label"]["label_table_rows_v1"] == EXPECTED_LABEL_ROWS, validation["label"]["label_table_rows_v1"]),
            row("MISSED_BUCKETS", validation["label"]["missed_bucket_counts_v1"] == EXPECTED_BUCKET_COUNTS, validation["label"]["missed_bucket_counts_v1"]),
            row("AMBIGUOUS_NOT_BAD_POSITIVE", validation["label"]["ambiguous_high_mfe_bad_positive_count_v1"] == 0, validation["label"]["ambiguous_high_mfe_bad_positive_count_v1"]),
            row("KEY_ALIGNMENT", validation["key_alignment"]["label_to_score_missing_keys_v1"] == 0 and validation["key_alignment"]["score_to_label_extra_keys_v1"] == 0, validation["key_alignment"]),
            row("FORBIDDEN_FEATURES", validation["feature"]["forbidden_feature_count_v1"] == 0, validation["feature"]["forbidden_feature_count_v1"]),
            row("DOWNSTREAM_R6_PLACEHOLDER", validation["downstream_r6_manifest_materialized_v1"], validation["downstream_r6_manifest_materialized_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# True R5.2 Rebuild Runner",
            "",
            f"Status: `{summary['prelaunch_status_v1']}`",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Training started: `{summary['training_started_v1']}`",
            f"- Foundation rows: `{summary['foundation_rows_v1']}`",
            f"- Label table rows: `{summary['label_table_rows_v1']}`",
            f"- AS_OF columns: `{summary['asof_columns_v1']}`",
            f"- Feature count: `{summary['feature_count_v1']}`",
            f"- Forbidden feature count: `{summary['forbidden_feature_count_v1']}`",
            "",
            "Dry/prelaunch writes scaffolding only unless the explicit training flag is used.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    spec_dir: Path = DEFAULT_SPEC_DIR,
    output_dir: Path | None = None,
    foundation_score_package_dir: Path | None = None,
    label_table_path: Path | None = None,
    run_true_rebuild: bool = False,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    spec_dir = spec_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    _ensure_output_namespace_clean(output_dir)

    spec = _load_spec_package(spec_dir)
    score_dir = _score_package_dir(spec, foundation_score_package_dir)
    label_path = _label_table_path(spec, label_table_path)
    score = pd.read_parquet(score_dir / SCORE_FRAME)
    score_summary = _read_json(score_dir / SCORE_SUMMARY)
    label_table = pd.read_csv(label_path)

    score_validation = _validate_score_package(score_dir, score, score_summary)
    label_validation = _validate_label_table(label_table, spec["loader_spec"])
    key_alignment = _validate_key_alignment(score, label_table)
    feature_names, feature_preflight, feature_summary, forbidden_scan = _feature_matrix_preflight(score, spec["input_contract"])
    label_weight_manifest = _label_weight_manifest(label_table, spec["loader_spec"])
    training_frame = _merge_training_frame(score, label_table)

    r6_placeholder = {
        "layer_name": "R5_2_DOWNSTREAM_R6_INPUT_MANIFEST_PLACEHOLDER_V1",
        "downstream_contract_present_v1": bool(spec.get("r6_contract")),
        "future_score_columns_v1": spec["r6_contract"].get("r5_2_score_columns_for_r6_v1", []),
        "future_base_membership_flags_v1": spec["r6_contract"].get("base_membership_flags_for_r6_v1", []),
        "r6_retrain_started_v1": False,
        "blocked_until_training_execution_v1": True,
    }
    validation = {
        "score": score_validation,
        "label": label_validation,
        "key_alignment": key_alignment,
        "feature": feature_summary,
        "downstream_r6_manifest_materialized_v1": True,
    }

    training_summary: dict[str, Any] = {"training_started_v1": False}
    objective_eval: dict[str, Any] | None = None
    base_audit: dict[str, Any] | None = None
    compare_to_v3: dict[str, Any] | None = None
    missed_390_recovery: dict[str, Any] | None = None
    safety_guard: dict[str, Any] | None = None
    downstream_lock: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    if run_true_rebuild:
        training_summary = _execute_training_scaffold(
            output_dir=output_dir,
            training_frame=training_frame,
            feature_names=feature_names,
            model_config=spec["model_config"],
        )
        prediction = pd.read_parquet(output_dir / TRAINING_OUTPUT_FILES["prediction_view"])
        scored = _join_predictions(training_frame, prediction)
        objective_eval = _label_objective_eval(scored, label_weight_manifest)
        base_audit = _base_membership_audit(scored)
        compare_to_v3 = _compare_to_v3(base_audit)
        missed_390_recovery = _missed_390_recovery(scored)
        safety_guard = _safety_guard(scored, base_audit, feature_summary["forbidden_feature_count_v1"], key_alignment)
        downstream_lock = {
            "layer_name": "DOWNSTREAM_R6_INPUT_PACKAGE_LOCK_V1",
            "score_package_path_v1": str(output_dir / TRAINING_OUTPUT_FILES["score_package"]),
            "prediction_view_path_v1": str(output_dir / TRAINING_OUTPUT_FILES["prediction_view"]),
            "base_membership_path_v1": str(output_dir / TRAINING_OUTPUT_FILES["base_membership"]),
            "score_columns_for_r6_v1": spec["r6_contract"].get("r5_2_score_columns_for_r6_v1", []),
            "base_flag_for_r6_v1": BASE_FLAG_COL,
            "old_flags_not_allowed_v1": [
                "r5_2_original_base_flag_v1",
                "r5_2_v1_base_flag_v1",
                "r5_2_v2_base_flag_v1",
                "r5_2_v3_base_flag_v1",
            ],
            "expected_r6_comparison_baseline_v1": {
                "old_r6_v3_v1": {"bad_v1": 82, "tail_v1": 51},
                "wednesday_benchmark_v1": {"bad_v1": 180, "tail_v1": 149},
            },
        }
        _write_json(output_dir / TRAINING_OUTPUT_FILES["eval_summary"], objective_eval)
        _pocket_eval_report(scored).to_csv(output_dir / TRAINING_OUTPUT_FILES["pocket_eval"], index=False)
        _write_json(output_dir / TRAINING_OUTPUT_FILES["safety_guard"], safety_guard)
        _write_json(output_dir / TRAINING_OUTPUT_FILES["base_audit"], base_audit)
        _write_json(output_dir / TRAINING_OUTPUT_FILES["compare_to_v3"], compare_to_v3)
        _write_json(output_dir / TRAINING_OUTPUT_FILES["missed_390_recovery"], missed_390_recovery)
        _write_json(output_dir / TRAINING_OUTPUT_FILES["downstream_lock"], downstream_lock)
        outputs_written = all((output_dir / filename).exists() for filename in [
            TRAINING_OUTPUT_FILES["prediction_view"],
            TRAINING_OUTPUT_FILES["score_package"],
            TRAINING_OUTPUT_FILES["base_membership"],
            TRAINING_OUTPUT_FILES["eval_summary"],
            TRAINING_OUTPUT_FILES["pocket_eval"],
            TRAINING_OUTPUT_FILES["safety_guard"],
        ])
        gate = _gate(
            training_completed=True,
            outputs_written=outputs_written,
            safety_pass=bool(safety_guard["safety_pass_v1"]),
            objective_eval=objective_eval,
            compare=compare_to_v3,
            recovery=missed_390_recovery,
            r6_manifest_ready=True,
        )
        training_summary.update(
            {
                "gate_decision_v1": gate["decision_v1"],
                "bad_rows_in_base_v1": base_audit["bad_rows_in_base_v1"],
                "tail_rows_in_base_v1": base_audit["tail_rows_in_base_v1"],
                "precision_v1": base_audit["precision_v1"],
                "worst_loso_v1": base_audit["worst_loso_v1"],
                "safety_pass_v1": safety_guard["safety_pass_v1"],
                "missed_390_now_base_eligible_v1": missed_390_recovery["now_r5_2_base_eligible_v1"],
            }
        )
        _write_json(output_dir / TRAINING_OUTPUT_FILES["training_summary"], training_summary)
        _write_json(
            output_dir / TRAINING_OUTPUT_FILES["model_manifest"],
            {
                "layer_name": "R5_2_MODEL_MANIFEST_V1",
                "head_specs_v1": NEW_HEAD_SPECS,
                "model_dir_v1": str(output_dir / "models"),
            },
        )
        _write_json(output_dir / TRAINING_OUTPUT_FILES["config_manifest"], {"layer_name": "R5_2_CONFIG_MANIFEST_V1", "config_v1": training_summary["config_v1"]})
        pd.DataFrame({"feature_v1": feature_names}).to_csv(output_dir / TRAINING_OUTPUT_FILES["feature_manifest"], index=False)
        _write_json(
            output_dir / TRAINING_OUTPUT_FILES["r6_manifest"],
            {
                "layer_name": "R5_2_DOWNSTREAM_R6_INPUT_MANIFEST_V1",
                "score_package_v1": str(output_dir / TRAINING_OUTPUT_FILES["score_package"]),
                "base_membership_v1": str(output_dir / TRAINING_OUTPUT_FILES["base_membership"]),
                "prediction_view_v1": str(output_dir / TRAINING_OUTPUT_FILES["prediction_view"]),
                "score_columns_for_r6_v1": spec["r6_contract"].get("r5_2_score_columns_for_r6_v1", []),
                "base_flag_for_r6_v1": BASE_FLAG_COL,
                "r6_contract_v1": spec["r6_contract"],
            },
        )
        _write_json(output_dir / TRAINING_OUTPUT_FILES["gate"], gate)
        _write_json(
            output_dir / TRAINING_OUTPUT_FILES["next_action"],
            {
                "layer_name": "NEXT_ACTION_LOCK_V1",
                "next_action_v1": _next_action_from_gate(gate),
                "blocked_action_v1": "RUN_R6_RETRAIN_NOW_WITHOUT_EXPLICIT_FLAG",
            },
        )

    prelaunch_status = "PASS"
    decision = "DRY_PRELAUNCH_COMPLETED"
    if run_true_rebuild:
        decision = gate["decision_v1"] if gate is not None else "NOT_ESTABLISHED"
    next_action = "NEXT_AGENT_MAY_RUN_TRUE_R5_2_REBUILD_WITH_EXPLICIT_FLAG" if not run_true_rebuild else _next_action_from_gate(gate or {"decision_v1": "NOT_ESTABLISHED"})
    blocked_action = "RUN_TRAINING_WITHOUT_EXPLICIT_FLAG"
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "spec_dir_v1": str(spec_dir),
        "foundation_score_package_dir_v1": str(score_dir),
        "label_table_path_v1": str(label_path),
        "decision_v1": decision,
        "prelaunch_status_v1": prelaunch_status,
        "training_started_v1": bool(run_true_rebuild),
        "explicit_training_flag_v1": bool(run_true_rebuild),
        "foundation_rows_v1": score_validation["foundation_rows_v1"],
        "active_rows_v1": score_validation["active_rows_v1"],
        "quarantine_rows_v1": score_validation["quarantine_rows_v1"],
        "label_table_rows_v1": label_validation["label_table_rows_v1"],
        "asof_columns_v1": score_validation["asof_columns_v1"],
        "missed_bad_tail_represented_v1": label_validation["missed_bad_tail_represented_v1"],
        "missed_bucket_counts_v1": label_validation["missed_bucket_counts_v1"],
        "ambiguous_high_mfe_bad_positive_count_v1": label_validation["ambiguous_high_mfe_bad_positive_count_v1"],
        "runner_protect_bad_positive_count_v1": label_validation["runner_protect_bad_positive_count_v1"],
        "feature_count_v1": feature_summary["feature_count_v1"],
        "feature_families_v1": feature_summary["feature_families_v1"],
        "forbidden_feature_count_v1": feature_summary["forbidden_feature_count_v1"],
        "key_alignment_v1": key_alignment,
        "downstream_r6_manifest_placeholder_written_v1": True,
        "score_package_written_v1": bool(run_true_rebuild and (output_dir / TRAINING_OUTPUT_FILES["score_package"]).exists()),
        "base_membership_written_v1": bool(run_true_rebuild and (output_dir / TRAINING_OUTPUT_FILES["base_membership"]).exists()),
        "downstream_r6_input_manifest_written_v1": bool(run_true_rebuild and (output_dir / TRAINING_OUTPUT_FILES["r6_manifest"]).exists()),
        "bad_rows_in_base_v1": None if base_audit is None else base_audit["bad_rows_in_base_v1"],
        "tail_rows_in_base_v1": None if base_audit is None else base_audit["tail_rows_in_base_v1"],
        "precision_v1": None if base_audit is None else base_audit["precision_v1"],
        "worst_loso_v1": None if base_audit is None else base_audit["worst_loso_v1"],
        "safety_pass_v1": None if safety_guard is None else safety_guard["safety_pass_v1"],
        "missed_390_now_base_eligible_v1": None if missed_390_recovery is None else missed_390_recovery["now_r5_2_base_eligible_v1"],
        "missed_390_still_not_recoverable_v1": None if missed_390_recovery is None else missed_390_recovery["still_not_recoverable_v1"],
        "true_rebuild_gate_v1": None if gate is None else gate["decision_v1"],
        "next_action_v1": next_action,
        "blocked_action_v1": blocked_action,
        "hard_status_v1": {
            "BEVIST": [
                "True R5.2 rebuild runner loads the locked spec package and validates Monday foundation, label table, weights, features, keys, and downstream R6 contract.",
                "Training execution completed with explicit flag." if run_true_rebuild else "Dry/prelaunch completed without starting training.",
                "Training execution is gated by an explicit flag.",
            ],
            "INDIKERT": [
                "The rebuilt score package is ready for R6 only if the gate is PASS." if run_true_rebuild else "The same runner can execute the multi-head rebuild path when the explicit flag is used.",
            ],
            "IKKE_ETABLERT": [
                "Downstream R6 uplift is not established until a separate explicit R6 retrain/eval is run.",
            ],
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": decision,
        "prelaunch_status_v1": prelaunch_status,
        "training_started_v1": bool(run_true_rebuild),
        "next_action_v1": next_action,
        "blocked_action_v1": blocked_action,
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "spec_files_v1": SPEC_FILES,
        "dry_output_files_v1": DRY_OUTPUT_FILES,
        "training_output_files_v1": TRAINING_OUTPUT_FILES,
        "input_files_v1": {
            "score_frame_v1": str(score_dir / SCORE_FRAME),
            "score_summary_v1": str(score_dir / SCORE_SUMMARY),
            "label_table_v1": str(label_path),
        },
    }
    prelaunch_report = {
        "layer_name": "R5_2_REBUILD_PRELAUNCH_VALIDATION_V1",
        "status_v1": prelaunch_status,
        "score_validation_v1": score_validation,
        "label_validation_v1": label_validation,
        "key_alignment_v1": key_alignment,
        "feature_summary_v1": feature_summary,
        "output_namespace_clean_v1": True,
        "downstream_r6_consumption_contract_present_v1": bool(spec.get("r6_contract")),
    }
    audit = _audit_rows(summary, validation)

    _write_json(output_dir / DRY_OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / DRY_OUTPUT_FILES["status"], status)
    _write_json(output_dir / DRY_OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / DRY_OUTPUT_FILES["prelaunch_report"], prelaunch_report)
    _write_json(output_dir / DRY_OUTPUT_FILES["feature_matrix_json"], feature_summary)
    feature_preflight.to_csv(output_dir / DRY_OUTPUT_FILES["feature_matrix_csv"], index=False)
    label_weight_manifest.to_csv(output_dir / DRY_OUTPUT_FILES["label_weight_manifest"], index=False)
    forbidden_scan.to_csv(output_dir / DRY_OUTPUT_FILES["forbidden_feature_scan"], index=False)
    _write_json(output_dir / DRY_OUTPUT_FILES["r6_placeholder"], r6_placeholder)
    audit.to_csv(output_dir / DRY_OUTPUT_FILES["audit"], index=False)
    (output_dir / DRY_OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--foundation-score-package-dir", type=Path, default=None)
    parser.add_argument("--label-table-path", type=Path, default=None)
    parser.add_argument("--run-true-r5-2-rebuild", action="store_true")
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        foundation_score_package_dir=args.foundation_score_package_dir,
        label_table_path=args.label_table_path,
        run_true_rebuild=args.run_true_r5_2_rebuild,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
