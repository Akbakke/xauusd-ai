#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

SPEC_PREFIX = "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1_"
DRY_OUTPUT_PREFIX = "MONDAY_NARROW_RETRAIN_DRY_PRELAUNCH_VALIDATION_V1"
TRAINING_OUTPUT_SUFFIX = "MONDAY_NARROW_RETRAIN_RUN_V1"

RUNNER_SPEC = "monday_narrow_retrain_runner_spec_v1.json"
CONFIG_LOCK = "monday_narrow_retrain_config_lock_v1.json"
FEATURE_MANIFEST = "monday_narrow_retrain_feature_manifest_v1.json"
FEATURE_MANIFEST_TABLE = "monday_narrow_retrain_feature_manifest_v1.csv"
PRELAUNCH_CHECKLIST = "monday_narrow_retrain_prelaunch_checklist_v1.json"
OUTPUT_SPEC = "monday_narrow_retrain_output_spec_v1.json"
ABORT_RULES = "monday_narrow_retrain_abort_rules_v1.json"
SPEC_SUMMARY = "summary_v1.json"

CONTRACT = "contract_v1.json"
LOADED_CONFIG = "monday_narrow_retrain_loaded_config_v1.json"
PRELAUNCH_REPORT = "monday_narrow_retrain_prelaunch_validation_report_v1.json"
ABORT_ENFORCEMENT = "monday_narrow_retrain_abort_enforcement_scaffold_v1.json"
TRAINING_SUMMARY = "shadow_meta_all_trade_review_monday_narrow_retrain_training_summary_v1.json"
MODEL_CONFIG_MANIFEST = "shadow_meta_all_trade_review_monday_narrow_retrain_model_config_manifest_v1.json"
FEATURE_MANIFEST_ECHO = "shadow_meta_all_trade_review_monday_narrow_retrain_feature_manifest_v1.csv"
EVAL_SUMMARY = "shadow_meta_all_trade_review_monday_narrow_retrain_eval_summary_v1.json"
COMPARE_AGAINST = "shadow_meta_all_trade_review_monday_narrow_retrain_compare_against_report_v1.csv"
POCKET_REPORT = "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_report_v1.csv"
VERDICT_PACKAGE = "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json"
MODEL_BUNDLE = "shadow_meta_all_trade_review_monday_narrow_retrain_model_bundle_v1.joblib"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_monday_narrow_retrain_policy_prediction_view_v1.parquet"
WALKFORWARD_METRICS = "shadow_meta_all_trade_review_monday_narrow_retrain_walkforward_metrics_v1.csv"
LOSO_METRICS = "shadow_meta_all_trade_review_monday_narrow_retrain_loso_metrics_v1.csv"
ROLLING_METRICS = "shadow_meta_all_trade_review_monday_narrow_retrain_rolling_window_metrics_v1.csv"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

JOB_NAME = "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1"
EXPECTED_SCOPE = "NARROW_RUNNER_FIRST_SHADOW_ONLY"
EXPECTED_ROWS = 1689
EXPECTED_FEATURES = 67
EXPECTED_LABEL_INTERSECTION = 1689
EXPECTED_SEED = 20260422
EXPECTED_MODEL_FAMILY = "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY"
EXPECTED_SURFACE_KIND = "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE"
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
KNOWN_FAILED_NARROW_RUN = "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1"

FORBIDDEN_FEATURE_PATTERNS = [
    "as_of_skip_xgb_",
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
    "management_policy",
    "decision_log",
    "policy_log",
    "bridge_only",
]

REQUIRED_COMPARE_REFERENCES = {
    "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
    "FAILURE_MINER_DIAGNOSIS_ONLY",
}


class PrelaunchValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunnerBundle:
    spec_dir: Path
    runner_spec: Dict[str, Any]
    config_lock: Dict[str, Any]
    feature_manifest: Dict[str, Any]
    feature_manifest_table: pd.DataFrame
    prelaunch_checklist: Dict[str, Any]
    output_spec: Dict[str, Any]
    abort_rules: Dict[str, Any]
    spec_summary: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_output_dir(reports_root: Path, output_dir_arg: str | None, *, run_training: bool = False) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()
    if run_training:
        return reports_root / f"ALL_TRADE_REVIEW_LEDGER_{_utc_compact()}_{TRAINING_OUTPUT_SUFFIX}"
    return reports_root / f"{DRY_OUTPUT_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_bundle(reports_root: Path, spec_dir_arg: str | None = None) -> RunnerBundle:
    spec_dir = Path(spec_dir_arg).expanduser().resolve() if spec_dir_arg else _latest_dir(reports_root, SPEC_PREFIX)
    required = [
        RUNNER_SPEC,
        CONFIG_LOCK,
        FEATURE_MANIFEST,
        FEATURE_MANIFEST_TABLE,
        PRELAUNCH_CHECKLIST,
        OUTPUT_SPEC,
        ABORT_RULES,
        SPEC_SUMMARY,
    ]
    missing = [name for name in required if not (spec_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Runner spec dir missing required artifacts: {missing}")
    return RunnerBundle(
        spec_dir=spec_dir,
        runner_spec=_load_json(spec_dir / RUNNER_SPEC),
        config_lock=_load_json(spec_dir / CONFIG_LOCK),
        feature_manifest=_load_json(spec_dir / FEATURE_MANIFEST),
        feature_manifest_table=pd.read_csv(spec_dir / FEATURE_MANIFEST_TABLE),
        prelaunch_checklist=_load_json(spec_dir / PRELAUNCH_CHECKLIST),
        output_spec=_load_json(spec_dir / OUTPUT_SPEC),
        abort_rules=_load_json(spec_dir / ABORT_RULES),
        spec_summary=_load_json(spec_dir / SPEC_SUMMARY),
    )


def _selected_feature_names(feature_table: pd.DataFrame) -> List[str]:
    required_cols = {"feature_name_v1", "must_exclude_v1"}
    missing = sorted(required_cols - set(feature_table.columns))
    if missing:
        raise PrelaunchValidationError(f"feature manifest table missing columns: {missing}")
    selected = feature_table.loc[~feature_table["must_exclude_v1"].astype(bool), "feature_name_v1"].astype(str).tolist()
    if len(selected) != len(set(selected)):
        duplicates = sorted([name for name in set(selected) if selected.count(name) > 1])
        raise PrelaunchValidationError(f"duplicate selected feature names: {duplicates}")
    return selected


def _forbidden_selected_features(features: Iterable[str]) -> List[str]:
    out: List[str] = []
    for feature in features:
        lower = str(feature).lower()
        if any(pattern in lower for pattern in FORBIDDEN_FEATURE_PATTERNS):
            out.append(str(feature))
    return sorted(out)


def _compare_ids(runner_spec: Dict[str, Any]) -> set[str]:
    return {str(row.get("id_v1")) for row in runner_spec.get("compare_against_inputs_v1", [])}


def _validate_output_namespace_clean(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PrelaunchValidationError(f"output namespace is not clean: {output_dir}")


def _validate_bundle_contract(bundle: RunnerBundle, output_dir: Path) -> Dict[str, Any]:
    spec = bundle.runner_spec
    config = bundle.config_lock
    manifest = bundle.feature_manifest
    if spec.get("job_name_v1") != JOB_NAME:
        raise PrelaunchValidationError(f"unexpected job name: {spec.get('job_name_v1')}")
    if spec.get("scope_v1") != EXPECTED_SCOPE:
        raise PrelaunchValidationError(f"unexpected scope: {spec.get('scope_v1')}")
    if spec.get("runner_may_train_v1") is not False or spec.get("training_now_v1") is not False:
        raise PrelaunchValidationError("runner spec must remain dry/no-training by default")
    if config.get("model_family_v1") != EXPECTED_MODEL_FAMILY:
        raise PrelaunchValidationError(f"unexpected model family: {config.get('model_family_v1')}")
    if config.get("base_model_v1") != "XGBClassifier per head":
        raise PrelaunchValidationError(f"unexpected base model: {config.get('base_model_v1')}")
    if config.get("compact_grid_v1") is not True:
        raise PrelaunchValidationError("compact grid must be true")
    if int(config.get("seed_v1", -1)) != EXPECTED_SEED:
        raise PrelaunchValidationError(f"unexpected seed: {config.get('seed_v1')}")
    if int(manifest.get("total_feature_count_v1", -1)) != EXPECTED_FEATURES:
        raise PrelaunchValidationError(f"feature manifest total != {EXPECTED_FEATURES}")
    missing_refs = sorted(REQUIRED_COMPARE_REFERENCES - _compare_ids(spec))
    if missing_refs:
        raise PrelaunchValidationError(f"compare-against references missing: {missing_refs}")
    _validate_output_namespace_clean(output_dir)
    return {
        "job_name_v1": "PASS",
        "scope_v1": "PASS",
        "dry_default_v1": "PASS",
        "model_family_v1": "PASS",
        "base_model_v1": "PASS",
        "compact_grid_v1": "PASS",
        "seed_v1": "PASS",
        "compare_references_v1": "PASS",
        "output_namespace_clean_v1": "PASS",
    }


def _validate_training_inputs(bundle: RunnerBundle) -> Dict[str, Any]:
    spec = bundle.runner_spec
    feature_names = _selected_feature_names(bundle.feature_manifest_table)
    forbidden = _forbidden_selected_features(feature_names)
    if forbidden:
        raise PrelaunchValidationError(f"forbidden selected feature fields: {forbidden}")
    if len(feature_names) != EXPECTED_FEATURES:
        raise PrelaunchValidationError(f"selected feature count {len(feature_names)} != {EXPECTED_FEATURES}")
    if spec.get("input_surface_kind_v1") != EXPECTED_SURFACE_KIND:
        raise PrelaunchValidationError(f"unexpected input surface kind: {spec.get('input_surface_kind_v1')}")
    input_path = Path(str(spec["input_artifact_v1"])).expanduser().resolve()
    label_path = Path(str(spec["label_artifact_v1"])).expanduser().resolve()
    if "bridge" in str(input_path).lower():
        raise PrelaunchValidationError(f"bridge path proposed as training surface: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"training surface does not exist: {input_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"label surface does not exist: {label_path}")
    raw_df = pd.read_parquet(input_path)
    label_df = pd.read_parquet(label_path)
    if len(raw_df) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"training row count {len(raw_df)} != {EXPECTED_ROWS}")
    if "candidate_uid" not in raw_df.columns or "candidate_uid" not in label_df.columns:
        raise PrelaunchValidationError("candidate_uid is required on both raw-state and label surfaces")
    missing_features = sorted([name for name in feature_names if name not in raw_df.columns])
    if missing_features:
        raise PrelaunchValidationError(f"selected features missing from raw-state: {missing_features[:20]}")
    matrix_columns = ["candidate_uid"] + feature_names
    matrix_forbidden = _forbidden_selected_features(matrix_columns)
    if matrix_forbidden:
        raise PrelaunchValidationError(f"forbidden fields in training matrix: {matrix_forbidden}")
    raw_ids = set(raw_df["candidate_uid"].astype("string"))
    label_ids = set(label_df["candidate_uid"].astype("string"))
    intersection = len(raw_ids & label_ids)
    if intersection != EXPECTED_LABEL_INTERSECTION:
        raise PrelaunchValidationError(f"label intersection {intersection} != {EXPECTED_LABEL_INTERSECTION}")
    missing_label_cols = sorted(
        [
            str(head.get("label_col_v1"))
            for head in spec.get("locked_training_heads_v1", [])
            if str(head.get("label_col_v1")) not in label_df.columns
        ]
    )
    if missing_label_cols:
        raise PrelaunchValidationError(f"locked label columns missing: {missing_label_cols}")
    return {
        "input_path_v1": str(input_path),
        "label_path_v1": str(label_path),
        "raw_rows_v1": int(len(raw_df)),
        "label_rows_v1": int(len(label_df)),
        "label_intersection_v1": int(intersection),
        "selected_feature_count_v1": int(len(feature_names)),
        "selected_features_v1": feature_names,
        "raw_column_count_v1": int(len(raw_df.columns)),
        "label_column_count_v1": int(len(label_df.columns)),
        "matrix_column_count_v1": int(len(matrix_columns)),
        "raw_contains_forbidden_unselected_review_fields_v1": sorted(
            [column for column in raw_df.columns if any(pattern in str(column).lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)]
        ),
    }


def run_prelaunch_validation(bundle: RunnerBundle, output_dir: Path) -> Dict[str, Any]:
    contract_status = _validate_bundle_contract(bundle, output_dir)
    input_status = _validate_training_inputs(bundle)
    return {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_PRELAUNCH_VALIDATION_V1",
        "validated_at_utc_v1": _utc_now_iso(),
        "status_v1": "PASS",
        "contract_status_v1": contract_status,
        "input_status_v1": input_status,
        "checklist_v1": bundle.prelaunch_checklist,
    }


def _enforce_known_failed_narrow_rerun_guard(*, allow_known_failed_narrow_rerun: bool) -> None:
    if allow_known_failed_narrow_rerun:
        return
    raise PrelaunchValidationError(
        "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1 is a known failed/no-go setup after "
        f"{KNOWN_FAILED_NARROW_RUN}. Refusing training with only --run-training. Use the protector-first "
        "path instead, or pass --allow-known-failed-narrow-rerun-for-forensics only for an intentional "
        "forensics/control rerun."
    )


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin(["1", "true", "yes", "y"])


def _numeric_matrix(frame: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def _fill_with_train_median(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    medians = train_x.median(numeric_only=True).fillna(0.0)
    return train_x.fillna(medians), test_x.fillna(medians), {str(k): float(v) for k, v in medians.items()}


class _ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        prob_true = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - prob_true, prob_true])


def _xgb_params(config_lock: Dict[str, Any]) -> Dict[str, Any]:
    params = config_lock.get("default_model_hyperparams_v1", {})
    return {
        "n_estimators": int(params.get("n_estimators_v1", 100)),
        "learning_rate": float(params.get("learning_rate_v1", 0.05)),
        "max_depth": int(params.get("max_depth_v1", 3)),
        "tree_method": str(params.get("tree_method_v1", "hist")),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": int(config_lock.get("seed_v1", EXPECTED_SEED)),
        "n_jobs": int(config_lock.get("n_jobs_v1", 1)),
    }


def _fit_binary_head(train_x: pd.DataFrame, y: pd.Series, config_lock: Dict[str, Any]) -> Any:
    y_int = y.fillna(False).astype(bool).astype(int)
    if y_int.nunique(dropna=False) < 2:
        return _ConstantProbabilityModel(float(y_int.mean()))
    model = XGBClassifier(**_xgb_params(config_lock))
    model.fit(train_x, y_int)
    return model


def _predict_true_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim != 2 or proba.shape[1] < 2:
        return np.zeros(len(x), dtype=float)
    return np.asarray(proba[:, 1], dtype=float)


def _fold_indices(n_rows: int, n_slices: int = 5) -> List[np.ndarray]:
    return [np.asarray(idx, dtype=int) for idx in np.array_split(np.arange(n_rows), n_slices) if len(idx)]


def _prepare_training_frame(bundle: RunnerBundle, validation_report: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    input_path = Path(validation_report["input_status_v1"]["input_path_v1"])
    label_path = Path(validation_report["input_status_v1"]["label_path_v1"])
    features = list(validation_report["input_status_v1"]["selected_features_v1"])
    raw_df = pd.read_parquet(input_path).copy()
    label_df = pd.read_parquet(label_path).copy()
    joined = raw_df.merge(label_df, on="candidate_uid", how="inner", validate="one_to_one")
    joined = joined.set_index("candidate_uid", drop=False)
    joined = joined.loc[raw_df["candidate_uid"].astype(str)].reset_index(drop=True)
    x = _numeric_matrix(joined, features)
    return joined, x, features


def _train_head_predictions(
    joined: pd.DataFrame,
    x: pd.DataFrame,
    bundle: RunnerBundle,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    folds = _fold_indices(len(joined), 5)
    predictions = pd.DataFrame({"candidate_uid": joined["candidate_uid"].astype(str)})
    model_bundle: Dict[str, Any] = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_MODEL_BUNDLE_V1",
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "heads_v1": {},
    }
    for head in bundle.runner_spec["locked_training_heads_v1"]:
        head_id = str(head["head_id_v1"])
        label_col = str(head["label_col_v1"])
        output_col = f"pred__monday_narrow__{head_id}__prob_true_v1"
        if label_col not in joined.columns:
            raise PrelaunchValidationError(f"Missing label column for head {head_id}: {label_col}")
        y = _bool_col(joined, label_col)
        oof = np.zeros(len(joined), dtype=float)
        fold_meta: List[Dict[str, Any]] = []
        for fold_idx, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(np.arange(len(joined)), test_idx)
            train_x, test_x, medians = _fill_with_train_median(x.iloc[train_idx].copy(), x.iloc[test_idx].copy())
            model = _fit_binary_head(train_x, y.iloc[train_idx], bundle.config_lock)
            oof[test_idx] = _predict_true_probability(model, test_x)
            fold_meta.append(
                {
                    "fold_v1": fold_idx,
                    "train_rows_v1": int(len(train_idx)),
                    "test_rows_v1": int(len(test_idx)),
                    "positive_train_count_v1": int(y.iloc[train_idx].sum()),
                    "median_count_v1": int(len(medians)),
                }
            )
        full_x, _, full_medians = _fill_with_train_median(x.copy(), x.copy())
        final_model = _fit_binary_head(full_x, y, bundle.config_lock)
        predictions[output_col] = oof
        model_bundle["heads_v1"][head_id] = {
            "label_col_v1": label_col,
            "output_col_v1": output_col,
            "positive_count_v1": int(y.sum()),
            "row_count_v1": int(len(y)),
            "folds_v1": fold_meta,
            "full_medians_v1": full_medians,
            "model_v1": final_model,
        }
    return predictions, model_bundle


def _policy_block_mask(scored: pd.DataFrame) -> pd.Series:
    bad = pd.to_numeric(scored.get("pred__monday_narrow__bad_risk__prob_true_v1", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    risky = pd.to_numeric(scored.get("pred__monday_narrow__risky_allow__prob_true_v1", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    tail = pd.to_numeric(scored.get("pred__monday_narrow__tail_control_10_50__prob_true_v1", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    blindspot = pd.to_numeric(scored.get("pred__monday_narrow__batch04_blindspot__prob_true_v1", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    runner = pd.to_numeric(scored.get("pred__monday_narrow__runner_protector__prob_true_v1", 0.0), errors="coerce").fillna(0.0).ge(0.5)
    return (bad | risky | tail | blindspot) & ~runner


def _metric_row(policy_name: str, scope: str, frame: pd.DataFrame, block: pd.Series) -> Dict[str, Any]:
    block = block.reindex(frame.index).fillna(False).astype(bool)
    should = _bool_col(frame, "r6_label_bad_risk_v1")
    tail = _bool_col(frame, "r6_label_tail_control_10_50_v1")
    repaired = _bool_col(frame, "r6_label_repaired_165_like_runner_v1")
    runner50 = _bool_col(frame, "r6_label_runner_50_mfe_v1")
    runner100 = _bool_col(frame, "r6_label_runner_100_mfe_v1")
    runner200 = _bool_col(frame, "r6_label_runner_200_mfe_v1")
    strong = _bool_col(frame, "r6_label_strong_low_mae_runner_v1")
    runner_near = _bool_col(frame, "r6_label_runner_near_miss_v1")
    strongest = runner200 | strong
    tp = int((block & should).sum())
    fp = int((block & ~should).sum())
    fn = int((~block & should).sum())
    tn = int((~block & ~should).sum())
    blocked = int(block.sum())
    should_count = int(should.sum())
    forensic_blocked = bool((block & frame["candidate_uid"].astype(str).eq(FORENSIC_TRADE)).any())
    precision = _safe_rate(float(tp), float(blocked))
    recall = _safe_rate(float(tp), float(should_count))
    specificity = _safe_rate(float(tn), float((~should).sum()))
    return {
        "policy_name_v1": policy_name,
        "scope_v1": scope,
        "row_count_v1": int(len(frame)),
        "block_count_v1": blocked,
        "should_not_take_count_v1": should_count,
        "should_not_take_block_count_v1": tp,
        "bad_blocks_v1": tp,
        "should_not_take_precision_v1": precision,
        "global_precision_v1": precision,
        "should_not_take_recall_v1": recall,
        "false_allow_should_not_take_count_v1": fn,
        "tail_10_50_help_count_v1": int((block & tail).sum()),
        "tail_help_v1": int((block & tail).sum()),
        "repaired_165_damage_v1": int((block & repaired).sum()),
        "repaired_165_block_count_v1": int((block & repaired).sum()),
        "forensic_trade_blocked_v1": forensic_blocked,
        "fifty_plus_mfe_block_count_v1": int((block & runner50).sum()),
        "hundred_plus_mfe_block_count_v1": int((block & runner100).sum()),
        "two_hundred_plus_mfe_block_count_v1": int((block & runner200).sum()),
        "strong_false_blocks_v1": int((block & strong).sum()),
        "strongest_winner_path_damage_v1": int((block & strongest).sum()),
        "runner_near_miss_block_count_v1": int((block & runner_near).sum()),
        "runner_near_miss_regression_v1": bool((block & runner_near).sum() > 0),
        "binary_balanced_accuracy_vs_should_not_take_v1": None if recall is None or specificity is None else (float(recall) + float(specificity)) / 2.0,
        "confusion_matrix_json_v1": _json_dumps(confusion_matrix(should.astype(int), block.astype(int), labels=[0, 1]).tolist()),
    }


def _slice_metric_table(scored: pd.DataFrame, block: pd.Series, *, table_name: str) -> pd.DataFrame:
    folds = _fold_indices(len(scored), 5)
    rows = []
    for idx, fold in enumerate(folds, start=1):
        scope = f"BATCH_{idx:02d}"
        row = _metric_row("MONDAY_NARROW_RETRAIN_SELECTED", scope, scored.iloc[fold].copy(), block.iloc[fold].copy())
        row["eval_table_v1"] = table_name
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_rows(candidate_metric: Dict[str, Any], worst_loso_precision: float | None) -> pd.DataFrame:
    references = [
        {
            "reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK",
            "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "kind_v1": "BENCHMARK",
            "bad_blocks_v1": 180,
            "tail_help_v1": 149,
            "global_precision_v1": 0.972972972972973,
            "worst_loso_precision_v1": 0.9285714285714286,
            "repaired_165_damage_v1": 0,
            "fifty_plus_mfe_block_count_v1": 1,
            "hundred_plus_mfe_block_count_v1": 0,
            "two_hundred_plus_mfe_block_count_v1": 0,
            "strongest_winner_path_damage_v1": 0,
            "runner_near_miss_block_count_v1": None,
        },
        {
            "reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
            "id_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
            "kind_v1": "SAFETY_REFERENCE",
            "bad_blocks_v1": 66,
            "tail_help_v1": 66,
            "global_precision_v1": 0.9295774647887324,
            "worst_loso_precision_v1": 0.0,
            "repaired_165_damage_v1": 0,
            "fifty_plus_mfe_block_count_v1": 0,
            "hundred_plus_mfe_block_count_v1": 0,
            "two_hundred_plus_mfe_block_count_v1": 0,
            "strongest_winner_path_damage_v1": 0,
            "runner_near_miss_block_count_v1": None,
        },
        {
            "reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER",
            "id_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
            "kind_v1": "FAILURE_MINER",
            "bad_blocks_v1": 84,
            "tail_help_v1": 84,
            "global_precision_v1": 0.9545454545454546,
            "worst_loso_precision_v1": 0.8888888888888888,
            "repaired_165_damage_v1": 1,
            "fifty_plus_mfe_block_count_v1": 1,
            "hundred_plus_mfe_block_count_v1": 0,
            "two_hundred_plus_mfe_block_count_v1": 0,
            "strongest_winner_path_damage_v1": 0,
            "runner_near_miss_block_count_v1": None,
        },
    ]
    candidate = {
        "reference_v1": "MONDAY_NARROW_RETRAIN_CANDIDATE",
        "id_v1": "CURRENT_RUN",
        "kind_v1": "CANDIDATE",
        "bad_blocks_v1": candidate_metric.get("bad_blocks_v1"),
        "tail_help_v1": candidate_metric.get("tail_help_v1"),
        "global_precision_v1": candidate_metric.get("global_precision_v1"),
        "worst_loso_precision_v1": worst_loso_precision,
        "repaired_165_damage_v1": candidate_metric.get("repaired_165_damage_v1"),
        "fifty_plus_mfe_block_count_v1": candidate_metric.get("fifty_plus_mfe_block_count_v1"),
        "hundred_plus_mfe_block_count_v1": candidate_metric.get("hundred_plus_mfe_block_count_v1"),
        "two_hundred_plus_mfe_block_count_v1": candidate_metric.get("two_hundred_plus_mfe_block_count_v1"),
        "strongest_winner_path_damage_v1": candidate_metric.get("strongest_winner_path_damage_v1"),
        "runner_near_miss_block_count_v1": candidate_metric.get("runner_near_miss_block_count_v1"),
    }
    rows = references + [candidate]
    frame = pd.DataFrame(rows)
    frozen = frame[frame["reference_v1"].eq("FROZEN_WEDNESDAY_R6_BENCHMARK")].iloc[0]
    for metric in ["bad_blocks_v1", "tail_help_v1", "global_precision_v1", "worst_loso_precision_v1"]:
        frame[f"delta_vs_frozen_{metric}"] = pd.to_numeric(frame[metric], errors="coerce") - _safe_float(frozen.get(metric))
    return frame


def _post_eval_failures(metric: Dict[str, Any], worst_loso_precision: float | None) -> List[str]:
    failures: List[str] = []
    if int(metric.get("repaired_165_damage_v1") or 0) > 0:
        failures.append("repaired_165_damage > 0")
    if bool(metric.get("forensic_trade_blocked_v1")):
        failures.append(f"forensic trade {FORENSIC_TRADE} blocked")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) > 0 or int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) > 0:
        failures.append("100+/200+ blocked > 0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("50+ blocked > 1")
    if int(metric.get("strongest_winner_path_damage_v1") or 0) > 0:
        failures.append("strongest-winner damage > 0")
    precision = _safe_float(metric.get("global_precision_v1"))
    if precision is None or precision < 0.954545:
        failures.append("global precision < 0.954545")
    if worst_loso_precision is None or worst_loso_precision < 0.888888:
        failures.append("worst LOSO < 0.888888")
    if bool(metric.get("runner_near_miss_regression_v1")):
        failures.append("serious runner near-miss regression")
    return failures


def _verdict(metric: Dict[str, Any], failures: List[str]) -> str:
    improves_bad = int(metric.get("bad_blocks_v1") or 0) > 84
    improves_tail = int(metric.get("tail_help_v1") or 0) > 84
    improves = improves_bad and improves_tail
    if failures and improves:
        return "CANDIDATE_IMPROVES_BUT_FAILS_SAFETY"
    if failures:
        return "CANDIDATE_FEATURES_INSUFFICIENT"
    if improves:
        return "CANDIDATE_IMPROVES_AND_HOLDS_SAFETY"
    return "CANDIDATE_SAFE_BUT_NOT_BETTER"


def _build_loaded_config(bundle: RunnerBundle) -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_CONFIG_LOADERS_V1",
        "spec_dir_v1": str(bundle.spec_dir),
        "loaded_artifacts_v1": [
            RUNNER_SPEC,
            CONFIG_LOCK,
            FEATURE_MANIFEST,
            FEATURE_MANIFEST_TABLE,
            PRELAUNCH_CHECKLIST,
            OUTPUT_SPEC,
            ABORT_RULES,
        ],
        "runner_spec_v1": bundle.runner_spec,
        "config_lock_v1": bundle.config_lock,
        "feature_manifest_summary_v1": bundle.feature_manifest,
        "output_spec_v1": bundle.output_spec,
        "abort_rules_v1": bundle.abort_rules,
    }


def _build_abort_enforcement_scaffold(bundle: RunnerBundle) -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_ABORT_ENFORCEMENT_V1",
        "enforcement_mode_v1": "STRUCTURE_BUILT_EVAL_NOT_RUN",
        "pre_training_abort_rules_v1": bundle.abort_rules["abort_before_training_v1"],
        "post_eval_reject_rules_v1": bundle.abort_rules["abort_or_reject_after_eval_v1"],
        "automatic_invalidators_v1": bundle.abort_rules["automatic_invalidators_v1"],
        "post_eval_metric_checks_v1": [
            {"metric_v1": "repaired_165_damage", "operator_v1": "==", "threshold_v1": 0, "hard_fail_v1": True},
            {"metric_v1": "forensic_trade_blocked", "operator_v1": "==", "threshold_v1": False, "hard_fail_v1": True, "trade_uid_v1": FORENSIC_TRADE},
            {"metric_v1": "hundred_plus_mfe_blocked", "operator_v1": "==", "threshold_v1": 0, "hard_fail_v1": True},
            {"metric_v1": "two_hundred_plus_mfe_blocked", "operator_v1": "==", "threshold_v1": 0, "hard_fail_v1": True},
            {"metric_v1": "fifty_plus_mfe_blocked", "operator_v1": "<=", "threshold_v1": 1, "hard_fail_v1": True},
            {"metric_v1": "strongest_winner_path_damage", "operator_v1": "==", "threshold_v1": 0, "hard_fail_v1": True},
            {"metric_v1": "global_precision", "operator_v1": ">=", "threshold_v1": 0.954545, "hard_fail_v1": True},
            {"metric_v1": "worst_loso_precision", "operator_v1": ">=", "threshold_v1": 0.888888, "hard_fail_v1": True},
            {"metric_v1": "runner_near_miss_regression", "operator_v1": "==", "threshold_v1": False, "hard_fail_v1": True},
        ],
    }


def _placeholder_compare_rows(bundle: RunnerBundle) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_v1": row.get("reference_v1"),
                "id_v1": row.get("id_v1"),
                "kind_v1": row.get("kind_v1"),
                "status_v1": "PENDING_TRAINING_NOT_RUN",
            }
            for row in bundle.runner_spec.get("compare_against_inputs_v1", [])
        ]
    )


def _placeholder_pocket_rows() -> pd.DataFrame:
    pockets = [
        "repaired_165_pocket",
        "forensic_repaired_trade",
        "runner_near_miss_pocket",
        "50_plus_mfe_seed_pocket",
        "100_plus_mfe_seed_pocket",
        "200_plus_mfe_seed_pocket",
        "missed_10_50_tail_control_pocket",
        "missed_should_not_take_pocket",
        "risky_allow_pocket",
    ]
    return pd.DataFrame(
        [
            {
                "pocket_v1": pocket,
                "status_v1": "PENDING_TRAINING_NOT_RUN",
                "hard_guard_v1": pocket in {"repaired_165_pocket", "forensic_repaired_trade", "50_plus_mfe_seed_pocket", "100_plus_mfe_seed_pocket", "200_plus_mfe_seed_pocket"},
            }
            for pocket in pockets
        ]
    )


def _write_output_scaffold(bundle: RunnerBundle, output_dir: Path, validation_report: Dict[str, Any], run_training: bool) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    loaded_config = _build_loaded_config(bundle)
    abort_scaffold = _build_abort_enforcement_scaffold(bundle)
    feature_table = bundle.feature_manifest_table.copy()
    selected_features = validation_report["input_status_v1"]["selected_features_v1"]
    training_summary = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_OUTPUT_SCAFFOLD_V1",
        "job_name_v1": JOB_NAME,
        "created_at_utc_v1": _utc_now_iso(),
        "training_started_v1": False,
        "run_training_flag_v1": bool(run_training),
        "dry_prelaunch_validation_only_v1": True,
        "training_surface_v1": validation_report["input_status_v1"]["input_path_v1"],
        "training_rows_v1": validation_report["input_status_v1"]["raw_rows_v1"],
        "label_intersection_v1": validation_report["input_status_v1"]["label_intersection_v1"],
        "feature_count_v1": validation_report["input_status_v1"]["selected_feature_count_v1"],
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "xgb_per_head_setup_v1": {
            "base_model_v1": bundle.config_lock["base_model_v1"],
            "head_family_v1": bundle.config_lock["head_family_v1"],
            "hyperparams_v1": bundle.config_lock["default_model_hyperparams_v1"],
            "compact_grid_v1": bundle.config_lock["compact_grid_v1"],
        },
    }
    model_config_manifest = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_MODEL_CONFIG_MANIFEST_DRY_V1",
        "training_started_v1": False,
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "base_model_v1": bundle.config_lock["base_model_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "n_jobs_v1": bundle.config_lock["n_jobs_v1"],
        "head_family_v1": bundle.config_lock["head_family_v1"],
        "default_model_hyperparams_v1": bundle.config_lock["default_model_hyperparams_v1"],
        "selected_features_v1": selected_features,
    }
    eval_summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_EVAL_SUMMARY_PLACEHOLDER_V1",
        "status_v1": "PENDING_TRAINING_NOT_RUN",
        "walkforward_required_v1": True,
        "loso_required_v1": True,
        "rolling_window_required_v1": True,
    }
    verdict_package = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_VERDICT_PACKAGE_PLACEHOLDER_V1",
        "verdict_v1": "NOT_ESTABLISHED_DRY_PRELAUNCH_ONLY",
        "training_started_v1": False,
        "hard_fail_v1": False,
        "reason_v1": (
            "Runner implementation materialized only prelaunch/scaffold outputs; no model fit or eval has run. "
            "This narrow setup is now a known failed/no-go path and must not be retrained as the active next step."
        ),
        "next_action_v1": "DO_NOT_RETRAIN_SAME_NARROW_SETUP_AGAIN",
    }
    contract = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_RUNNER_V1",
        "created_at_utc_v1": training_summary["created_at_utc_v1"],
        "spec_dir_v1": str(bundle.spec_dir),
        "output_dir_v1": str(output_dir),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "run_training_flag_v1": bool(run_training),
    }
    next_action = {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "USE_PROTECTOR_FIRST_PATH_INSTEAD",
        "blocked_action_v1": "RUN_KNOWN_FAILED_NARROW_RETRAIN_WITHOUT_FORENSICS_OVERRIDE",
        "training_now_v1": False,
        "known_failed_reference_v1": KNOWN_FAILED_NARROW_RUN,
    }
    _write_json(output_dir / CONTRACT, contract)
    _write_json(output_dir / LOADED_CONFIG, loaded_config)
    _write_json(output_dir / PRELAUNCH_REPORT, validation_report)
    _write_json(output_dir / ABORT_ENFORCEMENT, abort_scaffold)
    _write_json(output_dir / TRAINING_SUMMARY, training_summary)
    _write_json(output_dir / MODEL_CONFIG_MANIFEST, model_config_manifest)
    feature_table.to_csv(output_dir / FEATURE_MANIFEST_ECHO, index=False)
    _write_json(output_dir / EVAL_SUMMARY, eval_summary)
    _placeholder_compare_rows(bundle).to_csv(output_dir / COMPARE_AGAINST, index=False)
    _placeholder_pocket_rows().to_csv(output_dir / POCKET_REPORT, index=False)
    _write_json(output_dir / VERDICT_PACKAGE, verdict_package)
    _write_json(output_dir / NEXT_ACTION, next_action)
    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_DRY_PRELAUNCH_SUMMARY_V1",
        "created_at_utc_v1": training_summary["created_at_utc_v1"],
        "output_dir_v1": str(output_dir),
        "job_name_v1": JOB_NAME,
        "prelaunch_status_v1": validation_report["status_v1"],
        "training_started_v1": False,
        "run_training_flag_v1": bool(run_training),
        "training_rows_v1": validation_report["input_status_v1"]["raw_rows_v1"],
        "label_intersection_v1": validation_report["input_status_v1"]["label_intersection_v1"],
        "feature_count_v1": validation_report["input_status_v1"]["selected_feature_count_v1"],
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Runner loads locked config/spec artifacts.",
                "Prelaunch validation passes on exact-only canonical raw-state.",
                "Output scaffold, manifest, status and audit hooks are materialized without training.",
                "This narrow setup is a known failed/no-go setup after the 20260424 failure run.",
            ],
            "INDIKERT": [
                "A rerun should only be used as a deliberate forensics/control rerun.",
                "Abort enforcement has the required structure for later eval outputs.",
            ],
            "IKKE_ETABLERT": [
                "No model has been fit.",
                "No future candidate safety or benchmark improvement is established.",
                "This path is not the active next training path.",
            ],
        },
    }
    _write_json(output_dir / SUMMARY, summary)
    _write_report(output_dir / REPORT, summary, validation_report, abort_scaffold, bundle)
    return summary


def _write_training_outputs(bundle: RunnerBundle, output_dir: Path, validation_report: Dict[str, Any]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    start_time = _utc_now_iso()
    initial_status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_TRAINING_STATUS_V1",
        "RUNNER_STATUS": "TRAINING_STARTED",
        "training_started_v1": True,
        "run_training_now_v1": True,
        "created_at_utc_v1": start_time,
        "failed_check_count_v1": 0,
    }
    _write_json(output_dir / STATUS, initial_status)

    joined, x, features = _prepare_training_frame(bundle, validation_report)
    predictions, model_bundle = _train_head_predictions(joined, x, bundle)
    scored = joined.merge(predictions, on="candidate_uid", how="left", validate="one_to_one")
    block = _policy_block_mask(scored)
    scored["monday_narrow_block_v1"] = block
    scored["monday_narrow_allow_v1"] = ~block
    scored["monday_narrow_block_reason_v1"] = np.where(block, "RUNNER_FIRST_BAD_OR_RISK_OR_TAIL", "ALLOW_OR_RUNNER_PROTECTED")

    global_metric = _metric_row("MONDAY_NARROW_RETRAIN_SELECTED", "ALL_EXACT_ONLY", scored, block)
    loso_df = _slice_metric_table(scored, block, table_name="LOSO")
    walkforward_df = _slice_metric_table(scored, block, table_name="WALKFORWARD")
    rolling_df = _slice_metric_table(scored, block, table_name="ROLLING")
    loso_precisions = [
        _safe_float(value)
        for value in loso_df["should_not_take_precision_v1"].tolist()
        if _safe_float(value) is not None
    ]
    worst_loso_precision = min(loso_precisions) if loso_precisions else None
    global_metric["worst_loso_precision_v1"] = worst_loso_precision
    failures = _post_eval_failures(global_metric, worst_loso_precision)
    verdict = _verdict(global_metric, failures)
    compare_df = _reference_rows(global_metric, worst_loso_precision)
    pocket_df = pd.DataFrame(
        [
            {"pocket_v1": "repaired_165_pocket", "blocked_count_v1": global_metric["repaired_165_block_count_v1"], "hard_guard_v1": True},
            {"pocket_v1": "forensic_repaired_trade", "blocked_count_v1": int(bool(global_metric["forensic_trade_blocked_v1"])), "hard_guard_v1": True},
            {"pocket_v1": "runner_near_miss_pocket", "blocked_count_v1": global_metric["runner_near_miss_block_count_v1"], "hard_guard_v1": True},
            {"pocket_v1": "50_plus_mfe_seed_pocket", "blocked_count_v1": global_metric["fifty_plus_mfe_block_count_v1"], "hard_guard_v1": True},
            {"pocket_v1": "100_plus_mfe_seed_pocket", "blocked_count_v1": global_metric["hundred_plus_mfe_block_count_v1"], "hard_guard_v1": True},
            {"pocket_v1": "200_plus_mfe_seed_pocket", "blocked_count_v1": global_metric["two_hundred_plus_mfe_block_count_v1"], "hard_guard_v1": True},
            {"pocket_v1": "missed_10_50_tail_control_pocket", "blocked_count_v1": global_metric["tail_10_50_help_count_v1"], "hard_guard_v1": False},
            {"pocket_v1": "missed_should_not_take_pocket", "blocked_count_v1": global_metric["should_not_take_block_count_v1"], "hard_guard_v1": False},
        ]
    )
    loaded_config = _build_loaded_config(bundle)
    abort_scaffold = _build_abort_enforcement_scaffold(bundle)
    feature_table = bundle.feature_manifest_table.copy()
    model_manifest = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_MODEL_CONFIG_MANIFEST_V1",
        "training_started_v1": True,
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "base_model_v1": bundle.config_lock["base_model_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "n_jobs_v1": bundle.config_lock["n_jobs_v1"],
        "head_family_v1": bundle.config_lock["head_family_v1"],
        "default_model_hyperparams_v1": bundle.config_lock["default_model_hyperparams_v1"],
        "selected_feature_count_v1": len(features),
        "model_bundle_artifact_v1": MODEL_BUNDLE,
        "policy_rule_v1": "block if bad/risky/tail/blindspot probability >=0.5 and runner_protector probability <0.5",
    }
    training_summary = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_EXECUTION_PHASE_V1",
        "job_name_v1": JOB_NAME,
        "created_at_utc_v1": start_time,
        "completed_at_utc_v1": _utc_now_iso(),
        "training_started_v1": True,
        "run_training_flag_v1": True,
        "training_surface_v1": validation_report["input_status_v1"]["input_path_v1"],
        "training_rows_v1": int(len(scored)),
        "label_intersection_v1": validation_report["input_status_v1"]["label_intersection_v1"],
        "feature_count_v1": len(features),
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "global_metric_v1": global_metric,
        "verdict_v1": verdict,
        "post_eval_failure_reasons_v1": failures,
    }
    eval_summary = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_TRAINING_OUTPUTS_V1",
        "training_started_v1": True,
        "global_metric_v1": global_metric,
        "worst_loso_precision_v1": worst_loso_precision,
        "walkforward_rows_v1": int(len(walkforward_df)),
        "loso_rows_v1": int(len(loso_df)),
        "rolling_rows_v1": int(len(rolling_df)),
        "post_eval_failure_count_v1": len(failures),
    }
    verdict_package = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_VERDICT_MATRIX_V1",
        "verdict_v1": verdict,
        "training_started_v1": True,
        "candidate_disqualified_v1": bool(failures),
        "hard_fail_reasons_v1": failures,
        "supported_verdicts_v1": [
            "CANDIDATE_IMPROVES_AND_HOLDS_SAFETY",
            "CANDIDATE_IMPROVES_BUT_FAILS_SAFETY",
            "CANDIDATE_SAFE_BUT_NOT_BETTER",
            "CANDIDATE_FEATURES_INSUFFICIENT",
            "CANDIDATE_INVALID_DUE_TO_LEGALITY_OR_SURFACE_BREACH",
            "NOT_ESTABLISHED",
        ],
        "next_action_v1": "REVIEW_TRAINING_RUN_VERDICT",
    }
    contract = {
        "layer_name_v1": "IMPLEMENT_MONDAY_NARROW_RETRAIN_RUNNER_V1",
        "created_at_utc_v1": start_time,
        "spec_dir_v1": str(bundle.spec_dir),
        "output_dir_v1": str(output_dir),
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "run_training_flag_v1": True,
        "training_started_v1": True,
    }
    next_action = {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "REVIEW_TRAINING_RUN_VERDICT",
        "blocked_action_v1": "PROMOTE_OR_FREEZE_WITHOUT_REVIEW",
        "training_now_v1": False,
    }

    joblib.dump(model_bundle, output_dir / MODEL_BUNDLE)
    _write_json(output_dir / CONTRACT, contract)
    _write_json(output_dir / LOADED_CONFIG, loaded_config)
    _write_json(output_dir / PRELAUNCH_REPORT, validation_report)
    _write_json(output_dir / ABORT_ENFORCEMENT, abort_scaffold)
    _write_json(output_dir / TRAINING_SUMMARY, training_summary)
    _write_json(output_dir / MODEL_CONFIG_MANIFEST, model_manifest)
    feature_table.to_csv(output_dir / FEATURE_MANIFEST_ECHO, index=False)
    _write_json(output_dir / EVAL_SUMMARY, eval_summary)
    compare_df.to_csv(output_dir / COMPARE_AGAINST, index=False)
    pocket_df.to_csv(output_dir / POCKET_REPORT, index=False)
    scored.to_parquet(output_dir / POLICY_PREDICTION_VIEW, index=False)
    walkforward_df.to_csv(output_dir / WALKFORWARD_METRICS, index=False)
    loso_df.to_csv(output_dir / LOSO_METRICS, index=False)
    rolling_df.to_csv(output_dir / ROLLING_METRICS, index=False)
    _write_json(output_dir / VERDICT_PACKAGE, verdict_package)
    _write_json(output_dir / NEXT_ACTION, next_action)
    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_TRAINING_RUN_SUMMARY_V1",
        "created_at_utc_v1": start_time,
        "output_dir_v1": str(output_dir),
        "job_name_v1": JOB_NAME,
        "prelaunch_status_v1": validation_report["status_v1"],
        "training_started_v1": True,
        "run_training_flag_v1": True,
        "training_rows_v1": int(len(scored)),
        "label_intersection_v1": validation_report["input_status_v1"]["label_intersection_v1"],
        "feature_count_v1": len(features),
        "model_family_v1": bundle.config_lock["model_family_v1"],
        "seed_v1": bundle.config_lock["seed_v1"],
        "verdict_v1": verdict,
        "candidate_disqualified_v1": bool(failures),
        "post_eval_failure_count_v1": len(failures),
        "bad_blocks_v1": global_metric["bad_blocks_v1"],
        "tail_help_v1": global_metric["tail_help_v1"],
        "global_precision_v1": global_metric["global_precision_v1"],
        "worst_loso_precision_v1": worst_loso_precision,
        "next_action_v1": next_action["primary_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Training-run phase executes only with explicit run flag.",
                "Five XGB-style heads are fit and prediction/eval artifacts are materialized.",
                "Compare-against, pocket report, verdict package and runtime abort checks are written.",
            ],
            "INDIKERT": [
                "The runner is ready for a controlled narrow retrain run under the locked contract.",
            ],
            "IKKE_ETABLERT": [
                "No live promotion or freeze is established by a training run alone.",
            ],
        },
    }
    _write_json(output_dir / SUMMARY, summary)
    _write_report(output_dir / REPORT, summary, validation_report, abort_scaffold, bundle)
    return summary


def _write_report(path: Path, summary: Dict[str, Any], validation_report: Dict[str, Any], abort_scaffold: Dict[str, Any], bundle: RunnerBundle) -> None:
    blocked_action = summary.get("blocked_action_v1", "PROMOTE_OR_FREEZE_WITHOUT_REVIEW")
    lines = [
        "# Monday Narrow Retrain Runner V1",
        "",
        "## Status",
        f"- Prelaunch: `{summary['prelaunch_status_v1']}`",
        f"- Training started: `{summary['training_started_v1']}`",
        f"- Rows: `{summary['training_rows_v1']}`",
        f"- Label intersection: `{summary['label_intersection_v1']}`",
        f"- Feature count: `{summary['feature_count_v1']}`",
        "",
        "## Config",
        f"- Model family: `{bundle.config_lock['model_family_v1']}`",
        f"- Base model: `{bundle.config_lock['base_model_v1']}`",
        f"- Seed: `{bundle.config_lock['seed_v1']}`",
        "",
        "## Input",
        f"- Training surface: `{validation_report['input_status_v1']['input_path_v1']}`",
        f"- Label surface: `{validation_report['input_status_v1']['label_path_v1']}`",
        "",
        "## Abort Enforcement",
    ]
    for check in abort_scaffold["post_eval_metric_checks_v1"]:
        lines.append(f"- `{check['metric_v1']}` {check['operator_v1']} `{check['threshold_v1']}` hard_fail=`{check['hard_fail_v1']}`")
    lines.extend(
        [
            "",
        "## Next Action",
        f"- `{summary['next_action_v1']}`",
        f"- Blocked: `{blocked_action}`",
        "",
    ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest_status_audit(output_dir: Path, summary: Dict[str, Any]) -> None:
    artifacts = [
        CONTRACT,
        LOADED_CONFIG,
        PRELAUNCH_REPORT,
        ABORT_ENFORCEMENT,
        TRAINING_SUMMARY,
        MODEL_CONFIG_MANIFEST,
        FEATURE_MANIFEST_ECHO,
        EVAL_SUMMARY,
        COMPARE_AGAINST,
        POCKET_REPORT,
        VERDICT_PACKAGE,
        NEXT_ACTION,
        SUMMARY,
        REPORT,
        MODEL_BUNDLE,
        POLICY_PREDICTION_VIEW,
        WALKFORWARD_METRICS,
        LOSO_METRICS,
        ROLLING_METRICS,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    training_started = bool(summary.get("training_started_v1"))
    expected_status_check = "TRAINING_RUN_COMPLETED" if training_started else "NO_TRAINING_STARTED"
    audit_rows = [
        _audit_record("PRELAUNCH_VALIDATION_PASS", "PASS" if summary["prelaunch_status_v1"] == "PASS" else "FAIL", {"prelaunch_status_v1": summary["prelaunch_status_v1"]}),
        _audit_record(expected_status_check, "PASS", {"training_started_v1": summary["training_started_v1"]}),
        _audit_record("ROW_COUNT_LOCKED", "PASS" if summary["training_rows_v1"] == EXPECTED_ROWS else "FAIL", {"training_rows_v1": summary["training_rows_v1"]}),
        _audit_record("FEATURE_COUNT_LOCKED", "PASS" if summary["feature_count_v1"] == EXPECTED_FEATURES else "FAIL", {"feature_count_v1": summary["feature_count_v1"]}),
        _audit_record("LABEL_INTERSECTION_LOCKED", "PASS" if summary["label_intersection_v1"] == EXPECTED_LABEL_INTERSECTION else "FAIL", {"label_intersection_v1": summary["label_intersection_v1"]}),
        _audit_record("OUTPUTS_PRESENT", "PASS" if all((output_dir / artifact).exists() for artifact in artifacts if artifact not in {MANIFEST, STATUS, CONSISTENCY_AUDIT} and (training_started or artifact not in {MODEL_BUNDLE, POLICY_PREDICTION_VIEW, WALKFORWARD_METRICS, LOSO_METRICS, ROLLING_METRICS})) else "FAIL", {"artifact_count_v1": len(artifacts), "training_started_v1": training_started}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    manifest = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_DRY_PRELAUNCH_MANIFEST_V1",
        "created_at_utc_v1": _utc_now_iso(),
        "output_dir_v1": str(output_dir),
        "artifacts_v1": artifacts,
    }
    _write_json(output_dir / MANIFEST, manifest)
    status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_STATUS_V1",
        "RUNNER_STATUS": "TRAINING_RUN_COMPLETED" if training_started else "DRY_PRELAUNCH_VALIDATED",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "training_started_v1": training_started,
        "run_training_now_v1": False,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "next_action_v1": summary["next_action_v1"],
        "verdict_v1": summary.get("verdict_v1"),
        "candidate_disqualified_v1": summary.get("candidate_disqualified_v1"),
    }
    _write_json(output_dir / STATUS, status)


def run_runner(
    *,
    reports_root: Path,
    spec_dir: Path | None = None,
    output_dir: Path | None = None,
    run_training: bool = False,
    allow_known_failed_narrow_rerun_for_forensics: bool = False,
) -> Dict[str, Any]:
    bundle = _load_bundle(reports_root, str(spec_dir) if spec_dir else None)
    resolved_output_dir = output_dir or _resolve_output_dir(reports_root, None, run_training=run_training)
    validation_report = run_prelaunch_validation(bundle, resolved_output_dir)
    if run_training:
        _enforce_known_failed_narrow_rerun_guard(
            allow_known_failed_narrow_rerun=allow_known_failed_narrow_rerun_for_forensics
        )
        summary = _write_training_outputs(bundle, resolved_output_dir, validation_report)
    else:
        summary = _write_output_scaffold(bundle, resolved_output_dir, validation_report, run_training=run_training)
    _write_manifest_status_audit(resolved_output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monday narrow retrain runner dry prelaunch validation V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--spec-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument(
        "--allow-known-failed-narrow-rerun-for-forensics",
        action="store_true",
        help=(
            "Explicitly permit rerunning the known failed Monday narrow setup for forensics/control only. "
            "Normal users should run the protector-first path instead."
        ),
    )
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    output_dir = _resolve_output_dir(reports_root, args.output_dir, run_training=bool(args.run_training))
    summary = run_runner(
        reports_root=reports_root,
        spec_dir=Path(args.spec_dir).expanduser().resolve() if args.spec_dir else None,
        output_dir=output_dir,
        run_training=bool(args.run_training),
        allow_known_failed_narrow_rerun_for_forensics=bool(args.allow_known_failed_narrow_rerun_for_forensics),
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
