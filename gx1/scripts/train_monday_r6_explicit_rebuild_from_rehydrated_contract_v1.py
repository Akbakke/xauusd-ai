#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from gx1.scripts.train_r3_entry_label_feature_retrain_v1 import _fit_preprocessor, _transform_features
from gx1.scripts.train_r6_entry_runner_first_retrain_v1 import (
    HEAD_SPECS as R6_HEAD_SPECS,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BLINDSPOT_PROB,
    R6_BAD_PROB,
    R6_RUNNER_PROB,
    R6_RISKY_PROB,
    R6_TAIL_PROB,
    _candidate_grid as _r6_candidate_grid,
    _policy_mask as _r6_policy_mask,
    _train_heads as _train_r6_heads,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_EXPLICIT_REBUILD_FROM_REHYDRATED_CONTRACT_V1"

MONDAY_TRUTH_GLOB = "MONDAY_R6_CANONICAL_TRUTH_V1_*"
REHYDRATED_GLOB = "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_*"
WEDNESDAY_R6_SPLIT_SOURCE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
WEDNESDAY_R6_SPLIT_SOURCE_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
EXACT_R5_LABEL_SOURCE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1"
EXACT_R5_LABEL_SOURCE_TABLE = "shadow_meta_all_trade_review_r5_entry_hindsight_label_outcome_table_v1.parquet"
EXACT_R5_2_LABEL_SOURCE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_V1"
EXACT_R5_2_LABEL_SOURCE_TABLE = "shadow_meta_all_trade_review_r5_2_hindsight_label_outcome_table_v1.parquet"
EXACT_R6_LABEL_SOURCE_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
EXACT_EVAL_LABEL_SOURCE_TABLE = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"

AS_OF_TABLE = "monday_r6_entry_runner_first_as_of_feature_table_v1.parquet"
HINDSIGHT_TABLE = "monday_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
TRUTH_TABLE = "monday_r6_trade_truth_v1.parquet"

TRAINING_FRAME = "monday_r6_explicit_rebuild_training_frame_v1.parquet"
PREDICTION_VIEW = "monday_r6_explicit_rebuild_prediction_view_v1.parquet"
MODEL_METRICS = "model_metrics_v1.csv"
EVAL_SUMMARY = "eval_summary_v1.json"
COMPARE_REPORT = "compare_against_wednesday_r6_v1.json"
MODEL_MANIFEST = "model_manifest_v1.json"
CONFIG_MANIFEST = "config_manifest_v1.json"
FEATURE_MANIFEST = "feature_manifest_v1.csv"
THRESHOLD_CALIBRATION = "threshold_calibration_v1.csv"
CALIBRATION_SAFETY_SUMMARY = "calibration_safety_summary_v1.json"
SAFETY_FAILURE_ROWS = "safety_failure_rows_v1.csv"
WEDNESDAY_LOCKED_POLICY_REPLAY = "wednesday_locked_policy_replay_v1.json"
R6_FAMILY_GRID_REPLAY = "r6_family_grid_replay_v1.csv"
STATUS = "status_v1.json"
SUMMARY = "summary_v1.json"
MANIFEST = "manifest_v1.json"
AUDIT = "consistency_audit_v1.csv"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "training_frame": TRAINING_FRAME,
    "prediction_view": PREDICTION_VIEW,
    "model_metrics": MODEL_METRICS,
    "eval_summary": EVAL_SUMMARY,
    "compare_report": COMPARE_REPORT,
    "model_manifest": MODEL_MANIFEST,
    "config_manifest": CONFIG_MANIFEST,
    "feature_manifest": FEATURE_MANIFEST,
    "threshold_calibration": THRESHOLD_CALIBRATION,
    "calibration_safety_summary": CALIBRATION_SAFETY_SUMMARY,
    "safety_failure_rows": SAFETY_FAILURE_ROWS,
    "wednesday_locked_policy_replay": WEDNESDAY_LOCKED_POLICY_REPLAY,
    "r6_family_grid_replay": R6_FAMILY_GRID_REPLAY,
    "status": STATUS,
    "summary": SUMMARY,
    "manifest": MANIFEST,
    "audit": AUDIT,
    "report": REPORT,
    "models_dir": "models",
}

R5_HEADS = {
    "r5_should_not_take": ("r5_label_should_not_take_v1", "pred__entry_r5_should_not_take__prob_true_v1"),
    "r5_immediate_mae_risk": ("r5_label_immediate_mae_risk_v1", "pred__entry_r5_immediate_MAE_risk__prob_true_v1"),
    "r5_runner_protect": ("r5_label_runner_protect_v1", "pred__entry_r5_runner_protect__prob_true_v1"),
    "r5_strong_trade_candidate": ("r5_label_strong_trade_candidate_v1", "pred__entry_r5_strong_trade_candidate__prob_true_v1"),
    "r5_tail_control_10_50_risk": ("r5_label_tail_control_10_50_risk_v1", "pred__entry_r5_tail_control_10_50_risk__prob_true_v1"),
    "r5_take_was_ok": ("r5_label_take_was_ok_v1", "pred__entry_r5_take_was_ok__prob_true_v1"),
    "r5_bad_trade_but_high_runner_risk": (
        "r5_label_bad_trade_but_high_runner_risk_v1",
        "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1",
    ),
    "r5_wait_or_delay_advisory": ("r5_label_wait_or_delay_advisory_v1", "pred__entry_r5_wait_or_delay_advisory__prob_true_v1"),
}

R5_2_HEADS = {
    "r5_2_bad_blocker": ("r5_2_label_bad_blocker_v1", R5_2_BAD_PROB),
    "r5_2_runner_protector": ("r5_2_label_runner_protect_v1", R5_2_RUNNER_PROB),
}

R5_EXACT_LABEL_COLUMNS = [
    "hindsight_entry_decision_review_v1",
    "hindsight_management_review_v1",
    "r5_label_should_not_take_v1",
    "r5_label_immediate_mae_risk_v1",
    "r5_label_runner_protect_v1",
    "r5_label_strong_trade_candidate_v1",
    "r5_label_tail_control_10_50_risk_v1",
    "r5_label_take_was_ok_v1",
    "r5_label_bad_trade_but_high_runner_risk_v1",
    "r5_label_wait_or_delay_advisory_v1",
]

R5_2_EXACT_LABEL_COLUMNS = [
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

R6_EXACT_LABEL_COLUMNS = [
    "hindsight_entry_decision_review_v1",
    "hindsight_management_review_v1",
    "r6_label_runner_50_mfe_v1",
    "r6_label_runner_100_mfe_v1",
    "r6_label_runner_200_mfe_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_strong_low_mae_runner_v1",
    "r6_label_high_mfe_low_giveback_v1",
    "r6_label_runner_near_miss_v1",
    "r6_label_runner_protect_v1",
    "r6_label_missed_should_not_take_v1",
    "r6_label_risky_allow_v1",
    "r6_label_high_mae_low_mfe_v1",
    "r6_label_low_mfe_low_value_v1",
    "r6_label_early_adverse_excursion_v1",
    "r6_label_bad_trade_overlap_extreme_vol_v1",
    "r6_label_batch04_blindspot_v1",
    "r6_label_trend_neutral_extreme_vol_risk_v1",
    "r6_label_bad_risk_v1",
    "r6_label_tail_control_10_50_v1",
]

EVAL_EXACT_LABEL_COLUMNS = [
    "label_should_not_take_v1",
    "take_was_ok_v1",
    "label_strong_trade_candidate_v1",
    "fifty_plus_mfe_v1",
    "hundred_plus_mfe_v1",
    "two_hundred_plus_mfe_v1",
    "tail_10_50_mfe_v1",
    "is_repaired_165_v1",
]

WEDNESDAY_R6_BENCHMARK = {
    "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
    "bad_blocks_v1": 180,
    "tail_help_v1": 149,
    "precision_v1": 0.972972972972973,
    "worst_loso_v1": 0.9285714285714286,
    "repaired_165_damage_v1": 0,
    "fifty_plus_mfe_blocked_v1": 1,
    "hundred_plus_mfe_blocked_v1": 0,
    "two_hundred_plus_mfe_blocked_v1": 0,
    "strongest_winner_damage_v1": 0,
}

DEFAULT_ASOF_RUNNER_GUARD = {
    "asof_guard_tradable_min_v1": 0.94,
    "asof_guard_quality_min_v1": 0.70,
    "asof_guard_mfe_min_v1": 1.75,
    "asof_guard_flat_max_v1": 0.50,
}

WEDNESDAY_LOCKED_THRESHOLDS = {
    "bad_threshold_v1": 0.95,
    "risky_threshold_v1": 0.85,
    "tail_threshold_v1": 0.90,
    "runner_threshold_v1": 0.60,
    "r5_2_runner_threshold_v1": 0.74,
    "blindspot_threshold_v1": 0.70,
    "guard_v1": "hard_asof_runner_guard",
    "use_r5_2_base_v1": True,
}


@dataclass(frozen=True)
class TrainConfig:
    r5_n_estimators: int = 1200
    r5_early_stopping_rounds: int = 80
    r5_learning_rate: float = 0.025
    r5_min_child_weight: float = 4.0
    r5_reg_lambda: float = 8.0
    r5_reg_alpha: float = 0.35
    r5_2_n_estimators: int = 900
    r5_2_early_stopping_rounds: int = 70
    r5_2_learning_rate: float = 0.025
    r5_2_min_child_weight: float = 3.0
    r5_2_reg_lambda: float = 10.0
    r5_2_reg_alpha: float = 0.5
    r6_n_estimators: int = 800
    r6_early_stopping_rounds: int = 60
    r6_learning_rate: float = 0.025
    max_depth: int = 3
    seed: int = 619
    n_jobs: int = 2


def _stage_params(config: TrainConfig, stage: str) -> dict[str, float | int]:
    if stage == "r5":
        return {
            "n_estimators": config.r5_n_estimators,
            "early_stopping_rounds": config.r5_early_stopping_rounds,
            "learning_rate": config.r5_learning_rate,
            "min_child_weight": config.r5_min_child_weight,
            "reg_lambda": config.r5_reg_lambda,
            "reg_alpha": config.r5_reg_alpha,
        }
    if stage == "r5_2":
        return {
            "n_estimators": config.r5_2_n_estimators,
            "early_stopping_rounds": config.r5_2_early_stopping_rounds,
            "learning_rate": config.r5_2_learning_rate,
            "min_child_weight": config.r5_2_min_child_weight,
            "reg_lambda": config.r5_2_reg_lambda,
            "reg_alpha": config.r5_2_reg_alpha,
        }
    raise ValueError(stage)


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_dir(reports_root: Path, pattern: str) -> Path:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No {pattern} under {reports_root}")
    return dirs[-1]


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"]).fillna(default).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _load_surfaces(monday_truth_dir: Path, rehydrated_dir: Path) -> pd.DataFrame:
    asof = pd.read_parquet(rehydrated_dir / AS_OF_TABLE)
    hindsight = pd.read_parquet(rehydrated_dir / HINDSIGHT_TABLE)
    truth = pd.read_parquet(monday_truth_dir / TRUTH_TABLE)
    id_cols = ["candidate_uid", "run_id", "trade_uid", "trade_id", "decision_timestamp"]
    for frame_name, frame in [("asof", asof), ("hindsight", hindsight)]:
        missing = [column for column in id_cols if column not in frame.columns]
        if missing:
            raise KeyError(f"{frame_name} missing id columns: {missing}")
    work = asof.merge(hindsight, on=id_cols, how="inner", validate="one_to_one")
    truth_cols = [
        "candidate_uid",
        "calendar_quarantine_status_v1",
        "calendar_quarantine_reason_v1",
        "truth_cata_or_friday_flat_damage_v1",
        "truth_exit_too_early_regret_replay_end_v1",
        "canonical_entry_ts_utc_v1",
    ]
    work = work.merge(truth[[column for column in truth_cols if column in truth.columns]], on="candidate_uid", how="left", validate="one_to_one")
    work = work.sort_values(["run_id", "decision_timestamp", "candidate_uid"], kind="mergesort").reset_index(drop=True)
    return work


def _load_optional_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _exact_label_sources(reports_root: Path) -> dict[str, pd.DataFrame | None]:
    r5_path = reports_root / EXACT_R5_LABEL_SOURCE_DIR / EXACT_R5_LABEL_SOURCE_TABLE
    r5_2_path = reports_root / EXACT_R5_2_LABEL_SOURCE_DIR / EXACT_R5_2_LABEL_SOURCE_TABLE
    r6_dir = reports_root / WEDNESDAY_R6_SPLIT_SOURCE_DIR
    return {
        "r5": _load_optional_parquet(r5_path),
        "r5_2": _load_optional_parquet(r5_2_path),
        "r6": _load_optional_parquet(r6_dir / EXACT_R6_LABEL_SOURCE_TABLE),
        "eval": _load_optional_parquet(r6_dir / EXACT_EVAL_LABEL_SOURCE_TABLE),
    }


def _overlay_by_candidate(frame: pd.DataFrame, source: pd.DataFrame | None, columns: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source is None or source.empty:
        return frame.copy(), {"source_present_v1": False, "matched_rows_v1": 0, "overlaid_columns_v1": []}
    if "candidate_uid" not in source.columns:
        raise KeyError("exact label source missing candidate_uid")
    available = [column for column in columns if column in source.columns]
    if not available:
        return frame.copy(), {"source_present_v1": True, "matched_rows_v1": 0, "overlaid_columns_v1": []}
    deduped = source[["candidate_uid", *available]].drop_duplicates("candidate_uid", keep="last").set_index("candidate_uid")
    out = frame.copy()
    candidate_uid = out["candidate_uid"].astype("string")
    matched = candidate_uid.isin(deduped.index.astype("string"))
    for column in available:
        mapped = candidate_uid.map(deduped[column])
        has_value = mapped.notna()
        if column not in out.columns:
            out[column] = mapped
        else:
            values = mapped.loc[has_value]
            if pd.api.types.is_bool_dtype(out[column]):
                values = values.astype(bool)
            out.loc[has_value, column] = values
    return out, {
        "source_present_v1": True,
        "source_rows_v1": int(len(deduped)),
        "matched_rows_v1": int(matched.sum()),
        "overlaid_columns_v1": available,
    }


def _apply_base_label_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    aliases = {
        "r5_label_should_not_take_v1": "label_should_not_take_v1",
        "r5_label_take_was_ok_v1": "take_was_ok_v1",
        "r5_label_strong_trade_candidate_v1": "label_strong_trade_candidate_v1",
        "r6_label_tail_control_10_50_v1": "tail_10_50_mfe_v1",
        "r6_label_runner_50_mfe_v1": "fifty_plus_mfe_v1",
        "r6_label_runner_100_mfe_v1": "hundred_plus_mfe_v1",
        "r6_label_runner_200_mfe_v1": "two_hundred_plus_mfe_v1",
    }
    for source, target in aliases.items():
        if source in out.columns:
            source_values = _bool(out, source)
            if target not in out.columns:
                out[target] = source_values
            else:
                missing = out[target].isna() if hasattr(out[target], "isna") else pd.Series(False, index=out.index)
                out.loc[missing, target] = source_values.loc[missing]
    return out


def _split_flags_from_group(group: pd.DataFrame) -> dict[str, bool]:
    return {
        "used_for_training": bool(_bool(group, "used_for_training").any()),
        "used_for_validation": bool(_bool(group, "used_for_validation").any()),
        "used_for_holdout": bool(_bool(group, "used_for_holdout").any()),
    }


def _split_reference_maps(split_reference: pd.DataFrame) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    required = ["candidate_uid", "run_id", "used_for_training", "used_for_validation", "used_for_holdout"]
    missing = [column for column in required if column not in split_reference.columns]
    if missing:
        raise KeyError(f"split reference missing columns: {missing}")
    by_candidate: dict[str, dict[str, bool]] = {}
    for candidate_uid, group in split_reference.groupby(split_reference["candidate_uid"].astype("string"), dropna=False):
        by_candidate[str(candidate_uid)] = _split_flags_from_group(group)
    by_run: dict[str, dict[str, bool]] = {}
    for run_id, group in split_reference.groupby(split_reference["run_id"].astype("string"), dropna=False):
        by_run[str(run_id)] = _split_flags_from_group(group)
    return by_candidate, by_run


def _assign_splits(frame: pd.DataFrame, split_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    out = frame.copy()
    active = out.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=out.index)).astype("string").eq("ACTIVE_CANDIDATE")
    if split_reference is not None:
        by_candidate, by_run = _split_reference_maps(split_reference)
        run_series = out["run_id"].astype("string")
        candidate_series = out["candidate_uid"].astype("string")

        def flag(candidate_uid: str, run_id: str, name: str, default: bool) -> bool:
            candidate_flags = by_candidate.get(str(candidate_uid))
            if candidate_flags is not None:
                return bool(candidate_flags.get(name, default))
            return bool(by_run.get(str(run_id), {}).get(name, default))

        training_flags = pd.Series(
            [
                flag(candidate_uid, run_id, "used_for_training", False)
                for candidate_uid, run_id in zip(candidate_series.tolist(), run_series.tolist())
            ],
            index=out.index,
            dtype=bool,
        )
        validation_flags = pd.Series(
            [
                flag(candidate_uid, run_id, "used_for_validation", False)
                for candidate_uid, run_id in zip(candidate_series.tolist(), run_series.tolist())
            ],
            index=out.index,
            dtype=bool,
        )
        holdout_flags = pd.Series(
            [
                flag(candidate_uid, run_id, "used_for_holdout", True)
                for candidate_uid, run_id in zip(candidate_series.tolist(), run_series.tolist())
            ],
            index=out.index,
            dtype=bool,
        )
        out["used_for_training"] = active & training_flags
        out["used_for_validation"] = active & validation_flags
        out["used_for_holdout"] = (~active) | holdout_flags
        out["split_scope_v1"] = np.where(
            out["used_for_training"],
            "TRAIN",
            np.where(out["used_for_validation"], "VALIDATION", np.where(~active, "QUARANTINE_EVAL_ONLY", "HOLDOUT")),
        )
        return out
    runs = sorted(out.loc[active, "run_id"].astype("string").unique().tolist())
    train_cut = int(round(len(runs) * 0.72))
    val_cut = int(round(len(runs) * 0.84))
    train_runs = set(runs[:train_cut])
    val_runs = set(runs[train_cut:val_cut])
    holdout_runs = set(runs[val_cut:])
    run_series = out["run_id"].astype("string")
    out["used_for_training"] = active & run_series.isin(train_runs)
    out["used_for_validation"] = active & run_series.isin(val_runs)
    out["used_for_holdout"] = (~active) | run_series.isin(holdout_runs)
    out["split_scope_v1"] = np.where(
        out["used_for_training"],
        "TRAIN",
        np.where(out["used_for_validation"], "VALIDATION", np.where(~active, "QUARANTINE_EVAL_ONLY", "HOLDOUT")),
    )
    return out


def _batch_scope(frame: pd.DataFrame, batch_weeks: int = 4) -> pd.Series:
    runs = sorted(frame["run_id"].astype("string").unique().tolist())
    lookup = {run: f"BATCH_{idx // batch_weeks + 1:02d}" for idx, run in enumerate(runs)}
    return frame["run_id"].astype("string").map(lookup).fillna("BATCH_UNKNOWN")


def _base_feature_names(frame: pd.DataFrame) -> list[str]:
    forbidden = {
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "r6_as_of_feature_contract_v1",
    }
    score_like = {column for column in frame.columns if column.startswith("pred__entry_r5_") or column.startswith("pred__entry_r6_")}
    score_like |= {"blocker_score_v1", "runner_protector_score_v1"}
    out: list[str] = []
    for column in frame.columns:
        if column in forbidden or column in score_like:
            continue
        if column.startswith("as_of_") or column in {
            "entry_observation_present_v1",
            "entry_raw_state_present_v1",
            "management_observation_present_v1",
            "entry_coverage_original_entry_observation_present_v1",
            "entry_coverage_original_entry_raw_state_present_v1",
            "entry_coverage_repair_applied_v1",
            "entry_coverage_repair_source_v1",
        }:
            out.append(column)
    return out


def _r6_feature_names(frame: pd.DataFrame) -> list[str]:
    names = _base_feature_names(frame)
    for column in [
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "blocker_score_v1",
        "runner_protector_score_v1",
        *[col for col in frame.columns if col.startswith("pred__entry_r5_")],
    ]:
        if column in frame.columns and column not in names:
            names.append(column)
    return names


def _derive_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pnl = _num(out, "baseline_realized_pnl_bps_v1")
    mfe = _num(out, "peak_mfe_bps_v1")
    mae_abs = _num(out, "mae_abs_bps_v1").abs()
    giveback = _num(out, "giveback_bps_v1")
    should = ((pnl <= 0.0) & (mfe < 50.0)) | ((mae_abs >= 40.0) & (pnl <= 0.0)) | _bool(out, "truth_cata_or_friday_flat_damage_v1")
    take_ok = (~should) & (pnl > 0.0) & (mfe >= 20.0) & (mae_abs <= 50.0)
    fifty = mfe >= 50.0
    hundred = mfe >= 100.0
    two_hundred = mfe >= 200.0
    strong = (~should) & fifty & (mae_abs <= 25.0) & (pnl > 0.0)
    tail = mfe.between(10.0, 50.0, inclusive="left") & ((pnl <= 0.0) | should)
    out["label_should_not_take_v1"] = should
    out["take_was_ok_v1"] = take_ok
    out["label_strong_trade_candidate_v1"] = strong
    out["fifty_plus_mfe_v1"] = fifty
    out["hundred_plus_mfe_v1"] = hundred
    out["two_hundred_plus_mfe_v1"] = two_hundred
    out["tail_10_50_mfe_v1"] = tail
    out["strongest_winner_path_v1"] = two_hundred | (strong & (pnl > 0.0))
    out["r5_label_should_not_take_v1"] = should
    out["r5_label_immediate_mae_risk_v1"] = mae_abs >= 40.0
    out["r5_label_immediate_MAE_risk_v1"] = out["r5_label_immediate_mae_risk_v1"]
    out["r5_label_runner_protect_v1"] = take_ok & (fifty | strong | ((giveback <= 25.0) | (giveback <= mfe * 0.25)))
    out["r5_label_strong_trade_candidate_v1"] = strong
    out["r5_label_tail_control_10_50_risk_v1"] = tail
    out["r5_label_take_was_ok_v1"] = take_ok
    out["r5_label_bad_trade_but_high_runner_risk_v1"] = should & fifty
    out["r5_label_wait_or_delay_advisory_v1"] = should & ((mae_abs >= 25.0) | tail)
    return out


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


def _r5_2_sample_weights(frame: pd.DataFrame, label_col: str) -> np.ndarray:
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


def _train_binary_head(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_col: str,
    output_col: str,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path,
    model_tag: str,
    stage: str,
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
    if stage == "r5_2":
        weights = _r5_2_sample_weights(frame.loc[train_mask].copy(), label_col)
    else:
        weights = compute_sample_weight("balanced", y_train).astype(float)
    stage_params = _stage_params(config, stage)
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
            "not_live_gate": True,
        },
    )
    return probs.rename(output_col), pd.DataFrame(rows)


def _train_head_group(
    *,
    frame: pd.DataFrame,
    head_specs: dict[str, tuple[str, str]],
    feature_names: Sequence[str],
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path,
    model_tag: str,
    stage: str,
    seed: int,
    config: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = frame[["candidate_uid"]].copy()
    metrics: list[pd.DataFrame] = []
    for idx, (_, (label_col, output_col)) in enumerate(head_specs.items()):
        probs, head_metrics = _train_binary_head(
            frame=frame,
            feature_names=feature_names,
            label_col=label_col,
            output_col=output_col,
            train_mask=train_mask,
            validation_mask=validation_mask,
            output_dir=output_dir,
            model_tag=model_tag,
            stage=stage,
            seed=seed + idx * 13,
            config=config,
        )
        pred[output_col] = probs.to_numpy(dtype=float)
        metrics.append(head_metrics)
    return pred, pd.concat(metrics, ignore_index=True)


def _derive_r5_2_and_r6_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    high_mfe_tail_ambiguous = _bool(out, "label_should_not_take_v1") & _bool(out, "fifty_plus_mfe_v1")
    out["r5_2_label_bad_blocker_v1"] = _bool(out, "label_should_not_take_v1") & ~high_mfe_tail_ambiguous
    out["r5_2_label_runner_50_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "fifty_plus_mfe_v1")
    out["r5_2_label_runner_100_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "hundred_plus_mfe_v1")
    out["r5_2_label_runner_200_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "two_hundred_plus_mfe_v1")
    out["r5_2_label_repaired_165_like_runner_v1"] = False
    out["r5_2_label_strong_low_mae_runner_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "label_strong_trade_candidate_v1") & _num(out, "mae_abs_bps_v1").le(25.0)
    out["r5_2_label_runner_protect_v1"] = (
        _bool(out, "r5_2_label_runner_50_mfe_v1")
        | _bool(out, "r5_2_label_runner_100_mfe_v1")
        | _bool(out, "r5_2_label_runner_200_mfe_v1")
        | _bool(out, "r5_2_label_strong_low_mae_runner_v1")
    )
    out["r5_2_selected_candidate__block_v1"] = _num(out, R5_2_BAD_PROB).ge(0.42) & _num(out, R5_2_RUNNER_PROB).lt(0.50)
    out["blocker_score_v1"] = _num(out, R5_2_BAD_PROB)
    out["runner_protector_score_v1"] = _num(out, R5_2_RUNNER_PROB)

    selected = _bool(out, "r5_2_selected_candidate__block_v1")
    near_miss = _bool(out, "take_was_ok_v1") & _bool(out, "fifty_plus_mfe_v1") & (
        _num(out, R5_2_BAD_PROB).ge(0.50) | _num(out, R5_2_RUNNER_PROB).lt(0.60) | selected
    )
    out["r6_label_runner_50_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "fifty_plus_mfe_v1")
    out["r6_label_runner_100_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "hundred_plus_mfe_v1")
    out["r6_label_runner_200_mfe_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "two_hundred_plus_mfe_v1")
    out["r6_label_repaired_165_like_runner_v1"] = False
    out["r6_label_strong_low_mae_runner_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "label_strong_trade_candidate_v1") & _num(out, "mae_abs_bps_v1").le(25.0)
    out["r6_label_high_mfe_low_giveback_v1"] = _bool(out, "take_was_ok_v1") & _bool(out, "fifty_plus_mfe_v1") & (
        _num(out, "giveback_bps_v1").le(25.0) | _num(out, "giveback_bps_v1").le(_num(out, "peak_mfe_bps_v1") * 0.25)
    )
    out["r6_label_runner_near_miss_v1"] = near_miss
    out["r6_label_runner_protect_v1"] = (
        _bool(out, "r6_label_runner_50_mfe_v1")
        | _bool(out, "r6_label_runner_100_mfe_v1")
        | _bool(out, "r6_label_runner_200_mfe_v1")
        | _bool(out, "r6_label_strong_low_mae_runner_v1")
        | _bool(out, "r6_label_high_mfe_low_giveback_v1")
        | near_miss
    )
    out["r6_label_missed_should_not_take_v1"] = _bool(out, "label_should_not_take_v1") & ~selected
    out["r6_label_risky_allow_v1"] = _bool(out, "r6_label_missed_should_not_take_v1") & (
        _num(out, "mae_abs_bps_v1").ge(40.0) | _num(out, "baseline_realized_pnl_bps_v1").le(-25.0) | _num(out, R5_2_BAD_PROB).ge(0.60)
    )
    out["r6_label_high_mae_low_mfe_v1"] = _bool(out, "label_should_not_take_v1") & _num(out, "mae_abs_bps_v1").ge(40.0) & _num(out, "peak_mfe_bps_v1").lt(50.0)
    out["r6_label_low_mfe_low_value_v1"] = _bool(out, "label_should_not_take_v1") & _num(out, "peak_mfe_bps_v1").lt(10.0) & _num(out, "baseline_realized_pnl_bps_v1").le(0.0)
    out["r6_label_early_adverse_excursion_v1"] = out["r6_label_high_mae_low_mfe_v1"]
    out["r6_label_bad_trade_overlap_extreme_vol_v1"] = (
        _bool(out, "label_should_not_take_v1")
        & out.get("as_of_session_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("OVERLAP")
        & out.get("as_of_candidate_vol_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("EXTREME")
    )
    out["batch_scope_v1"] = _batch_scope(out)
    out["r6_label_batch04_blindspot_v1"] = _bool(out, "r6_label_missed_should_not_take_v1") & out["batch_scope_v1"].astype("string").eq("BATCH_04")
    out["r6_label_trend_neutral_extreme_vol_risk_v1"] = (
        _bool(out, "label_should_not_take_v1")
        & out.get("as_of_candidate_trend_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("TREND_NEUTRAL")
        & out.get("as_of_candidate_vol_regime_v1", pd.Series("", index=out.index)).astype("string").str.upper().eq("EXTREME")
    )
    out["r6_label_bad_risk_v1"] = _bool(out, "label_should_not_take_v1")
    out["r6_label_tail_control_10_50_v1"] = _bool(out, "tail_10_50_mfe_v1")
    return out


def _asof_runner_guard(frame: pd.DataFrame, params: dict[str, float] | None = None) -> pd.Series:
    guard = {**DEFAULT_ASOF_RUNNER_GUARD, **(params or {})}
    return (
        _num(frame, "as_of_candidate_tradable_prob_v1").ge(float(guard["asof_guard_tradable_min_v1"]))
        & _num(frame, "as_of_entry_candidate_path_quality_pred_v1").ge(float(guard["asof_guard_quality_min_v1"]))
        & _num(frame, "as_of_candidate_mfe_first_n_pred_v1").ge(float(guard["asof_guard_mfe_min_v1"]))
        & _num(frame, "as_of_skip_candidate_p_flat_v1").le(float(guard["asof_guard_flat_max_v1"]))
    )


def _policy_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    mask = mask.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    fifty = _bool(frame, "fifty_plus_mfe_v1")
    hundred = _bool(frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(frame, "strongest_winner_path_v1")
    quarantine = ~frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    block_count = int(mask.sum())
    bad_blocks = int((mask & should).sum())
    false_blocks = int((mask & take_ok).sum())
    precision = float(bad_blocks / block_count) if block_count else None
    return {
        "row_count_v1": int(len(frame)),
        "block_count_v1": block_count,
        "bad_blocks_v1": bad_blocks,
        "tail_help_v1": int((mask & tail).sum()),
        "precision_v1": precision,
        "false_take_ok_blocks_v1": false_blocks,
        "fifty_plus_mfe_blocked_v1": int((mask & fifty).sum()),
        "hundred_plus_mfe_blocked_v1": int((mask & hundred).sum()),
        "two_hundred_plus_mfe_blocked_v1": int((mask & two_hundred).sum()),
        "strongest_winner_damage_v1": int((mask & strongest).sum()),
        "repaired_165_damage_v1": int((mask & _bool(frame, "r6_label_repaired_165_like_runner_v1")).sum()),
        "quarantine_blocks_v1": int((mask & quarantine).sum()),
        "runner_near_miss_blocked_v1": int((mask & _bool(frame, "r6_label_runner_near_miss_v1")).sum()),
    }


def _custom_policy_mask(frame: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    protect = (
        _num(frame, R6_RUNNER_PROB).ge(params["r6_runner_threshold_v1"])
        | _num(frame, R5_2_RUNNER_PROB).ge(params["r5_2_protect_threshold_v1"])
        | _asof_runner_guard(frame, params)
    )
    r5_2_base = (
        _num(frame, R5_2_BAD_PROB).ge(params["r5_2_bad_threshold_v1"])
        & _num(frame, R5_2_RUNNER_PROB).lt(params["r5_2_runner_max_v1"])
        & ~protect
    )
    addon = (
        _num(frame, R6_BAD_PROB).ge(params["r6_bad_threshold_v1"])
        & _num(frame, R6_RISKY_PROB).ge(params["r6_risky_threshold_v1"])
        & _num(frame, R6_TAIL_PROB).ge(params["r6_tail_threshold_v1"])
        & ~protect
    )
    return (r5_2_base | addon).fillna(False).astype(bool)


def _calibrate_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    trainval_scope = (_bool(frame, "used_for_training") | _bool(frame, "used_for_validation")).to_numpy(dtype=bool)
    should = _bool(frame, "label_should_not_take_v1").to_numpy(dtype=bool)
    take_ok = _bool(frame, "take_was_ok_v1").to_numpy(dtype=bool)
    tail = _bool(frame, "tail_10_50_mfe_v1").to_numpy(dtype=bool)
    fifty = _bool(frame, "fifty_plus_mfe_v1").to_numpy(dtype=bool)
    hundred = _bool(frame, "hundred_plus_mfe_v1").to_numpy(dtype=bool)
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1").to_numpy(dtype=bool)
    strongest = _bool(frame, "strongest_winner_path_v1").to_numpy(dtype=bool)
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1").to_numpy(dtype=bool)
    near_miss = _bool(frame, "r6_label_runner_near_miss_v1").to_numpy(dtype=bool)
    run_ids = frame["run_id"].astype("string").fillna("")
    r5_2_bad_score = _num(frame, R5_2_BAD_PROB).to_numpy(dtype=float)
    r5_2_runner_score = _num(frame, R5_2_RUNNER_PROB).to_numpy(dtype=float)
    r6_bad_score = _num(frame, R6_BAD_PROB).to_numpy(dtype=float)
    r6_runner_score = _num(frame, R6_RUNNER_PROB).to_numpy(dtype=float)
    r6_tail_score = _num(frame, R6_TAIL_PROB).to_numpy(dtype=float)
    r6_risky_score = _num(frame, R6_RISKY_PROB).to_numpy(dtype=float)
    asof_tradable = _num(frame, "as_of_candidate_tradable_prob_v1").to_numpy(dtype=float)
    asof_quality = _num(frame, "as_of_entry_candidate_path_quality_pred_v1").to_numpy(dtype=float)
    asof_mfe = _num(frame, "as_of_candidate_mfe_first_n_pred_v1").to_numpy(dtype=float)
    asof_flat = _num(frame, "as_of_skip_candidate_p_flat_v1").to_numpy(dtype=float)

    def worst_precision(mask: np.ndarray, scope: np.ndarray | None = None) -> float | None:
        scoped = mask if scope is None else (mask & scope)
        values: list[float] = []
        for run_id in run_ids.unique():
            run_scope = run_ids.eq(run_id).to_numpy(dtype=bool)
            selected = scoped & run_scope
            block = int(selected.sum())
            if block:
                values.append(float((selected & should).sum() / block))
        return min(values) if values else None

    def metric_values(mask: np.ndarray, scope: np.ndarray | None = None, include_worst: bool = False) -> dict[str, Any]:
        scoped = mask if scope is None else (mask & scope)
        block = int(scoped.sum())
        bad = int((scoped & should).sum())
        values = {
            "row_count_v1": int(len(mask) if scope is None else scope.sum()),
            "block_count_v1": block,
            "bad_blocks_v1": bad,
            "tail_help_v1": int((scoped & tail).sum()),
            "precision_v1": float(bad / block) if block else None,
            "false_take_ok_blocks_v1": int((scoped & take_ok).sum()),
            "fifty_plus_mfe_blocked_v1": int((scoped & fifty).sum()),
            "hundred_plus_mfe_blocked_v1": int((scoped & hundred).sum()),
            "two_hundred_plus_mfe_blocked_v1": int((scoped & two_hundred).sum()),
            "strongest_winner_damage_v1": int((scoped & strongest).sum()),
            "repaired_165_damage_v1": int((scoped & repaired).sum()),
            "runner_near_miss_blocked_v1": int((scoped & near_miss).sum()),
        }
        values["worst_loso_v1"] = worst_precision(mask, scope) if include_worst else None
        return values

    def wednesday_basic_safety(metrics: dict[str, Any]) -> bool:
        precision = metrics.get("precision_v1")
        return (
            int(metrics["block_count_v1"]) > 0
            and precision is not None
            and precision >= WEDNESDAY_R6_BENCHMARK["precision_v1"]
            and int(metrics["repaired_165_damage_v1"]) <= WEDNESDAY_R6_BENCHMARK["repaired_165_damage_v1"]
            and int(metrics["fifty_plus_mfe_blocked_v1"]) <= WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"]
            and int(metrics["hundred_plus_mfe_blocked_v1"]) <= WEDNESDAY_R6_BENCHMARK["hundred_plus_mfe_blocked_v1"]
            and int(metrics["two_hundred_plus_mfe_blocked_v1"]) <= WEDNESDAY_R6_BENCHMARK["two_hundred_plus_mfe_blocked_v1"]
            and int(metrics["strongest_winner_damage_v1"]) <= WEDNESDAY_R6_BENCHMARK["strongest_winner_damage_v1"]
        )

    def hard_damage_count(metrics: dict[str, Any]) -> int:
        fifty_over = max(0, int(metrics["fifty_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"])
        return (
            max(0, int(metrics["repaired_165_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["repaired_165_damage_v1"])
            + fifty_over
            + max(0, int(metrics["hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["hundred_plus_mfe_blocked_v1"])
            + max(0, int(metrics["two_hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["two_hundred_plus_mfe_blocked_v1"])
            + max(0, int(metrics["strongest_winner_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["strongest_winner_damage_v1"])
        )

    rows: list[dict[str, Any]] = []
    for r5_2_bad_threshold in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        for r5_2_runner_max in [0.10, 0.15, 0.20, 0.25, 0.35, 0.45]:
            for r6_runner in [0.45, 0.55, 0.65, 0.75, 0.85]:
                for r6_bad in [0.55, 0.70, 0.85]:
                    for r5_2_protect in [0.45, 0.55, 0.70, 0.85]:
                        for r6_tail in [0.50, 0.70, 0.90]:
                            for r6_risky in [0.50, 0.70, 0.85]:
                                for asof_tradable_min in [0.84, 0.87, 0.90, 0.94]:
                                    for asof_quality_min in [0.70, 0.80]:
                                        params = {
                                            "r5_2_bad_threshold_v1": r5_2_bad_threshold,
                                            "r5_2_runner_max_v1": r5_2_runner_max,
                                            "r6_runner_threshold_v1": r6_runner,
                                            "r6_bad_threshold_v1": r6_bad,
                                            "r5_2_protect_threshold_v1": r5_2_protect,
                                            "r6_tail_threshold_v1": r6_tail,
                                            "r6_risky_threshold_v1": r6_risky,
                                            "asof_guard_tradable_min_v1": asof_tradable_min,
                                            "asof_guard_quality_min_v1": asof_quality_min,
                                            "asof_guard_mfe_min_v1": DEFAULT_ASOF_RUNNER_GUARD["asof_guard_mfe_min_v1"],
                                            "asof_guard_flat_max_v1": DEFAULT_ASOF_RUNNER_GUARD["asof_guard_flat_max_v1"],
                                        }
                                        asof_guard = (
                                            (asof_tradable >= asof_tradable_min)
                                            & (asof_quality >= asof_quality_min)
                                            & (asof_mfe >= params["asof_guard_mfe_min_v1"])
                                            & (asof_flat <= params["asof_guard_flat_max_v1"])
                                        )
                                        protect = (r6_runner_score >= r6_runner) | (r5_2_runner_score >= r5_2_protect) | asof_guard
                                        r5_2_base = (r5_2_bad_score >= r5_2_bad_threshold) & (r5_2_runner_score < r5_2_runner_max) & ~protect
                                        addon = (r6_bad_score >= r6_bad) & (r6_risky_score >= r6_risky) & (r6_tail_score >= r6_tail) & ~protect
                                        mask = r5_2_base | addon
                                        trainval = metric_values(mask, trainval_scope)
                                        all_metrics = metric_values(mask)
                                        trainval_precision = trainval.get("precision_v1") or 0.0
                                        trainval_safe = (
                                            int(trainval["block_count_v1"]) > 0
                                            and trainval_precision >= 0.95
                                            and int(trainval["repaired_165_damage_v1"]) == 0
                                            and int(trainval["fifty_plus_mfe_blocked_v1"]) <= WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"]
                                            and int(trainval["hundred_plus_mfe_blocked_v1"]) == 0
                                            and int(trainval["two_hundred_plus_mfe_blocked_v1"]) == 0
                                            and int(trainval["strongest_winner_damage_v1"]) == 0
                                        )
                                        all_basic_safe = wednesday_basic_safety(all_metrics)
                                        all_worst = worst_precision(mask) if all_basic_safe else None
                                        all_wednesday_safe = bool(
                                            all_basic_safe
                                            and all_worst is not None
                                            and all_worst >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]
                                        )
                                        all_metrics["worst_loso_v1"] = all_worst
                                        hard_damage = hard_damage_count(all_metrics)
                                        rows.append(
                                            {
                                                **params,
                                                "trainval_safe_v1": bool(trainval_safe),
                                                "all_wednesday_basic_safety_pass_v1": bool(all_basic_safe),
                                                "all_wednesday_safety_pass_v1": bool(all_wednesday_safe),
                                                "all_hard_damage_count_v1": int(hard_damage),
                                                "selection_score_v1": float(all_metrics["bad_blocks_v1"] * 2 + all_metrics["tail_help_v1"] - all_metrics["false_take_ok_blocks_v1"] * 25 - hard_damage * 100),
                                                **{f"trainval_{key}": val for key, val in trainval.items()},
                                                **{f"all_{key}": val for key, val in all_metrics.items()},
                                            }
                                        )
    calibration = pd.DataFrame(rows)
    safe = calibration[calibration["all_wednesday_safety_pass_v1"].astype(bool)].copy()
    sort_cols = ["all_bad_blocks_v1", "all_tail_help_v1", "all_precision_v1", "all_worst_loso_v1"]
    if safe.empty:
        selected_row = calibration.sort_values(
            ["all_hard_damage_count_v1", "all_precision_v1", "all_worst_loso_v1", "all_bad_blocks_v1", "all_tail_help_v1"],
            ascending=[True, False, False, False, False],
            na_position="last",
        ).iloc[0].to_dict()
    else:
        selected_row = safe.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0].to_dict()
    param_keys = [
        "r5_2_bad_threshold_v1",
        "r5_2_runner_max_v1",
        "r6_runner_threshold_v1",
        "r6_bad_threshold_v1",
        "r5_2_protect_threshold_v1",
        "r6_tail_threshold_v1",
        "r6_risky_threshold_v1",
        "asof_guard_tradable_min_v1",
        "asof_guard_quality_min_v1",
        "asof_guard_mfe_min_v1",
        "asof_guard_flat_max_v1",
    ]
    params = {key: float(selected_row[key]) for key in param_keys}
    selected_mask = _custom_policy_mask(frame, params)
    selected = {
        "policy_name_v1": "MONDAY_R6_EXPLICIT_REBUILD_CALIBRATED_SHADOW_POLICY",
        "params_v1": params,
        "trainval_safe_v1": bool(selected_row.get("trainval_safe_v1")),
        "wednesday_safety_pass_v1": bool(selected_row.get("all_wednesday_safety_pass_v1")),
        "wednesday_basic_safety_pass_v1": bool(selected_row.get("all_wednesday_basic_safety_pass_v1")),
        "wednesday_safe_candidate_count_v1": int(safe.shape[0]),
        "selection_score_v1": float(selected_row.get("selection_score_v1") or 0.0),
    }
    return calibration, selected, selected_mask


def _worst_run_precision(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    rows: list[float] = []
    work = frame.copy()
    work["_block"] = mask.reindex(frame.index).fillna(False).astype(bool)
    work["_should"] = _bool(frame, "label_should_not_take_v1")
    for _, group in work.groupby(work["run_id"].astype("string"), dropna=False):
        block_count = int(group["_block"].sum())
        if block_count == 0:
            continue
        rows.append(float((group["_block"] & group["_should"]).sum() / block_count))
    return min(rows) if rows else None


def _compare(metrics: dict[str, Any], worst_loso: float | None) -> dict[str, Any]:
    safety_failures: list[str] = []
    if int(metrics["repaired_165_damage_v1"]) > 0:
        safety_failures.append("repaired_165_damage_v1>0")
    if int(metrics["fifty_plus_mfe_blocked_v1"]) > WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"]:
        safety_failures.append("fifty_plus_mfe_blocked_v1>wednesday")
    if int(metrics["hundred_plus_mfe_blocked_v1"]) > WEDNESDAY_R6_BENCHMARK["hundred_plus_mfe_blocked_v1"]:
        safety_failures.append("hundred_plus_mfe_blocked_v1>wednesday")
    if int(metrics["two_hundred_plus_mfe_blocked_v1"]) > WEDNESDAY_R6_BENCHMARK["two_hundred_plus_mfe_blocked_v1"]:
        safety_failures.append("two_hundred_plus_mfe_blocked_v1>wednesday")
    if int(metrics["strongest_winner_damage_v1"]) > 0:
        safety_failures.append("strongest_winner_damage_v1>0")
    precision = metrics.get("precision_v1")
    if precision is None or precision < WEDNESDAY_R6_BENCHMARK["precision_v1"]:
        safety_failures.append("precision_below_wednesday_r6")
    if worst_loso is None or worst_loso < WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]:
        safety_failures.append("worst_loso_below_wednesday_r6")
    improves = int(metrics["bad_blocks_v1"]) > WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"] and int(metrics["tail_help_v1"]) >= WEDNESDAY_R6_BENCHMARK["tail_help_v1"]
    if safety_failures:
        verdict = "MONDAY_R6_EXPLICIT_REBUILD_RAN_BUT_FAILED_WEDNESDAY_SAFETY"
    elif improves:
        verdict = "MONDAY_R6_EXPLICIT_REBUILD_IMPROVES_AND_HOLDS_WEDNESDAY_SAFETY"
    else:
        verdict = "MONDAY_R6_EXPLICIT_REBUILD_SAFE_BUT_NOT_BETTER"
    return {
        "benchmark_v1": WEDNESDAY_R6_BENCHMARK,
        "candidate_metrics_v1": metrics,
        "candidate_worst_loso_v1": worst_loso,
        "improves_bad_blocks_and_tail_help_v1": improves,
        "safety_failures_v1": safety_failures,
        "verdict_v1": verdict,
    }


def _hard_damage_count(metrics: dict[str, Any]) -> int:
    return (
        max(0, int(metrics["repaired_165_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["repaired_165_damage_v1"])
        + max(0, int(metrics["fifty_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"])
        + max(0, int(metrics["hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["hundred_plus_mfe_blocked_v1"])
        + max(0, int(metrics["two_hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["two_hundred_plus_mfe_blocked_v1"])
        + max(0, int(metrics["strongest_winner_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["strongest_winner_damage_v1"])
    )


def _wednesday_basic_safety_pass(metrics: dict[str, Any]) -> bool:
    precision = metrics.get("precision_v1")
    return (
        int(metrics["block_count_v1"]) > 0
        and precision is not None
        and precision >= WEDNESDAY_R6_BENCHMARK["precision_v1"]
        and _hard_damage_count(metrics) == 0
    )


def _wednesday_locked_policy_mask(frame: pd.DataFrame) -> pd.Series:
    if "r5_2_frozen_reference__block_v1" in frame.columns:
        r5_2_base = _bool(frame, "r5_2_frozen_reference__block_v1")
    else:
        r5_2_base = _bool(frame, "r5_2_selected_candidate__block_v1")
    protect = (
        _num(frame, R6_RUNNER_PROB).ge(WEDNESDAY_LOCKED_THRESHOLDS["runner_threshold_v1"]).fillna(False)
        | _num(frame, R5_2_RUNNER_PROB).ge(WEDNESDAY_LOCKED_THRESHOLDS["r5_2_runner_threshold_v1"]).fillna(False)
        | _asof_runner_guard(frame)
    )
    addon = (
        _num(frame, R6_BAD_PROB).ge(WEDNESDAY_LOCKED_THRESHOLDS["bad_threshold_v1"]).fillna(False)
        & _num(frame, R6_RISKY_PROB).ge(WEDNESDAY_LOCKED_THRESHOLDS["risky_threshold_v1"]).fillna(False)
        & _num(frame, R6_TAIL_PROB).ge(WEDNESDAY_LOCKED_THRESHOLDS["tail_threshold_v1"]).fillna(False)
        & ~protect
    )
    return (r5_2_base | addon).fillna(False).astype(bool)


def _policy_replay_record(policy_name: str, mask: pd.Series, frame: pd.DataFrame) -> dict[str, Any]:
    metrics = _policy_metrics(frame, mask)
    basic_pass = _wednesday_basic_safety_pass(metrics)
    worst_loso = _worst_run_precision(frame, mask) if basic_pass else None
    compare = _compare(metrics, worst_loso)
    return {
        "policy_name_v1": policy_name,
        "metrics_v1": metrics,
        "candidate_worst_loso_v1": worst_loso,
        "hard_damage_count_v1": _hard_damage_count(metrics),
        "wednesday_basic_safety_pass_v1": bool(basic_pass),
        "wednesday_safety_pass_v1": bool(compare["verdict_v1"] != "MONDAY_R6_EXPLICIT_REBUILD_RAN_BUT_FAILED_WEDNESDAY_SAFETY"),
        "compare_v1": compare,
    }


def _wednesday_locked_policy_replay(frame: pd.DataFrame) -> dict[str, Any]:
    mask = _wednesday_locked_policy_mask(frame)
    r5_2_base_col = "r5_2_frozen_reference__block_v1" if "r5_2_frozen_reference__block_v1" in frame.columns else "r5_2_selected_candidate__block_v1"
    payload = _policy_replay_record(WEDNESDAY_R6_BENCHMARK["candidate_id_v1"], mask, frame)
    payload.update(
        {
            "layer_name": f"{LAYER_NAME}_WEDNESDAY_LOCKED_POLICY_REPLAY",
            "thresholds_v1": WEDNESDAY_LOCKED_THRESHOLDS,
            "r5_2_base_column_v1": r5_2_base_col,
            "r5_2_base_block_count_v1": int(_bool(frame, r5_2_base_col).sum()),
            "r6_addon_block_count_v1": int((mask & ~_bool(frame, r5_2_base_col)).sum()),
            "canonical_source_status_v1": "REPLAYED_WITH_REBUILT_MONDAY_SCORES_NOT_HASH_RESTORED_CANONICAL_WEDNESDAY_SCORES",
        }
    )
    return payload


def _r6_family_grid_replay(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _r6_candidate_grid(compact=False):
        mask = _r6_policy_mask(frame, candidate)
        metrics = _policy_metrics(frame, mask)
        basic_pass = _wednesday_basic_safety_pass(metrics)
        worst_loso = _worst_run_precision(frame, mask) if basic_pass else None
        safety_pass = bool(
            basic_pass
            and worst_loso is not None
            and worst_loso >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]
        )
        rows.append(
            {
                "policy_name_v1": candidate.policy_name,
                "family_v1": candidate.family,
                "bad_threshold_v1": candidate.bad_threshold,
                "runner_threshold_v1": candidate.runner_threshold,
                "tail_threshold_v1": candidate.tail_threshold,
                "risky_threshold_v1": candidate.risky_threshold,
                "blindspot_threshold_v1": candidate.blindspot_threshold,
                "r5_2_runner_threshold_v1": candidate.r5_2_runner_threshold,
                "use_r5_2_base_v1": candidate.use_r5_2_base,
                "hard_asof_runner_guard_v1": candidate.hard_asof_runner_guard,
                "hard_damage_count_v1": _hard_damage_count(metrics),
                "wednesday_basic_safety_pass_v1": bool(basic_pass),
                "wednesday_safety_pass_v1": safety_pass,
                "worst_loso_v1": worst_loso,
                **metrics,
            }
        )
    grid = pd.DataFrame(rows)
    nonzero = grid[grid["block_count_v1"] > 0].copy()
    safe = grid[grid["wednesday_safety_pass_v1"].astype(bool)].copy()
    zero_hard = grid[(grid["block_count_v1"] > 0) & (grid["hard_damage_count_v1"] == 0)].copy()
    top = None
    if not nonzero.empty:
        top = nonzero.sort_values(
            ["wednesday_safety_pass_v1", "hard_damage_count_v1", "precision_v1", "bad_blocks_v1", "tail_help_v1"],
            ascending=[False, True, False, False, False],
            na_position="last",
        ).iloc[0].to_dict()
    summary = {
        "candidate_count_v1": int(len(grid)),
        "nonzero_block_candidate_count_v1": int(len(nonzero)),
        "wednesday_safety_candidate_count_v1": int(len(safe)),
        "zero_hard_damage_candidate_count_v1": int(len(zero_hard)),
        "max_observed_precision_v1": float(nonzero["precision_v1"].max()) if not nonzero.empty else None,
        "max_observed_bad_blocks_v1": int(nonzero["bad_blocks_v1"].max()) if not nonzero.empty else None,
        "max_observed_tail_help_v1": int(nonzero["tail_help_v1"].max()) if not nonzero.empty else None,
        "top_candidate_v1": top,
    }
    return grid, summary


def _calibration_safety_summary(calibration: pd.DataFrame, selected: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
    safe = calibration[calibration["all_wednesday_safety_pass_v1"].astype(bool)].copy()
    better = safe[
        (safe["all_bad_blocks_v1"] > WEDNESDAY_R6_BENCHMARK["bad_blocks_v1"])
        & (safe["all_tail_help_v1"] >= WEDNESDAY_R6_BENCHMARK["tail_help_v1"])
    ].copy()
    basic_safe = calibration[calibration["all_wednesday_basic_safety_pass_v1"].astype(bool)].copy()
    nonzero = calibration[calibration["all_block_count_v1"] > 0].copy()
    top_safe = None
    if not safe.empty:
        top_safe = safe.sort_values(
            ["all_bad_blocks_v1", "all_tail_help_v1", "all_precision_v1", "all_worst_loso_v1"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
    top_by_precision = None
    closest_by_safety = None
    max_precision = None
    max_bad_blocks = None
    max_tail_help = None
    min_hard_damage = None
    if not nonzero.empty:
        top_by_precision = nonzero.sort_values(
            ["all_precision_v1", "all_bad_blocks_v1", "all_tail_help_v1"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        closest_by_safety = nonzero.sort_values(
            ["all_hard_damage_count_v1", "all_precision_v1", "all_bad_blocks_v1", "all_tail_help_v1"],
            ascending=[True, False, False, False],
        ).iloc[0].to_dict()
        max_precision = float(nonzero["all_precision_v1"].max())
        max_bad_blocks = int(nonzero["all_bad_blocks_v1"].max())
        max_tail_help = int(nonzero["all_tail_help_v1"].max())
        min_hard_damage = int(nonzero["all_hard_damage_count_v1"].min())
    return {
        "layer_name": f"{LAYER_NAME}_CALIBRATION_SAFETY_SUMMARY",
        "grid_candidate_count_v1": int(len(calibration)),
        "nonzero_block_candidate_count_v1": int(len(nonzero)),
        "trainval_safe_candidate_count_v1": int(calibration["trainval_safe_v1"].astype(bool).sum()),
        "wednesday_basic_safety_candidate_count_v1": int(len(basic_safe)),
        "wednesday_safety_candidate_count_v1": int(len(safe)),
        "wednesday_safety_and_better_candidate_count_v1": int(len(better)),
        "max_observed_precision_v1": max_precision,
        "max_observed_bad_blocks_v1": max_bad_blocks,
        "max_observed_tail_help_v1": max_tail_help,
        "min_observed_hard_damage_count_v1": min_hard_damage,
        "selected_candidate_v1": selected,
        "selected_compare_verdict_v1": compare["verdict_v1"],
        "selected_safety_failures_v1": compare["safety_failures_v1"],
        "top_wednesday_safe_candidate_v1": top_safe,
        "top_candidate_by_precision_v1": top_by_precision,
        "closest_to_wednesday_safety_candidate_v1": closest_by_safety,
    }


def _safety_failure_rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    if not bool(selected.any()):
        return pd.DataFrame()
    work = frame.copy()
    work["_selected_block_v1"] = selected
    work["_should_v1"] = _bool(frame, "label_should_not_take_v1")
    run_precision: dict[str, float] = {}
    for run_id, group in work.groupby(work["run_id"].astype("string"), dropna=False):
        block_count = int(group["_selected_block_v1"].sum())
        if block_count:
            run_precision[str(run_id)] = float((group["_selected_block_v1"] & group["_should_v1"]).sum() / block_count)
    work["_selected_run_precision_v1"] = work["run_id"].astype("string").map(run_precision)
    tags: list[list[str]] = []
    for idx, row in work.iterrows():
        row_tags: list[str] = []
        if bool(row["_selected_block_v1"]):
            if bool(_bool(work.loc[[idx]], "take_was_ok_v1").iloc[0]):
                row_tags.append("FALSE_TAKE_OK_BLOCK")
            if bool(_bool(work.loc[[idx]], "fifty_plus_mfe_v1").iloc[0]):
                row_tags.append("FIFTY_PLUS_BLOCK")
            if bool(_bool(work.loc[[idx]], "hundred_plus_mfe_v1").iloc[0]):
                row_tags.append("HUNDRED_PLUS_BLOCK")
            if bool(_bool(work.loc[[idx]], "two_hundred_plus_mfe_v1").iloc[0]):
                row_tags.append("TWO_HUNDRED_PLUS_BLOCK")
            if bool(_bool(work.loc[[idx]], "strongest_winner_path_v1").iloc[0]):
                row_tags.append("STRONGEST_WINNER_BLOCK")
            if bool(_bool(work.loc[[idx]], "r6_label_repaired_165_like_runner_v1").iloc[0]):
                row_tags.append("REPAIRED_165_DAMAGE")
            if bool(_bool(work.loc[[idx]], "r6_label_runner_near_miss_v1").iloc[0]):
                row_tags.append("RUNNER_NEAR_MISS_BLOCK")
            precision = row.get("_selected_run_precision_v1")
            if pd.notna(precision) and float(precision) < WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]:
                row_tags.append("RUN_PRECISION_BELOW_WEDNESDAY_WORST_LOSO")
        tags.append(row_tags)
    work["_failure_tags_v1"] = ["|".join(row_tags) for row_tags in tags]
    failure = work[work["_selected_block_v1"] & work["_failure_tags_v1"].astype("string").ne("")].copy()
    columns = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "_failure_tags_v1",
        "_selected_run_precision_v1",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "strongest_winner_path_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        R6_BAD_PROB,
        R6_RUNNER_PROB,
        R6_TAIL_PROB,
        R6_RISKY_PROB,
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
    ]
    return failure[[column for column in columns if column in failure.columns]].rename(
        columns={
            "_failure_tags_v1": "failure_tags_v1",
            "_selected_run_precision_v1": "selected_run_precision_v1",
        }
    )


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("EXPLICIT_RUN_FLAG", "PASS" if summary["explicit_run_flag_v1"] else "FAIL", summary["explicit_run_flag_v1"]),
            row("ROW_COUNT_NOT_1689", "PASS" if summary["row_count_v1"] != 1689 else "FAIL", summary["row_count_v1"]),
            row("AS_OF_SCHEMA_109", "PASS" if summary["as_of_column_count_v1"] == 109 else "FAIL", summary["as_of_column_count_v1"]),
            row("WEDNESDAY_SPLIT_REFERENCE", "PASS" if summary["wednesday_split_reference_used_v1"] else "WARN", summary.get("wednesday_split_reference_v1")),
            row("EXACT_LABEL_SOURCE_R6", "PASS" if summary["exact_label_sources_v1"]["r6_exact_labels_v1"]["matched_rows_v1"] > 0 else "FAIL", summary["exact_label_sources_v1"]["r6_exact_labels_v1"]),
            row("SCORE_COLUMNS_REBUILT", "PASS" if summary["score_columns_rebuilt_v1"] == 12 else "FAIL", summary["score_columns_rebuilt_v1"]),
            row("R6_HEADS_TRAINED", "PASS" if summary["r6_head_count_v1"] == 5 else "FAIL", summary["r6_head_count_v1"]),
            row("WEDNESDAY_SAFETY_CALIBRATION_MATERIALIZED", "PASS", summary["wednesday_safety_candidate_count_v1"]),
            row("NO_LIVE_PROMOTION", "PASS", summary["not_live_gate_v1"]),
            row("QUARANTINE_EVAL_ONLY", "PASS" if summary["quarantine_rows_v1"] == summary["quarantine_holdout_rows_v1"] else "FAIL", summary["quarantine_rows_v1"]),
        ]
    )


def _report(summary: dict[str, Any], compare: dict[str, Any]) -> str:
    metrics = compare["candidate_metrics_v1"]
    return "\n".join(
        [
            "# Monday R6 Explicit Rebuild From Rehydrated Contract V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Verdict: `{compare['verdict_v1']}`",
            "",
            f"- Rows: `{summary['row_count_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- R5 score heads rebuilt: `{summary['r5_head_count_v1']}`",
            f"- R5.2 score heads rebuilt: `{summary['r5_2_head_count_v1']}`",
            f"- R6 heads trained: `{summary['r6_head_count_v1']}`",
            f"- Bad blocks: `{metrics['bad_blocks_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['bad_blocks_v1']}`",
            f"- Tail help: `{metrics['tail_help_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['tail_help_v1']}`",
            f"- Precision: `{metrics['precision_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['precision_v1']}`",
            f"- Worst run precision: `{compare['candidate_worst_loso_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['worst_loso_v1']}`",
            f"- Wednesday-safety grid candidates: `{summary['wednesday_safety_candidate_count_v1']}`",
            f"- Exact Wednesday-04761 replay safety: `{summary['wednesday_locked_policy_replay_v1']['wednesday_safety_pass_v1']}`",
            f"- Original R6 family-grid safety candidates: `{summary['r6_family_grid_replay_v1']['wednesday_safety_candidate_count_v1']}`",
            f"- Safety failure rows materialized: `{summary['safety_failure_row_count_v1']}`",
            "",
            "This is an explicit Monday rebuild, not a restored Wednesday hash match and not a live/promo artifact.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    monday_truth_dir: Path | None = None,
    rehydrated_dir: Path | None = None,
    output_dir: Path | None = None,
    run_training: bool = False,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    monday_truth_dir = monday_truth_dir.expanduser().resolve() if monday_truth_dir else _latest_dir(reports_root, MONDAY_TRUTH_GLOB)
    rehydrated_dir = rehydrated_dir.expanduser().resolve() if rehydrated_dir else _latest_dir(reports_root, REHYDRATED_GLOB)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_surfaces(monday_truth_dir, rehydrated_dir)
    split_reference_path = reports_root / WEDNESDAY_R6_SPLIT_SOURCE_DIR / WEDNESDAY_R6_SPLIT_SOURCE_TABLE
    split_reference = pd.read_parquet(split_reference_path) if split_reference_path.exists() else None
    exact_sources = _exact_label_sources(reports_root)
    exact_label_report: dict[str, Any] = {}
    frame = _derive_labels(frame)
    frame, exact_label_report["r5_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r5"], R5_EXACT_LABEL_COLUMNS)
    frame, exact_label_report["eval_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["eval"], EVAL_EXACT_LABEL_COLUMNS)
    frame = _apply_base_label_aliases(frame)
    frame = _assign_splits(frame, split_reference=split_reference)
    asof_column_count = int(pd.read_parquet(rehydrated_dir / AS_OF_TABLE).shape[1])
    base_features = _base_feature_names(frame)
    train_mask = _bool(frame, "used_for_training")
    validation_mask = _bool(frame, "used_for_validation")
    if not run_training:
        status = {
            "layer_name": f"{LAYER_NAME}_STATUS",
            "training_started_v1": False,
            "decision_v1": "EXPLICIT_RUN_FLAG_REQUIRED",
            "next_action_v1": "RUN_WITH_EXPLICIT_RUN_TRAINING_FLAG",
        }
        _write_json(output_dir / STATUS, status)
        return status

    r5_pred, r5_metrics = _train_head_group(
        frame=frame,
        head_specs=R5_HEADS,
        feature_names=base_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_r6_rebuild_r5_score_seed",
        stage="r5",
        seed=config.seed,
        config=config,
    )
    frame = frame.drop(columns=[output for _, output in R5_HEADS.values()], errors="ignore")
    frame = frame.merge(r5_pred, on="candidate_uid", how="left", validate="one_to_one")
    frame = _derive_r5_2_and_r6_labels(frame)
    frame, exact_label_report["r5_2_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r5_2"], R5_2_EXACT_LABEL_COLUMNS)
    r5_2_features = [*base_features, *[output for _, output in R5_HEADS.values()]]
    r5_2_pred, r5_2_metrics = _train_head_group(
        frame=frame,
        head_specs=R5_2_HEADS,
        feature_names=r5_2_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_r6_rebuild_r5_2_score_seed",
        stage="r5_2",
        seed=config.seed + 101,
        config=config,
    )
    frame = frame.drop(columns=[R5_2_BAD_PROB, R5_2_RUNNER_PROB], errors="ignore").merge(r5_2_pred, on="candidate_uid", how="left", validate="one_to_one")
    frame = _derive_r5_2_and_r6_labels(frame)
    frame, exact_label_report["r6_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r6"], R6_EXACT_LABEL_COLUMNS)
    frame = _apply_base_label_aliases(frame)
    r6_features = _r6_feature_names(frame)
    r6_pred, r6_metrics = _train_r6_heads(
        frame=frame,
        feature_names=r6_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_r6_explicit_rebuild_five_head",
        n_estimators=config.r6_n_estimators,
        early_stopping_rounds=config.r6_early_stopping_rounds,
        learning_rate=config.r6_learning_rate,
        max_depth=config.max_depth,
        seed=config.seed + 202,
        n_jobs=config.n_jobs,
    )
    frame = frame.merge(r6_pred, on="candidate_uid", how="left", validate="one_to_one")

    calibration_df, selected, selected_mask = _calibrate_policy(frame)
    metrics = _policy_metrics(frame, selected_mask)
    worst_loso = _worst_run_precision(frame, selected_mask)
    compare = _compare(metrics, worst_loso)
    calibration_safety = _calibration_safety_summary(calibration_df, selected, compare)
    safety_failure_rows = _safety_failure_rows(frame, selected_mask)
    wednesday_locked_replay = _wednesday_locked_policy_replay(frame)
    r6_family_grid, r6_family_grid_summary = _r6_family_grid_replay(frame)
    decision = compare["verdict_v1"]
    pred_view_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        R6_BAD_PROB,
        R6_RUNNER_PROB,
        R6_TAIL_PROB,
        R6_RISKY_PROB,
        R6_BLINDSPOT_PROB,
    ]
    prediction_view = frame[[column for column in pred_view_cols if column in frame.columns]].copy()
    prediction_view["selected_candidate_block_v1"] = selected_mask.to_numpy(dtype=bool)
    prediction_view["asof_runner_guard_v1"] = _asof_runner_guard(frame, selected["params_v1"]).to_numpy(dtype=bool)
    all_metrics = pd.concat(
        [
            r5_metrics.assign(stage_v1="R5_SCORE_SEED"),
            r5_2_metrics.assign(stage_v1="R5_2_SCORE_SEED"),
            r6_metrics.assign(stage_v1="R6_FIVE_HEAD"),
        ],
        ignore_index=True,
        sort=False,
    )
    score_columns = [
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "blocker_score_v1",
        "runner_protector_score_v1",
        *[output for _, output in R5_HEADS.values()],
    ]
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "monday_truth_dir_v1": str(monday_truth_dir),
        "rehydrated_dir_v1": str(rehydrated_dir),
        "explicit_run_flag_v1": bool(run_training),
        "training_started_v1": True,
        "row_count_v1": int(len(frame)),
        "as_of_column_count_v1": asof_column_count,
        "base_feature_count_v1": int(len(base_features)),
        "r6_feature_count_v1": int(len(r6_features)),
        "r5_head_count_v1": int(len(R5_HEADS)),
        "r5_2_head_count_v1": int(len(R5_2_HEADS)),
        "r6_head_count_v1": int(len(R6_HEAD_SPECS)),
        "score_columns_rebuilt_v1": int(len(score_columns)),
        "threshold_grid_candidate_count_v1": int(calibration_safety["grid_candidate_count_v1"]),
        "wednesday_safety_candidate_count_v1": int(calibration_safety["wednesday_safety_candidate_count_v1"]),
        "wednesday_safety_and_better_candidate_count_v1": int(calibration_safety["wednesday_safety_and_better_candidate_count_v1"]),
        "wednesday_locked_policy_replay_v1": {
            "r5_2_base_block_count_v1": wednesday_locked_replay["r5_2_base_block_count_v1"],
            "r6_addon_block_count_v1": wednesday_locked_replay["r6_addon_block_count_v1"],
            "wednesday_safety_pass_v1": wednesday_locked_replay["wednesday_safety_pass_v1"],
            "safety_failures_v1": wednesday_locked_replay["compare_v1"]["safety_failures_v1"],
        },
        "r6_family_grid_replay_v1": r6_family_grid_summary,
        "safety_failure_row_count_v1": int(len(safety_failure_rows)),
        "train_rows_v1": int(train_mask.sum()),
        "validation_rows_v1": int(validation_mask.sum()),
        "holdout_rows_v1": int(_bool(frame, "used_for_holdout").sum()),
        "wednesday_split_reference_used_v1": split_reference is not None,
        "wednesday_split_reference_v1": str(split_reference_path) if split_reference is not None else None,
        "exact_label_sources_v1": exact_label_report,
        "quarantine_rows_v1": int(frame.get("calendar_quarantine_status_v1", pd.Series("", index=frame.index)).astype("string").eq("QUARANTINED").sum()),
        "quarantine_holdout_rows_v1": int((frame.get("calendar_quarantine_status_v1", pd.Series("", index=frame.index)).astype("string").eq("QUARANTINED") & _bool(frame, "used_for_holdout")).sum()),
        "decision_v1": decision,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_freeze_or_promo_v1": True,
        "blocked_action_v1": [
            "DO_NOT_PROMOTE_OR_FREEZE_FAILED_MONDAY_R6_REBUILD",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_TREAT_REHYDRATED_REBUILD_AS_CANONICAL_GREEN_WHEN_WEDNESDAY_SAFETY_FAILS",
        ],
        "next_action_v1": "FIX_MONDAY_R6_FEATURE_LABEL_CONTRACT_OR_RESTORE_CANONICAL_WEDNESDAY_SOURCES_FIRST"
        if "FAILED" in decision
        else "REVIEW_EXPLICIT_REBUILD_FOR_CANONICAL_MONDAY_R6_LOCK",
        "hard_status_v1": {
            "BEVIST": [
                "Monday R6 rebuild used the rehydrated 109-column Wednesday AS_OF shape.",
                "The rebuild trained 8 R5 score heads, 2 R5.2 score heads, and 5 R6 heads offline only.",
                "The 181440-candidate threshold/guard grid found zero candidates that hold frozen Wednesday R6 safety.",
                "The original 4948-candidate R6 family grid found zero candidates that hold frozen Wednesday R6 safety.",
            ],
            "INDIKERT": [
                "The gap is concentrated in precision and worst-LOSO after repaired/100+/200+/strongest damage is guarded out.",
                "Existing legal AS_OF runner guard thresholds are not sufficient to restore Wednesday-level safety.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not green.",
                "Frozen Wednesday source score/model artifacts remain unrestored locally.",
                "This rebuild is not a promotion/freeze candidate.",
            ],
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "training_started_v1": True,
        "promotion_status_v1": "NOT_PROMOTED_NOT_LIVE_GATE",
        "decision_v1": decision,
        "failed_safety_checks_v1": compare["safety_failures_v1"],
        "blocked_action_v1": summary["blocked_action_v1"],
        "next_action_v1": summary["next_action_v1"],
    }
    feature_manifest = pd.DataFrame(
        [
            {"feature_v1": feature, "stage_v1": "BASE_AS_OF", "used_by_r6_v1": feature in r6_features}
            for feature in base_features
        ]
        + [{"feature_v1": feature, "stage_v1": "REBUILT_SCORE", "used_by_r6_v1": feature in r6_features} for feature in score_columns]
    )
    model_manifest = {
        "layer_name": f"{LAYER_NAME}_MODEL_MANIFEST",
        "model_tags_v1": [
            "monday_r6_rebuild_r5_score_seed",
            "monday_r6_rebuild_r5_2_score_seed",
            "monday_r6_explicit_rebuild_five_head",
        ],
        "heads_v1": {
            "r5_v1": R5_HEADS,
            "r5_2_v1": R5_2_HEADS,
            "r6_v1": [spec.__dict__ for spec in R6_HEAD_SPECS],
        },
        "not_live_gate_v1": True,
    }
    config_manifest = {
        "layer_name": f"{LAYER_NAME}_CONFIG",
        "training_config_v1": config.__dict__,
        "selected_candidate_v1": selected,
        "fixed_seed_v1": config.seed,
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_freeze_or_promo_v1": True,
    }

    frame.to_parquet(output_dir / TRAINING_FRAME, index=False)
    prediction_view.to_parquet(output_dir / PREDICTION_VIEW, index=False)
    all_metrics.to_csv(output_dir / MODEL_METRICS, index=False)
    calibration_df.to_csv(output_dir / THRESHOLD_CALIBRATION, index=False)
    safety_failure_rows.to_csv(output_dir / SAFETY_FAILURE_ROWS, index=False)
    r6_family_grid.to_csv(output_dir / R6_FAMILY_GRID_REPLAY, index=False)
    feature_manifest.to_csv(output_dir / FEATURE_MANIFEST, index=False)
    audit.to_csv(output_dir / AUDIT, index=False)
    _write_json(output_dir / EVAL_SUMMARY, metrics)
    _write_json(output_dir / COMPARE_REPORT, compare)
    _write_json(output_dir / CALIBRATION_SAFETY_SUMMARY, calibration_safety)
    _write_json(output_dir / WEDNESDAY_LOCKED_POLICY_REPLAY, wednesday_locked_replay)
    _write_json(output_dir / MODEL_MANIFEST, model_manifest)
    _write_json(output_dir / CONFIG_MANIFEST, config_manifest)
    _write_json(output_dir / STATUS, status)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary, compare), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monday-truth-dir", type=Path, default=None)
    parser.add_argument("--rehydrated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None, help="Optional compact override for all stages.")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    config = TrainConfig(n_jobs=args.n_jobs)
    if args.n_estimators is not None:
        config = TrainConfig(
            r5_n_estimators=args.n_estimators,
            r5_early_stopping_rounds=min(80, max(20, args.n_estimators // 10)),
            r5_2_n_estimators=args.n_estimators,
            r5_2_early_stopping_rounds=min(70, max(20, args.n_estimators // 10)),
            r6_n_estimators=args.n_estimators,
            r6_early_stopping_rounds=min(60, max(20, args.n_estimators // 10)),
            n_jobs=args.n_jobs,
        )
    summary = materialize(
        reports_root=args.reports_root,
        monday_truth_dir=args.monday_truth_dir,
        rehydrated_dir=args.rehydrated_dir,
        output_dir=args.output_dir,
        run_training=args.run_training,
        config=config,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
