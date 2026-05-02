#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_HEADS,
    R5_2_RUNNER_PROB,
    R5_HEADS,
    TrainConfig,
    WEDNESDAY_R6_BENCHMARK,
    _base_feature_names,
    _bool,
    _derive_r5_2_and_r6_labels,
    _num,
    _train_head_group,
    _worst_run_precision,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1"
FOUNDATION_GLOB = "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_*"

FOUNDATION_FRAME = "monday_r6_foundation_training_frame_pre_score_v1.parquet"
FOUNDATION_AS_OF = "monday_r6_foundation_as_of_109_v1.parquet"
FOUNDATION_SUMMARY = "summary_v1.json"

SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
R5_PREDICTION_VIEW = "monday_r5_score_prediction_view_v1.parquet"
R5_1_PREDICTION_VIEW = "monday_r5_1_score_prediction_view_v1.parquet"
R5_2_PREDICTION_VIEW = "monday_r5_2_score_prediction_view_v1.parquet"
MODEL_METRICS = "model_metrics_v1.csv"
R5_1_CALIBRATION = "r5_1_policy_calibration_v1.csv"
R5_2_CALIBRATION = "r5_2_policy_calibration_v1.csv"
R5_2_BASE_MEMBERSHIP_CONTRACT_FILE = "r5_2_base_membership_contract_v1.json"
SCORE_SUMMARY = "score_rebuild_summary_v1.json"
MODEL_MANIFEST = "model_manifest_v1.json"
CONFIG_MANIFEST = "config_manifest_v1.json"
FEATURE_MANIFEST = "feature_manifest_v1.csv"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"
STATUS = "status_v1.json"
SUMMARY = "summary_v1.json"
MANIFEST = "manifest_v1.json"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "score_frame": SCORE_FRAME,
    "r5_prediction_view": R5_PREDICTION_VIEW,
    "r5_1_prediction_view": R5_1_PREDICTION_VIEW,
    "r5_2_prediction_view": R5_2_PREDICTION_VIEW,
    "model_metrics": MODEL_METRICS,
    "r5_1_calibration": R5_1_CALIBRATION,
    "r5_2_calibration": R5_2_CALIBRATION,
    "r5_2_base_membership_contract": R5_2_BASE_MEMBERSHIP_CONTRACT_FILE,
    "score_summary": SCORE_SUMMARY,
    "model_manifest": MODEL_MANIFEST,
    "config_manifest": CONFIG_MANIFEST,
    "feature_manifest": FEATURE_MANIFEST,
    "audit": CONSISTENCY_AUDIT,
    "status": STATUS,
    "summary": SUMMARY,
    "manifest": MANIFEST,
    "report": REPORT,
    "models_dir": "models",
}

EXPECTED_FOUNDATION_ROWS = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_AS_OF_COLUMNS = 109
FORBIDDEN_BASELINE_ROWS = {1689, 1852}
R5_2_BASE_MEMBERSHIP_CONTRACT_V1 = {
    "contract_id_v1": "R5_2_BASE_MEMBERSHIP_CONTRACT_MAE_CONFIRMED_RECALL_EXTENSION_V1",
    "base_rule_v1": "calibrated_r5_2_bad_score_and_runner_max",
    "extension_rule_v1": "r5_2_bad_score_at_or_above_selected_threshold AND r5_immediate_mae_score>=0.75 AND r5_runner<0.85 AND r5_1_runner<0.85 AND r5_2_runner<0.60",
    "uses_new_feature_surface_v1": False,
    "uses_existing_scores_only_v1": True,
    "requires_wednesday_safety_gate_v1": True,
    "extension_min_r5_immediate_mae_score_v1": 0.75,
    "extension_r5_runner_max_v1": 0.85,
    "extension_r5_1_runner_max_v1": 0.85,
    "extension_r5_2_runner_max_v1": 0.60,
}
R5_2_BASE_MEMBERSHIP_CONTRACT_V2 = {
    "contract_id_v1": "R5_2_BASE_MEMBERSHIP_CONTRACT_SAFE_RECALL_EXTENSION_V2",
    "source_rule_v1": "PARALLEL_MONDAY_R6_RECALL_RECOVERY_SCAN_V1.LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1.best_safe_bad_candidate_v1",
    "base_rule_v1": "original_calibrated_r5_2_base_then_v1_extension_then_v2_safe_recall_extension",
    "extension_rule_v1": "r5_2_bad_score>=0.35 AND r5_immediate_mae_score>=0.75 AND r5_runner<0.45 AND r5_1_runner<0.45 AND r5_2_runner<0.35",
    "uses_new_feature_surface_v1": False,
    "uses_existing_scores_only_v1": True,
    "requires_wednesday_safety_gate_v1": True,
    "extension_bad_source_v1": "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "extension_bad_threshold_v1": 0.35,
    "extension_min_r5_immediate_mae_score_v1": 0.75,
    "extension_r5_runner_max_v1": 0.45,
    "extension_r5_1_runner_max_v1": 0.45,
    "extension_r5_2_runner_max_v1": 0.35,
    "exclude_asof_runner_guard_v1": False,
    "expected_current_bad_tail_v1": [76, 48],
    "expected_v2_bad_tail_v1": [78, 49],
    "expected_v2_incremental_bad_tail_uplift_v1": [2, 1],
    "expected_precision_v1": 1.0,
    "expected_worst_loso_v1": 1.0,
}
R5_2_BASE_MEMBERSHIP_CONTRACT_V3 = {
    "contract_id_v1": "R5_2_BASE_MEMBERSHIP_CONTRACT_SAFE_RECALL_EXTENSION_V3",
    "source_rule_v1": "EXTEND_R5_2_BASE_CONTRACT_V3_ONLY_IF_SAFE.V3_R5_R5_1_R5_2_AGREEMENT_LANE.best_safe_candidate_v1",
    "base_rule_v1": "original_calibrated_r5_2_base_then_v1_extension_then_v2_safe_recall_extension_then_v3_score_agreement_extension",
    "extension_rule_v1": "r5_should_not_take_score>=0.35 AND r5_1_bad_score>=0.85 AND r5_2_bad_score>=0.35 AND r5_runner<0.55 AND r5_1_runner<0.55 AND r5_2_runner<0.55",
    "uses_new_feature_surface_v1": False,
    "uses_existing_scores_only_v1": True,
    "requires_wednesday_safety_gate_v1": True,
    "extension_r5_bad_source_v1": "pred__entry_r5_should_not_take__prob_true_v1",
    "extension_r5_bad_threshold_v1": 0.35,
    "extension_r5_1_bad_source_v1": "r5_1_bad_blocker_score_v1",
    "extension_r5_1_bad_threshold_v1": 0.85,
    "extension_r5_2_bad_source_v1": R5_2_BAD_PROB,
    "extension_r5_2_bad_threshold_v1": 0.35,
    "extension_r5_runner_max_v1": 0.55,
    "extension_r5_1_runner_max_v1": 0.55,
    "extension_r5_2_runner_max_v1": 0.55,
    "expected_current_bad_tail_v1": [78, 49],
    "expected_v3_bad_tail_v1": [82, 51],
    "expected_v3_incremental_bad_tail_uplift_v1": [4, 2],
    "expected_precision_v1": 1.0,
    "expected_worst_loso_v1": 1.0,
    "uplift_class_v1": "ONLY_TINY_SAFE_V3_FOUND",
}
R5_2_BASE_MEMBERSHIP_CONTRACT = R5_2_BASE_MEMBERSHIP_CONTRACT_V3


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _latest_dir(reports_root: Path, pattern: str, required_file: str) -> Path:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    if not dirs:
        raise FileNotFoundError(f"No {pattern} with {required_file} under {reports_root}")
    return dirs[-1]


def _validate_foundation(foundation_dir: Path, frame: pd.DataFrame, asof: pd.DataFrame, summary: dict[str, Any]) -> None:
    if len(frame) in FORBIDDEN_BASELINE_ROWS:
        raise RuntimeError(f"Refuses forbidden narrow/active-only row count: {len(frame)}")
    if int(len(frame)) != EXPECTED_FOUNDATION_ROWS:
        raise RuntimeError(f"Expected Monday actual foundation rows {EXPECTED_FOUNDATION_ROWS}, observed {len(frame)}")
    if int(asof.shape[1]) != EXPECTED_AS_OF_COLUMNS:
        raise RuntimeError(f"Expected {EXPECTED_AS_OF_COLUMNS} AS_OF columns, observed {asof.shape[1]}")
    active = frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    if int(active.sum()) != EXPECTED_ACTIVE_ROWS or int((~active).sum()) != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, observed {int(active.sum())}/{int((~active).sum())}")
    if summary.get("decision_v1") != "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT":
        raise RuntimeError(f"Foundation is not green: {foundation_dir} decision={summary.get('decision_v1')}")
    missing = [column for column in ["candidate_uid", "used_for_training", "used_for_validation", "used_for_holdout"] if column not in frame.columns]
    if missing:
        raise KeyError(f"Foundation frame missing required columns: {missing}")


def _policy_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    fifty = _bool(frame, "fifty_plus_mfe_v1")
    hundred = _bool(frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(frame, "strongest_winner_path_v1")
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1")
    block = int(selected.sum())
    bad = int((selected & should).sum())
    return {
        "block_count_v1": block,
        "bad_blocks_v1": bad,
        "tail_help_v1": int((selected & tail).sum()),
        "precision_v1": float(bad / block) if block else None,
        "false_take_ok_blocks_v1": int((selected & take_ok).sum()),
        "fifty_plus_mfe_blocked_v1": int((selected & fifty).sum()),
        "hundred_plus_mfe_blocked_v1": int((selected & hundred).sum()),
        "two_hundred_plus_mfe_blocked_v1": int((selected & two_hundred).sum()),
        "strongest_winner_damage_v1": int((selected & strongest).sum()),
        "repaired_165_damage_v1": int((selected & repaired).sum()),
    }


def _hard_damage_count(metrics: dict[str, Any]) -> int:
    fifty_over = max(0, int(metrics["fifty_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"])
    return (
        max(0, int(metrics["repaired_165_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["repaired_165_damage_v1"])
        + fifty_over
        + max(0, int(metrics["hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["hundred_plus_mfe_blocked_v1"])
        + max(0, int(metrics["two_hundred_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["two_hundred_plus_mfe_blocked_v1"])
        + max(0, int(metrics["strongest_winner_damage_v1"]) - WEDNESDAY_R6_BENCHMARK["strongest_winner_damage_v1"])
    )


def _wednesday_safety_pass(frame: pd.DataFrame, mask: pd.Series) -> tuple[bool, float | None, int]:
    metrics = _policy_metrics(frame, mask)
    worst_loso = _worst_run_precision(frame, mask)
    hard_damage = _hard_damage_count(metrics)
    precision = metrics.get("precision_v1")
    return (
        bool(
            int(metrics["block_count_v1"]) > 0
            and precision is not None
            and float(precision) >= WEDNESDAY_R6_BENCHMARK["precision_v1"]
            and worst_loso is not None
            and float(worst_loso) >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]
            and hard_damage == 0
        ),
        worst_loso,
        hard_damage,
    )


def _r5_2_contract_extension_mask(frame: pd.DataFrame, selected_bad_threshold: float) -> pd.Series:
    return (
        _num(frame, R5_2_BAD_PROB).ge(float(selected_bad_threshold)).fillna(False)
        & _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_min_r5_immediate_mae_score_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V1["extension_r5_2_runner_max_v1"])).fillna(False)
    ).fillna(False)


def _r5_2_contract_v2_extension_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, R5_2_BAD_PROB).ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_bad_threshold_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_min_r5_immediate_mae_score_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V2["extension_r5_2_runner_max_v1"])).fillna(False)
    ).fillna(False)


def _r5_2_contract_v3_extension_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, "pred__entry_r5_should_not_take__prob_true_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_bad_threshold_v1"])).fillna(False)
        & _num(frame, "r5_1_bad_blocker_score_v1").ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_bad_threshold_v1"])).fillna(False)
        & _num(frame, R5_2_BAD_PROB).ge(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_bad_threshold_v1"])).fillna(False)
        & _num(frame, "pred__entry_r5_runner_protect__prob_true_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_runner_max_v1"])).fillna(False)
        & _num(frame, "r5_1_runner_guard_score_v1").lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_1_runner_max_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER_PROB).lt(float(R5_2_BASE_MEMBERSHIP_CONTRACT_V3["extension_r5_2_runner_max_v1"])).fillna(False)
    ).fillna(False)


def _calibrate_r5_1(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    should_prob = _num(frame, "pred__entry_r5_should_not_take__prob_true_v1")
    mae_prob = _num(frame, "pred__entry_r5_immediate_MAE_risk__prob_true_v1")
    tail_prob = _num(frame, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1")
    runner_prob = _num(frame, "pred__entry_r5_runner_protect__prob_true_v1")
    strong_prob = _num(frame, "pred__entry_r5_strong_trade_candidate__prob_true_v1")
    take_ok_prob = _num(frame, "pred__entry_r5_take_was_ok__prob_true_v1")
    rows: list[dict[str, Any]] = []
    for bad_threshold in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for mae_threshold in [0.45, 0.55, 0.65, 0.75]:
            for tail_threshold in [0.45, 0.55, 0.70]:
                for runner_protect_threshold in [0.45, 0.55, 0.70, 0.85]:
                    raw_bad_score = np.maximum.reduce(
                        [
                            should_prob.fillna(0.0).to_numpy(dtype=float),
                            mae_prob.fillna(0.0).to_numpy(dtype=float),
                            tail_prob.fillna(0.0).to_numpy(dtype=float) * 0.8,
                        ]
                    )
                    protect_score = np.maximum.reduce(
                        [
                            runner_prob.fillna(0.0).to_numpy(dtype=float),
                            strong_prob.fillna(0.0).to_numpy(dtype=float),
                            take_ok_prob.fillna(0.0).to_numpy(dtype=float) * 0.8,
                        ]
                    )
                    mask = (
                        ((should_prob >= bad_threshold) | (mae_prob >= mae_threshold) | (tail_prob >= tail_threshold))
                        & (pd.Series(protect_score, index=frame.index) < runner_protect_threshold)
                    ).fillna(False)
                    metrics = _policy_metrics(frame, mask)
                    hard_damage = (
                        int(metrics["repaired_165_damage_v1"])
                        + max(0, int(metrics["fifty_plus_mfe_blocked_v1"]) - WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"])
                        + int(metrics["hundred_plus_mfe_blocked_v1"])
                        + int(metrics["two_hundred_plus_mfe_blocked_v1"])
                        + int(metrics["strongest_winner_damage_v1"])
                    )
                    rows.append(
                        {
                            "bad_threshold_v1": bad_threshold,
                            "mae_threshold_v1": mae_threshold,
                            "tail_threshold_v1": tail_threshold,
                            "runner_protect_threshold_v1": runner_protect_threshold,
                            "hard_damage_count_v1": hard_damage,
                            "selection_score_v1": float(metrics["bad_blocks_v1"] * 2 + metrics["tail_help_v1"] - metrics["false_take_ok_blocks_v1"] * 20 - hard_damage * 100),
                            **metrics,
                        }
                    )
    calibration = pd.DataFrame(rows)
    selected_row = calibration.sort_values(
        ["hard_damage_count_v1", "precision_v1", "bad_blocks_v1", "tail_help_v1"],
        ascending=[True, False, False, False],
        na_position="last",
    ).iloc[0].to_dict()
    raw_bad_score = np.maximum.reduce(
        [
            should_prob.fillna(0.0).to_numpy(dtype=float),
            mae_prob.fillna(0.0).to_numpy(dtype=float),
            tail_prob.fillna(0.0).to_numpy(dtype=float) * 0.8,
        ]
    )
    protect_score = np.maximum.reduce(
        [
            runner_prob.fillna(0.0).to_numpy(dtype=float),
            strong_prob.fillna(0.0).to_numpy(dtype=float),
            take_ok_prob.fillna(0.0).to_numpy(dtype=float) * 0.8,
        ]
    )
    selected_mask = (
        ((should_prob >= float(selected_row["bad_threshold_v1"])) | (mae_prob >= float(selected_row["mae_threshold_v1"])) | (tail_prob >= float(selected_row["tail_threshold_v1"])))
        & (pd.Series(protect_score, index=frame.index) < float(selected_row["runner_protect_threshold_v1"]))
    ).fillna(False)
    prediction = frame[["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "split_scope_v1", "calendar_quarantine_status_v1"]].copy()
    prediction["r5_1_bad_blocker_score_v1"] = raw_bad_score
    prediction["r5_1_runner_guard_score_v1"] = protect_score
    prediction["r5_1_selected_candidate__block_v1"] = selected_mask.to_numpy(dtype=bool)
    selected = {
        "policy_name_v1": "MONDAY_R5_1_FOUNDATION_REBUILD_CALIBRATED_POLICY",
        "params_v1": {
            "bad_threshold_v1": float(selected_row["bad_threshold_v1"]),
            "mae_threshold_v1": float(selected_row["mae_threshold_v1"]),
            "tail_threshold_v1": float(selected_row["tail_threshold_v1"]),
            "runner_protect_threshold_v1": float(selected_row["runner_protect_threshold_v1"]),
        },
        "metrics_v1": _policy_metrics(frame, selected_mask),
        "hard_damage_count_v1": int(selected_row["hard_damage_count_v1"]),
    }
    return calibration, selected, prediction


def _calibrate_r5_2(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    bad_score = _num(frame, R5_2_BAD_PROB)
    runner_score = _num(frame, R5_2_RUNNER_PROB)
    rows: list[dict[str, Any]] = []
    bad_thresholds = sorted(
        set(
            [
                0.18,
                0.19,
                0.20,
                0.22,
                0.25,
                0.30,
                0.35,
                *[
                    float(bad_score.quantile(q))
                    for q in [0.75, 0.85, 0.90, 0.95, 0.98]
                    if pd.notna(bad_score.quantile(q))
                ],
            ]
        )
    )
    for bad_threshold in bad_thresholds:
        for runner_max in [0.16, 0.20, 0.25, 0.35, 0.45, 0.60]:
            mask = (bad_score >= bad_threshold) & (runner_score < runner_max)
            metrics = _policy_metrics(frame, mask)
            worst_loso = _worst_run_precision(frame, mask)
            hard_damage = _hard_damage_count(metrics)
            rows.append(
                {
                    "candidate_type_v1": "BASE_GRID",
                    "bad_threshold_v1": float(bad_threshold),
                    "runner_max_v1": float(runner_max),
                    "extension_applied_v1": False,
                    "extension_added_rows_v1": 0,
                    "hard_damage_count_v1": int(hard_damage),
                    "worst_loso_v1": worst_loso,
                    "wednesday_safety_pass_v1": bool(
                        int(metrics["block_count_v1"]) > 0
                        and (metrics["precision_v1"] is not None and float(metrics["precision_v1"]) >= WEDNESDAY_R6_BENCHMARK["precision_v1"])
                        and worst_loso is not None
                        and float(worst_loso) >= WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]
                        and hard_damage == 0
                    ),
                    "selection_score_v1": float(metrics["bad_blocks_v1"] * 3 + metrics["tail_help_v1"] - metrics["false_take_ok_blocks_v1"] * 30 - hard_damage * 120),
                    **metrics,
                }
            )
    calibration = pd.DataFrame(rows)
    nonzero = calibration[calibration["block_count_v1"] > 0].copy()
    safe_recall = nonzero[nonzero["wednesday_safety_pass_v1"].astype(bool)].copy()
    selectable = safe_recall if not safe_recall.empty else (nonzero if not nonzero.empty else calibration)
    selected_row = selectable.sort_values(
        ["hard_damage_count_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1"],
        ascending=[True, False, False, False],
        na_position="last",
    ).iloc[0].to_dict()
    selected_mask = (
        (bad_score >= float(selected_row["bad_threshold_v1"]))
        & (runner_score < float(selected_row["runner_max_v1"]))
    ).fillna(False)
    base_metrics = _policy_metrics(frame, selected_mask)
    original_base_mask = selected_mask.copy()
    v1_extension_mask = _r5_2_contract_extension_mask(frame, float(selected_row["bad_threshold_v1"]))
    v1_contract_mask = (original_base_mask | v1_extension_mask).fillna(False)
    v1_safety_pass, v1_worst_loso, v1_hard_damage = _wednesday_safety_pass(frame, v1_contract_mask)
    v1_metrics = _policy_metrics(frame, v1_contract_mask)
    v1_added_mask = v1_contract_mask & ~original_base_mask
    v1_row = {
        "candidate_type_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V1["contract_id_v1"],
        "bad_threshold_v1": float(selected_row["bad_threshold_v1"]),
        "runner_max_v1": float(selected_row["runner_max_v1"]),
        "extension_applied_v1": True,
        "extension_added_rows_v1": int(v1_added_mask.sum()),
        "hard_damage_count_v1": int(v1_hard_damage),
        "worst_loso_v1": v1_worst_loso,
        "wednesday_safety_pass_v1": bool(v1_safety_pass),
        "selection_score_v1": float(v1_metrics["bad_blocks_v1"] * 3 + v1_metrics["tail_help_v1"] - v1_metrics["false_take_ok_blocks_v1"] * 30 - v1_hard_damage * 120),
        **v1_metrics,
    }
    calibration = pd.concat([calibration, pd.DataFrame([v1_row])], ignore_index=True)
    v1_improves_or_holds = (
        int(v1_metrics["bad_blocks_v1"]) >= int(base_metrics["bad_blocks_v1"])
        and int(v1_metrics["tail_help_v1"]) >= int(base_metrics["tail_help_v1"])
    )
    v1_applied = bool(v1_safety_pass and v1_improves_or_holds and int(v1_added_mask.sum()) > 0)
    prior_to_v2_mask = v1_contract_mask if v1_applied else original_base_mask
    prior_to_v2_metrics = v1_metrics if v1_applied else base_metrics

    v2_extension_mask = _r5_2_contract_v2_extension_mask(frame)
    v2_contract_mask = (prior_to_v2_mask | v2_extension_mask).fillna(False)
    v2_safety_pass, v2_worst_loso, v2_hard_damage = _wednesday_safety_pass(frame, v2_contract_mask)
    v2_metrics = _policy_metrics(frame, v2_contract_mask)
    v2_added_mask = v2_contract_mask & ~prior_to_v2_mask
    v2_row = {
        "candidate_type_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"],
        "bad_threshold_v1": float(selected_row["bad_threshold_v1"]),
        "runner_max_v1": float(selected_row["runner_max_v1"]),
        "extension_applied_v1": True,
        "extension_added_rows_v1": int(v2_added_mask.sum()),
        "hard_damage_count_v1": int(v2_hard_damage),
        "worst_loso_v1": v2_worst_loso,
        "wednesday_safety_pass_v1": bool(v2_safety_pass),
        "selection_score_v1": float(v2_metrics["bad_blocks_v1"] * 3 + v2_metrics["tail_help_v1"] - v2_metrics["false_take_ok_blocks_v1"] * 30 - v2_hard_damage * 120),
        **v2_metrics,
    }
    calibration = pd.concat([calibration, pd.DataFrame([v2_row])], ignore_index=True)
    v2_improves_or_holds = (
        int(v2_metrics["bad_blocks_v1"]) >= int(prior_to_v2_metrics["bad_blocks_v1"])
        and int(v2_metrics["tail_help_v1"]) >= int(prior_to_v2_metrics["tail_help_v1"])
    )
    v2_applied = bool(v2_safety_pass and v2_improves_or_holds and int(v2_added_mask.sum()) > 0)
    prior_to_v3_mask = v2_contract_mask if v2_applied else (v1_contract_mask if v1_applied else original_base_mask)
    prior_to_v3_metrics = v2_metrics if v2_applied else (v1_metrics if v1_applied else base_metrics)

    v3_extension_mask = _r5_2_contract_v3_extension_mask(frame)
    v3_contract_mask = (prior_to_v3_mask | v3_extension_mask).fillna(False)
    v3_safety_pass, v3_worst_loso, v3_hard_damage = _wednesday_safety_pass(frame, v3_contract_mask)
    v3_metrics = _policy_metrics(frame, v3_contract_mask)
    v3_added_mask = v3_contract_mask & ~prior_to_v3_mask
    v3_row = {
        "candidate_type_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"],
        "bad_threshold_v1": float(selected_row["bad_threshold_v1"]),
        "runner_max_v1": float(selected_row["runner_max_v1"]),
        "extension_applied_v1": True,
        "extension_added_rows_v1": int(v3_added_mask.sum()),
        "hard_damage_count_v1": int(v3_hard_damage),
        "worst_loso_v1": v3_worst_loso,
        "wednesday_safety_pass_v1": bool(v3_safety_pass),
        "selection_score_v1": float(v3_metrics["bad_blocks_v1"] * 3 + v3_metrics["tail_help_v1"] - v3_metrics["false_take_ok_blocks_v1"] * 30 - v3_hard_damage * 120),
        **v3_metrics,
    }
    calibration = pd.concat([calibration, pd.DataFrame([v3_row])], ignore_index=True)
    v3_improves_or_holds = (
        int(v3_metrics["bad_blocks_v1"]) >= int(prior_to_v3_metrics["bad_blocks_v1"])
        and int(v3_metrics["tail_help_v1"]) >= int(prior_to_v3_metrics["tail_help_v1"])
    )
    v3_applied = bool(v3_safety_pass and v3_improves_or_holds and int(v3_added_mask.sum()) > 0)
    if v3_applied:
        selected_mask = v3_contract_mask
        selected_row = v3_row
    elif v2_applied:
        selected_mask = v2_contract_mask
        selected_row = v2_row
    elif v1_applied:
        selected_mask = v1_contract_mask
        selected_row = v1_row
    active_contract = R5_2_BASE_MEMBERSHIP_CONTRACT_V3 if v3_applied else (R5_2_BASE_MEMBERSHIP_CONTRACT_V2 if v2_applied else (R5_2_BASE_MEMBERSHIP_CONTRACT_V1 if v1_applied else None))
    final_added_vs_original = selected_mask & ~original_base_mask
    selected = {
        "policy_name_v1": "MONDAY_R5_2_FOUNDATION_REBUILD_CALIBRATED_POLICY",
        "params_v1": {
            "bad_threshold_v1": float(selected_row["bad_threshold_v1"]),
            "runner_max_v1": float(selected_row["runner_max_v1"]),
        },
        "metrics_v1": _policy_metrics(frame, selected_mask),
        "hard_damage_count_v1": int(selected_row["hard_damage_count_v1"]),
        "worst_loso_v1": selected_row.get("worst_loso_v1"),
        "wednesday_safety_pass_v1": bool(selected_row.get("wednesday_safety_pass_v1")),
        "base_metrics_before_contract_v1": base_metrics,
        "base_membership_contract_v1": active_contract,
        "base_membership_contracts_v1": [R5_2_BASE_MEMBERSHIP_CONTRACT_V1, R5_2_BASE_MEMBERSHIP_CONTRACT_V2, R5_2_BASE_MEMBERSHIP_CONTRACT_V3],
        "base_membership_contract_applied_v1": bool(v1_applied or v2_applied or v3_applied),
        "base_membership_active_contract_id_v1": active_contract["contract_id_v1"] if active_contract else None,
        "base_membership_contract_safety_pass_v1": bool(v3_safety_pass if v3_applied else (v2_safety_pass if v2_applied else v1_safety_pass)),
        "base_membership_contract_added_rows_v1": int(final_added_vs_original.sum()) if (v1_applied or v2_applied or v3_applied) else 0,
        "base_membership_contract_added_bad_blocks_v1": int((final_added_vs_original & _bool(frame, "label_should_not_take_v1")).sum()) if (v1_applied or v2_applied or v3_applied) else 0,
        "base_membership_contract_added_tail_help_v1": int((final_added_vs_original & _bool(frame, "tail_10_50_mfe_v1")).sum()) if (v1_applied or v2_applied or v3_applied) else 0,
        "v1_base_membership_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V1,
        "v1_contract_applied_v1": v1_applied,
        "v1_contract_safety_pass_v1": bool(v1_safety_pass),
        "v1_contract_metrics_v1": v1_metrics,
        "v1_contract_added_rows_v1": int(v1_added_mask.sum()) if v1_applied else 0,
        "v1_contract_added_bad_blocks_v1": int((v1_added_mask & _bool(frame, "label_should_not_take_v1")).sum()) if v1_applied else 0,
        "v1_contract_added_tail_help_v1": int((v1_added_mask & _bool(frame, "tail_10_50_mfe_v1")).sum()) if v1_applied else 0,
        "v2_base_membership_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
        "v2_contract_applied_v1": v2_applied,
        "v2_contract_safety_pass_v1": bool(v2_safety_pass),
        "v2_contract_metrics_v1": v2_metrics,
        "v2_contract_incremental_added_rows_v1": int(v2_added_mask.sum()) if v2_applied else 0,
        "v2_contract_incremental_added_bad_blocks_v1": int((v2_added_mask & _bool(frame, "label_should_not_take_v1")).sum()) if v2_applied else 0,
        "v2_contract_incremental_added_tail_help_v1": int((v2_added_mask & _bool(frame, "tail_10_50_mfe_v1")).sum()) if v2_applied else 0,
        "v2_contract_total_added_rows_vs_original_v1": int(final_added_vs_original.sum()) if v2_applied else 0,
        "v2_contract_total_added_bad_blocks_vs_original_v1": int((final_added_vs_original & _bool(frame, "label_should_not_take_v1")).sum()) if v2_applied else 0,
        "v2_contract_total_added_tail_help_vs_original_v1": int((final_added_vs_original & _bool(frame, "tail_10_50_mfe_v1")).sum()) if v2_applied else 0,
        "v3_base_membership_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
        "v3_contract_applied_v1": v3_applied,
        "v3_contract_safety_pass_v1": bool(v3_safety_pass),
        "v3_contract_metrics_v1": v3_metrics,
        "v3_contract_incremental_added_rows_v1": int(v3_added_mask.sum()) if v3_applied else 0,
        "v3_contract_incremental_added_bad_blocks_v1": int((v3_added_mask & _bool(frame, "label_should_not_take_v1")).sum()) if v3_applied else 0,
        "v3_contract_incremental_added_tail_help_v1": int((v3_added_mask & _bool(frame, "tail_10_50_mfe_v1")).sum()) if v3_applied else 0,
        "v3_contract_total_added_rows_vs_original_v1": int(final_added_vs_original.sum()) if v3_applied else 0,
        "v3_contract_total_added_bad_blocks_vs_original_v1": int((final_added_vs_original & _bool(frame, "label_should_not_take_v1")).sum()) if v3_applied else 0,
        "v3_contract_total_added_tail_help_vs_original_v1": int((final_added_vs_original & _bool(frame, "tail_10_50_mfe_v1")).sum()) if v3_applied else 0,
        "nonzero_candidate_count_v1": int(len(nonzero)),
        "safe_recall_candidate_count_v1": int(len(safe_recall)),
        "selection_order_v1": "wednesday_safety_then_bad_blocks_tail_help_precision_then_v1_mae_confirmed_extension_then_v2_safe_recall_extension_then_v3_score_agreement_extension",
    }
    return calibration, selected, selected_mask


def _feature_manifest(base_features: list[str], r5_2_features: list[str], score_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in base_features:
        rows.append({"feature_v1": feature, "stage_v1": "BASE_AS_OF", "used_by_r5_v1": True, "used_by_r5_2_v1": feature in r5_2_features})
    for feature in r5_2_features:
        if feature not in base_features:
            rows.append({"feature_v1": feature, "stage_v1": "REBUILT_SCORE_OR_POLICY", "used_by_r5_v1": False, "used_by_r5_2_v1": True})
    for column in score_columns:
        rows.append({"feature_v1": column, "stage_v1": "OUTPUT_SCORE", "used_by_r5_v1": False, "used_by_r5_2_v1": False})
    return pd.DataFrame(rows)


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("EXPLICIT_SCORE_REBUILD_FLAG", "PASS" if summary["explicit_score_rebuild_flag_v1"] else "FAIL", summary["explicit_score_rebuild_flag_v1"]),
            row("FOUNDATION_ROW_COUNT_1914", "PASS" if summary["row_count_v1"] == EXPECTED_FOUNDATION_ROWS else "FAIL", summary["row_count_v1"]),
            row("AS_OF_109", "PASS" if summary["as_of_column_count_v1"] == EXPECTED_AS_OF_COLUMNS else "FAIL", summary["as_of_column_count_v1"]),
            row("QUARANTINE_ROWS_HELD", "PASS" if summary["quarantine_rows_v1"] == EXPECTED_QUARANTINE_ROWS else "FAIL", summary["quarantine_rows_v1"]),
            row("R5_HEADS_REBUILT", "PASS" if summary["r5_head_count_v1"] == len(R5_HEADS) else "FAIL", summary["r5_head_count_v1"]),
            row("R5_1_POLICY_MATERIALIZED", "PASS" if summary["r5_1_policy_materialized_v1"] else "FAIL", summary["r5_1_policy_materialized_v1"]),
            row("R5_2_HEADS_REBUILT", "PASS" if summary["r5_2_head_count_v1"] == len(R5_2_HEADS) else "FAIL", summary["r5_2_head_count_v1"]),
            row("NO_R6_HEADS_TRAINED", "PASS" if summary["r6_heads_trained_v1"] is False else "FAIL", summary["r6_heads_trained_v1"]),
            row("NO_FREEZE_OR_PROMOTION", "PASS", summary["not_freeze_or_promo_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Foundation Score Rebuild V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rows: `{summary['row_count_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- R5 heads rebuilt: `{summary['r5_head_count_v1']}`",
            f"- R5.1 policy materialized: `{summary['r5_1_policy_materialized_v1']}`",
            f"- R5.2 heads rebuilt: `{summary['r5_2_head_count_v1']}`",
            f"- Score columns written: `{summary['score_column_count_v1']}`",
            "",
            "This is offline score rebuild only. No R6 heads, freeze, promotion, live gate, or controller change was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    foundation_dir: Path | None = None,
    output_dir: Path | None = None,
    run_score_rebuild: bool = False,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    foundation_dir = foundation_dir.expanduser().resolve() if foundation_dir else _latest_dir(reports_root, FOUNDATION_GLOB, FOUNDATION_FRAME)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_score_rebuild:
        status = {
            "layer_name": f"{LAYER_NAME}_STATUS",
            "decision_v1": "EXPLICIT_SCORE_REBUILD_FLAG_REQUIRED",
            "next_action_v1": "RUN_WITH_EXPLICIT_RUN_SCORE_REBUILD_FLAG",
            "training_started_v1": False,
            "score_rebuild_started_v1": False,
        }
        _write_json(output_dir / STATUS, status)
        return status

    frame = pd.read_parquet(foundation_dir / FOUNDATION_FRAME)
    asof = pd.read_parquet(foundation_dir / FOUNDATION_AS_OF)
    foundation_summary = _read_json(foundation_dir / FOUNDATION_SUMMARY)
    _validate_foundation(foundation_dir, frame, asof, foundation_summary)

    train_mask = _bool(frame, "used_for_training")
    validation_mask = _bool(frame, "used_for_validation")
    active = frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    base_features = _base_feature_names(frame)

    r5_pred, r5_metrics = _train_head_group(
        frame=frame,
        head_specs=R5_HEADS,
        feature_names=base_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_foundation_r5_score_seed",
        stage="r5",
        seed=config.seed,
        config=config,
    )
    scored = frame.drop(columns=[output for _, output in R5_HEADS.values()], errors="ignore").merge(r5_pred, on="candidate_uid", how="left", validate="one_to_one")
    r5_1_calibration, r5_1_selected, r5_1_pred = _calibrate_r5_1(scored)
    scored = scored.merge(
        r5_1_pred[["candidate_uid", "r5_1_bad_blocker_score_v1", "r5_1_runner_guard_score_v1", "r5_1_selected_candidate__block_v1"]],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    scored = _derive_r5_2_and_r6_labels(scored)
    r5_2_features = [
        *base_features,
        *[output for _, output in R5_HEADS.values()],
        "r5_1_bad_blocker_score_v1",
        "r5_1_runner_guard_score_v1",
        "r5_1_selected_candidate__block_v1",
    ]
    r5_2_pred, r5_2_metrics = _train_head_group(
        frame=scored,
        head_specs=R5_2_HEADS,
        feature_names=r5_2_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_foundation_r5_2_score_seed",
        stage="r5_2",
        seed=config.seed + 101,
        config=config,
    )
    scored = scored.drop(columns=[R5_2_BAD_PROB, R5_2_RUNNER_PROB], errors="ignore").merge(r5_2_pred, on="candidate_uid", how="left", validate="one_to_one")
    scored = _derive_r5_2_and_r6_labels(scored)

    r5_block = (_num(scored, "pred__entry_r5_should_not_take__prob_true_v1") >= 0.5) & (_num(scored, "pred__entry_r5_runner_protect__prob_true_v1") < 0.5)
    scored["r5_selected_candidate__block_v1"] = r5_block.fillna(False).astype(bool)
    r5_2_calibration, r5_2_selected, r5_2_selected_mask = _calibrate_r5_2(scored)
    scored["r5_2_selected_candidate__block_v1"] = r5_2_selected_mask.to_numpy(dtype=bool)
    scored["blocker_score_v1"] = _num(scored, R5_2_BAD_PROB)
    scored["runner_protector_score_v1"] = _num(scored, R5_2_RUNNER_PROB)

    id_cols = ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "split_scope_v1", "calendar_quarantine_status_v1"]
    r5_cols = [*id_cols, *[output for _, output in R5_HEADS.values()], "r5_selected_candidate__block_v1"]
    r5_2_cols = [*id_cols, R5_2_BAD_PROB, R5_2_RUNNER_PROB, "r5_2_selected_candidate__block_v1", "blocker_score_v1", "runner_protector_score_v1"]
    r5_prediction = scored[[column for column in r5_cols if column in scored.columns]].copy()
    r5_2_prediction = scored[[column for column in r5_2_cols if column in scored.columns]].copy()
    score_columns = [
        *[output for _, output in R5_HEADS.values()],
        "r5_selected_candidate__block_v1",
        "r5_1_bad_blocker_score_v1",
        "r5_1_runner_guard_score_v1",
        "r5_1_selected_candidate__block_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "r5_2_selected_candidate__block_v1",
        "blocker_score_v1",
        "runner_protector_score_v1",
    ]
    metrics = pd.concat(
        [
            r5_metrics.assign(stage_v1="R5_SCORE_REBUILD"),
            r5_2_metrics.assign(stage_v1="R5_2_SCORE_REBUILD"),
        ],
        ignore_index=True,
    )
    score_summary = {
        "r5_selected_policy_metrics_v1": _policy_metrics(scored, scored["r5_selected_candidate__block_v1"]),
        "r5_1_selected_policy_v1": r5_1_selected,
        "r5_2_selected_policy_v1": r5_2_selected,
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
        "next_action_v1": "RUN_MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_WITH_EXPLICIT_FLAG",
        "explicit_score_rebuild_flag_v1": True,
        "training_started_v1": True,
        "score_rebuild_started_v1": True,
        "row_count_v1": int(len(scored)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "train_rows_v1": int(train_mask.sum()),
        "validation_rows_v1": int(validation_mask.sum()),
        "holdout_rows_v1": int(_bool(scored, "used_for_holdout").sum()),
        "as_of_column_count_v1": int(asof.shape[1]),
        "base_feature_count_v1": int(len(base_features)),
        "r5_2_feature_count_v1": int(len(r5_2_features)),
        "r5_head_count_v1": int(len(R5_HEADS)),
        "r5_1_policy_materialized_v1": True,
        "r5_2_head_count_v1": int(len(R5_2_HEADS)),
        "r6_heads_trained_v1": False,
        "score_column_count_v1": int(len(score_columns)),
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_freeze_or_promo_v1": True,
        "blocked_action_v1": [
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_SCORE_REBUILD",
            "DO_NOT_TREAT_R5_R5_1_R5_2_SCORE_REBUILD_AS_FINAL_R6",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_ACTIVE_ONLY_1852_AS_FULL_FOUNDATION",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "R5 score heads were rebuilt on Monday actual 1914-row foundation.",
                "R5.1 calibrated score/policy layer was materialized.",
                "R5.2 score heads were rebuilt using Monday foundation plus R5/R5.1 scores.",
                "No R6 heads, freeze, promotion, live gate, or controller change was run.",
            ],
            "INDIKERT": [
                "This score package is ready as input to the explicit Monday R6 rebuild stage.",
            ],
            "IKKE_ETABLERT": [
                "A green Monday R6 candidate is not established by this score rebuild alone.",
                "Frozen Wednesday source hashes are still not restored locally.",
            ],
        },
    }
    audit = _audit(summary)
    feature_manifest = _feature_manifest(base_features, r5_2_features, score_columns)
    model_manifest = {
        "layer_name": f"{LAYER_NAME}_MODEL_MANIFEST",
        "model_families_v1": ["monday_foundation_r5_score_seed", "monday_foundation_r5_2_score_seed"],
        "r5_heads_v1": R5_HEADS,
        "r5_1_policy_v1": r5_1_selected,
        "r5_2_policy_v1": r5_2_selected,
        "r5_2_heads_v1": R5_2_HEADS,
        "not_live_gate_v1": True,
    }
    config_manifest = {
        "layer_name": f"{LAYER_NAME}_CONFIG_MANIFEST",
        "config_v1": config.__dict__,
        "explicit_score_rebuild_flag_v1": True,
        "foundation_dir_v1": str(foundation_dir),
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "training_started_v1": True,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
    }

    scored.to_parquet(output_dir / SCORE_FRAME, index=False)
    r5_prediction.to_parquet(output_dir / R5_PREDICTION_VIEW, index=False)
    r5_1_pred.to_parquet(output_dir / R5_1_PREDICTION_VIEW, index=False)
    r5_2_prediction.to_parquet(output_dir / R5_2_PREDICTION_VIEW, index=False)
    metrics.to_csv(output_dir / MODEL_METRICS, index=False)
    r5_1_calibration.to_csv(output_dir / R5_1_CALIBRATION, index=False)
    r5_2_calibration.to_csv(output_dir / R5_2_CALIBRATION, index=False)
    feature_manifest.to_csv(output_dir / FEATURE_MANIFEST, index=False)
    audit.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(output_dir / SCORE_SUMMARY, score_summary)
    _write_json(
        output_dir / R5_2_BASE_MEMBERSHIP_CONTRACT_FILE,
        {
            "active_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT,
            "v1_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V1,
            "v2_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
            "v3_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
            "contract_lineage_v1": [
                R5_2_BASE_MEMBERSHIP_CONTRACT_V1["contract_id_v1"],
                R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"],
                R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"],
            ],
        },
    )
    _write_json(output_dir / MODEL_MANIFEST, model_manifest)
    _write_json(output_dir / CONFIG_MANIFEST, config_manifest)
    _write_json(output_dir / STATUS, {"layer_name": f"{LAYER_NAME}_STATUS", "decision_v1": summary["decision_v1"], "next_action_v1": summary["next_action_v1"], "training_started_v1": True})
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--foundation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-score-rebuild", action="store_true")
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        foundation_dir=args.foundation_dir,
        output_dir=args.output_dir,
        run_score_rebuild=args.run_score_rebuild,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
