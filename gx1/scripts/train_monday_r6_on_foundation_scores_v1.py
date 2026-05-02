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
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_HEAD_SPECS,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    TrainConfig,
    WEDNESDAY_R6_BENCHMARK,
    _asof_runner_guard,
    _base_feature_names,
    _bool,
    _calibrate_policy,
    _calibration_safety_summary,
    _compare,
    _jsonable,
    _policy_metrics,
    _r6_candidate_grid,
    _r6_family_grid_replay,
    _r6_feature_names,
    _r6_policy_mask,
    _safety_failure_rows,
    _train_r6_heads,
    _wednesday_locked_policy_replay,
    _worst_run_precision,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1"
SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"

SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
SCORE_SUMMARY = "summary_v1.json"

TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
PREDICTION_VIEW = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
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

EXPECTED_SCORE_ROWS = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_AS_OF_COLUMNS = 109
EXPECTED_BASE_FEATURES = 88
EXPECTED_R5_HEADS = 8
EXPECTED_R5_2_HEADS = 2
FORBIDDEN_BASELINE_ROWS = {1689, 1852}

FOUNDATION_SCORE_CONTEXT_COLUMNS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_strong_trade_candidate__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "pred__entry_r5_take_was_ok__prob_true_v1",
    "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1",
    "pred__entry_r5_wait_or_delay_advisory__prob_true_v1",
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

R6_LABEL_COLUMNS = [spec.label_col for spec in R6_HEAD_SPECS]
R6_OUTPUT_COLUMNS = [spec.output_col for spec in R6_HEAD_SPECS]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _latest_dir(reports_root: Path, pattern: str, required_file: str) -> Path:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    if not dirs:
        raise FileNotFoundError(f"No {pattern} with {required_file} under {reports_root}")
    return dirs[-1]


def _validate_score_package(score_dir: Path, frame: pd.DataFrame, score_summary: dict[str, Any]) -> None:
    if len(frame) in FORBIDDEN_BASELINE_ROWS:
        raise RuntimeError(f"Refuses forbidden narrow/active-only row count: {len(frame)}")
    if int(len(frame)) != EXPECTED_SCORE_ROWS:
        raise RuntimeError(f"Expected Monday R6 foundation score rows {EXPECTED_SCORE_ROWS}, observed {len(frame)}")
    if score_summary.get("decision_v1") != "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED":
        raise RuntimeError(f"Score package is not green: {score_dir} decision={score_summary.get('decision_v1')}")
    if int(score_summary.get("as_of_column_count_v1") or 0) != EXPECTED_AS_OF_COLUMNS:
        raise RuntimeError(f"Expected {EXPECTED_AS_OF_COLUMNS} AS_OF columns in score summary, observed {score_summary.get('as_of_column_count_v1')}")
    if int(score_summary.get("base_feature_count_v1") or 0) != EXPECTED_BASE_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_BASE_FEATURES} base features, observed {score_summary.get('base_feature_count_v1')}")
    if int(score_summary.get("r5_head_count_v1") or 0) != EXPECTED_R5_HEADS:
        raise RuntimeError(f"Expected {EXPECTED_R5_HEADS} R5 heads, observed {score_summary.get('r5_head_count_v1')}")
    if int(score_summary.get("r5_2_head_count_v1") or 0) != EXPECTED_R5_2_HEADS:
        raise RuntimeError(f"Expected {EXPECTED_R5_2_HEADS} R5.2 heads, observed {score_summary.get('r5_2_head_count_v1')}")
    if bool(score_summary.get("r6_heads_trained_v1")):
        raise RuntimeError("Score package must not already contain trained R6 heads")
    active = frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    if int(active.sum()) != EXPECTED_ACTIVE_ROWS or int((~active).sum()) != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, observed {int(active.sum())}/{int((~active).sum())}")
    required = [
        "candidate_uid",
        "run_id",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        *R6_LABEL_COLUMNS,
        *FOUNDATION_SCORE_CONTEXT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Foundation score frame missing required columns: {missing}")


def _foundation_r6_feature_names(frame: pd.DataFrame) -> list[str]:
    names = _r6_feature_names(frame)
    for column in FOUNDATION_SCORE_CONTEXT_COLUMNS:
        if column in frame.columns and column not in names:
            names.append(column)
    return names


def _verdict_from_compare(compare: dict[str, Any]) -> str:
    verdict = str(compare.get("verdict_v1") or "")
    if verdict.endswith("IMPROVES_AND_HOLDS_WEDNESDAY_SAFETY"):
        return "MONDAY_R6_ON_FOUNDATION_SCORES_IMPROVES_AND_HOLDS_WEDNESDAY_SAFETY"
    if verdict.endswith("SAFE_BUT_NOT_BETTER"):
        return "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER"
    if verdict.endswith("FAILED_WEDNESDAY_SAFETY"):
        return "MONDAY_R6_ON_FOUNDATION_SCORES_RAN_BUT_FAILED_WEDNESDAY_SAFETY"
    return "MONDAY_R6_ON_FOUNDATION_SCORES_NOT_ESTABLISHED"


def _family_grid_safe_candidate(frame: pd.DataFrame, grid: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.Series | None, dict[str, Any] | None]:
    if grid.empty or "wednesday_safety_pass_v1" not in grid.columns:
        return None, None, None
    safe = grid[grid["wednesday_safety_pass_v1"].astype(bool)].copy()
    if safe.empty:
        return None, None, None
    selected_row = safe.sort_values(
        ["bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False],
        na_position="last",
    ).iloc[0].to_dict()
    policy_name = str(selected_row["policy_name_v1"])
    candidate = next((item for item in _r6_candidate_grid(compact=False) if item.policy_name == policy_name), None)
    if candidate is None:
        return None, None, None
    mask = _r6_policy_mask(frame, candidate)
    metrics = _policy_metrics(frame, mask)
    worst_loso = _worst_run_precision(frame, mask)
    compare = _compare(metrics, worst_loso)
    selected = {
        "policy_name_v1": policy_name,
        "policy_source_v1": "R6_FAMILY_GRID_SAFE_CANDIDATE",
        "family_v1": candidate.family,
        "params_v1": {
            "bad_threshold_v1": candidate.bad_threshold,
            "runner_threshold_v1": candidate.runner_threshold,
            "tail_threshold_v1": candidate.tail_threshold,
            "risky_threshold_v1": candidate.risky_threshold,
            "blindspot_threshold_v1": candidate.blindspot_threshold,
            "r5_2_runner_threshold_v1": candidate.r5_2_runner_threshold,
            "use_r5_2_base_v1": candidate.use_r5_2_base,
            "hard_asof_runner_guard_v1": candidate.hard_asof_runner_guard,
        },
        "metrics_v1": metrics,
        "candidate_worst_loso_v1": worst_loso,
        "compare_v1": compare,
    }
    return selected, mask, compare


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("EXPLICIT_R6_REBUILD_FLAG", "PASS" if summary["explicit_r6_rebuild_flag_v1"] else "FAIL", summary["explicit_r6_rebuild_flag_v1"]),
            row("ROW_COUNT_1914", "PASS" if summary["row_count_v1"] == EXPECTED_SCORE_ROWS else "FAIL", summary["row_count_v1"]),
            row("ROW_COUNT_NOT_1689_OR_1852", "PASS" if summary["row_count_v1"] not in FORBIDDEN_BASELINE_ROWS else "FAIL", summary["row_count_v1"]),
            row("AS_OF_SCHEMA_109", "PASS" if summary["as_of_column_count_v1"] == EXPECTED_AS_OF_COLUMNS else "FAIL", summary["as_of_column_count_v1"]),
            row("BASE_FEATURES_88", "PASS" if summary["base_feature_count_v1"] == EXPECTED_BASE_FEATURES else "FAIL", summary["base_feature_count_v1"]),
            row("FOUNDATION_SCORE_CONTEXT_PRESENT", "PASS" if summary["foundation_score_context_column_count_v1"] == len(FOUNDATION_SCORE_CONTEXT_COLUMNS) else "FAIL", summary["foundation_score_context_column_count_v1"]),
            row("R6_HEADS_TRAINED", "PASS" if summary["r6_head_count_v1"] == len(R6_HEAD_SPECS) else "FAIL", summary["r6_head_count_v1"]),
            row("NO_FREEZE_OR_PROMOTION", "PASS", summary["not_freeze_or_promo_v1"]),
            row("NO_LIVE_GATE", "PASS", summary["not_live_gate_v1"]),
            row("QUARANTINE_EVAL_ONLY", "PASS" if summary["quarantine_rows_v1"] == summary["quarantine_holdout_rows_v1"] else "FAIL", summary["quarantine_rows_v1"]),
        ]
    )


def _report(summary: dict[str, Any], compare: dict[str, Any]) -> str:
    metrics = compare["candidate_metrics_v1"]
    return "\n".join(
        [
            "# Monday R6 Rebuild On Foundation Scores V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Compare verdict: `{compare['verdict_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rows: `{summary['row_count_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- Base features: `{summary['base_feature_count_v1']}`",
            f"- Foundation score context columns: `{summary['foundation_score_context_column_count_v1']}`",
            f"- R6 feature count: `{summary['r6_feature_count_v1']}`",
            f"- R6 heads trained: `{summary['r6_head_count_v1']}`",
            f"- Selected policy source: `{summary['selected_policy_source_v1']}`",
            f"- Bad blocks: `{metrics['bad_blocks_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['bad_blocks_v1']}`",
            f"- Tail help: `{metrics['tail_help_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['tail_help_v1']}`",
            f"- Precision: `{metrics['precision_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['precision_v1']}`",
            f"- Worst run precision: `{compare['candidate_worst_loso_v1']}` vs Wednesday `{WEDNESDAY_R6_BENCHMARK['worst_loso_v1']}`",
            f"- Safety failure rows: `{summary['safety_failure_row_count_v1']}`",
            "",
            "This is an explicit offline Monday R6 rebuild on the actual foundation score package. It is not freeze, promotion, live gate, or controller execution.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    score_dir: Path | None = None,
    output_dir: Path | None = None,
    run_r6_rebuild: bool = False,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    score_dir = score_dir.expanduser().resolve() if score_dir else _latest_dir(reports_root, SCORE_GLOB, SCORE_FRAME)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_r6_rebuild:
        status = {
            "layer_name": f"{LAYER_NAME}_STATUS",
            "decision_v1": "EXPLICIT_R6_REBUILD_FLAG_REQUIRED",
            "next_action_v1": "RUN_WITH_EXPLICIT_RUN_R6_REBUILD_FLAG",
            "training_started_v1": False,
            "r6_training_started_v1": False,
        }
        _write_json(output_dir / STATUS, status)
        return status

    frame = pd.read_parquet(score_dir / SCORE_FRAME)
    score_summary = _read_json(score_dir / SCORE_SUMMARY)
    _validate_score_package(score_dir, frame, score_summary)

    train_mask = _bool(frame, "used_for_training")
    validation_mask = _bool(frame, "used_for_validation")
    holdout_mask = _bool(frame, "used_for_holdout")
    quarantine = frame.get("calendar_quarantine_status_v1", pd.Series("", index=frame.index)).astype("string").eq("QUARANTINED")
    base_features = _base_feature_names(frame)
    r6_features = _foundation_r6_feature_names(frame)

    r6_pred, r6_metrics = _train_r6_heads(
        frame=frame.drop(columns=R6_OUTPUT_COLUMNS, errors="ignore"),
        feature_names=r6_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=output_dir,
        model_tag="monday_r6_on_foundation_scores_five_head",
        n_estimators=config.r6_n_estimators,
        early_stopping_rounds=config.r6_early_stopping_rounds,
        learning_rate=config.r6_learning_rate,
        max_depth=config.max_depth,
        seed=config.seed + 303,
        n_jobs=config.n_jobs,
    )
    frame = frame.drop(columns=R6_OUTPUT_COLUMNS, errors="ignore").merge(r6_pred, on="candidate_uid", how="left", validate="one_to_one")

    calibration_df, custom_selected, custom_selected_mask = _calibrate_policy(frame)
    custom_metrics = _policy_metrics(frame, custom_selected_mask)
    custom_worst_loso = _worst_run_precision(frame, custom_selected_mask)
    custom_compare = _compare(custom_metrics, custom_worst_loso)
    calibration_safety = _calibration_safety_summary(calibration_df, custom_selected, custom_compare)
    wednesday_locked_replay = _wednesday_locked_policy_replay(frame)
    r6_family_grid, r6_family_grid_summary = _r6_family_grid_replay(frame)
    family_selected, family_mask, family_compare = _family_grid_safe_candidate(frame, r6_family_grid)
    selected_policy_source = "CUSTOM_THRESHOLD_GRID"
    selected = custom_selected
    selected_mask = custom_selected_mask
    compare = custom_compare
    if custom_compare["verdict_v1"].endswith("FAILED_WEDNESDAY_SAFETY") and family_selected is not None and family_mask is not None and family_compare is not None:
        selected_policy_source = "R6_FAMILY_GRID_SAFE_CANDIDATE"
        selected = family_selected
        selected_mask = family_mask
        compare = family_compare
    metrics = _policy_metrics(frame, selected_mask)
    safety_failure_rows = _safety_failure_rows(frame, selected_mask)
    decision = _verdict_from_compare(compare)

    frame["selected_candidate_block_v1"] = selected_mask.to_numpy(dtype=bool)
    frame["asof_runner_guard_v1"] = _asof_runner_guard(frame, selected["params_v1"]).to_numpy(dtype=bool)
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
        "selected_candidate_block_v1",
        "asof_runner_guard_v1",
    ]
    prediction_view = frame[[column for column in pred_view_cols if column in frame.columns]].copy()

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "score_dir_v1": str(score_dir),
        "score_source_decision_v1": score_summary.get("decision_v1"),
        "explicit_r6_rebuild_flag_v1": True,
        "training_started_v1": True,
        "r6_training_started_v1": True,
        "row_count_v1": int(len(frame)),
        "active_rows_v1": int((~quarantine).sum()),
        "quarantine_rows_v1": int(quarantine.sum()),
        "quarantine_holdout_rows_v1": int((quarantine & holdout_mask).sum()),
        "train_rows_v1": int(train_mask.sum()),
        "validation_rows_v1": int(validation_mask.sum()),
        "holdout_rows_v1": int(holdout_mask.sum()),
        "as_of_column_count_v1": int(score_summary.get("as_of_column_count_v1") or EXPECTED_AS_OF_COLUMNS),
        "base_feature_count_v1": int(len(base_features)),
        "foundation_score_context_column_count_v1": int(sum(column in frame.columns for column in FOUNDATION_SCORE_CONTEXT_COLUMNS)),
        "r6_feature_count_v1": int(len(r6_features)),
        "r6_head_count_v1": int(len(R6_HEAD_SPECS)),
        "threshold_grid_candidate_count_v1": int(calibration_safety["grid_candidate_count_v1"]),
        "custom_threshold_grid_wednesday_safety_candidate_count_v1": int(calibration_safety["wednesday_safety_candidate_count_v1"]),
        "custom_threshold_grid_wednesday_safety_and_better_candidate_count_v1": int(calibration_safety["wednesday_safety_and_better_candidate_count_v1"]),
        "family_grid_wednesday_safety_candidate_count_v1": int(r6_family_grid_summary["wednesday_safety_candidate_count_v1"]),
        "selected_policy_wednesday_safety_pass_v1": bool(compare["verdict_v1"] != "MONDAY_R6_EXPLICIT_REBUILD_RAN_BUT_FAILED_WEDNESDAY_SAFETY"),
        "selected_policy_improves_bad_blocks_and_tail_help_v1": bool(compare["improves_bad_blocks_and_tail_help_v1"]),
        "wednesday_locked_policy_replay_v1": {
            "r5_2_base_block_count_v1": wednesday_locked_replay["r5_2_base_block_count_v1"],
            "r6_addon_block_count_v1": wednesday_locked_replay["r6_addon_block_count_v1"],
            "wednesday_safety_pass_v1": wednesday_locked_replay["wednesday_safety_pass_v1"],
            "safety_failures_v1": wednesday_locked_replay["compare_v1"]["safety_failures_v1"],
        },
        "r6_family_grid_replay_v1": r6_family_grid_summary,
        "safety_failure_row_count_v1": int(len(safety_failure_rows)),
        "decision_v1": decision,
        "compare_verdict_v1": compare["verdict_v1"],
        "failed_safety_checks_v1": compare["safety_failures_v1"],
        "selected_policy_source_v1": selected_policy_source,
        "custom_threshold_grid_policy_v1": {
            "selected_candidate_v1": custom_selected,
            "compare_v1": custom_compare,
        },
        "family_grid_selected_policy_v1": family_selected,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_freeze_or_promo_v1": True,
        "blocked_action_v1": [
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_R6_REBUILD",
            "DO_NOT_CHANGE_LIVE_OR_CONTROLLER",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_ACTIVE_ONLY_1852_AS_FULL_FOUNDATION",
        ],
        "next_action_v1": (
            "REVIEW_MONDAY_R6_FOR_CANONICAL_LOCK"
            if decision == "MONDAY_R6_ON_FOUNDATION_SCORES_IMPROVES_AND_HOLDS_WEDNESDAY_SAFETY"
            else "INVESTIGATE_MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK"
            if decision == "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER"
            else "FIX_MONDAY_R6_SAFETY_OR_LABEL_FEATURE_CONTRACT_FIRST"
        ),
        "hard_status_v1": {
            "BEVIST": [
                "Monday R6 five-head layer was rebuilt on the actual 1914-row foundation score package.",
                "The rebuild used 109 AS_OF schema count, 88 base features, and the full foundation score context.",
                "No freeze, promotion, live gate, or controller change was run.",
            ],
            "INDIKERT": [
                "The compare package determines whether this rebuilt Monday R6 can proceed to canonical lock review.",
            ],
            "IKKE_ETABLERT": [
                "Canonical/live Monday R6 is not established by training alone.",
                "Frozen Wednesday source hashes are still not restored locally.",
            ],
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "training_started_v1": True,
        "r6_training_started_v1": True,
        "promotion_status_v1": "NOT_PROMOTED_NOT_LIVE_GATE",
        "decision_v1": decision,
        "failed_safety_checks_v1": compare["safety_failures_v1"],
        "blocked_action_v1": summary["blocked_action_v1"],
        "next_action_v1": summary["next_action_v1"],
    }
    score_features = [column for column in FOUNDATION_SCORE_CONTEXT_COLUMNS if column in frame.columns]
    feature_manifest = pd.DataFrame(
        [{"feature_v1": feature, "stage_v1": "BASE_AS_OF_OR_COVERAGE", "used_by_r6_v1": feature in r6_features} for feature in base_features]
        + [{"feature_v1": feature, "stage_v1": "FOUNDATION_SCORE_CONTEXT", "used_by_r6_v1": feature in r6_features} for feature in score_features]
        + [{"feature_v1": feature, "stage_v1": "R6_OUTPUT_SCORE", "used_by_r6_v1": False} for feature in R6_OUTPUT_COLUMNS]
    )
    model_manifest = {
        "layer_name": f"{LAYER_NAME}_MODEL_MANIFEST",
        "model_tags_v1": ["monday_r6_on_foundation_scores_five_head"],
        "heads_v1": {"r6_v1": [spec.__dict__ for spec in R6_HEAD_SPECS]},
        "selected_policy_v1": selected,
        "not_live_gate_v1": True,
    }
    config_manifest = {
        "layer_name": f"{LAYER_NAME}_CONFIG",
        "training_config_v1": config.__dict__,
        "selected_candidate_v1": selected,
        "fixed_seed_v1": config.seed,
        "score_dir_v1": str(score_dir),
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
    r6_metrics.assign(stage_v1="R6_FIVE_HEAD_ON_FOUNDATION_SCORES").to_csv(output_dir / MODEL_METRICS, index=False)
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
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-r6-rebuild", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None, help="Optional compact override for R6 heads.")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    config = TrainConfig(n_jobs=args.n_jobs)
    if args.n_estimators is not None:
        config = TrainConfig(
            r6_n_estimators=args.n_estimators,
            r6_early_stopping_rounds=min(60, max(20, args.n_estimators // 10)),
            n_jobs=args.n_jobs,
        )
    summary = materialize(
        reports_root=args.reports_root,
        score_dir=args.score_dir,
        output_dir=args.output_dir,
        run_r6_rebuild=args.run_r6_rebuild,
        config=config,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
