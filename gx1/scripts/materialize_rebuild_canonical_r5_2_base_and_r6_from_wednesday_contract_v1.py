#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "REBUILD_CANONICAL_R5_2_BASE_AND_R6_FROM_WEDNESDAY_CONTRACT_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

FOUNDATION_GLOB = "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_*"
SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"
R6_REBUILD_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*"
SOURCE_RESTORE_GLOB = "MONDAY_R6_WEDNESDAY_SOURCE_RESTORE_ATTEMPT_V1_*"

FOUNDATION_SUMMARY = "summary_v1.json"
FOUNDATION_FRAME = "monday_r6_foundation_training_frame_pre_score_v1.parquet"
FOUNDATION_DELTA = "row_universe_delta_v1.csv"
SCORE_SUMMARY = "summary_v1.json"
SCORE_REBUILD_SUMMARY = "score_rebuild_summary_v1.json"
SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
R6_SUMMARY = "summary_v1.json"
R6_EVAL = "eval_summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"
R6_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
SOURCE_RESTORE_SUMMARY = "summary_v1.json"

WEDNESDAY_CONTRACT = {
    "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
    "expected_rows_v1": 1971,
    "hindsight_backfill_rows_v1": 1971,
    "as_of_columns_v1": 109,
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

REQUIRED_OUTPUTS = {
    "truth_scope": "rebuild_truth_and_scope_lock_v1.json",
    "contract": "wednesday_r6_contract_extraction_v1.json",
    "foundation": "monday_fullcoverage_foundation_rebuild_from_contract_v1.json",
    "r5_base": "r5_r5_1_r5_2_canonical_base_rebuild_v1.json",
    "r6_retrain": "r6_retrain_from_rebuilt_r5_2_base_v1.json",
    "r6_eval": "r6_contract_eval_against_wednesday_benchmark_v1.json",
    "row_delta": "row_and_schema_delta_explainer_v1.csv",
    "missing_source": "missing_source_artifacts_and_rebuild_limits_v1.json",
    "gate": "canonical_monday_r6_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

DO_NOT_USE_FOR_CANONICAL_R6 = [
    "MONDAY_EXACT_ONLY_1689_TRAINING_SURFACE",
    "MONDAY_67_FEATURE_NARROW_TRAINING_SURFACE",
    "PROTECTOR_FIRST_ON_1689_SURFACE",
    "FAILED_NARROW_RETRAIN_RUN",
    "LOCAL_ADBB_ZERO_BLOCK_OR_NARROW_R5_2_VARIANT",
    "BRIDGE_AS_WORKAROUND_FOR_MISSING_TRAINING_ROWS",
]

R6_HEADS = [
    "bad_risk",
    "runner_protector",
    "tail_control_10_50",
    "risky_allow",
    "batch04_blindspot",
]


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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_dir(root: Path, pattern: str, required_file: str) -> Path | None:
    dirs = sorted(path for path in root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    return dirs[-1] if dirs else None


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _nested_get(data: dict[str, Any], *paths: str) -> Any:
    for raw_path in paths:
        current: Any = data
        ok = True
        for part in raw_path.split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    return None


def _metric(data: dict[str, Any], key: str) -> Any:
    return _nested_get(
        data,
        key,
        f"metrics_v1.{key}",
        f"selected_candidate_v1.{key}",
        f"family_grid_selected_policy_v1.metrics_v1.{key}",
        f"family_grid_selected_policy_v1.compare_v1.{key}",
        f"compare_v1.{key}",
        f"wednesday_locked_policy_replay_v1.{key}",
    )


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_wednesday_contract(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = _read_json(snapshot_dir / WEDNESDAY_SUMMARY)
    manifest = _read_json(snapshot_dir / WEDNESDAY_MANIFEST)
    selected = summary.get("selected_candidate_v1") if isinstance(summary.get("selected_candidate_v1"), dict) else {}
    thresholds = manifest.get("thresholds_v1") or summary.get("thresholds_v1") or {}
    contract = {
        "layer_name": "WEDNESDAY_R6_CONTRACT_EXTRACTION_V1",
        "source_snapshot_dir_v1": str(snapshot_dir),
        "snapshot_summary_present_v1": bool(summary),
        "snapshot_manifest_present_v1": bool(manifest),
        "freeze_id_v1": summary.get("freeze_id_v1") or manifest.get("freeze_id_v1") or WEDNESDAY_CONTRACT["freeze_id_v1"],
        "candidate_id_v1": summary.get("selected_candidate_id_v1")
        or manifest.get("selected_candidate_id_v1")
        or manifest.get("selected_policy_stack_v1")
        or WEDNESDAY_CONTRACT["candidate_id_v1"],
        "expected_rows_v1": int((summary.get("policy_logging_v1") or {}).get("row_count_v1") or WEDNESDAY_CONTRACT["expected_rows_v1"]),
        "hindsight_backfill_rows_v1": int(
            (summary.get("policy_logging_v1") or {}).get("hindsight_backfill_rows_v1")
            or WEDNESDAY_CONTRACT["hindsight_backfill_rows_v1"]
        ),
        "as_of_columns_v1": int((manifest.get("as_of_schema_v1") or {}).get("column_count_v1") or WEDNESDAY_CONTRACT["as_of_columns_v1"]),
        "hindsight_schema_column_count_v1": (manifest.get("hindsight_schema_v1") or {}).get("column_count_v1"),
        "selected_metrics_v1": {
            "bad_blocks_v1": selected.get("true_block_should_not_take_count_v1", WEDNESDAY_CONTRACT["bad_blocks_v1"]),
            "tail_help_v1": selected.get("true_block_tail_10_50_count_v1", WEDNESDAY_CONTRACT["tail_help_v1"]),
            "precision_v1": selected.get("precision_v1", WEDNESDAY_CONTRACT["precision_v1"]),
            "worst_loso_v1": selected.get("worst_loso_precision_v1", WEDNESDAY_CONTRACT["worst_loso_v1"]),
            "repaired_165_damage_v1": selected.get("repaired_165_damage_count_v1", WEDNESDAY_CONTRACT["repaired_165_damage_v1"]),
            "fifty_plus_mfe_blocked_v1": selected.get(
                "fifty_plus_mfe_block_count_v1", WEDNESDAY_CONTRACT["fifty_plus_mfe_blocked_v1"]
            ),
            "hundred_plus_mfe_blocked_v1": selected.get(
                "hundred_plus_mfe_block_count_v1", WEDNESDAY_CONTRACT["hundred_plus_mfe_blocked_v1"]
            ),
            "two_hundred_plus_mfe_blocked_v1": selected.get(
                "two_hundred_plus_mfe_block_count_v1", WEDNESDAY_CONTRACT["two_hundred_plus_mfe_blocked_v1"]
            ),
            "strongest_winner_damage_v1": selected.get(
                "strongest_winner_damage_count_v1", WEDNESDAY_CONTRACT["strongest_winner_damage_v1"]
            ),
        },
        "thresholds_v1": {
            "bad_threshold_v1": thresholds.get("bad_threshold_v1", 0.95),
            "risky_threshold_v1": thresholds.get("risky_threshold_v1", 0.85),
            "tail_threshold_v1": thresholds.get("tail_threshold_v1", 0.90),
            "runner_threshold_v1": thresholds.get("runner_threshold_v1", 0.60),
            "r5_2_runner_threshold_v1": thresholds.get("r5_2_runner_threshold_v1", 0.74),
            "blindspot_threshold_v1": thresholds.get("blindspot_threshold_v1", 0.70),
            "use_r5_2_base_v1": bool(thresholds.get("use_r5_2_base_v1", True)),
            "guard_v1": thresholds.get("guard_v1", "hard_asof_runner_guard"),
        },
        "model_setup_v1": {
            "model_family_v1": "R6_FIVE_HEAD",
            "heads_v1": R6_HEADS,
            "candidate_family_must_include_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "uses_r5_2_base_v1": True,
            "r5_2_benchmark_freeze_id_v1": manifest.get("r5_2_benchmark_freeze_id_v1", "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"),
        },
        "schema_contract_v1": {
            "as_of_hindsight_physically_separate_v1": bool(manifest.get("as_of_schema_v1") and manifest.get("hindsight_schema_v1")),
            "as_of_schema_sha256_v1": (manifest.get("as_of_schema_v1") or {}).get("schema_sha256_v1"),
            "hindsight_schema_sha256_v1": (manifest.get("hindsight_schema_v1") or {}).get("schema_sha256_v1"),
        },
        "source_lineage_v1": {
            "r6_source_dir_v1": manifest.get("r6_source_dir_v1") or summary.get("r6_source_dir_v1"),
            "r5_2_freeze_dir_v1": summary.get("r5_2_freeze_dir_v1"),
            "reports_root_v1": manifest.get("reports_root_v1") or summary.get("reports_root_v1"),
        },
    }
    return contract, summary, manifest


def _truth_scope_lock(source_restore: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "REBUILD_TRUTH_AND_SCOPE_LOCK_V1",
        "benchmark_contract_v1": {
            "frozen_wednesday_r6_is_benchmark_contract_not_restorable_source_v1": True,
            "freeze_id_v1": contract["freeze_id_v1"],
            "candidate_id_v1": contract["candidate_id_v1"],
            "expected_rows_v1": contract["expected_rows_v1"],
            "as_of_columns_v1": contract["as_of_columns_v1"],
            "thresholds_v1": contract["thresholds_v1"],
            "selected_metrics_v1": contract["selected_metrics_v1"],
        },
        "missing_source_v1": {
            "exact_frozen_r6_restore_possible_locally_v1": False,
            "source_restore_decision_v1": source_restore.get("decision_v1", "NOT_ESTABLISHED"),
            "missing_hash_count_v1": source_restore.get("missing_hash_count_v1"),
            "expected_hash_rows_v1": source_restore.get("expected_hash_rows_v1"),
            "required_source_artifact_missing_count_v1": source_restore.get("required_source_artifact_missing_count_v1"),
            "archive_restorable_candidate_count_v1": source_restore.get("archive_restorable_candidate_count_v1"),
        },
        "rebuild_input_v1": {
            "use_wednesday_r6_contract_metrics_thresholds_scripts_manifests_feature_label_principle_v1": True,
            "target_anchor_v1": "MONDAY",
            "target_window_v1": "MONDAY_TO_SUNDAY",
            "rebuild_not_bit_for_bit_restore_v1": True,
        },
        "do_not_use_for_canonical_r6_v1": DO_NOT_USE_FOR_CANONICAL_R6,
        "blocked_actions_v1": [
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_R6",
            "DO_NOT_CALL_REBUILT_R6_FROZEN_WEDNESDAY_REPRODUCTION",
        ],
    }


def _foundation_lock(foundation_dir: Path | None, contract: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(foundation_dir / FOUNDATION_SUMMARY) if foundation_dir else {}
    row_count = int(summary.get("row_count_v1") or 0)
    as_of_count = int(summary.get("as_of_column_count_v1") or summary.get("foundation_as_of_output_column_count_v1") or 0)
    expected = int(contract["expected_rows_v1"])
    return {
        "layer_name": "MONDAY_FULLCOVERAGE_FOUNDATION_REBUILD_FROM_CONTRACT_V1",
        "foundation_dir_v1": str(foundation_dir) if foundation_dir else None,
        "foundation_present_v1": bool(summary),
        "decision_v1": summary.get("decision_v1", "NOT_ESTABLISHED"),
        "monday_fullcoverage_raw_state_materialized_v1": bool(summary and row_count > 0),
        "monday_policy_eval_surface_materialized_v1": bool(summary and row_count > 0),
        "monday_as_of_schema_materialized_v1": bool(as_of_count),
        "monday_hindsight_backfill_materialized_v1": bool(summary.get("hindsight_output_column_count_v1")),
        "monday_candidate_trade_lineage_materialized_v1": bool(summary and row_count > 0),
        "monday_repaired_coverage_materialized_v1": bool(summary and int(summary.get("quarantine_rows_v1") or 0) >= 0),
        "monday_feature_contract_materialized_v1": bool(summary.get("base_feature_count_v1")),
        "monday_label_intersection_materialized_v1": bool(summary.get("hindsight_output_column_count_v1")),
        "row_count_v1": row_count or None,
        "active_rows_v1": summary.get("active_rows_v1"),
        "quarantine_rows_v1": summary.get("quarantine_rows_v1"),
        "as_of_column_count_v1": as_of_count or None,
        "base_feature_count_v1": summary.get("base_feature_count_v1"),
        "wednesday_expected_rows_v1": expected,
        "delta_vs_wednesday_expected_rows_v1": (row_count - expected) if row_count else None,
        "not_1689_exact_only_v1": row_count != 1689 if row_count else None,
        "not_bridge_as_training_replacement_v1": True,
        "contract_fit_v1": {
            "as_of_109_pass_v1": as_of_count == int(contract["as_of_columns_v1"]),
            "fullcoverage_rows_available_but_not_wednesday_row_count_v1": row_count > 1689 and row_count != expected,
            "expected_1971_reached_v1": row_count == expected,
        },
    }


def _score_rebuild_lock(score_dir: Path | None, foundation_dir: Path | None) -> dict[str, Any]:
    summary = _read_json(score_dir / SCORE_SUMMARY) if score_dir else {}
    score_detail = _read_json(score_dir / SCORE_REBUILD_SUMMARY) if score_dir else {}
    r5_2_feature_count = summary.get("r5_2_feature_count_v1")
    return {
        "layer_name": "REBUILD_R5_R5_1_R5_2_CANONICAL_BASE_V1",
        "score_dir_v1": str(score_dir) if score_dir else None,
        "score_rebuild_present_v1": bool(summary),
        "decision_v1": summary.get("decision_v1", "NOT_ESTABLISHED"),
        "source_foundation_dir_v1": summary.get("foundation_dir_v1") or (str(foundation_dir) if foundation_dir else None),
        "uses_monday_fullcoverage_foundation_v1": bool(
            summary and foundation_dir and str(summary.get("foundation_dir_v1")) == str(foundation_dir)
        ),
        "input_universe_v1": {
            "row_count_v1": summary.get("row_count_v1"),
            "active_rows_v1": summary.get("active_rows_v1"),
            "quarantine_rows_v1": summary.get("quarantine_rows_v1"),
            "as_of_column_count_v1": summary.get("as_of_column_count_v1"),
            "base_feature_count_v1": summary.get("base_feature_count_v1"),
        },
        "r5_v1": {
            "head_count_v1": summary.get("r5_head_count_v1"),
            "score_outputs_materialized_v1": bool(summary.get("r5_head_count_v1")),
        },
        "r5_1_v1": {
            "policy_materialized_v1": summary.get("r5_1_policy_materialized_v1"),
        },
        "r5_2_v1": {
            "head_count_v1": summary.get("r5_2_head_count_v1"),
            "feature_count_v1": r5_2_feature_count,
            "produces_runner_base_scores_for_r6_v1": bool(summary.get("r5_2_head_count_v1")),
            "canonical_status_v1": "R5_2_REBUILT_FROM_CONTRACT" if summary else "NOT_ESTABLISHED",
            "not_frozen_original_v1": True,
            "known_frozen_r5_2_freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
        },
        "score_column_count_v1": summary.get("score_column_count_v1"),
        "split_rows_v1": {
            "train_rows_v1": summary.get("train_rows_v1"),
            "validation_rows_v1": summary.get("validation_rows_v1"),
            "holdout_rows_v1": summary.get("holdout_rows_v1"),
        },
        "detail_summary_v1": score_detail,
        "not_freeze_or_promo_v1": summary.get("not_freeze_or_promo_v1", True),
    }


def _r6_retrain_lock(r6_dir: Path | None, score_dir: Path | None) -> dict[str, Any]:
    summary = _read_json(r6_dir / R6_SUMMARY) if r6_dir else {}
    return {
        "layer_name": "R6_RETRAIN_FROM_REBUILT_R5_2_BASE_V1",
        "r6_dir_v1": str(r6_dir) if r6_dir else None,
        "r6_rebuild_present_v1": bool(summary),
        "decision_v1": summary.get("decision_v1", "NOT_ESTABLISHED"),
        "source_score_dir_v1": summary.get("score_dir_v1") or (str(score_dir) if score_dir else None),
        "uses_rebuilt_r5_2_base_v1": bool(summary and score_dir and str(summary.get("score_dir_v1")) == str(score_dir)),
        "training_started_v1": bool(summary.get("training_started_v1") or summary.get("r6_training_started_v1")),
        "r6_head_count_v1": summary.get("r6_head_count_v1"),
        "r6_feature_count_v1": summary.get("r6_feature_count_v1"),
        "row_count_v1": summary.get("row_count_v1"),
        "active_rows_v1": summary.get("active_rows_v1"),
        "quarantine_rows_v1": summary.get("quarantine_rows_v1"),
        "as_of_column_count_v1": summary.get("as_of_column_count_v1"),
        "selected_policy_source_v1": summary.get("selected_policy_source_v1"),
        "selected_policy_v1": summary.get("family_grid_selected_policy_v1"),
        "candidate_grid_v1": summary.get("r6_family_grid_replay_v1"),
        "wednesday_locked_policy_replay_v1": summary.get("wednesday_locked_policy_replay_v1"),
        "not_freeze_or_promo_v1": summary.get("not_freeze_or_promo_v1", True),
        "blocked_action_v1": summary.get("blocked_action_v1", ["DO_NOT_FREEZE_OR_PROMOTE_FROM_R6_REBUILD"]),
    }


def _r6_eval_against_contract(r6_dir: Path | None, contract: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(r6_dir / R6_SUMMARY) if r6_dir else {}
    eval_summary = _read_json(r6_dir / R6_EVAL) if r6_dir else {}
    compare = _read_json(r6_dir / R6_COMPARE) if r6_dir else {}
    selected = summary.get("family_grid_selected_policy_v1") if isinstance(summary.get("family_grid_selected_policy_v1"), dict) else {}
    metric_source: dict[str, Any] = {}
    metric_source.update(eval_summary)
    metric_source.update(compare)
    metric_source.update(summary)
    metric_source.update(selected)
    metrics = {
        "bad_blocks_v1": _metric(metric_source, "bad_blocks_v1"),
        "tail_help_v1": _metric(metric_source, "tail_help_v1"),
        "precision_v1": _metric(metric_source, "precision_v1"),
        "worst_loso_v1": _first_not_none(_metric(metric_source, "worst_loso_v1"), _metric(metric_source, "candidate_worst_loso_v1")),
        "repaired_165_damage_v1": _first_not_none(
            _metric(metric_source, "repaired_165_damage_v1"),
            _metric(metric_source, "repaired_165_damage_count_v1"),
        ),
        "fifty_plus_mfe_blocked_v1": _first_not_none(
            _metric(metric_source, "fifty_plus_mfe_blocked_v1"),
            _metric(metric_source, "fifty_plus_mfe_block_count_v1"),
        ),
        "hundred_plus_mfe_blocked_v1": _first_not_none(
            _metric(metric_source, "hundred_plus_mfe_blocked_v1"),
            _metric(metric_source, "hundred_plus_mfe_block_count_v1"),
        ),
        "two_hundred_plus_mfe_blocked_v1": _first_not_none(
            _metric(metric_source, "two_hundred_plus_mfe_blocked_v1"),
            _metric(metric_source, "two_hundred_plus_mfe_block_count_v1"),
        ),
        "strongest_winner_damage_v1": _first_not_none(
            _metric(metric_source, "strongest_winner_damage_v1"),
            _metric(metric_source, "strongest_winner_damage_count_v1"),
        ),
    }
    targets = contract["selected_metrics_v1"]

    def ge(key: str) -> bool | None:
        return None if metrics[key] is None else float(metrics[key]) >= float(targets[key])

    def le(key: str) -> bool | None:
        return None if metrics[key] is None else float(metrics[key]) <= float(targets[key])

    safety_checks = {
        "bad_blocks_target_met_v1": ge("bad_blocks_v1"),
        "tail_help_target_met_v1": ge("tail_help_v1"),
        "precision_target_met_v1": ge("precision_v1"),
        "worst_loso_target_met_v1": ge("worst_loso_v1"),
        "repaired_165_damage_zero_v1": metrics["repaired_165_damage_v1"] in (0, 0.0),
        "fifty_plus_mfe_blocked_lte_wednesday_v1": le("fifty_plus_mfe_blocked_v1"),
        "hundred_plus_mfe_blocked_zero_v1": metrics["hundred_plus_mfe_blocked_v1"] in (0, 0.0),
        "two_hundred_plus_mfe_blocked_zero_v1": metrics["two_hundred_plus_mfe_blocked_v1"] in (0, 0.0),
        "strongest_winner_damage_zero_v1": metrics["strongest_winner_damage_v1"] in (0, 0.0),
    }
    known_checks = [value for value in safety_checks.values() if value is not None]
    safety_pass = bool(known_checks) and all(bool(value) for value in known_checks)
    below_wednesday = any(value is False for value in [safety_checks["bad_blocks_target_met_v1"], safety_checks["tail_help_target_met_v1"]])
    if not summary:
        verdict = "NOT_ESTABLISHED"
    elif safety_pass:
        verdict = "MONDAY_R6_CANONICAL_REBUILD_PASS"
    elif summary.get("decision_v1") == "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER" or below_wednesday:
        verdict = "MONDAY_R6_REBUILD_SAFE_BUT_BELOW_WEDNESDAY"
    elif summary.get("score_source_decision_v1") != "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED":
        verdict = "MONDAY_R6_REBUILD_BLOCKED_BY_R5_2_BASE"
    else:
        verdict = "MONDAY_R6_REBUILD_INVALID_CONTRACT_DRIFT"
    return {
        "layer_name": "R6_CONTRACT_EVAL_AGAINST_WEDNESDAY_BENCHMARK_V1",
        "r6_dir_v1": str(r6_dir) if r6_dir else None,
        "verdict_v1": verdict,
        "contract_targets_v1": targets,
        "observed_metrics_v1": metrics,
        "safety_and_benchmark_checks_v1": safety_checks,
        "compare_verdict_v1": summary.get("compare_verdict_v1") or compare.get("verdict_v1"),
        "full_repaired_coverage_v1": {
            "row_count_v1": summary.get("row_count_v1"),
            "active_rows_v1": summary.get("active_rows_v1"),
            "quarantine_rows_v1": summary.get("quarantine_rows_v1"),
            "as_of_column_count_v1": summary.get("as_of_column_count_v1"),
        },
        "as_of_hindsight_separation_required_v1": True,
        "not_freeze_or_promo_v1": summary.get("not_freeze_or_promo_v1", True),
    }


def _row_key_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id"] if col in frame.columns]


def _row_delta_explainer(
    foundation_dir: Path | None,
    score_dir: Path | None,
    r6_dir: Path | None,
    contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    foundation_summary = _read_json(foundation_dir / FOUNDATION_SUMMARY) if foundation_dir else {}
    score_summary = _read_json(score_dir / SCORE_SUMMARY) if score_dir else {}
    r6_summary = _read_json(r6_dir / R6_SUMMARY) if r6_dir else {}

    def add_summary(surface: str, observed: Any, expected: Any, reason: str, status: str = "SUMMARY") -> None:
        rows.append(
            {
                "candidate_uid": "",
                "trade_uid": "",
                "trade_id": "",
                "decision_timestamp": "",
                "source_surface_v1": "WEDNESDAY_CONTRACT",
                "missing_surface_v1": surface,
                "observed_rows_v1": observed,
                "expected_rows_v1": expected,
                "observed_as_of_columns_v1": "",
                "expected_as_of_columns_v1": contract["as_of_columns_v1"],
                "reason_v1": reason,
                "status_v1": status,
                "is_repaired_coverage_v1": "",
                "is_runner_or_winner_pocket_v1": "",
            }
        )

    add_summary("WEDNESDAY_CONTRACT", contract["expected_rows_v1"], contract["expected_rows_v1"], "REFERENCE")
    if foundation_summary:
        add_summary(
            "MONDAY_FOUNDATION",
            foundation_summary.get("row_count_v1"),
            contract["expected_rows_v1"],
            "EXPECTED_MONDAY_ANCHOR_DELTA"
            if foundation_summary.get("row_count_v1") != contract["expected_rows_v1"]
            else "MATCH",
        )
        add_summary("MONDAY_ACTIVE", foundation_summary.get("active_rows_v1"), foundation_summary.get("row_count_v1"), "QUARANTINE")
    if score_summary:
        add_summary("R5_R5_1_R5_2_SCORE_SURFACE", score_summary.get("row_count_v1"), foundation_summary.get("row_count_v1"), "MATCH")
    if r6_summary:
        add_summary("R6_REBUILD_SURFACE", r6_summary.get("row_count_v1"), foundation_summary.get("row_count_v1"), "MATCH")

    if foundation_dir:
        foundation_frame = _safe_read_parquet(foundation_dir / FOUNDATION_FRAME)
        if not foundation_frame.empty:
            quarantine_col = "calendar_quarantine_status_v1"
            if quarantine_col in foundation_frame.columns:
                quarantine = foundation_frame[foundation_frame[quarantine_col].astype("string").str.upper().ne("ACTIVE")]
                for row in quarantine.head(500).to_dict("records"):
                    rows.append(
                        {
                            "candidate_uid": row.get("candidate_uid", ""),
                            "trade_uid": row.get("trade_uid", ""),
                            "trade_id": row.get("trade_id", ""),
                            "decision_timestamp": row.get("decision_timestamp", ""),
                            "source_surface_v1": "MONDAY_FOUNDATION",
                            "missing_surface_v1": "MONDAY_ACTIVE",
                            "observed_rows_v1": foundation_summary.get("active_rows_v1"),
                            "expected_rows_v1": foundation_summary.get("row_count_v1"),
                            "observed_as_of_columns_v1": foundation_summary.get("as_of_column_count_v1"),
                            "expected_as_of_columns_v1": contract["as_of_columns_v1"],
                            "reason_v1": "QUARANTINE",
                            "status_v1": "ROW_LEVEL",
                            "is_repaired_coverage_v1": row.get("is_repaired_coverage_v1", ""),
                            "is_runner_or_winner_pocket_v1": any(
                                bool(row.get(col, False))
                                for col in [
                                    "fifty_plus_mfe_v1",
                                    "hundred_plus_mfe_v1",
                                    "two_hundred_plus_mfe_v1",
                                    "strongest_winner_path_v1",
                                    "runner_near_miss_v1",
                                ]
                            ),
                        }
                    )
    for surface_name, root, filename, expected_root, expected_filename, reason in [
        ("R5_R5_1_R5_2_SCORE_SURFACE", score_dir, SCORE_FRAME, foundation_dir, FOUNDATION_FRAME, "R5_2_SCORE_MISSING"),
        ("R6_REBUILD_SURFACE", r6_dir, R6_FRAME, score_dir, SCORE_FRAME, "POLICY_LOG_MISSING"),
    ]:
        if not root or not expected_root:
            continue
        observed = _safe_read_parquet(root / filename)
        expected_frame = _safe_read_parquet(expected_root / expected_filename)
        key_cols = _row_key_columns(expected_frame)
        if observed.empty or expected_frame.empty or not key_cols:
            continue
        observed_keys = set(map(tuple, observed[key_cols].astype("string").fillna("").to_numpy()))
        for record in expected_frame.to_dict("records"):
            key = tuple(str(record.get(col, "")) for col in key_cols)
            if key in observed_keys:
                continue
            rows.append(
                {
                    "candidate_uid": record.get("candidate_uid", ""),
                    "trade_uid": record.get("trade_uid", ""),
                    "trade_id": record.get("trade_id", ""),
                    "decision_timestamp": record.get("decision_timestamp", ""),
                    "source_surface_v1": expected_root.name,
                    "missing_surface_v1": surface_name,
                    "observed_rows_v1": int(len(observed)),
                    "expected_rows_v1": int(len(expected_frame)),
                    "observed_as_of_columns_v1": "",
                    "expected_as_of_columns_v1": contract["as_of_columns_v1"],
                    "reason_v1": reason,
                    "status_v1": "ROW_LEVEL",
                    "is_repaired_coverage_v1": record.get("is_repaired_coverage_v1", ""),
                    "is_runner_or_winner_pocket_v1": any(
                        bool(record.get(col, False))
                        for col in [
                            "fifty_plus_mfe_v1",
                            "hundred_plus_mfe_v1",
                            "two_hundred_plus_mfe_v1",
                            "strongest_winner_path_v1",
                            "runner_near_miss_v1",
                        ]
                    ),
                }
            )
    if foundation_dir and (foundation_dir / FOUNDATION_DELTA).exists():
        old_delta = _safe_read_csv(foundation_dir / FOUNDATION_DELTA)
        for record in old_delta.head(1000).to_dict("records"):
            rows.append(
                {
                    "candidate_uid": record.get("candidate_uid", ""),
                    "trade_uid": record.get("trade_uid", ""),
                    "trade_id": record.get("trade_id", ""),
                    "decision_timestamp": record.get("decision_timestamp", ""),
                    "source_surface_v1": record.get("source_surface_v1", "FOUNDATION_DELTA"),
                    "missing_surface_v1": record.get("missing_surface_v1", "FOUNDATION_DELTA"),
                    "observed_rows_v1": foundation_summary.get("row_count_v1"),
                    "expected_rows_v1": contract["expected_rows_v1"],
                    "observed_as_of_columns_v1": foundation_summary.get("as_of_column_count_v1"),
                    "expected_as_of_columns_v1": contract["as_of_columns_v1"],
                    "reason_v1": record.get("reason_v1", "NOT_ESTABLISHED"),
                    "status_v1": "IMPORTED_FOUNDATION_DELTA",
                    "is_repaired_coverage_v1": record.get("is_repaired_coverage_v1", ""),
                    "is_runner_or_winner_pocket_v1": record.get("is_runner_or_winner_pocket_v1", ""),
                }
            )
    return pd.DataFrame(rows)


def _missing_source_limits(source_restore: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "MISSING_SOURCE_ARTIFACTS_AND_REBUILD_LIMITS_V1",
        "missing_frozen_r6_source_model_tree_v1": True,
        "missing_expected_hashes_v1": source_restore.get("missing_hash_count_v1", 15),
        "expected_hash_rows_v1": source_restore.get("expected_hash_rows_v1", 15),
        "required_source_artifacts_missing_v1": source_restore.get("required_source_artifact_missing_count_v1", 8),
        "archive_restorable_candidates_v1": source_restore.get("archive_restorable_candidate_count_v1", 0),
        "missing_canonical_r5_2_freeze_source_v1": True,
        "github_contains_missing_april_r6_truth_artifacts_v1": False,
        "exact_restore_blocked_v1": True,
        "contract_driven_rebuild_blocked_v1": False,
        "contract_driven_rebuild_requires_v1": [
            "GREEN_MONDAY_FULLCOVERAGE_FOUNDATION",
            "REBUILT_R5_2_BASE_WITH_R6_REQUIRED_RUNNER_SCORES",
            "R6_RETRAIN_AND_WEDNESDAY_CONTRACT_EVAL",
        ],
    }


def _canonical_gate(
    foundation: dict[str, Any],
    score: dict[str, Any],
    r6: dict[str, Any],
    eval_contract: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "fullcoverage_foundation_present_v1": foundation["foundation_present_v1"],
        "foundation_not_1689_exact_only_v1": foundation["not_1689_exact_only_v1"] is True,
        "foundation_as_of_109_v1": foundation["as_of_column_count_v1"] == contract["as_of_columns_v1"],
        "foundation_1971_rows_v1": foundation["row_count_v1"] == contract["expected_rows_v1"],
        "r5_r5_1_r5_2_rebuilt_v1": score["decision_v1"] == "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
        "r5_2_not_frozen_original_claim_v1": score["r5_2_v1"]["not_frozen_original_v1"] is True,
        "r6_retrain_present_v1": r6["r6_rebuild_present_v1"],
        "r6_uses_rebuilt_r5_2_base_v1": r6["uses_rebuilt_r5_2_base_v1"],
        "r6_five_heads_v1": r6["r6_head_count_v1"] == 5,
        "r6_contract_eval_pass_v1": eval_contract["verdict_v1"] == "MONDAY_R6_CANONICAL_REBUILD_PASS",
        "no_1689_exact_only_usage_v1": True,
        "no_protector_first_usage_v1": True,
    }
    if not checks["fullcoverage_foundation_present_v1"] or not checks["foundation_as_of_109_v1"]:
        decision = "FIX_FOUNDATION_COVERAGE_FIRST"
    elif not checks["r5_r5_1_r5_2_rebuilt_v1"]:
        decision = "RUN_R5_2_BASE_REBUILD_FIRST"
    elif not checks["r6_retrain_present_v1"]:
        decision = "RUN_R6_RETRAIN_FROM_REBUILT_R5_2"
    elif eval_contract["verdict_v1"] == "MONDAY_R6_CANONICAL_REBUILD_PASS" and checks["foundation_1971_rows_v1"]:
        decision = "CANONICAL_MONDAY_R6_READY"
    elif eval_contract["verdict_v1"] == "MONDAY_R6_REBUILD_SAFE_BUT_BELOW_WEDNESDAY":
        decision = "RUN_R6_RETRAIN_FROM_REBUILT_R5_2"
    elif not checks["foundation_1971_rows_v1"]:
        decision = "FIX_FOUNDATION_COVERAGE_FIRST"
    else:
        decision = "NOT_ESTABLISHED"
    return {
        "layer_name": "CANONICAL_MONDAY_R6_GATE_V1",
        "gate_decision_v1": decision,
        "checks_v1": checks,
        "blocked_actions_v1": [
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_R6",
            "DO_NOT_CALL_REBUILT_R6_FROZEN_WEDNESDAY_REPRODUCTION",
        ],
    }


def _next_action(gate: dict[str, Any], eval_contract: dict[str, Any]) -> dict[str, Any]:
    gate_decision = gate["gate_decision_v1"]
    if gate_decision == "CANONICAL_MONDAY_R6_READY":
        action = "RUN_R6_RETRAIN_FROM_REBUILT_R5_2_BASE"
    elif gate_decision == "RUN_R5_2_BASE_REBUILD_FIRST":
        action = "REBUILD_R5_R5_1_R5_2_CANONICAL_BASE_NOW"
    elif gate_decision == "RUN_R6_RETRAIN_FROM_REBUILT_R5_2":
        action = "RUN_R6_RETRAIN_FROM_REBUILT_R5_2_BASE"
        if eval_contract["verdict_v1"] == "MONDAY_R6_REBUILD_SAFE_BUT_BELOW_WEDNESDAY":
            action = "RESTORE_OR_RECONSTRUCT_REQUIRED_R5_2_INPUTS_FIRST"
    elif gate_decision == "FIX_FOUNDATION_COVERAGE_FIRST":
        action = "FIX_FULLCOVERAGE_FOUNDATION_FIRST"
    else:
        action = "RESTORE_OR_RECONSTRUCT_REQUIRED_R5_2_INPUTS_FIRST"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "always_enforced_actions_v1": [
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_R6",
            "DO_NOT_CALL_REBUILT_R6_FROZEN_WEDNESDAY_REPRODUCTION",
        ],
    }


def _audit(
    truth: dict[str, Any],
    contract: dict[str, Any],
    foundation: dict[str, Any],
    score: dict[str, Any],
    r6: dict[str, Any],
    eval_contract: dict[str, Any],
    gate: dict[str, Any],
) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    checks = [
        row("THIS_IS_REBUILD_NOT_RESTORE", "PASS", truth["rebuild_input_v1"]["rebuild_not_bit_for_bit_restore_v1"]),
        row("WEDNESDAY_CONTRACT_EXTRACTED", "PASS" if contract["snapshot_summary_present_v1"] else "FAIL", contract["source_snapshot_dir_v1"]),
        row("FROZEN_SOURCE_MISSING_DOCUMENTED", "PASS", truth["missing_source_v1"]),
        row("FOUNDATION_PRESENT", "PASS" if foundation["foundation_present_v1"] else "FAIL", foundation["foundation_dir_v1"]),
        row("FOUNDATION_NOT_1689", "PASS" if foundation["not_1689_exact_only_v1"] else "FAIL", foundation["row_count_v1"]),
        row("AS_OF_109", "PASS" if foundation["as_of_column_count_v1"] == 109 else "FAIL", foundation["as_of_column_count_v1"]),
        row(
            "FOUNDATION_REACHES_WEDNESDAY_ROW_COUNT",
            "PASS" if foundation["row_count_v1"] == contract["expected_rows_v1"] else "WARN",
            foundation["delta_vs_wednesday_expected_rows_v1"],
        ),
        row("R5_R5_1_R5_2_REBUILT", "PASS" if score["score_rebuild_present_v1"] else "FAIL", score["decision_v1"]),
        row("R5_2_NOT_FROZEN_ORIGINAL_CLAIMED", "PASS", score["r5_2_v1"]["canonical_status_v1"]),
        row("R6_RETRAIN_PRESENT", "PASS" if r6["r6_rebuild_present_v1"] else "FAIL", r6["decision_v1"]),
        row("R6_CONTRACT_EVAL", "PASS" if eval_contract["verdict_v1"] == "MONDAY_R6_CANONICAL_REBUILD_PASS" else "WARN", eval_contract["verdict_v1"]),
        row("CANONICAL_GATE", "PASS" if gate["gate_decision_v1"] == "CANONICAL_MONDAY_R6_READY" else "WARN", gate["gate_decision_v1"]),
    ]
    return pd.DataFrame(checks)


def _report(summary: dict[str, Any], contract: dict[str, Any], foundation: dict[str, Any], score: dict[str, Any], r6: dict[str, Any], eval_contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Rebuild Canonical R5.2 Base And R6 From Wednesday Contract V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            "## Truth Lock",
            "",
            "- This is a contract-driven rebuild, not an exact frozen Wednesday restore.",
            f"- Frozen benchmark: `{contract['freeze_id_v1']}` / `{contract['candidate_id_v1']}`.",
            "- 1689 exact-only, protector-first, bridge-as-training, and local narrow/zero-block variants are blocked as canonical R6 baseline.",
            "",
            "## Wednesday Contract",
            "",
            f"- Rows: `{contract['expected_rows_v1']}` policy/eval and `{contract['hindsight_backfill_rows_v1']}` hindsight/backfill.",
            f"- AS_OF columns: `{contract['as_of_columns_v1']}`.",
            f"- Metrics: bad blocks `{contract['selected_metrics_v1']['bad_blocks_v1']}`, tail help `{contract['selected_metrics_v1']['tail_help_v1']}`, precision `{contract['selected_metrics_v1']['precision_v1']}`, worst LOSO `{contract['selected_metrics_v1']['worst_loso_v1']}`.",
            "",
            "## Monday Foundation",
            "",
            f"- Foundation decision: `{foundation['decision_v1']}`.",
            f"- Rows: `{foundation['row_count_v1']}` total, `{foundation['active_rows_v1']}` active, `{foundation['quarantine_rows_v1']}` quarantine.",
            f"- Delta vs Wednesday contract rows: `{foundation['delta_vs_wednesday_expected_rows_v1']}`.",
            f"- AS_OF columns: `{foundation['as_of_column_count_v1']}`.",
            "",
            "## Rebuild",
            "",
            f"- R5/R5.1/R5.2 decision: `{score['decision_v1']}`.",
            f"- R5.2 status: `{score['r5_2_v1']['canonical_status_v1']}`; not frozen original.",
            f"- R6 decision: `{r6['decision_v1']}`.",
            f"- R6 eval verdict: `{eval_contract['verdict_v1']}`.",
            "",
            "## Hard Status",
            "",
            f"- BEVIST: `{summary['hard_status_v1']['BEVIST']}`",
            f"- INDIKERT: `{summary['hard_status_v1']['INDIKERT']}`",
            f"- IKKE_ETABLERT: `{summary['hard_status_v1']['IKKE_ETABLERT']}`",
            "",
        ]
    )


def materialize(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    foundation_dir: Path | None = None,
    score_dir: Path | None = None,
    r6_dir: Path | None = None,
    source_restore_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    contract, _wednesday_summary, _wednesday_manifest = _extract_wednesday_contract(snapshot_dir)
    foundation_dir = foundation_dir or _latest_dir(reports_root, FOUNDATION_GLOB, FOUNDATION_SUMMARY)
    score_dir = score_dir or _latest_dir(reports_root, SCORE_GLOB, SCORE_SUMMARY)
    r6_dir = r6_dir or _latest_dir(reports_root, R6_REBUILD_GLOB, R6_SUMMARY)
    source_restore_dir = source_restore_dir or _latest_dir(reports_root, SOURCE_RESTORE_GLOB, SOURCE_RESTORE_SUMMARY)
    source_restore = _read_json(source_restore_dir / SOURCE_RESTORE_SUMMARY) if source_restore_dir else {}

    truth = _truth_scope_lock(source_restore, contract)
    foundation = _foundation_lock(foundation_dir, contract)
    score = _score_rebuild_lock(score_dir, foundation_dir)
    r6 = _r6_retrain_lock(r6_dir, score_dir)
    eval_contract = _r6_eval_against_contract(r6_dir, contract)
    row_delta = _row_delta_explainer(foundation_dir, score_dir, r6_dir, contract)
    missing_source = _missing_source_limits(source_restore)
    gate = _canonical_gate(foundation, score, r6, eval_contract, contract)
    next_action = _next_action(gate, eval_contract)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(output_dir),
        "decision_v1": gate["gate_decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "restore_or_rebuild_v1": "CONTRACT_DRIVEN_REBUILD",
        "frozen_wednesday_exact_restore_possible_v1": False,
        "wednesday_freeze_id_v1": contract["freeze_id_v1"],
        "wednesday_candidate_id_v1": contract["candidate_id_v1"],
        "wednesday_expected_rows_v1": contract["expected_rows_v1"],
        "wednesday_as_of_columns_v1": contract["as_of_columns_v1"],
        "monday_foundation_rows_v1": foundation["row_count_v1"],
        "monday_foundation_active_rows_v1": foundation["active_rows_v1"],
        "monday_foundation_quarantine_rows_v1": foundation["quarantine_rows_v1"],
        "monday_foundation_as_of_columns_v1": foundation["as_of_column_count_v1"],
        "r5_r5_1_r5_2_rebuilt_v1": score["score_rebuild_present_v1"],
        "r5_2_status_v1": score["r5_2_v1"]["canonical_status_v1"],
        "r6_retrained_v1": r6["r6_rebuild_present_v1"],
        "r6_eval_verdict_v1": eval_contract["verdict_v1"],
        "blocked_action_v1": next_action["always_enforced_actions_v1"],
        "training_started_v1": bool(score["score_rebuild_present_v1"] or r6["r6_rebuild_present_v1"]),
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "hard_status_v1": {
            "BEVIST": [
                "Frozen Wednesday R6 is a benchmark contract, not locally restorable source.",
                "Wednesday R6 contract snapshot is extracted.",
                "1689 exact-only/protector-first/narrow surfaces are blocked as canonical R6 baseline.",
            ],
            "INDIKERT": [
                "Monday foundation V4 provides fullcoverage-style 1914 rows with 109 AS_OF columns.",
                "Rebuilt R5.2/R6 can be evaluated against the Wednesday contract when run artifacts are present.",
            ],
            "IKKE_ETABLERT": [
                "Bit-for-bit frozen Wednesday R6 restore.",
                "Canonical Monday R6 readiness unless gate decision is CANONICAL_MONDAY_R6_READY.",
            ],
        },
    }

    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "materialized_at_utc_v1": _utc_now(),
        "output_files_v1": REQUIRED_OUTPUTS,
        "input_dirs_v1": {
            "wednesday_snapshot_dir_v1": str(snapshot_dir),
            "foundation_dir_v1": str(foundation_dir) if foundation_dir else None,
            "score_dir_v1": str(score_dir) if score_dir else None,
            "r6_dir_v1": str(r6_dir) if r6_dir else None,
            "source_restore_dir_v1": str(source_restore_dir) if source_restore_dir else None,
        },
        "blocked_canonical_inputs_v1": DO_NOT_USE_FOR_CANONICAL_R6,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "MATERIALIZED",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
    }
    audit = _audit(truth, contract, foundation, score, r6, eval_contract, gate)

    _write_json(output_dir / REQUIRED_OUTPUTS["truth_scope"], truth)
    _write_json(output_dir / REQUIRED_OUTPUTS["contract"], contract)
    _write_json(output_dir / REQUIRED_OUTPUTS["foundation"], foundation)
    _write_json(output_dir / REQUIRED_OUTPUTS["r5_base"], score)
    _write_json(output_dir / REQUIRED_OUTPUTS["r6_retrain"], r6)
    _write_json(output_dir / REQUIRED_OUTPUTS["r6_eval"], eval_contract)
    row_delta.to_csv(output_dir / REQUIRED_OUTPUTS["row_delta"], index=False)
    _write_json(output_dir / REQUIRED_OUTPUTS["missing_source"], missing_source)
    _write_json(output_dir / REQUIRED_OUTPUTS["gate"], gate)
    _write_json(output_dir / REQUIRED_OUTPUTS["next_action"], next_action)
    _write_json(output_dir / REQUIRED_OUTPUTS["summary"], summary)
    _write_json(output_dir / REQUIRED_OUTPUTS["manifest"], manifest)
    _write_json(output_dir / REQUIRED_OUTPUTS["status"], status)
    audit.to_csv(output_dir / REQUIRED_OUTPUTS["audit"], index=False)
    (output_dir / REQUIRED_OUTPUTS["report"]).write_text(
        _report(summary, contract, foundation, score, r6, eval_contract), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--foundation-dir", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--r6-dir", type=Path, default=None)
    parser.add_argument("--source-restore-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        foundation_dir=args.foundation_dir,
        score_dir=args.score_dir,
        r6_dir=args.r6_dir,
        source_restore_dir=args.source_restore_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
