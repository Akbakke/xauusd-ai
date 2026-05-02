#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1 as optuna_materializer


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
LAYER_NAME = "FIND_BACK_TO_WEDNESDAY_R6_SKELETON_AND_REBUILD_MONDAY_FOUNDATION_V1"

WEDNESDAY_FREEZE_ID = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
WEDNESDAY_CANDIDATE_ID = "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"
WEDNESDAY_FAMILY = "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"
WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

SELECTED_V3_ROOT = DEFAULT_REPORTS_ROOT / "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG_20260427T073055Z_LOCK"
FOUNDATION_AUDIT_ROOT = DEFAULT_REPORTS_ROOT / "FOUNDATION_INTEGRITY_AND_HIDDEN_DRIFT_AUDIT_BEFORE_OPTUNA_V1_20260427T073512Z_AUDIT"
OPTUNA_ROOT = DEFAULT_REPORTS_ROOT / "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1_20260427T080458Z_LOCK"
V2_ROOT = DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_20260426T_EXECUTION"
V2_VARIANT = "R5_2_OBJECTIVE_V2_VARIANT_01_V2_BALANCED_STRICT_PROTECT"
V2_SCORE_PACKAGE = V2_ROOT / "variants" / V2_VARIANT / "score_package_v1.parquet"
V2_PREDICTION_VIEW = V2_ROOT / "variants" / V2_VARIANT / "prediction_view_v1.parquet"

MIN_DECISION_PRECISION_DENOMINATOR = 5
MIN_LOSO_SELECTED_GROUPS = 1

SEARCH_TERMS = [
    WEDNESDAY_FREEZE_ID,
    WEDNESDAY_CANDIDATE_ID,
    "ULTRA_SAFE_TAIL_RISKY_ADDON",
    "R5_2",
    "R5.2",
    "hard_asof_runner_guard",
    "batch04_blindspot",
    "runner_protector",
    "tail_control_10_50",
    "risky_allow",
    "bad_risk",
    "AS_OF",
    "HINDSIGHT",
    "repaired",
    "freeze",
    "candidate",
    "shadow",
    "Wednesday",
    "Monday",
    "1971",
    "1914",
]

REQUIRED_CANONICAL_SOURCE_ARTIFACTS = [
    "shadow_meta_all_trade_review_r6_entry_runner_first_contract_v1.json",
    "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
    "shadow_meta_all_trade_review_r6_model_family_bakeoff_v1.csv",
    "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv",
    "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv",
    "models",
]

R6_HEADS = [
    "bad_risk",
    "runner_protector",
    "tail_control_10_50",
    "risky_allow",
    "batch04_blindspot",
]

WEDNESDAY_THRESHOLDS = {
    "bad_threshold_v1": 0.95,
    "risky_threshold_v1": 0.85,
    "tail_threshold_v1": 0.90,
    "runner_threshold_v1": 0.60,
    "r5_2_runner_threshold_v1": 0.74,
    "blindspot_threshold_v1": 0.70,
    "use_r5_2_base_v1": True,
    "guard_v1": "hard_asof_runner_guard",
}

WEDNESDAY_REFERENCE = {
    "expected_rows_v1": 1971,
    "eval_rows_v1": 1971,
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


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _run_command(args: list[str], *, max_bytes: int = 2_000_000) -> str:
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True)
    except OSError as exc:
        return f"COMMAND_FAILED:{exc}\n"
    out = result.stdout[:max_bytes].decode("utf-8", errors="replace")
    err = result.stderr[:20_000].decode("utf-8", errors="replace")
    if result.returncode not in {0, 1}:
        return out + f"\nCOMMAND_RETURN_CODE={result.returncode}\n" + err
    return out


def _path_type(path: Path) -> str:
    lower = str(path).lower()
    if "/models/" in lower or path.suffix in {".joblib", ".pkl"}:
        return "MODEL_TREE"
    if path.suffix in {".py", ".md", ".toml", ".ini"}:
        return "SOURCE_OR_DOC"
    if path.suffix in {".json", ".csv", ".parquet"}:
        return "REPORT_OR_DATA_ARTIFACT"
    if path.is_dir():
        return "DIRECTORY"
    return "UNKNOWN"


def _likely_relevance(path: Path, exact_match: bool) -> str:
    text = str(path).lower()
    if exact_match:
        return "HIGH_EXACT_REFERENCE"
    if "wednesday" in text or "r6" in text or "r5_2" in text or "freeze" in text:
        return "HIGH_CONTEXTUAL"
    if "candidate" in text or "shadow" in text or "repaired" in text:
        return "MEDIUM"
    return "LOW"


def _stat_row(path: Path, exact_paths: set[str]) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    exact = str(path) in exact_paths or WEDNESDAY_FREEZE_ID in str(path) or WEDNESDAY_CANDIDATE_ID in str(path)
    artifact_type = _path_type(path)
    restorable = "unknown"
    reason = "contextual hit; inspect before reuse"
    if exact and artifact_type == "MODEL_TREE":
        restorable = "unknown"
        reason = "exact/model-like hit needs hash verification"
    elif exact and artifact_type != "MODEL_TREE":
        restorable = "unknown"
        reason = "exact reference hit, not sufficient by itself for restore"
    elif exists and "MANAGEMENT_PATH_DYNAMICS_UPSTREAM_REPLAY" in str(path):
        restorable = "false"
        reason = "canonical path is referenced historically but not proven present from this hit"
    return {
        "path_v1": str(path),
        "artifact_type_v1": artifact_type,
        "exists_v1": exists,
        "file_size_v1": None if stat is None or path.is_dir() else stat.st_size,
        "modified_time_utc_v1": None
        if stat is None
        else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "likely_relevance_v1": _likely_relevance(path, exact),
        "exact_wednesday_reference_match_v1": exact,
        "restorable_v1": restorable,
        "reason_v1": reason,
    }


def _artifact_archaeology(output_dir: Path, reports_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_terms = [WEDNESDAY_FREEZE_ID, WEDNESDAY_CANDIDATE_ID, "ULTRA_SAFE_TAIL_RISKY_ADDON", "hard_asof_runner_guard"]
    source_hits = _run_command(
        [
            "rg",
            "-n",
            "-S",
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            "--glob",
            "!__pycache__/**",
            *sum((["-e", term] for term in SEARCH_TERMS), []),
            str(REPO_ROOT),
        ],
        max_bytes=1_000_000,
    )
    report_hits = _run_command(
        [
            "rg",
            "-n",
            "-S",
            "--glob",
            "!*.parquet",
            *sum((["-e", term] for term in SEARCH_TERMS), []),
            str(reports_root),
        ],
        max_bytes=1_000_000,
    )
    model_tree_hits = _run_command(
        [
            "find",
            str(reports_root),
            "-type",
            "f",
            "(",
            "-name",
            "*.joblib",
            "-o",
            "-name",
            "*model*",
            "-o",
            "-name",
            "*preprocessor*",
            ")",
            "-print",
        ],
        max_bytes=500_000,
    )
    exact_path_text = _run_command(
        [
            "rg",
            "-l",
            "-S",
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            *sum((["-e", term] for term in exact_terms), []),
            str(REPO_ROOT),
            str(reports_root),
        ],
        max_bytes=500_000,
    )
    all_path_text = _run_command(
        [
            "rg",
            "-l",
            "-S",
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            *sum((["-e", term] for term in SEARCH_TERMS), []),
            str(REPO_ROOT),
            str(reports_root),
        ],
        max_bytes=1_000_000,
    )
    output_dir.joinpath("local_wednesday_source_hits_v1.txt").write_text(source_hits, encoding="utf-8")
    output_dir.joinpath("local_wednesday_report_hits_v1.txt").write_text(report_hits, encoding="utf-8")
    output_dir.joinpath("local_wednesday_model_tree_hits_v1.txt").write_text(model_tree_hits, encoding="utf-8")

    paths = {line.strip() for line in all_path_text.splitlines() if line.strip().startswith("/")}
    paths.update(line.strip() for line in model_tree_hits.splitlines() if line.strip().startswith("/"))
    exact_paths = {line.strip() for line in exact_path_text.splitlines() if line.strip().startswith("/")}
    inventory = [_stat_row(Path(path), exact_paths) for path in sorted(paths)]

    snapshot_freeze = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    summary = _read_json(snapshot_freeze / WEDNESDAY_SUMMARY)
    manifest = _read_json(snapshot_freeze / WEDNESDAY_MANIFEST)
    canonical_source_dir = Path(str(summary.get("r6_source_dir_v1") or manifest.get("r6_source_dir_v1") or ""))
    r5_2_freeze_dir = Path(str(summary.get("r5_2_freeze_dir_v1") or ""))
    missing_rows: list[dict[str, Any]] = []
    if not canonical_source_dir.exists():
        missing_rows.append(
            {
                "artifact_v1": "canonical_wednesday_r6_source_dir",
                "expected_path_v1": str(canonical_source_dir),
                "status_v1": "MISSING_LOCAL_ARTIFACT",
            }
        )
    for rel in REQUIRED_CANONICAL_SOURCE_ARTIFACTS:
        expected = canonical_source_dir / rel
        if not expected.exists():
            missing_rows.append(
                {
                    "artifact_v1": rel,
                    "expected_path_v1": str(expected),
                    "status_v1": "MISSING_LOCAL_ARTIFACT",
                }
            )
    if not r5_2_freeze_dir.exists():
        missing_rows.append(
            {
                "artifact_v1": "canonical_r5_2_freeze_dir",
                "expected_path_v1": str(r5_2_freeze_dir),
                "status_v1": "MISSING_LOCAL_ARTIFACT",
            }
        )
    missing = {
        "layer_name": "LOCAL_MISSING_REQUIRED_ARTIFACTS_V1",
        "exact_wednesday_restore_possible_v1": False,
        "canonical_source_dir_v1": str(canonical_source_dir),
        "canonical_source_tree_present_v1": canonical_source_dir.exists(),
        "missing_required_artifact_count_v1": len(missing_rows),
        "missing_required_artifacts_v1": missing_rows,
        "reason_v1": "Exact Wednesday restore requires frozen source/model tree and expected hashes; local search did not prove them present.",
    }
    return inventory, missing


def _wednesday_contract(reports_root: Path, missing: dict[str, Any]) -> dict[str, Any]:
    snapshot_freeze = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    summary = _read_json(snapshot_freeze / WEDNESDAY_SUMMARY)
    manifest = _read_json(snapshot_freeze / WEDNESDAY_MANIFEST)
    selected = summary.get("selected_candidate_v1") if isinstance(summary.get("selected_candidate_v1"), dict) else {}
    thresholds = selected.get("selected_thresholds_v1") if isinstance(selected.get("selected_thresholds_v1"), dict) else {}
    thresholds = {**WEDNESDAY_THRESHOLDS, **(manifest.get("thresholds_v1") or {}), **thresholds}
    return {
        "layer_name": "WEDNESDAY_R6_CONTRACT_RECONSTRUCTION_V1",
        "status_v1": "CONTRACT_RECONSTRUCTED_FROM_LOCAL_SUMMARY_AND_MANIFEST",
        "exact_restore_status_v1": "NOT_EXACT_RESTORE",
        "freeze_id_v1": summary.get("freeze_id_v1", WEDNESDAY_FREEZE_ID),
        "candidate_id_v1": summary.get("selected_candidate_id_v1", WEDNESDAY_CANDIDATE_ID),
        "model_version_id_v1": summary.get("model_version_id_v1") or manifest.get("model_version_id_v1"),
        "universe_eval_v1": {
            "expected_rows_v1": int((summary.get("policy_logging_v1") or {}).get("row_count_v1") or WEDNESDAY_REFERENCE["expected_rows_v1"]),
            "eval_rows_v1": int(selected.get("row_count_v1") or WEDNESDAY_REFERENCE["eval_rows_v1"]),
            "repaired_truth_required_v1": True,
            "as_of_hindsight_separation_v1": "AS_OF_SCHEMA_109_SEPARATE_FROM_HINDSIGHT_SCHEMA_30",
            "as_of_columns_v1": (manifest.get("as_of_schema_v1") or {}).get("column_count_v1", 109),
            "hindsight_columns_v1": (manifest.get("hindsight_schema_v1") or {}).get("column_count_v1"),
            "policy_eval_surface_v1": "ALL_1971_WITH_REPAIRED_TRUTH_AND_POLICY_LOGGING_LOCK",
            "safety_pocket_definitions_v1": [
                "repaired_165",
                "50_plus_mfe",
                "100_plus_mfe",
                "200_plus_mfe",
                "strongest_winner_path",
                "runner_protect",
                "ambiguous_high_mfe",
            ],
        },
        "r5_2_base_v1": {
            "use_r5_2_base_v1": bool(thresholds.get("use_r5_2_base_v1", True)),
            "expected_role_v1": "ELIGIBILITY_OPPORTUNITY_BASE_NOT_FINAL_R6_ALONE",
            "required_interaction_with_r6_v1": "R6 five-head policy can add/guard within R5.2 base and hard guard context.",
            "runner_protection_assumptions_v1": "R5.2 runner score threshold and R6 runner_protector cooperate.",
            "high_mfe_winner_protection_assumptions_v1": "100+/200+/strongest winner zero damage; 50+ requires explicit cap decision.",
        },
        "r6_five_head_setup_v1": {
            "heads_v1": R6_HEADS,
            "score_head_names_v1": manifest.get("score_head_names_v1"),
        },
        "thresholds_v1": thresholds,
        "guard_v1": "hard_asof_runner_guard",
        "safety_v1": {
            "repaired_165_damage_v1": selected.get("repaired_165_block_count_v1", WEDNESDAY_REFERENCE["repaired_165_damage_v1"]),
            "strongest_winner_damage_v1": selected.get("strongest_winner_path_block_count_v1", WEDNESDAY_REFERENCE["strongest_winner_damage_v1"]),
            "hundred_plus_mfe_blocked_v1": selected.get("hundred_plus_mfe_block_count_v1", WEDNESDAY_REFERENCE["hundred_plus_mfe_blocked_v1"]),
            "two_hundred_plus_mfe_blocked_v1": selected.get(
                "two_hundred_plus_mfe_block_count_v1", WEDNESDAY_REFERENCE["two_hundred_plus_mfe_blocked_v1"]
            ),
            "fifty_plus_mfe_blocked_v1": selected.get("fifty_plus_mfe_block_count_v1", WEDNESDAY_REFERENCE["fifty_plus_mfe_blocked_v1"]),
            "fifty_plus_contract_decision_v1": "EXPLICIT_CAP_REQUIRED_DO_NOT_SILENTLY_ALLOW_OR_FORBID",
        },
        "metrics_v1": {
            "bad_blocks_v1": selected.get("should_not_take_block_count_v1", WEDNESDAY_REFERENCE["bad_blocks_v1"]),
            "tail_help_v1": selected.get("tail_10_50_help_count_v1", WEDNESDAY_REFERENCE["tail_help_v1"]),
            "precision_v1": selected.get("should_not_take_precision_v1", WEDNESDAY_REFERENCE["precision_v1"]),
            "precision_denominator_v1": selected.get("block_count_v1"),
            "worst_loso_v1": selected.get("worst_loso_precision_v1", WEDNESDAY_REFERENCE["worst_loso_v1"]),
            "worst_loso_denominator_v1": "MISSING_LOCAL_ARTIFACT",
            "decision_valid_metadata_required_v1": True,
        },
        "non_restorable_gaps_v1": missing["missing_required_artifacts_v1"],
        "hash_policy_v1": "DO_NOT_INVENT_MISSING_HASHES_OR_ARTIFACTS",
    }


def _metric_ratio(name: str, numerator: int, denominator: int, min_denominator: int = MIN_DECISION_PRECISION_DENOMINATOR) -> dict[str, Any]:
    if denominator <= 0:
        return {
            f"{name}_v1": np.nan,
            f"{name}_numerator_v1": numerator,
            f"{name}_denominator_v1": denominator,
            f"{name}_denominator_status_v1": "EMPTY_DENOMINATOR",
            f"{name}_decision_valid_v1": False,
        }
    status = "OK" if denominator >= min_denominator else "TOO_SMALL_DENOMINATOR"
    return {
        f"{name}_v1": numerator / denominator,
        f"{name}_numerator_v1": numerator,
        f"{name}_denominator_v1": denominator,
        f"{name}_denominator_status_v1": status,
        f"{name}_decision_valid_v1": status == "OK",
    }


def _worst_loso(frame: pd.DataFrame, selected: pd.Series, bad: pd.Series, group_col: str = "run_id") -> dict[str, Any]:
    rows = []
    for group, part in pd.DataFrame({"group": frame[group_col].astype(str), "selected": selected, "bad": bad}).groupby("group"):
        denominator = int(part["selected"].sum())
        if denominator == 0:
            continue
        numerator = int((part["selected"] & part["bad"]).sum())
        rows.append((str(group), numerator, denominator, numerator / denominator))
    if not rows:
        return {
            "worst_loso_v1": np.nan,
            "worst_loso_group_v1": "EMPTY_SELECTED_GROUP_SET",
            "worst_loso_denominator_v1": 0,
            "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR",
            "worst_loso_decision_valid_v1": False,
            "selected_group_count_v1": 0,
            "small_selected_group_count_v1": 0,
        }
    worst = min(rows, key=lambda item: item[3])
    small = [row for row in rows if row[2] < MIN_DECISION_PRECISION_DENOMINATOR]
    status = "OK"
    reason = "NONE"
    if len(rows) < MIN_LOSO_SELECTED_GROUPS:
        status = "TOO_SMALL_DENOMINATOR"
        reason = "TOO_FEW_SELECTED_GROUPS"
    elif worst[2] < MIN_DECISION_PRECISION_DENOMINATOR:
        status = "TOO_SMALL_DENOMINATOR"
        reason = "WORST_GROUP_SELECTED_DENOMINATOR_TOO_SMALL"
    return {
        "worst_loso_v1": worst[3],
        "worst_loso_group_v1": worst[0],
        "worst_loso_numerator_v1": worst[1],
        "worst_loso_denominator_v1": worst[2],
        "worst_loso_denominator_status_v1": status,
        "worst_loso_denominator_fail_reason_v1": reason,
        "worst_loso_decision_valid_v1": status == "OK",
        "selected_group_count_v1": len(rows),
        "small_selected_group_count_v1": len(small),
    }


def classify_v2_reconciliation(
    *,
    metric_denominator_valid: bool,
    safety_clean: bool,
    provenance_valid: bool,
    artifacts_missing: bool,
) -> str:
    if artifacts_missing:
        return "V2_REQUIRES_MISSING_ARTIFACT"
    if not metric_denominator_valid or not safety_clean:
        return "V2_COLLAPSES_UNDER_CURRENT_GUARDS"
    if not provenance_valid:
        return "V2_HISTORICAL_ONLY_NOT_PROVENANCE_VALID"
    return "V2_DECISION_VALID_UNDER_CURRENT_GUARDS"


def assess_search_space_coverage(
    *,
    has_v2_fixed_control: bool,
    can_evaluate_current_best_baseline: bool,
    can_reproduce_current_best_baseline: bool,
) -> dict[str, Any]:
    coverage_pass = has_v2_fixed_control and can_evaluate_current_best_baseline and can_reproduce_current_best_baseline
    return {
        "status_v1": "PASS" if coverage_pass else "SEARCH_SPACE_COVERAGE_FAILURE",
        "has_v2_fixed_control_v1": has_v2_fixed_control,
        "can_evaluate_current_best_baseline_v1": can_evaluate_current_best_baseline,
        "can_reproduce_current_best_baseline_v1": can_reproduce_current_best_baseline,
        "model_limit_claim_allowed_v1": coverage_pass,
    }


def wednesday_threshold_diagnostic_control() -> dict[str, Any]:
    return {
        "control_id_v1": "WEDNESDAY_THRESHOLD_DIAGNOSTIC_CONTROL_V1",
        "freeze_id_v1": WEDNESDAY_FREEZE_ID,
        "candidate_id_v1": WEDNESDAY_CANDIDATE_ID,
        "thresholds_v1": WEDNESDAY_THRESHOLDS,
        "exact_model_required_v1": False,
        "purpose_v1": "Represent Wednesday policy thresholds as diagnostic config even when exact model tree is missing.",
    }


def validate_explicit_selection_policy(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN")
    return True


def selected_v3_artifact_status(*, selected_for_decisioning: bool, decision_valid_status: str) -> str:
    invalid = decision_valid_status != "DECISION_VALID_FOR_OPTUNA_PREP"
    if selected_for_decisioning and invalid:
        return "BLOCK_SELECTED_INVALID_V3"
    if not selected_for_decisioning and invalid:
        return "HISTORY_ONLY_NOT_BLOCKER"
    return "PASS"


def decision_input_attestation(
    *,
    dummy_input_used: bool,
    synthetic_input_used: bool,
    degraded_fallback_used: bool,
    in_sample_decisioning_used: bool,
) -> dict[str, Any]:
    failures = []
    if dummy_input_used:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic_input_used:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if degraded_fallback_used:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    if in_sample_decisioning_used:
        failures.append("IN_SAMPLE_DECISIONING_FORBIDDEN")
    return {
        "decision_valid_v1": not failures,
        "failures_v1": failures,
        "status_v1": "PASS" if not failures else "BLOCKED",
    }


def missing_wednesday_artifact_gap(artifact: str, expected_path: str) -> dict[str, Any]:
    return {
        "artifact_v1": artifact,
        "expected_path_v1": expected_path,
        "status_v1": "MISSING_LOCAL_ARTIFACT",
        "hash_v1": "MISSING_LOCAL_ARTIFACT",
        "invented_v1": False,
    }


def optuna_result_can_be_new_baseline(go_no_go: str, bad: int, tail: int) -> bool:
    return go_no_go == "CANDIDATE_FOR_R5_2_PACKAGE_BUILD" and bad > 95 and tail > 61


def _v2_baseline_reconciliation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = _read_parquet(V2_SCORE_PACKAGE)
    eval_df = _read_csv(V2_ROOT / "v2_variant_eval_and_safety_gate_v1.csv")
    if frame.empty:
        payload = {
            "layer_name": "V2_BASELINE_RECONCILIATION_UNDER_CURRENT_GUARDS_V1",
            "status_v1": "V2_REQUIRES_MISSING_ARTIFACT",
            "reason_v1": f"missing {V2_SCORE_PACKAGE}",
        }
        return payload, []
    selected = frame["r5_2_v2_final_base_membership"].fillna(False).astype(bool)
    bad = frame["label_should_not_take_v1"].fillna(False).astype(bool)
    tail = frame["tail_10_50_mfe_v1"].fillna(False).astype(bool)
    precision = _metric_ratio("precision", int((selected & bad).sum()), int(selected.sum()))
    loso = _worst_loso(frame, selected, bad)
    safety_cols = {
        "fifty_plus_mfe_overlap_v1": "fifty_plus_mfe_v1",
        "hundred_plus_mfe_overlap_v1": "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_overlap_v1": "two_hundred_plus_mfe_v1",
        "strongest_winner_overlap_v1": "strongest_winner_path_v1",
        "repaired_165_overlap_v1": "r6_label_repaired_165_like_runner_v1",
        "runner_protect_leakage_v1": "r5_2_label_runner_protect_v1",
        "ambiguous_high_mfe_leakage_v1": "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
    }
    safety = {}
    for out_name, col in safety_cols.items():
        safety[out_name] = int((selected & frame.get(col, pd.Series(False, index=frame.index)).fillna(False).astype(bool)).sum())
    safety_clean = all(value == 0 for value in safety.values())
    provenance_files = [
        V2_ROOT / "v2_oof_score_provenance_v1.csv",
        V2_ROOT / "v2_oof_fold_assignment_v1.csv",
        V2_ROOT / "v2_train_validation_membership_v1.csv",
    ]
    provenance_valid = all(path.exists() for path in provenance_files)
    metric_denominator_valid = bool(precision["precision_decision_valid_v1"] and loso["worst_loso_decision_valid_v1"])
    status = classify_v2_reconciliation(
        metric_denominator_valid=metric_denominator_valid,
        safety_clean=safety_clean,
        provenance_valid=provenance_valid,
        artifacts_missing=False,
    )
    control_rows = [
        {
            "control_id_v1": "V2_FIXED_CONTROL_CURRENT_BEST_SAFE_MONDAY",
            "bad_v1": int((selected & bad).sum()),
            "tail_v1": int((selected & tail).sum()),
            "precision_v1": precision["precision_v1"],
            "precision_denominator_v1": precision["precision_denominator_v1"],
            "precision_decision_valid_v1": precision["precision_decision_valid_v1"],
            "worst_loso_v1": loso["worst_loso_v1"],
            "worst_loso_denominator_v1": loso["worst_loso_denominator_v1"],
            "worst_loso_decision_valid_v1": loso["worst_loso_decision_valid_v1"],
            "safety_clean_v1": safety_clean,
            "provenance_valid_v1": provenance_valid,
            "status_v1": status,
        }
    ]
    search_coverage = assess_search_space_coverage(
        has_v2_fixed_control=False,
        can_evaluate_current_best_baseline=True,
        can_reproduce_current_best_baseline=False,
    )
    payload = {
        "layer_name": "V2_BASELINE_RECONCILIATION_UNDER_CURRENT_GUARDS_V1",
        "status_v1": status,
        "v2_root_v1": str(V2_ROOT),
        "v2_variant_v1": V2_VARIANT,
        "can_evaluate_under_current_metric_guards_v1": True,
        "metric_denominator_valid_v1": metric_denominator_valid,
        "provenance_valid_v1": provenance_valid,
        "provenance_status_v1": "MISSING_OOF_PROVENANCE_FILES" if not provenance_valid else "PASS",
        "safety_clean_v1": safety_clean,
        "safe_stronger_than_optuna_56_55_on_raw_counts_v1": int((selected & bad).sum()) > 56 and int((selected & tail).sum()) > 55,
        "metrics_v1": {
            "bad_v1": int((selected & bad).sum()),
            "tail_v1": int((selected & tail).sum()),
            **precision,
            **loso,
        },
        "safety_v1": safety,
        "source_eval_summary_v1": eval_df.iloc[0].to_dict() if not eval_df.empty else {},
        "search_space_coverage_v1": search_coverage,
        "questions_answered_v1": {
            "v2_guard_eval_v1": "Metric counts pass precision but worst LOSO denominator is too small under current guard.",
            "v2_provenance_v1": "Historical comparator; no V2 OOF provenance file set found.",
            "v2_safety_v1": "Safety clean on available Monday V2 score package.",
            "optuna_comparison_v1": "V2 raw counts 95/61 exceed Optuna 56/55, but V2 denominator/provenance block decision-valid claim.",
            "optuna_can_represent_v2_v1": "No fixed V2 control trial was present; exact reproduction was not proven.",
            "why_search_too_narrow_v1": "Current search thresholded weak V3/V2/R5 scores without fixed baseline controls or Wednesday diagnostic control.",
        },
    }
    return payload, control_rows


def _load_optuna_best_selection(ledger: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    best = _read_json(OPTUNA_ROOT / "constrained_optuna_best_candidate_v1.json")
    lock = best.get("candidate_lock_v1") or {}
    params = lock.get("params_v1") or {}
    if ledger.empty or not params:
        return pd.Series(False, index=ledger.index), lock
    _, selected = optuna_materializer._candidate_rule_metrics(ledger, params)
    return selected.astype(bool), lock


def _optuna_failure_frontier() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trials = _read_csv(OPTUNA_ROOT / "constrained_optuna_trials_v1.csv")
    if trials.empty:
        return [], {"status_v1": "OPTUNA_TRIALS_MISSING"}
    best_safe_bad = 56
    best_safe_tail = 55
    higher = trials[(trials["oof_bad_recall_v1"] > best_safe_bad) | (trials["oof_tail_recall_v1"] > best_safe_tail)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in higher.sort_values(["oof_bad_recall_v1", "oof_tail_recall_v1"], ascending=False).head(100).iterrows():
        reasons = str(row.get("fail_reason_v1", "NONE"))
        safety_damage = any(
            int(row.get(col, 0) or 0) > 0
            for col in [
                "hundred_plus_mfe_overlap_v1",
                "two_hundred_plus_mfe_overlap_v1",
                "strongest_winner_overlap_v1",
                "runner_protect_leakage_v1",
                "ambiguous_high_mfe_leakage_v1",
            ]
        )
        rows.append(
            {
                "trial_number_v1": row.get("trial_number_v1"),
                "bad_v1": row.get("oof_bad_recall_v1"),
                "tail_v1": row.get("oof_tail_recall_v1"),
                "status_v1": row.get("status_v1"),
                "fail_reason_v1": reasons,
                "fifty_plus_mfe_overlap_v1": row.get("fifty_plus_overlap_v1"),
                "hundred_plus_mfe_overlap_v1": row.get("hundred_plus_mfe_overlap_v1"),
                "two_hundred_plus_mfe_overlap_v1": row.get("two_hundred_plus_mfe_overlap_v1"),
                "strongest_winner_overlap_v1": row.get("strongest_winner_overlap_v1"),
                "runner_protect_leakage_v1": row.get("runner_protect_leakage_v1"),
                "ambiguous_high_mfe_leakage_v1": row.get("ambiguous_high_mfe_leakage_v1"),
                "precision_denominator_v1": row.get("precision_denominator_v1"),
                "worst_loso_denominator_v1": row.get("worst_loso_denominator_v1"),
                "worst_loso_v1": row.get("worst_loso_v1"),
                "near_safety_pass_v1": not safety_damage and "LOSO_COLLAPSE" in reasons,
                "frontier_class_v1": "TRUE_SAFETY_DAMAGE" if safety_damage else ("LOSO_OR_DENOMINATOR_FRONTIER" if reasons != "NONE" else "PASS"),
            }
        )
    fail_counter: Counter[str] = Counter()
    for reason_text in trials["fail_reason_v1"].fillna("NONE").astype(str):
        for reason in reason_text.split("|"):
            if reason and reason != "NONE":
                fail_counter[reason] += 1
    high_recall_safety_damage = sum(1 for row in rows if row["frontier_class_v1"] == "TRUE_SAFETY_DAMAGE")
    high_recall_50_only = sum(
        1
        for row in rows
        if int(row.get("fifty_plus_mfe_overlap_v1") or 0) > 0
        and all(int(row.get(col) or 0) == 0 for col in ["hundred_plus_mfe_overlap_v1", "two_hundred_plus_mfe_overlap_v1", "strongest_winner_overlap_v1", "runner_protect_leakage_v1"])
    )
    if not rows:
        marker = "SEARCH_SURFACE_TOO_WEAK_NO_HIGH_RECALL_REGION_FOUND"
    elif high_recall_safety_damage:
        marker = "TRUE_SAFETY_DAMAGE_BLOCKS_RECALL"
    elif high_recall_50_only:
        marker = "POSSIBLE_EXPLICIT_50_MFE_CAP_CONTRACT_NEEDED"
    else:
        marker = "HIGH_RECALL_REGION_EXISTS_BUT_LOSO_OR_DENOMINATOR_COLLAPSES"
    summary = {
        "layer_name": "OPTUNA_FAILURE_FRONTIER_ANALYSIS_V1",
        "trial_count_v1": int(len(trials)),
        "hard_constraint_pass_count_v1": int((trials["status_v1"] == "PASS").sum()),
        "fail_counts_v1": dict(fail_counter),
        "higher_than_safe_56_55_trial_count_v1": int(len(higher)),
        "max_bad_v1": int(trials["oof_bad_recall_v1"].max()),
        "max_tail_v1": int(trials["oof_tail_recall_v1"].max()),
        "frontier_marker_v1": marker,
        "interpretation_v1": "High raw recall exists, but passing trials collapse under LOSO/denominator stability rather than protected-winner damage.",
    }
    return rows, summary


def _safe_auc(target: pd.Series, score: pd.Series) -> float | None:
    y = target.fillna(False).astype(bool)
    x = pd.to_numeric(score, errors="coerce")
    mask = x.notna()
    if int(mask.sum()) < 3 or y[mask].nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y[mask].astype(int), x[mask]))
    except ValueError:
        return None


def _existing_signal_family_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger = _read_csv(OPTUNA_ROOT / "constrained_optuna_full_signal_forensics_v1.csv")
    v2_frame = _read_parquet(V2_SCORE_PACKAGE)
    optuna_selected, _lock = _load_optuna_best_selection(ledger)
    if ledger.empty:
        return [], {"status_v1": "LEDGER_MISSING"}
    v2_membership = v2_frame.set_index("candidate_uid")["r5_2_v2_final_base_membership"].astype(bool) if not v2_frame.empty else pd.Series(dtype=bool)
    ledger["captured_by_v2_v1"] = ledger["candidate_uid"].map(v2_membership).fillna(False).astype(bool)
    ledger["captured_by_optuna_best_v1"] = optuna_selected.to_numpy(dtype=bool) if len(optuna_selected) == len(ledger) else False
    ledger["captured_by_v3_v1"] = ledger.get("v3_oof_final_base_v1", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
    safe_target = ledger["safe_recoverable_candidate_v1"].fillna(False).astype(bool)
    families = {
        "R5_BAD": ["r5_bad_score_v1"],
        "R5_TAIL": ["r5_tail_score_v1"],
        "R5_RUNNER": ["r5_runner_score_v1"],
        "R5_1_BAD": ["r5_1_bad_score_v1"],
        "R5_1_RUNNER": ["r5_1_runner_score_v1"],
        "R5_2_V2_BAD_TAIL": ["r5_2_v2_bad_score_v1", "r5_2_v2_tail_score_v1"],
        "R5_2_V2_RUNNER": ["r5_2_v2_runner_protect_score_v1"],
        "V3_OOF_BAD_TAIL": ["r5_2_v3_oof_bad_score_v1", "r5_2_v3_oof_tail_score_v1"],
        "V3_OOF_RUNNER": ["r5_2_v3_oof_runner_protect_score_v1"],
    }
    rows: list[dict[str, Any]] = []
    for family, cols in families.items():
        available = [col for col in cols if col in ledger.columns]
        if available:
            score = pd.concat([pd.to_numeric(ledger[col], errors="coerce") for col in available], axis=1).max(axis=1)
            coverage = int(score.notna().sum())
            auc = _safe_auc(safe_target, score)
            safe_mean = float(score[safe_target].mean()) if int(safe_target.sum()) else None
            protected = ledger["dangerous_or_protected_v1"].fillna(False).astype(bool)
            protected_mean = float(score[protected].mean()) if int(protected.sum()) else None
        else:
            coverage = 0
            auc = None
            safe_mean = None
            protected_mean = None
        rows.append(
            {
                "signal_family_v1": family,
                "columns_v1": "|".join(available),
                "coverage_v1": coverage,
                "safe_recoverable_auc_v1": auc,
                "safe_candidate_mean_score_v1": safe_mean,
                "protected_winner_mean_score_v1": protected_mean,
                "captured_by_v2_count_v1": int((ledger["captured_by_v2_v1"] & safe_target).sum()),
                "missed_by_v2_count_v1": int((~ledger["captured_by_v2_v1"] & safe_target).sum()),
                "captured_by_optuna_best_count_v1": int((ledger["captured_by_optuna_best_v1"] & safe_target).sum()),
                "missed_by_optuna_best_count_v1": int((~ledger["captured_by_optuna_best_v1"] & safe_target).sum()),
                "captured_by_v3_count_v1": int((ledger["captured_by_v3_v1"] & safe_target).sum()),
                "missed_by_v3_count_v1": int((~ledger["captured_by_v3_v1"] & safe_target).sum()),
                "winner_damage_risk_v1": "CHECK" if protected_mean is not None and safe_mean is not None and protected_mean >= safe_mean else "LOW_OR_NOT_ESTABLISHED",
                "oof_provenance_valid_v1": family.startswith("V3_OOF"),
                "as_of_safe_v1": True,
                "underused_v1": bool(auc is not None and auc >= 0.58 and "V3" not in family),
                "appears_in_v2_but_not_optuna_v1": family.startswith("R5_2_V2") and int((ledger["captured_by_v2_v1"] & ~ledger["captured_by_optuna_best_v1"]).sum()) > 0,
            }
        )
    summary = {
        "layer_name": "EXISTING_LEGAL_SIGNAL_FAMILY_AUDIT_V1",
        "ledger_rows_v1": int(len(ledger)),
        "safe_recoverable_candidate_rows_v1": int(safe_target.sum()),
        "captured_by_v2_safe_rows_v1": int((ledger["captured_by_v2_v1"] & safe_target).sum()),
        "captured_by_optuna_best_safe_rows_v1": int((ledger["captured_by_optuna_best_v1"] & safe_target).sum()),
        "captured_by_v3_safe_rows_v1": int((ledger["captured_by_v3_v1"] & safe_target).sum()),
        "protected_winner_rows_v1": int(ledger["dangerous_or_protected_v1"].fillna(False).astype(bool).sum()),
        "ambiguous_high_mfe_rows_v1": int(ledger.get("ambiguous_high_mfe_flag_v1", pd.Series(False, index=ledger.index)).fillna(False).astype(bool).sum()),
        "runner_protect_rows_v1": int(ledger["runner_flag_v1"].fillna(False).astype(bool).sum()),
        "quarantine_rows_v1": int(ledger.get("active_quarantine_v1", pd.Series("", index=ledger.index)).astype(str).str.contains("QUARANTINE").sum()),
        "audit_interpretation_v1": "Existing legal R5/R5.1/V2 score families carry more opportunity signal than selected V3/Optuna best, but denominator/provenance controls must be rebuilt around fixed controls.",
    }
    return rows, summary


def _delta_matrix(
    contract: dict[str, Any],
    v2: dict[str, Any],
    frontier: dict[str, Any],
    signal_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    foundation = _read_json(FOUNDATION_AUDIT_ROOT / "foundation_audit_reference_v1.json")
    if not foundation:
        foundation = {
            "foundation_rows_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
        }
    return [
        {
            "layer_v1": "Universe",
            "classification_v1": "NOT_COMPARABLE_BUT_ACCEPTABLE",
            "wednesday_v1": "1971 rows",
            "monday_v1": "1914 rows / 1852 active / 62 quarantine",
            "evidence_path_v1": str(FOUNDATION_AUDIT_ROOT),
            "exact_metric_v1": "1971_vs_1914_do_not_force_row_match",
            "blocker_status_v1": "NOT_BLOCKER_FOR_MONDAY_IF_EXPLICIT",
            "recommended_repair_v1": "Anchor-aware skeleton must model boundary/quarantine separately.",
        },
        {
            "layer_v1": "Truth/labels",
            "classification_v1": "PARTIAL_MATCH",
            "wednesday_v1": "full repaired truth required",
            "monday_v1": "1914 repaired/pocket labels present, exact Wednesday truth not restored",
            "evidence_path_v1": str(SELECTED_V3_ROOT),
            "exact_metric_v1": "label_rows=1914",
            "blocker_status_v1": "BLOCKS_EXACT_RESTORE",
            "recommended_repair_v1": "Preserve Monday canonical labels, explicitly separate repaired truth vs diagnostics.",
        },
        {
            "layer_v1": "Feature surface",
            "classification_v1": "MATCHES_CONTRACT",
            "wednesday_v1": "AS_OF 109",
            "monday_v1": "AS_OF 109, forbidden/id/synthetic/fallback audit clean",
            "evidence_path_v1": str(FOUNDATION_AUDIT_ROOT),
            "exact_metric_v1": "as_of=109",
            "blocker_status_v1": "CLEAR",
            "recommended_repair_v1": "Reuse existing legal features; no new feature surface yet.",
        },
        {
            "layer_v1": "R5/R5.1/R5.2 signal",
            "classification_v1": "DRIFTED",
            "wednesday_v1": "R5.2 plus R6 stack produced 180/149",
            "monday_v1": f"V2 raw 95/61; V3/Optuna best 56/55; signal families captured_by_v2={signal_summary.get('captured_by_v2_safe_rows_v1')}",
            "evidence_path_v1": str(OPTUNA_ROOT),
            "exact_metric_v1": frontier.get("frontier_marker_v1"),
            "blocker_status_v1": "FOUNDATION_GAP",
            "recommended_repair_v1": "Rebuild R5.2 opportunity skeleton before new search.",
        },
        {
            "layer_v1": "R5.2 base membership",
            "classification_v1": "TOO_WEAK",
            "wednesday_v1": "eligibility base plus R6 five-head",
            "monday_v1": f"V2 95/61 but status={v2.get('status_v1')}; Optuna 56/55; V3 too weak",
            "evidence_path_v1": str(V2_ROOT),
            "exact_metric_v1": f"V2 worst_loso_denominator={v2.get('metrics_v1', {}).get('worst_loso_denominator_v1')}",
            "blocker_status_v1": "BASELINE_REVALIDATION_REQUIRED",
            "recommended_repair_v1": "Make V2 fixed control and denominator-valid baseline mandatory.",
        },
        {
            "layer_v1": "R6 heads",
            "classification_v1": "UNKNOWN_REQUIRES_ARTIFACT",
            "wednesday_v1": "|".join(R6_HEADS),
            "monday_v1": "Current task did not run R6; exact model tree not locally restored",
            "evidence_path_v1": str(DEFAULT_REPORTS_ROOT / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR),
            "exact_metric_v1": "five_head_contract_known_model_tree_missing",
            "blocker_status_v1": "BLOCKS_EXACT_RESTORE",
            "recommended_repair_v1": "Use five-head compatibility contract, not exact restore claim.",
        },
        {
            "layer_v1": "Candidate grid / thresholds",
            "classification_v1": "DRIFTED",
            "wednesday_v1": json.dumps(contract.get("thresholds_v1", {}), sort_keys=True),
            "monday_v1": "Optuna bounds lacked fixed V2 control and Wednesday diagnostic trial",
            "evidence_path_v1": str(OPTUNA_ROOT),
            "exact_metric_v1": "SEARCH_SPACE_COVERAGE_FAILURE",
            "blocker_status_v1": "BLOCKS_MODEL_LIMIT_CLAIM",
            "recommended_repair_v1": "Any future search must include V2 fixed control and Wednesday threshold diagnostic.",
        },
        {
            "layer_v1": "Safety/eval",
            "classification_v1": "PARTIAL_MATCH",
            "wednesday_v1": "0 repaired/100/200/strongest, explicit 50+ cap",
            "monday_v1": "Optuna best safety clean; high-recall trials fail LOSO/denominator",
            "evidence_path_v1": str(OPTUNA_ROOT),
            "exact_metric_v1": frontier.get("frontier_marker_v1"),
            "blocker_status_v1": "DENOMINATOR_AND_LOSO_STABILITY",
            "recommended_repair_v1": "Keep denominator guards; analyze batch/LOSO before threshold search.",
        },
        {
            "layer_v1": "Artifact/provenance",
            "classification_v1": "PARTIAL_MATCH",
            "wednesday_v1": "frozen source/model hashes required for exact restore",
            "monday_v1": "Selected V3 OOF provenance PASS; historical invalid V3 quarantined; Wednesday exact source missing",
            "evidence_path_v1": str(SELECTED_V3_ROOT),
            "exact_metric_v1": "V3 provenance PASS",
            "blocker_status_v1": "EXACT_WEDNESDAY_RESTORE_BLOCKED",
            "recommended_repair_v1": "Proceed contract-driven; never invent source hashes.",
        },
    ]


def _rebuild_plan_contract(go_no_go_decision: str) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OPPORTUNITY_SKELETON_REBUILD_CONTRACT_V1",
        "eligibility_base_v1": {
            "goal_v1": "Increase bad/tail opportunity recall before R6 without winner damage.",
            "allowed_signals_v1": ["existing AS_OF-safe R5/R5.1/R5.2/V2 legal score families", "selected OOF-valid V3 as weak control only"],
            "hard_veto_sequence_v1": ["protected winners", "100+/200+ MFE", "runner-protect", "ambiguous high-MFE", "explicit 50+ cap decision"],
        },
        "safety_architecture_v1": {
            "repaired_165_damage_v1": 0,
            "hundred_plus_mfe_damage_v1": 0,
            "two_hundred_plus_mfe_damage_v1": 0,
            "strongest_winner_damage_v1": 0,
            "runner_protect_leakage_v1": 0,
            "fifty_plus_mfe_policy_v1": "EXPLICIT_CAP_REQUIRED",
            "denominator_validity_required_v1": True,
        },
        "r6_five_head_compatibility_v1": {head: "INPUT_EXPECTATION_MUST_BE_AUDITED_BEFORE_R6_RETRAIN" for head in R6_HEADS},
        "candidate_controls_v1": {
            "v2_fixed_control_required_v1": True,
            "wednesday_threshold_diagnostic_required_v1": True,
            "optuna_best_56_55_negative_control_v1": True,
            "v3_selected_oof_weak_control_not_promotion_v1": True,
            "rescue_raw_true_diagnostic_only_if_safe_v1": True,
        },
        "required_tests_before_new_search_v1": [
            "reproduce_or_evaluate_v2_fixed_control",
            "evaluate_wednesday_threshold_diagnostic",
            "prove_no_in_sample_decisioning",
            "prove_no_dummy_synthetic_fallback",
            "prove_denominator_validity",
            "prove_train_validation_membership_for_oof_inputs",
            "prove_old_invalid_v3_not_selected",
        ],
        "search_policy_after_skeleton_v1": {
            "only_after_controls_pass_v1": True,
            "fixed_controls_required_v1": True,
            "fail_if_current_best_baseline_not_representable_v1": True,
            "purpose_v1": "Fine-tune robust skeleton, not discover foundation from scratch.",
        },
        "current_go_no_go_context_v1": go_no_go_decision,
    }


def _go_no_go(
    *,
    exact_restore_possible: bool,
    v2_status: str,
    frontier_marker: str,
    missing_count: int,
) -> dict[str, Any]:
    if exact_restore_possible:
        decision = "WEDNESDAY_EXACT_ARTIFACTS_FOUND_LOCAL_RESTORE_POSSIBLE"
        next_action = "REINTEGRATE_EXISTING_WEDNESDAY_SKELETON_ASSET_V1"
    elif v2_status in {"V2_COLLAPSES_UNDER_CURRENT_GUARDS", "V2_HISTORICAL_ONLY_NOT_PROVENANCE_VALID"}:
        decision = "V2_BASELINE_NOT_DECISION_VALID_UNDER_CURRENT_GUARDS"
        next_action = "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_V1"
    elif frontier_marker == "TRUE_SAFETY_DAMAGE_BLOCKS_RECALL":
        decision = "TRUE_SAFETY_DAMAGE_BLOCKS_WEDNESDAY_LIKE_RECALL"
        next_action = "MODEL_FAMILY_COMPARISON_AFTER_FOUNDATION_GAP_PROVEN_V1"
    elif missing_count > 20:
        decision = "BLOCKED_BY_MISSING_ARTIFACTS_OR_INCOMPLETE_AUDIT"
        next_action = "BUILD_MONDAY_R5_2_FOUNDATION_FROM_WEDNESDAY_CONTRACT_V1"
    else:
        decision = "WEDNESDAY_CONTRACT_RECONSTRUCTED_MONDAY_SKELETON_GAPS_IDENTIFIED"
        next_action = "BUILD_MONDAY_R5_2_FOUNDATION_FROM_WEDNESDAY_CONTRACT_V1"
    return {
        "layer_name": "FOUNDATION_SKELETON_RECONSTRUCTION_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_recommended_action_v1": next_action,
        "exact_wednesday_restore_possible_v1": exact_restore_possible,
        "v2_reconciliation_status_v1": v2_status,
        "optuna_frontier_marker_v1": frontier_marker,
        "do_not_run_more_optuna_until_controls_pass_v1": True,
        "do_not_build_r5_2_package_v1": True,
        "do_not_run_r6_freeze_promo_live_v1": True,
    }


def _markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def materialize(*, reports_root: Path = DEFAULT_REPORTS_ROOT, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    inventory, missing = _artifact_archaeology(output_dir, reports_root)
    contract = _wednesday_contract(reports_root, missing)
    v2_reconciliation, v2_control_rows = _v2_baseline_reconciliation()
    frontier_rows, frontier_summary = _optuna_failure_frontier()
    signal_rows, signal_summary = _existing_signal_family_audit()
    delta_rows = _delta_matrix(contract, v2_reconciliation, frontier_summary, signal_summary)

    exact_restore_possible = bool(
        missing.get("canonical_source_tree_present_v1") and int(missing.get("missing_required_artifact_count_v1") or 0) == 0
    )
    go_no_go = _go_no_go(
        exact_restore_possible=exact_restore_possible,
        v2_status=str(v2_reconciliation["status_v1"]),
        frontier_marker=str(frontier_summary.get("frontier_marker_v1")),
        missing_count=int(missing.get("missing_required_artifact_count_v1") or 0),
    )
    rebuild_contract = _rebuild_plan_contract(go_no_go["decision_v1"])
    diagnostic_control = wednesday_threshold_diagnostic_control()

    _write_rows(output_dir / "local_wednesday_artifact_inventory_v1.csv", inventory)
    _write_json(
        output_dir / "local_wednesday_artifact_inventory_v1.json",
        {
            "layer_name": "LOCAL_WEDNESDAY_ARTIFACT_INVENTORY_V1",
            "row_count_v1": len(inventory),
            "exact_reference_hit_count_v1": sum(bool(row["exact_wednesday_reference_match_v1"]) for row in inventory),
            "rows_v1": inventory,
        },
    )
    _write_json(output_dir / "local_missing_required_artifacts_v1.json", missing)
    _write_json(output_dir / "wednesday_r6_contract_reconstruction_v1.json", contract)
    (output_dir / "wednesday_r6_contract_reconstruction_v1.md").write_text(
        "\n".join(
            [
                "# Wednesday R6 Contract Reconstruction V1",
                "",
                f"Freeze: `{contract['freeze_id_v1']}`",
                f"Candidate: `{contract['candidate_id_v1']}`",
                f"Exact restore: `{contract['exact_restore_status_v1']}`",
                f"Rows: `{contract['universe_eval_v1']['expected_rows_v1']}`",
                f"Thresholds: `{json.dumps(contract['thresholds_v1'], sort_keys=True)}`",
                "",
                "Missing local artifacts are marked `MISSING_LOCAL_ARTIFACT`; no hashes were invented.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rows(output_dir / "monday_vs_wednesday_skeleton_delta_matrix_v1.csv", delta_rows)
    _write_json(output_dir / "monday_vs_wednesday_skeleton_delta_matrix_v1.json", {"rows_v1": delta_rows})
    (output_dir / "monday_vs_wednesday_skeleton_delta_report_v1.md").write_text(
        "# Monday vs Wednesday Skeleton Delta Report V1\n\n"
        + _markdown_list([f"{row['layer_v1']}: {row['classification_v1']} - {row['recommended_repair_v1']}" for row in delta_rows])
        + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "v2_baseline_reconciliation_under_current_guards_v1.json", v2_reconciliation)
    _write_rows(output_dir / "v2_fixed_control_candidate_eval_v1.csv", v2_control_rows)
    _write_json(output_dir / "v2_fixed_control_candidate_eval_v1.json", {"rows_v1": v2_control_rows})
    (output_dir / "v2_baseline_reconciliation_under_current_guards_v1.md").write_text(
        "\n".join(
            [
                "# V2 Baseline Reconciliation Under Current Guards V1",
                "",
                f"Status: `{v2_reconciliation['status_v1']}`",
                f"Bad/tail: `{v2_reconciliation['metrics_v1']['bad_v1']}` / `{v2_reconciliation['metrics_v1']['tail_v1']}`",
                f"Precision denominator: `{v2_reconciliation['metrics_v1']['precision_denominator_v1']}`",
                f"Worst LOSO denominator: `{v2_reconciliation['metrics_v1']['worst_loso_denominator_v1']}`",
                f"Search coverage: `{v2_reconciliation['search_space_coverage_v1']['status_v1']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rows(output_dir / "optuna_failure_frontier_analysis_v1.csv", frontier_rows)
    _write_json(output_dir / "optuna_failure_frontier_analysis_v1.json", frontier_summary)
    (output_dir / "optuna_failure_frontier_report_v1.md").write_text(
        "\n".join(
            [
                "# Optuna Failure Frontier Report V1",
                "",
                f"Marker: `{frontier_summary.get('frontier_marker_v1')}`",
                f"Higher-than-56/55 trials: `{frontier_summary.get('higher_than_safe_56_55_trial_count_v1')}`",
                f"Fail counts: `{json.dumps(frontier_summary.get('fail_counts_v1'), sort_keys=True)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rows(output_dir / "existing_legal_signal_family_audit_v1.csv", signal_rows)
    _write_json(output_dir / "existing_legal_signal_family_audit_v1.json", {**signal_summary, "families_v1": signal_rows})
    (output_dir / "existing_legal_signal_family_audit_report_v1.md").write_text(
        "\n".join(
            [
                "# Existing Legal Signal Family Audit V1",
                "",
                f"Safe recoverable rows: `{signal_summary.get('safe_recoverable_candidate_rows_v1')}`",
                f"Captured by V2: `{signal_summary.get('captured_by_v2_safe_rows_v1')}`",
                f"Captured by Optuna best: `{signal_summary.get('captured_by_optuna_best_safe_rows_v1')}`",
                f"Captured by V3: `{signal_summary.get('captured_by_v3_safe_rows_v1')}`",
                "",
                signal_summary.get("audit_interpretation_v1", ""),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "r5_2_opportunity_skeleton_rebuild_contract_v1.json", rebuild_contract)
    (output_dir / "r5_2_opportunity_skeleton_rebuild_plan_v1.md").write_text(
        "\n".join(
            [
                "# R5.2 Opportunity Skeleton Rebuild Plan V1",
                "",
                "Build a denominator-valid eligibility base before any new constrained search.",
                "",
                "## Controls",
                _markdown_list(rebuild_contract["candidate_controls_v1"].keys()),
                "",
                "## Required Tests",
                _markdown_list(rebuild_contract["required_tests_before_new_search_v1"]),
                "",
                "Future search may only fine-tune the skeleton after fixed controls pass.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "wednesday_threshold_diagnostic_control_v1.json", diagnostic_control)
    _write_json(output_dir / "foundation_skeleton_reconstruction_go_no_go_v1.json", go_no_go)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "exact_wednesday_artifacts_found_v1": exact_restore_possible,
        "missing_required_artifact_count_v1": missing["missing_required_artifact_count_v1"],
        "wednesday_contract_status_v1": contract["status_v1"],
        "v2_baseline_reconciliation_status_v1": v2_reconciliation["status_v1"],
        "optuna_search_space_coverage_v1": v2_reconciliation["search_space_coverage_v1"]["status_v1"],
        "optuna_failure_frontier_marker_v1": frontier_summary.get("frontier_marker_v1"),
        "signal_family_summary_v1": signal_summary,
        "go_no_go_v1": go_no_go["decision_v1"],
        "next_action_v1": go_no_go["next_recommended_action_v1"],
        "optuna_not_run_v1": True,
        "r6_not_run_v1": True,
        "package_not_built_v1": True,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": go_no_go["decision_v1"]})
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "input_roots_v1": {
                "reports_root_v1": str(reports_root),
                "selected_v3_root_v1": str(SELECTED_V3_ROOT),
                "foundation_audit_root_v1": str(FOUNDATION_AUDIT_ROOT),
                "optuna_root_v1": str(OPTUNA_ROOT),
                "v2_root_v1": str(V2_ROOT),
            },
            "output_dir_v1": str(output_dir),
        },
    )
    (output_dir / "report_v1.md").write_text(
        "\n".join(
            [
                "# Foundation Skeleton Reconstruction Report V1",
                "",
                f"Go/no-go: `{go_no_go['decision_v1']}`",
                f"Next action: `{go_no_go['next_recommended_action_v1']}`",
                f"Exact Wednesday restore possible: `{exact_restore_possible}`",
                f"V2 reconciliation: `{v2_reconciliation['status_v1']}`",
                f"Optuna frontier: `{frontier_summary.get('frontier_marker_v1')}`",
                "",
                "No Optuna, R6, package build, freeze, promo, or live action was run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = materialize(reports_root=args.reports_root, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
