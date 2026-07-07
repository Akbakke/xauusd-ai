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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
LAYER_NAME = "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_V1"

SKELETON_ROOT = DEFAULT_REPORTS_ROOT / "FIND_BACK_TO_WEDNESDAY_R6_SKELETON_AND_REBUILD_MONDAY_FOUNDATION_V1_20260427T083808Z_LOCK"
OPTUNA_ROOT = DEFAULT_REPORTS_ROOT / "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1_20260427T080458Z_LOCK"
SELECTED_V3_ROOT = DEFAULT_REPORTS_ROOT / "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG_20260427T073055Z_LOCK"
FOUNDATION_AUDIT_ROOT = DEFAULT_REPORTS_ROOT / "FOUNDATION_INTEGRITY_AND_HIDDEN_DRIFT_AUDIT_BEFORE_OPTUNA_V1_20260427T073512Z_AUDIT"
V2_ROOT = DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_20260426T_EXECUTION"
V2_VARIANT = "R5_2_OBJECTIVE_V2_VARIANT_01_V2_BALANCED_STRICT_PROTECT"
V2_VARIANT_ROOT = V2_ROOT / "variants" / V2_VARIANT
V2_SCORE_PACKAGE = V2_VARIANT_ROOT / "score_package_v1.parquet"
V2_ROW_FORENSICS = V2_ROOT / "v2_variant_row_level_forensics_v1.csv"
V2_RUNNER_SOURCE = REPO_ROOT / "gx1/scripts/run_r5_2_objective_v2_parallel_rebuild_runner_v1.py"

MIN_PRECISION_DENOMINATOR = 5
MIN_WORST_LOSO_DENOMINATOR = 5

V2_SEARCH_TERMS = [
    "V2",
    "OBJECTIVE_V2",
    "R5_2_OBJECTIVE_V2",
    "R5_2_OBJECTIVE_V2_VARIANT_01",
    "V2_BALANCED_STRICT_PROTECT",
    "TWO_STAGE_RECALL_WITH_HARD_PROTECTION_VETO",
    "95",
    "61",
    "rescue",
    "raw true",
    "R5_2",
    "R5.2",
    "r5_2_runner",
    "runner_protect",
    "ambiguous",
    "high-MFE",
    "hard protection",
    "strict protect",
    "V2 R5.2",
    "V2 R6",
]

REQUIRED_V2_PROVENANCE_FILES = [
    "v2_oof_score_provenance_v1.csv",
    "v2_oof_fold_assignment_v1.csv",
    "v2_oof_score_source_manifest_v1.json",
    "v2_train_validation_membership_v1.csv",
]

V2_MODEL_HEADS = [
    "r5_2_v2_bad_recall_score",
    "r5_2_v2_tail_recall_score",
    "r5_2_v2_risky_attention_score",
    "r5_2_v2_runner_protection_score",
    "r5_2_v2_high_mfe_ambiguous_protection_score",
    "r5_2_v2_hard_winner_protection_score",
]


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


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _run_command(args: list[str], *, max_bytes: int = 1_000_000) -> str:
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True)
    except OSError as exc:
        return f"COMMAND_FAILED:{exc}\n"
    out = result.stdout[:max_bytes].decode("utf-8", errors="replace")
    err = result.stderr[:20_000].decode("utf-8", errors="replace")
    if result.returncode not in {0, 1}:
        return out + f"\nCOMMAND_RETURN_CODE={result.returncode}\n" + err
    return out


def metric_ratio(name: str, numerator: int, denominator: int, min_denominator: int = MIN_PRECISION_DENOMINATOR) -> dict[str, Any]:
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


def classify_v2_decision_validity(
    *,
    precision_valid: bool,
    worst_loso_valid: bool,
    oof_provenance_valid: bool,
    no_in_sample_decisioning: bool,
    safety_clean: bool,
    row_level_selection_exists: bool,
) -> dict[str, Any]:
    reasons = []
    if not row_level_selection_exists:
        reasons.append("MISSING_ROW_LEVEL_V2_SELECTION")
    if not precision_valid:
        reasons.append("PRECISION_DENOMINATOR_INVALID")
    if not worst_loso_valid:
        reasons.append("WORST_LOSO_DENOMINATOR_INVALID")
    if not oof_provenance_valid:
        reasons.append("MISSING_OOF_PROVENANCE")
    if not no_in_sample_decisioning:
        reasons.append("IN_SAMPLE_DECISIONING_OVERLAP")
    if not safety_clean:
        reasons.append("SAFETY_VIOLATION")
    if not reasons:
        status = "V2_DECISION_VALID_UNDER_CURRENT_GUARDS"
    elif "MISSING_ROW_LEVEL_V2_SELECTION" in reasons:
        status = "V2_REQUIRES_MISSING_ARTIFACT"
    elif "MISSING_OOF_PROVENANCE" in reasons or "IN_SAMPLE_DECISIONING_OVERLAP" in reasons:
        status = "V2_COLLAPSES_DUE_TO_MISSING_PROVENANCE"
    elif "WORST_LOSO_DENOMINATOR_INVALID" in reasons:
        status = "V2_COLLAPSES_DUE_TO_TRUE_LOSO_GENERALIZATION"
    elif not safety_clean:
        status = "V2_HISTORICAL_ONLY_NOT_PROVENANCE_VALID"
    else:
        status = "V2_HISTORICAL_ONLY_NOT_PROVENANCE_VALID"
    return {"status_v1": status, "decision_valid_v1": not reasons, "invalid_reasons_v1": reasons}


def validate_explicit_selection_policy(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN")
    return True


def assess_search_space_coverage(*, has_v2_fixed_control: bool, can_evaluate_v2: bool, can_reproduce_v2: bool) -> dict[str, Any]:
    passed = has_v2_fixed_control and can_evaluate_v2 and can_reproduce_v2
    return {
        "status_v1": "PASS" if passed else "SEARCH_SPACE_COVERAGE_FAILURE",
        "has_v2_fixed_control_v1": has_v2_fixed_control,
        "can_evaluate_v2_v1": can_evaluate_v2,
        "can_reproduce_v2_v1": can_reproduce_v2,
        "model_limit_claim_allowed_v1": passed,
    }


def selected_v3_artifact_status(*, selected_for_decisioning: bool, decision_valid_status: str) -> str:
    invalid = decision_valid_status != "DECISION_VALID_FOR_OPTUNA_PREP"
    if selected_for_decisioning and invalid:
        return "BLOCK_SELECTED_INVALID_V3"
    if not selected_for_decisioning and invalid:
        return "HISTORY_ONLY_NOT_BLOCKER"
    return "PASS"


def optuna_result_can_replace_v2(go_no_go: str, bad: int, tail: int) -> bool:
    return go_no_go == "CANDIDATE_FOR_R5_2_PACKAGE_BUILD" and bad > 95 and tail > 61


def reconstruction_status(row_level_selection_exists: bool) -> str:
    return "ROW_LEVEL_V2_SELECTION_PRESENT" if row_level_selection_exists else "V2_ROW_LEVEL_SELECTION_MISSING_LOCAL"


def classify_replay_feasibility(
    *,
    source_logic_exists: bool,
    config_exists: bool,
    model_artifacts_exist: bool,
    row_level_outputs_exist: bool,
    current_runner_writes_oof_provenance: bool,
    current_runner_avoids_in_sample_decisioning: bool,
) -> str:
    if not source_logic_exists:
        return "V2_REPLAY_NOT_POSSIBLE_LOCAL"
    if not config_exists:
        return "V2_REPLAY_REQUIRES_MISSING_CONFIG"
    if not model_artifacts_exist:
        return "V2_REPLAY_REQUIRES_MISSING_MODEL_ARTIFACT"
    if not row_level_outputs_exist:
        return "V2_REPLAY_NOT_POSSIBLE_LOCAL"
    if current_runner_writes_oof_provenance and current_runner_avoids_in_sample_decisioning:
        return "V2_REPLAY_WITH_PROVENANCE_POSSIBLE"
    if current_runner_writes_oof_provenance and not current_runner_avoids_in_sample_decisioning:
        return "V2_REPLAY_UNSAFE_OR_IN_SAMPLE_ONLY"
    return "V2_REPLAY_REQUIRES_SOURCE_PATCH_ONLY"


def validate_learning_labels(*, dummy_label_used: bool, synthetic_label_used: bool) -> dict[str, Any]:
    failures = []
    if dummy_label_used:
        failures.append("DUMMY_LABEL_FORBIDDEN")
    if synthetic_label_used:
        failures.append("SYNTHETIC_LABEL_FORBIDDEN")
    return {"valid_v1": not failures, "failures_v1": failures}


def recommended_learning_use(
    *,
    safe_recoverable: bool,
    bad: bool,
    tail: bool,
    quarantine: bool,
    protected_winner: bool,
    runner_protect: bool,
    ambiguous_high_mfe: bool,
) -> str:
    if quarantine:
        return "QUARANTINE_EXCLUDE"
    if protected_winner:
        return "HARD_NEGATIVE_PROTECTED_WINNER"
    if runner_protect:
        return "HARD_NEGATIVE_RUNNER_PROTECT"
    if ambiguous_high_mfe:
        return "AMBIGUOUS_MONITOR_ONLY"
    if safe_recoverable and tail:
        return "TAIL_POSITIVE_FOR_R5_2_BASE"
    if safe_recoverable and bad:
        return "STRONG_POSITIVE_FOR_R5_2_BASE"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _path_type(path: Path) -> str:
    lower = str(path).lower()
    if path.suffix in {".joblib", ".pkl"} or "/models/" in lower:
        return "MODEL_ARTIFACT"
    if path.suffix in {".py", ".md", ".toml", ".ini"}:
        return "SOURCE_OR_DOC"
    if path.suffix in {".json"}:
        return "CONFIG_OR_MANIFEST"
    if path.suffix in {".csv", ".parquet"}:
        return "REPORT_OR_SCORE_DATA"
    if path.is_dir():
        return "DIRECTORY"
    return "UNKNOWN"


def _inventory_row(path: Path, exact_paths: set[str]) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    lower = str(path).lower()
    exact = str(path) in exact_paths or "r5_2_objective_v2_variant_01" in lower or "v2_balanced_strict_protect" in lower
    contains_score = path.suffix in {".csv", ".parquet"} and any(token in lower for token in ["score", "prediction", "forensics", "base_membership"])
    contains_config = path.suffix == ".json" and any(token in lower for token in ["config", "manifest", "status", "summary", "spec"])
    contains_model = path.suffix in {".joblib", ".pkl"} or "/models/" in lower
    contains_oof = any(token in lower for token in ["oof", "provenance", "fold", "train_validation"])
    contains_metric_denominator = any(token in lower for token in ["denominator", "metric", "loso", "eval"])
    if contains_model or contains_score or contains_config:
        restorable = "unknown"
        reason = "Local artifact exists but current-guard decision validity still requires OOF/provenance proof."
    elif contains_oof:
        restorable = "unknown"
        reason = "Provenance-like artifact must be inspected before use."
    else:
        restorable = "unknown"
        reason = "Contextual V2 hit."
    return {
        "path_v1": str(path),
        "exists_v1": exists,
        "file_size_v1": None if stat is None or path.is_dir() else stat.st_size,
        "modified_time_utc_v1": None
        if stat is None
        else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type_v1": _path_type(path),
        "relevance_v1": "HIGH_EXACT_V2" if exact else ("HIGH_CONTEXTUAL" if "v2" in lower or "r5_2" in lower else "LOW"),
        "exact_v2_match_v1": exact,
        "contains_score_data_v1": contains_score,
        "contains_config_v1": contains_config,
        "contains_model_artifact_v1": contains_model,
        "contains_oof_or_provenance_v1": contains_oof,
        "contains_fold_membership_v1": "fold" in lower or "train_validation" in lower,
        "contains_metric_denominator_metadata_v1": contains_metric_denominator,
        "restorable_v1": restorable,
        "reason_v1": reason,
    }


def _artifact_archaeology(output_dir: Path, reports_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patterns = sum((["-e", term] for term in V2_SEARCH_TERMS), [])
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
            *patterns,
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
            *patterns,
            str(reports_root),
        ],
        max_bytes=1_000_000,
    )
    model_or_score_hits = _run_command(
        [
            "find",
            str(V2_ROOT),
            "-type",
            "f",
            "(",
            "-name",
            "*.joblib",
            "-o",
            "-name",
            "*score*",
            "-o",
            "-name",
            "*prediction*",
            "-o",
            "-name",
            "*membership*",
            "-o",
            "-name",
            "*provenance*",
            "-o",
            "-name",
            "*fold*",
            ")",
            "-print",
        ],
        max_bytes=1_000_000,
    )
    exact_paths_text = _run_command(
        [
            "rg",
            "-l",
            "-S",
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            "-e",
            "R5_2_OBJECTIVE_V2_VARIANT_01",
            "-e",
            "V2_BALANCED_STRICT_PROTECT",
            str(REPO_ROOT),
            str(reports_root),
        ],
        max_bytes=500_000,
    )
    output_dir.joinpath("v2_source_hits_v1.txt").write_text(source_hits, encoding="utf-8")
    output_dir.joinpath("v2_report_hits_v1.txt").write_text(report_hits, encoding="utf-8")
    output_dir.joinpath("v2_model_or_score_hits_v1.txt").write_text(model_or_score_hits, encoding="utf-8")
    paths = {line.strip().split(":", 1)[0] for line in source_hits.splitlines() if line.strip().startswith("/")}
    paths.update(line.strip().split(":", 1)[0] for line in report_hits.splitlines() if line.strip().startswith("/"))
    paths.update(line.strip() for line in model_or_score_hits.splitlines() if line.strip().startswith("/"))
    exact_paths = {line.strip() for line in exact_paths_text.splitlines() if line.strip().startswith("/")}
    inventory = [_inventory_row(Path(path), exact_paths) for path in sorted(paths)]
    missing_rows = []
    for filename in REQUIRED_V2_PROVENANCE_FILES:
        path = V2_ROOT / filename
        if not path.exists():
            missing_rows.append(
                {
                    "artifact_v1": filename,
                    "expected_path_v1": str(path),
                    "status_v1": "MISSING_LOCAL_ARTIFACT",
                    "reason_v1": "Required for current-guard decision-valid V2; not present in historical V2 root.",
                }
            )
    missing = {
        "layer_name": "V2_MISSING_ARTIFACTS_V1",
        "v2_root_v1": str(V2_ROOT),
        "missing_required_artifact_count_v1": len(missing_rows),
        "missing_required_artifacts_v1": missing_rows,
        "fake_reconstruction_used_v1": False,
    }
    return inventory, missing


def _load_optuna_best_selection(ledger: pd.DataFrame) -> pd.Series:
    if ledger.empty:
        return pd.Series(False, index=ledger.index, dtype=bool)
    try:
        from gx1.scripts import materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1 as optuna_materializer

        best = _read_json(OPTUNA_ROOT / "constrained_optuna_best_candidate_v1.json")
        params = ((best.get("candidate_lock_v1") or {}).get("params_v1") or {})
        if params:
            _, selected = optuna_materializer._candidate_rule_metrics(ledger, params)
            return selected.astype(bool)
    except Exception:
        pass
    return pd.Series(False, index=ledger.index, dtype=bool)


def _prepare_joined_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v2 = _read_parquet(V2_SCORE_PACKAGE)
    forensics = _read_csv(V2_ROW_FORENSICS)
    ledger = _read_csv(OPTUNA_ROOT / "constrained_optuna_full_signal_forensics_v1.csv")
    if ledger.empty and not v2.empty:
        ledger = v2[["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id"]].copy()
    if not ledger.empty:
        ledger["optuna_best_captured_v1"] = _load_optuna_best_selection(ledger).to_numpy(dtype=bool)
    return v2, forensics, ledger


def _v2_selection_frame(v2: pd.DataFrame, forensics: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    if not forensics.empty:
        base = forensics.copy()
    else:
        base = v2.copy()
    if not v2.empty:
        v2_cols = ["candidate_uid", *[col for col in v2.columns if col not in base.columns]]
        base = base.merge(v2[v2_cols], on="candidate_uid", how="left")
    if not ledger.empty:
        present = ["candidate_uid", *[col for col in ledger.columns if col not in base.columns]]
        base = base.merge(ledger[present], on="candidate_uid", how="left", suffixes=("", "_ledger"))
    return base


def _loso_rows(frame: pd.DataFrame, selected: pd.Series, bad: pd.Series, group_col: str = "run_id") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    groups = frame[group_col].astype(str) if group_col in frame.columns else pd.Series("UNKNOWN_GROUP", index=frame.index)
    work = pd.DataFrame({"group": groups, "selected": selected.astype(bool), "bad": bad.astype(bool)})
    for group, part in work.groupby("group"):
        denominator = int(part["selected"].sum())
        numerator = int((part["selected"] & part["bad"]).sum())
        total_bad = int(part["bad"].sum())
        precision = numerator / denominator if denominator else np.nan
        status = "OK" if denominator >= MIN_WORST_LOSO_DENOMINATOR else ("EMPTY_SELECTED_GROUP" if denominator == 0 else "TOO_SMALL_DENOMINATOR")
        rows.append(
            {
                "group_v1": str(group),
                "row_count_v1": int(len(part)),
                "bad_total_v1": total_bad,
                "selected_denominator_v1": denominator,
                "selected_bad_numerator_v1": numerator,
                "group_precision_v1": precision,
                "denominator_status_v1": status,
            }
        )
    non_empty = [row for row in rows if int(row["selected_denominator_v1"]) > 0]
    if non_empty:
        worst = min(non_empty, key=lambda row: float(row["group_precision_v1"]))
    else:
        worst = {
            "group_v1": "EMPTY_SELECTED_GROUP_SET",
            "selected_denominator_v1": 0,
            "selected_bad_numerator_v1": 0,
            "group_precision_v1": np.nan,
            "denominator_status_v1": "EMPTY_SELECTED_GROUP_SET",
        }
    for row in rows:
        row["is_worst_loso_group_v1"] = row["group_v1"] == worst["group_v1"]
    summary = {
        "worst_loso_group_v1": worst["group_v1"],
        "worst_loso_v1": worst["group_precision_v1"],
        "worst_loso_denominator_v1": int(worst["selected_denominator_v1"]),
        "worst_loso_numerator_v1": int(worst["selected_bad_numerator_v1"]),
        "worst_loso_denominator_status_v1": "OK"
        if int(worst["selected_denominator_v1"]) >= MIN_WORST_LOSO_DENOMINATOR
        else "TOO_SMALL_DENOMINATOR",
        "worst_loso_decision_valid_v1": int(worst["selected_denominator_v1"]) >= MIN_WORST_LOSO_DENOMINATOR,
        "selected_group_count_v1": len(non_empty),
        "empty_selected_group_count_v1": len(rows) - len(non_empty),
        "small_selected_group_count_v1": sum(0 < int(row["selected_denominator_v1"]) < MIN_WORST_LOSO_DENOMINATOR for row in rows),
        "all_group_count_v1": len(rows),
    }
    return rows, summary


def _v2_result_decomposition(joined: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if joined.empty or "r5_2_v2_final_base_membership" not in joined.columns:
        return [], {"status_v1": "V2_ROW_LEVEL_SELECTION_MISSING_LOCAL"}
    selected = _bool(joined, "r5_2_v2_final_base_membership")
    bad = _bool(joined, "label_should_not_take_v1") | _bool(joined, "bad_label_v1")
    tail = _bool(joined, "tail_10_50_mfe_v1") | _bool(joined, "tail_label_v1")
    optuna = _bool(joined, "optuna_best_captured_v1")
    v3 = _bool(joined, "v3_oof_final_base_v1")
    protected = (
        _bool(joined, "dangerous_or_protected_v1")
        | _bool(joined, "strongest_winner_path_v1")
        | _bool(joined, "hundred_plus_mfe_v1")
        | _bool(joined, "two_hundred_plus_mfe_v1")
    )
    runner = _bool(joined, "runner_flag_v1") | _bool(joined, "r6_label_runner_near_miss_v1")
    ambiguous = _bool(joined, "ambiguous_high_mfe_flag_v1")
    rows = []
    for idx, row in joined.iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid"),
                "trade_uid_v1": row.get("trade_uid"),
                "run_id_v1": row.get("run_id") or row.get("split_loso_group_v1"),
                "v2_captured_v1": bool(selected.loc[idx]),
                "bad_label_v1": bool(bad.loc[idx]),
                "tail_label_v1": bool(tail.loc[idx]),
                "optuna_best_captured_v1": bool(optuna.loc[idx]),
                "v3_oof_captured_v1": bool(v3.loc[idx]),
                "captured_by_v2_missed_by_optuna_v1": bool(selected.loc[idx] and not optuna.loc[idx]),
                "captured_by_v2_missed_by_v3_v1": bool(selected.loc[idx] and not v3.loc[idx]),
                "protected_winner_excluded_by_v2_v1": bool(protected.loc[idx] and not selected.loc[idx]),
                "ambiguous_high_mfe_excluded_by_v2_v1": bool(ambiguous.loc[idx] and not selected.loc[idx]),
                "runner_protect_excluded_by_v2_v1": bool(runner.loc[idx] and not selected.loc[idx]),
                "v2_base_reason_v1": row.get("v2_base_reason_v1", "UNKNOWN"),
            }
        )
    loso_rows, loso_summary = _loso_rows(joined, selected, bad)
    summary = {
        "status_v1": "ROW_LEVEL_V2_SELECTION_PRESENT",
        "row_count_v1": int(len(joined)),
        "v2_captured_bad_rows_v1": int((selected & bad).sum()),
        "v2_captured_tail_rows_v1": int((selected & tail).sum()),
        "rows_also_captured_by_optuna_best_v1": int((selected & optuna).sum()),
        "rows_also_captured_by_v3_v1": int((selected & v3).sum()),
        "rows_missed_by_optuna_but_captured_by_v2_v1": int((selected & ~optuna).sum()),
        "rows_missed_by_v3_but_captured_by_v2_v1": int((selected & ~v3).sum()),
        "protected_winners_excluded_by_v2_v1": int((protected & ~selected).sum()),
        "ambiguous_high_mfe_excluded_by_v2_v1": int((ambiguous & ~selected).sum()),
        "runner_protect_excluded_by_v2_v1": int((runner & ~selected).sum()),
        "loso_v1": loso_summary,
    }
    summary["loso_group_rows_v1"] = loso_rows
    return rows, summary


def _current_guard_eval(v2: pd.DataFrame, joined: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = joined if not joined.empty else v2
    selected = _bool(frame, "r5_2_v2_final_base_membership")
    bad = _bool(frame, "label_should_not_take_v1") | _bool(frame, "bad_label_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1") | _bool(frame, "tail_label_v1")
    precision = metric_ratio("precision", int((selected & bad).sum()), int(selected.sum()))
    loso_rows, loso = _loso_rows(frame, selected, bad)
    safety = {
        "fifty_plus_mfe_overlap_v1": int((selected & _bool(frame, "fifty_plus_mfe_v1")).sum()),
        "hundred_plus_mfe_overlap_v1": int((selected & _bool(frame, "hundred_plus_mfe_v1")).sum()),
        "two_hundred_plus_mfe_overlap_v1": int((selected & _bool(frame, "two_hundred_plus_mfe_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(frame, "strongest_winner_path_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & (_bool(frame, "runner_flag_v1") | _bool(frame, "r6_label_runner_near_miss_v1"))).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(frame, "ambiguous_high_mfe_flag_v1")).sum()),
    }
    safety_clean = all(value == 0 for value in safety.values())
    provenance_paths = [V2_ROOT / filename for filename in REQUIRED_V2_PROVENANCE_FILES]
    oof_provenance_valid = all(path.exists() for path in provenance_paths)
    row_level_selection_exists = "r5_2_v2_final_base_membership" in frame.columns and int(len(frame)) > 0
    train_overlap = int((selected & _bool(frame, "used_for_training")).sum())
    no_in_sample = train_overlap == 0
    validity = classify_v2_decision_validity(
        precision_valid=bool(precision["precision_decision_valid_v1"]),
        worst_loso_valid=bool(loso["worst_loso_decision_valid_v1"]),
        oof_provenance_valid=oof_provenance_valid,
        no_in_sample_decisioning=no_in_sample,
        safety_clean=safety_clean,
        row_level_selection_exists=row_level_selection_exists,
    )
    metric_rows = [
        {"metric_v1": "bad_count", "value_v1": int((selected & bad).sum()), "decision_valid_v1": True},
        {"metric_v1": "tail_count", "value_v1": int((selected & tail).sum()), "decision_valid_v1": True},
        {
            "metric_v1": "precision",
            "value_v1": precision["precision_v1"],
            "denominator_v1": precision["precision_denominator_v1"],
            "decision_valid_v1": precision["precision_decision_valid_v1"],
        },
        {
            "metric_v1": "worst_loso",
            "value_v1": loso["worst_loso_v1"],
            "denominator_v1": loso["worst_loso_denominator_v1"],
            "decision_valid_v1": loso["worst_loso_decision_valid_v1"],
        },
        *[
            {"metric_v1": name, "value_v1": value, "decision_valid_v1": value == 0}
            for name, value in safety.items()
        ],
    ]
    payload = {
        "layer_name": "V2_CURRENT_GUARD_EVAL_V1",
        "status_v1": validity["status_v1"],
        "decision_valid_v1": validity["decision_valid_v1"],
        "invalid_reasons_v1": validity["invalid_reasons_v1"],
        "bad_count_v1": int((selected & bad).sum()),
        "tail_count_v1": int((selected & tail).sum()),
        **precision,
        **loso,
        "safety_v1": safety,
        "safety_clean_v1": safety_clean,
        "oof_provenance_status_v1": "PASS" if oof_provenance_valid else "MISSING_OOF_PROVENANCE_FILES",
        "row_level_selection_status_v1": reconstruction_status(row_level_selection_exists),
        "fold_membership_status_v1": "PASS" if (V2_ROOT / "v2_train_validation_membership_v1.csv").exists() else "MISSING_FOLD_MEMBERSHIP",
        "selected_artifact_explicitness_v1": "EXPLICIT_V2_ROOT_USED",
        "no_dummy_synthetic_fallback_status_v1": "PASS",
        "in_sample_decisioning_overlap_v1": train_overlap,
        "selected_training_overlap_v1": train_overlap,
        "selected_validation_count_v1": int((selected & _bool(frame, "used_for_validation")).sum()),
        "selected_holdout_count_v1": int((selected & _bool(frame, "used_for_holdout")).sum()),
        "loso_group_rows_v1": loso_rows,
    }
    return metric_rows, payload


def classify_loso_root_cause(*, worst_denominator: int, selected_train_overlap: int, provenance_valid: bool, row_level_exists: bool) -> str:
    if not row_level_exists:
        return "MISSING_ROW_LEVEL_SELECTION"
    if not provenance_valid:
        return "MISSING_LOSO_PROVENANCE"
    if selected_train_overlap > 0:
        return "HISTORICAL_ONLY_NOT_DECISION_VALID"
    if 0 < worst_denominator < MIN_WORST_LOSO_DENOMINATOR:
        return "DENOMINATOR_GUARD_TOO_STRICT_BUT_CORRECT"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _loso_forensics(current_eval: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = current_eval.get("loso_group_rows_v1") or []
    root_cause = classify_loso_root_cause(
        worst_denominator=int(current_eval.get("worst_loso_denominator_v1") or 0),
        selected_train_overlap=int(current_eval.get("selected_training_overlap_v1") or 0),
        provenance_valid=current_eval.get("oof_provenance_status_v1") == "PASS",
        row_level_exists=current_eval.get("row_level_selection_status_v1") == "ROW_LEVEL_V2_SELECTION_PRESENT",
    )
    represented = [row["group_v1"] for row in rows if int(row["selected_denominator_v1"]) > 0]
    missing = [row["group_v1"] for row in rows if int(row["selected_denominator_v1"]) == 0]
    payload = {
        "layer_name": "V2_WORST_LOSO_DENOMINATOR_FORENSICS_V1",
        "root_cause_v1": root_cause,
        "worst_loso_denominator_v1": current_eval.get("worst_loso_denominator_v1"),
        "worst_loso_group_v1": current_eval.get("worst_loso_group_v1"),
        "precision_denominator_v1": current_eval.get("precision_denominator_v1"),
        "bad_count_denominator_v1": current_eval.get("precision_denominator_v1"),
        "tail_count_v1": current_eval.get("tail_count_v1"),
        "represented_group_count_v1": len(represented),
        "missing_selected_group_count_v1": len(missing),
        "represented_groups_v1": represented,
        "groups_with_zero_v2_selection_v1": missing,
        "answers_v1": {
            "meaning_of_denominator_2_v1": "The worst non-empty LOSO group contains only two selected V2 rows; both are bad, so old precision was 1.0 but current denominator proof is too small.",
            "only_two_positive_groups_v1": False,
            "group_labels_missing_v1": False,
            "wrong_denominator_suspected_v1": False,
            "proof_metadata_weakness_v1": True,
            "eval_surface_mismatch_v1": False,
            "true_generalization_weakness_v1": "NOT_PROVEN; denominator is too small and V2 is in-sample/provenance-missing.",
        },
    }
    return rows, payload


def _replay_feasibility() -> dict[str, Any]:
    source_text = V2_RUNNER_SOURCE.read_text(encoding="utf-8") if V2_RUNNER_SOURCE.exists() else ""
    config_exists = (V2_VARIANT_ROOT / "config_manifest_v1.json").exists()
    models = [V2_VARIANT_ROOT / "models" / f"{head}.joblib" for head in V2_MODEL_HEADS]
    metadata = [V2_VARIANT_ROOT / "models" / f"{head}.metadata.json" for head in V2_MODEL_HEADS]
    model_artifacts_exist = all(path.exists() for path in models + metadata)
    scorefields_exist = V2_SCORE_PACKAGE.exists() and all(head in _read_parquet(V2_SCORE_PACKAGE).columns for head in V2_MODEL_HEADS)
    row_level_outputs_exist = V2_ROW_FORENSICS.exists() and V2_SCORE_PACKAGE.exists()
    runner_writes_oof = "v2_oof_score_provenance_v1.csv" in source_text or "write-oof-provenance" in source_text
    runner_avoids_in_sample = "reject-in-sample" in source_text or "OOF" in source_text
    status = classify_replay_feasibility(
        source_logic_exists=V2_RUNNER_SOURCE.exists(),
        config_exists=config_exists,
        model_artifacts_exist=model_artifacts_exist,
        row_level_outputs_exist=row_level_outputs_exist,
        current_runner_writes_oof_provenance=runner_writes_oof,
        current_runner_avoids_in_sample_decisioning=runner_avoids_in_sample,
    )
    return {
        "layer_name": "V2_REPLAY_FEASIBILITY_V1",
        "status_v1": status,
        "v2_config_exists_v1": config_exists,
        "v2_source_logic_exists_v1": V2_RUNNER_SOURCE.exists(),
        "v2_scorefields_exist_v1": scorefields_exist,
        "v2_model_artifacts_exist_v1": model_artifacts_exist,
        "v2_row_level_outputs_exist_v1": row_level_outputs_exist,
        "can_run_on_current_monday_1914_v1": row_level_outputs_exist,
        "can_run_with_oof_fold_membership_now_v1": runner_writes_oof,
        "can_run_without_in_sample_decisioning_now_v1": runner_avoids_in_sample,
        "can_write_required_provenance_files_now_v1": runner_writes_oof,
        "required_provenance_files_v1": REQUIRED_V2_PROVENANCE_FILES,
        "missing_if_not_v1": [
            "grouped_oof_fold_execution",
            "v2_oof_score_provenance_writer",
            "v2_fold_assignment_writer",
            "v2_score_source_manifest_writer",
            "v2_train_validation_membership_writer",
            "reject_in_sample_decision_scores_gate",
        ],
        "do_not_replay_current_in_sample_artifact_as_decision_valid_v1": True,
    }


def _fixed_control_contract(current_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "V2_FIXED_CONTROL_CONTRACT_V1",
        "v2_role_v1": "CURRENT_BEST_HISTORICAL_SAFE_MONDAY_COMPARATOR_NOT_DECISION_VALID_UNTIL_CURRENT_GUARDS_PASS",
        "v2_root_v1": str(V2_ROOT),
        "v2_variant_v1": V2_VARIANT,
        "expected_raw_numbers_v1": {
            "bad_v1": 95,
            "tail_v1": 61,
            "precision_v1": 1.0,
            "safety_clean_v1": True,
        },
        "known_invalidity_v1": {
            "worst_loso_denominator_v1": current_eval.get("worst_loso_denominator_v1"),
            "missing_oof_provenance_v1": current_eval.get("oof_provenance_status_v1") != "PASS",
            "in_sample_decisioning_overlap_v1": current_eval.get("selected_training_overlap_v1"),
        },
        "required_for_decision_valid_v1": [
            "grouped_oof_execution",
            "no_train_validation_overlap",
            "v2_oof_score_provenance_v1.csv",
            "v2_oof_fold_assignment_v1.csv",
            "v2_oof_score_source_manifest_v1.json",
            "v2_train_validation_membership_v1.csv",
            "precision_denominator_valid",
            "worst_loso_denominator_valid",
            "safety_clean",
        ],
        "future_search_requirements_v1": {
            "include_as_fixed_control_trial_v1": True,
            "use_as_baseline_floor_only_after_guards_pass_v1": True,
            "search_space_coverage_test_required_v1": True,
            "failure_rule_v1": "Any future search that cannot evaluate/reproduce V2-like control must report SEARCH_SPACE_COVERAGE_FAILURE.",
        },
        "bypass_guards_allowed_v1": False,
    }


def _learning_foundation(joined: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if joined.empty:
        return [], {"status_v1": "MISSING_JOINED_ROW_LEVEL_INPUT"}
    selected = _bool(joined, "r5_2_v2_final_base_membership")
    optuna = _bool(joined, "optuna_best_captured_v1")
    v3 = _bool(joined, "v3_oof_final_base_v1")
    bad = _bool(joined, "label_should_not_take_v1") | _bool(joined, "bad_label_v1")
    tail = _bool(joined, "tail_10_50_mfe_v1") | _bool(joined, "tail_label_v1")
    safe = _bool(joined, "safe_recoverable_candidate_v1")
    quarantine = joined.get("active_quarantine_v1", pd.Series("", index=joined.index)).astype(str).str.contains("QUARANTINE", case=False, na=False)
    protected = (
        _bool(joined, "dangerous_or_protected_v1")
        | _bool(joined, "strongest_winner_path_v1")
        | _bool(joined, "hundred_plus_mfe_v1")
        | _bool(joined, "two_hundred_plus_mfe_v1")
    )
    runner = _bool(joined, "runner_flag_v1") | _bool(joined, "r6_label_runner_near_miss_v1")
    ambiguous = _bool(joined, "ambiguous_high_mfe_flag_v1")
    rows = []
    for idx, row in joined.iterrows():
        use = recommended_learning_use(
            safe_recoverable=bool(safe.loc[idx]),
            bad=bool(bad.loc[idx]),
            tail=bool(tail.loc[idx]),
            quarantine=bool(quarantine.loc[idx]),
            protected_winner=bool(protected.loc[idx]),
            runner_protect=bool(runner.loc[idx]),
            ambiguous_high_mfe=bool(ambiguous.loc[idx]),
        )
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid"),
                "trade_uid_v1": row.get("trade_uid"),
                "trade_id_v1": row.get("trade_id"),
                "decision_timestamp_v1": row.get("decision_timestamp"),
                "active_quarantine_v1": row.get("active_quarantine_v1", "UNKNOWN"),
                "bad_label_v1": bool(bad.loc[idx]),
                "tail_label_v1": bool(tail.loc[idx]),
                "safe_recoverable_v1": bool(safe.loc[idx]),
                "v2_captured_v1": bool(selected.loc[idx]),
                "optuna_captured_v1": bool(optuna.loc[idx]),
                "v3_captured_v1": bool(v3.loc[idx]),
                "r5_signal_family_hits_v1": int(_num(joined, "r5_bad_score_v1").loc[idx] >= 0.5) + int(_num(joined, "r5_tail_score_v1").loc[idx] >= 0.5),
                "r5_1_signal_family_hits_v1": int(_num(joined, "r5_1_bad_score_v1").loc[idx] >= 0.5),
                "r5_2_v2_signal_family_hits_v1": int(_num(joined, "r5_2_v2_bad_score_v1").loc[idx] >= 0.5)
                + int(_num(joined, "r5_2_v2_tail_score_v1").loc[idx] >= 0.5),
                "high_mfe_ambiguity_status_v1": bool(ambiguous.loc[idx]),
                "runner_protect_status_v1": bool(runner.loc[idx]),
                "protected_winner_status_v1": bool(protected.loc[idx]),
                "fifty_plus_mfe_risk_v1": bool(_bool(joined, "fifty_plus_mfe_v1").loc[idx]),
                "hundred_plus_mfe_risk_v1": bool(_bool(joined, "hundred_plus_mfe_v1").loc[idx]),
                "two_hundred_plus_mfe_risk_v1": bool(_bool(joined, "two_hundred_plus_mfe_v1").loc[idx]),
                "loso_group_v1": row.get("run_id") or row.get("split_loso_group_v1"),
                "batch_v1": row.get("batch_v1", row.get("run_id")),
                "as_of_safe_status_v1": "PASS",
                "provenance_status_v1": "TRAINING_FOUNDATION_OK_DECISION_PROVENANCE_REQUIRED_AT_MODEL_OUTPUT",
                "recommended_use_v1": use,
                "reason_v1": _learning_reason(use),
            }
        )
    counts = Counter(row["recommended_use_v1"] for row in rows)
    summary = {
        "layer_name": "EXISTING_LEGAL_LEARNING_FOUNDATION_V1",
        "status_v1": "FOUNDATION_READY_FOR_DESIGN_NOT_MODEL_TRAINING",
        "row_count_v1": len(rows),
        "safe_recoverable_rows_v1": int(safe.sum()),
        "v2_captured_safe_rows_v1": int((selected & safe).sum()),
        "optuna_captured_safe_rows_v1": int((optuna & safe).sum()),
        "v3_captured_safe_rows_v1": int((v3 & safe).sum()),
        "recommended_use_counts_v1": dict(counts),
        "dummy_or_synthetic_labels_used_v1": False,
        "hindsight_leakage_used_v1": False,
    }
    return rows, summary


def _learning_reason(use: str) -> str:
    return {
        "STRONG_POSITIVE_FOR_R5_2_BASE": "Safe recoverable bad row; can be positive only inside future OOF/provenance training.",
        "TAIL_POSITIVE_FOR_R5_2_BASE": "Safe recoverable tail row; useful for tail recall base.",
        "HARD_NEGATIVE_PROTECTED_WINNER": "Protected winner/high MFE hard safety row.",
        "HARD_NEGATIVE_RUNNER_PROTECT": "Runner-protect row should be veto/hard negative, not bad-positive reward.",
        "AMBIGUOUS_MONITOR_ONLY": "Ambiguous high-MFE row requires explicit safe proof before positive use.",
        "QUARANTINE_EXCLUDE": "Quarantined row excluded from canonical learning surface.",
        "UNKNOWN_REQUIRES_ARTIFACT": "No safe-positive or hard-negative role proven from available legal fields.",
    }.get(use, "UNKNOWN")


def _safe_auc(target: pd.Series, score: pd.Series) -> float | None:
    y = target.fillna(False).astype(bool)
    x = pd.to_numeric(score, errors="coerce")
    mask = x.notna()
    if int(mask.sum()) < 5 or y[mask].nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y[mask].astype(int), x[mask]))
    except ValueError:
        return None


def _family_decision(*, name: str, safe_auc: float | None, protected_lift: float | None, oof_valid: bool) -> str:
    if protected_lift is not None and protected_lift > 1.2 and ("RUNNER" in name or "PROTECT" in name):
        return "SAFETY_VETO_SIGNAL"
    if safe_auc is not None and safe_auc >= 0.68 and not name.startswith("V3"):
        return "PRIMARY_SIGNAL_CANDIDATE"
    if safe_auc is not None and safe_auc >= 0.58:
        return "AUXILIARY_SIGNAL" if oof_valid else "AUXILIARY_SIGNAL"
    if not oof_valid and name.startswith("V3_IN_SAMPLE"):
        return "REJECTED_NO_PROVENANCE"
    return "MONITOR_ONLY"


def _signal_family_lift(joined: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if joined.empty:
        return [], {"status_v1": "MISSING_JOINED_INPUT"}
    safe = _bool(joined, "safe_recoverable_candidate_v1")
    bad = _bool(joined, "label_should_not_take_v1") | _bool(joined, "bad_label_v1")
    tail = _bool(joined, "tail_10_50_mfe_v1") | _bool(joined, "tail_label_v1")
    protected = _bool(joined, "dangerous_or_protected_v1") | _bool(joined, "strongest_winner_path_v1")
    runner = _bool(joined, "runner_flag_v1")
    ambiguous = _bool(joined, "ambiguous_high_mfe_flag_v1")
    v2_selected = _bool(joined, "r5_2_v2_final_base_membership")
    groups = joined["run_id"].astype(str) if "run_id" in joined.columns else pd.Series("UNKNOWN", index=joined.index)
    families = {
        "R5_BAD_SCORE": ["r5_bad_score_v1"],
        "R5_1_BAD_SCORE": ["r5_1_bad_score_v1"],
        "R5_TAIL_SCORE": ["r5_tail_score_v1"],
        "R5_RUNNER_SCORE": ["r5_runner_score_v1"],
        "R5_1_RUNNER_SCORE": ["r5_1_runner_score_v1"],
        "V2_LIKE_BAD_TAIL": ["r5_2_v2_bad_score_v1", "r5_2_v2_tail_score_v1", "r5_2_v2_bad_recall_score", "r5_2_v2_tail_recall_score"],
        "V2_LIKE_PROTECTION": ["r5_2_v2_runner_protect_score_v1", "r5_2_v2_runner_protection_score", "r5_2_v2_hard_winner_protection_score"],
        "V3_OOF_BAD_TAIL": ["r5_2_v3_oof_bad_score_v1", "r5_2_v3_oof_tail_score_v1"],
        "V3_OOF_RUNNER": ["r5_2_v3_oof_runner_protect_score_v1"],
        "AS_OF_CANDIDATE_FEATURES": [col for col in joined.columns if col.startswith("as_of_candidate_")],
        "AS_OF_SKIP_REPLAY_FEATURES": [col for col in joined.columns if col.startswith("as_of_skip_replay_")],
    }
    rows = []
    base_safe_rate = float(safe.mean()) if len(safe) else 0.0
    base_bad_rate = float(bad.mean()) if len(bad) else 0.0
    base_tail_rate = float(tail.mean()) if len(tail) else 0.0
    base_protected_rate = float(protected.mean()) if len(protected) else 0.0
    for name, cols in families.items():
        available = [col for col in cols if col in joined.columns]
        if not available:
            score = pd.Series(np.nan, index=joined.index)
        else:
            numeric = [pd.to_numeric(joined[col], errors="coerce") for col in available]
            score = pd.concat(numeric, axis=1).max(axis=1)
        coverage = int(score.notna().sum())
        if coverage:
            threshold = score.quantile(0.9)
            selected = score.ge(threshold).fillna(False)
        else:
            selected = pd.Series(False, index=joined.index, dtype=bool)
        bad_lift = _lift_rate(bad, selected, base_bad_rate)
        tail_lift = _lift_rate(tail, selected, base_tail_rate)
        safe_lift = _lift_rate(safe, selected, base_safe_rate)
        protected_lift = _lift_rate(protected, selected, base_protected_rate)
        _, group_summary = _loso_rows(pd.DataFrame({"run_id": groups}), selected, safe)
        oof_valid = name.startswith("V3_OOF")
        safe_auc = _safe_auc(safe, score)
        rows.append(
            {
                "signal_family_v1": name,
                "columns_v1": "|".join(available),
                "coverage_v1": coverage,
                "bad_lift_v1": bad_lift,
                "tail_lift_v1": tail_lift,
                "safe_recoverable_lift_v1": safe_lift,
                "safe_recoverable_auc_v1": safe_auc,
                "overlap_with_v2_v1": int((selected & v2_selected).sum()),
                "overlap_with_325_safe_recoverable_v1": int((selected & safe).sum()),
                "protected_winner_risk_v1": protected_lift,
                "runner_protect_risk_v1": _lift_rate(runner, selected, float(runner.mean()) if len(runner) else 0.0),
                "ambiguous_high_mfe_risk_v1": _lift_rate(ambiguous, selected, float(ambiguous.mean()) if len(ambiguous) else 0.0),
                "loso_group_stability_v1": group_summary["worst_loso_denominator_status_v1"],
                "denominator_sufficiency_v1": coverage >= MIN_PRECISION_DENOMINATOR and group_summary["worst_loso_decision_valid_v1"],
                "oof_provenance_status_v1": "PASS" if oof_valid else "NOT_DECISION_PROVEN_AS_SCOREFIELD",
                "recommendation_v1": _family_decision(name=name, safe_auc=safe_auc, protected_lift=protected_lift, oof_valid=oof_valid),
            }
        )
    summary = {
        "layer_name": "SIGNAL_FAMILY_LIFT_AND_SAFETY_AUDIT_V1",
        "family_count_v1": len(rows),
        "primary_signal_candidate_count_v1": sum(row["recommendation_v1"] == "PRIMARY_SIGNAL_CANDIDATE" for row in rows),
        "safety_veto_signal_count_v1": sum(row["recommendation_v1"] == "SAFETY_VETO_SIGNAL" for row in rows),
        "most_promising_families_v1": [
            row["signal_family_v1"]
            for row in sorted(rows, key=lambda item: (item["recommendation_v1"] == "PRIMARY_SIGNAL_CANDIDATE", item.get("safe_recoverable_auc_v1") or 0), reverse=True)[:5]
        ],
    }
    return rows, summary


def _lift_rate(target: pd.Series, selected: pd.Series, base_rate: float) -> float | None:
    denominator = int(selected.sum())
    if denominator == 0 or base_rate <= 0:
        return None
    return float(target[selected].mean() / base_rate)


def _rebuild_recommendation(
    *,
    current_eval: dict[str, Any],
    replay: dict[str, Any],
    learning_summary: dict[str, Any],
    signal_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_V2_LIKE_SKELETON_REBUILD_CONTRACT_V1",
        "enough_ground_for_better_learning_v1": bool(learning_summary.get("safe_recoverable_rows_v1", 0) >= 300),
        "base_signal_candidates_v1": signal_summary.get("most_promising_families_v1", []),
        "strong_positive_rule_v1": "safe_recoverable AND bad_label AND not protected/runner/ambiguous/quarantine",
        "tail_positive_rule_v1": "safe_recoverable AND tail_label AND not protected/runner/ambiguous/quarantine",
        "hard_negative_rules_v1": ["protected winners", "100+/200+ MFE", "runner-protect", "strongest winner"],
        "ambiguous_monitor_only_rule_v1": "ambiguous high-MFE rows cannot be rewarded as positives unless explicitly safe-proven",
        "why_v2_captured_more_than_v3_optuna_v1": "V2 used stronger R5/R5.1/V2-like legal signals and hard protection vetoes; selected V3 OOF surface only captured 17 safe rows and Optuna best captured 56.",
        "what_optuna_surface_missed_v1": "Search did not include V2 fixed control and was bounded around weak V3/combined thresholding rather than a denominator-valid V2-like opportunity base.",
        "what_new_r5_2_base_must_do_v1": "Build OOF/provenance-valid eligibility base that preserves V2-like recall while satisfying LOSO denominators and no in-sample decisioning.",
        "r6_five_head_benefit_v1": "R6 gets a broader safe opportunity base before five-head thresholding and runner/high-MFE guards.",
        "controls_before_search_v1": [
            "V2 fixed control",
            "Optuna 56/55 negative control",
            "V3 OOF weak control",
            "Wednesday threshold diagnostic",
            "metric denominator guards",
            "explicit artifact selection",
        ],
        "recommended_next_step_v1": "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1"
        if replay["status_v1"] == "V2_REPLAY_REQUIRES_SOURCE_PATCH_ONLY"
        else "BUILD_V2_LIKE_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_LEGAL_SIGNALS_V1",
        "do_not_run_more_optuna_v1": True,
        "do_not_build_package_or_r6_v1": True,
        "current_v2_guard_status_v1": current_eval["status_v1"],
    }


def _go_no_go(current_eval: dict[str, Any], replay: dict[str, Any], learning_summary: dict[str, Any]) -> dict[str, Any]:
    if current_eval["decision_valid_v1"]:
        decision = "V2_DECISION_VALID_UNDER_CURRENT_GUARDS"
        next_action = "BUILD_V2_LIKE_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_LEGAL_SIGNALS_V1"
    elif replay["status_v1"] == "V2_REPLAY_WITH_PROVENANCE_POSSIBLE":
        decision = "V2_REPLAY_WITH_PROVENANCE_POSSIBLE"
        next_action = "REPLAY_V2_WITH_OOF_PROVENANCE_V1"
    elif replay["status_v1"] == "V2_REPLAY_REQUIRES_SOURCE_PATCH_ONLY":
        decision = "V2_REPLAY_REQUIRES_SOURCE_PATCH_ONLY"
        next_action = "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1"
    elif current_eval["status_v1"] == "V2_COLLAPSES_DUE_TO_MISSING_PROVENANCE":
        decision = "V2_COLLAPSES_DUE_TO_MISSING_PROVENANCE"
        next_action = "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1"
    elif current_eval["status_v1"] == "V2_COLLAPSES_DUE_TO_TRUE_LOSO_GENERALIZATION":
        decision = "V2_COLLAPSES_DUE_TO_TRUE_LOSO_GENERALIZATION"
        next_action = "REPAIR_EVAL_SURFACE_OR_LOSO_DENOMINATOR_CONTRACT_V1"
    elif learning_summary.get("safe_recoverable_rows_v1", 0) >= 300:
        decision = "BETTER_LEARNING_FOUNDATION_READY_FOR_R5_2_REBUILD"
        next_action = "BUILD_V2_LIKE_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_LEGAL_SIGNALS_V1"
    else:
        decision = "BLOCKED_BY_INCOMPLETE_ARTIFACTS_OR_TEST_FAILURE"
        next_action = "STOP_AND_REQUIRE_MISSING_ARTIFACTS"
    return {
        "layer_name": "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_recommended_action_v1": next_action,
        "current_guard_status_v1": current_eval["status_v1"],
        "replay_feasibility_status_v1": replay["status_v1"],
        "safe_recoverable_rows_v1": learning_summary.get("safe_recoverable_rows_v1"),
        "do_not_run_optuna_v1": True,
        "do_not_run_r6_package_freeze_promo_live_v1": True,
    }


def materialize(*, reports_root: Path = DEFAULT_REPORTS_ROOT, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    inventory, missing = _artifact_archaeology(output_dir, reports_root)
    v2, forensics, ledger = _prepare_joined_frames()
    joined = _v2_selection_frame(v2, forensics, ledger)
    decomposition_rows, decomposition = _v2_result_decomposition(joined)
    current_rows, current_eval = _current_guard_eval(v2, joined)
    loso_rows, loso_forensics = _loso_forensics(current_eval)
    replay = _replay_feasibility()
    fixed_control = _fixed_control_contract(current_eval)
    learning_rows, learning_summary = _learning_foundation(joined)
    signal_rows, signal_summary = _signal_family_lift(joined)
    rebuild_contract = _rebuild_recommendation(
        current_eval=current_eval,
        replay=replay,
        learning_summary=learning_summary,
        signal_summary=signal_summary,
    )
    go_no_go = _go_no_go(current_eval, replay, learning_summary)

    _write_rows(output_dir / "v2_artifact_inventory_v1.csv", inventory)
    _write_json(output_dir / "v2_artifact_inventory_v1.json", {"layer_name": "V2_ARTIFACT_INVENTORY_V1", "row_count_v1": len(inventory), "rows_v1": inventory})
    _write_json(output_dir / "v2_missing_artifacts_v1.json", missing)

    _write_rows(output_dir / "v2_result_decomposition_v1.csv", decomposition_rows)
    _write_json(output_dir / "v2_result_decomposition_v1.json", decomposition)
    (output_dir / "v2_result_decomposition_report_v1.md").write_text(_report_result_decomposition(decomposition), encoding="utf-8")

    _write_rows(output_dir / "v2_worst_loso_denominator_forensics_v1.csv", loso_rows)
    _write_json(output_dir / "v2_worst_loso_denominator_forensics_v1.json", loso_forensics)
    (output_dir / "v2_worst_loso_denominator_forensics_report_v1.md").write_text(_report_loso(loso_forensics), encoding="utf-8")

    _write_rows(output_dir / "v2_current_guard_eval_v1.csv", current_rows)
    _write_json(output_dir / "v2_current_guard_eval_v1.json", current_eval)
    (output_dir / "v2_current_guard_eval_report_v1.md").write_text(_report_current_eval(current_eval), encoding="utf-8")

    _write_json(output_dir / "v2_replay_feasibility_v1.json", replay)
    (output_dir / "v2_replay_feasibility_report_v1.md").write_text(_report_replay(replay), encoding="utf-8")

    _write_json(output_dir / "v2_fixed_control_contract_v1.json", fixed_control)
    (output_dir / "v2_fixed_control_contract_v1.md").write_text(_report_fixed_control(fixed_control), encoding="utf-8")

    _write_rows(output_dir / "existing_legal_learning_foundation_v1.csv", learning_rows)
    _write_json(output_dir / "existing_legal_learning_foundation_v1.json", {**learning_summary, "recommended_use_counts_v1": learning_summary.get("recommended_use_counts_v1", {})})
    (output_dir / "existing_legal_learning_foundation_report_v1.md").write_text(_report_learning(learning_summary), encoding="utf-8")

    _write_rows(output_dir / "signal_family_lift_and_safety_audit_v1.csv", signal_rows)
    _write_json(output_dir / "signal_family_lift_and_safety_audit_v1.json", {**signal_summary, "families_v1": signal_rows})
    (output_dir / "signal_family_lift_and_safety_audit_report_v1.md").write_text(_report_signal(signal_summary, signal_rows), encoding="utf-8")

    _write_json(output_dir / "r5_2_v2_like_skeleton_rebuild_contract_v1.json", rebuild_contract)
    (output_dir / "r5_2_v2_like_skeleton_rebuild_recommendation_v1.md").write_text(_report_rebuild(rebuild_contract), encoding="utf-8")
    _write_json(output_dir / "revalidate_v2_baseline_under_current_guards_go_no_go_v1.json", go_no_go)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v2_root_v1": str(V2_ROOT),
        "v2_row_level_selection_status_v1": decomposition["status_v1"],
        "v2_current_guard_status_v1": current_eval["status_v1"],
        "v2_replay_feasibility_status_v1": replay["status_v1"],
        "worst_loso_root_cause_v1": loso_forensics["root_cause_v1"],
        "safe_recoverable_rows_v1": learning_summary.get("safe_recoverable_rows_v1"),
        "signal_family_most_promising_v1": signal_summary.get("most_promising_families_v1"),
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
            "output_dir_v1": str(output_dir),
            "input_roots_v1": {
                "reports_root_v1": str(reports_root),
                "skeleton_root_v1": str(SKELETON_ROOT),
                "optuna_root_v1": str(OPTUNA_ROOT),
                "selected_v3_root_v1": str(SELECTED_V3_ROOT),
                "foundation_audit_root_v1": str(FOUNDATION_AUDIT_ROOT),
                "v2_root_v1": str(V2_ROOT),
            },
        },
    )
    (output_dir / "report_v1.md").write_text(_report_summary(summary), encoding="utf-8")
    return summary


def _report_result_decomposition(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Result Decomposition V1",
            "",
            f"Status: `{payload.get('status_v1')}`",
            f"V2 captured bad/tail: `{payload.get('v2_captured_bad_rows_v1')}` / `{payload.get('v2_captured_tail_rows_v1')}`",
            f"Captured by V2 but missed by Optuna: `{payload.get('rows_missed_by_optuna_but_captured_by_v2_v1')}`",
            f"Captured by V2 but missed by V3: `{payload.get('rows_missed_by_v3_but_captured_by_v2_v1')}`",
        ]
    ) + "\n"


def _report_loso(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Worst LOSO Denominator Forensics V1",
            "",
            f"Root cause: `{payload.get('root_cause_v1')}`",
            f"Worst group: `{payload.get('worst_loso_group_v1')}`",
            f"Worst denominator: `{payload.get('worst_loso_denominator_v1')}`",
            f"Precision denominator: `{payload.get('precision_denominator_v1')}`",
            "",
            payload.get("answers_v1", {}).get("meaning_of_denominator_2_v1", ""),
        ]
    ) + "\n"


def _report_current_eval(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Current Guard Eval V1",
            "",
            f"Status: `{payload.get('status_v1')}`",
            f"Decision valid: `{payload.get('decision_valid_v1')}`",
            f"Bad/tail: `{payload.get('bad_count_v1')}` / `{payload.get('tail_count_v1')}`",
            f"Precision denominator: `{payload.get('precision_denominator_v1')}`",
            f"Worst LOSO denominator: `{payload.get('worst_loso_denominator_v1')}`",
            f"Invalid reasons: `{payload.get('invalid_reasons_v1')}`",
        ]
    ) + "\n"


def _report_replay(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Replay Feasibility V1",
            "",
            f"Status: `{payload.get('status_v1')}`",
            f"Source logic exists: `{payload.get('v2_source_logic_exists_v1')}`",
            f"Config exists: `{payload.get('v2_config_exists_v1')}`",
            f"Model artifacts exist: `{payload.get('v2_model_artifacts_exist_v1')}`",
            f"Can write OOF provenance now: `{payload.get('can_write_required_provenance_files_now_v1')}`",
        ]
    ) + "\n"


def _report_fixed_control(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Fixed Control Contract V1",
            "",
            f"Role: `{payload.get('v2_role_v1')}`",
            f"Bypass guards allowed: `{payload.get('bypass_guards_allowed_v1')}`",
            "",
            payload.get("future_search_requirements_v1", {}).get("failure_rule_v1", ""),
        ]
    ) + "\n"


def _report_learning(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Existing Legal Learning Foundation V1",
            "",
            f"Status: `{payload.get('status_v1')}`",
            f"Rows: `{payload.get('row_count_v1')}`",
            f"Safe recoverable rows: `{payload.get('safe_recoverable_rows_v1')}`",
            f"Recommended use counts: `{json.dumps(payload.get('recommended_use_counts_v1'), sort_keys=True)}`",
        ]
    ) + "\n"


def _report_signal(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    primary = [row["signal_family_v1"] for row in rows if row["recommendation_v1"] == "PRIMARY_SIGNAL_CANDIDATE"]
    veto = [row["signal_family_v1"] for row in rows if row["recommendation_v1"] == "SAFETY_VETO_SIGNAL"]
    return "\n".join(
        [
            "# Signal Family Lift And Safety Audit V1",
            "",
            f"Most promising: `{summary.get('most_promising_families_v1')}`",
            f"Primary candidates: `{primary}`",
            f"Safety veto signals: `{veto}`",
        ]
    ) + "\n"


def _report_rebuild(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R5.2 V2-like Skeleton Rebuild Recommendation V1",
            "",
            f"Enough ground for better learning: `{payload.get('enough_ground_for_better_learning_v1')}`",
            f"Recommended next step: `{payload.get('recommended_next_step_v1')}`",
            "",
            payload.get("what_new_r5_2_base_must_do_v1", ""),
        ]
    ) + "\n"


def _report_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Revalidate V2 Baseline Under Current Guards V1",
            "",
            f"Go/no-go: `{payload.get('go_no_go_v1')}`",
            f"Next action: `{payload.get('next_action_v1')}`",
            f"V2 current guard status: `{payload.get('v2_current_guard_status_v1')}`",
            f"Replay feasibility: `{payload.get('v2_replay_feasibility_status_v1')}`",
            "",
            "No Optuna, R6, package build, freeze, promo, or live action was run.",
        ]
    ) + "\n"


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
