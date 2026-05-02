#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts.run_r5_2_objective_v3_parallel_rebuild_runner_v1 import (
    FORENSIC_REPAIRED_CANDIDATE_UID,
    REQUIRED_KEYS,
    _feature_names,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_V3_OOF_DIR = (
    DEFAULT_REPORTS_ROOT
    / "RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_AND_STRATEGY_GATE_V1_20260426T_EXECUTION_OOF_20260426T190850Z"
)
DEFAULT_V3_IN_SAMPLE_DIR = (
    DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_AND_STRATEGY_GATE_V1_20260426T_EXECUTION"
)
DEFAULT_OPTUNA_DIR = DEFAULT_REPORTS_ROOT / "INSTALL_OPTUNA_AND_RUN_CONSTRAINED_OBJECTIVE_SEARCH_V1_20260426T_EXECUTION_FINAL"
LAYER_NAME = "FOUNDATION_INTEGRITY_AND_HIDDEN_DRIFT_AUDIT_BEFORE_OPTUNA_V1"
EXPECTED_ROWS = 1914
EXPECTED_ACTIVE = 1852
EXPECTED_QUARANTINE = 62
EXPECTED_ASOF_COLUMNS = 109
EXPECTED_V3_FEATURE_COUNT = 97
FORBIDDEN_FEATURE_PATTERNS = [
    "hindsight",
    "exit_",
    "exittruth",
    "management_",
    "bridge",
    "readiness",
    "protector_first",
    "diagnostic",
    "narrow",
]
ID_LEAKAGE_PATTERNS = ["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "row_id"]
SYNTHETIC_PATTERNS = ["dummy", "synthetic", "fake", "placeholder", "default_fill", "zero_default"]
OUTPUT_FILES = [
    "active_score_artifact_selection_v1.json",
    "active_canonical_source_graph_lock_v1.json",
    "feature_matrix_truth_audit_v1.csv",
    "target_and_label_table_audit_v1.json",
    "grouped_oof_and_split_integrity_audit_v1.json",
    "selected_v3_oof_artifact_audit_v1.json",
    "oof_score_provenance_audit_v1.csv",
    "historical_invalid_v3_artifacts_v1.csv",
    "fallback_fail_closed_audit_v1.csv",
    "hardcoded_path_and_spec_resolution_audit_v1.csv",
    "r5_2_r6_pass_through_consistency_audit_v1.json",
    "metric_and_eval_contract_audit_v1.json",
    "hidden_drift_summary_and_go_no_go_v1.json",
    "next_action_lock_v1.json",
    "summary_v1.json",
    "report_v1.md",
    "manifest_v1.json",
    "status_v1.json",
    "consistency_audit_v1.csv",
]
ACTIVE_SELECTION_CONTRACT = "ACTIVE_SCORE_ARTIFACT_SELECTION_V1"
REQUIRED_SELECTED_V3_FILES = [
    "v3_oof_score_provenance_v1.csv",
    "v3_oof_fold_assignment_v1.csv",
    "v3_oof_score_source_manifest_v1.json",
    "v3_train_validation_membership_v1.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _active_score_selection_contract(selected_root: Path) -> dict[str, Any]:
    return {
        "contract": ACTIVE_SELECTION_CONTRACT,
        "decisioning_stage": "PRE_OPTUNA",
        "selection_policy": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "selected_artifacts": {"v3_oof_scores": str(selected_root)},
        "requirements": {
            "oof_score_provenance_required": True,
            "fold_assignment_required": True,
            "score_source_manifest_required": True,
            "train_validation_membership_required": True,
            "metric_denominator_decision_valid_required": True,
        },
    }


def _resolve_selected_v3_root(
    *,
    v3_oof_dir: Path,
    selected_v3_oof_artifact_root: Path | None,
    active_score_artifact_selection: Path | None,
    require_explicit_artifact_selection: bool,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    failures: list[str] = []
    contract: dict[str, Any] = {}
    contract_root: Path | None = None
    if active_score_artifact_selection is not None:
        if not active_score_artifact_selection.exists():
            failures.append(f"ACTIVE_SELECTION_CONTRACT_MISSING:{active_score_artifact_selection}")
        else:
            contract = _read_json(active_score_artifact_selection)
            if contract.get("contract") != ACTIVE_SELECTION_CONTRACT:
                failures.append("ACTIVE_SELECTION_CONTRACT_NAME_INVALID")
            if contract.get("decisioning_stage") != "PRE_OPTUNA":
                failures.append("ACTIVE_SELECTION_STAGE_INVALID")
            if contract.get("selection_policy") != "EXPLICIT_ONLY_NO_LATEST_GLOB":
                failures.append("ACTIVE_SELECTION_POLICY_INVALID")
            root_value = (contract.get("selected_artifacts") or {}).get("v3_oof_scores")
            if root_value:
                contract_root = Path(str(root_value)).expanduser().resolve()
            else:
                failures.append("ACTIVE_SELECTION_MISSING_V3_OOF_SCORES_ROOT")

    selected_root = selected_v3_oof_artifact_root.expanduser().resolve() if selected_v3_oof_artifact_root is not None else contract_root
    if selected_root is None and not require_explicit_artifact_selection:
        selected_root = v3_oof_dir.expanduser().resolve()
        contract = _active_score_selection_contract(selected_root)
        contract["selection_source_v1"] = "LEGACY_COMPAT_EXPLICIT_V3_OOF_DIR_ARGUMENT"
    if selected_v3_oof_artifact_root is not None and contract_root is not None and selected_v3_oof_artifact_root.expanduser().resolve() != contract_root:
        failures.append("ACTIVE_SELECTION_ROOT_MISMATCH")
    if require_explicit_artifact_selection and selected_root is None:
        failures.append("EXPLICIT_SELECTED_V3_OOF_ARTIFACT_ROOT_REQUIRED")
    if selected_root is not None and not contract:
        contract = _active_score_selection_contract(selected_root)
    if selected_root is not None and not selected_root.exists():
        failures.append(f"SELECTED_V3_OOF_ARTIFACT_ROOT_MISSING:{selected_root}")
    return selected_root, contract, failures


def _write_selection_blocked_output(output_dir: Path, selection_contract: dict[str, Any], selection_failures: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    decision = (
        "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"
        if any(failure.startswith("SELECTED_ARTIFACT") or "INVALID" in failure for failure in selection_failures)
        else "EXPLICIT_ARTIFACT_SELECTION_REQUIRED"
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": decision,
        "next_action_v1": "SELECT_NEW_V3_OOF_ARTIFACT_ROOT_EXPLICITLY",
        "foundation_rows_v1": None,
        "asof_columns_v1": None,
        "oof_provenance_missing_count_v1": None,
        "metric_contract_failure_count_v1": None,
        "historical_invalid_v3_artifact_status_v1": "NOT_EVALUATED_NO_SELECTED_ROOT",
        "foundation_clean_ready_for_optuna_v1": False,
        "foundation_clean_for_constrained_optuna_v1": False,
        "hard_failures_v1": {"selection": selection_failures},
    }
    _write_json(output_dir / "active_score_artifact_selection_v1.json", selection_contract or {"contract": ACTIVE_SELECTION_CONTRACT, "status_v1": "MISSING"})
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "status_v1": decision})
    _write_json(output_dir / "hidden_drift_summary_and_go_no_go_v1.json", summary)
    _write_csv(output_dir / "consistency_audit_v1.csv", [{"check_v1": "active_artifact_selection", "status_v1": "FAIL", "evidence_v1": "|".join(selection_failures)}])
    (output_dir / "report_v1.md").write_text(
        "# Foundation Integrity And Hidden Drift Audit Before Optuna V1\n\n"
        f"Decision: `{decision}`\n\n"
        "No implicit latest/glob artifact selection was used.\n",
        encoding="utf-8",
    )
    return summary


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _load_inputs(v3_oof_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    manifest = _read_json(v3_oof_dir / "manifest_v1.json")
    paths = manifest["input_paths_v1"]
    score = pd.read_parquet(paths["score_package_v1"])
    label = pd.read_csv(paths["label_table_v1"])
    variant_index = pd.read_csv(v3_oof_dir / "v3_variant_outputs_index_v1.csv")
    best_lock = _read_json(v3_oof_dir / "best_v3_variant_downstream_r6_input_lock_v1.json")
    best_variant = str(best_lock.get("best_variant_id_v1") or variant_index.iloc[0]["variant_id_v1"])
    best_row = variant_index[variant_index["variant_id_v1"].astype(str).eq(best_variant)]
    if best_row.empty:
        best_row = variant_index.iloc[[0]]
    variant_dir = Path(str(best_row.iloc[0]["variant_dir_v1"]))
    pred = pd.read_parquet(variant_dir / "prediction_view_v1.parquet")
    provenance = None
    if (variant_dir / "v3_oof_score_provenance_v1.csv").exists():
        provenance = pd.read_csv(variant_dir / "v3_oof_score_provenance_v1.csv")
    elif (v3_oof_dir / "v3_oof_score_provenance_v1.csv").exists():
        provenance = pd.read_csv(v3_oof_dir / "v3_oof_score_provenance_v1.csv")
        if "variant_id_v1" in provenance.columns:
            provenance = provenance[provenance["variant_id_v1"].astype(str).eq(best_variant)].copy()
    return manifest, score, label, variant_index, pred, provenance


def _asset_info(path: Path, role: str, consumers: list[str], allowed: bool) -> dict[str, Any]:
    row_count = None
    col_count = None
    key_columns: list[str] = []
    if path.exists() and path.is_file():
        try:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                row_count = len(frame)
                col_count = len(frame.columns)
                key_columns = [column for column in REQUIRED_KEYS if column in frame.columns]
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                row_count = len(frame)
                col_count = len(frame.columns)
                key_columns = [column for column in REQUIRED_KEYS if column in frame.columns]
        except Exception:
            pass
    return {
        "path_v1": str(path),
        "run_id_v1": path.parent.name,
        "row_count_v1": row_count,
        "column_count_v1": col_count,
        "key_columns_v1": key_columns,
        "role_v1": role,
        "downstream_consumers_v1": consumers,
        "allowed_for_optuna_search_v1": allowed,
    }


def _source_graph(
    v3_oof_dir: Path,
    v3_in_sample_dir: Path,
    optuna_dir: Path,
    manifest: dict[str, Any],
    variant_index: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    paths = {key: Path(value) for key, value in manifest["input_paths_v1"].items() if key.endswith("_v1")}
    best_lock = _read_json(v3_oof_dir / "best_v3_variant_downstream_r6_input_lock_v1.json")
    assets = [
        _asset_info(paths["score_package_v1"], "CANONICAL_INPUT", ["V3", "Optuna preflight/search"], True),
        _asset_info(paths["label_table_v1"], "CANONICAL_INPUT", ["V3 target table", "Optuna audit"], True),
        _asset_info(paths["foundation_summary_v1"], "REUSE_INPUT", ["foundation row/schema validation"], True),
        _asset_info(paths["feature_inventory_v1"], "REUSE_INPUT", ["feature legality audit"], True),
        _asset_info(paths["downstream_r6_lock_v1"], "REUSE_INPUT", ["V2 source lineage"], True),
        _asset_info(v3_oof_dir / "v3_variant_outputs_index_v1.csv", "EVAL_ONLY", ["V3 OOF comparison"], True),
        _asset_info(v3_in_sample_dir / "v3_variant_outputs_index_v1.csv", "DIAGNOSTIC_ONLY", ["in-sample autopsy only"], False),
        _asset_info(optuna_dir / "constrained_optuna_trials_v1.csv", "DIAGNOSTIC_ONLY", ["prior Optuna result reference"], False),
    ]
    if "score_package_path_v1" in best_lock:
        assets.append(_asset_info(Path(best_lock["score_package_path_v1"]), "BLOCKED_DO_NOT_USE", ["none until V3 gate passes"], False))
    blocked = [
        "1689 exact-only",
        "narrow/protector-first",
        "bridge/readiness as training",
        "raw unsafe true R5.2",
        "V3 pre-veto base as final",
        "V3 in-sample as decision source",
    ]
    hard_failures = []
    for asset in assets:
        lower = asset["path_v1"].lower()
        if asset["role_v1"] == "CANONICAL_INPUT" and any(token in lower for token in ["1689", "exact_only", "narrow", "protector", "bridge", "readiness", "diagnostic"]):
            hard_failures.append(f"forbidden_canonical_asset:{asset['path_v1']}")
    if not variant_index.empty and "r5_2_v3_base_membership_pre_veto" in str(best_lock).lower() and "r5_2_v3_final_base_membership" not in str(best_lock).lower():
        hard_failures.append("pre_veto_base_used_as_final")
    return {
        "layer_name": "ACTIVE_CANONICAL_SOURCE_GRAPH_LOCK_V1",
        "assets_v1": assets,
        "blocked_assets_v1": blocked,
        "hard_failures_v1": hard_failures,
        "source_graph_pass_v1": not hard_failures,
    }, hard_failures


def _feature_audit(score: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    features, families = _feature_names(score)
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    duplicated = pd.Series(features).duplicated().sum()
    for feature in features:
        series = score[feature]
        lower = feature.lower()
        forbidden = [pattern for pattern in FORBIDDEN_FEATURE_PATTERNS if pattern in lower]
        id_leak = [pattern for pattern in ID_LEAKAGE_PATTERNS if pattern in lower]
        synthetic = [pattern for pattern in SYNTHETIC_PATTERNS if pattern in lower]
        null_rate = float(series.isna().mean())
        constant = bool(series.nunique(dropna=False) <= 1)
        rows.append(
            {
                "feature_name_v1": feature,
                "family_v1": next((name for name, cols in families.items() if feature in cols), "UNKNOWN"),
                "dtype_v1": str(series.dtype),
                "null_rate_v1": null_rate,
                "constant_v1": constant,
                "forbidden_patterns_v1": "|".join(forbidden),
                "id_leakage_patterns_v1": "|".join(id_leak),
                "synthetic_patterns_v1": "|".join(synthetic),
                "asof_legal_v1": feature.startswith("as_of_") or feature.startswith("pred__entry_r5_") or feature.startswith("r5_1_"),
                "status_v1": "FAIL" if forbidden or id_leak or synthetic else "PASS",
            }
        )
        if forbidden:
            hard_failures.append(f"forbidden_feature:{feature}")
        if id_leak:
            hard_failures.append(f"id_leakage_feature:{feature}")
        if synthetic:
            hard_failures.append(f"synthetic_or_dummy_feature:{feature}")
    if len(score) != EXPECTED_ROWS:
        hard_failures.append(f"feature_row_count_mismatch:{len(score)}")
    if len(features) != EXPECTED_V3_FEATURE_COUNT:
        hard_failures.append(f"feature_count_mismatch:{len(features)}")
    if duplicated:
        hard_failures.append(f"duplicate_features:{duplicated}")
    summary = {
        "row_count_v1": len(score),
        "feature_count_v1": len(features),
        "feature_families_v1": {name: len(cols) for name, cols in families.items()},
        "duplicate_feature_count_v1": int(duplicated),
        "constant_feature_count_v1": int(sum(row["constant_v1"] for row in rows)),
        "max_null_rate_v1": max((row["null_rate_v1"] for row in rows), default=0.0),
        "hard_failures_v1": hard_failures,
        "feature_matrix_pass_v1": not hard_failures,
    }
    return pd.DataFrame(rows), summary, hard_failures


def _target_audit(score: pd.DataFrame, label: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, Any], list[str]]:
    hard_failures: list[str] = []
    for column in REQUIRED_KEYS:
        score[column] = score[column].astype(str)
        label[column] = label[column].astype(str)
    score_keys = set(map(tuple, score[REQUIRED_KEYS].to_numpy()))
    label_keys = set(map(tuple, label[REQUIRED_KEYS].to_numpy()))
    missing = score_keys - label_keys
    extra = label_keys - score_keys
    bucket = label.get("new_r5_2_label_bucket_v1", pd.Series("", index=label.index)).astype(str)
    bad_target = _bool(label, "bad_eligibility_target_v1")
    tail_target = _bool(label, "tail_eligibility_target_v1")
    ambiguous = bucket.eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD") | _bool(label, "ambiguous_high_mfe_monitor_v1")
    runner = bucket.eq("RUNNER_PROTECT_TARGET") | _bool(label, "runner_protect_target_v1")
    monitor = bucket.eq("IGNORE_OR_MONITOR_ONLY")
    hard_protect = _bool(label, "hundred_plus_mfe_v1") | _bool(label, "two_hundred_plus_mfe_v1") | _bool(label, "strongest_winner_path_v1") | _bool(label, "r6_label_repaired_165_like_runner_v1")
    leakage_features = [feature for feature in feature_names if feature in label.columns and any(token in feature.lower() for token in ["label", "target", "mfe", "mae", "hindsight"])]
    ambiguous_bad = int((ambiguous & bad_target).sum())
    runner_bad = int((runner & bad_target).sum())
    monitor_positive = int((monitor & (bad_target | tail_target)).sum())
    if len(label) != EXPECTED_ROWS:
        hard_failures.append(f"label_row_count_mismatch:{len(label)}")
    if missing or extra:
        hard_failures.append(f"key_alignment_mismatch:{len(missing)}:{len(extra)}")
    if ambiguous_bad:
        hard_failures.append("ambiguous_high_mfe_bad_positive")
    if runner_bad:
        hard_failures.append("runner_protect_bad_positive")
    if monitor_positive:
        hard_failures.append("monitor_only_positive_target")
    if leakage_features:
        hard_failures.append(f"label_leakage_features:{'|'.join(leakage_features)}")
    return {
        "layer_name": "TARGET_AND_LABEL_TABLE_AUDIT_V1",
        "row_count_v1": len(label),
        "key_alignment_missing_from_label_v1": len(missing),
        "key_alignment_extra_in_label_v1": len(extra),
        "bucket_counts_v1": {str(k): int(v) for k, v in bucket.value_counts().to_dict().items()},
        "bad_target_rows_v1": int(bad_target.sum()),
        "tail_target_rows_v1": int(tail_target.sum()),
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad,
        "runner_protect_bad_positive_count_v1": runner_bad,
        "monitor_only_positive_target_count_v1": monitor_positive,
        "hard_protected_rows_v1": int(hard_protect.sum()),
        "label_leakage_features_in_model_matrix_v1": leakage_features,
        "hard_failures_v1": hard_failures,
        "target_label_pass_v1": not hard_failures,
    }, hard_failures


def _split_audit(score: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    hard_failures: list[str] = []
    group = "run_id" if "run_id" in score.columns else None
    if group is None:
        hard_failures.append("missing_group_key")
        group_counts = {}
    else:
        group_counts = score[group].astype(str).value_counts()
    duplicate_trade_groups = 0
    if group:
        duplicate_trade_groups = int(score.groupby("trade_uid")[group].nunique().gt(1).sum())
        if duplicate_trade_groups:
            hard_failures.append(f"trade_uid_crosses_groups:{duplicate_trade_groups}")
    label_per_group = {}
    if group:
        tmp = score.assign(_bad=_bool(score, "label_should_not_take_v1"), _tail=_bool(score, "tail_10_50_mfe_v1"))
        label_per_group = {
            "groups_v1": int(tmp[group].nunique()),
            "min_group_size_v1": int(group_counts.min()),
            "max_group_size_v1": int(group_counts.max()),
            "groups_with_bad_v1": int(tmp.groupby(group)["_bad"].sum().gt(0).sum()),
            "groups_with_tail_v1": int(tmp.groupby(group)["_tail"].sum().gt(0).sum()),
            "groups_with_protected_50_v1": int(tmp.assign(_p=_bool(tmp, "fifty_plus_mfe_v1")).groupby(group)["_p"].sum().gt(0).sum()),
        }
    return {
        "layer_name": "GROUPED_OOF_AND_SPLIT_INTEGRITY_AUDIT_V1",
        "group_key_v1": group,
        "grouped_oof_expected_v1": True,
        "same_trade_in_multiple_groups_v1": duplicate_trade_groups,
        "group_size_distribution_v1": label_per_group,
        "active_rows_v1": int(score.get("calendar_quarantine_status_v1", pd.Series("", index=score.index)).astype(str).eq("ACTIVE_CANDIDATE").sum()),
        "quarantine_rows_v1": int((~score.get("calendar_quarantine_status_v1", pd.Series("", index=score.index)).astype(str).eq("ACTIVE_CANDIDATE")).sum()),
        "worst_loso_zero_can_be_fold_bug_v1": "NOT_ESTABLISHED_WITHOUT_ROW_LEVEL_FOLD_PROVENANCE",
        "hard_failures_v1": hard_failures,
        "split_integrity_pass_v1": not hard_failures,
    }, hard_failures


def _oof_provenance_audit(pred: pd.DataFrame, provenance: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    required_provenance_cols = {
        "candidate_uid",
        "trade_uid",
        "decision_timestamp",
        "variant_id_v1",
        "score_field_v1",
        "fold_id_v1",
        "group_key_v1",
        "train_validation_membership_v1",
        "source_model_fold_v1",
        "score_source_v1",
        "row_was_in_training_for_source_model_v1",
        "in_sample_score_used_v1",
        "fallback_score_used_v1",
        "synthetic_score_used_v1",
    }
    provenance_has_required_columns = provenance is not None and required_provenance_cols.issubset(set(provenance.columns))
    for column in [col for col in pred.columns if col.startswith("r5_2_v3_") and col.endswith("_score")]:
        coverage = float(pred[column].notna().mean())
        field_provenance = pd.DataFrame()
        if provenance_has_required_columns:
            field_provenance = provenance[provenance["score_field_v1"].astype(str).eq(column)].copy()
        has_complete_provenance = (
            not field_provenance.empty
            and len(field_provenance) == len(pred)
            and field_provenance["score_source_v1"].astype(str).eq("OOF").all()
            and not field_provenance["row_was_in_training_for_source_model_v1"].astype(bool).any()
            and not field_provenance["in_sample_score_used_v1"].astype(bool).any()
            and not field_provenance["fallback_score_used_v1"].astype(bool).any()
            and not field_provenance["synthetic_score_used_v1"].astype(bool).any()
            and not field_provenance["fold_id_v1"].isna().any()
            and not field_provenance["group_key_v1"].isna().any()
        )
        row = {
            "score_field_v1": column,
            "source_model_run_v1": "R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER_V1",
            "fold_id_column_present_v1": has_complete_provenance,
            "train_validation_membership_present_v1": has_complete_provenance,
            "score_source_column_present_v1": has_complete_provenance,
            "score_came_from_model_that_trained_on_row_v1": False if has_complete_provenance else "NOT_MATERIALIZED",
            "row_coverage_v1": coverage,
            "provenance_row_count_v1": int(len(field_provenance)),
            "status_v1": "PASS" if has_complete_provenance and coverage == 1.0 else "FAIL_SCORE_PROVENANCE_MISSING",
        }
        rows.append(row)
        if not has_complete_provenance:
            hard_failures.append(f"missing_oof_provenance:{column}")
        if coverage < 1.0:
            hard_failures.append(f"incomplete_oof_coverage:{column}:{coverage}")
    return pd.DataFrame(rows), hard_failures


def _best_variant_id(v3_oof_dir: Path) -> str | None:
    best_lock = _read_json(v3_oof_dir / "best_v3_variant_downstream_r6_input_lock_v1.json")
    if best_lock.get("best_variant_id_v1"):
        return str(best_lock["best_variant_id_v1"])
    index_path = v3_oof_dir / "v3_variant_outputs_index_v1.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        if not index.empty and "variant_id_v1" in index.columns:
            return str(index.iloc[0]["variant_id_v1"])
    return None


def _root_has_invalidated_status(v3_oof_dir: Path) -> bool:
    def invalid_value(payload: Any, parent_key: str = "") -> bool:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in {"reconstruction_status_v1", "status_v1", "decision_valid_status_v1"} and value == "INVALID_FOR_OPTUNA_DECISIONING":
                    return True
                if key == "invalidated_status_v1" and value not in {"", None, "NOT_INVALIDATED"}:
                    return True
                if key in {"decision_valid_for_pre_optuna_v1", "existing_v3_oof_scores_valid_for_optuna_v1", "decision_valid_v1"} and value is False:
                    return True
                if invalid_value(value, key):
                    return True
        elif isinstance(payload, list):
            return any(invalid_value(item, parent_key) for item in payload)
        return False

    for name in [
        "summary_v1.json",
        "status_v1.json",
        "v3_oof_score_source_manifest_v1.json",
        "v3_oof_score_provenance_reconstruction_or_invalidation_v1.json",
        "oof_score_provenance_validation_v1.json",
    ]:
        path = v3_oof_dir / name
        if not path.exists():
            continue
        if invalid_value(_read_json(path)):
            return True
    return False


def _selected_v3_oof_artifact_audit(v3_oof_dir: Path, pred: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    file_status = {}
    for name in REQUIRED_SELECTED_V3_FILES:
        exists = (v3_oof_dir / name).exists()
        file_status[name] = "PRESENT" if exists else "MISSING"
        if not exists:
            failures.append(f"SELECTED_ARTIFACT_MISSING_REQUIRED_FILE:{name}")
    if _root_has_invalidated_status(v3_oof_dir):
        failures.append("SELECTED_ARTIFACT_MARKED_INVALID_FOR_OPTUNA_DECISIONING")
    if failures:
        return {
            "layer_name": "SELECTED_V3_OOF_ARTIFACT_AUDIT_V1",
            "selected_v3_oof_artifact_root_v1": str(v3_oof_dir),
            "required_file_status_v1": file_status,
            "status_v1": "FAIL",
            "failure_reasons_v1": failures,
        }, failures

    provenance = pd.read_csv(v3_oof_dir / "v3_oof_score_provenance_v1.csv")
    fold_assignment = pd.read_csv(v3_oof_dir / "v3_oof_fold_assignment_v1.csv")
    membership = pd.read_csv(v3_oof_dir / "v3_train_validation_membership_v1.csv")
    source_manifest = _read_json(v3_oof_dir / "v3_oof_score_source_manifest_v1.json")
    best_variant = _best_variant_id(v3_oof_dir)
    if best_variant is None:
        failures.append("SELECTED_ARTIFACT_MISSING_BEST_VARIANT_ID")
    if best_variant is not None and "variant_id_v1" in provenance.columns:
        provenance = provenance[provenance["variant_id_v1"].astype(str).eq(best_variant)].copy()
    if best_variant is not None and "variant_id_v1" in fold_assignment.columns:
        fold_assignment = fold_assignment[fold_assignment["variant_id_v1"].astype(str).eq(best_variant)].copy()
    if best_variant is not None and "variant_id_v1" in membership.columns:
        membership = membership[membership["variant_id_v1"].astype(str).eq(best_variant)].copy()

    required_provenance_cols = {
        *REQUIRED_KEYS,
        "variant_id_v1",
        "score_field_v1",
        "fold_id_v1",
        "group_key_v1",
        "train_validation_membership_v1",
        "source_model_fold_v1",
        "model_source_identifier_v1",
        "score_source_v1",
        "feature_matrix_hash_v1",
        "feature_matrix_columns_hash_v1",
        "label_table_hash_v1",
        "config_hash_v1",
        "seed_v1",
        "decision_valid_v1",
        "decision_valid_status_v1",
        "oof_provenance_status_v1",
        "row_was_in_training_for_source_model_v1",
        "in_sample_score_used_v1",
        "fallback_score_used_v1",
        "synthetic_score_used_v1",
    }
    missing_cols = sorted(required_provenance_cols.difference(provenance.columns))
    if missing_cols:
        failures.append(f"SELECTED_PROVENANCE_MISSING_COLUMNS:{missing_cols}")
    expected_keys = set(map(tuple, pred[REQUIRED_KEYS].astype(str).to_numpy()))
    per_field: list[dict[str, Any]] = []
    if not missing_cols:
        for score_field in [column for column in pred.columns if column.startswith("r5_2_v3_") and column.endswith("_score")]:
            field_rows = provenance[provenance["score_field_v1"].astype(str).eq(score_field)].copy()
            field_keys = set(map(tuple, field_rows[REQUIRED_KEYS].astype(str).to_numpy()))
            duplicate_count = int(field_rows.duplicated(subset=[*REQUIRED_KEYS, "score_field_v1"]).sum())
            missing_keys = expected_keys - field_keys
            extra_keys = field_keys - expected_keys
            status = "PASS"
            if len(field_rows) != len(pred) or missing_keys or extra_keys or duplicate_count:
                status = "FAIL"
                failures.append(
                    f"SELECTED_PROVENANCE_ROW_SOURCE_MISMATCH:{score_field}:rows={len(field_rows)}:missing={len(missing_keys)}:extra={len(extra_keys)}:dupes={duplicate_count}"
                )
            if not field_rows["score_source_v1"].astype(str).eq("OOF").all():
                status = "FAIL"
                failures.append(f"SELECTED_PROVENANCE_SCORE_SOURCE_NOT_OOF:{score_field}")
            for bool_col, failure_name in [
                ("row_was_in_training_for_source_model_v1", "ROW_SCORED_BY_MODEL_TRAINED_ON_ROW"),
                ("in_sample_score_used_v1", "IN_SAMPLE_SCORE_MARKED_AS_OOF"),
                ("fallback_score_used_v1", "FALLBACK_SCORE_USED"),
                ("synthetic_score_used_v1", "SYNTHETIC_SCORE_USED"),
            ]:
                if field_rows[bool_col].astype(bool).any():
                    status = "FAIL"
                    failures.append(f"SELECTED_PROVENANCE_{failure_name}:{score_field}")
            if not field_rows["decision_valid_v1"].astype(bool).all():
                status = "FAIL"
                failures.append(f"SELECTED_PROVENANCE_DECISION_VALID_FALSE:{score_field}")
            if not field_rows["oof_provenance_status_v1"].astype(str).eq("PASS").all():
                status = "FAIL"
                failures.append(f"SELECTED_PROVENANCE_STATUS_NOT_PASS:{score_field}")
            for hash_col in ["feature_matrix_hash_v1", "feature_matrix_columns_hash_v1", "label_table_hash_v1", "config_hash_v1"]:
                if field_rows[hash_col].astype(str).str.len().lt(12).any():
                    status = "FAIL"
                    failures.append(f"SELECTED_PROVENANCE_HASH_MISSING:{score_field}:{hash_col}")
            per_field.append(
                {
                    "score_field_v1": score_field,
                    "provenance_rows_v1": int(len(field_rows)),
                    "expected_rows_v1": int(len(pred)),
                    "duplicate_rows_v1": duplicate_count,
                    "missing_key_count_v1": int(len(missing_keys)),
                    "extra_key_count_v1": int(len(extra_keys)),
                    "status_v1": status,
                }
            )

    required_membership = {"variant_id_v1", "score_field_v1", "fold_id_v1", "group_key_v1", "train_validation_membership_v1"}
    missing_membership = sorted(required_membership.difference(membership.columns))
    if missing_membership:
        failures.append(f"SELECTED_MEMBERSHIP_MISSING_COLUMNS:{missing_membership}")
    else:
        overlap = membership.groupby(["variant_id_v1", "score_field_v1", "fold_id_v1", "group_key_v1"])["train_validation_membership_v1"].nunique()
        if bool((overlap > 1).any()):
            failures.append("SELECTED_MEMBERSHIP_TRAIN_VALIDATION_OVERLAP")

    registry = source_manifest.get("scorefield_registry_v1")
    if not isinstance(registry, list) or not registry:
        failures.append("SELECTED_SOURCE_MANIFEST_SCOREFIELD_REGISTRY_MISSING")
    else:
        for row in registry:
            if bool(row.get("decision_valid_v1")) is not True:
                failures.append(f"SELECTED_SCOREFIELD_DECISION_VALID_FALSE:{row.get('score_field_v1')}")
            if row.get("oof_provenance_status_v1") != "PASS":
                failures.append(f"SELECTED_SCOREFIELD_PROVENANCE_NOT_PASS:{row.get('score_field_v1')}")
    if source_manifest.get("decision_valid_for_pre_optuna_v1") is not True:
        failures.append("SELECTED_SOURCE_MANIFEST_DECISION_VALID_FALSE")
    if source_manifest.get("oof_provenance_status_v1") != "PASS":
        failures.append("SELECTED_SOURCE_MANIFEST_OOF_PROVENANCE_NOT_PASS")

    metric_status = "PASS"
    eval_path = v3_oof_dir / "v3_variant_eval_and_safety_gate_v1.csv"
    if not eval_path.exists():
        metric_status = "FAIL"
        failures.append("SELECTED_METRIC_EVAL_FILE_MISSING")
    else:
        eval_frame = pd.read_csv(eval_path)
        best_eval = eval_frame[eval_frame["variant_id_v1"].astype(str).eq(str(best_variant))].copy() if best_variant is not None and "variant_id_v1" in eval_frame.columns else pd.DataFrame()
        if best_eval.empty:
            metric_status = "FAIL"
            failures.append("SELECTED_METRIC_BEST_VARIANT_ROW_MISSING")
        else:
            metric_row = best_eval.iloc[0]
            for column in ["precision_decision_valid_v1", "worst_loso_decision_valid_v1"]:
                if column not in best_eval.columns or bool(metric_row.get(column)) is not True:
                    metric_status = "FAIL"
                    failures.append(f"SELECTED_METRIC_DENOMINATOR_INVALID:{column}")

    status = "PASS" if not failures else "FAIL"
    return {
        "layer_name": "SELECTED_V3_OOF_ARTIFACT_AUDIT_V1",
        "selected_v3_oof_artifact_root_v1": str(v3_oof_dir),
        "best_variant_id_v1": best_variant,
        "required_file_status_v1": file_status,
        "score_field_audit_v1": per_field,
        "metric_denominator_status_v1": metric_status,
        "status_v1": status,
        "failure_reasons_v1": failures,
    }, failures


def _historical_invalid_v3_artifacts(reports_root: Path, selected_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    selected_resolved = selected_root.expanduser().resolve()
    for marker_name in [
        "v3_oof_score_provenance_reconstruction_or_invalidation_v1.json",
        "oof_score_provenance_validation_v1.json",
        "v3_oof_score_source_manifest_v1.json",
    ]:
        for marker in sorted(reports_root.glob(f"*/{marker_name}")):
            root = marker.parent.expanduser().resolve()
            text = marker.read_text(encoding="utf-8")
            invalid = "INVALID_FOR_OPTUNA_DECISIONING" in text or "FAIL_MISSING_PROVENANCE" in text or '"decision_valid_for_pre_optuna_v1": false' in text
            if not invalid:
                continue
            selected = root == selected_resolved
            status = "SELECTED_INVALID_BLOCKER" if selected else "QUARANTINED_NOT_SELECTED_HISTORY_ONLY"
            rows.append(
                {
                    "artifact_root_v1": str(root),
                    "marker_file_v1": str(marker),
                    "selected_for_decisioning_v1": selected,
                    "status_v1": status,
                }
            )
            if selected:
                failures.append(f"HISTORICAL_INVALID_ARTIFACT_SELECTED:{root}")
    if not rows:
        rows.append(
            {
                "artifact_root_v1": "",
                "marker_file_v1": "",
                "selected_for_decisioning_v1": False,
                "status_v1": "NO_HISTORICAL_INVALID_MARKERS_FOUND",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(), failures


def _fallback_audit() -> tuple[pd.DataFrame, list[str]]:
    files = [
        Path("gx1/scripts/run_r5_2_objective_v3_parallel_rebuild_runner_v1.py"),
        Path("gx1/scripts/materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1.py"),
        Path("gx1/scripts/materialize_parallel_r5_2_v3_and_r6_head_recall_search_v1.py"),
        Path("gx1/scripts/materialize_r6_retrain_from_best_r5_2_objective_v2_variant_v1.py"),
    ]
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    patterns = [
        ("missing_input_raise", re.compile(r"raise (FileNotFoundError|RuntimeError|BlockedMissingRequiredInput)")),
        ("default_path", re.compile(r"DEFAULT_[A-Z0-9_]+\\s*=")),
        ("fallback_default_value", re.compile(r"get\([^\n]+,\s*(False|True|0|0\.0|''|\"\"|\{\}|\[\])")),
        ("fillna_default", re.compile(r"fillna\((False|True|0|0\.0)\)")),
        ("safe_empty_default_metric", re.compile(r"default=1\.0|if final_count == 0|precision = 1\.0 if")),
    ]
    for file in files:
        if not file.exists():
            continue
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            for fallback_type, pattern in patterns:
                if pattern.search(line):
                    classification = "NEEDS_FAIL_CLOSED" if fallback_type in {"safe_empty_default_metric", "fallback_default_value"} else "INTENTIONAL_SAFE_FALLBACK"
                    can_affect = fallback_type in {"safe_empty_default_metric", "default_path", "fallback_default_value"}
                    rows.append(
                        {
                            "file_v1": str(file),
                            "line_v1": lineno,
                            "function_or_context_v1": "source_scan",
                            "fallback_type_v1": fallback_type,
                            "fallback_condition_v1": line.strip(),
                            "result_v1": "see_source",
                            "fail_open_or_closed_v1": "POTENTIAL_FAIL_OPEN" if fallback_type in {"safe_empty_default_metric", "fallback_default_value"} else "FAIL_CLOSED_OR_PINNED_DEFAULT",
                            "can_affect_r5_2_r6_optuna_v1": can_affect,
                            "classification_v1": classification,
                            "required_fix_v1": "ADD_EXPLICIT_STATUS_OR_DENOMINATOR_WARNING" if fallback_type == "safe_empty_default_metric" else "",
                        }
                    )
    # Missing inputs generally raise, but metric empty-denominator defaults need explicit guard before more search.
    if any(row["fallback_type_v1"] == "safe_empty_default_metric" for row in rows):
        hard_failures.append("metric_empty_denominator_defaults_need_guard")
    return pd.DataFrame(rows), hard_failures


def _path_audit() -> tuple[pd.DataFrame, list[str]]:
    files = [
        Path("gx1/scripts/run_r5_2_objective_v3_parallel_rebuild_runner_v1.py"),
        Path("gx1/scripts/materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1.py"),
        Path("gx1/scripts/materialize_parallel_r5_2_v3_and_r6_head_recall_search_v1.py"),
        Path("gx1/scripts/materialize_r6_retrain_from_best_r5_2_objective_v2_variant_v1.py"),
    ]
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8") if file.exists() else ""
        has_cli = "argparse" in text and "--output-dir" in text
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "DEFAULT_" in line or "/home/andre2/GX1_DATA" in line:
                rows.append(
                    {
                        "file_v1": str(file),
                        "line_v1": lineno,
                        "path_or_default_v1": line.strip(),
                        "cli_or_spec_override_present_v1": has_cli,
                        "points_to_canonical_data_v1": "truth_e2e_sanity" in line or "DEFAULT_REPORTS_ROOT" in line,
                        "classification_v1": "ACCEPTABLE_LOCAL_DEFAULT_WITH_OVERRIDE" if has_cli else "SHOULD_BE_CLI_CONFIG",
                        "repro_risk_v1": "PINNED_DEFAULT_CAN_BECOME_STALE" if has_cli else "CANONICALITY_RISK",
                    }
                )
        if not has_cli:
            hard_failures.append(f"no_cli_override:{file}")
    return pd.DataFrame(rows), hard_failures


def _pass_through_audit(v3_oof_dir: Path) -> tuple[dict[str, Any], list[str]]:
    hard_failures: list[str] = []
    references = {
        "v2_added_rows_pass_through_v1": "2/2",
        "v3_added_rows_pass_through_v1": "4/4",
        "rescue_added_rows_pass_through_v1": "6/6",
        "v2_objective_rows_pass_through_v1": "7/7",
    }
    best_lock = _read_json(v3_oof_dir / "best_v3_variant_downstream_r6_input_lock_v1.json")
    ready = bool(best_lock.get("ready_for_downstream_r6_v1"))
    if ready:
        base_flag = best_lock.get("required_r6_base_flag_v1")
        if base_flag != "r5_2_v3_final_base_membership":
            hard_failures.append(f"wrong_v3_base_flag:{base_flag}")
    return {
        "layer_name": "R5_2_R6_PASS_THROUGH_CONSISTENCY_AUDIT_V1",
        "known_pass_through_facts_v1": references,
        "current_v3_oof_downstream_ready_v1": ready,
        "current_v3_oof_block_reason_v1": best_lock.get("failure_reason_v1", "not ready") if not ready else "READY",
        "pre_veto_flags_blocked_as_final_v1": True,
        "raw_unsafe_flags_blocked_v1": True,
        "manifest_actual_rows_agree_v1": "NOT_APPLICABLE_NO_PASSING_V3_OOF_R6_INPUT",
        "hard_failures_v1": hard_failures,
        "pass_through_contract_pass_v1": not hard_failures,
    }, hard_failures


def _metric_audit() -> tuple[dict[str, Any], list[str]]:
    files = [
        Path("gx1/scripts/run_r5_2_objective_v3_parallel_rebuild_runner_v1.py"),
        Path("gx1/scripts/materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1.py"),
    ]
    findings: list[str] = []
    hard_failures: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8") if file.exists() else ""
        if "default=1.0" in text or "final_count == 0" in text or "NO_SELECTED_GROUPS" in text or "precision = 1.0 if" in text:
            findings.append(f"empty_denominator_can_report_1_0:{file}")
        if "denominator_status_v1" not in text or "decision_valid_v1" not in text:
            findings.append(f"denominator_guard_metadata_missing:{file}")
    if findings:
        hard_failures.append("precision_or_worst_loso_empty_denominator_guard_missing")
    return {
        "layer_name": "METRIC_AND_EVAL_CONTRACT_AUDIT_V1",
        "bad_blocks_definition_v1": "selected/base rows with label_should_not_take_v1=true",
        "tail_help_definition_v1": "selected/base rows with tail_10_50_mfe_v1=true",
        "precision_definition_v1": "bad_blocks / selected_or_base_count",
        "worst_loso_definition_v1": "minimum per run_id precision among selected groups",
        "repaired_damage_definition_v1": "selected r6_label_repaired_165_like_runner_v1 or forensic repaired candidate",
        "high_mfe_damage_definitions_v1": ["fifty_plus_mfe_v1", "hundred_plus_mfe_v1", "two_hundred_plus_mfe_v1", "strongest_winner_path_v1"],
        "active_quarantine_contract_v1": "included in foundation with explicit calendar_quarantine_status_v1",
        "wednesday_benchmark_role_v1": "COMPARATOR_ONLY_NOT_TRAINING_TARGET",
        "empty_denominator_findings_v1": findings,
        "hard_failures_v1": hard_failures,
        "metric_contract_pass_v1": not hard_failures,
    }, hard_failures


def _decide(failures: dict[str, list[str]]) -> tuple[str, str]:
    priority = [
        ("selection", "FIX_ACTIVE_SCORE_ARTIFACT_SELECTION_FIRST"),
        ("feature", "FIX_FEATURE_MATRIX_DRIFT_FIRST"),
        ("label", "FIX_LABEL_TARGET_DRIFT_FIRST"),
        ("split", "FIX_OOF_SPLIT_INTEGRITY_FIRST"),
        ("selected_artifact", "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"),
        ("provenance", "FIX_SCORE_PROVENANCE_FIRST"),
        ("fallback", "FIX_FAIL_OPEN_FALLBACKS_FIRST"),
        ("path", "FIX_PATH_SPEC_RESOLUTION_FIRST"),
        ("metric", "FIX_METRIC_CONTRACT_FIRST"),
        ("history", "FIX_ACTIVE_SCORE_ARTIFACT_SELECTION_FIRST"),
        ("source_graph", "FIX_PATH_SPEC_RESOLUTION_FIRST"),
        ("pass_through", "FIX_METRIC_CONTRACT_FIRST"),
    ]
    for key, decision in priority:
        if failures.get(key):
            return decision, decision
    return "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA", "INSTALL_OPTUNA_AND_RUN_CONSTRAINED_OBJECTIVE_SEARCH"


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_oof_dir: Path = DEFAULT_V3_OOF_DIR,
    v3_in_sample_dir: Path = DEFAULT_V3_IN_SAMPLE_DIR,
    optuna_dir: Path = DEFAULT_OPTUNA_DIR,
    selected_v3_oof_artifact_root: Path | None = None,
    active_score_artifact_selection: Path | None = None,
    require_explicit_artifact_selection: bool = False,
    reject_invalidated_decision_scorefields: bool = True,
    fail_on_missing_oof_provenance: bool = True,
    fail_on_invalid_metric_denominator: bool = True,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_AUDIT"
    selected_root, selection_contract, selection_failures = _resolve_selected_v3_root(
        v3_oof_dir=v3_oof_dir,
        selected_v3_oof_artifact_root=selected_v3_oof_artifact_root,
        active_score_artifact_selection=active_score_artifact_selection,
        require_explicit_artifact_selection=require_explicit_artifact_selection,
    )
    if selected_root is None or any(failure.startswith("SELECTED_V3_OOF_ARTIFACT_ROOT_MISSING") for failure in selection_failures):
        return _write_selection_blocked_output(output_dir, selection_contract, selection_failures)
    output_dir.mkdir(parents=True, exist_ok=False)
    v3_oof_dir = selected_root
    missing_root_inputs = [
        name
        for name in ["manifest_v1.json", "v3_variant_outputs_index_v1.csv", "best_v3_variant_downstream_r6_input_lock_v1.json"]
        if not (v3_oof_dir / name).exists()
    ]
    if missing_root_inputs:
        selection_failures = [*selection_failures, f"SELECTED_ARTIFACT_NOT_A_V3_OOF_ROOT:{missing_root_inputs}"]
        output_dir.rmdir()
        return _write_selection_blocked_output(output_dir, selection_contract, selection_failures)
    manifest, score, label, variant_index, pred, provenance = _load_inputs(v3_oof_dir)
    feature_audit, feature_summary, feature_failures = _feature_audit(score)
    features, _families = _feature_names(score)
    source_graph, graph_failures = _source_graph(v3_oof_dir, v3_in_sample_dir, optuna_dir, manifest, variant_index)
    target_audit, label_failures = _target_audit(score.copy(), label.copy(), features)
    split_audit, split_failures = _split_audit(score)
    provenance_audit, provenance_failures = _oof_provenance_audit(pred, provenance)
    selected_artifact_audit, selected_artifact_failures = _selected_v3_oof_artifact_audit(v3_oof_dir, pred)
    historical_invalid_artifacts, history_failures = _historical_invalid_v3_artifacts(reports_root, v3_oof_dir)
    fallback_audit, fallback_failures = _fallback_audit()
    path_audit, path_failures = _path_audit()
    pass_through_audit, pass_failures = _pass_through_audit(v3_oof_dir)
    metric_audit, metric_failures = _metric_audit()
    if not fail_on_missing_oof_provenance:
        provenance_failures.append("MISSING_OOF_PROVENANCE_FAIL_CLOSED_FLAG_DISABLED")
    if not reject_invalidated_decision_scorefields:
        selected_artifact_failures.append("INVALIDATED_DECISION_SCOREFIELD_REJECTION_FLAG_DISABLED")
    if not fail_on_invalid_metric_denominator:
        metric_failures.append("INVALID_METRIC_DENOMINATOR_FAIL_CLOSED_FLAG_DISABLED")
    failures = {
        "selection": selection_failures,
        "source_graph": graph_failures,
        "feature": feature_failures,
        "label": label_failures,
        "split": split_failures,
        "selected_artifact": selected_artifact_failures,
        "provenance": provenance_failures,
        "fallback": fallback_failures,
        "path": path_failures,
        "pass_through": pass_failures,
        "metric": metric_failures,
        "history": history_failures,
    }
    decision, next_action = _decide(failures)
    go_no_go = {
        "layer_name": "HIDDEN_DRIFT_SUMMARY_AND_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "failure_counts_v1": {key: len(value) for key, value in failures.items()},
        "hard_failures_v1": failures,
        "foundation_clean_ready_for_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "foundation_clean_for_constrained_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "blocked_action_v1": "RUN_OPTUNA_OR_R6_BEFORE_FOUNDATION_AUDIT_FIX",
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": decision,
        "next_action_v1": next_action,
        "foundation_rows_v1": int(len(score)),
        "label_rows_v1": int(len(label)),
        "feature_count_v1": int(feature_summary["feature_count_v1"]),
        "oof_provenance_missing_count_v1": int(len(provenance_failures)),
        "selected_v3_oof_provenance_status_v1": selected_artifact_audit["status_v1"],
        "metric_contract_failure_count_v1": int(len(metric_failures)),
        "historical_invalid_v3_artifact_status_v1": "SELECTED_INVALID_BLOCKER" if history_failures else "QUARANTINED_NOT_SELECTED_HISTORY_ONLY",
        "foundation_clean_ready_for_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "foundation_clean_for_constrained_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "hard_status_v1": "BEVIST" if decision != "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA" else "INDIKERT",
    }
    status = {**summary, "status_v1": decision}
    manifest_out = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_dirs_v1": {
            "v3_oof_dir_v1": str(v3_oof_dir),
            "selected_v3_oof_artifact_root_v1": str(v3_oof_dir),
            "v3_in_sample_dir_v1": str(v3_in_sample_dir),
            "optuna_reference_dir_v1": str(optuna_dir),
        },
        "output_files_v1": {name: str(output_dir / name) for name in OUTPUT_FILES},
    }
    consistency = [
        {"check_v1": "active_artifact_selection", "status_v1": "PASS" if not selection_failures else "FAIL", "evidence_v1": len(selection_failures)},
        {"check_v1": "source_graph", "status_v1": "PASS" if not graph_failures else "FAIL", "evidence_v1": len(graph_failures)},
        {"check_v1": "feature_matrix", "status_v1": "PASS" if not feature_failures else "FAIL", "evidence_v1": len(feature_failures)},
        {"check_v1": "label_target", "status_v1": "PASS" if not label_failures else "FAIL", "evidence_v1": len(label_failures)},
        {"check_v1": "split_integrity", "status_v1": "PASS" if not split_failures else "FAIL", "evidence_v1": len(split_failures)},
        {"check_v1": "selected_v3_oof_artifact", "status_v1": "PASS" if not selected_artifact_failures else "FAIL", "evidence_v1": len(selected_artifact_failures)},
        {"check_v1": "oof_score_provenance", "status_v1": "PASS" if not provenance_failures else "FAIL", "evidence_v1": len(provenance_failures)},
        {"check_v1": "historical_invalid_v3_artifacts", "status_v1": "PASS" if not history_failures else "FAIL", "evidence_v1": len(history_failures)},
        {"check_v1": "fallback_fail_closed", "status_v1": "PASS" if not fallback_failures else "FAIL", "evidence_v1": len(fallback_failures)},
        {"check_v1": "metric_contract", "status_v1": "PASS" if not metric_failures else "FAIL", "evidence_v1": len(metric_failures)},
    ]
    _write_json(output_dir / "active_score_artifact_selection_v1.json", selection_contract)
    _write_json(output_dir / "active_canonical_source_graph_lock_v1.json", source_graph)
    feature_audit.to_csv(output_dir / "feature_matrix_truth_audit_v1.csv", index=False)
    _write_json(output_dir / "target_and_label_table_audit_v1.json", target_audit)
    _write_json(output_dir / "grouped_oof_and_split_integrity_audit_v1.json", split_audit)
    _write_json(output_dir / "selected_v3_oof_artifact_audit_v1.json", selected_artifact_audit)
    provenance_audit.to_csv(output_dir / "oof_score_provenance_audit_v1.csv", index=False)
    historical_invalid_artifacts.to_csv(output_dir / "historical_invalid_v3_artifacts_v1.csv", index=False)
    fallback_audit.to_csv(output_dir / "fallback_fail_closed_audit_v1.csv", index=False)
    path_audit.to_csv(output_dir / "hardcoded_path_and_spec_resolution_audit_v1.csv", index=False)
    _write_json(output_dir / "r5_2_r6_pass_through_consistency_audit_v1.json", pass_through_audit)
    _write_json(output_dir / "metric_and_eval_contract_audit_v1.json", metric_audit)
    _write_json(output_dir / "hidden_drift_summary_and_go_no_go_v1.json", go_no_go)
    _write_json(output_dir / "next_action_lock_v1.json", next_lock)
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", status)
    _write_json(output_dir / "manifest_v1.json", manifest_out)
    _write_csv(output_dir / "consistency_audit_v1.csv", consistency)
    report = "\n".join(
        [
            "# Foundation Integrity And Hidden Drift Audit Before Optuna V1",
            "",
            f"Decision: `{decision}`",
            f"Next action: `{next_action}`",
            "",
            f"- Foundation rows: `{len(score)}`",
            f"- Label rows: `{len(label)}`",
            f"- Feature count: `{feature_summary['feature_count_v1']}`",
            f"- Selected V3 OOF artifact: `{v3_oof_dir}`",
            f"- Selected V3 OOF provenance: `{selected_artifact_audit['status_v1']}`",
            f"- OOF provenance failures: `{len(provenance_failures)}`",
            f"- Metric contract failures: `{len(metric_failures)}`",
            f"- Historical invalid V3 artifacts: `{summary['historical_invalid_v3_artifact_status_v1']}`",
            "",
            "This audit did not run Optuna, training, R6, or materialize a new feature surface.",
        ]
    )
    (output_dir / "report_v1.md").write_text(report + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v3-oof-dir", type=Path, default=DEFAULT_V3_OOF_DIR)
    parser.add_argument("--selected-v3-oof-artifact-root", type=Path, default=None)
    parser.add_argument("--active-score-artifact-selection", type=Path, default=None)
    parser.add_argument("--require-explicit-artifact-selection", action="store_true")
    parser.add_argument("--reject-invalidated-decision-scorefields", action="store_true")
    parser.add_argument("--fail-on-missing-oof-provenance", action="store_true")
    parser.add_argument("--fail-on-invalid-metric-denominator", action="store_true")
    parser.add_argument("--v3-in-sample-dir", type=Path, default=DEFAULT_V3_IN_SAMPLE_DIR)
    parser.add_argument("--optuna-reference-dir", type=Path, default=DEFAULT_OPTUNA_DIR)
    args = parser.parse_args(argv)
    materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_oof_dir=args.v3_oof_dir,
        v3_in_sample_dir=args.v3_in_sample_dir,
        optuna_dir=args.optuna_reference_dir,
        selected_v3_oof_artifact_root=args.selected_v3_oof_artifact_root,
        active_score_artifact_selection=args.active_score_artifact_selection,
        require_explicit_artifact_selection=args.require_explicit_artifact_selection,
        reject_invalidated_decision_scorefields=args.reject_invalidated_decision_scorefields or not args.require_explicit_artifact_selection,
        fail_on_missing_oof_provenance=args.fail_on_missing_oof_provenance or not args.require_explicit_artifact_selection,
        fail_on_invalid_metric_denominator=args.fail_on_invalid_metric_denominator or not args.require_explicit_artifact_selection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
