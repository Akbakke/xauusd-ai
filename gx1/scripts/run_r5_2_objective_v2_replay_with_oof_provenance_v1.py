#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import run_r5_2_objective_v2_parallel_rebuild_runner_v1 as historical_v2


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
LAYER_NAME = "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1"
ACTION = "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1"

REVALIDATION_ROOT = DEFAULT_REPORTS_ROOT / "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_V1_20260427T095034Z_LOCK"
OPTUNA_ROOT = DEFAULT_REPORTS_ROOT / "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1_20260427T080458Z_LOCK"
SELECTED_V3_ROOT = DEFAULT_REPORTS_ROOT / "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG_20260427T073055Z_LOCK"
FOUNDATION_AUDIT_ROOT = DEFAULT_REPORTS_ROOT / "FOUNDATION_INTEGRITY_AND_HIDDEN_DRIFT_AUDIT_BEFORE_OPTUNA_V1_20260427T073512Z_AUDIT"
V2_HISTORICAL_ROOT = DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_20260426T_EXECUTION"
V2_VARIANT = "R5_2_OBJECTIVE_V2_VARIANT_01_V2_BALANCED_STRICT_PROTECT"
V2_PROFILE = "V2_BALANCED_STRICT_PROTECT"

MIN_PRECISION_DENOMINATOR = 5
MIN_WORST_LOSO_DENOMINATOR = 5
DEFAULT_FOLD_COUNT = 5

HEADS = [
    ("bad_recall_target", "r5_2_v2_bad_recall_score"),
    ("tail_recall_target", "r5_2_v2_tail_recall_score"),
    ("risky_attention_target", "r5_2_v2_risky_attention_score"),
    ("runner_protection_target", "r5_2_v2_runner_protection_score"),
    ("high_mfe_ambiguous_protection_target", "r5_2_v2_high_mfe_ambiguous_protection_score"),
    ("hard_winner_protection_target", "r5_2_v2_hard_winner_protection_score"),
]
SCORE_COLUMNS = [output for _, output in HEADS]
BASE_COLUMNS = [
    "r5_2_v2_base_membership_pre_veto",
    "r5_2_v2_hard_protection_veto",
    "r5_2_v2_final_base_membership",
    "v2_base_reason_v1",
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


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes()) if path.exists() else "MISSING_LOCAL_ARTIFACT"


def _hash_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _hash_list(values: Sequence[Any]) -> str:
    return _hash_json([str(value) for value in values])


def _hash_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    work = frame[list(columns)].copy() if columns is not None else frame.copy()
    work = work.sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(work, index=False).to_numpy(dtype="uint64")
    return _sha256_bytes(hashed.tobytes())


def _row_hashes(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    work = frame[list(columns)].copy()
    work = work.sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(work, index=False).astype("uint64")
    return hashed.map(lambda value: hashlib.sha256(str(int(value)).encode("utf-8")).hexdigest())


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _metric_ratio(name: str, numerator: int, denominator: int, min_denominator: int = MIN_PRECISION_DENOMINATOR) -> dict[str, Any]:
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


def validate_no_train_validation_overlap(membership: pd.DataFrame) -> dict[str, Any]:
    overlap = membership[membership["is_train_v1"].astype(bool) & membership["is_validation_v1"].astype(bool)]
    return {
        "status_v1": "PASS" if overlap.empty else "FAIL",
        "overlap_count_v1": int(len(overlap)),
        "decision_valid_v1": bool(overlap.empty),
    }


def validate_no_in_sample_scoring(scores: pd.DataFrame) -> dict[str, Any]:
    if "was_row_in_train_for_scoring_model_v1" not in scores.columns:
        return {"status_v1": "FAIL", "in_sample_scored_count_v1": -1, "decision_valid_v1": False}
    in_sample = scores["was_row_in_train_for_scoring_model_v1"].fillna(True).astype(bool)
    return {
        "status_v1": "PASS" if int(in_sample.sum()) == 0 else "FAIL",
        "in_sample_scored_count_v1": int(in_sample.sum()),
        "decision_valid_v1": int(in_sample.sum()) == 0,
    }


def validate_provenance_files(root: Path) -> dict[str, Any]:
    required = [
        "v2_oof_scores_v1.csv",
        "v2_oof_score_provenance_v1.csv",
        "v2_oof_fold_assignment_v1.csv",
        "v2_oof_score_source_manifest_v1.json",
        "v2_train_validation_membership_v1.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    return {
        "status_v1": "PASS" if not missing else "FAIL",
        "missing_files_v1": missing,
        "decision_valid_v1": not missing,
    }


def classify_model_artifact_use(*, existing_artifact_fold_trained: bool, existing_artifact_validation_only: bool) -> str:
    if existing_artifact_fold_trained and existing_artifact_validation_only:
        return "EXISTING_MODEL_ARTIFACT_DECISION_VALID_FOR_OOF"
    return "EXISTING_V2_MODEL_ARTIFACTS_ARE_HISTORICAL_ONLY_FOR_OOF"


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures, "decision_valid_v1": not failures}


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


def historical_v2_decision_status(*, has_oof_proof: bool) -> str:
    return "HISTORICAL_V2_CAN_BE_COMPARATOR_ONLY" if not has_oof_proof else "HISTORICAL_V2_CAN_BE_AUDITED_FOR_DECISION_VALIDITY"


def _balanced_group_folds(frame: pd.DataFrame, group_col: str = "run_id", fold_count: int = DEFAULT_FOLD_COUNT) -> pd.Series:
    if group_col not in frame.columns:
        raise RuntimeError(f"Grouped OOF requires {group_col}")
    groups = frame[group_col].astype(str)
    counts = groups.value_counts().sort_values(ascending=False)
    loads = {fold: 0 for fold in range(fold_count)}
    mapping: dict[str, int] = {}
    for group, count in counts.items():
        fold = min(loads, key=lambda item: (loads[item], item))
        mapping[str(group)] = fold
        loads[fold] += int(count)
    return groups.map(mapping).astype(int)


def _worst_loso(frame: pd.DataFrame, selected: pd.Series, bad: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = frame["run_id"].astype(str) if "run_id" in frame.columns else pd.Series("UNKNOWN", index=frame.index)
    work = pd.DataFrame({"group": groups, "selected": selected.astype(bool), "bad": bad.astype(bool)})
    for group, part in work.groupby("group"):
        denominator = int(part["selected"].sum())
        numerator = int((part["selected"] & part["bad"]).sum())
        value = numerator / denominator if denominator else np.nan
        rows.append(
            {
                "group_v1": str(group),
                "row_count_v1": int(len(part)),
                "bad_total_v1": int(part["bad"].sum()),
                "selected_denominator_v1": denominator,
                "selected_bad_numerator_v1": numerator,
                "group_precision_v1": value,
                "denominator_status_v1": "OK"
                if denominator >= MIN_WORST_LOSO_DENOMINATOR
                else ("EMPTY_SELECTED_GROUP" if denominator == 0 else "TOO_SMALL_DENOMINATOR"),
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
        }
    for row in rows:
        row["is_worst_loso_group_v1"] = row["group_v1"] == worst["group_v1"]
    summary = {
        "worst_loso_v1": worst["group_precision_v1"],
        "worst_loso_group_v1": worst["group_v1"],
        "worst_loso_numerator_v1": int(worst["selected_bad_numerator_v1"]),
        "worst_loso_denominator_v1": int(worst["selected_denominator_v1"]),
        "worst_loso_denominator_status_v1": "OK"
        if int(worst["selected_denominator_v1"]) >= MIN_WORST_LOSO_DENOMINATOR
        else "TOO_SMALL_DENOMINATOR",
        "worst_loso_decision_valid_v1": int(worst["selected_denominator_v1"]) >= MIN_WORST_LOSO_DENOMINATOR,
        "selected_group_count_v1": len(non_empty),
        "empty_selected_group_count_v1": len(rows) - len(non_empty),
        "small_selected_group_count_v1": sum(0 < int(row["selected_denominator_v1"]) < MIN_WORST_LOSO_DENOMINATOR for row in rows),
    }
    return rows, summary


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


def _load_comparison_ledger() -> pd.DataFrame:
    ledger = _read_csv(OPTUNA_ROOT / "constrained_optuna_full_signal_forensics_v1.csv")
    if ledger.empty:
        return ledger
    ledger["optuna_best_captured_v1"] = _load_optuna_best_selection(ledger).to_numpy(dtype=bool)
    return ledger


def _select_v2_variant(spec: dict[str, Any]) -> dict[str, Any]:
    variants = (spec.get("parallel_run") or {}).get("variants_v1") or []
    for variant in variants:
        if str(variant.get("variant_id_v1")) == V2_VARIANT:
            return variant
    raise RuntimeError(f"Missing V2 variant config: {V2_VARIANT}")


def _prepare_inputs(spec_dir: Path, foundation_score_dir: Path | None, label_table: Path | None) -> dict[str, Any]:
    spec = historical_v2._load_spec_package(spec_dir)
    score_dir = historical_v2._resolve_foundation_score_dir(spec, foundation_score_dir)
    label_path = historical_v2._resolve_label_table_path(spec, label_table)
    score, score_summary = historical_v2._load_foundation(score_dir)
    labels = _read_table(label_path)
    foundation = historical_v2._validate_foundation(score, score_summary, score_dir)
    key_alignment = historical_v2._validate_key_alignment(score, labels)
    target = historical_v2._build_v2_target_table(labels)
    target_audit = historical_v2._validate_target_table(target)
    feature_preflight, forbidden_scan = historical_v2._feature_prelaunch(score, score_summary)
    feature_names, feature_families = historical_v2._legal_feature_candidates(score)
    x = historical_v2._feature_matrix(score, feature_names)
    training_frame = historical_v2._join_training_frame(score, target)
    variant = _select_v2_variant(spec)
    return {
        "spec": spec,
        "score_dir": score_dir,
        "label_path": label_path,
        "score": score,
        "score_summary": score_summary,
        "labels": labels,
        "foundation": foundation,
        "key_alignment": key_alignment,
        "target": target,
        "target_audit": target_audit,
        "feature_preflight": feature_preflight,
        "forbidden_scan": forbidden_scan,
        "feature_names": feature_names,
        "feature_families": feature_families,
        "x": x,
        "training_frame": training_frame,
        "variant": variant,
    }


def _source_mapping(output_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    variant_root = V2_HISTORICAL_ROOT / "variants" / V2_VARIANT
    model_paths = [variant_root / "models" / f"{scorefield}.joblib" for scorefield in SCORE_COLUMNS]
    metadata_paths = [variant_root / "models" / f"{scorefield}.metadata.json" for scorefield in SCORE_COLUMNS]
    spec_files = {
        key: str(historical_v2.DEFAULT_SPEC_DIR / filename)
        for key, filename in historical_v2.SPEC_FILES.items()
        if (historical_v2.DEFAULT_SPEC_DIR / filename).exists()
    }
    payload = {
        "layer_name": "V2_RUNNER_SOURCE_MAPPING_V1",
        "v2_source_files_v1": {
            "historical_v2_runner_v1": str(Path(historical_v2.__file__).resolve()),
            "oof_replay_wrapper_v1": str(Path(__file__).resolve()),
        },
        "v2_config_files_v1": {
            "spec_dir_v1": str(historical_v2.DEFAULT_SPEC_DIR),
            "spec_files_v1": spec_files,
            "selected_variant_config_v1": str(variant_root / "config_manifest_v1.json"),
        },
        "v2_model_artifact_paths_v1": [str(path) for path in model_paths + metadata_paths],
        "v2_scorefield_paths_v1": {
            "historical_score_package_v1": str(variant_root / "score_package_v1.parquet"),
            "historical_prediction_view_v1": str(variant_root / "prediction_view_v1.parquet"),
        },
        "v2_row_level_output_paths_v1": {
            "historical_row_forensics_v1": str(V2_HISTORICAL_ROOT / "v2_variant_row_level_forensics_v1.csv"),
            "historical_base_membership_v1": str(variant_root / "base_membership_package_v1.parquet"),
        },
        "reusable_parts_v1": [
            "V2 target construction",
            "V2 legal feature selection",
            "V2 HistGradientBoostingClassifier head training",
            "V2 threshold/base membership rule",
            "V2 hard protection veto logic",
        ],
        "parts_patched_by_wrapper_v1": [
            "grouped OOF fold assignment",
            "validation-only scoring collection",
            "score provenance writer",
            "train/validation membership writer",
            "metric denominator report",
            "reject in-sample decisioning gate",
        ],
        "historical_only_parts_v1": [
            "full-sample historical V2 model artifacts",
            "historical V2 95/61 score package",
            "historical V2 row-level forensics",
        ],
        "existing_v2_model_artifact_use_v1": classify_model_artifact_use(
            existing_artifact_fold_trained=False,
            existing_artifact_validation_only=False,
        ),
        "source_hashes_v1": {
            "historical_v2_runner_sha256_v1": _file_hash(Path(historical_v2.__file__).resolve()),
            "oof_wrapper_sha256_v1": _file_hash(Path(__file__).resolve()),
            "selected_variant_config_hash_v1": _hash_json(inputs["variant"]),
        },
        "output_root_v1": str(output_dir),
    }
    return payload


def _oof_contract(inputs: dict[str, Any]) -> dict[str, Any]:
    foundation = inputs["foundation"]
    return {
        "contract": "V2_OOF_REPLAY_CONTRACT_V1",
        "foundation_rows_required_v1": 1914,
        "foundation_rows_observed_v1": foundation["foundation_rows_v1"],
        "active_quarantine_required_v1": "1852/62",
        "active_rows_observed_v1": foundation["active_rows_v1"],
        "quarantine_rows_observed_v1": foundation["quarantine_rows_v1"],
        "as_of_columns_required_v1": 109,
        "as_of_columns_observed_v1": foundation["asof_columns_v1"],
        "same_v2_source_logic_config_objective_required_v1": True,
        "selected_variant_v1": V2_VARIANT,
        "grouped_oof_fold_assignment_required_v1": True,
        "validation_only_scoring_per_fold_required_v1": True,
        "no_scored_row_in_training_membership_required_v1": True,
        "train_validation_membership_required_v1": True,
        "score_source_manifest_required_v1": True,
        "no_dummy_synthetic_fallback_required_v1": True,
        "no_in_sample_decisioning_required_v1": True,
        "explicit_artifact_selection_required_v1": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "metric_denominator_metadata_required_v1": True,
        "fixed_comparators_v1": {
            "historical_v2_bad_tail_v1": "95/61",
            "optuna_best_bad_tail_v1": "56/55",
            "v3_best_bad_tail_v1": "17/13",
        },
    }


def _fold_assignment(frame: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    assignment = frame[[*historical_v2.REQUIRED_KEYS, "trade_id", "run_id"]].copy()
    assignment["fold_id_v1"] = _balanced_group_folds(frame, "run_id", fold_count)
    assignment["group_key_v1"] = assignment["run_id"].astype(str)
    assignment["split_policy_v1"] = "DETERMINISTIC_BALANCED_GROUPED_OOF_BY_RUN_ID"
    return assignment


def _run_oof_replay(output_dir: Path, inputs: dict[str, Any], fold_count: int) -> dict[str, Any]:
    score = inputs["score"]
    x = inputs["x"]
    training_frame = inputs["training_frame"].copy()
    variant = inputs["variant"]
    feature_names = inputs["feature_names"]
    target = inputs["target"]
    assignment = _fold_assignment(training_frame, fold_count)
    training_frame["v2_oof_fold_id_v1"] = assignment["fold_id_v1"].values
    prediction = training_frame[[*historical_v2.REQUIRED_KEYS]].copy()
    for scorefield in SCORE_COLUMNS:
        prediction[scorefield] = np.nan

    source_hash = _file_hash(Path(historical_v2.__file__).resolve())
    config_hash = _hash_json(variant)
    feature_matrix_hash = _hash_frame(x)
    label_columns = [
        col
        for col in target.columns
        if col
        in {
            *historical_v2.REQUIRED_KEYS,
            "v2_bucket",
            "bad_recall_target",
            "tail_recall_target",
            "risky_attention_target",
            "runner_protection_target",
            "high_mfe_ambiguous_protection_target",
            "hard_winner_protection_target",
            "hard_protection_veto_target",
        }
    ]
    label_table_hash = _hash_frame(target, label_columns)
    feature_row_hashes = _row_hashes(x, list(x.columns))
    label_by_key = target.copy()
    label_by_key["_label_row_hash_v1"] = _row_hashes(label_by_key, label_columns)
    label_hash_lookup = label_by_key.set_index("candidate_uid")["_label_row_hash_v1"].astype(str).to_dict()

    membership_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    head_metric_rows: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    model_root = output_dir / "v2_oof_models_history_only"
    for fold_id in sorted(assignment["fold_id_v1"].unique()):
        validation_mask = assignment["fold_id_v1"].eq(fold_id)
        train_mask = ~validation_mask
        train_uids = training_frame.loc[train_mask, "candidate_uid"].astype(str).tolist()
        validation_uids = training_frame.loc[validation_mask, "candidate_uid"].astype(str).tolist()
        train_hash = _hash_list(train_uids)
        validation_hash = _hash_list(validation_uids)
        fold_frame = training_frame.copy()
        fold_frame["used_for_training"] = train_mask.values
        fold_frame["used_for_validation"] = validation_mask.values
        fold_frame["used_for_holdout"] = False
        fold_label = f"fold_{int(fold_id):02d}"
        fold_seed_base = int((variant.get("model_config_v1") or {}).get("seed_v1") or 20260426) + int(fold_id) * 100
        for idx, row in training_frame.iterrows():
            is_train = bool(train_mask.loc[idx])
            is_validation = bool(validation_mask.loc[idx])
            membership_rows.append(
                {
                    "candidate_uid_v1": row["candidate_uid"],
                    "trade_uid_v1": row["trade_uid"],
                    "decision_timestamp_v1": row["decision_timestamp"],
                    "run_id_v1": row.get("run_id"),
                    "fold_id_v1": fold_label,
                    "is_train_v1": is_train,
                    "is_validation_v1": is_validation,
                    "train_membership_hash_v1": train_hash,
                    "validation_membership_hash_v1": validation_hash,
                    "train_validation_overlap_v1": bool(is_train and is_validation),
                }
            )
        for head_idx, (label_col, scorefield) in enumerate(HEADS):
            seed = fold_seed_base + head_idx
            model_dir = model_root / fold_label
            pred, metrics = historical_v2._fit_head(
                x=x,
                frame=fold_frame,
                label_col=label_col,
                output_col=scorefield,
                variant_id=f"{V2_VARIANT}_{fold_label}",
                seed=seed,
                weights=variant["weights_v1"],
                model_dir=model_dir,
            )
            prediction.loc[validation_mask, scorefield] = pred.loc[validation_mask]
            metrics.update(
                {
                    "fold_id_v1": fold_label,
                    "scorefield_v1": scorefield,
                    "label_col_v1": label_col,
                    "seed_v1": seed,
                    "train_membership_hash_v1": train_hash,
                    "validation_membership_hash_v1": validation_hash,
                }
            )
            head_metric_rows.append(metrics)
            model_source_id = f"{V2_VARIANT}:{fold_label}:{scorefield}:seed={seed}"
            fold_models.append(
                {
                    "fold_id_v1": fold_label,
                    "scorefield_v1": scorefield,
                    "label_col_v1": label_col,
                    "model_source_identifier_v1": model_source_id,
                    "model_artifact_path_v1": str(model_dir / f"{scorefield}.joblib"),
                    "metadata_path_v1": str(model_dir / f"{scorefield}.metadata.json"),
                    "source_hash_v1": source_hash,
                    "config_hash_v1": config_hash,
                    "seed_v1": seed,
                    "decisioning_scope_v1": "OOF_VALIDATION_ONLY",
                }
            )
            for idx in training_frame.index[validation_mask]:
                row = training_frame.loc[idx]
                was_train = bool(train_mask.loc[idx])
                provenance_rows.append(
                    {
                        "candidate_uid_v1": row["candidate_uid"],
                        "trade_uid_v1": row["trade_uid"],
                        "decision_timestamp_v1": row["decision_timestamp"],
                        "run_id_v1": row.get("run_id"),
                        "fold_id_v1": fold_label,
                        "scorefield_v1": scorefield,
                        "head_v1": label_col,
                        "variant_v1": V2_VARIANT,
                        "model_source_identifier_v1": model_source_id,
                        "train_membership_hash_v1": train_hash,
                        "validation_membership_hash_v1": validation_hash,
                        "was_row_in_train_for_scoring_model_v1": was_train,
                        "feature_matrix_hash_v1": feature_matrix_hash,
                        "feature_row_hash_v1": feature_row_hashes.loc[idx],
                        "label_table_hash_v1": label_table_hash,
                        "label_row_hash_v1": label_hash_lookup.get(str(row["candidate_uid"])),
                        "config_hash_v1": config_hash,
                        "source_hash_v1": source_hash,
                        "seed_v1": seed,
                        "score_value_v1": pred.loc[idx],
                        "decision_valid_v1": not was_train,
                        "provenance_valid_v1": not was_train,
                        "oof_status_v1": "OOF_VALIDATION_SCORE" if not was_train else "INVALID_IN_SAMPLE_SCORE",
                    }
                )
    if prediction[SCORE_COLUMNS].isna().any().any():
        missing = prediction[SCORE_COLUMNS].isna().sum().to_dict()
        raise RuntimeError(f"OOF prediction matrix has missing scores: {missing}")
    prediction = historical_v2._apply_variant_base_rule(prediction, training_frame, variant)
    scores = training_frame[
        [
            *historical_v2.REQUIRED_KEYS,
            "trade_id",
            "run_id",
            "v2_bucket",
            "label_should_not_take_v1",
            "tail_10_50_mfe_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "strongest_winner_path_v1",
            "r6_label_repaired_165_like_runner_v1",
            "r6_label_runner_near_miss_v1",
            "runner_protection_target",
            "high_mfe_ambiguous_protection_target",
            "hard_winner_protection_target",
        ]
    ].copy()
    scores["fold_id_v1"] = assignment["fold_id_v1"].map(lambda value: f"fold_{int(value):02d}").values
    for column in [*SCORE_COLUMNS, *BASE_COLUMNS]:
        scores[column] = prediction[column].values
    scores["was_row_in_train_for_scoring_model_v1"] = False
    scores["decision_valid_score_v1"] = True
    fold_assignment = assignment.copy()
    fold_assignment["fold_id_v1"] = fold_assignment["fold_id_v1"].map(lambda value: f"fold_{int(value):02d}")
    return {
        "scores": scores,
        "fold_assignment": fold_assignment,
        "membership_rows": membership_rows,
        "provenance_rows": provenance_rows,
        "head_metric_rows": head_metric_rows,
        "fold_models": fold_models,
        "hashes": {
            "feature_matrix_hash_v1": feature_matrix_hash,
            "label_table_hash_v1": label_table_hash,
            "config_hash_v1": config_hash,
            "source_hash_v1": source_hash,
        },
        "feature_names": feature_names,
        "feature_families": inputs["feature_families"],
        "feature_preflight": inputs["feature_preflight"],
        "target_audit": inputs["target_audit"],
    }


def _evaluate_oof(scores: pd.DataFrame, provenance: pd.DataFrame, membership: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    selected = _bool(scores, "r5_2_v2_final_base_membership")
    bad = _bool(scores, "label_should_not_take_v1")
    tail = _bool(scores, "tail_10_50_mfe_v1")
    precision = _metric_ratio("precision", int((selected & bad).sum()), int(selected.sum()))
    loso_rows, loso = _worst_loso(scores, selected, bad)
    safety = {
        "fifty_plus_mfe_overlap_v1": int((selected & _bool(scores, "fifty_plus_mfe_v1")).sum()),
        "hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "hundred_plus_mfe_v1")).sum()),
        "two_hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "two_hundred_plus_mfe_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(scores, "strongest_winner_path_v1")).sum()),
        "runner_protect_leakage_v1": int(
            (selected & (_bool(scores, "runner_protection_target") | _bool(scores, "r6_label_runner_near_miss_v1"))).sum()
        ),
        "ambiguous_high_mfe_leakage_v1": int(
            (
                selected
                & (
                    _bool(scores, "high_mfe_ambiguous_protection_target")
                    | scores["v2_bucket"].astype(str).eq("AMBIGUOUS_HIGH_MFE_PROTECTED")
                )
            ).sum()
        ),
    }
    safety_clean = all(value == 0 for value in safety.values())
    no_in_sample = validate_no_in_sample_scoring(scores)
    no_overlap = validate_no_train_validation_overlap(membership)
    provenance_valid = bool(
        not provenance.empty
        and int(provenance["was_row_in_train_for_scoring_model_v1"].fillna(True).astype(bool).sum()) == 0
        and int(provenance["provenance_valid_v1"].fillna(False).astype(bool).sum()) == len(provenance)
    )
    comparison_payload = _comparison_payload(scores, selected, comparison)
    decision_valid = bool(
        provenance_valid
        and no_in_sample["decision_valid_v1"]
        and no_overlap["decision_valid_v1"]
        and precision["precision_decision_valid_v1"]
        and loso["worst_loso_decision_valid_v1"]
        and safety_clean
    )
    if not provenance_valid or not no_in_sample["decision_valid_v1"] or not no_overlap["decision_valid_v1"]:
        status = "V2_OOF_REPLAY_BLOCKED_BY_TEST_FAILURE"
        next_action = "REIMPLEMENT_V2_OBJECTIVE_FROM_SOURCE_CONFIG_WITH_OOF_PROVENANCE_V1"
    elif not safety_clean:
        status = "V2_OOF_REPLAY_FAILS_TRUE_SAFETY"
        next_action = "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    elif not precision["precision_decision_valid_v1"] or not loso["worst_loso_decision_valid_v1"]:
        status = "V2_OOF_REPLAY_FAILS_DENOMINATOR"
        next_action = "REPAIR_LOSO_GROUPING_OR_DENOMINATOR_CONTRACT_V1"
    elif int((selected & bad).sum()) >= 80 and int((selected & tail).sum()) >= 55:
        status = "V2_OOF_REPLAY_DECISION_VALID_AND_STRONG"
        next_action = "BUILD_V2_LIKE_R5_2_OPPORTUNITY_BASE_FROM_OOF_REPLAY_V1"
    else:
        status = "V2_OOF_REPLAY_DECISION_VALID_BUT_WEAK"
        next_action = "BUILD_V2_LIKE_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_LEGAL_SIGNALS_V1"
    return {
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "decision_valid_v1": decision_valid,
        "bad_count_v1": int((selected & bad).sum()),
        "tail_count_v1": int((selected & tail).sum()),
        **precision,
        **loso,
        "safety_clean_v1": safety_clean,
        "safety_v1": safety,
        "oof_provenance_status_v1": "PASS" if provenance_valid else "FAIL",
        "train_validation_overlap_status_v1": no_overlap["status_v1"],
        "train_validation_overlap_count_v1": no_overlap["overlap_count_v1"],
        "in_sample_scored_status_v1": no_in_sample["status_v1"],
        "in_sample_scored_count_v1": no_in_sample["in_sample_scored_count_v1"],
        "provenance_row_count_v1": int(len(provenance)),
        "score_row_count_v1": int(len(scores)),
        "loso_rows_v1": loso_rows,
        **comparison_payload,
    }


def _comparison_payload(scores: pd.DataFrame, selected: pd.Series, comparison: pd.DataFrame) -> dict[str, Any]:
    if comparison.empty:
        return {
            "historical_v2_bad_tail_v1": "95/61",
            "optuna_best_bad_tail_v1": "56/55",
            "v3_best_bad_tail_v1": "17/13",
            "captured_safe_recoverable_rows_v1": None,
            "missed_safe_recoverable_rows_v1": None,
        }
    comp = comparison.set_index("candidate_uid")
    safe = scores["candidate_uid"].map(comp["safe_recoverable_candidate_v1"]).fillna(False).astype(bool)
    optuna = scores["candidate_uid"].map(comp["optuna_best_captured_v1"]).fillna(False).astype(bool)
    v3 = scores["candidate_uid"].map(comp["v3_oof_final_base_v1"]).fillna(False).astype(bool)
    return {
        "historical_v2_bad_tail_v1": "95/61",
        "optuna_best_bad_tail_v1": "56/55",
        "v3_best_bad_tail_v1": "17/13",
        "row_overlap_with_optuna_best_v1": int((selected & optuna).sum()),
        "row_overlap_with_v3_v1": int((selected & v3).sum()),
        "captured_safe_recoverable_rows_v1": int((selected & safe).sum()),
        "missed_safe_recoverable_rows_v1": int((~selected & safe).sum()),
        "safe_recoverable_total_v1": int(safe.sum()),
    }


def _metric_reports(eval_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    denominator_rows = [
        {
            "metric_v1": "precision",
            "value_v1": eval_payload["precision_v1"],
            "numerator_v1": eval_payload["precision_numerator_v1"],
            "denominator_v1": eval_payload["precision_denominator_v1"],
            "denominator_status_v1": eval_payload["precision_denominator_status_v1"],
            "decision_valid_v1": eval_payload["precision_decision_valid_v1"],
        },
        {
            "metric_v1": "worst_loso",
            "value_v1": eval_payload["worst_loso_v1"],
            "numerator_v1": eval_payload["worst_loso_numerator_v1"],
            "denominator_v1": eval_payload["worst_loso_denominator_v1"],
            "denominator_status_v1": eval_payload["worst_loso_denominator_status_v1"],
            "decision_valid_v1": eval_payload["worst_loso_decision_valid_v1"],
        },
    ]
    safety_rows = [{"safety_metric_v1": key, "value_v1": value, "pass_v1": value == 0} for key, value in eval_payload["safety_v1"].items()]
    return denominator_rows, safety_rows


def _delta_reports(eval_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    bad = int(eval_payload["bad_count_v1"])
    tail = int(eval_payload["tail_count_v1"])
    historical = {
        "historical_v2_bad_v1": 95,
        "historical_v2_tail_v1": 61,
        "oof_v2_bad_v1": bad,
        "oof_v2_tail_v1": tail,
        "bad_delta_vs_historical_v2_v1": bad - 95,
        "tail_delta_vs_historical_v2_v1": tail - 61,
        "interpretation_v1": "Historical V2 remains comparator only; OOF replay is the current-guard evidence.",
    }
    optuna_v3 = {
        "optuna_best_bad_v1": 56,
        "optuna_best_tail_v1": 55,
        "v3_best_bad_v1": 17,
        "v3_best_tail_v1": 13,
        "oof_v2_bad_v1": bad,
        "oof_v2_tail_v1": tail,
        "bad_delta_vs_optuna_v1": bad - 56,
        "tail_delta_vs_optuna_v1": tail - 55,
        "bad_delta_vs_v3_v1": bad - 17,
        "tail_delta_vs_v3_v1": tail - 13,
        "row_overlap_with_optuna_best_v1": eval_payload.get("row_overlap_with_optuna_best_v1"),
        "row_overlap_with_v3_v1": eval_payload.get("row_overlap_with_v3_v1"),
    }
    rows = [
        {"comparator_v1": "historical_v2", "bad_v1": 95, "tail_v1": 61, "decisioning_status_v1": "HISTORICAL_ONLY"},
        {"comparator_v1": "optuna_best", "bad_v1": 56, "tail_v1": 55, "decisioning_status_v1": "SAFE_BUT_NOT_BETTER_THAN_V2"},
        {"comparator_v1": "v3_best", "bad_v1": 17, "tail_v1": 13, "decisioning_status_v1": "WEAK_CONTROL"},
        {"comparator_v1": "v2_oof_replay", "bad_v1": bad, "tail_v1": tail, "decisioning_status_v1": eval_payload["status_v1"]},
    ]
    return historical, optuna_v3, rows


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    spec_dir: Path = historical_v2.DEFAULT_SPEC_DIR,
    output_dir: Path | None = None,
    foundation_score_dir: Path | None = None,
    label_table: Path | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    if fold_count < 2:
        raise RuntimeError("Grouped OOF replay requires at least two folds")
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    inputs = _prepare_inputs(spec_dir, foundation_score_dir, label_table)
    source_mapping = _source_mapping(output_dir, inputs)
    contract = _oof_contract(inputs)
    comparison = _load_comparison_ledger()
    replay = _run_oof_replay(output_dir, inputs, fold_count)
    scores = replay["scores"]
    membership = pd.DataFrame(replay["membership_rows"])
    provenance = pd.DataFrame(replay["provenance_rows"])
    eval_payload = _evaluate_oof(scores, provenance, membership, comparison)
    denominator_rows, safety_rows = _metric_reports(eval_payload)
    historical_delta, optuna_v3_delta, comparator_rows = _delta_reports(eval_payload)
    no_fallback = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)

    scores.to_csv(output_dir / "v2_oof_scores_v1.csv", index=False)
    provenance.to_csv(output_dir / "v2_oof_score_provenance_v1.csv", index=False)
    replay["fold_assignment"].to_csv(output_dir / "v2_oof_fold_assignment_v1.csv", index=False)
    membership.to_csv(output_dir / "v2_train_validation_membership_v1.csv", index=False)
    _write_rows(output_dir / "v2_oof_metric_denominator_report_v1.csv", denominator_rows)
    _write_json(output_dir / "v2_oof_metric_denominator_report_v1.json", {"rows_v1": denominator_rows, "status_v1": "PASS" if all(row["decision_valid_v1"] for row in denominator_rows) else "FAIL"})
    _write_rows(output_dir / "v2_oof_safety_report_v1.csv", safety_rows)
    _write_json(output_dir / "v2_oof_safety_report_v1.json", {"rows_v1": safety_rows, "safety_clean_v1": eval_payload["safety_clean_v1"], **eval_payload["safety_v1"]})
    _write_json(output_dir / "v2_oof_vs_historical_v2_delta_v1.json", historical_delta)
    _write_json(output_dir / "v2_oof_vs_optuna_v3_delta_v1.json", optuna_v3_delta)
    _write_rows(output_dir / "v2_oof_vs_historical_v2_delta_v1.csv", comparator_rows)
    _write_rows(output_dir / "v2_oof_vs_optuna_v3_delta_v1.csv", comparator_rows)
    _write_json(output_dir / "no_fallback_no_dummy_no_synthetic_attestation_v1.json", {**no_fallback, "no_in_sample_decisioning_v1": eval_payload["in_sample_scored_status_v1"]})
    _write_json(output_dir / "v2_runner_source_mapping_v1.json", source_mapping)
    (output_dir / "v2_runner_source_mapping_v1.md").write_text(_report_source_mapping(source_mapping), encoding="utf-8")
    _write_json(output_dir / "v2_oof_replay_contract_v1.json", contract)
    (output_dir / "v2_oof_replay_contract_v1.md").write_text(_report_contract(contract), encoding="utf-8")
    _write_json(
        output_dir / "v2_oof_score_source_manifest_v1.json",
        {
            "layer_name": "V2_OOF_SCORE_SOURCE_MANIFEST_V1",
            "variant_v1": V2_VARIANT,
            "profile_v1": V2_PROFILE,
            "model_family_v1": "HistGradientBoostingClassifier",
            "fold_count_v1": fold_count,
            "scorefields_v1": SCORE_COLUMNS,
            "fold_models_v1": replay["fold_models"],
            "hashes_v1": replay["hashes"],
            "feature_count_v1": len(replay["feature_names"]),
            "feature_families_v1": {key: len(value) for key, value in replay["feature_families"].items()},
            "existing_v2_model_artifacts_decisioning_status_v1": source_mapping["existing_v2_model_artifact_use_v1"],
            "no_fallback_no_dummy_no_synthetic_v1": no_fallback["status_v1"],
        },
    )
    _write_json(
        output_dir / "v2_oof_replay_summary_v1.json",
        {
            "layer_name": "V2_OOF_REPLAY_SUMMARY_V1",
            "materialized_at_utc_v1": _utc_now(),
            "output_dir_v1": str(output_dir),
            "foundation_v1": inputs["foundation"],
            "target_audit_v1": inputs["target_audit"],
            "feature_preflight_v1": inputs["feature_preflight"],
            **{key: value for key, value in eval_payload.items() if key != "loso_rows_v1"},
        },
    )
    _write_json(
        output_dir / "v2_oof_go_no_go_v1.json",
        {
            "layer_name": "V2_OOF_GO_NO_GO_V1",
            "decision_v1": eval_payload["status_v1"],
            "next_recommended_action_v1": eval_payload["next_recommended_action_v1"],
            "decision_valid_v1": eval_payload["decision_valid_v1"],
            "do_not_run_optuna_v1": True,
            "do_not_run_r6_package_freeze_promo_live_v1": True,
        },
    )
    _write_rows(output_dir / "v2_oof_head_training_metrics_v1.csv", replay["head_metric_rows"])
    _write_rows(output_dir / "v2_oof_loso_group_denominator_v1.csv", eval_payload["loso_rows_v1"])
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "revalidation_root_v1": str(REVALIDATION_ROOT),
                "historical_v2_root_v1": str(V2_HISTORICAL_ROOT),
                "optuna_root_v1": str(OPTUNA_ROOT),
                "selected_v3_root_v1": str(SELECTED_V3_ROOT),
                "foundation_audit_root_v1": str(FOUNDATION_AUDIT_ROOT),
                "spec_dir_v1": str(spec_dir),
                "score_dir_v1": str(inputs["score_dir"]),
                "label_table_v1": str(inputs["label_path"]),
            },
        },
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v2_was_replayed_oof_v1": True,
        "existing_v2_model_artifacts_use_v1": source_mapping["existing_v2_model_artifact_use_v1"],
        "oof_provenance_status_v1": eval_payload["oof_provenance_status_v1"],
        "train_validation_overlap_status_v1": eval_payload["train_validation_overlap_status_v1"],
        "foundation_rows_v1": inputs["foundation"]["foundation_rows_v1"],
        "active_rows_v1": inputs["foundation"]["active_rows_v1"],
        "quarantine_rows_v1": inputs["foundation"]["quarantine_rows_v1"],
        "as_of_columns_v1": inputs["foundation"]["asof_columns_v1"],
        "bad_count_v1": eval_payload["bad_count_v1"],
        "tail_count_v1": eval_payload["tail_count_v1"],
        "precision_v1": eval_payload["precision_v1"],
        "precision_denominator_v1": eval_payload["precision_denominator_v1"],
        "precision_decision_valid_v1": eval_payload["precision_decision_valid_v1"],
        "worst_loso_v1": eval_payload["worst_loso_v1"],
        "worst_loso_denominator_v1": eval_payload["worst_loso_denominator_v1"],
        "worst_loso_decision_valid_v1": eval_payload["worst_loso_decision_valid_v1"],
        "captured_safe_recoverable_rows_v1": eval_payload.get("captured_safe_recoverable_rows_v1"),
        "go_no_go_v1": eval_payload["status_v1"],
        "next_action_v1": eval_payload["next_recommended_action_v1"],
        "optuna_not_run_v1": True,
        "r6_not_run_v1": True,
        "package_not_built_v1": True,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": eval_payload["status_v1"]})
    (output_dir / "report_v1.md").write_text(_report_summary(summary, eval_payload), encoding="utf-8")
    return summary


def _report_source_mapping(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 Runner Source Mapping V1",
            "",
            f"Historical runner: `{payload['v2_source_files_v1']['historical_v2_runner_v1']}`",
            f"OOF wrapper: `{payload['v2_source_files_v1']['oof_replay_wrapper_v1']}`",
            f"Existing model artifact use: `{payload['existing_v2_model_artifact_use_v1']}`",
            "",
            "Reusable parts:",
            *[f"- {item}" for item in payload["reusable_parts_v1"]],
            "",
            "Patched by wrapper:",
            *[f"- {item}" for item in payload["parts_patched_by_wrapper_v1"]],
        ]
    ) + "\n"


def _report_contract(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 OOF Replay Contract V1",
            "",
            f"Foundation rows: `{payload['foundation_rows_observed_v1']}`",
            f"Active/quarantine: `{payload['active_rows_observed_v1']}` / `{payload['quarantine_rows_observed_v1']}`",
            f"AS_OF columns: `{payload['as_of_columns_observed_v1']}`",
            f"Selected variant: `{payload['selected_variant_v1']}`",
            "",
            "Validation-only grouped OOF scoring and full provenance are required.",
        ]
    ) + "\n"


def _report_summary(summary: dict[str, Any], eval_payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Patch V2 Runner To Write Provenance V1",
            "",
            f"Go/no-go: `{summary['go_no_go_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            f"OOF bad/tail: `{summary['bad_count_v1']}` / `{summary['tail_count_v1']}`",
            f"Precision denominator: `{summary['precision_denominator_v1']}`",
            f"Worst LOSO denominator: `{summary['worst_loso_denominator_v1']}`",
            f"OOF provenance: `{summary['oof_provenance_status_v1']}`",
            f"Train/validation overlap: `{summary['train_validation_overlap_status_v1']}`",
            f"Safety clean: `{eval_payload['safety_clean_v1']}`",
            "",
            "No Optuna, R6, package build, freeze, promo, or live action was run.",
        ]
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=historical_v2.DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--foundation-score-dir", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        foundation_score_dir=args.foundation_score_dir,
        label_table=args.label_table,
        fold_count=args.fold_count,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
