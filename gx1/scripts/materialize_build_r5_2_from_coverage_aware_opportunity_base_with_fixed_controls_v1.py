#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import run_r5_2_objective_v2_parallel_rebuild_runner_v1 as historical_v2
from gx1.scripts import run_r5_2_objective_v2_replay_with_oof_provenance_v1 as v2_replay


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
ACTION = "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1"
LAYER_NAME = ACTION
COVERAGE_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1_20260427T142902Z_LOCK"
LOW_SUPPORT_POLICY_ROOT = DEFAULT_REPORTS_ROOT / "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_V1_20260427T140733Z_LOCK"
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
DENOMINATOR_TARGET = 5
DEFAULT_FOLD_COUNT = 5
SELECTED_OPPORTUNITY_VARIANT = "COVERAGE_AWARE_RUN_ID_BALANCED"

SCOREFIELDS = [
    ("coverage_bad_target_v1", "r5_2_coverage_bad_score_v1"),
    ("coverage_tail_target_v1", "r5_2_coverage_tail_score_v1"),
    ("coverage_hard_veto_target_v1", "r5_2_coverage_hard_veto_score_v1"),
]

THRESHOLD_CANDIDATES = [
    {
        "threshold_candidate_id_v1": "SAFETY_FIRST",
        "bad_threshold_v1": 0.80,
        "tail_threshold_v1": 0.85,
        "hard_veto_max_v1": 0.30,
        "policy_v1": "HIGH_CONFIDENCE_BAD_OR_TAIL_WITH_LOW_VETO_SCORE",
    },
    {
        "threshold_candidate_id_v1": "CONSERVATIVE",
        "bad_threshold_v1": 0.65,
        "tail_threshold_v1": 0.70,
        "hard_veto_max_v1": 0.50,
        "policy_v1": "CONSERVATIVE_BAD_OR_TAIL",
    },
    {
        "threshold_candidate_id_v1": "BALANCED",
        "bad_threshold_v1": 0.50,
        "tail_threshold_v1": 0.55,
        "hard_veto_max_v1": 0.70,
        "policy_v1": "BALANCED_BAD_TAIL_RECALL",
    },
    {
        "threshold_candidate_id_v1": "TAIL_FOCUSED",
        "bad_threshold_v1": 0.55,
        "tail_threshold_v1": 0.35,
        "hard_veto_max_v1": 0.75,
        "policy_v1": "TAIL_RECALL_WITH_BAD_CONFIRMATION",
    },
    {
        "threshold_candidate_id_v1": "RECALL",
        "bad_threshold_v1": 0.35,
        "tail_threshold_v1": 0.40,
        "hard_veto_max_v1": 0.80,
        "policy_v1": "RECALL_ORIENTED_WITH_HARD_VETOES",
    },
]

FIXED_CONTROLS = [
    {"control_v1": "historical_v2", "bad_v1": 95, "tail_v1": 61, "role_v1": "BLUEPRINT_COMPARATOR_ONLY_NOT_DECISION_VALID"},
    {"control_v1": "v2_oof", "bad_v1": 69, "tail_v1": 53, "role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL_STRICT_LOSO_INVALID"},
    {"control_v1": "policy_73", "bad_v1": 73, "tail_v1": 55, "role_v1": "POLICY_ALLOWED_BASE_CONTROL"},
    {"control_v1": "optuna", "bad_v1": 56, "tail_v1": 55, "role_v1": "WEAK_SEARCH_SPACE_CONTROL"},
    {"control_v1": "v3", "bad_v1": 17, "tail_v1": 13, "role_v1": "WEAK_OOF_CONTROL"},
    {"control_v1": "coverage_aware_skeleton_proxy", "bad_v1": 188, "tail_v1": 136, "role_v1": "TRAINING_OPPORTUNITY_ONLY_NOT_DECISIONING"},
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_loso_guard_not_weakened(min_denominator: int) -> bool:
    if int(min_denominator) < DENOMINATOR_TARGET:
        raise RuntimeError("LOSO_DENOMINATOR_GUARD_WEAKENING_FORBIDDEN")
    return True


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures, "decision_valid_v1": not failures}


def validate_no_forbidden_actions(*, optuna: bool, r6: bool, package: bool, freeze: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_forbidden_features(feature_names: Sequence[str]) -> dict[str, Any]:
    id_leakage_names = {"candidate_uid", "trade_uid", "trade_id", "run_id", "decision_timestamp"}
    rows = []
    for feature in feature_names:
        lower = str(feature).lower()
        matches = [pattern for pattern in historical_v2.FORBIDDEN_FEATURE_PATTERNS if pattern in lower]
        if lower in id_leakage_names or lower.endswith("_uid") or lower.endswith("_id"):
            matches.append("id_leakage_key")
        rows.append({"feature_v1": str(feature), "matches_v1": matches})
    forbidden = [row for row in rows if row["matches_v1"]]
    return {"status_v1": "PASS" if not forbidden else "FAIL", "forbidden_features_v1": forbidden, "decision_valid_v1": not forbidden}


def validate_no_hindsight_features(feature_names: Sequence[str]) -> dict[str, Any]:
    patterns = ["hindsight", "future_", "post_decision"]
    forbidden = [str(feature) for feature in feature_names if any(pattern in str(feature).lower() for pattern in patterns)]
    return {"status_v1": "PASS" if not forbidden else "FAIL", "hindsight_features_v1": forbidden, "decision_valid_v1": not forbidden}


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


def validate_oof_provenance_complete(scores: pd.DataFrame, provenance: pd.DataFrame, scorefields: Sequence[str] = tuple(col for _, col in SCOREFIELDS)) -> dict[str, Any]:
    if scores.empty or provenance.empty:
        return {"status_v1": "FAIL", "missing_provenance_rows_v1": -1, "decision_valid_v1": False}
    expected = {
        (str(row["candidate_uid_v1"]), str(scorefield))
        for _, row in scores.iterrows()
        for scorefield in scorefields
    }
    observed = {
        (str(row["candidate_uid_v1"]), str(row["scorefield_v1"]))
        for _, row in provenance.iterrows()
    }
    missing = sorted(expected - observed)
    invalid = int(provenance["provenance_valid_v1"].fillna(False).astype(bool).eq(False).sum()) if "provenance_valid_v1" in provenance.columns else len(provenance)
    return {
        "status_v1": "PASS" if not missing and invalid == 0 else "FAIL",
        "missing_provenance_rows_v1": int(len(missing)),
        "invalid_provenance_rows_v1": invalid,
        "decision_valid_v1": not missing and invalid == 0,
    }


def historical_v2_role() -> str:
    return "BLUEPRINT_COMPARATOR_ONLY_NOT_DECISION_VALID"


def weak_control_can_be_baseline(control_name: str) -> bool:
    return str(control_name).lower() not in {"optuna", "v3"}


def candidate_final_promotion_allowed(*, structural_low_support_selected: bool, strict_loso_decision_valid: bool, explicit_exception_gate: bool) -> bool:
    return bool(strict_loso_decision_valid and ((not structural_low_support_selected) or explicit_exception_gate))


def _positive_role(role: str) -> bool:
    return role in {
        "CORE_V2_OOF_POSITIVE",
        "CORE_V2_OOF_TAIL_POSITIVE",
        "COVERAGE_EXPANSION_STRONG_BAD",
        "COVERAGE_EXPANSION_TAIL",
        "COVERAGE_EXPANSION_RUN_ID_SUPPORT",
        "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE",
    }


def row_is_hard_veto(row: pd.Series | dict[str, Any]) -> bool:
    return bool(
        str(row.get("active_quarantine_v1", "")).upper() != "ACTIVE_CANDIDATE"
        or _as_bool(row.get("protected_winner_status_v1"))
        or _as_bool(row.get("runner_protect_status_v1"))
        or _as_bool(row.get("ambiguous_high_mfe_status_v1"))
        or _as_bool(row.get("fifty_plus_mfe_risk_v1"))
        or _as_bool(row.get("hundred_plus_mfe_risk_v1"))
        or _as_bool(row.get("two_hundred_plus_mfe_risk_v1"))
    )


def positive_row_has_evidence(row: pd.Series | dict[str, Any]) -> bool:
    if _as_bool(row.get("v2_oof_captured_v1")) or _as_bool(row.get("historical_v2_captured_v1")):
        return True
    for column in [
        "r5_bad_score_signal_bucket_v1",
        "r5_1_bad_score_signal_bucket_v1",
        "r5_tail_score_signal_bucket_v1",
        "v2_like_bad_tail_signal_bucket_v1",
        "v3_oof_signal_bucket_v1",
    ]:
        if str(row.get(column, "NONE")) in {"STRONG", "SUPPORT"}:
            return True
    return int(row.get("existing_legal_signal_evidence_count_v1", 0) or 0) > 0


def row_can_be_training_positive(row: pd.Series | dict[str, Any]) -> bool:
    return bool(_positive_role(str(row.get("opportunity_role_v1", ""))) and positive_row_has_evidence(row) and not row_is_hard_veto(row))


def classify_target_class(row: pd.Series | dict[str, Any]) -> str:
    role = str(row.get("opportunity_role_v1", "UNKNOWN_REQUIRES_ARTIFACT"))
    if role == "QUARANTINE_EXCLUDE":
        return "EXCLUDE_QUARANTINE"
    if role == "HARD_NEGATIVE_PROTECTED_WINNER":
        return "HARD_NEGATIVE_PROTECTED_WINNER"
    if role == "HARD_NEGATIVE_RUNNER_PROTECT":
        return "HARD_NEGATIVE_RUNNER_PROTECT"
    if role == "HARD_NEGATIVE_HIGH_MFE_UNSAFE":
        return "HARD_NEGATIVE_HIGH_MFE_UNSAFE"
    if role == "AMBIGUOUS_MONITOR_ONLY":
        return "MONITOR_ONLY_AMBIGUOUS"
    if role == "CORE_V2_OOF_TAIL_POSITIVE" or role == "COVERAGE_EXPANSION_TAIL":
        return "POSITIVE_TAIL"
    if role == "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE":
        return "POSITIVE_LOW_SUPPORT_TRAINING_ONLY"
    if role in {"CORE_V2_OOF_POSITIVE", "COVERAGE_EXPANSION_STRONG_BAD", "COVERAGE_EXPANSION_RUN_ID_SUPPORT"}:
        return "POSITIVE_STRONG_BAD"
    return "EXCLUDE_UNKNOWN"


def training_weight_value(tier: str) -> float:
    return {
        "CORE_HIGH_WEIGHT": 3.0,
        "TAIL_HIGH_WEIGHT": 3.0,
        "COVERAGE_MEDIUM_WEIGHT": 1.5,
        "LOW_SUPPORT_LOW_WEIGHT": 0.75,
        "HARD_NEGATIVE_HIGH_WEIGHT": 5.0,
        "MONITOR_ZERO_WEIGHT": 0.0,
        "EXCLUDE_ZERO_WEIGHT": 0.0,
        "UNKNOWN_ZERO_WEIGHT": 0.0,
    }.get(str(tier), 0.0)


def validate_training_target_table(target: pd.DataFrame) -> dict[str, Any]:
    quarantine_positive = int((_bool(target, "exclude_v1") & (_bool(target, "bad_target_v1") | _bool(target, "tail_target_v1"))).sum())
    protected_positive = int((_bool(target, "protected_winner_status_v1") & (_bool(target, "bad_target_v1") | _bool(target, "tail_target_v1"))).sum())
    runner_positive = int((_bool(target, "runner_protect_status_v1") & (_bool(target, "bad_target_v1") | _bool(target, "tail_target_v1"))).sum())
    ambiguous_positive = int((_bool(target, "ambiguous_high_mfe_status_v1") & (_bool(target, "bad_target_v1") | _bool(target, "tail_target_v1"))).sum())
    monitor_positive = int((_bool(target, "monitor_only_v1") & (_bool(target, "bad_target_v1") | _bool(target, "tail_target_v1"))).sum())
    failures = []
    if quarantine_positive:
        failures.append("QUARANTINE_POSITIVE_FORBIDDEN")
    if protected_positive:
        failures.append("PROTECTED_WINNER_POSITIVE_FORBIDDEN")
    if runner_positive:
        failures.append("RUNNER_PROTECT_POSITIVE_FORBIDDEN")
    if ambiguous_positive:
        failures.append("AMBIGUOUS_HIGH_MFE_POSITIVE_FORBIDDEN")
    if monitor_positive:
        failures.append("MONITOR_ONLY_POSITIVE_FORBIDDEN")
    return {
        "status_v1": "PASS" if not failures else "FAIL",
        "failures_v1": failures,
        "quarantine_positive_count_v1": quarantine_positive,
        "protected_positive_count_v1": protected_positive,
        "runner_positive_count_v1": runner_positive,
        "ambiguous_positive_count_v1": ambiguous_positive,
        "monitor_positive_count_v1": monitor_positive,
    }


def threshold_candidate_passes_safety(row: dict[str, Any]) -> bool:
    safety_keys = [
        "fifty_plus_mfe_overlap_v1",
        "hundred_plus_mfe_overlap_v1",
        "two_hundred_plus_mfe_overlap_v1",
        "strongest_winner_overlap_v1",
        "protected_winner_selected_v1",
        "runner_protect_leakage_v1",
        "ambiguous_high_mfe_leakage_v1",
        "quarantine_selected_v1",
    ]
    return all(int(row.get(key, 0) or 0) == 0 for key in safety_keys)


def _metric_ratio(name: str, numerator: int, denominator: int, min_denominator: int = DENOMINATOR_TARGET) -> dict[str, Any]:
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


def _load_coverage_inputs(coverage_root: Path, policy_root: Path) -> dict[str, Any]:
    return {
        "coverage_rows": pd.read_csv(coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv"),
        "coverage_variants": pd.read_csv(coverage_root / "coverage_aware_r5_2_base_variants_v1.csv"),
        "coverage_recommendation": _read_json(coverage_root / "coverage_aware_r5_2_opportunity_base_recommendation_v1.json"),
        "coverage_contract": _read_json(coverage_root / "next_r5_2_rebuild_from_coverage_aware_base_contract_v1.json"),
        "low_support_registry": pd.read_csv(policy_root / "structural_low_support_run_id_registry_v1.csv"),
        "low_support_policy": _read_json(policy_root / "run_id_low_support_policy_v1.json"),
    }


def _target_class_flags(target_class: str) -> dict[str, bool]:
    return {
        "bad_target_v1": target_class in {"POSITIVE_STRONG_BAD", "POSITIVE_TAIL", "POSITIVE_LOW_SUPPORT_TRAINING_ONLY"},
        "tail_target_v1": target_class == "POSITIVE_TAIL",
        "hard_negative_v1": target_class.startswith("HARD_NEGATIVE"),
        "monitor_only_v1": target_class == "MONITOR_ONLY_AMBIGUOUS",
        "exclude_v1": target_class.startswith("EXCLUDE"),
    }


def _build_training_target_table(coverage_rows: pd.DataFrame, training_frame: pd.DataFrame) -> pd.DataFrame:
    rows = coverage_rows.copy()
    rows["candidate_uid"] = rows["candidate_uid_v1"].astype(str)
    frame = training_frame.copy()
    frame["candidate_uid"] = frame["candidate_uid"].astype(str)
    merged = frame[
        [
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "run_id",
            "label_should_not_take_v1",
            "tail_10_50_mfe_v1",
        ]
    ].merge(rows, on="candidate_uid", how="left", validate="one_to_one")
    if merged["opportunity_role_v1"].isna().any():
        raise RuntimeError("Coverage-aware opportunity rows do not align one-to-one with foundation training frame")
    merged["target_class_v1"] = merged.apply(classify_target_class, axis=1)
    for key in ["bad_target_v1", "tail_target_v1", "hard_negative_v1", "monitor_only_v1", "exclude_v1"]:
        merged[key] = merged["target_class_v1"].map(lambda target_class: _target_class_flags(str(target_class))[key])
    merged["coverage_bad_target_v1"] = merged["bad_target_v1"].astype(bool)
    merged["coverage_tail_target_v1"] = merged["tail_target_v1"].astype(bool)
    merged["coverage_hard_veto_target_v1"] = merged["hard_negative_v1"].astype(bool)
    merged["training_weight_value_v1"] = merged["training_weight_tier_v1"].map(training_weight_value).astype(float)
    merged["source_evidence_v1"] = merged.apply(_source_evidence, axis=1)
    merged["reason_v1"] = merged["coverage_reason_v1"].astype(str)
    validation = validate_training_target_table(merged)
    if validation["status_v1"] != "PASS":
        raise RuntimeError(f"Invalid R5.2 training target table: {validation}")
    return merged


def _source_evidence(row: pd.Series) -> str:
    evidence = []
    if _as_bool(row.get("v2_oof_captured_v1")):
        evidence.append("V2_OOF")
    if _as_bool(row.get("historical_v2_captured_v1")):
        evidence.append("HISTORICAL_V2_BLUEPRINT")
    for column, name in [
        ("r5_bad_score_signal_bucket_v1", "R5_BAD_SCORE"),
        ("r5_1_bad_score_signal_bucket_v1", "R5_1_BAD_SCORE"),
        ("r5_tail_score_signal_bucket_v1", "R5_TAIL_SCORE"),
        ("v2_like_bad_tail_signal_bucket_v1", "V2_LIKE_BAD_TAIL"),
        ("v3_oof_signal_bucket_v1", "V3_OOF"),
    ]:
        if str(row.get(column, "NONE")) in {"STRONG", "SUPPORT"}:
            evidence.append(f"{name}:{row.get(column)}")
    return "|".join(evidence) if evidence else "NO_POSITIVE_SIGNAL_EVIDENCE"


def _target_output_rows(target: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "trade_id_v1",
        "decision_timestamp_v1",
        "run_id_v1",
        "active_quarantine_v1",
        "opportunity_role_v1",
        "training_weight_tier_v1",
        "target_class_v1",
        "bad_target_v1",
        "tail_target_v1",
        "hard_negative_v1",
        "monitor_only_v1",
        "exclude_v1",
        "run_id_policy_class_v1",
        "structural_low_support_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "existing_legal_signal_evidence_count_v1",
        "source_evidence_v1",
        "reason_v1",
    ]
    return target[[field for field in fields if field in target.columns]].to_dict("records")


def _fold_assignment(frame: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    assignment = frame[["candidate_uid", "trade_uid", "decision_timestamp", "trade_id", "run_id"]].copy()
    assignment["fold_id_v1"] = v2_replay._balanced_group_folds(frame, "run_id", fold_count)
    assignment["group_key_v1"] = assignment["run_id"].astype(str)
    assignment["split_policy_v1"] = "DETERMINISTIC_BALANCED_GROUPED_OOF_BY_RUN_ID"
    assignment = assignment.rename(
        columns={
            "candidate_uid": "candidate_uid_v1",
            "trade_uid": "trade_uid_v1",
            "decision_timestamp": "decision_timestamp_v1",
            "trade_id": "trade_id_v1",
            "run_id": "run_id_v1",
        }
    )
    return assignment


def _fit_oof_head(
    *,
    x: pd.DataFrame,
    frame: pd.DataFrame,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    label_col: str,
    weight_col: str,
    output_col: str,
    model_source_id: str,
    seed: int,
    model_dir: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    y = frame[label_col].astype(bool).astype(int)
    y_train = y.loc[train_mask]
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError(f"DEGRADED_CONSTANT_MODEL_FORBIDDEN_FOR_{output_col}")
    model = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    sample_weight = frame.loc[train_mask, weight_col].astype(float).to_numpy()
    model.fit(x.loc[train_mask], y_train, sample_weight=sample_weight)
    pred = pd.Series(np.nan, index=frame.index, dtype="float64")
    pred.loc[validation_mask] = model.predict_proba(x.loc[validation_mask])[:, 1]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{output_col}.joblib")
    yy = y.loc[validation_mask].to_numpy(dtype=int)
    pp = pred.loc[validation_mask].to_numpy(dtype=float)
    auc = float(roc_auc_score(yy, pp)) if len(set(yy.tolist())) >= 2 else None
    metrics = {
        "model_source_identifier_v1": model_source_id,
        "head_v1": label_col,
        "scorefield_v1": output_col,
        "seed_v1": seed,
        "model_family_v1": "HistGradientBoostingClassifier",
        "train_rows_v1": int(train_mask.sum()),
        "validation_rows_v1": int(validation_mask.sum()),
        "positive_train_rows_v1": int(y_train.sum()),
        "positive_validation_rows_v1": int(y.loc[validation_mask].sum()),
        "validation_roc_auc_v1": auc,
        "constant_model_v1": False,
    }
    _write_json(
        model_dir / f"{output_col}.metadata.json",
        {
            "model_source_identifier_v1": model_source_id,
            "label_col_v1": label_col,
            "output_col_v1": output_col,
            "seed_v1": seed,
            "model_family_v1": "HistGradientBoostingClassifier",
            "scope_v1": "GROUPED_OOF_VALIDATION_ONLY",
        },
    )
    return pred.rename(output_col), metrics


def _run_grouped_oof(
    output_dir: Path,
    *,
    inputs: dict[str, Any],
    target: pd.DataFrame,
    fold_count: int,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    x = inputs["x"]
    feature_names = inputs["feature_names"]
    training_frame = inputs["training_frame"].copy()
    frame = training_frame.merge(
        target[
            [
                "candidate_uid",
                "target_class_v1",
                "coverage_bad_target_v1",
                "coverage_tail_target_v1",
                "coverage_hard_veto_target_v1",
                "training_weight_value_v1",
                "opportunity_role_v1",
                "training_weight_tier_v1",
                "evaluation_role_v1",
                "run_id_policy_class_v1",
                "structural_low_support_v1",
                "zero_denominator_group_v1",
                "training_opportunity_allowed_v1",
                "final_promotion_evidence_allowed_v1",
                "source_evidence_v1",
            ]
        ],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    assignment = _fold_assignment(frame, fold_count)
    fold_ids = assignment["fold_id_v1"].astype(int)
    source_hash = _file_hash(Path(__file__).resolve())
    v2_source_hash = _file_hash(Path(historical_v2.__file__).resolve())
    config_hash = _hash_json(config_payload)
    feature_matrix_hash = _hash_frame(x)
    target_columns = [
        "candidate_uid",
        "target_class_v1",
        "coverage_bad_target_v1",
        "coverage_tail_target_v1",
        "coverage_hard_veto_target_v1",
        "training_weight_value_v1",
        "opportunity_role_v1",
        "training_weight_tier_v1",
    ]
    label_table_hash = _hash_frame(frame, target_columns)
    feature_row_hashes = _row_hashes(x, list(x.columns))
    label_row_hashes = _row_hashes(frame, target_columns)

    prediction = frame[["candidate_uid", "trade_uid", "decision_timestamp", "trade_id", "run_id"]].copy()
    for _, scorefield in SCOREFIELDS:
        prediction[scorefield] = np.nan

    membership_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    head_metric_rows: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    model_root = output_dir / "r5_2_oof_models_history_only"
    for fold_id in sorted(fold_ids.unique()):
        validation_mask = fold_ids.eq(fold_id)
        train_mask = ~validation_mask
        train_uids = frame.loc[train_mask, "candidate_uid"].astype(str).tolist()
        validation_uids = frame.loc[validation_mask, "candidate_uid"].astype(str).tolist()
        train_hash = _hash_list(train_uids)
        validation_hash = _hash_list(validation_uids)
        fold_label = f"fold_{int(fold_id):02d}"
        for idx, row in frame.iterrows():
            is_train = bool(train_mask.loc[idx])
            is_validation = bool(validation_mask.loc[idx])
            membership_rows.append(
                {
                    "candidate_uid_v1": row["candidate_uid"],
                    "trade_uid_v1": row["trade_uid"],
                    "decision_timestamp_v1": row["decision_timestamp"],
                    "trade_id_v1": row["trade_id"],
                    "run_id_v1": row["run_id"],
                    "fold_id_v1": fold_label,
                    "is_train_v1": is_train,
                    "is_validation_v1": is_validation,
                    "train_membership_hash_v1": train_hash,
                    "validation_membership_hash_v1": validation_hash,
                    "train_validation_overlap_v1": bool(is_train and is_validation),
                }
            )
        for head_idx, (label_col, scorefield) in enumerate(SCOREFIELDS):
            seed = 20260427 + int(fold_id) * 100 + head_idx
            model_source_id = f"{ACTION}:{fold_label}:{scorefield}:seed={seed}"
            pred, metrics = _fit_oof_head(
                x=x,
                frame=frame,
                train_mask=train_mask,
                validation_mask=validation_mask,
                label_col=label_col,
                weight_col="training_weight_value_v1",
                output_col=scorefield,
                model_source_id=model_source_id,
                seed=seed,
                model_dir=model_root / fold_label,
            )
            prediction.loc[validation_mask, scorefield] = pred.loc[validation_mask]
            metrics.update(
                {
                    "fold_id_v1": fold_label,
                    "train_membership_hash_v1": train_hash,
                    "validation_membership_hash_v1": validation_hash,
                }
            )
            head_metric_rows.append(metrics)
            fold_models.append(
                {
                    "fold_id_v1": fold_label,
                    "scorefield_v1": scorefield,
                    "label_col_v1": label_col,
                    "model_source_identifier_v1": model_source_id,
                    "model_artifact_path_v1": str(model_root / fold_label / f"{scorefield}.joblib"),
                    "metadata_path_v1": str(model_root / fold_label / f"{scorefield}.metadata.json"),
                    "source_hash_v1": source_hash,
                    "v2_training_utility_source_hash_v1": v2_source_hash,
                    "config_hash_v1": config_hash,
                    "seed_v1": seed,
                    "decisioning_scope_v1": "OOF_VALIDATION_ONLY",
                }
            )
            for idx in frame.index[validation_mask]:
                row = frame.loc[idx]
                provenance_rows.append(
                    {
                        "candidate_uid_v1": row["candidate_uid"],
                        "trade_uid_v1": row["trade_uid"],
                        "decision_timestamp_v1": row["decision_timestamp"],
                        "trade_id_v1": row["trade_id"],
                        "run_id_v1": row["run_id"],
                        "fold_id_v1": fold_label,
                        "scorefield_v1": scorefield,
                        "head_v1": label_col,
                        "variant_v1": SELECTED_OPPORTUNITY_VARIANT,
                        "model_source_identifier_v1": model_source_id,
                        "train_membership_hash_v1": train_hash,
                        "validation_membership_hash_v1": validation_hash,
                        "was_row_in_train_for_scoring_model_v1": False,
                        "feature_matrix_hash_v1": feature_matrix_hash,
                        "feature_row_hash_v1": feature_row_hashes.loc[idx],
                        "label_table_hash_v1": label_table_hash,
                        "label_row_hash_v1": label_row_hashes.loc[idx],
                        "config_hash_v1": config_hash,
                        "source_hash_v1": source_hash,
                        "v2_training_utility_source_hash_v1": v2_source_hash,
                        "seed_v1": seed,
                        "score_value_v1": pred.loc[idx],
                        "decision_valid_v1": True,
                        "provenance_valid_v1": True,
                        "oof_status_v1": "OOF_VALIDATION_SCORE",
                    }
                )
    if prediction[[scorefield for _, scorefield in SCOREFIELDS]].isna().any().any():
        missing = prediction[[scorefield for _, scorefield in SCOREFIELDS]].isna().sum().to_dict()
        raise RuntimeError(f"OOF prediction matrix has missing scores: {missing}")
    fold_assignment = assignment.copy()
    fold_assignment["fold_id_v1"] = fold_assignment["fold_id_v1"].map(lambda value: f"fold_{int(value):02d}")
    scores = frame[
        [
            "candidate_uid",
            "trade_uid",
            "decision_timestamp",
            "trade_id",
            "run_id",
            "label_should_not_take_v1",
            "tail_10_50_mfe_v1",
            "opportunity_role_v1",
            "training_weight_tier_v1",
            "evaluation_role_v1",
            "run_id_policy_class_v1",
            "structural_low_support_v1",
            "zero_denominator_group_v1",
            "training_opportunity_allowed_v1",
            "final_promotion_evidence_allowed_v1",
        ]
    ].copy()
    scores = scores.rename(
        columns={
            "candidate_uid": "candidate_uid_v1",
            "trade_uid": "trade_uid_v1",
            "decision_timestamp": "decision_timestamp_v1",
            "trade_id": "trade_id_v1",
            "run_id": "run_id_v1",
        }
    )
    coverage_cols = [
        "active_quarantine_v1",
        "bad_label_v1",
        "tail_label_v1",
        "safe_recoverable_v1",
        "v2_oof_captured_v1",
        "historical_v2_captured_v1",
        "optuna_captured_v1",
        "v3_captured_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "existing_legal_signal_evidence_count_v1",
        "source_evidence_v1",
    ]
    scores = scores.merge(target[["candidate_uid_v1", *coverage_cols]], on="candidate_uid_v1", how="left", validate="one_to_one")
    scores["fold_id_v1"] = fold_assignment["fold_id_v1"].values
    for _, scorefield in SCOREFIELDS:
        scores[scorefield] = prediction[scorefield].values
    scores["was_row_in_train_for_scoring_model_v1"] = False
    scores["decision_valid_score_v1"] = True
    return {
        "scores": scores,
        "fold_assignment": fold_assignment,
        "membership": pd.DataFrame(membership_rows),
        "provenance": pd.DataFrame(provenance_rows),
        "head_metrics": pd.DataFrame(head_metric_rows),
        "fold_models": fold_models,
        "hashes": {
            "feature_matrix_hash_v1": feature_matrix_hash,
            "label_table_hash_v1": label_table_hash,
            "config_hash_v1": config_hash,
            "source_hash_v1": source_hash,
            "v2_training_utility_source_hash_v1": v2_source_hash,
        },
        "feature_names": feature_names,
        "feature_families": inputs["feature_families"],
        "feature_preflight": inputs["feature_preflight"],
        "target_validation": validate_training_target_table(target),
    }


def _eligible_for_selection(scores: pd.DataFrame) -> pd.Series:
    evidence = scores.apply(positive_row_has_evidence, axis=1)
    hard_veto = (
        scores["active_quarantine_v1"].astype(str).str.upper().ne("ACTIVE_CANDIDATE")
        | _bool(scores, "protected_winner_status_v1")
        | _bool(scores, "runner_protect_status_v1")
        | _bool(scores, "ambiguous_high_mfe_status_v1")
        | _bool(scores, "fifty_plus_mfe_risk_v1")
        | _bool(scores, "hundred_plus_mfe_risk_v1")
        | _bool(scores, "two_hundred_plus_mfe_risk_v1")
    )
    return _bool(scores, "training_opportunity_allowed_v1") & evidence & ~hard_veto


def _safety_counts(scores: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    return {
        "fifty_plus_mfe_overlap_v1": int((selected & _bool(scores, "fifty_plus_mfe_risk_v1")).sum()),
        "hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "hundred_plus_mfe_risk_v1")).sum()),
        "two_hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "two_hundred_plus_mfe_risk_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(scores, "protected_winner_status_v1")).sum()),
        "protected_winner_selected_v1": int((selected & _bool(scores, "protected_winner_status_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & _bool(scores, "runner_protect_status_v1")).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(scores, "ambiguous_high_mfe_status_v1")).sum()),
        "quarantine_selected_v1": int((selected & scores["active_quarantine_v1"].astype(str).str.upper().ne("ACTIVE_CANDIDATE")).sum()),
    }


def _loso_rows(scores: pd.DataFrame, selected: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    bad = _bool(scores, "bad_label_v1")
    groups = scores["run_id_v1"].astype(str)
    for group, part in pd.DataFrame({"selected": selected.astype(bool), "bad": bad, "group": groups}).groupby("group"):
        denominator = int(part["selected"].sum())
        numerator = int((part["selected"] & part["bad"]).sum())
        precision = numerator / denominator if denominator else np.nan
        rows.append(
            {
                "run_id_v1": str(group),
                "selected_denominator_v1": denominator,
                "selected_bad_numerator_v1": numerator,
                "group_precision_v1": precision,
                "denominator_status_v1": "OK"
                if denominator >= DENOMINATOR_TARGET
                else ("EMPTY_SELECTED_GROUP" if denominator == 0 else "TOO_SMALL_DENOMINATOR"),
            }
        )
    non_empty = [row for row in rows if int(row["selected_denominator_v1"]) > 0]
    worst = min(non_empty, key=lambda row: float(row["group_precision_v1"])) if non_empty else {
        "run_id_v1": "EMPTY_SELECTED_GROUP_SET",
        "selected_denominator_v1": 0,
        "selected_bad_numerator_v1": 0,
        "group_precision_v1": np.nan,
    }
    evaluable = [row for row in rows if int(row["selected_denominator_v1"]) >= DENOMINATOR_TARGET]
    evaluable_worst = min(evaluable, key=lambda row: float(row["group_precision_v1"])) if evaluable else None
    for row in rows:
        row["is_worst_loso_group_v1"] = row["run_id_v1"] == worst["run_id_v1"]
    summary = {
        "strict_all_run_id_worst_loso_v1": worst["group_precision_v1"],
        "strict_all_run_id_worst_loso_group_v1": worst["run_id_v1"],
        "strict_all_run_id_worst_loso_numerator_v1": int(worst["selected_bad_numerator_v1"]),
        "strict_all_run_id_worst_loso_denominator_v1": int(worst["selected_denominator_v1"]),
        "strict_all_run_id_worst_loso_denominator_status_v1": "OK"
        if int(worst["selected_denominator_v1"]) >= DENOMINATOR_TARGET
        else "TOO_SMALL_DENOMINATOR",
        "strict_all_run_id_decision_valid_v1": int(worst["selected_denominator_v1"]) >= DENOMINATOR_TARGET,
        "selected_low_support_group_count_v1": int(sum(0 < int(row["selected_denominator_v1"]) < DENOMINATOR_TARGET for row in rows)),
        "zero_selected_group_count_v1": int(sum(int(row["selected_denominator_v1"]) == 0 for row in rows)),
        "evaluable_group_count_v1": int(len(evaluable)),
        "evaluable_groups_loso_v1": None if evaluable_worst is None else evaluable_worst["group_precision_v1"],
        "evaluable_groups_denominator_min_v1": 0 if not evaluable else min(int(row["selected_denominator_v1"]) for row in evaluable),
        "evaluable_groups_decision_valid_v1": bool(evaluable),
    }
    return rows, summary


def _evaluate_selection(scores: pd.DataFrame, selected: pd.Series, candidate_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = selected.reindex(scores.index).fillna(False).astype(bool)
    bad = _bool(scores, "bad_label_v1")
    tail = _bool(scores, "tail_label_v1")
    precision = _metric_ratio("precision", int((selected & bad).sum()), int(selected.sum()))
    loso_rows, loso = _loso_rows(scores, selected)
    safety = _safety_counts(scores, selected)
    safety_clean = threshold_candidate_passes_safety(safety)
    structural_groups = int((selected & _bool(scores, "structural_low_support_v1")).groupby(scores["run_id_v1"].astype(str)).any().sum())
    final_allowed = candidate_final_promotion_allowed(
        structural_low_support_selected=structural_groups > 0,
        strict_loso_decision_valid=bool(loso["strict_all_run_id_decision_valid_v1"]),
        explicit_exception_gate=False,
    )
    payload = {
        "threshold_candidate_id_v1": candidate_id,
        "selected_rows_v1": int(selected.sum()),
        "bad_count_v1": int((selected & bad).sum()),
        "tail_count_v1": int((selected & tail).sum()),
        **precision,
        **loso,
        "structural_low_support_selected_group_count_v1": structural_groups,
        "training_opportunity_allowed_v1": True,
        "final_promotion_allowed_v1": final_allowed,
        **safety,
        "safety_clean_v1": safety_clean,
        "recommendation_status_v1": _threshold_status(safety_clean, precision, loso, structural_groups),
    }
    return payload, loso_rows


def _threshold_status(safety_clean: bool, precision: dict[str, Any], loso: dict[str, Any], structural_groups: int) -> str:
    if not safety_clean:
        return "FAIL_TRUE_SAFETY"
    if not precision["precision_decision_valid_v1"]:
        return "FAIL_PRECISION_DENOMINATOR"
    if not loso["strict_all_run_id_decision_valid_v1"]:
        return "TRAINING_ALLOWED_STRICT_LOSO_LOW_SUPPORT_VISIBLE"
    if structural_groups:
        return "TRAINING_ALLOWED_FINAL_PROMOTION_BLOCKED_STRUCTURAL_LOW_SUPPORT"
    return "PASS_FOR_CANDIDATE_EVAL_NOT_PROMOTION"


def _threshold_grid(scores: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.Series], list[dict[str, Any]]]:
    eligible = _eligible_for_selection(scores)
    rows = []
    selections: dict[str, pd.Series] = {}
    loso_detail: list[dict[str, Any]] = []
    for config in THRESHOLD_CANDIDATES:
        selected = (
            eligible
            & (
                scores["r5_2_coverage_bad_score_v1"].ge(float(config["bad_threshold_v1"]))
                | scores["r5_2_coverage_tail_score_v1"].ge(float(config["tail_threshold_v1"]))
            )
            & scores["r5_2_coverage_hard_veto_score_v1"].le(float(config["hard_veto_max_v1"]))
        )
        candidate_id = str(config["threshold_candidate_id_v1"])
        metrics, loso_rows = _evaluate_selection(scores, selected, candidate_id)
        rows.append({**config, **metrics})
        selections[candidate_id] = selected
        for row in loso_rows:
            loso_detail.append({"threshold_candidate_id_v1": candidate_id, **row})
    return rows, selections, loso_detail


def _select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in rows if threshold_candidate_passes_safety(row) and bool(row.get("precision_decision_valid_v1"))]
    if not safe:
        return rows[0]
    return sorted(
        safe,
        key=lambda row: (
            int(row["bad_count_v1"]),
            int(row["tail_count_v1"]),
            float(row.get("precision_v1") or 0.0),
            int(row.get("selected_rows_v1") or 0),
        ),
        reverse=True,
    )[0]


def _best_candidate_status(best: dict[str, Any], provenance_ok: bool, no_in_sample_ok: bool, no_overlap_ok: bool) -> tuple[str, str]:
    if not provenance_ok or not no_in_sample_ok or not no_overlap_ok:
        return "R5_2_CANDIDATE_FAILS_PROVENANCE_OR_IN_SAMPLE_GUARD", "REPAIR_R5_2_OOF_PROVENANCE_OR_SPLIT_V1"
    if not threshold_candidate_passes_safety(best):
        return "R5_2_CANDIDATE_FAILS_TRUE_SAFETY", "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    bad = int(best["bad_count_v1"])
    tail = int(best["tail_count_v1"])
    if bad >= 95 and tail >= 61:
        return "R5_2_CANDIDATE_APPROACHES_OR_BEATS_HISTORICAL_V2_BUT_FINAL_PROMOTION_BLOCKED", "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1"
    if bad > 69 or tail > 53:
        return "R5_2_CANDIDATE_STRONGER_THAN_V2_OOF_BUT_FINAL_PROMOTION_BLOCKED", "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1"
    if bad >= 56 and tail >= 13:
        return "R5_2_CANDIDATE_SAFE_BUT_NOT_BETTER_THAN_V2_OOF", "REVISIT_OPPORTUNITY_BASE_OR_SIGNAL_FAMILY_AUDIT_V1"
    return "R5_2_CANDIDATE_TOO_WEAK_OR_UNSTABLE", "DEEPEN_EXISTING_LEGAL_SIGNAL_FAMILY_AUDIT_V1"


def _fixed_control_comparison(best: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for control in FIXED_CONTROLS:
        out.append(
            {
                **control,
                "candidate_bad_v1": best["bad_count_v1"],
                "candidate_tail_v1": best["tail_count_v1"],
                "bad_delta_v1": int(best["bad_count_v1"]) - int(control["bad_v1"]),
                "tail_delta_v1": int(best["tail_count_v1"]) - int(control["tail_v1"]),
            }
        )
    return out


def _rebuild_contract(inputs: dict[str, Any], coverage_root: Path, policy_root: Path) -> dict[str, Any]:
    foundation = inputs["foundation"]
    return {
        "contract": "R5_2_FROM_COVERAGE_AWARE_BASE_REBUILD_CONTRACT_V1",
        "input_opportunity_base_v1": SELECTED_OPPORTUNITY_VARIANT,
        "coverage_aware_base_root_v1": str(coverage_root),
        "low_support_policy_root_v1": str(policy_root),
        "foundation_rows_required_v1": 1914,
        "foundation_rows_observed_v1": foundation["foundation_rows_v1"],
        "active_rows_required_v1": 1852,
        "active_rows_observed_v1": foundation["active_rows_v1"],
        "quarantine_rows_required_v1": 62,
        "quarantine_rows_observed_v1": foundation["quarantine_rows_v1"],
        "as_of_columns_required_v1": 109,
        "as_of_columns_observed_v1": foundation["asof_columns_v1"],
        "grouped_oof_execution_required_v1": True,
        "validation_only_scoring_required_v1": True,
        "train_validation_membership_required_v1": True,
        "score_provenance_required_v1": True,
        "score_source_manifest_required_v1": True,
        "feature_label_config_hash_required_v1": True,
        "no_in_sample_decisioning_required_v1": True,
        "no_dummy_synthetic_fallback_required_v1": True,
        "hard_negatives_veto_rows_respected_v1": True,
        "low_support_policy_applied_v1": True,
        "strict_loso_reported_v1": True,
        "low_support_registry_reported_v1": True,
        "final_promotion_allowed_v1": False,
    }


def _source_mapping(output_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_TRAINING_SOURCE_MAPPING_V1",
        "existing_training_utilities_used_v1": True,
        "existing_feature_builder_used_v1": True,
        "existing_label_source_tables_used_v1": True,
        "new_model_logic_introduced_v1": False,
        "new_feature_surface_introduced_v1": False,
        "behavior_copied_reimplemented_v1": False,
        "thin_orchestrator_required_v1": True,
        "reason_v1": "Coverage-aware target mapping and fixed-control OOF orchestration are new, while the feature loader, legal feature selection, grouped split style, provenance hashes, and HistGradientBoostingClassifier family are reused from the V2 path.",
        "source_files_v1": {
            "current_materializer_v1": str(Path(__file__).resolve()),
            "historical_v2_runner_v1": str(Path(historical_v2.__file__).resolve()),
            "v2_oof_replay_runner_v1": str(Path(v2_replay.__file__).resolve()),
        },
        "feature_source_v1": str(inputs["score_dir"]),
        "label_source_v1": str(inputs["label_path"]),
        "output_root_v1": str(output_dir),
    }


def _reports(
    output_dir: Path,
    *,
    contract: dict[str, Any],
    target_summary: dict[str, Any],
    source_mapping: dict[str, Any],
    eval_summary: dict[str, Any],
    threshold_rows: list[dict[str, Any]],
    best: dict[str, Any],
    go_no_go: dict[str, Any],
) -> None:
    _write_report(
        output_dir / "r5_2_from_coverage_aware_base_rebuild_contract_v1.md",
        [
            "# R5.2 From Coverage-Aware Base Rebuild Contract V1",
            "",
            f"Input opportunity base: `{contract['input_opportunity_base_v1']}`",
            f"Foundation rows: `{contract['foundation_rows_observed_v1']}`",
            f"Active/quarantine: `{contract['active_rows_observed_v1']}` / `{contract['quarantine_rows_observed_v1']}`",
            f"AS_OF columns: `{contract['as_of_columns_observed_v1']}`",
            "Grouped OOF, validation-only scoring, provenance, low-support reporting, and final-promotion block are required.",
        ],
    )
    _write_report(
        output_dir / "r5_2_training_target_table_report_v1.md",
        [
            "# R5.2 Training Target Table V1",
            "",
            f"Rows: `{target_summary['rows_v1']}`",
            f"Positive bad target rows: `{target_summary['bad_target_rows_v1']}`",
            f"Positive tail target rows: `{target_summary['tail_target_rows_v1']}`",
            f"Hard negatives: `{target_summary['hard_negative_rows_v1']}`",
            f"Excluded rows: `{target_summary['excluded_rows_v1']}`",
        ],
    )
    _write_report(
        output_dir / "r5_2_training_source_mapping_v1.md",
        [
            "# R5.2 Training Source Mapping V1",
            "",
            f"Existing feature builder used: `{source_mapping['existing_feature_builder_used_v1']}`",
            f"Existing label/source tables used: `{source_mapping['existing_label_source_tables_used_v1']}`",
            f"New feature surface introduced: `{source_mapping['new_feature_surface_introduced_v1']}`",
            f"Reason: {source_mapping['reason_v1']}",
        ],
    )
    _write_report(
        output_dir / "r5_2_oof_eval_summary_v1.md",
        [
            "# R5.2 OOF Eval Summary V1",
            "",
            f"Best threshold candidate: `{best['threshold_candidate_id_v1']}`",
            f"Bad/tail: `{best['bad_count_v1']}` / `{best['tail_count_v1']}`",
            f"Precision: `{best['precision_v1']}` denominator `{best['precision_denominator_v1']}`",
            f"Strict LOSO: `{best['strict_all_run_id_worst_loso_v1']}` denominator `{best['strict_all_run_id_worst_loso_denominator_v1']}`",
            f"Safety clean: `{best['safety_clean_v1']}`",
            f"Final promotion allowed: `{best['final_promotion_allowed_v1']}`",
        ],
    )
    _write_report(
        output_dir / "r5_2_threshold_selection_report_v1.md",
        [
            "# R5.2 Threshold Selection Report V1",
            "",
            "Small fixed deterministic grid only; no Optuna or broad sweep was run.",
            f"Candidates evaluated: `{len(threshold_rows)}`",
            f"Selected candidate: `{best['threshold_candidate_id_v1']}`",
            f"Selected status: `{best['recommendation_status_v1']}`",
        ],
    )
    _write_report(
        output_dir / "r5_2_best_candidate_from_coverage_aware_base_v1.md",
        [
            "# R5.2 Best Candidate From Coverage-Aware Base V1",
            "",
            f"Status: `{go_no_go['decision_v1']}`",
            f"Next: `{go_no_go['next_recommended_action_v1']}`",
            f"Best candidate: `{best['threshold_candidate_id_v1']}`",
            f"Bad/tail: `{best['bad_count_v1']}` / `{best['tail_count_v1']}`",
            "This is not packaged, not R6-ready, and not final-promotion-valid while structural low-support remains.",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build R5.2 From Coverage-Aware Opportunity Base V1",
            "",
            f"Go/no-go: `{go_no_go['decision_v1']}`",
            f"Best candidate: `{best['threshold_candidate_id_v1']}`",
            f"Bad/tail: `{best['bad_count_v1']}` / `{best['tail_count_v1']}`",
            f"OOF provenance: `{eval_summary['oof_provenance_status_v1']}`",
            f"Train/validation overlap: `{eval_summary['train_validation_overlap_count_v1']}`",
            f"Strict LOSO decision-valid: `{best['strict_all_run_id_decision_valid_v1']}`",
            f"Final promotion allowed: `{best['final_promotion_allowed_v1']}`",
            f"Next: `{go_no_go['next_recommended_action_v1']}`",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    coverage_root: Path = COVERAGE_ROOT,
    low_support_policy_root: Path = LOW_SUPPORT_POLICY_ROOT,
    v2_oof_root: Path = V2_OOF_ROOT,
    spec_dir: Path = historical_v2.DEFAULT_SPEC_DIR,
    foundation_score_dir: Path | None = None,
    label_table: Path | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    if fold_count < 2:
        raise RuntimeError("Grouped OOF candidate rebuild requires at least two folds")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    validate_loso_guard_not_weakened(DENOMINATOR_TARGET)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "coverage_rows_sha256_v1": _file_hash(coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv"),
        "low_support_registry_sha256_v1": _file_hash(low_support_policy_root / "structural_low_support_run_id_registry_v1.csv"),
    }
    coverage_inputs = _load_coverage_inputs(coverage_root, low_support_policy_root)
    inputs = v2_replay._prepare_inputs(spec_dir, foundation_score_dir, label_table)
    contract = _rebuild_contract(inputs, coverage_root, low_support_policy_root)
    feature_check = validate_no_forbidden_features(inputs["feature_names"])
    hindsight_check = validate_no_hindsight_features(inputs["feature_names"])
    if feature_check["status_v1"] != "PASS":
        raise RuntimeError(f"Forbidden features in R5.2 rebuild feature matrix: {feature_check}")
    if hindsight_check["status_v1"] != "PASS":
        raise RuntimeError(f"Hindsight features in R5.2 rebuild feature matrix: {hindsight_check}")

    target = _build_training_target_table(coverage_inputs["coverage_rows"], inputs["training_frame"])
    target_summary = {
        "rows_v1": int(len(target)),
        "bad_target_rows_v1": int(_bool(target, "bad_target_v1").sum()),
        "tail_target_rows_v1": int(_bool(target, "tail_target_v1").sum()),
        "hard_negative_rows_v1": int(_bool(target, "hard_negative_v1").sum()),
        "monitor_only_rows_v1": int(_bool(target, "monitor_only_v1").sum()),
        "excluded_rows_v1": int(_bool(target, "exclude_v1").sum()),
        "target_class_counts_v1": {str(key): int(value) for key, value in target["target_class_v1"].value_counts().to_dict().items()},
    }
    config_payload = {
        "action_v1": ACTION,
        "selected_opportunity_variant_v1": SELECTED_OPPORTUNITY_VARIANT,
        "threshold_candidates_v1": THRESHOLD_CANDIDATES,
        "fold_count_v1": fold_count,
        "denominator_target_v1": DENOMINATOR_TARGET,
        "coverage_recommendation_v1": coverage_inputs["coverage_recommendation"],
    }
    oof = _run_grouped_oof(output_dir, inputs=inputs, target=target, fold_count=fold_count, config_payload=config_payload)
    scores = oof["scores"]
    provenance = oof["provenance"]
    membership = oof["membership"]
    threshold_rows, selections, loso_detail = _threshold_grid(scores)
    best = _select_best_threshold(threshold_rows)
    best_selection = selections[str(best["threshold_candidate_id_v1"])]
    scores["r5_2_best_candidate_selected_v1"] = best_selection.values
    no_in_sample = validate_no_in_sample_scoring(scores)
    no_overlap = validate_no_train_validation_overlap(membership)
    provenance_check = validate_oof_provenance_complete(scores, provenance)
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    no_forbidden = validate_no_forbidden_actions(optuna=False, r6=False, package=False, freeze=False, live=False)
    status, next_action = _best_candidate_status(
        best,
        provenance_ok=provenance_check["decision_valid_v1"],
        no_in_sample_ok=no_in_sample["decision_valid_v1"],
        no_overlap_ok=no_overlap["decision_valid_v1"],
    )
    eval_summary = {
        "layer_name": "R5_2_OOF_EVAL_SUMMARY_V1",
        "best_threshold_candidate_v1": best["threshold_candidate_id_v1"],
        "oof_provenance_status_v1": provenance_check["status_v1"],
        "oof_provenance_decision_valid_v1": provenance_check["decision_valid_v1"],
        "train_validation_overlap_status_v1": no_overlap["status_v1"],
        "train_validation_overlap_count_v1": no_overlap["overlap_count_v1"],
        "in_sample_scored_status_v1": no_in_sample["status_v1"],
        "in_sample_scored_count_v1": no_in_sample["in_sample_scored_count_v1"],
        **best,
    }
    fixed_comparison = _fixed_control_comparison(best)
    denominator_rows = [
        {
            "metric_v1": "precision",
            "value_v1": best["precision_v1"],
            "numerator_v1": best["precision_numerator_v1"],
            "denominator_v1": best["precision_denominator_v1"],
            "denominator_status_v1": best["precision_denominator_status_v1"],
            "decision_valid_v1": best["precision_decision_valid_v1"],
        },
        {
            "metric_v1": "strict_all_run_id_worst_loso",
            "value_v1": best["strict_all_run_id_worst_loso_v1"],
            "numerator_v1": best["strict_all_run_id_worst_loso_numerator_v1"],
            "denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
            "denominator_status_v1": best["strict_all_run_id_worst_loso_denominator_status_v1"],
            "decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        },
    ]
    safety_rows = [
        {"safety_metric_v1": key, "value_v1": best[key], "pass_v1": int(best[key]) == 0}
        for key in [
            "fifty_plus_mfe_overlap_v1",
            "hundred_plus_mfe_overlap_v1",
            "two_hundred_plus_mfe_overlap_v1",
            "strongest_winner_overlap_v1",
            "protected_winner_selected_v1",
            "runner_protect_leakage_v1",
            "ambiguous_high_mfe_leakage_v1",
            "quarantine_selected_v1",
        ]
    ]
    low_support_report = {
        "strict_all_run_id_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
        "selected_low_support_group_count_v1": best["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best["structural_low_support_selected_group_count_v1"],
        "zero_selected_group_count_v1": best["zero_selected_group_count_v1"],
        "evaluable_group_count_v1": best["evaluable_group_count_v1"],
        "evaluable_groups_loso_v1": best["evaluable_groups_loso_v1"],
        "final_promotion_allowed_v1": best["final_promotion_allowed_v1"],
    }
    go_no_go = {
        "layer_name": "R5_2_FROM_COVERAGE_AWARE_BASE_GO_NO_GO_V1",
        "decision_v1": status,
        "next_recommended_action_v1": next_action,
        "best_threshold_candidate_v1": best["threshold_candidate_id_v1"],
        "final_promotion_allowed_v1": False,
        "package_built_v1": False,
        "r6_run_v1": False,
        "optuna_run_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    best_candidate = {
        "layer_name": "R5_2_BEST_CANDIDATE_FROM_COVERAGE_AWARE_BASE_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "best_threshold_candidate_v1": best,
        "fixed_control_comparison_v1": fixed_comparison,
        "final_promotion_block_reason_v1": "STRUCTURAL_LOW_SUPPORT_REMAINS_VISIBLE_NO_EXCEPTION_GATE",
        "not_package_built_v1": True,
        "not_r6_ready_v1": True,
        "not_freeze_promo_live_ready_v1": True,
    }
    input_hashes_after = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "coverage_rows_sha256_v1": _file_hash(coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv"),
        "low_support_registry_sha256_v1": _file_hash(low_support_policy_root / "structural_low_support_run_id_registry_v1.csv"),
    }
    unchanged = input_hashes_before == input_hashes_after
    source_mapping = _source_mapping(output_dir, inputs)
    attestation = {
        **source_mapping,
        "input_artifact_hash_integrity_pass_v1": unchanged,
        "v2_scores_provenance_model_objective_thresholds_unchanged_v1": unchanged,
        "no_unnecessary_reimplementation_v1": True,
    }

    pd.DataFrame(_target_output_rows(target)).to_csv(output_dir / "r5_2_training_target_table_v1.csv", index=False)
    _write_json(output_dir / "r5_2_training_target_table_v1.json", {"summary_v1": target_summary, "rows_v1": _target_output_rows(target)})
    scores.to_csv(output_dir / "r5_2_oof_scores_v1.csv", index=False)
    provenance.to_csv(output_dir / "r5_2_oof_score_provenance_v1.csv", index=False)
    oof["fold_assignment"].to_csv(output_dir / "r5_2_oof_fold_assignment_v1.csv", index=False)
    membership.to_csv(output_dir / "r5_2_train_validation_membership_v1.csv", index=False)
    oof["head_metrics"].to_csv(output_dir / "r5_2_oof_head_training_metrics_v1.csv", index=False)
    _write_json(output_dir / "r5_2_from_coverage_aware_base_rebuild_contract_v1.json", contract)
    _write_json(output_dir / "r5_2_training_source_mapping_v1.json", source_mapping)
    _write_json(output_dir / "r5_2_no_unnecessary_reimplementation_attestation_v1.json", attestation)
    _write_json(
        output_dir / "r5_2_score_source_manifest_v1.json",
        {
            "layer_name": "R5_2_SCORE_SOURCE_MANIFEST_V1",
            "scorefields_v1": [scorefield for _, scorefield in SCOREFIELDS],
            "fold_count_v1": fold_count,
            "fold_models_v1": oof["fold_models"],
            "model_family_v1": "HistGradientBoostingClassifier",
            "selected_opportunity_variant_v1": SELECTED_OPPORTUNITY_VARIANT,
            "feature_count_v1": len(oof["feature_names"]),
            "feature_families_v1": {key: len(value) for key, value in oof["feature_families"].items()},
            "no_new_feature_surface_v1": True,
        },
    )
    _write_json(
        output_dir / "r5_2_feature_label_hash_manifest_v1.json",
        {
            "layer_name": "R5_2_FEATURE_LABEL_HASH_MANIFEST_V1",
            "hashes_v1": oof["hashes"],
            "feature_check_v1": feature_check,
            "hindsight_check_v1": hindsight_check,
        },
    )
    _write_json(
        output_dir / "r5_2_no_in_sample_decisioning_attestation_v1.json",
        {
            "layer_name": "R5_2_NO_IN_SAMPLE_DECISIONING_ATTESTATION_V1",
            **no_in_sample,
            "train_validation_overlap_v1": no_overlap,
        },
    )
    _write_json(output_dir / "no_fallback_no_dummy_no_synthetic_attestation_v1.json", {**no_dummy, "no_forbidden_actions_v1": no_forbidden})
    _write_json(output_dir / "r5_2_oof_eval_summary_v1.json", eval_summary)
    _write_rows(output_dir / "r5_2_oof_eval_metrics_v1.csv", [eval_summary])
    _write_rows(output_dir / "r5_2_oof_metric_denominator_report_v1.csv", denominator_rows)
    _write_json(output_dir / "r5_2_oof_metric_denominator_report_v1.json", {"rows_v1": denominator_rows})
    _write_rows(output_dir / "r5_2_oof_safety_report_v1.csv", safety_rows)
    _write_json(output_dir / "r5_2_oof_safety_report_v1.json", {"rows_v1": safety_rows, "safety_clean_v1": best["safety_clean_v1"]})
    _write_rows(output_dir / "r5_2_oof_low_support_report_v1.csv", [low_support_report])
    _write_json(output_dir / "r5_2_oof_low_support_report_v1.json", low_support_report)
    _write_rows(output_dir / "r5_2_oof_fixed_control_comparison_v1.csv", fixed_comparison)
    _write_json(output_dir / "r5_2_oof_fixed_control_comparison_v1.json", {"controls_v1": fixed_comparison})
    _write_rows(output_dir / "r5_2_threshold_candidate_grid_v1.csv", threshold_rows)
    _write_json(output_dir / "r5_2_threshold_candidate_grid_v1.json", {"threshold_candidates_v1": threshold_rows})
    _write_rows(output_dir / "r5_2_oof_loso_group_detail_v1.csv", loso_detail)
    _write_json(output_dir / "r5_2_best_candidate_from_coverage_aware_base_v1.json", best_candidate)
    _write_json(output_dir / "r5_2_from_coverage_aware_base_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "coverage_root_v1": str(coverage_root),
                "low_support_policy_root_v1": str(low_support_policy_root),
                "v2_oof_root_v1": str(v2_oof_root),
                "spec_dir_v1": str(spec_dir),
                "score_dir_v1": str(inputs["score_dir"]),
                "label_table_v1": str(inputs["label_path"]),
            },
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_artifacts_unchanged_v1": unchanged,
        },
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "coverage_root_v1": str(coverage_root),
        "low_support_policy_root_v1": str(low_support_policy_root),
        "v2_oof_root_v1": str(v2_oof_root),
        "v2_scores_provenance_model_objective_thresholds_unchanged_v1": unchanged,
        "foundation_rows_v1": inputs["foundation"]["foundation_rows_v1"],
        "active_rows_v1": inputs["foundation"]["active_rows_v1"],
        "quarantine_rows_v1": inputs["foundation"]["quarantine_rows_v1"],
        "as_of_columns_v1": inputs["foundation"]["asof_columns_v1"],
        "training_target_summary_v1": target_summary,
        "best_threshold_candidate_v1": best["threshold_candidate_id_v1"],
        "bad_count_v1": best["bad_count_v1"],
        "tail_count_v1": best["tail_count_v1"],
        "precision_v1": best["precision_v1"],
        "precision_denominator_v1": best["precision_denominator_v1"],
        "precision_decision_valid_v1": best["precision_decision_valid_v1"],
        "strict_all_run_id_worst_loso_v1": best["strict_all_run_id_worst_loso_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
        "strict_all_run_id_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_group_count_v1": best["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best["structural_low_support_selected_group_count_v1"],
        "safety_clean_v1": best["safety_clean_v1"],
        "oof_provenance_status_v1": provenance_check["status_v1"],
        "train_validation_overlap_count_v1": no_overlap["overlap_count_v1"],
        "in_sample_scored_count_v1": no_in_sample["in_sample_scored_count_v1"],
        "final_promotion_allowed_v1": False,
        "go_no_go_v1": status,
        "next_recommended_action_v1": next_action,
        "optuna_not_run_v1": True,
        "r6_not_run_v1": True,
        "package_not_built_v1": True,
        "freeze_promo_live_not_run_v1": True,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": status})
    _reports(
        output_dir,
        contract=contract,
        target_summary=target_summary,
        source_mapping=source_mapping,
        eval_summary=eval_summary,
        threshold_rows=threshold_rows,
        best=best,
        go_no_go=go_no_go,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--coverage-aware-root", type=Path, default=COVERAGE_ROOT)
    parser.add_argument("--low-support-policy-root", type=Path, default=LOW_SUPPORT_POLICY_ROOT)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=historical_v2.DEFAULT_SPEC_DIR)
    parser.add_argument("--foundation-score-dir", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        coverage_root=args.coverage_aware_root,
        low_support_policy_root=args.low_support_policy_root,
        v2_oof_root=args.v2_oof_root,
        spec_dir=args.spec_dir,
        foundation_score_dir=args.foundation_score_dir,
        label_table=args.label_table,
        fold_count=args.fold_count,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
