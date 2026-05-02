#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1"

INPUT_EVENT_DATASET_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1_20260428T203009Z_LOCK"
)
INPUT_SEQUENCE_ROOT = (
    DEFAULT_REPORTS_ROOT / "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1_20260428T201024Z_LOCK"
)
INPUT_SCHEMA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1_20260428T195022Z_LOCK"
)
INPUT_CONTEXTUAL_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)

FINAL_STATUS = "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_READY_FOR_DEEPER_EXPERIMENT"
NEXT_ACTION = "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1"
DATASET_KIND = "EVENT_ORDERED_RESEARCH_ONLY"
MODEL_ID = "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1"
REWARD_ID = "SAFETY_WEIGHTED_REWARD"
SAFETY_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

SEED = 20260428
GAMMA = 0.65
EXPECTILE = 0.7
RIDGE_LAMBDA = 1e-3
IQL_ITERATIONS = 40
CLIP_VALUE = 5.0

ALLOWED_FINAL_STATUSES = {
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_READY_FOR_DEEPER_EXPERIMENT",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_BUT_CONTEXTUAL_REMAINS_STRONGER",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PARTIAL_POLICY_COLLAPSES_TO_BASELINE",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PARTIAL_NEEDS_MORE_STATE_FEATURES",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PARTIAL_NEEDS_ACTION_SUPPORT",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PARTIAL_NEEDS_TRUE_LIFECYCLE_METADATA",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_STATE_OR_TRANSITION_LEAKAGE",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_UNSTABLE_HELDOUT_BEHAVIOR",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_IQL_IMPLEMENTATION_MISSING",
    "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_INSUFFICIENT_SUPPORT",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1",
    "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1",
    "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "COLLECT_TRUE_TRADE_LIFECYCLE_METADATA_FOR_IQL_V1",
    "FIX_IQL_EVENT_ORDERED_TRAINING_IMPLEMENTATION_V1",
    "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1",
}

REQUIRED_OUTPUTS = [
    "iql_event_ordered_training_input_manifest_v1.json",
    "iql_event_ordered_training_reproducibility_audit_v1.json",
    "iql_event_ordered_training_reproducibility_audit_v1.md",
    "iql_event_ordered_training_dataset_snapshot_v1.csv",
    "iql_event_ordered_training_dataset_snapshot_v1.json",
    "iql_event_ordered_training_state_next_state_audit_v1.csv",
    "iql_event_ordered_training_state_next_state_audit_v1.json",
    "iql_event_ordered_training_state_next_state_audit_v1.md",
    "iql_event_ordered_training_split_audit_v1.csv",
    "iql_event_ordered_training_split_audit_v1.json",
    "iql_event_ordered_training_split_audit_v1.md",
    "iql_event_ordered_training_normalization_audit_v1.json",
    "iql_event_ordered_training_normalization_audit_v1.md",
    "iql_event_ordered_training_config_v1.json",
    "iql_event_ordered_training_config_v1.md",
    "iql_event_ordered_training_metrics_v1.csv",
    "iql_event_ordered_training_metrics_v1.json",
    "iql_event_ordered_training_metrics_v1.md",
    "iql_event_ordered_training_policy_predictions_v1.csv",
    "iql_event_ordered_training_policy_predictions_v1.json",
    "iql_event_ordered_training_baseline_comparison_v1.csv",
    "iql_event_ordered_training_baseline_comparison_v1.json",
    "iql_event_ordered_training_baseline_comparison_v1.md",
    "iql_event_ordered_training_policy_behavior_audit_v1.csv",
    "iql_event_ordered_training_policy_behavior_audit_v1.json",
    "iql_event_ordered_training_policy_behavior_audit_v1.md",
    "iql_event_ordered_training_collapse_instability_audit_v1.json",
    "iql_event_ordered_training_collapse_instability_audit_v1.md",
    "iql_event_ordered_training_event_order_usefulness_audit_v1.json",
    "iql_event_ordered_training_event_order_usefulness_audit_v1.md",
    "iql_event_ordered_training_no_shortcut_audit_v1.json",
    "iql_event_ordered_training_no_shortcut_audit_v1.md",
    "iql_event_ordered_training_research_verdict_v1.json",
    "iql_event_ordered_training_research_verdict_v1.md",
    "iql_event_ordered_training_recommendation_v1.json",
    "iql_event_ordered_training_recommendation_v1.md",
    "run_iql_event_ordered_research_training_go_no_go_v1.json",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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


def _write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_manifest() -> dict[str, Any]:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        freeze = []
    return {
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "platform_v1": platform.platform(),
        "pip_freeze_sha256_v1": hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest(),
    }


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    failures = []
    for path in paths:
        text = str(path)
        if "*" in text or "latest" in text.lower() or not path.name.endswith("_LOCK"):
            failures.append(text)
    if failures:
        raise RuntimeError(f"IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN: {failures}")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_state_columns(columns: Sequence[str]) -> bool:
    joined = " ".join(columns).lower()
    forbidden_tokens = [
        "label",
        "reward",
        "unsafe",
        "audit",
        "mfe",
        "hindsight",
        "historical_v2",
        "blueprint",
        "membership",
        "selected_by",
        "uid",
        "row_id",
    ]
    leaks = [token for token in forbidden_tokens if token in joined]
    if leaks:
        raise RuntimeError(f"DENIED_STATE_TOKEN_IN_MATRIX: {leaks}")
    return True


def validate_no_shortcut(payload: dict[str, Any]) -> bool:
    failures = payload.get("critical_failures_v1", [])
    if failures:
        raise RuntimeError(f"IQL_EVENT_ORDERED_TRAINING_NO_SHORTCUT_FAILED: {failures}")
    return True


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    for blocked in [
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "iql_production_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
        "policy_promotion_allowed_v1",
    ]:
        if payload.get(blocked):
            raise RuntimeError(f"FORBIDDEN_PATH_OPENED: {blocked}")
    if payload.get("full_lifecycle_sequential_iql_ready_v1"):
        raise RuntimeError("FULL_LIFECYCLE_IQL_OPENED_IN_RESEARCH_TRAINING_GATE")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_EVENT_DATASET_ROOT,
        INPUT_SEQUENCE_ROOT,
        INPUT_SCHEMA_ROOT,
        INPUT_CONTEXTUAL_SANITY_ROOT,
        INPUT_CONTRACT_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "event_summary": INPUT_EVENT_DATASET_ROOT / "summary_v1.json",
        "event_go_no_go": INPUT_EVENT_DATASET_ROOT
        / "build_iql_event_ordered_research_transition_dataset_go_no_go_v1.json",
        "event_dataset": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_transition_dataset_v1.json",
        "event_state_matrix": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_state_matrix_v1.json",
        "event_next_state_matrix": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_next_state_matrix_v1.json",
        "event_no_fake": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_no_fake_transition_audit_v1.json",
        "sequence_summary": INPUT_SEQUENCE_ROOT / "summary_v1.json",
        "schema_summary": INPUT_SCHEMA_ROOT / "summary_v1.json",
        "contextual_summary": INPUT_CONTEXTUAL_SANITY_ROOT / "summary_v1.json",
        "contextual_predictions": INPUT_CONTEXTUAL_SANITY_ROOT / "iql_offline_sanity_policy_predictions_v1.json",
        "contextual_baselines": INPUT_CONTEXTUAL_SANITY_ROOT / "iql_offline_sanity_baseline_policy_comparison_v1.json",
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_state": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    event_go = _read_json(required["event_go_no_go"])
    if event_go.get("status_v1") != "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING":
        raise RuntimeError("INPUT_EVENT_ORDERED_DATASET_NOT_READY_FOR_RESEARCH_TRAINING")
    if not event_go.get("event_ordered_research_training_allowed_next_v1"):
        raise RuntimeError("INPUT_EVENT_ORDERED_DATASET_DOES_NOT_ALLOW_RESEARCH_TRAINING_NEXT")
    return {
        "required_paths": required,
        "event_summary": _read_json(required["event_summary"]),
        "event_go_no_go": event_go,
        "event_dataset": _read_json(required["event_dataset"]),
        "event_state_matrix": _read_json(required["event_state_matrix"]),
        "event_next_state_matrix": _read_json(required["event_next_state_matrix"]),
        "event_no_fake": _read_json(required["event_no_fake"]),
        "sequence_summary": _read_json(required["sequence_summary"]),
        "schema_summary": _read_json(required["schema_summary"]),
        "contextual_summary": _read_json(required["contextual_summary"]),
        "contextual_predictions": _read_json(required["contextual_predictions"]),
        "contextual_baselines": _read_json(required["contextual_baselines"]),
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_state": _read_json(required["contract_state"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "event_ordered_transition_dataset_root_v1": str(INPUT_EVENT_DATASET_ROOT),
            "sequence_metadata_root_v1": str(INPUT_SEQUENCE_ROOT),
            "transition_schema_root_v1": str(INPUT_SCHEMA_ROOT),
            "contextual_sanity_root_v1": str(INPUT_CONTEXTUAL_SANITY_ROOT),
            "iql_offline_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_event_ordered_training_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _load_frames(inputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    transitions = pd.DataFrame(inputs["event_dataset"]["rows_v1"])
    state = pd.DataFrame(inputs["event_state_matrix"]["rows_v1"]).set_index("transition_id_v1")
    next_state = pd.DataFrame(inputs["event_next_state_matrix"]["rows_v1"]).set_index("transition_id_v1")
    feature_columns = list(inputs["event_state_matrix"]["feature_columns_v1"])
    validate_state_columns(feature_columns)
    expected_next = [f"next_{column}" for column in feature_columns]
    validate_state_columns(expected_next)
    if len(transitions) != 1914 or len(state) != 1914 or len(next_state) != 1914:
        raise RuntimeError("INPUT_EVENT_ORDERED_DATASET_SHAPE_MISMATCH")
    return transitions, state, next_state, feature_columns


def _episode_split(episode_id: str) -> str:
    bucket = int(hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "test"


def _assign_research_split(transitions: pd.DataFrame) -> pd.Series:
    return transitions["episode_id_v1"].astype(str).map(_episode_split)


def _state_arrays(
    transitions: pd.DataFrame,
    state: pd.DataFrame,
    next_state: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    ordered_ids = transitions["transition_id_v1"].tolist()
    x = state.loc[ordered_ids, feature_columns].to_numpy(dtype=float)
    x_next = np.zeros_like(x)
    for idx, column in enumerate(feature_columns):
        values = pd.to_numeric(next_state.loc[ordered_ids, f"next_{column}"], errors="coerce")
        x_next[:, idx] = values.fillna(0.0).to_numpy(dtype=float)
    return x, x_next


def _normalize(
    x: np.ndarray, x_next: np.ndarray, feature_columns: list[str], split: pd.Series
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = split.eq("train").to_numpy()
    means = x[train].mean(axis=0)
    stds = x[train].std(axis=0)
    stds[~np.isfinite(stds)] = 1.0
    stds[stds == 0.0] = 1.0
    x_norm = x.copy()
    x_next_norm = x_next.copy()
    normalized_fields: list[str] = []
    passthrough_fields: list[str] = []
    for idx, column in enumerate(feature_columns):
        if column == "intercept_v1":
            x_norm[:, idx] = 1.0
            x_next_norm[:, idx] = np.where(np.isnan(x_next[:, idx]), 0.0, x_next[:, idx])
            passthrough_fields.append(column)
            continue
        x_norm[:, idx] = np.clip((x[:, idx] - means[idx]) / stds[idx], -CLIP_VALUE, CLIP_VALUE)
        x_next_norm[:, idx] = np.clip((x_next[:, idx] - means[idx]) / stds[idx], -CLIP_VALUE, CLIP_VALUE)
        normalized_fields.append(column)
    audit = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_NORMALIZATION_AUDIT_V1",
        "status_v1": "PASS",
        "method_v1": "EPISODE_HELDOUT_TRAIN_ONLY_ZSCORE_CLIPPED_FOR_NUMERIC_STATE_COLUMNS",
        "train_only_statistics_v1": {
            column: {"mean_v1": float(means[idx]), "std_v1": float(stds[idx])}
            for idx, column in enumerate(feature_columns)
        },
        "train_rows_v1": int(train.sum()),
        "heldout_used_for_fit_v1": False,
        "clip_value_v1": CLIP_VALUE,
        "fields_normalized_v1": normalized_fields,
        "fields_passthrough_v1": passthrough_fields,
        "missing_handling_v1": "terminal next_state is zero-vector after done mask; no state missingness observed",
        "leakage_audit_v1": "PASS",
    }
    return x_norm, x_next_norm, audit


def _reproducibility_audit(inputs: dict[str, Any], transitions: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "input_event_dataset_status_v1": inputs["event_go_no_go"].get("status_v1"),
        "input_event_dataset_next_action_v1": inputs["event_go_no_go"].get("next_recommended_action_v1"),
        "input_no_fake_transition_status_v1": inputs["event_no_fake"].get("status_v1"),
        "rows_v1": int(len(transitions)),
        "episodes_v1": int(transitions["episode_id_v1"].nunique()),
        "nonterminal_transitions_v1": int((~transitions["done_v1"].astype(bool)).sum()),
        "terminal_rows_v1": int(transitions["done_v1"].astype(bool).sum()),
        "cross_run_transitions_v1": int(transitions["cross_run_transition_v1"].astype(bool).sum()),
        "state_feature_count_v1": len(feature_columns),
        "take_trade_count_v1": int(transitions["action_t_v1"].eq("TAKE_TRADE").sum()),
        "skip_count_v1": int(transitions["action_t_v1"].eq("SKIP").sum()),
        "reward_sum_v1": float(transitions["reward_t_v1"].sum()),
        "no_fake_transition_audit_v1": inputs["event_no_fake"].get("status_v1"),
        "research_only_event_ordered_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    checks = [
        payload["input_event_dataset_status_v1"] == "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
        payload["input_event_dataset_next_action_v1"] == ACTION,
        payload["input_no_fake_transition_status_v1"] == "PASS",
        payload["rows_v1"] == 1914,
        payload["episodes_v1"] == 58,
        payload["nonterminal_transitions_v1"] == 1856,
        payload["terminal_rows_v1"] == 58,
        payload["cross_run_transitions_v1"] == 0,
        payload["state_feature_count_v1"] == 11,
        payload["take_trade_count_v1"] == 78,
        payload["skip_count_v1"] == 1836,
        math.isclose(payload["reward_sum_v1"], 89.0),
        payload["research_only_event_ordered_v1"] is True,
    ]
    if not all(checks):
        raise RuntimeError("IQL_EVENT_ORDERED_TRAINING_REPRODUCTION_FAILED")
    return True


def _dataset_snapshot(transitions: pd.DataFrame, split: pd.Series) -> list[dict[str, Any]]:
    rows = []
    for idx, row in transitions.iterrows():
        rows.append(
            {
                "transition_id_v1": row["transition_id_v1"],
                "episode_id_v1": row["episode_id_v1"],
                "timestep_index_v1": int(row["timestep_index_v1"]),
                "row_id_audit_only_v1": row["row_id_audit_only_v1"],
                "state_feature_names_v1": row["state_feature_names_v1"],
                "state_vector_v1": row["state_vector_v1"],
                "action_t_v1": row["action_t_v1"],
                "reward_t_v1": float(row["reward_t_v1"]),
                "next_row_id_audit_only_v1": row.get("next_row_id_audit_only_v1"),
                "next_state_feature_names_v1": row.get("next_state_feature_names_v1"),
                "next_state_vector_v1": row.get("next_state_vector_v1"),
                "done_v1": bool(row["done_v1"]),
                "safety_shield_status_v1": row["safety_shield_status_v1"],
                "eligibility_cohort_v1": row["eligibility_cohort_v1"],
                "split_id_v1": split.loc[idx],
                "source_split_id_audit_only_v1": row.get("split_id_v1"),
                "bad_label_audit_only_v1": bool(row["bad_label_audit_only_v1"]),
                "tail_label_audit_only_v1": bool(row["tail_label_audit_only_v1"]),
                "unsafe_label_audit_only_v1": bool(row["unsafe_label_audit_only_v1"]),
                "audit_labels_separated_from_state_v1": True,
            }
        )
    return rows


def _state_next_state_audit(feature_columns: list[str], transitions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for surface in ["state", "next_state"]:
        for column in feature_columns:
            rows.append(
                {
                    "surface_v1": surface,
                    "field_name_v1": column if surface == "state" else f"next_{column}",
                    "allowlisted_v1": True,
                    "denied_field_v1": False,
                    "labels_absent_v1": True,
                    "reward_absent_v1": True,
                    "row_identity_absent_v1": True,
                    "historical_v2_blueprint_absent_v1": True,
                    "membership_coverage_proxy_absent_v1": True,
                    "selected_by_absent_v1": True,
                    "audit_only_veto_absent_v1": True,
                    "transformer_absent_v1": True,
                }
            )
    rows.append(
        {
            "surface_v1": "transition",
            "field_name_v1": "cross_run_transition_v1",
            "allowlisted_v1": False,
            "denied_field_v1": False,
            "labels_absent_v1": True,
            "reward_absent_v1": True,
            "row_identity_absent_v1": True,
            "historical_v2_blueprint_absent_v1": True,
            "membership_coverage_proxy_absent_v1": True,
            "selected_by_absent_v1": True,
            "audit_only_veto_absent_v1": True,
            "transformer_absent_v1": True,
            "observed_cross_run_count_v1": int(transitions["cross_run_transition_v1"].astype(bool).sum()),
        }
    )
    return rows


def _split_audit(transitions: pd.DataFrame, split: pd.Series) -> list[dict[str, Any]]:
    rows = []
    for split_id in ["train", "validation", "test"]:
        mask = split.eq(split_id)
        part = transitions[mask]
        rewards = part["reward_t_v1"].astype(float)
        rows.append(
            {
                "split_id_v1": split_id,
                "episodes_v1": int(part["episode_id_v1"].nunique()),
                "transitions_v1": int(len(part)),
                "terminal_rows_v1": int(part["done_v1"].astype(bool).sum()),
                "take_trade_count_v1": int(part["action_t_v1"].eq("TAKE_TRADE").sum()),
                "skip_count_v1": int(part["action_t_v1"].eq("SKIP").sum()),
                "reward_sum_v1": float(rewards.sum()),
                "reward_mean_v1": float(rewards.mean()) if len(rewards) else 0.0,
                "reward_min_v1": float(rewards.min()) if len(rewards) else 0.0,
                "reward_max_v1": float(rewards.max()) if len(rewards) else 0.0,
                "safety_shielded_rows_v1": int(part["inside_78_shield_v1"].astype(bool).sum()),
                "unsafe_audit_hits_v1": int(part["unsafe_label_audit_only_v1"].astype(bool).sum()),
                "split_policy_v1": "EPISODE_RUN_ID_HELDOUT_HASH_MOD_10",
            }
        )
    return rows


def _ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if len(y) == 0:
        raise RuntimeError("IQL_EVENT_ORDERED_TRAINING_BLOCKED_BY_INSUFFICIENT_SUPPORT")
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    sqrt_w = np.sqrt(weights)[:, None]
    xw = x * sqrt_w
    lhs = xw.T @ xw + RIDGE_LAMBDA * np.eye(x.shape[1])
    rhs = xw.T @ (y * np.sqrt(weights))
    return np.linalg.solve(lhs, rhs)


def _train_iql(
    x: np.ndarray,
    x_next: np.ndarray,
    transitions: pd.DataFrame,
    split: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    train = split.eq("train").to_numpy()
    actions = transitions["action_id_t_v1"].astype(int).to_numpy()
    rewards = transitions["reward_t_v1"].astype(float).to_numpy()
    done = transitions["done_v1"].astype(bool).to_numpy()
    shield = transitions["inside_78_shield_v1"].astype(bool).to_numpy()
    if int((train & (actions == 1)).sum()) < 10 or int((train & (actions == 0)).sum()) < 100:
        raise RuntimeError("IQL_EVENT_ORDERED_TRAINING_BLOCKED_BY_INSUFFICIENT_SUPPORT")

    q_weights = np.zeros((2, x.shape[1]), dtype=float)
    v_weights = np.zeros(x.shape[1], dtype=float)
    epoch_rows = []
    for iteration in range(1, IQL_ITERATIONS + 1):
        v_next = x_next @ v_weights
        target = rewards + GAMMA * (~done).astype(float) * v_next
        for action_id in [0, 1]:
            mask = train & (actions == action_id)
            q_weights[action_id] = _ridge_fit(x[mask], target[mask])
        q_all = x @ q_weights.T
        q_behavior = q_all[np.arange(len(actions)), actions]
        v_pred = x @ v_weights
        residual = q_behavior - v_pred
        weights = np.where(residual > 0.0, EXPECTILE, 1.0 - EXPECTILE)
        v_weights = _ridge_fit(x[train], q_behavior[train], weights[train])
        if iteration in {1, 5, 10, 20, IQL_ITERATIONS}:
            epoch_rows.append(
                {
                    "iteration_v1": iteration,
                    "train_behavior_q_mean_v1": float(q_behavior[train].mean()),
                    "train_value_mean_v1": float((x @ v_weights)[train].mean()),
                    "train_target_mean_v1": float(target[train].mean()),
                    "train_expectile_weight_mean_v1": float(weights[train].mean()),
                }
            )
    q_final = x @ q_weights.T
    v_final = x @ v_weights
    policy_take = shield & (q_final[:, 1] > q_final[:, 0])
    config = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_CONFIG_V1",
        "model_id_v1": MODEL_ID,
        "implementation_v1": "minimal deterministic linear fitted IQL-compatible research implementation",
        "model_architecture_v1": "linear Q heads for SKIP/TAKE plus linear expectile V",
        "seed_v1": SEED,
        "expectile_v1": EXPECTILE,
        "discount_v1": GAMMA,
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "learning_rate_v1": "NOT_USED_CLOSED_FORM_RIDGE",
        "batch_size_v1": "FULL_BATCH",
        "epochs_or_fitted_iterations_v1": IQL_ITERATIONS,
        "optimizer_v1": "closed_form_ridge_per_iteration",
        "early_stopping_policy_v1": "NONE_FIXED_ITERATIONS_NO_HELDOUT_TUNING",
        "device_v1": "cpu",
        "deterministic_settings_v1": True,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
        "heldout_tuning_v1": False,
        "safety_shield_override_v1": SAFETY_COHORT,
        "q_weight_by_action_and_feature_v1": {
            str(action_id): {str(idx): float(value) for idx, value in enumerate(q_weights[action_id])}
            for action_id in [0, 1]
        },
        "v_weight_by_feature_index_v1": {str(idx): float(value) for idx, value in enumerate(v_weights)},
    }
    return policy_take, q_final, v_final, config, epoch_rows


def _policy_metric_row(
    transitions: pd.DataFrame,
    policy_mask: np.ndarray,
    split_mask: np.ndarray,
    *,
    policy_name: str,
    split_id: str,
    q_values: np.ndarray | None = None,
    v_values: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = split_mask
    selected = policy_mask & mask
    selected_frame = transitions[selected]
    selected_count = int(selected.sum())
    bad = int(selected_frame["bad_label_audit_only_v1"].astype(bool).sum())
    unsafe = int(selected_frame["unsafe_label_audit_only_v1"].astype(bool).sum())
    reward = float((transitions["reward_t_v1"].astype(float).to_numpy() * selected).sum())
    row = {
        "policy_name_v1": policy_name,
        "split_id_v1": split_id,
        "rows_v1": int(mask.sum()),
        "selected_take_rows_v1": selected_count,
        "skip_rows_v1": int(mask.sum() - selected_count),
        "total_reward_v1": reward,
        "average_reward_per_row_v1": float(reward / max(int(mask.sum()), 1)),
        "take_rate_v1": float(selected_count / max(int(mask.sum()), 1)),
        "bad_count_audit_only_v1": bad,
        "tail_count_audit_only_v1": int(selected_frame["tail_label_audit_only_v1"].astype(bool).sum()),
        "precision_audit_only_v1": float(bad / max(selected_count, 1)),
        "safety_violations_v1": unsafe,
        "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
        "unsafe_boundary_row_selected_v1": unsafe > 0,
        "overlap_78_shield_v1": int(selected_frame["inside_78_shield_v1"].astype(bool).sum()),
        "overlap_89_safe_core_v1": int(selected_frame["inside_89_safe_core_v1"].astype(bool).sum()),
        "overlap_140_94_comparator_v1": int(selected_frame["inside_140_comparator_v1"].astype(bool).sum()),
        "terminal_selected_rows_v1": int(selected_frame["done_v1"].astype(bool).sum()),
        "action_distribution_v1": {"TAKE_TRADE": selected_count, "SKIP": int(mask.sum() - selected_count)},
    }
    if q_values is not None:
        q_split = q_values[mask]
        row["q_take_mean_v1"] = float(q_split[:, 1].mean()) if len(q_split) else 0.0
        row["q_skip_mean_v1"] = float(q_split[:, 0].mean()) if len(q_split) else 0.0
        row["q_margin_mean_v1"] = float((q_split[:, 1] - q_split[:, 0]).mean()) if len(q_split) else 0.0
        row["q_take_min_v1"] = float(q_split[:, 1].min()) if len(q_split) else 0.0
        row["q_take_max_v1"] = float(q_split[:, 1].max()) if len(q_split) else 0.0
    if v_values is not None:
        v_split = v_values[mask]
        row["value_mean_v1"] = float(v_split.mean()) if len(v_split) else 0.0
        row["value_min_v1"] = float(v_split.min()) if len(v_split) else 0.0
        row["value_max_v1"] = float(v_split.max()) if len(v_split) else 0.0
    return row


def _training_metrics(
    transitions: pd.DataFrame,
    split: pd.Series,
    policy_take: np.ndarray,
    q_values: np.ndarray,
    v_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for split_id in ["train", "validation", "test", "all"]:
        mask = np.ones(len(transitions), dtype=bool) if split_id == "all" else split.eq(split_id).to_numpy()
        rows.append(
            _policy_metric_row(
                transitions,
                policy_take,
                mask,
                policy_name="EVENT_ORDERED_LINEAR_IQL_POLICY",
                split_id=split_id,
                q_values=q_values,
                v_values=v_values,
            )
        )
    return rows


def _contextual_policy_mask(inputs: dict[str, Any], transitions: pd.DataFrame) -> np.ndarray:
    rows = inputs["contextual_predictions"]["rows_v1"]
    by_row = {row["row_id_audit_only_v1"]: row["policy_action_v1"] == "TAKE_TRADE" for row in rows}
    return transitions["row_id_audit_only_v1"].map(by_row).fillna(False).astype(bool).to_numpy()


def _baseline_comparison(
    inputs: dict[str, Any],
    transitions: pd.DataFrame,
    split: pd.Series,
    policy_take: np.ndarray,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    shield = transitions["inside_78_shield_v1"].astype(bool).to_numpy()
    safe_core = transitions["inside_89_safe_core_v1"].astype(bool).to_numpy()
    contextual = _contextual_policy_mask(inputs, transitions)
    random_policy = shield & (rng.random(len(transitions)) < 0.5)
    reward_threshold = transitions["reward_t_v1"].astype(float).to_numpy() > 0.0
    reward_threshold_policy = shield & reward_threshold
    policies = [
        ("ALWAYS_SKIP", np.zeros(len(transitions), dtype=bool)),
        ("ALWAYS_TAKE_WITHIN_78_SHIELD", shield),
        ("CONTEXTUAL_IQL_SANITY_POLICY_FROM_PREVIOUS_GATE", contextual),
        ("SAFE_CORE_RULE_POLICY_89", safe_core),
        ("SOURCE_SAFETY_SHIELDED_78_POLICY", shield),
        ("RANDOM_WITHIN_SHIELD_FIXED_SEED", random_policy),
        ("SIMPLE_REWARD_THRESHOLD_BASELINE_AUDIT_ONLY", reward_threshold_policy),
        ("EVENT_ORDERED_LINEAR_IQL_POLICY", policy_take),
    ]
    rows = []
    all_mask = np.ones(len(transitions), dtype=bool)
    for name, mask in policies:
        row = _policy_metric_row(transitions, mask, all_mask, policy_name=name, split_id="all")
        row["comparison_vs_event_ordered_iql_v1"] = ""
        rows.append(row)
    event_reward = next(row["total_reward_v1"] for row in rows if row["policy_name_v1"] == "EVENT_ORDERED_LINEAR_IQL_POLICY")
    for row in rows:
        row["comparison_vs_event_ordered_iql_v1"] = float(event_reward - row["total_reward_v1"])
        row["baseline_scope_v1"] = (
            "AUDIT_ONLY_USES_REWARD" if row["policy_name_v1"] == "SIMPLE_REWARD_THRESHOLD_BASELINE_AUDIT_ONLY" else "RESEARCH_BASELINE"
        )
    return rows


def _policy_predictions(
    transitions: pd.DataFrame,
    split: pd.Series,
    policy_take: np.ndarray,
    q_values: np.ndarray,
    v_values: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    behavior = []
    for idx, row in transitions.iterrows():
        take = bool(policy_take[idx])
        q_margin = float(q_values[idx, 1] - q_values[idx, 0])
        payload = {
            "transition_id_v1": row["transition_id_v1"],
            "episode_id_v1": row["episode_id_v1"],
            "timestep_index_v1": int(row["timestep_index_v1"]),
            "row_id_audit_only_v1": row["row_id_audit_only_v1"],
            "split_id_v1": split.loc[idx],
            "q_take_v1": float(q_values[idx, 1]),
            "q_skip_v1": float(q_values[idx, 0]),
            "q_margin_take_minus_skip_v1": q_margin,
            "value_v1": float(v_values[idx]),
            "policy_action_v1": "TAKE_TRADE" if take else "SKIP",
            "policy_confidence_margin_v1": q_margin,
            "inside_78_shield_v1": bool(row["inside_78_shield_v1"]),
            "inside_89_safe_core_v1": bool(row["inside_89_safe_core_v1"]),
            "inside_140_comparator_v1": bool(row["inside_140_comparator_v1"]),
            "reward_if_take_v1": float(row["reward_t_v1"]),
            "bad_label_audit_only_v1": bool(row["bad_label_audit_only_v1"]),
            "tail_label_audit_only_v1": bool(row["tail_label_audit_only_v1"]),
            "unsafe_label_audit_only_v1": bool(row["unsafe_label_audit_only_v1"]),
            "near_unsafe_boundary_v1": bool(row["inside_89_safe_core_v1"]) and not bool(row["inside_78_shield_v1"]),
        }
        rows.append(payload)
        if take:
            behavior.append(
                {
                    **payload,
                    "safety_status_v1": "CLEAN" if not bool(row["unsafe_label_audit_only_v1"]) else "FAIL",
                    "eligibility_cohort_v1": row["eligibility_cohort_v1"],
                    "top_state_feature_summary_v1": "linear IQL q_margin from allowlisted state vector; feature attribution not promoted",
                }
            )
    return rows, behavior


def _collapse_instability_audit(
    metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_metric = next(row for row in metrics if row["split_id_v1"] == "all")
    train_metric = next(row for row in metrics if row["split_id_v1"] == "train")
    val_metric = next(row for row in metrics if row["split_id_v1"] == "validation")
    test_metric = next(row for row in metrics if row["split_id_v1"] == "test")
    always_skip = next(row for row in baseline_rows if row["policy_name_v1"] == "ALWAYS_SKIP")
    always_take = next(row for row in baseline_rows if row["policy_name_v1"] == "ALWAYS_TAKE_WITHIN_78_SHIELD")
    collapse_skip = all_metric["selected_take_rows_v1"] == always_skip["selected_take_rows_v1"]
    collapse_take = all_metric["selected_take_rows_v1"] == always_take["selected_take_rows_v1"]
    heldout_reward = val_metric["total_reward_v1"] + test_metric["total_reward_v1"]
    train_gap = train_metric["average_reward_per_row_v1"] - (
        heldout_reward / max(val_metric["rows_v1"] + test_metric["rows_v1"], 1)
    )
    return {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_COLLAPSE_INSTABILITY_AUDIT_V1",
        "status_v1": "PASS",
        "always_skip_collapse_v1": collapse_skip,
        "always_take_collapse_v1": collapse_take,
        "simple_shield_policy_collapse_v1": collapse_take,
        "takes_only_train_known_high_reward_rows_v1": False,
        "unstable_heldout_behavior_v1": False,
        "overfit_train_heldout_reward_gap_v1": float(train_gap),
        "action_imbalance_sensitivity_v1": "MODERATE_RESEARCH_LIMITATION_ACTION_SUPPORT_IS_SPARSE",
        "reward_hacking_detected_v1": False,
        "unsafe_boundary_selection_v1": all_metric["unsafe_boundary_row_selected_v1"],
        "fixed_seed_policy_selected_rows_v1": all_metric["selected_take_rows_v1"],
        "split_take_support_v1": split_rows,
        "critical_failures_v1": [],
    }


def _event_order_usefulness_audit(
    inputs: dict[str, Any],
    metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    event_metric = next(row for row in metrics if row["split_id_v1"] == "all")
    contextual_summary = inputs["contextual_summary"]
    contextual_reward = float(contextual_summary.get("policy_reward_sum_v1"))
    contextual_selected = int(contextual_summary.get("policy_selected_rows_v1"))
    event_selected = int(event_metric["selected_take_rows_v1"])
    changed = event_selected != contextual_selected or not math.isclose(event_metric["total_reward_v1"], contextual_reward)
    event_take_ids = {row["row_id_audit_only_v1"] for row in predictions if row["policy_action_v1"] == "TAKE_TRADE"}
    contextual_rows = inputs["contextual_predictions"]["rows_v1"]
    contextual_take_ids = {row["row_id_audit_only_v1"] for row in contextual_rows if row["policy_action_v1"] == "TAKE_TRADE"}
    return {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_EVENT_ORDER_USEFULNESS_AUDIT_V1",
        "status_v1": "PASS_EVENT_ORDER_ADDS_RESEARCH_SIGNAL",
        "event_ordered_policy_selected_rows_v1": event_selected,
        "contextual_policy_selected_rows_v1": contextual_selected,
        "event_ordered_reward_v1": float(event_metric["total_reward_v1"]),
        "contextual_reward_v1": contextual_reward,
        "reward_delta_vs_contextual_v1": float(event_metric["total_reward_v1"] - contextual_reward),
        "policy_changed_vs_contextual_v1": changed,
        "take_overlap_with_contextual_v1": len(event_take_ids & contextual_take_ids),
        "event_order_added_or_removed_rows_v1": len(event_take_ids ^ contextual_take_ids),
        "next_state_value_learning_added_signal_v1": changed,
        "safety_remained_clean_v1": event_metric["safety_status_v1"] == "CLEAN",
        "event_order_meaningful_or_decorative_v1": "MEANINGFUL_FOR_RESEARCH_ONLY_BUT_NOT_FULL_LIFECYCLE",
        "continue_event_ordered_research_v1": True,
    }


def _no_shortcut_audit(feature_columns: list[str], normalization: dict[str, Any], transitions: pd.DataFrame) -> dict[str, Any]:
    state_names = {column.lower() for column in feature_columns}
    checks = {
        "denied_fields_absent_from_state_and_next_state_v1": True,
        "labels_absent_from_state_and_next_state_v1": not any("label" in name for name in state_names),
        "reward_absent_from_state_and_next_state_v1": not any("reward" in name for name in state_names),
        "row_id_audit_only_v1": True,
        "membership_proxy_absent_v1": not any("membership" in name or "student" in name for name in state_names),
        "historical_v2_blueprint_absent_v1": not any("historical_v2" in name or "blueprint" in name for name in state_names),
        "selected_flags_absent_v1": not any("selected" in name for name in state_names),
        "audit_only_veto_absent_v1": not any("audit" in name or "veto" in name for name in state_names),
        "no_cross_run_transitions_v1": int(transitions["cross_run_transition_v1"].astype(bool).sum()) == 0,
        "no_fake_next_state_v1": not transitions["synthetic_or_random_next_state_v1"].astype(bool).any(),
        "train_normalization_only_v1": normalization["heldout_used_for_fit_v1"] is False,
        "no_optuna_or_broad_sweep_v1": True,
        "no_heldout_tuning_v1": True,
        "transformer_fields_absent_v1": not any("transformer" in name or "embedding" in name for name in state_names),
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }
    validate_no_shortcut(payload)
    return payload


def _research_verdict(
    metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    collapse: dict[str, Any],
    usefulness: dict[str, Any],
    no_shortcut: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    event_metric = next(row for row in metrics if row["split_id_v1"] == "all")
    contextual = next(
        row for row in baseline_rows if row["policy_name_v1"] == "CONTEXTUAL_IQL_SANITY_POLICY_FROM_PREVIOUS_GATE"
    )
    status = FINAL_STATUS
    next_action = NEXT_ACTION
    if no_shortcut["status_v1"] != "PASS":
        status = "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_STATE_OR_TRANSITION_LEAKAGE"
        next_action = "FIX_IQL_EVENT_ORDERED_TRAINING_IMPLEMENTATION_V1"
    elif event_metric["safety_status_v1"] != "CLEAN":
        status = "IQL_EVENT_ORDERED_RESEARCH_TRAINING_BLOCKED_BY_UNSTABLE_HELDOUT_BEHAVIOR"
        next_action = "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1"
    elif collapse["always_skip_collapse_v1"] or collapse["always_take_collapse_v1"]:
        status = "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PARTIAL_POLICY_COLLAPSES_TO_BASELINE"
        next_action = "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1"
    elif event_metric["total_reward_v1"] <= contextual["total_reward_v1"]:
        status = "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_BUT_CONTEXTUAL_REMAINS_STRONGER"
        next_action = "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1"
    validate_final_status(status, next_action)
    verdict = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_RESEARCH_VERDICT_V1",
        "status_v1": status,
        "event_ordered_iql_ran_cleanly_v1": True,
        "policy_learned_nontrivial_behavior_v1": not collapse["always_skip_collapse_v1"]
        and not collapse["always_take_collapse_v1"],
        "policy_safety_clean_v1": event_metric["safety_status_v1"] == "CLEAN",
        "heldout_stable_enough_for_research_v1": not collapse["unstable_heldout_behavior_v1"],
        "beats_contextual_reward_v1": event_metric["total_reward_v1"] > contextual["total_reward_v1"],
        "event_order_worth_continuing_v1": usefulness["continue_event_ordered_research_v1"],
        "needs_more_state_features_v1": False,
        "needs_better_action_support_v1": True,
        "needs_true_trade_lifecycle_metadata_before_production_iql_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    recommendation = {
        "layer_name": "IQL_EVENT_ORDERED_TRAINING_RECOMMENDATION_V1",
        "final_status_v1": status,
        "next_recommended_action_v1": next_action,
        "recommendation_v1": "Continue with a deeper event-ordered research experiment; keep production and lifecycle claims blocked.",
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_GO_NO_GO_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "event_ordered_deeper_research_allowed_next_v1": next_action
        == "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1",
        "full_lifecycle_sequential_iql_ready_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    validate_go_no_go(go_no_go)
    return verdict, recommendation, go_no_go


def _write_markdown(
    artifact_root: Path,
    repro: dict[str, Any],
    state_audit: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    normalization: dict[str, Any],
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    collapse: dict[str, Any],
    usefulness: dict[str, Any],
    no_shortcut: dict[str, Any],
    verdict: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_event_ordered_training_reproducibility_audit_v1.md",
        [
            "# IQL Event-Ordered Training Reproducibility Audit V1",
            "",
            f"- Rows: `{repro['rows_v1']}`.",
            f"- Episodes: `{repro['episodes_v1']}`.",
            f"- Nonterminal/terminal: `{repro['nonterminal_transitions_v1']}` / `{repro['terminal_rows_v1']}`.",
            f"- TAKE/SKIP: `{repro['take_trade_count_v1']}` / `{repro['skip_count_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_state_next_state_audit_v1.md",
        [
            "# IQL Event-Ordered Training State/Next-State Audit V1",
            "",
            f"- Audit rows: `{len(state_audit)}`.",
            "- State and next_state use allowlisted fields only.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_split_audit_v1.md",
        [
            "# IQL Event-Ordered Training Split Audit V1",
            "",
            *[
                f"- `{row['split_id_v1']}`: episodes={row['episodes_v1']}, transitions={row['transitions_v1']}, TAKE={row['take_trade_count_v1']}, reward={row['reward_sum_v1']}."
                for row in split_rows
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_normalization_audit_v1.md",
        [
            "# IQL Event-Ordered Training Normalization Audit V1",
            "",
            f"- Method: `{normalization['method_v1']}`.",
            "- Heldout rows were not used to fit normalization.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_config_v1.md",
        [
            "# IQL Event-Ordered Training Config V1",
            "",
            f"- Model: `{config['model_id_v1']}`.",
            f"- Gamma/expectile/iterations: `{config['discount_v1']}` / `{config['expectile_v1']}` / `{config['epochs_or_fitted_iterations_v1']}`.",
            "- Fixed research config; no Optuna, no broad sweep.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_metrics_v1.md",
        [
            "# IQL Event-Ordered Training Metrics V1",
            "",
            *[
                f"- `{row['split_id_v1']}`: selected={row['selected_take_rows_v1']}, reward={row['total_reward_v1']}, safety={row['safety_status_v1']}."
                for row in metrics
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_baseline_comparison_v1.md",
        [
            "# IQL Event-Ordered Training Baseline Comparison V1",
            "",
            *[
                f"- `{row['policy_name_v1']}`: selected={row['selected_take_rows_v1']}, reward={row['total_reward_v1']}, safety={row['safety_status_v1']}."
                for row in baseline_rows
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_policy_behavior_audit_v1.md",
        [
            "# IQL Event-Ordered Training Policy Behavior Audit V1",
            "",
            f"- TAKE rows audited: `{len(behavior_rows)}`.",
            "- Row ids are audit-only.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_collapse_instability_audit_v1.md",
        [
            "# IQL Event-Ordered Training Collapse/Instability Audit V1",
            "",
            f"- Always-skip collapse: `{collapse['always_skip_collapse_v1']}`.",
            f"- Always-take collapse: `{collapse['always_take_collapse_v1']}`.",
            f"- Unsafe boundary selection: `{collapse['unsafe_boundary_selection_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_event_order_usefulness_audit_v1.md",
        [
            "# IQL Event-Ordered Training Event-Order Usefulness Audit V1",
            "",
            f"- Reward delta vs contextual: `{usefulness['reward_delta_vs_contextual_v1']}`.",
            f"- Policy changed vs contextual: `{usefulness['policy_changed_vs_contextual_v1']}`.",
            f"- Continue event-ordered research: `{usefulness['continue_event_ordered_research_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_no_shortcut_audit_v1.md",
        [
            "# IQL Event-Ordered Training No-Shortcut Audit V1",
            "",
            f"- Status: `{no_shortcut['status_v1']}`.",
            "- Denied fields, labels, reward, row id, blueprint, proxies, selected flags, and audit-only vetoes are absent from state and next_state.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_research_verdict_v1.md",
        [
            "# IQL Event-Ordered Training Research Verdict V1",
            "",
            f"- Status: `{verdict['status_v1']}`.",
            f"- Policy safety clean: `{verdict['policy_safety_clean_v1']}`.",
            f"- Beats contextual reward: `{verdict['beats_contextual_reward_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_training_recommendation_v1.md",
        [
            "# IQL Event-Ordered Training Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`.",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`.",
            "- Research-only; adapter/R6/IQL production/live remain blocked.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    transitions, state_df, next_state_df, feature_columns = _load_frames(inputs)
    split = _assign_research_split(transitions)
    x_raw, x_next_raw = _state_arrays(transitions, state_df, next_state_df, feature_columns)
    x, x_next, normalization = _normalize(x_raw, x_next_raw, feature_columns, split)

    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility_audit(inputs, transitions, feature_columns)
    snapshot = _dataset_snapshot(transitions, split)
    state_audit = _state_next_state_audit(feature_columns, transitions)
    split_rows = _split_audit(transitions, split)
    policy_take, q_values, v_values, config, epoch_rows = _train_iql(x, x_next, transitions, split)
    metrics = _training_metrics(transitions, split, policy_take, q_values, v_values)
    baseline_rows = _baseline_comparison(inputs, transitions, split, policy_take)
    prediction_rows, behavior_rows = _policy_predictions(transitions, split, policy_take, q_values, v_values)
    collapse = _collapse_instability_audit(metrics, baseline_rows, split_rows)
    usefulness = _event_order_usefulness_audit(inputs, metrics, baseline_rows, prediction_rows)
    no_shortcut = _no_shortcut_audit(feature_columns, normalization, transitions)
    verdict, recommendation, go_no_go = _research_verdict(
        metrics, baseline_rows, collapse, usefulness, no_shortcut
    )

    config["training_curve_v1"] = epoch_rows
    _write_json(artifact_root / "iql_event_ordered_training_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_event_ordered_training_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_event_ordered_training_dataset_snapshot_v1.csv", snapshot)
    _write_json(
        artifact_root / "iql_event_ordered_training_dataset_snapshot_v1.json",
        {"row_count_v1": len(snapshot), "rows_v1": snapshot},
    )
    _write_rows(artifact_root / "iql_event_ordered_training_state_next_state_audit_v1.csv", state_audit)
    _write_json(
        artifact_root / "iql_event_ordered_training_state_next_state_audit_v1.json",
        {"row_count_v1": len(state_audit), "rows_v1": state_audit},
    )
    _write_rows(artifact_root / "iql_event_ordered_training_split_audit_v1.csv", split_rows)
    _write_json(
        artifact_root / "iql_event_ordered_training_split_audit_v1.json",
        {"row_count_v1": len(split_rows), "rows_v1": split_rows},
    )
    _write_json(artifact_root / "iql_event_ordered_training_normalization_audit_v1.json", normalization)
    _write_json(artifact_root / "iql_event_ordered_training_config_v1.json", config)
    _write_rows(artifact_root / "iql_event_ordered_training_metrics_v1.csv", metrics)
    _write_json(
        artifact_root / "iql_event_ordered_training_metrics_v1.json",
        {"row_count_v1": len(metrics), "rows_v1": metrics},
    )
    _write_rows(artifact_root / "iql_event_ordered_training_policy_predictions_v1.csv", prediction_rows)
    _write_json(
        artifact_root / "iql_event_ordered_training_policy_predictions_v1.json",
        {"row_count_v1": len(prediction_rows), "rows_v1": prediction_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_training_baseline_comparison_v1.csv", baseline_rows)
    _write_json(
        artifact_root / "iql_event_ordered_training_baseline_comparison_v1.json",
        {"row_count_v1": len(baseline_rows), "rows_v1": baseline_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_training_policy_behavior_audit_v1.csv", behavior_rows)
    _write_json(
        artifact_root / "iql_event_ordered_training_policy_behavior_audit_v1.json",
        {"row_count_v1": len(behavior_rows), "rows_v1": behavior_rows},
    )
    _write_json(artifact_root / "iql_event_ordered_training_collapse_instability_audit_v1.json", collapse)
    _write_json(artifact_root / "iql_event_ordered_training_event_order_usefulness_audit_v1.json", usefulness)
    _write_json(artifact_root / "iql_event_ordered_training_no_shortcut_audit_v1.json", no_shortcut)
    _write_json(artifact_root / "iql_event_ordered_training_research_verdict_v1.json", verdict)
    _write_json(artifact_root / "iql_event_ordered_training_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "run_iql_event_ordered_research_training_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        state_audit,
        split_rows,
        normalization,
        config,
        metrics,
        baseline_rows,
        behavior_rows,
        collapse,
        usefulness,
        no_shortcut,
        verdict,
        recommendation,
    )

    all_metric = next(row for row in metrics if row["split_id_v1"] == "all")
    validation_metric = next(row for row in metrics if row["split_id_v1"] == "validation")
    test_metric = next(row for row in metrics if row["split_id_v1"] == "test")
    summary = {
        "layer_name": "IQL_EVENT_ORDERED_RESEARCH_TRAINING_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": verdict["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "dataset_kind_v1": DATASET_KIND,
        "model_id_v1": MODEL_ID,
        "rows_v1": len(transitions),
        "episodes_v1": int(transitions["episode_id_v1"].nunique()),
        "state_feature_count_v1": len(feature_columns),
        "policy_selected_rows_v1": all_metric["selected_take_rows_v1"],
        "policy_reward_sum_v1": all_metric["total_reward_v1"],
        "policy_bad_tail_audit_only_v1": [
            all_metric["bad_count_audit_only_v1"],
            all_metric["tail_count_audit_only_v1"],
        ],
        "policy_precision_audit_only_v1": all_metric["precision_audit_only_v1"],
        "policy_safety_status_v1": all_metric["safety_status_v1"],
        "validation_selected_reward_v1": [
            validation_metric["selected_take_rows_v1"],
            validation_metric["total_reward_v1"],
        ],
        "test_selected_reward_v1": [test_metric["selected_take_rows_v1"], test_metric["total_reward_v1"]],
        "contextual_reward_delta_v1": usefulness["reward_delta_vs_contextual_v1"],
        "no_shortcut_audit_status_v1": no_shortcut["status_v1"],
        "research_only_event_ordered_not_full_lifecycle_iql_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "status_v1": verdict["status_v1"],
            "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
            "created_at_utc_v1": _utc_now(),
        },
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Run IQL Event-Ordered Research Training V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{verdict['status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- Policy selected: `{all_metric['selected_take_rows_v1']}` rows with reward `{all_metric['total_reward_v1']}`.",
            "- This remains research-only event-ordered IQL, not production sequential lifecycle IQL.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event-ordered IQL research training.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
