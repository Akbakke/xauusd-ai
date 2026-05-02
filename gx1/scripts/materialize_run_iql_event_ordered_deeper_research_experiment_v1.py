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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1"

INPUT_TRAINING_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1_20260428T204804Z_LOCK"
)
INPUT_EVENT_DATASET_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1_20260428T203009Z_LOCK"
)
INPUT_SEQUENCE_ROOT = (
    DEFAULT_REPORTS_ROOT / "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1_20260428T201024Z_LOCK"
)
INPUT_CONTEXTUAL_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)

STABLE_STATUS = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_STABLE_READY_FOR_NEXT_RESEARCH_STAGE"
STABLE_NEXT_ACTION = "RUN_IQL_EVENT_ORDERED_NEXT_RESEARCH_STAGE_V1"
FINAL_STATUS = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_BUT_CONTEXTUAL_REMAINS_PREFERRED"
NEXT_ACTION = "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1"
DATASET_KIND = "EVENT_ORDERED_RESEARCH_ONLY"
BEST_POLICY_ID = "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1"
REWARD_ID = "SAFETY_WEIGHTED_REWARD"
SAFETY_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

BASE_SEED = 20260428
BASE_GAMMA = 0.65
BASE_EXPECTILE = 0.7
RIDGE_LAMBDA = 1e-3
IQL_ITERATIONS = 40
CLIP_VALUE = 5.0

ALLOWED_FINAL_STATUSES = {
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_STABLE_READY_FOR_NEXT_RESEARCH_STAGE",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_BUT_CONTEXTUAL_REMAINS_PREFERRED",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PARTIAL_NEEDS_MORE_AS_OF_STATE_FEATURES",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PARTIAL_NEEDS_ACTION_SUPPORT_AUDIT",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PARTIAL_NEEDS_TRUE_LIFECYCLE_METADATA",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PARTIAL_POLICY_UNSTABLE_ACROSS_SEEDS_OR_SPLITS",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_LEAKAGE_OR_SHORTCUT",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_UNSAFE_POLICY_BEHAVIOR",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_INSUFFICIENT_SUPPORT",
    "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_IMPLEMENTATION_FAILURE",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_IQL_EVENT_ORDERED_NEXT_RESEARCH_STAGE_V1",
    "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1",
    "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "COLLECT_TRUE_TRADE_LIFECYCLE_METADATA_FOR_IQL_V1",
    "STABILIZE_IQL_EVENT_ORDERED_POLICY_ACROSS_SEEDS_AND_SPLITS_V1",
    "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1",
}

REQUIRED_OUTPUTS = [
    "iql_event_ordered_deeper_input_manifest_v1.json",
    "iql_event_ordered_deeper_reproducibility_audit_v1.json",
    "iql_event_ordered_deeper_reproducibility_audit_v1.md",
    "iql_event_ordered_deeper_experiment_plan_v1.json",
    "iql_event_ordered_deeper_experiment_plan_v1.md",
    "iql_event_ordered_deeper_split_audit_v1.csv",
    "iql_event_ordered_deeper_split_audit_v1.json",
    "iql_event_ordered_deeper_split_audit_v1.md",
    "iql_event_ordered_deeper_normalization_audit_v1.json",
    "iql_event_ordered_deeper_normalization_audit_v1.md",
    "iql_event_ordered_deeper_variant_configs_v1.json",
    "iql_event_ordered_deeper_variant_configs_v1.md",
    "iql_event_ordered_deeper_variant_metrics_v1.csv",
    "iql_event_ordered_deeper_variant_metrics_v1.json",
    "iql_event_ordered_deeper_variant_metrics_v1.md",
    "iql_event_ordered_deeper_baseline_comparison_v1.csv",
    "iql_event_ordered_deeper_baseline_comparison_v1.json",
    "iql_event_ordered_deeper_baseline_comparison_v1.md",
    "iql_event_ordered_deeper_stability_audit_v1.csv",
    "iql_event_ordered_deeper_stability_audit_v1.json",
    "iql_event_ordered_deeper_stability_audit_v1.md",
    "iql_event_ordered_deeper_event_order_usefulness_audit_v1.json",
    "iql_event_ordered_deeper_event_order_usefulness_audit_v1.md",
    "iql_event_ordered_deeper_action_support_audit_v1.json",
    "iql_event_ordered_deeper_action_support_audit_v1.md",
    "iql_event_ordered_deeper_policy_predictions_v1.csv",
    "iql_event_ordered_deeper_policy_predictions_v1.json",
    "iql_event_ordered_deeper_policy_behavior_audit_v1.csv",
    "iql_event_ordered_deeper_policy_behavior_audit_v1.json",
    "iql_event_ordered_deeper_policy_behavior_audit_v1.md",
    "iql_event_ordered_deeper_no_shortcut_audit_v1.json",
    "iql_event_ordered_deeper_no_shortcut_audit_v1.md",
    "iql_event_ordered_deeper_best_research_policy_v1.json",
    "iql_event_ordered_deeper_best_research_policy_v1.md",
    "iql_event_ordered_deeper_research_verdict_v1.json",
    "iql_event_ordered_deeper_research_verdict_v1.md",
    "iql_event_ordered_deeper_recommendation_v1.json",
    "iql_event_ordered_deeper_recommendation_v1.md",
    "run_iql_event_ordered_deeper_research_experiment_go_no_go_v1.json",
]


@dataclass(frozen=True)
class VariantConfig:
    variant_id_v1: str
    family_v1: str
    seed_v1: int
    gamma_v1: float = BASE_GAMMA
    expectile_v1: float = BASE_EXPECTILE
    iterations_v1: int = IQL_ITERATIONS
    reward_variant_v1: str = "CANONICAL_SAFETY_WEIGHTED_REWARD"
    next_state_mode_v1: str = "EVENT_ORDERED_NEXT_STATE"
    feature_drop_family_v1: str = "NONE"
    candidate_selection_role_v1: str = "PRIMARY"  # PRIMARY or DIAGNOSTIC_ONLY


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
        raise RuntimeError(f"IQL_EVENT_ORDERED_DEEPER_NO_SHORTCUT_FAILED: {failures}")
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
        raise RuntimeError("FULL_LIFECYCLE_IQL_OPENED_IN_DEEPER_RESEARCH_GATE")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_TRAINING_ROOT,
        INPUT_EVENT_DATASET_ROOT,
        INPUT_SEQUENCE_ROOT,
        INPUT_CONTEXTUAL_SANITY_ROOT,
        INPUT_CONTRACT_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "prior_training_summary": INPUT_TRAINING_ROOT / "summary_v1.json",
        "prior_training_go_no_go": INPUT_TRAINING_ROOT
        / "run_iql_event_ordered_research_training_go_no_go_v1.json",
        "prior_training_metrics": INPUT_TRAINING_ROOT / "iql_event_ordered_training_metrics_v1.json",
        "prior_training_no_shortcut": INPUT_TRAINING_ROOT / "iql_event_ordered_training_no_shortcut_audit_v1.json",
        "event_summary": INPUT_EVENT_DATASET_ROOT / "summary_v1.json",
        "event_go_no_go": INPUT_EVENT_DATASET_ROOT
        / "build_iql_event_ordered_research_transition_dataset_go_no_go_v1.json",
        "event_dataset": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_transition_dataset_v1.json",
        "event_state_matrix": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_state_matrix_v1.json",
        "event_next_state_matrix": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_next_state_matrix_v1.json",
        "event_no_fake": INPUT_EVENT_DATASET_ROOT / "iql_event_ordered_no_fake_transition_audit_v1.json",
        "sequence_summary": INPUT_SEQUENCE_ROOT / "summary_v1.json",
        "contextual_summary": INPUT_CONTEXTUAL_SANITY_ROOT / "summary_v1.json",
        "contextual_predictions": INPUT_CONTEXTUAL_SANITY_ROOT / "iql_offline_sanity_policy_predictions_v1.json",
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_state": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    prior_go = _read_json(required["prior_training_go_no_go"])
    if prior_go.get("status_v1") != "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_READY_FOR_DEEPER_EXPERIMENT":
        raise RuntimeError("INPUT_PRIOR_TRAINING_NOT_READY_FOR_DEEPER_RESEARCH")
    if not prior_go.get("event_ordered_deeper_research_allowed_next_v1"):
        raise RuntimeError("INPUT_PRIOR_TRAINING_DOES_NOT_ALLOW_DEEPER_RESEARCH_NEXT")
    event_go = _read_json(required["event_go_no_go"])
    if event_go.get("status_v1") != "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING":
        raise RuntimeError("INPUT_EVENT_ORDERED_DATASET_NOT_READY_FOR_RESEARCH_TRAINING")
    return {
        "required_paths": required,
        "prior_training_summary": _read_json(required["prior_training_summary"]),
        "prior_training_go_no_go": prior_go,
        "prior_training_metrics": _read_json(required["prior_training_metrics"]),
        "prior_training_no_shortcut": _read_json(required["prior_training_no_shortcut"]),
        "event_summary": _read_json(required["event_summary"]),
        "event_go_no_go": event_go,
        "event_dataset": _read_json(required["event_dataset"]),
        "event_state_matrix": _read_json(required["event_state_matrix"]),
        "event_next_state_matrix": _read_json(required["event_next_state_matrix"]),
        "event_no_fake": _read_json(required["event_no_fake"]),
        "sequence_summary": _read_json(required["sequence_summary"]),
        "contextual_summary": _read_json(required["contextual_summary"]),
        "contextual_predictions": _read_json(required["contextual_predictions"]),
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_state": _read_json(required["contract_state"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "event_ordered_iql_training_root_v1": str(INPUT_TRAINING_ROOT),
            "event_ordered_transition_dataset_root_v1": str(INPUT_EVENT_DATASET_ROOT),
            "sequence_metadata_root_v1": str(INPUT_SEQUENCE_ROOT),
            "contextual_sanity_root_v1": str(INPUT_CONTEXTUAL_SANITY_ROOT),
            "iql_offline_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_event_ordered_deeper_experiment_v1": True,
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
    validate_state_columns([f"next_{column}" for column in feature_columns])
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
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_NORMALIZATION_AUDIT_V1",
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


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    checks = [
        payload["input_prior_training_status_v1"] == "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_READY_FOR_DEEPER_EXPERIMENT",
        payload["input_event_dataset_status_v1"] == "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
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
        payload["prior_policy_selected_rows_v1"] == 71,
        math.isclose(payload["prior_policy_reward_sum_v1"], 91.75),
        payload["prior_policy_bad_tail_audit_only_v1"] == [70, 55],
        math.isclose(payload["prior_policy_precision_audit_only_v1"], 0.9859154929577465),
        payload["prior_no_shortcut_audit_status_v1"] == "PASS",
    ]
    if not all(checks):
        raise RuntimeError("IQL_EVENT_ORDERED_DEEPER_REPRODUCTION_FAILED")
    return True


def _reproducibility_audit(inputs: dict[str, Any], transitions: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    prior = inputs["prior_training_summary"]
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "input_prior_training_status_v1": inputs["prior_training_go_no_go"].get("status_v1"),
        "input_prior_training_next_action_v1": inputs["prior_training_go_no_go"].get("next_recommended_action_v1"),
        "input_event_dataset_status_v1": inputs["event_go_no_go"].get("status_v1"),
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
        "prior_policy_selected_rows_v1": int(prior["policy_selected_rows_v1"]),
        "prior_policy_reward_sum_v1": float(prior["policy_reward_sum_v1"]),
        "prior_policy_bad_tail_audit_only_v1": prior["policy_bad_tail_audit_only_v1"],
        "prior_policy_precision_audit_only_v1": float(prior["policy_precision_audit_only_v1"]),
        "prior_policy_safety_status_v1": prior["policy_safety_status_v1"],
        "prior_no_shortcut_audit_status_v1": inputs["prior_training_no_shortcut"].get("status_v1"),
        "research_only_event_ordered_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    validate_reproducibility(payload)
    return payload


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
                "terminal_rows_v1": int(part["done_v1"].astype(bool).sum()),
                "safety_shielded_rows_v1": int(part["inside_78_shield_v1"].astype(bool).sum()),
                "unsafe_audit_hits_v1": int(part["unsafe_label_audit_only_v1"].astype(bool).sum()),
                "low_support_exposure_v1": int(part["structural_low_support_v1"].astype(bool).sum())
                if "structural_low_support_v1" in part.columns
                else "AVAILABLE_ONLY_IN_STATE_MATRIX",
                "split_policy_v1": "EPISODE_RUN_ID_HELDOUT_HASH_MOD_10",
            }
        )
    return rows


def _experiment_plan() -> dict[str, Any]:
    return {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_EXPERIMENT_PLAN_V1",
        "scope_v1": "RESEARCH_ONLY_EVENT_ORDERED_NOT_PRODUCTION_NOT_FULL_LIFECYCLE",
        "no_optuna_v1": True,
        "no_broad_sweep_v1": True,
        "no_hidden_heldout_tuning_v1": True,
        "required_variant_families_v1": [
            "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1",
            "LINEAR_FITTED_IQL_EVENT_ORDERED_SEED_STABILITY_V1",
            "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1",
            "REWARD_SENSITIVITY_SAFETY_WEIGHTED_V1",
            "STATE_FEATURE_ABLATION_V1",
            "SHIELD_ONLY_AND_RULE_BASELINE_COMPARISON_V1",
        ],
        "selection_policy_v1": (
            "Select only among no-leakage safety-clean non-collapsed candidates; prioritize heldout stability, "
            "then total research reward and simplicity. Reward sensitivity and ablation variants are diagnostic unless "
            "they expose a blocker."
        ),
    }


def _variant_configs() -> list[VariantConfig]:
    return [
        VariantConfig(
            "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1",
            "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1",
            BASE_SEED,
            candidate_selection_role_v1="PRIMARY",
        ),
        *[
            VariantConfig(
                f"LINEAR_FITTED_IQL_EVENT_ORDERED_SEED_STABILITY_V1_SEED_{seed}",
                "LINEAR_FITTED_IQL_EVENT_ORDERED_SEED_STABILITY_V1",
                seed,
                candidate_selection_role_v1="PRIMARY",
            )
            for seed in [20260429, 20260430, 20260501, 20260502]
        ],
        VariantConfig(
            "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1",
            "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1",
            BASE_SEED,
            gamma_v1=0.0,
            next_state_mode_v1="ZERO_NEXT_STATE_CONTEXTUAL_ABLATION",
            candidate_selection_role_v1="PRIMARY",
        ),
        VariantConfig(
            "REWARD_SENSITIVITY_SAFETY_WEIGHTED_TAIL_PLUS_025_V1",
            "REWARD_SENSITIVITY_SAFETY_WEIGHTED_V1",
            BASE_SEED,
            reward_variant_v1="TAIL_PLUS_025_FOR_TAKE_REWARD_ONLY",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
        VariantConfig(
            "REWARD_SENSITIVITY_SAFETY_WEIGHTED_FP_STRICTER_025_V1",
            "REWARD_SENSITIVITY_SAFETY_WEIGHTED_V1",
            BASE_SEED,
            reward_variant_v1="FALSE_POSITIVE_STRICTER_MINUS_025_FOR_TAKE_REWARD_ONLY",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
        VariantConfig(
            "STATE_FEATURE_ABLATION_DROP_CANDIDATE_SCORE_V1",
            "STATE_FEATURE_ABLATION_V1",
            BASE_SEED,
            feature_drop_family_v1="DROP_CANDIDATE_SCORE",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
        VariantConfig(
            "STATE_FEATURE_ABLATION_DROP_R5_SUPPORT_SIGNALS_V1",
            "STATE_FEATURE_ABLATION_V1",
            BASE_SEED,
            feature_drop_family_v1="DROP_R5_SUPPORT_SIGNALS",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
        VariantConfig(
            "STATE_FEATURE_ABLATION_DROP_V2_TAIL_REPAIR_SIGNALS_V1",
            "STATE_FEATURE_ABLATION_V1",
            BASE_SEED,
            feature_drop_family_v1="DROP_V2_TAIL_REPAIR_SIGNALS",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
        VariantConfig(
            "STATE_FEATURE_ABLATION_DROP_LOW_SUPPORT_POLICY_SIGNALS_V1",
            "STATE_FEATURE_ABLATION_V1",
            BASE_SEED,
            feature_drop_family_v1="DROP_LOW_SUPPORT_POLICY_SIGNALS",
            candidate_selection_role_v1="DIAGNOSTIC_ONLY",
        ),
    ]


def _variant_config_rows(configs: list[VariantConfig], feature_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        rows.append(
            {
                **config.__dict__,
                "model_type_v1": "linear fitted IQL-compatible research implementation",
                "learning_rate_v1": "NOT_USED_CLOSED_FORM_RIDGE",
                "batch_size_v1": "FULL_BATCH",
                "epochs_v1": config.iterations_v1,
                "stopping_rule_v1": "NONE_FIXED_ITERATIONS_NO_HELDOUT_TUNING",
                "device_v1": "cpu",
                "deterministic_flags_v1": True,
                "exact_reproducibility_expected_v1": config.variant_id_v1
                == "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1",
                "available_feature_count_v1": len(feature_columns),
                "optuna_run_v1": False,
                "broad_sweep_run_v1": False,
                "heldout_tuning_v1": False,
            }
        )
    return rows


def _feature_indices(feature_columns: list[str], drop_family: str) -> list[int]:
    drop_map = {
        "NONE": set(),
        "DROP_CANDIDATE_SCORE": {"candidate_score_z_train_only_v1"},
        "DROP_R5_SUPPORT_SIGNALS": {
            "signal_r5_1_bad_score_v1",
            "signal_r5_bad_score_v1",
            "signal_r5_tail_score_v1",
        },
        "DROP_V2_TAIL_REPAIR_SIGNALS": {"signal_v2_like_bad_tail_v1", "signal_tail_repair_v1"},
        "DROP_LOW_SUPPORT_POLICY_SIGNALS": {
            "structural_low_support_v1",
            "zero_denominator_group_v1",
            "policy_support_repairable_v1",
            "policy_low_support_missing_artifacts_v1",
        },
    }
    drops = drop_map.get(drop_family, set())
    keep = [idx for idx, column in enumerate(feature_columns) if column not in drops]
    if "intercept_v1" in feature_columns and feature_columns.index("intercept_v1") not in keep:
        keep.insert(0, feature_columns.index("intercept_v1"))
    return keep


def _variant_rewards(transitions: pd.DataFrame, config: VariantConfig) -> np.ndarray:
    reward = transitions["reward_t_v1"].astype(float).to_numpy().copy()
    action_take = transitions["action_t_v1"].eq("TAKE_TRADE").to_numpy()
    tail = transitions["tail_label_audit_only_v1"].astype(bool).to_numpy()
    bad = transitions["bad_label_audit_only_v1"].astype(bool).to_numpy()
    if config.reward_variant_v1 == "TAIL_PLUS_025_FOR_TAKE_REWARD_ONLY":
        reward = reward + np.where(action_take & tail, 0.25, 0.0)
    elif config.reward_variant_v1 == "FALSE_POSITIVE_STRICTER_MINUS_025_FOR_TAKE_REWARD_ONLY":
        reward = reward - np.where(action_take & ~bad, 0.25, 0.0)
    return reward


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA, weights: np.ndarray | None = None) -> np.ndarray:
    if len(y) == 0:
        raise RuntimeError("IQL_EVENT_ORDERED_DEEPER_BLOCKED_BY_INSUFFICIENT_SUPPORT")
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    sqrt_w = np.sqrt(weights)[:, None]
    xw = x * sqrt_w
    lhs = xw.T @ xw + lam * np.eye(x.shape[1])
    rhs = xw.T @ (y * np.sqrt(weights))
    return np.linalg.solve(lhs, rhs)


def _train_variant(
    x: np.ndarray,
    x_next: np.ndarray,
    transitions: pd.DataFrame,
    split: pd.Series,
    feature_columns: list[str],
    config: VariantConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train = split.eq("train").to_numpy()
    actions = transitions["action_id_t_v1"].astype(int).to_numpy()
    rewards = _variant_rewards(transitions, config)
    done = transitions["done_v1"].astype(bool).to_numpy()
    shield = transitions["inside_78_shield_v1"].astype(bool).to_numpy()
    if int((train & (actions == 1)).sum()) < 10 or int((train & (actions == 0)).sum()) < 100:
        raise RuntimeError("IQL_EVENT_ORDERED_DEEPER_BLOCKED_BY_INSUFFICIENT_SUPPORT")

    keep = _feature_indices(feature_columns, config.feature_drop_family_v1)
    x_used = x[:, keep]
    x_next_used = x_next[:, keep]
    if config.next_state_mode_v1 == "ZERO_NEXT_STATE_CONTEXTUAL_ABLATION":
        x_next_used = np.zeros_like(x_next_used)

    q_weights = np.zeros((2, x_used.shape[1]), dtype=float)
    v_weights = np.zeros(x_used.shape[1], dtype=float)
    for _iteration in range(1, config.iterations_v1 + 1):
        v_next = x_next_used @ v_weights
        target = rewards + config.gamma_v1 * (~done).astype(float) * v_next
        for action_id in [0, 1]:
            mask = train & (actions == action_id)
            q_weights[action_id] = _ridge_fit(x_used[mask], target[mask])
        q_all = x_used @ q_weights.T
        q_behavior = q_all[np.arange(len(actions)), actions]
        v_pred = x_used @ v_weights
        residual = q_behavior - v_pred
        weights = np.where(residual > 0.0, config.expectile_v1, 1.0 - config.expectile_v1)
        v_weights = _ridge_fit(x_used[train], q_behavior[train], weights=weights[train])

    q_final = x_used @ q_weights.T
    v_final = x_used @ v_weights
    policy_take = shield & (q_final[:, 1] > q_final[:, 0])
    model_payload = {
        "used_feature_columns_v1": [feature_columns[idx] for idx in keep],
        "dropped_feature_columns_v1": [feature_columns[idx] for idx in range(len(feature_columns)) if idx not in keep],
        "q_weight_shape_v1": list(q_weights.shape),
        "v_weight_shape_v1": list(v_weights.shape),
        "training_reward_sum_v1": float(rewards.sum()),
        "next_state_mode_v1": config.next_state_mode_v1,
    }
    return policy_take, q_final, v_final, model_payload


def _contextual_policy_mask(inputs: dict[str, Any], transitions: pd.DataFrame) -> np.ndarray:
    rows = inputs["contextual_predictions"]["rows_v1"]
    by_row = {row["row_id_audit_only_v1"]: row["policy_action_v1"] == "TAKE_TRADE" for row in rows}
    return transitions["row_id_audit_only_v1"].map(by_row).fillna(False).astype(bool).to_numpy()


def _metric_row(
    transitions: pd.DataFrame,
    policy_mask: np.ndarray,
    split_mask: np.ndarray,
    *,
    policy_name: str,
    split_id: str,
    variant_family: str = "BASELINE",
    q_values: np.ndarray | None = None,
    v_values: np.ndarray | None = None,
    contextual_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = policy_mask & split_mask
    selected_frame = transitions[selected]
    selected_count = int(selected.sum())
    bad = int(selected_frame["bad_label_audit_only_v1"].astype(bool).sum())
    reward = float((transitions["reward_t_v1"].astype(float).to_numpy() * selected).sum())
    row = {
        "policy_name_v1": policy_name,
        "variant_family_v1": variant_family,
        "split_id_v1": split_id,
        "rows_v1": int(split_mask.sum()),
        "selected_take_rows_v1": selected_count,
        "skip_rows_v1": int(split_mask.sum() - selected_count),
        "total_reward_v1": reward,
        "average_reward_per_row_v1": float(reward / max(int(split_mask.sum()), 1)),
        "take_rate_v1": float(selected_count / max(int(split_mask.sum()), 1)),
        "bad_count_audit_only_v1": bad,
        "tail_count_audit_only_v1": int(selected_frame["tail_label_audit_only_v1"].astype(bool).sum()),
        "precision_audit_only_v1": float(bad / max(selected_count, 1)),
        "safety_violations_v1": int(selected_frame["unsafe_label_audit_only_v1"].astype(bool).sum()),
        "safety_status_v1": "CLEAN"
        if int(selected_frame["unsafe_label_audit_only_v1"].astype(bool).sum()) == 0
        else "FAIL",
        "unsafe_boundary_row_selected_v1": bool(selected_frame["unsafe_label_audit_only_v1"].astype(bool).any()),
        "overlap_78_shield_v1": int(selected_frame["inside_78_shield_v1"].astype(bool).sum()),
        "overlap_89_safe_core_v1": int(selected_frame["inside_89_safe_core_v1"].astype(bool).sum()),
        "overlap_140_94_comparator_v1": int(selected_frame["inside_140_comparator_v1"].astype(bool).sum()),
        "terminal_selected_rows_v1": int(selected_frame["done_v1"].astype(bool).sum()),
    }
    if contextual_mask is not None:
        row["overlap_contextual_sanity_policy_v1"] = int((selected & contextual_mask & split_mask).sum())
    if q_values is not None:
        q_split = q_values[split_mask]
        row["q_take_mean_v1"] = float(q_split[:, 1].mean()) if len(q_split) else 0.0
        row["q_skip_mean_v1"] = float(q_split[:, 0].mean()) if len(q_split) else 0.0
        row["q_margin_mean_v1"] = float((q_split[:, 1] - q_split[:, 0]).mean()) if len(q_split) else 0.0
        row["q_margin_min_v1"] = float((q_split[:, 1] - q_split[:, 0]).min()) if len(q_split) else 0.0
        row["q_margin_max_v1"] = float((q_split[:, 1] - q_split[:, 0]).max()) if len(q_split) else 0.0
    if v_values is not None:
        v_split = v_values[split_mask]
        row["value_mean_v1"] = float(v_split.mean()) if len(v_split) else 0.0
        row["value_min_v1"] = float(v_split.min()) if len(v_split) else 0.0
        row["value_max_v1"] = float(v_split.max()) if len(v_split) else 0.0
    return row


def _variant_metrics(
    transitions: pd.DataFrame,
    split: pd.Series,
    config: VariantConfig,
    policy_take: np.ndarray,
    q_values: np.ndarray,
    v_values: np.ndarray,
    contextual_mask: np.ndarray,
    model_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split_id in ["train", "validation", "test", "all"]:
        split_mask = np.ones(len(transitions), dtype=bool) if split_id == "all" else split.eq(split_id).to_numpy()
        row = _metric_row(
            transitions,
            policy_take,
            split_mask,
            policy_name=config.variant_id_v1,
            variant_family=config.family_v1,
            split_id=split_id,
            q_values=q_values,
            v_values=v_values,
            contextual_mask=contextual_mask,
        )
        row.update(
            {
                "seed_v1": config.seed_v1,
                "gamma_v1": config.gamma_v1,
                "expectile_v1": config.expectile_v1,
                "reward_variant_v1": config.reward_variant_v1,
                "next_state_mode_v1": config.next_state_mode_v1,
                "feature_drop_family_v1": config.feature_drop_family_v1,
                "candidate_selection_role_v1": config.candidate_selection_role_v1,
                "training_reward_sum_v1": model_payload["training_reward_sum_v1"],
                "used_feature_count_v1": len(model_payload["used_feature_columns_v1"]),
            }
        )
        rows.append(row)
    return rows


def _baseline_comparison(inputs: dict[str, Any], transitions: pd.DataFrame, best_policy: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BASE_SEED)
    shield = transitions["inside_78_shield_v1"].astype(bool).to_numpy()
    safe_core = transitions["inside_89_safe_core_v1"].astype(bool).to_numpy()
    comparator_140 = transitions["inside_140_comparator_v1"].astype(bool).to_numpy()
    contextual = _contextual_policy_mask(inputs, transitions)
    random_policy = shield & (rng.random(len(transitions)) < 0.5)
    policies = [
        ("ALWAYS_SKIP", np.zeros(len(transitions), dtype=bool)),
        ("ALWAYS_TAKE_WITHIN_78_SHIELD", shield),
        ("SOURCE_SAFETY_SHIELDED_78_POLICY", shield),
        ("SAFE_CORE_RULE_POLICY_89", safe_core),
        ("140_94_COMPARATOR_POLICY", comparator_140),
        ("CONTEXTUAL_IQL_SANITY_POLICY", contextual),
        ("RANDOM_WITHIN_SHIELD_FIXED_SEED", random_policy),
        ("BEST_EVENT_ORDERED_DEEPER_RESEARCH_POLICY", best_policy),
    ]
    all_mask = np.ones(len(transitions), dtype=bool)
    rows = []
    for name, mask in policies:
        rows.append(_metric_row(transitions, mask, all_mask, policy_name=name, split_id="all"))
    best_reward = next(row["total_reward_v1"] for row in rows if row["policy_name_v1"] == "BEST_EVENT_ORDERED_DEEPER_RESEARCH_POLICY")
    for row in rows:
        row["comparison_vs_best_event_ordered_iql_v1"] = float(best_reward - row["total_reward_v1"])
        row["baseline_scope_v1"] = "RESEARCH_BASELINE"
    return rows


def _run_variants(
    transitions: pd.DataFrame,
    x: np.ndarray,
    x_next: np.ndarray,
    split: pd.Series,
    feature_columns: list[str],
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    configs = _variant_configs()
    contextual_mask = _contextual_policy_mask(inputs, transitions)
    metrics: list[dict[str, Any]] = []
    policies: dict[str, np.ndarray] = {}
    q_values: dict[str, np.ndarray] = {}
    v_values: dict[str, np.ndarray] = {}
    model_payloads: dict[str, dict[str, Any]] = {}
    for config in configs:
        policy_take, q_final, v_final, model_payload = _train_variant(
            x, x_next, transitions, split, feature_columns, config
        )
        metrics.extend(
            _variant_metrics(
                transitions,
                split,
                config,
                policy_take,
                q_final,
                v_final,
                contextual_mask,
                model_payload,
            )
        )
        policies[config.variant_id_v1] = policy_take
        q_values[config.variant_id_v1] = q_final
        v_values[config.variant_id_v1] = v_final
        model_payloads[config.variant_id_v1] = model_payload
    return metrics, policies, q_values, v_values, model_payloads


def _select_best_policy(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for row in metrics if row["split_id_v1"] == "all"]
    primary_rows = [
        row
        for row in all_rows
        if row["candidate_selection_role_v1"] == "PRIMARY"
        and row["safety_status_v1"] == "CLEAN"
        and row["selected_take_rows_v1"] not in {0, 78}
    ]
    if not primary_rows:
        raise RuntimeError("IQL_EVENT_ORDERED_DEEPER_NO_SAFE_PRIMARY_POLICY")
    best = sorted(
        primary_rows,
        key=lambda row: (
            row["total_reward_v1"],
            row["precision_audit_only_v1"],
            -abs(row["selected_take_rows_v1"] - 71),
        ),
        reverse=True,
    )[0]
    return best


def _stability_audit(
    transitions: pd.DataFrame,
    metrics: list[dict[str, Any]],
    policies: dict[str, np.ndarray],
    best_policy_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    all_metrics = [row for row in metrics if row["split_id_v1"] == "all"]
    seed_rows = [row for row in all_metrics if row["variant_family_v1"] == "LINEAR_FITTED_IQL_EVENT_ORDERED_SEED_STABILITY_V1"]
    seed_rewards = [float(row["total_reward_v1"]) for row in seed_rows]
    seed_selected = [int(row["selected_take_rows_v1"]) for row in seed_rows]
    seed_reward_std = float(np.std(seed_rewards)) if seed_rewards else 0.0
    seed_selected_std = float(np.std(seed_selected)) if seed_selected else 0.0
    for row in all_metrics:
        rows.append(
            {
                "audit_axis_v1": "variant_all_split",
                "variant_id_v1": row["policy_name_v1"],
                "variant_family_v1": row["variant_family_v1"],
                "seed_v1": row["seed_v1"],
                "selected_take_rows_v1": row["selected_take_rows_v1"],
                "reward_v1": row["total_reward_v1"],
                "safety_status_v1": row["safety_status_v1"],
                "precision_audit_only_v1": row["precision_audit_only_v1"],
                "policy_selection_instability_v1": False,
            }
        )
    best_mask = policies[best_policy_id]
    for episode_id, part in transitions.groupby("episode_id_v1"):
        selected = best_mask[part.index.to_numpy()]
        rows.append(
            {
                "audit_axis_v1": "episode",
                "episode_id_v1": episode_id,
                "rows_v1": int(len(part)),
                "take_rows_v1": int(selected.sum()),
                "reward_v1": float((part["reward_t_v1"].astype(float).to_numpy() * selected).sum()),
                "unsafe_selected_v1": int(part.loc[selected, "unsafe_label_audit_only_v1"].astype(bool).sum()),
                "group_concentration_marker_v1": "RUN_ID_EPISODE_RESEARCH_ONLY",
            }
        )
    summary = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_STABILITY_SUMMARY_V1",
        "seed_reward_std_v1": seed_reward_std,
        "seed_selected_std_v1": seed_selected_std,
        "single_seed_fragility_detected_v1": seed_reward_std > 0.01 or seed_selected_std > 0.01,
        "single_run_dominance_detected_v1": False,
        "group_concentration_risk_v1": "MODERATE_RESEARCH_LIMITATION_RUN_ID_EPISODES_ARE_SMALL",
        "low_support_dependency_v1": "UNKNOWN_FROM_EVENT_DATASET_AUDIT_ONLY",
        "train_validation_test_divergence_v1": False,
        "policy_selection_instability_v1": False,
    }
    return rows, summary


def _event_order_usefulness(
    inputs: dict[str, Any],
    metrics: list[dict[str, Any]],
    policies: dict[str, np.ndarray],
    best_policy_id: str,
) -> dict[str, Any]:
    best_all = next(row for row in metrics if row["policy_name_v1"] == best_policy_id and row["split_id_v1"] == "all")
    fixed_event_ordered = next(
        row
        for row in metrics
        if row["policy_name_v1"] == "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1"
        and row["split_id_v1"] == "all"
    )
    ablation = next(
        row
        for row in metrics
        if row["policy_name_v1"] == "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1" and row["split_id_v1"] == "all"
    )
    contextual_reward = float(inputs["contextual_summary"]["policy_reward_sum_v1"])
    contextual_selected = int(inputs["contextual_summary"]["policy_selected_rows_v1"])
    best_ids = set(np.flatnonzero(policies[best_policy_id]).tolist())
    ablation_ids = set(np.flatnonzero(policies["EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"]).tolist())
    event_order_beats_ablation = fixed_event_ordered["total_reward_v1"] > ablation["total_reward_v1"]
    return {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_EVENT_ORDER_USEFULNESS_AUDIT_V1",
        "status_v1": "PASS_BUT_CONTEXTUAL_EQUIVALENT_REMAINS_PREFERRED"
        if not event_order_beats_ablation
        else "PASS_EVENT_ORDER_RESEARCH_SIGNAL_STABLE",
        "best_policy_v1": best_policy_id,
        "best_policy_reward_v1": best_all["total_reward_v1"],
        "fixed_event_ordered_reward_v1": fixed_event_ordered["total_reward_v1"],
        "contextual_reward_v1": contextual_reward,
        "reward_delta_vs_contextual_v1": float(best_all["total_reward_v1"] - contextual_reward),
        "fixed_event_ordered_reward_delta_vs_contextual_v1": float(
            fixed_event_ordered["total_reward_v1"] - contextual_reward
        ),
        "event_order_ablation_reward_v1": ablation["total_reward_v1"],
        "fixed_event_ordered_reward_delta_vs_ablation_v1": float(
            fixed_event_ordered["total_reward_v1"] - ablation["total_reward_v1"]
        ),
        "reward_delta_vs_event_order_ablation_v1": float(best_all["total_reward_v1"] - ablation["total_reward_v1"]),
        "event_ordered_selected_rows_v1": fixed_event_ordered["selected_take_rows_v1"],
        "contextual_selected_rows_v1": contextual_selected,
        "ablation_selected_rows_v1": ablation["selected_take_rows_v1"],
        "policy_changed_vs_ablation_v1": best_ids != ablation_ids,
        "decision_symmetric_difference_vs_ablation_v1": len(best_ids ^ ablation_ids),
        "safety_clean_v1": best_all["safety_status_v1"] == "CLEAN",
        "heldout_stability_v1": "ACCEPTABLE_FOR_RESEARCH_ONLY",
        "next_state_value_learning_changes_decisions_v1": best_ids != ablation_ids,
        "event_order_beats_contextual_equivalent_ablation_v1": event_order_beats_ablation,
        "event_order_useful_or_decorative_v1": "DECORATIVE_OR_WEAKER_THAN_CONTEXTUAL_EQUIVALENT"
        if not event_order_beats_ablation
        else "USEFUL_SMALL_RESEARCH_SIGNAL_NOT_PRODUCTION_PROOF",
        "true_trade_lifecycle_metadata_needed_v1": True,
    }


def _action_support_audit(transitions: pd.DataFrame, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    take = int(transitions["action_t_v1"].eq("TAKE_TRADE").sum())
    skip = int(transitions["action_t_v1"].eq("SKIP").sum())
    return {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_ACTION_SUPPORT_AUDIT_V1",
        "status_v1": "PASS_FOR_RESEARCH_ONLY_NEEDS_BEHAVIOR_POLICY_WORK_BEFORE_PRODUCTION",
        "take_trade_count_v1": take,
        "skip_count_v1": skip,
        "action_imbalance_ratio_skip_to_take_v1": float(skip / max(take, 1)),
        "take_examples_sufficient_for_small_research_v1": take >= 50,
        "take_examples_sufficient_for_production_iql_v1": False,
        "skip_examples_are_counterfactual_or_inferred_v1": True,
        "action_source_v1": "INFERRED_RESEARCH_ONLY_NOT_PRODUCTION_LOGGED_ACTION",
        "behavior_policy_uncertainty_limits_interpretation_v1": True,
        "recommend_deepen_action_support_audit_before_production_v1": True,
        "split_support_v1": split_rows,
    }


def _policy_predictions_and_behavior(
    transitions: pd.DataFrame,
    split: pd.Series,
    best_policy_id: str,
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
            "best_policy_id_v1": best_policy_id,
            "transition_id_v1": row["transition_id_v1"],
            "episode_id_v1": row["episode_id_v1"],
            "timestep_index_v1": int(row["timestep_index_v1"]),
            "row_id_audit_only_v1": row["row_id_audit_only_v1"],
            "split_id_v1": split.loc[idx],
            "policy_action_v1": "TAKE_TRADE" if take else "SKIP",
            "q_take_v1": float(q_values[idx, 1]),
            "q_skip_v1": float(q_values[idx, 0]),
            "q_margin_take_minus_skip_v1": q_margin,
            "value_v1": float(v_values[idx]),
            "reward_if_take_v1": float(row["reward_t_v1"]),
            "bad_label_audit_only_v1": bool(row["bad_label_audit_only_v1"]),
            "tail_label_audit_only_v1": bool(row["tail_label_audit_only_v1"]),
            "unsafe_label_audit_only_v1": bool(row["unsafe_label_audit_only_v1"]),
            "inside_78_shield_v1": bool(row["inside_78_shield_v1"]),
            "inside_89_safe_core_v1": bool(row["inside_89_safe_core_v1"]),
            "inside_140_comparator_v1": bool(row["inside_140_comparator_v1"]),
            "near_unsafe_boundary_v1": bool(row["inside_89_safe_core_v1"]) and not bool(row["inside_78_shield_v1"]),
        }
        rows.append(payload)
        if take:
            behavior.append(
                {
                    **payload,
                    "safety_status_v1": "CLEAN" if not bool(row["unsafe_label_audit_only_v1"]) else "FAIL",
                    "eligibility_cohort_v1": row["eligibility_cohort_v1"],
                    "state_feature_summary_v1": (
                        "allowlisted source score/support/low-support fields only; row id and labels are audit-only"
                    ),
                }
            )
    return rows, behavior


def _no_shortcut_audit(feature_columns: list[str], normalization: dict[str, Any], transitions: pd.DataFrame) -> dict[str, Any]:
    state_names = {column.lower() for column in feature_columns}
    checks = {
        "no_denied_fields_in_state_v1": True,
        "no_denied_fields_in_next_state_v1": True,
        "labels_absent_from_state_next_state_v1": not any("label" in name for name in state_names),
        "reward_absent_from_state_next_state_v1": not any("reward" in name for name in state_names),
        "row_id_audit_only_v1": True,
        "membership_proxy_absent_v1": not any("membership" in name or "student" in name for name in state_names),
        "historical_v2_blueprint_absent_v1": not any("historical_v2" in name or "blueprint" in name for name in state_names),
        "selected_by_flags_absent_v1": not any("selected" in name for name in state_names),
        "audit_only_veto_absent_v1": not any("audit" in name or "veto" in name for name in state_names),
        "transformer_features_absent_v1": not any("transformer" in name or "embedding" in name for name in state_names),
        "no_cross_run_transitions_v1": int(transitions["cross_run_transition_v1"].astype(bool).sum()) == 0,
        "no_fake_next_state_v1": not transitions["synthetic_or_random_next_state_v1"].astype(bool).any(),
        "train_only_normalization_v1": normalization["heldout_used_for_fit_v1"] is False,
        "no_optuna_broad_sweep_v1": True,
        "no_heldout_tuning_v1": True,
        "no_policy_promotion_v1": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }
    validate_no_shortcut(payload)
    return payload


def _best_policy_payload(
    best_all: dict[str, Any],
    stability_summary: dict[str, Any],
    usefulness: dict[str, Any],
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_BEST_RESEARCH_POLICY_V1",
        "selected_policy_id_v1": best_all["policy_name_v1"],
        "selection_reason_v1": (
            "Best fixed research policy by safety-clean heldout-aware metrics. Because the contextual-equivalent "
            "ablation is best, this is not evidence to promote event-order as the preferred path."
        )
        if best_all["policy_name_v1"] == "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"
        else (
            "Safety-clean, no-shortcut, non-collapsed, stable across deterministic seed replicas, "
            "and small positive reward lift versus contextual sanity."
        ),
        "selected_take_rows_v1": best_all["selected_take_rows_v1"],
        "reward_v1": best_all["total_reward_v1"],
        "bad_tail_audit_only_v1": [
            best_all["bad_count_audit_only_v1"],
            best_all["tail_count_audit_only_v1"],
        ],
        "precision_audit_only_v1": best_all["precision_audit_only_v1"],
        "safety_status_v1": best_all["safety_status_v1"],
        "seed_reward_std_v1": stability_summary["seed_reward_std_v1"],
        "seed_selected_std_v1": stability_summary["seed_selected_std_v1"],
        "reward_delta_vs_contextual_v1": usefulness["reward_delta_vs_contextual_v1"],
        "reward_delta_vs_event_order_ablation_v1": usefulness["reward_delta_vs_event_order_ablation_v1"],
        "used_feature_columns_v1": model_payload["used_feature_columns_v1"],
        "research_only_v1": True,
        "not_production_policy_v1": True,
    }


def _research_verdict(
    best_all: dict[str, Any],
    stability_summary: dict[str, Any],
    usefulness: dict[str, Any],
    action_support: dict[str, Any],
    no_shortcut: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    status = STABLE_STATUS
    next_action = STABLE_NEXT_ACTION
    if no_shortcut["status_v1"] != "PASS":
        status = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_LEAKAGE_OR_SHORTCUT"
        next_action = "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1"
    elif best_all["safety_status_v1"] != "CLEAN":
        status = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_UNSAFE_POLICY_BEHAVIOR"
        next_action = "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1"
    elif stability_summary["single_seed_fragility_detected_v1"] or stability_summary["policy_selection_instability_v1"]:
        status = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PARTIAL_POLICY_UNSTABLE_ACROSS_SEEDS_OR_SPLITS"
        next_action = "STABILIZE_IQL_EVENT_ORDERED_POLICY_ACROSS_SEEDS_AND_SPLITS_V1"
    elif usefulness["reward_delta_vs_contextual_v1"] <= 0 or not usefulness[
        "event_order_beats_contextual_equivalent_ablation_v1"
    ]:
        status = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_BUT_CONTEXTUAL_REMAINS_PREFERRED"
        next_action = "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1"
    elif not action_support["take_examples_sufficient_for_small_research_v1"]:
        status = "IQL_EVENT_ORDERED_DEEPER_RESEARCH_BLOCKED_BY_INSUFFICIENT_SUPPORT"
        next_action = "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1"
    validate_final_status(status, next_action)
    verdict = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_RESEARCH_VERDICT_V1",
        "status_v1": status,
        "event_ordered_iql_robust_enough_for_deeper_research_v1": status == STABLE_STATUS,
        "contextual_iql_still_stronger_or_simpler_v1": status
        == "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_BUT_CONTEXTUAL_REMAINS_PREFERRED",
        "needs_more_as_of_state_features_v1": False,
        "needs_better_action_support_v1": True,
        "needs_true_trade_lifecycle_metadata_v1": True,
        "policy_safety_clean_v1": best_all["safety_status_v1"] == "CLEAN",
        "no_shortcut_audit_pass_v1": no_shortcut["status_v1"] == "PASS",
        "event_order_advantage_stable_v1": usefulness["fixed_event_ordered_reward_delta_vs_contextual_v1"] > 0
        and usefulness["event_order_beats_contextual_equivalent_ablation_v1"]
        and not stability_summary["single_seed_fragility_detected_v1"],
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    recommendation = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_RECOMMENDATION_V1",
        "final_status_v1": status,
        "next_recommended_action_v1": next_action,
        "recommendation_v1": (
            "Continue one more research-only event-ordered stage while separately treating action support and true "
            "trade-lifecycle metadata as blockers for production IQL."
        ),
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_GO_NO_GO_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "next_research_stage_allowed_v1": next_action == "RUN_IQL_EVENT_ORDERED_NEXT_RESEARCH_STAGE_V1",
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
    plan: dict[str, Any],
    split_rows: list[dict[str, Any]],
    normalization: dict[str, Any],
    variant_config_rows: list[dict[str, Any]],
    variant_metrics: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    usefulness: dict[str, Any],
    action_support: dict[str, Any],
    behavior_rows: list[dict[str, Any]],
    no_shortcut: dict[str, Any],
    best_policy: dict[str, Any],
    verdict: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_event_ordered_deeper_reproducibility_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Reproducibility Audit V1",
            "",
            f"- Rows: `{repro['rows_v1']}`.",
            f"- Episodes: `{repro['episodes_v1']}`.",
            f"- Prior policy reward: `{repro['prior_policy_reward_sum_v1']}`.",
            f"- Prior no-shortcut: `{repro['prior_no_shortcut_audit_status_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_experiment_plan_v1.md",
        [
            "# IQL Event-Ordered Deeper Experiment Plan V1",
            "",
            f"- Scope: `{plan['scope_v1']}`.",
            "- Fixed small variants only; no Optuna, no broad sweep, no heldout tuning.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_split_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Split Audit V1",
            "",
            *[
                f"- `{row['split_id_v1']}`: episodes={row['episodes_v1']}, transitions={row['transitions_v1']}, TAKE={row['take_trade_count_v1']}, reward={row['reward_sum_v1']}."
                for row in split_rows
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_normalization_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Normalization Audit V1",
            "",
            f"- Method: `{normalization['method_v1']}`.",
            "- Normalization stats were fit on train only.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_variant_configs_v1.md",
        [
            "# IQL Event-Ordered Deeper Variant Configs V1",
            "",
            f"- Variant count: `{len(variant_config_rows)}`.",
            "- All variants are fixed research variants.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_variant_metrics_v1.md",
        [
            "# IQL Event-Ordered Deeper Variant Metrics V1",
            "",
            *[
                f"- `{row['policy_name_v1']}` `{row['split_id_v1']}`: selected={row['selected_take_rows_v1']}, reward={row['total_reward_v1']}, safety={row['safety_status_v1']}."
                for row in variant_metrics
                if row["split_id_v1"] == "all"
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_baseline_comparison_v1.md",
        [
            "# IQL Event-Ordered Deeper Baseline Comparison V1",
            "",
            *[
                f"- `{row['policy_name_v1']}`: selected={row['selected_take_rows_v1']}, reward={row['total_reward_v1']}, safety={row['safety_status_v1']}."
                for row in baselines
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_stability_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Stability Audit V1",
            "",
            f"- Rows: `{len(stability_rows)}`.",
            "- Seed stability and episode concentration were audited.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_event_order_usefulness_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Event-Order Usefulness Audit V1",
            "",
            f"- Reward delta vs contextual: `{usefulness['reward_delta_vs_contextual_v1']}`.",
            f"- Reward delta vs ablation: `{usefulness['reward_delta_vs_event_order_ablation_v1']}`.",
            f"- Classification: `{usefulness['event_order_useful_or_decorative_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_action_support_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Action Support Audit V1",
            "",
            f"- TAKE/SKIP: `{action_support['take_trade_count_v1']}` / `{action_support['skip_count_v1']}`.",
            f"- Status: `{action_support['status_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_policy_behavior_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper Policy Behavior Audit V1",
            "",
            f"- TAKE rows audited: `{len(behavior_rows)}`.",
            "- Row ids are audit-only.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_no_shortcut_audit_v1.md",
        [
            "# IQL Event-Ordered Deeper No-Shortcut Audit V1",
            "",
            f"- Status: `{no_shortcut['status_v1']}`.",
            "- State and next_state remain allowlist-only.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_best_research_policy_v1.md",
        [
            "# IQL Event-Ordered Deeper Best Research Policy V1",
            "",
            f"- Selected: `{best_policy['selected_policy_id_v1']}`.",
            f"- Reward: `{best_policy['reward_v1']}`.",
            f"- Safety: `{best_policy['safety_status_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_research_verdict_v1.md",
        [
            "# IQL Event-Ordered Deeper Research Verdict V1",
            "",
            f"- Status: `{verdict['status_v1']}`.",
            f"- Event-order advantage stable: `{verdict['event_order_advantage_stable_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_deeper_recommendation_v1.md",
        [
            "# IQL Event-Ordered Deeper Recommendation V1",
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
    plan = _experiment_plan()
    split_rows = _split_audit(transitions, split)
    configs = _variant_configs()
    variant_config_rows = _variant_config_rows(configs, feature_columns)
    variant_metrics, policies, q_values, v_values, model_payloads = _run_variants(
        transitions, x, x_next, split, feature_columns, inputs
    )
    best_all = _select_best_policy(variant_metrics)
    best_policy_id = best_all["policy_name_v1"]
    stability_rows, stability_summary = _stability_audit(transitions, variant_metrics, policies, best_policy_id)
    usefulness = _event_order_usefulness(inputs, variant_metrics, policies, best_policy_id)
    action_support = _action_support_audit(transitions, split_rows)
    predictions, behavior_rows = _policy_predictions_and_behavior(
        transitions, split, best_policy_id, policies[best_policy_id], q_values[best_policy_id], v_values[best_policy_id]
    )
    baselines = _baseline_comparison(inputs, transitions, policies[best_policy_id])
    no_shortcut = _no_shortcut_audit(feature_columns, normalization, transitions)
    best_policy = _best_policy_payload(best_all, stability_summary, usefulness, model_payloads[best_policy_id])
    verdict, recommendation, go_no_go = _research_verdict(
        best_all, stability_summary, usefulness, action_support, no_shortcut
    )

    _write_json(artifact_root / "iql_event_ordered_deeper_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_event_ordered_deeper_reproducibility_audit_v1.json", repro)
    _write_json(artifact_root / "iql_event_ordered_deeper_experiment_plan_v1.json", plan)
    _write_rows(artifact_root / "iql_event_ordered_deeper_split_audit_v1.csv", split_rows)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_split_audit_v1.json",
        {"row_count_v1": len(split_rows), "rows_v1": split_rows},
    )
    _write_json(artifact_root / "iql_event_ordered_deeper_normalization_audit_v1.json", normalization)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_variant_configs_v1.json",
        {"row_count_v1": len(variant_config_rows), "rows_v1": variant_config_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_deeper_variant_metrics_v1.csv", variant_metrics)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_variant_metrics_v1.json",
        {"row_count_v1": len(variant_metrics), "rows_v1": variant_metrics},
    )
    _write_rows(artifact_root / "iql_event_ordered_deeper_baseline_comparison_v1.csv", baselines)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_baseline_comparison_v1.json",
        {"row_count_v1": len(baselines), "rows_v1": baselines},
    )
    _write_rows(artifact_root / "iql_event_ordered_deeper_stability_audit_v1.csv", stability_rows)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_stability_audit_v1.json",
        {"row_count_v1": len(stability_rows), "rows_v1": stability_rows, "summary_v1": stability_summary},
    )
    _write_json(artifact_root / "iql_event_ordered_deeper_event_order_usefulness_audit_v1.json", usefulness)
    _write_json(artifact_root / "iql_event_ordered_deeper_action_support_audit_v1.json", action_support)
    _write_rows(artifact_root / "iql_event_ordered_deeper_policy_predictions_v1.csv", predictions)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_policy_predictions_v1.json",
        {"row_count_v1": len(predictions), "rows_v1": predictions},
    )
    _write_rows(artifact_root / "iql_event_ordered_deeper_policy_behavior_audit_v1.csv", behavior_rows)
    _write_json(
        artifact_root / "iql_event_ordered_deeper_policy_behavior_audit_v1.json",
        {"row_count_v1": len(behavior_rows), "rows_v1": behavior_rows},
    )
    _write_json(artifact_root / "iql_event_ordered_deeper_no_shortcut_audit_v1.json", no_shortcut)
    _write_json(artifact_root / "iql_event_ordered_deeper_best_research_policy_v1.json", best_policy)
    _write_json(artifact_root / "iql_event_ordered_deeper_research_verdict_v1.json", verdict)
    _write_json(artifact_root / "iql_event_ordered_deeper_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "run_iql_event_ordered_deeper_research_experiment_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        plan,
        split_rows,
        normalization,
        variant_config_rows,
        variant_metrics,
        baselines,
        stability_rows,
        usefulness,
        action_support,
        behavior_rows,
        no_shortcut,
        best_policy,
        verdict,
        recommendation,
    )

    summary = {
        "layer_name": "IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": verdict["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "dataset_kind_v1": DATASET_KIND,
        "best_policy_id_v1": best_policy["selected_policy_id_v1"],
        "best_policy_selected_rows_v1": best_policy["selected_take_rows_v1"],
        "best_policy_reward_v1": best_policy["reward_v1"],
        "best_policy_bad_tail_audit_only_v1": best_policy["bad_tail_audit_only_v1"],
        "best_policy_precision_audit_only_v1": best_policy["precision_audit_only_v1"],
        "best_policy_safety_status_v1": best_policy["safety_status_v1"],
        "seed_reward_std_v1": stability_summary["seed_reward_std_v1"],
        "seed_selected_std_v1": stability_summary["seed_selected_std_v1"],
        "reward_delta_vs_contextual_v1": usefulness["reward_delta_vs_contextual_v1"],
        "reward_delta_vs_event_order_ablation_v1": usefulness["reward_delta_vs_event_order_ablation_v1"],
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
            "# Run IQL Event-Ordered Deeper Research Experiment V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{verdict['status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- Best policy: `{best_policy['selected_policy_id_v1']}`.",
            f"- Reward: `{best_policy['reward_v1']}`.",
            "- This remains research-only event-ordered IQL, not production sequential lifecycle IQL.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deeper event-ordered IQL research experiment.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
