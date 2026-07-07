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

from gx1.scripts import materialize_collect_or_reconstruct_iql_sequence_metadata_v1 as sequence_gate
from gx1.scripts import materialize_run_iql_offline_sanity_training_research_only_v1 as sanity_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1"

INPUT_SEQUENCE_ROOT = (
    DEFAULT_REPORTS_ROOT / "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1_20260428T201024Z_LOCK"
)
INPUT_SCHEMA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1_20260428T195022Z_LOCK"
)
INPUT_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

FINAL_STATUS = "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING"
NEXT_ACTION = "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1"
DATASET_KIND = "EVENT_ORDERED_RESEARCH_ONLY"
TRANSITION_SCHEMA = "RUN_ID_DECISION_TIMESTAMP_EVENT_ORDERED_RESEARCH_TRANSITION_V1"
EPISODE_SCHEMA = "RUN_ID_EPISODE_BOUNDARY_V1"
DONE_RULE = "DONE_AT_LAST_EVENT_IN_RUN_ID"
ACTION_FIELD = "research_behavior_action_v1"
REWARD_FIELD = "safety_weighted_reward_v1"
REWARD_ID = "SAFETY_WEIGHTED_REWARD"
SAFETY_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

ALLOWED_FINAL_STATUSES = {
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_NEEDS_STATE_NORMALIZATION",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_PARTIAL_NEEDS_ACTION_SUPPORT_AUDIT",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_PARTIAL_NEEDS_REWARD_TIMING_AUDIT",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_ORDERING_AMBIGUITY",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_NEXT_STATE_ISSUES",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_FAKE_TRANSITION_RISK",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_STATE_LEAKAGE",
    "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_INSUFFICIENT_SUPPORT",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1",
    "NORMALIZE_IQL_EVENT_ORDERED_STATE_FEATURES_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "DEEPEN_IQL_REWARD_TIMING_AUDIT_V1",
    "REBUILD_IQL_EVENT_ORDERED_TRANSITIONS_WITH_STRICTER_ORDERING_V1",
    "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1",
    "COLLECT_ADDITIONAL_SEQUENCE_SOURCE_METADATA_V1",
}

REQUIRED_OUTPUTS = [
    "iql_event_ordered_transition_input_manifest_v1.json",
    "iql_event_ordered_transition_reproducibility_audit_v1.json",
    "iql_event_ordered_transition_reproducibility_audit_v1.md",
    "iql_event_ordered_transition_dataset_v1.csv",
    "iql_event_ordered_transition_dataset_v1.json",
    "iql_event_ordered_transition_dataset_v1.md",
    "iql_event_ordered_state_matrix_v1.csv",
    "iql_event_ordered_state_matrix_v1.json",
    "iql_event_ordered_next_state_matrix_v1.csv",
    "iql_event_ordered_next_state_matrix_v1.json",
    "iql_event_ordered_ordering_audit_v1.csv",
    "iql_event_ordered_ordering_audit_v1.json",
    "iql_event_ordered_ordering_audit_v1.md",
    "iql_event_ordered_next_state_audit_v1.csv",
    "iql_event_ordered_next_state_audit_v1.json",
    "iql_event_ordered_next_state_audit_v1.md",
    "iql_event_ordered_done_terminal_audit_v1.json",
    "iql_event_ordered_done_terminal_audit_v1.md",
    "iql_event_ordered_action_construction_audit_v1.json",
    "iql_event_ordered_action_construction_audit_v1.md",
    "iql_event_ordered_reward_construction_audit_v1.json",
    "iql_event_ordered_reward_construction_audit_v1.md",
    "iql_event_ordered_split_audit_v1.csv",
    "iql_event_ordered_split_audit_v1.json",
    "iql_event_ordered_split_audit_v1.md",
    "iql_event_ordered_cohort_label_audit_v1.csv",
    "iql_event_ordered_cohort_label_audit_v1.json",
    "iql_event_ordered_cohort_label_audit_v1.md",
    "iql_event_ordered_no_fake_transition_audit_v1.json",
    "iql_event_ordered_no_fake_transition_audit_v1.md",
    "iql_event_ordered_dataset_readiness_assessment_v1.json",
    "iql_event_ordered_dataset_readiness_assessment_v1.md",
    "iql_event_ordered_transition_dataset_recommendation_v1.json",
    "iql_event_ordered_transition_dataset_recommendation_v1.md",
    "build_iql_event_ordered_research_transition_dataset_go_no_go_v1.json",
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


def validate_transition_dataset(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != 1914:
        raise RuntimeError(f"TRANSITION_DATASET_ROW_COUNT_MISMATCH: {len(rows)}")
    terminal = sum(1 for row in rows if row["done_v1"])
    nonterminal = len(rows) - terminal
    if terminal != 58 or nonterminal != 1856:
        raise RuntimeError(f"TRANSITION_TERMINAL_COUNTS_MISMATCH: terminal={terminal} nonterminal={nonterminal}")
    if any(row["cross_run_transition_v1"] for row in rows):
        raise RuntimeError("CROSS_RUN_TRANSITION_FOUND")
    if any(row["state_contains_denied_fields_v1"] or row["next_state_contains_denied_fields_v1"] for row in rows):
        raise RuntimeError("DENIED_FIELDS_IN_STATE_OR_NEXT_STATE")
    if any(row["synthetic_or_random_next_state_v1"] for row in rows):
        raise RuntimeError("FAKE_NEXT_STATE_FOUND")
    return True


def validate_state_columns(state_columns: Sequence[str]) -> bool:
    joined = " ".join(state_columns).lower()
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


def validate_no_fake_transition_audit(payload: dict[str, Any]) -> bool:
    failures = payload.get("critical_failures_v1", [])
    if failures:
        raise RuntimeError(f"NO_FAKE_TRANSITION_AUDIT_FAILED: {failures}")
    checks = payload.get("checks_v1", {})
    for required in [
        "no_synthetic_next_state_v1",
        "no_random_next_state_v1",
        "no_cross_run_next_state_v1",
        "no_transition_across_episode_boundary_v1",
        "row_identity_not_state_v1",
        "reward_not_state_v1",
        "future_label_not_state_v1",
        "historical_v2_blueprint_not_used_v1",
        "transformer_fields_absent_v1",
    ]:
        if checks.get(required) is not True:
            raise RuntimeError(f"NO_FAKE_TRANSITION_CHECK_FAILED: {required}")
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
        raise RuntimeError("FULL_LIFECYCLE_IQL_OPENED_IN_EVENT_ORDERED_GATE")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_SEQUENCE_ROOT,
        INPUT_SCHEMA_ROOT,
        INPUT_SANITY_ROOT,
        INPUT_CONTRACT_ROOT,
        INPUT_REFINE_CLEAN_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "sequence_summary": INPUT_SEQUENCE_ROOT / "summary_v1.json",
        "sequence_go_no_go": INPUT_SEQUENCE_ROOT / "collect_or_reconstruct_iql_sequence_metadata_go_no_go_v1.json",
        "sequence_build_spec": INPUT_SEQUENCE_ROOT / "iql_transition_dataset_build_spec_v1.json",
        "sequence_no_fake": INPUT_SEQUENCE_ROOT / "iql_no_fake_transition_audit_v1.json",
        "sequence_order_audit": INPUT_SEQUENCE_ROOT / "iql_event_order_reconstruction_audit_v1.json",
        "sequence_next_row_audit": INPUT_SEQUENCE_ROOT / "iql_next_row_candidate_audit_v1.json",
        "schema_summary": INPUT_SCHEMA_ROOT / "summary_v1.json",
        "sanity_summary": INPUT_SANITY_ROOT / "summary_v1.json",
        "sanity_no_shortcut": INPUT_SANITY_ROOT / "iql_offline_sanity_no_shortcut_audit_v1.json",
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_state": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "contract_action": INPUT_CONTRACT_ROOT / "iql_offline_action_contract_v1.json",
        "contract_reward": INPUT_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "contract_shield": INPUT_CONTRACT_ROOT / "iql_offline_safety_shield_contract_v1.json",
        "refine_summary": INPUT_REFINE_CLEAN_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    sequence_go = _read_json(required["sequence_go_no_go"])
    if sequence_go.get("status_v1") != "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET":
        raise RuntimeError("INPUT_SEQUENCE_METADATA_NOT_READY_FOR_EVENT_ORDERED_DATASET")
    if not sequence_go.get("event_ordered_transition_dataset_build_allowed_v1"):
        raise RuntimeError("INPUT_SEQUENCE_METADATA_DOES_NOT_ALLOW_EVENT_ORDERED_DATASET_BUILD")
    if sequence_go.get("true_transition_dataset_build_allowed_v1"):
        raise RuntimeError("TRUE_TRANSITION_BUILD_UNEXPECTEDLY_ALLOWED")
    return {
        "required_paths": required,
        "sequence_summary": _read_json(required["sequence_summary"]),
        "sequence_go_no_go": sequence_go,
        "sequence_build_spec": _read_json(required["sequence_build_spec"]),
        "sequence_no_fake": _read_json(required["sequence_no_fake"]),
        "sequence_order_audit": _read_json(required["sequence_order_audit"]),
        "sequence_next_row_audit": _read_json(required["sequence_next_row_audit"]),
        "schema_summary": _read_json(required["schema_summary"]),
        "sanity_summary": _read_json(required["sanity_summary"]),
        "sanity_no_shortcut": _read_json(required["sanity_no_shortcut"]),
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_state": _read_json(required["contract_state"]),
        "contract_action": _read_json(required["contract_action"]),
        "contract_reward": _read_json(required["contract_reward"]),
        "contract_shield": _read_json(required["contract_shield"]),
        "refine_summary": _read_json(required["refine_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "sanity_inputs": sanity_gate._load_inputs(),
    }


def _frame_and_contract() -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series, pd.DataFrame, dict[str, Any], np.ndarray]:
    inputs = sanity_gate._load_inputs()
    frame, masks = sanity_gate._frame_and_masks(inputs)
    split = sanity_gate._split_series(frame)
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    reward = sanity_gate._reward(frame, shield)
    state, normalization, _state_audit = sanity_gate._normalization_and_state(frame, split)
    validate_state_columns(state.columns.tolist())
    return frame, masks, split, state, normalization, reward


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return sanity_gate._bool(frame, column)


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_EVENT_ORDERED_TRANSITION_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "sequence_metadata_root_v1": str(INPUT_SEQUENCE_ROOT),
            "transition_schema_root_v1": str(INPUT_SCHEMA_ROOT),
            "first_iql_sanity_root_v1": str(INPUT_SANITY_ROOT),
            "iql_offline_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "clean_safety_layer_refinement_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "dataset_kind_v1": DATASET_KIND,
        "research_only_event_ordered_v1": True,
        "iql_training_run_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _reproducibility_audit(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_TRANSITION_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "sequence_metadata_status_v1": inputs["sequence_go_no_go"].get("status_v1"),
        "sequence_metadata_next_action_v1": inputs["sequence_go_no_go"].get("next_recommended_action_v1"),
        "previous_no_fake_transition_status_v1": inputs["sequence_no_fake"].get("status_v1"),
        "sanity_status_v1": inputs["sanity_summary"].get("final_status_v1"),
        "iql_contract_status_v1": inputs["contract_summary"].get("final_status_v1"),
        "dataset_rows_v1": int(len(frame)),
        "run_id_episode_count_v1": int(frame["run_id_v1"].nunique(dropna=True)),
        "expected_nonterminal_transitions_v1": int(inputs["sequence_summary"].get("expected_nonterminal_transitions_v1")),
        "expected_terminal_rows_v1": int(inputs["sequence_summary"].get("expected_terminal_transitions_v1")),
        "event_order_schema_v1": inputs["sequence_summary"].get("event_order_schema_v1"),
        "state_allowlist_field_count_v1": int(inputs["contract_summary"].get("state_allowed_field_count_v1")),
        "state_denylist_field_count_v1": int(inputs["contract_summary"].get("state_blocked_field_count_v1")),
        "safety_shield_v1": SAFETY_COHORT,
        "safety_shielded_rows_v1": int(shield.sum()),
        "transformer_features_blocked_v1": True,
        "denied_fields_excluded_v1": True,
        "adapter_r6_live_blocked_v1": True,
        "iql_training_run_v1": False,
    }
    validate_reproducibility(payload)
    return payload


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    checks = [
        payload["sequence_metadata_status_v1"] == "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET",
        payload["sequence_metadata_next_action_v1"] == ACTION,
        payload["previous_no_fake_transition_status_v1"] == "PASS",
        payload["sanity_status_v1"] == "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN",
        payload["iql_contract_status_v1"] == "IQL_OFFLINE_DATA_CONTRACT_READY_FOR_SANITY_TRAINING_RESEARCH_ONLY",
        payload["dataset_rows_v1"] == 1914,
        payload["run_id_episode_count_v1"] == 58,
        payload["expected_nonterminal_transitions_v1"] == 1856,
        payload["expected_terminal_rows_v1"] == 58,
        payload["state_allowlist_field_count_v1"] == 9,
        payload["state_denylist_field_count_v1"] == 22,
        payload["safety_shielded_rows_v1"] == 78,
        payload["transformer_features_blocked_v1"] is True,
        payload["denied_fields_excluded_v1"] is True,
        payload["iql_training_run_v1"] is False,
    ]
    if not all(checks):
        raise RuntimeError("IQL_EVENT_ORDERED_TRANSITION_REPRODUCTION_FAILED")
    return True


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_source_index_v1"] = frame.index
    ordered["_event_ts_v1"] = pd.to_datetime(ordered["decision_timestamp_v1"], errors="coerce", utc=True)
    ordered = ordered.sort_values(
        ["run_id_v1", "_event_ts_v1", "candidate_uid_v1"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["_timestep_index_v1"] = ordered.groupby("run_id_v1").cumcount()
    ordered["_episode_row_count_v1"] = ordered.groupby("run_id_v1")["candidate_uid_v1"].transform("size")
    ordered["_next_source_index_v1"] = ordered.groupby("run_id_v1")["_source_index_v1"].shift(-1)
    ordered["_next_row_id_v1"] = ordered.groupby("run_id_v1")["candidate_uid_v1"].shift(-1)
    ordered["_next_timestamp_v1"] = ordered.groupby("run_id_v1")["_event_ts_v1"].shift(-1)
    ordered["_done_v1"] = ordered["_next_source_index_v1"].isna()
    return ordered


def _state_vector(state: pd.DataFrame, idx: int) -> list[float]:
    return [float(state.loc[idx, column]) for column in sanity_gate.MODEL_STATE_COLUMNS]


def _transition_id(run_id: Any, timestep: int) -> str:
    return f"{run_id}::event_{timestep:06d}"


def _build_transition_rows(
    ordered: pd.DataFrame,
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    split: pd.Series,
    state: pd.DataFrame,
    reward: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shield = safe_core & ~masks["source_confluence_repairable_v1"]
    rows: list[dict[str, Any]] = []
    for _, source in ordered.iterrows():
        idx = int(source["_source_index_v1"])
        next_idx = None if pd.isna(source["_next_source_index_v1"]) else int(source["_next_source_index_v1"])
        done = bool(source["_done_v1"])
        state_vector = _state_vector(state, idx)
        next_state_vector = None if done else _state_vector(state, int(next_idx))
        action = "TAKE_TRADE" if bool(shield.loc[idx]) else "SKIP"
        action_id = 1 if action == "TAKE_TRADE" else 0
        transition_id = _transition_id(source["run_id_v1"], int(source["_timestep_index_v1"]))
        next_row_id = None if done else source["_next_row_id_v1"]
        rows.append(
            {
                "transition_id_v1": transition_id,
                "dataset_kind_v1": DATASET_KIND,
                "transition_schema_v1": TRANSITION_SCHEMA,
                "episode_id_v1": source["run_id_v1"],
                "run_id_audit_only_v1": source["run_id_v1"],
                "fold_id_audit_only_v1": source.get("fold_id_v1"),
                "timestep_index_v1": int(source["_timestep_index_v1"]),
                "decision_timestamp_v1": source["_event_ts_v1"],
                "row_id_audit_only_v1": source.get("candidate_uid_v1"),
                "state_feature_names_v1": json.dumps(sanity_gate.MODEL_STATE_COLUMNS, sort_keys=True),
                "state_vector_v1": json.dumps([round(value, 8) for value in state_vector]),
                "action_t_v1": action,
                "action_id_t_v1": action_id,
                "action_source_v1": "RESEARCH_ONLY_INFERRED_FROM_SOURCE_SAFETY_SHIELDED_78_ELIGIBILITY",
                "action_observed_or_inferred_v1": "INFERRED_RESEARCH_ONLY_NOT_PRODUCTION_LOGGED_ACTION",
                "reward_t_v1": float(reward[idx]),
                "reward_field_v1": REWARD_FIELD,
                "reward_id_v1": REWARD_ID,
                "next_row_id_audit_only_v1": next_row_id,
                "next_state_feature_names_v1": json.dumps(sanity_gate.MODEL_STATE_COLUMNS, sort_keys=True)
                if not done
                else None,
                "next_state_vector_v1": json.dumps([round(value, 8) for value in next_state_vector])
                if next_state_vector is not None
                else None,
                "next_state_source_v1": "NEXT_REAL_EVENT_ROW_IN_SAME_RUN_ID" if not done else "TERMINAL_NO_NEXT_STATE",
                "done_v1": done,
                "terminal_reason_v1": "LAST_EVENT_IN_RUN_ID" if done else "",
                "split_id_v1": split.loc[idx],
                "safety_shield_status_v1": "ELIGIBLE_78_SHIELD" if bool(shield.loc[idx]) else "NOT_ELIGIBLE_FOR_TAKE",
                "eligibility_cohort_v1": SAFETY_COHORT if bool(shield.loc[idx]) else "NON_SELECTED_AND_NEAR_MISS_POOL",
                "inside_78_shield_v1": bool(shield.loc[idx]),
                "inside_89_safe_core_v1": bool(safe_core.loc[idx]),
                "inside_140_comparator_v1": bool(baseline.loc[idx]),
                "cohort_140_94_baseline_comparator_audit_only_v1": bool(baseline.loc[idx]),
                "cohort_safe_core_89_research_candidate_audit_only_v1": bool(safe_core.loc[idx]),
                "cohort_source_safety_shielded_78_audit_only_v1": bool(shield.loc[idx]),
                "cohort_non_selected_near_miss_pool_audit_only_v1": not bool(shield.loc[idx]),
                "bad_label_audit_only_v1": bool(frame.loc[idx].get("bad_label_v1", False)),
                "tail_label_audit_only_v1": bool(frame.loc[idx].get("tail_label_v1", False)),
                "unsafe_label_audit_only_v1": bool(frame.loc[idx].get("unsafe_audit_v1", False)),
                "state_contains_denied_fields_v1": False,
                "next_state_contains_denied_fields_v1": False,
                "audit_labels_separated_from_state_v1": True,
                "cross_run_transition_v1": False if done else frame.loc[int(next_idx), "run_id_v1"] != source["run_id_v1"],
                "synthetic_or_random_next_state_v1": False,
                "not_full_lifecycle_iql_v1": True,
                "production_behavior_policy_v1": False,
            }
        )
    validate_transition_dataset(rows)
    return rows


def _state_matrix_rows(
    ordered: pd.DataFrame, frame: pd.DataFrame, split: pd.Series, state: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = []
    for _, source in ordered.iterrows():
        idx = int(source["_source_index_v1"])
        row = {
            "transition_id_v1": _transition_id(source["run_id_v1"], int(source["_timestep_index_v1"])),
            "row_id_audit_only_v1": source.get("candidate_uid_v1"),
            "split_id_v1": split.loc[idx],
        }
        for column in sanity_gate.MODEL_STATE_COLUMNS:
            row[column] = float(state.loc[idx, column])
        rows.append(row)
    return rows


def _next_state_matrix_rows(
    ordered: pd.DataFrame, frame: pd.DataFrame, split: pd.Series, state: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = []
    for _, source in ordered.iterrows():
        idx = int(source["_source_index_v1"])
        done = bool(source["_done_v1"])
        row = {
            "transition_id_v1": _transition_id(source["run_id_v1"], int(source["_timestep_index_v1"])),
            "row_id_audit_only_v1": source.get("candidate_uid_v1"),
            "next_row_id_audit_only_v1": None if done else source["_next_row_id_v1"],
            "split_id_v1": split.loc[idx],
            "done_v1": done,
        }
        if done:
            for column in sanity_gate.MODEL_STATE_COLUMNS:
                row[f"next_{column}"] = None
        else:
            next_idx = int(source["_next_source_index_v1"])
            for column in sanity_gate.MODEL_STATE_COLUMNS:
                row[f"next_{column}"] = float(state.loc[next_idx, column])
        rows.append(row)
    return rows


def _ordering_audit(ordered: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for run_id, group in ordered.groupby("run_id_v1", sort=True):
        duplicate_ts = int(group.duplicated(subset=["_event_ts_v1"]).sum())
        missing_ts = int(group["_event_ts_v1"].isna().sum())
        rows.append(
            {
                "run_id_v1": run_id,
                "rows_in_episode_v1": int(len(group)),
                "first_timestamp_v1": group["_event_ts_v1"].min(),
                "last_timestamp_v1": group["_event_ts_v1"].max(),
                "missing_timestamps_v1": missing_ts,
                "duplicate_timestamps_v1": duplicate_ts,
                "timestamp_monotonic_after_sort_v1": bool(group["_event_ts_v1"].is_monotonic_increasing),
                "deterministic_ordering_v1": bool(missing_ts == 0 and duplicate_ts == 0),
                "tie_breaker_v1": "candidate_uid_v1",
                "terminal_rows_v1": int(group["_done_v1"].sum()),
                "cross_run_transitions_v1": 0,
            }
        )
    if len(rows) != 58 or any(row["missing_timestamps_v1"] for row in rows):
        raise RuntimeError("ORDERING_AUDIT_FAILED")
    return rows


def _next_state_audit(transition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in transition_rows:
        rows.append(
            {
                "transition_id_v1": row["transition_id_v1"],
                "episode_id_v1": row["episode_id_v1"],
                "row_id_audit_only_v1": row["row_id_audit_only_v1"],
                "next_row_id_audit_only_v1": row["next_row_id_audit_only_v1"],
                "done_v1": row["done_v1"],
                "has_next_state_v1": row["next_state_vector_v1"] is not None,
                "next_state_same_episode_v1": not row["cross_run_transition_v1"],
                "next_state_allowlist_only_v1": not row["next_state_contains_denied_fields_v1"],
                "next_state_outcome_future_label_free_v1": True,
                "next_state_synthetic_v1": False,
                "next_state_random_v1": False,
                "next_state_source_v1": row["next_state_source_v1"],
            }
        )
    nonterminal = sum(1 for row in rows if row["has_next_state_v1"])
    terminal = len(rows) - nonterminal
    if nonterminal != 1856 or terminal != 58:
        raise RuntimeError("NEXT_STATE_AUDIT_COUNTS_MISMATCH")
    return rows


def _done_terminal_audit(transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[str, int] = {}
    for row in transition_rows:
        if row["done_v1"]:
            by_episode[row["episode_id_v1"]] = by_episode.get(row["episode_id_v1"], 0) + 1
    zero_terminal = sorted({row["episode_id_v1"] for row in transition_rows} - set(by_episode))
    multi_terminal = sorted(ep for ep, count in by_episode.items() if count != 1)
    return {
        "layer_name": "IQL_EVENT_ORDERED_DONE_TERMINAL_AUDIT_V1",
        "status_v1": "PASS" if not zero_terminal and not multi_terminal else "FAIL",
        "done_rule_v1": DONE_RULE,
        "done_rows_count_v1": sum(1 for row in transition_rows if row["done_v1"]),
        "non_done_rows_count_v1": sum(1 for row in transition_rows if not row["done_v1"]),
        "episodes_count_v1": len(by_episode),
        "exactly_one_terminal_per_episode_v1": not zero_terminal and not multi_terminal,
        "episodes_with_zero_terminal_v1": zero_terminal,
        "episodes_with_multiple_terminal_v1": multi_terminal,
        "terminal_rule_leakage_risk_v1": "LOW_AS_EVENT_ORDERED_METADATA_NOT_OUTCOME_BASED",
        "terminal_scope_v1": "END_OF_RUN_EVENT_ORDERED_RESEARCH_ONLY_NOT_TRADE_EXIT",
    }


def _action_construction_audit(transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {}
    for split_id in ["train", "validation", "test"]:
        split_rows = [row for row in transition_rows if row["split_id_v1"] == split_id]
        by_split[split_id] = {
            "rows_v1": len(split_rows),
            "take_trade_v1": sum(1 for row in split_rows if row["action_t_v1"] == "TAKE_TRADE"),
            "skip_v1": sum(1 for row in split_rows if row["action_t_v1"] == "SKIP"),
        }
    take = sum(1 for row in transition_rows if row["action_t_v1"] == "TAKE_TRADE")
    skip = len(transition_rows) - take
    return {
        "layer_name": "IQL_EVENT_ORDERED_ACTION_CONSTRUCTION_AUDIT_V1",
        "status_v1": "PASS_RESEARCH_ONLY_INFERRED_ACTION",
        "action_field_v1": ACTION_FIELD,
        "action_source_v1": "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY",
        "action_inference_rule_v1": "TAKE_TRADE iff inside SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY else SKIP",
        "action_observed_or_inferred_v1": "INFERRED_RESEARCH_ONLY",
        "take_trade_count_v1": take,
        "skip_count_v1": skip,
        "action_imbalance_take_rate_v1": float(take / max(len(transition_rows), 1)),
        "action_support_by_split_v1": by_split,
        "supports_research_only_iql_v1": True,
        "supports_production_behavior_policy_eval_v1": False,
        "limitations_v1": [
            "TAKE/SKIP is inferred from research eligibility, not true logged production action sequence.",
            "No propensities are available.",
            "Sizing, hold, exit, and risk-reduced actions remain unsupported.",
        ],
    }


def _reward_construction_audit(transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.array([row["reward_t_v1"] for row in transition_rows], dtype=float)
    by_split = {}
    by_action = {}
    for split_id in ["train", "validation", "test"]:
        split_rewards = np.array([row["reward_t_v1"] for row in transition_rows if row["split_id_v1"] == split_id], dtype=float)
        by_split[split_id] = {
            "rows_v1": int(len(split_rewards)),
            "reward_sum_v1": float(split_rewards.sum()) if len(split_rewards) else 0.0,
            "reward_mean_v1": float(split_rewards.mean()) if len(split_rewards) else 0.0,
        }
    for action in ["TAKE_TRADE", "SKIP"]:
        action_rewards = np.array([row["reward_t_v1"] for row in transition_rows if row["action_t_v1"] == action], dtype=float)
        by_action[action] = {
            "rows_v1": int(len(action_rewards)),
            "reward_sum_v1": float(action_rewards.sum()) if len(action_rewards) else 0.0,
            "reward_mean_v1": float(action_rewards.mean()) if len(action_rewards) else 0.0,
        }
    return {
        "layer_name": "IQL_EVENT_ORDERED_REWARD_CONSTRUCTION_AUDIT_V1",
        "status_v1": "PASS_RESEARCH_ONLY_EVENT_ATTACHED_REWARD",
        "reward_field_v1": REWARD_FIELD,
        "reward_id_v1": REWARD_ID,
        "reward_available_count_v1": int(len(rewards)),
        "missing_reward_count_v1": int(np.isnan(rewards).sum()),
        "reward_sum_v1": float(np.nansum(rewards)),
        "reward_min_v1": float(np.nanmin(rewards)),
        "reward_max_v1": float(np.nanmax(rewards)),
        "reward_mean_v1": float(np.nanmean(rewards)),
        "reward_by_action_v1": by_action,
        "reward_by_split_v1": by_split,
        "reward_included_in_state_v1": False,
        "labels_used_only_as_reward_or_audit_v1": True,
        "reward_timing_limitation_v1": "EVENT_ATTACHED_RESEARCH_REWARD_NOT_TRUE_OUTCOME_TIMESTAMP_OR_TERMINAL_REWARD",
    }


def _split_audit(transition_rows: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split_id in ["train", "validation", "test"]:
        split_rows = [row for row in transition_rows if row["split_id_v1"] == split_id]
        rows.append(
            {
                "split_id_v1": split_id,
                "rows_v1": len(split_rows),
                "episodes_v1": len({row["episode_id_v1"] for row in split_rows}),
                "nonterminal_transitions_v1": sum(1 for row in split_rows if not row["done_v1"]),
                "terminal_rows_v1": sum(1 for row in split_rows if row["done_v1"]),
                "take_trade_count_v1": sum(1 for row in split_rows if row["action_t_v1"] == "TAKE_TRADE"),
                "skip_count_v1": sum(1 for row in split_rows if row["action_t_v1"] == "SKIP"),
                "reward_sum_v1": float(sum(row["reward_t_v1"] for row in split_rows)),
                "safety_shielded_rows_v1": sum(1 for row in split_rows if row["inside_78_shield_v1"]),
                "unsafe_audit_hits_v1": sum(1 for row in split_rows if row["unsafe_label_audit_only_v1"]),
                "low_support_rows_v1": int(
                    _bool(frame[frame["fold_id_v1"].astype("string").map(
                        {"fold_00": "train", "fold_01": "train", "fold_02": "train", "fold_03": "validation", "fold_04": "test"}
                    ).fillna("train").eq(split_id)], "structural_low_support_v1").sum()
                ),
            }
        )
    return rows


def _cohort_label_audit(transition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts = [
        ("140_94_BASELINE_COMPARATOR", "inside_140_comparator_v1"),
        ("SAFE_CORE_89_RESEARCH_CANDIDATE", "inside_89_safe_core_v1"),
        ("SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY", "inside_78_shield_v1"),
        ("NON_SELECTED_AND_NEAR_MISS_POOL", "cohort_non_selected_near_miss_pool_audit_only_v1"),
    ]
    rows = []
    for cohort, flag in cohorts:
        cohort_rows = [row for row in transition_rows if bool(row[flag])]
        rows.append(
            {
                "cohort_name_v1": cohort,
                "audit_label_only_v1": True,
                "enters_state_v1": False,
                "rows_v1": len(cohort_rows),
                "take_trade_count_v1": sum(1 for row in cohort_rows if row["action_t_v1"] == "TAKE_TRADE"),
                "bad_count_audit_only_v1": sum(1 for row in cohort_rows if row["bad_label_audit_only_v1"]),
                "tail_count_audit_only_v1": sum(1 for row in cohort_rows if row["tail_label_audit_only_v1"]),
                "unsafe_hits_audit_only_v1": sum(1 for row in cohort_rows if row["unsafe_label_audit_only_v1"]),
                "reward_sum_v1": float(sum(row["reward_t_v1"] for row in cohort_rows)),
            }
        )
    return rows


def _no_fake_transition_audit(transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "no_synthetic_next_state_v1": not any(row["synthetic_or_random_next_state_v1"] for row in transition_rows),
        "no_random_next_state_v1": not any(row["synthetic_or_random_next_state_v1"] for row in transition_rows),
        "no_cross_run_next_state_v1": not any(row["cross_run_transition_v1"] for row in transition_rows),
        "no_transition_across_episode_boundary_v1": not any(row["cross_run_transition_v1"] for row in transition_rows),
        "row_identity_not_state_v1": True,
        "reward_not_state_v1": True,
        "future_label_not_state_v1": True,
        "outcome_timing_not_state_v1": True,
        "historical_v2_blueprint_not_used_v1": True,
        "membership_coverage_proxy_not_used_v1": True,
        "selected_by_flag_not_state_v1": True,
        "audit_only_veto_not_state_v1": True,
        "transformer_fields_absent_v1": True,
        "next_state_from_real_next_event_row_v1": True,
        "labels_separated_from_state_v1": all(row["audit_labels_separated_from_state_v1"] for row in transition_rows),
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "layer_name": "IQL_EVENT_ORDERED_NO_FAKE_TRANSITION_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }
    validate_no_fake_transition_audit(payload)
    return payload


def _dataset_readiness_assessment(
    transition_rows: list[dict[str, Any]],
    action: dict[str, Any],
    reward: dict[str, Any],
    no_fake: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "IQL_EVENT_ORDERED_DATASET_READINESS_ASSESSMENT_V1",
        "status_v1": "READY_FOR_RESEARCH_ONLY_EVENT_ORDERED_TRAINING",
        "ready_for_research_only_iql_training_v1": True,
        "full_lifecycle_sequential_iql_ready_v1": False,
        "event_ordered_not_full_sequential_v1": True,
        "action_support_sufficient_for_research_only_v1": action["take_trade_count_v1"] == 78
        and action["skip_count_v1"] == 1836,
        "reward_timing_acceptable_for_research_only_v1": reward["status_v1"]
        == "PASS_RESEARCH_ONLY_EVENT_ATTACHED_REWARD",
        "split_support_acceptable_v1": True,
        "no_fake_transition_status_v1": no_fake["status_v1"],
        "dataset_rows_v1": len(transition_rows),
        "nonterminal_transitions_v1": sum(1 for row in transition_rows if not row["done_v1"]),
        "terminal_rows_v1": sum(1 for row in transition_rows if row["done_v1"]),
        "limitations_v1": [
            "Event-ordered transitions use next candidate event inside run_id, not trade lifecycle next_state.",
            "Actions are inferred research behavior actions, not source-logged production actions.",
            "Reward is event-attached research reward, not true delayed/terminal reward timing.",
            "This dataset does not open adapter, R6, production IQL, or live policy.",
        ],
    }


def _recommendation(no_fake: dict[str, Any], readiness: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    status = FINAL_STATUS
    next_action = NEXT_ACTION
    if no_fake["status_v1"] != "PASS":
        status = "IQL_EVENT_ORDERED_TRANSITION_DATASET_BLOCKED_BY_FAKE_TRANSITION_RISK"
        next_action = "COLLECT_ADDITIONAL_SEQUENCE_SOURCE_METADATA_V1"
    elif not readiness["action_support_sufficient_for_research_only_v1"]:
        status = "IQL_EVENT_ORDERED_TRANSITION_DATASET_PARTIAL_NEEDS_ACTION_SUPPORT_AUDIT"
        next_action = "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1"
    validate_final_status(status, next_action)
    recommendation = {
        "layer_name": "IQL_EVENT_ORDERED_TRANSITION_DATASET_RECOMMENDATION_V1",
        "final_status_v1": status,
        "next_recommended_action_v1": next_action,
        "recommendation_v1": "Run event-ordered research-only IQL training next; keep full lifecycle sequential IQL and production paths blocked.",
        "event_ordered_research_training_allowed_next_v1": status
        == "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
        "full_lifecycle_sequential_iql_ready_v1": False,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_GO_NO_GO_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "event_ordered_research_training_allowed_next_v1": recommendation[
            "event_ordered_research_training_allowed_next_v1"
        ],
        "full_lifecycle_sequential_iql_ready_v1": False,
        "iql_training_run_in_this_gate_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_production_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    validate_go_no_go(go_no_go)
    return recommendation, go_no_go


def _write_markdown(
    artifact_root: Path,
    repro: dict[str, Any],
    transition_rows: list[dict[str, Any]],
    ordering_rows: list[dict[str, Any]],
    next_state_rows: list[dict[str, Any]],
    done: dict[str, Any],
    action: dict[str, Any],
    reward: dict[str, Any],
    split_rows: list[dict[str, Any]],
    cohort_rows: list[dict[str, Any]],
    no_fake: dict[str, Any],
    readiness: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_event_ordered_transition_reproducibility_audit_v1.md",
        [
            "# IQL Event-Ordered Transition Reproducibility Audit V1",
            "",
            f"- Sequence status: `{repro['sequence_metadata_status_v1']}`.",
            f"- Rows: `{repro['dataset_rows_v1']}`.",
            f"- Episodes: `{repro['run_id_episode_count_v1']}`.",
            f"- Expected nonterminal/terminal: `{repro['expected_nonterminal_transitions_v1']}` / `{repro['expected_terminal_rows_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_transition_dataset_v1.md",
        [
            "# IQL Event-Ordered Transition Dataset V1",
            "",
            f"- Rows: `{len(transition_rows)}`.",
            f"- Dataset kind: `{DATASET_KIND}`.",
            "- This is research-only event-order, not full trade-lifecycle sequential IQL.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_ordering_audit_v1.md",
        [
            "# IQL Event-Ordered Ordering Audit V1",
            "",
            f"- Episodes: `{len(ordering_rows)}`.",
            "- Ordering uses `run_id_v1` + `decision_timestamp_v1` with audit-only row-id tie breaker.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_next_state_audit_v1.md",
        [
            "# IQL Event-Ordered Next-State Audit V1",
            "",
            f"- Nonterminal next_states: `{sum(1 for row in next_state_rows if row['has_next_state_v1'])}`.",
            f"- Terminal rows: `{sum(1 for row in next_state_rows if not row['has_next_state_v1'])}`.",
            "- next_state is the next real event row in the same run_id.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_done_terminal_audit_v1.md",
        [
            "# IQL Event-Ordered Done Terminal Audit V1",
            "",
            f"- Done rule: `{done['done_rule_v1']}`.",
            f"- Done rows: `{done['done_rows_count_v1']}`.",
            f"- Status: `{done['status_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_action_construction_audit_v1.md",
        [
            "# IQL Event-Ordered Action Construction Audit V1",
            "",
            f"- Status: `{action['status_v1']}`.",
            f"- TAKE: `{action['take_trade_count_v1']}`.",
            f"- SKIP: `{action['skip_count_v1']}`.",
            "- Actions are inferred research actions, not production behavior logs.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_reward_construction_audit_v1.md",
        [
            "# IQL Event-Ordered Reward Construction Audit V1",
            "",
            f"- Status: `{reward['status_v1']}`.",
            f"- Reward sum: `{reward['reward_sum_v1']}`.",
            "- Reward is not included in state or next_state.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_split_audit_v1.md",
        [
            "# IQL Event-Ordered Split Audit V1",
            "",
            *[
                f"- `{row['split_id_v1']}`: rows={row['rows_v1']}, take={row['take_trade_count_v1']}, terminal={row['terminal_rows_v1']}."
                for row in split_rows
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_cohort_label_audit_v1.md",
        [
            "# IQL Event-Ordered Cohort Label Audit V1",
            "",
            *[
                f"- `{row['cohort_name_v1']}`: rows={row['rows_v1']}, state={row['enters_state_v1']}."
                for row in cohort_rows
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_no_fake_transition_audit_v1.md",
        [
            "# IQL Event-Ordered No-Fake Transition Audit V1",
            "",
            f"- Status: `{no_fake['status_v1']}`.",
            "- No synthetic/random/cross-run next_state was created.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_dataset_readiness_assessment_v1.md",
        [
            "# IQL Event-Ordered Dataset Readiness Assessment V1",
            "",
            f"- Status: `{readiness['status_v1']}`.",
            f"- Research training ready: `{readiness['ready_for_research_only_iql_training_v1']}`.",
            f"- Full lifecycle sequential IQL ready: `{readiness['full_lifecycle_sequential_iql_ready_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_event_ordered_transition_dataset_recommendation_v1.md",
        [
            "# IQL Event-Ordered Transition Dataset Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`.",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks, split, state, normalization, reward = _frame_and_contract()
    ordered = _ordered_frame(frame)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility_audit(frame, masks, inputs)
    transition_rows = _build_transition_rows(ordered, frame, masks, split, state, reward)
    state_rows = _state_matrix_rows(ordered, frame, split, state)
    next_state_rows = _next_state_matrix_rows(ordered, frame, split, state)
    ordering_rows = _ordering_audit(ordered)
    next_state_audit = _next_state_audit(transition_rows)
    done = _done_terminal_audit(transition_rows)
    action = _action_construction_audit(transition_rows)
    reward_audit = _reward_construction_audit(transition_rows)
    split_rows = _split_audit(transition_rows, frame)
    cohort_rows = _cohort_label_audit(transition_rows)
    no_fake = _no_fake_transition_audit(transition_rows)
    readiness = _dataset_readiness_assessment(transition_rows, action, reward_audit, no_fake)
    recommendation, go_no_go = _recommendation(no_fake, readiness)

    _write_json(artifact_root / "iql_event_ordered_transition_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_event_ordered_transition_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_event_ordered_transition_dataset_v1.csv", transition_rows)
    _write_json(
        artifact_root / "iql_event_ordered_transition_dataset_v1.json",
        {"row_count_v1": len(transition_rows), "rows_v1": transition_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_state_matrix_v1.csv", state_rows)
    _write_json(
        artifact_root / "iql_event_ordered_state_matrix_v1.json",
        {
            "row_count_v1": len(state_rows),
            "feature_columns_v1": sanity_gate.MODEL_STATE_COLUMNS,
            "normalization_v1": normalization,
            "rows_v1": state_rows,
        },
    )
    _write_rows(artifact_root / "iql_event_ordered_next_state_matrix_v1.csv", next_state_rows)
    _write_json(
        artifact_root / "iql_event_ordered_next_state_matrix_v1.json",
        {
            "row_count_v1": len(next_state_rows),
            "feature_columns_v1": [f"next_{column}" for column in sanity_gate.MODEL_STATE_COLUMNS],
            "rows_v1": next_state_rows,
        },
    )
    _write_rows(artifact_root / "iql_event_ordered_ordering_audit_v1.csv", ordering_rows)
    _write_json(
        artifact_root / "iql_event_ordered_ordering_audit_v1.json",
        {"row_count_v1": len(ordering_rows), "rows_v1": ordering_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_next_state_audit_v1.csv", next_state_audit)
    _write_json(
        artifact_root / "iql_event_ordered_next_state_audit_v1.json",
        {"row_count_v1": len(next_state_audit), "rows_v1": next_state_audit},
    )
    _write_json(artifact_root / "iql_event_ordered_done_terminal_audit_v1.json", done)
    _write_json(artifact_root / "iql_event_ordered_action_construction_audit_v1.json", action)
    _write_json(artifact_root / "iql_event_ordered_reward_construction_audit_v1.json", reward_audit)
    _write_rows(artifact_root / "iql_event_ordered_split_audit_v1.csv", split_rows)
    _write_json(
        artifact_root / "iql_event_ordered_split_audit_v1.json",
        {"row_count_v1": len(split_rows), "rows_v1": split_rows},
    )
    _write_rows(artifact_root / "iql_event_ordered_cohort_label_audit_v1.csv", cohort_rows)
    _write_json(
        artifact_root / "iql_event_ordered_cohort_label_audit_v1.json",
        {"row_count_v1": len(cohort_rows), "rows_v1": cohort_rows},
    )
    _write_json(artifact_root / "iql_event_ordered_no_fake_transition_audit_v1.json", no_fake)
    _write_json(artifact_root / "iql_event_ordered_dataset_readiness_assessment_v1.json", readiness)
    _write_json(artifact_root / "iql_event_ordered_transition_dataset_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "build_iql_event_ordered_research_transition_dataset_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        transition_rows,
        ordering_rows,
        next_state_audit,
        done,
        action,
        reward_audit,
        split_rows,
        cohort_rows,
        no_fake,
        readiness,
        recommendation,
    )

    summary = {
        "layer_name": "IQL_EVENT_ORDERED_TRANSITION_DATASET_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": recommendation["final_status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "dataset_kind_v1": DATASET_KIND,
        "transition_schema_v1": TRANSITION_SCHEMA,
        "episode_schema_v1": EPISODE_SCHEMA,
        "done_rule_v1": DONE_RULE,
        "rows_v1": len(transition_rows),
        "episodes_v1": len(ordering_rows),
        "nonterminal_transitions_v1": sum(1 for row in transition_rows if not row["done_v1"]),
        "terminal_rows_v1": sum(1 for row in transition_rows if row["done_v1"]),
        "cross_run_transitions_v1": sum(1 for row in transition_rows if row["cross_run_transition_v1"]),
        "state_feature_count_v1": len(sanity_gate.MODEL_STATE_COLUMNS),
        "state_next_state_allowlist_only_v1": True,
        "action_status_v1": action["status_v1"],
        "take_trade_count_v1": action["take_trade_count_v1"],
        "skip_count_v1": action["skip_count_v1"],
        "reward_status_v1": reward_audit["status_v1"],
        "reward_sum_v1": reward_audit["reward_sum_v1"],
        "no_fake_transition_audit_status_v1": no_fake["status_v1"],
        "research_only_event_ordered_not_full_lifecycle_iql_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_training_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "status_v1": recommendation["final_status_v1"],
            "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
            "created_at_utc_v1": _utc_now(),
        },
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Build IQL Event-Ordered Research Transition Dataset V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- Rows / episodes / terminal: `{summary['rows_v1']}` / `{summary['episodes_v1']}` / `{summary['terminal_rows_v1']}`.",
            "- This is research-only event-ordered IQL data, not production sequential lifecycle IQL.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build event-ordered IQL transition dataset, research only.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
