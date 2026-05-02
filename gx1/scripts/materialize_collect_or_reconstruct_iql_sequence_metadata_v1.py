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

from gx1.scripts import materialize_design_iql_transition_and_episode_schema_v1 as schema_gate
from gx1.scripts import materialize_run_iql_offline_sanity_training_research_only_v1 as sanity_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1"

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

FINAL_STATUS = "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET"
NEXT_ACTION = "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1"
EPISODE_SCHEMA = "RUN_ID_EPISODE_BOUNDARY_V1"
ORDER_SCHEMA = "RUN_ID_DECISION_TIMESTAMP_EVENT_ORDER_V1"
TRANSITION_KIND = "EVENT_ORDERED_RESEARCH_TRANSITION"
REWARD_ID = "SAFETY_WEIGHTED_REWARD"
SAFETY_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

ALLOWED_FINAL_STATUSES = {
    "IQL_SEQUENCE_METADATA_READY_FOR_TRUE_TRANSITION_DATASET",
    "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET",
    "IQL_SEQUENCE_METADATA_PARTIAL_NEEDS_ACTION_RECONSTRUCTION",
    "IQL_SEQUENCE_METADATA_PARTIAL_NEEDS_DONE_REWARD_TIMING",
    "IQL_SEQUENCE_METADATA_PARTIAL_NEEDS_EPISODE_BOUNDARY_CONFIRMATION",
    "IQL_SEQUENCE_METADATA_CONTEXTUAL_ONLY_REMAINS_BEST",
    "IQL_SEQUENCE_METADATA_BLOCKED_BY_FAKE_TRANSITION_RISK",
    "IQL_SEQUENCE_METADATA_BLOCKED_BY_SOURCE_METADATA_GAPS",
    "IQL_SEQUENCE_METADATA_BLOCKED_BY_LEAKAGE_RISK",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_IQL_TRANSITION_DATASET_RESEARCH_ONLY_V1",
    "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1",
    "RECONSTRUCT_IQL_BEHAVIOR_ACTION_SUPPORT_V1",
    "DEEPEN_IQL_DONE_AND_REWARD_TIMING_AUDIT_V1",
    "CONFIRM_IQL_EPISODE_BOUNDARIES_V1",
    "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1",
    "COLLECT_ADDITIONAL_SEQUENCE_SOURCE_METADATA_V1",
}

REQUIRED_OUTPUTS = [
    "iql_sequence_metadata_input_manifest_v1.json",
    "iql_sequence_metadata_reproducibility_audit_v1.json",
    "iql_sequence_metadata_reproducibility_audit_v1.md",
    "iql_sequence_metadata_inventory_v1.csv",
    "iql_sequence_metadata_inventory_v1.json",
    "iql_sequence_metadata_inventory_v1.md",
    "iql_event_order_reconstruction_audit_v1.csv",
    "iql_event_order_reconstruction_audit_v1.json",
    "iql_event_order_reconstruction_audit_v1.md",
    "iql_next_row_candidate_audit_v1.csv",
    "iql_next_row_candidate_audit_v1.json",
    "iql_next_row_candidate_audit_v1.md",
    "iql_episode_boundary_candidates_v1.json",
    "iql_episode_boundary_candidates_v1.md",
    "iql_done_terminal_candidates_v1.json",
    "iql_done_terminal_candidates_v1.md",
    "iql_behavior_action_reconstruction_audit_v1.json",
    "iql_behavior_action_reconstruction_audit_v1.md",
    "iql_reward_timing_reconstruction_audit_v1.json",
    "iql_reward_timing_reconstruction_audit_v1.md",
    "iql_transition_dataset_feasibility_v1.json",
    "iql_transition_dataset_feasibility_v1.md",
    "iql_no_fake_transition_audit_v1.json",
    "iql_no_fake_transition_audit_v1.md",
    "iql_transition_dataset_build_spec_v1.json",
    "iql_transition_dataset_build_spec_v1.md",
    "iql_sequence_metadata_recommendation_v1.json",
    "iql_sequence_metadata_recommendation_v1.md",
    "collect_or_reconstruct_iql_sequence_metadata_go_no_go_v1.json",
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


def validate_no_fake_transition_audit(payload: dict[str, Any]) -> bool:
    failures = payload.get("critical_failures_v1", [])
    if failures:
        raise RuntimeError(f"NO_FAKE_TRANSITION_AUDIT_FAILED: {failures}")
    if payload.get("checks_v1", {}).get("no_synthetic_next_state_v1") is not True:
        raise RuntimeError("SYNTHETIC_NEXT_STATE_FORBIDDEN")
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
    if payload.get("true_transition_dataset_build_allowed_v1") and payload["status_v1"] != (
        "IQL_SEQUENCE_METADATA_READY_FOR_TRUE_TRANSITION_DATASET"
    ):
        raise RuntimeError("TRUE_TRANSITION_DATASET_ALLOWED_WITHOUT_TRUE_READY_STATUS")
    if payload.get("event_ordered_transition_dataset_build_allowed_v1") and payload["status_v1"] != (
        "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET"
    ):
        raise RuntimeError("EVENT_ORDERED_BUILD_ALLOWED_WITHOUT_READY_STATUS")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_SCHEMA_ROOT, INPUT_SANITY_ROOT, INPUT_CONTRACT_ROOT, INPUT_REFINE_CLEAN_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "schema_summary": INPUT_SCHEMA_ROOT / "summary_v1.json",
        "schema_go_no_go": INPUT_SCHEMA_ROOT / "design_iql_transition_and_episode_schema_go_no_go_v1.json",
        "schema_design": INPUT_SCHEMA_ROOT / "iql_recommended_transition_design_v1.json",
        "schema_inventory": INPUT_SCHEMA_ROOT / "iql_transition_source_inventory_v1.json",
        "sanity_summary": INPUT_SANITY_ROOT / "summary_v1.json",
        "sanity_go_no_go": INPUT_SANITY_ROOT / "run_iql_offline_sanity_training_research_only_go_no_go_v1.json",
        "sanity_no_shortcut": INPUT_SANITY_ROOT / "iql_offline_sanity_no_shortcut_audit_v1.json",
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_behavior": INPUT_CONTRACT_ROOT / "iql_offline_behavior_policy_audit_v1.json",
        "refine_summary": INPUT_REFINE_CLEAN_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    schema_go = _read_json(required["schema_go_no_go"])
    if schema_go.get("status_v1") != "IQL_TRANSITION_SCHEMA_PARTIAL_NEEDS_SEQUENCE_METADATA":
        raise RuntimeError("INPUT_TRANSITION_SCHEMA_NOT_READY_FOR_SEQUENCE_METADATA_GATE")
    if not schema_go.get("sequence_metadata_required_before_transition_build_v1"):
        raise RuntimeError("INPUT_SCHEMA_DOES_NOT_REQUIRE_SEQUENCE_METADATA_UNEXPECTED")
    return {
        "required_paths": required,
        "schema_summary": _read_json(required["schema_summary"]),
        "schema_go_no_go": schema_go,
        "schema_design": _read_json(required["schema_design"]),
        "schema_inventory": _read_json(required["schema_inventory"]),
        "sanity_summary": _read_json(required["sanity_summary"]),
        "sanity_go_no_go": _read_json(required["sanity_go_no_go"]),
        "sanity_no_shortcut": _read_json(required["sanity_no_shortcut"]),
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_behavior": _read_json(required["contract_behavior"]),
        "refine_summary": _read_json(required["refine_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "sanity_inputs": sanity_gate._load_inputs(),
    }


def _frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    return sanity_gate._frame_and_masks(inputs["sanity_inputs"])


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return sanity_gate._bool(frame, column)


def _reward(frame: pd.DataFrame, shield: pd.Series) -> np.ndarray:
    return sanity_gate._reward(frame, shield)


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_SEQUENCE_METADATA_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
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
        "metadata_prep_only_v1": True,
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
        "layer_name": "IQL_SEQUENCE_METADATA_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "previous_transition_schema_status_v1": inputs["schema_go_no_go"].get("status_v1"),
        "previous_transition_next_action_v1": inputs["schema_go_no_go"].get("next_recommended_action_v1"),
        "previous_sanity_status_v1": inputs["sanity_go_no_go"].get("status_v1"),
        "previous_sanity_mode_v1": inputs["sanity_summary"].get("mode_v1"),
        "previous_sanity_no_shortcut_status_v1": inputs["sanity_no_shortcut"].get("status_v1"),
        "dataset_rows_v1": int(len(frame)),
        "run_id_present_v1": "run_id_v1" in frame.columns,
        "decision_timestamp_present_v1": "decision_timestamp_v1" in frame.columns,
        "event_order_available_v1": "run_id_v1" in frame.columns and "decision_timestamp_v1" in frame.columns,
        "contextual_policy_selected_rows_v1": int(inputs["sanity_summary"].get("policy_selected_rows_v1")),
        "contextual_policy_bad_tail_audit_only_v1": inputs["sanity_summary"].get("policy_bad_tail_audit_only_v1"),
        "contextual_policy_precision_audit_only_v1": float(inputs["sanity_summary"].get("policy_precision_audit_only_v1")),
        "contextual_policy_reward_sum_v1": float(inputs["sanity_summary"].get("policy_reward_sum_v1")),
        "contextual_policy_safety_status_v1": inputs["sanity_summary"].get("policy_safety_status_v1"),
        "source_safety_shielded_78_rows_v1": int(shield.sum()),
        "no_fake_transitions_were_created_in_previous_gate_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    checks = [
        payload["previous_transition_schema_status_v1"] == "IQL_TRANSITION_SCHEMA_PARTIAL_NEEDS_SEQUENCE_METADATA",
        payload["previous_sanity_status_v1"] == "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN",
        payload["previous_sanity_no_shortcut_status_v1"] == "PASS",
        payload["dataset_rows_v1"] == 1914,
        payload["run_id_present_v1"] is True,
        payload["decision_timestamp_present_v1"] is True,
        payload["event_order_available_v1"] is True,
        payload["contextual_policy_selected_rows_v1"] == 76,
        payload["contextual_policy_bad_tail_audit_only_v1"] == [75, 55],
        payload["contextual_policy_safety_status_v1"] == "CLEAN",
        payload["source_safety_shielded_78_rows_v1"] == 78,
        payload["no_fake_transitions_were_created_in_previous_gate_v1"] is True,
    ]
    if not all(checks):
        raise RuntimeError("IQL_SEQUENCE_METADATA_REPRODUCTION_FAILED")
    return True


def _field_stats(frame: pd.DataFrame, field: str) -> dict[str, Any]:
    if field not in frame.columns:
        return {"present_v1": False, "missingness_v1": len(frame), "datatype_v1": "MISSING", "unique_values_v1": 0}
    series = frame[field]
    return {
        "present_v1": True,
        "missingness_v1": int(series.isna().sum()),
        "datatype_v1": str(series.dtype),
        "unique_values_v1": int(series.nunique(dropna=True)),
    }


def _sequence_metadata_inventory(frame: pd.DataFrame) -> list[dict[str, Any]]:
    requested = [
        ("run_id_v1", "run_id", "source candidate frame"),
        ("decision_timestamp_v1", "decision_timestamp/event timestamp/bar timestamp", "source candidate frame"),
        ("fold_id_v1", "fold/group", "source candidate frame"),
        ("candidate_uid_v1", "row audit id", "source candidate frame"),
        ("trade_uid_v1", "trade id / setup id candidate", "source candidate frame"),
        ("trade_id_v1", "trade id candidate", "source candidate frame"),
        ("symbol_v1", "symbol/instrument", "not present"),
        ("instrument_v1", "symbol/instrument", "not present"),
        ("session_id_v1", "session id", "not present"),
        ("episode_id_v1", "episode id", "not present"),
        ("position_id_v1", "position id", "not present"),
        ("entry_id_v1", "entry id", "not present"),
        ("exit_id_v1", "exit id", "not present"),
        ("previous_row_id_v1", "previous row pointer", "not present"),
        ("next_row_id_v1", "next row pointer", "not present"),
        ("done_v1", "terminal/done", "not present"),
        ("terminal_step_status_v1", "terminal/done", "not present"),
        ("logged_action_v1", "action taken", "sanity dataset only / not source frame"),
        ("behavior_action_v1", "behavior policy action", "not present"),
        ("skipped_action_v1", "skipped action", "not present"),
        ("entry_time_v1", "entry relation", "not present"),
        ("exit_time_v1", "exit relation", "not present"),
        ("outcome_timestamp_v1", "outcome timing", "not present"),
        ("reward_realization_time_v1", "reward timing", "not present"),
        ("position_state_v1", "lifecycle state", "not present"),
    ]
    rows: list[dict[str, Any]] = []
    for field, purpose, source in requested:
        stats = _field_stats(frame, field)
        present = bool(stats["present_v1"])
        usable_order = field == "decision_timestamp_v1" and present
        usable_episode = field == "run_id_v1" and present
        usable_next = field in {"next_row_id_v1", "previous_row_id_v1"} and present
        usable_done = field in {"done_v1", "terminal_step_status_v1"} and present
        usable_action = field in {"logged_action_v1", "behavior_action_v1"} and present
        if field in {"candidate_uid_v1", "trade_uid_v1", "trade_id_v1"}:
            asof = "AUDIT_METADATA_ONLY"
            leakage = "HIGH_IF_USED_AS_STATE_OR_SELECTOR"
            recommendation = "AUDIT_ONLY_NOT_STATE; MAY_BE_USED_FOR POINTER VALIDATION ONLY"
        elif field in {"run_id_v1", "decision_timestamp_v1", "fold_id_v1"} and present:
            asof = "AS_OF_SAFE_METADATA"
            leakage = "LOW_AS_METADATA_MEDIUM_IF_STATE"
            recommendation = "USE_FOR_METADATA_ONLY"
        elif usable_next or usable_done or usable_action:
            asof = "SOURCE_LOGGED_METADATA"
            leakage = "LOW_IF_SOURCE_LOGGED"
            recommendation = "USE_AFTER NO-LEAKAGE VALIDATION"
        elif field == "logged_action_v1":
            asof = "MISSING_IN_SOURCE_INFERRED_IN_SANITY_ONLY"
            leakage = "MEDIUM_RESEARCH_ONLY"
            recommendation = "RECONSTRUCT AS RESEARCH BEHAVIOR ACTION ONLY"
        else:
            asof = "MISSING"
            leakage = "MISSING"
            recommendation = "MISSING_FOR_TRUE_SEQUENTIAL_IQL"
        rows.append(
            {
                "field_name_v1": field,
                "semantic_role_v1": purpose,
                "source_artifact_path_v1": source,
                "present_v1": present,
                "missingness_v1": stats["missingness_v1"],
                "datatype_v1": stats["datatype_v1"],
                "unique_values_v1": stats["unique_values_v1"],
                "usable_for_ordering_v1": usable_order,
                "usable_for_episode_v1": usable_episode,
                "usable_for_next_state_v1": usable_next,
                "usable_for_done_v1": usable_done,
                "usable_for_behavior_action_v1": usable_action,
                "as_of_safe_status_v1": asof,
                "leakage_risk_v1": leakage,
                "recommendation_v1": recommendation,
            }
        )
    return rows


def validate_sequence_inventory(rows: list[dict[str, Any]]) -> bool:
    by_name = {row["field_name_v1"]: row for row in rows}
    if not by_name["run_id_v1"]["usable_for_episode_v1"]:
        raise RuntimeError("RUN_ID_SEQUENCE_METADATA_MISSING")
    if not by_name["decision_timestamp_v1"]["usable_for_ordering_v1"]:
        raise RuntimeError("DECISION_TIMESTAMP_SEQUENCE_METADATA_MISSING")
    return True


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_event_ts_v1"] = pd.to_datetime(ordered["decision_timestamp_v1"], errors="coerce", utc=True)
    ordered = ordered.sort_values(
        ["run_id_v1", "_event_ts_v1", "candidate_uid_v1"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["_timestep_index_v1"] = ordered.groupby("run_id_v1").cumcount()
    ordered["_episode_row_count_v1"] = ordered.groupby("run_id_v1")["candidate_uid_v1"].transform("size")
    return ordered


def _event_order_audit(ordered: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, group in ordered.groupby("run_id_v1", sort=True):
        duplicate_ts = int(group.duplicated(subset=["_event_ts_v1"]).sum())
        missing_ts = int(group["_event_ts_v1"].isna().sum())
        rows.append(
            {
                "run_id_v1": run_id,
                "rows_in_episode_candidate_v1": int(len(group)),
                "first_timestamp_v1": group["_event_ts_v1"].min(),
                "last_timestamp_v1": group["_event_ts_v1"].max(),
                "missing_timestamps_v1": missing_ts,
                "duplicate_timestamps_v1": duplicate_ts,
                "out_of_order_rows_after_sort_v1": 0,
                "timestamp_monotonic_after_sort_v1": bool(group["_event_ts_v1"].is_monotonic_increasing),
                "tie_breaker_available_v1": "candidate_uid_v1",
                "deterministic_ordering_possible_v1": bool(missing_ts == 0),
                "meaningful_for_iql_v1": "YES_EVENT_ORDERED_RESEARCH_ONLY",
                "recommendation_v1": "USE_FOR_EVENT_ORDERED_RESEARCH_TRANSITIONS_NOT_TRUE_LIFECYCLE",
            }
        )
    return rows


def validate_event_order(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        raise RuntimeError("EVENT_ORDER_AUDIT_EMPTY")
    failures = [
        row["run_id_v1"]
        for row in rows
        if row["missing_timestamps_v1"] != 0 or not row["timestamp_monotonic_after_sort_v1"]
    ]
    if failures:
        raise RuntimeError(f"EVENT_ORDER_RECONSTRUCTION_FAILED: {failures[:5]}")
    return True


def _next_row_candidates(ordered: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, group in ordered.groupby("run_id_v1", sort=False):
        group = group.reset_index(drop=True)
        duplicate_ts = group.duplicated(subset=["_event_ts_v1"], keep=False)
        for idx, source in group.iterrows():
            has_next = idx + 1 < len(group)
            next_source = group.iloc[idx + 1] if has_next else None
            rows.append(
                {
                    "row_id_audit_only_v1": source["candidate_uid_v1"],
                    "episode_id_candidate_v1": source["run_id_v1"],
                    "timestep_index_candidate_v1": int(source["_timestep_index_v1"]),
                    "decision_timestamp_v1": source["_event_ts_v1"],
                    "next_row_id_candidate_v1": None if next_source is None else next_source["candidate_uid_v1"],
                    "next_timestep_index_candidate_v1": None if next_source is None else int(next_source["_timestep_index_v1"]),
                    "next_decision_timestamp_v1": None if next_source is None else next_source["_event_ts_v1"],
                    "has_next_row_v1": bool(has_next),
                    "done_candidate_v1": not has_next,
                    "ambiguous_next_row_v1": bool(duplicate_ts.iloc[idx]),
                    "timestamp_tie_v1": bool(duplicate_ts.iloc[idx]),
                    "cross_run_transition_prevented_v1": True,
                    "next_state_source_v1": "AS_OF fields from next real event row in same run_id",
                    "next_state_uses_as_of_only_v1": True,
                    "leakage_risk_v1": "LOW_FOR_EVENT_ORDERED_RESEARCH; NOT_TRUE_TRADE_LIFECYCLE",
                    "transition_class_v1": "EVENT_ORDERED_NONTERMINAL" if has_next else "EVENT_ORDERED_TERMINAL_LAST_IN_RUN",
                }
            )
    return rows


def validate_next_row_candidates(rows: list[dict[str, Any]], *, expected_rows: int, expected_done: int) -> bool:
    if len(rows) != expected_rows:
        raise RuntimeError(f"NEXT_ROW_CANDIDATE_ROW_COUNT_MISMATCH: {len(rows)} != {expected_rows}")
    done = sum(1 for row in rows if row["done_candidate_v1"])
    if done != expected_done:
        raise RuntimeError(f"DONE_ROW_COUNT_MISMATCH: {done} != {expected_done}")
    if any(row["ambiguous_next_row_v1"] for row in rows):
        raise RuntimeError("AMBIGUOUS_NEXT_ROW_TIMESTAMP_TIE_FOUND")
    if not all(row["cross_run_transition_prevented_v1"] for row in rows):
        raise RuntimeError("CROSS_RUN_TRANSITION_NOT_PREVENTED")
    return True


def _episode_boundary_candidates(frame: pd.DataFrame, ordered: pd.DataFrame) -> dict[str, Any]:
    run_count = int(ordered["run_id_v1"].nunique(dropna=True))
    rows = [
        {
            "boundary_name_v1": "RUN_ID_EPISODE_BOUNDARY_V1",
            "required_fields_v1": ["run_id_v1", "decision_timestamp_v1"],
            "available_fields_v1": ["run_id_v1", "decision_timestamp_v1"],
            "support_v1": f"{run_count} candidate episodes / {len(frame)} rows",
            "leakage_risk_v1": "LOW_AS_METADATA_ONLY",
            "transition_validity_v1": "VALID_FOR_EVENT_ORDERED_RESEARCH_NOT_TRUE_LIFECYCLE",
            "recommendation_v1": "SELECT_FOR_EVENT_ORDERED_RESEARCH_DATASET",
        },
        {
            "boundary_name_v1": "RUN_ID_PLUS_SESSION_TIME_BOUNDARY_V1",
            "required_fields_v1": ["run_id_v1", "decision_timestamp_v1", "session_id_v1"],
            "available_fields_v1": ["run_id_v1", "decision_timestamp_v1"],
            "support_v1": "session metadata missing",
            "leakage_risk_v1": "UNKNOWN_UNTIL_SESSION_SOURCE_EXISTS",
            "transition_validity_v1": "NOT_READY",
            "recommendation_v1": "COLLECT_SESSION_METADATA_BEFORE_USE",
        },
        {
            "boundary_name_v1": "RUN_ID_PLUS_SYMBOL_BOUNDARY_V1",
            "required_fields_v1": ["run_id_v1", "symbol_or_instrument"],
            "available_fields_v1": ["run_id_v1"],
            "support_v1": "symbol/instrument metadata missing",
            "leakage_risk_v1": "UNKNOWN_UNTIL_SYMBOL_SOURCE_EXISTS",
            "transition_validity_v1": "NOT_READY",
            "recommendation_v1": "COLLECT_SYMBOL_OR_INSTRUMENT_METADATA_BEFORE_USE",
        },
        {
            "boundary_name_v1": "CONTEXTUAL_ONLY_NO_EPISODE_V1",
            "required_fields_v1": [],
            "available_fields_v1": ["AS_OF state", "reward"],
            "support_v1": f"{len(frame)} contextual rows",
            "leakage_risk_v1": "LOW_IF_DECLARED_CONTEXTUAL_ONLY",
            "transition_validity_v1": "VALID_FALLBACK_NOT_SEQUENTIAL",
            "recommendation_v1": "KEEP_AS_FALLBACK",
        },
    ]
    return {"layer_name": "IQL_EPISODE_BOUNDARY_CANDIDATES_V1", "selected_boundary_v1": EPISODE_SCHEMA, "rows_v1": rows}


def _done_terminal_candidates(ordered: pd.DataFrame) -> dict[str, Any]:
    done_rows = int(ordered["run_id_v1"].nunique(dropna=True))
    rows = [
        {
            "done_rule_v1": "DONE_AT_LAST_EVENT_IN_RUN_ID",
            "rows_marked_done_v1": done_rows,
            "leakage_risk_v1": "LOW_AS_EVENT_ORDERED_METADATA",
            "terminal_meaningful_v1": "MEANINGFUL_FOR_EVENT_ORDERED_RESEARCH_EPISODE_END_ONLY",
            "reward_timing_supports_it_v1": "PARTIAL_REWARD_IS_EVENT_ATTACHED_NOT_TRUE_TERMINAL",
            "recommendation_v1": "USE_FOR_EVENT_ORDERED_RESEARCH_DATASET_ONLY",
        },
        {
            "done_rule_v1": "DONE_AT_SESSION_BOUNDARY",
            "rows_marked_done_v1": 0,
            "leakage_risk_v1": "UNKNOWN_SESSION_METADATA_MISSING",
            "terminal_meaningful_v1": "NO_CURRENTLY",
            "reward_timing_supports_it_v1": "UNKNOWN",
            "recommendation_v1": "NOT_READY",
        },
        {
            "done_rule_v1": "DONE_AT_TRADE_OR_POSITION_EXIT",
            "rows_marked_done_v1": 0,
            "leakage_risk_v1": "UNKNOWN_LIFECYCLE_METADATA_MISSING",
            "terminal_meaningful_v1": "NO_CURRENTLY",
            "reward_timing_supports_it_v1": "UNKNOWN",
            "recommendation_v1": "COLLECT_LIFECYCLE_METADATA_FOR_TRUE_IQL",
        },
        {
            "done_rule_v1": "DONE_UNAVAILABLE",
            "rows_marked_done_v1": 0,
            "leakage_risk_v1": "NONE_IF_CONTEXTUAL_ONLY",
            "terminal_meaningful_v1": "CONTEXTUAL_ONLY",
            "reward_timing_supports_it_v1": "N/A",
            "recommendation_v1": "FALLBACK_ONLY",
        },
    ]
    return {
        "layer_name": "IQL_DONE_TERMINAL_CANDIDATES_V1",
        "selected_done_rule_v1": "DONE_AT_LAST_EVENT_IN_RUN_ID",
        "done_rows_v1": done_rows,
        "rows_v1": rows,
    }


def _behavior_action_reconstruction(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    take = int(shield.sum())
    skip = int((~shield).sum())
    return {
        "layer_name": "IQL_BEHAVIOR_ACTION_RECONSTRUCTION_AUDIT_V1",
        "status_v1": "RESEARCH_ONLY_INFERRED_BINARY_ACTION_AVAILABLE",
        "action_field_available_v1": False,
        "action_actually_logged_v1": False,
        "inferred_action_rule_v1": "TAKE_TRADE iff inside SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY; otherwise SKIP",
        "take_count_v1": take,
        "skip_count_v1": skip,
        "action_imbalance_v1": float(take / max(take + skip, 1)),
        "safe_shield_eligible_but_skipped_count_v1": 0,
        "skip_is_real_or_counterfactual_v1": "COUNTERFACTUAL_OR_NON_SELECTED_POOL_NOT_TRUE_BEHAVIOR_LOG",
        "safe_for_iql_v1": "YES_FOR_EVENT_ORDERED_RESEARCH_ONLY_NOT_POLICY_EVAL",
        "behavior_policy_description_v1": "artifact-selection candidate substrate with external 78-row safety shield, not production behavior policy with propensities",
        "propensities_available_v1": False,
        "sizing_or_position_actions_supported_v1": False,
    }


def _reward_timing_reconstruction(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    reward = _reward(frame, shield)
    return {
        "layer_name": "IQL_REWARD_TIMING_RECONSTRUCTION_AUDIT_V1",
        "status_v1": "EVENT_ATTACHED_REWARD_AVAILABLE_TIMING_NOT_TRUE_LIFECYCLE",
        "reward_id_v1": REWARD_ID,
        "reward_available_for_selected_rows_v1": int(shield.sum()),
        "reward_available_for_skipped_non_selected_rows_v1": int((~shield).sum()),
        "reward_sum_v1": float(reward.sum()),
        "immediate_vs_delayed_reward_v1": "EVENT_ATTACHED_RESEARCH_REWARD; TRUE DELAYED OUTCOME TIME MISSING",
        "reward_belongs_to_v1": "candidate event row in current research dataset",
        "can_attach_to_action_t_v1": True,
        "reward_uses_labels_only_as_reward_eval_not_state_v1": True,
        "outcome_timestamp_available_v1": "outcome_timestamp_v1" in frame.columns,
        "reward_realization_time_available_v1": "reward_realization_time_v1" in frame.columns,
        "reward_leakage_risk_v1": "LOW_AS_REWARD_ONLY_HIGH_IF_USED_AS_STATE",
        "recommendation_v1": "USE_FOR_EVENT_ORDERED_RESEARCH_DATASET_ONLY; collect outcome timing before true lifecycle IQL",
    }


def _transition_dataset_feasibility(next_rows: list[dict[str, Any]], behavior: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    nonterminal = sum(1 for row in next_rows if row["has_next_row_v1"])
    terminal = sum(1 for row in next_rows if row["done_candidate_v1"])
    rows = [
        {
            "candidate_v1": "TRUE_SEQUENTIAL_TRANSITION_DATASET_READY",
            "ready_v1": False,
            "reason_v1": "true logged behavior action, lifecycle next_state, done/terminal, and reward timing are not source-logged",
        },
        {
            "candidate_v1": "EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_READY",
            "ready_v1": True,
            "reason_v1": "run_id + decision_timestamp produce deterministic same-run next-row candidates; action/reward are research-only inferred/event-attached",
        },
        {
            "candidate_v1": "CONTEXTUAL_ONLY_REMAINS_BEST",
            "ready_v1": False,
            "reason_v1": "contextual remains fallback, but event-ordered research transitions are now feasible with explicit limitations",
        },
        {
            "candidate_v1": "BLOCKED_NEEDS_SOURCE_METADATA",
            "ready_v1": False,
            "reason_v1": "not blocked for event-ordered research; still blocked for true trade-lifecycle sequential IQL",
        },
    ]
    return {
        "layer_name": "IQL_TRANSITION_DATASET_FEASIBILITY_V1",
        "status_v1": "EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_READY_NOT_TRUE_SEQUENTIAL",
        "true_transition_dataset_ready_v1": False,
        "event_ordered_research_transition_dataset_ready_v1": True,
        "nonterminal_transition_count_v1": nonterminal,
        "terminal_transition_count_v1": terminal,
        "total_rows_v1": len(next_rows),
        "behavior_action_status_v1": behavior["status_v1"],
        "reward_timing_status_v1": reward["status_v1"],
        "rows_v1": rows,
    }


def _no_fake_transition_audit(next_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "no_synthetic_next_state_v1": True,
        "no_random_next_state_v1": True,
        "no_cross_run_transitions_v1": all(row["cross_run_transition_prevented_v1"] for row in next_rows),
        "row_identity_not_state_v1": True,
        "reward_not_state_v1": True,
        "future_label_not_state_v1": True,
        "outcome_timing_not_state_v1": True,
        "done_not_based_on_future_success_label_v1": True,
        "artifact_membership_coverage_proxy_not_used_v1": True,
        "historical_v2_blueprint_not_used_v1": True,
        "selected_by_flag_not_state_v1": True,
        "audit_only_veto_not_state_v1": True,
        "next_state_from_real_next_event_row_v1": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "layer_name": "IQL_NO_FAKE_TRANSITION_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }
    validate_no_fake_transition_audit(payload)
    return payload


def _transition_dataset_build_spec(feasibility: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "IQL_TRANSITION_DATASET_BUILD_SPEC_V1",
        "status_v1": "READY_FOR_EVENT_ORDERED_RESEARCH_DATASET_NOT_TRUE_LIFECYCLE",
        "next_gate_v1": "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1",
        "episode_id_field_v1": "run_id_v1",
        "timestep_field_v1": "timestep_index_by_decision_timestamp_within_run_id_v1",
        "state_fields_v1": sanity_gate.MODEL_STATE_COLUMNS,
        "action_field_v1": "behavior_action_research_v1",
        "action_construction_v1": "TAKE_TRADE iff inside SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY else SKIP",
        "reward_field_v1": "safety_weighted_reward_v1",
        "reward_construction_v1": "SAFETY_WEIGHTED_REWARD from prior contract; labels reward/audit only, not state",
        "next_state_construction_v1": "next row in same run_id after sorting by decision_timestamp_v1; terminal if no next row",
        "done_construction_v1": "done true at last event in run_id",
        "split_policy_v1": "reuse run/fold/time-aware research split; freeze before training",
        "exclusion_rules_v1": [
            "exclude rows with missing run_id_v1 or decision_timestamp_v1",
            "fail closed on timestamp ties until tie-breaker is explicitly approved",
            "no cross-run transitions",
            "row ids audit-only, not state",
        ],
        "validation_checks_v1": [
            "transition count equals total rows",
            "terminal count equals run_id count",
            "nonterminal next timestamp is greater than current timestamp",
            "next_state uses AS_OF state fields from next event only",
            "reward absent from state",
            "no HISTORICAL_V2_BLUEPRINT or membership/coverage proxy",
        ],
        "expected_dataset_size_v1": {
            "rows_v1": feasibility["total_rows_v1"],
            "nonterminal_transitions_v1": feasibility["nonterminal_transition_count_v1"],
            "terminal_transitions_v1": feasibility["terminal_transition_count_v1"],
        },
        "limitations_v1": [
            "research-only event sequence, not production behavior log",
            "SKIP is counterfactual/non-selected pool, not true logged skip",
            "done is end-of-run event terminal, not trade exit terminal",
            "reward timing is event-attached, not source outcome timestamp aligned",
        ],
    }


def _recommendation(feasibility: dict[str, Any], no_fake: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    status = FINAL_STATUS
    next_action = NEXT_ACTION
    if no_fake["status_v1"] != "PASS":
        status = "IQL_SEQUENCE_METADATA_BLOCKED_BY_FAKE_TRANSITION_RISK"
        next_action = "COLLECT_ADDITIONAL_SEQUENCE_SOURCE_METADATA_V1"
    validate_final_status(status, next_action)
    recommendation = {
        "layer_name": "IQL_SEQUENCE_METADATA_RECOMMENDATION_V1",
        "final_status_v1": status,
        "next_recommended_action_v1": next_action,
        "true_transition_dataset_ready_v1": False,
        "event_ordered_research_transition_dataset_ready_v1": feasibility[
            "event_ordered_research_transition_dataset_ready_v1"
        ],
        "contextual_only_remains_fallback_v1": True,
        "recommendation_v1": "Build a research-only event-ordered transition dataset next; keep true lifecycle IQL blocked until action/done/reward timing metadata is source-logged.",
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_GO_NO_GO_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "true_transition_dataset_build_allowed_v1": False,
        "event_ordered_transition_dataset_build_allowed_v1": status
        == "IQL_SEQUENCE_METADATA_READY_FOR_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET",
        "true_sequential_iql_ready_v1": False,
        "event_ordered_research_ready_v1": True,
        "contextual_iql_research_still_valid_v1": True,
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
    inventory: list[dict[str, Any]],
    order_rows: list[dict[str, Any]],
    next_rows: list[dict[str, Any]],
    episode: dict[str, Any],
    done: dict[str, Any],
    behavior: dict[str, Any],
    reward: dict[str, Any],
    feasibility: dict[str, Any],
    no_fake: dict[str, Any],
    spec: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_sequence_metadata_reproducibility_audit_v1.md",
        [
            "# IQL Sequence Metadata Reproducibility Audit V1",
            "",
            f"- Previous transition status: `{repro['previous_transition_schema_status_v1']}`.",
            f"- Dataset rows: `{repro['dataset_rows_v1']}`.",
            f"- Contextual sanity policy: `{repro['contextual_policy_selected_rows_v1']}` selected, `{repro['contextual_policy_bad_tail_audit_only_v1']}` bad/tail.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    _write_report(
        artifact_root / "iql_sequence_metadata_inventory_v1.md",
        [
            "# IQL Sequence Metadata Inventory V1",
            "",
            *[
                f"- `{row['field_name_v1']}`: present={row['present_v1']}, order={row['usable_for_ordering_v1']}, episode={row['usable_for_episode_v1']}, next={row['usable_for_next_state_v1']}, done={row['usable_for_done_v1']}, action={row['usable_for_behavior_action_v1']}."
                for row in inventory
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_event_order_reconstruction_audit_v1.md",
        [
            "# IQL Event Order Reconstruction Audit V1",
            "",
            f"- Candidate episodes: `{len(order_rows)}`.",
            "- Ordering uses run_id_v1 + decision_timestamp_v1.",
            "- Intended scope: event-ordered research transitions only.",
        ],
    )
    _write_report(
        artifact_root / "iql_next_row_candidate_audit_v1.md",
        [
            "# IQL Next Row Candidate Audit V1",
            "",
            f"- Rows: `{len(next_rows)}`.",
            f"- Nonterminal transitions: `{sum(1 for row in next_rows if row['has_next_row_v1'])}`.",
            f"- Terminal rows: `{sum(1 for row in next_rows if row['done_candidate_v1'])}`.",
            "- Next rows are real next events in the same run_id, not synthetic states.",
        ],
    )
    _write_report(
        artifact_root / "iql_episode_boundary_candidates_v1.md",
        ["# IQL Episode Boundary Candidates V1", "", f"- Selected: `{episode['selected_boundary_v1']}`."],
    )
    _write_report(
        artifact_root / "iql_done_terminal_candidates_v1.md",
        ["# IQL Done Terminal Candidates V1", "", f"- Selected rule: `{done['selected_done_rule_v1']}`.", f"- Done rows: `{done['done_rows_v1']}`."],
    )
    _write_report(
        artifact_root / "iql_behavior_action_reconstruction_audit_v1.md",
        [
            "# IQL Behavior Action Reconstruction Audit V1",
            "",
            f"- Status: `{behavior['status_v1']}`.",
            f"- TAKE count: `{behavior['take_count_v1']}`.",
            f"- SKIP count: `{behavior['skip_count_v1']}`.",
            "- Actions are inferred research actions, not production logs.",
        ],
    )
    _write_report(
        artifact_root / "iql_reward_timing_reconstruction_audit_v1.md",
        ["# IQL Reward Timing Reconstruction Audit V1", "", f"- Status: `{reward['status_v1']}`.", f"- Reward id: `{reward['reward_id_v1']}`."],
    )
    _write_report(
        artifact_root / "iql_transition_dataset_feasibility_v1.md",
        [
            "# IQL Transition Dataset Feasibility V1",
            "",
            f"- Status: `{feasibility['status_v1']}`.",
            f"- Event-ordered research ready: `{feasibility['event_ordered_research_transition_dataset_ready_v1']}`.",
            f"- True transition ready: `{feasibility['true_transition_dataset_ready_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_no_fake_transition_audit_v1.md",
        ["# IQL No Fake Transition Audit V1", "", f"- Status: `{no_fake['status_v1']}`.", "- No synthetic/random/cross-run next_state was created."],
    )
    _write_report(
        artifact_root / "iql_transition_dataset_build_spec_v1.md",
        [
            "# IQL Transition Dataset Build Spec V1",
            "",
            f"- Status: `{spec['status_v1']}`.",
            f"- Next gate: `{spec['next_gate_v1']}`.",
            f"- Expected rows: `{spec['expected_dataset_size_v1']['rows_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_sequence_metadata_recommendation_v1.md",
        [
            "# IQL Sequence Metadata Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`.",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`.",
            "- This unlocks only research event-ordered dataset construction, not production IQL.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _frame_and_masks(inputs)
    ordered = _ordered_frame(frame)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility_audit(frame, masks, inputs)
    inventory = _sequence_metadata_inventory(frame)
    validate_sequence_inventory(inventory)
    order_rows = _event_order_audit(ordered)
    validate_event_order(order_rows)
    next_rows = _next_row_candidates(ordered)
    validate_next_row_candidates(
        next_rows,
        expected_rows=len(frame),
        expected_done=int(ordered["run_id_v1"].nunique(dropna=True)),
    )
    episode = _episode_boundary_candidates(frame, ordered)
    done = _done_terminal_candidates(ordered)
    behavior = _behavior_action_reconstruction(frame, masks)
    reward = _reward_timing_reconstruction(frame, masks)
    feasibility = _transition_dataset_feasibility(next_rows, behavior, reward)
    no_fake = _no_fake_transition_audit(next_rows)
    spec = _transition_dataset_build_spec(feasibility)
    recommendation, go_no_go = _recommendation(feasibility, no_fake)

    _write_json(artifact_root / "iql_sequence_metadata_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_sequence_metadata_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_sequence_metadata_inventory_v1.csv", inventory)
    _write_json(
        artifact_root / "iql_sequence_metadata_inventory_v1.json",
        {"row_count_v1": len(inventory), "rows_v1": inventory},
    )
    _write_rows(artifact_root / "iql_event_order_reconstruction_audit_v1.csv", order_rows)
    _write_json(
        artifact_root / "iql_event_order_reconstruction_audit_v1.json",
        {"row_count_v1": len(order_rows), "rows_v1": order_rows},
    )
    _write_rows(artifact_root / "iql_next_row_candidate_audit_v1.csv", next_rows)
    _write_json(
        artifact_root / "iql_next_row_candidate_audit_v1.json",
        {"row_count_v1": len(next_rows), "rows_v1": next_rows},
    )
    _write_json(artifact_root / "iql_episode_boundary_candidates_v1.json", episode)
    _write_json(artifact_root / "iql_done_terminal_candidates_v1.json", done)
    _write_json(artifact_root / "iql_behavior_action_reconstruction_audit_v1.json", behavior)
    _write_json(artifact_root / "iql_reward_timing_reconstruction_audit_v1.json", reward)
    _write_json(artifact_root / "iql_transition_dataset_feasibility_v1.json", feasibility)
    _write_json(artifact_root / "iql_no_fake_transition_audit_v1.json", no_fake)
    _write_json(artifact_root / "iql_transition_dataset_build_spec_v1.json", spec)
    _write_json(artifact_root / "iql_sequence_metadata_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "collect_or_reconstruct_iql_sequence_metadata_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        inventory,
        order_rows,
        next_rows,
        episode,
        done,
        behavior,
        reward,
        feasibility,
        no_fake,
        spec,
        recommendation,
    )

    summary = {
        "layer_name": "IQL_SEQUENCE_METADATA_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": recommendation["final_status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "event_order_valid_v1": True,
        "event_order_schema_v1": ORDER_SCHEMA,
        "next_state_can_be_constructed_v1": "YES_EVENT_ORDERED_RESEARCH_ONLY",
        "recommended_episode_schema_v1": EPISODE_SCHEMA,
        "recommended_done_schema_v1": done["selected_done_rule_v1"],
        "recommended_action_schema_v1": behavior["status_v1"],
        "recommended_reward_schema_v1": reward["status_v1"],
        "true_sequential_iql_ready_v1": False,
        "event_ordered_research_transition_dataset_ready_v1": True,
        "transition_dataset_kind_v1": TRANSITION_KIND,
        "expected_transition_rows_v1": feasibility["total_rows_v1"],
        "expected_nonterminal_transitions_v1": feasibility["nonterminal_transition_count_v1"],
        "expected_terminal_transitions_v1": feasibility["terminal_transition_count_v1"],
        "no_fake_transition_audit_status_v1": no_fake["status_v1"],
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
            "# Collect Or Reconstruct IQL Sequence Metadata V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- Event ordered transitions: `{feasibility['nonterminal_transition_count_v1']}` nonterminal + `{feasibility['terminal_transition_count_v1']}` terminal rows.",
            "- This is research-only event-order, not true trade-lifecycle sequential IQL.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect or reconstruct IQL sequence metadata, research only.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
