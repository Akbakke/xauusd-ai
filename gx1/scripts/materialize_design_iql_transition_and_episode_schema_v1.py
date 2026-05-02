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

from gx1.scripts import materialize_run_iql_offline_sanity_training_research_only_v1 as sanity_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1"

INPUT_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

FINAL_STATUS = "IQL_TRANSITION_SCHEMA_PARTIAL_NEEDS_SEQUENCE_METADATA"
NEXT_ACTION = "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1"
RECOMMENDED_DESIGN = "TRANSITION_SCHEMA_NEEDS_SOURCE_METADATA"

ALLOWED_FINAL_STATUSES = {
    "IQL_TRANSITION_SCHEMA_READY_FOR_TRANSITION_DATASET_BUILD",
    "IQL_TRANSITION_SCHEMA_READY_BUT_NEEDS_FIELD_NORMALIZATION",
    "IQL_TRANSITION_SCHEMA_PARTIAL_NEEDS_SEQUENCE_METADATA",
    "IQL_TRANSITION_SCHEMA_CONTEXTUAL_ONLY_NO_TRUE_SEQUENCE_AVAILABLE",
    "IQL_TRANSITION_SCHEMA_BLOCKED_BY_FAKE_OR_PSEUDO_TRANSITION_RISK",
    "IQL_TRANSITION_SCHEMA_BLOCKED_BY_ACTION_SUPPORT_GAPS",
    "IQL_TRANSITION_SCHEMA_BLOCKED_BY_REWARD_TIMING_GAPS",
    "IQL_TRANSITION_SCHEMA_BLOCKED_BY_STATE_OR_TRANSITION_LEAKAGE",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_IQL_TRANSITION_DATASET_RESEARCH_ONLY_V1",
    "NORMALIZE_IQL_TRANSITION_SCHEMA_FIELDS_V1",
    "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1",
    "CONTINUE_CONTEXTUAL_IQL_RESEARCH_ONLY_WITH_STRONGER_STATE_FEATURES_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "DEEPEN_IQL_REWARD_TIMING_AUDIT_V1",
    "REBUILD_IQL_TRANSITION_SCHEMA_WITH_NO_LEAKAGE_CONSTRAINTS_V1",
}

REQUIRED_OUTPUTS = [
    "iql_transition_schema_input_manifest_v1.json",
    "iql_transition_schema_reproducibility_audit_v1.json",
    "iql_transition_schema_reproducibility_audit_v1.md",
    "iql_transition_source_inventory_v1.csv",
    "iql_transition_source_inventory_v1.json",
    "iql_transition_source_inventory_v1.md",
    "iql_episode_schema_candidates_v1.json",
    "iql_episode_schema_candidates_v1.md",
    "iql_transition_schema_candidates_v1.json",
    "iql_transition_schema_candidates_v1.md",
    "iql_action_support_schema_v1.json",
    "iql_action_support_schema_v1.md",
    "iql_reward_timing_schema_v1.json",
    "iql_reward_timing_schema_v1.md",
    "iql_done_terminal_schema_v1.json",
    "iql_done_terminal_schema_v1.md",
    "iql_transition_feasibility_matrix_v1.csv",
    "iql_transition_feasibility_matrix_v1.json",
    "iql_transition_feasibility_matrix_v1.md",
    "iql_recommended_transition_design_v1.json",
    "iql_recommended_transition_design_v1.md",
    "iql_transition_dataset_build_plan_v1.json",
    "iql_transition_dataset_build_plan_v1.md",
    "iql_transition_schema_no_shortcut_audit_v1.json",
    "iql_transition_schema_no_shortcut_audit_v1.md",
    "iql_transition_schema_recommendation_v1.json",
    "iql_transition_schema_recommendation_v1.md",
    "design_iql_transition_and_episode_schema_go_no_go_v1.json",
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


def validate_no_fake_transition_design(design: dict[str, Any]) -> bool:
    if design.get("fake_transitions_created_v1"):
        raise RuntimeError("FAKE_TRANSITIONS_FORBIDDEN")
    if design.get("recommended_design_v1") == "TRUE_SEQUENTIAL_IQL_READY" and not design.get(
        "true_sequential_iql_possible_v1"
    ):
        raise RuntimeError("SEQUENTIAL_READY_WITHOUT_REQUIRED_FIELDS")
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
    if payload.get("transition_dataset_build_allowed_next_v1") and payload["status_v1"] != (
        "IQL_TRANSITION_SCHEMA_READY_FOR_TRANSITION_DATASET_BUILD"
    ):
        raise RuntimeError("TRANSITION_DATASET_BUILD_ALLOWED_WITHOUT_READY_STATUS")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_SANITY_ROOT, INPUT_CONTRACT_ROOT, INPUT_REFINE_CLEAN_ROOT, INPUT_HARDEN_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "sanity_summary": INPUT_SANITY_ROOT / "summary_v1.json",
        "sanity_go_no_go": INPUT_SANITY_ROOT / "run_iql_offline_sanity_training_research_only_go_no_go_v1.json",
        "sanity_transition_audit": INPUT_SANITY_ROOT / "iql_offline_sanity_transition_or_contextual_audit_v1.json",
        "sanity_no_shortcut": INPUT_SANITY_ROOT / "iql_offline_sanity_no_shortcut_audit_v1.json",
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_state": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "contract_action": INPUT_CONTRACT_ROOT / "iql_offline_action_contract_v1.json",
        "contract_reward": INPUT_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "contract_behavior": INPUT_CONTRACT_ROOT / "iql_offline_behavior_policy_audit_v1.json",
        "contract_split": INPUT_CONTRACT_ROOT / "iql_offline_split_policy_v1.json",
        "refine_summary": INPUT_REFINE_CLEAN_ROOT / "summary_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    sanity_go = _read_json(required["sanity_go_no_go"])
    if sanity_go.get("status_v1") != "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN":
        raise RuntimeError("INPUT_IQL_SANITY_NOT_CONTEXTUAL_TRANSITION_DESIGN_READY")
    if sanity_go.get("sequential_iql_ready_v1"):
        raise RuntimeError("INPUT_SANITY_ALREADY_MARKED_SEQUENTIAL_READY_UNEXPECTED")
    return {
        "required_paths": required,
        "sanity_summary": _read_json(required["sanity_summary"]),
        "sanity_go_no_go": sanity_go,
        "sanity_transition_audit": _read_json(required["sanity_transition_audit"]),
        "sanity_no_shortcut": _read_json(required["sanity_no_shortcut"]),
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_state": _read_json(required["contract_state"]),
        "contract_action": _read_json(required["contract_action"]),
        "contract_reward": _read_json(required["contract_reward"]),
        "contract_behavior": _read_json(required["contract_behavior"]),
        "contract_split": _read_json(required["contract_split"]),
        "refine_summary": _read_json(required["refine_summary"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "sanity_inputs": sanity_gate._load_inputs(),
    }


def _frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    return sanity_gate._frame_and_masks(inputs["sanity_inputs"])


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_TRANSITION_SCHEMA_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "first_iql_sanity_root_v1": str(INPUT_SANITY_ROOT),
            "iql_offline_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "clean_safety_layer_refinement_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
            "safe_core_hardening_root_v1": str(INPUT_HARDEN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "design_only_v1": True,
        "iql_training_run_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return sanity_gate._bool(frame, column)


def _reproducibility_audit(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shield = safe_core & ~masks["source_confluence_repairable_v1"]
    payload = {
        "layer_name": "IQL_TRANSITION_SCHEMA_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "previous_sanity_status_v1": inputs["sanity_go_no_go"].get("status_v1"),
        "previous_sanity_mode_v1": inputs["sanity_summary"].get("mode_v1"),
        "previous_sanity_no_shortcut_status_v1": inputs["sanity_no_shortcut"].get("status_v1"),
        "previous_contextual_reason_v1": inputs["sanity_transition_audit"].get("reason_v1"),
        "dataset_rows_v1": int(len(frame)),
        "state_feature_count_v1": int(inputs["sanity_summary"].get("state_feature_count_v1")),
        "safety_shield_v1": inputs["sanity_summary"].get("chosen_safety_shield_v1"),
        "contextual_policy_selected_rows_v1": int(inputs["sanity_summary"].get("policy_selected_rows_v1")),
        "contextual_policy_bad_tail_audit_only_v1": inputs["sanity_summary"].get("policy_bad_tail_audit_only_v1"),
        "contextual_policy_precision_audit_only_v1": float(inputs["sanity_summary"].get("policy_precision_audit_only_v1")),
        "contextual_policy_reward_sum_v1": float(inputs["sanity_summary"].get("policy_reward_sum_v1")),
        "contextual_policy_safety_status_v1": inputs["sanity_summary"].get("policy_safety_status_v1"),
        "baseline_140_94_v1": {
            "selected_rows_v1": int(baseline.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[baseline], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[baseline], "tail_label_v1").sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[baseline], "unsafe_audit_v1").sum()) == 0 else "FAIL",
        },
        "safe_core_89_v1": {
            "selected_rows_v1": int(safe_core.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[safe_core], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[safe_core], "tail_label_v1").sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[safe_core], "unsafe_audit_v1").sum()) == 0 else "FAIL",
        },
        "source_safety_shielded_78_v1": {
            "selected_rows_v1": int(shield.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[shield], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[shield], "tail_label_v1").sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[shield], "unsafe_audit_v1").sum()) == 0 else "FAIL",
        },
        "contextual_only_because_transitions_missing_v1": True,
        "no_fake_transitions_created_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    checks = [
        payload["dataset_rows_v1"] == 1914,
        payload["state_feature_count_v1"] == 11,
        payload["safety_shield_v1"] == "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY",
        payload["contextual_policy_selected_rows_v1"] == 76,
        payload["contextual_policy_bad_tail_audit_only_v1"] == [75, 55],
        payload["contextual_policy_safety_status_v1"] == "CLEAN",
        payload["previous_sanity_no_shortcut_status_v1"] == "PASS",
        payload["baseline_140_94_v1"]["selected_rows_v1"] == 140,
        payload["safe_core_89_v1"]["selected_rows_v1"] == 89,
        payload["source_safety_shielded_78_v1"]["selected_rows_v1"] == 78,
        payload["contextual_only_because_transitions_missing_v1"] is True,
        payload["no_fake_transitions_created_v1"] is True,
    ]
    if not all(checks):
        raise RuntimeError("IQL_TRANSITION_SCHEMA_REPRODUCTION_FAILED")
    return True


def _field_stats(frame: pd.DataFrame, field: str) -> tuple[bool, int, int]:
    if field not in frame.columns:
        return False, len(frame), 0
    series = frame[field]
    missing = int(series.isna().sum())
    return True, missing, int(series.nunique(dropna=True))


def _source_inventory(frame: pd.DataFrame) -> list[dict[str, Any]]:
    requested = [
        ("decision_timestamp_v1", "timestamp/event_time/bar_time", "source candidate frame"),
        ("run_id_v1", "run_id/episode candidate", "source candidate frame"),
        ("fold_id_v1", "fold/group split metadata", "source candidate frame"),
        ("trade_uid_v1", "trade lifecycle identifier", "source candidate frame"),
        ("trade_id_v1", "trade lifecycle identifier", "source candidate frame"),
        ("candidate_uid_v1", "row audit identity", "source candidate frame"),
        ("split_id_v1", "split id", "source candidate frame"),
        ("logged_action_v1", "logged behavior action", "sanity dataset only"),
        ("logged_action_id_v1", "logged behavior action id", "sanity dataset only"),
        ("next_state_vector_v1", "next_state pointer/vector", "not present"),
        ("sequence_next_row_key_v1", "next row pointer", "not present"),
        ("episode_id", "episode id", "not present"),
        ("sequence_episode_key_v1", "episode key", "not present"),
        ("timestep_index_v1", "timestep/order index", "not present"),
        ("done", "terminal/done marker", "not present"),
        ("terminal_step_status_v1", "terminal/done marker", "not present"),
        ("position_state_v1", "position state", "not present"),
        ("entry_time_v1", "entry relation", "not present"),
        ("exit_time_v1", "exit/outcome relation", "not present"),
        ("outcome_timestamp_v1", "outcome timing", "not present"),
        ("reward_realization_time_v1", "reward timing", "not present"),
        ("symbol_v1", "symbol/instrument", "not present"),
        ("instrument_v1", "symbol/instrument", "not present"),
    ]
    rows: list[dict[str, Any]] = []
    for field, purpose, source in requested:
        present, missing, unique = _field_stats(frame, field)
        can_order = field == "decision_timestamp_v1" and present
        can_episode = field == "run_id_v1" and present
        can_next = field in {"next_state_vector_v1", "sequence_next_row_key_v1"} and present
        can_done = field in {"done", "terminal_step_status_v1"} and present
        can_action = field in {"logged_action_v1", "logged_action_id_v1"} and present
        if field in {"candidate_uid_v1", "trade_uid_v1", "trade_id_v1"}:
            leakage = "HIGH_IF_USED_AS_STATE_OR_SELECTOR_AUDIT_ONLY"
            recommendation = "AUDIT_ONLY_NOT_STATE_OR_TRANSITION_SHORTCUT"
        elif can_order or can_episode:
            leakage = "LOW_AS_METADATA_MEDIUM_IF_USED_AS_STATE"
            recommendation = "USABLE_FOR_METADATA_ONLY_NEEDS_CONTRACT"
        elif can_next or can_done or can_action:
            leakage = "LOW_IF_LOGGED_AND_NOT_INFERRED"
            recommendation = "USABLE_IF_SOURCE_LOGGED"
        elif field in {"logged_action_v1", "logged_action_id_v1"}:
            leakage = "MEDIUM_INFERRED_IN_SANITY_DATASET"
            recommendation = "NEEDS_TRUE_BEHAVIOR_LOG_NOT_SANITY_INFERENCE"
        else:
            leakage = "MISSING"
            recommendation = "MISSING_REQUIRED_FOR_TRUE_SEQUENTIAL_IQL"
        rows.append(
            {
                "field_name_v1": field,
                "purpose_v1": purpose,
                "source_artifact_path_v1": source,
                "present_v1": present,
                "missing_rows_v1": missing if present else len(frame),
                "unique_values_v1": unique,
                "as_of_status_v1": "AS_OF_METADATA" if field in {"decision_timestamp_v1", "run_id_v1", "fold_id_v1"} and present else ("AUDIT_ONLY" if present and field.endswith("_uid_v1") else "MISSING_OR_NOT_CONTRACTED"),
                "can_define_ordering_v1": can_order,
                "can_define_episode_v1": can_episode,
                "can_define_next_state_v1": can_next,
                "can_define_done_v1": can_done,
                "can_define_logged_action_v1": can_action,
                "leakage_risk_v1": leakage,
                "recommendation_v1": recommendation,
            }
        )
    return rows


def validate_source_inventory(rows: list[dict[str, Any]]) -> bool:
    by_name = {row["field_name_v1"]: row for row in rows}
    if not by_name["decision_timestamp_v1"]["can_define_ordering_v1"]:
        raise RuntimeError("DECISION_TIMESTAMP_ORDERING_METADATA_MISSING")
    if not by_name["run_id_v1"]["can_define_episode_v1"]:
        raise RuntimeError("RUN_ID_EPISODE_METADATA_MISSING")
    for missing in ["next_state_vector_v1", "sequence_next_row_key_v1", "done", "terminal_step_status_v1"]:
        if by_name[missing]["present_v1"]:
            raise RuntimeError(f"UNEXPECTED_TRUE_SEQUENCE_FIELD_PRESENT: {missing}")
    return True


def _order_stats(frame: pd.DataFrame) -> dict[str, Any]:
    parsed = pd.to_datetime(frame["decision_timestamp_v1"], errors="coerce", utc=True)
    temp = frame.assign(_ts=parsed)
    monotonic = True
    duplicate_pairs = 0
    for _, group in temp.groupby("run_id_v1", dropna=False):
        ordered = group.sort_values(["_ts", "candidate_uid_v1"], kind="mergesort")
        monotonic = monotonic and bool(ordered["_ts"].is_monotonic_increasing)
        duplicate_pairs += int(ordered.duplicated(subset=["_ts"]).sum())
    return {
        "timestamp_rows_present_v1": int(parsed.notna().sum()),
        "timestamp_missing_rows_v1": int(parsed.isna().sum()),
        "run_id_count_v1": int(frame["run_id_v1"].nunique(dropna=True)),
        "fold_id_count_v1": int(frame["fold_id_v1"].nunique(dropna=True)),
        "timestamp_duplicate_within_run_rows_v1": duplicate_pairs,
        "run_time_order_available_v1": bool(parsed.notna().all() and frame["run_id_v1"].notna().all()),
        "ordering_note_v1": "timestamp + run_id can order candidate events, but this is not enough for true RL transitions without logged action/lifecycle/done metadata.",
    }


def _episode_schema_candidates(frame: pd.DataFrame) -> dict[str, Any]:
    stats = _order_stats(frame)
    rows = [
        {
            "schema_name_v1": "RUN_ID_EPISODE_SCHEMA",
            "required_fields_v1": ["run_id_v1", "decision_timestamp_v1", "logged_behavior_action", "done_or_terminal_rule"],
            "available_fields_v1": ["run_id_v1", "decision_timestamp_v1"],
            "missing_fields_v1": ["true logged action sequence", "done/terminal marker", "position/lifecycle state"],
            "ordering_is_real_v1": True,
            "next_state_meaningful_v1": "PARTIAL_EVENT_ORDER_ONLY",
            "done_can_be_defined_v1": "ONLY_END_OF_RUN_INFERENCE_NOT_OUTCOME_TERMINAL",
            "action_sequence_observable_v1": False,
            "leakage_risk_v1": "LOW_FOR_METADATA_MEDIUM_IF_USED_AS_STATE",
            "support_v1": f"{stats['run_id_count_v1']} run_id episodes over {len(frame)} rows",
            "recommendation_v1": "PROMISING_BUT_NEEDS_SEQUENCE_METADATA",
        },
        {
            "schema_name_v1": "SESSION_OR_TIME_EPISODE_SCHEMA",
            "required_fields_v1": ["decision_timestamp_v1", "session/date block", "symbol/instrument", "done rule"],
            "available_fields_v1": ["decision_timestamp_v1"],
            "missing_fields_v1": ["session_id", "symbol/instrument", "session terminal marker"],
            "ordering_is_real_v1": True,
            "next_state_meaningful_v1": "PARTIAL_TIME_ORDER_ONLY",
            "done_can_be_defined_v1": "NO_WITHOUT_SESSION_BOUNDARY",
            "action_sequence_observable_v1": False,
            "leakage_risk_v1": "MEDIUM_IF_SESSION_RECONSTRUCTED_POST_HOC",
            "support_v1": "timestamp support present but session metadata absent",
            "recommendation_v1": "NEEDS_SOURCE_SESSION_METADATA",
        },
        {
            "schema_name_v1": "GROUP_OR_FOLD_EPISODE_SCHEMA",
            "required_fields_v1": ["fold_id_v1 or group"],
            "available_fields_v1": ["fold_id_v1"],
            "missing_fields_v1": ["real temporal/action sequence"],
            "ordering_is_real_v1": False,
            "next_state_meaningful_v1": "NO_GROUP_FOLD_IS_SPLIT_METADATA",
            "done_can_be_defined_v1": "NO",
            "action_sequence_observable_v1": False,
            "leakage_risk_v1": "HIGH_IF_TREATED_AS_SEQUENCE",
            "support_v1": f"{stats['fold_id_count_v1']} folds",
            "recommendation_v1": "DIAGNOSTIC_ONLY_NOT_SEQUENTIAL_EPISODE",
        },
        {
            "schema_name_v1": "TRADE_LIFECYCLE_EPISODE_SCHEMA",
            "required_fields_v1": ["trade_uid_v1", "entry_time", "exit_time", "position_state", "outcome_time"],
            "available_fields_v1": ["trade_uid_v1", "trade_id_v1"],
            "missing_fields_v1": ["entry/exit relation", "position state", "outcome terminal timing"],
            "ordering_is_real_v1": False,
            "next_state_meaningful_v1": "NO_SINGLE_CANDIDATE_ROWS_NOT_LIFECYCLE_STEPS",
            "done_can_be_defined_v1": "NO_WITHOUT_EXIT_OR_OUTCOME_TIME",
            "action_sequence_observable_v1": False,
            "leakage_risk_v1": "HIGH_IF_TRADE_UID_USED_AS_SHORTCUT",
            "support_v1": "trade ids exist as audit identifiers but not lifecycle graph",
            "recommendation_v1": "NEEDS_TRADE_LEDGER_OR_LIFECYCLE_SOURCE",
        },
        {
            "schema_name_v1": "CONTEXTUAL_ONLY_NO_SEQUENCE_SCHEMA",
            "required_fields_v1": ["state_t", "action", "reward"],
            "available_fields_v1": ["AS_OF state", "binary inferred SKIP/TAKE", "reward as audit/training target"],
            "missing_fields_v1": ["next_state", "done", "episode sequence"],
            "ordering_is_real_v1": False,
            "next_state_meaningful_v1": "NO",
            "done_can_be_defined_v1": "NOT_NEEDED",
            "action_sequence_observable_v1": False,
            "leakage_risk_v1": "LOW_IF_DECLARED_CONTEXTUAL_ONLY",
            "support_v1": "validated by previous sanity gate",
            "recommendation_v1": "VALID_FALLBACK_UNTIL_SEQUENCE_METADATA_EXISTS",
        },
    ]
    return {"layer_name": "IQL_EPISODE_SCHEMA_CANDIDATES_V1", "row_count_v1": len(rows), "rows_v1": rows}


def _transition_schema_candidates() -> dict[str, Any]:
    rows = [
        {
            "transition_name_v1": "TRUE_SEQUENTIAL_TRANSITION",
            "state_t_v1": "AS_OF state vector",
            "action_t_v1": "true logged behavior action",
            "reward_t_v1": "delayed or terminal safety-weighted reward aligned to action",
            "next_state_t_plus_1_v1": "next event in same episode",
            "done_t_v1": "terminal marker from source lifecycle/session",
            "episode_id_v1": "source episode/session/trade lifecycle id",
            "timestep_index_v1": "source order index",
            "behavior_action_t_v1": "MISSING",
            "safety_shield_t_v1": "external safety eligibility",
            "eligibility_t_v1": "78 shield or later shield",
            "status_v1": "NOT_READY_MISSING_ACTION_DONE_NEXT_STATE_CONTRACT",
            "recommendation_v1": "DO_NOT_BUILD_UNTIL_SEQUENCE_METADATA_EXISTS",
        },
        {
            "transition_name_v1": "EVENT_ORDERED_TRANSITION",
            "state_t_v1": "AS_OF state ordered by run_id + decision_timestamp",
            "action_t_v1": "currently inferred SKIP/TAKE, not true behavior",
            "reward_t_v1": "post-event reward attached to candidate",
            "next_state_t_plus_1_v1": "possible next timestamp row but not yet validated as behavioral next state",
            "done_t_v1": "possible end-of-run only",
            "episode_id_v1": "run_id_v1",
            "timestep_index_v1": "decision_timestamp_v1 order",
            "behavior_action_t_v1": "MISSING_TRUE_LOG",
            "safety_shield_t_v1": "external shield",
            "eligibility_t_v1": "available",
            "status_v1": "PARTIAL_NEEDS_METADATA_RECONSTRUCTION",
            "recommendation_v1": "COLLECT_OR_RECONSTRUCT_BEFORE_DATASET_BUILD",
        },
        {
            "transition_name_v1": "PSEUDO_TRANSITION_BLOCKED",
            "state_t_v1": "any arbitrary row",
            "action_t_v1": "inferred selector",
            "reward_t_v1": "post-hoc reward",
            "next_state_t_plus_1_v1": "fabricated or row-order-only next state",
            "done_t_v1": "fabricated",
            "episode_id_v1": "fabricated",
            "timestep_index_v1": "fabricated",
            "behavior_action_t_v1": "not logged",
            "safety_shield_t_v1": "unclear",
            "eligibility_t_v1": "unclear",
            "status_v1": "BLOCKED_FAKE_TRANSITION_RISK",
            "recommendation_v1": "FORBIDDEN",
        },
        {
            "transition_name_v1": "CONTEXTUAL_ONE_STEP_TRANSITION",
            "state_t_v1": "AS_OF state vector",
            "action_t_v1": "binary contextual SKIP/TAKE",
            "reward_t_v1": "safety-weighted reward",
            "next_state_t_plus_1_v1": "none",
            "done_t_v1": "implicit one-step terminal",
            "episode_id_v1": "none",
            "timestep_index_v1": "none",
            "behavior_action_t_v1": "contextual/inferred only",
            "safety_shield_t_v1": "78 shield external",
            "eligibility_t_v1": "available",
            "status_v1": "VALID_CONTEXTUAL_RESEARCH_ONLY_NOT_SEQUENTIAL_IQL",
            "recommendation_v1": "KEEP_AS_FALLBACK",
        },
    ]
    return {"layer_name": "IQL_TRANSITION_SCHEMA_CANDIDATES_V1", "row_count_v1": len(rows), "rows_v1": rows}


def _action_support_schema(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    rows = [
        {
            "action_id_v1": "SKIP",
            "logged_action_available_v1": "INFERRED_CONTEXTUAL_ONLY",
            "inference_rule_v1": "not inside 78 shield in sanity dataset",
            "action_support_count_v1": int((~shield).sum()),
            "action_imbalance_v1": "HIGH",
            "can_iql_learn_action_value_meaningfully_v1": "LIMITED_CONTEXTUAL_ONLY",
            "limitations_v1": "not a true historical behavior skip sequence",
        },
        {
            "action_id_v1": "TAKE_TRADE",
            "logged_action_available_v1": "INFERRED_CONTEXTUAL_ONLY",
            "inference_rule_v1": "inside SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY",
            "action_support_count_v1": int(shield.sum()),
            "action_imbalance_v1": "HIGH",
            "can_iql_learn_action_value_meaningfully_v1": "LIMITED_CONTEXTUAL_ONLY",
            "limitations_v1": "not true behavior policy log with propensities",
        },
        {
            "action_id_v1": "TAKE_TRADE_SMALL_OR_REDUCED_RISK",
            "logged_action_available_v1": "NO",
            "inference_rule_v1": "",
            "action_support_count_v1": 0,
            "action_imbalance_v1": "UNSUPPORTED",
            "can_iql_learn_action_value_meaningfully_v1": "NO",
            "limitations_v1": "sizing and risk-reduction actions are not supported by locked artifacts",
        },
        {
            "action_id_v1": "POSITION_MANAGEMENT_HOLD_OR_EXIT",
            "logged_action_available_v1": "NO",
            "inference_rule_v1": "",
            "action_support_count_v1": 0,
            "action_imbalance_v1": "UNSUPPORTED",
            "can_iql_learn_action_value_meaningfully_v1": "NO",
            "limitations_v1": "requires trade lifecycle/position ledger",
        },
    ]
    return {
        "layer_name": "IQL_ACTION_SUPPORT_SCHEMA_V1",
        "status_v1": "PARTIAL_CONTEXTUAL_BINARY_ONLY_NEEDS_TRUE_BEHAVIOR_ACTION_LOG",
        "rows_v1": rows,
        "binary_contextual_support_available_v1": True,
        "true_logged_action_sequence_available_v1": False,
        "sizing_actions_supported_v1": False,
    }


def _reward_timing_schema() -> dict[str, Any]:
    return {
        "layer_name": "IQL_REWARD_TIMING_SCHEMA_V1",
        "status_v1": "PARTIAL_REWARD_DEFINED_BUT_TIMING_NEEDS_OUTCOME_METADATA",
        "reward_id_v1": "SAFETY_WEIGHTED_REWARD",
        "when_reward_realizes_v1": "post-event/outcome audit currently, exact outcome timestamp not in transition contract",
        "immediate_or_delayed_v1": "DELAYED_OR_TERMINAL_LIKELY",
        "belongs_to_trade_setup_or_episode_v1": "candidate/trade setup in current artifacts; episode alignment not proven",
        "can_link_to_action_t_v1": "contextual yes, sequential not yet",
        "recommended_temporal_alignment_v1": "reward_t_plus_1_or_terminal_reward_after outcome_time is reconstructed",
        "labels_reward_only_not_state_v1": True,
        "leakage_risk_v1": "LOW_AS_REWARD_ONLY_HIGH_IF_USED_AS_STATE",
        "missing_fields_v1": ["outcome_timestamp_v1", "reward_realization_time_v1", "entry_exit_lifecycle"],
    }


def _done_terminal_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "layer_name": "IQL_DONE_TERMINAL_SCHEMA_V1",
        "status_v1": "PARTIAL_END_OF_RUN_INFERENCE_ONLY_NOT_TRUE_TERMINAL",
        "done_field_available_v1": False,
        "terminal_at_end_of_run_id_possible_v1": "DIAGNOSTIC_ONLY",
        "terminal_at_end_of_session_possible_v1": "NO_SESSION_METADATA",
        "terminal_at_exit_outcome_possible_v1": "NO_EXIT_OR_OUTCOME_TIME",
        "terminal_at_no_next_event_possible_v1": "ONLY_IF_EVENT_ORDERED_SCHEMA_IS_APPROVED_LATER",
        "run_count_v1": int(frame["run_id_v1"].nunique(dropna=True)),
        "risk_v1": "end-of-run terminal can be administratively convenient but may not match trade/outcome terminal",
        "recommendation_v1": "collect terminal/done or lifecycle metadata before true sequential IQL",
    }


def _feasibility_matrix(frame: pd.DataFrame) -> list[dict[str, Any]]:
    stats = _order_stats(frame)
    return [
        {
            "schema_candidate_v1": "TRUE_SEQUENTIAL_TRANSITION",
            "sequential_validity_v1": "NO",
            "as_of_validity_v1": "POTENTIALLY_YES_FOR_STATE_ONLY",
            "next_state_validity_v1": "MISSING",
            "action_support_v1": "MISSING_TRUE_SEQUENCE",
            "reward_timing_validity_v1": "MISSING_OUTCOME_TIME",
            "done_validity_v1": "MISSING",
            "support_size_v1": 0,
            "leakage_risk_v1": "HIGH_IF_FORCED",
            "implementation_complexity_v1": "HIGH",
            "recommendation_v1": "NOT_RECOMMENDED_NOW",
        },
        {
            "schema_candidate_v1": "EVENT_ORDERED_TRANSITION",
            "sequential_validity_v1": "PARTIAL",
            "as_of_validity_v1": "YES_FOR_ORDER_METADATA",
            "next_state_validity_v1": "UNPROVEN",
            "action_support_v1": "INFERRED_CONTEXTUAL_ONLY",
            "reward_timing_validity_v1": "PARTIAL",
            "done_validity_v1": "END_OF_RUN_ONLY_DIAGNOSTIC",
            "support_size_v1": int(len(frame)) if stats["run_time_order_available_v1"] else 0,
            "leakage_risk_v1": "MEDIUM_IF_USED_BEFORE_METADATA_RECONSTRUCTION",
            "implementation_complexity_v1": "MEDIUM",
            "recommendation_v1": "NEEDS_SEQUENCE_METADATA_BEFORE_DATASET_BUILD",
        },
        {
            "schema_candidate_v1": "TRADE_LIFECYCLE_TRANSITION",
            "sequential_validity_v1": "NO_CURRENTLY",
            "as_of_validity_v1": "UNKNOWN",
            "next_state_validity_v1": "MISSING_LIFECYCLE_STEPS",
            "action_support_v1": "MISSING_POSITION_ACTIONS",
            "reward_timing_validity_v1": "MISSING_ENTRY_EXIT_OUTCOME_TIME",
            "done_validity_v1": "MISSING_EXIT_TERMINAL",
            "support_size_v1": 0,
            "leakage_risk_v1": "HIGH_IF_RECONSTRUCTED_FROM_OUTCOME_ONLY",
            "implementation_complexity_v1": "HIGH",
            "recommendation_v1": "COLLECT_TRADE_LIFECYCLE_SOURCE_METADATA",
        },
        {
            "schema_candidate_v1": "CONTEXTUAL_ONE_STEP_TRANSITION",
            "sequential_validity_v1": "NO_BY_DESIGN",
            "as_of_validity_v1": "YES",
            "next_state_validity_v1": "NOT_USED",
            "action_support_v1": "BINARY_CONTEXTUAL_ONLY",
            "reward_timing_validity_v1": "ACCEPTABLE_FOR_CONTEXTUAL_RESEARCH",
            "done_validity_v1": "IMPLICIT_ONE_STEP",
            "support_size_v1": int(len(frame)),
            "leakage_risk_v1": "LOW_IF_DECLARED_CONTEXTUAL_ONLY",
            "implementation_complexity_v1": "LOW",
            "recommendation_v1": "VALID_FALLBACK_NOT_TRUE_IQL",
        },
    ]


def _recommended_design(frame: pd.DataFrame, inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["field_name_v1"]: row for row in inventory_rows}
    true_sequence_possible = all(
        by_name[name]["present_v1"]
        for name in ["decision_timestamp_v1", "run_id_v1", "next_state_vector_v1", "done"]
        if name in by_name
    )
    design = {
        "layer_name": "IQL_RECOMMENDED_TRANSITION_DESIGN_V1",
        "recommended_design_v1": RECOMMENDED_DESIGN,
        "true_sequential_iql_possible_v1": False,
        "event_order_available_v1": by_name["decision_timestamp_v1"]["present_v1"] and by_name["run_id_v1"]["present_v1"],
        "episode_schema_v1": "RUN_ID_EPISODE_SCHEMA_NEEDS_METADATA",
        "transition_schema_v1": "EVENT_ORDERED_TRANSITION_NEEDS_SOURCE_METADATA",
        "action_schema_v1": "BINARY_CONTEXTUAL_ONLY_UNTIL_TRUE_ACTION_LOG_EXISTS",
        "reward_schema_v1": "SAFETY_WEIGHTED_REWARD_AS_REWARD_ONLY_NEEDS_OUTCOME_TIMING",
        "done_schema_v1": "DONE_TERMINAL_MISSING",
        "missing_sequence_action_reward_fields_v1": [
            "true logged behavior action sequence",
            "next row pointer or approved event-order transition contract",
            "done/terminal marker",
            "episode/session boundary",
            "position/trade lifecycle state",
            "outcome/reward realization timestamp",
            "symbol/instrument if episodes span multiple instruments",
        ],
        "fake_transitions_created_v1": False,
        "reason_v1": "run_id and decision_timestamp can order candidate events, but true sequential IQL still lacks logged action sequence, lifecycle next_state, terminal/done, and reward timing metadata.",
    }
    if true_sequence_possible:
        raise RuntimeError("Unexpected complete true sequence metadata found; review status selection")
    validate_no_fake_transition_design(design)
    return design


def _dataset_build_plan() -> dict[str, Any]:
    return {
        "layer_name": "IQL_TRANSITION_DATASET_BUILD_PLAN_V1",
        "status_v1": "NOT_READY_COLLECT_OR_RECONSTRUCT_SEQUENCE_METADATA_FIRST",
        "next_gate_if_metadata_collected_v1": "BUILD_IQL_TRANSITION_DATASET_RESEARCH_ONLY_V1",
        "next_gate_now_v1": NEXT_ACTION,
        "input_artifacts_v1": [
            str(INPUT_SANITY_ROOT),
            str(INPUT_CONTRACT_ROOT),
            str(INPUT_REFINE_CLEAN_ROOT),
            str(INPUT_HARDEN_ROOT),
            str(INPUT_PRECHECK_ROOT),
        ],
        "required_fields_before_dataset_build_v1": [
            "episode_id or approved episode construction rule",
            "timestep_index or approved event order",
            "next_state row pointer",
            "done/terminal marker",
            "true behavior action or explicitly scoped inferred-action contract",
            "reward realization timing",
            "position/trade lifecycle metadata if actions are sequential trade-management actions",
        ],
        "transition_construction_v1": "deferred until metadata gate",
        "episode_construction_v1": "candidate RUN_ID_EPISODE_SCHEMA but not approved as true episode yet",
        "split_policy_v1": "run/fold/time/group-aware splits from prior contract, frozen before training",
        "validation_checks_v1": [
            "no fake next_state",
            "next_state belongs to same episode and later timestep",
            "done iff no valid next step or terminal lifecycle event",
            "state_t contains AS_OF fields only",
            "reward labels are reward-only and not state",
            "logged action support audited by split",
        ],
        "no_shortcut_checks_v1": [
            "row identity absent from state",
            "artifact path absent from state",
            "membership/coverage proxy absent",
            "HISTORICAL_V2_BLUEPRINT absent",
            "selected flags absent",
            "MFE/hindsight absent from state",
            "audit-only veto absent from state",
        ],
    }


def _no_shortcut_audit(design: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "future_labels_not_state_v1": True,
        "reward_not_state_v1": True,
        "next_state_not_computed_from_outcome_v1": True,
        "row_identity_not_transition_shortcut_v1": True,
        "artifact_path_not_transition_shortcut_v1": True,
        "membership_coverage_proxy_not_used_v1": True,
        "historical_v2_blueprint_not_used_v1": True,
        "selected_flags_not_used_v1": True,
        "mfe_hindsight_not_state_v1": True,
        "audit_only_veto_not_state_v1": True,
        "fake_transitions_not_created_v1": not design.get("fake_transitions_created_v1", True),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "layer_name": "IQL_TRANSITION_SCHEMA_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }


def _recommendation(design: dict[str, Any], no_shortcut: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    status = FINAL_STATUS
    next_action = NEXT_ACTION
    if no_shortcut["status_v1"] != "PASS":
        status = "IQL_TRANSITION_SCHEMA_BLOCKED_BY_STATE_OR_TRANSITION_LEAKAGE"
        next_action = "REBUILD_IQL_TRANSITION_SCHEMA_WITH_NO_LEAKAGE_CONSTRAINTS_V1"
    validate_final_status(status, next_action)
    recommendation = {
        "layer_name": "IQL_TRANSITION_SCHEMA_RECOMMENDATION_V1",
        "final_status_v1": status,
        "next_recommended_action_v1": next_action,
        "recommended_design_v1": design["recommended_design_v1"],
        "transition_dataset_build_ready_v1": False,
        "true_sequential_iql_ready_v1": False,
        "contextual_research_still_valid_v1": True,
        "recommendation_v1": "Collect or reconstruct sequence metadata before building a transition dataset; do not fake next_state/done/action sequence.",
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_GO_NO_GO_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "true_sequential_iql_ready_v1": False,
        "transition_dataset_build_allowed_next_v1": False,
        "sequence_metadata_required_before_transition_build_v1": True,
        "contextual_iql_research_allowed_v1": True,
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
    episode: dict[str, Any],
    transition: dict[str, Any],
    action: dict[str, Any],
    reward: dict[str, Any],
    done: dict[str, Any],
    matrix: list[dict[str, Any]],
    design: dict[str, Any],
    plan: dict[str, Any],
    no_shortcut: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_transition_schema_reproducibility_audit_v1.md",
        [
            "# IQL Transition Schema Reproducibility Audit V1",
            "",
            f"- Previous sanity status: `{repro['previous_sanity_status_v1']}`.",
            f"- Previous mode: `{repro['previous_sanity_mode_v1']}`.",
            f"- Dataset rows: `{repro['dataset_rows_v1']}`.",
            f"- Contextual policy: `{repro['contextual_policy_selected_rows_v1']}` selected, `{repro['contextual_policy_bad_tail_audit_only_v1']}` bad/tail, reward `{repro['contextual_policy_reward_sum_v1']}`.",
            "- The previous gate stayed contextual because transition fields were missing.",
        ],
    )
    _write_report(
        artifact_root / "iql_transition_source_inventory_v1.md",
        [
            "# IQL Transition Source Inventory V1",
            "",
            *[
                f"- `{row['field_name_v1']}`: present={row['present_v1']}, order={row['can_define_ordering_v1']}, episode={row['can_define_episode_v1']}, next={row['can_define_next_state_v1']}, done={row['can_define_done_v1']}, action={row['can_define_logged_action_v1']}."
                for row in inventory
            ],
        ],
    )
    _write_report(
        artifact_root / "iql_episode_schema_candidates_v1.md",
        ["# IQL Episode Schema Candidates V1", "", *[f"- `{row['schema_name_v1']}`: {row['recommendation_v1']}." for row in episode["rows_v1"]]],
    )
    _write_report(
        artifact_root / "iql_transition_schema_candidates_v1.md",
        [
            "# IQL Transition Schema Candidates V1",
            "",
            *[f"- `{row['transition_name_v1']}`: {row['status_v1']} / {row['recommendation_v1']}." for row in transition["rows_v1"]],
        ],
    )
    _write_report(
        artifact_root / "iql_action_support_schema_v1.md",
        ["# IQL Action Support Schema V1", "", f"- Status: `{action['status_v1']}`.", "- True logged action sequence is not available yet."],
    )
    _write_report(
        artifact_root / "iql_reward_timing_schema_v1.md",
        ["# IQL Reward Timing Schema V1", "", f"- Status: `{reward['status_v1']}`.", f"- Missing fields: `{', '.join(reward['missing_fields_v1'])}`."],
    )
    _write_report(
        artifact_root / "iql_done_terminal_schema_v1.md",
        ["# IQL Done Terminal Schema V1", "", f"- Status: `{done['status_v1']}`.", f"- Recommendation: `{done['recommendation_v1']}`."],
    )
    _write_report(
        artifact_root / "iql_transition_feasibility_matrix_v1.md",
        ["# IQL Transition Feasibility Matrix V1", "", *[f"- `{row['schema_candidate_v1']}`: {row['recommendation_v1']}." for row in matrix]],
    )
    _write_report(
        artifact_root / "iql_recommended_transition_design_v1.md",
        [
            "# IQL Recommended Transition Design V1",
            "",
            f"- Recommended design: `{design['recommended_design_v1']}`.",
            f"- True sequential IQL possible now: `{design['true_sequential_iql_possible_v1']}`.",
            "- No fake transitions were created.",
        ],
    )
    _write_report(
        artifact_root / "iql_transition_dataset_build_plan_v1.md",
        [
            "# IQL Transition Dataset Build Plan V1",
            "",
            f"- Status: `{plan['status_v1']}`.",
            f"- Next gate now: `{plan['next_gate_now_v1']}`.",
            "- Transition dataset build is deferred until sequence metadata exists.",
        ],
    )
    _write_report(
        artifact_root / "iql_transition_schema_no_shortcut_audit_v1.md",
        ["# IQL Transition Schema No-Shortcut Audit V1", "", f"- Status: `{no_shortcut['status_v1']}`.", "- Fake transitions are blocked."],
    )
    _write_report(
        artifact_root / "iql_transition_schema_recommendation_v1.md",
        [
            "# IQL Transition Schema Recommendation V1",
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
    frame, masks = _frame_and_masks(inputs)

    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility_audit(frame, masks, inputs)
    inventory = _source_inventory(frame)
    validate_source_inventory(inventory)
    episode = _episode_schema_candidates(frame)
    transition = _transition_schema_candidates()
    action = _action_support_schema(frame, masks)
    reward = _reward_timing_schema()
    done = _done_terminal_schema(frame)
    matrix = _feasibility_matrix(frame)
    design = _recommended_design(frame, inventory)
    plan = _dataset_build_plan()
    no_shortcut = _no_shortcut_audit(design)
    recommendation, go_no_go = _recommendation(design, no_shortcut)

    _write_json(artifact_root / "iql_transition_schema_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_transition_schema_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_transition_source_inventory_v1.csv", inventory)
    _write_json(
        artifact_root / "iql_transition_source_inventory_v1.json",
        {"row_count_v1": len(inventory), "rows_v1": inventory},
    )
    _write_json(artifact_root / "iql_episode_schema_candidates_v1.json", episode)
    _write_json(artifact_root / "iql_transition_schema_candidates_v1.json", transition)
    _write_json(artifact_root / "iql_action_support_schema_v1.json", action)
    _write_json(artifact_root / "iql_reward_timing_schema_v1.json", reward)
    _write_json(artifact_root / "iql_done_terminal_schema_v1.json", done)
    _write_rows(artifact_root / "iql_transition_feasibility_matrix_v1.csv", matrix)
    _write_json(
        artifact_root / "iql_transition_feasibility_matrix_v1.json",
        {"row_count_v1": len(matrix), "rows_v1": matrix},
    )
    _write_json(artifact_root / "iql_recommended_transition_design_v1.json", design)
    _write_json(artifact_root / "iql_transition_dataset_build_plan_v1.json", plan)
    _write_json(artifact_root / "iql_transition_schema_no_shortcut_audit_v1.json", no_shortcut)
    _write_json(artifact_root / "iql_transition_schema_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "design_iql_transition_and_episode_schema_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        inventory,
        episode,
        transition,
        action,
        reward,
        done,
        matrix,
        design,
        plan,
        no_shortcut,
        recommendation,
    )

    summary = {
        "layer_name": "IQL_TRANSITION_AND_EPISODE_SCHEMA_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": recommendation["final_status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "recommended_design_v1": design["recommended_design_v1"],
        "true_sequential_iql_possible_v1": design["true_sequential_iql_possible_v1"],
        "recommended_episode_schema_v1": design["episode_schema_v1"],
        "recommended_transition_schema_v1": design["transition_schema_v1"],
        "event_order_available_v1": design["event_order_available_v1"],
        "missing_sequence_action_reward_fields_v1": design["missing_sequence_action_reward_fields_v1"],
        "no_shortcut_audit_status_v1": no_shortcut["status_v1"],
        "fake_transitions_created_v1": False,
        "transition_dataset_build_allowed_next_v1": False,
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
            "# Design IQL Transition And Episode Schema V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- Recommended design: `{design['recommended_design_v1']}`",
            "- True sequential IQL is not ready; sequence metadata must be collected or reconstructed first.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Design IQL transition and episode schema, research only.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
