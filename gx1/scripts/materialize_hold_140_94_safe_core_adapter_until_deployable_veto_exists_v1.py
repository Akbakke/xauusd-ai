#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1"

INPUT_VETO_MAPPING_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1_20260428T101045Z_LOCK"
)
INPUT_ADAPTER_MAPPING_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1_20260428T090840Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_SIMPLIFY_ROOT = DEFAULT_REPORTS_ROOT / "SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T083415Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "140_94_SAFE_CORE_ADAPTER_HELD_UNTIL_DEPLOYABLE_VETO"
NEXT_ACTION = "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1"

EXPECTED_SELECTED = 89
EXPECTED_RECOVERED = 86
EXPECTED_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607

ALLOWED_FINAL_STATUSES = {
    "140_94_SAFE_CORE_ADAPTER_HELD_UNTIL_DEPLOYABLE_VETO",
    "140_94_SAFE_CORE_HOLD_BLOCKED_BY_MISSING_ARTIFACTS",
    "140_94_SAFE_CORE_HOLD_BLOCKED_BY_INCONSISTENT_REPRODUCTION",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1",
    "RESOLVE_MISSING_SAFE_CORE_HOLD_ARTIFACTS_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
}

REQUIRED_OUTPUTS = [
    "hold_140_94_safe_core_input_manifest_v1.json",
    "hold_140_94_safe_core_reproducibility_audit_v1.json",
    "hold_140_94_safe_core_reproducibility_audit_v1.md",
    "hold_140_94_safe_core_blocker_audit_v1.json",
    "hold_140_94_safe_core_blocker_audit_v1.md",
    "hold_140_94_safe_core_decision_record_v1.json",
    "hold_140_94_safe_core_decision_record_v1.md",
    "hold_140_94_safe_core_blocker_contract_v1.json",
    "hold_140_94_safe_core_blocker_contract_v1.md",
    "hold_140_94_safe_core_current_state_v1.csv",
    "hold_140_94_safe_core_current_state_v1.json",
    "hold_140_94_safe_core_current_state_v1.md",
    "hold_140_94_safe_core_restart_conditions_v1.json",
    "hold_140_94_safe_core_restart_conditions_v1.md",
    "hold_140_94_safe_core_next_step_recommendation_v1.json",
    "hold_140_94_safe_core_next_step_recommendation_v1.md",
    "hold_140_94_safe_core_anti_shortcut_audit_v1.json",
    "hold_140_94_safe_core_anti_shortcut_audit_v1.md",
    "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json",
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
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30).splitlines()
    except Exception as exc:  # pragma: no cover
        freeze = [f"PIP_FREEZE_UNAVAILABLE: {exc}"]
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


def validate_no_forbidden_actions(
    *,
    r6: bool = False,
    adapter: bool = False,
    iql: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
    optuna: bool = False,
) -> dict[str, Any]:
    failures = []
    if r6:
        failures.append("R6_FORBIDDEN")
    if adapter:
        failures.append("ADAPTER_BUILD_FORBIDDEN")
    if iql:
        failures.append("IQL_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    expected = {
        "selected_rows_v1": EXPECTED_SELECTED,
        "recovered_original_140_rows_v1": EXPECTED_RECOVERED,
        "extra_rows_v1": EXPECTED_EXTRA,
        "bad_count_audit_only_v1": EXPECTED_BAD,
        "tail_count_audit_only_v1": EXPECTED_TAIL,
        "safety_status_v1": "CLEAN",
        "hard_safety_veto_status_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if abs(float(payload.get("precision_audit_only_v1", -1.0)) - EXPECTED_PRECISION) > 1e-12:
        failures["precision_audit_only_v1"] = payload.get("precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"SAFE_CORE_HOLD_REPRODUCTION_FAILED: {failures}")
    return True


def validate_blocker_contract(payload: dict[str, Any]) -> bool:
    if payload.get("adapter_may_resume_now_v1") is not False:
        raise RuntimeError("ADAPTER_MUST_NOT_RESUME_WITHOUT_DEPLOYABLE_VETO")
    conditions = payload.get("required_conditions_v1", [])
    if not conditions or all(condition.get("current_value_v1") is True for condition in conditions):
        raise RuntimeError("BLOCKER_CONTRACT_MUST_KEEP_AT_LEAST_ONE_CONDITION_FALSE")
    names = {condition.get("condition_id_v1") for condition in conditions}
    required = {
        "DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_EXISTS",
        "VETO_STOPS_UNSAFE_EXTRA_ROW",
        "NO_ROW_IDENTITY_SHORTCUT",
        "NO_AUDIT_ONLY_LABELS_OR_HINDSIGHT",
        "CLEAN_SIMULATED_ADAPTER_DRY_RUN",
    }
    missing = required - names
    if missing:
        raise RuntimeError(f"BLOCKER_CONTRACT_MISSING_REQUIRED_CONDITIONS: {sorted(missing)}")
    return True


def validate_decision_record(payload: dict[str, Any]) -> bool:
    required_false = [
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "iql_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
        "missing_54_expansion_active_v1",
    ]
    failures = [key for key in required_false if payload.get(key) is not False]
    required_true = [
        "safe_core_preserved_as_best_current_adapter_candidate_v1",
        "best_lane_185_139_comparator_only_v1",
        "plus45_diagnostic_only_v1",
    ]
    failures.extend(key for key in required_true if payload.get(key) is not True)
    if failures:
        raise RuntimeError(f"SAFE_CORE_HOLD_DECISION_RECORD_INVALID: {failures}")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"SAFE_CORE_HOLD_REQUIRED_OUTPUTS_MISSING: {missing}")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_VETO_MAPPING_ROOT,
        INPUT_ADAPTER_MAPPING_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_SIMPLIFY_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "veto_summary": INPUT_VETO_MAPPING_ROOT / "summary_v1.json",
        "veto_go_no_go": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_safe_core_veto_mapping_audit_go_no_go_v1.json",
        "veto_final_decision": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_final_veto_decision_v1.json",
        "veto_adapter_readiness": INPUT_VETO_MAPPING_ROOT
        / "deepen_140_94_adapter_readiness_after_veto_mapping_v1.json",
        "veto_blocker_audit": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_hard_safety_veto_blocker_audit_v1.json",
        "veto_unsafe_extra": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_unsafe_extra_without_veto_audit_v1.json",
        "veto_remaining_extra_3": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_remaining_extra_3_veto_decision_v1.json",
        "veto_candidate_dry_runs": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_candidate_veto_dry_run_results_v1.json",
        "adapter_mapping_summary": INPUT_ADAPTER_MAPPING_ROOT / "summary_v1.json",
        "adapter_mapping_go_no_go": INPUT_ADAPTER_MAPPING_ROOT
        / "build_140_94_safe_core_adapter_input_mapping_go_no_go_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "harden_go_no_go": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
        "simplify_summary": INPUT_SIMPLIFY_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    veto_go = _read_json(required["veto_go_no_go"])
    if veto_go.get("status_v1") != "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_AUDIT_ONLY_VETO":
        raise RuntimeError("INPUT_VETO_MAPPING_STATUS_NOT_AUDIT_ONLY_BLOCKED")
    return {
        "required_paths": required,
        "veto_summary": _read_json(required["veto_summary"]),
        "veto_go_no_go": veto_go,
        "veto_final_decision": _read_json(required["veto_final_decision"]),
        "veto_adapter_readiness": _read_json(required["veto_adapter_readiness"]),
        "veto_blocker_audit": _read_json(required["veto_blocker_audit"]),
        "veto_unsafe_extra": _read_json(required["veto_unsafe_extra"]),
        "veto_remaining_extra_3": _read_json(required["veto_remaining_extra_3"]),
        "veto_candidate_dry_runs": _read_json(required["veto_candidate_dry_runs"]),
        "adapter_mapping_summary": _read_json(required["adapter_mapping_summary"]),
        "adapter_mapping_go_no_go": _read_json(required["adapter_mapping_go_no_go"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "simplify_summary": _read_json(required["simplify_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "HOLD_140_94_SAFE_CORE_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "veto_mapping_audit_root_v1": str(INPUT_VETO_MAPPING_ROOT),
            "adapter_input_mapping_root_v1": str(INPUT_ADAPTER_MAPPING_ROOT),
            "harden_root_v1": str(INPUT_HARDEN_ROOT),
            "simplify_root_v1": str(INPUT_SIMPLIFY_ROOT),
            "precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _reproducibility(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = inputs["veto_summary"]
    payload = {
        "layer_name": "HOLD_140_94_SAFE_CORE_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": summary.get("safe_core_rule_id_v1"),
        "selected_rows_v1": summary.get("selected_rows_v1"),
        "recovered_original_140_rows_v1": summary.get("recovered_original_140_rows_v1"),
        "extra_rows_v1": summary.get("extra_rows_v1"),
        "bad_count_audit_only_v1": summary.get("bad_tail_audit_only_v1", [None, None])[0],
        "tail_count_audit_only_v1": summary.get("bad_tail_audit_only_v1", [None, None])[1],
        "precision_audit_only_v1": summary.get("precision_audit_only_v1"),
        "safety_status_v1": summary.get("safety_status_v1"),
        "hard_safety_veto_status_v1": summary.get("hard_safety_veto_mapping_result_v1"),
        "unsafe_extra_without_hard_veto_rows_v1": summary.get("unsafe_extra_without_hard_veto_rows_v1"),
        "remaining_extra_3_decision_v1": summary.get("remaining_extra_3_decision_v1"),
        "source_veto_mapping_status_v1": inputs["veto_go_no_go"].get("status_v1"),
    }
    validate_reproducibility(payload)
    return payload


def _blocker_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    hard_blocker = inputs["veto_blocker_audit"]
    final = inputs["veto_final_decision"]
    unsafe = inputs["veto_unsafe_extra"]
    dry = inputs["veto_candidate_dry_runs"].get("rows_v1", [])
    signal_shape = [
        row for row in dry if str(row.get("candidate_veto_name_v1", "")).startswith("AS_OF_SIGNAL_SHAPE")
    ]
    return {
        "layer_name": "HOLD_140_94_SAFE_CORE_BLOCKER_AUDIT_V1",
        "primary_blocker_v1": "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO",
        "hard_safety_veto_status_v1": final.get("hard_safety_veto_final_status_v1"),
        "hard_safety_veto_deployable_v1": False,
        "unsafe_extra_without_hard_veto_rows_v1": unsafe.get("row_count_v1"),
        "unsafe_extra_rows_v1": unsafe.get("rows_v1", []),
        "signal_shape_as_of_vetoes_found_v1": [row.get("candidate_veto_name_v1") for row in signal_shape],
        "signal_shape_as_of_vetoes_sufficient_v1": False,
        "reason_signal_shape_vetoes_not_sufficient_v1": hard_blocker.get("reason_signal_shape_candidates_rejected_v1"),
        "row_identity_veto_rejected_v1": True,
        "row_identity_veto_rejection_reason_v1": "Row identity is a forbidden shortcut and cannot be deployable adapter logic.",
        "remaining_extra_3_status_v1": final.get("false_positive_extra_3_veto_status_v1"),
        "remaining_extra_3_safety_clean_v1": inputs["veto_remaining_extra_3"].get("summary_v1", {}).get(
            "safety_status_v1"
        )
        == "CLEAN",
        "adapter_build_blocker_v1": "Adapter would admit one unsafe extra row unless a deployable AS_OF hard safety veto exists.",
        "status_v1": "BLOCKER_REPRODUCED_ADAPTER_MUST_HOLD",
    }


def _decision_record() -> dict[str, Any]:
    payload = {
        "layer_name": "HOLD_140_94_SAFE_CORE_DECISION_RECORD_V1",
        "safe_core_preserved_as_best_current_adapter_candidate_v1": True,
        "safe_core_hold_reason_v1": "Audit-only hard safety veto is not deployable, and without it one unsafe extra row appears.",
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "missing_54_expansion_active_v1": False,
        "missing_54_expansion_status_v1": "SEPARATE_NOT_ACTIVE_MAINLINE",
        "best_lane_185_139_comparator_only_v1": True,
        "plus45_diagnostic_only_v1": True,
        "final_promotion_allowed_v1": False,
        "hold_is_model_training_gate_v1": False,
        "hold_is_adapter_gate_v1": False,
    }
    validate_decision_record(payload)
    return payload


def _blocker_contract() -> dict[str, Any]:
    conditions = [
        {
            "condition_id_v1": "DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_EXISTS",
            "required_value_v1": True,
            "current_value_v1": False,
            "reason_v1": "Current hard safety veto is diagnostic/audit-only.",
        },
        {
            "condition_id_v1": "VETO_STOPS_UNSAFE_EXTRA_ROW",
            "required_value_v1": True,
            "current_value_v1": True,
            "reason_v1": "The audit veto stops it, but deployable lineage is missing.",
        },
        {
            "condition_id_v1": "VETO_DOES_NOT_CUT_UNACCEPTABLY_MANY_SAFE_CORE_ROWS",
            "required_value_v1": True,
            "current_value_v1": False,
            "reason_v1": "Available AS_OF signal-shape vetoes cut too many good safe-core rows.",
        },
        {
            "condition_id_v1": "NO_ROW_IDENTITY_SHORTCUT",
            "required_value_v1": True,
            "current_value_v1": True,
            "reason_v1": "Row identity veto was explicitly rejected.",
        },
        {
            "condition_id_v1": "NO_AUDIT_ONLY_LABELS_OR_HINDSIGHT",
            "required_value_v1": True,
            "current_value_v1": False,
            "reason_v1": "Current exact hard safety veto uses audit-only safety labels/fields.",
        },
        {
            "condition_id_v1": "VETO_COMPUTABLE_BEFORE_OUTCOME",
            "required_value_v1": True,
            "current_value_v1": False,
            "reason_v1": "No deployable before-outcome lineage has been proven.",
        },
        {
            "condition_id_v1": "CLEAN_SIMULATED_ADAPTER_DRY_RUN",
            "required_value_v1": True,
            "current_value_v1": False,
            "reason_v1": "Exact dry-run is clean only with audit-only hard safety veto.",
        },
        {
            "condition_id_v1": "NO_SHORTCUT_AUDIT_PASSES",
            "required_value_v1": True,
            "current_value_v1": True,
            "reason_v1": "This hold gate introduces no shortcut and keeps adapter blocked.",
        },
    ]
    payload = {
        "layer_name": "HOLD_140_94_SAFE_CORE_BLOCKER_CONTRACT_V1",
        "adapter_may_resume_now_v1": False,
        "required_conditions_v1": conditions,
        "restart_gate_required_v1": "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1",
    }
    validate_blocker_contract(payload)
    return payload


def _current_state_rows(inputs: dict[str, Any], repro: dict[str, Any], blocker: dict[str, Any]) -> list[dict[str, Any]]:
    unsafe_rows = blocker.get("unsafe_extra_rows_v1", [])
    unsafe_uid = unsafe_rows[0].get("candidate_uid_v1") if unsafe_rows else "UNKNOWN"
    summary = inputs["adapter_mapping_summary"]
    return [
        {"state_item_v1": "safe_core_rule_id", "value_v1": SAFE_CORE_RULE_ID, "status_v1": "PRESERVED"},
        {"state_item_v1": "selected_rows", "value_v1": repro["selected_rows_v1"], "status_v1": "REPRODUCED"},
        {
            "state_item_v1": "recovered_original_140_rows",
            "value_v1": repro["recovered_original_140_rows_v1"],
            "status_v1": "REPRODUCED",
        },
        {"state_item_v1": "extra_rows", "value_v1": repro["extra_rows_v1"], "status_v1": "REPRODUCED"},
        {"state_item_v1": "bad_tail_audit_only", "value_v1": "86/55", "status_v1": "REPRODUCED"},
        {"state_item_v1": "precision_audit_only", "value_v1": EXPECTED_PRECISION, "status_v1": "REPRODUCED"},
        {"state_item_v1": "safety_status", "value_v1": "CLEAN", "status_v1": "REPRODUCED"},
        {
            "state_item_v1": "unsafe_extra_without_hard_veto",
            "value_v1": unsafe_uid,
            "status_v1": "BLOCKER_REPRODUCED",
        },
        {
            "state_item_v1": "required_positive_signals",
            "value_v1": "score >= 0.95 + R5_1_BAD_SCORE + V2_LIKE_BAD_TAIL",
            "status_v1": "MAPPED",
        },
        {
            "state_item_v1": "required_vetoes",
            "value_v1": "hard safety veto + low-support missing-artifact veto + later false-positive veto decision",
            "status_v1": "PARTIALLY_UNMAPPED",
        },
        {
            "state_item_v1": "mapped_adapter_fields",
            "value_v1": summary.get("mapped_adapter_fields_v1"),
            "status_v1": "MAPPED",
        },
        {
            "state_item_v1": "unmapped_blocking_fields",
            "value_v1": summary.get("unmapped_blocking_fields_v1"),
            "status_v1": "BLOCKING",
        },
        {
            "state_item_v1": "adapter_ready_fields",
            "value_v1": "tail-repaired score; R5_1 support; V2-like support",
            "status_v1": "READY",
        },
        {
            "state_item_v1": "not_adapter_ready_fields",
            "value_v1": "deployable hard safety veto; low-support normalization; false-positive veto for 3 extras",
            "status_v1": "BLOCKING",
        },
    ]


def _restart_conditions() -> dict[str, Any]:
    return {
        "layer_name": "HOLD_140_94_SAFE_CORE_RESTART_CONDITIONS_V1",
        "adapter_reopens_only_if_v1": [
            "a deployable AS_OF hard safety veto exists",
            "the veto stops the unsafe extra row",
            "the veto keeps acceptable safe-core retention",
            "the veto uses no row identity shortcut",
            "the veto uses no audit-only labels, hindsight, MFE, or final outcome",
            "the veto is computable before outcome",
            "simulated adapter dry-run is clean",
            "no-shortcut audit passes",
            "PROJECT_STATE.md and DECISION_LOG.md are explicitly updated",
        ],
        "current_restart_allowed_v1": False,
        "required_future_gate_v1": "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1",
    }


def _recommendation() -> dict[str, Any]:
    return {
        "layer_name": "HOLD_140_94_SAFE_CORE_NEXT_STEP_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "rationale_v1": [
            "Safe-core is the best current adapter candidate, but adapter safety depends on an audit-only hard veto.",
            "Without the hard safety veto, one unsafe extra row enters.",
            "Signal-shape AS_OF vetoes are too coarse, and row identity is forbidden.",
            "The next gate should discover a deployable AS_OF hard safety veto before any adapter/R6/IQL work.",
        ],
        "mainline_v1": "HOLD_SAFE_CORE_UNTIL_DEPLOYABLE_VETO",
    }


def _anti_shortcut() -> dict[str, Any]:
    return {
        "layer_name": "HOLD_140_94_SAFE_CORE_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_HOLD_LOCK_NO_SHORTCUTS",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_run_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_model_training_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_implicit_latest_glob_selection_v1": True,
        "no_dummy_synthetic_fallback_v1": True,
        "row_identity_veto_rejected_v1": True,
        "audit_only_veto_not_promoted_v1": True,
        "best_lane_185_139_comparator_only_v1": True,
        "plus45_diagnostic_only_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }


def _go_no_go(repro: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "layer_name": "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "hard_safety_veto_status_v1": repro["hard_safety_veto_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": repro["unsafe_extra_without_hard_veto_rows_v1"],
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    return payload


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    blocker: dict[str, Any],
    decision: dict[str, Any],
    contract: dict[str, Any],
    current_rows: list[dict[str, Any]],
    restart: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "hold_140_94_safe_core_reproducibility_audit_v1.md",
        [
            "# Hold 140/94 Safe-Core Reproducibility Audit V1",
            "",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_audit_only_v1']} / {repro['tail_count_audit_only_v1']}`",
            f"- Safety: `{repro['safety_status_v1']}`",
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_blocker_audit_v1.md",
        [
            "# Hold 140/94 Safe-Core Blocker Audit V1",
            "",
            f"- Hard safety veto status: `{blocker['hard_safety_veto_status_v1']}`",
            f"- Unsafe extras without hard veto: `{blocker['unsafe_extra_without_hard_veto_rows_v1']}`",
            "- Signal-shape AS_OF vetoes are not sufficient because they remove too many good safe-core rows.",
            "- Row-identity veto remains forbidden.",
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_decision_record_v1.md",
        [
            "# Hold 140/94 Safe-Core Decision Record V1",
            "",
            f"- Adapter allowed: `{decision['adapter_build_allowed_v1']}`",
            f"- R6 allowed: `{decision['r6_allowed_v1']}`",
            f"- IQL allowed: `{decision['iql_allowed_v1']}`",
            "- Safe-core is preserved as the best current adapter candidate, but held.",
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_blocker_contract_v1.md",
        [
            "# Hold 140/94 Safe-Core Blocker Contract V1",
            "",
            f"- Adapter may resume now: `{contract['adapter_may_resume_now_v1']}`",
            *[
                f"- `{row['condition_id_v1']}`: current `{row['current_value_v1']}`, required `{row['required_value_v1']}`"
                for row in contract["required_conditions_v1"]
            ],
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_current_state_v1.md",
        [
            "# Hold 140/94 Safe-Core Current State V1",
            "",
            *[f"- `{row['state_item_v1']}`: `{row['value_v1']}` ({row['status_v1']})" for row in current_rows],
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_restart_conditions_v1.md",
        [
            "# Hold 140/94 Safe-Core Restart Conditions V1",
            "",
            f"- Current restart allowed: `{restart['current_restart_allowed_v1']}`",
            *[f"- {item}" for item in restart["adapter_reopens_only_if_v1"]],
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_next_step_recommendation_v1.md",
        [
            "# Hold 140/94 Safe-Core Next Step Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
        ],
    )
    _write_report(
        root / "hold_140_94_safe_core_anti_shortcut_audit_v1.md",
        [
            "# Hold 140/94 Safe-Core Anti-Shortcut Audit V1",
            "",
            "- No R6, adapter, IQL, package, freeze, promo, live, Optuna, broad sweep, or model training was run.",
            "- Audit-only veto was not promoted; row identity remains forbidden.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(inputs)
    blocker = _blocker_audit(inputs)
    decision = _decision_record()
    contract = _blocker_contract()
    current_rows = _current_state_rows(inputs, repro, blocker)
    restart = _restart_conditions()
    recommendation = _recommendation()
    anti = _anti_shortcut()
    go_no_go = _go_no_go(repro)

    _write_json(artifact_root / "hold_140_94_safe_core_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "hold_140_94_safe_core_reproducibility_audit_v1.json", repro)
    _write_json(artifact_root / "hold_140_94_safe_core_blocker_audit_v1.json", blocker)
    _write_json(artifact_root / "hold_140_94_safe_core_decision_record_v1.json", decision)
    _write_json(artifact_root / "hold_140_94_safe_core_blocker_contract_v1.json", contract)
    _write_rows(artifact_root / "hold_140_94_safe_core_current_state_v1.csv", current_rows)
    _write_json(
        artifact_root / "hold_140_94_safe_core_current_state_v1.json",
        {"row_count_v1": len(current_rows), "rows_v1": current_rows},
    )
    _write_json(artifact_root / "hold_140_94_safe_core_restart_conditions_v1.json", restart)
    _write_json(artifact_root / "hold_140_94_safe_core_next_step_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "hold_140_94_safe_core_anti_shortcut_audit_v1.json", anti)
    _write_json(
        artifact_root / "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json",
        go_no_go,
    )

    summary = {
        "layer_name": "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "adapter_held_v1": True,
        "r6_held_v1": True,
        "iql_held_v1": True,
        "hard_safety_veto_status_v1": repro["hard_safety_veto_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": repro["unsafe_extra_without_hard_veto_rows_v1"],
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": FINAL_STATUS, "next_recommended_action_v1": NEXT_ACTION, "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Hold 140/94 Safe-Core Adapter Until Deployable Veto Exists V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Safe-core: `{SAFE_CORE_RULE_ID}`",
            f"- Result: `{EXPECTED_SELECTED}` selected, `{EXPECTED_RECOVERED}` original-140 recovered, `{EXPECTED_EXTRA}` extra.",
            f"- Hard safety veto: `{repro['hard_safety_veto_status_v1']}`",
            "- Adapter, R6, IQL, package/freeze/promo/live remain blocked.",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(artifact_root, repro, blocker, decision, contract, current_rows, restart, recommendation)
    validate_required_outputs(artifact_root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = materialize(args.artifact_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
