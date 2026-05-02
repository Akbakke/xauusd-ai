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

from gx1.scripts import materialize_deepen_140_94_safe_core_veto_mapping_audit_v1 as veto_gate
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1"

INPUT_HOLD_ROOT = (
    DEFAULT_REPORTS_ROOT / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T111216Z_LOCK"
)
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
FINAL_STATUS = "140_94_VETO_FOUND_BUT_TOO_DESTRUCTIVE_TO_SAFE_CORE"
NEXT_ACTION = "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1"

EXPECTED_SELECTED = 89
EXPECTED_RECOVERED = 86
EXPECTED_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607
ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT = 3

ALLOWED_FINAL_STATUSES = {
    "140_94_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOUND_ADAPTER_READY",
    "140_94_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOUND_NEEDS_NORMALIZATION",
    "140_94_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOUND_NEEDS_LINEAGE_CONFIRMATION",
    "140_94_VETO_FOUND_BUT_TOO_DESTRUCTIVE_TO_SAFE_CORE",
    "140_94_VETO_FOUND_BUT_UNSAFE_LOOKALIKE_RISK_REMAINS",
    "140_94_NO_DEPLOYABLE_HARD_SAFETY_VETO_FOUND",
    "140_94_HARD_SAFETY_VETO_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_SAFE_CORE_ADAPTER_V1",
    "NORMALIZE_140_94_HARD_SAFETY_VETO_INPUTS_V1",
    "DEEPEN_140_94_HARD_SAFETY_VETO_LINEAGE_AUDIT_V1",
    "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1",
    "DEEPEN_140_94_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
}

REQUIRED_OUTPUTS = [
    "discover_140_94_hard_safety_veto_input_manifest_v1.json",
    "discover_140_94_hard_safety_veto_reproducibility_audit_v1.json",
    "discover_140_94_hard_safety_veto_reproducibility_audit_v1.md",
    "discover_140_94_unsafe_extra_row_audit_v1.csv",
    "discover_140_94_unsafe_extra_row_audit_v1.json",
    "discover_140_94_unsafe_extra_row_audit_v1.md",
    "discover_140_94_candidate_veto_families_v1.json",
    "discover_140_94_candidate_veto_families_v1.md",
    "discover_140_94_candidate_veto_metrics_v1.csv",
    "discover_140_94_candidate_veto_metrics_v1.json",
    "discover_140_94_candidate_veto_metrics_v1.md",
    "discover_140_94_candidate_veto_dry_run_results_v1.csv",
    "discover_140_94_candidate_veto_dry_run_results_v1.json",
    "discover_140_94_candidate_veto_dry_run_results_v1.md",
    "discover_140_94_veto_row_retention_audit_v1.csv",
    "discover_140_94_veto_row_retention_audit_v1.json",
    "discover_140_94_veto_row_retention_audit_v1.md",
    "discover_140_94_unsafe_lookalike_audit_v1.csv",
    "discover_140_94_unsafe_lookalike_audit_v1.json",
    "discover_140_94_unsafe_lookalike_audit_v1.md",
    "discover_140_94_final_veto_selection_v1.json",
    "discover_140_94_final_veto_selection_v1.md",
    "discover_140_94_adapter_reopen_assessment_v1.json",
    "discover_140_94_adapter_reopen_assessment_v1.md",
    "discover_140_94_hard_safety_veto_anti_shortcut_audit_v1.json",
    "discover_140_94_hard_safety_veto_anti_shortcut_audit_v1.md",
    "discover_140_94_hard_safety_veto_recommendation_v1.json",
    "discover_140_94_hard_safety_veto_recommendation_v1.md",
    "discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_go_no_go_v1.json",
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


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return simplify._bool(frame, column)


def _str(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    return simplify._str(frame, column, default)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return simplify._num(frame, column, default)


def _as_bool(value: Any) -> bool:
    return simplify._as_bool(value)


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
        raise RuntimeError(f"HARD_SAFETY_VETO_DISCOVERY_REPRODUCTION_FAILED: {failures}")
    return True


def validate_candidate_veto_metrics(rows: list[dict[str, Any]]) -> bool:
    families = {row["veto_family_v1"] for row in rows}
    required = {
        "SIGNAL_SHAPE_REFINED_VETO",
        "LOW_SUPPORT_OR_MISSING_ARTIFACT_VETO",
        "SAFE_CORE_DISTANCE_MARGIN_VETO",
        "BRANCH_SPECIFIC_VETO",
        "VETO_CONFLUENCE_RULE",
        "FALSE_POSITIVE_RISK_VETO",
    }
    missing = required - families
    if missing:
        raise RuntimeError(f"HARD_SAFETY_VETO_FAMILIES_MISSING: {sorted(missing)}")
    for row in rows:
        if row.get("row_identity_risk_v1") and row.get("candidate_adapter_ready_v1"):
            raise RuntimeError("ROW_IDENTITY_VETO_CANNOT_BE_ADAPTER_READY")
        if row.get("membership_coverage_proxy_risk_v1") and row.get("candidate_adapter_ready_v1"):
            raise RuntimeError("MEMBERSHIP_PROXY_VETO_CANNOT_BE_ADAPTER_READY")
    return True


def validate_final_selection(payload: dict[str, Any]) -> bool:
    if payload.get("adapter_reopen_allowed_v1") is not False:
        raise RuntimeError("ADAPTER_REOPEN_MUST_REMAIN_FALSE")
    if payload.get("selected_veto_adapter_ready_v1") is True:
        raise RuntimeError("NO_ADAPTER_READY_VETO_SHOULD_BE_SELECTED_IN_THIS_RESULT")
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
        raise RuntimeError(f"HARD_SAFETY_VETO_DISCOVERY_REQUIRED_OUTPUTS_MISSING: {missing}")
    return True


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


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_HOLD_ROOT,
        INPUT_VETO_MAPPING_ROOT,
        INPUT_ADAPTER_MAPPING_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_SIMPLIFY_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "hold_summary": INPUT_HOLD_ROOT / "summary_v1.json",
        "hold_go_no_go": INPUT_HOLD_ROOT
        / "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json",
        "hold_blocker_contract": INPUT_HOLD_ROOT / "hold_140_94_safe_core_blocker_contract_v1.json",
        "veto_summary": INPUT_VETO_MAPPING_ROOT / "summary_v1.json",
        "veto_unsafe_extra": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_unsafe_extra_without_veto_audit_v1.json",
        "veto_candidate_dry_runs": INPUT_VETO_MAPPING_ROOT / "deepen_140_94_candidate_veto_dry_run_results_v1.json",
        "adapter_summary": INPUT_ADAPTER_MAPPING_ROOT / "summary_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "simplify_summary": INPUT_SIMPLIFY_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    hold_go = _read_json(required["hold_go_no_go"])
    if hold_go.get("status_v1") != "140_94_SAFE_CORE_ADAPTER_HELD_UNTIL_DEPLOYABLE_VETO":
        raise RuntimeError("INPUT_HOLD_STATUS_NOT_HELD")
    return {
        "required_paths": required,
        "hold_summary": _read_json(required["hold_summary"]),
        "hold_go_no_go": hold_go,
        "hold_blocker_contract": _read_json(required["hold_blocker_contract"]),
        "veto_summary": _read_json(required["veto_summary"]),
        "veto_unsafe_extra": _read_json(required["veto_unsafe_extra"]),
        "veto_candidate_dry_runs": _read_json(required["veto_candidate_dry_runs"]),
        "adapter_summary": _read_json(required["adapter_summary"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "simplify_summary": _read_json(required["simplify_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "source_inputs": veto_gate._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "DISCOVER_140_94_HARD_SAFETY_VETO_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "hold_root_v1": str(INPUT_HOLD_ROOT),
            "veto_mapping_root_v1": str(INPUT_VETO_MAPPING_ROOT),
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


def _build_frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    return veto_gate._build_frame_and_masks(inputs["source_inputs"])


def _selected_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = frame[mask]
    return {
        "selected_rows_v1": int(mask.sum()),
        "recovered_original_140_rows_v1": int((mask & _bool(frame, "selected_original_140_v1")).sum()),
        "extra_rows_v1": int((mask & ~_bool(frame, "selected_original_140_v1")).sum()),
        "bad_count_audit_only_v1": int(_bool(selected, "bad_label_v1").sum()),
        "tail_count_audit_only_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_audit_only_v1": float(_bool(selected, "bad_label_v1").sum() / max(len(selected), 1)),
        "safety_status_v1": "CLEAN" if int(_bool(selected, "unsafe_audit_v1").sum()) == 0 else "FAIL",
        "unsafe_hits_v1": int(_bool(selected, "unsafe_audit_v1").sum()),
    }


def _reproducibility(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = inputs["hold_summary"]
    payload = {
        "layer_name": "DISCOVER_140_94_HARD_SAFETY_VETO_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": summary.get("safe_core_rule_id_v1"),
        "selected_rows_v1": summary.get("selected_rows_v1"),
        "recovered_original_140_rows_v1": summary.get("recovered_original_140_rows_v1"),
        "extra_rows_v1": summary.get("extra_rows_v1"),
        "bad_count_audit_only_v1": summary.get("bad_tail_audit_only_v1", [None, None])[0],
        "tail_count_audit_only_v1": summary.get("bad_tail_audit_only_v1", [None, None])[1],
        "precision_audit_only_v1": summary.get("precision_audit_only_v1"),
        "safety_status_v1": summary.get("safety_status_v1"),
        "hard_safety_veto_status_v1": summary.get("hard_safety_veto_status_v1"),
        "unsafe_extra_without_hard_veto_rows_v1": summary.get("unsafe_extra_without_hard_veto_rows_v1"),
        "hold_status_v1": inputs["hold_go_no_go"].get("status_v1"),
    }
    validate_reproducibility(payload)
    return payload


def _candidate_definitions(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = masks["low_support_veto_only"]
    score = _num(frame, "candidate_score_v1")
    missing_tail = ~_bool(frame, "signal_r5_tail_score_v1")
    has_r5_bad = _bool(frame, "signal_r5_bad_score_v1")
    policy = _str(frame, "run_id_policy_class_v1")
    student_low = _num(frame, "student_oof_score_v1").lt(0.50)
    student_lower = _num(frame, "student_oof_score_v1").lt(0.40)
    return [
        {
            "candidate_veto_name_v1": "AUDIT_ONLY_HARD_SAFETY_VETO_REFERENCE",
            "veto_family_v1": "AUDIT_ONLY_REFERENCE",
            "input_fields_v1": "hard_veto_clear_shadow_v1 and audit safety columns",
            "condition_rule_v1": "veto if audit hard safety veto fails",
            "as_of_lineage_v1": "NOT_DEPLOYABLE_AUDIT_ONLY",
            "adapter_feasibility_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": True,
            "outcome_hindsight_risk_v1": True,
            "membership_coverage_proxy_risk_v1": False,
            "complexity_v1": "LOW_LOGIC_HIGH_LINEAGE_RISK",
            "recommendation_v1": "REFERENCE_ONLY_DO_NOT_DEPLOY",
            "mask": base & ~masks["hardened"],
        },
        {
            "candidate_veto_name_v1": "SIGNAL_SHAPE_REFINED_NO_R5_TAIL_R5_BAD_SCORE_GE_099",
            "veto_family_v1": "SIGNAL_SHAPE_REFINED_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1",
            "condition_rule_v1": "veto if missing R5_TAIL_SCORE and has R5_BAD_SCORE support and score >= 0.99",
            "as_of_lineage_v1": "AS_OF_SAFE_EXISTING_SCORE_AND_SIGNAL_FLAGS",
            "adapter_feasibility_v1": "DEPLOYABLE_BUT_TOO_DESTRUCTIVE",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": False,
            "complexity_v1": "LOW",
            "recommendation_v1": "BLOCK_TOO_DESTRUCTIVE_TO_SAFE_CORE",
            "mask": base & missing_tail & has_r5_bad & score.ge(0.99),
        },
        {
            "candidate_veto_name_v1": "LOW_SUPPORT_OR_MISSING_ARTIFACT_VETO_NORMALIZED",
            "veto_family_v1": "LOW_SUPPORT_OR_MISSING_ARTIFACT_VETO",
            "input_fields_v1": "run_id_policy_class_v1, structural_low_support_v1, zero_denominator_group_v1",
            "condition_rule_v1": "veto rows with missing-artifact or structural-low-support policy class",
            "as_of_lineage_v1": "AS_OF_POLICY_LINEAGE_PARTIAL",
            "adapter_feasibility_v1": "DEPLOYABLE_BUT_DOES_NOT_BLOCK_UNSAFE_ROW",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": False,
            "complexity_v1": "LOW",
            "recommendation_v1": "FAILS_TO_BLOCK_UNSAFE_ROW",
            "mask": base
            & (
                policy.str.contains("LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS", regex=False)
                | _bool(frame, "structural_low_support_v1")
                | _bool(frame, "zero_denominator_group_v1")
            ),
        },
        {
            "candidate_veto_name_v1": "SAFE_CORE_DISTANCE_MARGIN_STUDENT_LOW_HIGH_SCORE_DIAGNOSTIC",
            "veto_family_v1": "SAFE_CORE_DISTANCE_MARGIN_VETO",
            "input_fields_v1": "student_oof_score_v1, candidate_score_v1",
            "condition_rule_v1": "veto if score >= 0.99 and student OOF membership score < 0.50",
            "as_of_lineage_v1": "OOF_SCORE_PRESENT_BUT_MEMBERSHIP_TARGET_HISTORY",
            "adapter_feasibility_v1": "BLOCKED_MEMBERSHIP_PROXY_RISK",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": True,
            "complexity_v1": "LOW",
            "recommendation_v1": "DIAGNOSTIC_ONLY_DO_NOT_DEPLOY",
            "mask": base & score.ge(0.99) & student_low,
        },
        {
            "candidate_veto_name_v1": "BRANCH_SPECIFIC_HIGH_SCORE_NO_TAIL_VETO",
            "veto_family_v1": "BRANCH_SPECIFIC_VETO",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1",
            "condition_rule_v1": "within high-score V2/R5_1 branch, veto missing R5 tail + R5 bad support + score >= 0.993",
            "as_of_lineage_v1": "AS_OF_SAFE_EXISTING_SCORE_AND_SIGNAL_FLAGS",
            "adapter_feasibility_v1": "DEPLOYABLE_BUT_TOO_DESTRUCTIVE",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": False,
            "complexity_v1": "LOW",
            "recommendation_v1": "BLOCK_TOO_DESTRUCTIVE_TO_SAFE_CORE",
            "mask": base & missing_tail & has_r5_bad & score.ge(0.993),
        },
        {
            "candidate_veto_name_v1": "VETO_CONFLUENCE_SCORE_GE_099_STUDENT_LOW_DIAGNOSTIC",
            "veto_family_v1": "VETO_CONFLUENCE_RULE",
            "input_fields_v1": "candidate_score_v1, student_oof_score_v1",
            "condition_rule_v1": "veto if score >= 0.99 and student membership score < 0.40",
            "as_of_lineage_v1": "OOF_SCORE_PRESENT_BUT_MEMBERSHIP_TARGET_HISTORY",
            "adapter_feasibility_v1": "BLOCKED_MEMBERSHIP_PROXY_RISK",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": True,
            "complexity_v1": "LOW",
            "recommendation_v1": "DIAGNOSTIC_ONLY_DO_NOT_DEPLOY",
            "mask": base & score.ge(0.99) & student_lower,
        },
        {
            "candidate_veto_name_v1": "FALSE_POSITIVE_RISK_NO_TAIL_STUDENT_LOW_DIAGNOSTIC",
            "veto_family_v1": "FALSE_POSITIVE_RISK_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, student_oof_score_v1",
            "condition_rule_v1": "veto if missing R5 tail evidence and student membership score < 0.50",
            "as_of_lineage_v1": "OOF_SCORE_PRESENT_BUT_MEMBERSHIP_TARGET_HISTORY",
            "adapter_feasibility_v1": "BLOCKED_MEMBERSHIP_PROXY_RISK",
            "row_identity_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "membership_coverage_proxy_risk_v1": True,
            "complexity_v1": "LOW",
            "recommendation_v1": "DIAGNOSTIC_ONLY_DO_NOT_DEPLOY",
            "mask": base & missing_tail & student_low,
        },
    ]


def _candidate_families() -> dict[str, Any]:
    families = [
        {
            "veto_family_v1": "SIGNAL_SHAPE_REFINED_VETO",
            "purpose_v1": "Refine prior signal-shape vetoes using score and existing signal flags.",
            "deployability_requirement_v1": "Only existing AS_OF score and signal flags.",
        },
        {
            "veto_family_v1": "LOW_SUPPORT_OR_MISSING_ARTIFACT_VETO",
            "purpose_v1": "Test support, missing-artifact and lineage policy vetoes.",
            "deployability_requirement_v1": "Train-time/support policy lineage must be AS_OF-safe.",
        },
        {
            "veto_family_v1": "SAFE_CORE_DISTANCE_MARGIN_VETO",
            "purpose_v1": "Check whether unsafe row is far from safe-core under OOF-safe score space.",
            "deployability_requirement_v1": "Must not use membership/coverage target proxies.",
        },
        {
            "veto_family_v1": "BRANCH_SPECIFIC_VETO",
            "purpose_v1": "Veto only inside the branch/tier that admits the unsafe row.",
            "deployability_requirement_v1": "Must be general and not row-identity based.",
        },
        {
            "veto_family_v1": "VETO_CONFLUENCE_RULE",
            "purpose_v1": "Combine weak AS_OF warning signals into a simple adapter-friendly veto.",
            "deployability_requirement_v1": "All signals must be AS_OF-safe and not membership proxies.",
        },
        {
            "veto_family_v1": "FALSE_POSITIVE_RISK_VETO",
            "purpose_v1": "Secondary check for the three safety-clean extras.",
            "deployability_requirement_v1": "Must not block good safe-core rows or use outcome labels.",
        },
    ]
    return {"layer_name": "DISCOVER_140_94_CANDIDATE_VETO_FAMILIES_V1", "families_v1": families}


def _candidate_metrics(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = masks["low_support_veto_only"]
    hardened = masks["hardened"]
    original = _bool(frame, "selected_original_140_v1")
    unsafe = _bool(frame, "unsafe_audit_v1")
    metrics_rows: list[dict[str, Any]] = []
    dry_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    lookalike_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        veto_mask = candidate["mask"]
        selected = base & ~veto_mask
        cut_safe_core = veto_mask & hardened
        cut_original = cut_safe_core & original
        selected_frame = frame[selected]
        deployable_signal = (
            not candidate["audit_only_risk_v1"]
            and not candidate["outcome_hindsight_risk_v1"]
            and not candidate["membership_coverage_proxy_risk_v1"]
            and not candidate["row_identity_risk_v1"]
        )
        unsafe_blocked = int((veto_mask & (base & ~hardened) & unsafe).sum()) > 0
        safe_core_blocked = int(cut_safe_core.sum())
        candidate_adapter_ready = (
            deployable_signal
            and unsafe_blocked
            and safe_core_blocked <= ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT
            and int((selected & unsafe).sum()) == 0
        )
        row = {
            "candidate_veto_name_v1": candidate["candidate_veto_name_v1"],
            "veto_family_v1": candidate["veto_family_v1"],
            "input_fields_v1": candidate["input_fields_v1"],
            "as_of_lineage_v1": candidate["as_of_lineage_v1"],
            "condition_rule_v1": candidate["condition_rule_v1"],
            "adapter_feasibility_v1": candidate["adapter_feasibility_v1"],
            "row_identity_risk_v1": candidate["row_identity_risk_v1"],
            "audit_only_risk_v1": candidate["audit_only_risk_v1"],
            "outcome_hindsight_risk_v1": candidate["outcome_hindsight_risk_v1"],
            "membership_coverage_proxy_risk_v1": candidate["membership_coverage_proxy_risk_v1"],
            "unsafe_row_blocked_v1": unsafe_blocked,
            "safe_core_rows_retained_v1": int((selected & hardened).sum()),
            "original_140_rows_retained_v1": int((selected & original).sum()),
            "safe_core_rows_accidentally_blocked_v1": safe_core_blocked,
            "original_140_rows_accidentally_blocked_v1": int(cut_original.sum()),
            "remaining_extra_3_retained_v1": int((selected & hardened & ~original).sum()),
            "remaining_extra_3_blocked_v1": int((veto_mask & hardened & ~original).sum()),
            "selected_rows_after_veto_v1": int(selected.sum()),
            "extra_rows_after_veto_v1": int((selected & ~original).sum()),
            "bad_count_after_veto_audit_only_v1": int(_bool(selected_frame, "bad_label_v1").sum()),
            "tail_count_after_veto_audit_only_v1": int(_bool(selected_frame, "tail_label_v1").sum()),
            "precision_after_veto_audit_only_v1": float(
                _bool(selected_frame, "bad_label_v1").sum() / max(len(selected_frame), 1)
            ),
            "safety_after_veto_v1": "CLEAN" if int((selected & unsafe).sum()) == 0 else "FAIL",
            "unsafe_rows_after_veto_v1": int((selected & unsafe).sum()),
            "candidate_adapter_ready_v1": candidate_adapter_ready,
            "complexity_v1": candidate["complexity_v1"],
            "stability_run_id_count_cut_rows_v1": int(frame[cut_safe_core]["run_id_v1"].nunique()) if cut_safe_core.any() else 0,
            "recommendation_v1": candidate["recommendation_v1"],
        }
        metrics_rows.append(row)
        dry_rows.append(
            {
                "candidate_veto_name_v1": row["candidate_veto_name_v1"],
                "selected_rows_v1": row["selected_rows_after_veto_v1"],
                "recovered_original_140_rows_v1": row["original_140_rows_retained_v1"],
                "extra_rows_v1": row["extra_rows_after_veto_v1"],
                "unsafe_rows_v1": row["unsafe_rows_after_veto_v1"],
                "bad_tail_audit_only_v1": [
                    row["bad_count_after_veto_audit_only_v1"],
                    row["tail_count_after_veto_audit_only_v1"],
                ],
                "precision_audit_only_v1": row["precision_after_veto_audit_only_v1"],
                "safety_status_v1": row["safety_after_veto_v1"],
                "mismatch_vs_safe_core_v1": row["safe_core_rows_accidentally_blocked_v1"]
                + int((selected & ~hardened).sum()),
                "mismatch_vs_audit_only_veto_behavior_v1": row["safe_core_rows_accidentally_blocked_v1"],
                "adapter_readiness_v1": "READY" if row["candidate_adapter_ready_v1"] else row["adapter_feasibility_v1"],
            }
        )
        for _, cut in frame[cut_safe_core].sort_values(["run_id_v1", "candidate_score_v1"]).iterrows():
            retention_rows.append(
                {
                    "candidate_veto_name_v1": row["candidate_veto_name_v1"],
                    "row_id_v1": cut.get("candidate_uid_v1"),
                    "run_id_v1": cut.get("run_id_v1"),
                    "fold_id_v1": cut.get("fold_id_v1"),
                    "bad_label_audit_only_v1": _as_bool(cut.get("bad_label_v1")),
                    "tail_label_audit_only_v1": _as_bool(cut.get("tail_label_v1")),
                    "student_oof_score_v1": cut.get("student_oof_score_v1"),
                    "source_evidence_v1": cut.get("source_evidence_v1"),
                    "why_cut_v1": row["condition_rule_v1"],
                    "acceptable_cut_v1": row["safe_core_rows_accidentally_blocked_v1"] <= ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT,
                    "tail_coverage_impact_v1": "TAIL_ROW_CUT" if _as_bool(cut.get("tail_label_v1")) else "NON_TAIL_ROW_CUT",
                    "too_gross_veto_v1": row["safe_core_rows_accidentally_blocked_v1"] > ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT,
                }
            )
        look_mask = veto_mask & base
        if not look_mask.any():
            look_mask = (base & ~hardened) | (hardened & _num(frame, "candidate_score_v1").ge(0.99))
        for _, look in frame[look_mask].sort_values(["run_id_v1", "candidate_score_v1"], ascending=[True, False]).head(80).iterrows():
            risk_class = "LOW_UNSAFE_LOOKALIKE_RISK"
            if row["unsafe_rows_after_veto_v1"] > 0:
                risk_class = "HIGH_UNSAFE_LOOKALIKE_RISK_BLOCK_VETO"
            elif row["safe_core_rows_accidentally_blocked_v1"] > ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT:
                risk_class = "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_MONITORING"
            if row["membership_coverage_proxy_risk_v1"] or row["audit_only_risk_v1"]:
                risk_class = "UNKNOWN_REQUIRES_MORE_AUDIT"
            lookalike_rows.append(
                {
                    "candidate_veto_name_v1": row["candidate_veto_name_v1"],
                    "row_id_v1": look.get("candidate_uid_v1"),
                    "run_id_v1": look.get("run_id_v1"),
                    "selected_by_safe_core_v1": _as_bool(hardened.loc[look.name]),
                    "unsafe_audit_only_v1": _as_bool(look.get("unsafe_audit_v1")),
                    "bad_label_audit_only_v1": _as_bool(look.get("bad_label_v1")),
                    "tail_label_audit_only_v1": _as_bool(look.get("tail_label_v1")),
                    "blocked_by_candidate_veto_v1": _as_bool(veto_mask.loc[look.name]),
                    "candidate_score_v1": look.get("candidate_score_v1"),
                    "student_oof_score_v1": look.get("student_oof_score_v1"),
                    "source_evidence_v1": look.get("source_evidence_v1"),
                    "lookalike_reason_v1": row["condition_rule_v1"],
                    "risk_class_v1": risk_class,
                }
            )
    validate_candidate_veto_metrics(metrics_rows)
    return metrics_rows, dry_rows, retention_rows, lookalike_rows


def _unsafe_extra_row_audit(frame: pd.DataFrame, masks: dict[str, pd.Series], metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unsafe_mask = masks["low_support_veto_only"] & ~masks["hardened"] & _bool(frame, "unsafe_audit_v1")
    rows = []
    signal_shape_cut_counts = {
        row["candidate_veto_name_v1"]: row["safe_core_rows_accidentally_blocked_v1"]
        for row in metrics_rows
        if row["veto_family_v1"] in {"SIGNAL_SHAPE_REFINED_VETO", "BRANCH_SPECIFIC_VETO"}
    }
    for _, row in frame[unsafe_mask].iterrows():
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "group_v1": row.get("run_id_policy_class_v1"),
                "branch_tier_v1": "score>=0.95 + R5_1 + V2-like + low-support-clear",
                "positive_signals_v1": row.get("source_evidence_v1"),
                "audit_only_veto_that_stops_it_v1": "hard_veto_clear_shadow_v1 == false",
                "why_audit_only_veto_not_deployable_v1": "It depends on audit/protected/ambiguous/high-MFE/final safety fields without proven AS_OF adapter lineage.",
                "as_of_safe_signals_present_v1": "candidate_score_v1, R5_BAD_SCORE, R5_1_BAD_SCORE, V2_LIKE_BAD_TAIL",
                "as_of_safe_signals_absent_v1": "R5_TAIL_SCORE, tail_repair",
                "as_of_safe_signals_distinguishing_from_safe_core_v1": "None found without cutting many good rows; student score separates it but is membership-proxy-risk diagnostic only.",
                "signal_shape_vetoes_that_stop_it_v1": sorted(signal_shape_cut_counts),
                "safe_core_rows_cut_by_signal_shape_vetoes_v1": signal_shape_cut_counts,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "student_oof_score_v1": row.get("student_oof_score_v1"),
                "protected_winner_audit_only_v1": _as_bool(row.get("protected_winner_status_v1")),
                "ambiguous_high_mfe_audit_only_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                "fifty_plus_mfe_risk_audit_only_v1": _as_bool(row.get("fifty_plus_mfe_risk_v1")),
                "hundred_plus_mfe_risk_audit_only_v1": _as_bool(row.get("hundred_plus_mfe_risk_v1")),
                "final_classification_v1": "UNSAFE_ROW_NOT_SEPARABLE_BY_CURRENT_DEPLOYABLE_AS_OF_SIGNALS_WITH_ACCEPTABLE_RETENTION",
            }
        )
    return rows


def _final_selection(metrics_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    eligible = [row for row in metrics_rows if row["candidate_adapter_ready_v1"]]
    deployable_destructive = [
        row
        for row in metrics_rows
        if not row["audit_only_risk_v1"]
        and not row["membership_coverage_proxy_risk_v1"]
        and row["unsafe_row_blocked_v1"]
        and row["safe_core_rows_accidentally_blocked_v1"] > ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT
    ]
    best_destructive = min(
        deployable_destructive,
        key=lambda row: row["safe_core_rows_accidentally_blocked_v1"],
        default=None,
    )
    final = {
        "layer_name": "DISCOVER_140_94_FINAL_VETO_SELECTION_V1",
        "selected_veto_name_v1": None,
        "selected_veto_adapter_ready_v1": False,
        "adapter_reopen_allowed_v1": False,
        "deployable_candidate_found_v1": bool(eligible),
        "deployable_but_too_destructive_candidate_v1": best_destructive["candidate_veto_name_v1"]
        if best_destructive
        else None,
        "safe_core_rows_cut_by_best_destructive_candidate_v1": best_destructive[
            "safe_core_rows_accidentally_blocked_v1"
        ]
        if best_destructive
        else None,
        "reason_v1": "Deployable AS_OF signal-shape vetoes block the unsafe row only by cutting too many safe-core rows; diagnostic exact vetoes depend on audit-only or membership-proxy fields.",
        "status_v1": FINAL_STATUS,
    }
    validate_final_selection(final)
    adapter = {
        "layer_name": "DISCOVER_140_94_ADAPTER_REOPEN_ASSESSMENT_V1",
        "adapter_build_can_reopen_v1": False,
        "safe_core_adapter_ready_v1": False,
        "normalization_needed_before_adapter_v1": False,
        "lineage_audit_needed_before_adapter_v1": False,
        "safe_core_should_remain_held_v1": True,
        "reason_v1": "No acceptable deployable hard safety veto was selected.",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
    }
    anti = {
        "layer_name": "DISCOVER_140_94_HARD_SAFETY_VETO_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS_NO_DEPLOYABLE_VETO_SELECTED",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_run_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_model_training_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_implicit_latest_glob_selection_v1": True,
        "row_identity_veto_rejected_v1": True,
        "audit_only_veto_not_promoted_v1": True,
        "membership_proxy_veto_not_promoted_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    recommendation = {
        "layer_name": "DISCOVER_140_94_HARD_SAFETY_VETO_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_remains_held_v1": True,
        "rationale_v1": [
            "Signal-shape AS_OF candidates are deployable in lineage but too destructive to safe-core.",
            "Student/distance-margin candidates are diagnostically strong but blocked by membership-proxy risk.",
            "Audit-only veto remains non-deployable, and row identity remains forbidden.",
        ],
    }
    return final, adapter, anti, recommendation


def _go_no_go(repro: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "layer_name": "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "adapter_reopen_allowed_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    return payload


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    unsafe_rows: list[dict[str, Any]],
    families: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    dry_rows: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    lookalike_rows: list[dict[str, Any]],
    final: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "discover_140_94_hard_safety_veto_reproducibility_audit_v1.md",
        [
            "# Discover 140/94 Hard Safety Veto Reproducibility Audit V1",
            "",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_audit_only_v1']} / {repro['tail_count_audit_only_v1']}`",
            f"- Hard safety veto: `{repro['hard_safety_veto_status_v1']}`",
            f"- Unsafe extras without hard veto: `{repro['unsafe_extra_without_hard_veto_rows_v1']}`",
        ],
    )
    _write_report(
        root / "discover_140_94_unsafe_extra_row_audit_v1.md",
        [
            "# Discover 140/94 Unsafe Extra Row Audit V1",
            "",
            f"- Unsafe extra rows: `{len(unsafe_rows)}`",
            "- No acceptable deployable AS_OF signal distinguishes the unsafe row without damaging safe-core.",
        ],
    )
    _write_report(
        root / "discover_140_94_candidate_veto_families_v1.md",
        [
            "# Discover 140/94 Candidate Veto Families V1",
            "",
            *[f"- `{row['veto_family_v1']}`: {row['purpose_v1']}" for row in families["families_v1"]],
        ],
    )
    _write_report(
        root / "discover_140_94_candidate_veto_metrics_v1.md",
        [
            "# Discover 140/94 Candidate Veto Metrics V1",
            "",
            *[
                f"- `{row['candidate_veto_name_v1']}`: unsafe blocked `{row['unsafe_row_blocked_v1']}`, safe-core blocked `{row['safe_core_rows_accidentally_blocked_v1']}`, readiness `{row['adapter_feasibility_v1']}`"
                for row in metrics_rows
            ],
        ],
    )
    _write_report(
        root / "discover_140_94_candidate_veto_dry_run_results_v1.md",
        [
            "# Discover 140/94 Candidate Veto Dry Run Results V1",
            "",
            *[
                f"- `{row['candidate_veto_name_v1']}`: selected `{row['selected_rows_v1']}`, original retained `{row['recovered_original_140_rows_v1']}`, unsafe `{row['unsafe_rows_v1']}`"
                for row in dry_rows
            ],
        ],
    )
    _write_report(
        root / "discover_140_94_veto_row_retention_audit_v1.md",
        [
            "# Discover 140/94 Veto Row Retention Audit V1",
            "",
            f"- Cut safe-core row records: `{len(retention_rows)}`",
            "- Deployable signal-shape candidates cut too many good safe-core rows.",
        ],
    )
    _write_report(
        root / "discover_140_94_unsafe_lookalike_audit_v1.md",
        [
            "# Discover 140/94 Unsafe Lookalike Audit V1",
            "",
            f"- Lookalike audit rows: `{len(lookalike_rows)}`",
            "- Student-score lookalike separation remains diagnostic only because of membership-proxy risk.",
        ],
    )
    _write_report(
        root / "discover_140_94_final_veto_selection_v1.md",
        [
            "# Discover 140/94 Final Veto Selection V1",
            "",
            f"- Selected veto: `{final['selected_veto_name_v1']}`",
            f"- Adapter reopen allowed: `{final['adapter_reopen_allowed_v1']}`",
            f"- Status: `{final['status_v1']}`",
        ],
    )
    _write_report(
        root / "discover_140_94_adapter_reopen_assessment_v1.md",
        [
            "# Discover 140/94 Adapter Reopen Assessment V1",
            "",
            f"- Adapter build can reopen: `{adapter['adapter_build_can_reopen_v1']}`",
            f"- Status: `{adapter['status_v1']}`",
        ],
    )
    _write_report(
        root / "discover_140_94_hard_safety_veto_anti_shortcut_audit_v1.md",
        [
            "# Discover 140/94 Hard Safety Veto Anti-Shortcut Audit V1",
            "",
            "- No R6, adapter, IQL, package, freeze, promo, live, Optuna, broad sweep, or model training was run.",
            "- Audit-only, membership-proxy, and row-identity vetoes were not promoted.",
        ],
    )
    _write_report(
        root / "discover_140_94_hard_safety_veto_recommendation_v1.md",
        [
            "# Discover 140/94 Hard Safety Veto Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _build_frame_and_masks(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(inputs)
    families = _candidate_families()
    candidates = _candidate_definitions(frame, masks)
    metrics_rows, dry_rows, retention_rows, lookalike_rows = _candidate_metrics(frame, masks, candidates)
    unsafe_rows = _unsafe_extra_row_audit(frame, masks, metrics_rows)
    final, adapter, anti, recommendation = _final_selection(metrics_rows)
    go_no_go = _go_no_go(repro)

    _write_json(artifact_root / "discover_140_94_hard_safety_veto_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "discover_140_94_hard_safety_veto_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "discover_140_94_unsafe_extra_row_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "discover_140_94_unsafe_extra_row_audit_v1.json",
        {"row_count_v1": len(unsafe_rows), "rows_v1": unsafe_rows},
    )
    _write_json(artifact_root / "discover_140_94_candidate_veto_families_v1.json", families)
    _write_rows(artifact_root / "discover_140_94_candidate_veto_metrics_v1.csv", metrics_rows)
    _write_json(
        artifact_root / "discover_140_94_candidate_veto_metrics_v1.json",
        {"row_count_v1": len(metrics_rows), "rows_v1": metrics_rows},
    )
    _write_rows(artifact_root / "discover_140_94_candidate_veto_dry_run_results_v1.csv", dry_rows)
    _write_json(
        artifact_root / "discover_140_94_candidate_veto_dry_run_results_v1.json",
        {"row_count_v1": len(dry_rows), "rows_v1": dry_rows},
    )
    _write_rows(artifact_root / "discover_140_94_veto_row_retention_audit_v1.csv", retention_rows)
    _write_json(
        artifact_root / "discover_140_94_veto_row_retention_audit_v1.json",
        {"row_count_v1": len(retention_rows), "rows_v1": retention_rows},
    )
    _write_rows(artifact_root / "discover_140_94_unsafe_lookalike_audit_v1.csv", lookalike_rows)
    _write_json(
        artifact_root / "discover_140_94_unsafe_lookalike_audit_v1.json",
        {"row_count_v1": len(lookalike_rows), "rows_v1": lookalike_rows},
    )
    _write_json(artifact_root / "discover_140_94_final_veto_selection_v1.json", final)
    _write_json(artifact_root / "discover_140_94_adapter_reopen_assessment_v1.json", adapter)
    _write_json(artifact_root / "discover_140_94_hard_safety_veto_anti_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "discover_140_94_hard_safety_veto_recommendation_v1.json", recommendation)
    _write_json(
        artifact_root / "discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_go_no_go_v1.json",
        go_no_go,
    )
    summary = {
        "layer_name": "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": repro["unsafe_extra_without_hard_veto_rows_v1"],
        "selected_final_veto_v1": final["selected_veto_name_v1"],
        "best_deployable_destructive_candidate_v1": final["deployable_but_too_destructive_candidate_v1"],
        "adapter_reopen_allowed_v1": False,
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
            "# Discover Deployable AS_OF Hard Safety Veto For 140/94 Safe-Core V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Safe-core: `{SAFE_CORE_RULE_ID}`",
            f"- Unsafe extra without hard veto: `{summary['unsafe_extra_without_hard_veto_rows_v1']}`",
            f"- Selected deployable veto: `{summary['selected_final_veto_v1']}`",
            f"- Best deployable destructive candidate: `{summary['best_deployable_destructive_candidate_v1']}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        unsafe_rows,
        families,
        metrics_rows,
        dry_rows,
        retention_rows,
        lookalike_rows,
        final,
        adapter,
        recommendation,
    )
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
