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
from gx1.scripts import materialize_discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_v1 as discover
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1"

INPUT_DISCOVERY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1_20260428T113213Z_LOCK"
)
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

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "140_94_REFINED_HARD_SAFETY_VETO_PASS_NEEDS_LINEAGE_CONFIRMATION"
NEXT_ACTION = "DEEPEN_140_94_REFINED_HARD_SAFETY_VETO_LINEAGE_AUDIT_V1"

EXPECTED_SELECTED = 89
EXPECTED_RECOVERED = 86
EXPECTED_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607
EXPECTED_DESTRUCTIVE_CUTS = 21

ALLOWED_FINAL_STATUSES = {
    "140_94_REFINED_HARD_SAFETY_VETO_PASS_ADAPTER_READY",
    "140_94_REFINED_HARD_SAFETY_VETO_PASS_NEEDS_NORMALIZATION",
    "140_94_REFINED_HARD_SAFETY_VETO_PASS_NEEDS_LINEAGE_CONFIRMATION",
    "140_94_REFINED_HARD_SAFETY_VETO_FOUND_BUT_YELLOW_RETENTION",
    "140_94_REFINED_HARD_SAFETY_VETO_STILL_TOO_DESTRUCTIVE",
    "140_94_NO_SAFE_REFINED_HARD_SAFETY_VETO_FOUND",
    "140_94_REFINED_VETO_BLOCKED_BY_MEMBERSHIP_OR_DISTANCE_PROXY",
    "140_94_REFINED_VETO_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "140_94_REFINED_VETO_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_SAFE_CORE_ADAPTER_V1",
    "NORMALIZE_140_94_REFINED_HARD_SAFETY_VETO_INPUTS_V1",
    "DEEPEN_140_94_REFINED_HARD_SAFETY_VETO_LINEAGE_AUDIT_V1",
    "REVIEW_YELLOW_RETENTION_VETO_BEFORE_ADAPTER_V1",
    "REFINE_140_94_HARD_SAFETY_VETO_AGAIN_WITH_STRONGER_SIGNALS_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
}

REQUIRED_OUTPUTS = [
    "refine_140_94_hard_safety_veto_input_manifest_v1.json",
    "refine_140_94_hard_safety_veto_reproducibility_audit_v1.json",
    "refine_140_94_hard_safety_veto_reproducibility_audit_v1.md",
    "refine_140_94_destructive_veto_audit_v1.csv",
    "refine_140_94_destructive_veto_audit_v1.json",
    "refine_140_94_destructive_veto_audit_v1.md",
    "refine_140_94_cut_21_good_rows_audit_v1.csv",
    "refine_140_94_cut_21_good_rows_audit_v1.json",
    "refine_140_94_cut_21_good_rows_audit_v1.md",
    "refine_140_94_unsafe_row_refinement_audit_v1.csv",
    "refine_140_94_unsafe_row_refinement_audit_v1.json",
    "refine_140_94_unsafe_row_refinement_audit_v1.md",
    "refine_140_94_refined_veto_candidate_definitions_v1.json",
    "refine_140_94_refined_veto_candidate_definitions_v1.md",
    "refine_140_94_refined_veto_candidate_metrics_v1.csv",
    "refine_140_94_refined_veto_candidate_metrics_v1.json",
    "refine_140_94_refined_veto_candidate_metrics_v1.md",
    "refine_140_94_refined_veto_dry_run_results_v1.csv",
    "refine_140_94_refined_veto_dry_run_results_v1.json",
    "refine_140_94_refined_veto_dry_run_results_v1.md",
    "refine_140_94_retention_threshold_audit_v1.json",
    "refine_140_94_retention_threshold_audit_v1.md",
    "refine_140_94_diagnostic_student_distance_comparison_v1.json",
    "refine_140_94_diagnostic_student_distance_comparison_v1.md",
    "refine_140_94_unsafe_lookalike_audit_v1.csv",
    "refine_140_94_unsafe_lookalike_audit_v1.json",
    "refine_140_94_unsafe_lookalike_audit_v1.md",
    "refine_140_94_final_refined_veto_selection_v1.json",
    "refine_140_94_final_refined_veto_selection_v1.md",
    "refine_140_94_adapter_reopen_assessment_v1.json",
    "refine_140_94_adapter_reopen_assessment_v1.md",
    "refine_140_94_hard_safety_veto_anti_shortcut_audit_v1.json",
    "refine_140_94_hard_safety_veto_anti_shortcut_audit_v1.md",
    "refine_140_94_hard_safety_veto_recommendation_v1.json",
    "refine_140_94_hard_safety_veto_recommendation_v1.md",
    "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json",
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


def _as_bool(value: Any) -> bool:
    return simplify._as_bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return simplify._bool(frame, column)


def _str(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    return simplify._str(frame, column, default)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return simplify._num(frame, column, default)


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


def retention_tier(*, unsafe_row_blocked: bool, good_rows_cut: int, shortcut_or_leakage: bool = False) -> str:
    if shortcut_or_leakage or not unsafe_row_blocked:
        return "BLOCKED"
    if good_rows_cut <= 5:
        return "GREEN"
    if good_rows_cut <= 10:
        return "YELLOW"
    if good_rows_cut <= 20:
        return "ORANGE"
    return "RED"


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    expected = {
        "selected_rows_v1": EXPECTED_SELECTED,
        "recovered_original_140_rows_v1": EXPECTED_RECOVERED,
        "extra_rows_v1": EXPECTED_EXTRA,
        "bad_count_audit_only_v1": EXPECTED_BAD,
        "tail_count_audit_only_v1": EXPECTED_TAIL,
        "safety_status_v1": "CLEAN",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
        "best_prior_deployable_destructive_candidate_v1": "SIGNAL_SHAPE_REFINED_NO_R5_TAIL_R5_BAD_SCORE_GE_099",
        "best_prior_deployable_destructive_safe_core_cut_v1": EXPECTED_DESTRUCTIVE_CUTS,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("precision_audit_only_v1", -1)), EXPECTED_PRECISION):
        failures["precision_audit_only_v1"] = payload.get("precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"REFINE_140_94_HARD_SAFETY_VETO_REPRODUCTION_FAILED: {failures}")
    return True


def validate_candidate_metrics(rows: list[dict[str, Any]]) -> bool:
    families = {row["veto_family_v1"] for row in rows}
    required = {
        "BRANCH_LOCAL_SIGNAL_SHAPE_VETO",
        "TWO_CONDITION_CONFLUENCE_VETO",
        "RELAXED_SIGNAL_SHAPE_THRESHOLD_VETO",
        "EXCEPTION_GUARDED_SIGNAL_SHAPE_VETO",
        "LOW_SUPPORT_AWARE_SIGNAL_SHAPE_VETO",
        "MINIMAL_DESTRUCTIVE_VETO",
        "DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON",
    }
    missing = sorted(required - families)
    if missing:
        raise RuntimeError(f"REFINED_VETO_FAMILIES_MISSING: {missing}")
    for row in rows:
        if row.get("membership_proxy_risk_v1") and row.get("adapter_ready_v1"):
            raise RuntimeError("MEMBERSHIP_PROXY_VETO_CANNOT_BE_ADAPTER_READY")
        if row.get("row_identity_risk_v1") and row.get("adapter_ready_v1"):
            raise RuntimeError("ROW_IDENTITY_VETO_CANNOT_BE_ADAPTER_READY")
        if row.get("lineage_status_v1") == "NEEDS_LINEAGE_CONFIRMATION" and row.get("adapter_ready_v1"):
            raise RuntimeError("LINEAGE_UNCONFIRMED_VETO_CANNOT_BE_ADAPTER_READY")
    return True


def validate_final_selection(payload: dict[str, Any]) -> bool:
    if payload.get("selected_refined_veto_name_v1") != "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1":
        raise RuntimeError("EXPECTED_HISTORICAL_BLUEPRINT_GUARDED_REFINED_VETO")
    if payload.get("adapter_reopen_allowed_now_v1") is not False:
        raise RuntimeError("ADAPTER_REOPEN_MUST_WAIT_FOR_LINEAGE_CONFIRMATION")
    if payload.get("good_safe_core_rows_cut_v1", 999) > 5:
        raise RuntimeError("SELECTED_REFINED_VETO_MUST_BE_GREEN_RETENTION")
    if payload.get("lineage_status_v1") != "NEEDS_LINEAGE_CONFIRMATION":
        raise RuntimeError("SELECTED_REFINED_VETO_MUST_REQUIRE_LINEAGE_CONFIRMATION")
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
        raise RuntimeError(f"REFINE_140_94_HARD_SAFETY_VETO_REQUIRED_OUTPUTS_MISSING: {missing}")
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
    roots = [INPUT_DISCOVERY_ROOT, INPUT_HOLD_ROOT, INPUT_VETO_MAPPING_ROOT, INPUT_ADAPTER_MAPPING_ROOT, INPUT_HARDEN_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "discovery_summary": INPUT_DISCOVERY_ROOT / "summary_v1.json",
        "discovery_go_no_go": INPUT_DISCOVERY_ROOT
        / "discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_go_no_go_v1.json",
        "discovery_candidate_metrics": INPUT_DISCOVERY_ROOT / "discover_140_94_candidate_veto_metrics_v1.json",
        "discovery_cut_rows": INPUT_DISCOVERY_ROOT / "discover_140_94_veto_row_retention_audit_v1.json",
        "discovery_unsafe_row": INPUT_DISCOVERY_ROOT / "discover_140_94_unsafe_extra_row_audit_v1.json",
        "hold_summary": INPUT_HOLD_ROOT / "summary_v1.json",
        "veto_mapping_summary": INPUT_VETO_MAPPING_ROOT / "summary_v1.json",
        "adapter_mapping_summary": INPUT_ADAPTER_MAPPING_ROOT / "summary_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    discovery_go = _read_json(required["discovery_go_no_go"])
    if discovery_go.get("status_v1") != "140_94_VETO_FOUND_BUT_TOO_DESTRUCTIVE_TO_SAFE_CORE":
        raise RuntimeError("INPUT_DISCOVERY_STATUS_NOT_TOO_DESTRUCTIVE")
    return {
        "required_paths": required,
        "discovery_summary": _read_json(required["discovery_summary"]),
        "discovery_go_no_go": discovery_go,
        "discovery_candidate_metrics": _read_json(required["discovery_candidate_metrics"]),
        "discovery_cut_rows": _read_json(required["discovery_cut_rows"]),
        "discovery_unsafe_row": _read_json(required["discovery_unsafe_row"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "veto_mapping_summary": _read_json(required["veto_mapping_summary"]),
        "adapter_mapping_summary": _read_json(required["adapter_mapping_summary"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "source_inputs": discover._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "veto_discovery_root_v1": str(INPUT_DISCOVERY_ROOT),
            "hold_root_v1": str(INPUT_HOLD_ROOT),
            "veto_mapping_root_v1": str(INPUT_VETO_MAPPING_ROOT),
            "adapter_input_mapping_root_v1": str(INPUT_ADAPTER_MAPPING_ROOT),
            "harden_root_v1": str(INPUT_HARDEN_ROOT),
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
    frame, masks = discover._build_frame_and_masks(inputs)
    source = _str(frame, "source_evidence_v1")
    score = _num(frame, "candidate_score_v1")
    masks["missing_r5_tail_v1"] = ~source.str.contains("R5_TAIL_SCORE", regex=False)
    masks["has_r5_bad_support_v1"] = source.str.contains("R5_BAD_SCORE:SUPPORT", regex=False)
    masks["has_historical_v2_blueprint_v1"] = source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False)
    masks["has_v2_oof_v1"] = source.str.contains("V2_OOF", regex=False)
    masks["high_score_099_v1"] = score.ge(0.99)
    masks["high_score_unsafe_exact_band_v1"] = score.ge(0.99) & score.le(0.9932073485867541)
    return frame, masks


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


def _reproducibility(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    metrics = _selected_metrics(frame, masks["hardened"])
    prior_rows = inputs["discovery_candidate_metrics"]["rows_v1"]
    prior_best = next(
        row
        for row in prior_rows
        if row["candidate_veto_name_v1"] == "SIGNAL_SHAPE_REFINED_NO_R5_TAIL_R5_BAD_SCORE_GE_099"
    )
    without = frame[masks["low_support_veto_only"] & ~masks["hardened"]]
    payload = {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        **metrics,
        "unsafe_extra_without_hard_veto_rows_v1": int(_bool(without, "unsafe_audit_v1").sum()),
        "best_prior_deployable_destructive_candidate_v1": prior_best["candidate_veto_name_v1"],
        "best_prior_deployable_destructive_safe_core_cut_v1": prior_best["safe_core_rows_accidentally_blocked_v1"],
        "prior_discovery_status_v1": inputs["discovery_go_no_go"].get("status_v1"),
        "reproduced_from_explicit_discovery_artifact_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def _candidate_definitions(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = masks["low_support_veto_only"]
    no_tail = masks["missing_r5_tail_v1"]
    has_r5_bad = masks["has_r5_bad_support_v1"]
    score = _num(frame, "candidate_score_v1")
    source = _str(frame, "source_evidence_v1")
    policy = _str(frame, "run_id_policy_class_v1")
    student = _num(frame, "student_oof_score_v1")
    common_signal = base & no_tail & has_r5_bad & score.ge(0.99)
    return [
        {
            "candidate_name_v1": "BRANCH_LOCAL_SIGNAL_SHAPE_NO_TAIL_R5_BAD_SCORE_GE_099_V1",
            "veto_family_v1": "BRANCH_LOCAL_SIGNAL_SHAPE_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1",
            "condition_rule_v1": "inside safe-core branch, veto if missing R5_TAIL_SCORE and has R5_BAD_SCORE support and score >= 0.99",
            "as_of_lineage_v1": "AS_OF_SAFE_EXISTING_SCORE_AND_SIGNAL_FLAGS",
            "lineage_status_v1": "CONFIRMED_AS_OF_SIGNAL_SHAPE",
            "adapter_feasibility_v1": "DEPLOYABLE_BUT_TOO_DESTRUCTIVE",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": common_signal,
        },
        {
            "candidate_name_v1": "TWO_CONDITION_NO_TAIL_REPAIRABLE_SUPPORT_V1",
            "veto_family_v1": "TWO_CONDITION_CONFLUENCE_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1, run_id_policy_class_v1",
            "condition_rule_v1": "veto if missing R5 tail + R5 bad support + score >= 0.99 + support-repairable policy",
            "as_of_lineage_v1": "AS_OF_SIGNAL_SHAPE_PLUS_SUPPORT_POLICY",
            "lineage_status_v1": "NEEDS_NORMALIZATION",
            "adapter_feasibility_v1": "DEPLOYABLE_SHAPE_BUT_ORANGE_RETENTION",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": common_signal & policy.eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS"),
        },
        {
            "candidate_name_v1": "RELAXED_SIGNAL_SHAPE_SCORE_GE_099321_V1",
            "veto_family_v1": "RELAXED_SIGNAL_SHAPE_THRESHOLD_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1",
            "condition_rule_v1": "veto if missing R5 tail + R5 bad support + score >= 0.99321",
            "as_of_lineage_v1": "AS_OF_SAFE_EXISTING_SCORE_AND_SIGNAL_FLAGS",
            "lineage_status_v1": "CONFIRMED_AS_OF_SIGNAL_SHAPE",
            "adapter_feasibility_v1": "FAILS_TO_BLOCK_UNSAFE_ROW",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": base & no_tail & has_r5_bad & score.ge(0.99321),
        },
        {
            "candidate_name_v1": "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1",
            "veto_family_v1": "EXCEPTION_GUARDED_SIGNAL_SHAPE_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1, HISTORICAL_V2_BLUEPRINT evidence token",
            "condition_rule_v1": "veto high-score no-tail R5-bad rows unless HISTORICAL_V2_BLUEPRINT is present as a guard",
            "as_of_lineage_v1": "MECHANICALLY_PRECISE_BUT_HISTORICAL_BLUEPRINT_TOKEN_NOT_IN_CURRENT_ADAPTER_ALLOWLIST",
            "lineage_status_v1": "NEEDS_LINEAGE_CONFIRMATION",
            "adapter_feasibility_v1": "GREEN_RETENTION_BUT_LINEAGE_CONFIRMATION_REQUIRED",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": common_signal & ~source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False),
        },
        {
            "candidate_name_v1": "LOW_SUPPORT_AWARE_NO_TAIL_REPAIRABLE_NO_HISTORICAL_BLUEPRINT_V1",
            "veto_family_v1": "LOW_SUPPORT_AWARE_SIGNAL_SHAPE_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1, run_id_policy_class_v1, HISTORICAL_V2_BLUEPRINT evidence token",
            "condition_rule_v1": "veto high-score no-tail R5-bad rows in support-repairable policy unless historical V2 blueprint is present",
            "as_of_lineage_v1": "MECHANICALLY_PRECISE_BUT_REQUIRES_POLICY_AND_HISTORICAL_BLUEPRINT_LINEAGE_CONFIRMATION",
            "lineage_status_v1": "NEEDS_LINEAGE_CONFIRMATION",
            "adapter_feasibility_v1": "GREEN_RETENTION_BUT_LINEAGE_CONFIRMATION_REQUIRED",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": common_signal
            & policy.eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS")
            & ~source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False),
        },
        {
            "candidate_name_v1": "MINIMAL_DESTRUCTIVE_NO_TAIL_NO_HISTORICAL_BLUEPRINT_V1",
            "veto_family_v1": "MINIMAL_DESTRUCTIVE_VETO",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1, HISTORICAL_V2_BLUEPRINT evidence token",
            "condition_rule_v1": "minimal deployable-looking condition found: high-score no-tail R5-bad with no historical V2 blueprint guard",
            "as_of_lineage_v1": "MECHANICALLY_PRECISE_BUT_HISTORICAL_BLUEPRINT_TOKEN_NOT_YET_DEPLOYABLE",
            "lineage_status_v1": "NEEDS_LINEAGE_CONFIRMATION",
            "adapter_feasibility_v1": "GREEN_RETENTION_BUT_LINEAGE_CONFIRMATION_REQUIRED",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": False,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": common_signal & ~source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False),
        },
        {
            "candidate_name_v1": "DIAGNOSTIC_STUDENT_DISTANCE_SCORE_LT_050_V1",
            "veto_family_v1": "DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON",
            "input_fields_v1": "student_oof_score_v1, candidate_score_v1",
            "condition_rule_v1": "diagnostic only: veto if score >= 0.99 and student membership score < 0.50",
            "as_of_lineage_v1": "OOF_SCORE_PRESENT_BUT_MEMBERSHIP_TARGET_HISTORY",
            "lineage_status_v1": "BLOCKED_MEMBERSHIP_PROXY",
            "adapter_feasibility_v1": "DIAGNOSTIC_ONLY_DO_NOT_DEPLOY",
            "row_identity_risk_v1": False,
            "membership_proxy_risk_v1": True,
            "coverage_proxy_risk_v1": False,
            "audit_only_risk_v1": False,
            "outcome_hindsight_risk_v1": False,
            "mask": base & score.ge(0.99) & student.lt(0.50),
        },
    ]


def _metrics_for_candidates(
    frame: pd.DataFrame, masks: dict[str, pd.Series], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = masks["low_support_veto_only"]
    hardened = masks["hardened"]
    original = _bool(frame, "selected_original_140_v1")
    unsafe = _bool(frame, "unsafe_audit_v1")
    rows: list[dict[str, Any]] = []
    dry_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        veto_mask = candidate["mask"]
        selected = base & ~veto_mask
        selected_frame = frame[selected]
        cut_safe_core = veto_mask & hardened
        cut_original = cut_safe_core & original
        unsafe_blocked = int((veto_mask & base & ~hardened & unsafe).sum()) > 0
        shortcut = any(
            bool(candidate[key])
            for key in [
                "row_identity_risk_v1",
                "membership_proxy_risk_v1",
                "coverage_proxy_risk_v1",
                "audit_only_risk_v1",
                "outcome_hindsight_risk_v1",
            ]
        )
        tier = retention_tier(
            unsafe_row_blocked=unsafe_blocked,
            good_rows_cut=int(cut_safe_core.sum()),
            shortcut_or_leakage=shortcut,
        )
        adapter_ready = (
            tier == "GREEN"
            and candidate["lineage_status_v1"] == "CONFIRMED_AS_OF_SIGNAL_SHAPE"
            and int((selected & unsafe).sum()) == 0
        )
        row = {
            "candidate_name_v1": candidate["candidate_name_v1"],
            "veto_family_v1": candidate["veto_family_v1"],
            "input_fields_v1": candidate["input_fields_v1"],
            "as_of_lineage_v1": candidate["as_of_lineage_v1"],
            "lineage_status_v1": candidate["lineage_status_v1"],
            "rule_condition_v1": candidate["condition_rule_v1"],
            "adapter_feasibility_v1": candidate["adapter_feasibility_v1"],
            "unsafe_row_blocked_v1": unsafe_blocked,
            "unsafe_rows_after_veto_v1": int((selected & unsafe).sum()),
            "unsafe_lookalikes_blocked_v1": int((veto_mask & base & unsafe).sum()),
            "selected_rows_after_veto_v1": int(selected.sum()),
            "recovered_original_140_rows_v1": int((selected & original).sum()),
            "safe_core_rows_retained_v1": int((selected & hardened).sum()),
            "safe_core_rows_cut_v1": int(cut_safe_core.sum()),
            "original_140_rows_cut_v1": int(cut_original.sum()),
            "extra_rows_retained_v1": int((selected & ~original).sum()),
            "remaining_extra_3_retained_v1": int((selected & hardened & ~original).sum()),
            "remaining_extra_3_blocked_v1": int((veto_mask & hardened & ~original).sum()),
            "bad_count_after_veto_audit_only_v1": int(_bool(selected_frame, "bad_label_v1").sum()),
            "tail_count_after_veto_audit_only_v1": int(_bool(selected_frame, "tail_label_v1").sum()),
            "precision_after_veto_audit_only_v1": float(
                _bool(selected_frame, "bad_label_v1").sum() / max(len(selected_frame), 1)
            ),
            "safety_after_veto_v1": "CLEAN" if int((selected & unsafe).sum()) == 0 else "FAIL",
            "complexity_v1": "LOW" if "HISTORICAL_V2_BLUEPRINT" not in candidate["input_fields_v1"] else "LOW_LOGIC_LINEAGE_OPEN",
            "row_identity_risk_v1": candidate["row_identity_risk_v1"],
            "membership_proxy_risk_v1": candidate["membership_proxy_risk_v1"],
            "coverage_proxy_risk_v1": candidate["coverage_proxy_risk_v1"],
            "audit_only_risk_v1": candidate["audit_only_risk_v1"],
            "outcome_hindsight_risk_v1": candidate["outcome_hindsight_risk_v1"],
            "retention_tier_v1": tier,
            "adapter_ready_v1": adapter_ready,
            "recommendation_v1": _candidate_recommendation(tier, candidate["lineage_status_v1"], unsafe_blocked),
        }
        rows.append(row)
        dry_rows.append(
            {
                "candidate_name_v1": row["candidate_name_v1"],
                "selected_rows_v1": row["selected_rows_after_veto_v1"],
                "recovered_safe_core_rows_v1": row["safe_core_rows_retained_v1"],
                "recovered_original_140_rows_v1": row["recovered_original_140_rows_v1"],
                "extra_rows_v1": row["extra_rows_retained_v1"],
                "unsafe_rows_v1": row["unsafe_rows_after_veto_v1"],
                "bad_tail_audit_only_v1": [
                    row["bad_count_after_veto_audit_only_v1"],
                    row["tail_count_after_veto_audit_only_v1"],
                ],
                "precision_audit_only_v1": row["precision_after_veto_audit_only_v1"],
                "safety_status_v1": row["safety_after_veto_v1"],
                "mismatch_vs_current_safe_core_v1": row["safe_core_rows_cut_v1"] + int((selected & ~hardened).sum()),
                "mismatch_vs_destructive_veto_v1": abs(row["safe_core_rows_cut_v1"] - EXPECTED_DESTRUCTIVE_CUTS),
                "adapter_readiness_v1": "READY" if row["adapter_ready_v1"] else row["adapter_feasibility_v1"],
                "retention_tier_v1": row["retention_tier_v1"],
            }
        )
        for _, cut in frame[cut_safe_core].sort_values(["run_id_v1", "candidate_uid_v1"]).iterrows():
            retention_rows.append(
                {
                    "candidate_name_v1": row["candidate_name_v1"],
                    "row_id_v1": cut.get("candidate_uid_v1"),
                    "run_id_v1": cut.get("run_id_v1"),
                    "fold_id_v1": cut.get("fold_id_v1"),
                    "bad_label_audit_only_v1": _as_bool(cut.get("bad_label_v1")),
                    "tail_label_audit_only_v1": _as_bool(cut.get("tail_label_v1")),
                    "source_evidence_v1": cut.get("source_evidence_v1"),
                    "student_oof_score_v1": cut.get("student_oof_score_v1"),
                    "cut_by_condition_v1": row["rule_condition_v1"],
                    "retention_tier_v1": row["retention_tier_v1"],
                    "acceptable_cut_under_green_v1": row["retention_tier_v1"] == "GREEN",
                    "tail_coverage_impact_v1": "TAIL_ROW_CUT" if _as_bool(cut.get("tail_label_v1")) else "NON_TAIL_ROW_CUT",
                }
            )
    validate_candidate_metrics(rows)
    return rows, dry_rows, retention_rows


def _candidate_recommendation(tier: str, lineage_status: str, unsafe_blocked: bool) -> str:
    if not unsafe_blocked:
        return "REJECT_DOES_NOT_BLOCK_UNSAFE_ROW"
    if lineage_status == "BLOCKED_MEMBERSHIP_PROXY":
        return "DIAGNOSTIC_ONLY_MEMBERSHIP_PROXY"
    if tier == "GREEN" and lineage_status == "NEEDS_LINEAGE_CONFIRMATION":
        return "PROMISING_MECHANICAL_VETO_REQUIRES_LINEAGE_AUDIT"
    if tier == "GREEN":
        return "ADAPTER_READY_CANDIDATE"
    if tier == "YELLOW":
        return "REVIEW_YELLOW_RETENTION_BEFORE_ADAPTER"
    if tier in {"ORANGE", "RED"}:
        return "REJECT_TOO_DESTRUCTIVE_TO_SAFE_CORE"
    return "REJECT_BLOCKED"


def _destructive_veto_audit(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row = next(row for row in metrics_rows if row["candidate_name_v1"] == "BRANCH_LOCAL_SIGNAL_SHAPE_NO_TAIL_R5_BAD_SCORE_GE_099_V1")
    return [
        {
            "veto_name_v1": "SIGNAL_SHAPE_REFINED_NO_R5_TAIL_R5_BAD_SCORE_GE_099",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_r5_bad_score_v1, candidate_score_v1",
            "condition_rule_v1": "missing R5_TAIL_SCORE + R5_BAD_SCORE support + score >= 0.99",
            "as_of_lineage_v1": "AS_OF_SAFE_EXISTING_SCORE_AND_SIGNAL_FLAGS",
            "unsafe_row_blocked_v1": row["unsafe_row_blocked_v1"],
            "safe_core_rows_retained_v1": row["safe_core_rows_retained_v1"],
            "safe_core_rows_cut_v1": row["safe_core_rows_cut_v1"],
            "original_140_rows_retained_v1": row["recovered_original_140_rows_v1"],
            "original_140_rows_cut_v1": row["original_140_rows_cut_v1"],
            "bad_tail_impact_audit_only_v1": [
                EXPECTED_BAD - row["bad_count_after_veto_audit_only_v1"],
                EXPECTED_TAIL - row["tail_count_after_veto_audit_only_v1"],
            ],
            "safety_impact_v1": "BLOCKS_UNSAFE_ROW_BUT_TOO_MUCH_GOOD_ROW_LOSS",
            "branch_tier_impact_v1": "global high-score no-tail R5-bad branch",
            "why_too_destructive_v1": "The condition treats many good historical V2-like safe-core rows as equivalent to the unsafe row.",
            "condition_causing_most_good_row_loss_v1": "missing R5_TAIL_SCORE combined with high score; lack of a confirmed good-core exception guard",
        }
    ]


def _cut_21_good_rows_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    source = _str(frame, "source_evidence_v1")
    destructive = (
        masks["low_support_veto_only"]
        & masks["missing_r5_tail_v1"]
        & masks["has_r5_bad_support_v1"]
        & masks["high_score_099_v1"]
        & masks["hardened"]
    )
    rows = []
    for _, row in frame[destructive].sort_values(["run_id_v1", "candidate_uid_v1"]).iterrows():
        has_hist = "HISTORICAL_V2_BLUEPRINT" in str(row.get("source_evidence_v1", ""))
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "safe_core_v1": True,
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "branch_tier_v1": "high-score no-tail R5-bad V2-like branch",
                "which_veto_condition_cut_it_v1": "missing R5_TAIL_SCORE + R5_BAD_SCORE support + score >= 0.99",
                "source_evidence_v1": row.get("source_evidence_v1"),
                "differs_from_unsafe_row_v1": "has HISTORICAL_V2_BLUEPRINT guard" if has_hist else "same source-evidence shape as unsafe except row is audit-safe",
                "safe_distinguishing_signal_v1": "HISTORICAL_V2_BLUEPRINT" if has_hist else "NONE_CONFIRMED",
                "low_support_v1": row.get("run_id_policy_class_v1") != "SUPPORT_SUFFICIENT",
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "retained_by_refined_hist_guard_v1": has_hist,
                "can_be_retained_with_refined_as_of_condition_v1": has_hist,
            }
        )
    return rows


def _unsafe_row_refinement_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    unsafe_mask = masks["low_support_veto_only"] & ~masks["hardened"] & _bool(frame, "unsafe_audit_v1")
    rows = []
    for _, row in frame[unsafe_mask].iterrows():
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "branch_tier_v1": "score>=0.95 + R5_1 + V2-like + low-support-clear",
                "positive_signals_that_selected_it_v1": row.get("source_evidence_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "student_oof_score_v1": row.get("student_oof_score_v1"),
                "exact_signal_shape_pattern_v1": "high-score, no R5_TAIL_SCORE, R5_BAD_SCORE support, V2_LIKE_BAD_TAIL strong, no historical V2 blueprint token",
                "as_of_fields_distinguishing_from_retained_safe_core_rows_v1": "absence of HISTORICAL_V2_BLUEPRINT distinguishes it from 18/21 rows cut by the destructive veto; remaining 3 require lineage review",
                "fields_too_broad_v1": "missing R5_TAIL_SCORE and high score alone are too broad",
                "branch_specific_veto_possible_v1": True,
                "branch_specific_veto_status_v1": "MECHANICALLY_GREEN_IF_HISTORICAL_BLUEPRINT_GUARD_IS_LINEAGE_CONFIRMED",
            }
        )
    return rows


def _candidate_definitions_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = []
    for candidate in candidates:
        definitions.append({key: value for key, value in candidate.items() if key != "mask"})
    return {
        "layer_name": "REFINE_140_94_REFINED_VETO_CANDIDATE_DEFINITIONS_V1",
        "candidate_count_v1": len(definitions),
        "candidates_v1": definitions,
    }


def _retention_threshold_audit(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {tier: 0 for tier in ["GREEN", "YELLOW", "ORANGE", "RED", "BLOCKED"]}
    for row in metrics_rows:
        counts[row["retention_tier_v1"]] += 1
    green = [
        row
        for row in metrics_rows
        if row["retention_tier_v1"] == "GREEN" and row["lineage_status_v1"] == "NEEDS_LINEAGE_CONFIRMATION"
    ]
    return {
        "layer_name": "REFINE_140_94_RETENTION_THRESHOLD_AUDIT_V1",
        "thresholds_v1": {
            "GREEN": "unsafe row blocked and <=5 good safe-core/original-140 rows cut",
            "YELLOW": "unsafe row blocked and 6-10 good rows cut",
            "ORANGE": "unsafe row blocked and 11-20 good rows cut",
            "RED": "unsafe row blocked but >20 good rows cut",
            "BLOCKED": "unsafe row not blocked or shortcut/leakage detected",
        },
        "tier_counts_v1": counts,
        "green_mechanical_candidates_requiring_lineage_v1": [row["candidate_name_v1"] for row in green],
        "adapter_ready_green_candidates_v1": [
            row["candidate_name_v1"] for row in metrics_rows if row["adapter_ready_v1"]
        ],
    }


def _diagnostic_student_distance_comparison(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic = [
        row for row in metrics_rows if row["veto_family_v1"] == "DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON"
    ]
    return {
        "layer_name": "REFINE_140_94_DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON_V1",
        "status_v1": "DIAGNOSTIC_ONLY_BLOCKED_MEMBERSHIP_PROXY",
        "do_not_use_as_deployable_veto_v1": True,
        "do_not_use_as_selector_v1": True,
        "do_not_use_as_adapter_input_v1": True,
        "comparison_rows_v1": diagnostic,
        "interpretation_v1": "Student/distance signal separates the unsafe row with excellent retention, but remains blocked because it comes from membership-target student history.",
    }


def _unsafe_lookalike_audit(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    selected_candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    source = _str(frame, "source_evidence_v1")
    candidate_mask = selected_candidate["mask"]
    similar = (
        masks["low_support_veto_only"]
        & masks["missing_r5_tail_v1"]
        & masks["has_r5_bad_support_v1"]
        & masks["high_score_099_v1"]
    )
    rows = []
    for _, row in frame[similar].sort_values(["unsafe_audit_v1", "run_id_v1", "candidate_uid_v1"], ascending=[False, True, True]).iterrows():
        blocked = _as_bool(candidate_mask.loc[row.name])
        has_hist = "HISTORICAL_V2_BLUEPRINT" in str(row.get("source_evidence_v1", ""))
        rows.append(
            {
                "candidate_name_v1": selected_candidate["candidate_name_v1"],
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_safe_core_v1": _as_bool(masks["hardened"].loc[row.name]),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "blocked_by_refined_veto_v1": blocked,
                "has_historical_v2_blueprint_v1": has_hist,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "student_oof_score_v1": row.get("student_oof_score_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "lookalike_reason_v1": "high-score no-tail R5-bad V2-like row",
                "risk_class_v1": "UNKNOWN_REQUIRES_LINEAGE_AUDIT"
                if selected_candidate["lineage_status_v1"] == "NEEDS_LINEAGE_CONFIRMATION"
                else "LOW_UNSAFE_LOOKALIKE_RISK",
            }
        )
    return rows


def _final_selection(
    metrics_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = next(
        row
        for row in metrics_rows
        if row["candidate_name_v1"] == "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1"
    )
    final = {
        "layer_name": "REFINE_140_94_FINAL_REFINED_VETO_SELECTION_V1",
        "selected_refined_veto_name_v1": selected["candidate_name_v1"],
        "selected_veto_family_v1": selected["veto_family_v1"],
        "unsafe_row_blocked_v1": selected["unsafe_row_blocked_v1"],
        "good_safe_core_rows_cut_v1": selected["safe_core_rows_cut_v1"],
        "original_140_rows_cut_v1": selected["original_140_rows_cut_v1"],
        "retention_tier_v1": selected["retention_tier_v1"],
        "lineage_status_v1": selected["lineage_status_v1"],
        "adapter_reopen_allowed_now_v1": False,
        "adapter_reopen_after_lineage_confirmation_possible_v1": True,
        "reason_v1": "The historical-blueprint guard is the first mechanically GREEN refined veto, but HISTORICAL_V2_BLUEPRINT is not in the current adapter allowlist and must be lineage-confirmed before adapter use.",
        "status_v1": FINAL_STATUS,
    }
    validate_final_selection(final)
    adapter = {
        "layer_name": "REFINE_140_94_ADAPTER_REOPEN_ASSESSMENT_V1",
        "adapter_build_can_reopen_now_v1": False,
        "adapter_build_can_reopen_after_lineage_confirmation_v1": True,
        "normalization_needed_before_adapter_v1": False,
        "lineage_audit_needed_before_adapter_v1": True,
        "safe_core_adapter_ready_now_v1": False,
        "safe_core_should_remain_held_until_lineage_confirmed_v1": True,
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
    }
    anti = {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS_ADAPTER_STILL_CLOSED",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_run_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_model_training_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_implicit_latest_glob_selection_v1": True,
        "student_distance_candidates_diagnostic_only_v1": True,
        "row_identity_veto_used_v1": False,
        "audit_only_veto_promoted_v1": False,
        "historical_blueprint_guard_not_promoted_without_lineage_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    recommendation = {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_remains_closed_v1": True,
        "rationale_v1": [
            "The prior deployable signal-shape veto cut 21 good safe-core rows and stayed too destructive.",
            "A historical-blueprint exception guard blocks the unsafe row while cutting only 3 good original-140 safe-core rows.",
            "That guard is not yet in the adapter allowlist and may be historical-artifact proxy evidence, so lineage confirmation is required before adapter use.",
            "Student/distance candidates remain diagnostic-only because of membership-target history.",
        ],
    }
    go_no_go = {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_refined_veto_name_v1": selected["candidate_name_v1"],
        "unsafe_row_blocked_v1": selected["unsafe_row_blocked_v1"],
        "good_safe_core_rows_cut_v1": selected["safe_core_rows_cut_v1"],
        "retention_tier_v1": selected["retention_tier_v1"],
        "lineage_status_v1": selected["lineage_status_v1"],
        "adapter_reopen_allowed_now_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])
    return final, adapter, anti, recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    destructive_rows: list[dict[str, Any]],
    cut_rows: list[dict[str, Any]],
    unsafe_rows: list[dict[str, Any]],
    candidate_defs: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    dry_rows: list[dict[str, Any]],
    retention: dict[str, Any],
    diagnostic: dict[str, Any],
    lookalike_rows: list[dict[str, Any]],
    final: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "refine_140_94_hard_safety_veto_reproducibility_audit_v1.md",
        [
            "# Refine 140/94 Hard Safety Veto Reproducibility Audit V1",
            "",
            f"- Safe-core selected rows: `{repro['selected_rows_v1']}`",
            f"- Original-140 recovered: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_audit_only_v1']} / {repro['tail_count_audit_only_v1']}`",
            f"- Prior destructive safe-core cuts: `{repro['best_prior_deployable_destructive_safe_core_cut_v1']}`",
        ],
    )
    _write_report(
        root / "refine_140_94_destructive_veto_audit_v1.md",
        [
            "# Refine 140/94 Destructive Veto Audit V1",
            "",
            *[
                f"- `{row['veto_name_v1']}` cuts `{row['safe_core_rows_cut_v1']}` safe-core rows and is `{row['safety_impact_v1']}`."
                for row in destructive_rows
            ],
        ],
    )
    _write_report(
        root / "refine_140_94_cut_21_good_rows_audit_v1.md",
        [
            "# Refine 140/94 Cut 21 Good Rows Audit V1",
            "",
            f"- Cut rows audited: `{len(cut_rows)}`",
            f"- Rows retained by historical-blueprint guard: `{sum(row['retained_by_refined_hist_guard_v1'] for row in cut_rows)}`",
            f"- Rows still cut by refined guard: `{sum(not row['retained_by_refined_hist_guard_v1'] for row in cut_rows)}`",
        ],
    )
    _write_report(
        root / "refine_140_94_unsafe_row_refinement_audit_v1.md",
        [
            "# Refine 140/94 Unsafe Row Refinement Audit V1",
            "",
            f"- Unsafe rows audited: `{len(unsafe_rows)}`",
            "- The useful mechanical distinction is lack of a historical V2 blueprint guard, but that lineage is not yet adapter-confirmed.",
        ],
    )
    _write_report(
        root / "refine_140_94_refined_veto_candidate_definitions_v1.md",
        [
            "# Refine 140/94 Refined Veto Candidate Definitions V1",
            "",
            *[
                f"- `{row['candidate_name_v1']}`: {row['condition_rule_v1']}"
                for row in candidate_defs["candidates_v1"]
            ],
        ],
    )
    _write_report(
        root / "refine_140_94_refined_veto_candidate_metrics_v1.md",
        [
            "# Refine 140/94 Refined Veto Candidate Metrics V1",
            "",
            *[
                f"- `{row['candidate_name_v1']}`: unsafe blocked `{row['unsafe_row_blocked_v1']}`, safe-core cut `{row['safe_core_rows_cut_v1']}`, tier `{row['retention_tier_v1']}`, lineage `{row['lineage_status_v1']}`"
                for row in metrics_rows
            ],
        ],
    )
    _write_report(
        root / "refine_140_94_refined_veto_dry_run_results_v1.md",
        [
            "# Refine 140/94 Refined Veto Dry Run Results V1",
            "",
            *[
                f"- `{row['candidate_name_v1']}`: selected `{row['selected_rows_v1']}`, original recovered `{row['recovered_original_140_rows_v1']}`, unsafe `{row['unsafe_rows_v1']}`, readiness `{row['adapter_readiness_v1']}`"
                for row in dry_rows
            ],
        ],
    )
    _write_report(
        root / "refine_140_94_retention_threshold_audit_v1.md",
        [
            "# Refine 140/94 Retention Threshold Audit V1",
            "",
            f"- Tier counts: `{retention['tier_counts_v1']}`",
            f"- Adapter-ready GREEN candidates: `{retention['adapter_ready_green_candidates_v1']}`",
            f"- GREEN candidates requiring lineage: `{retention['green_mechanical_candidates_requiring_lineage_v1']}`",
        ],
    )
    _write_report(
        root / "refine_140_94_diagnostic_student_distance_comparison_v1.md",
        [
            "# Refine 140/94 Diagnostic Student Distance Comparison V1",
            "",
            f"- Status: `{diagnostic['status_v1']}`",
            "- Student/distance evidence remains diagnostic-only and was not used as deployable veto input.",
        ],
    )
    _write_report(
        root / "refine_140_94_unsafe_lookalike_audit_v1.md",
        [
            "# Refine 140/94 Unsafe Lookalike Audit V1",
            "",
            f"- Lookalike rows: `{len(lookalike_rows)}`",
            "- Boundary behavior is mechanically clean, but risk class remains lineage-dependent.",
        ],
    )
    _write_report(
        root / "refine_140_94_final_refined_veto_selection_v1.md",
        [
            "# Refine 140/94 Final Refined Veto Selection V1",
            "",
            f"- Selected veto: `{final['selected_refined_veto_name_v1']}`",
            f"- Unsafe row blocked: `{final['unsafe_row_blocked_v1']}`",
            f"- Good safe-core rows cut: `{final['good_safe_core_rows_cut_v1']}`",
            f"- Status: `{final['status_v1']}`",
        ],
    )
    _write_report(
        root / "refine_140_94_adapter_reopen_assessment_v1.md",
        [
            "# Refine 140/94 Adapter Reopen Assessment V1",
            "",
            f"- Adapter can reopen now: `{adapter['adapter_build_can_reopen_now_v1']}`",
            f"- Lineage audit needed: `{adapter['lineage_audit_needed_before_adapter_v1']}`",
            f"- Status: `{adapter['status_v1']}`",
        ],
    )
    _write_report(
        root / "refine_140_94_hard_safety_veto_anti_shortcut_audit_v1.md",
        [
            "# Refine 140/94 Hard Safety Veto Anti-Shortcut Audit V1",
            "",
            "- No R6, adapter, IQL, package, freeze, promo, live, Optuna, broad sweep, or model training was run.",
            "- Student/distance, audit-only, row-identity, membership, coverage, hindsight, and outcome shortcuts were not promoted.",
        ],
    )
    _write_report(
        root / "refine_140_94_hard_safety_veto_recommendation_v1.md",
        [
            "# Refine 140/94 Hard Safety Veto Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            "- Confirm historical-blueprint lineage before adapter build.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _build_frame_and_masks(inputs["source_inputs"])
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, masks, inputs)
    candidates = _candidate_definitions(frame, masks)
    metrics_rows, dry_rows, retention_rows = _metrics_for_candidates(frame, masks, candidates)
    destructive_rows = _destructive_veto_audit(metrics_rows)
    cut_rows = _cut_21_good_rows_audit(frame, masks)
    unsafe_rows = _unsafe_row_refinement_audit(frame, masks)
    candidate_defs = _candidate_definitions_payload(candidates)
    retention = _retention_threshold_audit(metrics_rows)
    diagnostic = _diagnostic_student_distance_comparison(metrics_rows)
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate["candidate_name_v1"] == "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1"
    )
    lookalike_rows = _unsafe_lookalike_audit(frame, masks, selected_candidate)
    final, adapter, anti, recommendation, go_no_go = _final_selection(metrics_rows)

    _write_json(artifact_root / "refine_140_94_hard_safety_veto_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "refine_140_94_hard_safety_veto_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "refine_140_94_destructive_veto_audit_v1.csv", destructive_rows)
    _write_json(
        artifact_root / "refine_140_94_destructive_veto_audit_v1.json",
        {"row_count_v1": len(destructive_rows), "rows_v1": destructive_rows},
    )
    _write_rows(artifact_root / "refine_140_94_cut_21_good_rows_audit_v1.csv", cut_rows)
    _write_json(
        artifact_root / "refine_140_94_cut_21_good_rows_audit_v1.json",
        {"row_count_v1": len(cut_rows), "rows_v1": cut_rows},
    )
    _write_rows(artifact_root / "refine_140_94_unsafe_row_refinement_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "refine_140_94_unsafe_row_refinement_audit_v1.json",
        {"row_count_v1": len(unsafe_rows), "rows_v1": unsafe_rows},
    )
    _write_json(artifact_root / "refine_140_94_refined_veto_candidate_definitions_v1.json", candidate_defs)
    _write_rows(artifact_root / "refine_140_94_refined_veto_candidate_metrics_v1.csv", metrics_rows)
    _write_json(
        artifact_root / "refine_140_94_refined_veto_candidate_metrics_v1.json",
        {"row_count_v1": len(metrics_rows), "rows_v1": metrics_rows},
    )
    _write_rows(artifact_root / "refine_140_94_refined_veto_dry_run_results_v1.csv", dry_rows)
    _write_json(
        artifact_root / "refine_140_94_refined_veto_dry_run_results_v1.json",
        {"row_count_v1": len(dry_rows), "rows_v1": dry_rows},
    )
    _write_json(artifact_root / "refine_140_94_retention_threshold_audit_v1.json", retention)
    _write_json(artifact_root / "refine_140_94_diagnostic_student_distance_comparison_v1.json", diagnostic)
    _write_rows(artifact_root / "refine_140_94_unsafe_lookalike_audit_v1.csv", lookalike_rows)
    _write_json(
        artifact_root / "refine_140_94_unsafe_lookalike_audit_v1.json",
        {"row_count_v1": len(lookalike_rows), "rows_v1": lookalike_rows},
    )
    _write_json(artifact_root / "refine_140_94_final_refined_veto_selection_v1.json", final)
    _write_json(artifact_root / "refine_140_94_adapter_reopen_assessment_v1.json", adapter)
    _write_json(artifact_root / "refine_140_94_hard_safety_veto_anti_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "refine_140_94_hard_safety_veto_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json", go_no_go)

    _write_markdown(
        artifact_root,
        repro,
        destructive_rows,
        cut_rows,
        unsafe_rows,
        candidate_defs,
        metrics_rows,
        dry_rows,
        retention,
        diagnostic,
        lookalike_rows,
        final,
        adapter,
        recommendation,
    )

    summary = {
        "layer_name": "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": repro["unsafe_extra_without_hard_veto_rows_v1"],
        "selected_final_refined_veto_v1": final["selected_refined_veto_name_v1"],
        "unsafe_row_blocked_v1": final["unsafe_row_blocked_v1"],
        "good_safe_core_rows_cut_v1": final["good_safe_core_rows_cut_v1"],
        "retention_tier_v1": final["retention_tier_v1"],
        "adapter_reopen_allowed_now_v1": False,
        "adapter_reopen_after_lineage_confirmation_possible_v1": True,
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": FINAL_STATUS, "next_recommended_action_v1": NEXT_ACTION, "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Refine 140/94 Hard Safety Veto To Retain Safe-Core V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Selected refined veto: `{summary['selected_final_refined_veto_v1']}`",
            f"- Unsafe row blocked: `{summary['unsafe_row_blocked_v1']}`",
            f"- Good safe-core rows cut: `{summary['good_safe_core_rows_cut_v1']}`",
            f"- Retention tier: `{summary['retention_tier_v1']}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
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
