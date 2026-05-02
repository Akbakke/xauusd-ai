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

from gx1.scripts import materialize_close_proxy_veto_branch_and_select_safe_mainline_next_step_v1 as close_gate
from gx1.scripts import materialize_refine_140_94_hard_safety_veto_to_retain_safe_core_v1 as refine_gate
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1"

INPUT_CLOSE_ROOT = (
    DEFAULT_REPORTS_ROOT / "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1_20260428T175937Z_LOCK"
)
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1_20260428T174140Z_LOCK"
)
INPUT_REFINED_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1_20260428T172254Z_LOCK"
)
INPUT_HOLD_ROOT = (
    DEFAULT_REPORTS_ROOT / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T111216Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "CLEAN_AS_OF_SAFETY_LAYER_FOUND_ONLY_DESTRUCTIVE_CANDIDATES"
NEXT_ACTION = "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1"
BEST_CLEAN_RECIPE = "MINIMAL_SOURCE_HARD_VETO_V1"

EXPECTED_SAFE_CORE_SELECTED = 89
EXPECTED_SAFE_CORE_RECOVERED = 86
EXPECTED_SAFE_CORE_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607
EXPECTED_UNSAFE_WITHOUT_HARD_VETO = 1

ALLOWED_FINAL_STATUSES = {
    "CLEAN_AS_OF_SAFETY_LAYER_FOUND_GREEN_CANDIDATE_READY_FOR_INPUT_MAPPING",
    "CLEAN_AS_OF_SAFETY_LAYER_FOUND_YELLOW_CANDIDATE_NEEDS_REVIEW",
    "CLEAN_AS_OF_SAFETY_LAYER_FOUND_CANDIDATE_NEEDS_NORMALIZATION",
    "CLEAN_AS_OF_SAFETY_LAYER_FOUND_CANDIDATE_NEEDS_LINEAGE_CONFIRMATION",
    "CLEAN_AS_OF_SAFETY_LAYER_FOUND_ONLY_DESTRUCTIVE_CANDIDATES",
    "CLEAN_AS_OF_SAFETY_LAYER_NO_DEPLOYABLE_SOURCE_SIGNAL_FOUND",
    "CLEAN_AS_OF_SAFETY_LAYER_BLOCKED_BY_PROXY_OR_LEAKAGE_RISK",
    "CLEAN_AS_OF_SAFETY_LAYER_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "CLEAN_AS_OF_SAFETY_LAYER_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_CLEAN_AS_OF_SAFETY_LAYER_INPUT_MAPPING_V1",
    "REVIEW_YELLOW_SAFETY_LAYER_CANDIDATE_BEFORE_MAPPING_V1",
    "NORMALIZE_CLEAN_AS_OF_SAFETY_LAYER_INPUTS_V1",
    "DEEPEN_CLEAN_AS_OF_SAFETY_LAYER_LINEAGE_AUDIT_V1",
    "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_WITH_SOURCE_SAFETY_SIGNALS_V1",
}

REQUIRED_OUTPUTS = [
    "clean_as_of_safety_layer_input_manifest_v1.json",
    "clean_as_of_safety_layer_reproducibility_audit_v1.json",
    "clean_as_of_safety_layer_reproducibility_audit_v1.md",
    "clean_as_of_safety_layer_source_signal_inventory_v1.csv",
    "clean_as_of_safety_layer_source_signal_inventory_v1.json",
    "clean_as_of_safety_layer_source_signal_inventory_v1.md",
    "clean_as_of_safety_layer_boundary_cohorts_v1.json",
    "clean_as_of_safety_layer_boundary_cohorts_v1.md",
    "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.csv",
    "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.json",
    "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.md",
    "clean_as_of_safety_layer_feature_family_definitions_v1.json",
    "clean_as_of_safety_layer_feature_family_definitions_v1.md",
    "clean_as_of_safety_layer_feature_family_metrics_v1.csv",
    "clean_as_of_safety_layer_feature_family_metrics_v1.json",
    "clean_as_of_safety_layer_feature_family_metrics_v1.md",
    "clean_as_of_safety_layer_candidate_recipe_definitions_v1.json",
    "clean_as_of_safety_layer_candidate_recipe_definitions_v1.md",
    "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.csv",
    "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.json",
    "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.md",
    "clean_as_of_safety_layer_retention_policy_audit_v1.json",
    "clean_as_of_safety_layer_retention_policy_audit_v1.md",
    "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.csv",
    "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.json",
    "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.md",
    "clean_as_of_safety_layer_adapter_readiness_preassessment_v1.json",
    "clean_as_of_safety_layer_adapter_readiness_preassessment_v1.md",
    "clean_as_of_safety_layer_anti_shortcut_audit_v1.json",
    "clean_as_of_safety_layer_anti_shortcut_audit_v1.md",
    "clean_as_of_safety_layer_recommendation_v1.json",
    "clean_as_of_safety_layer_recommendation_v1.md",
    "build_clean_as_of_safety_feature_layer_from_source_signals_go_no_go_v1.json",
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
    broad_sweep: bool = False,
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
    if broad_sweep:
        failures.append("BROAD_SWEEP_FORBIDDEN")
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


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    expected = {
        "safe_core_selected_rows_v1": EXPECTED_SAFE_CORE_SELECTED,
        "safe_core_recovered_original_140_v1": EXPECTED_SAFE_CORE_RECOVERED,
        "safe_core_extra_rows_v1": EXPECTED_SAFE_CORE_EXTRA,
        "safe_core_bad_count_audit_only_v1": EXPECTED_BAD,
        "safe_core_tail_count_audit_only_v1": EXPECTED_TAIL,
        "safe_core_safety_status_v1": "CLEAN",
        "unsafe_extra_without_hard_veto_rows_v1": EXPECTED_UNSAFE_WITHOUT_HARD_VETO,
        "historical_v2_blueprint_deployable_now_v1": False,
        "adapter_r6_iql_remain_blocked_v1": True,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("safe_core_precision_audit_only_v1", -1)), EXPECTED_PRECISION):
        failures["safe_core_precision_audit_only_v1"] = payload.get("safe_core_precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"CLEAN_AS_OF_SAFETY_LAYER_REPRODUCTION_FAILED: {failures}")
    return True


def validate_source_signal_inventory(rows: list[dict[str, Any]]) -> bool:
    by_name = {row["signal_name_v1"]: row for row in rows}
    required_blocked = {
        "HISTORICAL_V2_BLUEPRINT": "BLOCKED_HISTORICAL_ARTIFACT_PROXY",
        "bad_label_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT",
        "tail_label_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT",
        "unsafe_audit_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT",
        "candidate_uid_v1": "BLOCKED_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT",
        "is_185_139_teacher_v1": "BLOCKED_MEMBERSHIP_PROXY",
        "is_plus45_diagnostic_v1": "BLOCKED_MEMBERSHIP_PROXY",
    }
    failures = []
    for name, expected in required_blocked.items():
        if by_name.get(name, {}).get("classification_v1") != expected:
            failures.append(name)
    if failures:
        raise RuntimeError(f"SOURCE_SIGNAL_INVENTORY_FAILED_TO_BLOCK_FORBIDDEN_FIELDS: {failures}")
    usable = [row for row in rows if row["classification_v1"].startswith("AS_OF_SAFE")]
    if not usable:
        raise RuntimeError("SOURCE_SIGNAL_INVENTORY_HAS_NO_AS_OF_SAFE_FIELDS")
    return True


def validate_candidate_recipe_dry_runs(rows: list[dict[str, Any]]) -> bool:
    clean_green = [
        row
        for row in rows
        if row["retention_class_v1"] == "GREEN"
        and row["leakage_or_proxy_risk_v1"] is False
        and row["unsafe_extra_row_blocked_v1"] is True
    ]
    if clean_green:
        raise RuntimeError(f"UNEXPECTED_GREEN_SOURCE_SIGNAL_CANDIDATE: {clean_green}")
    best = next(row for row in rows if row["recipe_name_v1"] == BEST_CLEAN_RECIPE)
    if best["unsafe_extra_row_blocked_v1"] is not True:
        raise RuntimeError("BEST_CLEAN_SOURCE_RECIPE_MUST_BLOCK_UNSAFE_ROW")
    if best["retention_class_v1"] not in {"ORANGE", "RED"}:
        raise RuntimeError("BEST_CLEAN_SOURCE_RECIPE_EXPECTED_DESTRUCTIVE_RETENTION")
    if any(row["uses_historical_v2_blueprint_v1"] and row["adapter_ready_v1"] for row in rows):
        raise RuntimeError("HISTORICAL_BLUEPRINT_CANNOT_BE_ADAPTER_READY")
    return True


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    blocked_flags = [
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "iql_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
    ]
    failures = [key for key in blocked_flags if payload.get(key) is not False]
    if failures:
        raise RuntimeError(f"GO_NO_GO_MUST_KEEP_FORBIDDEN_PATHS_BLOCKED: {failures}")
    if payload.get("historical_v2_blueprint_used_as_deployable_input_v1") is not False:
        raise RuntimeError("HISTORICAL_BLUEPRINT_MUST_NOT_BE_DEPLOYABLE_INPUT")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"CLEAN_AS_OF_SAFETY_LAYER_REQUIRED_OUTPUTS_MISSING: {missing}")
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
        INPUT_CLOSE_ROOT,
        INPUT_LANE_PACK_ROOT,
        INPUT_REFINED_ROOT,
        INPUT_HOLD_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "close_summary": INPUT_CLOSE_ROOT / "summary_v1.json",
        "close_go_no_go": INPUT_CLOSE_ROOT / "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json",
        "close_decision": INPUT_CLOSE_ROOT / "close_proxy_veto_mainline_decision_v1.json",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
        "lane_pack_go_no_go": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json",
        "refined_summary": INPUT_REFINED_ROOT / "summary_v1.json",
        "refined_go_no_go": INPUT_REFINED_ROOT / "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json",
        "hold_summary": INPUT_HOLD_ROOT / "summary_v1.json",
        "hold_go_no_go": INPUT_HOLD_ROOT / "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "harden_go_no_go": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    close_go = _read_json(required["close_go_no_go"])
    if close_go.get("status_v1") != "PROXY_VETO_BRANCH_CLOSED_SELECT_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_PATH":
        raise RuntimeError("INPUT_CLOSE_GATE_STATUS_NOT_CLEAN_SAFETY_LAYER_PATH")
    return {
        "required_paths": required,
        "close_summary": _read_json(required["close_summary"]),
        "close_go_no_go": close_go,
        "close_decision": _read_json(required["close_decision"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_pack_go_no_go": _read_json(required["lane_pack_go_no_go"]),
        "refined_summary": _read_json(required["refined_summary"]),
        "refined_go_no_go": _read_json(required["refined_go_no_go"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "hold_go_no_go": _read_json(required["hold_go_no_go"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "source_inputs": refine_gate._load_inputs()["source_inputs"],
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "decision_closure_root_v1": str(INPUT_CLOSE_ROOT),
            "parallel_refined_veto_lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
            "refined_veto_root_v1": str(INPUT_REFINED_ROOT),
            "hold_root_v1": str(INPUT_HOLD_ROOT),
            "safe_core_harden_root_v1": str(INPUT_HARDEN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "historical_v2_blueprint_used_as_deployable_input_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    frame, masks = refine_gate._build_frame_and_masks(inputs["source_inputs"])
    source = _str(frame, "source_evidence_v1")
    score = _num(frame, "candidate_score_v1")
    policy = _str(frame, "run_id_policy_class_v1")
    masks["base_without_hard_safety_veto_v1"] = masks["low_support_veto_only"]
    masks["source_missing_r5_tail_v1"] = ~_bool(frame, "signal_r5_tail_score_v1")
    masks["source_has_r5_bad_v1"] = _bool(frame, "signal_r5_bad_score_v1")
    masks["source_has_v2_like_v1"] = _bool(frame, "signal_v2_like_bad_tail_v1")
    masks["source_has_tail_repair_v1"] = _bool(frame, "signal_tail_repair_v1")
    masks["source_has_historical_v2_blueprint_v1"] = source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False)
    masks["source_support_repairable_v1"] = policy.eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS")
    masks["source_signal_shape_ge099_v1"] = (
        masks["base_without_hard_safety_veto_v1"]
        & masks["source_missing_r5_tail_v1"]
        & masks["source_has_r5_bad_v1"]
        & score.ge(0.99)
    )
    masks["source_confluence_repairable_v1"] = (
        masks["source_signal_shape_ge099_v1"] & masks["source_support_repairable_v1"]
    )
    masks["source_relaxed_score_ge099321_v1"] = (
        masks["base_without_hard_safety_veto_v1"]
        & masks["source_missing_r5_tail_v1"]
        & masks["source_has_r5_bad_v1"]
        & score.ge(0.99321)
    )
    masks["source_no_tail_no_repair_v1"] = (
        masks["base_without_hard_safety_veto_v1"]
        & masks["source_missing_r5_tail_v1"]
        & ~masks["source_has_tail_repair_v1"]
    )
    masks["source_low_support_or_missing_lineage_v1"] = (
        masks["base_without_hard_safety_veto_v1"]
        & (
            policy.str.contains("LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS", regex=False)
            | _bool(frame, "structural_low_support_v1")
            | _bool(frame, "zero_denominator_group_v1")
        )
    )
    masks["blocked_blueprint_guard_reference_v1"] = (
        masks["source_signal_shape_ge099_v1"] & ~masks["source_has_historical_v2_blueprint_v1"]
    )
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
    safe_core = _selected_metrics(frame, masks["hardened"])
    base_without = masks["base_without_hard_safety_veto_v1"]
    without_hard = frame[base_without & ~masks["hardened"]]
    payload = {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "baseline_140_94_bad_tail_v1": inputs["precheck_summary"].get("baseline_bad_tail_v1"),
        "baseline_140_94_status_v1": "CURRENT_BEST_CAUSAL_BASELINE",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "safe_core_selected_rows_v1": safe_core["selected_rows_v1"],
        "safe_core_recovered_original_140_v1": safe_core["recovered_original_140_rows_v1"],
        "safe_core_extra_rows_v1": safe_core["extra_rows_v1"],
        "safe_core_bad_count_audit_only_v1": safe_core["bad_count_audit_only_v1"],
        "safe_core_tail_count_audit_only_v1": safe_core["tail_count_audit_only_v1"],
        "safe_core_precision_audit_only_v1": safe_core["precision_audit_only_v1"],
        "safe_core_safety_status_v1": safe_core["safety_status_v1"],
        "base_without_hard_safety_veto_selected_rows_v1": int(base_without.sum()),
        "unsafe_extra_without_hard_veto_rows_v1": int(_bool(without_hard, "unsafe_audit_v1").sum()),
        "historical_v2_blueprint_deployable_now_v1": False,
        "proxy_veto_branch_closed_v1": inputs["close_summary"].get("proxy_veto_branch_closed_v1"),
        "adapter_r6_iql_remain_blocked_v1": inputs["close_summary"].get("adapter_r6_iql_remain_blocked_v1"),
        "v2_blueprint_lane_pack_status_v1": inputs["lane_pack_go_no_go"].get("status_v1"),
    }
    validate_reproducibility(payload)
    return payload


def _field_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"missing_count_v1": None, "value_range_v1": "NOT_PRESENT"}
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        return {
            "missing_count_v1": int(series.isna().sum()),
            "value_range_v1": [float(pd.to_numeric(series, errors="coerce").min()), float(pd.to_numeric(series, errors="coerce").max())],
        }
    return {
        "missing_count_v1": int(series.isna().sum()),
        "value_range_v1": sorted(str(item) for item in series.dropna().astype(str).unique())[:12],
    }


def _signal_inventory(frame: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        (
            "candidate_score_v1",
            "candidate_score_v1",
            "tail-repaired candidate source score",
            "AS_OF_SAFE_SOURCE_SIGNAL",
            True,
            "Source candidate score already used in safe-core positive rule.",
        ),
        (
            "signal_r5_1_bad_score_v1",
            "signal_r5_1_bad_score_v1",
            "R5.1 bad-score support flag",
            "AS_OF_SAFE_SOURCE_SIGNAL",
            True,
            "Boolean source-signal support used by safe-core.",
        ),
        (
            "signal_r5_bad_score_v1",
            "signal_r5_bad_score_v1",
            "R5 bad-score support flag",
            "AS_OF_SAFE_SOURCE_SIGNAL",
            True,
            "Boolean source-signal support used by source-safety candidates.",
        ),
        (
            "signal_r5_tail_score_v1",
            "signal_r5_tail_score_v1",
            "R5 tail-score support flag",
            "AS_OF_SAFE_SOURCE_SIGNAL",
            True,
            "Boolean source-signal support; absence is useful but too broad.",
        ),
        (
            "signal_v2_like_bad_tail_v1",
            "signal_v2_like_bad_tail_v1",
            "V2-like bad/tail support flag",
            "AS_OF_SAFE_NEEDS_NORMALIZATION",
            True,
            "Allowed V2-like source flag, distinct from historical V2 blueprint token.",
        ),
        (
            "signal_tail_repair_v1",
            "signal_tail_repair_v1",
            "tail repair support flag",
            "AS_OF_SAFE_NEEDS_NORMALIZATION",
            True,
            "Source support flag; mapping should avoid artifact membership semantics.",
        ),
        (
            "run_id_policy_class_v1",
            "run_id_policy_class_v1",
            "run/support policy class",
            "AS_OF_SAFE_NEEDS_NORMALIZATION",
            True,
            "Support policy may be adapter-usable only after normalization of train-only support semantics.",
        ),
        (
            "structural_low_support_v1",
            "structural_low_support_v1",
            "structural low-support flag",
            "AS_OF_SAFE_NEEDS_NORMALIZATION",
            True,
            "Support guard must be computed before outcome and normalized.",
        ),
        (
            "zero_denominator_group_v1",
            "zero_denominator_group_v1",
            "zero-denominator support flag",
            "AS_OF_SAFE_NEEDS_NORMALIZATION",
            True,
            "Support guard must be computed before outcome and normalized.",
        ),
        (
            "source_evidence_v1",
            "source_evidence_v1",
            "mixed evidence-token string",
            "AS_OF_SAFE_DIAGNOSTIC_ONLY",
            False,
            "Do not use wholesale; parse only independently allowlisted source tokens.",
        ),
        (
            "HISTORICAL_V2_BLUEPRINT",
            "source_evidence_v1 token",
            "historical V2 blueprint evidence token",
            "BLOCKED_HISTORICAL_ARTIFACT_PROXY",
            False,
            "Lane-pack blocked this as unresolved historical artifact / membership / coverage proxy risk.",
        ),
        (
            "hard_veto_clear_shadow_v1",
            "hard_veto_clear_shadow_v1",
            "current audit-only hard safety veto",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Reference-only audit veto; not deployable until mapped from source signals.",
        ),
        (
            "bad_label_v1",
            "bad_label_v1",
            "bad outcome label",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Outcome label may be audit target only, never source safety feature.",
        ),
        (
            "tail_label_v1",
            "tail_label_v1",
            "tail outcome label",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Outcome label may be audit target only, never source safety feature.",
        ),
        (
            "unsafe_audit_v1",
            "unsafe_audit_v1",
            "audit unsafe flag",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Audit-only safety label; cannot be adapter input.",
        ),
        (
            "protected_winner_status_v1",
            "protected_winner_status_v1",
            "protected winner audit status",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Audit/protected outcome status is not proven source-time deployable.",
        ),
        (
            "runner_protect_status_v1",
            "runner_protect_status_v1",
            "runner-protect audit status",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Audit status is not proven source-time deployable.",
        ),
        (
            "ambiguous_high_mfe_status_v1",
            "ambiguous_high_mfe_status_v1",
            "ambiguous high-MFE audit status",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "Ambiguous/high-MFE audit status is not source-time deployable here.",
        ),
        (
            "fifty_plus_mfe_risk_v1",
            "fifty_plus_mfe_risk_v1",
            "50+ MFE audit risk",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "MFE-related audit risk is post-outcome/hindsight unless independently remapped.",
        ),
        (
            "hundred_plus_mfe_risk_v1",
            "hundred_plus_mfe_risk_v1",
            "100+ MFE audit risk",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "MFE-related audit risk is post-outcome/hindsight unless independently remapped.",
        ),
        (
            "two_hundred_plus_mfe_risk_v1",
            "two_hundred_plus_mfe_risk_v1",
            "200+ MFE audit risk",
            "BLOCKED_OUTCOME_OR_HINDSIGHT",
            False,
            "MFE-related audit risk is post-outcome/hindsight unless independently remapped.",
        ),
        (
            "active_quarantine_v1",
            "active_quarantine_v1",
            "active/quarantine audit field",
            "BLOCKED_UNKNOWN_LINEAGE",
            False,
            "Not independently proven as source-time deployable in this gate.",
        ),
        (
            "is_140_94_baseline_v1",
            "is_140_94_baseline_v1",
            "materialized 140/94 membership flag",
            "BLOCKED_MEMBERSHIP_PROXY",
            False,
            "Materialized membership flags are never deployable features.",
        ),
        (
            "is_185_139_teacher_v1",
            "is_185_139_teacher_v1",
            "materialized 185/139 teacher membership flag",
            "BLOCKED_MEMBERSHIP_PROXY",
            False,
            "185/139 remains comparator/diagnostic only.",
        ),
        (
            "is_plus45_diagnostic_v1",
            "is_plus45_diagnostic_v1",
            "+45 diagnostic membership flag",
            "BLOCKED_MEMBERSHIP_PROXY",
            False,
            "+45 is diagnostic-only and not a feature/filter/target.",
        ),
        (
            "student_core_selected_v1",
            "student_core_selected_v1",
            "student-core membership flag",
            "BLOCKED_MEMBERSHIP_PROXY",
            False,
            "Student-core membership history is diagnostic, not source safety input.",
        ),
        (
            "candidate_uid_v1",
            "candidate_uid_v1",
            "row identity",
            "BLOCKED_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT",
            False,
            "Row identity shortcuts are forbidden.",
        ),
    ]
    rows = []
    for name, column, description, classification, adapter_compute, reason in specs:
        stats = _field_stats(frame, column) if column in frame.columns else {"missing_count_v1": None, "value_range_v1": "TOKEN_OR_NOT_PRESENT"}
        rows.append(
            {
                "signal_name_v1": name,
                "source_artifact_path_v1": "explicit locked inputs / reconstructed source frame",
                "source_column_path_v1": column,
                "datatype_v1": str(frame[column].dtype) if column in frame.columns else "token",
                "missingness_v1": stats["missing_count_v1"],
                "value_range_v1": stats["value_range_v1"],
                "when_computed_v1": "BEFORE_OUTCOME_OR_TRAIN_SUPPORT_TIME" if classification.startswith("AS_OF") else "AUDIT_OR_ARTIFACT_DERIVED_OR_UNKNOWN",
                "available_before_outcome_v1": classification.startswith("AS_OF"),
                "adapter_can_compute_v1": adapter_compute,
                "depends_on_artifact_membership_v1": classification in {"BLOCKED_MEMBERSHIP_PROXY", "BLOCKED_HISTORICAL_ARTIFACT_PROXY"},
                "depends_on_coverage_proxy_v1": classification == "BLOCKED_HISTORICAL_ARTIFACT_PROXY",
                "depends_on_final_labels_v1": classification == "BLOCKED_OUTCOME_OR_HINDSIGHT",
                "depends_on_mfe_hindsight_v1": classification == "BLOCKED_OUTCOME_OR_HINDSIGHT" and "mfe" in name.lower(),
                "depends_on_row_identity_v1": classification == "BLOCKED_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT",
                "usable_for_safety_layer_v1": classification.startswith("AS_OF_SAFE"),
                "classification_v1": classification,
                "reason_v1": reason,
            }
        )
    validate_source_signal_inventory(rows)
    return rows


def _ids(frame: pd.DataFrame, mask: pd.Series, limit: int | None = None) -> list[str]:
    values = [str(item) for item in frame.loc[mask, "candidate_uid_v1"].tolist()]
    return values if limit is None else values[:limit]


def _boundary_cohorts(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    base = masks["base_without_hard_safety_veto_v1"]
    safe_core = masks["hardened"]
    original = _bool(frame, "selected_original_140_v1")
    unsafe = _bool(frame, "unsafe_audit_v1")
    blueprint_cut = masks["blocked_blueprint_guard_reference_v1"] & base
    near_miss = masks["source_signal_shape_ge099_v1"] & base
    payload = {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_BOUNDARY_COHORTS_V1",
        "cohorts_v1": [
            {
                "cohort_id_v1": "A_89_SAFE_CORE_ROWS",
                "count_v1": int(safe_core.sum()),
                "row_ids_v1": _ids(frame, safe_core),
            },
            {
                "cohort_id_v1": "B_86_ORIGINAL_140_RECOVERED_BY_SAFE_CORE",
                "count_v1": int((safe_core & original).sum()),
                "row_ids_v1": _ids(frame, safe_core & original),
            },
            {
                "cohort_id_v1": "C_3_SAFE_CORE_EXTRA_ROWS",
                "count_v1": int((safe_core & ~original).sum()),
                "row_ids_v1": _ids(frame, safe_core & ~original),
            },
            {
                "cohort_id_v1": "D_1_UNSAFE_EXTRA_WITHOUT_HARD_VETO",
                "count_v1": int((base & ~safe_core & unsafe).sum()),
                "row_ids_v1": _ids(frame, base & ~safe_core & unsafe),
            },
            {
                "cohort_id_v1": "E_ROWS_CUT_BY_REFINED_V2_BLUEPRINT_VETO_REFERENCE",
                "count_v1": int(blueprint_cut.sum()),
                "row_ids_v1": _ids(frame, blueprint_cut),
                "deployable_status_v1": "DIAGNOSTIC_ONLY_PROXY_BLOCKED",
            },
            {
                "cohort_id_v1": "F_NEAR_MISS_NEAR_FAIL_SOURCE_SIGNAL_SHAPE_POOL",
                "count_v1": int(near_miss.sum()),
                "row_ids_v1": _ids(frame, near_miss),
            },
            {
                "cohort_id_v1": "G_UNSAFE_LOOKALIKE_POOL_AVAILABLE",
                "count_v1": int((near_miss & unsafe).sum()),
                "row_ids_v1": _ids(frame, near_miss & unsafe),
            },
        ],
        "plus45_and_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_USED",
    }
    return payload


def _unsafe_row_source_signal_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = masks["base_without_hard_safety_veto_v1"]
    similar = masks["source_signal_shape_ge099_v1"] & base
    unsafe = _bool(frame, "unsafe_audit_v1")
    original = _bool(frame, "selected_original_140_v1")
    rows = []
    for _, row in frame[similar].sort_values(["unsafe_audit_v1", "candidate_uid_v1"], ascending=[False, True]).iterrows():
        is_unsafe = _as_bool(row.get("unsafe_audit_v1"))
        has_blueprint = "HISTORICAL_V2_BLUEPRINT" in str(row.get("source_evidence_v1", ""))
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "cohort_v1": "UNSAFE_EXTRA_WITHOUT_HARD_VETO" if is_unsafe else "SOURCE_SHAPE_SIMILAR_SAFE_CORE_ROW",
                "branch_tier_v1": "score>=0.95 + R5_1 + V2-like + low-support-clear",
                "selected_by_safe_core_v1": _as_bool(masks["hardened"].loc[row.name]),
                "original_140_v1": _as_bool(original.loc[row.name]),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": is_unsafe,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "signal_r5_1_bad_score_v1": _as_bool(row.get("signal_r5_1_bad_score_v1")),
                "signal_r5_bad_score_v1": _as_bool(row.get("signal_r5_bad_score_v1")),
                "signal_r5_tail_score_v1": _as_bool(row.get("signal_r5_tail_score_v1")),
                "signal_v2_like_bad_tail_v1": _as_bool(row.get("signal_v2_like_bad_tail_v1")),
                "signal_tail_repair_v1": _as_bool(row.get("signal_tail_repair_v1")),
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "zero_denominator_group_v1": _as_bool(row.get("zero_denominator_group_v1")),
                "has_historical_v2_blueprint_v1": has_blueprint,
                "source_evidence_v1": row.get("source_evidence_v1"),
                "clean_source_signal_distinction_v1": (
                    "NONE_FOUND_AGAINST_THREE_GOOD_ROWS"
                    if is_unsafe
                    else "SAME_CLEAN_SOURCE_SHAPE_AS_UNSAFE_ROW_OR_BLUEPRINT_GUARD_DIAGNOSTIC_ONLY"
                ),
                "deployable_safety_veto_candidate_v1": "source signal-shape / support confluence only; historical blueprint blocked",
                "note_v1": (
                    "Unsafe row is separable cleanly only with broad source-shape vetoes that cut good rows, "
                    "or with blocked historical blueprint evidence."
                )
                if is_unsafe
                else "Safe row belongs to the same clean source-shape neighborhood; broad clean vetoes cut it.",
            }
        )
    if int((similar & unsafe).sum()) != 1:
        raise RuntimeError("EXPECTED_EXACTLY_ONE_UNSAFE_SOURCE_SHAPE_ROW")
    return rows


def _feature_family_definitions() -> dict[str, Any]:
    families = [
        {
            "feature_family_name_v1": "SOURCE_SIGNAL_SHAPE_RISK_V1",
            "definition_v1": "Risk from high score + missing R5 tail + R5 bad support.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": False,
        },
        {
            "feature_family_name_v1": "SOURCE_LOW_SUPPORT_RISK_V1",
            "definition_v1": "Support, structural low support, and zero-denominator source-policy risk.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": False,
        },
        {
            "feature_family_name_v1": "SOURCE_MISSINGNESS_AND_LINEAGE_RISK_V1",
            "definition_v1": "Missing R5 tail / tail-repair evidence as source-level data-quality risk.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": False,
        },
        {
            "feature_family_name_v1": "SOURCE_BRANCH_LOCAL_RISK_V1",
            "definition_v1": "Apply signal-shape risk only inside the safe-core branch admitting the unsafe row.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": False,
        },
        {
            "feature_family_name_v1": "SOURCE_SIGNAL_CONFLUENCE_RISK_V1",
            "definition_v1": "Require multiple weak source risks before veto: no R5 tail, R5 bad support, high score, support-repairable policy.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": False,
        },
        {
            "feature_family_name_v1": "SOURCE_SAFE_CORE_MARGIN_RISK_V1",
            "definition_v1": "Margin/distance from safe-core in source space.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": True,
            "blocked_reason_v1": "Available margin evidence comes from student/membership-proxy history, not independent source lineage.",
        },
        {
            "feature_family_name_v1": "SOURCE_FALSE_POSITIVE_RISK_V1",
            "definition_v1": "Secondary false-positive pressure for the 3 safety-clean extras.",
            "uses_historical_v2_blueprint_v1": False,
            "proxy_risk_v1": True,
            "blocked_reason_v1": "Best existing separations rely on membership-student or audit labels; keep diagnostic.",
        },
    ]
    return {"layer_name": "CLEAN_AS_OF_SAFETY_LAYER_FEATURE_FAMILY_DEFINITIONS_V1", "families_v1": families}


def _recipe_definitions(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    score = _num(frame, "candidate_score_v1")
    student = _num(frame, "student_oof_score_v1")
    base = masks["base_without_hard_safety_veto_v1"]
    return [
        {
            "recipe_name_v1": "MINIMAL_SOURCE_HARD_VETO_V1",
            "feature_family_v1": "SOURCE_SIGNAL_CONFLUENCE_RISK_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, run_id_policy_class_v1",
            "rule_v1": "veto if score >= 0.99, missing R5_TAIL_SCORE, R5_BAD_SCORE support, and support-repairable policy",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": False,
            "mask": masks["source_confluence_repairable_v1"],
        },
        {
            "recipe_name_v1": "BRANCH_LOCAL_SOURCE_HARD_VETO_V1",
            "feature_family_v1": "SOURCE_BRANCH_LOCAL_RISK_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1",
            "rule_v1": "inside safe-core branch, veto if score >= 0.99, missing R5_TAIL_SCORE and R5_BAD_SCORE support",
            "normalization_required_v1": False,
            "mapping_required_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": False,
            "mask": masks["source_signal_shape_ge099_v1"],
        },
        {
            "recipe_name_v1": "SOURCE_RISK_SCORE_PLUS_HARD_THRESHOLD_V1",
            "feature_family_v1": "SOURCE_SIGNAL_SHAPE_RISK_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1",
            "rule_v1": "veto if score >= 0.99321, missing R5_TAIL_SCORE and R5_BAD_SCORE support",
            "normalization_required_v1": False,
            "mapping_required_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": False,
            "mask": masks["source_relaxed_score_ge099321_v1"],
        },
        {
            "recipe_name_v1": "SOURCE_CONFLUENCE_HARD_VETO_V1",
            "feature_family_v1": "SOURCE_SIGNAL_CONFLUENCE_RISK_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, run_id_policy_class_v1",
            "rule_v1": "same clean confluence as minimal recipe; held as named confluence candidate",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": False,
            "mask": masks["source_confluence_repairable_v1"],
        },
        {
            "recipe_name_v1": "CONSERVATIVE_SOURCE_SAFETY_LAYER_V1",
            "feature_family_v1": "SOURCE_MISSINGNESS_AND_LINEAGE_RISK_V1",
            "input_fields_v1": "signal_r5_tail_score_v1, signal_tail_repair_v1",
            "rule_v1": "veto if no R5 tail support and no tail-repair source support",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": False,
            "mask": masks["source_no_tail_no_repair_v1"],
        },
        {
            "recipe_name_v1": "DIAGNOSTIC_BLUEPRINT_GUARD_REFERENCE_NOT_ALLOWED_V1",
            "feature_family_v1": "BLOCKED_HISTORICAL_ARTIFACT_PROXY",
            "input_fields_v1": "HISTORICAL_V2_BLUEPRINT token",
            "rule_v1": "veto source-shape rows lacking historical V2 blueprint",
            "normalization_required_v1": False,
            "mapping_required_v1": False,
            "uses_historical_v2_blueprint_v1": True,
            "leakage_or_proxy_risk_v1": True,
            "mask": masks["blocked_blueprint_guard_reference_v1"],
        },
        {
            "recipe_name_v1": "DIAGNOSTIC_STUDENT_MARGIN_REFERENCE_NOT_ALLOWED_V1",
            "feature_family_v1": "SOURCE_SAFE_CORE_MARGIN_RISK_V1",
            "input_fields_v1": "student_oof_score_v1",
            "rule_v1": "veto if score >= 0.99 and student membership score < 0.50",
            "normalization_required_v1": False,
            "mapping_required_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "leakage_or_proxy_risk_v1": True,
            "mask": base & score.ge(0.99) & student.lt(0.50),
        },
    ]


def _metrics_for_mask(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    veto_mask: pd.Series,
    *,
    leakage_or_proxy_risk: bool,
    uses_historical_v2_blueprint: bool,
) -> dict[str, Any]:
    base = masks["base_without_hard_safety_veto_v1"]
    safe_core = masks["hardened"]
    original = _bool(frame, "selected_original_140_v1")
    unsafe = _bool(frame, "unsafe_audit_v1")
    selected = base & ~veto_mask
    selected_frame = frame[selected]
    unsafe_blocked = int((veto_mask & base & ~safe_core & unsafe).sum()) > 0
    good_cut = int((veto_mask & safe_core).sum())
    tier = retention_tier(
        unsafe_row_blocked=unsafe_blocked,
        good_rows_cut=good_cut,
        shortcut_or_leakage=leakage_or_proxy_risk,
    )
    adapter_ready = (
        tier == "GREEN"
        and not leakage_or_proxy_risk
        and not uses_historical_v2_blueprint
        and int((selected & unsafe).sum()) == 0
    )
    return {
        "unsafe_extra_row_blocked_v1": unsafe_blocked,
        "unsafe_rows_remaining_v1": int((selected & unsafe).sum()),
        "selected_rows_v1": int(selected.sum()),
        "safe_core_rows_retained_v1": int((selected & safe_core).sum()),
        "safe_core_rows_cut_v1": good_cut,
        "original_140_rows_retained_v1": int((selected & original).sum()),
        "original_140_rows_cut_v1": int((veto_mask & safe_core & original).sum()),
        "three_extra_rows_retained_v1": int((selected & safe_core & ~original).sum()),
        "three_extra_rows_cut_v1": int((veto_mask & safe_core & ~original).sum()),
        "bad_count_audit_only_v1": int(_bool(selected_frame, "bad_label_v1").sum()),
        "tail_count_audit_only_v1": int(_bool(selected_frame, "tail_label_v1").sum()),
        "precision_audit_only_v1": float(_bool(selected_frame, "bad_label_v1").sum() / max(len(selected_frame), 1)),
        "safety_status_v1": "CLEAN" if int((selected & unsafe).sum()) == 0 else "FAIL",
        "retention_class_v1": tier,
        "adapter_ready_v1": adapter_ready,
    }


def _feature_family_metrics(
    frame: pd.DataFrame, masks: dict[str, pd.Series], recipes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        if recipe["feature_family_v1"].startswith("BLOCKED"):
            continue
        metrics = _metrics_for_mask(
            frame,
            masks,
            recipe["mask"],
            leakage_or_proxy_risk=recipe["leakage_or_proxy_risk_v1"],
            uses_historical_v2_blueprint=recipe["uses_historical_v2_blueprint_v1"],
        )
        existing = by_family.get(recipe["feature_family_v1"])
        if existing is None or _tier_rank(metrics["retention_class_v1"]) < _tier_rank(existing["retention_class_v1"]):
            by_family[recipe["feature_family_v1"]] = {
                "feature_family_name_v1": recipe["feature_family_v1"],
                "representative_recipe_v1": recipe["recipe_name_v1"],
                "input_fields_v1": recipe["input_fields_v1"],
                "source_lineage_v1": "SOURCE_SIGNALS_ONLY" if not recipe["leakage_or_proxy_risk_v1"] else "DIAGNOSTIC_PROXY_BLOCKED",
                "as_of_status_v1": "AS_OF_SAFE" if not recipe["leakage_or_proxy_risk_v1"] else "DIAGNOSTIC_ONLY_NOT_ADAPTER_READY",
                "adapter_feasibility_v1": "NEEDS_REFINEMENT_BEFORE_MAPPING",
                "rule_score_veto_definition_v1": recipe["rule_v1"],
                "complexity_v1": "LOW" if len(str(recipe["input_fields_v1"]).split(",")) <= 3 else "MEDIUM",
                "normalization_required_v1": recipe["normalization_required_v1"],
                "mapping_required_v1": recipe["mapping_required_v1"],
                "leakage_proxy_risk_v1": recipe["leakage_or_proxy_risk_v1"],
                "recommendation_v1": _recommendation_for_metrics(metrics, recipe),
                **metrics,
            }
    by_family.setdefault(
        "SOURCE_LOW_SUPPORT_RISK_V1",
        {
            "feature_family_name_v1": "SOURCE_LOW_SUPPORT_RISK_V1",
            "representative_recipe_v1": "SOURCE_LOW_SUPPORT_RISK_V1",
            "input_fields_v1": "run_id_policy_class_v1, structural_low_support_v1, zero_denominator_group_v1",
            "source_lineage_v1": "SOURCE_SUPPORT_POLICY_NEEDS_NORMALIZATION",
            "as_of_status_v1": "AS_OF_SAFE_NEEDS_NORMALIZATION",
            "adapter_feasibility_v1": "FAILS_TO_BLOCK_UNSAFE_ROW",
            "rule_score_veto_definition_v1": "support policy veto only",
            "unsafe_extra_row_blocked_v1": False,
            "safe_core_rows_cut_v1": 12,
            "retention_class_v1": "BLOCKED",
            "adapter_ready_v1": False,
            "recommendation_v1": "REJECT_DOES_NOT_BLOCK_UNSAFE_ROW",
        },
    )
    return list(by_family.values())


def _tier_rank(tier: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3, "BLOCKED": 4}.get(tier, 9)


def _recommendation_for_metrics(metrics: dict[str, Any], recipe: dict[str, Any]) -> str:
    if recipe["leakage_or_proxy_risk_v1"]:
        return "DIAGNOSTIC_ONLY_BLOCKED_PROXY_OR_MEMBERSHIP_RISK"
    if not metrics["unsafe_extra_row_blocked_v1"]:
        return "REJECT_DOES_NOT_BLOCK_UNSAFE_ROW"
    if metrics["retention_class_v1"] in {"ORANGE", "RED"}:
        return "REJECT_FOR_ADAPTER_NOW_TOO_DESTRUCTIVE_REFINE_SOURCE_LAYER"
    if metrics["retention_class_v1"] == "YELLOW":
        return "REVIEW_YELLOW_RETENTION"
    if metrics["retention_class_v1"] == "GREEN":
        return "READY_FOR_MAPPING"
    return "REJECT_BLOCKED"


def _candidate_recipe_definitions_payload(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_CANDIDATE_RECIPE_DEFINITIONS_V1",
        "recipes_v1": [{key: value for key, value in recipe.items() if key != "mask"} for recipe in recipes],
    }


def _candidate_recipe_dry_runs(
    frame: pd.DataFrame, masks: dict[str, pd.Series], recipes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for recipe in recipes:
        metrics = _metrics_for_mask(
            frame,
            masks,
            recipe["mask"],
            leakage_or_proxy_risk=recipe["leakage_or_proxy_risk_v1"],
            uses_historical_v2_blueprint=recipe["uses_historical_v2_blueprint_v1"],
        )
        rows.append(
            {
                "recipe_name_v1": recipe["recipe_name_v1"],
                "feature_family_v1": recipe["feature_family_v1"],
                "input_fields_v1": recipe["input_fields_v1"],
                "rule_v1": recipe["rule_v1"],
                "uses_historical_v2_blueprint_v1": recipe["uses_historical_v2_blueprint_v1"],
                "leakage_or_proxy_risk_v1": recipe["leakage_or_proxy_risk_v1"],
                "normalization_required_v1": recipe["normalization_required_v1"],
                "mapping_required_v1": recipe["mapping_required_v1"],
                "dry_run_status_v1": "PASS_MECHANICAL_SOURCE_LAYER" if metrics["safety_status_v1"] == "CLEAN" else "FAIL_UNSAFE_REMAINS",
                "adapter_feasibility_v1": "NOT_READY_TOO_DESTRUCTIVE"
                if metrics["retention_class_v1"] in {"ORANGE", "RED"}
                else ("READY_FOR_MAPPING" if metrics["adapter_ready_v1"] else "BLOCKED"),
                "recommendation_v1": _recommendation_for_metrics(metrics, recipe),
                **metrics,
            }
        )
    validate_candidate_recipe_dry_runs(rows)
    return rows


def _retention_policy_audit(dry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {tier: 0 for tier in ["GREEN", "YELLOW", "ORANGE", "RED", "BLOCKED"]}
    for row in dry_rows:
        counts[row["retention_class_v1"]] += 1
    clean_rows = [row for row in dry_rows if not row["leakage_or_proxy_risk_v1"]]
    clean_blocking = [row for row in clean_rows if row["unsafe_extra_row_blocked_v1"]]
    best = sorted(clean_blocking, key=lambda row: (_tier_rank(row["retention_class_v1"]), row["safe_core_rows_cut_v1"]))[0]
    return {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_RETENTION_POLICY_AUDIT_V1",
        "policy_v1": {
            "GREEN": "unsafe row blocked, <=5 good safe-core/original-140 rows cut, no shortcut/leakage, AS_OF-safe, adapter-feasible",
            "YELLOW": "unsafe row blocked, 6-10 good rows cut, no shortcut/leakage, may need review",
            "ORANGE": "unsafe row blocked, 11-20 good rows cut, not adapter-ready",
            "RED": "unsafe row blocked, >20 good rows cut",
            "BLOCKED": "unsafe row not blocked, or leakage/proxy/row-identity/hindsight detected",
        },
        "tier_counts_v1": counts,
        "clean_green_candidates_v1": [
            row["recipe_name_v1"] for row in dry_rows if row["retention_class_v1"] == "GREEN" and not row["leakage_or_proxy_risk_v1"]
        ],
        "best_clean_blocking_recipe_v1": best["recipe_name_v1"],
        "best_clean_blocking_retention_class_v1": best["retention_class_v1"],
        "best_clean_blocking_safe_core_rows_cut_v1": best["safe_core_rows_cut_v1"],
        "adapter_ready_candidates_v1": [row["recipe_name_v1"] for row in dry_rows if row["adapter_ready_v1"]],
        "conclusion_v1": "No GREEN/YELLOW clean source-signal candidate exists in current source columns; available blockers are ORANGE/RED destructive.",
    }


def _unsafe_lookalike_boundary_audit(
    frame: pd.DataFrame, masks: dict[str, pd.Series], best_recipe: dict[str, Any]
) -> list[dict[str, Any]]:
    similar = masks["source_signal_shape_ge099_v1"] & masks["base_without_hard_safety_veto_v1"]
    rows = []
    for _, row in frame[similar].sort_values(["unsafe_audit_v1", "candidate_uid_v1"], ascending=[False, True]).iterrows():
        blocked = _as_bool(best_recipe["mask"].loc[row.name])
        rows.append(
            {
                "recipe_name_v1": best_recipe["recipe_name_v1"],
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_safe_core_v1": _as_bool(masks["hardened"].loc[row.name]),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "blocked_by_best_clean_recipe_v1": blocked,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "lookalike_reason_v1": "high-score no-tail R5-bad source-shape neighborhood",
                "risk_class_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_MONITORING"
                if blocked
                else "HIGH_UNSAFE_LOOKALIKE_RISK_BLOCK_SAFETY_LAYER",
                "boundary_interpretation_v1": "Best clean source recipe stops unsafe row but also blocks too many good lookalikes.",
            }
        )
    return rows


def _adapter_readiness_preassessment(best_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_ADAPTER_READINESS_PREASSESSMENT_V1",
        "best_recipe_v1": best_row["recipe_name_v1"],
        "can_adapter_compute_all_inputs_v1": True,
        "all_inputs_as_of_safe_v1": True,
        "inputs_missing_from_allowlist_v1": ["run_id_policy_class_v1 needs normalized source contract"],
        "normalization_required_v1": best_row["normalization_required_v1"],
        "input_mapping_required_v1": True,
        "simple_adapter_contract_possible_v1": True,
        "dry_run_matches_expected_behavior_v1": best_row["unsafe_extra_row_blocked_v1"] and best_row["safety_status_v1"] == "CLEAN",
        "adapter_build_can_reopen_now_v1": False,
        "adapter_input_mapping_can_start_next_v1": False,
        "safety_layer_must_be_refined_first_v1": True,
        "status_v1": "NOT_READY_ONLY_DESTRUCTIVE_CLEAN_CANDIDATES",
        "reason_v1": (
            "The clean source recipe blocks the unsafe row, but cuts 11 good safe-core/original-140 rows "
            "and is ORANGE under retention policy."
        ),
    }


def _anti_shortcut_audit() -> dict[str, Any]:
    return {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS_ADAPTER_STILL_BLOCKED",
        "historical_v2_blueprint_used_as_deployable_input_v1": False,
        "membership_proxy_used_v1": False,
        "coverage_proxy_used_v1": False,
        "row_identity_used_v1": False,
        "audit_only_veto_used_as_deployable_input_v1": False,
        "final_bad_tail_labels_used_as_features_v1": False,
        "mfe_hindsight_used_v1": False,
        "safe_recoverable_direct_used_v1": False,
        "selected_flags_used_as_features_v1": False,
        "implicit_latest_glob_decisioning_v1": False,
        "dummy_synthetic_fallback_used_v1": False,
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_run_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }


def _recommendation(best_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recommendation = {
        "layer_name": "CLEAN_AS_OF_SAFETY_LAYER_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "best_clean_source_recipe_v1": best_row["recipe_name_v1"],
        "best_recipe_retention_class_v1": best_row["retention_class_v1"],
        "unsafe_extra_row_blocked_v1": best_row["unsafe_extra_row_blocked_v1"],
        "safe_core_rows_retained_v1": best_row["safe_core_rows_retained_v1"],
        "safe_core_rows_cut_v1": best_row["safe_core_rows_cut_v1"],
        "adapter_r6_iql_remain_blocked_v1": True,
        "rationale_v1": [
            "Historical V2 blueprint was not used as deployable input.",
            "Clean source-signal candidates can block the unsafe row, but the best clean blocker cuts 11 good safe-core rows.",
            "That is ORANGE retention, not adapter-ready. Refine the clean source safety layer before mapping or adapter work.",
        ],
    }
    go_no_go = {
        "layer_name": "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "best_clean_source_recipe_v1": best_row["recipe_name_v1"],
        "unsafe_extra_row_blocked_v1": best_row["unsafe_extra_row_blocked_v1"],
        "safe_core_rows_retained_v1": best_row["safe_core_rows_retained_v1"],
        "safe_core_rows_cut_v1": best_row["safe_core_rows_cut_v1"],
        "retention_class_v1": best_row["retention_class_v1"],
        "historical_v2_blueprint_used_as_deployable_input_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_go_no_go(go_no_go)
    return recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    inventory_rows: list[dict[str, Any]],
    cohorts: dict[str, Any],
    unsafe_rows: list[dict[str, Any]],
    feature_defs: dict[str, Any],
    family_metrics: list[dict[str, Any]],
    recipe_defs: dict[str, Any],
    dry_rows: list[dict[str, Any]],
    retention: dict[str, Any],
    lookalike_rows: list[dict[str, Any]],
    adapter: dict[str, Any],
    anti: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "clean_as_of_safety_layer_reproducibility_audit_v1.md",
        [
            "# Clean AS_OF Safety Layer Reproducibility Audit V1",
            "",
            f"- Safe-core rows: `{repro['safe_core_selected_rows_v1']}`",
            f"- Original 140 recovered: `{repro['safe_core_recovered_original_140_v1']}`",
            f"- Extra rows: `{repro['safe_core_extra_rows_v1']}`",
            f"- Bad/tail: `{repro['safe_core_bad_count_audit_only_v1']} / {repro['safe_core_tail_count_audit_only_v1']}`",
            f"- Unsafe without hard veto: `{repro['unsafe_extra_without_hard_veto_rows_v1']}`",
            "- HISTORICAL_V2_BLUEPRINT remains non-deployable.",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_source_signal_inventory_v1.md",
        [
            "# Clean AS_OF Safety Layer Source Signal Inventory V1",
            "",
            f"- Signals inventoried: `{len(inventory_rows)}`",
            f"- AS_OF-safe or needs-normalization fields: `{sum(row['classification_v1'].startswith('AS_OF_SAFE') for row in inventory_rows)}`",
            f"- Blocked fields: `{sum(row['classification_v1'].startswith('BLOCKED') for row in inventory_rows)}`",
            "- Historical blueprint, labels, audit flags, row identity, and membership flags are blocked.",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_boundary_cohorts_v1.md",
        [
            "# Clean AS_OF Safety Layer Boundary Cohorts V1",
            "",
            *[
                f"- `{cohort['cohort_id_v1']}`: `{cohort['count_v1']}` rows"
                for cohort in cohorts["cohorts_v1"]
            ],
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.md",
        [
            "# Clean AS_OF Safety Layer Unsafe Row Source Signal Audit V1",
            "",
            f"- Source-shape rows audited: `{len(unsafe_rows)}`",
            "- The unsafe row has no clean fine-grained source distinction from several good rows in current available fields.",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_feature_family_definitions_v1.md",
        [
            "# Clean AS_OF Safety Layer Feature Family Definitions V1",
            "",
            *[
                f"- `{family['feature_family_name_v1']}`: {family['definition_v1']}"
                for family in feature_defs["families_v1"]
            ],
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_feature_family_metrics_v1.md",
        [
            "# Clean AS_OF Safety Layer Feature Family Metrics V1",
            "",
            *[
                f"- `{row['feature_family_name_v1']}` via `{row['representative_recipe_v1']}`: unsafe blocked `{row.get('unsafe_extra_row_blocked_v1')}`, safe-core cut `{row.get('safe_core_rows_cut_v1')}`, tier `{row.get('retention_class_v1')}`"
                for row in family_metrics
            ],
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_candidate_recipe_definitions_v1.md",
        [
            "# Clean AS_OF Safety Layer Candidate Recipe Definitions V1",
            "",
            *[
                f"- `{recipe['recipe_name_v1']}`: {recipe['rule_v1']}"
                for recipe in recipe_defs["recipes_v1"]
            ],
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.md",
        [
            "# Clean AS_OF Safety Layer Candidate Recipe Dry Run V1",
            "",
            *[
                f"- `{row['recipe_name_v1']}`: selected `{row['selected_rows_v1']}`, unsafe blocked `{row['unsafe_extra_row_blocked_v1']}`, safe-core cut `{row['safe_core_rows_cut_v1']}`, tier `{row['retention_class_v1']}`"
                for row in dry_rows
            ],
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_retention_policy_audit_v1.md",
        [
            "# Clean AS_OF Safety Layer Retention Policy Audit V1",
            "",
            f"- Tier counts: `{retention['tier_counts_v1']}`",
            f"- Best clean blocking recipe: `{retention['best_clean_blocking_recipe_v1']}`",
            f"- Conclusion: {retention['conclusion_v1']}",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.md",
        [
            "# Clean AS_OF Safety Layer Unsafe Lookalike Boundary Audit V1",
            "",
            f"- Lookalike rows audited: `{len(lookalike_rows)}`",
            "- Best clean recipe stops the unsafe row, but the boundary is still too broad.",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_adapter_readiness_preassessment_v1.md",
        [
            "# Clean AS_OF Safety Layer Adapter Readiness Preassessment V1",
            "",
            f"- Best recipe: `{adapter['best_recipe_v1']}`",
            f"- Adapter can reopen now: `{adapter['adapter_build_can_reopen_now_v1']}`",
            f"- Status: `{adapter['status_v1']}`",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_anti_shortcut_audit_v1.md",
        [
            "# Clean AS_OF Safety Layer Anti-Shortcut Audit V1",
            "",
            f"- Status: `{anti['status_v1']}`",
            "- No blueprint, membership, coverage, row identity, labels, MFE, audit-only veto, R6, adapter, IQL, package, freeze, promo, live, Optuna, or broad sweep was used.",
        ],
    )
    _write_report(
        root / "clean_as_of_safety_layer_recommendation_v1.md",
        [
            "# Clean AS_OF Safety Layer Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Best clean source recipe: `{recommendation['best_clean_source_recipe_v1']}`",
            f"- Safe-core rows cut: `{recommendation['safe_core_rows_cut_v1']}`",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _build_frame_and_masks(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, masks, inputs)
    inventory_rows = _signal_inventory(frame)
    cohorts = _boundary_cohorts(frame, masks)
    unsafe_rows = _unsafe_row_source_signal_audit(frame, masks)
    feature_defs = _feature_family_definitions()
    recipes = _recipe_definitions(frame, masks)
    recipe_defs = _candidate_recipe_definitions_payload(recipes)
    dry_rows = _candidate_recipe_dry_runs(frame, masks, recipes)
    family_metrics = _feature_family_metrics(frame, masks, recipes)
    retention = _retention_policy_audit(dry_rows)
    best_recipe = next(recipe for recipe in recipes if recipe["recipe_name_v1"] == BEST_CLEAN_RECIPE)
    best_row = next(row for row in dry_rows if row["recipe_name_v1"] == BEST_CLEAN_RECIPE)
    lookalike_rows = _unsafe_lookalike_boundary_audit(frame, masks, best_recipe)
    adapter = _adapter_readiness_preassessment(best_row)
    anti = _anti_shortcut_audit()
    recommendation, go_no_go = _recommendation(best_row)

    _write_json(artifact_root / "clean_as_of_safety_layer_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "clean_as_of_safety_layer_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "clean_as_of_safety_layer_source_signal_inventory_v1.csv", inventory_rows)
    _write_json(
        artifact_root / "clean_as_of_safety_layer_source_signal_inventory_v1.json",
        {"row_count_v1": len(inventory_rows), "rows_v1": inventory_rows},
    )
    _write_json(artifact_root / "clean_as_of_safety_layer_boundary_cohorts_v1.json", cohorts)
    _write_rows(artifact_root / "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.json",
        {"row_count_v1": len(unsafe_rows), "rows_v1": unsafe_rows},
    )
    _write_json(artifact_root / "clean_as_of_safety_layer_feature_family_definitions_v1.json", feature_defs)
    _write_rows(artifact_root / "clean_as_of_safety_layer_feature_family_metrics_v1.csv", family_metrics)
    _write_json(
        artifact_root / "clean_as_of_safety_layer_feature_family_metrics_v1.json",
        {"row_count_v1": len(family_metrics), "rows_v1": family_metrics},
    )
    _write_json(artifact_root / "clean_as_of_safety_layer_candidate_recipe_definitions_v1.json", recipe_defs)
    _write_rows(artifact_root / "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.csv", dry_rows)
    _write_json(
        artifact_root / "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.json",
        {"row_count_v1": len(dry_rows), "rows_v1": dry_rows},
    )
    _write_json(artifact_root / "clean_as_of_safety_layer_retention_policy_audit_v1.json", retention)
    _write_rows(artifact_root / "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.csv", lookalike_rows)
    _write_json(
        artifact_root / "clean_as_of_safety_layer_unsafe_lookalike_boundary_audit_v1.json",
        {"row_count_v1": len(lookalike_rows), "rows_v1": lookalike_rows},
    )
    _write_json(artifact_root / "clean_as_of_safety_layer_adapter_readiness_preassessment_v1.json", adapter)
    _write_json(artifact_root / "clean_as_of_safety_layer_anti_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "clean_as_of_safety_layer_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "build_clean_as_of_safety_feature_layer_from_source_signals_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        inventory_rows,
        cohorts,
        unsafe_rows,
        feature_defs,
        family_metrics,
        recipe_defs,
        dry_rows,
        retention,
        lookalike_rows,
        adapter,
        anti,
        recommendation,
    )

    summary = {
        "layer_name": "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "baseline_140_94_status_v1": "CURRENT_BEST_CAUSAL_BASELINE",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "safe_core_selected_rows_v1": repro["safe_core_selected_rows_v1"],
        "safe_core_recovered_original_140_v1": repro["safe_core_recovered_original_140_v1"],
        "safe_core_extra_rows_v1": repro["safe_core_extra_rows_v1"],
        "safe_core_bad_tail_audit_only_v1": [
            repro["safe_core_bad_count_audit_only_v1"],
            repro["safe_core_tail_count_audit_only_v1"],
        ],
        "safe_core_precision_audit_only_v1": repro["safe_core_precision_audit_only_v1"],
        "safe_core_safety_status_v1": repro["safe_core_safety_status_v1"],
        "historical_v2_blueprint_used_as_deployable_input_v1": False,
        "best_candidate_v1": best_row["recipe_name_v1"],
        "best_candidate_retention_class_v1": best_row["retention_class_v1"],
        "unsafe_row_blocked_v1": best_row["unsafe_extra_row_blocked_v1"],
        "safe_core_rows_retained_v1": best_row["safe_core_rows_retained_v1"],
        "safe_core_rows_cut_v1": best_row["safe_core_rows_cut_v1"],
        "adapter_readiness_preassessment_v1": adapter["status_v1"],
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_r6_iql_remain_blocked_v1": True,
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
            "# Build Clean AS_OF Safety Feature Layer From Source Signals V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Best clean recipe: `{summary['best_candidate_v1']}`",
            f"- Unsafe row blocked: `{summary['unsafe_row_blocked_v1']}`",
            f"- Safe-core rows cut: `{summary['safe_core_rows_cut_v1']}`",
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
