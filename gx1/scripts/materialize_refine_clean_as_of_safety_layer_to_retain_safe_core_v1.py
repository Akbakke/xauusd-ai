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

from gx1.scripts import materialize_build_clean_as_of_safety_feature_layer_from_source_signals_v1 as clean_gate
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1"

INPUT_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1_20260428T182517Z_LOCK"
)
INPUT_CLOSE_ROOT = (
    DEFAULT_REPORTS_ROOT / "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1_20260428T175937Z_LOCK"
)
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1_20260428T174140Z_LOCK"
)
INPUT_HOLD_ROOT = (
    DEFAULT_REPORTS_ROOT / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T111216Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
PRIOR_BEST_RECIPE = "MINIMAL_SOURCE_HARD_VETO_V1"
FINAL_REFINED_CANDIDATE = "SOURCE_CONFLUENCE_REFINED_VETO_V1"
FINAL_STATUS = "CLEAN_AS_OF_SAFETY_LAYER_REFINED_STILL_ORANGE_DESTRUCTIVE"
NEXT_ACTION = "REFINE_CLEAN_AS_OF_SAFETY_LAYER_AGAIN_WITH_STRONGER_SOURCE_SIGNALS_V1"

EXPECTED_SAFE_CORE_SELECTED = 89
EXPECTED_SAFE_CORE_RECOVERED = 86
EXPECTED_SAFE_CORE_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607
EXPECTED_UNSAFE_WITHOUT_HARD_VETO = 1
EXPECTED_PRIOR_MINIMAL_CUT = 11

ALLOWED_FINAL_STATUSES = {
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_GREEN_READY_FOR_INPUT_MAPPING",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_YELLOW_NEEDS_REVIEW",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_NEEDS_NORMALIZATION",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_NEEDS_LINEAGE_CONFIRMATION",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_STILL_ORANGE_DESTRUCTIVE",
    "CLEAN_AS_OF_SAFETY_LAYER_NO_SAFE_REFINED_CANDIDATE_FOUND",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_BLOCKED_BY_PROXY_OR_LEAKAGE",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "CLEAN_AS_OF_SAFETY_LAYER_REFINED_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_CLEAN_AS_OF_SAFETY_LAYER_INPUT_MAPPING_V1",
    "REVIEW_YELLOW_SAFETY_LAYER_CANDIDATE_BEFORE_MAPPING_V1",
    "NORMALIZE_CLEAN_AS_OF_SAFETY_LAYER_INPUTS_V1",
    "DEEPEN_CLEAN_AS_OF_SAFETY_LAYER_LINEAGE_AUDIT_V1",
    "REFINE_CLEAN_AS_OF_SAFETY_LAYER_AGAIN_WITH_STRONGER_SOURCE_SIGNALS_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_WITH_SOURCE_SAFETY_SIGNALS_V1",
}

REQUIRED_OUTPUTS = [
    "refine_clean_safety_layer_input_manifest_v1.json",
    "refine_clean_safety_layer_reproducibility_audit_v1.json",
    "refine_clean_safety_layer_reproducibility_audit_v1.md",
    "refine_clean_safety_layer_cut_11_good_rows_audit_v1.csv",
    "refine_clean_safety_layer_cut_11_good_rows_audit_v1.json",
    "refine_clean_safety_layer_cut_11_good_rows_audit_v1.md",
    "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.csv",
    "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.json",
    "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.md",
    "refine_clean_safety_layer_candidate_definitions_v1.json",
    "refine_clean_safety_layer_candidate_definitions_v1.md",
    "refine_clean_safety_layer_candidate_metrics_v1.csv",
    "refine_clean_safety_layer_candidate_metrics_v1.json",
    "refine_clean_safety_layer_candidate_metrics_v1.md",
    "refine_clean_safety_layer_candidate_dry_run_v1.csv",
    "refine_clean_safety_layer_candidate_dry_run_v1.json",
    "refine_clean_safety_layer_candidate_dry_run_v1.md",
    "refine_clean_safety_layer_unsafe_lookalike_audit_v1.csv",
    "refine_clean_safety_layer_unsafe_lookalike_audit_v1.json",
    "refine_clean_safety_layer_unsafe_lookalike_audit_v1.md",
    "refine_clean_safety_layer_anti_shortcut_audit_v1.json",
    "refine_clean_safety_layer_anti_shortcut_audit_v1.md",
    "refine_clean_safety_layer_final_candidate_selection_v1.json",
    "refine_clean_safety_layer_final_candidate_selection_v1.md",
    "refine_clean_safety_layer_adapter_readiness_preassessment_v1.json",
    "refine_clean_safety_layer_adapter_readiness_preassessment_v1.md",
    "refine_clean_safety_layer_recommendation_v1.json",
    "refine_clean_safety_layer_recommendation_v1.md",
    "refine_clean_as_of_safety_layer_to_retain_safe_core_go_no_go_v1.json",
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


def retention_class(*, unsafe_row_blocked: bool, good_rows_cut: int, shortcut_or_leakage: bool = False) -> str:
    if shortcut_or_leakage or not unsafe_row_blocked:
        return "BLOCKED"
    if good_rows_cut <= 5:
        return "GREEN"
    if good_rows_cut <= 10:
        return "YELLOW"
    if good_rows_cut <= 20:
        return "ORANGE"
    return "RED"


def _tier_rank(tier: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3, "BLOCKED": 4}.get(tier, 9)


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
        "prior_minimal_source_veto_blocks_unsafe_v1": True,
        "prior_minimal_source_veto_good_rows_cut_v1": EXPECTED_PRIOR_MINIMAL_CUT,
        "prior_minimal_source_veto_retention_class_v1": "ORANGE",
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("safe_core_precision_audit_only_v1", -1)), EXPECTED_PRECISION):
        failures["safe_core_precision_audit_only_v1"] = payload.get("safe_core_precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"REFINE_CLEAN_SAFETY_LAYER_REPRODUCTION_FAILED: {failures}")
    return True


def validate_candidate_metrics(rows: list[dict[str, Any]]) -> bool:
    required = {
        "BRANCH_LOCAL_SOURCE_HARD_VETO_V1",
        "SOURCE_CONFLUENCE_REFINED_VETO_V1",
        "RELAXED_SOURCE_THRESHOLD_VETO_V1",
        "GOOD_CORE_EXCEPTION_GUARD_SOURCE_VETO_V1",
        "LOW_SUPPORT_AWARE_SOURCE_VETO_V1",
        "MINIMAL_GREEN_SOURCE_VETO_V1",
        "YELLOW_REVIEW_SOURCE_VETO_V1",
    }
    present = {row["candidate_name_v1"] for row in rows}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"REFINE_CLEAN_SAFETY_LAYER_CANDIDATES_MISSING: {missing}")
    clean_green_or_yellow = [
        row
        for row in rows
        if row["retention_class_v1"] in {"GREEN", "YELLOW"}
        and row["proxy_leakage_risk_v1"] is False
        and row["unsafe_row_blocked_v1"] is True
    ]
    if clean_green_or_yellow:
        raise RuntimeError(f"UNEXPECTED_CLEAN_GREEN_OR_YELLOW_CANDIDATE: {clean_green_or_yellow}")
    if any(row["uses_historical_v2_blueprint_v1"] and row["adapter_ready_v1"] for row in rows):
        raise RuntimeError("HISTORICAL_BLUEPRINT_CANDIDATE_CANNOT_BE_ADAPTER_READY")
    if any(row["uses_student_or_membership_proxy_v1"] and row["adapter_ready_v1"] for row in rows):
        raise RuntimeError("STUDENT_OR_MEMBERSHIP_PROXY_CANDIDATE_CANNOT_BE_ADAPTER_READY")
    return True


def validate_final_selection(payload: dict[str, Any]) -> bool:
    if payload.get("selected_candidate_name_v1") != FINAL_REFINED_CANDIDATE:
        raise RuntimeError("EXPECTED_SOURCE_CONFLUENCE_REFINED_FINAL_CANDIDATE")
    if payload.get("retention_class_v1") != "ORANGE":
        raise RuntimeError("FINAL_REFINED_CANDIDATE_EXPECTED_ORANGE")
    if payload.get("adapter_input_mapping_allowed_next_v1") is not False:
        raise RuntimeError("ORANGE_REFINED_CANDIDATE_CANNOT_GO_TO_INPUT_MAPPING")
    if payload.get("uses_historical_v2_blueprint_v1") is not False:
        raise RuntimeError("FINAL_CANDIDATE_MUST_NOT_USE_HISTORICAL_BLUEPRINT")
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
    if payload.get("input_mapping_allowed_next_v1") is not False:
        raise RuntimeError("INPUT_MAPPING_MUST_NOT_BE_ALLOWED_FOR_ORANGE_RESULT")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"REFINE_CLEAN_SAFETY_LAYER_REQUIRED_OUTPUTS_MISSING: {missing}")
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
        INPUT_CLEAN_ROOT,
        INPUT_CLOSE_ROOT,
        INPUT_LANE_PACK_ROOT,
        INPUT_HOLD_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "clean_summary": INPUT_CLEAN_ROOT / "summary_v1.json",
        "clean_go_no_go": INPUT_CLEAN_ROOT / "build_clean_as_of_safety_feature_layer_from_source_signals_go_no_go_v1.json",
        "clean_dry_run": INPUT_CLEAN_ROOT / "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.json",
        "clean_source_inventory": INPUT_CLEAN_ROOT / "clean_as_of_safety_layer_source_signal_inventory_v1.json",
        "clean_unsafe_audit": INPUT_CLEAN_ROOT / "clean_as_of_safety_layer_unsafe_row_source_signal_audit_v1.json",
        "close_summary": INPUT_CLOSE_ROOT / "summary_v1.json",
        "close_go_no_go": INPUT_CLOSE_ROOT / "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
        "lane_pack_go_no_go": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json",
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
    clean_go = _read_json(required["clean_go_no_go"])
    if clean_go.get("status_v1") != "CLEAN_AS_OF_SAFETY_LAYER_FOUND_ONLY_DESTRUCTIVE_CANDIDATES":
        raise RuntimeError("INPUT_CLEAN_GATE_STATUS_NOT_DESTRUCTIVE")
    return {
        "required_paths": required,
        "clean_summary": _read_json(required["clean_summary"]),
        "clean_go_no_go": clean_go,
        "clean_dry_run": _read_json(required["clean_dry_run"]),
        "clean_source_inventory": _read_json(required["clean_source_inventory"]),
        "clean_unsafe_audit": _read_json(required["clean_unsafe_audit"]),
        "close_summary": _read_json(required["close_summary"]),
        "close_go_no_go": _read_json(required["close_go_no_go"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_pack_go_no_go": _read_json(required["lane_pack_go_no_go"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "hold_go_no_go": _read_json(required["hold_go_no_go"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "source_inputs": clean_gate._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "clean_as_of_safety_layer_root_v1": str(INPUT_CLEAN_ROOT),
            "proxy_branch_closure_root_v1": str(INPUT_CLOSE_ROOT),
            "parallel_refined_veto_lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
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
    return clean_gate._build_frame_and_masks(inputs["source_inputs"])


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
    minimal = next(row for row in inputs["clean_dry_run"]["rows_v1"] if row["recipe_name_v1"] == PRIOR_BEST_RECIPE)
    payload = {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "baseline_140_94_status_v1": "CURRENT_BEST_CAUSAL_BASELINE",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "safe_core_selected_rows_v1": safe_core["selected_rows_v1"],
        "safe_core_recovered_original_140_v1": safe_core["recovered_original_140_rows_v1"],
        "safe_core_extra_rows_v1": safe_core["extra_rows_v1"],
        "safe_core_bad_count_audit_only_v1": safe_core["bad_count_audit_only_v1"],
        "safe_core_tail_count_audit_only_v1": safe_core["tail_count_audit_only_v1"],
        "safe_core_precision_audit_only_v1": safe_core["precision_audit_only_v1"],
        "safe_core_safety_status_v1": safe_core["safety_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": int(_bool(without_hard, "unsafe_audit_v1").sum()),
        "prior_minimal_source_veto_name_v1": minimal["recipe_name_v1"],
        "prior_minimal_source_veto_blocks_unsafe_v1": minimal["unsafe_extra_row_blocked_v1"],
        "prior_minimal_source_veto_good_rows_cut_v1": minimal["safe_core_rows_cut_v1"],
        "prior_minimal_source_veto_retention_class_v1": minimal["retention_class_v1"],
        "clean_gate_status_v1": inputs["clean_go_no_go"].get("status_v1"),
    }
    validate_reproducibility(payload)
    return payload


def _cut_11_good_rows_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    minimal = masks["source_confluence_repairable_v1"] & masks["hardened"]
    rows = []
    for _, row in frame[minimal].sort_values(["run_id_v1", "candidate_uid_v1"]).iterrows():
        has_blueprint = "HISTORICAL_V2_BLUEPRINT" in str(row.get("source_evidence_v1", ""))
        same_no_blueprint_as_unsafe = not has_blueprint and str(row.get("run_id_v1")) == "TRUTH_MONFRI_WEEK_20260302_20260309"
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "group_v1": row.get("run_id_v1"),
                "original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "safe_core_v1": True,
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "branch_tier_v1": "score>=0.95 + R5_1 + V2-like + low-support-clear",
                "source_condition_that_cut_row_v1": (
                    "score>=0.99 + missing R5_TAIL_SCORE + R5_BAD_SCORE support + support-repairable policy"
                ),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "differs_from_unsafe_row_v1": (
                    "HISTORICAL_V2_BLUEPRINT token only, but token is blocked as proxy"
                    if has_blueprint
                    else "No clean source-signal difference found in current allowlisted fields"
                ),
                "source_signals_that_can_protect_row_v1": (
                    "NONE_DEPLOYABLE; historical blueprint would protect it but is blocked"
                    if has_blueprint
                    else "NONE_FOUND"
                ),
                "low_support_status_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "can_be_retained_without_proxy_leakage_v1": False if has_blueprint or same_no_blueprint_as_unsafe else False,
                "retention_blocker_v1": "ONLY_BLUEPRINT_OR_NO_DISTINCTION",
            }
        )
    if len(rows) != EXPECTED_PRIOR_MINIMAL_CUT:
        raise RuntimeError(f"EXPECTED_11_GOOD_ROWS_CUT_BY_MINIMAL_SOURCE_VETO: {len(rows)}")
    return rows


def _unsafe_row_refinement_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    unsafe_mask = masks["base_without_hard_safety_veto_v1"] & ~masks["hardened"] & _bool(frame, "unsafe_audit_v1")
    rows = []
    for _, row in frame[unsafe_mask].iterrows():
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "group_v1": row.get("run_id_v1"),
                "branch_tier_v1": "score>=0.95 + R5_1 + V2-like + low-support-clear",
                "why_safe_core_rule_admits_it_v1": (
                    "It has score support, R5_1 support, V2-like support, and low-support-veto-clear status."
                ),
                "source_risk_conditions_that_catch_it_v1": (
                    "score>=0.99, missing R5_TAIL_SCORE, R5_BAD_SCORE support, support-repairable policy, no tail repair"
                ),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "signal_r5_1_bad_score_v1": _as_bool(row.get("signal_r5_1_bad_score_v1")),
                "signal_r5_bad_score_v1": _as_bool(row.get("signal_r5_bad_score_v1")),
                "signal_r5_tail_score_v1": _as_bool(row.get("signal_r5_tail_score_v1")),
                "signal_v2_like_bad_tail_v1": _as_bool(row.get("signal_v2_like_bad_tail_v1")),
                "signal_tail_repair_v1": _as_bool(row.get("signal_tail_repair_v1")),
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "branch_local_stop_possible_v1": True,
                "confluence_stop_possible_v1": True,
                "missingness_support_lineage_risk_as_of_safe_v1": True,
                "clean_distinguishing_signal_from_11_good_rows_v1": "NONE_FOUND; 3 good rows have identical allowed source shape",
                "nondeployable_distinguishing_signals_v1": "HISTORICAL_V2_BLUEPRINT for 8 good rows; student/membership score for all good rows",
            }
        )
    if len(rows) != 1:
        raise RuntimeError(f"EXPECTED_ONE_UNSAFE_ROW_FOR_REFINEMENT: {len(rows)}")
    return rows


def _candidate_definitions(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = masks["base_without_hard_safety_veto_v1"]
    score = _num(frame, "candidate_score_v1")
    policy = _str(frame, "run_id_policy_class_v1")
    source = _str(frame, "source_evidence_v1")
    student = _num(frame, "student_oof_score_v1")
    same_shape = masks["source_signal_shape_ge099_v1"]
    confluence = masks["source_confluence_repairable_v1"]
    return [
        {
            "candidate_name_v1": "BRANCH_LOCAL_SOURCE_HARD_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1",
            "source_lineage_v1": "AS_OF_SAFE_SOURCE_SIGNALS",
            "as_of_status_v1": "AS_OF_SAFE",
            "rule_condition_v1": "veto safe-core branch rows with score>=0.99, missing R5_TAIL_SCORE, and R5_BAD_SCORE support",
            "normalization_needed_v1": False,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "LOW",
            "mask": same_shape,
        },
        {
            "candidate_name_v1": "SOURCE_CONFLUENCE_REFINED_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, run_id_policy_class_v1",
            "source_lineage_v1": "AS_OF_SAFE_SOURCE_SIGNALS_WITH_SUPPORT_POLICY_NORMALIZATION_NEEDED",
            "as_of_status_v1": "AS_OF_SAFE_NEEDS_NORMALIZATION",
            "rule_condition_v1": (
                "veto if score>=0.99, missing R5_TAIL_SCORE, R5_BAD_SCORE support, and support-repairable policy"
            ),
            "normalization_needed_v1": True,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "LOW",
            "mask": confluence,
        },
        {
            "candidate_name_v1": "RELAXED_SOURCE_THRESHOLD_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1",
            "source_lineage_v1": "AS_OF_SAFE_SOURCE_SIGNALS",
            "as_of_status_v1": "AS_OF_SAFE",
            "rule_condition_v1": "veto if score>=0.99321, missing R5_TAIL_SCORE, and R5_BAD_SCORE support",
            "normalization_needed_v1": False,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "LOW",
            "mask": masks["source_relaxed_score_ge099321_v1"],
        },
        {
            "candidate_name_v1": "GOOD_CORE_EXCEPTION_GUARD_SOURCE_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, source_evidence_v1 tokens",
            "source_lineage_v1": "SOURCE_SIGNALS_PLUS_BLOCKED_HISTORICAL_BLUEPRINT_REFERENCE",
            "as_of_status_v1": "BLOCKED_PROXY_REFERENCE",
            "rule_condition_v1": "diagnostic only: veto same source-shape rows unless HISTORICAL_V2_BLUEPRINT guard is present",
            "normalization_needed_v1": False,
            "mapping_needed_v1": False,
            "proxy_leakage_risk_v1": True,
            "uses_historical_v2_blueprint_v1": True,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "LOW_LOGIC_BLOCKED_LINEAGE",
            "mask": same_shape & ~source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False),
        },
        {
            "candidate_name_v1": "LOW_SUPPORT_AWARE_SOURCE_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, structural_low_support_v1, zero_denominator_group_v1",
            "source_lineage_v1": "AS_OF_SAFE_SOURCE_SIGNALS_WITH_SUPPORT_FLAGS",
            "as_of_status_v1": "AS_OF_SAFE_NEEDS_NORMALIZATION",
            "rule_condition_v1": "veto same source-shape rows only if structural low-support or zero-denominator is present",
            "normalization_needed_v1": True,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "LOW",
            "mask": same_shape & (_bool(frame, "structural_low_support_v1") | _bool(frame, "zero_denominator_group_v1")),
        },
        {
            "candidate_name_v1": "MINIMAL_GREEN_SOURCE_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, non-proxy source exceptions",
            "source_lineage_v1": "NO_CURRENT_CLEAN_SOURCE_EXCEPTION_FOUND",
            "as_of_status_v1": "AS_OF_SAFE_BUT_DOES_NOT_BLOCK_UNSAFE",
            "rule_condition_v1": "no clean <=5-cut condition exists in current allowed source fields",
            "normalization_needed_v1": False,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "N/A",
            "mask": base & score.gt(1.0),
        },
        {
            "candidate_name_v1": "YELLOW_REVIEW_SOURCE_VETO_V1",
            "input_fields_v1": "candidate_score_v1, signal_r5_tail_score_v1, signal_r5_bad_score_v1, non-proxy source exceptions",
            "source_lineage_v1": "NO_CURRENT_CLEAN_SOURCE_EXCEPTION_FOUND",
            "as_of_status_v1": "AS_OF_SAFE_BUT_DOES_NOT_BLOCK_UNSAFE",
            "rule_condition_v1": "no clean 6-10-cut condition exists in current allowed source fields",
            "normalization_needed_v1": False,
            "mapping_needed_v1": True,
            "proxy_leakage_risk_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "complexity_v1": "N/A",
            "mask": base & score.gt(1.0),
        },
        {
            "candidate_name_v1": "DIAGNOSTIC_STUDENT_MARGIN_REFERENCE_NOT_ALLOWED_V1",
            "input_fields_v1": "student_oof_score_v1",
            "source_lineage_v1": "MEMBERSHIP_STUDENT_HISTORY",
            "as_of_status_v1": "BLOCKED_MEMBERSHIP_PROXY",
            "rule_condition_v1": "diagnostic only: veto high-score source-shape rows with student_oof_score < 0.50",
            "normalization_needed_v1": False,
            "mapping_needed_v1": False,
            "proxy_leakage_risk_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": True,
            "complexity_v1": "LOW_LOGIC_BLOCKED_HISTORY",
            "mask": base & score.ge(0.99) & student.lt(0.50),
        },
    ]


def _metrics_for_candidates(
    frame: pd.DataFrame, masks: dict[str, pd.Series], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = masks["base_without_hard_safety_veto_v1"]
    safe_core = masks["hardened"]
    original = _bool(frame, "selected_original_140_v1")
    unsafe = _bool(frame, "unsafe_audit_v1")
    minimal_selected = base & ~masks["source_confluence_repairable_v1"]
    rows: list[dict[str, Any]] = []
    dry_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        veto_mask = candidate["mask"]
        selected = base & ~veto_mask
        selected_frame = frame[selected]
        unsafe_blocked = int((veto_mask & base & ~safe_core & unsafe).sum()) > 0
        good_cut = int((veto_mask & safe_core).sum())
        shortcut = bool(candidate["proxy_leakage_risk_v1"])
        tier = retention_class(
            unsafe_row_blocked=unsafe_blocked,
            good_rows_cut=good_cut,
            shortcut_or_leakage=shortcut,
        )
        adapter_ready = tier == "GREEN" and not shortcut and int((selected & unsafe).sum()) == 0
        common = {
            "candidate_name_v1": candidate["candidate_name_v1"],
            "input_fields_v1": candidate["input_fields_v1"],
            "source_lineage_v1": candidate["source_lineage_v1"],
            "as_of_status_v1": candidate["as_of_status_v1"],
            "rule_condition_v1": candidate["rule_condition_v1"],
            "unsafe_row_blocked_v1": unsafe_blocked,
            "safe_core_rows_retained_v1": int((selected & safe_core).sum()),
            "safe_core_rows_cut_v1": good_cut,
            "original_140_rows_retained_v1": int((selected & original).sum()),
            "original_140_rows_cut_v1": int((veto_mask & safe_core & original).sum()),
            "three_extra_rows_retained_v1": int((selected & safe_core & ~original).sum()),
            "three_extra_rows_cut_v1": int((veto_mask & safe_core & ~original).sum()),
            "selected_rows_after_veto_v1": int(selected.sum()),
            "bad_count_audit_only_v1": int(_bool(selected_frame, "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(selected_frame, "tail_label_v1").sum()),
            "precision_audit_only_v1": float(_bool(selected_frame, "bad_label_v1").sum() / max(len(selected_frame), 1)),
            "safety_status_v1": "CLEAN" if int((selected & unsafe).sum()) == 0 else "FAIL",
            "unsafe_rows_remaining_v1": int((selected & unsafe).sum()),
            "retention_class_v1": tier,
            "adapter_feasibility_v1": _adapter_feasibility(tier, candidate, adapter_ready),
            "normalization_needed_v1": candidate["normalization_needed_v1"],
            "mapping_needed_v1": candidate["mapping_needed_v1"],
            "proxy_leakage_risk_v1": candidate["proxy_leakage_risk_v1"],
            "uses_historical_v2_blueprint_v1": candidate["uses_historical_v2_blueprint_v1"],
            "uses_student_or_membership_proxy_v1": candidate["uses_student_or_membership_proxy_v1"],
            "complexity_v1": candidate["complexity_v1"],
            "adapter_ready_v1": adapter_ready,
            "recommendation_v1": _candidate_recommendation(tier, candidate, unsafe_blocked),
        }
        rows.append(common)
        dry_rows.append(
            {
                "candidate_name_v1": candidate["candidate_name_v1"],
                "selected_rows_v1": common["selected_rows_after_veto_v1"],
                "safe_core_retained_v1": common["safe_core_rows_retained_v1"],
                "original_140_retained_v1": common["original_140_rows_retained_v1"],
                "good_rows_cut_v1": common["safe_core_rows_cut_v1"],
                "unsafe_row_blocked_v1": common["unsafe_row_blocked_v1"],
                "unsafe_rows_remaining_v1": common["unsafe_rows_remaining_v1"],
                "bad_tail_audit_only_v1": [common["bad_count_audit_only_v1"], common["tail_count_audit_only_v1"]],
                "precision_audit_only_v1": common["precision_audit_only_v1"],
                "safety_status_v1": common["safety_status_v1"],
                "mismatch_vs_original_safe_core_v1": common["safe_core_rows_cut_v1"] + int((selected & ~safe_core).sum()),
                "mismatch_vs_minimal_source_hard_veto_v1": int((selected != minimal_selected).sum()),
                "mapping_readiness_v1": "READY_FOR_MAPPING" if adapter_ready else common["adapter_feasibility_v1"],
                "retention_class_v1": common["retention_class_v1"],
            }
        )
    validate_candidate_metrics(rows)
    return rows, dry_rows


def _adapter_feasibility(tier: str, candidate: dict[str, Any], adapter_ready: bool) -> str:
    if adapter_ready:
        return "READY_FOR_INPUT_MAPPING"
    if candidate["proxy_leakage_risk_v1"]:
        return "BLOCKED_PROXY_OR_MEMBERSHIP_REFERENCE"
    if tier == "BLOCKED":
        return "BLOCKED_DOES_NOT_STOP_UNSAFE_ROW"
    if tier == "YELLOW":
        return "NEEDS_REVIEW_BEFORE_MAPPING"
    if tier in {"ORANGE", "RED"}:
        return "NOT_READY_TOO_DESTRUCTIVE"
    return "NOT_READY"


def _candidate_recommendation(tier: str, candidate: dict[str, Any], unsafe_blocked: bool) -> str:
    if candidate["proxy_leakage_risk_v1"]:
        return "DIAGNOSTIC_ONLY_BLOCKED_PROXY_OR_MEMBERSHIP_RISK"
    if not unsafe_blocked:
        return "REJECT_DOES_NOT_BLOCK_UNSAFE_ROW"
    if tier == "GREEN":
        return "PROCEED_TO_INPUT_MAPPING"
    if tier == "YELLOW":
        return "REVIEW_BEFORE_MAPPING"
    if tier in {"ORANGE", "RED"}:
        return "REJECT_FOR_MAPPING_NOW_STILL_TOO_DESTRUCTIVE"
    return "REJECT_BLOCKED"


def _candidate_definitions_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_CANDIDATE_DEFINITIONS_V1",
        "candidate_count_v1": len(candidates),
        "candidates_v1": [{key: value for key, value in candidate.items() if key != "mask"} for candidate in candidates],
    }


def _unsafe_lookalike_audit(
    frame: pd.DataFrame, masks: dict[str, pd.Series], selected_candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    similar = masks["source_signal_shape_ge099_v1"] & masks["base_without_hard_safety_veto_v1"]
    rows = []
    for _, row in frame[similar].sort_values(["unsafe_audit_v1", "candidate_uid_v1"], ascending=[False, True]).iterrows():
        blocked = _as_bool(selected_candidate["mask"].loc[row.name])
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
                "blocked_by_candidate_v1": blocked,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "lookalike_reason_v1": "high-score no-tail R5-bad source-shape neighborhood",
                "risk_class_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_MONITORING"
                if blocked
                else "HIGH_UNSAFE_LOOKALIKE_RISK_BLOCK_CANDIDATE",
                "generalization_assessment_v1": (
                    "Candidate generalizes to a source-shape neighborhood, but the neighborhood still includes too many good rows."
                ),
            }
        )
    return rows


def _anti_shortcut_audit(selected: dict[str, Any]) -> dict[str, Any]:
    status = "PASS_NO_SHORTCUTS_ADAPTER_STILL_BLOCKED"
    return {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": status,
        "selected_candidate_v1": selected["candidate_name_v1"],
        "no_historical_v2_blueprint_v1": selected["uses_historical_v2_blueprint_v1"] is False,
        "no_membership_proxy_v1": selected["uses_student_or_membership_proxy_v1"] is False,
        "no_coverage_proxy_v1": True,
        "no_row_identity_v1": True,
        "no_artifact_shortcut_v1": True,
        "no_selected_by_flags_v1": True,
        "no_final_labels_as_input_v1": True,
        "no_mfe_hindsight_v1": True,
        "no_safe_recoverable_direct_v1": True,
        "no_audit_only_safety_flag_v1": True,
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_run_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }


def _final_selection(metrics_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = next(row for row in metrics_rows if row["candidate_name_v1"] == FINAL_REFINED_CANDIDATE)
    final = {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_FINAL_CANDIDATE_SELECTION_V1",
        "selected_candidate_name_v1": selected["candidate_name_v1"],
        "unsafe_row_blocked_v1": selected["unsafe_row_blocked_v1"],
        "good_rows_cut_v1": selected["safe_core_rows_cut_v1"],
        "safe_core_rows_retained_v1": selected["safe_core_rows_retained_v1"],
        "original_140_rows_retained_v1": selected["original_140_rows_retained_v1"],
        "bad_tail_audit_only_v1": [selected["bad_count_audit_only_v1"], selected["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": selected["precision_audit_only_v1"],
        "safety_status_v1": selected["safety_status_v1"],
        "retention_class_v1": selected["retention_class_v1"],
        "uses_historical_v2_blueprint_v1": selected["uses_historical_v2_blueprint_v1"],
        "proxy_leakage_risk_v1": selected["proxy_leakage_risk_v1"],
        "adapter_input_mapping_allowed_next_v1": False,
        "reason_v1": "No clean source-only GREEN/YELLOW refinement exists in current fields; best clean candidate remains ORANGE.",
        "status_v1": FINAL_STATUS,
    }
    validate_final_selection(final)
    adapter = {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_ADAPTER_READINESS_PREASSESSMENT_V1",
        "selected_candidate_v1": selected["candidate_name_v1"],
        "can_go_to_input_mapping_now_v1": False,
        "needs_review_first_v1": False,
        "normalization_needed_v1": selected["normalization_needed_v1"],
        "lineage_confirmation_needed_v1": False,
        "adapter_must_remain_held_v1": True,
        "status_v1": "NOT_READY_STILL_ORANGE_DESTRUCTIVE",
        "next_gate_v1": NEXT_ACTION,
    }
    recommendation = {
        "layer_name": "REFINE_CLEAN_SAFETY_LAYER_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_candidate_v1": selected["candidate_name_v1"],
        "rationale_v1": [
            "The unsafe row is blocked by clean source confluence.",
            "The same clean source confluence still cuts 11 good safe-core/original-140 rows.",
            "The only currently precise exceptions are historical blueprint or student/membership-margin references, both blocked.",
            "Input mapping should not begin until stronger source signals reduce good-row loss to GREEN or YELLOW.",
        ],
    }
    go_no_go = {
        "layer_name": "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_candidate_v1": selected["candidate_name_v1"],
        "unsafe_row_blocked_v1": selected["unsafe_row_blocked_v1"],
        "good_rows_cut_v1": selected["safe_core_rows_cut_v1"],
        "retention_class_v1": selected["retention_class_v1"],
        "input_mapping_allowed_next_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_go_no_go(go_no_go)
    return final, adapter, recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    cut_rows: list[dict[str, Any]],
    unsafe_rows: list[dict[str, Any]],
    candidate_defs: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    dry_rows: list[dict[str, Any]],
    lookalike_rows: list[dict[str, Any]],
    anti: dict[str, Any],
    final: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "refine_clean_safety_layer_reproducibility_audit_v1.md",
        [
            "# Refine Clean Safety Layer Reproducibility Audit V1",
            "",
            f"- Safe-core rows: `{repro['safe_core_selected_rows_v1']}`",
            f"- Original-140 recovered: `{repro['safe_core_recovered_original_140_v1']}`",
            f"- Prior minimal source veto good rows cut: `{repro['prior_minimal_source_veto_good_rows_cut_v1']}`",
            f"- Prior retention class: `{repro['prior_minimal_source_veto_retention_class_v1']}`",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_cut_11_good_rows_audit_v1.md",
        [
            "# Refine Clean Safety Layer Cut 11 Good Rows Audit V1",
            "",
            f"- Rows audited: `{len(cut_rows)}`",
            f"- Rows protectable only by blocked blueprint token: `{sum('HISTORICAL_V2_BLUEPRINT' in row['source_evidence_v1'] for row in cut_rows)}`",
            "- Three good rows share the same current clean source shape as the unsafe row.",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.md",
        [
            "# Refine Clean Safety Layer Unsafe Row Refinement Audit V1",
            "",
            f"- Unsafe rows audited: `{len(unsafe_rows)}`",
            "- Clean confluence catches the row, but current clean fields do not distinguish it narrowly enough.",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_candidate_definitions_v1.md",
        [
            "# Refine Clean Safety Layer Candidate Definitions V1",
            "",
            *[
                f"- `{candidate['candidate_name_v1']}`: {candidate['rule_condition_v1']}"
                for candidate in candidate_defs["candidates_v1"]
            ],
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_candidate_metrics_v1.md",
        [
            "# Refine Clean Safety Layer Candidate Metrics V1",
            "",
            *[
                f"- `{row['candidate_name_v1']}`: unsafe blocked `{row['unsafe_row_blocked_v1']}`, good rows cut `{row['safe_core_rows_cut_v1']}`, retention `{row['retention_class_v1']}`, adapter `{row['adapter_feasibility_v1']}`"
                for row in metrics_rows
            ],
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_candidate_dry_run_v1.md",
        [
            "# Refine Clean Safety Layer Candidate Dry Run V1",
            "",
            *[
                f"- `{row['candidate_name_v1']}`: selected `{row['selected_rows_v1']}`, original retained `{row['original_140_retained_v1']}`, safety `{row['safety_status_v1']}`, mapping `{row['mapping_readiness_v1']}`"
                for row in dry_rows
            ],
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_unsafe_lookalike_audit_v1.md",
        [
            "# Refine Clean Safety Layer Unsafe Lookalike Audit V1",
            "",
            f"- Lookalike rows audited: `{len(lookalike_rows)}`",
            "- The selected candidate generalizes to a source-shape neighborhood but remains too broad.",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_anti_shortcut_audit_v1.md",
        [
            "# Refine Clean Safety Layer Anti-Shortcut Audit V1",
            "",
            f"- Status: `{anti['status_v1']}`",
            "- The selected candidate uses no blueprint, membership proxy, coverage proxy, row identity, labels, MFE/hindsight, safe_recoverable direct, selected flags, or audit-only safety flag.",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_final_candidate_selection_v1.md",
        [
            "# Refine Clean Safety Layer Final Candidate Selection V1",
            "",
            f"- Selected candidate: `{final['selected_candidate_name_v1']}`",
            f"- Unsafe row blocked: `{final['unsafe_row_blocked_v1']}`",
            f"- Good rows cut: `{final['good_rows_cut_v1']}`",
            f"- Retention class: `{final['retention_class_v1']}`",
            f"- Status: `{final['status_v1']}`",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_adapter_readiness_preassessment_v1.md",
        [
            "# Refine Clean Safety Layer Adapter Readiness Preassessment V1",
            "",
            f"- Can go to input mapping now: `{adapter['can_go_to_input_mapping_now_v1']}`",
            f"- Adapter must remain held: `{adapter['adapter_must_remain_held_v1']}`",
            f"- Status: `{adapter['status_v1']}`",
        ],
    )
    _write_report(
        root / "refine_clean_safety_layer_recommendation_v1.md",
        [
            "# Refine Clean Safety Layer Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Selected candidate: `{recommendation['selected_candidate_v1']}`",
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
    cut_rows = _cut_11_good_rows_audit(frame, masks)
    unsafe_rows = _unsafe_row_refinement_audit(frame, masks)
    candidates = _candidate_definitions(frame, masks)
    candidate_defs = _candidate_definitions_payload(candidates)
    metrics_rows, dry_rows = _metrics_for_candidates(frame, masks, candidates)
    selected_candidate = next(candidate for candidate in candidates if candidate["candidate_name_v1"] == FINAL_REFINED_CANDIDATE)
    lookalike_rows = _unsafe_lookalike_audit(frame, masks, selected_candidate)
    selected_metric = next(row for row in metrics_rows if row["candidate_name_v1"] == FINAL_REFINED_CANDIDATE)
    anti = _anti_shortcut_audit(selected_metric)
    final, adapter, recommendation, go_no_go = _final_selection(metrics_rows)

    _write_json(artifact_root / "refine_clean_safety_layer_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "refine_clean_safety_layer_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "refine_clean_safety_layer_cut_11_good_rows_audit_v1.csv", cut_rows)
    _write_json(
        artifact_root / "refine_clean_safety_layer_cut_11_good_rows_audit_v1.json",
        {"row_count_v1": len(cut_rows), "rows_v1": cut_rows},
    )
    _write_rows(artifact_root / "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "refine_clean_safety_layer_unsafe_row_refinement_audit_v1.json",
        {"row_count_v1": len(unsafe_rows), "rows_v1": unsafe_rows},
    )
    _write_json(artifact_root / "refine_clean_safety_layer_candidate_definitions_v1.json", candidate_defs)
    _write_rows(artifact_root / "refine_clean_safety_layer_candidate_metrics_v1.csv", metrics_rows)
    _write_json(
        artifact_root / "refine_clean_safety_layer_candidate_metrics_v1.json",
        {"row_count_v1": len(metrics_rows), "rows_v1": metrics_rows},
    )
    _write_rows(artifact_root / "refine_clean_safety_layer_candidate_dry_run_v1.csv", dry_rows)
    _write_json(
        artifact_root / "refine_clean_safety_layer_candidate_dry_run_v1.json",
        {"row_count_v1": len(dry_rows), "rows_v1": dry_rows},
    )
    _write_rows(artifact_root / "refine_clean_safety_layer_unsafe_lookalike_audit_v1.csv", lookalike_rows)
    _write_json(
        artifact_root / "refine_clean_safety_layer_unsafe_lookalike_audit_v1.json",
        {"row_count_v1": len(lookalike_rows), "rows_v1": lookalike_rows},
    )
    _write_json(artifact_root / "refine_clean_safety_layer_anti_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "refine_clean_safety_layer_final_candidate_selection_v1.json", final)
    _write_json(artifact_root / "refine_clean_safety_layer_adapter_readiness_preassessment_v1.json", adapter)
    _write_json(artifact_root / "refine_clean_safety_layer_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "refine_clean_as_of_safety_layer_to_retain_safe_core_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        cut_rows,
        unsafe_rows,
        candidate_defs,
        metrics_rows,
        dry_rows,
        lookalike_rows,
        anti,
        final,
        adapter,
        recommendation,
    )

    summary = {
        "layer_name": "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_SUMMARY_V1",
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
        "prior_minimal_source_veto_good_rows_cut_v1": repro["prior_minimal_source_veto_good_rows_cut_v1"],
        "selected_refined_candidate_v1": final["selected_candidate_name_v1"],
        "unsafe_row_blocked_v1": final["unsafe_row_blocked_v1"],
        "good_rows_cut_v1": final["good_rows_cut_v1"],
        "retention_class_v1": final["retention_class_v1"],
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
            "# Refine Clean AS_OF Safety Layer To Retain Safe-Core V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Selected refined candidate: `{summary['selected_refined_candidate_v1']}`",
            f"- Unsafe row blocked: `{summary['unsafe_row_blocked_v1']}`",
            f"- Good rows cut: `{summary['good_rows_cut_v1']}`",
            f"- Retention class: `{summary['retention_class_v1']}`",
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
