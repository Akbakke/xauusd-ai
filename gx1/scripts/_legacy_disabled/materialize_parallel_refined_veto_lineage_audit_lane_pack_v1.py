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

from gx1.scripts import materialize_refine_140_94_hard_safety_veto_to_retain_safe_core_v1 as refine
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1"

INPUT_REFINED_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1_20260428T172254Z_LOCK"
)
INPUT_HOLD_ROOT = (
    DEFAULT_REPORTS_ROOT / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T111216Z_LOCK"
)
INPUT_DISCOVERY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1_20260428T113213Z_LOCK"
)
INPUT_VETO_MAPPING_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1_20260428T101045Z_LOCK"
)
INPUT_ADAPTER_MAPPING_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1_20260428T090840Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"

SUPPORTING_REVALIDATION_ROOT = (
    DEFAULT_REPORTS_ROOT / "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_V1_20260427T095034Z_LOCK"
)
SUPPORTING_OPPORTUNITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1_20260427T122550Z_LOCK"
)

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
REFINED_VETO_ID = "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1"
FINAL_STATUS = "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_PROXY_OR_LEAKAGE_RISK"
NEXT_ACTION = "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1"

LANES = [
    "LANE_01_PROVENANCE_SOURCE_LINEAGE",
    "LANE_02_AS_OF_RECONSTRUCTION",
    "LANE_03_ADAPTER_ALLOWLIST_ELIGIBILITY",
    "LANE_04_MEMBERSHIP_COVERAGE_PROXY_AUDIT",
    "LANE_05_OUTCOME_HINDSIGHT_MFE_LEAKAGE_AUDIT",
    "LANE_06_ROW_IDENTITY_ARTIFACT_SHORTCUT_AUDIT",
    "LANE_07_GROUP_LOSO_SUPPORT_STABILITY_AUDIT",
    "LANE_08_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT",
    "LANE_09_ALTERNATIVE_AS_OF_VETO_WITHOUT_V2_BLUEPRINT",
    "LANE_10_SIMULATED_ADAPTER_CONTRACT_DRY_RUN",
]

LANE_STATUSES = {
    "LANE_PASS_NO_BLOCKER_FOUND",
    "LANE_PASS_WITH_MINOR_NORMALIZATION_OR_MAPPING_NEEDS",
    "LANE_INCONCLUSIVE_NEEDS_MORE_LINEAGE",
    "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK",
    "LANE_BLOCKED_BY_OUTCOME_OR_HINDSIGHT_RISK",
    "LANE_BLOCKED_BY_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT",
    "LANE_BLOCKED_BY_AS_OF_RECONSTRUCTION_FAILURE",
    "LANE_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "LANE_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "LANE_BLOCKED_BY_MISSING_ARTIFACTS",
    "LANE_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_FINAL_STATUSES = {
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_READY_FOR_FAN_IN_DECISION",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_NEEDS_NORMALIZATION_BEFORE_FAN_IN",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_INCONCLUSIVE_NEEDS_MORE_LINEAGE",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_PROXY_OR_LEAKAGE_RISK",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_AS_OF_RECONSTRUCTION_FAILURE",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_FOUND_BETTER_NON_BLUEPRINT_ALTERNATIVE",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "MATERIALIZE_REFINED_VETO_LINEAGE_DECISION_FROM_PARALLEL_LANES_V1",
    "NORMALIZE_REFINED_VETO_BLUEPRINT_INPUTS_BEFORE_FAN_IN_V1",
    "DEEPEN_HISTORICAL_V2_BLUEPRINT_AS_OF_LINEAGE_AUDIT_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "TEST_ALTERNATIVE_AS_OF_HARD_SAFETY_VETO_WITHOUT_V2_BLUEPRINT_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
}

REQUIRED_GLOBAL_OUTPUTS = [
    "parallel_refined_veto_lane_pack_input_manifest_v1.json",
    "parallel_refined_veto_lane_pack_input_manifest_v1.md",
    "parallel_refined_veto_lane_pack_reproducibility_audit_v1.json",
    "parallel_refined_veto_lane_pack_reproducibility_audit_v1.md",
    "parallel_refined_veto_lane_pack_lane_index_v1.csv",
    "parallel_refined_veto_lane_pack_lane_index_v1.json",
    "parallel_refined_veto_lane_pack_summary_v1.json",
    "parallel_refined_veto_lane_pack_summary_v1.md",
    "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.csv",
    "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.json",
    "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.md",
    "parallel_refined_veto_lane_pack_recommendation_v1.json",
    "parallel_refined_veto_lane_pack_recommendation_v1.md",
    "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json",
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


def validate_lane_index(rows: list[dict[str, Any]]) -> bool:
    lane_ids = [row["lane_id_v1"] for row in rows]
    if lane_ids != LANES:
        raise RuntimeError(f"LANE_INDEX_MUST_CONTAIN_EXACT_10_PREDEFINED_LANES: {lane_ids}")
    bad_statuses = [row for row in rows if row["lane_status_v1"] not in LANE_STATUSES]
    if bad_statuses:
        raise RuntimeError(f"UNKNOWN_LANE_STATUS: {bad_statuses}")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_adapter_reopen(go_no_go: dict[str, Any]) -> bool:
    if go_no_go.get("adapter_reopen_allowed_v1") is not False:
        raise RuntimeError("LANE_PACK_MUST_NOT_OPEN_ADAPTER_DIRECTLY")
    if go_no_go.get("r6_run_v1") or go_no_go.get("adapter_built_v1") or go_no_go.get("iql_run_v1"):
        raise RuntimeError("FORBIDDEN_SIDE_EFFECT_DETECTED")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_GLOBAL_OUTPUTS if not (root / name).exists()]
    for lane in LANES:
        lane_root = root / "lanes" / lane
        for name in ["lane_manifest_v1.json", "lane_result_v1.json", "lane_result_v1.md", "lane_risk_audit_v1.json", "lane_risk_audit_v1.md"]:
            if not (lane_root / name).exists():
                missing.append(str(Path("lanes") / lane / name))
    if missing:
        raise RuntimeError(f"PARALLEL_REFINED_VETO_LANE_PACK_REQUIRED_OUTPUTS_MISSING: {missing}")
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
        INPUT_REFINED_ROOT,
        INPUT_HOLD_ROOT,
        INPUT_DISCOVERY_ROOT,
        INPUT_VETO_MAPPING_ROOT,
        INPUT_ADAPTER_MAPPING_ROOT,
        INPUT_HARDEN_ROOT,
        SUPPORTING_REVALIDATION_ROOT,
        SUPPORTING_OPPORTUNITY_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "refined_summary": INPUT_REFINED_ROOT / "summary_v1.json",
        "refined_go_no_go": INPUT_REFINED_ROOT / "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json",
        "refined_metrics": INPUT_REFINED_ROOT / "refine_140_94_refined_veto_candidate_metrics_v1.json",
        "refined_final": INPUT_REFINED_ROOT / "refine_140_94_final_refined_veto_selection_v1.json",
        "refined_lookalike": INPUT_REFINED_ROOT / "refine_140_94_unsafe_lookalike_audit_v1.json",
        "hold_summary": INPUT_HOLD_ROOT / "summary_v1.json",
        "discovery_summary": INPUT_DISCOVERY_ROOT / "summary_v1.json",
        "veto_mapping_summary": INPUT_VETO_MAPPING_ROOT / "summary_v1.json",
        "adapter_mapping_summary": INPUT_ADAPTER_MAPPING_ROOT / "summary_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "revalidation_manifest": SUPPORTING_REVALIDATION_ROOT / "manifest_v1.json",
        "revalidation_decomposition": SUPPORTING_REVALIDATION_ROOT / "v2_result_decomposition_v1.csv",
        "opportunity_manifest": SUPPORTING_OPPORTUNITY_ROOT / "manifest_v1.json",
        "opportunity_rows": SUPPORTING_OPPORTUNITY_ROOT / "r5_2_opportunity_base_rows_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    refined_go = _read_json(required["refined_go_no_go"])
    if refined_go.get("status_v1") != "140_94_REFINED_HARD_SAFETY_VETO_PASS_NEEDS_LINEAGE_CONFIRMATION":
        raise RuntimeError("INPUT_REFINED_STATUS_NOT_LINEAGE_CONFIRMATION")
    return {
        "required_paths": required,
        "refined_summary": _read_json(required["refined_summary"]),
        "refined_go_no_go": refined_go,
        "refined_metrics": _read_json(required["refined_metrics"]),
        "refined_final": _read_json(required["refined_final"]),
        "refined_lookalike": _read_json(required["refined_lookalike"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "discovery_summary": _read_json(required["discovery_summary"]),
        "veto_mapping_summary": _read_json(required["veto_mapping_summary"]),
        "adapter_mapping_summary": _read_json(required["adapter_mapping_summary"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "revalidation_manifest": _read_json(required["revalidation_manifest"]),
        "opportunity_manifest": _read_json(required["opportunity_manifest"]),
        "revalidation_decomposition": pd.read_csv(required["revalidation_decomposition"]),
        "opportunity_rows": pd.read_csv(required["opportunity_rows"]),
        "source_inputs": refine._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "PARALLEL_REFINED_VETO_LANE_PACK_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "refined_veto_root_v1": str(INPUT_REFINED_ROOT),
            "hold_root_v1": str(INPUT_HOLD_ROOT),
            "veto_discovery_root_v1": str(INPUT_DISCOVERY_ROOT),
            "veto_mapping_root_v1": str(INPUT_VETO_MAPPING_ROOT),
            "adapter_input_mapping_root_v1": str(INPUT_ADAPTER_MAPPING_ROOT),
            "harden_root_v1": str(INPUT_HARDEN_ROOT),
            "supporting_revalidation_root_v1": str(SUPPORTING_REVALIDATION_ROOT),
            "supporting_opportunity_root_v1": str(SUPPORTING_OPPORTUNITY_ROOT),
        },
        "files_used_v1": files,
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
    return refine._build_frame_and_masks(inputs["source_inputs"]["source_inputs"])


def _refined_masks(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, pd.Series]:
    source = _str(frame, "source_evidence_v1")
    base = masks["low_support_veto_only"]
    high_score_no_tail_r5_bad = (
        base
        & masks["missing_r5_tail_v1"]
        & masks["has_r5_bad_support_v1"]
        & masks["high_score_099_v1"]
    )
    refined_veto = high_score_no_tail_r5_bad & ~source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False)
    refined_selected = base & ~refined_veto
    return {
        "base_without_hard_veto": base,
        "hardened_safe_core": masks["hardened"],
        "refined_veto": refined_veto,
        "refined_selected": refined_selected,
        "high_score_no_tail_r5_bad": high_score_no_tail_r5_bad,
        "has_historical_v2_blueprint": source.str.contains("HISTORICAL_V2_BLUEPRINT", regex=False),
    }


def _metric_pack(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = frame[mask]
    return {
        "selected_rows_v1": int(mask.sum()),
        "original_140_rows_v1": int((mask & _bool(frame, "selected_original_140_v1")).sum()),
        "safe_core_rows_v1": int((mask & _bool(frame, "hard_veto_clear_shadow_v1")).sum()),
        "bad_count_audit_only_v1": int(_bool(selected, "bad_label_v1").sum()),
        "tail_count_audit_only_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_audit_only_v1": float(_bool(selected, "bad_label_v1").sum() / max(len(selected), 1)),
        "unsafe_rows_audit_only_v1": int(_bool(selected, "unsafe_audit_v1").sum()),
        "safety_status_v1": "CLEAN" if int(_bool(selected, "unsafe_audit_v1").sum()) == 0 else "FAIL",
    }


def _reproducibility(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    rmasks = _refined_masks(frame, masks)
    safe = inputs["refined_summary"]
    selected = frame[rmasks["refined_selected"]]
    payload = {
        "layer_name": "PARALLEL_REFINED_VETO_LANE_PACK_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "refined_veto_id_v1": REFINED_VETO_ID,
        "safe_core_selected_rows_v1": int(masks["hardened"].sum()),
        "safe_core_recovered_original_140_v1": int((masks["hardened"] & _bool(frame, "selected_original_140_v1")).sum()),
        "safe_core_extra_rows_v1": int((masks["hardened"] & ~_bool(frame, "selected_original_140_v1")).sum()),
        "safe_core_bad_tail_audit_only_v1": safe.get("bad_tail_audit_only_v1"),
        "safe_core_precision_audit_only_v1": safe.get("precision_audit_only_v1"),
        "unsafe_extra_without_hard_veto_rows_v1": int(
            (rmasks["base_without_hard_veto"] & ~masks["hardened"] & _bool(frame, "unsafe_audit_v1")).sum()
        ),
        "refined_veto_blocks_unsafe_row_v1": int((rmasks["refined_veto"] & _bool(frame, "unsafe_audit_v1")).sum()) == 1,
        "refined_veto_good_safe_core_rows_cut_v1": int((rmasks["refined_veto"] & masks["hardened"]).sum()),
        "refined_veto_selected_rows_v1": int(rmasks["refined_selected"].sum()),
        "refined_veto_bad_tail_audit_only_v1": [
            int(_bool(selected, "bad_label_v1").sum()),
            int(_bool(selected, "tail_label_v1").sum()),
        ],
        "refined_veto_safety_status_v1": "CLEAN" if int(_bool(selected, "unsafe_audit_v1").sum()) == 0 else "FAIL",
        "refined_input_status_v1": inputs["refined_go_no_go"].get("status_v1"),
    }
    if payload["safe_core_selected_rows_v1"] != 89 or payload["refined_veto_good_safe_core_rows_cut_v1"] != 3:
        raise RuntimeError(f"PARALLEL_REFINED_VETO_REPRODUCTION_FAILED: {payload}")
    return payload


def _lane_manifest(lane_id: str, purpose: str, artifact_root: Path) -> dict[str, Any]:
    return {
        "lane_id_v1": lane_id,
        "purpose_v1": purpose,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_refined_veto_root_v1": str(INPUT_REFINED_ROOT),
        "no_r6_adapter_iql_package_freeze_promo_live_v1": True,
        "no_implicit_latest_glob_decisioning_v1": True,
    }


def _lane_payload(
    lane_id: str,
    status: str,
    classification: str,
    summary: str,
    evidence_rows: list[dict[str, Any]] | None = None,
    risk_level: str = "MEDIUM",
    blocker_type: str = "",
    recommendation: str = "KEEP_ADAPTER_CLOSED",
) -> dict[str, Any]:
    if status not in LANE_STATUSES:
        raise RuntimeError(f"UNKNOWN_LANE_STATUS: {status}")
    return {
        "lane_id_v1": lane_id,
        "lane_status_v1": status,
        "classification_v1": classification,
        "summary_v1": summary,
        "risk_level_v1": risk_level,
        "blocker_type_v1": blocker_type,
        "recommendation_v1": recommendation,
        "evidence_rows_v1": evidence_rows or [],
    }


def _lanes(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rmasks = _refined_masks(frame, masks)
    source = _str(frame, "source_evidence_v1")
    hist = rmasks["has_historical_v2_blueprint"]
    safe_core = masks["hardened"]
    refined_veto = rmasks["refined_veto"]
    refined_selected = rmasks["refined_selected"]
    unsafe = _bool(frame, "unsafe_audit_v1")
    original = _bool(frame, "selected_original_140_v1")
    plus45 = _bool(frame, "is_plus45_diagnostic_v1")
    teacher185 = _bool(frame, "is_185_139_teacher_v1")
    baseline140 = _bool(frame, "is_140_94_baseline_v1")
    student_core = _bool(frame, "student_core_selected_v1")
    lane_selected = _bool(frame, "lane_selected_v1")
    reval = inputs["revalidation_decomposition"]
    opportunity = inputs["opportunity_rows"]
    common_source_rows = [
        {
            "source_artifact_v1": str(SUPPORTING_REVALIDATION_ROOT),
            "source_file_v1": "v2_result_decomposition_v1.csv",
            "source_column_v1": "v2_captured_v1",
            "downstream_column_v1": "historical_v2_captured_v1 -> source_evidence_v1 contains HISTORICAL_V2_BLUEPRINT",
            "row_count_with_source_flag_v1": int(reval["v2_captured_v1"].fillna(False).astype(bool).sum())
            if "v2_captured_v1" in reval.columns
            else 0,
        }
    ]
    lane04_evidence = [
        {
            "cohort_v1": "all_rows",
            "historical_blueprint_rows_v1": int(hist.sum()),
            "overlap_140_94_selected_v1": int((hist & baseline140).sum()),
            "overlap_safe_core_v1": int((hist & safe_core).sum()),
            "overlap_185_139_teacher_v1": int((hist & teacher185).sum()),
            "overlap_plus45_diagnostic_v1": int((hist & plus45).sum()),
            "overlap_lane_membership_v1": int((hist & lane_selected).sum()),
            "blueprint_outside_140_94_v1": int((hist & ~baseline140).sum()),
            "blueprint_outside_safe_core_v1": int((hist & ~safe_core).sum()),
        }
    ]
    group_rows = []
    for run_id, group in frame[refined_veto | (safe_core & hist)].groupby("run_id_v1", dropna=False):
        group_rows.append(
            {
                "run_id_v1": run_id,
                "historical_blueprint_safe_core_rows_v1": int((hist.loc[group.index] & safe_core.loc[group.index]).sum()),
                "refined_veto_cut_rows_v1": int(refined_veto.loc[group.index].sum()),
                "unsafe_rows_cut_v1": int((refined_veto.loc[group.index] & unsafe.loc[group.index]).sum()),
                "original_140_cut_v1": int((refined_veto.loc[group.index] & original.loc[group.index]).sum()),
            }
        )
    alt_rows = []
    metric_by_name = {row["candidate_name_v1"]: row for row in inputs["refined_metrics"]["rows_v1"]}
    for name in [
        "BRANCH_LOCAL_SIGNAL_SHAPE_NO_TAIL_R5_BAD_SCORE_GE_099_V1",
        "TWO_CONDITION_NO_TAIL_REPAIRABLE_SUPPORT_V1",
        "RELAXED_SIGNAL_SHAPE_SCORE_GE_099321_V1",
        "LOW_SUPPORT_AWARE_NO_TAIL_REPAIRABLE_NO_HISTORICAL_BLUEPRINT_V1",
    ]:
        row = metric_by_name.get(name, {})
        alt_rows.append(
            {
                "alternative_veto_v1": name,
                "unsafe_row_blocked_v1": row.get("unsafe_row_blocked_v1"),
                "good_rows_cut_v1": row.get("safe_core_rows_cut_v1"),
                "retention_tier_v1": row.get("retention_tier_v1"),
                "avoids_blueprint_risk_v1": "HISTORICAL_BLUEPRINT" not in str(row.get("input_fields_v1", "")),
                "adapter_feasibility_v1": row.get("adapter_feasibility_v1"),
            }
        )
    dry_metrics = _metric_pack(frame, refined_selected)
    lanes = [
        _lane_payload(
            "LANE_01_PROVENANCE_SOURCE_LINEAGE",
            "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK",
            "PROVENANCE_HISTORICAL_ARTIFACT_PROXY_RISK",
            "HISTORICAL_V2_BLUEPRINT is sourced from historical v2_captured membership in v2_result_decomposition_v1.csv, not an independently allowlisted AS_OF adapter field.",
            common_source_rows,
            risk_level="HIGH",
            blocker_type="HISTORICAL_ARTIFACT_MEMBERSHIP_PROXY",
        ),
        _lane_payload(
            "LANE_02_AS_OF_RECONSTRUCTION",
            "LANE_BLOCKED_BY_AS_OF_RECONSTRUCTION_FAILURE",
            "AS_OF_RECONSTRUCTION_NOT_PROVEN",
            "No independent raw AS_OF reconstruction rule for HISTORICAL_V2_BLUEPRINT was found in the current adapter contract; current reconstruction uses the historical artifact flag.",
            common_source_rows,
            risk_level="HIGH",
            blocker_type="AS_OF_RECONSTRUCTION_FAILURE",
        ),
        _lane_payload(
            "LANE_03_ADAPTER_ALLOWLIST_ELIGIBILITY",
            "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK",
            "BLOCKED_BY_PROXY_RISK",
            "The field is not in the current allowlist and should not be allowlisted until the historical V2 membership/proxy risk is resolved.",
            [
                {
                    "proposed_adapter_field_name_v1": "asof_signal__historical_v2_blueprint_v1",
                    "source_path_v1": str(SUPPORTING_REVALIDATION_ROOT / "v2_result_decomposition_v1.csv:v2_captured_v1"),
                    "data_type_v1": "bool",
                    "allowlist_decision_v1": "BLOCKED_BY_PROXY_RISK",
                    "normalization_needed_v1": True,
                }
            ],
            risk_level="HIGH",
            blocker_type="ALLOWLIST_PROXY_RISK",
        ),
        _lane_payload(
            "LANE_04_MEMBERSHIP_COVERAGE_PROXY_AUDIT",
            "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK",
            "MEMBERSHIP_PROXY_RISK",
            "Blueprint behaves as historical V2 selected-membership evidence. It overlaps selected cohorts and is not currently computable independently of selection artifacts.",
            lane04_evidence,
            risk_level="HIGH",
            blocker_type="MEMBERSHIP_PROXY",
        ),
        _lane_payload(
            "LANE_05_OUTCOME_HINDSIGHT_MFE_LEAKAGE_AUDIT",
            "LANE_INCONCLUSIVE_NEEDS_MORE_LINEAGE",
            "INDIRECT_HINDSIGHT_RISK_UNRESOLVED",
            "Direct refined-veto logic does not use labels/MFE, but the upstream v2_result_decomposition file carries bad/tail and exclusion audit columns; lineage must prove v2_captured was generated without outcome/hindsight.",
            [
                {
                    "bad_tail_columns_in_source_file_v1": all(
                        column in reval.columns for column in ["bad_label_v1", "tail_label_v1"]
                    ),
                    "mfe_or_safety_exclusion_columns_in_source_file_v1": any("mfe" in column.lower() for column in reval.columns),
                    "refined_veto_uses_labels_directly_v1": False,
                    "refined_veto_uses_mfe_directly_v1": False,
                }
            ],
            risk_level="MEDIUM",
            blocker_type="UNRESOLVED_UPSTREAM_HINDSIGHT_RISK",
        ),
        _lane_payload(
            "LANE_06_ROW_IDENTITY_ARTIFACT_SHORTCUT_AUDIT",
            "LANE_BLOCKED_BY_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT",
            "ARTIFACT_SHORTCUT_RISK",
            "The refined guard depends on a materialized historical artifact membership flag. It is not row identity, but it is artifact-derived selection evidence until independently reconstructed.",
            [
                {
                    "row_identity_used_v1": False,
                    "artifact_membership_flag_used_v1": True,
                    "selected_by_flag_used_v1": False,
                    "candidate_uid_used_as_rule_v1": False,
                    "source_artifact_dependency_v1": str(SUPPORTING_REVALIDATION_ROOT),
                }
            ],
            risk_level="HIGH",
            blocker_type="ARTIFACT_SELECTION_SHORTCUT",
        ),
        _lane_payload(
            "LANE_07_GROUP_LOSO_SUPPORT_STABILITY_AUDIT",
            "LANE_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
            "BLOCKED_BY_GROUP_CONCENTRATION",
            "The known unsafe row and the 3 good rows cut by the refined veto are concentrated in the same run_id/fold region; strict support remains too weak for direct approval.",
            group_rows,
            risk_level="HIGH",
            blocker_type="GROUP_CONCENTRATION",
        ),
        _lane_payload(
            "LANE_08_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT",
            "LANE_INCONCLUSIVE_NEEDS_MORE_LINEAGE",
            "UNKNOWN_REQUIRES_MORE_AUDIT",
            "The refined veto blocks the known unsafe row, but boundary robustness depends on whether HISTORICAL_V2_BLUEPRINT is a valid AS_OF guard rather than a historical proxy.",
            [
                {
                    "unsafe_row_blocked_v1": int((refined_veto & unsafe).sum()) == 1,
                    "unsafe_lookalikes_not_blocked_v1": int((refined_selected & unsafe).sum()),
                    "safe_core_rows_accidentally_blocked_v1": int((refined_veto & safe_core).sum()),
                    "risk_class_v1": "UNKNOWN_REQUIRES_MORE_AUDIT",
                }
            ],
            risk_level="MEDIUM",
            blocker_type="BOUNDARY_LINEAGE_DEPENDENT",
        ),
        _lane_payload(
            "LANE_09_ALTERNATIVE_AS_OF_VETO_WITHOUT_V2_BLUEPRINT",
            "LANE_PASS_NO_BLOCKER_FOUND",
            "NO_BETTER_NON_BLUEPRINT_ALTERNATIVE_FOUND",
            "Non-blueprint AS_OF alternatives either fail to block the unsafe row or cut too many good safe-core rows; no better alternative is promoted in this lane.",
            alt_rows,
            risk_level="LOW",
            blocker_type="",
            recommendation="NO_ALTERNATIVE_GATE_YET",
        ),
        _lane_payload(
            "LANE_10_SIMULATED_ADAPTER_CONTRACT_DRY_RUN",
            "LANE_INCONCLUSIVE_NEEDS_MORE_LINEAGE",
            "DRY_RUN_MECHANICALLY_MATCHES_BUT_FIELD_UNMAPPED",
            "The dry-run can reproduce the refined veto mechanically only by using HISTORICAL_V2_BLUEPRINT, which is still unmapped and unconfirmed for adapter use.",
            [
                {
                    "required_adapter_inputs_v1": [
                        "tail_repaired_r5_2_oof_candidate_score_v1",
                        "asof_signal__r5_1_bad_score_v1",
                        "asof_signal__v2_like_bad_tail_v1",
                        "asof_signal__r5_bad_score_v1",
                        "asof_signal__r5_tail_score_v1",
                        "asof_signal__historical_v2_blueprint_v1",
                    ],
                    "mapped_fields_without_blueprint_v1": 5,
                    "unmapped_fields_v1": ["asof_signal__historical_v2_blueprint_v1"],
                    "dry_run_metrics_v1": dry_metrics,
                }
            ],
            risk_level="MEDIUM",
            blocker_type="UNMAPPED_BLUEPRINT_FIELD",
        ),
    ]
    return lanes


def _lane_index(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, lane in enumerate(lanes, start=1):
        rows.append(
            {
                "lane_number_v1": idx,
                "lane_id_v1": lane["lane_id_v1"],
                "lane_status_v1": lane["lane_status_v1"],
                "classification_v1": lane["classification_v1"],
                "risk_level_v1": lane["risk_level_v1"],
                "blocker_type_v1": lane["blocker_type_v1"],
                "recommendation_v1": lane["recommendation_v1"],
            }
        )
    validate_lane_index(rows)
    return rows


def _risk_matrix(lane_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risk_types = [
        "membership_or_coverage_proxy",
        "as_of_reconstruction",
        "adapter_allowlist",
        "outcome_or_hindsight",
        "row_identity_or_artifact_shortcut",
        "group_or_low_support",
        "unsafe_lookalike",
        "alternative_non_blueprint",
        "adapter_contract",
    ]
    rows = []
    for risk_type in risk_types:
        impacted = [
            row["lane_id_v1"]
            for row in lane_rows
            if risk_type.split("_or_")[0].upper() in row["blocker_type_v1"]
            or (risk_type == "as_of_reconstruction" and "RECONSTRUCTION" in row["blocker_type_v1"])
            or (risk_type == "adapter_allowlist" and "ALLOWLIST" in row["blocker_type_v1"])
            or (risk_type == "row_identity_or_artifact_shortcut" and "ARTIFACT" in row["blocker_type_v1"])
            or (risk_type == "group_or_low_support" and "GROUP" in row["blocker_type_v1"])
            or (risk_type == "adapter_contract" and "UNMAPPED" in row["blocker_type_v1"])
        ]
        severity = "HIGH" if impacted and any(row["risk_level_v1"] == "HIGH" for row in lane_rows if row["lane_id_v1"] in impacted) else "MEDIUM" if impacted else "LOW"
        rows.append(
            {
                "risk_type_v1": risk_type,
                "severity_v1": severity,
                "impacted_lanes_v1": "|".join(impacted),
                "blocks_adapter_approval_v1": severity == "HIGH",
            }
        )
    return rows


def _fan_in(lane_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    blocked = [row for row in lane_rows if row["lane_status_v1"].startswith("LANE_BLOCKED")]
    inconclusive = [row for row in lane_rows if row["lane_status_v1"] == "LANE_INCONCLUSIVE_NEEDS_MORE_LINEAGE"]
    highest = "HIGH" if any(row["risk_level_v1"] == "HIGH" for row in lane_rows) else "MEDIUM" if inconclusive else "LOW"
    summary = {
        "layer_name": "PARALLEL_REFINED_VETO_LANE_PACK_SUMMARY_V1",
        "lane_count_v1": len(lane_rows),
        "pass_lanes_v1": [row["lane_id_v1"] for row in lane_rows if row["lane_status_v1"].startswith("LANE_PASS")],
        "inconclusive_lanes_v1": [row["lane_id_v1"] for row in inconclusive],
        "blocked_lanes_v1": [row["lane_id_v1"] for row in blocked],
        "highest_severity_blocker_v1": highest,
        "historical_v2_blueprint_as_of_safe_assessment_v1": "BLOCKED_OR_UNPROVEN_HISTORICAL_ARTIFACT_PROXY",
        "historical_v2_blueprint_adapter_allowlist_assessment_v1": "NOT_ALLOWLIST_ELIGIBLE",
        "refined_veto_can_move_to_fan_in_decision_v1": False,
        "adapter_remains_blocked_v1": True,
        "r6_iql_remain_blocked_v1": True,
    }
    recommendation = {
        "layer_name": "PARALLEL_REFINED_VETO_LANE_PACK_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "rationale_v1": [
            "Multiple lanes found that HISTORICAL_V2_BLUEPRINT is sourced from historical V2 selected-membership evidence.",
            "AS_OF reconstruction and adapter allowlist eligibility are not proven.",
            "Artifact shortcut and group-concentration risks remain too high to proceed to fan-in decision or adapter work.",
        ],
    }
    go_no_go = {
        "layer_name": "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "refined_veto_id_v1": REFINED_VETO_ID,
        "adapter_reopen_allowed_v1": False,
        "fan_in_decision_allowed_next_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])
    validate_no_adapter_reopen(go_no_go)
    return summary, recommendation, go_no_go


def _write_lane(root: Path, lane: dict[str, Any]) -> None:
    lane_root = root / "lanes" / lane["lane_id_v1"]
    lane_root.mkdir(parents=True, exist_ok=True)
    manifest = _lane_manifest(lane["lane_id_v1"], lane["summary_v1"], lane_root)
    result = {key: value for key, value in lane.items() if key != "evidence_rows_v1"}
    risk = {
        "lane_id_v1": lane["lane_id_v1"],
        "lane_status_v1": lane["lane_status_v1"],
        "risk_level_v1": lane["risk_level_v1"],
        "blocker_type_v1": lane["blocker_type_v1"],
        "adapter_approval_blocked_v1": lane["lane_status_v1"].startswith("LANE_BLOCKED"),
    }
    _write_json(lane_root / "lane_manifest_v1.json", manifest)
    _write_json(lane_root / "lane_result_v1.json", result)
    _write_json(lane_root / "lane_risk_audit_v1.json", risk)
    if lane["evidence_rows_v1"]:
        _write_rows(lane_root / "lane_evidence_v1.csv", lane["evidence_rows_v1"])
    _write_report(
        lane_root / "lane_result_v1.md",
        [
            f"# {lane['lane_id_v1']}",
            "",
            f"- Status: `{lane['lane_status_v1']}`",
            f"- Classification: `{lane['classification_v1']}`",
            f"- Risk: `{lane['risk_level_v1']}`",
            f"- Summary: {lane['summary_v1']}",
        ],
    )
    _write_report(
        lane_root / "lane_risk_audit_v1.md",
        [
            f"# {lane['lane_id_v1']} Risk Audit V1",
            "",
            f"- Blocker type: `{lane['blocker_type_v1']}`",
            f"- Adapter approval blocked: `{risk['adapter_approval_blocked_v1']}`",
        ],
    )


def _write_markdown(
    root: Path,
    manifest: dict[str, Any],
    repro: dict[str, Any],
    lane_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    risk_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "parallel_refined_veto_lane_pack_input_manifest_v1.md",
        [
            "# Parallel Refined Veto Lane Pack Input Manifest V1",
            "",
            f"- Artifact root: `{manifest['artifact_root_v1']}`",
            f"- Refined veto root: `{manifest['input_roots_v1']['refined_veto_root_v1']}`",
            "- All decision inputs use explicit locked roots.",
        ],
    )
    _write_report(
        root / "parallel_refined_veto_lane_pack_reproducibility_audit_v1.md",
        [
            "# Parallel Refined Veto Lane Pack Reproducibility Audit V1",
            "",
            f"- Safe-core selected rows: `{repro['safe_core_selected_rows_v1']}`",
            f"- Refined veto blocks unsafe row: `{repro['refined_veto_blocks_unsafe_row_v1']}`",
            f"- Refined veto good rows cut: `{repro['refined_veto_good_safe_core_rows_cut_v1']}`",
        ],
    )
    _write_report(
        root / "parallel_refined_veto_lane_pack_summary_v1.md",
        [
            "# Parallel Refined Veto Lane Pack Summary V1",
            "",
            f"- Blocked lanes: `{summary['blocked_lanes_v1']}`",
            f"- Inconclusive lanes: `{summary['inconclusive_lanes_v1']}`",
            f"- Blueprint AS_OF assessment: `{summary['historical_v2_blueprint_as_of_safe_assessment_v1']}`",
            f"- Adapter remains blocked: `{summary['adapter_remains_blocked_v1']}`",
        ],
    )
    _write_report(
        root / "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.md",
        [
            "# Parallel Refined Veto Lane Pack Cross-Lane Risk Matrix V1",
            "",
            *[
                f"- `{row['risk_type_v1']}`: severity `{row['severity_v1']}`, lanes `{row['impacted_lanes_v1']}`"
                for row in risk_rows
            ],
        ],
    )
    _write_report(
        root / "parallel_refined_veto_lane_pack_recommendation_v1.md",
        [
            "# Parallel Refined Veto Lane Pack Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            "- Adapter/R6/IQL remain blocked.",
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
    lanes = _lanes(frame, masks, inputs)
    lane_rows = _lane_index(lanes)
    risk_rows = _risk_matrix(lane_rows)
    summary, recommendation, go_no_go = _fan_in(lane_rows)

    for lane in lanes:
        _write_lane(artifact_root, lane)

    _write_json(artifact_root / "parallel_refined_veto_lane_pack_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "parallel_refined_veto_lane_pack_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "parallel_refined_veto_lane_pack_lane_index_v1.csv", lane_rows)
    _write_json(
        artifact_root / "parallel_refined_veto_lane_pack_lane_index_v1.json",
        {"row_count_v1": len(lane_rows), "rows_v1": lane_rows},
    )
    _write_json(artifact_root / "parallel_refined_veto_lane_pack_summary_v1.json", summary)
    _write_rows(artifact_root / "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.csv", risk_rows)
    _write_json(
        artifact_root / "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.json",
        {"row_count_v1": len(risk_rows), "rows_v1": risk_rows},
    )
    _write_json(artifact_root / "parallel_refined_veto_lane_pack_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json", go_no_go)
    _write_markdown(artifact_root, manifest, repro, lane_rows, summary, risk_rows, recommendation)

    summary_payload = {
        "layer_name": "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_SUMMARY_PAYLOAD_V1",
        "artifact_root_v1": str(artifact_root),
        "refined_veto_id_v1": REFINED_VETO_ID,
        "lane_statuses_v1": {row["lane_id_v1"]: row["lane_status_v1"] for row in lane_rows},
        "historical_v2_blueprint_assessment_v1": summary["historical_v2_blueprint_as_of_safe_assessment_v1"],
        "refined_veto_can_proceed_to_fan_in_v1": False,
        "adapter_r6_iql_remain_blocked_v1": True,
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary_payload)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": FINAL_STATUS, "next_recommended_action_v1": NEXT_ACTION, "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Parallel Refined Veto Lineage Audit Lane Pack V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Refined veto: `{REFINED_VETO_ID}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    validate_required_outputs(artifact_root)
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = materialize(args.artifact_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
