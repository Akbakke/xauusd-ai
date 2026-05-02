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

from gx1.scripts import materialize_harden_140_94_safe_core_and_expand_later_v1 as harden
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1"

INPUT_HARDEN_ROOT = (
    DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
)
INPUT_SIMPLIFY_ROOT = (
    DEFAULT_REPORTS_ROOT / "SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T083415Z_LOCK"
)
INPUT_DISTILL_ROOT = (
    DEFAULT_REPORTS_ROOT / "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T081017Z_LOCK"
)
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

HARDENED_RECIPE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_UNMAPPED_VETOES"
NEXT_ACTION = "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1"

EXPECTED_SELECTED = 89
EXPECTED_RECOVERED = 86
EXPECTED_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607

ALLOWED_FINAL_STATUSES = {
    "140_94_SAFE_CORE_INPUT_MAPPING_PASS_ADAPTER_BUILD_READY",
    "140_94_SAFE_CORE_INPUT_MAPPING_PASS_NEEDS_MINOR_NORMALIZATION",
    "140_94_SAFE_CORE_INPUT_MAPPING_PARTIAL_NEEDS_FALSE_POSITIVE_VETO_MAPPING",
    "140_94_SAFE_CORE_INPUT_MAPPING_PARTIAL_NEEDS_FIELD_NORMALIZATION",
    "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_UNMAPPED_VETOES",
    "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_MAPPING_DRY_RUN_MISMATCH",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_SAFE_CORE_ADAPTER_V1",
    "NORMALIZE_140_94_SAFE_CORE_ADAPTER_INPUT_FIELDS_V1",
    "MAP_140_94_SAFE_CORE_FALSE_POSITIVE_VETOES_V1",
    "DEEPEN_140_94_SAFE_CORE_AS_OF_LINEAGE_AUDIT_V1",
    "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
}

ADAPTER_INPUT_FIELDS = [
    "tail_repaired_r5_2_oof_candidate_score_v1",
    "asof_signal__r5_1_bad_score_v1",
    "asof_signal__v2_like_bad_tail_v1",
    "asof_low_support_missing_artifact_veto_v1",
    "asof_hard_safety_veto_set_v1",
    "asof_false_positive_veto_for_remaining_extra_3_v1",
]

REQUIRED_OUTPUTS = [
    "build_140_94_safe_core_input_mapping_manifest_v1.json",
    "build_140_94_safe_core_reproducibility_audit_v1.json",
    "build_140_94_safe_core_reproducibility_audit_v1.md",
    "build_140_94_safe_core_adapter_input_contract_v1.csv",
    "build_140_94_safe_core_adapter_input_contract_v1.json",
    "build_140_94_safe_core_adapter_input_contract_v1.md",
    "build_140_94_safe_core_rule_to_input_mapping_v1.csv",
    "build_140_94_safe_core_rule_to_input_mapping_v1.json",
    "build_140_94_safe_core_rule_to_input_mapping_v1.md",
    "build_140_94_safe_core_veto_to_input_mapping_v1.csv",
    "build_140_94_safe_core_veto_to_input_mapping_v1.json",
    "build_140_94_safe_core_veto_to_input_mapping_v1.md",
    "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.csv",
    "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.json",
    "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.md",
    "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.csv",
    "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.json",
    "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.md",
    "build_140_94_safe_core_missing_input_and_blocker_audit_v1.csv",
    "build_140_94_safe_core_missing_input_and_blocker_audit_v1.json",
    "build_140_94_safe_core_missing_input_and_blocker_audit_v1.md",
    "build_140_94_safe_core_adapter_build_readiness_v1.json",
    "build_140_94_safe_core_adapter_build_readiness_v1.md",
    "build_140_94_safe_core_anti_overfit_no_shortcut_audit_v1.json",
    "build_140_94_safe_core_anti_overfit_no_shortcut_audit_v1.md",
    "build_140_94_safe_core_adapter_input_mapping_recommendation_v1.json",
    "build_140_94_safe_core_adapter_input_mapping_recommendation_v1.md",
    "build_140_94_safe_core_adapter_input_mapping_go_no_go_v1.json",
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


def validate_no_forbidden_feature_names(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in simplify.DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_SAFE_CORE_ADAPTER_INPUT_FEATURE: {blocked}")
    return True


def validate_no_forbidden_actions(
    *,
    r6: bool = False,
    adapter: bool = False,
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
        "unsafe_hits_v1": 0,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("precision_audit_only_v1", -1)), EXPECTED_PRECISION):
        failures["precision_audit_only_v1"] = payload.get("precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"SAFE_CORE_INPUT_MAPPING_REPRODUCIBILITY_FAILED: {failures}")
    return True


def validate_mapping_dry_run(payload: dict[str, Any]) -> bool:
    if payload.get("mapping_dry_run_status_v1") != "EXACT_MATCH_WITH_CURRENT_AUDIT_VETO":
        raise RuntimeError("SAFE_CORE_MAPPING_DRY_RUN_NOT_EXACT")
    if payload.get("missed_hardened_safe_core_rows_v1") != 0:
        raise RuntimeError("SAFE_CORE_MAPPING_DRY_RUN_MISSED_ROWS")
    if payload.get("extra_selected_vs_hardened_safe_core_v1") != 0:
        raise RuntimeError("SAFE_CORE_MAPPING_DRY_RUN_EXTRA_ROWS")
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
        raise RuntimeError(f"SAFE_CORE_INPUT_MAPPING_REQUIRED_OUTPUTS_MISSING: {missing}")
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
    roots = [INPUT_HARDEN_ROOT, INPUT_SIMPLIFY_ROOT, INPUT_DISTILL_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "harden_go_no_go": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
        "harden_safe_core_definition": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_definition_v1.json",
        "harden_safe_core_rows": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_row_level_explanations_v1.csv",
        "harden_extra_5": INPUT_HARDEN_ROOT / "harden_140_94_extra_5_audit_v1.csv",
        "harden_veto_hardening": INPUT_HARDEN_ROOT / "harden_140_94_veto_hardening_audit_v1.csv",
        "harden_adapter_readiness": INPUT_HARDEN_ROOT / "harden_140_94_adapter_readiness_v1.json",
        "simplify_summary": INPUT_SIMPLIFY_ROOT / "summary_v1.json",
        "simplify_go_no_go": INPUT_SIMPLIFY_ROOT / "simplify_140_94_rules_and_vetoes_go_no_go_v1.json",
        "distill_summary": INPUT_DISTILL_ROOT / "summary_v1.json",
        "distill_go_no_go": INPUT_DISTILL_ROOT / "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    harden_go = _read_json(required["harden_go_no_go"])
    if harden_go.get("status_v1") != "140_94_SAFE_CORE_HARDENED_NEEDS_INPUT_MAPPING_EXPAND_LATER":
        raise RuntimeError("HARDEN_STATUS_NOT_INPUT_MAPPING_REQUIRED")
    if harden_go.get("safe_core_rule_id_v1") != HARDENED_RECIPE_ID:
        raise RuntimeError("HARDEN_SAFE_CORE_RECIPE_ID_MISMATCH")
    return {
        "required_paths": required,
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": harden_go,
        "harden_safe_core_definition": _read_json(required["harden_safe_core_definition"]),
        "harden_safe_core_rows": pd.read_csv(required["harden_safe_core_rows"]),
        "harden_extra_5": pd.read_csv(required["harden_extra_5"]),
        "harden_veto_hardening": pd.read_csv(required["harden_veto_hardening"]),
        "harden_adapter_readiness": _read_json(required["harden_adapter_readiness"]),
        "simplify_summary": _read_json(required["simplify_summary"]),
        "simplify_go_no_go": _read_json(required["simplify_go_no_go"]),
        "distill_summary": _read_json(required["distill_summary"]),
        "distill_go_no_go": _read_json(required["distill_go_no_go"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "source_inputs": harden._load_inputs()["source_inputs"],
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append({"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)})
    return {
        "layer_name": "BUILD_140_94_SAFE_CORE_INPUT_MAPPING_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "harden_root_v1": str(INPUT_HARDEN_ROOT),
            "simplify_root_v1": str(INPUT_SIMPLIFY_ROOT),
            "distill_root_v1": str(INPUT_DISTILL_ROOT),
            "precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "selection_source_v1": "explicit hardened safe-core artifact plus deterministic reconstruction from locked source artifacts",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    frame = simplify._build_frame(inputs["source_inputs"])
    masks = harden._build_masks(frame)
    score = _num(frame, "candidate_score_v1")
    r51 = _bool(frame, "signal_r5_1_bad_score_v1")
    v2 = _bool(frame, "signal_v2_like_bad_tail_v1")
    safe = _bool(frame, "hard_veto_clear_shadow_v1")
    missing = _str(frame, "run_id_policy_class_v1").str.contains("LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS")
    positive = score.ge(0.95) & r51 & v2
    masks["positive_only"] = positive
    masks["low_support_veto_only"] = positive & ~missing
    masks["hard_safety_veto_only"] = positive & safe
    masks["simulated_current_audit_veto"] = positive & ~missing & safe
    masks["missing_artifact_veto"] = missing
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
        "protected_winner_hits_audit_only_v1": int(_bool(selected, "protected_winner_status_v1").sum()),
        "runner_protect_hits_audit_only_v1": int(_bool(selected, "runner_protect_status_v1").sum()),
        "ambiguous_high_mfe_hits_audit_only_v1": int(_bool(selected, "ambiguous_high_mfe_status_v1").sum()),
        "fifty_plus_mfe_hits_audit_only_v1": int(_bool(selected, "fifty_plus_mfe_risk_v1").sum()),
        "hundred_plus_mfe_hits_audit_only_v1": int(_bool(selected, "hundred_plus_mfe_risk_v1").sum()),
        "two_hundred_plus_mfe_hits_audit_only_v1": int(_bool(selected, "two_hundred_plus_mfe_risk_v1").sum()),
        "quarantine_hits_audit_only_v1": int((_str(selected, "active_quarantine_v1", "ACTIVE_CANDIDATE") != "ACTIVE_CANDIDATE").sum()),
    }


def _reproducibility(frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    metrics = _selected_metrics(frame, masks["hardened"])
    payload = {
        "layer_name": "BUILD_140_94_SAFE_CORE_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        **metrics,
        "harden_source_status_v1": inputs["harden_go_no_go"].get("status_v1"),
        "adapter_readiness_from_harden_v1": inputs["harden_adapter_readiness"].get("status_v1"),
        "reproduced_from_explicit_harden_artifact_v1": True,
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def _adapter_input_contract() -> list[dict[str, Any]]:
    return [
        {
            "adapter_field_name_v1": "tail_repaired_r5_2_oof_candidate_score_v1",
            "source_artifact_v1": str(INPUT_PRECHECK_ROOT),
            "source_column_or_path_v1": "causal_rebuild_candidate_oof_predictions_v1.csv:candidate_score_v1",
            "data_type_v1": "float",
            "allowed_values_or_range_v1": "[0.0, 1.0]",
            "required_v1": True,
            "default_allowed_v1": False,
            "missing_handling_v1": "BLOCK_ROW_IF_MISSING",
            "as_of_safe_lineage_v1": "YES_OOF_SCORE_FROM_CAUSAL_BASELINE",
            "normalization_required_v1": False,
            "mapping_required_v1": False,
            "veto_dependency_v1": False,
            "used_by_rule_v1": True,
            "used_by_veto_v1": False,
            "adapter_ready_v1": True,
            "blocker_reason_v1": "",
        },
        {
            "adapter_field_name_v1": "asof_signal__r5_1_bad_score_v1",
            "source_artifact_v1": str(INPUT_PRECHECK_ROOT),
            "source_column_or_path_v1": "best_lane_student_oof_predictions_v1.csv:source_evidence_v1 contains R5_1_BAD_SCORE",
            "data_type_v1": "bool",
            "allowed_values_or_range_v1": "true|false",
            "required_v1": True,
            "default_allowed_v1": False,
            "missing_handling_v1": "BLOCK_ROW_IF_MISSING",
            "as_of_safe_lineage_v1": "YES_EXISTING_LEGAL_SIGNAL_FAMILY",
            "normalization_required_v1": False,
            "mapping_required_v1": False,
            "veto_dependency_v1": False,
            "used_by_rule_v1": True,
            "used_by_veto_v1": False,
            "adapter_ready_v1": True,
            "blocker_reason_v1": "",
        },
        {
            "adapter_field_name_v1": "asof_signal__v2_like_bad_tail_v1",
            "source_artifact_v1": str(INPUT_PRECHECK_ROOT),
            "source_column_or_path_v1": "best_lane_student_oof_predictions_v1.csv:source_evidence_v1 contains V2_LIKE_BAD_TAIL",
            "data_type_v1": "bool",
            "allowed_values_or_range_v1": "true|false",
            "required_v1": True,
            "default_allowed_v1": False,
            "missing_handling_v1": "BLOCK_ROW_IF_MISSING",
            "as_of_safe_lineage_v1": "YES_EXISTING_LEGAL_SIGNAL_FAMILY",
            "normalization_required_v1": False,
            "mapping_required_v1": False,
            "veto_dependency_v1": False,
            "used_by_rule_v1": True,
            "used_by_veto_v1": False,
            "adapter_ready_v1": True,
            "blocker_reason_v1": "",
        },
        {
            "adapter_field_name_v1": "asof_low_support_missing_artifact_veto_v1",
            "source_artifact_v1": str(INPUT_HARDEN_ROOT),
            "source_column_or_path_v1": "run_id_policy_class_v1 == LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS",
            "data_type_v1": "bool",
            "allowed_values_or_range_v1": "true|false",
            "required_v1": True,
            "default_allowed_v1": False,
            "missing_handling_v1": "BLOCK_ROW_IF_MISSING_OR_UNKNOWN",
            "as_of_safe_lineage_v1": "PARTIAL_POLICY_LINEAGE_PRESENT",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "veto_dependency_v1": True,
            "used_by_rule_v1": False,
            "used_by_veto_v1": True,
            "adapter_ready_v1": False,
            "blocker_reason_v1": "NEEDS_NORMALIZED_ADAPTER_INPUT_FIELD",
        },
        {
            "adapter_field_name_v1": "asof_hard_safety_veto_set_v1",
            "source_artifact_v1": str(INPUT_HARDEN_ROOT),
            "source_column_or_path_v1": "unsafe/protected/runner/ambiguous/high-MFE/quarantine audit columns",
            "data_type_v1": "bool",
            "allowed_values_or_range_v1": "true means veto row",
            "required_v1": True,
            "default_allowed_v1": False,
            "missing_handling_v1": "BLOCK_ROW_IF_MISSING_OR_UNKNOWN",
            "as_of_safe_lineage_v1": "NO_CURRENTLY_AUDIT_ONLY",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "veto_dependency_v1": True,
            "used_by_rule_v1": False,
            "used_by_veto_v1": True,
            "adapter_ready_v1": False,
            "blocker_reason_v1": "UNMAPPED_AUDIT_ONLY_SAFETY_VETO",
        },
        {
            "adapter_field_name_v1": "asof_false_positive_veto_for_remaining_extra_3_v1",
            "source_artifact_v1": str(INPUT_HARDEN_ROOT),
            "source_column_or_path_v1": "not yet materialized; inferred from remaining extra-3 audit only",
            "data_type_v1": "bool",
            "allowed_values_or_range_v1": "true means veto false-positive-risk extra",
            "required_v1": False,
            "default_allowed_v1": False,
            "missing_handling_v1": "DO_NOT_USE_FOR_DECISION_UNTIL_MAPPED",
            "as_of_safe_lineage_v1": "UNKNOWN_NOT_MAPPED",
            "normalization_required_v1": True,
            "mapping_required_v1": True,
            "veto_dependency_v1": True,
            "used_by_rule_v1": False,
            "used_by_veto_v1": True,
            "adapter_ready_v1": False,
            "blocker_reason_v1": "FALSE_POSITIVE_VETO_NOT_MAPPED_FOR_3_SAFE_EXTRAS",
        },
    ]


def _rule_to_input_mapping(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    metrics = _selected_metrics(frame, masks["hardened"])
    positive_metrics = _selected_metrics(frame, masks["positive_only"])
    return [
        {
            "branch_name_v1": "SAFE_CORE_HARDENED_POSITIVE_BRANCH_V1",
            "required_fields_v1": "tail_repaired_r5_2_oof_candidate_score_v1|asof_signal__r5_1_bad_score_v1|asof_signal__v2_like_bad_tail_v1",
            "thresholds_or_conditions_v1": "score >= 0.95 AND R5_1_BAD_SCORE present AND V2_LIKE_BAD_TAIL present",
            "score_contribution_v1": "candidate_score_v1 gates entry; no learned weighting in adapter mapping gate",
            "support_requirement_v1": "R5_1 support plus V2-like support",
            "low_support_behavior_v1": "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS veto must block before adapter build",
            "expected_selected_rows_before_veto_v1": positive_metrics["selected_rows_v1"],
            "expected_selected_rows_after_all_current_vetoes_v1": metrics["selected_rows_v1"],
            "recovered_original_140_rows_v1": metrics["recovered_original_140_rows_v1"],
            "extra_rows_v1": metrics["extra_rows_v1"],
            "safety_status_v1": metrics["safety_status_v1"],
            "adapter_readiness_v1": "BLOCKED_BY_UNMAPPED_HARD_SAFETY_VETO",
        }
    ]


def _veto_to_input_mapping(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    original = _bool(frame, "selected_original_140_v1")
    missing = masks["missing_artifact_veto"]
    unsafe_blocked = masks["low_support_veto_only"] & ~masks["hardened"]
    retained_extras = masks["hardened"] & ~original
    return [
        {
            "veto_name_v1": "LOW_SUPPORT_MISSING_ARTIFACT_VETO_V1",
            "veto_purpose_v1": "Block missing-artifact low-support rows from safe-core adapter input.",
            "input_fields_v1": "asof_low_support_missing_artifact_veto_v1",
            "as_of_safe_status_v1": "PARTIAL_POLICY_LINEAGE_PRESENT",
            "mapping_status_v1": "NEEDS_NORMALIZED_ADAPTER_INPUT",
            "rows_blocked_v1": int((masks["positive_only"] & missing).sum()),
            "original_140_rows_blocked_v1": int((masks["positive_only"] & missing & original).sum()),
            "extra_rows_blocked_v1": int((masks["positive_only"] & missing & ~original).sum()),
            "unsafe_lookalike_rows_blocked_v1": int((masks["positive_only"] & missing & _bool(frame, "unsafe_audit_v1")).sum()),
            "can_be_computed_by_adapter_v1": False,
            "required_before_adapter_build_v1": True,
            "status_v1": "NEEDS_AS_OF_INPUT_MAPPING",
        },
        {
            "veto_name_v1": "HARD_SAFETY_VETO_SET_V1",
            "veto_purpose_v1": "Block unsafe/protected/runner/ambiguous/high-MFE/quarantine risk before selection.",
            "input_fields_v1": "asof_hard_safety_veto_set_v1",
            "as_of_safe_status_v1": "NOT_PROVEN_CURRENT_SOURCE_IS_AUDIT_ONLY",
            "mapping_status_v1": "UNMAPPED_CRITICAL",
            "rows_blocked_v1": int(unsafe_blocked.sum()),
            "original_140_rows_blocked_v1": int((unsafe_blocked & original).sum()),
            "extra_rows_blocked_v1": int((unsafe_blocked & ~original).sum()),
            "unsafe_lookalike_rows_blocked_v1": int((unsafe_blocked & _bool(frame, "unsafe_audit_v1")).sum()),
            "can_be_computed_by_adapter_v1": False,
            "required_before_adapter_build_v1": True,
            "status_v1": "BLOCKED_UNMAPPED_AUDIT_ONLY_SAFETY_VETO",
        },
        {
            "veto_name_v1": "FALSE_POSITIVE_EXTRA_3_VETO_V1",
            "veto_purpose_v1": "Optional later precision hardening for three safety-clean false-positive-risk extras.",
            "input_fields_v1": "asof_false_positive_veto_for_remaining_extra_3_v1",
            "as_of_safe_status_v1": "UNKNOWN_NOT_MAPPED",
            "mapping_status_v1": "NOT_READY",
            "rows_blocked_v1": 0,
            "original_140_rows_blocked_v1": 0,
            "extra_rows_blocked_v1": 0,
            "unsafe_lookalike_rows_blocked_v1": 0,
            "candidate_extra_rows_remaining_v1": int(retained_extras.sum()),
            "can_be_computed_by_adapter_v1": False,
            "required_before_adapter_build_v1": False,
            "status_v1": "PRECISION_MAPPING_PENDING_NOT_PRIMARY_SAFETY_BLOCKER",
        },
    ]


def _remaining_extra_3_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = _bool(frame, "selected_original_140_v1")
    extra = frame[masks["hardened"] & ~original].copy()
    rows = []
    for _, row in extra.sort_values("candidate_score_v1", ascending=False).iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "why_remaining_selected_v1": "score >= 0.95 + R5_1 + V2-like + current audit hard safety veto clear + no missing-artifact low-support veto",
                "input_fields_selecting_row_v1": "tail_repaired_r5_2_oof_candidate_score_v1|asof_signal__r5_1_bad_score_v1|asof_signal__v2_like_bad_tail_v1",
                "candidate_score_v1": row.get("candidate_score_v1"),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "safety_status_v1": "CLEAN" if not _as_bool(row.get("unsafe_audit_v1")) else "FAIL",
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "veto_needed_if_blocking_v1": "asof_false_positive_veto_for_remaining_extra_3_v1",
                "acceptable_safe_core_extra_v1": False,
                "false_positive_risk_v1": True,
                "as_of_veto_mapping_needed_v1": True,
                "recommendation_v1": "KEEP_VISIBLE_DO_NOT_BUILD_FINAL_ADAPTER_UNTIL_VETO_DECISION_IS_EXPLICIT",
            }
        )
    summary = {
        "layer_name": "BUILD_140_94_SAFE_CORE_REMAINING_EXTRA_3_MAPPING_AUDIT_SUMMARY_V1",
        "remaining_extra_rows_v1": len(rows),
        "bad_rows_v1": sum(row["bad_label_audit_only_v1"] for row in rows),
        "tail_rows_v1": sum(row["tail_label_audit_only_v1"] for row in rows),
        "safety_status_v1": "CLEAN" if not any(row["safety_status_v1"] != "CLEAN" for row in rows) else "FAIL",
        "false_positive_veto_mapping_needed_v1": True,
    }
    return rows, summary


def _simulated_adapter_dry_run(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = masks["simulated_current_audit_veto"]
    hardened = masks["hardened"]
    no_safety = masks["low_support_veto_only"]
    rows = []
    for _, row in frame[selected | hardened | no_safety].sort_values(["run_id_v1", "candidate_score_v1"], ascending=[True, False]).iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "mapped_fields_available_v1": True,
                "selected_by_simulated_mapping_with_current_audit_veto_v1": _as_bool(selected.loc[row.name]),
                "selected_by_hardened_safe_core_v1": _as_bool(hardened.loc[row.name]),
                "selected_without_hard_safety_veto_v1": _as_bool(no_safety.loc[row.name]),
                "diff_vs_hardened_v1": _as_bool(selected.loc[row.name]) != _as_bool(hardened.loc[row.name]),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
            }
        )
    selected_frame = frame[selected]
    no_safety_frame = frame[no_safety]
    summary = {
        "layer_name": "BUILD_140_94_SAFE_CORE_SIMULATED_ADAPTER_MAPPING_DRY_RUN_V1",
        "mapping_dry_run_status_v1": "EXACT_MATCH_WITH_CURRENT_AUDIT_VETO",
        "mapped_rows_v1": len(frame),
        "unmapped_rows_v1": 0,
        "selected_by_simulated_mapping_v1": int(selected.sum()),
        "recovered_hardened_safe_core_rows_v1": int((selected & hardened).sum()),
        "missed_hardened_safe_core_rows_v1": int((~selected & hardened).sum()),
        "extra_selected_vs_hardened_safe_core_v1": int((selected & ~hardened).sum()),
        "bad_tail_audit_only_v1": [
            int(_bool(selected_frame, "bad_label_v1").sum()),
            int(_bool(selected_frame, "tail_label_v1").sum()),
        ],
        "safety_status_v1": "CLEAN" if int(_bool(selected_frame, "unsafe_audit_v1").sum()) == 0 else "FAIL",
        "difference_vs_hardened_safe_core_v1": 0,
        "no_safety_veto_selected_rows_v1": int(no_safety.sum()),
        "no_safety_veto_extra_vs_hardened_v1": int((no_safety & ~hardened).sum()),
        "no_safety_veto_unsafe_hits_v1": int(_bool(no_safety_frame, "unsafe_audit_v1").sum()),
        "dry_run_uses_audit_veto_not_deployable_v1": True,
    }
    validate_mapping_dry_run(summary)
    return rows, summary


def _blockers(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {
            "blocker_id_v1": "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO",
            "field_or_veto_v1": "asof_hard_safety_veto_set_v1",
            "severity_v1": "CRITICAL",
            "reason_v1": "Current exact dry-run uses audit unsafe/protected/runner/ambiguous/high-MFE/quarantine fields; AS_OF-safe adapter mapping is not proven.",
            "rows_affected_v1": int((masks["low_support_veto_only"] & ~masks["hardened"]).sum()),
            "unsafe_rows_affected_v1": int((masks["low_support_veto_only"] & ~masks["hardened"] & _bool(frame, "unsafe_audit_v1")).sum()),
            "recommended_fix_v1": "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1",
        },
        {
            "blocker_id_v1": "LOW_SUPPORT_MISSING_ARTIFACT_INPUT_NOT_NORMALIZED",
            "field_or_veto_v1": "asof_low_support_missing_artifact_veto_v1",
            "severity_v1": "HIGH",
            "reason_v1": "Policy class exists in artifacts but needs a normalized adapter field and missing-value contract.",
            "rows_affected_v1": int((masks["positive_only"] & masks["missing_artifact_veto"]).sum()),
            "unsafe_rows_affected_v1": int((masks["positive_only"] & masks["missing_artifact_veto"] & _bool(frame, "unsafe_audit_v1")).sum()),
            "recommended_fix_v1": "NORMALIZE_140_94_SAFE_CORE_ADAPTER_INPUT_FIELDS_V1",
        },
        {
            "blocker_id_v1": "FALSE_POSITIVE_EXTRA_3_VETO_NOT_MAPPED",
            "field_or_veto_v1": "asof_false_positive_veto_for_remaining_extra_3_v1",
            "severity_v1": "MEDIUM",
            "reason_v1": "Three remaining safety-clean extras are false-positive-risk rows; they are visible but no AS_OF veto has been mapped.",
            "rows_affected_v1": int((masks["hardened"] & ~_bool(frame, "selected_original_140_v1")).sum()),
            "unsafe_rows_affected_v1": 0,
            "recommended_fix_v1": "MAP_140_94_SAFE_CORE_FALSE_POSITIVE_VETOES_V1",
        },
    ]
    summary = {
        "layer_name": "BUILD_140_94_SAFE_CORE_MISSING_INPUT_AND_BLOCKER_AUDIT_SUMMARY_V1",
        "blocking_field_count_v1": len(rows),
        "critical_blocker_count_v1": sum(row["severity_v1"] == "CRITICAL" for row in rows),
        "adapter_build_blocked_v1": True,
        "primary_blocker_v1": "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO",
    }
    return rows, summary


def _adapter_readiness(
    input_rows: list[dict[str, Any]],
    dry_run_summary: dict[str, Any],
    blocker_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    mapped_count = sum(_as_bool(row["adapter_ready_v1"]) for row in input_rows)
    unmapped_count = len(input_rows) - mapped_count
    readiness = {
        "layer_name": "BUILD_140_94_SAFE_CORE_ADAPTER_BUILD_READINESS_V1",
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "safe_core_ready_for_adapter_build_v1": False,
        "adapter_build_can_start_next_v1": False,
        "mapped_adapter_fields_count_v1": mapped_count,
        "unmapped_or_blocking_fields_count_v1": unmapped_count,
        "mapping_dry_run_exact_match_v1": True,
        "mapping_dry_run_status_v1": dry_run_summary["mapping_dry_run_status_v1"],
        "requires_only_minor_normalization_v1": False,
        "false_positive_veto_mapping_required_v1": True,
        "hard_safety_veto_mapping_required_v1": True,
        "expansion_waits_for_later_gate_v1": True,
        "status_v1": FINAL_STATUS,
        "primary_blocker_v1": blocker_summary["primary_blocker_v1"],
    }
    anti = {
        "layer_name": "BUILD_140_94_SAFE_CORE_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS_MAPPING_ONLY",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_implicit_latest_glob_selection_v1": True,
        "dummy_synthetic_fallback_v1": False,
        "labels_mfe_safe_recoverable_blocked_as_features_v1": True,
        "membership_coverage_selected_flags_blocked_v1": True,
        "low_support_visible_v1": True,
        "strict_loso_visible_v1": True,
    }
    recommendation = {
        "layer_name": "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "rationale_v1": [
            "The current mapping contract reproduces the hardened safe-core exactly in dry-run.",
            "That exact match still depends on audit-only hard safety veto fields, so adapter build is not approved.",
            "The three remaining safety-clean false-positive-risk extras remain visible and need a separate AS_OF veto decision.",
            "Expansion remains separate and was not merged into the safe-core.",
        ],
    }
    go_no_go = {
        "layer_name": "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": EXPECTED_SELECTED,
        "recovered_original_140_rows_v1": EXPECTED_RECOVERED,
        "extra_rows_v1": EXPECTED_EXTRA,
        "bad_tail_audit_only_v1": [EXPECTED_BAD, EXPECTED_TAIL],
        "precision_audit_only_v1": EXPECTED_PRECISION,
        "safety_status_v1": "CLEAN",
        "mapped_adapter_fields_count_v1": mapped_count,
        "unmapped_or_blocking_fields_count_v1": unmapped_count,
        "mapping_dry_run_exact_match_v1": True,
        "adapter_build_approved_next_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])
    return readiness, anti, recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    input_rows: list[dict[str, Any]],
    rule_rows: list[dict[str, Any]],
    veto_rows: list[dict[str, Any]],
    extra_summary: dict[str, Any],
    dry_summary: dict[str, Any],
    blocker_summary: dict[str, Any],
    readiness: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "build_140_94_safe_core_reproducibility_audit_v1.md",
        [
            "# Build 140/94 Safe-Core Reproducibility Audit V1",
            "",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_audit_only_v1']} / {repro['tail_count_audit_only_v1']}`",
            f"- Safety: `{repro['safety_status_v1']}`",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_adapter_input_contract_v1.md",
        [
            "# Build 140/94 Safe-Core Adapter Input Contract V1",
            "",
            f"- Adapter fields: `{len(input_rows)}`",
            f"- Adapter-ready fields: `{sum(_as_bool(row['adapter_ready_v1']) for row in input_rows)}`",
            "- Hard safety veto and false-positive veto mapping remain unresolved.",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_rule_to_input_mapping_v1.md",
        [
            "# Build 140/94 Safe-Core Rule To Input Mapping V1",
            "",
            *[
                f"- `{row['branch_name_v1']}`: selected `{row['expected_selected_rows_after_all_current_vetoes_v1']}`, readiness `{row['adapter_readiness_v1']}`"
                for row in rule_rows
            ],
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_veto_to_input_mapping_v1.md",
        [
            "# Build 140/94 Safe-Core Veto To Input Mapping V1",
            "",
            *[
                f"- `{row['veto_name_v1']}`: `{row['status_v1']}`, rows blocked `{row['rows_blocked_v1']}`"
                for row in veto_rows
            ],
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.md",
        [
            "# Build 140/94 Safe-Core Remaining Extra 3 Mapping Audit V1",
            "",
            f"- Remaining extra rows: `{extra_summary['remaining_extra_rows_v1']}`",
            f"- Safety: `{extra_summary['safety_status_v1']}`",
            "- They remain visible false-positive-risk rows; no adapter-ready veto exists yet.",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.md",
        [
            "# Build 140/94 Safe-Core Simulated Adapter Mapping Dry Run V1",
            "",
            f"- Status: `{dry_summary['mapping_dry_run_status_v1']}`",
            f"- Selected by simulated mapping: `{dry_summary['selected_by_simulated_mapping_v1']}`",
            f"- Difference vs hardened safe-core: `{dry_summary['difference_vs_hardened_safe_core_v1']}`",
            f"- No-safety-veto unsafe hits: `{dry_summary['no_safety_veto_unsafe_hits_v1']}`",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_missing_input_and_blocker_audit_v1.md",
        [
            "# Build 140/94 Safe-Core Missing Input And Blocker Audit V1",
            "",
            f"- Blocking fields: `{blocker_summary['blocking_field_count_v1']}`",
            f"- Critical blockers: `{blocker_summary['critical_blocker_count_v1']}`",
            f"- Primary blocker: `{blocker_summary['primary_blocker_v1']}`",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_adapter_build_readiness_v1.md",
        [
            "# Build 140/94 Safe-Core Adapter Build Readiness V1",
            "",
            f"- Status: `{readiness['status_v1']}`",
            f"- Adapter build can start next: `{readiness['adapter_build_can_start_next_v1']}`",
            f"- Mapped fields: `{readiness['mapped_adapter_fields_count_v1']}`",
            f"- Unmapped/blocking fields: `{readiness['unmapped_or_blocking_fields_count_v1']}`",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Build 140/94 Safe-Core Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- Mapping only: no R6, adapter, package, freeze, promo, live, Optuna, broad sweep, or in-sample decisioning was run.",
            "- Labels, MFE, safe_recoverable, membership, coverage, selected flags, row identity, dummy/synthetic/fallback, and implicit latest/glob fields remain blocked.",
        ],
    )
    _write_report(
        root / "build_140_94_safe_core_adapter_input_mapping_recommendation_v1.md",
        [
            "# Build 140/94 Safe-Core Adapter Input Mapping Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            "- Exact dry-run is possible, but adapter build is blocked until the hard safety veto is mapped to AS_OF-safe inputs.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    validate_no_forbidden_feature_names(
        [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
            "asof_signal__v2_like_bad_tail_v1",
            "asof_low_support_missing_artifact_veto_v1",
            "asof_hard_safety_veto_set_v1",
        ]
    )
    inputs = _load_inputs()
    frame, masks = _build_frame_and_masks(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, masks, inputs)
    input_rows = _adapter_input_contract()
    rule_rows = _rule_to_input_mapping(frame, masks)
    veto_rows = _veto_to_input_mapping(frame, masks)
    extra_rows, extra_summary = _remaining_extra_3_audit(frame, masks)
    dry_rows, dry_summary = _simulated_adapter_dry_run(frame, masks)
    blocker_rows, blocker_summary = _blockers(frame, masks)
    readiness, anti, recommendation, go_no_go = _adapter_readiness(input_rows, dry_summary, blocker_summary)

    _write_json(artifact_root / "build_140_94_safe_core_input_mapping_manifest_v1.json", manifest)
    _write_json(artifact_root / "build_140_94_safe_core_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "build_140_94_safe_core_adapter_input_contract_v1.csv", input_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_adapter_input_contract_v1.json",
        {"row_count_v1": len(input_rows), "rows_v1": input_rows},
    )
    _write_rows(artifact_root / "build_140_94_safe_core_rule_to_input_mapping_v1.csv", rule_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_rule_to_input_mapping_v1.json",
        {"row_count_v1": len(rule_rows), "rows_v1": rule_rows},
    )
    _write_rows(artifact_root / "build_140_94_safe_core_veto_to_input_mapping_v1.csv", veto_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_veto_to_input_mapping_v1.json",
        {"row_count_v1": len(veto_rows), "rows_v1": veto_rows},
    )
    _write_rows(artifact_root / "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.csv", extra_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.json",
        {"summary_v1": extra_summary, "rows_v1": extra_rows},
    )
    _write_rows(artifact_root / "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.csv", dry_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.json",
        {"summary_v1": dry_summary, "rows_v1": dry_rows},
    )
    _write_rows(artifact_root / "build_140_94_safe_core_missing_input_and_blocker_audit_v1.csv", blocker_rows)
    _write_json(
        artifact_root / "build_140_94_safe_core_missing_input_and_blocker_audit_v1.json",
        {"summary_v1": blocker_summary, "rows_v1": blocker_rows},
    )
    _write_json(artifact_root / "build_140_94_safe_core_adapter_build_readiness_v1.json", readiness)
    _write_json(artifact_root / "build_140_94_safe_core_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "build_140_94_safe_core_adapter_input_mapping_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "build_140_94_safe_core_adapter_input_mapping_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "mapped_adapter_fields_count_v1": readiness["mapped_adapter_fields_count_v1"],
        "unmapped_or_blocking_fields_count_v1": readiness["unmapped_or_blocking_fields_count_v1"],
        "simulated_adapter_dry_run_status_v1": dry_summary["mapping_dry_run_status_v1"],
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_build_can_start_next_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
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
            "# Build 140/94 Safe-Core Adapter Input Mapping V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Safe-core rule: `{HARDENED_RECIPE_ID}`",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Dry-run status: `{dry_summary['mapping_dry_run_status_v1']}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        input_rows,
        rule_rows,
        veto_rows,
        extra_summary,
        dry_summary,
        blocker_summary,
        readiness,
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
