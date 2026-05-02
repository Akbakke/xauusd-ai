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

from gx1.scripts import materialize_build_140_94_safe_core_adapter_input_mapping_v1 as input_mapping
from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1"

INPUT_MAPPING_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1_20260428T090840Z_LOCK"
)
INPUT_HARDEN_ROOT = (
    DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
)
INPUT_SIMPLIFY_ROOT = (
    DEFAULT_REPORTS_ROOT / "SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T083415Z_LOCK"
)
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

HARDENED_RECIPE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_AUDIT_ONLY_VETO"
NEXT_ACTION = "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1"

EXPECTED_SELECTED = 89
EXPECTED_RECOVERED = 86
EXPECTED_EXTRA = 3
EXPECTED_BAD = 86
EXPECTED_TAIL = 55
EXPECTED_PRECISION = 0.9662921348314607

ALLOWED_FINAL_STATUSES = {
    "140_94_SAFE_CORE_VETO_MAPPING_PASS_ADAPTER_READY",
    "140_94_SAFE_CORE_VETO_MAPPING_PASS_NEEDS_MINOR_NORMALIZATION",
    "140_94_SAFE_CORE_VETO_MAPPING_PARTIAL_NEEDS_FALSE_POSITIVE_VETO_DECISION",
    "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_AUDIT_ONLY_VETO",
    "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_DRY_RUN_MISMATCH",
    "140_94_SAFE_CORE_VETO_MAPPING_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_SAFE_CORE_ADAPTER_V1",
    "NORMALIZE_140_94_SAFE_CORE_VETO_INPUTS_V1",
    "MAP_140_94_SAFE_CORE_FALSE_POSITIVE_VETOES_V1",
    "DEEPEN_140_94_SAFE_CORE_AS_OF_LINEAGE_AUDIT_V1",
    "RETURN_TO_140_94_SAFE_CORE_HARDENING_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
}

REQUIRED_OUTPUTS = [
    "deepen_140_94_veto_mapping_input_manifest_v1.json",
    "deepen_140_94_veto_mapping_reproducibility_audit_v1.json",
    "deepen_140_94_veto_mapping_reproducibility_audit_v1.md",
    "deepen_140_94_unmapped_veto_fields_audit_v1.csv",
    "deepen_140_94_unmapped_veto_fields_audit_v1.json",
    "deepen_140_94_unmapped_veto_fields_audit_v1.md",
    "deepen_140_94_hard_safety_veto_blocker_audit_v1.json",
    "deepen_140_94_hard_safety_veto_blocker_audit_v1.md",
    "deepen_140_94_candidate_as_of_veto_mappings_v1.csv",
    "deepen_140_94_candidate_as_of_veto_mappings_v1.json",
    "deepen_140_94_candidate_as_of_veto_mappings_v1.md",
    "deepen_140_94_unsafe_extra_without_veto_audit_v1.csv",
    "deepen_140_94_unsafe_extra_without_veto_audit_v1.json",
    "deepen_140_94_unsafe_extra_without_veto_audit_v1.md",
    "deepen_140_94_remaining_extra_3_veto_decision_v1.csv",
    "deepen_140_94_remaining_extra_3_veto_decision_v1.json",
    "deepen_140_94_remaining_extra_3_veto_decision_v1.md",
    "deepen_140_94_candidate_veto_dry_run_results_v1.csv",
    "deepen_140_94_candidate_veto_dry_run_results_v1.json",
    "deepen_140_94_candidate_veto_dry_run_results_v1.md",
    "deepen_140_94_final_veto_decision_v1.json",
    "deepen_140_94_final_veto_decision_v1.md",
    "deepen_140_94_adapter_readiness_after_veto_mapping_v1.json",
    "deepen_140_94_adapter_readiness_after_veto_mapping_v1.md",
    "deepen_140_94_veto_mapping_anti_overfit_no_shortcut_audit_v1.json",
    "deepen_140_94_veto_mapping_anti_overfit_no_shortcut_audit_v1.md",
    "deepen_140_94_veto_mapping_recommendation_v1.json",
    "deepen_140_94_veto_mapping_recommendation_v1.md",
    "deepen_140_94_safe_core_veto_mapping_audit_go_no_go_v1.json",
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
        raise RuntimeError(f"FORBIDDEN_SAFE_CORE_VETO_MAPPING_FEATURE: {blocked}")
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
        "without_hard_safety_veto_unsafe_extra_rows_v1": 1,
        "simulated_adapter_dry_run_status_v1": "EXACT_MATCH_WITH_CURRENT_AUDIT_VETO",
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("precision_audit_only_v1", -1)), EXPECTED_PRECISION):
        failures["precision_audit_only_v1"] = payload.get("precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"SAFE_CORE_VETO_MAPPING_REPRODUCIBILITY_FAILED: {failures}")
    return True


def validate_candidate_veto_rows(rows: list[dict[str, Any]]) -> bool:
    ids = {row["candidate_veto_name_v1"] for row in rows}
    required = {
        "CURRENT_AUDIT_HARD_SAFETY_VETO_REFERENCE",
        "AS_OF_SIGNAL_SHAPE_VETO_MISSING_R5_TAIL_AND_SCORE_GE_099",
        "AS_OF_SIGNAL_SHAPE_VETO_MISSING_R5_TAIL",
        "ROW_IDENTITY_SPECIFIC_VETO_FORBIDDEN",
    }
    missing = sorted(required - ids)
    if missing:
        raise RuntimeError(f"SAFE_CORE_VETO_CANDIDATE_SET_INCOMPLETE: {missing}")
    return True


def validate_final_veto_decision(payload: dict[str, Any]) -> bool:
    if payload.get("hard_safety_veto_final_status_v1") != "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE":
        raise RuntimeError("HARD_SAFETY_VETO_DECISION_MUST_REMAIN_BLOCKED")
    if payload.get("adapter_build_can_start_next_v1") is not False:
        raise RuntimeError("ADAPTER_BUILD_MUST_NOT_BE_APPROVED")
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
        raise RuntimeError(f"SAFE_CORE_VETO_MAPPING_REQUIRED_OUTPUTS_MISSING: {missing}")
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
    roots = [INPUT_MAPPING_ROOT, INPUT_HARDEN_ROOT, INPUT_SIMPLIFY_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "mapping_summary": INPUT_MAPPING_ROOT / "summary_v1.json",
        "mapping_go_no_go": INPUT_MAPPING_ROOT / "build_140_94_safe_core_adapter_input_mapping_go_no_go_v1.json",
        "mapping_input_contract": INPUT_MAPPING_ROOT / "build_140_94_safe_core_adapter_input_contract_v1.csv",
        "mapping_blockers": INPUT_MAPPING_ROOT / "build_140_94_safe_core_missing_input_and_blocker_audit_v1.csv",
        "mapping_dry_run": INPUT_MAPPING_ROOT / "build_140_94_safe_core_simulated_adapter_mapping_dry_run_v1.json",
        "mapping_extra_3": INPUT_MAPPING_ROOT / "build_140_94_safe_core_remaining_extra_3_mapping_audit_v1.csv",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "harden_go_no_go": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
        "simplify_summary": INPUT_SIMPLIFY_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    mapping_go = _read_json(required["mapping_go_no_go"])
    if mapping_go.get("status_v1") != "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_UNMAPPED_VETOES":
        raise RuntimeError("INPUT_MAPPING_STATUS_NOT_UNMAPPED_VETO_BLOCKED")
    return {
        "required_paths": required,
        "mapping_summary": _read_json(required["mapping_summary"]),
        "mapping_go_no_go": mapping_go,
        "mapping_input_contract": pd.read_csv(required["mapping_input_contract"]),
        "mapping_blockers": pd.read_csv(required["mapping_blockers"]),
        "mapping_dry_run": _read_json(required["mapping_dry_run"]),
        "mapping_extra_3": pd.read_csv(required["mapping_extra_3"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "simplify_summary": _read_json(required["simplify_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "source_inputs": input_mapping._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append({"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)})
    return {
        "layer_name": "DEEPEN_140_94_VETO_MAPPING_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "adapter_input_mapping_root_v1": str(INPUT_MAPPING_ROOT),
            "harden_root_v1": str(INPUT_HARDEN_ROOT),
            "simplify_root_v1": str(INPUT_SIMPLIFY_ROOT),
            "precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    frame, masks = input_mapping._build_frame_and_masks(inputs["source_inputs"])
    source = _str(frame, "source_evidence_v1")
    masks["veto_missing_r5_tail"] = ~source.str.contains("R5_TAIL_SCORE", regex=False)
    masks["veto_r5_bad_support"] = source.str.contains("R5_BAD_SCORE:SUPPORT", regex=False)
    masks["veto_missing_r5_tail_and_score_ge_099"] = masks["veto_missing_r5_tail"] & _num(frame, "candidate_score_v1").ge(0.99)
    masks["veto_row_identity_specific_forbidden"] = _str(frame, "candidate_uid_v1").eq(
        "TRUTH_MONFRI_WEEK_20260302_20260309:0:cand::001052:5ec4105e4e1a"
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
    metrics = _selected_metrics(frame, masks["hardened"])
    without = frame[masks["low_support_veto_only"] & ~masks["hardened"]]
    payload = {
        "layer_name": "DEEPEN_140_94_VETO_MAPPING_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        **metrics,
        "simulated_adapter_dry_run_status_v1": inputs["mapping_summary"].get("simulated_adapter_dry_run_status_v1"),
        "without_hard_safety_veto_selected_rows_v1": int(masks["low_support_veto_only"].sum()),
        "without_hard_safety_veto_unsafe_extra_rows_v1": int(_bool(without, "unsafe_audit_v1").sum()),
        "input_mapping_status_v1": inputs["mapping_go_no_go"].get("status_v1"),
        "reproduced_from_explicit_input_mapping_artifact_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def _unmapped_veto_fields(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    blockers = inputs["mapping_blockers"].to_dict("records")
    for row in blockers:
        field = row["field_or_veto_v1"]
        rows.append(
            {
                "veto_name_v1": row["blocker_id_v1"],
                "source_artifact_v1": str(INPUT_MAPPING_ROOT),
                "source_column_or_path_v1": field,
                "current_role_v1": "adapter blocker",
                "why_it_blocks_adapter_v1": row["reason_v1"],
                "audit_only_v1": field == "asof_hard_safety_veto_set_v1",
                "as_of_safe_lineage_exists_v1": field == "asof_low_support_missing_artifact_veto_v1",
                "mapping_candidate_exists_v1": field != "asof_false_positive_veto_for_remaining_extra_3_v1",
                "normalization_needed_v1": True,
                "severity_v1": row["severity_v1"],
                "rows_affected_v1": int(row["rows_affected_v1"]),
                "recommended_fix_v1": row["recommended_fix_v1"],
            }
        )
    summary = {
        "layer_name": "DEEPEN_140_94_UNMAPPED_VETO_FIELDS_AUDIT_SUMMARY_V1",
        "unmapped_veto_field_count_v1": len(rows),
        "audit_only_field_count_v1": sum(_as_bool(row["audit_only_v1"]) for row in rows),
        "primary_blocker_v1": "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO",
    }
    return rows, summary


def _unsafe_extra_rows(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    unsafe = frame[masks["low_support_veto_only"] & ~masks["hardened"]].copy()
    rows = []
    for _, row in unsafe.iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_rule_branch_v1": "score >= 0.95 + R5_1 + V2-like + low-support veto clear",
                "why_current_audit_veto_blocks_it_v1": "audit columns mark protected winner, ambiguous high-MFE, 50+ MFE and 100+ MFE risk",
                "candidate_score_v1": row.get("candidate_score_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "protected_winner_audit_only_v1": _as_bool(row.get("protected_winner_status_v1")),
                "runner_protect_audit_only_v1": _as_bool(row.get("runner_protect_status_v1")),
                "ambiguous_high_mfe_audit_only_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                "fifty_plus_mfe_risk_audit_only_v1": _as_bool(row.get("fifty_plus_mfe_risk_v1")),
                "hundred_plus_mfe_risk_audit_only_v1": _as_bool(row.get("hundred_plus_mfe_risk_v1")),
                "candidate_as_of_veto_can_block_it_v1": "signal-shape vetoes can block it but also block many good safe-core rows",
                "similar_unsafe_lookalikes_exist_v1": True,
                "blocks_good_rows_if_signal_shape_veto_used_v1": True,
                "final_classification_v1": "UNSAFE_ROW_REQUIRES_DEPLOYABLE_HARD_SAFETY_VETO_NOT_SIGNAL_SHAPE_SHORTCUT",
            }
        )
    return rows


def _candidate_veto_rows(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = masks["low_support_veto_only"]
    original = _bool(frame, "selected_original_140_v1")
    candidates = [
        {
            "candidate_veto_name_v1": "CURRENT_AUDIT_HARD_SAFETY_VETO_REFERENCE",
            "input_fields_v1": "unsafe/protected/runner/ambiguous/high-MFE/quarantine audit columns",
            "rule_condition_v1": "veto if audit safety is not clean",
            "as_of_lineage_v1": "NOT_PROVEN_AUDIT_ONLY",
            "adapter_feasibility_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "mask": base & ~masks["hardened"],
            "complexity_v1": "LOW_LOGIC_HIGH_LINEAGE_RISK",
            "recommendation_v1": "BLOCK_ADAPTER_UNTIL_DEPLOYABLE_AS_OF_MAPPING_EXISTS",
        },
        {
            "candidate_veto_name_v1": "AS_OF_SIGNAL_SHAPE_VETO_MISSING_R5_TAIL_AND_SCORE_GE_099",
            "input_fields_v1": "source_evidence_v1 contains no R5_TAIL_SCORE; candidate_score_v1 >= 0.99",
            "rule_condition_v1": "veto if missing R5 tail evidence and score >= 0.99",
            "as_of_lineage_v1": "AS_OF_SAFE_CANDIDATE_BUT_NOT_ACCEPTABLE",
            "adapter_feasibility_v1": "BLOCKS_TOO_MANY_GOOD_ROWS",
            "mask": base & masks["veto_missing_r5_tail_and_score_ge_099"],
            "complexity_v1": "LOW",
            "recommendation_v1": "REJECT_DUE_TO_SAFE_CORE_DAMAGE",
        },
        {
            "candidate_veto_name_v1": "AS_OF_SIGNAL_SHAPE_VETO_MISSING_R5_TAIL",
            "input_fields_v1": "source_evidence_v1 contains no R5_TAIL_SCORE",
            "rule_condition_v1": "veto if missing R5 tail evidence",
            "as_of_lineage_v1": "AS_OF_SAFE_CANDIDATE_BUT_NOT_ACCEPTABLE",
            "adapter_feasibility_v1": "BLOCKS_TOO_MANY_GOOD_ROWS",
            "mask": base & masks["veto_missing_r5_tail"],
            "complexity_v1": "LOW",
            "recommendation_v1": "REJECT_DUE_TO_SAFE_CORE_DAMAGE",
        },
        {
            "candidate_veto_name_v1": "ROW_IDENTITY_SPECIFIC_VETO_FORBIDDEN",
            "input_fields_v1": "candidate_uid_v1",
            "rule_condition_v1": "veto exact unsafe row identity",
            "as_of_lineage_v1": "BLOCKED_ROW_IDENTITY_SHORTCUT",
            "adapter_feasibility_v1": "FORBIDDEN",
            "mask": base & masks["veto_row_identity_specific_forbidden"],
            "complexity_v1": "LOW_BUT_FORBIDDEN",
            "recommendation_v1": "REJECT_ROW_IDENTITY_LEAKAGE",
        },
    ]
    rows = []
    for candidate in candidates:
        veto_mask = candidate.pop("mask")
        selected = base & ~veto_mask
        selected_frame = frame[selected]
        rows.append(
            {
                **candidate,
                "rows_blocked_v1": int(veto_mask.sum()),
                "unsafe_rows_blocked_v1": int((veto_mask & _bool(frame, "unsafe_audit_v1")).sum()),
                "original_140_rows_accidentally_blocked_v1": int((veto_mask & original).sum()),
                "safe_core_rows_accidentally_blocked_v1": int((veto_mask & masks["hardened"]).sum()),
                "remaining_extra_3_blocked_or_retained_v1": int((veto_mask & masks["hardened"] & ~original).sum()),
                "false_positive_risk_impact_v1": "NONE" if int((veto_mask & masks["hardened"] & ~original).sum()) == 0 else "BLOCKS_SOME_EXTRAS",
                "selected_rows_after_veto_v1": int(selected.sum()),
                "unsafe_rows_after_veto_v1": int(_bool(selected_frame, "unsafe_audit_v1").sum()),
                "bad_tail_after_veto_audit_only_v1": [
                    int(_bool(selected_frame, "bad_label_v1").sum()),
                    int(_bool(selected_frame, "tail_label_v1").sum()),
                ],
                "precision_after_veto_audit_only_v1": float(
                    _bool(selected_frame, "bad_label_v1").sum() / max(len(selected_frame), 1)
                ),
            }
        )
    validate_candidate_veto_rows(rows)
    return rows


def _remaining_extra_3(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "why_selected_v1": "score >= 0.95 + R5_1 + V2-like + current audit hard safety veto clear + no missing-artifact low-support veto",
                "safety_status_v1": "CLEAN" if not _as_bool(row.get("unsafe_audit_v1")) else "FAIL",
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "should_remain_allowed_v1": False,
                "should_be_blocked_by_false_positive_veto_v1": True,
                "blocking_harms_original_140_recovery_v1": False,
                "acceptable_adapter_extra_v1": False,
                "decision_v1": "REQUIRES_FALSE_POSITIVE_VETO_DECISION_AFTER_HARD_SAFETY_LINEAGE",
            }
        )
    summary = {
        "layer_name": "DEEPEN_140_94_REMAINING_EXTRA_3_VETO_DECISION_SUMMARY_V1",
        "remaining_extra_rows_v1": len(rows),
        "safety_status_v1": "CLEAN" if not any(row["safety_status_v1"] != "CLEAN" for row in rows) else "FAIL",
        "false_positive_veto_needed_v1": True,
        "hard_safety_lineage_is_primary_blocker_v1": True,
    }
    return rows, summary


def _dry_run_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidate_rows:
        mismatch = candidate["candidate_veto_name_v1"] != "CURRENT_AUDIT_HARD_SAFETY_VETO_REFERENCE"
        readiness = "NOT_READY"
        if candidate["candidate_veto_name_v1"] == "CURRENT_AUDIT_HARD_SAFETY_VETO_REFERENCE":
            readiness = "EXACT_MATCH_BUT_AUDIT_ONLY_NOT_DEPLOYABLE"
        elif candidate["candidate_veto_name_v1"] == "ROW_IDENTITY_SPECIFIC_VETO_FORBIDDEN":
            readiness = "FORBIDDEN_ROW_IDENTITY_SHORTCUT"
        elif candidate["unsafe_rows_after_veto_v1"] == 0:
            readiness = "SAFETY_CLEAN_BUT_DAMAGES_SAFE_CORE"
        rows.append(
            {
                "candidate_veto_name_v1": candidate["candidate_veto_name_v1"],
                "selected_rows_v1": candidate["selected_rows_after_veto_v1"],
                "recovered_safe_core_rows_v1": 89 - candidate["safe_core_rows_accidentally_blocked_v1"],
                "recovered_original_140_rows_v1": 86 - candidate["original_140_rows_accidentally_blocked_v1"],
                "extra_rows_v1": candidate["selected_rows_after_veto_v1"] - (86 - candidate["original_140_rows_accidentally_blocked_v1"]),
                "unsafe_rows_v1": candidate["unsafe_rows_after_veto_v1"],
                "bad_tail_audit_only_v1": candidate["bad_tail_after_veto_audit_only_v1"],
                "precision_audit_only_v1": candidate["precision_after_veto_audit_only_v1"],
                "safety_status_v1": "CLEAN" if candidate["unsafe_rows_after_veto_v1"] == 0 else "FAIL",
                "mismatch_vs_current_safe_core_v1": mismatch,
                "adapter_readiness_v1": readiness,
            }
        )
    return rows


def _hard_safety_blocker_audit(unsafe_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "layer_name": "DEEPEN_140_94_HARD_SAFETY_VETO_BLOCKER_AUDIT_V1",
        "primary_blocker_v1": "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO",
        "what_veto_stops_v1": "one unsafe extra row that passes score/R5_1/V2-like positive rules and low-support veto",
        "unsafe_row_count_without_veto_v1": len(unsafe_rows),
        "unsafe_rows_v1": unsafe_rows,
        "signals_that_selected_unsafe_row_v1": "score >= 0.95, R5_1_BAD_SCORE support, V2_LIKE_BAD_TAIL strong, low-support missing-artifact veto clear",
        "as_of_signal_shape_candidates_found_v1": [
            row["candidate_veto_name_v1"]
            for row in candidate_rows
            if row["candidate_veto_name_v1"].startswith("AS_OF_SIGNAL_SHAPE")
        ],
        "as_of_signal_shape_candidates_acceptable_v1": False,
        "reason_signal_shape_candidates_rejected_v1": "They block the unsafe row only by also blocking many original/hardened safe-core rows.",
        "can_current_hard_veto_be_computed_before_outcome_v1": False,
        "artifact_audit_hindsight_dependency_v1": True,
        "required_for_adapter_use_v1": "Map protected/ambiguous/high-MFE/quarantine risk to independently proven AS_OF-safe veto inputs, or hold adapter.",
        "status_v1": "AUDIT_ONLY_NOT_DEPLOYABLE",
    }


def _final_decision(
    candidate_rows: list[dict[str, Any]],
    extra_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    decisions = {
        "layer_name": "DEEPEN_140_94_FINAL_VETO_DECISION_V1",
        "hard_safety_veto_final_status_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "low_support_missing_artifact_veto_status_v1": "NEEDS_NORMALIZATION",
        "false_positive_extra_3_veto_status_v1": "NEEDS_FALSE_POSITIVE_DECISION_AFTER_HARD_SAFETY",
        "candidate_veto_decisions_v1": {
            row["candidate_veto_name_v1"]: (
                "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
                if row["candidate_veto_name_v1"] == "CURRENT_AUDIT_HARD_SAFETY_VETO_REFERENCE"
                else "BLOCKED_ROW_IDENTITY_LEAKAGE"
                if row["candidate_veto_name_v1"] == "ROW_IDENTITY_SPECIFIC_VETO_FORBIDDEN"
                else "BLOCKED_DAMAGES_SAFE_CORE"
            )
            for row in candidate_rows
        },
        "adapter_build_can_start_next_v1": False,
    }
    validate_final_veto_decision(decisions)
    readiness = {
        "layer_name": "DEEPEN_140_94_ADAPTER_READINESS_AFTER_VETO_MAPPING_V1",
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "adapter_build_can_start_next_v1": False,
        "hard_safety_veto_still_audit_only_v1": True,
        "normalization_needed_v1": True,
        "false_positive_veto_needed_v1": extra_summary["false_positive_veto_needed_v1"],
        "return_to_hardening_recommended_v1": False,
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
    }
    anti = {
        "layer_name": "DEEPEN_140_94_VETO_MAPPING_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS_AUDIT_ONLY",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_implicit_latest_glob_selection_v1": True,
        "row_identity_veto_rejected_v1": True,
        "labels_mfe_safe_recoverable_blocked_as_features_v1": True,
        "membership_coverage_selected_flags_blocked_v1": True,
        "dummy_synthetic_fallback_v1": False,
    }
    recommendation = {
        "layer_name": "DEEPEN_140_94_VETO_MAPPING_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "rationale_v1": [
            "The current audit hard safety veto is the only tested veto that exactly preserves the 89-row safe-core and blocks the unsafe extra.",
            "Signal-shape AS_OF candidates block the unsafe row but also remove too many original/hardened safe-core rows.",
            "A row-id-specific veto would be a forbidden shortcut.",
            "Hold adapter build until a deployable AS_OF hard safety veto exists.",
        ],
    }
    go_no_go = {
        "layer_name": "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": EXPECTED_SELECTED,
        "recovered_original_140_rows_v1": EXPECTED_RECOVERED,
        "extra_rows_v1": EXPECTED_EXTRA,
        "bad_tail_audit_only_v1": [EXPECTED_BAD, EXPECTED_TAIL],
        "precision_audit_only_v1": EXPECTED_PRECISION,
        "safety_status_v1": "CLEAN",
        "hard_safety_veto_status_v1": decisions["hard_safety_veto_final_status_v1"],
        "adapter_build_approved_next_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])
    return decisions, readiness, anti, recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    unmapped_summary: dict[str, Any],
    hard_blocker: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    unsafe_rows: list[dict[str, Any]],
    extra_summary: dict[str, Any],
    dry_rows: list[dict[str, Any]],
    final_decision: dict[str, Any],
    readiness: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "deepen_140_94_veto_mapping_reproducibility_audit_v1.md",
        [
            "# Deepen 140/94 Veto Mapping Reproducibility Audit V1",
            "",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['extra_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_audit_only_v1']} / {repro['tail_count_audit_only_v1']}`",
            f"- Without hard safety veto unsafe extras: `{repro['without_hard_safety_veto_unsafe_extra_rows_v1']}`",
        ],
    )
    _write_report(
        root / "deepen_140_94_unmapped_veto_fields_audit_v1.md",
        [
            "# Deepen 140/94 Unmapped Veto Fields Audit V1",
            "",
            f"- Unmapped veto fields: `{unmapped_summary['unmapped_veto_field_count_v1']}`",
            f"- Audit-only fields: `{unmapped_summary['audit_only_field_count_v1']}`",
            f"- Primary blocker: `{unmapped_summary['primary_blocker_v1']}`",
        ],
    )
    _write_report(
        root / "deepen_140_94_hard_safety_veto_blocker_audit_v1.md",
        [
            "# Deepen 140/94 Hard Safety Veto Blocker Audit V1",
            "",
            f"- Status: `{hard_blocker['status_v1']}`",
            f"- Unsafe rows without veto: `{hard_blocker['unsafe_row_count_without_veto_v1']}`",
            f"- Current hard veto computable before outcome: `{hard_blocker['can_current_hard_veto_be_computed_before_outcome_v1']}`",
        ],
    )
    _write_report(
        root / "deepen_140_94_candidate_as_of_veto_mappings_v1.md",
        [
            "# Deepen 140/94 Candidate AS_OF Veto Mappings V1",
            "",
            *[
                f"- `{row['candidate_veto_name_v1']}`: blocks `{row['rows_blocked_v1']}`, unsafe blocked `{row['unsafe_rows_blocked_v1']}`, readiness `{row['adapter_feasibility_v1']}`"
                for row in candidate_rows
            ],
        ],
    )
    _write_report(
        root / "deepen_140_94_unsafe_extra_without_veto_audit_v1.md",
        [
            "# Deepen 140/94 Unsafe Extra Without Veto Audit V1",
            "",
            f"- Unsafe rows: `{len(unsafe_rows)}`",
            "- The unsafe row is selected by the positive branch and requires a deployable hard safety veto, not a signal-shape shortcut.",
        ],
    )
    _write_report(
        root / "deepen_140_94_remaining_extra_3_veto_decision_v1.md",
        [
            "# Deepen 140/94 Remaining Extra 3 Veto Decision V1",
            "",
            f"- Remaining extra rows: `{extra_summary['remaining_extra_rows_v1']}`",
            f"- Safety: `{extra_summary['safety_status_v1']}`",
            "- They require a later false-positive veto decision, but hard safety lineage is still the primary blocker.",
        ],
    )
    _write_report(
        root / "deepen_140_94_candidate_veto_dry_run_results_v1.md",
        [
            "# Deepen 140/94 Candidate Veto Dry Run Results V1",
            "",
            *[
                f"- `{row['candidate_veto_name_v1']}`: selected `{row['selected_rows_v1']}`, unsafe `{row['unsafe_rows_v1']}`, readiness `{row['adapter_readiness_v1']}`"
                for row in dry_rows
            ],
        ],
    )
    _write_report(
        root / "deepen_140_94_final_veto_decision_v1.md",
        [
            "# Deepen 140/94 Final Veto Decision V1",
            "",
            f"- Hard safety veto: `{final_decision['hard_safety_veto_final_status_v1']}`",
            f"- Adapter build can start next: `{final_decision['adapter_build_can_start_next_v1']}`",
        ],
    )
    _write_report(
        root / "deepen_140_94_adapter_readiness_after_veto_mapping_v1.md",
        [
            "# Deepen 140/94 Adapter Readiness After Veto Mapping V1",
            "",
            f"- Status: `{readiness['status_v1']}`",
            f"- Adapter build can start next: `{readiness['adapter_build_can_start_next_v1']}`",
            f"- Hard safety veto still audit-only: `{readiness['hard_safety_veto_still_audit_only_v1']}`",
        ],
    )
    _write_report(
        root / "deepen_140_94_veto_mapping_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Deepen 140/94 Veto Mapping Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- No R6, adapter, package, freeze, promo, live, Optuna, broad sweep, or in-sample decisioning was run.",
            "- Row-identity veto was explicitly rejected; audit-only fields remain non-deployable.",
        ],
    )
    _write_report(
        root / "deepen_140_94_veto_mapping_recommendation_v1.md",
        [
            "# Deepen 140/94 Veto Mapping Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            "- Hold adapter build until a deployable AS_OF hard safety veto exists.",
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
    unmapped_rows, unmapped_summary = _unmapped_veto_fields(inputs)
    unsafe_rows = _unsafe_extra_rows(frame, masks)
    candidate_rows = _candidate_veto_rows(frame, masks)
    hard_blocker = _hard_safety_blocker_audit(unsafe_rows, candidate_rows)
    extra_rows, extra_summary = _remaining_extra_3(frame, masks)
    dry_rows = _dry_run_rows(candidate_rows)
    final_decision, readiness, anti, recommendation, go_no_go = _final_decision(candidate_rows, extra_summary)

    _write_json(artifact_root / "deepen_140_94_veto_mapping_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "deepen_140_94_veto_mapping_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "deepen_140_94_unmapped_veto_fields_audit_v1.csv", unmapped_rows)
    _write_json(
        artifact_root / "deepen_140_94_unmapped_veto_fields_audit_v1.json",
        {"summary_v1": unmapped_summary, "rows_v1": unmapped_rows},
    )
    _write_json(artifact_root / "deepen_140_94_hard_safety_veto_blocker_audit_v1.json", hard_blocker)
    _write_rows(artifact_root / "deepen_140_94_candidate_as_of_veto_mappings_v1.csv", candidate_rows)
    _write_json(
        artifact_root / "deepen_140_94_candidate_as_of_veto_mappings_v1.json",
        {"row_count_v1": len(candidate_rows), "rows_v1": candidate_rows},
    )
    _write_rows(artifact_root / "deepen_140_94_unsafe_extra_without_veto_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "deepen_140_94_unsafe_extra_without_veto_audit_v1.json",
        {"row_count_v1": len(unsafe_rows), "rows_v1": unsafe_rows},
    )
    _write_rows(artifact_root / "deepen_140_94_remaining_extra_3_veto_decision_v1.csv", extra_rows)
    _write_json(
        artifact_root / "deepen_140_94_remaining_extra_3_veto_decision_v1.json",
        {"summary_v1": extra_summary, "rows_v1": extra_rows},
    )
    _write_rows(artifact_root / "deepen_140_94_candidate_veto_dry_run_results_v1.csv", dry_rows)
    _write_json(
        artifact_root / "deepen_140_94_candidate_veto_dry_run_results_v1.json",
        {"row_count_v1": len(dry_rows), "rows_v1": dry_rows},
    )
    _write_json(artifact_root / "deepen_140_94_final_veto_decision_v1.json", final_decision)
    _write_json(artifact_root / "deepen_140_94_adapter_readiness_after_veto_mapping_v1.json", readiness)
    _write_json(artifact_root / "deepen_140_94_veto_mapping_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "deepen_140_94_veto_mapping_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "deepen_140_94_safe_core_veto_mapping_audit_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": repro["selected_rows_v1"],
        "recovered_original_140_rows_v1": repro["recovered_original_140_rows_v1"],
        "extra_rows_v1": repro["extra_rows_v1"],
        "bad_tail_audit_only_v1": [repro["bad_count_audit_only_v1"], repro["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": repro["precision_audit_only_v1"],
        "safety_status_v1": repro["safety_status_v1"],
        "hard_safety_veto_mapping_result_v1": final_decision["hard_safety_veto_final_status_v1"],
        "unsafe_extra_without_hard_veto_rows_v1": repro["without_hard_safety_veto_unsafe_extra_rows_v1"],
        "remaining_extra_3_decision_v1": final_decision["false_positive_extra_3_veto_status_v1"],
        "adapter_build_can_start_next_v1": False,
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
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
            "# Deepen 140/94 Safe-Core Veto Mapping Audit V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Safe-core rule: `{HARDENED_RECIPE_ID}`",
            f"- Hard safety veto mapping: `{summary['hard_safety_veto_mapping_result_v1']}`",
            f"- Unsafe extras without hard veto: `{summary['unsafe_extra_without_hard_veto_rows_v1']}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        unmapped_summary,
        hard_blocker,
        candidate_rows,
        unsafe_rows,
        extra_summary,
        dry_rows,
        final_decision,
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
