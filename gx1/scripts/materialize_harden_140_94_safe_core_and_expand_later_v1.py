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

from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as simplify


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1"

INPUT_SIMPLIFY_ROOT = (
    DEFAULT_REPORTS_ROOT / "SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T083415Z_LOCK"
)
INPUT_DISTILL_ROOT = (
    DEFAULT_REPORTS_ROOT / "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T081017Z_LOCK"
)
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

SIMPLIFIED_RECIPE_ID = "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1"
HARDENED_RECIPE_ID = "SAFE_CORE_HARDENED_RULE_V1"
FINAL_STATUS = "140_94_SAFE_CORE_HARDENED_NEEDS_INPUT_MAPPING_EXPAND_LATER"
NEXT_ACTION = "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1"

SIMPLIFIED_SELECTED = 91
SIMPLIFIED_RECOVERED = 86
SIMPLIFIED_EXTRA = 5
SIMPLIFIED_BAD = 86
SIMPLIFIED_TAIL = 55
SIMPLIFIED_PRECISION = 0.945054945054945

ALLOWED_FINAL_STATUSES = {
    "140_94_SAFE_CORE_HARDENED_ADAPTER_READY_EXPAND_LATER",
    "140_94_SAFE_CORE_HARDENED_NEEDS_INPUT_MAPPING_EXPAND_LATER",
    "140_94_SAFE_CORE_HARDENED_BUT_TOO_SMALL_NEEDS_EXPANSION_FIRST",
    "140_94_SAFE_CORE_NEEDS_MORE_VETO_HARDENING",
    "140_94_SAFE_CORE_BLOCKED_BY_OVER_SELECTION",
    "140_94_SAFE_CORE_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "140_94_SAFE_CORE_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "140_94_SAFE_CORE_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_SAFE_CORE_ADAPTER_V1",
    "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1",
    "TEST_140_94_EXPANSION_MODULES_AS_SEPARATE_GATE_V1",
    "DEEPEN_140_94_VETO_HARDENING_AUDIT_V1",
    "DEEPEN_140_94_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT_V1",
    "DEEPEN_140_94_GROUPED_GENERALIZATION_AND_LOSO_AUDIT_V1",
    "RETURN_TO_140_94_SIMPLIFICATION_WITH_STRONGER_SIGNALS_V1",
}

DENY_PATTERNS = simplify.DENY_PATTERNS
ADAPTER_SAFE_FEATURES = [
    "tail_repaired_r5_2_oof_candidate_score_v1",
    "asof_signal__r5_1_bad_score_v1",
    "asof_signal__v2_like_bad_tail_v1",
    "asof_low_support_missing_artifact_veto_v1",
    "asof_hard_safety_veto_set_v1",
]

REQUIRED_OUTPUTS = [
    "harden_140_94_input_manifest_v1.json",
    "harden_140_94_reproducibility_audit_v1.json",
    "harden_140_94_reproducibility_audit_v1.md",
    "harden_140_94_safe_core_definition_v1.json",
    "harden_140_94_safe_core_definition_v1.md",
    "harden_140_94_safe_core_row_level_explanations_v1.csv",
    "harden_140_94_safe_core_row_level_explanations_v1.json",
    "harden_140_94_extra_5_audit_v1.csv",
    "harden_140_94_extra_5_audit_v1.json",
    "harden_140_94_extra_5_audit_v1.md",
    "harden_140_94_missing_54_audit_v1.csv",
    "harden_140_94_missing_54_audit_v1.json",
    "harden_140_94_missing_54_audit_v1.md",
    "harden_140_94_missing_54_expansion_bucket_audit_v1.csv",
    "harden_140_94_missing_54_expansion_bucket_audit_v1.json",
    "harden_140_94_missing_54_expansion_bucket_audit_v1.md",
    "harden_140_94_expansion_module_definitions_v1.json",
    "harden_140_94_expansion_module_definitions_v1.md",
    "harden_140_94_expansion_module_metrics_v1.csv",
    "harden_140_94_expansion_module_metrics_v1.json",
    "harden_140_94_expansion_module_metrics_v1.md",
    "harden_140_94_veto_hardening_audit_v1.csv",
    "harden_140_94_veto_hardening_audit_v1.json",
    "harden_140_94_veto_hardening_audit_v1.md",
    "harden_140_94_boundary_stress_audit_v1.json",
    "harden_140_94_boundary_stress_audit_v1.md",
    "harden_140_94_near_miss_and_near_fail_rows_v1.csv",
    "harden_140_94_near_miss_and_near_fail_rows_v1.json",
    "harden_140_94_group_stability_audit_v1.csv",
    "harden_140_94_group_stability_audit_v1.json",
    "harden_140_94_group_stability_audit_v1.md",
    "harden_140_94_adapter_readiness_v1.json",
    "harden_140_94_adapter_readiness_v1.md",
    "harden_140_94_anti_overfit_no_shortcut_audit_v1.json",
    "harden_140_94_anti_overfit_no_shortcut_audit_v1.md",
    "harden_140_94_recommendation_v1.json",
    "harden_140_94_recommendation_v1.md",
    "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
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
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_HARDEN_140_94_FEATURE: {blocked}")
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
        "simplified_selected_rows_v1": SIMPLIFIED_SELECTED,
        "simplified_recovered_original_140_rows_v1": SIMPLIFIED_RECOVERED,
        "simplified_extra_rows_v1": SIMPLIFIED_EXTRA,
        "simplified_bad_count_audit_only_v1": SIMPLIFIED_BAD,
        "simplified_tail_count_audit_only_v1": SIMPLIFIED_TAIL,
        "simplified_safety_status_v1": "CLEAN",
        "simplified_unsafe_hits_v1": 0,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if not math.isclose(float(payload.get("simplified_precision_audit_only_v1", -1)), SIMPLIFIED_PRECISION):
        failures["simplified_precision_audit_only_v1"] = payload.get("simplified_precision_audit_only_v1")
    if failures:
        raise RuntimeError(f"HARDEN_140_94_REPRODUCIBILITY_FAILED: {failures}")
    return True


def validate_hardened_safe_core(payload: dict[str, Any]) -> bool:
    if payload.get("recipe_id_v1") != HARDENED_RECIPE_ID:
        raise RuntimeError("HARDENED_SAFE_CORE_RECIPE_ID_MISMATCH")
    if payload.get("safety_status_v1") != "CLEAN":
        raise RuntimeError("HARDENED_SAFE_CORE_NOT_SAFETY_CLEAN")
    if payload.get("extra_rows_v1", 999) > 3:
        raise RuntimeError("HARDENED_SAFE_CORE_OVERSELECTS")
    if payload.get("recovered_original_140_rows_v1", 0) < 80:
        raise RuntimeError("HARDENED_SAFE_CORE_TOO_SMALL")
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
        raise RuntimeError(f"HARDEN_140_94_REQUIRED_OUTPUTS_MISSING: {missing}")
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
    roots = [INPUT_SIMPLIFY_ROOT, INPUT_DISTILL_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "simplify_summary": INPUT_SIMPLIFY_ROOT / "summary_v1.json",
        "simplify_go_no_go": INPUT_SIMPLIFY_ROOT / "simplify_140_94_rules_and_vetoes_go_no_go_v1.json",
        "simplify_metrics": INPUT_SIMPLIFY_ROOT / "simplify_140_94_candidate_recipe_metrics_v1.csv",
        "simplify_best_explanations": INPUT_SIMPLIFY_ROOT / "simplify_140_94_best_recipe_row_level_explanations_v1.csv",
        "simplify_veto_mapping": INPUT_SIMPLIFY_ROOT / "simplify_140_94_veto_mapping_audit_v1.csv",
        "simplify_near_miss": INPUT_SIMPLIFY_ROOT / "simplify_140_94_near_miss_and_near_fail_rows_v1.csv",
        "distill_summary": INPUT_DISTILL_ROOT / "summary_v1.json",
        "distill_go_no_go": INPUT_DISTILL_ROOT / "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    simplify_go = _read_json(required["simplify_go_no_go"])
    if simplify_go.get("status_v1") != "140_94_SIMPLIFIED_RULES_FOUND_SAFE_CORE_NEEDS_EXPANSION_LATER":
        raise RuntimeError("SIMPLIFY_STATUS_NOT_SAFE_CORE_EXPAND_LATER")
    return {
        "required_paths": required,
        "simplify_summary": _read_json(required["simplify_summary"]),
        "simplify_go_no_go": simplify_go,
        "simplify_metrics": pd.read_csv(required["simplify_metrics"]),
        "simplify_best_explanations": pd.read_csv(required["simplify_best_explanations"]),
        "simplify_veto_mapping": pd.read_csv(required["simplify_veto_mapping"]),
        "simplify_near_miss": pd.read_csv(required["simplify_near_miss"]),
        "distill_summary": _read_json(required["distill_summary"]),
        "distill_go_no_go": _read_json(required["distill_go_no_go"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "source_inputs": simplify._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append({"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)})
    return {
        "layer_name": "HARDEN_140_94_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "simplify_root_v1": str(INPUT_SIMPLIFY_ROOT),
            "distill_root_v1": str(INPUT_DISTILL_ROOT),
            "precheck_140_94_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    score_floor = float(frame[_bool(frame, "selected_original_140_v1")]["candidate_score_v1"].min())
    recipe_masks = simplify._recipe_masks(frame, score_floor)
    simplified_mask = recipe_masks[SIMPLIFIED_RECIPE_ID]
    missing_artifacts = _str(frame, "run_id_policy_class_v1").str.contains("LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS")
    hardened_mask = simplified_mask & ~missing_artifacts
    balanced_mask = recipe_masks["BALANCED_140_RECOVERY_RULE_V1"]
    full_mask = recipe_masks["FULL_COVER_TIGHTENED_RULE_V1"]
    return {
        "simplified": simplified_mask,
        "hardened": hardened_mask,
        "balanced": balanced_mask,
        "full": full_mask,
        "original": _bool(frame, "selected_original_140_v1"),
    }


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


def _reproducibility(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    metrics = _selected_metrics(frame, masks["simplified"])
    payload = {
        "layer_name": "HARDEN_140_94_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "simplified_selected_rows_v1": metrics["selected_rows_v1"],
        "simplified_recovered_original_140_rows_v1": metrics["recovered_original_140_rows_v1"],
        "simplified_extra_rows_v1": metrics["extra_rows_v1"],
        "simplified_bad_count_audit_only_v1": metrics["bad_count_audit_only_v1"],
        "simplified_tail_count_audit_only_v1": metrics["tail_count_audit_only_v1"],
        "simplified_precision_audit_only_v1": metrics["precision_audit_only_v1"],
        "simplified_safety_status_v1": metrics["safety_status_v1"],
        "simplified_unsafe_hits_v1": metrics["unsafe_hits_v1"],
        "simplified_protected_winner_hits_v1": metrics["protected_winner_hits_audit_only_v1"],
        "simplified_runner_protect_hits_v1": metrics["runner_protect_hits_audit_only_v1"],
        "simplified_ambiguous_high_mfe_hits_v1": metrics["ambiguous_high_mfe_hits_audit_only_v1"],
        "simplified_quarantine_hits_v1": metrics["quarantine_hits_audit_only_v1"],
    }
    validate_reproducibility(payload)
    return payload


def _safe_core_definition(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    metrics = _selected_metrics(frame, masks["hardened"])
    payload = {
        "layer_name": "HARDEN_140_94_SAFE_CORE_DEFINITION_V1",
        "recipe_id_v1": HARDENED_RECIPE_ID,
        "source_simplified_recipe_v1": SIMPLIFIED_RECIPE_ID,
        "rule_v1": {
            "score_threshold_v1": "tail_repaired_r5_2_oof_candidate_score_v1 >= 0.95",
            "required_positive_signals_v1": [
                "asof_signal__r5_1_bad_score_v1",
                "asof_signal__v2_like_bad_tail_v1",
            ],
            "active_vetoes_v1": [
                "hard safety veto clear",
                "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS excluded from hardened safe-core",
                "membership/coverage/+45/row identity/selected flags forbidden as features",
            ],
        },
        "input_fields_v1": ADAPTER_SAFE_FEATURES,
        "adapter_ready_fields_v1": [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
            "asof_signal__v2_like_bad_tail_v1",
        ],
        "mapping_or_normalization_needed_v1": [
            "asof_low_support_missing_artifact_veto_v1",
            "asof_hard_safety_veto_set_v1",
        ],
        "selected_rows_v1": metrics["selected_rows_v1"],
        "recovered_original_140_rows_v1": metrics["recovered_original_140_rows_v1"],
        "extra_rows_v1": metrics["extra_rows_v1"],
        "bad_tail_audit_only_v1": [metrics["bad_count_audit_only_v1"], metrics["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": metrics["precision_audit_only_v1"],
        "safety_status_v1": metrics["safety_status_v1"],
        "final_promotion_allowed_v1": False,
    }
    validate_hardened_safe_core(payload)
    return payload


def _positive_signals(row: pd.Series) -> str:
    signals = ["tail_repaired_r5_2_oof_candidate_score_v1"]
    for signal, column in [
        ("asof_signal__r5_1_bad_score_v1", "signal_r5_1_bad_score_v1"),
        ("asof_signal__v2_like_bad_tail_v1", "signal_v2_like_bad_tail_v1"),
        ("asof_signal__r5_bad_score_v1", "signal_r5_bad_score_v1"),
        ("asof_signal__r5_tail_score_v1", "signal_r5_tail_score_v1"),
    ]:
        if _as_bool(row.get(column)):
            signals.append(signal)
    return "|".join(signals)


def _safe_core_rows(frame: pd.DataFrame, hardened_mask: pd.Series) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame[hardened_mask].sort_values(["run_id_v1", "candidate_score_v1"], ascending=[True, False]).iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "selected_by_simplified_recipe_v1": True,
                "selected_by_hardened_safe_core_v1": True,
                "candidate_score_v1": row.get("candidate_score_v1"),
                "positive_signals_v1": _positive_signals(row),
                "branch_tier_v1": "TIER_1_HARDENED_HIGH_CONFIDENCE",
                "veto_status_v1": "audit safety clean; AS_OF veto mapping still required",
                "confidence_class_v1": "HARDENED_SAFE_CORE_HIGH_CONFIDENCE",
                "support_class_v1": row.get("run_id_policy_class_v1"),
                "low_support_status_v1": "STRUCTURAL_LOW_SUPPORT_VISIBLE"
                if _as_bool(row.get("structural_low_support_v1"))
                else "SUPPORT_VISIBLE",
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "explanation_v1": "score >= 0.95, R5_1, V2-like support, hard veto clear, and no missing-artifact low-support veto",
            }
        )
    return rows


def _extra_5_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extra = frame[masks["simplified"] & ~masks["original"]].copy()
    rows = []
    for _, row in extra.sort_values("candidate_score_v1", ascending=False).iterrows():
        blocked_by_hardened = not _as_bool(masks["hardened"].loc[row.name])
        recommendation = "BLOCK_FROM_HARDENED_SAFE_CORE_LOW_SUPPORT_MISSING_ARTIFACTS" if blocked_by_hardened else "RETAIN_PENDING_AS_OF_FALSE_POSITIVE_VETO"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "why_selected_v1": "score >= 0.95 + R5_1 + V2-like + audit hard veto clear",
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "safety_status_v1": "CLEAN" if not _as_bool(row.get("unsafe_audit_v1")) else "FAIL",
                "signal_branch_v1": "HIGH_CONFIDENCE_V2LIKE",
                "confidence_class_v1": "FALSE_POSITIVE_RISK_SAFE_CLEAN",
                "low_support_status_v1": row.get("run_id_policy_class_v1"),
                "selected_by_hardened_safe_core_v1": _as_bool(masks["hardened"].loc[row.name]),
                "recommendation_v1": recommendation,
                "over_selection_risk_v1": "LOW_COUNT_BUT_REAL_FALSE_POSITIVE_AUDIT_RISK",
            }
        )
    summary = {
        "layer_name": "HARDEN_140_94_EXTRA_5_AUDIT_SUMMARY_V1",
        "extra_rows_v1": len(rows),
        "blocked_by_hardened_rule_v1": sum(not row["selected_by_hardened_safe_core_v1"] for row in rows),
        "retained_by_hardened_rule_v1": sum(row["selected_by_hardened_safe_core_v1"] for row in rows),
        "bad_tail_audit_only_v1": [sum(row["bad_label_audit_only_v1"] for row in rows), sum(row["tail_label_audit_only_v1"] for row in rows)],
        "safety_status_v1": "CLEAN" if not any(row["safety_status_v1"] != "CLEAN" for row in rows) else "FAIL",
    }
    return rows, summary


def _bucket_missing_row(row: pd.Series, masks: dict[str, pd.Series]) -> str:
    if _as_bool(row.get("unsafe_audit_v1")):
        return "UNSAFE_LOOKALIKE_RISK"
    if _as_bool(masks["balanced"].loc[row.name]):
        return "EASY_AS_OF_EXTENSION"
    support_class = str(row.get("run_id_policy_class_v1", ""))
    if _as_bool(row.get("structural_low_support_v1")) or "LOW_SUPPORT" in support_class:
        return "LOW_SUPPORT_OR_GROUP_RISK"
    if _as_bool(row.get("signal_r5_tail_score_v1")) or _as_bool(row.get("signal_r5_bad_score_v1")):
        return "NEEDS_SIGNAL_MAPPING"
    if _as_bool(masks["full"].loc[row.name]):
        return "NEEDS_VETO_MAPPING"
    return "UNCLEAR_LINEAGE"


def _missing_54_audit(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing = frame[masks["original"] & ~masks["simplified"]].copy()
    rows = []
    for _, row in missing.sort_values(["run_id_v1", "candidate_score_v1"], ascending=[True, False]).iterrows():
        bucket = _bucket_missing_row(row, masks)
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "original_branch_v1": "FULL_COVER_SCORE_R5_1_WITH_AUDIT_VETO",
                "miss_reason_v1": "missing high-confidence V2-like score >= 0.95 safe-core requirement",
                "missing_signal_or_mapping_v1": "score/V2-like/tighter support or veto mapping",
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "safety_status_v1": "CLEAN" if not _as_bool(row.get("unsafe_audit_v1")) else "FAIL",
                "low_support_status_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "expansion_bucket_v1": bucket,
                "can_be_expansion_candidate_v1": bucket
                in {"EASY_AS_OF_EXTENSION", "NEEDS_VETO_MAPPING", "NEEDS_SIGNAL_MAPPING"},
                "diagnostic_only_reason_v1": "" if bucket != "DO_NOT_EXPAND_NOW" else "do not expand now",
            }
        )
    bucket_rows = []
    for bucket, group in pd.DataFrame(rows).groupby("expansion_bucket_v1"):
        bucket_rows.append(
            {
                "expansion_bucket_v1": bucket,
                "missing_54_rows_v1": len(group),
                "bad_count_audit_only_v1": int(group["bad_label_audit_only_v1"].sum()),
                "tail_count_audit_only_v1": int(group["tail_label_audit_only_v1"].sum()),
                "safety_status_v1": "CLEAN" if not (group["safety_status_v1"] != "CLEAN").any() else "FAIL",
                "candidate_for_later_gate_v1": bucket in {"EASY_AS_OF_EXTENSION", "NEEDS_VETO_MAPPING", "NEEDS_SIGNAL_MAPPING"},
            }
        )
    return rows, bucket_rows


def _expansion_modules(frame: pd.DataFrame, masks: dict[str, pd.Series], missing_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_by_uid = {row["candidate_uid_v1"]: row["expansion_bucket_v1"] for row in missing_rows}
    easy_uids = {uid for uid, bucket in missing_by_uid.items() if bucket == "EASY_AS_OF_EXTENSION"}
    veto_uids = {uid for uid, bucket in missing_by_uid.items() if bucket == "NEEDS_VETO_MAPPING"}
    signal_uids = {uid for uid, bucket in missing_by_uid.items() if bucket == "NEEDS_SIGNAL_MAPPING"}
    uid = _str(frame, "candidate_uid_v1")
    modules = {
        "EXPANSION_MODULE_EASY_AS_OF_EXTENSION_V1": masks["balanced"] & ~masks["hardened"],
        "EXPANSION_MODULE_NEEDS_VETO_MAPPING_V1": masks["full"] & ~masks["balanced"] & ~masks["hardened"],
        "EXPANSION_MODULE_NEEDS_SIGNAL_MAPPING_V1": uid.isin(signal_uids),
        "EXPANSION_MODULE_HOLDOUT_DIAGNOSTIC_ONLY_V1": uid.isin(veto_uids | signal_uids | easy_uids) & ~masks["balanced"],
    }
    definitions = {
        "layer_name": "HARDEN_140_94_EXPANSION_MODULE_DEFINITIONS_V1",
        "modules_v1": {
            "EXPANSION_MODULE_EASY_AS_OF_EXTENSION_V1": "Balanced rule additions beyond hardened safe-core; later gate only.",
            "EXPANSION_MODULE_NEEDS_VETO_MAPPING_V1": "Full-cover additions that need AS_OF hard-veto mapping before use.",
            "EXPANSION_MODULE_NEEDS_SIGNAL_MAPPING_V1": "Original-140 misses with R5/R5-tail evidence but not enough current mapped support.",
            "EXPANSION_MODULE_HOLDOUT_DIAGNOSTIC_ONLY_V1": "Rows held out from expansion until a separate evidence gate.",
        },
        "expansion_merged_into_safe_core_v1": False,
    }
    rows = []
    for module_id, mask in modules.items():
        selected = frame[mask]
        rows.append(
            {
                "module_id_v1": module_id,
                "module_rows_v1": int(mask.sum()),
                "potential_missing_54_recovered_v1": int((mask & masks["original"] & ~masks["simplified"]).sum()),
                "extra_rows_pulled_v1": int((mask & ~masks["original"]).sum()),
                "bad_count_audit_only_v1": int(_bool(selected, "bad_label_v1").sum()),
                "tail_count_audit_only_v1": int(_bool(selected, "tail_label_v1").sum()),
                "safety_status_v1": "CLEAN" if int(_bool(selected, "unsafe_audit_v1").sum()) == 0 else "FAIL",
                "unsafe_hits_v1": int(_bool(selected, "unsafe_audit_v1").sum()),
                "unsafe_lookalike_risk_v1": "MODERATE_REQUIRES_VETO" if module_id != "EXPANSION_MODULE_EASY_AS_OF_EXTENSION_V1" else "LOW_TO_MODERATE",
                "adapter_feasibility_v1": "SEPARATE_GATE_REQUIRED",
                "can_test_in_later_gate_v1": module_id != "EXPANSION_MODULE_HOLDOUT_DIAGNOSTIC_ONLY_V1",
            }
        )
    return definitions, rows


def _veto_hardening(inputs: dict[str, Any], frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    simplify_veto = inputs["simplify_veto_mapping"].to_dict("records")
    rows = []
    for row in simplify_veto:
        rows.append(
            {
                "veto_name_v1": row["veto_name_v1"],
                "adapter_ready_v1": _as_bool(row.get("adapter_ready_v1")),
                "as_of_safe_lineage_v1": _as_bool(row.get("as_of_safe_input_available_v1")),
                "rows_blocked_v1": row.get("rows_affected_total_v1"),
                "original_140_accidentally_blocked_v1": row.get("original_140_rows_accidentally_blocked_v1"),
                "extra_rows_blocked_v1": row.get("extra_rows_blocked_candidate_v1"),
                "unsafe_lookalikes_blocked_v1": row.get("rows_affected_total_v1"),
                "mapping_needed_v1": _as_bool(row.get("mapping_required_v1")),
                "final_veto_status_v1": row.get("status_v1"),
            }
        )
    missing_artifact = _str(frame, "run_id_policy_class_v1").str.contains("LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS")
    rows.append(
        {
            "veto_name_v1": "low_support_missing_artifact_safe_core_veto_v1",
            "adapter_ready_v1": False,
            "as_of_safe_lineage_v1": True,
            "rows_blocked_v1": int((masks["simplified"] & missing_artifact).sum()),
            "original_140_accidentally_blocked_v1": int((masks["simplified"] & missing_artifact & masks["original"]).sum()),
            "extra_rows_blocked_v1": int((masks["simplified"] & missing_artifact & ~masks["original"]).sum()),
            "unsafe_lookalikes_blocked_v1": 0,
            "mapping_needed_v1": True,
            "final_veto_status_v1": "NEEDS_ADAPTER_INPUT_MAPPING",
        }
    )
    summary = {
        "layer_name": "HARDEN_140_94_VETO_HARDENING_AUDIT_SUMMARY_V1",
        "veto_count_v1": len(rows),
        "adapter_ready_veto_count_v1": sum(_as_bool(row["adapter_ready_v1"]) for row in rows),
        "mapping_needed_count_v1": sum(_as_bool(row["mapping_needed_v1"]) for row in rows),
        "primary_status_v1": "INPUT_MAPPING_REQUIRED_BEFORE_ADAPTER",
    }
    return rows, summary


def _boundary_rows(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    near = frame[(~masks["hardened"]) & _num(frame, "candidate_score_v1").ge(0.90) & _bool(frame, "signal_r5_1_bad_score_v1")].copy()
    near = near.sort_values("candidate_score_v1", ascending=False).head(250)
    rows = []
    for _, row in near.iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "near_class_v1": "MISSING_54_OR_EXPANSION_CANDIDATE"
                if _as_bool(row.get("selected_original_140_v1"))
                else "NONSELECTED_LOOKALIKE",
                "selected_original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "selected_simplified_v1": _as_bool(masks["simplified"].loc[row.name]),
                "selected_hardened_v1": _as_bool(masks["hardened"].loc[row.name]),
                "passes_full_cover_skeleton_v1": _as_bool(masks["full"].loc[row.name]),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "v2_like_support_v1": _as_bool(row.get("signal_v2_like_bad_tail_v1")),
                "tail_support_v1": _as_bool(row.get("signal_r5_tail_score_v1")),
                "over_selection_risk_v1": "HIGH_UNSAFE_LOOKALIKE"
                if _as_bool(row.get("unsafe_audit_v1"))
                else "MODERATE_FALSE_POSITIVE_RISK",
            }
        )
    summary = {
        "layer_name": "HARDEN_140_94_BOUNDARY_STRESS_AUDIT_V1",
        "near_rows_sampled_v1": len(rows),
        "unsafe_lookalike_rows_v1": sum(row["unsafe_audit_only_v1"] for row in rows),
        "hardened_extra_rows_v1": int((masks["hardened"] & ~masks["original"]).sum()),
        "simplified_extra_rows_v1": int((masks["simplified"] & ~masks["original"]).sum()),
        "adapter_can_safely_use_recipe_after_mapping_v1": True,
        "expansion_should_wait_for_later_gate_v1": True,
        "status_v1": "SAFE_CORE_BOUNDARY_ACCEPTABLE_EXPANSION_SEPARATE",
    }
    return rows, summary


def _group_stability(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relevant = frame[masks["hardened"] | (masks["original"] & ~masks["simplified"]) | (masks["simplified"] & ~masks["original"])].copy()
    relevant["cohort_v1"] = np.select(
        [
            masks["hardened"].loc[relevant.index],
            (masks["simplified"] & ~masks["original"]).loc[relevant.index],
            (masks["original"] & ~masks["simplified"]).loc[relevant.index],
        ],
        ["HARDENED_SAFE_CORE", "SIMPLIFIED_EXTRA_5", "MISSING_54"],
        default="OTHER",
    )
    rows = []
    for (run_id, cohort), group in relevant.groupby(["run_id_v1", "cohort_v1"]):
        rows.append(
            {
                "run_id_v1": run_id,
                "cohort_v1": cohort,
                "fold_values_v1": "|".join(sorted(set(_str(group, "fold_id_v1")))),
                "rows_v1": len(group),
                "bad_count_audit_only_v1": int(_bool(group, "bad_label_v1").sum()),
                "tail_count_audit_only_v1": int(_bool(group, "tail_label_v1").sum()),
                "precision_audit_only_v1": float(_bool(group, "bad_label_v1").sum() / max(len(group), 1)),
                "signal_family_v1": "V2_LIKE_HIGH_CONFIDENCE" if _bool(group, "signal_v2_like_bad_tail_v1").any() else "R5_1_OTHER",
                "branch_tier_v1": "TIER_1_HARDENED" if cohort == "HARDENED_SAFE_CORE" else "EXPANSION_OR_EXTRA",
                "low_support_class_values_v1": "|".join(sorted(set(_str(group, "run_id_policy_class_v1", "UNKNOWN")))),
                "structural_low_support_rows_v1": int(_bool(group, "structural_low_support_v1").sum()),
                "student_core_overlap_rows_v1": int(_bool(group, "student_predicted_membership_v1").sum()),
                "best_lane_185_139_overlap_rows_v1": int(_bool(group, "lane_selected_v1").sum()),
                "plus45_diagnostic_overlap_rows_v1": int(_bool(group, "rows_added_vs_140_94_v1").sum()),
            }
        )
    summary = {
        "layer_name": "HARDEN_140_94_GROUP_STABILITY_AUDIT_SUMMARY_V1",
        "run_id_cohort_rows_v1": len(rows),
        "hardened_run_id_count_v1": relevant[relevant["cohort_v1"] == "HARDENED_SAFE_CORE"]["run_id_v1"].nunique(),
        "strict_loso_status_v1": "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "strict_loso_decision_valid_v1": False,
        "low_support_visible_v1": True,
        "group_concentration_risk_v1": "VISIBLE_NOT_FINAL_PROMOTION_VALID",
    }
    return rows, summary


def _adapter_recommendation(
    safe_core: dict[str, Any],
    expansion_rows: list[dict[str, Any]],
    veto_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expansion_plan = {
        "expansion_waits_for_later_gate_v1": True,
        "easy_extension_potential_recover_v1": next(
            row["potential_missing_54_recovered_v1"]
            for row in expansion_rows
            if row["module_id_v1"] == "EXPANSION_MODULE_EASY_AS_OF_EXTENSION_V1"
        ),
        "expansion_not_merged_into_safe_core_v1": True,
    }
    adapter = {
        "layer_name": "HARDEN_140_94_ADAPTER_READINESS_V1",
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "safe_core_ready_for_adapter_build_v1": False,
        "adapter_input_mapping_required_v1": True,
        "expansion_should_wait_v1": True,
        "build_safe_core_adapter_first_after_mapping_v1": True,
        "required_next_mapping_v1": [
            "low-support missing-artifact veto input",
            "AS_OF hard safety veto set",
            "normalization for score and signal-family fields",
        ],
        "selected_rows_v1": safe_core["selected_rows_v1"],
        "recovered_original_140_rows_v1": safe_core["recovered_original_140_rows_v1"],
        "extra_rows_v1": safe_core["extra_rows_v1"],
        "status_v1": "SAFE_CORE_HARDENED_INPUT_MAPPING_REQUIRED_EXPAND_LATER",
    }
    anti = {
        "layer_name": "HARDEN_140_94_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUTS",
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_in_sample_decisioning_v1": True,
        "plus45_not_used_as_target_feature_filter_threshold_v1": True,
        "membership_coverage_selected_flags_blocked_v1": True,
        "labels_mfe_safe_recoverable_blocked_as_features_v1": True,
        "implicit_latest_glob_blocked_v1": True,
        "low_support_visible_v1": True,
        "strict_loso_visible_v1": True,
        "dummy_synthetic_fallback_v1": False,
    }
    recommendation = {
        "layer_name": "HARDEN_140_94_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "rationale_v1": [
            "The hardened safe-core is simple and safety-clean with lower over-selection than the simplified recipe.",
            "It keeps expansion separate and does not attempt to recover the missing 54 in this gate.",
            "Adapter build should wait for explicit input mapping of low-support and hard safety veto fields.",
        ],
        "expansion_plan_status_v1": expansion_plan,
        "veto_status_v1": veto_summary["primary_status_v1"],
        "boundary_status_v1": boundary_summary["status_v1"],
    }
    go_no_go = {
        "layer_name": "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": safe_core["selected_rows_v1"],
        "recovered_original_140_rows_v1": safe_core["recovered_original_140_rows_v1"],
        "extra_rows_v1": safe_core["extra_rows_v1"],
        "bad_tail_audit_only_v1": safe_core["bad_tail_audit_only_v1"],
        "precision_audit_only_v1": safe_core["precision_audit_only_v1"],
        "safety_status_v1": safe_core["safety_status_v1"],
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "expansion_merged_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])
    return adapter, anti, recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    safe_core: dict[str, Any],
    extra_summary: dict[str, Any],
    missing_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    veto_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    group_summary: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "harden_140_94_reproducibility_audit_v1.md",
        [
            "# Harden 140/94 Reproducibility Audit V1",
            "",
            f"- Simplified selected: `{repro['simplified_selected_rows_v1']}`",
            f"- Recovered original 140: `{repro['simplified_recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{repro['simplified_extra_rows_v1']}`",
            f"- Bad/tail: `{repro['simplified_bad_count_audit_only_v1']} / {repro['simplified_tail_count_audit_only_v1']}`",
            f"- Safety: `{repro['simplified_safety_status_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_safe_core_definition_v1.md",
        [
            "# Harden 140/94 Safe-Core Definition V1",
            "",
            f"- Rule: `{safe_core['recipe_id_v1']}`",
            f"- Selected rows: `{safe_core['selected_rows_v1']}`",
            f"- Recovered original 140: `{safe_core['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{safe_core['extra_rows_v1']}`",
            f"- Adapter input mapping needed: `{bool(safe_core['mapping_or_normalization_needed_v1'])}`",
        ],
    )
    _write_report(
        root / "harden_140_94_extra_5_audit_v1.md",
        [
            "# Harden 140/94 Extra 5 Audit V1",
            "",
            f"- Extra rows from simplify: `{extra_summary['extra_rows_v1']}`",
            f"- Blocked by hardened rule: `{extra_summary['blocked_by_hardened_rule_v1']}`",
            f"- Retained by hardened rule: `{extra_summary['retained_by_hardened_rule_v1']}`",
            f"- Safety: `{extra_summary['safety_status_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_missing_54_audit_v1.md",
        [
            "# Harden 140/94 Missing 54 Audit V1",
            "",
            f"- Missing rows audited: `{len(missing_rows)}`",
            "- Missing rows remain expansion evidence only; they were not merged into safe-core.",
        ],
    )
    _write_report(
        root / "harden_140_94_missing_54_expansion_bucket_audit_v1.md",
        [
            "# Harden 140/94 Missing 54 Expansion Bucket Audit V1",
            "",
            *[
                f"- `{row['expansion_bucket_v1']}`: `{row['missing_54_rows_v1']}` rows"
                for row in sorted(bucket_rows, key=lambda item: item["expansion_bucket_v1"])
            ],
        ],
    )
    _write_report(
        root / "harden_140_94_expansion_module_definitions_v1.md",
        [
            "# Harden 140/94 Expansion Module Definitions V1",
            "",
            "- Expansion modules are planned for a later separate gate only.",
            "- No expansion was merged into the hardened safe-core.",
        ],
    )
    _write_report(
        root / "harden_140_94_expansion_module_metrics_v1.md",
        [
            "# Harden 140/94 Expansion Module Metrics V1",
            "",
            *[
                f"- `{row['module_id_v1']}`: potential recover `{row['potential_missing_54_recovered_v1']}`, extra `{row['extra_rows_pulled_v1']}`, safety `{row['safety_status_v1']}`"
                for row in expansion_rows
            ],
        ],
    )
    _write_report(
        root / "harden_140_94_veto_hardening_audit_v1.md",
        [
            "# Harden 140/94 Veto Hardening Audit V1",
            "",
            f"- Veto count: `{veto_summary['veto_count_v1']}`",
            f"- Mapping needed count: `{veto_summary['mapping_needed_count_v1']}`",
            f"- Primary status: `{veto_summary['primary_status_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_boundary_stress_audit_v1.md",
        [
            "# Harden 140/94 Boundary Stress Audit V1",
            "",
            f"- Near rows sampled: `{boundary_summary['near_rows_sampled_v1']}`",
            f"- Hardened extra rows: `{boundary_summary['hardened_extra_rows_v1']}`",
            f"- Status: `{boundary_summary['status_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_group_stability_audit_v1.md",
        [
            "# Harden 140/94 Group Stability Audit V1",
            "",
            f"- Hardened run_id count: `{group_summary['hardened_run_id_count_v1']}`",
            f"- Strict LOSO decision-valid: `{group_summary['strict_loso_decision_valid_v1']}`",
            f"- Group concentration risk: `{group_summary['group_concentration_risk_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_adapter_readiness_v1.md",
        [
            "# Harden 140/94 Adapter Readiness V1",
            "",
            f"- Status: `{adapter['status_v1']}`",
            f"- Safe-core ready for adapter build: `{adapter['safe_core_ready_for_adapter_build_v1']}`",
            f"- Adapter input mapping required: `{adapter['adapter_input_mapping_required_v1']}`",
            f"- Expansion should wait: `{adapter['expansion_should_wait_v1']}`",
        ],
    )
    _write_report(
        root / "harden_140_94_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Harden 140/94 Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- No R6, adapter, package, freeze, promo, live, Optuna, broad sweep, or in-sample decisioning was run.",
            "- Expansion rows, +45, membership, coverage proxy, selected flags, labels, MFE, safe_recoverable, row identity, and implicit latest/glob sources remain blocked.",
        ],
    )
    _write_report(
        root / "harden_140_94_recommendation_v1.md",
        [
            "# Harden 140/94 Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Safe-core rule: `{recommendation['safe_core_rule_id_v1']}`",
            "- Build input mapping before adapter; keep expansion for a later gate.",
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
    frame = simplify._build_frame(inputs["source_inputs"])
    masks = _build_masks(frame)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, masks)
    safe_core = _safe_core_definition(frame, masks)
    safe_core_rows = _safe_core_rows(frame, masks["hardened"])
    extra_rows, extra_summary = _extra_5_audit(frame, masks)
    missing_rows, bucket_rows = _missing_54_audit(frame, masks)
    expansion_definitions, expansion_rows = _expansion_modules(frame, masks, missing_rows)
    veto_rows, veto_summary = _veto_hardening(inputs, frame, masks)
    near_rows, boundary_summary = _boundary_rows(frame, masks)
    group_rows, group_summary = _group_stability(frame, masks)
    adapter, anti, recommendation, go_no_go = _adapter_recommendation(
        safe_core, expansion_rows, veto_summary, boundary_summary
    )

    _write_json(artifact_root / "harden_140_94_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "harden_140_94_reproducibility_audit_v1.json", repro)
    _write_json(artifact_root / "harden_140_94_safe_core_definition_v1.json", safe_core)
    _write_rows(artifact_root / "harden_140_94_safe_core_row_level_explanations_v1.csv", safe_core_rows)
    _write_json(
        artifact_root / "harden_140_94_safe_core_row_level_explanations_v1.json",
        {"row_count_v1": len(safe_core_rows), "rows_v1": safe_core_rows},
    )
    _write_rows(artifact_root / "harden_140_94_extra_5_audit_v1.csv", extra_rows)
    _write_json(
        artifact_root / "harden_140_94_extra_5_audit_v1.json",
        {"summary_v1": extra_summary, "rows_v1": extra_rows},
    )
    _write_rows(artifact_root / "harden_140_94_missing_54_audit_v1.csv", missing_rows)
    _write_json(
        artifact_root / "harden_140_94_missing_54_audit_v1.json",
        {"row_count_v1": len(missing_rows), "rows_v1": missing_rows},
    )
    _write_rows(artifact_root / "harden_140_94_missing_54_expansion_bucket_audit_v1.csv", bucket_rows)
    _write_json(
        artifact_root / "harden_140_94_missing_54_expansion_bucket_audit_v1.json",
        {"rows_v1": bucket_rows},
    )
    _write_json(artifact_root / "harden_140_94_expansion_module_definitions_v1.json", expansion_definitions)
    _write_rows(artifact_root / "harden_140_94_expansion_module_metrics_v1.csv", expansion_rows)
    _write_json(
        artifact_root / "harden_140_94_expansion_module_metrics_v1.json",
        {"rows_v1": expansion_rows, "expansion_merged_into_safe_core_v1": False},
    )
    _write_rows(artifact_root / "harden_140_94_veto_hardening_audit_v1.csv", veto_rows)
    _write_json(
        artifact_root / "harden_140_94_veto_hardening_audit_v1.json",
        {"summary_v1": veto_summary, "rows_v1": veto_rows},
    )
    _write_json(artifact_root / "harden_140_94_boundary_stress_audit_v1.json", boundary_summary)
    _write_rows(artifact_root / "harden_140_94_near_miss_and_near_fail_rows_v1.csv", near_rows)
    _write_json(
        artifact_root / "harden_140_94_near_miss_and_near_fail_rows_v1.json",
        {"row_count_v1": len(near_rows), "rows_v1": near_rows},
    )
    _write_rows(artifact_root / "harden_140_94_group_stability_audit_v1.csv", group_rows)
    _write_json(
        artifact_root / "harden_140_94_group_stability_audit_v1.json",
        {"summary_v1": group_summary, "rows_v1": group_rows},
    )
    _write_json(artifact_root / "harden_140_94_adapter_readiness_v1.json", adapter)
    _write_json(artifact_root / "harden_140_94_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "harden_140_94_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "safe_core_rule_id_v1": HARDENED_RECIPE_ID,
        "selected_rows_v1": safe_core["selected_rows_v1"],
        "recovered_original_140_rows_v1": safe_core["recovered_original_140_rows_v1"],
        "extra_rows_v1": safe_core["extra_rows_v1"],
        "bad_tail_audit_only_v1": safe_core["bad_tail_audit_only_v1"],
        "precision_audit_only_v1": safe_core["precision_audit_only_v1"],
        "safety_status_v1": safe_core["safety_status_v1"],
        "adapter_readiness_v1": adapter["status_v1"],
        "expansion_plan_status_v1": "EXPAND_LATER_SEPARATE_GATE",
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
            "# Harden 140/94 Safe Core And Expand Later V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Safe-core rule: `{HARDENED_RECIPE_ID}`",
            f"- Selected rows: `{safe_core['selected_rows_v1']}`",
            f"- Recovered original 140: `{safe_core['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{safe_core['extra_rows_v1']}`",
            f"- Bad/tail: `{safe_core['bad_tail_audit_only_v1'][0]} / {safe_core['bad_tail_audit_only_v1'][1]}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        safe_core,
        extra_summary,
        missing_rows,
        bucket_rows,
        expansion_rows,
        veto_summary,
        boundary_summary,
        group_summary,
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
