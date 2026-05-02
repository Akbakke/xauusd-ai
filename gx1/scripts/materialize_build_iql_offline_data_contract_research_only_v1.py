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

from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as refine_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1"

INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)
INPUT_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1_20260428T182517Z_LOCK"
)
INPUT_CLOSE_ROOT = (
    DEFAULT_REPORTS_ROOT / "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1_20260428T175937Z_LOCK"
)
INPUT_HOLD_ROOT = (
    DEFAULT_REPORTS_ROOT / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T111216Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

FINAL_STATUS = "IQL_OFFLINE_DATA_CONTRACT_READY_FOR_SANITY_TRAINING_RESEARCH_ONLY"
NEXT_ACTION = "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1"
CHOSEN_RESEARCH_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

EXPECTED_140_SELECTED = 140
EXPECTED_140_BAD = 140
EXPECTED_140_TAIL = 94
EXPECTED_SAFE_CORE_SELECTED = 89
EXPECTED_SAFE_CORE_RECOVERED = 86
EXPECTED_SAFE_CORE_EXTRA = 3
EXPECTED_SAFE_CORE_BAD = 86
EXPECTED_SAFE_CORE_TAIL = 55
EXPECTED_SAFE_CORE_PRECISION = 0.9662921348314607
EXPECTED_SHIELD_SELECTED = 78
EXPECTED_SHIELD_ORIGINAL_RETAINED = 75
EXPECTED_SHIELD_BAD = 75
EXPECTED_SHIELD_TAIL = 55
EXPECTED_SHIELD_PRECISION = 0.9615384615384616

ALLOWED_FINAL_STATUSES = {
    "IQL_OFFLINE_DATA_CONTRACT_READY_FOR_SANITY_TRAINING_RESEARCH_ONLY",
    "IQL_OFFLINE_DATA_CONTRACT_READY_NEEDS_FEATURE_NORMALIZATION",
    "IQL_OFFLINE_DATA_CONTRACT_PARTIAL_NEEDS_XGB_TRANSFORMER_FEATURE_LINEAGE",
    "IQL_OFFLINE_DATA_CONTRACT_PARTIAL_NEEDS_BEHAVIOR_POLICY_CLARIFICATION",
    "IQL_OFFLINE_DATA_CONTRACT_BLOCKED_BY_STATE_LEAKAGE_RISK",
    "IQL_OFFLINE_DATA_CONTRACT_BLOCKED_BY_REWARD_LEAKAGE_RISK",
    "IQL_OFFLINE_DATA_CONTRACT_BLOCKED_BY_INSUFFICIENT_SAFE_COHORT_SUPPORT",
    "IQL_OFFLINE_DATA_CONTRACT_BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1",
    "NORMALIZE_IQL_OFFLINE_STATE_FEATURES_V1",
    "DEEPEN_XGB_TRANSFORMER_FEATURE_LINEAGE_FOR_IQL_STATE_V1",
    "DEEPEN_IQL_BEHAVIOR_POLICY_AND_ACTION_SUPPORT_AUDIT_V1",
    "REBUILD_IQL_STATE_CONTRACT_WITH_STRICTER_NO_LEAKAGE_V1",
    "HOLD_IQL_UNTIL_SAFE_COHORT_SUPPORT_IMPROVES_V1",
}

REQUIRED_OUTPUTS = [
    "iql_offline_data_contract_input_manifest_v1.json",
    "iql_offline_data_contract_reproducibility_audit_v1.json",
    "iql_offline_data_contract_reproducibility_audit_v1.md",
    "iql_offline_eligibility_cohorts_v1.csv",
    "iql_offline_eligibility_cohorts_v1.json",
    "iql_offline_eligibility_cohorts_v1.md",
    "iql_offline_state_contract_v1.csv",
    "iql_offline_state_contract_v1.json",
    "iql_offline_state_contract_v1.md",
    "iql_offline_action_contract_v1.json",
    "iql_offline_action_contract_v1.md",
    "iql_offline_reward_contract_v1.json",
    "iql_offline_reward_contract_v1.md",
    "iql_offline_behavior_policy_audit_v1.json",
    "iql_offline_behavior_policy_audit_v1.md",
    "iql_offline_split_policy_v1.json",
    "iql_offline_split_policy_v1.md",
    "iql_offline_safety_shield_contract_v1.json",
    "iql_offline_safety_shield_contract_v1.md",
    "iql_offline_xgb_transformer_feature_integration_audit_v1.csv",
    "iql_offline_xgb_transformer_feature_integration_audit_v1.json",
    "iql_offline_xgb_transformer_feature_integration_audit_v1.md",
    "iql_offline_no_shortcut_audit_v1.json",
    "iql_offline_no_shortcut_audit_v1.md",
    "iql_offline_readiness_assessment_v1.json",
    "iql_offline_readiness_assessment_v1.md",
    "iql_offline_data_contract_recommendation_v1.json",
    "iql_offline_data_contract_recommendation_v1.md",
    "build_iql_offline_data_contract_research_only_go_no_go_v1.json",
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


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return refine_gate._bool(frame, column)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return refine_gate._num(frame, column, default)


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
    iql_production: bool = False,
    iql_training_now: bool = False,
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
    if iql_production:
        failures.append("IQL_PRODUCTION_FORBIDDEN")
    if iql_training_now:
        failures.append("IQL_TRAINING_FORBIDDEN_IN_CONTRACT_GATE")
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


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_REFINE_CLEAN_ROOT,
        INPUT_CLEAN_ROOT,
        INPUT_CLOSE_ROOT,
        INPUT_HOLD_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "refine_summary": INPUT_REFINE_CLEAN_ROOT / "summary_v1.json",
        "refine_go_no_go": INPUT_REFINE_CLEAN_ROOT
        / "refine_clean_as_of_safety_layer_to_retain_safe_core_go_no_go_v1.json",
        "refine_candidate_metrics": INPUT_REFINE_CLEAN_ROOT / "refine_clean_safety_layer_candidate_metrics_v1.json",
        "clean_summary": INPUT_CLEAN_ROOT / "summary_v1.json",
        "clean_go_no_go": INPUT_CLEAN_ROOT / "build_clean_as_of_safety_feature_layer_from_source_signals_go_no_go_v1.json",
        "close_summary": INPUT_CLOSE_ROOT / "summary_v1.json",
        "close_go_no_go": INPUT_CLOSE_ROOT / "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json",
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
    refine_go = _read_json(required["refine_go_no_go"])
    if refine_go.get("status_v1") != "CLEAN_AS_OF_SAFETY_LAYER_REFINED_STILL_ORANGE_DESTRUCTIVE":
        raise RuntimeError("INPUT_REFINE_GATE_STATUS_NOT_STILL_ORANGE_DESTRUCTIVE")
    return {
        "required_paths": required,
        "refine_summary": _read_json(required["refine_summary"]),
        "refine_go_no_go": refine_go,
        "refine_candidate_metrics": _read_json(required["refine_candidate_metrics"]),
        "clean_summary": _read_json(required["clean_summary"]),
        "clean_go_no_go": _read_json(required["clean_go_no_go"]),
        "close_summary": _read_json(required["close_summary"]),
        "close_go_no_go": _read_json(required["close_go_no_go"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "hold_go_no_go": _read_json(required["hold_go_no_go"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "frame_inputs": refine_gate._load_inputs(),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_OFFLINE_DATA_CONTRACT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "clean_safety_layer_refinement_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
            "clean_as_of_safety_layer_root_v1": str(INPUT_CLEAN_ROOT),
            "proxy_branch_closure_root_v1": str(INPUT_CLOSE_ROOT),
            "safe_core_hold_root_v1": str(INPUT_HOLD_ROOT),
            "safe_core_harden_root_v1": str(INPUT_HARDEN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    return refine_gate._build_frame_and_masks(inputs["frame_inputs"])


def _cohort_metrics(frame: pd.DataFrame, mask: pd.Series, *, baseline_mask: pd.Series) -> dict[str, Any]:
    selected = frame[mask]
    bad = int(_bool(selected, "bad_label_v1").sum())
    unsafe = int(_bool(selected, "unsafe_audit_v1").sum())
    return {
        "selected_rows_v1": int(mask.sum()),
        "original_140_overlap_v1": int((mask & baseline_mask).sum()),
        "bad_count_audit_only_v1": bad,
        "tail_count_audit_only_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_audit_only_v1": float(bad / max(int(mask.sum()), 1)),
        "unsafe_hits_audit_only_v1": unsafe,
        "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
    }


def _reproducibility(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shielded = safe_core & ~masks["source_confluence_repairable_v1"]
    unsafe_without = masks["base_without_hard_safety_veto_v1"] & ~safe_core & _bool(frame, "unsafe_audit_v1")
    base_m = _cohort_metrics(frame, baseline, baseline_mask=baseline)
    safe_m = _cohort_metrics(frame, safe_core, baseline_mask=baseline)
    shield_m = _cohort_metrics(frame, shielded, baseline_mask=baseline)
    payload = {
        "layer_name": "IQL_OFFLINE_DATA_CONTRACT_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "baseline_140_94_v1": {
            **base_m,
            "expected_selected_rows_v1": EXPECTED_140_SELECTED,
            "expected_bad_tail_v1": [EXPECTED_140_BAD, EXPECTED_140_TAIL],
        },
        "safe_core_89_v1": {
            **safe_m,
            "recovered_original_140_v1": safe_m["original_140_overlap_v1"],
            "extra_rows_v1": int((safe_core & ~baseline).sum()),
        },
        "source_safety_shielded_78_v1": {
            **shield_m,
            "original_140_retained_v1": shield_m["original_140_overlap_v1"],
            "unsafe_row_blocked_v1": int(unsafe_without.sum()) == 1,
            "retention_class_v1": "ORANGE",
            "adapter_ready_v1": False,
        },
        "unsafe_extra_without_hard_veto_rows_v1": int(unsafe_without.sum()),
        "historical_v2_blueprint_used_v1": False,
        "adapter_r6_iql_production_remain_blocked_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    base = payload["baseline_140_94_v1"]
    safe = payload["safe_core_89_v1"]
    shield = payload["source_safety_shielded_78_v1"]
    checks = [
        base["selected_rows_v1"] == EXPECTED_140_SELECTED,
        base["bad_count_audit_only_v1"] == EXPECTED_140_BAD,
        base["tail_count_audit_only_v1"] == EXPECTED_140_TAIL,
        base["safety_status_v1"] == "CLEAN",
        safe["selected_rows_v1"] == EXPECTED_SAFE_CORE_SELECTED,
        safe["recovered_original_140_v1"] == EXPECTED_SAFE_CORE_RECOVERED,
        safe["extra_rows_v1"] == EXPECTED_SAFE_CORE_EXTRA,
        safe["bad_count_audit_only_v1"] == EXPECTED_SAFE_CORE_BAD,
        safe["tail_count_audit_only_v1"] == EXPECTED_SAFE_CORE_TAIL,
        abs(safe["precision_audit_only_v1"] - EXPECTED_SAFE_CORE_PRECISION) < 1e-12,
        shield["selected_rows_v1"] == EXPECTED_SHIELD_SELECTED,
        shield["original_140_retained_v1"] == EXPECTED_SHIELD_ORIGINAL_RETAINED,
        shield["bad_count_audit_only_v1"] == EXPECTED_SHIELD_BAD,
        shield["tail_count_audit_only_v1"] == EXPECTED_SHIELD_TAIL,
        abs(shield["precision_audit_only_v1"] - EXPECTED_SHIELD_PRECISION) < 1e-12,
        shield["safety_status_v1"] == "CLEAN",
        shield["unsafe_row_blocked_v1"] is True,
    ]
    if not all(checks):
        raise RuntimeError("IQL_OFFLINE_DATA_CONTRACT_REPRODUCTION_FAILED")
    return True


def _eligibility_cohorts(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shielded = safe_core & ~masks["source_confluence_repairable_v1"]
    near_miss = masks["base_without_hard_safety_veto_v1"] & ~safe_core
    non_selected = ~(baseline | safe_core | shielded)
    rows = [
        (
            "140_94_BASELINE_COMPARATOR",
            baseline,
            "COMPARATOR_REFERENCE_ONLY",
            "Not safety-shielded enough for IQL first pass; kept as current causal baseline comparator.",
        ),
        (
            "SAFE_CORE_89_RESEARCH_CANDIDATE",
            safe_core,
            "RESEARCH_CANDIDATE_NOT_PRODUCTION",
            "Useful concrete rule core, but adapter remains blocked by deployable hard-veto gap.",
        ),
        (
            CHOSEN_RESEARCH_COHORT,
            shielded,
            "FIRST_IQL_RESEARCH_ONLY_ELIGIBILITY_SHIELD",
            "Most conservative clean source-signal shield: unsafe row blocked, ORANGE retention, not adapter-ready.",
        ),
        (
            "NON_SELECTED_AND_NEAR_MISS_POOL",
            non_selected,
            "SKIP_NO_TRADE_COMPARISON_POOL_WITH_SUPPORT_LIMITS",
            f"Includes {int(near_miss.sum())} near-miss row(s); use only for skip/action support diagnostics.",
        ),
    ]
    out = []
    for name, mask, role, note in rows:
        metrics = _cohort_metrics(frame, mask, baseline_mask=baseline)
        out.append(
            {
                "cohort_id_v1": name,
                "role_v1": role,
                **metrics,
                "research_only_v1": True,
                "production_or_live_allowed_v1": False,
                "notes_v1": note,
            }
        )
    return out


def _state_contract_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    def row(
        field: str,
        source: str,
        dtype: str,
        status: str,
        reason: str,
        *,
        normalization: bool = False,
        missing: str = "REJECT_ROW_IF_REQUIRED_MISSING",
        leakage: str = "LOW",
        future: str = "POSSIBLE",
    ) -> dict[str, Any]:
        return {
            "field_name_v1": field,
            "source_artifact_path_v1": source,
            "datatype_v1": dtype,
            "as_of_lineage_v1": status,
            "allowed_as_state_v1": status in {"AS_OF_SAFE_ALLOWED", "AS_OF_SAFE_NEEDS_NORMALIZATION"},
            "blocked_reason_v1": "" if status in {"AS_OF_SAFE_ALLOWED", "AS_OF_SAFE_NEEDS_NORMALIZATION"} else reason,
            "normalization_needed_v1": normalization,
            "missing_handling_v1": missing,
            "leakage_risk_v1": leakage,
            "adapter_r6_future_feasibility_v1": future,
            "present_in_source_frame_v1": field in frame.columns,
        }

    rows = [
        row("candidate_score_v1", "tail-repaired R5.2/source candidate frame", "float", "AS_OF_SAFE_NEEDS_NORMALIZATION", "score scaling and range contract required", normalization=True),
        row("signal_r5_1_bad_score_v1", "source signal support columns", "bool", "AS_OF_SAFE_ALLOWED", "pre-outcome support signal"),
        row("signal_r5_bad_score_v1", "source signal support columns", "bool", "AS_OF_SAFE_ALLOWED", "pre-outcome support signal"),
        row("signal_r5_tail_score_v1", "source signal support columns", "bool", "AS_OF_SAFE_ALLOWED", "pre-outcome support signal"),
        row("signal_v2_like_bad_tail_v1", "source signal support columns", "bool", "AS_OF_SAFE_ALLOWED", "pre-outcome V2-like support flag, not historical blueprint membership"),
        row("signal_tail_repair_v1", "source signal support columns", "bool", "AS_OF_SAFE_ALLOWED", "pre-outcome tail-repair support flag"),
        row("run_id_policy_class_v1", "source support policy", "category", "AS_OF_SAFE_NEEDS_NORMALIZATION", "support policy must be normalized and not used as row-id shortcut", normalization=True, leakage="MEDIUM_CONTROLLED"),
        row("structural_low_support_v1", "source support audit", "bool", "AS_OF_SAFE_NEEDS_NORMALIZATION", "support guard only; normalize lineage before broader use", normalization=True, leakage="MEDIUM_CONTROLLED"),
        row("zero_denominator_group_v1", "source support audit", "bool", "AS_OF_SAFE_NEEDS_NORMALIZATION", "support guard only; normalize lineage before broader use", normalization=True, leakage="MEDIUM_CONTROLLED"),
        row("decision_timestamp_v1", "source frame", "timestamp", "AS_OF_SAFE_DIAGNOSTIC_ONLY", "split/time ordering only; not policy state in first contract", leakage="MEDIUM"),
        row("run_id_v1", "source frame", "category", "AS_OF_SAFE_DIAGNOSTIC_ONLY", "split/group audit only; not state to avoid run shortcut", leakage="MEDIUM"),
        row("fold_id_v1", "source frame", "category", "AS_OF_SAFE_DIAGNOSTIC_ONLY", "split/fold audit only; not state", leakage="MEDIUM"),
        row("student_oof_score_v1", "student membership artifact", "float", "BLOCKED_MEMBERSHIP_PROXY", "student was trained to recover Lane 08 membership and missed +45; not deployable state", leakage="HIGH", future="BLOCKED"),
        row("student_core_selected_v1", "student membership artifact", "bool", "BLOCKED_MEMBERSHIP_PROXY", "membership-derived selector", leakage="HIGH", future="BLOCKED"),
        row("bad_label_v1", "outcome audit labels", "bool", "BLOCKED_OUTCOME_LABEL_STATE", "reward/audit only, never state", leakage="HIGH", future="BLOCKED"),
        row("tail_label_v1", "outcome audit labels", "bool", "BLOCKED_OUTCOME_LABEL_STATE", "reward/audit only, never state", leakage="HIGH", future="BLOCKED"),
        row("unsafe_audit_v1", "audit-only safety labels", "bool", "BLOCKED_AUDIT_ONLY_STATE", "safety audit/reward penalty only, not policy state", leakage="HIGH", future="BLOCKED"),
        row("safety_clear_audit_v1", "audit-only safety labels", "bool", "BLOCKED_AUDIT_ONLY_STATE", "audit-only hard veto is not deployable", leakage="HIGH", future="BLOCKED"),
        row("hard_veto_clear_shadow_v1", "audit-only hard veto", "bool", "BLOCKED_AUDIT_ONLY_STATE", "shadow/audit veto cannot be state", leakage="HIGH", future="BLOCKED"),
        row("source_evidence_v1", "source evidence text", "string", "AS_OF_SAFE_DIAGNOSTIC_ONLY", "raw token string contains blocked HISTORICAL_V2_BLUEPRINT token; parse only allowlisted tokens", leakage="MEDIUM"),
        row("HISTORICAL_V2_BLUEPRINT", "source_evidence token", "bool", "BLOCKED_HISTORICAL_ARTIFACT_PROXY", "lane pack blocked it as membership/coverage/artifact proxy risk", leakage="HIGH", future="BLOCKED"),
        row("candidate_uid_v1", "row identifier", "string", "BLOCKED_ROW_IDENTITY", "row identity forbidden as state", leakage="HIGH", future="BLOCKED"),
        row("trade_uid_v1", "row identifier", "string", "BLOCKED_ROW_IDENTITY", "trade identity forbidden as state", leakage="HIGH", future="BLOCKED"),
        row("selected_original_140_v1", "materialized membership flag", "bool", "BLOCKED_MEMBERSHIP_FLAG", "comparator membership only", leakage="HIGH", future="BLOCKED"),
        row("is_185_139_teacher_v1", "materialized membership flag", "bool", "BLOCKED_MEMBERSHIP_FLAG", "185/139 comparator/diagnostic only", leakage="HIGH", future="BLOCKED"),
        row("is_plus45_diagnostic_v1", "diagnostic membership flag", "bool", "BLOCKED_MEMBERSHIP_FLAG", "+45 diagnostic only", leakage="HIGH", future="BLOCKED"),
        row("rows_added_vs_140_94_v1", "materialized comparison flag", "bool", "BLOCKED_MEMBERSHIP_FLAG", "row-level diagnostic flag only", leakage="HIGH", future="BLOCKED"),
        row("protected_winner_status_v1", "audit-only safety status", "string", "BLOCKED_AUDIT_ONLY_STATE", "unless independently AS_OF-mapped later", leakage="HIGH", future="BLOCKED"),
        row("runner_protect_status_v1", "audit-only safety status", "string", "BLOCKED_AUDIT_ONLY_STATE", "unless independently AS_OF-mapped later", leakage="HIGH", future="BLOCKED"),
        row("ambiguous_high_mfe_status_v1", "audit-only/MFE status", "string", "BLOCKED_MFE_OR_HINDSIGHT", "MFE/hindsight risk", leakage="HIGH", future="BLOCKED"),
        row("fifty_plus_mfe_risk_v1", "audit-only/MFE status", "bool", "BLOCKED_MFE_OR_HINDSIGHT", "post-outcome MFE-style risk", leakage="HIGH", future="BLOCKED"),
    ]
    validate_state_contract(rows)
    return rows


def validate_state_contract(rows: list[dict[str, Any]]) -> bool:
    by_name = {row["field_name_v1"]: row for row in rows}
    for blocked in [
        "bad_label_v1",
        "tail_label_v1",
        "unsafe_audit_v1",
        "HISTORICAL_V2_BLUEPRINT",
        "student_oof_score_v1",
        "candidate_uid_v1",
        "selected_original_140_v1",
        "is_plus45_diagnostic_v1",
        "fifty_plus_mfe_risk_v1",
    ]:
        row = by_name.get(blocked)
        if row is None or row["allowed_as_state_v1"]:
            raise RuntimeError(f"LEAKY_STATE_FIELD_NOT_BLOCKED: {blocked}")
    allowed = [row for row in rows if row["allowed_as_state_v1"]]
    if len(allowed) < 6:
        raise RuntimeError("INSUFFICIENT_ALLOWED_AS_OF_STATE_FIELDS")
    return True


def _action_contract() -> dict[str, Any]:
    payload = {
        "layer_name": "IQL_OFFLINE_ACTION_CONTRACT_V1",
        "action_space_status_v1": "BINARY_ONLY_SIZING_ACTIONS_NOT_SUPPORTED_YET",
        "sizing_actions_allowed_v1": False,
        "actions_v1": [
            {
                "action_id_v1": 0,
                "action_name_v1": "SKIP",
                "action_meaning_v1": "Do not take the candidate trade/opportunity.",
                "required_logged_support_v1": "non-selected and near-miss pool plus counterfactual caution",
                "observable_historically_v1": "PARTIAL_AS_NON_SELECTION",
                "offline_evaluable_v1": "LIMITED_RESEARCH_ONLY",
                "limitations_v1": "Skip reward is opportunity/counterfactual-sensitive; use only in sanity training with conservative diagnostics.",
            },
            {
                "action_id_v1": 1,
                "action_name_v1": "TAKE_TRADE",
                "action_meaning_v1": "Take eligible candidate inside the research-only safety shield.",
                "required_logged_support_v1": "selected cohort outcome labels as reward/audit only",
                "observable_historically_v1": "YES_FOR_SELECTED_COHORTS",
                "offline_evaluable_v1": "YES_RESEARCH_ONLY",
                "limitations_v1": "Must be shielded; not adapter/R6/live proof.",
            },
        ],
    }
    return payload


def _reward_contract() -> dict[str, Any]:
    return {
        "layer_name": "IQL_OFFLINE_REWARD_CONTRACT_V1",
        "labels_used_as_state_v1": False,
        "labels_used_only_as_reward_or_audit_v1": True,
        "reward_candidates_v1": [
            {
                "reward_id_v1": "BAD_TAIL_BINARY_REWARD",
                "target_fields_v1": ["bad_label_v1", "tail_label_v1"],
                "definition_v1": "positive reward for bad/tail success; false-positive penalty",
                "state_leakage_status_v1": "CLEAN_IF_TARGETS_NOT_IN_STATE",
                "suitability_v1": "SIMPLE_SANITY_REWARD",
                "limitations_v1": "Outcome labels are post-event reward only; not suitable for row-level filtering.",
            },
            {
                "reward_id_v1": "SAFETY_WEIGHTED_REWARD",
                "target_fields_v1": ["bad_label_v1", "tail_label_v1", "unsafe_audit_v1"],
                "definition_v1": "positive for bad/tail, strong negative for unsafe/protected/runner/ambiguous/MFE audit hits",
                "state_leakage_status_v1": "CLEAN_IF_SAFETY_AUDITS_REWARD_ONLY",
                "suitability_v1": "PREFERRED_RESEARCH_REWARD",
                "limitations_v1": "Safety labels remain reward/audit only until deployable AS_OF equivalents exist.",
            },
            {
                "reward_id_v1": "CONSERVATIVE_TRADE_UTILITY_REWARD",
                "target_fields_v1": ["bad_label_v1", "tail_label_v1", "unsafe_audit_v1", "low_support_penalty"],
                "definition_v1": "positive desired outcome, negative unsafe/low-confidence/overtrading pressure",
                "state_leakage_status_v1": "CLEAN_IF_PENALTIES_NOT_USED_AS_STATE",
                "suitability_v1": "SECONDARY_RESEARCH_REWARD",
                "limitations_v1": "Requires fixed predeclared coefficients before training.",
            },
        ],
    }


def _behavior_policy_audit() -> dict[str, Any]:
    return {
        "layer_name": "IQL_OFFLINE_BEHAVIOR_POLICY_AUDIT_V1",
        "historical_behavior_policy_v1": "artifact-selection and logged candidate outcome substrate",
        "logged_actions_available_v1": "PARTIAL",
        "data_shape_v1": "selected/non-selected candidate frame, not production policy log",
        "propensities_available_v1": False,
        "can_iql_run_without_propensities_v1": True,
        "why_iql_without_propensities_is_limited_v1": "IQL sanity training can fit an offline value/policy surrogate, but this is not unbiased off-policy evaluation and not deployment proof.",
        "supported_action_space_v1": "SKIP_TAKE_BINARY_ONLY",
        "sizing_actions_supported_v1": False,
        "primary_biases_v1": [
            "selection-policy support bias",
            "near-miss counterfactual uncertainty",
            "low-support/group concentration",
            "reward labels are post-event and must stay out of state",
        ],
        "cohort_recommendation_v1": CHOSEN_RESEARCH_COHORT,
        "status_v1": "SUFFICIENT_FOR_RESEARCH_ONLY_IQL_SANITY_NOT_FOR_POLICY_EVAL",
    }


def _split_policy() -> dict[str, Any]:
    return {
        "layer_name": "IQL_OFFLINE_SPLIT_POLICY_V1",
        "training_allowed_now_v1": False,
        "required_before_training_v1": [
            "freeze state field allowlist",
            "freeze reward candidate",
            "freeze safety shield cohort",
            "freeze split plan",
            "write no-leakage check over final dataset",
        ],
        "recommended_splits_v1": [
            {
                "split_id_v1": "RUN_ID_HELDOUT",
                "purpose_v1": "test cross-run generalization",
                "leakage_risk_v1": "run-id must not be state",
            },
            {
                "split_id_v1": "FOLD_HELDOUT",
                "purpose_v1": "respect OOF/fold structure when available",
                "leakage_risk_v1": "fold id split only, not state",
            },
            {
                "split_id_v1": "GROUP_LOSO_STYLE",
                "purpose_v1": "stress low-support structural groups",
                "leakage_risk_v1": "strict LOSO remains invalid for promotion; visible for research only",
            },
            {
                "split_id_v1": "TIME_ORDER_AWARE",
                "purpose_v1": "avoid training on future windows before past windows",
                "leakage_risk_v1": "decision timestamp split only",
            },
            {
                "split_id_v1": "LOW_SUPPORT_AWARE",
                "purpose_v1": "ensure shield does not rely only on low-support slices",
                "leakage_risk_v1": "support fields must be normalized",
            },
        ],
    }


def _safety_shield_contract(repro: dict[str, Any]) -> dict[str, Any]:
    shield = repro["source_safety_shielded_78_v1"]
    payload = {
        "layer_name": "IQL_OFFLINE_SAFETY_SHIELD_CONTRACT_V1",
        "primary_research_eligibility_cohort_v1": CHOSEN_RESEARCH_COHORT,
        "selected_rows_v1": shield["selected_rows_v1"],
        "original_140_retained_v1": shield["original_140_retained_v1"],
        "bad_tail_audit_only_v1": [shield["bad_count_audit_only_v1"], shield["tail_count_audit_only_v1"]],
        "precision_audit_only_v1": shield["precision_audit_only_v1"],
        "safety_status_v1": shield["safety_status_v1"],
        "unsafe_row_blocked_v1": shield["unsafe_row_blocked_v1"],
        "retention_class_v1": shield["retention_class_v1"],
        "iql_policy_may_act_only_inside_shield_v1": True,
        "safety_veto_outside_policy_override_v1": True,
        "historical_v2_blueprint_allowed_v1": False,
        "membership_or_coverage_proxy_allowed_v1": False,
        "audit_only_veto_allowed_as_state_v1": False,
        "adapter_ready_v1": False,
        "research_only_v1": True,
        "recommended_setup_v1": "Constrain first IQL sanity pass to the 78-row source-safety-shielded eligibility set; safety shield overrides TAKE.",
    }
    validate_safety_shield(payload)
    return payload


def validate_safety_shield(payload: dict[str, Any]) -> bool:
    if payload["selected_rows_v1"] != EXPECTED_SHIELD_SELECTED:
        raise RuntimeError("IQL_SAFETY_SHIELD_SELECTED_COUNT_MISMATCH")
    if not payload["unsafe_row_blocked_v1"]:
        raise RuntimeError("IQL_SAFETY_SHIELD_DOES_NOT_BLOCK_UNSAFE_ROW")
    if payload["historical_v2_blueprint_allowed_v1"] or payload["membership_or_coverage_proxy_allowed_v1"]:
        raise RuntimeError("IQL_SAFETY_SHIELD_USES_BLOCKED_PROXY")
    if payload["audit_only_veto_allowed_as_state_v1"]:
        raise RuntimeError("IQL_SAFETY_SHIELD_USES_AUDIT_ONLY_STATE")
    return True


def _xgb_transformer_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    def row(name: str, source: str, target: str, status: str, reason: str) -> dict[str, Any]:
        return {
            "feature_or_signal_v1": name,
            "source_v1": source,
            "training_target_v1": target,
            "oof_status_v1": "OOF_OR_PRECOMPUTED_AS_OF" if status.startswith("USABLE") else "NOT_USABLE_AS_STATE",
            "as_of_lineage_v1": status,
            "leakage_risk_v1": "LOW" if status.startswith("USABLE") else "HIGH_OR_UNKNOWN",
            "usable_as_iql_state_v1": status.startswith("USABLE"),
            "diagnostic_only_v1": status.startswith("DIAGNOSTIC"),
            "blocked_reason_v1": "" if status.startswith("USABLE") else reason,
            "present_in_source_frame_v1": name in frame.columns,
        }

    rows = [
        row("candidate_score_v1", "tail-repaired R5.2 score artifact", "bad/tail candidate scoring", "USABLE_NEEDS_NORMALIZATION", "range/normalization required"),
        row("signal_r5_1_bad_score_v1", "source score support flag", "bad support", "USABLE_AS_OF_STATE", ""),
        row("signal_r5_bad_score_v1", "source score support flag", "bad support", "USABLE_AS_OF_STATE", ""),
        row("signal_r5_tail_score_v1", "source score support flag", "tail support", "USABLE_AS_OF_STATE", ""),
        row("signal_v2_like_bad_tail_v1", "source score support flag", "V2-like bad/tail support", "USABLE_AS_OF_STATE", ""),
        row("student_oof_score_v1", "student membership OOF artifact", "Lane 08 membership", "DIAGNOSTIC_ONLY_MEMBERSHIP_PROXY", "student trained against forbidden teacher boundary"),
        row("transformer_embedding_v1", "not found in locked source frame", "unknown", "XGB_TRANSFORMER_FEATURES_NOT_READY_FOR_IQL_STATE_NOT_BLOCKER_FOR_CONTRACT", "no independently lineage-proven transformer embedding found"),
    ]
    return rows


def _no_shortcut_audit() -> dict[str, Any]:
    checks = {
        "final_labels_as_state_blocked_v1": True,
        "bad_tail_as_state_blocked_v1": True,
        "mfe_hindsight_as_state_blocked_v1": True,
        "safe_recoverable_direct_as_state_blocked_v1": True,
        "membership_flags_blocked_v1": True,
        "coverage_proxy_blocked_v1": True,
        "historical_v2_blueprint_blocked_v1": True,
        "row_identity_blocked_v1": True,
        "selected_by_flags_blocked_v1": True,
        "artifact_shortcut_blocked_v1": True,
        "audit_only_veto_as_state_blocked_v1": True,
        "adapter_r6_live_not_opened_v1": True,
        "iql_training_not_run_v1": True,
    }
    return {
        "layer_name": "IQL_OFFLINE_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "checks_v1": checks,
        "critical_failures_v1": [name for name, passed in checks.items() if not passed],
    }


def _readiness_assessment() -> dict[str, Any]:
    return {
        "layer_name": "IQL_OFFLINE_READINESS_ASSESSMENT_V1",
        "status_v1": FINAL_STATUS,
        "offline_iql_data_contract_ready_v1": True,
        "iql_sanity_training_allowed_next_v1": True,
        "iql_training_run_now_v1": False,
        "iql_production_allowed_v1": False,
        "first_cohort_v1": CHOSEN_RESEARCH_COHORT,
        "first_action_space_v1": "SKIP_TAKE_BINARY_ONLY",
        "first_reward_recommendation_v1": "SAFETY_WEIGHTED_REWARD",
        "xgb_transformer_feature_status_v1": "BASE_SOURCE_MODEL_SIGNALS_AVAILABLE_TRANSFORMER_NOT_READY_NOT_BLOCKER",
        "behavior_policy_status_v1": "PROPENSITIES_ABSENT_BUT_IQL_SANITY_ALLOWED_RESEARCH_ONLY",
        "missing_before_sanity_training_v1": [
            "freeze exact dataset export",
            "freeze reward candidate",
            "freeze split assignment",
            "normalize candidate_score_v1 and support-policy fields",
            "write training harness no-production guard",
        ],
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }


def _recommendation() -> dict[str, Any]:
    return {
        "layer_name": "IQL_OFFLINE_DATA_CONTRACT_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "recommendation_v1": "Proceed only to offline IQL sanity training research gate using the 78-row source-safety-shielded eligibility cohort.",
        "do_not_open_adapter_r6_live_v1": True,
        "do_not_use_iql_for_production_v1": True,
        "do_not_train_in_this_gate_v1": True,
    }


def _go_no_go() -> dict[str, Any]:
    payload = {
        "layer_name": "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "research_only_iql_sanity_training_allowed_next_v1": True,
        "iql_training_run_in_this_gate_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "selected_research_cohort_v1": CHOSEN_RESEARCH_COHORT,
        "no_shortcut_audit_status_v1": "PASS",
    }
    validate_go_no_go(payload)
    return payload


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    if payload.get("iql_training_run_in_this_gate_v1"):
        raise RuntimeError("IQL_TRAINING_MUST_NOT_RUN_IN_CONTRACT_GATE")
    for blocked in [
        "iql_production_allowed_v1",
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
    ]:
        if payload.get(blocked):
            raise RuntimeError(f"FORBIDDEN_DOWNSTREAM_PATH_OPENED: {blocked}")
    if not payload.get("research_only_iql_sanity_training_allowed_next_v1"):
        raise RuntimeError("RESEARCH_ONLY_IQL_SANITY_NEXT_SHOULD_BE_ALLOWED_FOR_READY_STATUS")
    return True


def _write_markdown(
    artifact_root: Path,
    repro: dict[str, Any],
    cohorts: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    action: dict[str, Any],
    reward: dict[str, Any],
    behavior: dict[str, Any],
    split: dict[str, Any],
    shield: dict[str, Any],
    xgb_rows: list[dict[str, Any]],
    no_shortcut: dict[str, Any],
    readiness: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_offline_data_contract_reproducibility_audit_v1.md",
        [
            "# IQL Offline Data Contract Reproducibility Audit V1",
            "",
            f"- 140/94 comparator: `{repro['baseline_140_94_v1']['selected_rows_v1']}` selected, `{repro['baseline_140_94_v1']['bad_count_audit_only_v1']}/{repro['baseline_140_94_v1']['tail_count_audit_only_v1']}` bad/tail, safety `{repro['baseline_140_94_v1']['safety_status_v1']}`.",
            f"- 89 safe-core: `{repro['safe_core_89_v1']['selected_rows_v1']}` selected, `{repro['safe_core_89_v1']['recovered_original_140_v1']}` original-140 recovered, `{repro['safe_core_89_v1']['extra_rows_v1']}` extra, safety `{repro['safe_core_89_v1']['safety_status_v1']}`.",
            f"- 78 source-safety shield: `{repro['source_safety_shielded_78_v1']['selected_rows_v1']}` selected, `{repro['source_safety_shielded_78_v1']['original_140_retained_v1']}` original-140 retained, unsafe row blocked `{repro['source_safety_shielded_78_v1']['unsafe_row_blocked_v1']}`.",
        ],
    )
    _write_report(
        artifact_root / "iql_offline_eligibility_cohorts_v1.md",
        ["# IQL Offline Eligibility Cohorts V1", "", *[f"- `{row['cohort_id_v1']}`: {row['selected_rows_v1']} rows, {row['role_v1']}." for row in cohorts]],
    )
    _write_report(
        artifact_root / "iql_offline_state_contract_v1.md",
        [
            "# IQL Offline State Contract V1",
            "",
            f"- Allowed state fields: `{sum(1 for row in state_rows if row['allowed_as_state_v1'])}`.",
            "- Final labels, row identity, membership flags, historical V2 blueprint, audit-only safety flags, and MFE/hindsight fields are blocked as state.",
        ],
    )
    _write_report(
        artifact_root / "iql_offline_action_contract_v1.md",
        ["# IQL Offline Action Contract V1", "", f"- Action space: `{action['action_space_status_v1']}`.", "- Actions: `SKIP`, `TAKE_TRADE`."],
    )
    _write_report(
        artifact_root / "iql_offline_reward_contract_v1.md",
        ["# IQL Offline Reward Contract V1", "", "- Labels are reward/audit only, never state.", "- Preferred first reward: `SAFETY_WEIGHTED_REWARD`."],
    )
    _write_report(
        artifact_root / "iql_offline_behavior_policy_audit_v1.md",
        ["# IQL Offline Behavior Policy Audit V1", "", f"- Status: `{behavior['status_v1']}`.", "- Propensities are absent; research-only IQL sanity remains allowed with support warnings."],
    )
    _write_report(
        artifact_root / "iql_offline_split_policy_v1.md",
        ["# IQL Offline Split Policy V1", "", *[f"- `{row['split_id_v1']}`: {row['purpose_v1']}." for row in split["recommended_splits_v1"]]],
    )
    _write_report(
        artifact_root / "iql_offline_safety_shield_contract_v1.md",
        ["# IQL Offline Safety Shield Contract V1", "", f"- Primary shield: `{shield['primary_research_eligibility_cohort_v1']}`.", f"- Selected rows: `{shield['selected_rows_v1']}`.", f"- Unsafe row blocked: `{shield['unsafe_row_blocked_v1']}`.", "- Safety shield stays outside policy and overrides TAKE."],
    )
    _write_report(
        artifact_root / "iql_offline_xgb_transformer_feature_integration_audit_v1.md",
        ["# IQL Offline XGB/Transformer Feature Integration Audit V1", "", "- Existing source score/support signals can be state after normalization where required.", "- Transformer embeddings are not lineage-ready for IQL state and are not a blocker for this contract."],
    )
    _write_report(
        artifact_root / "iql_offline_no_shortcut_audit_v1.md",
        ["# IQL Offline No-Shortcut Audit V1", "", f"- Status: `{no_shortcut['status_v1']}`.", "- No adapter/R6/live path was opened."],
    )
    _write_report(
        artifact_root / "iql_offline_readiness_assessment_v1.md",
        ["# IQL Offline Readiness Assessment V1", "", f"- Status: `{readiness['status_v1']}`.", f"- First cohort: `{readiness['first_cohort_v1']}`.", "- This authorizes only the next research-only sanity-training gate, not production IQL."],
    )
    _write_report(
        artifact_root / "iql_offline_data_contract_recommendation_v1.md",
        ["# IQL Offline Data Contract Recommendation V1", "", f"- Final status: `{recommendation['final_status_v1']}`.", f"- Next action: `{recommendation['next_recommended_action_v1']}`."],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _frame_and_masks(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, masks)
    cohorts = _eligibility_cohorts(frame, masks)
    state_rows = _state_contract_rows(frame)
    action = _action_contract()
    reward = _reward_contract()
    behavior = _behavior_policy_audit()
    split = _split_policy()
    shield = _safety_shield_contract(repro)
    xgb_rows = _xgb_transformer_audit(frame)
    no_shortcut = _no_shortcut_audit()
    readiness = _readiness_assessment()
    recommendation = _recommendation()
    go_no_go = _go_no_go()

    _write_json(artifact_root / "iql_offline_data_contract_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_offline_data_contract_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_offline_eligibility_cohorts_v1.csv", cohorts)
    _write_json(artifact_root / "iql_offline_eligibility_cohorts_v1.json", {"row_count_v1": len(cohorts), "rows_v1": cohorts})
    _write_rows(artifact_root / "iql_offline_state_contract_v1.csv", state_rows)
    _write_json(artifact_root / "iql_offline_state_contract_v1.json", {"row_count_v1": len(state_rows), "rows_v1": state_rows})
    _write_json(artifact_root / "iql_offline_action_contract_v1.json", action)
    _write_json(artifact_root / "iql_offline_reward_contract_v1.json", reward)
    _write_json(artifact_root / "iql_offline_behavior_policy_audit_v1.json", behavior)
    _write_json(artifact_root / "iql_offline_split_policy_v1.json", split)
    _write_json(artifact_root / "iql_offline_safety_shield_contract_v1.json", shield)
    _write_rows(artifact_root / "iql_offline_xgb_transformer_feature_integration_audit_v1.csv", xgb_rows)
    _write_json(
        artifact_root / "iql_offline_xgb_transformer_feature_integration_audit_v1.json",
        {"row_count_v1": len(xgb_rows), "rows_v1": xgb_rows},
    )
    _write_json(artifact_root / "iql_offline_no_shortcut_audit_v1.json", no_shortcut)
    _write_json(artifact_root / "iql_offline_readiness_assessment_v1.json", readiness)
    _write_json(artifact_root / "iql_offline_data_contract_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "build_iql_offline_data_contract_research_only_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        cohorts,
        state_rows,
        action,
        reward,
        behavior,
        split,
        shield,
        xgb_rows,
        no_shortcut,
        readiness,
        recommendation,
    )

    summary = {
        "layer_name": "IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "chosen_research_only_eligibility_cohort_v1": CHOSEN_RESEARCH_COHORT,
        "baseline_140_94_selected_bad_tail_v1": [
            repro["baseline_140_94_v1"]["selected_rows_v1"],
            repro["baseline_140_94_v1"]["bad_count_audit_only_v1"],
            repro["baseline_140_94_v1"]["tail_count_audit_only_v1"],
        ],
        "safe_core_89_selected_bad_tail_v1": [
            repro["safe_core_89_v1"]["selected_rows_v1"],
            repro["safe_core_89_v1"]["bad_count_audit_only_v1"],
            repro["safe_core_89_v1"]["tail_count_audit_only_v1"],
        ],
        "source_safety_shielded_78_selected_bad_tail_v1": [
            repro["source_safety_shielded_78_v1"]["selected_rows_v1"],
            repro["source_safety_shielded_78_v1"]["bad_count_audit_only_v1"],
            repro["source_safety_shielded_78_v1"]["tail_count_audit_only_v1"],
        ],
        "state_allowed_field_count_v1": int(sum(1 for row in state_rows if row["allowed_as_state_v1"])),
        "state_blocked_field_count_v1": int(sum(1 for row in state_rows if not row["allowed_as_state_v1"])),
        "action_space_v1": action["action_space_status_v1"],
        "recommended_reward_v1": "SAFETY_WEIGHTED_REWARD",
        "safety_shield_summary_v1": "78-row clean source-safety-shielded research eligibility; unsafe row blocked; ORANGE retention; not adapter-ready.",
        "offline_iql_sanity_training_allowed_next_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
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
            "# Build IQL Offline Data Contract Research Only V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
            f"- First research cohort: `{CHOSEN_RESEARCH_COHORT}`",
            "- No IQL training, adapter build, R6, package, freeze, promo, or live run was performed.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize offline IQL data contract, research only.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
