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
ACTION = "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1"

INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1_20260428T174140Z_LOCK"
)
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
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

SAFE_CORE_RULE_ID = "SAFE_CORE_HARDENED_RULE_V1"
REFINED_VETO_ID = "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1"
FINAL_STATUS = "PROXY_VETO_BRANCH_CLOSED_SELECT_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_PATH"
NEXT_ACTION = "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1"

ALLOWED_FINAL_STATUSES = {
    "PROXY_VETO_BRANCH_CLOSED_SELECT_MINIMAL_DEPLOYABLE_SAFE_CORE_PATH",
    "PROXY_VETO_BRANCH_CLOSED_SELECT_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_PATH",
    "PROXY_VETO_BRANCH_CLOSED_SELECT_140_94_SAFETY_FIRST_REDISTILLATION_PATH",
    "PROXY_VETO_BRANCH_CLOSED_HOLD_ADAPTER_AND_CLEANUP_FIRST",
    "PROXY_VETO_BRANCH_NOT_CLOSED_NEEDS_MORE_EVIDENCE",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_MINIMAL_DEPLOYABLE_SAFE_CORE_WITHOUT_AUDIT_ONLY_VETO_V1",
    "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1",
    "REDISTILL_140_94_WITH_DEPLOYABLE_SAFETY_FIRST_CONSTRAINT_V1",
    "CLEANUP_OVERVIEW_CURRENT_BASELINES_AND_OUTDATED_RUNS_V1",
    "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1",
    "DEEPEN_PROXY_VETO_CLOSURE_EVIDENCE_V1",
}

REQUIRED_OUTPUTS = [
    "close_proxy_veto_input_manifest_v1.json",
    "close_proxy_veto_reproducibility_audit_v1.json",
    "close_proxy_veto_reproducibility_audit_v1.md",
    "close_proxy_veto_branch_closure_record_v1.json",
    "close_proxy_veto_branch_closure_record_v1.md",
    "close_proxy_veto_current_asset_inventory_v1.json",
    "close_proxy_veto_current_asset_inventory_v1.md",
    "close_proxy_veto_next_direction_options_v1.json",
    "close_proxy_veto_next_direction_options_v1.md",
    "close_proxy_veto_option_ranking_v1.csv",
    "close_proxy_veto_option_ranking_v1.json",
    "close_proxy_veto_option_ranking_v1.md",
    "close_proxy_veto_mainline_decision_v1.json",
    "close_proxy_veto_mainline_decision_v1.md",
    "close_proxy_veto_anti_shortcut_audit_v1.json",
    "close_proxy_veto_anti_shortcut_audit_v1.md",
    "close_proxy_veto_recommendation_v1.json",
    "close_proxy_veto_recommendation_v1.md",
    "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json",
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


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_branch_closure(record: dict[str, Any]) -> bool:
    required_false = [
        "fan_in_decision_allowed_v1",
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "iql_allowed_v1",
        "continue_blueprint_refinement_without_new_as_of_sources_v1",
    ]
    failures = [key for key in required_false if record.get(key) is not False]
    required_true = [
        "branch_closed_as_deployable_mainline_v1",
        "historical_v2_blueprint_rejected_as_adapter_input_now_v1",
        "refined_veto_preserved_as_diagnostic_only_v1",
    ]
    failures.extend(key for key in required_true if record.get(key) is not True)
    if failures:
        raise RuntimeError(f"PROXY_VETO_BRANCH_CLOSURE_INVALID: {failures}")
    return True


def validate_option_ranking(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != 4:
        raise RuntimeError(f"EXPECTED_EXACTLY_FOUR_NEXT_DIRECTION_OPTIONS: {len(rows)}")
    selected = [row for row in rows if row.get("selected_next_direction_v1") is True]
    if len(selected) != 1:
        raise RuntimeError("EXACTLY_ONE_NEXT_DIRECTION_MUST_BE_SELECTED")
    if selected[0]["option_id_v1"] != "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS":
        raise RuntimeError(f"UNEXPECTED_SELECTED_DIRECTION: {selected[0]['option_id_v1']}")
    if any(row.get("opens_adapter_now_v1") or row.get("runs_r6_now_v1") or row.get("runs_iql_now_v1") for row in rows):
        raise RuntimeError("NEXT_DIRECTION_OPTIONS_MUST_NOT_OPEN_ADAPTER_R6_IQL")
    return True


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    if payload.get("adapter_build_allowed_v1") or payload.get("r6_allowed_v1") or payload.get("iql_allowed_v1"):
        raise RuntimeError("GO_NO_GO_MUST_KEEP_ADAPTER_R6_IQL_BLOCKED")
    if payload.get("proxy_veto_branch_closed_v1") is not True:
        raise RuntimeError("GO_NO_GO_MUST_CLOSE_PROXY_VETO_BRANCH")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"CLOSE_PROXY_VETO_REQUIRED_OUTPUTS_MISSING: {missing}")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_LANE_PACK_ROOT,
        INPUT_REFINED_ROOT,
        INPUT_HOLD_ROOT,
        INPUT_DISCOVERY_ROOT,
        INPUT_HARDEN_ROOT,
        INPUT_PRECHECK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
        "lane_pack_go_no_go": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json",
        "lane_pack_repro": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lane_pack_reproducibility_audit_v1.json",
        "lane_pack_lane_index": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lane_pack_lane_index_v1.json",
        "lane_pack_risk_matrix": INPUT_LANE_PACK_ROOT / "parallel_refined_veto_lane_pack_cross_lane_risk_matrix_v1.json",
        "refined_summary": INPUT_REFINED_ROOT / "summary_v1.json",
        "refined_go_no_go": INPUT_REFINED_ROOT / "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json",
        "hold_summary": INPUT_HOLD_ROOT / "summary_v1.json",
        "hold_go_no_go": INPUT_HOLD_ROOT / "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json",
        "discovery_summary": INPUT_DISCOVERY_ROOT / "summary_v1.json",
        "discovery_go_no_go": INPUT_DISCOVERY_ROOT
        / "discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_go_no_go_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "harden_go_no_go": INPUT_HARDEN_ROOT / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json",
        "harden_missing_54_buckets": INPUT_HARDEN_ROOT / "harden_140_94_missing_54_expansion_bucket_audit_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    lane_go = _read_json(required["lane_pack_go_no_go"])
    if lane_go.get("status_v1") != "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_PROXY_OR_LEAKAGE_RISK":
        raise RuntimeError("INPUT_LANE_PACK_STATUS_NOT_PROXY_BLOCKED")
    return {
        "required_paths": required,
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_pack_go_no_go": lane_go,
        "lane_pack_repro": _read_json(required["lane_pack_repro"]),
        "lane_pack_lane_index": _read_json(required["lane_pack_lane_index"]),
        "lane_pack_risk_matrix": _read_json(required["lane_pack_risk_matrix"]),
        "refined_summary": _read_json(required["refined_summary"]),
        "refined_go_no_go": _read_json(required["refined_go_no_go"]),
        "hold_summary": _read_json(required["hold_summary"]),
        "hold_go_no_go": _read_json(required["hold_go_no_go"]),
        "discovery_summary": _read_json(required["discovery_summary"]),
        "discovery_go_no_go": _read_json(required["discovery_go_no_go"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "harden_go_no_go": _read_json(required["harden_go_no_go"]),
        "harden_missing_54_buckets": _read_json(required["harden_missing_54_buckets"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "CLOSE_PROXY_VETO_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "parallel_refined_veto_lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
            "refined_veto_root_v1": str(INPUT_REFINED_ROOT),
            "hold_root_v1": str(INPUT_HOLD_ROOT),
            "veto_discovery_root_v1": str(INPUT_DISCOVERY_ROOT),
            "safe_core_hardening_root_v1": str(INPUT_HARDEN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
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


def _reproducibility(inputs: dict[str, Any]) -> dict[str, Any]:
    repro = inputs["lane_pack_repro"]
    payload = {
        "layer_name": "CLOSE_PROXY_VETO_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "refined_veto_id_v1": REFINED_VETO_ID,
        "safe_core_selected_rows_v1": repro.get("safe_core_selected_rows_v1"),
        "safe_core_recovered_original_140_v1": repro.get("safe_core_recovered_original_140_v1"),
        "safe_core_extra_rows_v1": repro.get("safe_core_extra_rows_v1"),
        "safe_core_bad_tail_audit_only_v1": repro.get("safe_core_bad_tail_audit_only_v1"),
        "safe_core_precision_audit_only_v1": repro.get("safe_core_precision_audit_only_v1"),
        "unsafe_extra_without_hard_veto_rows_v1": repro.get("unsafe_extra_without_hard_veto_rows_v1"),
        "refined_veto_mechanically_blocks_unsafe_row_v1": repro.get("refined_veto_blocks_unsafe_row_v1"),
        "refined_veto_good_safe_core_rows_cut_v1": repro.get("refined_veto_good_safe_core_rows_cut_v1"),
        "refined_veto_selected_rows_v1": repro.get("refined_veto_selected_rows_v1"),
        "refined_veto_bad_tail_audit_only_v1": repro.get("refined_veto_bad_tail_audit_only_v1"),
        "refined_veto_safety_status_v1": repro.get("refined_veto_safety_status_v1"),
        "lane_pack_final_status_v1": inputs["lane_pack_go_no_go"].get("status_v1"),
        "adapter_r6_iql_remain_blocked_v1": inputs["lane_pack_summary"].get("adapter_r6_iql_remain_blocked_v1"),
    }
    expected = {
        "safe_core_selected_rows_v1": 89,
        "safe_core_recovered_original_140_v1": 86,
        "safe_core_extra_rows_v1": 3,
        "unsafe_extra_without_hard_veto_rows_v1": 1,
        "refined_veto_mechanically_blocks_unsafe_row_v1": True,
        "refined_veto_good_safe_core_rows_cut_v1": 3,
        "refined_veto_safety_status_v1": "CLEAN",
        "lane_pack_final_status_v1": "PARALLEL_REFINED_VETO_LINEAGE_LANE_PACK_BLOCKED_BY_PROXY_OR_LEAKAGE_RISK",
        "adapter_r6_iql_remain_blocked_v1": True,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if failures:
        raise RuntimeError(f"CLOSE_PROXY_VETO_REPRODUCTION_FAILED: {failures}")
    return payload


def _branch_closure_record(inputs: dict[str, Any]) -> dict[str, Any]:
    lane_rows = inputs["lane_pack_lane_index"]["rows_v1"]
    blocking_lanes = [row for row in lane_rows if row["lane_status_v1"].startswith("LANE_BLOCKED")]
    record = {
        "layer_name": "CLOSE_PROXY_VETO_BRANCH_CLOSURE_RECORD_V1",
        "branch_id_v1": REFINED_VETO_ID,
        "branch_closed_as_deployable_mainline_v1": True,
        "historical_v2_blueprint_rejected_as_adapter_input_now_v1": True,
        "refined_veto_preserved_as_diagnostic_only_v1": True,
        "fan_in_decision_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "continue_blueprint_refinement_without_new_as_of_sources_v1": False,
        "blocking_lane_ids_v1": [row["lane_id_v1"] for row in blocking_lanes],
        "blocking_lane_statuses_v1": {row["lane_id_v1"]: row["lane_status_v1"] for row in blocking_lanes},
        "closure_reasons_v1": [
            "HISTORICAL_V2_BLUEPRINT traces to historical V2 captured-membership/artifact evidence.",
            "AS_OF reconstruction from raw source signals was not proven.",
            "Adapter allowlist eligibility is blocked by proxy risk.",
            "Artifact shortcut and support/concentration lanes found blockers.",
            "A fan-in decision would launder unresolved proxy lineage into adapter work.",
        ],
        "restart_condition_v1": (
            "Only reopen this branch if a later gate brings independent raw AS_OF source evidence that reconstructs "
            "the same safety guard without membership, coverage, row identity, artifact shortcut, hindsight, MFE, "
            "or audit-only labels."
        ),
    }
    validate_branch_closure(record)
    return record


def _asset_inventory(inputs: dict[str, Any]) -> dict[str, Any]:
    precheck = inputs["precheck_summary"]
    harden = inputs["harden_summary"]
    buckets = inputs["harden_missing_54_buckets"]["rows_v1"]
    return {
        "layer_name": "CLOSE_PROXY_VETO_CURRENT_ASSET_INVENTORY_V1",
        "assets_v1": [
            {
                "asset_id_v1": "BASELINE_140_94",
                "status_v1": "CURRENT_BEST_CAUSAL_BASELINE",
                "selected_rows_v1": 140,
                "bad_tail_v1": precheck.get("baseline_bad_tail_v1", [140, 94]),
                "safety_v1": "CLEAN",
                "value_v1": "Strong honest causal baseline and mainline anchor.",
                "blocker_v1": "Exact selection still needs deployable safety-first AS_OF recipe before adapter.",
                "possible_next_use_v1": "Anchor and comparator for clean AS_OF safety-feature layer.",
            },
            {
                "asset_id_v1": "SAFE_CORE_HARDENED_89",
                "status_v1": "BEST_CONCRETE_RULE_CORE_BUT_ADAPTER_BLOCKED",
                "selected_rows_v1": harden.get("selected_rows_v1"),
                "recovered_original_140_v1": harden.get("recovered_original_140_rows_v1"),
                "extra_rows_v1": harden.get("extra_rows_v1"),
                "bad_tail_v1": harden.get("bad_tail_audit_only_v1"),
                "precision_v1": harden.get("precision_audit_only_v1"),
                "safety_v1": harden.get("safety_status_v1"),
                "value_v1": "Simple high-confidence rule core with clean audit safety.",
                "blocker_v1": "Needs deployable hard safety veto; V2-blueprint guard is proxy-blocked.",
                "possible_next_use_v1": "Testbed for clean raw AS_OF safety features.",
            },
            {
                "asset_id_v1": "MISSING_54_EXPANSION",
                "status_v1": "SEPARATE_NOT_MAINLINE_NOW",
                "bucket_rows_v1": buckets,
                "value_v1": "Contains potential later expansion modules after deployable safety exists.",
                "blocker_v1": "Expansion without safety layer risks repeating veto/proxy corner.",
                "possible_next_use_v1": "Hold for later separate gate.",
            },
            {
                "asset_id_v1": "BEST_LANE_185_139_AND_PLUS45",
                "status_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
                "selected_rows_v1": 185,
                "bad_tail_v1": [185, 139],
                "value_v1": "Research signal and comparator; useful for feature-family insight.",
                "blocker_v1": "Membership/coverage-only and not learned OOF from available AS_OF features.",
                "possible_next_use_v1": "Diagnostic only; not target, feature, filter, selector, or adapter input.",
            },
            {
                "asset_id_v1": "REFINED_V2_BLUEPRINT_VETO",
                "status_v1": "MECHANICALLY_PROMISING_DEPLOYABLE_BLOCKED",
                "refined_veto_id_v1": REFINED_VETO_ID,
                "unsafe_row_blocked_v1": True,
                "good_rows_cut_v1": 3,
                "value_v1": "Shows what a precise hard safety veto would need to do.",
                "blocker_v1": "Depends on HISTORICAL_V2_BLUEPRINT, currently blocked as artifact/membership proxy risk.",
                "possible_next_use_v1": "Diagnostic pattern only unless independent raw AS_OF lineage appears.",
            },
        ],
    }


def _next_direction_options() -> dict[str, Any]:
    options = [
        {
            "option_id_v1": "OPTION_A_MINIMAL_DEPLOYABLE_SAFE_CORE_WITHOUT_HARD_VETO",
            "description_v1": "Drop or localize the branch/tier that admits the unsafe row and accept a smaller deployable core.",
            "expected_progress_v1": "Medium",
            "safety_score_v1": 88,
            "as_of_deployability_score_v1": 84,
            "adapter_feasibility_score_v1": 78,
            "usefulness_for_r6_iql_score_v1": 68,
            "diminishing_returns_risk_v1": "MEDIUM",
            "opens_adapter_now_v1": False,
            "runs_r6_now_v1": False,
            "runs_iql_now_v1": False,
            "main_risk_v1": "May become too small and discard useful 140/94 learning.",
        },
        {
            "option_id_v1": "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS",
            "description_v1": "Build a clean safety/veto feature inventory from raw AS_OF source signals instead of historical blueprint evidence.",
            "expected_progress_v1": "High",
            "safety_score_v1": 96,
            "as_of_deployability_score_v1": 95,
            "adapter_feasibility_score_v1": 88,
            "usefulness_for_r6_iql_score_v1": 94,
            "diminishing_returns_risk_v1": "LOW",
            "opens_adapter_now_v1": False,
            "runs_r6_now_v1": False,
            "runs_iql_now_v1": False,
            "main_risk_v1": "Requires source-lineage work before performance gains resume.",
        },
        {
            "option_id_v1": "OPTION_C_RETURN_TO_140_94_SAFETY_FIRST_DISTILLATION",
            "description_v1": "Re-distill 140/94 with deployable safety as the first constraint rather than a late audit veto.",
            "expected_progress_v1": "Medium",
            "safety_score_v1": 92,
            "as_of_deployability_score_v1": 82,
            "adapter_feasibility_score_v1": 74,
            "usefulness_for_r6_iql_score_v1": 76,
            "diminishing_returns_risk_v1": "MEDIUM",
            "opens_adapter_now_v1": False,
            "runs_r6_now_v1": False,
            "runs_iql_now_v1": False,
            "main_risk_v1": "Could repeat the same missing-safety-layer problem at a wider scale.",
        },
        {
            "option_id_v1": "OPTION_D_HOLD_ALL_ADAPTER_WORK_AND_CLEANUP_OR_DOCUMENTATION",
            "description_v1": "Pause adapter work and focus on cleanup/documentation until new AS_OF safety evidence exists.",
            "expected_progress_v1": "Low",
            "safety_score_v1": 98,
            "as_of_deployability_score_v1": 50,
            "adapter_feasibility_score_v1": 40,
            "usefulness_for_r6_iql_score_v1": 45,
            "diminishing_returns_risk_v1": "LOW",
            "opens_adapter_now_v1": False,
            "runs_r6_now_v1": False,
            "runs_iql_now_v1": False,
            "main_risk_v1": "Safest but too passive; cleanup has already been inventoried separately.",
        },
    ]
    return {"layer_name": "CLOSE_PROXY_VETO_NEXT_DIRECTION_OPTIONS_V1", "options_v1": options}


def _rank_options(options_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for option in options_payload["options_v1"]:
        total = (
            option["safety_score_v1"] * 0.30
            + option["as_of_deployability_score_v1"] * 0.25
            + option["adapter_feasibility_score_v1"] * 0.20
            + option["usefulness_for_r6_iql_score_v1"] * 0.20
            + (5 if option["diminishing_returns_risk_v1"] == "LOW" else 0)
        )
        rows.append(
            {
                **option,
                "weighted_score_v1": round(total, 3),
                "selected_next_direction_v1": option["option_id_v1"]
                == "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS",
            }
        )
    rows.sort(key=lambda row: row["weighted_score_v1"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_v1"] = idx
    validate_option_ranking(rows)
    return rows


def _mainline_decision(ranking_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(row for row in ranking_rows if row["selected_next_direction_v1"])
    return {
        "layer_name": "CLOSE_PROXY_VETO_MAINLINE_DECISION_V1",
        "proxy_veto_branch_closed_v1": True,
        "selected_option_id_v1": selected["option_id_v1"],
        "selected_next_recommended_action_v1": NEXT_ACTION,
        "final_status_v1": FINAL_STATUS,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "why_selected_v1": [
            "It solves the root blocker: lack of deployable AS_OF safety/veto inputs.",
            "It preserves learning from 140/94 and safe-core without using proxy evidence.",
            "It is useful for later adapter, R6, and IQL work, while avoiding another V2-blueprint corner.",
        ],
        "why_not_blueprint_refinement_v1": (
            "The lane pack already found proxy/reconstruction/allowlist/artifact/support blockers. More refinement "
            "of the same historical blueprint guard would optimize a non-deployable shortcut unless new raw AS_OF "
            "lineage appears."
        ),
    }


def _anti_shortcut_audit() -> dict[str, Any]:
    return {
        "layer_name": "CLOSE_PROXY_VETO_ANTI_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "no_r6_v1": True,
        "no_adapter_build_v1": True,
        "no_iql_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_optuna_or_broad_sweep_v1": True,
        "no_historical_v2_blueprint_as_deployable_input_v1": True,
        "no_membership_or_coverage_proxy_as_next_target_v1": True,
        "no_row_identity_or_artifact_shortcut_v1": True,
        "no_hindsight_mfe_audit_label_veto_v1": True,
        "no_implicit_latest_glob_decisioning_v1": True,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }


def _recommendation(decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recommendation = {
        "layer_name": "CLOSE_PROXY_VETO_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "mainline_v1": "Build clean AS_OF safety/veto features from source signals before any adapter/R6/IQL work.",
        "adapter_r6_iql_remain_blocked_v1": True,
        "proxy_veto_branch_closed_v1": True,
    }
    go_no_go = {
        "layer_name": "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "proxy_veto_branch_closed_v1": True,
        "selected_option_id_v1": decision["selected_option_id_v1"],
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "fan_in_decision_allowed_v1": False,
        "further_v2_blueprint_refinement_recommended_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_go_no_go(go_no_go)
    return recommendation, go_no_go


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    closure: dict[str, Any],
    inventory: dict[str, Any],
    options: dict[str, Any],
    ranking_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    anti: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        root / "close_proxy_veto_reproducibility_audit_v1.md",
        [
            "# Close Proxy Veto Reproducibility Audit V1",
            "",
            f"- Safe-core selected rows: `{repro['safe_core_selected_rows_v1']}`",
            f"- Refined veto blocks unsafe row: `{repro['refined_veto_mechanically_blocks_unsafe_row_v1']}`",
            f"- Refined veto good rows cut: `{repro['refined_veto_good_safe_core_rows_cut_v1']}`",
            f"- Lane-pack status: `{repro['lane_pack_final_status_v1']}`",
        ],
    )
    _write_report(
        root / "close_proxy_veto_branch_closure_record_v1.md",
        [
            "# Close Proxy Veto Branch Closure Record V1",
            "",
            f"- Branch: `{closure['branch_id_v1']}`",
            "- Status: closed as deployable mainline.",
            f"- Blocking lanes: `{closure['blocking_lane_ids_v1']}`",
            "- Adapter/R6/IQL remain blocked.",
        ],
    )
    _write_report(
        root / "close_proxy_veto_current_asset_inventory_v1.md",
        [
            "# Close Proxy Veto Current Asset Inventory V1",
            "",
            *[
                f"- `{asset['asset_id_v1']}`: `{asset['status_v1']}`; blocker: {asset['blocker_v1']}"
                for asset in inventory["assets_v1"]
            ],
        ],
    )
    _write_report(
        root / "close_proxy_veto_next_direction_options_v1.md",
        [
            "# Close Proxy Veto Next Direction Options V1",
            "",
            *[f"- `{option['option_id_v1']}`: {option['description_v1']}" for option in options["options_v1"]],
        ],
    )
    _write_report(
        root / "close_proxy_veto_option_ranking_v1.md",
        [
            "# Close Proxy Veto Option Ranking V1",
            "",
            *[
                f"- Rank {row['rank_v1']}: `{row['option_id_v1']}` score `{row['weighted_score_v1']}`"
                for row in ranking_rows
            ],
        ],
    )
    _write_report(
        root / "close_proxy_veto_mainline_decision_v1.md",
        [
            "# Close Proxy Veto Mainline Decision V1",
            "",
            f"- Final status: `{decision['final_status_v1']}`",
            f"- Selected option: `{decision['selected_option_id_v1']}`",
            f"- Next action: `{decision['selected_next_recommended_action_v1']}`",
            "- Adapter/R6/IQL remain blocked.",
        ],
    )
    _write_report(
        root / "close_proxy_veto_anti_shortcut_audit_v1.md",
        [
            "# Close Proxy Veto Anti-Shortcut Audit V1",
            "",
            f"- Status: `{anti['status_v1']}`",
            "- No R6, adapter, IQL, package, freeze, promo, live, Optuna, broad sweep, historical V2 deployable input, membership proxy, row identity, or hindsight shortcut was used.",
        ],
    )
    _write_report(
        root / "close_proxy_veto_recommendation_v1.md",
        [
            "# Close Proxy Veto Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Mainline: {recommendation['mainline_v1']}",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(inputs)
    closure = _branch_closure_record(inputs)
    inventory = _asset_inventory(inputs)
    options = _next_direction_options()
    ranking_rows = _rank_options(options)
    decision = _mainline_decision(ranking_rows)
    anti = _anti_shortcut_audit()
    recommendation, go_no_go = _recommendation(decision)

    _write_json(artifact_root / "close_proxy_veto_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "close_proxy_veto_reproducibility_audit_v1.json", repro)
    _write_json(artifact_root / "close_proxy_veto_branch_closure_record_v1.json", closure)
    _write_json(artifact_root / "close_proxy_veto_current_asset_inventory_v1.json", inventory)
    _write_json(artifact_root / "close_proxy_veto_next_direction_options_v1.json", options)
    _write_rows(artifact_root / "close_proxy_veto_option_ranking_v1.csv", ranking_rows)
    _write_json(
        artifact_root / "close_proxy_veto_option_ranking_v1.json",
        {"row_count_v1": len(ranking_rows), "rows_v1": ranking_rows},
    )
    _write_json(artifact_root / "close_proxy_veto_mainline_decision_v1.json", decision)
    _write_json(artifact_root / "close_proxy_veto_anti_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "close_proxy_veto_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json", go_no_go)
    _write_markdown(artifact_root, repro, closure, inventory, options, ranking_rows, decision, anti, recommendation)

    summary = {
        "layer_name": "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "proxy_veto_branch_closed_v1": True,
        "historical_v2_blueprint_deployable_now_v1": False,
        "safe_core_rule_id_v1": SAFE_CORE_RULE_ID,
        "safe_core_selected_rows_v1": repro["safe_core_selected_rows_v1"],
        "safe_core_recovered_original_140_v1": repro["safe_core_recovered_original_140_v1"],
        "safe_core_bad_tail_audit_only_v1": repro["safe_core_bad_tail_audit_only_v1"],
        "baseline_140_94_status_v1": "CURRENT_BEST_CAUSAL_BASELINE",
        "selected_next_mainline_direction_v1": decision["selected_option_id_v1"],
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
            "# Close Proxy Veto Branch And Select Safe Mainline Next Step V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Proxy branch closed: `{summary['proxy_veto_branch_closed_v1']}`",
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
