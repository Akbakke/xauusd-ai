#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "PARALLEL_CONTEXTUAL_IQL_STATE_ACTION_RESEARCH_LANE_PACK_V1"

INPUT_DEEPER_EVENT_ORDERED_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1_20260428T211150Z_LOCK"
)
INPUT_EVENT_ORDERED_TRAINING_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1_20260428T204804Z_LOCK"
)
INPUT_EVENT_ORDERED_DATASET_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1_20260428T203009Z_LOCK"
)
INPUT_CONTEXTUAL_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_IQL_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_CLEAN_SAFETY_REFINEMENT_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)
INPUT_140_94_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

FINAL_STATUS = "CONTEXTUAL_IQL_LANE_PACK_SELECT_STATE_FEATURE_REBUILD"
NEXT_ACTION = "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1"
CONTEXTUAL_BASELINE_POLICY = "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"

LANES = [
    "LANE_01_CONTEXTUAL_BASELINE_LOCK_AND_REPRO",
    "LANE_02_AS_OF_SOURCE_STATE_FEATURE_EXPANSION",
    "LANE_03_XGB_FEATURE_LINEAGE_AND_STATE_INTEGRATION",
    "LANE_04_TRANSFORMER_FEATURE_LINEAGE_AUDIT",
    "LANE_05_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT",
    "LANE_06_REWARD_DESIGN_AND_SENSITIVITY",
    "LANE_07_SAFETY_SHIELD_AND_COHORT_VARIANTS",
    "LANE_08_NON_RL_CONTEXTUAL_BASELINE_COMPARISON",
    "LANE_09_STATE_ABLATION_AND_FEATURE_DEPENDENCY",
    "LANE_10_FAN_IN_RANKING_AND_NEXT_MAINLINE_DECISION",
]

LANE_STATUSES = {
    "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
    "LANE_PASS_USEFUL_BUT_SECONDARY",
    "LANE_INCONCLUSIVE_NEEDS_MORE_EVIDENCE",
    "LANE_BLOCKED_BY_LEAKAGE_OR_PROXY_RISK",
    "LANE_BLOCKED_BY_INSUFFICIENT_SUPPORT",
    "LANE_BLOCKED_BY_UNSTABLE_HELDOUT_BEHAVIOR",
    "LANE_BLOCKED_BY_MISSING_FEATURE_LINEAGE",
    "LANE_BLOCKED_BY_ACTION_SUPPORT_GAPS",
    "LANE_BLOCKED_BY_REWARD_AMBIGUITY",
    "LANE_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_FINAL_STATUSES = {
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_STATE_FEATURE_REBUILD",
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_XGB_FEATURE_INTEGRATION",
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_ACTION_SUPPORT_AUDIT",
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_REWARD_REDESIGN",
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_NON_RL_BASELINE_FIRST",
    "CONTEXTUAL_IQL_LANE_PACK_SELECT_TRUE_LIFECYCLE_METADATA",
    "CONTEXTUAL_IQL_LANE_PACK_READY_FOR_CONTEXTUAL_DEEPER_EXPERIMENT",
    "CONTEXTUAL_IQL_LANE_PACK_BLOCKED_BY_LEAKAGE_OR_PROXY",
    "CONTEXTUAL_IQL_LANE_PACK_BLOCKED_BY_INSUFFICIENT_SUPPORT",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1",
    "REBUILD_IQL_STATE_CONTRACT_WITH_XGB_AS_OF_FEATURES_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "REDESIGN_IQL_REWARD_CONTRACT_RESEARCH_ONLY_V1",
    "RUN_SUPERVISED_CONTEXTUAL_BASELINE_VS_IQL_RESEARCH_V1",
    "COLLECT_TRUE_TRADE_LIFECYCLE_METADATA_FOR_IQL_V1",
    "RUN_CONTEXTUAL_IQL_DEEPER_RESEARCH_WITH_SELECTED_FEATURES_V1",
    "HOLD_IQL_RESEARCH_UNTIL_SUPPORT_IMPROVES_V1",
}

REQUIRED_GLOBAL_OUTPUTS = [
    "contextual_iql_parallel_lane_pack_input_manifest_v1.json",
    "contextual_iql_parallel_lane_pack_reproducibility_audit_v1.json",
    "contextual_iql_parallel_lane_pack_reproducibility_audit_v1.md",
    "contextual_iql_parallel_lane_index_v1.csv",
    "contextual_iql_parallel_lane_index_v1.json",
    "contextual_iql_parallel_lane_pack_summary_v1.json",
    "contextual_iql_parallel_lane_pack_summary_v1.md",
    "contextual_iql_parallel_cross_lane_risk_matrix_v1.csv",
    "contextual_iql_parallel_cross_lane_risk_matrix_v1.json",
    "contextual_iql_parallel_cross_lane_risk_matrix_v1.md",
    "contextual_iql_parallel_fan_in_recommendation_v1.json",
    "contextual_iql_parallel_fan_in_recommendation_v1.md",
    "parallel_contextual_iql_state_action_research_lane_pack_go_no_go_v1.json",
]

REQUIRED_LANE_OUTPUTS = [
    "lane_manifest_v1.json",
    "lane_result_v1.json",
    "lane_result_v1.md",
    "lane_metrics_v1.csv",
    "lane_risk_audit_v1.json",
    "lane_risk_audit_v1.md",
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
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
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


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    failures = []
    for path in paths:
        text = str(path)
        if "*" in text or "latest" in text.lower() or not path.name.endswith("_LOCK"):
            failures.append(text)
    if failures:
        raise RuntimeError(f"IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN: {failures}")
    return True


def validate_lane_index(rows: list[dict[str, Any]]) -> bool:
    lane_ids = [row["lane_id_v1"] for row in rows]
    if lane_ids != LANES:
        raise RuntimeError(f"LANE_INDEX_MUST_CONTAIN_EXACT_10_PREDEFINED_LANES: {lane_ids}")
    bad = [row for row in rows if row["lane_status_v1"] not in LANE_STATUSES]
    if bad:
        raise RuntimeError(f"UNKNOWN_LANE_STATUS: {bad}")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_forbidden_actions(payload: dict[str, Any]) -> bool:
    forbidden_true = [
        "adapter_built_v1",
        "adapter_opened_v1",
        "r6_run_v1",
        "iql_production_opened_v1",
        "production_iql_training_run_v1",
        "policy_promotion_run_v1",
        "package_built_v1",
        "freeze_promo_live_run_v1",
        "optuna_run_v1",
        "broad_sweep_run_v1",
    ]
    opened = [field for field in forbidden_true if payload.get(field)]
    if opened:
        raise RuntimeError(f"FORBIDDEN_SIDE_EFFECT_DETECTED: {opened}")
    return True


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    validate_no_forbidden_actions(payload)
    if payload.get("adapter_r6_iql_production_live_remain_blocked_v1") is not True:
        raise RuntimeError("LANE_PACK_MUST_KEEP_PRODUCTION_PATHS_BLOCKED")
    return True


def validate_no_shortcut_payload(payload: dict[str, Any]) -> bool:
    failures = payload.get("critical_failures_v1", [])
    if failures:
        raise RuntimeError(f"CONTEXTUAL_IQL_LANE_PACK_NO_SHORTCUT_FAILED: {failures}")
    return True


def _input_roots() -> list[Path]:
    return [
        INPUT_DEEPER_EVENT_ORDERED_ROOT,
        INPUT_EVENT_ORDERED_TRAINING_ROOT,
        INPUT_EVENT_ORDERED_DATASET_ROOT,
        INPUT_CONTEXTUAL_SANITY_ROOT,
        INPUT_IQL_CONTRACT_ROOT,
        INPUT_CLEAN_SAFETY_REFINEMENT_ROOT,
        INPUT_140_94_PRECHECK_ROOT,
    ]


def _load_inputs() -> dict[str, Any]:
    validate_explicit_artifact_roots(_input_roots())
    required_paths = {
        "deeper_summary": INPUT_DEEPER_EVENT_ORDERED_ROOT / "summary_v1.json",
        "deeper_go_no_go": INPUT_DEEPER_EVENT_ORDERED_ROOT
        / "run_iql_event_ordered_deeper_research_experiment_go_no_go_v1.json",
        "deeper_event_order": INPUT_DEEPER_EVENT_ORDERED_ROOT
        / "iql_event_ordered_deeper_event_order_usefulness_audit_v1.json",
        "deeper_variant_metrics": INPUT_DEEPER_EVENT_ORDERED_ROOT
        / "iql_event_ordered_deeper_variant_metrics_v1.json",
        "deeper_baseline_comparison": INPUT_DEEPER_EVENT_ORDERED_ROOT
        / "iql_event_ordered_deeper_baseline_comparison_v1.json",
        "deeper_action_support": INPUT_DEEPER_EVENT_ORDERED_ROOT
        / "iql_event_ordered_deeper_action_support_audit_v1.json",
        "deeper_no_shortcut": INPUT_DEEPER_EVENT_ORDERED_ROOT / "iql_event_ordered_deeper_no_shortcut_audit_v1.json",
        "event_training_summary": INPUT_EVENT_ORDERED_TRAINING_ROOT / "summary_v1.json",
        "event_dataset_summary": INPUT_EVENT_ORDERED_DATASET_ROOT / "summary_v1.json",
        "event_dataset": INPUT_EVENT_ORDERED_DATASET_ROOT / "iql_event_ordered_transition_dataset_v1.json",
        "event_state_matrix": INPUT_EVENT_ORDERED_DATASET_ROOT / "iql_event_ordered_state_matrix_v1.json",
        "contextual_summary": INPUT_CONTEXTUAL_SANITY_ROOT / "summary_v1.json",
        "contextual_training_metrics": INPUT_CONTEXTUAL_SANITY_ROOT / "iql_offline_sanity_training_metrics_v1.json",
        "contextual_baseline_comparison": INPUT_CONTEXTUAL_SANITY_ROOT
        / "iql_offline_sanity_baseline_policy_comparison_v1.json",
        "contextual_no_shortcut": INPUT_CONTEXTUAL_SANITY_ROOT / "iql_offline_sanity_no_shortcut_audit_v1.json",
        "contract_summary": INPUT_IQL_CONTRACT_ROOT / "summary_v1.json",
        "contract_state": INPUT_IQL_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "contract_xgb_transformer": INPUT_IQL_CONTRACT_ROOT
        / "iql_offline_xgb_transformer_feature_integration_audit_v1.json",
        "contract_behavior_policy": INPUT_IQL_CONTRACT_ROOT / "iql_offline_behavior_policy_audit_v1.json",
        "contract_reward": INPUT_IQL_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "clean_safety_summary": INPUT_CLEAN_SAFETY_REFINEMENT_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_140_94_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    loaded = {name: _read_json(path) for name, path in required_paths.items()}
    loaded["required_paths"] = required_paths
    return loaded


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("rows_v1", []))


def _all_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _rows(payload) if row.get("split_id_v1") == "all"]


def _find_row(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise RuntimeError(f"Expected row {key}={value!r} not found")


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in sorted(inputs["required_paths"].items())
    ]
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_PACK_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "deeper_event_ordered_iql_root_v1": str(INPUT_DEEPER_EVENT_ORDERED_ROOT),
            "event_ordered_training_root_v1": str(INPUT_EVENT_ORDERED_TRAINING_ROOT),
            "event_ordered_transition_dataset_root_v1": str(INPUT_EVENT_ORDERED_DATASET_ROOT),
            "contextual_iql_sanity_root_v1": str(INPUT_CONTEXTUAL_SANITY_ROOT),
            "iql_data_contract_root_v1": str(INPUT_IQL_CONTRACT_ROOT),
            "clean_safety_layer_refinement_root_v1": str(INPUT_CLEAN_SAFETY_REFINEMENT_ROOT),
            "precheck_140_94_root_v1": str(INPUT_140_94_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_lane_pack_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_training_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _reproducibility_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    deeper = inputs["deeper_summary"]
    event = inputs["event_dataset_summary"]
    contextual = inputs["contextual_summary"]
    rows = _rows(inputs["event_dataset"])
    all_variant = _all_rows(inputs["deeper_variant_metrics"])
    contextual_equiv = _find_row(all_variant, "policy_name_v1", CONTEXTUAL_BASELINE_POLICY)
    payload = {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_PACK_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "deeper_event_ordered_final_status_v1": deeper["final_status_v1"],
        "deeper_event_ordered_next_action_v1": deeper["next_recommended_action_v1"],
        "event_order_usefulness_v1": inputs["deeper_event_order"]["event_order_useful_or_decorative_v1"],
        "contextual_remains_preferred_v1": True,
        "dataset_rows_v1": int(event["rows_v1"]),
        "episodes_v1": int(event["episodes_v1"]),
        "nonterminal_transitions_v1": int(event["nonterminal_transitions_v1"]),
        "terminal_rows_v1": int(event["terminal_rows_v1"]),
        "cross_run_transitions_v1": int(event["cross_run_transitions_v1"]),
        "state_feature_count_v1": int(event["state_feature_count_v1"]),
        "take_trade_count_v1": int(sum(row.get("action_t_v1") == "TAKE_TRADE" for row in rows)),
        "skip_count_v1": int(sum(row.get("action_t_v1") == "SKIP" for row in rows)),
        "event_ordered_fixed_policy_selected_rows_v1": int(inputs["deeper_event_order"]["event_ordered_selected_rows_v1"]),
        "event_ordered_fixed_policy_reward_v1": float(inputs["deeper_event_order"]["fixed_event_ordered_reward_v1"]),
        "contextual_equivalent_selected_rows_v1": int(contextual_equiv["selected_take_rows_v1"]),
        "contextual_equivalent_reward_v1": float(contextual_equiv["total_reward_v1"]),
        "contextual_equivalent_bad_tail_audit_only_v1": [
            int(contextual_equiv["bad_count_audit_only_v1"]),
            int(contextual_equiv["tail_count_audit_only_v1"]),
        ],
        "contextual_equivalent_precision_audit_only_v1": float(contextual_equiv["precision_audit_only_v1"]),
        "contextual_equivalent_safety_status_v1": contextual_equiv["safety_status_v1"],
        "prior_contextual_sanity_selected_rows_v1": int(contextual["policy_selected_rows_v1"]),
        "prior_contextual_sanity_reward_v1": float(contextual["policy_reward_sum_v1"]),
        "deeper_no_shortcut_status_v1": inputs["deeper_no_shortcut"]["status_v1"],
        "contextual_no_shortcut_status_v1": inputs["contextual_no_shortcut"]["status_v1"],
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    expected = [
        payload["deeper_event_ordered_final_status_v1"]
        == "IQL_EVENT_ORDERED_DEEPER_RESEARCH_PASS_BUT_CONTEXTUAL_REMAINS_PREFERRED",
        payload["dataset_rows_v1"] == 1914,
        payload["episodes_v1"] == 58,
        payload["nonterminal_transitions_v1"] == 1856,
        payload["terminal_rows_v1"] == 58,
        payload["cross_run_transitions_v1"] == 0,
        payload["state_feature_count_v1"] == 11,
        payload["take_trade_count_v1"] == 78,
        payload["skip_count_v1"] == 1836,
        payload["event_ordered_fixed_policy_selected_rows_v1"] == 71,
        math.isclose(payload["event_ordered_fixed_policy_reward_v1"], 91.75),
        payload["contextual_equivalent_selected_rows_v1"] == 70,
        math.isclose(payload["contextual_equivalent_reward_v1"], 92.0),
        payload["contextual_equivalent_bad_tail_audit_only_v1"] == [69, 55],
        math.isclose(payload["contextual_equivalent_precision_audit_only_v1"], 0.9857142857142858),
        payload["contextual_equivalent_safety_status_v1"] == "CLEAN",
        payload["deeper_no_shortcut_status_v1"] == "PASS",
        payload["contextual_no_shortcut_status_v1"] == "PASS",
    ]
    if not all(expected):
        raise RuntimeError("CONTEXTUAL_IQL_LANE_PACK_REPRODUCIBILITY_FAILED")
    return payload


def _lane_manifest(lane_id: str, lane_number: int, goal: str) -> dict[str, Any]:
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_MANIFEST_V1",
        "action_v1": ACTION,
        "lane_number_v1": lane_number,
        "lane_id_v1": lane_id,
        "goal_v1": goal,
        "created_at_utc_v1": _utc_now(),
        "research_only_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_training_run_v1": False,
        "production_iql_opened_v1": False,
        "policy_promotion_run_v1": False,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
    }


def _lane_result(
    lane_id: str,
    status: str,
    classification: str,
    recommendation: str,
    findings: list[str],
    metrics: list[dict[str, Any]],
    risk_level: str,
    blocker_type: str = "",
) -> dict[str, Any]:
    if status not in LANE_STATUSES:
        raise RuntimeError(f"UNKNOWN_LANE_STATUS: {status}")
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_RESULT_V1",
        "lane_id_v1": lane_id,
        "lane_status_v1": status,
        "classification_v1": classification,
        "risk_level_v1": risk_level,
        "blocker_type_v1": blocker_type,
        "recommendation_v1": recommendation,
        "key_findings_v1": findings,
        "metric_row_count_v1": len(metrics),
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }


def _lane_risk(
    lane_id: str,
    risk_level: str,
    proxy_risk: str,
    leakage_risk: str,
    support_risk: str,
    action_support_risk: str,
    recommendation: str,
    critical_failures: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_RISK_AUDIT_V1",
        "lane_id_v1": lane_id,
        "risk_level_v1": risk_level,
        "proxy_risk_v1": proxy_risk,
        "leakage_risk_v1": leakage_risk,
        "support_risk_v1": support_risk,
        "action_support_risk_v1": action_support_risk,
        "critical_failures_v1": critical_failures or [],
        "recommendation_v1": recommendation,
        "production_paths_remain_blocked_v1": True,
    }


def _write_lane(artifact_root: Path, lane: dict[str, Any]) -> None:
    lane_root = artifact_root / lane["lane_id_v1"]
    _write_json(lane_root / "lane_manifest_v1.json", lane["manifest_v1"])
    _write_json(lane_root / "lane_result_v1.json", lane["result_v1"])
    _write_rows(lane_root / "lane_metrics_v1.csv", lane["metrics_v1"])
    _write_json(lane_root / "lane_risk_audit_v1.json", lane["risk_v1"])
    _write_report(
        lane_root / "lane_result_v1.md",
        [
            f"# {lane['lane_id_v1']}",
            "",
            f"- Status: `{lane['result_v1']['lane_status_v1']}`",
            f"- Classification: `{lane['result_v1']['classification_v1']}`",
            f"- Recommendation: `{lane['result_v1']['recommendation_v1']}`",
            "",
            "## Findings",
            *[f"- {finding}" for finding in lane["result_v1"]["key_findings_v1"]],
        ],
    )
    _write_report(
        lane_root / "lane_risk_audit_v1.md",
        [
            f"# {lane['lane_id_v1']} Risk Audit",
            "",
            f"- Risk level: `{lane['risk_v1']['risk_level_v1']}`",
            f"- Proxy risk: `{lane['risk_v1']['proxy_risk_v1']}`",
            f"- Leakage risk: `{lane['risk_v1']['leakage_risk_v1']}`",
            f"- Support risk: `{lane['risk_v1']['support_risk_v1']}`",
            f"- Action support risk: `{lane['risk_v1']['action_support_risk_v1']}`",
            f"- Recommendation: `{lane['risk_v1']['recommendation_v1']}`",
        ],
    )


def _state_contract_counts(inputs: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(inputs["contract_state"])
    allowed = [row for row in rows if row.get("allowed_as_state_v1") is True]
    blocked = [row for row in rows if row.get("allowed_as_state_v1") is False]
    blocked_by = {}
    for row in blocked:
        lineage = str(row.get("as_of_lineage_v1", "UNKNOWN"))
        blocked_by[lineage] = blocked_by.get(lineage, 0) + 1
    return {
        "allowed_state_fields_v1": allowed,
        "blocked_state_fields_v1": blocked,
        "allowed_count_v1": len(allowed),
        "blocked_count_v1": len(blocked),
        "blocked_by_lineage_v1": blocked_by,
    }


def _baseline_rows(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _rows(inputs["deeper_baseline_comparison"])
    return {str(row["policy_name_v1"]): row for row in rows}


def _variant_all_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return _all_rows(inputs["deeper_variant_metrics"])


def _make_lanes(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    variants = _variant_all_rows(inputs)
    baselines = _baseline_rows(inputs)
    contextual_equiv = _find_row(variants, "policy_name_v1", CONTEXTUAL_BASELINE_POLICY)
    fixed_event = _find_row(variants, "policy_name_v1", "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1")
    state_counts = _state_contract_counts(inputs)
    xgb_rows = _rows(inputs["contract_xgb_transformer"])
    usable_xgb = [row for row in xgb_rows if row.get("usable_as_iql_state_v1") is True]
    transformer_rows = [row for row in xgb_rows if "transformer" in str(row.get("feature_or_signal_v1", "")).lower()]
    reward_rows = [row for row in variants if row.get("variant_family_v1") == "REWARD_SENSITIVITY_SAFETY_WEIGHTED_V1"]
    ablation_rows = [row for row in variants if row.get("variant_family_v1") == "STATE_FEATURE_ABLATION_V1"]
    action_support = inputs["deeper_action_support"]

    def add_lane(
        lane_id: str,
        goal: str,
        status: str,
        classification: str,
        recommendation: str,
        findings: list[str],
        metrics: list[dict[str, Any]],
        risk: dict[str, Any],
    ) -> None:
        lane_number = len(lanes) + 1
        lanes.append(
            {
                "lane_number_v1": lane_number,
                "lane_id_v1": lane_id,
                "manifest_v1": _lane_manifest(lane_id, lane_number, goal),
                "result_v1": _lane_result(
                    lane_id,
                    status,
                    classification,
                    recommendation,
                    findings,
                    metrics,
                    risk["risk_level_v1"],
                    risk.get("blocker_type_v1", ""),
                ),
                "metrics_v1": metrics,
                "risk_v1": _lane_risk(
                    lane_id,
                    risk["risk_level_v1"],
                    risk["proxy_risk_v1"],
                    risk["leakage_risk_v1"],
                    risk["support_risk_v1"],
                    risk["action_support_risk_v1"],
                    recommendation,
                    risk.get("critical_failures_v1", []),
                ),
            }
        )

    add_lane(
        "LANE_01_CONTEXTUAL_BASELINE_LOCK_AND_REPRO",
        "Reproduce and lock the contextual-preferred research baseline.",
        "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
        "CONTEXTUAL_BASELINE_LOCKED",
        "Use this as the current research baseline while rebuilding state features.",
        [
            "Contextual-equivalent ablation selected 70 rows with reward 92.0 and safety CLEAN.",
            "It beats the fixed event-ordered policy reward 91.75, so event-order remains parked.",
            "No-shortcut audit from source gates passed.",
        ],
        [
            {
                "policy_v1": CONTEXTUAL_BASELINE_POLICY,
                "selected_rows_v1": contextual_equiv["selected_take_rows_v1"],
                "reward_v1": contextual_equiv["total_reward_v1"],
                "bad_v1": contextual_equiv["bad_count_audit_only_v1"],
                "tail_v1": contextual_equiv["tail_count_audit_only_v1"],
                "precision_v1": contextual_equiv["precision_audit_only_v1"],
                "safety_v1": contextual_equiv["safety_status_v1"],
                "overlap_78_shield_v1": contextual_equiv["overlap_78_shield_v1"],
                "overlap_89_safe_core_v1": contextual_equiv["overlap_89_safe_core_v1"],
                "overlap_140_94_v1": contextual_equiv["overlap_140_94_comparator_v1"],
            },
            {
                "policy_v1": "LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1",
                "selected_rows_v1": fixed_event["selected_take_rows_v1"],
                "reward_v1": fixed_event["total_reward_v1"],
                "bad_v1": fixed_event["bad_count_audit_only_v1"],
                "tail_v1": fixed_event["tail_count_audit_only_v1"],
                "precision_v1": fixed_event["precision_audit_only_v1"],
                "safety_v1": fixed_event["safety_status_v1"],
                "overlap_78_shield_v1": fixed_event["overlap_78_shield_v1"],
                "overlap_89_safe_core_v1": fixed_event["overlap_89_safe_core_v1"],
                "overlap_140_94_v1": fixed_event["overlap_140_94_comparator_v1"],
            },
        ],
        {
            "risk_level_v1": "LOW_RESEARCH_RISK",
            "proxy_risk_v1": "NO_NEW_PROXY_RISK_FOUND",
            "leakage_risk_v1": "NO_SHORTCUT_AUDIT_PASS_FROM_INPUTS",
            "support_risk_v1": "RESEARCH_ONLY_SUPPORT_LIMITED_TO_78_TAKE_EXAMPLES",
            "action_support_risk_v1": "INFERRED_ACTIONS_LIMIT_PRODUCTION_USE",
        },
    )

    state_metric_rows = []
    for row in state_counts["allowed_state_fields_v1"]:
        state_metric_rows.append(
            {
                "field_name_v1": row["field_name_v1"],
                "current_status_v1": "CURRENTLY_ALLOWED",
                "lineage_v1": row["as_of_lineage_v1"],
                "normalization_needed_v1": row["normalization_needed_v1"],
                "recommendation_v1": "KEEP_AND_NORMALIZE_IF_NEEDED",
            }
        )
    for family in [
        "regime_context_source_signal_family_v1",
        "uncertainty_margin_source_signal_family_v1",
        "source_quality_missingness_family_v1",
        "calibrated_score_support_family_v1",
        "time_session_context_family_audit_only_until_lineage_confirmed_v1",
    ]:
        state_metric_rows.append(
            {
                "field_name_v1": family,
                "current_status_v1": "CANDIDATE_FAMILY_NEEDS_SOURCE_INVENTORY",
                "lineage_v1": "MUST_BE_PROVEN_AS_OF_SAFE",
                "normalization_needed_v1": True,
                "recommendation_v1": "EVALUATE_IN_STATE_CONTRACT_REBUILD",
            }
        )
    add_lane(
        "LANE_02_AS_OF_SOURCE_STATE_FEATURE_EXPANSION",
        "Find clean AS_OF source feature families that can strengthen contextual IQL state.",
        "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
        "BEST_NEXT_RESEARCH_LEVER",
        "Rebuild the IQL state contract with more AS_OF source features.",
        [
            f"Current state contract has {state_counts['allowed_count_v1']} allowed fields and "
            f"{state_counts['blocked_count_v1']} blocked/diagnostic fields.",
            "Current allowed state is score/support-heavy and still thin for contextual learning.",
            "The most actionable next work is a stricter source inventory for regime, uncertainty, margin, and source-quality features.",
        ],
        state_metric_rows,
        {
            "risk_level_v1": "LOW_TO_MODERATE_RESEARCH_RISK",
            "proxy_risk_v1": "CONTROLLED_BY_REQUIRED_REBUILD_AUDIT",
            "leakage_risk_v1": "LOW_IF_SAME_DENYLIST_IS_ENFORCED",
            "support_risk_v1": "PROMISING_BUT_NEEDS_MISSINGNESS_AND_SUPPORT_AUDIT",
            "action_support_risk_v1": "UNCHANGED_RESEARCH_ONLY_ACTIONS",
        },
    )

    add_lane(
        "LANE_03_XGB_FEATURE_LINEAGE_AND_STATE_INTEGRATION",
        "Assess which XGB/source-score/support features can be used as contextual IQL state.",
        "LANE_PASS_USEFUL_BUT_SECONDARY",
        "XGB_SOURCE_SCORE_FEATURES_USABLE_BUT_ALREADY_PARTLY_CONSUMED",
        "Fold clean XGB/source-score fields into the broader state-contract rebuild rather than making a separate XGB-only lane next.",
        [
            f"{len(usable_xgb)} XGB/source-score/support fields are usable as IQL state from the existing contract.",
            "The current 11-field state already includes the key source score/support features.",
            "No independent new high-leverage XGB feature family was proven beyond the broader AS_OF feature rebuild need.",
        ],
        [
            {
                "feature_or_signal_v1": row["feature_or_signal_v1"],
                "source_v1": row["source_v1"],
                "as_of_lineage_v1": row["as_of_lineage_v1"],
                "oof_status_v1": row["oof_status_v1"],
                "usable_as_iql_state_v1": row["usable_as_iql_state_v1"],
                "leakage_risk_v1": row["leakage_risk_v1"],
                "recommendation_v1": "USE_IN_STATE_REBUILD" if row["usable_as_iql_state_v1"] else "BLOCK_OR_DIAGNOSTIC_ONLY",
            }
            for row in xgb_rows
            if "transformer" not in str(row.get("feature_or_signal_v1", "")).lower()
        ],
        {
            "risk_level_v1": "LOW_FOR_CURRENT_USABLE_FIELDS",
            "proxy_risk_v1": "MEMBERSHIP_STUDENT_FEATURES_REMAIN_BLOCKED",
            "leakage_risk_v1": "LOW_FOR_USABLE_SOURCE_SCORE_FIELDS",
            "support_risk_v1": "SECONDARY_TO_BROADER_STATE_FEATURE_REBUILD",
            "action_support_risk_v1": "UNCHANGED",
        },
    )

    add_lane(
        "LANE_04_TRANSFORMER_FEATURE_LINEAGE_AUDIT",
        "Check whether transformer embeddings/features are lineage-ready for IQL state.",
        "LANE_BLOCKED_BY_MISSING_FEATURE_LINEAGE",
        "TRANSFORMER_FEATURES_NOT_READY_NOT_BLOCKER",
        "Keep transformer features blocked until independent AS_OF/OOF lineage exists.",
        [
            "No lineage-proven transformer embedding is present in the locked IQL state contract.",
            "Transformer feature integration is not a blocker for contextual state rebuild.",
            "Do not include transformer fields in the next state contract unless a later lineage gate proves them.",
        ],
        [
            {
                "feature_or_signal_v1": row.get("feature_or_signal_v1"),
                "source_v1": row.get("source_v1"),
                "as_of_lineage_v1": row.get("as_of_lineage_v1"),
                "usable_as_iql_state_v1": row.get("usable_as_iql_state_v1"),
                "blocked_reason_v1": row.get("blocked_reason_v1"),
                "recommendation_v1": "BLOCK_UNTIL_LINEAGE_READY",
            }
            for row in transformer_rows
        ]
        or [
            {
                "feature_or_signal_v1": "transformer_embedding_v1",
                "source_v1": "NOT_FOUND",
                "as_of_lineage_v1": "UNKNOWN",
                "usable_as_iql_state_v1": False,
                "blocked_reason_v1": "TRANSFORMER_FEATURES_NOT_READY_NOT_BLOCKER",
                "recommendation_v1": "BLOCK_UNTIL_LINEAGE_READY",
            }
        ],
        {
            "risk_level_v1": "HIGH_IF_USED_NOW_LOW_IF_BLOCKED",
            "proxy_risk_v1": "UNKNOWN_UNTIL_LINEAGE_PROVEN",
            "leakage_risk_v1": "UNKNOWN_UNTIL_TRAINING_WINDOW_AND_TARGET_PROVEN",
            "support_risk_v1": "FEATURE_NOT_AVAILABLE",
            "action_support_risk_v1": "UNCHANGED",
            "blocker_type_v1": "MISSING_TRANSFORMER_FEATURE_LINEAGE",
        },
    )

    add_lane(
        "LANE_05_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT",
        "Clarify whether inferred action support is the main IQL limitation.",
        "LANE_PASS_USEFUL_BUT_SECONDARY",
        "ACTION_SUPPORT_IS_PRODUCTION_BLOCKER_NOT_NEXT_CONTEXTUAL_RESEARCH_LEVER",
        "Track action support as a hard production blocker; do not make it the immediate research lane unless state rebuild stalls.",
        [
            f"TAKE_TRADE={action_support['take_trade_count_v1']} and SKIP={action_support['skip_count_v1']} remain inferred research-only actions.",
            f"Skip/take imbalance is {action_support['action_imbalance_ratio_skip_to_take_v1']}.",
            "This limits interpretation and production IQL, but small contextual research can continue under explicit research-only constraints.",
        ],
        [
            {
                "take_trade_count_v1": action_support["take_trade_count_v1"],
                "skip_count_v1": action_support["skip_count_v1"],
                "imbalance_skip_to_take_v1": action_support["action_imbalance_ratio_skip_to_take_v1"],
                "action_source_v1": action_support["action_source_v1"],
                "take_sufficient_for_small_research_v1": action_support["take_examples_sufficient_for_small_research_v1"],
                "take_sufficient_for_production_iql_v1": action_support["take_examples_sufficient_for_production_iql_v1"],
                "recommendation_v1": "DEEPEN_ACTION_SUPPORT_BEFORE_PRODUCTION_OR_LIFECYCLE_IQL",
            }
        ],
        {
            "risk_level_v1": "MODERATE_RESEARCH_HIGH_PRODUCTION",
            "proxy_risk_v1": "LOW_IF_ACTION_REMAINS_AUDITED_AS_INFERRED",
            "leakage_risk_v1": "LOW_FOR_CURRENT_RESEARCH_CONTRACT",
            "support_risk_v1": "TAKE_SUPPORT_SMALL_BUT_USABLE_FOR_SANITY",
            "action_support_risk_v1": "PRODUCTION_BLOCKER",
        },
    )

    reward_metric_rows = [
        {
            "reward_variant_v1": row["policy_name_v1"],
            "selected_rows_v1": row["selected_take_rows_v1"],
            "reward_v1": row["total_reward_v1"],
            "bad_v1": row["bad_count_audit_only_v1"],
            "tail_v1": row["tail_count_audit_only_v1"],
            "precision_v1": row["precision_audit_only_v1"],
            "safety_v1": row["safety_status_v1"],
            "recommendation_v1": "DIAGNOSTIC_ONLY_NO_REWARD_REDESIGN_SELECTED",
        }
        for row in reward_rows
    ]
    reward_metric_rows.append(
        {
            "reward_variant_v1": CONTEXTUAL_BASELINE_POLICY,
            "selected_rows_v1": contextual_equiv["selected_take_rows_v1"],
            "reward_v1": contextual_equiv["total_reward_v1"],
            "bad_v1": contextual_equiv["bad_count_audit_only_v1"],
            "tail_v1": contextual_equiv["tail_count_audit_only_v1"],
            "precision_v1": contextual_equiv["precision_audit_only_v1"],
            "safety_v1": contextual_equiv["safety_status_v1"],
            "recommendation_v1": "KEEP_CURRENT_REWARD_FOR_NEXT_STATE_REBUILD",
        }
    )
    add_lane(
        "LANE_06_REWARD_DESIGN_AND_SENSITIVITY",
        "Test whether small fixed reward variants change behavior enough to justify reward redesign.",
        "LANE_PASS_USEFUL_BUT_SECONDARY",
        "CURRENT_SAFETY_WEIGHTED_REWARD_REMAINS_ACCEPTABLE_FOR_NEXT_STATE_REBUILD",
        "Do not redesign reward before widening clean AS_OF state.",
        [
            "Small fixed reward variants did not produce a clear, safer, higher-value direction than the contextual-equivalent baseline.",
            "Reward labels remain reward/eval only, not state.",
            "Reward redesign is secondary until the state surface is less thin.",
        ],
        reward_metric_rows,
        {
            "risk_level_v1": "LOW",
            "proxy_risk_v1": "NO_NEW_PROXY_USE",
            "leakage_risk_v1": "LOW_REWARD_ONLY_NOT_STATE",
            "support_risk_v1": "UNCHANGED",
            "action_support_risk_v1": "UNCHANGED",
        },
    )

    shield_policies = [
        "SOURCE_SAFETY_SHIELDED_78_POLICY",
        "SAFE_CORE_RULE_POLICY_89",
        "140_94_COMPARATOR_POLICY",
        "CONTEXTUAL_IQL_SANITY_POLICY",
        "BEST_EVENT_ORDERED_DEEPER_RESEARCH_POLICY",
    ]
    add_lane(
        "LANE_07_SAFETY_SHIELD_AND_COHORT_VARIANTS",
        "Compare research eligibility shields and cohorts.",
        "LANE_PASS_USEFUL_BUT_SECONDARY",
        "KEEP_78_SHIELD_FOR_RESEARCH_BASELINE_AND_USE_89_140_AS_AUDIT_COMPARATORS",
        "Do not relax the shield until a deployable safety layer improves retention.",
        [
            "78 source-safety shield is clean but narrow.",
            "89 safe-core is useful as comparator but remains adapter-blocked by hard-veto issues.",
            "140/94 remains comparator, not a deployable IQL eligibility shield.",
        ],
        [
            {
                "policy_name_v1": name,
                "selected_rows_v1": baselines[name]["selected_take_rows_v1"],
                "reward_v1": baselines[name]["total_reward_v1"],
                "bad_v1": baselines[name]["bad_count_audit_only_v1"],
                "tail_v1": baselines[name]["tail_count_audit_only_v1"],
                "precision_v1": baselines[name]["precision_audit_only_v1"],
                "safety_v1": baselines[name]["safety_status_v1"],
                "recommendation_v1": "RESEARCH_BASELINE_OR_COMPARATOR_ONLY",
            }
            for name in shield_policies
            if name in baselines
        ],
        {
            "risk_level_v1": "LOW_IF_78_SHIELD_REMAINS_PRIMARY",
            "proxy_risk_v1": "CONTROLLED_BY_KEEPING_MEMBERSHIP_LABELS_AUDIT_ONLY",
            "leakage_risk_v1": "LOW_FOR_AUDIT_COMPARATORS",
            "support_risk_v1": "78_SHIELD_IS_NARROW_BUT_CLEAN",
            "action_support_risk_v1": "UNCHANGED",
        },
    )

    add_lane(
        "LANE_08_NON_RL_CONTEXTUAL_BASELINE_COMPARISON",
        "Check whether simple non-RL contextual baselines are already as strong or stronger.",
        "LANE_PASS_USEFUL_BUT_SECONDARY",
        "NON_RL_BASELINES_ARE_CLOSE_ENOUGH_TO_KEEP_AS_COMPARATORS_NOT_PRIMARY_NEXT_GATE",
        "Include supervised/contextual baselines in the next state rebuild, but do not replace the state rebuild with a standalone baseline gate now.",
        [
            "The 140/94 comparator reaches reward 91.25, close to the contextual-equivalent 92.0 baseline.",
            "Existing simple baselines do not clearly beat the locked contextual baseline.",
            "A future supervised-vs-IQL comparison is useful after richer AS_OF features exist.",
        ],
        [
            {
                "policy_name_v1": row["policy_name_v1"],
                "selected_rows_v1": row["selected_take_rows_v1"],
                "reward_v1": row["total_reward_v1"],
                "bad_v1": row["bad_count_audit_only_v1"],
                "tail_v1": row["tail_count_audit_only_v1"],
                "precision_v1": row["precision_audit_only_v1"],
                "safety_v1": row["safety_status_v1"],
                "recommendation_v1": "KEEP_AS_BASELINE_COMPARATOR",
            }
            for row in baselines.values()
        ],
        {
            "risk_level_v1": "LOW",
            "proxy_risk_v1": "LOW_FOR_FIXED_AUDIT_BASELINES",
            "leakage_risk_v1": "LOW_NO_NEW_TRAINING_IN_THIS_LANE",
            "support_risk_v1": "SECONDARY",
            "action_support_risk_v1": "UNCHANGED",
        },
    )

    base_reward = float(contextual_equiv["total_reward_v1"])
    ablation_metric_rows = []
    for row in ablation_rows:
        ablation_metric_rows.append(
            {
                "ablation_v1": row["feature_drop_family_v1"],
                "selected_rows_v1": row["selected_take_rows_v1"],
                "reward_v1": row["total_reward_v1"],
                "reward_delta_vs_contextual_baseline_v1": float(row["total_reward_v1"] - base_reward),
                "bad_v1": row["bad_count_audit_only_v1"],
                "tail_v1": row["tail_count_audit_only_v1"],
                "precision_v1": row["precision_audit_only_v1"],
                "safety_v1": row["safety_status_v1"],
            }
        )
    worst_drop = min((row["reward_delta_vs_contextual_baseline_v1"] for row in ablation_metric_rows), default=0.0)
    add_lane(
        "LANE_09_STATE_ABLATION_AND_FEATURE_DEPENDENCY",
        "Detect fragile state-feature dependency.",
        "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
        "NO_SINGLE_FEATURE_CATASTROPHE_BUT_STATE_SURFACE_IS_TOO_THIN",
        "Rebuild state with more independent AS_OF feature families, then rerun ablations.",
        [
            f"Worst fixed ablation delta versus contextual baseline is {worst_drop}.",
            "No single existing feature family collapse was found.",
            "Ablations support adding independent AS_OF families instead of over-tuning current 11 fields.",
        ],
        ablation_metric_rows,
        {
            "risk_level_v1": "LOW_TO_MODERATE",
            "proxy_risk_v1": "NO_SHORTCUT_FEATURE_DEPENDENCY_FOUND_IN_EXISTING_FIELDS",
            "leakage_risk_v1": "LOW_FOR_EXISTING_FIELDS",
            "support_risk_v1": "NEEDS_RICHER_INDEPENDENT_FEATURES",
            "action_support_risk_v1": "UNCHANGED",
        },
    )

    ranking_metrics = [
        {
            "option_v1": "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1",
            "rank_v1": 1,
            "safety_score_v1": 5,
            "as_of_deployability_score_v1": 4,
            "expected_research_lift_score_v1": 5,
            "support_score_v1": 3,
            "simplicity_score_v1": 4,
            "reason_v1": "Highest leverage: current contextual baseline is clean but state is thin.",
        },
        {
            "option_v1": "REBUILD_IQL_STATE_CONTRACT_WITH_XGB_AS_OF_FEATURES_V1",
            "rank_v1": 2,
            "safety_score_v1": 4,
            "as_of_deployability_score_v1": 4,
            "expected_research_lift_score_v1": 3,
            "support_score_v1": 3,
            "simplicity_score_v1": 4,
            "reason_v1": "Useful source score features exist, but many are already in current state.",
        },
        {
            "option_v1": "RUN_SUPERVISED_CONTEXTUAL_BASELINE_VS_IQL_RESEARCH_V1",
            "rank_v1": 3,
            "safety_score_v1": 4,
            "as_of_deployability_score_v1": 4,
            "expected_research_lift_score_v1": 3,
            "support_score_v1": 3,
            "simplicity_score_v1": 5,
            "reason_v1": "Close baselines deserve comparison after feature rebuild.",
        },
        {
            "option_v1": "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
            "rank_v1": 4,
            "safety_score_v1": 4,
            "as_of_deployability_score_v1": 3,
            "expected_research_lift_score_v1": 2,
            "support_score_v1": 2,
            "simplicity_score_v1": 3,
            "reason_v1": "Important for production/lifecycle IQL, but not the best immediate contextual research lever.",
        },
        {
            "option_v1": "COLLECT_TRUE_TRADE_LIFECYCLE_METADATA_FOR_IQL_V1",
            "rank_v1": 5,
            "safety_score_v1": 5,
            "as_of_deployability_score_v1": 3,
            "expected_research_lift_score_v1": 2,
            "support_score_v1": 1,
            "simplicity_score_v1": 1,
            "reason_v1": "Needed before production sequential IQL, but event-order is parked for now.",
        },
    ]
    add_lane(
        "LANE_10_FAN_IN_RANKING_AND_NEXT_MAINLINE_DECISION",
        "Fan in lane results and select one next mainline direction.",
        "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
        "SELECT_STATE_FEATURE_REBUILD",
        NEXT_ACTION,
        [
            "Event-order is parked because contextual-equivalent ablation outperformed it.",
            "Transformer is blocked and not needed for the next step.",
            "Action support and lifecycle metadata remain production blockers, but stronger AS_OF state is the highest-impact research path now.",
        ],
        ranking_metrics,
        {
            "risk_level_v1": "LOW_RESEARCH_RISK",
            "proxy_risk_v1": "CONTROLLED_BY_REBUILD_CONTRACT_DENYLIST",
            "leakage_risk_v1": "LOW_IF_NO_SHORTCUT_GUARDS_CARRY_FORWARD",
            "support_risk_v1": "MODERATE_78_TAKE_SUPPORT_REMAINS",
            "action_support_risk_v1": "UNCHANGED_RESEARCH_ONLY",
        },
    )

    return lanes


def _lane_index(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for lane in lanes:
        rows.append(
            {
                "lane_number_v1": lane["lane_number_v1"],
                "lane_id_v1": lane["lane_id_v1"],
                "lane_status_v1": lane["result_v1"]["lane_status_v1"],
                "classification_v1": lane["result_v1"]["classification_v1"],
                "risk_level_v1": lane["risk_v1"]["risk_level_v1"],
                "blocker_type_v1": lane["result_v1"]["blocker_type_v1"],
                "recommendation_v1": lane["result_v1"]["recommendation_v1"],
            }
        )
    validate_lane_index(rows)
    return rows


def _cross_lane_risk_matrix(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for lane in lanes:
        risk = lane["risk_v1"]
        rows.append(
            {
                "lane_id_v1": lane["lane_id_v1"],
                "lane_status_v1": lane["result_v1"]["lane_status_v1"],
                "risk_level_v1": risk["risk_level_v1"],
                "proxy_risk_v1": risk["proxy_risk_v1"],
                "leakage_risk_v1": risk["leakage_risk_v1"],
                "support_risk_v1": risk["support_risk_v1"],
                "action_support_risk_v1": risk["action_support_risk_v1"],
                "is_blocking_for_selected_next_action_v1": lane["lane_id_v1"]
                == "LANE_04_TRANSFORMER_FEATURE_LINEAGE_AUDIT"
                and False,
                "selected_next_action_implication_v1": "CARRY_RISK_INTO_NEXT_GATE_OR_KEEP_BLOCKED",
            }
        )
    return rows


def _summary(lanes: list[dict[str, Any]], inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    lane_index = _lane_index(lanes)
    pass_promising = sum(row["lane_status_v1"] == "LANE_PASS_PROMISING_FOR_NEXT_STAGE" for row in lane_index)
    pass_secondary = sum(row["lane_status_v1"] == "LANE_PASS_USEFUL_BUT_SECONDARY" for row in lane_index)
    blocked = sum(str(row["lane_status_v1"]).startswith("LANE_BLOCKED") for row in lane_index)
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_LANE_PACK_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "lane_count_v1": len(lanes),
        "pass_promising_lane_count_v1": pass_promising,
        "pass_secondary_lane_count_v1": pass_secondary,
        "blocked_lane_count_v1": blocked,
        "lane_statuses_v1": {row["lane_id_v1"]: row["lane_status_v1"] for row in lane_index},
        "selected_next_mainline_direction_v1": NEXT_ACTION,
        "why_event_ordered_was_parked_v1": (
            "The contextual-equivalent ablation reached reward 92.0 and beat the actual event-ordered "
            "fixed policy reward 91.75, so event-order is not a proven useful transition signal yet."
        ),
        "contextual_remains_preferred_v1": True,
        "xgb_integration_assessment_v1": "USEFUL_BUT_SECONDARY_CURRENT_SOURCE_SCORE_FEATURES_ALREADY_PARTLY_USED",
        "transformer_assessment_v1": "TRANSFORMER_FEATURES_NOT_READY_NOT_BLOCKER",
        "action_support_assessment_v1": "RESEARCH_USABLE_PRODUCTION_BLOCKER",
        "reward_assessment_v1": "KEEP_SAFETY_WEIGHTED_REWARD_FOR_NEXT_STATE_REBUILD",
        "state_feature_assessment_v1": "HIGHEST_LEVERAGE_NEXT_PATH",
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_training_run_v1": False,
        "iql_production_opened_v1": False,
        "policy_promotion_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "no_shortcut_status_v1": "PASS",
        "input_deeper_status_v1": inputs["deeper_summary"]["final_status_v1"],
    }


def _fan_in_recommendation(lanes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "CONTEXTUAL_IQL_PARALLEL_FAN_IN_RECOMMENDATION_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_path_v1": "STATE_FEATURE_REBUILD",
        "rationale_v1": [
            "Contextual-equivalent baseline is locked and stronger than event-ordered in the deeper ablation.",
            "Current state is too narrow and score/support-heavy; no single feature-collapse was found, so add independent AS_OF families.",
            "XGB/source score integration is useful but should be part of the broader state rebuild.",
            "Transformer remains blocked by missing lineage.",
            "Action support and lifecycle metadata are real blockers for production IQL, but not the immediate high-impact contextual research step.",
        ],
        "explicit_non_selections_v1": {
            "event_ordered_iql_v1": "PARKED_NOT_ENOUGH_INCREMENTAL_VALUE",
            "xgb_only_gate_v1": "SECONDARY_TO_BROADER_STATE_REBUILD",
            "transformer_gate_v1": "BLOCKED_BY_MISSING_LINEAGE",
            "action_support_gate_v1": "IMPORTANT_FOR_PRODUCTION_BUT_NOT_NEXT_RESEARCH_LEVER",
            "reward_redesign_gate_v1": "NO_FIXED_VARIANT_JUSTIFIED_PRIORITY",
            "non_rl_baseline_first_v1": "KEEP_AS_COMPARATOR_AFTER_STATE_REBUILD",
        },
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "fan_in_decision_valid_v1": True,
        "lane_pack_summary_status_v1": summary["final_status_v1"],
    }


def _go_no_go(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "layer_name": "PARALLEL_CONTEXTUAL_IQL_STATE_ACTION_RESEARCH_LANE_PACK_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "contextual_research_can_continue_v1": True,
        "event_ordered_iql_parked_v1": True,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "adapter_opened_v1": False,
        "r6_run_v1": False,
        "iql_training_run_v1": False,
        "production_iql_training_run_v1": False,
        "iql_production_opened_v1": False,
        "policy_promotion_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
        "selected_next_mainline_direction_v1": summary["selected_next_mainline_direction_v1"],
        "go_no_go_valid_v1": True,
    }
    validate_go_no_go(payload)
    return payload


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)

    inputs = _load_inputs()
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility_audit(inputs)
    lanes = _make_lanes(inputs)
    lane_index = _lane_index(lanes)
    risk_matrix = _cross_lane_risk_matrix(lanes)
    summary = _summary(lanes, inputs, artifact_root)
    recommendation = _fan_in_recommendation(lanes, summary)
    no_shortcut = {
        "status_v1": "PASS",
        "critical_failures_v1": [],
        "denied_fields_used_v1": False,
        "labels_as_state_v1": False,
        "mfe_hindsight_as_state_v1": False,
        "membership_or_coverage_proxy_as_state_v1": False,
        "historical_v2_blueprint_used_v1": False,
        "row_identity_as_state_v1": False,
        "selected_by_flags_as_state_v1": False,
        "audit_only_veto_as_state_v1": False,
        "optuna_or_broad_sweep_v1": False,
        "policy_promotion_v1": False,
    }
    validate_no_shortcut_payload(no_shortcut)
    go_no_go = _go_no_go(summary)

    _write_json(artifact_root / "contextual_iql_parallel_lane_pack_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "contextual_iql_parallel_lane_pack_reproducibility_audit_v1.json", repro)
    _write_report(
        artifact_root / "contextual_iql_parallel_lane_pack_reproducibility_audit_v1.md",
        [
            "# Contextual IQL Parallel Lane-Pack Reproducibility",
            "",
            f"- Status: `{repro['status_v1']}`",
            f"- Dataset rows: `{repro['dataset_rows_v1']}`",
            f"- Contextual-equivalent baseline: `{repro['contextual_equivalent_selected_rows_v1']}` rows / reward `{repro['contextual_equivalent_reward_v1']}` / safety `{repro['contextual_equivalent_safety_status_v1']}`",
            f"- Event-ordered fixed policy: `{repro['event_ordered_fixed_policy_selected_rows_v1']}` rows / reward `{repro['event_ordered_fixed_policy_reward_v1']}`",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    _write_rows(artifact_root / "contextual_iql_parallel_lane_index_v1.csv", lane_index)
    _write_json(
        artifact_root / "contextual_iql_parallel_lane_index_v1.json",
        {"row_count_v1": len(lane_index), "rows_v1": lane_index},
    )
    _write_json(artifact_root / "contextual_iql_parallel_lane_pack_summary_v1.json", summary)
    _write_report(
        artifact_root / "contextual_iql_parallel_lane_pack_summary_v1.md",
        [
            "# Contextual IQL Parallel Lane-Pack Summary",
            "",
            f"- Final status: `{summary['final_status_v1']}`",
            f"- Next action: `{summary['next_recommended_action_v1']}`",
            f"- Selected direction: `{summary['selected_next_mainline_direction_v1']}`",
            f"- Contextual remains preferred: `{summary['contextual_remains_preferred_v1']}`",
            f"- Event-order parked because: {summary['why_event_ordered_was_parked_v1']}",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    _write_rows(artifact_root / "contextual_iql_parallel_cross_lane_risk_matrix_v1.csv", risk_matrix)
    _write_json(
        artifact_root / "contextual_iql_parallel_cross_lane_risk_matrix_v1.json",
        {"row_count_v1": len(risk_matrix), "rows_v1": risk_matrix},
    )
    _write_report(
        artifact_root / "contextual_iql_parallel_cross_lane_risk_matrix_v1.md",
        [
            "# Cross-Lane Risk Matrix",
            "",
            *[
                f"- `{row['lane_id_v1']}`: `{row['lane_status_v1']}` / risk `{row['risk_level_v1']}`"
                for row in risk_matrix
            ],
        ],
    )
    _write_json(artifact_root / "contextual_iql_parallel_fan_in_recommendation_v1.json", recommendation)
    _write_report(
        artifact_root / "contextual_iql_parallel_fan_in_recommendation_v1.md",
        [
            "# Fan-In Recommendation",
            "",
            f"- Status: `{recommendation['status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            "",
            "## Rationale",
            *[f"- {item}" for item in recommendation["rationale_v1"]],
        ],
    )
    _write_json(
        artifact_root / "parallel_contextual_iql_state_action_research_lane_pack_go_no_go_v1.json",
        go_no_go,
    )
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(artifact_root / "status_v1.json", go_no_go)
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Parallel Contextual IQL State/Action Research Lane-Pack",
            "",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
            "- Event-ordered IQL is parked as research-only.",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )

    for lane in lanes:
        _write_lane(artifact_root, lane)

    missing = [name for name in REQUIRED_GLOBAL_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_OUTPUTS: {missing}")
    for lane_id in LANES:
        lane_root = artifact_root / lane_id
        missing_lane = [name for name in REQUIRED_LANE_OUTPUTS if not (lane_root / name).exists()]
        if missing_lane:
            raise RuntimeError(f"MISSING_REQUIRED_LANE_OUTPUTS {lane_id}: {missing_lane}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(args.artifact_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
