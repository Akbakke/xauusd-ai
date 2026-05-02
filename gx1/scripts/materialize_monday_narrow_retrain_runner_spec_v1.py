#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1"
JOB_SPEC_PREFIX = "MONDAY_NARROW_RETRAIN_JOB_SPEC_V1_"
SCOPE_PLAN_PREFIX = "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1_"

CONTRACT = "contract_v1.json"
RUNNER_SPEC = "monday_narrow_retrain_runner_spec_v1.json"
CONFIG_LOCK = "monday_narrow_retrain_config_lock_v1.json"
FEATURE_MANIFEST = "monday_narrow_retrain_feature_manifest_v1.json"
FEATURE_MANIFEST_TABLE = "monday_narrow_retrain_feature_manifest_v1.csv"
PRELAUNCH_CHECKLIST = "monday_narrow_retrain_prelaunch_checklist_v1.json"
OUTPUT_SPEC = "monday_narrow_retrain_output_spec_v1.json"
ABORT_RULES = "monday_narrow_retrain_abort_rules_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

JOB_TRAINING_SPEC = "training_job_spec_lock_v1.json"
JOB_INPUT_CONTRACT = "training_input_contract_v1.json"
JOB_LABEL_LOCK = "label_and_target_lock_v1.json"
JOB_MODEL_CONFIG = "model_and_training_configuration_spec_v1.json"
JOB_OUTPUT_SPEC = "output_artifact_spec_v1.json"
JOB_VERDICT_MATRIX = "eval_verdict_matrix_v1.json"
JOB_PRE_RUN = "pre_run_validation_checklist_v1.json"
JOB_POST_RUN = "post_run_eval_checklist_v1.json"
JOB_ABORT = "no_go_and_abort_protocol_v1.json"
JOB_SUMMARY = "summary_v1.json"
JOB_STATUS = "status_v1.json"

SCOPE_FEATURE_LOCK = "feature_set_lock_v1.json"

JOB_NAME = "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1"
SCOPE = "NARROW_RUNNER_FIRST_SHADOW_ONLY"
BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"
MONDAY_R6_ROLE = "FAILURE_MINER_DIAGNOSIS_ONLY"
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

NEW_PROXY_FEATURES = [
    "as_of_pre_entry_vol_exp_comp_score_v1",
    "as_of_pre_entry_directional_asymmetry_score_v1",
    "as_of_pre_entry_swing_retracement_alignment_score_v1",
    "as_of_pre_entry_tail_leakage_pocket_score_v1",
    "as_of_pre_entry_runner_protection_guard_score_v1",
]

FORBIDDEN_PATTERNS = [
    "as_of_skip_xgb_",
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    job_spec_dir = _latest_dir(reports_root, JOB_SPEC_PREFIX)
    scope_dir = _latest_dir(reports_root, SCOPE_PLAN_PREFIX)
    payload = {
        "job_spec_dir": job_spec_dir,
        "scope_dir": scope_dir,
        "job_training_spec": _load_json(job_spec_dir / JOB_TRAINING_SPEC),
        "job_input_contract": _load_json(job_spec_dir / JOB_INPUT_CONTRACT),
        "job_label_lock": _load_json(job_spec_dir / JOB_LABEL_LOCK),
        "job_model_config": _load_json(job_spec_dir / JOB_MODEL_CONFIG),
        "job_output_spec": _load_json(job_spec_dir / JOB_OUTPUT_SPEC),
        "job_verdict_matrix": _load_json(job_spec_dir / JOB_VERDICT_MATRIX),
        "job_pre_run": _load_json(job_spec_dir / JOB_PRE_RUN),
        "job_post_run": _load_json(job_spec_dir / JOB_POST_RUN),
        "job_abort": _load_json(job_spec_dir / JOB_ABORT),
        "job_summary": _load_json(job_spec_dir / JOB_SUMMARY),
        "job_status": _load_json(job_spec_dir / JOB_STATUS),
        "scope_feature_lock": _load_json(scope_dir / SCOPE_FEATURE_LOCK),
    }
    return payload


def _feature_rows(feature_lock: Dict[str, Any]) -> pd.DataFrame:
    baseline = feature_lock["baseline_training_features_v1"]["feature_names_v1"]
    proxy_meta = feature_lock["new_proxy_features_v1"]
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(baseline, start=1):
        rows.append(
            {
                "feature_name_v1": name,
                "feature_group_v1": "LOCKED_BASELINE",
                "manifest_order_v1": idx,
                "legal_status_v1": "PRE_ENTRY_LEGAL",
                "source_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
                "role_v1": "Existing legal baseline feature on exact-only training surface.",
                "pockets_helped_v1": "",
                "must_exclude_v1": False,
            }
        )
    for idx, name in enumerate(NEW_PROXY_FEATURES, start=len(baseline) + 1):
        meta = proxy_meta.get(name, {})
        rows.append(
            {
                "feature_name_v1": name,
                "feature_group_v1": "LOCKED_NEW_PROXY",
                "manifest_order_v1": idx,
                "legal_status_v1": "PRE_ENTRY_LEGAL",
                "source_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
                "role_v1": meta.get("role_v1", ""),
                "pockets_helped_v1": "|".join(meta.get("pockets_helped_v1", [])),
                "must_exclude_v1": False,
            }
        )
    return pd.DataFrame(rows)


def _build_runner_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    training_spec = payload["job_training_spec"]
    input_contract = payload["job_input_contract"]
    label_lock = payload["job_label_lock"]
    model_config = payload["job_model_config"]
    abort = payload["job_abort"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1",
        "job_name_v1": JOB_NAME,
        "scope_v1": SCOPE,
        "execution_mode_v1": "SPEC_ONLY_DO_NOT_RUN_TRAINING",
        "training_now_v1": False,
        "input_artifact_v1": input_contract["valid_input_v1"]["training_feature_surface_path_v1"],
        "input_surface_kind_v1": input_contract["valid_input_v1"]["training_feature_surface_kind_v1"],
        "expected_training_rows_v1": input_contract["valid_input_v1"]["expected_training_row_count_v1"],
        "label_artifact_v1": label_lock["target_surface_v1"]["artifact_v1"],
        "label_contract_v1": training_spec["target_label_contract_v1"],
        "locked_training_heads_v1": label_lock["locked_training_heads_v1"],
        "feature_manifest_artifacts_v1": [
            FEATURE_MANIFEST,
            FEATURE_MANIFEST_TABLE,
        ],
        "feature_manifest_loading_rule_v1": "Load feature list from monday_narrow_retrain_feature_manifest_v1.csv and hard-fail if actual matrix differs.",
        "split_eval_setup_v1": {
            "walkforward_required_v1": model_config["evaluation_structure_v1"]["walkforward_required_v1"],
            "loso_required_v1": model_config["evaluation_structure_v1"]["loso_required_v1"],
            "rolling_window_required_v1": model_config["evaluation_structure_v1"]["rolling_window_required_v1"],
            "batch_weeks_v1": model_config["evaluation_structure_v1"]["batch_weeks_v1"],
            "batch04_batch05_reporting_v1": model_config["evaluation_structure_v1"]["batch04_batch05_reporting_v1"],
        },
        "output_namespace_v1": "ALL_TRADE_REVIEW_LEDGER_<timestamp>_MONDAY_NARROW_RETRAIN_RUN_V1",
        "required_manifests_status_v1": [
            "training summary JSON",
            "model/config manifest",
            "feature manifest",
            "eval summary",
            "compare-against report",
            "pocket report",
            "verdict package",
            "manifest_v1.json",
            "status_v1.json",
            "consistency_audit_v1.csv",
        ],
        "compare_against_inputs_v1": training_spec["eval_package_v1"]["compare_against_v1"],
        "hard_fail_before_start_v1": abort["do_not_start_if_v1"]
        + input_contract["input_validation_hard_fail_v1"],
        "hard_fail_after_eval_v1": abort["reject_after_run_if_v1"],
        "runner_may_train_v1": False,
        "runner_may_prepare_config_v1": True,
    }


def _build_config_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_config = payload["job_model_config"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_CONFIG_LOCK_V1",
        "model_family_v1": "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY",
        "base_model_v1": "XGBClassifier per head",
        "compact_grid_v1": True,
        "seed_v1": model_config["reproducibility_v1"]["seed_v1"],
        "n_jobs_v1": model_config["reproducibility_v1"]["n_jobs_v1"],
        "training_mode_v1": model_config["training_mode_v1"],
        "head_family_v1": model_config["model_family_v1"]["head_family_v1"],
        "default_model_hyperparams_v1": model_config["default_model_hyperparams_v1"],
        "split_eval_lock_v1": model_config["evaluation_structure_v1"],
        "must_hold_constant_v1": model_config["must_hold_constant_vs_prior_r6_v1"]
        + [
            "exact-only canonical training surface",
            "five selected legal proxy features",
            "feature count = 67",
            "training row count = 1689",
            "label intersection = 1689",
        ],
        "allowed_to_adjust_in_narrow_slice_v1": model_config["allowed_to_change_in_this_narrow_slice_v1"]
        + [
            "per-head threshold search inside compact grid",
            "calibration inside existing R6-style family",
        ],
        "not_allowed_to_change_v1": model_config["not_allowed_to_change_v1"]
        + [
            "management/exit truth usage",
            "policy-log / decision-log field usage",
            "bridge-only rows/signals",
            "as_of_skip_xgb_* fields",
            "deferred feature candidates",
            "live/controller behavior",
        ],
    }


def _build_feature_manifest(payload: Dict[str, Any], feature_df: pd.DataFrame) -> Dict[str, Any]:
    feature_lock = payload["scope_feature_lock"]
    exclusions = feature_lock["explicit_exclusions_v1"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_FEATURE_MANIFEST_V1",
        "feature_manifest_table_v1": FEATURE_MANIFEST_TABLE,
        "baseline_feature_count_v1": int(feature_lock["baseline_training_features_v1"]["feature_count_v1"]),
        "new_proxy_feature_count_v1": len(NEW_PROXY_FEATURES),
        "total_feature_count_v1": int(len(feature_df)),
        "baseline_features_v1": feature_lock["baseline_training_features_v1"]["feature_names_v1"],
        "new_proxy_features_v1": NEW_PROXY_FEATURES,
        "explicit_exclusion_list_v1": {
            "bridge_only_rows_or_signals_v1": True,
            "management_exit_truth_v1": True,
            "policy_decision_log_fields_v1": True,
            "as_of_skip_xgb_fields_v1": "as_of_skip_xgb_*",
            "deferred_candidates_v1": [
                "pre_entry_session_pocket_runner_expectancy_v1",
                "pre_entry_adverse_first_risk_proxy_v1",
                "spread_cost_pressure_hardening_v1",
            ],
            "forbidden_sources_v1": exclusions["forbidden_sources_v1"],
            "forbidden_field_examples_v1": exclusions["forbidden_field_examples_v1"],
        },
        "feature_legality_rule_v1": "All included fields must be available at entry-anchor or earlier on the exact-only canonical entry raw-state surface.",
    }


def _build_prelaunch_checklist(payload: Dict[str, Any], feature_df: pd.DataFrame) -> Dict[str, Any]:
    input_contract = payload["job_input_contract"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_PRELAUNCH_CHECKLIST_V1",
        "checks_v1": [
            {
                "check_id_v1": "TRAINING_SURFACE_EXACT_ONLY_CANONICAL_RAW_STATE",
                "required_v1": input_contract["valid_input_v1"]["training_feature_surface_path_v1"],
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "ROW_COUNT_AND_POPULATION",
                "required_v1": input_contract["valid_input_v1"]["expected_training_row_count_v1"],
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "BRIDGE_NOT_USED_AS_TRAINING_SURFACE",
                "required_v1": "bridge rows/signals absent from training matrix",
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "FEATURE_LEGALITY",
                "required_v1": "no management/exit truth, policy-log fields, as_of_skip_xgb_* fields, or deferred candidates",
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "FEATURE_COUNT",
                "required_v1": int(len(feature_df)),
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "LABEL_INTERSECTION",
                "required_v1": input_contract["valid_input_v1"]["expected_exact_label_intersection_v1"],
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "COMPARE_REFERENCES_RESOLVE",
                "required_v1": [BENCHMARK, MONDAY_SAFETY_REFERENCE, MONDAY_R6_ROLE],
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "OUTPUT_NAMESPACE_CLEAN",
                "required_v1": "append-only output namespace does not already exist",
                "hard_fail_v1": True,
            },
            {
                "check_id_v1": "REQUIRED_CONTRACT_FILES_PRESENT",
                "required_v1": [
                    CONTRACT,
                    RUNNER_SPEC,
                    CONFIG_LOCK,
                    FEATURE_MANIFEST,
                    FEATURE_MANIFEST_TABLE,
                    OUTPUT_SPEC,
                    ABORT_RULES,
                ],
                "hard_fail_v1": True,
            },
        ],
    }


def _build_output_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_spec = payload["job_output_spec"]["required_artifacts_v1"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_OUTPUT_SPEC_V1",
        "future_output_namespace_v1": "ALL_TRADE_REVIEW_LEDGER_<timestamp>_MONDAY_NARROW_RETRAIN_RUN_V1",
        "required_outputs_v1": [
            {
                "output_id_v1": "TRAINING_SUMMARY",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_summary_v1.json",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "MODEL_CONFIG_MANIFEST",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_model_config_manifest_v1.json",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "FEATURE_MANIFEST",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_feature_manifest_v1.csv",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "EVAL_SUMMARY",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_eval_summary_v1.json",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "COMPARE_AGAINST_REPORT",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_head_to_head_vs_frozen_r6_r5_1_monday_r6_v1.csv",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "POCKET_REPORT",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_guard_eval_v1.csv",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "VERDICT_PACKAGE",
                "required_artifact_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_eval_verdict_package_v1.json",
                "hard_required_v1": True,
            },
            {
                "output_id_v1": "STATUS_MANIFEST_AUDIT",
                "required_artifact_v1": "manifest_v1.json + status_v1.json + consistency_audit_v1.csv",
                "hard_required_v1": True,
            },
        ],
        "inherited_required_artifacts_from_job_spec_v1": base_spec,
    }


def _build_abort_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    abort = payload["job_abort"]
    return {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_ABORT_RULES_V1",
        "abort_before_training_v1": [
            "bridge used as training surface",
            "illegal fields in matrix",
            "management/exit truth in matrix",
            "policy/decision-log field in matrix",
            "as_of_skip_xgb_* field in matrix",
            "training row count != 1689",
            "label intersection != 1689",
            "feature count != 67",
            "compare references missing",
            "output namespace not clean",
        ]
        + abort["do_not_start_if_v1"],
        "abort_or_reject_after_eval_v1": [
            "repaired_165_damage > 0",
            f"forensic trade {FORENSIC_TRADE} blocked",
            "100+/200+ blocked > 0",
            "50+ blocked > 1",
            "strongest-winner damage > 0",
            "global precision < 0.954545",
            "worst LOSO < 0.888888",
            "serious runner near-miss regression",
        ]
        + abort["reject_after_run_if_v1"],
        "automatic_invalidators_v1": abort["automatic_invalidators_v1"]
        + [
            "bridge rows/signals in training matrix",
            "missing pocket report",
            "missing compare-against report",
        ],
        "reporting_rule_v1": "Every abort must be materialized in status, consistency audit and verdict package. Do not smooth or downgrade hard fails.",
    }


def _build_next_action() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "DO_NOT_RETRAIN_SAME_NARROW_SETUP_AGAIN",
        "blocked_action_v1": "RUN_KNOWN_FAILED_NARROW_RETRAIN_AS_ACTIVE_PATH",
        "supporting_actions_v1": [
            "KEEP_THIS_SPEC_AS_HISTORICAL_CONTEXT_ONLY",
            "USE_PROTECTOR_FIRST_PATH_INSTEAD",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
        "known_failed_reference_v1": "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1",
        "training_now_v1": False,
    }


def _write_report(
    path: Path,
    runner_spec: Dict[str, Any],
    config_lock: Dict[str, Any],
    feature_manifest: Dict[str, Any],
    prelaunch: Dict[str, Any],
    output_spec: Dict[str, Any],
    abort_rules: Dict[str, Any],
    next_action: Dict[str, Any],
) -> None:
    lines = [
        "# Monday Narrow Retrain Runner Spec V1",
        "",
        "## Runner",
        f"- Job: `{runner_spec['job_name_v1']}`",
        f"- Scope: `{runner_spec['scope_v1']}`",
        f"- Training now: `{runner_spec['training_now_v1']}`",
        f"- Input: `{runner_spec['input_artifact_v1']}`",
        f"- Labels: `{runner_spec['label_artifact_v1']}`",
        "",
        "## Config",
        f"- Model family: `{config_lock['model_family_v1']}`",
        f"- Base model: `{config_lock['base_model_v1']}`",
        f"- Seed: `{config_lock['seed_v1']}`",
        f"- Compact grid: `{config_lock['compact_grid_v1']}`",
        "",
        "## Feature Manifest",
        f"- Baseline features: `{feature_manifest['baseline_feature_count_v1']}`",
        f"- New proxies: `{feature_manifest['new_proxy_feature_count_v1']}`",
        f"- Total: `{feature_manifest['total_feature_count_v1']}`",
        "",
        "## Prelaunch",
    ]
    for check in prelaunch["checks_v1"]:
        lines.append(f"- `{check['check_id_v1']}` hard_fail=`{check['hard_fail_v1']}`")
    lines.extend(["", "## Outputs"])
    for output in output_spec["required_outputs_v1"]:
        lines.append(f"- `{output['output_id_v1']}` -> `{output['required_artifact_v1']}`")
    lines.extend(["", "## Abort Rules"])
    for rule in abort_rules["abort_or_reject_after_eval_v1"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## Next Action",
            f"- `{next_action['primary_action_v1']}`",
            f"- Blocked: `{next_action['blocked_action_v1']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday narrow retrain runner/config spec V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_inputs(reports_root)
    feature_df = _feature_rows(payload["scope_feature_lock"])
    runner_spec = _build_runner_spec(payload)
    config_lock = _build_config_lock(payload)
    feature_manifest = _build_feature_manifest(payload, feature_df)
    prelaunch = _build_prelaunch_checklist(payload, feature_df)
    output_spec = _build_output_spec(payload)
    abort_rules = _build_abort_rules(payload)
    next_action = _build_next_action()

    contract = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_CONFIG_SPEC_CONTRACT_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "job_spec_dir_v1": str(payload["job_spec_dir"]),
        "scope_dir_v1": str(payload["scope_dir"]),
        "job_name_v1": JOB_NAME,
        "scope_v1": SCOPE,
        "contract_type_v1": "RUNNER_AND_CONFIG_SPEC_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "not_promo_or_freeze_v1": True,
    }

    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_SUMMARY_V1",
        "materialized_at_utc_v1": contract["materialized_at_utc_v1"],
        "extension_dir_v1": str(extension_dir),
        "job_name_v1": JOB_NAME,
        "scope_v1": SCOPE,
        "training_now_v1": False,
        "runner_spec_complete_v1": True,
        "config_lock_complete_v1": True,
        "baseline_feature_count_v1": feature_manifest["baseline_feature_count_v1"],
        "new_proxy_feature_count_v1": feature_manifest["new_proxy_feature_count_v1"],
        "total_feature_count_v1": feature_manifest["total_feature_count_v1"],
        "training_row_count_v1": runner_spec["expected_training_rows_v1"],
        "label_intersection_v1": payload["job_summary"]["exact_label_intersection_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Runner/config specification is materialized without starting training.",
                "Feature manifest contains 62 baseline features plus 5 legal proxies.",
                "Bridge rows/signals remain forbidden as training input.",
                "Prelaunch checks, output requirements and abort rules are locked.",
            ],
            "INDIKERT": [
                "This spec remains useful as historical context for the failed narrow path.",
                "Any control rerun must be explicitly marked as forensics, not active pipeline progress.",
            ],
            "IKKE_ETABLERT": [
                "That a future trained candidate beats frozen Wednesday-R6.",
                "That a future run passes safety until the runner is implemented and eval artifacts exist.",
                "That this narrow setup should be retrained again.",
            ],
        },
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / RUNNER_SPEC, runner_spec)
    _write_json(extension_dir / CONFIG_LOCK, config_lock)
    _write_json(extension_dir / FEATURE_MANIFEST, feature_manifest)
    feature_df.to_csv(extension_dir / FEATURE_MANIFEST_TABLE, index=False)
    _write_json(extension_dir / PRELAUNCH_CHECKLIST, prelaunch)
    _write_json(extension_dir / OUTPUT_SPEC, output_spec)
    _write_json(extension_dir / ABORT_RULES, abort_rules)
    _write_json(extension_dir / NEXT_ACTION, next_action)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, runner_spec, config_lock, feature_manifest, prelaunch, output_spec, abort_rules, next_action)

    audit_rows = [
        _audit_record(
            "JOB_SPEC_READY",
            "PASS" if payload["job_status"].get("failed_check_count_v1") == 0 else "FAIL",
            {"job_spec_dir_v1": str(payload["job_spec_dir"]), "failed_check_count_v1": payload["job_status"].get("failed_check_count_v1")},
        ),
        _audit_record(
            "RUNNER_SPEC_IS_NOT_TRAINING",
            "PASS" if runner_spec["training_now_v1"] is False and runner_spec["runner_may_train_v1"] is False else "FAIL",
            {"training_now_v1": runner_spec["training_now_v1"], "runner_may_train_v1": runner_spec["runner_may_train_v1"]},
        ),
        _audit_record(
            "CONFIG_LOCK_MATCHES_R6_FAMILY",
            "PASS" if config_lock["model_family_v1"] == "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY" and config_lock["seed_v1"] == 20260422 else "FAIL",
            {"model_family_v1": config_lock["model_family_v1"], "seed_v1": config_lock["seed_v1"]},
        ),
        _audit_record(
            "FEATURE_MANIFEST_67_LOCKED",
            "PASS" if feature_manifest["baseline_feature_count_v1"] == 62 and feature_manifest["new_proxy_feature_count_v1"] == 5 and len(feature_df) == 67 else "FAIL",
            {
                "baseline_feature_count_v1": feature_manifest["baseline_feature_count_v1"],
                "new_proxy_feature_count_v1": feature_manifest["new_proxy_feature_count_v1"],
                "total_feature_count_v1": len(feature_df),
            },
        ),
        _audit_record(
            "FORBIDDEN_FEATURES_EXCLUDED",
            "PASS" if not any(any(pattern in name for pattern in FORBIDDEN_PATTERNS) for name in feature_df["feature_name_v1"].astype(str)) else "FAIL",
            {"forbidden_patterns_v1": FORBIDDEN_PATTERNS},
        ),
        _audit_record(
            "NEXT_ACTION_KNOWN_FAILED_NO_GO",
            "PASS"
            if next_action["primary_action_v1"] == "DO_NOT_RETRAIN_SAME_NARROW_SETUP_AGAIN"
            and next_action["blocked_action_v1"] == "RUN_KNOWN_FAILED_NARROW_RETRAIN_AS_ACTIVE_PATH"
            else "FAIL",
            {"primary_action_v1": next_action["primary_action_v1"], "blocked_action_v1": next_action["blocked_action_v1"]},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    manifest = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_MANIFEST_V1",
        "materialized_at_utc_v1": contract["materialized_at_utc_v1"],
        "extension_dir_v1": str(extension_dir),
        "artifacts_v1": [
            CONTRACT,
            RUNNER_SPEC,
            CONFIG_LOCK,
            FEATURE_MANIFEST,
            FEATURE_MANIFEST_TABLE,
            PRELAUNCH_CHECKLIST,
            OUTPUT_SPEC,
            ABORT_RULES,
            NEXT_ACTION,
            SUMMARY,
            REPORT,
            MANIFEST,
            STATUS,
            CONSISTENCY_AUDIT,
        ],
    }
    _write_json(extension_dir / MANIFEST, manifest)
    status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "runner_implementation_allowed_next_v1": False,
        "historical_context_only_v1": True,
        "training_run_allowed_now_v1": False,
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
