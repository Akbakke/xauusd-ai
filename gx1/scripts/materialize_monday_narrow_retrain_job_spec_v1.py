#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_NARROW_RETRAIN_JOB_SPEC_V1"

SCOPE_PLAN_PREFIX = "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1_"
READINESS_PREFIX = "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1_"
BRIDGE_PREFIX = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
R6_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

ENTRY_RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
ENTRY_RAW_CONTRACT = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"
ENTRY_RAW_CONTRACT_SUMMARY = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"

R6_HINDSIGHT = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
R6_POLICY_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"

CONTRACT = "contract_v1.json"
TRAINING_JOB_SPEC = "training_job_spec_lock_v1.json"
TRAINING_INPUT_CONTRACT = "training_input_contract_v1.json"
LABEL_TARGET_LOCK = "label_and_target_lock_v1.json"
MODEL_CONFIG_SPEC = "model_and_training_configuration_spec_v1.json"
OUTPUT_ARTIFACT_SPEC = "output_artifact_spec_v1.json"
EVAL_VERDICT_MATRIX = "eval_verdict_matrix_v1.json"
PRE_RUN_CHECKLIST = "pre_run_validation_checklist_v1.json"
POST_RUN_CHECKLIST = "post_run_eval_checklist_v1.json"
NO_GO_ABORT = "no_go_and_abort_protocol_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

SCOPE_PLAN = "narrow_retrain_plan_v1.json"
TRAINING_SURFACE_LOCK = "training_surface_lock_v1.json"
FEATURE_SET_LOCK = "feature_set_lock_v1.json"
OBJECTIVE_LOCK = "training_objective_and_priority_lock_v1.json"
EVAL_GUARD_PLAN = "eval_and_regression_guard_plan_v1.json"
TRAINING_IO_LOCK = "training_run_inputs_and_outputs_lock_v1.json"
STOP_CONDITIONS = "stop_conditions_and_no_go_cases_v1.json"
EXECUTION_ORDER = "narrow_retrain_execution_order_v1.json"
SCOPE_SUMMARY = "summary_v1.json"

READINESS_DECISION = "readiness_decision_v1.json"
BRIDGE_SUMMARY = "summary_v1.json"

SELECTED_PROXIES = [
    "as_of_pre_entry_vol_exp_comp_score_v1",
    "as_of_pre_entry_directional_asymmetry_score_v1",
    "as_of_pre_entry_swing_retracement_alignment_score_v1",
    "as_of_pre_entry_tail_leakage_pocket_score_v1",
    "as_of_pre_entry_runner_protection_guard_score_v1",
]

PRIMARY_HEADS = [
    {
        "head_id_v1": "runner_protector",
        "label_col_v1": "r6_label_runner_protect_v1",
        "role_v1": "PRIMARY_RUNNER_PROTECTION",
    },
    {
        "head_id_v1": "bad_risk",
        "label_col_v1": "r6_label_bad_risk_v1",
        "role_v1": "PRIMARY_BAD_RISK",
    },
]

SECONDARY_HEADS = [
    {
        "head_id_v1": "tail_control_10_50",
        "label_col_v1": "r6_label_tail_control_10_50_v1",
        "role_v1": "SECONDARY_TAIL_CONTROL",
    },
    {
        "head_id_v1": "risky_allow",
        "label_col_v1": "r6_label_risky_allow_v1",
        "role_v1": "SECONDARY_RISKY_ALLOW",
    },
    {
        "head_id_v1": "batch04_blindspot",
        "label_col_v1": "r6_label_batch04_blindspot_v1",
        "role_v1": "STABILITY_BLINDSPOT_MONITOR",
    },
]

BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"
MONDAY_R6_ROLE = "FAILURE_MINER_DIAGNOSIS_ONLY"
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"


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
    scope_dir = _latest_dir(reports_root, SCOPE_PLAN_PREFIX)
    readiness_dir = _latest_dir(reports_root, READINESS_PREFIX)
    bridge_dir = _latest_dir(reports_root, BRIDGE_PREFIX)
    ledger_dir = reports_root / CANONICAL_LEDGER_DIRNAME
    r6_dir = reports_root / R6_DIRNAME
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Missing canonical ledger dir: {ledger_dir}")
    if not r6_dir.exists():
        raise FileNotFoundError(f"Missing R6 dir: {r6_dir}")
    raw_df = pd.read_parquet(ledger_dir / ENTRY_RAW_STATE)
    hindsight_df = pd.read_parquet(r6_dir / R6_HINDSIGHT)
    return {
        "scope_dir": scope_dir,
        "readiness_dir": readiness_dir,
        "bridge_dir": bridge_dir,
        "ledger_dir": ledger_dir,
        "r6_dir": r6_dir,
        "scope_summary": _load_json(scope_dir / SCOPE_SUMMARY),
        "scope_plan": _load_json(scope_dir / SCOPE_PLAN),
        "training_surface_lock": _load_json(scope_dir / TRAINING_SURFACE_LOCK),
        "feature_set_lock": _load_json(scope_dir / FEATURE_SET_LOCK),
        "objective_lock": _load_json(scope_dir / OBJECTIVE_LOCK),
        "eval_guard_plan": _load_json(scope_dir / EVAL_GUARD_PLAN),
        "training_io_lock": _load_json(scope_dir / TRAINING_IO_LOCK),
        "stop_conditions": _load_json(scope_dir / STOP_CONDITIONS),
        "execution_order": _load_json(scope_dir / EXECUTION_ORDER),
        "readiness_decision": _load_json(readiness_dir / READINESS_DECISION),
        "bridge_summary": _load_json(bridge_dir / BRIDGE_SUMMARY),
        "raw_contract_df": pd.read_csv(ledger_dir / ENTRY_RAW_CONTRACT),
        "raw_contract_summary": _load_json(ledger_dir / ENTRY_RAW_CONTRACT_SUMMARY),
        "raw_df": raw_df,
        "hindsight_df": hindsight_df,
    }


def _feature_name_column(df: pd.DataFrame) -> str:
    for candidate in ("feature_name_v1", "field_name_v1", "feature_name"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not resolve feature-name column from {list(df.columns)}")


def _available_label_columns(hindsight_df: pd.DataFrame) -> List[str]:
    return sorted([column for column in hindsight_df.columns if str(column).startswith("r6_label_")])


def _build_training_job_spec(payload: Dict[str, Any], training_feature_count: int, label_row_count: int) -> Dict[str, Any]:
    return {
        "layer_name_v1": "TRAINING_JOB_SPEC_LOCK_V1",
        "job_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1",
        "purpose_v1": (
            "Run a narrow, runner-first, shadow-only entry retrain on the exact-only canonical Monday training surface, "
            "using the locked five legal proxies plus existing legal baseline features under the existing R6-family contract."
        ),
        "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
        "input_surface_v1": {
            "feature_surface_v1": str(payload["ledger_dir"] / ENTRY_RAW_STATE),
            "feature_row_count_v1": int(len(payload["raw_df"])),
            "label_surface_v1": str(payload["r6_dir"] / R6_HINDSIGHT),
            "label_row_count_exact_intersection_v1": label_row_count,
            "feature_count_v1": training_feature_count,
        },
        "feature_set_v1": {
            "baseline_feature_count_v1": int(payload["feature_set_lock"]["baseline_training_features_v1"]["feature_count_v1"]),
            "new_proxy_feature_count_v1": len(SELECTED_PROXIES),
            "total_locked_training_feature_count_v1": training_feature_count,
            "new_proxy_features_v1": SELECTED_PROXIES,
        },
        "target_label_contract_v1": "LOCKED_R6_HINDSIGHT_LABEL_SURFACE_FILTERED_TO_EXACT_ONLY_CANDIDATES",
        "model_policy_family_continuity_v1": {
            "continue_family_v1": "EXISTING_ENTRY_RUNNER_FIRST_SHADOW_FAMILY",
            "continue_head_family_v1": [head["head_id_v1"] for head in PRIMARY_HEADS + SECONDARY_HEADS],
            "no_new_policy_family_v1": True,
        },
        "output_namespace_rule_v1": "ALL_TRADE_REVIEW_LEDGER_<timestamp>_MONDAY_NARROW_RETRAIN_RUN_V1",
        "eval_package_v1": {
            "compare_against_v1": payload["eval_guard_plan"]["compare_against_v1"],
            "pocket_reporting_required_v1": [
                "repaired_165_pocket",
                "forensic_repaired_trade",
                "runner_near_miss_pocket",
                "50_plus_mfe_seed_pocket",
                "missed_10_50_tail_control_pocket",
                "missed_should_not_take_pocket",
                "risky_allow_pocket",
            ],
        },
        "verdict_possibilities_v1": [
            "CANDIDATE_IMPROVES_AND_HOLDS_SAFETY",
            "CANDIDATE_IMPROVES_BUT_FAILS_SAFETY",
            "CANDIDATE_SAFE_BUT_NOT_BETTER",
            "CANDIDATE_FEATURES_INSUFFICIENT",
            "CANDIDATE_INVALID_DUE_TO_LEGALITY_OR_SURFACE_BREACH",
            "NOT_ESTABLISHED",
        ],
    }


def _build_training_input_contract(payload: Dict[str, Any], training_feature_count: int, label_row_count: int) -> Dict[str, Any]:
    raw_contract_df = payload["raw_contract_df"]
    feature_col = _feature_name_column(raw_contract_df)
    return {
        "layer_name_v1": "TRAINING_INPUT_CONTRACT_V1",
        "valid_input_v1": {
            "training_feature_surface_path_v1": str(payload["ledger_dir"] / ENTRY_RAW_STATE),
            "training_feature_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "expected_training_row_count_v1": int(len(payload["raw_df"])),
            "expected_exact_label_intersection_v1": label_row_count,
            "feature_contract_path_v1": str(payload["ledger_dir"] / ENTRY_RAW_CONTRACT),
            "feature_contract_summary_path_v1": str(payload["ledger_dir"] / ENTRY_RAW_CONTRACT_SUMMARY),
            "included_feature_count_v1": training_feature_count,
            "included_feature_families_v1": payload["feature_set_lock"]["baseline_training_features_v1"]["feature_families_v1"],
            "included_new_proxies_v1": SELECTED_PROXIES,
        },
        "invalid_input_v1": {
            "bridge_surface_path_v1": str(payload["bridge_dir"] / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
            "forbidden_sources_v1": payload["feature_set_lock"]["explicit_exclusions_v1"]["forbidden_sources_v1"],
            "forbidden_field_examples_v1": payload["feature_set_lock"]["explicit_exclusions_v1"]["forbidden_field_examples_v1"],
            "bridge_rows_forbidden_v1": True,
        },
        "input_validation_hard_fail_v1": [
            "training surface path != locked exact-only parquet",
            "training row count != 1689",
            "label intersection != 1689",
            "missing selected proxy fields",
            "forbidden management/exit or policy-log fields present in training matrix",
            "any bridge-only rows included in training population",
            "target-adjacent as_of_skip_xgb_* fields included",
            "feature contract drift without explicit new lock package",
        ],
        "pre_run_checks_required_v1": [
            "surface path exact match",
            "row count exact match",
            "candidate_uid intersection exact match",
            "feature schema exact check",
            "forbidden field absence check",
        ],
        "contract_note_v1": (
            "Only exact-only canonical raw-state rows and locked legal features are valid inputs. "
            "Bridge visibility is for eval/readiness only and must hard-fail if proposed for training."
        ),
    }


def _build_label_target_lock(payload: Dict[str, Any], label_row_count: int) -> Dict[str, Any]:
    available_labels = _available_label_columns(payload["hindsight_df"])
    label_heads = PRIMARY_HEADS + SECONDARY_HEADS
    return {
        "layer_name_v1": "LABEL_AND_TARGET_LOCK_V1",
        "target_surface_v1": {
            "artifact_v1": str(payload["r6_dir"] / R6_HINDSIGHT),
            "surface_kind_v1": "R6_HINDSIGHT_LABEL_OUTCOME_TABLE_FILTERED_TO_EXACT_ONLY_CANDIDATES",
            "exact_training_intersection_row_count_v1": label_row_count,
            "available_r6_label_columns_v1": available_labels,
        },
        "locked_training_heads_v1": label_heads,
        "label_definition_v1": {
            "runner_first_priority_v1": "Expressed through runner_protector head, repaired/runner guardrails, and eval weighting priority; not by changing AS_OF legality.",
            "primary_interest_v1": [
                "r6_label_runner_protect_v1",
                "r6_label_bad_risk_v1",
            ],
            "secondary_interest_v1": [
                "r6_label_tail_control_10_50_v1",
                "r6_label_risky_allow_v1",
                "r6_label_batch04_blindspot_v1",
            ],
            "positive_negative_note_v1": (
                "Positives remain defined by the existing locked hindsight label contract from the R6 family; "
                "this spec does not reopen label semantics."
            ),
        },
        "what_must_not_change_v1": [
            "Do not redefine label semantics in this narrow retrain job.",
            "Do not replace the hindsight label surface with bridge-only eval rows.",
            "Do not collapse AS_OF and HINDSIGHT into one training table.",
        ],
    }


def _build_model_config_spec() -> Dict[str, Any]:
    return {
        "layer_name_v1": "MODEL_AND_TRAINING_CONFIGURATION_SPEC_V1",
        "model_family_v1": {
            "base_model_v1": "XGBClassifier per head",
            "head_family_v1": [head["head_id_v1"] for head in PRIMARY_HEADS + SECONDARY_HEADS],
            "continuity_v1": "Continue the same five-head R6-style shadow family; do not introduce new model families.",
        },
        "training_mode_v1": "SHADOW_RESEARCH_ONLY_NOT_LIVE_NOT_CONTROLLER",
        "evaluation_structure_v1": {
            "walkforward_required_v1": True,
            "loso_required_v1": True,
            "rolling_window_required_v1": True,
            "batch_weeks_v1": 15,
            "batch04_batch05_reporting_v1": "BATCH_04 must be reported; BATCH_05 must be null if absent, not fail.",
        },
        "reproducibility_v1": {
            "seed_v1": 20260422,
            "n_jobs_v1": 4,
            "expected_ledger_count_v1": 1689,
            "compact_grid_only_v1": True,
        },
        "default_model_hyperparams_v1": {
            "n_estimators_v1": 800,
            "early_stopping_rounds_v1": 60,
            "learning_rate_v1": 0.025,
            "max_depth_v1": 3,
            "tree_method_v1": "hist",
        },
        "must_hold_constant_vs_prior_r6_v1": [
            "five-head family continuity",
            "shadow-only mode",
            "walkforward + LOSO + rolling eval structure",
            "benchmark hierarchy",
            "runner-first safety contract",
        ],
        "allowed_to_change_in_this_narrow_slice_v1": [
            "feature matrix with the five new legal proxies included",
            "candidate grid restricted to compact-only mode",
            "exact-only row count expectation from 1971 fullcoverage to 1689 exact-only",
        ],
        "not_allowed_to_change_v1": [
            "policy family",
            "bridge-as-training-surface rule",
            "safety contract",
            "promotion status",
            "controller behavior",
        ],
    }


def _build_output_artifact_spec() -> Dict[str, Any]:
    return {
        "layer_name_v1": "OUTPUT_ARTIFACT_SPEC_V1",
        "future_output_namespace_v1": "ALL_TRADE_REVIEW_LEDGER_<timestamp>_MONDAY_NARROW_RETRAIN_RUN_V1",
        "required_artifacts_v1": [
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_summary_v1.json",
                "filetype_v1": "json",
                "minimum_content_v1": "top-level training summary, selected candidate, comparator deltas, safety verdict",
                "why_needed_v1": "Human-readable primary outcome and machine-readable summary.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_model_family_bakeoff_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "candidate rows, thresholds, policy family, score, global metrics",
                "why_needed_v1": "Compare candidate families under the locked narrow scope.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_threshold_calibration_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "calibration and threshold rows per head/candidate",
                "why_needed_v1": "Trace threshold choices and compare with prior R6 family.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_walkforward_metrics_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "walkforward metrics per reference and selected candidate",
                "why_needed_v1": "Enforce chronological eval consistency.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_loso_metrics_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "LOSO metrics, per-slice safety status, batch04/batch05 reporting",
                "why_needed_v1": "Protect against worst-slice regression.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_rolling_window_metrics_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "rolling window metrics per reference and selected candidate",
                "why_needed_v1": "Stress robustness beyond one split view.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_policy_prediction_view_v1.parquet",
                "filetype_v1": "parquet",
                "minimum_content_v1": "candidate_uid-level predictions and selected policy decision fields",
                "why_needed_v1": "Supports downstream pocket reporting and head-to-head comparison.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_guard_eval_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "repaired, forensic, runner near-miss, 50+/100+/200+ pockets",
                "why_needed_v1": "Direct proof of safety on the most critical pockets.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_head_to_head_vs_frozen_r6_r5_1_monday_r6_v1.csv",
                "filetype_v1": "csv",
                "minimum_content_v1": "selected candidate vs benchmark, safety reference, and failure-miner",
                "why_needed_v1": "Locks the required comparator hierarchy into the run output.",
            },
            {
                "artifact_name_v1": "shadow_meta_all_trade_review_monday_narrow_retrain_eval_verdict_package_v1.json",
                "filetype_v1": "json",
                "minimum_content_v1": "final verdict, hard-fail reasons, next-step recommendation",
                "why_needed_v1": "One authoritative verdict artifact per later training run.",
            },
            {
                "artifact_name_v1": "manifest_v1.json / status_v1.json / consistency_audit_v1.csv",
                "filetype_v1": "json+csv",
                "minimum_content_v1": "artifact list, job status, audit pass/fail rows",
                "why_needed_v1": "Operational traceability and guard enforcement.",
            },
        ],
    }


def _build_eval_verdict_matrix() -> Dict[str, Any]:
    return {
        "layer_name_v1": "EVAL_VERDICT_MATRIX_V1",
        "verdicts_v1": [
            {
                "verdict_v1": "CANDIDATE_IMPROVES_AND_HOLDS_SAFETY",
                "trigger_v1": [
                    "all hard safety guards pass",
                    "bad blocks > Monday-native R6",
                    "tail help > Monday-native R6",
                    "global precision and worst LOSO hold at or above locked floors",
                ],
                "hard_fail_overrides_v1": [
                    "any legality or surface breach",
                    "any repaired/forensic/winner hard fail",
                ],
                "next_step_v1": "candidate may proceed to later shadow-candidate review, not live gate",
            },
            {
                "verdict_v1": "CANDIDATE_IMPROVES_BUT_FAILS_SAFETY",
                "trigger_v1": [
                    "some metrics improve over Monday-native R6",
                    "one or more hard safety guards fail",
                ],
                "hard_fail_overrides_v1": ["safety failure makes candidate unusable regardless of metric gain"],
                "next_step_v1": "reject as usable candidate; keep as failure-miner only",
            },
            {
                "verdict_v1": "CANDIDATE_SAFE_BUT_NOT_BETTER",
                "trigger_v1": [
                    "all hard safety guards pass",
                    "no meaningful improvement over Monday-native R6 on primary objectives",
                ],
                "hard_fail_overrides_v1": [],
                "next_step_v1": "keep benchmark/reference hierarchy unchanged; likely features still insufficient",
            },
            {
                "verdict_v1": "CANDIDATE_FEATURES_INSUFFICIENT",
                "trigger_v1": [
                    "training completes",
                    "candidate neither improves meaningfully nor justifies safety confidence uplift",
                ],
                "hard_fail_overrides_v1": [],
                "next_step_v1": "mine failures and consider another narrow feature uplift",
            },
            {
                "verdict_v1": "CANDIDATE_INVALID_DUE_TO_LEGALITY_OR_SURFACE_BREACH",
                "trigger_v1": [
                    "bridge used as training surface",
                    "forbidden feature family appears in training",
                    "AS_OF/HINDSIGHT boundary broken",
                ],
                "hard_fail_overrides_v1": ["automatic invalidation"],
                "next_step_v1": "reject run entirely and repair job runner/spec",
            },
            {
                "verdict_v1": "NOT_ESTABLISHED",
                "trigger_v1": [
                    "required outputs missing",
                    "required audits not run",
                    "evaluation incomplete",
                ],
                "hard_fail_overrides_v1": ["missing verdict package or missing pocket audit"],
                "next_step_v1": "treat run as incomplete, not as model evidence",
            },
        ],
    }


def _build_pre_run_checklist() -> Dict[str, Any]:
    return {
        "layer_name_v1": "PRE_RUN_VALIDATION_CHECKLIST_V1",
        "checks_v1": [
            "training feature surface path matches locked exact-only parquet",
            "training feature row count = 1689",
            "feature schema matches locked contract",
            "all five new proxies present",
            "no forbidden management/exit or policy-log fields in training matrix",
            "no bridge-only rows proposed for training",
            "compare-against references resolve to frozen Wednesday-R6, Monday R5.1, Monday-native R6",
            "contract/spec files present",
            "output namespace resolved to append-only run dir",
            "seed/config lock present",
            "no pre-run no-go case already triggered",
        ],
    }


def _build_post_run_checklist() -> Dict[str, Any]:
    return {
        "layer_name_v1": "POST_RUN_EVAL_CHECKLIST_V1",
        "checks_v1": [
            "all required output artifacts exist",
            "eval/regression guards executed",
            "pocket guard report exists",
            "repaired-165 and forensic trade verified explicitly",
            "runner near-miss pocket evaluated explicitly",
            "compare-against report vs frozen R6 / R5.1 / Monday R6 exists",
            "final verdict package materialized",
            "no-go cases checked post-run",
            "candidate status explicitly locked as invalid / failure-miner / safe-but-not-better / improved",
        ],
    }


def _build_no_go_abort_protocol() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NO_GO_AND_ABORT_PROTOCOL_V1",
        "do_not_start_if_v1": [
            "readiness or scope decision is no longer green for planning",
            "training surface mismatch",
            "bridge proposed as training surface",
            "legality check fails",
            "forbidden field detected pre-run",
        ],
        "reject_after_run_if_v1": [
            "repaired_165_damage > 0",
            f"forensic trade {FORENSIC_TRADE} remains blocked",
            "100+/200+ blocked > 0",
            "strongest-winner damage > 0",
            "50+ MFE blocked > 1",
            "global precision or worst LOSO regresses below locked floor",
            "runner near-miss worsens without compensating safety evidence",
        ],
        "automatic_invalidators_v1": [
            "legality breach",
            "surface breach",
            "missing required artifacts",
            "missing final verdict package",
        ],
        "reporting_rule_v1": "Hard fails must be explicit in status, consistency audit, and eval verdict package; no smoothing or silent downgrade.",
    }


def _build_next_action() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "NEXT_AGENT_MAY_WRITE_TRAINING_RUNNER_SPEC",
        "supporting_actions_v1": [
            "NEXT_AGENT_MAY_PREPARE_CONFIGS_BUT_NOT_RUN",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
    }


def _write_report(
    path: Path,
    training_job_spec: Dict[str, Any],
    input_contract: Dict[str, Any],
    label_lock: Dict[str, Any],
    model_config: Dict[str, Any],
    output_spec: Dict[str, Any],
    verdict_matrix: Dict[str, Any],
    pre_run: Dict[str, Any],
    post_run: Dict[str, Any],
    no_go: Dict[str, Any],
    next_action: Dict[str, Any],
) -> None:
    lines = [
        "# Monday Narrow Retrain Job Spec V1",
        "",
        "## Job",
        f"- Job name: `{training_job_spec['job_name_v1']}`",
        f"- Scope: `{training_job_spec['scope_v1']}`",
        f"- Purpose: {training_job_spec['purpose_v1']}",
        "",
        "## Input Lock",
        f"- Training surface: `{input_contract['valid_input_v1']['training_feature_surface_path_v1']}`",
        f"- Expected training rows: `{input_contract['valid_input_v1']['expected_training_row_count_v1']}`",
        f"- Expected exact label intersection: `{input_contract['valid_input_v1']['expected_exact_label_intersection_v1']}`",
        "",
        "## Labels",
        f"- Label surface: `{label_lock['target_surface_v1']['artifact_v1']}`",
        f"- Exact label rows: `{label_lock['target_surface_v1']['exact_training_intersection_row_count_v1']}`",
        "",
        "## Model Config",
        f"- Base model: `{model_config['model_family_v1']['base_model_v1']}`",
        f"- Head family: `{model_config['model_family_v1']['head_family_v1']}`",
        f"- Seed: `{model_config['reproducibility_v1']['seed_v1']}`",
        "",
        "## Outputs",
    ]
    for artifact in output_spec["required_artifacts_v1"]:
        lines.append(f"- `{artifact['artifact_name_v1']}`: {artifact['why_needed_v1']}")
    lines.extend(
        [
            "",
            "## Verdicts",
        ]
    )
    for row in verdict_matrix["verdicts_v1"]:
        lines.append(f"- `{row['verdict_v1']}`")
    lines.extend(
        [
            "",
            "## Pre-Run Checklist",
        ]
    )
    for check in pre_run["checks_v1"]:
        lines.append(f"- {check}")
    lines.extend(
        [
            "",
            "## Post-Run Checklist",
        ]
    )
    for check in post_run["checks_v1"]:
        lines.append(f"- {check}")
    lines.extend(
        [
            "",
            "## Abort / No-Go",
        ]
    )
    for item in no_go["reject_after_run_if_v1"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Action",
            f"- `{next_action['primary_action_v1']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday narrow retrain job spec V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_inputs(reports_root)

    raw_candidate_ids = set(payload["raw_df"]["candidate_uid"].astype("string"))
    hindsight_candidate_ids = set(payload["hindsight_df"]["candidate_uid"].astype("string"))
    exact_label_intersection = len(raw_candidate_ids & hindsight_candidate_ids)

    baseline_feature_count = int(payload["feature_set_lock"]["baseline_training_features_v1"]["feature_count_v1"])
    total_feature_count = baseline_feature_count + len(SELECTED_PROXIES)

    training_job_spec = _build_training_job_spec(payload, training_feature_count=total_feature_count, label_row_count=exact_label_intersection)
    training_input_contract = _build_training_input_contract(payload, training_feature_count=total_feature_count, label_row_count=exact_label_intersection)
    label_target_lock = _build_label_target_lock(payload, label_row_count=exact_label_intersection)
    model_config = _build_model_config_spec()
    output_spec = _build_output_artifact_spec()
    verdict_matrix = _build_eval_verdict_matrix()
    pre_run = _build_pre_run_checklist()
    post_run = _build_post_run_checklist()
    no_go = _build_no_go_abort_protocol()
    next_action = _build_next_action()

    _write_json(extension_dir / TRAINING_JOB_SPEC, training_job_spec)
    _write_json(extension_dir / TRAINING_INPUT_CONTRACT, training_input_contract)
    _write_json(extension_dir / LABEL_TARGET_LOCK, label_target_lock)
    _write_json(extension_dir / MODEL_CONFIG_SPEC, model_config)
    _write_json(extension_dir / OUTPUT_ARTIFACT_SPEC, output_spec)
    _write_json(extension_dir / EVAL_VERDICT_MATRIX, verdict_matrix)
    _write_json(extension_dir / PRE_RUN_CHECKLIST, pre_run)
    _write_json(extension_dir / POST_RUN_CHECKLIST, post_run)
    _write_json(extension_dir / NO_GO_ABORT, no_go)
    _write_json(extension_dir / NEXT_ACTION, next_action)

    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_JOB_SPEC_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "job_name_v1": training_job_spec["job_name_v1"],
        "scope_v1": training_job_spec["scope_v1"],
        "training_surface_v1": training_input_contract["valid_input_v1"]["training_feature_surface_path_v1"],
        "training_row_count_v1": training_input_contract["valid_input_v1"]["expected_training_row_count_v1"],
        "exact_label_intersection_v1": training_input_contract["valid_input_v1"]["expected_exact_label_intersection_v1"],
        "total_training_feature_count_v1": training_job_spec["input_surface_v1"]["feature_count_v1"],
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "next_action_v1": next_action["primary_action_v1"],
        "training_now_v1": False,
        "hard_status_division_v1": {
            "BEVIST": [
                "The later training job is now specified tightly enough to be executed mechanically by a future agent.",
                "Exact-only canonical raw-state remains the only allowed training surface.",
                "The five new proxies plus baseline features are locked as the training feature set.",
                "Hard guards, verdicts, and abort conditions are all specified before any run starts.",
            ],
            "INDIKERT": [
                "The next agent may write the concrete training runner/config spec without reopening strategy scope.",
                "The future run can stay comparable to the existing R6 family while remaining narrow.",
            ],
            "IKKE_ETABLERT": [
                "That the future run will beat frozen Wednesday-R6.",
                "That the future run will be safe until the guards are actually executed on real outputs.",
            ],
        },
    }
    _write_json(extension_dir / SUMMARY, summary)

    contract = {
        "layer_name_v1": "CONTRACT_V1",
        "job_v1": "MONDAY_NARROW_RETRAIN_JOB_SPEC_V1",
        "read_only_v1": True,
        "not_replay_v1": True,
        "not_training_v1": True,
        "not_policy_change_v1": True,
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "do_not_use_bridge_as_training_surface_v1": True,
    }
    _write_json(extension_dir / CONTRACT, contract)

    _write_report(
        extension_dir / REPORT,
        training_job_spec=training_job_spec,
        input_contract=training_input_contract,
        label_lock=label_target_lock,
        model_config=model_config,
        output_spec=output_spec,
        verdict_matrix=verdict_matrix,
        pre_run=pre_run,
        post_run=post_run,
        no_go=no_go,
        next_action=next_action,
    )

    manifest = {
        "layer_name_v1": "MANIFEST_V1",
        "generated_at_utc_v1": _utc_now_iso(),
        "artifacts_v1": [
            CONTRACT,
            TRAINING_JOB_SPEC,
            TRAINING_INPUT_CONTRACT,
            LABEL_TARGET_LOCK,
            MODEL_CONFIG_SPEC,
            OUTPUT_ARTIFACT_SPEC,
            EVAL_VERDICT_MATRIX,
            PRE_RUN_CHECKLIST,
            POST_RUN_CHECKLIST,
            NO_GO_ABORT,
            NEXT_ACTION,
            SUMMARY,
            REPORT,
        ],
    }
    _write_json(extension_dir / MANIFEST, manifest)

    audit_rows = [
        _audit_record(
            "READINESS_SCOPE_PLAN_PRESENT",
            "PASS" if payload["readiness_decision"].get("decision_v1") == "READY_TO_PLAN_NARROW_RETRAIN" else "FAIL",
            {"readiness_decision_v1": payload["readiness_decision"].get("decision_v1")},
        ),
        _audit_record(
            "TRAINING_SURFACE_EXACT_ONLY_LOCKED",
            "PASS" if training_input_contract["valid_input_v1"]["expected_training_row_count_v1"] == 1689 else "FAIL",
            {"training_row_count_v1": training_input_contract["valid_input_v1"]["expected_training_row_count_v1"]},
        ),
        _audit_record(
            "LABEL_INTERSECTION_LOCKED",
            "PASS" if exact_label_intersection == 1689 else "FAIL",
            {"exact_label_intersection_v1": exact_label_intersection},
        ),
        _audit_record(
            "SELECTED_PROXY_COUNT_LOCKED",
            "PASS" if len(SELECTED_PROXIES) == 5 else "FAIL",
            {"selected_proxy_count_v1": len(SELECTED_PROXIES)},
        ),
        _audit_record(
            "NEXT_ACTION_IS_SPEC_ONLY",
            "PASS" if next_action["primary_action_v1"] == "NEXT_AGENT_MAY_WRITE_TRAINING_RUNNER_SPEC" else "FAIL",
            {"primary_action_v1": next_action["primary_action_v1"]},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_JOB_SPEC_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "not_replay_v1": True,
        "not_training_v1": True,
        "not_policy_change_v1": True,
        "spec_only_v1": True,
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
