#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

PROTECTOR_SPEC_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_V1_"
NARROW_RUNNER_SPEC_PREFIX = "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1_"
EXTENSION_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_SPEC_V1"

CONTRACT = "contract_v1.json"
RUNNER_SPEC = "protector_first_runner_spec_v1.json"
CONFIG_LOCK = "protector_first_config_lock_v1.json"
DECISION_CONTRACT = "protector_first_decision_contract_v1.json"
OBJECTIVE_LABEL_REVIEW = "protector_first_objective_label_review_spec_v1.json"
FEATURE_SURFACE_LOCK = "protector_first_feature_and_surface_lock_v1.json"
EVAL_VERDICT_MATRIX = "protector_first_eval_and_verdict_matrix_v1.json"
PRELAUNCH_CHECKLIST = "protector_first_prelaunch_checklist_v1.json"
ABORT_RULES = "protector_first_abort_rules_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

PROTECTOR_SUMMARY = "summary_v1.json"
PROTECTOR_SCOPE = "protector_first_experiment_scope_v1.json"
PROTECTOR_ARCHITECTURE = "protector_architecture_choice_lock_v1.json"
PROTECTOR_SIGNAL = "protector_signal_translation_contract_v1.json"
PROTECTOR_OBJECTIVE = "objective_and_label_review_lock_v1.json"
PROTECTOR_EVAL = "protector_first_eval_matrix_v1.json"
PROTECTOR_NO_GO = "no_go_constraints_v1.json"
PROTECTOR_IMPLEMENTATION = "experiment_implementation_shape_v1.json"
PROTECTOR_GO_NO_GO = "go_or_no_go_next_step_v1.json"
PROTECTOR_STATUS = "status_v1.json"

NARROW_RUNNER = "monday_narrow_retrain_runner_spec_v1.json"
NARROW_CONFIG = "monday_narrow_retrain_config_lock_v1.json"
NARROW_FEATURE = "monday_narrow_retrain_feature_manifest_v1.json"
NARROW_FEATURE_TABLE = "monday_narrow_retrain_feature_manifest_v1.csv"

JOB_NAME = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_V1"
RUNNER_NAME = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_V1"
ARCHITECTURE = "PROTECTOR_FIRST_VETO_OR_DAMPER"
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

FROZEN_WEDNESDAY_R6 = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_R5_1 = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"
NARROW_FAILURE_REFERENCE = "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1"

FORBIDDEN_FIELD_PATTERNS = [
    "as_of_skip_xgb_",
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
    "management_policy",
    "decision_log",
    "policy_log",
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


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted([p for p in reports_root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name)
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_required_dir(path: Path, required: List[str], label: str) -> Dict[str, Any]:
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing required artifacts: {missing}")
    return {name: _load_json(path / name) for name in required if name.endswith(".json")}


def _load_inputs(reports_root: Path, protector_spec_dir_arg: str | None, narrow_runner_spec_dir_arg: str | None) -> Dict[str, Any]:
    protector_dir = Path(protector_spec_dir_arg).expanduser().resolve() if protector_spec_dir_arg else _latest_dir(reports_root, PROTECTOR_SPEC_PREFIX)
    narrow_dir = Path(narrow_runner_spec_dir_arg).expanduser().resolve() if narrow_runner_spec_dir_arg else _latest_dir(reports_root, NARROW_RUNNER_SPEC_PREFIX)

    protector = _load_required_dir(
        protector_dir,
        [
            PROTECTOR_SUMMARY,
            PROTECTOR_SCOPE,
            PROTECTOR_ARCHITECTURE,
            PROTECTOR_SIGNAL,
            PROTECTOR_OBJECTIVE,
            PROTECTOR_EVAL,
            PROTECTOR_NO_GO,
            PROTECTOR_IMPLEMENTATION,
            PROTECTOR_GO_NO_GO,
            PROTECTOR_STATUS,
        ],
        "Protector-first spec dir",
    )
    narrow = _load_required_dir(
        narrow_dir,
        [
            NARROW_RUNNER,
            NARROW_CONFIG,
            NARROW_FEATURE,
        ],
        "Monday narrow runner spec dir",
    )
    feature_table_path = narrow_dir / NARROW_FEATURE_TABLE
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Monday narrow runner spec dir missing required artifact: {NARROW_FEATURE_TABLE}")
    feature_df = pd.read_csv(feature_table_path)
    return {
        "protector_dir": protector_dir,
        "narrow_dir": narrow_dir,
        "protector": protector,
        "narrow": narrow,
        "feature_df": feature_df,
    }


def _build_decision_contract(protector: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_DECISION_CONTRACT_V1",
        "architecture_v1": ARCHITECTURE,
        "protector_has_decision_power_v1": True,
        "hard_protector_veto_v1": [
            {
                "pocket_v1": "forensic_repaired_trade",
                "rule_v1": f"Never block {FORENSIC_TRADE} when it is present and covered.",
                "hard_fail_if_violated_v1": True,
            },
            {
                "pocket_v1": "repaired_165_like_pockets",
                "rule_v1": "Veto block when repaired-pocket protection criteria trigger.",
                "hard_fail_if_violated_v1": True,
            },
            {
                "pocket_v1": "strongest_winner",
                "rule_v1": "Veto block on strongest-winner protected cases.",
                "hard_fail_if_violated_v1": True,
            },
            {
                "pocket_v1": "100_plus_200_plus_winner_pockets",
                "rule_v1": "Veto block on 100+/200+ winner pockets.",
                "hard_fail_if_violated_v1": True,
            },
        ],
        "soft_damper_v1": [
            {
                "pocket_v1": "runner_near_miss",
                "rule_v1": "If protector is high/moderate, require stronger blocker evidence before blocking and report conflict.",
            },
            {
                "pocket_v1": "50_plus_mfe_seed_pockets",
                "rule_v1": "Dampen blocker pressure on 50+ MFE seed pockets unless bad-risk evidence clearly dominates.",
            },
        ],
        "blocker_evidence_when_protector_high_v1": [
            "blocker must exceed protector-adjusted evidence requirement",
            "bad-risk must dominate runner-protection margin",
            "conflict row must be emitted to blocker-vs-protector conflict summary",
            "winner hard-fail pockets override blocker evidence",
        ],
        "conflict_resolution_order_v1": [
            "1. hard safety veto pockets",
            "2. protector-first veto/damper",
            "3. blocker evidence after protector adjustment",
            "4. final shadow decision",
            "5. conflict summary/audit emission",
        ],
        "conflict_summary_required_fields_v1": [
            "candidate_uid",
            "pocket_tag",
            "blocker_score",
            "protector_score",
            "protector_action",
            "blocker_action_before_protection",
            "final_shadow_action",
            "score_margin",
            "override_or_damper_reason",
        ],
        "source_signal_contract_v1": protector[PROTECTOR_SIGNAL],
    }


def _build_feature_surface_lock(narrow: Dict[str, Any], feature_df: pd.DataFrame) -> Dict[str, Any]:
    feature_manifest = narrow[NARROW_FEATURE]
    return {
        "layer_name_v1": "PROTECTOR_FIRST_FEATURE_AND_SURFACE_LOCK_V1",
        "training_surface_v1": narrow[NARROW_RUNNER]["input_artifact_v1"],
        "training_surface_kind_v1": narrow[NARROW_RUNNER]["input_surface_kind_v1"],
        "expected_training_rows_v1": narrow[NARROW_RUNNER]["expected_training_rows_v1"],
        "bridge_as_training_surface_allowed_v1": False,
        "management_exit_truth_as_features_allowed_v1": False,
        "policy_controller_fields_allowed_v1": False,
        "feature_count_v1": int(feature_manifest["total_feature_count_v1"]),
        "baseline_feature_count_v1": int(feature_manifest["baseline_feature_count_v1"]),
        "new_proxy_feature_count_v1": int(feature_manifest["new_proxy_feature_count_v1"]),
        "new_proxy_features_reused_v1": feature_manifest["new_proxy_features_v1"],
        "feature_manifest_source_v1": NARROW_FEATURE_TABLE,
        "protector_only_metadata_eval_not_model_features_v1": [
            "forensic_repaired_trade_tag",
            "repaired_165_like_pocket_tag",
            "runner_near_miss_pocket_tag",
            "strongest_winner_tag",
            "50_plus_mfe_seed_pocket_tag",
            "100_plus_200_plus_winner_pocket_tag",
            "protector_override_reason",
            "blocker_protector_conflict_reason",
        ],
        "forbidden_field_patterns_v1": FORBIDDEN_FIELD_PATTERNS,
        "hard_fail_rule_v1": "Hard-fail if forbidden fields, bridge-only rows/signals, or management/exit truth appear in the feature matrix.",
        "feature_names_v1": feature_df["feature_name_v1"].astype(str).tolist(),
    }


def _build_objective_label_review_spec(protector: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_SPEC_V1",
        "review_required_before_training_v1": True,
        "labels_to_recheck_v1": [
            "runner_protect",
            "runner_near_miss",
            "strongest_winner",
            "100_plus_winner",
            "200_plus_winner",
            "repaired_165_safety",
            "bad_risk_vs_runner_conflict",
        ],
        "costs_to_weight_harder_v1": [
            "winner_damage_cost",
            "strongest_winner_damage_cost",
            "100_plus_block_cost",
            "200_plus_block_cost",
            "runner_near_miss_block_cost",
            "repaired_165_damage_cost",
        ],
        "winner_damage_pricing_v1": "Winner damage must be priced as hard safety cost, not ordinary false-block cost.",
        "runner_near_miss_treatment_v1": "Runner near-miss must be treated as protected conflict class; blocker may not dominate without stronger evidence.",
        "strongest_100_200_repaired_inclusion_v1": "These pockets must be explicit hard-fail eval pockets and candidate-selection constraints.",
        "held_constant_first_attempt_v1": [
            "pre-entry legality boundary",
            "exact-only training surface",
            "67-feature manifest unless a later spec explicitly changes it",
            "shadow-only/no-live/no-policy-controller scope",
        ],
        "training_stop_if_review_not_green_v1": [
            "missing runner-protect label review",
            "missing strongest-winner cost review",
            "missing 100+/200+ winner zero-tolerance review",
            "missing repaired-165 safety review",
            "missing conflict-label review for bad-risk vs runner-protection",
        ],
        "source_objective_lock_v1": protector[PROTECTOR_OBJECTIVE],
    }


def _build_eval_verdict_matrix(protector: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_EVAL_AND_VERDICT_MATRIX_V1",
        "compare_against_v1": [
            {"reference_v1": "FROZEN_WEDNESDAY_R6", "id_v1": FROZEN_WEDNESDAY_R6},
            {"reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE", "id_v1": MONDAY_R5_1},
            {"reference_v1": "NARROW_FAILURE_RUN_HARD_NEGATIVE", "id_v1": NARROW_FAILURE_REFERENCE},
        ],
        "hard_safety_requirements_v1": {
            "repaired_165_damage_v1": "== 0",
            "forensic_trade_unblocked_v1": FORENSIC_TRADE,
            "hundred_plus_mfe_blocked_v1": "== 0",
            "two_hundred_plus_mfe_blocked_v1": "== 0",
            "fifty_plus_mfe_blocked_v1": "<= 1",
            "strongest_winner_damage_v1": "== 0",
            "runner_near_miss_regression_v1": "false",
            "precision_v1": "must_not_collapse_vs_locked_floor",
            "worst_loso_v1": "must_not_collapse_vs_locked_floor",
        },
        "protection_specific_metrics_v1": [
            "protector_over_block_override_count",
            "protected_winner_retention",
            "runner_protector_effectiveness",
            "blocker_vs_protector_conflict_summary",
            "protector_saved_50_plus_count",
            "protector_saved_100_200_plus_count",
        ],
        "verdicts_v1": [
            {
                "verdict_v1": "PROTECTOR_FIRST_SAFE_AND_BETTER",
                "condition_v1": "passes all hard safety requirements and improves protection vs failure-run without unacceptable precision/LOSO collapse",
            },
            {
                "verdict_v1": "PROTECTOR_FIRST_SAFE_BUT_NOT_BETTER",
                "condition_v1": "passes safety but does not improve enough versus references",
            },
            {
                "verdict_v1": "PROTECTOR_FIRST_FAILS_SAFETY",
                "condition_v1": "any hard safety requirement fails",
            },
            {
                "verdict_v1": "PROTECTOR_FIRST_INVALID_SURFACE_OR_LEGALITY",
                "condition_v1": "bridge/truth/policy surface breach or missing contract",
            },
            {
                "verdict_v1": "NOT_ESTABLISHED",
                "condition_v1": "required eval artifacts or conflict summaries are missing",
            },
        ],
        "source_eval_matrix_v1": protector[PROTECTOR_EVAL],
    }


def _build_runner_spec(
    narrow: Dict[str, Any],
    feature_surface: Dict[str, Any],
    decision_contract: Dict[str, Any],
    eval_matrix: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_SPEC_V1",
        "job_name_v1": JOB_NAME,
        "runner_name_v1": RUNNER_NAME,
        "execution_mode_v1": "SPEC_ONLY_DO_NOT_TRAIN",
        "training_now_v1": False,
        "replay_now_v1": False,
        "policy_controller_change_v1": False,
        "input_training_surface_v1": feature_surface["training_surface_v1"],
        "training_surface_kind_v1": feature_surface["training_surface_kind_v1"],
        "expected_training_rows_v1": feature_surface["expected_training_rows_v1"],
        "feature_set_v1": {
            "feature_count_v1": feature_surface["feature_count_v1"],
            "baseline_feature_count_v1": feature_surface["baseline_feature_count_v1"],
            "new_proxy_feature_count_v1": feature_surface["new_proxy_feature_count_v1"],
            "new_proxy_features_reused_v1": feature_surface["new_proxy_features_reused_v1"],
        },
        "label_target_contract_v1": {
            "label_artifact_v1": narrow[NARROW_RUNNER]["label_artifact_v1"],
            "label_contract_v1": narrow[NARROW_RUNNER]["label_contract_v1"],
            "objective_label_review_required_before_training_v1": True,
        },
        "protector_first_decision_contract_v1": {
            "artifact_v1": DECISION_CONTRACT,
            "architecture_v1": decision_contract["architecture_v1"],
            "hard_veto_count_v1": len(decision_contract["hard_protector_veto_v1"]),
            "soft_damper_count_v1": len(decision_contract["soft_damper_v1"]),
        },
        "output_namespace_v1": "ALL_TRADE_REVIEW_LEDGER_<timestamp>_PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUN_V1",
        "required_manifests_status_v1": [
            "training_summary_v1.json",
            "protector_first_model_config_manifest_v1.json",
            "protector_first_feature_manifest_echo_v1.csv",
            "protector_first_eval_summary_v1.json",
            "protector_first_compare_against_report_v1.csv",
            "protector_first_pocket_report_v1.csv",
            "protector_first_conflict_summary_v1.csv",
            "protector_first_verdict_package_v1.json",
            "manifest_v1.json",
            "status_v1.json",
            "consistency_audit_v1.csv",
        ],
        "prelaunch_checks_artifact_v1": PRELAUNCH_CHECKLIST,
        "abort_no_go_artifact_v1": ABORT_RULES,
        "verdict_matrix_artifact_v1": EVAL_VERDICT_MATRIX,
        "verdicts_supported_v1": [row["verdict_v1"] for row in eval_matrix["verdicts_v1"]],
    }


def _build_config_lock(protector: Dict[str, Any]) -> Dict[str, Any]:
    architecture = protector[PROTECTOR_ARCHITECTURE]
    return {
        "layer_name_v1": "PROTECTOR_FIRST_CONFIG_LOCK_V1",
        "shadow_only_v1": True,
        "not_live_gate_v1": True,
        "not_policy_controller_v1": True,
        "bridge_as_training_surface_allowed_v1": False,
        "management_exit_truth_as_entry_features_allowed_v1": False,
        "architecture_v1": ARCHITECTURE,
        "model_parts_v1": architecture["model_vs_decision_contract_v1"]["model_parts_v1"],
        "decision_contract_parts_v1": architecture["model_vs_decision_contract_v1"]["decision_contract_parts_v1"],
        "can_change_in_this_experiment_v1": [
            "protector-first shadow decision contract",
            "protector/blocker conflict summary outputs",
            "objective/label review gates",
            "winner-damage candidate-selection costs if later training is authorized",
        ],
        "cannot_change_v1": [
            "live/controller behavior",
            "bridge as training surface",
            "management/exit truth as entry features",
            "policy-log or decision-log fields as entry features",
            "frozen Wednesday-R6 benchmark",
            "Monday R5.1 safety reference",
            "narrow failure-run hard negative reference",
            "training without explicit future run authorization",
        ],
        "source_architecture_choice_v1": architecture,
    }


def _build_prelaunch_checklist() -> Dict[str, Any]:
    checks = [
        ("INPUT_SURFACE_VALID", "exact-only canonical raw-state surface resolves and row population matches locked expectation"),
        ("BRIDGE_NOT_TRAINING_SURFACE", "no bridge-only rows/signals in training matrix"),
        ("FORBIDDEN_FIELDS_ABSENT", "no management/exit truth, policy/decision-log fields, as_of_skip_xgb_* or path-dynamics truth fields"),
        ("OBJECTIVE_LABEL_REVIEW_GREEN", "objective/label review is completed or explicitly locked as green before any training"),
        ("PROTECTOR_DECISION_CONTRACT_PRESENT", f"{DECISION_CONTRACT} exists and architecture is {ARCHITECTURE}"),
        ("EVAL_POCKETS_PRESENT", "forensic, repaired-165, runner near-miss, 50+/100+/200+ and strongest-winner pockets resolve"),
        ("COMPARE_REFERENCES_PRESENT", "frozen Wednesday-R6, Monday R5.1 and narrow failure-run references resolve"),
        ("OUTPUT_NAMESPACE_CLEAN", "append-only output namespace does not already exist"),
        ("EXPLICIT_RUN_FLAG_REQUIRED", "future runner refuses to train unless explicit run flag is set"),
    ]
    return {
        "layer_name_v1": "PROTECTOR_FIRST_PRELAUNCH_CHECKLIST_V1",
        "checks_v1": [
            {"check_id_v1": check_id, "required_v1": required, "hard_fail_v1": True}
            for check_id, required in checks
        ],
    }


def _build_abort_rules() -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_ABORT_RULES_V1",
        "abort_before_training_v1": [
            "bridge used as training surface",
            "forbidden feature used",
            "objective/label review not green",
            "protector contract missing",
            "forensic repaired trade not covered",
            "repaired/runner/100+/200+ eval pockets missing",
            "compare references missing",
            "output namespace dirty",
            "explicit run flag missing for any future training attempt",
        ],
        "reject_after_eval_v1": [
            "repaired_165_damage > 0",
            f"forensic trade {FORENSIC_TRADE} blocked",
            "100+/200+ blocked > 0",
            "50+ blocked > 1",
            "strongest-winner damage > 0",
            "runner near-miss worsened",
            "precision/worst LOSO collapsed",
            "protector/blocker conflict summary missing",
            "hard safety fail after eval",
        ],
        "reporting_rule_v1": "Every abort or reject must be written to status, verdict package and consistency audit; do not downgrade hard fails.",
    }


def _build_next_action() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "NEXT_AGENT_MAY_IMPLEMENT_PROTECTOR_FIRST_RUNNER",
        "blocked_action_v1": "RUN_TRAINING_NOW",
        "supporting_locks_v1": [
            "DO_NOT_TRAIN",
            "DO_NOT_REPLAY",
            "DO_NOT_TOUCH_POLICY_CONTROLLER",
            "KEEP_FAILURE_RUN_AS_HARD_NEGATIVE_REFERENCE",
            "REQUIRE_OBJECTIVE_LABEL_REVIEW_GREEN_BEFORE_TRAINING",
        ],
    }


def _write_report(path: Path, summary: Dict[str, Any], runner_spec: Dict[str, Any], decision: Dict[str, Any], action: Dict[str, Any]) -> None:
    lines = [
        "# Protector-First Shadow Experiment Runner Spec V1",
        "",
        "## Decision",
        f"- `{action['primary_action_v1']}`",
        f"- Blocked: `{action['blocked_action_v1']}`",
        "",
        "## Runner",
        f"- Job: `{runner_spec['job_name_v1']}`",
        f"- Architecture: `{summary['architecture_v1']}`",
        f"- Training now: `{runner_spec['training_now_v1']}`",
        f"- Replay now: `{runner_spec['replay_now_v1']}`",
        "",
        "## Protection",
        f"- Hard veto rules: `{len(decision['hard_protector_veto_v1'])}`",
        f"- Soft damper rules: `{len(decision['soft_damper_v1'])}`",
        "- Blocker/protector conflicts are required output, not optional debug.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize protector-first shadow experiment runner/config spec V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--protector-spec-dir", default=None)
    parser.add_argument("--narrow-runner-spec-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    payload = _load_inputs(reports_root, args.protector_spec_dir, args.narrow_runner_spec_dir)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    materialized_at = _utc_now_iso()
    decision_contract = _build_decision_contract(payload["protector"])
    feature_surface = _build_feature_surface_lock(payload["narrow"], payload["feature_df"])
    objective_review = _build_objective_label_review_spec(payload["protector"])
    eval_matrix = _build_eval_verdict_matrix(payload["protector"])
    runner_spec = _build_runner_spec(payload["narrow"], feature_surface, decision_contract, eval_matrix)
    config_lock = _build_config_lock(payload["protector"])
    prelaunch = _build_prelaunch_checklist()
    abort_rules = _build_abort_rules()
    next_action = _build_next_action()

    contract = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_CONFIG_SPEC_CONTRACT_V1",
        "materialized_at_utc_v1": materialized_at,
        "protector_spec_dir_v1": str(payload["protector_dir"]),
        "narrow_runner_spec_dir_v1": str(payload["narrow_dir"]),
        "contract_type_v1": "RUNNER_CONFIG_SPEC_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
        "not_freeze_promo_live_v1": True,
    }
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_SPEC_SUMMARY_V1",
        "materialized_at_utc_v1": materialized_at,
        "extension_dir_v1": str(extension_dir),
        "job_name_v1": JOB_NAME,
        "architecture_v1": ARCHITECTURE,
        "runner_config_spec_complete_v1": True,
        "training_now_v1": False,
        "replay_now_v1": False,
        "policy_controller_change_v1": False,
        "feature_count_v1": feature_surface["feature_count_v1"],
        "hard_veto_rule_count_v1": len(decision_contract["hard_protector_veto_v1"]),
        "soft_damper_rule_count_v1": len(decision_contract["soft_damper_v1"]),
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Runner/config spec is materialized without training, replay or policy/controller changes.",
                "Architecture is locked to PROTECTOR_FIRST_VETO_OR_DAMPER.",
                "Hard protector and soft damper decision contract is specified.",
                "Bridge/truth/policy fields remain forbidden as training inputs.",
            ],
            "INDIKERT": [
                "Objective/label review must gate any future training.",
                "Protector-first runner implementation can now be specified mechanically.",
            ],
            "IKKE_ETABLERT": [
                "That a future protector-first run will beat frozen Wednesday-R6.",
                "That objective/labels are green for training before the review is performed.",
                "That training should run now.",
            ],
        },
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / RUNNER_SPEC, runner_spec)
    _write_json(extension_dir / CONFIG_LOCK, config_lock)
    _write_json(extension_dir / DECISION_CONTRACT, decision_contract)
    _write_json(extension_dir / OBJECTIVE_LABEL_REVIEW, objective_review)
    _write_json(extension_dir / FEATURE_SURFACE_LOCK, feature_surface)
    _write_json(extension_dir / EVAL_VERDICT_MATRIX, eval_matrix)
    _write_json(extension_dir / PRELAUNCH_CHECKLIST, prelaunch)
    _write_json(extension_dir / ABORT_RULES, abort_rules)
    _write_json(extension_dir / NEXT_ACTION, next_action)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, summary, runner_spec, decision_contract, next_action)

    artifacts = [
        CONTRACT,
        RUNNER_SPEC,
        CONFIG_LOCK,
        DECISION_CONTRACT,
        OBJECTIVE_LABEL_REVIEW,
        FEATURE_SURFACE_LOCK,
        EVAL_VERDICT_MATRIX,
        PRELAUNCH_CHECKLIST,
        ABORT_RULES,
        NEXT_ACTION,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    feature_names = payload["feature_df"]["feature_name_v1"].astype(str).tolist()
    audit_rows = [
        _audit_record(
            "SOURCE_SPEC_READ",
            "PASS" if payload["protector"][PROTECTOR_STATUS]["failed_check_count_v1"] == 0 else "FAIL",
            {"protector_spec_dir_v1": str(payload["protector_dir"])},
        ),
        _audit_record(
            "ARCHITECTURE_LOCKED",
            "PASS" if config_lock["architecture_v1"] == ARCHITECTURE else "FAIL",
            {"architecture_v1": config_lock["architecture_v1"]},
        ),
        _audit_record(
            "RUNNER_SPEC_NOT_TRAINING",
            "PASS" if not runner_spec["training_now_v1"] and not runner_spec["replay_now_v1"] else "FAIL",
            {"training_now_v1": runner_spec["training_now_v1"], "replay_now_v1": runner_spec["replay_now_v1"]},
        ),
        _audit_record(
            "FEATURE_SURFACE_LOCKED",
            "PASS" if feature_surface["feature_count_v1"] == 67 and feature_surface["bridge_as_training_surface_allowed_v1"] is False else "FAIL",
            {"feature_count_v1": feature_surface["feature_count_v1"], "bridge_allowed_v1": feature_surface["bridge_as_training_surface_allowed_v1"]},
        ),
        _audit_record(
            "FORBIDDEN_FIELDS_EXCLUDED",
            "PASS" if not any(any(pattern in name for pattern in FORBIDDEN_FIELD_PATTERNS) for name in feature_names) else "FAIL",
            {"forbidden_patterns_v1": FORBIDDEN_FIELD_PATTERNS},
        ),
        _audit_record(
            "DECISION_CONTRACT_HAS_POWER",
            "PASS" if decision_contract["protector_has_decision_power_v1"] else "FAIL",
            {"hard_veto_count_v1": len(decision_contract["hard_protector_veto_v1"]), "soft_damper_count_v1": len(decision_contract["soft_damper_v1"])},
        ),
        _audit_record(
            "NEXT_ACTION_NO_TRAINING",
            "PASS" if next_action["blocked_action_v1"] == "RUN_TRAINING_NOW" else "FAIL",
            next_action,
        ),
        _audit_record(
            "OUTPUTS_PRESENT",
            "PASS" if all((extension_dir / artifact).exists() for artifact in artifacts if artifact not in {MANIFEST, STATUS, CONSISTENCY_AUDIT}) else "FAIL",
            {"artifact_count_v1": len(artifacts)},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_SPEC_MANIFEST_V1",
        "materialized_at_utc_v1": materialized_at,
        "extension_dir_v1": str(extension_dir),
        "source_protector_spec_dir_v1": str(payload["protector_dir"]),
        "source_narrow_runner_spec_dir_v1": str(payload["narrow_dir"]),
        "artifacts_v1": artifacts,
    }
    _write_json(extension_dir / MANIFEST, manifest)
    failed_checks = int(audit_df["status_v1"].astype("string").ne("PASS").sum())
    status = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_SPEC_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY" if failed_checks == 0 else "MATERIALIZED_WITH_FAILED_CHECKS",
        "failed_check_count_v1": failed_checks,
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
        "next_action_v1": next_action["primary_action_v1"],
        "training_run_allowed_now_v1": False,
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
