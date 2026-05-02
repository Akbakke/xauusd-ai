#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

DESIGN_LOCK_PREFIX = "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_V1_"
EXTENSION_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_V1"

CONTRACT = "contract_v1.json"
EXPERIMENT_SCOPE = "protector_first_experiment_scope_v1.json"
ARCHITECTURE_CHOICE = "protector_architecture_choice_lock_v1.json"
SIGNAL_TRANSLATION_CONTRACT = "protector_signal_translation_contract_v1.json"
OBJECTIVE_LABEL_REVIEW_LOCK = "objective_and_label_review_lock_v1.json"
EVAL_MATRIX = "protector_first_eval_matrix_v1.json"
NO_GO_CONSTRAINTS = "no_go_constraints_v1.json"
IMPLEMENTATION_SHAPE = "experiment_implementation_shape_v1.json"
GO_NO_GO = "go_or_no_go_next_step_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

DESIGN_SUMMARY = "summary_v1.json"
DESIGN_LOCK = "protection_first_design_lock_v1.json"
DESIGN_ARCH_OPTIONS = "runner_protector_architecture_options_v1.json"
DESIGN_SIGNAL_TRANSLATION = "runner_protection_signal_translation_v1.json"
DESIGN_OBJECTIVE_REVIEW = "objective_label_and_head_balance_review_v1.json"
DESIGN_EVAL_CONTRACT = "protection_first_eval_contract_v1.json"
DESIGN_NO_GO = "no_go_for_same_setup_retrain_v1.json"
DESIGN_NEXT_OPTIONS = "next_experiment_shape_options_v1.json"
DESIGN_GO_NO_GO = "go_or_no_go_next_step_v1.json"


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


def _load_design_lock(reports_root: Path, design_lock_dir_arg: str | None) -> Dict[str, Any]:
    design_lock_dir = Path(design_lock_dir_arg).expanduser().resolve() if design_lock_dir_arg else _latest_dir(reports_root, DESIGN_LOCK_PREFIX)
    required = [
        DESIGN_SUMMARY,
        DESIGN_LOCK,
        DESIGN_ARCH_OPTIONS,
        DESIGN_SIGNAL_TRANSLATION,
        DESIGN_OBJECTIVE_REVIEW,
        DESIGN_EVAL_CONTRACT,
        DESIGN_NO_GO,
        DESIGN_NEXT_OPTIONS,
        DESIGN_GO_NO_GO,
    ]
    missing = [name for name in required if not (design_lock_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Design-lock dir missing required artifacts: {missing}")
    return {
        "design_lock_dir": design_lock_dir,
        "summary": _load_json(design_lock_dir / DESIGN_SUMMARY),
        "design_lock": _load_json(design_lock_dir / DESIGN_LOCK),
        "arch_options": _load_json(design_lock_dir / DESIGN_ARCH_OPTIONS),
        "signal_translation": _load_json(design_lock_dir / DESIGN_SIGNAL_TRANSLATION),
        "objective_review": _load_json(design_lock_dir / DESIGN_OBJECTIVE_REVIEW),
        "eval_contract": _load_json(design_lock_dir / DESIGN_EVAL_CONTRACT),
        "no_go": _load_json(design_lock_dir / DESIGN_NO_GO),
        "next_options": _load_json(design_lock_dir / DESIGN_NEXT_OPTIONS),
        "go_no_go": _load_json(design_lock_dir / DESIGN_GO_NO_GO),
    }


def _option_by_id(payload: Dict[str, Any], option_id: str) -> Dict[str, Any]:
    for row in payload["arch_options"]["options_v1"]:
        if row["option_id_v1"] == option_id:
            return row
    raise KeyError(f"Missing architecture option: {option_id}")


def _build_scope(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_EXPERIMENT_SCOPE_V1",
        "experiment_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_V1",
        "purpose_v1": "Test whether runner protection must get decision priority before any blocker expansion is allowed.",
        "scope_locks_v1": {
            "protection_first_not_blocker_first_v1": True,
            "shadow_only_v1": True,
            "not_freeze_promo_live_v1": True,
            "not_same_narrow_setup_rerun_v1": True,
            "not_pure_threshold_fix_v1": True,
            "not_training_now_v1": True,
            "not_replay_now_v1": True,
        },
        "what_changes_v1": [
            "Protector is evaluated before blocker expansion in shadow-eval contract.",
            "High runner-protection can veto or dampen blocker action for protected runner pockets.",
            "Winner-damage cost becomes a first-class experiment constraint rather than an after-the-fact metric.",
        ],
        "what_stays_constant_v1": [
            "Exact-only canonical training surface remains the only allowed future training surface unless separately changed.",
            "Pre-entry legality boundary remains unchanged.",
            "No bridge rows are used as training rows.",
            "Frozen Wednesday-R6 remains benchmark.",
            "Monday R5.1 remains safety reference.",
            "Failure run remains hard negative reference.",
        ],
        "why_not_rerun_v1": "The previous narrow run collapsed because blocker heads overrode protection; re-running the same blocker-first setup would test random variance, not a repaired design.",
        "source_decision_v1": payload["go_no_go"]["decision_v1"],
    }


def _build_architecture_choice(payload: Dict[str, Any]) -> Dict[str, Any]:
    chosen = _option_by_id(payload, "PROTECTOR_FIRST_VETO_OR_DAMPER")
    return {
        "layer_name_v1": "PROTECTOR_ARCHITECTURE_CHOICE_LOCK_V1",
        "chosen_primary_design_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
        "chosen_option_details_v1": chosen,
        "secondary_support_components_v1": [
            {
                "component_v1": "OBJECTIVE_REWEIGHT_WITH_HARD_WINNER_COST",
                "role_v1": "Used as candidate-selection/eval constraint so winner damage cannot be treated as cheap collateral.",
                "phase_v1": "SPEC_AND_REVIEW_BEFORE_TRAINING",
            },
            {
                "component_v1": "STRONGER_EXPLICIT_PROTECTOR_HEAD_REVIEW",
                "role_v1": "Review label/objective changes needed if a later model-runner is authorized.",
                "phase_v1": "DESIGN_REVIEW_FIRST",
            },
        ],
        "why_chosen_first_v1": "It directly repairs the observed failure mode: blocker heads got final decision power even when runner-protection signal existed.",
        "why_not_stronger_head_first_v1": "A stronger head can still fail if the decision contract lets blocker heads dominate calibrated protection.",
        "why_not_separation_model_first_v1": "Runner-vs-bad separation is promising but larger and higher-overfit risk; it should follow a simpler protector-first contract test.",
        "why_not_objective_reweight_only_first_v1": "Winner-cost reweighting is needed, but by itself may still leave ambiguous blocker/protector ordering unresolved.",
        "model_vs_decision_contract_v1": {
            "model_parts_v1": [
                "Existing/new protector score source remains model/scoring surface in a future runner.",
                "Optional later stronger protector head remains model work and is not authorized here.",
            ],
            "decision_contract_parts_v1": [
                "Protector evaluated before block action.",
                "High-protection rows require veto or dampening before block can proceed.",
                "Protector/blocker conflict must be logged as first-class eval evidence.",
            ],
        },
    }


def _build_signal_translation_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload["signal_translation"]["starting_point_v1"]
    return {
        "layer_name_v1": "PROTECTOR_SIGNAL_TRANSLATION_CONTRACT_V1",
        "raw_signals_allowed_v1": [
            "as_of_pre_entry_runner_protection_guard_score_v1",
            "runner_protector_score_or_existing_runner_output_if_present",
            "as_of_pre_entry_directional_asymmetry_score_v1",
            "as_of_pre_entry_swing_retracement_alignment_score_v1",
            "as_of_pre_entry_vol_exp_comp_score_v1",
            "protected-pocket tags from readiness/eval surface only for evaluation, not as training features",
        ],
        "raw_signals_forbidden_v1": [
            "management_exit_anchor_truth",
            "policy_decision_log_fields_as_entry_features",
            "bridge_only_rows_as_training_surface",
            "same_trade_future_mfe_mae_truth_as_entry_feature",
        ],
        "translation_contract_v1": [
            "Normalize allowed raw protection inputs into a protector-strength score.",
            "Compare protector-strength against blocker pressure before final block decision in shadow eval.",
            "If protector-strength is high and hard winner-risk is present, protector veto applies.",
            "If protector-strength is moderate and bad-risk is high, blocker score is dampened and conflict is logged.",
            "If protector-strength is low, blocker proceeds normally subject to existing hard safety guards.",
        ],
        "decision_power_v1": {
            "hard_protector_behavior_v1": "Veto block on protected 100+/200+, strongest-winner, forensic repaired trade, and repaired-165-like pockets when protection criteria trigger.",
            "soft_damper_behavior_v1": "Reduce or require higher evidence for block on ambiguous runner-near-miss and 50+ seed pockets.",
            "blocker_no_longer_uncontrolled_v1": True,
        },
        "why_previous_translation_failed_v1": payload["signal_translation"]["why_raw_signal_failed_v1"],
        "forensics_signal_context_v1": source,
    }


def _build_objective_label_review_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "OBJECTIVE_AND_LABEL_REVIEW_LOCK_V1",
        "probable_prior_objective_failure_v1": "Winner damage was penalized too weakly relative to bad-block recall, allowing high collateral damage to pass as a candidate.",
        "must_review_or_redesign_v1": [
            {
                "area_v1": "strongest_winner",
                "required_review_v1": "Treat any strongest-winner damage as hard cost, not normal false block.",
                "label_change_needed_v1": "LIKELY",
            },
            {
                "area_v1": "runner_near_miss",
                "required_review_v1": "Sharpen runner-near-miss protection labels and conflict cases versus bad-risk labels.",
                "label_change_needed_v1": "LIKELY",
            },
            {
                "area_v1": "100_200_plus_winners",
                "required_review_v1": "Give 100+/200+ winner blocks explicit zero-tolerance cost.",
                "label_change_needed_v1": "POSSIBLE",
            },
            {
                "area_v1": "repaired_165_safety",
                "required_review_v1": "Preserve repaired-pocket safety as hard eval and candidate-selection constraint.",
                "label_change_needed_v1": "POSSIBLE",
            },
        ],
        "what_must_be_reweighted_v1": [
            "winner_damage_cost",
            "runner_near_miss_block_cost",
            "100_200_plus_block_cost",
            "strongest_winner_damage_cost",
            "protector_blocker_conflict_cost",
        ],
        "held_constant_in_first_experiment_v1": [
            "No new broad feature family.",
            "No bridge-as-training-surface.",
            "No policy/controller implementation.",
            "No freeze/promo/live gate.",
        ],
        "threshold_only_insufficient_v1": True,
        "source_review_v1": payload["objective_review"],
    }


def _build_eval_matrix(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTOR_FIRST_EVAL_MATRIX_V1",
        "compare_against_v1": [
            "frozen_wednesday_r6_benchmark",
            "monday_r5_1_safety_reference",
            "monday_narrow_failure_run_hard_negative_reference",
        ],
        "hard_fail_metrics_v1": {
            "repaired_165_damage_v1": "== 0",
            "forensic_trade_blocked_v1": "false",
            "hundred_plus_mfe_blocked_v1": "== 0",
            "two_hundred_plus_mfe_blocked_v1": "== 0",
            "fifty_plus_mfe_blocked_v1": "<= 1",
            "strongest_winner_damage_v1": "== 0",
            "runner_near_miss_regression_v1": "false",
            "global_precision_v1": "must_not_collapse",
            "worst_loso_v1": "must_not_collapse",
        },
        "protection_specific_metrics_v1": {
            "protector_over_block_override_count_v1": "count rows where protector veto/damper changes blocker decision",
            "protected_winner_retention_v1": "protected winners retained divided by protected winners at risk",
            "runner_protector_effectiveness_v1": "saved runners minus protected bad trades allowed, reported by pocket",
            "blocker_vs_protector_conflict_summary_v1": "counts and score margins where blocker and protector disagree",
            "protector_saved_50_plus_v1": "count of 50+ runner seeds saved by protection",
            "protector_saved_100_200_plus_v1": "count of 100+/200+ runners saved by protection",
        },
        "monitor_only_metrics_v1": {
            "bad_block_recall_delta_v1": "monitor, not allowed to override hard winner safety",
            "tail_help_delta_v1": "monitor, not allowed to override hard winner safety",
            "protector_allow_rate_v1": "monitor for over-broad protection",
        },
        "source_eval_contract_v1": payload["eval_contract"],
    }


def _build_no_go_constraints() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NO_GO_CONSTRAINTS_V1",
        "forbidden_v1": [
            "Do not rerun same narrow setup.",
            "Do not use bridge as training surface.",
            "Do not use management/exit truth as entry features.",
            "Do not change policy/controller.",
            "Do not introduce live-gate logic.",
            "Do not use pure threshold adjustment as the main mechanism.",
            "Do not freeze or promote this experiment.",
            "Do not treat raw protection signal presence as sufficient without decision effect.",
        ],
        "hard_abort_if_seen_v1": [
            "bridge_rows_in_training_matrix",
            "management_exit_truth_in_entry_feature_matrix",
            "policy_controller_code_change",
            "live_gate_or_close_authority_change",
            "threshold_only_experiment_claimed_as_protector_first",
        ],
    }


def _build_implementation_shape() -> Dict[str, Any]:
    return {
        "layer_name_v1": "EXPERIMENT_IMPLEMENTATION_SHAPE_V1",
        "next_job_type_v1": "RUNNER_JOB_SPEC_JOB",
        "not_training_job_yet_v1": True,
        "likely_files_to_touch_v1": [
            "gx1/scripts/materialize_protector_first_shadow_experiment_runner_spec_v1.py",
            "tests/test_materialize_protector_first_shadow_experiment_runner_spec_v1.py",
            "optionally gx1/scripts/run_monday_narrow_retrain_runner_v1.py only if reused later behind explicit flag",
        ],
        "must_build_before_training_can_be_authorized_v1": [
            "protector-first runner/config spec",
            "protector/blocker conflict logging schema",
            "veto/damper decision contract in shadow-eval spec",
            "objective/label review lock for winner-cost handling",
            "prelaunch validation proving no policy/controller/live changes",
        ],
        "tests_required_v1": [
            "spec materializes without training",
            "same narrow setup rerun is rejected",
            "bridge-as-training-surface is rejected",
            "management/exit truth as entry feature is rejected",
            "protector-veto/damper metrics are required outputs",
            "hard winner-safety guards are present",
        ],
        "append_only_requirements_v1": [
            "write new artifacts under PROTECTOR_FIRST_SHADOW_EXPERIMENT namespace",
            "do not overwrite frozen R6, Monday R5.1, or Monday failure-run artifacts",
            "do not mutate canonical training surface",
        ],
    }


def _build_go_no_go() -> Dict[str, Any]:
    return {
        "layer_name_v1": "GO_OR_NO_GO_NEXT_STEP_V1",
        "decision_v1": "DESIGN_PROTECTOR_FIRST_RUNNER_SPEC_NEXT",
        "supporting_locks_v1": [
            "REVIEW_LABELS_AND_OBJECTIVE_FIRST",
            "DO_NOT_TRAIN_YET",
            "KEEP_FAILURE_RUN_AS_HARD_NEGATIVE_REFERENCE",
        ],
        "why_v1": "The experiment form is now specific enough to write a runner/config spec, but no training is authorized until that spec and prelaunch guards exist.",
    }


def _write_report(path: Path, summary: Dict[str, Any], architecture: Dict[str, Any], go_no_go: Dict[str, Any]) -> None:
    lines = [
        "# Protector-First Shadow Experiment Spec V1",
        "",
        "## Decision",
        f"- `{go_no_go['decision_v1']}`",
        "",
        "## Chosen Architecture",
        f"- `{architecture['chosen_primary_design_v1']}`",
        "",
        "## Core Lock",
        "- Protection gets decision power before blocker expansion.",
        "- This is shadow-only, not training, not replay, not threshold tuning, not policy/controller.",
        "- The previous narrow failure run remains a hard negative reference.",
        "",
        "## Next",
        "- Write protector-first runner/config spec next; do not train yet.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize protector-first shadow experiment spec after protection-first design lock.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--design-lock-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    payload = _load_design_lock(reports_root, args.design_lock_dir)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    materialized_at = _utc_now_iso()
    scope = _build_scope(payload)
    architecture = _build_architecture_choice(payload)
    signal_contract = _build_signal_translation_contract(payload)
    objective_lock = _build_objective_label_review_lock(payload)
    eval_matrix = _build_eval_matrix(payload)
    no_go = _build_no_go_constraints()
    implementation_shape = _build_implementation_shape()
    go_no_go = _build_go_no_go()

    contract = {
        "layer_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_CONTRACT_V1",
        "materialized_at_utc_v1": materialized_at,
        "design_lock_dir_v1": str(payload["design_lock_dir"]),
        "experiment_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_V1",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_threshold_tuning_v1": True,
        "not_policy_controller_change_v1": True,
        "not_freeze_promo_live_v1": True,
        "append_only_namespace_v1": str(extension_dir),
    }
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_SUMMARY_V1",
        "materialized_at_utc_v1": materialized_at,
        "design_lock_dir_v1": str(payload["design_lock_dir"]),
        "extension_dir_v1": str(extension_dir),
        "chosen_architecture_v1": architecture["chosen_primary_design_v1"],
        "decision_v1": go_no_go["decision_v1"],
        "objective_labels_must_be_reviewed_v1": True,
        "protection_gets_decision_power_v1": True,
        "do_not_train_yet_v1": True,
        "hard_status_division_v1": {
            "BEVIST": [
                "The next experiment must be protection-first and shadow-only.",
                "The same narrow setup is explicitly forbidden as a rerun.",
                "Protector-first veto/damper is locked as the first architecture to specify.",
            ],
            "INDIKERT": [
                "Objective/label balance likely needs review because winner damage was too cheap.",
                "Raw protection signals may still be useful if translated into decision force.",
            ],
            "IKKE_ETABLERT": [
                "That protector-first veto/damper will beat frozen Wednesday-R6.",
                "That labels can remain unchanged in a later training run.",
                "That any training should start before runner/config prelaunch guards are written.",
            ],
        },
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / EXPERIMENT_SCOPE, scope)
    _write_json(extension_dir / ARCHITECTURE_CHOICE, architecture)
    _write_json(extension_dir / SIGNAL_TRANSLATION_CONTRACT, signal_contract)
    _write_json(extension_dir / OBJECTIVE_LABEL_REVIEW_LOCK, objective_lock)
    _write_json(extension_dir / EVAL_MATRIX, eval_matrix)
    _write_json(extension_dir / NO_GO_CONSTRAINTS, no_go)
    _write_json(extension_dir / IMPLEMENTATION_SHAPE, implementation_shape)
    _write_json(extension_dir / GO_NO_GO, go_no_go)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, summary, architecture, go_no_go)

    artifacts = [
        CONTRACT,
        EXPERIMENT_SCOPE,
        ARCHITECTURE_CHOICE,
        SIGNAL_TRANSLATION_CONTRACT,
        OBJECTIVE_LABEL_REVIEW_LOCK,
        EVAL_MATRIX,
        NO_GO_CONSTRAINTS,
        IMPLEMENTATION_SHAPE,
        GO_NO_GO,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    audit_rows = [
        _audit_record(
            "SOURCE_DESIGN_LOCK_READ",
            "PASS" if payload["go_no_go"]["decision_v1"] == "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT" else "FAIL",
            {"source_decision_v1": payload["go_no_go"]["decision_v1"]},
        ),
        _audit_record(
            "SCOPE_IS_SHADOW_ONLY",
            "PASS" if scope["scope_locks_v1"]["shadow_only_v1"] and scope["scope_locks_v1"]["not_training_now_v1"] else "FAIL",
            scope["scope_locks_v1"],
        ),
        _audit_record(
            "ARCHITECTURE_CHOICE_LOCKED",
            "PASS" if architecture["chosen_primary_design_v1"] == "PROTECTOR_FIRST_VETO_OR_DAMPER" else "FAIL",
            {"chosen_primary_design_v1": architecture["chosen_primary_design_v1"]},
        ),
        _audit_record(
            "PROTECTION_DECISION_POWER_LOCKED",
            "PASS" if signal_contract["decision_power_v1"]["blocker_no_longer_uncontrolled_v1"] else "FAIL",
            signal_contract["decision_power_v1"],
        ),
        _audit_record(
            "NO_GO_CONSTRAINTS_LOCKED",
            "PASS" if "Do not rerun same narrow setup." in no_go["forbidden_v1"] else "FAIL",
            {"forbidden_count_v1": len(no_go["forbidden_v1"])},
        ),
        _audit_record(
            "NEXT_STEP_LOCKED",
            "PASS" if go_no_go["decision_v1"] == "DESIGN_PROTECTOR_FIRST_RUNNER_SPEC_NEXT" else "FAIL",
            go_no_go,
        ),
        _audit_record(
            "OUTPUTS_PRESENT",
            "PASS" if all((extension_dir / a).exists() for a in artifacts if a not in {MANIFEST, STATUS, CONSISTENCY_AUDIT}) else "FAIL",
            {"artifact_count_v1": len(artifacts)},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_MANIFEST_V1",
        "materialized_at_utc_v1": materialized_at,
        "extension_dir_v1": str(extension_dir),
        "source_design_lock_dir_v1": str(payload["design_lock_dir"]),
        "artifacts_v1": artifacts,
    }
    _write_json(extension_dir / MANIFEST, manifest)
    failed_checks = int(audit_df["status_v1"].astype("string").ne("PASS").sum())
    status = {
        "layer_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY" if failed_checks == 0 else "MATERIALIZED_WITH_FAILED_CHECKS",
        "failed_check_count_v1": failed_checks,
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_threshold_tuning_v1": True,
        "not_policy_controller_change_v1": True,
        "decision_v1": go_no_go["decision_v1"],
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
