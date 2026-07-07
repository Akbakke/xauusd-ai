#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

FORENSICS_PREFIX = "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_V1_"
EXTENSION_PREFIX = "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_V1"

CONTRACT = "contract_v1.json"
PROTECTION_FIRST_DESIGN_LOCK = "protection_first_design_lock_v1.json"
RUNNER_ARCHITECTURE_OPTIONS = "runner_protector_architecture_options_v1.json"
RUNNER_ARCHITECTURE_OPTIONS_TABLE = "runner_protector_architecture_options_v1.csv"
SIGNAL_TRANSLATION = "runner_protection_signal_translation_v1.json"
OBJECTIVE_LABEL_HEAD_BALANCE = "objective_label_and_head_balance_review_v1.json"
EVAL_CONTRACT = "protection_first_eval_contract_v1.json"
NO_GO_SAME_SETUP = "no_go_for_same_setup_retrain_v1.json"
NEXT_EXPERIMENT_OPTIONS = "next_experiment_shape_options_v1.json"
NEXT_EXPERIMENT_OPTIONS_TABLE = "next_experiment_shape_options_v1.csv"
GO_NO_GO = "go_or_no_go_next_step_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

FORENSICS_SUMMARY = "summary_v1.json"
FORENSICS_GO_NO_GO = "go_or_no_go_next_step_v1.json"
FORENSICS_RUNNER_ANALYSIS = "runner_protection_failure_analysis_v1.json"
FORENSICS_FEATURE_REVIEW = "feature_proxy_behavior_review_v1.json"
FORENSICS_STRONGEST = "strongest_winner_damage_forensics_v1.json"
FORENSICS_TAIL = "tail_help_vs_bad_block_decomposition_v1.json"


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


def _load_forensics(reports_root: Path, forensics_dir_arg: str | None) -> Dict[str, Any]:
    forensics_dir = Path(forensics_dir_arg).expanduser().resolve() if forensics_dir_arg else _latest_dir(reports_root, FORENSICS_PREFIX)
    required = [
        FORENSICS_SUMMARY,
        FORENSICS_GO_NO_GO,
        FORENSICS_RUNNER_ANALYSIS,
        FORENSICS_FEATURE_REVIEW,
        FORENSICS_STRONGEST,
        FORENSICS_TAIL,
    ]
    missing = [name for name in required if not (forensics_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Forensics dir missing required artifacts: {missing}")
    return {
        "forensics_dir": forensics_dir,
        "summary": _load_json(forensics_dir / FORENSICS_SUMMARY),
        "go_no_go": _load_json(forensics_dir / FORENSICS_GO_NO_GO),
        "runner_analysis": _load_json(forensics_dir / FORENSICS_RUNNER_ANALYSIS),
        "feature_review": _load_json(forensics_dir / FORENSICS_FEATURE_REVIEW),
        "strongest": _load_json(forensics_dir / FORENSICS_STRONGEST),
        "tail": _load_json(forensics_dir / FORENSICS_TAIL),
    }


def _build_design_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTION_FIRST_DESIGN_LOCK_V1",
        "primary_conclusion_v1": "BLOCKER_HEADS_OVERRODE_RUNNER_PROTECTION",
        "locked_direction_v1": "PROTECTION_FIRST_NOT_BLOCKER_FIRST",
        "explicit_locks_v1": {
            "blocker_heads_overrode_runner_protection_v1": True,
            "do_not_retrain_same_narrow_setup_now_v1": True,
            "next_phase_must_be_protection_first_v1": True,
            "runner_protection_cannot_be_only_another_feature_v1": True,
        },
        "evidence_v1": {
            "main_failure_v1": payload["summary"]["main_failure_v1"],
            "global_precision_v1": payload["summary"]["global_precision_v1"],
            "strongest_winner_damage_v1": payload["summary"]["strongest_winner_damage_v1"],
            "runner_near_miss_blocked_v1": payload["summary"]["runner_near_miss_blocked_v1"],
            "runner_analysis_v1": payload["runner_analysis"]["main_failure_mode_v1"],
        },
        "design_note_v1": "Protection must become an ordering/constraint/objective property, not merely a weak input feature competing with blocker heads.",
    }


def _architecture_options() -> tuple[Dict[str, Any], pd.DataFrame]:
    rows = [
        {
            "option_id_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
            "definition_v1": "Evaluate runner-protection before block decision; high protection vetoes or dampens blocker output in shadow eval.",
            "how_protection_strengthens_v1": "Protection changes decision ordering, so blocker heads cannot override protected runner pockets by default.",
            "difference_from_current_v1": "Current setup uses protection as a learned head/feature; this adds explicit protection-first decision semantics.",
            "benefits_v1": "Directly targets 50+/100+/200+ and strongest-winner damage.",
            "risk_v1": "May reduce bad-block recall if protector is too broad.",
            "r6_family_compatible_v1": True,
            "requires_new_label_or_objective_v1": "Decision logic first; labels can stay initially, but eval contract must change.",
            "priority_v1": "HIGH",
        },
        {
            "option_id_v1": "STRONGER_EXPLICIT_PROTECTOR_HEAD",
            "definition_v1": "Train protector head with stronger runner/near-miss labels and calibration target.",
            "how_protection_strengthens_v1": "Raises model runner_protector probability on known runner pockets before blocker combination.",
            "difference_from_current_v1": "Current runner_protector under-asserted protection; this changes head objective/weighting/calibration.",
            "benefits_v1": "Keeps R6-style five-head family intact.",
            "risk_v1": "Still fails if decision rule lets blocker heads dominate calibrated protector.",
            "r6_family_compatible_v1": True,
            "requires_new_label_or_objective_v1": "Requires objective/weighting change; labels may need sharpening.",
            "priority_v1": "HIGH",
        },
        {
            "option_id_v1": "RUNNER_VS_BAD_SEPARATION_MODEL",
            "definition_v1": "Add explicit separation model for risky setup vs late runner seed, used before block expansion.",
            "how_protection_strengthens_v1": "Separates ambiguous bad-looking runners from genuine bad-risk cases.",
            "difference_from_current_v1": "Current heads predict bad and runner independently; this models the conflict directly.",
            "benefits_v1": "Targets runner near-miss and strongest-winner collateral.",
            "risk_v1": "More complexity and risk of overfitting if not tightly scoped.",
            "r6_family_compatible_v1": "PARTIAL",
            "requires_new_label_or_objective_v1": "Likely requires new pairwise/conflict labels.",
            "priority_v1": "MEDIUM",
        },
        {
            "option_id_v1": "WINNER_DAMAGE_COST_REWEIGHT_OBJECTIVE",
            "definition_v1": "Reweight training/eval objective so winner damage carries much higher cost than missed bad block.",
            "how_protection_strengthens_v1": "Makes 50+/100+/200+ and strongest-winner mistakes expensive during candidate selection.",
            "difference_from_current_v1": "Current setup optimizes blocker recall enough to allow large collateral damage.",
            "benefits_v1": "Aligns model selection with hard safety contract.",
            "risk_v1": "Can become overly conservative if not paired with good bad-risk discrimination.",
            "r6_family_compatible_v1": True,
            "requires_new_label_or_objective_v1": "Requires objective/candidate-selection reweighting.",
            "priority_v1": "HIGH",
        },
    ]
    payload = {"layer_name_v1": "RUNNER_PROTECTOR_ARCHITECTURE_OPTIONS_V1", "options_v1": rows}
    return payload, pd.DataFrame(rows)


def _signal_translation(payload: Dict[str, Any]) -> Dict[str, Any]:
    guard = payload["runner_analysis"]["guard_score_diagnosis_v1"]
    return {
        "layer_name_v1": "RUNNER_PROTECTION_SIGNAL_TRANSLATION_V1",
        "starting_point_v1": {
            "raw_runner_guard_score_exists_v1": True,
            "raw_runner_guard_mean_on_blocked_runners_v1": guard["raw_runner_guard_mean_on_blocked_runners_v1"],
            "model_runner_protector_mean_on_blocked_runners_v1": guard["model_runner_protector_mean_on_blocked_runners_v1"],
            "bad_score_mean_on_blocked_runners_v1": guard["bad_score_mean_on_blocked_runners_v1"],
        },
        "why_raw_signal_failed_v1": "The raw guard entered as an ordinary feature. The learned runner_protector output remained tiny on blocked runners, so blocker heads won the decision.",
        "problem_classification_v1": {
            "weak_label_v1": "POSSIBLE",
            "weak_objective_v1": "LIKELY",
            "wrong_head_balance_v1": "LIKELY",
            "wrong_decision_contract_v1": "PROVEN_BY_COLLAPSE",
            "combination_v1": "LIKELY_MAIN_CAUSE",
        },
        "missing_translation_v1": "A protection score must become a calibrated veto/damper or high-cost constraint, not merely a feature competing with bad/risky/tail heads.",
    }


def _objective_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "OBJECTIVE_LABEL_AND_HEAD_BALANCE_REVIEW_V1",
        "does_current_objective_reward_too_much_blocking_v1": True,
        "winner_damage_penalized_too_weakly_v1": True,
        "needs_stronger_negative_costs_v1": [
            "runner_near_miss",
            "strongest_winner",
            "50_plus_mfe",
            "100_plus_mfe",
            "200_plus_mfe",
        ],
        "labels_need_sharpening_v1": "INDICATED_FOR_RUNNER_PROTECTOR_AND_RUNNER_VS_BAD_CONFLICT",
        "head_balance_must_change_v1": True,
        "threshold_only_solution_v1": "INSUFFICIENT",
        "why_v1": "The candidate had more bad blocks than Monday-native R6 but destroyed precision and winner safety, which is an objective/head-balance failure, not a marginal threshold issue.",
    }


def _eval_contract() -> Dict[str, Any]:
    return {
        "layer_name_v1": "PROTECTION_FIRST_EVAL_CONTRACT_V1",
        "hard_fail_guardrails_v1": {
            "repaired_165_damage_v1": "== 0",
            "forensic_trade_blocked_v1": "false",
            "hundred_plus_mfe_blocked_v1": "== 0",
            "two_hundred_plus_mfe_blocked_v1": "== 0",
            "fifty_plus_mfe_blocked_v1": "<= 1",
            "strongest_winner_damage_v1": "== 0",
            "runner_near_miss_regression_v1": "false",
            "global_precision_floor_v1": "must not collapse below locked floor",
            "worst_loso_floor_v1": "must not collapse below locked floor",
        },
        "new_protection_metrics_v1": [
            "protector_veto_count_v1",
            "protector_saved_50_plus_count_v1",
            "protector_saved_100_200_plus_count_v1",
            "blocked_runner_protector_score_distribution_v1",
            "bad_score_minus_runner_score_margin_on_blocked_winners_v1",
            "runner_near_miss_saved_vs_blocked_v1",
        ],
        "monitor_only_v1": [
            "bad_blocks_delta_vs_monday_failure_run",
            "tail_help_delta_vs_monday_failure_run",
            "protector_over_allow_rate",
        ],
        "decision_effect_requirement_v1": "Protection must show actual decision effect, not just raw signal presence.",
    }


def _no_go_same_setup() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NO_GO_FOR_SAME_SETUP_RETRAIN_V1",
        "no_go_locks_v1": [
            "Do not retrain same narrow setup again.",
            "Do not use pure threshold adjustment as the main fix.",
            "Do not just try a new random run.",
            "Do not treat this as marginal regression.",
            "Do not let blocker-first objective remain dominant.",
        ],
        "required_before_retrain_v1": [
            "Protection-first decision or objective design is locked.",
            "Runner-protection head/objective/label balance is explicitly strengthened.",
            "Hard winner-safety metrics are included before candidate selection.",
        ],
    }


def _experiment_options() -> tuple[Dict[str, Any], pd.DataFrame]:
    rows = [
        {
            "experiment_shape_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT",
            "what_changes_v1": "Add protector-first veto/damper in shadow eval and measure saved runners vs lost bad blocks.",
            "what_stays_constant_v1": "Exact-only training surface, feature legality, benchmark hierarchy.",
            "change_size_v1": "MEDIUM",
            "risk_v1": "MEDIUM",
            "recommended_v1": True,
            "why_v1": "Directly addresses blocker-over-protector collapse.",
        },
        {
            "experiment_shape_v1": "OBJECTIVE_REWEIGHT_EXPERIMENT",
            "what_changes_v1": "Increase cost of winner/runner damage during model or candidate selection.",
            "what_stays_constant_v1": "R6-style family and feature surface.",
            "change_size_v1": "MEDIUM",
            "risk_v1": "MEDIUM",
            "recommended_v1": True,
            "why_v1": "Aligns optimization with hard safety contract.",
        },
        {
            "experiment_shape_v1": "RUNNER_VS_BAD_SEPARATION_EXPERIMENT",
            "what_changes_v1": "Introduce conflict/separation model for bad-looking runner seeds.",
            "what_stays_constant_v1": "Pre-entry legality and no live/controller promotion.",
            "change_size_v1": "LARGER",
            "risk_v1": "HIGHER",
            "recommended_v1": False,
            "why_v1": "Promising but should follow simpler protector-first test.",
        },
        {
            "experiment_shape_v1": "LABEL_CONTRACT_REVIEW_FIRST",
            "what_changes_v1": "Review runner_protect and runner-vs-bad labels before more model work.",
            "what_stays_constant_v1": "No training until label changes are locked.",
            "change_size_v1": "SMALL_TO_MEDIUM",
            "risk_v1": "LOW",
            "recommended_v1": True,
            "why_v1": "Forensics indicates weak protector translation may be label/objective related.",
        },
    ]
    return {"layer_name_v1": "NEXT_EXPERIMENT_SHAPE_OPTIONS_V1", "options_v1": rows}, pd.DataFrame(rows)


def _go_no_go() -> Dict[str, Any]:
    return {
        "layer_name_v1": "GO_OR_NO_GO_NEXT_STEP_V1",
        "decision_v1": "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT",
        "supporting_locks_v1": [
            "DO_NOT_RETRAIN_SAME_SETUP_AGAIN",
            "REVIEW_OBJECTIVE_AND_LABELS_BEFORE_MODEL_WORK",
            "KEEP_FAILURE_RUN_AS_HARD_NEGATIVE_REFERENCE",
        ],
        "why_v1": "The realistic next step is a protector-first shadow experiment design, not another blocker-first retrain.",
    }


def _write_report(path: Path, summary: Dict[str, Any], design: Dict[str, Any], go_no_go: Dict[str, Any]) -> None:
    lines = [
        "# Monday Protection-First Design Lock V1",
        "",
        "## Decision",
        f"- `{go_no_go['decision_v1']}`",
        "",
        "## Main Failure",
        f"- `{summary['main_failure_v1']}`",
        f"- Precision: `{summary['global_precision_v1']}`",
        f"- Strongest-winner damage: `{summary['strongest_winner_damage_v1']}`",
        "",
        "## Lock",
        f"- {design['design_note_v1']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize protection-first design lock after Monday narrow retrain collapse.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--forensics-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    payload = _load_forensics(reports_root, args.forensics_dir)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    design_lock = _build_design_lock(payload)
    arch_options, arch_table = _architecture_options()
    signal_translation = _signal_translation(payload)
    objective_review = _objective_review(payload)
    eval_contract = _eval_contract()
    no_go = _no_go_same_setup()
    experiment_options, experiment_table = _experiment_options()
    go_no_go = _go_no_go()

    summary = {
        "layer_name_v1": "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "forensics_dir_v1": str(payload["forensics_dir"]),
        "extension_dir_v1": str(extension_dir),
        "main_failure_v1": payload["summary"]["main_failure_v1"],
        "global_precision_v1": payload["summary"]["global_precision_v1"],
        "strongest_winner_damage_v1": payload["summary"]["strongest_winner_damage_v1"],
        "runner_near_miss_blocked_v1": payload["summary"]["runner_near_miss_blocked_v1"],
        "recommended_next_step_v1": go_no_go["decision_v1"],
        "do_not_retrain_same_setup_again_v1": True,
        "protection_first_required_v1": True,
        "hard_status_division_v1": {
            "BEVIST": [
                "Blocker heads overrode runner protection in the narrow retrain failure run.",
                "The same narrow setup must not be retrained again now.",
                "Protection must be part of architecture/objective/decision contract, not just another feature.",
            ],
            "INDIKERT": [
                "Protector-first veto/damper plus objective review is the most realistic next experiment shape.",
                "Some proxies still carry signal, but their translation into protection failed.",
            ],
            "IKKE_ETABLERT": [
                "That any proposed architecture option will beat frozen R6.",
                "That threshold tuning alone can recover safety.",
            ],
        },
    }
    contract = {
        "layer_name_v1": "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_CONTRACT_V1",
        "materialized_at_utc_v1": summary["materialized_at_utc_v1"],
        "forensics_dir_v1": str(payload["forensics_dir"]),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "not_threshold_tuning_v1": True,
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / PROTECTION_FIRST_DESIGN_LOCK, design_lock)
    _write_json(extension_dir / RUNNER_ARCHITECTURE_OPTIONS, arch_options)
    arch_table.to_csv(extension_dir / RUNNER_ARCHITECTURE_OPTIONS_TABLE, index=False)
    _write_json(extension_dir / SIGNAL_TRANSLATION, signal_translation)
    _write_json(extension_dir / OBJECTIVE_LABEL_HEAD_BALANCE, objective_review)
    _write_json(extension_dir / EVAL_CONTRACT, eval_contract)
    _write_json(extension_dir / NO_GO_SAME_SETUP, no_go)
    _write_json(extension_dir / NEXT_EXPERIMENT_OPTIONS, experiment_options)
    experiment_table.to_csv(extension_dir / NEXT_EXPERIMENT_OPTIONS_TABLE, index=False)
    _write_json(extension_dir / GO_NO_GO, go_no_go)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, summary, design_lock, go_no_go)

    artifacts = [
        CONTRACT,
        PROTECTION_FIRST_DESIGN_LOCK,
        RUNNER_ARCHITECTURE_OPTIONS,
        RUNNER_ARCHITECTURE_OPTIONS_TABLE,
        SIGNAL_TRANSLATION,
        OBJECTIVE_LABEL_HEAD_BALANCE,
        EVAL_CONTRACT,
        NO_GO_SAME_SETUP,
        NEXT_EXPERIMENT_OPTIONS,
        NEXT_EXPERIMENT_OPTIONS_TABLE,
        GO_NO_GO,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    audit_rows = [
        _audit_record("FORENSICS_DECISION_READ", "PASS" if payload["go_no_go"]["decision_v1"] == "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN" else "FAIL", {"forensics_decision_v1": payload["go_no_go"]["decision_v1"]}),
        _audit_record("PROTECTION_FIRST_LOCKED", "PASS" if design_lock["explicit_locks_v1"]["next_phase_must_be_protection_first_v1"] else "FAIL", design_lock["explicit_locks_v1"]),
        _audit_record("SAME_SETUP_NO_GO_LOCKED", "PASS" if summary["do_not_retrain_same_setup_again_v1"] else "FAIL", summary),
        _audit_record("NEXT_EXPERIMENT_LOCKED", "PASS" if go_no_go["decision_v1"] == "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT" else "FAIL", go_no_go),
        _audit_record("OUTPUTS_PRESENT", "PASS" if all((extension_dir / a).exists() for a in artifacts if a not in {MANIFEST, STATUS, CONSISTENCY_AUDIT}) else "FAIL", {"artifact_count_v1": len(artifacts)}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    manifest = {
        "layer_name_v1": "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_MANIFEST_V1",
        "materialized_at_utc_v1": summary["materialized_at_utc_v1"],
        "extension_dir_v1": str(extension_dir),
        "artifacts_v1": artifacts,
    }
    _write_json(extension_dir / MANIFEST, manifest)
    status = {
        "layer_name_v1": "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_STATUS_V1",
        "DESIGN_LOCK_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "decision_v1": go_no_go["decision_v1"],
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
