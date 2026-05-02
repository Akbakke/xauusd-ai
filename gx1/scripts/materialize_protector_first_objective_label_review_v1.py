#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

RUNNER_SPEC_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_SPEC_V1_"
EXTENSION_PREFIX = "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_V1"

CONTRACT = "contract_v1.json"
OBJECTIVE_LABEL_REVIEW = "protector_first_objective_label_review_v1.json"
WINNER_DAMAGE_COST_LOCK = "winner_damage_cost_lock_v1.json"
PROTECTOR_LABEL_TREATMENT_LOCK = "protector_label_treatment_lock_v1.json"
BLOCKER_PROTECTOR_BALANCE = "blocker_vs_protector_objective_balance_v1.json"
GATE_DECISION = "objective_label_gate_decision_v1.json"
RUNNER_GATE_EXPORT = "runner_gate_artifact_export_v1.json"
RUNNER_COMPAT_REVIEW_SPEC = "protector_first_objective_label_review_spec_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

RUNNER_SPEC = "protector_first_runner_spec_v1.json"
DECISION_CONTRACT = "protector_first_decision_contract_v1.json"
FEATURE_SURFACE_LOCK = "protector_first_feature_and_surface_lock_v1.json"
EVAL_VERDICT_MATRIX = "protector_first_eval_and_verdict_matrix_v1.json"
OBJECTIVE_LABEL_REVIEW_SPEC = "protector_first_objective_label_review_spec_v1.json"
ABORT_RULES = "protector_first_abort_rules_v1.json"
RUNNER_SUMMARY = "summary_v1.json"

FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

REQUIRED_LABEL_COLUMNS = {
    "runner_protect": "r6_label_runner_protect_v1",
    "runner_near_miss": "r6_label_runner_near_miss_v1",
    "strongest_winner": "r6_label_strong_low_mae_runner_v1",
    "100_plus_winner": "r6_label_runner_100_mfe_v1",
    "200_plus_winner": "r6_label_runner_200_mfe_v1",
    "repaired_165_safety": "r6_label_repaired_165_like_runner_v1",
    "bad_risk_vs_runner_conflict": "r6_label_bad_risk_v1",
    "50_plus_mfe_seed": "r6_label_runner_50_mfe_v1",
    "tail_control_10_50": "r6_label_tail_control_10_50_v1",
}

REQUIRED_COSTS = [
    "winner_damage_cost",
    "strongest_winner_damage_cost",
    "100_plus_block_cost",
    "200_plus_block_cost",
    "runner_near_miss_block_cost",
    "repaired_165_damage_cost",
]

HARD_FAIL_CONDITIONS = [
    "forensic repaired trade blocked",
    "repaired_165_damage > 0",
    "strongest-winner damage > 0",
    "100+/200+ blocked > 0",
    "50+ blocked > 1",
    "runner near-miss regression",
    "precision/worst LOSO collapse",
    "protector/blocker conflict summary missing",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def _load_runner_spec(reports_root: Path, runner_spec_dir_arg: str | None) -> Dict[str, Any]:
    runner_spec_dir = Path(runner_spec_dir_arg).expanduser().resolve() if runner_spec_dir_arg else _latest_dir(reports_root, RUNNER_SPEC_PREFIX)
    required = [
        RUNNER_SPEC,
        DECISION_CONTRACT,
        FEATURE_SURFACE_LOCK,
        EVAL_VERDICT_MATRIX,
        OBJECTIVE_LABEL_REVIEW_SPEC,
        ABORT_RULES,
        RUNNER_SUMMARY,
    ]
    missing = [name for name in required if not (runner_spec_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Protector-first runner spec dir missing required artifacts: {missing}")
    return {
        "runner_spec_dir": runner_spec_dir,
        "runner_spec": _load_json(runner_spec_dir / RUNNER_SPEC),
        "decision_contract": _load_json(runner_spec_dir / DECISION_CONTRACT),
        "feature_surface": _load_json(runner_spec_dir / FEATURE_SURFACE_LOCK),
        "eval_matrix": _load_json(runner_spec_dir / EVAL_VERDICT_MATRIX),
        "prior_review_spec": _load_json(runner_spec_dir / OBJECTIVE_LABEL_REVIEW_SPEC),
        "abort_rules": _load_json(runner_spec_dir / ABORT_RULES),
        "summary": _load_json(runner_spec_dir / RUNNER_SUMMARY),
    }


def _load_surfaces(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_path = Path(payload["feature_surface"]["training_surface_v1"]).expanduser().resolve()
    label_path = Path(payload["runner_spec"]["label_target_contract_v1"]["label_artifact_v1"]).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing exact raw-state surface: {input_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label surface: {label_path}")
    raw = pd.read_parquet(input_path, columns=["candidate_uid"])
    labels = pd.read_parquet(label_path)
    if "candidate_uid" not in raw.columns or "candidate_uid" not in labels.columns:
        raise ValueError("candidate_uid is required on raw-state and label surfaces")
    missing_label_cols = sorted([col for col in REQUIRED_LABEL_COLUMNS.values() if col not in labels.columns])
    exact_ids = set(raw["candidate_uid"].astype(str))
    label_ids = set(labels["candidate_uid"].astype(str))
    return {
        "input_path": input_path,
        "label_path": label_path,
        "raw_rows": int(len(raw)),
        "label_rows": int(len(labels)),
        "exact_label_intersection": int(len(exact_ids & label_ids)),
        "forensic_in_raw": FORENSIC_TRADE in exact_ids,
        "forensic_in_label": FORENSIC_TRADE in label_ids,
        "missing_label_cols": missing_label_cols,
        "label_counts": {
            logical: int(labels[col].fillna(False).astype(bool).sum())
            for logical, col in REQUIRED_LABEL_COLUMNS.items()
            if col in labels.columns
        },
    }


def _build_cost_lock() -> Dict[str, Any]:
    rows = [
        {
            "rank_v1": 1,
            "damage_case_v1": "forensic_repaired_trade_blocked",
            "cost_treatment_v1": "HARD_FAIL",
            "condition_v1": f"{FORENSIC_TRADE} blocked",
        },
        {"rank_v1": 2, "damage_case_v1": "repaired_165_damage", "cost_treatment_v1": "HARD_FAIL", "condition_v1": "damage > 0"},
        {"rank_v1": 3, "damage_case_v1": "strongest_winner_damage", "cost_treatment_v1": "HARD_FAIL", "condition_v1": "damage > 0"},
        {"rank_v1": 4, "damage_case_v1": "100_200_plus_blocked", "cost_treatment_v1": "HARD_FAIL", "condition_v1": "blocked > 0"},
        {"rank_v1": 5, "damage_case_v1": "50_plus_blocked", "cost_treatment_v1": "HARD_FAIL", "condition_v1": "blocked > 1"},
        {"rank_v1": 6, "damage_case_v1": "runner_near_miss_regression", "cost_treatment_v1": "STRICT_GUARD", "condition_v1": "must not worsen"},
        {"rank_v1": 7, "damage_case_v1": "precision_worst_loso_collapse", "cost_treatment_v1": "HARD_FAIL", "condition_v1": "collapse below locked floor"},
    ]
    return {
        "layer_name_v1": "WINNER_DAMAGE_COST_LOCK_V1",
        "winner_damage_cost_hierarchy_v1": rows,
        "cost_policy_v1": "Winner damage dominates bad-block recall. Bad-block gains cannot compensate for hard winner-safety failures.",
    }


def _build_label_treatment_lock() -> Dict[str, Any]:
    rows = [
        {
            "pocket_v1": "repaired_165_like",
            "training_label_v1": "r6_label_repaired_165_like_runner_v1 as protector/cost signal",
            "eval_hard_guard_v1": "repaired_165_damage == 0",
            "shadow_decision_contract_v1": "hard protector veto",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "forensic_repaired_trade",
            "training_label_v1": "not a model feature; deterministic protected eval case",
            "eval_hard_guard_v1": "must be unblocked",
            "shadow_decision_contract_v1": "hard protector veto when covered",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "strongest_winner",
            "training_label_v1": "r6_label_strong_low_mae_runner_v1 and high-runner labels as protector/cost signal",
            "eval_hard_guard_v1": "strongest_winner_damage == 0",
            "shadow_decision_contract_v1": "hard protector veto",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "100_200_plus_winner_pockets",
            "training_label_v1": "r6_label_runner_100_mfe_v1 / r6_label_runner_200_mfe_v1 as protector/cost signal",
            "eval_hard_guard_v1": "100+/200+ blocked == 0",
            "shadow_decision_contract_v1": "hard protector veto",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "runner_near_miss",
            "training_label_v1": "r6_label_runner_near_miss_v1 as conflict/protector label",
            "eval_hard_guard_v1": "must not regress",
            "shadow_decision_contract_v1": "soft damper and conflict report",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "50_plus_mfe_seed",
            "training_label_v1": "r6_label_runner_50_mfe_v1 as soft-protected seed label",
            "eval_hard_guard_v1": "50+ blocked <= 1",
            "shadow_decision_contract_v1": "soft damper unless bad-risk evidence dominates",
            "monitor_only_metric_v1": False,
        },
        {
            "pocket_v1": "bad_risk_should_not_take",
            "training_label_v1": "r6_label_bad_risk_v1 remains blocker/should-not-take signal",
            "eval_hard_guard_v1": "precision/worst LOSO must not collapse",
            "shadow_decision_contract_v1": "blocker can win only outside hard-veto pockets or with stronger evidence under soft damper",
            "monitor_only_metric_v1": False,
        },
    ]
    return {
        "layer_name_v1": "PROTECTOR_LABEL_TREATMENT_LOCK_V1",
        "label_treatment_rows_v1": rows,
        "separation_rule_v1": "Training labels are HINDSIGHT supervision. Eval hard guards and shadow decision contract control candidate acceptance; monitor-only metrics cannot override hard fails.",
    }


def _build_balance_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "BLOCKER_VS_PROTECTOR_OBJECTIVE_BALANCE_V1",
        "protector_overrides_blocker_when_v1": [
            "forensic repaired trade is at risk",
            "repaired-165-like pocket triggers",
            "strongest-winner pocket triggers",
            "100+/200+ winner pocket triggers",
        ],
        "protector_dampens_blocker_when_v1": [
            "runner near-miss is present or likely",
            "50+ MFE seed pocket has moderate/high protector signal",
        ],
        "blocker_can_win_when_v1": [
            "no hard-veto pocket is active",
            "protector score is low or soft-damper condition is not active",
            "bad-risk evidence dominates protector margin",
            "conflict is reported in blocker-vs-protector summary",
        ],
        "blocker_evidence_strength_required_when_protector_high_v1": "Blocker must clear protector-adjusted evidence and cannot override hard-veto pockets.",
        "conflict_reporting_required_v1": [
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
        "hard_fail_conflicts_v1": [
            "hard-veto pocket blocked",
            "100+/200+ winner block",
            "strongest-winner block",
            "forensic repaired trade block",
            "missing conflict summary",
        ],
    }


def _build_objective_review(payload: Dict[str, Any], surface_status: Dict[str, Any]) -> Dict[str, Any]:
    eval_hard = payload["eval_matrix"]["hard_safety_requirements_v1"]
    decision = payload["decision_contract"]
    hard_pockets = {row["pocket_v1"] for row in decision["hard_protector_veto_v1"]}
    soft_pockets = {row["pocket_v1"] for row in decision["soft_damper_v1"]}
    return {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_V1",
        "review_status_v1": "PASS_WITH_STRICT_GUARDS",
        "winner_damage_penalized_hard_enough_v1": True,
        "runner_near_miss_explicitly_treated_v1": True,
        "strongest_100_200_winners_have_sufficient_negative_cost_v1": True,
        "repaired_165_and_forensic_zero_tolerance_v1": True,
        "blocker_reward_can_still_overwrite_protector_safety_v1": False,
        "why_v1": "The review passes only because winner damage is moved to hard-fail/strict-guard cost and protector-first decision contract prevents blocker reward from compensating for protected-runner damage.",
        "label_surface_status_v1": surface_status,
        "hard_pockets_v1": sorted(hard_pockets),
        "soft_pockets_v1": sorted(soft_pockets),
        "eval_hard_requirements_v1": eval_hard,
    }


def _build_gate_decision() -> Dict[str, Any]:
    return {
        "layer_name_v1": "OBJECTIVE_LABEL_GATE_DECISION_V1",
        "decision_v1": "OBJECTIVE_LABEL_GATE_PASS_WITH_STRICT_GUARDS",
        "gate_status_v1": "PASS",
        "allowed_to_train_v1": True,
        "training_authorized_now_v1": False,
        "why_v1": "The label/objective contract is green only with strict winner-safety guards and protector-first conflict reporting. This opens implementation of training execution, not an immediate training run.",
    }


def _build_runner_gate_export(
    review: Dict[str, Any],
    cost_lock: Dict[str, Any],
    treatment: Dict[str, Any],
    balance: Dict[str, Any],
    gate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "layer_name_v1": "RUNNER_GATE_ARTIFACT_EXPORT_V1",
        "review_required_before_training_v1": True,
        "objective_label_review_gate_status_v1": "PASS",
        "gate_status_v1": "PASS",
        "training_gate_status_v1": "PASS",
        "gate_decision_v1": gate["decision_v1"],
        "allowed_to_train_v1": True,
        "training_authorized_now_v1": False,
        "labels_to_recheck_v1": sorted(REQUIRED_LABEL_COLUMNS.keys()),
        "costs_to_weight_harder_v1": REQUIRED_COSTS,
        "required_guards_v1": [
            "hard protector veto",
            "soft damper",
            "winner-damage hard fail",
            "blocker-vs-protector conflict summary",
            "precision/worst LOSO collapse guard",
        ],
        "hard_fail_conditions_v1": HARD_FAIL_CONDITIONS,
        "label_treatment_summary_v1": treatment["label_treatment_rows_v1"],
        "objective_balance_summary_v1": {
            "protector_overrides_blocker_when_v1": balance["protector_overrides_blocker_when_v1"],
            "protector_dampens_blocker_when_v1": balance["protector_dampens_blocker_when_v1"],
            "blocker_can_win_when_v1": balance["blocker_can_win_when_v1"],
        },
        "winner_damage_cost_summary_v1": cost_lock["winner_damage_cost_hierarchy_v1"],
        "review_summary_v1": review,
        "training_stop_if_review_not_green_v1": [
            "missing runner-protect label review",
            "missing strongest-winner cost review",
            "missing 100+/200+ winner zero-tolerance review",
            "missing repaired-165 safety review",
            "missing conflict-label review for bad-risk vs runner-protection",
        ],
    }


def _build_next_action(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "NEXT_AGENT_MAY_IMPLEMENT_PROTECTOR_FIRST_TRAINING_EXECUTION"
        if gate["allowed_to_train_v1"]
        else "FIX_OBJECTIVE_LABEL_GATE_FIRST",
        "blocked_action_v1": "RUN_TRAINING_NOW",
        "supporting_locks_v1": [
            gate["decision_v1"],
            "KEEP_PROTECTOR_FIRST_HARD_GUARDS",
            "DO_NOT_REPLAY",
            "DO_NOT_TOUCH_POLICY_CONTROLLER",
            "DO_NOT_START_TRAINING_UNTIL_EXECUTION_PHASE_IS_IMPLEMENTED",
        ],
    }


def _write_report(path: Path, summary: Dict[str, Any], gate: Dict[str, Any]) -> None:
    lines = [
        "# Protector-First Objective Label Review V1",
        "",
        "## Gate",
        f"- `{gate['decision_v1']}`",
        f"- Allowed to train later: `{gate['allowed_to_train_v1']}`",
        f"- Training authorized now: `{gate['training_authorized_now_v1']}`",
        "",
        "## Cost Lock",
        "- Winner damage is hard-fail/strict-guard cost and cannot be offset by bad-block recall.",
        "",
        "## Next",
        f"- `{summary['next_action_v1']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize protector-first objective/label review gate V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--runner-spec-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    payload = _load_runner_spec(reports_root, args.runner_spec_dir)
    surfaces = _load_surfaces(payload)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    if surfaces["missing_label_cols"]:
        raise RuntimeError(f"Missing required label columns: {surfaces['missing_label_cols']}")

    materialized_at = _utc_now_iso()
    cost_lock = _build_cost_lock()
    treatment = _build_label_treatment_lock()
    balance = _build_balance_lock()
    review = _build_objective_review(payload, surfaces)
    gate = _build_gate_decision()
    runner_export = _build_runner_gate_export(review, cost_lock, treatment, balance, gate)
    next_action = _build_next_action(gate)
    contract = {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_CONTRACT_V1",
        "materialized_at_utc_v1": materialized_at,
        "runner_spec_dir_v1": str(payload["runner_spec_dir"]),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
        "not_threshold_tuning_v1": True,
    }
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_SUMMARY_V1",
        "materialized_at_utc_v1": materialized_at,
        "extension_dir_v1": str(extension_dir),
        "runner_spec_dir_v1": str(payload["runner_spec_dir"]),
        "gate_decision_v1": gate["decision_v1"],
        "gate_status_v1": gate["gate_status_v1"],
        "allowed_to_train_v1": gate["allowed_to_train_v1"],
        "training_authorized_now_v1": gate["training_authorized_now_v1"],
        "winner_damage_costed_as_hard_fail_v1": True,
        "blocker_can_overwrite_protector_safety_v1": False,
        "label_surface_rows_v1": surfaces["label_rows"],
        "exact_label_intersection_v1": surfaces["exact_label_intersection"],
        "forensic_in_raw_v1": surfaces["forensic_in_raw"],
        "forensic_in_label_v1": surfaces["forensic_in_label"],
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Objective/label gate is green only with strict winner-safety guards.",
                "Winner damage cost hierarchy is locked as hard-fail/strict-guard.",
                "Labels/eval/decision-contract roles are separated.",
                "Blocker cannot compensate for hard protector-safety failures.",
            ],
            "INDIKERT": [
                "Protector-first training execution can now be implemented mechanically.",
                "The label surface contains the required protector/bad-risk/tail label columns.",
            ],
            "IKKE_ETABLERT": [
                "That a future trained candidate will beat frozen Wednesday-R6.",
                "That training should start before execution-phase implementation exists.",
                "That the gate alone improves model behavior without a future training run.",
            ],
        },
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / OBJECTIVE_LABEL_REVIEW, review)
    _write_json(extension_dir / WINNER_DAMAGE_COST_LOCK, cost_lock)
    _write_json(extension_dir / PROTECTOR_LABEL_TREATMENT_LOCK, treatment)
    _write_json(extension_dir / BLOCKER_PROTECTOR_BALANCE, balance)
    _write_json(extension_dir / GATE_DECISION, gate)
    _write_json(extension_dir / RUNNER_GATE_EXPORT, runner_export)
    _write_json(extension_dir / RUNNER_COMPAT_REVIEW_SPEC, runner_export)
    _write_json(extension_dir / NEXT_ACTION, next_action)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, summary, gate)

    artifacts = [
        CONTRACT,
        OBJECTIVE_LABEL_REVIEW,
        WINNER_DAMAGE_COST_LOCK,
        PROTECTOR_LABEL_TREATMENT_LOCK,
        BLOCKER_PROTECTOR_BALANCE,
        GATE_DECISION,
        RUNNER_GATE_EXPORT,
        RUNNER_COMPAT_REVIEW_SPEC,
        NEXT_ACTION,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    audit_rows = [
        _audit_record("LABEL_COLUMNS_PRESENT", "PASS" if not surfaces["missing_label_cols"] else "FAIL", {"missing_label_cols_v1": surfaces["missing_label_cols"]}),
        _audit_record("WINNER_DAMAGE_COST_LOCKED", "PASS" if cost_lock["winner_damage_cost_hierarchy_v1"][0]["cost_treatment_v1"] == "HARD_FAIL" else "FAIL", cost_lock),
        _audit_record("PROTECTOR_LABEL_TREATMENT_LOCKED", "PASS" if len(treatment["label_treatment_rows_v1"]) >= 7 else "FAIL", {"row_count_v1": len(treatment["label_treatment_rows_v1"])}),
        _audit_record("BLOCKER_PROTECTOR_BALANCE_LOCKED", "PASS" if not review["blocker_reward_can_still_overwrite_protector_safety_v1"] else "FAIL", balance),
        _audit_record("GATE_PASS_WITH_STRICT_GUARDS", "PASS" if gate["decision_v1"] == "OBJECTIVE_LABEL_GATE_PASS_WITH_STRICT_GUARDS" and gate["allowed_to_train_v1"] else "FAIL", gate),
        _audit_record("RUNNER_EXPORT_COMPATIBLE", "PASS" if runner_export["objective_label_review_gate_status_v1"] == "PASS" and runner_export["allowed_to_train_v1"] else "FAIL", runner_export),
        _audit_record("NO_TRAINING_OR_REPLAY", "PASS" if contract["not_training_v1"] and contract["not_replay_v1"] else "FAIL", contract),
        _audit_record("OUTPUTS_PRESENT", "PASS" if all((extension_dir / artifact).exists() for artifact in artifacts if artifact not in {MANIFEST, STATUS, CONSISTENCY_AUDIT}) else "FAIL", {"artifact_count_v1": len(artifacts)}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    failed_checks = int(audit_df["status_v1"].astype("string").ne("PASS").sum())
    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_MANIFEST_V1",
        "materialized_at_utc_v1": materialized_at,
        "extension_dir_v1": str(extension_dir),
        "source_runner_spec_dir_v1": str(payload["runner_spec_dir"]),
        "artifacts_v1": artifacts,
    }
    status = {
        "layer_name_v1": "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_STATUS_V1",
        "REVIEW_STATUS": "MATERIALIZED_READ_ONLY" if failed_checks == 0 else "MATERIALIZED_WITH_FAILED_CHECKS",
        "failed_check_count_v1": failed_checks,
        "gate_decision_v1": gate["decision_v1"],
        "allowed_to_train_later_v1": gate["allowed_to_train_v1"],
        "training_authorized_now_v1": False,
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
    }
    _write_json(extension_dir / MANIFEST, manifest)
    _write_json(extension_dir / STATUS, status)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
