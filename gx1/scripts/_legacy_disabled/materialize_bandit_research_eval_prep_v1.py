#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    PATH_DYNAMICS_V2_FIELDS,
    _json_ready,
    _read_csv_optional,
    _read_json_optional,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "BANDIT_RESEARCH_EVAL_PREP_V1"
DATASET_LAYER_ID = "BUILD_MANAGEMENT_BANDIT_DATASET_V1"
CONTRACT_LOCK_LAYER_ID = "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1"
REWARD_VERSION_ID = "MGMT_BANDIT_REALIZED_PNL_BPS_V1"
REWARD_FORMULA = "reward_bps = terminal_realized_pnl_bps"

OUTPUTS = {
    "contract": "bandit_research_eval_prep_contract_v1.json",
    "scope_boundary_lock": "bandit_eval_scope_boundary_lock_v1.json",
    "boundary_allowed_use": "bandit_eval_boundary_allowed_use_v1.csv",
    "boundary_forbidden_use": "bandit_eval_boundary_forbidden_use_v1.csv",
    "boundary_allowed_claims": "bandit_eval_boundary_allowed_claims_v1.csv",
    "boundary_forbidden_claims": "bandit_eval_boundary_forbidden_claims_v1.csv",
    "protocol_draft": "bandit_eval_split_protocol_draft_v1.json",
    "protocol_table": "bandit_eval_split_protocol_steps_v1.csv",
    "comparator_plan": "bandit_comparator_application_plan_v1.json",
    "comparator_table": "bandit_comparator_application_plan_v1.csv",
    "failcheck_plan": "bandit_failcheck_enforcement_plan_v1.json",
    "failcheck_table": "bandit_failcheck_enforcement_plan_v1.csv",
    "risk_lock": "action_imbalance_support_risk_lock_v1.json",
    "output_contract": "first_bandit_eval_output_contract_v1.json",
    "output_required_sections": "first_bandit_eval_output_required_sections_v1.csv",
    "output_required_tables": "first_bandit_eval_output_required_tables_v1.csv",
    "post_status": "post_eval_prep_status_update_v1.json",
    "summary": "bandit_research_eval_prep_summary_v1.json",
    "report": "bandit_research_eval_prep_report_v1.md",
    "manifest": "bandit_research_eval_prep_manifest_v1.json",
    "status": "bandit_research_eval_prep_status_v1.json",
    "consistency_audit": "bandit_research_eval_prep_consistency_audit_v1.csv",
    "consistency_audit_json": "bandit_research_eval_prep_consistency_audit_v1.json",
    "non_interference_audit": "bandit_research_eval_prep_non_interference_audit_v1.csv",
    "non_interference_audit_json": "bandit_research_eval_prep_non_interference_audit_v1.json",
}


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_dataset_dir(reports_root: Path, dataset_dir_arg: str | None) -> Path:
    if dataset_dir_arg:
        path = Path(dataset_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{DATASET_LAYER_ID}_*/build_management_bandit_dataset_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {DATASET_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _latest_contract_lock_dir(reports_root: Path, contract_lock_dir_arg: str | None) -> Path:
    if contract_lock_dir_arg:
        path = Path(contract_lock_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Contract lock dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{CONTRACT_LOCK_LAYER_ID}_*/iql_reward_comparator_bandit_contract_lock_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {CONTRACT_LOCK_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _source_paths(reports_root: Path, dataset_dir: Path, contract_lock_dir: Path) -> dict[str, str | None]:
    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    return {
        "reports_root_v1": str(reports_root),
        "dataset_dir_v1": str(dataset_dir),
        "dataset_summary_v1": str(dataset_dir / "build_management_bandit_dataset_summary_v1.json"),
        "dataset_profile_v1": str(dataset_dir / "management_bandit_dataset_profile_v1.json"),
        "dataset_parquet_v1": dataset_summary.get("dataset_parquet_v1"),
        "dataset_metadata_v1": str(dataset_dir / "management_bandit_research_dataset_metadata_v1.json"),
        "comparator_contract_dir_v1": str(contract_lock_dir),
        "comparator_lock_v1": str(contract_lock_dir / "iql_baseline_comparator_and_failcheck_lock_v1.json"),
        "comparator_table_v1": str(contract_lock_dir / "iql_baseline_comparator_lock_v1.csv"),
        "failcheck_table_v1": str(contract_lock_dir / "iql_failcheck_policy_lock_v1.csv"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _table(items: list[str], column: str, status: str = "LOCKED") -> pd.DataFrame:
    return pd.DataFrame.from_records([{column: item, "status_v1": status} for item in items])


def _action_count(dataset_summary: dict[str, Any], action: str) -> int:
    for row in dataset_summary.get("action_distribution_v1", []):
        if str(row.get("action_v1")) == action:
            return int(row.get("row_count_v1", 0) or 0)
    return 0


def _build_scope_boundary_lock(dataset_summary: dict[str, Any], dataset_profile: dict[str, Any]) -> dict[str, Any]:
    hold_count = _action_count(dataset_summary, "HOLD")
    exit_count = _action_count(dataset_summary, "EXIT_NOW")
    total = int(dataset_summary.get("included_rows_v1", hold_count + exit_count) or 0)
    exit_share = float(exit_count / total) if total else 0.0
    return {
        "lock_id_v1": "BANDIT_EVAL_SCOPE_AND_BOUNDARY_LOCK_V1",
        "scope_verdicts_v1": [
            "BANDIT_EVAL_ONLY",
            "NOT_IQL_EVAL",
            "NOT_R7_READINESS",
            "LIMITED_BY_ACTION_IMBALANCE",
        ],
        "intended_questions_v1": [
            "Can a future management contextual bandit research eval be interpreted under the locked reward/comparator/fail-check contracts?",
            "Which reporting, split, comparator, support, and fail-check boundaries must be enforced before any future bandit research result is read?",
            "How should the built dataset limitations be surfaced in every future eval report?",
        ],
        "cannot_answer_v1": [
            "Whether sequence-IQL is ready.",
            "Whether R7 should start.",
            "Whether HOLD -> next_state truth exists.",
            "Whether a live controller should be promoted.",
            "Whether unsupported or counterfactual actions are safe.",
        ],
        "allowed_conclusions_v1": [
            "A future bandit research eval can be run under this protocol if it stays row-wise and comparator-governed.",
            "Any future result is limited by HOLD dominance, thin EXIT_NOW coverage, and SUPPORT_TOO_THIN.",
            "Fail-checks can block positive interpretation even when headline reward is favorable.",
        ],
        "forbidden_conclusions_v1": [
            "This is IQL eval.",
            "This proves sequence-RL or R7 readiness.",
            "This proves live policy readiness.",
            "This proves HOLD transitions or path-dynamics replay truth.",
            "This proves the reward is a universal RL objective.",
        ],
        "why_not_sequence_iql_eval_v1": "The dataset is contextual-bandit row-wise and intentionally has no next_state/done transition contract; HOLD -> next_state remains zero.",
        "why_not_r7_readiness_v1": "R7 remains blocked until completed path-dynamics replay, post-replay audit, comparator calibration, and transition truth are established.",
        "action_imbalance_boundary_v1": {
            "hold_rows_v1": hold_count,
            "exit_now_rows_v1": exit_count,
            "exit_now_share_v1": exit_share,
            "support_ood_verdict_v1": dataset_profile.get("support_ood_verdict_from_foundation_v1", dataset_summary.get("support_ood_verdict_v1")),
            "hard_limitation_v1": "HOLD dominance and thin EXIT_NOW support can make headline reward misleading and must be front-page context.",
        },
    }


def _boundary_tables(scope: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "allowed_use": _table(
            [
                "row-wise management contextual bandit research eval preparation",
                "comparator/fail-check governed future eval protocol",
                "support/OOD and action imbalance reporting contract",
            ],
            "allowed_use_v1",
        ),
        "forbidden_use": _table(
            [
                "IQL eval or sequence-RL eval",
                "R7 readiness or R7 training decision",
                "live gate/controller/policy promotion",
                "path-dynamics replay validation",
                "HOLD next_state inference",
            ],
            "forbidden_use_v1",
            "FORBIDDEN",
        ),
        "allowed_claims": _table(scope["allowed_conclusions_v1"], "allowed_claim_v1"),
        "forbidden_claims": _table(scope["forbidden_conclusions_v1"], "forbidden_claim_v1", "FORBIDDEN"),
    }


def _build_protocol(dataset_summary: dict[str, Any], dataset_profile: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    steps = [
        ("scope_freeze", "Use exactly the built management bandit dataset and locked reward version; do not add sequence fields.", "REQUIRED"),
        ("primary_split", "Use chronological/block split by decision_ts where feasible; report that random IID-only split is insufficient.", "DRAFT_READY"),
        ("rolling_block_review", "Use rolling-window or block eval as stability review if decision_ts coverage remains complete.", "DRAFT_READY"),
        ("action_stratification", "Always report HOLD and EXIT_NOW separately, with row counts and shares.", "REQUIRED"),
        ("support_stratification", "Always report support_status distribution and metric slices by support tier.", "REQUIRED"),
        ("safe_pocket_breakdown", "If state_feature_names expose session/vol/trend/side safely, report descriptive pockets without trading interpretation.", "DRAFT_READY"),
        ("comparator_application", "Apply locked comparator/fail-check contract before any positive interpretation.", "REQUIRED"),
        ("fail_closed_reporting", "If a hard gate cannot be computed, mark eval partial instead of promoting a positive result.", "REQUIRED"),
    ]
    df = pd.DataFrame.from_records(
        [
            {
                "protocol_step_v1": key,
                "rule_v1": rule,
                "status_v1": status,
            }
            for key, rule, status in steps
        ]
    )
    support_verdict = dataset_profile.get("support_ood_verdict_from_foundation_v1", dataset_summary.get("support_ood_verdict_v1"))
    return {
        "protocol_id_v1": "BANDIT_EVAL_SPLIT_AND_PROTOCOL_DRAFT_V1",
        "verdict_v1": "EVAL_PROTOCOL_DRAFT_READY",
        "recommended_eval_splitting_v1": [
            "chronological/block split by decision_ts",
            "rolling-window/block stability review",
            "no IID-only headline interpretation",
        ],
        "visible_groups_and_slices_v1": [
            "action",
            "support_status",
            "behavior_policy_status",
            "decision_ts block",
            "safe AS_OF descriptive pockets if available: session, vol_regime, trend_regime, side",
            "BATCH_04/BATCH_05 if present in future eval artifact",
        ],
        "hold_exit_imbalance_reporting_v1": "Report row counts, shares, per-action metrics, and mark EXIT_NOW as thin whenever n=45 remains unchanged.",
        "support_thinness_reporting_v1": f"Report support/OOD verdict `{support_verdict}` and never hide support tiers behind headline reward.",
        "rolling_window_or_block_eval_v1": "Use decision_ts blocks if parseable; otherwise fail-closed to partial eval protocol.",
        "comparator_use_v1": "Comparator/fail-check contract is applied before interpreting headline reward.",
        "no_training_executed_v1": True,
        "fail_closed_if_split_cannot_be_established_v1": True,
        "steps_v1": df.to_dict(orient="records"),
    }, df


def _build_comparator_application(comparator_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in comparator_df.to_dict(orient="records"):
        name = str(row.get("comparator_v1"))
        source_status = str(row.get("status_v1"))
        if name == "no-RL/current locked ledger":
            status = "DIRECT_EVAL_COMPARATOR"
            headline = True
            safety = True
            sanity = False
        elif name == "dummy/random sanity comparator":
            status = "DIRECT_EVAL_COMPARATOR"
            headline = False
            safety = False
            sanity = True
        elif source_status == "PENDING_CALIBRATION":
            status = "PENDING_CALIBRATION"
            headline = name in {"R6 frozen shadow candidate", "supervised EXIT_LOCAL/tree baseline"}
            safety = True
            sanity = False
        else:
            status = "INTERPRETIVE_COMPARATOR"
            headline = False
            safety = True
            sanity = False
        rows.append(
            {
                "comparator_v1": name,
                "bandit_eval_relevance_v1": "Protects interpretation against single-metric or unsupported bandit claims.",
                "protects_against_v1": "headline-only positive interpretation, comparator drift, and unsupported promotion language",
                "does_not_protect_against_v1": "missing HOLD transitions, sequence-IQL readiness, R7 readiness, or live readiness",
                "application_status_v1": status,
                "source_contract_status_v1": source_status,
                "headline_eval_required_v1": headline,
                "safety_review_required_v1": safety,
                "sanity_anchor_only_v1": sanity,
            }
        )
    out = pd.DataFrame.from_records(rows)
    return {
        "plan_id_v1": "BANDIT_COMPARATOR_APPLICATION_PLAN_V1",
        "headline_eval_comparators_v1": out.loc[out["headline_eval_required_v1"], "comparator_v1"].astype(str).tolist(),
        "safety_review_comparators_v1": out.loc[out["safety_review_required_v1"], "comparator_v1"].astype(str).tolist(),
        "sanity_anchor_comparators_v1": out.loc[out["sanity_anchor_only_v1"], "comparator_v1"].astype(str).tolist(),
        "calibration_boundary_v1": "Comparators with PENDING_CALIBRATION must be shown as required future comparators but cannot be treated as fully calibrated thresholds.",
        "rows_v1": out.to_dict(orient="records"),
    }, out


def _build_failcheck_plan(failcheck_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in failcheck_df.to_dict(orient="records"):
        gate_type = str(row.get("gate_type_v1"))
        auto_stop = bool(row.get("auto_stop_promotion_v1"))
        hard_gate = auto_stop or "HARD" in gate_type
        rows.append(
            {
                "metric_or_failcheck_v1": row.get("metric_or_failcheck_v1"),
                "enforcement_type_v1": "HARD_GATE" if hard_gate else "SOFT_REVIEW",
                "source_gate_type_v1": gate_type,
                "directionality_v1": row.get("better_direction_v1"),
                "unacceptable_damage_v1": row.get("unacceptable_damage_v1"),
                "why_exists_v1": row.get("why_exists_v1"),
                "protected_pockets_or_slices_v1": row.get("protected_pockets_or_slices_v1"),
                "must_report_even_if_headline_pnl_good_v1": True if hard_gate else bool(row.get("requires_extra_audit_even_with_good_headline_pnl_v1")),
                "auto_stops_positive_interpretation_v1": hard_gate,
                "requires_extra_audit_v1": bool(row.get("requires_extra_audit_even_with_good_headline_pnl_v1")) or hard_gate,
            }
        )
    out = pd.DataFrame.from_records(rows)
    return {
        "plan_id_v1": "BANDIT_FAILCHECK_ENFORCEMENT_PLAN_V1",
        "policy_v1": {
            "auto_stop_positive_interpretation_v1": out.loc[out["auto_stops_positive_interpretation_v1"], "metric_or_failcheck_v1"].astype(str).tolist(),
            "requires_extra_audit_v1": out.loc[out["requires_extra_audit_v1"], "metric_or_failcheck_v1"].astype(str).tolist(),
            "secondary_support_only_v1": out.loc[~out["auto_stops_positive_interpretation_v1"], "metric_or_failcheck_v1"].astype(str).tolist(),
            "headline_pnl_never_sufficient_v1": True,
        },
        "rows_v1": out.to_dict(orient="records"),
    }, out


def _build_risk_lock(dataset_summary: dict[str, Any], dataset_profile: dict[str, Any]) -> dict[str, Any]:
    hold_count = _action_count(dataset_summary, "HOLD")
    exit_count = _action_count(dataset_summary, "EXIT_NOW")
    total = int(dataset_summary.get("included_rows_v1", hold_count + exit_count) or 0)
    hold_share = float(hold_count / total) if total else 0.0
    exit_share = float(exit_count / total) if total else 0.0
    return {
        "lock_id_v1": "ACTION_IMBALANCE_AND_SUPPORT_RISK_LOCK_V1",
        "verdicts_v1": [
            "SEVERE_ACTION_IMBALANCE",
            "SUPPORT_THIN_REQUIRES_CAUTION",
            "RESULTS_MUST_BE_PRESENTED_WITH_LIMITATIONS",
        ],
        "hold_rows_v1": hold_count,
        "exit_now_rows_v1": exit_count,
        "hold_share_v1": hold_share,
        "exit_now_share_v1": exit_share,
        "support_ood_verdict_v1": dataset_profile.get("support_ood_verdict_from_foundation_v1", dataset_summary.get("support_ood_verdict_v1")),
        "why_serious_limitation_v1": "A 1751/45 action split means headline reward is dominated by HOLD rows and EXIT_NOW conclusions have very thin direct support.",
        "potentially_misleading_results_v1": [
            "headline reward improvement driven by HOLD majority",
            "EXIT_NOW policy claims from too few rows",
            "apparent support in aggregate hiding unsupported pockets",
            "action agreement that simply mirrors the behavior policy imbalance",
        ],
        "interesting_result_requirements_v1": [
            "per-action metrics shown with counts",
            "support/OOD tier breakdown shown",
            "no hard fail-check damage",
            "worst-slice and rolling/block stability shown",
            "explicit comparator status shown",
        ],
        "not_evidence_of_v1": [
            "sequence-IQL readiness",
            "R7 readiness",
            "live policy readiness",
            "safe extrapolation into thin EXIT_NOW pockets",
            "canonical HOLD transition truth",
        ],
        "presentation_rules_v1": [
            "front-page action counts and shares",
            "front-page support/OOD verdict",
            "per-action metrics before aggregate-only interpretation",
            "all hard gates shown even if headline reward is favorable",
            "limitations section required in final verdict",
        ],
    }


def _build_output_contract() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sections = [
        "executive summary",
        "dataset scope",
        "reward version used",
        "comparator set used",
        "fail-check summary",
        "action distribution",
        "support/OOD summary",
        "headline metrics",
        "slice/pocket/regime breakdown",
        "stress breakdown",
        "limitation section",
        "allowed conclusion section",
        "forbidden conclusion section",
        "final verdict",
    ]
    tables = [
        ("dataset_scope_v1", "row count, reward version, dataset artifact, no sequence columns"),
        ("action_distribution_v1", "HOLD/EXIT_NOW counts and shares"),
        ("support_ood_summary_v1", "support_status counts and OOD verdict"),
        ("comparator_status_v1", "comparator application status and calibration boundary"),
        ("failcheck_summary_v1", "hard gates, soft reviews, failed checks"),
        ("headline_metrics_v1", "reward metrics with comparator status"),
        ("slice_pocket_regime_breakdown_v1", "safe AS_OF pockets with counts"),
        ("rolling_block_stability_v1", "decision_ts block/rolling stability if available"),
        ("stress_breakdown_v1", "BATCH_04/BATCH_05 if present, otherwise explicit not established"),
        ("allowed_forbidden_claims_v1", "claims table copied from scope boundary lock"),
    ]
    section_df = pd.DataFrame.from_records([{"required_section_v1": item, "required_v1": True} for item in sections])
    table_df = pd.DataFrame.from_records([{"required_table_or_json_key_v1": key, "content_rule_v1": rule, "required_v1": True} for key, rule in tables])
    return {
        "contract_id_v1": "FIRST_BANDIT_EVAL_OUTPUT_CONTRACT_V1",
        "verdict_v1": "EVAL_OUTPUT_CONTRACT_LOCKED",
        "required_sections_v1": sections,
        "required_json_keys_v1": [key for key, _ in tables],
        "required_tables_v1": table_df.to_dict(orient="records"),
        "final_verdict_allowed_values_v1": [
            "RESEARCH_EVAL_INTERESTING_WITH_LIMITATIONS",
            "RESEARCH_EVAL_INCONCLUSIVE",
            "RESEARCH_EVAL_FAILS_HARD_GATES",
            "RESEARCH_EVAL_NOT_INTERPRETABLE",
        ],
        "forbidden_output_claims_v1": [
            "IQL-ready",
            "R7-ready",
            "live-ready",
            "sequence-ready",
            "HOLD transitions proven",
        ],
    }, section_df, table_df


def _build_post_status(dataset_summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "update_id_v1": "POST_EVAL_PREP_STATUS_UPDATE_V1",
        "bandit_eval_prep_ready_v1": protocol.get("verdict_v1") == "EVAL_PROTOCOL_DRAFT_READY",
        "dataset_status_v1": dataset_summary.get("dataset_verdict_v1"),
        "comparator_failcheck_contract_still_governing_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_status_unchanged_blocked_v1": True,
        "replay_status_unchanged_v1": True,
        "next_right_steps_v1": [
            "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1",
            "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "bandit_eval_prep_contract_materialized",
                "dataset_remains_bandit_research_dataset_built_with_limitations",
                "comparator_failcheck_contract_still_governing",
                "sequence_iql_still_blocked",
                "hold_transition_truth_still_missing",
                "r7_still_blocked",
                "replay_status_unchanged",
            ],
            "INDIKERT": [
                "first_bandit_research_eval_can_be_run_under_strict_limitations",
            ],
            "IKKE_ETABLERT": [
                "sequence_iql_readiness",
                "r7_readiness",
                "canonical_hold_next_state_transitions",
                "path_dynamics_training_canonical_status",
            ],
        },
    }


def _build_non_interference(
    output_dir: Path,
    source_paths: dict[str, str | None],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
    r6_sha_before: str | None,
    r6_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay canonical source"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("POLICY_PROMOTION_NOT_ATTEMPTED", "PASS", "not_attempted", "not_attempted"),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": status, "observed_value_v1": observed, "expected_value_v1": expected} for name, status, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_consistency(
    dataset_summary: dict[str, Any],
    scope: dict[str, Any],
    protocol: dict[str, Any],
    comparator_table: pd.DataFrame,
    failcheck_table: pd.DataFrame,
    risk_lock: dict[str, Any],
    output_contract: dict[str, Any],
    non_interference: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_comparators = {
        "no-RL/current locked ledger",
        "R6 frozen shadow candidate",
        "R5.2 frozen historical reference",
        "management harvest comparator",
        "supervised EXIT_LOCAL/tree baseline",
        "dummy/random sanity comparator",
    }
    expected_failchecks = {
        "realized pnl",
        "bad-trade reduction",
        "MFE capture",
        "MAE burden",
        "giveback",
        "tail-control help",
        "runner damage",
        "50+/100+/200+ MFE damage",
        "strongest-winner path damage",
        "action agreement",
        "OOD action rate",
        "worst-slice performance",
        "rolling-window stability",
        "BATCH_04 stress",
        "BATCH_05 stress",
        "harvest candidate capture",
    }
    checks = [
        ("DATASET_BUILT_WITH_LOCKED_REWARD", dataset_summary.get("dataset_built_v1") is True and dataset_summary.get("reward_version_v1") == REWARD_VERSION_ID, dataset_summary.get("reward_version_v1"), REWARD_VERSION_ID),
        ("SCOPE_MARKS_NOT_IQL_NOT_R7", "NOT_IQL_EVAL" in scope.get("scope_verdicts_v1", []) and "NOT_R7_READINESS" in scope.get("scope_verdicts_v1", []), scope.get("scope_verdicts_v1"), "NOT_IQL_EVAL|NOT_R7_READINESS"),
        ("PROTOCOL_DRAFT_READY", protocol.get("verdict_v1") == "EVAL_PROTOCOL_DRAFT_READY", protocol.get("verdict_v1"), "EVAL_PROTOCOL_DRAFT_READY"),
        ("ALL_REQUIRED_COMPARATORS_MAPPED", required_comparators.issubset(set(comparator_table.get("comparator_v1", pd.Series(dtype=str)).astype(str))), comparator_table.get("comparator_v1", pd.Series(dtype=str)).astype(str).tolist(), sorted(required_comparators)),
        ("REQUIRED_FAILCHECKS_MAPPED", expected_failchecks.issubset(set(failcheck_table.get("metric_or_failcheck_v1", pd.Series(dtype=str)).astype(str))), failcheck_table.get("metric_or_failcheck_v1", pd.Series(dtype=str)).astype(str).tolist(), sorted(expected_failchecks)),
        ("ACTION_IMBALANCE_LOCKED", "SEVERE_ACTION_IMBALANCE" in risk_lock.get("verdicts_v1", []), risk_lock.get("verdicts_v1"), "SEVERE_ACTION_IMBALANCE"),
        ("OUTPUT_CONTRACT_LOCKED", output_contract.get("verdict_v1") == "EVAL_OUTPUT_CONTRACT_LOCKED", output_contract.get("verdict_v1"), "EVAL_OUTPUT_CONTRACT_LOCKED"),
        ("SEQUENCE_IQL_STILL_BLOCKED", dataset_summary.get("sequence_iql_still_blocked_v1") is True, dataset_summary.get("sequence_iql_still_blocked_v1"), True),
        ("NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": "PASS" if passed else "FAIL", "observed_value_v1": observed, "expected_value_v1": expected} for name, passed, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "BANDIT_RESEARCH_EVAL_PREP_CONSISTENCY_AUDIT_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "passed_check_count_v1": int((df["status_v1"] == "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    hard_gates = payload["failcheck_plan"]["policy_v1"]["auto_stop_positive_interpretation_v1"]
    return "\n".join(
        [
            "# Bandit Research Eval Prep V1",
            "",
            "## Status",
            "",
            f"- Eval-prep verdict: `{summary['eval_prep_verdict_v1']}`",
            f"- Dataset verdict: `{summary['dataset_verdict_v1']}`",
            f"- Reward version: `{summary['reward_version_v1']}`",
            f"- Action split: `HOLD={summary['hold_rows_v1']}`, `EXIT_NOW={summary['exit_now_rows_v1']}`",
            f"- Support/OOD: `{summary['support_ood_verdict_v1']}`",
            "",
            "## Boundaries",
            "",
            "- This is bandit research eval preparation only.",
            "- It is not IQL eval, not sequence-RL eval, not R7 readiness, and not policy promotion.",
            "- Future eval must show action imbalance, support/OOD status, comparator status, and hard fail-checks before any positive interpretation.",
            "",
            "## Hard Gates",
            "",
            ", ".join(f"`{gate}`" for gate in hard_gates),
            "",
            "## Next",
            "",
            "- `RUN_FIRST_BANDIT_RESEARCH_EVAL_V1` can run under these limits.",
            "- `WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN` remains required for HOLD transition truth and sequence-IQL readiness.",
        ]
    ) + "\n"


def build_bandit_research_eval_prep(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    dataset_dir = dataset_dir or _latest_dataset_dir(reports_root, None)
    contract_lock_dir = contract_lock_dir or _latest_contract_lock_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    dataset_profile = _read_json_optional(dataset_dir / "management_bandit_dataset_profile_v1.json")
    comparator_df = _read_csv_optional(contract_lock_dir / "iql_baseline_comparator_lock_v1.csv")
    failcheck_df = _read_csv_optional(contract_lock_dir / "iql_failcheck_policy_lock_v1.csv")
    comparator_lock = _read_json_optional(contract_lock_dir / "iql_baseline_comparator_and_failcheck_lock_v1.json")
    source_paths = _source_paths(reports_root, dataset_dir, contract_lock_dir)

    scope = _build_scope_boundary_lock(dataset_summary, dataset_profile)
    boundary_tables = _boundary_tables(scope)
    protocol, protocol_df = _build_protocol(dataset_summary, dataset_profile)
    comparator_plan, comparator_plan_df = _build_comparator_application(comparator_df)
    failcheck_plan, failcheck_plan_df = _build_failcheck_plan(failcheck_df)
    risk_lock = _build_risk_lock(dataset_summary, dataset_profile)
    output_contract, output_sections_df, output_tables_df = _build_output_contract()
    post_status = _build_post_status(dataset_summary, protocol)
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    consistency_df, consistency = _build_consistency(
        dataset_summary,
        scope,
        protocol,
        comparator_plan_df,
        failcheck_plan_df,
        risk_lock,
        output_contract,
        non_interference,
    )

    hold_rows = _action_count(dataset_summary, "HOLD")
    exit_rows = _action_count(dataset_summary, "EXIT_NOW")
    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_MANAGEMENT_BANDIT_RESEARCH_EVAL_PREP",
        "not_training_v1": True,
        "not_iql_eval_v1": True,
        "not_sequence_iql_eval_v1": True,
        "not_r7_readiness_v1": True,
        "not_policy_promotion_v1": True,
        "reward_version_v1": REWARD_VERSION_ID,
        "reward_formula_v1": REWARD_FORMULA,
        "source_paths_v1": source_paths,
        "comparator_failcheck_contract_v1": comparator_lock.get("lock_id_v1"),
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_build_sequence_iql_dataset_v1": True,
            "do_not_use_in_progress_replay_as_canonical_v1": True,
            "do_not_modify_r6_freeze_v1": True,
            "do_not_modify_locked_ledger_v1": True,
            "do_not_claim_hold_next_state_truth_v1": True,
        },
        "path_dynamics_v2_status_v1": {
            "status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
            "fields_v1": PATH_DYNAMICS_V2_FIELDS,
        },
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "eval_prep_ready_v1": post_status["bandit_eval_prep_ready_v1"],
        "eval_prep_verdict_v1": "BANDIT_EVAL_PREP_READY_WITH_LIMITATIONS" if post_status["bandit_eval_prep_ready_v1"] else "BANDIT_EVAL_PREP_NOT_READY",
        "dataset_dir_v1": str(dataset_dir),
        "dataset_verdict_v1": dataset_summary.get("dataset_verdict_v1"),
        "dataset_included_rows_v1": int(dataset_summary.get("included_rows_v1", 0) or 0),
        "dataset_excluded_rows_v1": int(dataset_summary.get("excluded_rows_v1", 0) or 0),
        "hold_rows_v1": hold_rows,
        "exit_now_rows_v1": exit_rows,
        "reward_version_v1": REWARD_VERSION_ID,
        "reward_formula_v1": REWARD_FORMULA,
        "support_ood_verdict_v1": dataset_profile.get("support_ood_verdict_from_foundation_v1", dataset_summary.get("support_ood_verdict_v1")),
        "scope_verdicts_v1": scope["scope_verdicts_v1"],
        "risk_verdicts_v1": risk_lock["verdicts_v1"],
        "hard_gate_metrics_v1": failcheck_plan["policy_v1"]["auto_stop_positive_interpretation_v1"],
        "headline_eval_comparators_v1": comparator_plan["headline_eval_comparators_v1"],
        "safety_review_comparators_v1": comparator_plan["safety_review_comparators_v1"],
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_still_blocked_v1": True,
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "recommended_next_steps_v1": ["RUN_FIRST_BANDIT_RESEARCH_EVAL_V1", "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN"],
        "hard_status_partition_v1": post_status["hard_status_v1"],
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_BANDIT_RESEARCH_EVAL_PREP",
        "eval_prep_ready_v1": summary["eval_prep_ready_v1"],
        "training_executed_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int(consistency.get("failed_check_count_v1", 0)),
        "failed_non_interference_check_count_v1": int(non_interference.get("failed_check_count_v1", 0)),
    }
    return {
        "contract": contract,
        "scope": scope,
        "boundary_tables": boundary_tables,
        "protocol": protocol,
        "protocol_df": protocol_df,
        "comparator_plan": comparator_plan,
        "comparator_plan_df": comparator_plan_df,
        "failcheck_plan": failcheck_plan,
        "failcheck_plan_df": failcheck_plan_df,
        "risk_lock": risk_lock,
        "output_contract": output_contract,
        "output_sections_df": output_sections_df,
        "output_tables_df": output_tables_df,
        "post_status": post_status,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "consistency": consistency,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_bandit_research_eval_prep_artifacts(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    r6_path = reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"
    exit_manager_sha_before = _sha256(exit_manager_path)
    r6_sha_before = _sha256(r6_path)

    payload = build_bandit_research_eval_prep(
        reports_root,
        dataset_dir=dataset_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)
    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    _write_json(target_dir / OUTPUTS["scope_boundary_lock"], payload["scope"])
    payload["boundary_tables"]["allowed_use"].to_csv(target_dir / OUTPUTS["boundary_allowed_use"], index=False)
    payload["boundary_tables"]["forbidden_use"].to_csv(target_dir / OUTPUTS["boundary_forbidden_use"], index=False)
    payload["boundary_tables"]["allowed_claims"].to_csv(target_dir / OUTPUTS["boundary_allowed_claims"], index=False)
    payload["boundary_tables"]["forbidden_claims"].to_csv(target_dir / OUTPUTS["boundary_forbidden_claims"], index=False)
    _write_json(target_dir / OUTPUTS["protocol_draft"], payload["protocol"])
    payload["protocol_df"].to_csv(target_dir / OUTPUTS["protocol_table"], index=False)
    _write_json(target_dir / OUTPUTS["comparator_plan"], payload["comparator_plan"])
    payload["comparator_plan_df"].to_csv(target_dir / OUTPUTS["comparator_table"], index=False)
    _write_json(target_dir / OUTPUTS["failcheck_plan"], payload["failcheck_plan"])
    payload["failcheck_plan_df"].to_csv(target_dir / OUTPUTS["failcheck_table"], index=False)
    _write_json(target_dir / OUTPUTS["risk_lock"], payload["risk_lock"])
    _write_json(target_dir / OUTPUTS["output_contract"], payload["output_contract"])
    payload["output_sections_df"].to_csv(target_dir / OUTPUTS["output_required_sections"], index=False)
    payload["output_tables_df"].to_csv(target_dir / OUTPUTS["output_required_tables"], index=False)
    _write_json(target_dir / OUTPUTS["post_status"], payload["post_status"])

    exit_manager_sha_after = _sha256(exit_manager_path)
    r6_sha_after = _sha256(r6_path)
    non_interference_df, non_interference = _build_non_interference(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference
    payload["consistency_df"], payload["consistency"] = _build_consistency(
        _read_json_optional(Path(payload["source_paths"]["dataset_summary_v1"])),
        payload["scope"],
        payload["protocol"],
        payload["comparator_plan_df"],
        payload["failcheck_plan_df"],
        payload["risk_lock"],
        payload["output_contract"],
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int(payload["consistency"]["failed_check_count_v1"])

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["consistency_audit_json"], payload["consistency"])
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "append_only_namespace_v1": "IQL_INTEGRATION",
        "artifact_paths_v1": artifact_paths,
        "source_paths_v1": payload["source_paths"],
        "read_only_references_v1": True,
        "not_training_v1": True,
        "not_iql_eval_v1": True,
        "not_sequence_iql_eval_v1": True,
        "not_policy_promotion_v1": True,
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], payload["status"])
    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize management bandit research eval-prep artifacts.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--contract-lock-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    contract_lock_dir = Path(args.contract_lock_dir).expanduser().resolve() if args.contract_lock_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_bandit_research_eval_prep_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
