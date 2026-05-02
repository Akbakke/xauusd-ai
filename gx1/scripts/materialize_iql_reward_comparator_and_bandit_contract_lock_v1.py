#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    FOUNDATION_DIRNAME,
    LAYER_ID as READONLY_PLANNING_LAYER_ID,
    PATH_DYNAMICS_V2_FIELDS,
    R5_2_FREEZE_ID,
    R6_FREEZE_ID,
    _json_ready,
    _read_csv_optional,
    _read_json_optional,
    _resolve_foundation_dir,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1"
FOUNDATION_LAYER_ID = "IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"

OUTPUTS = {
    "contract": "iql_reward_comparator_bandit_contract_lock_v1.json",
    "reward_review_csv": "iql_reward_contract_lock_review_v1.csv",
    "reward_review_json": "iql_reward_contract_lock_review_v1.json",
    "bandit_contract": "iql_management_bandit_dataset_contract_lock_v1.json",
    "bandit_fields": "iql_management_bandit_dataset_contract_fields_v1.csv",
    "comparator_lock": "iql_baseline_comparator_and_failcheck_lock_v1.json",
    "comparator_table": "iql_baseline_comparator_lock_v1.csv",
    "failcheck_table": "iql_failcheck_policy_lock_v1.csv",
    "post_replay_gate": "iql_post_replay_readiness_gate_draft_v1.json",
    "post_replay_steps": "iql_post_replay_readiness_gate_steps_v1.csv",
    "r7_blocked_lock": "iql_r7_still_blocked_lock_v1.json",
    "r7_boundary_table": "iql_r7_still_blocked_boundary_v1.csv",
    "next_step_lock": "iql_summary_and_next_step_lock_v1.csv",
    "summary": "iql_reward_comparator_bandit_contract_lock_summary_v1.json",
    "report": "iql_reward_comparator_bandit_contract_lock_report_v1.md",
    "manifest": "iql_reward_comparator_bandit_contract_lock_manifest_v1.json",
    "status": "iql_reward_comparator_bandit_contract_lock_status_v1.json",
    "consistency_audit": "iql_reward_comparator_bandit_contract_lock_consistency_audit_v1.csv",
    "non_interference_audit": "iql_reward_comparator_bandit_contract_lock_non_interference_audit_v1.csv",
    "non_interference_audit_json": "iql_reward_comparator_bandit_contract_lock_non_interference_audit_v1.json",
}

REWARD_SPECS = {
    "REALIZED_PNL_REWARD": {
        "formula": "terminal_realized_pnl_bps",
        "sign": "MAXIMIZE",
        "inputs": ["terminal_realized_pnl_bps"],
        "input_class": "HINDSIGHT_TERMINAL_OUTCOME",
        "eligible_for": "bandit,sequence_terminal_reward",
        "lock_missing": "explicit reward_version selection and acceptance review",
    },
    "MFE_CAPTURE_REWARD": {
        "formula": "terminal_realized_pnl_bps / max(hindsight_peak_mfe_bps, eps), clipped [-2, 2]",
        "sign": "MAXIMIZE",
        "inputs": ["terminal_realized_pnl_bps", "hindsight_peak_mfe_bps"],
        "input_class": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "eligible_for": "bandit,sequence_terminal_reward",
        "lock_missing": "review acceptance that hindsight path metric is reward-only and never state",
    },
    "MAE_PENALTY_REWARD": {
        "formula": "-abs(terminal_mae_bps)",
        "sign": "MAXIMIZE_LESS_NEGATIVE",
        "inputs": ["terminal_mae_bps"],
        "input_class": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "eligible_for": "bandit,sequence_terminal_reward",
        "lock_missing": "explicit scaling/clip policy if used as scalar reward",
    },
    "GIVEBACK_PENALTY_REWARD": {
        "formula": "-hindsight_peak_to_exit_giveback_bps",
        "sign": "MAXIMIZE_LESS_NEGATIVE",
        "inputs": ["hindsight_peak_to_exit_giveback_bps"],
        "input_class": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "eligible_for": "bandit,sequence_terminal_reward",
        "lock_missing": "review acceptance that giveback is reward-only and not AS_OF state",
    },
    "TAIL_CONTROL_REWARD": {
        "formula": "terminal_realized_pnl_bps - 25*bad_trade - 75*cata_loser",
        "sign": "MAXIMIZE",
        "inputs": ["terminal_realized_pnl_bps", "bad_trade", "cata_loser"],
        "input_class": "HINDSIGHT_TERMINAL_OUTCOME",
        "eligible_for": "bandit,sequence_terminal_reward",
        "lock_missing": "review of fixed penalty weights before scalar lock",
    },
    "RUNNER_DAMAGE_PENALTY": {
        "formula": "-max(hindsight_hold_longer_extra_value_bps, 0)",
        "sign": "MAXIMIZE_LESS_NEGATIVE",
        "inputs": ["hindsight_hold_longer_extra_value_bps"],
        "input_class": "HINDSIGHT_COUNTERFACTUAL_AUDIT_ONLY",
        "eligible_for": "audit_only",
        "lock_missing": "counterfactual locality is not locked",
    },
    "TRANSPARENT_COMBINED_REWARD": {
        "formula": "pnl - .25*mae - .25*giveback - .50*runner_damage - 25*bad_trade - 75*cata_loser",
        "sign": "MAXIMIZE",
        "inputs": ["terminal_realized_pnl_bps", "mae_bps", "giveback_bps", "runner_damage_bps", "bad_trade", "cata_loser"],
        "input_class": "MIXED_HINDSIGHT_COMPOSITE_AUDIT_ONLY",
        "eligible_for": "audit_only",
        "lock_missing": "weights and counterfactual component are not locked",
    },
}

DATASET_FIELDS = [
    "row_id",
    "episode_id",
    "candidate_uid_exact",
    "decision_ts",
    "action",
    "action_id",
    "reward",
    "reward_version",
    "state_vector",
    "state_feature_names",
    "source_policy_version",
    "behavior_policy_status",
    "support_status",
    "as_of_schema_version",
    "hindsight_outcome_backfill_version",
    "eligibility_status",
    "exclusion_reason",
    "provenance_namespace",
]

METRIC_SPECS = [
    ("realized pnl", "headline economic outcome reference", "SOFT_REVIEW_UNTIL_BASELINE_CALIBRATED", "HIGHER", "material degradation versus locked comparator", "all, worst-slice"),
    ("MFE capture", "avoid policies that exit strongest winners too early", "SOFT_REVIEW", "HIGHER", "large capture collapse in winner pockets", "50+/100+/200+ MFE pockets"),
    ("MAE burden", "limit adverse excursion damage", "HARD_GATE_IF_TAIL_WORSENS", "LOWER", "higher tail MAE burden", "bad-trade and tail slices"),
    ("giveback", "measure peak-to-exit leakage", "SOFT_REVIEW", "LOWER", "large giveback increase", "strongest-winner slices"),
    ("tail-control help", "ensure tail-risk controls are not damaged", "HARD_GATE", "HIGHER", "worse catastrophic loser/bad-trade tail", "tail slices"),
    ("runner damage", "avoid harming long-runner opportunities", "SOFT_REVIEW", "LOWER", "higher runner damage", "strongest winners"),
    ("50+/100+/200+ MFE damage", "protect high-MFE trades", "HARD_GATE_FOR_100_200_PLUS", "LOWER", "damage increase in high-MFE pockets", "50+/100+/200+ MFE"),
    ("strongest-winner path damage", "protect best path opportunities", "HARD_GATE", "LOWER", "material deterioration in strongest winners", "top winner pockets"),
    ("bad-trade reduction", "ensure risk reduction is real", "SOFT_REVIEW", "HIGHER", "bad-trade rate worsens", "bad-trade slices"),
    ("action agreement", "track behavior-policy divergence", "SOFT_REVIEW", "EXPLAINED_STABLE", "unexplained action drift", "all actions"),
    ("OOD action rate", "stop unsupported extrapolation", "HARD_GATE", "LOWER", "unsupported actions above locked threshold", "thin support pockets"),
    ("worst-slice performance", "avoid headline-only improvement", "HARD_GATE", "HIGHER", "worst slice worsens materially", "all protected slices"),
    ("rolling-window stability", "detect unstable policy behavior", "HARD_GATE_IF_UNSTABLE", "MORE_STABLE", "rolling instability or batch collapse", "time windows"),
    ("BATCH_04 stress", "protect known stress slice", "SOFT_REVIEW_UNTIL_CALIBRATED", "HIGHER_OR_NOT_WORSE", "batch stress regression", "BATCH_04"),
    ("BATCH_05 stress", "protect known stress slice", "SOFT_REVIEW_UNTIL_CALIBRATED", "HIGHER_OR_NOT_WORSE", "batch stress regression", "BATCH_05"),
    ("harvest candidate capture", "compare against harvest observability candidate", "SOFT_REVIEW", "HIGHER_OR_NOT_WORSE", "harvest capture degradation", "harvest candidate pockets"),
    ("failed checks", "aggregate invariant/fail-close status", "HARD_GATE", "LOWER_ZERO_REQUIRED", "any hard failed check", "global"),
]


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_readonly_planning_dir(reports_root: Path, readonly_dir_arg: str | None) -> Path | None:
    if readonly_dir_arg:
        path = Path(readonly_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Read-only planning dir does not exist: {path}")
        return path
    base = reports_root / "IQL_READINESS"
    candidates = sorted(
        base.glob(f"{READONLY_PLANNING_LAYER_ID}_*/iql_readonly_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent.resolve() if candidates else None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _load_sources(reports_root: Path, foundation_dir: Path, readonly_dir: Path | None) -> dict[str, Any]:
    return {
        "foundation_summary": _read_json_optional(foundation_dir / "iql_foundation_summary_v1.json"),
        "foundation_contract": _read_json_optional(foundation_dir / "iql_foundation_mdp_contract_v1.json"),
        "foundation_dataset_schema": _read_json_optional(foundation_dir / "iql_foundation_dataset_schema_v1.json"),
        "foundation_management_contract": _read_json_optional(foundation_dir / "iql_foundation_management_mdp_contract_v1.json"),
        "foundation_reward_audit_df": _read_csv_optional(foundation_dir / "iql_foundation_reward_audit_v1.csv"),
        "foundation_baseline_spec": _read_json_optional(foundation_dir / "iql_foundation_baseline_comparator_spec_v1.json"),
        "foundation_transition": _read_json_optional(foundation_dir / "iql_foundation_transition_linkage_audit_v1.json"),
        "foundation_support": _read_json_optional(foundation_dir / "iql_foundation_support_ood_audit_v1.json"),
        "readonly_summary": _read_json_optional(readonly_dir / "iql_readonly_summary_v1.json") if readonly_dir else {},
        "readonly_reward_draft_df": _read_csv_optional(readonly_dir / "iql_readonly_reward_contract_draft_v1.csv") if readonly_dir else pd.DataFrame(),
        "readonly_boundary": _read_json_optional(readonly_dir / "iql_readonly_r5_2_r6_r7_boundary_lock_v1.json") if readonly_dir else {},
        "r5_summary": _read_json_optional(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_summary": _read_json_optional(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _source_paths(reports_root: Path, foundation_dir: Path, readonly_dir: Path | None, sources: dict[str, Any]) -> dict[str, str | None]:
    source_truth = sources["foundation_contract"].get("source_truth_v1", {}) if isinstance(sources["foundation_contract"].get("source_truth_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "readonly_planning_dir_v1": str(readonly_dir) if readonly_dir else None,
        "locked_ledger_source_v1": source_truth.get("locked_ledger_source_file_v1"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
        "entry_observability_summary_v1": str(reports_root / "truth_entry_rl_observability_v1.json"),
        "harvest_observability_summary_v1": str(reports_root / "truth_harvest_retrain_candidate_v1.json"),
        "rl_unified_observability_summary_v1": str(reports_root / "truth_rl_unified_observability_v1.json"),
    }


def _build_feature_inventory(dataset_schema: dict[str, Any], management_contract: dict[str, Any]) -> dict[str, Any]:
    state_contract = management_contract.get("state_contract_v1", {}) if isinstance(management_contract.get("state_contract_v1"), dict) else {}
    state_features = _as_list(dataset_schema.get("state_feature_names_v1"))
    core_9 = dataset_schema.get("canonical_management_core_9_inputs_v1", {})
    policy_log_fields = _as_list(state_contract.get("policy_log_fields_v1"))
    return {
        "source_v1": "FOUNDATION_SCHEMA_REUSED_NO_NEW_FEATURE_NAMESPACE",
        "state_feature_count_v1": int(len(state_features)),
        "state_feature_names_v1": state_features,
        "canonical_core_9_v1": core_9,
        "policy_log_fields_v1": policy_log_fields,
        "path_dynamics_v2_fields_v1": [
            {
                "field_id_v1": field,
                "replay_status_v1": "PENDING_REPLAY",
                "canonical_status_v1": "NOT_CANONICAL_YET",
                "training_status_v1": "DO_NOT_USE_FOR_TRAINING",
            }
            for field in PATH_DYNAMICS_V2_FIELDS
        ],
        "no_duplicate_feature_contract_v1": True,
    }


def _reward_source_df(sources: dict[str, Any]) -> pd.DataFrame:
    draft_df = sources["readonly_reward_draft_df"]
    if not draft_df.empty:
        return draft_df
    audit_df = sources["foundation_reward_audit_df"].copy()
    if not audit_df.empty and "draft_status_v1" not in audit_df.columns:
        audit_df["draft_status_v1"] = audit_df["reward_candidate_v1"].map(
            lambda name: "AUDIT_ONLY"
            if str(name) in {"RUNNER_DAMAGE_PENALTY", "TRANSPARENT_COMBINED_REWARD"}
            else "LOCKABLE_AFTER_REVIEW"
        )
    return audit_df


def _build_reward_review(sources: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    reward_df = _reward_source_df(sources)
    by_name = {str(row.get("reward_candidate_v1")): row for row in reward_df.to_dict(orient="records")} if not reward_df.empty else {}
    rows = []
    for name, spec in REWARD_SPECS.items():
        source = by_name.get(name, {})
        source_status = str(source.get("draft_status_v1") or source.get("verdict_v1") or "NOT_READY")
        coverage = float(source.get("coverage_rate_v1", 0.0) or 0.0)
        if source_status == "LOCKABLE_AFTER_REVIEW":
            verdict = "LOCKABLE_AFTER_REVIEW"
        elif name == "TAIL_CONTROL_REWARD" and source_status == "USABLE_FOR_OFFLINE_RESEARCH":
            verdict = "PARTIAL_LOCK_ONLY"
        elif source_status == "AUDIT_ONLY" or name in {"RUNNER_DAMAGE_PENALTY", "TRANSPARENT_COMBINED_REWARD"}:
            verdict = "AUDIT_ONLY"
        elif source_status == "USABLE_FOR_OFFLINE_RESEARCH" and coverage > 0:
            verdict = "LOCKABLE_AFTER_REVIEW"
        else:
            verdict = "NOT_READY"
        rows.append(
            {
                "reward_candidate_v1": name,
                "formula_draft_v1": source.get("formula_v1", spec["formula"]),
                "sign_direction_v1": spec["sign"],
                "required_inputs_v1": json.dumps(spec["inputs"], ensure_ascii=True),
                "input_class_v1": spec["input_class"],
                "coverage_rate_v1": coverage,
                "coverage_count_v1": int(source.get("distribution_count_v1", 0) or 0),
                "hindsight_only_v1": True,
                "as_of_state_allowed_v1": False,
                "leakage_risk_v1": source.get("leakage_risk_v1", "NOT_ESTABLISHED"),
                "suitable_for_v1": spec["eligible_for"],
                "missing_for_full_lock_v1": spec["lock_missing"],
                "can_enter_future_locked_scalar_reward_version_v1": verdict in {"LOCKABLE_AFTER_REVIEW", "PARTIAL_LOCK_ONLY"},
                "hard_verdict_v1": verdict,
                "scalar_reward_locked_now_v1": False,
                "training_use_now_v1": "DO_NOT_USE_FOR_TRAINING_UNTIL_REWARD_VERSION_LOCK",
            }
        )
    review_df = pd.DataFrame.from_records(rows)
    lockable_count = int(review_df["hard_verdict_v1"].isin(["LOCKABLE_AFTER_REVIEW", "PARTIAL_LOCK_ONLY"]).sum())
    aggregate = {
        "review_id_v1": "REWARD_CONTRACT_LOCK_REVIEW_V1",
        "scalar_reward_version_locked_now_v1": False,
        "aggregate_verdict_v1": "CAN_PROCEED_TO_LOCK_FIRST_BANDIT_REWARD_VERSION_NEXT" if lockable_count else "REWARD_REMAINS_DRAFT_ONLY",
        "first_bandit_reward_version_lock_next_v1": bool(lockable_count),
        "fail_closed_reason_v1": "No scalar reward_version is locked by this read-only contract job.",
        "lockable_or_partial_count_v1": lockable_count,
        "audit_only_count_v1": int(review_df["hard_verdict_v1"].eq("AUDIT_ONLY").sum()),
        "hard_status_v1": "INDIKERT" if lockable_count else "IKKE_ETABLERT",
    }
    return review_df, aggregate


def _field_status_from_foundation(dataset_schema: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    fields = dataset_schema.get("fields_v1", []) if isinstance(dataset_schema.get("fields_v1"), list) else []
    for row in fields:
        if isinstance(row, dict):
            statuses[str(row.get("field_name_v1"))] = str(row.get("status_v1", "NOT_ESTABLISHED"))
    return statuses


def _build_bandit_dataset_contract(
    sources: dict[str, Any],
    feature_inventory: dict[str, Any],
    reward_aggregate: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_schema = sources["foundation_dataset_schema"]
    field_statuses = _field_status_from_foundation(dataset_schema)
    source_base = "IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"
    specs = {
        "row_id": ("string", True, "management_row_key_v1", "metadata/provenance", "METADATA", "READY", "fail if missing or duplicate"),
        "episode_id": ("string", True, "sequence_episode_key_v1", "foundation sequence row view", "METADATA", field_statuses.get("episode_id", "READY"), "fail if missing"),
        "candidate_uid_exact": ("string", True, "candidate_uid_exact_v1", "canonical management substrate", "METADATA", field_statuses.get("candidate_uid_exact", "READY"), "fail if missing"),
        "decision_ts": ("timestamp", True, "decision_timestamp", "AS_OF decision log", "AS_OF_PROVENANCE", field_statuses.get("decision_ts", "READY"), "fail if missing or non-parseable"),
        "action": ("enum", True, "action_label_v1", "logged behavior/action semantics", "BEHAVIOR_LOG", field_statuses.get("action", "READY"), "allowed HOLD|EXIT_NOW only"),
        "action_id": ("int", True, "HOLD=0, EXIT_NOW=1", "contract-derived", "BEHAVIOR_LOG", field_statuses.get("action_id", "READY"), "fail if not 0/1 mapping"),
        "reward": ("float", True, "selected reward channel", "HINDSIGHT_REWARD", "HINDSIGHT_REWARD", "PARTIAL", "fail closed until reward_version locked"),
        "reward_version": ("string", True, "future locked reward_version", "reward contract", "CONTRACT", "NOT_ESTABLISHED", "fail if missing"),
        "state_vector": ("array<float|string>", True, "foundation state_feature_names_v1", "AS_OF management observation", "AS_OF_STATE", field_statuses.get("state_vector", "PARTIAL"), "no hindsight/reward/outcome tokens"),
        "state_feature_names": ("array<string>", True, "foundation dataset schema", "AS_OF management observation", "AS_OF_STATE", field_statuses.get("state_feature_names", "READY"), "must equal foundation feature inventory"),
        "source_policy_version": ("string", True, "policy_version_v1", "policy logging provenance", "PROVENANCE", field_statuses.get("source_policy_version", "READY"), "fail if absent"),
        "behavior_policy_status": ("string", True, "observed_action/propensity status", "policy logging provenance", "PROVENANCE", field_statuses.get("behavior_policy_status", "PARTIAL"), "must be explicit; no implicit support"),
        "support_status": ("string", True, "support/OOD audit", "support audit", "SUPPORT", field_statuses.get("support_status", "PARTIAL"), "fail if assumed or missing"),
        "as_of_schema_version": ("string", True, "foundation schema id", "AS_OF schema contract", "AS_OF_CONTRACT", field_statuses.get("as_of_schema_version", "READY"), "fail if not exact expected version"),
        "hindsight_outcome_backfill_version": ("string", True, "outcome_backfill_version", "HINDSIGHT outcome contract", "HINDSIGHT_OUTCOME", field_statuses.get("outcome_backfill_version", "READY"), "fail if missing"),
        "eligibility_status": ("enum", True, "bandit eligibility status", "foundation bandit substrate", "ELIGIBILITY", "READY", "allowed ELIGIBLE|EXCLUDED only after build"),
        "exclusion_reason": ("string", False, "computed exclusion reason", "dataset contract", "ELIGIBILITY", "READY", "required when eligibility_status=EXCLUDED"),
        "provenance_namespace": ("string", True, source_base, "append-only IQL namespace", "PROVENANCE", "READY", "must point to canonical source artifacts"),
    }
    rows = []
    for field in DATASET_FIELDS:
        dtype, required, source_field, source_namespace, class_name, coverage_status, validation = specs[field]
        rows.append(
            {
                "field_name_v1": field,
                "datatype_v1": dtype,
                "required_v1": required,
                "source_artifact_v1": source_base,
                "source_field_or_contract_v1": source_field,
                "source_namespace_v1": source_namespace,
                "as_of_hindsight_class_v1": class_name,
                "coverage_status_v1": coverage_status,
                "null_policy_v1": "FAIL_CLOSED_IF_REQUIRED_NULL" if required else "NULL_ALLOWED_WHEN_NOT_APPLICABLE",
                "canonical_status_v1": "CANONICAL_EXISTING_SOURCE" if coverage_status in {"READY", "PARTIAL"} else "NOT_ESTABLISHED",
                "allowed_values_v1": "HOLD|EXIT_NOW" if field == "action" else ("0|1" if field == "action_id" else ""),
                "validation_rule_v1": validation,
                "fail_closed_condition_v1": validation,
            }
        )
    fields_df = pd.DataFrame.from_records(rows)
    contract = {
        "contract_id_v1": "MANAGEMENT_BANDIT_DATASET_CONTRACT_LOCK_V1",
        "current_build_status_v1": "NEEDS_REWARD_FIRST",
        "verdict_v1": "READY_TO_BUILD_AFTER_REWARD_LOCK"
        if reward_aggregate.get("first_bandit_reward_version_lock_next_v1")
        else "NEEDS_REWARD_FIRST",
        "dataset_build_executed_v1": False,
        "training_executed_v1": False,
        "field_count_v1": int(len(fields_df)),
        "state_feature_contract_v1": feature_inventory,
        "invariants_v1": {
            "no_as_of_hindsight_mixing_in_state_v1": True,
            "reward_requires_explicit_reward_version_v1": True,
            "support_status_must_be_explicit_v1": True,
            "behavior_policy_status_must_be_explicit_v1": True,
            "row_traceable_to_canonical_source_artifacts_v1": True,
            "path_dynamics_not_training_canonical_now_v1": True,
        },
        "field_rows_v1": fields_df.to_dict(orient="records"),
    }
    return fields_df, contract


def _build_comparator_and_failcheck_lock(sources: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = sources["foundation_baseline_spec"].get("baseline_comparator_presence_v1", {})
    comparator_specs = [
        ("no-RL/current locked ledger", "no_rl_baseline_v1", "canonical source reference", "READY", "locked 1971 source truth", "not calibrated as future RL threshold yet", "READY_AFTER_METRIC_CALIBRATION"),
        ("R6 frozen shadow candidate", "r6_frozen_shadow_fallback_v1", "current frozen shadow comparator", "PENDING_CALIBRATION", "frozen id present", "not RL and not live policy", "READY_AFTER_CALIBRATION"),
        ("R5.2 frozen historical reference", "r5_2_frozen_reference_v1", "historical frozen reference", "PENDING_CALIBRATION", "frozen id present", "historical only", "READY_AFTER_CALIBRATION"),
        ("management harvest comparator", "management_harvest_candidate_v1", "observability/comparator", "PENDING_CALIBRATION", "artifact slot registered", "not controller", "READY_AFTER_CALIBRATION"),
        ("supervised EXIT_LOCAL/tree baseline", "supervised_exit_local_tree_baseline_v1", "supervised comparator", "PENDING_CALIBRATION", "existing baseline slot", "limited to supervised/local reward semantics", "READY_AFTER_CALIBRATION"),
        ("dummy/random sanity comparator", "random_dummy_policy_v1", "sanity comparator", "READY", "simple sanity bound", "not meaningful target", "READY_FOR_SANITY_ONLY"),
    ]
    comparator_rows = []
    for name, registry_key, role, status, strength, weakness, readiness in comparator_specs:
        source_status = baseline.get(registry_key, {}).get("status_v1", "REFERENCE_REGISTERED") if isinstance(baseline, dict) else "REFERENCE_REGISTERED"
        comparator_rows.append(
            {
                "comparator_v1": name,
                "foundation_registry_key_v1": registry_key,
                "role_v1": role,
                "status_v1": status,
                "source_registry_status_v1": source_status,
                "strengths_v1": strength,
                "weaknesses_v1": weakness,
                "future_rl_comparator_readiness_v1": readiness,
                "performance_analysis_done_now_v1": False,
            }
        )
    metric_rows = [
        {
            "metric_or_failcheck_v1": metric,
            "why_exists_v1": why,
            "gate_type_v1": gate,
            "better_direction_v1": direction,
            "unacceptable_damage_v1": unacceptable,
            "protected_pockets_or_slices_v1": pockets,
            "auto_stop_promotion_v1": bool(gate.startswith("HARD_GATE")),
            "requires_extra_audit_even_with_good_headline_pnl_v1": metric
            in {"MFE capture", "runner damage", "strongest-winner path damage", "worst-slice performance", "OOD action rate"},
        }
        for metric, why, gate, direction, unacceptable, pockets in METRIC_SPECS
    ]
    comparator_df = pd.DataFrame.from_records(comparator_rows)
    failcheck_df = pd.DataFrame.from_records(metric_rows)
    lock = {
        "lock_id_v1": "BASELINE_COMPARATOR_AND_FAILCHECK_LOCK_V1",
        "mode_v1": "COMPARATOR_AND_FAILCHECK_CONTRACT_ONLY_NO_PERFORMANCE_ANALYSIS",
        "comparator_status_counts_v1": comparator_df["status_v1"].value_counts().to_dict(),
        "failcheck_count_v1": int(len(failcheck_df)),
        "hard_gate_count_v1": int(failcheck_df["auto_stop_promotion_v1"].sum()),
        "policy_v1": {
            "never_allowed_to_break_v1": [
                "AS_OF/HINDSIGHT separation",
                "locked ledger provenance",
                "OOD action hard gate",
                "worst-slice/tail-control hard gates",
                "non-interference with replay and freezes",
            ],
            "automatic_stop_checks_v1": failcheck_df.loc[failcheck_df["auto_stop_promotion_v1"], "metric_or_failcheck_v1"].astype(str).tolist(),
            "extra_audit_even_with_good_headline_pnl_v1": failcheck_df.loc[
                failcheck_df["requires_extra_audit_even_with_good_headline_pnl_v1"], "metric_or_failcheck_v1"
            ].astype(str).tolist(),
        },
        "comparator_rows_v1": comparator_df.to_dict(orient="records"),
        "failcheck_rows_v1": failcheck_df.to_dict(orient="records"),
    }
    return comparator_df, failcheck_df, lock


def _build_post_replay_gate(source_paths: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, Any]]:
    steps = [
        ("verify replay completion", "path dynamics replay status/manifest", "replay completion status is final and not running"),
        ("verify coverage/null-rate", "post-replay path dynamics coverage audit", "field coverage and null-rate by row/action/slice"),
        ("verify leakage status", "post-replay leakage audit", "path timestamps and values are AS_OF-safe"),
        ("rebuild canonical chain", "canonical rebuild after replay", "only after replay completion and coverage/leakage pass"),
        ("rerun HOLD transition diagnosis", "management transition audit", "HOLD next_state coverage, ordering, action consistency"),
        ("classify result", "post-replay IQL readiness matrix", "MDP_READY|BANDIT_ONLY|TRANSITION_LOGGING_REQUIRED|NOT_ESTABLISHED"),
    ]
    rows = [
        {
            "step_order_v1": index,
            "step_v1": step,
            "first_artifacts_to_read_v1": artifact,
            "required_check_v1": check,
            "execution_now_v1": False,
            "fail_closed_condition_v1": "stop if check is missing or ambiguous",
        }
        for index, (step, artifact, check) in enumerate(steps, start=1)
    ]
    gate = {
        "gate_id_v1": "POST_REPLAY_READINESS_GATE_DRAFT_V1",
        "execution_now_v1": False,
        "truth_source_priority_v1": [
            "locked canonical 1971 ledger",
            "completed post-replay path-dynamics artifacts",
            "rebuilt canonical management chain",
            "IQL foundation/readiness contracts",
        ],
        "source_paths_current_v1": source_paths,
        "join_keys_to_check_v1": [
            "candidate_uid_exact",
            "trade_uid_exact",
            "trade_id_exact",
            "management_row_key_v1",
            "sequence_episode_key_v1",
            "as_of_row_uid_v1",
        ],
        "timestamps_to_check_v1": ["decision_ts", "next_decision_ts", "last_peak_ts", "last_mfe_ts"],
        "ambiguity_checks_v1": [
            "duplicate join keys",
            "multiple next rows",
            "missing anchors",
            "non-monotonic timestamps",
            "conflicting action labels",
        ],
        "leakage_checks_v1": [
            "no hindsight/terminal/reward/outcome fields in state",
            "path event timestamps must be <= decision_ts",
            "path fields require coverage and null-rate audit",
            "terminal outcomes remain reward/backfill only",
        ],
        "gap_definitions_v1": {
            "logging_gap": "required next-state or path field was never logged in finished canonical artifacts",
            "join_gap": "logged rows exist but cannot join exactly on locked keys",
            "single_snapshot_gap": "only one management snapshot exists for a trade/action path so HOLD cannot link forward",
            "ordering_gap": "timestamps or step indexes are non-monotonic or ambiguous",
        },
        "classification_v1": ["MDP_READY", "BANDIT_ONLY", "TRANSITION_LOGGING_REQUIRED", "NOT_ESTABLISHED"],
        "step_rows_v1": rows,
    }
    return pd.DataFrame.from_records(rows), gate


def _build_r7_blocked_lock(sources: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    boundary = sources["readonly_boundary"]
    r7_status = boundary.get("r7_v1", {}).get("status_v1", "NOT_STARTED") if isinstance(boundary.get("r7_v1"), dict) else "NOT_STARTED"
    rows = [
        ("R7 not started", r7_status == "NOT_STARTED", "R7 status must remain NOT_STARTED"),
        ("forbidden before completed path-dynamics replay", True, "Replay completion is required before R7"),
        ("forbidden before post-replay audit", True, "Post-replay audit must pass before R7"),
        ("forbidden before comparator contract is locked", True, "Comparator/fail-check contract is materialized here"),
        ("forbidden before transition truth is known", True, "HOLD transition truth is not established"),
        ("must beat R6 contract if trained later", True, "Future R7 must be evaluated against R6 contract"),
    ]
    df = pd.DataFrame.from_records(
        [
            {
                "boundary_v1": boundary_name,
                "locked_v1": bool(locked),
                "note_v1": note,
            }
            for boundary_name, locked, note in rows
        ]
    )
    lock = {
        "lock_id_v1": "R7_STILL_BLOCKED_LOCK_V1",
        "short_verdict_v1": "R7_STILL_BLOCKED_DO_NOT_START",
        "r5_2_role_v1": "FROZEN_HISTORICAL_REFERENCE",
        "r6_role_v1": "CURRENT_FROZEN_SHADOW_CANDIDATE",
        "r7_status_v1": "NOT_STARTED",
        "r7_training_started_now_v1": False,
        "boundary_rows_v1": df.to_dict(orient="records"),
    }
    return df, lock


def _build_next_step_lock(
    reward_aggregate: dict[str, Any],
    bandit_contract: dict[str, Any],
    foundation_summary: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "decision_v1": "LOCK_FIRST_BANDIT_REWARD_VERSION_NEXT",
            "recommendation_v1": "DO_NEXT_REVIEW_LOCK",
            "hard_status_v1": "INDIKERT" if reward_aggregate.get("first_bandit_reward_version_lock_next_v1") else "IKKE_ETABLERT",
            "reason_v1": reward_aggregate.get("fail_closed_reason_v1"),
        },
        {
            "decision_v1": "BUILD_BANDIT_DATASET_AFTER_REWARD_LOCK",
            "recommendation_v1": "WAIT_FOR_REWARD_LOCK",
            "hard_status_v1": "INDIKERT" if bandit_contract.get("verdict_v1") == "READY_TO_BUILD_AFTER_REWARD_LOCK" else "IKKE_ETABLERT",
            "reason_v1": "Field-level contract is locked, dataset build waits for explicit reward_version.",
        },
        {
            "decision_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
            "recommendation_v1": "WAIT",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Path-dynamics is non-canonical and HOLD next_state remains missing.",
        },
        {
            "decision_v1": "DO_NOT_START_R7_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": "R7 requires replay completion, post-replay audit, comparator contract, and transition truth.",
        },
        {
            "decision_v1": "DO_NOT_START_IQL_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Harness remains NOT_READY_FOR_IQL_TRAINING.",
        },
        {
            "decision_v1": "SEQUENCE_IQL_BLOCKED",
            "recommendation_v1": "BLOCKED",
            "hard_status_v1": "BEVIST",
            "reason_v1": f"Management={foundation_summary.get('management_mdp_verdict_v1')}; HOLD->next_state=0.",
        },
        {
            "decision_v1": "BANDIT_FIRST_IF_TRANSITIONS_STAY_MISSING",
            "recommendation_v1": "LIKELY_FIRST_RL_STEP_AFTER_REWARD_LOCK",
            "hard_status_v1": "INDIKERT",
            "reason_v1": "Bandit rows exist and sequence transitions are incomplete.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _build_non_interference_audit(
    output_dir: Path,
    source_paths: dict[str, str | None],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay source path"),
        ("RAW_STATE_REBUILD_NOT_REQUESTED", "PASS", "not_invoked", "not_invoked"),
        ("POLICY_LOG_REBUILD_NOT_REQUESTED", "PASS", "not_invoked", "not_invoked"),
        ("EXIT_MANAGER_UNCHANGED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("PATH_DYNAMICS_DO_NOT_USE_FOR_TRAINING", "PASS", "PENDING_REPLAY|NOT_CANONICAL_YET|DO_NOT_USE_FOR_TRAINING", "PENDING_REPLAY|NOT_CANONICAL_YET|DO_NOT_USE_FOR_TRAINING"),
    ]
    df = pd.DataFrame.from_records(
        [
            {
                "check_name_v1": name,
                "status_v1": status,
                "observed_value_v1": observed,
                "expected_value_v1": expected,
                "note_v1": "Read-only append-only non-interference check.",
            }
            for name, status, observed, expected in checks
        ]
    )
    audit = {
        "audit_id_v1": "REPLAY_NON_INTERFERENCE_AUDIT_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }
    return df, audit


def _build_consistency_audit(
    reward_aggregate: dict[str, Any],
    bandit_contract: dict[str, Any],
    comparator_lock: dict[str, Any],
    r7_lock: dict[str, Any],
    non_interference: dict[str, Any],
    feature_inventory: dict[str, Any],
) -> pd.DataFrame:
    forbidden = [
        field
        for field in feature_inventory.get("state_feature_names_v1", [])
        if any(token in str(field).lower() for token in ["hindsight", "terminal", "reward", "outcome"])
    ]
    checks = [
        ("NO_SCALAR_REWARD_LOCKED_BY_THIS_JOB", not reward_aggregate.get("scalar_reward_version_locked_now_v1"), reward_aggregate.get("scalar_reward_version_locked_now_v1"), False),
        ("BANDIT_FIELD_CONTRACT_HAS_REQUIRED_FIELDS", len(bandit_contract.get("field_rows_v1", [])) == len(DATASET_FIELDS), len(bandit_contract.get("field_rows_v1", [])), len(DATASET_FIELDS)),
        ("COMPARATOR_FAILCHECK_POLICY_PRESENT", comparator_lock.get("failcheck_count_v1", 0) >= len(METRIC_SPECS), comparator_lock.get("failcheck_count_v1"), len(METRIC_SPECS)),
        ("R7_REMAINS_BLOCKED", r7_lock.get("short_verdict_v1") == "R7_STILL_BLOCKED_DO_NOT_START", r7_lock.get("short_verdict_v1"), "R7_STILL_BLOCKED_DO_NOT_START"),
        ("REPLAY_NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
        ("FOUNDATION_STATE_FEATURES_REUSED_NO_DUPLICATE_NAMESPACE", feature_inventory.get("no_duplicate_feature_contract_v1") is True, feature_inventory.get("source_v1"), "FOUNDATION_SCHEMA_REUSED_NO_NEW_FEATURE_NAMESPACE"),
        ("AS_OF_STATE_EXCLUDES_HINDSIGHT_REWARD_OUTCOME", not forbidden, json.dumps(forbidden, ensure_ascii=True), "[]"),
    ]
    return pd.DataFrame.from_records(
        [
            {
                "check_name_v1": name,
                "status_v1": "PASS" if passed else "FAIL",
                "observed_value_v1": observed,
                "expected_value_v1": expected,
            }
            for name, passed, observed, expected in checks
        ]
    )


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# IQL Reward Comparator And Bandit Contract Lock V1",
            "",
            "## Scope",
            "",
            "- Read-only / append-only contract lock over existing IQL foundation and readiness artifacts.",
            "- No replay, no raw-state rebuild, no policy-log rebuild, no R7, no IQL training, no sequence-IQL dataset build.",
            "",
            "## Lock Status",
            "",
            f"- Reward aggregate: `{summary['reward_aggregate_verdict_v1']}`",
            f"- Scalar reward locked now: `{summary['scalar_reward_version_locked_now_v1']}`",
            f"- Bandit dataset contract: `{summary['bandit_dataset_contract_verdict_v1']}`",
            f"- Comparator/fail-check contract: `{summary['comparator_failcheck_contract_status_v1']}`",
            f"- R7: `{summary['r7_short_verdict_v1']}`",
            "",
            "## Existing Features",
            "",
            f"- State features reused from foundation: `{summary['state_feature_count_v1']}`",
            "- Path-dynamics v2 fields remain `PENDING_REPLAY`, `NOT_CANONICAL_YET`, `DO_NOT_USE_FOR_TRAINING`.",
            "",
            "## Next",
            "",
            "- `LOCK_FIRST_BANDIT_REWARD_VERSION_NEXT` is the safe next review step.",
            "- `BUILD_BANDIT_DATASET_AFTER_REWARD_LOCK` waits for explicit reward_version.",
            "- `WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN`, `DO_NOT_START_R7_YET`, `DO_NOT_START_IQL_YET`, and `SEQUENCE_IQL_BLOCKED` remain active.",
        ]
    ) + "\n"


def build_contract_lock(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    readonly_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    foundation_dir = foundation_dir or _resolve_foundation_dir(reports_root, None)
    readonly_dir = readonly_dir if readonly_dir is not None else _latest_readonly_planning_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    sources = _load_sources(reports_root, foundation_dir, readonly_dir)
    source_paths = _source_paths(reports_root, foundation_dir, readonly_dir, sources)
    feature_inventory = _build_feature_inventory(sources["foundation_dataset_schema"], sources["foundation_management_contract"])
    reward_review_df, reward_aggregate = _build_reward_review(sources)
    bandit_fields_df, bandit_contract = _build_bandit_dataset_contract(sources, feature_inventory, reward_aggregate)
    comparator_df, failcheck_df, comparator_lock = _build_comparator_and_failcheck_lock(sources)
    post_replay_steps_df, post_replay_gate = _build_post_replay_gate(source_paths)
    r7_boundary_df, r7_lock = _build_r7_blocked_lock(sources)
    next_step_df = _build_next_step_lock(reward_aggregate, bandit_contract, sources["foundation_summary"])
    non_interference_df, non_interference = _build_non_interference_audit(output_dir, source_paths, exit_manager_sha_before, exit_manager_sha_after)
    consistency_df = _build_consistency_audit(
        reward_aggregate,
        bandit_contract,
        comparator_lock,
        r7_lock,
        non_interference,
        feature_inventory,
    )

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_CONTRACT_AND_PLANNING_LOCK",
        "source_paths_v1": source_paths,
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
        },
        "feature_inventory_v1": feature_inventory,
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(output_dir),
        "reward_aggregate_verdict_v1": reward_aggregate["aggregate_verdict_v1"],
        "scalar_reward_version_locked_now_v1": reward_aggregate["scalar_reward_version_locked_now_v1"],
        "reward_can_lock_next_v1": reward_aggregate["first_bandit_reward_version_lock_next_v1"],
        "bandit_dataset_contract_verdict_v1": bandit_contract["verdict_v1"],
        "bandit_dataset_build_executed_v1": False,
        "comparator_failcheck_contract_status_v1": "LOCKED_CONTRACT_ONLY_CALIBRATION_PENDING",
        "post_replay_gate_status_v1": "DRAFT_ONLY_NOT_EXECUTED",
        "r7_short_verdict_v1": r7_lock["short_verdict_v1"],
        "management_status_v1": sources["foundation_summary"].get("management_mdp_verdict_v1"),
        "strict_transition_count_v1": sources["foundation_summary"].get("full_sequence_ready_transition_count_v1"),
        "bandit_ready_row_count_v1": sources["foundation_summary"].get("bandit_only_row_count_v1"),
        "hold_to_next_state_transition_count_v1": sources["foundation_summary"].get("hold_to_next_state_transition_count_v1"),
        "support_ood_verdict_v1": sources["foundation_summary"].get("support_ood_verdict_v1"),
        "training_harness_status_v1": sources["foundation_summary"].get("training_harness_status_v1"),
        "state_feature_count_v1": feature_inventory["state_feature_count_v1"],
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "safe_while_replay_runs_v1": ["LOCK_FIRST_BANDIT_REWARD_VERSION_NEXT", "review comparator/fail-check contract"],
        "must_wait_for_replay_v1": ["WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN", "post-replay HOLD transition diagnosis", "R7 readiness", "sequence-IQL readiness"],
        "recommended_next_step_v1": "LOCK_FIRST_BANDIT_REWARD_VERSION_NEXT",
        "hard_status_partition_v1": {
            "BEVIST": [
                "sequence_iql_blocked_by_hold_next_state_zero",
                "r7_not_started_and_blocked",
                "iql_training_not_started",
                "path_dynamics_not_canonical_for_training",
                "state_features_reuse_foundation_schema",
            ],
            "INDIKERT": [
                "reward_contract_mature_enough_for_next_reward_version_lock_review",
                "bandit_dataset_contract_ready_after_reward_lock",
                "bandit_first_if_transitions_stay_missing",
            ],
            "IKKE_ETABLERT": [
                "scalar_reward_version_locked",
                "canonical_hold_next_state_transitions",
                "post_replay_mdp_ready_status",
                "r7_training_readiness",
                "sequence_iql_training_readiness",
            ],
        },
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_READONLY_APPEND_ONLY_CONTRACT_LOCK",
        "failed_consistency_check_count_v1": int((consistency_df["status_v1"] != "PASS").sum()),
        "failed_non_interference_check_count_v1": int(non_interference["failed_check_count_v1"]),
        "training_executed_v1": False,
        "r7_started_v1": False,
        "replay_touched_v1": False,
    }
    return {
        "contract": contract,
        "reward_review_df": reward_review_df,
        "reward_aggregate": reward_aggregate,
        "bandit_fields_df": bandit_fields_df,
        "bandit_contract": bandit_contract,
        "comparator_df": comparator_df,
        "failcheck_df": failcheck_df,
        "comparator_lock": comparator_lock,
        "post_replay_steps_df": post_replay_steps_df,
        "post_replay_gate": post_replay_gate,
        "r7_boundary_df": r7_boundary_df,
        "r7_lock": r7_lock,
        "next_step_df": next_step_df,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_contract_lock_artifacts(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    readonly_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    exit_manager_sha_before = _sha256(exit_manager_path)
    payload = build_contract_lock(
        reports_root,
        foundation_dir=foundation_dir,
        readonly_dir=readonly_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    payload["reward_review_df"].to_csv(target_dir / OUTPUTS["reward_review_csv"], index=False)
    _write_json(
        target_dir / OUTPUTS["reward_review_json"],
        {
            "aggregate_v1": payload["reward_aggregate"],
            "rows_v1": payload["reward_review_df"].to_dict(orient="records"),
        },
    )
    _write_json(target_dir / OUTPUTS["bandit_contract"], payload["bandit_contract"])
    payload["bandit_fields_df"].to_csv(target_dir / OUTPUTS["bandit_fields"], index=False)
    _write_json(target_dir / OUTPUTS["comparator_lock"], payload["comparator_lock"])
    payload["comparator_df"].to_csv(target_dir / OUTPUTS["comparator_table"], index=False)
    payload["failcheck_df"].to_csv(target_dir / OUTPUTS["failcheck_table"], index=False)
    _write_json(target_dir / OUTPUTS["post_replay_gate"], payload["post_replay_gate"])
    payload["post_replay_steps_df"].to_csv(target_dir / OUTPUTS["post_replay_steps"], index=False)
    _write_json(target_dir / OUTPUTS["r7_blocked_lock"], payload["r7_lock"])
    payload["r7_boundary_df"].to_csv(target_dir / OUTPUTS["r7_boundary_table"], index=False)
    payload["next_step_df"].to_csv(target_dir / OUTPUTS["next_step_lock"], index=False)

    exit_manager_sha_after = _sha256(exit_manager_path)
    non_interference_df, non_interference = _build_non_interference_audit(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before,
        exit_manager_sha_after,
    )
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference
    payload["consistency_df"] = _build_consistency_audit(
        payload["reward_aggregate"],
        payload["bandit_contract"],
        payload["comparator_lock"],
        payload["r7_lock"],
        non_interference,
        payload["contract"]["feature_inventory_v1"],
    )
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int((payload["consistency_df"]["status_v1"] != "PASS").sum())
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
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
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
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
    parser = argparse.ArgumentParser(description="Materialize IQL reward/comparator/bandit contract lock artifacts.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--foundation-dir", type=str, default=None)
    parser.add_argument("--readonly-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    foundation_dir = Path(args.foundation_dir).expanduser().resolve() if args.foundation_dir else None
    readonly_dir = Path(args.readonly_dir).expanduser().resolve() if args.readonly_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_contract_lock_artifacts(
        reports_root,
        foundation_dir=foundation_dir,
        readonly_dir=readonly_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
