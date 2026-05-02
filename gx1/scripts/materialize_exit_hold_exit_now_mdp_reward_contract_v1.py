#!/usr/bin/env python3
"""Lock the MDP and reward contract for exit-side HOLD/EXIT_NOW IQL research.

This gate is a research-only contract lock. It does not train any model, it
does not modify exit_manager or any runtime, and it does not open R6/adapter/
freeze/promo/live. It produces the locked MDP/reward semantics that the next
exit-side gates (state-feature contract, action-support augmentation, split &
leakage audit, off-policy eval harness, sanity training) must respect.

Hard design choices locked here:

  - State at bar t depends only on bars [0, t-1] within the same trade plus
    entry-context snapshot. AS_OF only.
  - Action set = {HOLD, EXIT_NOW}. Binary, no partial exits, no re-entry.
  - HOLD reward = 0 immediate. Terminal reward = exit-PNL evaluated under one
    of the locked reward variants (REALIZED_PNL, MFE_CAPTURE, MAE_PENALTY,
    GIVEBACK_PENALTY, TRANSPARENT_COMBINED). RUNNER_DAMAGE remains audit-only.
  - Episode = one trade. Episode_id = candidate_uid_v1.
  - Default discount gamma = 0.99 per M5 bar; sensitivity range
    [0.95, 0.999] enumerated for later gates.
  - Specific state fields are forbidden because they leak the realized exit
    (bar_count_v1, is_terminal_v1, exit_reason, all *_replay_end_obs,
    post_exit_*, trade_uid_v1, candidate_uid_v1).

The gate also writes an explicit pre-train dependency graph: training cannot
proceed until the next four gates pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1"

INPUT_PER_BAR_SCAFFOLD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T100845Z_LOCK"
)
INPUT_REBUILD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_20260429T081445Z_LOCK"
)

# Reuse helpers from contract_gate
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Locked design choices
# ---------------------------------------------------------------------------

ACTION_SET_V1 = {
    "HOLD": {
        "action_id_v1": 0,
        "meaning_v1": "continue holding the open position into the next bar",
        "transition_v1": "next bar of same trade if available; else episode terminates",
    },
    "EXIT_NOW": {
        "action_id_v1": 1,
        "meaning_v1": "close the open position at this bar's close (mid)",
        "transition_v1": "episode terminates; reward applied",
    },
}

HOLD_REWARD_LOCK_V1 = {
    "scheme_v1": "ZERO_IMMEDIATE_TERMINAL_ONLY",
    "hold_immediate_reward_v1": 0.0,
    "rationale_v1": (
        "Avoids per-bar shaping that can dominate offline value estimates. "
        "Bellman backup propagates terminal exit reward back through HOLD "
        "transitions controlled by the discount factor."
    ),
    "rejected_alternatives_v1": [
        {
            "alternative_v1": "PER_BAR_MARK_TO_MARKET_DELTA",
            "rejected_reason_v1": "Sums to terminal but distorts intermediate value, encourages chasing every bar",
        },
        {
            "alternative_v1": "HYBRID_SHAPING_PLUS_TERMINAL",
            "rejected_reason_v1": "Adds tunable shaping coefficient that becomes a free parameter and a leakage surface",
        },
    ],
}

TERMINAL_REWARD_VARIANTS_V1 = [
    {
        "reward_id_v1": "REALIZED_PNL_REWARD",
        "formula_v1": "pnl_bps_at_exit_close",
        "input_class_v1": "HINDSIGHT_TERMINAL_OUTCOME_REWARD_ONLY",
        "applies_to_action_v1": "EXIT_NOW_OR_FORCED_TERMINAL",
    },
    {
        "reward_id_v1": "MFE_CAPTURE_REWARD",
        "formula_v1": "pnl_bps_at_exit_close / max(mfe_bps_so_far_at_exit, eps), clipped [-2, 2]",
        "input_class_v1": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "applies_to_action_v1": "EXIT_NOW_OR_FORCED_TERMINAL",
    },
    {
        "reward_id_v1": "MAE_PENALTY_REWARD",
        "formula_v1": "pnl_bps_at_exit_close - 0.5 * abs(mae_bps_so_far_at_exit)",
        "input_class_v1": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "applies_to_action_v1": "EXIT_NOW_OR_FORCED_TERMINAL",
    },
    {
        "reward_id_v1": "GIVEBACK_PENALTY_REWARD",
        "formula_v1": "-max(mfe_bps_so_far_at_exit - pnl_bps_at_exit_close, 0)",
        "input_class_v1": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "applies_to_action_v1": "EXIT_NOW_OR_FORCED_TERMINAL",
    },
    {
        "reward_id_v1": "TRANSPARENT_COMBINED_REWARD",
        "formula_v1": "pnl - 0.25*abs(mae) - 0.25*max(mfe-pnl, 0)",
        "input_class_v1": "MIXED_HINDSIGHT_COMPOSITE_REWARD_ONLY",
        "applies_to_action_v1": "EXIT_NOW_OR_FORCED_TERMINAL",
    },
    {
        "reward_id_v1": "RUNNER_DAMAGE_PENALTY",
        "formula_v1": "-max(hindsight_hold_longer_extra_value_bps, 0)",
        "input_class_v1": "HINDSIGHT_COUNTERFACTUAL_AUDIT_ONLY",
        "applies_to_action_v1": "AUDIT_ONLY_NOT_TRAINING",
    },
]

EPISODE_DEFINITION_V1 = {
    "episode_id_field_v1": "candidate_uid_v1",
    "timestep_field_v1": "bar_index_v1",
    "terminal_definition_v1": (
        "Either agent action = EXIT_NOW, or bar is the realized exit bar in "
        "the historical trade (FORCED_TERMINAL). Both terminate the episode "
        "with the chosen reward variant evaluated at that bar's exit-PNL."
    ),
    "max_episode_length_bars_v1": 6000,
    "min_episode_length_bars_v1": 1,
}

TRANSITION_SEMANTICS_V1 = {
    "hold_at_non_terminal_v1": "transitions to next bar of same trade",
    "hold_at_realized_exit_bar_v1": (
        "FORCED_TERMINAL_HOLD: the agent wants to continue holding but the "
        "historical trade exited here. Episode terminates with reward "
        "evaluated at this bar's exit-PNL. This is a data-limit, not an "
        "agent choice. Off-policy evaluation must treat these as forced."
    ),
    "exit_now_at_any_bar_v1": "EXIT_NOW_TERMINAL: episode terminates; reward at this bar's exit-PNL",
    "no_re_entry_v1": True,
    "no_partial_exits_v1": True,
    "deterministic_v1": True,
}

DISCOUNT_LOCK_V1 = {
    "default_gamma_v1": 0.99,
    "bar_resolution_v1": "M5",
    "sensitivity_range_v1": [0.95, 0.97, 0.99, 0.995, 0.999],
    "rationale_v1": (
        "0.99 per M5 bar gives ~0.886 weight on a 12-bar horizon (1h) and "
        "~0.604 on 50 bars (4h). For shorter sniper-style trades the choice "
        "barely matters; for longer holds it matters significantly. Sensitivity "
        "must be enumerated in later gates before locking a single gamma."
    ),
}

STATE_REQUIREMENT_V1 = {
    "must_include_categories_v1": [
        {
            "category_v1": "TRADE_STATE_RUNNING",
            "must_include_fields_v1": [
                "running_pnl_at_close_bps",
                "running_mfe_bps",
                "running_mae_bps",
                "running_giveback_from_peak_bps",
                "bars_held",
            ],
            "lineage_v1": "AS_OF_FROM_BARS_LE_T_MINUS_1",
            "already_in_scaffold_v1": True,
        },
        {
            "category_v1": "MARKET_STATE_AT_BAR",
            "must_include_fields_v1": [
                "atr_bps_now",
                "session_id",
                "trend_regime_id",
                "vol_regime_id",
                "spread_bps",
            ],
            "lineage_v1": "AS_OF_AT_BAR_T",
            "already_in_scaffold_v1": False,
            "must_be_added_in_v1": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        },
        {
            "category_v1": "ENTRY_CONTEXT_SNAPSHOT",
            "must_include_fields_v1": [
                "p_long_entry",
                "p_hat_entry",
                "uncertainty_entry",
                "entropy_entry",
                "margin_entry",
                "side_v1",
            ],
            "lineage_v1": "AS_OF_AT_TRADE_OPEN",
            "already_in_scaffold_v1": False,
            "must_be_added_in_v1": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        },
    ],
    "leakage_audit_required_v1": True,
}

FORBIDDEN_STATE_FIELDS_V1 = [
    "bar_count_v1",
    "is_terminal_v1",
    "exit_reason",
    "exit_reason_v1",
    "exit_price_used",
    "exit_bid",
    "exit_ask",
    "exit_spread_bps",
    "exit_price_used_v1",
    "post_exit_mfe_bps",
    "post_exit_mfe_12b_bps",
    "post_exit_mae_12b_bps",
    "post_exit_mfe_bps_replay_end_obs",
    "early_exit_regret",
    "early_exit_regret_replay_end_obs",
    "early_exit_regret_threshold_bps",
    "early_exit_regret_threshold_bps_replay_end_obs",
    "candidate_uid_v1",
    "trade_uid_v1",
    "trade_id_v1",
    "trade_id",
    "close_ts_utc_meta_v1",
    "duration_bars",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "future_bar_high_v1",
    "future_bar_low_v1",
    "future_bar_close_v1",
]

NO_SHORTCUT_AXIOMS_V1 = [
    {
        "axiom_v1": "STATE_AT_BAR_T_DEPENDS_ONLY_ON_BARS_LE_T_MINUS_1_OR_T_OPEN_HIGH_LOW",
        "interpretation_v1": (
            "State features at decision-bar t may use bar t's open and "
            "running stats from bars 0..t-1 plus bar t's open/high/low/close "
            "if these are observable when the decision is made. They must "
            "not use bars > t."
        ),
    },
    {
        "axiom_v1": "REWARD_IS_TERMINAL_ONLY_NEVER_IN_STATE",
        "interpretation_v1": (
            "Reward variants are computed at episode terminal only. The "
            "computed reward value is never added back to state at any bar."
        ),
    },
    {
        "axiom_v1": "EPISODE_LENGTH_IS_NOT_OBSERVABLE_AT_DECISION_TIME",
        "interpretation_v1": (
            "bar_count_v1 (total trade length) is forbidden in state because "
            "it leaks the realized exit. Only bars_held (= bar_index_v1) is "
            "allowed."
        ),
    },
    {
        "axiom_v1": "EXIT_REASON_AND_EXIT_PRICE_NEVER_IN_STATE",
        "interpretation_v1": (
            "exit_reason, exit_price_used, exit_bid, exit_ask are realized-"
            "outcome fields. They must never appear in state."
        ),
    },
    {
        "axiom_v1": "POST_EXIT_FIELDS_NEVER_IN_STATE",
        "interpretation_v1": (
            "Any post_exit_* or *_replay_end_obs field is computed after "
            "trade close and must never appear in state at any bar."
        ),
    },
    {
        "axiom_v1": "ROW_IDENTITY_NEVER_IN_STATE",
        "interpretation_v1": (
            "candidate_uid_v1, trade_uid_v1, trade_id are row identifiers; "
            "they must never appear in state."
        ),
    },
    {
        "axiom_v1": "AGGREGATE_OUTCOME_FIELDS_NEVER_IN_STATE",
        "interpretation_v1": (
            "Trade-level aggregates pnl_bps, mfe_bps, mae_bps, duration_bars "
            "are realized outcomes; only bar-running counterparts may appear "
            "in state."
        ),
    },
]

ACTION_SUPPORT_REQUIREMENT_V1 = {
    "current_dataset_v1": {
        "logged_hold_count_v1": 167536,
        "logged_exit_now_count_v1": 1724,
        "hold_to_exit_now_ratio_v1": 167536 / 1724,
        "interpretation_v1": (
            "Logged actions reflect 'trade still open' (HOLD) versus 'realized "
            "exit bar' (EXIT_NOW). Intermediate bars have no logged "
            "counterfactual EXIT_NOW outcome. This is not a true behavior "
            "policy with action choice at every bar."
        ),
    },
    "training_blocked_until_v1": [
        {
            "blocker_v1": "COUNTERFACTUAL_EXIT_NOW_AUGMENTATION",
            "must_be_resolved_in_v1": "EXIT_ACTION_SUPPORT_AUGMENT_V1",
            "augmentation_method_v1": (
                "For every non-terminal bar t, synthesize an EXIT_NOW sample "
                "with reward = exit-PNL(close at bar t). The augmented "
                "dataset has both HOLD and EXIT_NOW samples for every bar, "
                "with HOLD's reward being 0 immediate plus the bellman backup "
                "of next-bar value, and EXIT_NOW's reward being the locked "
                "terminal reward variant evaluated at bar t's close."
            ),
            "behavior_policy_propensity_v1": (
                "Logged HOLD has propensity 1 at non-terminal, 0 at "
                "realized-exit-bar. Logged EXIT_NOW has propensity 1 only "
                "at realized-exit-bar. Synthetic EXIT_NOW counterfactuals "
                "have no propensity (off-policy generation)."
            ),
        }
    ],
    "before_augmentation_iql_training_forbidden_v1": True,
}

PRE_TRAIN_DEPENDENCY_GRAPH_V1 = [
    {
        "gate_id_v1": "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1",
        "gate_role_v1": "MDP_LOCK_THIS_GATE",
        "must_pass_before_v1": ["EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1"],
        "produces_v1": "MDP/reward/transition/discount/forbidden-state lock",
    },
    {
        "gate_id_v1": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        "gate_role_v1": "STATE_FEATURES_LOCK",
        "must_pass_before_v1": ["EXIT_ACTION_SUPPORT_AUGMENT_V1"],
        "produces_v1": "Per-bar state vector contract incorporating CTX36 subset, no-shortcut audit",
    },
    {
        "gate_id_v1": "EXIT_ACTION_SUPPORT_AUGMENT_V1",
        "gate_role_v1": "ACTION_AUGMENTATION",
        "must_pass_before_v1": ["EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1"],
        "produces_v1": "Counterfactual EXIT_NOW samples per bar, behavior-policy propensity logging",
    },
    {
        "gate_id_v1": "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1",
        "gate_role_v1": "SPLIT_AND_AUDIT",
        "must_pass_before_v1": ["EXIT_OFF_POLICY_EVAL_HARNESS_V1"],
        "produces_v1": "Per-trade or per-week split, intra-trade-leakage audit, time-leakage audit",
    },
    {
        "gate_id_v1": "EXIT_OFF_POLICY_EVAL_HARNESS_V1",
        "gate_role_v1": "EVAL_HARNESS",
        "must_pass_before_v1": ["EXIT_PER_BAR_SANITY_TRAINING_V1"],
        "produces_v1": "Off-policy comparator vs current exit_manager and supervised baseline",
    },
    {
        "gate_id_v1": "EXIT_PER_BAR_SANITY_TRAINING_V1",
        "gate_role_v1": "FIRST_TRAINING_GATE",
        "must_pass_before_v1": ["EXIT_PER_BAR_REWARD_VARIANT_COMPARATOR_V1"],
        "produces_v1": "First contextual one-step IQL sanity on the augmented dataset, research-only",
    },
]

ALLOWED_FINAL_STATUSES = {
    "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_LOCKED_PRE_TRAIN_DEPENDENCIES_ENUMERATED",
    "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_BLOCKED_BY_INPUT_LOCK_MISSING",
    "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
    "HOLD_UNTIL_INPUT_LOCKS_RESOLVED_V1",
}

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def validate_action_set(action_set: dict[str, Any]) -> bool:
    if set(action_set.keys()) != {"HOLD", "EXIT_NOW"}:
        raise RuntimeError("ACTION_SET_MUST_BE_BINARY_HOLD_EXIT_NOW")
    if action_set["HOLD"]["action_id_v1"] != 0 or action_set["EXIT_NOW"]["action_id_v1"] != 1:
        raise RuntimeError("ACTION_IDS_MUST_BE_FIXED_0_AND_1")
    return True


def validate_hold_reward_lock(payload: dict[str, Any]) -> bool:
    if payload["scheme_v1"] != "ZERO_IMMEDIATE_TERMINAL_ONLY":
        raise RuntimeError("HOLD_REWARD_SCHEME_NOT_LOCKED")
    if payload["hold_immediate_reward_v1"] != 0.0:
        raise RuntimeError("HOLD_IMMEDIATE_REWARD_MUST_BE_ZERO")
    return True


def validate_terminal_reward_variants(variants: list[dict[str, Any]]) -> bool:
    required_ids = {
        "REALIZED_PNL_REWARD",
        "MFE_CAPTURE_REWARD",
        "MAE_PENALTY_REWARD",
        "GIVEBACK_PENALTY_REWARD",
        "TRANSPARENT_COMBINED_REWARD",
        "RUNNER_DAMAGE_PENALTY",
    }
    have = {v["reward_id_v1"] for v in variants}
    if required_ids != have:
        raise RuntimeError(
            f"TERMINAL_REWARD_VARIANT_SET_MISMATCH: missing={required_ids - have} extra={have - required_ids}"
        )
    audit_only = [v for v in variants if v["applies_to_action_v1"] == "AUDIT_ONLY_NOT_TRAINING"]
    if len(audit_only) != 1 or audit_only[0]["reward_id_v1"] != "RUNNER_DAMAGE_PENALTY":
        raise RuntimeError("RUNNER_DAMAGE_PENALTY_MUST_BE_AUDIT_ONLY")
    return True


def validate_discount_lock(payload: dict[str, Any]) -> bool:
    g = payload["default_gamma_v1"]
    if not (0.5 <= g < 1.0):
        raise RuntimeError("DEFAULT_GAMMA_OUTSIDE_VALID_RANGE")
    if not all(0.5 <= x < 1.0 for x in payload["sensitivity_range_v1"]):
        raise RuntimeError("SENSITIVITY_GAMMA_VALUE_OUTSIDE_VALID_RANGE")
    return True


def validate_forbidden_state_fields(forbidden: list[str]) -> bool:
    must_be_in = {
        "bar_count_v1",
        "is_terminal_v1",
        "exit_reason",
        "post_exit_mfe_bps",
        "candidate_uid_v1",
        "trade_uid_v1",
        "pnl_bps",
        "mfe_bps",
        "mae_bps",
        "duration_bars",
    }
    missing = must_be_in - set(forbidden)
    if missing:
        raise RuntimeError(f"FORBIDDEN_STATE_FIELDS_MISSING_REQUIRED: {missing}")
    return True


def validate_action_support_requirement(req: dict[str, Any]) -> bool:
    if not req["before_augmentation_iql_training_forbidden_v1"]:
        raise RuntimeError("ACTION_SUPPORT_REQUIREMENT_MUST_BLOCK_TRAINING")
    blockers = req["training_blocked_until_v1"]
    if not any(
        b["blocker_v1"] == "COUNTERFACTUAL_EXIT_NOW_AUGMENTATION" for b in blockers
    ):
        raise RuntimeError("ACTION_SUPPORT_REQUIREMENT_MUST_REQUIRE_AUGMENTATION")
    return True


def validate_dependency_graph(graph: list[dict[str, Any]]) -> bool:
    ids = [g["gate_id_v1"] for g in graph]
    if ids[0] != "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1":
        raise RuntimeError("DEPENDENCY_GRAPH_MUST_START_WITH_THIS_GATE")
    if ids[-1] != "EXIT_PER_BAR_SANITY_TRAINING_V1":
        raise RuntimeError("DEPENDENCY_GRAPH_MUST_END_WITH_FIRST_TRAINING")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_PER_BAR_SCAFFOLD_ROOT, INPUT_REBUILD_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "per_bar_scaffold_summary": INPUT_PER_BAR_SCAFFOLD_ROOT / "summary_v1.json",
        "per_bar_reconstruction_summary": INPUT_PER_BAR_SCAFFOLD_ROOT
        / "PER_BAR_TRAJECTORY_V1"
        / "per_bar_reconstruction_summary_v1.json",
        "rebuild_reward_variants_contract": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "iql_entry_iql_reward_variants_contract_v2.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    return {
        "required_paths": required,
        "per_bar_scaffold_summary": _read_json(required["per_bar_scaffold_summary"]),
        "per_bar_reconstruction_summary": _read_json(
            required["per_bar_reconstruction_summary"]
        ),
        "rebuild_reward_variants_contract": _read_json(
            required["rebuild_reward_variants_contract"]
        ),
    }


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "per_bar_scaffold_root_v1": str(INPUT_PER_BAR_SCAFFOLD_ROOT),
            "rebuild_root_v1": str(INPUT_REBUILD_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _self_consistency_audit() -> dict[str, Any]:
    validate_action_set(ACTION_SET_V1)
    validate_hold_reward_lock(HOLD_REWARD_LOCK_V1)
    validate_terminal_reward_variants(TERMINAL_REWARD_VARIANTS_V1)
    validate_discount_lock(DISCOUNT_LOCK_V1)
    validate_forbidden_state_fields(FORBIDDEN_STATE_FIELDS_V1)
    validate_action_support_requirement(ACTION_SUPPORT_REQUIREMENT_V1)
    validate_dependency_graph(PRE_TRAIN_DEPENDENCY_GRAPH_V1)
    return {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_SELF_CONSISTENCY_AUDIT_V1",
        "status_v1": "PASS",
        "checks_v1": {
            "action_set_binary_v1": True,
            "hold_reward_zero_immediate_v1": True,
            "terminal_reward_variants_complete_v1": True,
            "runner_damage_audit_only_v1": True,
            "discount_in_valid_range_v1": True,
            "forbidden_state_fields_complete_v1": True,
            "action_support_blocks_training_v1": True,
            "dependency_graph_well_ordered_v1": True,
        },
        "research_only_v1": True,
    }


def _build_mdp_contract() -> dict[str, Any]:
    return {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_CONTRACT_V1",
        "scope_v1": "RESEARCH_ONLY_OFFLINE_RL_LOCK",
        "action_set_v1": ACTION_SET_V1,
        "hold_reward_lock_v1": HOLD_REWARD_LOCK_V1,
        "terminal_reward_variants_v1": TERMINAL_REWARD_VARIANTS_V1,
        "episode_definition_v1": EPISODE_DEFINITION_V1,
        "transition_semantics_v1": TRANSITION_SEMANTICS_V1,
        "discount_lock_v1": DISCOUNT_LOCK_V1,
        "state_requirements_v1": STATE_REQUIREMENT_V1,
        "forbidden_state_fields_v1": FORBIDDEN_STATE_FIELDS_V1,
        "no_shortcut_axioms_v1": NO_SHORTCUT_AXIOMS_V1,
        "action_support_requirement_v1": ACTION_SUPPORT_REQUIREMENT_V1,
        "research_only_v1": True,
        "iql_training_now_v1": False,
        "training_allowed_after_dependencies_v1": False,
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    _write_json(artifact_root / "input_manifest_v1.json", _build_input_manifest(inputs, artifact_root))

    mdp_contract = _build_mdp_contract()
    _write_json(artifact_root / "mdp_contract_v1.json", mdp_contract)
    _write_json(artifact_root / "action_contract_v1.json", {"action_set_v1": ACTION_SET_V1})
    _write_json(
        artifact_root / "reward_contract_v1.json",
        {
            "hold_reward_lock_v1": HOLD_REWARD_LOCK_V1,
            "terminal_reward_variants_v1": TERMINAL_REWARD_VARIANTS_V1,
        },
    )
    _write_json(
        artifact_root / "transition_semantics_v1.json",
        {
            "transition_semantics_v1": TRANSITION_SEMANTICS_V1,
            "episode_definition_v1": EPISODE_DEFINITION_V1,
        },
    )
    _write_json(artifact_root / "discount_factor_lock_v1.json", DISCOUNT_LOCK_V1)
    _write_json(artifact_root / "state_schema_requirements_v1.json", STATE_REQUIREMENT_V1)
    _write_json(
        artifact_root / "no_shortcut_axioms_v1.json",
        {"no_shortcut_axioms_v1": NO_SHORTCUT_AXIOMS_V1, "forbidden_state_fields_v1": FORBIDDEN_STATE_FIELDS_V1},
    )
    _write_json(
        artifact_root / "action_support_requirement_v1.json", ACTION_SUPPORT_REQUIREMENT_V1
    )
    _write_json(
        artifact_root / "pre_train_dependency_graph_v1.json",
        {"pre_train_dependency_graph_v1": PRE_TRAIN_DEPENDENCY_GRAPH_V1},
    )

    audit = _self_consistency_audit()
    _write_json(artifact_root / "mdp_self_consistency_audit_v1.json", audit)

    status = "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_LOCKED_PRE_TRAIN_DEPENDENCIES_ENUMERATED"
    next_action = "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1"
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": (
            "MDP and reward semantics are locked for exit-side IQL research. The "
            "next gate (EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1) must define the "
            "explicit per-bar state vector incorporating the required state "
            "categories: TRADE_STATE_RUNNING (already in scaffold), "
            "MARKET_STATE_AT_BAR (must be added via M5 join), and "
            "ENTRY_CONTEXT_SNAPSHOT (must be added via trade-open join). "
            "Training remains forbidden until all five pre-train dependencies pass."
        ),
        "action_set_v1": list(ACTION_SET_V1.keys()),
        "hold_reward_scheme_v1": HOLD_REWARD_LOCK_V1["scheme_v1"],
        "terminal_reward_variant_count_v1": len(TERMINAL_REWARD_VARIANTS_V1),
        "trainable_reward_variant_count_v1": len(
            [
                v
                for v in TERMINAL_REWARD_VARIANTS_V1
                if v["applies_to_action_v1"] != "AUDIT_ONLY_NOT_TRAINING"
            ]
        ),
        "default_discount_v1": DISCOUNT_LOCK_V1["default_gamma_v1"],
        "forbidden_state_field_count_v1": len(FORBIDDEN_STATE_FIELDS_V1),
        "no_shortcut_axiom_count_v1": len(NO_SHORTCUT_AXIOMS_V1),
        "pre_train_gate_count_v1": len(PRE_TRAIN_DEPENDENCY_GRAPH_V1),
        "training_blocked_v1": True,
        "next_pre_train_gate_v1": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        "self_consistency_status_v1": audit["status_v1"],
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "This gate is a research-only contract lock. It does not open "
            "adapter, R6, IQL production/live, freeze, promo, or live, and "
            "does not modify exit_manager or any runtime. Training of any "
            "exit-side IQL/bandit model remains BLOCKED until all five "
            "pre-train dependency gates pass."
        ),
    }
    _write_json(
        artifact_root / "exit_hold_exit_now_mdp_reward_contract_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Exit HOLD/EXIT_NOW MDP and Reward Contract V1",
        "",
        "## Final status",
        f"- `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** until all five pre-train dependency gates pass.",
        "",
        "## Locked design choices",
        f"- Action set: `{list(ACTION_SET_V1.keys())}` (binary, no partial exits, no re-entry)",
        f"- HOLD reward: `{HOLD_REWARD_LOCK_V1['scheme_v1']}` (= {HOLD_REWARD_LOCK_V1['hold_immediate_reward_v1']})",
        f"- Terminal reward variants: {len(TERMINAL_REWARD_VARIANTS_V1)} ({summary['trainable_reward_variant_count_v1']} trainable, 1 audit-only)",
        f"- Default discount γ: {DISCOUNT_LOCK_V1['default_gamma_v1']} per M5 bar",
        f"- Episode = one trade, episode_id = `candidate_uid_v1`, timestep = `bar_index_v1`",
        f"- Forbidden state fields: {len(FORBIDDEN_STATE_FIELDS_V1)} (incl. bar_count_v1, exit_reason, post_exit_*, pnl_bps, mfe_bps, mae_bps)",
        f"- No-shortcut axioms: {len(NO_SHORTCUT_AXIOMS_V1)}",
        "",
        "## Pre-train dependency graph",
    ]
    for g in PRE_TRAIN_DEPENDENCY_GRAPH_V1:
        report_lines.append(
            f"- `{g['gate_id_v1']}` → role `{g['gate_role_v1']}`"
        )
    report_lines.extend([
        "",
        "## Action support requirement",
        f"- Logged HOLD: {ACTION_SUPPORT_REQUIREMENT_V1['current_dataset_v1']['logged_hold_count_v1']}",
        f"- Logged EXIT_NOW: {ACTION_SUPPORT_REQUIREMENT_V1['current_dataset_v1']['logged_exit_now_count_v1']}",
        f"- Hold:exit ratio: {ACTION_SUPPORT_REQUIREMENT_V1['current_dataset_v1']['hold_to_exit_now_ratio_v1']:.0f}:1",
        "- Counterfactual EXIT_NOW augmentation is REQUIRED before training.",
        "",
        "## Recommendation",
        summary["recommendation_v1"],
    ])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root / "exit_hold_exit_now_mdp_reward_contract_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "mdp_contract": str(artifact_root / "mdp_contract_v1.json"),
            "action_contract": str(artifact_root / "action_contract_v1.json"),
            "reward_contract": str(artifact_root / "reward_contract_v1.json"),
            "transition_semantics": str(artifact_root / "transition_semantics_v1.json"),
            "discount_factor_lock": str(artifact_root / "discount_factor_lock_v1.json"),
            "state_schema_requirements": str(
                artifact_root / "state_schema_requirements_v1.json"
            ),
            "no_shortcut_axioms": str(artifact_root / "no_shortcut_axioms_v1.json"),
            "action_support_requirement": str(
                artifact_root / "action_support_requirement_v1.json"
            ),
            "pre_train_dependency_graph": str(
                artifact_root / "pre_train_dependency_graph_v1.json"
            ),
            "self_consistency_audit": str(
                artifact_root / "mdp_self_consistency_audit_v1.json"
            ),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
