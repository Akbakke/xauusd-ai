"""Exact learned fitted-Q and genuine auxiliary training objective.

Entry supervision depends on the declared training phase. The base joint
phase supervises Entry from the frozen TRAIN Exit teacher's first-state value;
the Entry-only phase (operator decision 2026-10-09, owner
``entry_observed_market_v1``) supervises Entry from observed executable market
outcomes without any Exit model or Exit gradients. The trainer persists the
exact phase with the bundle so metadata never advertises a target that was
not trained (rule 2a: one origin per decision-influencing value).
"""

from __future__ import annotations

from typing import Any, Mapping

from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    DYNAMIC_M5_JOINT_TASK_NAMES,
    joint_task_weighting_objective_contract,
    joint_task_names_for_phase,
)


SCHEMA_VERSION = "entry_model_native_training_objective_v14"
JOINT_FITTED_Q_TRAINING_PHASE = "joint_fitted_q"
ENTRY_ONLY_TRAINING_PHASE = "entry_only"
ENTRY_ONLY_M5_PATH_TRAINING_PHASE = "entry_only_observed_m5_path"
ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE = "entry_only_dynamic_m5_path"
# Operator decision 12.10.2026: Entry-only supervision from the observed market outcome at
# 20 trading days (owner ``entry_observed_market_horizons_v1``); 1 and 5 days observed only.
ENTRY_ONLY_HORIZON_TRAINING_PHASE = "entry_only_observed_horizon_20d"
ENTRY_TRAINING_PHASES = (JOINT_FITTED_Q_TRAINING_PHASE, ENTRY_ONLY_TRAINING_PHASE, ENTRY_ONLY_HORIZON_TRAINING_PHASE, ENTRY_ONLY_M5_PATH_TRAINING_PHASE, ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
ENTRY_ONLY_TRAINING_PHASES = (ENTRY_ONLY_TRAINING_PHASE, ENTRY_ONLY_HORIZON_TRAINING_PHASE, ENTRY_ONLY_M5_PATH_TRAINING_PHASE, ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
ENTRY_ACTION_Q_TARGET_BY_PHASE = {
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE: "no_legacy_entry_action_q_emitted_dynamic_component_supervision_only",
    ENTRY_ONLY_M5_PATH_TRAINING_PHASE: "signed_executable_mfe_minus_conservative_mae_before_first_executable_mfe_bps",
    JOINT_FITTED_Q_TRAINING_PHASE: (
        "stop_gradient_frozen_train_exit_target_model_first_state_value"
    ),
    ENTRY_ONLY_TRAINING_PHASE: (
        "observed_executable_terminal_cash_return_net_of_declared_costs_"
        "at_frozen_train_horizon_without_exit_model"
    ),
    ENTRY_ONLY_HORIZON_TRAINING_PHASE: (
        "observed_executable_terminal_cash_return_net_of_declared_costs_"
        "at_20_trading_days_on_the_session_clock_without_exit_model"
    ),
}
EXIT_GRADIENTS_ACTIVE_BY_PHASE = {
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE: False,
    ENTRY_ONLY_M5_PATH_TRAINING_PHASE: False,
    JOINT_FITTED_Q_TRAINING_PHASE: True,
    ENTRY_ONLY_TRAINING_PHASE: False,
    ENTRY_ONLY_HORIZON_TRAINING_PHASE: False,
}



def active_joint_task_names(entry_training_phase: str) -> tuple[str, ...]:
    phase = require_entry_training_phase(entry_training_phase, context="ENTRY_ACTIVE_TASKS")
    if phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        return DYNAMIC_M5_JOINT_TASK_NAMES
    if phase == ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
        return ("entry_action_q", "forecast_return_bps")
    if phase in ENTRY_ONLY_TRAINING_PHASES:
        return tuple(name for name in JOINT_TASK_NAMES if name != "unified_exit_action")
    return JOINT_TASK_NAMES


def require_entry_training_phase(value: Any, *, context: str) -> str:
    if value not in ENTRY_TRAINING_PHASES:
        raise RuntimeError(
            f"[{context}_ENTRY_TRAINING_PHASE_INVALID] got={value!r} "
            f"expected one of {ENTRY_TRAINING_PHASES}"
        )
    return str(value)


def require_native_serving_training_phase(value: Any, *, context: str) -> str:
    """Reject unadmitted dynamic candidates; all remaining serving guards apply."""
    phase = require_entry_training_phase(value, context=context)
    if phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        raise RuntimeError(f"[{context}_DYNAMIC_ENTRY_CANDIDATE_NATIVE_AUTHORITY_NOT_ADMITTED]")
    return phase


def _expected_objective(entry_training_phase: str) -> dict[str, Any]:
    phase = require_entry_training_phase(
        entry_training_phase, context="ENTRY_MODEL_NATIVE_OBJECTIVE"
    )
    expected = {
        "all_advertised_heads_supervised": True,
        "entry_training_phase": phase,
        "entry_action_q_loss": "masked_raw_bps_mean_squared_error",
        "entry_action_q_target": ENTRY_ACTION_Q_TARGET_BY_PHASE[phase],
        "exit_gradients_active": EXIT_GRADIENTS_ACTIVE_BY_PHASE[phase],
        "unified_exit_action_q_loss": "masked_raw_bps_mean_squared_error",
        "target_units": "raw_native_units",
        "entry_decision_authority": (
            "unique_argmax(entry_action_q_bps_over_valid_actions)"
        ),
        "classification_or_probability_loss_authority": False,
        "fixed_horizon_target_authority": False,
        "handwritten_distribution_forcing": False,
        "fixed_relative_task_weights": False,
        "handwritten_rank_losses": False,
        "handwritten_composite_weights": False,
        "handwritten_gate_regularization": False,
        "fixed_target_normalization_scales": False,
        "joint_task_names": list(joint_task_names_for_phase(phase)),
        "joint_task_weighting": joint_task_weighting_objective_contract(entry_training_phase=phase),
    }
    if phase == ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
        expected["all_advertised_heads_supervised"] = False
        expected["active_joint_task_names"] = list(active_joint_task_names(phase))
        expected["inactive_legacy_auxiliary_heads"] = ["position_size", "dip", "timing", "tail_risk", "vol_forecast", "side_mae", "trendline_event"]
        expected["entry_primary_target_contract"] = entry_observed_m5_path_contract()
        expected["entry_action_q_value_semantics"] = "expected_price_path_utility_not_realizable_cash_return"
        expected["long_horizon_forecasts"] = "independently_masked_learning_support_without_entry_authority"
    if phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
        from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_ENDPOINT_KINDS
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_loss_scale_contract
        expected.update({
            "all_advertised_heads_supervised": True,
            "active_joint_task_names": list(active_joint_task_names(phase)),
            "entry_action_q_loss": "inactive_no_legacy_bps_authority",
            "unified_exit_action_q_loss": "inactive_no_exit_targets_or_gradients",
            "entry_decision_authority": "none_dynamic_quality_bps_candidate_not_admitted",
            "entry_primary_target_contract": entry_observed_dynamic_m5_path_contract(),
            "dynamic_component_losses": {
                name: "independently_masked_TRAIN_target_RMS_scaled_squared_error"
                for name in DYNAMIC_M5_JOINT_TASK_NAMES
            },
            "dynamic_loss_unit_contract": dynamic_m5_loss_scale_contract(),
            "train_fitted_loss_scales": True,
            "raw_bps_component_diagnostics_preserved": True,
            "dynamic_path_endpoint_kinds": list(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS),
            "dynamic_path_loss_reduction": "all_observed_cells_of_both_endpoint_kinds_per_task",
            "dynamic_candidate_uses_endpoint_kind": "terminal_open_inclusive",
            "dynamic_candidate_has_supervised_target": False,
            "dynamic_candidate_quality_contract": entry_observed_dynamic_m5_path_contract()["candidate_quality_contract"],
            "dynamic_candidate_is_unconditional_expected_quality": False,
            "dynamic_candidate_is_cash": False,
            "dynamic_candidate_native_authority_admitted": False,
            "inactive_legacy_auxiliary_heads": ["entry_action_q", "position_size", "dip", "timing", "tail_risk", "vol_forecast", "side_mae", "trendline_event"],
            "long_horizon_forecasts": "independently_masked_learning_support_without_entry_authority",
        })
    return expected


def require_training_objective_contract(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_MISSING]")
    observed = dict(value)
    phase = observed.get("entry_training_phase")
    if phase not in ENTRY_TRAINING_PHASES:
        raise RuntimeError(
            f"[{context}_TRAINING_OBJECTIVE_CONTRACT_INVALID] "
            f"entry_training_phase={phase!r} expected one of {ENTRY_TRAINING_PHASES}"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        **_expected_objective(str(phase)),
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected).intersection(observed)
            if observed[key] != expected[key]
        )
        raise RuntimeError(
            f"[{context}_TRAINING_OBJECTIVE_CONTRACT_INVALID] "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    return expected


def training_objective_contract_metadata(
    *, entry_training_phase: str = JOINT_FITTED_Q_TRAINING_PHASE,
) -> dict[str, Any]:
    """Return the exact objective for one declared phase.

    The keyword default is the base joint design, which every pre-phase bundle
    was trained under; the trainer always passes the phase it actually runs.
    """

    return require_training_objective_contract(
        {
            "schema_version": SCHEMA_VERSION,
            **_expected_objective(entry_training_phase),
        },
        context="ENTRY_MODEL_NATIVE_EXPORT",
    )
