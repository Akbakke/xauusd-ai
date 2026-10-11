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
    joint_task_weighting_objective_contract,
)


SCHEMA_VERSION = "entry_model_native_training_objective_v11"
JOINT_FITTED_Q_TRAINING_PHASE = "joint_fitted_q"
ENTRY_ONLY_TRAINING_PHASE = "entry_only"
ENTRY_TRAINING_PHASES = (JOINT_FITTED_Q_TRAINING_PHASE, ENTRY_ONLY_TRAINING_PHASE)
ENTRY_ACTION_Q_TARGET_BY_PHASE = {
    JOINT_FITTED_Q_TRAINING_PHASE: (
        "stop_gradient_frozen_train_exit_target_model_first_state_value"
    ),
    ENTRY_ONLY_TRAINING_PHASE: (
        "observed_executable_terminal_cash_return_net_of_declared_costs_"
        "at_frozen_train_horizon_without_exit_model"
    ),
}
EXIT_GRADIENTS_ACTIVE_BY_PHASE = {
    JOINT_FITTED_Q_TRAINING_PHASE: True,
    ENTRY_ONLY_TRAINING_PHASE: False,
}


def require_entry_training_phase(value: Any, *, context: str) -> str:
    if value not in ENTRY_TRAINING_PHASES:
        raise RuntimeError(
            f"[{context}_ENTRY_TRAINING_PHASE_INVALID] got={value!r} "
            f"expected one of {ENTRY_TRAINING_PHASES}"
        )
    return str(value)


def _expected_objective(entry_training_phase: str) -> dict[str, Any]:
    phase = require_entry_training_phase(
        entry_training_phase, context="ENTRY_MODEL_NATIVE_OBJECTIVE"
    )
    return {
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
        "joint_task_names": list(JOINT_TASK_NAMES),
        "joint_task_weighting": joint_task_weighting_objective_contract(),
    }


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
