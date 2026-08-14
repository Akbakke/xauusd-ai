"""Exact learned fitted-Q and genuine auxiliary training objective."""

from __future__ import annotations

from typing import Any, Mapping

from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_objective_contract,
)


SCHEMA_VERSION = "entry_model_native_training_objective_v10"


def _expected_objective() -> dict[str, Any]:
    return {
        "all_advertised_heads_supervised": True,
        "entry_action_q_loss": "masked_raw_bps_mean_squared_error",
        "entry_action_q_target": (
            "stop_gradient_frozen_train_exit_target_model_first_state_value"
        ),
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
    expected = {
        "schema_version": SCHEMA_VERSION,
        **_expected_objective(),
    }
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_MISSING]")
    observed = dict(value)
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


def training_objective_contract_metadata() -> dict[str, Any]:
    return require_training_objective_contract(
        {
            "schema_version": SCHEMA_VERSION,
            **_expected_objective(),
        },
        context="ENTRY_MODEL_NATIVE_EXPORT",
    )

