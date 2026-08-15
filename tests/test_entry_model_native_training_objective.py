from __future__ import annotations

import pytest

from gx1.contracts.entry_fitted_q_v1 import entry_fitted_q_contract
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_objective_contract,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    require_training_objective_contract,
    training_objective_contract_metadata,
)


def test_training_objective_contract_is_exact_and_learned() -> None:
    payload = training_objective_contract_metadata()

    assert payload["all_advertised_heads_supervised"] is True
    assert payload["fixed_relative_task_weights"] is False
    assert payload["handwritten_rank_losses"] is False
    assert payload["handwritten_composite_weights"] is False
    assert payload["handwritten_gate_regularization"] is False
    assert payload["fixed_target_normalization_scales"] is False
    assert payload["handwritten_distribution_forcing"] is False
    assert payload["target_units"] == "raw_native_units"
    # Fitted-Q replaced every classification/probability objective: the only
    # Entry authority is the raw-bps Q argmax, and its target is the frozen
    # TRAIN Exit teacher rather than any fixed forward horizon.
    assert payload["classification_or_probability_loss_authority"] is False
    assert payload["fixed_horizon_target_authority"] is False
    assert payload["entry_action_q_loss"] == "masked_raw_bps_mean_squared_error"
    assert payload["unified_exit_action_q_loss"] == (
        "masked_raw_bps_mean_squared_error"
    )
    assert payload["entry_action_q_target"] == (
        "stop_gradient_frozen_train_exit_target_model_first_state_value"
    )
    # The objective may not invent its own decision rule: it must restate the
    # fitted-Q owner's decision verbatim.
    assert payload["entry_decision_authority"] == (
        entry_fitted_q_contract()["decision"]
    )
    assert payload["joint_task_names"] == list(JOINT_TASK_NAMES)
    assert payload["joint_task_weighting"] == joint_task_weighting_objective_contract()
    assert require_training_objective_contract(payload, context="TEST") == payload


@pytest.mark.parametrize(
    "field,bad",
    [
        ("fixed_relative_task_weights", True),
        ("handwritten_rank_losses", True),
        ("handwritten_composite_weights", True),
        ("handwritten_gate_regularization", True),
        ("fixed_target_normalization_scales", True),
        ("handwritten_distribution_forcing", True),
        ("classification_or_probability_loss_authority", True),
        ("fixed_horizon_target_authority", True),
        ("entry_action_q_loss", "weighted_cross_entropy"),
        ("unified_exit_action_q_loss", "weighted_cross_entropy"),
        ("entry_action_q_target", "pathwise_forward_outcome"),
        ("entry_decision_authority", "expected_utility_threshold"),
        ("target_units", "scaled_bps"),
    ],
)
def test_training_objective_rejects_contract_drift(field: str, bad: object) -> None:
    payload = training_objective_contract_metadata()
    payload[field] = bad

    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        require_training_objective_contract(payload, context="TEST")


def test_training_objective_rejects_extra_surface() -> None:
    payload = training_objective_contract_metadata()
    payload["fixed_positive_loss_weights"] = {"direction": 1.0}

    with pytest.raises(
        RuntimeError,
        match=r"TRAINING_OBJECTIVE_CONTRACT_INVALID.*"
        r"unexpected=\['fixed_positive_loss_weights'\]",
    ):
        require_training_objective_contract(payload, context="TEST")
