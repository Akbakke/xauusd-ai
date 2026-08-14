from __future__ import annotations

import pytest

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
    assert payload["handwritten_direction_distribution_forcing"] is False
    assert payload["handwritten_hierarchical_distribution_forcing"] is False
    assert payload["main_direction_loss"] == "unweighted_cross_entropy"
    assert payload["mtf_direction_loss"] == "unweighted_cross_entropy"
    assert payload["target_units"] == "raw_native_units"
    assert payload["path_quality_heteroscedastic_nll"] is False
    assert payload["joint_log_variance_role"] == (
        "learned_cross_task_precision_only"
    )
    assert payload["distributional_quantiles_define_output_functionals"] is True
    assert payload["offline_rl_expectile_tau_defines_value_functional"] is True
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
        ("handwritten_direction_distribution_forcing", True),
        ("handwritten_hierarchical_distribution_forcing", True),
        ("main_direction_loss", "weighted_cross_entropy"),
        ("mtf_direction_loss", "weighted_cross_entropy"),
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

    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_SURFACE_INVALID"):
        require_training_objective_contract(payload, context="TEST")
