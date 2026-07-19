from __future__ import annotations

import pytest

from gx1.contracts.entry_model_native_training_objective_v1 import (
    FIXED_POSITIVE_LOSS_WEIGHTS,
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    active_loss_weight_failures,
    require_training_objective_contract,
    training_objective_contract_metadata,
)


def _weights() -> dict[str, float]:
    return {name: 1.0 for name in REQUIRED_POSITIVE_LOSS_WEIGHTS}


def test_training_objective_contract_is_exact_and_positive() -> None:
    payload = training_objective_contract_metadata(_weights())

    assert payload["fixed_positive_loss_weights"] == FIXED_POSITIVE_LOSS_WEIGHTS
    assert payload["all_advertised_heads_supervised"] is True
    assert require_training_objective_contract(payload, context="TEST") == payload


@pytest.mark.parametrize("name", REQUIRED_POSITIVE_LOSS_WEIGHTS)
def test_training_objective_rejects_every_missing_weight(name: str) -> None:
    weights = _weights()
    del weights[name]

    assert f"{name}=missing" in active_loss_weight_failures(weights)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_training_objective_rejects_nonpositive_or_nonfinite(value: float) -> None:
    weights = _weights()
    weights[REQUIRED_POSITIVE_LOSS_WEIGHTS[0]] = value

    assert active_loss_weight_failures(weights)


def test_training_objective_rejects_extra_weight_and_fixed_weight_drift() -> None:
    weights = _weights()
    weights["UNCONTRACTED_LOSS"] = 1.0
    assert "UNCONTRACTED_LOSS=unexpected" in active_loss_weight_failures(weights)

    payload = training_objective_contract_metadata(_weights())
    payload["fixed_positive_loss_weights"]["position_size"] = 0.0
    with pytest.raises(RuntimeError, match="FIXED_WEIGHTS_INVALID"):
        require_training_objective_contract(payload, context="TEST")
