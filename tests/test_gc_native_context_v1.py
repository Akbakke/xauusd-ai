"""Mask/normalization algebra only; real-data native wiring has its own gate."""
import numpy as np
import pytest
import torch
from gx1.features import gc_native_context_v1 as gc
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    fit_surface_normalization, apply_surface_normalization, invert_surface_normalization,
    require_surface_normalization,
)


def test_observed_population_excludes_missing_storage_and_roundtrips():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    raw = np.array([[0, 0], [10, 1], [20, 1], [30, 1], [0, 0]], np.float32)
    contract = fit_surface_normalization(raw, surface='ctx_cont', field_names=names)
    require_surface_normalization(contract, surface='ctx_cont', field_names=names)
    assert contract['center'][0] == 20
    assert contract['scale'][0] == 10
    assert contract['binary_mask'] == [0, 1]
    normalized = apply_surface_normalization(raw, contract)
    assert normalized[0, 0] == normalized[-1, 0] == 0
    np.testing.assert_allclose(invert_surface_normalization(normalized, contract), raw, atol=1e-6)
    tensor = torch.tensor(raw, requires_grad=True)
    preliminary = torch.asinh((tensor - torch.tensor(contract['center'])) / torch.tensor(contract['scale']))
    result = gc.mask_normalized_torch(tensor, preliminary, names)
    result[:, 0].sum().backward()
    assert tensor.grad[0, 0] == tensor.grad[-1, 0] == 0
    assert tensor.grad[1, 0] != 0


def test_nonzero_value_behind_absent_mask_is_rejected():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    raw = np.array([[5, 0], [10, 1], [20, 1]], np.float32)
    with pytest.raises(RuntimeError, match='MASKED_STORAGE'):
        fit_surface_normalization(raw, surface='ctx_cont', field_names=names)
    with pytest.raises(RuntimeError, match='MASKED_STORAGE'):
        gc.mask_normalized_torch(torch.tensor(raw), torch.tensor(raw), names)


def test_missing_physical_train_observations_fail_closed():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    with pytest.raises(RuntimeError, match='NO_PHYSICAL_TRAIN_OBSERVATIONS'):
        fit_surface_normalization(np.zeros((4, 2), np.float32), surface='ctx_cont', field_names=names)


def test_every_value_has_exactly_one_availability_and_specialist_owner():
    assert len(gc.VALUE_FIELDS) == len(gc.OBSERVED_FIELDS)
    assert set(gc.SPECIALIST_OWNERS) == set(gc.CONTEXT_FIELDS)
    assert len(gc.masking_pairs(gc.CONTEXT_FIELDS)) == len(gc.VALUE_FIELDS)
    assert len(gc.masking_pairs(['ctx_cont.' + n for n in gc.CONTEXT_FIELDS])) == len(gc.VALUE_FIELDS)
    with pytest.raises(RuntimeError, match='OBSERVATION_FIELD_MISSING'):
        gc.masking_pairs([gc.VALUE_FIELDS[0]])
