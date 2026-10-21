"""Synthetic algebra/mechanics for the adopted fixed-grid query conditioning.

These tests call the actual constructor/decoder with the existing small-model
fixture. They qualify neither genuine data parity nor learned Entry quality.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    DYNAMIC_M1_PREFIX_ENDPOINT_KINDS,
    DYNAMIC_M1_PREFIX_HORIZON_MINUTES,
    DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
)
from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as owner
from tests.test_entry_v10_ctx_model_shapes import _make_dynamic_m5_model


@contextmanager
def _legacy_query_coordinates():
    """Independent raw-coordinate reference using the same constructor/RNG draws."""
    with patch.object(owner, "DYNAMIC_M5_QUERY_CENTER_MINUTES", 0.0), \
            patch.object(owner, "DYNAMIC_M5_QUERY_SCALE_MINUTES", float(DYNAMIC_M1_PREFIX_HORIZON_MINUTES[-1])):
        yield


@pytest.fixture(scope="module")
def initial_pair():
    previous_rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(158)
        with _legacy_query_coordinates():
            legacy = _make_dynamic_m5_model(dropout=0.0).eval()
        legacy_rng = torch.get_rng_state().clone()
        torch.manual_seed(158)
        standardized = _make_dynamic_m5_model(dropout=0.0).eval()
        standardized_rng = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(previous_rng)
    return legacy, standardized, legacy_rng, standardized_rng


def _bits(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _decode_pair(initial_pair, hidden, *, chunk=None):
    legacy, standardized, _, _ = initial_pair
    with _legacy_query_coordinates():
        old = legacy._decode_dynamic_m5_curves(hidden, query_chunk_size=chunk)
    new = standardized._decode_dynamic_m5_curves(hidden, query_chunk_size=chunk)
    return old, new


def test_declared_grid_has_analytic_center_unit_population_variance_and_no_rng_draws():
    grid = torch.tensor(DYNAMIC_M1_PREFIX_HORIZON_MINUTES, dtype=torch.float64)
    assert owner.DYNAMIC_M5_QUERY_CENTER_MINUTES == float(grid.mean()) == 240.5
    assert owner.DYNAMIC_M5_QUERY_SCALE_MINUTES == math.sqrt((len(grid) ** 2 - 1) / 12.0)
    rng = torch.get_rng_state().clone()
    normalized = owner.normalize_dynamic_m5_query_minutes(grid)
    assert torch.equal(torch.get_rng_state(), rng)
    epsilon = torch.finfo(grid.dtype).eps
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.0, dtype=grid.dtype), rtol=0, atol=epsilon)
    torch.testing.assert_close(normalized.square().mean(), torch.tensor(1.0, dtype=grid.dtype), rtol=0, atol=epsilon)
    assert torch.equal(normalized, -normalized.flip(0))
    assert normalized[0] < 0 < normalized[-1]


def test_fresh_initialization_preserves_rng_state_keys_and_every_other_parameter_bit(initial_pair):
    legacy, standardized, legacy_rng, standardized_rng = initial_pair
    assert torch.equal(legacy_rng, standardized_rng)
    before, after = legacy.state_dict(), standardized.state_dict()
    assert before.keys() == after.keys()
    weight_key = "dynamic_m5_query_projection.0.weight"
    bias_key = "dynamic_m5_query_projection.0.bias"
    for name in before:
        if name not in (weight_key, bias_key):
            assert torch.equal(_bits(before[name]), _bits(after[name])), name
    original_weight = before[weight_key]
    expected_weight = original_weight.clone()
    expected_weight[:, -2] *= owner.DYNAMIC_M5_QUERY_SCALE_MINUTES / DYNAMIC_M1_PREFIX_HORIZON_MINUTES[-1]
    expected_bias = before[bias_key] + original_weight[:, -2] * (
        owner.DYNAMIC_M5_QUERY_CENTER_MINUTES / DYNAMIC_M1_PREFIX_HORIZON_MINUTES[-1])
    assert torch.equal(_bits(after[weight_key]), _bits(expected_weight))
    assert torch.equal(_bits(after[bias_key]), _bits(expected_bias))
    assert torch.equal(_bits(after[weight_key][:, :-2]), _bits(original_weight[:, :-2]))
    assert torch.equal(_bits(after[weight_key][:, -1]), _bits(original_weight[:, -1]))


@pytest.mark.parametrize("chunk", [None, 17])
def test_actual_decoder_matches_raw_initial_function_components_candidates_and_ties(initial_pair, chunk):
    _, model, _, _ = initial_pair
    hidden = torch.linspace(-0.5, 0.5, 2 * model.cfg.d_model).reshape(2, model.cfg.d_model)
    with torch.no_grad():
        old, new = _decode_pair(initial_pair, hidden, chunk=chunk)
    assert old.keys() == new.keys()
    for name, expected in old.items():
        if expected.is_floating_point():
            # Reuse the declared genuine component parity tolerance, in bps.
            torch.testing.assert_close(new[name], expected, rtol=0, atol=1e-4)
        else:
            assert torch.equal(new[name], expected), name
    old_q, new_q = (output["dynamic_entry_action_quality_bps_candidate"] for output in (old, new))
    assert torch.equal(old_q == old_q.max(dim=1, keepdim=True).values,
                       new_q == new_q.max(dim=1, keepdim=True).values)


def test_actual_decoder_uses_same_standardized_coordinate_for_both_endpoints_and_chunks(initial_pair):
    _, model, _, _ = initial_pair
    inputs = []
    hook = model.dynamic_m5_query_projection.register_forward_pre_hook(
        lambda _module, arguments: inputs.append(arguments[0].detach().clone()))
    try:
        with torch.no_grad():
            model._decode_dynamic_m5_curves(torch.zeros(1, model.cfg.d_model), query_chunk_size=17)
    finally:
        hook.remove()
    chunks_per_endpoint = math.ceil(len(DYNAMIC_M1_PREFIX_HORIZON_MINUTES) / 17)
    assert len(inputs) == len(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS) * chunks_per_endpoint
    expected = owner.normalize_dynamic_m5_query_minutes(torch.tensor(DYNAMIC_M1_PREFIX_HORIZON_MINUTES, dtype=torch.float32))
    for code, _kind in enumerate(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS):
        actual = torch.cat(inputs[code * chunks_per_endpoint:(code + 1) * chunks_per_endpoint], dim=1)
        assert torch.equal(actual[0, :, -2], expected)
        assert torch.equal(actual[0, :, -1], torch.full_like(expected, code))


def test_analytic_flat_and_exact_three_action_horizon_ties_are_preserved(initial_pair):
    legacy, standardized, _, _ = initial_pair
    heads = ("head_dynamic_terminal_net_markout", "head_dynamic_pre_mfe_mae_upper")
    saved = [{name: deepcopy(getattr(model, name).state_dict()) for name in heads}
             for model in (legacy, standardized)]
    try:
        with torch.no_grad():
            for model in (legacy, standardized):
                model.head_dynamic_terminal_net_markout.weight.zero_()
                model.head_dynamic_terminal_net_markout.bias.fill_(float(torch.nn.functional.softplus(torch.tensor(0.0))))
                model.head_dynamic_pre_mfe_mae_upper.weight.zero_()
                model.head_dynamic_pre_mfe_mae_upper.bias.zero_()
            old, new = _decode_pair(initial_pair, torch.zeros(1, standardized.cfg.d_model), chunk=17)
        for output in (old, new):
            assert bool((output["dynamic_entry_action_quality_bps_candidate"] == 0).all())
            assert bool((output["dynamic_entry_quality_bps_candidate"] == 0).all())
            assert bool(output["dynamic_entry_quality_best_horizon_mask"].all())
            assert bool((output["dynamic_terminal_net_markout_bps"][..., 2] == 0).all())
        assert torch.equal(old["dynamic_entry_quality_best_horizon_mask"], new["dynamic_entry_quality_best_horizon_mask"])
    finally:
        for model, state in zip((legacy, standardized), saved):
            for name in heads:
                getattr(model, name).load_state_dict(state[name], strict=True)


def test_query_chain_rule_gives_same_bps_per_minute_at_function_matched_start(initial_pair):
    legacy, standardized, _, _ = initial_pair
    observed = []
    for model, is_legacy in ((legacy, True), (standardized, False)):
        inputs = []
        hook = model.dynamic_m5_query_projection.register_forward_pre_hook(
            lambda _module, arguments: inputs.append(arguments[0]))
        hidden = torch.linspace(-0.5, 0.5, 2 * model.cfg.d_model).reshape(2, model.cfg.d_model).requires_grad_()
        try:
            if is_legacy:
                with _legacy_query_coordinates():
                    output = model._decode_dynamic_m5_curves(hidden)
            else:
                output = model._decode_dynamic_m5_curves(hidden)
        finally:
            hook.remove()
        gradient = torch.autograd.grad(output["dynamic_terminal_net_markout_bps"][..., 0].sum(),
                                       tuple(inputs), allow_unused=True)
        endpoint = DYNAMIC_M1_PREFIX_ENDPOINT_KINDS.index(DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT)
        assert gradient[endpoint] is not None and gradient[1 - endpoint] is None
        coordinate_scale = (DYNAMIC_M1_PREFIX_HORIZON_MINUTES[-1] if is_legacy
                            else owner.DYNAMIC_M5_QUERY_SCALE_MINUTES)
        observed.append(gradient[endpoint][..., -2] / coordinate_scale)
    torch.testing.assert_close(observed[1], observed[0])
    minutes = torch.tensor(DYNAMIC_M1_PREFIX_HORIZON_MINUTES, dtype=torch.float64, requires_grad=True)
    normalized = owner.normalize_dynamic_m5_query_minutes(minutes)
    derivative = torch.autograd.grad(normalized.sum(), minutes)[0]
    assert torch.equal(derivative, torch.full_like(minutes, 1 / owner.DYNAMIC_M5_QUERY_SCALE_MINUTES))
    assert all(parameter.grad is None for model in (legacy, standardized) for parameter in model.parameters())
