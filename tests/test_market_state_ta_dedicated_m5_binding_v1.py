"""Synthetic identity, weights-only warmstart and restart; no market fit."""
import copy

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
from tests.test_market_state_native_F_private_state_v1 import _model as gc_stub, _seed, _update

DEVICE = torch.device('cpu')
ORDER = torch.arange(29574, dtype=torch.int64)
COMMIT = 'a' * 40


def _model(monkeypatch, dedicated=False):
    backbone = gc_stub(monkeypatch).backbone
    profile = compile_full_ta_entry_execution_profile()
    backbone.entry_execution_profile = {k: profile[k] for k in ('name', 'contract_sha256')}
    backbone.gc_mtf_history = None
    model = NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        backbone, dedicated_m5_side=dedicated))
    model.state_model.feature_view_identity = {
        'contract_sha256': ('3' if dedicated else '1') * 64,
        'prefix_normalization': {'contract_sha256': '2' * 64},
        'research_objective_sha256': model.state_model.research_objective['contract_sha256']}
    return model


def _case(monkeypatch):
    _seed()
    model = _model(monkeypatch, True)
    recipe = fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
    optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    _seed()
    return model, optimizer, recipe, binding


def _capture(model, optimizer, recipe, updates):
    return state.capture_fold1_fit_state(
        model, optimizer, numerical_recipe=recipe,
        feature_view_identity=model.state_model.feature_view_identity,
        parent_order=ORDER, cursor=16 * updates, optimizer_updates=updates,
        skipped_batches=0, source_commit=COMMIT, device=DEVICE)


def test_warmstart_copies_every_legacy_tensor_and_only_clones_side_rows(monkeypatch):
    source = _model(monkeypatch)
    with torch.no_grad():
        source.state_model.query_head.weight.add_(0.5)
        source.future_head.bias.sub_(0.25)
    target = _model(monkeypatch, True)
    before_source = digest(source.state_dict())
    rng = torch.random.get_rng_state().clone()
    result = state.copy_legacy_weights_to_dedicated_m5(
        source, target, expected_legacy_model_sha256=before_source)
    assert digest(source.state_dict()) == before_source
    assert torch.equal(rng, torch.random.get_rng_state())
    assert result['added_parameters'] == 2 * (source.state_model._hidden_width + 1)
    assert result['optimizer_updates'] == result['cursor'] == 0
    assert result['fresh_actual_initial_measurement_required']
    for name, value in source.state_dict().items():
        if name in {'state_model.query_objective_sha256', 'native_F_objective_sha256'}:
            assert not torch.equal(value, target.state_dict()[name])
        else:
            assert torch.equal(value, target.state_dict()[name]), name
    for name, original in result['copied_side_rows'].items():
        assert torch.equal(target.state_dict()[name], source.state_dict()[original][1:2])
    optimizer, binding = state.build_fold1_fit_optimizer(target,
        numerical_recipe=fit.ta_sf_numerical_recipe(dedicated_m5_side=True))
    assert not optimizer.state
    assert set(result['copied_side_rows']) <= set(binding['active_parameter_names'])


@pytest.mark.parametrize('fault', ['source_digest', 'normalization', 'mode'])
def test_bad_warmstart_rejected_before_destination_mutation(monkeypatch, fault):
    source, target = _model(monkeypatch), _model(monkeypatch, True)
    expected = digest(source.state_dict())
    if fault == 'source_digest':
        expected = '0' * 64
    elif fault == 'normalization':
        target.state_model.feature_view_identity['prefix_normalization'] = {'changed': True}
    else:
        target = _model(monkeypatch)
    before = digest(target.state_dict())
    with pytest.raises(RuntimeError, match='M5_WARMSTART'):
        state.copy_legacy_weights_to_dedicated_m5(
            source, target, expected_legacy_model_sha256=expected)
    assert digest(target.state_dict()) == before


def test_new_head_optimizer_state_resumes_bit_exactly(monkeypatch):
    reference, opt, recipe, binding = _case(monkeypatch)
    assert {'state_model.m5_side_head.weight', 'future_m5_side_head.weight'} <= set(
        binding['active_parameter_names'])
    for step in range(2):
        _update(reference, opt, step)
    expected = _capture(reference, opt, recipe, 2)
    interrupted, opt, _, _ = _case(monkeypatch)
    _update(interrupted, opt, 0)
    saved = _capture(interrupted, opt, recipe, 1)
    resumed, opt, _, _ = _case(monkeypatch)
    restored = state.restore_fold1_fit_state(
        saved, resumed, opt, numerical_recipe=recipe,
        feature_view_identity=resumed.state_model.feature_view_identity,
        parent_order=ORDER, source_commit=COMMIT, device=DEVICE)
    assert restored['cursor'] == 16 and restored['optimizer_updates'] == 1
    _update(resumed, opt, 1)
    actual = _capture(resumed, opt, recipe, 2)
    for key in ('model_state_sha256', 'buffer_state_sha256',
                'optimizer_state_sha256', 'rng_state_sha256'):
        assert actual[key] == expected[key], key


def test_legacy_numerical_identity_cannot_silently_train_new_heads(monkeypatch):
    model = _model(monkeypatch, True)
    with pytest.raises(RuntimeError, match='TA_SF_NUMERICAL_RECIPE'):
        state.build_fold1_fit_optimizer(model, numerical_recipe=fit.ta_sf_numerical_recipe())


def test_prior_rebinding_preserves_all_frozen_counts_and_logits():
    value = objective._seal({
        'schema_version': 'gx1_native_S_F_frozen_endpoint_priors_v1',
        'fold_id': 1, 'native_query_k': list(objective.NATIVE_K),
        'loss_contract_sha256': objective.contract(ta_only=True)['contract_sha256'],
        'tasks': {task: {'raw_logits': np.arange(10).reshape(5, 2).tolist(),
                         'counts': {'synthetic': 37}} for task in ('state', 'future')}})
    original = copy.deepcopy(value)
    rebound = objective.rebind_endpoint_priors_for_dedicated_m5(value)
    assert value == original and rebound['tasks'] == value['tasks']
    assert {k: v for k, v in rebound.items()
            if k not in ('contract_sha256', 'loss_contract_sha256')} == {
        k: v for k, v in value.items() if k not in ('contract_sha256', 'loss_contract_sha256')}
    with pytest.raises(ValueError, match='PRIOR_BINDING'):
        objective.require_endpoint_priors(rebound, ta_only=True)
    objective.require_endpoint_priors(rebound, ta_only=True, dedicated_m5_side=True)
