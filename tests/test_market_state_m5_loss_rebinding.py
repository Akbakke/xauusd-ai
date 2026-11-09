"""Loss rebinding preserves a strict zero-step state; no market/model forward."""
import copy

import pytest
import torch

from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _model, _seed,
)

NEW_COMMIT = 'b' * 40


def _case(monkeypatch):
    _seed()
    source = _model(monkeypatch, True)
    source.register_buffer('nonpersistent_witness', torch.tensor([1.25]), persistent=False)
    source.eval()
    source.future_m5_side_head.train()
    with torch.no_grad():
        source.state_model.m5_side_head.weight.add_(0.75)
        source.future_m5_side_head.bias.sub_(0.25)
    old_recipe = fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
    new_recipe = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=True, m5_equal_observation=True)
    source_opt, _ = state.build_fold1_fit_optimizer(source, numerical_recipe=old_recipe)
    _seed()
    saved = _capture(source, source_opt, old_recipe, 0)
    target = _model(monkeypatch, True)
    target.register_buffer('nonpersistent_witness', torch.tensor([-8.0]), persistent=False)
    optimizer, _ = state.build_fold1_fit_optimizer(target, numerical_recipe=old_recipe)
    kwargs = {
        'expected_model_state_sha256': saved['model_state_sha256'],
        'new_numerical_recipe': new_recipe,
        'feature_view_identity': target.state_model.feature_view_identity,
        'parent_order': ORDER, 'source_commit': COMMIT,
        'new_source_commit': NEW_COMMIT, 'device': DEVICE,
    }
    return saved, target, optimizer, kwargs


def _snapshot(model, optimizer):
    _, _, _, _, _, trainer = state._deps()
    return {
        'model': old._tree_sha(model.state_dict()),
        'buffers': old._tree_sha(dict(model.named_buffers())),
        'optimizer': old._tree_sha(optimizer.state_dict()),
        'rng': old._tree_sha(trainer._attended_session_rng_state(device=DEVICE)),
        'modes': {name: module.training for name, module in model.named_modules()},
        'parameters': [(name, id(p), p.requires_grad)
                       for name, p in model.named_parameters(remove_duplicate=False)],
        'buffer_ids': [(name, id(b)) for name, b in model.named_buffers()],
        'gradients': [(name, None if p.grad is None else old._tree_sha(p.grad))
                      for name, p in model.named_parameters()],
    }


def test_rebinding_changes_only_loss_source_and_seals_then_roundtrips(monkeypatch):
    saved, target, optimizer, kwargs = _case(monkeypatch)
    saved_sha = old._tree_sha(saved)
    before = _snapshot(target, optimizer)
    assert before['model'] != old._tree_sha(saved['model_state'])
    rebound, evidence = state.rebind_m5_equal_observation_initial_state(
        saved, target, optimizer, **kwargs)
    after = _snapshot(target, optimizer)
    assert old._tree_sha(saved) == saved_sha
    assert {key for key in saved if old._tree_sha(saved[key]) != old._tree_sha(rebound[key])} == {
        'binding', 'source_commit', 'contract_sha256'}
    assert {key for key in saved['binding']
            if old._tree_sha(saved['binding'][key]) != old._tree_sha(rebound['binding'][key])} == {
        'numerical_recipe', 'contract_sha256'}
    assert after['parameters'] == before['parameters']
    assert after['buffer_ids'] == before['buffer_ids']
    assert after['model'] == old._tree_sha(saved['model_state'])
    assert after['buffers'] == saved['buffer_state_sha256']
    assert after['optimizer'] == saved['optimizer_state_sha256']
    assert after['rng'] == saved['rng_state_sha256']
    assert after['modes'] == saved['module_training']
    assert not optimizer.state
    assert evidence['model_state_sha256'] == kwargs['expected_model_state_sha256']
    assert evidence['parent_order_sha256'] == old._tree_sha(ORDER)
    assert evidence['optimizer_group_parameter_names'] == saved['binding']['optimizer_group_parameter_names']
    assert evidence['cursor'] == evidence['optimizer_updates'] == evidence['skipped_batches'] == 0
    assert evidence['optimizer_steps_performed'] == 0
    assert evidence['parameter_identity_unchanged'] and evidence['buffer_identity_unchanged']

    restored = _model(monkeypatch, True)
    restored.register_buffer('nonpersistent_witness', torch.zeros(1), persistent=False)
    opt, _ = state.build_fold1_fit_optimizer(
        restored, numerical_recipe=kwargs['new_numerical_recipe'])
    result = state.restore_fold1_fit_state(
        rebound, restored, opt, numerical_recipe=kwargs['new_numerical_recipe'],
        feature_view_identity=restored.state_model.feature_view_identity,
        parent_order=ORDER, source_commit=NEW_COMMIT, device=DEVICE)
    assert result == {'cursor': 0, 'optimizer_updates': 0, 'skipped_batches': 0}
    again = state.capture_fold1_fit_state(
        restored, opt, numerical_recipe=kwargs['new_numerical_recipe'],
        feature_view_identity=restored.state_model.feature_view_identity,
        parent_order=ORDER, **result, source_commit=NEW_COMMIT, device=DEVICE)
    assert again['contract_sha256'] == rebound['contract_sha256']

    for value, recipe, commit in (
        (saved, kwargs['new_numerical_recipe'], COMMIT),
        (rebound, saved['binding']['numerical_recipe'], NEW_COMMIT),
    ):
        unchanged = _snapshot(restored, opt)
        with pytest.raises(RuntimeError, match='EXACT_BINDING'):
            state.restore_fold1_fit_state(
                value, restored, opt, numerical_recipe=recipe,
                feature_view_identity=restored.state_model.feature_view_identity,
                parent_order=ORDER, source_commit=commit, device=DEVICE)
        assert _snapshot(restored, opt) == unchanged


@pytest.mark.parametrize('fault, reason', [
    ('expected_model', 'M5_LOSS_REBIND_MODEL_DIGEST'),
    ('old_numerical', 'M5_LOSS_REBIND_OLD_NUMERICAL'),
    ('new_default', 'M5_LOSS_REBIND_NEW_NUMERICAL'),
    ('new_profile', 'M5_LOSS_REBIND_NEW_NUMERICAL'),
    ('new_source', 'M5_LOSS_REBIND_NEW_SOURCE'),
    ('consumed_batch', 'M5_LOSS_REBIND_ZERO_PROGRESS'),
    ('bool_cursor', 'M5_LOSS_REBIND_ZERO_PROGRESS'),
    ('destination_optimizer', 'M5_LOSS_REBIND_EMPTY_OPTIMIZER'),
    ('saved_optimizer', 'M5_LOSS_REBIND_EMPTY_OPTIMIZER'),
    ('old_source', 'EXACT_BINDING'),
    ('source_seal', 'STATE_SEAL'),
    ('saved_model', 'STATE_DIGESTS'),
    ('parent_order', 'PARENT_ORDER'),
    ('default_model', 'M5_LOSS_REBIND_MODE'),
    ('pending_gradient', 'PARTIAL_GRADIENT'),
])
def test_invalid_rebinding_is_rejected_before_any_destination_mutation(
        monkeypatch, fault, reason):
    saved, target, optimizer, kwargs = _case(monkeypatch)
    if fault == 'expected_model':
        kwargs['expected_model_state_sha256'] = '0' * 64
    elif fault == 'old_numerical':
        saved['binding']['numerical_recipe'] = copy.deepcopy(kwargs['new_numerical_recipe'])
    elif fault == 'new_default':
        kwargs['new_numerical_recipe'] = copy.deepcopy(saved['binding']['numerical_recipe'])
    elif fault == 'new_profile':
        kwargs['new_numerical_recipe']['m5_side_weighting_profile']['fit_rows'] -= 1
    elif fault == 'new_source':
        kwargs['new_source_commit'] = 'invalid'
    elif fault == 'consumed_batch':
        saved['cursor'], saved['skipped_batches'] = 16, 1
    elif fault == 'bool_cursor':
        saved['cursor'] = False
    elif fault == 'destination_optimizer':
        optimizer.state[next(iter(optimizer.param_groups[0]['params']))] = {'step': torch.tensor(1.)}
    elif fault == 'saved_optimizer':
        saved['optimizer_state']['state'][0] = {'step': torch.tensor(1.)}
    elif fault == 'old_source':
        kwargs['source_commit'] = 'c' * 40
    elif fault == 'source_seal':
        saved['contract_sha256'] = '0' * 64
    elif fault == 'saved_model':
        saved['model_state']['future_m5_side_head.bias'].add_(0.125)
        saved['contract_sha256'] = old._tree_sha({
            key: value for key, value in saved.items() if key != 'contract_sha256'})
    elif fault == 'parent_order':
        kwargs['parent_order'] = ORDER.flip(0)
    elif fault == 'default_model':
        target = _model(monkeypatch)
        optimizer, _ = state.build_fold1_fit_optimizer(
            target, numerical_recipe=fit.ta_sf_numerical_recipe())
        kwargs['feature_view_identity'] = target.state_model.feature_view_identity
    elif fault == 'pending_gradient':
        target.future_m5_side_head.bias.grad = torch.ones_like(target.future_m5_side_head.bias)

    before = _snapshot(target, optimizer)
    saved_sha = old._tree_sha(saved)
    with pytest.raises(RuntimeError, match=reason):
        state.rebind_m5_equal_observation_initial_state(saved, target, optimizer, **kwargs)
    assert _snapshot(target, optimizer) == before
    assert old._tree_sha(saved) == saved_sha
