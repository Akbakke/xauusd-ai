"""Small real-Adam private continuation checks; no market data or native fit."""
import copy

import pytest
import torch

from gx1.audit import run_market_state_ta_m5_paired_batch_order_v1 as paired
from gx1.audit import run_market_state_ta_m5_repeated_pass_v1 as worker
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from tests.test_market_state_m5_loss_rebinding import _snapshot
from tests.test_market_state_ta_dedicated_m5_binding_v1 import COMMIT, DEVICE, ORDER, _model, _seed
from tests.test_market_state_native_F_private_state_v1 import _update


def _ref(name):
    return {'path': '/synthetic/' + name, 'sha256': '9' * 64}


def _seal(value):
    value['evidence_sha256'] = old._tree_sha(
        {k: v for k, v in value.items() if k != 'evidence_sha256'})


def _case(monkeypatch):
    _seed()
    source = _model(monkeypatch, True)
    source.register_buffer('mutable_witness', torch.tensor([1.25]))
    source.register_buffer('nonpersistent_witness', torch.tensor([2.5]), persistent=False)
    numerical = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    source_opt, _ = state.build_fold1_fit_optimizer(source, numerical_recipe=numerical)
    for step in range(2):
        _update(source, source_opt, step)
    source.eval()
    source.future_m5_side_head.train()
    indices = [(i * 1847) // 255 for i in range(256)]
    arms = {'CONTROL': indices, 'INTERLEAVED': [
        indices[r * 16 + c] for c in range(16) for r in range(16)]}
    orders, _ = paired.build_execution_orders(ORDER.numpy(), arms)
    order = torch.from_numpy(orders['CONTROL'])
    # Synthetic boundary metadata is 256; genuine toy Adam states retain step=2.
    # This exercises the legal partial-state history, not a claimed native replay.
    captured = state.capture_fold1_fit_state(
        source, source_opt, numerical_recipe=numerical,
        feature_view_identity=source.state_model.feature_view_identity,
        parent_order=order, cursor=4096, optimizer_updates=256, skipped_batches=0,
        source_commit=COMMIT, device=DEVICE)
    batch_sha = worker.scope.campaign.canonical_sha256(indices)
    order_sha = old._tree_sha(order)
    saved = {key: copy.deepcopy(captured[key]) for key in (
        'model_state', 'buffer_state', 'optimizer_state', 'rng_state', 'module_training',
        'model_state_sha256', 'buffer_state_sha256', 'optimizer_state_sha256', 'rng_state_sha256')}
    saved.update(
        schema_version=paired.CAPSULE_SCHEMA, arm='CONTROL', recipe=_ref('paired_recipe.json'),
        source_commit=COMMIT, original_checkpoint=_ref('original_initial.pt'),
        canonical_resume_authority=False, test_data_used=False, parent_order=order.clone(),
        parent_cursor=4096, physical_optimizer_updates=256, pending_gradients=0,
        parameter_binding=copy.deepcopy(captured['binding']),
        original_numerical_recipe_reference=numerical,
        method_state=dict(arm_batch_indices=indices, batch_order_sha256=batch_sha,
            full_order_sha256=order_sha, original_fit_batch_starts=[16 * b for b in indices],
            unselected_suffix_consumed=False, loss_and_optimizer_unchanged=True))
    _seal(saved)
    target = _model(monkeypatch, True)
    target.register_buffer('mutable_witness', torch.tensor([-8.]))
    target.register_buffer('nonpersistent_witness', torch.tensor([-9.]), persistent=False)
    optimizer, _ = state.build_fold1_fit_optimizer(target, numerical_recipe=numerical)
    origin = dict(
        paired_result=dict(source_commit=COMMIT, arm_progress={'CONTROL': dict(
            parent_order_sha256=order_sha, batch_order_sha256=batch_sha)}),
        paired_recipe=dict(arm_batch_indices=arms,
            input_refs={'original_initial_checkpoint': _ref('original_initial.pt')}),
        measurements={name: {'evaluated_model_state_sha256': saved['model_state_sha256']}
                      for name in ('CONTROL_TRAIN', 'CONTROL_LATER')})
    kwargs = dict(recipe={'numerical_recipe': numerical, 'input_refs': {
        'paired_recipe': _ref('paired_recipe.json'), 'control_evidence': _ref('control.pt')}},
        origin=origin, parent_order=order,
        feature_view_identity=target.state_model.feature_view_identity, device=DEVICE)
    return saved, target, optimizer, kwargs


def test_restores_actual_moments_order_modes_rng_and_objects(monkeypatch):
    saved, model, optimizer, kwargs = _case(monkeypatch)
    before = _snapshot(model, optimizer)
    saved_sha = old._tree_sha(saved)
    assert not optimizer.state
    assert any(bool(v['exp_avg'].ne(0).any()) for v in saved['optimizer_state']['state'].values())
    proof = worker.restore_control_evidence(saved, model, optimizer, **kwargs)
    after = _snapshot(model, optimizer)
    assert old._tree_sha(saved) == saved_sha
    assert after['parameters'] == before['parameters']
    assert after['buffer_ids'] == before['buffer_ids']
    assert after['model'] == old._tree_sha(saved['model_state'])
    assert after['buffers'] == saved['buffer_state_sha256']
    assert after['optimizer'] == saved['optimizer_state_sha256']
    assert after['rng'] == saved['rng_state_sha256']
    assert after['modes'] == saved['module_training']
    assert all(grad is None for _, grad in after['gradients'])
    active_names = saved['parameter_binding']['optimizer_group_parameter_names'][0]
    actual_names = {id(p): name for name, p in state._parameters(model)[0]}
    assert [actual_names[id(p)] for p in optimizer.param_groups[0]['params']] == active_names
    assert {float(v['step']) for v in optimizer.state.values()} == {2.}
    assert proof['optimizer_parameter_states'] == len(saved['optimizer_state']['state']) > 0
    assert proof['all_numerical_state_exact'] and proof['parameter_objects_preserved']
    assert proof['canonical_resume_authority'] is False


@pytest.mark.parametrize('fault, reason', [
    ('seal', 'CONTROL_EVIDENCE_SEAL'),
    ('binding', 'CONTROL_BINDING'),
    ('model_digest', 'CONTROL_DIGEST:model_state'),
    ('negative_moment', 'ADAMW_SECOND_MOMENT'),
    ('parameter_order', 'OPTIMIZER_STATE_GROUPS'),
    ('immutable_buffer', 'CONTROL_IMMUTABLE_BUFFER'),
    ('rng_dtype', 'RNG_TENSOR'),
    ('mode_type', 'CONTROL_MODULE_MODES'),
    ('bound_model', 'CONTROL_MODEL_IDENTITY'),
])
def test_rejects_corruption_before_any_destination_mutation(monkeypatch, fault, reason):
    saved, model, optimizer, kwargs = _case(monkeypatch)
    if fault == 'seal':
        saved['evidence_sha256'] = '0' * 64
    elif fault == 'binding':
        saved['parameter_binding']['optimizer_group_parameter_names'][0].reverse()
    elif fault == 'model_digest':
        saved['model_state']['state_model.m5_side_head.weight'].add_(1.)
    elif fault == 'negative_moment':
        next(iter(saved['optimizer_state']['state'].values()))['exp_avg_sq'].fill_(-1.)
    elif fault == 'parameter_order':
        saved['optimizer_state']['param_groups'][0]['params'].reverse()
    elif fault == 'immutable_buffer':
        for key in ('buffer_state', 'model_state'):
            saved[key]['native_F_objective_sha256'][0] ^= 1
    elif fault == 'rng_dtype':
        saved['rng_state']['torch_cpu'] = saved['rng_state']['torch_cpu'].to(torch.float32)
    elif fault == 'mode_type':
        saved['module_training'][''] = 1
    else:
        kwargs['origin']['measurements']['CONTROL_LATER']['evaluated_model_state_sha256'] = '0' * 64
    if fault != 'seal':
        _seal(saved)
    before, saved_sha = _snapshot(model, optimizer), old._tree_sha(saved)
    with pytest.raises(RuntimeError, match=reason):
        worker.restore_control_evidence(saved, model, optimizer, **kwargs)
    assert _snapshot(model, optimizer) == before
    assert old._tree_sha(saved) == saved_sha


def _measurements():
    values = {}
    for panel in ('TRAIN', 'LATER'):
        for epoch, score in (('INITIAL', .35), ('CONTROL', .30), ('E2', .20), ('E3', .33)):
            values[epoch + '_' + panel] = dict(rows=16, observation_chain_sha256='a' * 64,
                scores={'model': {task: {'M5': {'conditional_side': {'count': 8}}}
                                  for task in worker.TASKS}},
                means={'model': {task: {'M5': {'conditional_side': {'brier': score, 'log': score}}}
                                 for task in worker.TASKS},
                       'prior': {task: {'M5': {'conditional_side': {'brier': .4, 'log': .4}}}
                                 for task in worker.TASKS}})
    return values


def test_e3_remains_primary_when_only_e2_passes_and_inputs_stay_unchanged():
    values = _measurements()
    before = copy.deepcopy(values)
    e2 = worker.compare_fixed_pass(values, 'E2', baseline_score_atol=1e-6)
    e3 = worker.compare_fixed_pass(values, 'E3', baseline_score_atol=1e-6)
    assert e2['decision'] == 'PASS_ALL_FOUR_CELLS' and e2['primary'] is False
    assert e3['decision'] == 'FAIL_COMPARISON_NO_PROMOTION' and e3['primary'] is True
    assert e2['checkpoint_selection'] is e3['checkpoint_selection'] is False
    assert e3['candidate'] == 'E3' and e3['CONTROL'] == 'cached_E1'
    assert all('E3' in cell and 'INTERLEAVED' not in cell for cell in e3['cells'].values())
    assert values == before


@pytest.mark.parametrize('fault, reason', [
    ('observations', 'MATCHED_OBSERVATIONS'),
    ('prior', 'FROZEN_PRIOR'),
    ('pass', 'FIXED_MEASUREMENT_PASS'),
])
def test_fixed_comparison_rejects_changed_population_prior_or_selected_pass(fault, reason):
    values, epoch = _measurements(), 'E3'
    if fault == 'observations':
        values['E3_TRAIN']['observation_chain_sha256'] = 'b' * 64
    elif fault == 'prior':
        values['E3_LATER']['means']['prior']['future']['M5']['conditional_side']['brier'] = .41
    else:
        epoch = 'E4'
    with pytest.raises(RuntimeError, match=reason):
        worker.compare_fixed_pass(values, epoch, baseline_score_atol=1e-6)
