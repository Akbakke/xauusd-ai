from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F

from gx1.audit.ta_current_update_direction_v1 import (
    ARMS, CALLS_PER_BATCH, FINAL_STEP, CurrentUpdateDirectionClosureError,
    measure_current_update_direction, nominal_last_adam_direction,
)


def graph():
    """Small synthetic shared graph; no GX1 model or market inputs."""
    module = torch.nn.Linear(3, 4)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12).reshape(4, 3) / 31 - .2)
        module.bias.copy_(torch.arange(4) / 17 - .1)
    module.register_parameter('absent', torch.nn.Parameter(torch.tensor([2., 3.])))
    module.register_parameter('zero', torch.nn.Parameter(torch.tensor([7.])))
    module.register_buffer('input', torch.arange(12).reshape(4, 3) / 13 - .4)
    module.eval()
    hidden = module(module.input) + 0 * module.zero
    ws = torch.arange(40).reshape(4, 10) / 43 - .2
    wf = torch.arange(40).flip(0).reshape(4, 10) / 47 - .3
    state = (hidden @ ws).reshape(4, 5, 2)
    future = (hidden @ wf).reshape(4, 5, 2)
    targets = (torch.arange(40).reshape(4, 5, 2) % 3 == 0).float()
    sw = torch.arange(40).reshape(4, 5, 2) / 83
    fw = torch.arange(40).flip(0).reshape(4, 5, 2) / 71
    sw[::2, 2, :] = 0; fw[1::2, 3, :] = 0
    sw[:, 0, 1] = torch.tensor([1., 0., 1., 0.]) * (1799 / 10010 / 5)
    fw[:, 0, 1] = torch.tensor([0., 1., 1., 1.]) * (1844 / 11909 / 5)
    sl = F.binary_cross_entropy_with_logits(state, targets, reduction='none') * sw
    fl = F.binary_cross_entropy_with_logits(future, 1 - targets, reduction='none') * fw
    parts = dict(state_m5_side=sl[:, 0, 1].sum(), future_m5_side=fl[:, 0, 1].sum(),
        rest=sl[:, :, 0].sum() + sl[:, 1:, 1].sum() + fl[:, :, 0].sum() + fl[:, 1:, 1].sum(),
        future_all_side=fl[:, :, 1].sum())
    report = dict(loss=sl.sum() + fl.sum(), state_activity_side_logits=state, future_activity_side_logits=future)
    groups = {'projection': [('weight', module.weight), ('bias', module.bias)],
              'unused': [('absent', module.absent), ('zero', module.zero)]}
    return module, report, groups, parts, wf


def flat(values, pairs):
    return torch.cat([torch.zeros(p.numel(), dtype=torch.float64) if value is None
                      else value.detach().reshape(-1).double().clone()
                      for value, (_, p) in zip(values, pairs)])


def test_eight_calls_original_side_partition_analytic_probes_and_passivity(monkeypatch):
    module, report, groups, parts, wf = graph()
    pairs = [entry for entries in groups.values() for entry in entries]
    params = tuple(p for _, p in pairs)
    expected = {arm: torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                for arm, loss in {**parts, 'total': report['loss']}.items()}
    for arm, cols in [('future_m5_mean_logit', [1]), ('future_non_m5_mean_logit', [3, 5, 7, 9])]:
        hidden_derivative = wf[:, cols].mean(dim=1)
        expected[arm] = ((hidden_derivative[:, None] @ module.input.mean(dim=0)[None, :]),
                         hidden_derivative, None, torch.zeros(1))
    before = {name: value.clone() for name, value in module.state_dict().items()}
    modes = {name: child.training for name, child in module.named_modules()}
    flags = {name: (id(p), p.requires_grad) for name, p in pairs}
    for _, p in pairs:
        p.grad = torch.full_like(p, 2.25)
    grads = [(id(p.grad), p.grad.clone()) for _, p in pairs]
    rng = torch.get_rng_state().clone(); calls = []; original = torch.autograd.grad
    def recorded(*args, **kwargs):
        calls.append(kwargs['retain_graph'])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, 'grad', recorded)
    counts = {'autograd_grad_calls': 3, 'unrelated': 9}
    result, vectors = measure_current_update_direction(report, groups, counts=counts, torch=torch)
    assert counts == {'autograd_grad_calls': 11, 'unrelated': 9}
    assert result['autograd_grad_calls'] == CALLS_PER_BATCH == 8 and calls == [True] * 7 + [False]
    assert result['closure_passed'] and result['mixing_partition']['autograd_grad_calls'] == 4
    offset = 0
    for group, entries in groups.items():
        assert tuple(vectors[group]) == ARMS
        for arm in ARMS:
            reference = flat(expected[arm][offset:offset + len(entries)], entries)
            value = vectors[group][arm]
            assert value.dtype == torch.float64 and value.device.type == 'cpu'
            assert not value.requires_grad and value.grad_fn is None
            torch.testing.assert_close(value, reference, rtol=3e-6, atol=3e-8)
        offset += len(entries)
    assert all(n > 0 for n in result['future_logit_seed_support']['side_nonzero_by_tf'].values())
    assert result['future_logit_seed_support']['activity_seed_nonzero_elements'] == 0
    assert not torch.equal(vectors['projection']['future_all_side'], vectors['projection']['future_m5_side'])
    assert {name: child.training for name, child in module.named_modules()} == modes
    assert {name: (id(p), p.requires_grad) for name, p in pairs} == flags
    assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())
    assert all(id(p.grad) == identity and torch.equal(p.grad, value) for (_, p), (identity, value) in zip(pairs, grads))
    assert torch.equal(rng, torch.get_rng_state())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('future_loss', ['unused', 'zero'])
def test_missing_future_seed_and_exact_zero_keep_distinct_metadata(future_loss):
    _, report, groups, _, _ = graph()
    report['loss'] = report['state_activity_side_logits'].square().sum()
    if future_loss == 'zero':
        report['loss'] = report['loss'] + report['future_activity_side_logits'].sum() * 0
    result, vectors = measure_current_update_direction(report, groups, counts={'autograd_grad_calls': 0}, torch=torch)
    assert result['future_logit_seed_support']['gradient_state'] is None if future_loss == 'unused' else result['future_logit_seed_support']['gradient_state'] == 'zero'
    for group in groups:
        assert torch.count_nonzero(vectors[group]['future_all_side']).item() == 0
    for arm in ARMS:
        states = result['groups']['unused']['arms'][arm]['tensor_gradient_state']
        assert states == {'absent': None, 'zero': 'zero'}
    assert torch.count_nonzero(vectors['projection']['future_m5_mean_logit']).item() > 0


def test_unaccounted_direct_loss_fails_closure_with_eight_call_scalar_receipt():
    _, report, groups, _, _ = graph()
    report['loss'] = report['loss'] + .3 * groups['projection'][0][1].sum()
    counts = {'autograd_grad_calls': 0}
    with pytest.raises(CurrentUpdateDirectionClosureError) as caught:
        measure_current_update_direction(report, groups, counts=counts, torch=torch)
    result = caught.value.report_json
    assert counts['autograd_grad_calls'] == result['autograd_grad_calls'] == 8
    assert result['closure_passed'] is False and result['mixing_partition']['closure_passed'] is False
    assert set(result['extra_groups']['projection']) == set(ARMS[4:])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('fault,reason', [('shape', 'LOGITS'), ('alias', 'PARAMETER_ALIASES'),
                                         ('counter', 'INPUT'), ('frozen', 'PARAMETER_GRAD_DEVICE')])
def test_invalid_gradient_boundary_rejected_before_autograd(fault, reason):
    _, report, groups, _, _ = graph(); counts = {'autograd_grad_calls': 0}
    if fault == 'shape': report['future_activity_side_logits'] = report['future_activity_side_logits'][:, :4]
    elif fault == 'alias': groups['unused'].append(('alias', groups['projection'][0][1]))
    elif fault == 'counter': counts['autograd_grad_calls'] = False
    else: groups['projection'][0][1].requires_grad_(False)
    with pytest.raises(RuntimeError, match='TA_CURRENT_UPDATE_DIRECTION_' + reason):
        measure_current_update_direction(report, groups, counts=counts, torch=torch)
    assert counts['autograd_grad_calls'] == 0


def adam_fixture():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    recipe = ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_tf_activity=True, m5_equal_observation=True)
    p = torch.nn.Parameter(torch.tensor([[.25, -.4], [.7, -.2]], dtype=torch.float32))
    q = torch.nn.Parameter(torch.tensor([.13, -.12], dtype=torch.float32))
    kwargs = dict(recipe['optimizer']['kwargs']); kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.AdamW([p, q], **kwargs)
    for index, parameter in enumerate((p, q)):
        optimizer.state[parameter] = dict(step=torch.tensor(float(FINAL_STEP - 1)),
            exp_avg=torch.full_like(parameter, .07 * (index + 1)),
            exp_avg_sq=torch.full_like(parameter, .015 * (index + 1)))
        parameter.grad = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) / 11 - .15
    before = [parameter.detach().clone() for parameter in (p, q)]
    optimizer.step()
    after = [parameter.detach().clone() for parameter in (p, q)]
    names = ['projection.weight', 'projection.bias']; pairs = list(zip(names, (p, q)))
    state = dict(physical_optimizer_updates=FINAL_STEP, pending_gradients=0, numerical_recipe=recipe,
        parameter_binding=dict(numerical_recipe=copy.deepcopy(recipe), active_parameter_names=names,
            inactive_parameter_names=[], optimizer_group_parameter_names=[names],
            parameter_inventory=[dict(name=n, shape=list(t.shape), dtype=str(t.dtype)) for n, t in pairs]),
        model_state={n: t.detach().clone() for n, t in pairs}, optimizer_state=copy.deepcopy(optimizer.state_dict()))
    return state, {'projection': pairs}, recipe, before, after


def test_last_adam_direction_matches_synthetic_adamw_gradient_component_and_is_passive():
    saved, groups, recipe, before, after = adam_fixture()
    original = copy.deepcopy(saved); rng = torch.get_rng_state().clone()
    grad_before = [(id(p.grad), p.grad.clone()) for _, p in groups['projection']]
    result, vectors = nominal_last_adam_direction(saved, groups, numerical_recipe=recipe, torch=torch)
    kwargs = recipe['optimizer']['kwargs']; lr = kwargs['lr']; wd = kwargs['weight_decay']
    actual_gradient_component = torch.cat([(new.double() - old.double() + lr * wd * old.double()).flatten()
                                           for old, new in zip(before, after)])
    # One FP32 torch AdamW step rounds its parameter write; the helper is FP64.
    torch.testing.assert_close(vectors['projection'], actual_gradient_component, rtol=0,
                               atol=2 * torch.finfo(torch.float32).eps * max(float(x.abs().max()) for x in before))
    pieces = []
    beta1, beta2 = kwargs['betas']
    for pid in saved['optimizer_state']['param_groups'][0]['params']:
        m = saved['optimizer_state']['state'][pid]
        pieces.append((-lr * m['exp_avg'].double() / (1 - beta1 ** FINAL_STEP)
            / ((m['exp_avg_sq'].double() / (1 - beta2 ** FINAL_STEP)).sqrt() + kwargs['eps'])).flatten())
    torch.testing.assert_close(vectors['projection'], torch.cat(pieces), rtol=1e-15, atol=1e-18)
    assert result['step'] == FINAL_STEP and result['history_tensors_checked'] == 2
    assert result['decoupled_weight_decay_included'] is False
    assert result['optimizer_constructions'] == result['optimizer_steps'] == 0
    assert vectors['projection'].dtype == torch.float64 and vectors['projection'].device.type == 'cpu'
    assert not vectors['projection'].requires_grad and vectors['projection'].grad_fn is None
    assert torch.equal(rng, torch.get_rng_state())
    for name in original['model_state']:
        assert torch.equal(saved['model_state'][name], original['model_state'][name])
    for pid, moment in original['optimizer_state']['state'].items():
        assert all(torch.equal(saved['optimizer_state']['state'][pid][key], value) for key, value in moment.items())
    assert all(id(p.grad) == identity and torch.equal(p.grad, value)
               for (_, p), (identity, value) in zip(groups['projection'], grad_before))
    json.dumps(result, allow_nan=False)
    vectors['projection'].zero_()
    assert all(torch.equal(saved['optimizer_state']['state'][pid]['exp_avg'], moment['exp_avg'])
               for pid, moment in original['optimizer_state']['state'].items())


@pytest.mark.parametrize('fault,reason', [('recipe', 'NUMERICAL_RECIPE'), ('group', 'OPTIMIZER_GROUP'),
    ('names', 'PARAMETER_NAMES'), ('step', 'ADAM_STEP'), ('missing_history', 'OPTIMIZER_GROUP'),
    ('negative_v', 'ADAM_MOMENTS'), ('nonfinite', 'ADAM_MOMENTS'), ('shape', 'ADAM_MOMENTS'),
    ('maximize', 'OPTIMIZER_GROUP'), ('amsgrad', 'OPTIMIZER_GROUP'), ('selected', 'SELECTED_PARAMETERS')])
def test_last_adam_rejects_wrong_recipe_mapping_or_history(fault, reason):
    saved, groups, recipe, _, _ = adam_fixture()
    history = saved['optimizer_state']['state']; pid = next(iter(history))
    if fault == 'recipe': recipe = dict(recipe, seed=0)
    elif fault == 'group': saved['optimizer_state']['param_groups'].append(copy.deepcopy(saved['optimizer_state']['param_groups'][0]))
    elif fault == 'names': saved['parameter_binding']['optimizer_group_parameter_names'] = [['wrong']]
    elif fault == 'step': history[pid]['step'] -= 1
    elif fault == 'missing_history': del history[pid]
    elif fault == 'negative_v': history[pid]['exp_avg_sq'].fill_(-1)
    elif fault == 'nonfinite': history[pid]['exp_avg'].fill_(float('nan'))
    elif fault == 'shape': history[pid]['exp_avg'] = torch.ones(1)
    elif fault in ('maximize', 'amsgrad'): saved['optimizer_state']['param_groups'][0][fault] = True
    else: groups['projection'][0] = ('not_bound', groups['projection'][0][1])
    with pytest.raises(RuntimeError, match='TA_CURRENT_UPDATE_DIRECTION_' + reason):
        nominal_last_adam_direction(saved, groups, numerical_recipe=recipe, torch=torch)
