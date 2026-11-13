"""Current original-loss VJPs and retained last-Adam direction; no run owner.

The Adam vector excludes decoupled decay. It is neither a next optimizer step
nor an attribution of a finite historical output change. Callers own admission,
endpoint identity, numerical execution and model-state/passivity bookends.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.audit.ta_direction_mixing_gradient_v1 import (
    ARMS as MIXING_ARMS, CLOSURE_RELATIVE_L2_LIMIT, MixingGradientClosureError,
    measure_mixing_gradient_partition,
)

EXTRA_ARMS = ('future_all_side', 'future_m5_mean_logit', 'future_non_m5_mean_logit')
ARMS = MIXING_ARMS + EXTRA_ARMS
CALLS_PER_BATCH = 8
FINAL_STEP = 3698
NUMERICAL_SHA256 = '838f7a163565ff7b9c645a67c451a7cfac70045911c963d5cb6ce273dee96648'


class CurrentUpdateDirectionClosureError(RuntimeError):
    def __init__(self, report_json):
        super().__init__('TA_CURRENT_UPDATE_DIRECTION_CLOSURE')
        self.report_json = report_json


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_CURRENT_UPDATE_DIRECTION_' + reason)


def _pairs(parameter_groups, *, torch):
    _need(isinstance(parameter_groups, Mapping) and bool(parameter_groups), 'GROUPS')
    pairs = []
    for group, entries in parameter_groups.items():
        _need(isinstance(group, str) and bool(group) and isinstance(entries, (list, tuple))
              and bool(entries), 'GROUP')
        for entry in entries:
            _need(isinstance(entry, (list, tuple)) and len(entry) == 2, 'PARAMETER_PAIR')
            name, p = entry
            _need(isinstance(name, str) and bool(name) and isinstance(p, torch.nn.Parameter)
                  and p.is_leaf and p.dtype == torch.float32 and p.numel() > 0
                  and bool(torch.isfinite(p.detach()).all()), 'PARAMETER')
            pairs.append((name, p))
    _need(len({n for n, _ in pairs}) == len({id(p) for _, p in pairs}) == len(pairs), 'PARAMETER_ALIASES')
    return pairs


def _l2(v):
    return float(v.square().sum().sqrt().item())


def _geometry(a, b, *, torch):
    dot = float(torch.dot(a, b).item()); denominator = _l2(a) * _l2(b)
    _need(math.isfinite(dot) and math.isfinite(denominator), 'GEOMETRY_NONFINITE')
    return dict(dot=dot, cosine=dot / denominator if denominator else None)


def _pack(values, parameter_groups, *, torch):
    vectors = {}; reports = {}; offset = 0
    for group, entries in parameter_groups.items():
        pieces = []; states = {}; missing_elements = 0
        for (name, p), value in zip(entries, values[offset:offset + len(entries)]):
            if value is None:
                states[name] = None; missing_elements += p.numel()
                pieces.append(torch.zeros(p.numel(), dtype=torch.float64, device='cpu'))
            else:
                _need(value.shape == p.shape and value.dtype == p.dtype and value.device == p.device
                      and not value.requires_grad and bool(torch.isfinite(value).all()), 'DERIVATIVE')
                states[name] = 'nonzero' if torch.count_nonzero(value).item() else 'zero'
                pieces.append(value.detach().reshape(-1).to(device='cpu', dtype=torch.float64).clone())
        vector = torch.cat(pieces); vectors[group] = vector
        reports[group] = dict(l2=_l2(vector), elements=vector.numel(),
            nonzero_elements=int(torch.count_nonzero(vector).item()), missing_elements=missing_elements,
            missing_tensors=sum(v is None for v in states.values()),
            zero_tensors=sum(v == 'zero' for v in states.values()),
            nonzero_tensors=sum(v == 'nonzero' for v in states.values()), tensor_gradient_state=states)
        offset += len(entries)
    return reports, vectors


def measure_current_update_direction(report, parameter_groups, *, counts, torch):
    """Add four calls before the unchanged four-call original-loss partition.

    All-five-TF future-side seeds come from the ORIGINAL scalar loss, including
    masks/reductions. Probes are unmasked all-row mean raw logits. None and exact
    zero derivatives stay distinct in metadata; flat vectors use zero placeholders.
    Owned vectors are FP64 CPU tensors in the supplied parameter/group order.
    No parameter .grad, values, RNG or module mode is changed.
    """
    _need(isinstance(report, Mapping) and isinstance(counts, dict)
          and type(counts.get('autograd_grad_calls')) is int and counts['autograd_grad_calls'] >= 0, 'INPUT')
    _need(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), 'GRAD_CONTEXT')
    loss = report.get('loss')
    raws = tuple(report.get(task + '_activity_side_logits') for task in ('state', 'future'))
    _need(isinstance(loss, torch.Tensor) and loss.ndim == 0 and loss.dtype == torch.float32
          and loss.requires_grad and bool(torch.isfinite(loss)), 'LOSS')
    _need(all(isinstance(raw, torch.Tensor) and raw.ndim == 3 and raw.shape[0] > 0
              and tuple(raw.shape[1:]) == (5, 2) and raw.dtype == torch.float32
              and raw.device == loss.device and raw.requires_grad and bool(torch.isfinite(raw).all())
              for raw in raws) and raws[0].shape == raws[1].shape and raws[0] is not raws[1], 'LOGITS')
    pairs = _pairs(parameter_groups, torch=torch); parameters = tuple(p for _, p in pairs)
    _need(all(p.requires_grad and p.device == loss.device for p in parameters), 'PARAMETER_GRAD_DEVICE')
    start = counts['autograd_grad_calls']; future = raws[1]

    def grad(outputs, inputs, seeds=None):
        counts['autograd_grad_calls'] += 1
        return torch.autograd.grad(outputs, inputs, grad_outputs=seeds, retain_graph=True,
                                   create_graph=False, allow_unused=True)

    seed, = grad(loss, (future,))
    _need(seed is None or (seed.shape == future.shape and seed.dtype == future.dtype
          and seed.device == future.device and not seed.requires_grad and bool(torch.isfinite(seed).all())), 'LOGIT_SEED')
    side_seed = torch.zeros_like(future)
    if seed is not None:
        side_seed[:, :, 1] = seed.detach()[:, :, 1]
    support = dict(gradient_state=None if seed is None else ('nonzero' if torch.count_nonzero(seed).item() else 'zero'),
        full_nonzero_elements=None if seed is None else int(torch.count_nonzero(seed).item()),
        side_nonzero_elements=int(torch.count_nonzero(side_seed).item()),
        side_nonzero_by_tf={tf: int(torch.count_nonzero(side_seed[:, j, 1]).item())
                            for j, tf in enumerate(('M5', 'M15', 'H1', 'H4', 'D1'))},
        activity_seed_nonzero_elements=int(torch.count_nonzero(side_seed[:, :, 0]).item()))
    extra_vectors = {group: {} for group in parameter_groups}; extra_reports = {group: {} for group in parameter_groups}
    # Pack each call immediately; do not retain three extra GPU gradient tuples.
    for arm in EXTRA_ARMS:
        if arm == 'future_all_side':
            values = grad(future, parameters, side_seed)
        else:
            probe = future[:, 0, 1].mean() if arm == 'future_m5_mean_logit' else future[:, 1:, 1].mean()
            values = grad(probe, parameters)
        scalar, vectors = _pack(values, parameter_groups, torch=torch)
        for group in parameter_groups:
            extra_reports[group][arm] = scalar[group]; extra_vectors[group][arm] = vectors[group]
        del values, vectors
    del seed, side_seed
    try:
        mixing, vectors = measure_mixing_gradient_partition(report, parameter_groups, counts=counts, torch=torch)
    except MixingGradientClosureError as exc:
        raise CurrentUpdateDirectionClosureError(dict(
            schema_version='gx1_ta_current_update_direction_v1', autograd_grad_calls=counts['autograd_grad_calls'] - start,
            closure_passed=False, mixing_partition=exc.report_json, future_logit_seed_support=support,
            extra_groups=extra_reports)) from exc
    _need(counts['autograd_grad_calls'] - start == CALLS_PER_BATCH, 'CALL_COUNT')
    groups = {}
    for group in parameter_groups:
        vectors[group].update(extra_vectors[group]); v = vectors[group]
        other = v['total'] - v['future_all_side']
        groups[group] = dict(arms={**mixing['groups'][group]['arms'], **extra_reports[group]},
            future_side_vs_other_loss=_geometry(v['future_all_side'], other, torch=torch),
            observables_vs_loss={probe: {arm: _geometry(v[probe], v[arm], torch=torch)
                for arm in (*MIXING_ARMS, 'future_all_side')} for probe in EXTRA_ARMS[1:]},
            observables_vs_other_loss={probe: _geometry(v[probe], other, torch=torch) for probe in EXTRA_ARMS[1:]})
    return dict(schema_version='gx1_ta_current_update_direction_v1',
        autograd_grad_calls=counts['autograd_grad_calls'] - start, closure_passed=mixing['closure_passed'],
        closure_relative_l2_limit=CLOSURE_RELATIVE_L2_LIMIT, mixing_partition=mixing,
        future_logit_seed_support=support, groups=groups,
        interpretation='Negative observable dot loss-gradient implies an infinitesimal SGD increase; not an Adam or finite-change attribution.'), vectors


def nominal_last_adam_direction(saved_final, parameter_groups, *, numerical_recipe, torch):
    """CPU FP64 last retained Adam gradient component, excluding weight decay.

    No optimizer is created or stepped. The saved moments already include the
    historical clipped-gradient stream. This direction is not a next-step
    prediction, an FP32 update replay, or a finite output-change decomposition.
    Caller must first qualify actual checkpoint/model identities and provenance.
    """
    _need(isinstance(numerical_recipe, Mapping), 'NUMERICAL_RECIPE')
    recipe = dict(numerical_recipe); claimed = recipe.pop('contract_sha256', None)
    digest = campaign.canonical_sha256(recipe)
    _need(claimed == digest == NUMERICAL_SHA256, 'NUMERICAL_RECIPE')
    _need(isinstance(saved_final, Mapping) and saved_final.get('physical_optimizer_updates') == FINAL_STEP
          and saved_final.get('pending_gradients') == 0 and saved_final.get('numerical_recipe') == numerical_recipe, 'FINAL_STATE')
    binding = saved_final.get('parameter_binding'); optimizer = saved_final.get('optimizer_state'); state = saved_final.get('model_state')
    _need(isinstance(binding, Mapping) and isinstance(optimizer, Mapping) and isinstance(state, Mapping)
          and binding.get('numerical_recipe') == numerical_recipe and set(optimizer) == {'state', 'param_groups'}, 'FINAL_BINDING')
    active = binding.get('active_parameter_names'); inactive = binding.get('inactive_parameter_names')
    inventory = binding.get('parameter_inventory'); groups = optimizer['param_groups']; history = optimizer['state']
    _need(isinstance(active, list) and bool(active) and isinstance(inactive, list)
          and all(isinstance(n, str) and bool(n) for n in active + inactive)
          and len(set(active + inactive)) == len(active + inactive)
          and binding.get('optimizer_group_parameter_names') == [active]
          and isinstance(inventory, list) and all(isinstance(v, Mapping) and set(v) == {'name', 'shape', 'dtype'} for v in inventory)
          and len(inventory) == len(active) + len(inactive), 'PARAMETER_NAMES')
    by_name = {v['name']: v for v in inventory}
    _need(set(by_name) == set(active + inactive) and isinstance(groups, list) and len(groups) == 1
          and isinstance(groups[0], Mapping) and isinstance(history, Mapping), 'OPTIMIZER_GROUP')
    kwargs = numerical_recipe['optimizer']['kwargs']; group = groups[0]; ids = group.get('params')
    _need(numerical_recipe['optimizer']['name'] == 'torch.optim.AdamW'
          and set(group) == set(kwargs) | {'params'} and group.get('maximize') is False and group.get('amsgrad') is False
          and all((tuple(group[k]) == tuple(v) if k == 'betas' else type(group[k]) is type(v) and group[k] == v)
                  for k, v in kwargs.items())
          and isinstance(ids, list) and len(ids) == len(active) and all(type(i) is int for i in ids)
          and len(set(ids)) == len(ids) and set(history) == set(ids), 'OPTIMIZER_GROUP')
    moments = {}
    for name, pid in zip(active, ids):
        m = history[pid]; p = state.get(name); meta = by_name[name]
        _need(isinstance(p, torch.Tensor) and p.device.type == 'cpu' and p.dtype == torch.float32
              and list(p.shape) == meta['shape'] and meta['dtype'] == 'torch.float32'
              and bool(torch.isfinite(p).all()), 'STATE_TENSOR:' + name)
        _need(isinstance(m, Mapping) and set(m) == {'step', 'exp_avg', 'exp_avg_sq'}, 'ADAM_HISTORY:' + name)
        step = m['step']
        _need(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.ndim == 0
              and step.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(step))
              and float(step) == FINAL_STEP, 'ADAM_STEP:' + name)
        _need(all(isinstance(m[k], torch.Tensor) and m[k].device.type == 'cpu'
              and m[k].shape == p.shape and m[k].dtype == p.dtype and not m[k].requires_grad
              and bool(torch.isfinite(m[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))
              and bool((m['exp_avg_sq'] >= 0).all()), 'ADAM_MOMENTS:' + name)
        moments[name] = m
    pairs = _pairs(parameter_groups, torch=torch)
    _need(all(name in moments and p.shape == state[name].shape and p.dtype == state[name].dtype
              for name, p in pairs), 'SELECTED_PARAMETERS')
    lr = float(group['lr']); beta1, beta2 = map(float, group['betas']); eps = float(group['eps'])
    vectors = {}; reports = {}
    for name, entries in parameter_groups.items():
        pieces = []
        for parameter_name, _ in entries:
            m = moments[parameter_name]
            first = m['exp_avg'].detach().to(dtype=torch.float64)
            second = m['exp_avg_sq'].detach().to(dtype=torch.float64)
            direction = -lr * (first / (1 - beta1 ** FINAL_STEP)) / ((second / (1 - beta2 ** FINAL_STEP)).sqrt() + eps)
            _need(bool(torch.isfinite(direction).all()), 'ADAM_DIRECTION')
            pieces.append(direction.reshape(-1))
        vector = torch.cat(pieces); vectors[name] = vector
        reports[name] = dict(parameters=len(entries), elements=vector.numel(), l2=_l2(vector),
            nonzero_elements=int(torch.count_nonzero(vector).item()))
    return dict(schema_version='gx1_ta_current_nominal_last_adam_direction_v1', step=FINAL_STEP,
        numerical_recipe_sha256=claimed, optimizer_groups=1, history_tensors_checked=len(active),
        selected_parameter_tensors=len(pairs), groups=reports, decoupled_weight_decay_included=False,
        optimizer_constructions=0, optimizer_steps=0,
        interpretation='Nominal FP64 gradient component from the retained last Adam moments, excluding decay; not a next step or finite historical output change.'), vectors
