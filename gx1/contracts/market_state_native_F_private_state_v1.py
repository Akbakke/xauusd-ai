"""Exact AdamW and restart state for the bounded native S/F research probe.

No data IO, fit admission, scoring, TEST, Entry, Exit, broker or trading role.
The caller owns the fixed TRAIN schedule and atomic checkpoint publication.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import torch

from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts import market_state_time_query_private_state_v1 as private
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel, contract as model_contract
from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

SCHEMA = 'gx1_market_state_native_F_private_state_v1'
BINDING_SCHEMA = 'gx1_market_state_native_F_private_optimizer_binding_v1'
RECIPE_SCHEMA = 'gx1_market_state_native_F_private_numerical_v1'
PARENT_ROWS = tuple(range(2, 34))
BATCH_SIZE = 16
MAX_UPDATES = 2
F_WEIGHT = 1.0
_INACTIVE = tuple('state_model.' + prefix for prefix in private._INACTIVE_PREFIXES)
_IMMUTABLE = frozenset({'native_F_objective_sha256'} |
    {'state_model.' + name for name in private._QUERY_BUFFERS} |
    {'state_model.' + name for name in old.IMMUTABLE_BUFFER_NAMES})
_IMMUTABLE_PREFIXES = tuple('state_model.' + prefix for prefix in old.IMMUTABLE_BUFFER_PREFIXES)
_STATE_KEYS = frozenset({'schema_version', 'binding', 'model_state', 'buffer_state',
    'optimizer_state', 'rng_state', 'module_training', 'parent_order', 'cursor',
    'optimizer_updates', 'model_state_sha256', 'buffer_state_sha256',
    'optimizer_state_sha256', 'rng_state_sha256', 'contract_sha256'})


def _need(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_F_PRIVATE_STATE_' + reason)


def numerical_recipe() -> dict:
    base = dict(native_direction_research_recipe(batch_size=BATCH_SIZE,
                                                   max_updates=MAX_UPDATES))
    base.pop('contract_sha256')
    base['schema_version'] = RECIPE_SCHEMA
    base['research_objective_sha256'] = model_contract()['contract_sha256']
    base['native_F_loss_contract_sha256'] = loss_contract()['contract_sha256']
    base['future_weight'] = F_WEIGHT
    raw = json.dumps(base, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**base, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _recipe(value):
    _need(isinstance(value, Mapping) and dict(value) == numerical_recipe()
          and float(trainer._GRAD_CLIP_NORM) == value['grad_clip_norm'],
          'NUMERICAL_RECIPE')
    return copy.deepcopy(dict(value))


def _parameters(model):
    _need(type(model) is NativeFutureDualModel, 'EXACT_MODEL_TYPE')
    model.state_model._require_binding()
    model.backbone.require_input_normalization_state()
    pairs = list(model.named_parameters(remove_duplicate=False))
    _need(bool(pairs) and len({id(p) for _, p in pairs}) == len(pairs),
          'PARAMETER_ALIASES')
    active = [(name, p) for name, p in pairs if not name.startswith(_INACTIVE)]
    inactive = [(name, p) for name, p in pairs if name.startswith(_INACTIVE)]
    _need(bool(active) and bool(inactive)
          and {'state_model.query_projection.0.weight',
               'state_model.query_head.weight', 'future_head.weight'}
              <= {name for name, _ in active}
          and any(name.startswith('state_model.backbone.') for name, _ in active),
          'PARAMETER_PARTITION')
    _need(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all())
              for _, p in pairs), 'FINITE_FP32_PARAMETERS')
    return pairs, active, inactive


def _binding(model, optimizer, recipe, expected_view):
    pairs, active, inactive = _parameters(model)
    view = private._view(model.state_model, expected_view)
    _need(all(p.requires_grad for _, p in active)
          and all(not p.requires_grad for _, p in inactive), 'REQUIRES_GRAD_PARTITION')
    kwargs = copy.deepcopy(recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    _need(type(optimizer) is torch.optim.AdamW
          and old._tree_sha(optimizer.defaults) == old._tree_sha(kwargs)
          and len(optimizer.param_groups) == 1, 'ADAMW_DEFAULTS')
    group = optimizer.param_groups[0]
    _need(set(group) == set(kwargs) | {'params'}
          and old._tree_sha({key: group[key] for key in kwargs}) == old._tree_sha(kwargs)
          and [id(p) for p in group['params']] == [id(p) for _, p in active],
          'OPTIMIZER_GROUP_ORDER')
    value = {'schema_version': BINDING_SCHEMA,
             'numerical_recipe': copy.deepcopy(recipe),
             'feature_view_identity': view,
             'model_contract_sha256': model_contract()['contract_sha256'],
             'loss_contract_sha256': loss_contract()['contract_sha256'],
             'active_parameter_names': [name for name, _ in active],
             'inactive_parameter_names': [name for name, _ in inactive],
             'parameter_inventory': [{'name': name, 'shape': list(p.shape),
                                      'dtype': str(p.dtype)} for name, p in pairs],
             'optimizer_group_parameter_names': [[name for name, _ in active]],
             'ema_created': False, 'scheduler_created': False,
             'target_model_created': False}
    value['contract_sha256'] = old._tree_sha(value)
    return value


def build_optimizer(model, *, recipe):
    recipe = _recipe(recipe)
    old._no_partial_gradients(model)
    _, active, inactive = _parameters(model)
    for _, p in active:
        p.requires_grad_(True)
    for _, p in inactive:
        p.requires_grad_(False)
    kwargs = copy.deepcopy(recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.AdamW([p for _, p in active], **kwargs)
    return optimizer, _binding(model, optimizer, recipe,
                               model.state_model.feature_view_identity)


def _progress(cursor, updates, parent_order, recipe):
    order = old._order(parent_order)
    _need(tuple(order.tolist()) == PARENT_ROWS and type(cursor) is int
          and cursor in (0, 16, 32)
          and type(updates) is int and updates == cursor // BATCH_SIZE,
          'EXACT_PROGRESS')
    old._progress(cursor, updates, order, recipe, train_passes=1)
    return order


def capture(model, optimizer, *, recipe, feature_view_identity,
            parent_order, cursor, optimizer_updates, device):
    recipe = _recipe(recipe)
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = _progress(cursor, optimizer_updates, parent_order, recipe)
    weights = copy.deepcopy(model.state_dict())
    buffers = copy.deepcopy(dict(model.named_buffers()))
    old._tensor_map(weights, model.state_dict(), 'MODEL')
    old._tensor_map(buffers, dict(model.named_buffers()), 'BUFFER')
    optim = copy.deepcopy(optimizer.state_dict())
    old._optimizer_state(optim, optimizer, model, optimizer_updates)
    rng = copy.deepcopy(trainer._attended_session_rng_state(device=device))
    old._rng(rng, device=device, seed=recipe['seed'])
    value = {'schema_version': SCHEMA, 'binding': binding,
             'model_state': weights, 'buffer_state': buffers,
             'optimizer_state': optim, 'rng_state': rng,
             'module_training': {name: bool(module.training)
                                 for name, module in model.named_modules()},
             'parent_order': order, 'cursor': cursor,
             'optimizer_updates': optimizer_updates,
             'model_state_sha256': model_sha(weights),
             'buffer_state_sha256': old._tree_sha(buffers),
             'optimizer_state_sha256': old._tree_sha(optim),
             'rng_state_sha256': old._tree_sha(rng)}
    return {**value, 'contract_sha256': old._tree_sha(value)}


def restore(state, model, optimizer, *, recipe, feature_view_identity,
            parent_order, device):
    recipe = _recipe(recipe)
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = old._order(parent_order)
    _need(isinstance(state, Mapping) and set(state) == _STATE_KEYS
          and state['schema_version'] == SCHEMA
          and state['binding'] == binding, 'SCHEMA_OR_BINDING')
    _need(state['contract_sha256'] == old._tree_sha(
        {k: v for k, v in state.items() if k != 'contract_sha256'}), 'SEAL')
    _need(isinstance(state['parent_order'], torch.Tensor)
          and torch.equal(state['parent_order'].detach().cpu(), order),
          'PARENT_ORDER')
    _progress(state['cursor'], state['optimizer_updates'], order, recipe)
    old._tensor_map(state['model_state'], model.state_dict(), 'MODEL')
    old._tensor_map(state['buffer_state'], dict(model.named_buffers()), 'BUFFER')
    for name, value in state['buffer_state'].items():
        if name in state['model_state']:
            _need(torch.equal(value.detach().cpu(),
                              state['model_state'][name].detach().cpu()),
                  'PERSISTENT_BUFFER_PARITY:' + name)
    for name, value in model.named_buffers():
        if name in _IMMUTABLE or name.startswith(_IMMUTABLE_PREFIXES):
            _need(torch.equal(state['buffer_state'][name].detach().cpu(),
                              value.detach().cpu()), 'IMMUTABLE_BUFFER:' + name)
    modules = dict(model.named_modules())
    _need(isinstance(state['module_training'], Mapping)
          and set(state['module_training']) == set(modules)
          and all(type(v) is bool for v in state['module_training'].values()),
          'MODULE_MODES')
    old._optimizer_state(state['optimizer_state'], optimizer, model,
                         state['optimizer_updates'])
    old._rng(state['rng_state'], device=device, seed=recipe['seed'])
    _need(state['model_state_sha256'] == model_sha(state['model_state'])
          and state['buffer_state_sha256'] == old._tree_sha(state['buffer_state'])
          and state['optimizer_state_sha256'] == old._tree_sha(state['optimizer_state'])
          and state['rng_state_sha256'] == old._tree_sha(state['rng_state']),
          'STATE_DIGESTS')
    model.load_state_dict(state['model_state'], strict=True)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(state['buffer_state'][name].to(device))
    optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
    for name, training in state['module_training'].items():
        modules[name].training = training
    model.state_model._require_binding()
    model.backbone.require_input_normalization_state()
    _need(_binding(model, optimizer, recipe, feature_view_identity) == binding
          and model_sha(model.state_dict()) == state['model_state_sha256']
          and old._tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256']
          and old._tree_sha(optimizer.state_dict()) == state['optimizer_state_sha256'],
          'RESTORED_FULL_STATE')
    trainer._restore_attended_session_rng_state(state['rng_state'], device=device)
    _need(old._tree_sha(trainer._attended_session_rng_state(device=device))
          == state['rng_state_sha256'], 'RESTORED_RNG')
    return {'parent_order': order, 'cursor': state['cursor'],
            'optimizer_updates': state['optimizer_updates']}
