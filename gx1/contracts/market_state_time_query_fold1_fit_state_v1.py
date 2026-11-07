"""Full fold-1 time-query fit state at exact consumed TRAIN batch boundaries."""
from __future__ import annotations

import copy
from collections.abc import Mapping
import re

STATE_SCHEMA = 'gx1_market_state_time_query_fold1_fit_state_v1'
BINDING_SCHEMA = 'gx1_market_state_time_query_fold1_fit_optimizer_binding_v1'
_STATE_KEYS = frozenset({
    'schema_version', 'source_commit', 'fold_id', 'feature_view_sha256',
    'binding', 'model_state', 'buffer_state', 'optimizer_state', 'rng_state',
    'module_training', 'parent_order', 'cursor', 'optimizer_updates',
    'skipped_batches', 'model_state_sha256', 'buffer_state_sha256',
    'optimizer_state_sha256', 'rng_state_sha256', 'contract_sha256',
})
_QUERY_BUFFERS = frozenset((
    'query_tf_coordinates', 'query_native_k', 'query_objective_sha256',
    'backbone.research_unobserved_units_profile_sha256',
))
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TIME_QUERY_FOLD1_FIT_STATE_' + reason)


def _deps():
    import torch
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_private_state_v1 as private
    from gx1.contracts.market_state_time_query_model_v1 import TimeQueryMarketStateModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    return torch, old, private, TimeQueryMarketStateModel, canonical_model_state_sha256, trainer


def _numerical(value, model):
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import (
        fold1_numerical_recipe, MODEL_CONTRACT_SHA256,
    )
    _, _, _, Model, _, trainer = _deps()
    _need(type(model) is Model and isinstance(value, Mapping)
          and dict(value) == fold1_numerical_recipe()
          and value['research_objective_sha256'] == MODEL_CONTRACT_SHA256
          == model.research_objective['contract_sha256']
          and float(trainer._GRAD_CLIP_NORM) == value['grad_clip_norm'],
          'NUMERICAL_RECIPE')
    return copy.deepcopy(dict(value))


def _progress(cursor, updates, skipped, order, recipe):
    torch, old, _, _, _, _ = _deps()
    _need(type(cursor) is int and type(updates) is int and type(skipped) is int
          and isinstance(order, torch.Tensor) and order.dtype == torch.int64
          and order.ndim == 1 and order.numel() == 29574
          and 0 <= cursor <= 29574 and (cursor == 29574 or cursor % 16 == 0)
          and 0 <= updates <= recipe['max_updates']
          and 0 <= skipped <= recipe['max_updates']
          and updates + skipped == (cursor + 15) // 16,
          'EXACT_CONSUMED_BATCH_CURSOR')
    old._progress(cursor, updates, order, recipe)
    return (cursor + 15) // 16


def _binding(model, optimizer, numerical_recipe, feature_view_identity):
    torch, old, private, _, _, _ = _deps()
    pairs, active, inactive = private._parameters(model)
    view = private._view(model, feature_view_identity)
    _need(all(parameter.requires_grad for _, parameter in active)
          and all(not parameter.requires_grad for _, parameter in inactive),
          'PARAMETER_PARTITION')
    kwargs = copy.deepcopy(numerical_recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    _need(type(optimizer) is torch.optim.AdamW
          and old._tree_sha(optimizer.defaults) == old._tree_sha(kwargs)
          and len(optimizer.param_groups) == 1, 'OPTIMIZER_DEFAULTS')
    group = optimizer.param_groups[0]
    _need(set(group) == set(kwargs) | {'params'}
          and old._tree_sha({key: group[key] for key in kwargs}) == old._tree_sha(kwargs)
          and [id(parameter) for parameter in group['params']]
              == [id(parameter) for _, parameter in active], 'OPTIMIZER_GROUP_ORDER')
    binding = {
        'schema_version': BINDING_SCHEMA,
        'numerical_recipe': copy.deepcopy(numerical_recipe),
        'feature_view_identity': view,
        'research_objective': model.research_objective,
        'active_parameter_names': [name for name, _ in active],
        'inactive_parameter_names': [name for name, _ in inactive],
        'parameter_inventory': [
            {'name': name, 'shape': list(parameter.shape), 'dtype': str(parameter.dtype)}
            for name, parameter in pairs
        ],
        'optimizer_group_parameter_names': [[name for name, _ in active]],
        'ema_created': False, 'scheduler_created': False,
        'target_model_created': False,
    }
    binding['contract_sha256'] = old._tree_sha(binding)
    return binding


def build_fold1_fit_optimizer(model, *, numerical_recipe):
    """Train the existing backbone and shared query head; freeze old task heads."""
    torch, old, private, _, _, _ = _deps()
    recipe = _numerical(numerical_recipe, model)
    old._no_partial_gradients(model)
    _, active, inactive = private._parameters(model)
    for _, parameter in active:
        parameter.requires_grad_(True)
    for _, parameter in inactive:
        parameter.requires_grad_(False)
    kwargs = copy.deepcopy(recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.AdamW([parameter for _, parameter in active], **kwargs)
    return optimizer, _binding(model, optimizer, recipe, model.feature_view_identity)


def capture_fold1_fit_state(model, optimizer, *, numerical_recipe,
                            feature_view_identity, parent_order, cursor,
                            optimizer_updates, skipped_batches,
                            source_commit, device):
    """Capture model, AdamW, modes and RNG after each fully consumed batch."""
    _, old, private, _, digest, trainer = _deps()
    recipe = _numerical(numerical_recipe, model)
    _need(type(source_commit) is str and _COMMIT.fullmatch(source_commit) is not None,
          'SOURCE_COMMIT')
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = old._order(parent_order)
    _progress(cursor, optimizer_updates, skipped_batches, order, recipe)
    model_state = copy.deepcopy(model.state_dict())
    buffers = copy.deepcopy(dict(model.named_buffers()))
    old._tensor_map(model_state, model.state_dict(), 'MODEL')
    old._tensor_map(buffers, dict(model.named_buffers()), 'BUFFER')
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    old._optimizer_state(optimizer_state, optimizer, model, optimizer_updates)
    rng = copy.deepcopy(trainer._attended_session_rng_state(device=device))
    old._rng(rng, device=device, seed=recipe['seed'])
    value = {
        'schema_version': STATE_SCHEMA,
        'source_commit': source_commit,
        'fold_id': 1,
        'feature_view_sha256': binding['feature_view_identity']['contract_sha256'],
        'binding': binding,
        'model_state': model_state,
        'buffer_state': buffers,
        'optimizer_state': optimizer_state,
        'rng_state': rng,
        'module_training': {name: bool(module.training)
                            for name, module in model.named_modules()},
        'parent_order': order,
        'cursor': cursor,
        'optimizer_updates': optimizer_updates,
        'skipped_batches': skipped_batches,
        'model_state_sha256': digest(model_state),
        'buffer_state_sha256': old._tree_sha(buffers),
        'optimizer_state_sha256': old._tree_sha(optimizer_state),
        'rng_state_sha256': old._tree_sha(rng),
    }
    value['contract_sha256'] = old._tree_sha(value)
    return value


def restore_fold1_fit_state(state, model, optimizer, *, numerical_recipe,
                            feature_view_identity, parent_order,
                            source_commit, device):
    """Validate every immutable binding and digest before restoring state."""
    torch, old, private, _, digest, trainer = _deps()
    recipe = _numerical(numerical_recipe, model)
    _need(type(source_commit) is str and _COMMIT.fullmatch(source_commit) is not None,
          'SOURCE_COMMIT')
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = old._order(parent_order)
    _need(isinstance(state, Mapping) and set(state) == _STATE_KEYS
          and state['schema_version'] == STATE_SCHEMA
          and state['source_commit'] == source_commit
          and type(state['fold_id']) is int and state['fold_id'] == 1
          and state['feature_view_sha256'] == binding['feature_view_identity']['contract_sha256']
          and state['binding'] == binding,
          'EXACT_BINDING')
    _need(state['contract_sha256'] == old._tree_sha({
        key: value for key, value in state.items() if key != 'contract_sha256'
    }), 'STATE_SEAL')
    _need(isinstance(state['parent_order'], torch.Tensor)
          and state['parent_order'].dtype == torch.int64
          and torch.equal(state['parent_order'].detach().cpu(), order),
          'PARENT_ORDER')
    _progress(state['cursor'], state['optimizer_updates'],
              state['skipped_batches'], order, recipe)
    old._tensor_map(state['model_state'], model.state_dict(), 'MODEL')
    old._tensor_map(state['buffer_state'], dict(model.named_buffers()), 'BUFFER')
    for name, value in state['buffer_state'].items():
        if name in state['model_state']:
            _need(torch.equal(value.detach().cpu(),
                              state['model_state'][name].detach().cpu()),
                  'PERSISTENT_BUFFER:' + name)
    immutable = old.IMMUTABLE_BUFFER_NAMES | _QUERY_BUFFERS
    for name, buffer in model.named_buffers():
        if name in immutable or name.startswith(old.IMMUTABLE_BUFFER_PREFIXES):
            _need(torch.equal(state['buffer_state'][name].detach().cpu(),
                              buffer.detach().cpu()),
                  'IMMUTABLE_BUFFER:' + name)
    modes = dict(model.named_modules())
    _need(isinstance(state['module_training'], Mapping)
          and set(state['module_training']) == set(modes)
          and all(type(value) is bool for value in state['module_training'].values()),
          'MODULE_MODES')
    old._optimizer_state(state['optimizer_state'], optimizer, model,
                         state['optimizer_updates'])
    old._rng(state['rng_state'], device=device, seed=recipe['seed'])
    _need(state['model_state_sha256'] == digest(state['model_state'])
          and state['buffer_state_sha256'] == old._tree_sha(state['buffer_state'])
          and state['optimizer_state_sha256'] == old._tree_sha(state['optimizer_state'])
          and state['rng_state_sha256'] == old._tree_sha(state['rng_state']),
          'STATE_DIGESTS')
    model.load_state_dict(state['model_state'], strict=True)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(state['buffer_state'][name].to(device))
    optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
    for name, training in state['module_training'].items():
        modes[name].training = training
    model.backbone.require_input_normalization_state()
    model._require_binding()
    _need(_binding(model, optimizer, recipe, feature_view_identity) == binding
          and digest(model.state_dict()) == state['model_state_sha256']
          and old._tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256']
          and old._tree_sha(optimizer.state_dict()) == state['optimizer_state_sha256'],
          'RESTORED_FULL_STATE')
    trainer._restore_attended_session_rng_state(state['rng_state'], device=device)
    _need(old._tree_sha(trainer._attended_session_rng_state(device=device))
          == state['rng_state_sha256'], 'RESTORED_RNG')
    return {'cursor': state['cursor'],
            'optimizer_updates': state['optimizer_updates'],
            'skipped_batches': state['skipped_batches']}
