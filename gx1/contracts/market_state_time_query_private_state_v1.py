"""Exact in-memory optimizer and resume state for the private time-query probe.

No input IO, admission, launch, EMA, scheduler, target model or trading role.
The caller owns atomic checkpoint publication and the fixed TRAIN row schedule.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

STATE_SCHEMA = 'gx1_market_state_time_query_private_state_v1'
BINDING_SCHEMA = 'gx1_market_state_time_query_private_optimizer_binding_v1'
_STATE_KEYS = frozenset(('schema_version', 'binding', 'model_state', 'buffer_state',
    'optimizer_state', 'rng_state', 'module_training', 'parent_order', 'cursor',
    'optimizer_updates', 'model_state_sha256', 'buffer_state_sha256',
    'optimizer_state_sha256', 'rng_state_sha256', 'contract_sha256'))
_INACTIVE_PREFIXES = ('backbone.head_', 'backbone.dynamic_m5_query_projection.',
    'backbone.task_log_variances.', 'backbone.entry_decision_token.', 'backbone.exit_')
_QUERY_PARAMETERS = frozenset(('query_projection.0.weight', 'query_projection.0.bias',
    'query_head.weight', 'query_head.bias'))
_QUERY_BUFFERS = frozenset(('query_tf_coordinates', 'query_native_k',
    'query_objective_sha256', 'backbone.research_unobserved_units_profile_sha256'))


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TIME_QUERY_PRIVATE_STATE_' + reason)


def _deps():
    # The shared campaign imports this module for metadata before the CUDA guard.
    import torch
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
    from gx1.contracts.market_state_time_query_model_v1 import TimeQueryMarketStateModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    return torch, old, native_direction_research_recipe, TimeQueryMarketStateModel, canonical_model_state_sha256, trainer


def _numerical(value, model):
    _, _, _, Model, _, trainer = _deps()
    from gx1.contracts.market_state_time_query_private_campaign_v1 import (
        time_query_private_numerical_recipe,
    )
    _need(type(model) is Model, 'EXACT_MODEL_TYPE')
    _need(isinstance(value, Mapping) and dict(value) == time_query_private_numerical_recipe(),
        'CANONICAL_NUMERICAL_RECIPE')
    _need(value.get('research_objective_sha256') == model.research_objective['contract_sha256'],
        'RECIPE_OBJECTIVE')
    _need(float(trainer._GRAD_CLIP_NORM) == value['grad_clip_norm'],
        'CANONICAL_GRAD_CLIP')
    return copy.deepcopy(dict(value))


def _view(model, expected):
    actual = getattr(model, 'feature_view_identity', None)
    _need(isinstance(actual, Mapping) and isinstance(expected, Mapping)
        and set(actual) == set(expected) == {'contract_sha256', 'prefix_normalization',
            'research_objective_sha256'} and actual == expected, 'FEATURE_VIEW_IDENTITY')
    _need(actual['research_objective_sha256'] == model.research_objective['contract_sha256']
        and isinstance(actual['prefix_normalization'], Mapping), 'VIEW_OBJECTIVE')
    for sha in (actual['contract_sha256'], actual['research_objective_sha256']):
        _need(type(sha) is str and len(sha) == 64
            and all(char in '0123456789abcdef' for char in sha), 'VIEW_SHA256')
    return copy.deepcopy(dict(actual))


def _parameters(model):
    torch, old, _, Model, _, _ = _deps()
    _need(type(model) is Model, 'EXACT_MODEL_TYPE')
    model._require_binding()
    model.backbone.require_input_normalization_state()
    pairs = list(model.named_parameters(remove_duplicate=False))
    _need(bool(pairs) and len({id(parameter) for _, parameter in pairs}) == len(pairs),
        'PARAMETER_ALIASES')
    active = [(name, parameter) for name, parameter in pairs
        if not name.startswith(_INACTIVE_PREFIXES)]
    inactive = [(name, parameter) for name, parameter in pairs
        if name.startswith(_INACTIVE_PREFIXES)]
    _need(bool(active) and bool(inactive)
        and _QUERY_PARAMETERS <= {name for name, _ in active}
        and any(name.startswith('backbone.') for name, _ in active),
        'PARAMETER_PARTITION')
    _need(all(parameter.dtype == torch.float32
        and bool(torch.isfinite(parameter).all()) for _, parameter in pairs),
        'FINITE_FP32_PARAMETERS')
    return pairs, active, inactive


def _binding(model, optimizer, numerical_recipe, feature_view_identity):
    torch, old, _, _, _, _ = _deps()
    pairs, active, inactive = _parameters(model)
    _need(all(parameter.requires_grad for _, parameter in active)
        and all(not parameter.requires_grad for _, parameter in inactive),
        'REQUIRES_GRAD_PARTITION')
    kwargs = copy.deepcopy(numerical_recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    _need(type(optimizer) is torch.optim.AdamW
        and old._tree_sha(optimizer.defaults) == old._tree_sha(kwargs)
        and len(optimizer.param_groups) == 1, 'ADAMW_DEFAULTS')
    group = optimizer.param_groups[0]
    _need(set(group) == set(kwargs) | {'params'}
        and old._tree_sha({key: group[key] for key in kwargs}) == old._tree_sha(kwargs)
        and [id(parameter) for parameter in group['params']]
            == [id(parameter) for _, parameter in active], 'OPTIMIZER_GROUP_ORDER')
    value = {'schema_version': BINDING_SCHEMA,
        'numerical_recipe': copy.deepcopy(numerical_recipe),
        'feature_view_identity': _view(model, feature_view_identity),
        'research_objective': model.research_objective,
        'active_parameter_names': [name for name, _ in active],
        'inactive_parameter_names': [name for name, _ in inactive],
        'parameter_inventory': [{'name': name, 'shape': list(parameter.shape),
            'dtype': str(parameter.dtype)} for name, parameter in pairs],
        'optimizer_group_parameter_names': [[name for name, _ in active]],
        'ema_created': False, 'scheduler_created': False,
        'target_model_created': False}
    value['contract_sha256'] = old._tree_sha(value)
    return value


def build_time_query_private_optimizer(model, *, numerical_recipe):
    """Freeze unused old output heads; retain backbone and shared query decoder."""
    torch, old, _, _, _, _ = _deps()
    recipe = _numerical(numerical_recipe, model)
    old._no_partial_gradients(model)
    _, active, inactive = _parameters(model)
    for _, parameter in active:
        parameter.requires_grad_(True)
    for _, parameter in inactive:
        parameter.requires_grad_(False)
    kwargs = copy.deepcopy(recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.AdamW([parameter for _, parameter in active], **kwargs)
    return optimizer, _binding(model, optimizer, recipe, model.feature_view_identity)


def _progress(cursor, updates, order, recipe):
    torch, _, _, _, _, _ = _deps()
    from gx1.contracts.market_state_time_query_private_campaign_v1 import PARENT_ROWS
    _need(type(cursor) is int and cursor in (0, 16, 32)
        and type(updates) is int and updates == cursor // 16
        and torch.equal(order, torch.tensor(PARENT_ROWS, dtype=torch.int64))
        and recipe['batch_size'] == 16 and recipe['max_updates'] == 2,
        'PROGRESS')


def _sealed(value):
    _, old, _, _, _, _ = _deps()
    result = dict(value)
    result['contract_sha256'] = old._tree_sha(result)
    return result


def capture_time_query_private_state(model, optimizer, *, numerical_recipe,
        feature_view_identity, parent_order, cursor, optimizer_updates, device):
    """Capture independent model/AdamW/RNG state at an exact batch boundary."""
    _, old, _, _, digest, trainer = _deps()
    recipe = _numerical(numerical_recipe, model)
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = old._order(parent_order)
    _progress(cursor, optimizer_updates, order, recipe)
    model_state = copy.deepcopy(model.state_dict())
    buffers = copy.deepcopy(dict(model.named_buffers()))
    old._tensor_map(model_state, model.state_dict(), 'MODEL')
    old._tensor_map(buffers, dict(model.named_buffers()), 'BUFFER')
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    old._optimizer_state(optimizer_state, optimizer, model, optimizer_updates)
    rng = copy.deepcopy(trainer._attended_session_rng_state(device=device))
    old._rng(rng, device=device, seed=recipe['seed'])
    return _sealed({'schema_version': STATE_SCHEMA, 'binding': binding,
        'model_state': model_state, 'buffer_state': buffers,
        'optimizer_state': optimizer_state, 'rng_state': rng,
        'module_training': {name: bool(module.training)
            for name, module in model.named_modules()},
        'parent_order': order, 'cursor': cursor,
        'optimizer_updates': optimizer_updates,
        'model_state_sha256': digest(model_state),
        'buffer_state_sha256': old._tree_sha(buffers),
        'optimizer_state_sha256': old._tree_sha(optimizer_state),
        'rng_state_sha256': old._tree_sha(rng)})


def restore_time_query_private_state(state, model, optimizer, *, numerical_recipe,
        feature_view_identity, parent_order, device):
    """Validate complete checkpoint and immutable inputs before any mutation."""
    torch, old, _, _, digest, trainer = _deps()
    recipe = _numerical(numerical_recipe, model)
    old._device(model, device)
    old._no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity)
    order = old._order(parent_order)
    _need(isinstance(state, Mapping) and set(state) == _STATE_KEYS
        and state['schema_version'] == STATE_SCHEMA, 'SCHEMA')
    _need(state['binding'] == binding, 'BINDING_IDENTITY')
    _need(state['contract_sha256'] == old._tree_sha({key: value for key, value
        in state.items() if key != 'contract_sha256'}), 'SEAL')
    _need(isinstance(state['parent_order'], torch.Tensor)
        and state['parent_order'].dtype == torch.int64
        and torch.equal(state['parent_order'].detach().cpu(), order),
        'EXPECTED_PARENT_ORDER')
    _progress(state['cursor'], state['optimizer_updates'], order, recipe)
    old._tensor_map(state['model_state'], model.state_dict(), 'MODEL')
    old._tensor_map(state['buffer_state'], dict(model.named_buffers()), 'BUFFER')
    for name, value in state['buffer_state'].items():
        if name in state['model_state']:
            _need(torch.equal(value.detach().cpu(),
                state['model_state'][name].detach().cpu()),
                'PERSISTENT_BUFFER_PARITY:' + name)
    immutable_names = old.IMMUTABLE_BUFFER_NAMES | _QUERY_BUFFERS
    for name, buffer in model.named_buffers():
        if name in immutable_names or name.startswith(old.IMMUTABLE_BUFFER_PREFIXES):
            _need(torch.equal(state['buffer_state'][name].detach().cpu(),
                buffer.detach().cpu()), 'IMMUTABLE_BUFFER:' + name)
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
    return {'parent_order': order, 'cursor': state['cursor'],
        'optimizer_updates': state['optimizer_updates']}
