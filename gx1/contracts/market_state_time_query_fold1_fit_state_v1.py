"""Full fold-1 time-query fit state at exact consumed TRAIN batch boundaries."""
from __future__ import annotations

import copy
from collections.abc import Mapping
import re

STATE_SCHEMA = 'gx1_market_state_time_query_fold1_fit_state_v1'
BINDING_SCHEMA = 'gx1_market_state_time_query_fold1_fit_optimizer_binding_v1'
TA_STATE_SCHEMA = 'gx1_market_state_ta_sf_fold1_fit_state_v1'
TA_BINDING_SCHEMA = 'gx1_market_state_ta_sf_fold1_fit_optimizer_binding_v1'
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



def _is_ta_sf(model):
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    return type(model) is NativeFutureDualModel


def _feature_view(model):
    return model.state_model.feature_view_identity if _is_ta_sf(model) else model.feature_view_identity


def _require_model_binding(model):
    if not _is_ta_sf(model):
        model._require_binding()
        return
    torch, _, _, _, _, _ = _deps()
    from gx1.contracts.market_state_time_query_native_F_model_v1 import contract
    model._require_binding()
    _need(model.state_model._ta_only is True, 'TA_SF_PROFILE')
    expected = contract(ta_only=True,
        dedicated_m5_side=model.state_model._dedicated_m5_side,
        dedicated_non_m5_side=model.state_model._dedicated_non_m5_side,
        dedicated_tf_activity=model.state_model._dedicated_tf_activity)['contract_sha256']
    reference = torch.tensor(list(bytes.fromhex(expected)), dtype=torch.uint8,
                             device=model.future_head.weight.device)
    head = model.future_head
    _need(model.research_objective['contract_sha256'] == expected
          and torch.equal(model.native_F_objective_sha256, reference)
          and head.in_features == model.state_model._hidden_width
          and head.out_features == 2
          and head.weight.dtype == head.bias.dtype == torch.float32
          and head.weight.device == head.bias.device
              == model.state_model.query_native_k.device
          and bool(torch.isfinite(head.weight).all())
          and bool(torch.isfinite(head.bias).all()), 'TA_SF_HEAD_BINDING')


def _parameters(model):
    if _is_ta_sf(model):
        from gx1.contracts import market_state_native_F_private_state_v1 as dual
        _require_model_binding(model)
        return dual._parameters(model)
    _, _, private, _, _, _ = _deps()
    return private._parameters(model)


def _numerical(value, model):
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import (
        fold1_numerical_recipe, MODEL_CONTRACT_SHA256,
    )
    _, _, _, Model, _, trainer = _deps()
    if _is_ta_sf(model):
        from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
        from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
        _require_model_binding(model)
        _need(isinstance(value, Mapping) and dict(value) == ta_sf_numerical_recipe(
                  dedicated_m5_side=model.state_model._dedicated_m5_side,
                  dedicated_non_m5_side=model.state_model._dedicated_non_m5_side,
                  dedicated_tf_activity=model.state_model._dedicated_tf_activity,
                  m5_equal_observation=value.get('m5_equal_observation', False))
              and value['research_objective_sha256'] == model.research_objective['contract_sha256']
              and value['native_F_loss_contract_sha256'] == loss_contract(ta_only=True,
                  dedicated_m5_side=model.state_model._dedicated_m5_side,
                  dedicated_non_m5_side=model.state_model._dedicated_non_m5_side,
                  dedicated_tf_activity=model.state_model._dedicated_tf_activity,
                  m5_equal_observation=value.get('m5_equal_observation', False))['contract_sha256']
              and value['future_weight'] == 1.0
              and float(trainer._GRAD_CLIP_NORM) == value['grad_clip_norm'],
              'TA_SF_NUMERICAL_RECIPE')
        return copy.deepcopy(dict(value))
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
    pairs, active, inactive = _parameters(model)
    view = private._view(model.state_model if _is_ta_sf(model) else model,
                         feature_view_identity)
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
        'schema_version': TA_BINDING_SCHEMA if _is_ta_sf(model) else BINDING_SCHEMA,
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
    _, active, inactive = _parameters(model)
    for _, parameter in active:
        parameter.requires_grad_(True)
    for _, parameter in inactive:
        parameter.requires_grad_(False)
    kwargs = copy.deepcopy(recipe['optimizer']['kwargs'])
    kwargs['betas'] = tuple(kwargs['betas'])
    optimizer = torch.optim.AdamW([parameter for _, parameter in active], **kwargs)
    return optimizer, _binding(model, optimizer, recipe, _feature_view(model))


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
        'schema_version': TA_STATE_SCHEMA if _is_ta_sf(model) else STATE_SCHEMA,
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
          and state['schema_version'] == (TA_STATE_SCHEMA if _is_ta_sf(model) else STATE_SCHEMA)
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
    immutable_prefixes = old.IMMUTABLE_BUFFER_PREFIXES
    if _is_ta_sf(model):
        immutable = frozenset({'native_F_objective_sha256'} |
                              {'state_model.' + name for name in immutable})
        immutable_prefixes = tuple('state_model.' + prefix for prefix in immutable_prefixes)
    for name, buffer in model.named_buffers():
        if name in immutable or name.startswith(immutable_prefixes):
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
    _require_model_binding(model)
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


def copy_legacy_weights_to_dedicated_m5(legacy_model, model, *,
                                        expected_legacy_model_sha256):
    """Copy a strictly restored legacy predictor into the opt-in variant.

    The caller must restore the original full state first and bind its checkpoint
    provenance. This transfers weights/buffers only, never optimizer history or
    progress. A new empty AdamW, cursor zero and actual initial measurement are
    required. No RNG is consumed here.
    """
    torch, old, private, _, digest, _ = _deps()
    _need(_is_ta_sf(legacy_model) and _is_ta_sf(model)
          and legacy_model is not model
          and legacy_model.state_model._dedicated_m5_side is False
          and model.state_model._dedicated_m5_side is True, 'M5_WARMSTART_MODE')
    _require_model_binding(legacy_model)
    _require_model_binding(model)
    old._no_partial_gradients(legacy_model)
    old._no_partial_gradients(model)
    source_view, target_view = _feature_view(legacy_model), _feature_view(model)
    private._view(legacy_model.state_model, source_view)
    private._view(model.state_model, target_view)
    _need(source_view['prefix_normalization'] == target_view['prefix_normalization'],
          'M5_WARMSTART_NORMALIZATION')
    source, target = legacy_model.state_dict(), model.state_dict()
    _need(digest(source) == expected_legacy_model_sha256, 'M5_WARMSTART_SOURCE_DIGEST')
    clones = {
        'state_model.m5_side_head.weight': 'state_model.query_head.weight',
        'state_model.m5_side_head.bias': 'state_model.query_head.bias',
        'future_m5_side_head.weight': 'future_head.weight',
        'future_m5_side_head.bias': 'future_head.bias',
    }
    identities = {'state_model.query_objective_sha256', 'native_F_objective_sha256'}
    _need(set(target) == set(source) | set(clones), 'M5_WARMSTART_INVENTORY')
    for name, value in source.items():
        destination = target[name]
        _need(value.shape == destination.shape and value.dtype == destination.dtype
              and value.device == destination.device, 'M5_WARMSTART_TENSOR:' + name)
    for name, original in clones.items():
        _need(target[name].shape == source[original][1:2].shape
              and target[name].dtype == source[original].dtype
              and target[name].device == source[original].device,
              'M5_WARMSTART_SIDE_ROW:' + name)
    buffers = dict(model.named_buffers())
    old_buffers = dict(legacy_model.named_buffers())
    _need(set(buffers) == set(old_buffers), 'M5_WARMSTART_BUFFER_INVENTORY')
    # Fresh/current normalization and other immutable buffers must already agree.
    _need(all(torch.equal(buffers[name], old_buffers[name])
              for name in buffers if name not in identities),
          'M5_WARMSTART_BUFFER_VALUES')
    old_modes = dict(legacy_model.named_modules())
    modes = dict(model.named_modules())
    mode_clones = {'state_model.m5_side_head': 'state_model.query_head',
                   'future_m5_side_head': 'future_head'}
    _need(set(modes) == set(old_modes) | set(mode_clones), 'M5_WARMSTART_MODULES')
    with torch.no_grad():
        for name, value in source.items():
            if name not in identities:
                target[name].copy_(value)
        for name, original in clones.items():
            target[name].copy_(source[original][1:2])
    for name, module in modes.items():
        module.training = old_modes[mode_clones.get(name, name)].training
    _require_model_binding(model)
    _need(digest(legacy_model.state_dict()) == expected_legacy_model_sha256,
          'M5_WARMSTART_SOURCE_MUTATED')
    copied_state = model.state_dict()
    _need(all(torch.equal(copied_state[name], value)
              for name, value in source.items() if name not in identities),
          'M5_WARMSTART_COPY_PARITY')
    return {
        'schema_version': 'gx1_ta_dedicated_m5_weights_warmstart_v1',
        'source_model_state_sha256': expected_legacy_model_sha256,
        'model_state_sha256': digest(copied_state),
        'source_model_contract_sha256': legacy_model.research_objective['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'copied_side_rows': clones,
        'added_parameters': sum(target[name].numel() for name in clones),
        'optimizer_policy': 'fresh_AdamW_empty_history_no_inherited_updates',
        'cursor': 0, 'optimizer_updates': 0,
        'fresh_actual_initial_measurement_required': True,
        'prediction_parity_measured': False,
    }



def copy_dedicated_m5_weights_to_non_m5_sides(legacy_model, model, *,
                                             expected_legacy_model_sha256):
    """Clone four shared side rows per task after strict original-state restore.

    Existing M5, activity, query/backbone, normalization, modes and trainability
    are preserved. Only the two model-identity buffers differ. This transfers no
    optimizer or progress and admits no run; a fresh actual baseline is required.
    """
    torch, old, private, _, digest, _ = _deps()
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    _need(_is_ta_sf(legacy_model) and _is_ta_sf(model)
          and legacy_model is not model
          and legacy_model.state_model._dedicated_m5_side is True
          and model.state_model._dedicated_m5_side is True
          and legacy_model.state_model._dedicated_non_m5_side is False
          and model.state_model._dedicated_non_m5_side is True,
          'NON_M5_WARMSTART_MODE')
    _require_model_binding(legacy_model)
    _require_model_binding(model)
    old._no_partial_gradients(legacy_model)
    old._no_partial_gradients(model)
    source_view, target_view = _feature_view(legacy_model), _feature_view(model)
    private._view(legacy_model.state_model, source_view)
    private._view(model.state_model, target_view)
    _need(source_view['prefix_normalization'] == target_view['prefix_normalization'],
          'NON_M5_WARMSTART_NORMALIZATION')
    source, target = legacy_model.state_dict(), model.state_dict()
    _need(digest(source) == expected_legacy_model_sha256, 'NON_M5_WARMSTART_SOURCE_DIGEST')
    clones = {
        prefix + '.' + timeframe + '.' + kind: shared + '.' + kind
        for prefix, shared in (
            ('state_model.non_m5_side_heads', 'state_model.query_head'),
            ('future_non_m5_side_heads', 'future_head'))
        for timeframe in TIMEFRAMES[1:]
        for kind in ('weight', 'bias')
    }
    identities = {'state_model.query_objective_sha256', 'native_F_objective_sha256'}
    _need(set(target) == set(source) | set(clones), 'NON_M5_WARMSTART_INVENTORY')
    source_parameters = dict(legacy_model.named_parameters())
    target_parameters = dict(model.named_parameters())
    _need(not ({parameter.untyped_storage().data_ptr() for parameter in source_parameters.values()}
               & {parameter.untyped_storage().data_ptr() for parameter in target_parameters.values()}),
          'NON_M5_WARMSTART_PARAMETER_ALIASES')
    for name, value in source.items():
        destination = target[name]
        _need(value.shape == destination.shape and value.dtype == destination.dtype
              and value.device == destination.device, 'NON_M5_WARMSTART_TENSOR:' + name)
    for name, original in clones.items():
        _need(target[name].shape == source[original][1:2].shape
              and target[name].dtype == source[original].dtype
              and target[name].device == source[original].device,
              'NON_M5_WARMSTART_SIDE_ROW:' + name)
    buffers, old_buffers = dict(model.named_buffers()), dict(legacy_model.named_buffers())
    _need(set(buffers) == set(old_buffers), 'NON_M5_WARMSTART_BUFFER_INVENTORY')
    _need(all(torch.equal(buffers[name], old_buffers[name])
              for name in buffers if name not in identities), 'NON_M5_WARMSTART_BUFFER_VALUES')
    old_modes, modes = dict(legacy_model.named_modules()), dict(model.named_modules())
    mode_clones = {
        prefix + suffix: shared
        for prefix, shared in (
            ('state_model.non_m5_side_heads', 'state_model.query_head'),
            ('future_non_m5_side_heads', 'future_head'))
        for suffix in ('', *('.' + timeframe for timeframe in TIMEFRAMES[1:]))
    }
    _need(set(modes) == set(old_modes) | set(mode_clones), 'NON_M5_WARMSTART_MODULES')
    with torch.no_grad():
        for name, value in source.items():
            if name not in identities:
                target[name].copy_(value)
        for name, original in clones.items():
            target[name].copy_(source[original][1:2])
    for name, parameter in target_parameters.items():
        parameter.requires_grad_(source_parameters[clones.get(name, name)].requires_grad)
    for name, module in modes.items():
        module.training = old_modes[mode_clones.get(name, name)].training
    _require_model_binding(model)
    _need(digest(legacy_model.state_dict()) == expected_legacy_model_sha256,
          'NON_M5_WARMSTART_SOURCE_MUTATED')
    copied = model.state_dict()
    _need(all(torch.equal(copied[name], value)
              for name, value in source.items() if name not in identities)
          and all(torch.equal(copied[name], source[original][1:2])
                  for name, original in clones.items()), 'NON_M5_WARMSTART_COPY_PARITY')
    return {
        'schema_version': 'gx1_ta_non_m5_side_weights_warmstart_v1',
        'source_model_state_sha256': expected_legacy_model_sha256,
        'model_state_sha256': digest(copied),
        'source_model_contract_sha256': legacy_model.research_objective['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'copied_side_rows': clones,
        'added_parameters': sum(target[name].numel() for name in clones),
        'optimizer_policy': 'fresh_AdamW_empty_history_no_inherited_updates',
        'cursor': 0, 'optimizer_updates': 0,
        'fresh_actual_initial_measurement_required': True,
        'prediction_parity_measured': False,
    }



def copy_dedicated_m5_weights_to_tf_activity(legacy_model, model, *,
                                            expected_legacy_model_sha256):
    """Clone five activity rows per task from a strictly bound original model.

    This isolated variant keeps the existing M5 and shared non-M5 side heads.
    All existing tensors, modes and trainability are preserved except the two
    destination model-identity buffers. No RNG, optimizer history or progress is
    transferred; the caller binds original c171 provenance and a fresh baseline.
    """
    torch, old, private, _, digest, _ = _deps()
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    _need(_is_ta_sf(legacy_model) and _is_ta_sf(model)
          and legacy_model is not model
          and legacy_model.state_model._dedicated_m5_side is True
          and model.state_model._dedicated_m5_side is True
          and legacy_model.state_model._dedicated_non_m5_side is False
          and model.state_model._dedicated_non_m5_side is False
          and legacy_model.state_model._dedicated_tf_activity is False
          and model.state_model._dedicated_tf_activity is True,
          'TF_ACTIVITY_WARMSTART_MODE')
    _require_model_binding(legacy_model)
    _require_model_binding(model)
    old._no_partial_gradients(legacy_model)
    old._no_partial_gradients(model)
    source_view, target_view = _feature_view(legacy_model), _feature_view(model)
    private._view(legacy_model.state_model, source_view)
    private._view(model.state_model, target_view)
    _need(source_view['prefix_normalization'] == target_view['prefix_normalization'],
          'TF_ACTIVITY_WARMSTART_NORMALIZATION')
    source, target = legacy_model.state_dict(), model.state_dict()
    _need(type(expected_legacy_model_sha256) is str
          and re.fullmatch(r'[0-9a-f]{64}', expected_legacy_model_sha256) is not None
          and digest(source) == expected_legacy_model_sha256,
          'TF_ACTIVITY_WARMSTART_SOURCE_DIGEST')
    clones = {
        prefix + '.' + timeframe + '.' + kind: shared + '.' + kind
        for prefix, shared in (
            ('state_model.tf_activity_heads', 'state_model.query_head'),
            ('future_tf_activity_heads', 'future_head'))
        for timeframe in TIMEFRAMES
        for kind in ('weight', 'bias')
    }
    identities = {'state_model.query_objective_sha256', 'native_F_objective_sha256'}
    _need(set(target) == set(source) | set(clones), 'TF_ACTIVITY_WARMSTART_INVENTORY')
    old._tensor_map(source, source, 'TF_ACTIVITY_WARMSTART_SOURCE')
    old._tensor_map(target, target, 'TF_ACTIVITY_WARMSTART_TARGET')
    source_pairs = list(legacy_model.named_parameters(remove_duplicate=False))
    target_pairs = list(model.named_parameters(remove_duplicate=False))
    source_parameters, target_parameters = dict(source_pairs), dict(target_pairs)
    _need(set(target_parameters) == set(source_parameters) | set(clones)
          and all(len({id(parameter) for _, parameter in pairs}) == len(pairs)
                  and len({parameter.untyped_storage().data_ptr() for _, parameter in pairs}) == len(pairs)
                  for pairs in (source_pairs, target_pairs)),
          'TF_ACTIVITY_WARMSTART_PARAMETER_ALIASES')
    buffers, old_buffers = dict(model.named_buffers()), dict(legacy_model.named_buffers())
    source_storage = {value.untyped_storage().data_ptr()
                      for value in (*source_parameters.values(), *old_buffers.values()) if value.numel()}
    target_storage = {value.untyped_storage().data_ptr()
                      for value in (*target_parameters.values(), *buffers.values()) if value.numel()}
    _need(source_storage.isdisjoint(target_storage), 'TF_ACTIVITY_WARMSTART_STORAGE_ALIASES')
    for name, value in source.items():
        destination = target[name]
        _need(value.shape == destination.shape and value.dtype == destination.dtype
              and value.device == destination.device, 'TF_ACTIVITY_WARMSTART_TENSOR:' + name)
    for name, original in clones.items():
        _need(target[name].shape == source[original][0:1].shape
              and target[name].dtype == source[original].dtype
              and target[name].device == source[original].device,
              'TF_ACTIVITY_WARMSTART_ACTIVITY_ROW:' + name)
    _need(set(buffers) == set(old_buffers), 'TF_ACTIVITY_WARMSTART_BUFFER_INVENTORY')
    _need(all(torch.equal(buffers[name], old_buffers[name])
              for name in buffers if name not in identities), 'TF_ACTIVITY_WARMSTART_BUFFER_VALUES')
    old_modes, modes = dict(legacy_model.named_modules()), dict(model.named_modules())
    mode_clones = {
        prefix + suffix: shared
        for prefix, shared in (
            ('state_model.tf_activity_heads', 'state_model.query_head'),
            ('future_tf_activity_heads', 'future_head'))
        for suffix in ('', *('.' + timeframe for timeframe in TIMEFRAMES))
    }
    _need(set(modes) == set(old_modes) | set(mode_clones)
          and all(type(module.training) is bool for modules in (old_modes, modes)
                  for module in modules.values()), 'TF_ACTIVITY_WARMSTART_MODULES')
    with torch.no_grad():
        for name, value in source.items():
            if name not in identities:
                target[name].copy_(value)
        for name, original in clones.items():
            target[name].copy_(source[original][0:1])
    for name, parameter in target_parameters.items():
        parameter.requires_grad_(source_parameters[clones.get(name, name)].requires_grad)
    for name, module in modes.items():
        module.training = old_modes[mode_clones.get(name, name)].training
    _require_model_binding(model)
    _need(digest(legacy_model.state_dict()) == expected_legacy_model_sha256,
          'TF_ACTIVITY_WARMSTART_SOURCE_MUTATED')
    copied = model.state_dict()
    _need(all(torch.equal(copied[name], value)
              for name, value in source.items() if name not in identities)
          and all(torch.equal(copied[name], source[original][0:1])
                  for name, original in clones.items()), 'TF_ACTIVITY_WARMSTART_COPY_PARITY')
    return {
        'schema_version': 'gx1_ta_tf_activity_weights_warmstart_v1',
        'source_model_state_sha256': expected_legacy_model_sha256,
        'model_state_sha256': digest(copied),
        'source_model_contract_sha256': legacy_model.research_objective['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'copied_activity_rows': clones,
        'added_parameters': sum(target[name].numel() for name in clones),
        'optimizer_policy': 'fresh_AdamW_empty_history_no_inherited_updates',
        'cursor': 0, 'optimizer_updates': 0,
        'fresh_actual_initial_measurement_required': True,
        'prediction_parity_measured': False,
    }


def import_equal_weight_m5_side_heads(model, coefficients, *,
                                     expected_model_state_sha256,
                                     coefficient_artifact, coefficient_recipe):
    """Import four already-rounded FP32 tensors into a qualified TA M5 model.

    The caller owns strict legacy restore/migration, artifact/recipe file binding
    and execution admission. References here are provenance, not file reads.
    No optimizer, progress, mode, requires-grad or RNG state is changed. A fresh
    actual output baseline and checkpoint roundtrip remain the caller's work.
    """
    torch, old, private, _, digest, trainer = _deps()
    _need(_is_ta_sf(model) and model.state_model._dedicated_m5_side is True,
          'M5_IMPORT_MODE')
    _require_model_binding(model)
    old._no_partial_gradients(model)
    view = _feature_view(model)
    private._view(model.state_model, view)
    names = (
        'state_model.m5_side_head.weight', 'state_model.m5_side_head.bias',
        'future_m5_side_head.weight', 'future_m5_side_head.bias',
    )
    _need(isinstance(coefficients, Mapping) and set(coefficients) == set(names),
          'M5_IMPORT_COEFFICIENT_KEYS')
    refs = {}
    for label, ref in (('coefficient_artifact', coefficient_artifact),
                       ('coefficient_recipe', coefficient_recipe)):
        _need(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256'}
              and type(ref['path']) is str and ref['path'].startswith('/')
              and type(ref['sha256']) is str
              and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) is not None,
              'M5_IMPORT_REFERENCE:' + label)
        refs[label] = copy.deepcopy(dict(ref))
    pairs = list(model.named_parameters(remove_duplicate=False))
    _need(len({id(parameter) for _, parameter in pairs}) == len(pairs),
          'M5_IMPORT_PARAMETER_ALIASES')
    parameters = dict(pairs)
    target = model.state_dict()
    _need(digest(target) == expected_model_state_sha256, 'M5_IMPORT_SOURCE_DIGEST')
    for name in names:
        value = coefficients[name]
        _need(name in parameters and isinstance(value, torch.Tensor)
              and value.layout == torch.strided
              and value.dtype == target[name].dtype == torch.float32
              and value.shape == target[name].shape
              and bool(torch.isfinite(value).all()), 'M5_IMPORT_TENSOR:' + name)
    # Snapshot only the tiny imported tensors; hashes bind the unchanged state.
    incoming = {name: coefficients[name].detach().clone() for name in names}
    before = {name: target[name].detach().clone() for name in names}
    unchanged_sha = digest({name: value for name, value in target.items() if name not in names})
    buffer_sha = old._tree_sha(dict(model.named_buffers()))
    modes = {name: bool(module.training) for name, module in model.named_modules()}
    identities = [(name, id(parameter), parameter.requires_grad) for name, parameter in pairs]
    device = parameters[names[0]].device
    rng_sha = old._tree_sha(trainer._attended_session_rng_state(device=device))
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(incoming[name])
    _require_model_binding(model)
    old._no_partial_gradients(model)
    after = model.state_dict()
    _need(digest({name: value for name, value in after.items() if name not in names})
          == unchanged_sha, 'M5_IMPORT_OTHER_STATE_MUTATED')
    _need(old._tree_sha(dict(model.named_buffers())) == buffer_sha,
          'M5_IMPORT_BUFFERS_MUTATED')
    _need({name: bool(module.training) for name, module in model.named_modules()} == modes,
          'M5_IMPORT_MODES_MUTATED')
    _need([(name, id(parameter), parameter.requires_grad)
           for name, parameter in model.named_parameters(remove_duplicate=False)] == identities,
          'M5_IMPORT_PARAMETER_IDENTITY_MUTATED')
    _need(old._tree_sha(trainer._attended_session_rng_state(device=device)) == rng_sha,
          'M5_IMPORT_RNG_MUTATED')
    changes = {}
    for name in names:
        source_sha = digest({name: incoming[name]})
        after_sha = digest({name: after[name]})
        _need(after_sha == source_sha, 'M5_IMPORT_COPY_PARITY:' + name)
        delta = after[name].detach().double() - before[name].double()
        changes[name] = {
            'shape': list(after[name].shape), 'dtype': str(after[name].dtype),
            'before_sha256': digest({name: before[name]}),
            'after_sha256': after_sha, 'coefficient_sha256': source_sha,
            'changed_elements': int((delta != 0).sum().item()),
            'max_abs_change': float(delta.abs().max().item()),
        }
    return {
        'schema_version': 'gx1_ta_m5_equal_weight_head_import_v1',
        'source_model_state_sha256': expected_model_state_sha256,
        'model_state_sha256': digest(after),
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'feature_view_sha256': view['contract_sha256'],
        **refs, 'tensor_changes': changes,
        'unchanged_model_state_sha256': unchanged_sha,
        'buffer_state_sha256': buffer_sha, 'rng_state_sha256': rng_sha,
        'module_training': modes, 'parameter_identity_unchanged': True,
        'optimizer_steps_performed': 0,
        'fresh_actual_output_baseline_required': True,
    }


def rebind_m5_equal_observation_initial_state(saved, model, optimizer, *,
                                              expected_model_state_sha256,
                                              new_numerical_recipe,
                                              feature_view_identity,
                                              parent_order, source_commit,
                                              new_source_commit, device):
    """Strictly restore the old dedicated-M5 zero-step state, then rebind loss.

    The caller binds the checkpoint file, old source and expected model digest
    and owns execution admission. This performs no weight migration, optimizer
    reset or update: unchanged strict restore loads the old state, and unchanged
    capture records the same objects under the exact opt-in numerical recipe.
    """
    torch, old, _, _, _, _ = _deps()
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe

    _need(_is_ta_sf(model) and model.state_model._dedicated_m5_side is True,
          'M5_LOSS_REBIND_MODE')
    _require_model_binding(model)
    old_recipe = ta_sf_numerical_recipe(dedicated_m5_side=True)
    expected_new = ta_sf_numerical_recipe(
        dedicated_m5_side=True, m5_equal_observation=True)
    _need(isinstance(new_numerical_recipe, Mapping)
          and dict(new_numerical_recipe) == expected_new,
          'M5_LOSS_REBIND_NEW_NUMERICAL')
    new_recipe = _numerical(new_numerical_recipe, model)
    _need(type(new_source_commit) is str
          and _COMMIT.fullmatch(new_source_commit) is not None,
          'M5_LOSS_REBIND_NEW_SOURCE')
    _need(type(expected_model_state_sha256) is str
          and re.fullmatch(r'[0-9a-f]{64}', expected_model_state_sha256) is not None
          and isinstance(saved, Mapping)
          and saved.get('model_state_sha256') == expected_model_state_sha256,
          'M5_LOSS_REBIND_MODEL_DIGEST')
    _need(isinstance(saved.get('binding'), Mapping)
          and saved['binding'].get('numerical_recipe') == old_recipe,
          'M5_LOSS_REBIND_OLD_NUMERICAL')
    counters = ('cursor', 'optimizer_updates', 'skipped_batches')
    _need(all(type(saved.get(key)) is int and saved[key] == 0 for key in counters),
          'M5_LOSS_REBIND_ZERO_PROGRESS')
    _need(type(optimizer) is torch.optim.AdamW and not optimizer.state
          and isinstance(saved.get('optimizer_state'), Mapping)
          and isinstance(saved['optimizer_state'].get('state'), Mapping)
          and not saved['optimizer_state']['state'],
          'M5_LOSS_REBIND_EMPTY_OPTIMIZER')
    old._no_partial_gradients(model)
    identities = [(name, id(parameter), parameter.requires_grad)
                  for name, parameter in model.named_parameters(remove_duplicate=False)]
    buffer_ids = [(name, id(buffer)) for name, buffer in model.named_buffers()]

    # All original source, seal, tensor, optimizer, order and RNG checks apply.
    restored = restore_fold1_fit_state(
        saved, model, optimizer, numerical_recipe=old_recipe,
        feature_view_identity=feature_view_identity, parent_order=parent_order,
        source_commit=source_commit, device=device)
    rebound = capture_fold1_fit_state(
        model, optimizer, numerical_recipe=new_recipe,
        feature_view_identity=feature_view_identity, parent_order=parent_order,
        cursor=0, optimizer_updates=0, skipped_batches=0,
        source_commit=new_source_commit, device=device)

    digests = ('model_state_sha256', 'buffer_state_sha256',
               'optimizer_state_sha256', 'rng_state_sha256')
    _need(restored == {key: 0 for key in counters}
          and all(rebound[key] == saved[key] for key in digests)
          and all(rebound[key] == saved[key] for key in (
              'schema_version', 'fold_id', 'feature_view_sha256',
              'module_training', *counters))
          and torch.equal(rebound['parent_order'], saved['parent_order'].detach().cpu()),
          'M5_LOSS_REBIND_PRESERVED_STATE')
    excluded = {'numerical_recipe', 'contract_sha256'}
    _need({key: value for key, value in rebound['binding'].items() if key not in excluded}
          == {key: value for key, value in saved['binding'].items() if key not in excluded},
          'M5_LOSS_REBIND_PRESERVED_BINDING')
    _need([(name, id(parameter), parameter.requires_grad)
           for name, parameter in model.named_parameters(remove_duplicate=False)] == identities
          and [(name, id(buffer)) for name, buffer in model.named_buffers()] == buffer_ids
          and not optimizer.state, 'M5_LOSS_REBIND_OBJECT_IDENTITY')
    evidence = {
        'schema_version': 'gx1_ta_m5_equal_observation_initial_rebinding_v1',
        'source_state_sha256': saved['contract_sha256'],
        'state_sha256': rebound['contract_sha256'],
        'source_commit': source_commit, 'new_source_commit': new_source_commit,
        'old_numerical_recipe_sha256': old_recipe['contract_sha256'],
        'new_numerical_recipe_sha256': new_recipe['contract_sha256'],
        **{key: rebound[key] for key in digests},
        'parent_order_sha256': old._tree_sha(rebound['parent_order']),
        'module_training': copy.deepcopy(rebound['module_training']),
        'optimizer_group_parameter_names': copy.deepcopy(
            rebound['binding']['optimizer_group_parameter_names']),
        **restored,
        'parameter_identity_unchanged': True, 'buffer_identity_unchanged': True,
        'optimizer_steps_performed': 0,
        'changed_state_fields': [key for key in ('binding', 'source_commit', 'contract_sha256')
                                 if rebound[key] != saved[key]],
        'changed_binding_fields': ['numerical_recipe', 'contract_sha256'],
    }
    return rebound, evidence
