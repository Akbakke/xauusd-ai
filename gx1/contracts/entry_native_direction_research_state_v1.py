"""Pure-direction research optimizer and exact in-memory step-boundary state.

No file/data/admission IO, EMA, scheduler, target model or native phase aliases.
The caller owns fixed cohorts, loss-empty handling, source/fileSHA verification,
atomic persistence and actual execution admission. Semantic tree seals are not
serialized-file SHAs. Synthetic shells do not qualify real native training.
"""
from __future__ import annotations

import copy
import random
from collections.abc import Mapping

import numpy as np
import torch

from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
from gx1.contracts.entry_native_direction_research_v1 import NativeEntryDirectionResearch, native_direction_research_contract
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as _tree_sha

SCHEMA_VERSION = 'gx1_native_direction_research_state_v1'
MULTIPASS_SCHEMA_VERSION = 'gx1_native_direction_research_state_multipass_v1'
INACTIVE_PREFIXES = ('backbone.head_', 'backbone.dynamic_m5_query_projection.',
    'backbone.task_log_variances.', 'backbone.entry_decision_token.', 'backbone.exit_')
IMMUTABLE_BUFFER_NAMES = frozenset(('backbone.gc_mtf_layout_sha256', 'backbone.gc_mtf_normalization_sha256',
    'backbone.entry_execution_profile_contract_sha256', 'direction_horizon_minutes', 'bps_rms', 'bps_scale_sha256'))
IMMUTABLE_BUFFER_PREFIXES = ('backbone.input_norm_', 'backbone.gc_mtf_specialist_idx_')
STATE_KEYS = frozenset(('schema_version', 'binding', 'model_state', 'buffer_state', 'optimizer_state',
    'rng_state', 'module_training', 'parent_order', 'cursor', 'optimizer_updates', 'model_state_sha256',
    'buffer_state_sha256', 'optimizer_state_sha256', 'rng_state_sha256', 'contract_sha256'))


def _require(ok, reason):
    if not ok: raise RuntimeError('NATIVE_DIRECTION_RESEARCH_STATE_' + reason)


def _sealed(value):
    result = dict(value)
    result['contract_sha256'] = _tree_sha(result)
    return result


def _view(model, expected):
    value = getattr(model, 'feature_view_identity', None)
    _require(isinstance(value, Mapping) and set(value) == {'contract_sha256', 'prefix_normalization',
        'research_objective_sha256'} and value == expected, 'VIEW_IDENTITY')
    _require(value['research_objective_sha256'] == model.research_objective['contract_sha256'], 'OBJECTIVE_IDENTITY')
    prefix = value['prefix_normalization']
    _require(isinstance(prefix, Mapping) and set(prefix) == {'binding', 'contract_sha256', 'base_sha256',
        'gc_layout_sha256', 'gc_normalization_sha256', 'qualification_sha256'}, 'PREFIX_IDENTITY')
    for item in (value['contract_sha256'], value['research_objective_sha256'],
            *(prefix[key] for key in prefix if key != 'binding')):
        _require(type(item) is str and len(item) == 64 and all(c in '0123456789abcdef' for c in item), 'IDENTITY_SHA256')
    ref = prefix['binding']
    _require(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256'} and type(ref['path']) is str
        and ref['path'].startswith('/') and type(ref['sha256']) is str and len(ref['sha256']) == 64
        and all(c in '0123456789abcdef' for c in ref['sha256']), 'PREFIX_PLAINREF')
    return copy.deepcopy(dict(value))


def _parameters(model):
    _require(type(model) is NativeEntryDirectionResearch, 'EXACT_RESEARCH_TYPE')
    model._require_binding()
    model.backbone.require_input_normalization_state()
    pairs = list(model.named_parameters(remove_duplicate=False))
    _require(pairs and len({id(p) for _, p in pairs}) == len(pairs), 'PARAMETER_ALIASES')
    active = [(name, p) for name, p in pairs if not name.startswith(INACTIVE_PREFIXES)]
    inactive = [(name, p) for name, p in pairs if name.startswith(INACTIVE_PREFIXES)]
    _require(active and inactive and any(name.startswith('direction_head.') for name, _ in active), 'PARAMETER_PARTITIONS')
    _require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for _, p in pairs), 'FINITE_FP32_PARAMETERS')
    return pairs, active, inactive


def _no_partial_gradients(model):
    _require(all(p.grad is None for p in model.parameters()), 'PARTIAL_GRADIENTS')


def _kwargs(recipe):
    result = copy.deepcopy(recipe['optimizer']['kwargs']); result['betas'] = tuple(result['betas'])
    return result


def _binding(model, optimizer, recipe, feature_view_identity):
    pairs, active, inactive = _parameters(model)
    _require(recipe['research_objective_sha256'] == model.research_objective['contract_sha256'], 'RECIPE_OBJECTIVE')
    _require(all(p.requires_grad for _, p in active) and all(not p.requires_grad for _, p in inactive), 'REQUIRES_GRAD_PARTITION')
    _require(type(optimizer) is torch.optim.AdamW and _tree_sha(optimizer.defaults) == _tree_sha(_kwargs(recipe))
        and len(optimizer.param_groups) == 1, 'OPTIMIZER_TYPE_DEFAULTS')
    group = optimizer.param_groups[0]
    _require(set(group) == set(_kwargs(recipe)) | {'params'}
        and _tree_sha({key:group[key] for key in _kwargs(recipe)}) == _tree_sha(_kwargs(recipe))
        and [id(p) for p in group['params']] == [id(p) for _, p in active], 'OPTIMIZER_GROUP_ORDER')
    return _sealed({'schema_version': 'gx1_native_direction_research_optimizer_binding_v1',
        'recipe': recipe, 'feature_view_identity': _view(model, feature_view_identity),
        'research_objective': model.research_objective,
        **({'bps_scale_contract_sha256':model.bps_scale['contract_sha256']} if model.signed_bps else {}),
        'active_parameter_names': [name for name, _ in active],
        'inactive_parameter_names': [name for name, _ in inactive],
        'parameter_inventory': [{'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype)} for name, p in pairs],
        'optimizer_group_parameter_names': [[name for name, _ in active]],
        'ema_created': False, 'scheduler_created': False, 'target_model_created': False})


def build_native_direction_research_optimizer(model, *, recipe):
    """Freeze only explicit unused legacy outputs; keep every other path active.

    The caller handles loss=None without a step. For nonempty BCE, reuse
    trainer._optimizer_step_with_finite_gradients with no weight_ema argument.
    This builder neither seeds nor changes model modes/numerical policy.
    """
    recipe = require_native_direction_research_recipe(recipe)
    _require(recipe['research_objective_sha256'] == model.research_objective['contract_sha256'], 'RECIPE_OBJECTIVE')
    _no_partial_gradients(model)
    pairs, active, inactive = _parameters(model)
    _require(float(trainer._GRAD_CLIP_NORM) == recipe['grad_clip_norm'], 'CANONICAL_GRAD_CLIP')
    for _, parameter in active: parameter.requires_grad_(True)
    for _, parameter in inactive: parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for _, p in active], **_kwargs(recipe))
    return optimizer, _binding(model, optimizer, recipe, model.feature_view_identity)


def _device(model, device):
    _require(isinstance(device, torch.device) and device.type in ('cpu', 'cuda')
        and all(p.device == device for p in model.parameters())
        and all(b.device == device for b in model.buffers()), 'EXPLICIT_DEVICE')


def _order(value):
    _require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and value.ndim == 1
        and value.numel() > 0 and bool((value >= 0).all()) and value.unique().numel() == value.numel(), 'PARENT_ORDER')
    return value.detach().cpu().contiguous().clone()


def _progress(cursor, updates, order, recipe, *, train_passes=1):
    # cursor counts parent presentations, not fabricated or duplicated IDs.
    _require(type(train_passes) is int and train_passes in (1, 3), 'TRAIN_PASSES')
    n = order.numel(); batch = recipe['batch_size']; batches_per_pass = (n + batch - 1) // batch
    _require(type(cursor) is int and 0 <= cursor <= train_passes * n, 'CURSOR')
    completed, within = divmod(cursor, n)
    _require(within % batch == 0, 'CURSOR')
    if train_passes == 3:
        _require(recipe['max_updates'] == 3 * batches_per_pass, 'MULTIPASS_RECIPE_BUDGET')
    batches = completed * batches_per_pass + (within + batch - 1) // batch
    _require(type(updates) is int and 0 <= updates <= min(recipe['max_updates'], batches), 'OPTIMIZER_UPDATES')


def _tensor_map(value, expected, label):
    _require(isinstance(value, Mapping) and set(value) == set(expected), label + '_KEYS')
    for name, tensor in expected.items():
        item = value[name]
        _require(isinstance(item, torch.Tensor) and item.shape == tensor.shape and item.dtype == tensor.dtype
            and bool(torch.isfinite(item).all()), label + '_TENSOR:' + name)


def _optimizer_state(value, optimizer, model, updates):
    _require(isinstance(value, Mapping) and set(value) == {'state', 'param_groups'}
        and isinstance(value['state'], Mapping) and _tree_sha(value['param_groups']) == _tree_sha(optimizer.state_dict()['param_groups']), 'OPTIMIZER_STATE_GROUPS')
    parameters = optimizer.param_groups[0]['params']; ids = value['param_groups'][0]['params']
    _require(ids == list(range(len(parameters))) and set(value['state']) <= set(ids)
        and (bool(value['state']) if updates else not value['state']), 'OPTIMIZER_HISTORY')
    for index, item in value['state'].items():
        _require(type(index) is int and isinstance(item, Mapping) and set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'ADAMW_HISTORY_SCHEMA')
        step = item['step']; parameter = parameters[index]
        _require(isinstance(step, torch.Tensor) and step.ndim == 0 and step.dtype in (torch.float32, torch.float64)
            and bool(torch.isfinite(step)) and 0 < float(step) <= updates and float(step).is_integer(), 'ADAMW_STEP')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = item[key]
            _require(isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape and tensor.dtype == parameter.dtype
                and bool(torch.isfinite(tensor).all()), 'ADAMW_MOMENT')
        _require(bool((item['exp_avg_sq'] >= 0).all()), 'ADAMW_SECOND_MOMENT')


def _rng(value, *, device, seed):
    """Validate on private engines before mutating any global RNG/model state."""
    template = trainer._attended_session_rng_state(device=device)
    _require(isinstance(value, Mapping) and set(value) == set(template) and value['numpy_name'] == 'MT19937', 'RNG_SCHEMA')
    for name in ('numpy_keys', 'torch_cpu'):
        item = value[name]; expected = template[name]
        _require(isinstance(item, torch.Tensor) and item.dtype == expected.dtype and item.shape == expected.shape, 'RNG_TENSOR')
    _require(type(value['numpy_position']) is int and 0 <= value['numpy_position'] <= template['numpy_keys'].numel()
        and type(value['numpy_has_gauss']) is int
        and value['numpy_has_gauss'] in (0, 1) and type(value['numpy_cached']) is float
        and np.isfinite(value['numpy_cached']), 'RNG_NUMPY')
    try:
        random.Random(seed).setstate(value['python'])
        np.random.RandomState(seed).set_state((value['numpy_name'], value['numpy_keys'].detach().cpu().numpy(),
            value['numpy_position'], value['numpy_has_gauss'], value['numpy_cached']))
        torch.Generator(device='cpu').set_state(value['torch_cpu'].detach().cpu())
        if device.type == 'cuda':
            _require(isinstance(value['torch_cuda'], list) and len(value['torch_cuda']) == len(template['torch_cuda']), 'CUDA_RNG_COUNT')
            for index, item in enumerate(value['torch_cuda']):
                _require(isinstance(item, torch.Tensor) and item.dtype == torch.uint8
                    and item.shape == template['torch_cuda'][index].shape, 'CUDA_RNG_TENSOR')
                torch.Generator(device=torch.device('cuda', index)).set_state(item.detach().cpu())
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError('NATIVE_DIRECTION_RESEARCH_STATE_RNG_INVALID') from error


def capture_native_direction_research_state(model, optimizer, *, recipe, feature_view_identity,
        parent_order, cursor, optimizer_updates, device, train_passes=1):
    """Capture detached independent state; cursor can advance without an update."""
    recipe = require_native_direction_research_recipe(recipe); _device(model, device); _no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity); order = _order(parent_order)
    _progress(cursor, optimizer_updates, order, recipe, train_passes=train_passes)
    model_state = copy.deepcopy(model.state_dict()); buffers = copy.deepcopy(dict(model.named_buffers()))
    _tensor_map(model_state, model.state_dict(), 'MODEL'); _tensor_map(buffers, dict(model.named_buffers()), 'BUFFER')
    optimizer_state = copy.deepcopy(optimizer.state_dict()); _optimizer_state(optimizer_state, optimizer, model, optimizer_updates)
    rng = copy.deepcopy(trainer._attended_session_rng_state(device=device)); _rng(rng, device=device, seed=recipe['seed'])
    return _sealed({'schema_version': SCHEMA_VERSION if train_passes == 1 else MULTIPASS_SCHEMA_VERSION,
        **({'train_passes': train_passes} if train_passes != 1 else {}), 'binding': binding, 'model_state': model_state,
        'buffer_state': buffers, 'optimizer_state': optimizer_state, 'rng_state': rng,
        'module_training': {name: bool(module.training) for name, module in model.named_modules()},
        'parent_order': order, 'cursor': cursor, 'optimizer_updates': optimizer_updates,
        'model_state_sha256': canonical_model_state_sha256(model_state), 'buffer_state_sha256': _tree_sha(buffers),
        'optimizer_state_sha256': _tree_sha(optimizer_state), 'rng_state_sha256': _tree_sha(rng)})


def restore_native_direction_research_state(state, model, optimizer, *, recipe, feature_view_identity, parent_order, device, train_passes=1):
    """Validate the complete checkpoint before any restore mutation.

    Loading files/atomic publication and admission are deliberately caller-owned.
    No weights from a legacy native production session are accepted here.
    """
    recipe = require_native_direction_research_recipe(recipe); _device(model, device); _no_partial_gradients(model)
    binding = _binding(model, optimizer, recipe, feature_view_identity); order = _order(parent_order)
    _require(type(train_passes) is int and train_passes in (1, 3), 'TRAIN_PASSES')
    expected_keys = STATE_KEYS if train_passes == 1 else STATE_KEYS | {'train_passes'}
    expected_schema = SCHEMA_VERSION if train_passes == 1 else MULTIPASS_SCHEMA_VERSION
    _require(isinstance(state, Mapping) and set(state) == expected_keys and state['schema_version'] == expected_schema
        and (train_passes == 1 or (type(state['train_passes']) is int and state['train_passes'] == train_passes)), 'SCHEMA')
    _require(_tree_sha(state['binding']) == _tree_sha(binding), 'BINDING_IDENTITY')
    _require(state['contract_sha256'] == _tree_sha({key: value for key, value in state.items() if key != 'contract_sha256'}), 'SEAL')
    _require(isinstance(state['parent_order'], torch.Tensor) and state['parent_order'].dtype == torch.int64
        and torch.equal(state['parent_order'].detach().cpu(), order), 'EXPECTED_PARENT_ORDER')
    _progress(state['cursor'], state['optimizer_updates'], order, recipe, train_passes=train_passes)
    _tensor_map(state['model_state'], model.state_dict(), 'MODEL')
    _tensor_map(state['buffer_state'], dict(model.named_buffers()), 'BUFFER')
    for name, value in state['buffer_state'].items():
        if name in state['model_state']:
            _require(torch.equal(value.detach().cpu(), state['model_state'][name].detach().cpu()), 'PERSISTENT_BUFFER_PARITY')
    for name, buffer in model.named_buffers():
        if name in IMMUTABLE_BUFFER_NAMES or name.startswith(IMMUTABLE_BUFFER_PREFIXES):
            _require(torch.equal(state['buffer_state'][name].detach().cpu(), buffer.detach().cpu()), 'IMMUTABLE_BUFFER:' + name)
    modes = dict(model.named_modules())
    _require(isinstance(state['module_training'], Mapping) and set(state['module_training']) == set(modes)
        and all(type(value) is bool for value in state['module_training'].values()), 'MODULE_MODES')
    _optimizer_state(state['optimizer_state'], optimizer, model, state['optimizer_updates'])
    _rng(state['rng_state'], device=device, seed=recipe['seed'])
    _require(state['model_state_sha256'] == canonical_model_state_sha256(state['model_state'])
        and state['buffer_state_sha256'] == _tree_sha(state['buffer_state'])
        and state['optimizer_state_sha256'] == _tree_sha(state['optimizer_state'])
        and state['rng_state_sha256'] == _tree_sha(state['rng_state']), 'STATE_DIGESTS')
    model.load_state_dict(state['model_state'], strict=True)
    with torch.no_grad():
        for name, buffer in model.named_buffers():buffer.copy_(state['buffer_state'][name].to(device))
    optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
    for name, training in state['module_training'].items():modes[name].training = training
    model.backbone.require_input_normalization_state(); model._require_binding()
    _require(_binding(model, optimizer, recipe, feature_view_identity) == binding
        and canonical_model_state_sha256(model.state_dict()) == state['model_state_sha256']
        and _tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256']
        and _tree_sha(optimizer.state_dict()) == state['optimizer_state_sha256'], 'RESTORED_FULL_STATE')
    trainer._restore_attended_session_rng_state(state['rng_state'], device=device)
    _require(_tree_sha(trainer._attended_session_rng_state(device=device)) == state['rng_state_sha256'], 'RESTORED_RNG')
    return {'parent_order': order, 'cursor': state['cursor'], 'optimizer_updates': state['optimizer_updates']}


def migrate_one_pass_terminal_to_three_pass(state, model, optimizer, *, original_recipe, continuation_recipe,
        feature_view_identity, parent_order, device):
    """Strictly restore a completed one-pass state and bind it to three passes.

    No model update or IO occurs. The caller must bind and authorize the input
    checkpoint and independently verify actual prediction parity before training.
    """
    original_recipe = require_native_direction_research_recipe(original_recipe)
    continuation_recipe = require_native_direction_research_recipe(continuation_recipe)
    order = _order(parent_order); n = order.numel(); batch = original_recipe['batch_size']
    per_pass = (n + batch - 1) // batch
    _require(original_recipe['max_updates'] == per_pass
        and continuation_recipe['max_updates'] == 3 * per_pass
        and {k:v for k,v in original_recipe.items() if k not in ('max_updates','contract_sha256')}
            == {k:v for k,v in continuation_recipe.items() if k not in ('max_updates','contract_sha256')},
        'MIGRATION_RECIPE')
    _require(isinstance(state, Mapping) and state.get('cursor') == n
        and state.get('optimizer_updates') == per_pass, 'MIGRATION_NOT_FULL_ONE_PASS')
    restored = restore_native_direction_research_state(state, model, optimizer, recipe=original_recipe,
        feature_view_identity=feature_view_identity, parent_order=order, device=device)
    migrated = capture_native_direction_research_state(model, optimizer, recipe=continuation_recipe,
        feature_view_identity=feature_view_identity, parent_order=order, cursor=restored['cursor'],
        optimizer_updates=restored['optimizer_updates'], device=device, train_passes=3)
    _require(all(migrated[key] == state[key] for key in ('model_state_sha256', 'buffer_state_sha256',
        'optimizer_state_sha256', 'rng_state_sha256'))
        and migrated['module_training'] == state['module_training'], 'MIGRATION_FULL_STATE_PARITY')
    return migrated
