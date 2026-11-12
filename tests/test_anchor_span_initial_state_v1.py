"""Synthetic tensor import only: no market capsule, old model or optimizer."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from gx1.features.entry_specialist_feature_groups_v1 import (
    compile_full_ta_entry_execution_profile, classify_entry_specialist_feature, _canonical_sha256,
)
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as dual_contract
from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256 as campaign_sha256

PREFIX = 'state_model.backbone.'
FAMILY = 'chart_geometry_encoder'
SPANS = ('geomline_above_anchor_span_bars', 'geomline_below_anchor_span_bars')
SOURCE_PROFILE = '1' * 64
TFS = ('m5', 'm15', 'h1', 'h4', 'd1')


def _hash_tensor(value):
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)


class _Backbone(nn.Module):
    def __init__(self, profile):
        super().__init__()
        fields = profile['physical_field_names']
        local = profile['local_signal_indices_by_specialist'][FAMILY]
        mtf = profile['mtf_indices_by_specialist'][FAMILY]
        self.seq_proj = nn.Linear(len(fields['signal']), 128)
        self.snap_proj = nn.Linear(len(profile['generic_snap_signal_indices']), 128)
        self.specialist_proj = nn.ModuleDict({FAMILY: nn.Linear(len(local), 128)})
        self.mtf_family_proj = nn.ModuleDict({FAMILY: nn.Linear(len(mtf), 128)})
        self.mtf_feature_context_gate = nn.ModuleDict({
            tf + '__' + FAMILY: nn.Linear(128, len(mtf)) for tf in TFS})
        # Index-keyed parameters beyond both insertions exercise key renaming.
        self.signal_nominal_embeddings = nn.ModuleDict({str(len(fields['signal']) - 1): nn.Embedding(3, 128)})
        self.mtf_nominal_embeddings = nn.ModuleDict({tf + '_' + str(len(fields['mtf']) - 1): nn.Embedding(3, 128) for tf in TFS})
        self.register_buffer('entry_execution_profile_contract_sha256', _hash_tensor(profile['contract_sha256']))
        self.register_buffer('input_norm_contract_sha256', _hash_tensor('2' * 64))
        self.register_buffer('research_unobserved_units_profile_sha256', _hash_tensor('3' * 64))
        self.register_buffer('unchanged_scalar', torch.tensor(0.25))
        for surface in ('signal', 'ctx_cont', *('mtf_' + tf for tf in TFS)):
            width = len(fields['mtf' if surface.startswith('mtf_') else surface])
            for suffix in ('center', 'scale', 'binary_mask', 'categorical_mask'):
                value = (torch.ones(width) if suffix == 'scale' else torch.zeros(width))
                if suffix.endswith('mask'):
                    value = value.bool()
                self.register_buffer('input_norm_' + surface + '_' + suffix, value)
        routes = {
            'entry_signal_projection_idx': profile['signal_indices'],
            'generic_snap_idx': profile['generic_snap_signal_indices'],
            'temporal_alias_signal_idx': profile['temporal_alias_signal_indices'],
            'temporal_alias_ctx_cont_idx': profile['temporal_alias_ctx_cont_indices'],
            'input_norm_alias_signal_indices': profile['temporal_alias_signal_indices'],
            'input_norm_alias_ctx_cont_indices': profile['temporal_alias_ctx_cont_indices'],
        }
        for key, stems in (('local_signal_indices_by_specialist', ('specialist_idx_',)),
                           ('mtf_indices_by_specialist', ('multi_tf_specialist_idx_', 'multi_tf_feature_gate_idx_'))):
            for family, indices in profile[key].items():
                routes.update({stem + family: indices for stem in stems})
        for name, indices in routes.items():
            self.register_buffer(name, torch.tensor(indices, dtype=torch.int64),
                                 persistent=name.startswith('input_norm_alias_'))

    def require_input_normalization_state(self):
        pass


class _Model(nn.Module):
    def __init__(self, profile):
        super().__init__()
        self.state_model = nn.Module()
        self.state_model.backbone = _Backbone(profile)
        self.state_model.query_projection = nn.Linear(3, 2)
        self.state_model._ta_only = True
        self.state_model._dedicated_m5_side = True
        self.state_model._dedicated_non_m5_side = False
        self.state_model._dedicated_tf_activity = True
        flags = dict(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
        self.research_objective = dual_contract(**flags)
        self.state_model.register_buffer('query_objective_sha256', _hash_tensor(
            factorized_time_query_model_contract(**flags)['contract_sha256']))
        self.register_buffer('native_F_objective_sha256', _hash_tensor(self.research_objective['contract_sha256']))

    @property
    def backbone(self):
        return self.state_model.backbone


def _reseal(saved):
    saved['binding']['contract_sha256'] = old._tree_sha({key: value for key, value in saved['binding'].items() if key != 'contract_sha256'})
    saved['model_state_sha256'] = digest(saved['model_state'])
    for key in ('buffer_state', 'optimizer_state', 'rng_state'):
        saved[key + '_sha256'] = old._tree_sha(saved[key])
    saved['contract_sha256'] = old._tree_sha({key: value for key, value in saved.items() if key != 'contract_sha256'})


def _case(monkeypatch):
    profile = compile_full_ta_entry_execution_profile()
    fields = {key: list(value) for key, value in profile['physical_field_names'].items()}
    old_fields = {key: [name for name in names if name not in (
        tuple('chart.' + item for item in SPANS) if key == 'signal' else SPANS if key == 'mtf' else ())]
        for key, names in fields.items()}
    model = _Model(profile)
    monkeypatch.setattr(state, '_is_ta_sf', lambda value: value is model)
    monkeypatch.setattr(state, '_require_model_binding', lambda value: None)
    monkeypatch.setattr(state, '_numerical', lambda value, model: copy.deepcopy(value))
    monkeypatch.setattr(state, '_parameters', lambda value: (
        list(value.named_parameters()), list(value.named_parameters()), []))
    takes = {key: [names.index(name) for name in old_fields[key]] for key, names in fields.items()}
    def unrename(name):
        for base, surface in ((PREFIX + 'signal_nominal_embeddings.', 'signal'),
                              (PREFIX + 'mtf_nominal_embeddings.', 'mtf')):
            if name.startswith(base):
                token, *tail = name[len(base):].split('.')
                leader = ''
                if surface == 'mtf':
                    tf, token = token.rsplit('_', 1)
                    leader = tf + '_'
                return base + leader + str(takes[surface].index(int(token))) + ('.' + '.'.join(tail) if tail else '')
        return name
    old_chart = {surface: [name for name in old_fields[surface] if classify_entry_specialist_feature(name) == FAMILY]
                 for surface in ('signal', 'mtf')}
    chart_takes = {surface: [list(fields[surface][i] for i in profile[key][FAMILY]).index(name) for name in old_chart[surface]]
        for surface, key in (('signal', 'local_signal_indices_by_specialist'), ('mtf', 'mtf_indices_by_specialist'))}
    old_snap = [name for name in old_fields['signal'] if not name.startswith('ctx_cont.')]
    new_snap = [fields['signal'][i] for i in profile['generic_snap_signal_indices']]
    expanded = {
        PREFIX + 'seq_proj.weight': (1, takes['signal']),
        PREFIX + 'snap_proj.weight': (1, [new_snap.index(name) for name in old_snap]),
        PREFIX + 'specialist_proj.' + FAMILY + '.weight': (1, chart_takes['signal']),
        PREFIX + 'mtf_family_proj.' + FAMILY + '.weight': (1, chart_takes['mtf']),
    }
    for tf in TFS:
        for kind in ('weight', 'bias'):
            expanded[PREFIX + 'mtf_feature_context_gate.' + tf + '__' + FAMILY + '.' + kind] = (0, chart_takes['mtf'])
    parameters = {}
    for i, (name, target) in enumerate(model.named_parameters()):
        value = torch.arange(target.numel(), dtype=torch.float32).reshape(target.shape) / 8192 + i / 16
        if name in expanded:
            axis, indices = expanded[name]
            value = value.index_select(axis, torch.tensor(indices))
        parameters[unrename(name)] = value.clone()
    buffers = copy.deepcopy(dict(model.named_buffers()))
    for name, value in list(buffers.items()):
        short = name.removeprefix(PREFIX)
        if short == 'entry_execution_profile_contract_sha256':
            buffers[name] = _hash_tensor(SOURCE_PROFILE)
        elif short in ('input_norm_contract_sha256', 'research_unobserved_units_profile_sha256'):
            buffers[name] = _hash_tensor('5' * 64)
        elif short.startswith('input_norm_signal_'):
            buffers[name] = value[takes['signal']].clone()
        elif short.startswith('input_norm_mtf_'):
            buffers[name] = value[takes['mtf']].clone()
        elif short in ('entry_signal_projection_idx', 'generic_snap_idx', 'temporal_alias_signal_idx', 'input_norm_alias_signal_indices'):
            buffers[name] = torch.tensor([takes['signal'].index(int(i)) for i in value if int(i) in takes['signal']])
        elif short.startswith(('specialist_idx_', 'multi_tf_specialist_idx_', 'multi_tf_feature_gate_idx_')):
            surface = 'signal' if short.startswith('specialist_idx_') else 'mtf'
            buffers[name] = torch.tensor([takes[surface].index(int(i)) for i in value if int(i) in takes[surface]])
    def contract_seal(value):
        value = copy.deepcopy(value)
        value.pop('contract_sha256', None)
        return {**value, 'contract_sha256': _canonical_sha256(value)}
    flags = dict(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    old_s = factorized_time_query_model_contract(**flags)
    old_s['input_profile']['contract_sha256'] = SOURCE_PROFILE
    old_s = contract_seal(old_s)
    old_dual = dual_contract(**flags)
    old_dual['S_model_contract_sha256'] = old_s['contract_sha256']
    old_dual = contract_seal(old_dual)
    old_loss = loss_contract(**flags, m5_equal_observation=True)
    old_loss['dual_model_contract_sha256'] = old_dual['contract_sha256']
    old_loss = contract_seal(old_loss)
    recipe = ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_tf_activity=True, m5_equal_observation=True)
    recipe['research_objective_sha256'] = old_dual['contract_sha256']
    recipe['native_F_loss_contract_sha256'] = old_loss['contract_sha256']
    recipe['contract_sha256'] = campaign_sha256({k: v for k, v in recipe.items() if k != 'contract_sha256'})
    buffers['state_model.query_objective_sha256'] = _hash_tensor(old_s['contract_sha256'])
    buffers['native_F_objective_sha256'] = _hash_tensor(old_dual['contract_sha256'])
    source = {**parameters, **{name: value for name, value in buffers.items() if name in model.state_dict()}}
    names = list(parameters)
    binding = dict(schema_version=state.TA_BINDING_SCHEMA,
        feature_view_identity={'contract_sha256': '6' * 64, 'research_objective_sha256': old_s['contract_sha256']},
        research_objective=old_dual,
        numerical_recipe=recipe, active_parameter_names=names, inactive_parameter_names=[],
        optimizer_group_parameter_names=[names], parameter_inventory=[dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
            for name, value in parameters.items()])
    saved = dict(schema_version=state.TA_STATE_SCHEMA, source_commit='7' * 40, fold_id=1,
        feature_view_sha256='6' * 64, binding=binding, model_state=source, buffer_state=buffers,
        optimizer_state={'state': {}, 'param_groups': []}, rng_state={'synthetic': True},
        module_training={unrename(name): i % 2 == 0 for i, (name, _) in enumerate(model.named_modules())},
        parent_order=torch.arange(16), cursor=0, optimizer_updates=0, skipped_batches=0)
    _reseal(saved)
    return model, saved, old_fields, expanded


def _import(model, saved, fields, **kwargs):
    return state.import_anchor_span_activity_initial(saved, model, source_field_names=fields,
        source_profile_sha256=kwargs.get('source_profile_sha256', SOURCE_PROFILE),
        expected_model_state_sha256=kwargs.get('expected_model_state_sha256', saved['model_state_sha256']))


def test_exact_named_import_zeroes_fourteen_expansions_and_preserves_target_context(monkeypatch):
    model, saved, fields, expanded = _case(monkeypatch)
    for i, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(i % 2 == 0)
    model.eval()
    buffers = old._tree_sha(dict(model.named_buffers()))
    modes = {name: value.training for name, value in model.named_modules()}
    flags = {name: (id(value), value.requires_grad) for name, value in model.named_parameters()}
    source_sha, rng = old._tree_sha(saved), torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        pytest.fail('no optimizer or checkpoint decode in weights-only import')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    result = _import(model, saved, fields)
    assert result['expanded_parameter_tensors'] == 14
    assert result['added_parameters'] == 2314
    assert result['model_state_sha256'] == digest(model.state_dict())
    assert result['prediction_parity_measured'] is False and result['rng_restored'] is False
    assert buffers == old._tree_sha(dict(model.named_buffers()))
    assert modes == {name: value.training for name, value in model.named_modules()}
    assert flags == {name: (id(value), value.requires_grad) for name, value in model.named_parameters()}
    assert old._tree_sha(saved) == source_sha and torch.equal(torch.get_rng_state(), rng)
    assert all(value.grad is None for value in model.parameters())
    for old_name, name in result['parameter_name_map'].items():
        value = dict(model.named_parameters())[name]
        if name in expanded:
            axis, indices = expanded[name]
            extra = sorted(set(range(value.shape[axis])) - set(indices))
            assert torch.count_nonzero(value.index_select(axis, torch.tensor(extra))).item() == 0
            value = value.index_select(axis, torch.tensor(indices))
        assert digest({'tensor': value}) == digest({'tensor': saved['model_state'][old_name]})
    changed = {a: b for a, b in result['parameter_name_map'].items() if a != b}
    assert len(changed) == 6
    for old_name, name in changed.items():
        old_module, module = old_name.rsplit('.', 1)[0], name.rsplit('.', 1)[0]
        assert result['module_training'][module] == saved['module_training'][old_module]
    # Independent FP64 affine equality for finite arbitrary added-feature values.
    values = torch.linspace(-2, 3, 2 * 317, dtype=torch.float64).reshape(2, 317)
    take = [list(compile_full_ta_entry_execution_profile()['physical_field_names']['signal']).index(name) for name in fields['signal']]
    initial = values[:, take] @ saved['model_state'][PREFIX + 'seq_proj.weight'].double().T
    new = values @ model.backbone.seq_proj.weight.double().T
    torch.testing.assert_close(initial, new, rtol=0, atol=1e-11)


@pytest.mark.parametrize('fault', [
    'digest', 'seal', 'profile', 'field_order', 'field_missing', 'wrong_mode', 'nonempty_optimizer',
    'parameter_shape', 'parameter_extra', 'nonfinite', 'norm_value', 'signed_zero', 'routing', 'persistent_alias_routing',
    'mode_inventory', 'mode_type', 'partition', 'source_target_alias', 'partial_grad', 'loss_recipe',
    'dual_view_identity', 'numerical_hash_encoding',
])
def test_rejects_before_target_or_rng_mutation(monkeypatch, fault):
    model, saved, fields, _ = _case(monkeypatch)
    kwargs = {}
    if fault == 'digest':
        kwargs['expected_model_state_sha256'] = '0' * 64
    elif fault == 'seal':
        saved['contract_sha256'] = '0' * 64
    elif fault == 'profile':
        kwargs['source_profile_sha256'] = '0' * 64
    elif fault == 'field_order':
        fields['signal'][0:2] = fields['signal'][0:2][::-1]
    elif fault == 'field_missing':
        fields['mtf'].pop()
    elif fault == 'wrong_mode':
        model.state_model._dedicated_non_m5_side = True
    elif fault == 'nonempty_optimizer':
        saved['optimizer_state']['state'][0] = {'step': torch.tensor(1.)}
    elif fault in ('parameter_shape', 'nonfinite'):
        key = PREFIX + 'seq_proj.weight'
        if fault == 'parameter_shape':
            saved['model_state'][key] = saved['model_state'][key][:, :-1].clone()
            next(item for item in saved['binding']['parameter_inventory'] if item['name'] == key)['shape'] = list(saved['model_state'][key].shape)
        else:
            saved['model_state'][key][0, 0] = float('nan')
    elif fault == 'loss_recipe':
        saved['binding']['numerical_recipe']['future_weight'] = 2.0
    elif fault == 'dual_view_identity':
        saved['binding']['feature_view_identity']['research_objective_sha256'] = saved['binding']['research_objective']['contract_sha256']
    elif fault == 'numerical_hash_encoding':
        recipe = saved['binding']['numerical_recipe']
        recipe['contract_sha256'] = _canonical_sha256({k: v for k, v in recipe.items() if k != 'contract_sha256'})
    elif fault == 'parameter_extra':
        saved['model_state']['unexpected.weight'] = torch.ones(1)
    elif fault in ('norm_value', 'signed_zero'):
        key = PREFIX + 'input_norm_signal_center'
        saved['buffer_state'][key][0] = 0.5 if fault == 'norm_value' else -0.0
        saved['model_state'][key] = saved['buffer_state'][key].clone()
    elif fault == 'routing':
        saved['buffer_state'][PREFIX + 'generic_snap_idx'][0] = 999
    elif fault == 'persistent_alias_routing':
        key = PREFIX + 'input_norm_alias_signal_indices'
        saved['buffer_state'][key][0] = 999
        saved['model_state'][key] = saved['buffer_state'][key].clone()
    elif fault == 'mode_inventory':
        saved['module_training']['unexpected'] = True
    elif fault == 'mode_type':
        saved['module_training'][''] = 1
    elif fault == 'partition':
        saved['binding']['active_parameter_names'].reverse()
    elif fault == 'source_target_alias':
        key = 'state_model.query_projection.weight'
        saved['model_state'][key] = model.state_dict()[key]
    elif fault == 'partial_grad':
        model.backbone.seq_proj.weight.grad = torch.zeros_like(model.backbone.seq_proj.weight)
    if fault != 'seal':
        _reseal(saved)
    before = digest(model.state_dict())
    buffers, modes = old._tree_sha(dict(model.named_buffers())), {name: value.training for name, value in model.named_modules()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        _import(model, saved, fields, **kwargs)
    assert digest(model.state_dict()) == before
    assert old._tree_sha(dict(model.named_buffers())) == buffers
    assert {name: value.training for name, value in model.named_modules()} == modes
    assert torch.equal(torch.get_rng_state(), rng)


def test_fixed_span_short_mode_reuses_same_budget_and_forbids_scope_substitution():
    from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as worker
    from gx1.contracts import market_state_ta_anchor_span_short_learning_v1 as short
    from tests.test_ta_tf_activity_short_learning_worker import recipe
    value=recipe();value.update(run_scope=short.SCOPE,max_model_constructions=1,anchor_span_inputs=True)
    worker._require_execution_bounds(value,short_learning=True,tf_activity_learning=True,anchor_span_learning=True)
    for changed,kwargs in ((dict(value,max_model_constructions=2),dict(anchor_span_learning=True,tf_activity_learning=True)),
                           (value,dict(anchor_span_learning=False,tf_activity_learning=True)),
                           (value,dict(anchor_span_learning=True,tf_activity_learning=False))):
        with pytest.raises(RuntimeError):worker._require_execution_bounds(changed,short_learning=True,**kwargs)
