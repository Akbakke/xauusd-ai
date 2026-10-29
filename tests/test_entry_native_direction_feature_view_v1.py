"""Synthetic raw-view/constructor-spy checks, not genuine native gradients."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import entry_native_direction_feature_view_v1 as owner
from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix
from gx1.features import entry_specialist_feature_groups_v1 as groups
from gx1.features import htf_features as htf
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig
from tests.test_entry_native_direction_prefix_normalization_v1 import prefix_case


def _ref(name):
    return {'path': '/synthetic/' + name, 'sha256': '1' * 64}


def _seal(value):
    value = {key: item for key, item in value.items() if key != 'contract_sha256'}
    return {**value, 'contract_sha256': owner._sha(value)}


def _metadata():
    fields = groups.compile_compact_entry_execution_profile()['physical_field_names']
    signal = {family: [] for family in groups.MODEL_NATIVE_TRAINING_SPECIALISTS}
    for index, name in enumerate(fields['signal']):
        signal[groups.classify_entry_specialist_feature(name)].append(index)
    context = groups.model_native_context_specialist_routing_contract()
    context['temporal_alias_policy'] = groups.model_native_context_temporal_alias_policy(fields['signal'])
    mtf = {'feature_names': list(fields['mtf']), 'multi_tf_num_layers': 1, 'multi_tf_scale': 0.1}
    for tf, length in owner.history.WINDOW_ROWS.items():
        mtf[tf.lower() + '_seq_dim'] = len(fields['mtf']); mtf[tf.lower() + '_seq_len'] = length
    return {'seq_input_dim': len(fields['signal']), 'snap_input_dim': len(fields['signal']), 'seq_len': 96,
        'ctx_cont_dim': len(fields['ctx_cont']), 'ctx_cat_dim': len(fields['ctx_cat']), 'dropout': 0.0,
        'multi_tf': mtf, 'specialist_fusion': {'input_indices': signal, 'context_routing': context,
            'num_layers': 1, 'fusion_scale': 0.1, 'cross_family_fusion_scale': 0.1},
        # Constructor scalar provenance may retain this old identity; it must
        # never cause a legacy artifact read or override the new prefix stats.
        'gc_mtf_history_artifact': _ref('legacy223/SURFACES.json'),
        'entry_execution_profile': {'name': 'legacy', 'contract_sha256': '2' * 64}}


@pytest.fixture(scope='module')
def bound_case():
    case = prefix_case()
    # One real historical gap, without changing any current parent clock.
    case['entry_sequence_source']['times_ns'][:50] -= owner.BAR_NS
    sources = {key: _ref(key + '.json') for key in owner.SOURCE_REF_KEYS}
    case['source_bindings'] = {'ancestral264_plan': sources['raw_input_provider_plan'],
        'current_gc_layout': sources['raw_gc_layout'], 'current_gc_raw_transition': sources['raw_gc_transition'],
        'current_source_owners': {path: sources[key] for key, path in (
            ('mtf_window_source', 'gx1/features/htf_features.py'),
            ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
            ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py'))}, 'synthetic_only': True}
    qualification = prefix.qualify_native_direction_prefix_population(**case)
    fitted = prefix.fit_native_direction_prefix_normalization(qualification)
    metadata = _metadata()
    arguments = {'prefix_normalization': fitted, 'prefix_normalization_ref': _ref('prefix.json'),
        'constructor_metadata': metadata, 'constructor_metadata_ref': _ref('constructor.json'),
        'source_bindings': sources, 'max_batch_rows': 2}
    view = owner.bind_native_direction_feature_view(**arguments)
    return {'case': case, 'arguments': arguments, 'view': view}


def _inputs(bound):
    case = bound['case']
    # DataFrame deepcopy does not promise that copied attrs still share the
    # copied frame. Restore the exact cache-owner view before its real slicer.
    for frame in case['multi_tf_sources'].values():
        frame.attrs['feats_np'] = frame.to_numpy(dtype=np.float32, copy=False)
        frame.attrs['feats_np'].setflags(write=False)
        frame.attrs['ts_int64'].setflags(write=False)
    for surface in case['gc_surfaces'].values():
        for key in ('native_row_indices', 'available_ns', 'features'): surface[key].setflags(write=False)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'snap_x', 'ctx_cont', 'ctx_cat'): case[key].setflags(write=False)
    for key in ('signal', 'times_ns', 'sample_positions'): case['entry_sequence_source'][key].setflags(write=False)
    return {key: case[key] for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'entry_sequence_source',
        'snap_x', 'ctx_cont', 'ctx_cat', 'multi_tf_sources', 'gc_surfaces', 'gc_layout')} | {
            'batch_parent_rows': np.array([5, 12], dtype=np.int64)}


def test_binding_separates_file_and_content_identity_and_preserves_full_routes(bound_case):
    view, args = bound_case['view'], bound_case['arguments']
    assert owner.bind_native_direction_feature_view(**args, expected=view) == view
    isolated = owner.bind_native_direction_feature_view(**args)
    isolated['physical_widths']['signal'] += 1
    assert owner.bind_native_direction_feature_view(**args) == view
    assert view['prefix_normalization']['binding']['sha256'] != view['prefix_normalization']['contract_sha256']
    assert view['constructor_metadata']['binding']['sha256'] != view['constructor_metadata']['content_sha256']
    assert view['prefix_normalization']['gc_normalization_sha256'] == args['prefix_normalization']['gc_mtf_history']['normalization']['contract_sha256']
    assert view['research_objective']['native_forwards_per_research_forward'] == 1
    assert view['entry_execution_profile']['name'] == groups.GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
    assert len(args['constructor_metadata']['specialist_fusion']['input_indices']) == 8
    assert view['evidence_scope'] == owner.EVIDENCE_SCOPE


def test_raw_batch_uses_real_canonical_slicers_same_gc_ids_and_preserves_gap_and_later_rows(bound_case, monkeypatch):
    bound = deepcopy(bound_case); arguments = _inputs(bound)
    calls = []; gc_calls = []
    route, gather = owner.get_model_native_multi_tf_route_windows, owner.history.gather_gc_mtf_history_rows
    def record_route(*args, **kwargs):
        calls.append(kwargs)
        return route(*args, **kwargs)
    def record_gc(surface, **kwargs):
        gc_calls.append(kwargs)
        return gather(surface, **kwargs)
    monkeypatch.setattr(owner, 'get_model_native_multi_tf_route_windows', record_route)
    monkeypatch.setattr(owner.history, 'gather_gc_mtf_history_rows', record_gc)
    output = owner.gather_native_direction_feature_batch(bound['view'], **arguments)
    features, proof = output['features'], output['witness']
    assert len(calls) == 2 and len(gc_calls) == 10 and all(call['return_row_indices'] is True for call in calls)
    assert len(features) == 14 and proof['parent_rows'] == [5, 12]
    assert proof['decision_ns'][1] >= bound['view']['prefix']['cutoff_ns']  # later OOF inputs are legal
    assert features['seq_x'].shape == (2, 96, 2725) and features['ctx_cont'].shape == (2, 2542)
    for i, position in enumerate(proof['source_positions']):
        source = arguments['entry_sequence_source']
        assert np.array_equal(features['seq_x'][i], source['signal'][position - 95:position + 1])
        assert (np.diff(source['times_ns'][position - 95:position + 1]) > owner.BAR_NS).any()
    assert np.array_equal(features['seq_x'][:, -1], features['snap_x']) and proof['raw_values_not_normalized']
    assert np.array_equal(features['snap_x'], arguments['snap_x'][[1, 4]])
    for tf, length in owner.history.WINDOW_ROWS.items():
        assert features['seq_' + tf.lower()].shape == (2, length, 253)
        assert features['gc_seq_' + tf.lower()].shape == (2, length, arguments['gc_layout']['widths'][tf])
        for i, witness in enumerate(proof['multi_tf'][tf]):
            call = next(call for call in gc_calls[i * 5:(i + 1) * 5] if call['tf'] == tf)
            assert call['native_row_indices'].tolist() == witness['native_row_indices']
            assert call['available_ns'].tolist() == witness['available_ns']
            assert max(witness['available_ns']) <= proof['decision_ns'][i]
    assert not set(features).intersection({'sign', 'observed', 'decision_ns', 'terminal_time_ns', 'cutoff_ns'})


@pytest.mark.parametrize('fault', ['parent', 'decision', 'pointer', 'snapshot', 'alias', 'writable_axis', 'oversized'])
def test_selected_row_clock_alias_and_batch_guards_reject(bound_case, fault):
    bound = deepcopy(bound_case); case = bound['case']
    if fault == 'decision': case['decision_ns'][1] += 1
    if fault == 'pointer': case['entry_sequence_source']['sample_positions'][1] -= 1
    if fault == 'snapshot': case['snap_x'][1, 0] += 1
    if fault == 'alias': case['ctx_cont'][1, 0] += 1
    arguments = _inputs(bound)
    if fault == 'parent': arguments['batch_parent_rows'][0] = 999
    if fault == 'writable_axis': arguments['parent_rows'].setflags(write=True)
    if fault == 'oversized': arguments['batch_parent_rows'] = np.array([2, 5, 7], np.int64)
    with pytest.raises(RuntimeError): owner.gather_native_direction_feature_batch(bound['view'], **arguments)


@pytest.mark.parametrize('fault', ['id', 'clock'])
def test_gc_gather_rejects_any_different_historical_native_id_or_close(bound_case, fault):
    bound = deepcopy(bound_case); surface = bound['case']['gc_surfaces']['D1']
    if fault == 'id': surface['native_row_indices'] += 1000
    else: surface['available_ns'] += 60_000_000_000
    with pytest.raises(RuntimeError, match='GATHER_'):
        owner.gather_native_direction_feature_batch(bound['view'], **_inputs(bound))


@pytest.mark.parametrize('fault', ['scalar', 'routing', 'prefix', 'source'])
def test_binding_rejects_unbound_configuration_prefix_or_source_metadata(bound_case, fault):
    arguments = deepcopy(bound_case['arguments'])
    if fault == 'scalar': del arguments['constructor_metadata']['multi_tf']['multi_tf_scale']
    elif fault == 'routing': arguments['constructor_metadata']['specialist_fusion']['input_indices'].pop('chart_geometry_encoder')
    elif fault == 'source': arguments['source_bindings']['raw_gc_transition'] = _ref('wrong_transition.json')
    else:
        fitted = arguments['prefix_normalization']
        fitted['prefix_qualification']['proof']['prefix']['rows'] += 1
        fitted['prefix_qualification'] = _seal(fitted['prefix_qualification'])
        arguments['prefix_normalization'] = _seal(fitted)
    with pytest.raises(RuntimeError): owner.bind_native_direction_feature_view(**arguments)


def _constructor_spy(monkeypatch):
    calls = []; norm_calls = []
    def initialize(self, **kwargs):
        calls.append(kwargs)
        nn.Module.__init__(self)
        cfg = CtxModelConfig.__new__(CtxModelConfig)
        object.__setattr__(cfg, 'd_model', 128)
        object.__setattr__(cfg, 'entry_training_phase', kwargs['entry_training_phase'])
        self.cfg = cfg
        self.entry_training_phase = kwargs['entry_training_phase']
        self.entry_execution_profile = kwargs['entry_execution_profile']
        self.gc_mtf_history = kwargs['gc_mtf_history']
        self.entry_q_joint_in = nn.Linear(512, 128)
    def require_normalization(self): norm_calls.append(self)
    def forbidden(*_args, **_kwargs): pytest.fail('feature constructor reached old artifact join or forward')
    monkeypatch.setattr(owner.EntryV10CtxHybridTransformer, '__init__', initialize)
    monkeypatch.setattr(owner.EntryV10CtxHybridTransformer, 'require_input_normalization_state', require_normalization)
    monkeypatch.setattr(owner.EntryV10CtxHybridTransformer, 'forward', forbidden)
    monkeypatch.setattr(owner.history, 'read_gc_mtf_history_artifact_metadata', forbidden)
    return calls, norm_calls


def test_direct_constructor_spy_gets_full_eight_routes_new_stats_and_unchanged_dynamic_queries(bound_case, monkeypatch):
    calls, norm_calls = _constructor_spy(monkeypatch)
    args, view = bound_case['arguments'], bound_case['view']
    model = owner.build_native_direction_research_model(view, prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'], device=torch.device('cpu'))
    assert len(calls) == len(norm_calls) == 1
    kwargs = calls[0]
    assert tuple(kwargs['specialist_input_indices']) == tuple(groups.MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert len(kwargs['multi_tf_specialist_input_indices']) == 8
    assert kwargs['dynamic_m5_query_minutes'] == tuple(range(1, 481))
    assert kwargs['input_normalization'] == args['prefix_normalization']['base_normalization']
    assert kwargs['gc_mtf_history'] == args['prefix_normalization']['gc_mtf_history']
    assert kwargs['entry_execution_profile'] == view['entry_execution_profile']
    assert model.backbone is norm_calls[0] and model.direction_head.in_features == 128
    assert model.direction_head.out_features == 3 and not hasattr(model, 'entry_training_phase')
    assert model.feature_view_identity['contract_sha256'] == view['contract_sha256']


def test_constructor_rejects_metadata_change_before_any_spy_call(bound_case, monkeypatch):
    calls, _ = _constructor_spy(monkeypatch)
    args = deepcopy(bound_case['arguments']); args['constructor_metadata']['dropout'] = 0.2
    with pytest.raises(RuntimeError, match='EXPECTED_VIEW_MISMATCH'):
        owner.build_native_direction_research_model(bound_case['view'], prefix_normalization=args['prefix_normalization'],
            constructor_metadata=args['constructor_metadata'], device=torch.device('cpu'))
    assert calls == []


def test_gather_api_rejects_target_or_future_endpoint_arguments(bound_case):
    with pytest.raises(TypeError):
        owner.gather_native_direction_feature_batch(bound_case['view'], **_inputs(deepcopy(bound_case)),
            sign=np.ones((2, 3), dtype=np.int8))


def test_dense_view_binds_own_source_and_constructs_sixty_head_model(bound_case, monkeypatch):
    from gx1.contracts import entry_native_direction_dense_v1 as dense
    arguments = deepcopy(bound_case['arguments'])
    arguments['source_bindings'].update({key: _ref(key + '.py') for key in owner.DENSE_SOURCE_REF_KEYS})
    view = owner.bind_native_direction_feature_view(**arguments, dense_minutes=True)
    assert view['dense_minute_supervision'] is True
    assert view['research_objective'] == dense.dense_direction_research_contract()
    calls, norm_calls = _constructor_spy(monkeypatch)
    model = owner.build_native_direction_research_model(view, prefix_normalization=arguments['prefix_normalization'],
        constructor_metadata=arguments['constructor_metadata'], device=torch.device('cpu'))
    assert type(model) is dense.DenseNativeEntryDirectionResearch
    assert model.direction_head.out_features == 3 and model.dense_aux_head.out_features == 57
    assert len(calls) == len(norm_calls) == 1
    assert model.feature_view_identity['research_objective_sha256'] == view['research_objective']['contract_sha256']
    with pytest.raises(RuntimeError):
        owner.bind_native_direction_feature_view(**bound_case['arguments'], dense_minutes=True)
    with pytest.raises(RuntimeError):
        owner.bind_native_direction_feature_view(**arguments, dense_minutes=True, bps_scale={})


def test_dense_profile_restores_geometry_without_removing_compact_or_gc_fields(bound_case):
    from gx1.contracts import entry_exit_production_architecture_v1 as architecture
    history = bound_case['arguments']['prefix_normalization']['gc_mtf_history']
    old = groups.compile_gc_mtf_entry_execution_profile(history)
    new = groups.compile_gc_mtf_geometry_entry_execution_profile(history)
    assert new['name'] == groups.GC_MTF_GEOMETRY_ENTRY_EXECUTION_PROFILE_NAME
    assert new['local_specialists'] == new['mtf_specialists'] == tuple(groups.MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert len(new['local_specialists']) == 8
    assert len(set(new['signal_indices']) - set(old['signal_indices'])) == 32
    assert len(set(new['ctx_cont_indices']) - set(old['ctx_cont_indices'])) == 1
    assert len(set(new['mtf_indices']) - set(old['mtf_indices'])) == 39
    for field in ('signal_indices', 'ctx_cont_indices', 'mtf_indices'):
        assert set(old[field]).issubset(new[field])
    assert new['gc_mtf_history'] == old['gc_mtf_history']
    ref = {'name': new['name'], 'contract_sha256': new['contract_sha256']}
    assert groups.require_compact_entry_execution_profile_reference(ref, gc_mtf_history=history) == ref
    contract = architecture.entry_exit_production_architecture_contract(entry_execution_profile=ref, gc_mtf_history=history)
    assert contract['gc_mtf_history']['mtf_specialists'] == list(groups.MODEL_NATIVE_TRAINING_SPECIALISTS)
    arguments = deepcopy(bound_case['arguments']); arguments['source_bindings'].update({key: _ref(key + '.py') for key in owner.DENSE_SOURCE_REF_KEYS})
    view = owner.bind_native_direction_feature_view(**arguments, dense_minutes=True)
    assert view['entry_execution_profile'] == ref


def test_five_priority_price_only_view_binds_distinct_objective_and_same_routes(bound_case, monkeypatch):
    from gx1.contracts import entry_native_direction_five_priority_v1 as candidate
    args=deepcopy(bound_case['arguments'])
    args['source_bindings'].update({key:_ref(key+'.py') for key in owner.DENSE_SOURCE_REF_KEYS | owner.PRICE_MOMENTUM_SOURCE_REF_KEYS})
    old=owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True)
    args['source_bindings'].update({key:_ref(key+'.py') for key in owner.FIVE_PRIORITY_SOURCE_REF_KEYS})
    view=owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,five_minute_priority=True)
    assert view['five_minute_priority'] is True and view['research_objective']==candidate.five_priority_direction_contract()
    assert view['entry_execution_profile']==old['entry_execution_profile']
    assert view['physical_field_names_sha256']==old['physical_field_names_sha256']
    calls,norm_calls=_constructor_spy(monkeypatch)
    model=owner.build_native_direction_research_model(view,prefix_normalization=args['prefix_normalization'],constructor_metadata=args['constructor_metadata'],device=torch.device('cpu'))
    assert model.five_minute_priority and model.research_objective==view['research_objective'] and len(calls)==len(norm_calls)==1
    assert model.feature_view_identity['research_objective_sha256']==view['research_objective']['contract_sha256']
    with pytest.raises(RuntimeError):owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=False,five_minute_priority=True)
    with pytest.raises(RuntimeError):owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_normalization={},five_minute_priority=True)
    broken=deepcopy(args);broken['source_bindings'].pop('five_priority_source')
    with pytest.raises(RuntimeError):owner.bind_native_direction_feature_view(**broken,dense_minutes=True,price_momentum=True,five_minute_priority=True)
