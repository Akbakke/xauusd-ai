"""Synthetic routing metadata and Python shells; no fitted stats or native model.

Normalization validators are deliberate spies here. Their numerical contracts
have separate tests; these checks cover the explicit profile join and raw gather.
"""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_exit_production_architecture_v1 as architecture
from gx1.contracts import entry_native_direction_feature_view_v1 as view
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.features import entry_direction_momentum_profile_v1 as momentum
from gx1.features import entry_specialist_feature_groups_v1 as groups
from tests.test_entry_native_direction_feature_view_v1 import _metadata, _inputs, _ref, _seal
from tests.test_entry_native_direction_prefix_normalization_v1 import prefix_case, _hash, _sync_current


@pytest.fixture
def delegates(monkeypatch):
    profile = units.october_prefix_units_profile()
    layout = view.history.gc_mtf_history_layout()
    legacy = {'layout': layout, 'normalization': {'contract_sha256': '1' * 64}}
    marked = {'layout': layout, 'normalization': {'contract_sha256': '2' * 64}}
    calls = []

    def require_history(value, **kwargs):
        supplied = kwargs.get('unobserved_units_profile')
        calls.append(supplied)
        if value['normalization']['contract_sha256'] == '2' * 64:
            if supplied != profile:
                raise RuntimeError('SYNTHETIC_EXPLICIT_PROFILE_REQUIRED')
        else:
            assert supplied is None
        return value

    monkeypatch.setattr(view.history, 'require_gc_mtf_history', require_history)
    return profile, legacy, marked, calls


@pytest.mark.parametrize('compiler', [groups.compile_gc_mtf_entry_execution_profile,
    groups.compile_gc_mtf_geometry_entry_execution_profile, groups.compile_full_gc_mtf_entry_execution_profile,
    momentum.compile_price_momentum_profile])
def test_compilers_and_reference_delegates_preserve_widths_and_require_explicit_profile(delegates, compiler):
    profile, legacy, marked, calls = delegates
    previous = compiler(legacy)
    assert compiler(legacy, unobserved_units_profile=None) == previous
    calls.clear()
    current = compiler(marked, unobserved_units_profile=profile)
    assert calls and all(item == profile for item in calls)
    assert current['physical_field_names'] == previous['physical_field_names']
    assert current['signal_indices'] == previous['signal_indices']
    assert current['gc_mtf_widths'] == previous['gc_mtf_widths']
    assert current['contract_sha256'] != previous['contract_sha256']
    reference = {key: current[key] for key in ('name', 'contract_sha256')}
    assert momentum.require_entry_execution_profile_reference(reference, gc_mtf_history=marked,
        unobserved_units_profile=profile) == reference
    with pytest.raises(RuntimeError, match='EXPLICIT_PROFILE_REQUIRED'):
        momentum.require_entry_execution_profile_reference(reference, gc_mtf_history=marked)
    with pytest.raises(RuntimeError, match='EXECUTION_PROFILE_REQUIRED'):
        groups.require_compact_entry_execution_profile_reference(None, unobserved_units_profile=profile)


def test_architecture_threads_the_explicit_profile_at_each_delegate(delegates):
    profile, legacy, marked, calls = delegates
    compiled = groups.compile_full_gc_mtf_entry_execution_profile(marked, unobserved_units_profile=profile)
    reference = {key: compiled[key] for key in ('name', 'contract_sha256')}
    kwargs = {'entry_execution_profile': reference, 'gc_mtf_history': marked, 'unobserved_units_profile': profile}
    expected = architecture.entry_exit_production_architecture_contract(**kwargs)
    assert architecture.current_entry_exit_architecture_observation(**kwargs) == expected
    assert architecture.require_entry_exit_production_architecture(expected, context='SYNTHETIC',
        gc_mtf_history=marked, unobserved_units_profile=profile) == expected
    assert calls and all(item == profile for item in calls)
    with pytest.raises(RuntimeError, match='EXPLICIT_PROFILE_REQUIRED'):
        architecture.entry_exit_production_architecture_contract(entry_execution_profile=reference, gc_mtf_history=marked)
    assert architecture.entry_exit_production_architecture_contract() == architecture.entry_exit_production_architecture_contract(
        unobserved_units_profile=None)


@pytest.fixture
def binding_case(monkeypatch):
    case = prefix_case()
    original_loader = units.october_prefix_units_profile
    immutable = original_loader()
    profile = deepcopy(immutable)
    stop = len(case['expected_prefix_parent_rows'])
    prefix = {'rows': stop, 'cutoff_ns': case['cutoff_ns'], **{
        key + '_sha256': _hash(case[key][:stop]) for key in ('parent_rows', 'bar_start_ns', 'decision_ns')}}
    profile['fit_prefix'] = prefix
    profile = _seal(profile)
    monkeypatch.setattr(units, 'october_prefix_units_profile', lambda: deepcopy(profile))
    # Only synthetic inputs: preserve zero/unknown in the fit prefix and a later
    # observed value. No fitted statistic or qualification is claimed by this fixture.
    signal_columns = profile['surfaces']['signal']['columns']
    last = int(case['entry_sequence_source']['sample_positions'][stop - 1])
    for column in signal_columns:
        case['entry_sequence_source']['signal'][:last + 1, [column['value_index'], column['mask_index']]] = 0.
    column = signal_columns[3]
    position = int(case['entry_sequence_source']['sample_positions'][4])
    case['entry_sequence_source']['signal'][position, column['value_index']] = 73.
    case['entry_sequence_source']['signal'][position, column['mask_index']] = 1.
    _sync_current(case)
    for column in profile['surfaces']['gc_mtf_d1']['columns']:
        selected = case['gc_surfaces']['D1']['available_ns'] < prefix['cutoff_ns']
        case['gc_surfaces']['D1']['features'][selected, column['value_index']] = 0.
        case['gc_surfaces']['D1']['features'][selected, column['mask_index']] = 0.
    sources = {key: _ref(key + '.json') for key in view.SOURCE_REF_KEYS | view.DYNAMIC_STATE_SOURCE_REF_KEYS}
    provenance = {'ancestral264_plan': sources['raw_input_provider_plan'],
        'current_gc_layout': sources['raw_gc_layout'], 'current_gc_raw_transition': sources['raw_gc_transition'],
        'current_source_owners': {path: sources[key] for key, path in (
            ('mtf_window_source', 'gx1/features/htf_features.py'),
            ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
            ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py'))}}
    lineage = {'train_row_count': stop, 'context_fit_row_count': stop, 'local_fit_row_count': 100}
    unions = {tf: {'rows': 1, 'native_row_indices_sha256': '3' * 64, 'available_ns_sha256': '4' * 64}
        for tf in view.history.TIMEFRAMES}
    base = {'contract_sha256': '5' * 64, 'lineage': lineage, 'surfaces': {}}
    gc_history = {'layout': case['gc_layout'], 'normalization': {'contract_sha256': '6' * 64, 'surfaces': {},
        'provenance': {'train_end_ns': prefix['cutoff_ns'], 'row_union': unions}}}
    report = _seal({'fit_allowed': True, 'blockers': [], 'statistics_fits': 0,
        'unobserved_units_profile': profile, 'proof': {'prefix': prefix, 'full_axes': case['expected_axes'],
            'local': {'selected_unique_row_count': 100}, 'base_lineage': lineage, 'gc': unions, 'source_bindings': provenance}})
    fitted = _seal({'schema_version': 'gx1_native_direction_prefix_normalization_v1', 'base_normalization': base,
        'gc_mtf_history': gc_history, 'prefix_qualification': report, 'unobserved_units_profile': profile,
        'old055_or223_normalization_identity_reused': False, 'native_launch_authorized': False,
        'genuine_model_fit_authorized': False})
    calls = []

    def require_component(value, **kwargs):
        assert kwargs.get('unobserved_units_profile') == profile
        calls.append(('component', deepcopy(kwargs['unobserved_units_profile'])))
        return value

    def require_fitted(value, supplied):
        assert value == fitted and supplied == profile
        units.require_prefix_units_fit_identity(value['prefix_qualification']['proof']['prefix'], supplied)
        calls.append(('fitted', deepcopy(supplied)))
        return value

    monkeypatch.setattr(view.norm, 'require_input_normalization_contract', require_component)
    monkeypatch.setattr(view.history, 'require_gc_mtf_history', require_component)
    monkeypatch.setattr(view.prefix_owner, 'require_october_units_normalization_bindings', require_fitted)
    arguments = {'prefix_normalization': fitted, 'prefix_normalization_ref': _ref('synthetic-prefix.json'),
        'constructor_metadata': _metadata(), 'constructor_metadata_ref': _ref('constructor.json'),
        'source_bindings': sources, 'max_batch_rows': 2, 'full_feature_profile': True,
        'dynamic_state_supervision': True, 'current_state_first_loss_weight': .03,
        'first_flat_activity_weight': .03, 'native_endpoint_loss': True, 'unobserved_units_profile': profile}
    bound = view.bind_native_direction_feature_view(**arguments)
    return {'case': case, 'arguments': arguments, 'view': bound, 'profile': profile, 'immutable': immutable,
        'original_loader': original_loader, 'calls': calls}


def test_binding_copies_and_rebinds_full_profile_without_implicit_authority(binding_case):
    bound, args, profile = (binding_case[key] for key in ('view', 'arguments', 'profile'))
    assert bound['unobserved_units_profile'] == profile and bound['unobserved_units_profile'] is not profile
    assert bound['prefix_normalization']['unobserved_units_profile_sha256'] == profile['contract_sha256']
    assert view.bind_native_direction_feature_view(**args, expected=bound) == bound
    assert view._require_view(bound, unobserved_units_profile=profile) == bound
    assert any(kind == 'fitted' for kind, _ in binding_case['calls'])
    assert all(item == profile for _, item in binding_case['calls'])
    with pytest.raises(RuntimeError, match='EXPLICIT_UNOBSERVED_UNITS_PROFILE'):
        view._require_view(bound)
    without = {key: value for key, value in args.items() if key != 'unobserved_units_profile'}
    with pytest.raises(RuntimeError, match='PREFIX_NORMALIZATION_SCOPE'):
        view.bind_native_direction_feature_view(**without)
    with pytest.raises(RuntimeError, match='PROFILE_MIX'):
        view.bind_native_direction_feature_view(**args, zero_support_profile={})


@pytest.mark.parametrize('key', ['full_feature_profile', 'dynamic_state_supervision', 'native_endpoint_loss'])
def test_units_binding_requires_the_current_full_dynamic_v5_scope(binding_case, key):
    with pytest.raises(RuntimeError, match='UNOBSERVED_UNITS_FIT_BINDING'):
        view.bind_native_direction_feature_view(**{**binding_case['arguments'], key: False})


def test_resealed_markers_and_real_profile_on_small_population_fail(binding_case, monkeypatch):
    profile, bound = binding_case['profile'], deepcopy(binding_case['view'])
    bound['prefix_normalization']['unobserved_units_profile_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='UNOBSERVED_UNITS_VIEW_BINDING'):
        view._require_view(_seal(bound), unobserved_units_profile=profile)
    bound = deepcopy(binding_case['view'])
    del bound['unobserved_units_profile']
    with pytest.raises(RuntimeError, match='UNOBSERVED_UNITS_VIEW_BINDING'):
        view._require_view(_seal(bound))
    bound = deepcopy(binding_case['view'])
    immutable = binding_case['immutable']
    bound['unobserved_units_profile'] = immutable
    bound['prefix_normalization']['unobserved_units_profile_sha256'] = immutable['contract_sha256']
    monkeypatch.setattr(units, 'october_prefix_units_profile', binding_case['original_loader'])
    with pytest.raises(RuntimeError, match='FIT_IDENTITY'):
        view._require_view(_seal(bound), unobserved_units_profile=immutable)


def test_confirmed_age_view_binds_exact_october_profile_and_candidate_source(binding_case):
    args = deepcopy(binding_case['arguments'])
    control = binding_case['view']
    args['source_bindings']['confirmed_age_candidate_design'] = _ref('confirmed-age-candidate.json')
    args['confirmed_run_age_input'] = True
    candidate = view.bind_native_direction_feature_view(**args)
    assert candidate['confirmed_run_age_input'] is True
    assert candidate['prefix_normalization'] == control['prefix_normalization']
    assert candidate['physical_field_names_sha256'] == control['physical_field_names_sha256']
    assert candidate['entry_execution_profile'] == control['entry_execution_profile']
    assert candidate['research_objective']['contract_sha256'] != control['research_objective']['contract_sha256']
    assert view._require_view(candidate, unobserved_units_profile=binding_case['profile']) == candidate
    with pytest.raises(RuntimeError, match='SOURCE_REF_KEYS'):
        view.bind_native_direction_feature_view(**{**args,
            'source_bindings': binding_case['arguments']['source_bindings']})
    with pytest.raises(RuntimeError, match='UNOBSERVED_UNITS_FIT_BINDING'):
        view.bind_native_direction_feature_view(**{**args, 'native_endpoint_loss': False})
    with pytest.raises(RuntimeError, match='SOURCE_REF_KEYS'):
        view.bind_native_direction_feature_view(**{**args, 'confirmed_run_age_input': False})


def test_raw_gather_requires_opt_in_and_preserves_later_values_and_masks(binding_case):
    bound, profile = binding_case['view'], binding_case['profile']
    arguments = _inputs(binding_case)
    with pytest.raises(RuntimeError, match='EXPLICIT_UNOBSERVED_UNITS_PROFILE'):
        view.gather_native_direction_feature_batch(bound, **arguments)
    result = view.gather_native_direction_feature_batch(bound, **arguments, unobserved_units_profile=profile)
    features = result['features']
    np.testing.assert_array_equal(features['snap_x'], arguments['snap_x'][[1, 4]])
    np.testing.assert_array_equal(features['ctx_cont'], arguments['ctx_cont'][[1, 4]])
    column = profile['surfaces']['signal']['columns'][3]
    assert features['snap_x'][1, column['value_index']] == 73.
    assert features['snap_x'][1, column['mask_index']] == 1.
    for i, position in enumerate(result['witness']['source_positions']):
        np.testing.assert_array_equal(features['seq_x'][i], arguments['entry_sequence_source']['signal'][position - 95:position + 1])
    for tf in view.history.TIMEFRAMES:
        for i, witness in enumerate(result['witness']['multi_tf'][tf]):
            np.testing.assert_array_equal(features['gc_seq_' + tf.lower()][i],
                arguments['gc_surfaces'][tf]['features'][witness['native_row_indices']])
    assert result['witness']['raw_values_not_normalized'] is True


def test_october_qualification_provenance_binds_its_own_source_plan(binding_case, monkeypatch):
    args = deepcopy(binding_case['arguments'])
    proof = args['prefix_normalization']['prefix_qualification']['proof']
    plan = _ref('october-input-plan.json')
    proof['source_bindings'] = {'ancestral_264_plan': args['source_bindings']['raw_input_provider_plan'],
        'ancestral_267_plan': _ref('ancestral-267.json'), 'current_plan': plan,
        'prior_october_support': _ref('october-support.json')}
    args['prefix_normalization']['prefix_qualification'] = _seal(
        args['prefix_normalization']['prefix_qualification'])
    args['prefix_normalization'] = _seal(args['prefix_normalization'])
    args['source_bindings']['october_input_plan'] = plan
    args['source_bindings']['input_qualification'] = _ref('QUALIFICATION.json')
    monkeypatch.setattr(view.prefix_owner, 'require_october_units_normalization_bindings',
        lambda fitted, profile: units.require_prefix_units_fit_identity(
            fitted['prefix_qualification']['proof']['prefix'], profile))
    bound = view.bind_native_direction_feature_view(**args)
    assert bound['source_bindings']['october_input_plan'] == plan
    assert view._require_view(bound, unobserved_units_profile=binding_case['profile']) == bound
    wrong = deepcopy(args)
    wrong['source_bindings']['october_input_plan'] = _ref('wrong-plan.json')
    with pytest.raises(RuntimeError, match='OCTOBER_SOURCE_PROVENANCE_JOINS'):
        view.bind_native_direction_feature_view(**wrong)
    missing = deepcopy(args)
    del missing['source_bindings']['october_input_plan']
    with pytest.raises(RuntimeError, match='SOURCE_REF_KEYS'):
        view.bind_native_direction_feature_view(**missing)


def test_scoped_october_axes_allow_only_bound_fit_or_development_rows(binding_case):
    case, profile = binding_case['case'], binding_case['profile']
    n = binding_case['view']['prefix']['rows']
    names = ('parent_rows', 'bar_start_ns', 'decision_ns')
    hashes = lambda left, right: {name + '_sha256': _hash(case[name][left:right]) for name in names}
    scope = {'source': _ref('synthetic-october-axis.json'), 'rows': len(case['parent_rows']),
             **hashes(0, len(case['parent_rows'])),
             'fit': {'rows': n, 'cutoff_ns': int(case['cutoff_ns']), **hashes(0, n)},
             'held_out': {'rows': len(case['parent_rows']) - n,
                          'cutoff_ns': int(case['decision_ns'][-1] + view.BAR_NS),
                          **hashes(n, len(case['parent_rows']))}}
    control_args = deepcopy(binding_case['arguments'])
    control_args['october_observer_axes'] = scope
    control = view.bind_native_direction_feature_view(**control_args)
    candidate_args = deepcopy(control_args)
    candidate_args['source_bindings']['confirmed_age_candidate_design'] = _ref('age-design.json')
    candidate_args['confirmed_run_age_input'] = True
    candidate = view.bind_native_direction_feature_view(**candidate_args)
    assert candidate['full_axes']['rows'] == control['full_axes']['rows'] == len(case['parent_rows'])
    assert candidate['october_observer_axes'] == control['october_observer_axes'] == scope
    assert candidate['prefix_normalization']['contract_sha256'] == control['prefix_normalization']['contract_sha256']
    inputs = _inputs(binding_case)
    for name in names:
        inputs[name] = case[name][n:]
    for name in ('snap_x', 'ctx_cont', 'ctx_cat'):
        inputs[name] = case[name][n:]
    source = inputs['entry_sequence_source']
    stop = int(source['sample_positions'][-1]) + 1
    inputs['entry_sequence_source'] = {'signal': source['signal'][:stop],
        'times_ns': source['times_ns'][:stop], 'sample_positions': source['sample_positions'][n:],
        'audit': source['audit']}
    inputs['batch_parent_rows'] = case['parent_rows'][n:].copy()
    with pytest.raises(RuntimeError, match='BOUND_OBSERVER_AXES_REQUIRED'):
        view.gather_native_direction_feature_batch(candidate, **inputs,
            unobserved_units_profile=profile)
    for bound in (control, candidate):
        observer = view.NativeDirectionObserverAxes(bound, source=scope['source'],
            parent_rows=inputs['parent_rows'], bar_start_ns=inputs['bar_start_ns'],
            decision_ns=inputs['decision_ns'], role='score',
            unobserved_units_profile=profile)
        gathered = view.gather_native_direction_feature_batch(bound, **{
            **inputs, 'parent_rows': observer.parent_rows,
            'bar_start_ns': observer.bar_start_ns, 'decision_ns': observer.decision_ns,
            'observer_axes': observer}, unobserved_units_profile=profile)
        assert gathered['witness']['parent_rows'] == inputs['batch_parent_rows'].tolist()
        assert len(gathered['features']) == 14
    bad = inputs['decision_ns'].copy()
    bad[-1] += view.BAR_NS
    with pytest.raises(RuntimeError, match='OBSERVER_FIT_AXIS'):
        view.NativeDirectionObserverAxes(candidate, source=scope['source'],
            parent_rows=inputs['parent_rows'], bar_start_ns=inputs['bar_start_ns'],
            decision_ns=bad, role='score', unobserved_units_profile=profile)


def test_factory_rebinds_explicit_profile_before_forwarding_to_python_shell(binding_case, monkeypatch):
    calls = []

    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.entry_execution_profile = kwargs['entry_execution_profile']
            self.gc_mtf_history = kwargs['gc_mtf_history']
        def to(self, device):
            return self
        def require_input_normalization_state(self):
            calls.append('runtime-normalization')

    class WrapperShell:
        def __init__(self, backbone, **kwargs):
            self.backbone = backbone
            self.kwargs = kwargs

    monkeypatch.setattr(view, 'EntryV10CtxHybridTransformer', BackboneShell)
    monkeypatch.setattr(view, 'NativeDynamicMTFStateResearch', WrapperShell)
    bound, args, profile = (binding_case[key] for key in ('view', 'arguments', 'profile'))
    factory_args = {key: args[key] for key in ('prefix_normalization', 'constructor_metadata')}
    with pytest.raises(RuntimeError, match='PREFIX_NORMALIZATION_SCOPE'):
        view.build_native_direction_research_model(bound, **factory_args, device='synthetic-shell')
    assert calls == []
    result = view.build_native_direction_research_model(bound, **factory_args, device='synthetic-shell',
        unobserved_units_profile=profile)
    assert calls[0]['research_unobserved_units_profile'] == profile
    assert calls[0]['research_unobserved_units_profile'] is not profile
    assert 'zero_support_profile' not in calls[0]
    assert calls[1] == 'runtime-normalization'
    assert result.kwargs['native_endpoint_loss'] is True
    assert result.feature_view_identity['contract_sha256'] == bound['contract_sha256']
