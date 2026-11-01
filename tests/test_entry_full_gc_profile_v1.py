"""Full input metadata and raw gathers; no native model or market payload."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gx1.contracts import entry_exit_production_architecture_v1 as architecture
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
from gx1.features import entry_specialist_feature_groups_v1 as profiles
from gx1.features import entry_direction_momentum_profile_v1 as momentum
from tests.test_entry_native_direction_feature_view_v1 import bound_case, _inputs, _seal


def _ref(profile):
    return {key: profile[key] for key in ('name', 'contract_sha256')}


def test_full_profile_has_no_parked_field_and_exact_family_tf_coverage(bound_case, record_property):
    history = bound_case['arguments']['prefix_normalization']['gc_mtf_history']
    before = deepcopy(history)
    full = profiles.compile_full_gc_mtf_entry_execution_profile(history)
    fields = full['physical_field_names']
    families = tuple(profiles.MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert history == before
    assert full['active_field_names'] == fields
    assert full['timeframes'] == tuple(history['layout']['timeframes'])
    assert full['local_specialists'] == full['base_mtf_specialists'] == full['mtf_specialists'] == families
    for surface, indices, routing in (
        ('signal', 'signal_indices', 'local_signal_indices_by_specialist'),
        ('ctx_cont', 'ctx_cont_indices', 'ctx_cont_indices_by_specialist'),
        ('mtf', 'mtf_indices', 'mtf_indices_by_specialist'),
    ):
        assert full[indices] == tuple(range(len(fields[surface])))
        assert sorted(i for values in full[routing].values() for i in values) == list(range(len(fields[surface])))
    assert full['mtf_indices_by_specialist'] == profiles.require_multi_tf_specialist_routing_v4(fields['mtf'])
    for tf in full['timeframes']:
        assert full['gc_mtf_field_names'][tf] == history['layout']['field_names'][tf]
        assert full['gc_mtf_widths'][tf] == history['layout']['widths'][tf]
        assert full['gc_mtf_window_rows'][tf] == history['layout']['window_rows'][tf]
        assert sorted(i for values in full['gc_mtf_family_indices'][tf].values() for i in values) == list(range(full['gc_mtf_widths'][tf]))
    assert set(full['macro_context_fields']) <= set(fields['ctx_cont'])
    assert set(full['gc_context_fields']) <= set(fields['ctx_cont'])
    aliases = set(profiles.model_native_context_temporal_alias_policy(fields['signal'])['signal_indices'])
    assert full['generic_snap_signal_indices'] == tuple(i for i in full['signal_indices'] if i not in aliases)
    price = full['mtf_indices_by_specialist'][momentum.MOMENTUM_FAMILY]
    for name in momentum.PRICE_MOMENTUM_FIELDS:
        assert price.count(fields['mtf'].index(name)) == 1
    assert 'price_momentum_initialization' not in full
    record_property('full_profile_mapping_json', json.dumps({
        'physical_widths': full['physical_widths'], 'timeframes': full['timeframes'],
        'parked_fields': {key: 0 for key in fields},
        'local_family_counts': {key: len(value) for key, value in full['local_signal_indices_by_specialist'].items()},
        'xau_per_tf_family_counts': {key: len(value) for key, value in full['mtf_indices_by_specialist'].items()},
        'gc_per_tf_widths': full['gc_mtf_widths'],
        'gc_per_tf_family_counts': {tf: {key: len(value) for key, value in routes.items()}
            for tf, routes in full['gc_mtf_family_indices'].items()},
    }, sort_keys=True))


def test_full_compilation_preserves_all_legacy_profiles_and_requires_its_exact_binding(bound_case):
    history = bound_case['arguments']['prefix_normalization']['gc_mtf_history']
    builders = (profiles.compile_gc_mtf_entry_execution_profile,
        profiles.compile_gc_mtf_geometry_entry_execution_profile, momentum.compile_price_momentum_profile)
    before = [builder(history) for builder in builders]
    full = profiles.compile_full_gc_mtf_entry_execution_profile(history)
    assert [builder(history) for builder in builders] == before
    reference = _ref(full)
    assert momentum.require_entry_execution_profile_reference(reference, gc_mtf_history=history) == reference
    assert profiles.require_compact_entry_execution_profile_reference(None) is None
    with pytest.raises(RuntimeError):
        profiles.require_compact_entry_execution_profile_reference(None, gc_mtf_history=history)
    with pytest.raises(RuntimeError):
        profiles.require_compact_entry_execution_profile_reference(reference)
    with pytest.raises(RuntimeError):
        profiles.require_compact_entry_execution_profile_reference({**reference, 'contract_sha256': '0' * 64}, gc_mtf_history=history)
    with pytest.raises(RuntimeError):
        profiles.require_compact_entry_execution_profile_reference({**reference, 'name': reference['name'] + '_unknown'}, gc_mtf_history=history)
    observed = architecture.entry_exit_production_architecture_contract(entry_execution_profile=reference, gc_mtf_history=history)
    assert observed['entry_execution_profile'] == reference
    assert observed['gc_mtf_history']['per_tf_widths'] == full['gc_mtf_widths']
    assert observed['gc_mtf_history']['mtf_specialists'] == list(full['mtf_specialists'])
    assert observed['mtf']['per_tf_widths'] == {tf: len(full['physical_field_names']['mtf']) for tf in full['timeframes']}


def test_full_view_reuses_complete_physical_prefix_statistics_without_refit(bound_case):
    args = bound_case['arguments']
    old = view_owner.bind_native_direction_feature_view(**args)
    full = view_owner.bind_native_direction_feature_view(**args, full_feature_profile=True)
    assert full['full_feature_profile'] is True
    assert full['entry_execution_profile']['name'] == profiles.FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
    assert full['prefix_normalization'] == old['prefix_normalization']
    assert full['physical_field_names_sha256'] == old['physical_field_names_sha256']
    assert full['canonical_routing_sha256'] == old['canonical_routing_sha256']
    assert full['research_objective'] == old['research_objective']
    assert 'gc_relative_coordinates_sha256' not in full
    assert 'price_momentum_initialization' not in full
    compiled = profiles.compile_full_gc_mtf_entry_execution_profile(args['prefix_normalization']['gc_mtf_history'])
    fields = compiled['physical_field_names']
    base = args['prefix_normalization']['base_normalization']
    for name, stats in base['surfaces'].items():
        surface = 'mtf' if name.startswith('mtf_') else name
        assert stats['field_names'] == list(fields[surface])
    assert view_owner.bind_native_direction_feature_view(**args, full_feature_profile=True, expected=full) == full


def test_full_gather_preserves_identical_raw_bytes_masks_aliases_and_all_tf_clocks(bound_case):
    bound = deepcopy(bound_case)
    old = bound['view']
    full = view_owner.bind_native_direction_feature_view(**bound['arguments'], full_feature_profile=True)
    inputs = _inputs(bound)
    a = view_owner.gather_native_direction_feature_batch(old, **inputs)
    b = view_owner.gather_native_direction_feature_batch(full, **inputs)
    assert set(a['features']) == set(b['features']) and len(b['features']) == 14
    for name in a['features']:
        assert a['features'][name].dtype == b['features'][name].dtype
        assert a['features'][name].tobytes() == b['features'][name].tobytes()
    assert {k: v for k, v in a['witness'].items() if k != 'view_sha256'} == {
        k: v for k, v in b['witness'].items() if k != 'view_sha256'}
    for tf, witnesses in b['witness']['multi_tf'].items():
        assert len(witnesses) == len(inputs['batch_parent_rows'])
        for row, witness in enumerate(witnesses):
            assert max(witness['available_ns']) <= b['witness']['decision_ns'][row]


@pytest.mark.parametrize('fault', ['missing_flag', 'false_flag', 'compact_name'])
def test_full_view_cannot_silently_fall_back_to_compact(bound_case, fault):
    value = view_owner.bind_native_direction_feature_view(**bound_case['arguments'], full_feature_profile=True)
    if fault == 'missing_flag':
        del value['full_feature_profile']
    elif fault == 'false_flag':
        value['full_feature_profile'] = False
    else:
        value['entry_execution_profile'] = bound_case['view']['entry_execution_profile']
    with pytest.raises(RuntimeError):
        view_owner._require_view(_seal(value))


@pytest.mark.parametrize('extra', [dict(full_feature_profile=1), dict(full_feature_profile='true'),
    dict(full_feature_profile=True, dense_minutes=True, price_momentum=True),
    dict(full_feature_profile=True, price_momentum_initialization='linear_default')])
def test_full_view_does_not_implicitly_mix_old_input_variants(bound_case, extra):
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**bound_case['arguments'], **extra)


def test_explicit_relative_history_keeps_all_original_fields_and_binds_extra_anchors():
    from tests.test_entry_gc_mtf_relative_normalization_v1 import _case, _fit
    case = _case()
    history = {'layout': case[0], 'normalization': _fit(case)}
    full = profiles.compile_full_gc_mtf_entry_execution_profile(history)
    for tf in full['timeframes']:
        raw = history['layout']['field_names'][tf]
        assert full['gc_mtf_field_names'][tf][:-2] == raw
        assert full['gc_mtf_field_names'][tf][-2:] == ['gc_relative_anchor_usd', 'gc_relative_anchor_observed']
        assert full['gc_mtf_widths'][tf] == len(raw) + 2
        assert sorted(i for values in full['gc_mtf_family_indices'][tf].values() for i in values) == list(range(len(raw) + 2))
    assert full['gc_mtf_history']['normalization_sha256'] == history['normalization']['contract_sha256']
