"""Exact metadata/name remapping only; no arrays, statistics fit or native model."""
from copy import deepcopy

import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.features import entry_specialist_feature_groups_v1 as groups


@pytest.fixture(scope='module')
def physical():
    return groups.compile_full_ta_entry_execution_profile()['source_field_names']


def reseal(profile):
    profile['contract_sha256'] = units._sha({k: v for k, v in profile.items()
                                           if k != 'contract_sha256'})
    return profile


def test_old_profile_identities_remain_exact_and_new_profile_has_explicit_parent():
    october = units.october_prefix_units_profile()
    parent = units.mature_fold1_prefix_units_profile()
    before = units._canonical(parent)
    profile = units.anchor_span_fold1_prefix_units_profile()
    assert october['contract_sha256'] == '40d2c04dd57623cf90ab51ece742b4c94a9b156457e0dd65ea52c2ea0b7786c7'
    assert parent['contract_sha256'] == '8fe7982aaf6ef8de947997cc945669cb58e94c6729282a38f57e7a967d3013bc'
    assert profile['contract_sha256'] not in (october['contract_sha256'], parent['contract_sha256'])
    for key in ('plan', 'fields', 'fit_prefix', 'support_evidence', 'unobserved_semantics', 'authority'):
        assert units._canonical(profile[key]) == units._canonical(parent[key])
    for surface in ('ctx_cont', 'gc_mtf_d1'):
        assert units._canonical(profile['surfaces'][surface]) == units._canonical(parent['surfaces'][surface])
    provenance = profile['extension_provenance']
    assert provenance['parent_profile_name'] == parent['profile_name']
    assert provenance['parent_profile_sha256'] == parent['contract_sha256']
    assert provenance['physical_input_result'] == {
        'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json',
        'sha256': '8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb',
    }
    assert provenance['inserted_signal_fields'] == list(units.ANCHOR_SPAN_SIGNAL_FIELDS)
    assert provenance['new_observation_masks'] is provenance['statistics_fitted'] is False
    assert all(value is False for value in profile['authority'].values())
    assert units.require_prefix_units_profile(profile) == profile
    assert units._canonical(parent) == before
    profile['fields'][0]['fixed_scale'] = 999.
    assert units._canonical(units.mature_fold1_prefix_units_profile()) == before
    assert units.anchor_span_fold1_prefix_units_profile()['fields'][0]['fixed_scale'] == 1.


def test_signal_units_follow_exact_names_and_new_spans_receive_no_mask_or_unit(physical):
    profile = units.anchor_span_fold1_prefix_units_profile()
    parent = units.mature_fold1_prefix_units_profile()
    names = list(physical['signal'])
    old_names = [name for name in names if name not in units.ANCHOR_SPAN_SIGNAL_FIELDS]
    assert len(names) == 2727 and len(old_names) == 2725
    assert units._sha(names) == '99f626ba2a99113627ccc114b8635e6d984e344bf48d759f7058c2184c5d16db'
    assert units._sha(old_names) == parent['surfaces']['signal']['field_names_sha256']
    old = units.prefix_units_surface_columns(parent, surface='signal', field_names=old_names)
    new = units.prefix_units_surface_columns(profile, surface='signal', field_names=names)
    assert len(old) == len(new) == len(profile['fields']) == 51
    for index, spec in old.items():
        mapped = names.index(old_names[index])
        assert mapped == index + 2
        assert new[mapped] == {**spec, 'mask_index': names.index(spec['mask'])}
        assert new[mapped]['mask_index'] == spec['mask_index'] + 2
    for name in units.ANCHOR_SPAN_SIGNAL_FIELDS:
        assert names.index(name) not in new
        assert name not in {entry['field'] for entry in profile['fields']}
    assert units.prefix_units_surface_columns(profile, surface='mtf_m5', field_names=physical['mtf']) == {}
    with pytest.raises(RuntimeError, match='FIELD_NAMES'):
        units.prefix_units_surface_columns(parent, surface='signal', field_names=names)
    with pytest.raises(RuntimeError, match='FIELD_NAMES'):
        units.prefix_units_surface_columns(profile, surface='signal', field_names=old_names)


@pytest.mark.parametrize('surface', ('ctx_cont', 'gc_mtf_d1'))
def test_unchanged_surfaces_keep_identical_routing(surface, physical):
    names = history.gc_mtf_history_layout()['field_names']['D1'] if surface == 'gc_mtf_d1' else physical[surface]
    parent = units.mature_fold1_prefix_units_profile()
    profile = units.anchor_span_fold1_prefix_units_profile()
    assert len(names) == (399 if surface == 'gc_mtf_d1' else 2542)
    assert units.prefix_units_surface_columns(profile, surface=surface, field_names=names) == \
        units.prefix_units_surface_columns(parent, surface=surface, field_names=names)


@pytest.mark.parametrize('fault', ('parent', 'physical_result', 'inserted_field', 'unit',
                                  'value_index', 'mask_index', 'fit', 'support', 'authority', 'extra'))
def test_modified_and_resealed_extension_is_rejected(fault):
    profile = deepcopy(units.anchor_span_fold1_prefix_units_profile())
    if fault == 'parent':
        profile['extension_provenance']['parent_profile_sha256'] = '0' * 64
    elif fault == 'physical_result':
        profile['extension_provenance']['physical_input_result']['sha256'] = '0' * 64
    elif fault == 'inserted_field':
        profile['extension_provenance']['inserted_signal_fields'].reverse()
    elif fault == 'unit':
        profile['fields'][0]['fixed_scale'] = 2.
    elif fault in ('value_index', 'mask_index'):
        profile['surfaces']['signal']['columns'][0][fault] -= 2
    elif fault == 'fit':
        profile['fit_prefix']['rows'] += 1
    elif fault == 'support':
        profile['support_evidence']['sha256'] = '0' * 64
    elif fault == 'authority':
        profile['authority']['training_enabled'] = True
    else:
        profile['unrecognized_extension'] = True
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        units.require_prefix_units_profile(reseal(profile))


@pytest.mark.parametrize('fault', ('missing', 'duplicate', 'reorder_old'))
def test_current_canonical_surface_must_be_only_the_two_exact_insertions(monkeypatch, physical, fault):
    names = list(physical['signal'])
    if fault == 'missing':
        names.remove(units.ANCHOR_SPAN_SIGNAL_FIELDS[0])
    elif fault == 'duplicate':
        names[0] = units.ANCHOR_SPAN_SIGNAL_FIELDS[0]
    else:
        names[0], names[1] = names[1], names[0]
    monkeypatch.setattr(groups, 'compile_full_ta_entry_execution_profile',
                        lambda: {'source_field_names': {'signal': names}})
    with pytest.raises(RuntimeError, match='ANCHOR_SPAN_SIGNAL_NAMES'):
        units.anchor_span_fold1_prefix_units_profile()


def test_extension_requires_exact_parent_even_if_parent_is_resealed(monkeypatch):
    parent = deepcopy(units.mature_fold1_prefix_units_profile())
    parent['surfaces']['ctx_cont']['columns'][0]['mask_index'] += 1
    monkeypatch.setattr(units, 'mature_fold1_prefix_units_profile', lambda: reseal(parent))
    with pytest.raises(RuntimeError, match='ANCHOR_SPAN_PARENT'):
        units.anchor_span_fold1_prefix_units_profile()


def test_fit_prefix_is_preserved_but_profile_identities_cannot_be_substituted():
    parent = units.mature_fold1_prefix_units_profile()
    profile = units.anchor_span_fold1_prefix_units_profile()
    assert units.require_prefix_units_fit_identity(parent['fit_prefix'], profile) == parent['fit_prefix']
    bad = {**profile['fit_prefix'], 'rows': True}
    with pytest.raises(RuntimeError, match='FIT_IDENTITY'):
        units.require_prefix_units_fit_identity(bad, profile)
    for require, value in ((units.require_mature_fold1_prefix_units_profile, profile),
                           (units.require_october_prefix_units_profile, profile),
                           (units.require_anchor_span_fold1_prefix_units_profile, parent)):
        with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
            require(value)
