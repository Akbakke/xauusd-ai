"""Metadata/synthetic checks only; no market arrays, fitting or native model.

The optional full-wrapper fixture reads only the immutable e24 normalization
JSON. Its twelve new statistics are fabricated metadata with the declared row
counts; they do not qualify any new physical statistics.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def seal(value):
    value.pop('contract_sha256', None)
    value['contract_sha256'] = norm._canonical_sha256(value)
    return value


def statistics(surface, names, rows):
    """Fabricated continuous metadata, never a fitted physical statistic."""
    n = len(names)
    return extension._restamp_surface({
        'surface': surface, 'continuous_transform': norm.CONTINUOUS_TRANSFORM,
        'field_names': list(names), 'fit_row_count': rows,
        'center': [float(i + 2) for i in range(n)], 'scale': [float(i + 3) for i in range(n)],
        'binary_mask': [0] * n, 'categorical_mask': [0] * n,
        'categorical_domains': {}, 'scale_source': ['raw_iqr'] * n,
        'train_transformed_min': [-1.] * n, 'train_transformed_max': [2.] * n,
    })


def merge_case():
    old = statistics('signal', ['old_continuous', 'old_category'], 7)
    old.update(center=[-0., 0.], scale=[2., 1.], categorical_mask=[0, 1],
               categorical_domains={'old_category': [0, 1]},
               scale_source=['raw_iqr', 'categorical_embedding_identity'],
               train_transformed_min=[-1., 0.], train_transformed_max=[2., 1.],
               unobserved_units_profile_sha256='a' * 64, observed_fit_counts=[5, 7],
               unfitted_estimates={'synthetic_only': None})
    extension._restamp_surface(old)
    addition = statistics('signal', extension.span_names('signal'), 7)
    names = [old['field_names'][0], addition['field_names'][0],
             old['field_names'][1], addition['field_names'][1]]
    return old, addition, names


def test_pure_merge_preserves_old_bits_counts_domains_and_inputs_by_name():
    old, addition, names = merge_case()
    before = encoded([old, addition])
    merged = extension.merge_surface(old, addition, field_names=names, units_profile_sha='b' * 64)
    assert encoded([old, addition]) == before
    for key in extension.STAT_VECTORS:
        assert encoded([merged[key][0], merged[key][2]]) == encoded(old[key])
        assert encoded([merged[key][1], merged[key][3]]) == encoded(addition[key])
    assert merged['observed_fit_counts'] == [5, 7, 7, 7]
    assert merged['unfitted_estimates'] == old['unfitted_estimates']
    assert merged['unobserved_units_profile_sha256'] == 'b' * 64
    assert merged['categorical_domains'] == old['categorical_domains']
    assert merged['field_names_sha256'] == norm._field_names_sha256(names)
    assert merged['stats_sha256'] != old['stats_sha256']
    merged['center'][0] = 99.
    merged['categorical_domains']['old_category'].append(2)
    assert encoded([old, addition]) == before


@pytest.mark.parametrize('fault', ('duplicate', 'remove_old', 'reorder_old', 'unbound_extra'))
def test_merge_rejects_anything_beyond_the_ordered_name_addition(fault):
    old, addition, names = merge_case()
    if fault == 'duplicate': names.append(names[0])
    elif fault == 'remove_old': names.remove('old_category')
    elif fault == 'reorder_old': names[0], names[2] = names[2], names[0]
    else: names[1] = 'not_the_declared_addition'
    with pytest.raises(RuntimeError, match='EXACT_NAME_ADDITION'):
        extension.merge_surface(old, addition, field_names=names, units_profile_sha='b' * 64)


@pytest.mark.parametrize('fault', ('rows', 'surface'))
def test_merge_requires_the_same_declared_physical_population(fault):
    old, addition, names = merge_case()
    if fault == 'rows': addition['fit_row_count'] += 1
    else: addition['surface'] = 'mtf_m5'
    with pytest.raises(RuntimeError, match='SAME_PHYSICAL_POPULATION'):
        extension.merge_surface(old, addition, field_names=names, units_profile_sha='b' * 64)


@pytest.fixture(scope='module')
def metadata_case():
    path = Path(extension.PARENT_ARTIFACT['path'])
    if not path.is_file():
        pytest.skip('requires immutable e24 JSON metadata; no market payload is read')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == extension.PARENT_ARTIFACT['sha256']
    parent = json.loads(raw)['base_normalization']
    additions = {surface: statistics(surface, extension.span_names(surface),
                                    parent['surfaces'][surface]['fit_row_count'])
                 for surface in extension.SURFACES}
    before = encoded([parent, additions])
    def no_fit(*args, **kwargs):
        raise AssertionError('metadata extension must not call a fitter')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(norm, 'fit_surface_normalization', no_fit)
        monkeypatch.setattr(norm, 'fit_ctx_cat_contract', no_fit)
        expanded = extension.extend_anchor_span_normalization(parent, additions)
    assert encoded([parent, additions]) == before
    profile = units.anchor_span_fold1_prefix_units_profile()
    fields = compile_full_ta_entry_execution_profile()['source_field_names']
    kwargs = dict(expected_field_names={s: fields['mtf' if s.startswith('mtf_') else s]
                                       for s in norm.EXPECTED_SURFACES},
                  expected_ctx_cat_names=fields['ctx_cat'],
                  expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
                  unobserved_units_profile=profile)
    return parent, additions, expanded, kwargs


def test_full_metadata_wrapper_preserves_every_old_statistic_support_and_alias(metadata_case):
    parent, additions, expanded, kwargs = metadata_case
    assert norm.require_input_normalization_contract(expanded, **kwargs) == expanded
    for surface in norm.EXPECTED_SURFACES:
        old, new = parent['surfaces'][surface], expanded['surfaces'][surface]
        take = [new['field_names'].index(name) for name in old['field_names']]
        for key in extension.STAT_VECTORS:
            assert encoded([new[key][i] for i in take]) == encoded(old[key])
        assert old['fit_row_count'] == new['fit_row_count']
        assert encoded(old['categorical_domains']) == encoded(new['categorical_domains'])
        if 'observed_fit_counts' in old:
            assert [new['observed_fit_counts'][i] for i in take] == old['observed_fit_counts']
            assert new['unfitted_estimates'] == old['unfitted_estimates']
        if surface in extension.SURFACES:
            for j, name in enumerate(extension.span_names(surface)):
                index = new['field_names'].index(name)
                for key in extension.STAT_VECTORS:
                    assert new[key][index] == additions[surface][key][j]
                if surface == 'signal':
                    assert new['observed_fit_counts'][index] == 29782
    for old, new in zip(parent['temporal_aliases'], expanded['temporal_aliases']):
        for surface, key in (('signal', 'signal_index'), ('ctx_cont', 'ctx_cont_index')):
            assert parent['surfaces'][surface]['field_names'][old[key]] == \
                expanded['surfaces'][surface]['field_names'][new[key]]
    assert parent['ctx_cat'] == expanded['ctx_cat'] and parent['lineage'] == expanded['lineage']
    assert expanded['anchor_span_extension']['source_contract'] == parent
    assert expanded['anchor_span_extension']['parent_artifact'] == extension.PARENT_ARTIFACT
    assert expanded['anchor_span_extension']['physical_input_result'] == extension.PHYSICAL_RESULT
    assert expanded['anchor_span_extension']['preservation']['old_statistics_refitted'] is False


def test_existing_ta_projection_routes_new_spans_and_reconstructs_exactly(metadata_case):
    _, _, expanded, kwargs = metadata_case
    before = encoded(expanded)
    projected = norm.project_ta_input_normalization_contract(
        expanded, unobserved_units_profile=kwargs['unobserved_units_profile'])
    profile = compile_full_ta_entry_execution_profile()
    target = profile['physical_field_names']
    expected = {s: target['mtf' if s.startswith('mtf_') else s] for s in norm.EXPECTED_SURFACES}
    assert norm.require_input_normalization_contract(projected, expected_field_names=expected,
        expected_ctx_cat_names=target['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE) == projected
    assert encoded(expanded) == before
    assert projected['surfaces']['signal']['field_count'] == 317
    assert projected['surfaces']['ctx_cont']['field_count'] == 132
    for surface in extension.SURFACES:
        old, new = expanded['surfaces'][surface], projected['surfaces'][surface]
        for name in extension.span_names(surface):
            before_index, after_index = old['field_names'].index(name), new['field_names'].index(name)
            for key in extension.STAT_VECTORS:
                assert old[key][before_index] == new[key][after_index]
    assert projected['ta_projection']['source_contract'] == expanded
    assert projected['ta_projection']['new_fit_performed'] is False


@pytest.mark.parametrize('key', ('center', 'train_transformed_min'))
def test_preservation_detects_signed_zero_changes_at_stored_precision(metadata_case, key):
    parent, _, expanded, _ = metadata_case
    changed = deepcopy(expanded)
    old = parent['surfaces']['signal']
    index = next(i for i, value in enumerate(old[key]) if value == 0.)
    name = old['field_names'][index]
    mapped = changed['surfaces']['signal']['field_names'].index(name)
    changed['surfaces']['signal'][key][mapped] = -float(old[key][index])
    with pytest.raises(RuntimeError, match='OLD_FIELD_STAT:signal:' + key):
        extension.old_field_preservation(parent, changed)


@pytest.mark.parametrize('fault', ('name', 'population', 'missing_surface'))
def test_new_statistics_reject_wrong_names_or_population(metadata_case, fault):
    parent, additions, _, _ = metadata_case
    new = deepcopy(additions)
    if fault == 'missing_surface':
        new.pop('mtf_d1')
    elif fault == 'name':
        new['signal']['field_names'][0] = 'wrong_span'
        extension._restamp_surface(new['signal'])
    else:
        new['signal']['fit_row_count'] += 1
    with pytest.raises(RuntimeError, match='SIX_NEW_STAT_SURFACES|SURFACE_IDENTITY_INVALID|NEW_STAT_SCOPE'):
        extension.extend_anchor_span_normalization(parent, new)


@pytest.mark.parametrize('fault', ('old_stat', 'alias', 'lineage', 'physical_ref', 'new_population'))
def test_resealing_cannot_change_preserved_state_or_provenance(metadata_case, fault):
    _, _, original, kwargs = metadata_case
    changed = deepcopy(original)
    if fault == 'old_stat': changed['surfaces']['signal']['center'][0] += 1.
    elif fault == 'alias': changed['temporal_aliases'][0]['signal_index'] += 1
    elif fault == 'lineage': changed['lineage']['dataset_run_id'] += '_forged'
    elif fault == 'physical_ref': changed['anchor_span_extension']['physical_input_result']['sha256'] = '0' * 64
    else: changed['anchor_span_extension']['new_field_statistics']['signal']['fit_row_count'] += 1
    with pytest.raises(RuntimeError, match='EXACT_EXTENSION|NEW_STAT_SCOPE'):
        norm.require_input_normalization_contract(seal(changed), **kwargs)


def test_resealed_parent_and_wrong_consumer_are_rejected(metadata_case):
    parent, additions, expanded, kwargs = metadata_case
    changed = deepcopy(parent)
    changed['surfaces']['signal']['center'][0] += 1.
    with pytest.raises(RuntimeError, match='EXACT_PARENT'):
        extension.extend_anchor_span_normalization(seal(changed), additions)
    wrong = {**kwargs, 'expected_field_names': deepcopy(kwargs['expected_field_names'])}
    wrong['expected_field_names']['signal'] = list(wrong['expected_field_names']['signal'])[:-1]
    with pytest.raises(RuntimeError, match='CONSUMER_BINDING'):
        norm.require_input_normalization_contract(expanded, **wrong)
    with pytest.raises(RuntimeError, match='PROVENANCE_AND_UNITS_REQUIRED'):
        norm.require_input_normalization_contract(expanded, **{**kwargs, 'unobserved_units_profile': None})
