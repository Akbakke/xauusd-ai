"""Small sealed-profile and synthetic unit-routing checks; no physical fit or model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_model_native_input_normalization_v1 as normalization
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile


def _seal(value):
    value['contract_sha256'] = units._sha({k: v for k, v in value.items() if k != 'contract_sha256'})
    return value


def test_mature_fold1_plan_binds_new_prefix_and_only_old_declared_units():
    old = units.october_prefix_units_profile()
    new = units.mature_fold1_prefix_units_profile()
    path = Path(__file__).resolve().parents[1] / units.MATURE_FOLD1_PLAN_PATH
    raw = path.read_bytes()
    plan = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == units.MATURE_FOLD1_PLAN_SHA256
    assert plan['unit_source'] == {'path': units.PLAN_PATH, 'sha256': units.PLAN_SHA256,
                                   'reuse_scope': 'declared_source_units_and_physical_column_maps_only'}
    assert plan['support_evidence']['sha256'] == '6df90574ef2e414a0b544f69c25de35b488147d91ad413916691b973b525ff3f'
    assert new['plan'] == {'path': units.MATURE_FOLD1_PLAN_PATH, 'sha256': units.MATURE_FOLD1_PLAN_SHA256}
    assert new['fit_prefix'] == plan['fit_prefix']
    assert new['fit_prefix']['rows'] == 29574 and old['fit_prefix']['rows'] == 29849
    assert new['fit_prefix'] != old['fit_prefix']
    assert new['fields'] == old['fields'] and new['surfaces'] == old['surfaces']
    assert new['support_evidence'] == plan['support_evidence'] != old['support_evidence']
    assert all(value is False for value in new['authority'].values())
    assert units.require_prefix_units_profile(new) == new
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        units.require_october_prefix_units_profile(new)
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        units.require_mature_fold1_prefix_units_profile(old)


@pytest.mark.parametrize('fault', ['fit_prefix', 'source_unit', 'column', 'support_evidence', 'plan_sha', 'authority'])
def test_mature_fold1_resealed_profile_mutation_fails_closed(fault):
    profile = deepcopy(units.mature_fold1_prefix_units_profile())
    if fault == 'fit_prefix':
        profile['fit_prefix']['rows'] += 1
    elif fault == 'source_unit':
        profile['fields'][0]['source_unit'] = 'one_contract'
    elif fault == 'column':
        profile['surfaces']['signal']['columns'][0]['value_index'] += 1
    elif fault == 'support_evidence':
        profile['support_evidence']['sha256'] = '0' * 64
    elif fault == 'plan_sha':
        profile['plan']['sha256'] = '0' * 64
    else:
        profile['authority']['training_enabled'] = True
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        units.require_prefix_units_profile(_seal(profile))


def test_mature_fold1_changed_plan_bytes_fail_before_profile_construction(tmp_path, monkeypatch):
    changed = tmp_path / 'changed.json'
    changed.write_bytes(units._MATURE_FOLD1_PLAN_FILE.read_bytes() + b'\n')
    monkeypatch.setattr(units, '_MATURE_FOLD1_PLAN_FILE', changed)
    with pytest.raises(RuntimeError, match='MATURE_PLAN_SHA'):
        units.mature_fold1_prefix_units_profile()


@pytest.mark.parametrize('key', ['rows', 'cutoff_ns', 'parent_rows_sha256',
                                 'bar_start_ns_sha256', 'decision_ns_sha256'])
def test_mature_fold1_rejects_old_or_changed_fit_axis(key):
    profile = units.mature_fold1_prefix_units_profile()
    old = units.october_prefix_units_profile()
    assert units.require_prefix_units_fit_identity(profile['fit_prefix'], profile) == profile['fit_prefix']
    with pytest.raises(RuntimeError, match='FIT_IDENTITY'):
        units.require_prefix_units_fit_identity(old['fit_prefix'], profile)
    changed = dict(profile['fit_prefix'])
    changed[key] = changed[key] + 1 if type(changed[key]) is int else '0' * 64
    with pytest.raises(RuntimeError, match='FIT_IDENTITY'):
        units.require_prefix_units_fit_identity(changed, profile)


def test_synthetic_unobserved_surface_has_new_identity_and_preserves_mask():
    new = units.mature_fold1_prefix_units_profile()
    old = units.october_prefix_units_profile()
    names = compile_compact_entry_execution_profile()['physical_field_names']['signal']
    columns = units.prefix_units_surface_columns(new, surface='signal', field_names=names)
    assert len(columns) == 51
    raw = np.ones((8, len(names)), dtype=np.float32)
    for index, entry in columns.items():
        raw[:, [index, entry['mask_index']]] = 0.
    fitted = normalization.fit_surface_normalization(
        raw, surface='signal', field_names=names,
        semantic_categorical_domains=normalization.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
        allow_constant_train_fields=True, unobserved_units_profile=new)
    assert fitted['unobserved_units_profile_sha256'] == new['contract_sha256']
    assert normalization.require_surface_normalization(
        fitted, surface='signal', field_names=names, unobserved_units_profile=new) == fitted
    with pytest.raises(RuntimeError, match='UNITS_BINDING'):
        normalization.require_surface_normalization(
            fitted, surface='signal', field_names=names, unobserved_units_profile=old)
    sample = raw[:1].copy()
    index, entry = next((i, e) for i, e in columns.items() if not e['binary'])
    sample[0, index] = 1.
    sample[0, entry['mask_index']] = 1.
    transformed = normalization.apply_surface_normalization(sample, fitted)
    assert transformed[0, index] != 0. and transformed[0, entry['mask_index']] == 1.
    sample[0, index] = 0.
    sample[0, entry['mask_index']] = 0.
    transformed_unknown = normalization.apply_surface_normalization(sample, fitted)
    assert transformed_unknown[0, index] == transformed_unknown[0, entry['mask_index']] == 0.
