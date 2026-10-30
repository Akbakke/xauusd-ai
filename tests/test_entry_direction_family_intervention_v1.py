"""Synthetic canonical-owner ablation checks; no model or actual market data."""
from copy import deepcopy
import numpy as np
import pytest

from gx1.contracts import entry_direction_family_intervention_v1 as intervention
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.features import entry_direction_momentum_profile_v1 as momentum
from gx1.features import entry_specialist_feature_groups_v1 as owners
from tests.model_native_input_normalization_support import input_normalization_fixture
from tests.test_entry_gc_mtf_history_v1 import gc_history_fixture


@pytest.fixture(scope='module')
def case():
    bound = gc_history_fixture()[0]
    profile = momentum.compile_price_momentum_profile(bound)
    physical = profile['physical_field_names']
    base = input_normalization_fixture(signal_names=list(physical['signal']), mtf_names=list(physical['mtf']),
        per_tf_seq_lens=bound['layout']['window_rows'], entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    args = dict(profile_reference={k: profile[k] for k in ('name', 'contract_sha256')},
        gc_mtf_history=bound, base_normalization=base, gc_normalization=bound['normalization'])
    contract = intervention.compile_family_intervention(**args)
    features = {}
    for key, item in contract['surfaces'].items():
        shape = (2, len(item['field_names'])) if item['rows'] is None else (2, item['rows'], len(item['field_names']))
        raw = np.full(shape, 0.25, dtype=item['dtype'])
        for i in item['binary_indices']:
            raw[..., i] = 1
        for name, domain in item['categorical_domains'].items():
            raw[..., item['field_names'].index(name)] = domain[-1]
        for i in item['preserved_mask_indices']:
            raw[..., i] = 1
            raw[0, ..., i] = 0
        for v, m in item['mask_pairs']:
            raw[..., v] = np.where(raw[..., m] == 1, raw[..., v], 0)
        if 'ema_stack_aligned_v2' in item['field_names']:
            raw[..., item['field_names'].index('ema_stack_aligned_v2')] = 1
        for i, value in item['source_constants'].items():
            raw[..., int(i)] = value
        features[key] = raw
    for a in contract['aliases']:
        features['seq_x'][..., int(a['signal_index'])] = features['ctx_cont'][:, int(a['ctx_cont_index'])][:, None]
    features['snap_x'] = features['seq_x'][:, -1].copy()
    return args, contract, features


def apply(case, arm, features=None):
    _args, contract, original = case
    return intervention.apply_family_intervention_numpy(original if features is None else features, contract,
        arm=arm, expected_sha256=contract['contract_sha256'])


def test_all_canonical_routes_masks_macro_and_shared_dependencies(case):
    args, c, features = case
    assert set(c['families']) == set(owners.MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert set(c['arms']) == set(c['families']) | {'baseline', 'macro_context'}
    assert set(c['surfaces']) == set(features)
    assert len(c['surfaces']) == 14
    assert len(c['macro_context_fields']) == 61
    assert len(c['arms']['macro_context']['ctx_cont']) == 61
    for tf in args['gc_mtf_history']['layout']['timeframes']:
        s = c['surfaces']['gc_seq_' + tf.lower()]
        assert len(s['shared_mask_dependencies']) == 1
        d = s['shared_mask_dependencies'][0]
        assert d['value_family'] == 'momentum_flow_encoder'
        assert d['mask_family'] == 'session_regime_encoder'
        assert d['mask_index'] in s['preserved_mask_indices']
        assert s['normalization_sha256'] == args['gc_normalization']['surfaces']['gc_mtf_' + tf.lower()]['stats_sha256']
    for arm, routes in c['arms'].items():
        for key, indices in routes.items():
            assert not set(indices) & set(c['surfaces'][key]['preserved_mask_indices'])


def test_baseline_is_byte_identical_and_independent(case):
    _args, _c, original = case
    result = apply(case, 'baseline')
    for key in original:
        assert result[key].tobytes() == original[key].tobytes()
        assert not np.shares_memory(result[key], original[key])


@pytest.mark.parametrize('arm', [*owners.MODEL_NATIVE_TRAINING_SPECIALISTS, 'macro_context'])
def test_each_arm_only_changes_owned_values_and_keeps_all_aliases(case, arm):
    _args, c, original = case
    before = {k: x.tobytes() for k, x in original.items()}
    result = apply(case, arm)
    for key, item in c['surfaces'].items():
        assert original[key].tobytes() == before[key]
        indices = c['arms'][arm][key]
        other = sorted(set(range(original[key].shape[-1])) - set(indices))
        assert result[key][..., other].tobytes() == original[key][..., other].tobytes()
        pairs = dict(item['mask_pairs'])
        for i in indices:
            expected = item['replacement_raw'][i]
            if i in pairs:
                expected = np.where(original[key][..., pairs[i]] == 1, expected, 0)
            assert np.all(result[key][..., i] == expected)
    assert np.array_equal(result['seq_x'][:, -1], result['snap_x'])
    for a in c['aliases']:
        assert np.array_equal(result['snap_x'][:, a['signal_index']], result['ctx_cont'][:, a['ctx_cont_index']])


def test_continuous_values_normalize_to_zero_and_domains_are_explicit(case):
    args, c, _features = case
    for key, item in c['surfaces'].items():
        if key == 'ctx_cat':
            assert all('not_natural_neutral' in x for x in item['replacement_convention'])
            continue
        surface = ('signal' if key in ('seq_x', 'snap_x') else 'ctx_cont' if key == 'ctx_cont'
                   else 'gc_mtf_' + key.removeprefix('gc_seq_') if key.startswith('gc_seq_') else 'mtf_' + key.removeprefix('seq_'))
        stats = (args['gc_normalization'] if key.startswith('gc_seq_') else args['base_normalization'])['surfaces'][surface]
        for i, convention in enumerate(item['replacement_convention']):
            if convention == intervention.CONVENTION['continuous']:
                center = np.float32(stats['center'][i])
                assert np.float32(item['replacement_raw'][i]) == center
                assert np.arcsinh((np.float64(item['replacement_raw'][i]) - np.float64(center)) / stats['scale'][i]) == 0
        if 'ema_stack_aligned_v2' in item['field_names']:
            i = item['field_names'].index('ema_stack_aligned_v2')
            assert item['replacement_raw'][i] == 0
            assert item['replacement_convention'][i] == intervention.CONVENTION['ema_stack_aligned_v2']


@pytest.mark.parametrize('failure', ['extra_target', 'missing_surface', 'alias', 'dtype', 'nan', 'mask', 'unknown_arm'])
def test_invalid_input_fails_closed(case, failure):
    _args, c, original = case
    features = {k: x.copy() for k, x in original.items()}
    arm = 'baseline'
    if failure == 'extra_target': features['future_target'] = np.zeros(2)
    elif failure == 'missing_surface': del features['seq_d1']
    elif failure == 'alias': features['snap_x'][0, 0] += 1
    elif failure == 'dtype': features['seq_m5'] = features['seq_m5'].astype(np.float64)
    elif failure == 'nan': features['seq_m5'][0, 0, 0] = np.nan
    elif failure == 'mask':
        v, m = c['surfaces']['gc_seq_m5']['mask_pairs'][0]
        features['gc_seq_m5'][0, 0, m] = 0
        features['gc_seq_m5'][0, 0, v] = 1
    elif failure == 'unknown_arm': arm = 'invented'
    with pytest.raises(RuntimeError, match='FAMILY_INTERVENTION_'):
        apply(case, arm, features)


def test_profile_gc_and_contract_identity_are_checked(case):
    args, c, features = case
    bad = dict(args); bad['profile_reference'] = {**args['profile_reference'], 'contract_sha256': '0' * 64}
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        intervention.compile_family_intervention(**bad)
    bad = deepcopy(c); bad['surfaces']['seq_m5']['replacement_raw'][0] = 999
    with pytest.raises(RuntimeError, match='CONTRACT_HASH'):
        intervention.apply_family_intervention_numpy(features, bad, arm='baseline', expected_sha256=c['contract_sha256'])
    with pytest.raises(RuntimeError, match='CONTRACT_HASH'):
        intervention.apply_family_intervention_numpy(features, c, arm='baseline', expected_sha256='0' * 64)
    bad_args = deepcopy(args); bad_args['gc_normalization']['contract_sha256'] = '0' * 64
    with pytest.raises(RuntimeError):
        intervention.compile_family_intervention(**bad_args)


def test_source_constants_preserved_including_nonzero_declared_constant(case, monkeypatch):
    args, _c, original = case
    declarations = deepcopy(intervention.htf.MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4)
    # Exercise the generic owner contract with a nonzero synthetic declaration.
    declarations['D1']['session_change_flag'] = 1.0
    monkeypatch.setattr(intervention.htf, 'MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4', declarations)
    c = intervention.compile_family_intervention(**args)
    inputs = {k: x.copy() for k, x in original.items()}
    item = c['surfaces']['seq_d1']; i = item['field_names'].index('session_change_flag')
    inputs['seq_d1'][..., i] = 1.0
    assert item['source_constants'][str(i)] == 1.0
    for arm in c['arms']:
        assert i not in c['arms'][arm]['seq_d1']
        out = intervention.apply_family_intervention_numpy(inputs, c, arm=arm, expected_sha256=c['contract_sha256'])
        assert np.all(out['seq_d1'][..., i] == 1.0)
    inputs['seq_d1'][..., i] = 0.0
    with pytest.raises(RuntimeError, match='SOURCE_CONSTANT_DOMAIN'):
        intervention.apply_family_intervention_numpy(inputs, c, arm='baseline', expected_sha256=c['contract_sha256'])


def test_macro_inventory_is_canonical_source_bound(case):
    _args, c, _inputs = case
    owner = intervention.d1_macro_context_contract()
    assert c['macro_source_contract'] == owner
    assert set(c['macro_context_fields']) == set(owner['fields'])
    assert owner['specialist_owner'] == 'session_regime_encoder'
    assert len(owner['fields_by_group']['positioning_cot']) == 4
    assert len(owner['fields_by_group']['panel_oanda']) == 44
    assert len(owner['fields_by_group']['drivers']) == 13
