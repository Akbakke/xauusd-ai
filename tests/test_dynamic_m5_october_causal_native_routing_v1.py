"""Source-only October batch boundary: no native model or genuine data reads."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_dynamic_m5_october_causal_inputs_v1 as causal
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets_owner
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ref(path, value):
    path.write_text(json.dumps(value) + '\n')
    return {'path': str(path), 'sha256': _sha(path)}


@pytest.fixture
def tape(tmp_path):
    n, fit = 36172, 29849
    rows = np.arange(n, dtype=np.int64)
    issue = 1_700_000_000_000_000_000 + rows * 300_000_000_000
    state = np.full((n, 5), targets_owner.FLAT, np.int8)
    observed = np.ones((n, 5), bool)
    path = np.ones((n, 5), bool)
    age = np.ones((n, 5, 2), np.float32)
    arrays = {'parent_rows': rows, 'issue_ns': issue, 'current_state': state,
              'current_observed': observed, 'current_path_observed': path,
              'current_confirmed_run_age_features': age}
    digest = lambda array: hashlib.sha256(array.astype('<i8', copy=False).tobytes()).hexdigest()
    scope = {'rows': n,
             'fit': {'rows': fit, 'cutoff_ns': int(issue[fit]),
                     'parent_rows_sha256': digest(rows[:fit]),
                     'decision_ns_sha256': digest(issue[:fit])},
             'held_out': {'rows': n - fit, 'cutoff_ns': int(issue[-1] + 300_000_000_000),
                          'parent_rows_sha256': digest(rows[fit:]),
                          'decision_ns_sha256': digest(issue[fit:])},
             'parent_rows_sha256': digest(rows), 'decision_ns_sha256': digest(issue)}
    observer = _json_ref(tmp_path / 'observer.json', scope)
    design = _json_ref(tmp_path / 'design.json', {'role': 'candidate'})
    series = _json_ref(tmp_path / 'series.json', {'role': 'historical_pre_november'})
    descriptors = {}
    for name, array in arrays.items():
        path_name = tmp_path / (name + '.npy')
        np.save(path_name, array, allow_pickle=False)
        descriptors[name] = {'path': str(path_name), 'sha256': _sha(path_name),
                             'size_bytes': path_name.stat().st_size,
                             'dtype': array.dtype.name, 'shape': list(array.shape)}
    manifest = {'schema_version': 'gx1_dynamic_m5_october_causal_input_artifact_v1',
                'candidate_design': design, 'target_source_series': series,
                'observer_axes': observer, 'future_labels_included': False,
                'clock_basis': 'historical_event_time_only_online_delivery_not_qualified',
                'fit_parent_rows': fit, 'development_parent_rows': n-fit,
                'cutoff_ns_exclusive': scope['held_out']['cutoff_ns'],
                'parent_rows_sha256': scope['parent_rows_sha256'],
                'issue_ns_sha256': scope['decision_ns_sha256'], 'arrays': descriptors}
    binding = _json_ref(tmp_path / 'CAUSAL_INPUTS.json', manifest)
    kwargs = {'observer_axes_ref': observer, 'candidate_design_ref': design,
              'target_series_ref': series}
    return causal.load_october_causal_inputs(binding, **kwargs), arrays, scope, binding, kwargs


def test_verified_tape_accepts_shuffled_fit_and_separates_candidate_age(tape):
    source, arrays, scope, _, _ = tape
    selected = np.array([29848, 1], np.int64)
    issue = arrays['issue_ns'][selected]
    candidate = source.gather(parent_rows=selected, issue_ns=issue, role='fit', include_age=True)
    control = source.gather(parent_rows=selected, issue_ns=issue, role='fit', include_age=False)
    assert set(candidate) == set(control) | {'current_confirmed_run_age_features'}
    assert set(candidate).isdisjoint({'state', 'observed', 'maturity_ns', 'reason'})
    assert np.array_equal(candidate['current_state'], control['current_state'])
    assert np.array_equal(candidate['current_confirmed_run_age_features'], arrays['current_confirmed_run_age_features'][selected])
    with pytest.raises(RuntimeError, match='EXACT_PARENT_ISSUE_ROLE'):
        source.gather(parent_rows=selected, issue_ns=issue, role='score', include_age=True)
    with pytest.raises(RuntimeError, match='EXACT_PARENT_ISSUE_ROLE'):
        source.gather(parent_rows=selected, issue_ns=issue + 1, role='fit', include_age=True)
    assert source.gather(parent_rows=np.array([29849], np.int64),
                         issue_ns=arrays['issue_ns'][[29849]], role='score', include_age=True)['current_state'].shape == (1, 5)
    assert scope['fit']['cutoff_ns'] == int(arrays['issue_ns'][29849])


def test_native_batch_uses_causal_tape_only_for_october_and_matches_target(monkeypatch, tape):
    source, arrays, scope, _, _ = tape
    selected = np.array([29848, 1], np.int64)
    parent = arrays['parent_rows'][:29849]
    decision = arrays['issue_ns'][:29849]
    prepared = {'arrays': {'parent_rows': parent, 'decision_ns': decision,
                            'bar_start_ns': decision - 300_000_000_000},
                'october_role': 'fit', 'observer_axes': object(), 'october_units_profile': {'profile': 'bound'},
                'october_causal_inputs': source,
                'entry_sequence_source': None, 'current': {'snap': None, 'ctx_cont': None, 'ctx_cat': None},
                'cache': None, 'gc': {'surfaces': None, 'layout': None}}
    targets = {'dynamic_state_series': object(), 'train_cutoff_ns': scope['fit']['cutoff_ns'],
               'lead_minutes': np.array([5], np.int64)}
    def gather(*args, **kwargs):
        assert kwargs['unobserved_units_profile'] is prepared['october_units_profile']
        assert kwargs['observer_axes'] is prepared['observer_axes']
        return {'features': {'seq_x': np.zeros((2, 1, 1), np.float32)}, 'witness': {}}
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch', gather)
    def historical_query(*args, **kwargs):
        take = np.searchsorted(arrays['parent_rows'], kwargs['parent_rows'])
        n = len(take)
        return {'state': np.zeros((n, 5, 1), np.int8), 'observed': np.ones((n, 5, 1), bool),
                'issue_ns': kwargs['issue_ns'], 'validity_ns': np.zeros((n, 5, 1), np.int64),
                'maturity_ns': np.zeros((n, 5, 1), np.int64),
                'current_state': arrays['current_state'][take].copy(),
                'current_observed': arrays['current_observed'][take].copy(),
                'current_path_observed': arrays['current_path_observed'][take].copy(),
                'path_observed': np.ones((n, 5, 1), bool), 'reason': np.zeros((n, 5, 1), np.uint8)}
    monkeypatch.setattr(targets_owner, 'query_dynamic_mtf_states', historical_query)
    view = {'october_observer_axes': scope, 'current_state_first_loss_weight': 0.03,
            'confirmed_run_age_input': True}
    raw, target = worker._batch(prepared, targets, view, selected, np)
    assert 'current_confirmed_run_age_features' in raw['features']
    assert 'state' not in raw['features'] and 'observed' not in raw['features']
    assert np.array_equal(raw['features']['current_state'], target['current_state'])
    control = {k: v for k, v in view.items() if k != 'confirmed_run_age_input'}
    raw_control, _ = worker._batch(prepared, targets, control, selected, np)
    assert 'current_confirmed_run_age_features' not in raw_control['features']
    original = targets_owner.query_dynamic_mtf_states
    def mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        result['current_state'][0, 0] = targets_owner.DOWN
        return result
    monkeypatch.setattr(targets_owner, 'query_dynamic_mtf_states', mismatch)
    with pytest.raises(RuntimeError, match='CURRENT_INPUT_TARGET_ALIGNMENT'):
        worker._batch(prepared, targets, view, selected, np)


def test_artifact_bytes_and_masks_are_fail_closed(tape):
    _, _, _, binding, kwargs = tape
    path = kwargs['observer_axes_ref']['path']
    with open(path, 'ab') as stream:
        stream.write(b'x')
    with pytest.raises(RuntimeError, match='REF_IDENTITY'):
        causal.load_october_causal_inputs(binding, **kwargs)


def test_fit_role_does_not_decode_later_feature_values_or_allow_score_gather(tape):
    _, arrays, _, binding, kwargs = tape
    later = arrays['current_confirmed_run_age_features'].copy()
    later[29849, 0, 0] = np.nan
    path = Path(binding['path'])
    manifest = json.loads(path.read_text())
    age_path = manifest['arrays']['current_confirmed_run_age_features']['path']
    np.save(age_path, later, allow_pickle=False)
    manifest['arrays']['current_confirmed_run_age_features']['sha256'] = _sha(Path(age_path))
    binding = _json_ref(path, manifest)
    fit = causal.load_october_causal_inputs(binding, **kwargs, validation_role='fit')
    selected = np.array([29848], np.int64)
    assert fit.gather(parent_rows=selected, issue_ns=arrays['issue_ns'][selected],
                      role='fit', include_age=True)['current_confirmed_run_age_features'].shape == (1, 5, 2)
    with pytest.raises(RuntimeError, match='ROLE'):
        fit.gather(parent_rows=np.array([29849], np.int64), issue_ns=arrays['issue_ns'][[29849]],
                   role='score', include_age=True)
    with pytest.raises(RuntimeError, match='CURRENT_VALUE_MASK'):
        causal.load_october_causal_inputs(binding, **kwargs)
