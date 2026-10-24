"""Synthetic raw domain transitions; no genuine inputs, fitting or native calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.features import gc_native_context_v1 as gc
from gx1.features import htf_features as htf


LEGACY_RETOUCH_COLUMNS = frozenset({
    'ob_bull_first_retouch', 'ob_bear_first_retouch',
    'observed_proxy_ob_bull_first_retouch', 'observed_proxy_ob_bear_first_retouch',
})


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _seal(value):
    payload = {key: item for key, item in value.items() if key != 'contract_sha256'}
    return {**payload, 'contract_sha256': history._sha(payload)}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    return _ref(path)


def _publish(case):
    directory = case['directory']
    case['manifest']['layout'] = _write_json(directory / 'LAYOUT.json', case['legacy'])
    case['manifest']['normalization'] = _write_json(directory / 'NORMALIZATION.json', case['normalization'])
    case['binding'] = _write_json(directory / 'SURFACES.json', case['manifest'])


def _change_layout(case, mutate):
    mutate(case['legacy'])
    case['legacy'] = _seal(case['legacy'])
    case['normalization']['layout_sha256'] = case['legacy']['contract_sha256']
    case['normalization'] = _seal(case['normalization'])
    _publish(case)


def _change_array(case, tf, key, mutate):
    value = case['arrays'][tf][key]
    mutate(value)
    path = case['directory'] / f'GC_{tf}_{key}.npy'
    np.save(path, value, allow_pickle=False)
    case['manifest']['surfaces'][tf][key] = {
        **_ref(path), 'size_bytes': path.stat().st_size,
        'dtype': value.dtype.name, 'shape': list(value.shape),
    }
    _publish(case)


@pytest.fixture
def raw_artifact(tmp_path):
    current = history.gc_mtf_history_layout()
    legacy = deepcopy(current)
    retouches = {row['name'] for row in gc.REGISTRY['fields'] if row['column'] in LEGACY_RETOUCH_COLUMNS}
    assert len(retouches) == 14 and len(gc.SOURCE_BINARY_VALUE_FIELDS) == 76
    for tf in history.TIMEFRAMES:
        legacy['binary_fields'][tf] = [name for name in current['binary_fields'][tf]
            if name not in gc.SOURCE_BINARY_VALUE_FIELDS or name in retouches]
    legacy = _seal(legacy)
    # Historical statistics are intentionally opaque: the bridge may verify
    # their seal/link, but must never treat this as fitted current statistics.
    normalization = _seal({'schema_version': history.NORMALIZATION_SCHEMA_VERSION,
        'layout_sha256': legacy['contract_sha256'], 'surfaces': {}, 'provenance': {}})
    manifest = {'schema_version': 'gx1_gc_mtf_history_physical_surfaces_v1',
        'CONTROL_fit_rows': 0, 'TEST_fit_rows': 0, 'predictive_value_measured': False,
        'surfaces': {}}
    arrays = {}
    for tf in history.TIMEFRAMES:
        names = current['field_names'][tf]
        values = np.zeros((3, len(names)), dtype=np.float32)
        for index, name in enumerate(names):
            if name in gc.OBSERVED_FIELDS or name in gc.STRUCTURAL_VALUE_FIELDS:
                values[:, index] = 1.0
            if name in gc.SOURCE_BINARY_VALUE_FIELDS:
                values[:, index] = (0.0, 1.0, 0.0)
        arrays[tf] = {'native_row_indices': np.arange(3, dtype=np.int64),
            'available_ns': (np.arange(3, dtype=np.int64) + 1) * history.MINUTE_NS,
            'features': values}
        manifest['surfaces'][tf] = {}
        for key, value in arrays[tf].items():
            path = tmp_path / f'GC_{tf}_{key}.npy'
            np.save(path, value, allow_pickle=False)
            manifest['surfaces'][tf][key] = {**_ref(path), 'size_bytes': path.stat().st_size,
                'dtype': value.dtype.name, 'shape': list(value.shape)}
    case = {'directory': tmp_path, 'current': current, 'legacy': legacy,
        'normalization': normalization, 'manifest': manifest, 'arrays': arrays}
    _publish(case)
    return case


def test_raw_metadata_binds_three_json_seals_without_loading_arrays_or_statistics(raw_artifact, monkeypatch):
    opened = []
    read = history._read_gc_mtf_history_artifact_json

    def recorded(ref):
        opened.append(ref['path'])
        return read(ref)

    def forbidden(*_args, **_kwargs):
        pytest.fail('metadata reader reached array or fitted-normalization admission')

    monkeypatch.setattr(history, '_read_gc_mtf_history_artifact_json', recorded)
    monkeypatch.setattr(htf, '_load_verified_cache_npy', forbidden)
    monkeypatch.setattr(history, 'require_gc_mtf_history_normalization', forbidden)
    result = history.read_gc_mtf_history_raw_artifact_metadata(raw_artifact['binding'])
    assert set(opened) == {raw_artifact['binding']['path'],
        raw_artifact['manifest']['layout']['path'], raw_artifact['manifest']['normalization']['path']}
    assert len(opened) == 3 and set(result) == {'manifest', 'layout', 'transition'}
    transition = result['transition']
    assert result['layout'] == raw_artifact['current']
    assert transition['legacy_layout_sha256'] == raw_artifact['legacy']['contract_sha256']
    assert transition['current_layout_sha256'] == raw_artifact['current']['contract_sha256']
    assert sum(map(len, transition['binary_field_additions'].values())) == 62
    assert transition['array_descriptors'] == raw_artifact['manifest']['surfaces']
    assert not transition['normalization_applied'] and not transition['legacy_statistics_admitted_as_current']
    assert transition['raw_value_validation'] == {'performed': False}
    assert transition == _seal(transition)


def test_raw_loader_reuses_fifteen_canonical_readonly_mappings_and_validates_domains(raw_artifact, monkeypatch):
    verified = {}
    load = htf._load_verified_cache_npy

    def recorded(directory_fd, name, **kwargs):
        result = load(directory_fd, name, **kwargs)
        assert name not in verified
        verified[name] = result
        return result

    monkeypatch.setattr(htf, '_load_verified_cache_npy', recorded)
    result = history.load_gc_mtf_history_raw_artifact(raw_artifact['binding'])
    assert len(verified) == 15 and set(result) == {'manifest', 'layout', 'transition', 'surfaces'}
    for tf in history.TIMEFRAMES:
        assert result['surfaces'][tf]['field_names'] == raw_artifact['current']['field_names'][tf]
        for key, expected in raw_artifact['arrays'][tf].items():
            actual = result['surfaces'][tf][key]
            assert not actual.flags.writeable and np.array_equal(actual, expected)
            assert np.shares_memory(actual, verified[f'GC_{tf}_{key}.npy'])
    proof = result['transition']['raw_value_validation']
    assert proof['performed'] and proof['finite'] and proof['masked_storage']
    assert proof['rows_by_tf'] == {tf: 3 for tf in history.TIMEFRAMES}
    assert proof['binary_fields'] == raw_artifact['current']['binary_fields']
    assert result['transition'] == _seal(result['transition'])


@pytest.mark.parametrize('reader', [history.read_gc_mtf_history_artifact_metadata, history.load_gc_mtf_history_artifact])
def test_legacy_strict_wrapper_still_rejects_old_domain_layout(raw_artifact, reader):
    with pytest.raises(RuntimeError, match='LAYOUT_CONTRACT'):
        reader(raw_artifact['binding'])


@pytest.mark.parametrize('fault', ['field_order', 'width', 'mask_pair', 'category', 'registry',
    'missing_observation_mask', 'undeclared_binary', 'binary_order'])
def test_raw_transition_rejects_changes_outside_proven_binary_additions(raw_artifact, fault):
    def mutate(layout):
        tf = 'M5'
        if fault == 'field_order':
            layout['field_names'][tf][0:2] = layout['field_names'][tf][0:2][::-1]
        elif fault == 'width':
            layout['widths'][tf] += 1
        elif fault == 'mask_pair':
            layout['value_mask_pairs'][tf][0][1] += 1
        elif fault == 'category':
            layout['categorical_domains'][tf] = {'invented': [0, 1]}
        elif fault == 'registry':
            layout['source_registry']['sha256'] = '0' * 64
        elif fault == 'missing_observation_mask':
            observed = next(name for name in layout['binary_fields'][tf] if name in gc.OBSERVED_FIELDS)
            layout['binary_fields'][tf].remove(observed)
        elif fault == 'undeclared_binary':
            continuous = next(name for name in layout['field_names'][tf]
                if name not in raw_artifact['current']['binary_fields'][tf])
            layout['binary_fields'][tf].append(continuous)
            layout['binary_fields'][tf].sort(key=layout['field_names'][tf].index)
        else:
            layout['binary_fields'][tf].reverse()
    _change_layout(raw_artifact, mutate)
    with pytest.raises(RuntimeError, match='RAW_'):
        history.read_gc_mtf_history_raw_artifact_metadata(raw_artifact['binding'])


@pytest.mark.parametrize('fault', ['layout_seal', 'normalization_seal', 'normalization_layout_join'])
def test_raw_metadata_rejects_legacy_seal_or_layout_join_mismatch(raw_artifact, fault):
    if fault == 'layout_seal':
        raw_artifact['legacy']['contract_sha256'] = '0' * 64
    elif fault == 'normalization_seal':
        raw_artifact['normalization']['contract_sha256'] = '0' * 64
    else:
        raw_artifact['normalization']['layout_sha256'] = '0' * 64
        raw_artifact['normalization'] = _seal(raw_artifact['normalization'])
    _publish(raw_artifact)
    with pytest.raises(RuntimeError, match='RAW_LEGACY_'):
        history.read_gc_mtf_history_raw_artifact_metadata(raw_artifact['binding'])


@pytest.mark.parametrize('fault', ['path', 'shape', 'dtype'])
def test_raw_metadata_preserves_exact_canonical_array_descriptor_guards(raw_artifact, fault):
    descriptor = raw_artifact['manifest']['surfaces']['M5']['features']
    if fault == 'path':
        descriptor['path'] = str(raw_artifact['directory'] / 'OTHER.npy')
    elif fault == 'shape':
        descriptor['shape'][1] += 1
    else:
        descriptor['dtype'] = 'float64'
    _publish(raw_artifact)
    with pytest.raises(RuntimeError, match='ARTIFACT_ARRAY_'):
        history.read_gc_mtf_history_raw_artifact_metadata(raw_artifact['binding'])


def test_raw_loader_preserves_canonical_payload_sha_verification(raw_artifact):
    path = Path(raw_artifact['manifest']['surfaces']['M5']['features']['path'])
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(RuntimeError, match='HTF_V4_CACHE_SHA256_MISMATCH'):
        history.load_gc_mtf_history_raw_artifact(raw_artifact['binding'])


@pytest.mark.parametrize('fault', ['binary_two', 'masked_nonzero', 'nonfinite', 'repeated_id', 'clock_order'])
def test_raw_loader_validates_current_domains_masks_and_causal_axis_headers(raw_artifact, fault):
    tf = 'D1'
    names = raw_artifact['current']['field_names'][tf]
    value_index = next(i for i, name in enumerate(names)
        if name in gc.SOURCE_BINARY_VALUE_FIELDS and name not in gc.STRUCTURAL_VALUE_FIELDS)
    if fault == 'repeated_id':
        _change_array(raw_artifact, tf, 'native_row_indices', lambda x: x.__setitem__(1, x[0]))
        reason = 'SURFACE_NATIVE_ROWS'
    elif fault == 'clock_order':
        _change_array(raw_artifact, tf, 'available_ns', lambda x: x.__setitem__(1, x[0]))
        reason = 'SURFACE_CLOSED_CLOCK'
    else:
        def mutate(values):
            if fault == 'binary_two':
                values[0, value_index] = 2.0
            elif fault == 'masked_nonzero':
                mask_index = next(mask for value, mask in raw_artifact['current']['value_mask_pairs'][tf]
                    if value == value_index)
                values[0, value_index] = 1.0
                values[0, mask_index] = 0.0
            else:
                values[0, value_index] = np.nan
        _change_array(raw_artifact, tf, 'features', mutate)
        reason = {'binary_two': 'RAW_BINARY_DOMAIN', 'masked_nonzero': 'MASKED_STORAGE_INVALID',
            'nonfinite': 'RAW_NONFINITE'}[fault]
    with pytest.raises(RuntimeError, match=reason):
        history.load_gc_mtf_history_raw_artifact(raw_artifact['binding'])
