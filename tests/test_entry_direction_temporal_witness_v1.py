"""239 synthetic source qualification; no genuine data or model construction."""
import copy

import numpy as np
import pytest

from gx1.contracts.entry_direction_temporal_witness_v1 import (
    BOOK_SOURCE, TIMEFRAMES, apply_direction_temporal_book_prefix,
    fit_direction_temporal_book_prefix, project_direction_temporal_book_witness,
)

BASE = 1_704_492_000_000_000_000  # 2024-01-05 22:00UTC, genuine H4/D1 phase
MINUTE = 60_000_000_000
DURATIONS = {'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}


def _freeze(value):
    value.setflags(write=False)
    return value


def _fixture():
    decisions = np.asarray(BASE + np.arange(4) * 5 * MINUTE, np.int64)
    surfaces = {}
    for tf in TIMEFRAMES:
        width, vi, mi, value_name, mask_name = BOOK_SOURCE[tf]
        names = ['unused_' + str(i) for i in range(width)]
        names[vi], names[mi] = value_name, mask_name
        features = np.zeros((7, width), np.float32)
        features[:, vi] = [0.1, 0.0, 0.0, 0.75, -0.5, 0.25, -0.25]
        features[:, mi] = [1, 0, 1, 1, 1, 1, 1]
        surfaces[tf] = {'native_row_indices': _freeze(np.arange(10, 17, dtype=np.int64)),
                        'available_ns': _freeze(np.asarray(BASE + np.arange(-2, 5) * DURATIONS[tf] * MINUTE, np.int64)),
                        'features': _freeze(features), 'field_names': names}
    return surfaces, decisions


def _inputs(surfaces, decisions):
    ids, pairs = {}, {}
    for tf in TIMEFRAMES:
        surface = surfaces[tf]
        selected = np.searchsorted(surface['available_ns'], decisions, side='right') - 1
        ids[tf] = np.array(surface['native_row_indices'][selected], copy=True)
        _width, vi, mi, _v, _m = BOOK_SOURCE[tf]
        pairs[tf] = np.ascontiguousarray(surface['features'][np.ix_(selected, [vi, mi])])
    return ids, pairs


def _project(surfaces, decisions):
    ids, pairs = _inputs(surfaces, decisions)
    return project_direction_temporal_book_witness(surfaces, decisions,
        current_native_row_indices=ids, current_206_pairs=pairs)


def test_causal_current_previous_keep_parents_and_h4_d1_phases():
    surfaces, decisions = _fixture()
    witness = _project(surfaces, decisions)
    np.testing.assert_array_equal(witness['parent_decision_ns'], decisions)
    assert len(witness['current']) == len(decisions)
    for tf in TIMEFRAMES:
        np.testing.assert_array_equal(witness['previous_native_row_indices'][tf],
                                      witness['current_native_row_indices'][tf] - 1)
        assert np.all(witness['previous_available_ns'][tf] < witness['current_available_ns'][tf])
        assert np.all(witness['current_available_ns'][tf] <= decisions)
    for tf in ('H4', 'D1'):
        assert witness['current_available_ns'][tf][0] == BASE
        assert witness['previous_available_ns'][tf][0] == BASE - DURATIONS[tf] * MINUTE
    assert not witness['current'].flags.writeable
    assert not witness['previous'].flags.writeable
    assert witness['proof']['current_206_bit_parity'] is True


def test_future_canonical_current_id_is_rejected():
    surfaces, decisions = _fixture()
    ids, pairs = _inputs(surfaces, decisions)
    ids['D1'][0] += 1
    with pytest.raises(RuntimeError, match='CURRENT_ID_ASOF'):
        project_direction_temporal_book_witness(surfaces, decisions,
            current_native_row_indices=ids, current_206_pairs=pairs)


@pytest.mark.parametrize('missing_id', [11, 12])
def test_sparse_missing_current_or_previous_native_id_is_rejected(missing_id):
    surfaces, decisions = _fixture()
    ids, pairs = _inputs(surfaces, decisions)
    surface = surfaces['D1']
    keep = surface['native_row_indices'] != missing_id
    surfaces['D1'] = {**surface, **{name: _freeze(np.array(surface[name][keep], copy=True))
        for name in ('native_row_indices', 'available_ns', 'features')}}
    with pytest.raises(RuntimeError, match='CURRENT_ID_ASOF|PREVIOUS_ID_MISSING'):
        project_direction_temporal_book_witness(surfaces, decisions,
            current_native_row_indices=ids, current_206_pairs=pairs)


def test_observed_zero_and_unknown_zero_remain_distinct():
    surfaces, decisions = _fixture()
    witness = _project(surfaces, decisions)
    np.testing.assert_array_equal(witness['current'][0, :2], [0, 1])
    np.testing.assert_array_equal(witness['previous'][0, :2], [0, 0])
    normalization = fit_direction_temporal_book_prefix(witness, cutoff_ns=BASE + 10 * MINUTE)
    encoded = apply_direction_temporal_book_prefix(witness, normalization)
    assert encoded['current'][0, 0] != 0
    assert encoded['current'][0, 1] == 1
    np.testing.assert_array_equal(encoded['current_previous'][0, 10:12], [0, 0])


@pytest.mark.parametrize('column', [0, 1])
def test_current206_value_or_mask_bit_drift_is_rejected(column):
    surfaces, decisions = _fixture()
    ids, pairs = _inputs(surfaces, decisions)
    pairs['M5'][0, column] = np.nextafter(np.float32(0), np.float32(1)) if column == 0 else 0
    with pytest.raises(RuntimeError, match='CURRENT206_PAIR_BITS'):
        project_direction_temporal_book_witness(surfaces, decisions,
            current_native_row_indices=ids, current_206_pairs=pairs)


def test_later_source_rows_cannot_change_prefix_statistics():
    surfaces, decisions = _fixture()
    cutoff = BASE + 10 * MINUTE
    original = fit_direction_temporal_book_prefix(_project(surfaces, decisions), cutoff_ns=cutoff)
    poisoned = {}
    for tf, surface in surfaces.items():
        features = np.array(surface['features'], copy=True)
        features[surface['available_ns'] >= cutoff, BOOK_SOURCE[tf][1]] = -0.9
        poisoned[tf] = {**surface, 'features': _freeze(features)}
    changed = fit_direction_temporal_book_prefix(_project(poisoned, decisions), cutoff_ns=cutoff)
    assert changed == original
    assert original['fit_row_count'] == 2
    assert original['last_decision_ns'] < cutoff


def test_five_prefix_fits_and_same_frozen_stats_for_both_slots(monkeypatch):
    from gx1.contracts import entry_relative_price_snapshot_v1 as owner
    from gx1.contracts.entry_model_native_input_normalization_v1 import apply_surface_normalization
    surfaces, decisions = _fixture()
    witness = _project(surfaces, decisions)
    calls = []
    actual = owner._fit_surface
    def counted(values, spec, lane):
        calls.append((lane, len(values), tuple(spec['field_names'])))
        return actual(values, spec, lane)
    monkeypatch.setattr(owner, '_fit_surface', counted)
    normalization = fit_direction_temporal_book_prefix(witness, cutoff_ns=BASE + 10 * MINUTE)
    assert len(calls) == 5
    assert all(rows == 2 for _lane, rows, _fields in calls)
    frozen = copy.deepcopy(normalization)
    calls.clear()
    encoded = apply_direction_temporal_book_prefix(witness, normalization)
    assert calls == []
    assert normalization == frozen
    for slot, tf in enumerate(TIMEFRAMES):
        columns = slice(2 * slot, 2 * slot + 2)
        stats = normalization['surfaces'][tf]
        assert tuple(stats['field_names']) == BOOK_SOURCE[tf][3:]
        np.testing.assert_array_equal(encoded['current'][:, columns],
            apply_surface_normalization(witness['current'][:, columns], stats))
        np.testing.assert_array_equal(encoded['current_previous'][:, 10 + 2 * slot:12 + 2 * slot],
            apply_surface_normalization(witness['previous'][:, columns], stats))


def test_changed_prefix_statistics_are_rejected():
    surfaces, decisions = _fixture()
    witness = _project(surfaces, decisions)
    normalization = fit_direction_temporal_book_prefix(witness, cutoff_ns=BASE + 10 * MINUTE)
    normalization['surfaces']['M5']['scale'][0] *= 2
    with pytest.raises(RuntimeError, match='PREFIX_HASH'):
        apply_direction_temporal_book_prefix(witness, normalization)


def test_valid_prefix_from_other_witness_with_same_clocks_is_rejected():
    surfaces, decisions = _fixture()
    normalization = fit_direction_temporal_book_prefix(_project(surfaces, decisions),
        cutoff_ns=BASE + 10 * MINUTE)
    features = np.array(surfaces['M5']['features'], copy=True)
    features[2, BOOK_SOURCE['M5'][1]] = 0.25
    surfaces['M5'] = {**surfaces['M5'], 'features': _freeze(features)}
    other = _project(surfaces, decisions)
    with pytest.raises(RuntimeError, match='PREFIX_PARENT_BINDING'):
        apply_direction_temporal_book_prefix(other, normalization)


def test_unobserved_prefix_domain_fails_before_any_surface_fit(monkeypatch):
    from gx1.contracts import entry_relative_price_snapshot_v1 as owner
    surfaces, decisions = _fixture()
    features = np.array(surfaces['D1']['features'], copy=True)
    features[2, [BOOK_SOURCE['D1'][1], BOOK_SOURCE['D1'][2]]] = 0
    surfaces['D1'] = {**surfaces['D1'], 'features': _freeze(features)}
    def forbidden(*args, **kwargs):
        pytest.fail('No surface fit may start before all five domains have support')
    monkeypatch.setattr(owner, '_fit_surface', forbidden)
    with pytest.raises(RuntimeError, match='PREFIX_NO_CURRENT_OBSERVATIONS:D1'):
        fit_direction_temporal_book_prefix(_project(surfaces, decisions),
            cutoff_ns=BASE + 10 * MINUTE)
