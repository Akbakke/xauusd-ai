"""Synthetic relative-coordinate statistics only; no model or market payloads."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import gc_relative_price_coordinates_v1 as coordinates
from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract


def _seal(value):
    value['contract_sha256'] = history._sha({k: v for k, v in value.items() if k != 'contract_sha256'})
    return value


def _coordinates(layout):
    return coordinates.compile_gc_native_relative_price_layout(history_layout=layout,
        source_contract=build_gc_price_coordinate_source_contract(source_mode='native_current'), tick_size_raw=100_000_000)


def _case(*, rows=6, selected_rows=4):
    layout = history.gc_mtf_history_layout()
    relative = _coordinates(layout)
    surfaces, selected = {}, {}
    for tf in history.TIMEFRAMES:
        names = layout['field_names'][tf]
        values = (np.arange(rows, dtype=np.float32)[:, None] + 1
            + np.arange(len(names), dtype=np.float32)[None, :] / np.float32(1000)).astype(np.float32)
        for name in layout['binary_fields'][tf]:
            values[:, names.index(name)] = 1.
        item = relative['surfaces'][tf]
        for ordinal, (vi, unit) in enumerate(zip(item['level_value_indices'], item['level_units'])):
            price = 2000. + np.arange(rows, dtype=np.float32) * 100. + ordinal
            values[:, vi] = price / (.1 if unit == 'ticks' else 1.)
        selected[tf] = np.arange(selected_rows, dtype=np.int64)
        surfaces[tf] = {'native_row_indices': np.arange(rows, dtype=np.int64),
            'available_ns': (np.arange(rows, dtype=np.int64) + 1) * history.TIMEFRAME_MINUTES[tf] * history.MINUTE_NS,
            'features': values, 'field_names': list(names)}
    cutoff = int((selected_rows + 1) * history.TIMEFRAME_MINUTES['D1'] * history.MINUTE_NS)
    return layout, relative, surfaces, selected, cutoff


def _fit(case, *, relative=True):
    layout, coords, surfaces, selected, cutoff = case
    return history.fit_gc_mtf_history_normalization(surfaces, selected, train_start_ns=0,
        train_end_ns=cutoff, layout=layout, relative_price_coordinates=coords if relative else None)


def _mask_all_levels(case, tf, rows):
    _layout, coords, surfaces, _selected, _cutoff = case
    item = coords['surfaces'][tf]
    values = surfaces[tf]['features']
    values[np.ix_(rows, item['level_value_indices'])] = 0.
    values[np.ix_(rows, item['level_mask_indices'])] = 0.
    # Any nonlevel sharing a disabled original observation mask must stay unknown too.
    for vi, mi in _layout['value_mask_pairs'][tf]:
        values[values[:, mi] == 0., vi] = 0.


def test_omitted_and_explicit_none_keep_legacy_output_hash_and_helpers():
    case = _case()
    layout, _coords, surfaces, selected, cutoff = case
    old = history.fit_gc_mtf_history_normalization(surfaces, selected,
        train_start_ns=0, train_end_ns=cutoff, layout=layout)
    assert old == _fit(case, relative=False)
    assert old['schema_version'] == history.NORMALIZATION_SCHEMA_VERSION
    assert 'relative_price' not in old
    wrapped = {'layout': layout, 'normalization': old}
    assert history.gc_mtf_model_layout(wrapped) == layout
    assert history.gc_mtf_normalization_surfaces(wrapped) == old['surfaces']


def test_relative_fit_keeps_raw_inputs_masks_and_population_with_separate_observed_anchor():
    case = _case()
    layout, coords, surfaces, selected, _cutoff = case
    for tf in history.TIMEFRAMES:
        _mask_all_levels(case, tf, [0, 3])
    before = {tf: x['features'].copy() for tf, x in surfaces.items()}
    fitted = _fit(case)
    assert fitted['schema_version'] == history.RELATIVE_NORMALIZATION_SCHEMA_VERSION
    assert fitted['relative_price']['layout'] == coords
    for tf in history.TIMEFRAMES:
        np.testing.assert_array_equal(surfaces[tf]['features'], before[tf])
        transformed = coordinates.transform_gc_native_relative_price_block(before[tf][selected[tf]],
            layout=coords, timeframe=tf)
        observed = transformed['anchor_observed'].astype(bool)
        assert observed.tolist() == [False, True, True, False]
        anchor = fitted['relative_price']['anchor_surfaces'][tf]
        expected = norm.fit_surface_normalization(transformed['anchor_usd'][observed, None],
            surface='gc_anchor_' + tf.lower(), field_names=['gc_relative_anchor_usd'], row_count=2,
            semantic_categorical_domains={}, semantic_binary_fields=[], allow_constant_train_fields=True)
        assert anchor == expected
        assert anchor['center'] == [2150.]
        raw_stats = fitted['surfaces']['gc_mtf_' + tf.lower()]
        assert raw_stats['field_names'] == layout['field_names'][tf]
        assert raw_stats['fit_row_count'] == fitted['provenance']['row_union'][tf]['rows'] == 4
        expected_raw = norm.fit_surface_normalization(transformed['values'][:, :layout['widths'][tf]],
            surface='gc_mtf_' + tf.lower(), field_names=layout['field_names'][tf], row_count=4,
            semantic_categorical_domains=layout['categorical_domains'][tf])
        assert raw_stats == expected_raw
    wrapped = {'layout': layout, 'normalization': fitted}
    model = history.gc_mtf_model_layout(wrapped)
    assert {k: v for k, v in model.items() if k not in ('field_names', 'widths', 'family_indices')} == {
        k: v for k, v in layout.items() if k not in ('field_names', 'widths', 'family_indices')}
    for tf in history.TIMEFRAMES:
        assert model['widths'][tf] == layout['widths'][tf] + 2
        assert model['field_names'][tf][-2:] == ['gc_relative_anchor_usd', 'gc_relative_anchor_observed']
        assert model['family_indices'][tf] == coords['surfaces'][tf]['family_indices']
    flattened = history.gc_mtf_normalization_surfaces(wrapped)
    assert set(flattened) == {prefix + tf.lower() for tf in history.TIMEFRAMES for prefix in ('gc_mtf_', 'gc_anchor_')}
    assert layout == history.gc_mtf_history_layout()


def test_relative_fit_ignores_rows_outside_declared_train_union():
    case = _case()
    before = _fit(case)
    for tf, surface in case[2].items():
        # Change only nonbinary values of rows excluded from the declared fit union.
        names = case[0]['field_names'][tf]
        for i, name in enumerate(names):
            if name not in case[0]['binary_fields'][tf]:
                surface['features'][4:, i] *= 5.
    assert _fit(case) == before


@pytest.mark.parametrize('observed_rows', [0, 1])
def test_relative_fit_rejects_fewer_than_two_observed_anchor_rows(observed_rows):
    case = _case()
    _mask_all_levels(case, 'M5', list(range(observed_rows, 4)))
    with pytest.raises(RuntimeError, match='ANCHOR_OBSERVED_FIT:M5'):
        _fit(case)


def test_constant_observed_anchor_is_allowed_without_fabricating_missing_rows():
    case = _case()
    for tf in history.TIMEFRAMES:
        item = case[1]['surfaces'][tf]
        values = case[2][tf]['features']
        values[:, item['level_value_indices']] = values[0, item['level_value_indices']]
    fitted = _fit(case)
    for tf in history.TIMEFRAMES:
        anchor = fitted['relative_price']['anchor_surfaces'][tf]
        assert anchor['center'] == [2000.] and anchor['fit_row_count'] == 4
        assert anchor['scale'][0] > 0 and anchor['binary_mask'] == anchor['categorical_mask'] == [0]


@pytest.mark.parametrize('invalid_count', [1, 5, 4.0])
def test_resealed_anchor_row_count_cannot_exceed_raw_population_or_change_type(invalid_count):
    case = _case()
    fitted = _fit(case)
    fitted['relative_price']['anchor_surfaces']['M5']['fit_row_count'] = invalid_count
    _seal(fitted)
    with pytest.raises(RuntimeError):
        history.require_gc_mtf_history_normalization(fitted, layout=case[0])


@pytest.mark.parametrize('fault', ['schema', 'raw_width', 'missing_anchor_surface', 'anchor_field'])
def test_resealed_relative_metadata_changes_are_rejected(fault):
    case = _case()
    fitted = _fit(case)
    if fault == 'schema':
        fitted['schema_version'] = history.NORMALIZATION_SCHEMA_VERSION
    elif fault == 'raw_width':
        nested = fitted['relative_price']['layout']
        nested['surfaces']['M5']['raw_width'] += 1
        _seal(nested)
    elif fault == 'missing_anchor_surface':
        fitted['relative_price']['anchor_surfaces'].pop('M5')
    else:
        fitted['relative_price']['anchor_surfaces']['M5']['field_names'] = ['unbound_anchor']
    _seal(fitted)
    with pytest.raises(RuntimeError):
        history.require_gc_mtf_history_normalization(fitted, layout=case[0])


def test_prefix_opt_in_changes_only_gc_normalization_and_keeps_raw_qualification():
    from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix
    from tests.test_entry_native_direction_prefix_normalization_v1 import prefix_case
    case = prefix_case()
    qualified = prefix.qualify_native_direction_prefix_population(**case)
    assert qualified['report']['fit_allowed'] is True
    raw_report = deepcopy(qualified['report'])
    old = prefix.fit_native_direction_prefix_normalization(qualified)
    relative = prefix.fit_native_direction_prefix_normalization(qualified,
        gc_relative_coordinates=_coordinates(case['gc_layout']))
    assert qualified['report'] == raw_report
    assert relative['schema_version'] == old['schema_version'] == prefix.SCHEMA_VERSION
    assert relative['prefix_qualification'] == old['prefix_qualification'] == raw_report
    assert relative['base_normalization'] == old['base_normalization']
    assert relative['gc_mtf_history']['layout'] == old['gc_mtf_history']['layout']
    assert relative['gc_mtf_history']['normalization']['contract_sha256'] != old['gc_mtf_history']['normalization']['contract_sha256']
    assert relative['contract_sha256'] != old['contract_sha256']


def test_relative_transformation_uses_existing_block_limit(monkeypatch):
    from gx1.contracts.entry_direction_snapshot_data_v1 import MAX_BLOCK_ROWS
    case = _case(rows=MAX_BLOCK_ROWS + 2, selected_rows=MAX_BLOCK_ROWS + 1)
    original = coordinates.transform_gc_native_relative_price_block
    calls = []

    def observed(raw, *, layout, timeframe):
        calls.append((timeframe, len(raw)))
        return original(raw, layout=layout, timeframe=timeframe)

    monkeypatch.setattr(coordinates, 'transform_gc_native_relative_price_block', observed)
    _fit(case)
    assert calls == [(tf, size) for tf in history.TIMEFRAMES for size in (MAX_BLOCK_ROWS, 1)]
