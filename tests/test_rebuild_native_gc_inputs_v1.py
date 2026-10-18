"""Mechanics/algebra checks, not evidence of historical feature quality."""
import numpy as np

from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.gc_order_flow_v1 import L1FeatureCollectorV2, L1_FIELDS_V2
from gx1.scripts.rebuild_native_gc_inputs_v1 import l1_masks, selected_minutes


def test_order_counts_do_not_change_existing_book_features():
    origin = timing.STUDY_START_NS
    events = np.array([[origin, 100, 102, 20, 30, 2, 3],
                       [origin + timing.MINUTE_NS // 2, 100, 102, 20, 30, 4, 6]], np.int64)
    original = L1FeatureCollectorV2(np.array([0]), 1)
    extended = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    original.consume(events[:, :5])
    extended.consume(events)
    original.finish(origin + timing.MINUTE_NS)
    extended.finish(origin + timing.MINUTE_NS)
    a, am = original.values([True])
    b, bm = extended.values([True])
    assert np.array_equal(am, bm)
    for key in L1_FIELDS_V2:
        np.testing.assert_array_equal(a[key], b[key])
    values, mask = extended.order_count_values([True])
    assert mask.tolist() == [True]
    assert values['tw_bid_orders'][0] == 3
    assert values['tw_ask_orders'][0] == 4.5
    assert values['tw_bid_contracts_per_order'][0] == 7.5
    assert values['tw_ask_contracts_per_order'][0] == 7.5
    assert values['tw_order_count_imbalance'][0] == -0.2


def test_unavailable_counts_preserve_valid_book_and_mask_counts():
    origin = timing.STUDY_START_NS
    collector = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    collector.consume(np.array([[origin, 100, 102, 20, 30, 0, 3]], np.int64))
    collector.finish(origin + timing.MINUTE_NS)
    _, mask = collector.values([True])
    counts, count_mask = collector.order_count_values([True])
    assert mask.tolist() == [True]
    assert count_mask.tolist() == [False]
    assert all(np.isnan(v[0]) for v in counts.values())


def test_roll_is_available_one_minute_after_prior_day():
    daily = np.array([[10, 0, 0], [0, 20, 0]], np.uint64)
    selected, chosen, _ = selected_minutes(daily, [(2024, 6), (2024, 8)])
    day = 24 * 60
    assert chosen.tolist() == [-1, 0, 1]
    assert selected[day] == -1
    assert selected[day + 1] == 0
    assert selected[2 * day] == 0
    assert selected[2 * day + 1] == 1


def test_book_mask_requires_same_contract_clean_prior_and_complete_integral():
    keys = ('bad_bbo', 'gap', 'bad_recv', 'snapshot', 'reset',
            'duplicate_trade_signature', 'file_seam')
    quality = {k: np.zeros(5, np.uint64) for k in keys}
    quality.update(records=np.ones(5, np.uint64), clean_complete_bbo=np.ones(5, np.uint64))
    ofi, bbo = l1_masks(quality, np.array([0, 0, 1, 1, 1]),
                         np.array([timing.MINUTE_NS] * 4 + [0]))
    assert ofi.tolist() == [False, True, False, True, True]
    assert bbo.tolist() == [False, True, False, True, False]
    quality['gap'][3] = 1
    ofi, _ = l1_masks(quality, np.array([0, 0, 1, 1, 1]), np.full(5, timing.MINUTE_NS))
    assert not ofi[3] and not ofi[4]


def test_footprint_sums_fragmented_price_rows_and_keeps_unknown(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.scripts.rebuild_native_gc_inputs_v1 import write_footprint
    path = tmp_path / 'selected.parquet'
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
                        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
    with pq.ParquetWriter(path, schema) as writer:
        for buy, unknown in [(2, 1), (3, 0)]:
            writer.write_table(pa.table({'minute_index': [0], 'price_tick': [100],
                                         'buy': [buy], 'sell': [0], 'unknown': [unknown]}, schema=schema))
    target = tmp_path / 'footprint.parquet'
    count = write_footprint([path], np.array([timing.STUDY_START_NS]), np.array([0]),
        {'buy': np.array([5]), 'sell': np.array([0]), 'unknown': np.array([1])},
        {'volume': np.array([True]), 'delta': np.array([False])}, target)
    assert count == 1
    values = pq.read_table(target).to_pydict()
    assert values['buy'] == [5] and values['unknown'] == [1]
    assert values['signed_valid'] == [False]



def _derived_test_plan():
    from gx1.scripts.rebuild_native_gc_inputs_v1 import DERIVED_TIMEFRAME_SCHEMA, DERIVED_TIMEFRAME_MODE
    return {'schema_version': DERIVED_TIMEFRAME_SCHEMA, 'mode': DERIVED_TIMEFRAME_MODE,
        'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
        'study_start_inclusive': '2024-05-01T00:00:00Z', 'study_end_exclusive': '2026-07-01T00:00:00Z'}


def test_derived_mode_requires_new_explicit_schema_mode_and_original_scope():
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import derived_timeframe_mode
    assert not derived_timeframe_mode({'schema_version': 'gx1_native_raw_gc_input_rebuild_plan_v1'})
    assert derived_timeframe_mode(_derived_test_plan())
    for change in ({'mode': 'raw'}, {'timeframes': ['M5']},
            {'study_end_exclusive': '2026-07-02T00:00:00Z'}):
        plan = {**_derived_test_plan(), **change}
        with pytest.raises(RuntimeError, match='DERIVED_GC_PLAN_SCOPE'):
            derived_timeframe_mode(plan)
    with pytest.raises(RuntimeError, match='DERIVED_GC_PLAN_MODE'):
        derived_timeframe_mode({'schema_version': 'unknown', 'mode': 'derive_timeframes_from_completed_m1'})


def _write_completed_footprint(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([('bar_start_ns', pa.int64()), ('contract_index', pa.int16()),
        ('price_tick', pa.int64()), ('buy', pa.uint64()), ('sell', pa.uint64()),
        ('unknown', pa.uint64()), ('signed_valid', pa.bool_())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, row_group_size=1)


def test_completed_footprint_maps_actual_m1_clock_and_retains_all_side_quantities(tmp_path):
    import pyarrow.parquet as pq
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    origin = timing.STUDY_START_NS
    minutes = origin + np.arange(3, dtype=np.int64) * timing.MINUTE_NS
    path, output = tmp_path / 'footprint.parquet', tmp_path / 'ladder.parquet'
    # Published raw footprint order is by contract, not globally by M1 clock.
    _write_completed_footprint(path, [
        {'bar_start_ns': int(minutes[2]), 'contract_index': 0, 'price_tick': 100,
            'buy': 3, 'sell': 2, 'unknown': 1, 'signed_valid': False},
        {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 101,
            'buy': 4, 'sell': 0, 'unknown': 0, 'signed_valid': True},
        {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 102,
            'buy': 1, 'sell': 2, 'unknown': 0, 'signed_valid': True}])
    metrics = {'buy': np.array([5, 0, 3], np.uint64), 'sell': np.array([2, 0, 2], np.uint64),
        'unknown': np.array([0, 0, 1], np.uint64)}
    masks = {'volume': np.array([True, False, True]), 'delta': np.array([True, False, False])}
    assert stream_completed_footprint(path, minutes, np.array([1, 1, 0]), metrics, masks, output) == 3
    result = pq.read_table(output).to_pydict()
    assert result['minute_index'] == [2, 0, 0]
    assert result['unknown'] == [1, 0, 0]
    assert result['buy'] == [3, 4, 1]


def test_completed_footprint_rejects_wrong_side_sum_and_missing_admitted_minute(tmp_path):
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    origin = timing.STUDY_START_NS
    minutes = origin + np.arange(2, dtype=np.int64) * timing.MINUTE_NS
    path = tmp_path / 'footprint.parquet'
    _write_completed_footprint(path, [{'bar_start_ns': int(minutes[0]), 'contract_index': 0,
        'price_tick': 100, 'buy': 3, 'sell': 2, 'unknown': 0, 'signed_valid': True}])
    masks = {'volume': np.array([True, True]), 'delta': np.array([True, True])}
    metrics = {'buy': np.array([4, 1]), 'sell': np.array([2, 1]), 'unknown': np.array([0, 0])}
    with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_SIDE_CONSERVATION'):
        stream_completed_footprint(path, minutes, np.array([0, 0]), metrics, masks, tmp_path / 'bad-side.parquet')
    metrics['buy'][0] = 3
    with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_MINUTE_CONSERVATION'):
        stream_completed_footprint(path, minutes, np.array([0, 0]), metrics, masks, tmp_path / 'missing-minute.parquet')


def test_completed_footprint_rejects_contract_and_exact_clock_mismatch(tmp_path):
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    minutes = np.array([timing.STUDY_START_NS], np.int64)
    row = {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 100,
        'buy': 3, 'sell': 2, 'unknown': 0, 'signed_valid': True}
    metrics = {'buy': np.array([3]), 'sell': np.array([2]), 'unknown': np.array([0])}
    masks = {'volume': np.array([True]), 'delta': np.array([True])}
    for index, change in enumerate(({}, {'contract_index': 0, 'bar_start_ns': int(minutes[0]) + 1})):
        path = tmp_path / f'bad-{index}.parquet'
        _write_completed_footprint(path, [{**row, **change}])
        with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_M1_MAPPING'):
            stream_completed_footprint(path, minutes, np.array([0]), metrics, masks, tmp_path / f'out-{index}.parquet')


def test_derived_inputs_require_completed_original_inventory_and_exact_hashes(tmp_path):
    import hashlib
    import json
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import DERIVED_INPUT_NAMES, bound_derived_inputs
    inputs, artifacts = {}, {}
    for name, filename in DERIVED_INPUT_NAMES.items():
        path = tmp_path / filename
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[name] = {'path': str(path), 'sha256': digest}
        artifacts[filename] = {'sha256': digest, 'bytes': path.stat().st_size}
    terminal = tmp_path / 'TERMINAL.json'
    value = {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
             'test_data_used': False, 'artifacts': artifacts}
    terminal.write_text(json.dumps(value))
    inputs['source_terminal'] = {'path': str(terminal), 'sha256': hashlib.sha256(terminal.read_bytes()).hexdigest()}
    assert set(bound_derived_inputs({'inputs': inputs})) == set(inputs)
    value['exit_code'] = 1
    terminal.write_text(json.dumps(value))
    inputs['source_terminal']['sha256'] = hashlib.sha256(terminal.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match='DERIVED_GC_SOURCE_NOT_COMPLETE'):
        bound_derived_inputs({'inputs': inputs})
