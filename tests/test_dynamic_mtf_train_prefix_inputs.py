import numpy as np
import pytest

from gx1.audit.dynamic_mtf_train_prefix_inputs import _d1_prefix_masks


def test_d1_screen_uses_only_bound_mask_columns_strictly_before_cutoff():
    times = np.array([100, 200, 300, 400], dtype=np.int64)
    # The unexamined feature column is a price. At and beyond cutoff even
    # malformed mask values must have no effect on the prefix measurement.
    values = np.array([[3., 0., 1.], [4., 0., 0.], [5., 1., 1.], [6., 2., -1.]], np.float32)
    got = _d1_prefix_masks(times, values, [1, 2], 300)
    np.testing.assert_array_equal(got, np.array([[0, 1], [0, 0]], dtype=np.uint8))
    values[1, 2] = 2.
    with pytest.raises(ValueError, match='D1_MASK_DOMAIN'):
        _d1_prefix_masks(times, values, [1, 2], 300)


def test_bear_source_replay_receives_only_complete_pre_cutoff_rows(tmp_path, monkeypatch):
    import hashlib
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from gx1.audit import dynamic_mtf_gc_d1_bear_trace as trace
    from gx1.audit.dynamic_mtf_train_prefix_inputs import review_december_bear_source

    day = 86_400_000_000_000
    clock = np.arange(1, 5, dtype=np.int64) * day
    columns = {'available_ns': clock, 'bar_start_ns': clock - day, 'contract_index': [0] * 4,
               'observed_valid': [True] * 4, 'observed_contract_valid': [True] * 4}
    for name in ('open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume', 'first_minute_ns', 'last_minute_ns',
                 'clock_minutes', 'accepted_minutes'):
        columns['observed_' + name] = [1, 2, 3, -999]
    for name in trace.PARITY_FIELDS:
        columns['observed_proxy_' + name] = [0.] * 4
    columns['observed_proxy_ob_bear_state_valid'] = [False, False, True, True]
    table = tmp_path / 'TRADE_FEATURES_D1.parquet'
    pq.write_table(pa.table(columns), table)

    def binding(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def receipt(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return binding(path)

    source = binding(table)
    missing = ['gc_d1_observed_proxy_ob_bear_' + f for f in trace.BEAR_FIELDS]
    prefix = receipt('input.json', {'fit_cutoff_ns': 3 * day, 'exact_missing_fields': missing,
                     'raw_source_mask_parity': {'source': source, 'source_rows': 2,
                     'source_true_rows': {'observed_proxy_ob_bear_state_valid': 0}}})
    donor = receipt('terminal.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
                    'test_data_used': False, 'artifacts': {table.name: {'sha256': source['sha256'],
                    'bytes': table.stat().st_size}}})

    def inspect_prefix(surface):
        assert tuple(surface['available_ns']) == tuple(clock[:2])
        assert all(len(v) == 2 for v in surface.values())
        assert list(surface['observed_close_usd']) == [1, 2]
        return {}, [{}, {}]

    monkeypatch.setattr(trace, 'diagnostic_trace', inspect_prefix)
    result, rows = review_december_bear_source({'input_support': prefix, 'gc_donor_terminal': donor,
        'terminal': prefix, 'measured': {'missing_fields': missing}}, {'fit_cutoff_ns': 3 * day, 'source_dates': 2})
    assert len(rows) == result['source_dates'] == 2
    assert result['held_out_outcomes_used'] is False
