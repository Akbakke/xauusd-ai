"""Direct raw-event M5 clocks; no derived M1 table or source metadata."""
import csv
import gzip

import numpy as np
import pytest

from gx1.contracts import gc_continuous_source_v1 as source
from gx1.contracts import gc_source_disposition_v1 as timing


def scope(bars=4):
    return dict(schema_version=source.NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
                start_ns=timing.STUDY_START_NS,
                end_ns=timing.STUDY_START_NS + bars * source.NATIVE_M5_INTERVAL_NS,
                bar_duration_ns=source.NATIVE_M5_INTERVAL_NS)


def raw_file(tmp_path, offsets):
    bound = scope()
    path = tmp_path / 'mbp.csv.gz'
    rows = [dict(ts_recv=bound['start_ns'] + offset, instrument_id=42,
                 publisher_id=1, symbol='GCM4', action='T', side='B', size=i + 1,
                 price=100000000, flags=0, sequence=i, ts_event=bound['start_ns'] + offset)
            for i, offset in enumerate(offsets)]
    with gzip.open(path, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(source.COLUMNS))
        w.writeheader()
        w.writerows(rows)
    item = dict(path=str(path), raw_symbol='GCM4', start_ns=bound['start_ns'],
                end_ns=bound['end_ns'], instrument_id=42, publisher_id=1, max_records=len(rows))
    return bound, item


def test_raw_records_reduce_directly_to_300s_without_minute_arrays(tmp_path, monkeypatch):
    interval = source.NATIVE_M5_INTERVAL_NS
    bound, item = raw_file(tmp_path, [1, interval - 1, interval, 3 * interval + 1])
    original = np.zeros
    allocations = []
    def allocate(shape, *args, **kwargs):
        allocations.append(shape)
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(source.np, 'zeros', allocate)
    callbacks = []
    out = source.scan_symbol([item], 100000000, native_m5_scope=bound,
                            on_batch=lambda data, idx, trusted: callbacks.append(idx.copy()))
    assert out['volume'].tolist() == [3, 3, 0, 4]
    assert out['trades'].tolist() == [2, 1, 0, 1]
    assert np.concatenate(callbacks).tolist() == [0, 0, 1, 3]
    assert out['native_m5_reduction_clock'] == bound
    assert all(shape == 4 for shape in allocations)


@pytest.mark.parametrize('change', [dict(bar_duration_ns=timing.MINUTE_NS),
    dict(start_ns=timing.STUDY_START_NS + 1), dict(end_ns=timing.WINDOW_END_NS + 300000000000),
    dict(m1_parent={}), dict(start_ns=True)])
def test_native_scope_rejects_minute_clock_extra_parent_and_invalid_bounds(change):
    bound = scope()
    bound.update(change)
    with pytest.raises(ValueError, match='NATIVE_M5_REDUCTION_CLOCK'):
        source.reduction_clock_v1(bound)


def test_source_file_outside_explicit_native_scope_is_rejected_before_decode(tmp_path):
    bound, item = raw_file(tmp_path, [1])
    item['end_ns'] += source.NATIVE_M5_INTERVAL_NS
    with pytest.raises(ValueError, match='REDUCTION_SOURCE_SCOPE'):
        source.scan_symbol([item], 100000000, native_m5_scope=bound)


def test_native_segments_keep_adjacent_five_minute_observations_and_reset_gap():
    begin = timing.STUDY_START_NS
    stamps = begin + np.array([0, 1, 2, 4, 5]) * source.NATIVE_M5_INTERVAL_NS
    result = source.reset_segments(np.array([0, 0, 0, 0, 1]),
        np.ones(5, bool), stamps, input_interval_ns=source.NATIVE_M5_INTERVAL_NS)
    assert result.tolist() == [0, 0, 0, 1, 2]


def test_native_quality_rejects_minute_indices_on_raw_event_clock():
    import pyarrow as pa
    bound = scope()
    quality = source.L1QualityCollector(np.arange(4), 100000000, native_m5_scope=bound)
    # Only ts_recv is needed: clock mismatch must fail before quality decoding.
    data = {'ts_recv': pa.array([bound['start_ns'] + source.NATIVE_M5_INTERVAL_NS], type=pa.uint64())}
    with pytest.raises(ValueError, match='QUALITY_RAW_CLOCK_MISMATCH'):
        quality(data, np.array([5]), np.array([True]))


def test_native_incident_excludes_overlapping_bucket_and_first_clean_anchor():
    bound = scope(6)
    stamps = bound['start_ns'] + np.arange(6) * source.NATIVE_M5_INTERVAL_NS
    q = {name: np.zeros(6, np.uint64) for name in source.L1_QUALITY_METRICS}
    q['records'][:] = q['clean_complete_bbo'][:] = 1
    m = dict(schema_version='gx1_gc_vendor_source_incidents_v1', dataset='GLBX.MDP3',
             nontrade_action_semantics_admitted=False, incidents=[dict(incident_id='synthetic',
             source_url='synthetic://source', start_inclusive_utc='2024-05-01T00:04:00Z',
             end_exclusive_utc='2024-05-01T00:06:00Z')])
    out = source.source_incident_masks_v1(stamps, np.zeros(6, np.int16), q, m,
        input_interval_ns=source.NATIVE_M5_INTERVAL_NS)
    assert out['receive_clock_excluded'].tolist() == [True, True, False, False, False, False]
    assert out['book_event_state_excluded'].tolist() == [True, True, True, False, False, False]
    evidence = out['incidents'][0]
    assert evidence['bar_duration_ns'] == source.NATIVE_M5_INTERVAL_NS
    assert evidence['overlap_buckets'] == 2
    assert not any('m1' in key.lower() or 'minute' in key.lower() for key in evidence)
