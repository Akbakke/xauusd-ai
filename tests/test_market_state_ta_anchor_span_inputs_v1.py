"""Synthetic sidecar gates only: no market arrays, targets, models or fits."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as worker
from gx1.features.trendline_registry_v1 import TRENDLINE_REGISTRY_FEATURE_NAMES_V1


def sample():
    names = list(TRENDLINE_REGISTRY_FEATURE_NAMES_V1)
    values = np.zeros((3, len(names)), dtype=np.float32)
    for side, count, age, span in (
        ('above', [1, 0, 2], [40, 0, 50], [20, 0, 30]),
        ('below', [0, 1, 1], [0, 20, 25], [0, 10, 10]),
    ):
        for suffix, column in (('active_count', count), ('age_bars', age),
                               ('anchor_span_bars', span)):
            values[:, names.index('geomline_' + side + '_' + suffix)] = column
    values[0, names.index('geomline_touch_above')] = np.float32(-0.0)
    positions = np.asarray([100, 101, 102], dtype=np.int64)
    legacy = [name for name in names if name not in worker.FIELDS]
    observed = values[:, [names.index(name) for name in legacy]].copy()
    return names, values, observed, positions


def extract(values, observed, names, positions):
    return worker.compare_and_extract(values, observed, names, positions, lookback=3)


def test_old_fields_are_bit_exact_and_spans_are_owned_without_input_mutation():
    names, values, observed, positions = sample()
    before = [a.tobytes() for a in (values, observed, positions)]
    spans, report = extract(values, observed, names, positions)
    assert np.array_equal(spans, np.asarray([[20, 0], [0, 10], [30, 10]], np.float32))
    assert spans.dtype == np.float32 and spans.flags.owndata
    assert [a.tobytes() for a in (values, observed, positions)] == before
    assert report['rows'] == 3 and report['fields'] == 31
    assert report['float32_bit_mismatched_cells'] == 0
    assert report['legacy_values_sha256'] == hashlib.sha256(observed.tobytes()).hexdigest()
    assert report['new_fields']['above'] == dict(active_rows=2, absent_rows=1,
        unique_positive_spans=2, min_positive_span=20., max_positive_span=30.)
    spans[:] = 999
    assert values.tobytes() == before[0]


def test_signed_zero_old_field_mismatch_is_not_numerical_equality():
    names, values, observed, positions = sample()
    legacy = [name for name in names if name not in worker.FIELDS]
    observed[0, legacy.index('geomline_touch_above')] = np.float32(0.)
    with pytest.raises(RuntimeError, match='LEGACY_FLOAT32_BITS:1'):
        extract(values, observed, names, positions)


@pytest.mark.parametrize('fault', ('shape', 'nonfinite'))
def test_old_field_shape_and_nonfinite_fail_before_span_admission(fault):
    names, values, observed, positions = sample()
    if fault == 'shape':
        observed = observed[:-1]
    else:
        values[0, 0] = observed[0, 0] = np.float32(np.nan)
    with pytest.raises(RuntimeError, match='LEGACY_FINITE_SHAPE'):
        extract(values, observed, names, positions)


@pytest.mark.parametrize('field,row,value,error', (
    ('anchor_span_bars', 1, 5., 'SPAN_DOMAIN:above'),
    ('anchor_span_bars', 0, 0., 'SPAN_DOMAIN:above'),
    ('anchor_span_bars', 0, 1.5, 'SPAN_DOMAIN:above'),
    ('anchor_span_bars', 0, np.inf, 'SPAN_FINITE'),
    ('age_bars', 0, 40.5, 'SPAN_DOMAIN:above'),
    ('age_bars', 0, 200., 'ANCHOR_CAUSALITY:above'),
    ('anchor_span_bars', 0, 41., 'ANCHOR_CAUSALITY:above'),
))
def test_span_absence_integer_domain_and_both_anchor_causality(field, row, value, error):
    names, values, observed, positions = sample()
    values[row, names.index('geomline_above_' + field)] = value
    # The retained old fields agree: this case isolates the new span/anchor gate.
    legacy = [name for name in names if name not in worker.FIELDS]
    observed = values[:, [names.index(name) for name in legacy]].copy()
    with pytest.raises(RuntimeError, match=error):
        extract(values, observed, names, positions)


def test_exact_clock_mapping_preserves_gaps_and_rejects_unmatched_times():
    clock = np.asarray([0, 300, 900, 1200, 1800], dtype=np.int64) * 10**9
    times = clock[[1, 2, 4]].copy()
    assert np.array_equal(worker.selected_positions(clock, times), [1, 2, 4])
    assert worker.axis_sha(times) == hashlib.sha256(times.astype('<i8').tobytes()).hexdigest()
    for bad in (np.asarray([600, 900]), np.asarray([900, 2100])):
        with pytest.raises(RuntimeError, match='EXACT_SOURCE_CLOCK'):
            worker.selected_positions(clock, bad * 10**9)


def test_declared_intervals_match_retained_namespace_row_hashes():
    expected = {
        'local': 'aaf7bb699db0eb3602ef5c44ec195780b24368c788f424fddc611270a300d9a1',
        'M5': 'd5999966c354bf9dfba62fb21d451f3923a6b5fe6e2acd74f4aca1126d2a3266',
        'M15': 'b74d7583c7f8ad370ec8acbfc2aff3f0b57a82fa8b9dba042f0d92d366e10def',
        'H1': 'b3b0cf82a464ee5cc48478aa67d1951be1641e5235a4fa74c2a4380a2d63a240',
        'H4': '2370aabbecbd17b03758dd181a5fbc3c26399ccd9581a9c5eb361f39042ff518',
        'D1': 'd5cc7da7b09bdffc992a624c650a3ffc6d8d5911e245c39429d6d60a383bcda7',
    }
    for lane, (left, right) in worker.RANGES.items():
        rows = np.arange(left, right, dtype=np.int64)
        assert worker.row_sha(rows, lane) == expected[lane]
        assert worker.row_sha(rows, lane) != worker.axis_sha(rows)


@pytest.mark.parametrize('status,result,failure', (
    ('COMPLETE', None, None), ('COMPLETE', {'path': 'result'}, {'path': 'failure'}),
    ('FAILED_CONSUMED', {'path': 'result'}, None),
))
def test_terminal_rejects_success_failure_mismatch(status, result, failure):
    with pytest.raises(RuntimeError, match='TERMINAL_OUTCOME'):
        worker.terminal_record(status=status, plan_ref={}, result=result, failure=failure,
                               head='a' * 40, elapsed=0.)


def mocked_run(monkeypatch, tmp_path, fault):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    published = {}
    plan = {'source_owners': {}, 'inputs': {}}
    digest, head = 'a' * 64, 'b' * 40
    monkeypatch.setattr(worker, 'ROOT', tmp_path / 'run')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(cap, 'require_capped_cpu_audit_execution', lambda: {'synthetic': True})
    monkeypatch.setattr(worker, 'inspect', lambda *a, **k: (
        plan, {'path': 'plan', 'sha256': digest}, {'path': 'policy', 'sha256': digest}, head))
    monkeypatch.setattr(worker, 'sha', lambda _: digest)
    heads_read = []
    def check_output(args, **kw):
        if args[1] != 'rev-parse':
            return ''
        heads_read.append(None)
        return 'changed' if (fault == 'bookend' or
            (fault == 'final_bookend' and len(heads_read) == 2)) else head
    monkeypatch.setattr(worker.subprocess, 'check_output', check_output)
    monkeypatch.setattr(worker.signal, 'getitimer', lambda _: (0., 0.))
    monkeypatch.setattr(worker.signal, 'getsignal', lambda _: 'original')
    monkeypatch.setattr(worker.signal, 'signal', lambda *a: None)
    timers = []
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: timers.append(args))

    def publish(name, value):
        assert name not in published
        published[name] = deepcopy(value)
        return {'path': name, 'sha256': digest}

    def measure(_):
        publish('local_QUALIFICATION.json', {'status': 'synthetic_partial'})
        if fault == 'measure':
            raise RuntimeError('synthetic_measure_failure')
        return {'local': {'synthetic': True}}

    monkeypatch.setattr(worker, 'publish_json', publish)
    monkeypatch.setattr(worker, 'measure', measure)
    return published, timers, digest


@pytest.mark.parametrize('fault', ('none', 'measure', 'bookend', 'final_bookend'))
def test_run_preserves_partial_evidence_and_terminal_outcome(monkeypatch, tmp_path, fault):
    published, timers, digest = mocked_run(monkeypatch, tmp_path, fault)
    if fault == 'none':
        worker.run(digest)
    else:
        with pytest.raises(RuntimeError, match='synthetic_measure_failure|SOURCE_BOOKEND'):
            worker.run(digest)
    assert 'CLAIM.json' in published and 'local_QUALIFICATION.json' in published
    terminal = published['TERMINAL.json']
    assert terminal['status'] == ('COMPLETE' if fault == 'none' else 'FAILED_CONSUMED')
    assert terminal['exit_code'] == (0 if fault == 'none' else 1)
    assert terminal['source_unchanged'] == (fault not in ('bookend', 'final_bookend'))
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert terminal['test_data_used'] is False and timers[-1][1] == 0
    assert ('RESULT.json' in published) == (fault in ('none', 'final_bookend'))
    assert ('FAILED.json' in published) == (fault != 'none')
    if fault == 'none':
        result = published['RESULT.json']
        assert result['normalization_fitted'] is result['fresh_model_baseline_complete'] is False
        assert result['old_cache_modified'] is result['predictive_value_claimed'] is False


def test_execution_guard_fails_before_inspection_or_claim(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    def fail():
        raise RuntimeError('synthetic_cap_required')
    def forbidden(*a, **kw):
        raise AssertionError('must not inspect, claim or decode before cap')
    monkeypatch.setattr(cap, 'require_capped_cpu_audit_execution', fail)
    monkeypatch.setattr(worker, 'inspect', forbidden)
    monkeypatch.setattr(worker, 'publish_json', forbidden)
    monkeypatch.setattr(worker, 'measure', forbidden)
    with pytest.raises(RuntimeError, match='synthetic_cap_required'):
        worker.run('a' * 64)


@pytest.mark.parametrize('lanes,declared,ranges,accepted', (
    (None, None, ['local', 'M5', 'M15', 'H1', 'H4', 'D1'], True),
    (None, None, ['local', 'M5', 'M15'], False),
    (('local', 'M5', 'M15'), ['local', 'M5', 'M15'], ['local', 'M5', 'M15'], True),
    (('local', 'M5', 'M15'), ['local', 'M5'], ['local', 'M5', 'M15'], False),
    (('M5', 'M5'), ['M5', 'M5'], ['M5'], False),
    (('UNKNOWN',), ['UNKNOWN'], ['UNKNOWN'], False),
    ((), [], [], False),
    (('M5', 'local'), ['M5', 'local'], ['local', 'M5'], False),
))
def test_replay_subset_is_explicit_unique_ordered_and_default_stays_all(
        monkeypatch, tmp_path, lanes, declared, ranges, accepted):
    bounds = {lane: [1, 2] for lane in ranges}
    scope = dict(ranges=bounds, history_end_exclusive_ns=100)
    if declared is not None:
        scope['replay_lanes'] = declared
    plan = dict(scope=scope, output_root=str(tmp_path),
                inputs=dict(enriched={'path': 'synthetic'}, native_parts=[]), caches={})
    def stop_before_io(*args):
        raise RuntimeError('SYNTHETIC_IO_BOUNDARY')
    monkeypatch.setattr(worker, 'stat', stop_before_io)
    expected = 'SYNTHETIC_IO_BOUNDARY' if accepted else 'EXPLICIT_REPLAY_(LANES|SCOPE)'
    with pytest.raises(RuntimeError, match=expected):
        worker.measure(plan, ranges=bounds, cutoff=100, output_root=tmp_path, lanes=lanes)
