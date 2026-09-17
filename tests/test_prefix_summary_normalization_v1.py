"""Prefix normalization observes only admitted past states; lifecycle stays whole."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority, iter_physical_summary_samples,
)
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION, build_market_closure_authority, seal_exact_market_schedule,
)
from gx1.scripts import materialize_unified_exit_pilot_summary_fit_v1 as owner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, data):
    path.write_text(json.dumps(data, sort_keys=True) + '\n')


def _fixture(root: Path, *, poison_future=False):
    root.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2026-02-28T23:40:00Z', periods=40, freq='min')
    prices = 2000.0 + np.arange(40) * .2
    if poison_future:
        prices[20:] += 5000
    frame = pd.DataFrame({'time': times})
    for side, spread in [('bid', 0.0), ('ask', .3)]:
        for field, value in [('open', 0), ('high', .2), ('low', -.2), ('close', .1)]:
            frame[f'{side}_{field}'] = prices + spread + value
    m1 = root/'m1.parquet'; frame.to_parquet(m1, index=False)
    manifest = root/'m1.json'
    _write(manifest, {'split': 'train', 'output_parquet_sha256': _sha(m1),
                     'right_censor_time_utc_exclusive': '2026-06-01T00:00:00+00:00'})
    schedule = seal_exact_market_schedule({
        'schema_version': MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION, 'decision': 'PASS',
        'instrument': 'XAU_USD', 'timeframe': 'M1', 'coverage_start_utc': times[0].isoformat(),
        'coverage_end_utc_exclusive': (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        'interval_semantics': 'left_closed_right_open_utc',
        'source_method': 'externally_sourced_exact_xau_utc_closure_intervals_v1',
        'source_reference_sha256': '1'*64, 'intervals': [], 'test_data_used': False})
    schedule_path = root/'schedule.json'; _write(schedule_path, schedule)
    closure = build_market_closure_authority(
        m1_times=times, m1_source_path=m1, m1_source_sha256=_sha(m1),
        m1_source_manifest_path=manifest, m1_source_manifest_sha256=_sha(manifest),
        exact_schedule=schedule, exact_schedule_path=schedule_path,
        exact_schedule_file_sha256=_sha(schedule_path))
    closure_path = root/'closure.json'; _write(closure_path, closure)
    entries = root/'entries.parquet'
    pd.DataFrame({'time': times[[10, 14, 25]] - pd.Timedelta(minutes=5)}).to_parquet(entries, index=False)
    admission = root/'admission.json'
    _write(admission, {'decision': 'PASS', 'splits': {'train': {
        'rows': 3, 'parquet_path': str(entries), 'parquet_sha256': _sha(entries)}}})
    rows = root/'fit_rows.npy'; np.save(rows, np.array([0, 1], dtype='<i8'), allow_pickle=False)
    return dict(split='train', child_admission_path=admission, m1_path=m1,
                m1_manifest_path=manifest, closure_path=closure_path, output_dir=root/'out',
                publish=True, fit_entry_rows_path=rows,
                fit_cutoff_time_ns=pd.Timestamp('2026-03-01T00:00:00Z').value)


def test_materializer_preserves_lifecycles_and_future_prices_cannot_change_prefix_fit(tmp_path):
    kwargs = _fixture(tmp_path/'base')
    owner.materialize(**kwargs)
    original = owner._json(kwargs['output_dir']/'manifest.json')
    counts = np.load(kwargs['output_dir']/'successor_transition_counts.npy')
    limits = np.load(kwargs['output_dir']/'fit_state_stop_exclusive_by_entry.npy')
    assert counts.tolist() == [29, 25, 14]
    assert limits.tolist() == [10, 6, 0]
    assert original['successor_counts_sha256'] == hashlib.sha256(counts.tobytes()).hexdigest()
    assert original['entry_pair_population'] == 3
    scope = original['normalization_fit_population']
    assert scope['maximum_observed_decision_time_ns'] == kwargs['fit_cutoff_time_ns']
    assert scope['market_successor_counts_preserved'] is True
    authority = original['summary_sample_authority']
    assert authority['entry_pair_population'] == 3 and authority['fit_entry_pair_population'] == 2
    assert authority['successor_counts_sha256'] == original['successor_counts_sha256']
    poisoned = _fixture(tmp_path/'future_changed', poison_future=True)
    owner.materialize(**poisoned)
    changed = owner._json(poisoned['output_dir']/'manifest.json')
    assert original['m1_source_sha256'] != changed['m1_source_sha256']
    assert original['source_lineage_sha256'] != changed['source_lineage_sha256']
    assert authority['sample_stream_sha256'] == changed['summary_sample_authority']['sample_stream_sha256']
    for field in ('sample_values_sha256', 'surface', 'train_fit_rows'):
        assert original['lifetime_summary_normalization'][field] == changed['lifetime_summary_normalization'][field]
    # Original mode still includes every Entry and original physical state range.
    legacy = {**kwargs, 'output_dir': tmp_path/'legacy', 'fit_entry_rows_path': None, 'fit_cutoff_time_ns': None}
    owner.materialize(**legacy)
    old = owner._json(legacy['output_dir']/'manifest.json')
    assert 'normalization_fit_population' not in old
    assert 'fit_entry_pair_population' not in old['summary_sample_authority']
    np.testing.assert_array_equal(np.load(legacy['output_dir']/'successor_transition_counts.npy'), counts)
    assert old['lifetime_summary_normalization']['train_fit_rows'] > original['lifetime_summary_normalization']['train_fit_rows']


def test_exact_close_boundary_and_entry_identity_are_preserved():
    times = pd.date_range('2026-02-27T21:58Z', periods=3, freq='min').append(
        pd.date_range('2026-03-01T22:00Z', periods=3, freq='min'))
    starts, counts, rows = np.array([0, 1, 3]), np.array([5, 4, 2]), np.array([0, 1], dtype='<i8')
    before = counts.copy(); cutoff = pd.Timestamp('2026-02-27T22:01Z').value
    exact = owner._prefix_fit_state_stops(times=times, starts=starts, counts=counts, entry_rows=rows, cutoff_time_ns=cutoff)
    less = owner._prefix_fit_state_stops(times=times, starts=starts, counts=counts, entry_rows=rows, cutoff_time_ns=cutoff-1)
    assert exact.tolist() == [3, 2, 0] and less.tolist() == [2, 1, 0]
    np.testing.assert_array_equal(counts, before)
    samples = list(iter_physical_summary_samples(successor_transition_count_by_entry=counts.tolist(),
        source_lineage_sha256='a'*64, fit_state_stop_exclusive_by_entry=exact.tolist()))
    assert {s['entry_row_index'] for s in samples} == {0, 1}
    assert all(int(times.asi8[starts[s['entry_row_index']] + s['state_index']]) + 60_000_000_000 <= cutoff for s in samples)


@pytest.mark.parametrize('limits', [[1], [1, -1], [1, 11], [True, 0], [0, 0], [1., 0]])
def test_invalid_fit_limits_fail_closed(limits):
    with pytest.raises(RuntimeError, match='FIT_POPULATION_INVALID'):
        build_physical_summary_sample_authority(successor_transition_count_by_entry=[10, 10],
            source_lineage_sha256='a'*64, fit_state_stop_exclusive_by_entry=limits)


@pytest.mark.parametrize('mutation', ['duplicate', 'unordered', 'past_end', 'float', 'empty', 'after_cutoff'])
def test_invalid_entry_scope_is_rejected(mutation):
    rows = {'duplicate': [0, 0], 'unordered': [1, 0], 'past_end': [3],
            'float': [0., 1.], 'empty': [], 'after_cutoff': [2]}[mutation]
    with pytest.raises(RuntimeError, match='PREFIX_'):
        owner._prefix_fit_state_stops(times=pd.date_range('2026-03-01T00:00Z', periods=5, freq='min'),
            starts=np.array([0, 1, 3]), counts=np.array([4, 3, 1]),
            entry_rows=np.asarray(rows, dtype=np.float64 if mutation == 'float' else np.int64),
            cutoff_time_ns=pd.Timestamp('2026-03-01T00:03Z').value)


@pytest.mark.parametrize('split,rows,cutoff', [('val', Path('/unused'), 1), ('train', None, 1), ('train', Path('/unused'), None)])
def test_prefix_scope_rejects_before_any_file_read(split, rows, cutoff):
    with pytest.raises(RuntimeError, match='PREFIX_SCOPE_INVALID'):
        owner.materialize(split=split, child_admission_path=Path('/unused'), m1_path=Path('/unused'),
            m1_manifest_path=Path('/unused'), closure_path=Path('/unused'), output_dir=Path('/unused'),
            publish=False, fit_entry_rows_path=rows, fit_cutoff_time_ns=cutoff)
