"""Synthetic row merge and claimed output paths for the six-lane extension."""
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import qualify_market_state_ta_fivefold_span_extension_v1 as work


def _block(row, time):
    return {'row_indices': np.array([row], np.int64),
            'times_ns': np.array([time], np.int64),
            'native_bar_positions': np.array([row + 10], np.int64),
            'anchor_span_bars': np.array([[float(row), float(row + 1)]], np.float32)}


def _scope():
    rows = np.arange(2, dtype=np.int64)
    times = np.array([1_000_000_000_000, 1_300_000_000_000], np.int64)
    return {'old_range': [0, 1], 'new_range': [1, 2],
            'old_sidecar': {'path': 'synthetic-old', 'sha256': 'a' * 64},
            'required_superrange': [0, 2],
            'new_population': {'selected_unique_row_count': 1},
            'full_population': {
                'selected_unique_row_count': 2,
                'selected_row_indices_sha256': work.replay.row_sha(rows, 'local'),
                'available_ns_sha256': work.replay.axis_sha(
                    times + work.replay.BAR_NS['M5'] * 10**9)}}


def test_merge_requires_exact_adjacent_rows_and_availability():
    old, new = _block(0, 1_000_000_000_000), _block(1, 1_300_000_000_000)
    scope = _scope()
    merged = work.merge(old, new, lane='local', scope=scope)
    assert merged['row_indices'].tolist() == [0, 1]
    assert merged['anchor_span_bars'].tolist() == [[0., 1.], [1., 2.]]
    wrong = dict(scope, new_range=[2, 3])
    with pytest.raises(RuntimeError, match='ADJACENT_RANGES'):
        work.merge(old, new, lane='local', scope=wrong)
    changed = dict(new, times_ns=np.array([1_200_000_000_000], np.int64))
    with pytest.raises(RuntimeError, match='FULL_POPULATION'):
        work.merge(old, changed, lane='local', scope=scope)


@pytest.mark.parametrize('failure_lane', [None, 'H1'])
def test_claimed_run_writes_terminal_for_success_or_partial_failure(monkeypatch, tmp_path,
                                                                     failure_lane):
    out = tmp_path / 'RUN'
    monkeypatch.setattr(work, 'ROOT', out)
    monkeypatch.setattr(work, 'PLAN', tmp_path / 'PLAN.json')
    monkeypatch.setattr(work, 'SCOPE', {'new_rows_total': 6, 'merged_rows_total': 12})
    scopes = {lane: _scope() for lane in work.LANES}
    coverage = {'lanes': scopes}
    old = {'inputs': {'native_parts': []}, 'populations': {},
           'scope': {}, 'output_root': ''}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    local_qual = {'population': scopes['local']['new_population'],
                  'parity': {'float32_bit_mismatched_cells': 0, 'fields': 31},
                  'fresh_state_at_source_row_zero': True,
                  'artifact': {'path': 'local', 'sha256': 'a' * 64}}
    monkeypatch.setattr(work, 'preflight', lambda *_a, **_kw:
                        ({'coverage': {}, 'lane_scopes': scopes, 'source_owners': {},
                          'native_2025_part': {'path': 'native2025', 'sha256': 'c' * 64},
                          'reuse_local_qualification': {'path': 'qual', 'sha256': 'd' * 64}},
                         coverage, {}, old, 'head', policy_ref, local_qual))
    monkeypatch.setattr(work, 'bookend', lambda *_a: None)
    def measure(wrapper, **_kw):
        assert wrapper['scope']['history_end_exclusive_ns'] == work.CUTOFF_NS
        assert wrapper['scope']['array_output_budget_bytes'] == 4 * 1024**2
        assert wrapper['scope']['replay_lanes'] == list(work.NATIVE_LANES)
        assert len(wrapper['inputs']['native_parts']) == 1
        return {lane: {'population': scopes[lane]['new_population'],
                       'parity': {'float32_bit_mismatched_cells': 0, 'fields': 31},
                       'fresh_state_at_source_row_zero': True,
                       'artifact': {'path': lane, 'sha256': 'a' * 64}}
                for lane in work.NATIVE_LANES}
    monkeypatch.setattr(work.replay, 'measure', measure)
    monkeypatch.setattr(work, 'archive', lambda item, *, expected_range:
                        _block(expected_range[0],
                               1_000_000_000_000 if expected_range[0] == 0
                               else 1_300_000_000_000))
    original = work.merge
    seen = []
    def checked_merge(first, second, *, lane, scope):
        seen.append(lane)
        if lane == failure_lane:
            raise RuntimeError('synthetic physical merge fault')
        # Use the real merger with the local synthetic hash on every lane.
        local = dict(scope, full_population=_scope()['full_population'])
        return original(first, second, lane='local', scope=local)
    monkeypatch.setattr(work, 'merge', checked_merge)
    if failure_lane:
        with pytest.raises(RuntimeError, match='synthetic physical merge fault'):
            work.run(Path('plan'), 'a' * 64)
        assert (out / 'FAILED.json').is_file()
        assert seen == ['local', 'M5', 'M15', 'H1']
    else:
        result = work.run(Path('plan'), 'a' * 64)
        assert result['terminal']['path'] == str(out / 'TERMINAL.json')
        assert seen == list(work.LANES)
        assert (out / 'RESULT.json').is_file()
    terminal = json.loads((out / 'TERMINAL.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure_lane else 'COMPLETE')
    assert (out / 'CLAIM.json').is_file()
