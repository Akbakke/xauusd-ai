"""Focused guards for the TRAIN-only independent D1 comparison."""
import json

import numpy as np
import pytest

from gx1.audit.verify_d1_trendline_lifecycle_physical_v1 import (
    DAY_NS, NAMES, REPO, STAGE, TRAIN_CACHE_ROWS, TRAIN_END_NS,
    TRAIN_START_NS, WITNESSES, align_train_rows, align_witnesses,
    compare_train_rows, read_claim,
)


def axes():
    cache = np.empty(3889, dtype=np.int64)
    first = WITNESSES[0][2]
    cache[:3588] = np.arange(TRAIN_START_NS - DAY_NS - 3588,
                             TRAIN_START_NS - DAY_NS, dtype=np.int64)
    points = [(row, timestamp) for row, _, timestamp in WITNESSES]
    points.append((3761, TRAIN_END_NS - DAY_NS - 2 * 3600 * 10**9))
    cache[3588] = first
    for (left, start), (right, end) in zip(points, points[1:]):
        cache[left:right + 1] = np.linspace(
            start, end, right - left + 1, dtype=np.int64)
    cache[3762:] = np.arange(TRAIN_END_NS - DAY_NS,
                             TRAIN_END_NS - DAY_NS + (3889 - 3762),
                             dtype=np.int64)
    native = np.concatenate((np.arange(cache[0] - 269, cache[0], dtype=np.int64),
                             cache))
    records = [{'cache_row': row, 'native_bar_index': pos,
                'timestamp_ns': timestamp} for row, pos, timestamp in WITNESSES]
    return cache, native, records


def test_all_174_closed_d1_train_rows_and_four_anchors_align():
    cache, native, records = axes()
    rows, positions, timestamps = align_train_rows(cache, native)
    assert len(rows) == 174
    assert rows.tolist() == list(TRAIN_CACHE_ROWS)
    assert positions.tolist() == [row + 269 for row in TRAIN_CACHE_ROWS]
    assert WITNESSES[0][2] < TRAIN_START_NS
    assert WITNESSES[0][2] + DAY_NS >= TRAIN_START_NS
    assert align_witnesses(cache, native, records) == [x[1] for x in WITNESSES]
    cache[WITNESSES[1][0]] += 1
    with pytest.raises(RuntimeError, match='TRAIN_NATIVE_JOIN'):
        align_train_rows(cache, native)


def test_all_31_fields_and_19_lifecycle_fields_require_comparison():
    rows = np.asarray(TRAIN_CACHE_ROWS, dtype=np.int64)
    times = TRAIN_START_NS + np.arange(174, dtype=np.int64) * DAY_NS
    oracle = np.zeros((174, 31), np.float64)
    oracle[:, NAMES.index('geomline_bars_since_break_signed')] = np.nan
    cache = oracle.astype(np.float32)
    report = compare_train_rows(oracle, cache, rows, times)
    assert (report['compared_cells'], report['lifecycle_cells'],
            report['mismatched_cells']) == (5394, 3306, 0)
    assert report['all_train_window_cells_match'] is True
    assert 'geomline_break_up' in report['unexercised_lifecycle_fields']
    assert report['mismatch_examples'] == []
    cache[0, NAMES.index('geomline_bars_since_break_signed')] = 0
    cache[1, NAMES.index('geomline_break_up')] = 1
    report = compare_train_rows(oracle, cache, rows, times)
    assert report['mismatched_cells'] == report['lifecycle_mismatched_cells'] == 2
    assert report['anchor_mismatched_cells'] == 1
    assert len(report['mismatch_examples']) == 2


def test_discrete_mismatch_is_exact_and_float_tolerance_is_bounded():
    rows = np.asarray(TRAIN_CACHE_ROWS, dtype=np.int64)
    times = TRAIN_START_NS + np.arange(174, dtype=np.int64) * DAY_NS
    oracle = np.zeros((174, 31), np.float64)
    cache = np.zeros((174, 31), np.float32)
    slope = NAMES.index('geomline_above_slope_atr_per_bar')
    count = NAMES.index('geomline_above_touch_count')
    oracle[0, slope] = 0.5
    cache[0, slope] = np.float32(0.5 + 1e-6)
    assert compare_train_rows(oracle, cache, rows, times)['mismatched_cells'] == 0
    cache[0, slope] = np.float32(0.5 + 1e-4)
    cache[0, count] = np.float32(1e-7)
    report = compare_train_rows(oracle, cache, rows, times)
    assert report['mismatched_cells'] == 2
    assert report['lifecycle_mismatched_cells'] == 1
    with pytest.raises(RuntimeError, match='COMPARE_SHAPE'):
        compare_train_rows(oracle[:, :-1], cache[:, :-1], rows, times)


def test_claim_only_root_binds_plan_policy_worker_and_head(tmp_path):
    output = tmp_path / 'one_use'
    output.mkdir()
    plan = {'path': '/plan.json', 'sha256': 'a' * 64}
    worker = {'path': '/worker.py', 'sha256': 'b' * 64}
    claim = {
        'schema_version': 'gx1_d1_trendline_lifecycle_physical_claim_v1',
        'stage': STAGE, 'plan': plan,
        'policy': {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': 'c' * 64},
        'source_head': 'd' * 40, 'worker': worker,
        'source_only_oracle': True, 'test_data_used': False,
    }
    (output / 'CLAIM.json').write_text(json.dumps(claim))
    observed, ref, _ = read_claim(output, plan, worker, 'd' * 40)
    assert observed == claim and ref['sha256']
    with pytest.raises(RuntimeError, match='CLAIM_BINDING'):
        read_claim(output, plan, worker, 'e' * 40)
    (output / 'RESULT.json').write_text('{}')
    with pytest.raises(RuntimeError, match='CLAIM_ONLY_ROOT'):
        read_claim(output, plan, worker, 'd' * 40)
