"""Synthetic output comparisons only; no model, checkpoint or market inputs."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gx1.audit.observe_market_state_ta_current_numeric_routes_v1 import compare_pieces


def pieces():
    raw = np.linspace(-2, 2, 160, dtype=np.float32).reshape(16, 5, 2)
    query = np.tile(np.asarray([1, 3, 12, 48, 288], dtype=np.int64), (16, 1))
    result = dict(parents=np.arange(16, dtype=np.int64),
        issues=np.arange(16, dtype=np.int64) * 300_000_000_000,
        state_query=query.copy(), future_query=query.copy(),
        state_raw=raw.copy(), future_raw=-raw.copy())
    for task in ('state', 'future'):
        result[task + '_allowed'] = np.full((16, 5), 4, dtype=np.uint8)
        result[task + '_eligible'] = np.ones((16, 5), dtype=np.bool_)
        result[task + '_mature'] = np.ones((16, 5), dtype=np.bool_)
    return result


def clone(value):
    return {k: v.copy() for k, v in value.items()}


def test_identical_outputs_are_bitexact_finite_json_and_unchanged():
    reference = pieces(); actual = clone(reference)
    before = {k: v.tobytes() for k, v in actual.items()}
    report = compare_pieces(actual, reference, 1.25, 1.25, np=np)
    assert report['within_existing_tolerance'] is True
    assert report['axes_targets_bitexact'] is True and report['loss_max_abs'] == 0.
    assert set(report['raw']) == {'state_raw', 'future_raw'}
    for raw in report['raw'].values():
        assert raw['bitexact'] is True
        assert raw['max_abs'] == raw['max_scaled'] == raw['max_probability_abs'] == 0.
    assert all(actual[k].tobytes() == reference[k].tobytes() == before[k] for k in actual)
    json.dumps(report, allow_nan=False)


def test_raw_and_loss_threshold_exceedance_is_retained_instead_of_raised():
    reference = pieces(); reference['state_raw'].fill(0)
    actual = clone(reference); actual['state_raw'][0, 0, 1] = np.float32(3e-6)
    report = compare_pieces(actual, reference, 1., 1., np=np)
    assert report['within_existing_tolerance'] is False
    assert report['axes_targets_bitexact'] is True and report['loss_max_abs'] == 0.
    delta = float(actual['state_raw'][0, 0, 1])
    assert report['raw']['state_raw']['max_abs'] == delta
    assert report['raw']['state_raw']['max_scaled'] == pytest.approx(delta / 2e-6, rel=1e-14)
    assert report['raw']['state_raw']['bitexact'] is False
    loss_report = compare_pieces(reference, clone(reference), 5e-5, 0., np=np)
    assert loss_report['within_existing_tolerance'] is False and loss_report['loss_max_abs'] == 5e-5
    assert all(raw['bitexact'] for raw in loss_report['raw'].values())
    json.dumps(report, allow_nan=False); json.dumps(loss_report, allow_nan=False)


def test_target_or_parent_identity_change_is_rejected():
    reference = pieces()
    for key in ('future_allowed', 'state_eligible', 'parents'):
        actual = clone(reference)
        actual[key].flat[0] = 1 if key == 'future_allowed' else False if key == 'state_eligible' else 99
        with pytest.raises(RuntimeError):
            compare_pieces(actual, reference, 1., 1., np=np)


def test_bad_shape_dtype_and_nonfinite_values_are_rejected():
    reference = pieces()
    for fault in ('shape', 'dtype', 'nan', 'infinite_loss'):
        actual = clone(reference); loss = 1.
        if fault == 'shape': actual['state_raw'] = actual['state_raw'][:, :4]
        elif fault == 'dtype': actual['future_raw'] = actual['future_raw'].astype(np.float64)
        elif fault == 'nan': actual['future_raw'][0, 0, 0] = np.nan
        else: loss = float('inf')
        with pytest.raises(RuntimeError):
            compare_pieces(actual, reference, loss, 1., np=np)


def test_sigmoid_difference_matches_independent_scalar_math_and_small_delta_passes():
    reference = pieces(); actual = clone(reference)
    reference['state_raw'].fill(0); actual['state_raw'].fill(0)
    actual['state_raw'][0, 0, 1] = np.float32(1e-6)
    small = compare_pieces(actual, reference, 1. + 1e-6, 1., np=np)
    assert small['within_existing_tolerance'] is True
    assert small['raw']['state_raw']['bitexact'] is False
    reference['future_raw'].flat[:5] = np.asarray([-3., 0., 3., -1000., 1000.], np.float32)
    actual['future_raw'].flat[:5] = np.asarray([-2., .7, 4., -999., 999.], np.float32)
    def sigmoid(value):
        value = float(value)
        if value >= 0: return 1. / (1. + math.exp(-value))
        e = math.exp(value); return e / (1. + e)
    report = compare_pieces(actual, reference, 1., 1., np=np)
    expected = max(abs(sigmoid(a) - sigmoid(b)) for a, b in zip(actual['future_raw'].flat, reference['future_raw'].flat))
    assert report['raw']['future_raw']['max_probability_abs'] == pytest.approx(expected, rel=1e-13, abs=1e-15)
    assert report['within_existing_tolerance'] is False
    json.dumps(report, allow_nan=False)
