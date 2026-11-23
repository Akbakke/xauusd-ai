"""Fail-closed certificate projection on eight synthetic D1 price geometries."""
import copy
import hashlib
import json

import numpy as np
import pytest

from gx1.contracts import market_state_native_m5_targets_v1 as legacy
from gx1.contracts import native_future_state_provider_hlc_v1 as hlc

DAY = 86_400_000_000_000
TICK = 300_000_000_000
UPPER = int(np.datetime64('2026-01-01', 'ns').astype(np.int64))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _synthetic_case(monkeypatch):
    spec = legacy.fully_future_state_contract()
    first = int(np.datetime64('2025-01-07T22:00', 'ns').astype(np.int64))
    issues = first + np.arange(8, dtype=np.int64) * DAY
    duration = np.asarray([spec['tf_duration_ns'][tf] for tf in spec['timeframes']], np.int64)
    origin = np.asarray([spec['tf_origin_ns'][tf] for tf in spec['timeframes']], np.int64)
    starts = ((issues[:, None] - origin + duration - 1) // duration) * duration + origin
    ends = starts + duration
    # flat; up; down; both excursions; neither excursion; unknown ATR;
    # exact-zero range with unknown ATR; exact endpoint with zero ATR.
    high = np.array([100.5, 102., 102., 102., 101., 102., 100., 101.])
    low = np.array([100., 98., 98., 98., 100., 98., 100., 100.])
    close = np.array([100.25, 102., 98., 100., 100.5, 102., 100., 101.])
    opening = np.minimum(high, np.maximum(low, 100.))
    ohlc = np.repeat(np.column_stack((opening, high, low, close))[:, None, :], 5, axis=1)
    atr = np.repeat(np.array([2., 2., 2., 2., 2., np.nan, np.nan, 0.])[:, None], 5, axis=1)
    scale = 0.5 * atr
    observed_range = ohlc[:, :, 1] - ohlc[:, :, 2]
    whole = np.ones((8, 5), np.bool_)
    whole[:, 4] = False
    active = np.isfinite(scale) & (observed_range > 0) & (observed_range >= scale)
    flat = whole & ((observed_range == 0) | (np.isfinite(scale) & (observed_range < scale)))
    up_material = np.where(scale > 0, ohlc[:, :, 3] - ohlc[:, :, 2] >= scale,
                           ohlc[:, :, 3] - ohlc[:, :, 2] > 0)
    down_material = np.where(scale > 0, ohlc[:, :, 1] - ohlc[:, :, 3] >= scale,
                             ohlc[:, :, 1] - ohlc[:, :, 3] > 0)
    side = whole & active & (up_material ^ down_material)
    parent = {'future_state_contract': spec, 'issue_ns': issues,
        'target_start_ns': starts, 'validity_ns': ends,
        'target_present': np.ones((8, 5), np.bool_), 'mature': np.ones((8, 5), np.bool_),
        'nominal_complete': whole.copy(), 'state_window_complete': whole.copy(),
        'partial_range_active': active & ~whole, 'activity': active,
        'activity_known': active | flat, 'side_known': side, 'side_up': side & up_material,
        'target_ohlc': ohlc, 'target_range': observed_range, 'atr14_asof': atr,
        'state_activity_scale': scale, 'atr_asof_available_ns': np.repeat(issues[:, None], 5, axis=1),
        'state_expected_m5_count': np.tile(duration // TICK, (8, 1)),
        'state_missing_open_m5_count': (~whole).astype(np.int64),
        'state_endpoint_available_ns': ends - TICK,
        'parent_receipt': {'path': '/synthetic/parent.json', 'sha256': 'a' * 64}}
    candles = []
    for i, start in enumerate(starts[:, 4]):
        components = {}
        for name, offset in (('mid', 0.), ('bid', -.1), ('ask', .1)):
            components[name] = dict(zip(('o', 'h', 'l', 'c'),
                (format(float(v + offset), '.17g') for v in ohlc[i, 4])))
        timestamp = np.datetime_as_string(np.datetime64(int(start), 'ns'), unit='s')
        candles.append({'time': timestamp + '.000000000Z', 'complete': True,
                        'volume': 1, **components})
    raw = _json_bytes({'instrument': 'XAU_USD', 'granularity': 'D', 'candles': candles})
    bindings = {key: {'path': '/synthetic/' + key + '.json', 'sha256': 'a' * 64}
                for key in hlc.SOURCE_BINDINGS}
    bindings['reference_raw']['sha256'] = _sha(raw)
    monkeypatch.setattr(hlc, 'SOURCE_BINDINGS', bindings)
    monkeypatch.setattr(hlc, 'PROVIDER_ROWS', len(candles))
    monkeypatch.setattr(hlc, 'TRAIN_ISSUE_SHA256', _sha(issues.tobytes()))
    monkeypatch.setattr(hlc, 'TRAIN_UPPER_NS', UPPER)
    certificate = hlc.make_provider_hlc_certificate(raw,
        certificate_path='/synthetic/provider_hlc_certificate.json')
    return parent, certificate, UPPER


def _projected_case(monkeypatch):
    parent, certificate, upper = _synthetic_case(monkeypatch)
    projected = hlc.apply_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
        fold_upper_ns=upper, certificate=certificate)
    return parent, projected, certificate, upper


def _assert_same(left, right):
    assert set(left) == set(right)
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == right[key].dtype and value.shape == right[key].shape
            assert value.tobytes() == right[key].tobytes(), key
        else:
            assert value == right[key], key


def test_absent_certificate_returns_exact_old_payload_arrays_and_metadata(monkeypatch):
    parent, _, upper = _synthetic_case(monkeypatch)
    before = copy.deepcopy(parent)
    for kwargs in ({}, {'certificate': None}):
        result = hlc.apply_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
            fold_upper_ns=upper, **kwargs)
        assert result is parent
        _assert_same(result, before)
    assert 'provider_hlc_observed' not in parent
    assert parent['future_state_contract'] == legacy.fully_future_state_contract()


def test_authenticated_hlc_certifies_only_defined_d1_facts_preserving_clock_and_other_tf(monkeypatch):
    parent, projected, certificate, _ = _projected_case(monkeypatch)
    assert projected['activity_known'][:, 4].tolist() == [True, True, True, True, True, False, True, True]
    assert projected['activity'][:, 4].tolist() == [False, True, True, True, True, False, False, True]
    assert projected['side_known'][:, 4].tolist() == [False, True, True, False, False, False, False, True]
    assert projected['side_up'][:, 4].tolist() == [False, True, False, False, False, False, False, True]
    assert projected['provider_hlc_observed'][:, 4].all()
    assert not projected['provider_hlc_observed'][:, :4].any()
    assert not projected['state_window_complete'][:, 4].any()
    np.testing.assert_array_equal(projected['state_hlc_observed'],
        projected['state_window_complete'] | projected['provider_hlc_observed'])
    raw_facts = ('nominal_complete', 'state_window_complete', 'state_expected_m5_count',
        'state_missing_open_m5_count', 'state_endpoint_available_ns', 'target_ohlc',
        'target_range', 'atr14_asof', 'state_activity_scale', 'atr_asof_available_ns',
        'target_start_ns', 'validity_ns', 'target_present', 'mature', 'partial_range_active')
    for key in raw_facts:
        assert parent[key].tobytes() == projected[key].tobytes(), key
    for key, value in parent.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            assert value[:, :4].tobytes() == projected[key][:, :4].tobytes(), key
    assert projected['target_observation_certificate_ref'] == certificate['ref']
    assert parent['future_state_contract'] == legacy.fully_future_state_contract()
    restored = hlc.authenticate_provider_hlc_certificate_ref(certificate['ref'],
        metadata_bytes=hlc.certificate_metadata_bytes(certificate['metadata']),
        raw_response_bytes=certificate['raw_response'])
    assert restored == certificate


@pytest.mark.parametrize('bad', ['wrong_hash', 'wrong_TF', 'off_phase', 'duplicate_start',
                                'wrong_end', 'HLC_mismatch', 'false_maturity', 'fold_cutoff'])
def test_provider_hlc_rejects_bad_evidence_and_ineligible_target_windows(monkeypatch, bad):
    parent, certificate, upper = _synthetic_case(monkeypatch)
    if bad in ('wrong_TF', 'off_phase', 'duplicate_start'):
        raw = json.loads(certificate['raw_response'])
        if bad == 'wrong_TF':
            raw['granularity'] = 'H4'
        elif bad == 'off_phase':
            raw['candles'][0]['time'] = '2025-01-07T21:55:00.000000000Z'
        else:
            raw['candles'][1]['time'] = raw['candles'][0]['time']
        payload = _json_bytes(raw)
        # Authorize these synthetic bytes so the actual semantic check is reached.
        bindings = copy.deepcopy(hlc.SOURCE_BINDINGS)
        bindings['reference_raw']['sha256'] = _sha(payload)
        monkeypatch.setattr(hlc, 'SOURCE_BINDINGS', bindings)
        with pytest.raises(ValueError, match='NATIVE_PROVIDER_HLC_'):
            hlc.make_provider_hlc_certificate(payload,
                certificate_path='/synthetic/provider_hlc_certificate.json')
        return
    if bad == 'wrong_hash':
        certificate['ref']['sha256'] = 'f' * 64
    elif bad == 'wrong_end':
        parent['validity_ns'][0, 4] += TICK
    elif bad == 'HLC_mismatch':
        parent['target_ohlc'][0, 4, 3] += .01
    elif bad == 'false_maturity':
        parent['mature'][0, 4] = False
    else:
        upper += TICK
    with pytest.raises(ValueError, match='NATIVE_PROVIDER_HLC_'):
        hlc.apply_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
            fold_upper_ns=upper, certificate=certificate)
