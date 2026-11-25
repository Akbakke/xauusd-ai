"""Synthetic V3 source authentication, all-TF observation algebra and gradients."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from gx1.contracts import native_future_state_provider_hlc_v1 as hlc
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from tests.test_native_future_state_provider_hlc_v1 import _synthetic_case


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _case(monkeypatch):
    parent, _, upper = _synthetic_case(monkeypatch)
    whole = parent['state_window_complete'].copy()
    whole[:, 1:] = False
    parent['state_window_complete'] = whole
    parent['nominal_complete'] = whole.copy()
    parent['partial_range_active'] = parent['activity'] & ~whole
    ranges, scale = parent['target_range'], parent['state_activity_scale']
    flat = whole & ((ranges == 0) | (np.isfinite(scale) & (ranges < scale)))
    parent['activity_known'] = parent['activity'] | flat
    parent['side_known'] &= whole
    parent['side_up'] &= parent['side_known']
    bindings = copy.deepcopy(hlc.ALL_TF_SOURCE_BINDINGS)
    proof = _bytes({'status': 'PASS_RAW_MBA_CANONICAL_REPRODUCTION_WITH_DECLARED_FILTER',
        'measured': {'active_input_rows_strictly_before_availability_cutoff': len(parent['issue_ns'])}})
    bindings['m5_raw_integrity_acceptance'] = {'path': '/synthetic/m5_proof.json', 'sha256': _sha(proof)}
    responses, raw = {'M5': []}, {}
    for j, tf in enumerate(hlc.TIMEFRAMES[1:], start=1):
        candles = []
        for start, prices in zip(parent['target_start_ns'][:, j], parent['target_ohlc'][:, j]):
            mba = {name: dict(zip(('o', 'h', 'l', 'c'), (format(float(x + offset), '.17g') for x in prices)))
                for name, offset in (('mid', 0.), ('bid', -.1), ('ask', .1))}
            candles.append({'time': np.datetime_as_string(np.datetime64(int(start), 'ns'), unit='s') + 'Z',
                'complete': True, 'volume': 1, **mba})
        payload = _bytes({'instrument': 'XAU_USD', 'granularity': 'D' if tf == 'D1' else tf, 'candles': candles})
        path = '/synthetic/' + tf + '.json'
        raw[path] = payload
        responses[tf] = [{'response': {'path': path, 'sha256': _sha(payload)}, 'request': {
            'from': candles[0]['time'],
            'to': np.datetime_as_string(np.datetime64(int(parent['validity_ns'][-1, j]), 'ns'), unit='s') + 'Z',
            'granularity': 'D' if tf == 'D1' else tf, 'price': 'MBA', 'dailyAlignment': 22,
            'alignmentTimezone': 'UTC', 'includeFirst': True, 'smooth': False}}]
    manifest = _bytes({'schema_version': 'gx1_all_tf_bar_compare_v1',
        'm5_raw_integrity_acceptance': bindings['m5_raw_integrity_acceptance'],
        'seal_ns': int(np.datetime64('2026-07-01', 'ns').astype(np.int64)), 'provider_responses': responses})
    restored = _bytes({'status': 'COMPLETE_RESTORED_PROVIDER_OBSERVATION_PARITY',
        'all_higher_TF_13_fields_exact': True, 'current_native_inputs_changed': False,
        'targets_prior_baseline_changed': False, 'TEST_used': False,
        'timeframes': {tf: {'after': {'all_13_fields_exact': True, 'missing_native': 0,
            'extra_native': 0, 'mismatching_cells': 0, 'provider_bars': len(parent['issue_ns'])}}
            for tf in hlc.TIMEFRAMES[1:]}})
    evidence = {'provider_manifest': manifest, 'restoration_result': restored,
        'm5_raw_integrity_acceptance': proof}
    for key in hlc.ALL_TF_EVIDENCE_KEYS:
        bindings[key] = {'path': '/synthetic/' + key + '.json', 'sha256': _sha(evidence[key])}
    # The manifest reference to the M5 proof must name that exact same binding.
    bindings['m5_raw_integrity_acceptance']['path'] = '/synthetic/m5_proof.json'
    monkeypatch.setattr(hlc, 'ALL_TF_SOURCE_BINDINGS', bindings)
    cert = hlc.make_all_tf_provider_hlc_certificate(evidence, raw,
        certificate_path='/synthetic/all_tf_certificate.json')
    return parent, cert, upper


def _project(monkeypatch):
    parent, certificate, upper = _case(monkeypatch)
    result = hlc.apply_all_tf_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
        fold_upper_ns=upper, certificate=certificate)
    return parent, result, certificate, upper


def test_all_five_observations_preserve_exact_clock_prices_and_causal_scale(monkeypatch):
    parent, result, certificate, upper = _project(monkeypatch)
    assert not result['provider_hlc_observed'][:, 0].any()
    assert result['provider_hlc_observed'][:, 1:].all()
    np.testing.assert_array_equal(result['state_hlc_observed'], result['mature'])
    changed = {'activity_known', 'side_known', 'side_up', 'future_state_contract'}
    for key, value in parent.items():
        if key not in changed:
            if isinstance(value, np.ndarray):
                assert value.dtype == result[key].dtype and value.tobytes() == result[key].tobytes()
            else:
                assert value == result[key]
    assert result['activity_known'].sum(axis=0).tolist() == [7] * 5
    assert result['side_known'].sum(axis=0).tolist() == [3] * 5
    assert (~result['activity'][[0, 6]] & result['activity_known'][[0, 6]]).all()
    assert not result['side_known'][[3, 4, 5]].any()
    assert not result['activity_known'][5].any()  # positive range + missing ATR stays unknown
    restored = hlc.authenticate_all_tf_provider_hlc_certificate_ref(certificate['ref'],
        metadata_bytes=hlc.certificate_metadata_bytes(certificate['metadata']),
        evidence_bytes=certificate['evidence_bytes'], raw_responses=certificate['raw_responses'])
    assert restored == certificate
    assert set(certificate['metadata']['array_headers_by_tf']) == set(hlc.TIMEFRAMES)
    assert hlc.require_future_state_observation_contract(result['future_state_contract']) == result['future_state_contract']
    with pytest.raises(ValueError):
        hlc.require_provider_hlc_state_contract(result['future_state_contract'])
    with pytest.raises(ValueError):
        hlc.require_provider_hlc_certificate_ref(certificate['ref'])


@pytest.mark.parametrize('bad', ['missing_raw', 'extra_raw', 'renamed_raw', 'changed_raw',
    'missing_evidence', 'extra_evidence', 'changed_evidence', 'certificate_path', 'certificate_bytes'])
def test_all_tf_rejects_source_or_certificate_identity_changes(monkeypatch, bad):
    parent, cert, upper = _case(monkeypatch)
    cert = copy.deepcopy(cert)
    if bad == 'missing_raw': cert['raw_responses'].pop(next(iter(cert['raw_responses'])))
    elif bad == 'extra_raw': cert['raw_responses']['/synthetic/extra.json'] = b'{}'
    elif bad == 'renamed_raw': cert['raw_responses']['/synthetic/wrong.json'] = cert['raw_responses'].pop(next(iter(cert['raw_responses'])))
    elif bad == 'changed_raw': cert['raw_responses'][next(iter(cert['raw_responses']))] += b' '
    elif bad == 'missing_evidence': cert['evidence_bytes'].pop('provider_manifest')
    elif bad == 'extra_evidence': cert['evidence_bytes']['extra'] = b'{}'
    elif bad == 'changed_evidence': cert['evidence_bytes']['restoration_result'] += b' '
    elif bad == 'certificate_path': cert['ref']['path'] = '/synthetic/../other.json'
    else: cert['metadata']['M5_fresh_provider_flag'] = True
    with pytest.raises(ValueError):
        hlc.apply_all_tf_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
            fold_upper_ns=upper, certificate=cert)


@pytest.mark.parametrize('bad', ['issue', 'past_start', 'phase', 'duration', 'maturity',
    'float32', 'pointwise', 'scale', 'future_ATR', 'M5_incomplete'])
def test_all_tf_rejects_bad_parent_time_price_or_observation(monkeypatch, bad):
    parent, cert, upper = _case(monkeypatch)
    parent = copy.deepcopy(parent)
    if bad == 'issue': parent['issue_ns'][0] += 300_000_000_000
    elif bad == 'past_start': parent['target_start_ns'][0, 1] -= 900_000_000_000
    elif bad == 'phase': parent['target_start_ns'][0, 2] += 300_000_000_000
    elif bad == 'duration': parent['validity_ns'][0, 3] += 300_000_000_000
    elif bad == 'maturity': parent['validity_ns'][0, 4] = upper
    elif bad == 'float32': parent['target_ohlc'] = parent['target_ohlc'].astype(np.float32)
    elif bad == 'pointwise':
        parent['target_ohlc'][0, 1, 3] = np.nextafter(parent['target_ohlc'][0, 1, 3], np.inf)
    elif bad == 'scale': parent['state_activity_scale'][0, 1] += .01
    elif bad == 'future_ATR': parent['atr_asof_available_ns'][0, 1] = parent['issue_ns'][0] + 300_000_000_000
    else: parent['state_window_complete'][0, 0] = False
    with pytest.raises(ValueError):
        hlc.apply_all_tf_provider_hlc_observation(parent, issue_ns=parent['issue_ns'],
            fold_upper_ns=upper, certificate=cert)


def test_all_tf_free_head_gradients_prior_and_proper_score_use_same_masks(monkeypatch):
    _, truth, cert, upper = _project(monkeypatch)
    activity = torch.nn.Linear(8, 5, bias=False)
    side = torch.nn.Linear(8, 5, bias=False)
    torch.nn.init.zeros_(activity.weight)
    torch.nn.init.zeros_(side.weight)
    features = torch.eye(8)
    output = {'activity_logits': activity(features), 'side_logits': side(features)}
    kwargs = dict(model_output=output, truth=truth, fold_upper_ns=upper,
        target_contract=truth['future_state_contract'])
    loss = objective.native_future_state_loss(**kwargs)
    score = objective.native_future_state_score_sums(**kwargs)
    assert loss['counts']['activity'].tolist() == [7] * 5
    assert loss['counts']['side'].tolist() == [3] * 5
    assert float(loss['loss']) == pytest.approx(50 * np.log(2) / 35, abs=1e-6)
    assert sum(v['partial_log_loss_sum'] for v in score['per_tf'].values()) == pytest.approx(float(loss['loss']) * 35, abs=1e-5)
    for report in (loss, score):
        assert report['target_observation_certificate_ref'] == cert['ref']
        assert report['truth_sha256'] == objective.native_future_state_truth_sha256(truth,
            target_contract=truth['future_state_contract'])
    loss['loss'].backward()
    assert torch.count_nonzero(side.weight.grad[:, [0, 3, 4, 5, 6]]) == 0
    assert torch.count_nonzero(side.weight.grad[:, [1, 2, 7]]) == 15
    assert torch.count_nonzero(activity.weight.grad[:, 5]) == 0
    assert torch.count_nonzero(activity.weight.grad[:, [0, 1, 2, 3, 4, 6, 7]]) == 35
    prior = objective.fit_native_future_state_train_prior(truth=truth,
        fold_upper_ns=upper, target_contract=truth['future_state_contract'])
    assert all(prior['per_task']['activity'][tf]['class_counts'] == [2, 5] for tf in hlc.TIMEFRAMES)
    assert all(prior['per_task']['side'][tf]['class_counts'] == [1, 2] for tf in hlc.TIMEFRAMES)
    assert prior['fit_population']['truth_sha256'] == loss['truth_sha256']


@pytest.mark.parametrize('bad', ['M5_provider', 'missing_higher_provider', 'immature_provider',
    'observed_union', 'missing_certificate', 'wrong_version_ref'])
def test_v3_objective_rejects_forged_observation_masks(monkeypatch, bad):
    _, truth, cert, upper = _project(monkeypatch)
    truth = copy.deepcopy(truth)
    if bad == 'M5_provider': truth['provider_hlc_observed'][0, 0] = True
    elif bad == 'missing_higher_provider': truth['provider_hlc_observed'][0, 1] = False
    elif bad == 'immature_provider': truth['mature'][0, 1] = False
    elif bad == 'observed_union': truth['state_hlc_observed'][0, 1] = False
    elif bad == 'missing_certificate': del truth['target_observation_certificate_ref']
    else: truth['target_observation_certificate_ref']['target_contract_sha256'] = hlc.provider_hlc_state_contract()['contract_sha256']
    with pytest.raises(ValueError):
        objective.native_future_state_loss(model_output={name: torch.zeros(8, 5)
            for name in ('activity_logits', 'side_logits')}, truth=truth,
            fold_upper_ns=upper, target_contract=truth['future_state_contract'])


def test_actual_native_v3_readout_has_zero_unknown_side_parameter_gradient(monkeypatch):
    from gx1.contracts.market_state_ta_integrated_model_v1 import NativeFutureStateReadout
    _, truth, _, upper = _project(monkeypatch)
    target = truth['future_state_contract']
    prior = objective.fit_native_future_state_train_prior(truth=truth,
        fold_upper_ns=upper, target_contract=target)
    readout = NativeFutureStateReadout(d_model=8, target_contract=target, train_prior=prior)
    hidden = torch.eye(8, dtype=torch.float32)[:, None, :].repeat(1, 5, 1)
    output = readout(hidden)
    initial = objective.native_future_state_train_prior_logits(prior, rows=8,
        target_contract=target)
    assert readout.readout_contract['target_contract_sha256'] == target['contract_sha256']
    assert readout.readout_contract['train_prior_sha256'] == prior['contract_sha256']
    assert all(torch.equal(output[key], initial[key]) for key in initial)
    loss = objective.native_future_state_loss(model_output=output, truth=truth,
        fold_upper_ns=upper, target_contract=target)
    loss['loss'].backward()
    for tf in hlc.TIMEFRAMES:
        side_gradient = readout.side_heads[tf].weight.grad
        activity_gradient = readout.activity_heads[tf].weight.grad
        assert side_gradient.shape == activity_gradient.shape == (1, 8)
        assert torch.count_nonzero(side_gradient[:, [0, 3, 4, 5, 6]]) == 0
        assert torch.count_nonzero(side_gradient[:, [1, 2, 7]]) == 3
        assert torch.count_nonzero(activity_gradient[:, 5]) == 0
        assert torch.count_nonzero(activity_gradient[:, [0, 1, 2, 3, 4, 6, 7]]) == 7
        # Unknown-side active rows still train activity on the actual head.
        assert torch.count_nonzero(activity_gradient[:, [3, 4]]) == 2
