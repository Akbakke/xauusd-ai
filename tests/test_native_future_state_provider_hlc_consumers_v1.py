"""Small synthetic consumer checks for the explicit D1 HLC observation route."""
import copy

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract
from tests.test_native_future_state_provider_hlc_v1 import _projected_case


def _inputs(monkeypatch):
    parent, truth, certificate, upper = _projected_case(monkeypatch)
    output = {name: torch.zeros(len(truth['issue_ns']), 5, dtype=torch.float32,
                               requires_grad=True)
              for name in ('activity_logits', 'side_logits')}
    return parent, certificate, {'model_output': output, 'truth': truth,
        'fold_upper_ns': upper, 'target_contract': truth['future_state_contract']}


def test_hlc_consumer_counts_loss_score_and_unknown_side_gradients(monkeypatch):
    _, certificate, kwargs = _inputs(monkeypatch)
    result = objective.native_future_state_loss(**kwargs)
    assert result['count'] == 35
    assert result['counts']['activity'].tolist() == [7] * 5
    assert result['counts']['side'].tolist() == [3] * 5
    assert float(result['loss']) == pytest.approx(50 * np.log(2) / 35, abs=1e-6)
    score = objective.native_future_state_score_sums(**kwargs)
    d1 = score['per_tf']['D1']
    assert d1['activity']['count'] == 7
    assert d1['side']['count'] == 3
    assert d1['three_class']['count'] == 5
    assert sum(v['partial_log_loss_sum'] for v in score['per_tf'].values()) == pytest.approx(
        float(result['loss']) * result['count'], abs=1e-5)
    for report in (result, score):
        assert report['target_observation_certificate_ref'] == certificate['ref']
        assert report['truth_sha256'] == objective.native_future_state_truth_sha256(
            kwargs['truth'], target_contract=kwargs['target_contract'])
    result['loss'].backward()
    side_grad = kwargs['model_output']['side_logits'].grad
    assert torch.count_nonzero(side_grad[[0, 3, 4, 5, 6], 4]) == 0
    assert torch.count_nonzero(side_grad[[1, 2, 7], 4]) == 3
    activity_grad = kwargs['model_output']['activity_logits'].grad
    assert activity_grad[5, 4] == 0
    assert torch.count_nonzero(activity_grad[[0, 1, 2, 3, 4, 6, 7], 4]) == 7


def test_hlc_prior_support_and_hash_bind_contract_arrays_and_certificate(monkeypatch):
    parent, certificate, kwargs = _inputs(monkeypatch)
    fit = {k: kwargs[k] for k in ('truth', 'fold_upper_ns', 'target_contract')}
    prior = objective.fit_native_future_state_train_prior(**fit)
    assert prior['per_task']['activity']['D1']['class_counts'] == [2, 5]
    assert prior['per_task']['side']['D1']['class_counts'] == [1, 2]
    assert prior['fit_population']['truth_sha256'] == objective.native_future_state_truth_sha256(
        kwargs['truth'], target_contract=kwargs['target_contract'])
    changed = copy.deepcopy(kwargs['truth'])
    changed['target_observation_certificate_ref']['sha256'] = 'f' * 64
    assert objective.native_future_state_truth_sha256(changed,
        target_contract=kwargs['target_contract']) != prior['fit_population']['truth_sha256']
    changed = copy.deepcopy(kwargs['truth'])
    changed['provider_hlc_observed'][0, 4] = False
    assert objective.native_future_state_truth_sha256(changed,
        target_contract=kwargs['target_contract']) != prior['fit_population']['truth_sha256']
    old = objective.fit_native_future_state_train_prior(truth=parent,
        fold_upper_ns=kwargs['fold_upper_ns'], target_contract=fully_future_state_contract())
    assert prior['contract_sha256'] != old['contract_sha256']
    from gx1.audit import native_future_state_training_v1 as training
    take = np.array([0, 1, 3, 5], dtype=np.int64)
    sliced = training.slice_provider_hlc_truth(kwargs['truth'], take,
        target_contract=kwargs['target_contract'], fold_upper_ns=kwargs['fold_upper_ns'])
    assert set(sliced) == set(training.provider_hlc_truth_fields(kwargs['target_contract'])) | {
        'target_observation_certificate_ref'}
    assert sliced['target_observation_certificate_ref'] == certificate['ref']
    assert sliced['target_observation_certificate_ref'] is not kwargs['truth']['target_observation_certificate_ref']
    for key in training.provider_hlc_truth_fields(kwargs['target_contract']):
        np.testing.assert_array_equal(sliced[key], kwargs['truth'][key][take])
        assert not sliced[key].flags.writeable
    for value, spec in ((old, kwargs['target_contract']), (prior, fully_future_state_contract())):
        with pytest.raises(ValueError):
            objective.require_native_future_state_train_prior(value, target_contract=spec)


def test_v1_hash_and_unknown_side_behavior_are_preserved(monkeypatch):
    parent, _, kwargs = _inputs(monkeypatch)
    spec = fully_future_state_contract()
    assert objective.native_future_state_truth_sha256(parent,
        target_contract=spec) == objective._native_prior_array_sha({key: parent[key] for key in (
            'issue_ns', 'target_start_ns', 'validity_ns', 'target_present', 'mature',
            'nominal_complete', 'state_window_complete', 'partial_range_active',
            'activity_known', 'activity', 'side_known', 'side_up')})
    kwargs.update(truth=parent, target_contract=spec)
    result = objective.native_future_state_loss(**kwargs)
    assert result['counts']['activity'].tolist() == [7, 7, 7, 7, 5]
    assert result['counts']['side'].tolist() == [3, 3, 3, 3, 0]
    assert 'truth_sha256' not in result
    assert 'target_observation_certificate_ref' not in result
    result['loss'].backward()
    assert torch.count_nonzero(kwargs['model_output']['side_logits'].grad[:, 4]) == 0


@pytest.mark.parametrize('bad', ['non_D1_provider', 'observed_union',
                                'side_without_hlc', 'missing_certificate_ref'])
def test_v2_consumer_rejects_forged_observation_masks(monkeypatch, bad):
    _, _, kwargs = _inputs(monkeypatch)
    truth = kwargs['truth']
    if bad == 'non_D1_provider':
        truth['provider_hlc_observed'][0, 0] = True
    elif bad == 'observed_union':
        truth['state_hlc_observed'][0, 4] = False
    elif bad == 'side_without_hlc':
        truth['provider_hlc_observed'][1, 4] = False
        truth['state_hlc_observed'][1, 4] = False
    else:
        del truth['target_observation_certificate_ref']
    with pytest.raises(ValueError):
        objective.native_future_state_loss(**kwargs)


def test_tiny_readout_state_rejects_cross_version_identity_and_metadata_input(monkeypatch):
    from gx1.contracts.market_state_ta_integrated_model_v1 import NativeFutureStateReadout
    parent, certificate, kwargs = _inputs(monkeypatch)
    old_spec = fully_future_state_contract()
    old_prior = objective.fit_native_future_state_train_prior(truth=parent,
        fold_upper_ns=kwargs['fold_upper_ns'], target_contract=old_spec)
    new_prior = objective.fit_native_future_state_train_prior(**{
        k: kwargs[k] for k in ('truth', 'fold_upper_ns', 'target_contract')})
    old = NativeFutureStateReadout(d_model=2, target_contract=old_spec, train_prior=old_prior)
    new = NativeFutureStateReadout(d_model=2, target_contract=kwargs['target_contract'],
                                 train_prior=new_prior)
    assert old.readout_contract['contract_sha256'] != new.readout_contract['contract_sha256']
    for destination, state in ((new, old.state_dict()), (old, new.state_dict())):
        before = {k: v.clone() for k, v in destination.state_dict().items()}
        with pytest.raises(ValueError, match='FUTURE_READOUT_STATE_IDENTITY'):
            destination.load_state_dict(state)
        for name, value in destination.state_dict().items():
            assert torch.equal(value, before[name])
    hidden = torch.zeros(1, 5, 2, dtype=torch.float32)
    assert set(new(hidden)) == {'activity_logits', 'side_logits'}
    with pytest.raises(TypeError):
        new(hidden, target_observation_certificate_ref=certificate['ref'])
