"""Fixed integrated S/F plus masked structure objective arithmetic."""
import numpy as np
import pytest
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_ta_integrated_objective_v1 import (
    STRUCTURE_WEIGHT, contract, integrated_loss,
)
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def _inputs():
    state = torch.randn(3, 5, 2, dtype=torch.float32, requires_grad=True)
    future = torch.randn(3, 5, 2, dtype=torch.float32, requires_grad=True)
    structure = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)
    output = {'state_activity_side_logits': state,
              'state_log_probs': factorized_log_probabilities(state),
              'future_activity_side_logits': future,
              'future_log_probs': factorized_log_probabilities(future),
              'm5_structure_hit_logits': structure}
    allowed = np.array([[1, 4, 1, 4, 1], [4, 1, 4, 1, 4],
                        [2, 5, 7, 1, 4]], dtype=np.uint8)
    truth = {'allowed': allowed, 'eligible': np.ones((3, 5), dtype=np.bool_),
             'mature': np.ones((3, 5), dtype=np.bool_)}
    hits = np.array([[True, False], [False, True], [False, False]], dtype=np.bool_)
    observed = np.array([[True, True], [False, True], [False, False]], dtype=np.bool_)
    return output, truth, hits, observed, torch.tensor([NATIVE_K] * 3, dtype=torch.int64)


def test_integrated_loss_equals_bound_components_and_masks_unknown_gradients():
    output, truth, hits, observed, query = _inputs()
    report = integrated_loss(model_output=output, state_truth=truth,
        future_truth=truth, state_query_k=query,
        structure_hit=hits, structure_observed=observed)
    sf = combined_loss(state_raw=output['state_activity_side_logits'],
        future_raw=output['future_activity_side_logits'],
        state_truth=truth, future_truth=truth, state_query_k=query,
        future_weight=1.0, ta_only=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    structural = masked_structure_hit_sums(output['m5_structure_hit_logits'],
        torch.from_numpy(hits), torch.from_numpy(observed))
    expected_structure = (structural['loss_sums'] /
                          structural['observed_counts'].clamp(min=1)).mean()
    assert torch.equal(report['S_F_loss'], sf['loss'])
    assert torch.equal(report['structure_loss'], expected_structure)
    assert torch.equal(report['loss'], sf['loss'] + STRUCTURE_WEIGHT * expected_structure)
    assert report['structure_observed_counts'].tolist() == [1, 2]
    assert report['contract_sha256'] == contract()['contract_sha256']
    report['loss'].backward()
    gradient = output['m5_structure_hit_logits'].grad
    assert gradient[1, 0] == gradient[2, 0] == gradient[2, 1] == 0
    assert gradient[0, 0] != 0 and gradient[0, 1] != 0 and gradient[1, 1] != 0
    assert torch.count_nonzero(output['state_activity_side_logits'].grad)
    assert torch.count_nonzero(output['future_activity_side_logits'].grad)


def test_integrated_loss_rejects_unobserved_positive_and_broken_factorization():
    output, truth, hits, observed, query = _inputs()
    hits[1, 0] = True
    with pytest.raises(ValueError, match='STRUCTURE'):
        integrated_loss(model_output=output, state_truth=truth, future_truth=truth,
                        state_query_k=query, structure_hit=hits,
                        structure_observed=observed)
    hits[1, 0] = False
    output['state_log_probs'] = output['state_log_probs'] + 1
    with pytest.raises(ValueError, match='FACTOR_PROBABILITIES'):
        integrated_loss(model_output=output, state_truth=truth, future_truth=truth,
                        state_query_k=query, structure_hit=hits,
                        structure_observed=observed)


def test_structure_readout_uses_bool_known_labels():
    from gx1.audit.run_market_state_ta_integrated_fold2_fit_v1 import structure_readout

    logits = np.array([[2.0, -2.0], [-2.0, 2.0], [100.0, -100.0]], dtype=np.float32)
    hit = np.array([[True, False], [False, True], [False, False]], dtype=bool)
    observed = np.array([[True, True], [True, True], [False, False]], dtype=bool)
    result = structure_readout(logits, hit, observed, np.array([.5, .5]))
    assert result['high']['observed'] == result['low']['observed'] == 2
    assert result['high']['auc'] == result['low']['auc'] == 1.0
    assert result['high']['hit'] == result['low']['hit'] == 1
    assert result['high']['brier'] < result['high']['prior_brier']


def _native_bar_truth():
    issues = np.array([1, 2, 3], dtype=np.int64) * 300_000_000_000
    validity = np.broadcast_to(issues[:, None] + 300_000_000_000, (3, 5)).copy()
    mature = validity < 1_200_000_000_000
    target_rows = np.full((3, 5), -1, dtype=np.int64)
    target_rows[:2] = [[2] * 5, [3] * 5]
    target_rows[:2, 1:] = 2
    previous = mature.copy()
    open_known = np.zeros((3, 5), dtype=np.bool_)
    open_known[:2, 1:] = True
    signs = np.array([[-1, 0, 1, -1, 0], [0, 1, -1, 0, 1], [0] * 5], dtype=np.int8)
    return {'issue_ns': issues, 'validity_ns': validity,
            'target_row': target_rows, 'target_present': np.ones((3, 5), dtype=np.bool_),
            'mature': mature, 'observed_previous': previous, 'eligible': mature.copy(),
            'open_known_at_issue': open_known, 'open_sign': signs, 'close_sign': signs.copy()}


def test_native_bar_loss_keeps_equality_masks_known_open_and_matches_proper_scores():
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import (
        native_bar_direction_loss, native_bar_direction_score_sums,
    )
    output = {key: torch.zeros(3, 5, 3, requires_grad=True)
              for key in ('open_logits', 'close_logits')}
    kwargs = {'model_output': output, 'truth': _native_bar_truth(), 'fold_upper_ns': 1_200_000_000_000,
              'target_contract_sha256': target_contract()['contract_sha256']}
    result = native_bar_direction_loss(**kwargs)
    assert result['counts']['open'].tolist() == [2, 0, 0, 0, 0]
    assert result['counts']['close'].tolist() == [2] * 5
    assert float(result['loss']) == pytest.approx(np.log(3))
    result['loss'].backward()
    assert torch.count_nonzero(output['open_logits'].grad[:, 1:]) == 0
    assert torch.count_nonzero(output['close_logits'].grad[-1]) == 0
    assert output['close_logits'].grad[0, 1, 1] < 0  # equality is class 1
    score = native_bar_direction_score_sums(**kwargs)['per_task']
    assert score['open']['M5']['count'] == 2
    assert score['open_known']['H4']['count'] == 2
    assert score['close']['H4']['count'] == 2
    assert score['close']['H4']['unique_target_rows'] == 1
    assert score['close']['H4']['max_logit_ties'] == 2
    assert score['close']['H4']['log_loss_sum'] == pytest.approx(2 * np.log(3))
    assert score['close']['H4']['brier_sum'] == pytest.approx(4 / 3)
    assert np.asarray(score['close']['H4']['confusion_truth_by_prediction']).sum() == 2


def test_native_bar_objective_rejects_old_target_identity_and_false_maturity():
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    output = {key: torch.zeros(3, 5, 3) for key in ('open_logits', 'close_logits')}
    kwargs = {'model_output': output, 'truth': _native_bar_truth(), 'fold_upper_ns': 1_200_000_000_000,
              'target_contract_sha256': '0' * 64}
    with pytest.raises(ValueError, match='TARGET_CONTRACT'):
        native_bar_direction_loss(**kwargs)
    kwargs['target_contract_sha256'] = target_contract()['contract_sha256']
    kwargs['truth']['mature'][0, 0] = False
    with pytest.raises(ValueError, match='OBSERVABILITY'):
        native_bar_direction_loss(**kwargs)


def test_native_train_prior_uses_three_classes_predictive_masks_and_issue_weighting():
    from copy import deepcopy
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    truth=_native_bar_truth();before=deepcopy(truth)
    prior=owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=1_200_000_000_000,
        target_contract_sha256=target_contract()['contract_sha256'])
    assert owner.require_native_bar_train_prior(prior)==prior
    assert prior['per_task']['close']['M5']['class_counts']==[1,1,0]
    assert prior['per_task']['close']['M5']['probabilities']==[.4,.4,.2]
    assert prior['per_task']['close']['M15']['count']==2
    assert prior['per_task']['close']['M15']['unique_target_rows']==1
    assert prior['per_task']['open']['M5']['class_counts']==[1,1,0]
    for tf in ('M15','H1','H4','D1'):
        cell=prior['per_task']['open'][tf]
        assert cell['count']==0 and cell['class_counts']==[0,0,0]
        assert cell['probabilities']==[1/3]*3 and cell['train_observations_available']is False
    for key in truth:np.testing.assert_array_equal(truth[key],before[key])
    logits=owner.native_bar_train_prior_logits(prior,rows=3)
    assert all(value.dtype==np.float32 and value.shape==(3,5,3) and np.isfinite(value).all()for value in logits.values())
    scores=owner.native_bar_direction_score_sums(model_output=logits,truth=truth,
        fold_upper_ns=1_200_000_000_000,target_contract_sha256=target_contract()['contract_sha256'])['per_task']
    assert scores['close']['M5']['log_loss_sum']==pytest.approx(-2*np.log(.4),abs=1e-7)
    assert scores['close']['M5']['brier_sum']==pytest.approx(1.12,abs=1e-7)
    assert scores['close']['M15']['unique_target_rows']==1
    assert scores['open']['H4']['count']==0 and scores['open_known']['H4']['count']==2


def test_native_train_prior_is_frozen_for_control_and_rejects_unmatched_or_future_fit():
    from copy import deepcopy
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    from gx1.contracts.entry_model_native_input_normalization_v1 import NATIVE_M5_TRAIN_FIT_CUTOFF_NS
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    truth=_native_bar_truth();target_sha=target_contract()['contract_sha256']
    prior=owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=1_200_000_000_000,
        target_contract_sha256=target_sha);before=deepcopy(prior)
    control=deepcopy(truth);control['issue_ns']+=1_200_000_000_000;control['validity_ns']+=1_200_000_000_000
    control['target_row'][control['mature']]+=100
    control['open_sign']*=-1;control['close_sign']*=-1
    score=owner.native_bar_direction_score_sums(model_output=owner.native_bar_train_prior_logits(prior,rows=3),
        truth=control,fold_upper_ns=2_400_000_000_000,target_contract_sha256=target_sha)
    assert score['per_task']['close']['M5']['count']==2 and prior==before
    with pytest.raises(ValueError,match='TRAIN_CUTOFF_OR_AXIS'):
        owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=NATIVE_M5_TRAIN_FIT_CUTOFF_NS+300_000_000_000,
            target_contract_sha256=target_sha)
    with pytest.raises(ValueError,match='TARGET_CONTRACT'):
        owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=1_200_000_000_000,target_contract_sha256='0'*64)
    bad=deepcopy(prior);bad['per_task']['close']['M5']['probabilities']=[1/3]*3
    with pytest.raises(ValueError,match='FIXED_LAPLACE_ONE'):owner.require_native_bar_train_prior(bad)
    bad=deepcopy(prior);bad['fit_population']['scope']='CONTROL'
    with pytest.raises(ValueError,match='POPULATION'):owner.require_native_bar_train_prior(bad)
    bad_truth={key:value[::-1].copy()for key,value in truth.items()}
    with pytest.raises(ValueError,match='ISSUE_ORDER'):
        owner.fit_native_bar_train_prior(truth=bad_truth,fold_upper_ns=1_200_000_000_000,target_contract_sha256=target_sha)


def _native_noise_truth_fixture():
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    spec=noise_margin_dataset_contract(.3);truth=_native_bar_truth();shape=truth['target_row'].shape
    truth['open_sign'][0,0]=0;truth['close_sign'][0,2]=0
    truth.update(m5_count=truth['mature'].astype(np.int64),source_gap=np.zeros(shape,np.bool_),
        noise_atr14_asof=np.ones(shape,np.float64),
        noise_atr_asof_row=np.broadcast_to(np.arange(3,dtype=np.int64)[:,None],shape).copy(),
        noise_atr_asof_available_ns=np.broadcast_to(truth['issue_ns'][:,None],shape).copy(),
        noise_margin_contract=spec['noise_margin_contract'])
    truth['noise_atr14_asof'][0,[0,2]]=np.nan
    truth['noise_margin_price']=.3*truth['noise_atr14_asof']
    for task in ('open','close'):
        delta=truth[task+'_sign'].astype(np.float64)*np.array([.2,.5,1.])[:,None]
        delta[~truth['eligible']]=np.nan
        truth['noise_'+task+'_delta']=delta
        sign=np.zeros(shape,np.int8);sign[1]=truth[task+'_sign'][1]
        truth['noise_'+task+'_sign']=sign
        known=truth['eligible'].copy();known[0,0 if task=='close'else 2]=False
        truth['noise_'+task+'_known']=known
        truth['noise_'+task+'_eligible']=known.copy()if task=='close'else known&~truth['open_known_at_issue']
    return truth,spec


def test_noise_objective_prior_and_score_use_per_task_masks_and_neutral_class():
    from copy import deepcopy
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    truth,spec=_native_noise_truth_fixture();before=deepcopy(truth)
    output={key:torch.zeros(3,5,3,requires_grad=True)for key in ('open_logits','close_logits')}
    args={'truth':truth,'fold_upper_ns':1_200_000_000_000,
        'target_contract_sha256':spec['contract_sha256'],'target_contract':spec}
    result=owner.native_bar_direction_loss(model_output=output,**args)
    assert result['counts']['open'].tolist()==[2,0,0,0,0]
    assert result['counts']['close'].tolist()==[1,2,2,2,2]
    assert float(result['loss'])==pytest.approx(np.log(3))
    result['loss'].backward()
    assert torch.count_nonzero(output['close_logits'].grad[0,0])==0
    assert output['open_logits'].grad[0,0,1]<0  # Exact zero is known without ATR.
    assert torch.count_nonzero(output['open_logits'].grad[:,1:])==0
    assert torch.count_nonzero(output['close_logits'].grad[2])==0
    score=owner.native_bar_direction_score_sums(model_output=output,**args)
    assert score['class_names']==['DOWN','NEUTRAL_NET','UP']
    assert score['per_task']['close']['M15']['unique_target_rows']==1
    assert score['per_task']['open_known']['H1']['count']==1
    assert score['per_task']['close']['M5']['brier_sum']==pytest.approx(2/3)
    prior=owner.fit_native_bar_train_prior(**args)
    assert prior['per_task']['close']['M5']['class_counts']==[0,1,0]
    assert prior['per_task']['open']['M5']['class_counts']==[0,2,0]
    assert prior['per_task']['close']['M15']['class_counts']==[0,1,1]
    assert prior['per_task']['open']['D1']['train_observations_available']is False
    assert owner.require_native_bar_train_prior(prior,target_contract=spec)==prior
    logits=owner.native_bar_train_prior_logits(prior,rows=3,target_contract=spec)
    scored=owner.native_bar_direction_score_sums(model_output=logits,**args)
    assert scored['class_names_sha256']==prior['class_names_sha256']
    assert scored['per_task']['close']['M5']['log_loss_sum']==pytest.approx(np.log(2),abs=1e-7)
    for name,value in before.items():
        if isinstance(value,np.ndarray):assert truth[name].tobytes()==value.tobytes()
        else:assert truth[name]==value


@pytest.mark.parametrize('field,index,value,reason',[
    ('noise_close_known',(0,0),True,'MASKS_close'),
    ('noise_close_sign',(1,1),0,'SIGNS_close'),
    ('noise_open_eligible',(0,1),True,'MASKS_open'),
    ('noise_atr_asof_available_ns',(0,0),600_000_000_000,'ATR_ASOF'),
    ('noise_margin_price',(1,1),.2,'MARGIN_VALUES'),
    ('noise_close_delta',(2,0),0.,'IMMATURE_DELTA_close'),
])
def test_noise_objective_rejects_false_labels_masks_future_scale_and_uncensored_tail(field,index,value,reason):
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    truth,spec=_native_noise_truth_fixture();truth[field][index]=value
    output={key:torch.zeros(3,5,3)for key in ('open_logits','close_logits')}
    with pytest.raises(ValueError,match='NOISE_'+reason):
        owner.native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=1_200_000_000_000,
            target_contract_sha256=spec['contract_sha256'],target_contract=spec)


def test_noise_contract_prior_and_raw_defaults_never_alias_equal_or_other_margin():
    from copy import deepcopy
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract,contract
    truth,spec=_native_noise_truth_fixture();upper=1_200_000_000_000
    raw=owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=upper,target_contract_sha256=contract()['contract_sha256'])
    prior=owner.fit_native_bar_train_prior(truth=truth,fold_upper_ns=upper,
        target_contract_sha256=spec['contract_sha256'],target_contract=spec)
    assert owner.native_bar_objective_contract()==owner.native_bar_objective_contract(target_contract=None)
    assert owner.native_bar_train_prior_contract()==owner.native_bar_train_prior_contract(target_contract=None)
    assert 'class_names'not in raw and 'target_contract'not in raw
    assert raw['target_contract_sha256']!=prior['target_contract_sha256']
    for value,target in ((raw,spec),(prior,None),(prior,noise_margin_dataset_contract(.5))):
        with pytest.raises(ValueError,match='NATIVE_BAR_TRAIN_PRIOR_'):
            owner.require_native_bar_train_prior(value,target_contract=target)
    bad=deepcopy(prior);bad['class_names'][1]='FLAT'
    with pytest.raises(ValueError,match='NOISE_IDENTITY'):owner.require_native_bar_train_prior(bad,target_contract=spec)
    bad=deepcopy(prior);bad['per_task']['close']['M5']['probabilities']=[.5,.25,.25]
    with pytest.raises(ValueError,match='FIXED_LAPLACE_ONE'):owner.require_native_bar_train_prior(bad,target_contract=spec)
    output={key:torch.zeros(3,5,3)for key in ('open_logits','close_logits')}
    with pytest.raises(ValueError,match='NOISE_TARGET_CONTRACT'):
        owner.native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=upper,
            target_contract_sha256=contract()['contract_sha256'],target_contract=spec)
    with pytest.raises(ValueError,match='TARGET_CONTRACT'):
        owner.native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=upper,
            target_contract_sha256=spec['contract_sha256'])


def _future_state_inputs():
    from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract
    spec = fully_future_state_contract()
    tick = 300_000_000_000
    issues = np.arange(1, 4, dtype=np.int64) * tick
    duration = np.asarray([spec['tf_duration_ns'][tf] for tf in spec['timeframes']], np.int64)
    origin = np.asarray([spec['tf_origin_ns'][tf] for tf in spec['timeframes']], np.int64)
    starts = ((issues[:, None] - origin + duration - 1) // duration) * duration + origin
    cutoff = 2 * int(duration.max()) + tick
    known = np.ones((3, 5), dtype=np.bool_)
    active = np.array([[True] * 5, [False] * 5, [True] * 5], dtype=np.bool_)
    side_known = np.zeros((3, 5), dtype=np.bool_)
    side_known[2] = True
    side_up = np.zeros((3, 5), dtype=np.bool_)
    side_up[2] = [True, False, True, False, True]
    truth = {'issue_ns': issues, 'target_start_ns': starts, 'validity_ns': starts + duration,
        'target_present': known.copy(), 'mature': known.copy(), 'nominal_complete': known.copy(),
        'activity_known': known.copy(), 'activity': active, 'side_known': side_known,
        'side_up': side_up}
    output = {key: torch.zeros(3, 5, dtype=torch.float32, requires_grad=True)
              for key in ('activity_logits', 'side_logits')}
    return {'model_output': output, 'truth': truth, 'fold_upper_ns': cutoff,
            'target_contract': spec}


def test_native_future_state_partial_likelihood_factorization_and_score_support():
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    kwargs = _future_state_inputs()
    output = kwargs['model_output']
    logp = owner.native_future_state_log_probabilities(model_output=output,
        target_contract=kwargs['target_contract'])
    np.testing.assert_allclose(logp.detach().exp().numpy(),
        np.broadcast_to([.25, .5, .25], (3, 5, 3)), atol=1e-7)
    result = owner.native_future_state_loss(**kwargs)
    assert result['count'] == 15 and result['skip_update'] is False
    assert result['counts']['activity'].tolist() == [3] * 5
    assert result['counts']['side'].tolist() == [1] * 5
    assert float(result['loss']) == pytest.approx(20 * np.log(2) / 15)
    report = owner.native_future_state_score_sums(**kwargs)['per_tf']
    assert sum(cell['partial_log_loss_sum'] for cell in report.values()) == pytest.approx(
        float(result['loss']) * result['count'], abs=1e-6)
    assert sum(cell['activity']['brier_sum'] for cell in report.values()) == 15 / 4
    assert sum(cell['side']['brier_sum'] for cell in report.values()) == 5 / 4
    assert sum(cell['three_class']['count'] for cell in report.values()) == 10
    assert sum(cell['three_class']['brier_sum'] for cell in report.values()) == pytest.approx(6.25)
    assert report['D1']['activity']['unique_target_starts'] == 1
    result['loss'].backward()
    assert torch.count_nonzero(output['side_logits'].grad[:2]) == 0
    assert torch.count_nonzero(output['side_logits'].grad[2]) == 5


def test_native_future_state_unknown_side_has_zero_free_linear_parameter_gradient():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    kwargs = _future_state_inputs()
    truth = kwargs['truth']
    truth['activity'][:] = True
    truth['side_known'][:] = False
    truth['side_up'][:] = False
    hidden = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    activity_head = torch.nn.Linear(4, 5)
    side_head = torch.nn.Linear(4, 5)
    with torch.no_grad():
        for head in (activity_head, side_head):
            head.weight.zero_()
            head.bias.zero_()
    kwargs['model_output'] = {'activity_logits': activity_head(hidden), 'side_logits': side_head(hidden)}
    result = native_future_state_loss(**kwargs)
    result['loss'].backward()
    assert torch.count_nonzero(activity_head.weight.grad) == activity_head.weight.numel()
    assert torch.count_nonzero(activity_head.bias.grad) == activity_head.bias.numel()
    assert torch.equal(side_head.weight.grad, torch.zeros_like(side_head.weight))
    assert torch.equal(side_head.bias.grad, torch.zeros_like(side_head.bias))


def test_native_future_state_all_unknown_skips_and_keeps_both_gradients_zero():
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    kwargs = _future_state_inputs()
    for key in ('activity_known', 'activity', 'side_known', 'side_up'):
        kwargs['truth'][key][:] = False
    result = owner.native_future_state_loss(**kwargs)
    assert result['count'] == 0 and result['skip_update'] is True
    assert float(result['loss']) == 0
    result['loss'].backward()
    assert all(torch.equal(value.grad, torch.zeros_like(value))
               for value in kwargs['model_output'].values())
    report = owner.native_future_state_score_sums(**kwargs)['per_tf']
    assert all(cell['activity']['count'] == cell['side']['count'] ==
               cell['three_class']['count'] == 0 for cell in report.values())


@pytest.mark.parametrize('field,value', [
    ('activity_logits', torch.zeros(3, 5, dtype=torch.float64)),
    ('side_logits', torch.zeros(3, 4)),
    ('side_logits', torch.full((3, 5), float('nan'))),
])
def test_native_future_state_rejects_invalid_primitive_logits(field, value):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    kwargs = _future_state_inputs()
    kwargs['model_output'][field] = value
    with pytest.raises(ValueError, match='LOGITS'):
        native_future_state_loss(**kwargs)


@pytest.mark.parametrize('case', ['false_maturity', 'past_start', 'wrong_duration',
    'unknown_activity_value', 'uncertified_side_value', 'side_on_flat', 'incomplete_activity',
    'wrong_dtype', 'missing_axis', 'target_contract'])
def test_native_future_state_rejects_false_axes_masks_and_contract(case):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    kwargs = _future_state_inputs()
    truth = kwargs['truth']
    if case == 'false_maturity': truth['mature'][0, 0] = False
    elif case == 'past_start': truth['target_start_ns'][0, 0] = 0
    elif case == 'wrong_duration': truth['validity_ns'][0, 0] += 300_000_000_000
    elif case == 'unknown_activity_value': truth['activity_known'][0, 0] = False
    elif case == 'uncertified_side_value': truth['side_up'][0, 0] = True
    elif case == 'side_on_flat': truth['side_known'][1, 0] = True
    elif case == 'incomplete_activity': truth['nominal_complete'][0, 0] = False
    elif case == 'wrong_dtype': truth['activity'] = truth['activity'].astype(np.int8)
    elif case == 'missing_axis': del truth['target_start_ns']
    elif case == 'target_contract': kwargs['target_contract']['contract_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        native_future_state_loss(**kwargs)


def test_native_future_state_prior_uses_separate_train_support_and_frozen_output():
    from copy import deepcopy
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    kwargs = _future_state_inputs()
    truth = kwargs['truth']
    truth['activity_known'][:, 3] = False
    truth['activity'][:, 3] = False
    truth['side_known'][:, 3] = False
    truth['side_up'][:, 3] = False
    before = deepcopy(truth)
    fit_kwargs = {key: kwargs[key] for key in ('truth', 'fold_upper_ns', 'target_contract')}
    prior = owner.fit_native_future_state_train_prior(**fit_kwargs)
    assert owner.require_native_future_state_train_prior(prior,
        target_contract=kwargs['target_contract']) == prior
    assert prior['per_task']['activity']['M5']['class_counts'] == [1, 2]
    assert prior['per_task']['activity']['M5']['probabilities'] == [.4, .6]
    assert prior['per_task']['side']['M5']['class_counts'] == [0, 1]
    assert prior['per_task']['side']['M5']['probabilities'] == [1/3, 2/3]
    for task in ('activity', 'side'):
        assert prior['per_task'][task]['H4']['fit_status'] == 'UNFITTED'
        assert prior['per_task'][task]['H4']['probabilities'] == [.5, .5]
    for key in before: np.testing.assert_array_equal(truth[key], before[key])
    output = owner.native_future_state_train_prior_logits(prior, rows=3,
        target_contract=kwargs['target_contract'])
    assert all(value.dtype == torch.float32 and value.shape == (3, 5)
               and bool(torch.isfinite(value).all()) for value in output.values())
    score = owner.native_future_state_score_sums(**{**kwargs, 'model_output': output})
    assert score['per_tf']['M5']['partial_log_loss_sum'] == pytest.approx(
        -2*np.log(.6)-np.log(.4)-np.log(2/3), abs=1e-6)
    frozen = deepcopy(prior)
    truth['side_up'][2, 0] = False
    owner.native_future_state_score_sums(**{**kwargs, 'model_output': output})
    assert prior == frozen
    broken = deepcopy(prior)
    broken['per_task']['side']['M5']['probabilities'] = [.5, .5]
    with pytest.raises(ValueError, match='PRIOR_LAPLACE_ONE'):
        owner.require_native_future_state_train_prior(broken, target_contract=kwargs['target_contract'])
    broken = deepcopy(prior)
    broken['fit_population']['scope'] = 'CONTROL'
    with pytest.raises(ValueError, match='PRIOR_POPULATION'):
        owner.require_native_future_state_train_prior(broken, target_contract=kwargs['target_contract'])


def test_native_future_state_prior_rejects_later_fit_and_duplicate_issues():
    from gx1.contracts import market_state_ta_integrated_objective_v1 as owner
    from gx1.contracts.entry_model_native_input_normalization_v1 import NATIVE_M5_TRAIN_FIT_CUTOFF_NS
    kwargs = _future_state_inputs()
    fit_kwargs = {key: kwargs[key] for key in ('truth', 'fold_upper_ns', 'target_contract')}
    with pytest.raises(ValueError, match='PRIOR_TRAIN_CUTOFF'):
        owner.fit_native_future_state_train_prior(**{**fit_kwargs,
            'fold_upper_ns': NATIVE_M5_TRAIN_FIT_CUTOFF_NS + 300_000_000_000})
    kwargs['truth']['issue_ns'][1] = kwargs['truth']['issue_ns'][0]
    with pytest.raises(ValueError, match='PRIOR_ISSUE_ORDER'):
        owner.fit_native_future_state_train_prior(**fit_kwargs)


def test_native_future_state_rejects_off_phase_h4_and_d1_bars():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    for j in (3, 4):
        kwargs = _future_state_inputs()
        kwargs['truth']['target_start_ns'][0, j] += 300_000_000_000
        kwargs['truth']['validity_ns'][0, j] += 300_000_000_000
        with pytest.raises(ValueError, match='OBSERVABILITY'):
            native_future_state_loss(**kwargs)
