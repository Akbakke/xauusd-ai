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
