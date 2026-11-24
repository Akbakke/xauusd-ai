"""Synthetic canonical-target alignment, coverage and one-use authority tests."""
from __future__ import annotations
import copy,json,sys
import numpy as np
import pytest
import torch
from gx1.audit import diagnose_native_future_state_alignment_v1 as audit
from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract

T=300_000_000_000


def truth_for(issues):
    target=fully_future_state_contract(); issues=np.asarray(issues,np.int64); n=len(issues)
    duration=np.asarray([target['tf_duration_ns'][tf]for tf in audit.TF]); origin=np.asarray([target['tf_origin_ns'][tf]for tf in audit.TF])
    starts=((issues[:,None]-origin+duration-1)//duration)*duration+origin
    truth={k:np.zeros((n,5),np.bool_)for k in audit.TRUTH if k not in ('issue_ns','target_start_ns','validity_ns')}
    truth.update(issue_ns=issues,target_start_ns=starts,validity_ns=starts+duration)
    for k in ('target_present','mature','nominal_complete','state_window_complete','activity_known','activity','side_known','side_up'):truth[k][:]=True
    return truth,target


def sample():
    start=int(np.datetime64('2021-01-04T00:05','ns').astype(np.int64)); issues=start+np.arange(8,dtype=np.int64)*2*86400*10**9
    truth,target=truth_for(issues)
    # DOWN, FLAT, UP, unknown activity, active unknown side, UP, DOWN, FLAT.
    truth['side_up'][[0,1,3,4,6,7]]=False
    truth['side_known'][[1,3,4,7]]=False
    truth['activity'][[1,3,7]]=False;truth['activity_known'][3]=False
    a=np.tile(np.array([4,-4,4,4,4,4,4,-4],np.float32)[:,None],(1,5))
    s=np.tile(np.array([-4,0,4,-4,4,4,-4,0],np.float32)[:,None],(1,5))
    out={'activity_logits':a,'side_logits':s};prior={k:np.zeros_like(v)for k,v in out.items()}
    return out,truth,prior,target


def synthetic_support_prior(truth,target):
    masks=audit.objective._native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=audit.UPPER,target_contract=target)
    prior={'per_task':{},'fit_population':{'truth_sha256':audit.objective._native_prior_array_sha(truth)}}
    for task,label in (('activity','activity'),('side','side_up')):
        prior['per_task'][task]={}
        for j,tf in enumerate(audit.TF):
            take=masks[task][:,j]
            prior['per_task'][task][tf]={'count':int(take.sum()),'class_counts':np.bincount(truth[label][take,j].astype(np.int64),minlength=2).tolist(),
                'unique_target_starts':int(np.unique(truth['target_start_ns'][take,j]).size),'mask_sha256':audit.saved.array_sha(take)}
    return prior


def test_canonical_support_keeps_unknown_outside_down_flat_and_up_confusion():
    outputs,truth,prior,target=sample(); r=audit.score_split(outputs,truth,prior,target)
    assert set(r['per_tf'])==set(audit.TF)
    for cell in r['per_tf'].values():
        assert cell['conditional_side']['UP']==cell['conditional_side']['DOWN']==2
        assert cell['conditional_side']['count']==4 and cell['activity_known']==7
        assert cell['ACTIVE_known']==5 and cell['ACTIVE_side_UNKNOWN']==1 and cell['activity_UNKNOWN']==1
        c=cell['model_classification']
        assert c['known_truth_counts']=={'DOWN':2,'FLAT':2,'UP':2}
        assert c['known_truth_confusion']==[[2,0,0,0],[0,2,0,0],[0,0,2,0]]
        assert c['unresolved_truth']=={'count':2,'prediction_counts':[1,0,1,0]}
        assert c['recall']=={'DOWN':1.,'FLAT':1.,'UP':1.}
        assert sum(v['count']for v in cell['truth_strata_probabilities'].values())==8


def test_auc_and_proper_scores_use_certified_side_and_same_frozen_reference():
    outputs,truth,prior,target=sample(); r=audit.score_split(outputs,truth,prior,target)['per_tf']['M5']['conditional_side']
    p=1/(1+np.exp(-4.))
    assert r['model_AUC']==1. and r['prior_AUC']==.5
    assert r['model_Brier']==pytest.approx((1-p)**2)
    assert r['model_NLL']==pytest.approx(np.logaddexp(0.,-4.))
    assert r['prior_Brier']==.25 and r['prior_NLL']==pytest.approx(np.log(2))
    assert r['model_minus_prior_Brier']<0 and r['model_minus_prior_NLL']<0
    outputs['side_logits']=-outputs['side_logits']
    bad=audit.score_split(outputs,truth,prior,target)['per_tf']['M5']
    assert bad['conditional_side']['model_AUC']==0. and bad['model_classification']['recall']['DOWN']==0.


def test_exact_down_up_argmax_ties_are_unknown_not_a_direction_credit():
    outputs,truth,prior,target=sample(); outputs={k:np.zeros_like(v)for k,v in outputs.items()};outputs['activity_logits'][:]=100
    r=audit.score_split(outputs,truth,prior,target)['per_tf']['M5']; c=r['model_classification']
    assert c['known_truth_confusion']==[[0,0,0,2],[0,0,0,2],[0,0,0,2]]
    assert c['exact_argmax_ties']==8 and c['all_row_prediction_counts']==[0,0,0,8]
    assert c['recall']=={'DOWN':0.,'FLAT':0.,'UP':0.}
    assert r['coverage']['cells']['argmax_tie_UNKNOWN']['row_count']==8
    assert r['coverage']['cells']['singleton_argmax_tie_UNKNOWN']['row_count']==6


def test_zero_support_keeps_unknown_counts_and_undefined_metrics():
    outputs,truth,prior,target=sample()
    for k in ('activity_known','activity','side_known','side_up'):truth[k][:]=False
    r=audit.score_split(outputs,truth,prior,target)['per_tf']['D1']
    assert r['conditional_side']['count']==0 and r['conditional_side']['model_AUC']is None
    assert r['conditional_side']['model_Brier']is r['conditional_side']['prior_Brier']is None
    assert r['model_classification']['unresolved_truth']['count']==8
    assert r['model_classification']['recall']=={'DOWN':None,'FLAT':None,'UP':None}
    assert r['coverage']['cells']['activity_UNKNOWN']['row_count']==8
    assert r['fixed_output_side_intercept_derivative']['native_partial_NLL_derivative']is None


def test_one_side_auc_is_undefined_without_inventing_the_other_class():
    outputs,truth,prior,target=sample();truth['side_up'][truth['side_known']]=True
    r=audit.score_split(outputs,truth,prior,target)['per_tf']['M15']['conditional_side']
    assert r['count']==r['UP']==4 and r['DOWN']==0 and r['model_AUC']is None


def test_coverage_has_explicit_raw_and_unique_denominators_for_repeated_labels():
    issues=np.array([np.datetime64(x,'ns').astype(np.int64)for x in ('2021-01-04T00:05','2021-01-04T00:10','2021-01-05T00:05','2021-01-05T00:10')])
    truth,target=truth_for(issues);truth['side_up'][2:]=False
    outputs={k:np.zeros((4,5),np.float32)for k in audit.HEADS}
    r=audit.score_split(outputs,truth,outputs,target)['per_tf']['D1']['coverage'];c=r['cells']['side_known']
    assert c['row_count']==c['row_denominator']==4 and c['row_rate']==1.
    assert c['unique_identified_target_count']==c['unique_present_target_denominator']==2
    assert c['unique_any_issued_mask_rate']==1. and c['rows_without_target_identity']==0
    assert r['intrinsic_missing_reasons']=='NOT_IDENTIFIABLE_FROM_THE_12_CANONICAL_FIELDS'
    assert 'not independent' in r['unique_definition']


def test_all_and_each_train_year_are_same_model_strata_without_refit():
    issues=np.array([np.datetime64(f'{year}-01-04T00:{minute}','ns').astype(np.int64)for year in range(2021,2026)for minute in ('05','10')])
    truth,target=truth_for(issues);outputs={k:np.zeros((10,5),np.float32)for k in audit.HEADS}
    r=audit.analyse(outputs,truth,outputs,target)
    assert tuple(r)==audit.STRATA and r['ALL']['rows']==10
    assert all(r[str(y)]['rows']==2 and set(r[str(y)]['per_tf'])==set(audit.TF)for y in range(2021,2026))


def test_full_support_matches_original_prior_counts_classes_uniques_and_hashes():
    _,truth,_,target=sample();p=synthetic_support_prior(truth,target)
    proof=audit.verify_prior_support(truth,p,target)
    assert proof['all_counts_classes_unique_and_mask_hashes_match']is True and proof['prior_refitted']is False
    assert proof['per_task']['activity']['M5']['class_counts']==[2,5]
    assert proof['per_task']['side']['D1']['class_counts']==[2,2]


@pytest.mark.parametrize('field,value',[('count',99),('class_counts',[3,1]),('unique_target_starts',99),('mask_sha256','0'*64)])
def test_wrong_prior_population_accounting_fails_closed(field,value):
    _,truth,_,target=sample();p=synthetic_support_prior(truth,target);p['per_task']['side']['H4'][field]=value
    with pytest.raises(ValueError,match='FROZEN_PRIOR_side_H4_SUPPORT'):audit.verify_prior_support(truth,p,target)


def test_full_truth_hash_must_match_prior_population():
    _,truth,_,target=sample();p=synthetic_support_prior(truth,target);p['fit_population']['truth_sha256']='0'*64
    with pytest.raises(ValueError,match='FROZEN_PRIOR_TRUTH_HASH'):audit.verify_prior_support(truth,p,target)


def test_unsupported_side_boolean_cannot_be_counted_as_known_down():
    outputs,truth,prior,target=sample();truth['side_up'][4]=True
    with pytest.raises(ValueError,match='STATE_MASKS'):audit.score_split(outputs,truth,prior,target)


def lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
    monkeypatch.setattr(audit,'inspect',lambda *a,**k:({},binding,pref));monkeypatch.setattr(audit,'checked_inputs',lambda p:None);monkeypatch.setattr(audit,'source_clean',lambda p:True)
    original=audit.ref;monkeypatch.setattr(audit,'ref',lambda p:pref if p==audit.POLICY else binding if p==audit.PLAN else original(p))
    return binding,pref


def test_one_use_success_and_relaunch_rejection(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'calculate',lambda p:{'synthetic':True})
    r=audit.run('1'*64);terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['result']==r['result'] and terminal['remaining_invocations']==0
    assert terminal['relaunch_allowed']is False
    with pytest.raises(FileExistsError):audit.run('1'*64)


def test_failure_publishes_consumed_claim_and_no_success_result(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path)
    def fail(_):raise ValueError('synthetic')
    monkeypatch.setattr(audit,'calculate',fail)
    with pytest.raises(ValueError,match='synthetic'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED' and terminal['result']is None and terminal['remaining_invocations']==0
    assert not (audit.ROOT/'RESULT.json').exists()


def test_preflight_never_decodes_scores_or_creates_claim(monkeypatch,tmp_path,capsys):
    lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'calculate',lambda p:pytest.fail('preflight decoded payload'))
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','1'*64,'--preflight']);audit.main()
    r=json.loads(capsys.readouterr().out)
    assert r['status']=='PREFLIGHT_PASS' and r['array_payloads_decoded']==r['models']==r['forwards']==0
    assert r['claim_created']is False and not audit.ROOT.exists()


@pytest.mark.parametrize('flag',audit.saved.FLAGS)
def test_inspect_rejects_any_open_global_training_flag(monkeypatch,tmp_path,flag):
    monkeypatch.setattr(audit,'REPO',tmp_path);monkeypatch.setattr(audit,'PLAN',tmp_path/'PLAN.json');monkeypatch.setattr(audit,'POLICY',tmp_path/'POLICY.json');monkeypatch.setattr(audit,'ROOT',tmp_path/'OUT');monkeypatch.chdir(tmp_path)
    plan={'source_base_commit':'synthetic'};audit.PLAN.write_text(json.dumps(plan));binding=audit.ref(audit.PLAN)
    policy={k:False for k in audit.saved.FLAGS};policy[flag]=True;policy[audit.STAGE]=audit.gate_expected(binding,False);audit.POLICY.write_text(json.dumps(policy))
    monkeypatch.setattr(audit,'build_plan',lambda **k:plan);monkeypatch.setattr(audit,'source_clean',lambda p:True)
    with pytest.raises(ValueError,match='EXACT_ONE_USE_ALL_TRAINING_CLOSED'):audit.inspect(binding['sha256'],admitted=False)


# Separate saved-logit batch-risk audit. Existing alignment tests above stay unchanged.
def batch_risk_toy(*,heterogeneous=True):
    logits=np.zeros((4,5),np.float32);activity_known=np.ones((4,5),np.bool_)
    if heterogeneous:activity_known[2:,1:]=False
    side_known=np.zeros((4,5),np.bool_);side_known[:,0]=True
    side_up=np.zeros((4,5),np.bool_);side_up[:2,0]=True
    return logits,activity_known,side_known,side_up,np.zeros(5,np.float32)


def batch_risk_call(values,*,batch_rows=2):
    return audit.batch_risk_statistics(*values[:4],batch_rows=batch_rows,prior_logits=values[4])


def test_batch_risk_heterogeneous_activity_denominators_change_constant_optimum():
    result=batch_risk_call(batch_risk_toy());cell=result['per_tf']['M5']
    assert result['activity_denominator_total']==12 and result['updates']==2 and result['skipped_batches']==0
    assert cell['known_side_rows']==4 and cell['UP']==cell['DOWN']==2
    assert cell['constant_row_optimum']==.5
    assert cell['constant_mean_batch_optimum']==pytest.approx(1/6,abs=1e-15)
    assert cell['prior_gradient_global']==0 and cell['FINAL_gradient_global']==0
    assert cell['prior_gradient_mean_batch']==pytest.approx(.2,abs=1e-15)
    assert cell['FINAL_gradient_mean_batch']==pytest.approx(.2,abs=1e-15)


def test_batch_risk_fixed_activity_denominator_preserves_row_optimum_and_gradient():
    values=batch_risk_toy(heterogeneous=False);values[0][:,0]=np.array([.2,.7,-.3,1.2],np.float32)
    result=batch_risk_call(values);cell=result['per_tf']['M5']
    assert result['activity_denominator_total']==20
    assert cell['constant_row_optimum']==cell['constant_mean_batch_optimum']==.5
    assert cell['FINAL_gradient_global']==pytest.approx(cell['FINAL_gradient_mean_batch'],abs=1e-15)


def test_batch_risk_zero_denominator_batch_is_skipped_and_unsupported_tf_has_no_optimum():
    values=batch_risk_toy();values[1][:2]=False;values[2][:2]=False;values[3][:2]=False
    result=batch_risk_call(values)
    assert result['updates']==1 and result['skipped_batches']==1 and result['activity_denominator_total']==2
    empty=result['per_tf']['D1']
    assert empty['known_side_rows']==empty['UP']==empty['DOWN']==0
    assert empty['constant_row_optimum']is None and empty['constant_mean_batch_optimum']is None


def test_batch_risk_known_false_labels_are_down_and_unknown_false_labels_are_not_down():
    logits=np.zeros((2,5),np.float32);known=np.zeros((2,5),np.bool_);known[:,0]=True
    result=audit.batch_risk_statistics(logits,known,known,np.zeros_like(known),batch_rows=2,prior_logits=np.zeros(5,np.float32))
    assert result['per_tf']['M5']['DOWN']==2 and result['per_tf']['M5']['known_side_rows']==2
    assert result['per_tf']['M5']['constant_row_optimum']==0
    assert result['per_tf']['M5']['FINAL_gradient_global']==.5
    assert result['per_tf']['M15']['DOWN']==result['per_tf']['M15']['known_side_rows']==0


def test_batch_risk_inconsistent_unknown_side_or_activity_masks_fail_closed():
    for mode in ('side_without_activity','up_without_side'):
        values=batch_risk_toy()
        if mode=='side_without_activity':values[1][0,0]=False
        else:values[3][0,1]=True
        with pytest.raises(ValueError,match='MASK'):batch_risk_call(values)


def test_batch_risk_final_derivatives_use_f64_and_actual_activity_denominators():
    values=batch_risk_toy();values[0][:,0]=np.array([.1234567,-2.34567,1.234567,-.7654321],np.float32)
    result=batch_risk_call(values);cell=result['per_tf']['M5']
    q=1/(1+np.exp(-values[0][:,0].astype(np.float64)));residual=q-values[3][:,0].astype(np.float64)
    assert cell['FINAL_mean_q']==pytest.approx(float(q.mean()),abs=5e-16)
    assert cell['FINAL_gradient_global']==pytest.approx(float(residual.sum()/12),abs=5e-16)
    expected_mean=(residual[:2].sum()/10+residual[2:].sum()/2)/2
    assert cell['FINAL_gradient_mean_batch']==pytest.approx(float(expected_mean),abs=5e-16)


def test_batch_risk_laplace_prior_is_distinct_from_raw_and_mean_batch_optima():
    logits=np.zeros((3,5),np.float32);ak=np.ones((3,5),np.bool_);sk=np.zeros_like(ak);sk[:,0]=True
    up=np.zeros_like(ak);up[0,0]=True
    # Original prior encodes (UP+1)/(known+2), rather than the unregularized optimum.
    prior=np.full(5,np.float32(np.log(.4/.6)),np.float32)
    cell=audit.batch_risk_statistics(logits,ak,sk,up,batch_rows=2,prior_logits=prior)['per_tf']['M5']
    prior_q=float(1/(1+np.exp(-float(prior[0]))))
    assert cell['constant_row_optimum']==pytest.approx(1/3,abs=1e-15)
    assert cell['constant_mean_batch_optimum']==pytest.approx(.25,abs=1e-15)
    assert cell['prior_q']==pytest.approx(prior_q,abs=5e-16)
    assert cell['prior_gradient_global']==pytest.approx((3*prior_q-1)/15,abs=5e-16)
    assert cell['prior_q']!=cell['constant_row_optimum'] and cell['prior_q']!=cell['constant_mean_batch_optimum']


def batch_risk_lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'BATCH_ROOT',tmp_path/'OUT')
    binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,
        'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
    monkeypatch.setattr(audit,'batch_inspect',lambda *args,**kwargs:({},binding,pref))
    monkeypatch.setattr(audit,'batch_checked_inputs',lambda plan:None);monkeypatch.setattr(audit,'source_clean',lambda plan:True)
    original=audit.ref
    monkeypatch.setattr(audit,'ref',lambda path:pref if path==audit.POLICY else binding if path==audit.BATCH_PLAN else original(path))
    return binding,pref


def test_batch_risk_preflight_never_loads_payload_or_publishes_claim(monkeypatch,tmp_path,capsys):
    batch_risk_lifecycle(monkeypatch,tmp_path);seen=[]
    monkeypatch.setattr(audit,'batch_checked_inputs',lambda plan:seen.append('metadata_checked'))
    def forbidden(*args,**kwargs):raise AssertionError('preflight payload or claim')
    monkeypatch.setattr(audit.np,'load',forbidden);monkeypatch.setattr(audit.saved,'load_fields',forbidden)
    monkeypatch.setattr(audit,'batch_calculate',forbidden);monkeypatch.setattr(audit,'batch_publish',forbidden)
    monkeypatch.setattr(sys,'argv',['audit','--batch-risk-plan-sha256','1'*64,'--preflight']);audit.main()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='PREFLIGHT_BATCH_RISK_PASS' and result['claim_created']is False
    assert result['array_payloads_decoded']==result['models']==result['checkpoint_loads']==result['forwards']==result['fits']==0
    assert seen==['metadata_checked'] and not audit.BATCH_ROOT.exists()


def test_batch_risk_all_global_and_both_scope_training_flags_remain_false(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'REPO',tmp_path);monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit,'BATCH_PLAN',tmp_path/'PLAN.json');monkeypatch.setattr(audit,'POLICY',tmp_path/'POLICY.json')
    monkeypatch.setattr(audit,'BATCH_ROOT',tmp_path/'OUT')
    plan={'source_base_commit':'synthetic','inputs':{'batch_risk_design':{}}}
    audit.BATCH_PLAN.write_text(json.dumps(plan));binding=audit.ref(audit.BATCH_PLAN)
    policy={key:False for key in audit.saved.FLAGS};policy[audit.BATCH_STAGE]=audit.batch_gate_expected(binding,False)
    monkeypatch.setattr(audit,'batch_build_plan',lambda **kwargs:plan);monkeypatch.setattr(audit,'source_clean',lambda value:True)
    def checked():
        audit.POLICY.write_text(json.dumps(policy));return audit.batch_inspect(binding['sha256'],admitted=False)
    assert policy[audit.BATCH_STAGE]['training_authorized']is policy[audit.BATCH_STAGE]['test_authorized']is False
    assert checked()[0]==plan
    for flag in audit.saved.FLAGS:
        policy[flag]=True
        with pytest.raises(ValueError,match='BATCH_RISK_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
        policy[flag]=False
    for flag in ('training_authorized','test_authorized'):
        policy[audit.BATCH_STAGE][flag]=True
        with pytest.raises(ValueError,match='BATCH_RISK_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
        policy[audit.BATCH_STAGE][flag]=False


def test_batch_risk_exact_cli_rejects_repeated_preflight_before_inspection(monkeypatch):
    monkeypatch.setattr(audit,'batch_inspect',lambda *args,**kwargs:pytest.fail('invalid CLI reached authority'))
    monkeypatch.setattr(sys,'argv',['audit','--batch-risk-plan-sha256','1'*64,'--preflight','--preflight'])
    with pytest.raises(ValueError,match='BATCH_RISK_EXACT_CLI'):audit.main()


def test_batch_risk_failure_consumes_claim_and_retains_partial_progress(monkeypatch,tmp_path):
    batch_risk_lifecycle(monkeypatch,tmp_path)
    def fail(plan,*,progress):
        progress['phase']='synthetic stopped';raise ValueError('synthetic batch risk failure')
    monkeypatch.setattr(audit,'batch_calculate',fail)
    with pytest.raises(ValueError,match='synthetic batch risk failure'):audit.batch_run('1'*64)
    terminal=json.loads((audit.BATCH_ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED' and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert terminal['result']is None and terminal['progress']['phase']=='synthetic stopped'
    assert (audit.BATCH_ROOT/'CLAIM.json').exists() and not (audit.BATCH_ROOT/'RESULT.json').exists()


def test_batch_risk_success_is_durable_and_source_bookend_stops_later_success(monkeypatch,tmp_path):
    batch_risk_lifecycle(monkeypatch,tmp_path/'good');monkeypatch.setattr(audit,'batch_calculate',lambda plan,**kwargs:{'synthetic':True})
    result=audit.batch_run('1'*64);terminal=json.loads((audit.BATCH_ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['result']==result['result'] and terminal['remaining_invocations']==0
    assert terminal['relaunch_allowed']is False and json.loads((audit.BATCH_ROOT/'RESULT.json').read_text())['economic_edge_proven']is False
    batch_risk_lifecycle(monkeypatch,tmp_path/'drift');monkeypatch.setattr(audit,'source_clean',lambda plan:False)
    with pytest.raises(ValueError,match='BATCH_RISK_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):audit.batch_run('1'*64)
    terminal=json.loads((audit.BATCH_ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED' and not (audit.BATCH_ROOT/'RESULT.json').exists()
# Separate saved-state zero-gradient AdamW projection; no model or optimizer construction.
def moment_toy_options():
 return {'lr':.01,'betas':(.8,.9),'eps':1e-8,'weight_decay':.2,'amsgrad':False,'maximize':False,
  'foreach':False,'capturable':False,'differentiable':False,'fused':False}


def test_moment_projection_positive_first_moment_lowers_parameter_and_negative_pushes_up():
 p=np.zeros(2,np.float64);m=np.array([1.,-1.]);v=np.ones(2,np.float64)
 result=audit.moment_zero_gradient_delta(p,m,v,7,moment_toy_options())
 assert result['delta'][0]<0 and result['delta'][1]>0
 assert np.array_equal(result['decay_delta'],np.zeros(2))


def test_moment_projection_uses_actual_step_bias_corrections_and_resets_only_first_moment():
 options=moment_toy_options();p=np.array([2.,-3.]);m=np.array([1.,-2.]);v=np.array([4.,9.]);step=7
 original=audit.moment_zero_gradient_delta(p,m,v,step,options)
 reset=audit.moment_zero_gradient_delta(p,m,v,step,options,reset_first_moment=True)
 b1,b2=options['betas'];t=step+1
 expected_decay=-options['lr']*options['weight_decay']*p
 expected_moment=-options['lr']*(b1*m)/(1-b1**t)/(np.sqrt(b2*v/(1-b2**t))+options['eps'])
 np.testing.assert_allclose(original['moment_delta'],expected_moment,rtol=1e-14,atol=0)
 np.testing.assert_array_equal(original['decay_delta'],expected_decay)
 np.testing.assert_allclose(original['delta'],expected_decay+expected_moment,rtol=1e-14,atol=0)
 np.testing.assert_array_equal(reset['decay_delta'],original['decay_delta'])
 np.testing.assert_array_equal(reset['moment_delta'],np.zeros_like(p))
 np.testing.assert_array_equal(reset['delta'],expected_decay)
 wrong_step=audit.moment_zero_gradient_delta(p,m,v,0,options)
 assert not np.allclose(wrong_step['moment_delta'],original['moment_delta'],rtol=1e-6,atol=0)


def test_moment_projection_preserves_inputs_and_rejects_negative_v_or_invalid_options():
 p=np.array([2.,-3.]);m=np.array([1.,-2.]);v=np.array([4.,9.]);options=moment_toy_options()
 before=(p.copy(),m.copy(),v.copy(),copy.deepcopy(options))
 audit.moment_zero_gradient_delta(p,m,v,7,options,reset_first_moment=True)
 for actual,expected in zip((p,m,v),before[:3]):np.testing.assert_array_equal(actual,expected)
 assert options==before[3]
 with pytest.raises(ValueError):audit.moment_zero_gradient_delta(p,m,-v,7,options)
 for key,value in (('betas',(1.,.9)),('eps',-1e-8),('lr',-.01),('weight_decay',-.1)):
  invalid={**options,key:value}
  with pytest.raises(ValueError):audit.moment_zero_gradient_delta(p,m,v,7,invalid)


def moment_toy_projection():
 hidden=np.zeros((2,5,2),np.float32);hidden[:,:,0]=1.;logits=np.zeros((2,5),np.float32)
 ak=np.ones((2,5),np.bool_);sk=np.zeros((2,5),np.bool_);sk[:,0]=True;up=np.zeros_like(sk);heads={}
 for tf in audit.train.TF:
  heads[tf]={'weight':np.zeros(2,np.float64),'bias':np.zeros(1,np.float64),
   'weight_exp_avg':np.zeros(2,np.float64),'bias_exp_avg':np.zeros(1,np.float64),
   'weight_exp_avg_sq':np.ones(2,np.float64),'bias_exp_avg_sq':np.ones(1,np.float64),'step':7,'options':moment_toy_options()}
 return hidden,logits,ak,sk,up,heads


def test_moment_projection_weight_effect_uses_hidden_while_bias_effect_does_not():
 values=moment_toy_projection();values[-1]['M5']['weight_exp_avg'][0]=1.
 cell=audit.moment_projection_report(*values)['per_tf']['M5'];arm=cell['arms']['original_m']
 expected=audit.moment_zero_gradient_delta(np.zeros(2),np.array([1.,0.]),np.ones(2),7,moment_toy_options())['delta'][0]
 assert arm['bias_delta']==0 and arm['mean_delta_side_logit_known']==pytest.approx(expected,abs=1e-15)
 assert cell['arms']['reset_m']['mean_delta_side_logit_known']==0
 values[0][:]=0;values[-1]['M5']['weight_exp_avg'][:]=0;values[-1]['M5']['bias_exp_avg'][0]=-1.
 arm=audit.moment_projection_report(*values)['per_tf']['M5']['arms']['original_m']
 assert arm['projection_weight_delta_l2']==0 and arm['bias_delta']>0
 assert arm['mean_delta_side_logit_known']==arm['bias_delta']


def test_moment_projection_conditional_probability_is_not_scaled_by_activity_support():
 values=moment_toy_projection();values[1][:,0]=np.float32(.8);values[-1]['M5']['bias'][0]=float(np.float32(.8))
 dense=audit.moment_projection_report(*values);values[2][:]=False;values[2][:,0]=True
 sparse=audit.moment_projection_report(*values)
 a,b=dense['per_tf']['M5'],sparse['per_tf']['M5']
 assert dense['activity_denominator_all_TF']==10 and sparse['activity_denominator_all_TF']==2
 assert a['baseline']==b['baseline'] and a['baseline']['mean_q']==pytest.approx(1/(1+np.exp(-float(np.float32(.8)))),abs=1e-15)
 assert a['arms']['original_m']['mean_q']==b['arms']['original_m']['mean_q']
 assert b['arms']['original_m']['native_partial_directional_derivative']==pytest.approx(5*a['arms']['original_m']['native_partial_directional_derivative'],abs=1e-15)
 assert 'no joint' in dense['probability_semantics']


def test_moment_projection_native_partial_directional_derivative_and_alignment_match_fixed_hidden():
 values=moment_toy_projection();head=values[-1]['M5'];head['options']['weight_decay']=0.
 head['weight_exp_avg'][0]=1.;head['bias_exp_avg'][0]=.5
 cell=audit.moment_projection_report(*values)['per_tf']['M5'];arm=cell['arms']['original_m']
 dz=arm['mean_delta_side_logit_known'];assert dz<0
 assert arm['native_partial_directional_derivative']==pytest.approx(dz/10,abs=1e-15)
 assert arm['native_partial_moment_directional_derivative']==pytest.approx(dz/10,abs=1e-15)
 assert cell['fixed_cohort_gradient_dot_first_moment']==pytest.approx(.15,abs=1e-15)
 assert arm['NLL_delta']<0 and cell['arms']['reset_m']['native_partial_directional_derivative']==0
 effect=cell['first_moment_effect_original_minus_reset']
 assert effect['mean_delta_side_logit_known']==pytest.approx(dz,abs=1e-15)


def test_moment_projection_no_side_support_retains_unknown_metrics():
 values=moment_toy_projection();cell=audit.moment_projection_report(*values)['per_tf']['D1']
 assert cell['baseline']['count']==0 and cell['baseline']['mean_q']is None
 assert cell['baseline']['Brier']is cell['baseline']['NLL']is None
 for arm in cell['arms'].values():
  assert arm['count']==0 and arm['mean_q']is None and arm['mean_delta_side_logit_known']is None
  assert arm['native_partial_directional_derivative']is None
 assert cell['fixed_cohort_gradient_dot_first_moment']is None and cell['support_status'].startswith('NO_SUPPORT')


def moment_toy_checkpoint():
 import torch
 recipe=moment_toy_options();names=[];ids=[];model={};state={}
 for j,tf in enumerate(audit.train.TF):
  for k,shape in (('weight',(1,2)),('bias',(1,))):
   name=f'readout.side_heads.{tf}.{k}';index=101+2*j if k=='weight'else 5+3*j
   names.append(name);ids.append(index);model[name]=torch.full(shape,float(j+1),dtype=torch.float32)
   state[index]={'step':torch.tensor(7.,dtype=torch.float32),'exp_avg':torch.full(shape,float(j+.25),dtype=torch.float32),'exp_avg_sq':torch.ones(shape,dtype=torch.float32)}
 cp={'optimizer_state':{'state':state,'param_groups':[{**copy.deepcopy(recipe),'params':ids}]},'optimizer_parameter_names':names,
  'scheduler_state':None,'position':{'optimizer_updates':7},'model_state':model}
 return cp,recipe


def test_moment_projection_extracts_actual_nonordinal_parameter_ids_without_construction(monkeypatch):
 import torch
 cp,recipe=moment_toy_checkpoint()
 monkeypatch.setattr(torch.optim,'AdamW',lambda *args,**kwargs:pytest.fail('optimizer construction'))
 heads=audit.moment_extract_heads(cp,recipe,width=2)
 assert heads['M5']['parameter_ids']=={'weight':101,'bias':5}
 assert heads['D1']['parameter_ids']=={'weight':109,'bias':17}
 assert heads['M5']['parameter_names']['weight']=='readout.side_heads.M5.weight'
 assert heads['M5']['step']==7 and heads['M5']['weight'].dtype==np.float64 and heads['M5']['weight'].shape==(2,)
 np.testing.assert_array_equal(heads['M5']['weight_exp_avg'],np.full(2,.25))
 heads['M5']['weight'][0]=99
 assert float(cp['model_state']['readout.side_heads.M5.weight'][0,0])==1.


def test_moment_projection_extract_rejects_duplicate_ids_wrong_name_recipe_and_actual_step():
 import torch
 for mode in ('ids','name','recipe','step'):
  cp,recipe=moment_toy_checkpoint();group=cp['optimizer_state']['param_groups'][0]
  if mode=='ids':group['params'][1]=group['params'][0]
  elif mode=='name':cp['optimizer_parameter_names'][0]='readout.side_heads.FOREIGN.weight'
  elif mode=='recipe':group['amsgrad']=True
  else:cp['optimizer_state']['state'][group['params'][0]]['step']=torch.tensor(6.,dtype=torch.float32)
  with pytest.raises(ValueError,match='MOMENT_'):audit.moment_extract_heads(cp,recipe,width=2)


def moment_projection_lifecycle(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 monkeypatch.setattr(audit,'MOMENT_ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
 monkeypatch.setattr(guard,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
 monkeypatch.setattr(guard,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
 monkeypatch.setattr(audit,'moment_inspect',lambda *args,**kwargs:({},binding,pref));monkeypatch.setattr(audit,'moment_checked_inputs',lambda plan:None)
 monkeypatch.setattr(audit,'moment_source_clean',lambda plan:True);original=audit.ref
 monkeypatch.setattr(audit,'ref',lambda path:pref if path==audit.POLICY else binding if path==audit.MOMENT_PLAN else original(path))
 return binding,pref


def test_moment_projection_preflight_has_no_checkpoint_arrays_or_claim(monkeypatch,tmp_path,capsys):
 import json,sys,torch
 moment_projection_lifecycle(monkeypatch,tmp_path);seen=[]
 monkeypatch.setattr(audit,'moment_checked_inputs',lambda plan:seen.append('metadata'))
 def forbidden(*args,**kwargs):raise AssertionError('preflight CP/array/claim')
 monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(audit.np,'load',forbidden)
 monkeypatch.setattr(audit,'moment_calculate',forbidden);monkeypatch.setattr(audit,'moment_publish',forbidden)
 monkeypatch.setattr(sys,'argv',['audit','--projection-moment-plan-sha256','1'*64,'--preflight']);audit.main()
 result=json.loads(capsys.readouterr().out)
 assert result['status']=='PREFLIGHT_PROJECTION_MOMENT_PASS'and result['claim_created']is False
 assert result['checkpoint_payloads']==result['array_payloads_decoded']==result['models']==result['forwards']==result['optimizer_updates']==0
 assert seen==['metadata']and not audit.MOMENT_ROOT.exists()


def test_moment_projection_global_and_both_scope_training_flags_remain_closed(monkeypatch,tmp_path):
 import json
 monkeypatch.setattr(audit,'REPO',tmp_path);monkeypatch.chdir(tmp_path);monkeypatch.setattr(audit,'MOMENT_PLAN',tmp_path/'PLAN.json')
 monkeypatch.setattr(audit,'POLICY',tmp_path/'POLICY.json');monkeypatch.setattr(audit,'MOMENT_ROOT',tmp_path/'OUT')
 plan={'source_base_commit':'synthetic','inputs':{'design':{}}};audit.MOMENT_PLAN.write_text(json.dumps(plan));binding=audit.ref(audit.MOMENT_PLAN)
 flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
 policy={key:False for key in flags};policy[audit.MOMENT_STAGE]=audit.moment_gate_expected(binding,False)
 monkeypatch.setattr(audit,'moment_build_plan',lambda **kwargs:plan);monkeypatch.setattr(audit,'moment_source_clean',lambda value:True)
 def checked():
  audit.POLICY.write_text(json.dumps(policy));return audit.moment_inspect(binding['sha256'],admitted=False)
 assert policy[audit.MOMENT_STAGE]['training_authorized']is policy[audit.MOMENT_STAGE]['test_authorized']is False
 assert checked()[0]==plan
 for flag in flags:
  policy[flag]=True
  with pytest.raises(ValueError,match='MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
  policy[flag]=False
 for flag in ('training_authorized','test_authorized'):
  policy[audit.MOMENT_STAGE][flag]=True
  with pytest.raises(ValueError,match='MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
  policy[audit.MOMENT_STAGE][flag]=False


def test_moment_projection_exact_cli_rejects_duplicate_preflight_before_authority(monkeypatch):
 import sys
 monkeypatch.setattr(audit,'moment_inspect',lambda *args,**kwargs:pytest.fail('bad CLI reached authority'))
 monkeypatch.setattr(sys,'argv',['audit','--projection-moment-plan-sha256','1'*64,'--preflight','--preflight'])
 with pytest.raises(ValueError,match='MOMENT_EXACT_CLI'):audit.main()


def test_moment_projection_failure_consumes_claim_and_retains_progress(monkeypatch,tmp_path):
 import json
 moment_projection_lifecycle(monkeypatch,tmp_path)
 def fail(plan,*,progress):
  progress['phase']='synthetic stopped';raise ValueError('synthetic moment failure')
 monkeypatch.setattr(audit,'moment_calculate',fail)
 with pytest.raises(ValueError,match='synthetic moment failure'):audit.moment_run('1'*64)
 terminal=json.loads((audit.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert terminal['result']is None and terminal['progress']['phase']=='synthetic stopped'
 assert (audit.MOMENT_ROOT/'CLAIM.json').exists()and not (audit.MOMENT_ROOT/'RESULT.json').exists()


def test_moment_projection_success_is_durable_and_source_bookend_blocks_later_result(monkeypatch,tmp_path):
 import json
 moment_projection_lifecycle(monkeypatch,tmp_path/'good');monkeypatch.setattr(audit,'moment_calculate',lambda plan,**kwargs:{'synthetic':True})
 result=audit.moment_run('1'*64);terminal=json.loads((audit.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='COMPLETE'and terminal['result']==result['result']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert json.loads((audit.MOMENT_ROOT/'RESULT.json').read_text())['economic_edge_proven']is False
 moment_projection_lifecycle(monkeypatch,tmp_path/'drift');monkeypatch.setattr(audit,'moment_source_clean',lambda plan:False)
 with pytest.raises(ValueError,match='MOMENT_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):audit.moment_run('1'*64)
 terminal=json.loads((audit.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='FAILED_CONSUMED'and not (audit.MOMENT_ROOT/'RESULT.json').exists()


def test_moment_projection_real_builder_uses_frozen_owner_metadata_before_closed_receipt_gate(monkeypatch):
 design_ref={'path':str(audit.MOMENT_EROOT/'PROJECTION_MOMENT_DESIGN001.json'),'sha256':'a'*64}
 plan_ref={'path':'/synthetic/frozen_plan.json','sha256':'b'*64}
 result_ref={'path':str(audit.frozen_owner.ROOT/'RESULT.json'),'sha256':'c'*64}
 terminal_ref={'path':str(audit.frozen_owner.ROOT/'TERMINAL.json'),'sha256':'d'*64}
 design={'schema_version':'gx1_native_projection_moment_design_v1','execution_authorized':False,
  'execution':audit.MOMENT_EXECUTION,'budget':audit.MOMENT_BUDGET}
 seen=[]
 def reference(path):
  seen.append(path)
  if path==audit.frozen_owner.ROOT/'RESULT.json':return result_ref
  if path==audit.frozen_owner.ROOT/'TERMINAL.json':return terminal_ref
  raise AssertionError('unexpected metadata reference '+str(path))
 def read(binding):
  if binding==design_ref:return design
  if binding==result_ref:return {'plan':plan_ref}
  if binding==terminal_ref:return {'status':'FAILED_CONSUMED'}
  if binding==plan_ref:return {}
  raise AssertionError('unexpected metadata read '+str(binding))
 def forbidden(*args,**kwargs):raise AssertionError('CP/array payload before authentic frozen receipt')
 monkeypatch.setattr(audit,'ref',reference);monkeypatch.setattr(audit,'read',read)
 monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(audit.np,'load',forbidden)
 with pytest.raises(ValueError,match='MOMENT_AUTHENTIC_FROZEN_COHORT'):
  audit.moment_build_plan(design=design_ref)
 assert seen==[audit.frozen_owner.ROOT/'RESULT.json',audit.frozen_owner.ROOT/'TERMINAL.json']


# Provider-HLC fixed-output arithmetic; existing v1 tests and owners stay unchanged.
def hlc_batch_toy():
    values = batch_risk_toy()
    issues = np.array(['2021-12-31T23:55', '2022-01-01T00:00',
                       '2022-01-01T00:05', '2022-01-01T00:10'], dtype='datetime64[ns]').astype(np.int64)
    outputs = {'side_logits': values[0], 'activity_logits': np.zeros_like(values[0])}
    truth = {'issue_ns': issues, 'activity_known': values[1], 'side_known': values[2],
             'side_up': values[3], 'activity': values[1].copy()}
    prior = {'side_logits': values[4], 'activity_logits': values[4].copy()}
    return outputs, truth, prior


def test_provider_hlc_batch_decomposes_activity_and_side_with_native_denominator():
    outputs, truth, prior = hlc_batch_toy()
    report = audit.provider_hlc_batch_statistics(outputs, truth, batch_rows=2, prior_logits=prior)
    assert report['activity_denominator_total'] == 12
    for name in ('FINAL', 'prior'):
        loss = report['loss_decomposition'][name]
        assert loss['global_native']['activity'] == pytest.approx(np.log(2), abs=1e-15)
        assert loss['global_native']['side'] == pytest.approx(np.log(2) / 3, abs=1e-15)
        assert loss['global_native']['total'] == pytest.approx(4 * np.log(2) / 3, abs=1e-15)
        assert loss['mean_supported_B96_native']['activity'] == pytest.approx(np.log(2), abs=1e-15)
        assert loss['mean_supported_B96_native']['side'] == pytest.approx(.6 * np.log(2), abs=1e-15)
    assert report['per_tf']['M5']['constant_mean_batch_optimum'] == pytest.approx(1 / 6)


def test_provider_hlc_batch_year_crossing_preserves_full_original_denominator():
    outputs, truth, prior = hlc_batch_toy()
    report = audit.provider_hlc_batch_statistics(outputs, truth, batch_rows=2, prior_logits=prior)
    assert report['year_crossing_logical_batches'] == [1]
    first, second = report['issue_UTC_years']['2021'], report['issue_UTC_years']['2022']
    assert first['rows'] == 1 and second['rows'] == 3
    assert first['FINAL_gradient_mean_B96_contribution'][0] == pytest.approx(-.025, abs=1e-15)
    assert second['FINAL_gradient_mean_B96_contribution'][0] == pytest.approx(.225, abs=1e-15)
    assert first['FINAL']['mean_supported_B96_contribution']['side'] == pytest.approx(.05 * np.log(2), abs=1e-15)
    for name in ('FINAL', 'prior'):
        for task in ('activity', 'side'):
            assert sum(year[name]['global_native_contribution'][task] for year in (first, second)) == pytest.approx(report['loss_decomposition'][name]['global_native'][task], abs=1e-15)
            assert sum(year[name]['mean_supported_B96_contribution'][task] for year in (first, second)) == pytest.approx(report['loss_decomposition'][name]['mean_supported_B96_native'][task], abs=1e-15)
        for j, tf in enumerate(audit.TF):
            assert sum(year[name+'_gradient_global_contribution'][j] for year in (first, second)) == pytest.approx(report['per_tf'][tf][name+'_gradient_global'], abs=1e-15)
            assert sum(year[name+'_gradient_mean_B96_contribution'][j] for year in (first, second)) == pytest.approx(report['per_tf'][tf][name+'_gradient_mean_batch'], abs=1e-15)


def test_provider_hlc_batch_unknown_side_is_zero_without_filling_labels():
    outputs, truth, prior = hlc_batch_toy(); truth['side_known'][:] = False; truth['side_up'][:] = False
    outputs['side_logits'][:] = 100
    report = audit.provider_hlc_batch_statistics(outputs, truth, batch_rows=2, prior_logits=prior)
    assert report['loss_decomposition']['FINAL']['global_native']['side'] == 0
    for cell in report['per_tf'].values():
        assert cell['known_side_rows'] == 0 and cell['FINAL_gradient_global'] == 0
        assert cell['FINAL_gradient_conditional_side'] is None
    assert report['loss_decomposition']['FINAL']['global_native']['activity'] > 0


def test_provider_hlc_batch_no_support_keeps_undefined_risk_and_json_finite():
    outputs, truth, prior = hlc_batch_toy()
    for name in ('activity_known', 'activity', 'side_known', 'side_up'): truth[name][:] = False
    report = audit.provider_hlc_batch_statistics(outputs, truth, batch_rows=2, prior_logits=prior)
    assert report['updates'] == 0 and report['skipped_batches'] == 2
    assert report['loss_decomposition']['FINAL']['global_native']['total'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bad', ['activity_unknown', 'activity_dtype', 'activity_logits', 'prior_activity', 'issue_dtype'])
def test_provider_hlc_batch_rejects_invalid_new_arithmetic_inputs(bad):
    outputs, truth, prior = hlc_batch_toy()
    if bad == 'activity_unknown': truth['activity'][2, 1] = True
    elif bad == 'activity_dtype': truth['activity'] = truth['activity'].astype(np.int64)
    elif bad == 'activity_logits': outputs['activity_logits'][0, 0] = np.nan
    elif bad == 'prior_activity': prior['activity_logits'] = prior['activity_logits'].astype(np.float64)
    else: truth['issue_ns'] = truth['issue_ns'].astype(np.float64)
    with pytest.raises(ValueError): audit.provider_hlc_batch_statistics(outputs, truth, batch_rows=2, prior_logits=prior)


def hlc_batch_lifecycle(monkeypatch, tmp_path):
    import signal
    monkeypatch.setattr(audit, 'HLC_BATCH_ROOT', tmp_path / 'OUT')
    monkeypatch.setattr(audit, 'HLC_BATCH_PLAN', tmp_path / 'PLAN.json')
    monkeypatch.setattr(audit, 'POLICY', tmp_path / 'POLICY.json')
    audit.HLC_BATCH_PLAN.write_text('{}'); audit.POLICY.write_text('{}')
    binding, pref = audit.ref(audit.HLC_BATCH_PLAN), audit.ref(audit.POLICY)
    plan = {'inputs': {'capture': {'path': str(tmp_path / 'capture.npz')},
                       'target_arrays': {'path': str(tmp_path / 'targets.npz')}}}
    monkeypatch.setattr(audit, 'provider_hlc_batch_inspect', lambda *a, **k: (plan, binding, pref))
    monkeypatch.setattr(audit, 'provider_hlc_batch_checked_inputs', lambda plan: None)
    monkeypatch.setattr(audit, 'source_clean', lambda plan: True)
    monkeypatch.setattr(audit, 'require_capped_cpu_audit_execution', lambda: {'memory_max': 4*1024**3,
        'memory_high': 4*1024**3, 'swap': 512*1024**2, 'numerical_threads': 1, 'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    hooks = []; monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    timers = []; monkeypatch.setattr(signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(audit, 'provider_hlc_batch_calculate', lambda *a, **k: {'synthetic': True})
    return plan, binding, pref, hooks, timers


def test_provider_hlc_batch_preflight_exact_dispatch_has_no_claim_payload_or_quotes(monkeypatch, tmp_path, capsys):
    _, binding, _, _, _ = hlc_batch_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', ['audit', '--provider-hlc-batch-risk-plan-sha256', binding['sha256'], '--preflight'])
    audit.main(); result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PREFLIGHT_CONDITIONAL_PROVIDER_HLC_BATCH_PASS'
    assert result['array_payloads_decoded'] == result['cached_provider_records_parsed'] == result['models'] == 0
    assert not audit.HLC_BATCH_ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--preflight'])
    with pytest.raises(ValueError, match='HLC_BATCH_EXACT_CLI'): audit.main()


@pytest.mark.parametrize('failure', [None, 'calculate', 'bookend'])
def test_provider_hlc_batch_success_and_consumed_failure_reports(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers = hlc_batch_lifecycle(monkeypatch, tmp_path)
    if failure == 'calculate':
        monkeypatch.setattr(audit, 'provider_hlc_batch_calculate', lambda *a, **k: (_ for _ in ()).throw(ValueError('synthetic_calculate')))
    elif failure == 'bookend': monkeypatch.setattr(audit, 'source_clean', lambda plan: False)
    if failure:
        with pytest.raises(ValueError): audit.provider_hlc_batch_run(binding['sha256'])
    else:
        result = audit.provider_hlc_batch_run(binding['sha256'])
        assert result['result'] == audit.ref(audit.HLC_BATCH_ROOT / 'RESULT.json')
    terminal = json.loads((audit.HLC_BATCH_ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((audit.HLC_BATCH_ROOT / 'CLAIM.json').read_text())
    assert claim['policy_at_claim'] == pref and claim['training_authorized'] is False
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert (audit.HLC_BATCH_ROOT / 'RESULT.json').exists() is (failure is None)
    assert timers[-1] == (audit.signal.ITIMER_REAL, 0)


def test_provider_hlc_batch_audit_allows_only_two_npz_and_rejects_CP_features_network(monkeypatch, tmp_path):
    plan, binding, _, hooks, _ = hlc_batch_lifecycle(monkeypatch, tmp_path)
    audit.provider_hlc_batch_run(binding['sha256']); assert len(hooks) == 1
    for key in ('capture', 'target_arrays'): hooks[0]('open', (plan['inputs'][key]['path'],))
    for name in ('FINAL.pt', 'FEATURE_INPUTS.npz', 'foreign.parquet'):
        with pytest.raises(RuntimeError, match='HLC_BATCH_PAYLOAD_FORBIDDEN'): hooks[0]('open', (str(tmp_path / name),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='HLC_BATCH_NETWORK_FORBIDDEN'): hooks[0](event, ())


@pytest.mark.parametrize('admitted', [False, True])
def test_provider_hlc_batch_gate_keeps_training_and_test_closed(admitted):
    gate = audit.provider_hlc_batch_gate({'path': '/synthetic/plan.json', 'sha256': 'a'*64}, admitted)
    assert gate['execution_authorized'] is admitted and gate['remaining_invocations'] == int(admitted)
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False



def provider_hlc_builder_fixture(monkeypatch, tmp_path):
    from pathlib import Path
    from gx1.audit import native_future_state_training_v1 as training
    from gx1.contracts import native_future_state_provider_hlc_v1 as provider
    budget = {'HTTP_calls': 0,
     'MBA_OHLC_price_scalars_parsed': 15480,
     'NN_forwards': 0,
     'NPZ_ndarray_bytes_decoded': 67719050,
     'VAL_TEST_rows': 0,
     'analytic_logit_derivatives': 'Nonzero arithmetic onfixedoutputs allowed; no '
                                   'parametergrad/autograd/optimizer.',
     'autograd_calls': 0,
     'backward_calls': 0,
     'batch_rows': 96,
     'cached_provider_response_JSON_parses': 1,
     'cached_provider_response_bytes': 614909,
     'cached_provider_response_reads': 1,
     'capture_archive_loads': 1,
     'capture_fields': ['issue_ns', 'activity_logits', 'side_logits'],
     'certificate_array_memory_limit': 'Returnedarraysonly; '
                                       'intermediateallocationsarenotcounted/claimed; '
                                       'kernel4Gpeakcapapplies.',
     'certificate_metadata_bytes': 3460,
     'certificate_metadata_reads': 1,
     'certificate_reauthentications': 1,
     'checkpoint_loads': 0,
     'features': 0,
     'fits': 0,
     'last_batch_rows': 22,
     'models': 0,
     'new_labels': 0,
     'optimizer_constructions': 0,
     'optimizer_updates': 0,
     'original_batches': 3694,
     'parameter_gradient_calculations': 0,
     'prior_refits': 0,
     'provider_candle_records': 1290,
     'returned_certificate_arrays': {'mid_hlc_bytes': 30960,
                                     'target_start_ns_bytes': 10320,
                                     'total_bytes': 51600,
                                     'validity_ns_bytes': 10320},
     'rows': 354550,
     'source_timestamp_strings_parsed': 1290,
     'target_archive_loads': 1,
     'target_fields': ['issue_ns',
                       'target_start_ns',
                       'validity_ns',
                       'target_present',
                       'mature',
                       'nominal_complete',
                       'activity_known',
                       'activity',
                       'side_known',
                       'side_up',
                       'state_window_complete',
                       'partial_range_active',
                       'provider_hlc_observed',
                       'state_hlc_observed'],
     'volume_integers_checked': 1290}
    fields = ('issue_ns', 'target_start_ns', 'validity_ns', 'target_present', 'mature',
              'nominal_complete', 'state_window_complete', 'partial_range_active',
              'activity_known', 'activity', 'side_known', 'side_up',
              'provider_hlc_observed', 'state_hlc_observed')
    def binding(name): return {'path': str(tmp_path / (name + '.json')), 'sha256': 'a' * 64}
    roles = ('capture', 'certificate', 'completed_model_plan', 'completed_model_result',
             'completed_model_terminal', 'dataset', 'full_epoch_review', 'full_quality',
             'initial_result', 'main_goal', 'original_epoch_plan', 'previous_causal_design',
             'previous_post_epoch_scope', 'prior', 'provider_response',
             'qualification_requirements', 'target_acceptance', 'target_arrays')
    inputs = {key: binding(key) for key in roles}
    Path(inputs['provider_response']['path']).write_bytes(b'x' * 614909)
    Path(inputs['certificate']['path']).write_bytes(b'y' * 3460)
    source = {key: binding(key) for key in ('base_dataset', 'base_targets', 'native_manifest')}
    source['reference_raw'] = inputs['provider_response']
    target = {'schema_version': 'synthetic_v2', 'contract_sha256': 'b' * 64}
    truth_sha = 'c' * 64; issue_sha = 'd' * 64
    schedule = {'rows': 354550, 'batch_rows': 96, 'logical_batches': 3694,
                'batches_per_pass': 3694, 'passes': 1, 'shuffle': False,
                'drop_last': False, 'skipped_batches': 0, 'issue_ns_sha256': issue_sha,
                'batch_issue_ns_sha256': ['e' * 64] * 3694, 'update_flags': [True] * 3694}
    recipe = {'train_batch_rows': 96}
    meta = {'target_contract': target, 'dataset': inputs['dataset'], 'prior': inputs['prior'],
            'TRAIN_issue_ns_sha256': issue_sha, 'truth_sha256': truth_sha}
    original = {'metadata': meta, 'schedule': schedule, 'recipe': recipe,
                'model_contract': {'contract_sha256': 'f' * 64}}
    trajectory = {'schedule': copy.deepcopy(schedule), 'recipe': recipe,
                  'model_contract': original['model_contract'], 'target_contract': target}
    hashes = {'activity_logits': '1' * 64, 'side_logits': '2' * 64}
    report = {'trajectory': trajectory, 'actual_optimizer_updates': 3694, 'passes': 1,
              'full_TRAIN_rows': 354550, 'original_plan': inputs['original_epoch_plan'],
              'final_logits': inputs['capture'], 'final_quality': inputs['full_quality'],
              'whole_FINAL_score': {'truth_sha256': truth_sha, 'output_sha256': hashes}}
    result = {'plan': inputs['completed_model_plan'], 'source_inputs_policy_boot_unchanged': True,
              'test_data_used': False, 'report': report}
    terminal = {'status': 'COMPLETE', 'exit_code': 0, 'result': inputs['completed_model_result'],
                'plan': inputs['completed_model_plan'], 'relaunch_allowed': False}
    completed = {'metadata': {'original': copy.deepcopy(original)},
                 'source_owners': {'synthetic_numeric.py': 'a' * 64}}
    prior = {'fit_population': {'issue_rows': 354550, 'issue_ns_sha256': issue_sha,
             'cutoff_ns': audit.UPPER, 'truth_sha256': truth_sha}, 'contract_sha256': '3' * 64}
    quality = {'truth_sha256': truth_sha, 'output_sha256': hashes,
               'frozen_prior_contract_sha256': prior['contract_sha256'], 'rows': 354550}
    headers = {key: {'shape': [354550] if key == 'issue_ns' else [354550, 5],
                    'dtype': '<i8' if key in ('issue_ns', 'target_start_ns', 'validity_ns') else '|b1',
                    'sha256': issue_sha if key == 'issue_ns' else '4' * 64} for key in fields}
    certificate = {**inputs['certificate'], 'target_contract_sha256': target['contract_sha256'],
                   'source_binding_sha256': '5' * 64}
    dataset = {'contract': target, 'cutoff_ns': audit.UPPER, 'model_input': False,
               'arrays': inputs['target_arrays'], 'TRAIN_issue_ns_sha256': issue_sha,
               'target_observation_certificate_ref': certificate, 'array_headers': headers,
               **{key: source[key] for key in ('base_dataset', 'base_targets', 'native_manifest')}}
    design = {'schema_version': 'gx1_provider_hlc_conditional_batch_risk_design_v1',
              'execution_authorized': False, 'training_authorized': False,
              'test_authorized': False, 'no_more_epochs': True, 'model_parameters_changed': False,
              'execution': copy.deepcopy(audit.EXECUTION), 'basename': audit.HLC_BATCH_NAME,
              'stage': audit.HLC_BATCH_STAGE, 'plan_path': str(audit.HLC_BATCH_PLAN),
              'output_root': str(audit.HLC_BATCH_ROOT), 'inputs': inputs, 'budget': budget,
              'question': 'synthetic conditional algebra', 'method': [], 'limits': [],
              'new_conditional_scope_preconditions': {'causal_source_availability_claim': False},
              'expected_global_decomposition_from_existing_JSON': {}}
    values = {inputs[key]['path']: {} for key in inputs}
    values.update({audit.HLC_BATCH_DESIGN['path']: design, inputs['completed_model_result']['path']: result,
                   inputs['completed_model_terminal']['path']: terminal,
                   inputs['completed_model_plan']['path']: completed,
                   inputs['original_epoch_plan']['path']: original, inputs['dataset']['path']: dataset,
                   inputs['prior']['path']: prior, inputs['full_quality']['path']: quality,
                   inputs['initial_result']['path']: {'report': {'initial': {
                       'truth_sha256': truth_sha, 'initial_prior_exact': True}}}})
    monkeypatch.setattr(audit, 'read', lambda binding: values[binding['path']])
    monkeypatch.setattr(audit, 'provider_hlc_batch_reference', lambda binding: Path(binding['path']))
    monkeypatch.setattr(provider, 'SOURCE_BINDINGS', source)
    monkeypatch.setattr(provider, 'require_provider_hlc_state_contract', lambda value: value)
    monkeypatch.setattr(provider, 'require_provider_hlc_certificate_ref', lambda value: value)
    monkeypatch.setattr(training, 'provider_hlc_truth_fields', lambda target: fields)
    monkeypatch.setattr(audit.objective, 'require_native_future_state_train_prior', lambda value, **k: value)
    monkeypatch.setattr(audit.saved.native, 'sha', lambda path: 'a' * 64)
    monkeypatch.setattr(audit, 'provider_hlc_batch_source_bridge', lambda: {'synthetic_old_bytes_unchanged': True})
    checked = []; monkeypatch.setattr(audit, 'provider_hlc_batch_checked_inputs', checked.append)
    return design, original, dataset, prior, quality, result, terminal, completed, checked


def test_provider_hlc_builder_binds_v2_certificate_full_issue_schedule_and_explicit_decode_budget(monkeypatch, tmp_path):
    design, original, dataset, prior, _, _, _, _, checked = provider_hlc_builder_fixture(monkeypatch, tmp_path)
    plan = audit.provider_hlc_batch_build_plan(source_base_commit='1' * 40)
    assert checked == [plan]
    assert plan['schedule'] == original['schedule']
    assert plan['budget']['NPZ_ndarray_bytes_decoded'] == 67719050
    assert plan['budget']['provider_candle_records'] == 1290
    assert plan['budget']['MBA_OHLC_price_scalars_parsed'] == 15480
    assert plan['budget']['cached_provider_response_bytes'] == 614909
    assert plan['prior_contract_sha256'] == prior['contract_sha256']
    assert set(plan['target_array_headers']) == set(design['budget']['target_fields'])
    assert list(plan['target_array_headers']) != design['budget']['target_fields']
    assert 'source_rows' not in plan['capture_array_headers']
    assert 'known_activity_by_batch' not in plan['schedule']
    assert plan['target_contract'] == dataset['contract']
    assert all(plan['budget'][key] == 0 for key in ('models', 'NN_forwards', 'checkpoint_loads',
               'autograd_calls', 'backward_calls', 'parameter_gradient_calculations', 'optimizer_updates', 'fits'))


@pytest.mark.parametrize('bad', ['terminal', 'prior_truth', 'quality_output', 'dataset_contract',
    'schedule_flag', 'target_fields_duplicate', 'capture_fields', 'budget_bytes',
    'provider_count', 'no_more_epochs', 'numeric_owner'])
def test_provider_hlc_builder_rejects_mismatched_frozen_identity_or_budget(monkeypatch, tmp_path, bad):
    d, original, dataset, prior, quality, _, terminal, completed, _ = provider_hlc_builder_fixture(monkeypatch, tmp_path)
    if bad == 'terminal': terminal['status'] = 'FAILED'
    elif bad == 'prior_truth': prior['fit_population']['truth_sha256'] = '0' * 64
    elif bad == 'quality_output': quality['output_sha256'] = {'side_logits': '0' * 64}
    elif bad == 'dataset_contract': dataset['contract'] = {'foreign': True}
    elif bad == 'schedule_flag': original['schedule']['update_flags'][0] = False
    elif bad == 'target_fields_duplicate': d['budget']['target_fields'][0] = d['budget']['target_fields'][1]
    elif bad == 'capture_fields': d['budget']['capture_fields'].append('source_rows')
    elif bad == 'budget_bytes': d['budget']['NPZ_ndarray_bytes_decoded'] += 1
    elif bad == 'provider_count': d['budget']['provider_candle_records'] = 1289
    elif bad == 'no_more_epochs': d['no_more_epochs'] = False
    else: completed['source_owners']['synthetic_numeric.py'] = '0' * 64
    with pytest.raises(ValueError): audit.provider_hlc_batch_build_plan(source_base_commit='1' * 40)


@pytest.mark.parametrize('invalid_json_role', [False, True])
def test_provider_hlc_builder_hashes_real_markdown_goal_and_keeps_JSON_roles_strict(monkeypatch, tmp_path, invalid_json_role):
    from pathlib import Path
    d, _, _, _, _, _, _, _, _ = provider_hlc_builder_fixture(monkeypatch, tmp_path)
    goal = tmp_path / 'ORIGINAL_PROJECT_GOAL.md'
    goal.write_text('# Original project goal\n\nPreserve the genuine five-timeframe research goal.\n')
    # The synthetic fixture stubs file SHA to a*64; the actual reference owner is unchanged.
    goal_ref = {'path': str(goal), 'sha256': 'a' * 64}
    d['inputs']['main_goal'] = goal_ref
    json_role = d['inputs']['qualification_requirements']
    malformed = tmp_path / 'malformed_JSON_role.json'; malformed.write_text('# This is not JSON\n')
    previous_read = audit.read; previous_reference = audit.provider_hlc_batch_reference
    parsed, authenticated = [], []
    def read(binding):
        parsed.append(binding['path'])
        if binding['path'] == str(goal): return json.loads(goal.read_text())
        if invalid_json_role and binding == json_role: return json.loads(malformed.read_text())
        return previous_read(binding)
    def reference(binding):
        authenticated.append(binding)
        if binding == goal_ref:
            assert Path(binding['path']).suffix == '.md' and goal.read_text().startswith('# Original project goal')
        return previous_reference(binding)
    monkeypatch.setattr(audit, 'read', read)
    monkeypatch.setattr(audit, 'provider_hlc_batch_reference', reference)
    if invalid_json_role:
        with pytest.raises(json.JSONDecodeError): audit.provider_hlc_batch_build_plan(source_base_commit='1' * 40)
    else:
        plan = audit.provider_hlc_batch_build_plan(source_base_commit='1' * 40)
        assert plan['inputs']['main_goal'] == goal_ref
    assert goal_ref in authenticated and str(goal) not in parsed
    assert json_role['path'] in parsed


def test_provider_hlc_reference_real_markdown_requires_exact_immutable_bytes(tmp_path):
    goal = tmp_path / 'ORIGINAL_PROJECT_GOAL.md'; goal.write_text('# Original goal\n')
    binding = audit.ref(goal)
    assert audit.provider_hlc_batch_reference(binding) == goal
    goal.write_text('# Changed goal\n')
    with pytest.raises(ValueError, match='HLC_BATCH_CANONICAL_REFERENCE'):
        audit.provider_hlc_batch_reference(binding)
