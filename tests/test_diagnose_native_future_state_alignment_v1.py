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
