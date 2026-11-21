"""Synthetic source-clock, paired-loop and restart-compatible strict restore tests."""
import copy,json,random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gx1.audit import native_future_state_training_v1 as owner


def reference(name='plan.json'):return {'path':'/synthetic/'+name,'sha256':'a'*64}


def runtime(boot=701):
    from tests.test_native_future_state_cuda_qualification_v1 import runtime as make
    x=make();x['physical_windows_boot_id']=boot;x['wsl_boot_id']='boot'+str(boot);return x


def context_fixture(monkeypatch):
    from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior
    target=fully_future_state_contract();tick=300000000000;base=int(np.datetime64('2021-01-04T00:05','ns').astype(np.int64))
    issues=np.concatenate([base+d*owner.DAY+np.arange(20,dtype=np.int64)*tick for d in range(8)])
    n=len(issues);source=np.arange(n,dtype=np.int64);monkeypatch.setattr(owner,'TRAIN_ROWS',n);monkeypatch.setattr(owner,'TRAIN_SHA',owner.array_sha(issues))
    duration=np.array([target['tf_duration_ns'][t]for t in owner.TF],np.int64);origin=np.array([target['tf_origin_ns'][t]for t in owner.TF],np.int64)
    starts=((issues[:,None]-origin+duration-1)//duration)*duration+origin;complete=np.ones((n,5),np.bool_)
    active=np.broadcast_to((source%3!=0)[:,None],complete.shape).copy();up=active&np.broadcast_to((source%2==0)[:,None],complete.shape)
    truth={'issue_ns':issues,'target_start_ns':starts,'validity_ns':starts+duration,'target_present':complete.copy(),
        'mature':complete.copy(),'nominal_complete':complete.copy(),'state_window_complete':complete.copy(),
        'partial_range_active':np.zeros_like(complete),'activity_known':complete.copy(),'activity':active,
        'side_known':active.copy(),'side_up':up}
    prior=fit_native_future_state_train_prior(truth=truth,fold_upper_ns=owner.UPPER,target_contract=target)
    pop={'issue_rows':n,'selected_issue_ns_sha256':owner.TRAIN_SHA,'selected_source_rows_sha256':owner.array_sha(source),'cutoff_ns':owner.UPPER}
    samples=np.linspace(-1.,1.,n*5*2,dtype=np.float32).reshape(n,5,2)
    def gather(rows,*,max_rows):
        assert 0<len(rows)<=max_rows
        x=samples[rows].copy();out={'seq_x':x,'snap_x':x[:,-1].copy(),'ctx_cont':np.zeros((len(rows),2),np.float32),'ctx_cat':np.zeros((len(rows),1),np.int64)}
        out.update({'seq_'+t.lower():x[:,:4].copy()for t in owner.TF});return out
    provider=SimpleNamespace(issue_ns=issues,source_row=source,input_ready=np.ones(n,np.bool_),gather=gather)
    return {'provider':provider,'rows':source,'truth':truth,'prior':prior,'normalization':{'fit_population':pop}},target


def prepared(monkeypatch):
    context,target=context_fixture(monkeypatch);clock=owner.select_study_clock_rows({'provider':context['provider'],'rows':context['rows']})
    return context,target,clock,owner.prepare_study_cohort_from_context(context,clock_selection=clock,
        frozen_selection_ref=reference('selection.json'),target_contract=target)


def test_selector_is_clock_only_exact16_endpoints_and_sparse_day_has_no_topup(monkeypatch):
    context,target=context_fixture(monkeypatch)
    class ClockOnly(dict):
        def __getitem__(self,key):
            assert key in ('provider','rows'),'target/prior lookup before frozen clock'
            return super().__getitem__(key)
    clock=owner.select_study_clock_rows(ClockOnly(provider=context['provider'],rows=context['rows']))
    assert clock['manifest']['rows']==128 and clock['manifest']['selected_per_bucket']==[16]*8
    for d in range(8):
        ranks=clock['selected_positions'][d*16:(d+1)*16]
        assert np.array_equal(ranks,d*20+np.linspace(0,19,16,dtype=np.int64))
    assert not clock['selected_positions'].flags.writeable
    assert clock['manifest']['target_filter']is False and clock['manifest']['top_up']is False


def test_prepare_keeps_factory_prior_and_fits_only_one_selected_scoring_reference(monkeypatch):
    context,target,clock,out=prepared(monkeypatch);before=copy.deepcopy(context['prior'])
    assert out['factory_prior']==before and out['factory_prior']['fit_population']['issue_rows']==160
    assert out['matched_scoring_prior']['fit_population']['issue_rows']==128
    assert out['matched_scoring_prior']['contract_sha256']!=before['contract_sha256']
    assert out['matched_scoring_prior_fits']==1 and out['factory_prior_fits']==0
    assert set(out['batch_geometries'])=={'64','96','128'}
    assert out['batch_geometries']['64']['first_two_supported']is True
    assert out['whole_TRAIN_batch_geometries']['64']['schedule']['passes']==1
    assert all(not a.flags.writeable for a in out['truth'].values())
    assert set(out['support'])==set(owner.TF)


def test_changed_frozen_clock_is_rejected_before_truth_lookup(monkeypatch):
    context,target,clock,out=prepared(monkeypatch);bad=copy.deepcopy(clock);bad['manifest']['rows']+=1
    context['truth']=None
    with pytest.raises(ValueError,match='FROZEN_CLOCK_SELECTION'):
        owner.prepare_study_cohort_from_context(context,clock_selection=bad,frozen_selection_ref=reference(),target_contract=target)


def test_tie_aware_auc_and_one_class_undefined_without_nonconstant_credit():
    y=np.array([False,True,False,True]);x=np.ones(4,np.float32)
    result=owner.binary_rank_metrics(x,y);assert result['AUC']==.5 and not result['nonconstant']
    assert type(result['AUC'])is float  # weights_only-safe numeric metadata boundary.
    assert owner.binary_rank_metrics(np.array([0,1,0,1],np.float32),y)['AUC']==1.
    one=owner.binary_rank_metrics(x,np.ones(4,np.bool_));assert one['AUC']is None and one['AUC_undefined_reason']


def test_schedule_separates_cursor_updates_skips_and_fixed_probe_panels():
    source=np.arange(260,dtype=np.int64);issues=source*300000000000
    schedule=owner.make_schedule(source,issues,batch_rows=64,passes=2,known_activity_by_batch=[100,0,50,70,0])
    assert schedule['logical_batches']==10 and schedule['optimizer_updates']==6 and schedule['skipped_batches']==4
    assert owner.position(schedule,4)['optimizer_updates']==3
    panels=owner.diagnostic_panels(schedule);assert panels['candidate_batch_indices']==[0,1,3]
    assert panels['skipped_empty_batch_indices']==[1]and panels['supported_panels']==2
    limits=owner.limits_for(schedule,256)
    assert limits['model_forwards']==10+3*2+2 and limits['autograd_calls']==6 and limits['diagnostic_skips']==1
    assert owner.diagnostic_panels(schedule,main=True)['candidate_batch_indices']==[]
    with pytest.raises(ValueError):owner.position(schedule,True)


def test_main_checkpoint_cadence_counts_actual_updates_not_logical_unknown_tail():
    schedule={'rows':257*64,'batch_rows':64,'batches_per_pass':257,'logical_batches':257,
        'optimizer_updates':256,'known_activity_by_batch':[1]*256+[0]}
    limit=owner.limits_for(schedule,256,main=True)
    assert limit['checkpoint_writes']==2 and limit['backward_calls']==256 and limit['skipped_batches']==1


@pytest.mark.parametrize('main',[False,True])
@pytest.mark.parametrize('admitted',[False,True])
def test_exact_main_vs_paired_flags_and_separate_gate(main,admitted):
    flags={k:False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')}
    if admitted:
        flags['training_enabled']=flags['native_training_enabled']=True
        if main:flags['native_launch_authorized']=flags['full_epoch_training_allowed']=True
    owner.validate_training_flags(flags,main=main,admitted=admitted)
    if main and admitted:
        private={**flags,'native_launch_authorized':False,'full_epoch_training_allowed':False}
        with pytest.raises(ValueError):owner.validate_training_flags(private,main=True,admitted=True)
    gate=owner.gate_expected(reference(),main=main,admitted=admitted)
    assert ('main'in gate['lanes'])is main
    assert gate['status']==(('ADMITTED_EXACT_WHOLE_TRAIN_ONE_EPOCH'if main else'ADMITTED_EXACT_PAIRED_LEARNING')if admitted else'PREPARED_NOT_ADMITTED')


class TinyBody(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(2,2);self.dropout=torch.nn.Dropout(.05)
    def forward(self,**features):return {'entry_q_timeframe_hidden':self.dropout(self.layer(features['seq_x']))}


class TinyModel(torch.nn.Module):
    def __init__(self,prior,target):
        super().__init__()
        from gx1.contracts.market_state_ta_integrated_model_v1 import NativeFutureStateReadout
        self.backbone=TinyBody();self.readout=NativeFutureStateReadout(d_model=2,target_contract=target,train_prior=prior)
    def forward(self,**features):return self.readout(self.backbone(**features)['entry_q_timeframe_hidden'])


def fresh(prior,target):
    torch.manual_seed(20260911);np.random.seed(20260911);random.seed(20260911)
    model=TinyModel(prior,target);options={**owner.core.ADAMW_RECIPE,'betas':tuple(owner.core.ADAMW_RECIPE['betas'])}
    return model,torch.optim.AdamW(model.parameters(),**options)


def fake_rng(monkeypatch):
    from tests.test_native_future_state_cuda_qualification_v1 import fake_cuda_rng
    return fake_cuda_rng(monkeypatch)


def tiny_state(monkeypatch):
    context,target,clock,out=prepared(monkeypatch);fake_rng(monkeypatch)
    model,opt=fresh(context['prior'],target);schedule=out['batch_geometries']['64']['schedule']
    plan={'metadata':{'target_contract':target},'recipe':{'optimizer':owner.core.ADAMW_RECIPE,'gradient_clip_norm':1.},'schedule':schedule}
    current={'provider':context['provider'],'source_rows':out['selected_source_rows'],'truth':out['truth']}
    take=np.arange(64,dtype=np.int64);counts=owner.limits_for(schedule,64);counts=dict.fromkeys(counts,0)
    step=owner.training_step(model,opt,owner.packet(current,take),plan,counts,0)
    trajectory=owner.seal({'schedule':schedule,'runtime_compatibility':{k:runtime()[k]for k in ('library_versions','gpu')},
        'model':'fixed','normalization':'fixednorm','prior':'fixedprior','backbone_gradient_policy':None,'epoch_count':2})
    cp=owner.make_checkpoint(model,opt,trajectory=trajectory,original_plan_ref=reference('original.json'),lane='joint',
        cursor=1,steps=[step],scores={},producer_runtime=runtime())
    stream=__import__('io').BytesIO();torch.save(cp,stream)
    cp=torch.load(__import__('io').BytesIO(stream.getvalue()),map_location='cpu',weights_only=True)
    return context,target,out,model,opt,trajectory,cp


def test_generic_restore_allows_new_boot_with_same_immutable_function_and_original_plan(monkeypatch):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    second,second_opt=fresh(context['prior'],target)
    owner.strict_restore(second,second_opt,cp,trajectory=trajectory,original_plan_ref=reference('original.json'),
        expected_cursor=1,current_runtime=runtime(702),expected_state_proof=owner.checkpoint_proof(cp))
    assert owner.core.state_sha(second.state_dict())==owner.core.state_sha(cp['model_state'])
    assert owner.core.state_sha(second_opt.state_dict())==owner.core.state_sha(cp['optimizer_state'])


@pytest.mark.parametrize('bad',['model','normalization','prior','policy','order','epoch','optimizer','cursor','rng','stepdtype','library'])
def test_generic_restore_rejects_wrong_trajectory_or_state_before_parameter_copy(monkeypatch,bad):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    expected=copy.deepcopy(trajectory);live=runtime(702);proof=owner.checkpoint_proof(cp)
    if bad in ('model','normalization','prior'):expected[bad]='changed'
    elif bad=='policy':expected['backbone_gradient_policy']='M5_primary'
    elif bad=='order':expected['schedule']['batch_source_rows_sha256'][0]='0'*64
    elif bad=='epoch':expected['epoch_count']=1
    elif bad=='optimizer':cp['optimizer_state']['param_groups'][0]['lr']*=2
    elif bad=='cursor':cp['position']['logical_cursor']=True
    elif bad=='rng':cp['rng_state']['torch_cuda']=[torch.ones(4)]
    elif bad=='stepdtype':next(iter(cp['optimizer_state']['state'].values()))['step']=torch.tensor(1,dtype=torch.int64)
    elif bad=='library':live['library_versions']['torch']='changed'
    monkeypatch.setattr(model,'load_state_dict',lambda *a,**k:pytest.fail('parameter copy before complete validation'))
    with pytest.raises(ValueError):owner.strict_restore(model,opt,cp,trajectory=expected,original_plan_ref=reference('original.json'),expected_cursor=1,current_runtime=live,expected_state_proof=proof)


def test_original_checkpoint_plan_is_not_replaced_with_new_execution_plan(monkeypatch):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    with pytest.raises(ValueError,match='TRAJECTORY_CURSOR'):
        owner.strict_restore(model,opt,cp,trajectory=trajectory,original_plan_ref=reference('new_execution.json'),expected_cursor=1,current_runtime=runtime(702),expected_state_proof=owner.checkpoint_proof(cp))


def test_main_is_rejected_by_closed_specific_policy_before_execution_guard(monkeypatch):
    def closed(*a,**k):raise ValueError('EXACT_MODE_TRAINING_AUTHORITY')
    monkeypatch.setattr(owner,'inspect',closed)
    import gx1.contracts.gx1_capped_execution_v1 as guard
    monkeypatch.setattr(guard,'require_native_future_state_training_execution',lambda **k:pytest.fail('guard reached under closed policy'))
    with pytest.raises(ValueError,match='EXACT_MODE_TRAINING_AUTHORITY'):owner.run('a'*64,'main')


def test_supported_main_admission_reaches_guard_without_private_paired_authority(monkeypatch,tmp_path):
    live=runtime();plan={'execution':{},'stage':'synthetic'}
    monkeypatch.setattr(owner,'inspect',lambda *a,**k:(plan,reference(),reference('policy.json'),live))
    import gx1.contracts.gx1_capped_execution_v1 as guard
    def reached(**kwargs):raise RuntimeError('explicit-main-guard-reached')
    monkeypatch.setattr(guard,'require_native_future_state_training_execution',reached)
    with pytest.raises(RuntimeError,match='explicit-main-guard-reached'):owner.run('a'*64,'main')



def test_matched_zero_support_side_stays_unfitted_without_half_probability_credit(monkeypatch):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior
    context,target=context_fixture(monkeypatch);clock=owner.select_study_clock_rows(context)
    selected=clock['selected_positions'];context['truth']['side_known'][selected,4]=False;context['truth']['side_up'][selected,4]=False
    context['prior']=fit_native_future_state_train_prior(truth=context['truth'],fold_upper_ns=owner.UPPER,target_contract=target)
    out=owner.prepare_study_cohort_from_context(context,clock_selection=clock,frozen_selection_ref=reference(),target_contract=target)
    assert context['prior']['per_task']['side']['D1']['probabilities']is not None
    cell=out['matched_scoring_prior']['per_task']['side']['D1']
    assert cell['fit_status']=='UNFITTED'and cell['count']==0
    assert cell['probabilities']==[0.5,0.5]  # Canonical pseudocount storage, never a fitted estimate.
    scores=out['constant_reference_scores']['matched_selected_SCORING_ONLY']
    assert 'side.D1'in scores['UNFITTED_tasks']
    assert scores['per_tf']['D1']['side']['count']==0 and scores['per_tf']['D1']['side']['brier_sum']is None
    assert scores['per_tf']['D1']['three_class']['brier_sum']is None
    assert scores['per_tf']['M5']['side']['log_loss_sum']is not None
    assert scores['per_tf']['D1']['activity']['log_loss_sum']is not None
    ranks=owner.constant_rank_metrics(out['matched_scoring_prior'],out['truth'],target)
    assert ranks['D1']['side']['AUC']is None and ranks['D1']['side']['nonconstant']is None
    assert ranks['D1']['side']['undefined_reason']=='UNFITTED_PRIOR_NOT_ESTIMATED'



def test_negative_second_moment_rejected_even_with_matching_receipt_proof(monkeypatch):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    next(iter(cp['optimizer_state']['state'].values()))['exp_avg_sq'].fill_(-.01)
    monkeypatch.setattr(model,'load_state_dict',lambda *a,**k:pytest.fail('negative moment reached copy'))
    with pytest.raises(ValueError,match='NONNEGATIVE_SECOND_MOMENT'):
        owner.strict_restore(model,opt,cp,trajectory=trajectory,original_plan_ref=reference('original.json'),expected_cursor=1,
            current_runtime=runtime(702),expected_state_proof=owner.checkpoint_proof(cp))


@pytest.mark.parametrize('bad',['delete_used_backbone_slot','stale_backbone_step'])
def test_authentic_receipt_binds_observed_backbone_slot_inventory_and_steps(monkeypatch,bad):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    # A receipt authenticates the original full optimizer state, including used
    # backbone slots; no assumption is made that every parameter was used.
    proof=owner.checkpoint_proof(cp);index=cp['optimizer_state']['param_groups'][0]['params'][0]
    if bad=='delete_used_backbone_slot':del cp['optimizer_state']['state'][index]
    else:cp['optimizer_state']['state'][index]['step'].fill_(0.)
    monkeypatch.setattr(model,'load_state_dict',lambda *a,**k:pytest.fail('changed observed slot reached copy'))
    with pytest.raises(ValueError,match='ORIGINAL_CHECKPOINT_STATE_PROOF_BEFORE_COPY'):
        owner.strict_restore(model,opt,cp,trajectory=trajectory,original_plan_ref=reference('original.json'),expected_cursor=1,
            current_runtime=runtime(702),expected_state_proof=proof)



def test_two_pass_leaf_counts_and_own_cursor1_to2_replay_are_exact(monkeypatch,tmp_path):
    context,target,clock,out=prepared(monkeypatch);fake_rng(monkeypatch)
    schedule=out['batch_geometries']['64']['schedule'];source=out['selected_source_rows']
    data={'provider':context['provider'],'source_rows':source,'truth':out['truth'],'prior':context['prior'],
        'scoring_prior':out['matched_scoring_prior'],'batches':owner.core.all_score_batches(np.arange(len(source),dtype=np.int64),64)}
    plan={'mode':'paired_learning','metadata':{'target_contract':target},'schedule':schedule,
        'recipe':{'optimizer':owner.core.ADAMW_RECIPE,'gradient_clip_norm':1.,'train_batch_rows':64,'eval_batch_rows':64,'passes':2},
        'diagnostic_panels':owner.diagnostic_panels(schedule)}
    trajectory=owner.seal({'schedule':schedule,'runtime_compatibility':{k:runtime()[k]for k in ('library_versions','gpu')}})
    monkeypatch.setattr(owner,'trajectory_bindings',lambda *a:trajectory)
    monkeypatch.setattr(owner,'PAIR_ROOT',tmp_path/'paired')
    def initial(model,plan):return {'state':owner.core.state_sha(model.state_dict()),'rng':owner.core.state_sha(owner.core.training_rng())}
    monkeypatch.setattr(owner,'initial_identity',initial)
    pr=reference();live=runtime();root=owner.lane_root('joint');root.mkdir(parents=True)
    model,opt=fresh(context['prior'],target);limit=owner.limits_for(schedule,64);counts=dict.fromkeys(limit,0)
    counts['model_constructions']=counts['optimizer_constructions']=1
    def count(key):
        def hook(*a):counts[key]+=1
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    try:report=owner.run_training_loop(model,opt,data,plan,pr,root,'joint',live,counts)
    finally:
        for hook in hooks:hook.remove()
    assert counts==limit and set(report['scores'])=={'INITIAL','PASS1','PASS2'}
    assert report['logical_batches']==4 and report['actual_optimizer_updates']==4
    assert report['post_final_gradient_balance'][0]['report']['role'].startswith('post_final_two_pass')
    result=owner.publish(root/'RESULT.json',{'plan':pr,'report':report})
    owner.publish(root/'TERMINAL.json',{'result':result})
    second,second_opt=fresh(context['prior'],target);replayroot=owner.lane_root('joint_replay');replayroot.mkdir()
    replaylimit=owner.limits_for(schedule,64,replay=True);replaycounts=dict.fromkeys(replaylimit,0)
    replaycounts['model_constructions']=replaycounts['optimizer_constructions']=1
    def recount(key):
        def hook(*a):replaycounts[key]+=1
        return hook
    hooks=[second.register_forward_pre_hook(recount('model_forwards')),second.backbone.register_forward_pre_hook(recount('backbone_forwards'))]
    try:replayed=owner.run_training_loop(second,second_opt,data,plan,pr,replayroot,'joint_replay',live,replaycounts)
    finally:
        for hook in hooks:hook.remove()
    assert replayed['cold_replay_bit_exact']is True and replaycounts==replaylimit
    receipt=owner.read(report['final_checkpoint_receipt'])
    assert receipt['logical_cursor']==4 and receipt['optimizer_updates']==4 and receipt['original_plan']==pr


@pytest.mark.parametrize('kind',['recovery_probe','main_resume'])
def test_fixed_recovery_lanes_have_separate_roots_gates_and_walls(kind):
    gate=owner.gate_expected(reference(),kind=kind,admitted=True)
    assert set(gate['lanes'])=={kind}
    assert owner.lane_root(kind).name.startswith('NATIVE_FUTURE_STATE_')
    assert owner.execution(kind)['outer_seconds']==(1800 if kind=='recovery_probe'else 12000)



def test_checkpoint_receipt_is_durable_before_final_and_binds_full_original_state(monkeypatch,tmp_path):
    context,target,out,model,opt,trajectory,cp=tiny_state(monkeypatch)
    counts={'checkpoint_writes':0,'checkpoint_readbacks':0,'checkpoint_receipt_writes':0}
    value=owner.publish_checkpoint_receipt(tmp_path/'UPDATE000001.pt',cp,counts,execution_ref=reference('new_execution.json'))
    receipt=owner.read(value)
    assert counts=={'checkpoint_writes':1,'checkpoint_readbacks':1,'checkpoint_receipt_writes':1}
    assert receipt['checkpoint']==owner.ref(tmp_path/'UPDATE000001.pt')
    assert receipt['logical_cursor']==1 and receipt['optimizer_updates']==1
    assert receipt['original_plan']==reference('original.json')and receipt['execution_plan']==reference('new_execution.json')
    assert receipt['proof']==owner.checkpoint_proof(cp)
    assert receipt['producer_runtime']==cp['producer']['runtime']


def test_recovery_runtime_compatibility_allows_new_execution_boot_not_changed_cuda_libraries():
    original={k:runtime()[k]for k in ('library_versions','gpu')}
    assert owner.runtime_compatible(original,runtime(999))
    bad=runtime(999);bad['library_versions']['torch_cuda']='changed'
    with pytest.raises(ValueError,match='RUNTIME_COMPATIBILITY'):owner.runtime_compatible(original,bad)



def recovery_metadata_fixture(monkeypatch,*,main_resume=False,live_boot=702):
    arm='joint';plan_path=owner.MAIN_PLAN if main_resume else owner.PAIR_PLAN
    receipt_path=(owner.MAIN_ROOT/'INITIAL.json')if main_resume else(owner.PAIR_ROOT/arm/'CURSOR001.json')
    inputs={'original_plan':{'path':str(plan_path),'sha256':'a'*64},
        'checkpoint_receipt':{'path':str(receipt_path),'sha256':'b'*64},
        'runtime_proof':{'path':str(owner.ROOT/'RUNTIME_PROOF003.json'),'sha256':'c'*64}}
    if not main_resume:inputs['original_cursor2_proof']={'path':str(owner.PAIR_ROOT/arm/'CURSOR002_PROOF.json'),'sha256':'d'*64}
    schedule=owner.make_schedule(np.arange(256,dtype=np.int64),np.arange(256,dtype=np.int64)*300000000000,
        batch_rows=64,passes=1 if main_resume else 2,known_activity_by_batch=[10]*4)
    historical=runtime();live=runtime(live_boot)
    trajectory=owner.seal({'schedule':schedule,'runtime_compatibility':{k:historical[k]for k in ('library_versions','gpu')}})
    original={'mode':'whole_train_one_epoch'if main_resume else'paired_learning','backbone_gradient_policy':None,
        'metadata':{'runtime':historical},'schedule':schedule,'training_leaf_AST':{},'source_owners':{},
        'inputs':{},'recipe':{'eval_batch_rows':64}}
    proof={k:'e'*64 for k in ('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state',
        'training','optimizer_parameter_names','steps','scores')}
    cursor=0 if main_resume else 1
    proof['trajectory']=owner.core.state_sha(trajectory);proof['position']=owner.core.state_sha(owner.position(schedule,cursor))
    receipt={'schema_version':'gx1_native_future_state_training_checkpoint_receipt_v1',
        'checkpoint':{'path':str(receipt_path.with_suffix('.pt')),'sha256':'f'*64},
        'original_plan':inputs['original_plan'],'execution_plan':inputs['original_plan'],
        'producer_lane':'main'if main_resume else arm,'producer_runtime':historical,
        'logical_cursor':cursor,'optimizer_updates':cursor,'trajectory_sha256':trajectory['contract_sha256'],'proof':proof}
    expected={'proof':{**proof,'position':owner.core.state_sha(owner.position(schedule,2))}}
    values={str(plan_path):original,str(receipt_path):receipt,inputs['runtime_proof']['path']:live}
    if not main_resume:values[inputs['original_cursor2_proof']['path']]=expected
    def metadata_read(binding):
        assert binding['path'].endswith('.json'),'payload opened during metadata preflight'
        return values[binding['path']]
    monkeypatch.setattr(owner,'read',metadata_read)
    monkeypatch.setattr(owner,'trajectory_bindings',lambda *a:trajectory)
    monkeypatch.setattr(owner,'own_numeric_ast',lambda:{})
    monkeypatch.setattr(owner,'require_source_inventory',lambda *a:None)
    monkeypatch.setattr(owner,'expected_source_names',lambda:[])
    monkeypatch.setattr(owner,'verify_initial_source_bridge',lambda *a:None)
    for name in ('sha','ref','recovery_physical'):
        monkeypatch.setattr(owner,name,lambda *a,**k:pytest.fail('physical/hash I/O before metadata rejection'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint opened during metadata preflight'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('arrays opened during metadata preflight'))
    return inputs,original,receipt,live


@pytest.mark.parametrize('main_resume',[False,True])
@pytest.mark.parametrize('bad',['original_path','receipt_path','runtime_path','runtime_sha','receipt_ref_extra'])
def test_recovery_input_refs_reject_before_any_hash_or_open(monkeypatch,main_resume,bad):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch,main_resume=main_resume)
    if bad=='original_path':inputs['original_plan']['path']='/wrong/original.json'
    elif bad=='receipt_path':inputs['checkpoint_receipt']['path']='/wrong/CURSOR001.json'
    elif bad=='runtime_path':inputs['runtime_proof']['path']='/wrong/RUNTIME_PROOF003.json'
    elif bad=='runtime_sha':inputs['runtime_proof']['sha256']='A'*64
    else:inputs['checkpoint_receipt']['extra']=True
    monkeypatch.setattr(owner,'read',lambda *a:pytest.fail('invalid incoming reference was read/hashed'))
    with pytest.raises(ValueError,match='RECOVERY_(REFERENCE|RECEIPT|RUNTIME)|FIXED_RECOVERY'):
        owner.build_recovery_plan(inputs=inputs,main_resume=main_resume,source_base_commit=live['source_commit'])


@pytest.mark.parametrize('bad',['checkpoint_sha','checkpoint_path','missing_proof','extra_proof','proof_sha','execution_ref','producer_runtime'])
def test_recovery_receipt_headers_reject_without_payload_io(monkeypatch,bad):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch)
    if bad=='checkpoint_sha':receipt['checkpoint']['sha256']='not-a-sha'
    elif bad=='checkpoint_path':receipt['checkpoint']['path']=str(owner.PAIR_ROOT/'joint'/'OTHER.pt')
    elif bad=='missing_proof':del receipt['proof']['optimizer_state']
    elif bad=='extra_proof':receipt['proof']['extra']='e'*64
    elif bad=='proof_sha':receipt['proof']['model_state']='E'*64
    elif bad=='execution_ref':receipt['execution_plan']={'path':str(owner.PAIR_PLAN),'sha256':'bad'}
    else:receipt['producer_runtime']['physical_windows_boot_id']=True
    with pytest.raises(ValueError,match='RECOVERY_|FUTURE_CUDA_RUNTIME_'):
        owner.build_recovery_probe_plan(inputs=inputs,source_base_commit=live['source_commit'])


@pytest.mark.parametrize('bad',['same_windows','same_wsl','older_windows'])
def test_recovery_probe_requires_physical_and_wsl_new_boot_before_claim(monkeypatch,bad):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch)
    if bad=='same_windows':live['physical_windows_boot_id']=701
    elif bad=='same_wsl':live['wsl_boot_id']=original['metadata']['runtime']['wsl_boot_id']
    else:live['physical_windows_boot_id']=700
    with pytest.raises(ValueError,match='RECOVERY_PROBE_REQUIRES_PHYSICAL_NEW_BOOT'):
        owner.build_recovery_probe_plan(inputs=inputs,source_base_commit=live['source_commit'])


def test_recovery_probe_accepts_authenticated_new_boot_metadata_only(monkeypatch):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch)
    result=owner.build_recovery_probe_plan(inputs=inputs,source_base_commit=live['source_commit'])
    assert result['mode']=='recovery_probe'and result['training_authorized']is False
    assert result['metadata']['receipt']['producer_runtime']['physical_windows_boot_id']==701
    assert result['metadata']['runtime']['physical_windows_boot_id']==702
    assert 'verified_physical_recovery'not in result  # Requires actual full cursor2 numeric proof.


def test_main_resume_allows_same_boot_authenticated_metadata(monkeypatch):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch,main_resume=True,live_boot=701)
    result=owner.build_main_resume_plan(inputs=inputs,source_base_commit=live['source_commit'])
    assert result['mode']=='main_resume'and result['training_authorized']is False



def test_recovery_cursor2_reference_rejects_before_any_hash_or_open(monkeypatch):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch)
    inputs['original_cursor2_proof']['path']=str(owner.PAIR_ROOT/'m5_primary'/'CURSOR002_PROOF.json')
    monkeypatch.setattr(owner,'read',lambda *a:pytest.fail('wrong cursor2 role was read/hashed'))
    with pytest.raises(ValueError,match='RECOVERY_REFERENCE_PATH'):
        owner.build_recovery_probe_plan(inputs=inputs,source_base_commit=live['source_commit'])


def test_recovery_runtime_symlink_rejects_before_any_hash_or_open(monkeypatch):
    inputs,original,receipt,live=recovery_metadata_fixture(monkeypatch)
    monkeypatch.setattr(owner.Path,'is_symlink',lambda path:True)
    monkeypatch.setattr(owner,'read',lambda *a:pytest.fail('symlink runtime was read/hashed'))
    with pytest.raises(ValueError,match='RECOVERY_RUNTIME_SYMLINK'):
        owner.build_recovery_probe_plan(inputs=inputs,source_base_commit=live['source_commit'])
