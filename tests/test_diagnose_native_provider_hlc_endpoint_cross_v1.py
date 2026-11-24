"""Fixed endpoint algebra and admission boundaries; no genuine payload/native NN."""
import ast
import copy
import inspect as pyinspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from gx1.audit import diagnose_native_provider_hlc_endpoint_cross_v1 as m


def binding(name): return {'path':'/synthetic/'+name+'.json','sha256':'a'*64}


def parameter_fixture():
    p0 = {tf:{'weight':torch.zeros((1,128),dtype=torch.float32),'bias':torch.tensor([j/8],dtype=torch.float32)}for j,tf in enumerate(m.TF)}
    p1 = {tf:{'weight':torch.arange(128,dtype=torch.float32).reshape(1,128)/(128*128)*(j+1),
              'bias':torch.tensor([j/8+1/16],dtype=torch.float32)}for j,tf in enumerate(m.TF)}
    return p0,p1


@pytest.mark.parametrize('rows',[256,246])
def test_original_batch_FP32_TF_views_feed_linear_and_four_arms_preserve_rng(monkeypatch,rows):
    h0 = (torch.arange(rows*5*128,dtype=torch.float32).reshape(rows,5,128)%17)/32
    h1 = h0+1/8; p0,p1 = parameter_fixture(); seen = []; original = torch.nn.functional.linear
    def linear(h,w,b=None):
        seen.append((tuple(h.shape),h.stride(),h.dtype,h.device.type,tuple(w.shape),None if b is None else tuple(b.shape)))
        return original(h,w,b)
    monkeypatch.setattr(torch.nn.functional,'linear',linear)
    rng = torch.get_rng_state().clone(); before = m.core.state_sha((h0,h1,p0,p1))
    result = m.analytic_cross(h0,h1,p0,p1,device='cpu')
    assert len(seen)==30  # 20 crossed side cells and 10 algebraic projection terms; no module forward.
    assert all(shape==(rows,128)and stride==(640,1)and dtype==torch.float32 and device=='cpu'and weight==(1,128)
               and bias in (None,(1,))for shape,stride,dtype,device,weight,bias in seen)
    assert torch.equal(rng,torch.get_rng_state())and before==m.core.state_sha((h0,h1,p0,p1))
    b0 = np.array([p0[tf]['bias'].item()for tf in m.TF],np.float32)
    b1 = np.array([p1[tf]['bias'].item()for tf in m.TF],np.float32)
    assert np.array_equal(result['H0_W0'],np.repeat(b0[None],rows,axis=0))
    assert np.array_equal(result['H1_W0'],result['H0_W0'])
    diagonal,identity = m.cross_parities(result,result['H0_W0'],result['H1_W1'],b0,b1)
    assert all(p['qualified']for p in diagonal.values())and identity['qualified']
    summary = m.numeric_summary(result,h0,h1,p0,p1)
    assert set(summary)==set(m.TF)and all(summary[tf]['rows']==rows for tf in m.TF)
    assert summary['M5']['representation_change_at_final_W_mean_logit']>0
    json.dumps(summary,allow_nan=False)


@pytest.mark.parametrize('bad',['batch','width','dtype','nan','weight','bias','gradient'])
def test_analytic_input_errors_reach_their_intended_gate(bad):
    h0 = torch.zeros((256,5,128)); h1 = h0.clone(); p0,p1 = parameter_fixture()
    gate = 'ANALYTIC_CAPTURE_SHAPES'
    if bad=='batch': h0=h1=h0[:128]
    elif bad=='width': h0=h1=h0[:,:,:127]
    elif bad=='dtype': h0=h0.double()
    elif bad=='nan': h1[0,0,0]=float('nan')
    else:
        gate='ANALYTIC_PARAMETER_BINDING'
        if bad=='weight': p1['M5']['weight']=p1['M5']['weight'][:,:127]
        elif bad=='bias': p1['M5']['bias']=p1['M5']['bias'].double()
        else: p1['M5']['weight'].requires_grad_(True)
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_'+gate+'$'):
        m.analytic_cross(h0,h1,p0,p1,device='cpu')


@pytest.mark.parametrize('task',m.OUTPUTS)
def test_both_saved_final_native_heads_must_pass_unrelaxed_parity(task):
    batches=[np.arange(i*3,(i+1)*3,dtype=np.int64)for i in range(6)]
    saved={k:np.zeros((18,5),np.float32)for k in m.OUTPUTS}
    outputs=[{k:saved[k][b].copy()for k in m.OUTPUTS}for b in batches]
    assert len(m.saved_final_parity(outputs,batches,saved))==6
    outputs[3][task][0,4]=1e-3
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_SAVED_FINAL_NATIVE_PARITY$'):
        m.saved_final_parity(outputs,batches,saved)


@pytest.mark.parametrize('bad,gate',[('initial','CROSS_NATIVE_DIAGONAL_PARITY'),('final','CROSS_NATIVE_DIAGONAL_PARITY'),
                                     ('identity','CROSS_ADDITIVE_IDENTITY'),('zero_W','ZERO_INITIAL_W_BOTH_REPRESENTATIONS')])
def test_cross_failures_cannot_reach_a_numeric_interpretation(bad,gate):
    x={k:np.zeros((246,5),np.float32)for k in ('H0_W0','H0_W1','H1_W0','H1_W1','projection_on_initial','representation_change')}
    a=np.zeros((246,5),np.float32);b=a.copy();bias=np.zeros(5,np.float32)
    if bad=='initial': x['H0_W0'][0,0]=1.
    elif bad=='final': x['H1_W1'][0,0]=1.
    elif bad=='identity': x['representation_change'][0,0]=1.
    else: x['H1_W0'][0,0]=1.
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_'+gate+'$'):
        m.cross_parities(x,a,b,bias,bias)


def test_panel_is_original_six_batches_and_calendar_checked_before_any_model(monkeypatch):
    issues=np.linspace(np.datetime64('2021-01-01','ns').astype('int64'),np.datetime64('2025-12-31','ns').astype('int64'),m.ROWS,dtype=np.int64)
    source=np.arange(m.ROWS,dtype=np.int64)+100
    monkeypatch.setattr(m.train,'TRAIN_SHA',m.core.digest(issues))
    batches,report=m.select_panel(issues,source)
    assert [len(b)for b in batches]==[256]*5+[246]
    assert [x['original_score_batch_index_zero_based']for x in report]==list(m.BATCH_INDICES)
    assert [x['UTC_year']for x in report]==[2021,2022,2023,2024,2025,2025]
    # Authenticated clock can be well ordered and TRAIN-only yet fail this prespecified panel.
    issues=np.linspace(np.datetime64('2021-01-01','ns').astype('int64'),np.datetime64('2022-12-31','ns').astype('int64'),m.ROWS,dtype=np.int64)
    monkeypatch.setattr(m.train,'TRAIN_SHA',m.core.digest(issues))
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_PRESPECIFIED_CALENDAR_COVERAGE$'):
        m.select_panel(issues,source)


def ledger_fixture(monkeypatch):
    arrays={'a':np.arange(6,dtype=np.float32).reshape(2,3),'b':np.array([True,False])}
    headers={k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':m.core.digest(v)}for k,v in arrays.items()}
    inventory={'synthetic':{'binding':{'path':'/synthetic/archive.npz','sha256':'b'*64},'array_headers':headers,
                            'decoded_fields':['a','b'],'decoded_bytes':26}}
    class Archive:
        files=['a','b']
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def __getitem__(self,k): return arrays[k]
    calls=[]
    def load(path,**kwargs): calls.append((path,kwargs));return Archive()
    monkeypatch.setattr(m.np,'load',load)
    return arrays,inventory,calls


def test_runtime_ledger_counts_actual_getitem_arrays_bytes_and_one_open(monkeypatch):
    arrays,inventory,calls=ledger_fixture(monkeypatch); before=m.np.load
    with m.InputLedger(inventory)as ledger:
        with m.np.load('/synthetic/archive.npz',allow_pickle=False)as z:
            assert z.files==['a','b']and z['a']is arrays['a']and z['b']is arrays['b']
        assert ledger.verify()=={'synthetic':{'archive_loads':1,'decoded_fields':['a','b'],'decoded_bytes':26}}
    assert len(calls)==1 and m.np.load is before


@pytest.mark.parametrize('bad,gate',[('path','NPZ_ALLOWED_LOAD'),('pickle','NPZ_ALLOWED_LOAD'),('load2','NPZ_ONE_LOAD_PER_ARCHIVE'),
                                     ('field','NPZ_FIELD_BUDGET'),('duplicate','NPZ_FIELD_BUDGET'),('header','NPZ_DECODED_HEADER'),
                                     ('bytes','NPZ_BYTE_BUDGET'),('missing','EXACT_ACTUAL_INPUT_DECODES')])
def test_runtime_ledger_rejects_intended_budget_gate(monkeypatch,bad,gate):
    arrays,inventory,_=ledger_fixture(monkeypatch)
    if bad=='header': inventory['synthetic']['array_headers']['a']['shape']=[6]
    if bad=='bytes': inventory['synthetic']['decoded_bytes']=1
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_'+gate+'$'):
        with m.InputLedger(inventory)as ledger:
            if bad in ('path','pickle'):
                m.np.load('/wrong.npz'if bad=='path'else'/synthetic/archive.npz',allow_pickle=bad=='pickle')
            else:
                with m.np.load('/synthetic/archive.npz',allow_pickle=False)as z:
                    if bad=='load2': m.np.load('/synthetic/archive.npz',allow_pickle=False)
                    elif bad=='field': z['unknown']
                    else:
                        z['a']
                        if bad=='duplicate': z['a']
                        if bad=='missing': ledger.verify()


class SyntheticModel:
    def __init__(self): self.values={'p':torch.zeros(2),'clock':torch.tensor([7],dtype=torch.int64)};self.installs=0
    def state_dict(self): return self.values
    def named_buffers(self): return [('clock',self.values['clock'])]
    def parameters(self): return [self.values['p']]
    def load_state_dict(self,state,*,strict):
        assert strict is True;self.values={k:v.clone()for k,v in state.items()};self.installs+=1


def test_two_strict_whole_model_installs_preserve_rng_and_buffers(monkeypatch):
    model=SyntheticModel();counts={'model_state_installs':0};rng=torch.get_rng_state().clone()
    monkeypatch.setattr(m.core,'training_rng',lambda:{'cpu':torch.get_rng_state()})
    for value in (1.,2.):
        state={'p':torch.full((2,),value),'clock':torch.tensor([7],dtype=torch.int64)}
        proof=m.strict_model_install(model,state,counts)
        assert proof['strict']and proof['RNG_unchanged']and model.installs==counts['model_state_installs']
    assert counts['model_state_installs']==2 and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('bad,gate',[('keys','STRICT_MODEL_STATE'),('dtype','STRICT_MODEL_STATE'),('nan','STRICT_MODEL_STATE'),
                                     ('buffer','IMMUTABLE_MODEL_BUFFERS'),('budget','MODEL_INSTALL_BUDGET')])
def test_strict_install_never_accepts_partial_or_corrupt_state(monkeypatch,bad,gate):
    model=SyntheticModel();state={'p':torch.ones(2),'clock':torch.tensor([7],dtype=torch.int64)}
    counts={'model_state_installs':2 if bad=='budget'else 0}
    monkeypatch.setattr(m.core,'training_rng',lambda:{'cpu':torch.get_rng_state()})
    if bad=='keys':del state['p']
    elif bad=='dtype':state['p']=state['p'].double()
    elif bad=='nan':state['p'][0]=float('nan')
    elif bad=='buffer':state['clock'][0]=8
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_'+gate+'$'):
        m.strict_model_install(model,state,counts)
    assert model.installs==0


class NativeBufferPattern(torch.nn.Module):
    """Match the qualified backbone's derived, non-persistent projection index."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer('identity',torch.tensor([7],dtype=torch.int64))
        self.backbone = torch.nn.Module()
        self.backbone.register_buffer('entry_signal_projection_idx',torch.tensor([2,0,1],dtype=torch.long),persistent=False)


def test_native_nonpersistent_projection_buffer_survives_exact_persistent_CP_install(monkeypatch):
    model = NativeBufferPattern(); state = {k:v.clone()for k,v in model.state_dict().items()}
    state['weight'].fill_(1.)
    assert set(state)=={'weight','identity'}
    assert set(dict(model.named_buffers()))=={'identity','backbone.entry_signal_projection_idx'}
    before = m.core.state_sha(dict(model.named_buffers()))
    monkeypatch.setattr(m.core,'training_rng',lambda:{'cpu':torch.get_rng_state()})
    counts = {'model_state_installs':0}; proof = m.strict_model_install(model,state,counts)
    assert counts=={'model_state_installs':1}and torch.equal(model.weight,state['weight'])
    assert proof['all_factory_buffers_unchanged']is True
    assert proof['all_factory_buffers_sha256_before_and_after']==before==m.core.state_sha(dict(model.named_buffers()))


@pytest.mark.parametrize('mutation',['value','dtype'])
def test_strict_load_hook_mutation_of_nonpersistent_factory_buffer_is_rejected(monkeypatch,mutation):
    model = NativeBufferPattern(); state = {k:v.clone()for k,v in model.state_dict().items()}
    def mutate(module,_):
        if mutation=='value': module.backbone.entry_signal_projection_idx[0]+=1
        else: module.backbone.entry_signal_projection_idx=module.backbone.entry_signal_projection_idx.float()
    model.register_load_state_dict_post_hook(mutate)
    monkeypatch.setattr(m.core,'training_rng',lambda:{'cpu':torch.get_rng_state()})
    counts = {'model_state_installs':0}
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_FACTORY_BUFFERS_UNCHANGED$'):
        m.strict_model_install(model,state,counts)
    assert counts=={'model_state_installs':1}
    assert m.core.state_sha(model.state_dict())==m.core.state_sha(state)  # state_dict alone would miss this mutation.


def test_nonpersistent_buffer_must_not_be_fabricated_as_a_checkpoint_field(monkeypatch):
    model = NativeBufferPattern(); state = {k:v.clone()for k,v in model.state_dict().items()}
    state['backbone.entry_signal_projection_idx']=model.backbone.entry_signal_projection_idx.clone()
    monkeypatch.setattr(m.core,'training_rng',lambda:{'cpu':torch.get_rng_state()})
    counts = {'model_state_installs':0}
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_STRICT_MODEL_STATE$'):
        m.strict_model_install(model,state,counts)
    assert counts=={'model_state_installs':0}


@pytest.mark.parametrize('action,name',[
    (lambda:torch.optim.AdamW([]),'__init__'),(lambda:torch.autograd.grad(None,None),'grad'),
    (lambda:torch.autograd.backward(None),'backward'),(lambda:torch.tensor(0.).backward(),'backward'),
    (lambda:torch.set_rng_state(torch.get_rng_state()),'set_rng_state'),
    (lambda:torch.cuda.set_rng_state_all([]),'set_rng_state_all'),
    (lambda:m.core.restore_training_rng(None),'restore_training_rng'),(lambda:m.train.strict_restore(None,None,None,None),'strict_restore'),
    (lambda:torch.serialization.add_safe_globals([]),'add_safe_globals')])
def test_runtime_forbids_optimizer_gradients_checkpoint_rng_restore_and_fallback(action,name):
    with m.runtime_blockers()as attempts:
        with pytest.raises(RuntimeError,match='^PROVIDER_HLC_ENDPOINT_CROSS_FORBIDDEN_'+name+'$'):action()
        assert attempts==[name]


@pytest.mark.parametrize('admitted',[False,True])
def test_separate_readonly_oneuse_gate_never_grants_fit_or_test(admitted):
    gate=m.gate_expected(binding('plan'),admitted)
    assert gate['execution_authorized']is admitted and gate['remaining_invocations']==int(admitted)
    assert gate['training_authorized']is gate['test_authorized']is gate['relaunch_allowed']is False


def test_scope_requires_exact_frozen_env_canonical_CLI_and_common_trainer_guard(monkeypatch,tmp_path):
    repo=tmp_path/'repo';(repo/'configs/research').mkdir(parents=True);runs=tmp_path/'runs';runs.mkdir()
    monkeypatch.setattr(m,'REPO',repo);monkeypatch.setattr(m,'PLAN',repo/'configs/research'/m.PLAN.name)
    monkeypatch.setattr(m,'ROOT',runs/m.NAME);monkeypatch.chdir(repo)
    seen=[];monkeypatch.setattr(m,'require_guarded_cuda_trainer_execution',lambda:seen.append(True)or{'actual_guard':True})
    sha='c'*64;env={k:sha if v=='SHA'else v for k,v in m.ENV.items()}
    argv=[str(repo/'.venv/bin/python'),'-m',m.MODULE,'--plan-sha256',sha]
    assert m.require_scope(sha,environ=env,argv=argv)=={'actual_guard':True}and seen==[True]
    assert 'GX1_CPU_AFFINITY'not in env
    for key in m.ENV:
        bad={**env,key:env[key]+'wrong'}
        with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_EXACT_EXECUTION_ENV$'):
            m.require_scope(sha,environ=bad,argv=argv)
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_UNKNOWN_NATIVE_ENV$'):
        m.require_scope(sha,environ={**env,'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_LANE':''},argv=argv)
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_EXACT_PROCESS_ARGV$'):
        m.require_scope(sha,environ=env,argv=argv+['--lane','main'])
    assert seen==[True]


def test_shell_bridge_proves_all_old_bytes_and_rejects_unrelated_delta():
    assert m.shell_bridge()['normalized_sha256']==m.OLD_SHELL_SHA
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_SHELL_OLD_BYTES_UNCHANGED$'):
        m.shell_bridge((m.REPO/'scripts/gx1_capped_run.sh').read_text()+'\n# unrelated delta\n')


def test_preflight_reports_zero_payloads_models_cuda_and_no_claim(monkeypatch,capsys):
    sha='d'*64;calls=[];monkeypatch.setattr(m.sys,'argv',['endpoint','--plan-sha256',sha,'--preflight'])
    monkeypatch.setattr(m,'require_scope',lambda value,**kw:calls.append((value,kw))or{'guard':True})
    monkeypatch.setattr(m,'inspect',lambda value,**kw:({},binding('plan'),binding('policy'),{'cuda_initialized':False}))
    monkeypatch.setattr(m,'run',lambda *_:pytest.fail('preflight cannot call run'))
    m.main();report=json.loads(capsys.readouterr().out)
    assert report['status']=='EXACT_NATIVE_NONE_ENDPOINT_CROSS_PREFLIGHT_PASS'
    assert report['claim_created']is report['CUDA_initialized']is False
    assert all(report[k]==0 for k in ('checkpoint_deserializations','NPZ_arrays_decoded','models','NN_forwards'))
    assert calls==[(sha,{'preflight':True})]


def test_owner_preserves_original_validator_roles_native_route_and_exact_two_decode_loop():
    source=pyinspect.getsource(m.physical);tree=ast.parse(source)
    loads=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.load']
    assert len(loads)==1 and {k.arg:ast.literal_eval(k.value)for k in loads[0].keywords}=={'map_location':'cpu','weights_only':True}
    assert "(('initial','initial_checkpoint'),('final','None_final_checkpoint'))"in source
    assert 'readout.validate_initial(cp,meta[\'initial_receipt\'],meta[\'initial_base_plan\']'in source
    assert "plan['inputs']['initial_base_plan'],prior_bias)"in source
    assert 'readout.validate_final' in source and 'strict_model_install(model' in source
    calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert calls.count('model_owner.build_native_future_state_model')==1
    assert not any('optimizer' in call or 'restore' in call or 'backward' in call for call in calls)
    assert source.index('select_panel(')<source.index('build_native_future_state_model(')
    assert source.index('saved_final_parity(')<source.index('analytic_cross(')<source.index('numeric_summary(')
    assert m.BUDGET['checkpoint_deserializations']==2 and m.BUDGET['model_state_installs']==2
    assert m.BUDGET['NPZ_archive_loads']==3 and m.BUDGET['NPZ_arrays_decoded']==44 and m.BUDGET['NPZ_decoded_bytes']==1500935808
    assert m.EXECUTION['outer_seconds']==900 and m.EXECUTION['internal_seconds']==870


def metadata_fixture(monkeypatch):
    roles=['original_epoch_plan','None_final_plan','None_final_result','None_final_receipt','initial_base_plan','first_fresh_plan',
           'first_fresh_closure','initial_receipt','initial_result','initial_acceptance','full_quality','capture','initial_checkpoint','None_final_checkpoint']
    inputs={k:binding(k)for k in roles};contract={'contract_sha256':'6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0',
        'readout':{'d_model':128},'train_prior_sha256':'p'};target={'contract_sha256':'target'}
    metadata={'TRAIN_rows':m.ROWS,'TRAIN_issue_ns_sha256':'issue','truth_sha256':'truth','target_contract':target,'architecture':{},
              'runtime':{'observed_utc':'base'},'input_binding':{'input_normalization_sha256':'norm'}}
    original={'metadata':metadata,'model_contract':contract,'backbone_gradient_policy':None,'recipe':{'seed':7},
              'schedule':{'issue_ns_sha256':'issue','logical_batches':3694,'batches_per_pass':3694,'update_flags':[True]*3694}}
    base={'metadata':copy.deepcopy(metadata),'model_contract':contract,'recipe':{'seed':7}}
    first={**copy.deepcopy(base),'schema_version':'gx1_native_provider_hlc_all_tf_initial_segment_plan_v1','initial_mode':'first_fresh',
           'base_plan':inputs['initial_base_plan'],'parent_terminal':None,'initial_receipt':None,'prefix':{'cursor':0,'head':None,'rows_done':0}}
    first['metadata']['runtime']['observed_utc']='first'
    trajectory={'contract_sha256':'trajectory','model_contract':contract};proof={k:'a'*64 for k in ('trajectory','position','model_state','optimizer_state',
        'scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores')}
    proof.update(trajectory=m.core.state_sha(trajectory),position=m.core.state_sha(m.train.position(original['schedule'],3694)))
    ir={'plan':inputs['initial_base_plan'],'checkpoint':inputs['initial_checkpoint'],'model_state_sha256':'start','rng_sha256':'rng'}
    fr={'checkpoint':inputs['None_final_checkpoint'],'original_plan':inputs['original_epoch_plan'],'execution_plan':inputs['None_final_plan'],
        'producer_lane':'main','logical_cursor':3694,'optimizer_updates':3694,'trajectory_sha256':'trajectory','proof':proof}
    output={'activity_logits':'a','side_logits':'s'}
    report={'trajectory':trajectory,'actual_optimizer_updates':3694,'logical_batches':3694,'full_TRAIN_rows':m.ROWS,'passes':1,
            'original_plan':inputs['original_epoch_plan'],'final_checkpoint_receipt':inputs['None_final_receipt'],'final_quality':inputs['full_quality'],
            'final_logits':inputs['capture'],'fresh_initial_state_sha256':'start',
            'whole_FINAL_score':{'truth_sha256':'truth','output_sha256':output,'state_sha256_before_and_after':proof['model_state'],
                                 'rng_sha256_before_and_restored':proof['rng_state']}}
    result={'status':'COMPLETE_NONE_FULL_POPULATION_TRAIN_DIAGNOSTIC_EPOCH','report':report,'plan':inputs['None_final_plan'],'claim':binding('claim'),
            'source_inputs_policy_boot_unchanged':True,'source_unchanged':True,'test_data_used':False}
    closure={'schema_version':'gx1_native_provider_hlc_initial_segment_closure_v1','plan':inputs['first_fresh_plan'],
             'base_plan':inputs['initial_base_plan'],'initial_receipt':inputs['initial_receipt'],'status':'CLOSED_SEGMENT_REUSE_PROOF_PASS','errors':[],
             **dict.fromkeys(('M1_firewall_clear','model_RNG_grad_unchanged','physical_boot_unchanged','policy_unchanged','source_inputs_unchanged','wsl_boot_unchanged'),True),
             'model_RNG_proof':{'model_state_sha256':'start','rng_sha256':'rng',**dict.fromkeys(('constructor_training_mode_restored',
                 'fresh_constructor_no_weight_load','gradients_none','model_RNG_grad_unchanged'),True)},
             'counts':{'new_checkpoint_writes':1,'new_checkpoint_readbacks':1,**dict.fromkeys(('existing_checkpoint_loads','model_state_restores','optimizer_constructions','optimizer_steps'),0)}}
    meta={'original_epoch_plan':original,'None_final_plan':{'metadata':{'original':original,'trajectory':trajectory},'model_contract':contract,'backbone_gradient_policy':None},
          'None_final_result':result,'completed_terminal':{'status':'COMPLETE','exit_code':0,'result':inputs['None_final_result'],'plan':inputs['None_final_plan'],'claim':result['claim']},
          'initial_base_plan':base,'first_fresh_plan':first,'first_fresh_closure':closure,'initial_receipt':ir,'None_final_receipt':fr,
          'initial_result':{'plan':inputs['initial_base_plan'],'source_inputs_unchanged_across_segments':True,'each_segment_policy_boot_unchanged':True,
              'test_data_used':False,'report':{'whole_TRAIN_initialbaseline_qualified':True,'initial':{'initial_prior_exact':True},'initial_state_sha256':'start'}},
          'initial_acceptance':{'full_initialbaseline_qualified':True,'base_plan':inputs['initial_base_plan'],'result':inputs['initial_result'],
              'initial_receipt':inputs['initial_receipt'],'initializer':{'checkpoint':inputs['initial_checkpoint']}},
          'prior':{'contract_sha256':'p','fit_population':{'issue_rows':m.ROWS,'issue_ns_sha256':'issue','truth_sha256':'truth'}},
          'dataset':{'contract':target,'TRAIN_issue_ns_sha256':'issue'},'normalization':{'contract_sha256':'norm'},
          'input_binding':metadata['input_binding'],'full_quality':{'rows':m.ROWS,'truth_sha256':'truth','output_sha256':output,'frozen_prior_contract_sha256':'p'}}
    # Only independent metadata constructors are stubbed, with witnesses; no tested gate is bypassed.
    witnesses=[]
    monkeypatch.setattr(m.objective,'require_native_future_state_train_prior',lambda value,**kw:witnesses.append('prior')or value)
    monkeypatch.setattr(m.model_owner,'native_future_state_model_contract',lambda *a,**kw:witnesses.append('contract')or contract)
    monkeypatch.setattr(m.core,'future_cuda_runtime_matches',lambda a,b:witnesses.append(('runtime',a,b)))
    return inputs,meta,witnesses


def test_initial_receipt_keeps_real_BASE_and_first_fresh_is_separate_provenance(monkeypatch):
    inputs,meta,witnesses=metadata_fixture(monkeypatch)
    validated=m.validate_metadata(inputs,meta)
    assert validated['initial_base_plan']==meta['initial_base_plan']and validated['initial_receipt']['plan']==inputs['initial_base_plan']
    assert inputs['first_fresh_plan']!=inputs['initial_base_plan']and 'prior'in witnesses and 'contract'in witnesses
    assert witnesses[0][0]=='runtime'


@pytest.mark.parametrize('bad,gate',[('receipt_substitution','ACTUAL_INITIAL_BASE_IDENTITY'),('prefix','ACTUAL_INITIAL_BASE_IDENTITY'),
    ('closure','FIRST_FRESH_CLOSURE'),('RNG_provenance','FRESH_CONSTRUCTOR_PROVENANCE'),('old_load','FRESH_CONSTRUCTOR_PROVENANCE'),
    ('accepted','ACCEPTED_WHOLE_INITIAL'),('final_cursor','FINAL_RECEIPT_METADATA'),('final_proof','FINAL_RECEIPT_METADATA'),
    ('quality','MATCHED_FULL_TRUTH_PRIOR_OUTPUT'),('prior','MATCHED_FULL_TRUTH_PRIOR_OUTPUT'),('terminal','COMPLETED_TERMINAL')])
def test_metadata_forgeries_fail_their_intended_gate(monkeypatch,bad,gate):
    inputs,meta,_=metadata_fixture(monkeypatch)
    if bad=='receipt_substitution':meta['initial_receipt']['plan']=inputs['first_fresh_plan']
    elif bad=='prefix':meta['first_fresh_plan']['prefix']['cursor']=1
    elif bad=='closure':meta['first_fresh_closure']['source_inputs_unchanged']=False
    elif bad=='RNG_provenance':meta['first_fresh_closure']['model_RNG_proof']['rng_sha256']='different'
    elif bad=='old_load':meta['first_fresh_closure']['counts']['existing_checkpoint_loads']=1
    elif bad=='accepted':meta['initial_acceptance']['full_initialbaseline_qualified']=False
    elif bad=='final_cursor':meta['None_final_receipt']['logical_cursor']=True
    elif bad=='final_proof':meta['None_final_receipt']['proof']['position']='b'*64
    elif bad=='quality':meta['full_quality']['rows']-=1
    elif bad=='prior':meta['prior']['fit_population']['issue_ns_sha256']='wrong'
    else:meta['completed_terminal']['exit_code']=True
    with pytest.raises(ValueError,match='^PROVIDER_HLC_ENDPOINT_CROSS_'+gate+'$'):
        m.validate_metadata(inputs,meta)
