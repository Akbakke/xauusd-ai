"""Synthetic CUDA-authority metadata/checkpoint guards; no CUDA/model-data job."""
import copy
import io
import numpy as np
import pytest
import torch
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as owner


def reference(name='runtime.json'):
    return {'path':'/synthetic/'+name,'sha256':'a'*64}


def runtime():
    return {'schema_version':'gx1_native_future_state_cuda_runtime_proof_v1',
        'status':'VERIFIED_IDLE_RUNTIME_CURRENT_BOOT','observed_utc':'2026-11-21T12:00:00+00:00',
        'source_commit':'a'*40,'wsl_boot_id':'synthetic-current-boot','physical_windows_boot_id':701,
        'library_versions':{'python':'3.12','numpy':'2.0','torch':'2.8','torch_cuda':'12.8','cudnn':91000},
        'gpu':{'index':0,'uuid':owner.FUTURE_CUDA_GPU_UUID,'name':'synthetic3090','driver_version':'580',
            'total_memory_bytes':24*1024**3},
        'idle':{'gpu_compute_processes':[],'current_project_model_processes':[]},
        'cuda_initialized':False,'no_model_or_checkpoint_opened':True}


def test_cuda_exact_bounded_three_lane_authority_and_deadlines():
    limits=owner.FUTURE_CUDA_LIMITS
    assert [limits[k]['model_forwards']for k in ('continuous','replay','initial')]==[17,9,5540]
    assert [limits[k]['backward_calls']for k in ('continuous','replay','initial')]==[6,1,0]
    assert [limits[k]['autograd_calls']for k in ('continuous','replay','initial')]==[3,0,0]
    assert sum(v['optimizer_steps']for v in limits.values())==5
    assert limits['initial']['optimizer_constructions']==limits['initial']['checkpoint_loads']==0
    assert limits['initial']['checkpoint_writes']==limits['initial']['checkpoint_readbacks']==1
    assert all(v['prior_fits']==v['normalization_fits']==v['CONTROL_rows_scored']==v['TEST_rows']==0 for v in limits.values())
    assert owner.future_cuda_execution('mechanics')['internal_seconds']==1770
    assert owner.future_cuda_execution('initial')['internal_seconds']==5370


@pytest.mark.parametrize('bad',['status','extra','cuda','opened','bootbool','uuid','indexbool','memory',
    'cudnn','library','GPUbusy','CURRENTbusy','timezone'])
def test_cuda_runtime_proof_requires_known_idle_current_boot_domain(bad):
    value=runtime()
    if bad=='status':value['status']='old'
    elif bad=='extra':value['extra']=True
    elif bad=='cuda':value['cuda_initialized']=True
    elif bad=='opened':value['no_model_or_checkpoint_opened']=False
    elif bad=='bootbool':value['physical_windows_boot_id']=True
    elif bad=='uuid':value['gpu']['uuid']='other'
    elif bad=='indexbool':value['gpu']['index']=False
    elif bad=='memory':value['gpu']['total_memory_bytes']=owner.LARGER_VRAM_MAX
    elif bad=='cudnn':value['library_versions']['cudnn']=True
    elif bad=='library':value['library_versions'].pop('torch_cuda')
    elif bad=='GPUbusy':value['idle']['gpu_compute_processes']=[{'pid':123}]
    elif bad=='CURRENTbusy':value['idle']['current_project_model_processes']=[{'pid':123}]
    elif bad=='timezone':value['observed_utc']='2026-11-21T12:00:00'
    with pytest.raises(ValueError):owner.future_cuda_require_runtime(value)


def test_cuda_fresh_live_identity_has_no_invented_age_threshold_but_runtime_must_match():
    expected=runtime();actual=copy.deepcopy(expected);actual['observed_utc']='2026-11-22T14:00:00+00:00'
    owner.future_cuda_runtime_matches(expected,actual)
    actual['library_versions']['torch']='changed'
    with pytest.raises(ValueError,match='CURRENT_RUNTIME'):owner.future_cuda_runtime_matches(expected,actual)


@pytest.mark.parametrize('process_output',['N/A, Not Supported','123','No running processes found'])
def test_cuda_unknown_nvidia_process_query_is_never_guessed_empty(monkeypatch,process_output):
    outputs=iter(['0, '+owner.FUTURE_CUDA_GPU_UUID+', synthetic3090, 580, 24576',process_output])
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:next(outputs))
    with pytest.raises(ValueError,match='PROCESS_QUERY_SCHEMA'):owner.future_cuda_gpu_metadata()


def test_cuda_nvidia_metadata_empty_is_explicit_not_fallback(monkeypatch):
    outputs=iter(['0, '+owner.FUTURE_CUDA_GPU_UUID+', synthetic3090, 580, 24576',''])
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:next(outputs))
    gpu,processes=owner.future_cuda_gpu_metadata()
    assert gpu['total_memory_bytes']==24*1024**3 and processes==[]


def test_cuda_collector_rejects_context_before_sampling_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(owner,'future_cuda_gpu_metadata',lambda:pytest.fail('hardware query after forbidden context'))
    with pytest.raises(ValueError,match='CONTEXT_BEFORE_CLAIM'):owner.collect_future_cuda_runtime_metadata()


@pytest.mark.parametrize('kind',['mechanics','initial'])
@pytest.mark.parametrize('admitted',[False,True])
def test_cuda_exact_gate_closes_all_other_training_authorities(kind,admitted):
    ref=reference('plan.json');training=admitted and kind=='mechanics'
    policy={'training_enabled':training,'native_training_enabled':training,'native_launch_authorized':False,
        'full_epoch_training_allowed':False,'full_val_allowed':False,
        owner.FUTURE_CUDA_PARENT:{owner.FUTURE_CUDA_SCOPES[kind]['gate']:owner.future_cuda_gate_expected(ref,kind,admitted=admitted)}}
    owner.future_cuda_validate_gate(policy,ref,kind,admitted=admitted)
    policy['full_epoch_training_allowed']=True
    with pytest.raises(ValueError):owner.future_cuda_validate_gate(policy,ref,kind,admitted=admitted)


def test_cuda_boolean_invocation_count_is_not_integer_authority():
    ref=reference('plan.json');gate=owner.future_cuda_gate_expected(ref,'mechanics',admitted=True)
    gate['lanes']['continuous']['remaining_invocations']=True
    policy={'training_enabled':True,'native_training_enabled':True,'native_launch_authorized':False,
        'full_epoch_training_allowed':False,'full_val_allowed':False,
        owner.FUTURE_CUDA_PARENT:{owner.FUTURE_CUDA_SCOPES['mechanics']['gate']:gate}}
    with pytest.raises(ValueError,match='EXACT_ONEUSE'):owner.future_cuda_validate_gate(policy,ref,'mechanics',admitted=True)


@pytest.mark.parametrize('name',['/sealed/TEST.py','../escape.py','gx1/TEST.npz','gx1/extra.py'])
def test_cuda_exact_source_inventory_precedes_any_source_or_payload_hash(monkeypatch,name):
    owners=dict.fromkeys(owner.FUTURE_CUDA_SOURCE_PATHS,'a'*64);owners[name]='a'*64
    monkeypatch.setattr(owner,'sha',lambda *a:pytest.fail('hash before exact inventory'))
    with pytest.raises(ValueError,match='SOURCE_INVENTORY'):owner.future_cuda_source_inventory(owners)


def test_cuda_unexpected_input_roles_rejected_before_parent_read(monkeypatch):
    monkeypatch.setattr(owner,'future_cpu_read',lambda *a:pytest.fail('read before exact input roles'))
    with pytest.raises(ValueError,match='INPUT_ROLES'):owner.future_cuda_metadata({'checkpoint':reference('old.pt')},'mechanics',None)


def test_cuda_parity_reports_actual_absolute_relative_and_zero_reference_differences():
    expected=np.array([0.,1.],np.float32)
    exact=owner.future_cuda_array_parity(expected,expected.copy());assert exact['bitexact']and exact['qualified']
    near=owner.future_cuda_array_parity(expected,np.array([1e-7,1.+1e-6],np.float32))
    assert near['qualified']and not near['bitexact']and near['nonzero_difference_at_zero_reference']==1
    assert near['max_absolute_difference']>0 and near['max_relative_difference_nonzero_reference']>0
    assert not owner.future_cuda_array_parity(expected,np.array([0.,1.1],np.float32))['qualified']
    with pytest.raises(ValueError):owner.future_cuda_array_parity(expected,np.array([np.nan,1.],np.float32))


def fake_cuda_rng(monkeypatch):
    state=[torch.arange(16,dtype=torch.uint8)]
    def capture():return {**owner.future_cpu_rng(),'torch_cuda':[v.clone()for v in state]}
    def restore(value):
        assert len(value['torch_cuda'])==1 and value['torch_cuda'][0].dtype==torch.uint8
        owner.future_cpu_restore_rng({k:v for k,v in value.items()if k!='torch_cuda'})
        state[:]=[v.clone()for v in value['torch_cuda']]
    monkeypatch.setattr(owner,'training_rng',capture);monkeypatch.setattr(owner,'restore_training_rng',restore)
    return state


def test_cuda_weights_only_checkpoint_adapter_restores_cpu_and_cuda_rng_and_fourth_step(tmp_path,monkeypatch):
    from tests.test_native_future_state_cpu_mechanics_v1 import packet,reference as cpu_reference,zeros
    plan,prior,packets,model,opt=packet();plan.update(lane_group='mechanics',backbone_gradient_policy=None,
        inputs={'runtime_proof':reference()});fake=fake_cuda_rng(monkeypatch)
    steps=[];counts=zeros()
    for i in range(3):steps.append(owner.future_cpu_step(model,opt,packets[i],plan,counts,i))
    order=[{'synthetic':i}for i in range(4)];ref=cpu_reference()
    checkpoint=owner.future_cuda_checkpoint(model,opt,plan,ref,order,3,steps)
    binding=owner.future_cuda_publish_checkpoint(tmp_path/'cursor3.pt',checkpoint,counts)
    loaded=torch.load(binding['path'],map_location='cpu',weights_only=True)
    assert owner.state_sha(loaded)==owner.state_sha(checkpoint)
    step=owner.future_cpu_step(model,opt,packets[3],plan,counts,3);steps.append(step)
    final=owner.future_cuda_checkpoint(model,opt,plan,ref,order,4,steps)
    _,_,_,second,second_opt=packet();fake[0][0]^=1
    owner.future_cuda_restore(second,second_opt,loaded,plan,ref,order)
    replay=loaded['steps']+[owner.future_cpu_step(second,second_opt,packets[3],plan,zeros('replay'),3)]
    actual=owner.future_cuda_checkpoint(second,second_opt,plan,ref,order,4,replay)
    assert owner.future_cpu_proof(actual)==owner.future_cpu_proof(final)


def test_cuda_bad_rng_rejected_before_any_parameter_copy(monkeypatch):
    plan={'metadata':{},'recipe':{},'inputs':{'runtime_proof':reference()},'backbone_gradient_policy':None,'lane_group':'mechanics'}
    fake_cuda_rng(monkeypatch);value=owner.training_rng();value['torch_cuda']=[torch.ones(3)]
    cp={'schema_version':'gx1_native_future_state_cuda_qualification_checkpoint_v1',
        'bindings':owner.future_cuda_bindings(plan,reference()),'rng_state':value}
    monkeypatch.setattr(owner,'future_cpu_restore',lambda *a:pytest.fail('parameter-copy phase reached'))
    with pytest.raises(ValueError,match='RNG_HEADER_BEFORE_COPY'):
        owner.future_cuda_restore(None,None,cp,plan,reference(),[])


@pytest.mark.parametrize('lane',['continuous','replay','initial'])
def test_cuda_output_root_matches_canonical_guard_initial_has_no_extra_subdir(lane):
    kind='initial'if lane=='initial'else'mechanics'
    expected=owner.FUTURE_CUDA_SCOPES[kind]['root']
    if lane!='initial':expected=expected/lane
    assert owner.future_cuda_lane_root(lane)==expected


class TinyWitnessBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(2))
    def forward(self,**features):
        hidden=self.scale[None,None,:].expand(len(features['seq_x']),5,2)
        return {'entry_q_timeframe_hidden':hidden}


class TinyWitnessModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=TinyWitnessBackbone();self.head=torch.nn.Linear(2,2)
    def forward(self,**features):
        x=self.head(self.backbone(**features)['entry_q_timeframe_hidden'])
        return {'activity_logits':x[...,0],'side_logits':x[...,1]}


def witness_features(unique=False):
    out={'seq_x':torch.ones(2,3,2)}
    for j,tf in enumerate(('M5','M15','H1','H4','D1')):
        out['seq_'+tf.lower()]=torch.ones(2,j+1 if unique else 4,2)
    return out


def test_cuda_equal_TF_witness_observes_eval_no_grad_and_eligible_windows():
    model=TinyWitnessModel().eval();features=witness_features()
    with torch.no_grad():
        direct,hidden,_=owner.future_cuda_forward(model,features,capture=True)
        candidate,other,proof=owner.future_cuda_forward(model,features,equal_tf=True,capture=True)
    assert torch.equal(hidden,other)and all(torch.equal(v,candidate[k])for k,v in direct.items())
    assert proof['batch_eval_active']is proof['scope_enabled']is True
    assert proof['model_training']is proof['grad_enabled']is False
    assert len(proof['eligible_equal_length_TF_groups']['4'])==5


@pytest.mark.parametrize('bad',['training','grad','unique_lengths'])
def test_cuda_equal_TF_cannot_pass_a_silent_fallback(bad):
    model=TinyWitnessModel();model.train(bad=='training')
    with torch.set_grad_enabled(bad=='grad'):
        with pytest.raises(ValueError,match='EQUAL_TF_ACTUAL_ACTIVATION'):
            owner.future_cuda_forward(model,witness_features(bad=='unique_lengths'),equal_tf=True)


@pytest.mark.parametrize('initial',[False,True])
def test_cuda_batch_selector_keeps_unfiltered_axis_and_exact64_tail(monkeypatch,initial):
    from types import SimpleNamespace
    issues=owner.FUTURE_CPU_TRAIN_START+300_000_000_000*(1+np.arange(354550,dtype=np.int64))
    monkeypatch.setattr(owner,'FUTURE_CPU_TRAIN_SHA',owner.digest(issues))
    provider=SimpleNamespace(issue_ns=issues,input_ready=np.ones(len(issues),np.bool_))
    rows,batches=owner.future_cuda_batches(provider,initial=initial)
    assert np.array_equal(rows,np.arange(354550,dtype=np.int64))
    assert len(batches)==(5540 if initial else 4)and len(batches[-1])==(54 if initial else 64)
    assert np.array_equal(np.concatenate(batches),rows if initial else rows[:256])
