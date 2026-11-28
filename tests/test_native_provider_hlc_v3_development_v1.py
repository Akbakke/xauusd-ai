"""Focused metadata/IO guards; synthetic values, no CUDA/model or real arrays."""
import copy
import json
import subprocess
from pathlib import Path
import pytest
from gx1.audit import native_provider_hlc_learning_v1 as owner


def protocol():
    return json.loads(owner.DEV_PROTOCOL.read_text())


def test_exact_legacy_source_reversal():
    actual=owner.development_reverse_source((owner.REPO/owner.SELF).read_bytes())
    expected=subprocess.check_output(['git','show',owner.DEV_PARENT+':'+owner.SELF],cwd=owner.REPO)
    assert actual==expected


@pytest.mark.parametrize('mutation',['duplicate','extra_dispatch','legacy_edit'])
def test_source_reversal_rejects_other_changes(mutation):
    raw=(owner.REPO/owner.SELF).read_bytes()
    if mutation=='duplicate':raw+=b'# BEGIN GX1 V3 BOUNDED DEVELOPMENT 20261127\n# END GX1 V3 BOUNDED DEVELOPMENT 20261127\n'
    elif mutation=='extra_dispatch':raw+=b" if sys.argv[1:2]==['--v3-development-plan-sha256']:return main_v3_development()\n"
    else:raw=raw.replace(b'One explicitly admitted',b'Changed legacy admitted',1)
    try:actual=owner.development_reverse_source(raw)
    except ValueError:return
    expected=subprocess.check_output(['git','show',owner.DEV_PARENT+':'+owner.SELF],cwd=owner.REPO)
    assert actual!=expected


def test_recipe_derived_from_canonical_owner():
    p=protocol();recipe=owner.development_recipe(p)
    assert recipe==p['recipe']
    assert recipe['passes']==2 and recipe['train_batch_rows']==96 and recipe['eval_batch_rows']==256
    assert p['fit_checkpoint_cursors']==[1,432]


@pytest.mark.parametrize('field,value',[('passes',1),('seed',17),('allocator_phase_policy',{}),('optimizer',{})])
def test_recipe_rejects_changed_decisions(field,value):
    p=protocol();p['recipe'][field]=value
    with pytest.raises(ValueError):owner.development_recipe(p)


@pytest.mark.parametrize('phase',owner.DEV_PHASES)
def test_exact_phase_gate_and_flags(phase):
    binding={'path':'/synthetic/plan.json','sha256':'1'*64};gate=owner.development_gate(binding,phase,True)
    assert set(gate)=={'status','plan','execution_authorized','remaining_invocations','relaunch_allowed','training_authorized','test_authorized'}
    assert gate['training_authorized']==(phase!='cohort') and gate['remaining_invocations']==1
    policy={key:False for key in owner.DEV_FLAGS}
    if phase!='cohort':policy.update(training_enabled=True,native_training_enabled=True)
    owner.development_validate_flags(policy,phase,True)
    policy['full_epoch_training_allowed']=True
    with pytest.raises(ValueError):owner.development_validate_flags(policy,phase,True)


def test_json_roundtrip_precedes_any_publication():
    assert json.loads(owner.development_json_bytes({'value':(1,True,None)}))=={'value':[1,True,None]}
    with pytest.raises(ValueError):owner.development_json_bytes({'not_finite':float('nan')})
    with pytest.raises(ValueError):owner.development_json_bytes({'unsupported':object()})


@pytest.mark.parametrize('phase,steps',[('cohort',0),('warm',3),('cold',1),('fit',432)])
def test_fixed_operation_budgets(phase,steps):
    limits=owner.development_limits(phase)
    assert limits['optimizer_steps']==steps
    if phase=='fit':
        assert limits['model_forwards']==678 and limits['backward_calls']==432
        assert limits['autograd_calls']==9 and limits['checkpoint_writes']==limits['checkpoint_readbacks']==2
        assert limits['checkpoint_loads']==limits['model_restores']==0


@pytest.mark.parametrize('phase',owner.DEV_PHASES)
def test_payload_allowlist_is_phase_exact(phase):
    plan={'phase':phase,'inputs':{'feature_archive':{'path':'/synthetic/features.npz'},'target_arrays':{'path':'/synthetic/targets.npz'},'selected_cursor2_checkpoint':{'path':'/synthetic/technical/CURSOR002.pt'}}}
    allowed=owner.development_allowed_payloads(plan)
    owner.development_io_audit(allowed,'open',('/synthetic/features.npz','r',0))
    with pytest.raises(RuntimeError):owner.development_io_audit(allowed,'open',('/synthetic/other.npz','r',0))
    if phase=='cold':assert '/synthetic/technical/CURSOR002.pt'in allowed
    else:assert '/synthetic/technical/CURSOR002.pt'not in allowed
    if phase=='fit':
        assert str(owner.DEV_ROOTS[phase]/'FINAL.pt')in allowed
        assert str(owner.DEV_ROOTS[phase]/'INITIAL.pt')not in allowed


@pytest.mark.parametrize('event',['socket.connect','socket.getaddrinfo','urllib.Request'])
def test_network_always_denied(event):
    with pytest.raises(RuntimeError):owner.development_io_audit(set(),event,())

def original_fit():
    return owner.read(owner.DEV_RECOVERY_ORIGINAL)


def test_recovery_source_bridge_preserves_original_numeric_leaf():
    proof=owner.recovery_source_bridge()
    assert proof['original_development_nn_and_numeric_paths_unchanged']
    assert proof['producer_commit']==owner.DEV_RECOVERY_ANCHOR


@pytest.mark.parametrize('lane,updates,forwards',[('reference',2,2),('cold',1,1),('recovery',431,596)])
def test_recovery_operation_limits_are_derived_from_original(lane,updates,forwards):
    plan=original_fit();limits=owner.recovery_limits(lane,plan)
    assert limits['optimizer_steps']==limits['backward_calls']==updates
    assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==forwards
    assert limits['checkpoint_writes']==limits['checkpoint_readbacks']==limits['checkpoint_prepublication_loads']==1
    assert limits['checkpoint_loads']==limits['model_restores']==(lane!='reference')
    plan['schedule']['optimizer_updates']+=1
    with pytest.raises(ValueError):owner.recovery_limits(lane,plan)


@pytest.mark.parametrize('lane',owner.DEV_RECOVERY_LANES)
def test_recovery_payload_scope_allows_bound_hashes_and_rejects_unknown_initializer(lane):
    plan={'lane':lane,'inputs':{'feature_archive':{'path':'/s/features.npz'},'target_arrays':{'path':'/s/truth.npz'},'original_checkpoint':{'path':'/s/genuine/CURSOR001.pt'},'original_INITIAL_logits':{'path':'/s/genuine/INITIAL_LOGITS.npz'}}}
    allowed=owner.recovery_allowed_payloads(plan)
    assert '/s/genuine/CURSOR001.pt'in allowed and '/s/genuine/INITIAL_LOGITS.npz'in allowed
    for path in ['/s/technical/CURSOR002.pt','/s/unknown/INITIAL_LOGITS.npz']:
        with pytest.raises(RuntimeError):owner.development_io_audit(allowed,'open',(path,'r',0))


@pytest.mark.parametrize('field',('lane','runtime','original_plan','scores'))
def test_recovery_rejects_forged_producer_before_parameter_copy(monkeypatch,field):
    runtime={'actual':'original'};initial={'path':'/original/INITIAL_SCORE.json','sha256':'1'*64}
    plan={'metadata':{'receipt':{'producer_runtime':runtime,'proof':{}},'original':{'schedule':{}},'trajectory':{}},'inputs':{'original_INITIAL_score':initial}}
    cp={'producer':{'original_plan':owner.DEV_RECOVERY_ORIGINAL,'lane':'fit','runtime':runtime},'scores':{'INITIAL':initial}}
    if field=='scores':cp['scores']={'INITIAL':{**initial,'path':'/new/INITIAL_SCORE.json'}}
    else:cp['producer'][field]='forged'
    called=[];monkeypatch.setattr(owner.train,'strict_restore',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError):owner.recovery_restore(None,None,cp,plan,{})
    assert not called


def test_semantic_proof_excludes_producer_but_binds_every_state_component():
    keys=('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores')
    a={k:'state:'+k for k in keys};a.update(producer={'actual':'reference'},schema_version='v')
    b=copy.deepcopy(a);b['producer']={'actual':'cold'}
    assert owner.train.checkpoint_proof(a)==owner.train.checkpoint_proof(b)
    assert owner.core.state_sha(a)!=owner.core.state_sha(b)
    for key in keys:
        changed=copy.deepcopy(b);changed[key]='different'
        assert owner.train.checkpoint_proof(a)!=owner.train.checkpoint_proof(changed)


def fake_recovery(monkeypatch,lane,restore_lane='fit'):
    """Synthetic state-machine test: no tensors, model construction, CUDA or file IO."""
    import torch
    from gx1.contracts import market_state_ta_integrated_model_v1 as factory
    original=original_fit();runtime={'actual':'synthetic-runtime'};initial={'path':'/original/INITIAL_SCORE.json','sha256':'1'*64}
    binding={'path':'/new/'+lane+'.json','sha256':'2'*64};checkpoint_binding={'path':'/original/CURSOR001.pt','sha256':'3'*64}
    trajectory=owner.development_trajectory(original);rng={'value':0};events={'steps':[],'scores':[],'checkpoints':[],'json':[],'loads':[]}
    class Handle:
        def remove(self):pass
    class Part:
        def register_forward_pre_hook(self,fn):return Handle()
    class Model(Part):
        def __init__(self):self.value=0;self.training=True;self.backbone=Part();self.readout=Part();self.research_objective=original['model_contract']
        def state_dict(self):return {'weight':self.value}
        def named_parameters(self):return [('readout.weight',object())]
        def parameters(self):return []
    class Optimizer:
        def __init__(self,*a,**k):self.value=0;self.state={}
        def state_dict(self):return {'moment':self.value}
    monkeypatch.setattr(owner.core,'training_rng',lambda:dict(rng))
    monkeypatch.setattr(owner.core,'future_cuda_seed_and_runtime',lambda p:rng.update(value=0))
    original['initial_reference']['model_state_sha256']=owner.core.state_sha({'weight':0})
    original['initial_reference']['rng_sha256']=owner.core.state_sha({'value':0})
    model=Model();optimizer=Optimizer();model.value=optimizer.value=1;rng['value']=1
    first={'logical_step':1,'skip_update':False,'loss':1.}
    cp=owner.train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=owner.DEV_RECOVERY_ORIGINAL,lane='fit',cursor=1,steps=[first],scores={'INITIAL':initial},producer_runtime=runtime)
    receipt={'producer_runtime':runtime,'proof':owner.train.checkpoint_proof(cp)}
    model.value=optimizer.value=2;rng['value']=2
    expected=owner.train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=binding,lane='reference',cursor=2,steps=[first,{'logical_step':2,'skip_update':False,'loss':1.}],scores={'INITIAL':initial},producer_runtime=runtime)
    reference={'proof':owner.train.checkpoint_proof(expected)}
    plan={'lane':lane,'metadata':{'original':original,'receipt':receipt,'trajectory':trajectory},'inputs':{'original_INITIAL_score':initial,'original_checkpoint':checkpoint_binding,'reference_cursor2_receipt':{'path':'/reference/C2.json','sha256':'4'*64}},'limits':owner.recovery_limits(lane,original),'execution':original['execution'],'starting_cursor':0 if lane=='reference'else 1,'ending_cursor':432 if lane=='recovery'else 2}
    monkeypatch.setattr(factory,'build_native_future_state_model',lambda **k:Model())
    monkeypatch.setattr(torch.optim,'AdamW',Optimizer)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    context={'normalization':{},'input_binding':{},'prior':{},'target_contract':original['metadata']['target_contract'],'batches':list(range(216)),'schedule':original['schedule']}
    monkeypatch.setattr(owner,'development_nn_context',lambda p:context)
    monkeypatch.setattr(owner.adapter,'packet',lambda c,take:{'take':take})
    def step(m,opt,packet,p,counts,j):
        assert p is original
        events['steps'].append((j,packet['take']));m.value+=1;opt.value+=1;opt.state={'initialized':True};rng['value']+=1
        for k in ('optimizer_steps','backward_calls','model_forwards','backbone_forwards','readout_forwards'):counts[k]+=1
        return {'logical_step':j+1,'skip_update':False,'loss':1.}
    monkeypatch.setattr(owner.train,'training_step',step)
    def save(root,name,value,counts,execution):
        events['checkpoints'].append((name,copy.deepcopy(value)))
        for k in ('checkpoint_writes','checkpoint_readbacks','checkpoint_receipt_writes','checkpoint_prepublication_loads'):counts[k]+=1
        return {'path':str(root/Path(name).with_suffix('.json')),'sha256':'5'*64}
    monkeypatch.setattr(owner,'development_checkpoint',save)
    monkeypatch.setattr(owner,'development_publish',lambda root,name,value:events['json'].append((name,copy.deepcopy(value)))or{'path':str(root/name),'sha256':'6'*64})
    monkeypatch.setattr(owner,'ref',lambda path:checkpoint_binding)
    monkeypatch.setattr(torch,'load',lambda *a,**k:events['loads'].append(a[0])or copy.deepcopy(cp))
    def restore(m,opt,c,**kwargs):
        assert c['producer']['lane']==restore_lane and kwargs['expected_state_proof']==receipt['proof']
        m.value=c['model_state']['weight'];opt.value=c['optimizer_state']['moment'];rng.update(c['rng_state'])
        return c['position']
    monkeypatch.setattr(owner.train,'strict_restore',restore)
    monkeypatch.setattr(owner,'read',lambda ref:reference)
    monkeypatch.setattr(owner.train,'release_phase_allocator_cache',lambda *a,**k:None)
    def score(m,c,p,root,label,counts):
        assert p is original;events['scores'].append(label)
        for k in ('model_forwards','backbone_forwards','readout_forwards'):counts[k]+=81
        counts['logit_archive_writes']+=1;counts['logit_archive_readbacks']+=1
        return {'path':str(root/(label+'_SCORE.json')),'sha256':'7'*64}
    monkeypatch.setattr(owner,'development_score',score)
    def gradient(m,packet,p,counts):
        for k in ('model_forwards','backbone_forwards','readout_forwards'):counts[k]+=1
        counts['autograd_calls']+=3;return {'synthetic':True}
    monkeypatch.setattr(owner.core,'future_cuda_gradient_probe',gradient)
    return plan,binding,runtime,events,receipt,reference


@pytest.mark.parametrize('lane',owner.DEV_RECOVERY_LANES)
def test_exact_recovery_state_machine_preserves_original_trajectory(monkeypatch,tmp_path,lane):
    plan,binding,runtime,events,receipt,reference=fake_recovery(monkeypatch,lane)
    report=owner.recovery_nn(plan,binding,tmp_path,runtime)
    assert report['counts']==plan['limits']
    assert len(events['loads'])==(0 if lane=='reference' else 1)
    assert report['scores']['INITIAL']==plan['inputs']['original_INITIAL_score']
    assert report['original_uncheckpointed_updates']=='UNKNOWN'
    name,cp=events['checkpoints'][-1]
    assert cp['producer']['runtime']==runtime and cp['producer']['lane']==lane
    assert cp['producer']['original_plan']==(binding if lane=='reference'else owner.DEV_RECOVERY_ORIGINAL)
    if lane=='reference':
        assert events['steps']==[(0,0),(1,1)]and report['original_C1_all10_equal']
        assert owner.train.checkpoint_proof(cp)==reference['proof']
        assert events['json'][0][1]['actual_producer']['original_plan']==binding
    elif lane=='cold':
        assert events['steps']==[(1,1)]and report['all10_proofs_equal']
    else:
        assert len(events['steps'])==431 and events['steps'][0]==(1,1)and events['steps'][-1]==(431,215)
        assert events['scores']==['PASS1','PASS2']and cp['position']['optimizer_updates']==432
        assert len(cp['steps'])==432 and name=='FINAL.pt'


def test_reference_c1_mismatch_stops_before_second_update_or_checkpoint(monkeypatch,tmp_path):
    plan,binding,runtime,events,receipt,_=fake_recovery(monkeypatch,'reference')
    receipt['proof']['steps']='wrong'
    with pytest.raises(ValueError,match='INDEPENDENT_C1_ALL10'):owner.recovery_nn(plan,binding,tmp_path,runtime)
    assert events['steps']==[(0,0)]and not events['checkpoints']and not events['json']


def test_cold_c2_mismatch_never_publishes_replayed_checkpoint(monkeypatch,tmp_path):
    plan,binding,runtime,events,_,reference=fake_recovery(monkeypatch,'cold')
    reference['proof']['model_state']='wrong'
    with pytest.raises(ValueError,match='COLD_C2_ALL10'):owner.recovery_nn(plan,binding,tmp_path,runtime)
    assert events['steps']==[(1,1)]and not events['checkpoints']


@pytest.mark.parametrize('admitted',(False,True))
@pytest.mark.parametrize('field',('remaining_invocations','execution_authorized','relaunch_allowed','training_authorized','test_authorized'))
def test_recovery_gate_rejects_bool_integer_type_aliases(admitted,field):
    binding={'path':'/new/plan.json','sha256':'1'*64};gate=owner.recovery_gate(binding,'reference',admitted)
    owner.recovery_require_gate(gate,binding,'reference',admitted)
    gate[field]=bool(gate[field])if field=='remaining_invocations'else int(gate[field])
    with pytest.raises(ValueError,match='GATE_TYPES'):owner.recovery_require_gate(gate,binding,'reference',admitted)


def test_recovery_hash_checks_allow_all_declared_payloads_without_decoding(monkeypatch,tmp_path):
    inputs={}
    for role,suffix in [('feature_archive','.npz'),('target_arrays','.npz'),('original_checkpoint','.pt'),('original_INITIAL_logits','.npz')]:
        path=tmp_path/(role+suffix);path.write_bytes(b'hash-only synthetic bytes');inputs[role]=owner.ref(path)
    plan={'lane':'reference','inputs':inputs,'source_commit':'head','source_owners':{},'metadata_input_roles':[]}
    allowed=owner.recovery_allowed_payloads(plan);opened=[];real_open=Path.open
    def guarded(path,*a,**k):
        owner.development_io_audit(allowed,'open',(str(path),'r',0));opened.append(str(path));return real_open(path,*a,**k)
    monkeypatch.chdir(owner.REPO)
    monkeypatch.setattr(owner.subprocess,'check_output',lambda argv,**k:'work/gx1-current'if argv[1]=='branch'else'head')
    monkeypatch.setattr(Path,'open',guarded)
    owner.recovery_check_inputs(plan,physical=True)
    assert set(opened)=={v['path']for v in inputs.values()}
    with pytest.raises(RuntimeError):owner.development_io_audit(allowed,'open',(str(tmp_path/'undeclared.pt'),'r',0))


@pytest.mark.parametrize('bad',(None,'sha','lane','flags','protocol','counter','scope'))
def test_recovery_immutable_admission_snapshot_is_exact(monkeypatch,bad):
    binding={'path':'/new/plan.json','sha256':'1'*64};protocol={'path':'/new/protocol.json','sha256':'2'*64};snapshot_ref={'path':'/immutable/admission.json','sha256':'3'*64};policy_ref={'path':str(owner.POLICY),'sha256':snapshot_ref['sha256']}
    value={k:False for k in owner.DEV_FLAGS};value.update(training_enabled=True,native_training_enabled=True)
    parent={'protocol':protocol,'original_plan':owner.DEV_RECOVERY_ORIGINAL,'checkpoint_receipt':owner.DEV_RECOVERY_RECEIPT,'execution_authorized':True,'active_lane':'reference','main_fit_authorized':False,'test_authorized':False,'spending_authorized':False,'lane_gates':{'reference':owner.recovery_gate(binding,'reference',True)}};value[owner.DEV_RECOVERY_STAGE]=parent
    if bad=='sha':policy_ref['sha256']='4'*64
    if bad=='lane':parent['active_lane']='cold'
    if bad=='flags':value['full_epoch_training_allowed']=True
    if bad=='protocol':parent['protocol']={'path':'/wrong.json','sha256':'2'*64}
    if bad=='counter':parent['lane_gates']['reference']['remaining_invocations']=True
    if bad=='scope':parent['test_authorized']=True
    seen=[]
    monkeypatch.setattr(owner,'ref',lambda p:seen.append(str(p))or snapshot_ref)
    monkeypatch.setattr(owner,'read',lambda r:value)
    if bad is None:
        assert owner.recovery_admission_snapshot('reference',binding,protocol,policy_ref)==snapshot_ref
        assert seen==[str(owner.E/'V3_FIT004_EXACT_RECOVERY_REFERENCE_ADMITTED_POLICY_20261128_002.json')]
    else:
        with pytest.raises(ValueError):owner.recovery_admission_snapshot('reference',binding,protocol,policy_ref)


@pytest.mark.parametrize('changed',(False,True))
def test_own_admission_snapshot_is_part_of_six_bookend_inputs(monkeypatch,tmp_path,changed):
    binding={'path':str(owner.DEV_RECOVERY_PLANS['reference']),'sha256':'1'*64};pref={'path':str(owner.POLICY),'sha256':'2'*64};boot={'physical_windows_boot_id':4,'wsl_boot_id':'boot'}
    plan={'lane':'reference','protocol':{},'source_owners':{},'inputs':{},'source_commit':'head'}
    monkeypatch.setattr(owner,'ref',lambda p:pref if str(p)==str(owner.POLICY)else binding)
    monkeypatch.setattr(owner.core,'physical_windows_boot_id',lambda:4)
    real_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'boot'if str(p)=='/proc/sys/kernel/random/boot_id'else real_read(p,*a,**k))
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:'head')
    def snapshot(*a):
        if changed:raise ValueError('snapshot bytes changed')
        return {'path':'/immutable/snapshot.json','sha256':pref['sha256']}
    monkeypatch.setattr(owner,'recovery_admission_snapshot',snapshot)
    checks=owner.recovery_bookends(plan,binding,pref,boot)
    assert set(checks)=={'plan','sources','inputs','policy','boots','HEAD'}
    assert checks['inputs']is(not changed)and all(v for k,v in checks.items()if k!='inputs')

def test_direction_gradient_addition_preserves_prior_leaf_bridges():
    assert owner.direction_gradient_source_bridge()['old_numeric_and_recovery_functions_unchanged']
    assert owner.recovery_source_bridge()['original_development_nn_and_numeric_paths_unchanged']
    assert owner.development_source_bridge()['baseline74_numeric_functions_changed']==0


@pytest.mark.parametrize('admitted',(False,True))
@pytest.mark.parametrize('wrong',('remaining_invocations','execution_authorized','training_authorized'))
def test_direction_gradient_gate_types_and_zero_training_authority(admitted,wrong):
    ref={'path':'/s/plan.json','sha256':'1'*64};gate=owner.direction_gradient_gate(ref,admitted)
    owner.direction_gradient_require_gate(gate,ref,admitted)
    assert gate['training_authorized']is False
    gate[wrong]=bool(gate[wrong])if wrong=='remaining_invocations'else int(gate[wrong])
    with pytest.raises(ValueError):owner.direction_gradient_require_gate(gate,ref,admitted)


def tiny_direction_probe(monkeypatch,*,unknown=False,consume_rng=False,mutate_model=False):
    """Actual CPU autograd on explicit synthetic scalar signals; no native model/CUDA/data."""
    import torch
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.ModuleDict({name:torch.nn.Linear(1,1,bias=False)for name in MODEL_NATIVE_TRAINING_SPECIALISTS})
            self.readout=torch.nn.Module()
            self.readout.activity_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(1,1)for tf in owner.train.TF})
            self.readout.side_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(1,1)for tf in owner.train.TF})
            with torch.no_grad():
                for p in self.parameters():p.fill_(0.125)
                for heads in (self.readout.activity_heads,self.readout.side_heads):
                    for head in heads.values():head.bias.zero_()
        def forward(self,signal):
            if consume_rng:torch.rand(1)
            if mutate_model:
                with torch.no_grad():self.readout.activity_heads['M5'].bias.add_(0.1)
            hidden=sum(layer(signal)for layer in self.backbone.values())
            return {task+'_logits':torch.cat([head(hidden)for head in heads.values()],dim=1)for task,heads in [('activity',self.readout.activity_heads),('side',self.readout.side_heads)]}
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
    def forbidden(*a,**k):raise AssertionError('No optimizer mutation allowed')
    monkeypatch.setattr(optimizer,'step',forbidden);monkeypatch.setattr(optimizer,'zero_grad',forbidden)
    monkeypatch.setattr(owner.core,'training_rng',lambda:{'torch_cpu':torch.get_rng_state().clone()})
    monkeypatch.setattr(owner.core,'restore_training_rng',lambda value:torch.set_rng_state(value['torch_cpu']))
    monkeypatch.setattr(owner.core,'future_cuda_forward',lambda m,features:(m(**features),None,{}))
    labels={'activity':torch.ones(2,5),'side':torch.tensor([[0.]*5,[1.]*5])};masks={key:torch.ones(2,5,dtype=torch.bool)for key in labels}
    if unknown:masks['side'][:,0]=False
    def loss(*,model_output,**kwargs):
        sums={};counts={}
        for task in labels:
            raw=torch.nn.functional.binary_cross_entropy_with_logits(model_output[task+'_logits'],labels[task],reduction='none')
            sums[task]=torch.where(masks[task],raw,torch.zeros_like(raw)).sum(0);counts[task]=masks[task].sum(0)
        return {'loss_sums':sums,'counts':counts,'count':int(counts['activity'].sum())}
    monkeypatch.setattr(objective,'native_future_state_loss',loss)
    return torch,model,optimizer,{'features':{'signal':torch.tensor([[-1.],[1.]])},'truth':{}},{'metadata':{'target_contract':{}}}


@pytest.mark.parametrize('unknown',(False,True))
def test_direction_gradient_exact_terms_denominator_and_unchanged_state(monkeypatch,unknown):
    torch,m,opt,packet,original=tiny_direction_probe(monkeypatch,unknown=unknown)
    before={'model':owner.core.state_sha(m.state_dict()),'optimizer':owner.core.state_sha(opt.state_dict()),'rng':owner.core.state_sha(owner.core.training_rng())}
    counts={'autograd_calls':0}
    for _ in range(3):
        report=owner.direction_gradient_probe(m,opt,packet,original,counts)
        assert report['denominator_known_activity_cells']==10
        assert set(report['per_task'])=={task+'.'+tf for task in ('activity','side')for tf in owner.train.TF}
        assert report['temporary_eval_mode']and report['original_training_mode']is True
        assert all(report['observed_before_cleanup'].values())
        bias=report['per_task']['activity.M5']['readout']['readout.activity_heads.M5.bias']['signed_gradient']
        assert bias==pytest.approx(-0.1,abs=1e-7)
        assert len(report['shared_group_pairwise']['all_backbone'])==45
        if unknown:
            assert report['per_task']['side.M5']['status']=='UNKNOWN_NO_OBSERVED_TASK_SUPPORT'
            assert all(value==0. for value in report['per_task']['side.M5']['shared_group_l2'].values())
            assert report['shared_group_pairwise']['all_backbone']['activity.M5__side.M5']=={'status':'UNKNOWN_TASK_SUPPORT'}
    assert counts['autograd_calls']==30
    assert m.training is True and all(p.grad is None for p in m.parameters())
    assert before=={'model':owner.core.state_sha(m.state_dict()),'optimizer':owner.core.state_sha(opt.state_dict()),'rng':owner.core.state_sha(owner.core.training_rng())}


def test_direction_gradient_rng_consumption_is_detected_before_cleanup(monkeypatch):
    _,m,opt,packet,original=tiny_direction_probe(monkeypatch,consume_rng=True)
    before=owner.core.state_sha(owner.core.training_rng())
    with pytest.raises(ValueError,match='OBSERVED_STATE_MUTATION'):owner.direction_gradient_probe(m,opt,packet,original,{'autograd_calls':0})
    assert owner.core.state_sha(owner.core.training_rng())==before
    assert m.training is True


def test_direction_gradient_model_mutation_fails_without_hiding_it(monkeypatch):
    _,m,opt,packet,original=tiny_direction_probe(monkeypatch,mutate_model=True)
    with pytest.raises(ValueError):owner.direction_gradient_probe(m,opt,packet,original,{'autograd_calls':0})


def test_direction_gradient_refuses_existing_gradients(monkeypatch):
    torch,m,opt,packet,original=tiny_direction_probe(monkeypatch)
    first=next(m.parameters());first.grad=torch.ones_like(first)
    with pytest.raises(ValueError,match='INITIAL_GRAD_NONE'):owner.direction_gradient_probe(m,opt,packet,original,{'autograd_calls':0})


def test_direction_gradient_unknown_parameter_owner_fails(monkeypatch):
    torch,m,_,_,_=tiny_direction_probe(monkeypatch);m.extra=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='PARAMETER_OWNER'):owner.direction_gradient_parameter_groups(tuple(m.named_parameters()))


@pytest.mark.parametrize('bad',(None,'flags','sha','training','counter','design'))
def test_direction_gradient_immutable_snapshot_with_all_training_flags_closed(monkeypatch,bad):
    binding={'path':'/s/plan.json','sha256':'1'*64};snapshot_ref={'path':str(owner.DIRECTION_GRADIENT_POLICY_SNAPSHOT),'sha256':'2'*64};pref={'path':str(owner.POLICY),'sha256':'2'*64}
    policy={k:False for k in owner.DEV_FLAGS};policy[owner.DIRECTION_GRADIENT_STAGE]={**owner.direction_gradient_gate(binding,True),'design':owner.DIRECTION_GRADIENT_DESIGN}
    if bad=='flags':policy['training_enabled']=True
    if bad=='sha':pref['sha256']='3'*64
    if bad=='training':policy[owner.DIRECTION_GRADIENT_STAGE]['training_authorized']=True
    if bad=='counter':policy[owner.DIRECTION_GRADIENT_STAGE]['remaining_invocations']=True
    if bad=='design':policy[owner.DIRECTION_GRADIENT_STAGE]['design']={}
    monkeypatch.setattr(owner,'ref',lambda p:snapshot_ref);monkeypatch.setattr(owner,'read',lambda r:policy)
    if bad is None:assert owner.direction_gradient_admission_snapshot(binding,pref)==snapshot_ref
    else:
        with pytest.raises(ValueError):owner.direction_gradient_admission_snapshot(binding,pref)



def test_direction_gradient_pair_roles_preserve_math_without_false_M5_scope():
    import torch
    value=owner.direction_gradient_pair_measures((torch.tensor([1.]),),(torch.tensor([-2.]),),'activity.M15','side.H4')
    assert value['left_task']=='activity.M15'and value['right_task']=='side.H4'
    assert value['scope']=='pair_only_left_plus_right_not_all_ten_tasks'
    assert value['left_l2']==1. and value['right_l2']==2. and value['cross_dot']==-2. and value['cosine']==-1.
    assert value['left_dot_pair_sum']==-1. and value['raw_pair_sum_opposes_left']is True
    assert not any('M5'in k or 'g5'in k or 'gaux'in k or 'joint'in k for k in value)

def test_block_order_all_prior_leaf_bridges_preserve_numeric_functions():
    assert owner.block_order_source_bridge()['all_old_numeric_functions_unchanged']
    assert owner.direction_gradient_source_bridge()['old_numeric_and_recovery_functions_unchanged']
    assert owner.recovery_source_bridge()['original_development_nn_and_numeric_paths_unchanged']
    assert owner.development_source_bridge()['baseline74_numeric_functions_changed']==0


def test_block_order_private_rng_preserves_all_global_streams_and_whole_tail():
    import random
    import numpy as np
    import torch
    p=original_fit();before=owner.core.state_sha({'numpy':np.random.get_state(),'python':random.getstate(),'torch_cpu':torch.get_rng_state()})
    schedule=owner.block_order_schedule(p)
    assert owner.core.state_sha({'numpy':np.random.get_state(),'python':random.getstate(),'torch_cpu':torch.get_rng_state()})==before
    assert schedule==owner.block_order_schedule(p)
    for permutation in schedule['permutations']:
        assert sorted(permutation)==list(range(216))and permutation.count(215)==1
    assert schedule['original_batch_rows'][-1]==87 and sum(schedule['original_batch_rows'])==20727
    assert schedule['logical_original_batch_ids']==sum(schedule['permutations'],[])
    for j,batch in enumerate(schedule['logical_original_batch_ids']):
        assert schedule['ordered_batch_source_rows_sha256'][j]==p['schedule']['batch_source_rows_sha256'][batch]
        assert schedule['ordered_batch_issue_ns_sha256'][j]==p['schedule']['batch_issue_ns_sha256'][batch]
    assert owner.block_order_cursor(schedule,432)['next_original_batch']is None
    assert owner.block_order_cursor(schedule,1)['next_original_batch']['original_batch_id']==schedule['permutations'][0][1]


@pytest.mark.parametrize('bad',('duplicate','bool','within_pass_swap','source_hash','logical_flatten','seed'))
def test_block_order_rejects_mutated_or_unbound_schedules(bad):
    p=original_fit();s=owner.block_order_schedule(p)
    if bad=='duplicate':s['permutations'][0][0]=s['permutations'][0][1]
    if bad=='bool':s['permutations'][0][0]=True
    if bad=='within_pass_swap':s['permutations'][0][0],s['permutations'][0][1]=s['permutations'][0][1],s['permutations'][0][0]
    if bad=='source_hash':s['ordered_batch_source_rows_sha256'][0]='0'*64
    if bad=='logical_flatten':s['logical_original_batch_ids'][0]=s['logical_original_batch_ids'][1]
    if bad=='seed':s['seed']+=1
    with pytest.raises(ValueError):owner.block_order_validate_schedule(s,p)


@pytest.mark.parametrize('bad',('chronological_trajectory','wrong_producer','wrong_cursor','wrong_batch','permutation'))
def test_block_order_cross_restore_rejects_before_any_parameter_copy(monkeypatch,bad):
    original=original_fit();schedule=owner.block_order_schedule(original);binding={'path':'/new/order.json','sha256':'1'*64};plan={'metadata':{'original':original,'trajectory':owner.block_order_trajectory(original,schedule)},'schedule':schedule}
    cp={'trajectory':plan['metadata']['trajectory'],'producer':{'original_plan':binding},'position':owner.train.position(schedule,1),'steps':[{'logical_step':1,'skip_update':False,'order_binding':owner.block_order_step_binding(schedule,0)}]}
    if bad=='chronological_trajectory':cp['trajectory']=owner.development_trajectory(original)
    if bad=='wrong_producer':cp['producer']['original_plan']=owner.DEV_RECOVERY_ORIGINAL
    if bad=='wrong_cursor':cp['position']=owner.train.position(schedule,2)
    if bad=='wrong_batch':cp['steps'][0]['order_binding']['original_batch_id']=(cp['steps'][0]['order_binding']['original_batch_id']+1)%216
    if bad=='permutation':schedule['permutations'][0][0]=schedule['permutations'][0][1]
    called=[];monkeypatch.setattr(owner.train,'strict_restore',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError):owner.block_order_strict_restore(None,None,cp,plan,binding,1,{}, {})
    assert not called


@pytest.mark.parametrize('bad',(False,True))
def test_block_order_INITIAL_reuse_is_named_and_factory_bound(monkeypatch,bad):
    p=original_fit();lineage={'original_INITIAL_score':{'path':'/old/INITIAL.json','sha256':'1'*64},'original_INITIAL_logits':{'path':'/old/INITIAL.npz','sha256':'2'*64}}
    score={'phase':'INITIAL','rows':20727,'logits':lineage['original_INITIAL_logits'],'canonical_scores':{'same':1},'constant_scores':{'factory':{'same':1}},'truth_sha256':'truth'}
    cold={'status':'COMPLETE_V3_DEVELOPMENT_COLD','bookends':{'all':True},'report':{'cold_replay_bit_exact':True,'full_cursor3_proof':dict.fromkeys(range(10),'proof'),'technical_weights_never_fit_initializer':True}}
    if bad:score['canonical_scores']={'different':True}
    monkeypatch.setattr(owner,'read',lambda binding:cold if binding==p['inputs']['development_cold_result']else score)
    if bad:
        with pytest.raises(ValueError):owner.block_order_initial_reuse_metadata(p,lineage)
    else:
        proof=owner.block_order_initial_reuse_metadata(p,lineage)
        assert proof['INITIAL_score']==lineage['original_INITIAL_score']and proof['INITIAL_logits']==lineage['original_INITIAL_logits']
        assert proof['factory_model_sha256']==p['initial_reference']['model_state_sha256']and proof['factory_full_RNG_sha256']==p['initial_reference']['rng_sha256']
        assert proof['new_order_cold_resume_parity']=='NOT_MEASURED'


@pytest.mark.parametrize('truth_matches',(False,True))
def test_block_order_full_synthetic_loop_keeps_batches_scoring_and_factory(monkeypatch,tmp_path,truth_matches):
    import numpy as np
    plan0,binding,runtime,events,_,_=fake_recovery(monkeypatch,'reference')
    original=plan0['metadata']['original'];batches=owner.core.all_score_batches(np.arange(20727,dtype=np.int64),96);source=np.arange(20727,dtype=np.int64);issues=source+100
    schedule={k:copy.deepcopy(v)for k,v in original['schedule'].items()if k!='contract_sha256'}
    schedule['batch_source_rows_sha256']=[owner.core.digest(source[take])for take in batches]
    schedule['batch_issue_ns_sha256']=[owner.core.digest(issues[take])for take in batches]
    original['schedule']=owner.train.seal(schedule);order=owner.block_order_schedule(original)
    context={'normalization':{},'input_binding':{},'prior':{},'target_contract':original['metadata']['target_contract'],'batches':batches,'source_rows':source,'truth':{'issue_ns':issues},'truth_sha256':'synthetic-truth','schedule':original['schedule']}
    monkeypatch.setattr(owner,'development_nn_context',lambda p:context)
    limits=owner.development_limits('fit');limits.update(model_forwards=597,backbone_forwards=597,readout_forwards=597,logit_archive_writes=2,logit_archive_readbacks=2)
    plan={'metadata':{'original':original},'schedule':order,'limits':limits,'checkpoint_cursors':[1,432],'panel_indices':[0,107,214],'execution':original['execution'],'initial_reuse':{'INITIAL_score':plan0['inputs']['original_INITIAL_score'],'truth_sha256':'synthetic-truth'if truth_matches else'wrong-truth'}}
    if not truth_matches:
        with pytest.raises(ValueError,match='ACTUAL_INITIAL_TRUTH'):owner.block_order_physical(plan,binding,tmp_path,runtime)
        assert not events['steps']and not events['checkpoints']and not events['loads']
        return
    report=owner.block_order_physical(plan,binding,tmp_path,runtime)
    assert report['counts']==limits and not events['loads']
    assert len(events['steps'])==432 and events['scores']==['PASS1','PASS2']
    for j,(index,take)in enumerate(events['steps']):
        assert index==j and np.array_equal(take,batches[order['logical_original_batch_ids'][j]])
    assert [name for name,_ in events['checkpoints']]==['CURSOR001.pt','FINAL.pt']
    assert report['checkpoint_order_cursors']['1']['next_original_batch']['original_batch_id']==order['logical_original_batch_ids'][1]
    assert report['checkpoint_order_cursors']['432']['next_original_batch']is None
    assert report['scores']['INITIAL']==plan0['inputs']['original_INITIAL_score']
    initial=events['json'][0][1];assert initial['new_INITIAL_forwards']==initial['old_weight_checkpoint_loads']==0
    assert report['new_order_cold_resume_parity']=='NOT_MEASURED'


@pytest.mark.parametrize('admitted',(False,True))
@pytest.mark.parametrize('field',('remaining_invocations','execution_authorized','training_authorized'))
def test_block_order_typed_gate_and_only_exact_training_flags(admitted,field):
    binding={'path':'/new/order.json','sha256':'1'*64};gate=owner.block_order_gate(binding,admitted);owner.block_order_require_gate(gate,binding,admitted)
    p={k:(admitted and k in owner.DEV_FLAGS[:2])for k in owner.DEV_FLAGS};owner.block_order_flags(p,admitted)
    p['full_epoch_training_allowed']=True
    with pytest.raises(ValueError):owner.block_order_flags(p,admitted)
    gate[field]=bool(gate[field])if field=='remaining_invocations'else int(gate[field])
    with pytest.raises(ValueError):owner.block_order_require_gate(gate,binding,admitted)

def test_block_recovery_all_legacy_leaf_bridges_remain_exact():
    assert owner.block_recovery_source_bridge()['all_old_numeric_order_and_recovery_paths_unchanged']
    assert owner.block_order_source_bridge()['all_old_numeric_functions_unchanged']
    assert owner.direction_gradient_source_bridge()['old_numeric_and_recovery_functions_unchanged']
    assert owner.recovery_source_bridge()['original_development_nn_and_numeric_paths_unchanged']
    assert owner.development_source_bridge()['baseline74_numeric_functions_changed']==0


def fake_block_recovery(monkeypatch,lane):
    import numpy as np
    import torch
    from gx1.contracts import market_state_ta_integrated_model_v1 as factory
    p0,_,runtime,events,old_receipt,_=fake_recovery(monkeypatch,'reference',restore_lane='block_order')
    nn=p0['metadata']['original'];batches=owner.core.all_score_batches(np.arange(20727,dtype=np.int64),96);source=np.arange(20727,dtype=np.int64);issues=source+100
    canonical={k:copy.deepcopy(v)for k,v in nn['schedule'].items()if k!='contract_sha256'}
    canonical['batch_source_rows_sha256']=[owner.core.digest(source[take])for take in batches];canonical['batch_issue_ns_sha256']=[owner.core.digest(issues[take])for take in batches];nn['schedule']=owner.train.seal(canonical)
    schedule=owner.block_order_schedule(nn);trajectory=owner.block_order_trajectory(nn,schedule);original_ref={'path':'/original/order.json','sha256':'a'*64};initial=p0['inputs']['original_INITIAL_score']
    original={'metadata':{'original':nn,'trajectory':trajectory},'schedule':schedule,'recipe':nn['recipe'],'model_contract':nn['model_contract'],'initial_reference':nn['initial_reference'],'panel_indices':[0,107,214],'initial_reuse':{'truth_sha256':'synthetic-truth'}}
    context={'normalization':{},'input_binding':{},'prior':{},'target_contract':nn['metadata']['target_contract'],'batches':batches,'source_rows':source,'truth':{'issue_ns':issues},'truth_sha256':'synthetic-truth','schedule':nn['schedule']}
    def context_loader(p):
        assert p is nn
        return context
    monkeypatch.setattr(owner,'development_nn_context',context_loader)
    model=factory.build_native_future_state_model();optimizer=torch.optim.AdamW(model.parameters());owner.core.future_cuda_seed_and_runtime({})
    steps=[];dummy={k:0 for k in owner.development_limits('fit')}
    def make(cursor):
        return owner.train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=original_ref,lane='block_order',cursor=cursor,steps=copy.deepcopy(steps),scores={'INITIAL':initial},producer_runtime=runtime)
    for j in range(2):
        record=owner.train.training_step(model,optimizer,{'take':batches[schedule['logical_original_batch_ids'][j]]},nn,dummy,j);record['order_binding']=owner.block_order_step_binding(schedule,j);steps.append(record)
        if j==0:c1=make(1)
    reference={'proof':owner.train.checkpoint_proof(make(2))};receipt={'producer_runtime':runtime,'proof':owner.train.checkpoint_proof(c1)};old_receipt['proof']=receipt['proof']
    binding={'path':'/new/'+lane+'.json','sha256':'b'*64};cp_ref=p0['inputs']['original_checkpoint'];plan={'lane':lane,'metadata':{'original':original,'receipt':receipt,'trajectory':trajectory},'original_plan':original_ref,'checkpoint_receipt':{'path':'/original/CURSOR001.json','sha256':'c'*64},'inputs':{'original_INITIAL_score':initial,'original_checkpoint':cp_ref,'reference_cursor2_receipt':{'path':'/reference/C2.json','sha256':'d'*64}},'limits':owner.recovery_limits(lane,nn),'execution':nn['execution'],'starting_cursor':0 if lane=='reference'else 1,'ending_cursor':432 if lane=='recovery'else 2}
    monkeypatch.setattr(torch,'load',lambda *a,**k:events['loads'].append(a[0])or copy.deepcopy(c1));monkeypatch.setattr(owner,'read',lambda r:reference)
    for values in events.values():values.clear()
    return plan,binding,runtime,events,receipt,reference,c1,batches


@pytest.mark.parametrize('lane',('reference','cold','recovery'))
def test_block_recovery_synthetic_roles_preserve_exact_order_and_initializer(monkeypatch,tmp_path,lane):
    plan,binding,runtime,events,receipt,reference,c1,batches=fake_block_recovery(monkeypatch,lane)
    report=owner.block_recovery_physical(plan,binding,tmp_path,runtime);original=plan['metadata']['original'];ids=original['schedule']['logical_original_batch_ids']
    assert report['counts']==plan['limits']and report['scores']['INITIAL']==plan['inputs']['original_INITIAL_score']
    assert len(events['loads'])==(0 if lane=='reference'else 1)
    start=0 if lane=='reference'else 1;end=432 if lane=='recovery'else 2
    assert len(events['steps'])==end-start
    for j,(index,take)in zip(range(start,end),events['steps']):assert index==j and __import__('numpy').array_equal(take,batches[ids[j]])
    name,cp=events['checkpoints'][-1]
    assert cp['producer']=={'original_plan':binding if lane=='reference'else plan['original_plan'],'lane':lane,'runtime':runtime}
    assert cp['trajectory']==plan['metadata']['trajectory']and len(cp['steps'])==end
    assert name==('REFERENCE_CURSOR002.pt'if lane=='reference'else'REPLAYED_CURSOR002.pt'if lane=='cold'else'FINAL.pt')
    if lane=='reference':
        assert report['original_C1_all10_equal']and owner.train.checkpoint_proof(cp)==reference['proof']
        assert events['json'][0][1]['actual_producer']['original_plan']==binding
    if lane=='cold':assert report['all10_proofs_equal']and not events['scores']
    if lane=='recovery':assert events['scores']==['PASS1','PASS2']and report['new_optimizer_updates']==431 and report['final_order_cursor']['next_original_batch']is None


def test_block_recovery_reference_mismatch_stops_before_batch131(monkeypatch,tmp_path):
    plan,binding,runtime,events,receipt,*_=fake_block_recovery(monkeypatch,'reference');receipt['proof']['steps']='wrong'
    with pytest.raises(ValueError,match='REFERENCE_C1_ALL10'):owner.block_recovery_physical(plan,binding,tmp_path,runtime)
    assert len(events['steps'])==1 and not events['checkpoints']and not events['json']


def test_block_recovery_cold_mismatch_does_not_publish_C2(monkeypatch,tmp_path):
    plan,binding,runtime,events,receipt,reference,*_=fake_block_recovery(monkeypatch,'cold');reference['proof']['optimizer_state']='wrong'
    with pytest.raises(ValueError,match='COLD_C2_ALL10'):owner.block_recovery_physical(plan,binding,tmp_path,runtime)
    assert len(events['steps'])==1 and not events['checkpoints']


@pytest.mark.parametrize('bad',('lane','original_plan','runtime','trajectory','order_binding','scores'))
def test_block_recovery_rejects_old_or_forged_C1_before_generic_restore(monkeypatch,bad):
    plan,binding,runtime,events,receipt,reference,cp,_=fake_block_recovery(monkeypatch,'cold')
    if bad in('lane','original_plan','runtime'):cp['producer'][bad]='wrong'
    if bad=='trajectory':cp['trajectory']=owner.development_trajectory(plan['metadata']['original']['metadata']['original'])
    if bad=='order_binding':cp['steps'][0]['order_binding']['original_batch_id']=0
    if bad=='scores':cp['scores']={'INITIAL':{'path':'/new/INITIAL.json','sha256':'e'*64}}
    copied=[];monkeypatch.setattr(owner.train,'strict_restore',lambda *a,**k:copied.append(True))
    with pytest.raises(ValueError):owner.block_recovery_restore(None,None,cp,plan,runtime)
    assert not copied


@pytest.mark.parametrize('lane',('reference','cold','recovery'))
@pytest.mark.parametrize('bad',(False,True))
def test_block_recovery_snapshot_requires_typed_one_use_and_exact_role(monkeypatch,lane,bad):
    binding={'path':'/new/'+lane+'.json','sha256':'1'*64};protocol={'path':'/new/protocol.json','sha256':'2'*64};sr={'path':str(owner.block_recovery_snapshot_path(lane)),'sha256':'3'*64};pref={'path':str(owner.POLICY),'sha256':'3'*64}
    p={k:(k in owner.DEV_FLAGS[:2])for k in owner.DEV_FLAGS};p[owner.BLOCK_RECOVERY_STAGE]={'execution_authorized':True,'active_lane':lane,'protocol':protocol,'design':owner.BLOCK_RECOVERY_DESIGN,'main_fit_authorized':False,'test_authorized':False,'spending_authorized':False,'lane_gates':{lane:owner.block_recovery_gate(binding,lane,True)}}
    if bad:p[owner.BLOCK_RECOVERY_STAGE]['lane_gates'][lane]['remaining_invocations']=True
    monkeypatch.setattr(owner,'ref',lambda path:sr);monkeypatch.setattr(owner,'read',lambda ref:p)
    if bad:
        with pytest.raises(ValueError):owner.block_recovery_admission_snapshot(lane,binding,protocol,pref)
    else:assert owner.block_recovery_admission_snapshot(lane,binding,protocol,pref)==sr


def issue_mix_fixture():
    control=json.loads((owner.REPO/'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001.json').read_text())
    design=json.loads(Path(owner.ISSUE_MIX_DESIGN['path']).read_text())
    return control,design


def test_issue_mix_all_legacy_bridges_are_exact():
    assert owner.issue_mix_source_bridge()['all_old_numeric_context_scoring_and_restore_owners_unchanged']
    assert owner.block_recovery_source_bridge()['all_old_numeric_order_and_recovery_paths_unchanged']
    assert owner.block_order_source_bridge()['all_old_numeric_functions_unchanged']
    assert owner.direction_gradient_source_bridge()['old_numeric_and_recovery_functions_unchanged']
    assert owner.recovery_source_bridge()['original_development_nn_and_numeric_paths_unchanged']
    assert owner.development_source_bridge()['baseline74_numeric_functions_changed']==0


def test_issue_mix_private_membership_bijections_and_original_step_cardinalities():
    import numpy as np
    import random
    import torch
    control,design=issue_mix_fixture();before=owner.core.state_sha({'np':np.random.get_state(),'python':random.getstate(),'torch':torch.get_rng_state()})
    s=owner.issue_mix_schedule(control,design)
    assert owner.core.state_sha({'np':np.random.get_state(),'python':random.getstate(),'torch':torch.get_rng_state()})==before
    assert [i for i,n in enumerate(s['logical_step_rows'])if n==87]==[111,428]
    assert s['raw_permutation_state_sha256_by_pass']==design['ordering']['raw_permutation_state_sha256_by_pass']
    for p in range(2):
        parts=s['cohort_takes'][p*216:(p+1)*216]
        assert sorted(sum(parts,[]))==list(range(20727))
        assert all(all(b>a for a,b in zip(t,t[1:]))for t in parts)
        assert owner.core.state_sha(np.concatenate([np.asarray(t,dtype=np.int64)for t in parts]))==design['ordering']['actual_sorted_partition_stream_state_sha256_by_pass'][p]
    assert [len(t)for t in s['cohort_takes']]==design['ordering']['logical_step_rows']
    assert owner.issue_mix_validate_schedule(s,control,design)==s


@pytest.mark.parametrize('bad',('duplicate','bool','unsorted','cross_batch_swap','87_step','take_hash','raw_hash','stream_hash','control_schedule','old_schedule'))
def test_issue_mix_changed_membership_or_binding_is_rejected(bad):
    control,design=issue_mix_fixture();s=owner.issue_mix_schedule(control,design)
    if bad=='duplicate':s['cohort_takes'][0][0]=s['cohort_takes'][0][1]
    if bad=='bool':s['cohort_takes'][0][0]=True
    if bad=='unsorted':s['cohort_takes'][0].reverse()
    if bad=='cross_batch_swap':s['cohort_takes'][0][0],s['cohort_takes'][1][0]=s['cohort_takes'][1][0],s['cohort_takes'][0][0]
    if bad=='87_step':s['logical_step_rows'][111],s['logical_step_rows'][215]=s['logical_step_rows'][215],s['logical_step_rows'][111]
    if bad=='take_hash':s['cohort_take_state_sha256_by_logical_batch'][0]='0'*64
    if bad=='raw_hash':s['raw_permutation_state_sha256_by_pass'][0]='0'*64
    if bad=='stream_hash':s['actual_sorted_partition_stream_state_sha256_by_pass'][0]='0'*64
    if bad=='control_schedule':s['control_schedule_sha256']='0'*64
    if bad=='old_schedule':s=control['schedule']
    with pytest.raises(ValueError):owner.issue_mix_validate_schedule(s,control,design)


def test_issue_mix_role_limits_keep_exact_numeric_budget():
    p=original_fit()
    for lane,updates,forwards,archives,loads in [('reference',2,2,2,0),('cold',1,1,1,1),('fit',432,597,2,0)]:
        limits=owner.issue_mix_limits(lane,p)
        assert set(limits)==set(p['limits'])
        assert limits['optimizer_steps']==limits['backward_calls']==updates
        assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==forwards
        assert limits['checkpoint_writes']==limits['checkpoint_readbacks']==limits['checkpoint_receipt_writes']==limits['checkpoint_prepublication_loads']==archives
        assert limits['checkpoint_loads']==limits['model_restores']==loads
        assert limits['normalization_fits']==limits['prior_fits']==limits['skipped_batches']==0


def fake_issue_mix(monkeypatch,lane):
    import numpy as np
    import torch
    from gx1.contracts import market_state_ta_integrated_model_v1 as factory
    p0,_,runtime,events,_,_=fake_recovery(monkeypatch,'reference',restore_lane='reference');nn=p0['metadata']['original'];control,design=issue_mix_fixture();control=copy.deepcopy(control);control['metadata']['original']=nn
    source=np.arange(20727,dtype=np.int64);issues=source+100;batches=owner.core.all_score_batches(source,96)
    canonical={k:copy.deepcopy(v)for k,v in nn['schedule'].items()if k!='contract_sha256'};canonical['batch_source_rows_sha256']=[owner.core.digest(source[t])for t in batches];canonical['batch_issue_ns_sha256']=[owner.core.digest(issues[t])for t in batches];canonical['source_rows_sha256']=owner.core.digest(source);canonical['issue_ns_sha256']=owner.core.digest(issues);nn['schedule']=owner.train.seal(canonical);control['schedule']=owner.block_order_schedule(nn);design['ordering']['control_schedule_sha256']=control['schedule']['contract_sha256'];schedule=owner.issue_mix_schedule(control,design);control_ref={'path':'/original/order.json','sha256':'a'*64};trajectory=owner.issue_mix_trajectory(nn,schedule,control_ref)
    context={'normalization':{},'input_binding':{},'prior':{},'target_contract':nn['metadata']['target_contract'],'batches':batches,'source_rows':source,'truth':{'issue_ns':issues},'truth_sha256':'synthetic-truth','schedule':nn['schedule']}
    stream=owner.train.seal({'schedule_sha256':schedule['contract_sha256'],'batches':[{'logical_index':j,'rows':len(t),'cohort_take_state_sha256':owner.core.state_sha(np.asarray(t,dtype=np.int64)),'source_rows_sha256':owner.core.digest(source[t]),'issue_ns_sha256':owner.core.digest(issues[t]),'truth_sha256':str(j),'known_activity_cells':len(t)*5}for j,t in enumerate(schedule['cohort_takes'])]})
    reference_binding={'path':'/new/reference.json','sha256':'b'*64};binding=reference_binding if lane=='reference'else{'path':'/new/'+lane+'.json','sha256':'c'*64};initial=p0['inputs']['original_INITIAL_score'];cp_ref=p0['inputs']['original_checkpoint'];c1_ref={'path':'/new/REFERENCE_CURSOR001.json','sha256':'d'*64};c2_ref={'path':'/new/REFERENCE_CURSOR002.json','sha256':'e'*64};sr={'path':'/new/STREAM_BINDING.json','sha256':'f'*64}
    plan={'lane':lane,'metadata':{'original':nn,'control':control,'trajectory':trajectory},'original_plan':control_ref,'schedule':schedule,'parents':{}if lane=='reference'else{'reference':{'plan':reference_binding}},'inputs':{'reference_checkpoint':cp_ref,'reference_cursor1_receipt':c1_ref,'reference_cursor2_receipt':c2_ref,'reference_stream_binding':sr},'initial_reuse':{'INITIAL_score':initial,'truth_sha256':'synthetic-truth'},'limits':owner.issue_mix_limits(lane,nn),'execution':nn['execution'],'panel_indices':[0,107,214],'starting_cursor':1 if lane=='cold'else 0,'ending_cursor':432 if lane=='fit'else 2}
    monkeypatch.setattr(owner,'development_nn_context',lambda p:context);monkeypatch.setattr(owner,'issue_mix_actual_stream',lambda c,p:stream)
    model=factory.build_native_future_state_model();opt=torch.optim.AdamW(model.parameters());owner.core.future_cuda_seed_and_runtime({});steps=[];dummy=dict.fromkeys(plan['limits'],0)
    for j in range(2):
        rec=owner.train.training_step(model,opt,{'take':np.asarray(schedule['cohort_takes'][j],dtype=np.int64)},nn,dummy,j);rec['issue_mix_binding']=owner.issue_mix_step_binding(schedule,stream,j);steps.append(rec)
        cp=owner.train.make_checkpoint(model,opt,trajectory=trajectory,original_plan_ref=reference_binding,lane='reference',cursor=j+1,steps=copy.deepcopy(steps),scores={'INITIAL':initial},producer_runtime=runtime)
        if j==0:c1=cp
    c1_receipt={'proof':owner.train.checkpoint_proof(c1),'producer_runtime':runtime};c2_receipt={'proof':owner.train.checkpoint_proof(cp)}
    def read(binding):
        if binding==owner.ISSUE_MIX_DESIGN:return design
        if binding==c1_ref:return c1_receipt
        if binding==c2_ref:return c2_receipt
        if binding==sr:return stream
        raise AssertionError(binding)
    monkeypatch.setattr(owner,'read',read);monkeypatch.setattr(torch,'load',lambda *a,**k:events['loads'].append(a[0])or copy.deepcopy(c1))
    def restore(m,opt,c,**kwargs):
        assert kwargs['original_plan_ref']==reference_binding and kwargs['expected_state_proof']==c1_receipt['proof']
        m.value=c['model_state']['weight'];opt.value=c['optimizer_state']['moment']
        # The original synthetic RNG dictionary is owned by the core RNG closure.
        rng=owner.core.training_rng.__closure__[0].cell_contents;rng.update(c['rng_state']);return c['position']
    monkeypatch.setattr(owner.train,'strict_restore',restore)
    for v in events.values():v.clear()
    return plan,binding,runtime,events,c1,c1_receipt,c2_receipt,stream,context


@pytest.mark.parametrize('lane',('reference','cold','fit'))
def test_issue_mix_synthetic_roles_keep_factory_stream_scores_and_checkpoint_counts(monkeypatch,tmp_path,lane):
    import numpy as np
    plan,binding,runtime,events,c1,r1,r2,stream,context=fake_issue_mix(monkeypatch,lane);report=owner.issue_mix_physical(plan,binding,tmp_path,runtime)
    assert report['counts']==plan['limits']and report['scores']['INITIAL']==plan['initial_reuse']['INITIAL_score']
    assert len(events['loads'])==(1 if lane=='cold'else 0)
    assert len(events['steps'])==plan['ending_cursor']-plan['starting_cursor']
    for j,(index,take)in zip(range(plan['starting_cursor'],plan['ending_cursor']),events['steps']):assert index==j and np.array_equal(take,np.asarray(plan['schedule']['cohort_takes'][j],dtype=np.int64))
    assert [n for n,_ in events['checkpoints']]==(['REFERENCE_CURSOR001.pt','REFERENCE_CURSOR002.pt']if lane=='reference'else['REPLAYED_CURSOR002.pt']if lane=='cold'else['CURSOR001.pt','FINAL.pt'])
    assert events['scores']==(['PASS1','PASS2']if lane=='fit'else[])
    if lane=='cold':assert report['all10_proofs_equal']and report['cold_replay_bit_exact']and report['full_cursor2_proof']==r2['proof']
    if lane=='reference':assert report['full_cursor2_proof']==r2['proof']
    if lane=='fit':assert report['ending_cursor']==report['new_optimizer_updates']==432 and report['final_order_cursor']['next_issue_batch']is None


@pytest.mark.parametrize('bad',('lane','original_plan','runtime','trajectory','order_binding','scores','stream'))
def test_issue_mix_cold_wrong_lineage_rejected_before_copy(monkeypatch,bad):
    plan,binding,runtime,events,cp,r1,r2,stream,context=fake_issue_mix(monkeypatch,'cold')
    if bad in('lane','original_plan','runtime'):cp['producer'][bad]='wrong'
    if bad=='trajectory':cp['trajectory']=owner.development_trajectory(plan['metadata']['original'])
    if bad=='order_binding':cp['steps'][0]['issue_mix_binding']['cohort_take_state_sha256']='wrong'
    if bad=='scores':cp['scores']={'INITIAL':{'path':'/forged/INITIAL.json','sha256':'0'*64}}
    if bad=='stream':stream['batches'][0]['truth_sha256']='wrong'
    copied=[];monkeypatch.setattr(owner.train,'strict_restore',lambda *a,**k:copied.append(True))
    with pytest.raises(ValueError):owner.issue_mix_restore(None,None,cp,plan,runtime,stream)
    assert not copied


def test_issue_mix_cold_C2_mismatch_never_publishes_replay(monkeypatch,tmp_path):
    plan,binding,runtime,events,cp,r1,r2,*_=fake_issue_mix(monkeypatch,'cold');r2['proof']['optimizer_state']='wrong'
    with pytest.raises(ValueError,match='COLD_C2_ALL10'):owner.issue_mix_physical(plan,binding,tmp_path,runtime)
    assert len(events['steps'])==1 and not events['checkpoints']


def test_issue_mix_actual_stream_binds_same_source_issue_and_truth_slices(monkeypatch):
    import numpy as np
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    control,design=issue_mix_fixture();nn=control['metadata']['original'];source=np.arange(20727,dtype=np.int64);issues=source+100;s=owner.issue_mix_schedule(control,design);s['source_rows_sha256']=owner.core.digest(source);s['issue_ns_sha256']=owner.core.digest(issues);s=owner.train.seal({k:v for k,v in s.items()if k!='contract_sha256'})
    context={'source_rows':source,'truth':{'issue_ns':issues,'label':source%2},'truth_sha256':'truth','schedule':nn['schedule'],'target_contract':{}}
    plan={'schedule':s,'metadata':{'original':nn},'initial_reuse':{'truth_sha256':'truth'}};seen=[]
    def slice_truth(truth,take,**kwargs):seen.append(take.copy());return {k:v[take].copy()for k,v in truth.items()}
    monkeypatch.setattr(owner.train,'slice_provider_hlc_truth',slice_truth);monkeypatch.setattr(objective,'_native_future_state_truth',lambda t,**k:{'activity':np.ones((len(t['issue_ns']),5),dtype=np.bool_)});monkeypatch.setattr(objective,'native_future_state_truth_sha256',lambda t,**k:owner.core.state_sha(t))
    stream=owner.issue_mix_actual_stream(context,plan)
    assert len(seen)==432
    for j,take in enumerate(seen):
        record=stream['batches'][j];assert np.array_equal(take,np.asarray(s['cohort_takes'][j],dtype=np.int64))
        assert record['source_rows_sha256']==owner.core.digest(source[take])and record['issue_ns_sha256']==owner.core.digest(issues[take])
        assert record['truth_sha256']==owner.core.state_sha({k:v[take]for k,v in context['truth'].items()})
    context['truth']['label'][0]^=1
    assert owner.issue_mix_actual_stream(context,plan)['ordered_truth_binding_sha256']!=stream['ordered_truth_binding_sha256']
    context['truth_sha256']='wrong'
    with pytest.raises(ValueError,match='ACTUAL_COHORT_IDENTITY'):owner.issue_mix_actual_stream(context,plan)


@pytest.mark.parametrize('lane',('reference','cold','fit'))
@pytest.mark.parametrize('bad',(False,True))
def test_issue_mix_snapshot_preserves_exact_role_and_strict_gate_types(monkeypatch,lane,bad):
    binding={'path':'/new/'+lane+'.json','sha256':'1'*64};pr={'path':'/new/protocol.json','sha256':'2'*64};sr={'path':str(owner.issue_mix_snapshot_path(lane)),'sha256':'3'*64};pref={'path':str(owner.POLICY),'sha256':'3'*64};control,design=issue_mix_fixture()
    p={k:(k in owner.DEV_FLAGS[:2])for k in owner.DEV_FLAGS};p[owner.ISSUE_MIX_STAGE]={'execution_authorized':True,'active_lane':lane,'protocol':pr,'design':owner.ISSUE_MIX_DESIGN,'original_plan':design['evidence']['original_plan'],'main_fit_authorized':False,'test_authorized':False,'spending_authorized':False,'lane_gates':{lane:owner.issue_mix_gate(binding,lane,True)}}
    if bad:p[owner.ISSUE_MIX_STAGE]['lane_gates'][lane]['remaining_invocations']=True
    monkeypatch.setattr(owner,'ref',lambda path:sr);monkeypatch.setattr(owner,'read',lambda binding:design if binding==owner.ISSUE_MIX_DESIGN else p)
    if bad:
        with pytest.raises(ValueError):owner.issue_mix_admission_snapshot(lane,binding,pr,pref)
    else:assert owner.issue_mix_admission_snapshot(lane,binding,pr,pref)==sr
