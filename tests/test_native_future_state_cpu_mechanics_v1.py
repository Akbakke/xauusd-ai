"""Synthetic CPU optimizer/RNG/checkpoint mechanics, never genuine GX1 inputs."""
import copy
import io
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gx1.audit import qualify_native_m5_full_gc_direction_v1 as owner
from gx1.contracts.market_state_native_m5_targets_v1 import fully_future_state_contract
from gx1.contracts.market_state_ta_integrated_model_v1 import NativeFutureStateReadout
from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_future_state_train_prior


class TinyCPUStateModel(nn.Module):
    def __init__(self,prior,target):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(2,2),nn.Dropout(.05))
        self.readout=NativeFutureStateReadout(d_model=2,target_contract=target,train_prior=prior)

    def forward(self,**features):
        return self.readout(self.backbone(features['seq_x']))


def packet():
    target=fully_future_state_contract();tick=300_000_000_000
    issues=np.arange(1,65,dtype=np.int64)*tick
    duration=np.array([target['tf_duration_ns'][tf]for tf in target['timeframes']],np.int64)
    origin=np.array([target['tf_origin_ns'][tf]for tf in target['timeframes']],np.int64)
    starts=((issues[:,None]-origin+duration-1)//duration)*duration+origin
    complete=np.ones((64,5),np.bool_)
    active=np.broadcast_to((np.arange(64)%3!=0)[:,None],complete.shape).copy()
    side=active&np.broadcast_to((np.arange(64)%5!=0)[:,None],complete.shape)
    up=side&np.broadcast_to((np.arange(64)%2==0)[:,None],complete.shape)
    truth={'issue_ns':issues,'target_start_ns':starts,'validity_ns':starts+duration,
        'target_present':complete.copy(),'mature':complete.copy(),'nominal_complete':complete.copy(),
        'state_window_complete':complete.copy(),'partial_range_active':np.zeros_like(complete),
        'activity_known':complete.copy(),'activity':active,'side_known':side,'side_up':up}
    prior=fit_native_future_state_train_prior(truth=truth,fold_upper_ns=owner.FUTURE_CPU_TRAIN_UPPER,target_contract=target)
    recipe={'optimizer':copy.deepcopy(owner.ADAMW_RECIPE),'gradient_clip_norm':1.,'scheduler':None,
        'seed':20260911,'batch_rows':16,'logical_updates':4,'checkpoint_cursor':3}
    plan={'metadata':{'target_contract':target},'recipe':recipe}
    values=torch.linspace(-1.,1.,64*5*2).reshape(64,5,2)
    packets=[{'features':{'seq_x':values[i:i+16].clone()},'truth':{k:v[i:i+16].copy()for k,v in truth.items()}}
             for i in range(0,64,16)]
    torch.manual_seed(recipe['seed']);random.seed(recipe['seed']);np.random.seed(recipe['seed'])
    model=TinyCPUStateModel(prior,target)
    options={**recipe['optimizer'],'betas':tuple(recipe['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options)
    return plan,prior,packets,model,optimizer


def zeros(lane='continuous'):
    return dict.fromkeys(owner.FUTURE_CPU_LIMITS[lane],0)


def reference():
    return {'path':'/synthetic/mechanics.json','sha256':'a'*64}


def test_exact_two_lane_budget_is_five_physical_updates_not_eight():
    assert sum(v['optimizer_steps']for v in owner.FUTURE_CPU_LIMITS.values())==5
    assert sum(v['model_constructions']for v in owner.FUTURE_CPU_LIMITS.values())==2
    assert owner.FUTURE_CPU_LIMITS['continuous']['model_forwards']==13
    assert owner.FUTURE_CPU_LIMITS['replay']['model_forwards']==9
    assert owner.FUTURE_CPU_LIMITS['continuous']['autograd_calls']==3
    assert owner.FUTURE_CPU_LIMITS['replay']['checkpoint_loads']==1


def test_cpu_rng_weights_only_serialization_has_no_numpy_pickle_allowlist():
    before=owner.future_cpu_rng();stream=io.BytesIO();torch.save(before,stream)
    loaded=torch.load(io.BytesIO(stream.getvalue()),weights_only=True,map_location='cpu')
    assert owner.state_sha(before)==owner.state_sha(loaded)
    expected=(random.random(),np.random.random(),torch.rand(3))
    owner.future_cpu_restore_rng(loaded)
    actual=(random.random(),np.random.random(),torch.rand(3))
    assert actual[:2]==expected[:2]and torch.equal(actual[2],expected[2])
    owner.future_cpu_restore_rng(before)


def test_initial_all_four_batches_are_exact_prior_and_scores_preserve_rng_state():
    plan,prior,packets,model,opt=packet()
    rng=owner.state_sha(owner.future_cpu_rng());state=owner.state_sha(model.state_dict())
    result=owner.future_cpu_score(model,packets,plan,prior,initial=True)
    assert result['rows']==64 and len(result['batches'])==4 and result['initial_prior_exact']
    assert owner.state_sha(owner.future_cpu_rng())==rng
    assert owner.state_sha(model.state_dict())==state and model.training
    assert all(p.grad is None for p in model.parameters())


def test_first_data_update_and_post_update_shared_gradient_are_real_and_probe_is_read_only():
    plan,prior,packets,model,opt=packet();counts=zeros()
    report=owner.future_cpu_step(model,opt,packets[0],plan,counts,0)
    assert report['backbone_data_gradient_l2']==0 and report['readout_data_gradient_l2']>0
    assert report['backbone_actual_parameter_delta_l2']>=0
    state=owner.state_sha(model.state_dict());rng=owner.state_sha(owner.future_cpu_rng())
    probe=owner.future_cpu_gradient_probe(model,packets[0],plan,counts)
    assert counts['autograd_calls']==3
    assert probe['denominator_known_activity_cells']==80
    assert probe['M5_vs_auxiliary']['g5_l2']>0
    assert probe['M5_vs_auxiliary']['gaux_l2']>0
    assert owner.state_sha(model.state_dict())==state and owner.state_sha(owner.future_cpu_rng())==rng
    assert all(p.grad is None for p in model.parameters())


def test_cursor_three_weights_only_cold_restore_duplicates_only_fourth_update(tmp_path):
    plan,prior,packets,model,opt=packet();counts=zeros();steps=[]
    order=[{'source_rows_sha256':str(i)*64,'issue_ns_sha256':str(i+1)*64}for i in range(4)]
    for i in range(3):steps.append(owner.future_cpu_step(model,opt,packets[i],plan,counts,i))
    cp=owner.future_cpu_checkpoint(model,opt,plan,reference(),order,3,steps)
    binding=owner.future_cpu_publish_checkpoint(tmp_path/'cursor3.pt',cp,counts)
    loaded=torch.load(binding['path'],weights_only=True,map_location='cpu')
    assert owner.future_cpu_proof(loaded)==owner.future_cpu_proof(cp)
    from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError
    with pytest.raises(ImmutableEventAuthorityError):owner.future_cpu_publish_checkpoint(tmp_path/'cursor3.pt',cp,counts)
    steps.append(owner.future_cpu_step(model,opt,packets[3],plan,counts,3))
    final=owner.future_cpu_checkpoint(model,opt,plan,reference(),order,4,steps)
    # Different temporary initialization must be completely replaced by cursor3.
    torch.manual_seed(123)
    second=TinyCPUStateModel(prior,plan['metadata']['target_contract'])
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    second_opt=torch.optim.AdamW(second.parameters(),**options)
    owner.future_cpu_restore(second,second_opt,loaded,plan,reference(),order)
    replay_steps=list(loaded['steps']);replay_counts=zeros('replay')
    replay_steps.append(owner.future_cpu_step(second,second_opt,packets[3],plan,replay_counts,3))
    replay=owner.future_cpu_checkpoint(second,second_opt,plan,reference(),order,4,replay_steps)
    assert owner.future_cpu_proof(replay)==owner.future_cpu_proof(final)
    assert replay_counts['optimizer_steps']==replay_counts['backward_calls']==1


@pytest.mark.parametrize('bad',['cursor','scheduler','names','batch_order','optimizer_group','optimizer_tensor','rng'])
def test_bad_checkpoint_rejected_before_model_parameter_copy(bad):
    plan,prior,packets,model,opt=packet();counts=zeros();steps=[];order=[{'step':i}for i in range(4)]
    for i in range(3):steps.append(owner.future_cpu_step(model,opt,packets[i],plan,counts,i))
    stream=io.BytesIO();torch.save(owner.future_cpu_checkpoint(model,opt,plan,reference(),order,3,steps),stream)
    cp=torch.load(io.BytesIO(stream.getvalue()),weights_only=True,map_location='cpu')
    fresh=TinyCPUStateModel(prior,plan['metadata']['target_contract'])
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    fresh_opt=torch.optim.AdamW(fresh.parameters(),**options)
    before=owner.state_sha(fresh.state_dict())
    if bad=='cursor':cp['cursor']=4
    elif bad=='scheduler':cp['scheduler_state']={}
    elif bad=='names':cp['optimizer_parameter_names'].reverse()
    elif bad=='batch_order':cp['batch_order'].reverse()
    elif bad=='optimizer_group':cp['optimizer_state']['param_groups'][0]['lr']*=2
    elif bad=='optimizer_tensor':next(iter(cp['optimizer_state']['state'].values()))['exp_avg']=torch.zeros(9)
    elif bad=='rng':cp['rng_state']['torch_cpu']=torch.zeros(1,dtype=torch.uint8)
    with pytest.raises((ValueError,RuntimeError)):
        owner.future_cpu_restore(fresh,fresh_opt,cp,plan,reference(),order)
    assert owner.state_sha(fresh.state_dict())==before


@pytest.mark.parametrize('path',['/synthetic/weights.pt','/synthetic/targets.npz','relative.json','/synthetic/../bad.json'])
def test_preflight_rejects_nonjson_or_noncanonical_references_before_io(monkeypatch,path):
    monkeypatch.setattr(owner,'read',lambda *a:(_ for _ in()).throw(AssertionError('IO reached')))
    with pytest.raises(ValueError,match='JSON_REFERENCE'):
        owner.future_cpu_read({'path':path,'sha256':'a'*64})


def test_recipe_is_exact_qualified_main_optimizer_not_new_hyperparameters():
    recipe={'optimizer':copy.deepcopy(owner.ADAMW_RECIPE),'gradient_clip_norm':1.,'scheduler':None,'seed':20260911}
    result=owner.future_cpu_recipe({'recipe':recipe})
    assert result['optimizer']==recipe['optimizer']and result['batch_rows']==16
    recipe['optimizer']['lr']*=2
    with pytest.raises(ValueError,match='QUALIFIED_RECIPE'):owner.future_cpu_recipe({'recipe':recipe})


def test_prefix_selector_has_no_target_mask_argument_and_keeps_source_order(monkeypatch):
    import inspect
    issues=owner.FUTURE_CPU_TRAIN_START+np.arange(354550,dtype=np.int64)*300_000_000_000
    provider=SimpleNamespace(issue_ns=issues,input_ready=np.ones(len(issues),np.bool_))
    monkeypatch.setattr(owner,'FUTURE_CPU_TRAIN_SHA',owner.digest(issues))
    rows,batches=owner.future_cpu_batches(provider)
    assert list(inspect.signature(owner.future_cpu_batches).parameters)==['provider']
    assert len(rows)==354550 and np.array_equal(np.concatenate(batches),np.arange(64))
    assert len(batches)==4 and all(len(b)==16 for b in batches)


def test_new_cli_rejects_any_cuda_or_unbounded_extra_argument(monkeypatch):
    monkeypatch.setattr('sys.argv',['qualifier','--native-future-state-cpu-mechanics',
        '--plan-sha256','a'*64,'--lane','continuous','--device','cuda'])
    with pytest.raises(SystemExit)as error:owner.future_cpu_cli()
    assert error.value.code==2


def test_family_gradient_partition_reuses_tuples_and_reports_empty_and_shared_groups():
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    first,second=MODEL_NATIVE_TRAINING_SPECIALISTS[:2]
    names=(first+'.weight',second+'.bias','shared_mix.weight')
    left=(torch.tensor([3.,0.]),None,torch.tensor([1.]))
    right=(torch.tensor([0.,4.]),torch.tensor([-2.]),torch.tensor([-1.]))
    groups=owner.future_cpu_family_gradients(names,left,right)['per_group']
    assert groups[first]['parameter_tensor_count']==1
    assert groups[first]['g5_l2']==3 and groups[first]['gaux_l2']==4
    assert groups[first]['gtotal_l2']==5 and groups[first]['cross_dot']==0
    assert groups[first]['gaux_to_gM5_l2_ratio']==pytest.approx(4/3)
    assert groups[second]['gaux_to_gM5_l2_ratio']is None
    shared=groups['unassigned_shared_mix']
    assert shared['parameter_tensor_count']==1 and shared['cosine']==-1
    assert shared['gtotal_l2']==0 and shared['gtotal_nonzero_parameter_tensors']==0
    assert sum(g['parameter_tensor_count']for g in groups.values())==3
    empty=groups[MODEL_NATIVE_TRAINING_SPECIALISTS[-1]]
    assert empty['parameter_tensor_count']==0 and empty['cosine']is None


def test_checkpoint_disk_corruption_is_rejected_before_deserialization(monkeypatch,tmp_path):
    plan,prior,packets,model,opt=packet()
    cp=owner.future_cpu_checkpoint(model,opt,plan,reference(),[],0,[])
    original=owner.publish
    def corrupt_after_publish(path,payload):
        binding=original(path,payload)
        path.write_bytes(b'corrupted persisted checkpoint')
        return binding
    monkeypatch.setattr(owner,'publish',corrupt_after_publish)
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in()).throw(AssertionError('deserialization reached')))
    with pytest.raises(ValueError,match='CHECKPOINT_FILE_HASH'):
        owner.future_cpu_publish_checkpoint(tmp_path/'corrupt.pt',cp,zeros())


@pytest.mark.parametrize('bad',[None,'policy','claim_path','terminal_status'])
def test_replay_metadata_gate_authenticates_same_continuous_policy_before_payload(monkeypatch,bad):
    plan_ref=reference();policy_ref={'path':'/synthetic/NEXT_RUN_POLICY.json','sha256':'b'*64}
    claim_ref={'path':str(owner.FUTURE_CPU_ROOT/'continuous/CLAIM.json'),'sha256':'c'*64}
    terminal_ref={'path':str(owner.FUTURE_CPU_ROOT/'continuous/TERMINAL.json'),'sha256':'d'*64}
    terminal={'status':'COMPLETE_CONSUMED','exit_code':0,'plan':plan_ref,'claim':claim_ref}
    claim={'plan':plan_ref,'lane':'continuous','policy_at_claim':policy_ref}
    if bad=='policy':claim['policy_at_claim']={**policy_ref,'sha256':'e'*64}
    elif bad=='claim_path':terminal['claim']={**claim_ref,'path':'/synthetic/other_CLAIM.json'}
    elif bad=='terminal_status':terminal['status']='FAILED_CONSUMED'
    monkeypatch.setattr(owner,'ref',lambda path:terminal_ref if str(path)==terminal_ref['path']else None)
    def read_json(binding):
        if binding==terminal_ref:return terminal
        assert binding==claim_ref
        return claim
    monkeypatch.setattr(owner,'future_cpu_read',read_json)
    if bad is None:owner.future_cpu_replay_admission(plan_ref,policy_ref)
    else:
        with pytest.raises(ValueError,match='FUTURE_CPU_REPLAY_PARENT_TERMINAL|FUTURE_CPU_INTERLANE_POLICY_FREEZE'):
            owner.future_cpu_replay_admission(plan_ref,policy_ref)
