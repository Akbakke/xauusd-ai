"""Same-step observer checks with tiny CPU tensors; no genuine NN/data or launch."""
import copy,json
import numpy as np
import pytest
import torch
from gx1.audit import native_future_state_step_observer_v1 as observer_owner
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective


def synthetic_step(monkeypatch):
    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__();self.side_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(4,1)for tf in observer_owner.TF});self.activity_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(4,1)for tf in observer_owner.TF})
        def forward(self,x):
            return {task+'_logits':torch.cat([heads[tf](x)for tf in observer_owner.TF],dim=1)for task,heads in(('activity',self.activity_heads),('side',self.side_heads))}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.05));self.readout=Readout()
        def forward(self,x):return self.readout(self.backbone(x))
    torch.manual_seed(271828);model=Model();features=torch.linspace(-2,2,44,dtype=torch.float32).reshape(11,4);labels=np.zeros((11,5),np.bool_);labels[::3,:]=True;activity=np.ones((11,5),np.bool_);activity[-1,3:]=False;side=activity.copy();side[:4,4]=False;side[:,2]=False;masks={'activity':activity,'side':side};truth={'activity':labels.copy(),'side_up':labels.copy()}
    monkeypatch.setattr(objective,'_native_future_state_truth',lambda *a,**k:masks);monkeypatch.setattr(objective,'native_future_state_objective_contract',lambda **k:{'contract_sha256':'synthetic'})
    monkeypatch.setattr(core,'future_cuda_forward',lambda m,f:(m(f),None,None))
    plan={'metadata':{'target_contract':{'schema_version':'synthetic_cpu_observer'}},'recipe':{'gradient_clip_norm':.03,'optimizer':{'lr':.001,'betas':(.9,.999),'weight_decay':.01}}}
    return model,features,truth,masks,plan


def execute(model,features,truth,plan,callback):
    optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts={'backward_calls':0,'optimizer_steps':0};calls={'forward':0,'backward':0,'clip':0,'step':0,'zero':0,'train':0};original_backward=torch.Tensor.backward;original_clip=torch.nn.utils.clip_grad_norm_;original_step=optimizer.step;original_zero=optimizer.zero_grad;original_train=model.train
    def after_forward(*args):calls['forward']+=1
    hook=model.register_forward_hook(after_forward)
    def backward(*a,**k):calls['backward']+=1;return original_backward(*a,**k)
    def clip(*a,**k):calls['clip']+=1;return original_clip(*a,**k)
    def step(*a,**k):calls['step']+=1;return original_step(*a,**k)
    def zero(*a,**k):calls['zero']+=1;return original_zero(*a,**k)
    def train(*a,**k):calls['train']+=1;return original_train(*a,**k)
    with pytest.MonkeyPatch.context()as patch:
        patch.setattr(torch.Tensor,'backward',backward);patch.setattr(torch.nn.utils,'clip_grad_norm_',clip);patch.setattr(optimizer,'step',step);patch.setattr(optimizer,'zero_grad',zero);patch.setattr(model,'train',train)
        result=core.future_cuda_step(model,optimizer,{'features':features,'truth':truth},plan,counts,432,observer=callback)
    hook.remove();return optimizer,counts,calls,result


def test_observer_on_off_keeps_update_rng_modes_gradients_and_original_step_return_exact(monkeypatch):
    model,features,truth,masks,plan=synthetic_step(monkeypatch);original=copy.deepcopy(model);rng=torch.get_rng_state().clone();observed=observer_owner.NativeFutureStateStepObserver();without=execute(model,features,truth,plan,None);after_off=torch.get_rng_state().clone();torch.set_rng_state(rng);with_observer=execute(original,features,truth,plan,observed);after_on=torch.get_rng_state().clone()
    assert without[1:]==with_observer[1:]
    assert with_observer[2]=={'forward':1,'backward':1,'clip':1,'step':1,'zero':2,'train':1}
    assert core.state_sha(model.state_dict())==core.state_sha(original.state_dict())and core.state_sha(without[0].state_dict())==core.state_sha(with_observer[0].state_dict())
    assert torch.equal(after_off,after_on)
    assert all(p.grad is None for p in original.parameters())and all(m.training for m in original.modules())
    assert len(observed.reports)==1 and observed._pending is None
    record=observed.reports[0];assert record['A_b']==int(masks['activity'].sum())and record['logical_step']==433
    assert all(record['per_tf'][tf]['raw_side_bias_gradient_check']['within_F32_bound']for tf in observer_owner.TF)
    assert record['per_tf']['H1']['side_known']==0 and record['per_tf']['H1']['mean_probability_UP']is None and record['per_tf']['H1']['bias_gradient_pre_clip']==0.
    assert all(record['before_optimizer_step'][tf]['AdamW']['status']=='EMPTY_UNINITIALIZED'and record['after_optimizer_step'][tf]['AdamW']['step']==1. for tf in observer_owner.TF)
    assert json.loads(json.dumps(observed.reports,allow_nan=False))==observed.reports
    assert torch.equal(torch.get_rng_state(),after_on)


def test_actual_native_F32_autograd_error_is_bounded_with_F32_not_endpoint_precision():
    bias=torch.tensor([.137],dtype=torch.float32,requires_grad=True);z=torch.linspace(-7,4,257,dtype=torch.float32)+bias;labels=(torch.arange(257)%3==0);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,labels.float(),reduction='sum')/1285;loss.backward();p=torch.sigmoid(z.detach()).numpy();check=observer_owner.f32_bias_gradient_bound(p,labels.numpy(),1285);difference=abs(float(bias.grad)-check['analytic_from_actual_F32_sigmoid'])
    assert difference<=check['absolute_error_bound']and check['unit_roundoff']==float(np.finfo(np.float32).eps)/2
    assert check['absolute_error_bound']>1e-12 and difference>1e-12


def test_observer_rejects_wrong_native_bias_gradient_before_clip(monkeypatch):
    model,features,truth,masks,plan=synthetic_step(monkeypatch);optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);observer=observer_owner.NativeFutureStateStepObserver();output=model(features);loss=objective.native_future_state_loss(model_output=output,truth=truth,fold_upper_ns=core.FUTURE_CPU_TRAIN_UPPER,target_contract=plan['metadata']['target_contract']);observer('forward_loss',model=model,optimizer=optimizer,output=output,truth=truth,loss=loss,plan=plan,index=432);loss['loss'].backward();model.readout.side_heads['M5'].bias.grad.add_(.1)
    with pytest.raises(ValueError,match='NATIVE_F32_ANALYTIC_RAW_BIAS_GRADIENT'):observer('backward',model=model,optimizer=optimizer,index=432)


def test_scalar_AdamW_snapshots_do_not_alias_later_updates_or_create_empty_state(monkeypatch):
    model,features,truth,masks,plan=synthetic_step(monkeypatch);optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);before=observer_owner.bias_and_AdamW(model,optimizer);assert not optimizer.state
    loss=objective.native_future_state_loss(model_output=model(features),truth=truth,fold_upper_ns=core.FUTURE_CPU_TRAIN_UPPER,target_contract=plan['metadata']['target_contract']);loss['loss'].backward();optimizer.step();saved=observer_owner.bias_and_AdamW(model,optimizer);frozen=copy.deepcopy(saved)
    for state in optimizer.state.values():state['exp_avg'].add_(1);state['exp_avg_sq'].add_(1);state['step'].add_(1)
    assert saved==frozen and all(before[tf]['AdamW']['status']=='EMPTY_UNINITIALIZED'for tf in observer_owner.TF)


@pytest.mark.parametrize('phase',['backward','clip','step'])
def test_phase_order_rejects_unobserved_boundary_without_touching_model(monkeypatch,phase):
    model,features,truth,masks,plan=synthetic_step(monkeypatch);optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);before=core.state_sha(model.state_dict());rng=torch.get_rng_state().clone();observer=observer_owner.NativeFutureStateStepObserver()
    with pytest.raises(ValueError,match='PHASE_ORDER'):observer(phase,model=model,optimizer=optimizer,index=432)
    assert core.state_sha(model.state_dict())==before and torch.equal(torch.get_rng_state(),rng)and not optimizer.state
