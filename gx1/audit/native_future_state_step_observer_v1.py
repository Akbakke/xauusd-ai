"""Detached same-step observations. No forward/backward/update/mode/RNG authority."""
from __future__ import annotations
import math
import numpy as np
import torch
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.audit.qualify_native_m5_full_gc_direction_v1 import FUTURE_CPU_TRAIN_UPPER

TF=('M5','M15','H1','H4','D1')

def need(ok,reason):
    if not bool(ok):raise ValueError('NATIVE_STEP_OBSERVER_'+reason)

def scalar(value):
    if isinstance(value,torch.Tensor):
        need(value.numel()==1 and bool(torch.isfinite(value.detach()).all()),'FINITE_SCALAR')
        return float(value.detach().item())
    need(type(value)in(int,float)and math.isfinite(value),'FINITE_SCALAR')
    return float(value)

def side_bias(model,tf):
    bias=model.readout.side_heads[tf].bias
    need(isinstance(bias,torch.Tensor)and bias.dtype==torch.float32 and tuple(bias.shape)==(1,),'NATIVE_F32_SIDE_BIAS')
    return bias

def bias_and_AdamW(model,optimizer):
    """Copy scalars, never retain tensor aliases or create absent defaultdict slots."""
    result={}
    for tf in TF:
        bias=side_bias(model,tf);state=optimizer.state.get(bias)
        cell={'bias':scalar(bias)}
        if not state:cell['AdamW']={'status':'EMPTY_UNINITIALIZED'}
        else:
            need(set(state)=={'step','exp_avg','exp_avg_sq'},'EXACT_ADAMW_SCALAR_SLOT_KEYS')
            need(all(isinstance(state[k],torch.Tensor)and state[k].dtype==torch.float32 and tuple(state[k].shape)==(1,)for k in('exp_avg','exp_avg_sq')),'NATIVE_F32_ADAMW_BIAS_SLOTS')
            cell['AdamW']={'status':'OBSERVED','step':scalar(state['step']),'exp_avg':scalar(state['exp_avg']),'exp_avg_sq':scalar(state['exp_avg_sq'])}
        result[tf]=cell
    return result

def f32_bias_gradient_bound(probability,labels,denominator):
    """Bound F32 subtraction/division/reduction relative to the observed F32 sigmoid.

    gamma(n+4) covers n-term native bias reduction and four elementwise/scaling
    rounding stages. n*(n+4)*tiny also covers flush-to-zero underflow; this is a
    roundoff bound, not statistical uncertainty or FP64 endpoint precision.
    """
    p=np.asarray(probability);y=np.asarray(labels)
    need(p.dtype==np.float32 and y.dtype==np.bool_ and p.ndim==y.ndim==1 and p.shape==y.shape and denominator>0 and np.isfinite(p).all(),'GRADIENT_BOUND_DOMAIN')
    residual=p.astype(np.float64)-y.astype(np.float64);n=len(p);finfo=np.finfo(np.float32);u=float(finfo.eps)/2;operations=n+4
    need(operations*u<1,'F32_REDUCTION_BOUND_FINITE')
    gamma=operations*u/(1-operations*u);l1=float(np.abs(residual).sum())/denominator;underflow=float(n*operations*float(finfo.tiny));bound=float(gamma*l1+underflow)
    return {'analytic_from_actual_F32_sigmoid':float(residual.sum())/denominator,'absolute_error_bound':bound,'unit_roundoff':u,'gamma_operations':operations,'scaled_residual_l1':l1,'underflow_allowance':underflow,'scope':'Native F32 BCE subtraction/scaling/bias reduction versus observed F32 sigmoid; no FP64=1e-12 tolerance or optimizer-gradient claim.'}

class NativeFutureStateStepObserver:
    """Optional trusted callback for core.future_cuda_step; records stay external.

    Pass one instance as observer=. reports contains only completed JSON-native
    records. No tensor graph, optimizer slot or module hook is retained.
    """
    def __init__(self):
        self.reports=[];self._pending=None;self._next_phase='forward_loss'

    def __call__(self,phase,*,model,optimizer,index,output=None,truth=None,loss=None,plan=None,global_norm=None):
        need(phase==self._next_phase and type(index)is int and index>=0,'PHASE_ORDER_LOGICAL_INDEX')
        if phase=='forward_loss':
            need(self._pending is None and (not self.reports or index==self.reports[-1]['logical_index']+1),'CONTIGUOUS_OBSERVED_STEPS')
            z=output['side_logits'];need(z.dtype==torch.float32 and z.ndim==2 and z.shape[1]==5 and bool(torch.isfinite(z.detach()).all()),'ACTUAL_TRAIN_F32_OUTPUT')
            masks=objective._native_future_state_truth(truth,rows=len(z),fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,target_contract=plan['metadata']['target_contract'])
            A=int(masks['activity'].sum());need(A==loss['count']and A>0,'ALL5_ACTIVITY_DENOMINATOR')
            probability=torch.sigmoid(z.detach()).to('cpu').numpy().copy();logits=z.detach().to('cpu').numpy().copy();cells={}
            for j,tf in enumerate(TF):
                mask=masks['side'][:,j];y=truth['side_up'][mask,j];p=probability[mask,j];count=int(mask.sum())
                need(int(loss['counts']['side'][j])==count and int(loss['counts']['activity'][j])==int(masks['activity'][:,j].sum()),'ACTUAL_LOSS_KNOWN_COUNTS')
                cells[tf]={'activity_known':int(masks['activity'][:,j].sum()),'side_known':count,'UP':int(y.sum()),'A_b':A,'mean_logit':float(logits[mask,j].astype(np.float64).mean())if count else None,'mean_probability_UP':float(p.astype(np.float64).mean())if count else None,'UP_fraction':float(y.mean())if count else None,'raw_side_bias_gradient_check':f32_bias_gradient_bound(p,y,A)}
            self._pending={'schema_version':'gx1_native_actual_step_observation_v1','logical_index':index,'logical_step':index+1,'rows':len(z),'A_b':A,'per_tf':cells,'gradient_clip_norm':float(plan['recipe']['gradient_clip_norm']),'mode':'actual_TRAIN','scope':'Same-step detached observations; not reconstruction of historical steps, later generalization or economic edge.'};self._next_phase='backward';return
        need(self._pending is not None and self._pending['logical_index']==index,'SAME_STEP_BOUNDARIES');record=self._pending
        if phase=='backward':
            for tf in TF:
                bias=side_bias(model,tf);need(bias.grad is not None and bias.grad.dtype==torch.float32,'NATIVE_F32_BIAS_GRADIENT');actual=scalar(bias.grad);cell=record['per_tf'][tf];check=cell['raw_side_bias_gradient_check'];difference=actual-check['analytic_from_actual_F32_sigmoid'];check.update(actual_native_F32_preclip=actual,signed_native_minus_analytic=difference,absolute_native_minus_analytic=abs(difference),within_F32_bound=abs(difference)<=check['absolute_error_bound']);need(check['within_F32_bound'],'NATIVE_F32_ANALYTIC_RAW_BIAS_GRADIENT');cell['bias_gradient_pre_clip']=actual
            self._next_phase='clip';return
        if phase=='clip':
            record['global_gradient_norm_pre_clip']=scalar(global_norm)
            for tf in TF:record['per_tf'][tf]['bias_gradient_post_clip']=scalar(side_bias(model,tf).grad)
            record['before_optimizer_step']=bias_and_AdamW(model,optimizer);self._next_phase='step';return
        record['after_optimizer_step']=bias_and_AdamW(model,optimizer)
        for tf in TF:record['per_tf'][tf]['actual_bias_change']=record['after_optimizer_step'][tf]['bias']-record['before_optimizer_step'][tf]['bias']
        self.reports.append(record);self._pending=None;self._next_phase='forward_loss'
