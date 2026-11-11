"""Local original-loss VJPs at the shared query projection; no fit/run owner.

Dot products describe infinitesimal SGD at this graph. They are not historical
AdamW effects, causal attribution, later generalization or evidence of edge.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

LOSS_ARMS = ('state_m5_side','future_m5_side','state_h4_side','state_d1_side',
             'future_h4_side','future_d1_side','medium_side','activity')
PROBES = ('state_m5_mean_logit','future_m5_mean_logit')
ARMS = LOSS_ARMS + PROBES + ('total',)
ELEMENTS = 17280
CALLS_PER_BATCH = 11
CLOSURE_RELATIVE_L2_LIMIT = 1e-4
TIMEFRAMES = ('M5','M15','H1','H4','D1')


class QueryProjectionClosureError(RuntimeError):
    """Keep scalar evidence if direct loss paths escape the logit partition."""
    def __init__(self, report_json):
        super().__init__('TA_C171_QUERY_PROJECTION_CLOSURE')
        self.report_json = report_json


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_C171_QUERY_PROJECTION_' + reason)


def _l2(value):
    return float(value.square().sum().sqrt().item())


def summarize(vectors):
    """Pure FP64 CPU geometry on four disjoint parameter-coordinate groups.

    Negative mean-logit/proxy dot loss-gradient means that an infinitesimal
    negative-gradient SGD step increases that mean logit. No optimizer is run.
    """
    import torch
    _need(isinstance(vectors,Mapping) and set(vectors)==set(ARMS), 'VECTOR_ARMS')
    for value in vectors.values():
        _need(isinstance(value,torch.Tensor) and value.device.type=='cpu'
            and value.dtype==torch.float64 and tuple(value.shape)==(ELEMENTS,)
            and not value.requires_grad and bool(torch.isfinite(value).all().item()), 'VECTOR')
    groups = {'hidden_weight':{},'timeframe_weight':{},'query_fraction_weight':{},'bias':{}}
    for arm,value in vectors.items():
        weight=value[:17152].reshape(128,134)
        groups['hidden_weight'][arm]=weight[:,:128].reshape(-1)
        groups['timeframe_weight'][arm]=weight[:,128:133].reshape(-1)
        groups['query_fraction_weight'][arm]=weight[:,133].reshape(-1)
        groups['bias'][arm]=value[17152:]
    result={}
    for name,values in groups.items():
        norms={arm:_l2(value) for arm,value in values.items()}
        def comparison(first,second):
            dot=float(torch.dot(values[first],values[second]).item())
            denominator=norms[first]*norms[second]
            _need(math.isfinite(dot) and math.isfinite(denominator),'GEOMETRY_NONFINITE')
            return dict(dot=dot,cosine=dot/denominator if denominator else None)
        result[name]=dict(elements=values['total'].numel(),norms=norms,
            m5_loss_vs_loss_arm={first:{arm:comparison(first,arm) for arm in LOSS_ARMS}
                                 for first in LOSS_ARMS[:2]},
            m5_mean_logit_vs_loss_arm={first:{arm:comparison(first,arm) for arm in LOSS_ARMS}
                                       for first in PROBES})
    return result


def measure_query_projection_partition(report, projection, *, counts, torch):
    """Return scalar JSON and eleven owned FP64 CPU vectors of length17280.

    Original loss supplies all masks and normalization through its raw-logit
    derivatives. Eight exclusive VJPs partition these exact detached seeds.
    Two further VJPs probe mean M5-side logits over all16 rows, without masks.
    Missing derivatives get flat zero placeholders while tensor_gradient_state
    preserves None separately from exact-zero tensors. No .grad is accumulated.
    Caller owns admission, model mode, trainability and numerical execution.
    """
    _need(isinstance(report,Mapping) and isinstance(counts,dict)
        and type(counts.get('autograd_grad_calls')) is int
        and counts['autograd_grad_calls']>=0,'INPUT')
    _need(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),'GRAD_CONTEXT')
    _need(type(projection) is torch.nn.Linear and projection.in_features==134
        and projection.out_features==128 and projection.bias is not None,'PROJECTION')
    parameters=(projection.weight,projection.bias)
    loss=report.get('loss')
    raws=tuple(report.get(task+'_activity_side_logits') for task in ('state','future'))
    _need(isinstance(loss,torch.Tensor) and loss.ndim==0 and loss.dtype==torch.float32
        and loss.requires_grad and bool(torch.isfinite(loss).item()),'LOSS')
    _need(all(isinstance(raw,torch.Tensor) and tuple(raw.shape)==(16,5,2)
        and raw.dtype==torch.float32 and raw.device==loss.device and raw.requires_grad
        and bool(torch.isfinite(raw).all().item()) for raw in raws)
        and raws[0] is not raws[1],'LOGITS')
    _need(tuple(parameters[0].shape)==(128,134) and tuple(parameters[1].shape)==(128,)
        and all(isinstance(p,torch.nn.Parameter) and p.is_leaf and p.requires_grad
            and p.dtype==torch.float32 and p.device==loss.device
            and bool(torch.isfinite(p.detach()).all().item()) for p in parameters),'PARAMETERS')
    call_start=counts['autograd_grad_calls']
    def grad(outputs,inputs,*,seeds=None,retain_graph):
        counts['autograd_grad_calls']+=1
        return torch.autograd.grad(outputs,inputs,grad_outputs=seeds,
            retain_graph=retain_graph,create_graph=False,allow_unused=True)
    def checked(value,reference):
        if value is not None:
            _need(value.shape==reference.shape and value.dtype==reference.dtype
                and value.device==reference.device and not value.requires_grad
                and bool(torch.isfinite(value).all().item()),'DERIVATIVE')
        return value
    def status(value):
        return None if value is None else 'nonzero' if torch.count_nonzero(value).item() else 'zero'
    direct=grad(loss,(*raws,*parameters),retain_graph=True)
    raw_gradients=tuple(checked(value,raw) for value,raw in zip(direct[:2],raws))
    gradients={'total':tuple(checked(value,p) for value,p in zip(direct[2:],parameters))}
    seeds=tuple(torch.zeros_like(raw) if value is None else value.detach().clone()
                for raw,value in zip(raws,raw_gradients))
    side_cells={'state_m5_side':(0,0),'future_m5_side':(1,0),
        'state_h4_side':(0,3),'state_d1_side':(0,4),
        'future_h4_side':(1,3),'future_d1_side':(1,4)}
    seed_support={}
    for arm in LOSS_ARMS:
        arm_seeds=tuple(torch.zeros_like(raw) for raw in raws)
        if arm in side_cells:
            task,tf=side_cells[arm];arm_seeds[task][:,tf,1]=seeds[task][:,tf,1]
        elif arm=='medium_side':
            for actual,source in zip(arm_seeds,seeds):actual[:,1:3,1]=source[:,1:3,1]
        else:
            for actual,source in zip(arm_seeds,seeds):actual[:,:,0]=source[:,:,0]
        seed_support[arm]={task:int(torch.count_nonzero(value).item())
                           for task,value in zip(('state','future'),arm_seeds)}
        values=grad(raws,parameters,seeds=arm_seeds,retain_graph=True)
        gradients[arm]=tuple(checked(value,p) for value,p in zip(values,parameters))
    for task,arm in enumerate(PROBES):
        probe=tuple(torch.zeros_like(raw) for raw in raws)
        probe[task][:,0,1]=1./16
        values=grad(raws,parameters,seeds=probe,retain_graph=arm!=PROBES[-1])
        gradients[arm]=tuple(checked(value,p) for value,p in zip(values,parameters))
    _need(counts['autograd_grad_calls']-call_start==CALLS_PER_BATCH,'CALL_COUNT')
    vectors={};arm_reports={}
    for arm in ARMS:
        values=gradients[arm];states={};pieces=[];missing_elements=0
        for name,p,value in zip(('weight','bias'),parameters,values):
            states[name]=status(value)
            if value is None:
                missing_elements+=p.numel()
                pieces.append(torch.zeros(p.numel(),dtype=torch.float64,device='cpu'))
            else:pieces.append(value.detach().reshape(-1).to(device='cpu',dtype=torch.float64).clone())
        vector=torch.cat(pieces);vectors[arm]=vector
        arm_reports[arm]=dict(l2=_l2(vector),elements=ELEMENTS,
            nonzero_elements=int(torch.count_nonzero(vector).item()),missing_elements=missing_elements,
            tensor_gradient_state=states,missing_tensors=sum(v is None for v in states.values()),
            zero_tensors=sum(v=='zero' for v in states.values()),nonzero_tensors=sum(v=='nonzero' for v in states.values()))
    closure=torch.zeros(ELEMENTS,dtype=torch.float64,device='cpu')
    for arm in LOSS_ARMS:closure=closure+vectors[arm]
    closure=closure-vectors['total'];closure_l2=_l2(closure)
    denominator=sum(arm_reports[arm]['l2'] for arm in (*LOSS_ARMS,'total'))
    _need(math.isfinite(closure_l2) and math.isfinite(denominator),'CLOSURE_NONFINITE')
    relative=closure_l2/denominator if denominator else (0. if closure_l2==0 else None)
    passed=closure_l2==0 if denominator==0 else relative<=CLOSURE_RELATIVE_L2_LIMIT
    support={}
    for task,raw,value in zip(('state','future'),raws,raw_gradients):
        support[task]=dict(gradient_state=status(value),elements=raw.numel(),
            nonzero_elements=None if value is None else int(torch.count_nonzero(value).item()),
            by_timeframe={tf:{channel:None if value is None else int(torch.count_nonzero(value[:,j,k]).item())
                for k,channel in enumerate(('activity','side'))} for j,tf in enumerate(TIMEFRAMES)})
    measurement=dict(schema_version='gx1_ta_c171_query_projection_gradient_partition_v1',
        autograd_grad_calls=counts['autograd_grad_calls']-call_start,
        vector_layout=dict(elements=ELEMENTS,weight_shape=[128,134],weight_flat_elements=17152,
            bias_offset=17152,bias_elements=128,flatten_order='C'),
        loss_arms=list(LOSS_ARMS),probes=list(PROBES),arms=arm_reports,
        logit_gradient_support=support,loss_seed_nonzero_elements=seed_support,
        flat_vector_missing_derivative='zero placeholders; tensor_gradient_state preserves None',
        closure_l2=closure_l2,closure_sum_arm_l2=denominator,closure_relative_l2=relative,
        closure_relative_l2_limit=CLOSURE_RELATIVE_L2_LIMIT,closure_passed=passed,
        coordinate_groups=summarize(vectors),
        interpretation='Negative mean-logit dot loss-gradient predicts an infinitesimal SGD increase in that mean logit; not an AdamW or historical causal effect.')
    if not passed:raise QueryProjectionClosureError(measurement)
    return measurement,vectors
