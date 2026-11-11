"""Local loss-gradient partition on one existing graph; no model or fit owner."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import math

ARMS = ('state_m5_side', 'future_m5_side', 'rest', 'total')
CLOSURE_RELATIVE_L2_LIMIT = 1e-4


class MixingGradientClosureError(RuntimeError):
    """Preserve scalar evidence when the original loss has unaccounted paths."""
    def __init__(self, report_json):
        super().__init__('TA_MIXING_GRADIENT_CLOSURE')
        self.report_json = report_json


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_MIXING_GRADIENT_' + reason)


def measure_mixing_gradient_partition(report, parameter_groups, *, counts, torch):
    """Return scalar evidence and owned FP64 CPU vectors for the four arms.

    The exact original loss supplies the logit seeds, including its masks and
    normalization. Three VJPs partition those seeds without reimplementing loss.
    Missing derivatives have zero placeholders only in the flat-vector format;
    ``tensor_gradient_state`` retains None separately from exact zero tensors.
    No parameter .grad, RNG, parameter value or module mode is written here.
    The caller owns eval/precision, selected trainability and run admission.
    """
    _need(isinstance(report, Mapping) and isinstance(parameter_groups, Mapping)
          and bool(parameter_groups) and isinstance(counts, dict)
          and type(counts.get('autograd_grad_calls')) is int
          and counts['autograd_grad_calls'] >= 0, 'INPUT')
    _need(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), 'GRAD_CONTEXT')
    loss = report.get('loss')
    raws = tuple(report.get(task+'_activity_side_logits') for task in ('state','future'))
    _need(isinstance(loss,torch.Tensor) and loss.ndim == 0 and loss.dtype == torch.float32
          and loss.requires_grad and bool(torch.isfinite(loss).item()), 'LOSS')
    _need(all(isinstance(raw,torch.Tensor) and raw.dtype == torch.float32
              and raw.ndim == 3 and raw.shape[0] > 0 and tuple(raw.shape[1:]) == (5,2)
              and raw.device == loss.device and raw.requires_grad
              and bool(torch.isfinite(raw).all().item()) for raw in raws)
          and raws[0].shape == raws[1].shape and raws[0] is not raws[1], 'LOGITS')
    pairs = []
    for group, group_pairs in parameter_groups.items():
        _need(isinstance(group,str) and bool(group) and isinstance(group_pairs,(tuple,list))
              and bool(group_pairs), 'GROUP')
        for pair in group_pairs:
            _need(isinstance(pair,(tuple,list)) and len(pair) == 2, 'PARAMETER_PAIR')
            name, parameter = pair
            _need(isinstance(name,str) and bool(name) and isinstance(parameter,torch.nn.Parameter)
                  and parameter.is_leaf and parameter.requires_grad and parameter.dtype == torch.float32
                  and parameter.device == loss.device and parameter.numel() > 0
                  and bool(torch.isfinite(parameter.detach()).all().item()), 'PARAMETER')
            pairs.append((name,parameter))
    _need(len({name for name,_ in pairs}) == len({id(p) for _,p in pairs}) == len(pairs), 'PARAMETER_ALIASES')
    parameters = tuple(p for _,p in pairs)
    call_start = counts['autograd_grad_calls']

    def grad(outputs, inputs, *, seeds=None, retain_graph):
        counts['autograd_grad_calls'] += 1
        return torch.autograd.grad(outputs,inputs,grad_outputs=seeds,retain_graph=retain_graph,
            create_graph=False,allow_unused=True)

    def checked(value, reference):
        if value is not None:
            _need(value.shape == reference.shape and value.dtype == reference.dtype
                  and value.device == reference.device and not value.requires_grad
                  and bool(torch.isfinite(value).all().item()), 'DERIVATIVE')
        return value

    all_total = grad(loss, (*raws,*parameters),retain_graph=True)
    logit_gradients = tuple(checked(v,r) for v,r in zip(all_total[:2],raws))
    gradients = {'total':tuple(checked(v,p) for v,p in zip(all_total[2:],parameters))}
    seeds = tuple(torch.zeros_like(raw) if value is None else value.detach().clone()
                  for raw,value in zip(raws,logit_gradients))
    state_m5 = torch.zeros_like(raws[0]);state_m5[:,0,1] = seeds[0][:,0,1]
    future_m5 = torch.zeros_like(raws[1]);future_m5[:,0,1] = seeds[1][:,0,1]
    rest = tuple(value.clone() for value in seeds)
    for value in rest:
        value[:,0,1] = 0
    partition = {
        'state_m5_side':(state_m5,torch.zeros_like(raws[1])),
        'future_m5_side':(torch.zeros_like(raws[0]),future_m5),
        'rest':rest,
    }
    for arm, arm_seeds in partition.items():
        values = grad(raws,parameters,seeds=arm_seeds,retain_graph=arm != 'rest')
        gradients[arm] = tuple(checked(v,p) for v,p in zip(values,parameters))
    _need(counts['autograd_grad_calls']-call_start == 4, 'CALL_COUNT')

    def l2(value):
        return float(value.square().sum().sqrt().item())

    def pair_stats(first, second):
        dot = float(torch.dot(first,second).item())
        denominator = l2(first)*l2(second)
        return dict(dot=dot,cosine=None if denominator == 0 else dot/denominator)

    support = {}
    for task, raw, value in zip(('state','future'),raws,logit_gradients):
        support[task] = dict(gradient_state=None if value is None else (
            'zero' if torch.count_nonzero(value).item() == 0 else 'nonzero'),
            elements=raw.numel(),nonzero_elements=None if value is None else int(torch.count_nonzero(value).item()),
            m5_side_nonzero_elements=None if value is None else int(torch.count_nonzero(value[:,0,1]).item()),
            remaining_channel_nonzero_elements=None if value is None else (
                int(torch.count_nonzero(value).item())-int(torch.count_nonzero(value[:,0,1]).item())))
    result = dict(schema_version='gx1_ta_mixing_gradient_partition_v1',
        autograd_grad_calls=counts['autograd_grad_calls']-call_start,
        closure_relative_l2_limit=CLOSURE_RELATIVE_L2_LIMIT,
        flat_vector_missing_derivative='zero placeholders; tensor_gradient_state preserves None',
        logit_gradient_support=support,groups={})
    vectors = {}; offset = 0
    for group, group_pairs in parameter_groups.items():
        vectors[group] = {}; arm_reports = {}
        for arm in ARMS:
            values = gradients[arm][offset:offset+len(group_pairs)]
            status = {}; pieces = []; missing_elements = 0
            for (name,parameter),value in zip(group_pairs,values):
                status[name] = None if value is None else (
                    'zero' if torch.count_nonzero(value).item() == 0 else 'nonzero')
                if value is None:
                    missing_elements += parameter.numel()
                    pieces.append(torch.zeros(parameter.numel(),dtype=torch.float64,device='cpu'))
                else:
                    pieces.append(value.detach().reshape(-1).to(device='cpu',dtype=torch.float64).clone())
            vector = torch.cat(pieces);vectors[group][arm] = vector
            arm_reports[arm] = dict(l2=l2(vector),elements=vector.numel(),
                nonzero_elements=int(torch.count_nonzero(vector).item()),missing_elements=missing_elements,
                missing_tensors=sum(v is None for v in status.values()),
                zero_tensors=sum(v == 'zero' for v in status.values()),
                nonzero_tensors=sum(v == 'nonzero' for v in status.values()),tensor_gradient_state=status)
        v = vectors[group]
        closure = v['state_m5_side']+v['future_m5_side']+v['rest']-v['total']
        closure_l2 = l2(closure);denominator = sum(arm_reports[arm]['l2'] for arm in ARMS)
        relative = closure_l2/denominator if denominator else 0. if closure_l2 == 0 else None
        passed = (closure_l2 == 0 if denominator == 0 else relative <= CLOSURE_RELATIVE_L2_LIMIT)
        result['groups'][group] = dict(parameters=len(group_pairs),arms=arm_reports,
            pairwise={a+'__'+b:pair_stats(v[a],v[b]) for a,b in combinations(ARMS,2)},
            m5_side_sum_vs_rest=pair_stats(v['state_m5_side']+v['future_m5_side'],v['rest']),
            closure_l2=closure_l2,closure_sum_arm_l2=denominator,
            closure_relative_l2=relative,closure_passed=passed)
        offset += len(group_pairs)
    result['closure_passed'] = all(group['closure_passed'] for group in result['groups'].values())
    _need(all(math.isfinite(group['closure_l2']) and math.isfinite(group['closure_sum_arm_l2'])
              for group in result['groups'].values()), 'CLOSURE_NONFINITE')
    if not result['closure_passed']:
        raise MixingGradientClosureError(result)
    return result, vectors
