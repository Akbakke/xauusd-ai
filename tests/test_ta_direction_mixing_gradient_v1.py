from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from gx1.audit.ta_direction_mixing_gradient_v1 import (
    ARMS, CLOSURE_RELATIVE_L2_LIMIT, MixingGradientClosureError,
    measure_mixing_gradient_partition,
)


def fixture_graph(*, m5_scale=1.0, zero_loss=False):
    """Synthetic tensors only: three shared groups plus absent/zero derivatives."""
    a=torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4)/29-.2)
    b=torch.nn.Parameter(torch.arange(16,dtype=torch.float32).reshape(4,4)/37-.1)
    c=torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4)/23-.3)
    absent=torch.nn.Parameter(torch.tensor([7.,8.]))
    zero=torch.nn.Parameter(torch.tensor([3.]))
    groups={'family_axis_attn':[('family.a',a),('family.absent',absent)],
            'timeframe_axis_attn':[('timeframe.b',b),('timeframe.zero',zero)],
            'cross_tf_attn':[('cross.c',c)]}
    x=torch.arange(12,dtype=torch.float32).reshape(4,3)/11-.4
    hidden=torch.tanh((x@a)@b+x@c)+0*zero
    ws=torch.arange(40,dtype=torch.float32).reshape(4,10)/41-.3
    wf=torch.arange(40,dtype=torch.float32).flip(0).reshape(4,10)/47-.1
    state=(hidden@ws).reshape(4,5,2)
    future=(hidden@wf).reshape(4,5,2)
    target=(torch.arange(40).reshape(4,5,2)%3==0).float()
    weight_s=torch.arange(40,dtype=torch.float32).reshape(4,5,2)/53
    weight_f=torch.arange(40,dtype=torch.float32).reshape(4,5,2).flip(0)/67
    weight_s[::2,2,:]=0;weight_f[1::2,3,:]=0
    # Different masks and unequal pre-existing observation weights are inherited.
    weight_s[:,0,1]=torch.tensor([1.,0.,1.,0.])*(1799/10010/5)*m5_scale
    weight_f[:,0,1]=torch.tensor([0.,1.,1.,1.])*(1844/11909/5)*m5_scale
    cell_s=F.binary_cross_entropy_with_logits(state,target,reduction='none')*weight_s
    cell_f=F.binary_cross_entropy_with_logits(future,1-target,reduction='none')*weight_f
    components={
        'state_m5_side':cell_s[:,0,1].sum(),
        'future_m5_side':cell_f[:,0,1].sum(),
        'rest':cell_s[:,:,0].sum()+cell_s[:,1:,1].sum()+cell_f[:,:,0].sum()+cell_f[:,1:,1].sum(),
    }
    loss=sum(components.values())
    if zero_loss:
        components={k:v*0 for k,v in components.items()};loss=sum(components.values())
    report={'loss':loss,'state_activity_side_logits':state,'future_activity_side_logits':future}
    return report,groups,components,(weight_s,weight_f)


def assert_scalar_leaves(value):
    if isinstance(value,dict):
        for child in value.values():assert_scalar_leaves(child)
    else:
        assert value is None or type(value) in (str,int,float,bool)


@pytest.mark.parametrize('m5_scale',[1.,0.,2.75])
def test_partition_matches_independent_loss_components_and_inherits_weights(m5_scale):
    report,groups,components,weights=fixture_graph(m5_scale=m5_scale)
    pairs=[pair for group in groups.values() for pair in group]
    parameters=tuple(p for _,p in pairs)
    expected={arm:torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
              for arm,loss in {**components,'total':report['loss']}.items()}
    rng=torch.get_rng_state().clone()
    before={name:p.detach().clone() for name,p in pairs}
    for _,p in pairs:p.grad=torch.full_like(p,3.25)
    existing_grads=[(id(p.grad),p.grad.clone()) for _,p in pairs]
    flags=[(id(p),p.requires_grad) for _,p in pairs]
    counts={'autograd_grad_calls':7,'unrelated_counter':11}
    result,vectors=measure_mixing_gradient_partition(report,groups,counts=counts,torch=torch)
    assert counts=={'autograd_grad_calls':11,'unrelated_counter':11}
    assert result['autograd_grad_calls']==4 and result['closure_passed']
    assert result['closure_relative_l2_limit']==CLOSURE_RELATIVE_L2_LIMIT==1e-4
    assert tuple(vectors)==tuple(groups)
    offset=0
    for group,group_pairs in groups.items():
        for arm in ARMS:
            pieces=[]
            for (_,p),g in zip(group_pairs,expected[arm][offset:offset+len(group_pairs)]):
                pieces.append(torch.zeros(p.numel(),dtype=torch.float64) if g is None else g.detach().reshape(-1).double())
            reference=torch.cat(pieces);actual=vectors[group][arm]
            assert actual.dtype==torch.float64 and actual.device.type=='cpu' and actual.ndim==1
            assert actual.grad_fn is None and not actual.requires_grad
            torch.testing.assert_close(actual,reference,rtol=2e-6,atol=2e-8)
        closure=sum((vectors[group][arm] for arm in ARMS[:3]))-vectors[group]['total']
        assert result['groups'][group]['closure_l2']==float(closure.square().sum().sqrt())
        assert result['groups'][group]['closure_relative_l2']<=1e-4
        offset+=len(group_pairs)
    for task,weight in zip(('state','future'),weights):
        support=result['logit_gradient_support'][task]
        assert support['m5_side_nonzero_elements']==int(torch.count_nonzero(weight[:,0,1]))
        assert support['nonzero_elements']==int(torch.count_nonzero(weight))
    assert all(torch.equal(p.detach(),before[name]) for name,p in pairs)
    assert [(id(p),p.requires_grad) for _,p in pairs]==flags
    assert all(id(p.grad)==identity and torch.equal(p.grad,value)
               for (_,p),(identity,value) in zip(pairs,existing_grads))
    assert torch.equal(torch.get_rng_state(),rng)
    assert_scalar_leaves(result);json.dumps(result,allow_nan=False)


def test_missing_and_exact_zero_derivatives_remain_distinct():
    report,groups,_,_=fixture_graph()
    result,vectors=measure_mixing_gradient_partition(report,groups,counts={'autograd_grad_calls':0},torch=torch)
    for arm in ARMS:
        absent=result['groups']['family_axis_attn']['arms'][arm]
        zero=result['groups']['timeframe_axis_attn']['arms'][arm]
        assert absent['tensor_gradient_state']['family.absent'] is None
        assert absent['missing_tensors']==1 and absent['missing_elements']==2
        assert zero['tensor_gradient_state']['timeframe.zero']=='zero'
        assert zero['zero_tensors']==1 and zero['missing_tensors']==0
        assert torch.equal(vectors['family_axis_attn'][arm][-2:],torch.zeros(2,dtype=torch.float64))
        assert vectors['timeframe_axis_attn'][arm][-1].item()==0
    assert all(p.grad is None for pairs in groups.values() for _,p in pairs)


def test_unused_logit_tensor_is_reported_as_missing_without_imputed_m5_gradient():
    report,groups,_,_=fixture_graph()
    report['loss']=report['future_activity_side_logits'][:,:,0].square().sum()
    counts={'autograd_grad_calls':0}
    result,vectors=measure_mixing_gradient_partition(report,groups,counts=counts,torch=torch)
    support=result['logit_gradient_support']
    assert support['state']['gradient_state'] is None
    assert support['state']['m5_side_nonzero_elements'] is None
    assert support['future']['m5_side_nonzero_elements']==0
    for group in vectors:
        for arm in ('state_m5_side','future_m5_side'):
            assert torch.count_nonzero(vectors[group][arm]).item()==0
        assert torch.equal(vectors[group]['rest'],vectors[group]['total'])
    assert counts['autograd_grad_calls']==4


def test_zero_sum_requires_exact_zero_closure_and_cosines_are_undefined():
    report,groups,_,_=fixture_graph(zero_loss=True)
    result,vectors=measure_mixing_gradient_partition(report,groups,counts={'autograd_grad_calls':0},torch=torch)
    for group,entry in result['groups'].items():
        assert entry['closure_l2']==entry['closure_sum_arm_l2']==entry['closure_relative_l2']==0.
        assert all(p['dot']==0. and p['cosine'] is None for p in entry['pairwise'].values())
        assert entry['m5_side_sum_vs_rest']=={'dot':0.,'cosine':None}
        assert all(torch.count_nonzero(v).item()==0 for v in vectors[group].values())


def test_loss_path_outside_primitive_logits_fails_fixed_closure_and_retains_scalar_evidence():
    report,groups,_,_=fixture_graph()
    report['loss']=report['loss']+.2*groups['family_axis_attn'][0][1].sum()
    counts={'autograd_grad_calls':0}
    with pytest.raises(MixingGradientClosureError) as caught:
        measure_mixing_gradient_partition(report,groups,counts=counts,torch=torch)
    result=caught.value.report_json
    assert not result['closure_passed'] and counts['autograd_grad_calls']==4
    assert result['groups']['family_axis_attn']['closure_relative_l2']>1e-4
    assert all(p.grad is None for pairs in groups.values() for _,p in pairs)
    assert_scalar_leaves(result);json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('fault,reason',[('alias','PARAMETER_ALIASES'),('logits','LOGITS'),
                                       ('counter','INPUT'),('frozen','PARAMETER')])
def test_invalid_inputs_fail_before_any_autograd_call(fault,reason):
    report,groups,_,_=fixture_graph();counts={'autograd_grad_calls':0}
    if fault=='alias':groups['cross_tf_attn'].append(('duplicate',groups['family_axis_attn'][0][1]))
    elif fault=='logits':report['state_activity_side_logits']=report['state_activity_side_logits'][:,:4]
    elif fault=='counter':counts['autograd_grad_calls']=True
    else:groups['cross_tf_attn'][0][1].requires_grad_(False)
    with pytest.raises(RuntimeError,match='TA_MIXING_GRADIENT_'+reason):
        measure_mixing_gradient_partition(report,groups,counts=counts,torch=torch)
    assert counts['autograd_grad_calls']==(True if fault=='counter' else 0)
