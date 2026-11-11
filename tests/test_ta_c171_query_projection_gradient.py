"""Small synthetic graphs only; no GX1 model, data, checkpoint, optimizer or fit."""
import json
from types import SimpleNamespace

import pytest
import torch

from gx1.audit import ta_c171_query_projection_gradient_v1 as helper


def graph(*, use_bias=True):
    projection=torch.nn.Linear(134,128)
    x=torch.linspace(-.4,.6,16*5*134).reshape(16,5,134)
    coefficients=torch.stack((torch.linspace(-.2,.3,128),torch.linspace(.25,-.15,128)),dim=1)
    def raw(inputs):
        hidden=torch.nn.functional.linear(inputs,projection.weight,projection.bias if use_bias else None)
        return torch.nn.functional.gelu(hidden)@coefficients
    state=raw(x);future=raw(x.flip(0)*.7+.2)
    return projection,state,future


def weighted_components(state,future):
    # Unequal per-row/channel support and unequal per-TF scaling deliberately
    # live in the original loss; the helper must inherit rather than rebuild it.
    slots=torch.arange(160).reshape(16,5,2)
    sw=(slots%4!=0).float()*(slots%7+1)/torch.tensor([13.,17.,19.,23.,29.])[None,:,None]
    fw=(slots%3!=0).float()*(slots%5+1)/torch.tensor([11.,31.,37.,41.,43.])[None,:,None]
    s=state.square()*sw;f=(future-.3).square()*fw
    return {'state_m5_side':s[:,0,1].sum(),'future_m5_side':f[:,0,1].sum(),
        'state_h4_side':s[:,3,1].sum(),'state_d1_side':s[:,4,1].sum(),
        'future_h4_side':f[:,3,1].sum(),'future_d1_side':f[:,4,1].sum(),
        'medium_side':sum(x[:,j,1].sum() for x in (s,f) for j in (1,2)),
        'activity':sum(x[:,:,0].sum() for x in (s,f))}


def report(state,future,loss):
    return dict(loss=loss,state_activity_side_logits=state,future_activity_side_logits=future)


def flat(gradients,parameters):
    return torch.cat([torch.zeros(p.numel(),dtype=torch.float64) if g is None
        else g.detach().flatten().double().clone() for g,p in zip(gradients,parameters)])


def test_partition_matches_independent_loss_components_and_probe_gradients(monkeypatch):
    projection,s,f=graph();params=tuple(projection.parameters());components=weighted_components(s,f)
    loss=sum(components.values())
    expected={arm:flat(torch.autograd.grad(value,params,retain_graph=True,allow_unused=True),params)
              for arm,value in components.items()}
    for arm,raw in zip(helper.PROBES,(s,f)):
        expected[arm]=flat(torch.autograd.grad(raw[:,0,1].mean(),params,retain_graph=True),params)
    expected['total']=flat(torch.autograd.grad(loss,params,retain_graph=True),params)
    direct_seeds=torch.autograd.grad(loss,(s,f),retain_graph=True)
    real_grad=torch.autograd.grad;calls=[]
    def record(*args,**kwargs):
        calls.append(kwargs['retain_graph']);return real_grad(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',record)
    counts={'autograd_grad_calls':7}
    result,vectors=helper.measure_query_projection_partition(report(s,f,loss),projection,counts=counts,torch=torch)
    assert counts['autograd_grad_calls']==18 and result['autograd_grad_calls']==11
    assert calls==[True]*10+[False]
    assert set(vectors)==set(helper.ARMS) and len(vectors)==11
    assert result['closure_passed'] and result['closure_relative_l2']<=1e-4
    for arm in helper.ARMS:
        torch.testing.assert_close(vectors[arm],expected[arm],rtol=3e-6,atol=2e-6)
        assert vectors[arm].dtype==torch.float64 and vectors[arm].device.type=='cpu'
        assert vectors[arm].shape==(17280,) and not vectors[arm].requires_grad
        assert torch.equal(vectors[arm],vectors[arm].float().double())
    assert len({v.data_ptr() for v in vectors.values()})==11
    for task,seed in zip(('state','future'),direct_seeds):
        assert result['logit_gradient_support'][task]['nonzero_elements']==int(torch.count_nonzero(seed))
        for j,tf in enumerate(helper.TIMEFRAMES):
            assert result['logit_gradient_support'][task]['by_timeframe'][tf]=={
                'activity':int(torch.count_nonzero(seed[:,j,0])), 'side':int(torch.count_nonzero(seed[:,j,1]))}
    json.dumps(result,allow_nan=False)
    with pytest.raises(RuntimeError,match='backward through the graph a second time'):
        real_grad(loss,params)


def test_parameter_values_rng_modes_and_existing_gradients_are_unchanged():
    projection,s,f=graph();projection.eval()
    params=tuple(projection.parameters())
    for p in params:p.grad=torch.full_like(p,.125)
    values=[p.detach().clone() for p in params];gradient_values=[p.grad.clone() for p in params]
    identities=[(id(p),id(p.grad),p.requires_grad) for p in params]
    rng=torch.get_rng_state().clone();mode=projection.training
    _,vectors=helper.measure_query_projection_partition(report(s,f,(s.square().mean()+f.square().mean())),projection,
        counts={'autograd_grad_calls':0},torch=torch)
    assert torch.equal(rng,torch.get_rng_state()) and projection.training==mode
    for i,p in enumerate(params):
        assert (id(p),id(p.grad),p.requires_grad)==identities[i]
        assert torch.equal(p,values[i]) and torch.equal(p.grad,gradient_values[i])
    saved=vectors['future_m5_side'].clone();vectors['state_m5_side'].fill_(123.)
    assert torch.equal(saved,vectors['future_m5_side'])
    assert all(torch.equal(p,values[i]) for i,p in enumerate(params))


def test_none_and_exact_zero_stay_distinct_and_zero_closure_passes():
    projection,s,f=graph(use_bias=False)
    result,vectors=helper.measure_query_projection_partition(report(s,f,s.mul(0).sum()),projection,
        counts={'autograd_grad_calls':0},torch=torch)
    assert result['logit_gradient_support']['state']['gradient_state']=='zero'
    assert result['logit_gradient_support']['future']['gradient_state'] is None
    assert result['closure_sum_arm_l2']==result['closure_l2']==result['closure_relative_l2']==0.
    for arm in (*helper.LOSS_ARMS,'total'):
        assert result['arms'][arm]['tensor_gradient_state']=={'weight':'zero','bias':None}
        assert result['arms'][arm]['missing_elements']==128
        assert torch.count_nonzero(vectors[arm])==0
    assert torch.count_nonzero(vectors['state_m5_mean_logit'])>0
    assert result['arms']['state_m5_mean_logit']['tensor_gradient_state']['bias'] is None


def test_completely_unused_projection_is_missing_not_zero():
    projection=torch.nn.Linear(134,128)
    s=torch.ones(16,5,2,requires_grad=True);f=torch.ones(16,5,2,requires_grad=True)
    result,vectors=helper.measure_query_projection_partition(report(s,f,s.square().sum()),projection,
        counts={'autograd_grad_calls':0},torch=torch)
    for arm in helper.ARMS:
        assert result['arms'][arm]['tensor_gradient_state']=={'weight':None,'bias':None}
        assert result['arms'][arm]['missing_elements']==17280
        assert torch.count_nonzero(vectors[arm])==0
    assert result['closure_passed']


def test_zero_m5_loss_seed_does_not_remove_mean_logit_probe():
    projection,s,f=graph()
    loss=s[:,3,1].square().sum()+f[:,4,1].square().sum()
    result,vectors=helper.measure_query_projection_partition(report(s,f,loss),projection,
        counts={'autograd_grad_calls':0},torch=torch)
    for arm in ('state_m5_side','future_m5_side','medium_side','activity'):
        assert torch.count_nonzero(vectors[arm])==0
        assert result['loss_seed_nonzero_elements'][arm]=={'state':0,'future':0}
    assert all(torch.count_nonzero(vectors[arm])>0 for arm in helper.PROBES)


def test_unmediated_parameter_loss_fails_closed_with_scalar_evidence():
    projection,s,f=graph();counts={'autograd_grad_calls':0}
    loss=s.mul(0).sum()+projection.weight[0,0]
    with pytest.raises(helper.QueryProjectionClosureError) as error:
        helper.measure_query_projection_partition(report(s,f,loss),projection,counts=counts,torch=torch)
    result=error.value.report_json
    assert counts['autograd_grad_calls']==11 and not result['closure_passed']
    assert result['closure_l2']==1. and result['closure_relative_l2']==1.
    assert all(p.grad is None for p in projection.parameters())
    json.dumps(result,allow_nan=False)


def test_geometry_coordinates_dots_and_sign_convention_are_exact_and_pure():
    vectors={arm:torch.zeros(17280,dtype=torch.float64) for arm in helper.ARMS}
    locations={'hidden_weight':134+127,'timeframe_weight':11*134+130,
               'query_fraction_weight':5*134+133,'bias':17152+45}
    for i,(name,index) in enumerate(locations.items()):
        vectors['state_m5_side'][index]=[2.,3.,4.,5.][i]
        vectors['future_d1_side'][index]=[-1.,2.,-3.,4.][i]
        vectors['state_m5_mean_logit'][index]=[1.,-2.,3.,-4.][i]
    before={k:v.clone() for k,v in vectors.items()};result=helper.summarize(vectors)
    assert {k:v['elements'] for k,v in result.items()}=={
        'hidden_weight':16384,'timeframe_weight':640,'query_fraction_weight':128,'bias':128}
    assert sum(v['elements'] for v in result.values())==17280
    for i,name in enumerate(locations):
        assert result[name]['norms']['state_m5_side']==[2.,3.,4.,5.][i]
        assert result[name]['m5_loss_vs_loss_arm']['state_m5_side']['future_d1_side']['dot']==[-2.,6.,-12.,20.][i]
        probe=result[name]['m5_mean_logit_vs_loss_arm']['state_m5_mean_logit']['future_d1_side']
        assert probe=={'dot':[-1.,-4.,-9.,-16.][i],'cosine':-1.}
        assert result[name]['m5_loss_vs_loss_arm']['future_m5_side']['activity']['cosine'] is None
    assert all(torch.equal(before[k],v) for k,v in vectors.items())
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('bad',['arm','dtype','shape','nan'])
def test_summary_rejects_invalid_retained_vectors(bad):
    vectors={arm:torch.zeros(17280,dtype=torch.float64) for arm in helper.ARMS}
    if bad=='arm':vectors['extra']=vectors['total']
    if bad=='dtype':vectors['total']=vectors['total'].float()
    if bad=='shape':vectors['total']=vectors['total'][:-1]
    if bad=='nan':vectors['total'][0]=float('nan')
    with pytest.raises(RuntimeError,match='VECTOR'):helper.summarize(vectors)


def test_wrong_projection_or_batch_rejected_before_autograd():
    projection,s,f=graph();counts={'autograd_grad_calls':0}
    with pytest.raises(RuntimeError,match='PROJECTION'):
        helper.measure_query_projection_partition(report(s,f,s.sum()),torch.nn.Linear(133,128),counts=counts,torch=torch)
    with pytest.raises(RuntimeError,match='LOGITS'):
        helper.measure_query_projection_partition(report(s[:15],f[:15],s.sum()),projection,counts=counts,torch=torch)
    assert counts['autograd_grad_calls']==0
