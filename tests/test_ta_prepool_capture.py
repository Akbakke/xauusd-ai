"""Passive observation and terminal reporting; no native model or actual inputs."""
import json
import numpy as np
import pytest
import torch
from torch import nn
from gx1.audit import observe_market_state_ta_head_input_v1 as owner


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.tf_token_gate=nn.Linear(128,1)
        self.state_model=nn.Module()
        self.state_model.query_projection=nn.Sequential(nn.Linear(134,128),nn.GELU())

    def forward(self,x):
        pooled=x.mean(1)[:,None,:].expand(16,5,128)
        q=torch.cat((pooled,torch.zeros(16,5,6)),dim=-1)
        return (self.backbone.tf_token_gate(x),self.state_model.query_projection(q[:,None]),self.state_model.query_projection(q))


def test_hooks_preserve_outputs_parameters_rng_and_capture_actual_boundary():
    torch.manual_seed(7);m=Tiny().eval();x=torch.randn(16,5,128)
    before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.random.get_rng_state().clone()
    with torch.no_grad():plain=m(x)
    observed={};counts=dict(prepool_hook_calls=0,projection_input_hook_calls=0)
    handles=owner._prepool_hooks(m,observed,counts,np=np)
    try:
        with torch.no_grad():actual=m(x)
    finally:
        for h in handles:h.remove()
    assert all(torch.equal(a,b) for a,b in zip(plain,actual))
    assert np.array_equal(observed['prepool'],x.numpy())
    assert np.array_equal(observed['state_hidden'],x.mean(1).numpy())
    assert np.array_equal(observed['state_hidden'],observed['future_hidden'])
    assert counts==dict(prepool_hook_calls=1,projection_input_hook_calls=2)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    assert torch.equal(rng,torch.random.get_rng_state())
    assert not m.backbone.tf_token_gate._forward_pre_hooks
    assert not m.state_model.query_projection[0]._forward_pre_hooks


@pytest.mark.parametrize('kind',['shape','nan','duplicate'])
def test_bad_or_duplicate_capture_fails_and_hooks_can_be_removed(kind):
    m=Tiny();observed={};counts=dict(prepool_hook_calls=0,projection_input_hook_calls=0)
    handles=owner._prepool_hooks(m,observed,counts,np=np)
    x=torch.ones(16,5,128)
    if kind=='shape':x=x[:1]
    if kind=='nan':x[0,0,0]=float('nan')
    if kind=='duplicate':observed['prepool']=np.zeros((16,5,128),np.float32)
    try:
        with pytest.raises(RuntimeError,match='PREPOOL_'):
            m.backbone.tf_token_gate(x)
    finally:
        for h in handles:h.remove()
    assert not m.backbone.tf_token_gate._forward_pre_hooks
    assert counts['prepool_hook_calls']==0


@pytest.mark.parametrize('complete',[False,True])
def test_existing_publisher_retains_complete_or_failed_consumed_terminal(tmp_path,complete):
    (tmp_path/'CLAIM.json').write_text('{}')
    result=owner.cache._publish_fusion_outcome(
        dict(path=str(tmp_path/'PLAN.json'),sha256='a'*64),
        dict(source_unchanged=True,counts={k:0 for k in owner.PREPOOL_COUNTS},
             passivity={},interpretation_authorized=False),
        complete=complete,root=tmp_path,json_bytes_bound=owner.PREPOOL_BOUNDS['max_published_json_bytes'])
    payload=json.loads((tmp_path/'RESULT.json').read_text());terminal=json.loads((tmp_path/'TERMINAL.json').read_text())
    assert payload['status']==('COMPLETE' if complete else 'FAILED_CONSUMED')
    assert terminal['exit_code']==(0 if complete else 1)
    assert terminal['result']==result['result']
    assert terminal['no_relaunch_under_this_plan'] is True
    assert payload['training_enabled'] is payload['predictive_value_claimed'] is False
