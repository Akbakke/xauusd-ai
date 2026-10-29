import pytest
import torch
from torch.nn import functional as F
from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_binary_loss, dense_direction_research_contract
from gx1.contracts.entry_native_direction_five_priority_v1 import five_priority_direction_binary_loss, five_priority_direction_contract

MINUTE = 60_000_000_000


def batch():
    torch.manual_seed(7)
    z = torch.randn((3,60), dtype=torch.float32, requires_grad=True)
    sign = torch.where(torch.rand((3,60)) > .5, 1, -1).to(torch.int8)
    observed = torch.ones_like(sign,dtype=torch.bool)
    observed[0,5] = False; sign[0,5] = 0; sign[1,3] = 0
    d = torch.tensor([10**15,10**15+MINUTE,10**15+2*MINUTE],dtype=torch.int64)
    terminal = d[:,None] + torch.arange(1,61,dtype=torch.int64)[None,:]*MINUTE
    return z,sign,observed,d,terminal,int(d[-1]+45*MINUTE)


def test_weighted_loss_and_gradients_match_direct_bce_with_original_mask():
    z,s,o,d,t,c = batch()
    base = dense_direction_binary_loss(z, sign=s, observed=o, decision_ns=d, terminal_time_ns=t, cutoff_ns=c)
    candidate = five_priority_direction_binary_loss(z, sign=s, observed=o, decision_ns=d, terminal_time_ns=t, cutoff_ns=c)
    mask=base['eligible_mask']; weights=torch.ones_like(z); weights[:,4]=10
    expected=F.binary_cross_entropy_with_logits(z[mask],(s[mask]==1).to(z.dtype),weight=weights[mask],reduction='sum')/weights[mask].sum()
    assert torch.equal(mask,candidate['eligible_mask'])
    assert candidate['counts']==base['counts'] and candidate['per_horizon_counts']==base['per_horizon_counts']
    assert torch.equal(candidate['loss'],expected)
    g=torch.autograd.grad(candidate['loss'],z,retain_graph=True)[0]
    e=torch.autograd.grad(expected,z)[0]
    assert torch.equal(g,e) and torch.all(g[~mask] == 0)
    assert torch.all(g[mask] != 0)
    assert candidate['research_objective_sha256']==five_priority_direction_contract()['contract_sha256']
    assert candidate['research_objective_sha256']!=dense_direction_research_contract()['contract_sha256']


def test_all_zero_labels_produce_no_loss_or_update():
    z,s,o,d,t,c=batch();s.zero_()
    result=five_priority_direction_binary_loss(z,sign=s,observed=o,decision_ns=d,terminal_time_ns=t,cutoff_ns=c)
    assert result['loss'] is None and result['update_allowed'] is False
    assert result['counts']['eligible_cells']==0 and z.grad is None


def test_invalid_future_clock_and_unobserved_values_rejected():
    z,s,o,d,t,c=batch()
    with pytest.raises(RuntimeError):five_priority_direction_binary_loss(z,sign=s,observed=o,decision_ns=d,terminal_time_ns=t+1,cutoff_ns=c)
    s[0,5]=1
    with pytest.raises(RuntimeError):five_priority_direction_binary_loss(z,sign=s,observed=o,decision_ns=d,terminal_time_ns=t,cutoff_ns=c)


def test_priority_scales_five_minute_gradient_ten_times_other_heads():
    z,s,o,d,t,c=batch()
    b=dense_direction_binary_loss(z,sign=s,observed=o,decision_ns=d,terminal_time_ns=t,cutoff_ns=c)
    v=five_priority_direction_binary_loss(z,sign=s,observed=o,decision_ns=d,terminal_time_ns=t,cutoff_ns=c)
    gb=torch.autograd.grad(b['loss'],z,retain_graph=True)[0];gv=torch.autograd.grad(v['loss'],z)[0]
    factor=b['counts']['eligible_cells']/(b['counts']['eligible_cells']+9*b['per_horizon_counts'][4]['eligible_cells'])
    assert torch.allclose(gv[:,4], gb[:,4]*factor*10,rtol=1e-6,atol=1e-8)
    assert torch.allclose(gv[:,:4],gb[:,:4]*factor,rtol=1e-6,atol=1e-8)
    assert torch.allclose(gv[:,5:],gb[:,5:]*factor,rtol=1e-6,atol=1e-8)
