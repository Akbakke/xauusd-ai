"""Synthetic analytic checks for the read-only gradient instrument, not market evidence."""
import pytest
import torch
from torch import nn
from gx1.scripts.research_ta_campaign_v1 import entry_edge_gradient_summary

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.family_tf_context_gate = nn.Linear(2, 2, bias=False)
        self.family_tf_token_gate = nn.Linear(2, 2, bias=False)
        self.head_entry_action_q = nn.Linear(2, 1, bias=False)
    def forward(self, x):
        shared = self.family_tf_context_gate(self.encoder(x))
        return self.head_entry_action_q(shared), shared

def test_actual_vjps_distinguish_opposition_unused_and_zero_without_mutation():
    torch.manual_seed(7)
    model = Model()
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    q, shared = model(torch.tensor([[1.,2.],[2.,1.]]))
    loss = q.sum()
    result = entry_edge_gradient_summary(model, {
        "entry":loss, "contrast":loss, "common":loss*0, "flat":loss*0,
        "aux:opposed":-loss, "aux:zero":loss*0})
    for group in ("shared","routing","entry_head"):
        assert result["alignment"][group]["entry_auxiliary_cosine"] == pytest.approx(-1.)
        assert result["alignment"][group]["contrast_dot_joint"] == pytest.approx(0.)
        assert result["gradients"]["common"][group]["status"] == "connected_zero"
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters())

def test_auxiliary_without_head_connection_is_reported_unused():
    model=Model()
    q,shared=model(torch.tensor([[1.,2.]]))
    result=entry_edge_gradient_summary(model, {
        "entry":q.sum(), "contrast":q.sum(), "common":q.sum()*0, "flat":q.sum()*0,
        "aux:observed":shared.square().sum()})
    assert result["gradients"]["aux:observed"]["entry_head"]["status"] == "unused"
    assert result["gradients"]["aux:observed"]["shared"]["status"] == "nonzero"

def test_nonfinite_losses_and_existing_gradients_reject():
    model=Model()
    q,_=model(torch.ones(1,2))
    with pytest.raises(RuntimeError, match="LOSS_INVALID"):
        entry_edge_gradient_summary(model, {"entry":q.sum()*float("nan")})
    model.encoder.weight.grad=torch.zeros_like(model.encoder.weight)
    with pytest.raises(RuntimeError, match="SURFACE_INVALID"):
        entry_edge_gradient_summary(model, {"entry":q.sum()})
