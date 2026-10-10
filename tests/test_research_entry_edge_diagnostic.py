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

def test_entry_edge_purge_uses_outcome_availability_not_row_order():
    import pandas as pd
    import numpy as np
    from gx1.scripts.research_ta_campaign_v1 import entry_edge_fold_rows
    clock = pd.date_range("2013-12-29", "2014-01-03", freq="5min", tz="UTC")
    start, end = pd.Timestamp("2014-01-01", tz="UTC"), pd.Timestamp("2014-01-02", tz="UTC")
    fit, hold = entry_edge_fold_rows(clock, start, end, 5700, clock[0])
    assert clock[fit[-1]] == start - pd.Timedelta(minutes=105)
    assert clock[hold[0]] == start - pd.Timedelta(minutes=5)
    assert clock[hold[-1]] + pd.Timedelta(minutes=100) < end
    assert not np.intersect1d(fit, hold).size


def test_entry_edge_unique_positive_action_and_nonfinite_rejection():
    import numpy as np
    from gx1.scripts.research_ta_campaign_v1 import entry_edge_actions
    np.testing.assert_array_equal(entry_edge_actions([[1,0],[0,1],[1,1],[-1,-2],[0,0]]), [1,-1,0,0,0])
    with pytest.raises(RuntimeError, match="PREDICTION_INVALID"):
        entry_edge_actions([[float("nan"), 1]])


def test_entry_edge_economics_counts_open_position_and_executable_exit_cost(tmp_path):
    import json
    import hashlib
    import numpy as np
    import pandas as pd
    from gx1.scripts.research_ta_campaign_v1 import _entry_edge_portfolios
    times = pd.date_range("2014-01-01", periods=4, freq="min", tz="UTC")
    quotes = pd.DataFrame({"time":times, "open":[100.,101.,102.,103.],
                           "bid_open":[99.9,100.9,101.9,102.9],
                           "ask_open":[100.1,101.1,102.1,103.1]})
    tape = tmp_path / "m1.parquet"
    quotes.to_parquet(tape, index=False)
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({"parameters":{"execution_slippage":{"central_bps_per_execution":1.},
                                              "commission":{"bps_per_execution":1.}}}))
    def bind(p):
        return {"path":str(p), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    spec = {"m1_source":bind(tape), "cost_authority":bind(costs),
            "train_end_exclusive":"2014-01-01T00:04:00Z"}
    frame = pd.DataFrame({"decision_time":times[[0,2]], "constant_action":[0,0],
                          "additive_action":[1,-1], "interaction_action":[1,1]})
    scope = {"economics":{"annual_financing_cost_bps":[0.,0.], "seconds_per_year":31557600.}}
    result, monthly = _entry_edge_portfolios(frame, spec, scope, tmp_path)
    expected = (102.9*(1-2e-4) - 100.1*(1+2e-4)) / 100.1 * 1e4
    assert result["interaction"]["total_net_bps"] == pytest.approx(expected)
    assert result["interaction"]["open_units"] == 1
    assert result["interaction"]["position_entries"] == 1
    assert result["interaction"]["liquidation_reserve_bps"] > 0
    assert result["flat"]["total_net_bps"] == 0
    path = pd.read_parquet(tmp_path / "PORTFOLIO_additive.parquet")
    np.testing.assert_array_equal(path.held_units_after, [1,1,-1,-1])
    assert result["additive"]["position_entries"] == 2
    assert monthly["interaction"].sum() == pytest.approx(expected)
