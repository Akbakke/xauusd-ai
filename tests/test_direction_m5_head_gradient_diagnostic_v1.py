import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gx1.audit.direction_m5_head_gradient_diagnostic_v1 import gradients, reordered_weights


def test_head_reorder_preserves_one_five_and_fifteen_anchors():
    anchor = np.repeat(np.arange(3)[:, None], 4, axis=1)
    aux = np.repeat(np.arange(3, 60)[:, None], 4, axis=1)
    weights = reordered_weights(anchor, aux)
    assert tuple(weights[[0, 4, 14], 0]) == (0, 1, 2)
    assert sorted(weights[:, 0].tolist()) == list(range(60))
    with pytest.raises(RuntimeError): reordered_weights(anchor, aux[:56])


def test_hidden_gradients_match_autograd_with_missing_ties_and_maturity():
    rng = np.random.default_rng(17)
    rows, width, batch_size = 7, 6, 3
    weights = rng.normal(size=(60, width))
    hidden = rng.normal(size=(rows, width))
    logits = hidden @ weights.T + rng.normal(size=(rows, 60))
    sign = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(rows, 60))
    observed = rng.random((rows,60)) > 0.11; sign[~observed] = 0
    decisions = np.arange(rows, dtype=np.int64) * 60_000_000_000 + 10**15
    terminals = decisions[:, None] + np.arange(1,61,dtype=np.int64)[None,:]*60_000_000_000
    cutoff = int(decisions[-1] + 40 * 60_000_000_000)
    actual = gradients(logits, sign, observed, terminals, decisions, cutoff, weights, batch_size=batch_size)
    total5 = totalother = totaldot = totalcombined = 0.0
    elig = observed & (sign != 0) & (terminals < cutoff)
    for start in range(0,rows,batch_size):
        end = min(start+batch_size,rows); active = torch.tensor(elig[start:end]); n = int(active.sum())
        x = torch.tensor(hidden[start:end], dtype=torch.float64, requires_grad=True)
        w = torch.tensor(weights, dtype=torch.float64)
        biases = torch.tensor(logits[start:end] - hidden[start:end] @ weights.T, dtype=torch.float64)
        z = x @ w.T + biases
        labels = torch.tensor(sign[start:end] == 1, dtype=torch.float64)
        raw = F.binary_cross_entropy_with_logits(z, labels, reduction='none') * active / n
        g5 = torch.autograd.grad(raw[:,4].sum(), x, retain_graph=True)[0]
        gother = torch.autograd.grad(raw.sum()-raw[:,4].sum(), x)[0]
        total5 += (g5*g5).sum().item(); totalother += (gother*gother).sum().item()
        totaldot += (g5*gother).sum().item(); totalcombined += ((g5+gother)**2).sum().item()
    assert actual['five_norm'] == pytest.approx(total5**.5, abs=1e-14)
    assert actual['other_norm'] == pytest.approx(totalother**.5, abs=1e-14)
    assert actual['five_dot_other'] == pytest.approx(totaldot, abs=1e-14)
    assert actual['combined_norm'] == pytest.approx(totalcombined**.5, abs=1e-14)
    assert actual['five_vs_other_cosine'] == pytest.approx(totaldot/(total5*totalother)**.5, abs=1e-14)
    json.dumps(actual, allow_nan=False)  # Actual artifact must contain JSON-native scalar types.
    assert actual['eligible_cells'] == int(elig.sum())
    assert actual['eligible_five_minute_cells'] == int(elig[:,4].sum())


def test_bad_future_clock_and_missing_value_fail_closed():
    d = np.array([10**15],dtype=np.int64); t=d[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*60_000_000_000
    z=np.zeros((1,60));s=np.ones((1,60),dtype=np.int8);o=np.ones((1,60),dtype=bool);w=np.ones((60,4))
    with pytest.raises(RuntimeError): gradients(z,s,o,t+1,d,int(t.max()+1),w)
    o[0,0]=False
    with pytest.raises(RuntimeError): gradients(z,s,o,t,d,int(t.max()+1),w)
