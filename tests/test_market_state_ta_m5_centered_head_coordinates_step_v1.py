"""Small CPU coordinate/optimizer tests; no native model, data, or checkpoint.

The explicit reference really forwards through (h - mu, a), while the helper
keeps the ordinary forward and only changes optimizer coordinates. FP32 bounds
allow bias rebinding and the different backward reduction order over 128 terms.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gx1.audit.run_market_state_ta_m5_centered_head_coordinates_v1 import (
    centered_head_optimizer_step,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


ATOL, RTOL = 3e-7, 5e-6


class Toy(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.coupled = nn.Parameter(torch.tensor(0.2, dtype=dtype))
        self.state_model = nn.Module()
        self.state_model.m5_side_head = nn.Linear(128, 1, dtype=dtype)
        self.future_m5_side_head = nn.Linear(128, 1, dtype=dtype)
        with torch.no_grad():
            k = torch.arange(128, dtype=dtype)
            self.state_model.m5_side_head.weight.copy_(((k % 9 - 4) * .002)[None])
            self.future_m5_side_head.weight.copy_(((k % 7 - 3) * -.003)[None])
            self.state_model.m5_side_head.bias.fill_(.08)
            self.future_m5_side_head.bias.fill_(-.07)


def heads(model):
    return {"state": model.state_model.m5_side_head,
            "future": model.future_m5_side_head}


def means(dtype=torch.float32):
    return {"state": torch.linspace(.5, 1.5, 128, dtype=dtype),
            "future": torch.linspace(-.6, .9, 128, dtype=dtype)}


def inputs(dtype=torch.float32, step=0):
    return torch.linspace(.2, 1.7, 512, dtype=dtype).reshape(4, 128) + step * .03


def raw(model, x, mu=None):
    h = x + model.coupled * torch.linspace(-.4, .6, 128, dtype=x.dtype)
    return torch.cat([head(h if mu is None else h - mu[task])
                      for task, head in heads(model).items()], dim=1)


def loss(model, x, mu=None):
    target = torch.tensor([[1, 0], [1, 0], [0, 0], [1, 1]], dtype=x.dtype)
    return 17 * F.binary_cross_entropy_with_logits(raw(model, x, mu), target) \
        + .4 * (model.coupled + 1.2).square()


def centered_reference(model, mu):
    reference = deepcopy(model)
    with torch.no_grad():
        for task, head in heads(reference).items():
            # Independent higher-precision affine conversion, then FP32 storage.
            a = head.bias.double() + head.weight.double() @ mu[task].double()
            head.bias.copy_(a)
    return reference


def optimizer(model, populated=False):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.7, .93),
                            eps=2e-6, weight_decay=.04, foreach=False, fused=False)
    if populated:
        for p in model.parameters():
            k = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)
            opt.state[p] = {"step": torch.tensor(4.),
                            "exp_avg": .003 * (k.remainder(7) - 3),
                            "exp_avg_sq": .002 + .0001 * k.remainder(5)}
    return opt


def hyperparameters(opt):
    return [{k: deepcopy(v) for k, v in group.items() if k != "params"}
            for group in opt.param_groups]


def explicit_step(model, opt):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return norm


def canonical_step(model, opt):
    return trainer._optimizer_step_with_finite_gradients(model=model, optimizer=opt)


def assert_matches(model, opt, reference, refopt, mu):
    expected = dict(reference.named_parameters())
    for name, p in model.named_parameters():
        q = expected[name]
        value = q
        if name == "state_model.m5_side_head.bias":
            value = q.double() - reference.state_model.m5_side_head.weight.double() @ mu["state"].double()
        elif name == "future_m5_side_head.bias":
            value = q.double() - reference.future_m5_side_head.weight.double() @ mu["future"].double()
        torch.testing.assert_close(p.double(), value.double(), atol=ATOL, rtol=RTOL)
        assert set(opt.state[p]) == set(refopt.state[q])
        for key in opt.state[p]:
            if key == "step":
                assert torch.equal(opt.state[p][key], refopt.state[q][key])
            else:
                torch.testing.assert_close(opt.state[p][key], refopt.state[q][key], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(raw(model, inputs()), raw(reference, inputs(), mu), atol=2e-6, rtol=3e-6)


def test_affine_identity_preserves_logits_and_upstream_gradient():
    model = Toy(torch.float64)
    mu = means(torch.float64)
    reference = centered_reference(model, mu)
    x = inputs(torch.float64).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    torch.testing.assert_close(raw(model, x), raw(reference, xr, mu), atol=2e-13, rtol=2e-13)
    loss(model, x).backward()
    loss(reference, xr, mu).backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(model.coupled.grad, reference.coupled.grad, atol=2e-13, rtol=2e-13)
    for task, head in heads(model).items():
        refhead = heads(reference)[task]
        torch.testing.assert_close(refhead.bias.grad, head.bias.grad, atol=2e-13, rtol=2e-13)
        torch.testing.assert_close(refhead.weight.grad,
                                   head.weight.grad - head.bias.grad[:, None] * mu[task],
                                   atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("populated", [False, True], ids=["empty_adam", "existing_moments"])
def test_two_steps_match_explicit_centered_adamw_with_global_clip(populated):
    model, mu = Toy(), means()
    reference = centered_reference(model, mu)
    opt, refopt = optimizer(model, populated), optimizer(reference, populated)
    original_hyper = hyperparameters(opt)
    parameter_ids = [id(p) for p in opt.param_groups[0]["params"]]
    for step in range(2):
        loss(model, inputs(step=step)).backward()
        loss(reference, inputs(step=step), mu).backward()
        raw_norm = torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters()))
        expected_norm = torch.sqrt(sum(p.grad.double().square().sum() for p in reference.parameters()))
        assert expected_norm > 1 and abs(raw_norm - expected_norm) > .01
        coupled_grad = model.coupled.grad.clone()
        calls = []
        def callback():
            calls.append(1)
            torch.testing.assert_close(model.coupled.grad, coupled_grad, atol=0, rtol=0)
            for task, head in heads(model).items():
                torch.testing.assert_close(head.weight.grad, heads(reference)[task].weight.grad,
                                           atol=2e-6, rtol=3e-5)
            return canonical_step(model, opt)
        result = centered_head_optimizer_step(model=model, optimizer=opt, means=mu, optimizer_step=callback)
        norm = explicit_step(reference, refopt)
        assert calls == [1]
        assert float(result["gradient_norm"]) == pytest.approx(float(norm), rel=3e-6, abs=2e-6)
        assert_matches(model, opt, reference, refopt, mu)
        assert hyperparameters(opt) == original_hyper
        assert [id(p) for p in opt.param_groups[0]["params"]] == parameter_ids
        assert all(p.grad is None for p in model.parameters())


def test_connected_zero_gradients_keep_existing_adam_history():
    model, mu = Toy(), means()
    reference = centered_reference(model, mu)
    opt, refopt = optimizer(model, True), optimizer(reference, True)
    old_m = {name: opt.state[p]["exp_avg"].clone() for name, p in model.named_parameters()}
    for current in (model, reference):
        sum(p.sum() * 0 for p in current.parameters()).backward()
    result = centered_head_optimizer_step(model=model, optimizer=opt, means=mu,
                                         optimizer_step=lambda: canonical_step(model, opt))
    explicit_step(reference, refopt)
    assert float(result["gradient_norm"]) == 0
    assert_matches(model, opt, reference, refopt, mu)
    for name, p in model.named_parameters():
        assert float(opt.state[p]["step"]) == 5
        torch.testing.assert_close(opt.state[p]["exp_avg"], .7 * old_m[name], atol=1e-9, rtol=1e-6)


def test_uniform_decoupled_decay_commutes_with_affine_coordinates():
    model, mu = Toy(), means()
    opt = optimizer(model)
    # Make pure decay much larger than the FP32 comparison bound.
    opt.param_groups[0]["lr"] = .02
    opt.param_groups[0]["weight_decay"] = .3
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    sum(p.sum() * 0 for p in model.parameters()).backward()
    result = centered_head_optimizer_step(model=model, optimizer=opt, means=mu,
                                         optimizer_step=lambda: canonical_step(model, opt))
    factor = 1 - opt.param_groups[0]["lr"] * opt.param_groups[0]["weight_decay"]
    assert float(result["gradient_norm"]) == 0
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name] * factor, atol=1e-7, rtol=1e-6)
        assert float(opt.state[p]["step"]) == 1
        assert torch.count_nonzero(opt.state[p]["exp_avg"]) == 0
        assert torch.count_nonzero(opt.state[p]["exp_avg_sq"]) == 0


@pytest.mark.parametrize("after_step", [False, True], ids=["before_step", "after_step"])
def test_callback_exception_restores_ordinary_bias_without_claiming_rollback(after_step):
    model, mu = Toy(), means()
    reference = centered_reference(model, mu)
    opt, refopt = optimizer(model, True), optimizer(reference, True)
    loss(model, inputs()).backward()
    loss(reference, inputs(), mu).backward()
    original_hyper = hyperparameters(opt)
    calls = []
    def fail():
        calls.append(1)
        for task, head in heads(model).items():
            torch.testing.assert_close(head.bias, heads(reference)[task].bias, atol=ATOL, rtol=RTOL)
        if after_step:
            canonical_step(model, opt)
        raise RuntimeError("injected callback failure")
    with pytest.raises(RuntimeError, match="injected callback failure"):
        centered_head_optimizer_step(model=model, optimizer=opt, means=mu, optimizer_step=fail)
    if after_step:
        explicit_step(reference, refopt)
    assert calls == [1]
    assert_matches(model, opt, reference, refopt, mu)
    assert hyperparameters(opt) == original_hyper


@pytest.mark.parametrize("location", ["head", "coupled"])
def test_nonfinite_gradient_fails_before_optimizer_and_restores_bias(location):
    model, mu = Toy(), means()
    opt = optimizer(model, True)
    loss(model, inputs()).backward()
    bad = model.future_m5_side_head.weight if location == "head" else model.coupled
    bad.grad.reshape(-1)[0] = float("inf")
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    state_before = {name: {k: v.clone() for k, v in opt.state[p].items()}
                    for name, p in model.named_parameters()}
    step_calls = []
    handle = opt.register_step_pre_hook(lambda *_: step_calls.append(1))
    try:
        with pytest.raises(RuntimeError):
            centered_head_optimizer_step(model=model, optimizer=opt, means=mu,
                                         optimizer_step=lambda: canonical_step(model, opt))
    finally:
        handle.remove()
    assert step_calls == []
    for name, p in model.named_parameters():
        if location == "head":
            assert torch.equal(p, parameters_before[name])
        else:
            torch.testing.assert_close(p, parameters_before[name], atol=ATOL, rtol=RTOL)
        for key in state_before[name]:
            assert torch.equal(opt.state[p][key], state_before[name][key])
