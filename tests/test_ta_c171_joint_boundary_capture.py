"""Synthetic CPU boundaries only: no GX1 data, checkpoints, model or fit."""
import numpy as np
import pytest
import torch

from gx1.audit.ta_c171_joint_boundary_capture_v1 import capture_joint_boundaries


class BoundaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_model = torch.nn.Module()
        self.state_model.backbone = torch.nn.Module()
        self.state_model.backbone.entry_q_joint_norm = torch.nn.LayerNorm(512)
        self.state_model.backbone.entry_q_joint_in = torch.nn.Linear(512, 128)
        self.state_model.query_projection = torch.nn.Sequential(
            torch.nn.Linear(134, 128), torch.nn.GELU())
        self.state_model.register_buffer("query_tf_coordinates", torch.eye(5))
        self.dropout = torch.nn.Dropout(.2)

    def queries(self, hidden):
        tf = self.state_model.query_tf_coordinates[None].expand(16, 5, 5)
        fractions = hidden.new_tensor([1., .5, .25, .125, 1.])[None, :, None].expand(16, 5, 1)
        common = hidden[:, None, :].expand(16, 5, 128)
        state = torch.cat((common, tf, fractions), dim=-1)[:, None]
        future = torch.cat((common, tf, torch.ones_like(fractions)), dim=-1)
        return state, future

    def forward(self, source):
        source = self.dropout(source)
        hidden = self.state_model.backbone.entry_q_joint_in(
            self.state_model.backbone.entry_q_joint_norm(source)).tanh()
        state, future = self.queries(hidden)
        return (self.state_model.query_projection(state),
                self.state_model.query_projection(future))


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        return BoundaryModel()


def payloads(m):
    source = torch.arange(16 * 512, dtype=torch.float32).reshape(16, 512) / 1024
    hidden = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128) / 128
    state, future = m.queries(hidden)
    return source, state, future


def hooks(m):
    return {id(v): (tuple(v._forward_pre_hooks.items()), tuple(v._forward_hooks.items()))
            for v in m.modules()}


def emit(m, values):
    source, state, future = values
    m.state_model.backbone.entry_q_joint_norm(source)
    m.state_model.query_projection(state)
    m.state_model.query_projection(future)


def test_passive_outputs_rng_state_grad_flags_modes_hooks_and_owned_copies():
    with torch.random.fork_rng(devices=[]):
        m = model().train()
        m.state_model.backbone.entry_q_joint_norm.eval()
        next(m.parameters()).requires_grad_(False)
        next(m.state_model.query_projection.parameters()).grad = torch.ones(128, 134)
        before_state = {k: v.clone() for k, v in m.state_dict().items()}
        before_params = [(id(p), p.requires_grad, None if p.grad is None else p.grad.clone())
                         for p in m.parameters()]
        modes = [v.training for v in m.modules()]
        existing = m.state_model.query_projection.register_forward_pre_hook(lambda *args: None)
        before_hooks = hooks(m)
        source = payloads(m)[0].requires_grad_()
        rng = torch.get_rng_state().clone()
        expected = m(source)
        expected_rng = torch.get_rng_state().clone()
        expected_grad, = torch.autograd.grad(sum(v.sum() for v in expected), source)
        torch.set_rng_state(rng)
        counts = {"joint_source_captures": 7, "joint_hidden_captures": 8,
                  "query_input_observations": 9, "unrelated": 12}
        with capture_joint_boundaries(m, counts=counts, torch=torch) as captured:
            actual = m(source)
        actual_grad, = torch.autograd.grad(sum(v.sum() for v in actual), source)
        assert all(torch.equal(x, y) for x, y in zip(actual, expected))
        assert torch.equal(actual_grad, expected_grad)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert all(torch.equal(v, before_state[k]) for k, v in m.state_dict().items())
        for p, (identity, flag, grad) in zip(m.parameters(), before_params):
            assert id(p) == identity and p.requires_grad == flag
            assert p.grad is None if grad is None else torch.equal(p.grad, grad)
        assert [v.training for v in m.modules()] == modes and hooks(m) == before_hooks
        assert counts == {"joint_source_captures": 8, "joint_hidden_captures": 9,
                          "query_input_observations": 11, "unrelated": 12}
        assert {k: v.shape for k, v in captured.items()} == {
            "joint_source": (16, 512), "joint_hidden": (16, 128)}
        assert all(v.dtype == np.float32 and v.flags.owndata for v in captured.values())
        copied = {k: v.copy() for k, v in captured.items()}
        source.detach().add_(100)
        assert all(np.array_equal(captured[k], v) for k, v in copied.items())
        captured["joint_hidden"].fill(100)
        assert all(torch.equal(x, y) for x, y in zip(actual, expected))
        existing.remove()


@pytest.mark.parametrize("fault,error,partial,expected_counts", [
    ("source_shape", "TENSOR:joint_source", set(), (1, 0, 0)),
    ("source_dtype", "TENSOR:joint_source", set(), (1, 0, 0)),
    ("source_nan", "NONFINITE:joint_source", set(), (1, 0, 0)),
    ("state_shape", "TENSOR:state", {"joint_source"}, (1, 1, 1)),
    ("state_dtype", "TENSOR:state", {"joint_source"}, (1, 1, 1)),
    ("state_inf", "NONFINITE:state", {"joint_source"}, (1, 1, 1)),
    ("state_tf", "TF_INPUT", {"joint_source"}, (1, 1, 1)),
    ("state_hidden_tf", "HIDDEN_TIMEFRAMES", {"joint_source"}, (1, 1, 1)),
    ("state_hidden_signed_zero", "HIDDEN_TIMEFRAMES", {"joint_source"}, (1, 1, 1)),
    ("state_fraction_zero", "QUERY_FRACTIONS", {"joint_source"}, (1, 1, 1)),
    ("state_fraction_over", "QUERY_FRACTIONS", {"joint_source"}, (1, 1, 1)),
    ("state_m5_fraction", "QUERY_FRACTIONS", {"joint_source"}, (1, 1, 1)),
    ("future_hidden", "HIDDEN_STATE_FUTURE", {"joint_source", "joint_hidden"}, (1, 1, 2)),
    ("future_tf", "TF_INPUT", {"joint_source", "joint_hidden"}, (1, 1, 2)),
    ("future_fraction", "QUERY_FRACTIONS", {"joint_source", "joint_hidden"}, (1, 1, 2)),
])
def test_rejects_tensor_and_semantic_mismatches_preserving_partial(fault, error, partial, expected_counts):
    m = model();source, state, future = payloads(m)
    if fault == "source_shape": source = source[:15]
    elif fault == "source_dtype": source = source.double()
    elif fault == "source_nan": source[0, 0] = float("nan")
    elif fault == "state_shape": state = state[:, 0]
    elif fault == "state_dtype": state = state.double()
    elif fault == "state_inf": state[0, 0, 0, 0] = float("inf")
    elif fault == "state_tf": state[0, 0, 0, 128] = 0
    elif fault == "state_hidden_tf": state[0, 0, 1, 0] += 1
    elif fault == "state_hidden_signed_zero": state[0, 0, 1, 0] = -0.
    elif fault == "state_fraction_zero": state[0, 0, 1, 133] = 0
    elif fault == "state_fraction_over": state[0, 0, 1, 133] = 1.25
    elif fault == "state_m5_fraction": state[0, 0, 0, 133] = .5
    elif fault == "future_hidden": future[:, :, :128] += 1
    elif fault == "future_tf": future[0, 0, 128] = 0
    elif fault == "future_fraction": future[0, 1, 133] = .5
    before = hooks(m);counts = {}
    with pytest.raises(RuntimeError, match=error):
        with capture_joint_boundaries(m, counts=counts, torch=torch) as captured:
            emit(m, (source, state, future))
    assert set(captured) == partial and hooks(m) == before
    assert tuple(counts[k] for k in (
        "joint_source_captures", "joint_hidden_captures", "query_input_observations")) == expected_counts


@pytest.mark.parametrize("calls,error,expected_counts", [
    ("", "INCOMPLETE", (0, 0, 0)),
    ("n", "INCOMPLETE", (1, 0, 0)),
    ("ns", "INCOMPLETE", (1, 1, 1)),
    ("s", "QUERY_ORDER", (0, 1, 1)),
    ("nf", "TENSOR:state", (1, 1, 1)),
    ("nnsf", "SOURCE_ORDER", (2, 0, 0)),
    ("nssf", "TENSOR:future", (1, 1, 2)),
    ("nsff", "QUERY_ORDER", (1, 1, 3)),
])
def test_missing_duplicate_and_misordered_calls(calls, error, expected_counts):
    m = model();source, state, future = payloads(m);counts = {};before = hooks(m)
    actions = {"n": lambda: m.state_model.backbone.entry_q_joint_norm(source),
               "s": lambda: m.state_model.query_projection(state),
               "f": lambda: m.state_model.query_projection(future)}
    with pytest.raises(RuntimeError, match=error):
        with capture_joint_boundaries(m, counts=counts, torch=torch):
            for key in calls:
                actions[key]()
    assert hooks(m) == before
    assert tuple(counts[k] for k in (
        "joint_source_captures", "joint_hidden_captures", "query_input_observations")) == expected_counts


def test_swallowed_failure_rejects_exit_and_keeps_valid_copies():
    m = model();values = payloads(m);counts = {};before = hooks(m)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        with capture_joint_boundaries(m, counts=counts, torch=torch) as captured:
            emit(m, values)
            with pytest.raises(RuntimeError, match="QUERY_ORDER"):
                m.state_model.query_projection(values[2])
    assert set(captured) == {"joint_source", "joint_hidden"} and hooks(m) == before
    assert counts["query_input_observations"] == 3


def test_registration_and_caller_failure_remove_only_own_hooks(monkeypatch):
    m = model();existing = m.state_model.backbone.entry_q_joint_norm.register_forward_pre_hook(lambda *args: None)
    before = hooks(m)
    def reject(*args, **kwargs):
        raise ValueError("registration")
    with monkeypatch.context() as patch:
        patch.setattr(m.state_model.query_projection, "register_forward_pre_hook", reject)
        with pytest.raises(ValueError, match="registration"):
            with capture_joint_boundaries(m, counts={}, torch=torch):
                pytest.fail("registration admitted")
    assert hooks(m) == before
    with pytest.raises(ValueError, match="caller"):
        with capture_joint_boundaries(m, counts={}, torch=torch) as captured:
            emit(m, payloads(m))
            raise ValueError("caller")
    assert set(captured) == {"joint_source", "joint_hidden"} and hooks(m) == before
    existing.remove()


@pytest.mark.parametrize("fault,error", [
    ("norm", "MODULE_LAYOUT"), ("coordinates", "TF_COORDINATES"), ("counts", "COUNTS")
])
def test_invalid_setup_registers_no_hooks(fault, error):
    m = model();counts = {}
    if fault == "norm": m.state_model.backbone.entry_q_joint_norm = torch.nn.LayerNorm(128)
    elif fault == "coordinates": m.state_model.query_tf_coordinates[0, 0] = 2
    else: counts["joint_hidden_captures"] = True
    before = hooks(m)
    with pytest.raises(RuntimeError, match=error):
        with capture_joint_boundaries(m, counts=counts, torch=torch):
            pytest.fail("invalid setup admitted")
    assert hooks(m) == before
