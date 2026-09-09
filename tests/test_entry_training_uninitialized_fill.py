from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from gx1.contracts import entry_training_precision_v1 as policy
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_entry_v10_ctx_model_shapes import (
    _make_exit_episode_inputs,
    _make_inputs,
    _make_model,
)

POLICY = policy.EXPERIMENTAL_FP32_3090_NO_UNINITIALIZED_FILL


@pytest.fixture(autouse=True)
def restore_process_settings(monkeypatch):
    rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(
        trainer, "_TRAINING_PRECISION_POLICY", trainer._TRAINING_PRECISION_POLICY
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        yield
    finally:
        torch.set_rng_state(rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _equal_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _equal_tree(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_tree(a, b)
    else:
        assert left == right


def test_actual_model_full_episode_backward_adamw_and_ema_match_with_fill_on_and_off():
    """Real model/library parity on CPU; canonical CUDA TRAIN/VAL is separate evidence."""

    def run(fill):
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.manual_seed(20260908)
        trainer._TRAINING_PRECISION_POLICY = (
            policy.DETERMINISTIC_FP32 if fill else POLICY
        )
        model = _make_model(dropout=0.05).train()
        initial = copy.deepcopy(model.state_dict())
        assert all(
            torch.isfinite(t).all() for t in initial.values() if t.is_floating_point()
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0003, weight_decay=0.00001
        )
        ema = trainer._WeightEma(model, decay=0.9)
        seq, snap, cat, cont, mtf = _make_inputs(batch_size=1)
        entry = trainer._model_forward_fp32(
            model, seq, snap, ctx_cat=cat, ctx_cont=cont, **mtf
        )
        exit_inputs = _make_exit_episode_inputs(batch_size=1)
        exit_inputs["entry_decision_representation"] = entry[
            "entry_decision_representation"
        ].detach()
        episode = model.forward_exit_episode(**exit_inputs)
        heads = [
            "entry_action_q_bps",
            "side_mae_bps",
            "trendline_event_logits",
            "position_size_logit",
            "dip_pred",
            "forecast_pred",
            "timing_pred",
            "tail_risk_pred",
            "vol_forecast_pred",
        ]
        loss = (
            sum(entry[name].square().mean() for name in heads)
            + episode["exit_action_q_bps"].square().mean()
        )
        loss.backward()
        gradients = {
            name: value.grad.detach().clone()
            for name, value in model.named_parameters()
            if value.grad is not None
        }
        assert gradients and all(torch.isfinite(t).all() for t in gradients.values())
        trainer._optimizer_step_with_finite_gradients(
            model=model, optimizer=optimizer, weight_ema=ema
        )
        with ema.evaluating(model), torch.no_grad():
            validation = trainer._model_forward_fp32(
                model, seq, snap, ctx_cat=cat, ctx_cont=cont, **mtf
            )
        return {
            "initial_state": initial,
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "ema": copy.deepcopy(ema.checkpoint_state()),
            "entry_outputs": {
                name: tensor.detach().clone()
                for name, tensor in entry.items()
                if torch.is_tensor(tensor)
            },
            "exit_outputs": {
                name: tensor.detach().clone()
                for name, tensor in episode.items()
                if torch.is_tensor(tensor)
            },
            "ema_entry_outputs": {
                name: tensor.detach().clone()
                for name, tensor in validation.items()
                if torch.is_tensor(tensor)
            },
            "gradients": gradients,
            "loss": loss.item(),
            "torch_rng": torch.get_rng_state(),
        }

    reference = run(True)
    actual = run(False)
    _equal_tree(reference, actual)


def test_real_set_deterministic_applies_option_and_next_default_run_restores_it(
    monkeypatch,
):
    # Only CUDA hardware calls are mocked. Exercise the real process-global
    # setter and policy selection without touching a GPU in the CPU audit.
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    fractions = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: fractions.append(fraction),
    )
    trainer._set_deterministic(9, torch.device("cuda"), POLICY)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.utils.deterministic.fill_uninitialized_memory
    assert not torch.backends.cuda.matmul.allow_tf32
    assert fractions == [0.45]
    trainer._set_deterministic(9, torch.device("cpu"), policy.DETERMINISTIC_FP32)
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("selected", sorted(policy.TRAINING_PRECISION_POLICIES))
def test_only_explicit_no_fill_policies_disable_allocation_poisoning(selected):
    # Both bounded follow-up experiments use the measured no-fill baseline.
    # Every other declared policy retains allocation poisoning.
    explicitly_no_fill = {
        POLICY,
        policy.EXPERIMENTAL_FP32_3090_NO_FILL_KERNEL_PROFILE,
        policy.EXPERIMENTAL_BF16_3090_FP32_Q_HEADS_NO_FILL,
        policy.EXPERIMENTAL_FP32_3090_BATCHED_MTF_TEACHER_NO_FILL,
    }
    assert policy.deterministic_fill_uninitialized_memory(selected) is (
        selected not in explicitly_no_fill
    )


def test_unknown_policy_cannot_disable_filling():
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.deterministic_fill_uninitialized_memory("unknown")


def test_no_fill_policy_keeps_original_fp32_checks_and_resource_geometry():
    assert (
        policy.require_training_precision_policy(
            POLICY,
            device_type="cuda",
            execution_tier="canonical",
            profile="smoke",
            batch_size=8,
        )
        == POLICY
    )
    m = policy.training_precision_metadata(POLICY, device_type="cuda")
    assert m["deterministic_fill_uninitialized_memory"] is False
    assert m["deterministic_algorithms"] is True
    assert all(
        m[key] is False for key in ["autocast", "tf32", "compile", "gradient_scaler"]
    )
    assert all(
        m[key] == "float32"
        for key in [
            "parameter_dtype",
            "loss_reduction_dtype",
            "optimizer_state_dtype",
            "ema_dtype",
        ]
    )
    assert m["cuda_memory_fraction"] == 0.45 and m[
        "required_cuda_compute_capability"
    ] == [8, 6]
    assert policy.numerical_thread_count(POLICY) == 8
    assert policy.unified_exit_chunk_rows(POLICY, batch_size=8) == 8
    assert policy.candidate_checkpoint_interval(POLICY) == 64
    assert policy.candidate_validation_checkpoint_interval(POLICY) == 64
    assert policy.model_finite_check_mode(POLICY) is None


@pytest.mark.parametrize(
    "change",
    [
        {"device_type": "cpu"},
        {"profile": "unknown"},
        {"execution_tier": "attended_only"},
        {"batch_size": 10},
        {"batch_size": True},
    ],
)
def test_no_fill_policy_rejects_scope_expansion(change):
    kwargs = dict(
        device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=8
    )
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_training_precision_policy(POLICY, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": 2},
        {"grad_accum_steps": 2},
        {"subsample_rows": 511},
        {"subsample_rows": 513},
        {"subsample_rows": True},
    ],
)
def test_no_fill_policy_rejects_other_training_geometry(change):
    kwargs = dict(epochs=1, grad_accum_steps=1, subsample_rows=512)
    policy.require_local_precision_benchmark_geometry(POLICY, **kwargs)
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_local_precision_benchmark_geometry(POLICY, **kwargs)


@pytest.mark.parametrize(
    "capability,accepted", [((8, 6), True), ((8, 0), False), ((9, 0), False)]
)
def test_no_fill_capability_is_local_3090_only(monkeypatch, capability, accepted):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    if accepted:
        trainer._require_local_fp32_no_fill_capability()
    else:
        with pytest.raises(
            RuntimeError, match="LOCAL_FP32_NO_FILL_CAPABILITY_REQUIRED"
        ):
            trainer._require_local_fp32_no_fill_capability()


def test_declared_training_seed_initializes_every_checkpointed_cpu_rng(monkeypatch):
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    random.seed(71)
    trainer._set_deterministic(1337, torch.device("cpu"))
    first = trainer._attended_session_rng_state(device=torch.device("cpu"))
    assert first["python"] == random.Random(1337).getstate()
    random.random()
    np.random.random(5)
    torch.rand(7)
    trainer._set_deterministic(1337, torch.device("cpu"))
    second = trainer._attended_session_rng_state(device=torch.device("cpu"))
    _equal_tree(first, second)
    trainer._set_deterministic(97531, torch.device("cpu"))
    trainer._restore_attended_session_rng_state(first, device=torch.device("cpu"))
    _equal_tree(first, trainer._attended_session_rng_state(device=torch.device("cpu")))


def test_fresh_processes_use_declared_python_seed_not_startup_entropy():
    import json
    from pathlib import Path
    import subprocess
    import sys

    code = """
import json, random, torch
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
random.seed()
torch.set_num_threads = lambda count: None
trainer._set_deterministic(1337, torch.device('cpu'))
print(json.dumps([random.random() for _ in range(16)]))
"""
    expected = random.Random(1337)
    draws = [expected.random() for _ in range(16)]
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, check=True,
        )
        assert json.loads(result.stdout) == draws
