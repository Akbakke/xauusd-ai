from __future__ import annotations

import pytest

from gx1.contracts.entry_training_precision_v1 import (
    DETERMINISTIC_BF16_HOPPER,
    DETERMINISTIC_FP32,
    TrainingPrecisionPolicyError,
    candidate_checkpoint_interval,
    candidate_validation_checkpoint_interval,
    numerical_thread_count,
    require_training_precision_policy,
    training_precision_metadata,
    unified_exit_chunk_rows,
)


def test_existing_fp32_policy_is_unchanged() -> None:
    assert training_precision_metadata(
        DETERMINISTIC_FP32,
        device_type="cuda",
    ) == {
        "precision": "deterministic_fp32",
        "compile": False,
        "tf32": False,
        "autocast": False,
        "cuda_memory_fraction": 0.45,
    }
    assert numerical_thread_count(DETERMINISTIC_FP32) == 8
    assert unified_exit_chunk_rows(DETERMINISTIC_FP32, batch_size=64) == 8
    assert candidate_checkpoint_interval(DETERMINISTIC_FP32) == 64
    assert candidate_validation_checkpoint_interval(DETERMINISTIC_FP32) == 64


def test_hopper_policy_keeps_state_and_reductions_in_fp32() -> None:
    metadata = training_precision_metadata(
        DETERMINISTIC_BF16_HOPPER,
        device_type="cuda",
    )
    assert metadata["autocast"] is True
    assert metadata["autocast_dtype"] == "bfloat16"
    assert metadata["parameter_dtype"] == "float32"
    assert metadata["loss_reduction_dtype"] == "float32"
    assert metadata["optimizer_state_dtype"] == "float32"
    assert metadata["ema_dtype"] == "float32"
    assert metadata["gradient_scaler"] is False
    assert metadata["minimum_cuda_compute_capability"] == [9, 0]
    assert metadata["cuda_memory_fraction"] == 0.85
    assert numerical_thread_count(DETERMINISTIC_BF16_HOPPER) == 16
    assert unified_exit_chunk_rows(DETERMINISTIC_BF16_HOPPER, batch_size=32) == 32
    assert unified_exit_chunk_rows(DETERMINISTIC_BF16_HOPPER, batch_size=64) == 64
    assert candidate_checkpoint_interval(DETERMINISTIC_BF16_HOPPER) == 512
    assert candidate_validation_checkpoint_interval(DETERMINISTIC_BF16_HOPPER) == 128


@pytest.mark.parametrize("batch_size", [32, 64])
def test_hopper_candidate_accepts_only_large_explicit_batches(batch_size: int) -> None:
    assert require_training_precision_policy(
        DETERMINISTIC_BF16_HOPPER,
        device_type="cuda",
        execution_tier="canonical",
        profile="candidate",
        batch_size=batch_size,
    ) == DETERMINISTIC_BF16_HOPPER


@pytest.mark.parametrize(
    ("device_type", "execution_tier", "profile", "batch_size"),
    [
        ("cpu", "canonical", "candidate", 32),
        ("cuda", "attended_only", "smoke", 32),
        ("cuda", "canonical", "candidate", 8),
        ("cuda", "canonical", "candidate", 16),
    ],
)
def test_hopper_policy_fails_closed_outside_declared_surface(
    device_type: str,
    execution_tier: str,
    profile: str,
    batch_size: int,
) -> None:
    with pytest.raises(TrainingPrecisionPolicyError):
        require_training_precision_policy(
            DETERMINISTIC_BF16_HOPPER,
            device_type=device_type,
            execution_tier=execution_tier,
            profile=profile,
            batch_size=batch_size,
        )


def test_local_bf16_keeps_the_local_resource_geometry() -> None:
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_BF16_3090
    policy = EXPERIMENTAL_BF16_3090
    assert require_training_precision_policy(policy, device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=8) == policy
    metadata = training_precision_metadata(policy, device_type="cuda")
    assert metadata["cuda_memory_fraction"] == 0.45
    assert metadata["required_cuda_compute_capability"] == [8, 6]
    assert metadata["autocast_dtype"] == "bfloat16"
    assert metadata["native_bf16_required"] is True
    assert metadata["experimental_only"] is True
    assert metadata["gradient_scaler"] is False
    for key in ("parameter_dtype", "loss_reduction_dtype", "optimizer_state_dtype", "ema_dtype"):
        assert metadata[key] == "float32"
    assert numerical_thread_count(policy) == 8
    assert unified_exit_chunk_rows(policy, batch_size=8) == 8
    assert candidate_checkpoint_interval(policy) == 64
    assert candidate_validation_checkpoint_interval(policy) == 64


@pytest.mark.parametrize("overrides", [
    {"device_type": "cpu"}, {"execution_tier": "attended_only"},
    {"profile": "candidate"}, {"batch_size": 16}, {"batch_size": True},
])
def test_local_bf16_cannot_expand_execution_scope(overrides) -> None:
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_BF16_3090
    kwargs = dict(device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=8)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError, match="canonical CUDA smoke at batch 8"):
        require_training_precision_policy(EXPERIMENTAL_BF16_3090, **kwargs)


@pytest.mark.parametrize("overrides", [
    {"epochs": 2}, {"epochs": True}, {"grad_accum_steps": 2},
    {"subsample_rows": 0}, {"subsample_rows": 513},
])
def test_local_bf16_benchmark_is_bounded(overrides) -> None:
    from gx1.contracts.entry_training_precision_v1 import (
        EXPERIMENTAL_BF16_3090, require_local_precision_benchmark_geometry,
    )
    kwargs = dict(epochs=1, grad_accum_steps=1, subsample_rows=512)
    require_local_precision_benchmark_geometry(EXPERIMENTAL_BF16_3090, **kwargs)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError, match="local precision benchmark"):
        require_local_precision_benchmark_geometry(EXPERIMENTAL_BF16_3090, **kwargs)


@pytest.mark.parametrize("batch", [8, 10, 12, 16])
def test_local_full_exit_batch_is_explicit_fp32_with_unchanged_resource_limits(batch):
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH as policy
    assert require_training_precision_policy(policy, device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=batch) == policy
    metadata = training_precision_metadata(policy, device_type="cuda")
    assert metadata["autocast"] is False and metadata["tf32"] is False and metadata["compile"] is False
    assert metadata["cuda_memory_fraction"] == 0.45
    assert numerical_thread_count(policy) == 8
    assert unified_exit_chunk_rows(policy, batch_size=batch) == batch
    assert unified_exit_chunk_rows(DETERMINISTIC_FP32, batch_size=batch) == 8


@pytest.mark.parametrize("overrides", [
    {"device_type": "cpu"}, {"execution_tier": "attended_only"},
    {"profile": "candidate"}, {"batch_size": 9}, {"batch_size": 32},
    {"batch_size": True},
])
def test_local_full_exit_batch_rejects_undeclared_execution(overrides):
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH as policy
    kwargs = dict(device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=10)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError):
        require_training_precision_policy(policy, **kwargs)


@pytest.mark.parametrize("overrides", [{"epochs": 2}, {"grad_accum_steps": 2}, {"subsample_rows": 0}, {"subsample_rows": 513}])
def test_local_full_exit_batch_cannot_be_used_for_unbounded_training(overrides):
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH as policy, require_local_precision_benchmark_geometry
    kwargs = dict(epochs=1, grad_accum_steps=1, subsample_rows=512)
    require_local_precision_benchmark_geometry(policy, **kwargs)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError):
        require_local_precision_benchmark_geometry(policy, **kwargs)


_NEXT_LOCAL_POLICIES = (
    "experimental_bf16_3090_fp32_q_heads_no_fill",
    "experimental_fp32_3090_batched_mtf_teacher_no_fill",
)


@pytest.mark.parametrize("policy", _NEXT_LOCAL_POLICIES)
def test_next_local_experiments_are_bounded(policy):
    from gx1.contracts.entry_training_precision_v1 import (
        deterministic_fill_uninitialized_memory,
        require_local_precision_benchmark_geometry,
    )
    assert require_training_precision_policy(
        policy, device_type="cuda", execution_tier="canonical",
        profile="smoke", batch_size=8,
    ) == policy
    require_local_precision_benchmark_geometry(
        policy, epochs=1, grad_accum_steps=1, subsample_rows=512, profile="smoke",
    )
    metadata = training_precision_metadata(policy, device_type="cuda")
    assert metadata["precision"] == policy
    assert metadata["cuda_memory_fraction"] == 0.45
    assert metadata["required_cuda_compute_capability"] == [8, 6]
    assert metadata["experimental_only"] is True
    assert metadata["deterministic_algorithms"] is True
    assert metadata["deterministic_fill_uninitialized_memory"] is False
    assert deterministic_fill_uninitialized_memory(policy) is False
    assert metadata["compile"] is metadata["tf32"] is metadata["gradient_scaler"] is False
    for key in ("parameter_dtype", "loss_reduction_dtype", "optimizer_state_dtype", "ema_dtype"):
        assert metadata[key] == "float32"
    assert numerical_thread_count(policy) == 8
    assert unified_exit_chunk_rows(policy, batch_size=8) == 8
    assert candidate_checkpoint_interval(policy) == candidate_validation_checkpoint_interval(policy) == 64
    if policy == _NEXT_LOCAL_POLICIES[0]:
        assert metadata["autocast"] is True
        assert metadata["autocast_dtype"] == "bfloat16"
        assert metadata["raw_q_head_arithmetic_dtype"] == "float32"
        assert metadata["raw_q_head_autocast"] is False
        assert metadata["native_bf16_required"] is True
        assert "mtf_equal_length_batching" not in metadata
    else:
        assert metadata["autocast"] is False
        assert metadata["mtf_equal_length_batching"] == "eval_no_grad_only"
        assert metadata["mtf_batch_grouping"] == "same_encoder_same_sequence_length"
        assert "raw_q_head_arithmetic_dtype" not in metadata


@pytest.mark.parametrize("policy", _NEXT_LOCAL_POLICIES)
@pytest.mark.parametrize("overrides", [
    {"device_type": "cpu"}, {"execution_tier": "attended_only"},
    {"profile": "candidate"}, {"batch_size": 10}, {"batch_size": True},
])
def test_next_local_experiments_reject_other_launch_surfaces(policy, overrides):
    kwargs = dict(device_type="cuda", execution_tier="canonical", profile="smoke", batch_size=8)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError):
        require_training_precision_policy(policy, **kwargs)


@pytest.mark.parametrize("policy", _NEXT_LOCAL_POLICIES)
@pytest.mark.parametrize("overrides", [
    {"profile": "candidate"}, {"epochs": 2}, {"epochs": True},
    {"grad_accum_steps": 2}, {"grad_accum_steps": True},
    {"subsample_rows": 0}, {"subsample_rows": 128}, {"subsample_rows": 513},
    {"subsample_rows": 512.0},
])
def test_next_local_experiments_reject_changed_workload(policy, overrides):
    from gx1.contracts.entry_training_precision_v1 import require_local_precision_benchmark_geometry
    kwargs = dict(profile="smoke", epochs=1, grad_accum_steps=1, subsample_rows=512)
    kwargs.update(overrides)
    with pytest.raises(TrainingPrecisionPolicyError):
        require_local_precision_benchmark_geometry(policy, **kwargs)


@pytest.fixture
def _isolated_training_forward():
    """Run the real numerical owners without importing trainer or any datasets.

    Existing FP32 autocast/finite/profile contexts are neutral; these tests own
    only the phase-bound MHA policy, not an alternative precision implementation.
    """
    import ast
    import contextlib
    import random
    from pathlib import Path

    import numpy as np
    import torch

    from gx1.contracts.entry_model_native_training_objective_v1 import (
        ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, JOINT_FITTED_Q_TRAINING_PHASE,
    )

    path = Path(__file__).resolve().parents[1] / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    tree = ast.parse(path.read_text())
    names = {"_model_forward_fp32", "_training_mha_fastpath_context"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *definitions], type_ignores=[])
    namespace = {
        "torch": torch, "contextlib": contextlib,
        "DETERMINISTIC_FP32": DETERMINISTIC_FP32,
        "ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "_TRAINING_PRECISION_POLICY": DETERMINISTIC_FP32,
        "_training_autocast_context": contextlib.nullcontext,
        "_training_model_finite_check_context": contextlib.nullcontext,
        "kernel_profile_range": lambda label: contextlib.nullcontext(),
        "legacy_phase": JOINT_FITTED_Q_TRAINING_PHASE,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    set_fastpath = torch.backends.mha.set_fastpath_enabled
    previous = torch.backends.mha.get_fastpath_enabled()
    rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    try:
        yield namespace
    finally:
        set_fastpath(previous)
        torch.set_rng_state(rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


@pytest.mark.parametrize("kind", ["mha", "encoder"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("previous_fastpath", [False, True])
def test_dynamic_fp32_attention_uses_same_unfused_routes_across_grad_modes(
    _isolated_training_forward, kind, training, previous_fastpath,
):
    import random
    import numpy as np

    owner = _isolated_training_forward
    torch = owner["torch"]
    torch.manual_seed(619)
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    sdpa = lambda: (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
    sdpa_before = sdpa()

    class TinyAttention(torch.nn.Module):
        entry_training_phase = owner["ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE"]

        def __init__(self):
            super().__init__()
            self.layer = (torch.nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True) if kind == "mha"
                          else torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True))
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            assert torch.backends.mha.get_fastpath_enabled() is False
            assert sdpa() == sdpa_before
            output = self.layer(value, value, value, need_weights=False)[0] if kind == "mha" else self.layer(value)
            return {"output": output}

    model = TinyAttention().train(training)
    value = torch.randn(2, 5, 8, requires_grad=True)
    original_value = value.detach().clone()
    weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    outputs, operators = [], []
    for enabled in (False, True):
        with torch.set_grad_enabled(enabled):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                output = owner["_model_forward_fp32"](model, value)["output"]
            assert output.requires_grad is enabled and output.dtype == torch.float32
            outputs.append(output.detach().clone())
            operators.append({event.key for event in profile.key_averages()})
        assert torch.backends.mha.get_fastpath_enabled() is previous_fastpath
    assert model.calls == 2
    for observed in operators:
        assert "aten::_transformer_encoder_layer_fwd" not in observed
        assert "aten::_native_multi_head_attention" not in observed
        assert "aten::scaled_dot_product_attention" in observed
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=1e-4)
    assert torch.equal(value.detach(), original_value) and value.requires_grad and value.grad is None
    assert all(torch.equal(tensor, weights[name]) for name, tensor in model.state_dict().items())
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng[0] and np.array_equal(current_numpy_rng[1], numpy_rng[1])
    assert current_numpy_rng[2:] == numpy_rng[2:] and sdpa() == sdpa_before


@pytest.mark.parametrize("previous_fastpath", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_dynamic_fp32_attention_restores_nested_and_exception_lifetimes(
    _isolated_training_forward, previous_fastpath, fail,
):
    owner = _isolated_training_forward
    torch = owner["torch"]
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)

    class Probe(torch.nn.Module):
        entry_training_phase = owner["ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE"]

        def forward(self, value):
            assert torch.backends.mha.get_fastpath_enabled() is False
            with owner["_training_mha_fastpath_context"](self):
                assert torch.backends.mha.get_fastpath_enabled() is False
                if fail:
                    raise ValueError("model sentinel")
            assert torch.backends.mha.get_fastpath_enabled() is False
            return value * 2

    model = Probe().train()
    value = torch.ones(1, requires_grad=True)
    if fail:
        with pytest.raises(ValueError, match="model sentinel"):
            owner["_model_forward_fp32"](model, value)
    else:
        output = owner["_model_forward_fp32"](model, value)
        assert torch.equal(output, value * 2) and output.requires_grad
    assert torch.backends.mha.get_fastpath_enabled() is previous_fastpath
    assert model.training and value.grad is None


@pytest.mark.parametrize("previous_fastpath", [False, True])
@pytest.mark.parametrize("phase,precision", [
    ("missing", DETERMINISTIC_FP32), ("legacy", DETERMINISTIC_FP32),
    ("dynamic", DETERMINISTIC_BF16_HOPPER),
])
def test_attention_fastpath_leaves_other_phases_and_precisions_unchanged(
    _isolated_training_forward, phase, precision, previous_fastpath,
):
    owner = _isolated_training_forward
    torch = owner["torch"]
    owner["_TRAINING_PRECISION_POLICY"] = precision
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)

    class Probe(torch.nn.Module):
        def forward(self, value):
            assert torch.backends.mha.get_fastpath_enabled() is previous_fastpath
            return value * 2

    model = Probe()
    if phase != "missing":
        model.entry_training_phase = owner["legacy_phase"] if phase == "legacy" else owner["ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE"]
    value = torch.ones(1, requires_grad=True)
    output = owner["_model_forward_fp32"](model, value)
    assert torch.equal(output, value * 2) and output.requires_grad
    assert torch.backends.mha.get_fastpath_enabled() is previous_fastpath


def test_dynamic_fp32_attention_fails_closed_if_public_backend_refuses_disable(
    _isolated_training_forward, monkeypatch,
):
    owner = _isolated_training_forward
    torch = owner["torch"]
    torch.backends.mha.set_fastpath_enabled(True)
    actual_set = torch.backends.mha.set_fastpath_enabled
    attempts = []

    def refuse_disable(value):
        attempts.append(value)
        if value:
            actual_set(value)

    monkeypatch.setattr(torch.backends.mha, "set_fastpath_enabled", refuse_disable)

    class NeverRun(torch.nn.Module):
        entry_training_phase = owner["ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE"]

        def forward(self, value):
            raise AssertionError("model must not execute before the policy is enforced")

    with pytest.raises(RuntimeError, match="ENTRY_DYNAMIC_M5_MHA_FASTPATH_DISABLE_REQUIRED"):
        owner["_model_forward_fp32"](NeverRun(), torch.ones(1))
    assert attempts == [False, True]
    assert torch.backends.mha.get_fastpath_enabled() is True
