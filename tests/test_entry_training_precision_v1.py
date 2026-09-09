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
