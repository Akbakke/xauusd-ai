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
