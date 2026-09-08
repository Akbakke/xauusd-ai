"""Source-owned numerical policies for model-native Entry/Exit training."""

from __future__ import annotations

from typing import Any


DETERMINISTIC_FP32 = "deterministic_fp32"
DETERMINISTIC_BF16_HOPPER = "deterministic_bf16_hopper"
EXPERIMENTAL_BF16_3090 = "experimental_bf16_3090"
TRAINING_PRECISION_POLICIES = frozenset(
    {DETERMINISTIC_FP32, DETERMINISTIC_BF16_HOPPER, EXPERIMENTAL_BF16_3090}
)


class TrainingPrecisionPolicyError(RuntimeError):
    """The declared training precision policy is invalid."""


def require_training_precision_policy(
    value: Any,
    *,
    device_type: str,
    execution_tier: str,
    profile: str,
    batch_size: int,
) -> str:
    policy = str(value or "")
    if policy not in TRAINING_PRECISION_POLICIES:
        raise TrainingPrecisionPolicyError(
            f"precision_policy={policy!r} is not declared"
        )
    if policy == DETERMINISTIC_FP32:
        return policy
    if policy == EXPERIMENTAL_BF16_3090:
        if (device_type, execution_tier, profile) != ("cuda", "canonical", "smoke") or type(batch_size) is not int or batch_size != 8:
            raise TrainingPrecisionPolicyError(
                "experimental_bf16_3090 requires canonical CUDA smoke at batch 8"
            )
        return policy
    if (
        device_type != "cuda"
        or execution_tier != "canonical"
        or profile not in {"smoke", "candidate"}
        or isinstance(batch_size, bool)
        or int(batch_size) not in {8, 32, 64}
        or (profile == "candidate" and int(batch_size) not in {32, 64})
    ):
        raise TrainingPrecisionPolicyError(
            "deterministic_bf16_hopper requires canonical CUDA, a smoke batch "
            "of 8/32/64, or a candidate batch of 32/64"
        )
    return policy


def training_precision_metadata(
    policy: str,
    *,
    device_type: str,
) -> dict[str, Any]:
    if policy == DETERMINISTIC_FP32 and device_type in {"cpu", "cuda"}:
        return {
            "precision": DETERMINISTIC_FP32,
            "compile": False,
            "tf32": False,
            "autocast": False,
            "cuda_memory_fraction": 0.45 if device_type == "cuda" else None,
        }
    if policy == EXPERIMENTAL_BF16_3090 and device_type == "cuda":
        return {
            "precision": EXPERIMENTAL_BF16_3090,
            "parameter_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "loss_reduction_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "ema_dtype": "float32",
            "gradient_scaler": False,
            "compile": False,
            "tf32": False,
            "autocast": True,
            "deterministic_algorithms": True,
            "required_cuda_compute_capability": [8, 6],
            "native_bf16_required": True,
            "cuda_memory_fraction": 0.45,
            "experimental_only": True,
        }
    if policy == DETERMINISTIC_BF16_HOPPER and device_type == "cuda":
        return {
            "precision": DETERMINISTIC_BF16_HOPPER,
            "parameter_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "loss_reduction_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "ema_dtype": "float32",
            "gradient_scaler": False,
            "compile": False,
            "tf32": False,
            "autocast": True,
            "deterministic_algorithms": True,
            "minimum_cuda_compute_capability": [9, 0],
            "cuda_memory_fraction": 0.85,
        }
    raise TrainingPrecisionPolicyError(
        f"precision policy {policy!r} is invalid for device {device_type!r}"
    )


def numerical_thread_count(policy: str) -> int:
    if policy in {DETERMINISTIC_FP32, EXPERIMENTAL_BF16_3090}:
        return 8
    if policy == DETERMINISTIC_BF16_HOPPER:
        return 16
    raise TrainingPrecisionPolicyError(f"precision_policy={policy!r} is not declared")


def cuda_memory_fraction(policy: str) -> float:
    metadata = training_precision_metadata(policy, device_type="cuda")
    return float(metadata["cuda_memory_fraction"])


def unified_exit_chunk_rows(policy: str, *, batch_size: int) -> int:
    if policy in {DETERMINISTIC_FP32, EXPERIMENTAL_BF16_3090}:
        return min(8, int(batch_size))
    if policy == DETERMINISTIC_BF16_HOPPER:
        return min(64, int(batch_size))
    raise TrainingPrecisionPolicyError(f"precision_policy={policy!r} is not declared")


def candidate_checkpoint_interval(policy: str) -> int:
    if policy in {DETERMINISTIC_FP32, EXPERIMENTAL_BF16_3090}:
        return 64
    if policy == DETERMINISTIC_BF16_HOPPER:
        return 512
    raise TrainingPrecisionPolicyError(f"precision_policy={policy!r} is not declared")


def candidate_validation_checkpoint_interval(policy: str) -> int:
    if policy in {DETERMINISTIC_FP32, EXPERIMENTAL_BF16_3090}:
        return 64
    if policy == DETERMINISTIC_BF16_HOPPER:
        return 128
    raise TrainingPrecisionPolicyError(f"precision_policy={policy!r} is not declared")


def require_local_precision_benchmark_geometry(
    policy: str, *, epochs: int, grad_accum_steps: int, subsample_rows: int,
) -> None:
    """Keep the first precision comparison identical to the bounded baseline."""
    if policy != EXPERIMENTAL_BF16_3090:
        return
    if (
        type(epochs) is not int or epochs != 1
        or type(grad_accum_steps) is not int or grad_accum_steps != 1
        or type(subsample_rows) is not int or not 1 <= subsample_rows <= 512
    ):
        raise TrainingPrecisionPolicyError(
            "local precision benchmark requires one epoch, accumulation 1 and 1..512 rows"
        )
