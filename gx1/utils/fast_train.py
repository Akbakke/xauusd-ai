"""Centralized fast-training utilities.

Provides composable speedup primitives used by V10/V3 trainers and other
heavy compute scripts. All toggles are env-gated so behavior is reproducible
and opt-in (defaults preserve V1 behavior).

Env vars:
    GX1_FAST_TRAIN=1           — enable all bf16/TF32/compile defaults
    GX1_FAST_TRAIN_BF16=0      — opt-out bf16 autocast individually
    GX1_FAST_TRAIN_TF32=0      — opt-out TF32 matmul individually
    GX1_FAST_TRAIN_COMPILE=0   — opt-out torch.compile individually
    GX1_FAST_TRAIN_COMPILE_MODE — compile mode (default 'reduce-overhead')

Usage:
    from gx1.utils.fast_train import (
        apply_global_speedups, autocast_context, maybe_compile, build_cosine_warmup_scheduler,
    )

    apply_global_speedups()                     # call once at trainer start
    model = maybe_compile(model)                # after model construction
    with autocast_context(device):              # wrap forward
        out = model(...)
        loss = ...
    loss.backward()
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _fast_train_master() -> bool:
    return _env_on("GX1_FAST_TRAIN", "0")


def bf16_enabled() -> bool:
    """bf16 autocast on iff master flag is set AND not individually opted-out."""
    if not _fast_train_master():
        return False
    return os.environ.get("GX1_FAST_TRAIN_BF16", "1").strip().lower() in ("1", "true", "yes", "on")


def tf32_enabled() -> bool:
    if not _fast_train_master():
        return False
    return os.environ.get("GX1_FAST_TRAIN_TF32", "1").strip().lower() in ("1", "true", "yes", "on")


def compile_enabled() -> bool:
    if not _fast_train_master():
        return False
    return os.environ.get("GX1_FAST_TRAIN_COMPILE", "1").strip().lower() in ("1", "true", "yes", "on")


def apply_global_speedups() -> dict:
    """Apply process-wide CUDA/torch settings. Idempotent; safe to call multiple times.

    Returns a dict describing what was enabled (for logging).
    """
    report = {
        "fast_train_master": _fast_train_master(),
        "bf16_autocast": bf16_enabled(),
        "tf32_matmul": tf32_enabled(),
        "torch_compile": compile_enabled(),
    }
    if not _fast_train_master():
        return report

    # TF32 matmul on Ampere+ — free 1.5× for fp32 ops with negligible quality loss
    if tf32_enabled():
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    return report


def autocast_context(device: torch.device):
    """Returns torch.autocast(bfloat16) context if enabled and cuda, else nullcontext().

    bf16 is preferred over fp16 because it has fp32's dynamic range — no need for
    GradScaler, no NaN risk on overflows.
    """
    if bf16_enabled() and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_compile(model: torch.nn.Module, mode: Optional[str] = None) -> torch.nn.Module:
    """Wrap model with torch.compile if enabled. Returns model unchanged otherwise.

    `mode` defaults to env GX1_FAST_TRAIN_COMPILE_MODE or 'reduce-overhead'.
    Returns the original model on compile failure (compile is best-effort).
    """
    if not compile_enabled():
        return model
    mode = mode or os.environ.get("GX1_FAST_TRAIN_COMPILE_MODE", "reduce-overhead")
    try:
        return torch.compile(model, mode=mode)
    except Exception as e:
        print(f"[FAST_TRAIN] torch.compile failed ({e!r}) — using uncompiled model")
        return model


def build_cosine_warmup_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Linear warmup → cosine decay scheduler.

    During warmup_steps: lr ramps linearly 0 → base_lr.
    After warmup: cosine decay base_lr → base_lr * min_lr_ratio over remaining steps.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    warmup_steps = max(0, int(warmup_steps))

    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return LambdaLR(optimizer, lr_lambda)


def grad_accum_steps_from_env(default: int = 1) -> int:
    """Read GX1_FAST_TRAIN_GRAD_ACCUM env var (default 1 = no accumulation)."""
    raw = os.environ.get("GX1_FAST_TRAIN_GRAD_ACCUM", str(default))
    try:
        n = int(raw)
        return max(1, n)
    except Exception:
        return default


def loader_kwargs(num_workers: int, *, use_cuda: bool) -> dict:
    """Return DataLoader kwargs tuned for high throughput.

    Bumps prefetch_factor 2→4 when fast-train master is on.
    """
    pin_memory = bool(use_cuda)
    persistent_workers = bool(num_workers > 0)
    base_prefetch = 4 if _fast_train_master() else 2
    prefetch_factor = base_prefetch if num_workers > 0 else None
    return dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


__all__ = [
    "apply_global_speedups",
    "autocast_context",
    "maybe_compile",
    "build_cosine_warmup_scheduler",
    "grad_accum_steps_from_env",
    "loader_kwargs",
    "bf16_enabled",
    "tf32_enabled",
    "compile_enabled",
]
