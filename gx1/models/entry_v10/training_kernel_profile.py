"""One warmed optimizer-step trace for an explicit local diagnostic policy."""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch

from gx1.contracts.entry_training_precision_v1 import (
    EXPERIMENTAL_FP32_3090_KERNEL_PROFILE,
    EXPERIMENTAL_FP32_3090_NO_FILL_KERNEL_PROFILE,
)

_ACTIVE_PROFILE: ContextVar["_KernelProfileSession | None"] = ContextVar(
    "gx1_training_kernel_profile", default=None,
)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _KernelProfileSession:
    def __init__(self, output_dir: Path, *, include_cuda: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.include_cuda = include_cuda
        self.steps = 0
        self.trace_count = 0
        self.trace_step: int | None = None
        self._token = None
        self._profiler = None

    def _save_trace(self, profiler) -> None:
        if self.trace_count != 0 or self.steps != 9:
            raise RuntimeError("[ENTRY_KERNEL_PROFILE_TRACE_GEOMETRY_INVALID]")
        trace = self.output_dir / "optimizer_step_9.trace.json"
        if trace.exists():
            raise RuntimeError("[ENTRY_KERNEL_PROFILE_TRACE_ALREADY_EXISTS]")
        profiler.export_chrome_trace(str(trace))
        operators = []
        for event in profiler.key_averages():
            row = {"operator": str(event.key), "count": int(event.count),
                   "device_type": str(event.device_type)}
            for name in ("cpu_time_total", "self_cpu_time_total", "device_time_total", "self_device_time_total"):
                value = float(getattr(event, name))
                if not math.isfinite(value):
                    raise RuntimeError("[ENTRY_KERNEL_PROFILE_TIME_NONFINITE]")
                row[name + "_us"] = value
            operators.append(row)
        cuda_events = sum(str(event.device_type) == "DeviceType.CUDA" for event in profiler.events())
        self.trace_count = 1
        self.trace_step = self.steps
        _write_json(self.output_dir / "operators.json", {
            "schema_version": "gx1_local_kernel_profile_v1",
            "purpose": "operator_diagnosis_not_throughput_qualification",
            "profiled_optimizer_steps": [self.steps],
            "schedule": {"wait": 7, "warmup": 1, "active": 1, "repeat": 1},
            "record_shapes": False, "with_stack": False, "profile_memory": False,
            "cuda_requested": self.include_cuda, "cuda_event_count": cuda_events,
            "cuda_timing_available": cuda_events > 0,
            "status": "CPU_AND_CUDA_CAPTURED" if cuda_events > 0 else "CPU_ONLY_CAPTURED",
            "torch_version": str(torch.__version__),
            "trace": {"path": str(trace), "sha256": _sha256(trace), "size_bytes": trace.stat().st_size},
            "operators": operators,
            "limitations": ["Profiler perturbs runtime", "One warmed step is not an ETA or speedup benchmark", "Operator totals may include nested or overlapping work"],
        })

    def __enter__(self):
        if _ACTIVE_PROFILE.get() is not None:
            raise RuntimeError("[ENTRY_KERNEL_PROFILE_NESTED_SESSION_INVALID]")
        if (not self.output_dir.is_absolute() or self.output_dir.resolve() != self.output_dir
                or not self.output_dir.parent.is_dir()):
            raise RuntimeError("[ENTRY_KERNEL_PROFILE_OUTPUT_PATH_INVALID]")
        self.output_dir.mkdir(mode=0o700)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.include_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=7, warmup=1, active=1, repeat=1),
            on_trace_ready=self._save_trace,
            record_shapes=False, with_stack=False, profile_memory=False,
        )
        self._profiler.__enter__()
        self._token = _ACTIVE_PROFILE.set(self)
        return self

    def step(self) -> None:
        self.steps += 1
        self._profiler.step()

    def __exit__(self, exc_type, exc, traceback):
        closed = False
        try:
            self._profiler.__exit__(exc_type, exc, traceback)
            closed = True
        finally:
            _ACTIVE_PROFILE.reset(self._token)
            successful = closed and exc_type is None and self.trace_count == 1 and self.trace_step == 9
            _write_json(self.output_dir / "session.json", {
                "schema_version": "gx1_local_kernel_profile_session_v1",
                "terminal_success": successful,
                "completed_optimizer_steps": self.steps,
                "trace_count": self.trace_count,
                "trace_optimizer_step": self.trace_step,
                "exception_type": None if exc_type is None else exc_type.__name__,
            })
        if exc_type is None and not successful:
            raise RuntimeError("[ENTRY_KERNEL_PROFILE_REQUIRED_TRACE_MISSING]")
        return False


def profile_training_epoch(*, policy: Callable[[], str]):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            output = kwargs.get("kernel_profile_output_dir")
            enabled = policy() in {
                EXPERIMENTAL_FP32_3090_KERNEL_PROFILE,
                EXPERIMENTAL_FP32_3090_NO_FILL_KERNEL_PROFILE,
            }
            if enabled != (output is not None):
                raise RuntimeError("[ENTRY_KERNEL_PROFILE_POLICY_OUTPUT_MISMATCH]")
            if not enabled:
                return function(*args, **kwargs)
            with _KernelProfileSession(Path(output)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def kernel_profile_step() -> None:
    active = _ACTIVE_PROFILE.get()
    if active is not None:
        active.step()


def kernel_profile_range(label: str):
    if _ACTIVE_PROFILE.get() is None:
        return contextlib.nullcontext()
    return torch.profiler.record_function("GX1/" + label)
