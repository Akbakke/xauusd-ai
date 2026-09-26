"""Test isolation for process-global torch state (added 2026-09-26).

Some tests exercise the trainer's runtime configuration, which sets process-global
torch state (intra-op thread count, deterministic algorithms). Measured 2026-09-26:
after tests/test_candidate_training_session.py the process ran with 8 threads and
deterministic algorithms on, and a chunked-vs-full numerical parity test in
tests/test_entry_v10_ctx_model_shapes.py failed only in full-suite order. Every test
now returns the state it found, so results do not depend on test order.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_torch_global_state():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        yield
        return
    threads = torch.get_num_threads()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    yield
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    if torch.get_default_dtype() != dtype:
        torch.set_default_dtype(dtype)
    if (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) != (deterministic, warn_only):
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if torch.get_float32_matmul_precision() != precision:
        torch.set_float32_matmul_precision(precision)
