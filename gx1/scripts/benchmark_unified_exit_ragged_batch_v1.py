"""CPU microbenchmark for schema-neutral sampled-transition ragged batching."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from gx1.contracts.unified_exit_ragged_batch_v1 import (
    one_forward_one_backward,
    right_pad_arrays,
)


class _BenchmarkTailModel(nn.Module):
    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, *, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rows = self.network(values)
        last = lengths.to(torch.long) - 1
        return rows[torch.arange(len(rows)), last].reshape(len(rows), 2, 2)


def _measure(call: Callable[[], None], repeats: int, transition_count: int) -> dict:
    durations: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    median = statistics.median(durations)
    return {
        "median_seconds_per_entry_batch": median,
        "entry_batches_per_second": 1.0 / median,
        "sampled_transitions_per_second": transition_count / median,
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "peak_python_allocation_bytes": max(peaks or [0]),
    }


def benchmark(*, repeats: int, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lengths_list = [448, 512, 471, 503, 489, 460, 496, 477]
    tails = [
        rng.normal(size=(length, 32)).astype(np.float32)
        for length in lengths_list
    ]
    collated = right_pad_arrays(tails, max_rows_cap=512)
    values = torch.from_numpy(collated["values"])
    lengths = torch.from_numpy(collated["lengths"])
    valid = torch.from_numpy(
        rng.random((len(tails), 2, 2)) > 0.15
    ).to(torch.bool)
    valid[:, :, 1] = True
    weights = torch.from_numpy(
        rng.uniform(0.5, 2.0, size=len(tails)).astype(np.float32)
    )
    base_online = _BenchmarkTailModel(32, 128)
    base_target = copy.deepcopy(base_online)

    def run_batch(
        online: _BenchmarkTailModel, target: _BenchmarkTailModel
    ) -> tuple[torch.Tensor, torch.Tensor]:
        online.zero_grad(set_to_none=True)
        result = one_forward_one_backward(
            online_model=online,
            target_model=target,
            online_inputs={"values": values, "lengths": lengths},
            target_inputs={"values": values, "lengths": lengths},
            target_builder=lambda prediction: (prediction + 0.25, valid),
            importance_weight=weights,
        )
        return result["raw_loss"], online.network[0].weight.grad.detach().clone()

    denominator = sum(
        float(weights[index]) * int(valid[index].sum())
        for index in range(len(tails))
    )

    def run_loop(
        online: _BenchmarkTailModel, target: _BenchmarkTailModel
    ) -> tuple[torch.Tensor, torch.Tensor]:
        online.zero_grad(set_to_none=True)
        loss_sum = torch.tensor(0.0)
        for index, tail in enumerate(tails):
            value = torch.from_numpy(tail).unsqueeze(0)
            length = torch.tensor([len(tail)])
            with torch.no_grad():
                target_value = target(values=value, lengths=length) + 0.25
            prediction = online(values=value, lengths=length)
            row_loss = (
                (prediction - target_value).square()
                * valid[index : index + 1].to(prediction.dtype)
                * weights[index]
            ).sum()
            (row_loss / denominator).backward()
            loss_sum = loss_sum + row_loss.detach()
        return loss_sum / denominator, online.network[0].weight.grad.detach().clone()

    parity_batch = run_batch(copy.deepcopy(base_online), copy.deepcopy(base_target))
    parity_loop = run_loop(copy.deepcopy(base_online), copy.deepcopy(base_target))
    loss_parity = bool(
        torch.allclose(parity_batch[0], parity_loop[0], rtol=1e-6, atol=1e-8)
    )
    gradient_parity = bool(
        torch.allclose(parity_batch[1], parity_loop[1], rtol=1e-5, atol=1e-7)
    )
    if not loss_parity or not gradient_parity:
        raise RuntimeError("UNIFIED_EXIT_RAGGED_BATCH_BENCHMARK_PARITY_FAILED")

    batch_online = copy.deepcopy(base_online)
    batch_target = copy.deepcopy(base_target)
    loop_online = copy.deepcopy(base_online)
    loop_target = copy.deepcopy(base_target)
    batch = _measure(
        lambda: run_batch(batch_online, batch_target), repeats, len(tails)
    )
    loop = _measure(
        lambda: run_loop(loop_online, loop_target), repeats, len(tails)
    )
    return {
        "schema_version": "gx1_unified_exit_ragged_batch_benchmark_v1",
        "decision": "PASS",
        "device": "cpu",
        "test_data_used": False,
        "sampled_transition_count": len(tails),
        "tail_lengths": lengths_list,
        "tail_rows_cap": 512,
        "both_sides": True,
        "action_count": 2,
        "repeats": repeats,
        "loss_parity": loss_parity,
        "gradient_parity": gradient_parity,
        "unpadded_input_bytes": collated["unpadded_nbytes"],
        "padded_input_bytes": collated["padded_nbytes"],
        "padding_memory_overhead_x": (
            collated["padded_nbytes"] / collated["unpadded_nbytes"]
        ),
        "sample_loop": loop,
        "ragged_batch": batch,
        "median_speedup_x": (
            loop["median_seconds_per_entry_batch"]
            / batch["median_seconds_per_entry_batch"]
        ),
        "scope": (
            "one target forward plus one online forward plus one backward; "
            "schema-neutral sampled-transition tail batch"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3:
        raise RuntimeError("UNIFIED_EXIT_RAGGED_BATCH_BENCHMARK_REPEATS_INVALID")
    result = benchmark(repeats=args.repeats, seed=args.seed)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
