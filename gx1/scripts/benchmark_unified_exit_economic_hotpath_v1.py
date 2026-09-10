"""CPU-only scalar-vs-vector benchmark for lifecycle-v2 economic materialization."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    require_economic_training_projection,
)
from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    LazyUnifiedExitEconomicStepProviderV1,
)
from gx1.contracts.unified_exit_economics_objective_v2 import compose_economic_step
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import canonical_sha256


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("HOTPATH_BENCHMARK_JSON_INVALID")
    return value


def _measure(call: Callable[[], tuple[np.ndarray, np.ndarray]], repeats: int) -> dict:
    durations = []
    peak_bytes = []
    output = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        output = call()
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes.append(peak)
    assert output is not None
    median = statistics.median(durations)
    return {
        "median_seconds_per_pack": median,
        "packs_per_second": 1.0 / median,
        "min_seconds_per_pack": min(durations),
        "max_seconds_per_pack": max(durations),
        "peak_python_allocation_bytes": max(peak_bytes),
        "output_array_bytes": sum(array.nbytes for array in output),
        "output": output,
    }


def benchmark(*, authority_path: Path, readiness_path: Path, repeats: int) -> dict:
    authority = _read_json(authority_path)
    policy = _read_json(Path(authority["policy"]["path"]))
    tape_path = Path(policy["executable_bid_ask"]["parquet"]["path"])
    times = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(tape_path, columns=["time"])["time"], utc=True)
    )
    entry_start = int(times.searchsorted(pd.Timestamp("2025-06-02T00:00:00Z")))
    state_count = 512
    expected = pd.Timedelta(minutes=state_count)
    if times[entry_start + state_count] - times[entry_start] != expected:
        raise RuntimeError("HOTPATH_BENCHMARK_REQUIRES_CONTINUOUS_TRAIN_CLOCK")
    compact = pd.DataFrame(
        {
            "entry_row_index": [0],
            "entry_m1_start_row": [entry_start],
            "long_lifecycle_state_count": [state_count],
            "short_lifecycle_state_count": [state_count],
            "long_economic_terminal": [False],
            "short_economic_terminal": [False],
            "m1_source_sha256": [
                policy["executable_bid_ask"]["parquet"]["sha256"]
            ],
        }
    )
    manifest = {
        "split": "train",
        "test_accessed": False,
        "gap_classification_source_sha256": policy["executable_bid_ask"]["manifest"][
            "sha256"
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    readiness = _read_json(readiness_path)
    provider = LazyUnifiedExitEconomicStepProviderV1(
        compact_rows=compact,
        compact_manifest=manifest,
        economics_readiness=readiness,
        cost_parameter_authority_path=authority_path,
    )
    contract = readiness["economics_objective_contract"]
    economic_manifest = provider.economic_exit_step_manifest

    def scalar() -> tuple[np.ndarray, np.ndarray]:
        exit_envelope = provider(0, 0, "exit_now", 0, state_count)
        hold_envelope = provider(0, 0, "hold", 0, state_count - 1)
        for envelope in (exit_envelope, hold_envelope):
            declared = envelope["slice_sha256"]
            payload = {key: value for key, value in envelope.items() if key != "slice_sha256"}
            if declared != canonical_sha256(payload):
                raise RuntimeError("HOTPATH_BENCHMARK_SCALAR_HASH_INVALID")
        exit_reward = np.asarray(
            [
                compose_economic_step(step, contract=contract)[
                    "undiscounted_risk_adjusted_utility_increment_bps"
                ]
                for step in exit_envelope["steps"]
            ],
            dtype=np.float32,
        )
        hold_reward = np.asarray(
            [
                compose_economic_step(step, contract=contract)[
                    "undiscounted_risk_adjusted_utility_increment_bps"
                ]
                for step in hold_envelope["steps"]
            ],
            dtype=np.float32,
        )
        return exit_reward, hold_reward

    def vectorized() -> tuple[np.ndarray, np.ndarray]:
        projection = require_economic_training_projection(
            provider.materialize_training_projection(
                0, 0, 0, state_count, state_count - 1
            ),
            entry_row_index=0,
            side_index=0,
            start_state_index=0,
            stop_state_index=state_count,
            hold_stop_state_index=state_count - 1,
            economic_manifest=economic_manifest,
        )
        return (
            np.ascontiguousarray(projection["exit_reward_bps"], dtype=np.float32),
            np.ascontiguousarray(projection["hold_reward_bps"], dtype=np.float32),
        )

    scalar_output = scalar()
    vector_output = vectorized()
    parity = all(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
        for left, right in zip(scalar_output, vector_output)
    )
    if not parity:
        raise RuntimeError("HOTPATH_BENCHMARK_ARRAY_PARITY_FAILED")
    scalar_result = _measure(scalar, repeats)
    vector_result = _measure(vectorized, repeats)
    speedup = scalar_result["median_seconds_per_pack"] / vector_result[
        "median_seconds_per_pack"
    ]
    peak_reduction = scalar_result["peak_python_allocation_bytes"] / vector_result[
        "peak_python_allocation_bytes"
    ]
    for result in (scalar_result, vector_result):
        result.pop("output")
    return {
        "schema_version": "gx1_unified_exit_economic_hotpath_benchmark_v1",
        "decision": "PASS",
        "device": "cpu",
        "test_data_used": False,
        "split": "train",
        "entry_timestamp_utc": str(times[entry_start]),
        "state_count": state_count,
        "hold_transition_count": state_count - 1,
        "repeats": repeats,
        "float32_array_byte_parity": parity,
        "scalar": scalar_result,
        "vectorized": vector_result,
        "median_speedup_x": speedup,
        "peak_python_allocation_reduction_x": peak_reduction,
        "scope": "economic provider plus projection validation and float32 conversion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3:
        raise RuntimeError("HOTPATH_BENCHMARK_REPEATS_INVALID")
    result = benchmark(
        authority_path=args.authority.expanduser().resolve(),
        readiness_path=args.readiness.expanduser().resolve(),
        repeats=args.repeats,
    )
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
