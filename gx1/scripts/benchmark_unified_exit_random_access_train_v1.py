"""TRAIN-only CPU benchmark for canonical random-access adapter candidates."""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

import torch

from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    BENCHMARK_BUDGETS,
    BENCHMARK_TRANSITIONS_PER_ENTRY,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_training_items,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def benchmark_random_access_train_candidates_v1(
    *,
    adapter_factory: Callable[[int], Any],
    batch_sizes: Sequence[int],
    repeats: int = 3,
    max_selected_entries: int | None = None,
) -> dict[str, Any]:
    """Measure actual adapter materialization+collation without model/CUDA work.

    ``adapter_factory`` must return a separately configured immutable TRAIN
    adapter for the requested transition budget. A cap is permitted for a
    smoke measurement, but the receipt marks it non-authoritative.
    """

    sizes = tuple(batch_sizes)
    if (
        not callable(adapter_factory)
        or not sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes
        )
        or len(set(sizes)) != len(sizes)
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
        or (
            max_selected_entries is not None
            and (
                isinstance(max_selected_entries, bool)
                or not isinstance(max_selected_entries, int)
                or max_selected_entries < 1
            )
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_INVOCATION_INVALID")
    candidates: list[dict[str, Any]] = []
    for budget in BENCHMARK_BUDGETS:
        adapter = adapter_factory(budget)
        bindings = adapter.random_access_training_bindings_v1()
        contract = bindings["sampler_contract"]
        if (
            contract.get("split") != "train"
            or contract.get("transition_budget_per_epoch") != budget
            or contract.get("transitions_per_entry") != BENCHMARK_TRANSITIONS_PER_ENTRY
            or contract.get("test_data_used") is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_CONTRACT_INVALID")
        selected = adapter.random_access_selected_entry_rows_v1()
        expected_entries = budget // BENCHMARK_TRANSITIONS_PER_ENTRY
        if len(selected) != expected_entries:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_SCHEDULE_INVALID")
        measured = selected[:max_selected_entries]
        measurements: list[dict[str, Any]] = []
        for batch_size in sizes:
            durations: list[float] = []
            peaks: list[int] = []
            padded_bytes: list[int] = []
            for _repeat in range(repeats):
                gc.collect()
                tracemalloc.start()
                started = time.perf_counter()
                transition_count = 0
                batch_padded = 0
                for offset in range(0, len(measured), batch_size):
                    rows = measured[offset : offset + batch_size]
                    items = [
                        adapter.materialize_random_access_training_item_v1(
                            entry, outer_batch_index=position
                        )
                        for position, entry in enumerate(rows)
                    ]
                    if any(item is None for item in items):
                        raise RuntimeError(
                            "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_ITEM_MISSING"
                        )
                    collated = collate_random_access_training_items(
                        items,
                        outer_batch_size=len(rows),
                        sampler_contract=contract,
                        normalization_artifact=bindings["normalization_artifact"],
                        expected_m1_source_sha256=bindings["m1_source_sha256"],
                        expected_market_closure_authority_sha256=bindings[
                            "market_closure_authority_sha256"
                        ],
                        expected_economic_step_manifest_sha256=bindings[
                            "economic_step_manifest_sha256"
                        ],
                        expected_economics_objective_contract_sha256=bindings[
                            "economics_objective_contract_sha256"
                        ],
                        device=torch.device("cpu"),
                    )
                    transition_count += int(collated["transition_count"])
                    for model_inputs in (
                        collated["online_model_inputs"],
                        collated["target_model_inputs"],
                    ):
                        batch_padded += sum(
                            tensor.numel() * tensor.element_size()
                            for tensor in model_inputs.values()
                            if isinstance(tensor, torch.Tensor)
                        )
                duration = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if transition_count != len(measured) * BENCHMARK_TRANSITIONS_PER_ENTRY:
                    raise RuntimeError(
                        "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_TRANSITION_DRIFT"
                    )
                durations.append(duration)
                peaks.append(peak)
                padded_bytes.append(batch_padded)
            median = statistics.median(durations)
            measurements.append(
                {
                    "batch_size": batch_size,
                    "measured_entry_pairs": len(measured),
                    "measured_transitions": len(measured)
                    * BENCHMARK_TRANSITIONS_PER_ENTRY,
                    "median_seconds": median,
                    "entry_pairs_per_second": len(measured) / median,
                    "transitions_per_second": (
                        len(measured) * BENCHMARK_TRANSITIONS_PER_ENTRY / median
                    ),
                    "peak_python_allocation_bytes": max(peaks),
                    "model_input_tensor_bytes_per_repeat": int(
                        statistics.median(padded_bytes)
                    ),
                }
            )
        candidates.append(
            {
                "transition_budget_per_epoch": budget,
                "sampler_contract_sha256": contract["contract_sha256"],
                "full_selected_entry_pairs": expected_entries,
                "full_budget_measured": len(measured) == expected_entries,
                "batch_size_sweep": measurements,
            }
        )
    receipt = {
        "schema_version": "gx1_unified_exit_random_access_train_benchmark_v1",
        "decision": (
            "PASS" if max_selected_entries is None else "NON_AUTHORITATIVE_CAPPED_SMOKE"
        ),
        "device": "cpu",
        "candidate_budgets": list(BENCHMARK_BUDGETS),
        "transitions_per_entry": BENCHMARK_TRANSITIONS_PER_ENTRY,
        "batch_sizes": list(sizes),
        "repeats": repeats,
        "candidate_selection_performed": False,
        "selection_requires_external_preregistered_rule": True,
        "test_data_used": False,
        "candidates": candidates,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = ("benchmark_random_access_train_candidates_v1",)
