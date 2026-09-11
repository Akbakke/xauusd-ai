"""TRAIN-only CPU benchmark for canonical random-access adapter candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    BENCHMARK_BUDGETS,
    BENCHMARK_TRANSITIONS_PER_ENTRY,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_training_items,
)


AUTHORITATIVE_BATCH_SIZE = 16
AUTHORITATIVE_REPEATS = 1
MAX_PEAK_PYTHON_ALLOCATION_BYTES = 2 * 1024**3
MAX_PEAK_PADDED_MODEL_INPUT_BYTES = 1024**3
MAX_MEASURED_CPU_PREP_EPOCH_SECONDS = 30 * 60
ENTRY_PAIR_POPULATION = 65_295
SELECTION_RULE_VERSION = "minimum_population_cycle_epochs_with_30m_cpu_cap_v1"


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


def _tensor_bytes(value: Mapping[str, Any]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in value.values()
        if isinstance(tensor, torch.Tensor)
    )


def _selection_policy() -> dict[str, Any]:
    value = {
        "schema_version": "gx1_unified_exit_sampler_selection_policy_v1",
        "rule": SELECTION_RULE_VERSION,
        "authoritative_batch_size": AUTHORITATIVE_BATCH_SIZE,
        "authoritative_repeats": AUTHORITATIVE_REPEATS,
        "entry_pair_population": ENTRY_PAIR_POPULATION,
        "max_peak_python_allocation_bytes": MAX_PEAK_PYTHON_ALLOCATION_BYTES,
        "max_peak_padded_model_input_bytes": MAX_PEAK_PADDED_MODEL_INPUT_BYTES,
        "max_measured_cpu_prep_epoch_seconds": MAX_MEASURED_CPU_PREP_EPOCH_SECONDS,
        "eligibility": (
            "complete_once_at_batch16_with_cpu_epoch_and_memory_within_caps"
        ),
        "rank_order": [
            "population_cycle_epochs_ascending",
            "measured_epoch_seconds_ascending",
            "peak_padded_model_input_bytes_ascending",
            "peak_python_allocation_bytes_ascending",
            "transition_budget_per_epoch_ascending",
        ],
        "selection_uses_outcome_values": False,
        "test_data_used": False,
    }
    value["policy_sha256"] = _canonical_sha256(value)
    return value


def benchmark_random_access_train_candidates_v1(
    *,
    adapter_factory: Callable[[int], Any],
    batch_sizes: Sequence[int],
    repeats: int = 3,
    max_selected_entries: int | None = None,
    run_bindings: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure adapter materialization+collation without model/CUDA work."""

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
        or (run_bindings is not None and not isinstance(run_bindings, Mapping))
        or (progress_callback is not None and not callable(progress_callback))
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_INVOCATION_INVALID")
    authoritative = max_selected_entries is None
    if authoritative and (
        sizes != (AUTHORITATIVE_BATCH_SIZE,) or repeats != AUTHORITATIVE_REPEATS
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_PREREGISTRATION_MISMATCH"
        )
    policy = _selection_policy()
    candidates: list[dict[str, Any]] = []
    for budget in BENCHMARK_BUDGETS:
        adapter = adapter_factory(budget)
        bindings = adapter.random_access_training_bindings_v1()
        contract = bindings["sampler_contract"]
        if (
            contract.get("split") != "train"
            or contract.get("transition_budget_per_epoch") != budget
            or contract.get("transitions_per_entry") != BENCHMARK_TRANSITIONS_PER_ENTRY
            or contract.get("entry_pair_population") != ENTRY_PAIR_POPULATION
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
            materialize_durations: list[float] = []
            collate_durations: list[float] = []
            peaks: list[int] = []
            peak_padded_bytes: list[int] = []
            for repeat_index in range(repeats):
                gc.collect()
                tracemalloc.start()
                started = time.perf_counter()
                materialize_seconds = 0.0
                collate_seconds = 0.0
                transition_count = 0
                repeat_peak_padded_bytes = 0
                last_progress = started
                for offset in range(0, len(measured), batch_size):
                    rows = measured[offset : offset + batch_size]
                    materialize_started = time.perf_counter()
                    items = [
                        adapter.materialize_random_access_training_item_v1(
                            entry, outer_batch_index=position
                        )
                        for position, entry in enumerate(rows)
                    ]
                    materialize_seconds += time.perf_counter() - materialize_started
                    if any(item is None for item in items):
                        raise RuntimeError(
                            "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_ITEM_MISSING"
                        )
                    collate_started = time.perf_counter()
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
                    collate_seconds += time.perf_counter() - collate_started
                    transition_count += int(collated["transition_count"])
                    current_padded_bytes = sum(
                        _tensor_bytes(model_inputs)
                        for model_inputs in (
                            collated["online_model_inputs"],
                            collated["target_model_inputs"],
                        )
                    )
                    repeat_peak_padded_bytes = max(
                        repeat_peak_padded_bytes, current_padded_bytes
                    )
                    now = time.perf_counter()
                    if progress_callback is not None and now - last_progress >= 60.0:
                        completed_entries = min(offset + len(rows), len(measured))
                        progress_callback(
                            {
                                "event": "benchmark_progress",
                                "transition_budget_per_epoch": budget,
                                "batch_size": batch_size,
                                "repeat_index": repeat_index,
                                "completed_entry_pairs": completed_entries,
                                "total_entry_pairs": len(measured),
                                "elapsed_seconds": now - started,
                                "eta_seconds": (now - started)
                                * (len(measured) - completed_entries)
                                / completed_entries,
                            }
                        )
                        last_progress = now
                duration = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if transition_count != len(measured) * BENCHMARK_TRANSITIONS_PER_ENTRY:
                    raise RuntimeError(
                        "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_TRANSITION_DRIFT"
                    )
                durations.append(duration)
                materialize_durations.append(materialize_seconds)
                collate_durations.append(collate_seconds)
                peaks.append(peak)
                peak_padded_bytes.append(repeat_peak_padded_bytes)
            median = statistics.median(durations)
            full_budget_measured = len(measured) == expected_entries
            population_cycle_epochs = math.ceil(
                ENTRY_PAIR_POPULATION / expected_entries
            )
            measurements.append(
                {
                    "batch_size": batch_size,
                    "measured_entry_pairs": len(measured),
                    "measured_transitions": len(measured)
                    * BENCHMARK_TRANSITIONS_PER_ENTRY,
                    "median_seconds": median,
                    "materialize_seconds": statistics.median(materialize_durations),
                    "collate_seconds": statistics.median(collate_durations),
                    "entry_pairs_per_second": len(measured) / median,
                    "transitions_per_second": len(measured)
                    * BENCHMARK_TRANSITIONS_PER_ENTRY
                    / median,
                    "peak_python_allocation_bytes": max(peaks),
                    "peak_padded_model_input_bytes": max(peak_padded_bytes),
                    "population_cycle_epochs": population_cycle_epochs,
                    "measured_epoch_seconds": median if full_budget_measured else None,
                    "projected_entry_population_cycle_seconds": (
                        median * population_cycle_epochs
                        if full_budget_measured
                        else None
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
    decision = "NON_AUTHORITATIVE_CAPPED_SMOKE"
    selected_result: dict[str, Any] | None = None
    if authoritative:
        eligible: list[tuple[tuple[float | int, ...], dict[str, Any]]] = []
        for candidate in candidates:
            rows = candidate["batch_size_sweep"]
            if not candidate["full_budget_measured"] or len(rows) != 1:
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_INCOMPLETE")
            row = rows[0]
            if (
                row["batch_size"] != AUTHORITATIVE_BATCH_SIZE
                or row["measured_epoch_seconds"] > MAX_MEASURED_CPU_PREP_EPOCH_SECONDS
                or row["peak_python_allocation_bytes"]
                > MAX_PEAK_PYTHON_ALLOCATION_BYTES
                or row["peak_padded_model_input_bytes"]
                > MAX_PEAK_PADDED_MODEL_INPUT_BYTES
            ):
                continue
            rank = (
                row["population_cycle_epochs"],
                row["measured_epoch_seconds"],
                row["peak_padded_model_input_bytes"],
                row["peak_python_allocation_bytes"],
                candidate["transition_budget_per_epoch"],
            )
            eligible.append((rank, candidate))
        if not eligible:
            raise RuntimeError(
                "UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_NO_ELIGIBLE_CANDIDATE"
            )
        chosen = min(eligible, key=lambda item: item[0])[1]
        chosen_row = chosen["batch_size_sweep"][0]
        selected_result = {
            "transition_budget_per_epoch": chosen["transition_budget_per_epoch"],
            "sampler_contract_sha256": chosen["sampler_contract_sha256"],
            "batch_size": chosen_row["batch_size"],
            "population_cycle_epochs": chosen_row["population_cycle_epochs"],
            "measured_epoch_seconds": chosen_row["measured_epoch_seconds"],
            "projected_entry_population_cycle_seconds": chosen_row[
                "projected_entry_population_cycle_seconds"
            ],
        }
        decision = "PASS"
    receipt = {
        "schema_version": "gx1_unified_exit_random_access_train_benchmark_v2",
        "decision": decision,
        "device": "cpu",
        "candidate_budgets": list(BENCHMARK_BUDGETS),
        "transitions_per_entry": BENCHMARK_TRANSITIONS_PER_ENTRY,
        "batch_sizes": list(sizes),
        "repeats": repeats,
        "selection_policy": policy,
        "candidate_selection_performed": selected_result is not None,
        "selected_candidate": selected_result,
        "selection_uses_outcome_values": False,
        "test_data_used": False,
        "run_bindings": dict(run_bindings or {}),
        "candidates": candidates,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BENCHMARK_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root-manifest",
        "candidate-set",
        "composite-normalization",
        "economics-readiness",
        "train-cost-authority",
        "feature-lifecycle-root",
        "entry-train-parquet",
        "entry-train-manifest",
        "sequence-source-audit",
        "m5-prebuilt",
        "mtf-cache-dir",
        "output-json",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--m5-len", type=int, default=16)
    parser.add_argument("--m15-len", type=int, default=64)
    parser.add_argument("--h1-len", type=int, default=96)
    parser.add_argument("--h4-len", type=int, default=96)
    parser.add_argument("--d1-len", type=int, default=252)
    parser.add_argument("--batch-size", type=int, default=AUTHORITATIVE_BATCH_SIZE)
    parser.add_argument("--repeats", type=int, default=AUTHORITATIVE_REPEATS)
    parser.add_argument("--max-selected-entries", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(
        args.mtf_cache_dir.expanduser().resolve()
    )
    from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
    from gx1.contracts.unified_exit_random_access_train_factory_v1 import (
        build_random_access_train_adapter_factory_v1,
    )
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset

    manifest_path = args.entry_train_manifest.expanduser().resolve()
    entry_path = args.entry_train_parquet.expanduser().resolve()
    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=args.feature_lifecycle_root,
        entry_parquets={"train": entry_path},
        entry_manifest_bindings={
            "train": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
        },
        dataset_run_id=args.dataset_run_id,
        splits=("train",),
    )
    dataset = EntryV10CtxDataset(
        parquet_path=entry_path,
        seq_len=args.seq_len,
        m5_prebuilt_path=args.m5_prebuilt,
        per_tf_seq_lens={
            "M5": args.m5_len,
            "M15": args.m15_len,
            "H1": args.h1_len,
            "H4": args.h4_len,
            "D1": args.d1_len,
        },
        multi_tf_closed_bar=True,
        sequence_source_audit_json=args.sequence_source_audit,
    )
    factory = build_random_access_train_adapter_factory_v1(
        root_manifest_path=args.root_manifest,
        candidate_set_path=args.candidate_set,
        composite_normalization_path=args.composite_normalization,
        economics_readiness_path=args.economics_readiness,
        train_cost_authority_path=args.train_cost_authority,
        train_dataset=dataset,
        train_feature_source_owner=corpus.splits["train"],
    )
    path_bindings = {
        key: {"path": str(path.expanduser().resolve()), "sha256": file_sha256(path)}
        for key, path in {
            "root_manifest": args.root_manifest,
            "candidate_set": args.candidate_set,
            "composite_normalization": args.composite_normalization,
            "economics_readiness": args.economics_readiness,
            "train_cost_authority": args.train_cost_authority,
            "feature_lifecycle_root": args.feature_lifecycle_root,
            "entry_train_parquet": args.entry_train_parquet,
            "entry_train_manifest": args.entry_train_manifest,
            "sequence_source_audit": args.sequence_source_audit,
            "m5_prebuilt": args.m5_prebuilt,
        }.items()
    }
    receipt = benchmark_random_access_train_candidates_v1(
        adapter_factory=factory,
        batch_sizes=(args.batch_size,),
        repeats=args.repeats,
        max_selected_entries=args.max_selected_entries,
        run_bindings={
            "dataset_run_id": args.dataset_run_id,
            "mtf_cache_dir": str(args.mtf_cache_dir.expanduser().resolve()),
            "files": path_bindings,
        },
        progress_callback=lambda event: print(
            json.dumps(event, sort_keys=True, allow_nan=False), flush=True
        ),
    )
    _atomic_write_new_json(args.output_json, receipt)
    print(json.dumps({"event": "benchmark_complete", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("benchmark_random_access_train_candidates_v1", "main")
