"""TRAIN-only normalization and first-state bridge contracts for lifecycle-v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    fit_surface_normalization,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    LIFETIME_SUMMARY_FIELD_ORDER,
    lifetime_summary_registry,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    DURATION_BUCKETS,
    build_random_access_sampler_contract,
)


SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION = (
    "gx1_unified_exit_physical_summary_normalization_sample_v1"
)
SUMMARY_NORMALIZATION_SCHEMA_VERSION = (
    "gx1_unified_exit_lifetime_summary_normalization_v1"
)
FIRST_STATE_BRIDGE_SCHEMA_VERSION = "gx1_unified_exit_first_state_entry_bridge_v1"
BENCHMARK_CANDIDATE_SET_SCHEMA_VERSION = (
    "gx1_unified_exit_sampler_benchmark_candidate_set_v1"
)
BENCHMARK_BUDGETS = (32_768, 65_536, 131_072)
BENCHMARK_TRANSITIONS_PER_ENTRY = 4


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_PILOT_{label}_SHA_INVALID")
    return value


def build_sampler_benchmark_candidate_set(
    *, source_lineage_sha256: str, entry_pair_population: int
) -> dict[str, Any]:
    """Declare benchmark inputs without selecting or admitting an epoch budget."""

    lineage = _require_sha(source_lineage_sha256, "SAMPLER_LINEAGE")
    if (
        isinstance(entry_pair_population, bool)
        or not isinstance(entry_pair_population, int)
        or entry_pair_population < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_PILOT_ENTRY_POPULATION_INVALID")
    candidates = []
    for budget in BENCHMARK_BUDGETS:
        contract = build_random_access_sampler_contract(
            split="train",
            source_lineage_sha256=lineage,
            transition_budget_per_epoch=budget,
            transitions_per_entry=BENCHMARK_TRANSITIONS_PER_ENTRY,
            entry_pair_population=entry_pair_population,
        )
        candidates.append(
            {
                "status": "BENCHMARK_PENDING",
                "selected": False,
                "sampler_contract": contract,
            }
        )
    value = {
        "schema_version": BENCHMARK_CANDIDATE_SET_SCHEMA_VERSION,
        "decision": "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK",
        "source_lineage_sha256": lineage,
        "entry_pair_population": entry_pair_population,
        "transitions_per_entry": BENCHMARK_TRANSITIONS_PER_ENTRY,
        "candidates": candidates,
        "selected_sampler_contract_sha256": None,
        "selection_requires_measured_throughput_and_memory": True,
        "selection_uses_outcome_values": False,
        "test_data_used": False,
    }
    value["candidate_set_sha256"] = canonical_sha256(value)
    return value


def iter_physical_summary_samples(
    *, successor_transition_count_by_entry: Sequence[int], source_lineage_sha256: str
) -> Iterable[dict[str, Any]]:
    """Yield one deterministic state per physical Entry and eligible duration bucket.

    Sampling depends only on lifecycle length, Entry identity, bucket bounds and
    immutable lineage. It is independent of epoch budgets and all outcomes.
    """

    lineage = _require_sha(source_lineage_sha256, "SUMMARY_LINEAGE")
    counts = tuple(successor_transition_count_by_entry)
    if not counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts
    ):
        raise RuntimeError("UNIFIED_EXIT_PILOT_SUMMARY_POPULATION_INVALID")
    for entry_row_index, count in enumerate(counts):
        for bucket_index, (start, raw_stop) in enumerate(DURATION_BUCKETS):
            if start >= count:
                continue
            stop = min(count, raw_stop or count)
            seed = canonical_sha256(
                {
                    "schema_version": SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION,
                    "source_lineage_sha256": lineage,
                    "entry_row_index": entry_row_index,
                    "duration_bucket_index": bucket_index,
                    "start_inclusive": start,
                    "stop_exclusive": stop,
                }
            )
            state_index = start + int(seed, 16) % (stop - start)
            value = {
                "entry_row_index": entry_row_index,
                "duration_bucket_index": bucket_index,
                "state_index": state_index,
                "both_sides": True,
                "selection_uses_outcome_values": False,
            }
            value["sample_sha256"] = canonical_sha256(value)
            yield value


def build_physical_summary_sample_authority(
    *, successor_transition_count_by_entry: Sequence[int], source_lineage_sha256: str
) -> dict[str, Any]:
    counts = tuple(successor_transition_count_by_entry)
    samples = tuple(
        iter_physical_summary_samples(
            successor_transition_count_by_entry=counts,
            source_lineage_sha256=source_lineage_sha256,
        )
    )
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample["sample_sha256"].encode("ascii"))
        digest.update(b"\n")
    registry = lifetime_summary_registry()
    value = {
        "schema_version": SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION,
        "decision": "PASS",
        "source_lineage_sha256": _require_sha(
            source_lineage_sha256, "SUMMARY_LINEAGE"
        ),
        "entry_pair_population": len(counts),
        "successor_counts_sha256": hashlib.sha256(
            np.asarray(counts, dtype="<i8").tobytes()
        ).hexdigest(),
        "selection": "one_hash_selected_state_per_entry_per_eligible_duration_bucket_v1",
        "sample_count": len(samples),
        "side_rows_per_sample": 2,
        "fit_row_count": len(samples) * 2,
        "sample_stream_sha256": digest.hexdigest(),
        "lifetime_summary_registry_sha256": registry["registry_sha256"],
        "field_order_sha256": registry["field_order_sha256"],
        "epoch_sampler_independent": True,
        "selection_uses_outcome_values": False,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["authority_sha256"] = canonical_sha256(value)
    return value


def fit_lifetime_summary_normalization(
    *, values: Any, sample_authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = dict(sample_authority)
    claimed = authority.pop("authority_sha256", None)
    if (
        sample_authority.get("schema_version")
        != SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION
        or sample_authority.get("decision") != "PASS"
        or claimed != canonical_sha256(authority)
        or sample_authority.get("selection_uses_outcome_values") is not False
        or sample_authority.get("val_fit_rows") != 0
        or sample_authority.get("test_fit_rows") != 0
        or sample_authority.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_PILOT_SUMMARY_AUTHORITY_INVALID")
    matrix = np.ascontiguousarray(values, dtype="<f8")
    registry = lifetime_summary_registry()
    expected_shape = (int(sample_authority["fit_row_count"]), len(LIFETIME_SUMMARY_FIELD_ORDER))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise RuntimeError("UNIFIED_EXIT_PILOT_SUMMARY_VALUES_INVALID")
    surface = fit_surface_normalization(
        matrix,
        surface="lifetime_summary",
        field_names=LIFETIME_SUMMARY_FIELD_ORDER,
        row_count=matrix.shape[0],
    )
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in surface.items()
    }
    result = {
        "schema_version": SUMMARY_NORMALIZATION_SCHEMA_VERSION,
        "decision": "PASS",
        "fit_scope": "train_only_outcome_blind_physical_state_sample",
        "sample_authority_sha256": sample_authority["authority_sha256"],
        "sample_values_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "lifetime_summary_registry_sha256": registry["registry_sha256"],
        "field_order": list(LIFETIME_SUMMARY_FIELD_ORDER),
        "field_order_sha256": registry["field_order_sha256"],
        "surface": serializable,
        "train_fit_rows": matrix.shape[0],
        "val_fit_rows": 0,
        "val_mode": "apply_frozen_train_transform_only",
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    result["normalization_sha256"] = canonical_sha256(result)
    return result


def build_first_state_entry_bridge_witness(
    *,
    split: str,
    entry_times: Sequence[Any],
    m1_times: Sequence[Any],
    child_admission_sha256: str,
    child_parquet_sha256: str,
    entry_sequence_audit_sha256: str,
    m1_source_sha256: str,
    closure_authority_sha256: str,
    state_view_source_sha256: str,
    lifetime_summary_registry_sha256: str,
    train_normalization_sha256: str,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_PILOT_BRIDGE_SPLIT_INVALID")
    bindings = {
        name: _require_sha(value, name.upper())
        for name, value in {
            "child_admission": child_admission_sha256,
            "child_parquet": child_parquet_sha256,
            "entry_sequence_audit": entry_sequence_audit_sha256,
            "m1_source": m1_source_sha256,
            "closure_authority": closure_authority_sha256,
            "state_view_source": state_view_source_sha256,
            "lifetime_summary_registry": lifetime_summary_registry_sha256,
            "train_normalization": train_normalization_sha256,
        }.items()
    }
    entry = pd.DatetimeIndex(pd.to_datetime(entry_times, utc=True, errors="coerce")).as_unit("ns")
    m1 = pd.DatetimeIndex(pd.to_datetime(m1_times, utc=True, errors="coerce")).as_unit("ns")
    if (
        entry.empty
        or entry.hasnans
        or not entry.is_unique
        or not entry.is_monotonic_increasing
        or m1.empty
        or m1.hasnans
        or not m1.is_unique
        or not m1.is_monotonic_increasing
    ):
        raise RuntimeError("UNIFIED_EXIT_PILOT_BRIDGE_CLOCK_INVALID")
    expected = entry.asi8 + 5 * 60 * 1_000_000_000
    positions = np.searchsorted(m1.asi8, expected)
    if np.any(positions >= len(m1)) or not np.array_equal(m1.asi8[positions], expected):
        raise RuntimeError("UNIFIED_EXIT_PILOT_FIRST_STATE_MISSING")
    stream = np.column_stack(
        [np.arange(len(entry), dtype="<i8"), entry.asi8, positions.astype("<i8"), expected]
    ).astype("<i8", copy=False)
    value = {
        "schema_version": FIRST_STATE_BRIDGE_SCHEMA_VERSION,
        "decision": "PASS",
        "split": split,
        "entry_row_count": len(entry),
        "first_state_rule": "entry_m5_bar_start_plus_300_seconds_equals_first_exit_m1_bar_start",
        "bridge_stream_sha256": hashlib.sha256(stream.tobytes()).hexdigest(),
        "first_entry_time_utc": entry[0].isoformat(),
        "last_entry_time_utc": entry[-1].isoformat(),
        "first_state_min_utc": pd.Timestamp(expected[0], tz="UTC").isoformat(),
        "first_state_max_utc": pd.Timestamp(expected[-1], tz="UTC").isoformat(),
        "bindings": bindings,
        "entry_representation": ["seq", "snap", "ctx_cont", "ctx_cat"],
        "entry_to_exit_join_cardinality": "one_to_one_exact_clock",
        "normalization_mode": "frozen_train_transform",
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["witness_sha256"] = canonical_sha256(value)
    return value
