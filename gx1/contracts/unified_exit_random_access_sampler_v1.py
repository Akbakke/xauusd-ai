"""Outcome-blind, bounded transition sampling for unbounded Exit lifecycles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


RANDOM_ACCESS_SAMPLER_SCHEMA_VERSION = "gx1_unified_exit_random_access_sampler_v1"
RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION = "gx1_unified_exit_transition_sample_v1"
RANDOM_ACCESS_ANCHOR_SCHEMA_VERSION = "gx1_unified_exit_entry_anchor_sample_v1"
DURATION_BUCKETS: tuple[tuple[int, int | None], ...] = (
    (0, 1),
    (1, 4),
    (4, 16),
    (16, 64),
    (64, 256),
    (256, 1_024),
    (1_024, 4_096),
    (4_096, 16_384),
    (16_384, 65_536),
    (65_536, 262_144),
    (262_144, None),
)


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
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_{label}_SHA_INVALID")
    return value


def build_random_access_sampler_contract(
    *,
    split: str,
    source_lineage_sha256: str,
    transition_budget_per_epoch: int,
    transitions_per_entry: int,
    entry_pair_population: int,
) -> dict[str, Any]:
    """Seal an explicit bounded budget without selecting a large default."""

    lineage = _require_sha(source_lineage_sha256, "LINEAGE")
    integers = (
        transition_budget_per_epoch,
        transitions_per_entry,
        entry_pair_population,
    )
    if (
        split not in {"train", "val"}
        or any(isinstance(item, bool) or not isinstance(item, int) for item in integers)
        or transition_budget_per_epoch < 1
        or transitions_per_entry < 1
        or entry_pair_population < 1
        or transition_budget_per_epoch % transitions_per_entry != 0
        or transition_budget_per_epoch // transitions_per_entry
        > entry_pair_population
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BUDGET_INVALID")
    value = {
        "schema_version": RANDOM_ACCESS_SAMPLER_SCHEMA_VERSION,
        "decision": "PASS",
        "split": split,
        "source_lineage_sha256": lineage,
        "transition_budget_per_epoch": transition_budget_per_epoch,
        "transitions_per_entry": transitions_per_entry,
        "entry_pairs_per_epoch": transition_budget_per_epoch
        // transitions_per_entry,
        "entry_pair_population": entry_pair_population,
        "duration_buckets": [
            {"bucket_index": index, "start_inclusive": start, "end_exclusive": end}
            for index, (start, end) in enumerate(DURATION_BUCKETS)
        ],
        "entry_schedule": "outcome_blind_affine_population_cycle_v1",
        "bucket_schedule": "outcome_blind_cyclic_permutation_v1",
        "within_bucket_schedule": "outcome_blind_affine_cycle_v1",
        "sampling_target": "entry_pair_uniform_duration_bucket_uniform_v1",
        "entry_anchor_policy": "one_state_zero_no_loss_view_per_selected_entry_v1",
        "entry_anchor_views_per_selected_entry": 1,
        "anchors_excluded_from_transition_budget": True,
        "common_timeline_both_sides": True,
        "forbidden_seed_fields": ["price", "reward", "label", "pnl", "side"],
        "test_data_used": False,
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def require_random_access_sampler_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "contract_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_CONTRACT_INVALID")
    observed = dict(value)
    claimed = observed.pop("contract_sha256")
    rebuilt = build_random_access_sampler_contract(
        split=observed.get("split"),
        source_lineage_sha256=observed.get("source_lineage_sha256"),
        transition_budget_per_epoch=observed.get("transition_budget_per_epoch"),
        transitions_per_entry=observed.get("transitions_per_entry"),
        entry_pair_population=observed.get("entry_pair_population"),
    )
    if value != rebuilt or claimed != rebuilt["contract_sha256"]:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_CONTRACT_INVALID")
    return rebuilt


def _affine_permutation(*, count: int, seed_sha256: str) -> tuple[int, int]:
    seed = bytes.fromhex(seed_sha256)
    multiplier = int.from_bytes(seed[:16], "big") % count
    while math.gcd(multiplier, count) != 1:
        multiplier = (multiplier + 1) % count
    offset = int.from_bytes(seed[16:], "big") % count
    return multiplier, offset


def _seed(contract: Mapping[str, Any], *, epoch_index: int, scope: str, slot: int) -> str:
    return canonical_sha256(
        {
            "sampler_contract_sha256": contract["contract_sha256"],
            "source_lineage_sha256": contract["source_lineage_sha256"],
            "split": contract["split"],
            "epoch_index": epoch_index,
            "scope": scope,
            "slot": slot,
        }
    )


def duration_bucket_for_state(state_index: int) -> int:
    if isinstance(state_index, bool) or not isinstance(state_index, int) or state_index < 0:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_STATE_INDEX_INVALID")
    for bucket, (start, end) in enumerate(DURATION_BUCKETS):
        if state_index >= start and (end is None or state_index < end):
            return bucket
    raise AssertionError("duration buckets must cover all non-negative indices")


def schedule_random_access_epoch(
    *,
    sampler_contract: Mapping[str, Any],
    epoch_index: int,
    successor_transition_count_by_entry: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Choose exactly the declared number of causal t->t+1 transitions."""

    contract = require_random_access_sampler_contract(sampler_contract)
    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_EPOCH_INVALID")
    counts = tuple(successor_transition_count_by_entry)
    if (
        len(counts) != contract["entry_pair_population"]
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_POPULATION_INVALID")
    population = len(counts)
    selected_count = contract["entry_pairs_per_epoch"]
    # Keep the entry permutation fixed and advance a global cursor.  This makes
    # resumption exact and visits the full physical Entry-pair population before
    # repeating, even when one epoch intentionally selects only a prefix.
    entry_seed = _seed(contract, epoch_index=0, scope="entry", slot=0)
    entry_a, entry_b = _affine_permutation(count=population, seed_sha256=entry_seed)
    selected_entries = tuple(
        (entry_a * (epoch_index * selected_count + slot) + entry_b) % population
        for slot in range(selected_count)
    )
    samples: list[dict[str, Any]] = []
    for entry_slot, entry_row_index in enumerate(selected_entries):
        transition_count = counts[entry_row_index]
        eligible = [
            index
            for index, (start, _end) in enumerate(DURATION_BUCKETS)
            if start < transition_count
        ]
        bucket_seed = _seed(
            contract,
            epoch_index=epoch_index,
            scope=f"bucket:{entry_row_index}",
            slot=0,
        )
        bucket_a, bucket_b = _affine_permutation(
            count=len(eligible), seed_sha256=bucket_seed
        )
        for sample_slot in range(contract["transitions_per_entry"]):
            cycle = epoch_index * contract["transitions_per_entry"] + sample_slot
            bucket_index = eligible[(bucket_a * cycle + bucket_b) % len(eligible)]
            start, raw_end = DURATION_BUCKETS[bucket_index]
            stop = min(transition_count, raw_end or transition_count)
            bucket_count = stop - start
            within_seed = _seed(
                contract,
                epoch_index=0,
                scope=f"within:{entry_row_index}:{bucket_index}",
                slot=0,
            )
            within_a, within_b = _affine_permutation(
                count=bucket_count, seed_sha256=within_seed
            )
            visit = epoch_index * contract["transitions_per_entry"] + sample_slot
            state_index = start + (within_a * visit + within_b) % bucket_count
            sample = {
                "schema_version": RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION,
                "sampler_contract_sha256": contract["contract_sha256"],
                "epoch_index": epoch_index,
                "entry_slot": entry_slot,
                "entry_row_index": entry_row_index,
                "sample_slot": sample_slot,
                "duration_bucket_index": bucket_index,
                "duration_bucket_start_inclusive": start,
                "duration_bucket_stop_exclusive": stop,
                "eligible_transition_count_in_bucket": bucket_count,
                "eligible_duration_bucket_count": len(eligible),
                "state_index": state_index,
                "successor_state_index": state_index + 1,
                "sampling_probability": (
                    1.0 / population / len(eligible) / bucket_count
                ),
                "importance_weight": 1.0,
                "sample_role": "bellman_transition",
                "both_sides_share_timeline": True,
                "selection_uses_outcome_values": False,
            }
            sample["sample_sha256"] = canonical_sha256(sample)
            samples.append(sample)
    if len(samples) != contract["transition_budget_per_epoch"]:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BUDGET_DRIFT")
    return tuple(samples)


def schedule_random_access_entry_anchors(
    *, sampler_contract: Mapping[str, Any], epoch_index: int
) -> tuple[dict[str, Any], ...]:
    """Return mandatory state-zero bridge views outside the transition budget."""

    contract = require_random_access_sampler_contract(sampler_contract)
    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_EPOCH_INVALID")
    population = contract["entry_pair_population"]
    selected_count = contract["entry_pairs_per_epoch"]
    entry_seed = _seed(contract, epoch_index=0, scope="entry", slot=0)
    entry_a, entry_b = _affine_permutation(count=population, seed_sha256=entry_seed)
    anchors: list[dict[str, Any]] = []
    for entry_slot in range(selected_count):
        entry_row_index = (
            entry_a * (epoch_index * selected_count + entry_slot) + entry_b
        ) % population
        anchor = {
            "schema_version": RANDOM_ACCESS_ANCHOR_SCHEMA_VERSION,
            "sampler_contract_sha256": contract["contract_sha256"],
            "epoch_index": epoch_index,
            "entry_slot": entry_slot,
            "entry_row_index": entry_row_index,
            "state_index": 0,
            "sample_role": "entry_anchor_no_loss",
            "loss_weight": 0.0,
            "both_sides_share_timeline": True,
            "selection_uses_outcome_values": False,
        }
        anchor["anchor_sha256"] = canonical_sha256(anchor)
        anchors.append(anchor)
    return tuple(anchors)



def schedule_random_access_full_population_epoch(
    *,
    sampler_contract: Mapping[str, Any],
    epoch_index: int,
    successor_transition_count_by_entry: Sequence[int],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Cover each Entry once while preserving the measured sampler stream.

    Legacy epochs are bounded sampler chunks. A full epoch is one
    population-length interval of the same stream, including partial chunks.
    Existing transition/anchor bytes and seeds are retained.
    """
    contract = require_random_access_sampler_contract(sampler_contract)
    if (
        contract["split"] != "train"
        or isinstance(epoch_index, bool)
        or not isinstance(epoch_index, int)
        or epoch_index < 0
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_POPULATION_EPOCH_INVALID")
    population = contract["entry_pair_population"]
    chunk_size = contract["entry_pairs_per_epoch"]
    first = epoch_index * population
    stop = first + population
    samples: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    segments: list[dict[str, int]] = []
    cursor = first
    while cursor < stop:
        chunk_epoch, start_slot = divmod(cursor, chunk_size)
        count = min(chunk_size - start_slot, stop - cursor)
        stop_slot = start_slot + count
        chunk_samples = schedule_random_access_epoch(
            sampler_contract=contract,
            epoch_index=chunk_epoch,
            successor_transition_count_by_entry=successor_transition_count_by_entry,
        )
        chunk_anchors = schedule_random_access_entry_anchors(
            sampler_contract=contract, epoch_index=chunk_epoch
        )
        samples.extend(
            item for item in chunk_samples
            if start_slot <= item["entry_slot"] < stop_slot
        )
        anchors.extend(chunk_anchors[start_slot:stop_slot])
        segments.append({
            "sampler_chunk_index": chunk_epoch,
            "first_entry_slot": start_slot,
            "stop_entry_slot": stop_slot,
            "full_epoch_row_start": cursor - first,
        })
        cursor += count
    order = [item["entry_row_index"] for item in anchors]
    if (
        len(order) != population
        or len(set(order)) != population
        or set(order) != set(range(population))
        or len(samples) != population * contract["transitions_per_entry"]
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_POPULATION_COVERAGE_INVALID")
    metadata = {
        "schema_version": "gx1_unified_exit_full_population_epoch_v1",
        "epoch_index": epoch_index,
        "sampler_contract_sha256": contract["contract_sha256"],
        "entry_pair_count": population,
        "transition_count": len(samples),
        "entry_order_sha256": canonical_sha256(order),
        "global_entry_start": first,
        "global_entry_stop": stop,
        "segments": segments,
        "every_entry_pair_exactly_once": True,
        "transition_sampling_unchanged": True,
        "test_data_used": False,
    }
    metadata["schedule_sha256"] = canonical_sha256(metadata)
    return tuple(samples), tuple(anchors), metadata

def require_random_access_entry_anchor(
    value: Mapping[str, Any], *, sampler_contract: Mapping[str, Any]
) -> dict[str, Any]:
    contract = require_random_access_sampler_contract(sampler_contract)
    if not isinstance(value, Mapping) or "anchor_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ANCHOR_INVALID")
    observed = dict(value)
    claimed = observed.pop("anchor_sha256")
    if (
        set(observed)
        != {
            "schema_version",
            "sampler_contract_sha256",
            "epoch_index",
            "entry_slot",
            "entry_row_index",
            "state_index",
            "sample_role",
            "loss_weight",
            "both_sides_share_timeline",
            "selection_uses_outcome_values",
        }
        or observed["schema_version"] != RANDOM_ACCESS_ANCHOR_SCHEMA_VERSION
        or observed["sampler_contract_sha256"] != contract["contract_sha256"]
        or observed["state_index"] != 0
        or observed["sample_role"] != "entry_anchor_no_loss"
        or observed["loss_weight"] != 0.0
        or observed["both_sides_share_timeline"] is not True
        or observed["selection_uses_outcome_values"] is not False
        or claimed != canonical_sha256(observed)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ANCHOR_INVALID")
    observed["anchor_sha256"] = claimed
    return observed


def require_random_access_sample(
    value: Mapping[str, Any], *, sampler_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one scheduled transition without consulting outcome fields."""

    contract = require_random_access_sampler_contract(sampler_contract)
    if not isinstance(value, Mapping) or "sample_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SAMPLE_INVALID")
    observed = dict(value)
    claimed = observed.pop("sample_sha256")
    expected_keys = {
        "schema_version",
        "sampler_contract_sha256",
        "epoch_index",
        "entry_slot",
        "entry_row_index",
        "sample_slot",
        "duration_bucket_index",
        "duration_bucket_start_inclusive",
        "duration_bucket_stop_exclusive",
        "eligible_transition_count_in_bucket",
        "eligible_duration_bucket_count",
        "state_index",
        "successor_state_index",
        "sampling_probability",
        "importance_weight",
        "sample_role",
        "both_sides_share_timeline",
        "selection_uses_outcome_values",
    }
    bucket = observed.get("duration_bucket_index")
    probability = observed.get("sampling_probability")
    bucket_spec = DURATION_BUCKETS[bucket] if isinstance(bucket, int) and bucket in range(len(DURATION_BUCKETS)) else None
    bucket_start = observed.get("duration_bucket_start_inclusive")
    bucket_stop = observed.get("duration_bucket_stop_exclusive")
    bucket_count = observed.get("eligible_transition_count_in_bucket")
    if (
        set(observed) != expected_keys
        or observed.get("schema_version") != RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION
        or observed.get("sampler_contract_sha256") != contract["contract_sha256"]
        or not isinstance(bucket, int)
        or bucket not in range(len(DURATION_BUCKETS))
        or bucket_spec is None
        or bucket_start != bucket_spec[0]
        or not isinstance(bucket_stop, int)
        or bucket_stop <= bucket_start
        or (bucket_spec[1] is not None and bucket_stop > bucket_spec[1])
        or bucket_count != bucket_stop - bucket_start
        or not bucket_start <= observed.get("state_index", -1) < bucket_stop
        or observed.get("successor_state_index")
        != observed.get("state_index", -2) + 1
        or observed.get("both_sides_share_timeline") is not True
        or observed.get("selection_uses_outcome_values") is not False
        or not isinstance(probability, float)
        or not math.isfinite(probability)
        or probability <= 0.0
        or observed.get("importance_weight") != 1.0
        or observed.get("sample_role") != "bellman_transition"
        or claimed != canonical_sha256(observed)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SAMPLE_INVALID")
    observed["sample_sha256"] = claimed
    return observed


def bucket_coverage_report(
    *,
    sampler_contract: Mapping[str, Any],
    successor_transition_count_by_entry: Sequence[int],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report sampled counts against analytic denominators without expansion."""

    contract = require_random_access_sampler_contract(sampler_contract)
    eligible = [0] * len(DURATION_BUCKETS)
    for count in successor_transition_count_by_entry:
        for bucket, (start, raw_end) in enumerate(DURATION_BUCKETS):
            if count > start:
                eligible[bucket] += min(count, raw_end or count) - start
    sampled = [0] * len(DURATION_BUCKETS)
    unique: list[set[tuple[int, int]]] = [set() for _ in DURATION_BUCKETS]
    for raw in samples:
        sample = require_random_access_sample(raw, sampler_contract=contract)
        bucket = sample["duration_bucket_index"]
        sampled[bucket] += 1
        unique[bucket].add((sample["entry_row_index"], sample["state_index"]))
    report = {
        "schema_version": "gx1_unified_exit_random_access_bucket_coverage_v1",
        "sampler_contract_sha256": contract["contract_sha256"],
        "full_transition_population": sum(eligible),
        "sampled_transition_budget": len(samples),
        "buckets": [
            {
                "bucket_index": bucket,
                "eligible_transitions": eligible[bucket],
                "sampled_transitions": sampled[bucket],
                "unique_sampled_transitions": len(unique[bucket]),
                "coverage_fraction": (
                    len(unique[bucket]) / eligible[bucket] if eligible[bucket] else None
                ),
            }
            for bucket in range(len(DURATION_BUCKETS))
        ],
        "full_population_materialized": False,
        "test_data_used": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


__all__ = (
    "DURATION_BUCKETS",
    "RANDOM_ACCESS_ANCHOR_SCHEMA_VERSION",
    "RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION",
    "RANDOM_ACCESS_SAMPLER_SCHEMA_VERSION",
    "bucket_coverage_report",
    "build_random_access_sampler_contract",
    "duration_bucket_for_state",
    "require_random_access_entry_anchor",
    "require_random_access_sample",
    "require_random_access_sampler_contract",
    "schedule_random_access_epoch",
    "schedule_random_access_entry_anchors",
)
