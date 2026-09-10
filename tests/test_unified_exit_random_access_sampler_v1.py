from __future__ import annotations

import pytest

from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    DURATION_BUCKETS,
    bucket_coverage_report,
    build_random_access_sampler_contract,
    require_random_access_sample,
    schedule_random_access_epoch,
)
from gx1.scripts.materialize_unified_exit_random_access_sampler_v1 import (
    build_parser,
)


def _contract(*, budget: int = 12, per_entry: int = 3, population: int = 7):
    return build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="a" * 64,
        transition_budget_per_epoch=budget,
        transitions_per_entry=per_entry,
        entry_pair_population=population,
    )


def test_budget_and_k_are_cli_explicit_and_schedule_is_resume_exact() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--split",
                "train",
                "--source-lineage-sha256",
                "a" * 64,
                "--entry-pair-population",
                "7",
                "--output",
                "/tmp/unused.json",
            ]
        )
    contract = _contract()
    counts = [2, 10, 100, 1_000, 10_000, 100_000, 400_000]
    epoch = schedule_random_access_epoch(
        sampler_contract=contract,
        epoch_index=5,
        successor_transition_count_by_entry=counts,
    )
    assert epoch == schedule_random_access_epoch(
        sampler_contract=contract,
        epoch_index=5,
        successor_transition_count_by_entry=counts,
    )
    assert len(epoch) == 12
    assert all(item["successor_state_index"] == item["state_index"] + 1 for item in epoch)
    assert all(item["both_sides_share_timeline"] for item in epoch)
    assert all(item["selection_uses_outcome_values"] is False for item in epoch)
    assert all(item["importance_weight"] == 1.0 for item in epoch)
    assert all(item["sampling_probability"] > 0.0 for item in epoch)


def test_entry_cycle_visits_population_and_tail_bucket_is_reachable() -> None:
    contract = _contract(budget=2, per_entry=1, population=5)
    counts = [400_000] * 5
    selected = []
    seen_buckets = set()
    for epoch_index in range(30):
        samples = schedule_random_access_epoch(
            sampler_contract=contract,
            epoch_index=epoch_index,
            successor_transition_count_by_entry=counts,
        )
        selected.extend(item["entry_row_index"] for item in samples)
        seen_buckets.update(item["duration_bucket_index"] for item in samples)
    assert set(selected[:5]) == set(range(5))
    assert len(DURATION_BUCKETS) - 1 in seen_buckets


def test_report_is_analytic_and_tamper_fails_closed() -> None:
    contract = _contract()
    counts = [2, 10, 100, 1_000, 10_000, 100_000, 400_000]
    samples = schedule_random_access_epoch(
        sampler_contract=contract,
        epoch_index=0,
        successor_transition_count_by_entry=counts,
    )
    report = bucket_coverage_report(
        sampler_contract=contract,
        successor_transition_count_by_entry=counts,
        samples=samples,
    )
    assert report["full_transition_population"] == sum(counts)
    assert report["sampled_transition_budget"] == 12
    assert report["full_population_materialized"] is False
    tampered = dict(samples[0])
    tampered["state_index"] += 1
    with pytest.raises(RuntimeError, match="SAMPLE_INVALID"):
        require_random_access_sample(tampered, sampler_contract=contract)


def test_invalid_budget_relationship_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="BUDGET_INVALID"):
        _contract(budget=10, per_entry=3)
