from __future__ import annotations

import pytest

from gx1.contracts.unified_exit_dataset_adapter_v2 import UnifiedExitDatasetAdapterV2
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
    schedule_random_access_entry_anchors,
    schedule_random_access_epoch,
    schedule_random_access_full_population_epoch,
)


def _inputs():
    contract = build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="a" * 64,
        transition_budget_per_epoch=16,
        transitions_per_entry=4,
        entry_pair_population=7,
    )
    return contract, [2, 7, 25, 100, 700, 1000, 9000]


def test_full_epochs_preserve_stream_and_cover_every_entry_once():
    contract, counts = _inputs()
    expected_samples = []
    expected_anchors = []
    for chunk in range(7):
        expected_samples.extend(schedule_random_access_epoch(
            sampler_contract=contract, epoch_index=chunk,
            successor_transition_count_by_entry=counts,
        ))
        expected_anchors.extend(schedule_random_access_entry_anchors(
            sampler_contract=contract, epoch_index=chunk,
        ))
    for epoch in range(4):
        samples, anchors, metadata = schedule_random_access_full_population_epoch(
            sampler_contract=contract, epoch_index=epoch,
            successor_transition_count_by_entry=counts,
        )
        assert list(anchors) == expected_anchors[epoch * 7:(epoch + 1) * 7]
        assert list(samples) == expected_samples[epoch * 28:(epoch + 1) * 28]
        assert len(anchors) == 7
        assert {a["entry_row_index"] for a in anchors} == set(range(7))
        assert metadata["entry_pair_count"] == 7
        assert metadata["transition_count"] == 28
        assert metadata["global_entry_stop"] - metadata["global_entry_start"] == 7
        assert metadata["every_entry_pair_exactly_once"] is True


def test_adapter_exposes_full_order_across_partial_chunk_boundaries():
    contract, counts = _inputs()
    # Exercise the production scheduler API without unrelated feature materialization.
    adapter = UnifiedExitDatasetAdapterV2.__new__(UnifiedExitDatasetAdapterV2)
    adapter._manifest = {"split": "train"}
    adapter._random_access_train = {
        "sampler_contract": contract, "successor_counts": counts,
    }
    metadata = adapter.set_full_population_epoch_index(1)
    _, anchors, expected = schedule_random_access_full_population_epoch(
        sampler_contract=contract, epoch_index=1,
        successor_transition_count_by_entry=counts,
    )
    assert metadata == expected
    assert adapter.random_access_selected_entry_rows_v1() == tuple(
        item["entry_row_index"] for item in anchors
    )
    for anchor in anchors:
        entry = anchor["entry_row_index"]
        samples = adapter._random_access_train["samples_by_entry"][entry]
        assert len(samples) == 4
        assert all(sample["epoch_index"] == anchor["epoch_index"] for sample in samples)
    adapter.set_epoch_index(1)
    assert adapter.random_access_selected_entry_rows_v1() == tuple(
        a["entry_row_index"] for a in schedule_random_access_entry_anchors(
            sampler_contract=contract, epoch_index=1,
        )
    )
    assert "full_population_schedule" not in adapter._random_access_train
    with pytest.raises(RuntimeError, match="FULL_POPULATION_EPOCH_INVALID"):
        adapter.set_full_population_epoch_index(True)
    adapter._manifest = {"split": "val"}
    with pytest.raises(RuntimeError, match="FULL_POPULATION_EPOCH_INVALID"):
        adapter.set_full_population_epoch_index(0)
