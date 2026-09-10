from __future__ import annotations

from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)
from gx1.scripts import benchmark_unified_exit_random_access_train_v1 as benchmark


class _Adapter:
    def __init__(self, budget: int) -> None:
        self.contract = build_random_access_sampler_contract(
            split="train",
            source_lineage_sha256="1" * 64,
            transition_budget_per_epoch=budget,
            transitions_per_entry=4,
            entry_pair_population=65_295,
        )

    def random_access_training_bindings_v1(self):
        return {
            "sampler_contract": self.contract,
            "normalization_artifact": {},
            "m1_source_sha256": "2" * 64,
            "market_closure_authority_sha256": "3" * 64,
            "economic_step_manifest_sha256": "4" * 64,
            "economics_objective_contract_sha256": "5" * 64,
        }

    def random_access_selected_entry_rows_v1(self):
        return tuple(range(self.contract["entry_pairs_per_epoch"]))

    def materialize_random_access_training_item_v1(
        self, entry: int, *, outer_batch_index: int
    ):
        return {"entry": entry, "outer": outer_batch_index}


def test_benchmark_sweeps_preregistered_budgets_and_marks_cap(monkeypatch) -> None:
    def fake_collate(items, **_kwargs):
        return {
            "transition_count": len(items) * 4,
            "online_model_inputs": {},
            "target_model_inputs": {},
        }

    monkeypatch.setattr(benchmark, "collate_random_access_training_items", fake_collate)
    receipt = benchmark.benchmark_random_access_train_candidates_v1(
        adapter_factory=_Adapter,
        batch_sizes=(1, 2),
        repeats=1,
        max_selected_entries=2,
    )
    assert receipt["decision"] == "NON_AUTHORITATIVE_CAPPED_SMOKE"
    assert receipt["candidate_budgets"] == [32_768, 65_536, 131_072]
    assert receipt["transitions_per_entry"] == 4
    assert all(not item["full_budget_measured"] for item in receipt["candidates"])
    assert all(
        [row["batch_size"] for row in item["batch_size_sweep"]] == [1, 2]
        for item in receipt["candidates"]
    )
