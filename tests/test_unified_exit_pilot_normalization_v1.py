from __future__ import annotations

import copy

import numpy as np
import pytest

from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    LIFETIME_SUMMARY_FIELD_ORDER,
    lifetime_summary_registry,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    BENCHMARK_BUDGETS,
    build_first_state_entry_bridge_witness,
    build_physical_summary_sample_authority,
    build_sampler_benchmark_candidate_set,
    fit_lifetime_summary_normalization,
    iter_physical_summary_samples,
    require_lifetime_summary_normalization,
)


def test_benchmark_candidates_are_explicit_and_unselected() -> None:
    value = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="a" * 64,
        entry_pair_population=65_295,
    )
    assert value["decision"] == "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK"
    assert [
        item["sampler_contract"]["transition_budget_per_epoch"]
        for item in value["candidates"]
    ] == list(BENCHMARK_BUDGETS)
    assert all(item["selected"] is False for item in value["candidates"])
    assert all(
        item["sampler_contract"]["transitions_per_entry"] == 4
        for item in value["candidates"]
    )
    assert value["selected_sampler_contract_sha256"] is None
    assert value["selection_uses_outcome_values"] is False


def test_physical_summary_authority_is_deterministic_and_outcome_blind() -> None:
    counts = [1, 5, 70, 300_000]
    left = list(
        iter_physical_summary_samples(
            successor_transition_count_by_entry=counts,
            source_lineage_sha256="b" * 64,
        )
    )
    right = list(
        iter_physical_summary_samples(
            successor_transition_count_by_entry=counts,
            source_lineage_sha256="b" * 64,
        )
    )
    assert left == right
    assert all(item["selection_uses_outcome_values"] is False for item in left)
    assert all(set(item).isdisjoint({"reward", "pnl", "q", "price"}) for item in left)
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=counts,
        source_lineage_sha256="b" * 64,
    )
    assert authority["sample_count"] == len(left)
    assert authority["fit_row_count"] == len(left) * 2
    assert authority["epoch_sampler_independent"] is True
    assert authority["val_fit_rows"] == authority["test_fit_rows"] == 0


def test_lifetime_summary_fit_is_train_only_and_registry_bound() -> None:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[2, 20, 300],
        source_lineage_sha256="c" * 64,
    )
    rows = authority["fit_row_count"]
    base = np.arange(rows, dtype=np.float64)[:, None]
    values = np.concatenate(
        [np.log1p(base + offset + 1.0) for offset in range(7)], axis=1
    )
    fitted = fit_lifetime_summary_normalization(
        values=values,
        sample_authority=authority,
    )
    registry = lifetime_summary_registry()
    assert fitted["field_order"] == list(LIFETIME_SUMMARY_FIELD_ORDER)
    assert fitted["lifetime_summary_registry_sha256"] == registry["registry_sha256"]
    assert fitted["train_fit_rows"] == rows
    assert fitted["val_fit_rows"] == fitted["test_fit_rows"] == 0
    assert fitted["val_mode"] == "apply_frozen_train_transform_only"
    assert fitted["test_accessed"] is False


def test_lifetime_summary_normalization_validator_rejects_mutation() -> None:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[2, 20, 300],
        source_lineage_sha256="9" * 64,
    )
    rows = authority["fit_row_count"]
    values = np.arange(rows * 7, dtype=np.float64).reshape(rows, 7)
    fitted = fit_lifetime_summary_normalization(
        values=values,
        sample_authority=authority,
    )
    assert require_lifetime_summary_normalization(
        fitted,
        expected_sample_authority_sha256=authority["authority_sha256"],
    ) == fitted
    bad = copy.deepcopy(fitted)
    bad["surface"]["center"][0] += 1.0
    with pytest.raises(RuntimeError, match="SUMMARY_NORMALIZATION_INVALID"):
        require_lifetime_summary_normalization(bad)


def test_lifetime_summary_fit_rejects_mutated_authority() -> None:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[10],
        source_lineage_sha256="d" * 64,
    )
    bad = copy.deepcopy(authority)
    bad["selection_uses_outcome_values"] = True
    with pytest.raises(RuntimeError, match="SUMMARY_AUTHORITY_INVALID"):
        fit_lifetime_summary_normalization(
            values=np.ones((authority["fit_row_count"], 7)),
            sample_authority=bad,
        )


def test_first_state_bridge_binds_exact_clock_and_sources() -> None:
    m1 = np.arange(
        np.datetime64("2025-06-01T00:00"),
        np.datetime64("2025-06-01T01:00"),
        np.timedelta64(1, "m"),
    )
    entry = m1[[0, 5, 10]]
    kwargs = {
        "split": "train",
        "entry_times": entry,
        "m1_times": m1,
        "child_admission_sha256": "1" * 64,
        "child_parquet_sha256": "2" * 64,
        "entry_sequence_audit_sha256": "3" * 64,
        "m1_source_sha256": "4" * 64,
        "closure_authority_sha256": "5" * 64,
        "state_view_source_sha256": "6" * 64,
        "lifetime_summary_registry_sha256": "7" * 64,
        "train_normalization_sha256": "8" * 64,
        "m1_bid_open": np.linspace(100.0, 101.0, len(m1)),
        "m1_ask_open": np.linspace(100.1, 101.1, len(m1)),
    }
    witness = build_first_state_entry_bridge_witness(**kwargs)
    assert witness["entry_row_count"] == 3
    assert witness["entry_to_exit_join_cardinality"] == "one_to_one_exact_clock"
    assert witness["test_accessed"] is False
    assert witness["bindings"]["child_parquet"] == "2" * 64

    equal_ask = kwargs["m1_ask_open"].copy()
    equal_ask[5] = kwargs["m1_bid_open"][5]
    zero_spread = build_first_state_entry_bridge_witness(
        **{**kwargs, "m1_ask_open": equal_ask})
    assert zero_spread["entry_row_count"] == witness["entry_row_count"]
    assert zero_spread["entry_fill_binding_stream_sha256"] != witness["entry_fill_binding_stream_sha256"]

    broken = np.delete(m1, 5)
    with pytest.raises(RuntimeError, match="FIRST_STATE_MISSING"):
        build_first_state_entry_bridge_witness(**{**kwargs, "m1_times": broken})


def test_first_state_bridge_rejects_test_split() -> None:
    with pytest.raises(RuntimeError, match="BRIDGE_SPLIT_INVALID"):
        build_first_state_entry_bridge_witness(
            split="test",
            entry_times=["2025-01-01T00:00Z"],
            m1_times=["2025-01-01T00:05Z"],
            child_admission_sha256="1" * 64,
            child_parquet_sha256="2" * 64,
            entry_sequence_audit_sha256="3" * 64,
            m1_source_sha256="4" * 64,
            closure_authority_sha256="5" * 64,
            state_view_source_sha256="6" * 64,
            lifetime_summary_registry_sha256="7" * 64,
            train_normalization_sha256="8" * 64,
            m1_bid_open=[100.0],
            m1_ask_open=[100.1],
        )


@pytest.mark.parametrize("limits,expected_sha", [
    (None, "4f8fdd10d89bfffe11a629a5414994a6b7ff1ea87f36937025bcb43e2b6e0ff1"),
    ([1, 3, 20, 100], "225c07cdcb69f888e85a3ec780dce1a75b353d4d562d474a2c39be56194c692f"),
])
def test_streaming_authority_preserves_prechange_contract(limits, expected_sha):
    # Captured from the tuple-based owner, including stream order and all fields.
    value = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[1, 5, 70, 300_000],
        source_lineage_sha256="b" * 64,
        fit_state_stop_exclusive_by_entry=limits,
    )
    assert value["authority_sha256"] == expected_sha


def test_authority_does_not_retain_the_sample_population(monkeypatch):
    import weakref
    from gx1.contracts import unified_exit_pilot_normalization_v1 as owner

    class Sample(dict):
        pass

    previous = []
    def samples(**kwargs):
        for _ in range(100):
            # A for-loop may still hold its last sample when it asks for the next.
            assert sum(ref() is not None for ref in previous) <= 1
            sample = Sample(sample_sha256="a" * 64)
            previous.append(weakref.ref(sample))
            yield sample
            del sample

    monkeypatch.setattr(owner, "iter_physical_summary_samples", samples)
    value = owner.build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[100], source_lineage_sha256="b" * 64)
    assert value["sample_count"] == 100
    assert not any(ref() is not None for ref in previous)
