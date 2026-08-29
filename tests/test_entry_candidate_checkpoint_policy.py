from __future__ import annotations

import pytest

from gx1.contracts import entry_candidate_checkpoint_policy_v1 as policy


def _record(epoch: int, metric: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "metric": metric,
        "path": f"top_k/epoch_{epoch:04d}.pt",
        "sha256": f"{epoch:064x}",
    }


def test_policy_is_the_frozen_external_candidate_contract() -> None:
    observed = policy.checkpoint_policy_metadata()
    assert observed["max_epochs"] == 30
    assert observed["validation_frequency_epochs"] == 1
    assert observed["early_stop_patience"] == 5
    assert observed["minimum_epochs_before_stop"] == 2
    assert observed["save_top_k"] == 3
    assert observed["checkpoint_mode"] == "max"
    assert policy.require_checkpoint_policy(observed, context="TEST") == observed


def test_early_stop_waits_for_minimum_epochs_and_patience() -> None:
    assert policy.metric_improved(candidate=1.0, best=0.0, min_delta=0.0)
    assert not policy.metric_improved(candidate=1.0, best=1.0, min_delta=0.0)
    assert not policy.should_early_stop(
        completed_epochs=1,
        epochs_since_improve=5,
        patience=5,
        minimum_epochs_before_stop=2,
    )
    assert not policy.should_early_stop(
        completed_epochs=2,
        epochs_since_improve=4,
        patience=5,
        minimum_epochs_before_stop=2,
    )
    assert policy.should_early_stop(
        completed_epochs=2,
        epochs_since_improve=5,
        patience=5,
        minimum_epochs_before_stop=2,
    )


def test_top_k_is_deterministic_and_retains_only_the_three_best() -> None:
    records = [
        _record(1, 1.0),
        _record(2, 5.0),
        _record(3, 3.0),
        _record(4, 5.0),
        _record(5, 2.0),
    ]
    kept = policy.retain_top_k(records, top_k=3)
    assert [(row["epoch"], row["metric"]) for row in kept] == [
        (2, 5.0),
        (4, 5.0),
        (3, 3.0),
    ]


def test_policy_rejects_nonfinite_or_duplicate_checkpoint_records() -> None:
    with pytest.raises(RuntimeError, match="TOP_K_RECORD_INVALID"):
        policy.retain_top_k([_record(1, float("nan"))], top_k=3)
    with pytest.raises(RuntimeError, match="TOP_K_RECORD_INVALID"):
        policy.retain_top_k([_record(1, 1.0), _record(1, 2.0)], top_k=3)
