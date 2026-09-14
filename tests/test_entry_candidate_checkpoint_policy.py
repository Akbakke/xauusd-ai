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
    assert observed["schema_version"] == "gx1_entry_candidate_checkpoint_policy_v3"
    assert observed["max_epochs"] == 30
    assert observed["validation_frequency_epochs"] == 1
    assert observed["early_stop_patience"] == 5
    assert observed["minimum_epochs_before_stop"] == 1
    assert observed["save_top_k"] == 1
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


@pytest.mark.parametrize("marked", [False, True])
def test_native_monitor_is_bound_to_validated_accounting(marked):
    from tests.test_unified_exit_economics_objective_v2 import _contract
    objective = _contract(reward_accounting="liquidation_value_increments_v1" if marked else "terminal_cash_v2")
    monitor = policy.native_checkpoint_monitor(objective)
    assert monitor == (policy.MARKED_NET_CHECKPOINT_MONITOR if marked else policy.COUPLED_NET_CHECKPOINT_MONITOR)
    metadata = policy.checkpoint_policy_metadata(checkpoint_monitor=monitor)
    assert metadata["schema_version"] == (policy.MARKED_NET_SCHEMA_VERSION if marked else policy.COUPLED_NET_SCHEMA_VERSION)
    assert metadata["early_stop_patience"] == 5 and metadata["max_epochs"] == 30
    assert policy.require_checkpoint_policy(metadata, context="TEST", checkpoint_monitor=monitor) == metadata
    objective["contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="OBJECTIVE_CONTRACT_INVALID"):
        policy.native_checkpoint_monitor(objective)


def test_marked_selection_counts_losses_and_rejects_partial_or_fallback_scores():
    monitor = policy.MARKED_NET_CHECKPOINT_MONITOR
    metrics = {"marked_policy_evaluation": {"single_position_replay": {
        "full_cohort_authoritative": True, "net_cash_plus_open_value_bps_sum": -13.5}},
        "entry_exit_policy_metrics": {"mean_net_bps_per_entry": 1000.0}}
    assert policy.checkpoint_metric(metrics, checkpoint_monitor=monitor) == -13.5
    replay = metrics["marked_policy_evaluation"]["single_position_replay"]
    replay["full_cohort_authoritative"] = False
    with pytest.raises(RuntimeError, match="MARKED_CHECKPOINT_INCOMPLETE"):
        policy.checkpoint_metric(metrics, checkpoint_monitor=monitor)
    replay["full_cohort_authoritative"] = True
    for value in [None, True, float("nan"), float("inf")]:
        replay["net_cash_plus_open_value_bps_sum"] = value
        with pytest.raises(RuntimeError, match="CHECKPOINT_METRIC_INVALID"):
            policy.checkpoint_metric(metrics, checkpoint_monitor=monitor)
    del metrics["marked_policy_evaluation"]
    with pytest.raises(RuntimeError, match="MARKED_CHECKPOINT_INCOMPLETE"):
        policy.checkpoint_metric(metrics, checkpoint_monitor=monitor)
