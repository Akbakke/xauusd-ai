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
    assert observed["schema_version"] == "gx1_entry_candidate_checkpoint_policy_v6"
    assert observed["constant_action_checkpoint_eligible"] is False
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


def test_constant_action_validation_is_never_selectable() -> None:
    assert policy.constant_action_checkpoint_ineligible([0, 0, 256])
    assert policy.constant_action_checkpoint_ineligible([256, 0, 0])
    assert policy.constant_action_checkpoint_ineligible([0, 256, 0])
    assert not policy.constant_action_checkpoint_ineligible([1, 0, 255])
    assert not policy.constant_action_checkpoint_ineligible([100, 100, 56])
    assert policy.validation_entry_action_counts(
        {"entry_exit_policy_metrics": {"long_count": 3, "short_count": 0, "flat_count": 9}}
    ) == [3, 0, 9]
    assert policy.validation_entry_action_counts({"entry_action_counts": [1, 2, 3]}) == [1, 2, 3]
    for bad in (
        {},
        {"entry_action_counts": [0, 0, 0]},
        {"entry_action_counts": [1, 2]},
        {"entry_action_counts": [1, -1, 2]},
        {"entry_action_counts": [1.0, 2.0, 3.0]},
        {"entry_exit_policy_metrics": {"long_count": 1, "short_count": 2}},
    ):
        with pytest.raises(RuntimeError, match="ACTION_COUNTS_MISSING"):
            policy.validation_entry_action_counts(bad)


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


def _m5_checkpoint_prefix(tmp_path):
    from tests.test_entry_observed_market_horizons_learning import _single_epoch_design
    from tests.test_native_prefix_recipe import _write
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    design, witness = _single_epoch_design(tmp_path)
    single = require_m5_single_epoch_population(design)
    return {"schema_version":"gx1_candidate_prefix_training_binding_v1",
        "artifacts":{"design":_write(tmp_path/"checkpoint-design.json",design)},
        "entry_m5_path_single_epoch":{"population_witness":single["binding"],
            "train_rows":single["train_rows"],"last_batch_rows":single["last_batch_rows"],"drop_last":False},
        "train_rows":single["train_rows"],"maximum_optimizer_steps":single["optimizer_steps"],
        "maximum_completed_epochs":1,"physical_sources":{"train":{"parquet":witness["train_parquet"]}},
        "entry_observed_market":{role:{"identity":{"training_phase":witness["training_phase"]}}
            for role in ("train","control")}}


def test_m5_single_epoch_checkpoint_policy_requires_explicit_bound_prefix_and_keeps_legacy(tmp_path):
    prefix = _m5_checkpoint_prefix(tmp_path)
    monitor = policy.MARKED_NET_CHECKPOINT_MONITOR
    legacy = policy.checkpoint_policy_metadata(checkpoint_monitor=monitor)
    actual = policy.checkpoint_policy_metadata(checkpoint_monitor=monitor, prefix_training_binding=prefix)
    assert actual["schema_version"] == policy.M5_SINGLE_EPOCH_SCHEMA_VERSION
    assert actual["max_epochs"] == 1 and legacy["max_epochs"] == policy.MAX_EPOCHS == 30
    assert actual["entry_m5_path_single_epoch"]["optimizer_steps"] == prefix["maximum_optimizer_steps"]
    assert policy.require_checkpoint_policy(actual,context="TEST",checkpoint_monitor=monitor,
        prefix_training_binding=prefix) == actual
    for changed, bound in ((actual,None),({**legacy,"max_epochs":1},None),
            (legacy,prefix),({**actual,"max_epochs":2},prefix),({**actual,"max_epochs":True},prefix)):
        with pytest.raises(RuntimeError):
            policy.require_checkpoint_policy(changed,context="TEST",checkpoint_monitor=monitor,
                prefix_training_binding=bound)
    old_prefix = {k:v for k,v in prefix.items() if k != "entry_m5_path_single_epoch"}
    assert policy.checkpoint_policy_metadata(checkpoint_monitor=monitor,prefix_training_binding=old_prefix) == legacy
    with pytest.raises(RuntimeError,match="M5_EPOCH_MONITOR"):
        policy.checkpoint_policy_metadata(prefix_training_binding=prefix)


@pytest.mark.parametrize("fault",["steps","rows","epoch","witness","phase","source"])
def test_m5_checkpoint_policy_rejects_changed_scope_or_checkpoint_identity(tmp_path,fault):
    prefix = _m5_checkpoint_prefix(tmp_path)
    if fault == "steps":prefix["maximum_optimizer_steps"] += 1
    elif fault == "rows":prefix["train_rows"] += 1
    elif fault == "epoch":prefix["maximum_completed_epochs"] = 2
    elif fault == "witness":prefix["entry_m5_path_single_epoch"]["population_witness"] = {"path":"old","sha256":"0"*64}
    elif fault == "phase":prefix["entry_observed_market"]["train"]["identity"]["training_phase"] = "entry_only_observed_horizon_20d"
    else:prefix["physical_sources"]["train"]["parquet"] = {"path":"old","sha256":"0"*64}
    with pytest.raises(RuntimeError,match="M5_EPOCH_BINDING"):
        policy.checkpoint_policy_metadata(checkpoint_monitor=policy.MARKED_NET_CHECKPOINT_MONITOR,
            prefix_training_binding=prefix)


def _dynamic_m5_checkpoint_prefix(tmp_path):
    from tests.test_native_prefix_recipe import _dynamic_single_epoch_design, _write
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    design, witness = _dynamic_single_epoch_design(tmp_path, rows=17)
    single = require_m5_single_epoch_population(design)
    binding = _write(tmp_path / "dynamic-checkpoint-design.json", design)
    identity = {"training_phase": witness["training_phase"], "design": binding,
                "target_contract": witness["target_contract"]}
    return {"schema_version": "gx1_candidate_prefix_training_binding_v1",
        "entry_training_phase": witness["training_phase"], "fixed_final_model_variant": "ONLINE",
        "artifacts": {"design": binding},
        "entry_m5_path_single_epoch": {"population_witness": single["binding"],
            "train_rows": single["train_rows"], "last_batch_rows": single["last_batch_rows"], "drop_last": False},
        "train_rows": single["train_rows"], "maximum_optimizer_steps": single["optimizer_steps"],
        "maximum_completed_epochs": 1,
        "physical_sources": {"train": {"parquet": single["train_parquet"], "manifest": single["train_manifest"]}},
        "entry_observed_market": {role: {"identity": dict(identity)} for role in ("train", "control")}}


def test_dynamic_m5_fixed_online_checkpoint_policy_has_no_validation_or_economic_selection(tmp_path):
    prefix = _dynamic_m5_checkpoint_prefix(tmp_path)
    monitor = policy.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    actual = policy.checkpoint_policy_metadata(checkpoint_monitor=monitor, prefix_training_binding=prefix)
    assert actual == {
        "schema_version": policy.DYNAMIC_M5_FIXED_ONLINE_SCHEMA_VERSION,
        "checkpoint_monitor": monitor, "checkpoint_mode": "fixed_final_online", "max_epochs": 1,
        "entry_training_phase": prefix["entry_training_phase"],
        "entry_m5_path_single_epoch": {**prefix["entry_m5_path_single_epoch"], "optimizer_steps": 2},
        "selected_model_variant": "ONLINE", "selection_boundary": "complete_one_whole_physical_train_epoch",
        "validation_selection": False, "economic_selection": False,
        "early_stopping": False, "save_top_k": 0, "native_training_admitted": False}
    assert policy.require_checkpoint_policy(actual, context="UNIT", checkpoint_monitor=monitor,
        prefix_training_binding=prefix) == actual
    for field, value in (("max_epochs", 2), ("selected_model_variant", "EMA"),
                         ("validation_selection", True), ("save_top_k", 1)):
        with pytest.raises(RuntimeError, match="CHECKPOINT_POLICY_INVALID"):
            policy.require_checkpoint_policy({**actual, field: value}, context="UNIT", checkpoint_monitor=monitor,
                prefix_training_binding=prefix)
    for metrics in ({}, {monitor: 1.0}, {policy.CHECKPOINT_MONITOR: 99.0},
                    {"entry_exit_policy_metrics": {"mean_net_bps_per_entry": 1000.0}}):
        with pytest.raises(RuntimeError, match="FIXED_FINAL_NO_METRIC"):
            policy.checkpoint_metric(metrics, checkpoint_monitor=monitor)


@pytest.mark.parametrize("monitor", [policy.CHECKPOINT_MONITOR, policy.COUPLED_NET_CHECKPOINT_MONITOR,
                                    policy.MARKED_NET_CHECKPOINT_MONITOR])
def test_dynamic_m5_prefix_rejects_legacy_checkpoint_monitors(tmp_path, monitor):
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_FIXED_MONITOR_REQUIRED"):
        policy.checkpoint_policy_metadata(checkpoint_monitor=monitor,
            prefix_training_binding=_dynamic_m5_checkpoint_prefix(tmp_path))


@pytest.mark.parametrize("fault", ["no_prefix", "legacy", "no_witness", "phase", "manifest", "design", "target", "ema"])
def test_dynamic_fixed_monitor_requires_actual_dynamic_prefix_and_whole_train_witness(tmp_path, fault):
    prefix = _dynamic_m5_checkpoint_prefix(tmp_path)
    if fault == "no_prefix": prefix = None
    elif fault == "legacy":
        prefix = _m5_checkpoint_prefix(tmp_path)
    elif fault == "no_witness": prefix.pop("entry_m5_path_single_epoch")
    elif fault == "phase": prefix["entry_training_phase"] = "entry_only_observed_m5_path"
    elif fault == "manifest": prefix["physical_sources"]["train"]["manifest"] = {"path": "old", "sha256": "0" * 64}
    elif fault == "design": prefix["entry_observed_market"]["control"]["identity"]["design"] = {"path": "old", "sha256": "0" * 64}
    elif fault == "target": prefix["entry_observed_market"]["train"]["identity"]["target_contract"] = {}
    elif fault == "ema": prefix["fixed_final_model_variant"] = "EMA"
    with pytest.raises(RuntimeError, match="DYNAMIC_M5"):
        policy.checkpoint_policy_metadata(checkpoint_monitor=policy.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
            prefix_training_binding=prefix)
