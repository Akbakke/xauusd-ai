"""The 20-trading-day learning study admission (schema v2): synthetic contracts only."""

import numpy as np
import pytest

from gx1.contracts import entry_observed_market_horizons_v1 as horizons
from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from tests.test_native_learning_calibration_scope import scope  # noqa: F401
from tests.test_native_prefix_recipe import _bind

ELIGIBLE = np.arange(0, 120_000, 2, dtype=np.int64)  # frozen eligible TRAIN rows (complete 20-day outcomes)
POPULATION = np.arange(0, 7_000, dtype=np.int64)       # VAL rows with complete outcomes before the cutoff


@pytest.fixture
def study_v2(scope, tmp_path, monkeypatch):  # noqa: F811
    policy, recipe, write, save = scope
    policy.pop("native_learning_calibration")
    policy.update(full_epoch_training_allowed=False, full_val_allowed=False)
    recipe.pop("candidate_resume_origin")
    recipe.update(chronological_prefix={"fixture": True}, entry_observed_market_horizons={"fixture": "h"},
                  initialization="fresh_existing_model_constructor_no_checkpoint_weights",
                  run_id="ENTRY_D1_HORIZON_FIT_20261012_001", out_bundle_dir=str(tmp_path / "FIT" / "CANDIDATE_BUNDLE"))
    plan = {
        "schema_version": "gx1_entry_learning_study_v2", "study_id": "ENTRY_D1_HORIZON_LEARNING_STUDY_20261012",
        "root": str(tmp_path), "files": recipe["files"], "chronological_prefix": recipe["chronological_prefix"],
        "entry_observed_market_horizons": recipe["entry_observed_market_horizons"],
        "phases": {
            "fit": {"run_id": "ENTRY_D1_HORIZON_FIT_20261012_001", "optimizer_steps": 1024,
                    "train_observation_steps": [0, 64, 256, 1024], "control_observation_steps": [],
                    "max_invocations": 1, "unique_train_rows": 256},
            "curve": {"run_id": "ENTRY_D1_HORIZON_CURVE_20261012_001", "optimizer_steps": 4096,
                      "train_observation_steps": [0, 256, 1024, 4096], "control_observation_steps": [0, 4096],
                      "max_invocations": 1, "unique_train_rows": 65536}},
        "execution": {"test_data_used": False, "broker_access": False, "spending": False, "horizon_changes": False,
                      "full_epoch_training_allowed": False, "full_val_allowed": False,
                      "exit_training_authorized_without_entry_qualification": False},
        "model": {"unchanged_architecture": True, "signal_fields": 315, "context_fields": 132, "feature_families": 8,
                  "context_families": 9, "fresh_initialization_each_phase": True, "optimizer_and_loss_unchanged": True,
                  "seed": 20260911, "loss_horizon_trading_days": 20},
        "row_bindings": {},
    }
    order = ELIGIBLE[::-1].copy()
    arrays = {"epoch_order": order, "fit": order[:256], "train_probe": np.sort(order[:4096]),
              "control_population": POPULATION,
              "control": POPULATION[np.linspace(0, len(POPULATION) - 1, 4096, dtype=np.int64)]}
    for name, a in arrays.items():
        p = tmp_path / (name + ".npy")
        np.save(p, a)
        plan["row_bindings"][name] = _bind(p)
    # the real prefix result carries the eligible rows inside the coordinates block (require_physical_native_training_coordinates)
    monkeypatch.setattr(native, "require_chronological_prefix_recipe", lambda _: {
        "physical_coordinates": {"eligible_parent_rows": ELIGIBLE,
                                 "artifact": {"bindings": {"TRAIN_NATIVE_EPOCH0_ORDER": plan["row_bindings"]["epoch_order"]}}}})
    monkeypatch.setattr(horizons, "require_entry_observed_market_horizons_scope", lambda _: {
        "decision_horizon_trading_days": 20,
        "physical_sources": {"train": {"physical_rows": 120_000}, "val": {"physical_rows": 8192}},
        "entry_eligibility_cutoff_utc": {"train": "2025-05-02T00:00:00Z", "val": "2026-06-02T00:00:00Z"}})
    monkeypatch.setattr(native, "_complete_horizon_control_rows", lambda _: POPULATION)

    def seal():
        recipe["entry_learning_study"] = {"plan": write("plan.json", plan), "phase": "fit"}
        policy["entry_learning_study_run"] = {
            "selection": recipe["entry_learning_study"], "run_id": recipe["run_id"],
            "out_bundle_dir": recipe["out_bundle_dir"], "source_bindings_sha256": recipe["source_bindings_sha256"],
            "optimizer_steps": 1024, "max_invocations": 1, "test_data_used": False}
        save()
    seal()
    return policy, recipe, plan, seal, save, tmp_path


def test_v2_study_admits_the_bounded_horizon_phases_on_the_frozen_eligible_rows(study_v2):
    policy, recipe, plan, seal, save, tmp_path = study_v2
    study = native.require_entry_learning_study_run(recipe)
    assert study["phase"] == "fit" and study["phase_spec"]["optimizer_steps"] == 1024
    assert study["observed_scope"]["decision_horizon_trading_days"] == 20
    assert np.array_equal(np.sort(study["rows"]["epoch_order"]), ELIGIBLE)
    assert len(study["rows"]["control"]) == 4096 and np.isin(study["rows"]["control"], POPULATION).all()
    # curve phase: 4096 steps, one invocation, control observed at 0 and 4096
    recipe["run_id"] = "ENTRY_D1_HORIZON_CURVE_20261012_001"
    recipe["out_bundle_dir"] = str(tmp_path / "CURVE" / "CANDIDATE_BUNDLE")
    recipe["entry_learning_study"] = {"plan": recipe["entry_learning_study"]["plan"], "phase": "curve"}
    policy["entry_learning_study_run"] = {**policy["entry_learning_study_run"], "selection": recipe["entry_learning_study"],
                                          "run_id": recipe["run_id"], "out_bundle_dir": recipe["out_bundle_dir"], "optimizer_steps": 4096}
    save()
    curve = native.require_entry_learning_study_run(recipe, execution_budget={
        "stop_after_optimizer_steps": 4096, "stop_after_completed_val_epochs": None, "max_invocation_seconds": 12000})
    assert curve["phase_spec"]["control_observation_steps"] == [0, 4096] and curve["phase_spec"]["max_invocations"] == 1


@pytest.mark.parametrize("fault", ["both_designs", "model_surface", "control_outside_population", "order_not_eligible",
                                   "long_fit", "v1_schema_with_horizons"])
def test_v2_study_fails_closed(study_v2, fault):
    policy, recipe, plan, seal, save, tmp_path = study_v2
    if fault == "both_designs":
        recipe["entry_observed_market"] = {"fixture": "v1"}
        save()
    elif fault == "model_surface":
        plan["model"]["signal_fields"] = 254
        seal()
    elif fault == "control_outside_population":
        p = tmp_path / "control.npy"
        np.save(p, np.linspace(0, 8191, 4096, dtype=np.int64))
        plan["row_bindings"]["control"] = _bind(p)
        seal()
    elif fault == "order_not_eligible":
        p = tmp_path / "epoch_order.npy"
        np.save(p, np.arange(120_000, dtype=np.int64)[::-1].copy())
        plan["row_bindings"]["epoch_order"] = _bind(p)
        seal()
    elif fault == "long_fit":
        plan["phases"]["fit"]["optimizer_steps"] = 2048
        seal()
    else:
        plan["schema_version"] = "gx1_entry_learning_study_v1"
        seal()
    with pytest.raises(RuntimeError, match="ENTRY_STUDY"):
        native.require_entry_learning_study_run(recipe)


def test_action_references_report_long_short_flat_per_month():
    import pandas as pd
    from gx1.scripts.research_ta_campaign_v1 import entry_learning_action_references
    n = 10
    y = np.column_stack([np.linspace(-5, 5, n), np.linspace(5, -5, n), np.zeros(n)])
    q = np.column_stack([np.where(np.arange(n) % 2 == 0, 3.0, -1.0), np.where(np.arange(n) % 2 == 0, -1.0, 3.0), np.zeros(n)])
    time_ns = (pd.date_range("2026-01-01", periods=n, freq="7D", tz="UTC").asi8)
    r = entry_learning_action_references(q, y, time_ns)
    assert r["overall"]["always_long_mean_net_bps"] == pytest.approx(0.0) and r["overall"]["action_counts"] == [5, 5, 0]
    assert set(r["monthly"]) == {"2026-01", "2026-02", "2026-03"} and r["decisional"] is False
    assert 0.0 <= r["overall"]["non_flat_hit_rate"] <= 1.0
    with pytest.raises(RuntimeError, match="ARRAY_INVALID"):
        entry_learning_action_references(q[:, :2], y, time_ns)


def test_horizon_references_reject_censored_rows_and_match_cost_arithmetic(tmp_path, monkeypatch):
    import torch
    from gx1.scripts import research_ta_campaign_v1 as owner
    from tests.test_entry_observed_market_horizons_learning import dataset_scope
    from tests.test_unified_exit_prospective_cost_policy_v1 import END, START, _build, _load
    from gx1.contracts import entry_observed_market_v1 as observed
    authority = _load(_build(tmp_path), "parameter_authority")
    cost_terms = observed.entry_observed_market_economics(authority, coverage_start_utc=START, coverage_end_utc=END, verify_local_sources=False)
    dataset, hscope, frame = dataset_scope(tmp_path, cost_terms, rows=12, censor_row=5, cutoff_row=11)
    hscope["physical_sources"] = {"train": hscope["physical_sources"]["train"], "val": hscope["physical_sources"]["train"]}
    monkeypatch.setattr(horizons, "require_observed_entry_scope", lambda _: hscope)
    refs = owner.entry_learning_horizon_references({"fixture": True}, rows_by_role={"fit": np.array([0, 1, 2, 3], dtype=np.int64)})
    assert set(refs["targets"]["fit"]) == {1, 5, 20} and refs["targets"]["fit"][20].shape == (4, 3)
    gross = {h: torch.from_numpy(frame.iloc[:4][list(horizons.GROSS_TARGET_COLUMNS_BY_HORIZON[h])].to_numpy(dtype=np.float32)) for h in (1, 5, 20)}
    elapsed = {h: torch.from_numpy(frame.iloc[:4][horizons.ELAPSED_COLUMN_BY_HORIZON[h]].to_numpy(dtype=np.float64).astype(np.int64)) for h in (1, 5, 20)}
    expected = horizons.build_entry_observed_market_horizon_targets(gross_by_horizon=gross, elapsed_by_horizon=elapsed, economics=cost_terms)
    for h in (1, 5, 20):
        np.testing.assert_array_equal(refs["targets"]["fit"][h], expected[h]["target_bps"].numpy())
    assert refs["constant_bps_by_horizon"]["fit"]["20"] == pytest.approx(expected[20]["target_bps"].numpy().mean(axis=0).tolist())
    with pytest.raises(RuntimeError, match="CENSORED"):
        owner.entry_learning_horizon_references({"fixture": True}, rows_by_role={"fit": np.array([4, 5], dtype=np.int64)})
    with pytest.raises(RuntimeError, match="ROWS_INVALID"):
        owner.entry_learning_horizon_references({"fixture": True}, rows_by_role={"fit": np.array([2, 2], dtype=np.int64)})
