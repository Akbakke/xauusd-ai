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


WINDOW = ELIGIBLE[ELIGIBLE >= 60_000]  # the eligible rows whose decision lies in the recent window (30 000 rows)


@pytest.fixture
def study_successor(study_v2, monkeypatch):
    """A successor spec (operator decision 14.10.2026): fit/curve repeated byte for byte, the recent-window
    phase derived from the window size, the predecessor bound by hash."""
    policy, recipe, plan, seal, save, tmp_path = study_v2
    import copy
    predecessor = copy.deepcopy(plan)
    predecessor_binding = _bind(_write_plan(tmp_path / "predecessor.json", predecessor))
    order = plan_order = np.load(plan["row_bindings"]["epoch_order"]["path"], allow_pickle=False)
    in_order = order[np.isin(order, WINDOW)]
    for name, a in (("recent_year", WINDOW), ("recent_year_train_probe", np.sort(in_order[:4096]))):
        p = tmp_path / (name + ".npy")
        np.save(p, a)
        plan["row_bindings"][name] = _bind(p)
    spec = horizons.recent_window_phase_spec(int(len(WINDOW)))
    plan["phases"]["recent_year"] = {"run_id": "ENTRY_D1_HORIZON_RECENT_YEAR_20261014_001", **spec}
    plan["predecessor"] = predecessor_binding
    monkeypatch.setattr(native, "_recent_window_train_rows", lambda observed, eligible, *, window_days: WINDOW)
    recipe["run_id"] = "ENTRY_D1_HORIZON_RECENT_YEAR_20261014_001"
    recipe["out_bundle_dir"] = str(tmp_path / "RECENT_YEAR" / "CANDIDATE_BUNDLE")

    def seal_successor():
        recipe["entry_learning_study"] = {"plan": _bind(_write_plan(tmp_path / "successor.json", plan)), "phase": "recent_year"}
        policy["entry_learning_study_run"] = {
            "selection": recipe["entry_learning_study"], "run_id": recipe["run_id"],
            "out_bundle_dir": recipe["out_bundle_dir"], "source_bindings_sha256": recipe["source_bindings_sha256"],
            "optimizer_steps": plan["phases"]["recent_year"]["optimizer_steps"], "max_invocations": 1, "test_data_used": False}
        save()
    seal_successor()
    return policy, recipe, plan, seal_successor, save, tmp_path, spec, plan_order


def _write_plan(path, plan):
    import json
    path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    return path


def test_successor_admits_the_recent_window_phase_derived_from_the_window(study_successor):
    policy, recipe, plan, seal, save, tmp_path, spec, order = study_successor
    assert spec["optimizer_steps"] == (len(WINDOW) - 1) // 16 and spec["unique_train_rows"] == spec["optimizer_steps"] * 16
    assert spec["train_observation_steps"] == [0, 256, 1024, spec["optimizer_steps"]]
    assert spec["control_observation_steps"] == [0, spec["optimizer_steps"]] and spec["window_days"] == 365
    study = native.require_entry_learning_study_run(recipe, execution_budget={
        "stop_after_optimizer_steps": spec["optimizer_steps"], "stop_after_completed_val_epochs": None, "max_invocation_seconds": 12000})
    assert study["phase"] == "recent_year" and np.array_equal(study["rows"]["recent_year"], WINDOW)
    assert np.isin(study["rows"]["recent_year_train_probe"], WINDOW).all() and len(study["rows"]["recent_year_train_probe"]) == 4096
    # the loader order keeps the frozen order inside the window first; nothing is dropped or duplicated
    import torch
    from gx1.contracts.entry_observed_market_v1 import entry_learning_study_loader_order
    loader = entry_learning_study_loader_order(torch.as_tensor(order), study=study).numpy()
    assert len(loader) == len(order) and np.array_equal(np.sort(loader), np.sort(order))
    assert np.isin(loader[:len(WINDOW)], WINDOW).all() and not np.isin(loader[len(WINDOW):], WINDOW).any()
    assert np.array_equal(loader[:len(WINDOW)], order[np.isin(order, WINDOW)])
    assert np.array_equal(np.sort(loader[:4096]), study["rows"]["recent_year_train_probe"])


@pytest.mark.parametrize("fault", ["window_rows", "steps_literal", "predecessor_phase", "no_predecessor", "probe_rows"])
def test_successor_fails_closed(study_successor, fault):
    policy, recipe, plan, seal, save, tmp_path, spec, order = study_successor
    if fault == "window_rows":
        p = tmp_path / "recent_year.npy"
        np.save(p, WINDOW[:-1])
        plan["row_bindings"]["recent_year"] = _bind(p)
    elif fault == "steps_literal":
        plan["phases"]["recent_year"]["optimizer_steps"] = spec["optimizer_steps"] + 1
    elif fault == "predecessor_phase":
        plan["phases"]["curve"]["optimizer_steps"] = 8192
    elif fault == "no_predecessor":
        plan.pop("predecessor")
    else:
        p = tmp_path / "recent_year_train_probe.npy"
        np.save(p, np.sort(order[np.isin(order, WINDOW)][1:4097]))
        plan["row_bindings"]["recent_year_train_probe"] = _bind(p)
    seal()
    with pytest.raises(RuntimeError, match="ENTRY_STUDY|binding"):
        native.require_entry_learning_study_run(recipe)


def test_recent_window_rows_and_phase_spec_are_derived_not_literal():
    decisions = np.arange(0, 10) * 86_400 * 10**9  # one decision per day, day 0..9
    rows = horizons.recent_window_rows(decisions, cutoff_ns=int(decisions[9]), window_days=3)
    assert rows.tolist() == [7, 8, 9]  # (cutoff - 3 d, cutoff]
    with pytest.raises(RuntimeError, match="RECENT_WINDOW_TOO_SMALL"):
        horizons.recent_window_phase_spec(4096)
    spec = horizons.recent_window_phase_spec(65_148)
    assert spec["optimizer_steps"] == 4071 and spec["unique_train_rows"] == 65_136 and spec["max_invocations"] == 1


@pytest.fixture
def study_multi_window(study_v2, monkeypatch):
    """A successor spec with several named window phases (operator decision 15.10.2026): two seed variants on
    the 365-day window and one 180-day window at the design seed; fit/curve repeated byte for byte."""
    policy, recipe, plan, seal, save, tmp_path = study_v2
    import copy
    predecessor_binding = _bind(_write_plan(tmp_path / "pred2.json", copy.deepcopy(plan)))
    short = ELIGIBLE[ELIGIBLE >= 80_000]  # the 180-day window (synthetic): 20 000 rows
    windows = {"recent_365d_s20261015": (WINDOW, 365, 20261015),
               "recent_365d_s20261016": (WINDOW, 365, 20261016),
               "recent_180d": (short, 180, None)}
    order = np.load(plan["row_bindings"]["epoch_order"]["path"], allow_pickle=False)
    for name, (rows, days, seed) in windows.items():
        for key, a in ((name, rows), (name + "_train_probe", np.sort(order[np.isin(order, rows)][:4096]))):
            p = tmp_path / (key + ".npy")
            np.save(p, a)
            plan["row_bindings"][key] = _bind(p)
        spec = horizons.recent_window_phase_spec(int(len(rows)), window_days=days)
        if seed is not None:
            spec["seed"] = seed
        plan["phases"][name] = {"run_id": f"RUN_{name.upper()}", **spec}
    plan["predecessor"] = predecessor_binding

    def windows_by_days(observed, eligible, *, window_days):
        return {365: WINDOW, 180: short}[window_days]
    monkeypatch.setattr(native, "_recent_window_train_rows", windows_by_days)

    def select(name, seed=20260911):
        recipe["run_id"] = plan["phases"][name]["run_id"] if name in windows else recipe["run_id"]
        recipe["out_bundle_dir"] = str(tmp_path / name.upper() / "CANDIDATE_BUNDLE")
        recipe["trainer_cli"] = {**recipe.get("trainer_cli", {}), "seed": seed}
        recipe["entry_learning_study"] = {"plan": _bind(_write_plan(tmp_path / "multi.json", plan)), "phase": name}
        policy["entry_learning_study_run"] = {
            "selection": recipe["entry_learning_study"], "run_id": recipe["run_id"],
            "out_bundle_dir": recipe["out_bundle_dir"], "source_bindings_sha256": recipe["source_bindings_sha256"],
            "optimizer_steps": plan["phases"][name]["optimizer_steps"], "max_invocations": 1, "test_data_used": False}
        save()
    return policy, recipe, plan, select, tmp_path, windows


def test_multi_window_phases_admit_with_their_declared_seeds(study_multi_window):
    policy, recipe, plan, select, tmp_path, windows = study_multi_window
    select("recent_365d_s20261015", seed=20261015)
    study = native.require_entry_learning_study_run(recipe)
    assert study["phase_spec"]["seed"] == 20261015 and study["phase_spec"]["window_days"] == 365
    select("recent_180d", seed=20260911)
    study = native.require_entry_learning_study_run(recipe)
    assert "seed" not in study["phase_spec"] and study["phase_spec"]["window_days"] == 180
    assert study["phase_spec"]["optimizer_steps"] == (len(windows["recent_180d"][0]) - 1) // 16
    # the loader order and observer keys follow the phase name
    import torch
    from gx1.contracts.entry_observed_market_v1 import entry_learning_study_loader_order
    order = np.load(plan["row_bindings"]["epoch_order"]["path"], allow_pickle=False)
    loader = entry_learning_study_loader_order(torch.as_tensor(order), study=study).numpy()
    rows = windows["recent_180d"][0]
    assert np.isin(loader[:len(rows)], rows).all() and np.array_equal(np.sort(loader), np.sort(order))


@pytest.mark.parametrize("fault", ["undeclared_seed", "trainer_seed_mismatch", "fit_with_variant_seed", "bad_phase_name"])
def test_multi_window_fails_closed(study_multi_window, fault):
    policy, recipe, plan, select, tmp_path, windows = study_multi_window
    if fault == "undeclared_seed":
        plan["phases"]["recent_365d_s20261015"]["seed"] = 20269999
        select("recent_365d_s20261015", seed=20269999)
    elif fault == "trainer_seed_mismatch":
        select("recent_365d_s20261015", seed=20260911)
    elif fault == "fit_with_variant_seed":
        select("fit", seed=20261015)
        recipe["run_id"] = plan["phases"]["fit"]["run_id"]
        recipe["out_bundle_dir"] = str(tmp_path / "FIT" / "CANDIDATE_BUNDLE")
        recipe["entry_learning_study"] = {"plan": recipe["entry_learning_study"]["plan"], "phase": "fit"}
        policy["entry_learning_study_run"] = {**policy["entry_learning_study_run"],
                                              "selection": recipe["entry_learning_study"], "run_id": recipe["run_id"],
                                              "out_bundle_dir": recipe["out_bundle_dir"], "optimizer_steps": 1024}
    else:
        plan["phases"]["Recent-BAD!"] = plan["phases"].pop("recent_180d")
        select("recent_365d_s20261015", seed=20261015)
    with pytest.raises(RuntimeError, match="ENTRY_STUDY"):
        native.require_entry_learning_study_run(recipe)
