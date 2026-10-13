"""Horizon learning design, dataset binding and batch dispatch (operator decision 12.10.2026).

Synthetic bytes prove mechanics only: exact batch keys, admission by the measured eligibility
cutoff, fail-closed censoring, and the dispatchers that pick the 20-day loss target.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import entry_observed_market_horizons_v1 as horizons
from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_HORIZON_TRAINING_PHASE, ENTRY_ONLY_TRAINING_PHASES,
)
from tests.test_native_prefix_recipe import _bind, _write
from tests.test_unified_exit_prospective_cost_policy_v1 import END, START, _build, _load


@pytest.fixture
def cost_terms(tmp_path):
    authority = _load(_build(tmp_path), "parameter_authority")
    return observed.entry_observed_market_economics(
        authority, coverage_start_utc=START, coverage_end_utc=END, verify_local_sources=False)


def _frame(rows: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame({"time": pd.date_range("2025-04-01T00:00:00Z", periods=rows, freq="5min")})
    for h in horizons.OBSERVATION_HORIZONS_TRADING_DAYS:
        long_col, short_col = horizons.GROSS_TARGET_COLUMNS_BY_HORIZON[h]
        frame[long_col] = np.arange(rows, dtype=np.float32) * 3.0 + h
        frame[short_col] = -np.arange(rows, dtype=np.float32) * 2.0 - h
        frame[horizons.ELAPSED_COLUMN_BY_HORIZON[h]] = np.full(rows, 86_400.0 * h)
    return frame


def dataset_scope(tmp_path, cost_terms, *, rows: int = 6, censor_row: int | None = None, cutoff_row: int = 4,
                  complete_after_cutoff: bool = False):
    frame = _frame(rows)
    # the measured cutoff is the last complete decision: every later row is censored by construction
    censored = [] if censor_row is None else [censor_row]
    if not complete_after_cutoff:
        censored += list(range(cutoff_row + 1, rows))
    for row in censored:
        for h in horizons.OBSERVATION_HORIZONS_TRADING_DAYS:
            frame.loc[row, horizons.GROSS_TARGET_COLUMNS_BY_HORIZON[h][0]] = np.nan
    path = tmp_path / "market.parquet"
    frame.to_parquet(path, index=False)
    source = {
        "parquet": _bind(path), "manifest": _write(tmp_path / "manifest.json", {}),
        "physical_rows": len(frame),
        "clock_sha256": hashlib.sha256(frame.time.array.asi8.astype("<i8").tobytes()).hexdigest(),
    }
    dataset = SimpleNamespace(
        parquet_path=path, df=frame[["time"]].copy(),
        _policy_dependent_auxiliary_bound_rows=np.ones(len(frame), dtype=bool),
        _policy_dependent_auxiliary_binding={
            "mode": "original_physical_split_targets", "role": "TRAIN",
            "parent_entry_parquet": source["parquet"], "parent_entry_manifest": source["manifest"],
            "policies": {"direction_policy": {"policy_sha256": "a" * 64}},
        },
    )
    cutoff = frame["time"].iloc[cutoff_row] + pd.Timedelta(minutes=5)
    scope = {
        "binding": {"path": "/synthetic/horizons_design.json", "sha256": "b" * 64},
        "target_contract": horizons.entry_observed_market_horizons_contract(),
        "training_phase": ENTRY_ONLY_HORIZON_TRAINING_PHASE, "economics": cost_terms,
        "reference_horizon_m5_bars": 19, "elapsed_wall_clock_seconds": 5700,
        "direction_policy_sha256": "a" * 64, "target_m1_source_sha256": "c" * 64,
        "physical_sources": {"train": source},
        "train_control_cutoff": "2025-06-01T00:00:00Z",
        "development_control_entry_end_exclusive": "2026-07-01T00:00:00Z",
        "decision_horizon_trading_days": 20, "loss_target_batch_key": horizons.DECISION_BATCH_TARGET_KEY,
        "batch_target_keys": dict(horizons.BATCH_TARGET_KEY_BY_HORIZON),
        "entry_eligibility_cutoff_utc": {"train": str(cutoff), "val": "2026-06-01T00:00:00Z"},
    }
    return dataset, scope, frame


def test_training_phases_name_the_horizon_phase_as_entry_only():
    assert ENTRY_ONLY_HORIZON_TRAINING_PHASE in ENTRY_ONLY_TRAINING_PHASES
    assert horizons.OBSERVED_ONLY_BATCH_TARGET_KEYS == [
        "entry_observed_market_target_bps_h1d", "entry_observed_market_target_bps_h5d"]
    assert horizons.HORIZON_LABEL_COLUMNS[-3:] == [
        "y_long_gross_at_20d_bps", "y_short_gross_at_20d_bps", "elapsed_seconds_at_20d"]


def test_binding_admits_rows_up_to_the_cutoff_and_emits_the_three_exact_keys(tmp_path, cost_terms):
    dataset, scope, frame = dataset_scope(tmp_path, cost_terms)
    horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)
    binding = dataset._entry_observed_market_binding
    assert binding["admitted_rows"] == 5 and binding["auxiliary_bound_rows"] == 6
    assert binding["censored_rows_inside_cutoff"] == 0 and binding["rows_after_cutoff"] == 1
    assert binding["identity"]["training_phase"] == ENTRY_ONLY_HORIZON_TRAINING_PHASE
    assert binding["identity"]["loss_target_batch_key"] == "entry_observed_market_target_bps_h20d"
    assert dataset._entry_observed_market_admitted_rows.tolist() == [True] * 5 + [False]
    items = horizons.observed_entry_batch_items(dataset, 2)
    assert set(items) == set(horizons.BATCH_TARGET_KEY_BY_HORIZON.values())
    # the 20-day target is the v1 net arithmetic on the 20-day gross/elapsed columns
    expected = observed.build_entry_observed_market_targets(
        gross_return_bps=torch.tensor([[2 * 3.0 + 20, -2 * 2.0 - 20]], dtype=torch.float32),
        elapsed_wall_clock_seconds=torch.tensor([86_400 * 20]), economics=cost_terms)["target_bps"][0]
    assert torch.equal(items["entry_observed_market_target_bps_h20d"], expected)
    assert items["entry_observed_market_target_bps_h20d"][2] == 0.0
    # rows beyond the cutoff are not admitted and carry NaN, so a loader may never draw them
    assert bool(torch.isnan(horizons.observed_entry_batch_items(dataset, 5)["entry_observed_market_target_bps_h1d"]).all())
    with pytest.raises(RuntimeError, match="ALREADY_BOUND"):
        horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)


def test_a_censored_row_inside_the_cutoff_is_excluded_and_counted_and_v1_scope_is_rejected(tmp_path, cost_terms):
    dataset, scope, _ = dataset_scope(tmp_path, cost_terms, censor_row=1)
    horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)
    binding = dataset._entry_observed_market_binding
    assert binding["admitted_rows"] == 4 and binding["censored_rows_inside_cutoff"] == 1 and binding["rows_after_cutoff"] == 1
    assert dataset._entry_observed_market_admitted_rows.tolist() == [True, False, True, True, True, False]
    assert bool(torch.isnan(horizons.observed_entry_batch_items(dataset, 1)["entry_observed_market_target_bps_h20d"]).all())
    # a complete row after the measured cutoff contradicts the design
    dataset, scope, _ = dataset_scope(tmp_path, cost_terms, complete_after_cutoff=True)
    with pytest.raises(RuntimeError, match="COMPLETE_ROW_AFTER_ELIGIBILITY_CUTOFF"):
        horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)
    dataset, scope, _ = dataset_scope(tmp_path, cost_terms)
    wrong = {**scope, "training_phase": "entry_only", "target_contract": observed.entry_observed_market_contract()}
    with pytest.raises(RuntimeError, match="DATASET_SCOPE_INVALID"):
        horizons.bind_entry_observed_market_horizons_dataset(dataset, wrong)
    # the v1 binder refuses a horizon scope and vice versa
    with pytest.raises(RuntimeError, match="DATASET_SCOPE_INVALID"):
        observed.bind_entry_observed_market_dataset(dataset, scope)


def test_batch_dispatch_picks_the_20_day_loss_target_and_reports_every_horizon(tmp_path, cost_terms):
    dataset, scope, _ = dataset_scope(tmp_path, cost_terms)
    horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)
    binding = dataset._entry_observed_market_binding
    rows = [0, 3]
    batch = {key: torch.stack([horizons.observed_entry_batch_items(dataset, r)[key] for r in rows])
             for key in horizons.BATCH_TARGET_KEY_BY_HORIZON.values()}
    y, valid = horizons.observed_entry_batch_targets(batch, binding, device=torch.device("cpu"))
    assert torch.equal(y, batch["entry_observed_market_target_bps_h20d"]) and bool(valid.all())
    by_horizon = horizons.observed_entry_batch_targets_by_horizon(batch, binding, device=torch.device("cpu"))
    assert set(by_horizon) == {1, 5, 20} and torch.equal(by_horizon[20], y)
    with pytest.raises(RuntimeError, match="BATCH_TARGET_INVALID"):
        horizons.observed_entry_batch_targets({k: v for k, v in batch.items() if "h5d" not in k}, binding, device=torch.device("cpu"))
    # a v1 binding dispatches to the v1 key and reports no horizons
    v1_binding = {"identity": {"training_phase": "entry_only"}}
    v1_batch = {"entry_observed_market_target_bps": y.clone()}
    y1, _ = horizons.observed_entry_batch_targets(v1_batch, v1_binding, device=torch.device("cpu"))
    assert torch.equal(y1, y) and horizons.observed_entry_batch_targets_by_horizon(v1_batch, v1_binding, device=torch.device("cpu")) == {}


def test_batch_items_never_emit_both_target_families(tmp_path, cost_terms):
    dataset, scope, _ = dataset_scope(tmp_path, cost_terms)
    horizons.bind_entry_observed_market_horizons_dataset(dataset, scope)
    dataset._entry_observed_market_targets = np.zeros((6, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="AMBIGUOUS"):
        horizons.observed_entry_batch_items(dataset, 0)
    assert horizons.observed_entry_batch_items(SimpleNamespace(), 0) == {}


def test_recipe_scope_dispatch_requires_exactly_one_design(monkeypatch):
    calls = []
    monkeypatch.setattr(horizons, "require_entry_observed_market_horizons_scope", lambda recipe: calls.append("h") or {"decision_horizon_trading_days": 20})
    monkeypatch.setattr(observed, "require_entry_observed_market_scope", lambda recipe: calls.append("v1") or {"training_phase": "entry_only"})
    assert horizons.require_observed_entry_scope({}) is None
    assert horizons.require_observed_entry_scope({"entry_observed_market_horizons": {}})["decision_horizon_trading_days"] == 20
    assert horizons.require_observed_entry_scope({"entry_observed_market": {}})["training_phase"] == "entry_only"
    with pytest.raises(RuntimeError, match="AMBIGUOUS"):
        horizons.require_observed_entry_scope({"entry_observed_market": {}, "entry_observed_market_horizons": {}})
    assert calls == ["h", "v1"]


def test_horizon_design_validation_binds_decision_keys_cutoffs_and_source_manifests(tmp_path, monkeypatch, cost_terms):
    train_manifest = _write(tmp_path / "train.manifest.json", {"extra": {"observed_market_horizons": {
        "contract": horizons.entry_observed_market_horizons_contract(), "censored_or_boundary_rows_are_nan": True,
        "label_columns": horizons.HORIZON_LABEL_COLUMNS}}})
    val_manifest = _write(tmp_path / "val.manifest.json", {"extra": {"observed_market_horizons": {
        "contract": horizons.entry_observed_market_horizons_contract(), "censored_or_boundary_rows_are_nan": True,
        "label_columns": horizons.HORIZON_LABEL_COLUMNS}}})
    plan = {
        "decision_horizon_trading_days": 20, "loss_target_batch_key": horizons.DECISION_BATCH_TARGET_KEY,
        "observed_only_batch_keys": list(horizons.OBSERVED_ONLY_BATCH_TARGET_KEYS),
        "entry_eligibility_cutoff_utc": {"train": "2025-05-02T00:00:00Z", "val": "2026-06-02T00:00:00Z"},
    }
    base = {
        "binding": {"path": "/synthetic/design.json", "sha256": "b" * 64}, "plan": plan,
        "target_contract": horizons.entry_observed_market_horizons_contract(),
        "training_phase": ENTRY_ONLY_HORIZON_TRAINING_PHASE, "economics": cost_terms,
        "reference_horizon_m5_bars": 19, "elapsed_wall_clock_seconds": 5700,
        "direction_policy_sha256": "a" * 64, "target_m1_source_sha256": "c" * 64,
        "physical_sources": {"train": {"manifest": train_manifest}, "val": {"manifest": val_manifest}},
        "train_control_cutoff": "2025-06-01T00:00:00+00:00",
        "development_control_entry_end_exclusive": "2026-07-01T00:00:00+00:00",
    }
    seen = {}
    def fake_core(recipe, **kwargs):
        seen.update(kwargs)
        return base
    monkeypatch.setattr(observed, "_require_observed_market_plan", fake_core)
    scope = horizons.require_entry_observed_market_horizons_scope({"entry_observed_market_horizons": {}})
    assert seen["expected_training_phase"] == ENTRY_ONLY_HORIZON_TRAINING_PHASE
    assert seen["expected_schema"] == "gx1_entry_observed_market_horizons_v1_learning_design"
    assert scope["loss_target_batch_key"] == "entry_observed_market_target_bps_h20d"
    assert scope["entry_eligibility_cutoff_utc"]["train"].startswith("2025-05-02")
    identity = horizons.entry_observed_market_horizons_identity(scope)
    assert identity["decision_horizon_trading_days"] == 20 and identity["batch_target_keys"]["20"] == scope["loss_target_batch_key"]
    assert identity["target_contract"] == horizons.entry_observed_market_horizons_contract()
    for fault in ("late_cutoff", "wrong_horizon", "manifest_contract"):
        bad = {**base, "plan": dict(plan)}
        if fault == "late_cutoff":
            bad["plan"]["entry_eligibility_cutoff_utc"] = {"train": "2025-06-02T00:00:00Z", "val": "2026-06-02T00:00:00Z"}
        elif fault == "wrong_horizon":
            bad["plan"]["decision_horizon_trading_days"] = 5
        else:
            broken = _write(tmp_path / "broken.manifest.json", {"extra": {"observed_market_horizons": {
                "contract": {}, "censored_or_boundary_rows_are_nan": True, "label_columns": horizons.HORIZON_LABEL_COLUMNS}}})
            bad["physical_sources"] = {"train": {"manifest": broken}, "val": {"manifest": val_manifest}}
        monkeypatch.setattr(observed, "_require_observed_market_plan", lambda recipe, **kwargs: bad)
        with pytest.raises(RuntimeError, match="ENTRY_OBSERVED_HORIZONS_(DESIGN|SOURCE_MANIFEST)_INVALID"):
            horizons.require_entry_observed_market_horizons_scope({"entry_observed_market_horizons": {}})


def test_planned_prefix_steps_follow_the_declared_entry_only_horizon_phase():
    horizon = {"scope": {"entry_only_training_phase": ENTRY_ONLY_HORIZON_TRAINING_PHASE}}
    assert horizons.planned_prefix_optimizer_steps(horizon) == horizons.MINIMUM_PLAN_FIT_OPTIMIZER_STEPS == 1024
    assert horizons.planned_prefix_optimizer_steps({"scope": {"same_architecture": True}}) == 256
    assert horizons.planned_prefix_optimizer_steps({}) == horizons.V38_SMOKE_PLANNED_OPTIMIZER_STEPS
