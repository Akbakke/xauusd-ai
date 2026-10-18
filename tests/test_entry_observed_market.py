from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts.entry_observed_market_v1 import (
    GROSS_TARGET_COLUMNS, build_entry_observed_market_targets,
    entry_observed_market_contract, entry_observed_market_economics,
    require_entry_observed_market_economics,
)
from gx1.contracts.unified_exit_economics_objective_v2 import SECONDS_PER_YEAR
from tests.test_unified_exit_prospective_cost_policy_v1 import START, END, _build, _load


@pytest.fixture
def economics(tmp_path):
    authority = _load(_build(tmp_path), "parameter_authority")
    return entry_observed_market_economics(
        authority, coverage_start_utc=START, coverage_end_utc=END,
        verify_local_sources=False,
    )


def test_market_contract_is_teacher_free_and_preserves_existing_targets():
    value = entry_observed_market_contract()
    assert value["gross_target_columns"] == list(GROSS_TARGET_COLUMNS)
    assert value["action_order"] == ["LONG", "SHORT", "FLAT"]
    assert value["exit_model_used"] is False
    assert value["hindsight_best_exit_used"] is False
    assert value["holding_time_limit"] is False
    assert value["score_is_calibrated_probability"] is False
    assert value["retained_forecast_horizons_m5"] == [1, 5, 12, 24]


def test_net_markouts_preserve_signed_losses_exact_flat_and_elapsed_cost(economics):
    gross = torch.tensor([[12.0, -14.0], [-2.0, 1.0], [3.0, -4.0]], requires_grad=True)
    elapsed = torch.tensor([300, 5700, 172800], dtype=torch.int64)
    result = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    expected = torch.tensor([
        [12.0 - 4.0 - 540.0 * 300 / SECONDS_PER_YEAR, -18.0, 0.0],
        [-2.0 - 4.0 - 540.0 * 5700 / SECONDS_PER_YEAR, -3.0, 0.0],
        [3.0 - 4.0 - 540.0 * 172800 / SECONDS_PER_YEAR, -8.0, 0.0],
    ])
    assert torch.equal(result["target_bps"], expected)
    assert bool(result["target_valid"].all())
    assert not result["target_bps"].requires_grad
    assert gross.grad is None
    assert torch.equal(gross.detach(), torch.tensor([[12.0, -14.0], [-2.0, 1.0], [3.0, -4.0]]))


@pytest.mark.parametrize("defect", ["nan", "wrong_side_count", "zero_elapsed", "float_elapsed", "negative_elapsed"])
def test_incomplete_or_invalid_outcomes_never_become_flat_targets(economics, defect):
    gross = torch.tensor([[1.0, -2.0]])
    elapsed = torch.tensor([300], dtype=torch.int64)
    if defect == "nan": gross[0, 0] = float("nan")
    elif defect == "wrong_side_count": gross = gross[:, :1]
    elif defect == "zero_elapsed": elapsed[0] = 0
    elif defect == "negative_elapsed": elapsed[0] = -1
    else: elapsed = elapsed.float()
    with pytest.raises(RuntimeError, match="TARGET_INPUT_INVALID"):
        build_entry_observed_market_targets(
            gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
        )


@pytest.mark.parametrize("defect", ["execution_cost", "financing", "truth_claim", "extra_teacher"])
def test_economics_drift_is_rejected(economics, defect):
    changed = copy.deepcopy(economics)
    if defect == "execution_cost": changed["round_trip_execution_cost_bps"] = 0.0
    elif defect == "financing": changed["annual_financing_cost_bps"][0] = 0.0
    elif defect == "truth_claim": changed["historical_cost_truth_qualified"] = True
    else: changed["exit_target_model_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="ECONOMICS_INVALID"):
        require_entry_observed_market_economics(changed)


def test_net_target_does_not_depend_on_exit_state_or_torch_rng(economics):
    gross = torch.tensor([[10.0, -12.0], [-8.0, 6.0]])
    elapsed = torch.tensor([5700, 5700], dtype=torch.int64)
    first = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    torch.manual_seed(109)
    # The target API admits no model/teacher state, and consumes no random draw.
    before = torch.get_rng_state().clone()
    second = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(first[k], second[k]) for k in first)


# Synthetic fixtures below prove semantics, binding and masks, never market edge.
def _dynamic_case(tmp_path, economics, *, gap=False):
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from gx1.contracts import entry_causal_m1_outcomes_v1 as quotes
    from gx1.contracts import entry_model_native_aux_targets_v3 as aux
    from gx1.contracts import entry_observed_market_v1 as observed
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.unified_exit_market_closure_authority_v1 import m1_clock_sha256
    clock = pd.date_range("2025-06-02T00:00:00Z", periods=500, freq="min")
    price = 100. + np.arange(500) * .01
    frame = pd.DataFrame({"time": clock, "bid_open": price, "bid_high": price + .03,
        "bid_low": price, "ask_open": price + .01,
        "ask_high": price + .04, "ask_low": price + .01})
    if gap:
        frame = frame.drop(index=8).reset_index(drop=True)
    quote_path = tmp_path / "original_quotes.parquet"
    frame.to_parquet(quote_path, index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    binding = {"path": str(quote_path), "sha256": sha(quote_path), "size_bytes": quote_path.stat().st_size,
        "rows": len(frame), "clock_sha256": m1_clock_sha256(pd.DatetimeIndex(frame.time))}
    cutoff = clock[-1] + pd.Timedelta(minutes=1)
    index = quotes.prepare_dynamic_m1_prefix_index(frame, split_end_exclusive=cutoff,
        source_parquet_sha256=binding["sha256"])
    decisions = clock[[0, 10]]
    fill = quotes.build_dynamic_m1_fill_surface(index, m5_decision_times=decisions)
    query = dict(fill_surface=fill, query_entry_rows=np.arange(2, dtype=np.int64), horizon_minutes=np.ones(2, np.int64))
    first = quotes.query_dynamic_m1_prefix(index, **query)
    closed = quotes.query_dynamic_m1_prefix(index, endpoint_kind="closed_m1_prefix", **query)
    physical = pd.DataFrame({"time": decisions,
        "dynamic_entry_m1_row": fill.entry_m1_row.to_numpy(np.int64),
        "dynamic_first_prefix_observed": np.ones(2, bool),
        "dynamic_first_terminal_censor_reason": first["terminal_censor_reason"],
        "dynamic_first_path_censor_reason": closed["path_censor_reason"]})
    for j, side in enumerate(("long", "short")):
        physical["dynamic_first_terminal_observed_" + side] = first["terminal_gross_observed"][:, j]
        physical["dynamic_first_path_observed_" + side] = closed["path_observed"][:, j]
    for name in aux.MODEL_NATIVE_FORECAST_TARGET_COLUMNS:
        physical[name] = np.zeros(2, np.float32)
    for name in aux.MODEL_NATIVE_FORECAST_VALID_COLUMNS:
        physical[name] = np.zeros(2, bool)
    for name in horizons.HORIZON_LABEL_COLUMNS:
        physical[name] = np.nan
    path = tmp_path / "entries.parquet"
    physical.to_parquet(path, index=False)
    meta = {"schema_version": "gx1_dynamic_m5_physical_target_pointers_v1",
        "contract": quotes.dynamic_m1_prefix_contract(), "source_quote_binding": binding,
        "source_quote_columns": list(observed.DYNAMIC_M5_QUOTE_COLUMNS), "index_binding": index.binding(),
        "pointer_columns": list(observed.DYNAMIC_M5_POINTER_COLUMNS), "target_only_columns": True,
        "feature_inputs_include_future_masks": False, "dense_future_paths_materialized": False,
        "first_terminal_endpoint_kind": "terminal_open_inclusive", "first_path_endpoint_kind": "closed_m1_prefix",
        "first_prefix_horizon_minutes": 1, "closure_mode": "unknown_gaps_only_no_inferred_market_calendar"}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"extra": {"dynamic_m5_supervision": meta}}))
    source = {"parquet": {"path": str(path), "sha256": sha(path)},
        "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
        "physical_rows": 2, "clock_sha256": m1_clock_sha256(decisions)}
    dataset = SimpleNamespace(parquet_path=path, df=physical.copy(deep=True),
        aux_head_target_contract=aux.model_native_aux_target_contract_metadata(emission_scope=aux.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE),
        _policy_dependent_auxiliary_bound_rows=np.ones(2, bool),
        _policy_dependent_auxiliary_binding={"mode": aux.DYNAMIC_M5_PHYSICAL_BINDING_MODE, "role": "TRAIN",
            "parent_entry_parquet": source["parquet"], "parent_entry_manifest": source["manifest"]})
    scope = {"binding": {"path": "/synthetic/entry_design.json", "sha256": "a" * 64},
        "target_contract": observed.entry_observed_dynamic_m5_path_contract(),
        "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, "economics": economics,
        "query_minutes": quotes.DYNAMIC_M1_PREFIX_HORIZON_MINUTES, "target_m1_source_sha256": binding["sha256"],
        "physical_sources": {"train": source}, "train_control_cutoff": cutoff.isoformat(),
        "development_control_entry_end_exclusive": cutoff.isoformat()}
    return dataset, scope, index, fill


def test_dynamic_provider_real_file_binding_preserves_rows_and_independent_masks(tmp_path, economics):
    import numpy as np
    import pandas as pd
    from gx1.contracts import entry_observed_market_v1 as observed
    dataset, scope, index, fill = _dynamic_case(tmp_path, economics, gap=True)
    before = dataset.df.copy(deep=True)
    observed.bind_entry_observed_market_dataset(dataset, scope)
    pd.testing.assert_frame_equal(dataset.df, before)
    provider = dataset._entry_dynamic_m5_target_provider
    result = provider.targets_for_rows(np.array([1, 0, 1], np.int64))
    terminal = result[observed.DYNAMIC_M5_MARKOUT_VALID_KEY]
    path = result[observed.DYNAMIC_M5_PATH_VALID_KEY]
    assert terminal.shape == (3, 480, 3) and path.shape == (3, 480, 2)
    # Missing8min quote censors3min endpoint;4min endpoint still exists,
    # whereas4min extrema remain unknown. Shorter1/2min paths survive.
    assert path[1, :2].all()
    assert not terminal[1, 2].any()
    assert terminal[1, 3].all() and not path[1, 3].any()
    assert np.array_equal(terminal[..., 2], terminal[..., 0] & terminal[..., 1])
    assert (result[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][..., 2] == 0).all()
    assert (result[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][~terminal] == 0).all()
    assert (result[observed.DYNAMIC_M5_MFE_TARGET_KEY][~path] == 0).all()
    assert np.array_equal(result[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][0], result[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][2])
    binding = dataset._entry_observed_market_binding
    assert binding["dense_future_paths_materialized"] is False
    assert "reference_horizon_m5_bars" not in binding["identity"]
    assert not hasattr(dataset, "_entry_observed_market_targets")
    for bad in (np.array([-1]), np.array([2]), np.array([1.]), np.array([], np.int64)):
        with pytest.raises(RuntimeError, match="PARENT_ROWS_INVALID"):
            provider.targets_for_rows(bad)


@pytest.mark.parametrize("fault", ["input_clock", "global_pointer", "prefix_mask", "quote_hash", "source_sha", "wrong_phase", "legacy_contract", "legacy_binding_mode"])
def test_dynamic_physical_binding_rejects_mismatched_provenance(tmp_path, economics, fault):
    import hashlib
    from gx1.contracts import entry_observed_market_v1 as observed
    dataset, scope, index, fill = _dynamic_case(tmp_path, economics)
    if fault == "input_clock":
        import pandas as pd
        dataset.df.loc[0, "time"] += pd.Timedelta(minutes=5)
    elif fault in ("global_pointer", "prefix_mask"):
        import pandas as pd
        changed = pd.read_parquet(dataset.parquet_path)
        if fault == "global_pointer": changed.loc[0, "dynamic_entry_m1_row"] += 1
        else: changed.loc[0, "dynamic_first_terminal_observed_long"] = False
        changed.to_parquet(dataset.parquet_path, index=False)
        source = scope["physical_sources"]["train"]
        source["parquet"]["sha256"] = hashlib.sha256(dataset.parquet_path.read_bytes()).hexdigest()
    elif fault == "quote_hash":
        from pathlib import Path
        Path(tmp_path / "original_quotes.parquet").write_bytes(b"changed")
    elif fault == "source_sha": scope["target_m1_source_sha256"] = "c" * 64
    elif fault == "wrong_phase": scope["training_phase"] = "entry_only_m5_path"
    elif fault == "legacy_binding_mode": dataset._policy_dependent_auxiliary_binding["mode"] = "original_physical_split_targets"
    else: scope["target_contract"] = observed.entry_observed_m5_path_contract()
    with pytest.raises((RuntimeError, ValueError)):
        observed.bind_entry_dynamic_m5_dataset(dataset, scope)
    assert not hasattr(dataset, "_entry_dynamic_m5_target_provider")


def test_dynamic_provider_uses_existing_declared_cost_owner_at_actual_elapsed(tmp_path, economics):
    import numpy as np
    from gx1.contracts import entry_observed_market_v1 as observed
    from gx1.contracts import entry_causal_m1_outcomes_v1 as quotes
    dataset, scope, index, fill = _dynamic_case(tmp_path, economics)
    observed.bind_entry_dynamic_m5_dataset(dataset, scope)
    result = dataset._entry_dynamic_m5_target_provider.targets_for_rows(np.array([0], np.int64))
    raw = quotes.query_dynamic_m1_prefix(index, fill_surface=fill,
        query_entry_rows=np.array([0, 0]), horizon_minutes=np.array([1, 120]))
    expected = build_entry_observed_market_targets(gross_return_bps=torch.tensor(raw["terminal_gross_return_bps"], dtype=torch.float64),
        elapsed_wall_clock_seconds=torch.from_numpy(raw["elapsed_wall_seconds"]), economics=economics)["target_bps"].float().numpy()
    assert np.array_equal(result[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][0, [0, 119]], expected)
    path = result[observed.DYNAMIC_M5_PATH_VALID_KEY]
    # SHORT's favorable extreme stays signed negative on this up-only toy path.
    assert (result[observed.DYNAMIC_M5_MFE_TARGET_KEY][0, :1, 1] < 0).all()
    assert (result[observed.DYNAMIC_M5_MAE_TARGET_KEY][path] >= 0).all()


def test_dynamic_closed_path_tail_remains_active_without_terminal_quote(tmp_path, economics):
    import numpy as np
    import pandas as pd
    from gx1.contracts import entry_causal_m1_outcomes_v1 as quotes
    from gx1.contracts import entry_observed_market_v1 as observed
    dataset, scope, index, fill = _dynamic_case(tmp_path, economics)
    # This M5 decision fills at the last available23:59-equivalent M1 row.
    # The complete closed M1 bucket is known, whereas next-minute open is absent.
    last_decision = index.source.times[-1] - pd.Timedelta(minutes=5)
    # Use an aligned M5 decision whose fill is the final M1 bucket.
    aligned_last = last_decision.floor("5min") + pd.Timedelta(minutes=5)
    retained = index.source.times.searchsorted(aligned_last) + 1
    source = pd.DataFrame({"time": index.source.times[:retained], **{
        name: values[:retained] for name, values in index.source.values.items()}})
    prefix_index = quotes.prepare_dynamic_m1_prefix_index(source,
        split_end_exclusive=aligned_last + pd.Timedelta(minutes=1), source_parquet_sha256="c" * 64)
    tail_fill = quotes.build_dynamic_m1_fill_surface(prefix_index,
        m5_decision_times=pd.DatetimeIndex([aligned_last - pd.Timedelta(minutes=5)]))
    provider = observed.DynamicM5TargetProvider(index=prefix_index, fill_surface=tail_fill,
        admitted_rows=np.array([True]), economics=economics)
    result = provider.targets_for_rows(np.array([0], np.int64))
    assert not result[observed.DYNAMIC_M5_MARKOUT_VALID_KEY].any()
    assert not result[observed.DYNAMIC_M5_PATH_VALID_KEY].any()
    assert result[observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY][0, 0].all()
    assert not result[observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY][0, 1:].any()
    assert np.isfinite(result[observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY][0, 0]).all()
