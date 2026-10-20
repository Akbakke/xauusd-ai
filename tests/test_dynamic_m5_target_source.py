"""Small physical source projections; no model or full-feature Dataset is built."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gx1.contracts import entry_causal_m1_outcomes_v1 as quote
from gx1.contracts import entry_model_native_aux_targets_v3 as targets
from gx1.contracts import entry_observed_market_horizons_v1 as horizons
from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.unified_exit_market_closure_authority_v1 import m1_clock_sha256
from tests.test_entry_observed_market import economics


def _bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write(path, value):
    path.write_text(json.dumps(value))
    return _bind(path)


@pytest.fixture
def target_source_case(tmp_path, economics):
    times = pd.DatetimeIndex(list(pd.date_range("2025-06-01T00:05Z", periods=11, freq="min"))
        + list(pd.date_range("2025-06-03T00:05Z", periods=11, freq="min")))
    bid = 100. + np.arange(len(times), dtype=np.float64) / 100
    raw = pd.DataFrame({"time": times, "bid_open": bid, "ask_open": bid + .1,
        "bid_high": bid + .2, "ask_high": bid + .3,
        "bid_low": bid - .2, "ask_low": bid - .1})
    quote_path = tmp_path / "original_quotes.parquet"
    raw.to_parquet(quote_path, index=False)
    quote_binding = {**_bind(quote_path), "size_bytes": quote_path.stat().st_size,
        "rows": len(raw), "clock_sha256": m1_clock_sha256(times)}
    sources, expected = {}, {}
    for split, start, cutoff in (("train", "2025-06-01T00:00Z", "2025-06-03T00:00Z"),
                                 ("val", "2025-06-03T00:00Z", "2025-06-05T00:00Z")):
        clock = pd.date_range(start, periods=3, freq="5min")
        index = quote.prepare_dynamic_m1_prefix_index(raw, split_end_exclusive=cutoff,
            source_parquet_sha256=quote_binding["sha256"])
        fill = quote.build_dynamic_m1_fill_surface(index, m5_decision_times=clock)
        query = {"fill_surface": fill, "query_entry_rows": np.arange(3, dtype=np.int64),
            "horizon_minutes": np.ones(3, np.int64)}
        first = quote.query_dynamic_m1_prefix(index, **query)
        closed = quote.query_dynamic_m1_prefix(index, **query, endpoint_kind="closed_m1_prefix")
        frame = pd.DataFrame({"time": clock, "dynamic_entry_m1_row": fill.entry_m1_row.to_numpy(np.int64),
            "dynamic_first_prefix_observed": True,
            "dynamic_first_terminal_observed_long": first["terminal_gross_observed"][:, 0],
            "dynamic_first_terminal_observed_short": first["terminal_gross_observed"][:, 1],
            "dynamic_first_path_observed_long": closed["path_observed"][:, 0],
            "dynamic_first_path_observed_short": closed["path_observed"][:, 1],
            "dynamic_first_terminal_censor_reason": first["terminal_censor_reason"],
            "dynamic_first_path_censor_reason": closed["path_censor_reason"]})
        for horizon, name, mask in zip(targets.MODEL_NATIVE_AUX_FORECAST_HORIZONS,
                targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS, targets.MODEL_NATIVE_FORECAST_VALID_COLUMNS):
            frame[name] = np.array([horizon, -horizon, 0], np.float32)
            frame[mask] = np.array([True, True, False])
        for name in horizons.HORIZON_LABEL_COLUMNS:
            frame[name] = np.nan
        frame["unused_feature"] = np.array([901., 902., 903.])
        path = tmp_path / (split + ".parquet")
        frame.to_parquet(path, index=False)
        meta = {"schema_version": "gx1_dynamic_m5_physical_target_pointers_v1",
            "contract": quote.dynamic_m1_prefix_contract(), "source_quote_binding": quote_binding,
            "source_quote_columns": list(targets.DYNAMIC_M5_QUOTE_COLUMNS),
            "pointer_columns": list(targets.DYNAMIC_M5_POINTER_COLUMNS), "index_binding": index.binding(),
            "target_only_columns": True, "feature_inputs_include_future_masks": False,
            "dense_future_paths_materialized": False, "first_terminal_endpoint_kind": "terminal_open_inclusive",
            "first_path_endpoint_kind": "closed_m1_prefix", "first_prefix_horizon_minutes": 1,
            "closure_mode": "unknown_gaps_only_no_inferred_market_calendar"}
        sources[split] = {"parquet": _bind(path),
            "manifest": _write(tmp_path / (split + ".manifest.json"), {"extra": {"dynamic_m5_supervision": meta}}),
            "physical_rows": 3, "clock_sha256": hashlib.sha256(np.asarray(clock.asi8, dtype="<i8").tobytes()).hexdigest()}
        expected[split] = {"frame": frame, "index": index, "fill": fill, "first": first,
            "closed": closed, "metadata": meta}
    scope = {"binding": _write(tmp_path / "observed_plan.json", {}),
        "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "target_contract": observed.entry_observed_dynamic_m5_path_contract(),
        "query_minutes": quote.DYNAMIC_M1_PREFIX_HORIZON_MINUTES, "physical_sources": sources,
        "target_m1_source_sha256": quote_binding["sha256"], "economics": economics,
        "train_control_cutoff": "2025-06-03T00:00Z",
        "development_control_entry_end_exclusive": "2025-06-05T00:00Z"}
    return SimpleNamespace(scope=scope, expected=expected)


def _dataset(case, split):
    source = case.scope["physical_sources"][split]
    return SimpleNamespace(parquet_path=Path(source["parquet"]["path"]),
        df=case.expected[split]["frame"][["time", "unused_feature"]].copy(),
        aux_head_target_contract=targets.model_native_aux_target_contract_metadata(
            emission_scope=targets.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE),
        _policy_dependent_auxiliary_binding={"mode": targets.DYNAMIC_M5_PHYSICAL_BINDING_MODE,
            "role": "TRAIN" if split == "train" else "CONTROL256",
            "parent_entry_parquet": source["parquet"], "parent_entry_manifest": source["manifest"]},
        _policy_dependent_auxiliary_bound_rows=np.array([True, False, True]))


@pytest.mark.parametrize("split", ["train", "val"])
def test_target_source_projection_preserves_original_source_and_dataset_targets(target_source_case, monkeypatch, split):
    case = target_source_case
    expected = case.expected[split]
    path = Path(case.scope["physical_sources"][split]["parquet"]["path"])
    columns = ["time", *targets.DYNAMIC_M5_POINTER_COLUMNS, *targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
        *targets.MODEL_NATIVE_FORECAST_VALID_COLUMNS, *horizons.HORIZON_LABEL_COLUMNS]
    reads = []
    real_read = pq.read_table
    def projected_read(source, *args, **kwargs):
        if Path(source) == path:
            reads.append(kwargs.get("columns"))
        return real_read(source, *args, **kwargs)
    monkeypatch.setattr(pq, "read_table", projected_read)
    actual = observed.read_dynamic_m5_target_source(case.scope, split=split)
    assert set(actual) == {"frame", "index", "fill", "first", "closed", "first_valid", "metadata"}
    assert len(columns) == 26 and reads == [columns]
    pd.testing.assert_frame_equal(actual["frame"], expected["frame"][columns])
    pd.testing.assert_frame_equal(actual["fill"], expected["fill"])
    assert actual["index"].binding() == expected["index"].binding()
    assert actual["metadata"] == expected["metadata"]
    for name in ("first", "closed"):
        for key, value in expected[name].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[name][key], value)
            else:
                assert actual[name][key] == value
    assert actual["first_valid"].tolist() == [True, True, True]
    assert not actual["first"]["terminal_gross_observed"][-1].any()
    assert actual["closed"]["path_observed"][-1].all()
    dataset = _dataset(case, split)
    admitted = dataset._policy_dependent_auxiliary_bound_rows.copy()
    original_provider = observed.DynamicM5TargetProvider(index=expected["index"], fill_surface=expected["fill"],
        admitted_rows=admitted, economics=case.scope["economics"])
    original_targets = original_provider.targets_for_rows(np.array([0, 2], np.int64))
    original_forecast_valid = targets.require_m5_forecast_observation_storage(expected["frame"], context="TEST")
    original_support, original_support_valid = horizons.build_masked_support_forecast_targets(
        frame=expected["frame"], scope=case.scope, split=split, admitted_mask=admitted)
    observed.bind_entry_dynamic_m5_dataset(dataset, case.scope)
    assert reads == [columns, columns]
    actual_targets = dataset._entry_dynamic_m5_target_provider.targets_for_rows(np.array([0, 2], np.int64))
    assert actual_targets.keys() == original_targets.keys()
    for key, value in original_targets.items():
        assert actual_targets[key].dtype == value.dtype and actual_targets[key].shape == value.shape
        assert actual_targets[key].tobytes() == value.tobytes()
    np.testing.assert_array_equal(dataset._entry_observed_market_admitted_rows, admitted)
    np.testing.assert_array_equal(dataset._entry_observed_market_m5_forecast_valid, original_forecast_valid)
    np.testing.assert_array_equal(dataset._entry_observed_market_support_forecast, original_support)
    np.testing.assert_array_equal(dataset._entry_observed_market_support_forecast_valid, original_support_valid)
    binding = dataset._entry_observed_market_binding
    assert binding["physical_source"] == case.scope["physical_sources"][split]
    assert binding["source_quote_binding"] == expected["metadata"]["source_quote_binding"]
    assert binding["index_binding"] == expected["index"].binding()
    assert binding["admitted_rows"] == 2 and binding["first_closed_prefix_aux_only_rows"] == 1
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    assert binding["binding_sha256"] == observed._sha(unsigned)


@pytest.mark.parametrize("fault,error", [
    ("clock", "DYNAMIC_M5_DATASET_CLOCK_INVALID"),
    ("pointer", "DYNAMIC_M5_GLOBAL_QUOTE_POINTER_MISMATCH"),
    ("first_mask", "DYNAMIC_M5_FIRST_PREFIX_MASK_MISMATCH"),
    ("censor", "DYNAMIC_M5_FIRST_PREFIX_REASON_MISMATCH"),
])
def test_target_source_rejects_source_clock_pointer_mask_and_censor_drift(target_source_case, fault, error):
    case = target_source_case
    frame = case.expected["train"]["frame"].copy()
    if fault == "clock": frame.loc[1, "time"] = frame.loc[0, "time"]
    elif fault == "pointer": frame.loc[0, "dynamic_entry_m1_row"] += 1
    elif fault == "first_mask": frame.loc[0, "dynamic_first_terminal_observed_long"] = False
    else: frame.loc[0, "dynamic_first_path_censor_reason"] = "unknown_source_absence"
    source = case.scope["physical_sources"]["train"]
    path = Path(source["parquet"]["path"])
    frame.to_parquet(path, index=False)
    source["parquet"] = _bind(path)
    with pytest.raises(RuntimeError, match=error):
        observed.read_dynamic_m5_target_source(case.scope, split="train")


@pytest.mark.parametrize("fault,error", [
    ("input_clock", "DYNAMIC_M5_DATASET_CLOCK_INVALID"),
    ("auxiliary", "DYNAMIC_M5_DATASET_SOURCE_INVALID"),
])
def test_target_source_extraction_keeps_dataset_clock_and_auxiliary_joins(target_source_case, fault, error):
    case = target_source_case
    dataset = _dataset(case, "train")
    if fault == "input_clock": dataset.df.loc[0, "time"] += pd.Timedelta(minutes=5)
    else: dataset._policy_dependent_auxiliary_binding["parent_entry_manifest"] = {"path": "wrong", "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match=error):
        observed.bind_entry_dynamic_m5_dataset(dataset, case.scope)


def test_target_source_never_opens_test_split(target_source_case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid split reached physical source I/O")
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(observed, "_read_dynamic_m1_index", forbidden)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_TARGET_SOURCE_SCOPE_INVALID"):
        observed.read_dynamic_m5_target_source(target_source_case.scope, split="test")


@pytest.mark.parametrize("fault,error", [
    ("missing_stage", "DYNAMIC_M5_LOSS_FIT_NOT_EXACTLY_AUTHORIZED"),
    ("consumed_claim", "DYNAMIC_M5_LOSS_FIT_ALREADY_CONSUMED"),
    ("wrong_binding", "DYNAMIC_M5_LOSS_FIT_EXECUTION_BINDING_INVALID"),
])
def test_fit_cli_rejects_unadmitted_or_consumed_work_before_target_reads(tmp_path, monkeypatch, fault, error):
    import sys
    from gx1.scripts import fit_dynamic_m5_loss_scales_v1 as cli
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()
    runs.mkdir()
    output = runs / "synthetic_admission_only"
    output.mkdir()
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(cli, "RUNS_ROOT", runs)
    monkeypatch.setattr(cli, "_source_state", lambda: "b" * 40)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "admission_fixture")
    for key, value in {"GX1_CAPPED_CLASS": "audit", "GX1_CAPPED_MEMORY_BYTES": "4294967296",
                       "GX1_CAPPED_SWAP_BYTES": "536870912"}.items():
        monkeypatch.setenv(key, value)
    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted CLI reached target-data setup")
    monkeypatch.setattr(cli, "_prepare_targets", forbidden)
    plan = {"schema_version": cli.PLAN_SCHEMA, "run_id": "synthetic_admission_only",
        "output_dir": str(output), "prepared_source_head": "a" * 40,
        "input_recipe": _write(tmp_path / "unread_recipe.json", {}),
        "population_design": _write(tmp_path / "unread_design.json", {}),
        "chunk_rows": 2, "progress_interval_batches": 1, "expected_train_rows": 3}
    plan_ref = _write(tmp_path / "PLAN.json", plan)
    script_ref = _bind(cli.SCRIPT)
    policy = {"training_enabled": False, "full_epoch_training_allowed": False, "full_val_allowed": False,
        "dynamic_m5_loss_scale_fit_run": {"execution_authorized": True, "training_enabled": False,
            "planned_invocations": 1, "completed_invocations": 0, "relaunch_allowed": False,
            "plan": plan_ref, "script": script_ref}}
    if fault == "missing_stage":
        policy.pop("dynamic_m5_loss_scale_fit_run")
    policy_ref = _write(repo / "NEXT_RUN_POLICY.json", policy)
    binding = {"schema_version": cli.BINDING_SCHEMA, "source_head": "b" * 40,
        "policy_sha256": policy_ref["sha256"], "plan_sha256": plan_ref["sha256"],
        "script_sha256": script_ref["sha256"]}
    if fault == "wrong_binding":
        binding["plan_sha256"] = "0" * 64
    binding_ref = _write(tmp_path / "EXECUTION_BINDING.json", binding)
    if fault == "consumed_claim":
        (output / "CLAIM.json").write_bytes(b"previous immutable claim\n")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(RuntimeError, match=error):
        cli.main(["--plan-json", plan_ref["path"], "--plan-sha256", plan_ref["sha256"],
            "--execution-binding-json", binding_ref["path"], "--execution-binding-sha256", binding_ref["sha256"]])
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_fit_generator_consumes_actual_numpy_provider_and_last_partial_batch(target_source_case):
    import time
    from gx1.scripts import fit_dynamic_m5_loss_scales_v1 as cli
    case = target_source_case
    source = observed.read_dynamic_m5_target_source(case.scope, split="train")
    provider = observed.DynamicM5TargetProvider(index=source["index"], fill_surface=source["fill"],
        admitted_rows=source["first_valid"], economics=case.scope["economics"])
    support, support_valid = horizons.build_masked_support_forecast_targets(frame=source["frame"],
        scope=case.scope, split="train", admitted_mask=source["first_valid"])
    forecast = np.concatenate((source["frame"][list(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS)].to_numpy(dtype=np.float32), support), axis=1)
    valid = np.concatenate((targets.require_m5_forecast_observation_storage(source["frame"], context="TEST"), support_valid), axis=1)
    batches = list(cli._target_batches({"chunk_rows": 2, "progress_interval_batches": 1},
        np.arange(3, dtype=np.int64), provider, forecast, valid, time.monotonic()))
    assert [parents.tolist() for parents, _ in batches] == [[0, 1], [2]]
    expected_keys = {"dynamic_terminal_net_markout_bps", "dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps",
        "dynamic_closed_mfe_bps", "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred"}
    for parents, surfaces in batches:
        assert set(surfaces) == expected_keys
        original = provider.targets_for_rows(parents)
        for name, key in (
            ("dynamic_terminal_net_markout_bps", observed.DYNAMIC_M5_MARKOUT_TARGET_KEY),
            ("dynamic_mfe_bps", observed.DYNAMIC_M5_MFE_TARGET_KEY),
            ("dynamic_pre_mfe_mae_upper_bps", observed.DYNAMIC_M5_MAE_TARGET_KEY),
            ("dynamic_closed_mfe_bps", observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY),
            ("dynamic_closed_pre_mfe_mae_upper_bps", observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY),
        ):
            np.testing.assert_array_equal(surfaces[name][0], original[key])
        np.testing.assert_array_equal(surfaces["dynamic_terminal_net_markout_bps"][1], original[observed.DYNAMIC_M5_MARKOUT_VALID_KEY])
        np.testing.assert_array_equal(surfaces["dynamic_mfe_bps"][1], original[observed.DYNAMIC_M5_PATH_VALID_KEY])
        np.testing.assert_array_equal(surfaces["dynamic_closed_pre_mfe_mae_upper_bps"][1], original[observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY])
        np.testing.assert_array_equal(surfaces["forecast_pred"][0], forecast[parents])
        np.testing.assert_array_equal(surfaces["forecast_pred"][1], valid[parents])
        for name, (target, mask) in surfaces.items():
            assert isinstance(target, np.ndarray) and isinstance(mask, np.ndarray)
            assert target.dtype == np.float32 and mask.dtype == np.bool_
            shape = (len(parents), 10) if name == "forecast_pred" else (len(parents), 480, 3 if name == "dynamic_terminal_net_markout_bps" else 2)
            assert target.shape == mask.shape == shape
            assert np.isfinite(target).all() and np.all(target[~mask] == 0)
