"""Synthetic clocks and real native coordinate publication; no market/model work."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_unified_exit_chronological_coordinates_v1 as producer
from gx1.contracts import unified_exit_bounded_val_cohort_v1 as owner
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract, schedule_random_access_full_population_epoch)
from tests.test_chronological_measurement_binding import (
    physical_component_templates, physical_component_case, physical_prepared,
    physical_measurement, _binding, _write)


@pytest.mark.parametrize("population,chunk", [(17,4),(18,6),(19,19)])
def test_published_epoch_order_matches_native_full_population_schedule(population, chunk):
    sampler = build_random_access_sampler_contract(split="train", source_lineage_sha256="a"*64,
        transition_budget_per_epoch=chunk*4, transitions_per_entry=4, entry_pair_population=population)
    _, anchors, metadata = schedule_random_access_full_population_epoch(
        sampler_contract=sampler, epoch_index=0, successor_transition_count_by_entry=[3]*population)
    order = producer._epoch0_order(sampler)
    assert order.tolist() == [r["entry_row_index"] for r in anchors]
    assert metadata["every_entry_pair_exactly_once"] is True


def _arguments(case, output):
    return {"chronological_prefix":{key:value for key,value in case.prepared.value.items() if key!="native_coordinates"},
            "selected_sampler":case.prepared.case["coordinates"]["selected_sampler"], "output_dir":output}


def test_producer_publishes_exact_frozen_coordinates_and_refuses_overwrite(physical_measurement, tmp_path):
    c = physical_measurement; output = tmp_path/"published"
    args = _arguments(c, output)
    dry = producer.materialize(**args,publish=False)
    assert dry["published"] is False and not output.exists()
    result = producer.materialize(**args,publish=True)
    complete = json.loads(Path(result["completion"]["path"]).read_text())
    assert complete["decision"] == "NATIVE_AND_MEASUREMENT_COORDINATES_FROZEN_NO_LAUNCH_AUTHORITY"
    assert complete["test_data_used"] is False and complete["optimizer_steps"] == complete["model_forwards"] == 0
    native = json.loads(Path(result["chronological_prefix"]["native_coordinates"]["path"]).read_text())
    assert np.load(native["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"]["path"]).tolist() == c.prepared.case["order"].tolist()
    assert native["control_parent_rows"] == c.prepared.design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"]
    coordinates = json.loads(Path(result["measurement_coordinates"]["path"]).read_text())
    for role in ("train","control"):
        cohort = owner.build_chronological_measurement_cohort(
            result["chronological_prefix"]["design"],result["measurement_coordinates"],role=role)
        with np.load(coordinates["coordinates"][role]["path"]) as arrays:
            for key,values in c.arrays[role].items():
                np.testing.assert_array_equal(arrays[key],values)
        assert owner.require_bounded_val_cohort(cohort) == cohort
    before = {str(path):path.read_bytes() for path in output.iterdir()}
    for publish in (False,True):
        with pytest.raises(RuntimeError,match="OUTPUT_EXISTS"):
            producer.materialize(**args,publish=publish)
        assert before == {str(path):path.read_bytes() for path in output.iterdir()}


def test_failed_control_production_cannot_publish_completion_or_measurement_authority(
    physical_measurement,tmp_path,monkeypatch,
):
    output = tmp_path/"failed"
    original = producer._physical_measurement_arrays
    def fail_control(inputs):
        if inputs["source_split"] == "val": raise RuntimeError("injected control failure")
        return original(inputs)
    monkeypatch.setattr(producer,"_physical_measurement_arrays",fail_control)
    with pytest.raises(RuntimeError,match="injected control failure"):
        producer.materialize(**_arguments(physical_measurement,output),publish=True)
    assert (output/"NATIVE_COORDINATES.json").is_file()
    assert (output/"TRAIN_MEASUREMENT.npz").is_file()
    assert not (output/"MEASUREMENT_COORDINATES.json").exists()
    assert not (output/"COMPLETE.json").exists()


def test_publication_rejects_race_and_preserves_failed_stage(tmp_path,monkeypatch):
    output = tmp_path/"result.json"
    original = producer._publish_file_noreplace
    def race(stage, final):
        final.write_bytes(b"existing evidence")
        original(stage,final)
    monkeypatch.setattr(producer,"_publish_file_noreplace",race)
    with pytest.raises((RuntimeError,FileExistsError)):
        producer._publish(output, {"new":True})
    assert output.read_bytes() == b"existing evidence"
    assert len(list(tmp_path.glob(".result.json.*"))) == 1


def _support_inputs(tmp_path):
    from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
    times = pd.date_range("2025-01-03T00:00Z",periods=205,freq="min").asi8.copy()
    times[100:] += 2*24*60*60_000_000_000  # Real row clock includes a two-day gap.
    source = tmp_path/"m1.parquet"
    pd.DataFrame({"time":pd.to_datetime(times,utc=True)}).to_parquet(source,index=False)
    m1 = _binding(source)
    mb = _write(tmp_path/"m1.json",{"split":"train","decision":"PASS","test_accessed":False,
                                  "output_parquet_sha256":m1["sha256"]})
    frame = pd.DataFrame({"entry_row_index":[0,1],"entry_time_ns":times[[0,1]]-300_000_000_000,
                          "child_m1_start_row":[0,1],"first_state_time_ns":times[[0,1]],
                          "successor_transition_count":[200,3]})
    inputs={"manifest":{"source_bindings":{"m1_child":m1,"m1_child_manifest":mb}},
            "source_split":"train","expected":np.array([0,1]),"frame":frame,
            "expected_samples":[[0,90,199,90],[0,1,2,2]],
            "design":{"targets":{"reference_policy":reference_policy_contract()}}}
    return inputs,times


def test_reference_support_uses_actual_clock_gaps_horizon_and_censored_boundary(tmp_path):
    inputs,times = _support_inputs(tmp_path)
    arrays,_ = owner._physical_measurement_arrays(inputs)
    assert arrays["sampled_reference_end_close_ns"].tolist() == [
        (times[[120,200,200,200]]+60_000_000_000).tolist(),
        [int(times[4])+60_000_000_000]*4]
    assert arrays["anchor_reference_end_close_ns"].tolist() == (times[[120,4]]+60_000_000_000).tolist()
    assert arrays["sampled_reference_end_close_ns"][0,0] != times[0]+121*60_000_000_000


@pytest.mark.parametrize("fault",["wrong_split","stale_m1","wrong_first_state","outside_m1","no_successor"])
def test_reference_support_rejects_m1_binding_or_coordinate_fault(tmp_path,fault):
    inputs,_ = _support_inputs(tmp_path)
    sources=inputs["manifest"]["source_bindings"]
    if fault=="wrong_split":
        path=Path(sources["m1_child_manifest"]["path"]);value=json.loads(path.read_text());value["split"]="val"
        sources["m1_child_manifest"]=_write(path,value)
    elif fault=="stale_m1":
        path=Path(sources["m1_child"]["path"]);path.write_bytes(path.read_bytes()+b"changed")
    elif fault=="wrong_first_state":inputs["frame"].loc[0,"first_state_time_ns"]+=60_000_000_000
    elif fault=="outside_m1":inputs["frame"].loc[0,"successor_transition_count"]=999
    else:inputs["expected_samples"][0][0]=200
    with pytest.raises(RuntimeError):
        owner._physical_measurement_arrays(inputs)
