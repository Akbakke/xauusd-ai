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


def test_horizon_eligible_rows_are_complete_rows_at_or_before_the_measured_cutoff(tmp_path):
    from gx1.contracts.entry_observed_market_horizons_v1 import HORIZON_LABEL_COLUMNS
    frame = pd.DataFrame({"time": pd.date_range("2025-04-01T00:00:00Z", periods=6, freq="5min")})
    for column in HORIZON_LABEL_COLUMNS:
        frame[column] = np.arange(6, dtype=np.float64) + 1.0
    frame.loc[2, HORIZON_LABEL_COLUMNS[0]] = np.nan  # censored inside the cutoff: excluded, counted
    frame.loc[5, HORIZON_LABEL_COLUMNS[3]] = np.nan  # after the cutoff: censored by construction
    path = tmp_path / "train.parquet"
    frame.to_parquet(path, index=False)
    cutoff = str(frame["time"].iloc[4] + pd.Timedelta(minutes=5))
    found = producer.horizon_eligible_rows(path, cutoff_utc=cutoff)
    assert found["rows"].tolist() == [0, 1, 3, 4] and found["censored_rows_excluded"] == 1
    assert found["rows_after_cutoff"] == 1 and found["physical_rows"] == 6
    frame.loc[5, HORIZON_LABEL_COLUMNS[3]] = 9.0
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="COMPLETE_ROW_AFTER_CUTOFF"):
        producer.horizon_eligible_rows(path, cutoff_utc=cutoff)


def test_producer_publishes_v2_coordinates_over_the_horizon_eligible_subset(physical_measurement, tmp_path, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    c = physical_measurement
    output = tmp_path / "published_v2"
    full_order = c.prepared.case["order"]
    eligible = np.sort(full_order[1:]).astype(np.int64)  # drop one row: a censored entry
    monkeypatch.setattr(producer, "horizon_eligible_rows", lambda path, *, cutoff_utc: {
        "rows": eligible, "censored_rows_excluded": 1, "rows_after_cutoff": 0, "physical_rows": len(full_order)})
    design = _write(tmp_path / "horizons_design.json", {"schema_version": "fixture"})
    args = {**_arguments(c, output), "horizon_eligibility": {"horizons_design": design, "entry_eligibility_cutoff_utc": "2025-05-02T19:20:00Z"}}
    result = producer.materialize(**args, publish=True)
    native = json.loads(Path(result["chronological_prefix"]["native_coordinates"]["path"]).read_text())
    assert native["schema_version"] == campaign.PHYSICAL_COORDINATES_SCHEMA_V2
    assert native["eligibility"]["rule"] == campaign.HORIZON_ELIGIBILITY_RULE and native["eligibility"]["eligible_rows"] == len(eligible)
    assert native["eligibility"]["censored_rows_excluded"] == 1 and native["eligibility"]["horizons_design"] == design
    published_eligible = np.load(native["bindings"]["TRAIN_ELIGIBLE_PARENT_ROWS"]["path"])
    order = np.load(native["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"]["path"])
    assert published_eligible.tolist() == eligible.tolist()
    assert order.tolist() == full_order[np.isin(full_order, eligible)].tolist()
    # the frozen contract accepts the subset and the trainer's replay semantics (filtered full order)
    prefix = dict(result["chronological_prefix"])
    design_value = json.loads(Path(prefix["design"]["path"]).read_text())
    physical = campaign.require_physical_chronological_preprocessing(
        {k: v for k, v in prefix.items() if k != "native_coordinates"}, design=design_value,
        normalization=json.loads(Path(prefix["normalization_result"]["path"]).read_text()),
        labels=json.loads(Path(prefix["labels_result"]["path"]).read_text()))
    checked = campaign.require_physical_native_training_coordinates(prefix, design=design_value, physical=physical)
    assert checked["eligible_parent_rows"].tolist() == eligible.tolist()
    with pytest.raises(RuntimeError, match="HORIZON_ELIGIBILITY_INVALID"):
        producer.materialize(**{**args, "horizon_eligibility": {"horizons_design": design}, "output_dir": tmp_path / "x"}, publish=False)


def _m5_path_frame():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN
    frame = pd.DataFrame({'time': pd.date_range('2025-04-01T00:00Z', periods=6, freq='5min')})
    frame[M5_PATH_VALID_COLUMN] = [True, True, False, True, True, False]
    for column in M5_PATH_TARGET_COLUMNS:
        frame[column] = [1., -2., np.nan, 0., 3., np.nan]
    return frame


def test_m5_path_eligibility_ignores_absent_long_targets_and_uses_only_availability(tmp_path):
    from gx1.contracts.entry_observed_market_horizons_v1 import HORIZON_LABEL_COLUMNS
    frame = _m5_path_frame()
    path = tmp_path / 'train.parquet'
    cutoff = '2025-04-01T02:00Z'
    frame.to_parquet(path, index=False)
    found = producer.m5_path_eligible_rows(path, cutoff_utc=cutoff)
    assert found['rows'].tolist() == [0, 1, 3, 4]
    assert found['physical_rows'] == 6 and found['censored_rows_excluded'] == 2
    # Every long-horizon cell is missing, yet exactly the same M5 population remains.
    for column in HORIZON_LABEL_COLUMNS:
        frame[column] = np.nan
    frame.to_parquet(path, index=False)
    again = producer.m5_path_eligible_rows(path, cutoff_utc=cutoff)
    assert np.array_equal(found['rows'], again['rows'])


@pytest.mark.parametrize('fault', ['mask', 'partial', 'cross_cutoff', 'after_cutoff'])
def test_m5_path_eligibility_rejects_corrupt_availability_or_boundary(tmp_path, fault):
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN
    frame = _m5_path_frame()
    cutoff = '2025-04-01T02:00Z'
    if fault == 'mask':
        frame.loc[0, M5_PATH_VALID_COLUMN] = False
    elif fault == 'partial':
        frame.loc[0, M5_PATH_TARGET_COLUMNS[0]] = np.nan
    elif fault == 'cross_cutoff':
        cutoff = '2025-04-01T01:40Z'
    else:
        cutoff = '2025-04-01T00:10Z'
    path = tmp_path / 'bad.parquet'
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match='AVAILABILITY_INVALID|CROSSES_CUTOFF|COMPLETE_ROW_AFTER_CUTOFF'):
        producer.m5_path_eligible_rows(path, cutoff_utc=cutoff)


def test_m5_path_coordinates_preserve_filtered_native_order_and_reject_long_gating(physical_measurement, tmp_path, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
    c = physical_measurement
    full_order = c.prepared.case['order']
    eligible = np.sort(full_order[1:]).astype(np.int64)
    monkeypatch.setattr(producer, 'm5_path_eligible_rows', lambda path, *, cutoff_utc: {
        'rows': eligible, 'censored_rows_excluded': 1, 'rows_after_cutoff': 0, 'physical_rows': len(full_order)})
    args = {**_arguments(c, tmp_path/'path'), 'm5_path_eligibility': {
        'target_contract': entry_observed_m5_path_contract(),
        'entry_eligibility_cutoff_utc': c.prepared.design['calendar']['train_control_cutoff']}}
    result = producer.materialize(**args, publish=True)
    prefix = result['chronological_prefix']
    native_path = Path(prefix['native_coordinates']['path'])
    native = json.loads(native_path.read_text())
    assert native['schema_version'] == campaign.PHYSICAL_COORDINATES_SCHEMA_M5_PATH
    assert native['eligibility']['long_horizons_gate_primary_eligibility'] is False
    assert np.array_equal(np.load(native['bindings']['TRAIN_NATIVE_EPOCH0_ORDER']['path']), full_order[np.isin(full_order, eligible)])
    design = json.loads(Path(prefix['design']['path']).read_text())
    physical = campaign.require_physical_chronological_preprocessing(
        {k:v for k,v in prefix.items() if k != 'native_coordinates'}, design=design,
        normalization=json.loads(Path(prefix['normalization_result']['path']).read_text()),
        labels=json.loads(Path(prefix['labels_result']['path']).read_text()))
    for field, value in [('long_horizons_gate_primary_eligibility', True), ('target_contract', {}),
                         ('training_phase', 'entry_only_observed_horizon_20d'), ('censored_rows_excluded', 0)]:
        bad = copy.deepcopy(native)
        bad['eligibility'][field] = value
        bad['coordinates_sha256'] = campaign.native_sha256({k:v for k,v in bad.items() if k != 'coordinates_sha256'})
        bad_binding = _write(tmp_path/(field+'.json'), bad)
        with pytest.raises(RuntimeError, match='M5_PATH_ELIGIBILITY_INVALID|ROWS_INVALID'):
            campaign.require_physical_native_training_coordinates({**prefix, 'native_coordinates': bad_binding}, design=design, physical=physical)
    reordered = np.load(native['bindings']['TRAIN_NATIVE_EPOCH0_ORDER']['path']).copy()[::-1]
    bad_order = tmp_path/'bad_order.npy'
    np.save(bad_order, reordered, allow_pickle=False)
    bad = copy.deepcopy(native)
    bad['bindings']['TRAIN_NATIVE_EPOCH0_ORDER'] = _binding(bad_order)
    bad['coordinates_sha256'] = campaign.native_sha256({k:v for k,v in bad.items() if k != 'coordinates_sha256'})
    bad_binding = _write(tmp_path/'bad_order.json', bad)
    with pytest.raises(RuntimeError, match='M5_PATH_NATIVE_ORDER_MISMATCH'):
        campaign.require_physical_native_training_coordinates({**prefix, 'native_coordinates': bad_binding}, design=design, physical=physical)
    with pytest.raises(RuntimeError, match='M5_PATH_ELIGIBILITY_INVALID'):
        producer.materialize(**{**args, 'output_dir': tmp_path/'mixed', 'horizon_eligibility': {
            'horizons_design': prefix['design'], 'entry_eligibility_cutoff_utc': args['m5_path_eligibility']['entry_eligibility_cutoff_utc']}}, publish=False)
