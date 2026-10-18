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


def _dynamic_m5_pointer_frame(rows=6):
    frame = pd.DataFrame({"time": pd.date_range("2025-04-01T00:00Z", periods=rows, freq="5min"),
        "dynamic_entry_m1_row": np.arange(rows, dtype=np.int64) * 5,
        "dynamic_first_prefix_observed": np.ones(rows, dtype=bool),
        "dynamic_first_terminal_observed_long": np.ones(rows, dtype=bool),
        "dynamic_first_terminal_observed_short": np.ones(rows, dtype=bool),
        "dynamic_first_path_observed_long": np.zeros(rows, dtype=bool),
        "dynamic_first_path_observed_short": np.zeros(rows, dtype=bool),
        "dynamic_first_terminal_censor_reason": ["observed"] * rows,
        "dynamic_first_path_censor_reason": ["unknown_source_absence"] * rows})
    # A genuine closed-only prefix may be observed while its next quote is absent.
    frame.loc[1, ["dynamic_first_terminal_observed_long", "dynamic_first_terminal_observed_short"]] = False
    frame.loc[1, ["dynamic_first_path_observed_long", "dynamic_first_path_observed_short"]] = True
    frame.loc[1, "dynamic_first_terminal_censor_reason"] = "terminal_quote_missing"
    frame.loc[1, "dynamic_first_path_censor_reason"] = "observed"
    return frame


def test_dynamic_coordinate_eligibility_keeps_terminal_or_closed_prefix_without_95min_or_long_union(tmp_path):
    frame = _dynamic_m5_pointer_frame()
    path = tmp_path / "dynamic.parquet"
    frame.to_parquet(path, index=False)
    found = producer.dynamic_m5_eligible_rows(path, cutoff_utc="2025-04-01T00:40Z")
    np.testing.assert_array_equal(found["rows"], np.arange(6, dtype=np.int64))
    assert found["censored_rows_excluded"] == found["rows_after_cutoff"] == 0
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN
    frame[M5_PATH_VALID_COLUMN] = False
    for column in M5_PATH_TARGET_COLUMNS: frame[column] = np.nan
    from gx1.contracts.entry_observed_market_horizons_v1 import HORIZON_LABEL_COLUMNS
    for column in HORIZON_LABEL_COLUMNS: frame[column] = np.nan
    frame.to_parquet(path, index=False)
    again = producer.dynamic_m5_eligible_rows(path, cutoff_utc="2025-04-01T00:40Z")
    np.testing.assert_array_equal(found["rows"], again["rows"])


@pytest.mark.parametrize("fault", ["float_mask", "or_mask", "pointer", "reason", "terminal_at_cutoff", "path_after_cutoff"])
def test_dynamic_coordinate_eligibility_rejects_invented_observation_or_boundary(tmp_path, fault):
    frame = _dynamic_m5_pointer_frame()
    cutoff = "2025-04-01T00:40Z"
    if fault == "float_mask": frame["dynamic_first_prefix_observed"] = frame["dynamic_first_prefix_observed"].astype(float)
    elif fault == "or_mask": frame.loc[1, "dynamic_first_prefix_observed"] = False
    elif fault == "pointer": frame.loc[1, "dynamic_entry_m1_row"] = -1
    elif fault == "reason": frame.loc[1, "dynamic_first_path_censor_reason"] = "terminal_quote_missing"
    elif fault == "terminal_at_cutoff": cutoff = "2025-04-01T00:31Z"
    elif fault == "path_after_cutoff":
        frame.loc[5, ["dynamic_first_terminal_observed_long", "dynamic_first_terminal_observed_short"]] = False
        frame.loc[5, ["dynamic_first_path_observed_long", "dynamic_first_path_observed_short"]] = True
        frame.loc[5, "dynamic_first_terminal_censor_reason"] = "terminal_quote_missing"
        frame.loc[5, "dynamic_first_path_censor_reason"] = "observed"
        cutoff = "2025-04-01T00:30Z"
    path = tmp_path / "bad-dynamic.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_"):
        producer.dynamic_m5_eligible_rows(path, cutoff_utc=cutoff)


def test_dynamic_coordinate_closed_prefix_at_split_close_does_not_require_terminal_quote(tmp_path):
    frame = _dynamic_m5_pointer_frame(rows=2)
    path = tmp_path / "closed-prefix.parquet"
    frame.to_parquet(path, index=False)
    found = producer.dynamic_m5_eligible_rows(path, cutoff_utc="2025-04-01T00:11Z")
    assert found["rows"].tolist() == [0, 1]


def _dynamic_coordinate_case(tmp_path, *, rows=273):
    """Use the real existing single-epoch witness owner and real pointer/order files."""
    from tests.test_native_prefix_recipe import _dynamic_single_epoch_design
    from gx1.contracts.entry_observed_market_horizons_v1 import M5_SINGLE_EPOCH_SCOPE_KEY
    design, witness = _dynamic_single_epoch_design(tmp_path, rows=rows)
    train_path = Path(witness["train_parquet"]["path"])
    _dynamic_m5_pointer_frame(rows).to_parquet(train_path, index=False)
    witness["train_parquet"] = _binding(train_path)
    witness_binding = _write(tmp_path / "bound-dynamic-population.json", witness)
    design["scope"][M5_SINGLE_EPOCH_SCOPE_KEY] = witness_binding
    control_path = tmp_path / "dynamic-control.parquet"
    _dynamic_m5_pointer_frame(rows).to_parquet(control_path, index=False)
    control_manifest = _write(tmp_path / "dynamic-control.manifest.json", {"rows": rows})
    control_rows_path = tmp_path / "dynamic-control-parents.npy"
    np.save(control_rows_path, np.arange(256, dtype=np.int64), allow_pickle=False)
    physical = {"physical_sources": {
        "train": {"parquet": witness["train_parquet"], "manifest": witness["train_manifest"], "physical_rows": rows},
        "val": {"parquet": _binding(control_path), "manifest": control_manifest, "physical_rows": rows}},
        "train_rows": rows, "train_parent_rows": witness["eligible_parent_rows"],
        "control_parent_rows": _binding(control_rows_path)}
    design["calendar"].update(source_bindings=physical["physical_sources"],
        bindings={"CONTROL256_PARENT_ROWS": physical["control_parent_rows"]})
    design["budget"]["train_only_probe_entries"] = 256
    design["selection"] = {"seed": 20260911}
    prefix = {"design": _write(tmp_path / "dynamic-coordinate-design.json", design),
        "normalization_result": _write(tmp_path / "qualified-normalization.json", {"qualified_by_separate_owner": True}),
        "labels_result": _write(tmp_path / "qualified-labels.json", {"qualified_by_separate_owner": True})}
    return design, witness, witness_binding, prefix, physical


def _dynamic_coordinate_arguments(case, output):
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    design, witness, witness_binding, prefix, physical = case
    return {"chronological_prefix": prefix, "dynamic_population_witness": witness_binding,
        "dynamic_m5_eligibility": {"target_contract": entry_observed_dynamic_m5_path_contract(),
            "entry_eligibility_cutoff_utc": design["calendar"]["train_control_cutoff"]}, "output_dir": output}


def test_dynamic_coordinate_publication_uses_real_witness_and_never_exit_measurements(tmp_path, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts import unified_exit_selected_sampler_v1 as selected_owner
    case = _dynamic_coordinate_case(tmp_path)
    design, witness, witness_binding, prefix, physical = case
    # Full physical normalization/target admission has its own bounded tests.
    # The actual sampler/witness/coordinate/order chain below remains unstubbed.
    monkeypatch.setattr(producer, "require_physical_chronological_preprocessing", lambda *a, **k: physical)
    def forbidden(*args, **kwargs):
        raise AssertionError("dynamic native coordinates must not enter Exit machinery")
    monkeypatch.setattr(selected_owner, "require_selected_sampler_artifact", forbidden)
    monkeypatch.setattr(producer, "_physical_measurement_inputs", forbidden)
    monkeypatch.setattr(producer, "_physical_measurement_arrays", forbidden)
    output = tmp_path / "dynamic-native-only"
    args = _dynamic_coordinate_arguments(case, output)
    dry = producer.materialize(**args, publish=False)
    assert dry["entry_measurement_complete"] is False and not output.exists()
    result = producer.materialize(**args, publish=True)
    assert result["native_coordinates_complete"] is True
    assert result["entry_measurement_complete"] is result["native_launch_admitted"] is False
    publication = json.loads(Path(result["native_coordinate_result"]["path"]).read_text())
    assert publication["population_witness"] == witness_binding
    assert publication["model_forwards"] == publication["optimizer_steps"] == publication["fits"] == 0
    assert not (output / "COMPLETE.json").exists()
    assert not (output / "MEASUREMENT_COORDINATES.json").exists()
    assert not list(output.glob("*_MEASUREMENT.npz"))
    native = json.loads(Path(result["chronological_prefix"]["native_coordinates"]["path"]).read_text())
    assert native["schema_version"] == campaign.PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH
    assert "selected_sampler" not in native and "selected_sampler" not in publication
    checked = campaign.require_physical_native_training_coordinates(
        result["chronological_prefix"], design=design, physical=physical)
    assert checked["population_witness"] == witness_binding
    assert checked["native_sampler_contract"] == witness["native_sampler_contract"]
    assert "selected_sampler_binding" not in checked and "selected_sampler" not in checked
    np.testing.assert_array_equal(checked["epoch0_parent_order"],
                                 np.load(witness["epoch0_parent_order"]["path"]))
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        producer.materialize(**args, publish=True)


def test_dynamic_coordinate_validator_requires_new_identity_full_population_and_exact_native_order(tmp_path, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    case = _dynamic_coordinate_case(tmp_path)
    design, witness, witness_binding, prefix, physical = case
    monkeypatch.setattr(producer, "require_physical_chronological_preprocessing", lambda *a, **k: physical)
    result = producer.materialize(**_dynamic_coordinate_arguments(case, tmp_path / "native"), publish=True)
    prefix = result["chronological_prefix"]
    native = json.loads(Path(prefix["native_coordinates"]["path"]).read_text())
    def binding(value, suffix):
        value["coordinates_sha256"] = campaign.native_sha256({k: v for k, v in value.items() if k != "coordinates_sha256"})
        return _write(tmp_path / (suffix + ".json"), value)
    for field, value in (("training_phase", "entry_only_observed_m5_path"), ("emission_scope", "m5_primary_observed"),
            ("target_contract", {}), ("later_horizons_gate_primary_eligibility", True), ("first_prefix_horizon_minutes", 95)):
        bad = copy.deepcopy(native)
        bad["eligibility"][field] = value
        with pytest.raises(RuntimeError, match="DYNAMIC_M5_ELIGIBILITY_INVALID"):
            campaign.require_physical_native_training_coordinates({**prefix, "native_coordinates": binding(bad, field)}, design=design, physical=physical)
    bad = copy.deepcopy(native)
    op = tmp_path / "reversed-native-order.npy"
    np.save(op, np.load(witness["epoch0_parent_order"]["path"])[::-1], allow_pickle=False)
    bad["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"] = _binding(op)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_NATIVE_ORDER_MISMATCH"):
        campaign.require_physical_native_training_coordinates({**prefix, "native_coordinates": binding(bad, "reordered")}, design=design, physical=physical)
    bad = copy.deepcopy(native)
    bad["selected_sampler"] = witness_binding
    with pytest.raises(RuntimeError, match="DYNAMIC_EXIT_SAMPLER_FORBIDDEN"):
        campaign.require_physical_native_training_coordinates({**prefix, "native_coordinates": binding(bad, "exit-mixed")}, design=design, physical=physical)


@pytest.mark.parametrize("fault", ["missing", "exit_mixed", "other_path", "source", "order"])
def test_dynamic_sampler_requires_exact_real_population_source_and_order(tmp_path, fault):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts.entry_observed_market_horizons_v1 import M5_SINGLE_EPOCH_SCOPE_KEY
    design, witness, witness_binding, prefix, physical = _dynamic_coordinate_case(tmp_path)
    selected = None
    if fault == "missing": witness_binding = None
    elif fault == "exit_mixed": selected = witness_binding
    elif fault == "other_path": witness_binding = _write(tmp_path / "identical-witness-copy.json", witness)
    elif fault == "source": physical["physical_sources"]["train"]["parquet"] = {**witness["train_parquet"], "sha256": "b" * 64}
    elif fault == "order":
        op = tmp_path / "witness-wrong-order.npy"
        np.save(op, np.load(witness["epoch0_parent_order"]["path"])[::-1], allow_pickle=False)
        witness["epoch0_parent_order"] = _binding(op)
        witness_binding = _write(tmp_path / "wrong-order-witness.json", witness)
        design["scope"][M5_SINGLE_EPOCH_SCOPE_KEY] = witness_binding
    with pytest.raises(RuntimeError, match="DYNAMIC_POPULATION_"):
        campaign._physical_native_sampler(selected, artifacts=prefix, design=design, physical=physical,
                                          dynamic_population_witness=witness_binding)


def test_dynamic_native_coordinates_do_not_grant_fresh_campaign_admission(tmp_path):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    case = _dynamic_coordinate_case(tmp_path)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_MEASUREMENT_AND_LAUNCH_ADMISSION_PENDING"):
        campaign.require_fresh_physical_campaign_sampler({"chronological_prefix": case[3]})
