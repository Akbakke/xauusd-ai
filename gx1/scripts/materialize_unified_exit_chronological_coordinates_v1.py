"""Publish frozen native TRAIN order and TRAIN/CONTROL measurement coordinates.

This offline producer uses existing preprocessing, index and measured sampler
authorities. It never fits targets, initializes a model or grants native launch.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_file_noreplace
from gx1.contracts.local_random_access_campaign_v2 import file_sha256, read_bound_json, require_binding
from gx1.contracts.entry_model_native_training_run_lineage_v1 import deterministic_uniform_subsample_indices
from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
    HORIZON_ELIGIBILITY_RULE, PHYSICAL_COORDINATES_SCHEMA_V1, PHYSICAL_COORDINATES_SCHEMA_V2,
    M5_PATH_ELIGIBILITY_RULE, PHYSICAL_COORDINATES_SCHEMA_M5_PATH,
    DYNAMIC_M5_PATH_ELIGIBILITY_RULE, PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH,
    native_sha256, require_physical_chronological_preprocessing,
    require_physical_native_training_coordinates, _physical_native_sampler,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract, schedule_random_access_entry_anchors,
)
from gx1.contracts.unified_exit_bounded_val_cohort_v1 import (
    _physical_measurement_inputs, _physical_measurement_arrays,
    build_chronological_measurement_cohort,
)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _read(binding):
    checked = require_binding(binding, label="coordinate producer input", verify_file=True)
    return read_bound_json(Path(checked["path"]), checked["sha256"])


def _epoch0_order(sampler) -> np.ndarray:
    """Use native anchor chunks without materializing full-population transitions."""
    population = sampler["entry_pair_population"]
    chunk_size = sampler["entry_pairs_per_epoch"]
    order = np.empty(population, dtype=np.int64)
    for start in range(0, population, chunk_size):
        anchors = schedule_random_access_entry_anchors(
            sampler_contract=sampler, epoch_index=start // chunk_size)
        count = min(chunk_size, population - start)
        order[start:start+count] = [row["entry_row_index"] for row in anchors[:count]]
    if not np.array_equal(np.sort(order), np.arange(population, dtype=np.int64)):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_NATIVE_ORDER_INVALID")
    return order


def _publish(path: Path, value: Any, *, validate: Callable[[Path], None] | None = None):
    """Strict-load staged bytes, then publish without replacing any prior file."""
    if path.exists() or path.is_symlink():
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_OUTPUT_EXISTS")
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    stage = Path(name)
    # Failed staging is retained for the retention owner, never deleted here.
    with os.fdopen(fd, "wb") as stream:
        if path.suffix == ".npy":
            np.save(stream, value, allow_pickle=False)
        elif path.suffix == ".npz":
            np.savez(stream, **value)
        else:
            stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    if path.suffix == ".npy":
        observed = np.load(stage, allow_pickle=False)
        if observed.dtype != value.dtype or not np.array_equal(observed, value):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_STAGING_INVALID")
    elif path.suffix == ".npz":
        with np.load(stage, allow_pickle=False) as observed:
            if set(observed.files) != set(value) or any(
                    observed[key].dtype != item.dtype or not np.array_equal(observed[key], item)
                    for key, item in value.items()):
                raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_STAGING_INVALID")
    elif json.loads(stage.read_text()) != value:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_STAGING_INVALID")
    if validate is not None:
        validate(stage)
    _publish_file_noreplace(stage, path)
    _fsync_directory(path.parent)
    return _binding(path)


def horizon_eligible_rows(parquet_path: Path, *, cutoff_utc: str) -> dict[str, Any]:
    """TRAIN rows with complete observed outcomes at every trading-day horizon whose decision lies at
    or before the measured cutoff (operator decision 12.10.2026). Only outcome *availability* is read,
    never an outcome value; a complete row after the cutoff contradicts the measured design."""
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
    from gx1.contracts.entry_observed_market_horizons_v1 import HORIZON_LABEL_COLUMNS
    frame = pq.read_table(parquet_path, columns=["time", *HORIZON_LABEL_COLUMNS]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    eligible_time = clock + ENTRY_DECISION_BAR_SECONDS * 10**9 <= pd.Timestamp(cutoff_utc).value
    complete = np.isfinite(frame[HORIZON_LABEL_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    if bool((complete & ~eligible_time).any()):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_COMPLETE_ROW_AFTER_CUTOFF")
    rows = np.flatnonzero(eligible_time & complete).astype(np.int64)
    if len(rows) < 1:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_NO_ELIGIBLE_ROWS")
    return {"rows": rows, "censored_rows_excluded": int((eligible_time & ~complete).sum()),
            "rows_after_cutoff": int((~eligible_time).sum()), "physical_rows": int(len(frame))}


def m5_path_eligible_rows(parquet_path: Path, *, cutoff_utc: str) -> dict[str, Any]:
    """Primary M5 availability only; no1/5/20d label is read or required."""
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_causal_m1_outcomes_v1 import (
        M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN, M5_PATH_OBSERVATION_HORIZON_BARS,
    )
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
    from gx1.time.session_detector import m5_decision_availability
    frame = pq.read_table(parquet_path, columns=["time", M5_PATH_VALID_COLUMN, *M5_PATH_TARGET_COLUMNS]).to_pandas()
    decisions = m5_decision_availability(frame["time"]).as_unit("ns").asi8
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tz is None:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_CUTOFF_INVALID")
    valid = frame[M5_PATH_VALID_COLUMN].to_numpy()
    finite = np.isfinite(frame[list(M5_PATH_TARGET_COLUMNS)].to_numpy(dtype=np.float64))
    complete = finite.all(axis=1)
    if (valid.dtype != np.dtype(bool) or np.any(finite.any(axis=1) != complete)
            or not np.array_equal(complete, valid)):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_AVAILABILITY_INVALID")
    eligible_time = decisions <= cutoff.value
    if bool((complete & ~eligible_time).any()):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_COMPLETE_ROW_AFTER_CUTOFF")
    if bool((complete & (decisions + M5_PATH_OBSERVATION_HORIZON_BARS * ENTRY_DECISION_BAR_SECONDS * 10**9 > cutoff.value)).any()):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_TARGET_CROSSES_CUTOFF")
    rows = np.flatnonzero(eligible_time & complete).astype(np.int64)
    if len(rows) < 1:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_NO_ELIGIBLE_ROWS")
    return {"rows": rows, "censored_rows_excluded": int((eligible_time & ~complete).sum()),
        "rows_after_cutoff": int((~eligible_time).sum()), "physical_rows": int(len(frame))}



def dynamic_m5_eligible_rows(parquet_path: Path, *, cutoff_utc: str) -> dict[str, Any]:
    """Replay only declared first-prefix pointer availability; never fixed95/legacy tails.

    Completed source-bound receipt validation owns full physical masks/clock
    proofs; the provider owns quote reconstruction. This narrow selection does
    not turn a closed-prefix path into terminal-inclusive path supervision.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_model_native_aux_targets_v3 import DYNAMIC_M5_POINTER_COLUMNS
    from gx1.time.session_detector import m5_decision_availability
    frame = pq.read_table(parquet_path, columns=["time", *DYNAMIC_M5_POINTER_COLUMNS]).to_pandas()
    times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="raise")).as_unit("ns")
    cutoff = pd.Timestamp(cutoff_utc)
    if (pd.isna(cutoff) or cutoff.tzinfo is None or cutoff.utcoffset() != pd.Timedelta(0)
            or cutoff.floor("min") != cutoff or times.hasnans or not times.is_unique
            or not times.is_monotonic_increasing or not times.floor("5min").equals(times)):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_CLOCK_INVALID")
    decisions = m5_decision_availability(times).as_unit("ns").asi8
    pointer = frame[DYNAMIC_M5_POINTER_COLUMNS[0]].to_numpy()
    valid = frame[DYNAMIC_M5_POINTER_COLUMNS[1]].to_numpy()
    terminal = frame[list(DYNAMIC_M5_POINTER_COLUMNS[2:4])].to_numpy()
    path = frame[list(DYNAMIC_M5_POINTER_COLUMNS[4:6])].to_numpy()
    if (pointer.dtype != np.dtype("int64") or valid.dtype != np.dtype(bool)
            or terminal.dtype != np.dtype(bool) or path.dtype != np.dtype(bool)
            or not np.array_equal(valid, terminal.any(axis=1) | path.any(axis=1))
            or np.any(valid & (pointer < 0)) or np.any(pointer < -1)):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_AVAILABILITY_INVALID")
    for column, component in ((DYNAMIC_M5_POINTER_COLUMNS[6], terminal), (DYNAMIC_M5_POINTER_COLUMNS[7], path)):
        reasons = frame[column].to_numpy()
        if (not all(isinstance(reason, str) and bool(reason) for reason in reasons)
                or not np.array_equal(reasons == "observed", component.any(axis=1))):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_CENSOR_REASON_INVALID")
    first_terminal = decisions + 60 * 10**9
    if (np.any(terminal.any(axis=1) & (first_terminal >= cutoff.value))
            or np.any(path.any(axis=1) & (first_terminal > cutoff.value))):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_COMPONENT_CROSSES_CUTOFF")
    eligible_time = decisions < cutoff.value
    if np.any(valid & ~eligible_time):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_OBSERVED_ROW_AFTER_CUTOFF")
    rows = np.flatnonzero(valid & eligible_time).astype(np.int64)
    if not len(rows):
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_NO_ELIGIBLE_ROWS")
    return {"rows": rows, "censored_rows_excluded": int(np.count_nonzero(eligible_time & ~valid)),
            "rows_after_cutoff": int(np.count_nonzero(~eligible_time)), "physical_rows": len(frame)}


def materialize(*, chronological_prefix: Mapping[str, Any], selected_sampler: Mapping[str, str],
                output_dir: Path, publish: bool, horizon_eligibility: Mapping[str, Any] | None = None,
                m5_path_eligibility: Mapping[str, Any] | None = None,
                dynamic_m5_eligibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if set(chronological_prefix) != {"design", "normalization_result", "labels_result"}:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_PREFIX_INVALID")
    if sum(value is not None for value in (horizon_eligibility, m5_path_eligibility, dynamic_m5_eligibility)) > 1:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_ELIGIBILITY_INVALID_MIXED_SCOPES")
    if dynamic_m5_eligibility is not None:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
        if (set(dynamic_m5_eligibility) != {"target_contract", "entry_eligibility_cutoff_utc"}
                or dynamic_m5_eligibility["target_contract"] != entry_observed_dynamic_m5_path_contract()):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_ELIGIBILITY_INVALID")
    if horizon_eligibility is not None and set(horizon_eligibility) != {"horizons_design", "entry_eligibility_cutoff_utc"}:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_HORIZON_ELIGIBILITY_INVALID")
    if m5_path_eligibility is not None:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
        if (horizon_eligibility is not None or set(m5_path_eligibility) != {"target_contract", "entry_eligibility_cutoff_utc"}
                or m5_path_eligibility["target_contract"] != entry_observed_m5_path_contract()):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_ELIGIBILITY_INVALID")
    artifacts = {key:require_binding(value, label="coordinate preprocessing", verify_file=True)
                 for key,value in chronological_prefix.items()}
    design, normalization, labels = (_read(artifacts[key])
        for key in ("design", "normalization_result", "labels_result"))
    physical = require_physical_chronological_preprocessing(
        artifacts, design=design, normalization=normalization, labels=labels)
    if dynamic_m5_eligibility is not None:
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
        if (dynamic_m5_eligibility["entry_eligibility_cutoff_utc"] != design["calendar"]["train_control_cutoff"]
                or design.get("scope", {}).get("entry_only_training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
                or design["scope"].get("emission_scope") != M5_DYNAMIC_OBSERVED_EMISSION_SCOPE):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_SCOPE_OR_CUTOFF_INVALID")
    if m5_path_eligibility is not None and m5_path_eligibility["entry_eligibility_cutoff_utc"] != design["calendar"]["train_control_cutoff"]:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_CUTOFF_INVALID")
    selected_sampler = require_binding(selected_sampler, label="selected sampler", verify_file=True)
    selected = _physical_native_sampler(selected_sampler, artifacts=artifacts, design=design, physical=physical)
    dynamic_found = None
    if dynamic_m5_eligibility is not None:
        dynamic_found = dynamic_m5_eligible_rows(Path(physical["physical_sources"]["train"]["parquet"]["path"]),
            cutoff_utc=dynamic_m5_eligibility["entry_eligibility_cutoff_utc"])
        if (dynamic_found["physical_rows"] != physical["train_rows"]
                or dynamic_found["censored_rows_excluded"] != 0 or dynamic_found["rows_after_cutoff"] != 0
                or not np.array_equal(dynamic_found["rows"], np.arange(physical["train_rows"], dtype=np.int64))):
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_DYNAMIC_M5_POPULATION_MISMATCH")
    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_OUTPUT_EXISTS_OR_ALIASED")
    if not publish:
        return {"decision":"INPUTS_VERIFIED_COORDINATES_NOT_PRODUCED", "published":False,
                "train_rows":physical["train_rows"], "test_data_used":False,
                "model_forwards":0, "optimizer_steps":0, "fits":0}
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    sampler = selected["selected_sampler_contract"]
    order = _epoch0_order(sampler)
    eligibility = None
    if horizon_eligibility is not None:
        horizons_design = require_binding(horizon_eligibility["horizons_design"], label="horizon learning design", verify_file=True)
        found = horizon_eligible_rows(Path(physical["physical_sources"]["train"]["parquet"]["path"]),
                                      cutoff_utc=horizon_eligibility["entry_eligibility_cutoff_utc"])
        if found["physical_rows"] != physical["train_rows"]:
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_HORIZON_POPULATION_MISMATCH")
        eligible = found["rows"]
        # the native sampler's full order, filtered to the eligible subset (what the trainer replays)
        order = order[np.isin(order, eligible)]
        eligibility = {"rule": HORIZON_ELIGIBILITY_RULE, "horizons_design": horizons_design,
                       "entry_eligibility_cutoff_utc": str(horizon_eligibility["entry_eligibility_cutoff_utc"]),
                       "eligible_rows": int(len(eligible)), "censored_rows_excluded": found["censored_rows_excluded"],
                       "complete_at_all_observation_horizons": True}
    if m5_path_eligibility is not None:
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
        found = m5_path_eligible_rows(Path(physical["physical_sources"]["train"]["parquet"]["path"]),
            cutoff_utc=m5_path_eligibility["entry_eligibility_cutoff_utc"])
        if found["physical_rows"] != physical["train_rows"]:
            raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_M5_PATH_POPULATION_MISMATCH")
        eligible = found["rows"]
        order = order[np.isin(order, eligible)]
        eligibility = {"rule": M5_PATH_ELIGIBILITY_RULE, "training_phase": ENTRY_ONLY_M5_PATH_TRAINING_PHASE,
            "target_contract": m5_path_eligibility["target_contract"],
            "entry_eligibility_cutoff_utc": m5_path_eligibility["entry_eligibility_cutoff_utc"],
            "eligible_rows": int(len(eligible)), "censored_rows_excluded": found["censored_rows_excluded"],
            "long_horizons_gate_primary_eligibility": False}
    if dynamic_m5_eligibility is not None:
        eligible = dynamic_found["rows"]
        order = order[np.isin(order, eligible)]
        eligibility = {"rule": DYNAMIC_M5_PATH_ELIGIBILITY_RULE,
            "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
            "emission_scope": M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
            "target_contract": dynamic_m5_eligibility["target_contract"],
            "entry_eligibility_cutoff_utc": dynamic_m5_eligibility["entry_eligibility_cutoff_utc"],
            "eligible_rows": len(eligible), "censored_rows_excluded": 0,
            "later_horizons_gate_primary_eligibility": False, "first_prefix_horizon_minutes": 1}
    planned = order[:design["budget"]["maximum_trained_entry_rows"]]
    if len(planned) != design["budget"]["maximum_trained_entry_rows"]:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_PLANNED_ROWS_SHORT")
    positions = deterministic_uniform_subsample_indices(
        population_rows=len(planned), requested_rows=design["budget"]["train_only_probe_entries"],
        seed=design["selection"]["seed"], split_salt=0)
    if eligibility is None:
        bindings = {"TRAIN_ELIGIBLE_PARENT_ROWS":physical["train_parent_rows"]}
    else:
        bindings = {"TRAIN_ELIGIBLE_PARENT_ROWS":_publish(output/"TRAIN_ELIGIBLE_PARENT_ROWS.npy", eligible)}
    for key, rows in (("TRAIN_NATIVE_EPOCH0_ORDER",order), ("TRAIN_NATIVE4096_PARENT_ROWS",planned),
                      ("TRAIN256_PROBE_PARENT_ROWS",planned[positions])):
        bindings[key] = _publish(output/(key+".npy"), rows)
    native = {"schema_version":(PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH if dynamic_m5_eligibility is not None
        else PHYSICAL_COORDINATES_SCHEMA_M5_PATH if m5_path_eligibility is not None
        else PHYSICAL_COORDINATES_SCHEMA_V1 if eligibility is None else PHYSICAL_COORDINATES_SCHEMA_V2),
        **({} if eligibility is None else {"eligibility":eligibility}),
        "decision":"TRAIN_COORDINATES_FROZEN_NO_MODEL", "design":artifacts["design"],
        "train_source":physical["physical_sources"]["train"]["parquet"],
        "control_source":physical["physical_sources"]["val"]["parquet"],
        "control_parent_rows":physical["control_parent_rows"], "selected_sampler":selected_sampler,
        "bindings":bindings, "selection_uses_outcome_values":False, "test_data_used":False,
        "model_forwards":0, "optimizer_steps":0, "fits":0}
    native["coordinates_sha256"] = native_sha256(native)
    def verify_native(stage):
        require_physical_native_training_coordinates(
            {**artifacts,"native_coordinates":_binding(stage)}, design=design, physical=physical)
    artifacts["native_coordinates"] = _publish(output/"NATIVE_COORDINATES.json", native, validate=verify_native)
    root_binding = selected["random_access_root"]
    root = _read({"path":root_binding["path"],"sha256":root_binding["file_sha256"]})
    indexes = {split:{"index":{"path":root["splits"][split]["index_parquet_path"],
                               "sha256":root["splits"][split]["index_parquet_sha256"]},
                      "manifest":_binding(Path(root["splits"][split]["manifest_path"]))}
               for split in ("train","val")}
    val_population = physical["physical_sources"]["val"]["physical_rows"]
    result = {"schema_version":"gx1_physical_prefix_measurement_coordinates_v1",
        "decision":"TRAIN_AND_CONTROL_COORDINATES_FROZEN_NO_MODEL_MEASUREMENTS",
        "design":artifacts["design"], "chronological_prefix":artifacts, "source_indexes":indexes,
        "sampler_contracts":{"train":sampler, "val":build_random_access_sampler_contract(
            split="val", source_lineage_sha256=sampler["source_lineage_sha256"],
            transition_budget_per_epoch=val_population*sampler["transitions_per_entry"],
            transitions_per_entry=sampler["transitions_per_entry"], entry_pair_population=val_population)},
        "coordinates":{}, "train_samples_exactly_reused":True, "control_entry_ids_unchanged":True,
        "all_samples_preserved":True, "test_data_used":False, "model_forwards":0,"optimizer_steps":0,"fits":0}
    for role in ("train","control"):
        inputs = _physical_measurement_inputs(artifacts["design"], result, role=role)
        arrays, _ = _physical_measurement_arrays(inputs)
        result["coordinates"][role] = _publish(output/(role.upper()+"_MEASUREMENT.npz"), arrays)
    def verify_measurement(stage):
        for role in ("train","control"):
            build_chronological_measurement_cohort(artifacts["design"], _binding(stage), role=role)
    coordinates = _publish(output/"MEASUREMENT_COORDINATES.json", result, validate=verify_measurement)
    complete = {"schema_version":"gx1_physical_coordinate_publication_v1",
        "decision":"NATIVE_AND_MEASUREMENT_COORDINATES_FROZEN_NO_LAUNCH_AUTHORITY",
        "chronological_prefix":artifacts, "measurement_coordinates":coordinates,
        "selected_sampler":selected_sampler, "test_data_used":False,
        "model_forwards":0,"optimizer_steps":0,"fits":0}
    published = _publish(output/"COMPLETE.json", complete)
    return {"decision":complete["decision"],"published":True,"completion":published,
            "chronological_prefix":artifacts,"measurement_coordinates":coordinates}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("design","normalization-result","labels-result","selected-sampler"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--horizons-design", type=Path, default=None,
                        help="trading-day-horizon learning design: eligible rows = complete at every horizon at or before the cutoff")
    parser.add_argument("--m5-path-eligibility", action="store_true",
        help="canonical M5-path availability subset; long horizons do not filter primary rows")
    parser.add_argument("--dynamic-m5-eligibility", action="store_true",
        help="dynamic first-minute terminal OR closed-prefix availability; no later horizon union")
    parser.add_argument("--entry-eligibility-cutoff-utc", type=str, default=None)
    args = parser.parse_args(argv)
    selections = sum((args.horizons_design is not None, args.m5_path_eligibility, args.dynamic_m5_eligibility))
    if selections > 1 or (bool(selections) != (args.entry_eligibility_cutoff_utc is not None)):
        raise SystemExit("Choose one explicit horizon/fixed-M5/dynamic-M5 eligibility scope bound with --entry-eligibility-cutoff-utc")
    from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
    m5_path_eligibility = None if not args.m5_path_eligibility else {
        "target_contract": entry_observed_m5_path_contract(), "entry_eligibility_cutoff_utc": args.entry_eligibility_cutoff_utc}
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    dynamic_m5_eligibility = None if not args.dynamic_m5_eligibility else {
        "target_contract": entry_observed_dynamic_m5_path_contract(), "entry_eligibility_cutoff_utc": args.entry_eligibility_cutoff_utc}
    horizon_eligibility = None if args.horizons_design is None else {
        "horizons_design": _binding(args.horizons_design.expanduser().absolute()),
        "entry_eligibility_cutoff_utc": args.entry_eligibility_cutoff_utc}
    result = materialize(chronological_prefix={
        key:_binding(getattr(args,key).expanduser().absolute())
        for key in ("design","normalization_result","labels_result")},
        selected_sampler=_binding(args.selected_sampler.expanduser().absolute()),
        output_dir=args.output_dir, publish=args.publish, horizon_eligibility=horizon_eligibility,
        m5_path_eligibility=m5_path_eligibility, dynamic_m5_eligibility=dynamic_m5_eligibility)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
