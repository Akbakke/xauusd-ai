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


def materialize(*, chronological_prefix: Mapping[str, Any], selected_sampler: Mapping[str, str],
                output_dir: Path, publish: bool, horizon_eligibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if set(chronological_prefix) != {"design", "normalization_result", "labels_result"}:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_PREFIX_INVALID")
    if horizon_eligibility is not None and set(horizon_eligibility) != {"horizons_design", "entry_eligibility_cutoff_utc"}:
        raise RuntimeError("PHYSICAL_COORDINATE_PRODUCER_HORIZON_ELIGIBILITY_INVALID")
    artifacts = {key:require_binding(value, label="coordinate preprocessing", verify_file=True)
                 for key,value in chronological_prefix.items()}
    design, normalization, labels = (_read(artifacts[key])
        for key in ("design", "normalization_result", "labels_result"))
    physical = require_physical_chronological_preprocessing(
        artifacts, design=design, normalization=normalization, labels=labels)
    selected_sampler = require_binding(selected_sampler, label="selected sampler", verify_file=True)
    selected = _physical_native_sampler(selected_sampler, artifacts=artifacts, design=design, physical=physical)
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
    native = {"schema_version":PHYSICAL_COORDINATES_SCHEMA_V1 if eligibility is None else PHYSICAL_COORDINATES_SCHEMA_V2,
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
    parser.add_argument("--entry-eligibility-cutoff-utc", type=str, default=None)
    args = parser.parse_args(argv)
    if (args.horizons_design is None) != (args.entry_eligibility_cutoff_utc is None):
        raise SystemExit("--horizons-design and --entry-eligibility-cutoff-utc are bound together")
    horizon_eligibility = None if args.horizons_design is None else {
        "horizons_design": _binding(args.horizons_design.expanduser().absolute()),
        "entry_eligibility_cutoff_utc": args.entry_eligibility_cutoff_utc}
    result = materialize(chronological_prefix={
        key:_binding(getattr(args,key).expanduser().absolute())
        for key in ("design","normalization_result","labels_result")},
        selected_sampler=_binding(args.selected_sampler.expanduser().absolute()),
        output_dir=args.output_dir, publish=args.publish, horizon_eligibility=horizon_eligibility)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
