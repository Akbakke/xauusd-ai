"""Bind a read-only VAL subset to its predeclared frozen evaluation plan.

This is data identity, never permission to launch or expand an evaluation.
The native campaign remains the sole execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from gx1.contracts.local_random_access_campaign_v2 import (
    canonical_sha256, read_bound_json, require_binding,
)

SCHEMA = "gx1_bounded_val_cohort_v1"
CHRONOLOGICAL_SCHEMA = "gx1_chronological_development_control_cohort_v1"
MEASUREMENT_SCHEMA = "gx1_chronological_measurement_cohort_v1"
TRAIN_ROLLOUT_SCHEMA = "gx1_chronological_train_rollout_cohort_v1"


def build_bounded_val_cohort(plan_binding: Mapping[str, str]) -> dict[str, Any]:
    binding = require_binding(plan_binding, label="frozen evaluation plan", verify_file=True)
    plan = read_bound_json(Path(binding["path"]), binding["sha256"])
    if (
        plan.get("decision") != "FROZEN_CANDIDATE_AND_COHORT_NOT_LAUNCH_AUTHORITY"
        or plan.get("fit_frozen") is not True
        or plan.get("further_tuning_on_existing_train_control_forbidden") is not True
        or plan.get("execution", {}).get("training_enabled") is not False
        or plan.get("execution", {}).get("optimizer_steps") != 0
    ):
        raise RuntimeError("BOUNDED_VAL_FROZEN_PLAN_REQUIRED")
    selection = plan.get("selection", {})
    rows = selection.get("rows")
    population = selection.get("population_rows")
    requested = selection.get("requested_rows")
    if (
        not isinstance(rows, list)
        or type(population) is not int or population <= 0
        or type(requested) is not int or not 0 < requested < population
        or len(rows) != requested
        or selection.get("selection_uses_outcomes") is not False
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise RuntimeError("BOUNDED_VAL_SELECTION_INVALID")
    ids = [row.get("entry_row_index") for row in rows]
    if (any(type(i) is not int or not 0 <= i < population for i in ids)
            or ids != sorted(set(ids))):
        raise RuntimeError("BOUNDED_VAL_ENTRY_IDENTITIES_INVALID")
    index = require_binding(plan.get("val_index"), label="bounded VAL index", verify_file=True)
    value = {
        "schema_version": SCHEMA, "split": "val", "plan": binding,
        "source_index": index, "population_rows": population,
        "entry_row_indices": ids, "test_data_used": False,
    }
    value["cohort_sha256"] = canonical_sha256(value)
    return value



@lru_cache(maxsize=4)
def _chronological_control_coordinates(index_path, index_sha, original_path, original_sha,
                                       rows_path, rows_sha, cutoff_ns, end_ns, expected_rows):
    # Hash-bound immutable inputs are checked by the caller before cache lookup.
    # Cache only immutable coordinates, never market outcomes or model outputs.
    columns = ["entry_row_index", "parent_entry_row_index", "entry_time_ns",
               "first_state_time_ns", "parent_m1_start_row", "child_m1_start_row",
               "successor_transition_count", "lifecycle_state_count",
               "economic_terminal", "right_censored", "entry_bid", "entry_ask"]
    original = pd.read_parquet(original_path, columns=columns)
    frame = original if index_path == original_path and index_sha == original_sha else pd.read_parquet(index_path, columns=columns)
    rows = np.load(rows_path, allow_pickle=False)
    if (rows.dtype != np.dtype("int64") or rows.shape != (expected_rows,)
            or not np.array_equal(rows, np.unique(rows)) or np.any(rows < 0)
            or not frame.equals(original)
            or not np.array_equal(frame["entry_row_index"].to_numpy(), np.arange(len(frame)))
            or not frame["parent_entry_row_index"].is_unique):
        raise RuntimeError("BOUNDED_VAL_CHRONOLOGICAL_COORDINATES_INVALID")
    selected = frame.loc[frame["parent_entry_row_index"].isin(rows)]
    times = selected["entry_time_ns"].to_numpy(dtype="int64")
    if (len(selected) != expected_rows
            or selected["parent_entry_row_index"].tolist() != rows.tolist()
            or np.any(times < cutoff_ns) or np.any(times >= end_ns)):
        raise RuntimeError("BOUNDED_VAL_CHRONOLOGICAL_CONTROL_INVALID")
    return len(frame), tuple(int(i) for i in selected["entry_row_index"]), tuple(int(i) for i in rows)



@lru_cache(maxsize=4)
def _physical_control_coordinates(index_path, index_sha, manifest_path, manifest_sha,
                                  source_identity, rows_path, rows_sha, cutoff_ns, end_ns):
    """Verify the entire physical VAL mapping before selecting frozen controls.

    The caller verifies immutable file hashes before each cache lookup. Only
    coordinates are cached. Native input/economics admission remains separate.
    """
    from gx1.contracts import unified_exit_random_access_index_v1 as index_owner
    parquet_path, parquet_sha, parent_manifest_path, parent_manifest_sha, population, clock_sha = source_identity
    manifest = read_bound_json(Path(manifest_path), manifest_sha)
    # Check the declared parent/split before following any index source pointer.
    if (manifest.get("split") != "val"
            or manifest.get("schema_version") != index_owner.RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION
            or manifest.get("source_bindings", {}).get("parent_entry_parquet")
               != {"path": parquet_path, "sha256": parquet_sha}
            or manifest.get("source_bindings", {}).get("parent_entry_manifest")
               != {"path": parent_manifest_path, "sha256": parent_manifest_sha}
            or manifest.get("parent_entry_source_rows") != population
            or manifest.get("parent_entry_clock_sha256") != clock_sha):
        raise RuntimeError("BOUNDED_VAL_PHYSICAL_SOURCE_MISMATCH")
    frame = pd.read_parquet(index_path)
    checked = index_owner.require_random_access_index_manifest(
        manifest, expected_split="val", index_frame=frame,
        index_path=Path(index_path), verify_sources=False)
    expected = np.arange(population, dtype="int64")
    clock = pd.DatetimeIndex(pd.to_datetime(frame["entry_time_ns"], unit="ns", utc=True))
    if (len(frame) != population
            or any(frame[name].dtype != np.dtype("int64") for name in (
                "entry_row_index", "parent_entry_row_index", "entry_time_ns"))
            or not np.array_equal(frame["entry_row_index"].to_numpy(), expected)
            or not np.array_equal(frame["parent_entry_row_index"].to_numpy(), expected)
            or not clock.is_unique
            or index_owner._clock_sha256(clock) != clock_sha
            or checked["child_entry_clock_sha256"] != clock_sha
            or checked["parent_entry_row_indices_sha256"] != hashlib.sha256(
                np.ascontiguousarray(expected, dtype="<i8").tobytes()).hexdigest()
            or np.any(clock.asi8 < cutoff_ns) or np.any(clock.asi8 >= end_ns)):
        raise RuntimeError("BOUNDED_VAL_PHYSICAL_COORDINATES_INVALID")
    rows = np.load(rows_path, allow_pickle=False)
    if (rows.dtype != np.dtype("int64") or rows.shape != (256,)
            or not np.array_equal(rows, np.unique(rows))
            or np.any(rows < 0) or np.any(rows >= population)):
        raise RuntimeError("BOUNDED_VAL_PHYSICAL_CONTROL_INVALID")
    selected = frame.iloc[rows]
    return population, tuple(int(i) for i in selected["entry_row_index"]), tuple(int(i) for i in rows)


def build_chronological_control_cohort(
    design_binding: Mapping[str, str], *, source_index_binding: Mapping[str, str] | None = None,
    source_index_manifest_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind controls to their declared physical source, never grant run authority."""
    binding = require_binding(design_binding, label="chronological design", verify_file=True)
    design = read_bound_json(Path(binding["path"]), binding["sha256"])
    scope = design.get("scope", {})
    if (design.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
            or design.get("status") != "DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE"
            or scope.get("test_sealed") is not True
            or scope.get("existing_control_periods_are_reused_development") is not True
            or scope.get("native_launch_authorized") is not False
            or scope.get("one_experiment") is not True
            or design.get("selection", {}).get("control_entries") != 256
            or design.get("budget", {}).get("later_control_entries") != 256):
        raise RuntimeError("BOUNDED_VAL_CHRONOLOGICAL_DESIGN_INVALID")
    calendar = design["calendar"]
    cutoff = pd.Timestamp(calendar["train_control_cutoff"])
    end = pd.Timestamp(calendar["development_control_entry_end_exclusive"])
    if cutoff.tzinfo is None or end.tzinfo is None or cutoff >= end:
        raise RuntimeError("BOUNDED_VAL_CHRONOLOGICAL_WINDOW_INVALID")
    physical = any(key in calendar for key in (
        "physical_source_splits", "physical_coordinate_namespaces_are_separate", "source_bindings"))
    manifest_binding = None
    if physical:
        sources = calendar.get("source_bindings")
        if (calendar.get("physical_source_splits") != {"train": "train", "control": "val"}
                or calendar.get("physical_coordinate_namespaces_are_separate") is not True
                or "index" in calendar
                or not isinstance(sources, Mapping) or set(sources) != {"train", "val"}
                or any(not isinstance(sources[name], Mapping) for name in ("train", "val"))):
            raise RuntimeError("BOUNDED_VAL_PHYSICAL_DESIGN_INVALID")
        bound_sources = {
            split: {kind: require_binding(sources[split].get(kind),
                    label=f"physical {split} {kind}", verify_file=False)
                    for kind in ("parquet", "manifest")}
            for split in ("train", "val")}
        if (any(bound_sources["train"][kind]["path"] == bound_sources["val"][kind]["path"]
                for kind in ("parquet", "manifest"))
                or type(sources["val"].get("physical_rows")) is not int
                or sources["val"]["physical_rows"] < 256):
            raise RuntimeError("BOUNDED_VAL_PHYSICAL_SOURCE_MISMATCH")
        index = require_binding(source_index_binding, label="physical control VAL index", verify_file=True)
        manifest_binding = require_binding(source_index_manifest_binding,
            label="physical control VAL index manifest", verify_file=True)
        rows = require_binding(calendar["bindings"]["CONTROL256_PARENT_ROWS"],
            label="fixed control rows", verify_file=True)
        source = bound_sources["val"]
        population, children, parents = _physical_control_coordinates(
            index["path"], index["sha256"], manifest_binding["path"], manifest_binding["sha256"],
            (source["parquet"]["path"], source["parquet"]["sha256"],
             source["manifest"]["path"], source["manifest"]["sha256"],
             sources["val"]["physical_rows"], sources["val"].get("clock_sha256")),
            rows["path"], rows["sha256"], int(cutoff.value), int(end.value))
    else:
        if source_index_manifest_binding is not None:
            raise RuntimeError("BOUNDED_VAL_LEGACY_INDEX_MANIFEST_UNEXPECTED")
        original = require_binding(calendar["index"], label="design TRAIN index", verify_file=True)
        index = require_binding(source_index_binding or original, label="control TRAIN index", verify_file=True)
        rows = require_binding(calendar["bindings"]["CONTROL256_PARENT_ROWS"], label="fixed control rows", verify_file=True)
        population, children, parents = _chronological_control_coordinates(
            index["path"], index["sha256"], original["path"], original["sha256"],
            rows["path"], rows["sha256"], int(cutoff.value), int(end.value), 256)
    value = {
        "schema_version": CHRONOLOGICAL_SCHEMA, "split": "val", "source_split": "val" if physical else "train",
        "evaluation_role": "chronological_reused_development_control",
        "plan": binding, "source_index": index, "population_rows": population,
        "entry_row_indices": list(children), "parent_entry_row_indices": list(parents),
        "test_data_used": False,
    }
    if manifest_binding is not None:
        value["source_index_manifest"] = manifest_binding
    value["cohort_sha256"] = canonical_sha256(value)
    return value


def _physical_measurement_inputs(design_binding, result, *, role):
    """Join frozen measurement rows to actual physical indexes and native draws."""
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
        require_physical_chronological_preprocessing, require_physical_native_training_coordinates,
    )
    from gx1.contracts import unified_exit_random_access_index_v1 as index_owner
    from gx1.contracts.unified_exit_random_access_sampler_v1 import (
        build_random_access_sampler_contract, schedule_random_access_epoch,
    )

    def bound(value):
        return require_binding(value, label="physical measurement input", verify_file=True)
    def read(value):
        value = bound(value)
        return read_bound_json(Path(value["path"]), value["sha256"])
    artifacts = result.get("chronological_prefix")
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != {"design", "normalization_result", "labels_result", "native_coordinates"}
            or artifacts["design"] != design_binding):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_PREFIX_REQUIRED")
    design = read(design_binding)
    physical = require_physical_chronological_preprocessing(
        artifacts, design=design, normalization=read(artifacts["normalization_result"]),
        labels=read(artifacts["labels_result"]))
    coordinates = require_physical_native_training_coordinates(artifacts, design=design, physical=physical)
    selected = coordinates["selected_sampler"]
    root_binding = selected["random_access_root"]
    root = read({"path": root_binding["path"], "sha256": root_binding["file_sha256"]})
    indexes = result.get("source_indexes")
    if not isinstance(indexes, Mapping) or set(indexes) != {"train", "val"}:
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_INDEXES_REQUIRED")
    for split in ("train", "val"):
        binding = root["splits"][split]
        item = indexes[split]
        if (not isinstance(item, Mapping) or set(item) != {"index", "manifest"}
                or item["index"] != {"path": binding["index_parquet_path"], "sha256": binding["index_parquet_sha256"]}
                or item["manifest"]["path"] != binding["manifest_path"]):
            raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_INDEX_MISMATCH")
    control = build_chronological_control_cohort(design_binding,
        source_index_binding=indexes["val"]["index"],
        source_index_manifest_binding=indexes["val"]["manifest"])
    split = "train" if role == "train" else "val"
    source = physical["physical_sources"][split]
    index = bound(indexes[split]["index"])
    manifest_binding = bound(indexes[split]["manifest"])
    manifest = read(manifest_binding)
    if (manifest.get("split") != split
            or any(manifest.get("source_bindings", {}).get(f"parent_entry_{kind}") != source[kind]
                   for kind in ("parquet", "manifest"))
            or manifest.get("parent_entry_clock_sha256") != source["clock_sha256"]
            or manifest.get("parent_entry_source_rows") != source["physical_rows"]
            or manifest.get("manifest_sha256") != root["splits"][split]["manifest_sha256"]):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_PARENT_MISMATCH")
    frame = pd.read_parquet(index["path"])
    index_owner.require_random_access_index_manifest(manifest, expected_split=split,
        index_frame=frame, index_path=Path(index["path"]), verify_sources=False)
    population = source["physical_rows"]
    identity = np.arange(population, dtype=np.int64)
    clock = pd.DatetimeIndex(pd.to_datetime(frame["entry_time_ns"], unit="ns", utc=True))
    if (len(frame) != population
            or not np.array_equal(frame["entry_row_index"], identity)
            or not np.array_equal(frame["parent_entry_row_index"], identity)
            or index_owner._clock_sha256(clock) != source["clock_sha256"]
            or manifest.get("child_entry_clock_sha256") != source["clock_sha256"]):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_MAPPING_MISMATCH")
    expected = (coordinates["probe_parent_rows"] if role == "train"
                else np.asarray(control["parent_entry_row_indices"], dtype=np.int64))
    train_sampler = selected["selected_sampler_contract"]
    # CONTROL has no optimizer stream. Freeze one native epoch over its whole
    # physical population, with the existing four-draw policy and bound lineage.
    sampler = (train_sampler if role == "train" else build_random_access_sampler_contract(
        split="val", source_lineage_sha256=train_sampler["source_lineage_sha256"],
        transition_budget_per_epoch=population * train_sampler["transitions_per_entry"],
        transitions_per_entry=train_sampler["transitions_per_entry"], entry_pair_population=population))
    if (not isinstance(result.get("sampler_contracts"), Mapping)
            or set(result["sampler_contracts"]) != {"train", "val"}
            or result["sampler_contracts"][split] != sampler):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_SAMPLER_MISMATCH")
    samples = schedule_random_access_epoch(sampler_contract=sampler, epoch_index=0,
        successor_transition_count_by_entry=frame["successor_transition_count"].astype("int64").tolist())
    wanted = set(expected.tolist())
    draws = {parent: [] for parent in wanted}
    for sample in samples:
        if sample["entry_row_index"] in wanted:
            draws[sample["entry_row_index"]].append((sample["sample_slot"], sample["state_index"]))
    expected_samples = [[offset for _, offset in sorted(draws[int(parent)])] for parent in expected]
    if any(len(row) != 4 for row in expected_samples):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PHYSICAL_DRAW_COVERAGE_INVALID")
    return {"design": design, "control": control, "expected": expected, "frame": frame,
            "source_index": index, "population_rows": population, "source_split": split,
            "manifest": manifest, "expected_samples": expected_samples,
            "extra": {"source_index_manifest": manifest_binding,
                      "parent_entry_parquet": source["parquet"],
                      "sampler_contract_sha256": sampler["contract_sha256"],
                      **({"physical_control_cohort_sha256": control["cohort_sha256"]} if role == "control" else {})}}


def _physical_measurement_arrays(inputs):
    """Reconstruct the frozen draws' support from bound M1 rows, including gaps."""
    from gx1.contracts.unified_exit_reference_policy_v1 import require_reference_policy_contract
    from gx1.contracts.unified_exit_random_access_index_v1 import _clock
    manifest = inputs["manifest"]
    split = inputs["source_split"]
    bindings = manifest.get("source_bindings", {})
    mb = require_binding(bindings.get("m1_child_manifest"), label="measurement M1 manifest", verify_file=True)
    metadata = read_bound_json(Path(mb["path"]), mb["sha256"])
    source = require_binding(bindings.get("m1_child"), label="measurement M1", verify_file=False)
    if (metadata.get("split") != split or metadata.get("decision") != "PASS"
            or metadata.get("output_parquet_sha256") != source["sha256"]
            or metadata.get("test_accessed") is not False):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_M1_SOURCE_INVALID")
    source = require_binding(source, label="measurement M1", verify_file=True)
    times = _clock(pd.read_parquet(source["path"], columns=["time"])["time"], "MEASUREMENT_M1").asi8
    rows = np.asarray(inputs["expected"], dtype=np.int64)
    frame = inputs["frame"].iloc[rows]
    starts = frame["child_m1_start_row"].to_numpy(dtype=np.int64)
    counts = frame["successor_transition_count"].to_numpy(dtype=np.int64)
    offsets = np.asarray(inputs["expected_samples"], dtype=np.int64)
    policy = require_reference_policy_contract(inputs["design"]["targets"]["reference_policy"])
    if (np.any(starts < 0) or np.any(starts + counts >= len(times))
            or not np.array_equal(times[starts], frame["first_state_time_ns"])
            or np.any(offsets < 0) or np.any(offsets >= counts[:, None])):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_M1_COORDINATES_INVALID")
    lengths = np.minimum(policy["maximum_observed_backup_steps"], counts[:, None] - offsets)
    boundaries = starts[:, None] + offsets + lengths
    anchor = starts + np.minimum(policy["maximum_observed_backup_steps"], counts)
    # Match native decision availability: the boundary M1 bar must have closed.
    arrays = {"parent_rows": rows.copy(), "child_rows": frame["entry_row_index"].to_numpy(dtype=np.int64),
              "entry_time_ns": frame["entry_time_ns"].to_numpy(dtype=np.int64),
              "sampled_state_indices": offsets,
              "sampled_reference_end_close_ns": times[boundaries] + 60_000_000_000,
              "anchor_reference_end_close_ns": times[anchor] + 60_000_000_000}
    return arrays, {"source_m1": source, "source_m1_manifest": mb}


def build_chronological_measurement_cohort(
    design_binding: Mapping[str, str], coordinates_binding: Mapping[str, str], *, role: str,
) -> dict[str, Any]:
    """Bind frozen TRAIN/control measurements, without authorizing a rollout."""
    if role not in {"train", "control"}:
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_ROLE_INVALID")
    binding = require_binding(coordinates_binding, label="measurement coordinates", verify_file=True)
    result = read_bound_json(Path(binding["path"]), binding["sha256"])
    physical = result.get("schema_version") == "gx1_physical_prefix_measurement_coordinates_v1"
    inputs = _physical_measurement_inputs(design_binding, result, role=role) if physical else None
    control = inputs["control"] if physical else build_chronological_control_cohort(design_binding)
    if (result.get("schema_version") not in ("gx1_prefix_measurement_coordinates_v1",
                                           "gx1_physical_prefix_measurement_coordinates_v1")
            or result.get("decision") != "TRAIN_AND_CONTROL_COORDINATES_FROZEN_NO_MODEL_MEASUREMENTS"
            or result.get("design") != control["plan"]
            or (not physical and (result.get("source_index") != control["source_index"]
                                  or result.get("population_rows") != control["population_rows"]))
            or any(result.get(k) is not True for k in (
                "train_samples_exactly_reused", "control_entry_ids_unchanged", "all_samples_preserved"))
            or result.get("test_data_used") is not False
            or any(result.get(k) != 0 for k in ("model_forwards", "optimizer_steps", "fits"))):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_RESULT_INVALID")
    if physical:
        design, expected = inputs["design"], inputs["expected"]
        source_index, population = inputs["source_index"], inputs["population_rows"]
    else:
        design = read_bound_json(Path(control["plan"]["path"]), control["plan"]["sha256"])
        aux_binding = require_binding(result["auxiliary_policies"], label="prefix policies", verify_file=True)
        aux = read_bound_json(Path(aux_binding["path"]), aux_binding["sha256"])
        if aux.get("frozen_design") != control["plan"]:
            raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_PREFIX_MISMATCH")
        row_binding = (aux["bindings"]["TRAIN256_PROBE_PARENT_ROWS"] if role == "train"
                       else design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"])
        row_binding = require_binding(row_binding, label="frozen measurement rows", verify_file=True)
        expected = np.load(row_binding["path"], allow_pickle=False)
        source_index, population = control["source_index"], control["population_rows"]
    array_binding = require_binding(result["coordinates"][role], label="measurement array", verify_file=True)
    with np.load(array_binding["path"], allow_pickle=False) as arrays:
        required = {"parent_rows": (256,), "child_rows": (256,), "entry_time_ns": (256,),
                    "sampled_state_indices": (256, 4), "sampled_reference_end_close_ns": (256, 4),
                    "anchor_reference_end_close_ns": (256,)}
        if (set(arrays.files) != set(required)
                or any(arrays[k].dtype != np.dtype("int64") or arrays[k].shape != shape
                       for k, shape in required.items())):
            raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_ARRAY_INVALID")
        data = {k: arrays[k].copy() for k in required}
    parents, children, offsets = data["parent_rows"], data["child_rows"], data["sampled_state_indices"]
    if (not np.array_equal(parents, expected) or len(np.unique(parents)) != 256
            or len(np.unique(children)) != 256 or np.any(children < 0)
            or np.any(children >= population) or np.any(offsets < 0)):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_COORDINATES_INVALID")
    if physical and offsets.tolist() != inputs["expected_samples"]:
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_NATIVE_SAMPLES_CHANGED")
    if physical:
        actual, m1_bindings = _physical_measurement_arrays(inputs)
        if any(not np.array_equal(data[key], actual[key]) for key in actual):
            raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_M1_SUPPORT_MISMATCH")
        inputs["extra"].update(m1_bindings)

    frame = inputs["frame"] if physical else pd.read_parquet(source_index["path"], columns=[
        "entry_row_index", "parent_entry_row_index", "entry_time_ns", "successor_transition_count"])
    selected = frame.iloc[children]
    cutoff_key = "train_control_cutoff" if role == "train" else "development_control_entry_end_exclusive"
    start_key = "train_entry_start_inclusive" if role == "train" else "train_control_cutoff"
    start, cutoff = (pd.Timestamp(design["calendar"][k]) for k in (start_key, cutoff_key))
    if (start.tzinfo is None or cutoff.tzinfo is None or start >= cutoff
            or not np.array_equal(selected.entry_row_index, children)
            or not np.array_equal(selected.parent_entry_row_index, parents)
            or not np.array_equal(selected.entry_time_ns, data["entry_time_ns"])
            or np.any(data["entry_time_ns"] < start.value) or np.any(data["entry_time_ns"] >= cutoff.value)
            or np.any(offsets >= selected.successor_transition_count.to_numpy()[:, None])
            or np.any(data["sampled_reference_end_close_ns"] > cutoff.value)
            or np.any(data["anchor_reference_end_close_ns"] > cutoff.value)):
        raise RuntimeError("CHRONOLOGICAL_MEASUREMENT_SUPPORT_INVALID")
    value = {
        "schema_version": MEASUREMENT_SCHEMA, "split": "train" if role == "train" else "val",
        "source_split": inputs["source_split"] if physical else "train", "measurement_only": True, "measurement_role": role,
        "evaluation_role": ("chronological_training_probe" if role == "train"
                            else "chronological_reused_development_control"),
        "plan": control["plan"], "measurement_coordinates": binding,
        "source_index": source_index, "population_rows": population,
        **(inputs["extra"] if physical else {}),
        "entry_row_indices": children.tolist(), "parent_entry_row_indices": parents.tolist(),
        "sampled_state_indices": offsets.tolist(), "reference_cutoff_time_ns": int(cutoff.value),
        "test_data_used": False,
    }
    value["cohort_sha256"] = canonical_sha256(value)
    return value



def build_chronological_train_rollout_cohort(
    design_binding: Mapping[str, str], coordinates_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Reuse frozen TRAIN identities with an observation end, never launch authority."""
    probe = build_chronological_measurement_cohort(
        design_binding, coordinates_binding, role="train")
    # Rollout/one-position accounting requires chronological order. Preserve the
    # exact frozen identities and their parent mapping; no outcome-based selection.
    pairs = sorted(zip(probe["entry_row_indices"], probe["parent_entry_row_indices"]))
    value = {
        "schema_version": TRAIN_ROLLOUT_SCHEMA, "split": "train", "source_split": "train",
        "evaluation_role": "chronological_training_policy_rollout", "measurement_only": False,
        "plan": probe["plan"], "measurement_coordinates": probe["measurement_coordinates"],
        "source_index": probe["source_index"], "population_rows": probe["population_rows"],
        "entry_row_indices": [child for child, _ in pairs],
        "parent_entry_row_indices": [parent for _, parent in pairs],
        "observation_cutoff_time_ns": probe["reference_cutoff_time_ns"],
        "test_data_used": False,
    }
    value["cohort_sha256"] = canonical_sha256(value)
    return value


def require_bounded_val_cohort(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("schema_version") == TRAIN_ROLLOUT_SCHEMA:
        expected = build_chronological_train_rollout_cohort(
            value.get("plan"), value.get("measurement_coordinates"))
        if dict(value) != expected:
            raise RuntimeError("BOUNDED_VAL_COHORT_BINDING_MISMATCH")
        return expected
    if isinstance(value, Mapping) and value.get("schema_version") == MEASUREMENT_SCHEMA:
        expected = build_chronological_measurement_cohort(
            value.get("plan"), value.get("measurement_coordinates"), role=value.get("measurement_role"))
        if dict(value) != expected:
            raise RuntimeError("BOUNDED_VAL_COHORT_BINDING_MISMATCH")
        return expected
    if isinstance(value, Mapping) and value.get("schema_version") == CHRONOLOGICAL_SCHEMA:
        expected = build_chronological_control_cohort(
            value.get("plan"), source_index_binding=value.get("source_index"),
            source_index_manifest_binding=value.get("source_index_manifest"))
        if dict(value) != expected:
            raise RuntimeError("BOUNDED_VAL_COHORT_BINDING_MISMATCH")
        return expected
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "split", "plan", "source_index", "population_rows",
        "entry_row_indices", "test_data_used", "cohort_sha256",
    }:
        raise RuntimeError("BOUNDED_VAL_COHORT_INVALID")
    expected = build_bounded_val_cohort(value["plan"])
    if dict(value) != expected:
        raise RuntimeError("BOUNDED_VAL_COHORT_BINDING_MISMATCH")
    return expected
