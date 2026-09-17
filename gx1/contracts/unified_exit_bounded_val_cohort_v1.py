"""Bind a read-only VAL subset to its predeclared frozen evaluation plan.

This is data identity, never permission to launch or expand an evaluation.
The native campaign remains the sole execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from gx1.contracts.local_random_access_campaign_v2 import (
    canonical_sha256, read_bound_json, require_binding,
)

SCHEMA = "gx1_bounded_val_cohort_v1"
CHRONOLOGICAL_SCHEMA = "gx1_chronological_development_control_cohort_v1"


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


def build_chronological_control_cohort(
    design_binding: Mapping[str, str], *, source_index_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Use exactly the preselected later TRAIN rows; this grants no run authority."""
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
    original = require_binding(calendar["index"], label="design TRAIN index", verify_file=True)
    index = require_binding(source_index_binding or original, label="control TRAIN index", verify_file=True)
    rows = require_binding(calendar["bindings"]["CONTROL256_PARENT_ROWS"], label="fixed control rows", verify_file=True)
    population, children, parents = _chronological_control_coordinates(
        index["path"], index["sha256"], original["path"], original["sha256"],
        rows["path"], rows["sha256"], int(cutoff.value), int(end.value), 256)
    value = {
        "schema_version": CHRONOLOGICAL_SCHEMA, "split": "val", "source_split": "train",
        "evaluation_role": "chronological_reused_development_control",
        "plan": binding, "source_index": index, "population_rows": population,
        "entry_row_indices": list(children), "parent_entry_row_indices": list(parents),
        "test_data_used": False,
    }
    value["cohort_sha256"] = canonical_sha256(value)
    return value


def require_bounded_val_cohort(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("schema_version") == CHRONOLOGICAL_SCHEMA:
        expected = build_chronological_control_cohort(
            value.get("plan"), source_index_binding=value.get("source_index"))
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
