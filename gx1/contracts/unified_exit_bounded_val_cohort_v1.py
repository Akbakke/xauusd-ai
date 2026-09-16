"""Bind a read-only VAL subset to its predeclared frozen evaluation plan.

This is data identity, never permission to launch or expand an evaluation.
The native campaign remains the sole execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.local_random_access_campaign_v2 import (
    canonical_sha256, read_bound_json, require_binding,
)

SCHEMA = "gx1_bounded_val_cohort_v1"


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


def require_bounded_val_cohort(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "split", "plan", "source_index", "population_rows",
        "entry_row_indices", "test_data_used", "cohort_sha256",
    }:
        raise RuntimeError("BOUNDED_VAL_COHORT_INVALID")
    expected = build_bounded_val_cohort(value["plan"])
    if dict(value) != expected:
        raise RuntimeError("BOUNDED_VAL_COHORT_BINDING_MISMATCH")
    return expected
