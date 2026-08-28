"""Exact population lineage for model-native ENTRY training bundles.

The trainer and the strict bundle loader consume this one contract.  It keeps
the model-compute population explicit: a smoke bundle may use bounded,
deterministic TRAIN *and* VAL samples, whereas a candidate bundle must use
both complete populations.  Sampling never changes the fitted TRAIN input
normalisation population, V46 data, feature surface, or target definition.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


SCHEMA_VERSION = "entry_model_native_training_run_lineage_v3"
SAMPLING_SCHEMA_VERSION = "entry_model_native_population_sampling_v1"
FULL_POPULATION_ALGORITHM = "full_population_v1"
UNIFORM_SUBSAMPLE_ALGORITHM = "uniform_without_replacement_pcg64_v1"
# A pre-candidate integration may deliberately use one contiguous, causal
# TRAIN interval. It is a smoke-only evidence lane: candidate bundles still
# require both complete immutable populations below.
TEMPORAL_WINDOW_SUBSAMPLE_ALGORITHM = "contiguous_time_window_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "training_run_id",
        "dataset_run_id",
        "training_profile",
        "execution_tier",
        "requested_subsample_rows",
        "physical_train_rows",
        "effective_train_rows",
        "physical_val_rows",
        "effective_val_rows",
        "population_sampling",
    }
)
_SAMPLING_FIELDS = frozenset({"schema_version", "train", "val"})
_SPLIT_FIELDS = frozenset(
    {"algorithm", "population_rows", "selected_rows", "selection_sha256"}
)


class EntryModelNativeTrainingRunLineageError(RuntimeError):
    """Raised when a model bundle's train/validation population is unclear."""


def deterministic_uniform_subsample_indices(
    *,
    population_rows: int,
    requested_rows: int,
    seed: int,
    split_salt: int,
) -> np.ndarray:
    """Return sorted exact model-compute rows without using labels.

    ``split_salt`` prevents TRAIN and VAL from being two accidental views of
    the same random stream.  A full population has a deterministic complete
    index, which is also recorded in bundle lineage.
    """

    if type(population_rows) is not int or population_rows <= 0:
        raise ValueError("population_rows must be a positive exact int")
    if type(requested_rows) is not int or requested_rows < 0:
        raise ValueError("requested_rows must be a non-negative exact int")
    if type(seed) is not int or type(split_salt) is not int:
        raise ValueError("seed and split_salt must be exact ints")
    if requested_rows == 0 or requested_rows >= population_rows:
        return np.arange(population_rows, dtype=np.int64)
    sequence = np.random.SeedSequence([seed, split_salt])
    selected = np.random.default_rng(sequence).choice(
        population_rows,
        size=requested_rows,
        replace=False,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def population_selection_descriptor(
    *,
    population_rows: int,
    selected_indices: Sequence[int] | np.ndarray,
    algorithm: str,
) -> dict[str, Any]:
    """Bind the exact selected source-row indices without retaining labels."""

    if algorithm not in {
        FULL_POPULATION_ALGORITHM,
        UNIFORM_SUBSAMPLE_ALGORITHM,
        TEMPORAL_WINDOW_SUBSAMPLE_ALGORITHM,
    }:
        raise ValueError(f"unsupported population sampling algorithm: {algorithm!r}")
    indices = np.asarray(selected_indices, dtype=np.int64)
    if indices.ndim != 1 or int(indices.size) <= 0:
        raise ValueError("selected_indices must be a non-empty one-dimensional array")
    if int(population_rows) <= 0 or int(indices.size) > int(population_rows):
        raise ValueError("population/selection row counts are invalid")
    if np.any(indices < 0) or np.any(indices >= int(population_rows)):
        raise ValueError("selected indices fall outside the source population")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("selected indices must be sorted and unique")
    if algorithm == FULL_POPULATION_ALGORITHM and not np.array_equal(
        indices, np.arange(int(population_rows), dtype=np.int64)
    ):
        raise ValueError("full-population selection must contain every source row")
    return {
        "algorithm": algorithm,
        "population_rows": int(population_rows),
        "selected_rows": int(indices.size),
        "selection_sha256": hashlib.sha256(
            np.ascontiguousarray(indices.astype("<i8", copy=False)).tobytes()
        ).hexdigest(),
    }


def build_training_run_lineage(
    *,
    training_run_id: str,
    dataset_run_id: str,
    training_profile: str,
    execution_tier: str,
    requested_subsample_rows: int,
    physical_train_rows: int,
    train_selection: Mapping[str, Any],
    physical_val_rows: int,
    val_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the exact v3 lineage written to lock and metadata."""

    lineage = {
        "schema_version": SCHEMA_VERSION,
        "training_run_id": str(training_run_id),
        "dataset_run_id": str(dataset_run_id),
        "training_profile": str(training_profile),
        "execution_tier": str(execution_tier),
        "requested_subsample_rows": int(requested_subsample_rows),
        "physical_train_rows": int(physical_train_rows),
        "effective_train_rows": int(train_selection.get("selected_rows", -1)),
        "physical_val_rows": int(physical_val_rows),
        "effective_val_rows": int(val_selection.get("selected_rows", -1)),
        "population_sampling": {
            "schema_version": SAMPLING_SCHEMA_VERSION,
            "train": dict(train_selection),
            "val": dict(val_selection),
        },
    }
    return require_training_run_lineage(lineage)


def require_training_run_lineage(value: Any) -> dict[str, Any]:
    """Fail closed unless the TRAIN and VAL compute populations are exact."""

    if not isinstance(value, Mapping) or set(value) != REQUIRED_FIELDS:
        raise EntryModelNativeTrainingRunLineageError("run-lineage fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise EntryModelNativeTrainingRunLineageError("run-lineage schema is invalid")
    try:
        training_run_id = require_entry_run_id(value.get("training_run_id"))
        dataset_run_id = require_entry_run_id(value.get("dataset_run_id"))
    except Exception as exc:
        raise EntryModelNativeTrainingRunLineageError(
            f"run identities are invalid: {exc}"
        ) from exc
    if training_run_id == dataset_run_id:
        raise EntryModelNativeTrainingRunLineageError("training and dataset IDs collapse")
    profile = value.get("training_profile")
    execution_tier = value.get("execution_tier")
    requested = value.get("requested_subsample_rows")
    physical_train = value.get("physical_train_rows")
    effective_train = value.get("effective_train_rows")
    physical_val = value.get("physical_val_rows")
    effective_val = value.get("effective_val_rows")
    if (
        profile not in {"smoke", "candidate"}
        or not isinstance(execution_tier, str)
        or not execution_tier
        or any(type(item) is not int for item in (
            requested,
            physical_train,
            effective_train,
            physical_val,
            effective_val,
        ))
        or requested < 0
        or physical_train <= 0
        or not 0 < effective_train <= physical_train
        or physical_val <= 0
        or not 0 < effective_val <= physical_val
    ):
        raise EntryModelNativeTrainingRunLineageError("population counts are invalid")
    sampling = value.get("population_sampling")
    if not isinstance(sampling, Mapping) or set(sampling) != _SAMPLING_FIELDS:
        raise EntryModelNativeTrainingRunLineageError("sampling fields are not exact")
    if sampling.get("schema_version") != SAMPLING_SCHEMA_VERSION:
        raise EntryModelNativeTrainingRunLineageError("sampling schema is invalid")
    split_specs = {
        "train": (physical_train, effective_train),
        "val": (physical_val, effective_val),
    }
    normalized_sampling: dict[str, dict[str, Any]] = {}
    for split, (physical, effective) in split_specs.items():
        spec = sampling.get(split)
        if not isinstance(spec, Mapping) or set(spec) != _SPLIT_FIELDS:
            raise EntryModelNativeTrainingRunLineageError(
                f"{split} sampling fields are not exact"
            )
        algorithm = spec.get("algorithm")
        digest = spec.get("selection_sha256")
        if (
            algorithm not in {
                FULL_POPULATION_ALGORITHM,
                UNIFORM_SUBSAMPLE_ALGORITHM,
                TEMPORAL_WINDOW_SUBSAMPLE_ALGORITHM,
            }
            or type(spec.get("population_rows")) is not int
            or spec.get("population_rows") != physical
            or type(spec.get("selected_rows")) is not int
            or spec.get("selected_rows") != effective
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or (
                algorithm == FULL_POPULATION_ALGORITHM
                and effective != physical
            )
            or (
                algorithm == UNIFORM_SUBSAMPLE_ALGORITHM
                and effective >= physical
            )
            or (
                algorithm == TEMPORAL_WINDOW_SUBSAMPLE_ALGORITHM
                and effective >= physical
            )
        ):
            raise EntryModelNativeTrainingRunLineageError(
                f"{split} sampling descriptor is invalid"
            )
        normalized_sampling[split] = dict(spec)
    if profile == "candidate" and (
        requested != 0
        or effective_train != physical_train
        or effective_val != physical_val
        or any(
            normalized_sampling[split]["algorithm"] != FULL_POPULATION_ALGORITHM
            for split in ("train", "val")
        )
    ):
        raise EntryModelNativeTrainingRunLineageError(
            "candidate bundles require complete TRAIN and VAL populations"
        )
    if profile == "smoke" and requested <= 0:
        raise EntryModelNativeTrainingRunLineageError(
            "smoke bundles require an explicit positive compute budget"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "training_run_id": training_run_id,
        "dataset_run_id": dataset_run_id,
        "training_profile": profile,
        "execution_tier": execution_tier,
        "requested_subsample_rows": requested,
        "physical_train_rows": physical_train,
        "effective_train_rows": effective_train,
        "physical_val_rows": physical_val,
        "effective_val_rows": effective_val,
        "population_sampling": {
            "schema_version": SAMPLING_SCHEMA_VERSION,
            "train": normalized_sampling["train"],
            "val": normalized_sampling["val"],
        },
    }
