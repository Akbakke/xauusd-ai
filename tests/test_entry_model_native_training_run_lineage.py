from __future__ import annotations

from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_training_run_lineage_v1 import (
    FULL_POPULATION_ALGORITHM,
    SCHEMA_VERSION,
    UNIFORM_SUBSAMPLE_ALGORITHM,
    EntryModelNativeTrainingRunLineageError,
    build_training_run_lineage,
    deterministic_uniform_subsample_indices,
    population_selection_descriptor,
    require_training_run_lineage,
)


def _descriptor(*, rows: int, requested: int, salt: int) -> dict[str, object]:
    selected = deterministic_uniform_subsample_indices(
        population_rows=rows,
        requested_rows=requested,
        seed=1337,
        split_salt=salt,
    )
    return population_selection_descriptor(
        population_rows=rows,
        selected_indices=selected,
        algorithm=(
            UNIFORM_SUBSAMPLE_ALGORITHM
            if len(selected) < rows
            else FULL_POPULATION_ALGORITHM
        ),
    )


def test_smoke_lineage_binds_both_bounded_compute_populations() -> None:
    lineage = build_training_run_lineage(
        training_run_id="UNIT_SMOKE_TRAIN_20260828",
        dataset_run_id="UNIT_DATASET_RUN_20260828",
        training_profile="smoke",
        execution_tier="canonical",
        requested_subsample_rows=32,
        physical_train_rows=250,
        train_selection=_descriptor(rows=250, requested=32, salt=0),
        physical_val_rows=90,
        val_selection=_descriptor(rows=90, requested=32, salt=1),
    )

    assert lineage["schema_version"] == SCHEMA_VERSION
    assert lineage["effective_train_rows"] == 32
    assert lineage["effective_val_rows"] == 32
    assert lineage["population_sampling"]["train"]["algorithm"] == (
        UNIFORM_SUBSAMPLE_ALGORITHM
    )
    assert lineage["population_sampling"]["val"]["algorithm"] == (
        UNIFORM_SUBSAMPLE_ALGORITHM
    )
    assert lineage["population_sampling"]["train"]["selection_sha256"] != (
        lineage["population_sampling"]["val"]["selection_sha256"]
    )
    assert require_training_run_lineage(lineage) == lineage


def test_candidate_lineage_requires_complete_train_and_val() -> None:
    lineage = build_training_run_lineage(
        training_run_id="UNIT_CANDIDATE_TRAIN_20260828",
        dataset_run_id="UNIT_DATASET_RUN_20260828",
        training_profile="candidate",
        execution_tier="canonical",
        requested_subsample_rows=0,
        physical_train_rows=250,
        train_selection=_descriptor(rows=250, requested=0, salt=0),
        physical_val_rows=90,
        val_selection=_descriptor(rows=90, requested=0, salt=1),
    )
    assert lineage["effective_train_rows"] == 250
    assert lineage["effective_val_rows"] == 90

    broken = dict(lineage)
    broken["effective_val_rows"] = 32
    with pytest.raises(EntryModelNativeTrainingRunLineageError, match="val sampling"):
        require_training_run_lineage(broken)

    smoke_without_bound = dict(lineage)
    smoke_without_bound["training_profile"] = "smoke"
    with pytest.raises(EntryModelNativeTrainingRunLineageError, match="explicit positive"):
        require_training_run_lineage(smoke_without_bound)


def test_trainer_and_loader_share_one_lineage_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (
        root / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    ).read_text(encoding="utf-8")
    loader = (
        root / "gx1/models/entry_v10/entry_v10_bundle.py"
    ).read_text(encoding="utf-8")
    assert "build_training_run_lineage(" in trainer
    assert "require_training_run_lineage(" in loader
