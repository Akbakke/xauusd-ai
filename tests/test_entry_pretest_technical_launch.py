from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.scripts import run_entry_model_native_pretest_technical_train_v1 as launcher
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    canonical_json_sha256,
)
from tests.test_entry_model_native_pretest_technical_recipe import _recipe


def test_pretest_launcher_derives_every_runtime_value_from_recipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recipe = _recipe(tmp_path)
    recipe_path = (tmp_path / "pretest-recipe.json").resolve()
    recipe_path.write_text(json.dumps(recipe, sort_keys=True), encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        launcher,
        "require_training_recipe_execution_provenance",
        lambda **_kwargs: {"source_commit": recipe["source_commit"]},
    )

    command, environment, validated = launcher.build_pretest_technical_launch(
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha,
    )

    assert validated["run_id"] == recipe["run_id"]
    assert command[:10] == [
        str(launcher.CAPPED_RUNNER),
        "--class",
        "trainer",
        "--mem",
        "20G",
        "--swap",
        "512M",
        "--attended-smoke",
        "--",
        str(launcher.PYTHON),
    ]
    assert "--test-manifest-json" not in command
    assert "--test-parquet" not in command
    assert "--execution-tier" in command
    assert command[command.index("--execution-tier") + 1] == "attended_only"
    assert command[command.index("--train-time-window-start-utc") + 1] == (
        recipe["trainer_cli"]["train_time_window"]["start_utc"]
    )
    assert environment["GX1_ENTRY_DATASET_RUN_ID"] == recipe["dataset_run_id"]
    assert environment["GX1_ENTRY_TRAIN_PARQUET_SHA256"] == (
        recipe["artifact_bindings"]["train_parquet"]["sha256"]
    )
    assert environment["GX1_V10_MULTI_TF_V4_CACHE_DIR"].endswith("MULTI_TF")


def test_pretest_launcher_allows_guarded_canonical_smoke_bundle_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recipe = _recipe(tmp_path)
    cli = recipe["trainer_cli"]
    assert isinstance(cli, dict)
    cli["execution_tier"] = "canonical"
    cli["train_time_window"] = None
    recipe["trainer_cli_sha256"] = canonical_json_sha256(cli)
    recipe_path = (tmp_path / "pretest-canonical-recipe.json").resolve()
    recipe_path.write_text(json.dumps(recipe, sort_keys=True), encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        launcher,
        "require_training_recipe_execution_provenance",
        lambda **_kwargs: {"source_commit": recipe["source_commit"]},
    )

    command, _, _ = launcher.build_pretest_technical_launch(
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha,
    )

    assert command[:9] == [
        str(launcher.CAPPED_RUNNER),
        "--class",
        "trainer",
        "--mem",
        "20G",
        "--swap",
        "512M",
        "--",
        str(launcher.PYTHON),
    ]
    assert "--attended-smoke" not in command
    assert command[command.index("--execution-tier") + 1] == "canonical"
    assert "--train-time-window-start-utc" not in command
    assert "--train-time-window-end-utc" not in command


def test_pretest_candidate_launcher_requires_immutable_launch_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recipe = _recipe(tmp_path)
    recipe["profile"] = "candidate"
    recipe_path = (tmp_path / "pretest-candidate-recipe.json").resolve()
    recipe_path.write_text(json.dumps(recipe, sort_keys=True), encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        launcher,
        "require_training_recipe_execution_provenance",
        lambda **_kwargs: {"source_commit": recipe["source_commit"]},
    )

    with pytest.raises(
        launcher.PretestTechnicalLaunchError,
        match="requires an immutable candidate launch gate",
    ):
        launcher.build_pretest_technical_launch(
            recipe_path=recipe_path,
            recipe_sha256=recipe_sha,
        )
