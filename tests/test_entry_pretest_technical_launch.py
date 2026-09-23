from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert "--candidate-gate-json" not in command
    assert "--candidate-gate-sha256" not in command
    assert "--execution-tier" in command
    assert command[command.index("--execution-tier") + 1] == "attended_only"
    assert command[command.index("--precision-policy") + 1] == "deterministic_fp32"
    assert command[command.index("--train-time-window-start-utc") + 1] == (
        recipe["trainer_cli"]["train_time_window"]["start_utc"]
    )
    assert environment["GX1_ENTRY_DATASET_RUN_ID"] == recipe["dataset_run_id"]
    assert environment["GX1_ENTRY_TRAIN_PARQUET_SHA256"] == (
        recipe["artifact_bindings"]["train_parquet"]["sha256"]
    )
    assert environment["GX1_V10_MULTI_TF_V4_CACHE_DIR"].endswith("MULTI_TF")


@pytest.mark.parametrize("initialized", [False, True])
def test_pretest_launcher_allows_guarded_canonical_smoke_bundle_path(
    tmp_path: Path,
    monkeypatch,
    initialized: bool,
) -> None:
    recipe = _recipe(tmp_path)
    cli = recipe["trainer_cli"]
    assert isinstance(cli, dict)
    cli["execution_tier"] = "canonical"
    cli["train_time_window"] = None
    if initialized:
        cli.update({
            "precision_policy": "deterministic_fp32",
            "initial_checkpoint_path": str(tmp_path / "candidate_state.pt"),
            "initial_checkpoint_sha256": "b" * 64,
            "freeze_initial_teacher": True,
        })
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
    if initialized:
        assert command[command.index("--initial-checkpoint-path") + 1] == cli["initial_checkpoint_path"]
        assert command[command.index("--initial-checkpoint-sha256") + 1] == "b" * 64
        assert "--freeze-initial-teacher" in command
    else:
        assert "--initial-checkpoint-path" not in command


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


def _candidate_command(tmp_path: Path, monkeypatch):
    recipe = _recipe(tmp_path)
    recipe["profile"] = "candidate"
    recipe["trainer_cli"].update({
        "execution_tier": "canonical",
        "train_time_window": None,
        "epochs": 30,
        "early_stop_patience": 5,
        "subsample_rows": 0,
    })
    recipe["trainer_cli_sha256"] = canonical_json_sha256(recipe["trainer_cli"])
    recipe_path = tmp_path / "candidate-recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True), encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    gate_path = tmp_path / "candidate-gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        launcher, "require_training_recipe_execution_provenance",
        lambda **_kwargs: {"source_commit": recipe["source_commit"]},
    )
    # Gate-content validation has dedicated contract tests. Here the wrapper
    # accepts its evidence so the direct trainer boundary can be exercised.
    monkeypatch.setattr(
        launcher, "require_pretest_candidate_launch_gate", lambda *_args, **_kwargs: {},
    )
    command, _, _ = launcher.build_pretest_technical_launch(
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha,
        candidate_gate_path=gate_path,
        candidate_gate_sha256=gate_sha,
    )
    assert command[command.index("--candidate-gate-json") + 1] == str(gate_path)
    assert command[command.index("--candidate-gate-sha256") + 1] == gate_sha
    return command, recipe_path, recipe_sha, gate_path, gate_sha


@pytest.mark.parametrize("gate_mode", ["missing", "missing-sha", "changed", "valid"])
def test_direct_pretest_candidate_trainer_revalidates_gate_before_cuda(
    tmp_path: Path, monkeypatch, gate_mode: str,
) -> None:
    from gx1.contracts import entry_model_native_train_launch_v1 as source_contract
    from gx1.contracts import entry_pretest_candidate_launch_gate_v1 as gate_contract
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    command, recipe_path, recipe_sha, gate_path, gate_sha = _candidate_command(
        tmp_path, monkeypatch,
    )
    trainer_module = "gx1.models.entry_v10.entry_v10_ctx_train_v3"
    argv = command[command.index(trainer_module) + 1:]
    if gate_mode in {"missing", "missing-sha"}:
        flags = ("--candidate-gate-json", "--candidate-gate-sha256") if gate_mode == "missing" else ("--candidate-gate-sha256",)
        for flag in flags:
            position = argv.index(flag)
            del argv[position:position + 2]
    elif gate_mode == "changed":
        gate_path.write_text('{"changed":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [trainer_module, *argv])
    monkeypatch.setattr(trainer, "_require_trainer_cgroup_preflight", lambda: None)
    monkeypatch.setattr(trainer, "_enforce_canonical_train_env_contract", lambda: None)
    monkeypatch.setattr(
        source_contract, "require_training_recipe_execution_provenance",
        lambda **_kwargs: {},
    )
    gate_calls = []
    if gate_mode == "valid":
        def valid_gate(path, digest, **kwargs):
            gate_calls.append((path, digest, kwargs))
            return {}
        monkeypatch.setattr(gate_contract, "require_pretest_candidate_launch_gate", valid_gate)

    def cuda_boundary(**_kwargs):
        raise RuntimeError("TEST_REACHED_CUDA_BOUNDARY_AFTER_GATE")

    def forbidden_compute(*_args, **_kwargs):
        raise AssertionError("test must never allocate CUDA or train")

    monkeypatch.setattr(trainer, "_require_cuda_trainer_guard_execution", cuda_boundary)
    monkeypatch.setattr(trainer, "_resolve_device", forbidden_compute)
    monkeypatch.setattr(trainer, "run_train", forbidden_compute)
    expected_error = {
        "missing": "ENTRY_TRAIN_PRETEST_CANDIDATE_GATE_REQUIRED",
        "missing-sha": "ENTRY_TRAIN_PRETEST_CANDIDATE_GATE_REQUIRED",
        "changed": "ENTRY_TRAIN_PRETEST_CANDIDATE_GATE_REJECTED.*hash mismatch",
        "valid": "TEST_REACHED_CUDA_BOUNDARY_AFTER_GATE",
    }[gate_mode]
    with pytest.raises(RuntimeError, match=expected_error):
        trainer.main()
    if gate_mode == "valid":
        assert gate_calls == [(
            gate_path, gate_sha,
            {"expected_recipe_path": recipe_path, "expected_recipe_sha256": recipe_sha},
        )]


def test_legacy_trainer_recipe_does_not_require_pretest_gate(tmp_path: Path) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    recipe_path = tmp_path / "legacy-recipe.json"
    recipe_path.write_text(
        '{"schema_version":"entry_model_native_seq513_train_recipe_audit_v9"}',
        encoding="utf-8",
    )
    args = SimpleNamespace(recipe_audit_json=recipe_path, profile="candidate")
    trainer._require_pretest_recipe_cli_match(args)


def test_legacy_recipe_rejects_unbound_checkpoint_initialization(tmp_path: Path) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    recipe = tmp_path / "legacy.json"
    recipe.write_text('{"schema_version":"entry_model_native_seq513_train_recipe_audit_v9"}')
    args = SimpleNamespace(
        recipe_audit_json=recipe, profile="smoke",
        initial_checkpoint_path=tmp_path / "state.pt", initial_checkpoint_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="REQUIRES_PRETEST_RECIPE"):
        trainer._require_pretest_recipe_cli_match(args)
