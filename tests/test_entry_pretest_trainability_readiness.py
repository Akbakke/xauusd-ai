from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    canonical_json_sha256,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.scripts import verify_entry_pretest_trainability_readiness_v1 as readiness
from gx1.scripts import verify_entry_candidate_readiness_v1 as candidate_readiness
from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import artifact_binding
from tests.test_entry_model_native_pretest_technical_recipe import _recipe


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_recipe_files(recipe: dict[str, object]) -> None:
    artifacts = recipe["artifact_bindings"]
    assert isinstance(artifacts, dict)
    for index, binding in enumerate(artifacts.values()):
        assert isinstance(binding, dict)
        path = Path(str(binding["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"immutable-artifact-{index}".encode())
        binding["sha256"] = _sha(path)
    guard = recipe["test_guard_lineage"]
    assert isinstance(guard, dict)
    for name in (
        "train_manifest",
        "train_parquet",
        "val_manifest",
        "val_parquet",
        "dataset_build_proof",
        "full_input_liveness",
    ):
        guard[name] = dict(artifacts[name])
    recipe["artifact_bindings_sha256"] = canonical_json_sha256(artifacts)


def _write_recipe(path: Path, recipe: dict[str, object]) -> tuple[Path, str]:
    path.write_text(json.dumps(recipe, sort_keys=True), encoding="utf-8")
    return path.resolve(), _sha(path)


def _pretrain(path: Path, dataset_dir: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_path, _ = write_immutable_json_event(
        path.parent,
        "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "xau_direction_repair_pretrain_audit_v6",
            "decision": "PASS",
            "failures": [],
            "dataset_dir": dataset_dir,
            "data_splits": ["train", "val"],
            "contract_mode": "xau_seq513_model_native_direction_v20",
            "expected_signal_dim": 238,
            "large_artifact_hashes_verified": True,
            "require_mandatory_level_features": True,
            "require_inline_seq_structure": True,
            "require_xau_provenance": True,
        },
    )
    return event_path


def _recipes(tmp_path: Path) -> tuple[Path, str, Path, str, Path]:
    smoke = _recipe(tmp_path)
    _materialize_recipe_files(smoke)
    smoke_cli = smoke["trainer_cli"]
    assert isinstance(smoke_cli, dict)
    smoke_cli["execution_tier"] = "canonical"
    smoke_cli["train_time_window"] = None
    smoke["trainer_cli_sha256"] = canonical_json_sha256(smoke_cli)
    smoke["run_id"] = "PRETEST_SMOKE_RUN_V1"

    candidate = copy.deepcopy(smoke)
    candidate["profile"] = "candidate"
    candidate["run_id"] = "PRETEST_CANDIDATE_RUN_V1"
    candidate["out_bundle_dir"] = str((tmp_path / "candidate-bundle").resolve())
    candidate_cli = candidate["trainer_cli"]
    assert isinstance(candidate_cli, dict)
    candidate_cli["epochs"] = 30
    candidate_cli["subsample_rows"] = 0
    candidate["trainer_cli_sha256"] = canonical_json_sha256(candidate_cli)

    smoke_path, smoke_sha = _write_recipe(tmp_path / "smoke.json", smoke)
    candidate_path, candidate_sha = _write_recipe(tmp_path / "candidate.json", candidate)
    pretrain = _pretrain(
        tmp_path / "pretrain" / "placeholder.json", str(candidate["dataset_dir"])
    )
    return candidate_path, candidate_sha, smoke_path, smoke_sha, pretrain


def test_direct_pretest_trainability_binds_exact_recipes_and_pretrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate, candidate_sha, smoke, smoke_sha, pretrain = _recipes(tmp_path)
    monkeypatch.setattr(
        readiness,
        "require_training_recipe_execution_provenance",
        lambda **_kwargs: {"source_commit": "a" * 40},
    )

    report = readiness.run(
        readiness.build_parser().parse_args(
            [
                "--candidate-recipe-json", str(candidate),
                "--candidate-recipe-sha256", candidate_sha,
                "--smoke-recipe-json", str(smoke),
                "--smoke-recipe-sha256", smoke_sha,
                "--pretrain-audit-json", str(pretrain),
                "--repo-dir", str(tmp_path),
                "--out-dir", str(tmp_path / "out"),
            ]
        )
    )

    assert report["decision"] == readiness.READY_DECISION
    assert report["failures"] == []
    assert report["candidate_training_allowed"] is False
    assert report["input_bindings"]["candidate_recipe"]["sha256"] == candidate_sha
    assert candidate_readiness._trainability_contract_check(
        report,
        expected_candidate_recipe=artifact_binding(candidate),
    )["ok"] is True
    assert candidate_readiness._trainability_contract_check(
        report,
        expected_candidate_recipe=artifact_binding(smoke),
    )["ok"] is False


def test_direct_pretest_trainability_rejects_recipe_dataset_mismatch(tmp_path: Path) -> None:
    candidate, candidate_sha, smoke, smoke_sha, pretrain = _recipes(tmp_path)
    smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
    smoke_payload["dataset_run_id"] = "PRETEST_OTHER_DATASET_V1"
    smoke_payload["test_guard_lineage"]["dataset_run_id"] = (
        "PRETEST_OTHER_DATASET_V1"
    )
    smoke.write_text(json.dumps(smoke_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="1"):
        readiness.run(
            readiness.build_parser().parse_args(
                [
                    "--candidate-recipe-json", str(candidate),
                    "--candidate-recipe-sha256", candidate_sha,
                    "--smoke-recipe-json", str(smoke),
                    "--smoke-recipe-sha256", _sha(smoke),
                    "--pretrain-audit-json", str(pretrain),
                    "--repo-dir", str(tmp_path),
                    "--out-dir", str(tmp_path / "out"),
                ]
            )
        )
