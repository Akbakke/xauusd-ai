from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts import entry_model_native_train_launch_v1 as launch
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    MODEL_NATIVE_RECIPE_ENV_SHA256,
    model_native_recipe_env_contract_metadata,
    require_model_native_recipe_env,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_train_recipe_audit_v1 as producer,
)
from tests.entry_model_native_train_wrapper_support import (
    DATASET_RUN_ID,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_smoke_train.sh"


def test_recipe_env_is_one_exact_complete_value_source_contract() -> None:
    metadata = model_native_recipe_env_contract_metadata()

    assert len(MODEL_NATIVE_RECIPE_ENV) == 162
    assert len(MODEL_NATIVE_RECIPE_ENV_KEYS) == 162
    assert metadata == {
        "schema_version": "entry_model_native_seq513_train_recipe_env_v1",
        "count": 162,
        "sha256": MODEL_NATIVE_RECIPE_ENV_SHA256,
        "keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
    }
    assert require_model_native_recipe_env(MODEL_NATIVE_RECIPE_ENV) == dict(
        MODEL_NATIVE_RECIPE_ENV
    )


def test_every_recipe_value_is_consumed_by_the_only_trainer_source() -> None:
    trainer_path = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    trainer = trainer_path.read_text(encoding="utf-8")
    syntax = ast.parse(trainer)
    canonical_defaults = None
    for node in syntax.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if isinstance(node.target, ast.Name) and (
            node.target.id == "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS"
        ):
            canonical_defaults = ast.literal_eval(node.value)
            break

    missing = sorted(key for key in MODEL_NATIVE_RECIPE_ENV if key not in trainer)

    assert missing == []
    assert isinstance(canonical_defaults, dict)
    assert set(MODEL_NATIVE_RECIPE_ENV) == set(canonical_defaults).union(
        {"GX1_V10_CKPT_MONITOR"}
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed"))
def test_recipe_env_rejects_every_non_exact_surface(mutation: str) -> None:
    candidate = dict(MODEL_NATIVE_RECIPE_ENV)
    if mutation == "missing":
        candidate.pop("ENTRY_MTF_DIR_AUX_WEIGHT")
    elif mutation == "extra":
        candidate["ENTRY_UNAUDITED_PASS_THROUGH"] = "1"
    else:
        candidate["ENTRY_MTF_DIR_AUX_WEIGHT"] = "0"

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_RECIPE_ENV_MISMATCH"):
        require_model_native_recipe_env(candidate)


def test_recipe_producer_event_drives_exact_smoke_wrapper_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper_argv, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    out_dir = (tmp_path / "recipe_20260722T140000Z").resolve()
    producer_argv = [
        "--profile",
        "smoke",
        "--repo",
        str(REPO),
        "--wrapper-path",
        str(WRAPPER),
        *wrapper_argv,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    recipe_flag = producer_argv.index("--recipe-audit-json")
    del producer_argv[recipe_flag : recipe_flag + 2]
    monkeypatch.setattr(producer, "_clean_worktree", lambda _repo: None)

    event_path, event = producer.run(producer.build_parser().parse_args(producer_argv))

    assert event_path.is_file()
    assert event["json_path"] == str(event_path)
    assert event["trainer_env"] == MODEL_NATIVE_RECIPE_ENV
    assert event["dataset_run_id"] == DATASET_RUN_ID
    assert event["trainer_env_contract"] == model_native_recipe_env_contract_metadata()
    assert event["activation_authority"] is False
    assert event["report_only"] is True
    assert not any(event["side_effects_started"].values())
    assert set(event["source_bindings"]) == {
        "aux_target_contract",
        "control_surface",
        "launch_contract",
        "recipe_contract",
        "recipe_producer",
        "wrapper",
        "trainer",
        "capped_runner",
    }

    original_recipe = str(paths["recipe_audit_json"])
    validated_argv = [
        str(event_path) if value == original_recipe else value
        for value in wrapper_argv
    ]
    result = subprocess.run(
        ["bash", str(WRAPPER), *validated_argv, "--dry-run"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 smoke contract" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00" in result.stdout
    assert "ENTRY_TAIL_DIRECTION_CE_WEIGHT=0.35" in result.stdout


def test_recipe_producer_fails_before_publication_when_source_is_dirty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    out_dir = (tmp_path / "recipe_20260722T140001Z").resolve()
    producer_argv = [
        "--profile",
        "smoke",
        "--repo",
        str(REPO),
        "--wrapper-path",
        str(WRAPPER),
        *wrapper_argv,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    recipe_flag = producer_argv.index("--recipe-audit-json")
    del producer_argv[recipe_flag : recipe_flag + 2]

    def reject(_repo: Path) -> None:
        raise RuntimeError("TRAIN_RECIPE_SOURCE_WORKTREE_DIRTY")

    monkeypatch.setattr(producer, "_clean_worktree", reject)

    with pytest.raises(RuntimeError, match="TRAIN_RECIPE_SOURCE_WORKTREE_DIRTY"):
        producer.run(producer.build_parser().parse_args(producer_argv))
    assert not out_dir.exists()


@pytest.mark.parametrize("mutation", ("missing", "wrong_equation", "unknown"))
def test_launch_rejects_non_exact_split_aux_target_emission_proof(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    manifest_path = paths["train_manifest_json"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    emission = payload["extra"]["aux_head_target_contract"]
    if mutation == "missing":
        emission.pop("complete_rows_emitted")
    elif mutation == "wrong_equation":
        emission["candidate_rows_before_completeness"] += 1
    else:
        emission["allow_incomplete_rows"] = False

    with pytest.raises(
        launch.LaunchContractError,
        match="split manifest aux-target emission contract invalid",
    ):
        launch._validate_split_manifest(
            payload,
            path=manifest_path,
            parquet=paths["train_parquet"],
            m5_prebuilt=paths["m5_prebuilt_path"],
        )


def test_launch_derives_dataset_run_id_from_post_rebuild_and_all_splits(
    tmp_path: Path,
) -> None:
    _, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    payloads = {
        f"{split}_manifest_json": json.loads(
            paths[f"{split}_manifest_json"].read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test")
    }
    post_rebuild = json.loads(
        paths["post_rebuild_readiness_json"].read_text(encoding="utf-8")
    )

    assert (
        launch._dataset_run_id_from_launch_evidence(
            post_rebuild=post_rebuild,
            payloads=payloads,
        )
        == DATASET_RUN_ID
    )

    payloads["val_manifest_json"]["extra"]["entry_run_id"] = (
        "DIFFERENT_DATASET_RUN_ID"
    )
    with pytest.raises(
        launch.LaunchContractError,
        match="val split dataset run lineage mismatch",
    ):
        launch._dataset_run_id_from_launch_evidence(
            post_rebuild=post_rebuild,
            payloads=payloads,
        )
