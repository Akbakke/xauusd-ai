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
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_train.sh"


def test_recipe_env_is_one_exact_complete_value_source_contract() -> None:
    metadata = model_native_recipe_env_contract_metadata()

    # Wave C leaves only diagnostics/admission and optimization switches.
    # Registry operator parameters come only from immutable TRAIN-fit artifacts.
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS,
        MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE,
    )

    expected_count = 11
    assert len(MODEL_NATIVE_RECIPE_ENV) == expected_count
    assert len(MODEL_NATIVE_RECIPE_ENV_KEYS) == expected_count
    assert "ENTRY_DIRECTION_LOGIT_ADJUST_TAU" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_DIRECTION_CE_SCALE" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_TAIL_DIRECTION_CE_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP" not in MODEL_NATIVE_RECIPE_ENV
    assert not any(key.endswith("_POS_WEIGHT_CAP") for key in MODEL_NATIVE_RECIPE_ENV)
    # V30 package 5. The cosine switch carries no magnitude at all (T_max is
    # the declared --epochs budget, eta_min the library default 0.0), so it is
    # adopted ON.  V30 package 6: the weight-EMA key declares a HORIZON, not a
    # magnitude — "epoch" selects the owner's derivation (one epoch of
    # optimizer steps), "0.0" is the OFF sentinel, and nothing else is a
    # declared value, so no bare decay can ever be pinned here.
    assert set(MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS) <= set(
        MODEL_NATIVE_RECIPE_ENV_KEYS
    )
    assert MODEL_NATIVE_RECIPE_ENV["ENTRY_TRAIN_LR_COSINE_DECAY"] == "1"
    assert MODEL_NATIVE_RECIPE_ENV["ENTRY_TRAIN_WEIGHT_EMA_DECAY"] == (
        MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE
    )
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE == "0.0"
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE == "epoch"
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES == ("0.0", "epoch")
    assert not any(
        "REGISTRY" in key for key in MODEL_NATIVE_RECIPE_ENV
    )
    assert metadata == {
        "schema_version": "entry_model_native_seq513_train_recipe_env_v7",
        "count": expected_count,
        "sha256": MODEL_NATIVE_RECIPE_ENV_SHA256,
        "keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
    }
    assert require_model_native_recipe_env(MODEL_NATIVE_RECIPE_ENV) == dict(
        MODEL_NATIVE_RECIPE_ENV
    )


def test_every_recipe_value_is_consumed_by_the_only_trainer_source() -> None:
    trainer_path = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    trainer = trainer_path.read_text(encoding="utf-8")

    assert "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS" not in trainer
    assert "MODEL_NATIVE_RECIPE_ENV_KEYS" in trainer
    assert "for key in MODEL_NATIVE_RECIPE_ENV_KEYS" in trainer
    assert "require_model_native_recipe_env(recipe_env)" in trainer
    assert "[ENTRY_TRAIN_AMBIENT_CONTROL_FORBIDDEN]" in trainer


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed"))
def test_recipe_env_rejects_every_non_exact_surface(mutation: str) -> None:
    candidate = dict(MODEL_NATIVE_RECIPE_ENV)
    if mutation == "missing":
        candidate.pop("ENTRY_SYMMETRIC_NEGATIVES")
    elif mutation == "extra":
        candidate["ENTRY_UNAUDITED_PASS_THROUGH"] = "1"
    else:
        candidate["ENTRY_SYMMETRIC_NEGATIVES"] = "0"

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
    assert event["prefreeze_test_seal_lineage"]["dataset_run_id"] == DATASET_RUN_ID
    assert event["prefreeze_test_seal_lineage"]["access_proof"][
        "test_metrics_read"
    ] is False
    assert event["prefreeze_test_seal_lineage_sha256"] == (
        launch.canonical_json_sha256(event["prefreeze_test_seal_lineage"])
    )
    assert event["trainer_env_contract"] == model_native_recipe_env_contract_metadata()
    assert event["activation_authority"] is False
    assert event["report_only"] is True
    assert not any(event["side_effects_started"].values())
    assert set(event["source_bindings"]) == {
        "aux_target_contract",
        "control_surface",
        "launch_contract",
        "model",
        "test_seal_contract",
        "mtf_feature_builder",
        "mtf_smc_geometry",
        "mtf_specialist_routing",
        "recipe_contract",
        "recipe_producer",
            "wrapper",
            "trainer",
            "unified_exit_lifecycle_contract",
            "capped_runner",
    }

    original_recipe = str(paths["recipe_audit_json"])
    validated_argv = [
        str(event_path) if value == original_recipe else value
        for value in wrapper_argv
    ]
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--profile",
            "smoke",
            *validated_argv,
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 smoke contract" in result.stdout
    assert "ENTRY_HIER_TRADE_WEIGHT" not in result.stdout
    assert "ENTRY_MTF_DIR_AUX_WEIGHT" not in result.stdout


@pytest.mark.parametrize("invalid_dropout", ("-0.01", "1.0", "nan"))
def test_train_launch_rejects_invalid_explicit_dropout(
    tmp_path: Path,
    invalid_dropout: str,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    dropout_index = wrapper_argv.index("--dropout") + 1
    wrapper_argv[dropout_index] = invalid_dropout
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(launch.LaunchContractError, match="dropout"):
        launch._trainer_cli_contract(args)


def test_train_launch_rejects_every_noncanonical_wrapper_path(
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    noncanonical = tmp_path / "noncanonical_train.sh"
    noncanonical.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(noncanonical),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="canonical profile-explicit trainer wrapper",
    ):
        launch.validate_launch(args)


def test_candidate_launch_rejects_training_subsample(tmp_path: Path) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="candidate",
        wrapper=WRAPPER,
    )
    subsample_index = wrapper_argv.index("--subsample-rows") + 1
    wrapper_argv[subsample_index] = "512"
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "candidate",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="candidate training requires full TRAIN population",
    ):
        launch._trainer_cli_contract(args)


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


def test_recipe_rejects_training_run_id_equal_to_dataset_run_id(
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    run_id_index = wrapper_argv.index("--run-id") + 1
    wrapper_argv[run_id_index] = DATASET_RUN_ID
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="training run_id must differ from immutable dataset_run_id",
    ):
        launch.validate_launch(args)


def test_trainer_has_no_shadow_default_for_any_recipe_value() -> None:
    """The recipe owner must be the single origin of every training value.

    The trainer used to carry its own literal default at all 160 call sites.
    62 had drifted from the recipe and 30 of those to zero, which silently
    deletes the direction-balance, slice-guard and prior-match loss families.
    A second origin can never be allowed back.
    """
    import re
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
    )

    source = (REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    with_default = re.findall(r'_env_str\(\s*"[A-Z0-9_]+"\s*,', source)
    assert with_default == [], f"trainer regained shadow defaults: {with_default}"
    single_arg = re.findall(r'_env_str\(\s*"([A-Z0-9_]+)"\s*\)', source)
    indirect_keys = {
        *MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
        "GX1_CTX_CONTRACT",
    }
    assert set(single_arg) == set(MODEL_NATIVE_RECIPE_ENV_KEYS) - indirect_keys


def test_trainer_cli_has_no_numeric_execution_defaults() -> None:
    source = (REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    required_flags = (
        "--seed",
        "--device",
        "--batch_size",
        "--epochs",
        "--lr",
        "--seq_len",
        "--num-workers",
        "--early-stopping-patience",
        "--early-stopping-min-delta",
        "--multi-tf-scale",
        "--subsample-rows",
        "--grad-clip-norm",
        "--weight-decay",
    )
    tree = ast.parse(source)
    observed: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in required_flags
        ):
            continue
        observed[str(node.args[0].value)] = node
    assert set(observed) == set(required_flags)
    for flag, node in observed.items():
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        assert isinstance(keywords.get("required"), ast.Constant)
        assert keywords["required"].value is True, flag
        assert "default" not in keywords, flag


def test_env_reader_rejects_a_key_the_recipe_does_not_own() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    with pytest.raises(RuntimeError, match="RECIPE_KEY_UNKNOWN"):
        trainer._env_str("ENTRY_NOT_A_RECIPE_KEY")
