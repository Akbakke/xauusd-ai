from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_SIGNAL_DIM
from gx1.contracts.entry_model_native_train_launch_v1 import (
    LaunchContractError,
    _validate_feature_audit_signal_partition,
    artifact_binding,
    canonical_json_sha256,
)
from tests.entry_model_native_train_wrapper_support import (
    DATASET_RUN_ID,
    RUN_ID,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_train.sh"


def _run_raw(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_raw("--profile", "smoke", *args)


def _replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    updated[updated.index(flag) + 1] = value
    return updated


def _capped_command_tokens(stdout: str) -> list[str]:
    line = next(
        row for row in stdout.splitlines() if row.startswith("Capped smoke train command:")
    )
    return shlex.split(line.split(":", 1)[1].strip())


def test_smoke_wrapper_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_wrapper_is_the_only_physical_seq513_train_wrapper() -> None:
    assert WRAPPER.is_file()
    assert not WRAPPER.is_symlink()
    assert sorted(
        path.name
        for path in (REPO / "scripts").glob(
            "run_entry_model_native_seq513*train.sh"
        )
    ) == [WRAPPER.name]


def test_smoke_wrapper_requires_complete_explicit_contract() -> None:
    result = _run("--run-id", RUN_ID, "--dry-run")

    assert result.returncode == 2
    assert "missing required argument" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_smoke_wrapper_rejects_unknown_or_duplicate_arguments() -> None:
    unknown = _run("--smart-seq520", "--dry-run")
    duplicate = _run("--run-id", RUN_ID, "--run-id", RUN_ID, "--dry-run")

    assert unknown.returncode == 2
    assert "unknown argument" in unknown.stderr
    assert duplicate.returncode == 2
    assert "duplicate argument" in duplicate.stderr


def test_canonical_wrapper_rejects_missing_wrong_or_duplicate_profile() -> None:
    missing = _run_raw("--dry-run")
    wrong = _run_raw("--profile", "other", "--dry-run")
    duplicate = _run_raw(
        "--profile", "smoke", "--profile", "candidate", "--dry-run"
    )

    assert missing.returncode == 2
    assert "missing required argument: --profile" in missing.stderr
    assert wrong.returncode == 2
    assert "--profile must be exactly smoke or candidate" in wrong.stderr
    assert duplicate.returncode == 2
    assert "duplicate argument: --profile" in duplicate.stderr


@pytest.mark.parametrize(
    ("profile", "foreign_flag"),
    (
        ("smoke", "--candidate-readiness-json"),
        ("candidate", "--smoke-manifest-json"),
    ),
)
def test_canonical_wrapper_rejects_cross_profile_evidence(
    tmp_path: Path,
    profile: str,
    foreign_flag: str,
) -> None:
    args, paths = build_wrapper_contract(
        tmp_path,
        profile=profile,
        wrapper=WRAPPER,
    )
    result = _run_raw(
        "--profile",
        profile,
        *args,
        foreign_flag,
        str(paths["recipe_audit_json"]),
        "--dry-run",
    )

    assert result.returncode == 2
    assert f"{foreign_flag} is invalid for --profile {profile}" in result.stderr


def test_smoke_wrapper_validates_exact_contract_without_writes(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    env = os.environ.copy()
    env["ENTRY_STALE_WRAPPER_TEST"] = "must_be_scrubbed"
    env["GX1_STALE_WRAPPER_TEST"] = "must_be_scrubbed"
    result = subprocess.run(
        ["bash", str(WRAPPER), "--profile", "smoke", *args, "--dry-run"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 smoke contract" in result.stdout
    assert "Capped smoke train command:" in result.stdout
    assert "gx1_capped_run.sh" in result.stdout
    assert f"--dataset-run-id {DATASET_RUN_ID}" in result.stdout
    assert "GX1_ENTRY_M5_PREBUILT_SHA256=" in result.stdout
    assert "--prefreeze-test-seal-json" in result.stdout
    assert "--prefreeze-test-seal-sha256" in result.stdout
    assert "--specialist-audit-json" in result.stdout
    assert "--mtf-dir-scale-init" not in result.stdout
    assert "--enable-" not in result.stdout
    assert not paths["out_bundle_dir"].exists()
    command = _capped_command_tokens(result.stdout)
    runner_index = command.index(str(REPO / "scripts/gx1_capped_run.sh"))
    separator_index = command.index("--", runner_index)
    assert command[0] == "/usr/bin/env"
    assert command[runner_index + 1 : runner_index + 4] == [
        "--class",
        "trainer",
        "--mem",
    ]
    assert command[separator_index + 1 : separator_index + 4] == [
        str(REPO / ".venv/bin/python"),
        "-m",
        "gx1.models.entry_v10.entry_v10_ctx_train_v3",
    ]
    assert command[separator_index + 4] == "--train"
    for stale_key in ("ENTRY_STALE_WRAPPER_TEST", "GX1_STALE_WRAPPER_TEST"):
        stale_index = command.index(stale_key)
        assert command[stale_index - 1] == "-u"
        assert stale_index < runner_index
        assert stale_key not in command[separator_index + 1 :]


def test_smoke_launch_never_opens_or_stats_sealed_test_artifacts(
    tmp_path: Path,
) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    paths["test_manifest_json"].unlink()
    paths["test_parquet"].unlink()

    result = _run(*args, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Capped smoke train command:" in result.stdout
    assert "--prefreeze-test-seal-json" in result.stdout
    assert "--test-manifest-json" not in result.stdout
    assert "--test-parquet" not in result.stdout


def test_smoke_launch_rejects_changed_prefreeze_test_seal_event(
    tmp_path: Path,
) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    seal = paths["prefreeze_test_seal_json"]
    seal.write_bytes(seal.read_bytes() + b" ")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "pre-freeze TEST seal rejected" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_smoke_launch_rejects_extra_test_metrics_isolation_channel(
    tmp_path: Path,
) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    post_rebuild_path = paths["post_rebuild_readiness_json"]
    post_rebuild = json.loads(post_rebuild_path.read_text(encoding="utf-8"))
    post_rebuild["test_isolation"]["test_metrics_path"] = str(
        (tmp_path / "forbidden_test_metrics.json").resolve()
    )
    post_rebuild_path.write_text(
        json.dumps(post_rebuild, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["artifact_bindings"]["post_rebuild_readiness_json"] = artifact_binding(
        post_rebuild_path
    )
    recipe["artifact_bindings_sha256"] = canonical_json_sha256(
        recipe["artifact_bindings"]
    )
    recipe_path.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "TEST isolation proof keys are not exact" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_train_launch_rejects_nonproduction_mtf_window(tmp_path: Path) -> None:
    args, _paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    result = _run(
        *_replace_arg(args, "--per-tf-seq-len-m5", "15"),
        "--dry-run",
    )

    assert result.returncode == 2
    assert "windows must equal the frozen production architecture" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_train_launch_rejects_legacy_base_field_and_swapped_mandatory_prefix(
    tmp_path: Path,
) -> None:
    _args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    feature = json.loads(paths["feature_audit_json"].read_text(encoding="utf-8"))
    _validate_feature_audit_signal_partition(feature)

    stale_base = dict(feature)
    stale_base.pop("base_signal_dim")
    stale_base["base_seq_dim_v3"] = MODEL_NATIVE_BASE_SIGNAL_DIM
    with pytest.raises(LaunchContractError, match="base signal width mismatch"):
        _validate_feature_audit_signal_partition(stale_base)

    swapped = json.loads(json.dumps(feature))
    contract = swapped["model_native_signal_contract"]
    contract["selected_fields"][0], contract["selected_fields"][1] = (
        contract["selected_fields"][1],
        contract["selected_fields"][0],
    )
    with pytest.raises(
        LaunchContractError,
        match="mandatory_registry_prefix_order_violation",
    ):
        _validate_feature_audit_signal_partition(swapped)


def test_smoke_wrapper_rejects_mutable_pointer_path(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    mutable = paths["recipe_audit_json"].with_name("ENTRY_TRAIN_RECIPE_AUDIT_latest.json")
    mutable.write_bytes(paths["recipe_audit_json"].read_bytes())

    result = _run(*_replace_arg(args, "--recipe-audit-json", str(mutable)), "--dry-run")

    assert result.returncode == 2
    assert "mutable pointer" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_smoke_wrapper_rejects_incomplete_recipe_env(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["trainer_env"].pop("ENTRY_TRENDLINE_RAIL_AUX_WEIGHT")
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "MODEL_NATIVE_RECIPE_ENV_MISMATCH" in result.stderr
    assert "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT" in result.stderr


def test_smoke_wrapper_rejects_recipe_from_unrelated_source_commit(
    tmp_path: Path,
) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["source_commit"] = "a" * 40
    recipe_path.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "source_commit is not an ancestor" in result.stderr


def test_smoke_wrapper_rejects_stale_target_audit_schema(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)
    target_path = paths["target_audit_json"]
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["schema_version"] = "entry_target_foundation_audit_v1"
    target.pop("model_native_aux_target_contract")
    target_path.write_text(
        json.dumps(target, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "target audit schema mismatch" in result.stderr
    assert "Capped smoke train command:" not in result.stdout


def test_smoke_wrapper_source_is_exact_model_native_and_has_no_stale_launch_paths() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "MODEL_NATIVE_CONTRACT_MODE=xau_seq513_model_native_direction_v4" in text
    assert "MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native" in text
    assert "MODEL_NATIVE_SIGNAL_DIM=513" in text
    assert "PROFILE=smoke" not in text
    assert "PROFILE=candidate" not in text
    assert "--profile" in text
    assert "--recipe-audit-json" in text
    assert "--pretrain-audit-json" in text
    assert "--prefreeze-test-seal-json" in text
    assert "--prefreeze-test-seal-sha256" in text
    assert "--test-manifest-json" not in text
    assert "--test-parquet" not in text
    assert "test_metrics" not in lowered
    assert "--full-input-liveness-audit-json" in text
    assert "--execute" in text and "--run-id" in text
    assert 'TRAIN_CMD=(\n  "$PY" -m gx1.models.entry_v10.entry_v10_ctx_train_v3' in text
    assert 'RUN_CMD=(\n  "${ENV_COMMAND[@]}"\n  "$CAPPED_RUNNER" --class trainer' in text
    assert '-- "${TRAIN_CMD[@]}"' in text
    assert 'exec "${RUN_CMD[@]}"' in text
    for flag in (
        "--enable-pos-enc",
        "--enable-regime-film",
        "--enable-cross-tf-attn",
        "--enable-tf-agreement-head",
        "--enable-path-quality-variance-head",
        "--enable-position-size-head",
        "--enable-mtf-direction-head",
        "--enable-specialist-fusion",
        "--enable-hierarchical-entry-heads",
        "--enable-side-validity-head",
        "--enable-trendline-rail-head",
    ):
        assert flag not in text
    for stale in (
        "seq520",
        "tombstone",
        "run_manifest",
        "event_ledger",
        "neutral_external_tree_sidecar",
        "anchor_gate",
        "gx1_allow_legacy",
    ):
        assert stale not in lowered
