from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_SIGNAL_DIM
from gx1.contracts.entry_model_native_train_launch_v1 import (
    LaunchContractError,
    _validate_feature_audit_signal_partition,
)
from tests.entry_model_native_train_wrapper_support import (
    DATASET_RUN_ID,
    RUN_ID,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_smoke_train.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    updated[updated.index(flag) + 1] = value
    return updated


def test_smoke_wrapper_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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


def test_smoke_wrapper_validates_exact_contract_without_writes(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="smoke", wrapper=WRAPPER)

    result = _run(*args, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 smoke contract" in result.stdout
    assert "Capped smoke train command:" in result.stdout
    assert "gx1_capped_run.sh" in result.stdout
    assert f"--dataset-run-id {DATASET_RUN_ID}" in result.stdout
    assert "GX1_ENTRY_M5_PREBUILT_SHA256=" in result.stdout
    assert "--specialist-audit-json" in result.stdout
    assert "--mtf-dir-scale-init" not in result.stdout
    assert "--enable-" not in result.stdout
    assert not paths["out_bundle_dir"].exists()


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
    assert "--recipe-audit-json" in text
    assert "--pretrain-audit-json" in text
    assert "--full-input-liveness-audit-json" in text
    assert "--execute" in text and "--run-id" in text
    assert 'RUN_CMD=("$CAPPED_RUNNER" --mem "$MEMORY_CAP" --swap "$SWAP_CAP" --' in text
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
        "neutral_xgb",
        "anchor_gate",
        "gx1_allow_legacy",
    ):
        assert stale not in lowered
