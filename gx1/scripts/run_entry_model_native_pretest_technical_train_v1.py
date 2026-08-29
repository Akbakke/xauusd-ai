"""Launch exactly one recipe-bound, TRAIN/VAL-only technical CUDA smoke.

This is intentionally a narrow control surface.  It accepts an immutable
pre-TEST recipe and its digest, derives every trainer argument and permitted
environment variable from it, and then enters the normal capped trainer.  It
has no TEST input, no override flag and no candidate/promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    PretestTechnicalRecipeError,
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.contracts.entry_model_native_train_launch_v1 import (
    LaunchContractError,
    require_training_recipe_execution_provenance,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import MODEL_NATIVE_RECIPE_ENV


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "bin" / "python"
CAPPED_RUNNER = REPO / "scripts" / "gx1_capped_run.sh"
PRETEST_WRAPPER = Path(__file__).resolve()
_SHA256_HEX_LENGTH = 64

_ARTIFACT_ENV = {
    "train_manifest": "GX1_ENTRY_TRAIN_MANIFEST_SHA256",
    "val_manifest": "GX1_ENTRY_VAL_MANIFEST_SHA256",
    "train_parquet": "GX1_ENTRY_TRAIN_PARQUET_SHA256",
    "val_parquet": "GX1_ENTRY_VAL_PARQUET_SHA256",
    "m5_prebuilt": "GX1_ENTRY_M5_PREBUILT_SHA256",
    "unified_exit_lifecycle_manifest": (
        "GX1_ENTRY_UNIFIED_EXIT_LIFECYCLE_MANIFEST_SHA256"
    ),
    "train_sequence_source_reconstruction": (
        "GX1_ENTRY_TRAIN_SEQUENCE_SOURCE_AUDIT_SHA256"
    ),
    "val_sequence_source_reconstruction": (
        "GX1_ENTRY_VAL_SEQUENCE_SOURCE_AUDIT_SHA256"
    ),
}


class PretestTechnicalLaunchError(RuntimeError):
    """The pre-TEST technical launch surface is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_absolute(path: Path, *, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
    ):
        raise PretestTechnicalLaunchError(f"{label} must be an absolute regular file")
    return path


def _recipe(path: Path, expected_sha256: str) -> dict[str, Any]:
    recipe_path = _regular_absolute(path, label="recipe JSON")
    if len(expected_sha256) != _SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise PretestTechnicalLaunchError("recipe SHA-256 must be lowercase hex")
    if _sha256_file(recipe_path) != expected_sha256:
        raise PretestTechnicalLaunchError("recipe JSON does not match declared SHA-256")
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PretestTechnicalLaunchError("recipe JSON is not valid") from exc
    if not isinstance(payload, dict):
        raise PretestTechnicalLaunchError("recipe JSON root must be an object")
    return payload


def build_pretest_technical_launch(
    *,
    recipe_path: Path,
    recipe_sha256: str,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Validate the immutable recipe and derive the sole allowed command."""

    if not PYTHON.is_file() or not CAPPED_RUNNER.is_file():
        raise PretestTechnicalLaunchError("canonical Python or capped runner is unavailable")
    recipe = _recipe(recipe_path, recipe_sha256)
    try:
        validated = require_pretest_technical_recipe_metadata(recipe)
        artifact = validated["artifact_bindings"]
        cli = validated["trainer_cli"]
        provenance = require_training_recipe_execution_provenance(
            recipe_audit_path=recipe_path,
            recipe_audit_sha256=recipe_sha256,
            repo=REPO,
            profile=str(validated["profile"]),
            run_id=str(validated["run_id"]),
            dataset_run_id=str(validated["dataset_run_id"]),
            dataset_dir=Path(str(validated["dataset_dir"])),
            out_bundle_dir=Path(str(validated["out_bundle_dir"])),
        )
    except (LaunchContractError, PretestTechnicalRecipeError, OSError, ValueError) as exc:
        raise PretestTechnicalLaunchError(f"recipe provenance rejected: {exc}") from exc
    if not isinstance(artifact, Mapping) or not isinstance(cli, Mapping):
        raise PretestTechnicalLaunchError("validated recipe lost its exact bindings")
    if provenance.get("source_commit") != validated["source_commit"]:
        raise PretestTechnicalLaunchError("source provenance does not match recipe")
    execution_tier = str(cli["execution_tier"])
    device = str(cli["device"])
    if (execution_tier, device) not in {
        ("attended_only", "cuda"),
        ("canonical", "cuda"),
    }:
        raise PretestTechnicalLaunchError(
            "this launcher admits only guarded attended or canonical CUDA smoke"
        )

    def bound_path(name: str) -> str:
        value = artifact.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise PretestTechnicalLaunchError(f"recipe artifact missing: {name}")
        return str(value["path"])

    def bound_sha(name: str) -> str:
        value = artifact.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("sha256"), str):
            raise PretestTechnicalLaunchError(f"recipe artifact digest missing: {name}")
        return str(value["sha256"])

    guard = validated["test_guard_lineage"]
    if not isinstance(guard, Mapping) or not isinstance(guard.get("guard_event"), Mapping):
        raise PretestTechnicalLaunchError("recipe unopened-TEST guard is malformed")
    guard_event = guard["guard_event"]
    guard_path = str(guard_event.get("path") or "")
    guard_sha = str(guard_event.get("sha256") or "")
    if not guard_path or len(guard_sha) != _SHA256_HEX_LENGTH:
        raise PretestTechnicalLaunchError("recipe unopened-TEST guard is incomplete")
    window = cli["train_time_window"]
    if execution_tier == "attended_only" and not isinstance(window, Mapping):
        raise PretestTechnicalLaunchError(
            "attended technical smoke requires a train time window"
        )
    if execution_tier == "canonical" and window is not None:
        raise PretestTechnicalLaunchError(
            "canonical technical smoke must use deterministic uniform sampling"
        )
    cache_manifest = Path(bound_path("multi_tf_cache_manifest"))
    if cache_manifest.name != "manifest.json":
        raise PretestTechnicalLaunchError("multi-TF cache identity is invalid")

    trainer_command = [
        str(PYTHON), "-m", "gx1.models.entry_v10.entry_v10_ctx_train_v3",
        "--train", "--profile", str(validated["profile"]),
        "--execution-tier", str(cli["execution_tier"]),
        "--run-id", str(validated["run_id"]),
        "--dataset-run-id", str(validated["dataset_run_id"]),
        "--seed", str(cli["seed"]), "--device", str(cli["device"]),
        "--batch_size", str(cli["batch_size"]), "--epochs", str(cli["epochs"]),
        "--lr", str(cli["learning_rate"]), "--seq_len", str(cli["seq_len"]),
        "--train-manifest-json", bound_path("train_manifest"),
        "--val-manifest-json", bound_path("val_manifest"),
        "--train-parquet", bound_path("train_parquet"),
        "--val-parquet", bound_path("val_parquet"),
        "--train-sequence-source-audit-json", bound_path("train_sequence_source_reconstruction"),
        "--val-sequence-source-audit-json", bound_path("val_sequence_source_reconstruction"),
        "--recipe-audit-json", str(recipe_path), "--recipe-audit-sha256", recipe_sha256,
        "--prefreeze-test-seal-json", guard_path, "--prefreeze-test-seal-sha256", guard_sha,
        "--unified-exit-lifecycle-manifest-json", bound_path("unified_exit_lifecycle_manifest"),
        "--out_bundle_dir", str(validated["out_bundle_dir"]),
        "--gx1-data", str(cli["gx1_data_root"]), "--num-workers", str(cli["num_workers"]),
        "--early-stopping-patience", str(cli["early_stop_patience"]),
        "--early-stopping-min-delta", str(cli["early_stop_min_delta"]),
        "--minimum-epochs-before-stop", str(cli["minimum_epochs_before_stop"]),
        "--save-top-k", str(cli["save_top_k"]), "--m5-prebuilt-path", bound_path("m5_prebuilt"),
        "--multi-tf-num-layers", str(cli["multi_tf_num_layers"]),
        "--per-tf-seq-len-m5", str(cli["per_tf_seq_len_m5"]),
        "--per-tf-seq-len-m15", str(cli["per_tf_seq_len_m15"]),
        "--per-tf-seq-len-h1", str(cli["per_tf_seq_len_h1"]),
        "--per-tf-seq-len-h4", str(cli["per_tf_seq_len_h4"]),
        "--per-tf-seq-len-d1", str(cli["per_tf_seq_len_d1"]),
        "--multi-tf-scale", str(cli["multi_tf_scale"]),
        "--specialist-audit-json", bound_path("specialist_audit"),
        "--specialist-contract-mode", MODEL_NATIVE_CONTRACT_MODE,
        "--specialist-num-layers", str(cli["specialist_num_layers"]),
        "--specialist-fusion-scale", str(cli["specialist_fusion_scale"]),
        "--cross-family-fusion-scale", str(cli["cross_family_fusion_scale"]),
        "--grad-accum-steps", str(cli["grad_accum_steps"]),
        "--subsample-rows", str(cli["subsample_rows"]),
        "--grad-clip-norm", str(cli["grad_clip_norm"]),
        "--weight-decay", str(cli["weight_decay"]), "--dropout", str(cli["dropout"]),
    ]
    if isinstance(window, Mapping):
        trainer_command.extend((
            "--train-time-window-start-utc", str(window["start_utc"]),
            "--train-time-window-end-utc", str(window["end_utc"]),
        ))
    environment = dict(MODEL_NATIVE_RECIPE_ENV)
    environment.update({
        env_name: bound_sha(name)
        for name, env_name in _ARTIFACT_ENV.items()
    })
    environment["GX1_ENTRY_DATASET_RUN_ID"] = str(validated["dataset_run_id"])
    environment["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(cache_manifest.parent)
    # Canonical smoke remains guarded by gx1_capped_run's normal CUDA path;
    # only the attended diagnostic needs the shorter attended-only envelope.
    command = [
        str(CAPPED_RUNNER), "--class", "trainer", "--mem", "20G", "--swap", "512M",
    ]
    if execution_tier == "attended_only":
        command.append("--attended-smoke")
    command.extend(("--", *trainer_command))
    return command, environment, validated


def _scrubbed_environment(exact: Mapping[str, str]) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ENTRY_", "GX1_")) and key != "PYTHONPATH"
    }
    environment.update(exact)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        command, environment, recipe = build_pretest_technical_launch(
            recipe_path=args.recipe_json,
            recipe_sha256=str(args.recipe_sha256),
        )
    except PretestTechnicalLaunchError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({
            "schema_version": "gx1_pretest_technical_launch_dry_run_v1",
            "decision": "PASS",
            "run_id": recipe["run_id"],
            "test_accessed": False,
            "command": command,
            "environment": environment,
        }, sort_keys=True))
        return
    os.chdir(REPO)
    os.execvpe(command[0], command, _scrubbed_environment(environment))


if __name__ == "__main__":
    main()
