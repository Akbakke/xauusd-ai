"""Materialize a bounded TRAIN/VAL-only recipe for the canonical trainer.

No TEST artifact argument exists.  The sole TEST-related input is the
unopened-TEST guard event, whose validator reads only its own control-plane
bytes and rejects any physical TEST lineage for this route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    require_pretest_or_prefreeze_test_guard_lineage,
)
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    DECISION,
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION,
    SIDE_EFFECTS_ZERO,
    TECHNICAL_SCOPE,
    canonical_json_sha256,
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    recipe_source_bindings,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PRETEST_TECHNICAL_RECIPE_{label}_MISSING")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"PRETEST_TECHNICAL_RECIPE_{label}_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"PRETEST_TECHNICAL_RECIPE_{label}_INVALID")
    return payload


def _regular(path: str | Path, *, label: str, forbid_test: bool = True) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or candidate != candidate.resolve(strict=False)
        or candidate.is_symlink()
        or not candidate.is_file()
        or (forbid_test and "test" in candidate.name.lower())
    ):
        raise RuntimeError(f"PRETEST_TECHNICAL_RECIPE_{label}_PATH_INVALID")
    return candidate


def _physical_test_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "test" or "_test" in lowered or "-test" in lowered


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _require_pass(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("decision") != "PASS":
        raise RuntimeError(f"PRETEST_TECHNICAL_RECIPE_{label}_NOT_PASS")


def materialize_pretest_technical_recipe(
    *,
    repo: Path,
    wrapper_path: Path,
    profile: str,
    run_id: str,
    dataset_dir: Path,
    out_bundle_dir: Path,
    test_guard_json: Path,
    test_guard_sha256: str,
    train_manifest: Path,
    train_parquet: Path,
    val_manifest: Path,
    val_parquet: Path,
    dataset_build_proof: Path,
    full_input_liveness: Path,
    feature_audit: Path,
    target_audit: Path,
    specialist_audit: Path,
    execution_causality_audit: Path,
    train_sequence_source_reconstruction: Path,
    val_sequence_source_reconstruction: Path,
    unified_exit_lifecycle_manifest: Path,
    m5_prebuilt: Path,
    multi_tf_cache_manifest: Path,
    trainer_cli: Mapping[str, Any],
    out_json: Path,
    created_utc: str,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    wrapper_path = wrapper_path.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_DATASET_DIR_INVALID")
    if profile not in {"smoke", "candidate"}:
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_PROFILE_INVALID")
    run_id = require_entry_run_id(run_id)
    file_args = {
        "train_manifest": train_manifest,
        "train_parquet": train_parquet,
        "val_manifest": val_manifest,
        "val_parquet": val_parquet,
        "dataset_build_proof": dataset_build_proof,
        "full_input_liveness": full_input_liveness,
        "feature_audit": feature_audit,
        "target_audit": target_audit,
        "specialist_audit": specialist_audit,
        "execution_causality_audit": execution_causality_audit,
        "train_sequence_source_reconstruction": train_sequence_source_reconstruction,
        "val_sequence_source_reconstruction": val_sequence_source_reconstruction,
        "unified_exit_lifecycle_manifest": unified_exit_lifecycle_manifest,
        "m5_prebuilt": m5_prebuilt,
        "multi_tf_cache_manifest": multi_tf_cache_manifest,
    }
    if frozenset(file_args) != REQUIRED_ARTIFACTS:
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_ARTIFACT_SET_INVALID")
    files = {name: _regular(path, label=name.upper()) for name, path in file_args.items()}
    if any(files[name].parent != dataset_dir for name in (
        "train_manifest", "train_parquet", "val_manifest", "val_parquet", "dataset_build_proof",
        "full_input_liveness", "execution_causality_audit",
    )):
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_DATASET_ARTIFACT_PATH_INVALID")
    if out_bundle_dir.exists() or out_bundle_dir.is_symlink() or not out_bundle_dir.is_absolute():
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_OUTPUT_BUNDLE_PATH_INVALID")
    if (
        not out_json.is_absolute()
        or out_json.exists()
        or out_json.is_symlink()
        or not out_json.parent.is_dir()
        or _physical_test_name(out_json.name)
    ):
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_OUTPUT_PATH_INVALID")
    proof = _json(files["dataset_build_proof"], label="DATASET_BUILD_PROOF")
    dataset_run_id = require_entry_run_id(proof.get("entry_run_id"))
    pretest_guard = proof.get("pretest_test_guard")
    if (
        proof.get("pretest_only") is not True
        or not isinstance(pretest_guard, Mapping)
        or pretest_guard.get("test_accessed") is not False
    ):
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_DATASET_PROOF_NOT_STRICT_PRETEST")
    guard = require_pretest_or_prefreeze_test_guard_lineage(
        test_guard_json,
        test_guard_sha256,
        expected_dataset_run_id=dataset_run_id,
        expected_dataset_dir=dataset_dir,
    )
    if guard.get("schema_version") != "entry_model_native_pretest_test_guard_lineage_v1":
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_PHYSICAL_TEST_SEAL_FORBIDDEN")
    for name in (
        "full_input_liveness", "feature_audit", "target_audit", "specialist_audit",
        "execution_causality_audit", "train_sequence_source_reconstruction",
        "val_sequence_source_reconstruction",
    ):
        _require_pass(_json(files[name], label=name.upper()), label=name.upper())
    manifests = {
        split: _json(files[f"{split}_manifest"], label=f"{split.upper()}_MANIFEST")
        for split in ("train", "val")
    }
    for split, manifest in manifests.items():
        extra = manifest.get("extra")
        if (
            not isinstance(extra, Mapping)
            or extra.get("entry_run_id") != dataset_run_id
            or extra.get("pretest_only") is not True
            or manifest.get("output_data_path") != str(files[f"{split}_parquet"])
        ):
            raise RuntimeError("PRETEST_TECHNICAL_RECIPE_SPLIT_MANIFEST_INVALID")
    artifacts = {name: _artifact(path) for name, path in files.items()}
    for split in ("train", "val"):
        for kind in ("manifest", "parquet"):
            key = f"{split}_{kind}"
            if artifacts[key] != guard[key]:
                raise RuntimeError("PRETEST_TECHNICAL_RECIPE_GUARD_BINDING_MISMATCH")
    if not isinstance(trainer_cli, Mapping) or not trainer_cli:
        raise RuntimeError("PRETEST_TECHNICAL_RECIPE_TRAINER_CLI_INVALID")
    source_bindings = recipe_source_bindings(repo=repo, wrapper_path=wrapper_path)
    recipe: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "created_utc": created_utc,
        "technical_scope": TECHNICAL_SCOPE,
        "profile": profile,
        "run_id": run_id,
        "dataset_run_id": dataset_run_id,
        "dataset_dir": str(dataset_dir),
        "out_bundle_dir": str(out_bundle_dir),
        "test_guard_lineage": guard,
        "artifact_bindings": artifacts,
        "artifact_bindings_sha256": canonical_json_sha256(artifacts),
        "trainer_cli": dict(trainer_cli),
        "trainer_cli_sha256": canonical_json_sha256(trainer_cli),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
        "activation_authority": False,
        "report_only": True,
        "side_effects_started": dict(SIDE_EFFECTS_ZERO),
    }
    require_pretest_technical_recipe_metadata(
        recipe,
        expected_profile=profile,
        expected_run_id=run_id,
        expected_dataset_run_id=dataset_run_id,
        expected_dataset_dir=dataset_dir,
        expected_out_bundle_dir=out_bundle_dir,
    )
    raw = (json.dumps(recipe, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(out_json, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    return {"path": str(out_json), "sha256": _sha256_file(out_json), "recipe": recipe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wrapper-path", required=True)
    parser.add_argument("--profile", choices=("smoke", "candidate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-bundle-dir", required=True)
    parser.add_argument("--test-guard-json", required=True)
    parser.add_argument("--test-guard-sha256", required=True)
    for name in sorted(REQUIRED_ARTIFACTS):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--trainer-cli-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    try:
        trainer_cli = json.loads(args.trainer_cli_json)
    except ValueError as exc:
        parser.error(f"--trainer-cli-json must be one JSON object: {exc}")
    result = materialize_pretest_technical_recipe(
        repo=Path(args.repo),
        wrapper_path=Path(args.wrapper_path),
        profile=args.profile,
        run_id=args.run_id,
        dataset_dir=Path(args.dataset_dir),
        out_bundle_dir=Path(args.out_bundle_dir),
        test_guard_json=Path(args.test_guard_json),
        test_guard_sha256=args.test_guard_sha256,
        train_manifest=Path(args.train_manifest),
        train_parquet=Path(args.train_parquet),
        val_manifest=Path(args.val_manifest),
        val_parquet=Path(args.val_parquet),
        dataset_build_proof=Path(args.dataset_build_proof),
        full_input_liveness=Path(args.full_input_liveness),
        feature_audit=Path(args.feature_audit),
        target_audit=Path(args.target_audit),
        specialist_audit=Path(args.specialist_audit),
        execution_causality_audit=Path(args.execution_causality_audit),
        train_sequence_source_reconstruction=Path(args.train_sequence_source_reconstruction),
        val_sequence_source_reconstruction=Path(args.val_sequence_source_reconstruction),
        unified_exit_lifecycle_manifest=Path(args.unified_exit_lifecycle_manifest),
        m5_prebuilt=Path(args.m5_prebuilt),
        multi_tf_cache_manifest=Path(args.multi_tf_cache_manifest),
        trainer_cli=trainer_cli,
        out_json=Path(args.out_json),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
