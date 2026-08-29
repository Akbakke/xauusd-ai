from __future__ import annotations

from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PRETEST_TEST_GUARD_ACCESS_POLICY,
    PRETEST_TEST_GUARD_DECISION,
    PRETEST_TEST_GUARD_EVENT_PREFIX,
    PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION,
    PRETEST_TEST_GUARD_SCHEMA_VERSION,
    PRETEST_TEST_GUARD_VERIFICATION_MODE,
)
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    DECISION,
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION,
    SIDE_EFFECTS_ZERO,
    TECHNICAL_SCOPE,
    PretestTechnicalRecipeError,
    canonical_json_sha256,
    require_pretest_technical_recipe_metadata,
)


RUN_ID = "PRETEST_TECHNICAL_RECIPE_RUN_V1"
DATASET_RUN_ID = "PRETEST_TECHNICAL_DATASET_V1"


def _recipe(tmp_path: Path) -> dict[str, object]:
    dataset = (tmp_path / "dataset").resolve()
    authority = (tmp_path / "authority").resolve()
    dataset.mkdir()
    authority.mkdir()
    sha = "a" * 64
    artifacts: dict[str, dict[str, str]] = {}
    names = {
        "train_manifest": "entry_train.manifest.json",
        "val_manifest": "entry_val.manifest.json",
        "train_parquet": "entry_train.parquet",
        "val_parquet": "entry_val.parquet",
        "dataset_build_proof": "DATASET_BUILD_PROOF.json",
        "full_input_liveness": "ENTRY_FULL_INPUT_LIVENESS_20260830T010000Z.json",
        "feature_audit": "AUDIT_FEATURE/feature.json",
        "target_audit": "AUDIT_TARGET/target.json",
        "specialist_audit": "AUDIT_SPECIALIST/specialist.json",
        "execution_causality_audit": "execution.json",
        "train_sequence_source_reconstruction": "sequence_train.json",
        "val_sequence_source_reconstruction": "sequence_val.json",
        "unified_exit_lifecycle_manifest": "UNIFIED_EXIT_LIFECYCLE/manifest.json",
        "m5_prebuilt": "m5_enriched.parquet",
        "multi_tf_cache_manifest": "MULTI_TF/manifest.json",
    }
    for name in REQUIRED_ARTIFACTS:
        location = Path(names[name])
        artifacts[name] = {
            "path": str((dataset / location) if name not in {"m5_prebuilt", "multi_tf_cache_manifest"} else (tmp_path / location)),
            "sha256": sha,
        }
    guard = {
        "schema_version": PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION,
        "verification_mode": PRETEST_TEST_GUARD_VERIFICATION_MODE,
        "guard_event": {
            "path": str(authority / f"{PRETEST_TEST_GUARD_EVENT_PREFIX}_20260830T010203000000Z.json"),
            "sha256": sha,
            "schema_version": PRETEST_TEST_GUARD_SCHEMA_VERSION,
            "decision": PRETEST_TEST_GUARD_DECISION,
            "created_utc": "2026-08-30T01:02:03+00:00",
            "content_binding_sha256": sha,
        },
        "dataset_run_id": DATASET_RUN_ID,
        "dataset_dir": str(dataset),
        "split": "test",
        "access_policy": PRETEST_TEST_GUARD_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_boundary_utc": "2026-07-01T00:00:00+00:00",
        "train_manifest": artifacts["train_manifest"],
        "train_parquet": artifacts["train_parquet"],
        "val_manifest": artifacts["val_manifest"],
        "val_parquet": artifacts["val_parquet"],
        "dataset_build_proof": artifacts["dataset_build_proof"],
        "full_input_liveness": artifacts["full_input_liveness"],
        "access_proof": {
            "guard_event_bytes_hash_validated": True,
            "test_dataset_bytes_read": False,
            "test_manifest_bytes_read": False,
            "test_metrics_read": False,
            "test_paths_resolved_or_statted": False,
        },
    }
    source_bindings = {
        "python:trainer.py": {
            "path": str((tmp_path / "trainer.py").resolve()),
            "sha256": sha,
            "size_bytes": 1,
            "mtime_ns": 1,
            "device": 1,
            "inode": 1,
        }
    }
    trainer_cli = {"device": "cuda", "epochs": 1, "batch_size": 8}
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "created_utc": "2026-08-30T01:02:03+00:00",
        "technical_scope": TECHNICAL_SCOPE,
        "profile": "smoke",
        "run_id": RUN_ID,
        "dataset_run_id": DATASET_RUN_ID,
        "dataset_dir": str(dataset),
        "out_bundle_dir": str((tmp_path / "bundle").resolve()),
        "test_guard_lineage": guard,
        "artifact_bindings": artifacts,
        "artifact_bindings_sha256": canonical_json_sha256(artifacts),
        "trainer_cli": trainer_cli,
        "trainer_cli_sha256": canonical_json_sha256(trainer_cli),
        "source_commit": "b" * 40,
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
        "activation_authority": False,
        "report_only": True,
        "side_effects_started": SIDE_EFFECTS_ZERO,
    }


def test_pretest_recipe_accepts_only_a_bound_unopened_test_guard(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path)
    assert require_pretest_technical_recipe_metadata(
        recipe,
        expected_profile="smoke",
        expected_run_id=RUN_ID,
        expected_dataset_run_id=DATASET_RUN_ID,
        expected_dataset_dir=recipe["dataset_dir"],
        expected_out_bundle_dir=recipe["out_bundle_dir"],
    ) == recipe


def test_pretest_recipe_rejects_artifact_not_bound_to_guard(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path)
    bindings = dict(recipe["artifact_bindings"])
    bindings["train_parquet"] = dict(bindings["train_parquet"], sha256="b" * 64)
    recipe["artifact_bindings"] = bindings
    recipe["artifact_bindings_sha256"] = canonical_json_sha256(bindings)
    with pytest.raises(PretestTechnicalRecipeError, match="differs from unopened-TEST guard"):
        require_pretest_technical_recipe_metadata(recipe)
