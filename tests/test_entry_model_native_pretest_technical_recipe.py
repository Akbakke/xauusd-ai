from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

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
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _require_pretest_recipe_cli_match,
)
from gx1.scripts.materialize_entry_pretest_technical_recipe_v1 import (
    materialize_pretest_technical_recipe,
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
    trainer_cli = {
        "execution_tier": "attended_only",
        "device": "cuda",
        "seed": 1337,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.0003,
        "seq_len": 96,
        "early_stop_patience": 1,
        "early_stop_min_delta": 0.0,
        "minimum_epochs_before_stop": 1,
        "save_top_k": 1,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.00001,
        "dropout": 0.05,
        "multi_tf_scale": 0.5,
        "specialist_fusion_scale": 0.25,
        "cross_family_fusion_scale": 0.25,
        "subsample_rows": 32,
        "num_workers": 0,
        "multi_tf_num_layers": 2,
        "specialist_num_layers": 1,
        "grad_accum_steps": 1,
        "per_tf_seq_len_m5": 16,
        "per_tf_seq_len_m15": 64,
        "per_tf_seq_len_h1": 96,
        "per_tf_seq_len_h4": 96,
        "per_tf_seq_len_d1": 252,
        "gx1_data_root": "/tmp/gx1-data",
        "train_time_window": {
            "start_utc": "2024-12-01T00:00:00+00:00",
            "end_utc": "2025-06-01T00:00:00+00:00",
        },
    }
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


def test_pretest_recipe_materializer_rejects_noncanonical_wrapper_before_inputs(
    tmp_path: Path,
) -> None:
    """A bad wrapper must fail before the materializer can touch data inputs."""

    repo = (tmp_path / "repo").resolve()
    wrong_wrapper = (repo / "scripts" / "wrong-wrapper.sh").resolve()
    expected_wrapper = (
        repo / "gx1" / "scripts" / "run_entry_model_native_pretest_technical_train_v1.py"
    ).resolve()
    wrong_wrapper.parent.mkdir(parents=True)
    wrong_wrapper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    expected_wrapper.parent.mkdir(parents=True)
    expected_wrapper.write_text("# canonical wrapper fixture\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="WRAPPER_PATH_INVALID"):
        materialize_pretest_technical_recipe(
            repo=repo,
            wrapper_path=wrong_wrapper,
            profile="smoke",
            run_id=RUN_ID,
            dataset_dir=tmp_path / "missing-dataset",
            out_bundle_dir=tmp_path / "bundle",
            test_guard_json=tmp_path / "guard.json",
            test_guard_sha256="a" * 64,
            train_manifest=tmp_path / "train.manifest.json",
            train_parquet=tmp_path / "train.parquet",
            val_manifest=tmp_path / "val.manifest.json",
            val_parquet=tmp_path / "val.parquet",
            dataset_build_proof=tmp_path / "proof.json",
            full_input_liveness=tmp_path / "liveness.json",
            feature_audit=tmp_path / "feature.json",
            target_audit=tmp_path / "target.json",
            specialist_audit=tmp_path / "specialist.json",
            execution_causality_audit=tmp_path / "causality.json",
            train_sequence_source_reconstruction=tmp_path / "train-sequence.json",
            val_sequence_source_reconstruction=tmp_path / "val-sequence.json",
            unified_exit_lifecycle_manifest=tmp_path / "lifecycle.json",
            m5_prebuilt=tmp_path / "m5.parquet",
            multi_tf_cache_manifest=tmp_path / "cache.json",
            trainer_cli={},
            out_json=tmp_path / "recipe.json",
            created_utc="2026-08-30T01:02:03+00:00",
        )


def test_direct_trainer_rejects_any_cli_drift_from_pretest_recipe(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    cli = recipe["trainer_cli"]
    assert isinstance(cli, dict)
    args = SimpleNamespace(
        recipe_audit_json=recipe_path,
        profile="smoke",
        run_id=RUN_ID,
        dataset_run_id=DATASET_RUN_ID,
        train_parquet=Path(str(recipe["dataset_dir"])) / "entry_train.parquet",
        out_bundle_dir=Path(str(recipe["out_bundle_dir"])),
        execution_tier=cli["execution_tier"], device=cli["device"], seed=cli["seed"],
        epochs=cli["epochs"], batch_size=cli["batch_size"], lr=cli["learning_rate"],
        seq_len=cli["seq_len"], early_stopping_patience=cli["early_stop_patience"],
        early_stopping_min_delta=cli["early_stop_min_delta"],
        minimum_epochs_before_stop=cli["minimum_epochs_before_stop"], save_top_k=cli["save_top_k"],
        grad_clip_norm=cli["grad_clip_norm"], weight_decay=cli["weight_decay"], dropout=cli["dropout"],
        multi_tf_scale=cli["multi_tf_scale"], specialist_fusion_scale=cli["specialist_fusion_scale"],
        cross_family_fusion_scale=cli["cross_family_fusion_scale"], subsample_rows=cli["subsample_rows"],
        num_workers=cli["num_workers"], multi_tf_num_layers=cli["multi_tf_num_layers"],
        specialist_num_layers=cli["specialist_num_layers"], grad_accum_steps=cli["grad_accum_steps"],
        per_tf_seq_len_m5=cli["per_tf_seq_len_m5"], per_tf_seq_len_m15=cli["per_tf_seq_len_m15"],
        per_tf_seq_len_h1=cli["per_tf_seq_len_h1"], per_tf_seq_len_h4=cli["per_tf_seq_len_h4"],
        per_tf_seq_len_d1=cli["per_tf_seq_len_d1"], gx1_data=cli["gx1_data_root"],
        train_time_window_start_utc=cli["train_time_window"]["start_utc"],
        train_time_window_end_utc=cli["train_time_window"]["end_utc"],
    )
    _require_pretest_recipe_cli_match(args)
    args.batch_size = 16
    with pytest.raises(RuntimeError, match="CLI_MISMATCH"):
        _require_pretest_recipe_cli_match(args)
