"""Native full-TRAIN construction for the canonical candidate coordinator.

These internal functions consume already verified recipe/artifact bindings.
They do not provide a standalone launch authority or a second training loop.
"""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from gx1.contracts.unified_exit_random_access_train_factory_v1 import (
    build_random_access_train_adapter_factory_v1,
)
from gx1.contracts import entry_model_native_train_launch_v1 as launch_owner
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    require_selected_weight_ema_checkpoint_binding_v1,
)
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import RESULT_SCHEMA_VERSION
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import native_checkpoint_monitor


NATIVE_FULL_TRAIN_RECIPE_SCHEMA = "gx1_unified_exit_random_access_full_train_recipe_v1"
_DATA_FILES = frozenset({
    "random_access_root", "sampler_candidate_set", "child_composite_normalization",
    "economics_readiness", "train_cost_authority", "source_bundle_metadata",
    "feature_lifecycle_root", "entry_train_parquet", "entry_train_manifest",
    "entry_val_parquet", "entry_val_manifest", "m5_prebuilt", "sequence_source_audit",
})
_INITIALIZATION = "completed_smoke_ema_then_fresh_main_optimizer_ema_scheduler"


def _native_recipe_source_bindings(repo: Path) -> dict[str, Any]:
    wrapper = repo / "gx1/scripts/run_unified_exit_random_access_full_train_v1.py"
    paths = launch_owner.recipe_source_binding_paths(repo=repo, wrapper_path=wrapper)
    campaign_wrapper = repo / "gx1/scripts/run_unified_exit_native_candidate_window_v1.py"
    paths.update(launch_owner._recipe_local_python_import_closure(repo=repo, roots=[wrapper, campaign_wrapper]))
    return {key: launch_owner.artifact_binding(path) for key, path in paths.items()}


def _bound_artifact(binding: Any) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise RuntimeError("NATIVE_FULL_TRAIN_ARTIFACT_BINDING_INVALID")
    path = Path(binding["path"])
    if (
        not path.is_absolute() or not path.is_file() or path.is_symlink()
        or path.resolve() != path or val.file_sha256(path) != binding["sha256"]
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_ARTIFACT_BYTES_INVALID")
    return path


def _require_native_full_train_recipe(
    recipe_path: Path, recipe_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Verify native source/data/seed authority before a guarded invocation.

    A complete smoke VAL measures the seed; it is not candidate or live
    admission. The canonical candidate still applies its strict gate/head
    requirements and selects on its own complete full-VAL net-Bps results.
    """

    recipe = val._read(_bound_artifact({"path": str(recipe_path), "sha256": recipe_file_sha256}))
    required = {
        "schema_version", "profile", "source_repo", "source_commit",
        "source_bindings", "source_bindings_sha256", "run_id", "dataset_run_id",
        "gx1_data_root", "out_bundle_dir", "files", "seed_launch",
        "seed_authority", "smoke_full_val", "trainer_cli", "recipe_env",
        "val_limits", "initialization", "test_data_used", "recipe_sha256",
        "next_run_policy",
    }
    prefix_mode = "chronological_prefix" in recipe
    if (
        not required <= set(recipe)
        or set(recipe) - required - {"candidate_resume_origin", "native_calibration", "exit_backup_steps", "exit_reference_policy", "frozen_readout_evaluation", "chronological_prefix", "chronological_initial_measurement", "chronological_learning_measurement", "entry_gradient_diagnostic", "chronological_train_only_measurement", "chronological_entry_baseline"}
        or recipe["schema_version"] != NATIVE_FULL_TRAIN_RECIPE_SCHEMA
        or recipe["profile"] != "candidate" or recipe["test_data_used"] is not False
        or recipe["initialization"] != ("fresh_existing_model_constructor_no_checkpoint_weights" if prefix_mode else
                                        "frozen_online_readout_evaluation_only" if "frozen_readout_evaluation" in recipe else _INITIALIZATION)
        or recipe["recipe_sha256"] != val.canonical_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})
        or set(recipe["files"]) != _DATA_FILES
        or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(recipe["run_id"])) is None
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_RECIPE_INVALID")
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_calibration_run
    require_native_calibration_run(recipe)
    repo = Path(recipe["source_repo"])
    if repo != Path(__file__).resolve().parents[2]:
        raise RuntimeError("NATIVE_FULL_TRAIN_EXECUTION_SOURCE_INVALID")
    val._assert_clean_source(recipe)
    bindings = _native_recipe_source_bindings(repo)
    if (
        recipe["source_bindings"] != bindings
        or recipe["source_bindings_sha256"] != launch_owner.canonical_json_sha256(bindings)
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_SOURCE_CLOSURE_MISMATCH")
    env = trainer.require_model_native_recipe_env(recipe["recipe_env"])
    if any(os.environ.get(key, expected) != expected for key, expected in env.items()):
        raise RuntimeError("NATIVE_FULL_TRAIN_RECIPE_ENVIRONMENT_MISMATCH")
    files = {key: _bound_artifact(binding) for key, binding in recipe["files"].items()}
    if prefix_mode:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_prefix_recipe
        prefix = require_chronological_prefix_recipe(recipe)
        seed_launch = {"seed":prefix["design"]["initialization"]["seed"],
                       "learning_rate":0.0001, "weight_decay":0.0001}
        _require_prefix_component_bindings(recipe["chronological_prefix"], files=files,
            batch_size=16, seed=seed_launch["seed"], learning_rate=0.0001, weight_decay=0.0001)
    else:
        seed_launch_path = _bound_artifact(recipe["seed_launch"])
        seed_launch = val.require_launch_manifest(val._read(seed_launch_path))
        authority_path = _bound_artifact(recipe["seed_authority"])
        authority = val._load_final_authority(authority_path, recipe["seed_authority"]["sha256"])
        schedule = authority.get("full_population_schedule")
        if (
            authority.get("epoch_complete") is not True or authority.get("epoch_index") != 0
            or not isinstance(schedule, Mapping)
            or schedule.get("every_entry_pair_exactly_once") is not True
            or schedule.get("entry_pair_count") != authority["entry_pair_count"]
            or schedule.get("global_entry_start") != 0
            or schedule.get("global_entry_stop") != authority["entry_pair_count"]
            or authority["global_optimizer_steps"] != (authority["entry_pair_count"] + 15) // 16
        ):
            raise RuntimeError("NATIVE_FULL_TRAIN_COMPLETE_SEED_EPOCH_REQUIRED")
        smoke = val._read(_bound_artifact(recipe["smoke_full_val"]))
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_complete_val_observation
        require_complete_val_observation(smoke)
        checkpoint = require_selected_weight_ema_checkpoint_binding_v1(smoke["checkpoint_binding"])
        if (
            smoke.get("schema_version") != RESULT_SCHEMA_VERSION
            or smoke.get("test_data_used") is not False
            or smoke.get("entry_pair_cohort_size") != 5_508
            or smoke.get("compute_truncated_side_trade_count") != 0
            or smoke.get("semantic_result_sha256") != val.canonical_sha256({k: v for k, v in smoke.items() if k != "semantic_result_sha256"})
            or smoke.get("checkpoint_binding_sha256") != checkpoint["binding_sha256"]
            or checkpoint["checkpoint_file_sha256"] != authority["final_checkpoint_state"]["sha256"]
            or checkpoint["checkpoint_pointer_file_sha256"] != authority["final_checkpoint_pointer"]["sha256"]
            or checkpoint["launch_manifest_sha256"] != seed_launch["manifest_sha256"]
            or recipe["dataset_run_id"] != seed_launch["dataset_run_id"]
        ):
            raise RuntimeError("NATIVE_FULL_TRAIN_COMPLETE_SMOKE_BINDING_REQUIRED")
    # A technical smoke may finish with a selected trade censored at June-end.
    # Its unavailable score cannot select a checkpoint or establish trading quality.
    controls = {
        "epochs": 30, "batch_size": 16, "num_workers": 0, "grad_accum_steps": 1,
        "early_stopping_patience": 5, "early_stopping_min_delta": 0.0,
        "minimum_epochs_before_stop": 1, "save_top_k": 1,
        "precision_policy": "deterministic_fp32", "seed": seed_launch["seed"],
        "learning_rate": seed_launch["learning_rate"], "weight_decay": seed_launch["weight_decay"],
        "grad_clip_norm": float(trainer._GRAD_CLIP_NORM),
        "checkpoint_monitor": native_checkpoint_monitor(
            val._load_val_economics_readiness(files["economics_readiness"])["economics_objective_contract"]
        ),
    }
    if (
        recipe["trainer_cli"] != controls
        or any(type(recipe["trainer_cli"][key]) is not type(value) for key, value in controls.items())
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_CONTROLS_MISMATCH")
    limits = recipe["val_limits"]
    if (
        set(limits) not in (
            {"max_model_forwards", "max_state_views", "max_wall_seconds", "progress_interval_forwards"},
            {"max_model_forwards", "max_state_views", "max_wall_seconds", "progress_interval_forwards", "policy_batch_size"},
            {"max_model_forwards", "max_state_views", "max_wall_seconds", "progress_interval_forwards", "policy_batch_size", "cpu_pipeline_workers"},
        )
        or limits.get("policy_batch_size") != 256
        or limits.get("cpu_pipeline_workers") != 8
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or limits["max_wall_seconds"] != 10_800
        or limits["progress_interval_forwards"] != 64
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_VAL_WINDOW_INVALID")
    output = Path(recipe["out_bundle_dir"])
    if not output.is_absolute() or output.resolve() != output:
        raise RuntimeError("NATIVE_FULL_TRAIN_OUTPUT_PATH_INVALID")
    trainer._resolve_train_out_bundle_dir(output, recipe["gx1_data_root"])
    provenance = launch_owner.require_training_recipe_source_provenance_metadata({
        "schema_version": launch_owner.TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
        "recipe_audit_path": str(recipe_path), "recipe_audit_sha256": recipe_file_sha256,
        "source_commit": recipe["source_commit"], "source_bindings": bindings,
        "source_bindings_sha256": recipe["source_bindings_sha256"],
    }, context="NATIVE_FULL_TRAIN")
    return recipe, files, provenance



def _require_prefix_component_bindings(value, *, files, seed, batch_size, learning_rate, weight_decay):
    """Check completed preprocessing identity; never fit or grant launch authority."""
    if not isinstance(value, Mapping) or set(value) != {"design", "normalization_result", "labels_result"}:
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_BINDINGS_INVALID")
    design, normalization, labels = (val._read(_bound_artifact(value[name]))
                                    for name in ("design", "normalization_result", "labels_result"))
    if (design.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
            or design.get("status") != "DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE"
            or design["scope"].get("test_sealed") is not True
            or design["scope"].get("same_architecture") is not True
            or design["initialization"].get("mode") != "fresh_existing_model_constructor_no_checkpoint_weights"
            or type(seed) is not int or seed != design["initialization"]["seed"]
            or batch_size != 16 or learning_rate != 0.0001 or weight_decay != 0.0001
            or design["budget"]["planned_optimizer_steps"] != 256
            or design["budget"]["maximum_trained_entry_rows"] != 4096
            or normalization.get("schema_version") != "gx1_prefix_normalization_preparation_result_v1"
            or normalization.get("decision") != "PREFIX_NORMALIZATION_READY_NOT_NATIVE_BOUND"
            or normalization["frozen_design"] != value["design"]
            or normalization["scope"].get("control_fit_rows") != 0
            or normalization["scope"].get("test_fit_rows") != 0
            or normalization["scope"].get("test_accessed") is not False
            or normalization.get("original_market_successor_counts_exact_equal") is not True
            or normalization.get("parent_child_row_clock_identity_exact") is not True
            or labels.get("schema_version") != "gx1_prefix_policy_dependent_labels_v1"
            or labels["prefix_preparation"] != normalization["prefix_preparation"]):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_IDENTITY_INVALID")
    preparation = val._read(_bound_artifact(normalization["prefix_preparation"]))
    cutoff = int(val.pd.Timestamp(design["calendar"]["train_control_cutoff"]).value)
    if (preparation["frozen_design"] != value["design"]
            or int(val.pd.Timestamp(preparation["support_end_inclusive"]).value) != cutoff
            or normalization["fit_cutoff_time_ns"] != cutoff
            or normalization["fit_entry_rows"] != preparation["bindings"]["TRAIN_ELIGIBLE_PARENT_ROWS"]
            or labels["labels"]["TRAIN"]["row_binding"] != normalization["fit_entry_rows"]
            or labels["labels"]["CONTROL256"]["row_binding"] != design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"]
            or _bound_artifact(normalization["artifacts"]["COMPOSITE_NORMALIZATION.json"]) != files["child_composite_normalization"]
            or _bound_artifact(labels["parent_entry_parquet"]) != files["entry_train_parquet"]
            or _bound_artifact(labels["parent_entry_manifest"]) != files["entry_train_manifest"]
            or files["entry_val_parquet"] != files["entry_train_parquet"]
            or files["entry_val_manifest"] != files["entry_train_manifest"]):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_SOURCE_INVALID")
    rows = {name: np.load(_bound_artifact(preparation["bindings"][name]), allow_pickle=False)
            for name in ("TRAIN_ELIGIBLE_PARENT_ROWS", "TRAIN_NATIVE_EPOCH0_ORDER", "TRAIN_NATIVE4096_PARENT_ROWS", "TRAIN256_PROBE_PARENT_ROWS")}
    eligible, order, planned, probe = (rows[name] for name in rows)
    if (any(v.dtype != np.dtype("int64") or v.ndim != 1 for v in rows.values())
            or len(eligible) < 4096 or np.any(eligible < 0)
            or not np.array_equal(eligible, np.unique(eligible))
            or not np.array_equal(np.sort(order), eligible)
            or planned.shape != (4096,) or not np.array_equal(planned, order[:4096])
            or probe.shape != (256,) or len(np.unique(probe)) != 256 or not np.isin(probe, planned).all()):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_POPULATION_INVALID")
    composite = val.require_composite_normalization_binding(val._read(files["child_composite_normalization"]))
    if (composite["composite_normalization_sha256"] != normalization["composite_normalization_sha256"]
            or composite["base_feature_normalization"]["contract_sha256"] != normalization["base_contract_sha256"]
            or composite["lifetime_summary_normalization"]["normalization_sha256"] != normalization["summary_normalization_sha256"]):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_NORMALIZATION_INVALID")
    return {"bindings": dict(value), "design": design, "normalization": normalization,
            "labels": labels, "preparation": preparation, "cutoff_time_ns": cutoff,
            "eligible_parent_rows": eligible, "epoch0_parent_order": order}


def _fresh_prefix_model(*, metadata, normalization, device, seed):
    """Call the existing complete constructor, never any checkpoint loader."""
    trainer._set_deterministic(seed, device, "deterministic_fp32")
    model = val._model(metadata, normalization, device)
    if any(torch.count_nonzero(parameter.detach()).item() for parameter in model.task_log_variances.parameters()):
        raise RuntimeError("NATIVE_PREFIX_FRESH_TASK_WEIGHTS_INVALID")
    return model, {
        "initialization": "fresh_existing_model_constructor_no_checkpoint_weights",
        "seed": seed, "model_state_sha256": trainer._model_state_sha256(model),
        "input_normalization_sha256": normalization["contract_sha256"],
        "checkpoint_loaded": False,
    }


def _build_bound_full_train_components(
    *, files: Mapping[str, Path], dataset_run_id: str,
    seed_launch_path: Path | None, seed_authority_path: Path | None,
    seed_authority_file_sha256: str | None, device: torch.device,
    batch_size: int, epochs: int, seed: int,
    learning_rate: float, weight_decay: float,
    val_limits: Mapping[str, int],
    exit_backup_steps: int = 1,
    exit_reference_policy: Mapping[str, Any] | None = None,
    chronological_prefix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind legacy full TRAIN/June or explicit fresh-prefix components.

    Main training warm-starts from the completed smoke EMA. Its new candidate
    starts with a fresh optimizer and full-TRAIN EMA horizon; on subsequent
    invocations the coordinator restores its exact optimizer/EMA/target/RNG.
    Prefix mode reuses completed preprocessing and physical TRAIN for CONTROL256.
    No source checkpoint is modified and no sampler benchmark is rerun here.
    """

    if epochs != 30 or batch_size != 16 or type(exit_backup_steps) is not int or exit_backup_steps not in (1, 5):
        raise RuntimeError("NATIVE_FULL_TRAIN_DECLARED_GEOMETRY_INVALID")
    if device.type == "cuda":
        trainer._require_cuda_trainer_guard_execution(execution_tier="canonical")
    elif device.type != "cpu":
        raise RuntimeError("NATIVE_FULL_TRAIN_DEVICE_INVALID")
    trainer._set_deterministic(seed, device, "deterministic_fp32")
    prefix = None
    if chronological_prefix is not None:
        if any(value is not None for value in (seed_launch_path, seed_authority_path, seed_authority_file_sha256)):
            raise RuntimeError("NATIVE_PREFIX_OLD_INITIALIZATION_FORBIDDEN")
        prefix = _require_prefix_component_bindings(
            chronological_prefix, files=files, seed=seed, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay)
        if exit_backup_steps != 1 or exit_reference_policy != prefix["design"]["targets"]["reference_policy"]:
            raise RuntimeError("NATIVE_PREFIX_REFERENCE_POLICY_MISMATCH")
    else:
        source_launch = val.require_launch_manifest(val._read(seed_launch_path))
        authority = val._load_final_authority(
            seed_authority_path, seed_authority_file_sha256,
        )
        if (
            authority["launch_manifest"]["path"] != str(seed_launch_path)
            or authority["launch_manifest"]["sha256"] != val.file_sha256(seed_launch_path)
            or authority["launch_manifest_sha256"] != source_launch["manifest_sha256"]
            or authority["selected_batch_size"] != batch_size
            or source_launch["files"]["source_bundle_metadata"]["sha256"]
            != val.file_sha256(files["source_bundle_metadata"])
        ):
            raise RuntimeError("NATIVE_FULL_TRAIN_SEED_AUTHORITY_MISMATCH")
    train_window = val._read(files["entry_train_manifest"])["splits"]["train"]
    train_start, train_end = (val.pd.Timestamp(train_window[key]) for key in ("start", "end"))
    if (
        train_start != val.pd.Timestamp("2021-06-01T00:00:00Z")
        or train_end.tzinfo is None
        or not val.pd.Timestamp("2026-05-31T00:00:00Z") <= train_end <= val.pd.Timestamp("2026-06-01T00:00:00Z")
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_FIVE_YEAR_WINDOW_REQUIRED")

    meta = val._read(files["source_bundle_metadata"])
    val._bind_multi_tf_cache_from_source_bundle_metadata(meta)
    per_tf = {
        name.upper(): int(meta["multi_tf"][f"{name}_seq_len"])
        for name in ("m5", "m15", "h1", "h4", "d1")
    }
    file_bindings = {
        key: {"path": str(path), "sha256": val.file_sha256(path)}
        for key, path in files.items()
        if key in {"entry_train_parquet", "entry_train_manifest", "entry_val_parquet", "entry_val_manifest"}
    }
    val_audit = (files["sequence_source_audit"] if prefix is not None
                 else val._val_sequence_source_audit(meta, file_bindings))
    physical_splits = ("train",) if prefix is not None else ("train", "val")
    datasets = {
        split: val.EntryV10CtxDataset(
            files[f"entry_{split}_parquet"], seq_len=int(meta["seq_len"]),
            m5_prebuilt_path=files["m5_prebuilt"], per_tf_seq_lens=per_tf,
            multi_tf_closed_bar=True,
            sequence_source_audit_json=(files["sequence_source_audit"] if split == "train" else val_audit),
        ) for split in physical_splits
    }
    if prefix is not None:
        # Role-specific label replacement is atomic and shares unchanged input columns.
        # The copy precedes any lifecycle or label binding; workers are not running.
        datasets["val"] = copy.copy(datasets["train"])
        for split, role in (("train", "TRAIN"), ("val", "CONTROL256")):
            datasets[split].bind_policy_dependent_auxiliary_targets(
                result_path=Path(chronological_prefix["labels_result"]["path"]),
                expected_result_sha256=chronological_prefix["labels_result"]["sha256"],
                expected_design_sha256=chronological_prefix["design"]["sha256"], role=role)
        # Measurement shares the TRAIN inputs and bound labels, while lifecycle
        # materialization remains attached only to the optimizer's dataset.
        train_probe_ds = copy.copy(datasets["train"])
    corpus = val.UnifiedExitLifecycleCorpus(
        root_manifest_path=files["feature_lifecycle_root"],
        entry_parquets={split: files[f"entry_{split}_parquet"] for split in physical_splits},
        entry_manifest_bindings={split: file_bindings[f"entry_{split}_manifest"] for split in physical_splits},
        dataset_run_id=dataset_run_id, splits=physical_splits,
    )
    root_path = files["random_access_root"]
    root = val.require_random_access_index_root(val._read(root_path))
    from gx1.contracts.unified_exit_random_access_index_v1 import (
        LATEST_YEAR_ROOT_SCHEMA_VERSION, require_latest_year_index_root,
    )
    latest_year = root["schema_version"] == LATEST_YEAR_ROOT_SCHEMA_VERSION
    if latest_year and prefix is None:
        require_latest_year_index_root(root, expected_parent_bindings={
            split: {"parquet": file_bindings[f"entry_{split}_parquet"],
                    "manifest": file_bindings[f"entry_{split}_manifest"]}
            for split in ("train", "val")
        })
    train_binding = root["splits"]["train"]
    train_index = val.pd.read_parquet(
        Path(train_binding["index_parquet_path"]),
        columns=["entry_row_index", "parent_entry_row_index"],
    )
    children = train_index["entry_row_index"].astype("int64").tolist()
    parents = train_index["parent_entry_row_index"].astype("int64").tolist()
    if (
        children != list(range(len(train_index)))
        or sorted(parents) != list(range(len(datasets["train"])))
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_ENTIRE_PARENT_POPULATION_REQUIRED")
    factory = build_random_access_train_adapter_factory_v1(
        root_manifest_path=root_path,
        candidate_set_path=files["sampler_candidate_set"],
        composite_normalization_path=files["child_composite_normalization"],
        economics_readiness_path=files["economics_readiness"],
        train_cost_authority_path=files["train_cost_authority"],
        train_dataset=datasets["train"],
        train_feature_source_owner=corpus.splits["train"],
        backup_steps=exit_backup_steps,
        reference_policy=exit_reference_policy,
        reference_cutoff_time_ns=(prefix["cutoff_time_ns"] if prefix is not None else None),
    )
    # Keep the measured transition-sampler geometry, then select all Entry
    # pairs through the same full-population owner used by the completed year.
    adapter = factory(65_536)
    schedule = adapter.set_full_population_epoch_index(0)
    datasets["train"].bind_unified_exit_lifecycle_v2(adapter)
    datasets["train"].bind_random_access_entry_coordinate_mapping_v1(
        parent_entry_row_indices=parents, child_entry_row_indices=children,
    )

    source_split = "train" if prefix is not None else "val"
    val_binding = root["splits"][source_split]
    index_path = Path(val_binding["index_parquet_path"])
    manifest_path = Path(val_binding["manifest_path"])
    frame = val.pd.read_parquet(index_path)
    manifest = val.require_random_access_index_manifest(
        val._read(manifest_path), expected_split=source_split, index_frame=frame,
        index_path=index_path, verify_sources=False,
    )
    if (
        frame["entry_row_index"].astype("int64").tolist() != list(range(len(datasets["train"]) if prefix is not None else 5_508))
        or val_binding["index_parquet_sha256"] != val.file_sha256(index_path)
        or val_binding["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_JUNE_VAL_BINDING_INVALID")
    evaluation_cohort = None
    if prefix is not None:
        from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_control_cohort
        evaluation_cohort = build_chronological_control_cohort(
            chronological_prefix["design"], source_index_binding={"path": str(index_path), "sha256": val.file_sha256(index_path)})
    control_frame = frame if evaluation_cohort is None else frame.iloc[evaluation_cohort["entry_row_indices"]].copy()
    all_val_states = int(control_frame["lifecycle_state_count"].sum())
    if (
        val_limits["max_state_views"] < all_val_states
        or val_limits["max_model_forwards"] < all_val_states
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_VAL_COMPUTE_CANNOT_COVER_FULL_COHORT")
    parent_evidence = val.require_parent_entry_coordinate_equivalence(
        index_manifest=manifest, index_frame=frame, expected_split=source_split,
        parent_parquet=file_bindings["entry_val_parquet"],
        parent_manifest=file_bindings["entry_val_manifest"],
    )
    readiness = val._load_val_economics_readiness(files["economics_readiness"])
    provider = val._build_provider(
        frame=frame, manifest=manifest, readiness=readiness,
        cost_authority_path=files["train_cost_authority"],
    )
    state_paths = {
        key: val._source_path(manifest, name)
        for key, name in {
            "entry_parquet_path": "entry_parquet",
            "entry_manifest_path": "entry_manifest",
            "child_m1_path": "m1_child",
            "child_m1_manifest_path": "m1_child_manifest",
            "successor_counts_path": "successor_counts",
            "summary_manifest_path": "summary_manifest",
            "first_state_bridge_path": "first_state_bridge",
            "split_sequence_binding_path": "sequence_binding",
            "composite_normalization_path": "composite_normalization",
            "closure_authority_path": "closure_authority",
        }.items()
    }
    val_factory = val.RandomAccessValStateFactoryV1.from_artifacts(
        **state_paths, random_access_index_path=index_path,
        random_access_index_manifest_path=manifest_path,
        random_access_index_root_path=root_path, source_owner=corpus.splits[source_split],
        mtf_materializer=datasets["val"]._get_exit_multi_tf_episode_histories,
        economic_step_provider=provider,
        economic_step_manifest=provider.economic_exit_step_manifest,
        economics_objective_contract=readiness["economics_objective_contract"],
        **({"source_split": "train"} if prefix is not None else {}),
    )
    child = val.require_composite_normalization_binding(
        val._read(files["child_composite_normalization"]),
    )
    if prefix is not None:
        input_norm = child["base_feature_normalization"]["artifact"]["contract"]
        model, seed_binding = _fresh_prefix_model(metadata=meta, normalization=input_norm, device=device, seed=seed)
    else:
        input_norm = val.require_bootstrap_composite_normalization(
            val._read(Path(source_launch["files"]["bootstrap_composite_normalization"]["path"])),
        )["base_feature_normalization"]["artifact"]["base_artifact"]["contract"]
        model = val._model(meta, child["base_feature_normalization"]["artifact"]["contract"], device)
        seed_binding = val.load_selected_weight_ema_checkpoint_readonly_v1(
            pointer_path=Path(authority["final_checkpoint_pointer"]["path"]), model=model,
            expected_checkpoint_pointer_file_sha256=authority["final_checkpoint_pointer"]["sha256"],
            expected_launch_manifest_sha256=source_launch["manifest_sha256"],
            expected_selected_sampler_artifact_sha256=authority["selected_sampler_artifact_sha256"],
            expected_bootstrap_source_receipt_sha256=source_launch["bootstrap_source_receipt_sha256"],
            expected_base_normalization_sha256=authority["base_normalization_sha256"],
            expected_summary_normalization_sha256=authority["summary_normalization_sha256"],
            expected_batch_size=batch_size,
            expected_epoch_schedule_sha256=authority["epoch_schedule_sha256"],
            expected_weight_ema_decay=float(source_launch["weight_ema_decay"]),
        )
        val.bind_preserved_v7_input_normalization(model, input_norm)
    model.requires_grad_(True)
    model.train()
    joint = list(model.task_log_variances.parameters())
    joint_ids = {id(parameter) for parameter in joint}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in joint_ids], "weight_decay": weight_decay},
        {"params": joint, "weight_decay": 0.0},
    ], lr=learning_rate)
    effective_train_rows = (len(prefix["eligible_parent_rows"]) if prefix is not None
                            else root["latest_year_population"]["splits"]["train"]["selected_entry_row_count"]
                            if latest_year else len(datasets["train"]))
    ema_derivation = trainer.resolve_weight_ema_decay(
        trainer.ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED,
        train_rows=(len(prefix["eligible_parent_rows"]) if prefix is not None else len(datasets["train"])),
        batch_size=batch_size, grad_accum_steps=1,
    )
    weight_ema = trainer._WeightEma(model, float(ema_derivation["weight_ema_decay"]))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        if trainer.ENTRY_TRAIN_LR_COSINE_DECAY == 1 else None
    )
    val_context = {
        "frame": control_frame, "state_factory": val_factory,
        "parent_coordinate_evidence": parent_evidence, "val_sequence_audit": val_audit,
        **dict(val_limits),
    }
    if prefix is not None:
        val_context["evaluation_cohort"] = evaluation_cohort
        observed_order = trainer._candidate_training_epoch_order(
            datasets["train"], epoch_index=0,
            parent_population=torch.as_tensor(prefix["eligible_parent_rows"].copy(), dtype=torch.int64))
        if not np.array_equal(observed_order.numpy(), prefix["epoch0_parent_order"]):
            raise RuntimeError("NATIVE_PREFIX_NATIVE_ORDER_MISMATCH")
    trainer._native_candidate_val_context_binding(val_context)
    return {
        "model": model, "optimizer": optimizer, "weight_ema": weight_ema,
        "lr_scheduler": scheduler, "train_ds": datasets["train"],
        "val_ds": datasets["val"], "native_val_context": val_context,
        "input_normalization": input_norm, "metadata": meta, "per_tf_seq_lens": per_tf,
        "seed_binding": seed_binding, "weight_ema_derivation": ema_derivation,
        "full_population_schedule": schedule,
        **({"chronological_prefix": prefix["bindings"], "prefix_parent_population": prefix["eligible_parent_rows"],
            "train_probe_ds": train_probe_ds,
            "prefix_epoch0_parent_order": prefix["epoch0_parent_order"]} if prefix is not None else {}),
        "effective_train_rows": effective_train_rows,
        "unified_exit_lifecycle_evidence": {
            "schema_version": "gx1_native_candidate_input_lineage_v1",
            "root_manifest_sha256": root["root_sha256"],
            "splits": {"train": {"lifecycle_manifest_sha256": train_binding["manifest_sha256"]}},
            "feature_source_root_manifest_sha256": corpus.evidence["root_manifest_sha256"],
        },
    }


def _run_bound_full_train_candidate(
    *, components: Mapping[str, Any], files: Mapping[str, Path],
    device: torch.device, run_id: str, dataset_run_id: str,
    output: Path, gx1_data: str, seed: int, learning_rate: float,
    weight_decay: float, grad_clip_norm: float,
    recipe_source_provenance: Mapping[str, Any], execution_budget: Mapping[str, Any],
    execution_budget_sha256: str, invocation_started_monotonic: float,
    candidate_resume_origin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Enter the existing durable coordinator after the outer launch checks."""

    if device.type != "cuda":
        raise RuntimeError("NATIVE_FULL_TRAIN_CANDIDATE_REQUIRES_CUDA")
    trainer._require_cuda_trainer_guard_execution(execution_tier="canonical")
    trainer._guard_no_rl()
    trainer._GRAD_CLIP_NORM = float(grad_clip_norm)
    trainer._WEIGHT_DECAY = float(weight_decay)
    meta = components["metadata"]
    return trainer._run_resumable_candidate_training(
        **{key: components[key] for key in (
            "model", "optimizer", "weight_ema", "lr_scheduler", "train_ds", "val_ds",
            "native_val_context", "input_normalization", "per_tf_seq_lens",
            "unified_exit_lifecycle_evidence",
        )},
        device=device, effective_train_rows=components["effective_train_rows"],
        batch_size=16, num_workers=0, pin_memory=True,
        persistent_workers=False, prefetch_factor=None,
        epochs=30, early_stopping_patience=5, early_stopping_min_delta=0.0,
        minimum_epochs_before_stop=1, save_top_k=1,
        out_bundle_dir=output, gx1_data_override=gx1_data,
        run_id=run_id, dataset_run_id=dataset_run_id,
        train_parquet=files["entry_train_parquet"], val_parquet=files["entry_val_parquet"],
        m5_prebuilt_path=files["m5_prebuilt"],
        unified_exit_lifecycle_manifest_path=files["random_access_root"],
        seed=seed, grad_accum_steps=1, grad_clip_norm=grad_clip_norm,
        weight_decay=weight_decay, lr=learning_rate, dropout=float(meta["dropout"]),
        seq_len=int(meta["seq_len"]),
        multi_tf_num_layers=int(meta["multi_tf"]["multi_tf_num_layers"]),
        specialist_num_layers=int(meta["specialist_fusion"]["num_layers"]),
        multi_tf_scale=float(meta["multi_tf"]["multi_tf_scale"]),
        specialist_fusion_scale=float(meta["specialist_fusion"]["fusion_scale"]),
        cross_family_fusion_scale=float(meta["specialist_fusion"]["cross_family_fusion_scale"]),
        recipe_source_provenance=recipe_source_provenance,
        precision_policy="deterministic_fp32", execution_budget=execution_budget,
        execution_budget_sha256=execution_budget_sha256,
        invocation_started_monotonic=invocation_started_monotonic,
        checkpoint_monitor=native_checkpoint_monitor(
            val._load_val_economics_readiness(files["economics_readiness"])["economics_objective_contract"]
        ),
        candidate_resume_origin=candidate_resume_origin,
        **({"chronological_prefix": components["chronological_prefix"]} if "chronological_prefix" in components else {}),
    )


def run_guarded_native_candidate_invocation(
    *, recipe_path: Path, recipe_file_sha256: str,
    execution_budget_path: Path, execution_budget_file_sha256: str,
) -> dict[str, Any]:
    """Run one bound candidate window under the existing canonical GPU guard.

    The campaign launcher supplies the immutable per-window budget, including
    its expected resume pointer. This function never starts another process,
    creates a new budget, restarts a campaign or grants live/bundle authority.
    """

    started = time.monotonic()
    trainer._require_cuda_trainer_guard_execution(execution_tier="canonical")
    recipe, files, provenance = _require_native_full_train_recipe(
        recipe_path, recipe_file_sha256,
    )
    budget = launch_owner.require_candidate_execution_budget(
        execution_budget_path, execution_budget_file_sha256,
        recipe_path=recipe_path, recipe_sha256=recipe_file_sha256, recipe=recipe,
    )
    if (
        "resume_probe_val_rows" in budget
        or recipe["val_limits"]["max_wall_seconds"] + 60 >= budget["max_invocation_seconds"]
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_INVOCATION_BUDGET_INVALID")
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_run_scope
    require_native_run_scope(recipe, execution_budget=budget)
    controls = recipe["trainer_cli"]
    prefix_mode = "chronological_prefix" in recipe
    output = Path(recipe["out_bundle_dir"])
    device = trainer._resolve_device("cuda")
    components = _build_bound_full_train_components(
        files=files, dataset_run_id=recipe["dataset_run_id"],
        seed_launch_path=None if prefix_mode else Path(recipe["seed_launch"]["path"]),
        seed_authority_path=None if prefix_mode else Path(recipe["seed_authority"]["path"]),
        seed_authority_file_sha256=None if prefix_mode else recipe["seed_authority"]["sha256"],
        device=device, batch_size=controls["batch_size"], epochs=controls["epochs"],
        seed=controls["seed"], learning_rate=controls["learning_rate"],
        weight_decay=controls["weight_decay"], val_limits=recipe["val_limits"],
        exit_backup_steps=recipe.get("exit_backup_steps", 1),
        exit_reference_policy=recipe.get("exit_reference_policy"),
        **({"chronological_prefix":recipe["chronological_prefix"]} if prefix_mode else {}),
    )
    if not prefix_mode:
        smoke = val._read(Path(recipe["smoke_full_val"]["path"]))
        if components["seed_binding"]["model_state_sha256"] != smoke["checkpoint_binding"]["model_state_sha256"]:
            raise RuntimeError("NATIVE_FULL_TRAIN_ACTUAL_SEED_MODEL_MISMATCH")
    if "entry_gradient_diagnostic" in recipe:
        return _run_entry_gradient_diagnostic(components=components,recipe=recipe,device=device,output=output,
            recipe_file_sha256=recipe_file_sha256,invocation_started=started)
    if "frozen_readout_evaluation" in recipe:
        return _run_frozen_readout_validation(
            components=components, recipe=recipe, device=device, output=output,
            recipe_file_sha256=recipe_file_sha256, invocation_started=started)
    if "chronological_initial_measurement" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_initial_measurement
        initial_scope = require_chronological_initial_measurement(recipe, execution_budget=budget)
        _restore_prefix_initial_measurement_state(components=components, scope=initial_scope, device=device)
    if "chronological_learning_measurement" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_learning_measurement
        learning_scope = require_chronological_learning_measurement(recipe)
        _restore_prefix_initial_measurement_state(components=components, scope=learning_scope, device=device)
    try:
        result = _run_bound_full_train_candidate(
            components=components, files=files, device=device,
            run_id=recipe["run_id"], dataset_run_id=recipe["dataset_run_id"],
            output=output, gx1_data=recipe["gx1_data_root"], seed=controls["seed"],
            learning_rate=controls["learning_rate"], weight_decay=controls["weight_decay"],
            grad_clip_norm=controls["grad_clip_norm"],
            recipe_source_provenance=provenance, execution_budget=budget,
            execution_budget_sha256=execution_budget_file_sha256,
            invocation_started_monotonic=started,
            candidate_resume_origin=recipe.get("candidate_resume_origin"),
        )
    except trainer._CandidateExecutionPaused as paused:
        if "chronological_initial_measurement" in recipe:
            paused.evidence["chronological_initial_measurement"] = _run_prefix_initial_measurement(
                components=components, scope=initial_scope, recipe=recipe, output=output,
                device=device, invocation_started=started, pause_evidence=paused.evidence)
        if ("chronological_learning_measurement" in recipe
                and paused.evidence["reason"] == "optimizer_step_ceiling"
                and paused.evidence["global_optimizer_steps"] == 256):
            paused.evidence["chronological_final_measurement"] = _run_prefix_initial_measurement(
                components=components, scope=learning_scope, recipe=recipe, output=output,
                device=device, invocation_started=started, pause_evidence=paused.evidence, optimizer_steps=256)
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_calibration_run
        calibration = require_native_calibration_run(recipe)
        if (calibration is not None and calibration["report_only_val"]
                and paused.evidence["reason"] == "optimizer_step_ceiling"
                and paused.evidence["global_optimizer_steps"] == 32):
            paused.evidence["native_calibration_validation"] = _run_native_calibration_validation(
                components=components, recipe=recipe, output=output, device=device,
                invocation_started=started, pause_evidence=paused.evidence,
            )
        receipt = trainer._write_candidate_execution_pause_receipt(
            paused.evidence, out_bundle_dir=output,
            gx1_data_override=recipe["gx1_data_root"],
        )
        return {
            "decision": "PAUSED_RESUMABLE", "pause": paused.evidence,
            "pause_receipt": {"path": str(receipt), "sha256": val.file_sha256(receipt)},
            "recipe_file_sha256": recipe_file_sha256,
            "resume_state": _native_resume_state(components=components, output=output),
            "bundle_written": False, "test_data_used": False,
        }
    return {
        "decision": "COMPLETE", "session_directory": result["session_directory"],
        "best_epoch": result["best_epoch"], "last_epoch": result["last_epoch"],
        "early_stopped": result["early_stopped"],
        "best_mean_net_bps_per_entry": result["best_policy_pnl"],
        "selected_checkpoint": result["best_checkpoint"],
        "recipe_file_sha256": recipe_file_sha256,
        "resume_state": _native_resume_state(components=components, output=output),
        "bundle_written": False, "test_data_used": False,
    }



def _entry_gradient_pair(*, model, batch, target, valid, expected_prediction, device):
    """Same frozen eval inputs/targets, existing flag only; autograd never accumulates .grad."""
    if (model.training or target.shape != (16,3) or valid.shape != target.shape
            or valid.dtype != torch.bool or not bool(valid.all())
            or any(p.grad is not None for p in model.parameters())):
        raise RuntimeError("ENTRY_GRADIENT_INPUT_STATE_INVALID")
    names, parameters = zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    routing = [i for i,n in enumerate(names) if n.startswith(("family_tf_context_gate.","family_tf_token_gate."))]
    head = [i for i,n in enumerate(names) if n.startswith(("entry_q_joint_","head_entry_action_q."))]
    if not routing or not head:
        raise RuntimeError("ENTRY_GRADIENT_PARAMETER_SURFACE_MISSING")
    def gradients(loss):
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("ENTRY_GRADIENT_NONFINITE")
        return grads
    def vector(grads, indices):
        return torch.cat([(grads[i].detach().cpu().double().reshape(-1) if grads[i] is not None
                           else torch.zeros(parameters[i].numel(),dtype=torch.float64)) for i in indices])
    def describe(grads):
        return {"connected_parameter_names":[n for n,g in zip(names,grads) if g is not None],
                "nonzero_parameter_names":[n for n,g in zip(names,grads) if g is not None and bool(torch.count_nonzero(g))],
                "routing_l2_norm":float(torch.linalg.vector_norm(vector(grads,routing))),
                "entry_head_l2_norm":float(torch.linalg.vector_norm(vector(grads,head)))}
    def cosine(a,b):
        den=float(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))
        return max(-1.0,min(1.0,float(torch.dot(a,b))/den)) if den>0 else None
    report = {}
    first_head = None
    first_prediction = None
    for variant, detached in (("detached",True),("connected",False)):
        out = trainer._model_forward_fp32(model,batch["seq_x"].to(device),batch["snap_x"].to(device),
            ctx_cat=batch["ctx_cat"].to(device),ctx_cont=batch["ctx_cont"].to(device),
            **trainer._multi_tf_kwargs_from_batch(batch,device),liquidation_relative_values=detached)
        q=out["entry_action_q_bps"]
        # Reuse the existing native FP32 output-parity allowance. Inference
        # and autograd can select different reduction kernels. The two
        # gradient variants still require bit-exact outputs and head gradients.
        difference = float((q.detach()-expected_prediction).abs().max())
        actions_equal = torch.equal(q.detach().argmax(1), expected_prediction.argmax(1))
        print(json.dumps({"event":"ENTRY_GRADIENT_NUMERIC_PARITY", "variant":variant,
            "max_abs_q_difference_bps":difference, "absolute_tolerance_bps":1e-4,
            "relative_tolerance":0.0, "actions_equal":actions_equal}), flush=True)
        if (not actions_equal or not torch.allclose(q.detach(),expected_prediction,atol=1e-4,rtol=0.0)):
            raise RuntimeError("ENTRY_GRADIENT_FORWARD_VALUES_CHANGED")
        if first_prediction is None:first_prediction=q.detach().clone()
        elif not torch.equal(first_prediction,q.detach()):
            raise RuntimeError("ENTRY_GRADIENT_VARIANT_VALUES_CHANGED")
        raw=torch.nn.functional.mse_loss(q[valid],target[valid])
        precision=torch.exp(-model.task_log_variances["entry_action_q"])
        entry_grads=gradients(precision*raw)
        head_vector=vector(entry_grads,head)
        if first_head is None:first_head=head_vector
        elif not torch.equal(first_head,head_vector):
            raise RuntimeError("ENTRY_GRADIENT_HEAD_GRADIENT_CHANGED")
        aux=trainer.dip_forecast_task_losses(out,batch,device)
        aux["side_mae_bps"]=trainer._side_mae_auxiliary_loss(out,batch,device)[0]
        aux["trendline_event"]=trainer._trendline_event_aux_loss(out,batch,device)[0]
        position=trainer._require_active_aux_head_prediction(out,batch,output_name="position_size_logit",
            target_names=("y_position_size_target","y_position_size_mask"))
        mask=batch["y_position_size_mask"].to(device)
        if bool((mask.reshape(-1)==1.0).any()):
            aux["position_size"]=trainer._masked_position_size_mse(position,batch["y_position_size_target"].to(device),mask)
        aux_loss,_=trainer._joint_task_loss(model,aux)
        aux_grads=gradients(aux_loss)
        ev,av=vector(entry_grads,routing),vector(aux_grads,routing)
        sides={}
        for j,side in enumerate(("LONG","SHORT","FLAT")):
            # Each action's contribution uses the same denominator as joint Entry MSE.
            g=gradients(precision*((q[:,j]-target[:,j])**2).sum()/valid.sum())
            sides[side]=describe(g)
        report[variant]={"raw_entry_mse":float(raw.detach()),"entry_precision":float(precision.detach()),
            "cached_prediction_max_abs_difference_bps":difference,"cached_actions_equal":actions_equal,
            "entry":describe(entry_grads),"auxiliary":describe(aux_grads),"entry_by_action":sides,
            "entry_auxiliary_routing_cosine":cosine(ev,av),
            "entry_to_auxiliary_routing_norm_ratio":float(torch.linalg.vector_norm(ev)/torch.linalg.vector_norm(av)) if bool(torch.linalg.vector_norm(av)>0) else None}
        del out,q,raw,precision,entry_grads,aux,aux_loss,aux_grads,g
    if report["detached"]["entry"]["routing_l2_norm"] != 0 or any(p.grad is not None for p in parameters):
        raise RuntimeError("ENTRY_GRADIENT_BASELINE_OR_ACCUMULATION_CHANGED")
    return report


def _entry_signal_losses(q, target, valid):
    """Exact native Entry MSE decomposition, with no changed objective."""
    if (q.shape != (16,3) or target.shape != q.shape or valid.shape != q.shape
            or valid.dtype != torch.bool or not bool(valid.all())
            or not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(target).all())):
        raise RuntimeError("ENTRY_SIGNAL_INPUT_INVALID")
    error=q-target
    parts={"common":(2.0/3.0)*error[:,:2].mean(1).square().mean(),
           "contrast":(1.0/6.0)*(error[:,0]-error[:,1]).square().mean(),
           "flat":(1.0/3.0)*error[:,2].square().mean()}
    torch.testing.assert_close(sum(parts.values()),error.square().mean(),atol=1e-5,rtol=1e-5)
    return parts


def _entry_signal_pair(*,model,batch,target,valid,predictions,states,device,validate_inference=False):
    """Frozen forwards localize lost variation and Entry/aux gradient conflict.

    Routing and Entry-Q parameters are the Entry-private surfaces. These are
    pre-clipping eval gradients, not a reconstructed training optimizer step.
    """
    selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad and
              n.startswith(("family_tf_context_gate.","family_tf_token_gate.","entry_q_joint_","head_entry_action_q."))]
    names,parameters=zip(*selected)
    groups={"routing":[i for i,n in enumerate(names) if n.startswith("family_tf_")],
            "entry_head":[i for i,n in enumerate(names) if not n.startswith("family_tf_")]}
    if model.training or not all(groups.values()) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("ENTRY_SIGNAL_MODEL_INVALID")
    def spread(x):
        x=x.detach().cpu().double();center=x-x.mean(0);variance=center.square().mean(0)
        norm=torch.linalg.vector_norm(x,dim=1);normalized=x/norm.clamp_min(1e-30)[:,None]
        cosine=normalized@normalized.T
        return {"rows":len(x),"width":x.shape[1],"rms_feature_std":float(variance.mean().sqrt()),
                "mean_row_l2":float(norm.mean()),"mean_pairwise_cosine":float((cosine.sum()-cosine.diag().sum())/(len(x)*(len(x)-1)))}
    reports={}
    for variant in ("initial","final"):
        model.load_state_dict(states[variant],strict=True)
        expected=predictions[variant];inference_difference=None
        if validate_inference:
            with torch.inference_mode():
                inference=trainer._model_forward_fp32(model,batch["seq_x"].to(device),batch["snap_x"].to(device),
                    ctx_cat=batch["ctx_cat"].to(device),ctx_cont=batch["ctx_cont"].to(device),
                    **trainer._multi_tf_kwargs_from_batch(batch,device))["entry_action_q_bps"]
                inference_difference=float((inference-expected).abs().max())
                if not torch.allclose(inference,expected,atol=1e-4,rtol=0) or not torch.equal(inference.argmax(1),expected.argmax(1)):
                    raise RuntimeError("ENTRY_SIGNAL_CACHED_INFERENCE_CHANGED")
            del inference
        captured={}
        handle=model.entry_q_joint_norm.register_forward_pre_hook(lambda _m,args:captured.update(source=args[0]))
        try:
            out=trainer._model_forward_fp32(model,batch["seq_x"].to(device),batch["snap_x"].to(device),
                ctx_cat=batch["ctx_cat"].to(device),ctx_cont=batch["ctx_cont"].to(device),
                **trainer._multi_tf_kwargs_from_batch(batch,device))
        finally:handle.remove()
        q=out["entry_action_q_bps"];expected=predictions[variant]
        difference=float((q.detach()-expected).abs().max())
        print(json.dumps({"event":"ENTRY_SIGNAL_NUMERIC_PARITY","variant":variant,
            "max_abs_difference_bps":difference,"actions_equal":bool(torch.equal(q.detach().argmax(1),expected.argmax(1))),
            "cached_validation_mode":"canonical_inference" if validate_inference else "gradient",
            "inference_cached_max_abs_difference_bps":inference_difference,
            "absolute_tolerance_bps":1e-4}),flush=True)
        if ((not validate_inference and not torch.allclose(q.detach(),expected,atol=1e-4,rtol=0))
                or not torch.equal(q.detach().argmax(1),expected.argmax(1))):
            raise RuntimeError("ENTRY_SIGNAL_CACHED_PREDICTION_CHANGED")
        parts=_entry_signal_losses(q,target,valid)
        precision=torch.exp(-model.task_log_variances["entry_action_q"])
        losses={k:precision*v for k,v in parts.items()}
        losses["entry"]=sum(losses.values())
        aux=trainer.dip_forecast_task_losses(out,batch,device)
        aux["side_mae_bps"]=trainer._side_mae_auxiliary_loss(out,batch,device)[0]
        aux["trendline_event"]=trainer._trendline_event_aux_loss(out,batch,device)[0]
        position=trainer._require_active_aux_head_prediction(out,batch,output_name="position_size_logit",
            target_names=("y_position_size_target","y_position_size_mask"))
        mask=batch["y_position_size_mask"].to(device)
        if bool((mask.reshape(-1)==1).any()):
            aux["position_size"]=trainer._masked_position_size_mse(position,batch["y_position_size_target"].to(device),mask)
        losses["auxiliary"]=trainer._joint_task_loss(model,aux)[0]
        source=captured["source"];vectors={};gradients={}
        for task,loss in losses.items():
            grads=torch.autograd.grad(loss,(*parameters,source),retain_graph=True,allow_unused=True)
            vectors[task]={};gradients[task]={}
            for group,indices in groups.items():
                vector=torch.cat([grads[i].detach().cpu().double().reshape(-1) if grads[i] is not None
                                  else torch.zeros(parameters[i].numel(),dtype=torch.float64) for i in indices])
                vectors[task][group]=vector
                gradients[task][group]={"l2_norm":float(torch.linalg.vector_norm(vector)),
                    "connected_tensors":sum(grads[i] is not None for i in indices)}
            gradients[task]["joint_source_l2"]=float(torch.linalg.vector_norm(grads[-1].detach())) if grads[-1] is not None else None
        alignment={}
        for group in groups:
            alignment[group]={}
            for first,second in (("common","contrast"),("entry","auxiliary"),("contrast","auxiliary")):
                x,y=vectors[first][group],vectors[second][group];den=float(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y))
                alignment[group][first+"__"+second]=float(torch.dot(x,y)/den) if den else None
            g=vectors["contrast"][group];joint=vectors["entry"][group]+vectors["auxiliary"][group]
            alignment[group]["contrast_dot_entry_plus_aux_gradient"]=float(torch.dot(g,joint))
        reports[variant]={"model_state_sha256":trainer._model_state_sha256(model),"cached_max_abs_difference_bps":difference,
            "cached_validation_mode":"canonical_inference" if validate_inference else "gradient",
            "inference_cached_max_abs_difference_bps":inference_difference,
            "gradient_cached_within_1e_minus4_bps":bool(torch.allclose(q.detach(),expected,atol=1e-4,rtol=0)),
            "entry_precision":float(precision.detach()),"raw_entry_mse":float(sum(parts.values()).detach()),
            "raw_loss_decomposition":{k:float(v.detach()) for k,v in parts.items()},
            "representations":{**{name:spread(value) for name,value in zip(("local_m5","fused","mtf","global_context"),source.chunk(4,dim=1))},
                               "entry_hidden":spread(out["entry_q_joint_hidden"])},
            "prediction_std_by_action":q.detach().cpu().double().std(0,unbiased=False).tolist(),
            "gradients":gradients,"alignment":alignment}
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError("ENTRY_SIGNAL_GRAD_ACCUMULATED")
        del out,q,parts,losses,aux,source,captured,vectors,grads,position,precision,loss
    return reports


def _joint_probe_adam_delta(*, model, optimizer, saved_optimizer, gradients):
    """FP64 read-only next-step estimate with native clipping and saved AdamW moments."""
    named = dict(model.named_parameters())
    by_id = {id(p): n for n, p in named.items()}
    if set(gradients) != set(named) or len(optimizer.param_groups) != len(saved_optimizer['param_groups']):
        raise RuntimeError('JOINT_PROBE_OPTIMIZER_BINDING_INVALID')
    norms = {}
    for task_weights in (False, True):
        values = [g for n, g in gradients.items() if g is not None and n.startswith('task_log_variances.') == task_weights]
        norm = sum(float(g.square().sum()) for g in values) ** .5
        norms['task_weights' if task_weights else 'prediction'] = norm
    delta = {n: torch.zeros_like(p, device='cpu', dtype=torch.float64) for n, p in named.items()}
    seen = set()
    for live, saved in zip(optimizer.param_groups, saved_optimizer['param_groups']):
        if (len(live['params']) != len(saved['params']) or saved.get('amsgrad') or saved.get('maximize')
                or saved.get('differentiable') or saved.get('capturable')):
            raise RuntimeError('JOINT_PROBE_OPTIMIZER_VARIANT_INVALID')
        beta1, beta2 = saved['betas']
        for p, index in zip(live['params'], saved['params']):
            n = by_id[id(p)]
            if n in seen: raise RuntimeError('JOINT_PROBE_OPTIMIZER_DUPLICATE')
            seen.add(n)
            g = gradients[n]
            if g is None: continue  # AdamW skips parameters whose gradient is absent.
            if g.shape != p.shape or not bool(torch.isfinite(g).all()):
                raise RuntimeError('JOINT_PROBE_GRADIENT_INVALID')
            norm = norms['task_weights' if n.startswith('task_log_variances.') else 'prediction']
            clipped = g * min(1., float(trainer._GRAD_CLIP_NORM) / (norm + 1e-6))
            state = saved_optimizer['state'].get(index, {})
            step = int(state.get('step', 0)) + 1
            m = state.get('exp_avg', torch.zeros_like(p)).detach().cpu().double()
            v = state.get('exp_avg_sq', torch.zeros_like(p)).detach().cpu().double()
            if m.shape != p.shape or v.shape != p.shape or not bool(torch.isfinite(m).all() & torch.isfinite(v).all()) or bool((v < 0).any()):
                raise RuntimeError('JOINT_PROBE_MOMENT_INVALID')
            m = beta1 * m + (1 - beta1) * clipped
            v = beta2 * v + (1 - beta2) * clipped.square()
            delta[n] = (-saved['lr'] * m / (1 - beta1 ** step)
                        / (v.sqrt() / (1 - beta2 ** step) ** .5 + saved['eps'])
                        - saved['lr'] * saved['weight_decay'] * p.detach().cpu().double())
    if seen != set(named): raise RuntimeError('JOINT_PROBE_OPTIMIZER_PARAMETER_MISSING')
    return delta, norms


def _joint_update_probe(*, model, optimizer, saved_state, batch, target, valid, expected, dataset, device):
    """One frozen TRAIN16 eval probe of native Entry/Exit/aux gradients; never steps."""
    parameters = dict(model.named_parameters())
    if model.training or any(p.grad is not None for p in parameters.values()):
        raise RuntimeError('JOINT_PROBE_MODEL_STATE_INVALID')
    def cpu_grad(values):
        return {n: None if g is None else g.detach().cpu().double().clone() for n, g in zip(parameters, values)}
    def grad(loss):
        return cpu_grad(torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True, allow_unused=True))
    def add(*vectors):
        return {n: sum((v[n] for v in vectors if v[n] is not None), torch.zeros_like(p, device='cpu', dtype=torch.float64))
                if any(v[n] is not None for v in vectors) else None for n, p in parameters.items()}
    def forward(which):
        return trainer._model_forward_fp32(which, batch['seq_x'].to(device), batch['snap_x'].to(device),
            ctx_cat=batch['ctx_cat'].to(device), ctx_cont=batch['ctx_cont'].to(device),
            **trainer._multi_tf_kwargs_from_batch(batch, device))
    with torch.inference_mode():
        inference = forward(model)['entry_action_q_bps']
        difference = float((inference - expected).abs().max())
        if not torch.allclose(inference, expected, atol=1e-4, rtol=0) or not torch.equal(inference.argmax(1), expected.argmax(1)):
            raise RuntimeError('JOINT_PROBE_CACHED_INFERENCE_CHANGED')
    teacher = trainer._copy_frozen_prefix_reference_model(model)
    teacher.load_state_dict(saved_state['target_model_state'], strict=True)
    target_digest = trainer._model_state_sha256(teacher)
    if target_digest != val.canonical_model_state_sha256(saved_state['target_model_state']):
        raise RuntimeError('JOINT_PROBE_TEACHER_STATE_CHANGED')
    try:
        out = forward(model)
        q = out['entry_action_q_bps']
        if not torch.equal(q.detach().argmax(1), expected.argmax(1)):
            raise RuntimeError('JOINT_PROBE_GRADIENT_ACTION_CHANGED')
        with torch.no_grad(): teacher_out = forward(teacher)
        representation = out[trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY]
        token_gradient, exit_stats, measured_target, measured_valid = trainer._train_unified_exit_full_population(
            model=model, target_model=teacher, entry_decision_representations=representation,
            target_entry_decision_representations=teacher_out[trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY],
            entry_row_indices=batch['entry_row_index'], dataset=dataset, device=device, grad_accum_steps=1,
            exit_cooperation_gate_epoch=trainer._new_cooperation_gate_epoch_accumulator(trainer._UNIFIED_EXIT_COOPERATION_GATE_WIDTHS),
            exit_feature_tf_gate_epoch=trainer._new_feature_tf_gate_epoch_accumulator(trainer._UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE))
        if (not torch.equal(measured_valid, valid) or not torch.allclose(measured_target, target, atol=1e-4, rtol=0)
                or exit_stats.get('random_access_online_forward_calls') != 1
                or exit_stats.get('random_access_target_forward_calls') != 1
                or exit_stats.get('random_access_backward_calls') != 1):
            raise RuntimeError('JOINT_PROBE_NATIVE_EXIT_OR_TARGET_PARITY_INVALID')
        exit_direct = cpu_grad([p.grad for p in parameters.values()])
        exit_bridge = (representation * token_gradient).sum() + model.task_log_variances['unified_exit_action']
        exit_gradient = add(exit_direct, grad(exit_bridge))
        parts = _entry_signal_losses(q, measured_target, measured_valid)
        precision = torch.exp(-model.task_log_variances['entry_action_q'])
        entry_loss = trainer._joint_task_loss(model, {'entry_action_q': sum(parts.values())})[0]
        aux = trainer.dip_forecast_task_losses(out, batch, device)
        aux['side_mae_bps'] = trainer._side_mae_auxiliary_loss(out, batch, device)[0]
        aux['trendline_event'] = trainer._trendline_event_aux_loss(out, batch, device)[0]
        position = trainer._require_active_aux_head_prediction(out, batch, output_name='position_size_logit',
            target_names=('y_position_size_target', 'y_position_size_mask'))
        mask = batch['y_position_size_mask'].to(device)
        if bool((mask.reshape(-1) == 1).any()):
            aux['position_size'] = trainer._masked_position_size_mse(position, batch['y_position_size_target'].to(device), mask)
        aux_loss = trainer._joint_task_loss(model, aux)[0]
        vectors = {k: grad(precision * value) for k, value in parts.items()}
        vectors.update(entry=grad(entry_loss), auxiliary=grad(aux_loss), exit=exit_gradient)
        vectors['joint'] = add(vectors['entry'], vectors['auxiliary'], vectors['exit'])
        # Independent native-style accumulation checks all shared/private/task gradients.
        (entry_loss + aux_loss + exit_bridge).backward()
        actual = cpu_grad([p.grad for p in parameters.values()])
        max_error = 0.
        for n in parameters:
            a, b = actual[n], vectors['joint'][n]
            if (a is None) != (b is None): raise RuntimeError('JOINT_PROBE_GRADIENT_COVERAGE_MISMATCH')
            if a is not None:
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                max_error = max(max_error, float((a-b).abs().max()))
        groups = {
            'all_prediction': [n for n in parameters if not n.startswith('task_log_variances.')],
            'fuse': [n for n in parameters if n.startswith('fuse.')],
            'entry_head': [n for n in parameters if n.startswith(('entry_q_joint_', 'head_entry_action_q.'))],
            'exit': [n for n in parameters if n.startswith(('exit_', 'head_exit_action.'))]}
        if not all(groups.values()): raise RuntimeError('JOINT_PROBE_PARAMETER_SURFACE_MISSING')
        def dot(a, b, names):
            return sum(float((a[n]*b[n]).sum()) for n in names if a[n] is not None and b[n] is not None)
        directions = {}
        for label, gradients in [('joint', actual), ('without_current_auxiliary', add(vectors['entry'], vectors['exit']))]:
            delta, norms = _joint_probe_adam_delta(model=model, optimizer=optimizer,
                saved_optimizer=saved_state['optimizer_state'], gradients=gradients)
            directions[label] = {'preclip_norms': norms, 'by_parameter_group': {
                group: {'delta_l2': dot(delta,delta,names)**.5,
                        'first_order_weighted_loss_change': {task: dot(g,delta,names) for task,g in vectors.items() if task != 'joint'}}
                for group,names in groups.items()}}
        gradient_report = {group: {'parameter_tensors':len(names),
            'norms':{task:dot(g,g,names)**.5 for task,g in vectors.items()},
            'contrast_dot_auxiliary':dot(vectors['contrast'],vectors['auxiliary'],names),
            'exit_dot_auxiliary':dot(vectors['exit'],vectors['auxiliary'],names)} for group,names in groups.items()}
        return {'inference_cached_max_abs_difference_bps':difference,
            'gradient_cached_max_abs_difference_bps':float((q.detach()-expected).abs().max()),
            'native_target_max_abs_difference_bps':float((measured_target-target).abs().max()),
            'native_accumulation_max_abs_gradient_difference':max_error,
            'native_accumulation_all_parameter_gradients_matched':True,
            'target_model_state_sha256':target_digest,
            'raw_entry_mse':float(sum(parts.values()).detach()),
            'raw_entry_loss_decomposition':{k:float(v.detach()) for k,v in parts.items()},
            'raw_exit_mse':exit_stats['raw_loss'],'exit_transition_count':exit_stats['random_access_transition_count'],
            'gradients':gradient_report,'adam_directions':directions,
            'optimizer_steps':0,'model_forwards':5,
            'limitations':'One eval-mode TRAIN16, not a replay of historical training/dropout. FP64 first-order AdamW estimates with frozen saved moments, not executed finite-step gains. Removing current auxiliary gradients retains historical auxiliary momentum. No predictability, learning or economic claim.'}
    finally:
        for p in parameters.values(): p.grad = None


def _entry_representation_pair(*,model,batch,target,valid,predictions,states,device):
    """Two inference-only forwards on the exact saved TRAIN16; no fit or gradients."""
    if model.training or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("ENTRY_REPRESENTATION_MODEL_INVALID")
    def spread(value):
        x=value.detach().cpu().double()
        if x.ndim != 2 or len(x) != 16 or not bool(torch.isfinite(x).all()):
            raise RuntimeError("ENTRY_REPRESENTATION_SURFACE_INVALID")
        center=x-x.mean(0);norm=torch.linalg.vector_norm(x,dim=1)
        unit=x/norm.clamp_min(1e-30)[:,None];cosine=unit@unit.T
        return {"rows":len(x),"width":x.shape[1],"rms_feature_std":float(center.square().mean().sqrt()),
                "mean_row_l2":float(norm.mean()),
                "mean_pairwise_cosine":float((cosine.sum()-cosine.diag().sum())/(len(x)*(len(x)-1)))}
    reports={}
    for variant in ("initial","final"):
        model.load_state_dict(states[variant],strict=True)
        captured={};handles=[]
        def save(name,value):
            if name in captured:raise RuntimeError("ENTRY_REPRESENTATION_SURFACE_REPEATED")
            captured[name]=value
        def hook(name):
            return lambda _m,args,out:save(name,out)
        for name,module in (("main_fuse",model.fuse),("specialist_correction",model.specialist_out),
                            ("cross_tf_correction",model.cross_tf_out),("cooperation_correction",model.family_tf_cooperation_out),
                            ("joint_normalized",model.entry_q_joint_norm),("joint_linear",model.entry_q_joint_in)):
            handles.append(module.register_forward_hook(hook(name)))
        handles.append(model.fuse.register_forward_pre_hook(lambda _m,args:save("main_fuse_input",args[0])))
        handles.append(model.entry_q_joint_norm.register_forward_pre_hook(lambda _m,args:save("joint_source",args[0])))
        try:
            with torch.inference_mode():
                out=trainer._model_forward_fp32(model,batch["seq_x"].to(device),batch["snap_x"].to(device),
                    ctx_cat=batch["ctx_cat"].to(device),ctx_cont=batch["ctx_cont"].to(device),
                    **trainer._multi_tf_kwargs_from_batch(batch,device))
                q=out["entry_action_q_bps"];expected=predictions[variant]
                if (q.shape != (16,3) or expected.shape != q.shape or not bool(torch.isfinite(q).all())
                        or not torch.allclose(q,expected,atol=1e-4,rtol=0)
                        or not torch.equal(q.argmax(1),expected.argmax(1))):
                    raise RuntimeError("ENTRY_REPRESENTATION_CACHED_INFERENCE_CHANGED")
                parts=_entry_signal_losses(q,target,valid)
                surfaces={**captured,"entry_hidden":out["entry_q_joint_hidden"]}
                for prefix,name in (("raw_","joint_source"),("normalized_","joint_normalized")):
                    surfaces.update({prefix+k:v for k,v in zip(("local_m5","fused","mtf","global_context"),captured[name].chunk(4,dim=1))})
                surfaces.update({k:v for k,v in zip(("seq_pool","snap_hidden","context_before_fuse"),captured["main_fuse_input"].chunk(3,dim=1))})
                reports[variant]={"model_state_sha256":trainer._model_state_sha256(model),
                    "inference_cached_max_abs_difference_bps":float((q-expected).abs().max()),"cached_actions_identical":True,
                    "raw_entry_mse":float(sum(parts.values())),"raw_loss_decomposition":{k:float(v) for k,v in parts.items()},
                    "representations":{k:spread(v) for k,v in surfaces.items()},
                    "prediction_std_by_action":q.cpu().double().std(0,unbiased=False).tolist(),
                    "prediction_contrast_std_bps":float((q[:,0]-q[:,1]).double().std(unbiased=False)),
                    "target_contrast_std_bps":float((target[:,0]-target[:,1]).double().std(unbiased=False))}
        finally:
            for handle in handles:handle.remove()
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError("ENTRY_REPRESENTATION_GRAD_ACCUMULATED")
        del out,q,captured,surfaces,parts
    return reports


def _entry_forward_parity(*,model,batch,expected,device):
    """Measure the two numerical paths without fitting or changing tolerance."""
    if model.training or expected.shape != (16,3) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("ENTRY_FORWARD_PARITY_SCOPE_INVALID")
    values={"cached_reference":expected.detach().cpu().double()}
    for name,context in (("inference",torch.inference_mode),("gradient",torch.enable_grad)):
        with context():
            out=trainer._model_forward_fp32(model,batch["seq_x"].to(device),batch["snap_x"].to(device),
                ctx_cat=batch["ctx_cat"].to(device),ctx_cont=batch["ctx_cont"].to(device),
                **trainer._multi_tf_kwargs_from_batch(batch,device))
            q=out["entry_action_q_bps"]
            if q.shape != expected.shape or not bool(torch.isfinite(q).all()):
                raise RuntimeError("ENTRY_FORWARD_PARITY_NONFINITE_OR_SHAPE")
            values[name]=q.detach().cpu().double().clone()
        del out,q
    comparisons={}
    for left,right in (("inference","cached_reference"),("gradient","cached_reference"),("gradient","inference")):
        delta=values[left]-values[right]
        comparisons[left+"__"+right]={"max_abs_difference_bps":float(delta.abs().max()),
            "rms_difference_bps":float(delta.square().mean().sqrt()),
            "max_abs_difference_by_action_bps":delta.abs().amax(0).tolist(),
            "changed_actions":int((values[left].argmax(1)!=values[right].argmax(1)).sum()),
            "within_existing_1e_minus4_bps":bool(torch.allclose(values[left],values[right],atol=1e-4,rtol=0))}
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError("ENTRY_FORWARD_PARITY_GRAD_ACCUMULATED")
    report={"model_state_sha256":trainer._model_state_sha256(model),"comparisons":comparisons,
        "prediction_mean_by_action":{k:v.mean(0).tolist() for k,v in values.items()},
        "prediction_std_by_action":{k:v.std(0,unbiased=False).tolist() for k,v in values.items()},
        "comparison_tolerance_changed":False,"optimizer_steps":0,"model_forwards":2}
    print(json.dumps({"event":"ENTRY_FORWARD_PARITY_MEASURED",**report}),flush=True)
    return report


def _run_entry_gradient_diagnostic(*, components, recipe, device, output, recipe_file_sha256, invocation_started):
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_entry_gradient_diagnostic
    scope=require_entry_gradient_diagnostic(recipe)
    state_binding=scope["origin_resume_state"]["training_state"]
    pointer_binding=scope["origin_resume_state"]["training_pointer"]
    state=torch.load(_bound_artifact(state_binding),map_location="cpu",weights_only=False)
    model=components["model"]
    model.load_state_dict(state["model_state"],strict=True)
    expected=scope["result"]["model_state_sha256"]
    if (trainer._model_state_sha256(model)!=expected
            or val.canonical_model_state_sha256(state["target_model_state"])!=scope["result"]["target_model_state_sha256"]
            or state["global_optimizer_steps"]!=256):
        raise RuntimeError("ENTRY_GRADIENT_MODEL_STATE_MISMATCH")
    rows=scope["observation"]["diagnostics"]["bounded_entry_observations"][:16]
    parents=[r["parent_entry_row_index"] for r in rows]
    if parents!=scope["observation"]["cohort"]["parent_entry_row_indices"][:16]:
        raise RuntimeError("ENTRY_GRADIENT_TRAIN_ORDER_INVALID")
    signal=scope.get("cached_inputs") is not None
    if signal:
        cached=torch.load(_bound_artifact(scope["cached_inputs"]),map_location="cpu",weights_only=False)
        batch=cached["batch"]
    else:
        loader=val.DataLoader(components["train_probe_ds"],batch_size=16,sampler=val._ExactSampler(parents),
                              num_workers=0,generator=torch.Generator().manual_seed(0))
        batch=next(iter(loader))
    if batch["entry_row_index"].tolist()!=parents:
        raise RuntimeError("ENTRY_GRADIENT_INPUT_ROW_MISMATCH")
    target=torch.tensor([r["target_q_bps"] for r in rows],dtype=torch.float32,device=device)
    valid=torch.tensor([r["target_valid"] for r in rows],dtype=torch.bool,device=device)
    prediction=torch.tensor([r["predicted_q_bps"] for r in rows],dtype=torch.float32,device=device)
    directory=output.parent/"entry_gradient_diagnostic"
    if directory.exists() or directory.is_symlink():raise RuntimeError("ENTRY_GRADIENT_OUTPUT_EXISTS")
    directory.mkdir()
    cache=directory/"TRAIN16_INPUTS_AND_TARGETS.pt"
    # Preserve the expensive input materialization even if the contrast fails.
    torch.save({"batch":batch,"target":target.cpu(),"valid":valid.cpu(),"expected_prediction":prediction.cpu()},cache)
    modes=[(m,m.training) for m in model.modules()]
    rng=trainer._attended_session_rng_state(device=device)
    try:
        model.eval()
        if time.monotonic()-invocation_started>=11940:raise RuntimeError("ENTRY_GRADIENT_WALL_LIMIT")
        if signal:
            initial_rows=scope["initial_observation"]["diagnostics"]["bounded_entry_observations"][:16]
            initial_prediction=torch.tensor([r["predicted_q_bps"] for r in initial_rows],dtype=torch.float32,device=device)
            if scope["plan"].get("diagnostic_kind") == "final_joint_update":
                measured={"final":_joint_update_probe(model=model,optimizer=components["optimizer"],saved_state=state,
                    batch=batch,target=target,valid=valid,expected=prediction,dataset=components["train_ds"],device=device)}
            elif scope["plan"].get("diagnostic_kind") == "initial_final_entry_representations":
                measured=_entry_representation_pair(model=model,batch=batch,target=target,valid=valid,
                    predictions={"initial":initial_prediction,"final":prediction},
                    states={"initial":state["target_model_state"],"final":state["model_state"]},device=device)
            elif scope["plan"].get("diagnostic_kind") == "initial_final_forward_parity":
                measured={}
                for variant,weights,expected_prediction in (("initial",state["target_model_state"],initial_prediction),
                                                            ("final",state["model_state"],prediction)):
                    model.load_state_dict(weights,strict=True)
                    measured[variant]=_entry_forward_parity(model=model,batch=batch,expected=expected_prediction,device=device)
            else:
                measured=_entry_signal_pair(model=model,batch=batch,target=target,valid=valid,
                    predictions={"initial":initial_prediction,"final":prediction},
                    states={"initial":state["target_model_state"],"final":state["model_state"]},device=device,
                    validate_inference=scope["plan"].get("diagnostic_kind")=="initial_final_entry_signal_inference_checked")
        else:
            measured=_entry_gradient_pair(model=model,batch=batch,target=target,valid=valid,
                                         expected_prediction=prediction,device=device)
        if time.monotonic()-invocation_started>=11940:raise RuntimeError("ENTRY_GRADIENT_WALL_LIMIT")
    finally:
        if signal:model.load_state_dict(state["model_state"],strict=True)
        for m,training in modes:m.training=training
        trainer._restore_attended_session_rng_state(rng,device=device)
        if trainer._model_state_sha256(model)!=expected:raise RuntimeError("ENTRY_GRADIENT_MODEL_CHANGED")
        _bound_artifact(pointer_binding);_bound_artifact(state_binding)
    report={"schema_version":"gx1_entry_gradient_diagnostic_result_v1",
        "decision":"TRAIN_GRADIENT_MEASURED_NOT_LEARNING_OR_GENERALIZATION_PROOF",
        "plan":scope["plan_binding"],"source_commit":recipe["source_commit"],"model_state_sha256":expected,
        "training_state":state_binding,"training_pointer":pointer_binding,"input_cache":{"path":str(cache),"sha256":val.file_sha256(cache)},
        "selection":"first16_existing_frozen_TRAIN_probe","model_mode":"eval","measurements":measured,
        "variant_forward_values_and_entry_head_gradients_identical":True,
        "cached_prediction_absolute_tolerance_bps":1e-4,"cached_actions_identical":True,
        "model_and_original_checkpoint_preserved":True,
        "rng_restored":True,"optimizer_steps":0,"model_forwards":2,"control_forwards":0,"test_data_used":False,
        "limitations":"One reused TRAIN batch in eval mode. Finite connected gradients are not learning, improvement or future transfer.",
        "native_elapsed_seconds":time.monotonic()-invocation_started}
    if signal:
        report.update(schema_version="gx1_entry_signal_diagnostic_result_v1",
            decision="ENTRY_REPRESENTATION_AND_GRADIENT_SIGNAL_MEASURED_NO_OPTIMIZER_STEP",
            reused_input_cache=scope["cached_inputs"],variants=["initial","final"],
            exact_native_entry_mse_decomposition=True,
            limitations="One reused TRAIN16 in eval mode. Pre-clipping Entry/auxiliary gradients on Entry-private surfaces only; no Exit forward, full joint optimizer update, training-mode dropout, learning or generalization claim.")
        report.pop("variant_forward_values_and_entry_head_gradients_identical")
        if scope["plan"].get("diagnostic_kind") == "final_joint_update":
            report.update(schema_version="gx1_joint_update_diagnostic_result_v1",
                decision="JOINT_GRADIENT_AND_ADAM_DIRECTION_MEASURED_NO_OPTIMIZER_STEP",
                variants=["final"],model_forwards=5,
                limitations=measured["final"]["limitations"])
        if scope["plan"].get("diagnostic_kind") == "initial_final_entry_representations":
            report.update(schema_version="gx1_entry_representation_diagnostic_result_v1",
                decision="ENTRY_REPRESENTATIONS_MEASURED_NO_OPTIMIZER_STEP",model_forwards=2,backward_passes=0,
                cached_prediction_validation_mode="canonical_inference",input_binding_audit=scope["plan"]["input_binding_audit"],
                limitations="Two inference-only forwards on reused TRAIN16. Measured representation variation is not evidence of learning, predictability, Exit improvement, generalization or profitability.")
        if scope["plan"].get("diagnostic_kind") == "initial_final_entry_signal_inference_checked":
            report.update(schema_version="gx1_entry_signal_diagnostic_result_v2",model_forwards=4,
                cached_prediction_validation_mode="canonical_inference",cross_mode_numerical_parity_claimed=False,
                forward_parity_evidence=scope["plan"]["forward_parity_result"])
        if scope["plan"].get("diagnostic_kind") == "initial_final_forward_parity":
            report.update(schema_version="gx1_entry_forward_parity_result_v1",
                decision="ENTRY_FORWARD_PARITY_MEASURED_NO_OPTIMIZER_STEP",
                variants=["initial_inference","initial_gradient","final_inference","final_gradient"],
                model_forwards=4,exact_native_entry_mse_decomposition=False,
                limitations="Four frozen TRAIN16 forwards isolate inference/gradient paths on initial and final models. A completed report is not a parity PASS, a changed tolerance, learning or generalization evidence.")
    path=directory/"RESULT.json";trainer._candidate_training_session_atomic_write_json(path,report)
    print(json.dumps({"event":"ENTRY_GRADIENT_DIAGNOSTIC_COMPLETE","result":str(path),"optimizer_steps":0}),flush=True)
    return {"decision":"PAUSED_RESUMABLE","resume_state":scope["origin_resume_state"],
            "observation":{"path":str(path),"sha256":val.file_sha256(path)},"recipe_file_sha256":recipe_file_sha256,
            "bundle_written":False,"test_data_used":False}


def _restore_prefix_initial_measurement_state(*, components, scope, device):
    """Load the saved fresh state, never an exposed historical checkpoint."""
    initial = scope["initialization"]
    state = torch.load(_bound_artifact(initial["initial_state"]), map_location="cpu", weights_only=False)
    expected = initial["online_model_state_sha256"]
    if (state.get("schema_version") != "gx1_prefix_fresh_initial_state_v1"
            or state.get("chronological_prefix") != components["chronological_prefix"]
            or state.get("model_forwards") != 0 or state.get("optimizer_steps") != 0
            or state.get("checkpoint_loaded") is not False
            or state["optimizer_state"]["state"] or state["weight_ema_state"]["steps"] != 0
            or any(val.canonical_model_state_sha256(value) != expected for value in (
                state["model_state"], state["target_model_state"], state["weight_ema_state"]["shadow"]))
            or components["seed_binding"]["model_state_sha256"] != expected
            or components["weight_ema_derivation"] != initial["ema_derivation"]):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_STATE_MISMATCH")
    model = components["model"]
    model.load_state_dict(state["model_state"], strict=True)
    components["optimizer"].load_state_dict(state["optimizer_state"])
    components["weight_ema"].restore_checkpoint_state(state["weight_ema_state"], model=model)
    scheduler = components["lr_scheduler"]
    if (scheduler is None) != (state["lr_scheduler_state"] is None):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_SCHEDULER_MISMATCH")
    if scheduler is not None:
        scheduler.load_state_dict(state["lr_scheduler_state"])
    # The saved preparation ran on CPU. CUDA randomness starts from the same
    # declared seed; the CPU/Python/NumPy state comes from the saved preparation.
    trainer._restore_attended_session_rng_state(state["rng_state"], device=torch.device("cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(initial["seed_binding"]["seed"]))
    if trainer._model_state_sha256(model) != expected:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_MODEL_MISMATCH")


def _run_prefix_initial_measurement(*, components, scope, recipe, output, device,
                                  invocation_started, pause_evidence, optimizer_steps=0):
    """Measure initial or fixed final ONLINE state at a durable native boundary."""
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
    if (type(optimizer_steps) is not int or optimizer_steps not in (0, 256)
            or pause_evidence.get("reason") != "optimizer_step_ceiling" or pause_evidence.get("phase") != "train"
            or any(type(pause_evidence.get(k)) is not int or pause_evidence[k] != value
                   for k, value in (("epoch_index", 0), ("next_batch_offset", optimizer_steps),
                                    ("global_optimizer_steps", optimizer_steps)))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_PAUSE_INVALID")
    context = components["native_val_context"]
    deadline = min(invocation_started + 12000 - 60, time.monotonic() + context["max_wall_seconds"])
    if time.monotonic() >= deadline:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_INSUFFICIENT_WINDOW")
    directory = Path(pause_evidence["session_directory"])
    session = trainer._CandidateTrainingSession(out_bundle_dir=output,
        contract=val._read(directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME), read_only=True)
    before = val.file_sha256(session._active_path)
    if before != pause_evidence["active_pointer_sha256"]:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_POINTER_MISMATCH")
    state = session.load_checkpoint()
    target_hash = scope["initialization"]["online_model_state_sha256"]
    model = components["model"]
    expected = trainer._model_state_sha256(model)
    if (state["global_optimizer_steps"] != optimizer_steps
            or val.canonical_model_state_sha256(state["model_state"]) != expected
            or (optimizer_steps == 0 and (state["optimizer_state"]["state"] or expected != target_hash))
            or (optimizer_steps == 256 and (not state["optimizer_state"]["state"]
                                           or state["weight_ema_state"]["steps"] != 256))
            or val.canonical_model_state_sha256(state["target_model_state"]) != target_hash):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_CHECKPOINT_MISMATCH")
    target = trainer._copy_frozen_prefix_reference_model(model).to(device)
    target.load_state_dict(state["target_model_state"], strict=True)
    out = directory / ("initial_measurement" if optimizer_steps == 0 else "final_online_measurement")
    if out.exists() or out.is_symlink():
        raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_EXISTS")
    out.mkdir()
    modes = [(module, module.training) for module in model.modules()]
    rng = trainer._attended_session_rng_state(device=device)
    observations = {}
    try:
        model.eval()
        for role, dataset_key in (("train", "train_probe_ds"), ("control", "val_ds")):
            if role == "control" and recipe.get("chronological_train_only_measurement") is True:
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_WALL_LIMIT")
            cohort = build_chronological_measurement_cohort(recipe["chronological_prefix"]["design"],
                scope["measurement"]["coordinate_result"], role=role)
            _, diagnostics, _ = val._entry_representations(model=model, dataset=components[dataset_key],
                parent_rows=cohort["parent_entry_row_indices"], device=device, batch_size=16,
                candidate_target_model=target, candidate_state_factory=context["state_factory"],
                candidate_child_rows=cohort["entry_row_indices"], exit_boundary_model=target,
                evaluation_cohort=cohort)
            if (len(diagnostics["bounded_entry_observations"]) != 256
                    or len(diagnostics["bounded_exit_anchor_observations"]) != 256
                    or len(diagnostics["bounded_exit_sampled_observations"]) != 1024):
                raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_INCOMPLETE")
            if time.monotonic() >= deadline:
                raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_WALL_LIMIT")
            if optimizer_steps == 256:
                baseline = val._read(_bound_artifact(scope["initial_measurement"]["observations"][role]))
                if "entry_baseline" in scope:
                    if role != "train" or recipe.get("chronological_train_only_measurement") is not True:
                        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_TRAIN_ONLY_REQUIRED")
                    baseline["diagnostics"]["bounded_entry_observations"] = scope["entry_baseline"]["entry_observations"]
                if baseline["cohort"] != cohort or baseline["target_model_state_sha256"] != target_hash:
                    raise RuntimeError("NATIVE_PREFIX_FINAL_COHORT_OR_TEACHER_CHANGED")
                for key, prediction in (("bounded_entry_observations", "predicted_q_bps"),
                        ("bounded_exit_anchor_observations", "prediction_hold_bps"),
                        ("bounded_exit_sampled_observations", "prediction_hold_bps")):
                    old, new = baseline["diagnostics"][key], diagnostics[key]
                    without_predictions = lambda rows: [{k:v for k,v in row.items() if k != prediction} for row in rows]
                    if without_predictions(old) != without_predictions(new):
                        raise RuntimeError("NATIVE_PREFIX_FINAL_FROZEN_TARGET_CHANGED")
                    diagnostics[key] = [{**saved, prediction: fresh[prediction]} for saved, fresh in zip(old, new)]
            report = {"schema_version": ("gx1_prefix_initial_prediction_observation_v1" if optimizer_steps == 0
                                          else "gx1_prefix_final_online_prediction_observation_v1"),
                "role": role, "cohort": cohort, "model_state_sha256": expected,
                "target_model_state_sha256": target_hash, "diagnostics": diagnostics,
                "optimizer_steps": optimizer_steps, "test_data_used": False}
            path = out / (role.upper() + "_OBSERVATION.json")
            trainer._candidate_training_session_atomic_write_json(path, report)
            observations[role] = {"path": str(path), "sha256": val.file_sha256(path)}
            print(json.dumps({"event": "PREFIX_MEASUREMENT_ROLE_COMPLETE", "role": role,
                              "optimizer_steps": optimizer_steps, "entries": 256, "sampled_states": 1024}), flush=True)
    finally:
        for module, training in modes:
            module.training = training
        trainer._restore_attended_session_rng_state(rng, device=device)
        if val.file_sha256(session._active_path) != before or trainer._model_state_sha256(model) != expected:
            raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_CHANGED_STATE")
    result = {"schema_version": ("gx1_native_prefix_initial_measurement_v1" if optimizer_steps == 0
                                else "gx1_native_prefix_final_online_measurement_v1"),
        "decision": ("FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED" if optimizer_steps == 0
                     else "FIXED256_FINAL_ONLINE_MEASURED_PAIRED_LEARNING_REVIEW_REQUIRED"),
        "initialization_result": scope["artifacts"]["initialization_result"],
        "measurement_binding_result": scope["artifacts"]["measurement_binding_result"],
        "training_pointer_sha256": before, "model_state_sha256": expected,
        "target_model_state_sha256": target_hash, "observations": observations,
        "model_functions": dict(trainer._PREFIX_MODEL_FUNCTIONS),
        "measurement_roles": list(observations),
        "optimizer_steps": optimizer_steps, "teacher_refreshed": False, "economic_rollout": False,
        "test_data_used": False, "elapsed_native_seconds": time.monotonic() - invocation_started}
    if optimizer_steps == 256:
        result.update(initial_measurement=scope["artifacts"]["initial_measurement_result"],
                      initial_measurement_audit=scope["artifacts"]["initial_measurement_audit"],
                      selected_model_variant="ONLINE", frozen_targets_exactly_preserved=True)
        if "entry_baseline_result" in scope:
            result["derived_entry_target_baseline"] = scope["entry_baseline_result"]
    path = out / "RESULT.json"
    trainer._candidate_training_session_atomic_write_json(path, result)
    return {"path": str(path), "sha256": val.file_sha256(path)}


def _run_frozen_readout_validation(*, components, recipe, device, output,
                                   recipe_file_sha256, invocation_started):
    """Use the existing native evaluator; preserve the original TRAIN cursor."""
    import copy
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_frozen_readout_evaluation
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import bind_frozen_readout_checkpoint_v1
    scope = require_frozen_readout_evaluation(recipe)
    if device.type != "cuda":
        raise RuntimeError("NATIVE_FROZEN_READOUT_CUDA_REQUIRED")
    directory = output.parent / "frozen_readout_val"
    if directory.exists() or directory.is_symlink():
        raise RuntimeError("NATIVE_FROZEN_READOUT_OUTPUT_EXISTS")
    context = components["native_val_context"]
    if time.monotonic() - invocation_started + context["max_wall_seconds"] + 60 >= 12000:
        raise RuntimeError("NATIVE_FROZEN_READOUT_INSUFFICIENT_WINDOW")
    frame = context["frame"].iloc[scope["cohort"]["entry_row_indices"]].copy()
    planned = scope["plan"]["selection"]["rows"]
    # Check source identities again at the actual full native dataframe boundary.
    for actual, expected in zip(frame.to_dict("records"), planned):
        if any(int(actual[k]) != int(expected[k]) for k in ("entry_row_index", "parent_entry_row_index")):
            raise RuntimeError("NATIVE_FROZEN_READOUT_ENTRY_MAPPING_CHANGED")
    model = components["model"]
    target = copy.deepcopy(model)
    boundary = copy.deepcopy(model)
    binding = bind_frozen_readout_checkpoint_v1(plan_binding=scope["cohort"]["plan"],
        arm=scope["arm"], model=model, target_model=target, boundary_model=boundary)
    before = dict(scope["origin_resume_state"])
    result = val.evaluate_bound_full_val_v1(
        model=model, entry_dataset=components["val_ds"], frame=frame,
        state_factory=context["state_factory"], checkpoint_binding=binding,
        parent_coordinate_evidence=context["parent_coordinate_evidence"],
        val_sequence_audit=context["val_sequence_audit"], device=device, selected_batch_size=16,
        candidate_target_model=target, exit_boundary_model=boundary,
        evaluation_cohort=scope["cohort"], rollout_progress_path=directory/"ROLLOUT_PROGRESS.json",
        result_path=directory/"VAL_RESULT.json", max_forwards_this_invocation=context["max_model_forwards"],
        progress_interval_forwards=context["progress_interval_forwards"],
        compute_guard_max_model_forwards=context["max_model_forwards"],
        compute_guard_max_materialized_state_views=context["max_state_views"],
        compute_guard_max_wall_seconds=context["max_wall_seconds"],
        exit_policy_batch_size=context["policy_batch_size"], cpu_pipeline_workers=context["cpu_pipeline_workers"])
    after = require_frozen_readout_evaluation(recipe)["origin_resume_state"]
    if after != before:
        raise RuntimeError("NATIVE_FROZEN_READOUT_CHANGED_TRAINING_STATE")
    report = {"schema_version":"gx1_native_frozen_readout_evaluation_v1",
        "decision":"OBSERVATION_REQUIRES_REVIEW", "arm":scope["arm"],
        "checkpoint_binding":binding, "evaluation_cohort":scope["cohort"],
        "evaluation_decision":result["decision"], "origin_resume_state":before,
        "optimizer_steps":0, "training_enabled":False, "test_data_used":False,
        "native_invocation_elapsed_seconds":time.monotonic()-invocation_started}
    for key,name in (("progress","ROLLOUT_PROGRESS.json"),("result","VAL_RESULT.json")):
        path=directory/name
        if path.is_file():report[key]={"path":str(path),"sha256":val.file_sha256(path)}
    path=directory/"OBSERVATION.json"
    trainer._candidate_training_session_atomic_write_json(path,report)
    print(json.dumps({"event":"FROZEN_READOUT_VAL_COMPLETED", "arm":scope["arm"],
                      "decision":result["decision"], "observation":str(path)}),flush=True)
    # The evaluation never completes or advances the underlying training session.
    return {"decision":"PAUSED_RESUMABLE", "resume_state":before,
            "observation":{"path":str(path),"sha256":val.file_sha256(path)},
            "recipe_file_sha256":recipe_file_sha256, "bundle_written":False,"test_data_used":False}


def _run_native_calibration_validation(
    *, components: Mapping[str, Any], recipe: Mapping[str, Any], output: Path,
    device: torch.device, invocation_started: float, pause_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one normal native VAL window from a frozen partial-TRAIN snapshot.

    This never advances selection or training. The existing evaluator owns
    batch-256 parity, CPU-pipeline verification and the three-hour VAL budget.
    """
    import copy
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
        bind_candidate_weight_ema_validation_checkpoint_v1,
    )
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_calibration_run
    from gx1.contracts.local_random_access_campaign_v2 import read_bound_json

    calibration = require_native_calibration_run(recipe)
    if (calibration is None or not calibration["report_only_val"]
            or device.type != "cuda" or pause_evidence.get("phase") != "train"
            or any(type(pause_evidence.get(key)) is not int or pause_evidence[key] != wanted
                   for key, wanted in {"epoch_index": 0, "next_batch_offset": 32, "global_optimizer_steps": 32}.items())):
        raise RuntimeError("NATIVE_CALIBRATION_VAL_SCOPE_INVALID")
    elapsed = time.monotonic() - invocation_started
    context = components["native_val_context"]
    profile = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
               "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    if any(context.get(key) != wanted or recipe.get("val_limits", {}).get(key) != wanted
           for key, wanted in profile.items()):
        raise RuntimeError("NATIVE_CALIBRATION_VAL_PROFILE_MISMATCH")
    if elapsed + context["max_wall_seconds"] + 60 >= 12000:
        return {"executed": False, "reason": "insufficient_remaining_native_window",
                "elapsed_before_val_seconds": elapsed, "report_only": True}
    directory = Path(pause_evidence["session_directory"])
    contract_path = directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME
    contract = val._read(contract_path)
    session = trainer._CandidateTrainingSession(out_bundle_dir=output, contract=contract)
    provenance = contract["recipe_source_provenance"]
    bound_recipe = read_bound_json(Path(provenance["recipe_audit_path"]), provenance["recipe_audit_sha256"])
    if (require_native_calibration_run(bound_recipe) != calibration
            or any(bound_recipe.get("val_limits", {}).get(key) != wanted for key, wanted in profile.items())):
        raise RuntimeError("NATIVE_CALIBRATION_VAL_RECIPE_MISMATCH")
    report_directory = directory / "native_val" / "calibration_step_0032"
    report_path = report_directory / "CAPACITY_OBSERVATION.json"
    if report_path.exists() or report_path.is_symlink():
        raise RuntimeError("NATIVE_CALIBRATION_VAL_OBSERVATION_EXISTS")
    pointer_before = val.file_sha256(session._active_path)
    if pointer_before != pause_evidence["active_pointer_sha256"]:
        raise RuntimeError("NATIVE_CALIBRATION_VAL_POINTER_MISMATCH")
    state = session.load_checkpoint()
    model, ema = components["model"], components["weight_ema"]
    target = copy.deepcopy(model).to(device)
    target.load_state_dict(state["target_model_state"], strict=True)
    target.requires_grad_(False)
    target.eval()
    snapshot = session.save_validation_checkpoint(model=model, report_only=True)
    modes = [(module, module.training) for module in model.modules()]
    rng = trainer._attended_session_rng_state(device=device)
    val_started = time.monotonic()
    try:
        model.eval()
        with ema.evaluating(model):
            binding = bind_candidate_weight_ema_validation_checkpoint_v1(snapshot=snapshot, model=model)
            result = val.evaluate_bound_full_val_v1(
                model=model, entry_dataset=components["val_ds"], frame=context["frame"],
                state_factory=context["state_factory"], checkpoint_binding=binding,
                parent_coordinate_evidence=context["parent_coordinate_evidence"],
                val_sequence_audit=Path(context["val_sequence_audit"]), device=device,
                selected_batch_size=16, exit_policy_batch_size=context["policy_batch_size"],
                cpu_pipeline_workers=context["cpu_pipeline_workers"],
                rollout_progress_path=report_directory / "ROLLOUT_PROGRESS.json",
                result_path=report_directory / "VAL_RESULT.json",
                max_forwards_this_invocation=context["max_model_forwards"],
                progress_interval_forwards=context["progress_interval_forwards"],
                compute_guard_max_model_forwards=context["max_model_forwards"],
                compute_guard_max_materialized_state_views=context["max_state_views"],
                compute_guard_max_wall_seconds=context["max_wall_seconds"],
                candidate_target_model=target,
            )
    finally:
        for module, training in modes:
            module.training = training
        trainer._restore_attended_session_rng_state(rng, device=device)
        if val.file_sha256(session._active_path) != pointer_before:
            raise RuntimeError("NATIVE_CALIBRATION_VAL_CHANGED_TRAINING_POINTER")
    progress_path = report_directory / "ROLLOUT_PROGRESS.json"
    observed = val._read(progress_path)
    seconds = time.monotonic() - val_started
    report = {
        "schema_version": "gx1_native_calibration_val_observation_v1",
        "decision": "OBSERVATION_REQUIRES_REVIEW", "report_only": True,
        "native_calibration": calibration, "training_pointer_sha256": pointer_before,
        "snapshot": snapshot, "checkpoint_binding": binding,
        "native_val_profile": {key: context[key] for key in (
            "policy_batch_size", "cpu_pipeline_workers", "max_wall_seconds", "progress_interval_forwards")},
        "elapsed_before_val_seconds": elapsed, "native_val_seconds": seconds,
        "native_invocation_elapsed_seconds": time.monotonic() - invocation_started,
        "optimizer_steps_this_reference": 32 if calibration["arm"] == "reference" else 16,
        "global_optimizer_steps": 32, "val_decision": result["decision"],
        "model_forward_count": observed["model_forward_count"],
        "materialized_state_view_count": observed["materialized_state_view_count"],
        "val_states_per_second_including_entry_setup": observed["materialized_state_view_count"] / seconds,
        "val_progress": {"path": str(progress_path), "sha256": val.file_sha256(progress_path)},
        "checkpoint_selection_advanced": False, "learning_calibrated": False,
        "profitability_proven": False, "test_data_used": False,
    }
    trainer._candidate_training_session_atomic_write_json(report_path, report)
    print(json.dumps({"event": "NATIVE_CALIBRATION_VAL_WINDOW_COMPLETED",
                      "path": str(report_path), "sha256": val.file_sha256(report_path),
                      "native_val_seconds": seconds, "decision": result["decision"]}), flush=True)
    return {"executed": True, "path": str(report_path), "sha256": val.file_sha256(report_path),
            "report_only": True, "checkpoint_selection_advanced": False}


def _native_resume_state(*, components: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Expose existing durable state bindings to the campaign, without rewriting them."""
    directory = output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    pointer_path = directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    pointer = val._read(pointer_path)
    slot = pointer["slot"]
    if type(slot) is not int or slot not in (0, 1):
        raise RuntimeError("NATIVE_FULL_TRAIN_DURABLE_SLOT_INVALID")
    state_path = directory / trainer._CANDIDATE_TRAINING_STATE_FILENAMES[slot]
    state_binding = {"path": str(state_path), "sha256": pointer["state_sha256"]}
    _bound_artifact(state_binding)
    schedule = components["train_ds"]._unified_exit_lifecycle_v2.random_access_full_population_schedule_v1()
    if schedule["epoch_index"] != pointer["epoch_index"]:
        raise RuntimeError("NATIVE_FULL_TRAIN_DURABLE_SCHEDULE_MISMATCH")
    cursor_path = directory / "native_val" / f"epoch_{pointer['epoch_index'] + 1:04d}" / "ROLLOUT_PROGRESS.json"
    cursor_binding = None
    val_forwards = 0
    if pointer["phase"] == "validation" and cursor_path.exists():
        cursor = val._read(cursor_path)
        cursor_binding = {"path": str(cursor_path), "sha256": val.file_sha256(cursor_path)}
        val_forwards = int(cursor["model_forward_count"])
    return {
        "training_pointer": {"path": str(pointer_path), "sha256": val.file_sha256(pointer_path)},
        "training_state": state_binding, "active_val_cursor": cursor_binding,
        "active_val_model_forwards": val_forwards,
        "epoch_schedule_sha256": schedule["schedule_sha256"],
        **{key: pointer[key] for key in (
            "session_contract_sha256", "phase", "epoch_index", "next_batch_offset",
            "global_optimizer_steps", "complete",
        )},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-file-sha256", required=True)
    parser.add_argument("--execution-budget", type=Path, required=True)
    parser.add_argument("--execution-budget-file-sha256", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_guarded_native_candidate_invocation(
        recipe_path=args.recipe, recipe_file_sha256=args.recipe_file_sha256,
        execution_budget_path=args.execution_budget,
        execution_budget_file_sha256=args.execution_budget_file_sha256,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
