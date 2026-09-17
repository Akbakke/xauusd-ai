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
    if (
        not required <= set(recipe)
        or set(recipe) - required - {"candidate_resume_origin", "native_calibration", "exit_backup_steps", "exit_reference_policy", "frozen_readout_evaluation"}
        or recipe["schema_version"] != NATIVE_FULL_TRAIN_RECIPE_SCHEMA
        or recipe["profile"] != "candidate" or recipe["test_data_used"] is not False
        or recipe["initialization"] != ("frozen_online_readout_evaluation_only" if "frozen_readout_evaluation" in recipe else _INITIALIZATION)
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
    output = Path(recipe["out_bundle_dir"])
    device = trainer._resolve_device("cuda")
    components = _build_bound_full_train_components(
        files=files, dataset_run_id=recipe["dataset_run_id"],
        seed_launch_path=Path(recipe["seed_launch"]["path"]),
        seed_authority_path=Path(recipe["seed_authority"]["path"]),
        seed_authority_file_sha256=recipe["seed_authority"]["sha256"],
        device=device, batch_size=controls["batch_size"], epochs=controls["epochs"],
        seed=controls["seed"], learning_rate=controls["learning_rate"],
        weight_decay=controls["weight_decay"], val_limits=recipe["val_limits"],
        exit_backup_steps=recipe.get("exit_backup_steps", 1),
        exit_reference_policy=recipe.get("exit_reference_policy"),
    )
    smoke = val._read(Path(recipe["smoke_full_val"]["path"]))
    if components["seed_binding"]["model_state_sha256"] != smoke["checkpoint_binding"]["model_state_sha256"]:
        raise RuntimeError("NATIVE_FULL_TRAIN_ACTUAL_SEED_MODEL_MISMATCH")
    if "frozen_readout_evaluation" in recipe:
        return _run_frozen_readout_validation(
            components=components, recipe=recipe, device=device, output=output,
            recipe_file_sha256=recipe_file_sha256, invocation_started=started)
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
