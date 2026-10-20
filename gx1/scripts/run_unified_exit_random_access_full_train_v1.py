"""Native full-TRAIN construction for the canonical candidate coordinator.

These internal functions consume already verified recipe/artifact bindings.
They do not provide a standalone launch authority or a second training loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
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
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import (
    native_checkpoint_monitor, DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
)
from gx1.contracts.local_random_access_campaign_v2 import file_verification_scope
from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
    DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY, dynamic_m5_loss_fit_lineage,
    dynamic_m5_loss_scale_identity, load_dynamic_m5_loss_scales,
)


NATIVE_FULL_TRAIN_RECIPE_SCHEMA = "gx1_unified_exit_random_access_full_train_recipe_v1"
_DATA_FILES = frozenset({
    "random_access_root", "sampler_candidate_set", "child_composite_normalization",
    "economics_readiness", "train_cost_authority", "source_bundle_metadata",
    "feature_lifecycle_root", "entry_train_parquet", "entry_train_manifest",
    "entry_val_parquet", "entry_val_manifest", "m5_prebuilt", "sequence_source_audit",
})
_INITIALIZATION = "completed_smoke_ema_then_fresh_main_optimizer_ema_scheduler"
_DYNAMIC_M5_DATA_FILES = frozenset({
    "entry_train_parquet", "entry_train_manifest", "entry_val_parquet", "entry_val_manifest",
    "m5_prebuilt", "source_bundle_metadata", "sequence_source_audit", "val_sequence_source_audit",
    "entry_only_base_normalization", "entry_measurement_coordinates", "train_cost_authority",
})



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

    Legacy joint candidates keep their complete smoke/VAL selection contract.
    The explicit dynamic Entry phase consumes one bound physical TRAIN epoch
    with fixed ONLINE checkpointing; these checks do not grant launch authority.
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
    dynamic_phase = None
    if prefix_mode:
        prefix_value = recipe["chronological_prefix"]
        if not isinstance(prefix_value, Mapping) or "design" not in prefix_value:
            raise RuntimeError("NATIVE_FULL_TRAIN_RECIPE_INVALID")
        design = val._read(_bound_artifact(prefix_value["design"]))
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        if design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
            dynamic_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if (
        not required <= set(recipe)
        or set(recipe) - required - {"candidate_resume_origin", "native_calibration", "exit_backup_steps", "exit_reference_policy", "frozen_readout_evaluation", "chronological_prefix", "chronological_initial_measurement", "chronological_learning_measurement", "entry_gradient_diagnostic", "chronological_train_only_measurement", "chronological_entry_baseline", "chronological_learning_continuation", "entry_observed_market_horizons", "frozen_train_policy_evaluation", "frozen_entry_selector_probe", "entry_observed_market", "entry_learning_study", "entry_m5_path_single_epoch", "entry_execution_profile", "entry_dynamic_m5_technical_qualification", DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY}
        or recipe["schema_version"] != NATIVE_FULL_TRAIN_RECIPE_SCHEMA
        or recipe["profile"] != "candidate" or recipe["test_data_used"] is not False
        or ((DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY in recipe) != (dynamic_phase is not None))
        or ("entry_execution_profile" in recipe and dynamic_phase is None)
        or ("entry_dynamic_m5_technical_qualification" in recipe and dynamic_phase is None)
        or recipe["initialization"] != ("fresh_existing_model_constructor_no_checkpoint_weights" if prefix_mode else
                                        "frozen_online_readout_evaluation_only" if "frozen_readout_evaluation" in recipe else _INITIALIZATION)
        or recipe["recipe_sha256"] != val.canonical_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})
        or (set(recipe["files"]) != _DYNAMIC_M5_DATA_FILES if dynamic_phase is not None else
            (set(recipe["files"]) != _DATA_FILES
             and not (prefix_mode and set(recipe["files"]) in (
                 _DATA_FILES | {"selected_sampler"},
                 _DATA_FILES | {"selected_sampler", "val_sequence_source_audit"}))))
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
    env = (trainer.require_model_native_recipe_env(recipe["recipe_env"], entry_training_phase=dynamic_phase)
           if dynamic_phase is not None else trainer.require_model_native_recipe_env(recipe["recipe_env"]))
    if any(os.environ.get(key, expected) != expected for key, expected in env.items()):
        raise RuntimeError("NATIVE_FULL_TRAIN_RECIPE_ENVIRONMENT_MISMATCH")
    files = {key: _bound_artifact(binding) for key, binding in recipe["files"].items()}
    if prefix_mode:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_prefix_recipe
        require_chronological_prefix_recipe(recipe)  # gate only; the trainer rebinds the prefix itself
        # the prefix gate validated trainer_cli.seed against the named allowed set (design seed or a
        # declared window-phase variant); the launch carries that one origin, never a second literal
        seed_launch = {"seed":recipe["trainer_cli"]["seed"],
                       "learning_rate":0.0001, "weight_decay":0.0001}
        _require_component_sampler(files=files, chronological_prefix=recipe["chronological_prefix"])
        prefix = _require_prefix_component_bindings(recipe["chronological_prefix"], files=files,
            batch_size=16, seed=seed_launch["seed"], learning_rate=0.0001, weight_decay=0.0001)
        if dynamic_phase is not None:
            from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_dynamic_m5_execution_profile
            execution_profile = require_dynamic_m5_execution_profile(recipe)
            _require_dynamic_m5_constructor_binding(files=files, prefix=prefix,
                entry_execution_profile=execution_profile)
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
    if dynamic_phase is not None:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_recipe_metadata
        checked, actual_rows = require_native_recipe_metadata(
            {"path": str(recipe_path), "sha256": recipe_file_sha256},
            source_repo=repo, source_commit=recipe["source_commit"],
        )
        single = prefix.get("entry_m5_path_single_epoch")
        if (checked != recipe or single is None or type(actual_rows) is not int
                or actual_rows != single["train_rows"]
                or actual_rows != len(prefix["eligible_parent_rows"])
                or prefix["physical_preprocessing"].get("entry_training_phase") != dynamic_phase
                or recipe["trainer_cli"]["grad_clip_norm"] != float(trainer._GRAD_CLIP_NORM)):
            raise RuntimeError("NATIVE_FULL_TRAIN_DYNAMIC_IDENTITY_INVALID")
    else:
        controls = {
            "epochs": 1 if "entry_m5_path_single_epoch" in recipe else 30, "batch_size": 16, "num_workers": 0, "grad_accum_steps": 1,
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



def _require_dynamic_m5_constructor_binding(*, files, prefix, entry_execution_profile=None):
    """Check current geometry/routes/cache without constructing a model or reading values."""
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_SEQ_LEN, ordered_model_native_signal_fields,
        MODEL_NATIVE_CTX_CONT_FIELDS, MODEL_NATIVE_CTX_CAT_FIELDS,
    )
    from gx1.features.entry_specialist_feature_groups_v1 import (
        MODEL_NATIVE_TRAINING_SPECIALISTS, classify_entry_specialist_feature,
        require_model_native_context_specialist_routing, require_multi_tf_specialist_routing_v4,
    )
    metadata = val._read(files["source_bundle_metadata"])
    manifest = val._read(files["entry_train_manifest"])
    contract = prefix["normalization"]["contract"]
    feature = manifest["feature_contract"]
    signal = list(manifest["extra"]["model_native_signal_contract"]["fields"])
    surfaces, lineage = contract["surfaces"], contract["lineage"]
    if (signal != list(ordered_model_native_signal_fields(
                manifest["extra"]["model_native_signal_contract"]["selected_fields"]))
            or surfaces["signal"]["field_names"] != signal
            or feature["ctx_cont_names"] != list(MODEL_NATIVE_CTX_CONT_FIELDS)
            or surfaces["ctx_cont"]["field_names"] != feature["ctx_cont_names"]
            or feature["ctx_cat_names"] != list(MODEL_NATIVE_CTX_CAT_FIELDS)
            or contract["ctx_cat"]["field_names"] != feature["ctx_cat_names"]):
        raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_FIELDS_MISMATCH")
    dimensions = {"seq_input_dim": len(signal), "snap_input_dim": len(signal),
        "ctx_cont_dim": len(feature["ctx_cont_names"]), "ctx_cat_dim": len(feature["ctx_cat_names"]),
        "seq_len": MODEL_NATIVE_SEQ_LEN}
    if (metadata.get("schema_version") != "gx1_native_fresh_constructor_metadata_v1"
            or manifest["extra"]["seq_len"] != MODEL_NATIVE_SEQ_LEN
            or any(type(metadata.get(key)) is not int or metadata[key] != value for key, value in dimensions.items())):
        raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_GEOMETRY_MISMATCH")
    mtf = metadata.get("multi_tf", {})
    cache_binding = manifest["extra"]["multi_tf_cache_binding"]
    cache_path = _bound_artifact({"path": cache_binding["manifest_path"], "sha256": cache_binding["manifest_sha256"]})
    cache = val._read(cache_path)
    if (cache_path != Path(cache_binding["cache_dir"]) / "manifest.json"
            or mtf.get("shared_cache_dir") != cache_binding["cache_dir"]
            or mtf.get("shared_cache_manifest_path") != str(cache_path)
            or mtf.get("shared_cache_manifest_sha256") != cache_binding["manifest_sha256"]
            or mtf.get("shared_cache_identity_sha256") != cache_binding["cache_identity_sha256"]
            or cache.get("cache_identity_sha256") != cache_binding["cache_identity_sha256"]
            or lineage["mtf_cache_manifest_path"] != str(cache_path)
            or lineage["mtf_cache_manifest_sha256"] != cache_binding["manifest_sha256"]
            or mtf.get("feature_names") != cache["feature_names"]
            or lineage["per_tf_seq_lens"] != dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)):
        raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_CACHE_MISMATCH")
    require_multi_tf_specialist_routing_v4(mtf["feature_names"])
    for tf, window in PRODUCTION_MTF_PER_TF_WINDOW_BARS:
        width_key, window_key = f"{tf.lower()}_seq_dim", f"{tf.lower()}_seq_len"
        if (type(mtf.get(width_key)) is not int or mtf[width_key] != len(cache["feature_names"])
                or type(mtf.get(window_key)) is not int or mtf[window_key] != window
                or surfaces[f"mtf_{tf.lower()}"]["field_names"] != cache["feature_names"]):
            raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_MTF_GEOMETRY_MISMATCH")
    specialist = metadata.get("specialist_fusion", {})
    indices = specialist.get("input_indices")
    expected = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    for index, name in enumerate(signal):
        owner = classify_entry_specialist_feature(name)
        if owner not in expected:
            raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_SPECIALIST_MISMATCH")
        expected[owner].append(index)
    if (not isinstance(indices, Mapping) or indices != expected
            or any(type(index) is not int for values in indices.values() for index in values)):
        raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_SPECIALIST_MISMATCH")
    require_model_native_context_specialist_routing(
        specialist.get("context_routing"), ordered_signal_names=signal, context="NATIVE_DYNAMIC_M5_CONSTRUCTOR")
    from gx1.features.entry_specialist_feature_groups_v1 import (
        compile_compact_entry_execution_profile, require_compact_entry_execution_profile_reference,
    )
    execution_profile = require_compact_entry_execution_profile_reference(
        entry_execution_profile, context="NATIVE_DYNAMIC_M5_CONSTRUCTOR")
    if execution_profile is not None:
        physical = compile_compact_entry_execution_profile()["physical_field_names"]
        if (signal != list(physical["signal"])
                or feature["ctx_cont_names"] != list(physical["ctx_cont"])
                or feature["ctx_cat_names"] != list(physical["ctx_cat"])
                or mtf["feature_names"] != list(physical["mtf"])):
            raise RuntimeError("NATIVE_DYNAMIC_M5_CONSTRUCTOR_COMPACT_PHYSICAL_FIELDS_INVALID")


def _require_component_sampler(*, files, chronological_prefix):
    """Current physical TRAIN consumes its measured choice; legacy stays frozen."""
    current = False
    if chronological_prefix is not None:
        design = val._read(_bound_artifact(chronological_prefix["design"]))
        current = trainer._prefix_reference_model_functions(design) == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
    if not current:
        if "selected_sampler" in files:
            raise RuntimeError("NATIVE_FULL_TRAIN_LEGACY_SAMPLER_CHANGE_FORBIDDEN")
        return None
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
        if "selected_sampler" in files or require_m5_single_epoch_population(design) is None:
            raise RuntimeError("NATIVE_DYNAMIC_M5_EXIT_SAMPLER_FORBIDDEN")
        return None  # the exact Entry population witness owns this phase's order
    if "selected_sampler" not in files:
        raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_REQUIRED")
    from gx1.contracts.unified_exit_selected_sampler_v1 import (
        DIRECT_SELECTION_MODE, require_selected_sampler_artifact,
    )
    selected = require_selected_sampler_artifact(val._read(files["selected_sampler"]))
    if selected.get("selection_mode") != DIRECT_SELECTION_MODE:
        raise RuntimeError("NATIVE_PREFIX_CURRENT_SAMPLER_REQUIRED")
    for role, key in (("random_access_root", "random_access_root"),
                      ("candidate_set", "sampler_candidate_set")):
        binding = selected[role]
        if (binding["path"] != str(files[key])
                or binding["file_sha256"] != val.file_sha256(files[key])):
            raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_SOURCE_MISMATCH")
    if selected["batch_size"] != design["budget"]["train_batch_size"]:
        raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_BATCH_MISMATCH")
    from gx1.scripts.benchmark_unified_exit_random_access_train_v1 import _reference_workload_from_design
    if selected.get("reference_workload") != _reference_workload_from_design(design):
        raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_WORKLOAD_MISMATCH")
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_benchmarked_design_identity
    try:
        require_benchmarked_design_identity(
            selected.get("benchmark_design"), design_binding=chronological_prefix["design"], design=design)
    except RuntimeError as exc:
        raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_WORKLOAD_MISMATCH") from exc
    metadata = val._read(files["source_bundle_metadata"])
    expected_geometry = {
        "seq_len": int(metadata["seq_len"]),
        "per_tf_seq_lens": {
            tf.upper(): int(metadata["multi_tf"][f"{tf}_seq_len"])
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
        "multi_tf_closed_bar": True,
    }
    if selected.get("input_geometry") != expected_geometry:
        raise RuntimeError("NATIVE_PREFIX_MEASURED_SAMPLER_INPUT_GEOMETRY_MISMATCH")
    return selected


def allowed_prefix_training_seeds(design_seed):
    from gx1.contracts.entry_observed_market_horizons_v1 import allowed_prefix_training_seeds as owner
    return owner(design_seed)


def planned_prefix_optimizer_steps(design):
    from gx1.contracts.entry_observed_market_horizons_v1 import planned_prefix_optimizer_steps as owner
    return owner(design)


def _require_prefix_component_bindings(value, *, files, seed, batch_size, learning_rate, weight_decay):
    """Check completed preprocessing identity; never fit or grant launch authority."""
    if (not isinstance(value, Mapping) or set(value) not in (
            {"design", "normalization_result", "labels_result"},
            {"design", "normalization_result", "labels_result", "native_coordinates"})):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_BINDINGS_INVALID")
    design, normalization, labels = (val._read(_bound_artifact(value[name]))
                                    for name in ("design", "normalization_result", "labels_result"))
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        planned_prefix_training_rows, require_m5_single_epoch_population,
    )
    single_epoch = require_m5_single_epoch_population(design)
    functions = trainer._prefix_reference_model_functions(design)
    if (functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
            and (files["entry_val_parquet"] == files["entry_train_parquet"]
                 or files["entry_val_manifest"] == files["entry_train_manifest"])):
        raise RuntimeError("NATIVE_PREFIX_PHYSICAL_SPLIT_BINDING_REQUIRED")
    if (design.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
            or design.get("status") != "DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE"
            or design["scope"].get("test_sealed") is not True
            or design["scope"].get("same_architecture") is not True
            or design["initialization"].get("mode") != "fresh_existing_model_constructor_no_checkpoint_weights"
            or type(seed) is not int or seed not in allowed_prefix_training_seeds(design["initialization"]["seed"])
            or batch_size != 16 or learning_rate != 0.0001 or weight_decay != 0.0001
            or design["budget"]["planned_optimizer_steps"] != planned_prefix_optimizer_steps(design)
            or design["budget"]["maximum_trained_entry_rows"] != planned_prefix_training_rows(design)):
        raise RuntimeError("NATIVE_PREFIX_COMPONENT_IDENTITY_INVALID")
    if functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
            require_physical_chronological_preprocessing,
            require_physical_native_training_coordinates,
        )
        physical = require_physical_chronological_preprocessing(
            value, design=design, normalization=normalization, labels=labels)
        for split in ("train", "val"):
            for kind in ("parquet", "manifest"):
                if str(files[f"entry_{split}_{kind}"]) != physical["physical_sources"][split][kind]["path"]:
                    raise RuntimeError("NATIVE_PREFIX_COMPONENT_SOURCE_INVALID")
        dynamic = physical.get("entry_training_phase") == trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        normalization_key = "entry_only_base_normalization" if dynamic else "composite_normalization"
        file_key = "entry_only_base_normalization" if dynamic else "child_composite_normalization"
        if _bound_artifact(physical[normalization_key]) != files.get(file_key):
            raise RuntimeError("NATIVE_PREFIX_COMPONENT_NORMALIZATION_INVALID")
        coordinates = require_physical_native_training_coordinates(
            value, design=design, physical=physical)
        if dynamic:
            if ("selected_sampler" in files or single_epoch is None
                    or coordinates.get("population_witness") != single_epoch["binding"]
                    or coordinates.get("native_sampler_contract") != single_epoch["witness"]["native_sampler_contract"]
                    or "val_sequence_source_audit" not in files):
                raise RuntimeError("NATIVE_PREFIX_COMPONENT_COORDINATE_SOURCES_INVALID")
            for split, key in (("train", "sequence_source_audit"), ("val", "val_sequence_source_audit")):
                if _bound_artifact(physical["sequence_source_audits"][split]) != files.get(key):
                    raise RuntimeError("NATIVE_PREFIX_COMPONENT_COORDINATE_SOURCES_INVALID")
        elif (coordinates["selected_sampler_binding"]["path"] != str(files.get("selected_sampler"))
                or coordinates["selected_sampler_binding"]["sha256"] != val.file_sha256(files["selected_sampler"])
                or "val_sequence_source_audit" not in files):
            raise RuntimeError("NATIVE_PREFIX_COMPONENT_COORDINATE_SOURCES_INVALID")
        return {"bindings": dict(value), "design": design, "model_functions": functions,
                **({"entry_m5_path_single_epoch": single_epoch} if single_epoch is not None else {}),
                "normalization": normalization, "labels": labels,
                "physical_preprocessing": physical, "physical_coordinates": coordinates,
                "preparation": coordinates["artifact"],
                "cutoff_time_ns": int(val.pd.Timestamp(design["calendar"]["train_control_cutoff"]).value),
                "eligible_parent_rows": coordinates["eligible_parent_rows"],
                "epoch0_parent_order": coordinates["epoch0_parent_order"]}
    if "native_coordinates" in value:
        raise RuntimeError("NATIVE_PREFIX_LEGACY_COORDINATES_FORBIDDEN")
    if (normalization.get("schema_version") != "gx1_prefix_normalization_preparation_result_v1"
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
    return {"bindings": dict(value), "design": design,
            "model_functions": functions,
            "normalization": normalization,
            "labels": labels, "preparation": preparation, "cutoff_time_ns": cutoff,
            "eligible_parent_rows": eligible, "epoch0_parent_order": order}


def _require_component_model_phase(model, entry_training_phase):
    from gx1.contracts.entry_model_native_training_objective_v1 import require_entry_training_phase
    from gx1.contracts.entry_model_native_joint_task_weighting_v1 import joint_task_names_for_phase
    phase = require_entry_training_phase(entry_training_phase, context="NATIVE_COMPONENT_MODEL")
    tasks = getattr(model, "task_log_variances", None)
    if (getattr(model, "entry_training_phase", None) != phase
            or getattr(getattr(model, "cfg", None), "entry_training_phase", None) != phase
            or not isinstance(tasks, torch.nn.ParameterDict)
            or set(tasks) != set(joint_task_names_for_phase(phase))):
        raise RuntimeError("NATIVE_COMPONENT_MODEL_PHASE_OR_TASKS_MISMATCH")
    queries = trainer._dynamic_m5_model_kwargs(phase).get("dynamic_m5_query_minutes")
    if queries is not None:
        state = getattr(model, "dynamic_m5_query_horizons", None)
        configured = getattr(getattr(model, "cfg", None), "dynamic_m5_query_minutes", None)
        if (not isinstance(state, torch.Tensor) or state.dtype != torch.int64
                or tuple(state.shape) != (len(queries),)
                or not torch.equal(state, torch.tensor(queries, dtype=torch.int64, device=state.device))
                or not isinstance(configured, tuple) or any(type(value) is not int for value in configured)
                or configured != queries):
            raise RuntimeError("NATIVE_COMPONENT_MODEL_DYNAMIC_QUERY_MISMATCH")


def _fresh_prefix_model(*, metadata, normalization, device, seed,
                        entry_training_phase=trainer.JOINT_FITTED_Q_TRAINING_PHASE,
                        entry_execution_profile=None):
    """Call the existing complete constructor, never any checkpoint loader."""
    from gx1.features.entry_specialist_feature_groups_v1 import require_compact_entry_execution_profile_reference
    execution_profile = require_compact_entry_execution_profile_reference(
        entry_execution_profile, context="NATIVE_PREFIX_FRESH_MODEL")
    metadata_profile = require_compact_entry_execution_profile_reference(
        metadata.get("entry_execution_profile"), context="NATIVE_PREFIX_FRESH_METADATA")
    if execution_profile != metadata_profile:
        raise RuntimeError("NATIVE_PREFIX_FRESH_EXECUTION_PROFILE_MISMATCH")
    if execution_profile is not None and entry_training_phase != trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        raise RuntimeError("NATIVE_PREFIX_FRESH_COMPACT_DYNAMIC_PHASE_REQUIRED")
    trainer._set_deterministic(seed, device, "deterministic_fp32")
    model = val._model(metadata, normalization, device,
        **trainer._dynamic_m5_model_kwargs(entry_training_phase),
        **({"entry_execution_profile":execution_profile} if execution_profile is not None else {}))
    if execution_profile is not None and (getattr(model, "entry_execution_profile", None) != execution_profile
            or getattr(getattr(model, "cfg", None), "entry_execution_profile", None) != execution_profile):
        raise RuntimeError("NATIVE_PREFIX_FRESH_MODEL_EXECUTION_PROFILE_MISMATCH")
    _require_component_model_phase(model, entry_training_phase)
    if any(torch.count_nonzero(parameter.detach()).item() for parameter in model.task_log_variances.parameters()):
        raise RuntimeError("NATIVE_PREFIX_FRESH_TASK_WEIGHTS_INVALID")
    return model, {
        "initialization": "fresh_existing_model_constructor_no_checkpoint_weights",
        "seed": seed, "model_state_sha256": trainer._model_state_sha256(model),
        "input_normalization_sha256": normalization["contract_sha256"],
        "checkpoint_loaded": False,
        **({"entry_execution_profile":execution_profile} if execution_profile is not None else {}),
    }


def _bound_measurement_state_factory(*, manifest, index_path, manifest_path, root_path,
                                     frame, dataset, source_owner, readiness, cost_authority,
                                     source_split, evaluation_cohort=None):
    """Reuse the existing state factory with one explicit physical source."""
    provider = val._build_provider(frame=frame, manifest=manifest, readiness=readiness,
                                   cost_authority_path=cost_authority)
    state_paths = {
        key: val._source_path(manifest, name)
        for key, name in {
            "entry_parquet_path": "entry_parquet", "entry_manifest_path": "entry_manifest",
            "child_m1_path": "m1_child", "child_m1_manifest_path": "m1_child_manifest",
            "successor_counts_path": "successor_counts", "summary_manifest_path": "summary_manifest",
            "first_state_bridge_path": "first_state_bridge", "split_sequence_binding_path": "sequence_binding",
            "composite_normalization_path": "composite_normalization", "closure_authority_path": "closure_authority",
        }.items()
    }
    return val.RandomAccessValStateFactoryV1.from_artifacts(
        **state_paths, random_access_index_path=index_path,
        random_access_index_manifest_path=manifest_path, random_access_index_root_path=root_path,
        source_owner=source_owner, mtf_materializer=dataset._get_exit_multi_tf_episode_histories,
        economic_step_provider=provider, economic_step_manifest=provider.economic_exit_step_manifest,
        economics_objective_contract=readiness["economics_objective_contract"],
        source_split=source_split,
        **({"evaluation_cohort": evaluation_cohort} if evaluation_cohort is not None else {}),
    )


def _configure_component_training_phase(model, entry_phase):
    _require_component_model_phase(model, entry_phase)
    from gx1.contracts.entry_observed_market_v1 import is_inactive_phase_parameter
    from gx1.contracts.entry_model_native_training_objective_v1 import active_joint_task_names
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not is_inactive_phase_parameter(name, entry_phase))
    model.train()
    active_tasks = {name for name, parameter in model.task_log_variances.items() if parameter.requires_grad}
    if active_tasks != set(active_joint_task_names(entry_phase)):
        raise RuntimeError("NATIVE_COMPONENT_ACTIVE_TASK_VARIANCES_MISMATCH")


def _component_optimizer_state(*, model, entry_phase, train_rows, batch_size, epochs,
                               learning_rate, weight_decay):
    joint = [parameter for parameter in model.task_log_variances.parameters() if parameter.requires_grad]
    joint_ids = {id(parameter) for parameter in joint}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in joint_ids], "weight_decay": weight_decay},
        {"params": joint, "weight_decay": 0.0},
    ], lr=learning_rate)
    ema_derivation = trainer.resolve_weight_ema_decay(
        trainer.ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED,
        train_rows=train_rows,
        batch_size=batch_size, grad_accum_steps=1,
    )
    weight_ema = trainer._WeightEma(model, float(ema_derivation["weight_ema_decay"]),
        entry_training_phase=entry_phase)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        if trainer.ENTRY_TRAIN_LR_COSINE_DECAY == 1 else None
    )
    return optimizer, weight_ema, scheduler, ema_derivation


def publish_dynamic_m5_fresh_initialization(*, recipe_path: Path, recipe_file_sha256: str,
                                          output_dir: Path) -> dict[str, dict[str, str]]:
    """Publish genuine fresh CPU state, without Dataset construction or learning.

    The sealed preparation recipe carries final input/control identities but no
    initial/learning artifacts. Receipts bind its source closure and inputs,
    rather than its file hash, so final recipe/policy binding is acyclic.
    """
    import tempfile
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes, _publish_file_noreplace
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
        _require_dynamic_initial_artifacts, require_prefix_measurement_source_binding,
    )
    from gx1.contracts.unified_exit_random_access_state_view_v1 import _structured_sha256
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    recipe, files, _ = _require_native_full_train_recipe(recipe_path, recipe_file_sha256)
    phase = trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if (set(files) != _DYNAMIC_M5_DATA_FILES or "chronological_prefix" not in recipe
            or any(key in recipe for key in ("chronological_initial_measurement", "chronological_learning_measurement", "entry_dynamic_m5_technical_qualification"))):
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_RECIPE_REQUIRED")
    policy = val._read(_bound_artifact(recipe["next_run_policy"]))
    if policy.get("training_enabled") is not False or policy.get("full_epoch_training_allowed") is not False:
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_TRAINING_FORBIDDEN")
    controls = recipe["trainer_cli"]
    prefix = _require_prefix_component_bindings(recipe["chronological_prefix"], files=files,
        seed=controls["seed"], batch_size=controls["batch_size"],
        learning_rate=controls["learning_rate"], weight_decay=controls["weight_decay"])
    physical, single = prefix.get("physical_preprocessing", {}), prefix.get("entry_m5_path_single_epoch")
    if (physical.get("entry_training_phase") != phase or single is None
            or single["witness"]["training_phase"] != phase
            or prefix.get("model_functions") != trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
            or controls["epochs"] != 1 or controls["batch_size"] != 16
            or single["train_rows"] != len(prefix["eligible_parent_rows"])):
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_IDENTITY_INVALID")
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    with file_verification_scope():
        loss_scales = load_dynamic_m5_loss_scales(recipe[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY],
            expected_lineage=dynamic_m5_loss_fit_lineage(require_observed_entry_scope(recipe), single))
    loss_scale_identity = dynamic_m5_loss_scale_identity(loss_scales)
    output_dir = Path(output_dir)
    if (not output_dir.is_absolute() or output_dir != output_dir.resolve()
            or output_dir.exists() or output_dir.is_symlink()
            or not output_dir.parent.is_dir() or output_dir.parent != output_dir.parent.resolve()):
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_OUTPUT_INVALID")
    metadata = val._read(files["source_bundle_metadata"])
    normalization = val._read(files["entry_only_base_normalization"])["contract"]
    val._bind_multi_tf_cache_from_source_bundle_metadata(metadata)
    device = torch.device("cpu")
    model, seed_binding = _fresh_prefix_model(metadata=metadata, normalization=normalization,
        device=device, seed=controls["seed"], entry_training_phase=phase,
        **({"entry_execution_profile":metadata["entry_execution_profile"]} if metadata.get("entry_execution_profile") is not None else {}))
    _configure_component_training_phase(model, phase)
    optimizer, weight_ema, scheduler, ema_derivation = _component_optimizer_state(
        model=model, entry_phase=phase, train_rows=single["train_rows"],
        batch_size=controls["batch_size"], epochs=controls["epochs"],
        learning_rate=controls["learning_rate"], weight_decay=controls["weight_decay"])
    model_state = {name: value.detach() for name, value in model.state_dict().items()}
    expected = trainer._model_state_sha256(model)
    ema_state = weight_ema.checkpoint_state()
    if (any(value.device.type != "cpu" or value.requires_grad for value in model_state.values())
            or optimizer.state_dict()["state"] or ema_state["steps"] != 0
            or val.canonical_model_state_sha256(ema_state["shadow"]) != expected
            or expected != seed_binding["model_state_sha256"]):
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_STATE_INVALID")
    rng = trainer._attended_session_rng_state(device=device)
    def rng_identity(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {key: rng_identity(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rng_identity(item) for item in value]
        return value
    rng_sha256 = _structured_sha256(rng_identity(rng))
    closure = require_prefix_measurement_source_binding(recipe)
    common = {"entry_training_phase": phase, "target_model_role": "frozen_constructor_anchor_no_forward",
        DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY: loss_scale_identity,
        "exit_evaluation": False, "native_launch_authorized": False, "exit_model_forward_count": 0, **closure}
    initial = {"schema_version": "gx1_prefix_fresh_initialization_v1",
        "decision": "ACTUAL_NATIVE_FRESH_COMPONENTS_READY_NO_LEARNING_MEASURED", **common,
        "chronological_prefix": recipe["chronological_prefix"], "files": recipe["files"],
        "inherited_checkpoint_weights": False, "optimizer_state_empty": True, "target_frozen": True,
        "model_forwards": 0, "optimizer_steps": 0, "ema_steps": 0, "normalization_refits": 0,
        "online_model_state_sha256": expected, "target_model_state_sha256": expected, "ema_model_state_sha256": expected,
        "model_functions": prefix["model_functions"], "seed_binding": seed_binding, "ema_derivation": ema_derivation,
        "source_bindings": {"gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py":
            val.file_sha256(Path(recipe["source_repo"]) / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py")},
        "entry_only_base_normalization": physical["entry_only_base_normalization"],
        "base_contract_sha256": physical["base_contract_sha256"], "population_witness": single["binding"],
        "train_rows": single["train_rows"], "rng_state_sha256": rng_sha256,
        "optimizer_state_sha256": val.canonical_sha256(optimizer.state_dict()),
        "lr_scheduler_state_sha256": val.canonical_sha256(None if scheduler is None else scheduler.state_dict()),
        "native_training_admitted": False, "test_data_used": False}
    measurement = {"schema_version": "gx1_prefix_native_measurement_binding_result_v1",
        "decision": "FROZEN_TRAIN_CONTROL_COORDINATES_AND_NATIVE_MEASUREMENT_BINDING_READY", **common,
        "actual_model_forwards": 0, "optimizer_steps": 0, "test_data_used": False,
        "coordinate_result": recipe["files"]["entry_measurement_coordinates"]}
    # state_dict tensors are detached CPU values. The immutable archive may share
    # their storage for ONLINE and its identical no-forward constructor anchor.
    state = {"schema_version": "gx1_prefix_fresh_initial_state_v1",
        DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY: loss_scale_identity,
        "chronological_prefix": recipe["chronological_prefix"], "model_functions": prefix["model_functions"],
        "model_forwards": 0, "optimizer_steps": 0, "checkpoint_loaded": False,
        "model_state": model_state, "target_model_state": dict(model_state),
        "optimizer_state": optimizer.state_dict(), "weight_ema_state": ema_state,
        "lr_scheduler_state": None if scheduler is None else scheduler.state_dict(), "rng_state": rng}
    stage = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".staging.", dir=output_dir.parent))
    staged_state = stage / "INITIAL_STATE.pt"
    torch.save(state, staged_state)
    trainer._fsync_regular_file(staged_state)
    state_sha256 = val.file_sha256(staged_state)
    initial["initial_state"] = {"path": str(output_dir / staged_state.name), "sha256": state_sha256}
    del state, ema_state, model_state
    components = {"model": model, "optimizer": optimizer, "weight_ema": weight_ema, "lr_scheduler": scheduler,
        "chronological_prefix": recipe["chronological_prefix"], "model_functions": prefix["model_functions"],
        "seed_binding": seed_binding, "weight_ema_derivation": ema_derivation,
        DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY: loss_scale_identity}
    staged_initial = {**initial, "initial_state": {"path": str(staged_state), "sha256": state_sha256}}
    _restore_prefix_initial_measurement_state(components=components,
        scope={"initialization": staged_initial}, device=device)
    if (_structured_sha256(rng_identity(trainer._attended_session_rng_state(device=device))) != rng_sha256
            or trainer._model_state_sha256(model) != expected):
        raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_ROUNDTRIP_INVALID")
    _require_dynamic_initial_artifacts(recipe, prefix, initial, measurement)
    for name, value in (("INITIALIZATION.json", initial), ("MEASUREMENT_BINDING.json", measurement)):
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _publish_bytes(stage / name, encoded)
        if json.loads((stage / name).read_text()) != value:
            raise RuntimeError("NATIVE_DYNAMIC_M5_FRESH_PREPARATION_JSON_INVALID")
    _fsync_directory(stage)
    _publish_file_noreplace(stage, output_dir)
    _fsync_directory(output_dir.parent)
    return {key: {"path": str(output_dir / name), "sha256": val.file_sha256(output_dir / name)}
        for key, name in (("initialization_result", "INITIALIZATION.json"),
                          ("measurement_binding_result", "MEASUREMENT_BINDING.json"))}


def _build_dynamic_m5_entry_components(*, files, chronological_prefix, observed_entry_scope,
        dataset_run_id, device, batch_size, epochs, seed, learning_rate, weight_decay, val_limits,
        dynamic_m5_loss_scales=None):
    """Bind the existing Entry Dataset/model without constructing any Exit owner."""
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_horizons_v1 import bind_observed_entry_dataset
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
    phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    limits = {"max_model_forwards", "max_state_views", "max_wall_seconds", "progress_interval_forwards"}
    if (set(files) != _DYNAMIC_M5_DATA_FILES or epochs != 1 or batch_size != 16
            or not isinstance(observed_entry_scope, Mapping) or observed_entry_scope.get("training_phase") != phase
            or set(val_limits) != limits):
        raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_SCOPE_INVALID")
    # Metadata digests may be reused here; Dataset/model work starts after exit.
    with file_verification_scope():
        prefix = _require_prefix_component_bindings(chronological_prefix, files=files, seed=seed,
            batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
    physical = prefix.get("physical_preprocessing", {})
    single = prefix.get("entry_m5_path_single_epoch")
    if (physical.get("entry_training_phase") != phase or single is None
            or single["witness"]["training_phase"] != phase):
        raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_PHYSICAL_BINDING_REQUIRED")
    meta = val._read(files["source_bundle_metadata"])
    val._bind_multi_tf_cache_from_source_bundle_metadata(meta)
    per_tf = {tf.upper(): int(meta["multi_tf"][tf + "_seq_len"])
              for tf in ("m5", "m15", "h1", "h4", "d1")}
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    if int(meta["seq_len"]) != trainer.MODEL_NATIVE_SEQ_LEN or per_tf != dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS):
        raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_GEOMETRY_MISMATCH")
    normalization = val._read(files["entry_only_base_normalization"])["contract"]
    if (normalization["contract_sha256"] != physical["base_contract_sha256"]
            or normalization["lineage"]["dataset_run_id"] != dataset_run_id):
        raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_DATASET_IDENTITY_MISMATCH")
    with file_verification_scope():
        loss_scales = load_dynamic_m5_loss_scales(dynamic_m5_loss_scales,
            expected_lineage=dynamic_m5_loss_fit_lineage(observed_entry_scope, single))
    loss_scale_identity = dynamic_m5_loss_scale_identity(loss_scales)
    datasets = {}
    for split, role, audit_key in (("train", "TRAIN", "sequence_source_audit"),
                                  ("val", "CONTROL256", "val_sequence_source_audit")):
        source = physical["physical_sources"][split]
        for kind in ("parquet", "manifest"):
            if {"path": str(files[f"entry_{split}_{kind}"]), "sha256": val.file_sha256(files[f"entry_{split}_{kind}"])} != source[kind]:
                raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_SOURCE_MISMATCH")
        from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
        manifest_run = val._read(files[f"entry_{split}_manifest"]).get("extra", {}).get("entry_run_id")
        if require_entry_run_id(manifest_run) != dataset_run_id:
            raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_DATASET_IDENTITY_MISMATCH")
        dataset = val.EntryV10CtxDataset(files[f"entry_{split}_parquet"], seq_len=int(meta["seq_len"]),
            m5_prebuilt_path=files["m5_prebuilt"], per_tf_seq_lens=per_tf,
            multi_tf_closed_bar=True, sequence_source_audit_json=files[audit_key])
        if len(dataset) != source["physical_rows"]:
            raise RuntimeError("NATIVE_DYNAMIC_M5_COMPONENT_SOURCE_MISMATCH")
        dataset.bind_policy_dependent_auxiliary_targets(
            result_path=Path(chronological_prefix["labels_result"]["path"]),
            expected_result_sha256=chronological_prefix["labels_result"]["sha256"],
            expected_design_sha256=chronological_prefix["design"]["sha256"], role=role)
        bind_observed_entry_dataset(dataset, observed_entry_scope)
        trainer._bind_dynamic_m5_loss_scales(dataset, loss_scales, device=device)
        if getattr(dataset, "_unified_exit_lifecycle_v2", None) is not None:
            raise RuntimeError("NATIVE_DYNAMIC_M5_EXIT_LIFECYCLE_FORBIDDEN")
        datasets[split] = dataset
    train_probe = copy.copy(datasets["train"])
    identity = datasets["train"]._entry_observed_market_binding["identity"]
    queries = trainer._dynamic_m5_model_kwargs(phase)["dynamic_m5_query_minutes"]
    if (identity != datasets["val"]._entry_observed_market_binding["identity"]
            or any(tuple(binding.get("query_minutes", ())) != queries for binding in (identity, observed_entry_scope))):
        raise RuntimeError("NATIVE_COMPONENT_SCOPE_DYNAMIC_QUERY_MISMATCH")
    model, seed_binding = _fresh_prefix_model(metadata=meta, normalization=normalization,
        device=device, seed=seed, entry_training_phase=phase,
        **({"entry_execution_profile":meta["entry_execution_profile"]} if meta.get("entry_execution_profile") is not None else {}))
    _configure_component_training_phase(model, phase)
    # This owner establishes the strict population/order authority before the
    # epoch order is requested and before any optimizer is constructed.
    with file_verification_scope():
        session_binding, parents, expected_order = trainer._prefix_candidate_training_binding(
            chronological_prefix=chronological_prefix, train_ds=datasets["train"], val_ds=datasets["val"],
            train_parquet=files["entry_train_parquet"], val_parquet=files["entry_val_parquet"],
            input_normalization=normalization, model=model, seed=seed, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay)
    order = trainer._candidate_training_epoch_order(datasets["train"], epoch_index=0, parent_population=parents)
    if (not torch.equal(order, expected_order) or not np.array_equal(order.numpy(), prefix["epoch0_parent_order"])
            or parents.numel() != single["train_rows"]):
        raise RuntimeError("NATIVE_PREFIX_NATIVE_ORDER_MISMATCH")
    cohort = build_chronological_measurement_cohort(chronological_prefix["design"],
        {"path": str(files["entry_measurement_coordinates"]), "sha256": val.file_sha256(files["entry_measurement_coordinates"])}, role="control")
    context = {"entry_training_phase": phase, "entry_supervision_identity": identity,
        "physical_sources": physical["physical_sources"],
        "entry_only_base_normalization": physical["entry_only_base_normalization"],
        "base_contract_sha256": physical["base_contract_sha256"], "evaluation_cohort": cohort,
        "val_sequence_audit": physical["sequence_source_audits"]["val"], **dict(val_limits)}
    trainer._native_candidate_val_context_binding(context)
    optimizer, weight_ema, scheduler, ema_derivation = _component_optimizer_state(
        model=model, entry_phase=phase, train_rows=int(parents.numel()), batch_size=batch_size,
        epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay)
    schedule = {"schema_version": "gx1_entry_only_dynamic_m5_native_epoch_schedule_v1", "epoch_index": 0,
        "entry_pair_count": int(parents.numel()), "drop_last": False,
        "population_witness": single["binding"], "native_sampler_contract": single["witness"]["native_sampler_contract"]}
    schedule["schedule_sha256"] = val.canonical_sha256(schedule)
    return {"model": model, "optimizer": optimizer, "weight_ema": weight_ema, "lr_scheduler": scheduler,
        "train_ds": datasets["train"], "val_ds": datasets["val"], "train_probe_ds": train_probe,
        "native_val_context": context, "input_normalization": normalization, "metadata": meta,
        "per_tf_seq_lens": per_tf, "seed_binding": seed_binding, "weight_ema_derivation": ema_derivation,
        "full_population_schedule": schedule, "chronological_prefix": prefix["bindings"],
        "model_functions": prefix["model_functions"], "prefix_parent_population": parents.numpy().copy(),
        "prefix_epoch0_parent_order": order.numpy().copy(), "effective_train_rows": int(parents.numel()),
        "entry_m5_path_single_epoch": single, "entry_observed_market": identity,
        "entry_only_session_binding": session_binding,
        DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY: loss_scale_identity,
        "unified_exit_lifecycle_evidence": {"schema_version": "gx1_native_entry_only_dynamic_m5_input_lineage_v1",
            "physical_sources": physical["physical_sources"], "base_normalization": physical["entry_only_base_normalization"],
            "sequence_source_audits": physical["sequence_source_audits"], "exit_lifecycle_bound": False}}


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
    frozen_train_policy_scope: Mapping[str, Any] | None = None,
    observed_entry_scope: Mapping[str, Any] | None = None,
    dynamic_m5_loss_scales: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind legacy full TRAIN/June or explicit fresh-prefix components.

    Main training warm-starts from the completed smoke EMA. Its new candidate
    starts with a fresh optimizer and full-TRAIN EMA horizon; on subsequent
    invocations the coordinator restores its exact optimizer/EMA/target/RNG.
    Current prefix mode binds separate physical TRAIN/VAL and original targets.
    Historical prefix mode preserves its shared TRAIN source for CONTROL256.
    No source checkpoint is modified and no sampler benchmark is rerun here.
    """

    if epochs not in (1, 30) or (epochs == 1 and chronological_prefix is None) or batch_size != 16 or type(exit_backup_steps) is not int or exit_backup_steps not in (1, 5):
        raise RuntimeError("NATIVE_FULL_TRAIN_DECLARED_GEOMETRY_INVALID")
    if device.type == "cuda":
        trainer._require_cuda_trainer_guard_execution(execution_tier="canonical")
    elif device.type != "cpu":
        raise RuntimeError("NATIVE_FULL_TRAIN_DEVICE_INVALID")
    if chronological_prefix is not None:
        design = val._read(_bound_artifact(chronological_prefix["design"]))
        if design.get("scope", {}).get("entry_only_training_phase") == trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
            if (any(value is not None for value in (seed_launch_path, seed_authority_path, seed_authority_file_sha256,
                                                    exit_reference_policy, frozen_train_policy_scope))
                    or exit_backup_steps != 1):
                raise RuntimeError("NATIVE_DYNAMIC_M5_LEGACY_COMPONENT_FORBIDDEN")
            return _build_dynamic_m5_entry_components(files=files, chronological_prefix=chronological_prefix,
                observed_entry_scope=observed_entry_scope, dataset_run_id=dataset_run_id,
                device=device, batch_size=batch_size, epochs=epochs,
                seed=seed, learning_rate=learning_rate, weight_decay=weight_decay, val_limits=val_limits,
                dynamic_m5_loss_scales=dynamic_m5_loss_scales)
    if dynamic_m5_loss_scales is not None:
        raise RuntimeError("NATIVE_DYNAMIC_M5_LOSS_SCALE_PHASE_INVALID")
    trainer._set_deterministic(seed, device, "deterministic_fp32")
    selected_sampler = _require_component_sampler(
        files=files, chronological_prefix=chronological_prefix,
    )
    prefix = None
    if observed_entry_scope is not None and chronological_prefix is None:
        raise RuntimeError("NATIVE_OBSERVED_ENTRY_PREFIX_REQUIRED")
    if frozen_train_policy_scope is not None and (
            chronological_prefix is None
            or frozen_train_policy_scope["origin_recipe"]["chronological_prefix"] != chronological_prefix
            or frozen_train_policy_scope["plan"]["val_limits"] != val_limits):
        raise RuntimeError("FROZEN_TRAIN_POLICY_COMPONENT_SCOPE_INVALID")
    if chronological_prefix is not None:
        if any(value is not None for value in (seed_launch_path, seed_authority_path, seed_authority_file_sha256)):
            raise RuntimeError("NATIVE_PREFIX_OLD_INITIALIZATION_FORBIDDEN")
        prefix = _require_prefix_component_bindings(
            chronological_prefix, files=files, seed=seed, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay)
        if epochs != (1 if "entry_m5_path_single_epoch" in prefix else 30):
            raise RuntimeError("NATIVE_FULL_TRAIN_DECLARED_GEOMETRY_INVALID")
        if "entry_m5_path_single_epoch" in prefix and (observed_entry_scope is None
                or observed_entry_scope.get("training_phase") != prefix["entry_m5_path_single_epoch"]["witness"]["training_phase"]):
            raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_TARGET_MISMATCH")
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
    physical_prefix = prefix is not None and "physical_preprocessing" in prefix
    if physical_prefix and frozen_train_policy_scope is not None:
        raise RuntimeError("NATIVE_PHYSICAL_FROZEN_TRAIN_POLICY_COORDINATES_NOT_BOUND")
    train_window = val._read(files["entry_train_manifest"])["splits"]["train"]
    train_start, train_end = (val.pd.Timestamp(train_window[key]) for key in ("start", "end"))
    if physical_prefix:
        for split in ("train", "val"):
            declared = prefix["physical_preprocessing"]["physical_sources"][split]["declared_window"]
            observed = val._read(files[f"entry_{split}_manifest"])["splits"][split]
            if any(val.pd.Timestamp(observed[key]) != val.pd.Timestamp(declared[key]) for key in ("start", "end")):
                raise RuntimeError("NATIVE_PHYSICAL_COMPONENT_WINDOW_MISMATCH")
    elif (
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
    if physical_prefix:
        for split in ("train", "val"):
            for kind in ("parquet", "manifest"):
                if file_bindings[f"entry_{split}_{kind}"] != prefix["physical_preprocessing"]["physical_sources"][split][kind]:
                    raise RuntimeError("NATIVE_PHYSICAL_COMPONENT_SOURCE_BYTES_MISMATCH")
    val_audit = (files["val_sequence_source_audit"] if physical_prefix
                 else files["sequence_source_audit"] if prefix is not None
                 else val._val_sequence_source_audit(meta, file_bindings))
    physical_splits = ("train",) if prefix is not None and not physical_prefix else ("train", "val")
    datasets = {
        split: val.EntryV10CtxDataset(
            files[f"entry_{split}_parquet"], seq_len=int(meta["seq_len"]),
            m5_prebuilt_path=files["m5_prebuilt"], per_tf_seq_lens=per_tf,
            multi_tf_closed_bar=True,
            sequence_source_audit_json=(files["sequence_source_audit"] if split == "train" else val_audit),
        ) for split in physical_splits
    }
    if prefix is not None:
        # Physical TRAIN/VAL keep their own original target columns. Only the
        # historical prefix uses two role masks over a shared physical TRAIN.
        if not physical_prefix:
            datasets["val"] = copy.copy(datasets["train"])
        label_roles = (("train", "TRAIN"),) if frozen_train_policy_scope is not None else (("train", "TRAIN"), ("val", "CONTROL256"))
        for split, role in label_roles:
            datasets[split].bind_policy_dependent_auxiliary_targets(
                result_path=Path(chronological_prefix["labels_result"]["path"]),
                expected_result_sha256=chronological_prefix["labels_result"]["sha256"],
                expected_design_sha256=chronological_prefix["design"]["sha256"], role=role)
        if observed_entry_scope is not None:
            from gx1.contracts.entry_observed_market_horizons_v1 import bind_observed_entry_dataset
            if not physical_prefix or frozen_train_policy_scope is not None:
                raise RuntimeError("NATIVE_OBSERVED_ENTRY_PHYSICAL_SCOPE_REQUIRED")
            for split in ("train", "val"):
                bind_observed_entry_dataset(datasets[split], observed_entry_scope)
            admitted = getattr(datasets["train"], "_entry_observed_market_admitted_rows", None)
            if admitted is not None and not bool(admitted[np.asarray(prefix["eligible_parent_rows"], dtype=np.int64)].all()):
                # the frozen coordinates must only draw rows with complete horizon outcomes
                raise RuntimeError("NATIVE_PREFIX_ELIGIBLE_ROWS_NOT_ADMITTED_FOR_HORIZON_TARGET")
        # Measurement shares the TRAIN inputs and bound labels, while lifecycle
        # materialization remains attached only to the optimizer's dataset.
        train_probe_ds = copy.copy(datasets["train"])
        if frozen_train_policy_scope is not None:
            datasets["val"] = copy.copy(train_probe_ds)
    corpus_owner = (val.UnifiedExitLifecycleCorpus.from_random_access_index
                    if physical_prefix else val.UnifiedExitLifecycleCorpus)
    corpus = corpus_owner(
        root_manifest_path=(files["random_access_root"] if physical_prefix
                            else files["feature_lifecycle_root"]),
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
    # Historical recipes preserve their measured geometry. Current physical
    # TRAIN must carry a fresh choice bound to these exact candidate/root bytes.
    adapter = factory(selected_sampler["transition_budget_per_epoch"]
                      if selected_sampler is not None else 65_536)
    if selected_sampler is not None:
        actual = adapter.random_access_training_bindings_v1()
        if (actual["sampler_contract"] != selected_sampler["selected_sampler_contract"]
                or {key: actual[key] for key in ("reference_policy", "reference_cutoff_time_ns") if key in actual}
                    != selected_sampler["reference_workload"]):
            raise RuntimeError("NATIVE_PREFIX_ADAPTER_SAMPLER_MISMATCH")
    schedule = adapter.set_full_population_epoch_index(0)
    datasets["train"].bind_unified_exit_lifecycle_v2(adapter)
    datasets["train"].bind_random_access_entry_coordinate_mapping_v1(
        parent_entry_row_indices=parents, child_entry_row_indices=children,
    )

    source_split = "train" if prefix is not None and not physical_prefix else "val"
    val_binding = root["splits"][source_split]
    index_path = Path(val_binding["index_parquet_path"])
    manifest_path = Path(val_binding["manifest_path"])
    frame = val.pd.read_parquet(index_path)
    manifest = val.require_random_access_index_manifest(
        val._read(manifest_path), expected_split=source_split, index_frame=frame,
        index_path=index_path, verify_sources=False,
    )
    if (
        frame["entry_row_index"].astype("int64").tolist() != list(range(len(datasets["val"]) if physical_prefix else len(datasets["train"]) if prefix is not None else 5_508))
        or val_binding["index_parquet_sha256"] != val.file_sha256(index_path)
        or val_binding["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("NATIVE_FULL_TRAIN_JUNE_VAL_BINDING_INVALID")
    evaluation_cohort = None
    if frozen_train_policy_scope is not None:
        from gx1.contracts.unified_exit_bounded_val_cohort_v1 import require_bounded_val_cohort, TRAIN_ROLLOUT_SCHEMA
        evaluation_cohort = require_bounded_val_cohort(frozen_train_policy_scope["cohort"])
        if (evaluation_cohort["schema_version"] != TRAIN_ROLLOUT_SCHEMA
                or evaluation_cohort["source_index"] != {"path":str(index_path),"sha256":val.file_sha256(index_path)}):
            raise RuntimeError("FROZEN_TRAIN_POLICY_COMPONENT_COHORT_INVALID")
    elif prefix is not None:
        from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_control_cohort
        evaluation_cohort = build_chronological_control_cohort(
            chronological_prefix["design"],
            source_index_binding={"path": str(index_path), "sha256": val.file_sha256(index_path)},
            **({"source_index_manifest_binding": {"path": str(manifest_path), "sha256": val.file_sha256(manifest_path)}}
               if physical_prefix else {}),
        )
    control_frame = frame if evaluation_cohort is None else frame.iloc[evaluation_cohort["entry_row_indices"]].copy()
    all_val_states = int(control_frame["lifecycle_state_count"].sum())
    if frozen_train_policy_scope is None and (
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
    val_factory = _bound_measurement_state_factory(
        manifest=manifest, index_path=index_path, manifest_path=manifest_path, root_path=root_path,
        frame=frame, dataset=datasets["val"], source_owner=corpus.splits[source_split],
        readiness=readiness, cost_authority=files["train_cost_authority"], source_split=source_split,
        evaluation_cohort=evaluation_cohort if physical_prefix else None,
    )
    if physical_prefix:
        train_manifest_path = Path(train_binding["manifest_path"])
        train_measurement_factory = _bound_measurement_state_factory(
            manifest=val._read(train_manifest_path), index_path=Path(train_binding["index_parquet_path"]),
            manifest_path=train_manifest_path, root_path=root_path,
            frame=adapter._native_random_access_index, dataset=train_probe_ds,
            source_owner=corpus.splits["train"], readiness=readiness,
            cost_authority=files["train_cost_authority"], source_split="train",
        )
    if frozen_train_policy_scope is not None:
        # Retain physical lifecycles; only the observation support is bounded.
        end = np.searchsorted(val_factory.times.asi8 + 60_000_000_000,
                              evaluation_cohort["observation_cutoff_time_ns"], side="right")
        counts = np.minimum(control_frame["lifecycle_state_count"].to_numpy(dtype="int64"),
                            end - control_frame["child_m1_start_row"].to_numpy(dtype="int64"))
        if (len(counts) != 256 or np.any(counts <= 0)
                or val_limits["max_state_views"] != int(counts.sum())
                or val_limits["max_model_forwards"] != int(counts.max())):
            raise RuntimeError("FROZEN_TRAIN_POLICY_ACTUAL_FOOTPRINT_MISMATCH")
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
    from gx1.contracts.entry_model_native_training_objective_v1 import (
        JOINT_FITTED_Q_TRAINING_PHASE, ENTRY_ONLY_TRAINING_PHASES,
    )
    observed_binding = getattr(datasets["train"], "_entry_observed_market_binding", None)
    if observed_entry_scope is None:
        if observed_binding is not None:
            raise RuntimeError("NATIVE_FULL_TRAIN_EMA_PHASE_BINDING_MISMATCH")
        entry_phase = JOINT_FITTED_Q_TRAINING_PHASE
    else:
        identity = observed_binding.get("identity") if isinstance(observed_binding, Mapping) else None
        if (not isinstance(identity, Mapping) or identity.get("training_phase") not in ENTRY_ONLY_TRAINING_PHASES
                or identity["training_phase"] != observed_entry_scope.get("training_phase")):
            raise RuntimeError("NATIVE_FULL_TRAIN_EMA_PHASE_BINDING_MISMATCH")
        entry_phase = identity["training_phase"]
    queries = trainer._dynamic_m5_model_kwargs(entry_phase).get("dynamic_m5_query_minutes")
    if queries is not None:
        for binding in (observed_entry_scope, identity):
            declared = binding.get("query_minutes")
            if (not isinstance(declared, (tuple, list))
                    or any(type(value) is not int for value in declared)
                    or tuple(declared) != queries):
                raise RuntimeError("NATIVE_COMPONENT_SCOPE_DYNAMIC_QUERY_MISMATCH")
    child = val.require_composite_normalization_binding(
        val._read(files["child_composite_normalization"]),
    )
    if prefix is not None:
        input_norm = child["base_feature_normalization"]["artifact"]["contract"]
        model, seed_binding = _fresh_prefix_model(metadata=meta, normalization=input_norm, device=device, seed=seed,
            entry_training_phase=entry_phase)
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
    _configure_component_training_phase(model, entry_phase)
    effective_train_rows = (len(prefix["eligible_parent_rows"]) if prefix is not None
                            else root["latest_year_population"]["splits"]["train"]["selected_entry_row_count"]
                            if latest_year else len(datasets["train"]))
    optimizer, weight_ema, scheduler, ema_derivation = _component_optimizer_state(
        model=model, entry_phase=entry_phase,
        train_rows=(len(prefix["eligible_parent_rows"]) if prefix is not None else len(datasets["train"])),
        batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay)
    return {
        "model": model, "optimizer": optimizer, "weight_ema": weight_ema,
        "lr_scheduler": scheduler, "train_ds": datasets["train"],
        "val_ds": datasets["val"], "native_val_context": val_context,
        "input_normalization": input_norm, "metadata": meta, "per_tf_seq_lens": per_tf,
        "seed_binding": seed_binding, "weight_ema_derivation": ema_derivation,
        "full_population_schedule": schedule,
        **({"chronological_prefix": prefix["bindings"], "model_functions": prefix["model_functions"],
            "prefix_parent_population": prefix["eligible_parent_rows"],
            "train_probe_ds": train_probe_ds,
            "prefix_epoch0_parent_order": prefix["epoch0_parent_order"]} if prefix is not None else {}),
        "effective_train_rows": effective_train_rows,
        **({"entry_m5_path_single_epoch": prefix["entry_m5_path_single_epoch"]}
           if prefix is not None and "entry_m5_path_single_epoch" in prefix else {}),
        **({"entry_observed_market": datasets["train"]._entry_observed_market_binding["identity"]}
           if observed_entry_scope is not None else {}),
        **({"measurement_state_factories": {"train": train_measurement_factory, "control": val_factory}}
           if physical_prefix else {}),
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
        epochs=1 if "entry_m5_path_single_epoch" in components else 30, early_stopping_patience=5, early_stopping_min_delta=0.0,
        minimum_epochs_before_stop=1, save_top_k=(0 if "entry_only_session_binding" in components else 1),
        out_bundle_dir=output, gx1_data_override=gx1_data,
        run_id=run_id, dataset_run_id=dataset_run_id,
        train_parquet=files["entry_train_parquet"], val_parquet=files["entry_val_parquet"],
        m5_prebuilt_path=files["m5_prebuilt"],
        unified_exit_lifecycle_manifest_path=(None if "entry_only_session_binding" in components else files["random_access_root"]),
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
        checkpoint_monitor=(DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
            if "entry_only_session_binding" in components else native_checkpoint_monitor(
                val._load_val_economics_readiness(files["economics_readiness"])["economics_objective_contract"])),
        candidate_resume_origin=candidate_resume_origin,
        chronological_continuation=components.get("chronological_continuation"),
        entry_learning_study=components.get("entry_learning_study"),
        entry_learning_observer=components.get("entry_learning_observer"),
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
    with file_verification_scope():
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
    frozen_train_scope = None
    if "frozen_train_policy_evaluation" in recipe or "frozen_entry_selector_probe" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_frozen_train_policy_evaluation
        frozen_train_scope = require_frozen_train_policy_evaluation(recipe, execution_budget=budget)
    prefix_mode = "chronological_prefix" in recipe
    output = Path(recipe["out_bundle_dir"])
    device = trainer._resolve_device("cuda")
    component_started = time.monotonic()
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    observed_entry_scope = require_observed_entry_scope(recipe)
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
        **({"frozen_train_policy_scope":frozen_train_scope} if frozen_train_scope is not None else {}),
        **({"observed_entry_scope":observed_entry_scope} if observed_entry_scope is not None else {}),
        **({DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY: recipe[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY]}
           if DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY in recipe else {}),
    )
    print(json.dumps({"event": "NATIVE_COMPONENT_SETUP", "report_only": True,
                      "component_wall_seconds": time.monotonic() - component_started,
                      "invocation_elapsed_seconds": time.monotonic() - started}), flush=True)
    if not prefix_mode:
        smoke = val._read(Path(recipe["smoke_full_val"]["path"]))
        if components["seed_binding"]["model_state_sha256"] != smoke["checkpoint_binding"]["model_state_sha256"]:
            raise RuntimeError("NATIVE_FULL_TRAIN_ACTUAL_SEED_MODEL_MISMATCH")
    if frozen_train_scope is not None:
        if "entry_selector_probe" in frozen_train_scope:
            return _run_frozen_entry_representation_probe(components=components, scope=frozen_train_scope,
                recipe=recipe, device=device, output=output, recipe_file_sha256=recipe_file_sha256,
                invocation_started=started)
        return _run_frozen_train_policy_validation(components=components, scope=frozen_train_scope,
            recipe=recipe, device=device, output=output, recipe_file_sha256=recipe_file_sha256,
            invocation_started=started)
    if "entry_gradient_diagnostic" in recipe:
        return _run_entry_gradient_diagnostic(components=components,recipe=recipe,device=device,output=output,
            recipe_file_sha256=recipe_file_sha256,invocation_started=started)
    if "frozen_readout_evaluation" in recipe:
        return _run_frozen_readout_validation(
            components=components, recipe=recipe, device=device, output=output,
            recipe_file_sha256=recipe_file_sha256, invocation_started=started)
    if "entry_dynamic_m5_technical_qualification" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_dynamic_m5_technical_qualification
        with file_verification_scope():
            technical_scope = require_dynamic_m5_technical_qualification(recipe, execution_budget=budget)
        return _run_dynamic_m5_technical_qualification(components=components, scope=technical_scope,
            recipe=recipe, files=files, output=output, device=device, provenance=provenance,
            execution_budget=budget, recipe_file_sha256=recipe_file_sha256, invocation_started=started)
    if "chronological_initial_measurement" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_initial_measurement
        with file_verification_scope():
            initial_scope = require_chronological_initial_measurement(recipe, execution_budget=budget)
        _restore_prefix_initial_measurement_state(components=components, scope=initial_scope, device=device)
    if "chronological_learning_measurement" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_learning_measurement
        with file_verification_scope():
            learning_scope = require_chronological_learning_measurement(recipe)
            if "entry_m5_path_single_epoch" in recipe:
                from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_m5_single_epoch_run
                epoch_scope = require_m5_single_epoch_run(recipe, execution_budget=budget)
                learning_scope["entry_m5_path_single_epoch"] = epoch_scope["entry_m5_path_single_epoch"]
            from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_continuation
            continuation = require_chronological_continuation(recipe)
            if continuation:
                learning_scope["continuation"] = continuation
                components["chronological_continuation"] = continuation
        _restore_prefix_initial_measurement_state(components=components, scope=learning_scope, device=device)
    if "entry_learning_study" in recipe:
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_entry_learning_study_run
        study = require_entry_learning_study_run(recipe, execution_budget=budget)
        components["entry_learning_study"] = study
        components["entry_learning_observer"] = _entry_learning_study_observer(
            components=components, study=study, recipe=recipe, device=device, output=output)
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
                and learning_scope.get("continuation", {}).get("resume_equivalence") is not True
                and paused.evidence["reason"] == "optimizer_step_ceiling"
                and paused.evidence["global_optimizer_steps"] == (learning_scope["entry_m5_path_single_epoch"]["optimizer_steps"]
                    if "entry_m5_path_single_epoch" in learning_scope else learning_scope["continuation"]["plan"]["stop_after_optimizer_steps"] if learning_scope.get("continuation") else 256)):
            paused.evidence["chronological_final_measurement"] = _run_prefix_initial_measurement(
                components=components, scope=learning_scope, recipe=recipe, output=output,
                device=device, invocation_started=started, pause_evidence=paused.evidence, optimizer_steps=paused.evidence["global_optimizer_steps"])
        if "entry_m5_path_single_epoch" in components and paused.evidence["global_optimizer_steps"] == components["entry_m5_path_single_epoch"]["optimizer_steps"]:
            single = components["entry_m5_path_single_epoch"]
            if (paused.evidence["phase"] != "train" or paused.evidence["epoch_index"] != 0
                    or paused.evidence["next_batch_offset"] != single["optimizer_steps"]):
                raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_FINAL_CURSOR_INVALID")
            paused.evidence["entry_m5_path_single_epoch"] = {
                "population_witness": single["binding"], "unique_train_rows": single["train_rows"],
                "optimizer_steps": single["optimizer_steps"], "last_batch_rows": single["last_batch_rows"],
                "train_epochs_consumed": 1, "drop_last": False, "full_val_executed": False,
                "teacher_refreshed": False, "additional_epoch_authorized": False}
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



def _entry_learning_study_observer(*, components, study, recipe, device, output):
    """Observe only predeclared rows/steps, without changing training RNG or state."""
    from gx1.contracts.entry_observed_market_v1 import entry_learning_study_control_dataset
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        observed_entry_batch_targets, observed_entry_batch_targets_by_horizon,
    )
    from gx1.contracts.unified_exit_random_access_state_view_v1 import _structured_sha256
    from gx1.scripts.benchmark_unified_exit_random_access_train_v1 import _atomic_write_new_json
    def rng_identity(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {key:rng_identity(item) for key,item in value.items()}
        if isinstance(value, (list,tuple)):
            return [rng_identity(item) for item in value]
        return value
    def rng_hash():
        return _structured_sha256(rng_identity(trainer._attended_session_rng_state(device=device)))
    model = components["model"]
    datasets = {"train":components["train_probe_ds"]}
    window_phase = "window_days" in study["phase_spec"]
    if window_phase or study["phase"] in ("curve", "epoch"):
        datasets["control"] = entry_learning_study_control_dataset(components["val_ds"], study=study)
    directory = output.parent / "OBSERVATIONS"
    directory.mkdir(exist_ok=True)
    train_rows_key = study["phase"] + "_train_probe" if window_phase else ("fit" if study["phase"] == "fit" else "train_probe")
    rows_by_role = {"train":study["rows"][train_rows_key], "control":study["rows"]["control"]}
    for role, dataset in datasets.items():
        if not np.array_equal(dataset.indices, np.arange(len(dataset))):
            raise RuntimeError("ENTRY_STUDY_NONIDENTITY_DATASET_INDEX")
        if not dataset._policy_dependent_auxiliary_bound_rows[rows_by_role[role]].all():
            raise RuntimeError("ENTRY_STUDY_UNBOUND_EVALUATION_ROWS")

    def observe(*, optimizer_steps, session):
        roles = [role for role in datasets
                 if optimizer_steps in study["phase_spec"][role + "_observation_steps"]]
        if not roles:
            return
        original_mode = model.training
        state_hash = trainer._model_state_sha256(model)
        original_rng_hash = rng_hash()
        try:
            model.eval()
            for role in roles:
                target_path = directory / f"step_{optimizer_steps:06d}_{role}.json"
                if target_path.exists():
                    old = val._read(target_path)
                    if (old["model_state_sha256"] != state_hash
                            or old["entry_learning_study"] != study["selection"]
                            or old["optimizer_steps"] != optimizer_steps):
                        raise RuntimeError("ENTRY_STUDY_OBSERVATION_COLLISION")
                    continue
                rows = rows_by_role[role]
                dataset = datasets[role]
                # the bound design decides the exact loss key; an unbound dataset is the v1 key
                observed_binding = getattr(dataset, "_entry_observed_market_binding", None) or {"identity": {}}
                loader = val.DataLoader(dataset, batch_size=16, sampler=val._ExactSampler(rows.tolist()),
                    num_workers=0, generator=torch.Generator().manual_seed(0))
                predictions, targets = [], []
                targets_by_horizon = {}
                consumed = 0
                for batch in loader:
                    count = len(batch["entry_row_index"])
                    if batch["entry_row_index"].tolist() != rows[consumed:consumed+count].tolist():
                        raise RuntimeError("ENTRY_STUDY_OBSERVATION_ROW_MAPPING")
                    with torch.inference_mode():
                        result = trainer._model_forward_fp32(
                            model, batch["seq_x"].to(device), batch["snap_x"].to(device),
                            ctx_cat=batch["ctx_cat"].to(device), ctx_cont=batch["ctx_cont"].to(device),
                            **trainer._multi_tf_kwargs_from_batch(batch, device))
                        q = result["entry_action_q_bps"]
                        y, valid = observed_entry_batch_targets(batch, observed_binding, device=device)
                        for horizon, horizon_y in observed_entry_batch_targets_by_horizon(
                                batch, observed_binding, device=device).items():
                            targets_by_horizon.setdefault(horizon, []).append(horizon_y.detach().cpu().numpy().copy())
                    if (q.shape != (count,3) or q.dtype != torch.float32
                            or not bool(torch.isfinite(q).all()) or not bool(valid.all())):
                        raise RuntimeError("ENTRY_STUDY_OBSERVATION_INVALID")
                    predictions.append(q.detach().cpu().numpy().copy())
                    targets.append(y.detach().cpu().numpy().copy())
                    consumed += count
                if consumed != len(rows):
                    raise RuntimeError("ENTRY_STUDY_OBSERVATION_INCOMPLETE")
                q, y = np.concatenate(predictions), np.concatenate(targets)
                error = q.astype(float)-y.astype(float)
                contrast_error = error[:,0]-error[:,1]
                actions = np.argmax(q,axis=1)
                unique = (q == q.max(axis=1,keepdims=True)).sum(axis=1) == 1
                report = {"schema_version":"gx1_entry_learning_study_observation_v1",
                    "entry_learning_study":study["selection"], "role":role,
                    "optimizer_steps":optimizer_steps, "rows":rows.tolist(),
                    "time_ns":val.pd.DatetimeIndex(val.pd.to_datetime(dataset.df.iloc[rows]["time"],utc=True)).as_unit("ns").asi8.tolist(),
                    "source":study["observed_scope"]["physical_sources"]["train" if role=="train" else "val"],
                    "model_state_sha256":state_hash,
                    "native_recipe_source_bindings_sha256":recipe["source_bindings_sha256"],
                    "training_pointer_snapshot":val._read(session._active_path),
                    "training_pointer_origin":str(session._active_path),
                    "checkpoint_retention":"existing two-slot owner; intermediate model identity and predictions preserved here",
                    "training_rng_sha256":original_rng_hash,
                    "predictions_bps":q.tolist(),"targets_bps":y.tolist(),
                    "metrics":{"side_mse":np.mean(error[:,:2]**2,axis=0).tolist(),
                        "side_centered_mse":np.mean((error[:,:2]-error[:,:2].mean(axis=0))**2,axis=0).tolist(),
                        "contrast_mse":float(np.mean(contrast_error**2)),
                        "contrast_centered_mse":float(np.mean((contrast_error-contrast_error.mean())**2)),
                        "action_counts":np.bincount(actions,minlength=3).tolist(),
                        "ties":int((~unique).sum()),
                        "positive_estimated_entry_rows":int(((q[:,:2].max(axis=1)>0)&(actions!=2)&unique).sum())},
                    "economic_rollout":False,"exit_forwards":0,"test_data_used":False}
                if targets_by_horizon:
                    # Observation only: the same Q against every observed horizon target (rank-IC and
                    # MSE); the loss used exactly the design's decision horizon.
                    from scipy.stats import spearmanr
                    report["observed_horizons"] = {}
                    for horizon, chunks in sorted(targets_by_horizon.items()):
                        hy = np.concatenate(chunks)
                        herr = q.astype(float)-hy.astype(float)
                        contrast_q, contrast_y = q[:,0]-q[:,1], hy[:,0]-hy[:,1]
                        report["observed_horizons"][str(horizon)] = {
                            "targets_bps":hy.tolist(),
                            "side_mse":np.mean(herr[:,:2]**2,axis=0).tolist(),
                            "contrast_mse":float(np.mean((herr[:,0]-herr[:,1])**2)),
                            "contrast_rank_ic":float(spearmanr(contrast_q,contrast_y).statistic),
                            "is_loss_target":horizon == observed_binding["identity"]["decision_horizon_trading_days"]}
                if (trainer._model_state_sha256(model) != state_hash
                        or rng_hash() != original_rng_hash):
                    raise RuntimeError("ENTRY_STUDY_OBSERVATION_MUTATED_TRAINING")
                _atomic_write_new_json(target_path,report)
                print(json.dumps({"event":"ENTRY_STUDY_OBSERVATION","phase":study["phase"],
                    "role":role,"optimizer_steps":optimizer_steps,"rows":len(rows),
                    "metrics":report["metrics"]}),flush=True)
        finally:
            model.train(original_mode)
        if rng_hash() != original_rng_hash:
            raise RuntimeError("ENTRY_STUDY_OBSERVER_RNG_CHANGED")
    return observe


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


@contextmanager
def _joint_probe_exit_forward(model):
    """Retain cuDNN backward state only in dropout-free GRUs; verify eval parity."""
    recurrent = [m for m in model.modules() if isinstance(m, torch.nn.RNNBase)]
    if (not recurrent or any(m.training for m in model.modules())
            or any(not isinstance(m, torch.nn.GRU) or m.dropout != 0 for m in recurrent)):
        raise RuntimeError('JOINT_PROBE_RECURRENT_MODE_INVALID')
    original = model.forward_exit_random_access_batch
    had_override = 'forward_exit_random_access_batch' in model.__dict__
    report = {'online_calls': 0, 'max_abs_difference_bps': 0., 'actions_identical': False,
              'recurrent_modules': len(recurrent), 'dropout_enabled': False}
    def checked(**kwargs):
        report['online_calls'] += 1
        if report['online_calls'] != 1 or not torch.is_grad_enabled():
            raise RuntimeError('JOINT_PROBE_EXIT_FORWARD_COUNT_INVALID')
        with torch.inference_mode():
            expected = original(**kwargs)['exit_action_q_bps']
        try:
            for module in recurrent: module.training = True
            output = original(**kwargs)
        finally:
            for module in recurrent: module.training = False
        actual = output['exit_action_q_bps'].detach()
        report['max_abs_difference_bps'] = float((actual - expected).abs().max())
        report['actions_identical'] = torch.equal(actual.argmax(-1), expected.argmax(-1))
        if not torch.allclose(actual, expected, atol=1e-4, rtol=0) or not report['actions_identical']:
            raise RuntimeError('JOINT_PROBE_EXIT_INFERENCE_PARITY_INVALID')
        return output
    model.forward_exit_random_access_batch = checked
    try:
        yield report
        if report['online_calls'] != 1:
            raise RuntimeError('JOINT_PROBE_EXIT_FORWARD_COUNT_INVALID')
    finally:
        if had_override: model.forward_exit_random_access_batch = original
        else: delattr(model, 'forward_exit_random_access_batch')
        for module in recurrent: module.training = False


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
        with _joint_probe_exit_forward(model) as exit_parity:
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
            'optimizer_steps':0,'model_forwards':6,'exit_inference_parity':exit_parity,
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
            report.update(schema_version="gx1_joint_update_diagnostic_result_v2",
                decision="JOINT_GRADIENT_AND_ADAM_DIRECTION_MEASURED_NO_OPTIMIZER_STEP",
                variants=["final"],model_forwards=6,
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
    dynamic = getattr(components["model"], "entry_training_phase", None) == trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if dynamic and (not isinstance(components.get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY), Mapping)
            or initial.get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY) != components[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY]):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_LOSS_SCALE_MISMATCH")
    state = torch.load(_bound_artifact(initial["initial_state"]), map_location="cpu", weights_only=False)
    expected = initial["online_model_state_sha256"]
    functions = components.get("model_functions")
    if (functions not in (trainer._PREFIX_MODEL_FUNCTIONS, trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)
            or any("model_functions" in value and value["model_functions"] != functions
                   for value in (initial, state))
            or (functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
                and any(value.get("model_functions") != functions for value in (initial, state)))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_FUNCTION_BINDING_INVALID")
    if (state.get("schema_version") != "gx1_prefix_fresh_initial_state_v1"
            or (dynamic and state.get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY) != components[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY])
            or state.get("chronological_prefix") != components["chronological_prefix"]
            or state.get("model_forwards") != 0 or state.get("optimizer_steps") != 0
            or state.get("checkpoint_loaded") is not False
            or state["optimizer_state"]["state"] or state["weight_ema_state"]["steps"] != 0
            or any(val.canonical_model_state_sha256(value) != expected for value in (
                state["model_state"], state["target_model_state"], state["weight_ema_state"]["shadow"]))
            or components["seed_binding"]["model_state_sha256"] != expected
            or components["weight_ema_derivation"] != initial["ema_derivation"]):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_STATE_MISMATCH")
    # Empty optimizer history does not bind its parameter groups or schedule.
    # Compare with the fresh recipe-built components before mutating any object.
    optimizer = components["optimizer"]
    if val.canonical_sha256(state["optimizer_state"]) != val.canonical_sha256(optimizer.state_dict()):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_OPTIMIZER_STATE_MISMATCH")
    scheduler = components["lr_scheduler"]
    expected_schedule = None if scheduler is None else scheduler.state_dict()
    if val.canonical_sha256(state["lr_scheduler_state"]) != val.canonical_sha256(expected_schedule):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_SCHEDULER_MISMATCH")
    expected_ema = components["weight_ema"].checkpoint_state()
    saved_ema = state["weight_ema_state"]
    if (set(saved_ema) != set(expected_ema)
            or any(type(saved_ema[key]) is not type(expected_ema[key]) or saved_ema[key] != expected_ema[key]
                   for key in ("decay", "steps"))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_EMA_STATE_MISMATCH")
    model = components["model"]
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    components["weight_ema"].restore_checkpoint_state(saved_ema, model=model)
    if scheduler is not None:
        scheduler.load_state_dict(state["lr_scheduler_state"])
    # The saved preparation ran on CPU. CUDA randomness starts from the same
    # declared seed; the CPU/Python/NumPy state comes from the saved preparation.
    trainer._restore_attended_session_rng_state(state["rng_state"], device=torch.device("cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(initial["seed_binding"]["seed"]))
    if trainer._model_state_sha256(model) != expected:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_MODEL_MISMATCH")


def _dynamic_m5_technical_tree_sha256(value):
    """Use the existing typed ndarray digest for exact nested checkpoint evidence."""
    from gx1.contracts.unified_exit_random_access_state_view_v1 import _structured_sha256
    def project(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        if isinstance(item, Mapping):
            return {key: project(raw) for key, raw in item.items()}
        if isinstance(item, (list, tuple)):
            return [project(raw) for raw in item]
        return item
    return _structured_sha256(project(value))


def _dynamic_m5_technical_batch_manifest(batch):
    """Describe actual tensors by role, preserving each nested MTF leaf."""
    import hashlib
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
    inputs = {"seq_x", "snap_x", "ctx_cont", "ctx_cat", *("seq_" + tf.lower() for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES)}
    from gx1.contracts.entry_observed_market_v1 import (
        DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY, DYNAMIC_M5_MFE_TARGET_KEY,
        DYNAMIC_M5_MAE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY, DYNAMIC_M5_CLOSED_MFE_TARGET_KEY,
        DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
    )
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    pointers = {"entry_row_index", "dynamic_entry_parent_row_index", "dynamic_entry_m1_row"}
    targets = {DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_MAE_TARGET_KEY,
        DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY,
        *(f"y_forecast_ret_K{h}" for h in trainer.FORECAST_HORIZONS)}
    masks = {DYNAMIC_M5_MARKOUT_VALID_KEY, DYNAMIC_M5_PATH_VALID_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY}
    groups = {name: {} for name in ("inputs", "targets", "masks", "pointers", "other_batch_tensors")}
    def walk(value, path, root):
        if isinstance(value, torch.Tensor):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            group = ("inputs" if root in inputs else "pointers" if root in pointers else
                     "masks" if root in masks else "targets" if root in targets else
                     "other_batch_tensors")
            groups[group][path] = {"shape": list(value.shape), "dtype": str(value.dtype),
                "storage_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, path + "." + str(key), root)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + "." + str(index), root)
    for name, value in batch.items():
        walk(value, str(name), str(name))
    return groups


def _dynamic_m5_forward_parity(*, model, batch, device, diagnostics=None):
    """Same eval batch through the two actual curve-evaluation/training paths."""
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import DYNAMIC_M5_TECHNICAL_FIXED_SCOPE
    fields = ("dynamic_terminal_net_markout_bps", "dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps",
        "dynamic_closed_mfe_bps", "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred",
        "dynamic_entry_quality_bps_candidate", "dynamic_entry_action_quality_bps_candidate")
    exact = ("dynamic_query_horizon_minutes", "dynamic_entry_quality_best_horizon_mask")
    if (model.training or int(batch["seq_x"].shape[0]) != 16
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_STATE_INVALID")
    before = trainer._model_state_sha256(model)
    rng = trainer._attended_session_rng_state(device=device)
    input_hash = _dynamic_m5_technical_tree_sha256(batch)
    manifest = _dynamic_m5_technical_batch_manifest(batch)
    modes = [(module, module.training) for module in model.modules()]
    snapshots = []
    evidence = {} if diagnostics is None else diagnostics
    evidence.update(model_forwards=0, optimizer_steps=0, execution_passes=[], output_comparisons={})
    try:
        for context in (torch.no_grad, torch.enable_grad):
            trainer._restore_attended_session_rng_state(rng, device=device)
            with context():
                execution = {"context": context.__name__, "grad_enabled": torch.is_grad_enabled(),
                    "inference_mode_enabled": torch.is_inference_mode_enabled(), "model_training": model.training,
                    "training_module_count": sum(int(module.training) for module, _ in modes),
                    "device": str(device), "autocast_enabled": torch.is_autocast_enabled(),
                    "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
                    "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()}
                evidence["execution_passes"].append(execution)
                # Observe host-dispatched ATen attention operators on these two
                # existing forwards only. This is not a CUDA-kernel/timing probe;
                # module hooks would themselves disable the fastpath under test.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=False, profile_memory=False, with_stack=False) as profile:
                    out = trainer._model_forward_fp32(model, batch["seq_x"].to(device), batch["snap_x"].to(device),
                        ctx_cat=batch["ctx_cat"].to(device), ctx_cont=batch["ctx_cont"].to(device),
                        **trainer._multi_tf_kwargs_from_batch(batch, device))
                evidence["model_forwards"] += 1
                execution["host_attention_operator_calls"] = {
                    event.key: event.count for event in profile.key_averages()
                    if event.key.startswith("aten::") and ("attention" in event.key or "transformer" in event.key)}
                del profile
                losses, cells = trainer._dynamic_m5_task_loss_surfaces(out, batch, device, loss_units="train_rms")
                trainer._joint_task_loss(model, losses, loss_units="train_rms")
                values = {name: out[name].detach().cpu().clone() for name in (*fields, *exact)}
                if any(not bool(torch.isfinite(values[name]).all()) for name in fields):
                    raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_NONFINITE")
                queries = trainer._dynamic_m5_model_kwargs(model.entry_training_phase)["dynamic_m5_query_minutes"]
                query = values["dynamic_query_horizon_minutes"]
                quality, candidate, mask = (values[name] for name in ("dynamic_entry_quality_bps_candidate",
                    "dynamic_entry_action_quality_bps_candidate", "dynamic_entry_quality_best_horizon_mask"))
                if (query.dtype != torch.int64 or tuple(query.shape) != (len(queries),)
                        or not torch.equal(query, torch.tensor(queries, dtype=torch.int64))
                        or tuple(quality.shape) != (16, len(queries), 3) or tuple(candidate.shape) != (16, 3)
                        or mask.dtype != torch.bool or tuple(mask.shape) != (16, len(queries), 2)
                        or bool((quality[..., 2] != 0).any()) or bool((candidate[:, 2] != 0).any())
                        or not torch.equal(candidate[:, :2], quality[..., :2].amax(dim=1))
                        or not torch.equal(mask, quality[..., :2] == candidate[:, None, :2])):
                    raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_QUERY_OR_TIE_INVALID")
                snapshots.append((values, dict(cells)))
            del out, losses, values
        left, right = snapshots[0][0], snapshots[1][0]
        tolerance = DYNAMIC_M5_TECHNICAL_FIXED_SCOPE["parity_absolute_tolerance_bps"]
        comparisons = evidence["output_comparisons"]
        for name in fields:
            if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
                raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_SHAPE_CHANGED:" + name)
            delta = left[name].double() - right[name].double()
            worst = np.unravel_index(int(delta.abs().argmax()), tuple(delta.shape))
            comparisons[name] = {"shape": list(left[name].shape), "dtype": str(left[name].dtype),
                "max_absolute_difference_bps": float(delta.abs().max()),
                "rms_difference_bps": float(delta.square().mean().sqrt()),
                "absolute_tolerance_bps": tolerance, "relative_tolerance": 0.0,
                "cells_above_absolute_tolerance": int((delta.abs() > tolerance).sum()),
                "max_absolute_difference_index": [int(index) for index in worst],
                "left_at_max_absolute_difference_bps": float(left[name][worst]),
                "right_at_max_absolute_difference_bps": float(right[name][worst]),
                "left_max_absolute_output_bps": float(left[name].abs().max()),
                "right_max_absolute_output_bps": float(right[name].abs().max())}
            if not torch.allclose(left[name], right[name], atol=tolerance, rtol=0.0):
                raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_VALUES_CHANGED:" + name)
        if any(not torch.equal(left[name], right[name]) for name in exact) or snapshots[0][1] != snapshots[1][1]:
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_QUERY_MASK_CHANGED")
        winner = lambda value: value == value.amax(dim=1, keepdim=True)
        if not torch.equal(winner(left["dynamic_entry_action_quality_bps_candidate"]),
                           winner(right["dynamic_entry_action_quality_bps_candidate"])):
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_ACTION_OR_TIE_CHANGED")
        return {"model_forwards": 2, "optimizer_steps": 0, "model_state_sha256": before,
            "materialized_batch_sha256": input_hash, "typed_batch_manifest": manifest, "output_comparisons": comparisons,
            "query_and_best_horizon_masks_exact": True, "candidate_actions_and_ties_exact": True,
            "supervised_cells": snapshots[0][1],
            "reference_outputs_sha256": _dynamic_m5_technical_tree_sha256(left),
            "execution_passes": evidence["execution_passes"],
            "rng_exactly_preserved": True, "module_modes_exactly_preserved": True}
    finally:
        for module, training in modes:
            module.training = training
        trainer._restore_attended_session_rng_state(rng, device=device)
        if (trainer._model_state_sha256(model) != before or _dynamic_m5_technical_tree_sha256(batch) != input_hash
                or any(parameter.grad is not None for parameter in model.parameters())):
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_CHANGED_STATE")
        evidence["model_input_and_gradient_state_preserved"] = True
        evidence["rng_restored_exactly"] = (
            _dynamic_m5_technical_tree_sha256(trainer._attended_session_rng_state(device=device))
            == _dynamic_m5_technical_tree_sha256(rng))
        evidence["module_modes_restored_exactly"] = all(module.training == training for module, training in modes)


def _run_dynamic_m5_technical_qualification(*, components, scope, recipe, files, output, device,
        provenance, execution_budget, recipe_file_sha256, invocation_started):
    """One four-update mechanics proof using the unchanged durable coordinator."""
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    from gx1.contracts.entry_observed_market_v1 import is_inactive_phase_parameter
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
        DYNAMIC_M5_TECHNICAL_CURSOR_CEILING, require_dynamic_m5_technical_qualification,
    )
    with file_verification_scope():
        checked = require_dynamic_m5_technical_qualification(recipe, invocation_number=1, execution_budget=execution_budget)
    if checked["plan_binding"] != scope["plan_binding"] or device.type != "cuda":
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_SCOPE_CHANGED")
    directory = output.parent / "DYNAMIC_M5_TECHNICAL_QUALIFICATION"
    session_directory = output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    if any(path.exists() or path.is_symlink() for path in (directory, output, session_directory)):
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_OUTPUT_EXISTS")
    directory.mkdir(mode=0o700)
    trainer._fsync_directory(directory.parent)
    def write(path, value):
        _publish_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        return {"path": str(path), "sha256": val.file_sha256(path)}
    def deadline():
        if time.monotonic() - invocation_started >= execution_budget["max_invocation_seconds"] - 60:
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_WALL_LIMIT")
    controls = recipe["trainer_cli"]
    model = components["model"]
    phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    def fresh():
        optimizer, ema, scheduler, derivation = _component_optimizer_state(model=model, entry_phase=phase,
            train_rows=components["effective_train_rows"], batch_size=16, epochs=1,
            learning_rate=controls["learning_rate"], weight_decay=controls["weight_decay"])
        components.update(optimizer=optimizer, weight_ema=ema, lr_scheduler=scheduler, weight_ema_derivation=derivation)
        _restore_prefix_initial_measurement_state(components=components, scope=scope, device=device)
    fresh()
    initial_hash = trainer._model_state_sha256(model)
    inactive_hash = lambda: _dynamic_m5_technical_tree_sha256({name: parameter for name, parameter in model.named_parameters()
        if is_inactive_phase_parameter(name, phase)})
    initial_inactive_hash = inactive_hash()
    modes = [(module, module.training) for module in model.modules()]
    parity = {}
    def optimizer_ema_snapshot():
        return _dynamic_m5_technical_tree_sha256({"optimizer": components["optimizer"].state_dict(),
            "ema": components["weight_ema"].checkpoint_state(),
            "scheduler": None if components["lr_scheduler"] is None else components["lr_scheduler"].state_dict()})
    parity_training_state = optimizer_ema_snapshot()
    parity_rng = _dynamic_m5_technical_tree_sha256(trainer._attended_session_rng_state(device=device))
    try:
        model.eval()
        for role, dataset_key in (("train", "train_probe_ds"), ("control", "val_ds")):
            deadline()
            cohort, dataset = scope["cohorts"][role], components[dataset_key]
            cohort = _require_dynamic_m5_measurement_source(model=model, dataset=dataset, cohort=cohort,
                context=components["native_val_context"])
            if (cohort["cohort_sha256"] != scope["plan"]["measurement_cohort_sha256"][role]
                    or cohort.get("exit_evaluation") is not False
                    or getattr(dataset, "_entry_observed_market_binding", {}).get("identity") != components["entry_observed_market"]):
                raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_COHORT_CHANGED")
            parents = cohort["parent_entry_row_indices"][:16]
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=val._ExactSampler(parents),
                num_workers=0, generator=torch.Generator().manual_seed(0))
            batch = next(iter(loader))
            if batch["entry_row_index"].tolist() != parents or batch["dynamic_entry_parent_row_index"].tolist() != parents:
                raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_PARENT_CHANGED")
            batch = trainer._materialize_dynamic_m5_batch_targets(batch, dataset)
            parity[role] = {"cohort_sha256": cohort["cohort_sha256"], "parent_entry_rows": parents,
                "entry_time_ns": cohort["entry_time_ns"][:16], "source_clock_sha256": cohort["source_clock_sha256"],
                "dynamic_entry_m1_rows": batch["dynamic_entry_m1_row"].tolist()}
            diagnostics = {}
            try:
                parity[role].update(_dynamic_m5_forward_parity(
                    model=model, batch=batch, device=device, diagnostics=diagnostics))
            except RuntimeError as error:
                write(directory / "PARITY_FAILURE.json", {
                    "schema_version": "gx1_dynamic_m5_technical_parity_failure_v1",
                    "error": str(error), "failed_role": role, "cohort": parity[role],
                    "diagnostics": diagnostics, "previous_completed_roles": {
                        name: value for name, value in parity.items() if name != role},
                    "optimizer_updates": 0, "native_training_admitted": False, "test_data_used": False})
                raise
            del batch, loader
    finally:
        for module, training in modes:
            module.training = training
    if (trainer._model_state_sha256(model) != initial_hash or optimizer_ema_snapshot() != parity_training_state
            or _dynamic_m5_technical_tree_sha256(trainer._attended_session_rng_state(device=device)) != parity_rng
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_CHANGED_ONLINE_OR_TRAINING_STATE")
    actual_order = components["prefix_epoch0_parent_order"][:32].tolist()
    if actual_order != scope["plan"]["native_train_parent_rows"]:
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_TRAIN_ROWS_CHANGED")
    def run_segment(name, destination, steps, pointer):
        deadline()
        budget = {**dict(execution_budget), "stop_after_optimizer_steps": steps,
            "expected_active_pointer_sha256": pointer}
        binding = write(directory / (name + "_BUDGET.json"), budget)
        # Check each true child budget/hash through its unchanged source owner.
        launch_owner.require_candidate_execution_budget(Path(binding["path"]), binding["sha256"],
            recipe_path=Path(provenance["recipe_audit_path"]), recipe_sha256=recipe_file_sha256, recipe=recipe)
        try:
            _run_bound_full_train_candidate(components=components, files=files, device=device,
                run_id=recipe["run_id"], dataset_run_id=recipe["dataset_run_id"], output=destination,
                gx1_data=recipe["gx1_data_root"], seed=controls["seed"], learning_rate=controls["learning_rate"],
                weight_decay=controls["weight_decay"], grad_clip_norm=controls["grad_clip_norm"],
                recipe_source_provenance=provenance, execution_budget=budget,
                execution_budget_sha256=binding["sha256"], invocation_started_monotonic=invocation_started)
        except trainer._CandidateExecutionPaused as paused:
            evidence = dict(paused.evidence)
        else:
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_EXPECTED_PAUSE_MISSING")
        if (evidence.get("reason") != "optimizer_step_ceiling" or evidence.get("phase") != "train"
                or any(type(evidence.get(key)) is not int or evidence[key] != wanted
                       for key, wanted in (("epoch_index", 0), ("next_batch_offset", steps), ("global_optimizer_steps", steps)))
                or evidence.get("completed_val_epochs") != 0):
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_CURSOR_INVALID")
        if inactive_hash() != initial_inactive_hash:
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_INACTIVE_WEIGHTS_CHANGED")
        contract = val._read(Path(evidence["session_directory"]) / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME)
        session = trainer._CandidateTrainingSession(out_bundle_dir=destination, contract=contract, read_only=True)
        state = session.load_checkpoint()
        if (state["global_optimizer_steps"] != steps or state["next_batch_offset"] != steps
                or state["phase"] != "train" or state["epoch_index"] != 0 or state["complete"] is not False
                or val.canonical_model_state_sha256(state["target_model_state"]) != initial_hash
                or trainer._model_state_sha256(model) != val.canonical_model_state_sha256(state["model_state"])):
            raise RuntimeError("DYNAMIC_M5_TECHNICAL_SAVED_STATE_INVALID")
        fields = ("model_state", "target_model_state", "optimizer_state", "weight_ema_state", "lr_scheduler_state",
                  "rng_state", "epoch_order", "training_progress")
        hashes = {key: _dynamic_m5_technical_tree_sha256(state[key]) for key in fields}
        del state
        return evidence, hashes
    # Fresh restoration resets all state before each independent trajectory.
    fresh()
    reference_pause, reference_hashes = run_segment("UNINTERRUPTED", directory / "REFERENCE_BUNDLE", 2, None)
    fresh()
    first_pause, _ = run_segment("BEFORE_DURABLE_RELOAD", output, 1, None)
    # Deliberately reset the live state: the next call MUST restore the actual
    # saved step1 optimizer/model/EMA/scheduler/RNG via the existing coordinator.
    fresh()
    final_pause, resumed_hashes = run_segment("AFTER_DURABLE_RELOAD", output,
        DYNAMIC_M5_TECHNICAL_CURSOR_CEILING, first_pause["active_pointer_sha256"])
    if reference_hashes != resumed_hashes:
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_RESUME_NOT_EXACT")
    updates = reference_pause["global_optimizer_steps"] + first_pause["global_optimizer_steps"] + (
        final_pause["global_optimizer_steps"] - first_pause["global_optimizer_steps"])
    if updates != 4 or sum(value["model_forwards"] for value in parity.values()) != 4:
        raise RuntimeError("DYNAMIC_M5_TECHNICAL_COMPUTE_COUNT_INVALID")
    result = {"schema_version": "gx1_dynamic_m5_technical_qualification_result_v1",
        "decision": "PASS_DYNAMIC_M5_MECHANICS_PRIVATE_WEIGHTS_NOT_REUSABLE", "plan": scope["plan_binding"],
        "initialization_result": scope["artifacts"]["initialization_result"],
        "measurement_binding_result": scope["artifacts"]["measurement_binding_result"],
        "initial_measurement_result": scope["artifacts"]["initial_measurement_result"],
        "initial_measurement_audit": scope["artifacts"]["initial_measurement_audit"],
        "initial_model_state_sha256": initial_hash, "entry_execution_profile": recipe["entry_execution_profile"],
        "source_bindings_sha256": recipe["source_bindings_sha256"], "files": recipe["files"],
        "entry_observed_market": components["entry_observed_market"], "parity": parity,
        "native_train_parent_rows": actual_order, "technical_optimizer_updates": updates,
        "control_optimizer_updates": 0, "parity_model_forwards": 4,
        "resume_comparison_state_sha256": resumed_hashes, "resume_exact": True,
        "exit_model_forward_count": 0, "teacher_refreshed": False,
        "private_weights_reusable": False, "private_states_retained_for_audit_only": True,
        "full_epoch_training_allowed": False, "native_launch_authorized": False, "test_data_used": False}
    final_pause["entry_dynamic_m5_technical_qualification"] = write(directory / "RESULT.json", result)
    receipt = trainer._write_candidate_execution_pause_receipt(final_pause, out_bundle_dir=output,
        gx1_data_override=recipe["gx1_data_root"])
    return {"decision": "PAUSED_RESUMABLE", "pause": final_pause,
        "pause_receipt": {"path": str(receipt), "sha256": val.file_sha256(receipt)},
        "recipe_file_sha256": recipe_file_sha256, "resume_state": _native_resume_state(components=components, output=output),
        "bundle_written": False, "test_data_used": False}


def _require_dynamic_m5_measurement_source(*, model, dataset, cohort, context):
    """Shared actual Dataset/clock/parent join for observations and parity."""
    import hashlib
    import pandas as pd
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import require_bounded_val_cohort
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    cohort = require_bounded_val_cohort(cohort)
    binding = getattr(dataset, "_entry_observed_market_binding", {})
    source = context["physical_sources"].get(cohort.get("source_split"), {})
    parents = cohort.get("parent_entry_row_indices")
    if (not isinstance(dataset, val.EntryV10CtxDataset)
            or cohort.get("training_phase") != phase or cohort.get("exit_evaluation") is not False
            or cohort.get("entry_row_indices") != parents or len(parents) != 256
            or getattr(model, "entry_training_phase", None) != phase
            or binding.get("identity") != context["entry_supervision_identity"]
            or binding.get("role") != ("TRAIN" if cohort.get("measurement_role") == "train" else "CONTROL256")
            or binding.get("physical_source") != source
            or Path(dataset.parquet_path) != Path(source["parquet"]["path"])
            or source["parquet"] != cohort["parent_entry_parquet"]
            or source["manifest"] != cohort["parent_entry_manifest"]
            or len(dataset) != source["physical_rows"] or len(dataset) != cohort["population_rows"]
            or any(getattr(dataset, name, None) is not None for name in (
                "_unified_exit_lifecycle", "_unified_exit_lifecycle_v2"))):
        raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_SOURCE_INVALID")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.df["time"], utc=True)).as_unit("ns")
    if (hashlib.sha256(np.asarray(times.asi8, dtype="<i8").tobytes()).hexdigest() != cohort["source_clock_sha256"]
            or times.asi8[parents].tolist() != cohort["entry_time_ns"]):
        raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_CLOCK_INVALID")
    return cohort


def _dynamic_m5_prefix_observations(*, model, dataset, cohort, context, device, deadline):
    """Observe one frozen Entry cohort through the existing one-forward curve evaluator."""
    import hashlib
    import pandas as pd
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import require_bounded_val_cohort
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_v1 import (
        DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY,
        DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_MAE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY,
        DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
    )
    from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT
    phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    cohort = _require_dynamic_m5_measurement_source(model=model, dataset=dataset, cohort=cohort, context=context)
    parents = cohort["parent_entry_row_indices"]
    queries = tuple(trainer._dynamic_m5_model_kwargs(phase)["dynamic_m5_query_minutes"])
    specs = (
        ("dynamic_terminal_net_markout_bps", DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY),
        ("dynamic_mfe_bps", DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY),
        ("dynamic_pre_mfe_mae_upper_bps", DYNAMIC_M5_MAE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY),
        ("dynamic_closed_mfe_bps", DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY),
        ("dynamic_closed_pre_mfe_mae_upper_bps", DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY),
    )
    component_specs = (*specs, ("forecast_pred", "_measurement_forecast_target", "_measurement_forecast_mask"))
    accum = {name: {"target": hashlib.sha256(), "mask": hashlib.sha256(), "rows": 0,
        "counts": np.zeros((trainer.FORECAST_HEAD_DIM,) if index == 5 else (len(queries), 3 if index == 0 else 2), np.int64),
        "squared": np.zeros((trainer.FORECAST_HEAD_DIM,) if index == 5 else (len(queries), 3 if index == 0 else 2), np.float64),
        "absolute": np.zeros((trainer.FORECAST_HEAD_DIM,) if index == 5 else (len(queries), 3 if index == 0 else 2), np.float64)}
        for index, (name, _, _) in enumerate(component_specs)}
    traces, observed_parents, pointers = [], [], []
    forward_count = 0
    def observe(*, batch, out):
        nonlocal forward_count
        if time.monotonic() >= deadline:
            raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_WALL_LIMIT")
        rows = batch["entry_row_index"].tolist()
        start = len(observed_parents)
        if (rows != parents[start:start + len(rows)]
                or batch["dynamic_entry_parent_row_index"].tolist() != rows):
            raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_ORDER_INVALID")
        observed_parents.extend(rows)
        pointers.extend(batch["dynamic_entry_m1_row"].tolist())
        forward_count += 1
        if forward_count > context["max_model_forwards"] or len(observed_parents) > context["max_state_views"]:
            raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_COMPUTE_LIMIT")
        batch = {**batch,
            "_measurement_forecast_target": torch.cat((torch.stack([batch[f"y_forecast_ret_K{h}"]
                for h in trainer.FORECAST_HORIZONS], dim=1), batch[SUPPORT_FORECAST_TARGET_BATCH_KEY]), dim=1),
            "_measurement_forecast_mask": torch.cat((batch[M5_FORECAST_VALID_BATCH_KEY],
                batch[SUPPORT_FORECAST_VALID_BATCH_KEY]), dim=1)}
        values = {}
        for name, target_key, mask_key in component_specs:
            target = batch[target_key].detach().cpu().numpy()
            mask = batch[mask_key].detach().cpu().numpy()
            prediction = out[name].detach().float().cpu().numpy()
            row = accum[name]
            row["target"].update(np.asarray(target, dtype="<f4", order="C").tobytes())
            row["mask"].update(np.asarray(mask, dtype=np.bool_, order="C").tobytes())
            row["rows"] += int(mask.reshape(len(rows), -1).any(axis=1).sum())
            row["counts"] += mask.sum(axis=0)
            error = prediction.astype(np.float64) - target.astype(np.float64)
            row["squared"] += np.where(mask, error * error, 0).sum(axis=0)
            row["absolute"] += np.where(mask, np.abs(error), 0).sum(axis=0)
            values[name] = (prediction, target, mask)
        candidate = out["dynamic_entry_action_quality_bps_candidate"].detach().float().cpu().numpy()
        horizon_mask = out.get("dynamic_entry_quality_best_horizon_mask")
        horizon_values = out.get("dynamic_query_horizon_minutes")
        if (not isinstance(horizon_mask, torch.Tensor) or horizon_mask.dtype != torch.bool
                or tuple(horizon_mask.shape) != (len(rows), len(queries), 2)
                or not isinstance(horizon_values, torch.Tensor) or horizon_values.dtype != torch.int64
                or horizon_values.detach().cpu().tolist() != list(queries)):
            raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_QUERY_INVALID")
        horizon_mask = horizon_mask.detach().cpu().numpy()
        for i, parent in enumerate(rows):
            winners = np.flatnonzero(candidate[i] == candidate[i].max())
            action = ENTRY_FITTED_Q_ACTION_ORDER[int(winners[0])] if len(winners) == 1 else "TIED_FAIL_CLOSED"
            sides = []
            for side in range(2):
                best = np.flatnonzero(horizon_mask[i, :, side])
                if not len(best):
                    raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_HORIZON_EMPTY")
                detail = {"side": ENTRY_FITTED_Q_ACTION_ORDER[side],
                    "predicted_best_horizon_minutes": [queries[h] for h in best]}
                for name, _, _ in specs:
                    prediction, target, mask = values[name]
                    valid = mask[i, best, side]
                    detail[name] = {"observed_cells": int(valid.sum()),
                        "predicted_mean_bps": float(prediction[i, best, side].astype(np.float64).mean()),
                        **({"observed_target_mean_bps": float(target[i, best, side][valid].astype(np.float64).mean())}
                           if valid.any() else {})}
                sides.append(detail)
            traces.append({"entry_row_index": parent, "entry_time_ns": cohort["entry_time_ns"][start + i],
                "predicted_candidate_quality_bps": candidate[i].tolist(), "diagnostic_action": action,
                "diagnostic_abstained": action in ("FLAT", "TIED_FAIL_CLOSED"),
                "side_diagnostics": sides})
    if math.ceil(len(parents) / 16) > context["max_model_forwards"] or len(parents) > context["max_state_views"]:
        raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_COMPUTE_LIMIT")
    source_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=val._ExactSampler(parents),
        num_workers=0, generator=torch.Generator().manual_seed(0))
    class DeadlineLoader:
        def __init__(self):
            self.dataset = dataset
        def __iter__(self):
            for batch in source_loader:
                if time.monotonic() >= deadline:
                    raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_WALL_LIMIT")
                yield batch
    loader = DeadlineLoader()
    # The observer sees the exact materialized targets and outputs of this one evaluator call.
    _, _, _, _, statistics = trainer._validate_dynamic_m5_curves(
        model, loader, device, batch_observer=observe)
    if observed_parents != parents or statistics.get("curve_evaluation_rows") != len(parents):
        raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_INCOMPLETE")
    frozen, errors = {}, {}
    for name, _, _ in component_specs:
        row = accum[name]
        count = int(row["counts"].sum())
        frozen[name] = {"shape": [len(parents), *row["counts"].shape], "target_dtype": "float32",
            "mask_dtype": "bool", "target_storage_sha256": row["target"].hexdigest(),
            "observed_mask_sha256": row["mask"].hexdigest(), "observed_cells": count,
            "observed_rows": row["rows"],
            ("observed_cells_by_forecast_coordinate" if name == "forecast_pred" else "observed_cells_by_horizon_and_side"): row["counts"].tolist()}
        errors[name] = {"observed_cells": count, "observed_rows": row["rows"],
            "sum_squared_error_bps2": float(row["squared"].sum()),
            "sum_absolute_error_bps": float(row["absolute"].sum()),
            **({"mse_bps2": float(row["squared"].sum() / count),
                "mae_bps": float(row["absolute"].sum() / count)} if count else {}),
            ("sum_squared_error_by_forecast_coordinate_bps2" if name == "forecast_pred" else "sum_squared_error_by_horizon_and_side_bps2"): row["squared"].tolist(),
            ("sum_absolute_error_by_forecast_coordinate_bps" if name == "forecast_pred" else "sum_absolute_error_by_horizon_and_side_bps"): row["absolute"].tolist()}
    return {"dynamic_frozen_targets": {"entry_supervision_identity": context["entry_supervision_identity"],
            "parent_entry_rows": parents, "entry_time_ns": cohort["entry_time_ns"], "dynamic_entry_m1_rows": pointers,
            "query_minutes": list(queries), "base_forecast_horizons": list(trainer.FORECAST_HORIZONS),
            "support_forecast_layout": list(MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT), "components": frozen},
        "dynamic_component_errors": errors, "dynamic_entry_observations": traces,
        "dynamic_curve_statistics": statistics,
        "dynamic_candidate_diagnostic_coverage": {"entries": len(traces),
            "unique_action_counts": statistics["dynamic_candidate_unique_diagnostic_counts"],
            "tied_fail_closed_rows": statistics["dynamic_candidate_tied_diagnostic_rows"],
            "abstained_rows": sum(row["diagnostic_abstained"] for row in traces),
            "selected_fraction": sum(not row["diagnostic_abstained"] for row in traces) / len(traces)},
        "entry_model_forward_count": forward_count,
        "exit_model_forward_count": 0, "dynamic_candidate_native_authority_admitted": False,
        "candidate_target_comparison_uses_prediction_selected_horizons_only": True,
        "component_masks_have_independent_denominators": True,
        "realized_cash_or_native_policy_evaluated": False}


def _run_prefix_initial_measurement(*, components, scope, recipe, output, device,
                                  invocation_started, pause_evidence, optimizer_steps=0):
    """Measure initial or fixed final ONLINE state at a durable native boundary."""
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
    final_step = (scope["entry_m5_path_single_epoch"]["optimizer_steps"] if "entry_m5_path_single_epoch" in scope
                  else 512 if scope.get("continuation") else 256)
    if (type(optimizer_steps) is not int or optimizer_steps not in (0, final_step)
            or pause_evidence.get("reason") != "optimizer_step_ceiling" or pause_evidence.get("phase") != "train"
            or any(type(pause_evidence.get(k)) is not int or pause_evidence[k] != value
                   for k, value in (("epoch_index", 0), ("next_batch_offset", optimizer_steps),
                                    ("global_optimizer_steps", optimizer_steps)))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_PAUSE_INVALID")
    context = components["native_val_context"]
    entry_identity = components.get("entry_observed_market")
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    dynamic = isinstance(entry_identity, Mapping) and entry_identity.get("training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if dynamic != ("entry_only_session_binding" in components):
        raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_COMPONENT_MISMATCH")
    if (("entry_observed_market" in recipe or "entry_observed_market_horizons" in recipe)) != (entry_identity is not None):
        raise RuntimeError("NATIVE_PREFIX_OBSERVED_ENTRY_COMPONENT_MISMATCH")
    entry_evidence = {} if entry_identity is None else {"entry_observed_market": entry_identity}
    if dynamic:
        entry_evidence[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY] = components[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY]
    deadline = min(invocation_started + 12000 - 60, time.monotonic() + context["max_wall_seconds"])
    if time.monotonic() >= deadline:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_INSUFFICIENT_WINDOW")
    directory = Path(pause_evidence["session_directory"])
    session = trainer._CandidateTrainingSession(out_bundle_dir=output,
        contract=val._read(directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME), read_only=True)
    if entry_identity is not None:
        bound = session._contract.get("chronological_prefix", {}).get("entry_observed_market", {})
        if (set(bound) != {"train", "control"}
                or any(bound[role]["identity"] != entry_identity for role in bound)
                or any(getattr(components[key], "_entry_observed_market_binding", {}).get("identity") != entry_identity
                       for key in ("train_probe_ds", "val_ds"))):
            raise RuntimeError("NATIVE_PREFIX_OBSERVED_ENTRY_SESSION_MISMATCH")
    if dynamic:
        if (session._contract.get("chronological_prefix") != components["entry_only_session_binding"]
                or context.get("entry_training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
                or context.get("entry_supervision_identity") != entry_identity
                or components.get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY) != session._contract.get("chronological_prefix", {}).get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY)
                or "measurement_state_factories" in components):
            raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_SESSION_MISMATCH")
        context_binding = trainer._native_candidate_val_context_binding(context)
        if session._contract.get("native_bounded_control") != context_binding:
            raise RuntimeError("NATIVE_PREFIX_DYNAMIC_MEASUREMENT_CONTEXT_MISMATCH")
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
            or (optimizer_steps == final_step and (not state["optimizer_state"]["state"]
                                           or state["weight_ema_state"]["steps"] != final_step))
            or val.canonical_model_state_sha256(state["target_model_state"]) != target_hash):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_CHECKPOINT_MISMATCH")
    functions = trainer._prefix_reference_model_functions(
        val._read(_bound_artifact(recipe["chronological_prefix"]["design"])))
    if (session._contract.get("chronological_prefix", {}).get("model_functions") != functions
            or ("model_functions" in scope["initialization"]
                and scope["initialization"]["model_functions"] != functions)
            or (functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
                and scope["initialization"].get("model_functions") != functions)):
        raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_FUNCTION_BINDING_INVALID")
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_prefix_measurement_source_binding
    source_binding = require_prefix_measurement_source_binding(recipe)
    factories = components.get("measurement_state_factories")
    if not dynamic and functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS:
        if (not isinstance(factories, Mapping) or set(factories) != {"train", "control"}
                or any(getattr(factories[role], "source_split", None) != split
                       for role, split in (("train", "train"), ("control", "val")))):
            raise RuntimeError("NATIVE_PREFIX_PHYSICAL_MEASUREMENT_FACTORIES_REQUIRED")
    target = None
    if not dynamic:
        target = trainer._copy_frozen_prefix_reference_model(model, model_functions=functions).to(device)
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
            if not dynamic and functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS and role == "train":
                samples = components["train_ds"]._unified_exit_lifecycle_v2._random_access_train["samples_by_entry"]
                observed = [[int(item["state_index"]) for item in sorted(samples[child], key=lambda item: item["sample_slot"])]
                            for child in cohort["entry_row_indices"]]
                if observed != cohort["sampled_state_indices"]:
                    raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_TRAIN_SAMPLES_CHANGED")
            if dynamic:
                diagnostics = _dynamic_m5_prefix_observations(model=model, dataset=components[dataset_key],
                    cohort=cohort, context=context, device=device, deadline=deadline)
            else:
                _, diagnostics, _ = val._entry_representations(model=model, dataset=components[dataset_key],
                    parent_rows=cohort["parent_entry_row_indices"], device=device, batch_size=16,
                    candidate_target_model=target, candidate_state_factory=(factories[role] if functions == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS else context["state_factory"]),
                    candidate_child_rows=cohort["entry_row_indices"], exit_boundary_model=target,
                    evaluation_cohort=cohort)
                if (len(diagnostics["bounded_entry_observations"]) != 256
                        or len(diagnostics["bounded_exit_anchor_observations"]) != 256
                        or len(diagnostics["bounded_exit_sampled_observations"]) != 1024):
                    raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_INCOMPLETE")
            if time.monotonic() >= deadline:
                raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_WALL_LIMIT")
            if optimizer_steps == final_step:
                baseline = val._read(_bound_artifact(scope["initial_measurement"]["observations"][role]))
                if "entry_baseline" in scope and not dynamic:
                    if role != "train" or recipe.get("chronological_train_only_measurement") is not True:
                        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_TRAIN_ONLY_REQUIRED")
                    baseline["diagnostics"]["bounded_entry_observations"] = scope["entry_baseline"]["entry_observations"]
                if (baseline["cohort"] != cohort or baseline["target_model_state_sha256"] != target_hash
                        or baseline.get("entry_observed_market") != entry_identity):
                    raise RuntimeError("NATIVE_PREFIX_FINAL_COHORT_OR_TEACHER_CHANGED")
                if dynamic:
                    if ("entry_baseline" in scope
                            or baseline["diagnostics"].get("dynamic_frozen_targets") != diagnostics["dynamic_frozen_targets"]):
                        raise RuntimeError("NATIVE_PREFIX_FINAL_FROZEN_TARGET_CHANGED")
                else:
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
                "optimizer_steps": optimizer_steps, "test_data_used": False, **entry_evidence,
                **({"target_model_role": "frozen_constructor_anchor_no_forward", "exit_evaluation": False,
                    "native_launch_authorized": False} if dynamic else {})}
            path = out / (role.upper() + "_OBSERVATION.json")
            trainer._candidate_training_session_atomic_write_json(path, report)
            observations[role] = {"path": str(path), "sha256": val.file_sha256(path)}
            print(json.dumps({"event": "PREFIX_MEASUREMENT_ROLE_COMPLETE", "role": role,
                              "optimizer_steps": optimizer_steps, "entries": 256,
                              **({"entry_forwards": diagnostics["entry_model_forward_count"]} if dynamic else {"sampled_states": 1024})}), flush=True)
    finally:
        for module, training in modes:
            module.training = training
        trainer._restore_attended_session_rng_state(rng, device=device)
        if val.file_sha256(session._active_path) != before or trainer._model_state_sha256(model) != expected:
            raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_CHANGED_STATE")
    result = {"schema_version": ("gx1_native_prefix_initial_measurement_v1" if optimizer_steps == 0
                                else "gx1_native_prefix_final_online_measurement_v1"),
        "decision": ("FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED" if optimizer_steps == 0
                     else f"FIXED{optimizer_steps}_FINAL_ONLINE_MEASURED_PAIRED_LEARNING_REVIEW_REQUIRED"),
        "initialization_result": scope["artifacts"]["initialization_result"],
        "measurement_binding_result": scope["artifacts"]["measurement_binding_result"],
        "training_pointer_sha256": before, "model_state_sha256": expected,
        "target_model_state_sha256": target_hash, "observations": observations,
        "model_functions": dict(functions), **source_binding, **entry_evidence,
        "measurement_roles": list(observations),
        "optimizer_steps": optimizer_steps, "teacher_refreshed": False, "economic_rollout": False,
        "test_data_used": False, "elapsed_native_seconds": time.monotonic() - invocation_started,
        **({"target_model_role": "frozen_constructor_anchor_no_forward", "exit_evaluation": False,
            "native_launch_authorized": False} if dynamic else {})}
    if optimizer_steps == final_step:
        result.update(initial_measurement=scope["artifacts"]["initial_measurement_result"],
                      initial_measurement_audit=scope["artifacts"]["initial_measurement_audit"],
                      selected_model_variant="ONLINE", frozen_targets_exactly_preserved=True)
        if "entry_baseline_result" in scope:
            result["derived_entry_target_baseline"] = scope["entry_baseline_result"]
    path = out / "RESULT.json"
    trainer._candidate_training_session_atomic_write_json(path, result)
    return {"path": str(path), "sha256": val.file_sha256(path)}



def _capture_frozen_entry_representations(*, model, dataset, frame, state_factory,
                                          scope, binding, device):
    """Observe the existing head input without replacing any model output."""
    head = getattr(model, "head_entry_action_q", None)
    if (not isinstance(head, torch.nn.Linear) or head.in_features != 128 or head.out_features != 3
            or model.training or any(p.requires_grad for p in model.parameters())):
        raise RuntimeError("FROZEN_ENTRY_PROBE_MODEL_NOT_FROZEN")
    hidden = []
    def capture(_module, args):
        value = args[0]
        if (value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 128
                or not bool(torch.isfinite(value).all().item())):
            raise RuntimeError("FROZEN_ENTRY_PROBE_HIDDEN_INVALID")
        hidden.append(value.detach().cpu().clone())
        # Returning None leaves the actual head input unchanged.
    hook = head.register_forward_pre_hook(capture)
    try:
        tokens, diagnostics, q = val._entry_representations(
            model=model, dataset=dataset,
            parent_rows=frame["parent_entry_row_index"].astype("int64").tolist(),
            device=device, batch_size=16, evaluation_cohort=scope["cohort"])
    finally:
        hook.remove()
    x = torch.cat(hidden, dim=0) if hidden else torch.empty((0,128))
    probe = scope["entry_selector_probe"]
    expected_q = torch.tensor(probe["source_result"]["entry_policy_decisions"]["entry_action_q_bps"], dtype=torch.float32)
    if (len(hidden) != 16 or x.shape != (256,128) or q.shape != (256,3)
            or not torch.equal(q.cpu(), expected_q)):
        raise RuntimeError("FROZEN_ENTRY_PROBE_ORIGINAL_ENTRY_PARITY_FAILED")
    limits = scope["plan"]["val_limits"]
    contract, _ = state_factory.bind_rollout(
        entry_decision_representations=tokens,
        model_state_sha256=binding["model_state_sha256"],
        checkpoint_file_sha256=binding["checkpoint_file_sha256"],
        compute_guard_max_model_forwards=limits["max_model_forwards"],
        compute_guard_max_materialized_state_views=limits["max_state_views"],
        compute_guard_max_wall_seconds=limits["max_wall_seconds"],
        resumable_wall_limit=True, evaluation_cohort=scope["cohort"])
    if contract["contract_sha256"] != probe["source_result"]["contract_sha256"]:
        raise RuntimeError("FROZEN_ENTRY_PROBE_EXIT_CONTEXT_PARITY_FAILED")
    return {"entry_q_joint_hidden":x, "original_entry_q_bps":q.cpu(),
        "original_exit_tokens":tokens.detach().cpu(),
        "parent_rows":scope["cohort"]["parent_entry_row_indices"],
        "child_rows":scope["cohort"]["entry_row_indices"],
        "original_rollout_contract":contract, "entry_forward_count":len(hidden)}, diagnostics


def _run_frozen_entry_representation_probe(*, components, scope, recipe, device, output,
                                          recipe_file_sha256, invocation_started):
    """Native extraction only; never run Exit or fit/replace the Entry readout."""
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_frozen_train_policy_evaluation
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import bind_frozen_train_policy_checkpoint_v1
    if device.type != "cuda":
        raise RuntimeError("FROZEN_ENTRY_PROBE_CUDA_REQUIRED")
    directory = output.parent / "frozen_entry_representations"
    if directory.exists() or directory.is_symlink():
        raise RuntimeError("FROZEN_ENTRY_PROBE_OUTPUT_EXISTS")
    context = components["native_val_context"]
    frame = context["frame"]
    if (context.get("evaluation_cohort") != scope["cohort"]
            or frame["entry_row_index"].tolist() != scope["cohort"]["entry_row_indices"]
            or frame["parent_entry_row_index"].tolist() != scope["cohort"]["parent_entry_row_indices"]
            or context["state_factory"].source_split != "train"):
        raise RuntimeError("FROZEN_ENTRY_PROBE_CONTEXT_INVALID")
    model = components["model"]
    trainer._require_prefix_online_function(model)
    binding = bind_frozen_train_policy_checkpoint_v1(plan_binding=scope["plan_binding"], model=model)
    before = dict(scope["origin_resume_state"])
    directory.mkdir(parents=True)
    cache, diagnostics = _capture_frozen_entry_representations(model=model,
        dataset=components["train_probe_ds"], frame=frame, state_factory=context["state_factory"],
        scope=scope, binding=binding, device=device)
    after = require_frozen_train_policy_evaluation(recipe)["origin_resume_state"]
    if after != before or trainer._model_state_sha256(model) != binding["model_state_sha256"]:
        raise RuntimeError("FROZEN_ENTRY_PROBE_CHANGED_MODEL_OR_ORIGIN")
    cache_path = directory / "ENTRY_REPRESENTATIONS.pt"
    torch.save(cache, cache_path)
    report = {"schema_version":"gx1_native_frozen_entry_representation_observation_v1",
        "decision":"ORIGINAL_ENTRY_AND_EXIT_CONTEXT_PARITY_PASS_NOT_FIT_AUTHORITY",
        "plan":scope["entry_selector_probe"]["binding"], "checkpoint_binding":binding,
        "cache":{"path":str(cache_path),"sha256":val.file_sha256(cache_path)},
        "original_rollout_contract_sha256":cache["original_rollout_contract"]["contract_sha256"],
        "original_entry_q_exactly_preserved":True,"original_exit_context_exactly_preserved":True,
        "entry_forward_count":cache["entry_forward_count"],"exit_rollout_forwards":0,
        "origin_resume_state":before,"optimizer_steps":0,"new_fits":0,
        "training_enabled":False,"test_data_used":False,
        "entry_route_diagnostics":diagnostics,
        "native_invocation_elapsed_seconds":time.monotonic()-invocation_started}
    path = directory / "OBSERVATION.json"
    trainer._candidate_training_session_atomic_write_json(path, report)
    print(json.dumps({"event":"FROZEN_ENTRY_REPRESENTATIONS_COMPLETED","observation":str(path)}),flush=True)
    return {"decision":"PAUSED_RESUMABLE","resume_state":before,
        "observation":{"path":str(path),"sha256":val.file_sha256(path)},
        "recipe_file_sha256":recipe_file_sha256,"bundle_written":False,"test_data_used":False}


def _run_frozen_train_policy_validation(*, components, scope, recipe, device, output,
                                      recipe_file_sha256, invocation_started):
    """Evaluate the exact saved ONLINE512; preserve its source training cursor."""
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_frozen_train_policy_evaluation
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import bind_frozen_train_policy_checkpoint_v1
    if device.type != "cuda":
        raise RuntimeError("FROZEN_TRAIN_POLICY_CUDA_REQUIRED")
    directory = output.parent / "frozen_train_policy"
    if directory.exists() or directory.is_symlink():
        raise RuntimeError("FROZEN_TRAIN_POLICY_OUTPUT_EXISTS")
    context = components["native_val_context"]
    frame = context["frame"]
    if (context.get("evaluation_cohort") != scope["cohort"]
            or frame["entry_row_index"].tolist() != scope["cohort"]["entry_row_indices"]
            or frame["parent_entry_row_index"].tolist() != scope["cohort"]["parent_entry_row_indices"]
            or context["state_factory"].source_split != "train"):
        raise RuntimeError("FROZEN_TRAIN_POLICY_ACTUAL_CONTEXT_INVALID")
    if time.monotonic() - invocation_started + context["max_wall_seconds"] + 60 >= 12000:
        raise RuntimeError("FROZEN_TRAIN_POLICY_INSUFFICIENT_WINDOW")
    model = components["model"]
    # Reuse the existing function check for both parameter-free ONLINE norms.
    trainer._require_prefix_online_function(model)
    binding = bind_frozen_train_policy_checkpoint_v1(plan_binding=scope["plan_binding"], model=model)
    before = dict(scope["origin_resume_state"])
    directory.mkdir(parents=True)
    result = val.evaluate_bound_full_val_v1(
        model=model, entry_dataset=components["train_probe_ds"], frame=frame,
        state_factory=context["state_factory"], checkpoint_binding=binding,
        parent_coordinate_evidence=context["parent_coordinate_evidence"],
        val_sequence_audit=context["val_sequence_audit"], device=device, selected_batch_size=16,
        evaluation_cohort=scope["cohort"], rollout_progress_path=directory/"ROLLOUT_PROGRESS.json",
        result_path=directory/"TRAIN_RESULT.json", max_forwards_this_invocation=context["max_model_forwards"],
        progress_interval_forwards=context["progress_interval_forwards"],
        compute_guard_max_model_forwards=context["max_model_forwards"],
        compute_guard_max_materialized_state_views=context["max_state_views"],
        compute_guard_max_wall_seconds=context["max_wall_seconds"],
        exit_policy_batch_size=context["policy_batch_size"], cpu_pipeline_workers=context["cpu_pipeline_workers"])
    after = require_frozen_train_policy_evaluation(recipe)["origin_resume_state"]
    if after != before or trainer._model_state_sha256(model) != binding["model_state_sha256"]:
        raise RuntimeError("FROZEN_TRAIN_POLICY_CHANGED_MODEL_OR_ORIGIN")
    report = {"schema_version":"gx1_native_frozen_train_policy_observation_v1",
        "decision":"OBSERVATION_REQUIRES_REVIEW","checkpoint_binding":binding,"evaluation_cohort":scope["cohort"],
        "evaluation_decision":result["decision"],"origin_resume_state":before,
        "optimizer_steps":0,"training_enabled":False,"test_data_used":False,
        "native_invocation_elapsed_seconds":time.monotonic()-invocation_started}
    for key,name in (("progress","ROLLOUT_PROGRESS.json"),("result","TRAIN_RESULT.json")):
        path=directory/name
        if path.is_file(): report[key]={"path":str(path),"sha256":val.file_sha256(path)}
    path=directory/"OBSERVATION.json"
    trainer._candidate_training_session_atomic_write_json(path,report)
    print(json.dumps({"event":"FROZEN_TRAIN_POLICY_COMPLETED","decision":result["decision"],"observation":str(path)}),flush=True)
    return {"decision":"PAUSED_RESUMABLE","resume_state":before,
        "observation":{"path":str(path),"sha256":val.file_sha256(path)},
        "recipe_file_sha256":recipe_file_sha256,"bundle_written":False,"test_data_used":False}


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
    schedule = (components["full_population_schedule"] if "entry_only_session_binding" in components else
                components["train_ds"]._unified_exit_lifecycle_v2.random_access_full_population_schedule_v1())
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
