"""Fail-closed source contract for the lifecycle-v2 local CUDA smoke."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    _stats_sha256,
    require_input_normalization_contract,
)
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    canonical_sha256,
    require_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)

BOOTSTRAP_BASE_SCHEMA = "gx1_unified_exit_bootstrap_base_normalization_v1"
BOOTSTRAP_COMPOSITE_SCHEMA = "gx1_unified_exit_bootstrap_composite_normalization_v1"
BOOTSTRAP_SOURCE_SCHEMA = "gx1_unified_exit_random_access_bootstrap_source_receipt_v1"
SMOKE_SCHEMA = "gx1_unified_exit_random_access_cuda_smoke_manifest_v1"
_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUIRED_ARTIFACTS = frozenset(
    {
        "random_access_index_root",
        "final_bindings_bundle",
        "candidate_set",
        "child_composite_normalization",
        "bootstrap_composite_normalization",
        "train_closure_authority",
        "val_closure_authority",
        "economics_readiness",
        "train_cost_authority",
        "val_cost_authority",
        "state_view_source",
    }
)
_NORMALIZATION_KEYS = {
    "schema_version",
    "transform",
    "fit_scope",
    "fit_population",
    "fit_start_utc",
    "fit_end_utc",
    "continuous_transform",
    "continuous_inverse",
    "lineage",
    "ctx_cat",
    "temporal_aliases",
    "temporal_aliases_sha256",
    "surfaces",
    "surface_stats_sha256",
    "contract_sha256",
}


def _require_legacy_v7_normalization(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("contract_sha256", None)
    if (
        set(value) != _NORMALIZATION_KEYS
        or value.get("schema_version") != "entry_model_native_input_normalization_v7"
        or value.get("transform")
        != "shared_entry_exit_train_only_median_raw_iqr_asinh_v4"
        or value.get("fit_scope") != "train_only"
        or value.get("fit_population")
        != "unique_physical_train_rows_entry_exit_union_v2"
        or value.get("continuous_transform") != "asinh_affine_invertible_non_saturating"
        or value.get("continuous_inverse") != "x=sinh(z)*scale+center"
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_BASE_INVALID")
    lineage = value.get("lineage")
    surfaces = value.get("surfaces")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(surfaces, Mapping)
        or int(lineage.get("train_row_count", 0)) < 1
        or int(lineage.get("val_fit_row_count", -1)) != 0
        or int(lineage.get("test_fit_row_count", -1)) != 0
        or value.get("fit_start_utc") != lineage.get("train_time_min_utc")
        or value.get("fit_end_utc") != lineage.get("train_time_max_utc")
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_LINEAGE_INVALID")
    observed_hashes: dict[str, str] = {}
    for surface_name, surface in surfaces.items():
        if not isinstance(surface, Mapping):
            raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_SURFACE_INVALID")
        names = list(surface.get("field_names") or [])
        center = np.asarray(surface.get("center"), dtype=np.float32)
        scale = np.asarray(surface.get("scale"), dtype=np.float32)
        binary = np.asarray(surface.get("binary_mask"), dtype=np.uint8)
        categorical = np.asarray(surface.get("categorical_mask"), dtype=np.uint8)
        minimum = np.asarray(surface.get("train_transformed_min"), dtype=np.float64)
        maximum = np.asarray(surface.get("train_transformed_max"), dtype=np.float64)
        expected_shape = (len(names),)
        domains = surface.get("categorical_domains")
        if (
            surface.get("surface") != surface_name
            or int(surface.get("field_count", -1)) != len(names)
            or any(
                array.shape != expected_shape
                for array in (center, scale, binary, categorical, minimum, maximum)
            )
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or not (scale > 0).all()
            or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all()
            or not isinstance(domains, Mapping)
        ):
            raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_SURFACE_INVALID")
        expected_stats = _stats_sha256(
            field_names=names,
            center=center,
            scale=scale,
            binary_mask=binary,
            categorical_mask=categorical,
            categorical_domains=domains,
            train_transformed_min=minimum,
            train_transformed_max=maximum,
        )
        if surface.get("stats_sha256") != expected_stats:
            raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_SURFACE_HASH_INVALID")
        observed_hashes[str(surface_name)] = expected_stats
    if value.get("surface_stats_sha256") != observed_hashes:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_LEGACY_SURFACE_SET_INVALID")
    return dict(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_SHA_INVALID")
    return value


def _binding(value: Any, label: str, *, verify_file: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_BINDING_INVALID")
    path = Path(str(value.get("path", ""))).expanduser()
    digest = _sha(value.get("sha256"), label)
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_PATH_INVALID")
    if verify_file and (
        not path.is_file() or path.is_symlink() or file_sha256(path) != digest
    ):
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_FILE_INVALID")
    return {"path": str(path), "sha256": digest}


def _canonical_payload(
    value: Mapping[str, Any], hash_key: str, label: str
) -> dict[str, Any]:
    data = dict(value)
    claimed = _sha(data.pop(hash_key, None), label)
    if claimed != canonical_sha256(data):
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_PAYLOAD_INVALID")
    data[hash_key] = claimed
    return data


def build_bootstrap_base_normalization(
    *,
    source_bundle_metadata: Mapping[str, Any],
    source_bundle_metadata_path: str,
    source_bundle_metadata_file_sha256: str,
    checkpoint_input_normalization_sha256: str,
    child_base_artifact: Mapping[str, Any],
    pilot_val_start_utc: str,
) -> dict[str, Any]:
    old = source_bundle_metadata.get("input_normalization")
    child = child_base_artifact.get("contract")
    if not isinstance(old, Mapping) or not isinstance(child, Mapping):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BASE_NORMALIZATION_MISSING")
    old_names = {
        str(name): list(surface.get("field_names") or [])
        for name, surface in old.get("surfaces", {}).items()
    }
    child_names = {
        str(name): list(surface.get("field_names") or [])
        for name, surface in child.get("surfaces", {}).items()
    }
    checked_old = _require_legacy_v7_normalization(old)
    checked_child = require_input_normalization_contract(
        child,
        expected_field_names=child_names,
        expected_ctx_cat_names=list(child.get("ctx_cat", {}).get("field_names") or []),
    )
    lineage = checked_old["lineage"]
    if (
        checked_old["contract_sha256"]
        != _sha(checkpoint_input_normalization_sha256, "CHECKPOINT_NORMALIZATION")
        or old_names != child_names
        or checked_old["ctx_cat"]["field_names"]
        != checked_child["ctx_cat"]["field_names"]
        or checked_old["ctx_cat"]["domains"] != checked_child["ctx_cat"]["domains"]
        or checked_old["temporal_aliases"] != checked_child["temporal_aliases"]
        or lineage.get("val_fit_row_count") != 0
        or lineage.get("test_fit_row_count") != 0
        or str(lineage.get("train_time_max_utc", "")) >= str(pilot_val_start_utc)
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BASE_TRANSFER_INVALID")
    proof = source_bundle_metadata.get("input_normalization_fit_population_proof")
    provenance = source_bundle_metadata.get("recipe_source_provenance")
    source_bindings = (
        provenance.get("source_bindings") if isinstance(provenance, Mapping) else None
    )
    owner = (
        source_bindings.get(
            "python:gx1/contracts/entry_model_native_input_normalization_v1.py"
        )
        if isinstance(source_bindings, Mapping)
        else None
    )
    historical_commit = source_bundle_metadata.get("git_commit")
    if (
        not isinstance(proof, Mapping)
        or not isinstance(proof.get("proof_sha256"), str)
        or not isinstance(provenance, Mapping)
        or provenance.get("source_commit") != historical_commit
        or not isinstance(historical_commit, str)
        or _COMMIT.fullmatch(historical_commit) is None
        or not isinstance(owner, Mapping)
        or _SHA.fullmatch(str(owner.get("sha256", ""))) is None
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BASE_PROOF_MISSING")
    base_artifact = {
        "schema_version": "gx1_unified_exit_pilot_base_normalization_v1",
        "decision": "PASS",
        "contract": checked_old,
        "contract_sha256": checked_old["contract_sha256"],
        "population_witness_sha256": _sha(proof["proof_sha256"], "SOURCE_POPULATION"),
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value = {
        "schema_version": BOOTSTRAP_BASE_SCHEMA,
        "decision": "PASS",
        "transfer_mode": "preserve_checkpoint_train_fitted_base_normalization",
        "base_artifact": base_artifact,
        "contract_sha256": checked_old["contract_sha256"],
        "source_bundle_metadata": {
            "path": str(source_bundle_metadata_path),
            "sha256": _sha(source_bundle_metadata_file_sha256, "SOURCE_BUNDLE"),
        },
        "historical_normalization_owner": {
            "source_commit": historical_commit,
            "path": str(owner.get("path")),
            "sha256": str(owner.get("sha256")),
        },
        "feature_schema_matches_child_base": True,
        "child_base_contract_sha256": checked_child["contract_sha256"],
        "pilot_val_start_utc": str(pilot_val_start_utc),
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def require_bootstrap_base_normalization(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _canonical_payload(value, "artifact_sha256", "BOOTSTRAP_BASE")
    base_artifact = data.get("base_artifact")
    contract = (
        base_artifact.get("contract") if isinstance(base_artifact, Mapping) else None
    )
    if not isinstance(contract, Mapping):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_BASE_INVALID")
    checked = _require_legacy_v7_normalization(contract)
    if (
        data.get("schema_version") != BOOTSTRAP_BASE_SCHEMA
        or base_artifact.get("schema_version")
        != "gx1_unified_exit_pilot_base_normalization_v1"
        or base_artifact.get("decision") != "PASS"
        or base_artifact.get("contract_sha256") != checked["contract_sha256"]
        or base_artifact.get("val_fit_rows") != 0
        or base_artifact.get("test_fit_rows") != 0
        or base_artifact.get("test_accessed") is not False
        or data.get("decision") != "PASS"
        or data.get("transfer_mode")
        != "preserve_checkpoint_train_fitted_base_normalization"
        or data.get("contract_sha256") != checked["contract_sha256"]
        or data.get("feature_schema_matches_child_base") is not True
        or data.get("val_fit_rows") != 0
        or data.get("test_fit_rows") != 0
        or data.get("test_accessed") is not False
        or set(data.get("historical_normalization_owner", {}))
        != {"source_commit", "path", "sha256"}
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_BASE_INVALID")
    return dict(value)


def build_bootstrap_composite_normalization(
    *,
    bootstrap_base: Mapping[str, Any],
    base_path: str,
    base_file_sha256: str,
    child_composite: Mapping[str, Any],
) -> dict[str, Any]:
    checked_base = require_bootstrap_base_normalization(bootstrap_base)
    checked_child = require_composite_normalization_binding(child_composite)
    summary = require_lifetime_summary_normalization(
        checked_child["lifetime_summary_normalization"]
    )
    value = {
        "schema_version": BOOTSTRAP_COMPOSITE_SCHEMA,
        "decision": "PASS",
        "fit_scope": "historical_train_only_base_plus_pilot_train_only_summary",
        "base_feature_normalization": {
            "path": str(base_path),
            "file_sha256": _sha(base_file_sha256, "BOOTSTRAP_BASE_FILE"),
            "contract_sha256": checked_base["contract_sha256"],
            "artifact_sha256": checked_base["artifact_sha256"],
            "artifact": checked_base,
        },
        "lifetime_summary_normalization": summary,
        "summary_fit_manifest": dict(checked_child["summary_fit_manifest"]),
        "val_mode": "apply_frozen_train_transforms_only",
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["composite_normalization_sha256"] = canonical_sha256(value)
    return value


def require_bootstrap_composite_normalization(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    data = _canonical_payload(
        value, "composite_normalization_sha256", "BOOTSTRAP_COMPOSITE"
    )
    base_binding = data.get("base_feature_normalization")
    if not isinstance(base_binding, Mapping):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_COMPOSITE_INVALID")
    base = require_bootstrap_base_normalization(base_binding.get("artifact", {}))
    summary = require_lifetime_summary_normalization(
        data.get("lifetime_summary_normalization", {})
    )
    if (
        data.get("schema_version") != BOOTSTRAP_COMPOSITE_SCHEMA
        or data.get("decision") != "PASS"
        or data.get("fit_scope")
        != "historical_train_only_base_plus_pilot_train_only_summary"
        or base_binding.get("contract_sha256") != base["contract_sha256"]
        or base_binding.get("artifact_sha256") != base["artifact_sha256"]
        or data.get("val_mode") != "apply_frozen_train_transforms_only"
        or data.get("val_fit_rows") != 0
        or data.get("test_fit_rows") != 0
        or data.get("test_accessed") is not False
        or summary.get("val_fit_rows") != 0
        or summary.get("test_fit_rows") != 0
        or summary.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_COMPOSITE_INVALID")
    return dict(value)


def build_bootstrap_source_receipt(
    *, checkpoint: Mapping[str, Any], source_commit: str
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_INVALID")
    required = {
        "state_path",
        "state_file_sha256",
        "pointer_path",
        "pointer_file_sha256",
        "session_contract_path",
        "session_contract_file_sha256",
        "session_contract_sha256",
        "checkpoint_index",
        "slot",
        "phase",
        "epoch_index",
        "next_batch_offset",
        "global_optimizer_steps",
        "container_schema_version",
        "container_keyset_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "model_state_keyset_sha256",
        "model_state_key_count",
        "input_normalization_sha256",
        "contains_random_access_v2_state",
    }
    if set(checkpoint) != required or _COMMIT.fullmatch(source_commit) is None:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_INVALID")
    for key in (
        "state_file_sha256",
        "pointer_file_sha256",
        "session_contract_file_sha256",
        "session_contract_sha256",
        "container_keyset_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "model_state_keyset_sha256",
        "input_normalization_sha256",
    ):
        _sha(checkpoint[key], key.upper())
    if (
        checkpoint["container_schema_version"] != "gx1_candidate_training_session_v1"
        or checkpoint["contains_random_access_v2_state"] is not False
        or checkpoint["phase"] != "train"
        or checkpoint["slot"] not in (0, 1)
        or any(
            isinstance(checkpoint[key], bool)
            or not isinstance(checkpoint[key], int)
            or checkpoint[key] < lower
            for key, lower in (
                ("checkpoint_index", 1),
                ("epoch_index", 0),
                ("next_batch_offset", 0),
                ("global_optimizer_steps", 0),
                ("model_state_key_count", 1),
            )
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_INVALID")
    for key in ("state_path", "pointer_path", "session_contract_path"):
        path = Path(str(checkpoint[key]))
        if not path.is_absolute() or path.resolve() != path:
            raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_PATH_INVALID")
    value = {
        "schema_version": BOOTSTRAP_SOURCE_SCHEMA,
        "decision": "PASS_SOURCE_VERIFIED_BOOTSTRAP_NOT_EXECUTED",
        "migration_kind": "one_time_v1_entry_backbone_to_random_access_v2",
        "source_commit": source_commit,
        "checkpoint": dict(checkpoint),
        "architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "architecture_schema_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "base_normalization_policy": "preserve_checkpoint_buffers_and_bind_new_summary",
        "online_and_target_states_migrate_independently": True,
        "optimizer_scheduler_ema_reinitialized_for_v2": True,
        "old_progress_is_not_v2_resume": True,
        "strict_v2_restore_required_after_bootstrap": True,
        "test_accessed": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def require_bootstrap_source_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _canonical_payload(value, "receipt_sha256", "BOOTSTRAP_SOURCE")
    rebuilt = build_bootstrap_source_receipt(
        checkpoint=data.get("checkpoint", {}),
        source_commit=str(data.get("source_commit", "")),
    )
    comparable = dict(data)
    comparable.pop("receipt_sha256")
    if comparable != {k: v for k, v in rebuilt.items() if k != "receipt_sha256"}:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_SOURCE_INVALID")
    return dict(value)


def build_blocked_smoke_manifest(
    *,
    source_repo: str,
    source_commit: str,
    bootstrap_source: Mapping[str, Any],
    bootstrap_base: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    coordinator: Mapping[str, Mapping[str, str]],
    output_root: str,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_SOURCE_COMMIT_INVALID")
    repo = Path(source_repo)
    out = Path(output_root)
    if (
        not repo.is_absolute()
        or repo.resolve() != repo
        or not out.is_absolute()
        or out.resolve() != out
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_PATH_INVALID")
    checked_source = require_bootstrap_source_receipt(bootstrap_source)
    checked_base = require_bootstrap_base_normalization(bootstrap_base)
    if set(artifacts) != _REQUIRED_ARTIFACTS:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_ARTIFACT_SET_INVALID")
    checked_artifacts = {
        name: _binding(binding, name.upper(), verify_file=True)
        for name, binding in artifacts.items()
    }
    required_coordinator = {"contract", "controller", "telemetry", "installer"}
    if set(coordinator) != required_coordinator:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_COORDINATOR_INVALID")
    checked_coordinator = {
        name: _binding(binding, f"COORDINATOR_{name.upper()}", verify_file=True)
        for name, binding in coordinator.items()
    }
    child_composite = require_composite_normalization_binding(
        json.loads(
            Path(checked_artifacts["child_composite_normalization"]["path"]).read_text()
        )
    )
    bootstrap_composite = require_bootstrap_composite_normalization(
        json.loads(
            Path(
                checked_artifacts["bootstrap_composite_normalization"]["path"]
            ).read_text()
        )
    )
    if (
        bootstrap_composite["base_feature_normalization"]["contract_sha256"]
        != checked_base["base_artifact"]["contract_sha256"]
        or bootstrap_composite["lifetime_summary_normalization"]["normalization_sha256"]
        != child_composite["lifetime_summary_normalization"]["normalization_sha256"]
        or checked_source["checkpoint"]["input_normalization_sha256"]
        != checked_base["contract_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_NORMALIZATION_BINDING_INVALID")
    value = {
        "schema_version": SMOKE_SCHEMA,
        "decision": "BLOCKED_PENDING_AUTHORITATIVE_TRAIN_BENCHMARK_SELECTION",
        "source_repo": str(repo),
        "source_commit": source_commit,
        "output_root": str(out),
        "bootstrap_source_receipt_sha256": checked_source["receipt_sha256"],
        "bootstrap_base_normalization_sha256": checked_base["artifact_sha256"],
        "artifacts": checked_artifacts,
        "sampler_selection": {
            "status": "PENDING_AUTHORITATIVE_TRAIN_BENCHMARK_SELECTION_RECEIPT",
            "receipt_path": None,
            "receipt_file_sha256": None,
            "selected_sampler_contract_sha256": None,
        },
        "benchmark_matrix": {
            "precision_policies": ["deterministic_fp32"],
            "batch_sizes": [4, 8],
            "warmup_optimizer_steps_per_arm": 1,
            "measured_optimizer_steps_per_arm": 2,
            "checkpoint_after_each_arm": True,
            "fresh_process_strict_v2_resume_probe_after_each_arm": True,
            "selection_metric": "measured_steps_per_second_subject_to_12gib_vram_and_guard_pass",
            "precision_alternatives_forbidden": ["autocast", "tf32", "compile"],
        },
        "safety": {
            "physical_power_limit_w": 160,
            "maximum_observed_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
            "telemetry_interval_seconds": 1,
            "capped_runner_class": "trainer",
            "capped_memory": "20G",
            "capped_swap": "512M",
        },
        "coordinator": {
            **checked_coordinator,
            "zero_codex_polling": True,
            "human_status_cadence_seconds": 900,
            "reboot_between_cuda_arms": True,
            "install_scheduled_task_authorized": False,
            "reboot_authorized_by_this_manifest": False,
        },
        "resume_policy": {
            "bootstrap_invocation": "v1_source_once_only",
            "all_later_invocations": "strict_random_access_v2_checkpoint_only",
            "old_optimizer_or_progress_resume_forbidden": True,
            "pointer_must_bind_checkpoint_file_sha256": True,
        },
        "launcher_command": None,
        "cuda_execution_authorized": False,
        "test_data_used": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def require_smoke_manifest(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = _canonical_payload(value, "manifest_sha256", "MANIFEST")
    if (
        data.get("schema_version") != SMOKE_SCHEMA
        or data.get("decision")
        != "BLOCKED_PENDING_AUTHORITATIVE_TRAIN_BENCHMARK_SELECTION"
        or data.get("cuda_execution_authorized") is not False
        or data.get("launcher_command") is not None
        or data.get("test_data_used") is not False
        or data.get("sampler_selection")
        != {
            "status": "PENDING_AUTHORITATIVE_TRAIN_BENCHMARK_SELECTION_RECEIPT",
            "receipt_path": None,
            "receipt_file_sha256": None,
            "selected_sampler_contract_sha256": None,
        }
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_MANIFEST_NOT_BLOCKED")
    if set(data.get("artifacts", {})) != _REQUIRED_ARTIFACTS:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_ARTIFACT_SET_INVALID")
    if verify_files:
        for name, binding in data["artifacts"].items():
            _binding(binding, name.upper(), verify_file=True)
        for name in ("contract", "controller", "telemetry", "installer"):
            _binding(
                data["coordinator"][name],
                f"COORDINATOR_{name.upper()}",
                verify_file=True,
            )
    if data.get("benchmark_matrix") != {
        "precision_policies": ["deterministic_fp32"],
        "batch_sizes": [4, 8],
        "warmup_optimizer_steps_per_arm": 1,
        "measured_optimizer_steps_per_arm": 2,
        "checkpoint_after_each_arm": True,
        "fresh_process_strict_v2_resume_probe_after_each_arm": True,
        "selection_metric": "measured_steps_per_second_subject_to_12gib_vram_and_guard_pass",
        "precision_alternatives_forbidden": ["autocast", "tf32", "compile"],
    }:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BENCHMARK_MATRIX_INVALID")
    return dict(value)


__all__ = (
    "BOOTSTRAP_BASE_SCHEMA",
    "BOOTSTRAP_COMPOSITE_SCHEMA",
    "BOOTSTRAP_SOURCE_SCHEMA",
    "SMOKE_SCHEMA",
    "build_blocked_smoke_manifest",
    "build_bootstrap_base_normalization",
    "build_bootstrap_composite_normalization",
    "build_bootstrap_source_receipt",
    "file_sha256",
    "require_bootstrap_base_normalization",
    "require_bootstrap_composite_normalization",
    "require_bootstrap_source_receipt",
    "require_smoke_manifest",
)
