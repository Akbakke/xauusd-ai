"""Fail-closed source contract for the lifecycle-v2 local CUDA smoke."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    require_input_normalization_contract,
)
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)

BOOTSTRAP_BASE_SCHEMA = "gx1_unified_exit_bootstrap_base_normalization_v1"
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
    checked_old = require_input_normalization_contract(
        old,
        expected_field_names=old_names,
        expected_ctx_cat_names=list(old.get("ctx_cat", {}).get("field_names") or []),
    )
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
        or lineage.get("val_fit_row_count") != 0
        or lineage.get("test_fit_row_count") != 0
        or str(lineage.get("train_time_max_utc", "")) >= str(pilot_val_start_utc)
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BASE_TRANSFER_INVALID")
    proof = source_bundle_metadata.get("input_normalization_fit_population_proof")
    if not isinstance(proof, Mapping) or not isinstance(proof.get("proof_sha256"), str):
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
    names = {
        str(name): list(surface.get("field_names") or [])
        for name, surface in contract.get("surfaces", {}).items()
    }
    checked = require_input_normalization_contract(
        contract,
        expected_field_names=names,
        expected_ctx_cat_names=list(
            contract.get("ctx_cat", {}).get("field_names") or []
        ),
    )
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
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_BOOTSTRAP_BASE_INVALID")
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
    bootstrap_composite = require_composite_normalization_binding(
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
    "BOOTSTRAP_SOURCE_SCHEMA",
    "SMOKE_SCHEMA",
    "build_blocked_smoke_manifest",
    "build_bootstrap_base_normalization",
    "build_bootstrap_source_receipt",
    "file_sha256",
    "require_bootstrap_base_normalization",
    "require_bootstrap_source_receipt",
    "require_smoke_manifest",
)
