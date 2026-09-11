"""Full-year continuation of an exactly witnessed, completed sampler prefix."""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    canonical_sha256, file_sha256,
)
import json


SCHEMA_VERSION = "gx1_unified_exit_full_population_train_session_v1"


def _read(binding: Mapping[str, str], *, verify_files: bool) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_BINDING_INVALID")
    path = Path(binding["path"])
    if (
        not path.is_absolute() or path.resolve() != path or path.is_symlink()
        or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None
        or not path.is_file()
        or (verify_files and file_sha256(path) != binding["sha256"])
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_BINDING_INVALID")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_BINDING_INVALID")
    return value


def build_full_population_train_session(
    *,
    source_repo: Path,
    source_commit: str,
    checkpoint_dir: Path,
    prefix_authority_binding: Mapping[str, str],
    prefix_proof_binding: Mapping[str, str],
) -> dict[str, Any]:
    from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
        require_final_train_checkpoint_authority,
    )
    authority = require_final_train_checkpoint_authority(
        _read(prefix_authority_binding, verify_files=True), verify_files=True
    )
    proof = _read(prefix_proof_binding, verify_files=True)
    pointer = _read(proof["prior_checkpoint_pointer"], verify_files=True)
    selection = _read(authority["gpu_batch_selection"], verify_files=True)
    batch = int(authority["selected_batch_size"])
    count = int(proof["full_epoch_entry_pairs"])
    total = -(-count // batch)
    value = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_FULL_YEAR_CONTINUATION_ELIGIBLE",
        "phase": "epoch1",
        "source_repo": str(source_repo),
        "source_commit": source_commit,
        "predecessor_source_commit": authority["source_commit"],
        "prelaunch": authority["launch_manifest"],
        "prelaunch_manifest_sha256": authority["launch_manifest_sha256"],
        "gpu_batch_selection": authority["gpu_batch_selection"],
        "gpu_batch_selection_artifact_sha256": selection["artifact_sha256"],
        "selected_batch_size": batch,
        "prefix_checkpoint_authority": dict(prefix_authority_binding),
        "prefix_proof": dict(prefix_proof_binding),
        "prefix_checkpoint": proof["prior_checkpoint_pointer"],
        "prefix_epoch_schedule_sha256": pointer["epoch_schedule_sha256"],
        "initial_batch_offset": pointer["next_batch_offset"],
        "initial_global_step": pointer["global_step"],
        "checkpoint_dir": str(checkpoint_dir),
        "full_population_schedule": proof["full_population_schedule"],
        "entry_pairs_per_epoch": count,
        "transition_budget_per_epoch": proof["full_epoch_transitions"],
        "total_batches_per_epoch": total,
        "remaining_optimizer_steps": total - pointer["next_batch_offset"],
        "checkpoint_interval_optimizer_steps": selection["checkpoint_interval_optimizer_steps"],
        "test_data_used": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def require_full_population_train_session(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
        require_final_train_checkpoint_authority,
    )
    fields = {
        "schema_version", "decision", "phase", "source_repo", "source_commit",
        "predecessor_source_commit", "prelaunch", "prelaunch_manifest_sha256",
        "gpu_batch_selection", "gpu_batch_selection_artifact_sha256",
        "selected_batch_size", "prefix_checkpoint_authority", "prefix_proof",
        "prefix_checkpoint", "prefix_epoch_schedule_sha256", "initial_batch_offset",
        "initial_global_step", "checkpoint_dir", "full_population_schedule",
        "entry_pairs_per_epoch", "transition_budget_per_epoch",
        "total_batches_per_epoch", "remaining_optimizer_steps",
        "checkpoint_interval_optimizer_steps", "test_data_used", "manifest_sha256",
    }
    if (
        not isinstance(value, Mapping) or set(value) != fields
        or value["schema_version"] != SCHEMA_VERSION
        or value["decision"] != "PASS_FULL_YEAR_CONTINUATION_ELIGIBLE"
        or value["phase"] != "epoch1" or value["test_data_used"] is not False
        or value["manifest_sha256"] != canonical_sha256({
            k: v for k, v in value.items() if k != "manifest_sha256"
        })
        or re.fullmatch(r"[0-9a-f]{40}", str(value["source_commit"])) is None
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_SESSION_INVALID")
    for name in ("source_repo", "checkpoint_dir"):
        path = Path(value[name])
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_PATH_INVALID")
    for name in (
        "selected_batch_size", "initial_batch_offset", "initial_global_step",
        "entry_pairs_per_epoch", "transition_budget_per_epoch",
        "total_batches_per_epoch", "remaining_optimizer_steps",
        "checkpoint_interval_optimizer_steps",
    ):
        if type(value[name]) is not int or value[name] < 1:
            raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_COUNTER_INVALID")
    authority = require_final_train_checkpoint_authority(
        _read(value["prefix_checkpoint_authority"], verify_files=verify_files),
        verify_files=verify_files,
    )
    proof = _read(value["prefix_proof"], verify_files=verify_files)
    pointer = _read(value["prefix_checkpoint"], verify_files=verify_files)
    selection = _read(value["gpu_batch_selection"], verify_files=verify_files)
    launch = _read(value["prelaunch"], verify_files=verify_files)
    selected = _read(launch["files"]["selected_sampler"], verify_files=verify_files)
    count = selected["selected_sampler_contract"]["entry_pair_population"]
    batch = value["selected_batch_size"]
    schedule = value["full_population_schedule"]
    if (
        proof.get("decision") != "PASS_FULL_YEAR_COVERAGE_AND_COMPLETED_PREFIX"
        or proof.get("proof_sha256") != canonical_sha256({
            k: v for k, v in proof.items() if k != "proof_sha256"
        })
        or proof.get("prefix_transition_and_anchor_bytes_equal") is not True
        or proof.get("prefix_parent_order_equal") is not True
        or proof.get("legacy_sampler_functions_ast_unchanged") is not True
        or proof.get("test_data_used") is not False
        or schedule != proof["full_population_schedule"]
        or schedule.get("schedule_sha256") != canonical_sha256({
            k: v for k, v in schedule.items() if k != "schedule_sha256"
        })
        or schedule.get("epoch_index") != 0
        or schedule.get("global_entry_start") != 0
        or schedule.get("global_entry_stop") != count
        or schedule.get("every_entry_pair_exactly_once") is not True
        or schedule.get("sampler_contract_sha256") != selected["selected_sampler_contract_sha256"]
        or value["prefix_checkpoint"] != authority["final_checkpoint_pointer"]
        or value["prefix_checkpoint"] != proof["prior_checkpoint_pointer"]
        or proof["prior_checkpoint_state"] != {
            "path": pointer["state_path"], "sha256": pointer["state_file_sha256"]
        }
        or value["prefix_epoch_schedule_sha256"] != pointer["epoch_schedule_sha256"]
        or value["initial_batch_offset"] != pointer["next_batch_offset"]
        or value["initial_global_step"] != pointer["global_step"]
        or value["initial_batch_offset"] * batch != proof["prefix_entry_pairs"]
        or batch != pointer["batch_size"]
        or batch != authority["selected_batch_size"]
        or value["predecessor_source_commit"] != authority["source_commit"]
        or value["prelaunch"] != authority["launch_manifest"]
        or value["prelaunch_manifest_sha256"] != authority["launch_manifest_sha256"]
        or value["gpu_batch_selection"] != authority["gpu_batch_selection"]
        or value["gpu_batch_selection_artifact_sha256"] != selection["artifact_sha256"]
        or value["entry_pairs_per_epoch"] != count
        or value["transition_budget_per_epoch"] != count * selected["selected_sampler_contract"]["transitions_per_entry"]
        or value["total_batches_per_epoch"] != -(-count // batch)
        or value["remaining_optimizer_steps"] != value["total_batches_per_epoch"] - value["initial_batch_offset"]
        or value["checkpoint_interval_optimizer_steps"] != selection["checkpoint_interval_optimizer_steps"]
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_PREFIX_BINDING_INVALID")
    prior_dir = Path(value["prefix_checkpoint"]["path"]).parent
    target = Path(value["checkpoint_dir"])
    if target == prior_dir or prior_dir in target.parents:
        raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_PREFIX_OVERWRITE_FORBIDDEN")
    return dict(value)
