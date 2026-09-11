"""Strict post-GPU-selection session manifest for resume proof and epoch 1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.unified_exit_fixed_step_resume_equivalence_v1 import (
    require_equivalence,
)
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    file_sha256,
    require_selection,
)

SCHEMA_VERSION = "gx1_unified_exit_train_session_manifest_v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _binding(value: Any, verify: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_BINDING_INVALID")
    p = Path(str(value["path"]))
    if not p.is_absolute() or (
        verify
        and (not p.is_file() or p.is_symlink() or file_sha256(p) != value["sha256"])
    ):
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_BINDING_INVALID")
    return dict(value)


def build_train_session_manifest(
    *,
    phase: str,
    source_commit: str,
    prelaunch_binding: Mapping[str, str],
    prelaunch_manifest_sha256: str,
    gpu_selection_binding: Mapping[str, str],
    gpu_selection: Mapping[str, Any],
    resume_equivalence_binding: Mapping[str, str] | None = None,
    resume_equivalence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selection = require_selection(gpu_selection)
    if phase not in {"resume_proof", "epoch1"}:
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_PHASE_INVALID")
    if phase == "resume_proof" and (
        resume_equivalence_binding is not None or resume_equivalence is not None
    ):
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_EQ_PREMATURE")
    eq_sha = None
    if phase == "epoch1":
        if resume_equivalence_binding is None or resume_equivalence is None:
            raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_EQ_REQUIRED")
        eq = require_equivalence(resume_equivalence)
        if (
            eq["gpu_batch_selection_artifact_sha256"] != selection["artifact_sha256"]
            or eq["batch_size"] != selection["selected_batch_size"]
        ):
            raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_EQ_MISMATCH")
        eq_sha = eq["artifact_sha256"]
    value = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_RESUME_PROOF_ELIGIBLE"
        if phase == "resume_proof"
        else "PASS_EPOCH1_ELIGIBLE",
        "phase": phase,
        "source_commit": source_commit,
        "prelaunch": dict(prelaunch_binding),
        "prelaunch_manifest_sha256": prelaunch_manifest_sha256,
        "gpu_batch_selection": dict(gpu_selection_binding),
        "gpu_batch_selection_artifact_sha256": selection["artifact_sha256"],
        "selected_batch_size": selection["selected_batch_size"],
        "entry_pairs_per_epoch": 16384,
        "transition_budget_per_epoch": 65536,
        "total_batches_per_epoch": selection["total_batches_per_epoch"],
        "checkpoint_interval_optimizer_steps": selection[
            "checkpoint_interval_optimizer_steps"
        ],
        "resume_equivalence": dict(resume_equivalence_binding)
        if resume_equivalence_binding
        else None,
        "resume_equivalence_artifact_sha256": eq_sha,
        "test_data_used": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def require_train_session_manifest(
    value: Mapping[str, Any], *, expected_phase: str, verify_files: bool = True
) -> dict[str, Any]:
    from gx1.contracts.unified_exit_full_population_train_session_v1 import (
        SCHEMA_VERSION as FULL_POPULATION_SCHEMA,
        require_full_population_train_session,
    )
    if value.get("schema_version") == FULL_POPULATION_SCHEMA:
        if expected_phase != "epoch1":
            raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_PHASE_INVALID")
        return require_full_population_train_session(value, verify_files=verify_files)
    data = dict(value)
    claimed = data.pop("manifest_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("phase") != expected_phase
        or value.get("decision")
        != (
            "PASS_RESUME_PROOF_ELIGIBLE"
            if expected_phase == "resume_proof"
            else "PASS_EPOCH1_ELIGIBLE"
        )
        or value.get("entry_pairs_per_epoch") != 16384
        or value.get("transition_budget_per_epoch") != 65536
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_INVALID")
    _binding(value.get("prelaunch"), verify_files)
    sel_binding = _binding(value.get("gpu_batch_selection"), verify_files)
    selection = (
        require_selection(json.loads(Path(sel_binding["path"]).read_text()))
        if verify_files
        else None
    )
    if selection is not None and (
        selection["artifact_sha256"] != value.get("gpu_batch_selection_artifact_sha256")
        or selection["selected_batch_size"] != value.get("selected_batch_size")
        or selection["total_batches_per_epoch"] != value.get("total_batches_per_epoch")
        or selection["checkpoint_interval_optimizer_steps"]
        != value.get("checkpoint_interval_optimizer_steps")
    ):
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_SELECTION_INVALID")
    if expected_phase == "resume_proof":
        if (
            value.get("resume_equivalence") is not None
            or value.get("resume_equivalence_artifact_sha256") is not None
        ):
            raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_EQ_PREMATURE")
    else:
        eq_binding = _binding(value.get("resume_equivalence"), verify_files)
        if verify_files:
            eq = require_equivalence(json.loads(Path(eq_binding["path"]).read_text()))
            if eq["artifact_sha256"] != value.get("resume_equivalence_artifact_sha256"):
                raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_EQ_INVALID")
    return dict(value)
