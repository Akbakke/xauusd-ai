"""Final guarded epoch authority consumed by the full-cohort VAL launcher."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.local_random_access_campaign_v2 import (
    read_bound_json,
    require_plan,
    require_receipt_chain,
)
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import require_selection
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
    require_bootstrap_composite_normalization,
)
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    require_selected_sampler_artifact,
)
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    POINTER_SCHEMA,
    SCHEMA_VERSION as CHECKPOINT_SCHEMA,
    canonical_sha256 as checkpoint_sha256,
)
from gx1.contracts.unified_exit_train_session_manifest_v1 import (
    require_train_session_manifest,
)
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    require_launch_manifest,
)

SCHEMA_VERSION = "gx1_unified_exit_final_train_checkpoint_authority_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _binding(value: Any, *, verify_files: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_BINDING_INVALID")
    path = Path(str(value["path"]))
    sha = str(value["sha256"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or len(sha) != 64
        or any(ch not in "0123456789abcdef" for ch in sha)
        or verify_files
        and (not path.is_file() or path.is_symlink() or file_sha256(path) != sha)
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_BINDING_INVALID")
    return {"path": str(path), "sha256": sha}


def _read(value: Any) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _binding(value, verify_files=True)
    loaded = read_bound_json(Path(binding["path"]), binding["sha256"])
    if not isinstance(loaded, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_BOUND_JSON_INVALID")
    return binding, dict(loaded)


def _load_checkpoint(pointer_binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    pointer_ref, pointer = _read(pointer_binding)
    core = dict(pointer)
    claimed = core.pop("pointer_sha256", None)
    state_path = Path(str(pointer.get("state_path", "")))
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or claimed != checkpoint_sha256(core)
        or not state_path.is_absolute()
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_POINTER_INVALID")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    ema = state.get("weight_ema_state") if isinstance(state, Mapping) else None
    online = state.get("online_model_state") if isinstance(state, Mapping) else None
    target = state.get("target_model_state") if isinstance(state, Mapping) else None
    shadow = ema.get("shadow") if isinstance(ema, Mapping) else None
    parameter_names = ema.get("parameter_names") if isinstance(ema, Mapping) else None
    model_states_valid = (
        isinstance(online, Mapping)
        and isinstance(target, Mapping)
        and set(online) == set(target)
        and canonical_model_state_sha256(online)
        == state.get("online_model_state_sha256")
        and canonical_model_state_sha256(target)
        == state.get("target_model_state_sha256")
    )
    ema_valid = (
        isinstance(ema, Mapping)
        and set(ema) == {"decay", "steps", "parameter_names", "shadow"}
        and isinstance(parameter_names, list)
        and parameter_names == sorted(set(parameter_names))
        and bool(parameter_names)
        and isinstance(shadow, Mapping)
        and isinstance(online, Mapping)
        and set(shadow) == set(online)
        and set(parameter_names).issubset(online)
        and all(
            isinstance(shadow[name], torch.Tensor)
            and isinstance(online[name], torch.Tensor)
            and shadow[name].shape == online[name].shape
            and shadow[name].dtype == online[name].dtype
            and (
                not shadow[name].is_floating_point()
                or bool(torch.isfinite(shadow[name]).all().item())
            )
            for name in shadow
        )
    )
    if (
        not isinstance(state, Mapping)
        or not model_states_valid
        or not ema_valid
        or state.get("schema_version") != CHECKPOINT_SCHEMA
        or state.get("global_step") != pointer.get("global_step")
        or state.get("next_batch_offset") != pointer.get("next_batch_offset")
        or state.get("epoch_index") != pointer.get("epoch_index")
        or state.get("batch_size") != pointer.get("batch_size")
        or state.get("epoch_schedule_sha256") != pointer.get("epoch_schedule_sha256")
        or state.get("launch_manifest_sha256") != pointer.get("launch_manifest_sha256")
        or state.get("selected_sampler_artifact_sha256")
        != pointer.get("selected_sampler_artifact_sha256")
        or not isinstance(ema, Mapping)
        or ema.get("steps") != state.get("global_step")
        or state.get("test_data_used") is not False
        or state.get("old_v1_progress_reused") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_STATE_INVALID")
    state_ref = {"path": str(state_path), "sha256": file_sha256(state_path)}
    return pointer, dict(state), state_ref


def build_final_train_checkpoint_authority(
    *,
    campaign_plan_binding: Mapping[str, Any],
    campaign_receipt_bindings: Sequence[Mapping[str, Any]],
    gpu_selection_binding: Mapping[str, Any],
    train_session_binding: Mapping[str, Any],
) -> dict[str, Any]:
    plan_ref, raw_plan = _read(campaign_plan_binding)
    plan = require_plan(raw_plan, verify_files=True)
    if plan["phase"] != "selected_training":
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_CAMPAIGN_INVALID")
    raw_receipts = []
    receipt_refs = []
    for value in campaign_receipt_bindings:
        ref, raw = _read(value)
        receipt_refs.append(ref)
        raw_receipts.append(raw)
    receipts = require_receipt_chain(plan, raw_receipts, verify_files=True)
    invocations = plan["checked_invocations"]
    if (
        len(receipts) != len(invocations) - 1
        or not receipts
        or receipts[-1]["kind"] != "epoch1_window"
        or receipts[-1]["outcome"] != "COMPLETE"
        or receipts[-1]["guard_decision"] != "PASS"
        or invocations[len(receipts)]["kind"] != "full_val"
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_CAMPAIGN_INCOMPLETE")
    final_receipt = receipts[-1]
    selection_ref, raw_selection = _read(gpu_selection_binding)
    selection = require_selection(raw_selection, verify_files=True)
    session_ref, raw_session = _read(train_session_binding)
    session = require_train_session_manifest(
        raw_session, expected_phase="epoch1", verify_files=True
    )
    if (
        selection_ref != plan["selection_receipt"]
        or selection["artifact_sha256"] != plan["selection_artifact_sha256"]
        or session["gpu_batch_selection"] != selection_ref
        or session["gpu_batch_selection_artifact_sha256"]
        != selection["artifact_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_SELECTION_INVALID")
    launch_ref, raw_launch = _read(session["prelaunch"])
    launch = require_launch_manifest(raw_launch)
    files = launch["files"]
    selected_sampler = require_selected_sampler_artifact(
        read_bound_json(
            Path(files["selected_sampler"]["path"]),
            files["selected_sampler"]["sha256"],
        )
    )
    bootstrap_normalization = require_bootstrap_composite_normalization(
        read_bound_json(
            Path(files["bootstrap_composite_normalization"]["path"]),
            files["bootstrap_composite_normalization"]["sha256"],
        )
    )
    child_normalization = require_composite_normalization_binding(
        read_bound_json(
            Path(files["child_composite_normalization"]["path"]),
            files["child_composite_normalization"]["sha256"],
        )
    )
    expected_base_normalization_sha256 = bootstrap_normalization[
        "base_feature_normalization"
    ]["artifact"]["base_artifact"]["contract"]["contract_sha256"]
    expected_summary_normalization_sha256 = child_normalization[
        "lifetime_summary_normalization"
    ]["normalization_sha256"]
    pointer_ref = final_receipt["checkpoint_pointer_after"]
    pointer, state, state_ref = _load_checkpoint(pointer_ref)
    progress_ref, progress = _read(final_receipt["progress"])
    selected_batch = int(selection["selected_batch_size"])
    total_batches = int(selection["total_batches_per_epoch"])
    if (
        launch_ref != session["prelaunch"]
        or launch["manifest_sha256"] != session["prelaunch_manifest_sha256"]
        or launch["source_commit"] != plan["source_commit"]
        or selection["launch_manifest_sha256"] != launch["manifest_sha256"]
        or pointer_ref != progress["checkpoint_pointer"]
        or progress["outcome"] != "COMPLETE"
        or progress["next_batch_offset"] != total_batches
        or progress["total_batches"] != total_batches
        or progress["completed_units"] != total_batches
        or progress["total_units"] != total_batches
        or progress["global_optimizer_steps"] != total_batches
        or pointer["global_step"] != total_batches
        or pointer["next_batch_offset"] != total_batches
        or pointer["epoch_index"] != 0
        or pointer["batch_size"] != selected_batch
        or pointer["selected_sampler_artifact_sha256"]
        != selected_sampler["artifact_sha256"]
        or state["base_normalization_sha256"]
        != expected_base_normalization_sha256
        or state["summary_normalization_sha256"]
        != expected_summary_normalization_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_PROGRESS_INVALID")
    value = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_FULL_VAL_ELIGIBLE",
        "source_commit": plan["source_commit"],
        "campaign_plan": plan_ref,
        "campaign_plan_sha256": plan["plan_sha256"],
        "campaign_receipts": receipt_refs,
        "terminal_campaign_receipt": receipt_refs[-1],
        "terminal_campaign_receipt_sha256": final_receipt["receipt_sha256"],
        "terminal_invocation_sha256": final_receipt["invocation_sha256"],
        "gpu_batch_selection": selection_ref,
        "gpu_batch_selection_artifact_sha256": selection["artifact_sha256"],
        "train_session": session_ref,
        "train_session_manifest_sha256": session["manifest_sha256"],
        "resume_equivalence": dict(session["resume_equivalence"]),
        "resume_equivalence_artifact_sha256": session["resume_equivalence_artifact_sha256"],
        "launch_manifest": launch_ref,
        "launch_manifest_sha256": launch["manifest_sha256"],
        "final_checkpoint_pointer": pointer_ref,
        "final_checkpoint_pointer_sha256": pointer["pointer_sha256"],
        "final_checkpoint_state": state_ref,
        "final_progress": progress_ref,
        "final_progress_sha256": progress["progress_sha256"],
        "final_guard_log": final_receipt["guard_log"],
        "guard_decision": final_receipt["guard_decision"],
        "signed_guard_telemetry_owner": final_receipt["signed_guard_telemetry_owner"],
        "model_variant_for_val": "weight_ema",
        "selected_batch_size": selected_batch,
        "epoch_index": 0,
        "epoch_complete": True,
        "entry_pair_count": 16384,
        "transition_count": 65536,
        "total_batches": total_batches,
        "global_optimizer_steps": total_batches,
        "epoch_schedule_sha256": pointer["epoch_schedule_sha256"],
        "selected_sampler_artifact_sha256": pointer["selected_sampler_artifact_sha256"],
        "bootstrap_source_receipt_sha256": state["bootstrap_source_receipt_sha256"],
        "base_normalization_sha256": state["base_normalization_sha256"],
        "summary_normalization_sha256": state["summary_normalization_sha256"],
        "random_access_root": dict(files["random_access_root"]),
        "economics_readiness": dict(files["economics_readiness"]),
        "train_cost_authority": dict(files["train_cost_authority"]),
        "test_data_used": False,
    }
    value["authority_sha256"] = canonical_sha256(value)
    return value


def require_final_train_checkpoint_authority(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("authority_sha256", None)
    required = {
        "schema_version", "decision", "source_commit", "campaign_plan",
        "campaign_plan_sha256", "campaign_receipts", "terminal_campaign_receipt",
        "terminal_campaign_receipt_sha256", "terminal_invocation_sha256",
        "gpu_batch_selection", "gpu_batch_selection_artifact_sha256",
        "train_session", "train_session_manifest_sha256", "resume_equivalence",
        "resume_equivalence_artifact_sha256", "launch_manifest",
        "launch_manifest_sha256", "final_checkpoint_pointer",
        "final_checkpoint_pointer_sha256", "final_checkpoint_state", "final_progress",
        "final_progress_sha256", "final_guard_log", "guard_decision",
        "signed_guard_telemetry_owner", "model_variant_for_val",
        "selected_batch_size", "epoch_index", "epoch_complete", "entry_pair_count",
        "transition_count", "total_batches", "global_optimizer_steps",
        "epoch_schedule_sha256", "selected_sampler_artifact_sha256",
        "bootstrap_source_receipt_sha256", "base_normalization_sha256",
        "summary_normalization_sha256", "random_access_root", "economics_readiness",
        "train_cost_authority", "test_data_used", "authority_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != "PASS_FULL_VAL_ELIGIBLE"
        or value.get("model_variant_for_val") != "weight_ema"
        or value.get("selected_batch_size") not in (4, 8, 16)
        or value.get("epoch_index") != 0
        or value.get("epoch_complete") is not True
        or value.get("entry_pair_count") != 16384
        or value.get("transition_count") != 65536
        or value.get("total_batches")
        != -(-16384 // int(value.get("selected_batch_size", 1)))
        or value.get("global_optimizer_steps") != value.get("total_batches")
        or value.get("guard_decision") != "PASS"
        or value.get("signed_guard_telemetry_owner") != "gx1_guarded_trainer_exec.sh"
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_AUTHORITY_INVALID")
    if verify_files:
        rebuilt = build_final_train_checkpoint_authority(
            campaign_plan_binding=value["campaign_plan"],
            campaign_receipt_bindings=value["campaign_receipts"],
            gpu_selection_binding=value["gpu_batch_selection"],
            train_session_binding=value["train_session"],
        )
        if rebuilt != dict(value):
            raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_AUTHORITY_REBUILD_MISMATCH")
    return dict(value)


def publish_final_train_checkpoint_authority(value: Mapping[str, Any], output: Path) -> None:
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_OUTPUT_INVALID")
    checked = require_final_train_checkpoint_authority(value, verify_files=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(checked, sort_keys=True, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
