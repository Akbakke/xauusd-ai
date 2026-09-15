"""Read-only selection of the weight-EMA model from a strict Exit-v2 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gx1.contracts.entry_candidate_checkpoint_policy_v1 import MAX_EPOCHS
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    POINTER_SCHEMA,
    SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
    strict_load_random_access_v2_state,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")

VAL_CHECKPOINT_BINDING_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_val_weight_ema_checkpoint_v1"
)


def _equal_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.detach().cpu(), right.detach().cpu())
    )


CANDIDATE_VAL_BINDING_SCHEMA_VERSION = "gx1_candidate_weight_ema_val_binding_v1"


def bind_candidate_weight_ema_history_v1(
    *, session_contract_path: Path, session_contract_sha256: str,
) -> dict[str, Any] | None:
    """Bind retained EMA updates to the immutable economics transition origin."""
    from gx1.contracts.local_random_access_campaign_v2 import read_bound_json, require_binding

    contract = read_bound_json(session_contract_path, session_contract_sha256)
    provenance = contract.get("recipe_source_provenance")
    if provenance is None:
        return None
    recipe = read_bound_json(Path(provenance["recipe_audit_path"]), provenance["recipe_audit_sha256"])
    origin = recipe.get("candidate_resume_origin")
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
        OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA, OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME,
        OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_SCHEMA, OPTIMIZER_PROCEDURE_ORIGIN_CURSOR,
        OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256, require_optimizer_procedure_origin,
    )
    if isinstance(origin, Mapping) and origin.get("schema_version") == OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA:
        require_optimizer_procedure_origin(origin)
        inherited = bind_candidate_weight_ema_history_v1(
            session_contract_path=Path(origin["contract"]["path"]),
            session_contract_sha256=origin["contract"]["sha256"],
        )
        receipt_path = session_contract_path.parent / OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME
        receipt_binding = require_binding(
            {"path": str(receipt_path), "sha256": file_sha256(receipt_path)},
            label="optimizer procedure EMA history receipt", verify_file=True,
        )
        receipt = read_bound_json(receipt_path, receipt_binding["sha256"])
        preserved = receipt.get("preserved_state_fields")
        cursor = receipt.get("origin_cursor")
        if (inherited is None or inherited.get("optimizer_step_offset") != 19908
                or receipt.get("schema_version") != OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_SCHEMA
                or receipt.get("destination_session_contract_sha256") != session_contract_sha256
                or receipt.get("origin") != origin
                or not isinstance(cursor, Mapping) or set(cursor) != set(OPTIMIZER_PROCEDURE_ORIGIN_CURSOR)
                or any(cursor.get(k) != v or type(cursor.get(k)) is not type(v)
                       for k, v in OPTIMIZER_PROCEDURE_ORIGIN_CURSOR.items())
                or receipt.get("origin_state_sha256") != OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256
                or receipt.get("inherited_weight_ema_history") != inherited
                or type(receipt.get("global_optimizer_steps")) is not int
                or receipt["global_optimizer_steps"] != 5233
                or type(receipt.get("ema_internal_steps")) is not int
                or receipt["ema_internal_steps"] != 5233 + inherited["optimizer_step_offset"]
                or receipt.get("state_preserved") is not True
                or receipt.get("optimizer_procedure_changed") is not True
                or receipt.get("identical_future_trajectory_claimed") is not False
                or not isinstance(preserved, list) or any(type(x) is not str for x in preserved)
                or len(preserved) != len(set(preserved))
                or not {"model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                        "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"} <= set(preserved)):
            raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_EMA_HISTORY_INVALID")
        return {"optimizer_step_offset": inherited["optimizer_step_offset"],
                "transition_receipt": receipt_binding}
    if not isinstance(origin, Mapping) or origin.get("schema_version") != "gx1_candidate_economics_transition_origin_v1":
        return None
    old_contract = require_binding(origin["contract"], label="EMA history origin contract", verify_file=True)
    pointer_binding = require_binding(origin["pointer"], label="EMA history origin pointer", verify_file=True)
    pointer = read_bound_json(Path(pointer_binding["path"]), pointer_binding["sha256"])
    receipt_path = session_contract_path.parent / "CANDIDATE_ECONOMICS_TRANSITION.json"
    receipt_binding = require_binding(
        {"path": str(receipt_path), "sha256": file_sha256(receipt_path)},
        label="EMA history transition receipt", verify_file=True,
    )
    receipt = read_bound_json(receipt_path, receipt_binding["sha256"])
    offset = receipt.get("ema_internal_steps")
    cursor = receipt.get("origin_cursor")
    if (receipt.get("schema_version") != "gx1_candidate_economics_transition_receipt_v1"
            or receipt.get("destination_session_contract_sha256") != session_contract_sha256
            or receipt.get("origin") != origin
            or receipt.get("origin_state_sha256") != pointer.get("state_sha256")
            or pointer.get("session_contract_sha256") != old_contract["sha256"]
            or not isinstance(cursor, Mapping) or set(cursor) != {
                "checkpoint_index", "phase", "epoch_index", "next_batch_offset", "global_optimizer_steps", "complete"}
            or any(pointer.get(key) != value or type(pointer.get(key)) is not type(value) for key, value in cursor.items())
            or type(offset) is not int or offset < 1
            or type(pointer.get("global_optimizer_steps")) is not int
            or offset != pointer["global_optimizer_steps"]
            or type(receipt.get("new_objective_global_optimizer_steps")) is not int
            or receipt["new_objective_global_optimizer_steps"] != 0):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_EMA_HISTORY_INVALID")
    return {"optimizer_step_offset": offset, "transition_receipt": receipt_binding}


def _require_candidate_ema_history_offset(value: Any, *, contract_path: Path) -> int:
    from gx1.contracts.local_random_access_campaign_v2 import require_binding

    if (not isinstance(value, Mapping) or set(value) != {"optimizer_step_offset", "transition_receipt"}
            or type(value["optimizer_step_offset"]) is not int or value["optimizer_step_offset"] < 1):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_EMA_HISTORY_INVALID")
    binding = require_binding(value["transition_receipt"], label="EMA history receipt", verify_file=False)
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME
    if Path(binding["path"]) not in {
            contract_path.parent / "CANDIDATE_ECONOMICS_TRANSITION.json",
            contract_path.parent / OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME}:
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_EMA_HISTORY_INVALID")
    return value["optimizer_step_offset"]


def require_candidate_report_only_val_scope_v1(
    *, session_contract_path: Path, session_contract_sha256: str,
    training_pointer: Mapping[str, Any], verify_files: bool = True,
) -> None:
    """Authorize only the bound calibration's exact saved TRAIN-32 snapshot."""
    from gx1.contracts.local_random_access_campaign_v2 import (
        read_bound_json, require_binding, canonical_sha256 as pointer_file_sha256,
    )

    error = "UNIFIED_EXIT_CANDIDATE_REPORT_ONLY_VAL_SCOPE_INVALID"
    if not isinstance(training_pointer, Mapping) or set(training_pointer) != {"path", "sha256", "value"}:
        raise RuntimeError(error)
    source = require_binding({key: training_pointer[key] for key in ("path", "sha256")},
                             label="report-only TRAIN pointer", verify_file=False)
    pointer = training_pointer["value"]
    required = {"schema_version", "session_contract_sha256", "slot", "checkpoint_index", "state_sha256",
                "phase", "epoch_index", "next_batch_offset", "global_optimizer_steps", "complete"}
    if (Path(source["path"]) != session_contract_path.parent / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
            or not isinstance(pointer, Mapping) or set(pointer) != required
            or pointer_file_sha256(pointer) != source["sha256"]
            or pointer["schema_version"] != "gx1_candidate_training_session_v1"
            or pointer["session_contract_sha256"] != session_contract_sha256
            or pointer["phase"] != "train" or pointer["complete"] is not False
            or any(type(pointer[key]) is not int or pointer[key] != wanted for key, wanted in {
                "epoch_index": 0, "next_batch_offset": 32, "global_optimizer_steps": 32}.items())
            or type(pointer["checkpoint_index"]) is not int or pointer["checkpoint_index"] < 1
            or type(pointer["slot"]) is not int or pointer["slot"] not in (0, 1)
            or not isinstance(pointer["state_sha256"], str) or _SHA256.fullmatch(pointer["state_sha256"]) is None):
        raise RuntimeError(error)
    if verify_files:
        contract = read_bound_json(session_contract_path, session_contract_sha256)
        provenance = contract.get("recipe_source_provenance", {})
        if not isinstance(provenance, Mapping) or not {"recipe_audit_path", "recipe_audit_sha256"} <= set(provenance):
            raise RuntimeError(error)
        recipe = read_bound_json(Path(provenance["recipe_audit_path"]), provenance["recipe_audit_sha256"])
        calibration = recipe.get("native_calibration")
        if (not isinstance(calibration, Mapping) or set(calibration) != {"schema_version", "arm", "report_only_val"}
                or calibration["schema_version"] != "gx1_native_learning_calibration_run_v1"
                or type(calibration["arm"]) is not str or calibration["arm"] not in {"reference", "split"}
                or calibration["report_only_val"] is not True
                or any(recipe.get("val_limits", {}).get(key) != wanted for key, wanted in {
                    "policy_batch_size": 256, "cpu_pipeline_workers": 8,
                    "max_wall_seconds": 10800, "progress_interval_forwards": 64}.items())):
            raise RuntimeError(error)


def require_candidate_weight_ema_val_binding_v1(
    value: Mapping[str, Any], *, verify_files: bool = True,
) -> dict[str, Any]:
    expected = {
        "schema_version", "decision", "model_variant",
        "model_architecture_schema_version", "model_architecture_sha256",
        "model_state_sha256", "target_model_state_sha256",
        "online_model_state_sha256", "checkpoint_path", "checkpoint_file_sha256",
        "session_contract_sha256", "session_contract_path", "session_contract_file_sha256",
        "epoch_index", "global_step",
        "weight_ema_decay", "weight_ema_steps", "parameter_names_sha256",
        "online_buffers_preserved_exactly", "immutable_epoch_snapshot",
        "test_data_used", "binding_sha256",
    }
    if isinstance(value, Mapping) and "weight_ema_history" in value:
        expected.add("weight_ema_history")
    report_fields = {"snapshot_purpose", "report_only", "training_pointer"}
    report_only = isinstance(value, Mapping) and bool(set(value) & report_fields)
    if report_only:
        expected |= report_fields
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_BINDING_INVALID")
    result = dict(value)
    offset = (_require_candidate_ema_history_offset(
        result["weight_ema_history"], contract_path=Path(str(result["session_contract_path"]))
    ) if "weight_ema_history" in result else 0)
    core = {key: item for key, item in result.items() if key != "binding_sha256"}
    sha_keys = [key for key in result if key.endswith("sha256")]
    if (
        result["schema_version"] != CANDIDATE_VAL_BINDING_SCHEMA_VERSION
        or result["decision"] != "PASS" or result["model_variant"] != "weight_ema"
        or result["model_architecture_schema_version"] != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or result["model_architecture_sha256"] != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or any(not isinstance(result[key], str) or _SHA256.fullmatch(result[key]) is None for key in sha_keys)
        or result["binding_sha256"] != canonical_sha256(core)
        or type(result["epoch_index"]) is not int or not 0 <= result["epoch_index"] < MAX_EPOCHS
        or type(result["global_step"]) is not int or result["global_step"] < 1
        or type(result["weight_ema_steps"]) is not int or result["weight_ema_steps"] != result["global_step"] + offset
        or isinstance(result["weight_ema_decay"], bool)
        or not isinstance(result["weight_ema_decay"], (int, float))
        or not 0.0 < result["weight_ema_decay"] < 1.0
        or result["online_buffers_preserved_exactly"] is not True
        or result["immutable_epoch_snapshot"] is not (False if report_only else True)
        or (report_only and (result["report_only"] is not True
            or result["snapshot_purpose"] != "native_calibration_capacity"
            or result["epoch_index"] != 0 or result["global_step"] != 32))
        or result["test_data_used"] is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_BINDING_INVALID")
    path = Path(str(result["checkpoint_path"]))
    if (
        not path.is_absolute() or path.resolve() != path
        or (verify_files and (not path.is_file() or path.is_symlink() or file_sha256(path) != result["checkpoint_file_sha256"]))
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_BINDING_INVALID")
    contract_path = Path(str(result["session_contract_path"]))
    if (
        not contract_path.is_absolute() or contract_path.resolve() != contract_path
        or result["session_contract_file_sha256"] != result["session_contract_sha256"]
        or (verify_files and (not contract_path.is_file() or contract_path.is_symlink() or file_sha256(contract_path) != result["session_contract_file_sha256"]))
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_BINDING_INVALID")
    if report_only:
        if path.name != "calibration_step_0032.pt" or path.parent != contract_path.parent / "validation":
            raise RuntimeError("UNIFIED_EXIT_CANDIDATE_REPORT_ONLY_VAL_SCOPE_INVALID")
        require_candidate_report_only_val_scope_v1(
            session_contract_path=contract_path, session_contract_sha256=result["session_contract_sha256"],
            training_pointer=result["training_pointer"], verify_files=verify_files,
        )
    elif path.name == "calibration_step_0032.pt":
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_REPORT_ONLY_VAL_SCOPE_INVALID")
    if verify_files and result.get("weight_ema_history") != bind_candidate_weight_ema_history_v1(
        session_contract_path=contract_path, session_contract_sha256=result["session_contract_sha256"],
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_EMA_HISTORY_INVALID")
    return result


def bind_candidate_weight_ema_validation_checkpoint_v1(
    *, snapshot: Mapping[str, Any], model: nn.Module,
) -> dict[str, Any]:
    """Bind the actual EMA evaluator to its immutable canonical epoch snapshot."""

    path = Path(str(snapshot["path"]))
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_sha256(path) != snapshot["sha256"]:
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_SNAPSHOT_INVALID")
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {
        "schema_version", "session_contract_sha256", "session_contract_path",
        "session_contract_file_sha256", "epoch_index", "global_optimizer_steps",
        "model_variant", "model_state", "target_model_state", "online_model_state_sha256",
        "model_state_sha256", "target_model_state_sha256", "online_buffers", "parameter_names",
        "weight_ema_decay", "weight_ema_steps", "test_data_used",
    }
    if isinstance(value, Mapping) and "weight_ema_history" in value:
        expected_keys.add("weight_ema_history")
    report_fields = {"snapshot_purpose", "report_only", "training_pointer", "immutable_epoch_snapshot"}
    report_only = isinstance(value, Mapping) and bool(set(value) & report_fields)
    if report_only:
        expected_keys |= report_fields
    if (
        not isinstance(value, Mapping) or set(value) != expected_keys
        or value["schema_version"] != "gx1_candidate_validation_checkpoint_v1"
        or value["session_contract_sha256"] != snapshot["session_contract_sha256"]
        or value["epoch_index"] != snapshot["epoch_index"]
        or value["global_optimizer_steps"] != snapshot["global_optimizer_steps"]
        or value["model_variant"] != "weight_ema" or value["test_data_used"] is not False
        or model.training
        or getattr(model, "unified_exit_random_access_architecture_version", None) != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or value["parameter_names"] != sorted(name for name, _ in model.named_parameters())
        or set(value["model_state"]) != set(model.state_dict())
        or canonical_model_state_sha256(value["model_state"]) != value["model_state_sha256"]
        or canonical_model_state_sha256(model.state_dict()) != value["model_state_sha256"]
        or canonical_model_state_sha256(value["target_model_state"]) != value["target_model_state_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_SNAPSHOT_INVALID")
    buffers = {name: tensor for name, tensor in value["model_state"].items() if name not in value["parameter_names"]}
    if (
        set(buffers) != set(value["online_buffers"])
        or any(not torch.equal(tensor, value["online_buffers"][name]) for name, tensor in buffers.items())
        or bytes(value["model_state"]["unified_exit_random_access_architecture_sha256"].tolist()).hex() != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_CANDIDATE_VAL_SNAPSHOT_BUFFER_INVALID")
    result = {
        "schema_version": CANDIDATE_VAL_BINDING_SCHEMA_VERSION, "decision": "PASS", "model_variant": "weight_ema",
        "model_architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "model_architecture_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "model_state_sha256": value["model_state_sha256"],
        "target_model_state_sha256": value["target_model_state_sha256"],
        "online_model_state_sha256": value["online_model_state_sha256"],
        "checkpoint_path": str(path), "checkpoint_file_sha256": snapshot["sha256"],
        "session_contract_sha256": value["session_contract_sha256"],
        "session_contract_path": value["session_contract_path"],
        "session_contract_file_sha256": value["session_contract_file_sha256"],
        "epoch_index": value["epoch_index"], "global_step": value["global_optimizer_steps"],
        "weight_ema_decay": value["weight_ema_decay"], "weight_ema_steps": value["weight_ema_steps"],
        "parameter_names_sha256": canonical_sha256(value["parameter_names"]),
        "online_buffers_preserved_exactly": True, "immutable_epoch_snapshot": not report_only, "test_data_used": False,
    }
    if "weight_ema_history" in value:
        result["weight_ema_history"] = value["weight_ema_history"]
    if report_only:
        result.update({key: value[key] for key in report_fields})
    result["binding_sha256"] = canonical_sha256(result)
    return require_candidate_weight_ema_val_binding_v1(result)


def require_selected_weight_ema_checkpoint_binding_v1(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate the immutable read-only EMA selection receipt."""
    if isinstance(value, Mapping) and value.get("schema_version") == CANDIDATE_VAL_BINDING_SCHEMA_VERSION:
        return require_candidate_weight_ema_val_binding_v1(value, verify_files=verify_files)

    expected_keys = {
        "schema_version",
        "decision",
        "model_variant",
        "model_architecture_schema_version",
        "model_architecture_sha256",
        "model_state_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_pointer_path",
        "checkpoint_pointer_file_sha256",
        "checkpoint_pointer_sha256",
        "launch_manifest_sha256",
        "selected_sampler_artifact_sha256",
        "bootstrap_source_receipt_sha256",
        "base_normalization_sha256",
        "summary_normalization_sha256",
        "batch_size",
        "epoch_schedule_sha256",
        "epoch_index",
        "next_batch_offset",
        "global_step",
        "weight_ema_decay",
        "weight_ema_steps",
        "weight_ema_parameter_names_sha256",
        "online_buffers_preserved_exactly",
        "rng_mutated",
        "optimizer_or_scheduler_loaded",
        "test_data_used",
        "binding_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    result = dict(value)
    core = dict(result)
    claimed = core.pop("binding_sha256")
    sha_fields = (
        "model_architecture_sha256",
        "model_state_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "checkpoint_file_sha256",
        "checkpoint_pointer_file_sha256",
        "checkpoint_pointer_sha256",
        "launch_manifest_sha256",
        "selected_sampler_artifact_sha256",
        "bootstrap_source_receipt_sha256",
        "base_normalization_sha256",
        "summary_normalization_sha256",
        "epoch_schedule_sha256",
        "weight_ema_parameter_names_sha256",
    )
    if (
        result["schema_version"] != VAL_CHECKPOINT_BINDING_SCHEMA_VERSION
        or result["decision"] != "PASS"
        or result["model_variant"] != "weight_ema"
        or result["model_architecture_schema_version"]
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or result["model_architecture_sha256"] != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or any(
            not isinstance(result[name], str) or _SHA256.fullmatch(result[name]) is None
            for name in sha_fields
        )
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or claimed != canonical_sha256(core)
        or result["batch_size"] not in (4, 8, 16)
        or any(
            type(result[name]) is not int or result[name] < minimum
            for name, minimum in (
                ("epoch_index", 0),
                ("next_batch_offset", 0),
                ("global_step", 1),
                ("weight_ema_steps", 1),
            )
        )
        or result["weight_ema_steps"] != result["global_step"]
        or isinstance(result["weight_ema_decay"], bool)
        or not isinstance(result["weight_ema_decay"], (int, float))
        or not math.isfinite(float(result["weight_ema_decay"]))
        or not 0.0 < float(result["weight_ema_decay"]) < 1.0
        or result["online_buffers_preserved_exactly"] is not True
        or result["rng_mutated"] is not False
        or result["optimizer_or_scheduler_loaded"] is not False
        or result["test_data_used"] is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    for path_name, sha_name in (
        ("checkpoint_path", "checkpoint_file_sha256"),
        ("checkpoint_pointer_path", "checkpoint_pointer_file_sha256"),
    ):
        path = Path(str(result[path_name]))
        if (
            not path.is_absolute()
            or path.resolve() != path
            or (
                verify_files
                and (
                    not path.is_file()
                    or path.is_symlink()
                    or file_sha256(path) != result[sha_name]
                )
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    return result


def load_selected_weight_ema_checkpoint_readonly_v1(
    *,
    pointer_path: Path,
    model: nn.Module,
    expected_checkpoint_pointer_file_sha256: str,
    expected_launch_manifest_sha256: str,
    expected_selected_sampler_artifact_sha256: str,
    expected_bootstrap_source_receipt_sha256: str,
    expected_base_normalization_sha256: str,
    expected_summary_normalization_sha256: str,
    expected_batch_size: int,
    expected_epoch_schedule_sha256: str,
    expected_weight_ema_decay: float,
) -> dict[str, Any]:
    """Strictly load EMA parameters while preserving online buffers and all RNG."""

    path = pointer_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    if file_sha256(path) != expected_checkpoint_pointer_file_sha256:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    pointer = json.loads(path.read_text())
    pointer_core = dict(pointer)
    claimed_pointer = pointer_core.pop("pointer_sha256", None)
    state_path = Path(str(pointer.get("state_path", "")))
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or claimed_pointer != canonical_sha256(pointer_core)
        or pointer.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or pointer.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or pointer.get("batch_size") != expected_batch_size
        or pointer.get("epoch_schedule_sha256") != expected_epoch_schedule_sha256
        or not state_path.is_absolute()
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not torch.equal(torch.get_rng_state(), cpu_rng_before) or (
        torch.cuda.is_available()
        and any(
            not torch.equal(before, after)
            for before, after in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_LOAD_MUTATED_RNG")
    if (
        not isinstance(state, Mapping)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or state.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or state.get("bootstrap_source_receipt_sha256")
        != expected_bootstrap_source_receipt_sha256
        or state.get("base_normalization_sha256") != expected_base_normalization_sha256
        or state.get("summary_normalization_sha256")
        != expected_summary_normalization_sha256
        or state.get("batch_size") != expected_batch_size
        or state.get("epoch_schedule_sha256") != expected_epoch_schedule_sha256
        or state.get("global_step") != pointer.get("global_step")
        or state.get("next_batch_offset") != pointer.get("next_batch_offset")
        or state.get("epoch_index") != pointer.get("epoch_index")
        or state.get("old_v1_progress_reused") is not False
        or state.get("test_data_used") is not False
        or state.get("rng_contract") != "python_numpy_torch_cpu_and_all_cuda_v1"
        or state.get("bootstrap_model_receipts_sha256")
        != canonical_sha256(state.get("bootstrap_model_receipts"))
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_STATE_INVALID")
    online = state.get("online_model_state")
    target = state.get("target_model_state")
    expected = dict(model.state_dict())
    if (
        getattr(model, "unified_exit_random_access_architecture_version", None)
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or not isinstance(online, Mapping)
        or not isinstance(target, Mapping)
        or set(online) != set(expected)
        or set(target) != set(expected)
        or canonical_model_state_sha256(online)
        != state.get("online_model_state_sha256")
        or canonical_model_state_sha256(target)
        != state.get("target_model_state_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    for name, reference in expected.items():
        for observed in (online[name], target[name]):
            if (
                not isinstance(observed, torch.Tensor)
                or observed.shape != reference.shape
                or observed.dtype != reference.dtype
                or (
                    observed.is_floating_point()
                    and not bool(torch.isfinite(observed).all().item())
                )
            ):
                raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    marker = online.get("unified_exit_random_access_architecture_sha256")
    if (
        not isinstance(marker, torch.Tensor)
        or bytes(marker.detach().cpu().tolist()).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    ema = state.get("weight_ema_state")
    if not isinstance(ema, Mapping) or set(ema) != {
        "decay",
        "steps",
        "parameter_names",
        "shadow",
    }:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
    expected_parameter_names = sorted(name for name, _ in model.named_parameters())
    expected_parameter_name_set = set(expected_parameter_names)
    parameter_names_raw = ema["parameter_names"]
    shadow = ema["shadow"]
    if (
        not isinstance(parameter_names_raw, list)
        or parameter_names_raw != expected_parameter_names
        or not all(isinstance(name, str) for name in parameter_names_raw)
        or not isinstance(shadow, Mapping)
        or set(shadow) != set(expected)
        or type(ema["steps"]) is not int
        or ema["steps"] < 1
        or type(state.get("global_step")) is not int
        or ema["steps"] != state["global_step"]
        or isinstance(ema["decay"], bool)
        or not isinstance(ema["decay"], (int, float))
        or not math.isfinite(float(ema["decay"]))
        or float(ema["decay"]) != float(expected_weight_ema_decay)
        or not 0.0 < float(ema["decay"]) < 1.0
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
    selected: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        ema_tensor = shadow[name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or ema_tensor.shape != reference.shape
            or ema_tensor.dtype != reference.dtype
            or (
                ema_tensor.is_floating_point()
                and not bool(torch.isfinite(ema_tensor).all().item())
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
        if name in expected_parameter_name_set:
            selected[name] = ema_tensor
        else:
            if not _equal_tensor(ema_tensor, online[name]):
                raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_BUFFER_DRIFT")
            selected[name] = online[name]
    selected_digest = strict_load_random_access_v2_state(model, selected)
    model.requires_grad_(False)
    model.eval()
    binding = {
        "schema_version": VAL_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "decision": "PASS",
        "model_variant": "weight_ema",
        "model_architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "model_architecture_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "model_state_sha256": selected_digest,
        "online_model_state_sha256": state["online_model_state_sha256"],
        "target_model_state_sha256": state["target_model_state_sha256"],
        "checkpoint_path": str(state_path),
        "checkpoint_file_sha256": pointer["state_file_sha256"],
        "checkpoint_pointer_path": str(path),
        "checkpoint_pointer_file_sha256": file_sha256(path),
        "checkpoint_pointer_sha256": claimed_pointer,
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "selected_sampler_artifact_sha256": (expected_selected_sampler_artifact_sha256),
        "bootstrap_source_receipt_sha256": (expected_bootstrap_source_receipt_sha256),
        "base_normalization_sha256": expected_base_normalization_sha256,
        "summary_normalization_sha256": expected_summary_normalization_sha256,
        "batch_size": int(state["batch_size"]),
        "epoch_schedule_sha256": state["epoch_schedule_sha256"],
        "epoch_index": int(state["epoch_index"]),
        "next_batch_offset": int(state["next_batch_offset"]),
        "global_step": int(state["global_step"]),
        "weight_ema_decay": float(ema["decay"]),
        "weight_ema_steps": int(ema["steps"]),
        "weight_ema_parameter_names_sha256": canonical_sha256(expected_parameter_names),
        "online_buffers_preserved_exactly": True,
        "rng_mutated": False,
        "optimizer_or_scheduler_loaded": False,
        "test_data_used": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return require_selected_weight_ema_checkpoint_binding_v1(binding)


__all__ = (
    "VAL_CHECKPOINT_BINDING_SCHEMA_VERSION",
    "load_selected_weight_ema_checkpoint_readonly_v1",
    "require_selected_weight_ema_checkpoint_binding_v1",
)
