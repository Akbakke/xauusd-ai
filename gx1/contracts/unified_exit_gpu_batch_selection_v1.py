"""Outcome-blind post-smoke GPU batch selection for random-access Exit v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARM_SCHEMA = "gx1_unified_exit_cuda_smoke_arm_final_receipt_v1"
SELECTION_SCHEMA = "gx1_unified_exit_gpu_batch_selection_v1"
BATCHES = (4, 8, 16)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _require_binding(value: Any, *, verify_files: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BINDING_INVALID")
    path = Path(str(value["path"]))
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BINDING_INVALID")
    if verify_files and (
        not path.is_file() or path.is_symlink() or file_sha256(path) != value["sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BINDING_INVALID")
    return {"path": str(path), "sha256": str(value["sha256"])}


def require_arm_receipt(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("receipt_sha256", None)
    batch = value.get("batch_size")
    seconds = value.get("measured_train_seconds")
    measured_entries = value.get("measured_entry_rows")
    expected_transitions = 2 * int(batch) * 4 if batch in BATCHES else -1
    if (
        value.get("schema_version") != ARM_SCHEMA
        or value.get("decision") != "PASS"
        or batch not in BATCHES
        or value.get("precision_policy") != "deterministic_fp32"
        or value.get("warmup_optimizer_steps") != 1
        or value.get("measured_optimizer_steps") != 2
        or measured_entries != 2 * batch
        or value.get("transitions_per_entry") != 4
        or value.get("measured_transition_count") != expected_transitions
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0.0
        or value.get("measured_transitions_per_second")
        != expected_transitions / float(seconds)
        or value.get("guard_decision") != "PASS"
        or value.get("safety")
        != {
            "physical_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
        }
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_ARM_RECEIPT_INVALID")
    for key in ("progress", "checkpoint_pointer", "guard_log"):
        _require_binding(value.get(key), verify_files=verify_files)
    return dict(value)


def _load_arm_binding(
    binding: Mapping[str, Any], *, verify_files: bool
) -> tuple[dict[str, str], dict[str, Any]]:
    checked_binding = _require_binding(binding, verify_files=verify_files)
    if not verify_files:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_FILES_REQUIRED")
    receipt = require_arm_receipt(
        json.loads(Path(checked_binding["path"]).read_text()), verify_files=True
    )
    return checked_binding, receipt


def build_selection(
    arm_receipt_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    loaded = [
        _load_arm_binding(binding, verify_files=True)
        for binding in arm_receipt_bindings
    ]
    checked = [receipt for _binding_value, receipt in loaded]
    if sorted(receipt["batch_size"] for receipt in checked) != list(BATCHES):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_SET_INVALID")
    launch_hashes = {receipt["launch_manifest_sha256"] for receipt in checked}
    if len(launch_hashes) != 1:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_LAUNCH_MISMATCH")
    winner = min(
        checked,
        key=lambda receipt: (
            -float(receipt["measured_transitions_per_second"]),
            int(receipt["batch_size"]),
        ),
    )
    batch16 = next(receipt for receipt in checked if receipt["batch_size"] == 16)
    reference_replay_seconds = 64.0 * float(batch16["measured_train_seconds"]) / 2.0
    selected_step_seconds = float(winner["measured_train_seconds"]) / 2.0
    checkpoint_interval = max(
        1, min(256, round(reference_replay_seconds / selected_step_seconds))
    )
    binding_by_batch = {receipt["batch_size"]: binding for binding, receipt in loaded}
    value = {
        "schema_version": SELECTION_SCHEMA,
        "decision": "PASS",
        "launch_manifest_sha256": launch_hashes.pop(),
        "selection_metric": "highest_guarded_measured_optimizer_transitions_per_second",
        "tie_break": "lower_batch_size",
        "selected_batch_size": int(winner["batch_size"]),
        "selected_measured_transitions_per_second": float(
            winner["measured_transitions_per_second"]
        ),
        "entry_pairs_per_epoch": 16384,
        "transitions_per_entry": 4,
        "transition_budget_per_epoch": 65536,
        "total_batches_per_epoch": -(-16384 // int(winner["batch_size"])),
        "checkpoint_interval_optimizer_steps": checkpoint_interval,
        "checkpoint_interval_derivation": {
            "method": "match_batch16_64_step_measured_replay_time_v1",
            "batch16_reference_replay_seconds": reference_replay_seconds,
            "selected_step_seconds": selected_step_seconds,
            "minimum_steps": 1,
            "maximum_steps": 256,
        },
        "arm_receipts": {
            str(receipt["batch_size"]): {
                "path": binding_by_batch[receipt["batch_size"]]["path"],
                "file_sha256": binding_by_batch[receipt["batch_size"]]["sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
            for receipt in checked
        },
        "selection_uses_outcome_values": False,
        "test_data_used": False,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def require_selection(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("artifact_sha256", None)
    arms = value.get("arm_receipts")
    if (
        value.get("schema_version") != SELECTION_SCHEMA
        or value.get("decision") != "PASS"
        or value.get("selection_metric")
        != "highest_guarded_measured_optimizer_transitions_per_second"
        or value.get("tie_break") != "lower_batch_size"
        or not isinstance(arms, Mapping)
        or set(arms) != {"4", "8", "16"}
        or value.get("selection_uses_outcome_values") is not False
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_INVALID")
    if not verify_files:
        return dict(value)
    receipts = []
    for batch_text, binding in arms.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "file_sha256",
            "receipt_sha256",
        }:
            raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_BINDING_INVALID")
        checked_binding, receipt = _load_arm_binding(
            {"path": binding["path"], "sha256": binding["file_sha256"]},
            verify_files=True,
        )
        if (
            receipt["batch_size"] != int(batch_text)
            or receipt["receipt_sha256"] != binding["receipt_sha256"]
            or checked_binding["sha256"] != binding["file_sha256"]
            or receipt["launch_manifest_sha256"] != value["launch_manifest_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_BINDING_INVALID")
        receipts.append(receipt)
    winner = min(
        receipts,
        key=lambda arm: (
            -float(arm["measured_transitions_per_second"]),
            int(arm["batch_size"]),
        ),
    )
    selected_step_seconds = float(winner["measured_train_seconds"]) / 2.0
    batch16 = next(receipt for receipt in receipts if receipt["batch_size"] == 16)
    reference_replay_seconds = 64.0 * float(batch16["measured_train_seconds"]) / 2.0
    expected_interval = max(
        1, min(256, round(reference_replay_seconds / selected_step_seconds))
    )
    if (
        value.get("selected_batch_size") != winner["batch_size"]
        or value.get("selected_measured_transitions_per_second")
        != winner["measured_transitions_per_second"]
        or value.get("entry_pairs_per_epoch") != 16384
        or value.get("transitions_per_entry") != 4
        or value.get("transition_budget_per_epoch") != 65536
        or value.get("total_batches_per_epoch")
        != -(-16384 // int(winner["batch_size"]))
        or value.get("checkpoint_interval_optimizer_steps") != expected_interval
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_WINNER_INVALID")
    return dict(value)


__all__ = (
    "ARM_SCHEMA",
    "SELECTION_SCHEMA",
    "build_selection",
    "require_arm_receipt",
    "require_selection",
)
