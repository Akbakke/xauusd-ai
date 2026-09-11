"""Campaign-bound post-smoke GPU batch selection for random-access Exit v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gx1.contracts.local_random_access_campaign_v2 import (
    read_bound_json,
    require_invocation,
    require_plan,
    require_progress,
    require_receipt,
    require_receipt_chain,
)

ARM_SCHEMA = "gx1_unified_exit_cuda_smoke_arm_final_receipt_v1"
MEASUREMENT_SCHEMA = "gx1_unified_exit_cuda_smoke_measurement_v1"
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
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _require_binding(value: Any, *, verify_files: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BINDING_INVALID")
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
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BINDING_INVALID")
    return {"path": str(path), "sha256": sha}


def _read_binding(value: Any) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _require_binding(value, verify_files=True)
    loaded = read_bound_json(Path(binding["path"]), binding["sha256"])
    if not isinstance(loaded, Mapping):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_BOUND_JSON_INVALID")
    return binding, dict(loaded)


def _require_measurement(value: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("measurement_sha256", None)
    seconds = value.get("measured_train_seconds")
    expected_transitions = 8 * batch_size
    if (
        value.get("schema_version") != MEASUREMENT_SCHEMA
        or value.get("batch_size") != batch_size
        or value.get("warmup_optimizer_steps") != 1
        or value.get("measured_optimizer_steps") != 2
        or value.get("measured_entry_rows") != 2 * batch_size
        or value.get("transitions_per_entry") != 4
        or value.get("measured_transition_count") != expected_transitions
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0.0
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_MEASUREMENT_INVALID")
    return dict(value)


def _load_campaign_evidence(
    *,
    plan_binding: Mapping[str, Any],
    invocation_binding: Mapping[str, Any],
    campaign_receipt_binding: Mapping[str, Any],
    measurement_binding: Mapping[str, Any],
) -> dict[str, Any]:
    plan_ref, raw_plan = _read_binding(plan_binding)
    plan = require_plan(raw_plan, verify_files=True)
    invocation_ref, raw_invocation = _read_binding(invocation_binding)
    invocation = require_invocation(
        raw_invocation,
        source_repo=Path(plan["source_repo"]),
        source_commit=plan["source_commit"],
        verify_files=True,
    )
    if (
        plan["phase"] != "gpu_selection"
        or invocation["kind"] != "smoke_arm"
        or not any(
            item["path"] == invocation_ref["path"]
            and item["sha256"] == invocation_ref["sha256"]
            for item in plan["invocations"]
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_CAMPAIGN_INVALID")
    campaign_ref, raw_campaign_receipt = _read_binding(campaign_receipt_binding)
    campaign_receipt = require_receipt(
        raw_campaign_receipt,
        plan_sha256=plan["plan_sha256"],
        invocation=invocation,
        verify_files=True,
    )
    progress = require_progress(
        read_bound_json(
            Path(campaign_receipt["progress"]["path"]),
            campaign_receipt["progress"]["sha256"],
        ),
        plan_sha256=plan["plan_sha256"],
        invocation=invocation,
        expected_selection_receipt_sha256=None,
        verify_file=True,
    )
    measurement_ref, raw_measurement = _read_binding(measurement_binding)
    measurement = _require_measurement(
        raw_measurement, batch_size=int(invocation["batch_size"])
    )
    execution = read_bound_json(
        Path(invocation["execution_manifest"]["path"]),
        invocation["execution_manifest"]["sha256"],
    )
    prelaunch_ref = _require_binding(
        execution.get("prelaunch_manifest"), verify_files=True
    )
    prelaunch = read_bound_json(
        Path(prelaunch_ref["path"]), prelaunch_ref["sha256"]
    )
    launch_sha = measurement.get("launch_manifest_sha256")
    prelaunch_sha = prelaunch.get("manifest_sha256")
    if (
        not isinstance(launch_sha, str)
        or len(launch_sha) != 64
        or execution.get("prelaunch_manifest_sha256") != prelaunch_sha
        or launch_sha != prelaunch_sha
        or campaign_receipt["selection_receipt_sha256"] is not None
        or campaign_receipt["outcome"] != "COMPLETE"
        or campaign_receipt["guard_decision"] != "PASS"
        or campaign_receipt["signed_guard_telemetry_owner"]
        != "gx1_guarded_trainer_exec.sh"
        or progress["global_optimizer_steps"] != 3
        or progress["next_batch_offset"] != 3
        or progress["outcome"] != "COMPLETE"
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_CAMPAIGN_INVALID")
    return {
        "plan_ref": plan_ref,
        "plan": plan,
        "invocation_ref": invocation_ref,
        "invocation": invocation,
        "campaign_ref": campaign_ref,
        "campaign_receipt": campaign_receipt,
        "measurement_ref": measurement_ref,
        "measurement": measurement,
        "prelaunch_manifest": prelaunch_ref,
        "launch_manifest_sha256": launch_sha,
    }


def build_arm_receipt(
    *,
    campaign_plan_binding: Mapping[str, Any],
    campaign_invocation_binding: Mapping[str, Any],
    campaign_receipt_binding: Mapping[str, Any],
    measurement_binding: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _load_campaign_evidence(
        plan_binding=campaign_plan_binding,
        invocation_binding=campaign_invocation_binding,
        campaign_receipt_binding=campaign_receipt_binding,
        measurement_binding=measurement_binding,
    )
    measurement = evidence["measurement"]
    invocation = evidence["invocation"]
    campaign_receipt = evidence["campaign_receipt"]
    plan = evidence["plan"]
    batch = int(invocation["batch_size"])
    seconds = float(measurement["measured_train_seconds"])
    safety = {
        "physical_power_limit_w": plan["policy"]["physical_power_limit_w"],
        "maximum_actual_draw_w": plan["policy"]["maximum_actual_power_draw_w"],
        "maximum_core_temperature_c": plan["policy"]["maximum_core_temperature_c"],
        "maximum_memory_junction_temperature_c": plan["policy"]["maximum_memory_junction_temperature_c"],
        "maximum_vram_mib": plan["policy"]["maximum_vram_mib"],
    }
    value = {
        "schema_version": ARM_SCHEMA,
        "decision": "PASS",
        "campaign_plan": evidence["plan_ref"],
        "campaign_plan_sha256": plan["plan_sha256"],
        "campaign_invocation": evidence["invocation_ref"],
        "campaign_invocation_sha256": invocation["invocation_sha256"],
        "campaign_receipt": evidence["campaign_ref"],
        "campaign_receipt_sha256": campaign_receipt["receipt_sha256"],
        "measurement": evidence["measurement_ref"],
        "measurement_sha256": measurement["measurement_sha256"],
        "source_commit": plan["source_commit"],
        "prelaunch_manifest": evidence["prelaunch_manifest"],
        "launch_manifest_sha256": evidence["launch_manifest_sha256"],
        "batch_size": batch,
        "precision_policy": "deterministic_fp32",
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": 2,
        "measured_entry_rows": 2 * batch,
        "transitions_per_entry": 4,
        "measured_transition_count": 8 * batch,
        "measured_train_seconds": seconds,
        "measured_transitions_per_second": 8 * batch / seconds,
        "guard_decision": campaign_receipt["guard_decision"],
        "boot_identity_sha256": campaign_receipt["boot"]["identity_sha256"],
        "safety": safety,
        "progress": campaign_receipt["progress"],
        "checkpoint_pointer": campaign_receipt["checkpoint_pointer_after"],
        "guard_log": campaign_receipt["guard_log"],
        "test_data_used": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def require_arm_receipt(value: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("receipt_sha256", None)
    required = {
        "schema_version", "decision", "campaign_plan", "campaign_plan_sha256",
        "campaign_invocation", "campaign_invocation_sha256", "campaign_receipt",
        "campaign_receipt_sha256", "measurement", "measurement_sha256",
        "source_commit", "prelaunch_manifest", "launch_manifest_sha256",
        "batch_size", "precision_policy",
        "warmup_optimizer_steps", "measured_optimizer_steps", "measured_entry_rows",
        "transitions_per_entry", "measured_transition_count", "measured_train_seconds",
        "measured_transitions_per_second", "guard_decision", "boot_identity_sha256",
        "safety", "progress", "checkpoint_pointer", "guard_log", "test_data_used",
        "receipt_sha256",
    }
    batch = value.get("batch_size")
    seconds = value.get("measured_train_seconds")
    expected_transitions = 8 * int(batch) if batch in BATCHES else -1
    if (
        set(value) != required
        or value.get("schema_version") != ARM_SCHEMA
        or value.get("decision") != "PASS"
        or batch not in BATCHES
        or value.get("precision_policy") != "deterministic_fp32"
        or value.get("warmup_optimizer_steps") != 1
        or value.get("measured_optimizer_steps") != 2
        or value.get("measured_entry_rows") != 2 * batch
        or value.get("transitions_per_entry") != 4
        or value.get("measured_transition_count") != expected_transitions
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0.0
        or value.get("measured_transitions_per_second") != expected_transitions / float(seconds)
        or value.get("guard_decision") != "PASS"
        or value.get("safety") != {
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
    prelaunch_ref = _require_binding(
        value.get("prelaunch_manifest"), verify_files=verify_files
    )
    if value.get("prelaunch_manifest") != prelaunch_ref:
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_ARM_RECEIPT_INVALID")
    if verify_files:
        rebuilt = build_arm_receipt(
            campaign_plan_binding=value["campaign_plan"],
            campaign_invocation_binding=value["campaign_invocation"],
            campaign_receipt_binding=value["campaign_receipt"],
            measurement_binding=value["measurement"],
        )
        if rebuilt != dict(value):
            raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_ARM_RECEIPT_REBUILD_MISMATCH")
    return dict(value)


def _load_arm_binding(binding: Mapping[str, Any], *, verify_files: bool) -> tuple[dict[str, str], dict[str, Any]]:
    checked_binding = _require_binding(binding, verify_files=verify_files)
    if not verify_files:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_FILES_REQUIRED")
    receipt = require_arm_receipt(
        json.loads(Path(checked_binding["path"]).read_text()), verify_files=True
    )
    return checked_binding, receipt


def _checked_arm_set(arm_receipt_bindings: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, str], dict[str, Any]]]:
    loaded = [_load_arm_binding(binding, verify_files=True) for binding in arm_receipt_bindings]
    receipts = [receipt for _binding, receipt in loaded]
    if sorted(receipt["batch_size"] for receipt in receipts) != list(BATCHES):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_SET_INVALID")
    if len({receipt["campaign_plan_sha256"] for receipt in receipts}) != 1:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_CAMPAIGN_MISMATCH")
    plan_path = Path(receipts[0]["campaign_plan"]["path"])
    plan = require_plan(
        read_bound_json(plan_path, receipts[0]["campaign_plan"]["sha256"]),
        verify_files=True,
    )
    if any(receipt["campaign_plan"] != receipts[0]["campaign_plan"] for receipt in receipts):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_CAMPAIGN_MISMATCH")
    campaign_receipts = [
        read_bound_json(
            Path(receipt["campaign_receipt"]["path"]),
            receipt["campaign_receipt"]["sha256"],
        )
        for receipt in sorted(receipts, key=lambda item: item["campaign_invocation_sha256"])
    ]
    # Receipt-chain order is invocation-number order, never digest lexical order.
    campaign_receipts.sort(key=lambda item: int(item["invocation_number"]))
    checked_chain = require_receipt_chain(plan, campaign_receipts, verify_files=True)
    if len(checked_chain) != 3 or [item["kind"] for item in checked_chain] != ["smoke_arm"] * 3:
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_CAMPAIGN_CHAIN_INVALID")
    return loaded


def build_selection(arm_receipt_bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    loaded = _checked_arm_set(arm_receipt_bindings)
    checked = [receipt for _binding, receipt in loaded]
    launch_hashes = {receipt["launch_manifest_sha256"] for receipt in checked}
    source_commits = {receipt["source_commit"] for receipt in checked}
    prelaunch_bindings = {
        (receipt["prelaunch_manifest"]["path"], receipt["prelaunch_manifest"]["sha256"])
        for receipt in checked
    }
    if (
        len(launch_hashes) != 1
        or len(source_commits) != 1
        or len(prelaunch_bindings) != 1
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_LAUNCH_MISMATCH")
    winner = min(checked, key=lambda receipt: (-float(receipt["measured_transitions_per_second"]), int(receipt["batch_size"])))
    batch16 = next(receipt for receipt in checked if receipt["batch_size"] == 16)
    reference_replay_seconds = 64.0 * float(batch16["measured_train_seconds"]) / 2.0
    selected_step_seconds = float(winner["measured_train_seconds"]) / 2.0
    checkpoint_interval = max(1, min(256, round(reference_replay_seconds / selected_step_seconds)))
    binding_by_batch = {receipt["batch_size"]: binding for binding, receipt in loaded}
    value = {
        "schema_version": SELECTION_SCHEMA,
        "decision": "PASS",
        "campaign_plan": dict(checked[0]["campaign_plan"]),
        "campaign_plan_sha256": checked[0]["campaign_plan_sha256"],
        "source_commit": source_commits.pop(),
        "prelaunch_manifest": dict(checked[0]["prelaunch_manifest"]),
        "launch_manifest_sha256": launch_hashes.pop(),
        "selection_metric": "highest_guarded_measured_optimizer_transitions_per_second",
        "tie_break": "lower_batch_size",
        "selected_batch_size": int(winner["batch_size"]),
        "selected_measured_transitions_per_second": float(winner["measured_transitions_per_second"]),
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


def require_selection(value: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    prelaunch_ref = _require_binding(
        value.get("prelaunch_manifest"), verify_files=verify_files
    )
    data = dict(value)
    claimed = data.pop("artifact_sha256", None)
    arms = value.get("arm_receipts")
    if (
        value.get("schema_version") != SELECTION_SCHEMA
        or value.get("decision") != "PASS"
        or value.get("selection_metric") != "highest_guarded_measured_optimizer_transitions_per_second"
        or value.get("tie_break") != "lower_batch_size"
        or not isinstance(arms, Mapping)
        or set(arms) != {"4", "8", "16"}
        or value.get("prelaunch_manifest") != prelaunch_ref
        or value.get("selection_uses_outcome_values") is not False
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_INVALID")
    if not verify_files:
        return dict(value)
    bindings = []
    for batch_text, binding in arms.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256", "receipt_sha256"}:
            raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_ARM_BINDING_INVALID")
        bindings.append({"path": binding["path"], "sha256": binding["file_sha256"]})
    rebuilt = build_selection(bindings)
    if rebuilt != dict(value):
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_REBUILD_MISMATCH")
    return dict(value)


__all__ = (
    "ARM_SCHEMA", "MEASUREMENT_SCHEMA", "SELECTION_SCHEMA", "build_arm_receipt",
    "build_selection", "canonical_sha256", "file_sha256", "require_arm_receipt",
    "require_selection",
)
