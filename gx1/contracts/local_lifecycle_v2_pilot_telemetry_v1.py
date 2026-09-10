"""Machine JSON telemetry/ETA for the local lifecycle-v2 pilot."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Mapping

SCHEMA = "gx1_local_lifecycle_v2_pilot_telemetry_v1"
STATUS_SCHEMA = "gx1_local_lifecycle_v2_pilot_status_v1"
SAMPLE_SECONDS = 1
HUMAN_CADENCE_SECONDS = 900


class PilotTelemetryError(RuntimeError):
    pass


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PilotTelemetryError(f"{label} invalid")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PilotTelemetryError(f"{label} invalid") from exc
    if not math.isfinite(result):
        raise PilotTelemetryError(f"{label} invalid")
    return result


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PilotTelemetryError("UTC timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PilotTelemetryError("UTC timestamp required") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise PilotTelemetryError("UTC timestamp required")
    return result


def build_telemetry_sample(
    *, observed_utc: str, stage: str, invocation_number: int,
    gpu_uuid: str, power_limit_w: float, power_draw_w: float,
    core_temp_c: float, memory_temp_c: float, memory_used_mib: int,
    utilization_percent: int, progress: Mapping[str, Any] | None,
    previous_sample: Mapping[str, Any] | None,
) -> dict[str, Any]:
    now = _utc(observed_utc)
    if stage not in {"smoke", "epoch1"} or type(invocation_number) is not int or invocation_number < 1:
        raise PilotTelemetryError("sample identity invalid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise PilotTelemetryError("GPU identity invalid")
    limit = _finite(power_limit_w, "power limit")
    draw = _finite(power_draw_w, "power draw")
    core = _finite(core_temp_c, "core temperature")
    memory_temp = _finite(memory_temp_c, "memory temperature")
    if (type(memory_used_mib) is not int or memory_used_mib < 0
            or type(utilization_percent) is not int or not 0 <= utilization_percent <= 100):
        raise PilotTelemetryError("GPU sample invalid")
    breach = limit > 160.0 or draw > 160.0
    phase = None
    completed = total = None
    eta_seconds = None
    if progress is not None:
        if not isinstance(progress, Mapping) or set(progress) != {
            "schema_version", "phase", "completed_units", "total_units", "observed_utc"
        } or progress["schema_version"] != "gx1_local_lifecycle_v2_progress_v1":
            raise PilotTelemetryError("progress schema invalid")
        phase = progress["phase"]
        completed, total = progress["completed_units"], progress["total_units"]
        if (not isinstance(phase, str) or not phase or type(completed) is not int
                or type(total) is not int or not 0 <= completed <= total or total < 1):
            raise PilotTelemetryError("progress values invalid")
        progress_time = _utc(progress["observed_utc"])
        if progress_time > now or (now - progress_time).total_seconds() > 120:
            raise PilotTelemetryError("progress clock stale or future")
        if previous_sample is not None:
            prior_completed = previous_sample.get("completed_units")
            prior_time = _utc(previous_sample.get("observed_utc"))
            elapsed = (now - prior_time).total_seconds()
            if (isinstance(prior_completed, int) and completed > prior_completed
                    and elapsed > 0 and previous_sample.get("phase") == phase):
                eta_seconds = (total - completed) * elapsed / (completed - prior_completed)
    return {
        "schema_version": SCHEMA,
        "observed_utc": observed_utc,
        "stage": stage,
        "invocation_number": invocation_number,
        "gpu_uuid": gpu_uuid,
        "power_limit_w": limit,
        "power_draw_w": draw,
        "core_temp_c": core,
        "memory_temp_c": memory_temp,
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "phase": phase,
        "completed_units": completed,
        "total_units": total,
        "eta_seconds": eta_seconds,
        "safety_decision": "KILL_AND_BLOCK" if breach else "PASS",
    }


def build_status(
    sample: Mapping[str, Any], *, previous_status: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(sample, Mapping) or sample.get("schema_version") != SCHEMA:
        raise PilotTelemetryError("telemetry sample invalid")
    now = _utc(sample.get("observed_utc"))
    previous_phase = None if previous_status is None else previous_status.get("phase")
    previous_decision = None if previous_status is None else previous_status.get("safety_decision")
    meaningful = (
        previous_status is None
        or sample.get("phase") != previous_phase
        or sample.get("safety_decision") != previous_decision
        or sample.get("completed_units") == sample.get("total_units")
    )
    last_human = None if previous_status is None else previous_status.get("last_human_status_utc")
    due = meaningful or last_human is None or (now - _utc(last_human)).total_seconds() >= HUMAN_CADENCE_SECONDS
    return {
        "schema_version": STATUS_SCHEMA,
        "observed_utc": sample["observed_utc"],
        "stage": sample["stage"],
        "invocation_number": sample["invocation_number"],
        "phase": sample["phase"],
        "completed_units": sample["completed_units"],
        "total_units": sample["total_units"],
        "eta_seconds": sample["eta_seconds"],
        "power_limit_w": sample["power_limit_w"],
        "power_draw_w": sample["power_draw_w"],
        "safety_decision": sample["safety_decision"],
        "meaningful_event": meaningful,
        "human_status_due": due,
        "last_human_status_utc": sample["observed_utc"] if due else last_human,
        "human_status_cadence_seconds": HUMAN_CADENCE_SECONDS,
        "local_sample_seconds": SAMPLE_SECONDS,
        "codex_polling_required": False,
    }


def json_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


__all__ = [
    "HUMAN_CADENCE_SECONDS", "PilotTelemetryError", "SAMPLE_SECONDS",
    "build_status", "build_telemetry_sample", "json_line",
]
