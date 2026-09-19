"""Exact report-only performance evidence from one Hopper host smoke."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gx1_cloud_hopper_smoke_measurement_v1"
DECISION = "PASS_HOST_SMOKE_MEASURED_NOT_TRAINING_AUTHORITY"
PRECISION_POLICY = "deterministic_bf16_hopper"
WARMUP_OPTIMIZER_STEPS = 32
MEASURED_OPTIMIZER_STEPS = 256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BINDING_KEYS = frozenset({"path", "sha256"})
_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "source_commit",
        "run_id",
        "profile",
        "execution_tier",
        "precision_policy",
        "batch_size",
        "smoke_recipe",
        "cloud_host_profile",
        "physical_train_rows",
        "physical_val_rows",
        "sampled_train_rows",
        "sampled_val_rows",
        "warmup_optimizer_steps",
        "measured_optimizer_steps",
        "measured_train_rows",
        "measured_train_seconds",
        "measured_val_rows",
        "measured_val_seconds",
        "preflight_seconds",
        "checkpoint_write_seconds",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "test_accessed",
        "report_only",
        "activation_authority",
        "authority",
        "side_effects",
    }
)
_AUTHORITY = {
    "cloud_purchase": False,
    "training": False,
    "test": False,
    "promotion": False,
    "paper": False,
    "live": False,
}


class CloudTrainingSmokeMeasurementError(RuntimeError):
    """The Hopper smoke measurement is malformed or not source-bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_binding(path: Path, *, label: str) -> dict[str, str]:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise CloudTrainingSmokeMeasurementError(
            f"{label} is not an absolute regular file"
        )
    if any(parent.is_symlink() for parent in raw.parents):
        raise CloudTrainingSmokeMeasurementError(f"{label} path contains a symlink")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise CloudTrainingSmokeMeasurementError(f"{label} path is not canonical")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or frozenset(value) != _BINDING_KEYS:
        raise CloudTrainingSmokeMeasurementError(f"{label} binding keys are not exact")
    path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path, str) or _SHA256_RE.fullmatch(str(sha256 or "")) is None:
        raise CloudTrainingSmokeMeasurementError(f"{label} binding is invalid")
    observed = artifact_binding(Path(path), label=label)
    if observed != dict(value):
        raise CloudTrainingSmokeMeasurementError(f"{label} binding mismatch")
    return observed


def _integer(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CloudTrainingSmokeMeasurementError(f"{label} is invalid")
    return value


def _seconds(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloudTrainingSmokeMeasurementError(f"{label} is invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CloudTrainingSmokeMeasurementError(f"{label} must be positive and finite")
    return result


def require_cloud_training_smoke_measurement(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _KEYS:
        raise CloudTrainingSmokeMeasurementError("smoke measurement keys are not exact")
    payload = dict(value)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("decision") != DECISION
        or payload.get("profile") != "smoke"
        or payload.get("execution_tier") != "canonical"
        or payload.get("precision_policy") != PRECISION_POLICY
        or payload.get("test_accessed") is not False
        or payload.get("report_only") is not True
        or payload.get("activation_authority") is not False
        or payload.get("authority") != _AUTHORITY
        or payload.get("side_effects") != []
    ):
        raise CloudTrainingSmokeMeasurementError("smoke measurement boundary is invalid")
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
        raise CloudTrainingSmokeMeasurementError("source_commit is invalid")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
        raise CloudTrainingSmokeMeasurementError("run_id is invalid")
    payload["smoke_recipe"] = _binding(payload.get("smoke_recipe"), label="smoke recipe")
    payload["cloud_host_profile"] = _binding(
        payload.get("cloud_host_profile"), label="cloud host profile"
    )
    batch_size = _integer(payload.get("batch_size"), label="batch_size")
    if batch_size not in {32, 64}:
        raise CloudTrainingSmokeMeasurementError("batch_size must be 32 or 64")
    physical_train_rows = _integer(
        payload.get("physical_train_rows"), label="physical_train_rows"
    )
    physical_val_rows = _integer(
        payload.get("physical_val_rows"), label="physical_val_rows"
    )
    sampled_train_rows = _integer(
        payload.get("sampled_train_rows"), label="sampled_train_rows"
    )
    sampled_val_rows = _integer(
        payload.get("sampled_val_rows"), label="sampled_val_rows"
    )
    warmup_steps = _integer(
        payload.get("warmup_optimizer_steps"),
        label="warmup_optimizer_steps",
        minimum=0,
    )
    measured_steps = _integer(
        payload.get("measured_optimizer_steps"),
        label="measured_optimizer_steps",
    )
    measured_train_rows = _integer(
        payload.get("measured_train_rows"), label="measured_train_rows"
    )
    measured_val_rows = _integer(
        payload.get("measured_val_rows"), label="measured_val_rows"
    )
    if warmup_steps != WARMUP_OPTIMIZER_STEPS:
        raise CloudTrainingSmokeMeasurementError("warmup optimizer steps are not exact")
    if measured_steps != MEASURED_OPTIMIZER_STEPS:
        raise CloudTrainingSmokeMeasurementError("measured optimizer steps are not exact")
    if sampled_train_rows != (warmup_steps + measured_steps) * batch_size:
        raise CloudTrainingSmokeMeasurementError("sampled TRAIN geometry is not exact")
    if measured_train_rows != measured_steps * batch_size:
        raise CloudTrainingSmokeMeasurementError("measured TRAIN rows are not exact")
    if sampled_val_rows != measured_val_rows:
        raise CloudTrainingSmokeMeasurementError("measured VAL rows are not exact")
    if physical_train_rows <= sampled_train_rows or physical_val_rows <= sampled_val_rows:
        raise CloudTrainingSmokeMeasurementError("smoke populations are not bounded")
    payload["measured_train_seconds"] = _seconds(
        payload.get("measured_train_seconds"), label="measured_train_seconds"
    )
    payload["measured_val_seconds"] = _seconds(
        payload.get("measured_val_seconds"), label="measured_val_seconds"
    )
    payload["preflight_seconds"] = _seconds(
        payload.get("preflight_seconds"), label="preflight_seconds"
    )
    payload["checkpoint_write_seconds"] = _seconds(
        payload.get("checkpoint_write_seconds"), label="checkpoint_write_seconds"
    )
    if _SHA256_RE.fullmatch(str(payload.get("checkpoint_sha256") or "")) is None:
        raise CloudTrainingSmokeMeasurementError("checkpoint SHA-256 is invalid")
    _integer(payload.get("checkpoint_size_bytes"), label="checkpoint_size_bytes")
    return payload


def build_cloud_training_smoke_measurement(
    *,
    source_commit: str,
    run_id: str,
    smoke_recipe_path: Path,
    cloud_host_profile_path: Path,
    batch_size: int,
    physical_train_rows: int,
    physical_val_rows: int,
    sampled_train_rows: int,
    sampled_val_rows: int,
    measured_train_rows: int,
    measured_train_seconds: float,
    measured_val_seconds: float,
    preflight_seconds: float,
    checkpoint_write_seconds: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = artifact_binding(checkpoint_path, label="checkpoint")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "source_commit": source_commit,
        "run_id": run_id,
        "profile": "smoke",
        "execution_tier": "canonical",
        "precision_policy": PRECISION_POLICY,
        "batch_size": int(batch_size),
        "smoke_recipe": artifact_binding(smoke_recipe_path, label="smoke recipe"),
        "cloud_host_profile": artifact_binding(
            cloud_host_profile_path, label="cloud host profile"
        ),
        "physical_train_rows": int(physical_train_rows),
        "physical_val_rows": int(physical_val_rows),
        "sampled_train_rows": int(sampled_train_rows),
        "sampled_val_rows": int(sampled_val_rows),
        "warmup_optimizer_steps": WARMUP_OPTIMIZER_STEPS,
        "measured_optimizer_steps": MEASURED_OPTIMIZER_STEPS,
        "measured_train_rows": int(measured_train_rows),
        "measured_train_seconds": float(measured_train_seconds),
        "measured_val_rows": int(sampled_val_rows),
        "measured_val_seconds": float(measured_val_seconds),
        "preflight_seconds": float(preflight_seconds),
        "checkpoint_write_seconds": float(checkpoint_write_seconds),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_size_bytes": int(Path(checkpoint["path"]).stat().st_size),
        "test_accessed": False,
        "report_only": True,
        "activation_authority": False,
        "authority": dict(_AUTHORITY),
        "side_effects": [],
    }
    return require_cloud_training_smoke_measurement(payload)


__all__ = [
    "CloudTrainingSmokeMeasurementError",
    "DECISION",
    "MEASURED_OPTIMIZER_STEPS",
    "PRECISION_POLICY",
    "SCHEMA_VERSION",
    "WARMUP_OPTIMIZER_STEPS",
    "artifact_binding",
    "build_cloud_training_smoke_measurement",
    "require_cloud_training_smoke_measurement",
]
