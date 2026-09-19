"""Fail-closed capacity qualification for one measured Hopper training profile."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from gx1.contracts.cloud_training_smoke_measurement_v1 import (
    MEASURED_OPTIMIZER_STEPS,
    WARMUP_OPTIMIZER_STEPS,
    require_cloud_training_smoke_measurement,
)
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import MAX_EPOCHS
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_training_precision_v1 import (
    candidate_checkpoint_interval,
    candidate_validation_checkpoint_interval,
)

BENCHMARK_SCHEMA_VERSION = "cloud_hopper_training_benchmark_v1"
GATE_SCHEMA_VERSION = "cloud_training_capacity_gate_v1"
PRECISION_POLICY = "deterministic_bf16_hopper"
PASS_DECISION = "PASS_CAPACITY_QUALIFIED"
FAIL_DECISION = "FAIL_CAPACITY_NOT_QUALIFIED"
QUALIFICATION_TIME_LIMIT_SECONDS = 155_520
HARD_HOST_DEADLINE_SECONDS = 172_800
COST_CAP_NOK = Decimal(2500)
COST_BUFFER_FRACTION = Decimal("0.10")
RESTART_RESERVE_COUNT = 2
CHECKPOINT_WRITE_SAFETY_MULTIPLIER = Decimal(4)
MAX_JSON_BYTES = 128 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_TEST_TOKEN_RE = re.compile(r"(?:^|[^a-z0-9])test(?:[^a-z0-9]|$)", re.IGNORECASE)
_BINDING_KEYS = frozenset(("path", "sha256"))
_BENCHMARK_KEYS = frozenset(
    (
        "schema_version",
        "source_commit",
        "smoke_recipe",
        "smoke_bundle_commit",
        "cloud_host_profile",
        "precision_policy",
        "batch_size",
        "train_rows",
        "val_rows",
        "max_epochs",
        "warmup_optimizer_steps",
        "measured_optimizer_steps",
        "measured_train_rows",
        "measured_train_seconds",
        "measured_val_rows",
        "measured_val_seconds",
        "preflight_seconds",
        "checkpoint_write_seconds",
        "provider_price_usd_per_hour",
        "nok_per_usd",
        "fx_observed_utc",
        "hard_host_deadline_seconds",
    )
)
_AUTHORITY = {
    "cloud_purchase": False,
    "cuda": False,
    "training": False,
    "test": False,
    "activation": False,
}
_LIMIT_KEYS = frozenset(
    (
        "qualification_time_limit_seconds",
        "hard_host_deadline_seconds",
        "cost_cap_nok",
        "cost_buffer_fraction",
        "restart_reserve_count",
    )
)
_PROJECTION_KEYS = frozenset(
    (
        "measured_train_rows_per_second",
        "projected_train_seconds",
        "projected_val_seconds",
        "projected_preflight_seconds",
        "projected_checkpoint_seconds",
        "projected_restart_reserve_seconds",
        "projected_worst_case_seconds",
        "projected_worst_case_hours",
        "projected_unbuffered_cost_nok",
        "projected_buffered_cost_nok",
        "hard_deadline_buffered_cost_nok",
    )
)
_GATE_KEYS = frozenset(
    (
        "schema_version",
        "decision",
        "failures",
        "capacity_qualified",
        "report_only",
        "activation_authority",
        "authority",
        "side_effects",
        "benchmark",
        "source_commit",
        "smoke_recipe",
        "cloud_host_profile",
        "precision_policy",
        "batch_size",
        "limits",
        "projection",
    )
)


class CloudTrainingCapacityGateError(RuntimeError):
    """A benchmark or capacity decision is malformed, stale, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_test_token(path: Path) -> bool:
    return any(_TEST_TOKEN_RE.search(part) is not None for part in path.parts)


def artifact_binding(
    path: Path,
    *,
    label: str,
    forbid_test_like: bool = True,
) -> dict[str, str]:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise CloudTrainingCapacityGateError(
            f"{label} is not an absolute regular file: {raw}"
        )
    if any(parent.is_symlink() for parent in raw.parents):
        raise CloudTrainingCapacityGateError(f"{label} path contains a symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise CloudTrainingCapacityGateError(f"{label} path is not canonical: {raw}")
    if forbid_test_like and _contains_test_token(resolved):
        raise CloudTrainingCapacityGateError(f"{label} path is TEST-like: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    binding = artifact_binding(path, label=label)
    canonical = Path(binding["path"])
    try:
        if canonical.stat().st_size > MAX_JSON_BYTES:
            raise CloudTrainingCapacityGateError(f"{label} exceeds bounded JSON size")
        payload = json.loads(
            canonical.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CloudTrainingCapacityGateError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CloudTrainingCapacityGateError(f"{label} root is not an object")
    return payload


def _require_binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or frozenset(value) != _BINDING_KEYS:
        raise CloudTrainingCapacityGateError(f"{label} binding keys are not exact")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise CloudTrainingCapacityGateError(f"{label} binding values are invalid")
    if _SHA256_RE.fullmatch(digest) is None:
        raise CloudTrainingCapacityGateError(f"{label} SHA-256 is invalid")
    observed = artifact_binding(Path(path_value), label=label)
    if observed != {"path": path_value, "sha256": digest}:
        raise CloudTrainingCapacityGateError(f"{label} binding hash/path mismatch")
    return observed


def _require_int(
    value: Any,
    *,
    label: str,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CloudTrainingCapacityGateError(f"{label} is invalid")
    return value


def _require_positive_decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloudTrainingCapacityGateError(f"{label} is invalid")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise CloudTrainingCapacityGateError(f"{label} must be positive and finite")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0:
        raise CloudTrainingCapacityGateError(f"{label} must be positive and finite")
    return parsed


def _require_utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise CloudTrainingCapacityGateError(f"{label} is not exact UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CloudTrainingCapacityGateError(f"{label} is not valid UTC") from exc
    if parsed.tzinfo != timezone.utc:
        raise CloudTrainingCapacityGateError(f"{label} is not UTC")
    return value


def _read_committed_json(
    path: Path,
    *,
    expected_size_bytes: int,
    label: str,
) -> dict[str, Any]:
    if expected_size_bytes < 1 or expected_size_bytes > 16 * 1024 * 1024:
        raise CloudTrainingCapacityGateError(f"{label} size is invalid")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size_bytes:
        raise CloudTrainingCapacityGateError(f"{label} committed file is invalid")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CloudTrainingCapacityGateError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CloudTrainingCapacityGateError(f"{label} root is not an object")
    return value


def _require_bound_smoke_measurement(payload: Mapping[str, Any]) -> dict[str, Any]:
    commit_binding = payload["smoke_bundle_commit"]
    commit_path = Path(commit_binding["path"])
    if commit_path.name != BUNDLE_COMMIT_MANIFEST_NAME:
        raise CloudTrainingCapacityGateError("smoke bundle commit name is invalid")
    bundle_dir = commit_path.parent
    try:
        manifest = require_bundle_commit_manifest(bundle_dir)
    except RuntimeError as exc:
        raise CloudTrainingCapacityGateError("smoke bundle commit is invalid") from exc
    if artifact_binding(commit_path, label="smoke bundle commit") != commit_binding:
        raise CloudTrainingCapacityGateError("smoke bundle commit binding mismatch")
    artifacts = manifest["artifacts"]
    metadata = _read_committed_json(
        bundle_dir / "bundle_metadata.json",
        expected_size_bytes=int(artifacts["bundle_metadata.json"]["size_bytes"]),
        label="smoke bundle metadata",
    )
    lock = _read_committed_json(
        bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
        expected_size_bytes=int(artifacts["MASTER_TRANSFORMER_LOCK.json"]["size_bytes"]),
        label="smoke bundle lock",
    )
    try:
        measurement = require_cloud_training_smoke_measurement(
            metadata.get("cloud_hopper_smoke_measurement")
        )
        lock_measurement = require_cloud_training_smoke_measurement(
            lock.get("cloud_hopper_smoke_measurement")
        )
    except RuntimeError as exc:
        raise CloudTrainingCapacityGateError("smoke bundle measurement is invalid") from exc
    if measurement != lock_measurement:
        raise CloudTrainingCapacityGateError("smoke bundle measurement split-brain")
    model_artifact = artifacts["model_state_dict.pt"]
    lineage = metadata.get("run_lineage")
    provenance = metadata.get("recipe_source_provenance")
    if (
        metadata.get("git_commit") != measurement["source_commit"]
        or metadata.get("execution_tier") != "canonical"
        or metadata.get("batch_size") != measurement["batch_size"]
        or not isinstance(lineage, Mapping)
        or lineage.get("training_profile") != "smoke"
        or lineage.get("training_run_id") != measurement["run_id"]
        or lineage.get("physical_train_rows") != measurement["physical_train_rows"]
        or lineage.get("physical_val_rows") != measurement["physical_val_rows"]
        or lineage.get("effective_train_rows") != measurement["sampled_train_rows"]
        or lineage.get("effective_val_rows") != measurement["sampled_val_rows"]
        or not isinstance(provenance, Mapping)
        or provenance.get("recipe_audit_path") != measurement["smoke_recipe"]["path"]
        or provenance.get("recipe_audit_sha256")
        != measurement["smoke_recipe"]["sha256"]
        or provenance.get("source_commit") != measurement["source_commit"]
        or model_artifact.get("sha256") != measurement["checkpoint_sha256"]
        or model_artifact.get("size_bytes") != measurement["checkpoint_size_bytes"]
    ):
        raise CloudTrainingCapacityGateError(
            "smoke measurement differs from committed bundle evidence"
        )
    expected_values = {
        "source_commit": measurement["source_commit"],
        "smoke_recipe": measurement["smoke_recipe"],
        "cloud_host_profile": measurement["cloud_host_profile"],
        "precision_policy": measurement["precision_policy"],
        "batch_size": measurement["batch_size"],
        "train_rows": measurement["physical_train_rows"],
        "val_rows": measurement["physical_val_rows"],
        "warmup_optimizer_steps": measurement["warmup_optimizer_steps"],
        "measured_optimizer_steps": measurement["measured_optimizer_steps"],
        "measured_train_rows": measurement["measured_train_rows"],
        "measured_train_seconds": measurement["measured_train_seconds"],
        "measured_val_rows": measurement["measured_val_rows"],
        "measured_val_seconds": measurement["measured_val_seconds"],
        "preflight_seconds": measurement["preflight_seconds"],
        "checkpoint_write_seconds": measurement["checkpoint_write_seconds"],
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise CloudTrainingCapacityGateError(
                f"benchmark {key} differs from committed smoke measurement"
            )
    return measurement


def validate_hopper_benchmark_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _BENCHMARK_KEYS:
        raise CloudTrainingCapacityGateError("benchmark keys are not exact")
    payload = dict(value)
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise CloudTrainingCapacityGateError("benchmark schema_version is invalid")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or _COMMIT_RE.fullmatch(source_commit) is None
    ):
        raise CloudTrainingCapacityGateError("source_commit is invalid")
    payload["smoke_recipe"] = _require_binding(
        payload.get("smoke_recipe"), label="smoke recipe"
    )
    payload["smoke_bundle_commit"] = _require_binding(
        payload.get("smoke_bundle_commit"), label="smoke bundle commit"
    )
    payload["cloud_host_profile"] = _require_binding(
        payload.get("cloud_host_profile"), label="cloud host profile"
    )
    if payload.get("precision_policy") != PRECISION_POLICY:
        raise CloudTrainingCapacityGateError("precision_policy is invalid")
    batch_size = _require_int(payload.get("batch_size"), label="batch_size")
    if batch_size not in (32, 64):
        raise CloudTrainingCapacityGateError("batch_size must be 32 or 64")
    train_rows = _require_int(payload.get("train_rows"), label="train_rows")
    val_rows = _require_int(payload.get("val_rows"), label="val_rows")
    max_epochs = _require_int(payload.get("max_epochs"), label="max_epochs")
    if max_epochs != MAX_EPOCHS:
        raise CloudTrainingCapacityGateError("max_epochs is not the candidate maximum")
    warmup_steps = _require_int(
        payload.get("warmup_optimizer_steps"),
        label="warmup_optimizer_steps",
        minimum=0,
    )
    if warmup_steps != WARMUP_OPTIMIZER_STEPS:
        raise CloudTrainingCapacityGateError("warmup_optimizer_steps is not exact")
    measured_steps = _require_int(
        payload.get("measured_optimizer_steps"), label="measured_optimizer_steps"
    )
    if measured_steps != MEASURED_OPTIMIZER_STEPS:
        raise CloudTrainingCapacityGateError(
            "measured_optimizer_steps is not exact"
        )
    measured_rows = _require_int(
        payload.get("measured_train_rows"), label="measured_train_rows"
    )
    if measured_rows != measured_steps * batch_size:
        raise CloudTrainingCapacityGateError(
            "measured_train_rows must equal measured_optimizer_steps times batch_size"
        )
    # The immutable measurement owner fixes 32 warmup + 256 measured steps.
    # A duration floor here would reject fast hosts rather than improve the
    # sample: at 600 s / 256 steps even batch 64 cannot fit the physical
    # five-year TRAIN population inside the 43.2 h ceiling. Keep actual
    # synchronized positive duration; never stretch or pad measured time.
    _require_positive_decimal(
        payload.get("measured_train_seconds"), label="measured_train_seconds"
    )
    measured_val_rows = _require_int(
        payload.get("measured_val_rows"), label="measured_val_rows"
    )
    if measured_val_rows > val_rows:
        raise CloudTrainingCapacityGateError("measured_val_rows exceeds VAL population")
    _require_positive_decimal(
        payload.get("measured_val_seconds"), label="measured_val_seconds"
    )
    _require_positive_decimal(
        payload.get("preflight_seconds"), label="preflight_seconds"
    )
    _require_positive_decimal(
        payload.get("checkpoint_write_seconds"), label="checkpoint_write_seconds"
    )
    _require_positive_decimal(
        payload.get("provider_price_usd_per_hour"),
        label="provider_price_usd_per_hour",
    )
    _require_positive_decimal(payload.get("nok_per_usd"), label="nok_per_usd")
    _require_utc(payload.get("fx_observed_utc"), label="fx_observed_utc")
    hard_deadline = _require_int(
        payload.get("hard_host_deadline_seconds"),
        label="hard_host_deadline_seconds",
    )
    if hard_deadline != HARD_HOST_DEADLINE_SECONDS:
        raise CloudTrainingCapacityGateError(
            "hard_host_deadline_seconds must equal 172800"
        )
    _require_bound_smoke_measurement(payload)
    payload["train_rows"] = train_rows
    payload["val_rows"] = val_rows
    payload["max_epochs"] = max_epochs
    return payload


def require_hopper_benchmark(
    path: Path,
    sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise CloudTrainingCapacityGateError("benchmark SHA-256 is invalid")
    binding = artifact_binding(path, label="benchmark")
    if binding["sha256"] != sha256:
        raise CloudTrainingCapacityGateError("benchmark SHA-256 mismatch")
    payload = validate_hopper_benchmark_payload(
        _read_json_object(Path(binding["path"]), label="benchmark")
    )
    return payload, binding


def build_hopper_benchmark_from_smoke_bundle(bundle_dir: Path) -> dict[str, Any]:
    directory = Path(bundle_dir).expanduser()
    if (
        not directory.is_absolute()
        or directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve(strict=True) != directory
    ):
        raise CloudTrainingCapacityGateError("smoke bundle directory is invalid")
    try:
        manifest = require_bundle_commit_manifest(directory)
    except RuntimeError as exc:
        raise CloudTrainingCapacityGateError("smoke bundle commit is invalid") from exc
    manifest_path = directory / BUNDLE_COMMIT_MANIFEST_NAME
    metadata_artifact = manifest["artifacts"]["bundle_metadata.json"]
    metadata = _read_committed_json(
        directory / "bundle_metadata.json",
        expected_size_bytes=int(metadata_artifact["size_bytes"]),
        label="smoke bundle metadata",
    )
    try:
        measurement = require_cloud_training_smoke_measurement(
            metadata.get("cloud_hopper_smoke_measurement")
        )
    except RuntimeError as exc:
        raise CloudTrainingCapacityGateError("smoke bundle measurement is invalid") from exc
    from gx1.contracts.cloud_training_host_profile_v1 import (
        require_cloud_training_host_profile_file,
    )

    host_profile = require_cloud_training_host_profile_file(
        Path(measurement["cloud_host_profile"]["path"]),
        measurement["cloud_host_profile"]["sha256"],
        repo=Path("/"),
        verify_runtime=False,
    )
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "source_commit": measurement["source_commit"],
        "smoke_recipe": measurement["smoke_recipe"],
        "smoke_bundle_commit": artifact_binding(
            manifest_path, label="smoke bundle commit"
        ),
        "cloud_host_profile": measurement["cloud_host_profile"],
        "precision_policy": measurement["precision_policy"],
        "batch_size": measurement["batch_size"],
        "train_rows": measurement["physical_train_rows"],
        "val_rows": measurement["physical_val_rows"],
        "max_epochs": MAX_EPOCHS,
        "warmup_optimizer_steps": measurement["warmup_optimizer_steps"],
        "measured_optimizer_steps": measurement["measured_optimizer_steps"],
        "measured_train_rows": measurement["measured_train_rows"],
        "measured_train_seconds": measurement["measured_train_seconds"],
        "measured_val_rows": measurement["measured_val_rows"],
        "measured_val_seconds": measurement["measured_val_seconds"],
        "preflight_seconds": measurement["preflight_seconds"],
        "checkpoint_write_seconds": measurement["checkpoint_write_seconds"],
        "provider_price_usd_per_hour": host_profile["provider"][
            "hourly_price_usd"
        ],
        "nok_per_usd": host_profile["budget"]["nok_per_usd"],
        "fx_observed_utc": host_profile["budget"]["fx_observed_utc"],
        "hard_host_deadline_seconds": host_profile["budget"][
            "hard_deadline_seconds"
        ],
    }
    return validate_hopper_benchmark_payload(payload)


def _as_float(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CloudTrainingCapacityGateError("capacity projection is non-finite")
    return result


def evaluate_cloud_training_capacity(
    benchmark: Mapping[str, Any],
    *,
    benchmark_binding: Mapping[str, str],
) -> dict[str, Any]:
    payload = validate_hopper_benchmark_payload(benchmark)
    binding = _require_binding(benchmark_binding, label="benchmark")
    benchmark_path = Path(binding["path"])
    disk_payload = validate_hopper_benchmark_payload(
        _read_json_object(benchmark_path, label="benchmark")
    )
    if disk_payload != payload:
        raise CloudTrainingCapacityGateError(
            "benchmark payload does not match its immutable binding"
        )

    measured_rows = Decimal(payload["measured_train_rows"])
    measured_seconds = Decimal(str(payload["measured_train_seconds"]))
    train_rows_per_second = measured_rows / measured_seconds
    epoch_count = Decimal(payload["max_epochs"])
    projected_train_seconds = (
        Decimal(payload["train_rows"]) * epoch_count / train_rows_per_second
    )
    measured_val_rows_per_second = Decimal(payload["measured_val_rows"]) / Decimal(
        str(payload["measured_val_seconds"])
    )
    projected_val_seconds = (
        Decimal(payload["val_rows"])
        * epoch_count
        / measured_val_rows_per_second
    )
    projected_preflight_seconds = Decimal(str(payload["preflight_seconds"]))
    train_steps = math.ceil(payload["train_rows"] / payload["batch_size"])
    val_batches = math.ceil(payload["val_rows"] / payload["batch_size"])
    train_checkpoint_count = math.ceil(
        train_steps
        * payload["max_epochs"]
        / candidate_checkpoint_interval(PRECISION_POLICY)
    )
    val_checkpoint_count = math.ceil(
        val_batches
        * payload["max_epochs"]
        / candidate_validation_checkpoint_interval(PRECISION_POLICY)
    )
    projected_checkpoint_seconds = (
        Decimal(train_checkpoint_count + val_checkpoint_count)
        * Decimal(str(payload["checkpoint_write_seconds"]))
        * CHECKPOINT_WRITE_SAFETY_MULTIPLIER
    )
    projected_restart_seconds = (
        (
            projected_preflight_seconds
            + Decimal(str(payload["checkpoint_write_seconds"]))
            * CHECKPOINT_WRITE_SAFETY_MULTIPLIER
        )
        * RESTART_RESERVE_COUNT
    )
    projected_total_seconds = (
        projected_train_seconds
        + projected_val_seconds
        + projected_preflight_seconds
        + projected_checkpoint_seconds
        + projected_restart_seconds
    )
    price_usd_per_hour = Decimal(str(payload["provider_price_usd_per_hour"]))
    nok_per_usd = Decimal(str(payload["nok_per_usd"]))
    buffer_multiplier = Decimal(1) + COST_BUFFER_FRACTION
    projected_cost_nok = (
        projected_total_seconds / Decimal(3600) * price_usd_per_hour * nok_per_usd
    )
    buffered_projected_cost_nok = projected_cost_nok * buffer_multiplier
    hard_deadline_buffered_cost_nok = (
        Decimal(HARD_HOST_DEADLINE_SECONDS)
        / Decimal(3600)
        * price_usd_per_hour
        * nok_per_usd
        * buffer_multiplier
    )

    failures: list[str] = []
    if projected_total_seconds > QUALIFICATION_TIME_LIMIT_SECONDS:
        failures.append("PROJECTED_WORST_CASE_EXCEEDS_43_2_HOURS")
    if max(buffered_projected_cost_nok, hard_deadline_buffered_cost_nok) > COST_CAP_NOK:
        failures.append("BUFFERED_COST_EXCEEDS_2500_NOK")
    capacity_qualified = not failures
    projection = {
        "measured_train_rows_per_second": _as_float(train_rows_per_second),
        "projected_train_seconds": _as_float(projected_train_seconds),
        "projected_val_seconds": _as_float(projected_val_seconds),
        "projected_preflight_seconds": _as_float(projected_preflight_seconds),
        "projected_checkpoint_seconds": _as_float(projected_checkpoint_seconds),
        "projected_restart_reserve_seconds": _as_float(projected_restart_seconds),
        "projected_worst_case_seconds": _as_float(projected_total_seconds),
        "projected_worst_case_hours": _as_float(
            projected_total_seconds / Decimal(3600)
        ),
        "projected_unbuffered_cost_nok": _as_float(projected_cost_nok),
        "projected_buffered_cost_nok": _as_float(buffered_projected_cost_nok),
        "hard_deadline_buffered_cost_nok": _as_float(hard_deadline_buffered_cost_nok),
    }
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "decision": PASS_DECISION if capacity_qualified else FAIL_DECISION,
        "failures": failures,
        "capacity_qualified": capacity_qualified,
        "report_only": True,
        "activation_authority": False,
        "authority": dict(_AUTHORITY),
        "side_effects": [],
        "benchmark": binding,
        "source_commit": payload["source_commit"],
        "smoke_recipe": dict(payload["smoke_recipe"]),
        "cloud_host_profile": dict(payload["cloud_host_profile"]),
        "precision_policy": payload["precision_policy"],
        "batch_size": payload["batch_size"],
        "limits": {
            "qualification_time_limit_seconds": QUALIFICATION_TIME_LIMIT_SECONDS,
            "hard_host_deadline_seconds": HARD_HOST_DEADLINE_SECONDS,
            "cost_cap_nok": float(COST_CAP_NOK),
            "cost_buffer_fraction": float(COST_BUFFER_FRACTION),
            "restart_reserve_count": RESTART_RESERVE_COUNT,
        },
        "projection": projection,
    }


def validate_capacity_gate_payload(
    value: Any,
    *,
    expected_benchmark_path: Path,
    expected_benchmark_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _GATE_KEYS:
        raise CloudTrainingCapacityGateError("capacity gate keys are not exact")
    binding = _require_binding(value.get("benchmark"), label="benchmark")
    expected_binding = artifact_binding(expected_benchmark_path, label="benchmark")
    if expected_binding["sha256"] != expected_benchmark_sha256:
        raise CloudTrainingCapacityGateError("expected benchmark SHA-256 mismatch")
    if binding != expected_binding:
        raise CloudTrainingCapacityGateError("capacity gate benchmark mismatch")
    benchmark, _ = require_hopper_benchmark(Path(binding["path"]), binding["sha256"])
    expected = evaluate_cloud_training_capacity(benchmark, benchmark_binding=binding)
    if value != expected:
        raise CloudTrainingCapacityGateError(
            "capacity gate does not match the exact benchmark projection"
        )
    if (
        value.get("report_only") is not True
        or value.get("activation_authority") is not False
        or value.get("authority") != _AUTHORITY
        or value.get("side_effects") != []
    ):
        raise CloudTrainingCapacityGateError("capacity gate authority is invalid")
    limits = value.get("limits")
    projection = value.get("projection")
    if not isinstance(limits, Mapping) or frozenset(limits) != _LIMIT_KEYS:
        raise CloudTrainingCapacityGateError("capacity gate limit keys are not exact")
    if not isinstance(projection, Mapping) or frozenset(projection) != _PROJECTION_KEYS:
        raise CloudTrainingCapacityGateError(
            "capacity gate projection keys are not exact"
        )
    return dict(value)


def require_cloud_training_capacity_gate(
    path: Path,
    sha256: str,
    *,
    expected_benchmark_path: Path,
    expected_benchmark_sha256: str,
) -> dict[str, Any]:
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise CloudTrainingCapacityGateError("capacity gate SHA-256 is invalid")
    binding = artifact_binding(path, label="capacity gate", forbid_test_like=False)
    if binding["sha256"] != sha256:
        raise CloudTrainingCapacityGateError("capacity gate SHA-256 mismatch")
    payload = _read_json_object(Path(binding["path"]), label="capacity gate")
    return validate_capacity_gate_payload(
        payload,
        expected_benchmark_path=expected_benchmark_path,
        expected_benchmark_sha256=expected_benchmark_sha256,
    )


def require_cloud_training_capacity_gate_for_candidate(
    path: Path,
    sha256: str,
    *,
    expected_source_commit: str,
    expected_host_profile_path: Path,
    expected_host_profile_sha256: str,
    expected_batch_size: int,
) -> dict[str, Any]:
    """Require a PASS gate transitively bound to its real smoke and host."""

    if _COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise CloudTrainingCapacityGateError("expected source commit is invalid")
    gate_binding = artifact_binding(path, label="capacity gate", forbid_test_like=False)
    if gate_binding["sha256"] != sha256:
        raise CloudTrainingCapacityGateError("capacity gate SHA-256 mismatch")
    raw_gate = _read_json_object(Path(gate_binding["path"]), label="capacity gate")
    benchmark_binding = _require_binding(raw_gate.get("benchmark"), label="benchmark")
    gate = validate_capacity_gate_payload(
        raw_gate,
        expected_benchmark_path=Path(benchmark_binding["path"]),
        expected_benchmark_sha256=benchmark_binding["sha256"],
    )
    if gate["decision"] != PASS_DECISION or gate["capacity_qualified"] is not True:
        raise CloudTrainingCapacityGateError("capacity gate is not PASS")
    expected_host_binding = artifact_binding(
        expected_host_profile_path,
        label="cloud host profile",
    )
    if expected_host_binding["sha256"] != expected_host_profile_sha256:
        raise CloudTrainingCapacityGateError("expected host profile SHA-256 mismatch")
    if (
        gate["source_commit"] != expected_source_commit
        or gate["cloud_host_profile"] != expected_host_binding
        or gate["precision_policy"] != PRECISION_POLICY
        or gate["batch_size"] != expected_batch_size
    ):
        raise CloudTrainingCapacityGateError(
            "capacity gate source, host, precision or batch binding mismatch"
        )

    from gx1.contracts.cloud_training_host_profile_v1 import (
        require_cloud_training_host_profile_file,
    )
    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
        require_pretest_technical_recipe_metadata,
    )

    host_profile = require_cloud_training_host_profile_file(
        Path(expected_host_binding["path"]),
        expected_host_binding["sha256"],
        repo=Path("/"),
        verify_runtime=False,
    )
    if host_profile["source"]["commit"] != expected_source_commit:
        raise CloudTrainingCapacityGateError("host profile source commit mismatch")
    smoke_recipe = require_pretest_technical_recipe_metadata(
        _read_json_object(Path(gate["smoke_recipe"]["path"]), label="smoke recipe"),
        expected_profile="smoke",
    )
    smoke_cli = smoke_recipe["trainer_cli"]
    if (
        smoke_recipe["source_commit"] != expected_source_commit
        or smoke_cli.get("precision_policy") != PRECISION_POLICY
        or smoke_cli.get("batch_size") != expected_batch_size
        or smoke_cli.get("cloud_host_profile_path") != expected_host_binding["path"]
        or smoke_cli.get("cloud_host_profile_sha256") != expected_host_binding["sha256"]
    ):
        raise CloudTrainingCapacityGateError("capacity smoke recipe binding mismatch")
    benchmark, _ = require_hopper_benchmark(
        Path(benchmark_binding["path"]), benchmark_binding["sha256"]
    )
    if (
        float(benchmark["provider_price_usd_per_hour"])
        != float(host_profile["provider"]["hourly_price_usd"])
        or float(benchmark["nok_per_usd"])
        != float(host_profile["budget"]["nok_per_usd"])
        or benchmark["fx_observed_utc"] != host_profile["budget"]["fx_observed_utc"]
        or benchmark["hard_host_deadline_seconds"]
        != host_profile["budget"]["hard_deadline_seconds"]
    ):
        raise CloudTrainingCapacityGateError("capacity benchmark budget binding mismatch")
    return gate
