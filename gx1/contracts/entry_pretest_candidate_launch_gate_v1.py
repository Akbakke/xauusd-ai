"""Immutable launch gate for a TRAIN/VAL-only pre-TEST candidate.

The pre-TEST recipe intentionally carries no physical TEST input.  This
separate gate binds one such candidate recipe to the already materialized
smoke-bundle audit and candidate-readiness evidence.  It cannot grant TEST,
promotion, shadow, paper or live authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)


SCHEMA_VERSION = "entry_pretest_candidate_launch_gate_v1"
EVENT_PREFIX = "ENTRY_PRETEST_CANDIDATE_LAUNCH_GATE"
READY_DECISION = "READY_FOR_PRETEST_CANDIDATE_TRAINING"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_BINDING_KEYS = frozenset(("path", "sha256"))
AUTHORITY = {
    "candidate_training": True,
    "test": False,
    "promotion": False,
    "shadow": False,
    "paper": False,
    "live": False,
}
_GATE_KEYS = frozenset(
    (
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "authority",
        "activation_authority",
        "run_id",
        "dataset_run_id",
        "dataset_dir",
        "out_bundle_dir",
        "recipe",
        "candidate_readiness",
        "smoke_bundle_audit",
    )
)


class PretestCandidateLaunchGateError(RuntimeError):
    """A pre-TEST candidate launch gate is absent, stale or inconsistent."""


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_binding(path: Path) -> dict[str, str]:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise PretestCandidateLaunchGateError(
            f"candidate gate artifact is not an absolute regular file: {raw}"
        )
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise PretestCandidateLaunchGateError(
            f"candidate gate artifact path is not canonical: {raw}"
        )
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PretestCandidateLaunchGateError(
            f"{label} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise PretestCandidateLaunchGateError(f"{label} root is not an object")
    return payload


def _binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or frozenset(value) != _BINDING_KEYS:
        raise PretestCandidateLaunchGateError(f"{label} binding keys are not exact")
    path = Path(str(value.get("path") or ""))
    digest = str(value.get("sha256") or "")
    if not path.is_absolute() or _SHA_RE.fullmatch(digest) is None:
        raise PretestCandidateLaunchGateError(f"{label} binding is invalid")
    observed = artifact_binding(path)
    if observed != {"path": str(path), "sha256": digest}:
        raise PretestCandidateLaunchGateError(f"{label} binding hash mismatch")
    return observed


def _candidate_readiness(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    smoke_audit: Mapping[str, str],
) -> None:
    payload = _read_json(path, label="candidate readiness")
    if (
        payload.get("schema_version") != "entry_candidate_readiness_model_native_v1"
        or payload.get("decision") != "READY_FOR_CANDIDATE_TRAINING"
        or payload.get("failures") != []
        or payload.get("candidate_training_allowed") is not True
        or payload.get("promotion_shadow_live_allowed") is not False
        or payload.get("activation_authority") is not False
    ):
        raise PretestCandidateLaunchGateError("candidate readiness is not safe")
    dataset_dir = str(recipe["dataset_dir"])
    if any(
        payload.get(key) != dataset_dir
        for key in (
            "expected_smoke_dataset_dir",
            "dataset_dir",
            "smoke_bundle_dataset_dir",
        )
    ):
        raise PretestCandidateLaunchGateError("candidate readiness dataset mismatch")
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping) or frozenset(bindings) != {
        "smoke_bundle_audit",
        "specialist_audit",
        "trainability_readiness",
    }:
        raise PretestCandidateLaunchGateError("candidate readiness input set is invalid")
    if payload.get("input_bindings_sha256") != canonical_json_sha256(bindings):
        raise PretestCandidateLaunchGateError("candidate readiness binding hash mismatch")
    if bindings.get("smoke_bundle_audit") != dict(smoke_audit):
        raise PretestCandidateLaunchGateError(
            "candidate readiness does not bind the supplied smoke audit"
        )


def _smoke_audit(path: Path, *, recipe: Mapping[str, Any]) -> None:
    payload = _read_json(path, label="smoke bundle audit")
    if (
        payload.get("schema_version") != "entry_foundation_smoke_bundle_audit_v7"
        or payload.get("dataset_dir") != recipe["dataset_dir"]
    ):
        raise PretestCandidateLaunchGateError("smoke bundle audit is incompatible")


def validate_gate_payload(
    value: Mapping[str, Any] | Any,
    *,
    expected_recipe: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one gate and rehash every authority-bearing input."""

    if not isinstance(value, Mapping) or frozenset(value) != _GATE_KEYS:
        raise PretestCandidateLaunchGateError("candidate launch gate keys are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != READY_DECISION
        or value.get("failures") != []
        or value.get("authority") != AUTHORITY
        or value.get("activation_authority") is not False
    ):
        raise PretestCandidateLaunchGateError("candidate launch gate authority is invalid")
    recipe = _binding(value.get("recipe"), label="recipe")
    if recipe != dict(expected_recipe):
        raise PretestCandidateLaunchGateError("candidate launch gate recipe mismatch")
    readiness = _binding(value.get("candidate_readiness"), label="candidate readiness")
    smoke = _binding(value.get("smoke_bundle_audit"), label="smoke bundle audit")
    identity = {
        "run_id": value.get("run_id"),
        "dataset_run_id": value.get("dataset_run_id"),
        "dataset_dir": value.get("dataset_dir"),
        "out_bundle_dir": value.get("out_bundle_dir"),
    }
    for key, observed in identity.items():
        if not isinstance(observed, str) or not observed:
            raise PretestCandidateLaunchGateError(f"candidate launch gate {key} is invalid")
    recipe_payload = _read_json(Path(recipe["path"]), label="recipe")
    for key, observed in identity.items():
        if recipe_payload.get(key) != observed:
            raise PretestCandidateLaunchGateError(
                f"candidate launch gate {key} differs from recipe"
            )
    _smoke_audit(Path(smoke["path"]), recipe=identity)
    _candidate_readiness(
        Path(readiness["path"]), recipe=identity, smoke_audit=smoke
    )
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def require_pretest_candidate_launch_gate(
    gate_json: str | Path,
    gate_sha256: str,
    *,
    expected_recipe_path: str | Path,
    expected_recipe_sha256: str,
) -> dict[str, Any]:
    """Read and validate exactly one immutable gate before candidate launch."""

    path = Path(gate_json).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PretestCandidateLaunchGateError("candidate launch gate path is invalid")
    path = path.resolve(strict=True)
    if _SHA_RE.fullmatch(str(gate_sha256)) is None or sha256_file(path) != gate_sha256:
        raise PretestCandidateLaunchGateError("candidate launch gate hash mismatch")
    try:
        require_newest_immutable_event(path, EVENT_PREFIX)
    except ImmutableEventAuthorityError as exc:
        raise PretestCandidateLaunchGateError(
            "candidate launch gate is not current immutable authority"
        ) from exc
    payload = _read_json(path, label="candidate launch gate")
    if payload.get("json_path") != str(path):
        raise PretestCandidateLaunchGateError(
            "candidate launch gate self-reference mismatch"
        )
    return validate_gate_payload(
        payload,
        expected_recipe={
            "path": str(Path(expected_recipe_path).expanduser().resolve(strict=True)),
            "sha256": str(expected_recipe_sha256),
        },
    )
