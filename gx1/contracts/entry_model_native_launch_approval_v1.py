"""Exact immutable human-approval binding for model-native Entry launch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)


SCHEMA_VERSION = "entry_model_native_launch_approval_v2"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_LAUNCH_APPROVAL"
VEDTAK_SCHEMA_VERSION = "entry_model_native_launch_vedtak_v1"
VEDTAK_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_LAUNCH_VEDTAK"
PROJECT = "XAUUSD"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VEDTAK_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{7,127}$")
_SOURCE_BINDING_KEYS = frozenset({"json_path", "sha256"})
_VEDTAK_EVIDENCE_KEYS = frozenset(
    {
        "sizing_adoption",
        "joint_exit_sizing_proof",
        "sizing_runtime_parity",
        "model_native_serve_parity",
        "model_native_direction_pocket_audit",
        "adaptation_lifecycle",
    }
)
_VEDTAK_REQUEST_KEYS = frozenset(
    {
        "transaction_id",
        "accepted_bundle_dir",
        "bundle_commit_sha256",
        "target_registry_path",
        "target_launch_state_path",
        "operating_point",
        "evidence",
    }
)
_VEDTAK_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "project",
        "vedtak_id",
        "launch_request",
        "launch_request_sha256",
    }
)


class EntryLaunchApprovalError(RuntimeError):
    """Raised when launch approval is missing, mutable, or not cross-bound."""


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EntryLaunchApprovalError(f"{label} cannot be opened") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise EntryLaunchApprovalError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        current = os.stat(path, follow_symlinks=False)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise EntryLaunchApprovalError(f"{label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def require_launch_vedtak_id(value: Any) -> str:
    """Return one exact one-time approval ID without granting authority."""

    vedtak_id = str(value or "")
    if _VEDTAK_RE.fullmatch(vedtak_id) is None:
        raise EntryLaunchApprovalError("launch approval vedtak_id is invalid")
    return vedtak_id


def _source_binding(
    value: Mapping[str, Any] | Any,
    *,
    label: str,
    verify_file: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_BINDING_KEYS):
        raise EntryLaunchApprovalError(f"{label} binding schema mismatch")
    path = Path(str(value.get("json_path") or "")).expanduser()
    digest = str(value.get("sha256") or "").lower()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or _SHA256_RE.fullmatch(digest) is None
    ):
        raise EntryLaunchApprovalError(f"{label} binding is invalid")
    path = path.resolve()
    if verify_file:
        if not path.is_file():
            raise EntryLaunchApprovalError(f"{label} binding file is missing")
        encoded = _read_regular_bytes(path, label=label)
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise EntryLaunchApprovalError(
                f"{label} binding byte hash mismatch"
            )
    return {"json_path": str(path), "sha256": digest}


def launch_vedtak_request(
    *,
    transaction_id: str,
    accepted_bundle_dir: Path,
    bundle_commit_sha256: str,
    target_registry_path: Path,
    target_launch_state_path: Path,
    operating_point: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact request a pre-existing launch vedtak must authorize."""

    transaction = str(transaction_id or "")
    if _VEDTAK_RE.fullmatch(transaction) is None:
        raise EntryLaunchApprovalError("launch transaction_id is invalid")
    bundle = Path(accepted_bundle_dir).expanduser()
    registry = Path(target_registry_path).expanduser()
    state = Path(target_launch_state_path).expanduser()
    for path, label, require_dir in (
        (bundle, "accepted bundle", True),
        (registry, "target registry", False),
        (state, "target launch state", False),
    ):
        if not path.is_absolute() or path.is_symlink():
            raise EntryLaunchApprovalError(f"{label} path is invalid")
        if require_dir and not path.is_dir():
            raise EntryLaunchApprovalError(f"{label} directory is unavailable")
    commit_sha = str(bundle_commit_sha256 or "").lower()
    if _SHA256_RE.fullmatch(commit_sha) is None:
        raise EntryLaunchApprovalError("bundle commit SHA is invalid")
    if not isinstance(operating_point, Mapping):
        raise EntryLaunchApprovalError("launch operating point is invalid")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        _VEDTAK_EVIDENCE_KEYS
    ):
        raise EntryLaunchApprovalError("launch vedtak evidence set mismatch")
    canonical_evidence = {
        name: _source_binding(
            evidence[name],
            label=f"launch vedtak evidence.{name}",
            verify_file=True,
        )
        for name in sorted(_VEDTAK_EVIDENCE_KEYS)
    }
    return {
        "transaction_id": transaction,
        "accepted_bundle_dir": str(bundle.resolve()),
        "bundle_commit_sha256": commit_sha,
        "target_registry_path": str(registry.resolve()),
        "target_launch_state_path": str(state.resolve()),
        "operating_point": dict(operating_point),
        "evidence": canonical_evidence,
    }


def _validate_launch_vedtak(
    event_path: Path,
    *,
    expected_request: Mapping[str, Any],
    require_newest: bool,
) -> dict[str, Any]:
    path = Path(event_path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EntryLaunchApprovalError(
            "launch vedtak must be an explicit immutable absolute event"
        )
    path = path.resolve()
    if require_newest:
        try:
            require_newest_immutable_event(path, VEDTAK_EVENT_PREFIX)
        except ImmutableEventAuthorityError as exc:
            raise EntryLaunchApprovalError(
                f"launch vedtak is not newest immutable authority: {exc}"
            ) from exc
    try:
        encoded = _read_regular_bytes(path, label="launch vedtak event")
        raw = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchApprovalError("launch vedtak event is unreadable") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(_VEDTAK_EVENT_KEYS):
        raise EntryLaunchApprovalError("launch vedtak event schema mismatch")
    event = dict(raw)
    request = event.get("launch_request")
    if not isinstance(request, Mapping) or set(request) != set(_VEDTAK_REQUEST_KEYS):
        raise EntryLaunchApprovalError("launch vedtak request schema mismatch")
    expected = dict(expected_request)
    if (
        event.get("schema_version") != VEDTAK_SCHEMA_VERSION
        or event.get("decision") != "AUTHORIZE"
        or event.get("project") != PROJECT
        or require_launch_vedtak_id(event.get("vedtak_id"))
        != event.get("vedtak_id")
        or request != expected
        or event.get("launch_request_sha256") != _canonical_sha256(expected)
    ):
        raise EntryLaunchApprovalError(
            "launch vedtak does not authorize the exact launch request"
        )
    return {
        "schema_version": VEDTAK_SCHEMA_VERSION,
        "vedtak_id": str(event["vedtak_id"]),
        "event_path": str(path),
        "event_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def require_preexisting_launch_vedtak(
    event_path: Path,
    *,
    expected_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an independently published newest launch authorization."""

    return _validate_launch_vedtak(
        event_path,
        expected_request=expected_request,
        require_newest=True,
    )


def require_historical_launch_vedtak(
    event_path: Path,
    *,
    expected_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact historical vedtak bytes solely for crash recovery."""

    return _validate_launch_vedtak(
        event_path,
        expected_request=expected_request,
        require_newest=False,
    )


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EntryLaunchApprovalError(
            "launch state is not strict canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def launch_state_approval_payload_sha256(
    launch_state: Mapping[str, Any],
) -> str:
    """Hash the complete launch state with the circular approval slot removed."""

    payload = dict(launch_state)
    payload.pop("accepted_via_vedtak", None)
    return _canonical_sha256(payload)


def require_entry_launch_approval(
    launch_state: Mapping[str, Any],
    *,
    accepted_bundle: Path,
) -> dict[str, Any]:
    """Validate the newest one-time approval against the complete launch state."""

    binding_raw = launch_state.get("accepted_via_vedtak")
    if not isinstance(binding_raw, Mapping):
        raise EntryLaunchApprovalError(
            "accepted_via_vedtak must be an immutable approval binding"
        )
    binding = dict(binding_raw)
    if set(binding) != {
        "schema_version",
        "vedtak_id",
        "event_path",
        "event_sha256",
        "vedtak_authority",
    } or binding.get("schema_version") != SCHEMA_VERSION:
        raise EntryLaunchApprovalError("launch approval binding schema mismatch")
    vedtak_id = require_launch_vedtak_id(binding.get("vedtak_id"))
    vedtak_authority = _source_binding(
        binding.get("vedtak_authority"),
        label="launch vedtak authority",
        verify_file=True,
    )
    event_path = Path(str(binding.get("event_path") or "")).expanduser()
    event_sha = str(binding.get("event_sha256") or "").lower()
    if (
        not event_path.is_absolute()
        or event_path.is_symlink()
        or not event_path.is_file()
        or _SHA256_RE.fullmatch(event_sha) is None
    ):
        raise EntryLaunchApprovalError("launch approval event binding mismatch")
    event_encoded = _read_regular_bytes(
        event_path,
        label="launch approval event",
    )
    if hashlib.sha256(event_encoded).hexdigest() != event_sha:
        raise EntryLaunchApprovalError("launch approval event binding mismatch")
    try:
        require_newest_immutable_event(event_path, EVENT_PREFIX)
    except ImmutableEventAuthorityError as exc:
        raise EntryLaunchApprovalError(
            f"launch approval is not newest immutable authority: {exc}"
        ) from exc
    try:
        event_raw = json.loads(event_encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchApprovalError("launch approval event is unreadable") from exc
    if not isinstance(event_raw, Mapping):
        raise EntryLaunchApprovalError("launch approval event root is invalid")
    event = dict(event_raw)
    required_event_keys = {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "project",
        "vedtak_id",
        "accepted_bundle_dir",
        "bundle_commit_sha256",
        "launch_state_payload_sha256",
        "vedtak_authority",
    }
    if set(event) != required_event_keys:
        raise EntryLaunchApprovalError("launch approval event schema mismatch")
    bundle = Path(str(event.get("accepted_bundle_dir") or "")).expanduser()
    if (
        event.get("schema_version") != SCHEMA_VERSION
        or event.get("decision") != "ALLOW"
        or event.get("project") != PROJECT
        or event.get("vedtak_id") != vedtak_id
        or event.get("vedtak_authority") != vedtak_authority
        or not bundle.is_absolute()
        or bundle.resolve() != accepted_bundle.resolve()
        or event.get("launch_state_payload_sha256")
        != launch_state_approval_payload_sha256(launch_state)
    ):
        raise EntryLaunchApprovalError(
            "launch approval event does not bind the exact launch state"
        )
    commit = require_bundle_commit_manifest(accepted_bundle.resolve())
    if event.get("bundle_commit_sha256") != commit.get("commit_sha256"):
        raise EntryLaunchApprovalError(
            "launch approval event bundle commit mismatch"
        )
    return binding
