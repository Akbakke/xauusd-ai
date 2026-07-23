"""Exact immutable human-approval binding for model-native Entry launch."""

from __future__ import annotations

import hashlib
import json
import re
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


SCHEMA_VERSION = "entry_model_native_launch_approval_v1"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_LAUNCH_APPROVAL"
PROJECT = "XAUUSD"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VEDTAK_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{7,127}$")


class EntryLaunchApprovalError(RuntimeError):
    """Raised when launch approval is missing, mutable, or not cross-bound."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    } or binding.get("schema_version") != SCHEMA_VERSION:
        raise EntryLaunchApprovalError("launch approval binding schema mismatch")
    vedtak_id = str(binding.get("vedtak_id") or "")
    if _VEDTAK_RE.fullmatch(vedtak_id) is None:
        raise EntryLaunchApprovalError("launch approval vedtak_id is invalid")
    event_path = Path(str(binding.get("event_path") or "")).expanduser()
    event_sha = str(binding.get("event_sha256") or "").lower()
    if (
        not event_path.is_absolute()
        or event_path.is_symlink()
        or not event_path.is_file()
        or _SHA256_RE.fullmatch(event_sha) is None
        or _sha256_file(event_path) != event_sha
    ):
        raise EntryLaunchApprovalError("launch approval event binding mismatch")
    try:
        require_newest_immutable_event(event_path, EVENT_PREFIX)
    except ImmutableEventAuthorityError as exc:
        raise EntryLaunchApprovalError(
            f"launch approval is not newest immutable authority: {exc}"
        ) from exc
    try:
        event_raw = json.loads(event_path.read_text(encoding="utf-8"))
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
    }
    if set(event) != required_event_keys:
        raise EntryLaunchApprovalError("launch approval event schema mismatch")
    bundle = Path(str(event.get("accepted_bundle_dir") or "")).expanduser()
    if (
        event.get("schema_version") != SCHEMA_VERSION
        or event.get("decision") != "ALLOW"
        or event.get("project") != PROJECT
        or event.get("vedtak_id") != vedtak_id
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
