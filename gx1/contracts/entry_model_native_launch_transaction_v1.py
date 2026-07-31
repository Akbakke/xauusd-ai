"""Atomic authority barrier for model-native Entry launch publication.

The artifact registry and launch-state JSON are two separate filesystem
objects and therefore cannot be replaced by one kernel rename.  This contract
makes their *authority* transactional: a pre-published immutable commit event
binds the exact final bytes of both files.  Consumers accept Entry only when
both target files match that same newest COMMIT event.  Any partial update,
newer failure event, changed bundle, or changed registry is fail-closed.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.contracts.entry_model_native_launch_approval_v1 import (
    launch_vedtak_request,
    require_historical_launch_vedtak,
    require_preexisting_launch_vedtak,
)
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    select_latest_immutable_event,
)
from gx1.contracts.live_tail_publication_v1 import (
    LiveTailAuthorityError,
    require_live_tail_launch_authority,
)
from gx1.models.entry_v10.direction_decision_contract import (
    require_model_direction_operating_point,
)


SCHEMA_VERSION = "entry_model_native_launch_transaction_v1"
DECLARATION_SCHEMA_VERSION = "entry_model_native_launch_transaction_declaration_v1"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_LAUNCH_TRANSACTION"
PROJECT = "XAUUSD"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{7,127}$")
_DECLARATION_KEYS = frozenset(
    {"schema_version", "transaction_id", "event_dir"}
)
_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "project",
        "transaction_id",
        "vedtak_id",
        "target_registry_path",
        "target_launch_state_path",
        "registry_sha256",
        "launch_state_sha256",
        "registry_before_artifact",
        "launch_state_before_artifact",
        "accepted_bundle_dir",
        "bundle_commit_sha256",
        "vedtak_authority",
    }
)
_BACKUP_KEYS = frozenset({"path", "sha256"})
_SOURCE_BINDING_KEYS = frozenset({"json_path", "sha256"})
_LAUNCH_STATE_KEYS = frozenset(
    {
        "schema_version",
        "project",
        "updated_utc",
        "decision",
        "latest_terminal_event_id",
        "latest_terminal_event_decision",
        "decision_surface",
        "public_trade_flat_surface",
        "required_contract_mode",
        "required_signal_dim",
        "required_base_signal_dim",
        "required_selected_feature_count",
        "required_mandatory_causal_layer_feature_count",
        "required_train_ranked_remainder_feature_count",
        "required_mandatory_causal_layer_count",
        "required_ctx_cont_dim",
        "required_ctx_cat_dim",
        "dataset_event_id",
        "accepted_bundle_dir",
        "bundle_metadata_sha256",
        "sizing_adoption_mode",
        "sizing_authority_contract",
        "joint_exit_execution_proof_evidence",
        "sizing_runtime_parity_evidence",
        "serve_gate_evidence",
        "adaptation_lifecycle_evidence",
        "new_entry_live_tail_authority",
        "operating_point",
        "launch_transaction",
        "accepted_via_vedtak",
        "blockers",
    }
)
_V10_ENTRY_KEYS = frozenset(
    {
        "path",
        "status",
        "in_sample_only",
        "contract_mode",
        "operating_point",
        "launch_transaction_id",
        "bundle_commit_sha256",
        "vedtak",
    }
)


class EntryLaunchTransactionError(RuntimeError):
    """Launch files do not form one complete immutable transaction."""


@contextmanager
def entry_launch_authority_lock(
    *,
    registry_path: Path,
    launch_state_path: Path,
):
    """Serialize launch publication and the final broker mutation lease."""

    targets: list[Path] = []
    for raw, label in (
        (registry_path, "artifact registry"),
        (launch_state_path, "launch state"),
    ):
        path = Path(raw).expanduser()
        if not path.is_absolute() or path.is_symlink():
            raise EntryLaunchTransactionError(
                f"{label} lock target must be absolute and non-symlinked"
            )
        targets.append(path.resolve())
    identity = "\n".join(str(path) for path in targets).encode("utf-8")
    lock_path = Path("/tmp") / (
        ".gx1-entry-launch-"
        + hashlib.sha256(identity).hexdigest()
        + ".lock"
    )
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EntryLaunchTransactionError(
                "launch authority lock is not a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        _fail(f"{label} is not an absolute non-symlink file: {raw}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(raw, flags)
    except OSError as exc:
        raise EntryLaunchTransactionError(
            f"[ENTRY_LAUNCH_TRANSACTION_INVALID] {label} cannot be opened"
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        current = os.stat(raw, follow_symlinks=False)
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
            _fail(f"{label} changed while it was read")
    finally:
        os.close(fd)
    return raw.resolve(), b"".join(chunks)


def _fail(detail: str) -> None:
    raise EntryLaunchTransactionError(
        f"[ENTRY_LAUNCH_TRANSACTION_INVALID] {detail}"
    )


def _exact_mapping(
    value: Mapping[str, Any] | Any,
    keys: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    observed = dict(value)
    if set(observed) != set(keys):
        _fail(
            f"{label} keys mismatch: "
            f"missing={sorted(keys - set(observed))} "
            f"unexpected={sorted(set(observed) - keys)}"
        )
    return observed


def _sha(value: Any, *, label: str) -> str:
    parsed = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(parsed) is None:
        _fail(f"{label} is not an exact SHA-256")
    return parsed


def _absolute_regular(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        _fail(f"{label} is not an absolute regular file: {raw}")
    return raw.resolve()


def _absolute_target(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        _fail(f"{label} is not an absolute non-symlink path: {raw}")
    return raw.resolve()


def _backup(
    value: Mapping[str, Any] | Any,
    *,
    label: str,
    event_dir: Path | None = None,
    expected_name: str | None = None,
) -> dict[str, str]:
    observed = _exact_mapping(value, _BACKUP_KEYS, label=label)
    path = _absolute_regular(Path(str(observed["path"])), label=f"{label}.path")
    expected = _sha(observed["sha256"], label=f"{label}.sha256")
    if sha256_file(path) != expected:
        _fail(f"{label} byte hash mismatch")
    if event_dir is not None and path.parent != event_dir.resolve():
        _fail(f"{label} is outside the transaction event directory")
    if expected_name is not None and path.name != expected_name:
        _fail(f"{label} filename mismatch")
    return {"path": str(path), "sha256": expected}


def launch_transaction_declaration(
    *,
    transaction_id: str,
    event_dir: Path,
) -> dict[str, str]:
    """Return the exact non-circular declaration embedded in launch state."""

    transaction = str(transaction_id)
    if _ID_RE.fullmatch(transaction) is None:
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] transaction_id is invalid"
        )
    directory = Path(event_dir).expanduser()
    if not directory.is_absolute() or directory.is_symlink():
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] event_dir must be absolute"
        )
    return {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "transaction_id": transaction,
        "event_dir": str(directory.resolve()),
    }


def require_launch_transaction_commit_event(
    event_path: Path,
    *,
    transaction_dir: Path,
    target_launch_state_path: Path,
    target_registry_path: Path,
    expected_event_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable COMMIT structurally before use or recovery."""

    directory = Path(transaction_dir).expanduser()
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        _fail("transaction event directory is invalid")
    directory = directory.resolve()
    path, event_encoded = _read_regular_bytes(
        Path(event_path),
        label="launch transaction commit event",
    )
    if path.parent != directory:
        _fail("launch transaction commit is outside its event directory")
    if expected_event_sha256 is not None:
        expected_event_sha = _sha(
            expected_event_sha256,
            label="launch transaction commit event sha256",
        )
        if hashlib.sha256(event_encoded).hexdigest() != expected_event_sha:
            _fail("launch transaction commit event byte hash mismatch")
    try:
        raw = json.loads(event_encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] commit event is unreadable"
        ) from exc
    event = _exact_mapping(raw, _EVENT_KEYS, label="commit event")
    transaction_id = str(event.get("transaction_id") or "")
    vedtak_id = str(event.get("vedtak_id") or "")
    if (
        event.get("schema_version") != SCHEMA_VERSION
        or event.get("decision") != "COMMIT"
        or event.get("failures") != []
        or event.get("project") != PROJECT
        or _ID_RE.fullmatch(transaction_id) is None
        or _ID_RE.fullmatch(vedtak_id) is None
        or Path(str(event.get("json_path") or "")).expanduser().resolve() != path
    ):
        _fail("launch transaction commit authority fields are invalid")
    launch_target = _absolute_target(
        target_launch_state_path,
        label="launch-state target",
    )
    registry_target = _absolute_target(
        target_registry_path,
        label="registry target",
    )
    if (
        Path(str(event.get("target_launch_state_path") or ""))
        .expanduser()
        .resolve()
        != launch_target
        or Path(str(event.get("target_registry_path") or ""))
        .expanduser()
        .resolve()
        != registry_target
    ):
        _fail("commit target paths differ from configured authority paths")
    _sha(event.get("launch_state_sha256"), label="commit launch_state_sha256")
    _sha(event.get("registry_sha256"), label="commit registry_sha256")
    _backup(
        event.get("launch_state_before_artifact"),
        label="launch backup",
        event_dir=directory,
        expected_name=(
            f"ENTRY_MODEL_NATIVE_LAUNCH_BACKUP_{transaction_id}_state.json"
        ),
    )
    _backup(
        event.get("registry_before_artifact"),
        label="registry backup",
        event_dir=directory,
        expected_name=(
            f"ENTRY_MODEL_NATIVE_LAUNCH_BACKUP_{transaction_id}_registry.json"
        ),
    )
    bundle = Path(str(event.get("accepted_bundle_dir") or "")).expanduser()
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_dir():
        _fail("commit accepted bundle path is invalid")
    commit = require_bundle_commit_manifest(bundle.resolve())
    if event.get("bundle_commit_sha256") != commit.get("commit_sha256"):
        _fail("commit bundle inventory mismatch")
    authority = _exact_mapping(
        event.get("vedtak_authority"),
        _SOURCE_BINDING_KEYS,
        label="commit vedtak authority",
    )
    authority_path = _absolute_regular(
        Path(str(authority.get("json_path") or "")),
        label="commit vedtak authority event",
    )
    authority_sha = _sha(
        authority.get("sha256"),
        label="commit vedtak authority sha256",
    )
    _authority_path, authority_encoded = _read_regular_bytes(
        authority_path,
        label="commit vedtak authority event",
    )
    if hashlib.sha256(authority_encoded).hexdigest() != authority_sha:
        _fail("commit vedtak authority byte hash mismatch")
    try:
        authority_event = json.loads(authority_encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] commit vedtak authority is unreadable"
        ) from exc
    if not isinstance(authority_event, Mapping) or not isinstance(
        authority_event.get("launch_request"),
        Mapping,
    ):
        _fail("commit vedtak authority request is invalid")
    request = dict(authority_event["launch_request"])
    vedtak = require_historical_launch_vedtak(
        authority_path,
        expected_request=request,
    )
    if (
        vedtak.get("vedtak_id") != vedtak_id
        or request.get("transaction_id") != transaction_id
        or Path(str(request.get("accepted_bundle_dir") or ""))
        .expanduser()
        .resolve()
        != bundle.resolve()
        or request.get("bundle_commit_sha256") != event.get("bundle_commit_sha256")
        or Path(str(request.get("target_registry_path") or ""))
        .expanduser()
        .resolve()
        != registry_target
        or Path(str(request.get("target_launch_state_path") or ""))
        .expanduser()
        .resolve()
        != launch_target
    ):
        _fail("commit vedtak authority does not bind this transaction")
    return event


def require_entry_launch_transaction(
    launch_state: Mapping[str, Any],
    *,
    launch_state_bytes_path: Path,
    registry_bytes_path: Path,
    target_launch_state_path: Path,
    target_registry_path: Path,
    accepted_bundle: Path,
    expected_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one newest commit against both exact launch-authority files."""

    state = _exact_mapping(
        launch_state,
        _LAUNCH_STATE_KEYS,
        label="launch state",
    )
    if (
        state.get("schema_version") != "gx1_xau_direction_launch_state_v1"
        or state.get("project") != PROJECT
        or state.get("decision") != "ALLOW"
        or state.get("blockers") != []
    ):
        _fail("launch state is not an exact unblocked ALLOW authority")
    try:
        live_tail_authority = require_live_tail_launch_authority(
            state.get("new_entry_live_tail_authority")
        )
    except LiveTailAuthorityError as exc:
        _fail(f"new-Entry live-tail authority is invalid: {exc}")
    declaration = _exact_mapping(
        state.get("launch_transaction"),
        _DECLARATION_KEYS,
        label="launch_transaction",
    )
    if declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        _fail("launch_transaction schema mismatch")
    transaction_id = str(declaration.get("transaction_id") or "")
    if _ID_RE.fullmatch(transaction_id) is None:
        _fail("launch_transaction transaction_id is invalid")
    event_dir = Path(str(declaration.get("event_dir") or "")).expanduser()
    if (
        not event_dir.is_absolute()
        or event_dir.is_symlink()
        or not event_dir.is_dir()
    ):
        _fail("launch_transaction event_dir is unavailable")
    event_dir = event_dir.resolve()
    try:
        event_path = select_latest_immutable_event(event_dir, EVENT_PREFIX)
    except ImmutableEventAuthorityError as exc:
        raise EntryLaunchTransactionError(
            f"[ENTRY_LAUNCH_TRANSACTION_INVALID] event authority invalid: {exc}"
        ) from exc
    if event_path is None:
        _fail("launch transaction has no immutable event")
    event = require_launch_transaction_commit_event(
        event_path,
        transaction_dir=event_dir,
        target_launch_state_path=target_launch_state_path,
        target_registry_path=target_registry_path,
    )
    if event.get("transaction_id") != transaction_id:
        _fail("newest launch transaction is not the exact COMMIT authority")

    _launch_bytes_path, launch_encoded = _read_regular_bytes(
        launch_state_bytes_path,
        label="launch-state bytes",
    )
    _registry_bytes_path, registry_encoded = _read_regular_bytes(
        registry_bytes_path,
        label="registry bytes",
    )
    launch_target = _absolute_target(
        target_launch_state_path,
        label="launch-state target",
    )
    registry_target = _absolute_target(
        target_registry_path,
        label="registry target",
    )
    launch_sha = _sha(
        event.get("launch_state_sha256"),
        label="commit launch_state_sha256",
    )
    registry_sha = _sha(
        event.get("registry_sha256"),
        label="commit registry_sha256",
    )
    if hashlib.sha256(launch_encoded).hexdigest() != launch_sha:
        _fail("launch-state bytes differ from transaction commit")
    if hashlib.sha256(registry_encoded).hexdigest() != registry_sha:
        _fail("registry bytes differ from transaction commit")
    try:
        launch_from_bytes = json.loads(launch_encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] launch state is unreadable"
        ) from exc
    if launch_from_bytes != state:
        _fail("validated launch state differs from exact committed bytes")
    bundle = Path(accepted_bundle).expanduser()
    if (
        not bundle.is_absolute()
        or bundle.is_symlink()
        or not bundle.is_dir()
        or Path(str(event.get("accepted_bundle_dir") or ""))
        .expanduser()
        .resolve()
        != bundle.resolve()
    ):
        _fail("commit accepted bundle mismatch")
    bundle = bundle.resolve()
    commit = require_bundle_commit_manifest(bundle)
    if event.get("bundle_commit_sha256") != commit.get("commit_sha256"):
        _fail("commit bundle inventory mismatch")

    approval = state.get("accepted_via_vedtak")
    if (
        not isinstance(approval, Mapping)
        or event.get("vedtak_id") != approval.get("vedtak_id")
    ):
        _fail("commit vedtak differs from launch approval")

    try:
        registry_raw = json.loads(registry_encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchTransactionError(
            "[ENTRY_LAUNCH_TRANSACTION_INVALID] registry is unreadable"
        ) from exc
    if not isinstance(registry_raw, Mapping):
        _fail("registry root is invalid")
    registry = dict(registry_raw)
    if expected_registry is not None and registry != dict(expected_registry):
        _fail("caller registry object differs from exact committed bytes")
    active = registry.get("active")
    if (
        registry.get("schema_version") != "gx1_artifact_selection_v2"
        or registry.get("project") != PROJECT
        or not isinstance(active, Mapping)
    ):
        _fail("registry is not the exact XAUUSD authority")
    entry = _exact_mapping(
        active.get("v10_entry"),
        _V10_ENTRY_KEYS,
        label="registry active.v10_entry",
    )
    if (
        Path(str(entry.get("path") or "")).expanduser().resolve() != bundle
        or entry.get("status") != "ACTIVE"
        or entry.get("in_sample_only") is not False
        or entry.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
        or entry.get("launch_transaction_id") != transaction_id
        or entry.get("bundle_commit_sha256") != commit.get("commit_sha256")
        or entry.get("vedtak") != event.get("vedtak_id")
    ):
        _fail("registry active.v10_entry does not match transaction")
    operating_point = require_model_direction_operating_point(
        entry.get("operating_point"),
        context="entry launch transaction registry",
    )
    state_operating_point = require_model_direction_operating_point(
        state.get("operating_point"),
        context="entry launch transaction state",
    )
    if operating_point != state_operating_point:
        _fail("registry operating point differs from launch state")
    vedtak_binding = _exact_mapping(
        approval.get("vedtak_authority"),
        _SOURCE_BINDING_KEYS,
        label="launch vedtak authority",
    )
    if event.get("vedtak_authority") != vedtak_binding:
        _fail("commit vedtak authority differs from launch approval")
    request = launch_vedtak_request(
        transaction_id=transaction_id,
        accepted_bundle_dir=bundle,
        bundle_commit_sha256=str(commit.get("commit_sha256") or ""),
        target_registry_path=registry_target,
        target_launch_state_path=launch_target,
        operating_point=operating_point,
        evidence={
            "sizing_adoption": state["sizing_authority_contract"][
                "adoption_artifact"
            ],
            "joint_exit_sizing_proof": state[
                "joint_exit_execution_proof_evidence"
            ],
            "sizing_runtime_parity": state["sizing_runtime_parity_evidence"],
            "model_native_serve_parity": state["serve_gate_evidence"][
                "model_native_serve_parity"
            ],
            "model_native_direction_pocket_audit": state[
                "serve_gate_evidence"
            ]["model_native_direction_pocket_audit"],
            "adaptation_lifecycle": state["adaptation_lifecycle_evidence"],
            "live_tail_admission": live_tail_authority[
                "launch_admission"
            ],
        },
    )
    vedtak = require_preexisting_launch_vedtak(
        Path(str(vedtak_binding["json_path"])),
        expected_request=request,
    )
    if (
        vedtak["vedtak_id"] != event.get("vedtak_id")
        or vedtak["event_sha256"] != vedtak_binding["sha256"]
    ):
        _fail("launch vedtak authority identity mismatch")
    return event


__all__ = [
    "DECLARATION_SCHEMA_VERSION",
    "EVENT_PREFIX",
    "PROJECT",
    "SCHEMA_VERSION",
    "EntryLaunchTransactionError",
    "launch_transaction_declaration",
    "require_entry_launch_transaction",
    "require_launch_transaction_commit_event",
    "sha256_file",
]
