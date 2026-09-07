"""Fail-closed publication authority for timestamped immutable JSON events.

UTC timestamps identify events, not their generation order. The writer retains
the caller's timestamp and payload schema, and durably publishes an immutable
``.<event filename>.order`` witness before the event. A witness binds the event
bytes and the complete preceding recursive inventory under its authority-root
lock. Writer and reader must explicitly agree on that root and child-directory
glob; an event's output directory does not imply an ancestor authority scope.
Readers require a unique witnessed successor whenever more than one event is
present; legacy timestamps, mtimes and mutable mirrors cannot prove that order.
Order witnesses and their event histories must be retained and copied together.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "immutable_json_event_authority_v1"
PUBLICATION_ORDER_SCHEMA_VERSION = "immutable_json_event_publication_order_v1"
INVENTORY_SCHEMA_VERSION = "immutable_json_event_authority_inventory_v1"
_STAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}(?:\d{6})?Z)")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ORDER_KEYS = frozenset(
    {
        "schema_version",
        "json_path",
        "sha256",
        "authority_root",
        "scope_dir_glob",
        "predecessor_count",
        "predecessors_sha256",
    }
)


class ImmutableEventAuthorityError(RuntimeError):
    """Raised when an event inventory cannot establish unique authority."""


def _require_event_prefix(event_prefix: str) -> None:
    if not isinstance(event_prefix, str) or _PREFIX_RE.fullmatch(event_prefix) is None:
        raise ImmutableEventAuthorityError("immutable event prefix is invalid")


def _canonical_directory(path: Path) -> Path:
    requested = Path(path).expanduser().absolute()
    if any(parent.is_symlink() for parent in (requested, *requested.parents)):
        raise ImmutableEventAuthorityError(f"immutable event directory contains a symlink: {requested}")
    resolved = requested.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ImmutableEventAuthorityError(f"event inventory root is not a directory: {resolved}")
    return resolved


def _require_scope_glob(scope_dir_glob: str | None) -> None:
    if scope_dir_glob is not None and (
        not isinstance(scope_dir_glob, str)
        or not scope_dir_glob
        or scope_dir_glob in {".", ".."}
        or "/" in scope_dir_glob
        or "\\" in scope_dir_glob
        or "**" in scope_dir_glob
    ):
        raise ImmutableEventAuthorityError("event scope glob must match direct child directory names")


def _require_output_scope(output_dir: Path, authority_root: Path, scope_dir_glob: str | None) -> None:
    try:
        relative = output_dir.relative_to(authority_root)
    except ValueError as exc:
        raise ImmutableEventAuthorityError("event output is outside its authority scope") from exc
    if scope_dir_glob is not None and (
        not relative.parts or not fnmatch.fnmatchcase(relative.parts[0], scope_dir_glob)
    ):
        raise ImmutableEventAuthorityError("event output is outside its authority scope glob")


def _mkdir_durable(path: Path) -> None:
    missing = []
    directory = path
    while not directory.exists():
        missing.append(directory)
        directory = directory.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _canonical_directory(directory)
        _fsync_directory(directory.parent)


@contextmanager
def _locked_directory(root: Path, *, exclusive: bool) -> Iterator[None]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ImmutableEventAuthorityError(f"cannot lock immutable event directory: {root}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError as exc:
            raise ImmutableEventAuthorityError(f"cannot lock immutable event directory: {root}") from exc
        yield
    finally:
        os.close(descriptor)


def _order_path(event_path: Path) -> Path:
    return event_path.with_name(f".{event_path.name}.order")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value: {value}")
    return parsed


def _read_json_object(path: Path, *, max_bytes: int | None = None) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ImmutableEventAuthorityError(f"immutable event is not a regular file: {path}")
    try:
        if max_bytes is None:
            encoded = path.read_bytes()
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                observed = os.fstat(handle.fileno())
                if not stat.S_ISREG(observed.st_mode) or observed.st_size > max_bytes:
                    raise ImmutableEventAuthorityError(f"immutable document exceeds bounded read or changed type: {path}")
                encoded = handle.read(max_bytes + 1)
            if len(encoded) > max_bytes:
                raise ImmutableEventAuthorityError(f"immutable document exceeds bounded read: {path}")
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ImmutableEventAuthorityError(f"invalid immutable event JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ImmutableEventAuthorityError(f"immutable event root is not an object: {path}")
    return payload, encoded


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_file_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ImmutableEventAuthorityError(
            "atomic no-replace immutable event publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    ) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ImmutableEventAuthorityError(
                f"immutable event already exists: {destination}"
            )
        raise ImmutableEventAuthorityError(
            "atomic no-replace immutable event publication failed: "
            f"{os.strerror(code)}"
        )


def _event_time_from_name(path: Path, *, event_prefix: str) -> tuple[datetime, bool]:
    prefix = f"{event_prefix}_"
    suffix = ".json"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ImmutableEventAuthorityError(f"unexpected event filename: {path}")
    token = path.name[len(prefix) : -len(suffix)]
    match = _STAMP_RE.fullmatch(token)
    if match is None:
        raise ImmutableEventAuthorityError(
            f"immutable event filename lacks an exact UTC timestamp: {path}"
        )
    stamp = match.group("stamp")
    has_microseconds = len(stamp) == 22
    fmt = "%Y%m%dT%H%M%S%fZ" if has_microseconds else "%Y%m%dT%H%M%SZ"
    try:
        parsed = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ImmutableEventAuthorityError(f"invalid event timestamp in {path}") from exc
    return parsed, has_microseconds


def _created_utc(payload: dict[str, Any], *, path: Path) -> datetime:
    raw = payload.get("created_utc")
    if not isinstance(raw, str) or not raw.strip():
        raise ImmutableEventAuthorityError(f"immutable event lacks created_utc: {path}")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ImmutableEventAuthorityError(f"invalid created_utc in {path}: {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ImmutableEventAuthorityError(f"created_utc is not timezone-aware: {path}")
    return parsed.astimezone(timezone.utc)


def _validate_event(
    path: Path, *, event_prefix: str, max_bytes: int | None = None
) -> tuple[datetime, str]:
    filename_time, has_microseconds = _event_time_from_name(path, event_prefix=event_prefix)
    payload, encoded = _read_json_object(path, max_bytes=max_bytes)
    created = _created_utc(payload, path=path)
    comparable_created = created if has_microseconds else created.replace(microsecond=0)
    if comparable_created != filename_time:
        raise ImmutableEventAuthorityError(
            "filename timestamp does not match created_utc: "
            f"path={path} filename={filename_time.isoformat()} created={created.isoformat()}"
        )

    declared_raw = payload.get("json_path")
    if not isinstance(declared_raw, str) or not declared_raw.strip():
        raise ImmutableEventAuthorityError(f"immutable event lacks json_path: {path}")
    declared = Path(declared_raw).expanduser()
    if not declared.is_absolute() or declared.resolve() != path.resolve():
        raise ImmutableEventAuthorityError(
            f"immutable event json_path is not an exact self-reference: path={path} declared={declared}"
        )
    return created, hashlib.sha256(encoded).hexdigest()


def _inventory_paths(
    root: Path, event_prefix: str, *, scope_dir_glob: str | None
) -> tuple[list[Path], list[Path]]:
    excluded = {f"{event_prefix}_latest.json", f"{event_prefix}_MANIFEST.json"}
    events: set[Path] = set()
    orders: set[Path] = set()
    scopes = [root]
    if scope_dir_glob is not None:
        scopes = []
        for path in root.iterdir():
            if fnmatch.fnmatchcase(path.name, scope_dir_glob):
                if path.is_symlink():
                    raise ImmutableEventAuthorityError(f"immutable event scope is a symlink: {path}")
                if path.is_dir():
                    scopes.append(path)

    def scan_error(error: OSError) -> None:
        raise ImmutableEventAuthorityError(f"cannot scan immutable event authority scope: {error}") from error

    for scope in scopes:
        for directory, child_dirs, filenames in os.walk(scope, followlinks=False, onerror=scan_error):
            parent = Path(directory)
            for child in child_dirs:
                if (parent / child).is_symlink():
                    raise ImmutableEventAuthorityError(
                        f"immutable event scope contains a directory symlink: {parent / child}"
                    )
            for name in filenames:
                if name.startswith(f"{event_prefix}_") and name.endswith(".json") and name not in excluded:
                    events.add(parent / name)
                if name.startswith(f".{event_prefix}_") and name.endswith(".json.order"):
                    orders.add(parent / name)
            for name in child_dirs:
                if name.startswith(f"{event_prefix}_") and name.endswith(".json") and name not in excluded:
                    raise ImmutableEventAuthorityError(f"immutable event is not a regular file: {parent / name}")
                if name.startswith(f".{event_prefix}_") and name.endswith(".json.order"):
                    raise ImmutableEventAuthorityError(f"publication order witness is not a regular file: {parent / name}")
    return sorted(events), sorted(orders)


def _inventory_sha256(events: dict[Path, tuple[datetime, str]]) -> str:
    bindings = [[str(path), events[path][1]] for path in sorted(events)]
    encoded = json.dumps(bindings, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inventory_document_sizes(
    paths: tuple[list[Path], list[Path]],
    *,
    max_document_bytes: int | None,
    max_total_bytes: int | None,
    max_events: int | None,
) -> dict[Path, int]:
    event_paths, witness_paths = paths
    if max_events is not None and len(event_paths) > max_events:
        raise ImmutableEventAuthorityError("immutable authority inventory exceeds max_events")
    sizes: dict[Path, int] = {}
    total = 0
    for path in (*event_paths, *witness_paths):
        try:
            observed = path.lstat()
        except OSError as exc:
            raise ImmutableEventAuthorityError(f"cannot stat immutable authority document: {path}") from exc
        if not stat.S_ISREG(observed.st_mode):
            raise ImmutableEventAuthorityError(f"immutable authority document is not a regular file: {path}")
        if max_document_bytes is not None and observed.st_size > max_document_bytes:
            raise ImmutableEventAuthorityError(f"immutable authority document exceeds max_document_bytes: {path}")
        sizes[path] = observed.st_size
        total += observed.st_size
        if max_total_bytes is not None and total > max_total_bytes:
            raise ImmutableEventAuthorityError("immutable authority inventory exceeds max_total_bytes")
    return sizes


def _read_inventory(
    paths: tuple[list[Path], list[Path]],
    event_prefix: str,
    *,
    authority_root: Path,
    scope_dir_glob: str | None,
    document_sizes: dict[Path, int] | None = None,
) -> tuple[dict[Path, tuple[datetime, str]], dict[Path, dict[str, Any]]]:
    event_paths, order_paths = paths
    events = {
        path: _validate_event(
            path, event_prefix=event_prefix,
            max_bytes=document_sizes[path] if document_sizes is not None else None,
        ) for path in event_paths
    }
    orders: dict[Path, dict[str, Any]] = {}
    for path in order_paths:
        event_path = path.with_name(path.name[1:-len(".order")])
        _event_time_from_name(event_path, event_prefix=event_prefix)
        order, _ = _read_json_object(
            path, max_bytes=document_sizes[path] if document_sizes is not None else None
        )
        if (
            set(order) != _ORDER_KEYS
            or order.get("schema_version") != PUBLICATION_ORDER_SCHEMA_VERSION
            or order.get("json_path") != str(event_path)
            or type(order.get("predecessor_count")) is not int
            or order["predecessor_count"] < 0
            or not isinstance(order.get("sha256"), str)
            or _SHA_RE.fullmatch(order["sha256"]) is None
            or not isinstance(order.get("predecessors_sha256"), str)
            or _SHA_RE.fullmatch(order["predecessors_sha256"]) is None
        ):
            raise ImmutableEventAuthorityError(f"invalid immutable publication order witness: {path}")
        if order["authority_root"] != str(authority_root) or order["scope_dir_glob"] != scope_dir_glob:
            raise ImmutableEventAuthorityError(f"immutable publication authority scope mismatch: {path}")
        if event_path not in events:
            raise ImmutableEventAuthorityError(
                f"incomplete immutable publication; order witness has no event: {path}"
            )
        if order["sha256"] != events[event_path][1]:
            raise ImmutableEventAuthorityError(f"publication order event hash mismatch: {event_path}")
        orders[event_path] = order
    return events, orders


def _select_inventory(
    events: dict[Path, tuple[datetime, str]], orders: dict[Path, dict[str, Any]]
) -> Path | None:
    if not events:
        return None
    if len(events) == 1 and not orders:
        return next(iter(events))
    candidates = [
        path
        for path, order in orders.items()
        if order["predecessor_count"] == len(events) - 1
    ]
    if len(candidates) != 1:
        raise ImmutableEventAuthorityError(
            "immutable event publication order is unproven: "
            "require one witnessed successor covering the complete inventory; "
            "legacy timestamps and independent directories cannot establish generation order"
        )
    selected = candidates[0]
    predecessors = {path: value for path, value in events.items() if path != selected}
    if orders[selected]["predecessors_sha256"] != _inventory_sha256(predecessors):
        raise ImmutableEventAuthorityError(
            f"immutable publication predecessor inventory mismatch: {selected}"
        )
    if any(order["predecessor_count"] >= len(events) for order in orders.values()):
        raise ImmutableEventAuthorityError("immutable publication history is incomplete")
    return selected


def select_latest_immutable_event(
    root: Path,
    event_prefix: str,
    *,
    scope_dir_glob: str | None = None,
) -> Path | None:
    """Return the unique proven last-published immutable event below ``root``.

    ``scope_dir_glob`` restricts the inventory to matching direct child
    directories (for example ``smart_seq520_candidate_*``). Mutable latest
    mirrors and the conventional ``<prefix>_MANIFEST.json`` are excluded.
    All event identities and publication witnesses are checked. A sole legacy
    event remains readable; multiple legacy events or independent directory
    histories fail closed because wall time cannot prove generation order.
    A fresh owner-written successor may bind a validated, unordered legacy
    inventory without asserting an order among those historical events.
    Incomplete publication, changed history and malformed evidence block.
    """

    selected, _ = _validated_scope_inventory(root, event_prefix, scope_dir_glob=scope_dir_glob)
    return selected


def _validated_scope_inventory(
    root: Path, event_prefix: str, *, scope_dir_glob: str | None,
    max_document_bytes: int | None = None,
    max_total_bytes: int | None = None,
    max_events: int | None = None,
) -> tuple[Path | None, tuple[list[Path], list[Path]]]:
    _require_event_prefix(event_prefix)
    _require_scope_glob(scope_dir_glob)
    root = _canonical_directory(root)
    if not root.exists():
        return None, ([], [])
    with _locked_directory(root, exclusive=False):
        paths = _inventory_paths(root, event_prefix, scope_dir_glob=scope_dir_glob)
        limits = {
            "max_document_bytes": max_document_bytes,
            "max_total_bytes": max_total_bytes,
            "max_events": max_events,
        }
        document_sizes = (
            _inventory_document_sizes(paths, **limits)
            if any(value is not None for value in limits.values()) else None
        )
        events, orders = _read_inventory(
            paths, event_prefix, authority_root=root, scope_dir_glob=scope_dir_glob,
            document_sizes=document_sizes,
        )
        selected = _select_inventory(events, orders)
        if paths != _inventory_paths(root, event_prefix, scope_dir_glob=scope_dir_glob):
            raise ImmutableEventAuthorityError("immutable event inventory changed while validating")
        if document_sizes is not None and document_sizes != _inventory_document_sizes(paths, **limits):
            raise ImmutableEventAuthorityError("immutable document sizes changed while validating")
        return selected, paths


def require_newest_immutable_event(
    event_path: Path,
    event_prefix: str,
    *,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
) -> Path:
    """Require newest authority in the caller's exact recursive publication scope.

    The default root is the event's directory. Shared cross-run scope must be
    provided explicitly by the consuming contract, never inferred from a
    witness or from a common ancestor.
    """

    requested = Path(event_path).expanduser().absolute()
    if requested.is_symlink() or not requested.is_file():
        raise ImmutableEventAuthorityError(
            f"requested immutable event is not a regular file: {requested}"
        )
    event_path = _canonical_directory(requested.parent) / requested.name
    root = _canonical_directory(authority_root if authority_root is not None else event_path.parent)
    _require_scope_glob(scope_dir_glob)
    _require_output_scope(event_path.parent, root, scope_dir_glob)
    newest = select_latest_immutable_event(root, event_prefix, scope_dir_glob=scope_dir_glob)
    if newest is None:
        raise ImmutableEventAuthorityError(
            f"no immutable {event_prefix} event under {root}"
        )
    if newest != event_path:
        raise ImmutableEventAuthorityError(
            f"event is not the newest immutable {event_prefix} authority: "
            f"requested={event_path} newest={newest}"
        )
    return newest


def validated_immutable_event_authority_inventory(
    event_path: Path,
    event_prefix: str,
    *,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
    max_document_bytes: int | None = None,
    max_total_bytes: int | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Inspect validated scope paths without granting admission or cleanup.

    This read-only helper may discover scope from the referenced event's own
    witness, then validates the complete scope with the same reader contract.
    Unwitnessed legacy references require an explicit caller-owned scope; no
    ancestor search or order reconstruction is attempted. The referenced event
    need not be newest, so historical dependencies retain their entire scope.
    Admission consumers must instead use ``require_newest_immutable_event``
    with their own expected scope. This snapshot does not authorize deletion.
    Optional caller-owned nonnegative integer limits count all event and witness
    bytes, with ``max_events`` counting only events. Scope sizes are checked
    before JSON decoding, and reads are bounded by those sizes against growth.
    Discovering scope first requires the referenced event's own witness; that
    bootstrap read also obeys the byte limits. Limits default to unbounded and
    introduce no retention policy or numeric budget in this owner.
    """

    requested = Path(event_path).expanduser().absolute()
    _require_event_prefix(event_prefix)
    limits = {
        "max_document_bytes": max_document_bytes,
        "max_total_bytes": max_total_bytes,
        "max_events": max_events,
    }
    for name, value in limits.items():
        if value is not None and (type(value) is not int or value < 0):
            raise ImmutableEventAuthorityError(f"{name} must be a nonnegative integer or None")
    if max_events == 0:
        raise ImmutableEventAuthorityError("immutable authority inventory exceeds max_events")
    if requested.is_symlink() or not requested.is_file():
        raise ImmutableEventAuthorityError(f"immutable event is not a regular file: {requested}")
    _event_time_from_name(requested, event_prefix=event_prefix)
    requested = _canonical_directory(requested.parent) / requested.name
    if authority_root is None:
        if scope_dir_glob is not None:
            raise ImmutableEventAuthorityError("an explicit scope glob requires an authority root")
        witness_path = _order_path(requested)
        if not witness_path.exists() and not witness_path.is_symlink():
            raise ImmutableEventAuthorityError("legacy event has no declared authority scope")
        witness_sizes = _inventory_document_sizes(
            ([], [witness_path]), max_document_bytes=max_document_bytes,
            max_total_bytes=max_total_bytes, max_events=None,
        )
        witness, _ = _read_json_object(witness_path, max_bytes=witness_sizes[witness_path])
        declared_root = witness.get("authority_root")
        if not isinstance(declared_root, str) or not Path(declared_root).is_absolute():
            raise ImmutableEventAuthorityError("publication witness has no absolute authority root")
        authority_root = Path(declared_root)
        scope_dir_glob = witness.get("scope_dir_glob")
    root = _canonical_directory(authority_root)
    _require_scope_glob(scope_dir_glob)
    _require_output_scope(requested.parent, root, scope_dir_glob)
    selected, (event_paths, witness_paths) = _validated_scope_inventory(
        root, event_prefix, scope_dir_glob=scope_dir_glob, **limits
    )
    if selected is None or requested not in event_paths:
        raise ImmutableEventAuthorityError("referenced immutable event is absent from its authority scope")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "authority_root": str(root),
        "event_prefix": event_prefix,
        "scope_dir_glob": scope_dir_glob,
        "selected_event_path": str(selected),
        "event_paths": [str(path) for path in event_paths],
        "witness_paths": [str(path) for path in witness_paths],
    }


def next_immutable_event_created_utc(
    root: Path,
    event_prefix: str,
    *floors: datetime,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
) -> datetime:
    """Return a UTC time newer than floors and the exact recursive inventory.

    This is an allocation hint, not a reservation across processes. The writer
    serializes publication and records its order independently of wall time;
    callers must still handle no-replace timestamp collisions.
    """

    _require_event_prefix(event_prefix)
    _require_scope_glob(scope_dir_glob)
    output_dir = _canonical_directory(root)
    root = _canonical_directory(authority_root if authority_root is not None else output_dir)
    _require_output_scope(output_dir, root, scope_dir_glob)
    created = datetime.now(timezone.utc)
    for floor in floors:
        if floor.tzinfo is None:
            raise ImmutableEventAuthorityError("event time floor is not timezone-aware")
        observed = floor.astimezone(timezone.utc)
        if created <= observed:
            created = observed + timedelta(microseconds=1)
    if root.exists():
        with _locked_directory(root, exclusive=False):
            paths = _inventory_paths(root, event_prefix, scope_dir_glob=scope_dir_glob)
            events, orders = _read_inventory(
                paths, event_prefix, authority_root=root, scope_dir_glob=scope_dir_glob
            )
            if orders:
                _select_inventory(events, orders)
            for observed, _ in events.values():
                if created <= observed:
                    created = observed + timedelta(microseconds=1)
    return created


def write_immutable_json_event(
    root: Path,
    event_prefix: str,
    payload: dict[str, Any],
    *,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish one no-replace JSON event with durable publication-order proof.

    ``payload.created_utc`` is the event time and therefore determines the
    microsecond-resolution filename, even after clock rollback. It is not
    silently rewritten into logical time. The returned payload is a copy
    containing the authoritative absolute ``json_path``; its schema is unchanged.
    Publication is serialized per authority root, surviving restart through
    immutable order witnesses rather than a mutable clock counter or pointer.
    The witness is durable before event publication; an interrupted publication
    blocks readers and further writes instead of exposing an older GREEN.
    Existing events are never replaced, including on timestamp collision.
    A fresh event can supersede a validated legacy inventory, without inventing
    its internal order. No automatic repair of an incomplete history is allowed.
    The default authority root is ``root`` with a recursive inventory. Writers
    in separate run directories must declare the same explicit authority root
    and exact ``scope_dir_glob`` as their consumers.
    """

    _require_event_prefix(event_prefix)
    _require_scope_glob(scope_dir_glob)
    root = _canonical_directory(root)
    authority = _canonical_directory(authority_root if authority_root is not None else root)
    _require_output_scope(root, authority, scope_dir_glob)
    if not isinstance(payload, dict):
        raise ImmutableEventAuthorityError("immutable event payload must be an object")
    event = dict(payload)
    created = _created_utc(event, path=root / f"{event_prefix}_<pending>.json")
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{event_prefix}_{stamp}.json"
    event["json_path"] = str(path)
    try:
        encoded = (
            json.dumps(
                event,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ImmutableEventAuthorityError(
            f"immutable event payload is not strict JSON: {exc}"
        ) from exc
    _mkdir_durable(authority)
    with _locked_directory(authority, exclusive=True):
        _mkdir_durable(root)
        if path.exists() or path.is_symlink():
            raise ImmutableEventAuthorityError(f"immutable event already exists: {path}")
        paths = _inventory_paths(authority, event_prefix, scope_dir_glob=scope_dir_glob)
        events, orders = _read_inventory(
            paths, event_prefix, authority_root=authority, scope_dir_glob=scope_dir_glob
        )
        if orders:
            _select_inventory(events, orders)
        order = {
            "schema_version": PUBLICATION_ORDER_SCHEMA_VERSION,
            "json_path": str(path),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "authority_root": str(authority),
            "scope_dir_glob": scope_dir_glob,
            "predecessor_count": len(events),
            "predecessors_sha256": _inventory_sha256(events),
        }
        order_encoded = (json.dumps(order, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _publish_bytes(_order_path(path), order_encoded)
        _publish_bytes(path, encoded)
    return path, event


def _publish_bytes(path: Path, encoded: bytes) -> None:
    root = path.parent
    fd, stage_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging.",
        dir=str(root),
    )
    stage_path = Path(stage_name)
    published = False
    try:
        os.fchmod(fd, 0o644)
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(
                    f"short write while staging immutable event: {stage_path}"
                )
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        _publish_file_noreplace(stage_path, path)
        published = True
        _fsync_directory(root)
    finally:
        if fd >= 0:
            os.close(fd)
        if not published:
            stage_path.unlink(missing_ok=True)
