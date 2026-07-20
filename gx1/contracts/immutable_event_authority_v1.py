"""Fail-closed authority selection for timestamped immutable JSON events.

Filesystem mtimes and mutable ``*_latest.json`` mirrors are never authority.
The selected event is the unique newest event identified by its exact filename
timestamp after the payload's ``created_utc`` and ``json_path`` self-reference
have been verified.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "immutable_json_event_authority_v1"
_STAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}(?:\d{6})?Z)")


class ImmutableEventAuthorityError(RuntimeError):
    """Raised when an event inventory cannot establish unique authority."""


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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


def _validate_event(path: Path, *, event_prefix: str) -> datetime:
    if path.is_symlink() or not path.is_file():
        raise ImmutableEventAuthorityError(f"immutable event is not a regular file: {path}")
    filename_time, has_microseconds = _event_time_from_name(path, event_prefix=event_prefix)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutableEventAuthorityError(f"invalid immutable event JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ImmutableEventAuthorityError(f"immutable event root is not an object: {path}")

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
    return filename_time


def select_latest_immutable_event(
    root: Path,
    event_prefix: str,
    *,
    scope_dir_glob: str | None = None,
) -> Path | None:
    """Return the unique newest valid immutable event below ``root``.

    ``scope_dir_glob`` restricts the inventory to matching direct child
    directories (for example ``smart_seq520_candidate_*``). Mutable latest
    mirrors and the conventional ``<prefix>_MANIFEST.json`` are excluded.
    Every prefix-matching filename must have an exact timestamp. The unique
    newest event must then satisfy the complete payload contract. Historical
    payloads are not re-authorized here; they are outside the selection result.
    A malformed filename, malformed newest payload, or duplicate newest
    timestamp raises ``ImmutableEventAuthorityError``.
    """

    root = Path(root)
    if not root.exists():
        return None
    if not root.is_dir():
        raise ImmutableEventAuthorityError(f"event inventory root is not a directory: {root}")

    if scope_dir_glob is None:
        scopes = [root]
    else:
        scopes = sorted(path for path in root.glob(scope_dir_glob) if path.is_dir())
        if not scopes:
            return None

    latest_name = f"{event_prefix}_latest.json"
    manifest_name = f"{event_prefix}_MANIFEST.json"
    rows: list[tuple[datetime, Path]] = []
    seen_paths: set[Path] = set()
    for scope in scopes:
        for path in sorted(scope.rglob(f"{event_prefix}_*.json")):
            if path.name in {latest_name, manifest_name}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ImmutableEventAuthorityError(
                    f"immutable event is not a regular file: {path}"
                )
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            event_time, _ = _event_time_from_name(path, event_prefix=event_prefix)
            rows.append((event_time, resolved))

    if not rows:
        return None

    newest_time = max(event_time for event_time, _ in rows)
    newest_paths = [path for event_time, path in rows if event_time == newest_time]
    if len(newest_paths) != 1:
        details = ", ".join(str(path) for path in sorted(newest_paths))
        raise ImmutableEventAuthorityError(
            "duplicate newest immutable event timestamp ambiguity: "
            f"{newest_time.isoformat()}=[{details}]"
        )

    selected = newest_paths[0]
    validated_time = _validate_event(selected, event_prefix=event_prefix)
    if validated_time != newest_time:
        raise ImmutableEventAuthorityError(
            f"selected event timestamp changed during validation: {selected}"
        )
    return selected


def require_newest_immutable_event(event_path: Path, event_prefix: str) -> Path:
    """Require ``event_path`` to be the unique newest event in its directory."""

    requested = Path(event_path).expanduser().absolute()
    if requested.is_symlink() or not requested.is_file():
        raise ImmutableEventAuthorityError(
            f"requested immutable event is not a regular file: {requested}"
        )
    event_path = requested.resolve()
    newest = select_latest_immutable_event(event_path.parent, event_prefix)
    if newest is None:
        raise ImmutableEventAuthorityError(
            f"no immutable {event_prefix} event under {event_path.parent}"
        )
    if newest != event_path:
        raise ImmutableEventAuthorityError(
            f"event is not the newest immutable {event_prefix} authority: "
            f"requested={event_path} newest={newest}"
        )
    return newest


def next_immutable_event_created_utc(
    root: Path,
    event_prefix: str,
    *floors: datetime,
) -> datetime:
    """Return a UTC time strictly newer than clock, floors and event inventory."""

    root = Path(root).expanduser().resolve()
    created = datetime.now(timezone.utc)
    for floor in floors:
        if floor.tzinfo is None:
            raise ImmutableEventAuthorityError("event time floor is not timezone-aware")
        observed = floor.astimezone(timezone.utc)
        if created <= observed:
            created = observed + timedelta(microseconds=1)
    if root.exists():
        for path in root.glob(f"{event_prefix}_*.json"):
            if path.is_symlink() or not path.is_file():
                raise ImmutableEventAuthorityError(
                    f"invalid immutable event inventory member: {path}"
                )
            observed, _ = _event_time_from_name(path, event_prefix=event_prefix)
            if created <= observed:
                created = observed + timedelta(microseconds=1)
    return created


def write_immutable_json_event(
    root: Path,
    event_prefix: str,
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Atomically publish one no-replace JSON event with exact self-identity.

    ``payload.created_utc`` is the event time and therefore determines the
    microsecond-resolution filename. The returned payload is a copy containing
    the authoritative absolute ``json_path``. Existing events are never
    replaced, including on timestamp collision.
    """

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ImmutableEventAuthorityError(
            f"immutable event already exists for timestamp {stamp}: {path}"
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write while publishing immutable event: {path}")
            view = view[written:]
        os.fsync(fd)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
    finally:
        os.close(fd)
    _fsync_directory(root)
    return path, event
