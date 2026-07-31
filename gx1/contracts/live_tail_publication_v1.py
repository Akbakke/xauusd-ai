"""Immutable authority for continuously advancing XAU serving data.

One successful pair publication is source evidence, not proof that a live-tail
publisher remains healthy.  This contract records each strict pair successor
as an immutable publication event and admits serving only after two
consecutive, current, freshness-bounded events are revalidated against the
active pair pointer.

Process presence, mutable collector files, mtimes and retry loops have no
authority here.  A stale or structurally invalid event blocks new Entry
exposure; it never synthesizes a model decision.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    select_latest_immutable_event,
    write_immutable_json_event,
)
from gx1.execution.v12_state_from_prebuilt import (
    PAIR_PUBLISH_LOCK_FILENAME,
    PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
    read_prebuilt_pair_manifest,
    verify_prebuilt_pair,
)


LIVE_TAIL_PUBLICATION_SCHEMA_VERSION = "gx1_live_tail_publication_v1"
LIVE_TAIL_ADMISSION_SCHEMA_VERSION = "gx1_live_tail_admission_v1"
LIVE_TAIL_LAUNCH_AUTHORITY_SCHEMA_VERSION = (
    "gx1_live_tail_launch_authority_v1"
)
LIVE_TAIL_PUBLICATION_EVENT_PREFIX = "GX1_LIVE_TAIL_PUBLICATION"
LIVE_TAIL_ADMISSION_EVENT_PREFIX = "GX1_LIVE_TAIL_ADMISSION"
LIVE_TAIL_PUBLICATION_OWNER = (
    "gx1.execution.v12_canonical_incremental.publish_prebuilt_pair_successor"
)
LIVE_TAIL_ADMISSION_OWNER = (
    "gx1.contracts.live_tail_publication_v1."
    "publish_live_tail_admission_event"
)
LIVE_TAIL_M5_INTERVAL_SECONDS = 300
LIVE_TAIL_MAX_PUBLICATION_LATENCY_SECONDS = 90
LIVE_TAIL_VALIDITY_SECONDS = (
    LIVE_TAIL_M5_INTERVAL_SECONDS
    + LIVE_TAIL_MAX_PUBLICATION_LATENCY_SECONDS
)
LIVE_TAIL_MAX_CONSECUTIVE_EVENT_GAP_SECONDS = 600
_SHA256_HEX = frozenset("0123456789abcdef")
_EVENT_STAMP_RE = re.compile(
    r"^(?P<prefix>[A-Z0-9_]+)_"
    r"(?P<stamp>\d{8}T\d{12}Z)\.json$"
)


class LiveTailAuthorityError(RuntimeError):
    """Raised when immutable live-tail authority cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(raw: object, *, label: str) -> str:
    value = str(raw or "")
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise LiveTailAuthorityError(f"{label}_SHA256_INVALID")
    return value


def _read_exact_file(
    raw: object,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str, bytes]:
    path = Path(str(raw or "")).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise LiveTailAuthorityError(f"{label}_FILE_INVALID: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LiveTailAuthorityError(
            f"{label}_FILE_OPEN_FAILED: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise LiveTailAuthorityError(
                f"{label}_FILE_INVALID: {path}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
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
            raise LiveTailAuthorityError(
                f"{label}_FILE_CHANGED_DURING_READ"
            )
    finally:
        os.close(descriptor)
    path = path.resolve(strict=True)
    encoded = b"".join(chunks)
    observed = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, label=label)
        if observed != expected:
            raise LiveTailAuthorityError(f"{label}_SHA256_MISMATCH")
    return path, observed, encoded


def _require_exact_file(
    raw: object,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    path, digest, _encoded = _read_exact_file(
        raw,
        label=label,
        expected_sha256=expected_sha256,
    )
    return path, digest


def _utc(raw: object, *, label: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(pd.to_datetime(raw, errors="raise"))
    except Exception as exc:
        raise LiveTailAuthorityError(f"{label}_UTC_INVALID") from exc
    if pd.isna(value) or value.tzinfo is None:
        raise LiveTailAuthorityError(f"{label}_UTC_INVALID")
    return value.tz_convert("UTC")


def _created_utc(raw: pd.Timestamp | datetime | str | None) -> pd.Timestamp:
    if raw is None:
        return pd.Timestamp.now(tz="UTC")
    return _utc(raw, label="LIVE_TAIL_CREATED")


def _event_json(path: Path, *, expected_sha256: str | None) -> dict[str, Any]:
    if expected_sha256 is None:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_EVENT_EXPECTED_SHA256_REQUIRED"
        )
    exact_path, _digest, encoded = _read_exact_file(
        path,
        label="LIVE_TAIL_EVENT",
        expected_sha256=expected_sha256,
    )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise LiveTailAuthorityError("LIVE_TAIL_EVENT_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise LiveTailAuthorityError("LIVE_TAIL_EVENT_ROOT_INVALID")
    if payload.get("json_path") != str(exact_path):
        raise LiveTailAuthorityError("LIVE_TAIL_EVENT_SELF_PATH_MISMATCH")
    schema = payload.get("schema_version")
    if schema == LIVE_TAIL_PUBLICATION_SCHEMA_VERSION:
        expected_prefix = LIVE_TAIL_PUBLICATION_EVENT_PREFIX
    elif schema == LIVE_TAIL_ADMISSION_SCHEMA_VERSION:
        expected_prefix = LIVE_TAIL_ADMISSION_EVENT_PREFIX
    else:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_EVENT_SCHEMA_VERSION_INVALID"
        )
    match = _EVENT_STAMP_RE.fullmatch(exact_path.name)
    created = _utc(payload.get("created_utc"), label="LIVE_TAIL_EVENT_CREATED")
    if (
        match is None
        or match.group("prefix") != expected_prefix
        or created.strftime("%Y%m%dT%H%M%S%fZ")
        != match.group("stamp")
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_EVENT_FILENAME_TIME_MISMATCH"
        )
    return payload


def _artifact(path: Path) -> dict[str, str]:
    exact_path, digest = _require_exact_file(path, label="LIVE_TAIL_ARTIFACT")
    return {"path": str(exact_path), "sha256": digest}


def _pair_binding(
    *,
    pair_manifest_path: Path,
    generation_root: Path,
    require_active_pointer: bool,
) -> Any:
    binding = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    verify_prebuilt_pair(binding)
    if binding.generation_manifest_path is None:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_GENERATION_MANIFEST_REQUIRED"
        )
    if require_active_pointer:
        if (
            binding.manifest_path == binding.generation_manifest_path
            or binding.manifest_path.is_relative_to(
                Path(generation_root).resolve(strict=True)
            )
        ):
            raise LiveTailAuthorityError(
                "LIVE_TAIL_ACTIVE_PAIR_POINTER_REQUIRED"
            )
    elif binding.manifest_path != binding.generation_manifest_path:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_IMMUTABLE_GENERATION_MANIFEST_REQUIRED"
        )
    return binding


def _publication_failures(
    *,
    created: pd.Timestamp,
    binding: Any,
) -> tuple[list[str], dict[str, Any]]:
    lineage = binding.lineage
    native = lineage.get("native_sources")
    coverage = lineage.get("coverage")
    if not isinstance(native, dict) or not isinstance(coverage, dict):
        raise LiveTailAuthorityError("LIVE_TAIL_PAIR_LINEAGE_INVALID")
    m1 = native.get("m1")
    m5 = native.get("m5")
    if not isinstance(m1, dict) or not isinstance(m5, dict):
        raise LiveTailAuthorityError("LIVE_TAIL_NATIVE_LINEAGE_INVALID")

    latest_m1 = _utc(m1.get("time_max_utc"), label="LIVE_TAIL_NATIVE_M1_MAX")
    latest_m5 = _utc(m5.get("time_max_utc"), label="LIVE_TAIL_NATIVE_M5_MAX")
    canonical_m5 = _utc(
        coverage.get("canonical_time_max_utc"),
        label="LIVE_TAIL_CANONICAL_MAX",
    )
    decision_available = canonical_m5 + pd.Timedelta(minutes=5)
    expected_latest_closed = created.floor("5min") - pd.Timedelta(minutes=5)
    latency_seconds = float((created - decision_available).total_seconds())
    valid_until = decision_available + pd.Timedelta(
        seconds=LIVE_TAIL_VALIDITY_SECONDS
    )

    failures: list[str] = []
    if canonical_m5 != latest_m5:
        failures.append("canonical_m5_tail_does_not_equal_native_m5_tail")
    if latest_m1 != canonical_m5 + pd.Timedelta(minutes=4):
        failures.append(
            "native_m1_tail_is_not_exact_final_complete_m5_bucket"
        )
    if canonical_m5 != expected_latest_closed:
        failures.append("pair_does_not_cover_latest_complete_m5_at_publication")
    if latency_seconds < 0.0:
        failures.append("pair_decision_availability_is_in_the_future")
    elif latency_seconds > LIVE_TAIL_MAX_PUBLICATION_LATENCY_SECONDS:
        failures.append("pair_publication_latency_exceeds_contract")

    timing = {
        "latest_native_m1_bar_start_utc": latest_m1.isoformat(),
        "latest_native_m5_bar_start_utc": latest_m5.isoformat(),
        "canonical_m5_cutoff_utc": canonical_m5.isoformat(),
        "decision_available_utc": decision_available.isoformat(),
        "expected_latest_closed_m5_start_utc": (
            expected_latest_closed.isoformat()
        ),
        "publication_latency_seconds": latency_seconds,
        "max_publication_latency_seconds": (
            LIVE_TAIL_MAX_PUBLICATION_LATENCY_SECONDS
        ),
        "validity_seconds": LIVE_TAIL_VALIDITY_SECONDS,
        "valid_until_utc": valid_until.isoformat(),
    }
    return failures, timing


def publish_live_tail_publication_event(
    *,
    event_root: Path,
    pair_manifest_path: Path,
    generation_root: Path,
    previous_publication_json: Path | None = None,
    previous_publication_sha256: str | None = None,
    created_utc: pd.Timestamp | datetime | str | None = None,
    candidate_generation_manifest_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish immutable PASS/BLOCK evidence for the active pair successor."""

    if (previous_publication_json is None) != (
        previous_publication_sha256 is None
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PREVIOUS_PUBLICATION_IDENTITY_INCOMPLETE"
        )
    pair_pointer_path = Path(pair_manifest_path).expanduser()
    if (
        not pair_pointer_path.is_absolute()
        or pair_pointer_path.is_symlink()
        or pair_pointer_path.resolve().is_relative_to(
            Path(generation_root).resolve(strict=True)
        )
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PAIR_POINTER_PATH_INVALID"
        )
    precommit = candidate_generation_manifest_path is not None
    binding = _pair_binding(
        pair_manifest_path=(
            Path(candidate_generation_manifest_path)
            if precommit
            else pair_pointer_path
        ),
        generation_root=Path(generation_root),
        require_active_pointer=not precommit,
    )
    parent_pair_id = binding.lineage.get("parent_pair_generation_id")
    parent_manifest_sha = binding.lineage.get(
        "parent_pair_manifest_sha256"
    )
    if parent_pair_id is None or parent_manifest_sha is None:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PAIR_SUCCESSOR_REQUIRED"
        )
    parent_generation_manifest = (
        Path(generation_root)
        / str(parent_pair_id)
        / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
    )
    parent_binding = _pair_binding(
        pair_manifest_path=parent_generation_manifest,
        generation_root=Path(generation_root),
        require_active_pointer=False,
    )
    if (
        parent_binding.pair_generation_id != parent_pair_id
        or parent_binding.manifest_sha256 != parent_manifest_sha
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PARENT_PAIR_IDENTITY_MISMATCH"
        )

    previous: dict[str, str] | None = None
    if previous_publication_json is not None:
        previous_path = Path(previous_publication_json)
        previous_payload = require_live_tail_publication_event(
            previous_path,
            expected_sha256=str(previous_publication_sha256),
            require_pass=True,
            _validate_previous=False,
        )
        if (
            previous_payload["pair"]["pair_generation_id"]
            != parent_binding.pair_generation_id
            or previous_payload["pair"]["generation_manifest"]["sha256"]
            != parent_binding.manifest_sha256
        ):
            raise LiveTailAuthorityError(
                "LIVE_TAIL_PREVIOUS_PUBLICATION_PAIR_MISMATCH"
            )
        previous = {
            "path": str(previous_path.resolve(strict=True)),
            "sha256": _sha256_file(previous_path.resolve(strict=True)),
            "pair_generation_id": parent_binding.pair_generation_id,
        }

    created = _created_utc(created_utc)
    failures, timing = _publication_failures(
        created=created,
        binding=binding,
    )
    event = {
        "schema_version": LIVE_TAIL_PUBLICATION_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS" if not failures else "BLOCK",
        "failures": failures,
        "publisher_owner": LIVE_TAIL_PUBLICATION_OWNER,
        "pair": {
            "pointer": (
                {
                    "path": str(pair_pointer_path.resolve()),
                    "sha256": binding.manifest_sha256,
                }
                if precommit
                else _artifact(binding.manifest_path)
            ),
            "generation_manifest": _artifact(
                binding.generation_manifest_path
            ),
            "pair_generation_id": binding.pair_generation_id,
            "parent_pair_generation_id": parent_binding.pair_generation_id,
            "parent_pair_manifest_sha256": parent_binding.manifest_sha256,
            "native_sources": binding.lineage["native_sources"],
            "coverage": binding.lineage["coverage"],
        },
        "producer": {
            "git_commit": binding.lineage["producer_git_commit"],
            "source_inventory_sha256": binding.lineage[
                "producer_source_inventory_sha256"
            ],
        },
        "timing": timing,
        "previous_publication": previous,
    }
    path, written = write_immutable_json_event(
        Path(event_root),
        LIVE_TAIL_PUBLICATION_EVENT_PREFIX,
        event,
    )
    require_live_tail_publication_event(
        path,
        expected_sha256=_sha256_file(path),
    )
    return path, written


def require_live_tail_publication_event(
    event_path: Path,
    *,
    expected_sha256: str | None = None,
    require_pass: bool = False,
    _validate_previous: bool = True,
) -> dict[str, Any]:
    """Strict-load one immutable publication event without freshness reuse."""

    path = Path(event_path).expanduser().resolve(strict=True)
    event = _event_json(path, expected_sha256=expected_sha256)
    expected_fields = {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "publisher_owner",
        "pair",
        "producer",
        "timing",
        "previous_publication",
    }
    if set(event) != expected_fields:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_FIELDS_INVALID"
        )
    if (
        event["schema_version"] != LIVE_TAIL_PUBLICATION_SCHEMA_VERSION
        or event["publisher_owner"] != LIVE_TAIL_PUBLICATION_OWNER
        or event["decision"] not in {"PASS", "BLOCK"}
        or not isinstance(event["failures"], list)
        or any(
            not isinstance(failure, str) or not failure
            for failure in event["failures"]
        )
        or (event["decision"] == "PASS") != (not event["failures"])
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_DECISION_INVALID"
        )
    if require_pass and event["decision"] != "PASS":
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_NOT_PASS"
        )
    _utc(event["created_utc"], label="LIVE_TAIL_PUBLICATION_CREATED")

    pair = event["pair"]
    if not isinstance(pair, dict) or set(pair) != {
        "pointer",
        "generation_manifest",
        "pair_generation_id",
        "parent_pair_generation_id",
        "parent_pair_manifest_sha256",
        "native_sources",
        "coverage",
    }:
        raise LiveTailAuthorityError("LIVE_TAIL_PUBLICATION_PAIR_INVALID")
    generation = pair["generation_manifest"]
    pointer = pair["pointer"]
    for label, artifact in (
        ("LIVE_TAIL_POINTER", pointer),
        ("LIVE_TAIL_GENERATION", generation),
    ):
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256"}
        ):
            raise LiveTailAuthorityError(f"{label}_BINDING_INVALID")
        artifact_path = Path(str(artifact["path"] or "")).expanduser()
        if not artifact_path.is_absolute() or artifact_path.resolve() != artifact_path:
            raise LiveTailAuthorityError(f"{label}_PATH_INVALID")
        _require_sha256(artifact["sha256"], label=label)
    _require_exact_file(
        generation["path"],
        label="LIVE_TAIL_GENERATION",
        expected_sha256=generation["sha256"],
    )
    generation_path = Path(generation["path"])
    generation_root = generation_path.parent.parent
    binding = _pair_binding(
        pair_manifest_path=generation_path,
        generation_root=generation_root,
        require_active_pointer=False,
    )
    if (
        binding.pair_generation_id != pair["pair_generation_id"]
        or binding.manifest_sha256 != generation["sha256"]
        or binding.lineage["parent_pair_generation_id"]
        != pair["parent_pair_generation_id"]
        or binding.lineage["parent_pair_manifest_sha256"]
        != pair["parent_pair_manifest_sha256"]
        or binding.lineage["native_sources"] != pair["native_sources"]
        or binding.lineage["coverage"] != pair["coverage"]
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_PAIR_BINDING_MISMATCH"
        )
    producer = event["producer"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"git_commit", "source_inventory_sha256"}
        or producer["git_commit"] != binding.lineage["producer_git_commit"]
        or producer["source_inventory_sha256"]
        != binding.lineage["producer_source_inventory_sha256"]
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_PRODUCER_MISMATCH"
        )
    _require_sha256(
        producer["source_inventory_sha256"],
        label="LIVE_TAIL_PRODUCER_INVENTORY",
    )

    timing = event["timing"]
    if not isinstance(timing, dict) or set(timing) != {
        "latest_native_m1_bar_start_utc",
        "latest_native_m5_bar_start_utc",
        "canonical_m5_cutoff_utc",
        "decision_available_utc",
        "expected_latest_closed_m5_start_utc",
        "publication_latency_seconds",
        "max_publication_latency_seconds",
        "validity_seconds",
        "valid_until_utc",
    }:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_TIMING_INVALID"
        )
    rebuilt_failures, rebuilt_timing = _publication_failures(
        created=_utc(
            event["created_utc"],
            label="LIVE_TAIL_PUBLICATION_CREATED",
        ),
        binding=binding,
    )
    if timing != rebuilt_timing or event["failures"] != rebuilt_failures:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_PUBLICATION_TIMING_BINDING_MISMATCH"
        )
    previous = event["previous_publication"]
    if previous is not None:
        if not isinstance(previous, dict) or set(previous) != {
            "path",
            "sha256",
            "pair_generation_id",
        }:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_PREVIOUS_PUBLICATION_INVALID"
            )
        if (
            previous["pair_generation_id"]
            != pair["parent_pair_generation_id"]
        ):
            raise LiveTailAuthorityError(
                "LIVE_TAIL_PREVIOUS_PUBLICATION_BINDING_MISMATCH"
            )
        if _validate_previous:
            prior = require_live_tail_publication_event(
                Path(previous["path"]),
                expected_sha256=previous["sha256"],
                require_pass=True,
                _validate_previous=False,
            )
            if (
                prior["pair"]["pair_generation_id"]
                != previous["pair_generation_id"]
                or prior["pair"]["generation_manifest"]["sha256"]
                != pair["parent_pair_manifest_sha256"]
            ):
                raise LiveTailAuthorityError(
                    "LIVE_TAIL_PREVIOUS_PUBLICATION_BINDING_MISMATCH"
                )
    return event


def _publish_live_tail_admission_event_locked(
    *,
    event_root: Path,
    parent_publication_json: Path,
    parent_publication_sha256: str,
    child_publication_json: Path,
    child_publication_sha256: str,
    pair_manifest_path: Path,
    generation_root: Path,
    created_utc: pd.Timestamp | datetime | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish short-lived serving admission from two consecutive events."""

    parent_path = Path(parent_publication_json).resolve(strict=True)
    child_path = Path(child_publication_json).resolve(strict=True)
    parent = require_live_tail_publication_event(
        parent_path,
        expected_sha256=parent_publication_sha256,
        require_pass=True,
    )
    child = require_live_tail_publication_event(
        child_path,
        expected_sha256=child_publication_sha256,
        require_pass=True,
    )
    current = _pair_binding(
        pair_manifest_path=Path(pair_manifest_path),
        generation_root=Path(generation_root),
        require_active_pointer=True,
    )
    created = _created_utc(created_utc)
    failures: list[str] = []
    child_previous = child["previous_publication"]
    if (
        not isinstance(child_previous, dict)
        or child_previous.get("path") != str(parent_path)
        or child_previous.get("sha256") != parent_publication_sha256
        or child_previous.get("pair_generation_id")
        != parent["pair"]["pair_generation_id"]
    ):
        failures.append("child_does_not_bind_parent_publication")
    if (
        child["pair"]["parent_pair_generation_id"]
        != parent["pair"]["pair_generation_id"]
        or child["pair"]["parent_pair_manifest_sha256"]
        != parent["pair"]["generation_manifest"]["sha256"]
    ):
        failures.append("pair_parent_child_chain_mismatch")
    if (
        current.pair_generation_id
        != child["pair"]["pair_generation_id"]
        or current.manifest_sha256
        != child["pair"]["generation_manifest"]["sha256"]
    ):
        failures.append("active_pair_pointer_is_not_child_publication")
    if child["pair"]["pointer"] != {
        "path": str(current.manifest_path),
        "sha256": current.manifest_sha256,
    }:
        failures.append(
            "child_publication_pointer_is_not_active_pair"
        )
    if child["producer"] != parent["producer"]:
        failures.append("publisher_producer_identity_changed")

    parent_created = _utc(
        parent["created_utc"],
        label="LIVE_TAIL_PARENT_CREATED",
    )
    child_created = _utc(
        child["created_utc"],
        label="LIVE_TAIL_CHILD_CREATED",
    )
    if created < child_created:
        failures.append("admission_created_before_child_publication")
    event_gap = float((child_created - parent_created).total_seconds())
    if (
        event_gap <= 0.0
        or event_gap > LIVE_TAIL_MAX_CONSECUTIVE_EVENT_GAP_SECONDS
    ):
        failures.append("publication_event_gap_outside_contract")
    parent_cutoff = _utc(
        parent["timing"]["canonical_m5_cutoff_utc"],
        label="LIVE_TAIL_PARENT_CUTOFF",
    )
    child_cutoff = _utc(
        child["timing"]["canonical_m5_cutoff_utc"],
        label="LIVE_TAIL_CHILD_CUTOFF",
    )
    if child_cutoff <= parent_cutoff:
        failures.append("publication_pair_did_not_advance")
    valid_until = _utc(
        child["timing"]["valid_until_utc"],
        label="LIVE_TAIL_CHILD_VALID_UNTIL",
    )
    if created > valid_until:
        failures.append("child_publication_is_stale")

    event = {
        "schema_version": LIVE_TAIL_ADMISSION_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS" if not failures else "BLOCK",
        "failures": failures,
        "admission_owner": LIVE_TAIL_ADMISSION_OWNER,
        "parent_publication": {
            "path": str(parent_path),
            "sha256": parent_publication_sha256,
        },
        "child_publication": {
            "path": str(child_path),
            "sha256": child_publication_sha256,
        },
        "anchor_pair": {
            "pointer_path": str(current.manifest_path),
            "pointer_sha256": _sha256_file(current.manifest_path),
            "generation_manifest_path": str(
                current.generation_manifest_path
            ),
            "generation_manifest_sha256": current.manifest_sha256,
            "pair_generation_id": current.pair_generation_id,
        },
        "producer": child["producer"],
        "publication_event_gap_seconds": event_gap,
        "valid_until_utc": valid_until.isoformat(),
    }
    path, written = write_immutable_json_event(
        Path(event_root),
        LIVE_TAIL_ADMISSION_EVENT_PREFIX,
        event,
    )
    require_live_tail_admission_event(
        path,
        expected_sha256=_sha256_file(path),
        pair_manifest_path=Path(pair_manifest_path),
        generation_root=Path(generation_root),
        now_utc=created,
        require_pass=False,
    )
    return path, written


def publish_live_tail_admission_event(
    *,
    event_root: Path,
    parent_publication_json: Path,
    parent_publication_sha256: str,
    child_publication_json: Path,
    child_publication_sha256: str,
    pair_manifest_path: Path,
    generation_root: Path,
    created_utc: pd.Timestamp | datetime | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish admission while sharing the pair/broker mutation lease."""

    pointer = Path(pair_manifest_path).expanduser()
    if not pointer.is_absolute() or pointer.parent.is_symlink():
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_PAIR_POINTER_PATH_INVALID"
        )
    lock_path = (
        pointer.parent.resolve(strict=True)
        / PAIR_PUBLISH_LOCK_FILENAME
    )
    if lock_path.is_symlink():
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_PAIR_LOCK_INVALID"
        )
    with lock_path.open("a+b") as lock_handle:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise LiveTailAuthorityError(
                "LIVE_TAIL_ADMISSION_PAIR_LOCK_INVALID"
            )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _publish_live_tail_admission_event_locked(
                event_root=event_root,
                parent_publication_json=parent_publication_json,
                parent_publication_sha256=(
                    parent_publication_sha256
                ),
                child_publication_json=child_publication_json,
                child_publication_sha256=child_publication_sha256,
                pair_manifest_path=pointer,
                generation_root=generation_root,
                created_utc=created_utc,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def require_live_tail_admission_event(
    event_path: Path,
    *,
    pair_manifest_path: Path,
    generation_root: Path,
    expected_sha256: str | None = None,
    now_utc: pd.Timestamp | datetime | str | None = None,
    require_pass: bool = True,
    require_current: bool = True,
    enforce_freshness: bool = True,
) -> dict[str, Any]:
    """Revalidate short-lived live-tail authority against the current pointer."""

    path = Path(event_path).resolve(strict=True)
    event = _event_json(path, expected_sha256=expected_sha256)
    if set(event) != {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "admission_owner",
        "parent_publication",
        "child_publication",
        "anchor_pair",
        "producer",
        "publication_event_gap_seconds",
        "valid_until_utc",
    }:
        raise LiveTailAuthorityError("LIVE_TAIL_ADMISSION_FIELDS_INVALID")
    if (
        event["schema_version"] != LIVE_TAIL_ADMISSION_SCHEMA_VERSION
        or event["admission_owner"] != LIVE_TAIL_ADMISSION_OWNER
        or event["decision"] not in {"PASS", "BLOCK"}
        or not isinstance(event["failures"], list)
        or any(
            not isinstance(failure, str) or not failure
            for failure in event["failures"]
        )
        or (event["decision"] == "PASS") != (not event["failures"])
    ):
        raise LiveTailAuthorityError("LIVE_TAIL_ADMISSION_DECISION_INVALID")
    parent_binding = event["parent_publication"]
    child_binding = event["child_publication"]
    for label, binding in (
        ("LIVE_TAIL_PARENT_PUBLICATION", parent_binding),
        ("LIVE_TAIL_CHILD_PUBLICATION", child_binding),
    ):
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "sha256",
        }:
            raise LiveTailAuthorityError(f"{label}_INVALID")
    parent = require_live_tail_publication_event(
        Path(parent_binding["path"]),
        expected_sha256=parent_binding["sha256"],
        require_pass=True,
    )
    child = require_live_tail_publication_event(
        Path(child_binding["path"]),
        expected_sha256=child_binding["sha256"],
        require_pass=True,
    )
    anchor = event["anchor_pair"]
    if not isinstance(anchor, dict) or set(anchor) != {
        "pointer_path",
        "pointer_sha256",
        "generation_manifest_path",
        "generation_manifest_sha256",
        "pair_generation_id",
    }:
        raise LiveTailAuthorityError("LIVE_TAIL_ADMISSION_ANCHOR_INVALID")
    anchor_generation_path, _ = _require_exact_file(
        anchor["generation_manifest_path"],
        label="LIVE_TAIL_ADMISSION_ANCHOR_GENERATION",
        expected_sha256=anchor["generation_manifest_sha256"],
    )
    anchor_binding = _pair_binding(
        pair_manifest_path=anchor_generation_path,
        generation_root=Path(generation_root),
        require_active_pointer=False,
    )
    _require_sha256(
        anchor["pointer_sha256"],
        label="LIVE_TAIL_ADMISSION_ANCHOR_POINTER",
    )
    if (
        anchor["pointer_path"]
        != str(Path(pair_manifest_path).expanduser().resolve())
        or anchor["generation_manifest_path"]
        != str(anchor_binding.generation_manifest_path)
        or anchor["generation_manifest_sha256"]
        != anchor_binding.manifest_sha256
        or anchor["pair_generation_id"]
        != anchor_binding.pair_generation_id
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_ANCHOR_BINDING_MISMATCH"
        )
    if event["producer"] != child["producer"]:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_PRODUCER_BINDING_MISMATCH"
        )
    parent_created = _utc(
        parent["created_utc"],
        label="LIVE_TAIL_PARENT_CREATED",
    )
    child_created = _utc(
        child["created_utc"],
        label="LIVE_TAIL_CHILD_CREATED",
    )
    created = _utc(
        event["created_utc"],
        label="LIVE_TAIL_ADMISSION_CREATED",
    )
    gap = float((child_created - parent_created).total_seconds())
    if (
        event["publication_event_gap_seconds"] != gap
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_EVENT_GAP_INVALID"
        )
    valid_until = _utc(
        event["valid_until_utc"],
        label="LIVE_TAIL_ADMISSION_VALID_UNTIL",
    )
    if event["valid_until_utc"] != child["timing"]["valid_until_utc"]:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_VALIDITY_BINDING_MISMATCH"
        )
    rebuilt_failures: list[str] = []
    if (
        child["previous_publication"]
        != {
            "path": parent_binding["path"],
            "sha256": parent_binding["sha256"],
            "pair_generation_id": parent["pair"]["pair_generation_id"],
        }
    ):
        rebuilt_failures.append("child_does_not_bind_parent_publication")
    if (
        child["pair"]["parent_pair_generation_id"]
        != parent["pair"]["pair_generation_id"]
        or child["pair"]["parent_pair_manifest_sha256"]
        != parent["pair"]["generation_manifest"]["sha256"]
    ):
        rebuilt_failures.append("pair_parent_child_chain_mismatch")
    if (
        anchor_binding.pair_generation_id
        != child["pair"]["pair_generation_id"]
        or anchor_binding.manifest_sha256
        != child["pair"]["generation_manifest"]["sha256"]
    ):
        rebuilt_failures.append(
            "active_pair_pointer_is_not_child_publication"
        )
    if child["pair"]["pointer"] != {
        "path": anchor["pointer_path"],
        "sha256": anchor["pointer_sha256"],
    }:
        rebuilt_failures.append(
            "child_publication_pointer_is_not_active_pair"
        )
    if child["producer"] != parent["producer"]:
        rebuilt_failures.append("publisher_producer_identity_changed")
    if created < child_created:
        rebuilt_failures.append(
            "admission_created_before_child_publication"
        )
    if gap <= 0.0 or gap > LIVE_TAIL_MAX_CONSECUTIVE_EVENT_GAP_SECONDS:
        rebuilt_failures.append("publication_event_gap_outside_contract")
    if (
        _utc(
            child["timing"]["canonical_m5_cutoff_utc"],
            label="LIVE_TAIL_CHILD_CUTOFF",
        )
        <= _utc(
            parent["timing"]["canonical_m5_cutoff_utc"],
            label="LIVE_TAIL_PARENT_CUTOFF",
        )
    ):
        rebuilt_failures.append("publication_pair_did_not_advance")
    if created > valid_until:
        rebuilt_failures.append("child_publication_is_stale")
    if event["failures"] != rebuilt_failures:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_ADMISSION_FAILURE_BINDING_MISMATCH"
        )
    if require_pass and event["decision"] != "PASS":
        raise LiveTailAuthorityError("LIVE_TAIL_ADMISSION_NOT_PASS")
    if event["decision"] != "PASS":
        return event

    if require_current:
        current = _pair_binding(
            pair_manifest_path=Path(pair_manifest_path),
            generation_root=Path(generation_root),
            require_active_pointer=True,
        )
        expected_anchor = {
            "pointer_path": str(current.manifest_path),
            "pointer_sha256": _sha256_file(current.manifest_path),
            "generation_manifest_path": str(
                current.generation_manifest_path
            ),
            "generation_manifest_sha256": current.manifest_sha256,
            "pair_generation_id": current.pair_generation_id,
        }
        if anchor != expected_anchor:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_ADMISSION_CURRENT_PAIR_MISMATCH"
            )
    if enforce_freshness:
        now = _created_utc(now_utc)
        if now < created:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_ADMISSION_NOW_BEFORE_CREATED"
            )
        if now > valid_until:
            raise LiveTailAuthorityError("LIVE_TAIL_ADMISSION_EXPIRED")
    return event


def require_historical_live_tail_admission_event(
    event_path: Path,
    *,
    expected_sha256: str,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Validate immutable launch-time admission without reusing its freshness."""

    path = Path(event_path).expanduser().resolve(strict=True)
    raw = _event_json(path, expected_sha256=expected_sha256)
    anchor = raw.get("anchor_pair")
    if not isinstance(anchor, dict):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_HISTORICAL_ADMISSION_ANCHOR_INVALID"
        )
    generation_path = Path(
        str(anchor.get("generation_manifest_path") or "")
    ).expanduser()
    pointer_path = Path(str(anchor.get("pointer_path") or "")).expanduser()
    if (
        not generation_path.is_absolute()
        or not pointer_path.is_absolute()
        or generation_path.parent.parent.is_symlink()
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_HISTORICAL_ADMISSION_PATH_INVALID"
        )
    return require_live_tail_admission_event(
        path,
        expected_sha256=expected_sha256,
        pair_manifest_path=pointer_path,
        generation_root=generation_path.parent.parent,
        now_utc=raw.get("created_utc"),
        require_pass=require_pass,
        require_current=False,
        enforce_freshness=False,
    )


def live_tail_launch_authority(
    event_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Derive static new-Entry authority from one exact launch admission."""

    path = Path(event_path).expanduser().resolve(strict=True)
    expected = _require_sha256(
        expected_sha256,
        label="LIVE_TAIL_LAUNCH_ADMISSION",
    )
    event = require_historical_live_tail_admission_event(
        path,
        expected_sha256=expected,
    )
    parent_publication_path = Path(
        event["parent_publication"]["path"]
    ).resolve(strict=True)
    child_publication_path = Path(
        event["child_publication"]["path"]
    ).resolve(strict=True)
    if parent_publication_path.parent != child_publication_path.parent:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_LAUNCH_PUBLICATION_ROOT_SPLIT"
        )
    anchor = event["anchor_pair"]
    generation_root = Path(
        anchor["generation_manifest_path"]
    ).parent.parent.resolve(strict=True)
    authority = {
        "schema_version": LIVE_TAIL_LAUNCH_AUTHORITY_SCHEMA_VERSION,
        "launch_admission": {
            "json_path": str(path),
            "sha256": expected,
        },
        "admission_event_root": str(path.parent),
        "publication_event_root": str(parent_publication_path.parent),
        "pair_pointer_path": anchor["pointer_path"],
        "pair_generation_root": str(generation_root),
        "launch_anchor": {
            "pair_generation_id": anchor["pair_generation_id"],
            "generation_manifest_sha256": anchor[
                "generation_manifest_sha256"
            ],
        },
        "producer": event["producer"],
    }
    return require_live_tail_launch_authority(authority)


def require_live_tail_launch_authority(
    authority: object,
) -> dict[str, Any]:
    """Rebuild and compare the exact static authority stored in launch state."""

    expected_fields = {
        "schema_version",
        "launch_admission",
        "admission_event_root",
        "publication_event_root",
        "pair_pointer_path",
        "pair_generation_root",
        "launch_anchor",
        "producer",
    }
    if not isinstance(authority, dict) or set(authority) != expected_fields:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_LAUNCH_AUTHORITY_FIELDS_INVALID"
        )
    binding = authority["launch_admission"]
    if not isinstance(binding, dict) or set(binding) != {
        "json_path",
        "sha256",
    }:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_LAUNCH_ADMISSION_BINDING_INVALID"
        )
    path = Path(str(binding["json_path"])).expanduser().resolve(strict=True)
    event = require_historical_live_tail_admission_event(
        path,
        expected_sha256=str(binding["sha256"]),
    )
    parent_root = Path(
        event["parent_publication"]["path"]
    ).resolve(strict=True).parent
    child_root = Path(
        event["child_publication"]["path"]
    ).resolve(strict=True).parent
    anchor = event["anchor_pair"]
    generation_root = Path(
        anchor["generation_manifest_path"]
    ).parent.parent.resolve(strict=True)
    expected = {
        "schema_version": LIVE_TAIL_LAUNCH_AUTHORITY_SCHEMA_VERSION,
        "launch_admission": {
            "json_path": str(path),
            "sha256": _require_sha256(
                binding["sha256"],
                label="LIVE_TAIL_LAUNCH_ADMISSION",
            ),
        },
        "admission_event_root": str(path.parent),
        "publication_event_root": str(parent_root),
        "pair_pointer_path": anchor["pointer_path"],
        "pair_generation_root": str(generation_root),
        "launch_anchor": {
            "pair_generation_id": anchor["pair_generation_id"],
            "generation_manifest_sha256": anchor[
                "generation_manifest_sha256"
            ],
        },
        "producer": event["producer"],
    }
    if parent_root != child_root or authority != expected:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_LAUNCH_AUTHORITY_BINDING_MISMATCH"
        )
    return expected


def _require_pair_descends_from(
    *,
    descendant_generation_manifest_path: Path,
    generation_root: Path,
    ancestor_pair_generation_id: str,
    ancestor_generation_manifest_sha256: str,
) -> None:
    target_id = _require_sha256(
        ancestor_pair_generation_id,
        label="LIVE_TAIL_RUNTIME_ANCESTOR_PAIR",
    )
    target_sha = _require_sha256(
        ancestor_generation_manifest_sha256,
        label="LIVE_TAIL_RUNTIME_ANCESTOR_MANIFEST",
    )
    root = Path(generation_root).expanduser().resolve(strict=True)
    current_path = Path(
        descendant_generation_manifest_path
    ).expanduser().resolve(strict=True)
    seen: set[str] = set()
    while True:
        binding = read_prebuilt_pair_manifest(
            current_path,
            generation_root=root,
        )
        if binding.pair_generation_id in seen:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_RUNTIME_PAIR_LINEAGE_CYCLE"
            )
        seen.add(binding.pair_generation_id)
        if (
            binding.pair_generation_id == target_id
            and binding.manifest_sha256 == target_sha
        ):
            return
        parent_id = binding.lineage.get("parent_pair_generation_id")
        parent_sha = binding.lineage.get("parent_pair_manifest_sha256")
        if parent_id is None or parent_sha is None:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_RUNTIME_PAIR_NOT_DESCENDANT_OF_ANCHOR"
            )
        parent_path = (
            root
            / str(parent_id)
            / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
        )
        _require_exact_file(
            parent_path,
            label="LIVE_TAIL_RUNTIME_PARENT_GENERATION",
            expected_sha256=str(parent_sha),
        )
        current_path = parent_path


def require_newest_live_tail_runtime_admission(
    launch_admission_path: Path,
    *,
    launch_admission_sha256: str,
    expected_pair_generation_id: str | None = None,
    expected_generation_manifest_sha256: str | None = None,
    now_utc: pd.Timestamp | datetime | str | None = None,
) -> dict[str, Any]:
    """Return newest fresh admission chained to launch or the prior runtime pair."""

    if (expected_pair_generation_id is None) != (
        expected_generation_manifest_sha256 is None
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_EXPECTED_PAIR_IDENTITY_INCOMPLETE"
        )
    launch_path = Path(launch_admission_path).expanduser().resolve(strict=True)
    launch = require_historical_live_tail_admission_event(
        launch_path,
        expected_sha256=launch_admission_sha256,
    )
    event_root = launch_path.parent
    try:
        newest_path = select_latest_immutable_event(
            event_root,
            LIVE_TAIL_ADMISSION_EVENT_PREFIX,
        )
    except ImmutableEventAuthorityError as exc:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_EVENT_AUTHORITY_INVALID"
        ) from exc
    if newest_path is None:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_ADMISSION_MISSING"
        )
    launch_anchor = launch["anchor_pair"]
    launch_publication_root = Path(
        launch["parent_publication"]["path"]
    ).resolve(strict=True).parent
    if (
        Path(launch["child_publication"]["path"]).resolve(strict=True).parent
        != launch_publication_root
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_LAUNCH_PUBLICATION_ROOT_SPLIT"
        )
    try:
        newest_publication_path = select_latest_immutable_event(
            launch_publication_root,
            LIVE_TAIL_PUBLICATION_EVENT_PREFIX,
        )
    except ImmutableEventAuthorityError as exc:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_PUBLICATION_AUTHORITY_INVALID"
        ) from exc
    if newest_publication_path is None:
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_PUBLICATION_MISSING"
        )
    newest_publication_sha256 = _sha256_file(
        newest_publication_path
    )
    newest_publication = require_live_tail_publication_event(
        newest_publication_path,
        expected_sha256=newest_publication_sha256,
        require_pass=True,
    )
    generation_root = Path(
        launch_anchor["generation_manifest_path"]
    ).parent.parent
    current = require_live_tail_admission_event(
        newest_path,
        expected_sha256=_sha256_file(newest_path),
        pair_manifest_path=Path(launch_anchor["pointer_path"]),
        generation_root=generation_root,
        now_utc=now_utc,
        require_pass=True,
        require_current=True,
        enforce_freshness=True,
    )
    current_anchor = current["anchor_pair"]
    inventory: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for candidate in event_root.glob(
        f"{LIVE_TAIL_ADMISSION_EVENT_PREFIX}_*.json"
    ):
        if candidate.name in {
            f"{LIVE_TAIL_ADMISSION_EVENT_PREFIX}_latest.json",
            f"{LIVE_TAIL_ADMISSION_EVENT_PREFIX}_MANIFEST.json",
        }:
            continue
        payload = require_historical_live_tail_admission_event(
            candidate,
            expected_sha256=_sha256_file(candidate),
            require_pass=False,
        )
        inventory.append(
            (
                _utc(
                    payload["created_utc"],
                    label="LIVE_TAIL_RUNTIME_INVENTORY_CREATED",
                ),
                candidate.resolve(strict=True),
                payload,
            )
        )
    inventory.sort(key=lambda item: (item[0], str(item[1])))
    inventory_paths = [item[1] for item in inventory]
    if (
        launch_path not in inventory_paths
        or newest_path not in inventory_paths
        or not inventory
        or inventory[-1][1] != newest_path
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_ADMISSION_INVENTORY_INVALID"
        )
    current_position = inventory_paths.index(newest_path)
    previous_passes = [
        payload
        for _created, _path, payload in inventory[:current_position]
        if payload["decision"] == "PASS"
    ]
    if newest_path != launch_path:
        if not previous_passes:
            raise LiveTailAuthorityError(
                "LIVE_TAIL_RUNTIME_PREVIOUS_PASS_ADMISSION_MISSING"
            )
        prior_anchor = previous_passes[-1]["anchor_pair"]
        current_parent_publication = require_live_tail_publication_event(
            Path(current["parent_publication"]["path"]),
            expected_sha256=current["parent_publication"]["sha256"],
            require_pass=True,
        )
        if (
            current_parent_publication["pair"][
                "pair_generation_id"
            ]
            != prior_anchor["pair_generation_id"]
            or current_parent_publication["pair"][
                "generation_manifest"
            ]["sha256"]
            != prior_anchor["generation_manifest_sha256"]
        ):
            raise LiveTailAuthorityError(
                "LIVE_TAIL_RUNTIME_ADMISSION_NOT_MONOTONIC_FROM_PRIOR_PASS"
            )
    current_child_publication = current["child_publication"]
    if (
        Path(current_child_publication["path"]).resolve(strict=True)
        != newest_publication_path
        or current_child_publication["sha256"]
        != newest_publication_sha256
        or newest_publication["pair"]["pair_generation_id"]
        != current_anchor["pair_generation_id"]
        or newest_publication["pair"]["generation_manifest"]["sha256"]
        != current_anchor["generation_manifest_sha256"]
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_NEWEST_PUBLICATION_NOT_ADMITTED"
        )
    if (
        Path(current["parent_publication"]["path"]).resolve(strict=True).parent
        != launch_publication_root
        or Path(
            current["child_publication"]["path"]
        ).resolve(strict=True).parent
        != launch_publication_root
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_PUBLICATION_ROOT_MISMATCH"
        )
    current_generation_root = Path(
        current_anchor["generation_manifest_path"]
    ).parent.parent
    if (
        current["producer"] != launch["producer"]
        or current_generation_root.resolve(strict=True)
        != generation_root.resolve(strict=True)
        or _utc(
            current["created_utc"],
            label="LIVE_TAIL_RUNTIME_CURRENT_CREATED",
        )
        < _utc(
            launch["created_utc"],
            label="LIVE_TAIL_RUNTIME_LAUNCH_CREATED",
        )
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_LAUNCH_BINDING_MISMATCH"
        )
    # Every admitted runtime pair must remain in the immutable lineage rooted
    # at the launch anchor.  When a caller supplies a pair identity, it is the
    # exact generation used for model inference, not merely an allowed
    # ancestor: advancing the pointer between inference and order must force a
    # fresh decision rather than admit a stale one.
    _require_pair_descends_from(
        descendant_generation_manifest_path=Path(
            current_anchor["generation_manifest_path"]
        ),
        generation_root=generation_root,
        ancestor_pair_generation_id=str(
            launch_anchor["pair_generation_id"]
        ),
        ancestor_generation_manifest_sha256=str(
            launch_anchor["generation_manifest_sha256"]
        ),
    )
    if expected_pair_generation_id is not None and (
        current_anchor["pair_generation_id"]
        != _require_sha256(
            expected_pair_generation_id,
            label="LIVE_TAIL_RUNTIME_EXPECTED_PAIR",
        )
        or current_anchor["generation_manifest_sha256"]
        != _require_sha256(
            expected_generation_manifest_sha256,
            label="LIVE_TAIL_RUNTIME_EXPECTED_MANIFEST",
        )
    ):
        raise LiveTailAuthorityError(
            "LIVE_TAIL_RUNTIME_DECISION_PAIR_MISMATCH"
        )
    return {
        "schema_version": "gx1_live_tail_runtime_authority_v1",
        "launch_admission": {
            "path": str(launch_path),
            "sha256": launch_admission_sha256,
            "pair_generation_id": launch_anchor["pair_generation_id"],
            "generation_manifest_sha256": launch_anchor[
                "generation_manifest_sha256"
            ],
        },
        "current_admission": {
            "path": str(newest_path),
            "sha256": _sha256_file(newest_path),
            "pair_generation_id": current_anchor["pair_generation_id"],
            "generation_manifest_sha256": current_anchor[
                "generation_manifest_sha256"
            ],
            "valid_until_utc": current["valid_until_utc"],
        },
        "current_publication": {
            "path": str(newest_publication_path),
            "sha256": newest_publication_sha256,
            "pair_generation_id": newest_publication["pair"][
                "pair_generation_id"
            ],
            "generation_manifest_sha256": newest_publication["pair"][
                "generation_manifest"
            ]["sha256"],
        },
        "producer": current["producer"],
    }


def require_newest_live_tail_runtime_authority(
    launch_authority: object,
    *,
    expected_pair_generation_id: str | None = None,
    expected_generation_manifest_sha256: str | None = None,
    now_utc: pd.Timestamp | datetime | str | None = None,
) -> dict[str, Any]:
    """Resolve dynamic new-Entry admission from static launch authority."""

    authority = require_live_tail_launch_authority(launch_authority)
    return require_newest_live_tail_runtime_admission(
        Path(authority["launch_admission"]["json_path"]),
        launch_admission_sha256=authority["launch_admission"]["sha256"],
        expected_pair_generation_id=expected_pair_generation_id,
        expected_generation_manifest_sha256=(
            expected_generation_manifest_sha256
        ),
        now_utc=now_utc,
    )
