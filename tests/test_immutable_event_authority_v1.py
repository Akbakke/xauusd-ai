"""Synthetic filesystem fixtures prove authority mechanics, not model quality."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import pytest

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    next_immutable_event_created_utc,
    require_newest_immutable_event,
    select_latest_immutable_event,
    validated_immutable_event_authority_inventory,
    write_immutable_json_event,
)


PREFIX = "ENTRY_TEST_AUTHORITY"


def _write_event(
    root: Path,
    stamp: str,
    *,
    decision: str,
    created_utc: str | None = None,
) -> Path:
    path = root / f"{PREFIX}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    payload = {
        "created_utc": created_utc or parsed.isoformat(),
        "json_path": str(path.resolve()),
        "decision": decision,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _publish_event(
    root: Path,
    created: datetime,
    *,
    decision: str,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
) -> Path:
    path, _ = write_immutable_json_event(
        root,
        PREFIX,
        {"created_utc": created.isoformat(), "decision": decision},
        authority_root=authority_root,
        scope_dir_glob=scope_dir_glob,
    )
    return path


def _witness(path: Path) -> Path:
    return path.with_name(f".{path.name}.order")


def test_legacy_candidate_order_is_unproven_regardless_of_mtime(tmp_path: Path) -> None:
    older_green = _write_event(
        tmp_path / "model_native_seq513_candidate_old",
        "20260716T100000Z",
        decision="PASS",
    )
    newer_red = _write_event(
        tmp_path / "model_native_seq513_candidate_new",
        "20260716T110000Z",
        decision="FAIL",
    )
    os.utime(older_green, ns=(2_000_000_000, 2_000_000_000))
    os.utime(newer_red, ns=(1_000_000_000, 1_000_000_000))

    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        select_latest_immutable_event(
            tmp_path,
            PREFIX,
            scope_dir_glob="model_native_seq513_candidate_*",
        )


def test_explicit_event_must_be_newest_in_its_authority_directory(tmp_path: Path) -> None:
    created = datetime(2026, 7, 16, 10, tzinfo=timezone.utc)
    older_green = _publish_event(tmp_path, created, decision="PASS")
    newer_red = _publish_event(tmp_path, created + timedelta(hours=1), decision="FAIL")

    with pytest.raises(ImmutableEventAuthorityError, match="not the newest"):
        require_newest_immutable_event(older_green, PREFIX)
    assert require_newest_immutable_event(newer_red, PREFIX) == newer_red.resolve()


def test_explicit_event_authority_rejects_symlink_alias(tmp_path: Path) -> None:
    event = _write_event(tmp_path, "20260716T110000Z", decision="PASS")
    alias = tmp_path / "alias.json"
    alias.symlink_to(event)

    with pytest.raises(ImmutableEventAuthorityError, match="not a regular file"):
        require_newest_immutable_event(alias, PREFIX)


def test_independent_legacy_directories_cannot_establish_generation_order(tmp_path: Path) -> None:
    older = _write_event(tmp_path / "variant", "20260716T100000Z", decision="PASS")
    newer = _write_event(tmp_path, "20260716T110000Z", decision="FAIL")
    os.utime(older, ns=(2_000_000_000, 2_000_000_000))
    os.utime(newer, ns=(1_000_000_000, 1_000_000_000))

    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        select_latest_immutable_event(tmp_path, PREFIX)


def test_malformed_newest_event_fails_closed(tmp_path: Path) -> None:
    _write_event(tmp_path, "20260716T100000Z", decision="PASS")
    _write_event(
        tmp_path,
        "20260716T110000Z",
        decision="FAIL",
        created_utc="2026-07-16T11:00:01+00:00",
    )

    with pytest.raises(ImmutableEventAuthorityError, match="does not match created_utc"):
        select_latest_immutable_event(tmp_path, PREFIX)


def test_malformed_event_filename_fails_closed(tmp_path: Path) -> None:
    _write_event(tmp_path, "20260716T100000Z", decision="PASS")
    malformed = tmp_path / f"{PREFIX}_20260716Tbroken.json"
    malformed.write_text("{}", encoding="utf-8")

    with pytest.raises(ImmutableEventAuthorityError, match="exact UTC timestamp"):
        select_latest_immutable_event(tmp_path, PREFIX)


def test_duplicate_legacy_timestamp_across_candidates_fails_closed(tmp_path: Path) -> None:
    _write_event(
        tmp_path / "model_native_seq513_candidate_a",
        "20260716T110000Z",
        decision="PASS",
    )
    _write_event(
        tmp_path / "model_native_seq513_candidate_b",
        "20260716T110000Z",
        decision="FAIL",
    )

    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        select_latest_immutable_event(
            tmp_path,
            PREFIX,
            scope_dir_glob="model_native_seq513_candidate_*",
        )


def test_immutable_event_writer_self_binds_without_mutable_mirror(tmp_path: Path) -> None:
    created_utc = "2026-07-16T11:00:00.123456+00:00"
    event_path, event = write_immutable_json_event(
        tmp_path,
        PREFIX,
        {"created_utc": created_utc, "decision": "FAIL"},
    )
    assert event_path.name == f"{PREFIX}_20260716T110000123456Z.json"
    assert event["json_path"] == str(event_path)
    assert json.loads(event_path.read_text(encoding="utf-8"))["json_path"] == str(
        event_path
    )
    assert select_latest_immutable_event(tmp_path, PREFIX) == event_path
    assert not list(tmp_path.glob("*_latest.json"))
    event_bytes = event_path.read_bytes()
    witness_bytes = _witness(event_path).read_bytes()
    assert set(event) == {"created_utc", "decision", "json_path"}

    with pytest.raises(ImmutableEventAuthorityError, match="already exists"):
        write_immutable_json_event(
            tmp_path,
            PREFIX,
            {"created_utc": created_utc, "decision": "PASS"},
        )
    assert event_path.read_bytes() == event_bytes
    assert _witness(event_path).read_bytes() == witness_bytes


def test_immutable_event_is_hidden_and_fsynced_before_final_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    original_publish = authority._publish_file_noreplace
    observed: dict[str, object] = {}

    def capture_publish(source: Path, destination: Path) -> None:
        observed["source_hidden"] = source.name.startswith(".")
        observed["source_bytes"] = source.read_bytes()
        observed["destination_absent"] = not destination.exists()
        original_publish(source, destination)

    monkeypatch.setattr(authority, "_publish_file_noreplace", capture_publish)
    event_path, _ = write_immutable_json_event(
        tmp_path,
        PREFIX,
        {
            "created_utc": "2026-07-16T11:00:00.123456+00:00",
            "decision": "PASS",
        },
    )

    assert observed["source_hidden"] is True
    assert observed["destination_absent"] is True
    assert observed["source_bytes"] == event_path.read_bytes()
    assert not list(tmp_path.glob(".*.staging.*"))


def test_publication_witness_is_durable_before_event_becomes_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    operations: list[tuple[str, Path]] = []
    original_publish = authority._publish_file_noreplace
    original_sync = authority._fsync_directory

    def record_publish(source: Path, destination: Path) -> None:
        original_publish(source, destination)
        operations.append(("publish", destination))

    def record_sync(directory: Path) -> None:
        original_sync(directory)
        operations.append(("sync", directory))

    monkeypatch.setattr(authority, "_publish_file_noreplace", record_publish)
    monkeypatch.setattr(authority, "_fsync_directory", record_sync)
    path = _publish_event(
        tmp_path, datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc), decision="FAIL"
    )

    assert operations == [
        ("publish", _witness(path)),
        ("sync", tmp_path),
        ("publish", path),
        ("sync", tmp_path),
    ]


def test_next_event_time_is_strictly_newer_than_future_inventory(
    tmp_path: Path,
) -> None:
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    write_immutable_json_event(
        tmp_path,
        PREFIX,
        {"created_utc": future.isoformat(), "decision": "PASS"},
    )

    assert next_immutable_event_created_utc(tmp_path, PREFIX) == future + timedelta(
        microseconds=1
    )


def test_rollback_red_supersedes_green_without_rewriting_created_utc(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, 32, 121002, tzinfo=timezone.utc)
    green = _publish_event(tmp_path, created, decision="PASS")
    rolled_back = created - timedelta(seconds=61)
    red = _publish_event(tmp_path, rolled_back, decision="FAIL")
    os.utime(green, ns=(2_000_000_000, 2_000_000_000))
    os.utime(red, ns=(1_000_000_000, 1_000_000_000))

    assert json.loads(red.read_text(encoding="utf-8"))["created_utc"] == rolled_back.isoformat()
    assert max(green.name, red.name) == green.name
    assert select_latest_immutable_event(tmp_path, PREFIX) == red
    assert require_newest_immutable_event(red, PREFIX) == red
    with pytest.raises(ImmutableEventAuthorityError, match="not the newest"):
        require_newest_immutable_event(green, PREFIX)


def test_later_green_can_supersede_red_by_publication_not_clock(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    _publish_event(tmp_path, created, decision="PASS")
    _publish_event(tmp_path, created - timedelta(seconds=61), decision="FAIL")
    recovered = _publish_event(tmp_path, created - timedelta(seconds=62), decision="PASS")

    assert select_latest_immutable_event(tmp_path, PREFIX) == recovered


def test_unwitnessed_legacy_rollback_blocks_instead_of_guessing_order(tmp_path: Path) -> None:
    green = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    red = _write_event(tmp_path, "20260906T072931Z", decision="FAIL")

    for requested in (green, red):
        with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
            require_newest_immutable_event(requested, PREFIX)


def test_fresh_event_can_supersede_unordered_legacy_inventory(tmp_path: Path) -> None:
    green = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    red = _write_event(tmp_path, "20260906T072931Z", decision="FAIL")
    preserved = {path: path.read_bytes() for path in (green, red)}
    fresh = _publish_event(
        tmp_path, datetime(2026, 9, 6, 7, 29, tzinfo=timezone.utc), decision="FAIL"
    )

    assert select_latest_immutable_event(tmp_path, PREFIX) == fresh
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert not _witness(green).exists()
    assert not _witness(red).exists()


def test_independent_witnessed_directories_have_no_shared_order(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    first = _publish_event(tmp_path / "candidate_a", created, decision="PASS")
    second = _publish_event(
        tmp_path / "candidate_b", created - timedelta(seconds=61), decision="FAIL"
    )

    assert select_latest_immutable_event(first.parent, PREFIX) == first
    assert select_latest_immutable_event(second.parent, PREFIX) == second
    with pytest.raises(ImmutableEventAuthorityError, match="authority scope mismatch"):
        select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob="candidate_*")


@pytest.mark.parametrize("scope_dir_glob", [None, "candidate_*"])
def test_explicit_cross_run_scope_orders_a_b_c_through_clock_rollback(
    tmp_path: Path, scope_dir_glob: str | None
) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    paths = []
    for index, decision in enumerate(("PASS", "FAIL", "PASS")):
        path = _publish_event(
            tmp_path / f"candidate_{index}" / "reports",
            created - timedelta(seconds=61 * index),
            decision=decision,
            authority_root=tmp_path,
            scope_dir_glob=scope_dir_glob,
        )
        paths.append(path)
        assert select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob=scope_dir_glob) == path
        assert require_newest_immutable_event(
            path, PREFIX, authority_root=tmp_path, scope_dir_glob=scope_dir_glob
        ) == path
        witness = json.loads(_witness(path).read_text(encoding="utf-8"))
        assert witness["authority_root"] == str(tmp_path)
        assert witness["scope_dir_glob"] == scope_dir_glob
        assert witness["predecessor_count"] == index
    with pytest.raises(ImmutableEventAuthorityError, match="not the newest"):
        require_newest_immutable_event(
            paths[0], PREFIX, authority_root=tmp_path, scope_dir_glob=scope_dir_glob
        )
    with pytest.raises(ImmutableEventAuthorityError, match="authority scope mismatch"):
        require_newest_immutable_event(paths[-1], PREFIX)
    assert next_immutable_event_created_utc(
        paths[-1].parent,
        PREFIX,
        created,
        authority_root=tmp_path,
        scope_dir_glob=scope_dir_glob,
    ) > created


def test_default_writer_inventory_is_recursive_like_default_consumer(tmp_path: Path) -> None:
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    legacy = _write_event(
        tmp_path / "nested",
        future.strftime("%Y%m%dT%H%M%SZ"),
        decision="PASS",
        created_utc=future.isoformat(),
    )
    assert next_immutable_event_created_utc(tmp_path, PREFIX) == future + timedelta(microseconds=1)
    red = _publish_event(tmp_path, future - timedelta(seconds=61), decision="FAIL")

    assert select_latest_immutable_event(tmp_path, PREFIX) == red
    assert require_newest_immutable_event(red, PREFIX) == red
    assert json.loads(_witness(red).read_text(encoding="utf-8"))["predecessor_count"] == 1
    assert legacy.is_file()


def test_shared_scope_rejects_any_different_reader_filter(tmp_path: Path) -> None:
    path = _publish_event(
        tmp_path / "candidate_a",
        datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc),
        decision="PASS",
        authority_root=tmp_path,
        scope_dir_glob="candidate_*",
    )
    for other_glob in (None, "*", "candidate_a"):
        with pytest.raises(ImmutableEventAuthorityError, match="authority scope mismatch"):
            require_newest_immutable_event(
                path, PREFIX, authority_root=tmp_path, scope_dir_glob=other_glob
            )


def test_scope_filter_excludes_unrelated_histories_but_not_new_matching_runs(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    _publish_event(tmp_path / "unrelated", created, decision="FAIL")
    first = _publish_event(
        tmp_path / "candidate_a", created, decision="PASS",
        authority_root=tmp_path, scope_dir_glob="candidate_*",
    )
    assert select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob="candidate_*") == first
    second = _publish_event(
        tmp_path / "candidate_b", created - timedelta(seconds=61), decision="FAIL",
        authority_root=tmp_path, scope_dir_glob="candidate_*",
    )
    assert select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob="candidate_*") == second
    assert json.loads(_witness(second).read_text(encoding="utf-8"))["predecessor_count"] == 1


@pytest.mark.parametrize("scope_dir_glob", [None, "candidate_*"])
def test_cross_run_fresh_publication_preserves_unordered_legacy_history(
    tmp_path: Path, scope_dir_glob: str | None
) -> None:
    green = _write_event(tmp_path / "candidate_a", "20260906T073032Z", decision="PASS")
    red = _write_event(tmp_path / "candidate_b", "20260906T072931Z", decision="FAIL")
    before = {path: path.read_bytes() for path in (green, red)}
    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob=scope_dir_glob)
    newest = _publish_event(
        tmp_path / "candidate_c", datetime(2026, 9, 6, 7, 29, tzinfo=timezone.utc),
        decision="FAIL", authority_root=tmp_path, scope_dir_glob=scope_dir_glob,
    )
    assert select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob=scope_dir_glob) == newest
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("scope_dir_glob", ["", ".", "..", "**", "candidate_*/*", "../*"])
def test_invalid_scope_glob_fails_before_creating_outputs(tmp_path: Path, scope_dir_glob: str) -> None:
    with pytest.raises(ImmutableEventAuthorityError, match="scope glob"):
        _publish_event(
            tmp_path / "candidate_a", datetime.now(timezone.utc), decision="PASS",
            authority_root=tmp_path, scope_dir_glob=scope_dir_glob,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("outside", ["authority", "outside", "authority/unmatched"])
def test_writer_rejects_output_outside_explicit_scope_before_publication(
    tmp_path: Path, outside: str
) -> None:
    with pytest.raises(ImmutableEventAuthorityError, match="outside its authority scope"):
        _publish_event(
            tmp_path / outside, datetime.now(timezone.utc), decision="PASS",
            authority_root=tmp_path / "authority", scope_dir_glob="candidate_*",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("scoped", [False, True])
def test_directory_symlinks_cannot_hide_events_from_scope(tmp_path: Path, scoped: bool) -> None:
    outside = tmp_path / "outside"
    _write_event(outside, "20260906T073032Z", decision="FAIL")
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    (authority_root / "candidate_alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ImmutableEventAuthorityError, match="symlink"):
        select_latest_immutable_event(
            authority_root, PREFIX, scope_dir_glob="candidate_*" if scoped else None
        )


def test_authority_root_and_output_symlink_aliases_are_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ImmutableEventAuthorityError, match="symlink"):
        select_latest_immutable_event(alias, PREFIX)
    with pytest.raises(ImmutableEventAuthorityError, match="symlink"):
        _publish_event(alias / "candidate_a", datetime.now(timezone.utc), decision="PASS")


def test_readonly_inventory_helper_exposes_complete_shared_scope(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    paths = [
        _publish_event(
            tmp_path / f"candidate_{index}", created - timedelta(seconds=61 * index),
            decision="FAIL", authority_root=tmp_path, scope_dir_glob="candidate_*",
        )
        for index in range(3)
    ]
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    inventory = validated_immutable_event_authority_inventory(paths[0], PREFIX)

    assert inventory["schema_version"] == "immutable_json_event_authority_inventory_v1"
    assert inventory["authority_root"] == str(tmp_path)
    assert inventory["event_prefix"] == PREFIX
    assert inventory["scope_dir_glob"] == "candidate_*"
    assert inventory["selected_event_path"] == str(paths[-1])
    assert inventory["event_paths"] == sorted(str(path) for path in paths)
    assert inventory["witness_paths"] == sorted(str(_witness(path)) for path in paths)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    with pytest.raises(ImmutableEventAuthorityError, match="authority scope mismatch"):
        validated_immutable_event_authority_inventory(paths[0], PREFIX, authority_root=paths[0].parent)


def test_inventory_helper_requires_explicit_scope_for_unwitnessed_legacy(tmp_path: Path) -> None:
    path = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    with pytest.raises(ImmutableEventAuthorityError, match="legacy event has no declared authority scope"):
        validated_immutable_event_authority_inventory(path, PREFIX)
    inventory = validated_immutable_event_authority_inventory(path, PREFIX, authority_root=tmp_path)
    assert inventory["event_paths"] == [str(path)]
    assert inventory["witness_paths"] == []


@pytest.mark.parametrize("discover_scope", [False, True])
def test_inventory_limits_accept_exact_complete_scope_boundaries(
    tmp_path: Path, discover_scope: bool,
) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    paths = [
        _publish_event(
            tmp_path / f"candidate_{index}", created - timedelta(seconds=61 * index),
            decision="FAIL", authority_root=tmp_path, scope_dir_glob="candidate_*",
        ) for index in range(2)
    ]
    expected = validated_immutable_event_authority_inventory(paths[0], PREFIX)
    sizes = [Path(path).stat().st_size for path in expected["event_paths"] + expected["witness_paths"]]
    options = {} if discover_scope else {"authority_root": tmp_path, "scope_dir_glob": "candidate_*"}
    actual = validated_immutable_event_authority_inventory(
        paths[0], PREFIX, **options,
        max_document_bytes=max(sizes), max_total_bytes=sum(sizes), max_events=len(paths),
    )
    assert actual == expected


@pytest.mark.parametrize("discover_scope", [False, True])
@pytest.mark.parametrize("limit", ["max_document_bytes", "max_total_bytes", "max_events"])
def test_inventory_limits_reject_whole_scope_before_decoding_historical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, discover_scope: bool, limit: str,
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    first, _ = write_immutable_json_event(
        tmp_path / "candidate_a", PREFIX,
        {"created_utc": created.isoformat(), "decision": "PASS", "fixture_padding": "x" * 4096},
        authority_root=tmp_path, scope_dir_glob="candidate_*",
    )
    last = _publish_event(
        tmp_path / "candidate_b", created - timedelta(seconds=61), decision="FAIL",
        authority_root=tmp_path, scope_dir_glob="candidate_*",
    )
    documents = [first, last, _witness(first), _witness(last)]
    bounds = {
        "max_document_bytes": first.stat().st_size - 1,
        "max_total_bytes": sum(path.stat().st_size for path in documents) - 1,
        "max_events": 1,
    }
    decoded = []
    original_read = authority._read_json_object

    def record_decode(path: Path, **options: object) -> tuple[dict[str, object], bytes]:
        decoded.append(path)
        return original_read(path, **options)

    monkeypatch.setattr(authority, "_read_json_object", record_decode)
    options = {} if discover_scope else {"authority_root": tmp_path, "scope_dir_glob": "candidate_*"}
    with pytest.raises(ImmutableEventAuthorityError, match=limit):
        validated_immutable_event_authority_inventory(last, PREFIX, **options, **{limit: bounds[limit]})
    assert decoded == ([_witness(last)] if discover_scope else [])


@pytest.mark.parametrize("limit", ["max_document_bytes", "max_total_bytes"])
def test_scope_discovery_witness_obeys_limits_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    path = _publish_event(tmp_path, datetime.now(timezone.utc), decision="FAIL")
    witness_size = _witness(path).stat().st_size

    def reject_decode(*args: object, **options: object) -> None:
        pytest.fail("budget rejection must precede bootstrap JSON decoding")

    monkeypatch.setattr(authority, "_read_json_object", reject_decode)
    with pytest.raises(ImmutableEventAuthorityError, match=limit):
        validated_immutable_event_authority_inventory(path, PREFIX, **{limit: witness_size - 1})


@pytest.mark.parametrize("limit", ["max_document_bytes", "max_total_bytes", "max_events"])
@pytest.mark.parametrize("invalid", [True, -1, 1.5, "1"])
def test_inventory_limits_require_caller_owned_nonnegative_integers(
    tmp_path: Path, limit: str, invalid: object,
) -> None:
    with pytest.raises(ImmutableEventAuthorityError, match=f"{limit} must be a nonnegative integer"):
        validated_immutable_event_authority_inventory(tmp_path / "absent.json", PREFIX, **{limit: invalid})


@pytest.mark.parametrize("witness", [False, True])
@pytest.mark.parametrize("during_read", [False, True])
def test_inventory_bounded_reads_reject_growth_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, witness: bool, during_read: bool,
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    path = _publish_event(tmp_path, datetime.now(timezone.utc), decision="FAIL")
    target = _witness(path) if witness else path
    original_size = target.stat().st_size
    identity = target.stat()
    original_read = authority._read_json_object
    original_fstat = os.fstat
    original_fdopen = os.fdopen
    observed_bounds = []
    observed_read_sizes = []

    def grow_document() -> None:
        with target.open("ab") as handle:
            handle.write(b" ")

    def capture_bounded_read(document: Path, *, max_bytes: int | None = None) -> tuple[dict[str, object], bytes]:
        if document == target:
            observed_bounds.append(max_bytes)
            if not during_read:
                grow_document()
        return original_read(document, max_bytes=max_bytes)

    def grow_after_fstat(descriptor: int) -> os.stat_result:
        observed = original_fstat(descriptor)
        if observed.st_ino == identity.st_ino and observed.st_dev == identity.st_dev:
            grow_document()
        return observed

    def track_fdopen(descriptor: int, mode: str) -> object:
        handle = original_fdopen(descriptor, mode)
        observed = original_fstat(descriptor)
        if observed.st_ino != identity.st_ino or observed.st_dev != identity.st_dev:
            return handle

        class BoundedReader:
            def __enter__(self) -> BoundedReader:
                return self

            def __exit__(self, *args: object) -> object:
                return handle.__exit__(*args)

            def fileno(self) -> int:
                return handle.fileno()

            def read(self, size: int = -1) -> bytes:
                observed_read_sizes.append(size)
                assert size == original_size + 1
                return handle.read(size)

        return BoundedReader()

    monkeypatch.setattr(authority, "_read_json_object", capture_bounded_read)
    monkeypatch.setattr(os, "fdopen", track_fdopen)
    if during_read:
        monkeypatch.setattr(os, "fstat", grow_after_fstat)
    with pytest.raises(ImmutableEventAuthorityError, match="exceeds bounded read"):
        validated_immutable_event_authority_inventory(path, PREFIX, authority_root=tmp_path, max_events=1)
    assert observed_bounds == [original_size]
    assert observed_read_sizes == ([original_size + 1] if during_read else [])


@pytest.mark.parametrize("stamp", ["20260906T072931Z", "20260906T073133Z"])
def test_unwitnessed_addition_cannot_hide_behind_witnessed_green(
    tmp_path: Path, stamp: str
) -> None:
    _publish_event(tmp_path, datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc), decision="PASS")
    _write_event(tmp_path, stamp, decision="FAIL")

    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        select_latest_immutable_event(tmp_path, PREFIX)
    with pytest.raises(ImmutableEventAuthorityError, match="publication order is unproven"):
        next_immutable_event_created_utc(tmp_path, PREFIX)


@pytest.mark.parametrize("offset", [-61, 61])
def test_malformed_published_red_blocks_even_below_green_timestamp(
    tmp_path: Path, offset: int
) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    _publish_event(tmp_path, created, decision="PASS")
    red = _publish_event(tmp_path, created + timedelta(seconds=offset), decision="FAIL")
    red.write_text("{", encoding="utf-8")

    with pytest.raises(ImmutableEventAuthorityError, match="invalid immutable event JSON"):
        select_latest_immutable_event(tmp_path, PREFIX)


@pytest.mark.parametrize("corruption", ["payload", "missing_predecessor", "missing_witness"])
def test_changed_or_incomplete_witnessed_history_blocks(
    tmp_path: Path, corruption: str
) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    older = _publish_event(tmp_path, created, decision="PASS")
    newer = _publish_event(tmp_path, created + timedelta(seconds=1), decision="FAIL")
    if corruption == "payload":
        payload = json.loads(older.read_text(encoding="utf-8"))
        payload["decision"] = "FAIL"
        older.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "missing_predecessor":
        older.unlink()
        _witness(older).unlink()
    else:
        _witness(newer).unlink()

    with pytest.raises(ImmutableEventAuthorityError):
        select_latest_immutable_event(tmp_path, PREFIX)
    with pytest.raises(ImmutableEventAuthorityError):
        _publish_event(tmp_path, created + timedelta(seconds=2), decision="PASS")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "unrecognized"),
        ("json_path", "/wrong.json"),
        ("sha256", "0" * 64),
        ("authority_root", "/incorrect/authority"),
        ("scope_dir_glob", "candidate_*"),
        ("predecessors_sha256", "0" * 64),
        ("predecessor_count", True),
        ("predecessor_count", -1),
        ("predecessor_count", 2),
        ("unexpected", "field"),
    ],
)
def test_invalid_publication_witness_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _publish_event(
        tmp_path, datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc), decision="PASS"
    )
    witness_path = _witness(path)
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    witness[field] = value
    witness_path.write_text(json.dumps(witness), encoding="utf-8")

    with pytest.raises(ImmutableEventAuthorityError):
        select_latest_immutable_event(tmp_path, PREFIX)


@pytest.mark.parametrize("corruption", ["malformed_json", "symlink"])
def test_unreadable_or_aliased_order_witness_is_not_legacy_fallback(
    tmp_path: Path, corruption: str
) -> None:
    path = _publish_event(
        tmp_path, datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc), decision="PASS"
    )
    witness_path = _witness(path)
    if corruption == "malformed_json":
        witness_path.write_text("{", encoding="utf-8")
    else:
        target = tmp_path / "witness-copy.json"
        target.write_bytes(witness_path.read_bytes())
        witness_path.unlink()
        witness_path.symlink_to(target)

    with pytest.raises(ImmutableEventAuthorityError):
        select_latest_immutable_event(tmp_path, PREFIX)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("created_utc", "2026-09-06T07:30:32"),
        ("created_utc", "not a timestamp"),
        ("json_path", "relative.json"),
        ("json_path", "/incorrect/self-reference.json"),
    ],
)
def test_legacy_singleton_still_requires_exact_reader_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    path = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImmutableEventAuthorityError):
        select_latest_immutable_event(tmp_path, PREFIX)


@pytest.mark.parametrize(
    "invalid",
    ['"extra": NaN', '"extra": Infinity', '"extra": 1e999', '"decision": "FAIL"'],
)
def test_reader_rejects_non_strict_json_even_for_legacy_singleton(
    tmp_path: Path, invalid: str
) -> None:
    path = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    encoded = path.read_text(encoding="utf-8")
    path.write_text(encoded[:-1] + ", " + invalid + "}", encoding="utf-8")

    with pytest.raises(ImmutableEventAuthorityError, match="invalid immutable event JSON"):
        select_latest_immutable_event(tmp_path, PREFIX)


def test_serial_allocator_remains_monotone_across_clock_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    wall_time = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)

    class ControlledDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return wall_time

    monkeypatch.setattr(authority, "datetime", ControlledDatetime)
    created_times = []
    for sequence in range(4):
        created = next_immutable_event_created_utc(tmp_path, PREFIX)
        path = _publish_event(tmp_path, created, decision="FAIL")
        assert select_latest_immutable_event(tmp_path, PREFIX) == path
        assert json.loads(_witness(path).read_text(encoding="utf-8"))["predecessor_count"] == sequence
        created_times.append(created)
        wall_time -= timedelta(seconds=61)

    assert created_times == [created_times[0] + timedelta(microseconds=step) for step in range(4)]


def test_allocator_preserves_full_legacy_microseconds_and_excludes_mirrors(
    tmp_path: Path,
) -> None:
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(microsecond=987654)
    _write_event(
        tmp_path,
        future.strftime("%Y%m%dT%H%M%SZ"),
        decision="PASS",
        created_utc=future.isoformat(),
    )
    for suffix in ("latest", "MANIFEST"):
        (tmp_path / f"{PREFIX}_{suffix}.json").write_text("not authority", encoding="utf-8")

    assert next_immutable_event_created_utc(tmp_path, PREFIX) == future + timedelta(microseconds=1)
    assert next_immutable_event_created_utc(tmp_path, PREFIX, future + timedelta(seconds=1)) == (
        future + timedelta(seconds=1, microseconds=1)
    )
    with pytest.raises(ImmutableEventAuthorityError, match="not timezone-aware"):
        next_immutable_event_created_utc(tmp_path, PREFIX, future.replace(tzinfo=None))


@pytest.mark.parametrize("shared_scope", [False, True])
def test_interrupted_publication_leaves_durable_fail_closed_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_scope: bool
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    options = {"authority_root": tmp_path, "scope_dir_glob": "candidate_*"} if shared_scope else {}
    first_dir = tmp_path / "candidate_a" if shared_scope else tmp_path
    next_dir = tmp_path / "candidate_b" if shared_scope else tmp_path
    green = _publish_event(first_dir, created, decision="PASS", **options)
    original_publish = authority._publish_file_noreplace

    def fail_event_publish(source: Path, destination: Path) -> None:
        if destination.name.endswith(".json"):
            assert _witness(destination).is_file()
            raise OSError("injected event publication interruption")
        original_publish(source, destination)

    monkeypatch.setattr(authority, "_publish_file_noreplace", fail_event_publish)
    with pytest.raises(OSError, match="injected event publication interruption"):
        _publish_event(next_dir, created - timedelta(seconds=61), decision="FAIL", **options)
    monkeypatch.setattr(authority, "_publish_file_noreplace", original_publish)

    assert list(tmp_path.rglob(f"{PREFIX}_*.json")) == [green]
    assert len(list(tmp_path.rglob(f".{PREFIX}_*.json.order"))) == 2
    assert not list(tmp_path.rglob(".*.staging.*"))
    with pytest.raises(ImmutableEventAuthorityError, match="incomplete immutable publication"):
        select_latest_immutable_event(tmp_path, PREFIX, scope_dir_glob=options.get("scope_dir_glob"))
    with pytest.raises(ImmutableEventAuthorityError, match="incomplete immutable publication"):
        validated_immutable_event_authority_inventory(green, PREFIX)
    with pytest.raises(ImmutableEventAuthorityError, match="incomplete immutable publication"):
        _publish_event(next_dir, created + timedelta(seconds=1), decision="PASS", **options)


@pytest.mark.parametrize("shared_scope", [False, True])
def test_reader_waits_for_serialized_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_scope: bool
) -> None:
    from gx1.contracts import immutable_event_authority_v1 as authority

    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    options = {"authority_root": tmp_path, "scope_dir_glob": "candidate_*"} if shared_scope else {}
    first_dir = tmp_path / "candidate_a" if shared_scope else tmp_path
    next_dir = tmp_path / "candidate_b" if shared_scope else tmp_path
    _publish_event(first_dir, created, decision="PASS", **options)
    publishing = Event()
    release = Event()
    reading = Event()
    read_done = Event()
    original_publish = authority._publish_file_noreplace

    def pause_event_publish(source: Path, destination: Path) -> None:
        if destination.name.endswith(".json"):
            publishing.set()
            assert release.wait(timeout=10)
        original_publish(source, destination)

    def read_authority() -> Path | None:
        reading.set()
        try:
            return select_latest_immutable_event(
                tmp_path, PREFIX, scope_dir_glob=options.get("scope_dir_glob")
            )
        finally:
            read_done.set()

    monkeypatch.setattr(authority, "_publish_file_noreplace", pause_event_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(
            _publish_event, next_dir, created - timedelta(seconds=61), decision="FAIL", **options
        )
        try:
            assert publishing.wait(timeout=10)
            reader = pool.submit(read_authority)
            assert reading.wait(timeout=10)
            assert not read_done.wait(timeout=0.1)
        finally:
            release.set()
        red = writer.result(timeout=10)
        assert reader.result(timeout=10) == red


def _writer_command(
    root: Path, created: datetime, *, authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
) -> list[str]:
    source = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event\n"
        "path, event = write_immutable_json_event(Path(sys.argv[1]), sys.argv[2], "
        "{'created_utc': sys.argv[3], 'decision': 'FAIL'}, **json.loads(sys.argv[4]))\n"
        "print(path)\n"
    )
    options = {
        "authority_root": str(authority_root) if authority_root is not None else None,
        "scope_dir_glob": scope_dir_glob,
    }
    return [sys.executable, "-c", source, str(root), PREFIX, created.isoformat(), json.dumps(options)]


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("shared_scope", [False, True])
def test_process_writers_serialize_or_fail_no_replace(
    tmp_path: Path, collision: bool, shared_scope: bool
) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    commands = [
        _writer_command(
            tmp_path / f"candidate_{offset}" if shared_scope else tmp_path,
            created - timedelta(seconds=0 if collision else 61 * offset),
            authority_root=tmp_path if shared_scope else None,
            scope_dir_glob="candidate_*" if shared_scope else None,
        )
        for offset in range(4)
    ]
    processes = [
        subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for command in commands
    ]
    try:
        outputs = [process.communicate(timeout=20) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    successes = [
        Path(stdout.strip())
        for process, (stdout, _) in zip(processes, outputs)
        if process.returncode == 0
    ]

    assert len(successes) == (1 if collision and not shared_scope else 4), outputs
    for process, (_, stderr) in zip(processes, outputs):
        if process.returncode != 0:
            assert collision and not shared_scope and "immutable event already exists" in stderr
    witnessed_counts = sorted(
        json.loads(_witness(path).read_text(encoding="utf-8"))["predecessor_count"]
        for path in successes
    )
    assert witnessed_counts == list(range(len(successes)))
    selected = select_latest_immutable_event(
        tmp_path, PREFIX, scope_dir_glob="candidate_*" if shared_scope else None
    )
    assert selected in successes
    assert json.loads(_witness(selected).read_text(encoding="utf-8"))["predecessor_count"] == (
        len(successes) - 1
    )


def test_fresh_process_restarts_after_clock_rollback(tmp_path: Path) -> None:
    created = datetime(2026, 9, 6, 7, 30, tzinfo=timezone.utc)
    green = _publish_event(tmp_path, created, decision="PASS")
    completed = subprocess.run(
        _writer_command(tmp_path, created - timedelta(seconds=61)),
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    red = Path(completed.stdout.strip())

    assert red != green
    assert select_latest_immutable_event(tmp_path, PREFIX) == red
    assert next_immutable_event_created_utc(tmp_path, PREFIX, created) > created


def test_identical_legacy_timestamp_can_be_disambiguated_only_by_publication_proof(
    tmp_path: Path,
) -> None:
    legacy = _write_event(tmp_path, "20260906T073032Z", decision="PASS")
    witnessed = _publish_event(
        tmp_path, datetime(2026, 9, 6, 7, 30, 32, tzinfo=timezone.utc), decision="FAIL"
    )

    assert witnessed != legacy
    assert select_latest_immutable_event(tmp_path, PREFIX) == witnessed


@pytest.mark.parametrize("invalid", [Path("soft-string-pass-through"), float("nan")])
def test_immutable_event_writer_rejects_non_json_payload_without_artifact(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(ImmutableEventAuthorityError, match="not strict JSON"):
        write_immutable_json_event(
            tmp_path,
            PREFIX,
            {
                "created_utc": "2026-07-16T11:00:00.123456+00:00",
                "invalid": invalid,
            },
        )

    assert list(tmp_path.iterdir()) == []


_MIGRATED_PRODUCERS = (
    "audit_xau_direction_repair_pretrain_v1",
    "audit_entry_specialist_feature_groups_v1",
    "audit_entry_foundation_targets_v1",
    "audit_entry_foundation_features_v1",
    "evaluate_entry_candidate_selective_edge_v1",
)


def _producer_ast(name: str) -> ast.Module:
    path = Path(__file__).resolve().parents[1] / "gx1" / "scripts" / f"{name}.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _execute_statements(statements: list[ast.stmt], namespace: dict[str, object]) -> None:
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *statements],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<publication-statements>", "exec"), namespace)


def _sidecar_owners() -> dict[str, object]:
    parsed = _producer_ast("entry_candidate_prediction_evidence_v1")
    functions = [
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"atomic_write_text", "atomic_write_parquet_immutable"}
    ]
    assert len(functions) == 2
    namespace = {"Path": Path, "os": os, "tempfile": tempfile}
    _execute_statements(functions, namespace)
    return namespace


@pytest.mark.parametrize("producer", _MIGRATED_PRODUCERS)
def test_migrated_publication_statement_preserves_schema_order_and_existing_bytes(
    tmp_path: Path, producer: str,
) -> None:
    parsed = _producer_ast(producer)
    statements = [
        node for node in ast.walk(parsed)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "write_immutable_json_event"
    ]
    assert len(statements) == 1
    prefix = ast.literal_eval(statements[0].value.args[1])
    event_variable = statements[0].targets[0].elts[1].id
    namespace = {
        "out_dir": tmp_path,
        "json": json,
        "write_immutable_json_event": write_immutable_json_event,
        "_json_default": str,
    }
    created = datetime(2026, 9, 6, 7, 30, 32, 123456, tzinfo=timezone.utc)
    for decision, observed in (("PASS", created), ("FAIL", created - timedelta(seconds=61))):
        original = {"created_utc": observed.isoformat(), "decision": decision, "schema_version": "fixture_v1"}
        namespace["report"] = original
        _execute_statements(statements, namespace)
        event = namespace[event_variable]
        path = Path(event["json_path"])
        assert {key: event[key] for key in original} == original
        assert require_newest_immutable_event(path, prefix) == path
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ImmutableEventAuthorityError, match="already exists"):
        _execute_statements(statements, namespace)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    namespace["report"] = {**namespace["report"], "invalid": float("nan")}
    with pytest.raises(ValueError, match="Out of range float"):
        _execute_statements(statements, namespace)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("suffix", ["csv", "json", "md", "parquet"])
@pytest.mark.parametrize("racing_collision", [False, True])
def test_existing_sidecar_owners_never_replace_bytes_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, racing_collision: bool,
) -> None:
    namespace = _sidecar_owners()
    path = tmp_path / f"sidecar_20260906T073032123456Z.{suffix}"
    retained = b"retained original sidecar bytes"
    if racing_collision:
        original_link = os.link

        def publish_competing_file(source: Path, destination: Path) -> None:
            destination.write_bytes(retained)
            original_link(source, destination)

        monkeypatch.setattr(os, "link", publish_competing_file)
    else:
        path.write_bytes(retained)
    if suffix == "parquet":
        import pandas as pd

        with pytest.raises((RuntimeError, FileExistsError)):
            namespace["atomic_write_parquet_immutable"](pd.DataFrame({"value": [1.0]}), path)
    else:
        with pytest.raises((RuntimeError, FileExistsError)):
            namespace["atomic_write_text"](path, "replacement refused")
    assert path.read_bytes() == retained
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("producer", _MIGRATED_PRODUCERS[1:])
def test_migrated_markdown_publication_uses_existing_no_replace_owner(
    tmp_path: Path, producer: str,
) -> None:
    functions = [
        node for node in _producer_ast(producer).body
        if isinstance(node, ast.FunctionDef) and node.name == "_write_markdown"
    ]
    assert len(functions) == 1
    publication = functions[0].body[-1]
    assert isinstance(publication, ast.Expr)
    assert isinstance(publication.value, ast.Call)
    assert isinstance(publication.value.func, ast.Name)
    assert publication.value.func.id == "atomic_write_text"
    path = tmp_path / "event.md"
    namespace = {**_sidecar_owners(), "path": path, "lines": ["original markdown"]}
    _execute_statements([publication], namespace)
    original = path.read_bytes()
    namespace["lines"] = ["must not replace original"]
    with pytest.raises(RuntimeError, match="already exists"):
        _execute_statements([publication], namespace)
    assert path.read_bytes() == original


def test_evaluator_summary_path_does_not_overlap_authority_prefix(tmp_path: Path) -> None:
    parsed = _producer_ast("evaluate_entry_candidate_selective_edge_v1")
    assignments = [
        node for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "summary_path" for target in node.targets)
    ]
    assert len(assignments) == 1
    prefix = "ENTRY_CANDIDATE_SELECTIVE_EDGE"
    namespace = {"out_dir": tmp_path, "timestamp": "20260906T073032123456Z"}
    _execute_statements(assignments, namespace)
    summary_path = namespace["summary_path"]
    assert not summary_path.name.startswith(f"{prefix}_")
    _sidecar_owners()["atomic_write_text"](summary_path, "{}\n")
    path, _ = write_immutable_json_event(
        tmp_path, prefix,
        {"created_utc": "2026-09-06T07:30:32.123456+00:00", "summary_path": str(summary_path)},
    )
    assert select_latest_immutable_event(tmp_path, prefix) == path


def test_live_gate_report_producers_publish_immutable_authority() -> None:
    parity = Path("gx1/scripts/verify_model_native_serve_parity_v1.py").read_text(
        encoding="utf-8"
    )
    pocket = Path(
        "gx1/scripts/audit_model_native_direction_pockets_v1.py"
    ).read_text(encoding="utf-8")

    for source, prefix in (
        (parity, "MODEL_NATIVE_SERVE_PARITY"),
        (pocket, "MODEL_NATIVE_DIRECTION_POCKET_AUDIT"),
    ):
        assert "write_immutable_json_event(" in source
        assert f'"{prefix}_latest.json").write_text' not in source
    assert "replace_latest_json_mirror(" not in parity
    assert "replace_latest_json_mirror(" not in pocket
    assert "_latest.json" not in pocket
    assert "_latest.md" not in pocket
    assert 'report["created_utc"] = datetime.now(timezone.utc).isoformat()' in parity
    assert '"started_utc": datetime.now(timezone.utc).isoformat()' in parity

    prediction_producer = Path(
        "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py"
    ).read_text(encoding="utf-8")
    assert "event_created_utc = datetime.now(timezone.utc)" in prediction_producer
    assert 'timestamp = event_created_utc.strftime("%Y%m%dT%H%M%S%fZ")' in prediction_producer
    assert '"created_utc": event_created_utc.isoformat()' in prediction_producer


def test_control_surface_accepts_only_explicit_immutable_event_inputs() -> None:
    source = Path("scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")

    assert "readiness-report" not in source
    assert "select_latest_immutable_event" not in source
    assert "st_mtime" not in source
    assert "_latest.json" in source
    assert "mutable latest input is forbidden" in source
    assert "--rebuild-preflight-json" in source
    assert "--candidate-readiness-json" in source
    assert "--out-dir" in source
    assert "write_text(" not in source
    assert "replace_latest_json_mirror" not in source
