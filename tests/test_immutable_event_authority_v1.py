from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    next_immutable_event_created_utc,
    require_newest_immutable_event,
    select_latest_immutable_event,
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


def test_newest_filename_event_wins_even_when_older_green_has_newer_mtime(tmp_path: Path) -> None:
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

    selected = select_latest_immutable_event(
        tmp_path,
        PREFIX,
        scope_dir_glob="model_native_seq513_candidate_*",
    )

    assert selected == newer_red.resolve()
    assert json.loads(selected.read_text(encoding="utf-8"))["decision"] == "FAIL"


def test_explicit_event_must_be_newest_in_its_authority_directory(tmp_path: Path) -> None:
    older_green = _write_event(tmp_path, "20260716T100000Z", decision="PASS")
    newer_red = _write_event(tmp_path, "20260716T110000Z", decision="FAIL")

    with pytest.raises(ImmutableEventAuthorityError, match="not the newest"):
        require_newest_immutable_event(older_green, PREFIX)
    assert require_newest_immutable_event(newer_red, PREFIX) == newer_red.resolve()


def test_explicit_event_authority_rejects_symlink_alias(tmp_path: Path) -> None:
    event = _write_event(tmp_path, "20260716T110000Z", decision="PASS")
    alias = tmp_path / "alias.json"
    alias.symlink_to(event)

    with pytest.raises(ImmutableEventAuthorityError, match="not a regular file"):
        require_newest_immutable_event(alias, PREFIX)


def test_authority_selection_uses_immutable_timestamp_not_mtime(tmp_path: Path) -> None:
    older = _write_event(tmp_path / "variant", "20260716T100000Z", decision="PASS")
    newer = _write_event(tmp_path, "20260716T110000Z", decision="FAIL")
    os.utime(older, ns=(2_000_000_000, 2_000_000_000))
    os.utime(newer, ns=(1_000_000_000, 1_000_000_000))

    selected = select_latest_immutable_event(tmp_path, PREFIX)

    assert selected == newer.resolve()


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


def test_duplicate_newest_timestamp_across_candidates_fails_closed(tmp_path: Path) -> None:
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

    with pytest.raises(ImmutableEventAuthorityError, match="duplicate newest"):
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

    with pytest.raises(ImmutableEventAuthorityError, match="already exists"):
        write_immutable_json_event(
            tmp_path,
            PREFIX,
            {"created_utc": created_utc, "decision": "PASS"},
        )


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
