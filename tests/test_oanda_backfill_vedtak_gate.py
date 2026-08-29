from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from gx1.contracts import xau_tape_provenance_v1 as tape_contract
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    CANONICAL_NATIVE_SUCCESSOR_MODE,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    NATIVE_TIMEFRAME_POLICY,
    canonical_json_sha256,
    canonical_native_frame_from_oanda_response,
    canonical_native_rows_bytes,
    canonical_xau_source_descriptor_v1,
)
from gx1.scripts import backfill_xauusd_m5_from_oanda as canonical_backfill
from gx1.execution import v12_m1_to_m5_downsample as m1_downsample
from gx1_guards.gates import GateError


class _FakeOandaClient:
    env = "practice"
    base_url = "https://api-fxpractice.oanda.com/v3"

    def __init__(
        self,
        *,
        fail: bool = False,
        incomplete_only: bool = False,
        incomplete_position: int | None = None,
        bad_geometry: bool = False,
        timeframe: str = "M5",
        response_timeframe: str | None = None,
        price_offset: float = 0.0,
    ) -> None:
        self.fail = fail
        self.incomplete_only = incomplete_only
        self.incomplete_position = incomplete_position
        self.bad_geometry = bad_geometry
        self.timeframe = timeframe
        self.response_timeframe = response_timeframe or timeframe
        self.price_offset = float(price_offset)
        self.requests: list[dict[str, Any]] = []

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(
            {"method": method, "path": path, "params": dict(params)}
        )
        if self.fail:
            raise RuntimeError("network unavailable")
        start = pd.Timestamp(params["from"])
        end = pd.Timestamp(params["to"])
        bar_minutes = 1 if self.timeframe == "M1" else 5
        timestamps = pd.date_range(
            start,
            end - pd.Timedelta(minutes=bar_minutes),
            freq=f"{bar_minutes}min",
        )
        candles = []
        for position, timestamp in enumerate(timestamps):
            mid_high = (
                "1998.0"
                if self.bad_geometry and position == 0
                else str(2002.0 + self.price_offset)
            )
            candles.append(
                {
                    "complete": not (
                        self.incomplete_only
                        or position == self.incomplete_position
                    ),
                    "time": timestamp.isoformat().replace("+00:00", "Z"),
                    "volume": 10 + position,
                    "mid": {
                        "o": str(2000.0 + self.price_offset),
                        "h": mid_high,
                        "l": str(1999.0 + self.price_offset),
                        "c": str(2001.0 + self.price_offset),
                    },
                    "bid": {
                        "o": str(1999.5 + self.price_offset),
                        "h": str(2001.5 + self.price_offset),
                        "l": str(1998.5 + self.price_offset),
                        "c": str(2000.5 + self.price_offset),
                    },
                    "ask": {
                        "o": str(2000.5 + self.price_offset),
                        "h": str(2002.5 + self.price_offset),
                        "l": str(1999.5 + self.price_offset),
                        "c": str(2001.5 + self.price_offset),
                    },
                }
            )
        return {
            "instrument": "XAU_USD",
            "granularity": self.response_timeframe,
            "candles": candles,
        }


def _allow_clean_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        canonical_backfill,
        "_require_clean_repository",
        lambda _root, *, timeframe: "a" * 40,
    )


def materialize_native_xau_test_bundle(
    output: Path,
    *,
    timeframe: str = "M5",
    end_utc: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Build a real strict native bundle for downstream provenance fixtures."""

    with mock.patch.object(
        canonical_backfill,
        "_require_clean_repository",
        return_value="a" * 40,
    ):
        bar_minutes = 1 if timeframe == "M1" else 5
        return canonical_backfill.materialize_native_xau_snapshot(
            client=_FakeOandaClient(timeframe=timeframe),
            timeframe=timeframe,
            vedtak_id=f"XAU_NATIVE_{timeframe}_FIXTURE_V3",
            start_utc="2026-01-01T00:00:00Z",
            end_utc=(
                pd.Timestamp(end_utc)
                if end_utc is not None
                else pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(minutes=bar_minutes)
            ),
            out_root=output,
        )


def test_backfill_cli_requires_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = canonical_backfill
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2


def test_backfill_cli_rejects_invalid_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = canonical_backfill
    argv = [str(module.__file__), "--vedtak", "short"]
    argv.extend(["--publication-mode", "bootstrap"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(GateError, match="blocked"):
        module.main()


def test_backfill_cli_rejects_unapproved_history_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = canonical_backfill
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--vedtak",
            "XAU_NATIVE_M5_SCOPE_TEST_001",
            "--publication-mode",
            "bootstrap",
            "--timeframe",
            "M5",
            "--start-utc",
            "2026-01-01T00:00:00Z",
            "--end-utc",
            "2026-01-02T00:00:00Z",
            "--out-root",
            "/tmp/gx1-oanda-scope-test",
        ],
    )
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before offline scope gate"),
    )

    with pytest.raises(GateError, match="GX1_OANDA_HISTORY_INGEST_FORBIDDEN"):
        module.main()


def test_backfill_cli_approved_pretest_history_authorization_precedes_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = canonical_backfill
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--vedtak",
            "OANDA_M5_PRETEST_CURRENT_20260829",
            "--publication-mode",
            "bootstrap",
            "--timeframe",
            "M5",
            "--start-utc",
            "2019-01-01T00:00:00Z",
            "--end-utc",
            "2026-07-01T00:00:00Z",
            "--out-root",
            "/tmp/gx1-oanda-authorized-pretest-test",
        ],
    )
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("authorization should permit the next setup stage"),
    )

    with pytest.raises(pytest.fail.Exception, match="authorization should permit"):
        module.main()


def test_history_authorization_accepts_only_the_bound_successor_mode() -> None:
    from gx1.contracts.oanda_history_ingest_approval_v1 import (
        require_approved_oanda_history_ingest,
    )

    assert require_approved_oanda_history_ingest(
        vedtak_id="OANDA_M5_PRETEST_CURRENT_20260829",
        timeframe="M5",
        publication_mode=CANONICAL_NATIVE_SUCCESSOR_MODE,
        start_utc=None,
        end_utc="2026-08-29T14:55:00Z",
    ) == "OANDA_M5_PRETEST_CURRENT_20260829"


def test_m1_downsample_cannot_write_canonical_m5_root() -> None:
    with pytest.raises(RuntimeError, match="CANONICAL_M5_SINGLE_OWNER_VIOLATION"):
        m1_downsample.main()


@pytest.mark.parametrize("timeframe", ["M1", "M5"])
def test_native_materialization_is_source_bound_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeframe: str,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / f"native_{timeframe.lower()}"
    client = _FakeOandaClient(timeframe=timeframe)
    bar_minutes = 1 if timeframe == "M1" else 5

    manifest = canonical_backfill.materialize_native_xau_snapshot(
        client=client,
        timeframe=timeframe,
        vedtak_id=f"XAU_NATIVE_{timeframe}_TEST_001",
        start_utc="2024-12-31T23:59:00Z"
        if timeframe == "M1"
        else "2024-12-31T23:55:00Z",
        end_utc="2025-01-01T00:02:00Z"
        if timeframe == "M1"
        else "2025-01-01T00:10:00Z",
        out_root=output,
    )

    assert output.is_dir()
    assert manifest["row_count"] == 3
    assert manifest["year_rows"] == {"year=2024": 1, "year=2025": 2}
    assert manifest["request_chunk_days"] == NATIVE_TIMEFRAME_POLICY[timeframe][
        "request_chunk_days"
    ]
    assert manifest["schema_required_cols"] == list(
        CANONICAL_NATIVE_REQUIRED_COLUMNS
    )
    assert list(output.glob(".*.staging.*")) == []
    assert [request["params"]["price"] for request in client.requests] == ["MBA"]
    descriptor = canonical_xau_source_descriptor_v1(
        output,
        timeframe=timeframe,
    )
    assert descriptor["row_count"] == 3
    assert descriptor["canonical_rows_sha256"] == manifest["canonical_rows_sha256"]
    assert descriptor["bar_duration_seconds"] == bar_minutes * 60

    chunk_path = output / manifest["source_chunks"][0]["relative_path"]
    payload = json.loads(gzip.decompress(chunk_path.read_bytes()).decode("utf-8"))
    assert payload["request"]["instrument"] == "XAU_USD"
    assert payload["response"]["candles"][0]["complete"] is True


def test_native_successor_reuses_history_and_refetches_only_exact_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m1_parent"
    child = tmp_path / "native_m1_child"
    vedtak = "XAU_NATIVE_M1_SUCCESSOR_TEST_001"
    parent_client = _FakeOandaClient(timeframe="M1")
    parent_manifest = canonical_backfill.materialize_native_xau_snapshot(
        client=parent_client,
        timeframe="M1",
        vedtak_id=vedtak,
        start_utc="2024-12-30T23:59:00Z",
        end_utc="2025-01-03T00:00:00Z",
        out_root=parent,
    )
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M1",
    )
    successor_client = _FakeOandaClient(timeframe="M1")

    child_manifest = canonical_backfill.materialize_native_xau_successor(
        client=successor_client,
        timeframe="M1",
        vedtak_id=vedtak,
        end_utc="2025-01-03T00:03:00Z",
        out_root=child,
        parent_root=parent,
        expected_parent_manifest_sha256=parent_descriptor["manifest_sha256"],
    )
    child_descriptor = canonical_xau_source_descriptor_v1(
        child,
        timeframe="M1",
    )

    overlap_start = parent_manifest["source_chunks"][-1][
        "request_from_utc"
    ]
    assert len(parent_client.requests) == 2
    assert len(successor_client.requests) == 1
    assert successor_client.requests[0]["params"]["from"] == pd.Timestamp(
        overlap_start
    ).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    assert successor_client.requests[0]["params"]["from"] != (
        "2024-12-30T23:59:00.000000000Z"
    )
    assert child_manifest["schema_version"] == (
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
    )
    assert child_manifest["parent_source"]["root"] == str(parent)
    assert child_manifest["parent_source"]["manifest_sha256"] == (
        parent_descriptor["manifest_sha256"]
    )
    assert child_manifest["successor_append"] == {
        "overlap_start_utc": pd.Timestamp(overlap_start).isoformat(),
        "parent_end_utc_exclusive": "2025-01-03T00:00:00+00:00",
        "reused_source_chunks": 1,
        "refetched_source_chunks": 1,
        "parent_overlap_rows": 1,
        "appended_rows": 3,
        "overlap_rows_sha256": child_manifest["successor_append"][
            "overlap_rows_sha256"
        ],
    }
    assert child_descriptor["row_count"] == parent_descriptor["row_count"] + 3
    assert child_descriptor["requested_start_utc"] == (
        parent_descriptor["requested_start_utc"]
    )
    assert child_descriptor["requested_end_utc_exclusive"] == (
        "2025-01-03T00:03:00+00:00"
    )


def test_native_successor_rejects_wrong_parent_cas_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    client = _FakeOandaClient(timeframe="M5")
    child = tmp_path / "native_m5_child"

    with pytest.raises(RuntimeError, match="PARENT_MANIFEST_CAS_MISMATCH"):
        canonical_backfill.materialize_native_xau_successor(
            client=client,
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc="2026-01-01T00:10:00Z",
            out_root=child,
            parent_root=parent,
            expected_parent_manifest_sha256="f" * 64,
        )

    assert client.requests == []
    assert not child.exists()


def test_native_successor_rejects_output_nested_under_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )
    child = parent / "nested_child"
    client = _FakeOandaClient(timeframe="M5")

    with pytest.raises(RuntimeError, match="OUTPUT_PARENT_INVALID"):
        canonical_backfill.materialize_native_xau_successor(
            client=client,
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc="2026-01-01T00:10:00Z",
            out_root=child,
            parent_root=parent,
            expected_parent_manifest_sha256=parent_descriptor["manifest_sha256"],
        )

    assert client.requests == []
    assert not child.exists()


def test_native_successor_rejects_overlap_rewrite_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )
    client = _FakeOandaClient(timeframe="M5", price_offset=10.0)
    child = tmp_path / "native_m5_child"

    with pytest.raises(RuntimeError, match="SUCCESSOR_OVERLAP_REWRITE"):
        canonical_backfill.materialize_native_xau_successor(
            client=client,
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc="2026-01-01T00:10:00Z",
            out_root=child,
            parent_root=parent,
            expected_parent_manifest_sha256=(
                parent_descriptor["manifest_sha256"]
            ),
        )

    assert len(client.requests) == 1
    assert not child.exists()
    assert not list(tmp_path.glob(".native_m5_child.staging.*"))


def test_native_successor_descriptor_revalidates_parent_manifest_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    child = tmp_path / "native_m5_child"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )
    canonical_backfill.materialize_native_xau_successor(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
        end_utc="2026-01-01T00:10:00Z",
        out_root=child,
        parent_root=parent,
        expected_parent_manifest_sha256=parent_descriptor["manifest_sha256"],
    )
    parent_manifest = parent / "MANIFEST.json"
    parent_manifest.write_bytes(parent_manifest.read_bytes() + b"\n")

    with pytest.raises(
        RuntimeError,
        match="SUCCESSOR_PARENT_MANIFEST_CAS_MISMATCH",
    ):
        canonical_xau_source_descriptor_v1(child, timeframe="M5")


def test_native_grandchild_revalidates_bootstrap_ancestor_manifest_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    child = tmp_path / "native_m5_child"
    grandchild = tmp_path / "native_m5_grandchild"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )
    canonical_backfill.materialize_native_xau_successor(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
        end_utc="2026-01-01T00:10:00Z",
        out_root=child,
        parent_root=parent,
        expected_parent_manifest_sha256=parent_descriptor["manifest_sha256"],
    )
    child_descriptor = canonical_xau_source_descriptor_v1(
        child,
        timeframe="M5",
    )
    canonical_backfill.materialize_native_xau_successor(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
        end_utc="2026-01-01T00:15:00Z",
        out_root=grandchild,
        parent_root=child,
        expected_parent_manifest_sha256=child_descriptor["manifest_sha256"],
    )
    monkeypatch.setattr(
        tape_contract,
        "_MAX_CANONICAL_NATIVE_ANCESTOR_DEPTH",
        1,
    )
    with pytest.raises(RuntimeError, match="SUCCESSOR_ANCESTOR_DEPTH_EXCEEDED"):
        canonical_xau_source_descriptor_v1(grandchild, timeframe="M5")
    monkeypatch.setattr(
        tape_contract,
        "_MAX_CANONICAL_NATIVE_ANCESTOR_DEPTH",
        1_024,
    )
    parent_manifest = parent / "MANIFEST.json"
    parent_manifest.write_bytes(parent_manifest.read_bytes() + b"\n")

    with pytest.raises(
        RuntimeError,
        match="SUCCESSOR_ANCESTOR_MANIFEST_CAS_MISMATCH",
    ):
        canonical_xau_source_descriptor_v1(grandchild, timeframe="M5")


def test_native_successor_can_extend_a_valid_successor_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    child = tmp_path / "native_m5_child"
    grandchild = tmp_path / "native_m5_grandchild"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )
    canonical_backfill.materialize_native_xau_successor(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
        end_utc="2026-01-01T00:10:00Z",
        out_root=child,
        parent_root=parent,
        expected_parent_manifest_sha256=parent_descriptor["manifest_sha256"],
    )
    child_descriptor = canonical_xau_source_descriptor_v1(
        child,
        timeframe="M5",
    )

    canonical_backfill.materialize_native_xau_successor(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
        end_utc="2026-01-01T00:15:00Z",
        out_root=grandchild,
        parent_root=child,
        expected_parent_manifest_sha256=child_descriptor["manifest_sha256"],
    )
    grandchild_descriptor = canonical_xau_source_descriptor_v1(
        grandchild,
        timeframe="M5",
    )

    assert grandchild_descriptor["parent_source"]["schema_version"] == (
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
    )
    assert grandchild_descriptor["parent_source"]["manifest_sha256"] == (
        child_descriptor["manifest_sha256"]
    )
    assert grandchild_descriptor["row_count"] == (
        child_descriptor["row_count"] + 1
    )


def test_native_successor_parent_change_before_publish_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    child = tmp_path / "native_m5_child"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )

    def mutate_parent(**_kwargs: object) -> None:
        manifest_path = parent / "MANIFEST.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    monkeypatch.setattr(
        canonical_backfill,
        "_verify_producer_sources_unchanged",
        mutate_parent,
    )
    with pytest.raises(
        RuntimeError,
        match="SUCCESSOR_PARENT_CHANGED_BEFORE_PUBLISH",
    ):
        canonical_backfill.materialize_native_xau_successor(
            client=_FakeOandaClient(timeframe="M5"),
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc="2026-01-01T00:10:00Z",
            out_root=child,
            parent_root=parent,
            expected_parent_manifest_sha256=(
                parent_descriptor["manifest_sha256"]
            ),
        )

    assert not child.exists()
    assert not list(tmp_path.glob(".native_m5_child.staging.*"))


def test_native_successor_incomplete_refetch_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    parent = tmp_path / "native_m5_parent"
    child = tmp_path / "native_m5_child"
    materialize_native_xau_test_bundle(parent, timeframe="M5")
    parent_descriptor = canonical_xau_source_descriptor_v1(
        parent,
        timeframe="M5",
    )

    with pytest.raises(RuntimeError, match="INCOMPLETE_CANDLE_FORBIDDEN"):
        canonical_backfill.materialize_native_xau_successor(
            client=_FakeOandaClient(
                timeframe="M5",
                incomplete_only=True,
            ),
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc="2026-01-01T00:10:00Z",
            out_root=child,
            parent_root=parent,
            expected_parent_manifest_sha256=(
                parent_descriptor["manifest_sha256"]
            ),
        )

    assert not child.exists()
    assert not list(tmp_path.glob(".native_m5_child.staging.*"))


def test_native_request_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"

    with pytest.raises(RuntimeError, match="network unavailable"):
        canonical_backfill.materialize_native_xau_snapshot(
            client=_FakeOandaClient(fail=True),
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_TEST_002",
            start_utc="2024-12-31T23:55:00Z",
            end_utc="2025-01-01T00:10:00Z",
            out_root=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".native_m5.staging.*"))


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (_FakeOandaClient(incomplete_only=True), "INCOMPLETE_CANDLE_FORBIDDEN"),
        (_FakeOandaClient(bad_geometry=True), "OHLC_GEOMETRY_INVALID"),
    ],
)
def test_native_rejects_unproven_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeOandaClient,
    error: str,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / f"native_m5_{error.lower()}"

    with pytest.raises(RuntimeError, match=error):
        canonical_backfill.materialize_native_xau_snapshot(
            client=client,
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_TEST_003",
            start_utc="2024-12-31T23:55:00Z",
            end_utc="2025-01-01T00:10:00Z",
            out_root=output,
        )

    assert not output.exists()


def test_native_mixed_complete_response_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5_mixed_completion"

    with pytest.raises(RuntimeError, match="INCOMPLETE_CANDLE_FORBIDDEN"):
        canonical_backfill.materialize_native_xau_snapshot(
            client=_FakeOandaClient(
                timeframe="M5",
                incomplete_position=1,
            ),
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_MIXED_COMPLETION",
            start_utc="2025-01-01T00:00:00Z",
            end_utc="2025-01-01T00:15:00Z",
            out_root=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".native_m5_mixed_completion.staging.*"))


def test_native_validator_requires_zero_incomplete_candles(
    tmp_path: Path,
) -> None:
    output = tmp_path / "native_m5"
    materialize_native_xau_test_bundle(output, timeframe="M5")
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_chunks"][0]["incomplete_candles"] = 1
    manifest["source_chunks_sha256"] = canonical_json_sha256(
        manifest["source_chunks"]
    )
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_json_sha256(
        manifest_without_hash
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="INCOMPLETE_CANDLES_FORBIDDEN"):
        canonical_xau_source_descriptor_v1(output, timeframe="M5")


@pytest.mark.parametrize(
    ("timeframe", "start_utc", "error"),
    [
        ("M1", "2024-12-31T23:55:30Z", "START_UTC_NOT_M1_ALIGNED"),
        ("M5", "2024-12-31T23:56:00Z", "START_UTC_NOT_M5_ALIGNED"),
    ],
)
def test_native_rejects_misaligned_interval_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeframe: str,
    start_utc: str,
    error: str,
) -> None:
    _allow_clean_repo(monkeypatch)
    client = _FakeOandaClient(timeframe=timeframe)

    with pytest.raises(RuntimeError, match=error):
        canonical_backfill.materialize_native_xau_snapshot(
            client=client,
            timeframe=timeframe,
            vedtak_id=f"XAU_NATIVE_{timeframe}_ALIGNMENT_TEST",
            start_utc=start_utc,
            end_utc="2025-01-01T00:10:00Z",
            out_root=tmp_path / f"native_{timeframe.lower()}",
        )

    assert client.requests == []


def test_native_rejects_response_timeframe_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m1"

    with pytest.raises(
        RuntimeError,
        match="XAU_CANONICAL_M1_SOURCE_RESPONSE_GRANULARITY_MISMATCH",
    ):
        canonical_backfill.materialize_native_xau_snapshot(
            client=_FakeOandaClient(
                timeframe="M1",
                response_timeframe="M5",
            ),
            timeframe="M1",
            vedtak_id="XAU_NATIVE_M1_RESPONSE_TF_TEST",
            start_utc="2025-01-01T00:00:00Z",
            end_utc="2025-01-01T00:01:00Z",
            out_root=output,
        )

    assert not output.exists()


def test_native_parser_keeps_timeframe_identity_in_component_error() -> None:
    response = _FakeOandaClient(timeframe="M1")._request(
        "GET",
        "/instruments/XAU_USD/candles",
        params={
            "from": "2025-01-01T00:00:00.000000000Z",
            "to": "2025-01-01T00:01:00.000000000Z",
            "granularity": "M1",
            "price": "MBA",
        },
    )
    del response["candles"][0]["bid"]

    with pytest.raises(
        RuntimeError,
        match="XAU_CANONICAL_M1_SOURCE_PRICE_COMPONENT_MISSING",
    ):
        canonical_native_frame_from_oanda_response(
            response,
            timeframe="M1",
            request_start="2025-01-01T00:00:00Z",
            request_end="2025-01-01T00:01:00Z",
        )


def test_native_rejects_existing_output_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"
    output.mkdir()
    client = _FakeOandaClient()

    with pytest.raises(RuntimeError, match="IMMUTABLE_OUTPUT_EXISTS"):
        canonical_backfill.materialize_native_xau_snapshot(
            client=client,
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_TEST_004",
            start_utc="2024-12-31T23:55:00Z",
            end_utc="2025-01-01T00:10:00Z",
            out_root=output,
        )

    assert client.requests == []


def test_native_source_tamper_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"
    manifest = canonical_backfill.materialize_native_xau_snapshot(
        client=_FakeOandaClient(),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_TEST_005",
        start_utc="2024-12-31T23:55:00Z",
        end_utc="2025-01-01T00:10:00Z",
        out_root=output,
    )
    chunk = output / manifest["source_chunks"][0]["relative_path"]
    chunk.write_bytes(chunk.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="SOURCE_CHUNK_BINDING_MISMATCH"):
        canonical_xau_source_descriptor_v1(output, timeframe="M5")


def test_native_descriptor_rejects_cross_timeframe_admission(
    tmp_path: Path,
) -> None:
    output = tmp_path / "native_m1"
    materialize_native_xau_test_bundle(output, timeframe="M1")

    with pytest.raises(RuntimeError, match="XAU_CANONICAL_TIMEFRAME_MISMATCH"):
        canonical_xau_source_descriptor_v1(output, timeframe="M5")


def test_native_removed_legacy_modes_are_not_parseable() -> None:
    parser = canonical_backfill.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--vedtak",
                "XAU_NATIVE_M5_TEST_006",
                "--timeframe",
                "M5",
                "--repair-mode",
            ]
        )

    assert exc_info.value.code == 2


def test_native_cli_requires_explicit_publication_mode() -> None:
    parser = canonical_backfill.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--vedtak",
                "XAU_NATIVE_M5_EXPLICIT_MODE",
                "--timeframe",
                "M5",
                "--start-utc",
                "2026-01-01T00:00:00Z",
                "--end-utc",
                "2026-01-01T00:10:00Z",
                "--out-root",
                "/tmp/native-m5-explicit-mode",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--publication-mode", "successor"],
        [
            "--publication-mode",
            "successor",
            "--parent-root",
            "/tmp/parent",
            "--expected-parent-manifest-sha256",
            "invalid",
        ],
        [
            "--publication-mode",
            "bootstrap",
            "--parent-root",
            "/tmp/parent",
            "--expected-parent-manifest-sha256",
            "f" * 64,
        ],
    ],
)
def test_native_cli_rejects_incomplete_or_mixed_successor_contract_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(canonical_backfill.__file__),
            "--vedtak",
            "XAU_NATIVE_CLI_SUCCESSOR_TEST",
            "--timeframe",
            "M5",
            "--start-utc",
            "2026-01-01T00:00:00Z",
            "--end-utc",
            "2026-01-01T00:10:00Z",
            "--out-root",
            "/tmp/native-child",
            *extra,
        ],
    )
    monkeypatch.setattr(
        canonical_backfill,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loaded before CLI contract failed"),
    )

    with pytest.raises(SystemExit) as exc_info:
        canonical_backfill.main()

    assert exc_info.value.code == 2


def test_native_vectorized_row_encoding_is_byte_identical() -> None:
    client = _FakeOandaClient()
    response = client._request(
        "GET",
        "/instruments/XAU_USD/candles",
        params={
            "from": "2026-01-01T00:00:00.000000000Z",
            "to": "2026-01-01T00:10:00.000000000Z",
            "granularity": "M5",
            "price": "MBA",
        },
    )
    frame, _ = canonical_native_frame_from_oanda_response(
        response,
        timeframe="M5",
        request_start="2026-01-01T00:00:00Z",
        request_end="2026-01-01T00:10:00Z",
    )
    reference = b"".join(
        struct.pack(
            ">q12dq",
            int(pd.Timestamp(row[0]).value),
            *(float(value) for value in row[1:-1]),
            int(row[-1]),
        )
        for row in frame.itertuples(index=False, name=None)
    )

    assert (
        canonical_native_rows_bytes(frame, timeframe="M5")
        == reference
    )


def test_native_chunk_policy_is_exact_4320_slot_window() -> None:
    assert NATIVE_TIMEFRAME_POLICY == {
        "M1": {
            "bar_seconds": 60,
            "request_chunk_days": 3,
            "max_theoretical_slots": 4_320,
        },
        "M5": {
            "bar_seconds": 300,
            "request_chunk_days": 15,
            "max_theoretical_slots": 4_320,
        },
    }


def test_native_m1_streams_exact_policy_chunks_into_year_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m1_streamed"
    client = _FakeOandaClient(timeframe="M1")

    manifest = canonical_backfill.materialize_native_xau_snapshot(
        client=client,
        timeframe="M1",
        vedtak_id="XAU_NATIVE_M1_TEST_008",
        start_utc="2024-12-30T23:59:00Z",
        end_utc="2025-01-03T00:00:00Z",
        out_root=output,
    )

    assert len(client.requests) == 2
    assert client.requests[0]["params"]["from"] == "2024-12-30T23:59:00.000000000Z"
    assert client.requests[0]["params"]["to"] == "2025-01-02T23:59:00.000000000Z"
    assert client.requests[1]["params"]["to"] == "2025-01-03T00:00:00.000000000Z"
    assert manifest["row_count"] == 4_321
    assert manifest["year_rows"] == {"year=2024": 1_441, "year=2025": 2_880}
    descriptor = canonical_xau_source_descriptor_v1(output, timeframe="M1")
    assert descriptor["canonical_rows_sha256"] == manifest["canonical_rows_sha256"]


def test_native_root_is_accepted_as_current_tape_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A strict native-v3 M5 root is complete tape provenance for the seq513
    # source cascade: no repair lineage, no collector snapshot.
    from gx1.contracts.xau_tape_provenance_v1 import (
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        validate_xau_tape_provenance_v1,
    )

    _allow_clean_repo(monkeypatch)
    output = tmp_path / "m5_tape_native_v3"
    manifest = canonical_backfill.materialize_native_xau_snapshot(
        client=_FakeOandaClient(timeframe="M5"),
        timeframe="M5",
        vedtak_id="XAU_NATIVE_M5_TEST_002",
        start_utc="2024-12-31T23:55:00Z",
        end_utc="2025-01-01T00:10:00Z",
        out_root=output,
    )

    provenance = validate_xau_tape_provenance_v1(
        output,
        expected_run_id="XAU_SEQ513_TEST_RUN_01",
        require_current=True,
    )

    assert provenance["schema_version"] == CANONICAL_NATIVE_SOURCE_SCHEMA
    assert provenance["tape_root"] == str(output)
    assert provenance["explicit_vedtak_id"] == "XAU_NATIVE_M5_TEST_002"
    assert provenance["year_sha256"] == manifest["year_sha256"]
    assert provenance["time_max_utc"] == manifest["time_max_utc"]

    empty = tmp_path / "no_manifests"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="XAU_TAPE_MANIFEST"):
        validate_xau_tape_provenance_v1(
            empty,
            expected_run_id="XAU_SEQ513_TEST_RUN_01",
            require_current=True,
        )
