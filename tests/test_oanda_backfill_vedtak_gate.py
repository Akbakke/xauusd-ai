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

from gx1.contracts.xau_tape_provenance_v1 import (
    canonical_m5_rows_bytes,
    canonical_xau_source_descriptor_v1,
)
from gx1.scripts import backfill_xauusd_m5_bidask_2020_2025 as bounded_backfill
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
        bad_geometry: bool = False,
    ) -> None:
        self.fail = fail
        self.incomplete_only = incomplete_only
        self.bad_geometry = bad_geometry
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
        timestamps = pd.date_range(
            start,
            end - pd.Timedelta(minutes=5),
            freq="5min",
        )
        candles = []
        for position, timestamp in enumerate(timestamps):
            mid_high = "1998.0" if self.bad_geometry and position == 0 else "2002.0"
            candles.append(
                {
                    "complete": not self.incomplete_only,
                    "time": timestamp.isoformat().replace("+00:00", "Z"),
                    "volume": 10 + position,
                    "mid": {
                        "o": "2000.0",
                        "h": mid_high,
                        "l": "1999.0",
                        "c": "2001.0",
                    },
                    "bid": {
                        "o": "1999.5",
                        "h": "2001.5",
                        "l": "1998.5",
                        "c": "2000.5",
                    },
                    "ask": {
                        "o": "2000.5",
                        "h": "2002.5",
                        "l": "1999.5",
                        "c": "2001.5",
                    },
                }
            )
        return {
            "instrument": "XAU_USD",
            "granularity": "M5",
            "candles": candles,
        }


def _allow_clean_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        canonical_backfill,
        "_require_clean_repository",
        lambda _root: "a" * 40,
    )


def materialize_native_m5_test_bundle(output: Path) -> dict[str, Any]:
    """Build a real v2 native-M5 bundle for downstream provenance fixtures."""

    with mock.patch.object(
        canonical_backfill,
        "_require_clean_repository",
        return_value="a" * 40,
    ):
        return canonical_backfill.materialize_native_m5_snapshot(
            client=_FakeOandaClient(),
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V2",
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T00:05:00Z",
            out_root=output,
        )


@pytest.mark.parametrize("module", [bounded_backfill, canonical_backfill])
def test_backfill_cli_requires_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("module", [bounded_backfill, canonical_backfill])
def test_backfill_cli_rejects_invalid_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__), "--vedtak", "short"])
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(GateError, match="blocked"):
        module.main()


def test_m1_downsample_cannot_write_canonical_m5_root() -> None:
    with pytest.raises(RuntimeError, match="CANONICAL_M5_SINGLE_OWNER_VIOLATION"):
        m1_downsample.main()


def test_bounded_backfill_cannot_target_canonical_m5_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "canonical_m5"
    monkeypatch.setattr(
        bounded_backfill,
        "PROTECTED_CANONICAL_M5_ROOT",
        protected,
    )

    with pytest.raises(RuntimeError, match="CANONICAL_M5_SINGLE_OWNER_VIOLATION"):
        bounded_backfill._reject_canonical_root_output(
            protected / "year=2026" / "part-000.parquet"
        )


def test_native_m5_materialization_is_source_bound_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"
    client = _FakeOandaClient()

    manifest = canonical_backfill.materialize_native_m5_snapshot(
        client=client,
        vedtak_id="XAU_NATIVE_M5_TEST_001",
        start_utc="2024-12-31T23:55:00Z",
        end_utc="2025-01-01T00:10:00Z",
        out_root=output,
        chunk_days=1,
    )

    assert output.is_dir()
    assert manifest["row_count"] == 3
    assert manifest["year_rows"] == {"year=2024": 1, "year=2025": 2}
    assert list(output.glob(".*.staging.*")) == []
    assert [request["params"]["price"] for request in client.requests] == ["MBA"]
    descriptor = canonical_xau_source_descriptor_v1(output, timeframe="M5")
    assert descriptor["row_count"] == 3
    assert descriptor["canonical_rows_sha256"] == manifest["canonical_rows_sha256"]

    chunk_path = output / manifest["source_chunks"][0]["relative_path"]
    payload = json.loads(gzip.decompress(chunk_path.read_bytes()).decode("utf-8"))
    assert payload["request"]["instrument"] == "XAU_USD"
    assert payload["response"]["candles"][0]["complete"] is True


def test_native_m5_request_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"

    with pytest.raises(RuntimeError, match="network unavailable"):
        canonical_backfill.materialize_native_m5_snapshot(
            client=_FakeOandaClient(fail=True),
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
        (_FakeOandaClient(incomplete_only=True), "COMPLETE_SOURCE_EMPTY"),
        (_FakeOandaClient(bad_geometry=True), "OHLC_GEOMETRY_INVALID"),
    ],
)
def test_native_m5_rejects_unproven_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeOandaClient,
    error: str,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / f"native_m5_{error.lower()}"

    with pytest.raises(RuntimeError, match=error):
        canonical_backfill.materialize_native_m5_snapshot(
            client=client,
            vedtak_id="XAU_NATIVE_M5_TEST_003",
            start_utc="2024-12-31T23:55:00Z",
            end_utc="2025-01-01T00:10:00Z",
            out_root=output,
        )

    assert not output.exists()


def test_native_m5_rejects_existing_output_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"
    output.mkdir()
    client = _FakeOandaClient()

    with pytest.raises(RuntimeError, match="IMMUTABLE_OUTPUT_EXISTS"):
        canonical_backfill.materialize_native_m5_snapshot(
            client=client,
            vedtak_id="XAU_NATIVE_M5_TEST_004",
            start_utc="2024-12-31T23:55:00Z",
            end_utc="2025-01-01T00:10:00Z",
            out_root=output,
        )

    assert client.requests == []


def test_native_m5_source_tamper_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5"
    manifest = canonical_backfill.materialize_native_m5_snapshot(
        client=_FakeOandaClient(),
        vedtak_id="XAU_NATIVE_M5_TEST_005",
        start_utc="2024-12-31T23:55:00Z",
        end_utc="2025-01-01T00:10:00Z",
        out_root=output,
    )
    chunk = output / manifest["source_chunks"][0]["relative_path"]
    chunk.write_bytes(chunk.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="SOURCE_CHUNK_BINDING_MISMATCH"):
        canonical_xau_source_descriptor_v1(output, timeframe="M5")


def test_native_m5_removed_legacy_modes_are_not_parseable() -> None:
    parser = canonical_backfill.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--vedtak",
                "XAU_NATIVE_M5_TEST_006",
                "--repair-mode",
            ]
        )

    assert exc_info.value.code == 2


def test_native_m5_vectorized_row_encoding_is_byte_identical() -> None:
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
    frame, _ = canonical_backfill.canonical_m5_frame_from_oanda_response(
        response,
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

    assert canonical_m5_rows_bytes(frame) == reference


def test_native_m5_default_chunk_is_max_safe_oanda_window() -> None:
    args = canonical_backfill.build_parser().parse_args(
        ["--vedtak", "XAU_NATIVE_M5_TEST_007"]
    )

    assert args.chunk_days == 15


def test_native_m5_streams_multiple_chunks_into_year_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_clean_repo(monkeypatch)
    output = tmp_path / "native_m5_streamed"
    client = _FakeOandaClient()

    manifest = canonical_backfill.materialize_native_m5_snapshot(
        client=client,
        vedtak_id="XAU_NATIVE_M5_TEST_008",
        start_utc="2024-12-31T00:00:00Z",
        end_utc="2025-01-02T00:00:00Z",
        out_root=output,
        chunk_days=1,
    )

    assert len(client.requests) == 2
    assert manifest["row_count"] == 576
    assert manifest["year_rows"] == {"year=2024": 288, "year=2025": 288}
    descriptor = canonical_xau_source_descriptor_v1(output, timeframe="M5")
    assert descriptor["canonical_rows_sha256"] == manifest["canonical_rows_sha256"]
