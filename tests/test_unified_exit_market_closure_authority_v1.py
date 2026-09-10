from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
    build_market_closure_authority,
    closure_intervals_by_gap_after_row,
    m1_clock_sha256,
    require_market_closure_authority,
    seal_exact_market_schedule,
)


def _clock() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        list(pd.date_range("2026-07-03T21:55Z", periods=5, freq="min"))
        + list(pd.date_range("2026-07-05T22:05Z", periods=5, freq="min"))
        + list(pd.date_range("2026-07-06T10:00Z", periods=5, freq="min"))
    )


def _schedule(clock: pd.DatetimeIndex, *, extra: bool = False) -> dict:
    intervals = [
        {
            "kind": "weekend",
            "start_utc": (clock[4] + pd.Timedelta(minutes=1)).isoformat(),
            "end_utc_exclusive": clock[5].isoformat(),
            "source_event_id": "xau-weekend-2026-07-03",
        }
    ]
    if extra:
        intervals.append(
            {
                "kind": "holiday",
                "start_utc": "2026-07-06T12:00:00+00:00",
                "end_utc_exclusive": "2026-07-06T13:00:00+00:00",
                "source_event_id": "unobserved-holiday",
            }
        )
    return seal_exact_market_schedule(
        {
            "schema_version": MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
            "decision": "PASS",
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "coverage_start_utc": clock[0].isoformat(),
            "coverage_end_utc_exclusive": "2026-07-07T00:00:00+00:00",
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": (
                "externally_sourced_exact_xau_utc_closure_intervals_v1"
            ),
            "source_reference_sha256": "1" * 64,
            "intervals": intervals,
            "test_data_used": False,
        }
    )


def _authority(clock: pd.DatetimeIndex) -> dict:
    return build_market_closure_authority(
        m1_times=clock,
        m1_source_path=Path("/immutable/pretest.parquet"),
        m1_source_sha256="2" * 64,
        m1_source_manifest_path=Path("/immutable/pretest.manifest.json"),
        m1_source_manifest_sha256="3" * 64,
        exact_schedule=_schedule(clock),
        exact_schedule_path=Path("/immutable/xau.schedule.json"),
        exact_schedule_file_sha256="4" * 64,
    )


def test_exact_schedule_allows_weekend_but_not_unknown_gap() -> None:
    clock = _clock()
    authority = _authority(clock)
    gaps = closure_intervals_by_gap_after_row(authority)
    assert gaps[4]["classification"] == "declared_weekend_market_closure"
    assert gaps[4]["successor_across_gap_allowed"] is True
    assert gaps[9]["classification"] == "unknown_source_absence"
    assert gaps[9]["successor_across_gap_allowed"] is False
    assert authority["known_market_closure_count"] == 1
    assert authority["unknown_source_gap_count"] == 1
    require_market_closure_authority(
        authority,
        expected_m1_source_sha256="2" * 64,
        expected_m1_clock_sha256=m1_clock_sha256(clock),
    )


def test_unobserved_schedule_interval_and_tamper_fail_closed() -> None:
    clock = _clock()
    with pytest.raises(RuntimeError, match="INTERVAL_UNOBSERVED"):
        build_market_closure_authority(
            m1_times=clock,
            m1_source_path=Path("/immutable/pretest.parquet"),
            m1_source_sha256="2" * 64,
            m1_source_manifest_path=Path("/immutable/pretest.manifest.json"),
            m1_source_manifest_sha256="3" * 64,
            exact_schedule=_schedule(clock, extra=True),
            exact_schedule_path=Path("/immutable/xau.schedule.json"),
            exact_schedule_file_sha256="4" * 64,
        )
    authority = _authority(clock)
    authority["intervals"][0]["successor_across_gap_allowed"] = False
    with pytest.raises(RuntimeError, match="INTERVAL_INVALID"):
        require_market_closure_authority(
            authority,
            expected_m1_source_sha256="2" * 64,
            expected_m1_clock_sha256=m1_clock_sha256(clock),
        )
