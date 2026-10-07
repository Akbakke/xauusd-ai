from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
    UNKNOWN_GAPS_ONLY_SOURCE_METHOD,
    build_market_closure_authority,
    build_unknown_gap_only_schedule,
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


def _unknown_authority(clock: pd.DatetimeIndex, schedule: dict | None = None) -> dict:
    return build_market_closure_authority(
        m1_times=clock,
        m1_source_path=Path('/immutable/pretest.parquet'),
        m1_source_sha256='2' * 64,
        m1_source_manifest_path=Path('/immutable/pretest.manifest.json'),
        m1_source_manifest_sha256='3' * 64,
        exact_schedule=build_unknown_gap_only_schedule(clock) if schedule is None else schedule,
        exact_schedule_path=Path('/immutable/unknown-only.schedule.json'),
        exact_schedule_file_sha256='4' * 64,
    )


def test_observed_clock_policy_never_infers_weekend_or_other_closures() -> None:
    clock = _clock()
    schedule = build_unknown_gap_only_schedule(clock)
    assert schedule['source_method'] == UNKNOWN_GAPS_ONLY_SOURCE_METHOD
    assert schedule['source_reference_sha256'] == m1_clock_sha256(clock)
    assert schedule['intervals'] == []
    authority = _unknown_authority(clock)
    assert authority['observed_gap_count'] == authority['unknown_source_gap_count'] == 2
    assert authority['known_market_closure_count'] == 0
    assert authority['unknown_gap_semantics'] == 'right_censor_before_gap'
    assert all(item['classification'] == 'unknown_source_absence'
               and item['successor_across_gap_allowed'] is False
               for item in authority['intervals'])


def test_unknown_only_policy_rejects_any_declared_closure() -> None:
    clock = _clock()
    schedule = build_unknown_gap_only_schedule(clock)
    schedule.pop('schedule_sha256')
    schedule['intervals'] = _schedule(clock)['intervals']
    with pytest.raises(RuntimeError, match='UNKNOWN_GAP_POLICY_INVALID'):
        seal_exact_market_schedule(schedule)


@pytest.mark.parametrize('change', ['clock-reference', 'coverage-start', 'coverage-end'])
def test_unknown_only_policy_is_bound_to_the_entire_actual_clock(change) -> None:
    clock = _clock()
    schedule = build_unknown_gap_only_schedule(clock)
    schedule.pop('schedule_sha256')
    if change == 'clock-reference':
        schedule['source_reference_sha256'] = '1' * 64
    elif change == 'coverage-start':
        schedule['coverage_start_utc'] = (clock[0] - pd.Timedelta(minutes=1)).isoformat()
    else:
        schedule['coverage_end_utc_exclusive'] = (clock[-1] + pd.Timedelta(minutes=2)).isoformat()
    with pytest.raises(RuntimeError, match='UNKNOWN_GAP_CLOCK_BINDING_INVALID'):
        _unknown_authority(clock, seal_exact_market_schedule(schedule))


def test_unknown_only_continuous_clock_does_not_invent_a_gap() -> None:
    authority = _unknown_authority(pd.date_range('2026-06-01T00:00Z', periods=5, freq='min'))
    assert authority['observed_gap_count'] == authority['unknown_source_gap_count'] == 0
    assert authority['known_market_closure_count'] == 0
    assert authority['intervals'] == []


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
