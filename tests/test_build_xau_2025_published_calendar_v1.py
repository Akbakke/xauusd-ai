from __future__ import annotations

import pytest

from gx1.audit import build_xau_2025_published_calendar_v1 as calendar
from gx1.contracts import unified_exit_market_closure_authority_v1 as owner


def _holiday_lines() -> list[str]:
    lines = ["Note that all times are in Eastern Time."]
    for day, action in (
        ("31 December", ["Metals:", "close at 16:59"]),
        ("25 December", ["Metals:", "open at 18:01"]),
        ("24 December", ["Metals:", "close at 13:44"]),
        ("28 November", ["US energies and metals: Close early at 2:45 PM"]),
        ("27 November", ["US energies and metals: Close early at 2:30 PM, reopen at 6:00 PM"]),
        ("1 September", ["US energies and metals: Close early at 2:30 PM, reopen at 6:00 PM"]),
        ("4 July", ["US energies and metals close early at 1 pm"]),
        ("19 June", ["US energies and metals close early at 2:30 PM and reopen at 6:00 PM"]),
        ("26 May", ["US energies and metals close early at 2:30 PM and reopen at 6:00 PM"]),
        ("18 April", ["US energies and metals are closed"]),
        ("17 April", ["US energies and metals close early at 17:00 ET"]),
    ):
        lines.extend([f"{day} 2025", *action])
    return lines


def test_source_dates_veto_all_displayed_metals_exceptions() -> None:
    lines = _holiday_lines()
    lines.extend(["1 January 2026", "Metals:", "open at 18:01"])
    calendar._check_holiday(lines, "20260208124247")
    lines[1:1] = ["20 August 2025", "Metals: close at 10:00"]
    with pytest.raises(RuntimeError, match="HOLIDAY_DATE_SET"):
        calendar._check_holiday(lines, "20260208124247")


def test_extension_preserves_prefix_and_excludes_holiday_periods() -> None:
    prefix = {"kind": "daily_maintenance", "start_utc": "2025-04-01T20:59:00+00:00",
              "end_utc_exclusive": "2025-04-01T22:05:00+00:00", "source_event_id": "old"}
    old = {"coverage_start_utc": "2024-04-29T00:00:00+00:00",
           "intervals": [prefix], "holiday_veto_dates": [],
           "excluded_published_regular_intervals_due_to_holiday": [],
           "source_documents": [], "calendar_sha256": "old"}
    plan = {"timezone_database": {"path": "/usr/share/zoneinfo/America/New_York"},
            "coverage_end_utc_exclusive": "2026-01-01T00:00:00+00:00"}
    records = [{"role": "regular", "sha256": "a" * 64}]
    result = calendar._build_calendar(old, plan, records, owner)
    assert result["intervals"][0] == prefix
    assert result["holiday_veto_dates"] == list(calendar.METAL_DATES)
    assert {day for item in result["excluded_published_regular_intervals_due_to_holiday"]
            for day in item["veto_dates"]} == set(calendar.METAL_DATES)
    excluded_starts = {item["start_utc"] for item in result["excluded_published_regular_intervals_due_to_holiday"]}
    interval_starts = {item["start_utc"] for item in result["intervals"]}
    assert excluded_starts.isdisjoint(interval_starts)
    assert not any(item["start_utc"][:10] in {"2025-04-17", "2025-04-18"}
                   for item in result["intervals"])
    dst_weekend = next(item for item in result["intervals"]
                       if item["start_utc"] == "2025-10-31T20:59:00+00:00")
    assert dst_weekend["kind"] == "weekend"
    assert dst_weekend["end_utc_exclusive"] == "2025-11-02T23:05:00+00:00"
    assert result["coverage_end_utc_exclusive"] == "2026-01-01T00:00:00+00:00"
    assert not any(item["start_utc"].startswith("2025-12-31") for item in result["intervals"])
    assert result["calendar_sha256"] == owner.canonical_sha256(
        {key: value for key, value in result.items() if key != "calendar_sha256"})
