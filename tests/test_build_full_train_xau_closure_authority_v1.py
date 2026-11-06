from __future__ import annotations

from pathlib import Path

import pandas as pd

from gx1.audit import build_full_train_xau_closure_authority_v1 as worker
from gx1.contracts import unified_exit_market_closure_authority_v1 as owner


def test_only_fully_contained_observed_gap_is_exact() -> None:
    clock = pd.DatetimeIndex([
        "2025-05-02T20:59:00Z", "2025-05-04T22:04:00Z",
        "2025-05-04T22:05:00Z", "2025-05-06T10:00:00Z",
        "2025-05-06T10:03:00Z", "2025-05-07T20:57:00Z",
        "2025-05-07T22:05:00Z",
    ])
    intervals = [
        {"kind": "weekend", "start_utc": "2025-05-02T20:59:00+00:00",
         "end_utc_exclusive": "2025-05-04T22:05:00+00:00", "source_event_id": "regular-weekend"},
        {"kind": "daily_maintenance", "start_utc": "2025-05-07T20:59:00+00:00",
         "end_utc_exclusive": "2025-05-07T22:05:00+00:00", "source_event_id": "regular-daily"},
    ]
    proofs, exact, gaps = worker._certify_gaps(
        clock, {"calendar_sha256": "a" * 64}, intervals, 1234, owner)
    assert gaps == 5
    assert len(proofs) == len(exact) == 1
    assert proofs[0]["gap_after_physical_source_row"] == 1234
    assert proofs[0]["observed_gap"] == {
        "start_utc": "2025-05-02T21:00:00+00:00",
        "end_utc_exclusive": "2025-05-04T22:04:00+00:00",
    }
    schedule = owner.seal_exact_market_schedule({
        "schema_version": owner.MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
        "decision": "PASS", "instrument": "XAU_USD", "timeframe": "M1",
        "coverage_start_utc": clock[0].isoformat(),
        "coverage_end_utc_exclusive": (clock[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "interval_semantics": "left_closed_right_open_utc",
        "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1",
        "source_reference_sha256": "b" * 64, "intervals": exact, "test_data_used": False,
    })
    authority = owner.build_market_closure_authority(
        m1_times=clock, m1_source_path=Path("/source/train_m1.parquet"),
        m1_source_sha256="c" * 64,
        m1_source_manifest_path=Path("/source/manifest.json"),
        m1_source_manifest_sha256="d" * 64,
        exact_schedule=schedule, exact_schedule_path=Path("/source/schedule.json"),
        exact_schedule_file_sha256="e" * 64,
    )
    assert authority["known_market_closure_count"] == 1
    assert authority["unknown_source_gap_count"] == 4
    assert [item["successor_across_gap_allowed"] for item in authority["intervals"]] == [
        True, False, False, False, False]
