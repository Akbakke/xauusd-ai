"""Synthetic mechanics only; no GC source-quality or trading evidence."""
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts.gc_source_disposition_v1 import (
    F_BAD_TS_RECV,
    HOUR_NS,
    MINUTE_NS,
    NANOSECONDS,
    STUDY_START_NS,
    WINDOW_END_NS,
    audit_symbol_group,
    causal_bar_index,
    merge_presence_bitmaps,
    regular_globex_template_open,
    summarize_session_bins,
)


def ns(year, month, day, hour=0, minute=0, second=0):
    return int(datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()) * NANOSECONDS


HEADER = [
    "ts_recv", "symbol", "instrument_id", "publisher_id", "action", "side",
    "size", "flags", "ts_event", "price", "bid_px_00", "ask_px_00",
]


def write_source(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)


def declaration(path, symbol="GCZ5", instrument=42, start=None, end=None):
    start = start if start is not None else ns(2025, 1, 6, 0)
    end = end if end is not None else start + 3 * HOUR_NS
    return {
        "raw_symbol": symbol,
        "instrument_id": instrument,
        "publisher_id": 1,
        "start_ns": start,
        "end_ns": end,
        "path": str(path),
        "sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "compression": "gzip",
        "max_records": 100,
    }


def row(ts, symbol="GCZ5", instrument=42, action="T", side="B", size=1, flags=0):
    # Forbidden fields intentionally contain malformed text. The audit may not use them.
    return [str(ts), symbol, str(instrument), "1", action, side, str(size),
            str(flags), "not-a-timestamp", "not-a-price", "bad", "bad"]


def test_causal_match_uses_sixty_seconds_and_strict_test_end():
    start = ns(2025, 6, 1, 12)
    lookup = {start: 0}
    assert causal_bar_index(start + MINUTE_NS - 1, lookup) == 0
    assert causal_bar_index(start + MINUTE_NS, lookup) is None

    eligible_start = WINDOW_END_NS - 2 * MINUTE_NS
    last_start = WINDOW_END_NS - MINUTE_NS
    lookup = {eligible_start: 4, last_start: 5}
    assert causal_bar_index(eligible_start + MINUTE_NS - 1, lookup) == 4
    assert causal_bar_index(last_start, lookup) is None
    assert causal_bar_index(WINDOW_END_NS, lookup) is None


def test_bad_receive_timestamp_is_counted_and_excluded_from_causal_matches(tmp_path):
    start = ns(2025, 1, 6, 0)
    path = tmp_path / "gc.csv.gz"
    write_source(path, [
        row(start + 1, side="B", size=5),
        row(start + 2, side="A", size=4, flags=F_BAD_TS_RECV),
    ])
    report = audit_symbol_group(
        [declaration(path, start=start, end=start + HOUR_NS)],
        {start: 0},
    )
    assert report["counters"]["bad_receive_timestamp_records"] == 1
    assert report["counters"]["trade_records"] == 2
    assert report["counters"]["matched_trade_records_support"] == 1
    assert report["matched_support_trade_counts_by_side"] == {"B": 1, "A": 0, "N": 0}
    assert sum(report["_trade_presence"]) == 1


def test_unknown_aggressor_volume_stays_separate(tmp_path):
    start = ns(2025, 1, 6, 0)
    path = tmp_path / "gc.csv.gz"
    write_source(path, [
        row(start + 1, side="B", size=5),
        row(start + 2, side="A", size=3),
        row(start + 3, side="N", size=2),
    ])
    report = audit_symbol_group(
        [declaration(path, start=start, end=start + HOUR_NS)],
        {start: 0},
    )
    assert report["trade_volume_contracts_by_side"] == {"B": 5, "A": 3, "N": 2}
    assert report["known_aggressor_delta_contracts"] == 2
    assert report["unknown_aggressor_trade_volume_share"] == 0.2
    assert report["matched_support_trade_volume_contracts_by_side"] == {"B": 5, "A": 3, "N": 2}


def test_contract_metrics_remain_separate_while_union_counts_unique_bars(tmp_path):
    start = ns(2025, 1, 6, 0)
    gc_z5 = tmp_path / "z5.csv.gz"
    gc_h6 = tmp_path / "h6.csv.gz"
    write_source(gc_z5, [row(start + 1, symbol="GCZ5", instrument=42, size=5)])
    write_source(gc_h6, [row(start + 2, symbol="GCH6", instrument=43, size=7)])
    lookup = {start: 0}
    z5 = audit_symbol_group(
        [declaration(gc_z5, symbol="GCZ5", instrument=42, start=start, end=start + HOUR_NS)],
        lookup,
    )
    h6 = audit_symbol_group(
        [declaration(gc_h6, symbol="GCH6", instrument=43, start=start, end=start + HOUR_NS)],
        lookup,
    )
    assert z5["trade_volume_contracts_by_side"]["B"] == 5
    assert h6["trade_volume_contracts_by_side"]["B"] == 7
    assert sum(z5["_record_presence"]) == sum(h6["_record_presence"]) == 1
    assert sum(merge_presence_bitmaps([z5["_record_presence"], h6["_record_presence"]])) == 1


def test_schedule_template_and_unknown_holiday_event_silence_are_explicit():
    monday_maintenance = ns(2025, 1, 6, 22)  # 16:00 CT
    saturday = ns(2025, 1, 11, 18)
    sunday_open = ns(2025, 1, 12, 23)  # 17:00 CT
    assert not regular_globex_template_open(monday_maintenance)
    assert not regular_globex_template_open(saturday)
    assert regular_globex_template_open(sunday_open)

    start = ns(2025, 1, 6, 23)  # Monday 17:00 CT
    coverage = [(start, start + 3 * HOUR_NS)]
    summary = summarize_session_bins(coverage, {start // HOUR_NS, (start + 2 * HOUR_NS) // HOUR_NS})
    assert summary["regular_template_open_hour_bins_within_declared_file_windows"] == 3
    assert summary["event_silent_open_hour_bins"] == 1
    assert summary["historical_holiday_exception_status"] == "SCHEDULE_UNKNOWN"
    assert summary["event_silent_intervals"] == [{
        "start_utc": "2025-01-07T00:00:00Z",
        "end_utc": "2025-01-07T01:00:00Z",
        "hours": 1,
        "status": "EVENT_SILENCE_ON_REGULAR_TEMPLATE_OPEN_INTERVAL",
        "historical_holiday_exception_status": "SCHEDULE_UNKNOWN",
        "interpretation": "No good-ts_recv MBP-1 row was observed in these UTC hour bins; this is not proof of packet loss.",
    }]
