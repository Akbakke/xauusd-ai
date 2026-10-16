"""Causal timing and session disposition for frozen GC MBP-1 inputs.

Descriptive only: no feature building, roll choice, feed-failure inference, or
source admission.
"""
from __future__ import annotations

import csv
import gzip
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

GC_DISPOSITION_SCHEMA = "gx1_gc_source_disposition_v1"
NANOSECONDS = 1_000_000_000
MINUTE_NS = 60 * NANOSECONDS
HOUR_NS = 60 * MINUTE_NS
SUPPORT_START_NS = int(datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp()) * NANOSECONDS
STUDY_START_NS = int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp()) * NANOSECONDS
WINDOW_END_NS = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()) * NANOSECONDS
AVAILABILITY_OFFSET_NS = MINUTE_NS
F_BAD_TS_RECV = 1 << 3
F_MAYBE_BAD_BOOK = 1 << 2
UNDEF_TIMESTAMP = (1 << 64) - 1
UNDEF_SIZE = (1 << 32) - 1
SELECTED_GC_COLUMNS = (
    "ts_recv", "symbol", "instrument_id", "publisher_id",
    "action", "side", "size", "flags",
)
ALLOWED_ACTIONS = frozenset({"A", "C", "F", "M", "N", "R", "T"})
ALLOWED_SIDES = frozenset({"A", "B", "N"})
_CHICAGO = ZoneInfo("America/Chicago")


def _uint(value: str, name: str, maximum: int = UNDEF_TIMESTAMP) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"GC_DISPOSITION_INTEGER_REQUIRED:{name}")
    number = int(value)
    if not 0 <= number <= maximum:
        raise ValueError(f"GC_DISPOSITION_INTEGER_RANGE:{name}")
    return number


def causal_bar_index(ts_recv_ns: int, oanda_bar_index: Mapping[int, int]) -> int | None:
    """Return the exact M1 bar whose [t,t+60s) contains ts_recv and is available."""
    if ts_recv_ns < SUPPORT_START_NS or ts_recv_ns >= WINDOW_END_NS:
        return None
    bar_start_ns = (ts_recv_ns // MINUTE_NS) * MINUTE_NS
    index = oanda_bar_index.get(bar_start_ns)
    if index is None:
        return None
    decision_ns = bar_start_ns + AVAILABILITY_OFFSET_NS
    if decision_ns >= WINDOW_END_NS or not (bar_start_ns <= ts_recv_ns < decision_ns):
        return None
    return index


def regular_globex_template_open(hour_start_ns: int) -> bool:
    """Classify a UTC hour by the regular weekly/daily Gold Globex template."""
    local = datetime.fromtimestamp(
        (hour_start_ns + HOUR_NS // 2) / NANOSECONDS, timezone.utc
    ).astimezone(_CHICAGO)
    weekday, hour = local.weekday(), local.hour
    if weekday == 6:  # Sunday
        return hour >= 17
    if weekday == 4:  # Friday
        return hour < 16
    if weekday == 5:  # Saturday
        return False
    return hour != 16  # Monday-Thursday maintenance is 16:00-17:00 CT


def _iso_utc(ns: int) -> str:
    seconds, subsecond = divmod(ns, NANOSECONDS)
    stamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return stamp + (f".{subsecond:09d}" if subsecond else "") + "Z"


def merge_presence_bitmaps(bitmaps: Sequence[bytes]) -> bytes:
    """OR distinct-contract bar presence; never sum per-contract measures."""
    if not bitmaps:
        return b""
    width = len(bitmaps[0])
    if any(len(bitmap) != width for bitmap in bitmaps):
        raise ValueError("GC_DISPOSITION_BITMAP_LENGTH_MISMATCH")
    merged = bytearray(width)
    for bitmap in bitmaps:
        for index, present in enumerate(bitmap):
            if present:
                merged[index] = 1
    return bytes(merged)


def _coverage_intervals(items: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[int, int]], list[dict[str, str]]]:
    ordered = sorted(items, key=lambda item: int(item["start_ns"]))
    intervals: list[tuple[int, int]] = []
    gaps: list[dict[str, str]] = []
    previous_end = None
    for item in ordered:
        start, end = int(item["start_ns"]), int(item["end_ns"])
        if start % HOUR_NS or end % HOUR_NS or start >= end:
            raise ValueError("GC_DISPOSITION_DECLARATION_BOUNDARY_INVALID")
        if previous_end is not None:
            if start < previous_end:
                raise ValueError("GC_DISPOSITION_DECLARATION_OVERLAP")
            if start > previous_end:
                gaps.append({
                    "start_utc": _iso_utc(previous_end), "end_utc": _iso_utc(start),
                    "status": "NO_FILE_DECLARED_FOR_THIS_CONTRACT",
                })
        intervals.append((start, end))
        previous_end = end
    return intervals, gaps


def summarize_session_bins(
    coverage_intervals_ns: Sequence[tuple[int, int]],
    observed_hour_bins: set[int],
) -> dict[str, Any]:
    """Count template-open event-silent hours without inferring missing packets."""
    open_bins = observed_open_bins = closed_observed_bins = 0
    silent_runs: list[dict[str, Any]] = []
    run_start = run_end = None

    def close_run() -> None:
        nonlocal run_start, run_end
        if run_start is not None:
            silent_runs.append({
                "start_utc": _iso_utc(run_start),
                "end_utc": _iso_utc(run_end),
                "hours": (run_end - run_start) // HOUR_NS,
                "status": "EVENT_SILENCE_ON_REGULAR_TEMPLATE_OPEN_INTERVAL",
                "historical_holiday_exception_status": "SCHEDULE_UNKNOWN",
                "interpretation": "No good-ts_recv MBP-1 row was observed in these UTC hour bins; this is not proof of packet loss.",
            })
            run_start = run_end = None

    for start, end in coverage_intervals_ns:
        if start % HOUR_NS or end % HOUR_NS or start >= end:
            raise ValueError("GC_DISPOSITION_COVERAGE_BOUNDARY_INVALID")
        cursor = start
        while cursor + HOUR_NS <= end:
            if regular_globex_template_open(cursor):
                open_bins += 1
                if cursor // HOUR_NS in observed_hour_bins:
                    observed_open_bins += 1
                    close_run()
                else:
                    if run_start is None:
                        run_start = cursor
                    elif cursor != run_end:
                        close_run()
                        run_start = cursor
                    run_end = cursor + HOUR_NS
            else:
                close_run()
            cursor += HOUR_NS
        close_run()

    for hour_bin in observed_hour_bins:
        if not regular_globex_template_open(hour_bin * HOUR_NS):
            closed_observed_bins += 1

    return {
        "timezone": "America/Chicago",
        "regular_template": "Sunday 17:00 CT through Friday 16:00 CT, with daily 16:00-17:00 CT maintenance",
        "regular_template_open_hour_bins_within_declared_file_windows": open_bins,
        "observed_open_hour_bins_with_good_receive_timestamps": observed_open_bins,
        "event_silent_open_hour_bins": open_bins - observed_open_bins,
        "observed_hour_bins_in_regular_template_closed_time": closed_observed_bins,
        "historical_holiday_exception_status": "SCHEDULE_UNKNOWN",
        "event_silent_intervals": silent_runs,
        "inference_limit": "Regular weekly template only. No historical holiday exception is applied without exact official GC/Metals hours; event silence is not a feed-loss or packet-loss claim.",
    }


def audit_symbol_group(
    items: Sequence[Mapping[str, Any]],
    oanda_bar_index: Mapping[int, int],
) -> dict[str, Any]:
    """Stream all declared files for one outright, keeping contracts separate."""
    if not items:
        raise ValueError("GC_DISPOSITION_EMPTY_SYMBOL_GROUP")
    ordered = sorted(items, key=lambda item: int(item["start_ns"]))
    raw_symbol = str(ordered[0]["raw_symbol"])
    if any(item["raw_symbol"] != raw_symbol for item in ordered):
        raise ValueError("GC_DISPOSITION_SYMBOL_GROUP_MISMATCH")
    coverage, declaration_gaps = _coverage_intervals(ordered)
    width = len(oanda_bar_index)
    record_presence = bytearray(width)
    trade_presence = bytearray(width)
    study_record_presence = bytearray(width)
    study_trade_presence = bytearray(width)
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    trade_counts: Counter[str] = Counter()
    trade_volumes: Counter[str] = Counter()
    matched_trade_counts: Counter[str] = Counter()
    matched_trade_volumes: Counter[str] = Counter()
    study_trade_counts: Counter[str] = Counter()
    study_trade_volumes: Counter[str] = Counter()
    observed_hours: set[int] = set()
    file_reports: list[dict[str, Any]] = []

    for item in ordered:
        item_start, item_end = int(item["start_ns"]), int(item["end_ns"])
        path = Path(str(item["path"]))
        opener = gzip.open if item["compression"] == "gzip" else open
        previous_recv = None
        file_records = 0
        file_actions: Counter[str] = Counter()
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if (header is None or len(header) != len(set(header))
                    or any(name not in header for name in SELECTED_GC_COLUMNS)):
                raise ValueError("GC_DISPOSITION_HEADER_INVALID")
            column = {name: header.index(name) for name in SELECTED_GC_COLUMNS}
            for row in reader:
                if len(row) != len(header):
                    raise ValueError("GC_DISPOSITION_CSV_ROW_WIDTH_INVALID")
                file_records += 1
                counts["records"] += 1
                if file_records > int(item["max_records"]):
                    raise ValueError("GC_DISPOSITION_RECORD_BUDGET_EXCEEDED")
                action, side, symbol = row[column["action"]], row[column["side"]], row[column["symbol"]]
                if symbol != raw_symbol or action not in ALLOWED_ACTIONS or side not in ALLOWED_SIDES:
                    raise ValueError("GC_DISPOSITION_ACTION_SIDE_OR_SYMBOL_INVALID")
                if action == "R" and side != "N":
                    raise ValueError("GC_DISPOSITION_CLEAR_BOOK_SIDE_INVALID")
                recv = _uint(row[column["ts_recv"]], "ts_recv")
                instrument_id = _uint(row[column["instrument_id"]], "instrument_id", (1 << 32) - 1)
                publisher_id = _uint(row[column["publisher_id"]], "publisher_id", (1 << 16) - 1)
                flags = _uint(row[column["flags"]], "flags", 255)
                if (recv in {0, UNDEF_TIMESTAMP} or not item_start <= recv < item_end
                        or instrument_id != int(item["instrument_id"])
                        or publisher_id != int(item["publisher_id"])):
                    raise ValueError("GC_DISPOSITION_TIMESTAMP_OR_IDENTITY_INVALID")
                if previous_recv is not None and recv < previous_recv:
                    raise ValueError("GC_DISPOSITION_RECEIVE_TIME_REVERSAL")
                previous_recv = recv
                actions[action] += 1
                file_actions[action] += 1
                is_bad_recv = bool(flags & F_BAD_TS_RECV)
                counts["bad_receive_timestamp_records"] += int(is_bad_recv)
                counts["possible_bad_book_records"] += int(bool(flags & F_MAYBE_BAD_BOOK))

                size = None
                if action == "T":
                    size = _uint(row[column["size"]], "size", UNDEF_SIZE)
                    if size == 0:
                        raise ValueError("GC_DISPOSITION_ZERO_TRADE_SIZE")
                    counts["trade_records"] += 1
                    trade_counts[side] += 1
                    trade_volumes[side] += size
                    counts["unknown_aggressor_trades"] += int(side == "N")

                if is_bad_recv:
                    continue
                observed_hours.add(recv // HOUR_NS)
                matched_index = causal_bar_index(recv, oanda_bar_index)
                if matched_index is None:
                    continue
                counts["matched_records_support"] += 1
                record_presence[matched_index] = 1
                bar_start = (recv // MINUTE_NS) * MINUTE_NS
                in_study = bar_start >= STUDY_START_NS
                if in_study:
                    counts["matched_records_study"] += 1
                    study_record_presence[matched_index] = 1
                if size is None:
                    continue
                counts["matched_trade_records_support"] += 1
                matched_trade_counts[side] += 1
                matched_trade_volumes[side] += size
                trade_presence[matched_index] = 1
                if in_study:
                    counts["matched_trade_records_study"] += 1
                    study_trade_counts[side] += 1
                    study_trade_volumes[side] += size
                    study_trade_presence[matched_index] = 1

        if file_records == 0:
            raise ValueError("GC_DISPOSITION_EMPTY_SOURCE_FILE")
        print(
            f"GC_DISPOSITION_FILE_COMPLETE symbol={raw_symbol} records={file_records} "
            f"cumulative_symbol_records={counts['records']}",
            flush=True,
        )
        file_reports.append({
            "path": str(path),
            "sha256": str(item["sha256"]),
            "receipt_sha256": str(item["receipt_sha256"]),
            "declared_start_utc": _iso_utc(item_start),
            "declared_end_utc": _iso_utc(item_end),
            "records": file_records,
            "actions": dict(sorted(file_actions.items())),
        })

    total_trade_volume = sum(trade_volumes.values())
    return {
        "raw_symbol": raw_symbol,
        "source_files": file_reports,
        "declared_coverage_ranges_utc": [
            {"start_utc": _iso_utc(start), "end_utc": _iso_utc(end)}
            for start, end in coverage
        ],
        "gaps_between_declared_file_windows": declaration_gaps,
        "counters": dict(sorted(counts.items())),
        "actions": dict(sorted(actions.items())),
        "trade_counts_by_side": {side: trade_counts[side] for side in ("B", "A", "N")},
        "trade_volume_contracts_by_side": {side: trade_volumes[side] for side in ("B", "A", "N")},
        "known_aggressor_delta_contracts": trade_volumes["B"] - trade_volumes["A"],
        "unknown_aggressor_trade_volume_share": (trade_volumes["N"] / total_trade_volume) if total_trade_volume else None,
        "matched_support_trade_counts_by_side": {side: matched_trade_counts[side] for side in ("B", "A", "N")},
        "matched_support_trade_volume_contracts_by_side": {side: matched_trade_volumes[side] for side in ("B", "A", "N")},
        "matched_support_known_aggressor_delta_contracts": matched_trade_volumes["B"] - matched_trade_volumes["A"],
        "matched_study_trade_counts_by_side": {side: study_trade_counts[side] for side in ("B", "A", "N")},
        "matched_study_trade_volume_contracts_by_side": {side: study_trade_volumes[side] for side in ("B", "A", "N")},
        "matched_study_known_aggressor_delta_contracts": study_trade_volumes["B"] - study_trade_volumes["A"],
        "contract_session_template_disposition": summarize_session_bins(coverage, observed_hours),
        "_record_presence": bytes(record_presence),
        "_trade_presence": bytes(trade_presence),
        "_study_record_presence": bytes(study_record_presence),
        "_study_trade_presence": bytes(study_trade_presence),
    }
