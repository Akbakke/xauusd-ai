#!/usr/bin/env python3
"""Generate TRUTH Monday-week calendar entries for 2020-2024.

Background
----------
TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json currently lists 68 Monday-weeks
covering 2025 + Q1 2026. To replay 2020-2024 we extend the calendar with
~260 additional weeks following the same conventions:
  - Calendar week: MONDAY 00:00 UTC → next MONDAY 00:00 UTC (exclusive)
  - Trading start: same Monday 00:00 UTC (forex actually opens Sunday
    22:00 but the calendar's contract floors to Monday 00:00 UTC)
  - Friday flat cutoff: 20:55 UTC same week
  - Each week gets a unique run_id of the form
    TRUTH_MONFRI_WEEK_YYYYMMDD_YYYYMMDD (start day - end day).

This script writes a NEW calendar JSON
TRUTH_CALENDAR_REORG_MONDAY_WEEK_2020_2026_V1.json that is the union of
the existing 2025-2026 entries + the 2020-2024 entries we generate. The
existing canonical calendar is not modified.

Output:
  /home/andre2/GX1_DATA/reports/truth_e2e_sanity/TRUTH_CALENDAR_REORG_MONDAY_WEEK_2020_2026_V1.json

Research-only data prep. Output is consumed by
launch_truth_monday_week_replay_v1.py via --calendar argument.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXISTING_CALENDAR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"
)
DEFAULT_OUT_PATH = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/TRUTH_CALENDAR_REORG_MONDAY_WEEK_2020_2026_V1.json"
)
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2024  # inclusive; 2025+ comes from existing calendar
# BASE34_EXTENDED_2020_2026 prebuilt starts 2020-11-09 — earlier weeks would
# fail with [CHUNK 0] PREBUILT empty, so floor the calendar there.
DEFAULT_MIN_START_UTC = "2020-11-09T00:00:00+00:00"


def _first_monday_on_or_after(d: datetime) -> datetime:
    """Return the first Monday at 00:00 UTC on or after d."""
    days_ahead = (0 - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def _generate_week_entries(
    start_year: int, end_year: int, min_start_utc: datetime | None = None
) -> list[dict[str, Any]]:
    """Generate Monday-week entries [start_year-01-01, end_year-12-31].

    If min_start_utc is supplied, weeks whose Monday cursor falls before it
    are dropped (used to align with prebuilt-data availability).
    """
    entries: list[dict[str, Any]] = []
    floor = (
        _first_monday_on_or_after(min_start_utc)
        if min_start_utc is not None
        else None
    )
    cursor = _first_monday_on_or_after(
        datetime(start_year, 1, 1, tzinfo=timezone.utc)
    )
    if floor is not None and cursor < floor:
        cursor = floor
    cutoff = datetime(end_year + 1, 1, 1, tzinfo=timezone.utc)
    while cursor < cutoff:
        end_excl = cursor + timedelta(days=7)
        friday = cursor + timedelta(days=4)
        friday_flat_cutoff = friday.replace(hour=20, minute=55, second=0, microsecond=0)
        run_id = (
            f"TRUTH_MONFRI_WEEK_{cursor.strftime('%Y%m%d')}_"
            f"{end_excl.strftime('%Y%m%d')}"
        )
        entries.append(
            {
                "run_id": run_id,
                "calendar_start_utc": cursor.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "calendar_end_exclusive_utc": end_excl.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00"
                ),
                "trading_start_utc": cursor.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "friday_flat_cutoff_utc": friday_flat_cutoff.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00"
                ),
                "quarantine_status": "ACTIVE_CANDIDATE",
                "quarantine_reason": None,
                "weekend_no_action": True,
            }
        )
        cursor = end_excl
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend TRUTH calendar with Monday-weeks for 2020-2024."
    )
    parser.add_argument(
        "--existing-calendar", type=Path, default=DEFAULT_EXISTING_CALENDAR
    )
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--min-start-utc",
        type=str,
        default=DEFAULT_MIN_START_UTC,
        help="Drop weeks before this UTC timestamp (prebuilt-data floor).",
    )
    args = parser.parse_args()
    min_start_dt = (
        datetime.fromisoformat(args.min_start_utc)
        if args.min_start_utc
        else None
    )

    existing = {}
    if args.existing_calendar.exists():
        existing = json.loads(args.existing_calendar.read_text(encoding="utf-8"))
    existing_weeks = list(existing.get("full_monday_weeks", []))
    existing_run_ids = {w["run_id"] for w in existing_weeks}
    new_entries = _generate_week_entries(
        args.start_year, args.end_year, min_start_utc=min_start_dt
    )
    # Filter out any week that is already in the existing calendar.
    additions = [w for w in new_entries if w["run_id"] not in existing_run_ids]
    combined = additions + existing_weeks
    combined.sort(key=lambda w: w["trading_start_utc"])

    payload = dict(existing) if existing else {}
    payload["artifact_name"] = "TRUTH_CALENDAR_REORG_MONDAY_WEEK_2020_2026_V1"
    payload["full_monday_weeks"] = combined
    payload["weeks_added_v1"] = len(additions)
    payload["weeks_existing_v1"] = len(existing_weeks)
    payload["weeks_total_v1"] = len(combined)
    payload["created_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[CALENDAR] wrote {args.out_path} weeks_total={len(combined)} "
        f"added={len(additions)} existing={len(existing_weeks)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
