#!/usr/bin/env python3
"""Build the research-only US macro event calendar from stored raw source bytes.

Research evidence only (GX1_RULES.md rule 1): a public release schedule, never an Entry input.
Sources are fetched once, stored unchanged under ``<root>/raw`` and hashed into the manifest:

* FOMC scheduled meetings: federalreserve.gov ``fomchistorical<YEAR>.htm`` (2013-2020) and
  ``fomccalendars.htm`` (2021-). The statement date is the meeting's last day; unscheduled and
  cancelled meetings are excluded. Calendar-page rows are cross-checked against the date in their
  statement link (``monetaryYYYYMMDDa``).
* Employment Situation and CPI: ALFRED release dates (rid 50 and 10). ALFRED lists every
  publication, including revisions; the headline release is the first date in a month for the
  Employment Situation and the last date in a month for CPI (February seasonal-factor revisions
  precede the CPI release).

Release instants: FOMC statements 14:00 America/New_York (uniform from 2013), Employment Situation
and CPI 08:30 America/New_York; DST from the tz database.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
MONTHS = {m: i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
FOMC_HISTORICAL_YEARS = tuple(range(2013, 2021))
# First bar of the pair tape (vedtak OANDA_PAIR_PRETEST_2009_20260927); older publications are out of scope.
CALENDAR_START = dt.date(2009, 6, 1)
RELEASE_LOCAL_TIME = {"fomc": dt.time(14, 0), "nfp": dt.time(8, 30), "cpi": dt.time(8, 30)}
SOURCES = {
    "fomc_calendar": ("fed_fomccalendars.htm", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
    "nfp": ("alfred_release_dates_rid50_employment_situation.txt",
            "https://alfred.stlouisfed.org/release/downloaddates?rid=50&ff=txt"),
    "cpi": ("alfred_release_dates_rid10_consumer_price_index.txt",
            "https://alfred.stlouisfed.org/release/downloaddates?rid=10&ff=txt"),
    **{f"fomc_{y}": (f"fed_fomchistorical_{y}.htm",
                     f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
       for y in FOMC_HISTORICAL_YEARS},
}
_HISTORICAL = re.compile(
    r"(?P<m1>[A-Z][a-z]+)(?:/(?P<m2>[A-Z][a-z]+))?\s+(?P<d1>\d{1,2})(?:-(?P<d2>\d{1,2}))?"
    r"(?P<note>\s*\([a-z]+\))?\s+Meeting - (?P<year>\d{4})"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meeting_end(year: int, m1: str, m2: str | None, d1: str, d2: str | None) -> dt.date:
    month = MONTHS[m2 or m1]
    return dt.date(year, month, int(d2 or d1))


def fomc_historical(text: str, year: int) -> list[dt.date]:
    dates = set()
    for match in _HISTORICAL.finditer(text):
        if int(match["year"]) != year or match["note"]:
            continue
        dates.add(_meeting_end(year, match["m1"], match["m2"], match["d1"], match["d2"]))
    return sorted(dates)


def fomc_calendar(text: str) -> list[dt.date]:
    dates = []
    panels = re.split(r"<h4><a id=\"\d+\">(\d{4}) FOMC Meetings</a></h4>", text)
    for year, body in zip(panels[1::2], panels[2::2]):
        # Alternate rows carry a shading class: "row fomc-meeting" and "fomc-meeting--shaded row fomc-meeting".
        for row in re.split(r'class="(?:fomc-meeting--shaded )?row fomc-meeting"', body)[1:]:
            month = re.search(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>", row)
            day = re.search(r"fomc-meeting__date[^>]*>\s*([^<]+?)\s*</div>", row)
            if not month or not day or not re.fullmatch(r"\d{1,2}(-\d{1,2})?\*?", day[1]):
                continue  # unscheduled, cancelled or notation vote
            months = month[1].strip().split("/")
            days = day[1].rstrip("*").split("-")
            end = _meeting_end(int(year), months[0], months[1] if len(months) > 1 else None,
                               days[0], days[1] if len(days) > 1 else None)
            link = re.search(r"monetary(\d{8})a", row)
            if link and dt.datetime.strptime(link[1], "%Y%m%d").date() != end:
                raise RuntimeError(f"FOMC_CALENDAR_STATEMENT_DATE_MISMATCH: {end} vs {link[1]}")
            dates.append(end)
    return sorted(set(dates))


def alfred_headline_dates(text: str, *, keep: str) -> list[dt.date]:
    by_month: dict[tuple[int, int], list[dt.date]] = {}
    for raw in re.findall(r"^(\d{4}-\d{2}-\d{2})\s*$", text, re.M):
        day = dt.date.fromisoformat(raw)
        by_month.setdefault((day.year, day.month), []).append(day)
    pick = min if keep == "first" else max
    return sorted(pick(days) for days in by_month.values())


def release_utc(kind: str, day: dt.date) -> dt.datetime:
    local = dt.datetime.combine(day, RELEASE_LOCAL_TIME[kind], tzinfo=NEW_YORK)
    return local.astimezone(dt.timezone.utc)


def build(root: Path) -> dict:
    raw = root / "raw"
    text = {key: (raw / name).read_text(encoding="utf-8", errors="strict") for key, (name, _) in SOURCES.items()}
    fomc = set(fomc_calendar(text["fomc_calendar"]))
    for year in FOMC_HISTORICAL_YEARS:
        fomc.update(fomc_historical(text[f"fomc_{year}"], year))
    events = [("fomc", day) for day in sorted(fomc)]
    events += [("nfp", day) for day in alfred_headline_dates(text["nfp"], keep="first")]
    events += [("cpi", day) for day in alfred_headline_dates(text["cpi"], keep="last")]
    rows = sorted((release_utc(kind, day).isoformat(), kind, day.isoformat())
                  for kind, day in events if day >= CALENDAR_START)
    for _, kind, day in rows:
        if dt.date.fromisoformat(day).weekday() >= 5:
            raise RuntimeError(f"MACRO_EVENT_ON_WEEKEND: {kind} {day}")
    csv = "release_utc,event,release_date_new_york\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows)
    return {
        "csv": csv,
        "manifest": {
            "schema_version": "gx1_macro_event_calendar_research_v1",
            "purpose": "research measurement only (GX1_RULES.md rule 1); never an Entry input",
            "sources": {key: {"file": f"raw/{name}", "url": url, "sha256": _sha256(raw / name)}
                        for key, (name, url) in SOURCES.items()},
            "rules": {
                "fomc": "scheduled meetings only; statement on the last meeting day at 14:00 New York",
                "nfp": "ALFRED rid 50, first publication date per month, 08:30 New York",
                "cpi": "ALFRED rid 10, last publication date per month, 08:30 New York",
            },
            "counts": {kind: sum(1 for _, k, _ in rows if k == kind) for kind in ("fomc", "nfp", "cpi")},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fetched-utc", required=True, help="when the raw files were fetched")
    args = parser.parse_args()
    out_csv, out_manifest = args.root / "events.csv", args.root / "MANIFEST.json"
    if out_csv.exists() or out_manifest.exists():
        raise RuntimeError("MACRO_CALENDAR_OUTPUT_EXISTS")
    result = build(args.root)
    out_csv.write_text(result["csv"], encoding="utf-8")
    manifest = {**result["manifest"], "fetched_utc": args.fetched_utc,
                "events_csv": {"file": "events.csv", "sha256": _sha256(out_csv)}}
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
