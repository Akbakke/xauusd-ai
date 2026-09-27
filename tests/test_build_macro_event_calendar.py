"""Parsing rules of the research macro calendar (synthetic snippets prove only the parser)."""
from __future__ import annotations

import datetime as dt

import pytest

from gx1.scripts import build_macro_event_calendar_v1 as calendar


def test_historical_page_keeps_scheduled_meetings_and_cross_month_end_date():
    text = ("April/May 30-1 Meeting - 2019 ... March 15 (unscheduled) Meeting - 2019 ... "
            "March 17-18 (cancelled) Meeting - 2019 ... December 10-11 Meeting - 2019 ... June 5 Meeting - 2018")
    assert calendar.fomc_historical(text, 2019) == [dt.date(2019, 5, 1), dt.date(2019, 12, 11)]


def _row(shaded: bool, month: str, day: str, link: str = "") -> str:
    cls = "fomc-meeting--shaded row fomc-meeting" if shaded else "row fomc-meeting"
    return (f'<div class="{cls}"><div class="fomc-meeting__month x"><strong>{month}</strong></div>'
            f'<div class="fomc-meeting__date y">{day}</div>{link}</div>')


def test_calendar_page_reads_shaded_rows_skips_unscheduled_and_checks_statement_link():
    body = ('<h4><a id="1">2022 FOMC Meetings</a></h4>'
            + _row(False, "January", "25-26", '<a href="/x/monetary20220126a.htm">')
            + _row(True, "Apr/May", "3-4*")
            + _row(False, "March", "(unscheduled)"))
    assert calendar.fomc_calendar(body) == [dt.date(2022, 1, 26), dt.date(2022, 5, 4)]
    bad = '<h4><a id="1">2022 FOMC Meetings</a></h4>' + _row(False, "January", "25-26", '<a href="/x/monetary20220127a.htm">')
    with pytest.raises(RuntimeError, match="STATEMENT_DATE_MISMATCH"):
        calendar.fomc_calendar(bad)


def test_alfred_headline_rule_first_for_employment_last_for_cpi():
    text = "header\n2015-02-06\n2015-02-20\n2015-02-26\n2015-03-06\n"
    assert calendar.alfred_headline_dates(text, keep="first") == [dt.date(2015, 2, 6), dt.date(2015, 3, 6)]
    assert calendar.alfred_headline_dates(text, keep="last") == [dt.date(2015, 2, 26), dt.date(2015, 3, 6)]


def test_release_instants_follow_new_york_dst():
    assert calendar.release_utc("nfp", dt.date(2015, 1, 9)).isoformat() == "2015-01-09T13:30:00+00:00"
    assert calendar.release_utc("nfp", dt.date(2015, 7, 2)).isoformat() == "2015-07-02T12:30:00+00:00"
    assert calendar.release_utc("fomc", dt.date(2015, 12, 16)).isoformat() == "2015-12-16T19:00:00+00:00"
