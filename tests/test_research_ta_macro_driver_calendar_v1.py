"""The declared D+1 New York publication lag uses calendar dates, not 48 hours."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gx1.scripts import research_ta_campaign_v1 as owner


@pytest.mark.parametrize('observation_date', [
    '2024-03-08', '2024-03-09', '2024-03-10',
    '2024-11-01', '2024-11-02', '2024-11-03', '2024-11-04',
])
def test_driver_available_at_is_midnight_after_next_new_york_calendar_day(observation_date):
    day = date.fromisoformat(observation_date)
    expected = datetime.combine(day + timedelta(days=2), time(), ZoneInfo('America/New_York'))
    expected = pd.Timestamp(expected.astimezone(timezone.utc))
    actual = owner._driver_available_at(pd.DatetimeIndex([observation_date]))[0]
    assert actual == expected


def test_fall_back_observation_cannot_enter_one_calendar_day_early():
    available = owner._driver_available_at(pd.DatetimeIndex(['2024-11-02']))
    decision_times = pd.DatetimeIndex([
        '2024-11-04 04:55:00Z', '2024-11-04 05:00:00Z', '2024-11-04 05:05:00Z',
    ])
    values = pd.DataFrame({'driver': [1.]})
    joined = owner.align_asof(decision_times, pd.Series(available), values, inclusive=False)
    assert joined['driver'].isna().tolist() == [True, True, False]
    assert joined['driver'].iloc[-1] == 1.


@pytest.mark.parametrize('source_timezone', ['America/New_York', 'UTC'])
def test_driver_available_at_uses_the_new_york_date_for_aware_inputs(source_timezone):
    observed = pd.DatetimeIndex(['2024-11-03 00:30:00'], tz='America/New_York')
    observed = observed.tz_convert(source_timezone)
    actual = owner._driver_available_at(observed)
    assert actual[0] == pd.Timestamp('2024-11-05 05:00:00Z')
