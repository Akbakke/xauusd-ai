from __future__ import annotations

import re

import pandas as pd


_GRANULARITY_RE = re.compile(r"^(?P<unit>[MH])(?P<n>\d+)$")


def granularity_to_minutes(granularity: str) -> int:
    """
    Convert OANDA granularity names like M1/M5/M15/H1/H4 to minutes.
    """
    g = granularity.strip().upper()
    m = _GRANULARITY_RE.match(g)
    if not m:
        raise ValueError(f"Unsupported granularity: {granularity}")
    unit = m.group("unit")
    n = int(m.group("n"))
    if unit == "M":
        return n
    if unit == "H":
        return 60 * n
    raise ValueError(f"Unsupported granularity: {granularity}")


def granularity_to_pandas_freq(granularity: str) -> str:
    """
    Convert OANDA granularity name to a pandas frequency string.
    """
    minutes = granularity_to_minutes(granularity)
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours}H"
    return f"{minutes}min"


def granularity_to_timedelta(granularity: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=granularity_to_minutes(granularity))


def floor_timestamp(ts: pd.Timestamp, granularity: str) -> pd.Timestamp:
    return ts.floor(granularity_to_pandas_freq(granularity))

