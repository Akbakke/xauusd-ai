"""
Utility script for generating a tiny bid/ask M5 dataset that can be checked
into the repository and exercised in CI smoke tests.

Usage
-----
python -m gx1.tests.build_smoke_dataset

What it does
------------
* Loads the high-resolution 2025 bid/ask dataset from data/raw/.
* Filters down to a small ASIA-session window (2025-06-01 → 2025-06-03).
* Keeps only the columns expected by the replay pipeline
  (open, high, low, close, volume + bid/ask variants).
* Writes the trimmed sample to gx1/tests/data/xauusd_m5_smoke_2025_06_01_03.parquet.

This script intentionally keeps the output below a few MB so that it can live
inside the repo and be fetched in GitHub Actions without incurring huge costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SmokeDatasetConfig:
    source: Path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
    target: Path = Path("gx1/tests/data/xauusd_m5_smoke_2025_06_01_03.parquet")
    start_ts: str = "2025-06-01T00:00:00Z"
    end_ts: str = "2025-06-03T23:59:59Z"
    max_rows: int = 2000
    asia_hours_utc: tuple[int, ...] = tuple(range(0, 9))  # 00:00-08:55 UTC


REQUIRED_COLUMNS: Iterable[str] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)


def infer_asia_mask(index: pd.DatetimeIndex, asia_hours: Iterable[int]) -> pd.Series:
    """Return boolean mask for timestamps that fall in our ASIA hour bucket."""
    hours = index.tz_convert("UTC").hour
    return pd.Series(hours.isin(list(asia_hours)), index=index)


def build_smoke_dataset(config: SmokeDatasetConfig = SmokeDatasetConfig()) -> Path:
    if not config.source.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {config.source}. "
            "Please download the 2025 bid/ask dataset before building the smoke file."
        )

    config.target.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(config.source)
    # Normalize to datetime index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")

    start_ts = pd.Timestamp(config.start_ts).tz_convert("UTC")
    end_ts = pd.Timestamp(config.end_ts).tz_convert("UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    asia_mask = infer_asia_mask(df.index, config.asia_hours_utc)
    df = df[asia_mask]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns for smoke dataset: {missing_cols}")

    if len(df) > config.max_rows:
        df = df.iloc[: config.max_rows]

    # Preserve schema order for replay features.
    df = df.loc[:, list(REQUIRED_COLUMNS)]
    df.to_parquet(config.target)
    print(f"[SMOKE DATA] Wrote {len(df):,} rows to {config.target}")
    return config.target


if __name__ == "__main__":
    build_smoke_dataset()
