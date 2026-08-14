from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd

from gx1.contracts.entry_direction_target_policy_v1 import (
    fit_entry_direction_target_policy,
)


@lru_cache(maxsize=32)
def _cached_policy(
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
    train_start_utc: str,
    train_end_utc: str,
) -> dict[str, object]:
    train_start = pd.Timestamp(train_start_utc)
    train_end = pd.Timestamp(train_end_utc)
    time = pd.date_range(
        train_start,
        periods=1200,
        freq="5min",
    )
    if train_end < time[-1] + pd.Timedelta(minutes=5):
        raise RuntimeError("ENTRY_DIRECTION_POLICY_FIXTURE_TRAIN_RANGE_TOO_SHORT")
    phase = np.arange(len(time), dtype=np.float64)
    mid = 1800.0 + np.sin(phase / 8.0) * 2.0 + phase * 0.0005
    return fit_entry_direction_target_policy(
        closed_m5=pd.DataFrame(
            {
                "time": time,
                "bid_close": mid - 0.05,
                "ask_close": mid + 0.05,
            }
        ),
        train_start=train_start,
        train_end=train_end,
        source_parquet_sha256=source_parquet_sha256,
        tape_provenance_sha256=tape_provenance_sha256,
    )


def entry_direction_target_policy_fixture(
    *,
    source_parquet_sha256: str = "a" * 64,
    tape_provenance_sha256: str = "b" * 64,
    train_start_utc: str = "2020-01-01T00:00:00+00:00",
    train_end_utc: str = "2020-01-05T04:00:00+00:00",
) -> dict[str, object]:
    return deepcopy(
        _cached_policy(
            source_parquet_sha256,
            tape_provenance_sha256,
            train_start_utc,
            train_end_utc,
        )
    )
