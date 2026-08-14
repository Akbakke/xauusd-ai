from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from gx1.contracts.entry_position_size_target_policy_v1 import (
    fit_entry_position_size_target_policy,
)
from tests.entry_direction_target_policy_support import (
    entry_direction_target_policy_fixture,
)


_ARTIFACT_ROOT_OWNER = tempfile.TemporaryDirectory(
    prefix="gx1_position_size_policy_fixture."
)
_ARTIFACT_ROOT = Path(_ARTIFACT_ROOT_OWNER.name)


@lru_cache(maxsize=32)
def _cached_policy(
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
    train_start_utc: str,
    train_end_utc: str,
) -> dict[str, object]:
    direction = entry_direction_target_policy_fixture(
        source_parquet_sha256=source_parquet_sha256,
        tape_provenance_sha256=tape_provenance_sha256,
        train_start_utc=train_start_utc,
        train_end_utc=train_end_utc,
    )
    time = pd.date_range(pd.Timestamp(train_start_utc), periods=1200, freq="5min")
    phase = np.arange(len(time), dtype=np.float64)
    mid = 1800.0 + np.sin(phase / 8.0) * 2.0 + phase * 0.0005
    return fit_entry_position_size_target_policy(
        closed_m5=pd.DataFrame(
            {
                "time": time,
                "bid_close": mid - 0.05,
                "ask_close": mid + 0.05,
            }
        ),
        entry_direction_target_policy=direction,
        source_parquet_sha256=source_parquet_sha256,
        tape_provenance_sha256=tape_provenance_sha256,
        ecdf_artifact_path=(
            _ARTIFACT_ROOT
            / (
                hashlib.sha256(
                    (
                        f"{source_parquet_sha256}:{tape_provenance_sha256}:"
                        f"{train_start_utc}:{train_end_utc}"
                    ).encode("utf-8")
                ).hexdigest()
                + ".npy"
            )
        ),
    )


def entry_position_size_target_policy_fixture(
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
