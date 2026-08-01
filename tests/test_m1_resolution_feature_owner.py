from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features.htf_features import (
    MULTI_TF_FEATURE_COUNT_V4,
    build_multi_tf_per_bar_features_v4,
)


def _m1_frame(rows: int = 5_000) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01 00:00:00",
        periods=rows,
        freq="min",
        tz="UTC",
    )
    close = 1_800.0 + np.linspace(0.0, 20.0, rows) + np.sin(np.arange(rows) / 17.0)
    spread = np.full(rows, 0.2)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(rows, 100.0),
        },
        index=index,
    )


def test_v4_owner_accepts_causal_m1_clock_without_future_m5_bar() -> None:
    source = _m1_frame()
    features = build_multi_tf_per_bar_features_v4(
        source,
        base_bar_duration=pd.Timedelta(minutes=1),
    )

    assert tuple(features) == ("M5", "M15", "H1", "H4", "D1")
    for frame in features.values():
        assert frame.shape[1] == MULTI_TF_FEATURE_COUNT_V4
        assert 0 <= frame.attrs["causal_warmup_rows"] <= len(frame)

    last_source = source.index[-1]
    latest_m5_label = pd.Timestamp(features["M5"].index[-1])
    assert latest_m5_label <= last_source + pd.Timedelta(minutes=1) - pd.Timedelta(minutes=5)
