from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder
from tests.test_entry_observed_market_horizons_v1 import _m1_sessions


def test_horizon_label_frame_is_nan_beyond_split_end_and_keeps_every_row():
    m1 = _m1_sessions(weeks=3)
    decisions = pd.DatetimeIndex(["2026-01-06T10:00:00Z", "2026-01-08T10:00:00Z", "2026-01-21T10:00:00Z"])
    frame = builder._trading_day_horizon_label_frame(
        m5_decision_times=pd.Series(decisions), closed_m1=m1, split_end=pd.Timestamp("2026-01-16T22:00:00Z"))
    assert list(frame.columns) == ["time", *builder.TRADING_DAY_HORIZON_LABEL_COLUMNS]
    assert len(frame) == 3 and frame["time"].tolist() == decisions.tolist()
    # first row: 1 d and 5 d exits (Jan 7, Jan 13) lie before the split end; 20 d does not
    assert np.isfinite(frame["elapsed_seconds_at_1d"].iloc[0]) and np.isfinite(frame["elapsed_seconds_at_5d"].iloc[0])
    assert np.isnan(frame["elapsed_seconds_at_20d"].iloc[0]) and np.isnan(frame["y_long_gross_at_20d_bps"].iloc[0])
    # second row: 5 d exit on Jan 15 is inside; 1 d on Jan 9 inside
    assert np.isfinite(frame["y_long_gross_at_5d_bps"].iloc[1]) and frame["elapsed_seconds_at_5d"].iloc[1] == 7 * 86400
    # third row: every horizon crosses the split end
    assert frame[list(builder.TRADING_DAY_HORIZON_LABEL_COLUMNS)].iloc[2].isna().all()
    assert frame.dtypes[builder.TRADING_DAY_HORIZON_LABEL_COLUMNS[0]] == np.float32


def test_m5_path_label_owner_keeps_primary_rows_when_long_support_is_unavailable_and_censors_boundary():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN
    from tests.test_entry_causal_m1_outcomes import _m1_frame
    m1 = _m1_frame(periods=110)
    decisions = m1.time.iloc[[0, 5]]
    # First fill00:05 plus19 M5 bars ends01:40; second ends01:45.
    frame = builder._m5_path_label_frame(m5_decision_times=decisions, closed_m1=m1,
        split_end=pd.Timestamp('2026-01-05T01:40Z'))
    assert len(frame) == 2 and frame[M5_PATH_VALID_COLUMN].tolist() == [True, False]
    assert np.isfinite(frame[list(M5_PATH_TARGET_COLUMNS)].iloc[0]).all()
    assert frame[list(M5_PATH_TARGET_COLUMNS)].iloc[1].isna().all()
    assert not set(builder.TRADING_DAY_HORIZON_LABEL_COLUMNS) & set(frame.columns)
