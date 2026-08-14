from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    multi_tf_resample,
)
from gx1.scripts import materialize_entry_model_native_train_feature_ranker_v1 as ranker
from gx1.scripts.augment_forward_outcome_v2 import (
    attach_group_a_ctx_columns_parallel,
)


def _market_history() -> pd.DataFrame:
    rows = 75 * 24 * 12
    index = pd.date_range(
        "2025-01-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )
    position = np.arange(rows, dtype=np.float64)
    day = position / (24 * 12)
    volatility = (
        0.04
        + 0.02 * np.sin(day / 3.0)
        + 0.015 * (day < 14.0)
    )
    change = volatility * (
        0.55 * np.sin(position / 17.0)
        + 0.35 * np.sin(position / 61.0)
        + 0.10 * np.cos(position / 7.0)
    )
    close = 2_000.0 + np.cumsum(change)
    open_ = np.concatenate(([close[0]], close[:-1]))
    span = (
        np.abs(change)
        + 0.08
        + 0.04 * (1.0 + np.sin(position / 29.0))
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
        },
        index=index,
    )


def _minimal_valid_v4_mtf(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build contract-valid lookup tables; this test targets Group-A history."""
    result: dict[str, pd.DataFrame] = {}
    width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    # V30 package 3 (2026-08-13): build the axis through the ONE cadence+origin
    # owner. A bare `resample(rule)` kept pandas' midnight-UTC D1 origin and
    # would no longer match the production freshness check, which now uses the
    # trading-day D1 bin.
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        index = (
            multi_tf_resample(history, timeframe)
            .agg({"close": "last"})
            .dropna()
            .index
        )
        values = np.zeros((len(index), width), dtype=np.float32)
        values[:, 0] = np.float32(10.0)  # positive atr_bps_14
        frame = pd.DataFrame(
            values,
            index=index,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["feats_np"] = values
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        result[timeframe] = frame
    return result


def test_ranker_raw_distances_match_dataset_with_exact_common_history() -> None:
    common_history = _market_history()
    multi_tf = _minimal_valid_v4_mtf(common_history)
    decision = common_history.iloc[-12:].copy()
    decision["smc_swing_state"] = 0
    decision = decision.reset_index(names="time")

    dataset_values = attach_group_a_ctx_columns_parallel(
        decision.copy(),
        multi_tf=multi_tf,
        context_m5=common_history,
        journal_label="dataset_common_history_regression",
        workers=1,
    )
    ranker_values = ranker._attach_ranker_group_a_with_common_history(
        decision.copy(),
        multi_tf=multi_tf,
        context_m5=common_history,
        workers=1,
    )

    fields = ["dist_to_R1_atr", "dist_to_d1_hi_atr"]
    np.testing.assert_array_equal(
        ranker_values[fields].to_numpy(dtype=np.float32),
        dataset_values[fields].to_numpy(dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="COMMON_HISTORY_REQUIRED"):
        ranker._attach_ranker_group_a_with_common_history(
            decision.copy(),
            multi_tf=multi_tf,
            context_m5=pd.DataFrame(),
            workers=1,
        )
