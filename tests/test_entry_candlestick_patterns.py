import numpy as np
import pandas as pd

from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    build_entry_candlestick_pattern_layer,
    missing_candlestick_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


def test_candlestick_pattern_layer_builds_closed_bar_numeric_patterns() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=9, freq="5min", tz="UTC"),
            "open": [10.0, 10.4, 10.7, 10.2, 10.2, 9.7, 10.4, 10.8, 11.2],
            "high": [10.5, 10.8, 10.9, 10.3, 10.4, 10.8, 10.9, 11.3, 11.7],
            "low": [9.9, 10.3, 10.0, 9.6, 9.5, 9.6, 10.2, 10.7, 11.1],
            "close": [10.4, 10.7, 10.1, 10.0, 9.9, 10.7, 10.85, 11.25, 11.6],
        }
    )

    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}

    assert tuple(names) == CANDLESTICK_PATTERN_FEATURE_NAMES
    assert out.shape == (9, len(CANDLESTICK_PATTERN_FEATURE_NAMES))
    assert np.isfinite(out).all()
    assert out[0].sum() == 0.0
    assert out[5, idx["candle.pattern_hammer_bull_reversal_score"]] > 0.0
    assert out[6, idx["candle.pattern_bullish_engulfing_score"]] > 0.0
    assert out[8, idx["candle.pattern_bull_continuation_pressure"]] > 0.0


def test_candlestick_source_contract_and_specialist_routing() -> None:
    assert missing_candlestick_source_fields(["time", "open", "high", "low", "close"]) == []
    assert missing_candlestick_source_fields(["time", "open", "high", "close"]) == ["low"]
    assert (
        classify_entry_specialist_feature("candle.pattern_bullish_engulfing_score")
        == "price_action_candle_encoder"
    )
