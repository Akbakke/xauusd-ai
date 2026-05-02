from __future__ import annotations

import math

import pandas as pd
import pytest

from gx1.analysis.shadow_meta_v1 import _build_replay_market_pressure_fields_v1


def test_build_replay_market_pressure_fields_v1_uses_backward_rolling_windows() -> None:
    replay_df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 104.0, 103.0],
            "high": [101.0, 103.0, 102.0, 105.0, 104.0],
            "low": [99.0, 101.0, 100.0, 102.0, 102.0],
        }
    )

    result = _build_replay_market_pressure_fields_v1(replay_df, windows=(3,))

    assert math.isnan(float(result.loc[0, "as_of_entry_replay_window_range_3_bps_v1"]))
    assert math.isnan(float(result.loc[1, "as_of_entry_replay_window_range_3_bps_v1"]))

    expected_up_move = (103.0 - 100.0) / 103.0 * 1e4
    expected_down_move = (105.0 - 103.0) / 103.0 * 1e4
    expected_range = (105.0 - 100.0) / 103.0 * 1e4

    assert result.loc[4, "as_of_entry_replay_window_up_move_3_bps_v1"] == pytest.approx(expected_up_move)
    assert result.loc[4, "as_of_entry_replay_window_down_move_3_bps_v1"] == pytest.approx(expected_down_move)
    assert result.loc[4, "as_of_entry_replay_window_range_3_bps_v1"] == pytest.approx(expected_range)
    assert result.loc[4, "as_of_entry_replay_window_directional_imbalance_3_bps_v1"] == pytest.approx(
        expected_up_move - expected_down_move
    )
    assert result.loc[4, "as_of_entry_replay_window_close_in_range_3_v1"] == pytest.approx(0.6)
