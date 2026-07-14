import numpy as np
import pyarrow as pa

from gx1.scripts.repair_entry_xau_structural_utility_labels_v1 import _repair_table


def test_xau_structural_repair_rewrites_dependent_scalar_targets() -> None:
    table = pa.table(
        {
            "y_direction": pa.array([1], type=pa.int32()),
            "y_trade": pa.array([1.0], type=pa.float32()),
            "y_tradable": pa.array([1.0], type=pa.float32()),
            "y_side": pa.array([1], type=pa.int8()),
            "y_side_mask": pa.array([1.0], type=pa.float32()),
            "y_bad_path": pa.array([1.0], type=pa.float32()),
            "mae_first_n_bps": pa.array([20.0], type=pa.float32()),
            "mfe_first_n_bps": pa.array([0.0], type=pa.float32()),
            "path_quality_bps": pa.array([-20.0], type=pa.float32()),
            "y_quality_score": pa.array([0.0], type=pa.float32()),
            "y_position_size_target": pa.array([0.1], type=pa.float32()),
            "atr_bps": pa.array([10.0], type=pa.float32()),
            "mfe_long_first_n_bps": pa.array([30.0], type=pa.float32()),
            "mae_long_first_n_bps": pa.array([2.0], type=pa.float32()),
            "mfe_short_first_n_bps": pa.array([0.0], type=pa.float32()),
            "mae_short_first_n_bps": pa.array([20.0], type=pa.float32()),
            "y_direction_long_score_bps": pa.array([40.0], type=pa.float32()),
            "y_direction_short_score_bps": pa.array([45.0], type=pa.float32()),
            "y_long_path_utility_bps": pa.array([40.0], type=pa.float32()),
            "y_short_path_utility_bps": pa.array([45.0], type=pa.float32()),
            "y_long_bad_path": pa.array([0.0], type=pa.float32()),
            "y_short_bad_path": pa.array([0.0], type=pa.float32()),
            "y_long_expected_mae_bps": pa.array([2.0], type=pa.float32()),
            "y_short_expected_mae_bps": pa.array([20.0], type=pa.float32()),
            "y_rising_channel_support_touch": pa.array([1.0], type=pa.float32()),
            "y_falling_channel_resistance_touch": pa.array([0.0], type=pa.float32()),
            "y_support_retest_continuation": pa.array([1.0], type=pa.float32()),
            "y_resistance_retest_continuation": pa.array([0.0], type=pa.float32()),
            "y_countertrend_short_trap": pa.array([0.0], type=pa.float32()),
            "y_countertrend_long_trap": pa.array([0.0], type=pa.float32()),
            "y_long_high_mae_low_mfe_early_failure": pa.array([0.0], type=pa.float32()),
            "y_short_high_mae_low_mfe_early_failure": pa.array([0.0], type=pa.float32()),
        }
    )

    repaired, stats = _repair_table(table)
    row = {name: repaired[name].to_pylist()[0] for name in repaired.schema.names}

    assert row["y_direction"] == 0
    assert row["y_trade"] == 1.0
    assert row["y_tradable"] == 1.0
    assert row["y_side"] == 0
    assert row["y_side_mask"] == 1.0
    assert row["mfe_first_n_bps"] == 30.0
    assert row["mae_first_n_bps"] == 2.0
    assert row["path_quality_bps"] == 28.0
    assert row["y_quality_score"] == 28.0
    assert row["y_bad_path"] == 0.0
    assert row["y_direction_long_score_bps"] == row["y_long_path_utility_bps"]
    assert row["y_direction_short_score_bps"] == row["y_short_path_utility_bps"]
    assert np.isfinite(float(row["y_position_size_target"]))
    assert stats["short_to_long_rows"] == 1
