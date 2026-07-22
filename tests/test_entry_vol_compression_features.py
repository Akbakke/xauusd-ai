import numpy as np
import pytest

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_vol_compression_v1 import (
    VOL_COMPRESSION_FEATURE_NAMES,
    VOL_COMPRESSION_SOURCE_FIELDS,
    build_entry_vol_compression_layer,
    missing_entry_vol_compression_source_fields,
)


EXPECTED_VOL_COMPRESSION_FEATURE_NAMES = (
    "vol_compression.atr_percentile_blend",
    "vol_compression.atr_percentile_m5_minus_h1",
    "vol_compression.low_atr_percentile_pressure",
    "vol_compression.high_atr_percentile_pressure",
    "vol_compression.squeeze_state_score",
    "vol_compression.range_compression_stack",
    "vol_compression.compression_agreement_score",
    "vol_compression.compression_divergence_score",
    "vol_compression.compression_depth_score",
    "vol_compression.compression_persistence_score",
    "vol_compression.compression_release_impulse",
    "vol_compression.compression_release_up_pressure",
    "vol_compression.compression_release_down_pressure",
    "vol_compression.expansion_state_score",
    "vol_compression.expansion_acceleration_score",
    "vol_compression.expansion_direction_score",
    "vol_compression.expansion_direction_confidence",
    "vol_compression.high_vol_tail_risk",
    "vol_compression.high_vol_chop_tail_risk",
    "vol_compression.low_vol_breakout_setup",
    "vol_compression.low_vol_false_breakout_risk",
    "vol_compression.mtf_vol_agreement_score",
    "vol_compression.mtf_vol_divergence_score",
    "vol_compression.short_tf_vol_expansion_pressure",
    "vol_compression.higher_tf_vol_expansion_pressure",
    "vol_compression.squeeze_breakout_asym_score",
    "vol_compression.vol_term_structure_slope",
    "vol_compression.vol_forecast_confidence_score",
)


def _feature_names(extra: list[str] | None = None) -> list[str]:
    return list(dict.fromkeys([*VOL_COMPRESSION_SOURCE_FIELDS, *(extra or [])]))


def _matrix(names: list[str], n: int = 7) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("snap.atr_z", [-1.0, -1.4, -1.6, 1.1, 3.0, 1.3, -0.2])
    set_col("snap.rvol_20", [-1.0, -1.3, -1.5, 1.6, 3.5, 1.8, 0.2])
    set_col("snap._v1_range_z", [-0.8, -1.2, -1.5, 1.7, 3.3, 1.8, 0.4])
    set_col("snap._v1_bb_squeeze_20_2", [0.0, -0.7, -0.9, 0.5, 0.8, 0.5, -0.6])
    set_col("snap._v1_bb_bandwidth_delta_10", [0.0, -0.9, -1.0, 1.0, 1.3, 1.0, -0.6])
    set_col("snap.vol_ratio_5_20", [-0.5, -0.8, -1.0, 2.0, 3.0, 2.0, -0.5])
    set_col("snap.signed_vol_z_20", [0.0, 0.1, 0.2, 2.2, -0.2, -2.4, 0.0])
    set_col("snap.vol_z_20", [-0.6, -1.0, -1.2, 1.5, 3.0, 1.5, 0.1])
    set_col("snap.vol_pct_96", [0.4, 0.15, 0.10, 0.75, 0.98, 0.80, 0.50])
    set_col("ctx_cont.vol_pct_m5_1yr", [0.40, 0.12, 0.10, 0.80, 0.98, 0.82, 0.95])
    set_col("ctx_cont.vol_pct_h1_1yr", [0.42, 0.15, 0.12, 0.70, 0.90, 0.75, 0.10])
    set_col("ctx_cont.D1_atr_percentile_252", [0.45, 0.20, 0.15, 0.60, 0.96, 0.70, 0.20])
    set_col("ctx_cont.H1_range_compression_ratio", [1.0, 0.6, 0.5, 1.4, 2.0, 1.4, 0.55])
    set_col("ctx_cont.M15_range_compression_ratio", [1.0, 0.7, 0.5, 1.5, 2.0, 1.5, 0.6])
    set_col("ctx_cont.atr_ratio_m5_m15", [1.0, 0.6, 0.5, 2.2, 3.0, 2.1, 3.0])
    set_col("ctx_cont.atr_ratio_m5_h4", [1.0, 0.5, 0.5, 2.0, 3.2, 2.2, 3.5])
    set_col("ctx_cont.atr_ratio_m15_d1", [1.0, 0.7, 0.6, 1.5, 2.8, 1.7, 0.4])
    set_col("ctx_cont.atr_ratio_h1_d1", [1.0, 0.8, 0.7, 1.2, 2.5, 1.4, 0.3])
    set_col("ctx_cont.regime_tf_agreement_v3", [0.5, 0.8, 0.95, 0.8, 0.2, 0.8, 0.1])
    set_col("ctx_cont.regime_divergence_flag_v3", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    set_col("ctx_cat.atr_bucket", [2, 0, 0, 3, 4, 3, 2])
    set_col("ctx_cat.vol_regime_id", [2, 0, 0, 3, 4, 3, 2])
    set_col("snap.ret_1", [0.0, 0.2, 0.4, 10.0, -2.0, -10.0, 0.5])
    set_col("snap.ret_5", [0.0, 0.4, 0.8, 25.0, -5.0, -25.0, 1.0])
    set_col("snap.ret_20", [0.0, 0.5, 1.0, 40.0, -8.0, -40.0, 2.0])
    set_col("ctx_cont._v1h1_ema_diff", [0.0, 0.3, 0.5, 1.5, 0.0, -1.5, 0.0])
    set_col("ctx_cont._v1h4_ema_diff", [0.0, 0.3, 0.5, 1.2, 0.0, -1.2, 0.0])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", [0.0, 0.2, 0.4, 1.0, 0.0, -1.0, 0.0])
    set_col("ctx_cont.m15_trend_sign_canon_v2", [0.0, 0.2, 0.4, 1.0, 0.0, -1.0, 0.0])
    set_col("chart.foundation_compression_state", [0.0, 0.7, 0.9, 0.3, 0.1, 0.3, 0.7])
    set_col("chart.foundation_expansion_state", [0.0, 0.0, 0.1, 0.9, 0.8, 0.9, 0.1])
    set_col("chart.foundation_compression_release_trigger", [0.0, 0.1, 0.2, 4.5, 2.0, 4.5, 0.0])
    set_col("chart.foundation_compression_release_up", [0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0])
    set_col("chart.foundation_compression_release_down", [0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0])
    set_col("chart.foundation_impulse_direction", [0.0, 0.1, 0.2, 1.5, 0.0, -1.5, 0.0])
    set_col("snap._v1_kurt_r", [0.0, 0.1, 0.1, 1.0, 6.0, 1.0, 0.2])
    set_col("snap._v1_pk_sigma20", [0.0, -0.5, -0.8, 1.0, 4.0, 1.0, 0.0])
    return x


def test_vol_compression_feature_contract_is_stable() -> None:
    assert len(VOL_COMPRESSION_FEATURE_NAMES) == 28
    assert VOL_COMPRESSION_FEATURE_NAMES == EXPECTED_VOL_COMPRESSION_FEATURE_NAMES
    assert len(set(VOL_COMPRESSION_FEATURE_NAMES)) == 28


def test_vol_compression_layer_builds_requested_smart_features() -> None:
    names = _feature_names()
    out, out_names = build_entry_vol_compression_layer(_matrix(names), names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert tuple(out_names) == VOL_COMPRESSION_FEATURE_NAMES
    assert out.shape == (7, len(VOL_COMPRESSION_FEATURE_NAMES))
    assert np.isfinite(out).all()
    assert out[2, idx["vol_compression.compression_depth_score"]] > out[0, idx["vol_compression.compression_depth_score"]]
    assert out[2, idx["vol_compression.compression_agreement_score"]] > out[6, idx["vol_compression.compression_agreement_score"]]
    assert out[3, idx["vol_compression.compression_release_up_pressure"]] > out[3, idx["vol_compression.compression_release_down_pressure"]]
    assert out[5, idx["vol_compression.compression_release_down_pressure"]] > out[5, idx["vol_compression.compression_release_up_pressure"]]
    assert out[3, idx["vol_compression.expansion_direction_score"]] > 0.0
    assert out[5, idx["vol_compression.expansion_direction_score"]] < 0.0
    assert out[4, idx["vol_compression.high_vol_tail_risk"]] > out[1, idx["vol_compression.high_vol_tail_risk"]]
    assert out[2, idx["vol_compression.low_vol_breakout_setup"]] > out[4, idx["vol_compression.low_vol_breakout_setup"]]
    assert out[2, idx["vol_compression.mtf_vol_agreement_score"]] > out[6, idx["vol_compression.mtf_vol_agreement_score"]]
    assert out[6, idx["vol_compression.mtf_vol_divergence_score"]] > out[2, idx["vol_compression.mtf_vol_divergence_score"]]
    assert out[3, idx["vol_compression.short_tf_vol_expansion_pressure"]] > out[2, idx["vol_compression.short_tf_vol_expansion_pressure"]]
    blend = out[:, idx["vol_compression.atr_percentile_blend"]]
    low_tail = out[:, idx["vol_compression.low_atr_percentile_pressure"]]
    high_tail = out[:, idx["vol_compression.high_atr_percentile_pressure"]]
    assert not np.array_equal(blend, high_tail)
    assert not np.array_equal(blend, low_tail)
    assert np.all(low_tail[blend >= 0.5] == 0.0)
    assert np.all(high_tail[blend <= 0.5] == 0.0)


def test_vol_compression_layer_rejects_unprefixed_aliases() -> None:
    names = _feature_names()
    alias_names = ["atr_z" if name == "snap.atr_z" else name for name in names]

    assert missing_entry_vol_compression_source_fields(alias_names) == ["snap.atr_z"]
    with pytest.raises(RuntimeError, match="vol/compression required source fields missing"):
        build_entry_vol_compression_layer(np.zeros((2, len(alias_names)), dtype=np.float32), alias_names)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_vol_compression_layer_rejects_nonfinite_inputs(bad_value: float) -> None:
    names = _feature_names()
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    x[1, idx["snap.atr_z"]] = bad_value

    with pytest.raises(RuntimeError, match="vol/compression input matrix contains non-finite values"):
        build_entry_vol_compression_layer(x, names)


def test_vol_compression_layer_is_future_row_invariant_and_ignores_targets() -> None:
    names = _feature_names(extra=["target.forward_return_24", "future_mfe_bps", "y_direction", "reward_bps"])
    x = _matrix(names)
    base, base_names = build_entry_vol_compression_layer(x, names)
    source_idx = {name: i for i, name in enumerate(names)}

    for future_start in range(1, x.shape[0]):
        changed = x.copy()
        changed[future_start:, :] = changed[future_start:, :] * -7.0 + 3.0
        changed[:, source_idx["target.forward_return_24"]] = np.linspace(-1_000.0, 1_000.0, x.shape[0])
        changed[:, source_idx["future_mfe_bps"]] = 999_999.0
        changed[:, source_idx["y_direction"]] = np.arange(x.shape[0], dtype=np.float32)
        changed[:, source_idx["reward_bps"]] = -999_999.0
        for ratio_name in (
            "ctx_cont.H1_range_compression_ratio",
            "ctx_cont.M15_range_compression_ratio",
        ):
            changed[:, source_idx[ratio_name]] = x[:, source_idx[ratio_name]]
        mutated, mutated_names = build_entry_vol_compression_layer(changed, names)

        assert mutated_names == base_names
        np.testing.assert_allclose(mutated[:future_start], base[:future_start], rtol=0.0, atol=0.0)

    leakage_only = x.copy()
    leakage_only[:, source_idx["target.forward_return_24"]] = 123_456.0
    leakage_only[:, source_idx["future_mfe_bps"]] = -123_456.0
    leakage_only[:, source_idx["y_direction"]] = np.arange(x.shape[0], dtype=np.float32)
    leakage_only[:, source_idx["reward_bps"]] = 654_321.0
    leakage_out, _ = build_entry_vol_compression_layer(leakage_only, names)
    np.testing.assert_allclose(leakage_out, base, rtol=0.0, atol=0.0)


def test_vol_compression_source_contract_and_specialist_routing() -> None:
    assert missing_entry_vol_compression_source_fields(VOL_COMPRESSION_SOURCE_FIELDS) == []
    missing = missing_entry_vol_compression_source_fields(
        name for name in VOL_COMPRESSION_SOURCE_FIELDS if name != "ctx_cont.M15_range_compression_ratio"
    )
    assert missing == ["ctx_cont.M15_range_compression_ratio"]
    assert {classify_entry_specialist_feature(name) for name in VOL_COMPRESSION_FEATURE_NAMES} == {
        "vol_compression_encoder"
    }
    forbidden = ("future", "target", "y_", "mfe", "mae", "pnl", "reward", "label")
    assert not any(token in name for name in VOL_COMPRESSION_FEATURE_NAMES for token in forbidden)


def test_vol_compression_layer_rejects_empty_rows_width_mismatch_and_duplicate_names() -> None:
    names = list(VOL_COMPRESSION_SOURCE_FIELDS)

    with pytest.raises(RuntimeError, match="at least one row"):
        build_entry_vol_compression_layer(np.empty((0, len(names)), dtype=np.float32), names)
    with pytest.raises(RuntimeError, match="feature name count"):
        build_entry_vol_compression_layer(np.zeros((2, len(names)), dtype=np.float32), names[:-1])

    duplicate_names = [*names, names[-1]]
    with pytest.raises(RuntimeError, match="duplicate feature names"):
        build_entry_vol_compression_layer(
            np.zeros((2, len(duplicate_names)), dtype=np.float32),
            duplicate_names,
        )
