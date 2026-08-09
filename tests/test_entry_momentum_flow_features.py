import numpy as np
import pytest

from gx1.features.entry_momentum_flow_v1 import (
    MOMENTUM_FLOW_FEATURE_NAMES,
    MOMENTUM_FLOW_SOURCE_FIELDS,
    build_entry_momentum_flow_layer,
    missing_entry_momentum_flow_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


def _feature_names(extra: list[str] | None = None) -> list[str]:
    return list(dict.fromkeys([*MOMENTUM_FLOW_SOURCE_FIELDS, *(extra or [])]))


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        if name not in idx:
            return
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("snap.ret_1", [2.0, 5.0, 8.0, 1.0, -4.0, -7.0])
    set_col("snap.ret_5", [4.0, 12.0, 20.0, 10.0, -12.0, -20.0])
    set_col("snap.ret_20", [8.0, 18.0, 35.0, 40.0, -18.0, -35.0])
    # Producer domain (basic_v1): [0, 1] with neutral 0.5; bull rows close near
    # the bar high, bear rows near the bar low.
    set_col("snap._v1_clv", [0.6, 1.0, 1.0, 0.0, 0.0, 0.0])
    set_col("ctx_cont.micro_momentum_3", [0.2, 0.8, 1.0, 0.2, -0.8, -1.0])
    set_col("ctx_cont.micro_momentum_5", [0.3, 1.0, 1.3, 0.5, -1.0, -1.3])
    set_col("ctx_cont.micro_acceleration", [0.1, 0.4, 0.5, -0.5, -0.4, -0.5])
    set_col("ctx_cont.atr_bps", [4.0, 4.0, 5.0, 5.0, 4.0, 5.0])
    set_col("snap.atr_z", [0.1, 0.4, 0.9, 1.5, 0.4, 0.9])
    set_col("snap._v1_range_z", [0.1, 0.4, 0.9, 1.5, 0.4, 0.9])
    set_col("ctx_cont._v1h1_slope5", [1.0, 1.0, 1.0, -0.2, -1.0, -1.0])
    set_col("ctx_cont._v1h4_slope5", [1.0, 1.0, 1.0, -0.2, -1.0, -1.0])
    set_col("ctx_cont.d1_pct_change_5_canon_v2", [5.0, 8.0, 10.0, -2.0, -8.0, -10.0])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", [0.8, 1.0, 1.0, -0.2, -1.0, -1.0])
    set_col("ctx_cont.regime_tf_agreement_v3", [0.6, 0.8, 1.0, 0.2, 0.8, 1.0])
    set_col("ctx_cont.regime_divergence_flag_v3", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    for name in (
        "ctx_cont.dip_confirmed_m5_v3",
        "ctx_cont.dip_confirmed_m15_v3",
        "ctx_cont.dip_confirmed_h1_v3",
        "ctx_cont.dip_confirmed_h4_v3",
        "ctx_cont.dip_confirmed_d1_v3",
        "ctx_cont.dip_confirmed_mean_5tf",
        "ctx_cont.dip_confirmed_max_5tf",
    ):
        set_col(name, [0.0, 0.4, 0.8, 0.9, 0.4, 0.8])
    for name in (
        "ctx_cont.dip_proximity_h1_v3",
        "ctx_cont.dip_proximity_h4_v3",
        "ctx_cont.dip_proximity_d1_v3",
        "ctx_cont.dip_proximity_mean_h1h4d1",
    ):
        set_col(name, [0.0, 0.3, 0.7, 0.9, 0.3, 0.7])
    set_col("chart.foundation_impulse_direction", [0.0, 1.0, 1.0, 0.5, -1.0, -1.0])
    set_col("chart.foundation_impulse_pullback_alignment", [0.0, 0.8, 1.0, -0.5, 0.8, 1.0])
    set_col("chart.foundation_compression_release_up", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    set_col("chart.foundation_compression_release_down", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    set_col("candle.pattern_bull_continuation_pressure", [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    set_col("candle.pattern_bear_continuation_pressure", [0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    set_col("candle.pattern_bull_reversal_pressure", [0.0, 0.1, 0.1, 0.0, 0.2, 0.3])
    set_col("candle.pattern_bear_reversal_pressure", [0.0, 0.0, 0.2, 0.8, 0.1, 0.1])
    set_col("candle.pattern_body_share", [0.2, 0.7, 0.9, 0.8, 0.7, 0.9])
    set_col("candle.pattern_upper_wick_share", [0.0, 0.1, 0.1, 0.9, 0.5, 0.9])
    set_col("candle.pattern_lower_wick_share", [0.0, 0.6, 0.9, 0.1, 0.1, 0.1])
    return x


def test_momentum_flow_layer_builds_causal_derivatives() -> None:
    names = _feature_names()
    x = _matrix(names)
    out, out_names = build_entry_momentum_flow_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}
    neutral_wick = x.copy()
    source_idx = {name: i for i, name in enumerate(names)}
    neutral_wick[:, source_idx["candle.pattern_upper_wick_share"]] = 0.0
    neutral_wick[:, source_idx["candle.pattern_lower_wick_share"]] = 0.0
    out_neutral_wick, _ = build_entry_momentum_flow_layer(neutral_wick, names)

    assert tuple(out_names) == MOMENTUM_FLOW_FEATURE_NAMES
    assert len(MOMENTUM_FLOW_FEATURE_NAMES) == 26
    assert out.shape == (6, len(MOMENTUM_FLOW_FEATURE_NAMES))
    assert np.isfinite(out).all()
    assert out[2, idx["momentum.flow_bull_followthrough_score"]] > out[
        2, idx["momentum.flow_bear_followthrough_score"]
    ]
    assert out[5, idx["momentum.flow_bear_followthrough_score"]] > out[
        5, idx["momentum.flow_bull_followthrough_score"]
    ]
    assert out[2, idx["momentum.flow_dip_continuation_long_input"]] > out[
        2, idx["momentum.flow_bad_path_long_initial_pressure"]
    ]
    assert out[3, idx["momentum.flow_bad_path_long_initial_pressure"]] > out[
        2, idx["momentum.flow_bad_path_long_initial_pressure"]
    ]
    assert out[3, idx["momentum.flow_flow_divergence_pressure"]] > out[
        2, idx["momentum.flow_flow_divergence_pressure"]
    ]
    assert out[3, idx["momentum.flow_bull_exhaustion_pressure"]] > 0.0
    assert out[5, idx["momentum.flow_compression_release_followthrough_score"]] < 0.0
    assert out[2, idx["momentum.flow_bodyflow_bull_pressure"]] > out_neutral_wick[
        2, idx["momentum.flow_bodyflow_bull_pressure"]
    ]
    assert out[3, idx["momentum.flow_bodyflow_bear_pressure"]] > out_neutral_wick[
        3, idx["momentum.flow_bodyflow_bear_pressure"]
    ]


def test_momentum_flow_layer_rejects_nonfinite_and_is_row_causal() -> None:
    names = _feature_names(extra=["target.forward_return_24", "y_direction", "future_mfe_bps"])
    x = _matrix(names)
    out, out_names = build_entry_momentum_flow_layer(x, names)
    source_idx = {name: i for i, name in enumerate(names)}

    bad = x.copy()
    bad[1, source_idx["snap.ret_1"]] = np.nan
    bad[2, source_idx["snap.ret_5"]] = np.inf
    bad[3, source_idx["ctx_cont.micro_momentum_3"]] = -np.inf
    with pytest.raises(RuntimeError, match="momentum flow input matrix contains non-finite values"):
        build_entry_momentum_flow_layer(bad, names)

    future_changed = x.copy()
    future_changed[3:, :] = future_changed[3:, :] * -7.0 + 3.0
    future_changed[:, source_idx["target.forward_return_24"]] = np.linspace(-1_000.0, 1_000.0, x.shape[0])
    future_out, _ = build_entry_momentum_flow_layer(future_changed, names)
    np.testing.assert_allclose(out[:3], future_out[:3], rtol=0.0, atol=0.0)

    leakage_only = x.copy()
    leakage_only[:, source_idx["target.forward_return_24"]] = 999_999.0
    leakage_only[:, source_idx["y_direction"]] = np.arange(x.shape[0], dtype=np.float32)
    leakage_only[:, source_idx["future_mfe_bps"]] = -999_999.0
    leakage_out, _ = build_entry_momentum_flow_layer(leakage_only, names)
    np.testing.assert_allclose(out, leakage_out, rtol=0.0, atol=0.0)


def test_momentum_flow_source_contract_and_specialist_routing() -> None:
    names = list(MOMENTUM_FLOW_SOURCE_FIELDS)

    assert missing_entry_momentum_flow_source_fields(names) == []
    assert missing_entry_momentum_flow_source_fields(names[:-1]) == ["candle.pattern_lower_wick_share"]
    with pytest.raises(RuntimeError, match="momentum flow required source fields missing"):
        build_entry_momentum_flow_layer(np.zeros((2, len(names) - 1), dtype=np.float32), names[:-1])

    alias_names = ["ret_1" if name == "snap.ret_1" else name for name in names]
    assert missing_entry_momentum_flow_source_fields(alias_names) == ["snap.ret_1"]
    with pytest.raises(RuntimeError, match="momentum flow required source fields missing"):
        build_entry_momentum_flow_layer(np.zeros((2, len(alias_names)), dtype=np.float32), alias_names)

    assert classify_entry_specialist_feature("momentum.flow_clean_edge_score") == "momentum_flow_encoder"
    assert len(MOMENTUM_FLOW_FEATURE_NAMES) == 26
    assert len(set(MOMENTUM_FLOW_FEATURE_NAMES)) == 26
    forbidden = ("future", "target", "y_", "mfe", "mae", "pnl", "reward", "label")
    assert not any(token in name for name in MOMENTUM_FLOW_FEATURE_NAMES for token in forbidden)


def test_momentum_flow_layer_rejects_empty_rows_width_mismatch_and_duplicate_names() -> None:
    names = list(MOMENTUM_FLOW_SOURCE_FIELDS)

    with pytest.raises(RuntimeError, match="at least one row"):
        build_entry_momentum_flow_layer(np.empty((0, len(names)), dtype=np.float32), names)
    with pytest.raises(RuntimeError, match="feature name count"):
        build_entry_momentum_flow_layer(np.zeros((2, len(names)), dtype=np.float32), names[:-1])

    duplicate_names = [*names, names[-1]]
    with pytest.raises(RuntimeError, match="duplicate feature names"):
        build_entry_momentum_flow_layer(
            np.zeros((2, len(duplicate_names)), dtype=np.float32),
            duplicate_names,
        )
