import argparse
from pathlib import Path

import numpy as np

from gx1.features.entry_momentum_flow_v1 import (
    MOMENTUM_FLOW_FEATURE_NAMES,
    MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS,
    build_entry_momentum_flow_layer,
    missing_entry_momentum_flow_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.materialize_entry_momentum_flow_challenger_manifest_v1 import run


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("ret_1", [2.0, 5.0, 8.0, 1.0, -4.0, -7.0])
    set_col("ret_5", [4.0, 12.0, 20.0, 10.0, -12.0, -20.0])
    set_col("ret_20", [8.0, 18.0, 35.0, 40.0, -18.0, -35.0])
    set_col("_v1_clv", [0.2, 1.0, 1.3, -1.5, -1.0, -1.3])
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
    return x


def test_momentum_flow_layer_builds_causal_derivatives() -> None:
    names = list(
        dict.fromkeys(
            [
                *MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS,
                "ctx_cont.micro_momentum_3",
                "ctx_cont.micro_momentum_5",
                "ctx_cont.micro_acceleration",
                "ctx_cont.atr_bps",
                "snap.atr_z",
                "snap._v1_range_z",
                "ctx_cont._v1h1_slope5",
                "ctx_cont._v1h4_slope5",
                "ctx_cont.d1_pct_change_5_canon_v2",
                "ctx_cont.d1_ema_slope_20_canon_v2",
                "ctx_cont.regime_tf_agreement_v3",
                "ctx_cont.regime_divergence_flag_v3",
                "ctx_cont.dip_confirmed_m5_v3",
                "ctx_cont.dip_confirmed_m15_v3",
                "ctx_cont.dip_confirmed_h1_v3",
                "ctx_cont.dip_confirmed_h4_v3",
                "ctx_cont.dip_confirmed_d1_v3",
                "ctx_cont.dip_confirmed_mean_5tf",
                "ctx_cont.dip_confirmed_max_5tf",
                "ctx_cont.dip_proximity_h1_v3",
                "ctx_cont.dip_proximity_h4_v3",
                "ctx_cont.dip_proximity_d1_v3",
                "ctx_cont.dip_proximity_mean_h1h4d1",
                "chart.foundation_impulse_direction",
                "chart.foundation_impulse_pullback_alignment",
                "chart.foundation_compression_release_up",
                "chart.foundation_compression_release_down",
                "candle.pattern_bull_continuation_pressure",
                "candle.pattern_bear_continuation_pressure",
                "candle.pattern_bull_reversal_pressure",
                "candle.pattern_bear_reversal_pressure",
            ]
        )
    )
    x = _matrix(names)
    out, out_names = build_entry_momentum_flow_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert tuple(out_names) == MOMENTUM_FLOW_FEATURE_NAMES
    assert out.shape == (6, len(MOMENTUM_FLOW_FEATURE_NAMES))
    assert np.isfinite(out).all()
    assert out[2, idx["momentum.flow_bull_followthrough_score"]] > out[2, idx["momentum.flow_bear_followthrough_score"]]
    assert out[5, idx["momentum.flow_bear_followthrough_score"]] > out[5, idx["momentum.flow_bull_followthrough_score"]]
    assert out[2, idx["momentum.flow_dip_continuation_long_input"]] > out[2, idx["momentum.flow_dip_reversal_long_risk_input"]]
    assert out[3, idx["momentum.flow_bull_exhaustion_pressure"]] > 0.0
    assert out[5, idx["momentum.flow_compression_release_followthrough_score"]] < 0.0


def test_momentum_flow_source_contract_and_specialist_routing() -> None:
    names = [
        "snap.ret_1",
        "snap.ret_5",
        "snap.ret_20",
        "snap._v1_clv",
        "ctx_cont.micro_momentum_3",
        "ctx_cont.micro_momentum_5",
        "ctx_cont.micro_acceleration",
    ]

    assert missing_entry_momentum_flow_source_fields(names) == []
    assert missing_entry_momentum_flow_source_fields(names[:-1]) == ["micro_acceleration"]
    assert classify_entry_specialist_feature("momentum.flow_clean_edge_score") == "momentum_flow_encoder"


def test_momentum_flow_manifest_is_report_only(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            out_dir=str(tmp_path),
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_MOMENTUM_FLOW_CHALLENGER_REVIEW"
    assert report["side_effects_started"] == {
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
        "promotion": False,
    }
    assert report["manifest"]["selected_feature_count"] == len(MOMENTUM_FLOW_FEATURE_NAMES)
    assert all(row["specialist"] == "momentum_flow_encoder" for row in report["manifest"]["feature_rows"])
