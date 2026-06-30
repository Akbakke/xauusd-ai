import argparse
import json

import numpy as np

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_trend_ema_v1 import (
    TREND_EMA_FEATURE_NAMES,
    TREND_EMA_SOURCE_FIELDS,
    build_entry_trend_ema_layer,
    missing_trend_ema_source_fields,
)
from gx1.scripts.materialize_entry_trend_ema_extension_manifest_v1 import run as run_manifest


EXPECTED_TREND_EMA_FEATURE_NAMES = (
    "trend.ema_mtf_score",
    "trend.ema_stack_alignment_score",
    "trend.ema_stack_bull_pressure",
    "trend.ema_stack_bear_pressure",
    "trend.ema_inflect_up_pressure",
    "trend.ema_inflect_down_pressure",
    "trend.ema_slope_score",
    "trend.ema_slope_curvature",
    "trend.ema_slope_curvature_abs",
    "trend.ema_mtf_agreement_pressure",
    "trend.ema_mtf_divergence_pressure",
    "trend.ema_distance_fast_abs",
    "trend.ema_distance_stack_abs",
    "trend.ema_distance_stretch_pressure",
    "trend.ema_retrace_to_fast_long_pressure",
    "trend.ema_retrace_to_fast_short_pressure",
    "trend.ema_trend_age_mean",
    "trend.ema_trend_age_exhaustion_pressure",
    "trend.ema_late_reversal_risk",
    "trend.ema_d1_flip_pressure",
)


def _matrix(names: list[str], n: int = 7) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    down_to_up = [-2.0, -1.0, -0.2, 0.5, 1.4, 1.8, 2.0]
    up_to_down = [2.0, 1.2, 0.2, -0.4, -1.2, -1.8, -2.0]
    for name in (
        "snap._v1_ema_diff",
        "snap.ema20_slope",
        "snap.pos_vs_ema200",
        "snap._v1_close_ema_slope_3",
        "snap._v1_kama_slope_30",
        "snap._v1_tema_slope_20",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h1_slope3",
        "ctx_cont._v1h1_slope5",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont._v1h4_slope3",
        "ctx_cont._v1h4_slope5",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.d1_pct_change_5_canon_v2",
        "ctx_cont.m15_trend_sign_canon_v2",
    ):
        set_col(name, down_to_up)
    set_col("ctx_cont.distance_ema_fast", [3.0, 1.0, 0.2, 0.05, -0.05, -0.2, -1.0])
    set_col("ctx_cont.m5_trend_age_bars_norm_v2", [0.05, 0.10, 0.20, 0.40, 0.70, 0.90, 1.00])
    set_col("ctx_cont.m15_trend_age_bars_norm_v2", [0.05, 0.10, 0.25, 0.40, 0.70, 0.90, 1.00])
    set_col("ctx_cont.h1_trend_age_bars_norm_v2", [0.05, 0.15, 0.25, 0.45, 0.75, 0.95, 1.00])
    set_col("ctx_cont.h4_trend_age_bars_norm_v2", [0.05, 0.10, 0.20, 0.40, 0.70, 0.90, 1.00])
    set_col("ctx_cont.d1_trend_age_bars_norm_v2", [0.10, 0.15, 0.25, 0.40, 0.70, 0.90, 1.00])
    set_col("ctx_cont.regime_tf_agreement_v3", [0.2, 0.4, 0.6, 0.9, 1.0, 1.0, 1.0])
    set_col("ctx_cont.regime_stack_sum_v3", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 3.0])
    set_col("ctx_cont.regime_divergence_flag_v3", [0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.8])
    set_col("ctx_cont.d1_dist_roc_288_v3", [0.0, 0.0, 0.1, 0.2, 0.5, 1.0, 1.0])
    set_col("ctx_cont.d1_trend_age_mature_flag_v3", [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    set_col("ctx_cont.retracement_from_last_impulse", [0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 1.0])

    assert up_to_down
    return x


def _coherence_case(names: list[str], *, slope_sign: float) -> np.ndarray:
    x = np.zeros((5, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, value: float) -> None:
        x[:, idx[name]] = np.full(x.shape[0], value, dtype=np.float32)

    for name in (
        "snap._v1_ema_diff",
        "snap.pos_vs_ema200",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont.m15_trend_sign_canon_v2",
    ):
        set_col(name, 1.25)
    for name in (
        "snap.ema20_slope",
        "snap._v1_close_ema_slope_3",
        "snap._v1_kama_slope_30",
        "snap._v1_tema_slope_20",
        "ctx_cont._v1h1_slope3",
        "ctx_cont._v1h1_slope5",
        "ctx_cont._v1h4_slope3",
        "ctx_cont._v1h4_slope5",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.d1_pct_change_5_canon_v2",
    ):
        set_col(name, slope_sign * 1.25)
    set_col("ctx_cont.distance_ema_fast", 0.20)
    set_col("ctx_cont.regime_tf_agreement_v3", 1.00)
    set_col("ctx_cont.regime_stack_sum_v3", 3.00)
    set_col("ctx_cont.regime_divergence_flag_v3", 0.00)
    set_col("ctx_cont.d1_dist_roc_288_v3", 0.10)
    set_col("ctx_cont.d1_trend_age_mature_flag_v3", 0.20)
    set_col("ctx_cont.retracement_from_last_impulse", 0.50)
    for name in (
        "ctx_cont.m5_trend_age_bars_norm_v2",
        "ctx_cont.m15_trend_age_bars_norm_v2",
        "ctx_cont.h1_trend_age_bars_norm_v2",
        "ctx_cont.h4_trend_age_bars_norm_v2",
        "ctx_cont.d1_trend_age_bars_norm_v2",
    ):
        set_col(name, 0.40)
    return x


def test_trend_ema_feature_contract_is_stable() -> None:
    assert len(TREND_EMA_FEATURE_NAMES) == 20
    assert TREND_EMA_FEATURE_NAMES == EXPECTED_TREND_EMA_FEATURE_NAMES


def test_trend_ema_layer_builds_causal_specialist_features() -> None:
    names = list(TREND_EMA_SOURCE_FIELDS)
    out, out_names = build_entry_trend_ema_layer(_matrix(names), names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert out.shape == (7, len(TREND_EMA_FEATURE_NAMES))
    assert tuple(out_names) == TREND_EMA_FEATURE_NAMES
    assert np.isfinite(out).all()
    assert out[4, idx["trend.ema_stack_bull_pressure"]] > out[1, idx["trend.ema_stack_bull_pressure"]]
    assert out[1, idx["trend.ema_stack_bear_pressure"]] > out[4, idx["trend.ema_stack_bear_pressure"]]
    assert out[3, idx["trend.ema_inflect_up_pressure"]] > 0.0
    assert out[5, idx["trend.ema_trend_age_exhaustion_pressure"]] > out[2, idx["trend.ema_trend_age_exhaustion_pressure"]]
    assert out[4, idx["trend.ema_retrace_to_fast_long_pressure"]] > 0.0
    assert out[6, idx["trend.ema_late_reversal_risk"]] > out[2, idx["trend.ema_late_reversal_risk"]]


def test_trend_ema_layer_sanitizes_nonfinite_inputs() -> None:
    names = list(TREND_EMA_SOURCE_FIELDS)
    x = _matrix(names)
    source_idx = {name: i for i, name in enumerate(names)}
    x[1, source_idx["snap._v1_ema_diff"]] = np.nan
    x[2, source_idx["snap.ema20_slope"]] = np.inf
    x[3, source_idx["ctx_cont._v1h4_slope5"]] = -np.inf

    out, out_names = build_entry_trend_ema_layer(x, names)

    assert tuple(out_names) == EXPECTED_TREND_EMA_FEATURE_NAMES
    assert out.shape == (7, 20)
    assert np.isfinite(out).all()


def test_trend_ema_layer_rewards_ema_slope_coherence() -> None:
    names = list(TREND_EMA_SOURCE_FIELDS)
    coherent, out_names = build_entry_trend_ema_layer(_coherence_case(names, slope_sign=1.0), names)
    divergent, _ = build_entry_trend_ema_layer(_coherence_case(names, slope_sign=-1.0), names)
    idx = {name: i for i, name in enumerate(out_names)}

    row = 3
    assert coherent[row, idx["trend.ema_mtf_agreement_pressure"]] > divergent[row, idx["trend.ema_mtf_agreement_pressure"]]
    assert divergent[row, idx["trend.ema_mtf_divergence_pressure"]] > coherent[row, idx["trend.ema_mtf_divergence_pressure"]]
    assert coherent[row, idx["trend.ema_stack_alignment_score"]] > divergent[row, idx["trend.ema_stack_alignment_score"]]
    assert coherent[row, idx["trend.ema_stack_bull_pressure"]] > divergent[row, idx["trend.ema_stack_bull_pressure"]]


def test_trend_ema_source_contract_and_specialist_routing() -> None:
    assert missing_trend_ema_source_fields(TREND_EMA_SOURCE_FIELDS) == []
    missing = missing_trend_ema_source_fields(
        name for name in TREND_EMA_SOURCE_FIELDS if name != "ctx_cont._v1h4_slope5"
    )
    assert missing == ["ctx_cont._v1h4_slope5"]
    assert {classify_entry_specialist_feature(name) for name in TREND_EMA_FEATURE_NAMES} == {"trend_ema_encoder"}


def test_trend_ema_layer_does_not_read_future_rows() -> None:
    names = list(TREND_EMA_SOURCE_FIELDS)
    x = _matrix(names)
    baseline, _ = build_entry_trend_ema_layer(x, names)

    for future_start in range(1, x.shape[0]):
        changed = x.copy()
        changed[future_start:, :] = 99.0
        mutated, _ = build_entry_trend_ema_layer(changed, names)

        np.testing.assert_allclose(mutated[:future_start], baseline[:future_start], rtol=0.0, atol=0.0)


def test_trend_ema_manifest_is_report_only(tmp_path) -> None:
    report = run_manifest(
        argparse.Namespace(
            out_dir=str(tmp_path / "out"),
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_TREND_EMA_EXTENSION_REVIEW"
    assert report["manifest"]["selected_feature_count"] == len(TREND_EMA_FEATURE_NAMES)
    assert report["manifest"]["all_features_route_to_expected_specialist"] is True
    assert report["training_allowed"] is False
    assert report["shadow_live_promotion_allowed"] is False
    assert report["side_effects_started"] == {
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
        "promotion": False,
    }
    latest = tmp_path / "out" / "ENTRY_TREND_EMA_EXTENSION_REPORT_latest.json"
    assert json.loads(latest.read_text(encoding="utf-8"))["decision"] == report["decision"]
