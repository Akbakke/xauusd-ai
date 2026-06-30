import argparse
import json
from pathlib import Path

import numpy as np

from gx1.features.entry_smc_liquidity_quality_v1 import (
    SMC_LIQUIDITY_QUALITY_FEATURE_NAMES,
    SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS,
    build_entry_smc_liquidity_quality_layer,
    missing_smc_liquidity_quality_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.materialize_entry_smc_liquidity_quality_manifest_v1 import run as run_manifest


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        if name in idx:
            x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("snap.smc_sweep_up", [0, 0, 0, 0, 1, 0])
    set_col("snap.smc_sweep_down", [0, 1, 0, 0, 0, 0])
    set_col("snap.smc_sweep_size_atr", [0, 1.6, 0.7, 0, 1.4, 0.4])
    set_col("snap.smc_bars_since_sweep", [96, 0, 1, 8, 0, 1])
    set_col("snap.smc_premium_discount", [0.5, 0.18, 0.25, 0.5, 0.84, 0.78])
    set_col("snap.smc_choch", [0, 0, 1, 0, 0, 1])
    set_col("snap.body_pct", [0, 0.55, 0.25, 0, -0.50, -0.20])
    set_col("snap._v1_body_share_1", [0, 0.60, 0.40, 0, 0.55, 0.35])
    set_col("snap._v1_clv", [0, 0.9, 0.6, 0, -0.8, -0.5])
    set_col("snap.wick_asym", [0, -0.85, -0.25, 0, 0.75, 0.30])
    set_col("ctx_cont.wick_ratio", [0.5, 0.08, 0.20, 0.5, 0.92, 0.80])
    set_col("ctx_cont.smc_sweep_bull_pressure_last12", [0, 1, 0.3, 0, -1, -0.3])
    set_col("ctx_cont.smc_sweep_bull_pressure_last48", [0, 0.8, 0.2, 0, -0.8, -0.2])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0, 0.9, 0.5, 0, 0.8, 0.4])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0, 1, 0.8, 0.2, 1, 0.8])
    set_col("ctx_cont.smc_choch_recent_tau12", [0, 0.1, 0.8, 0, 0.1, 0.8])
    set_col("ctx_cont.smc_choch_recent_tau24", [0, 0.2, 0.7, 0, 0.2, 0.7])
    set_col("ctx_cont.sr_nearest_pivot_abs_atr", [8, 0.2, 0.4, 8, 0.2, 0.4])
    set_col("ctx_cont.sr_support_proximity_exp", [0.1, 1.0, 0.8, 0.1, 0.2, 0.3])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.1, 0.2, 0.3, 0.1, 1.0, 0.8])
    set_col("ctx_cont.sr_support_minus_resistance_prox", [0, 0.8, 0.5, 0, -0.8, -0.5])
    set_col("ctx_cont.liquidity_lo_nearest_abs_atr", [8, 0.1, 0.3, 8, 5, 3])
    set_col("ctx_cont.liquidity_hi_nearest_abs_atr", [8, 5, 3, 8, 0.1, 0.3])
    set_col("ctx_cont.liquidity_lo_minus_hi_prox", [0, 0.7, 0.4, 0, -0.7, -0.4])
    set_col("ctx_cont.dist_to_S1_atr", [8, 0.1, 0.3, 8, 4, 3])
    set_col("ctx_cont.dist_to_S2_atr", [8, 0.2, 0.5, 8, 5, 3])
    set_col("ctx_cont.dist_to_R1_atr", [8, 4, 3, 8, 0.1, 0.3])
    set_col("ctx_cont.dist_to_R2_atr", [8, 5, 3, 8, 0.2, 0.5])
    set_col("ctx_cont.dist_to_m5_lo_atr", [8, 0.1, 0.2, 8, 4, 3])
    set_col("ctx_cont.dist_to_m15_lo_atr", [8, 0.2, 0.3, 8, 4, 3])
    set_col("ctx_cont.dist_to_h1_lo_atr", [8, 0.3, 0.4, 8, 4, 3])
    set_col("ctx_cont.dist_to_h4_lo_atr", [8, 0.4, 0.5, 8, 4, 3])
    set_col("ctx_cont.dist_to_d1_lo_atr", [8, 0.5, 0.6, 8, 4, 3])
    set_col("ctx_cont.dist_to_m5_hi_atr", [8, 4, 3, 8, 0.1, 0.2])
    set_col("ctx_cont.dist_to_m15_hi_atr", [8, 4, 3, 8, 0.2, 0.3])
    set_col("ctx_cont.dist_to_h1_hi_atr", [8, 4, 3, 8, 0.3, 0.4])
    set_col("ctx_cont.dist_to_h4_hi_atr", [8, 4, 3, 8, 0.4, 0.5])
    set_col("ctx_cont.dist_to_d1_hi_atr", [8, 4, 3, 8, 0.5, 0.6])
    set_col("ctx_cont.dist_last_swing_low_atr", [8, 0.2, 0.3, 8, 4, 3])
    set_col("ctx_cont.dist_last_swing_high_atr", [8, 4, 3, 8, 0.2, 0.3])
    set_col("ctx_cont.bars_since_swing_low", [96, 1, 2, 96, 20, 21])
    set_col("ctx_cont.bars_since_swing_high", [96, 20, 21, 96, 1, 2])
    set_col("ctx_cont._v1h1_ema_diff", [0, 1.5, 1.0, 0, -1.5, -1.0])
    set_col("ctx_cont._v1h4_ema_diff", [0, 1.5, 1.0, 0, -1.5, -1.0])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", [0, 1.0, 0.8, 0, -1.0, -0.8])
    set_col("ctx_cont.m15_trend_sign_canon_v2", [0, 1.0, 1.0, 0, -1.0, -1.0])
    set_col("ctx_cont.regime_stack_sum_v3", [0, 2.0, 1.0, 0, -2.0, -1.0])
    set_col("chart.foundation_sweep_low_reclaim_up_proxy", [0, 3.5, 1.5, 0, 0, 0])
    set_col("chart.foundation_sweep_high_reclaim_down_proxy", [0, 0, 0, 0, 3.2, 1.2])
    set_col("chart.foundation_false_breakout_low_followthrough_up_proxy", [0, 2.2, 2.8, 0, 0, 0])
    set_col("chart.foundation_false_breakout_high_followthrough_down_proxy", [0, 0, 0, 0, 2.0, 2.7])
    return x


def test_smc_liquidity_quality_layer_builds_directional_closed_bar_features() -> None:
    names = list(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS) + [
        "chart.foundation_sweep_low_reclaim_up_proxy",
        "chart.foundation_sweep_high_reclaim_down_proxy",
        "chart.foundation_false_breakout_low_followthrough_up_proxy",
        "chart.foundation_false_breakout_high_followthrough_down_proxy",
    ]
    x = _matrix(names)
    out, out_names = build_entry_smc_liquidity_quality_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert tuple(out_names) == SMC_LIQUIDITY_QUALITY_FEATURE_NAMES
    assert out.shape == (6, len(out_names))
    assert np.isfinite(out).all()
    assert out[1, idx["chart.smc_liquidity_sweep_low_quality_long"]] > 0.5
    assert out[4, idx["chart.smc_liquidity_sweep_high_quality_short"]] > 0.5
    assert out[1, idx["chart.smc_liquidity_reclaim_confirmation_long"]] > out[0, idx["chart.smc_liquidity_reclaim_confirmation_long"]]
    assert out[4, idx["chart.smc_liquidity_reclaim_confirmation_short"]] > out[0, idx["chart.smc_liquidity_reclaim_confirmation_short"]]
    assert out[1, idx["chart.smc_liquidity_liquidity_pool_proximity_low"]] > out[0, idx["chart.smc_liquidity_liquidity_pool_proximity_low"]]
    assert out[4, idx["chart.smc_liquidity_liquidity_pool_proximity_high"]] > out[0, idx["chart.smc_liquidity_liquidity_pool_proximity_high"]]
    assert out[1, idx["chart.smc_liquidity_wick_rejection_strength_long"]] > out[4, idx["chart.smc_liquidity_wick_rejection_strength_long"]]
    assert out[4, idx["chart.smc_liquidity_wick_rejection_strength_short"]] > out[1, idx["chart.smc_liquidity_wick_rejection_strength_short"]]


def test_smc_liquidity_quality_layer_is_future_row_invariant() -> None:
    names = list(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS)
    x = _matrix(names)
    changed_future = x.copy()
    changed_future[4:, :] = changed_future[4:, :] * -7.0 + 3.0

    out, names_a = build_entry_smc_liquidity_quality_layer(x, names)
    changed, names_b = build_entry_smc_liquidity_quality_layer(changed_future, names)

    assert names_a == names_b
    np.testing.assert_allclose(out[:4], changed[:4], rtol=0.0, atol=1e-7)


def test_smc_liquidity_quality_contract_routes_to_smc_specialist() -> None:
    assert len(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS) == len(set(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS))
    assert missing_smc_liquidity_quality_source_fields(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS) == []
    assert missing_smc_liquidity_quality_source_fields(["snap.smc_sweep_up"])[:1] == ["snap.smc_sweep_down"]
    assert {classify_entry_specialist_feature(name) for name in SMC_LIQUIDITY_QUALITY_FEATURE_NAMES} == {
        "smc_liquidity_encoder"
    }


def test_smc_liquidity_quality_manifest_is_report_only(tmp_path: Path) -> None:
    report = run_manifest(argparse.Namespace(out_dir=str(tmp_path)))

    assert report["decision"] == "READY_FOR_FEATURE_AUDIT_REVIEW"
    assert report["training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert report["side_effects_started"] is False
    assert report["candidate_feature_count"] == len(SMC_LIQUIDITY_QUALITY_FEATURE_NAMES)
    assert set(report["feature_counts_by_specialist"]) == {"smc_liquidity_encoder"}
    assert (tmp_path / "ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_latest.json").exists()
    latest = json.loads((tmp_path / "ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_latest.json").read_text(encoding="utf-8"))
    assert latest["selected_features"] == list(SMC_LIQUIDITY_QUALITY_FEATURE_NAMES)
