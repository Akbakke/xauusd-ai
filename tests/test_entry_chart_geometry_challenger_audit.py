import argparse
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_SOURCE_FIELDS,
)
from gx1.scripts.audit_entry_chart_geometry_challenger_v1 import run


def _source_names() -> tuple[list[str], list[str]]:
    snap = [field.split(".", 1)[1] for field in CHART_GEOMETRY_SOURCE_FIELDS if field.startswith("snap.")]
    ctx = [field.split(".", 1)[1] for field in CHART_GEOMETRY_SOURCE_FIELDS if field.startswith("ctx_cont.")]
    return snap, ctx


def _rows(n: int, split_offset: int) -> pd.DataFrame:
    snap_names, ctx_names = _source_names()
    snap = np.zeros((n, len(snap_names)), dtype=np.float32)
    ctx = np.zeros((n, len(ctx_names)), dtype=np.float32)
    snap_idx = {name: i for i, name in enumerate(snap_names)}
    ctx_idx = {name: i for i, name in enumerate(ctx_names)}

    t = np.linspace(-2.0, 2.0, n, dtype=np.float32) + float(split_offset) * 0.1
    alt = np.sin(np.linspace(0.0, 4.0 * np.pi, n, dtype=np.float32) + split_offset)
    ramp01 = np.linspace(0.05, 0.95, n, dtype=np.float32)

    for name in snap_names:
        snap[:, snap_idx[name]] = alt
    for name in ctx_names:
        ctx[:, ctx_idx[name]] = t

    for name in ("_v1_ema_diff", "ema20_slope", "pos_vs_ema200"):
        snap[:, snap_idx[name]] = t
    for name in ("_v1h1_ema_diff", "_v1h1_slope5", "_v1h4_ema_diff", "_v1h4_slope5", "d1_ema_slope_20_canon_v2"):
        ctx[:, ctx_idx[name]] = t
    ctx[:, ctx_idx["m15_trend_sign_canon_v2"]] = np.sign(t)
    ctx[:, ctx_idx["regime_stack_sum_v3"]] = t
    ctx[:, ctx_idx["regime_tf_agreement_v3"]] = ramp01
    ctx[:, ctx_idx["regime_divergence_flag_v3"]] = (alt > 0.75).astype(np.float32)

    for name in (
        "dist_last_swing_high_atr",
        "dist_last_swing_low_atr",
        "dist_to_R1_atr",
        "dist_to_R2_atr",
        "dist_to_S1_atr",
        "dist_to_S2_atr",
        "dist_to_h1_hi_atr",
        "dist_to_h1_lo_atr",
        "dist_to_h4_hi_atr",
        "dist_to_h4_lo_atr",
        "dist_to_d1_hi_atr",
        "dist_to_d1_lo_atr",
        "sr_nearest_pivot_abs_atr",
    ):
        ctx[:, ctx_idx[name]] = np.abs(alt) + 0.1
    ctx[:, ctx_idx["bars_since_swing_high"]] = np.arange(n, dtype=np.float32) % 17
    ctx[:, ctx_idx["bars_since_swing_low"]] = np.arange(n, dtype=np.float32)[::-1] % 19
    ctx[:, ctx_idx["sr_support_proximity_exp"]] = ramp01
    ctx[:, ctx_idx["sr_resistance_proximity_exp"]] = ramp01[::-1]
    ctx[:, ctx_idx["sr_support_minus_resistance_prox"]] = t

    snap[:, snap_idx["smc_premium_discount"]] = ramp01
    ctx[:, ctx_idx["retracement_from_last_impulse"]] = ramp01
    ctx[:, ctx_idx["d1_close_pct_in_20day_range_canon_v2"]] = ramp01
    for name in ("smc_bos_up", "smc_bos_down", "smc_choch", "smc_sweep_up", "smc_sweep_down"):
        snap[:, snap_idx[name]] = (alt > 0.0).astype(np.float32)
    for name in (
        "smc_bos_pressure_last12",
        "smc_bos_pressure_last48",
        "smc_choch_recent_tau12",
        "smc_choch_recent_tau24",
        "smc_sweep_bull_pressure_last12",
        "smc_sweep_bull_pressure_last48",
        "smc_sweep_size_recent_tau12",
        "smc_sweep_recency_tau24",
        "wick_ratio",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "h1_trend_age_bars_norm_v2",
        "h4_trend_age_bars_norm_v2",
        "D1_atr_percentile_252",
    ):
        ctx[:, ctx_idx[name]] = ramp01
    snap[:, snap_idx["smc_sweep_size_atr"]] = ramp01
    snap[:, snap_idx["wick_asym"]] = alt
    snap[:, snap_idx["_v1_bb_squeeze_20_2"]] = ramp01
    snap[:, snap_idx["atr_z"]] = t
    snap[:, snap_idx["rvol_20"]] = ramp01
    snap[:, snap_idx["vol_ratio_5_20"]] = ramp01[::-1]

    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "snap": [row.tolist() for row in snap],
            "ctx_cont": [row.tolist() for row in ctx],
            "y_direction": np.arange(n, dtype=np.int64) % 3,
        }
    )


def test_chart_geometry_challenger_audit_writes_manifest(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    snap_names, ctx_names = _source_names()
    for offset, split in enumerate(("train", "val", "test")):
        path = dataset_dir / f"toy_{split}.parquet"
        pq.write_table(pa.Table.from_pandas(_rows(48, offset), preserve_index=False), path)
        manifest = {
            "extra": {
                "signal_bridge": {
                    "fields": snap_names,
                    "seq_input_dim": len(snap_names),
                    "seq_structure_extension_dim": 0,
                    "neutral_xgb_bridge": True,
                },
                "ctx_contract": {
                    "ctx_cont_names": ctx_names,
                    "ctx_cont_dim": len(ctx_names),
                },
            }
        }
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = run(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            out_dir=str(tmp_path / "out"),
            data_splits="train,val,test",
            batch_size=16,
            max_rows_per_split=0,
            liveness_epsilon=1e-7,
            near_constant_std=1e-12,
            min_generated_active_rate=0.0,
            min_generated_active_count=0,
            min_source_active_rate=0.0,
            min_source_active_count=0,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_CHALLENGER_DATASET_REBUILD"
    assert report["generated_feature_count"] == len(CHART_GEOMETRY_FEATURE_NAMES)
    assert report["challenger_manifest"]["activation_or_training_allowed"] is False
    assert report["challenger_manifest"]["dataset_rebuild_required_before_training"] is True
    assert report["trainable_in_current_contract"] is False
