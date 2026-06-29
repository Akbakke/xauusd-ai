import argparse
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    build_metric_rows,
    build_summary,
    run,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val"] * 6 + ["test"] * 6,
            "model": ["candidate"] * 12,
            "time": pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC"),
            "y_direction": [0, 1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 1],
            "trade_side": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "side": ["LONG", "SHORT", "LONG", "LONG", "SHORT", "SHORT"] * 2,
            "session": ["EU", "EU", "US", "ASIA", "US", "OVERLAP"] * 2,
            "vol_regime": ["1", "1", "2", "0", "2", "1"] * 2,
            "edge_score": [0.90, 0.80, 0.70, 0.10, 0.60, 0.20, 0.95, 0.85, 0.05, 0.75, 0.65, 0.15],
            "pnl_proxy_bps": [12.0, 10.0, 8.0, 0.0, 6.0, -5.0, 14.0, 11.0, 0.0, 7.0, 5.0, 3.0],
            "bad_path_prob": [0.1] * 12,
            "path_quality_pred": [1.0] * 12,
        }
    )


def test_metric_rows_include_required_replay_readiness_columns_and_session_slices() -> None:
    rows = build_metric_rows(_predictions(), top_fracs=[0.5])
    metrics = pd.DataFrame(rows)

    required = {"split", "model", "scope", "top_frac", "group", "n", "mean_pnl_bps", "win_rate", "direction_precision"}
    assert required.issubset(metrics.columns)
    assert "session=EU" in set(metrics["group"])
    assert "session=US" in set(metrics["group"])
    all_val = metrics[(metrics["split"] == "val") & (metrics["group"] == "ALL")].iloc[0]
    assert all_val["mean_pnl_bps"] > 0.0


def test_summary_uses_top5_and_top10_all_metrics() -> None:
    metrics = pd.DataFrame(build_metric_rows(_predictions(), top_fracs=[0.05, 0.10]))
    summary = build_summary(_predictions(), metrics)

    assert summary["splits"] == ["test", "val"]
    rows = {(row["split"], row["model"]): row for row in summary["summaries"]}
    assert rows[("val", "candidate")]["top5_all_mean_pnl_bps"] == 12.0
    assert rows[("test", "candidate")]["top10_all_mean_pnl_bps"] == 14.0


def test_selective_edge_requires_bundle_dir(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ENTRY_V10_BUNDLE_MISSING"):
        run(
            argparse.Namespace(
                bundle_dir=str(tmp_path / "missing"),
                no_xgb_bundle_dir="",
                dataset_dir="/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke",
                splits="val",
                top_fracs="0.05,0.10",
                model_name="candidate",
                device="cpu",
                batch_size=16,
                m5_prebuilt_path="/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet",
                multi_tf_cache_dir="",
                out_dir=str(tmp_path),
                require_no_xgb_ablation=True,
                fail_on_audit_fail=False,
                quiet=True,
            )
        )
