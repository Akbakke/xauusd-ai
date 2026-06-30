import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch

from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    SIGNAL_BRIDGE_NEUTRAL_VALUES,
    _neutralize_signal_bridge,
    _specialist_contract_snapshot,
    build_parser,
    build_metric_rows,
    build_no_xgb_ablation_diagnostics,
    build_summary,
    run,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
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


def _bundle_meta_for_contract(mode: str) -> dict:
    dim = 215 if mode == "challenger_seq215" else 146
    specialists = required_training_specialists_for_mode(mode)
    return {
        "seq_input_dim": dim,
        "snap_input_dim": dim,
        "specialist_fusion": {
            "enabled": True,
            "contract_mode": mode,
            "input_indices": {name: [idx] for idx, name in enumerate(specialists)},
            "specialist_model_contract": specialist_model_contract_for_mode(mode),
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_model_roles_match": True,
        },
    }


def test_specialist_contract_snapshot_accepts_foundation_six_specialists() -> None:
    snapshot = _specialist_contract_snapshot(_bundle_meta_for_contract("foundation_seq146"), "foundation_seq146")

    assert snapshot["failures"] == []
    assert snapshot["expected_signal_dim"] == 146
    assert snapshot["required_specialists_exact"] is True
    assert snapshot["chart_geometry_present"] is False
    assert snapshot["price_action_candle_present"] is False


def test_specialist_contract_snapshot_accepts_challenger_seq215_eight_specialists() -> None:
    snapshot = _specialist_contract_snapshot(_bundle_meta_for_contract("challenger_seq215"), "challenger_seq215")

    assert snapshot["failures"] == []
    assert snapshot["expected_signal_dim"] == 215
    assert snapshot["required_specialists_exact"] is True
    assert snapshot["chart_geometry_present"] is True
    assert snapshot["price_action_candle_present"] is True


def test_specialist_contract_snapshot_blocks_seq215_foundation_fallback() -> None:
    meta = _bundle_meta_for_contract("foundation_seq146")

    snapshot = _specialist_contract_snapshot(meta, "challenger_seq215")

    assert any("seq_input_dim mismatch" in failure for failure in snapshot["failures"])
    assert any("contract mode mismatch" in failure for failure in snapshot["failures"])
    assert any("chart_geometry_encoder" in failure for failure in snapshot["failures"])
    assert any("price_action_candle_encoder" in failure for failure in snapshot["failures"])


def test_parser_has_challenger_seq215_alias() -> None:
    args = build_parser().parse_args(["--bundle-dir", "/tmp/bundle", "--challenger-seq215"])

    assert args.contract_mode == "challenger_seq215"


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


def test_neutralize_signal_bridge_sets_only_bridge_slots() -> None:
    seq_x = torch.ones((2, 3, 10), dtype=torch.float32)
    snap_x = torch.full((2, 10), 2.0, dtype=torch.float32)

    _neutralize_signal_bridge(seq_x, snap_x)

    expected = torch.as_tensor(SIGNAL_BRIDGE_NEUTRAL_VALUES, dtype=torch.float32)
    assert torch.allclose(seq_x[..., : len(expected)], expected.view(1, 1, -1))
    assert torch.allclose(snap_x[..., : len(expected)], expected.view(1, -1))
    assert torch.equal(seq_x[..., len(expected) :], torch.ones((2, 3, 3), dtype=torch.float32))
    assert torch.equal(snap_x[..., len(expected) :], torch.full((2, 3), 2.0, dtype=torch.float32))


def test_no_xgb_ablation_diagnostics_measure_prediction_delta() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["val", "val", "val", "val"],
            "model": ["candidate", "candidate", "candidate_no_xgb", "candidate_no_xgb"],
            "time": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC").tolist() * 2,
            "p_long": [0.7, 0.2, 0.6, 0.2],
            "p_short": [0.2, 0.7, 0.3, 0.7],
            "p_flat": [0.1, 0.1, 0.1, 0.1],
            "edge_score": [0.6, 0.6, 0.5, 0.6],
            "trade_side": [0, 1, 0, 1],
            "pred_direction": [0, 1, 0, 1],
        }
    )

    diagnostics = build_no_xgb_ablation_diagnostics(predictions, model_name="candidate")

    assert diagnostics["available"] is True
    assert diagnostics["splits"]["val"]["comparable"] is True
    assert diagnostics["splits"]["val"]["max_abs_prob_delta"] == pytest.approx(0.1)
    assert diagnostics["splits"]["val"]["max_abs_edge_score_delta"] == pytest.approx(0.1)


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
