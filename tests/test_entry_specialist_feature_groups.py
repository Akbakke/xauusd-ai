import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import (
    FOUNDATION_OBJECTIVE_SPECIALISTS,
    SPECIALIST_MODEL_CONTRACT,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    classify_entry_specialist_feature,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.audit_entry_specialist_feature_groups_v1 import run


def test_entry_specialist_feature_classifier_maps_foundation_requirements() -> None:
    assert classify_entry_specialist_feature("chart.foundation_hh_state") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("chart.foundation_bos_up_age_bars") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("chart.foundation_choch_recent_tau24") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("chart.foundation_sweep_low_reclaim_up_proxy") == "smc_liquidity_encoder"
    assert classify_entry_specialist_feature("chart.foundation_compression_release_up") == "vol_compression_encoder"
    assert classify_entry_specialist_feature("chart.foundation_impulse_pullback_alignment") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("chart.foundation_eu_x_bos_balance") == "session_regime_encoder"
    assert classify_entry_specialist_feature("ema20_slope") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ret_5") == "momentum_flow_encoder"
    assert classify_entry_specialist_feature("p_long") == "neutral_bridge_anchor"


def test_specialist_feature_group_audit_passes_minimal_contract(tmp_path: Path) -> None:
    selected = list(
        dict.fromkeys(
            feature
            for features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.values()
            for feature in features
        )
    ) + [
        "chart.eu_x_bos",
        "chart.is_eu_only_x_pullback",
        "chart.eu_x_price_vs_ema200",
        "chart.premium_discount_x_level",
        "ctx_cont.sr_support_minus_resistance_prox",
        "ctx_cont.liquidity_hi_nearest_abs_atr",
        "chart.wick_level_x_level_prox",
    ]
    base = [
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "uncertainty_score",
        "margin_top1_top2",
        "entropy",
        "ema20_slope",
        "pos_vs_ema200",
        "_v1_ema_diff",
        "_v1_close_ema_slope_3",
        "_v1_kama_slope_30",
        "_v1_tema_slope_20",
        "_v1_atr14",
        "atr_z",
        "rvol_20",
        "_v1_bb_squeeze_20_2",
        "_v1_pk_sigma20",
        "_v1_range_z",
        "ret_1",
        "ret_5",
        "ret_20",
        "_v1_clv",
        "m5h1_momentum",
        "body_pct",
    ]
    fields = base + selected
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for split in ("train", "val", "test"):
        manifest = {
            "extra": {
                "signal_bridge": {
                    "fields": fields,
                    "seq_input_dim": len(fields),
                    "snap_input_dim": len(fields),
                    "seq_structure_extension_v1": {"features": selected},
                }
            }
        }
        (dataset_dir / f"sample_{split}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        snap = [
            (np.linspace(0.1, 1.0, len(fields), dtype=np.float32) * float(i + 1)).tolist()
            for i in range(8)
        ]
        pd.DataFrame({"snap": snap}).to_parquet(dataset_dir / f"sample_{split}.parquet", index=False)
    seq_manifest = tmp_path / "seq_manifest.json"
    seq_manifest.write_text(json.dumps({"selected_features": selected}), encoding="utf-8")

    report = run(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            seq_structure_manifest=str(seq_manifest),
            out_dir=str(tmp_path / "out"),
            data_splits="train,val,test",
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "PASS"
    assert report["architecture_contract"]["input_dim"] == len(fields)
    recommended = report["architecture_contract"]["recommended_fusion"]
    assert set(recommended["active_heads"]) == set(SPECIALIST_FUSION_ACTIVE_HEADS)
    assert set(recommended["blocked_heads"]) == set(SPECIALIST_FUSION_BLOCKED_HEADS)
    assert "hold_horizon" not in set(recommended["active_heads"])
    assert "hold_horizon" in set(recommended["blocked_heads"])
    assert report["specialist_input_liveness_all_live"] is True
    assert len(report["specialist_input_liveness"]) == 18
    assert report["foundation_objective_routing_all_present_and_expected"] is True
    assert report["specialist_model_contract_valid"] is True
    assert report["specialist_model_contract_failures"] == []
    assert set(report["specialist_model_contract"]) == set(SPECIALIST_MODEL_CONTRACT)
    expected_owned = {
        specialist: tuple(spec["owned_objectives"])
        for specialist, spec in SPECIALIST_MODEL_CONTRACT.items()
    }
    actual_owned = {
        specialist: tuple(spec["owned_objectives"])
        for specialist, spec in report["specialist_model_contract"].items()
    }
    assert actual_owned == expected_owned
    for specialist, spec in report["specialist_model_contract"].items():
        assert spec["model_role"]
        assert spec["primary_signal_families"]
        assert set(spec["supports_heads"]).issubset(set(SPECIALIST_FUSION_ACTIVE_HEADS))
    routing = {row["objective"]: row for row in report["foundation_objective_routing"]}
    assert set(routing) == set(FOUNDATION_OBJECTIVE_SPECIALISTS)
    for objective, expected_specialist in FOUNDATION_OBJECTIVE_SPECIALISTS.items():
        row = routing[objective]
        assert row["expected_specialist"] == expected_specialist
        assert row["present_count"] == row["required_count"]
        assert row["routed_to_expected_count"] == row["required_count"]
        assert row["missing_count"] == 0
        assert row["misrouted_count"] == 0
