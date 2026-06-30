import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import (
    CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT,
    CHALLENGER_SEQ215_TRAINING_SPECIALISTS,
    FOUNDATION_OBJECTIVE_SPECIALISTS,
    SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT,
    SMART_SEQ520_EXPECTED_SIGNAL_DIM,
    SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT,
    SMART_SEQ520_SMART_FAMILY_CONTRACT,
    SPECIALIST_AUDIT_CONTRACT_MODES,
    SPECIALIST_CONTRACT_MODES,
    SPECIALIST_MODEL_CONTRACT,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    classify_entry_specialist_feature,
    required_training_specialists_for_mode,
    smart_family_contract_for_mode,
    specialist_contract_training_allowed_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.audit_entry_specialist_feature_groups_v1 import _context_routing_failures, run


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


def test_entry_specialist_feature_classifier_maps_context_gate_fields() -> None:
    expected = {
        "ctx_cont.spread_bps": "session_regime_encoder",
        "ctx_cat.spread_bucket": "session_regime_encoder",
        "ctx_cat.vol_regime_id": "session_regime_encoder",
        "ctx_cat.atr_bucket": "vol_compression_encoder",
        "ctx_cont.is_us_only": "session_regime_encoder",
        "ctx_cont.is_eu_us_overlap": "session_regime_encoder",
        "ctx_cont.m5_regime_class_id_v2": "session_regime_encoder",
        "ctx_cont.m15_regime_class_id_v2": "session_regime_encoder",
        "ctx_cont.h1_regime_class_id_v2": "session_regime_encoder",
        "ctx_cont.h4_regime_class_id_v2": "session_regime_encoder",
        "ctx_cont.d1_regime_class_id_v2": "session_regime_encoder",
        "ctx_cont.regime_tf_agreement_v3": "session_regime_encoder",
        "ctx_cont.regime_stack_sum_v3": "session_regime_encoder",
        "ctx_cont.regime_divergence_flag_v3": "session_regime_encoder",
        "ctx_cont.d1_regime_changed_flag_v3": "session_regime_encoder",
        "ctx_cont.bars_since_d1_regime_change_v3": "session_regime_encoder",
        "ctx_cont.d1_dist_to_boundary_v3": "session_regime_encoder",
        "ctx_cont.retracement_from_last_impulse": "structure_swing_encoder",
        "ctx_cont.d1_pct_change_5_canon_v2": "trend_ema_encoder",
        "ctx_cont.d1_dist_roc_288_v3": "trend_ema_encoder",
        "ctx_cont.dip_proximity_h1_v3": "momentum_flow_encoder",
        "ctx_cont.dip_proximity_mean_h1h4d1": "momentum_flow_encoder",
        "ctx_cont.smc_premium_extreme_snap": "smc_liquidity_encoder",
    }

    assert {field: classify_entry_specialist_feature(field) for field in expected} == expected


def test_context_routing_failures_fail_closed_for_all_contract_modes() -> None:
    rows = [
        {
            "scope": "ctx_cont",
            "index": 0,
            "feature": "ctx_cont.unowned_context_feature_v1",
            "specialist": "unmapped",
        }
    ]

    for contract_mode in SPECIALIST_AUDIT_CONTRACT_MODES:
        failures = _context_routing_failures(rows, contract_mode=contract_mode)

        assert len(failures) == 1
        assert contract_mode in failures[0]
        assert "ctx_cont.unowned_context_feature_v1" in failures[0]


def test_smart_seq520_candidate_contract_is_audit_only_and_exact() -> None:
    assert SPECIALIST_CONTRACT_MODES == ("foundation_seq146", "challenger_seq215")
    assert "smart_seq520_candidate" in SPECIALIST_AUDIT_CONTRACT_MODES
    assert specialist_contract_training_allowed_for_mode("foundation_seq146") is True
    assert specialist_contract_training_allowed_for_mode("challenger_seq215") is True
    assert specialist_contract_training_allowed_for_mode("smart_seq520_candidate") is False
    assert required_training_specialists_for_mode("smart_seq520_candidate") == CHALLENGER_SEQ215_TRAINING_SPECIALISTS
    assert (
        specialist_model_contract_for_mode("smart_seq520_candidate")
        == CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT
    )

    family_contract = smart_family_contract_for_mode("smart_seq520_candidate")
    assert family_contract == SMART_SEQ520_SMART_FAMILY_CONTRACT
    assert len(family_contract) == 10
    assert sum(int(spec["expected_feature_count"]) for spec in family_contract.values()) == (
        SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT
    )
    for spec in family_contract.values():
        assert set(spec["owned_specialists"]).issubset(set(CHALLENGER_SEQ215_TRAINING_SPECIALISTS))
        assert sum(spec["expected_specialist_counts"].values()) == spec["expected_feature_count"]


def _foundation_selected_features() -> list[str]:
    return list(
        dict.fromkeys(
            feature
            for features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.values()
            for feature in features
        )
    )


def _fill_features(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(count)]


def _smart_seq520_fields() -> tuple[list[str], list[str]]:
    selected = _foundation_selected_features()
    filler_groups = [
        ("chart.structure_swing_smart_contract_fill_", 80),
        ("chart.smc_liquidity_smart_contract_fill_", 75),
        ("trend.ema_smart_contract_fill_", 55),
        ("vol_compression.smart_contract_fill_", 45),
        ("momentum.flow_smart_contract_fill_", 40),
        ("session_regime.smart_contract_fill_", 80),
        ("chart.geometry_smart_contract_fill_", 55),
        ("candle.pattern_smart_contract_fill_", 55),
    ]
    for prefix, count in filler_groups:
        for name in _fill_features(prefix, count):
            if len(selected) >= SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT:
                break
            if name not in selected:
                selected.append(name)
        if len(selected) >= SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT:
            break
    assert len(selected) == SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT

    base = [
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "uncertainty_score",
        "margin_top1_top2",
        "entropy",
    ]
    base.extend(_fill_features("trend.ema_base_contract_fill_", 6))
    base.extend(_fill_features("vol_compression.base_contract_fill_", 6))
    base.extend(_fill_features("momentum.flow_base_contract_fill_", 5))
    base.extend(_fill_features("session_regime.base_contract_fill_", 6))
    base.extend(_fill_features("chart.geometry_base_contract_fill_", 5))
    base.extend(_fill_features("candle.pattern_base_contract_fill_", 6))
    assert len(base) == 41
    fields = base + selected
    assert len(fields) == SMART_SEQ520_EXPECTED_SIGNAL_DIM
    assert len(set(fields)) == len(fields)
    return fields, selected


def _smart_family_counts() -> dict[str, int]:
    return {
        label: int(spec["expected_feature_count"])
        for label, spec in SMART_SEQ520_SMART_FAMILY_CONTRACT.items()
    }


def _write_smart_seq520_fixture(
    tmp_path: Path,
    *,
    family_counts: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    fields, selected = _smart_seq520_fields()
    dataset_dir = tmp_path / "smart_dataset"
    dataset_dir.mkdir()
    for split in ("train", "val", "test"):
        manifest = {
            "extra": {
                "signal_bridge": {
                    "fields": fields,
                    "seq_input_dim": len(fields),
                    "snap_input_dim": len(fields),
                    "seq_structure_extension_v1": {"features": selected},
                },
                "ctx_contract": {
                    "tag": "UNIT_CTX",
                    "ctx_cont_names": [
                        "spread_bps",
                        "is_us_only",
                        "regime_stack_sum_v3",
                        "d1_regime_changed_flag_v3",
                        "dip_proximity_h1_v3",
                        "sr_support_minus_resistance_prox",
                    ],
                    "ctx_cat_names": ["spread_bucket", "atr_bucket"],
                    "ctx_cont_dim": 6,
                    "ctx_cat_dim": 2,
                },
            }
        }
        (dataset_dir / f"sample_{split}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        snap = [
            (np.linspace(0.1, 1.0, len(fields), dtype=np.float32) * float(i + 1)).tolist()
            for i in range(8)
        ]
        pd.DataFrame({"snap": snap}).to_parquet(dataset_dir / f"sample_{split}.parquet", index=False)

    seq_manifest = tmp_path / "smart_seq520_manifest.json"
    seq_manifest.write_text(
        json.dumps(
            {
                "selected_features": selected,
                "manifest_variant": "smart_seq520_candidate",
                "smart_layers_included": True,
                "smart_layer_feature_counts": family_counts or _smart_family_counts(),
                "source_feature_counts": {
                    "foundation_sequence_extension": 105,
                    "chart_geometry_challenger": 41,
                    "candlestick_challenger": 28,
                    "smart_candidate_layers": SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT,
                },
                "expected_seq_snap_width": SMART_SEQ520_EXPECTED_SIGNAL_DIM,
                "dataset_rebuild_required_before_training": True,
                "training_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir, seq_manifest


def _audit_args(tmp_path: Path, dataset_dir: Path, seq_manifest: Path, *, contract_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        seq_structure_manifest=str(seq_manifest),
        out_dir=str(tmp_path / "out"),
        data_splits="train,val,test",
        contract_mode=contract_mode,
        fail_on_audit_fail=True,
        quiet=True,
    )


def test_specialist_feature_group_audit_passes_minimal_contract(tmp_path: Path) -> None:
    selected = _foundation_selected_features() + [
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
                },
                "ctx_contract": {
                    "tag": "UNIT_CTX",
                    "ctx_cont_names": [
                        "spread_bps",
                        "is_us_only",
                        "regime_stack_sum_v3",
                        "d1_regime_changed_flag_v3",
                        "dip_proximity_h1_v3",
                    ],
                    "ctx_cat_names": ["spread_bucket", "atr_bucket"],
                    "ctx_cont_dim": 5,
                    "ctx_cat_dim": 2,
                },
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
    assert report["signal_unmapped_count"] == 0
    assert report["signal_unmapped_fields"] == []
    assert report["signal_routing_all_mapped"] is True
    assert report["context_routing_unmapped_count"] == 0
    assert report["context_routing_unmapped_fields"] == []
    assert report["context_routing_all_mapped"] is True
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
    assert report["training_allowed"] is False
    assert report["contract_training_surface"]["registered_for_training_surfaces"] is True
    assert report["contract_training_surface"]["training_allowed_by_contract_mode"] is True
    assert report["smart_family_contract_required"] is False
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


def test_specialist_feature_group_audit_passes_smart_seq520_contract_prep(tmp_path: Path) -> None:
    dataset_dir, seq_manifest = _write_smart_seq520_fixture(tmp_path)

    report = run(_audit_args(tmp_path, dataset_dir, seq_manifest, contract_mode="smart_seq520_candidate"))

    assert report["decision"] == "PASS"
    assert report["contract_mode"] == "smart_seq520_candidate"
    assert report["signal_field_count"] == SMART_SEQ520_EXPECTED_SIGNAL_DIM
    assert report["selected_feature_count"] == SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT
    assert report["required_training_specialists"] == list(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert report["specialist_model_contract"] == CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT
    assert report["specialist_model_contract_valid"] is True
    assert report["training_allowed"] is False
    assert report["training_allowed_with_explicit_vedtak"] is False
    assert report["contract_training_surface"] == {
        "contract_mode": "smart_seq520_candidate",
        "registered_for_training_surfaces": False,
        "training_allowed_by_contract_mode": False,
        "training_allowed_by_this_audit": False,
        "training_allowed": False,
        "requires_separate_readiness_gate": True,
    }
    assert report["smart_family_contract_required"] is True
    assert report["smart_family_contract_valid"] is True
    assert report["smart_family_contract_failures"] == []
    assert len(report["smart_family_contract_rows"]) == 10
    assert all(row["feature_count_matches"] is True for row in report["smart_family_contract_rows"])
    assert len(report["specialist_input_liveness"]) == 24
    assert report["specialist_input_liveness_all_live"] is True


def test_specialist_feature_group_audit_fails_closed_on_smart_family_count_mismatch(tmp_path: Path) -> None:
    family_counts = _smart_family_counts()
    family_counts["trend_ema_smart_layer"] -= 1
    dataset_dir, seq_manifest = _write_smart_seq520_fixture(tmp_path, family_counts=family_counts)
    args = _audit_args(tmp_path, dataset_dir, seq_manifest, contract_mode="smart_seq520_candidate")
    args.fail_on_audit_fail = False

    report = run(args)

    assert report["decision"] == "FAIL"
    assert report["training_allowed"] is False
    assert report["smart_family_contract_valid"] is False
    assert any("smart_seq520 family count mismatch: trend_ema_smart_layer" in failure for failure in report["failures"])
