import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_SPECIALIST,
    FOUNDATION_OBJECTIVE_SPECIALISTS,
    MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_EXPECTED_SIGNAL_DIM,
    MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
    MODEL_NATIVE_SMART_FAMILY_CONTRACT,
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_CONTRACT_MODES,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    SPECIALIST_GROUPS,
    classify_entry_specialist_feature,
    require_model_native_specialist_contract_mode,
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
    assert classify_entry_specialist_feature("p_long") == FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    assert FORBIDDEN_LEGACY_BRIDGE_SPECIALIST not in SPECIALIST_GROUPS


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
        "ctx_cont.D1_dist_from_ema200_atr": "trend_ema_encoder",
        "ctx_cont.d1_pct_change_5_canon_v2": "trend_ema_encoder",
        "ctx_cont.d1_dist_roc_288_v3": "trend_ema_encoder",
        "ctx_cont.dip_proximity_h1_v3": "momentum_flow_encoder",
        "ctx_cont.dip_proximity_mean_h1h4d1": "momentum_flow_encoder",
        "ctx_cont.smc_premium_extreme_snap": "smc_liquidity_encoder",
    }

    assert {field: classify_entry_specialist_feature(field) for field in expected} == expected


def test_context_routing_failures_fail_closed_for_model_native_contract() -> None:
    rows = [
        {
            "scope": "ctx_cont",
            "index": 0,
            "feature": "ctx_cont.unowned_context_feature_v1",
            "specialist": "unmapped",
        }
    ]

    failures = _context_routing_failures(rows)

    assert len(failures) == 1
    assert MODEL_NATIVE_CONTRACT_MODE in failures[0]
    assert "ctx_cont.unowned_context_feature_v1" in failures[0]


def test_only_model_native_seq513_specialist_contract_is_registered() -> None:
    assert SPECIALIST_CONTRACT_MODES == (MODEL_NATIVE_CONTRACT_MODE,)
    assert tuple(SPECIALIST_GROUPS) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert tuple(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert require_model_native_specialist_contract_mode(MODEL_NATIVE_CONTRACT_MODE) == MODEL_NATIVE_CONTRACT_MODE
    assert specialist_contract_training_allowed_for_mode(MODEL_NATIVE_CONTRACT_MODE) is True
    assert required_training_specialists_for_mode(MODEL_NATIVE_CONTRACT_MODE) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert (
        specialist_model_contract_for_mode(MODEL_NATIVE_CONTRACT_MODE)
        == MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT
    )

    family_contract = smart_family_contract_for_mode(MODEL_NATIVE_CONTRACT_MODE)
    assert family_contract == MODEL_NATIVE_SMART_FAMILY_CONTRACT
    assert len(family_contract) == 10
    assert sum(int(spec["expected_feature_count"]) for spec in family_contract.values()) == (
        MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT
    )
    for spec in family_contract.values():
        assert set(spec["owned_specialists"]).issubset(set(MODEL_NATIVE_TRAINING_SPECIALISTS))
        assert sum(spec["expected_specialist_counts"].values()) == spec["expected_feature_count"]

    for stale_mode in (None, "", "foundation_seq146", "challenger_seq215", "smart_seq520_candidate"):
        with pytest.raises(ValueError, match="model-native specialist contract mode required"):
            require_model_native_specialist_contract_mode(stale_mode)


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


def _smart_seq513_fields() -> tuple[list[str], list[str]]:
    selected = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    selected.extend(
        feature
        for feature in _foundation_selected_features()
        if feature not in set(selected)
    )
    selected.extend(
        _fill_features(
            "session_regime.smart_contract_ranked_fill_",
            MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT - len(selected),
        )
    )
    assert len(selected) == MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT

    base = list(MODEL_NATIVE_BASE_FIELDS)
    assert len(base) == 34
    fields = base + selected
    assert len(fields) == MODEL_NATIVE_EXPECTED_SIGNAL_DIM
    assert len(set(fields)) == len(fields)
    return fields, selected


def _smart_family_counts() -> dict[str, int]:
    return {
        label: int(spec["expected_feature_count"])
        for label, spec in MODEL_NATIVE_SMART_FAMILY_CONTRACT.items()
    }


def _write_smart_seq513_fixture(
    tmp_path: Path,
    *,
    family_counts: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    fields, selected = _smart_seq513_fields()
    signal_contract = model_native_signal_contract_metadata(selected)
    dataset_dir = tmp_path / "smart_dataset"
    dataset_dir.mkdir()
    for split in ("train", "val", "test"):
        manifest = {
            "extra": {
                "model_native_signal_contract": signal_contract,
                "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
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

    seq_manifest = tmp_path / "smart_seq513_model_native_manifest.json"
    seq_manifest.write_text(
        json.dumps(
            {
                "selected_features": selected,
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "base_signal_feature_count": len(MODEL_NATIVE_BASE_FIELDS),
                "model_native_signal_contract": signal_contract,
                "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
                "smart_layers_included": True,
                "smart_layer_feature_counts": family_counts or _smart_family_counts(),
                "source_feature_counts": {
                    "smart_candidate_layers": MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
                    "mandatory_full_stack": MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
                    "ranked_remainder": (
                        MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT
                        - MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT
                    ),
                },
                "expected_seq_snap_width": MODEL_NATIVE_EXPECTED_SIGNAL_DIM,
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
        quiet=True,
    )


def test_specialist_feature_group_audit_passes_model_native_seq513_contract_prep(
    tmp_path: Path,
) -> None:
    dataset_dir, seq_manifest = _write_smart_seq513_fixture(tmp_path)

    report = run(
        _audit_args(
            tmp_path,
            dataset_dir,
            seq_manifest,
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )
    )

    assert report["decision"] == "PASS"
    assert report["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert report["signal_field_count"] == MODEL_NATIVE_EXPECTED_SIGNAL_DIM
    assert report["selected_feature_count"] == MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT
    assert report["required_training_specialists"] == list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert report["specialist_model_contract"] == MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT
    assert report["specialist_model_contract_valid"] is True
    assert report["training_allowed"] is False
    assert report["training_allowed_with_explicit_vedtak"] is False
    assert report["contract_training_surface"] == {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "registered_for_training_surfaces": True,
        "training_allowed_by_contract_mode": True,
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


def test_specialist_feature_group_audit_fails_closed_on_model_native_family_count_mismatch(
    tmp_path: Path,
) -> None:
    family_counts = _smart_family_counts()
    family_counts["trend_ema_smart_layer"] -= 1
    dataset_dir, seq_manifest = _write_smart_seq513_fixture(
        tmp_path,
        family_counts=family_counts,
    )
    args = _audit_args(
        tmp_path,
        dataset_dir,
        seq_manifest,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )

    with pytest.raises(SystemExit) as exc_info:
        run(args)

    assert exc_info.value.code == 2
    report_paths = list(
        Path(args.out_dir).glob("ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_*.json")
    )
    assert len(report_paths) == 1
    report_path = report_paths[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "FAIL"
    assert report["training_allowed"] is False
    assert report["smart_family_contract_valid"] is False
    assert all(
        row["feature_count_matches"] is True
        for row in report["smart_family_contract_rows"]
    )
    assert any(
        "model-native mandatory family count metadata stale" in failure
        for failure in report["failures"]
    )


def test_specialist_audit_recomputes_and_rejects_same_group_mandatory_swap(
    tmp_path: Path,
) -> None:
    dataset_dir, seq_manifest = _write_smart_seq513_fixture(tmp_path)
    payload = json.loads(seq_manifest.read_text(encoding="utf-8"))
    selected = list(payload["selected_features"])
    victim = "trend.ema_mtf_score"
    replacement = "trend.ema_adversarial_same_group_replacement"
    assert classify_entry_specialist_feature(victim) == classify_entry_specialist_feature(
        replacement
    )
    selected[selected.index(victim)] = replacement
    assert len(selected) == len(set(selected)) == MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT
    payload["selected_features"] = selected
    seq_manifest.write_text(json.dumps(payload), encoding="utf-8")
    args = _audit_args(
        tmp_path,
        dataset_dir,
        seq_manifest,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )

    with pytest.raises(SystemExit) as exc_info:
        run(args)

    assert exc_info.value.code == 2
    report_path = next(
        Path(args.out_dir).glob("ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_*.json")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trend_row = next(
        row
        for row in report["smart_family_contract_rows"]
        if row["family"] == "trend_ema_smart_layer"
    )
    assert trend_row["selected_feature_count"] == 19
    assert trend_row["emitted_signal_feature_count"] == 20
    assert trend_row["feature_count_matches"] is False
