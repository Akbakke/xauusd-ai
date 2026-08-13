import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTEXT_TAG,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_SPECIALIST,
    MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_EXPECTED_SIGNAL_DIM,
    MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_NOMINAL_CTX_CONT_FIELDS,
    MODEL_NATIVE_SMART_FAMILY_CONTRACT,
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_CONTRACT_MODES,
    SPECIALIST_GROUPS,
    classify_entry_specialist_feature,
    model_native_context_temporal_alias_policy,
    require_model_native_specialist_contract_mode,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    PRICE_DERIVED_FEATURE_NAMES,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.audit_entry_specialist_feature_groups_v1 import (
    _context_taxonomy_failures,
    _specialist_input_liveness_rows,
    run,
)
from tests.model_native_context_routing_support import (
    V24_TEMPORAL_ALIAS_SIGNAL_FIELDS,
)


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
    assert (
        classify_entry_specialist_feature("signed_vol_z_20")
        == "momentum_flow_encoder"
    )
    assert classify_entry_specialist_feature("p_long") == FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    assert FORBIDDEN_LEGACY_BRIDGE_SPECIALIST not in SPECIALIST_GROUPS


def test_local_ema_formula_family_has_one_exact_trend_owner() -> None:
    # V30 (2026-08-13): 15 = 11 + chart.local_kama_efficiency_30 (package 1)
    # + the three GAP-2/3 local age fields (package 2).
    assert len(PRICE_DERIVED_FEATURE_NAMES) == len(set(PRICE_DERIVED_FEATURE_NAMES)) == 15
    assert {
        field: classify_entry_specialist_feature(field)
        for field in PRICE_DERIVED_FEATURE_NAMES
    } == {
        field: "trend_ema_encoder"
        for field in PRICE_DERIVED_FEATURE_NAMES
    }
    # Execution spread remains session evidence; the exact EMA-spread fields
    # above must not inherit that broad lexical owner.
    assert (
        classify_entry_specialist_feature("ctx_cont.spread_bps")
        == "session_regime_encoder"
    )


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
        "ctx_cont.d1_pct_change_5_canon_v2": "momentum_flow_encoder",
        "ctx_cont.d1_dist_roc_288_v3": "momentum_flow_encoder",
        "ctx_cont.d1_rsi14_canon_v2": "momentum_flow_encoder",
        "ctx_cont.m15_rsi14_canon_v2": "momentum_flow_encoder",
        "ctx_cont.dip_proximity_h1_v3": "momentum_flow_encoder",
        "ctx_cont.dip_proximity_mean_h1h4d1": "momentum_flow_encoder",
        "ctx_cont.smc_premium_extreme_snap": "smc_liquidity_encoder",
    }

    assert {field: classify_entry_specialist_feature(field) for field in expected} == expected


def test_context_taxonomy_failures_fail_closed_for_model_native_contract() -> None:
    rows = [
        {
            "scope": "ctx_cont",
            "index": 0,
            "feature": "ctx_cont.unowned_context_feature_v1",
            "specialist": "unmapped",
        }
    ]

    failures = _context_taxonomy_failures(rows)

    assert len(failures) == 1
    assert MODEL_NATIVE_CONTRACT_MODE in failures[0]
    assert "ctx_cont.unowned_context_feature_v1" in failures[0]


def test_context_routing_partitions_numeric_and_nominal_semantics_exactly() -> None:
    routing = MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT
    numeric = {
        index
        for values in routing["ctx_cont_numeric_indices"].values()
        for index in values
    }
    nominal = {
        index
        for values in routing["ctx_cont_nominal_indices"].values()
        for index in values
    }

    assert not (numeric & nominal)
    assert numeric | nominal == set(range(len(MODEL_NATIVE_CTX_CONT_FIELDS)))
    assert {
        MODEL_NATIVE_CTX_CONT_FIELDS[index] for index in nominal
    } == set(MODEL_NATIVE_NOMINAL_CTX_CONT_FIELDS)
    assert routing["ctx_cont_nominal_indices"]["session_regime_encoder"] == sorted(
        nominal
    )
    assert routing["nominal_ctx_cont_cardinality"] == 5
    assert routing["nominal_ctx_cont_representation"] == (
        "learned_categorical_embedding"
    )


def test_temporal_alias_policy_is_artifact_derived_not_fixed_to_v24_count() -> None:
    v24 = model_native_context_temporal_alias_policy(
        V24_TEMPORAL_ALIAS_SIGNAL_FIELDS
    )
    future_fields = [
        *V24_TEMPORAL_ALIAS_SIGNAL_FIELDS[:-1],
        "ctx_cont.h1_regime_class_id_v2",
    ]
    future = model_native_context_temporal_alias_policy(future_fields)

    assert v24["alias_count"] == 82
    assert future["alias_count"] == 82
    assert future["signal_fields"] != v24["signal_fields"]
    assert future["signal_fields_sha256"] != v24["signal_fields_sha256"]


def test_specialist_liveness_uses_exact_sparse_event_support_contract(
    tmp_path: Path,
) -> None:
    fields = [
        "smc_choch",
        "candle.pattern_outside_after_inside_bull_breakout_score",
        "candle.pattern_outside_after_inside_bear_breakout_score",
        "chart.local_ema50_200_cross_up",
        "chart.local_ema50_200_cross_down",
    ]
    event_counts = [32, 16, 16, 128, 128]
    split_artifacts: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        values = np.zeros((20_000, len(fields)), dtype=np.float32)
        if split == "train":
            for index, count in enumerate(event_counts):
                values[:count, index] = 1.0
        path = tmp_path / f"sparse_{split}.parquet"
        pd.DataFrame({"snap": values.tolist()}).to_parquet(path, index=False)
        split_artifacts[split] = {"parquet_path": str(path)}

    _groups, feature_rows, _duplicates = _specialist_input_liveness_rows(
        split_artifacts,
        ["train", "val"],
        fields,
        (),
    )

    train = [row for row in feature_rows if row["split"] == "train"]
    assert [row["active_count"] for row in train] == event_counts
    assert {row["status"] for row in train} == {"ALLOWED_RARE_EVENT"}
    assert all(row["live"] is True for row in train)
    oos = [row for row in feature_rows if row["split"] != "train"]
    assert {row["status"] for row in oos} == {"OBSERVED_SINGLE_STATE"}
    assert all(row["live"] is True for row in oos)


def test_specialist_liveness_rejects_sparse_event_below_exact_support_floor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "below_floor.parquet"
    values = np.zeros((20_000, 1), dtype=np.float32)
    values[:31, 0] = 1.0
    pd.DataFrame({"snap": values.tolist()}).to_parquet(path, index=False)

    _groups, feature_rows, _duplicates = _specialist_input_liveness_rows(
        {"train": {"parquet_path": str(path)}},
        ["train"],
        ["smc_choch"],
        (),
    )

    assert feature_rows[0]["status"] == "FAIL"
    assert feature_rows[0]["status_reason"] == "rare_event_support_below_minimum"
    assert feature_rows[0]["live"] is False


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

    family_contract = MODEL_NATIVE_SMART_FAMILY_CONTRACT
    assert len(family_contract) == len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
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
        feature
        for feature in V24_TEMPORAL_ALIAS_SIGNAL_FIELDS
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
    for split in ("train", "val"):
        parquet_path = dataset_dir / f"sample_{split}.parquet"
        manifest = {
            "output_data_path": str(parquet_path.resolve()),
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
                    "tag": MODEL_NATIVE_CONTEXT_TAG,
                    "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
                    "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
                    "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
                    "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS),
                    "ctx_cont_fields_sha256": (
                        MODEL_NATIVE_CTX_CONT_FIELDS_SHA256
                    ),
                    "ctx_cat_fields_sha256": (
                        MODEL_NATIVE_CTX_CAT_FIELDS_SHA256
                    ),
                },
            }
        }
        snap = [
            (np.linspace(0.1, 1.0, len(fields), dtype=np.float32) * float(i + 1)).tolist()
            for i in range(8)
        ]
        pd.DataFrame({"snap": snap}).to_parquet(parquet_path, index=False)
        (dataset_dir / f"sample_{split}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

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
    split_args: dict[str, str] = {}
    for split in ("train", "val"):
        manifest = (dataset_dir / f"sample_{split}.manifest.json").resolve()
        parquet = (dataset_dir / f"sample_{split}.parquet").resolve()
        split_args[f"{split}_manifest_json"] = str(manifest)
        split_args[f"{split}_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        split_args[f"{split}_parquet_sha256"] = hashlib.sha256(
            parquet.read_bytes()
        ).hexdigest()
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        seq_structure_manifest=str(seq_manifest),
        out_dir=str(tmp_path / "out"),
        data_splits="train,val",
        contract_mode=contract_mode,
        quiet=True,
        **split_args,
    )


def test_specialist_feature_group_audit_passes_model_native_seq513_contract_prep(
    tmp_path: Path,
) -> None:
    dataset_dir, seq_manifest = _write_smart_seq513_fixture(tmp_path)
    for split in ("train", "val"):
        (dataset_dir / f"unbound_decoy_{split}.parquet").write_bytes(b"decoy")

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
    assert len(report["smart_family_contract_rows"]) == len(
        MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    )
    assert all(row["feature_count_matches"] is True for row in report["smart_family_contract_rows"])
    assert len(report["specialist_input_liveness"]) == 16
    assert report["specialist_input_liveness_all_live"] is True
    assert report["context_taxonomy_all_mapped"] is True
    assert report["context_specialist_routing_all_mapped"] is True
    assert report["context_specialist_routing_failure_count"] == 0
    alias_policy = report["architecture_contract"][
        "context_specialist_routing"
    ]["temporal_alias_policy"]
    assert alias_policy["alias_count"] == 82
    assert alias_policy["signal_fields"] == [
        field
        for field in _smart_seq513_fields()[0]
        if field in V24_TEMPORAL_ALIAS_SIGNAL_FIELDS
    ]
    # The executable owner (share_temporal_alias_stats_from_signal) copies
    # signal-fitted statistics into ctx_cont; the declaration names that
    # direction.
    assert alias_policy["statistics_owner"] == "signal"
    assert alias_policy["signal_alias_statistics_policy"] == (
        "bit_identical_copy_from_signal_train_stats"
    )
    assert set(report["split_artifacts"]) == {"train", "val"}


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
