import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gx1.contracts.entry_full_input_liveness_v1 import (
    classify_field_name_semantics,
)
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
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
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.features.micro_structure_v1 import MICRO_FEATURE_NAMES_V1
from gx1.scripts.audit_entry_specialist_feature_groups_v1 import (
    _context_taxonomy_failures,
    _specialist_input_liveness_rows,
    run,
)
from tests.model_native_context_routing_support import (
    TEMPORAL_ALIAS_SIGNAL_FIELDS,
)


def test_entry_specialist_feature_classifier_maps_foundation_requirements() -> None:
    assert classify_entry_specialist_feature("chart.foundation_bos_up_event_age_bars") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("chart.foundation_choch_event_age_bars") == "structure_swing_encoder"
    assert classify_entry_specialist_feature("ema20_slope_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ret_1") == "momentum_flow_encoder"
    # Both retired names must now fail LOUD rather than route somewhere plausible.
    assert classify_entry_specialist_feature("ema20_slope") == "forbidden_legacy_bridge"
    assert classify_entry_specialist_feature("ret_5") == "forbidden_legacy_bridge"


def test_local_micro_primitives_route_by_declared_formula_not_by_producer() -> None:
    """Each local price/quote primitive goes to the owner its formula names.

    ``gx1.features.micro_structure_v1`` is one PRODUCER, not one specialist
    concept.  Routing the whole module to chart_geometry_encoder put three
    returns, one price-vs-EMA distance and two single-bar close-location fields
    into the encoder declared for "identified support/resistance levels,
    persistent trendline state, line and level touch history, break and retest
    events, channel geometry" -- which describes none of them.  The expected
    owners below are hand-written from the formulas that module declares.
    """

    expected = {
        # (close[t]/close[t-3]-1)*10000, (close[t]/close[t-5]-1)*10000 and the
        # one-bar difference of consecutive returns -> "recent returns" and
        # "acceleration" are declared momentum_flow_encoder families.
        "close_return_3_bps": "momentum_flow_encoder",
        "close_return_5_bps": "momentum_flow_encoder",
        "close_return_acceleration_1_bps": "momentum_flow_encoder",
        # (close[t]-classic_sma_seeded_ema5[t])/close[t]*10000 -> "price-vs-EMA"
        # is a declared trend_ema_encoder family.
        "close_distance_from_ema5_bps": "trend_ema_encoder",
        # (high[t]-close[t])/(high[t]-low[t]) and 1 iff high[t]>low[t]: computed
        # from ONE bar's own high/low/close, and "close location and zero-range
        # identity" is a declared price_action_candle_encoder family.
        "close_distance_below_high_range_fraction": "price_action_candle_encoder",
        "close_range_observed": "price_action_candle_encoder",
    }
    # Fail closed if the producer grows a primitive nobody adjudicated.
    assert set(expected) == set(MICRO_FEATURE_NAMES_V1)
    assert {
        name: classify_entry_specialist_feature(f"ctx_cont.{name}")
        for name in MICRO_FEATURE_NAMES_V1
    } == expected
    # The bare form must reach the same owner as the ctx_cont-prefixed form.
    assert {
        name: classify_entry_specialist_feature(name)
        for name in MICRO_FEATURE_NAMES_V1
    } == expected


def test_retired_and_forbidden_bridge_fields_are_never_an_encoder_group() -> None:
    assert (
        classify_entry_specialist_feature("signed_vol_z_20")
        == FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    )
    assert classify_entry_specialist_feature("p_long") == FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    assert FORBIDDEN_LEGACY_BRIDGE_SPECIALIST not in SPECIALIST_GROUPS


def test_local_ema_formula_family_has_one_exact_trend_owner() -> None:
    # The width is NEVER restated here (rule 13).  A literal ``== 19`` stood in
    # this line until 2026-08-19 and went stale the moment
    # chart.local_ema50_200_spread_bps was retired for volatility coupling; the
    # binding that survives a width change is the executable one below --
    # the layer's ordered tuple IS the price_ema50_200_layer family of the
    # mandatory registry, with no duplicates and no member missing.
    assert len(PRICE_DERIVED_FEATURE_NAMES) == len(set(PRICE_DERIVED_FEATURE_NAMES))
    assert PRICE_DERIVED_FEATURE_NAMES
    assert (
        dict(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)["price_ema50_200_layer"]
        == PRICE_DERIVED_FEATURE_NAMES
    )
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
        "ctx_cat.session_id": "session_regime_encoder",
        "ctx_cont.m5_trend_state_age_bars_v2": "trend_ema_encoder",
        "ctx_cont.m15_trend_state_age_bars_v2": "trend_ema_encoder",
        "ctx_cont.h1_trend_state_age_bars_v2": "trend_ema_encoder",
        "ctx_cont.h4_trend_state_age_bars_v2": "trend_ema_encoder",
        "ctx_cont.d1_trend_state_age_bars_v2": "trend_ema_encoder",
        "ctx_cont.m15_ema5_20_spread_atr_canon_v2": "trend_ema_encoder",
        "ctx_cont.h4_mid_ema50_dist_atr_canon_v2": "trend_ema_encoder",
        "ctx_cont.retracement_from_last_impulse": "structure_swing_encoder",
        "ctx_cont.D1_dist_from_ema200_atr": "trend_ema_encoder",
        "ctx_cont.d1_change_5_bps_canon_v2": "momentum_flow_encoder",
        "ctx_cont.d1_dist_change_1bar_atr_v4": "momentum_flow_encoder",
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
    assert routing["nominal_ctx_cont_cardinality"] == 0
    assert routing["nominal_ctx_cont_representation"] == "none"


def test_temporal_alias_policy_is_artifact_derived_not_fixed_to_fixture_count() -> None:
    baseline = model_native_context_temporal_alias_policy(
        TEMPORAL_ALIAS_SIGNAL_FIELDS
    )
    future_fields = list(TEMPORAL_ALIAS_SIGNAL_FIELDS[:-1])
    future = model_native_context_temporal_alias_policy(future_fields)

    assert baseline["alias_count"] == len(TEMPORAL_ALIAS_SIGNAL_FIELDS)
    assert future["alias_count"] == len(TEMPORAL_ALIAS_SIGNAL_FIELDS) - 1
    assert future["signal_fields"] != baseline["signal_fields"]
    assert future["signal_fields_sha256"] != baseline["signal_fields_sha256"]


def test_specialist_liveness_uses_exact_sparse_event_support_contract(
    tmp_path: Path,
) -> None:
    fields = [
        "smc_choch",
        "chart.local_ema50_200_cross_up",
        "chart.local_ema50_200_cross_down",
    ]
    event_counts = [32, 128, 128]
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


def _smart_seq513_fields() -> tuple[list[str], list[str]]:
    selected = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    ]
    assert len(selected) == MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT

    base = list(MODEL_NATIVE_BASE_FIELDS)
    assert len(base) == len(MODEL_NATIVE_BASE_FIELDS)
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
        # The synthetic ramp must honour each field's own name contract: an
        # all-positive column declares every `_slope`/`_delta`/`_spread`/`_z`
        # field one-sided, which the liveness semantics gate rejects exactly as
        # it would on real bytes.
        _rows = 8
        _snap = np.stack(
            [
                np.linspace(0.1, 1.0, len(fields), dtype=np.float32)
                * np.float32(i + 1)
                for i in range(_rows)
            ]
        )
        _row_ramp = np.linspace(0.0, 1.0, _rows, dtype=np.float32)[:, None]
        for _index, _name in enumerate(fields):
            _semantics = classify_field_name_semantics(_name)
            # Each column must straddle zero AND stay distinct: a shared
            # ramp would make every signed field an exact duplicate, which the
            # audit rejects for its own (correct) reason.
            _scale = np.float32(1.0 + _index * 0.001)
            if _semantics == "signed":
                _snap[:, _index] = (_row_ramp[:, 0] - np.float32(0.5)) * _scale
            elif _semantics == "unit_interval":
                _snap[:, _index] = _row_ramp[:, 0] * np.float32(
                    1.0 - _index * 1e-5
                )
        snap = _snap.tolist()
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
                    "available_candidates": (
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
    assert alias_policy["alias_count"] == len(MODEL_NATIVE_CTX_CONT_FIELDS)
    assert alias_policy["signal_fields"] == [
        field
        for field in _smart_seq513_fields()[0]
        if field.startswith("ctx_cont.")
        and field.removeprefix("ctx_cont.") in MODEL_NATIVE_CTX_CONT_FIELDS
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
    family_counts["raw_mtf_trend_layer"] -= 1
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
    victim = next(
        feature
        for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        if family == "raw_mtf_trend_layer"
        for feature in features
    )
    replacement = "ctx_cont.adversarial_ema_diff_same_group_replacement"
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
        if row["family"] == "raw_mtf_trend_layer"
    )
    expected_count = len(
        dict(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)[
            "raw_mtf_trend_layer"
        ]
    )
    assert trend_row["selected_feature_count"] == expected_count - 1
    assert trend_row["emitted_signal_feature_count"] == expected_count
    assert trend_row["feature_count_matches"] is False


# ---------------------------------------------------------------------------
# Routing adjudication ledger (2026-08-19).
#
# WHY THIS EXISTS.  ``classify_entry_specialist_feature`` decides most fields
# by LEXICAL SUBSTRING, so a rename or an addition can hand a field a plausible
# WRONG specialist with nothing failing: the field is still live, still finite,
# still has a producer, and every gate stays green (CLAUDE.md rule 25).  Two
# measured instances in this wave:
#   * ``_v1_ema3_ema6_spread_frac`` -> ``_v1_ema3_ema6_spread_atr`` re-routed an
#     EMA trend field to session_regime_encoder, because ``spread`` matched.
#   * ``ema20_slope_atr`` returned vol_compression_encoder, because the ATR
#     DENOMINATOR matched before the EMA SUBJECT.
#
# WHAT IT IS.  A frozen record of the CURRENT owner of every field in the live
# classify-decided population, derived from the contract owners below.  It is a
# LEDGER, not a certificate: it proves nothing moved unnoticed.  Whether a given
# owner is CORRECT is asserted separately, field by field, in the tests above
# and below against what each producer computes.
#
# WHEN IT FAILS.  Add, rename, retire or re-route a field and this goes red.
# That is the point.  Do not paste the new value in; decide which specialist the
# producer's formula belongs to, record the reason, then update this ledger.
#
# NOT VACUOUS: the expected side is a frozen literal, not a re-read of the
# module's own tables, so it fails if the routing arithmetic changes.
# ---------------------------------------------------------------------------
EXPECTED_LIVE_SPECIALIST_ROUTING: dict[str, tuple[str, ...]] = {
    "structure_swing_encoder": (
        "bars_since_swing_high_break",
        "bars_since_swing_low_break",
        "chart.foundation_bos_down_event_age_bars",
        "chart.foundation_bos_up_event_age_bars",
        "chart.foundation_choch_event_age_bars",
        "consecutive_higher_highs_count",
        "consecutive_higher_lows_count",
        "consecutive_lower_highs_count",
        "consecutive_lower_lows_count",
        "ctx_cont.bars_since_swing_high",
        "ctx_cont.bars_since_swing_low",
        "ctx_cont.dist_last_swing_high_atr",
        "ctx_cont.dist_last_swing_low_atr",
        "ctx_cont.retracement_from_last_impulse",
        "ctx_cont.swing_impulse_present",
        "smc_bos_displacement_atr",
        "smc_bos_down",
        "smc_bos_up",
        "smc_choch",
        "smc_swing_state",
        "swing_high_break_displacement_atr",
        "swing_high_break_event",
        "swing_high_level_intact",
        "swing_high_sequence_delta_atr",
        "swing_low_break_displacement_atr",
        "swing_low_break_event",
        "swing_low_level_intact",
        "swing_low_sequence_delta_atr",
    ),
    "smc_liquidity_encoder": (
        "ctx_cont.dist_to_R1_atr",
        "ctx_cont.dist_to_R2_atr",
        "ctx_cont.dist_to_S1_atr",
        "ctx_cont.dist_to_S2_atr",
        "ctx_cont.dist_to_d1_hi_atr",
        "ctx_cont.dist_to_d1_lo_atr",
        "ctx_cont.dist_to_h1_hi_atr",
        "ctx_cont.dist_to_h1_lo_atr",
        "ctx_cont.dist_to_h4_hi_atr",
        "ctx_cont.dist_to_h4_lo_atr",
        "ctx_cont.dist_to_m15_hi_atr",
        "ctx_cont.dist_to_m15_lo_atr",
        "ctx_cont.dist_to_m5_hi_atr",
        "ctx_cont.dist_to_m5_lo_atr",
        "level_above2_dist_atr",
        "level_above2_present",
        "level_above_age_bars",
        "level_above_bars_since_touch",
        "level_above_dist_atr",
        "level_above_last_reaction_atr",
        "level_above_max_reaction_atr",
        "level_above_mean_reaction_atr",
        "level_above_recurrence_confirmed",
        "level_above_touch_count",
        "level_bars_since_break",
        "level_bars_since_break_signed",
        "level_below2_dist_atr",
        "level_below2_present",
        "level_below_age_bars",
        "level_below_bars_since_touch",
        "level_below_dist_atr",
        "level_below_last_reaction_atr",
        "level_below_max_reaction_atr",
        "level_below_mean_reaction_atr",
        "level_below_present",
        "level_below_recurrence_confirmed",
        "level_below_touch_count",
        "level_break_down_event",
        "level_break_up_event",
        "level_broken_touch_count",
        "level_retest_fail_signed",
        "level_retest_hold_signed",
        "smc_pivot_envelope_position",
        "smc_sweep_down_depth_atr",
        "smc_sweep_down_event",
        "smc_sweep_down_state",
        "smc_sweep_event_age_bars",
        "smc_sweep_up_depth_atr",
        "smc_sweep_up_event",
        "smc_sweep_up_state",
    ),
    "trend_ema_encoder": (
        "_v1_ema3_ema6_spread_atr",
        "_v1_ema_diff",
        "_v1_kama30_change_5_atr",
        "_v1_tema20_change_3_atr",
        "chart.local_ema200_slope_atr",
        "chart.local_ema50_200_bull_state",
        "chart.local_ema50_200_cross_down",
        "chart.local_ema50_200_cross_up",
        "chart.local_ema50_200_spread_accel_atr",
        "chart.local_ema50_200_spread_atr",
        "chart.local_ema50_200_spread_delta_atr",
        "chart.local_ema50_200_state_age_bars",
        "chart.local_ema50_slope_atr",
        "chart.local_kama_efficiency_30",
        "chart.local_price_vs_ema200_atr",
        "chart.local_price_vs_ema200_state_age_bars",
        "chart.local_price_vs_ema50_atr",
        "chart.local_price_vs_ema50_state_age_bars",
        "chart.local_price_x_ema200_cross_down",
        "chart.local_price_x_ema200_cross_up",
        "chart.local_price_x_ema50_cross_down",
        "chart.local_price_x_ema50_cross_up",
        "ctx_cont.D1_dist_from_ema200_atr",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h1_slope3",
        "ctx_cont._v1h1_slope5",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont._v1h4_slope3",
        "ctx_cont._v1h4_slope5",
        "ctx_cont.close_distance_from_ema5_bps",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.d1_ema_stack_aligned_v2",
        "ctx_cont.d1_trend_state_age_bars_v2",
        "ctx_cont.h1_ema_stack_aligned_v2",
        "ctx_cont.h1_trend_state_age_bars_v2",
        "ctx_cont.h4_ema_stack_aligned_v2",
        "ctx_cont.h4_mid_ema50_dist_atr_canon_v2",
        "ctx_cont.h4_trend_state_age_bars_v2",
        "ctx_cont.m15_ema5_20_spread_atr_canon_v2",
        "ctx_cont.m15_ema_stack_aligned_v2",
        "ctx_cont.m15_trend_state_age_bars_v2",
        "ctx_cont.m5_ema_stack_aligned_v2",
        "ctx_cont.m5_trend_state_age_bars_v2",
        "ema20_slope_atr",
    ),
    "vol_compression_encoder": (
        "_v1_atr14",
        "_v1_bb10_bandwidth_change_3",
        "_v1_bb_squeeze_20_2",
        "_v1_kurt_r",
        "_v1_pk_sigma20",
        "_v1_range_z",
        "atr_z",
        "ctx_cont._v1h1_atr_bps",
        "ctx_cont._v1h4_atr_bps",
        "ctx_cont.atr_bps",
        "ctx_cont.d1_atr14_bps_canon_v2",
        "ctx_cont.d1_range_z_20_canon_v2",
        "ctx_cont.m15_range_z_20_canon_v2",
        "rvol_20",
        "volatility.bars_in_squeeze",
        "volatility.duration_at_release",
        "volatility.squeeze_active",
        "volatility.squeeze_release_age_bars",
        "volatility.squeeze_release_event",
    ),
    "momentum_flow_encoder": (
        "bear_divergence_event",
        "bear_divergence_strength",
        "bull_divergence_event",
        "bull_divergence_strength",
        "ctx_cont._v1h1_rsi14_z",
        "ctx_cont._v1h4_rsi14_z",
        "ctx_cont.close_return_3_bps",
        "ctx_cont.close_return_5_bps",
        "ctx_cont.close_return_acceleration_1_bps",
        "ctx_cont.d1_change_5_bps_canon_v2",
        "ctx_cont.d1_dist_change_1bar_atr_v4",
        "ctx_cont.d1_rsi14_canon_v2",
        "ctx_cont.h1_rsi14_canon_v2",
        "ctx_cont.h4_rsi14_canon_v2",
        "ctx_cont.m15_rsi14_canon_v2",
        "ctx_cont.m5_rsi14_canon_v2",
        "divergence_event_age_bars",
        "mom20_sign_flip_down",
        "mom20_sign_flip_up",
        "mom_20_atr",
        "mom_5_atr",
        "ret_1",
        "ret_20",
        "rsi14_centered",
        "rsi14_delta_5",
        "rsi_cross_down_50",
        "rsi_cross_down_70",
        "rsi_cross_up_30",
        "rsi_cross_up_50",
        "rsi_extreme_event_age_bars",
        "vol_pct_96",
        "vol_ratio_5_20",
        "vol_z_20",
    ),
    "session_regime_encoder": (
        "ctx_cat.session_id",
        "ctx_cont.dow_sin",
        "ctx_cont.hour_cos",
        "ctx_cont.hour_sin",
        "ctx_cont.minutes_since_session_open",
        "ctx_cont.quote_range_asymmetry_bps",
        "ctx_cont.session_change_flag",
        "ctx_cont.spread_bps",
        "ctx_cont.spread_bps_delta_1",
        "ctx_cont.spread_extremes_sum_bps",
    ),
    "chart_geometry_encoder": (
        "chart.geomchan_active",
        "chart.geomchan_apex_proximity",
        "chart.geomchan_converging",
        "chart.geomchan_pos_0_1",
        "chart.geomchan_slope_atr_per_bar",
        "chart.geomchan_width_atr",
        "chart.geomline_above_active_count",
        "chart.geomline_above_age_bars",
        "chart.geomline_above_dist_atr",
        "chart.geomline_above_last_touch_age_bars",
        "chart.geomline_above_max_dev_atr",
        "chart.geomline_above_slope_atr_per_bar",
        "chart.geomline_above_touch_count",
        "chart.geomline_bars_since_break",
        "chart.geomline_below_active_count",
        "chart.geomline_below_age_bars",
        "chart.geomline_below_dist_atr",
        "chart.geomline_below_last_touch_age_bars",
        "chart.geomline_below_max_dev_atr",
        "chart.geomline_below_slope_atr_per_bar",
        "chart.geomline_below_touch_count",
        "chart.geomline_break_down",
        "chart.geomline_break_line_age_bars",
        "chart.geomline_break_line_touch_count",
        "chart.geomline_break_up",
        "chart.geomline_retest_fail_down",
        "chart.geomline_retest_fail_up",
        "chart.geomline_retest_hold_down",
        "chart.geomline_retest_hold_up",
        "chart.geomline_touch_above",
        "chart.geomline_touch_below",
        "ctx_cont.d1_close_pct_in_20day_range_canon_v2",
    ),
    "price_action_candle_encoder": (
        # ADJUDICATED 2026-08-19: ``body_pct`` leaves this ledger with the
        # field itself.  It was retired from the LOCAL surface earlier in this
        # wave (materialize_build_canonical_features_v1) and from the per-TF
        # lane in this commit, in both cases as an exact duplicate of
        # ``candle.raw_body_signed_range`` / its ``mtf_`` twin -- which stays
        # in THIS SAME specialist, one line below.  So the routing decision is
        # unchanged: the price-action candle owner keeps the body-share
        # evidence, now with the sign it used to discard.  This entry can only
        # be dropped once the signal contract's ``MODEL_NATIVE_BASE_FIELDS``
        # has dropped ``body_pct``; until that commit lands the field is still
        # in the live population and this ledger reports it unadjudicated.
        "candle.raw_bear_body_covers_previous_bull_body_event",
        "candle.raw_body_change_local_geometry",
        "candle.raw_body_contained_by_previous_flag",
        "candle.raw_body_contains_previous_flag",
        "candle.raw_body_overlap_previous_local_geometry",
        "candle.raw_body_signed_range",
        "candle.raw_bull_body_covers_previous_bear_body_event",
        "candle.raw_close_change_local_geometry",
        "candle.raw_high_change_local_geometry",
        "candle.raw_high_rejection_previous_high_event",
        "candle.raw_low_change_local_geometry",
        "candle.raw_low_rejection_previous_low_event",
        "candle.raw_lower_wick_share",
        "candle.raw_observed_body_direction_duration_bars",
        "candle.raw_observed_range_relation_duration_bars",
        "candle.raw_open_above_previous_high_local_geometry",
        "candle.raw_open_below_previous_low_local_geometry",
        "candle.raw_open_position_previous_range",
        "candle.raw_range_contained_by_previous_flag",
        "candle.raw_range_contains_previous_flag",
        "candle.raw_upper_wick_share",
        "ctx_cont.close_distance_below_high_range_fraction",
        "ctx_cont.close_range_observed",
    ),
}


def _live_classify_decided_population() -> tuple[str, ...]:
    """Every field whose specialist is decided by the classifier.

    Derived by executing the contract owners (rule 13), never restated: the
    frozen base block, the mandatory causal families, the code-owned candidate
    remainder and the declared continuous/categorical context surfaces.  The
    family x timeframe lane is deliberately NOT here -- on that lane
    ``MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4`` is the executing authority, and it
    is covered by its own assertions below.
    """

    ordered: list[str] = []
    seen: set[str] = set()
    for field in (
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        *(f"ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS),
        *(f"ctx_cat.{name}" for name in MODEL_NATIVE_CTX_CAT_FIELDS),
    ):
        if field not in seen:
            seen.add(field)
            ordered.append(field)
    return tuple(ordered)


def test_every_live_field_routes_to_its_adjudicated_specialist() -> None:
    population = _live_classify_decided_population()
    expected = {
        field: specialist
        for specialist, fields in EXPECTED_LIVE_SPECIALIST_ROUTING.items()
        for field in fields
    }
    observed = {field: classify_entry_specialist_feature(field) for field in population}

    unadjudicated = sorted(set(observed) - set(expected))
    retired = sorted(set(expected) - set(observed))
    assert not unadjudicated, (
        "new or renamed live fields have no adjudicated specialist; decide the "
        "owner from what the producer computes, then record it: "
        f"{[(name, observed[name]) for name in unadjudicated]}"
    )
    assert not retired, (
        "adjudicated fields are no longer in the live population; remove them "
        f"from the ledger: {retired}"
    )
    moved = {
        field: (expected[field], owner)
        for field, owner in observed.items()
        if expected[field] != owner
    }
    assert not moved, f"specialist routing moved (field: expected -> observed): {moved}"
    assert set(expected.values()) <= set(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert len(population) == len(expected)


def test_unit_suffix_never_outranks_the_quantity_the_field_measures() -> None:
    """The measured defect class: a normalization denominator deciding the owner.

    Every name below is an EMA/trend/momentum quantity whose only volatility
    content is the ATR it is divided by.  Before the repair the lexical block
    matched ``atr`` ahead of ``ema``/``slope``/``mom`` and handed all of them to
    vol_compression_encoder.  Hand-written expectations: this test fails if the
    routing arithmetic is wrong, and does not consult the module's own tables.
    """

    assert classify_entry_specialist_feature("ema20_slope_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema50_slope_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema200_slope_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema20_dist_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema50_dist_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema100_dist_atr") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("ema200_dist_atr") == "trend_ema_encoder"
    assert (
        classify_entry_specialist_feature("ema50_200_spread_atr") == "trend_ema_encoder"
    )
    assert classify_entry_specialist_feature("mom_5_atr") == "momentum_flow_encoder"
    assert classify_entry_specialist_feature("mom_20_atr") == "momentum_flow_encoder"
    assert (
        classify_entry_specialist_feature("close_open_atr")
        == "price_action_candle_encoder"
    )
    # ATR as the SUBJECT still belongs to the volatility owner.
    assert classify_entry_specialist_feature("atr_bps_14") == "vol_compression_encoder"
    assert classify_entry_specialist_feature("ctx_cont.atr_bps") == "vol_compression_encoder"
    assert classify_entry_specialist_feature("bb_width_atr") == "vol_compression_encoder"


def test_per_timeframe_lane_name_keeps_the_specialist_of_its_bare_field() -> None:
    """``m15_ema20_slope_atr_v2`` is the shipped form of ``ema20_slope_atr``.

    The per-timeframe projection emits ``f"{tf}_{output_name}_v2"``, and before
    the repair every one of these five names reached vol_compression_encoder.
    Hand-written expectations, one per emitted timeframe.
    """

    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        assert (
            classify_entry_specialist_feature(f"{timeframe}_ema20_slope_atr_v2")
            == "trend_ema_encoder"
        )
        assert (
            classify_entry_specialist_feature(f"{timeframe}_rsi14_centered_v2")
            == "momentum_flow_encoder"
        )
        assert (
            classify_entry_specialist_feature(f"{timeframe}_atr_bps_14_v2")
            == "vol_compression_encoder"
        )
        assert (
            classify_entry_specialist_feature(f"{timeframe}_ema_stack_aligned_v2")
            == "trend_ema_encoder"
        )
        assert (
            classify_entry_specialist_feature(f"{timeframe}_trend_state_age_bars_v2")
            == "trend_ema_encoder"
        )


def test_family_timeframe_lane_has_no_unclassifiable_or_contradicted_field() -> None:
    """One emitted name, one specialist, whichever lane it arrives on.

    ``adx14``, ``bb_position`` and ``mtf_smc_structure_bias`` returned
    ``unmapped`` -- a hard failure at the ranker, availability and manifest call
    sites had any of them ever entered the candidate pool.
    """

    assert classify_entry_specialist_feature("adx14") == "trend_ema_encoder"
    assert classify_entry_specialist_feature("bb_position") == "momentum_flow_encoder"
    assert (
        classify_entry_specialist_feature("mtf_smc_structure_bias")
        == "smc_liquidity_encoder"
    )
    assert classify_entry_specialist_feature("di_spread_signed") == "trend_ema_encoder"
    assert (
        classify_entry_specialist_feature("vwap_rolling5_slope_atr")
        == "session_regime_encoder"
    )
    unmapped = [
        name
        for name in MULTI_TF_PER_BAR_FEATURES_V4
        if classify_entry_specialist_feature(name)
        not in MODEL_NATIVE_TRAINING_SPECIALISTS
    ]
    assert not unmapped, unmapped
