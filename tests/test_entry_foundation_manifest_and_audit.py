import pandas as pd
import pytest
import numpy as np
import json

import gx1.scripts.audit_entry_foundation_features_v1 as foundation_audit
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
)
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.scripts.audit_entry_foundation_features_v1 import (
    REQUIRED_FOUNDATION_LIVENESS_FAMILIES,
    REQUIRED_FOUNDATION_OBJECTIVE_FEATURES,
    _family_liveness,
    _objective_liveness,
    _objective_coverage,
    _required_objective_liveness_failures,
    _required_family_liveness_failures,
    _required_source_field_liveness_failures,
    _is_neutral_constant_allowed,
    _load_emitted_contract,
    _source_field_liveness_rows,
    _stats_rows,
    _stream_split_liveness_rows,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _build_inline_seq_structure_extension,
    _normalize_time_utc,
    _resolve_seq_structure_extension,
    write_manifest,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_SPECIALIST_LAYER_FEATURES,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _selected_with_foundation(*, prefix: str) -> list[str]:
    selected = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    selected.extend(
        name for name in FOUNDATION_STRUCTURE_FEATURE_NAMES if name not in set(selected)
    )
    selected.extend(
        f"session_regime.{prefix}_{index:03d}"
        for index in range(MODEL_NATIVE_SELECTED_FEATURE_COUNT - len(selected))
    )
    return selected


def test_foundation_audit_loads_exact_model_native_emitted_contract(tmp_path) -> None:
    selected = _selected_with_foundation(prefix="audit_selected")
    signal_contract = model_native_signal_contract_metadata(selected)
    parquet = tmp_path / "entry_train.parquet"
    parquet.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "extra": {
                    "neutral_xgb_bridge": False,
                    "model_native_signal_contract": signal_contract,
                    "signal_bridge": {
                        "fields": signal_contract["fields"],
                        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        "seq_structure_extension_dim": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
                        "seq_structure_extension_v1": {
                            "features": selected,
                            "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                            "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                            "foundation_structure_all_required_selected": True,
                        },
                    },
                    "ctx_contract": {
                        "ctx_cont_names": list(ORDERED_CTX_CONT_NAMES_V3),
                        "ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_V3),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    emitted = _load_emitted_contract(parquet)

    assert emitted["signal_fields"] == signal_contract["fields"]
    assert emitted["seq_input_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert emitted["seq_structure_extension_dim"] == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert emitted["neutral_xgb_bridge"] is False


def test_xau_regime_agreement_feature_can_be_split_constant() -> None:
    assert _is_neutral_constant_allowed("session_regime.h4_d1_regime_sign_agreement") is True
    assert _is_neutral_constant_allowed("session_regime.some_other_context") is False


def test_dataset_manifest_uses_actual_v3_ctx_and_signal_contract(tmp_path) -> None:
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields()
    )
    manifest_path = write_manifest(
        output_path=tmp_path / "sample_train.parquet",
        build_command=["builder"],
        base28_manifest=tmp_path / "base_manifest.json",
        source_parquet_override=None,
        tape_root=tmp_path / "tape",
        extra={
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "neutral_xgb_bridge": False,
            "model_native_signal_contract": signal_contract,
            "signal_bridge": {
                "id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
                "fields": list(signal_contract["fields"]),
                "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "bridge_dim": 0,
                "bridge_source": None,
            },
            "ctx_contract": {
                "tag": "CTX6CAT5",
                "ctx_cont_dim": 142,
                "ctx_cat_dim": 5,
                "ctx_cont_names": list(ORDERED_CTX_CONT_NAMES_V3),
                "ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_V3),
                "ctx_cont_micro_features": ["micro_momentum_3"],
                "ctx_cont_swing_features": ["dist_last_swing_high_atr"],
            },
            "model_native_state_contract": {
                "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
                "feature_history_start_utc": "2020-11-01T00:00:00Z",
                "rank_fit_start_utc": "2020-11-09T00:00:00Z",
                "rank_fit_end_utc": "2025-09-30T23:59:59Z",
                "rank_reference_npz": "/immutable/rank_reference.npz",
                "rank_reference_npz_sha256": "a" * 64,
                "rank_reference_schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
                "rank_reference_sidecar_json": "/immutable/rank_reference.npz.json",
                "rank_reference_source_parquet": "/immutable/source.parquet",
                "rank_reference_source_parquet_sha256": "b" * 64,
                "rank_reference_fit_row_count": 123,
                "normalization_fit_scope": "train_only",
                "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
                "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
                "split_reset_allowed": False,
                "post_fit_rows_in_rank_reference": False,
                "runtime_rule_free": True,
                "explicit_vedtak_id": "MODEL_NATIVE_FOUNDATION_MANIFEST_PYTEST",
            },
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_contract"]["ctx_tag"] == "CTX6CAT5"
    assert manifest["feature_contract"]["ctx_cont_dim"] == 142
    assert manifest["feature_contract"]["ctx_cat_dim"] == 5
    assert manifest["feature_contract"]["signal_bridge_id"] == MODEL_NATIVE_SIGNAL_SCHEMA_VERSION
    assert manifest["feature_contract"]["signal_bridge_contract_sha256"] == signal_contract["static_contract_sha256"]
    assert manifest["feature_contract"]["signal_bridge_fields"] == signal_contract["fields"]
    assert not any("xgb" in key.lower() for key in manifest["inputs"])


def test_foundation_audit_stats_report_liveness_and_allow_proven_split_constant() -> None:
    names = [
        "chart.foundation_hh_state",
        "chart.foundation_bos_up_age_bars",
        "chart.foundation_sweep_low_reclaim_up_proxy",
        "session_regime.h4_d1_regime_sign_agreement",
    ]
    matrix = np.asarray(
        [
            [0.0, 96.0, 0.0, 1.0 / 3.0],
            [0.5, 0.0, 1.2, 1.0 / 3.0],
            [1.0, 1.0, 0.0, 1.0 / 3.0],
        ],
        dtype=np.float32,
    )

    rows = _stats_rows(
        matrix,
        names,
        split="train",
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
    )
    by_name = {row["feature"]: row for row in rows}
    assert by_name["chart.foundation_hh_state"]["family"] == "foundation_hh_hl_lh_ll"
    assert by_name["chart.foundation_bos_up_age_bars"]["family"] == "foundation_bos_choch_age"
    assert by_name["chart.foundation_sweep_low_reclaim_up_proxy"]["family"] == "foundation_sweep_reclaim"
    allowed = by_name["session_regime.h4_d1_regime_sign_agreement"]
    assert allowed["near_constant"] is True
    assert allowed["constant_allowed"] is True

    family_rows = _family_liveness(rows)
    families = {row["family"] for row in family_rows}
    assert "foundation_hh_hl_lh_ll" in families
    assert "foundation_bos_choch_age" in families
    assert "foundation_sweep_reclaim" in families


def test_required_foundation_family_liveness_is_checked_per_split() -> None:
    rows = []
    for split in ("train", "val"):
        for family in REQUIRED_FOUNDATION_LIVENESS_FAMILIES:
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "feature_count": 2,
                    "mean_active_rate": 0.20,
                    "min_active_rate": 0.10,
                    "near_constant_count": 0,
                    "nonfinite_count": 0,
                }
            )

    assert (
        _required_family_liveness_failures(
            rows,
            splits=["train", "val"],
            required_families=REQUIRED_FOUNDATION_LIVENESS_FAMILIES,
            min_mean_active_rate=0.01,
        )
        == []
    )

    bad_rows = [
        row
        for row in rows
        if not (row["split"] == "val" and row["family"] == "foundation_sweep_reclaim")
    ]
    bad_rows.append(
        {
            "split": "train",
            "family": "foundation_bos_choch_age",
            "feature_count": 2,
            "mean_active_rate": 0.0,
            "min_active_rate": 0.0,
            "near_constant_count": 0,
            "nonfinite_count": 0,
        }
    )

    failures = _required_family_liveness_failures(
        bad_rows,
        splits=["train", "val"],
        required_families=REQUIRED_FOUNDATION_LIVENESS_FAMILIES,
        min_mean_active_rate=0.01,
    )

    assert any("required foundation liveness family missing: foundation_sweep_reclaim" in item for item in failures)
    assert any("foundation_bos_choch_age" in item and "active rate too low" in item for item in failures)


def test_foundation_objective_liveness_is_checked_per_split() -> None:
    names = [
        "chart.foundation_hh_state",
        "chart.foundation_hl_state",
        "chart.foundation_lh_state",
        "chart.foundation_ll_state",
        "chart.foundation_structure_up_minus_down",
    ]
    matrix = np.asarray(
        [
            [0.0, 0.2, 0.0, 0.1, 0.1],
            [0.5, 0.3, 0.1, 0.0, 0.7],
            [1.0, 0.4, 0.2, 0.0, 1.2],
        ],
        dtype=np.float32,
    )
    rows = _stats_rows(
        matrix,
        names,
        split="train",
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
    )

    objective_rows = _objective_liveness(rows)
    by_objective = {row["objective"]: row for row in objective_rows}

    assert by_objective["hh_hl_lh_ll"]["observed_count"] == 5
    assert by_objective["hh_hl_lh_ll"]["missing_count"] == 0
    failures = _required_objective_liveness_failures(
        objective_rows,
        splits=["train"],
        required_objectives=("hh_hl_lh_ll",),
        min_mean_active_rate=0.01,
    )
    assert failures == []

    missing_failures = _required_objective_liveness_failures(
        objective_rows,
        splits=["train"],
        required_objectives=("bos_choch_age",),
        min_mean_active_rate=0.01,
    )
    assert any("required foundation objective has missing live features: bos_choch_age" in item for item in missing_failures)


def test_foundation_source_field_liveness_is_checked_per_split() -> None:
    signal_fields = [
        field.split(".", 1)[1]
        for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
        if field.startswith("snap.")
    ]
    ctx_cont_names = [
        field.split(".", 1)[1]
        for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
        if field.startswith("ctx_cont.")
    ]
    snap = np.zeros((4, len(signal_fields)), dtype=np.float32)
    ctx_cont = np.zeros((4, len(ctx_cont_names)), dtype=np.float32)
    snap[:, :] = np.arange(4, dtype=np.float32)[:, None]
    ctx_cont[:, :] = np.arange(4, dtype=np.float32)[:, None]

    rows = _source_field_liveness_rows(
        snap=snap,
        ctx_cont=ctx_cont,
        signal_fields=signal_fields,
        ctx_cont_names=ctx_cont_names,
        split="train",
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
        min_active_rate=0.0001,
        min_active_count=1,
    )

    assert len(rows) == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    assert all(row["observed"] for row in rows)
    assert all(row["live"] for row in rows)
    assert (
        _required_source_field_liveness_failures(
            rows,
            splits=["train"],
            required_source_fields=FOUNDATION_STRUCTURE_SOURCE_FIELDS,
            min_active_rate=0.0001,
            min_active_count=1,
        )
        == []
    )

    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["active_count"] = 0
    bad_rows[0]["active_rate"] = 0.0
    bad_rows[1]["near_constant"] = True
    failures = _required_source_field_liveness_failures(
        bad_rows,
        splits=["train", "val"],
        required_source_fields=FOUNDATION_STRUCTURE_SOURCE_FIELDS,
        min_active_rate=0.0001,
        min_active_count=1,
    )

    assert any("active count too low" in item for item in failures)
    assert any("near-constant" in item for item in failures)
    assert any("required foundation source-field liveness missing" in item for item in failures)


def _assert_liveness_rows_match(
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
    *,
    key: str,
) -> None:
    assert len(actual) == len(expected)
    actual_by_key = {str(row[key]): row for row in actual}
    for expected_row in expected:
        actual_row = actual_by_key[str(expected_row[key])]
        assert set(actual_row) == set(expected_row)
        for field, expected_value in expected_row.items():
            actual_value = actual_row[field]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
            else:
                assert actual_value == expected_value


def test_stream_split_liveness_matches_matrix_helpers(tmp_path) -> None:
    audit_features = [
        "chart.foundation_hh_state",
        "chart.foundation_bos_up_age_bars",
        "p_long",
    ]
    snap_source_names = [
        field.split(".", 1)[1]
        for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
        if field.startswith("snap.")
    ]
    signal_fields = list(dict.fromkeys(audit_features + snap_source_names))
    ctx_cont_names = list(
        dict.fromkeys(
            field.split(".", 1)[1]
            for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
            if field.startswith("ctx_cont.")
        )
    )
    row_count = 5
    snap = (
        np.arange(row_count * len(signal_fields), dtype=np.float32)
        .reshape(row_count, len(signal_fields))
        / np.float32(10.0)
    )
    ctx_cont = (
        np.arange(row_count * len(ctx_cont_names), dtype=np.float32)
        .reshape(row_count, len(ctx_cont_names))
        / np.float32(7.0)
    )
    snap[:, signal_fields.index("p_long")] = np.float32(1.0 / 3.0)
    snap[1, signal_fields.index("chart.foundation_bos_up_age_bars")] = np.nan
    if ctx_cont_names:
        ctx_cont[2, 0] = np.inf
    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "snap": [row.astype(np.float32) for row in snap],
            "ctx_cont": [row.astype(np.float32) for row in ctx_cont],
        }
    ).to_parquet(parquet_path, index=False)

    streamed_stats, streamed_source_rows = _stream_split_liveness_rows(
        parquet_path,
        split="train",
        signal_fields=signal_fields,
        ctx_cont_names=ctx_cont_names,
        audit_features=audit_features,
        batch_size=2,
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
        min_source_active_rate=0.0001,
        min_source_active_count=1,
    )
    audit_cols = [signal_fields.index(name) for name in audit_features]
    expected_stats = _stats_rows(
        snap[:, audit_cols],
        audit_features,
        split="train",
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
    )
    expected_source_rows = _source_field_liveness_rows(
        snap=snap,
        ctx_cont=ctx_cont,
        signal_fields=signal_fields,
        ctx_cont_names=ctx_cont_names,
        split="train",
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
        min_active_rate=0.0001,
        min_active_count=1,
    )

    _assert_liveness_rows_match(streamed_stats, expected_stats, key="feature")
    _assert_liveness_rows_match(streamed_source_rows, expected_source_rows, key="source_field")


def test_stream_split_source_scan_avoids_stacked_source_matrix(tmp_path, monkeypatch) -> None:
    audit_features = ["chart.foundation_hh_state"]
    snap_source_names = [
        field.split(".", 1)[1]
        for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
        if field.startswith("snap.")
    ]
    signal_fields = list(dict.fromkeys(audit_features + snap_source_names))
    ctx_cont_names = list(
        dict.fromkeys(
            field.split(".", 1)[1]
            for field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
            if field.startswith("ctx_cont.")
        )
    )
    row_count = 3
    snap = (
        np.arange(row_count * len(signal_fields), dtype=np.float32)
        .reshape(row_count, len(signal_fields))
        + np.float32(1.0)
    )
    ctx_cont = (
        np.arange(row_count * len(ctx_cont_names), dtype=np.float32)
        .reshape(row_count, len(ctx_cont_names))
        + np.float32(1.0)
    )
    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "snap": [row.astype(np.float32) for row in snap],
            "ctx_cont": [row.astype(np.float32) for row in ctx_cont],
        }
    ).to_parquet(parquet_path, index=False)

    original_add = foundation_audit._StreamingStatsAccumulator.add

    def _forbid_source_matrix_add(self, values):
        if self.names == list(FOUNDATION_STRUCTURE_SOURCE_FIELDS):
            raise AssertionError("source scan should not materialize a stacked source matrix")
        return original_add(self, values)

    monkeypatch.setattr(foundation_audit._StreamingStatsAccumulator, "add", _forbid_source_matrix_add)

    streamed_stats, streamed_source_rows = foundation_audit._stream_split_liveness_rows(
        parquet_path,
        split="train",
        signal_fields=signal_fields,
        ctx_cont_names=ctx_cont_names,
        audit_features=audit_features,
        batch_size=2,
        liveness_epsilon=1e-7,
        near_constant_std=1e-9,
        min_source_active_rate=0.0001,
        min_source_active_count=1,
    )

    assert len(streamed_stats) == len(audit_features)
    assert len(streamed_source_rows) == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    assert all(row["n"] == row_count for row in streamed_source_rows)


def test_foundation_objective_coverage_requires_exact_goal_features() -> None:
    rows, failures = _objective_coverage(list(FOUNDATION_STRUCTURE_FEATURE_NAMES))

    assert failures == []
    by_objective = {row["objective"]: row for row in rows}
    assert set(by_objective) == set(REQUIRED_FOUNDATION_OBJECTIVE_FEATURES)
    assert by_objective["hh_hl_lh_ll"]["present_count"] == 5
    assert by_objective["bos_choch_age"]["present_count"] == 8
    assert by_objective["sweep_reclaim_false_breakout"]["present_count"] == 5
    assert by_objective["compression_expansion"]["present_count"] == 5
    assert by_objective["impulse_pullback_phase"]["present_count"] == 6
    assert by_objective["session_x_structure"]["present_count"] == 28

    missing_session = [
        name
        for name in FOUNDATION_STRUCTURE_FEATURE_NAMES
        if name != "chart.foundation_overlap_x_sweep_reclaim_balance"
    ]
    bad_rows, bad_failures = _objective_coverage(missing_session)
    bad_session = next(row for row in bad_rows if row["objective"] == "session_x_structure")

    assert bad_session["missing"] == ["chart.foundation_overlap_x_sweep_reclaim_balance"]
    assert any("foundation objective coverage missing session_x_structure" in item for item in bad_failures)


def test_inline_seq_structure_extension_requires_explicit_source_parquet() -> None:
    merged = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")})
    for field in MODEL_NATIVE_BASE_FIELDS:
        merged[field] = 0.0

    with pytest.raises(RuntimeError, match="SEQ_STRUCTURE_INLINE_SOURCE_PARQUET_REQUIRED"):
        _build_inline_seq_structure_extension(
            merged,
            requested_features=["chart.foundation_hh_state", "chart.foundation_feature_that_does_not_exist"],
            ctx_cont_names=[],
            source_parquet=None,
        )


def test_dataset_builder_time_normalization_rejects_duplicates() -> None:
    duplicate = pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ]
        }
    )
    with pytest.raises(RuntimeError, match="TIME_DUPLICATE_ROWS"):
        _normalize_time_utc(duplicate, "time")


def test_inline_seq_structure_extension_can_materialize_all_smart_layers(tmp_path) -> None:
    periods = 12
    times = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    ctx_cont_names = set(ORDERED_CTX_CONT_NAMES_V3)
    ctx_cat_names = set(ORDERED_CTX_CAT_NAMES_V3)

    data = {"time": times}
    data.update({field: np.full(periods, 0.1, dtype=np.float32) for field in MODEL_NATIVE_BASE_FIELDS})
    data.update({field: np.full(periods, 0.2, dtype=np.float32) for field in ctx_cont_names})
    data.update({field: np.ones(periods, dtype=np.int64) for field in ctx_cat_names})
    merged = pd.DataFrame(data)
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "time": times,
            "open": np.linspace(1.0, 1.1, periods),
            "high": np.linspace(1.01, 1.11, periods),
            "low": np.linspace(0.99, 1.09, periods),
            "close": np.linspace(1.005, 1.105, periods),
            "mid": np.linspace(1.005, 1.105, periods),
            "_v1_atr14": np.full(periods, 0.01),
        }
    ).to_parquet(source)
    requested = [
        name
        for _, features in MODEL_NATIVE_SPECIALIST_LAYER_FEATURES
        for name in features
    ]

    out, names, meta = _build_inline_seq_structure_extension(
        merged,
        requested_features=requested,
        ctx_cont_names=sorted(ctx_cont_names),
        ctx_cat_names=sorted(ctx_cat_names),
        source_parquet=source,
    )

    assert out.shape == (periods, len(requested))
    assert names == requested
    assert np.isfinite(out).all()
    assert meta["feature_count"] == len(requested)
    assert meta["smart_generated_dim"] > 0
    assert {row["label"] for row in meta["smart_generated_layers"]} >= {
        "trend_ema_smart_layer",
        "smc_liquidity_quality_layer",
        "structure_swing_derivation_layer",
        "momentum_flow_smart_layer",
        "session_regime_interaction_layer",
        "vol_compression_smart_layer",
        "support_resistance_memory_layer",
        "mtf_confluence_layer",
    }


def test_exact_model_native_manifest_resolves_mandatory_inline_mode(tmp_path) -> None:
    selected = _selected_with_foundation(prefix="inline_selected")
    manifest = tmp_path / "sequence_structure_feature_layer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "entry_specialist_challenger_extension_manifest_v1",
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "base_signal_feature_count": 34,
                "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
                "manifest_only": True,
                "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                "foundation_structure_missing_feature_count": 0,
                "foundation_structure_all_required_selected": True,
                "selected_features": selected,
                "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
                "model_native_signal_contract": model_native_signal_contract_metadata(selected),
            }
        ),
        encoding="utf-8",
    )

    features, meta = _resolve_seq_structure_extension(manifest_path=manifest)

    assert features == selected
    assert meta["mode"] == "mandatory_inline_common_causal_history_v1"
    assert meta["foundation_structure_feature_version"] == FOUNDATION_STRUCTURE_FEATURE_VERSION
    assert meta["foundation_structure_feature_count"] == len(FOUNDATION_STRUCTURE_FEATURE_NAMES)
    assert meta["foundation_structure_missing_feature_count"] == 0
    assert meta["foundation_structure_all_required_selected"] is True
