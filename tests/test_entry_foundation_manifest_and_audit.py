import argparse
import pandas as pd
import pytest
import numpy as np
import json

import gx1.scripts.audit_entry_foundation_features_v1 as foundation_audit
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
    _source_field_liveness_rows,
    _stats_rows,
    _stream_split_liveness_rows,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    SIGNAL_FIELDS,
    SIGNAL_BRIDGE_ID_V3,
    _build_inline_seq_structure_extension,
    _resolve_seq_structure_extension,
    write_manifest,
)
from gx1.scripts.materialize_entry_feature_ai_inventory_v1 import SMART_LAYER_SOURCE_CONTRACTS
from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import SMART_LAYER_FEATURES
from gx1.scripts.materialize_sequence_structure_features_v1 import _requested_features
from gx1.scripts.repair_entry_seq215_manifest_provenance_v1 import run as run_seq215_manifest_repair


def test_sequence_structure_manifest_request_includes_all_foundation_features_once() -> None:
    requested, meta = _requested_features(
        ["chart.foundation_hh_state", "chart.some_promoted_feature", "chart.some_promoted_feature"],
        include_foundation_structure_features=True,
    )

    assert requested[0] == "chart.foundation_hh_state"
    assert requested[1] == "chart.some_promoted_feature"
    assert len(requested) == len(set(requested))
    assert set(FOUNDATION_STRUCTURE_FEATURE_NAMES).issubset(set(requested))
    assert meta["foundation_structure_features_required"] is True
    assert meta["foundation_structure_feature_count"] == len(FOUNDATION_STRUCTURE_FEATURE_NAMES)


def test_dataset_manifest_uses_actual_v3_ctx_and_signal_contract(tmp_path) -> None:
    manifest_path = write_manifest(
        output_path=tmp_path / "sample_train.parquet",
        build_command=["builder"],
        base28_manifest=tmp_path / "base_manifest.json",
        xgb_bundle=tmp_path / "xgb_bundle",
        tape_root=tmp_path / "tape",
        extra={
            "signal_bridge": {
                "id": SIGNAL_BRIDGE_ID_V3,
                "fields": ["p_long", "chart.foundation_hh_state"],
                "contract_sha256": "abc123",
            },
            "ctx_contract": {
                "tag": "CTX6CAT5",
                "ctx_cont_dim": 142,
                "ctx_cat_dim": 5,
                "ctx_cont_base_dim": 6,
                "ctx_cont_micro_features": ["micro_momentum_3"],
                "ctx_cont_swing_features": ["dist_last_swing_high_atr"],
            },
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_contract"]["ctx_tag"] == "CTX6CAT5"
    assert manifest["feature_contract"]["ctx_cont_dim"] == 142
    assert manifest["feature_contract"]["ctx_cat_dim"] == 5
    assert manifest["feature_contract"]["signal_bridge_id"] == SIGNAL_BRIDGE_ID_V3
    assert manifest["feature_contract"]["signal_bridge_contract_sha256"] == "abc123"
    assert manifest["feature_contract"]["signal_bridge_fields"] == ["p_long", "chart.foundation_hh_state"]


def test_seq215_manifest_repair_updates_stale_top_level_contract(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "reports"
    dataset_dir.mkdir()
    manifest = {
        "feature_contract": {
            "ctx_tag": "CTX6CAT6",
            "ctx_cont_dim": 142,
            "ctx_cat_dim": 5,
            "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V2",
        },
        "extra": {
            "signal_bridge": {
                "id": SIGNAL_BRIDGE_ID_V3,
                "fields": ["p_long"],
                "contract_sha256": "abc123",
            },
            "ctx_contract": {
                "tag": "CTX6CAT5",
                "ctx_cont_dim": 142,
                "ctx_cat_dim": 5,
                "ctx_cont_names": ["spread_bps"],
                "ctx_cat_names": ["spread_bucket"],
            },
        },
    }
    (dataset_dir / "sample_train.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dataset_dir / "DATASET_BUILD_PROOF.json").write_text(
        json.dumps({"ctx_tag": "CTX6CAT6", "ctx_cont_dim": 6, "ctx_cat_dim": 6, "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V2"}),
        encoding="utf-8",
    )

    report = run_seq215_manifest_repair(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            out_dir=str(out_dir),
            apply=True,
            quiet=True,
        )
    )

    repaired = json.loads((dataset_dir / "sample_train.manifest.json").read_text(encoding="utf-8"))
    proof = json.loads((dataset_dir / "DATASET_BUILD_PROOF.json").read_text(encoding="utf-8"))
    assert report["decision"] == "APPLIED"
    assert repaired["feature_contract"]["ctx_tag"] == "CTX6CAT5"
    assert repaired["feature_contract"]["signal_bridge_id"] == SIGNAL_BRIDGE_ID_V3
    assert repaired["extra"]["signal_bridge"]["ctx_cont_dim"] == 142
    assert proof["ctx_tag"] == "CTX6CAT5"
    assert proof["ctx_cont_dim"] == 142
    assert proof["ctx_cat_dim"] == 5


def test_foundation_audit_stats_report_liveness_and_allow_neutral_bridge_constant() -> None:
    names = [
        "chart.foundation_hh_state",
        "chart.foundation_bos_up_age_bars",
        "chart.foundation_sweep_low_reclaim_up_proxy",
        "p_long",
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
    assert by_name["p_long"]["near_constant"] is True
    assert by_name["p_long"]["constant_allowed"] is True

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


def test_inline_seq_structure_extension_fails_on_missing_requested_feature() -> None:
    merged = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")})
    for field in SIGNAL_FIELDS:
        merged[field] = 0.0

    with pytest.raises(RuntimeError, match="SEQ_STRUCTURE_INLINE_FEATURES_MISSING"):
        _build_inline_seq_structure_extension(
            merged,
            requested_features=["chart.foundation_hh_state", "chart.foundation_feature_that_does_not_exist"],
            ctx_cont_names=[],
            source_parquet=None,
        )


def test_inline_seq_structure_extension_can_materialize_all_smart_layers(tmp_path) -> None:
    periods = 12
    times = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    ctx_cont_names: set[str] = set()
    ctx_cat_names: set[str] = set()
    for contract in SMART_LAYER_SOURCE_CONTRACTS.values():
        for raw_name in tuple(contract["required_source_fields"]) + tuple(contract["optional_source_fields"]):
            name = str(raw_name)
            if name.startswith("ctx_cont."):
                ctx_cont_names.add(name.removeprefix("ctx_cont."))
            elif name.startswith("ctx_cat."):
                ctx_cat_names.add(name.removeprefix("ctx_cat."))

    data = {"time": times}
    data.update({field: np.full(periods, 0.1, dtype=np.float32) for field in SIGNAL_FIELDS})
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
        }
    ).to_parquet(source)
    requested = [name for _, features, _, _ in SMART_LAYER_FEATURES.values() for name in features]

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


def test_manifest_only_seq_structure_resolves_for_inline_mode(tmp_path) -> None:
    manifest = tmp_path / "sequence_structure_feature_layer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "sequence_structure_feature_layer_v1",
                "manifest_only": True,
                "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                "foundation_structure_missing_feature_count": 0,
                "foundation_structure_all_required_selected": True,
                "selected_features": ["chart.foundation_hh_state"],
                "parquet_path": None,
            }
        ),
        encoding="utf-8",
    )

    parquet_path, features, meta = _resolve_seq_structure_extension(
        parquet_path=None,
        manifest_path=manifest,
        allow_manifest_only_inline=True,
    )

    assert parquet_path is None
    assert features == ["chart.foundation_hh_state"]
    assert meta["manifest_only"] is True
    assert meta["foundation_structure_feature_version"] == FOUNDATION_STRUCTURE_FEATURE_VERSION
    assert meta["foundation_structure_feature_count"] == len(FOUNDATION_STRUCTURE_FEATURE_NAMES)
    assert meta["foundation_structure_missing_feature_count"] == 0
    assert meta["foundation_structure_all_required_selected"] is True
