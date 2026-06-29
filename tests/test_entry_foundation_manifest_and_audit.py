import pandas as pd
import pytest
import numpy as np
import json

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
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    SIGNAL_FIELDS,
    _build_inline_seq_structure_extension,
    _resolve_seq_structure_extension,
)
from gx1.scripts.materialize_sequence_structure_features_v1 import _requested_features


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
