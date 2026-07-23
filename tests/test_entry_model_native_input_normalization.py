from __future__ import annotations

import copy

import numpy as np
import pytest

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CAT_DOMAINS,
    EXPECTED_SURFACES,
    apply_surface_normalization,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    require_input_normalization_contract,
)


def _surface(name: str, *, width: int = 3) -> tuple[dict, list[str], np.ndarray]:
    names = [f"{name}_{index}" for index in range(width)]
    rows = np.arange(1, 129, dtype=np.float32)
    matrix = np.column_stack(
        [
            rows * 1000.0,
            (rows % 2 == 0).astype(np.float32),
            np.where(rows % 31 == 0, rows * 0.25, 0.0),
        ][:width]
    ).astype(np.float32)
    return (
        fit_surface_normalization(
            matrix,
            surface=name,
            field_names=names,
            column_chunk=2,
        ),
        names,
        matrix,
    )


def _contract() -> tuple[dict, dict[str, list[str]], dict[str, np.ndarray]]:
    surfaces = {}
    names = {}
    matrices = {}
    for surface in EXPECTED_SURFACES:
        fitted, field_names, matrix = _surface(surface)
        surfaces[surface] = fitted
        names[surface] = field_names
        matrices[surface] = matrix
    tf_lens = {tf: 96 for tf in ("M5", "M15", "H1", "H4", "D1")}
    tf_shifts = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
    tf_windows = {
        tf: {
            "left_index_inclusive": 10,
            "right_index_exclusive": 138,
            "selected_unique_row_count": 128,
            "selected_row_indices_sha256": "6" * 64,
            "selected_row_values_sha256": "7" * 64,
            "time_min_utc": "2021-01-04T00:00:00+00:00",
            "time_max_utc": "2026-05-31T23:55:00+00:00",
        }
        for tf in tf_lens
    }
    lineage = {
        "dataset_run_id": "NORMALIZATION_TEST_RUN",
        "train_parquet_path": "/immutable/train.parquet",
        "train_parquet_sha256": "1" * 64,
        "train_manifest_path": "/immutable/train.manifest.json",
        "train_manifest_sha256": "2" * 64,
        "train_row_count": 128,
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "train_time_min_utc": "2021-01-05T00:00:00+00:00",
        "train_time_max_utc": "2026-05-31T23:59:59+00:00",
        "m5_prebuilt_path": "/immutable/xauusd_m5.parquet",
        "m5_prebuilt_sha256": "3" * 64,
        "mtf_cache_manifest_path": "/immutable/mtf/manifest.json",
        "mtf_cache_manifest_sha256": "4" * 64,
        "mtf_builder_version": "test_mtf_v2",
        "mtf_feature_names_sha256": "5" * 64,
        "per_tf_seq_lens": tf_lens,
        "per_tf_shift_seconds": tf_shifts,
        "per_tf_fit_windows": tf_windows,
    }
    ctx_rows = np.column_stack(
        [
            np.arange(128) % len(domain)
            for domain in CTX_CAT_DOMAINS.values()
        ]
    )
    return (
        build_input_normalization_contract(
            fit_start_utc="2021-01-05T00:00:00+00:00",
            fit_end_utc="2026-05-31T23:59:59+00:00",
            surfaces=surfaces,
            ctx_cat=fit_ctx_cat_contract(
                ctx_rows,
                field_names=list(CTX_CAT_DOMAINS),
            ),
            lineage=lineage,
            temporal_aliases=[],
        ),
        names,
        matrices,
    )


def test_robust_fit_preserves_binary_and_scales_large_and_sparse_fields() -> None:
    fitted, _, matrix = _surface("signal")
    transformed = apply_surface_normalization(matrix, fitted)

    assert fitted["binary_mask"] == [0, 1, 0]
    assert fitted["scale_source"][0] == "iqr"
    assert fitted["scale_source"][1] == "binary_identity"
    assert fitted["scale_source"][2] == "median_positive_abs_deviation"
    np.testing.assert_array_equal(transformed[:, 1], matrix[:, 1])
    assert np.isfinite(transformed).all()
    assert float(np.max(np.abs(transformed))) <= 12.0


def test_fit_rejects_nonfinite_and_constant_nonbinary_fields() -> None:
    with pytest.raises(RuntimeError, match="NONFINITE"):
        fit_surface_normalization(
            np.array([[1.0], [np.nan]], dtype=np.float32),
            surface="signal",
            field_names=["bad"],
        )
    with pytest.raises(RuntimeError, match="UNSCALEABLE"):
        fit_surface_normalization(
            np.full((8, 1), 2.0, dtype=np.float32),
            surface="signal",
            field_names=["constant"],
        )


def test_contract_binds_all_surface_names_stats_and_fit_lineage() -> None:
    contract, names, _ = _contract()

    assert require_input_normalization_contract(
        contract,
        expected_field_names=names,
        expected_ctx_cat_names=list(CTX_CAT_DOMAINS),
    ) == contract
    assert len(contract["contract_sha256"]) == 64
    assert tuple(contract["surfaces"]) == EXPECTED_SURFACES

    tampered = copy.deepcopy(contract)
    tampered["surfaces"]["ctx_cont"]["scale"][0] *= 2.0
    with pytest.raises(RuntimeError, match="STATS_HASH_MISMATCH"):
        require_input_normalization_contract(
            tampered,
            expected_field_names=names,
            expected_ctx_cat_names=list(CTX_CAT_DOMAINS),
        )


def test_binary_contract_rejects_unseen_nonbinary_runtime_value() -> None:
    fitted, _, matrix = _surface("signal")
    changed = matrix.copy()
    changed[0, 1] = 0.5
    with pytest.raises(RuntimeError, match="BINARY_VALUE_INVALID"):
        apply_surface_normalization(changed, fitted)


def test_semantic_categorical_field_is_domain_checked_and_not_scaled() -> None:
    values = np.column_stack(
        [
            np.arange(25, dtype=np.float32),
            np.arange(25, dtype=np.float32) % 5,
        ]
    )
    fitted = fit_surface_normalization(
        values,
        surface="mtf_m5",
        field_names=["momentum", "regime_class_id"],
        semantic_categorical_domains={"regime_class_id": (0, 1, 2, 3, 4)},
    )
    transformed = apply_surface_normalization(values, fitted)

    assert fitted["categorical_mask"] == [0, 1]
    np.testing.assert_array_equal(transformed[:, 1], values[:, 1])
    changed = values.copy()
    changed[0, 1] = 5.0
    with pytest.raises(RuntimeError, match="CATEGORICAL_VALUE_INVALID"):
        apply_surface_normalization(changed, fitted)
