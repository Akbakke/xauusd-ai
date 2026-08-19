from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import torch

from gx1.contracts import entry_model_native_input_normalization_v1 as normalization_contract
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CAT_DOMAINS,
    EXPECTED_SURFACES,
    MatrixPopulationPart,
    apply_surface_normalization,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    invert_surface_normalization,
    require_input_normalization_contract,
)


def test_retired_ctx_owned_temporal_alias_statistics_api_is_absent() -> None:
    assert not hasattr(
        normalization_contract,
        "share_temporal_alias_stats_from_ctx",
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
        "entry_train_decision_row_count": 96,
        "exit_train_decision_row_count": 32,
        "local_fit_row_count": 128,
        "context_fit_row_count": 128,
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


def test_robust_fit_scales_large_two_valued_and_sparse_fields_alike() -> None:
    # v7: a two-valued column has no identity exception any more. It is fitted
    # by the same median/IQR route as everything else, which is injective on
    # two points, so nothing about the evidence changes -- only the claim that
    # the fit knew the field's domain.
    fitted, _, matrix = _surface("signal")
    transformed = apply_surface_normalization(matrix, fitted)

    assert fitted["binary_mask"] == [0, 0, 0]
    assert fitted["binary_field_count"] == 0
    assert fitted["scale_source"][0] == "raw_iqr"
    assert fitted["scale_source"][1] == "raw_iqr"
    assert fitted["scale_source"][2] == "median_positive_abs_deviation"
    assert np.unique(transformed[:, 1]).size == np.unique(matrix[:, 1]).size
    order = np.argsort(matrix[:, 1], kind="stable")
    assert np.all(np.diff(transformed[order, 1]) >= 0.0)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(
        invert_surface_normalization(transformed, fitted),
        matrix,
        rtol=2e-6,
        atol=2e-6,
    )


def test_sparse_burst_uses_data_deviation_without_tail_collapse() -> None:
    # Sparse-event evidence has a zero raw IQR. The scale is fitted from the
    # observed positive deviations; no tail-rate threshold changes it.
    rng = np.random.default_rng(7)
    values = np.zeros(10_000, dtype=np.float32)
    burst = rng.uniform(0.5, 1.0, size=1_500).astype(np.float32)
    values[:1_500] = burst
    matrix = values.reshape(-1, 1)

    fitted = fit_surface_normalization(
        matrix,
        surface="signal",
        field_names=["sparse_burst"],
    )
    assert fitted["scale_source"] == ["median_positive_abs_deviation"]
    transformed = apply_surface_normalization(matrix, fitted)
    assert np.isfinite(transformed).all()
    positive_raw = matrix[:, 0] > 0.0
    assert np.unique(transformed[positive_raw, 0]).size == np.unique(
        matrix[positive_raw, 0]
    ).size
    order = np.argsort(matrix[:, 0], kind="stable")
    assert np.all(np.diff(transformed[order, 0]) >= 0.0)


def test_extreme_tails_remain_distinct_monotonic_and_invertible() -> None:
    train = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    fitted = fit_surface_normalization(
        train,
        surface="signal",
        field_names=["tail"],
    )
    runtime = np.array(
        [[-1.0e20], [-1.0e12], [0.0], [1.0e12], [1.0e20]],
        dtype=np.float32,
    )
    transformed = apply_surface_normalization(runtime, fitted)
    assert np.isfinite(transformed).all()
    assert np.all(np.diff(transformed[:, 0]) > 0.0)
    assert np.unique(transformed[:, 0]).size == len(runtime)
    np.testing.assert_allclose(
        invert_surface_normalization(transformed, fitted),
        runtime,
        rtol=3e-6,
    )


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
    # Constant-one presence masks are not granted a binary identity exception:
    # TRAIN cannot learn the effect of their absent state.
    with pytest.raises(RuntimeError, match="UNSCALEABLE"):
        fit_surface_normalization(
            np.ones((8, 1), dtype=np.float32),
            surface="mtf_d1",
            field_names=["level_present"],
        )


def test_require_rejects_a_surface_that_carries_an_inferred_binary_mask() -> None:
    # A pre-v7 surface stamped ``binary_mask`` from its fit window. Those
    # center/scale values are a window artefact, not a fitted statistic, so the
    # surface is stale immutable bundle state and must not be re-admitted. The
    # mask is checked before the stats hash, so this test fails -- on the wrong
    # error code -- if the check is removed.
    fitted, names, _ = _surface("signal")
    stale = copy.deepcopy(fitted)
    stale["binary_mask"][1] = 1
    stale["binary_field_count"] = 1
    stale["center"][1] = 0.0
    stale["scale"][1] = 1.0
    stale["scale_source"][1] = "binary_identity"
    stale["train_transformed_min"][1] = 0.0
    stale["train_transformed_max"][1] = 1.0
    with pytest.raises(RuntimeError, match="INFERRED_BINARY_MASK_FORBIDDEN"):
        normalization_contract.require_surface_normalization(
            stale,
            surface="signal",
            field_names=names,
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
    support_mutation = copy.deepcopy(contract)
    support_mutation["surfaces"]["ctx_cont"]["train_transformed_max"][0] += 1.0
    with pytest.raises(RuntimeError, match="STATS_HASH_MISMATCH"):
        require_input_normalization_contract(
            support_mutation,
            expected_field_names=names,
            expected_ctx_cat_names=list(CTX_CAT_DOMAINS),
        )
    transform_mutation = copy.deepcopy(contract)
    transform_mutation["continuous_transform"] = "clipped"
    with pytest.raises(RuntimeError, match="CONTRACT_IDENTITY_INVALID"):
        require_input_normalization_contract(
            transform_mutation,
            expected_field_names=names,
            expected_ctx_cat_names=list(CTX_CAT_DOMAINS),
        )


def test_two_valued_fit_window_never_becomes_an_immutable_binary_domain() -> None:
    # THE regression this contract version exists for.
    #
    # ``ema_stack_aligned_v2`` emits exactly {-1.0, 0.0, +1.0} by construction:
    # htf_features fills every EMA-defined row with 0.0, then writes 1 on a
    # strictly ascending EMA stack and -1 on a strictly descending one. A TRAIN
    # window containing no bear stack therefore shows only {0, +1}. The pre-v7
    # fit read that window, stamped ``binary_mask``, and the first served -1
    # raised [ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID] -- so no Entry
    # action at all, not even FLAT, could be produced in a daily downtrend.
    train = np.where(
        np.arange(256) % 3 == 0,
        1.0,
        0.0,
    ).astype(np.float32).reshape(-1, 1)
    fitted = fit_surface_normalization(
        train,
        surface="mtf_d1",
        field_names=["ema_stack_aligned_v2"],
    )
    assert fitted["binary_mask"] == [0]
    assert fitted["categorical_mask"] == [0]
    assert fitted["scale_source"] == ["raw_iqr"]

    served = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    transformed = apply_surface_normalization(served, fitted)
    assert np.isfinite(transformed).all()
    assert np.all(np.diff(transformed[:, 0]) > 0.0)
    assert np.unique(transformed[:, 0]).size == 3
    np.testing.assert_allclose(
        invert_surface_normalization(transformed, fitted),
        served,
        rtol=2e-6,
        atol=2e-6,
    )


def test_declared_domain_must_be_the_zero_based_range_the_embedding_indexes() -> None:
    # The runtime encoder builds ``nn.Embedding(len(domain), d_model)`` and
    # indexes it with the raw field value, so a signed domain is an
    # out-of-range table lookup rather than a strict-load failure. Declaring
    # one must fail at the contract, both on fit and on load.
    signed = np.array([[-1.0], [0.0], [1.0]] * 8, dtype=np.float32)
    with pytest.raises(RuntimeError, match="CATEGORICAL_FIELDS_INVALID"):
        fit_surface_normalization(
            signed,
            surface="mtf_d1",
            field_names=["signed_state"],
            semantic_categorical_domains={"signed_state": (-1, 0, 1)},
        )
    gapped = np.array([[0.0], [1.0], [3.0]] * 8, dtype=np.float32)
    with pytest.raises(RuntimeError, match="CATEGORICAL_FIELDS_INVALID"):
        fit_surface_normalization(
            gapped,
            surface="mtf_d1",
            field_names=["gapped_state"],
            semantic_categorical_domains={"gapped_state": (0, 1, 3)},
        )

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
    stored = copy.deepcopy(fitted)
    stored["categorical_domains"]["regime_class_id"] = [0, 1, 2, 3, 5]
    with pytest.raises(RuntimeError, match="CATEGORICAL_CONTRACT_INVALID"):
        normalization_contract.require_surface_normalization(
            stored,
            surface="mtf_m5",
            field_names=["momentum", "regime_class_id"],
        )


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


def test_shared_population_fit_accepts_selected_sources_without_concatenation() -> None:
    entry = np.arange(1, 41, dtype=np.float32).reshape(-1, 1)
    exit_ = np.arange(101, 181, dtype=np.float32).reshape(-1, 1)
    fitted = fit_surface_normalization(
        [
            MatrixPopulationPart(entry, source="entry_m5"),
            MatrixPopulationPart(
                exit_,
                row_indices=np.arange(0, len(exit_), 2, dtype=np.int64),
                source="exit_m1",
            ),
        ],
        surface="signal",
        field_names=["shared_local"],
        row_count=80,
        column_chunk=32,
    )
    expected = np.concatenate([entry[:, 0], exit_[::2, 0]])
    assert fitted["fit_row_count"] == 80
    assert np.float32(fitted["center"][0]) == np.float32(np.median(expected))
    assert fitted["train_transformed_min"][0] < fitted["train_transformed_max"][0]


def test_numpy_torch_asinh_transform_parity_and_source_guards() -> None:
    values = np.array(
        [
            [-1.0e12, 0.0, 0.0],
            [-2.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [3.0, 1.0, 3.0],
            [1.0e12, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    fitted = fit_surface_normalization(
        values,
        surface="ctx_cont",
        field_names=["continuous", "two_valued", "category"],
        semantic_categorical_domains={"category": (0, 1, 2, 3, 4)},
    )
    numpy_result = apply_surface_normalization(values, fitted)
    raw = torch.from_numpy(values.copy())
    center = torch.tensor(fitted["center"], dtype=torch.float32)
    scale = torch.tensor(fitted["scale"], dtype=torch.float32)
    identity = torch.tensor(
        np.logical_or(fitted["binary_mask"], fitted["categorical_mask"]),
        dtype=torch.bool,
    )
    torch_result = torch.where(
        identity,
        raw.to(dtype=torch.float64),
        torch.asinh(
            (
                raw.to(dtype=torch.float64)
                - center.to(dtype=torch.float64)
            )
            / scale.to(dtype=torch.float64)
        ),
    ).to(dtype=raw.dtype)
    np.testing.assert_allclose(
        torch_result.numpy(),
        numpy_result,
        rtol=2e-6,
        atol=2e-6,
    )
    # Only the DECLARED categorical passes through as identity. The
    # two-valued column is fitted and transformed like any other field.
    np.testing.assert_array_equal(numpy_result[:, 2], values[:, 2])
    assert not np.array_equal(numpy_result[:, 1], values[:, 1])

    contract_source = inspect.getsource(normalization_contract)
    for retired in (
        "CLIP_ABS",
        "MAX_TRAIN_CLIP_RATE",
        "SCALE_FLOOR",
        "clip_cap_quantile",
        "saturated_presence_mask_identity",
        "np.clip",
    ):
        assert retired not in contract_source
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        EntryV10CtxHybridTransformer,
    )

    model_source = inspect.getsource(
        EntryV10CtxHybridTransformer._normalize_input_surface
    )
    assert "torch.asinh(" in model_source
    assert "torch.clamp(" not in model_source


def test_asinh_affine_avoids_float32_intermediate_overflow() -> None:
    train = np.array(
        [[-2.0e-38], [-1.0e-38], [0.0], [1.0e-38], [2.0e-38]],
        dtype=np.float32,
    )
    fitted = fit_surface_normalization(
        train,
        surface="signal",
        field_names=["tiny_scale"],
    )
    runtime = np.array(
        [[-np.finfo(np.float32).max], [np.finfo(np.float32).max]],
        dtype=np.float32,
    )
    transformed = apply_surface_normalization(runtime, fitted)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] < transformed[1, 0]
    np.testing.assert_allclose(
        invert_surface_normalization(transformed, fitted),
        runtime,
        rtol=2e-5,
    )


def test_trainer_cgroup_preflight_rejects_uncapped_and_audit_without_training() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    main_source = inspect.getsource(trainer.main)
    assert main_source.index("_require_trainer_cgroup_preflight()") < (
        main_source.index("_enforce_canonical_train_env_contract()")
    )
    assert main_source.index("_require_trainer_cgroup_preflight()") < (
        main_source.index("_resolve_explicit_train_split_artifacts(")
    )

    reads: list[str] = []
    memory = 10 * 1024**3
    swap = 512 * 1024**2
    files = {
        "/proc/self/cgroup": "0::/gx1-test.scope\n",
        "/sys/fs/cgroup/gx1-test.scope/memory.max": str(memory),
        "/sys/fs/cgroup/gx1-test.scope/memory.high": str(memory),
        "/sys/fs/cgroup/gx1-test.scope/memory.swap.max": str(swap),
        "/sys/fs/cgroup/gx1-test.scope/pids.max": "64",
    }

    def read_text(path) -> str:
        key = str(path)
        reads.append(key)
        return files[key]

    base_env = {
        "GX1_CAPPED_MEMORY_BYTES": str(memory),
        "GX1_CAPPED_SWAP_BYTES": str(swap),
        "GX1_CAPPED_TASKS_MAX": "64",
    }
    with pytest.raises(RuntimeError, match="CGROUP_CLASS_INVALID"):
        trainer._require_trainer_cgroup_preflight(
            environ=base_env,
            read_text=read_text,
        )
    assert reads == []
    with pytest.raises(RuntimeError, match="CGROUP_CLASS_INVALID"):
        trainer._require_trainer_cgroup_preflight(
            environ={**base_env, "GX1_CAPPED_CLASS": "audit"},
            read_text=read_text,
        )
    assert reads == []

    proof = trainer._require_trainer_cgroup_preflight(
        environ={**base_env, "GX1_CAPPED_CLASS": "trainer"},
        read_text=read_text,
    )
    assert proof["memory_max"] == proof["memory_high"] == memory
    assert proof["swap"] == swap
    assert proof["pids"] == 64

    with pytest.raises(RuntimeError, match="ENV_ACTUAL_MISMATCH"):
        trainer._require_trainer_cgroup_preflight(
            environ={
                **base_env,
                "GX1_CAPPED_CLASS": "trainer",
                "GX1_CAPPED_MEMORY_BYTES": str(9 * 1024**3),
            },
            read_text=read_text,
        )
