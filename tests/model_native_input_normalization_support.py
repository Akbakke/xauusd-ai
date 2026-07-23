"""Deterministic immutable normalization fixture for Entry model unit tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CAT_DOMAINS,
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    EXPECTED_SURFACES,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    share_temporal_alias_stats_from_ctx,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
)


def _selection_hash(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def input_normalization_fixture(
    *,
    signal_names: list[str],
    mtf_names: list[str],
    rows: int = 128,
    per_tf_seq_lens: dict[str, int] | None = None,
    dataset_run_id: str = "MODEL_NORMALIZATION_FIXTURE_V1",
) -> dict:
    row = np.arange(rows, dtype=np.float32)
    ctx = np.column_stack(
        [
            row * np.float32(0.01 + (index + 1) / 1000.0)
            + np.float32((index % 7) * 0.1)
            for index in range(len(MODEL_NATIVE_CTX_CONT_FIELDS))
        ]
    ).astype(np.float32)
    for name in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS:
        index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(name)
        ctx[:, index] = row % 5
    signal = np.column_stack(
        [
            row * np.float32(0.02 + (index + 1) / 1000.0)
            + np.float32((index % 5) * 0.2)
            for index in range(len(signal_names))
        ]
    ).astype(np.float32)
    alias_policy = model_native_context_temporal_alias_policy(signal_names)
    for alias in alias_policy["aliases"]:
        signal[:, int(alias["signal_index"])] = ctx[
            :, int(alias["ctx_cont_index"])
        ]

    ctx_surface = fit_surface_normalization(
        ctx,
        surface="ctx_cont",
        field_names=MODEL_NATIVE_CTX_CONT_FIELDS,
        semantic_categorical_domains=CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    )
    signal_surface = fit_surface_normalization(
        signal,
        surface="signal",
        field_names=signal_names,
    )
    signal_surface = share_temporal_alias_stats_from_ctx(
        signal_surface,
        ctx_surface,
        temporal_aliases=alias_policy["aliases"],
    )
    surfaces = {
        "signal": signal_surface,
        "ctx_cont": ctx_surface,
    }
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values = np.column_stack(
            [
                row * np.float32(0.03 + (index + 1) / 100.0)
                + np.float32((index % 3) * 0.1)
                for index in range(len(mtf_names))
            ]
        ).astype(np.float32)
        semantic = {}
        if "ema_stack_aligned_v2" in mtf_names:
            values[:, mtf_names.index("ema_stack_aligned_v2")] = (
                row % 3
            ) - 1
        if "regime_class_id" in mtf_names:
            values[:, mtf_names.index("regime_class_id")] = row % 5
            semantic = MTF_SEMANTIC_CATEGORICAL_DOMAINS
        surfaces[f"mtf_{tf}"] = fit_surface_normalization(
            values,
            surface=f"mtf_{tf}",
            field_names=mtf_names,
            semantic_categorical_domains=semantic,
        )
    assert tuple(surfaces) == EXPECTED_SURFACES

    ctx_cat = np.column_stack(
        [
            np.arange(rows, dtype=np.int64) % len(domain)
            for domain in CTX_CAT_DOMAINS.values()
        ]
    )
    windows = {
        tf: {
            "left_index_inclusive": 0,
            "right_index_exclusive": rows,
            "selected_unique_row_count": rows,
            "selected_row_indices_sha256": _selection_hash(f"{tf}:indices"),
            "selected_row_values_sha256": _selection_hash(f"{tf}:values"),
            "time_min_utc": "2021-01-01T00:00:00+00:00",
            "time_max_utc": "2021-01-01T10:35:00+00:00",
        }
        for tf in ("M5", "M15", "H1", "H4", "D1")
    }
    exact_seq_lens = per_tf_seq_lens or {
        tf: 4 for tf in ("M5", "M15", "H1", "H4", "D1")
    }
    lineage = {
        "dataset_run_id": dataset_run_id,
        "train_parquet_path": "/fixture/train.parquet",
        "train_parquet_sha256": "1" * 64,
        "train_manifest_path": "/fixture/train.manifest.json",
        "train_manifest_sha256": "2" * 64,
        "train_row_count": rows,
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "train_time_min_utc": "2021-01-01T00:00:00+00:00",
        "train_time_max_utc": "2021-01-01T10:35:00+00:00",
        "m5_prebuilt_path": "/fixture/xau_m5.parquet",
        "m5_prebuilt_sha256": "3" * 64,
        "mtf_cache_manifest_path": "/fixture/mtf/manifest.json",
        "mtf_cache_manifest_sha256": "4" * 64,
        "mtf_builder_version": "fixture_v1",
        "mtf_feature_names_sha256": hashlib.sha256(
            json.dumps(
                list(mtf_names),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "per_tf_seq_lens": dict(exact_seq_lens),
        "per_tf_shift_seconds": {
            "M5": 300,
            "M15": 900,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        },
        "per_tf_fit_windows": windows,
    }
    return build_input_normalization_contract(
        fit_start_utc=lineage["train_time_min_utc"],
        fit_end_utc=lineage["train_time_max_utc"],
        surfaces=surfaces,
        ctx_cat=fit_ctx_cat_contract(
            ctx_cat,
            field_names=MODEL_NATIVE_CTX_CAT_FIELDS,
        ),
        lineage=lineage,
        temporal_aliases=alias_policy["aliases"],
    )


def input_normalization_fit_population_proof_fixture(
    normalization_contract: dict,
) -> dict:
    lineage = normalization_contract["lineage"]
    mtf_populations = {}
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        window = lineage["per_tf_fit_windows"][tf]
        population = {
            "tf": tf,
            "selection": "union_of_causal_train_windows_each_source_row_once",
            "target_availability_shift_seconds": 300,
            "tf_shift_seconds": int(lineage["per_tf_shift_seconds"][tf]),
            "seq_len": int(lineage["per_tf_seq_lens"][tf]),
            "source_row_count": int(window["right_index_exclusive"]),
            "source_warmup_rows": 0,
            **window,
        }
        population["selection_proof_sha256"] = hashlib.sha256(
            json.dumps(
                population,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        mtf_populations[tf] = population
    proof = {
        "schema_version": (
            "entry_v10_train_input_normalization_population_proof_v1"
        ),
        "fit_scope": "train_only",
        "signal_population": (
            "one_full_train_snapshot_per_unique_decision_row"
        ),
        "ctx_cont_population": "one_full_train_row_per_decision_row",
        "ctx_cat_population": "one_full_train_row_per_decision_row",
        "sequence_population": (
            "validation_only_seq_last_equals_snap_not_flattened_for_fit"
        ),
        "train_decision_row_count": int(lineage["train_row_count"]),
        "train_decision_row_indices_sha256": "6" * 64,
        "train_decision_row_values_sha256": "7" * 64,
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "temporal_alias_count": len(
            normalization_contract["temporal_aliases"]
        ),
        "temporal_aliases_sha256": normalization_contract[
            "temporal_aliases_sha256"
        ],
        "mtf_populations": mtf_populations,
    }
    proof["proof_sha256"] = hashlib.sha256(
        json.dumps(
            proof,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return proof
