from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleSplit
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
    fit_surface_normalization,
    require_input_normalization_contract,
    FIT_POPULATION, DYNAMIC_M5_FIT_POPULATION,
)
from gx1.features import gc_native_context_v1 as gc_context
from gx1.features.micro_structure_v1 import range_observation_binary_fields, range_observation_masking_pairs
from gx1.features.basic_v1 import session_native_binary_event_fields
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, JOINT_FITTED_Q_TRAINING_PHASE,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
    MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4,
    MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4,
    MULTI_TF_RESAMPLE_RULES,
    build_multi_tf_v4_closed_timestamp_indices,
    load_multi_tf_v4_cache,
)
from gx1.models.entry_v10 import entry_v10_input_normalization as normalization
from gx1.models.entry_v10.entry_v10_input_normalization import (
    TrainNormalizationArtifacts,
    _validate_source_backed_train_inputs,
    fit_entry_v10_train_input_normalization,
    select_causal_mtf_fit_population,
)
from gx1.scripts.prebuild_multi_tf_cache_v4 import publish_multi_tf_v4_cache
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)

from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)

_V29_TEST_REGISTRY_CONSTANTS = synthetic_v29_registry_constants()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_names() -> list[str]:
    names = [f"test.signal_{index:03d}" for index in range(MODEL_NATIVE_SIGNAL_DIM)]
    names[7] = "ctx_cont.atr_bps"
    names[-1] = "ctx_cont.spread_bps"
    # The one declared semantic categorical on the signal surface must be
    # present so its exact integer domain is fitted, not standardized.
    for position, name in enumerate(SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS):
        names[3 + position] = name
    return names


def _semantic_categorical_signal_indices(signal_names: list[str]) -> list[int]:
    return [
        signal_names.index(name) for name in SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS
    ]


def _fill_semantic_categoricals(
    matrix: np.ndarray,
    signal_names: list[str],
    *,
    axis_rows: int,
) -> None:
    """Write exact in-domain integer values into every categorical column.

    A 2D snapshot/decision matrix gets one domain value per row; a 3D sequence
    matrix repeats each row's value across every step, so the last sequence
    step stays bit-identical to its snapshot.
    """
    for index, name in zip(
        _semantic_categorical_signal_indices(signal_names),
        SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
    ):
        domain = np.asarray(
            SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS[name], dtype=np.float32
        )
        values = domain[np.arange(axis_rows) % len(domain)]
        if matrix.ndim == 3:
            matrix[:, :, index] = values[:, None]
        else:
            matrix[:, index] = values


def _valid_observation_storage(values: np.ndarray, names) -> None:
    positions = {name: i for i, name in enumerate(names)}
    row = np.arange(len(values))
    binary = (gc_context.binary_fields(names) | range_observation_binary_fields(names)
              | session_native_binary_event_fields(names)
              | set(MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4).intersection(names))
    for name in binary:
        # Preserve the distinct source-valid patterns already supplied by MTF fixtures.
        if not np.isin(values[:, positions[name]], (0.0, 1.0)).all():
            values[:, positions[name]] = row % 2
    for vi, mi in (*gc_context.masking_pairs(names), *range_observation_masking_pairs(names)):
        values[values[:, mi] == 0, vi] = 0


def _mtf_values(rows: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.float32)[:, None]
    column = np.arange(len(MULTI_TF_PER_BAR_FEATURES_V4), dtype=np.float32)[
        None, :
    ]
    values = (
        row * (np.float32(0.25) + (column % 7.0) * np.float32(0.01))
        + np.sin((row + column) * np.float32(0.17)).astype(np.float32)
    ).astype(np.float32)
    ema_stack = list(MULTI_TF_PER_BAR_FEATURES_V4).index("ema_stack_aligned_v2")
    values[:, ema_stack] = (np.arange(rows) % 3 - 1).astype(np.float32)
    for offset, name in enumerate(MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4):
        index = list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)
        values[:, index] = (
            np.arange(rows) % (offset + 2) == offset
        ).astype(np.float32)
    # 2026-09-21 (F-18): every declared MTF semantic categorical must carry
    # exact in-domain integers, mirroring the signal-surface convention.
    for name, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
        position = list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)
        values[:, position] = np.asarray(domain, dtype=np.float32)[
            np.arange(rows) % len(domain)
        ]
    _valid_observation_storage(values, MULTI_TF_PER_BAR_FEATURES_V4)
    return values


def _mtf_frame(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    *,
    warmup_rows: int = 0,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        np.asarray(timestamps_ns, dtype=np.int64).astype("datetime64[ns]"),
        tz="UTC",
    )
    verified = np.ascontiguousarray(values, dtype=np.float32)
    frame = pd.DataFrame(
        verified,
        index=index,
        columns=MULTI_TF_PER_BAR_FEATURES_V4,
        copy=False,
    )
    frame.attrs["ts_int64"] = np.ascontiguousarray(
        timestamps_ns, dtype=np.int64
    )
    frame.attrs["feats_np"] = verified
    frame.attrs["causal_warmup_rows"] = int(warmup_rows)
    frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
    return frame


def _train_arrays(
    *,
    rows: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    row = np.arange(rows, dtype=np.float32)[:, None]
    signal_column = np.arange(MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32)[None, :]
    snap = (
        row * (np.float32(0.5) + (signal_column % 11.0) * np.float32(0.01))
        + signal_column * np.float32(0.001)
    ).astype(np.float32)

    ctx_column = np.arange(len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.float32)[
        None, :
    ]
    ctx_cont = (
        row * (np.float32(0.3) + (ctx_column % 13.0) * np.float32(0.01))
        + ctx_column * np.float32(0.002)
    ).astype(np.float32)
    signal_names = _signal_names()
    _valid_observation_storage(ctx_cont, MODEL_NATIVE_CTX_CONT_FIELDS)
    _fill_semantic_categoricals(snap, signal_names, axis_rows=rows)
    for field in ("atr_bps", "spread_bps"):
        snap[:, signal_names.index(f"ctx_cont.{field}")] = ctx_cont[
            :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
        ]

    # Deliberately make earlier temporal rows unlike snapshots.  They are
    # validated but must never be flattened into the signal fit population.
    seq = np.empty(
        (rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM),
        dtype=np.float32,
    )
    seq[:, :-1, :] = snap[:, None, :] - np.float32(50.0)
    seq[:, -1, :] = snap
    for field in ("atr_bps", "spread_bps"):
        seq[:, :-1, signal_names.index(f"ctx_cont.{field}")] = 0.0
    _fill_semantic_categoricals(seq, signal_names, axis_rows=rows)

    ctx_cat = np.column_stack(
        [
            (np.arange(rows) + offset) % 4
            for offset in range(len(MODEL_NATIVE_CTX_CAT_FIELDS))
        ]
    ).astype(np.int64)
    times = pd.date_range(
        "2026-02-15T00:00:00Z", periods=rows, freq="5min", tz="UTC"
    )
    return seq, snap, ctx_cont, ctx_cat, times


class _FakeTrainExitLifecycle:
    def __init__(self, signal_names: list[str]) -> None:
        rows = 600
        row = np.arange(rows, dtype=np.float32)[:, None]
        signal_column = np.arange(MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32)[
            None, :
        ]
        self.signal = (
            row * (np.float32(0.07) + signal_column % 9 * np.float32(0.003))
            + signal_column * np.float32(0.0007)
        ).astype(np.float32)
        ctx_column = np.arange(
            len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.float32
        )[None, :]
        self.ctx_cont = (
            row * (np.float32(0.05) + ctx_column % 11 * np.float32(0.002))
            + ctx_column * np.float32(0.0011)
        ).astype(np.float32)
        _valid_observation_storage(self.ctx_cont, MODEL_NATIVE_CTX_CONT_FIELDS)
        _fill_semantic_categoricals(self.signal, signal_names, axis_rows=rows)
        for field in ("atr_bps", "spread_bps"):
            self.signal[
                :, signal_names.index(f"ctx_cont.{field}")
            ] = self.ctx_cont[
                :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
            ]
        self.ctx_cat = np.column_stack(
            [
                (np.arange(rows) + offset) % 4
                for offset in range(len(MODEL_NATIVE_CTX_CAT_FIELDS))
            ]
        ).astype(np.int64)
        self.source_times_ns = pd.date_range(
            "2026-02-15T00:00:00Z",
            periods=rows,
            freq="1min",
            tz="UTC",
        ).asi8
        self.current_indices = np.asarray([500, 550], dtype=np.int64)
        self.local_indices = np.arange(21, 551, dtype=np.int64)

    def train_normalization_population(self) -> dict:
        return {
            "signal": self.signal,
            "ctx_cont": self.ctx_cont,
            "ctx_cat": self.ctx_cat,
            "local_row_indices": self.local_indices,
            "current_row_indices": self.current_indices,
            "current_decision_times_ns": self.source_times_ns[
                self.current_indices
            ]
            + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
            "source_times_ns": self.source_times_ns,
            "local_merged_intervals": ((21, 551),),
        }


def _artifacts(
    tmp_path: Path,
    *,
    signal_names: list[str],
    train_times: pd.DatetimeIndex,
) -> TrainNormalizationArtifacts:
    train_parquet = tmp_path / "train.parquet"
    train_parquet.write_bytes(b"immutable synthetic TRAIN parquet identity")
    m5_prebuilt = tmp_path / "xauusd_m5.parquet"

    source_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-03-10T00:00:00Z",
        freq="5min",
        tz="UTC",
    )
    pd.DataFrame({"time": source_index}).to_parquet(m5_prebuilt, index=False)
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(source_index)
    features = {
        tf: _mtf_frame(
            expected_indices[tf].asi8,
            _mtf_values(len(expected_indices[tf])) + np.float32(offset * 0.01),
        )
        for offset, tf in enumerate(MULTI_TF_RESAMPLE_RULES)
    }
    # Restore exact categorical domains after the harmless per-TF offset.
    ema_stack = list(MULTI_TF_PER_BAR_FEATURES_V4).index("ema_stack_aligned_v2")
    for frame in features.values():
        frame.attrs["feats_np"][:, ema_stack] = (
            np.arange(len(frame)) % 3 - 1
        ).astype(np.float32)
        for binary_offset, name in enumerate(
            MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4
        ):
            index = list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)
            frame.attrs["feats_np"][:, index] = (
                np.arange(len(frame)) % (binary_offset + 2) == binary_offset
            ).astype(np.float32)
        for name, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
            position = list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)
            frame.attrs["feats_np"][:, position] = np.asarray(
                domain, dtype=np.float32
            )[np.arange(len(frame)) % len(domain)]
    for timeframe, frame in features.items():
        _valid_observation_storage(frame.attrs["feats_np"], MULTI_TF_PER_BAR_FEATURES_V4)
        for name, value in MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4.get(timeframe, {}).items():
            frame.attrs["feats_np"][:, list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)] = value
    for tf_offset, (timeframe, frame) in enumerate(features.items()):
        scalar_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[
            timeframe
        ]
        scalar_values = (
            np.arange(len(frame), dtype=np.float32)[:, None]
            * (np.arange(len(scalar_fields), dtype=np.float32)[None, :] + 0.5)
            + np.float32(tf_offset * 100.0)
        ).astype(np.float32)
        frame.attrs["model_native_mtf_scalar_fields_v4"] = scalar_fields
        frame.attrs["model_native_mtf_scalars_np_v4"] = scalar_values
        frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = 0
        frame.attrs["model_native_mtf_scalar_contract_v4"] = (
            MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        )

    cache_dir = tmp_path / "mtf_cache"
    squeeze_artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    cache_manifest_path = publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=m5_prebuilt,
        expected_source_sha256=_sha256_file(m5_prebuilt),
        features=features,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=squeeze_artifacts,
    )
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    cache_binding = {
        "cache_dir": str(cache_dir),
        "manifest_path": str(cache_manifest_path),
        "manifest_sha256": _sha256_file(cache_manifest_path),
        "cache_identity_sha256": cache_manifest["cache_identity_sha256"],
        "m5_prebuilt_source": str(m5_prebuilt),
        "m5_prebuilt_source_sha256": _sha256_file(m5_prebuilt),
        "v29_registry_constants": _V29_TEST_REGISTRY_CONSTANTS,
        "volatility_squeeze_artifact_set": squeeze_artifacts.binding(),
    }
    manifest = {
        "output_data_path": str(train_parquet),
        "extra": {
            "entry_run_id": "SYNTHETIC_XAU_TRAIN",
            "multi_tf_cache_binding": cache_binding,
        },
        "feature_contract": {
            "signal_bridge_fields": signal_names,
            "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
            "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        },
        "ts_min_max_by_split": {
            "train": {
                "ts_min": train_times[0].isoformat(),
                "ts_max": train_times[-1].isoformat(),
            }
        },
    }
    train_manifest = tmp_path / "train.manifest.json"
    train_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return TrainNormalizationArtifacts(
        dataset_run_id="SYNTHETIC_XAU_TRAIN",
        train_parquet_path=train_parquet,
        train_manifest_path=train_manifest,
        m5_prebuilt_path=m5_prebuilt,
        mtf_cache_dir=cache_dir,
    )


@pytest.fixture()
def fit_fixture(tmp_path: Path) -> dict:
    seq, snap, ctx_cont, ctx_cat, times = _train_arrays()
    names = _signal_names()
    return {
        "seq": seq,
        "snap": snap,
        "ctx_cont": ctx_cont,
        "ctx_cat": ctx_cat,
        "times": times,
        "names": names,
        "exit_lifecycle": _FakeTrainExitLifecycle(names),
        "artifacts": _artifacts(
            tmp_path, signal_names=names, train_times=times
        ),
        "per_tf_seq_lens": {tf: 3 for tf in MULTI_TF_RESAMPLE_RULES},
    }


def _fit(values: dict, **kwargs: object) -> dict:
    return fit_entry_v10_train_input_normalization(
        train_seq=values["seq"],
        train_snap=values["snap"],
        train_ctx_cont=values["ctx_cont"],
        train_ctx_cat=values["ctx_cat"],
        train_times=values["times"],
        train_exit_lifecycle=values["exit_lifecycle"],
        ordered_signal_names=values["names"],
        per_tf_seq_lens=values["per_tf_seq_lens"],
        artifacts=values["artifacts"],
        row_chunk=5,
        **kwargs,
    )


def test_lifecycle_exposes_unique_train_m1_windows_without_surface_copy() -> None:
    lifecycle = object.__new__(UnifiedExitLifecycleSplit)
    lifecycle.split = "train"
    # This fixture's feature surface starts at the source clock's first row, so
    # source and feature coordinates coincide.
    lifecycle._feature_row_offset = 0
    lifecycle._episode_pointers = {
        (0, 0): (0, 500, 2000.0, 2000.1),
        (0, 1): (1, 500, 2000.0, 2000.1),
        (1, 0): (2, 520, 2000.0, 2000.1),
        (1, 1): (3, 520, 2000.0, 2000.1),
    }
    lifecycle._m1_times = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=1100,
        freq="1min",
    )
    lifecycle._m1_features = {
        "signal": np.zeros((1100, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32),
        "ctx_cont": np.zeros(
            (1100, len(MODEL_NATIVE_CTX_CONT_FIELDS)), dtype=np.float32
        ),
        "ctx_cat": np.zeros(
            (1100, len(MODEL_NATIVE_CTX_CAT_FIELDS)), dtype=np.int64
        ),
    }

    population = lifecycle.train_normalization_population()
    np.testing.assert_array_equal(
        population["current_row_indices"],
        np.arange(500, 1032, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        population["local_row_indices"],
        np.arange(21, 1032, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        population["current_decision_times_ns"],
        population["source_times_ns"][population["current_row_indices"]]
        + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
    )
    assert population["signal"] is lifecycle._m1_features["signal"]
    assert population["ctx_cont"] is lifecycle._m1_features["ctx_cont"]
    assert population["ctx_cat"] is lifecycle._m1_features["ctx_cat"]


def test_overlapping_mtf_windows_fit_each_source_row_once() -> None:
    source_times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=20, freq="1D", tz="UTC"
    ).asi8
    source = _mtf_frame(source_times, _mtf_values(len(source_times)))
    train_times = pd.DatetimeIndex(
        ["2026-01-11T00:00:00Z", "2026-01-12T00:00:00Z"]
    ).asi8

    selected, window, proof = select_causal_mtf_fit_population(
        tf="M5",
        source=source,
        train_times_ns=train_times,
        seq_len=3,
    )

    assert selected.shape == (4, len(MULTI_TF_PER_BAR_FEATURES_V4))
    assert window["left_index_inclusive"] == 8
    assert window["right_index_exclusive"] == 12
    assert window["selected_unique_row_count"] == 4
    assert proof["selection"] == (
        "union_of_entry_plus5_train_windows_each_cache_row_once"
    )


def test_unconsumed_future_does_not_change_selection_but_consumed_row_does() -> None:
    base_times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=20, freq="1D", tz="UTC"
    ).asi8
    base_values = _mtf_values(len(base_times))
    train_times = pd.DatetimeIndex(
        ["2026-01-11T00:00:00Z", "2026-01-12T00:00:00Z"]
    ).asi8
    selected_a, window_a, _ = select_causal_mtf_fit_population(
        tf="M5",
        source=_mtf_frame(base_times, base_values),
        train_times_ns=train_times,
        seq_len=3,
    )

    future_time = pd.Timestamp("2027-01-01T00:00:00Z").value
    future_values = np.vstack([base_values, _mtf_values(1) + np.float32(99.0)])
    ema_stack = list(MULTI_TF_PER_BAR_FEATURES_V4).index("ema_stack_aligned_v2")
    future_values[-1, ema_stack] = 0.0
    for name in MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4:
        future_values[-1, list(MULTI_TF_PER_BAR_FEATURES_V4).index(name)] = 0.0
    selected_b, window_b, _ = select_causal_mtf_fit_population(
        tf="M5",
        source=_mtf_frame(
            np.append(base_times, np.int64(future_time)), future_values
        ),
        train_times_ns=train_times,
        seq_len=3,
    )
    assert np.array_equal(selected_a, selected_b)
    assert (
        window_a["selected_row_indices_sha256"]
        == window_b["selected_row_indices_sha256"]
    )
    assert (
        window_a["selected_row_values_sha256"]
        == window_b["selected_row_values_sha256"]
    )
    stats_a = fit_surface_normalization(
        selected_a,
        surface="mtf_m5",
        field_names=MULTI_TF_PER_BAR_FEATURES_V4,
        semantic_categorical_domains=MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    )
    stats_b = fit_surface_normalization(
        selected_b,
        surface="mtf_m5",
        field_names=MULTI_TF_PER_BAR_FEATURES_V4,
        semantic_categorical_domains=MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    )
    assert stats_a["stats_sha256"] == stats_b["stats_sha256"]

    changed_values = base_values.copy()
    changed_values[9, 0] += np.float32(0.125)
    _, changed_window, _ = select_causal_mtf_fit_population(
        tf="M5",
        source=_mtf_frame(base_times, changed_values),
        train_times_ns=train_times,
        seq_len=3,
    )
    assert (
        changed_window["selected_row_indices_sha256"]
        == window_a["selected_row_indices_sha256"]
    )
    assert (
        changed_window["selected_row_values_sha256"]
        != window_a["selected_row_values_sha256"]
    )


def test_shared_train_fit_deduplicates_entry_exit_and_binds_all_lineage(
    fit_fixture: dict,
) -> None:
    result = _fit(fit_fixture)
    contract = result["normalization_contract"]
    proof = result["fit_population_proof"]
    rows = len(fit_fixture["times"])
    exit_lifecycle = fit_fixture["exit_lifecycle"]
    entry_local_rows = MODEL_NATIVE_SEQ_LEN + rows - 1
    exit_local_rows = int(exit_lifecycle.local_indices.size)
    context_rows = rows + int(exit_lifecycle.current_indices.size)

    assert contract["fit_scope"] == "train_only"
    assert contract["surfaces"]["signal"]["fit_row_count"] == (
        entry_local_rows + exit_local_rows
    )
    assert contract["surfaces"]["ctx_cont"]["fit_row_count"] == context_rows
    assert contract["ctx_cat"]["fit_row_count"] == context_rows
    assert contract["lineage"]["val_fit_row_count"] == 0
    assert contract["lineage"]["test_fit_row_count"] == 0
    assert contract["lineage"]["train_parquet_sha256"] == _sha256_file(
        fit_fixture["artifacts"].train_parquet_path
    )
    assert contract["lineage"]["train_manifest_sha256"] == _sha256_file(
        fit_fixture["artifacts"].train_manifest_path
    )
    assert contract["lineage"]["m5_prebuilt_sha256"] == _sha256_file(
        fit_fixture["artifacts"].m5_prebuilt_path
    )
    assert contract["lineage"]["mtf_cache_manifest_sha256"] == _sha256_file(
        fit_fixture["artifacts"].mtf_cache_dir / "manifest.json"
    )
    assert contract["lineage"]["mtf_feature_names_sha256"] == contract[
        "surfaces"
    ]["mtf_m5"]["field_names_sha256"]
    assert contract["surfaces"]["mtf_m5"]["field_count"] == len(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert contract["surfaces"]["mtf_m5"]["field_names"] == list(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert proof["train_decision_row_count"] == context_rows
    assert proof["entry_train_decision_row_count"] == rows
    assert proof["exit_train_decision_row_count"] == 2
    assert proof["local_populations"]["entry"][
        "selected_unique_row_count"
    ] == entry_local_rows
    assert proof["local_populations"]["exit"][
        "selected_unique_row_count"
    ] == exit_local_rows
    assert proof["mtf_populations"]["M5"]["routes"]["entry"][
        "enabled"
    ] is ("M5" in ENTRY_MTF_CONTEXT_TIMEFRAMES)
    assert proof["mtf_populations"]["M5"]["routes"]["exit"][
        "target_availability_shift_seconds"
    ] == 60
    assert proof["val_fit_row_count"] == proof["test_fit_row_count"] == 0

    non_alias_index = 0
    local_values = np.concatenate(
        [
            fit_fixture["seq"][0, :, non_alias_index],
            fit_fixture["snap"][1:, non_alias_index],
            exit_lifecycle.signal[
                exit_lifecycle.local_indices,
                non_alias_index,
            ],
        ]
    )
    expected_median = np.float32(
        np.median(local_values)
    )
    assert (
        np.float32(contract["surfaces"]["signal"]["center"][non_alias_index])
        == expected_median
    )


def test_shared_train_fit_uses_exit_state_bar_starts_once(
    fit_fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MTF selector, like the shared runtime owner, owns the +60s shift."""

    original = normalization.select_shared_causal_mtf_fit_population
    observed: list[np.ndarray] = []

    def capture_exit_clock(**kwargs: object):
        observed.append(np.asarray(kwargs["exit_train_times_ns"], dtype=np.int64))
        return original(**kwargs)

    monkeypatch.setattr(
        normalization,
        "select_shared_causal_mtf_fit_population",
        capture_exit_clock,
    )
    _fit(fit_fixture)

    lifecycle = fit_fixture["exit_lifecycle"]
    expected = lifecycle.source_times_ns[lifecycle.current_indices]
    assert len(observed) == len(MULTI_TF_RESAMPLE_RULES)
    for values in observed:
        np.testing.assert_array_equal(values, expected)


def test_shared_train_fit_reuses_explicit_verified_cache(
    fit_fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer's cache object is reused instead of loaded a second time."""

    cache = load_multi_tf_v4_cache(fit_fixture["artifacts"].mtf_cache_dir)

    def forbidden_reload(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("normalization must reuse the supplied cache")

    monkeypatch.setattr(normalization, "load_multi_tf_v4_cache", forbidden_reload)
    result = _fit(
        fit_fixture,
        prevalidated_multi_tf_cache=cache,
    )
    assert result["normalization_contract"]["contract_sha256"]


def test_shared_train_fit_rejects_dataset_cache_identity_tamper(
    fit_fixture: dict,
) -> None:
    manifest_path = fit_fixture["artifacts"].train_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra"]["multi_tf_cache_binding"][
        "cache_identity_sha256"
    ] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cache manifest lineage mismatch"):
        _fit(fit_fixture)


def test_alias_stats_are_shared_local_owned_and_dynamic(fit_fixture: dict) -> None:
    contract = _fit(fit_fixture)["normalization_contract"]
    assert [alias["signal_field"] for alias in contract["temporal_aliases"]] == [
        "ctx_cont.atr_bps",
        "ctx_cont.spread_bps",
    ]
    for alias in contract["temporal_aliases"]:
        signal_index = alias["signal_index"]
        ctx_index = alias["ctx_cont_index"]
        signal = contract["surfaces"]["signal"]
        ctx = contract["surfaces"]["ctx_cont"]
        assert np.float32(signal["center"][signal_index]).tobytes() == np.float32(
            ctx["center"][ctx_index]
        ).tobytes()
        assert np.float32(signal["scale"][signal_index]).tobytes() == np.float32(
            ctx["scale"][ctx_index]
        ).tobytes()
        assert signal["scale_source"][signal_index] == ctx["scale_source"][
            ctx_index
        ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("snap", "ENTRY_INPUT_NORMALIZATION_SEQ_LAST_SNAP_MISMATCH"),
        ("alias", "ENTRY_INPUT_NORMALIZATION_ALIAS_VALUE_MISMATCH"),
    ],
)
def test_snap_and_alias_mismatches_fail_closed(
    fit_fixture: dict,
    mutation: str,
    error: str,
) -> None:
    if mutation == "snap":
        fit_fixture["seq"] = fit_fixture["seq"].copy()
        fit_fixture["seq"][3, -1, 0] += np.float32(0.25)
    else:
        fit_fixture["ctx_cont"] = fit_fixture["ctx_cont"].copy()
        atr_index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index("atr_bps")
        fit_fixture["ctx_cont"][4, atr_index] += np.float32(0.25)
    with pytest.raises(RuntimeError, match=error):
        _fit(fit_fixture)


def test_fit_is_structurally_independent_of_downstream_subsamples(
    fit_fixture: dict,
) -> None:
    parameters = inspect.signature(
        fit_entry_v10_train_input_normalization
    ).parameters
    assert not any("sample" in name for name in parameters)

    downstream_subsample_a = np.arange(0, len(fit_fixture["times"]), 2)
    downstream_subsample_b = np.arange(1, len(fit_fixture["times"]), 3)
    assert not np.array_equal(
        downstream_subsample_a[: len(downstream_subsample_b)],
        downstream_subsample_b,
    )
    first = _fit(fit_fixture)
    second = _fit(fit_fixture)
    assert (
        first["normalization_contract"]["contract_sha256"]
        == second["normalization_contract"]["contract_sha256"]
    )
    assert (
        first["fit_population_proof"]["proof_sha256"]
        == second["fit_population_proof"]["proof_sha256"]
    )


def test_source_backed_normalization_rejects_unaudited_sequence_surface(
    fit_fixture: dict,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="ENTRY_INPUT_NORMALIZATION_SOURCE_BACKING_INVALID",
    ):
        _validate_source_backed_train_inputs(
            train_snap=fit_fixture["snap"],
            train_ctx_cont=fit_fixture["ctx_cont"],
            train_ctx_cat=fit_fixture["ctx_cat"],
            entry_sequence_source={
                "signal": fit_fixture["seq"][0],
                "times_ns": np.arange(MODEL_NATIVE_SEQ_LEN, dtype=np.int64),
                "sample_positions": np.arange(
                    len(fit_fixture["snap"]), dtype=np.int64
                ),
            },
            ordered_signal_names=fit_fixture["names"],
            temporal_aliases=[],
            row_chunk=5,
        )


def test_mtf_warmup_is_a_hard_failure() -> None:
    source_times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=20, freq="5min", tz="UTC"
    ).asi8
    source = _mtf_frame(
        source_times, _mtf_values(len(source_times)), warmup_rows=9
    )
    train_times = pd.DatetimeIndex(
        ["2026-01-01T00:50:00Z", "2026-01-01T00:55:00Z"]
    ).asi8
    with pytest.raises(
        RuntimeError, match="ENTRY_INPUT_NORMALIZATION_MTF_WARMUP_INCOMPLETE"
    ):
        select_causal_mtf_fit_population(
            tf="M5",
            source=source,
            train_times_ns=train_times,
            seq_len=3,
        )


def test_mtf_fit_rejects_off_grid_local_decision_clocks() -> None:
    entry_off_grid = np.asarray(
        [pd.Timestamp("2026-01-11T00:01:00Z").value], dtype=np.int64
    )
    with pytest.raises(
        RuntimeError,
        match="ENTRY_INPUT_NORMALIZATION_ROUTE_CLOCK_GRID_INVALID",
    ):
        normalization._validate_route_times(
            entry_off_grid,
            route="entry",
            allow_empty=False,
        )


@pytest.fixture()
def dynamic_fit_fixture(tmp_path, monkeypatch):
    from dataclasses import replace
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        ordered_model_native_signal_fields, model_native_signal_contract_metadata,
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
    from gx1.scripts.audit_entry_sequence_source_reconstruction_v1 import audit_sequence_source_reconstruction
    selected = (*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    names = list(ordered_model_native_signal_fields(selected))
    positions = np.asarray([95, 96, 101, 103], dtype=np.int64)
    source_times = pd.date_range("2026-02-14T16:05Z", periods=104, freq="5min").as_unit("ns")
    row = np.arange(104, dtype=np.float32)[:, None]
    source = (row * .25 + np.arange(len(names), dtype=np.float32)[None, :] * .01).astype(np.float32)
    context = (row * .5 + np.arange(len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.float32)[None, :] * .01).astype(np.float32)
    _valid_observation_storage(source, names)
    _valid_observation_storage(context, MODEL_NATIVE_CTX_CONT_FIELDS)
    _fill_semantic_categoricals(source, names, axis_rows=len(source))
    for alias in normalization._derive_temporal_aliases(names):
        source[:, alias["signal_index"]] = context[:, alias["ctx_cont_index"]]
    cat = np.column_stack([np.arange(104) % 4 for _ in MODEL_NATIVE_CTX_CAT_FIELDS]).astype(np.int64)
    times = source_times[positions]
    artifacts = _artifacts(tmp_path, signal_names=names, train_times=times)
    surface = tmp_path / "dynamic_m5_surface.parquet"
    pq.write_table(pa.table({"time": source_times, "signal": pa.array(source.tolist()),
                            "ctx_cont": pa.array(context.tolist()), "ctx_cat": pa.array(cat.tolist())}), surface)
    sidecar = Path(str(surface) + ".manifest.json")
    sidecar.write_text(json.dumps({"schema_version": "gx1_entry_exit_m5_feature_surface_v8",
        "output_parquet": str(surface), "output_parquet_sha256": _sha256_file(surface),
        "rows": len(source), "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS), "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS)}))
    sequence = np.stack([source[p - MODEL_NATIVE_SEQ_LEN + 1:p + 1] for p in positions])
    pq.write_table(pa.table({"time": times, "seq": pa.array(sequence.tolist()),
        "snap": pa.array(source[positions].tolist()), "ctx_cont": pa.array(context[positions].tolist()),
        "ctx_cat": pa.array(cat[positions].tolist())}), artifacts.train_parquet_path)
    registry = dict(gc_context.REGISTRY)
    registry.update(train_decision_rows=len(times), train_decision_clock_sha256=hashlib.sha256(
        normalization.m5_decision_availability(times).as_unit("ns").asi8.tobytes()).hexdigest())
    monkeypatch.setattr(gc_context, "REGISTRY", registry)
    manifest = json.loads(artifacts.train_manifest_path.read_text())
    manifest["extra"].update(emission_scope=M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
        gc_native_context=gc_context.contract(), model_native_signal_contract=model_native_signal_contract_metadata(selected),
        signal_bridge={"seq_structure_extension_v1": {"feature_surface": {
            "dataset_run_id": artifacts.dataset_run_id, "inline_split_recomputation": False,
            "manifest_path": str(sidecar), "manifest_sha256": _sha256_file(sidecar),
            "pair_generation_id": "synthetic_dynamic_source", "path": str(surface), "rows": len(source),
            "schema_version": "gx1_entry_exit_m5_feature_surface_v8", "sha256": _sha256_file(surface),
            "signal_manifest_sha256": "1" * 64, "time_alignment": "exact_entry_m5_source_timeline"}}})
    artifacts.train_manifest_path.write_text(json.dumps(manifest))
    audit = audit_sequence_source_reconstruction(parquet_path=artifacts.train_parquet_path,
                                                 manifest_path=artifacts.train_manifest_path)
    audit_path = tmp_path / "SOURCE_RECONSTRUCTION.json"
    audit_path.write_text(json.dumps(audit))
    artifacts = replace(artifacts, train_sequence_source_audit_path=audit_path)
    return {"train_seq": None, "train_snap": source[positions], "train_ctx_cont": context[positions],
        "train_ctx_cat": cat[positions], "train_times": times, "ordered_signal_names": names,
        "per_tf_seq_lens": {tf: 3 for tf in MULTI_TF_RESAMPLE_RULES}, "artifacts": artifacts,
        "entry_sequence_source": {"signal": source, "times_ns": source_times.asi8,
                                   "sample_positions": positions, "audit": audit},
        "entry_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE}


def test_dynamic_fit_uses_all_physical_train_rows_without_exit(dynamic_fit_fixture):
    result = fit_entry_v10_train_input_normalization(**dynamic_fit_fixture)
    contract, proof = result["normalization_contract"], result["fit_population_proof"]
    assert contract["fit_population"] == DYNAMIC_M5_FIT_POPULATION
    assert contract["lineage"]["entry_train_decision_row_count"] == 4
    assert contract["lineage"]["exit_train_decision_row_count"] == 0
    assert contract["lineage"]["context_fit_row_count"] == 4
    assert contract["surfaces"]["signal"]["fit_row_count"] == 104
    assert set(proof["context_populations"]) == set(proof["local_populations"]) == {"entry"}
    assert proof["gc_registry_whole_train_clock"]["exact_match"] is True
    for tf in MULTI_TF_RESAMPLE_RULES:
        routes = proof["mtf_populations"][tf]["routes"]
        assert routes["entry"]["enabled"] is True
        assert routes["entry"]["decision_row_count"] == 4
        assert routes["exit"]["enabled"] is False
        assert routes["exit"]["decision_row_count"] == 0
    fields = {name: surface["field_names"] for name, surface in contract["surfaces"].items()}
    require_input_normalization_contract(contract, expected_field_names=fields,
        expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS,
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    with pytest.raises(RuntimeError, match="PHASE_POPULATION_MISMATCH"):
        require_input_normalization_contract(contract, expected_field_names=fields,
            expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS,
            expected_entry_training_phase=JOINT_FITTED_Q_TRAINING_PHASE)
    joint_lie = dict(contract, fit_population=FIT_POPULATION)
    with pytest.raises(RuntimeError, match="TRAIN_ROWS_INVALID"):
        require_input_normalization_contract(joint_lie, expected_field_names=fields,
                                            expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS)
    exit_lie = dict(contract, lineage=dict(contract["lineage"], exit_train_decision_row_count=1))
    with pytest.raises(RuntimeError, match="ENTRY_ONLY_EXIT_ROWS_FORBIDDEN"):
        require_input_normalization_contract(exit_lie, expected_field_names=fields,
                                            expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS)


def test_dynamic_fit_rejects_whole_train_subset(dynamic_fit_fixture):
    values = dict(dynamic_fit_fixture)
    subset = [0, 2, 3]  # Same first/last timestamps cannot hide a missing physical middle row.
    for key in ("train_snap", "train_ctx_cont", "train_ctx_cat", "train_times"):
        values[key] = values[key][subset]
    values["entry_sequence_source"] = dict(values["entry_sequence_source"],
        sample_positions=values["entry_sequence_source"]["sample_positions"][subset])
    with pytest.raises(RuntimeError, match="WHOLE_TRAIN_CLOCK_INVALID"):
        fit_entry_v10_train_input_normalization(**values)


def test_dynamic_fit_rejects_registry_clock_mismatch(dynamic_fit_fixture, monkeypatch):
    monkeypatch.setitem(gc_context.REGISTRY, "train_decision_clock_sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="PHYSICAL_TRAIN_CLOCK_MISMATCH"):
        fit_entry_v10_train_input_normalization(**dynamic_fit_fixture)


@pytest.mark.parametrize("fault", ["exit", "dense", "unaudited"])
def test_dynamic_fit_requires_source_backing_without_exit(dynamic_fit_fixture, fault):
    values = dict(dynamic_fit_fixture)
    if fault == "exit": values["train_exit_lifecycle"] = object()
    elif fault == "dense": values["train_seq"] = object()
    else: values["entry_sequence_source"] = None
    with pytest.raises(RuntimeError, match="DYNAMIC_SOURCE_BACKING_REQUIRED_NO_EXIT"):
        fit_entry_v10_train_input_normalization(**values)


def test_source_backed_physical_row_hash_preserves_legacy_chunk_bytes():
    rows = normalization.DEFAULT_ROW_CHUNK + 17
    matrix = np.arange((rows + 9) * 3, dtype=np.float32).reshape(rows + 9, 3)
    indices = np.arange(7, 7 + rows, dtype=np.int64)
    times = np.arange(len(matrix), dtype=np.int64) * 300_000_000_000
    names = ["synthetic_a", "synthetic_b", "synthetic_c"]
    digest = hashlib.sha256()
    digest.update(b"entry_exit_shared_local_entry_m5_rows_v1\0")
    digest.update(bytes.fromhex(normalization._field_names_sha256(names)))
    normalization._update_physical_rows_hash(digest, physical_indices=indices,
        timestamps_ns=times[indices], values=matrix[indices])
    assert normalization._hash_selected_surface_rows(
        namespace="entry_exit_shared_local_entry_m5_rows_v1", matrix=matrix,
        row_indices=indices, timestamps_ns=times, field_names=names, dtype="<f4") == digest.hexdigest()


def test_joint_default_still_requires_exit_lifecycle(dynamic_fit_fixture):
    values = dict(dynamic_fit_fixture)
    values.pop("entry_training_phase")
    with pytest.raises(RuntimeError, match="EXIT_LIFECYCLE_REQUIRED"):
        fit_entry_v10_train_input_normalization(**values)
