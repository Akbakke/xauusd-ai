from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleSplit
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    fit_surface_normalization,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    build_multi_tf_v4_closed_timestamp_indices,
)
from gx1.models.entry_v10.entry_v10_input_normalization import (
    TrainNormalizationArtifacts,
    fit_entry_v10_train_input_normalization,
    select_causal_mtf_fit_population,
)
from gx1.scripts.prebuild_multi_tf_cache_v4 import publish_multi_tf_v4_cache


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_names() -> list[str]:
    names = [f"test.signal_{index:03d}" for index in range(MODEL_NATIVE_SIGNAL_DIM)]
    names[7] = "ctx_cont.atr_bps"
    names[311] = "ctx_cont.m5_regime_class_id_v2"
    return names


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
    regime = list(MULTI_TF_PER_BAR_FEATURES_V4).index("regime_class_id")
    values[:, ema_stack] = (np.arange(rows) % 3 - 1).astype(np.float32)
    values[:, regime] = (np.arange(rows) % 5).astype(np.float32)
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
    for field in (
        "m5_regime_class_id_v2",
        "m15_regime_class_id_v2",
        "h1_regime_class_id_v2",
        "h4_regime_class_id_v2",
        "d1_regime_class_id_v2",
    ):
        index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
        ctx_cont[:, index] = (np.arange(rows) % 5).astype(np.float32)

    signal_names = _signal_names()
    snap[:, signal_names.index("ctx_cont.atr_bps")] = ctx_cont[
        :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index("atr_bps")
    ]
    snap[:, signal_names.index("ctx_cont.m5_regime_class_id_v2")] = ctx_cont[
        :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index("m5_regime_class_id_v2")
    ]

    # Deliberately make earlier temporal rows unlike snapshots.  They are
    # validated but must never be flattened into the signal fit population.
    seq = np.empty(
        (rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM),
        dtype=np.float32,
    )
    seq[:, :-1, :] = snap[:, None, :] - np.float32(50.0)
    seq[:, -1, :] = snap
    categorical_alias = signal_names.index(
        "ctx_cont.m5_regime_class_id_v2"
    )
    seq[:, :-1, categorical_alias] = 0.0
    atr_alias = signal_names.index("ctx_cont.atr_bps")
    seq[:, :-1, atr_alias] = 0.0

    ctx_cat = np.column_stack(
        [
            np.arange(rows) % 4,
            np.arange(rows) % 5,
            (np.arange(rows) + 1) % 5,
            (np.arange(rows) + 2) % 5,
            np.arange(rows) % 3,
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
        for field in (
            "m5_regime_class_id_v2",
            "m15_regime_class_id_v2",
            "h1_regime_class_id_v2",
            "h4_regime_class_id_v2",
            "d1_regime_class_id_v2",
        ):
            index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
            self.ctx_cont[:, index] = np.arange(rows) % 5
        self.signal[:, signal_names.index("ctx_cont.atr_bps")] = self.ctx_cont[
            :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index("atr_bps")
        ]
        self.signal[
            :, signal_names.index("ctx_cont.m5_regime_class_id_v2")
        ] = self.ctx_cont[
            :, list(MODEL_NATIVE_CTX_CONT_FIELDS).index(
                "m5_regime_class_id_v2"
            )
        ]
        self.ctx_cat = np.column_stack(
            [
                np.arange(rows) % 4,
                np.arange(rows) % 5,
                (np.arange(rows) + 1) % 5,
                (np.arange(rows) + 2) % 5,
                np.arange(rows) % 3,
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
            ],
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
    regime = list(MULTI_TF_PER_BAR_FEATURES_V4).index("regime_class_id")
    for frame in features.values():
        frame.attrs["feats_np"][:, ema_stack] = (
            np.arange(len(frame)) % 3 - 1
        ).astype(np.float32)
        frame.attrs["feats_np"][:, regime] = (
            np.arange(len(frame)) % 5
        ).astype(np.float32)

    cache_dir = tmp_path / "mtf_cache"
    cache_manifest_path = publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=m5_prebuilt,
        expected_source_sha256=_sha256_file(m5_prebuilt),
        features=features,
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


def _fit(values: dict) -> dict:
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
    )


def test_lifecycle_exposes_unique_train_m1_windows_without_surface_copy() -> None:
    lifecycle = object.__new__(UnifiedExitLifecycleSplit)
    lifecycle.split = "train"
    lifecycle._selected_state = np.asarray(
        [[0, 10, -1, -1], [20, -1, -1, -1]],
        dtype=np.int16,
    )
    lifecycle._selected_start = np.asarray(
        [[500, 500, -1, -1], [520, -1, -1, -1]],
        dtype=np.int64,
    )
    lifecycle._m1_times = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=600,
        freq="1min",
    )
    lifecycle._m1_features = {
        "signal": np.zeros((600, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32),
        "ctx_cont": np.zeros(
            (600, len(MODEL_NATIVE_CTX_CONT_FIELDS)), dtype=np.float32
        ),
        "ctx_cat": np.zeros(
            (600, len(MODEL_NATIVE_CTX_CAT_FIELDS)), dtype=np.int64
        ),
    }

    population = lifecycle.train_normalization_population()
    np.testing.assert_array_equal(
        population["current_row_indices"],
        np.asarray([500, 510, 540], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        population["local_row_indices"],
        np.arange(21, 541, dtype=np.int64),
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
        "union_of_entry_plus5_exit_plus1_train_windows_each_cache_row_once"
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
    regime = list(MULTI_TF_PER_BAR_FEATURES_V4).index("regime_class_id")
    future_values[-1, ema_stack] = 0.0
    future_values[-1, regime] = 2.0
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
    ] is False
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
        "ctx_cont.m5_regime_class_id_v2",
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


def test_mtf_warmup_is_a_hard_failure() -> None:
    source_times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=20, freq="1D", tz="UTC"
    ).asi8
    source = _mtf_frame(
        source_times, _mtf_values(len(source_times)), warmup_rows=9
    )
    train_times = pd.DatetimeIndex(
        ["2026-01-11T00:00:00Z", "2026-01-12T00:00:00Z"]
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
