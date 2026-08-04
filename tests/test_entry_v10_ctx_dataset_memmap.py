from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.entry_v10_trainer_dataset_support import (
    aux_head_target_contract,
    install_multi_tf_stub,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _write_advanced_parquet(path: Path, *, times: list[str] | None = None) -> None:
    rows = 3
    seq_len = 2
    signal_dim = MODEL_NATIVE_SIGNAL_DIM
    ctx_cont_dim = MODEL_NATIVE_CTX_CONT_DIM
    ctx_cat_dim = MODEL_NATIVE_CTX_CAT_DIM

    seq = [
        [[float(row + step + col) for col in range(signal_dim)] for step in range(seq_len)]
        for row in range(rows)
    ]
    snap = [[float(row + col) for col in range(signal_dim)] for row in range(rows)]
    ctx_cont = [[float(row + col) for col in range(ctx_cont_dim)] for row in range(rows)]
    ctx_cat = [[int((row + col) % 3) for col in range(ctx_cat_dim)] for row in range(rows)]

    columns = {
            "time": pa.array(
                times or [f"2026-01-0{row + 1}T00:00:00Z" for row in range(rows)],
                type=pa.string(),
            ),
            "seq": pa.array(seq, type=pa.list_(pa.list_(pa.float64()))),
            "snap": pa.array(snap, type=pa.list_(pa.float64())),
            "ctx_cont": pa.array(ctx_cont, type=pa.list_(pa.float64())),
            "ctx_cat": pa.array(ctx_cat, type=pa.list_(pa.int64())),
            "mae_first_n_bps": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "y_early_move": pa.array([0.0, 1.0, 0.0], type=pa.float64()),
            "y_quality_score": pa.array([0.2, 0.4, 0.6], type=pa.float64()),
        }
    for target in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
        values = [0.0, 0.0, 0.0]
        if target == "y_direction":
            values = [0, 1, 2]
        elif target in ("y_tf_agreement_score", "y_position_size_target"):
            values = [0.5, 0.5, 0.5]
        columns[target] = pa.array(values)
    table = pa.table(columns)
    pq.write_table(table, path)
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.memmap_fixture"
        )
    )
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "extra": {
                    "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                    "model_native_signal_contract": signal_contract,
                    "aux_head_target_contract": aux_head_target_contract(),
                    "signal_bridge": {
                        "fields": signal_contract["fields"],
                        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_advanced_dataset_uses_memmap_when_nested_arrays_exceed_threshold(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path)

    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    assert trainer._NESTED_ARROW_BATCH_ROWS == 512
    assert trainer._MEMMAP_WRITEBACK_ROWS == 2048
    monkeypatch.setattr(trainer, "_NESTED_ARROW_BATCH_ROWS", 1)
    monkeypatch.setattr(trainer, "_MEMMAP_WRITEBACK_ROWS", 2)
    flush_calls = 0
    real_flush = trainer._flush_memmap_pages

    def _counting_flush(*arrays) -> None:
        nonlocal flush_calls
        flush_calls += 1
        real_flush(*arrays)

    monkeypatch.setattr(trainer, "_flush_memmap_pages", _counting_flush)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=2,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens={"M5": 2, "M15": 2, "H1": 2, "H4": 2, "D1": 2},
        multi_tf_closed_bar=True,
    )

    assert isinstance(ds._np_seq, np.memmap)
    assert isinstance(ds._np_snap, np.memmap)
    assert isinstance(ds._np_ctx_cont, np.memmap)
    assert isinstance(ds._np_ctx_cat, np.memmap)
    assert ds._np_seq.shape == (3, 2, MODEL_NATIVE_SIGNAL_DIM)
    assert ds._np_snap.shape == (3, MODEL_NATIVE_SIGNAL_DIM)
    assert ds._np_ctx_cont.shape == (3, MODEL_NATIVE_CTX_CONT_DIM)
    assert ds._np_ctx_cat.shape == (3, MODEL_NATIVE_CTX_CAT_DIM)
    assert len(ds) == 3

    sample = ds[1]

    assert tuple(sample["seq_x"].shape) == (2, MODEL_NATIVE_SIGNAL_DIM)
    assert tuple(sample["snap_x"].shape) == (MODEL_NATIVE_SIGNAL_DIM,)
    assert tuple(sample["ctx_cont"].shape) == (MODEL_NATIVE_CTX_CONT_DIM,)
    assert tuple(sample["ctx_cat"].shape) == (MODEL_NATIVE_CTX_CAT_DIM,)
    assert int(sample["y"].item()) == 1
    assert "y_teacher_bad_long" not in sample
    assert "y_teacher_winner_long" not in sample
    assert memmap_root.exists()
    assert flush_calls == 2


def test_advanced_dataset_rejects_unsorted_time_rows(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(
        parquet_path,
        times=[
            "2026-01-02T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
    )
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="ENTRY_V10_CTX_ADVANCED_TIME_ORDER_FAIL"):
        trainer.EntryV10CtxDataset(
            parquet_path,
            seq_len=2,
            m5_prebuilt_path=m5_path,
            per_tf_seq_lens={"M5": 2, "M15": 2, "H1": 2, "H4": 2, "D1": 2},
            multi_tf_closed_bar=True,
        )


def test_dataset_rejects_missing_timeframe_length_without_global_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(parquet_path)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="ENTRY_PER_TF_SEQ_LEN_CONTRACT_INVALID",
    ):
        trainer.EntryV10CtxDataset(
            parquet_path,
            seq_len=2,
            m5_prebuilt_path=m5_path,
            per_tf_seq_lens={"M5": 2, "H4": 2, "D1": 2},
            multi_tf_closed_bar=True,
        )


def test_compact_materialized_rows_preserves_original_row_lookup(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path)

    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)
    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=2,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens={"M5": 2, "M15": 2, "H1": 2, "H4": 2, "D1": 2},
        multi_tf_closed_bar=True,
    )

    ds.indices = np.asarray([0, 2], dtype=np.int64)
    ds.compact_materialized_rows(ds.indices)

    assert not isinstance(ds._np_seq, np.memmap)
    assert ds._np_seq.shape == (2, 2, MODEL_NATIVE_SIGNAL_DIM)
    assert np.array_equal(ds._compact_row_indices, np.asarray([0, 2]))
    sample = ds[1]
    assert int(sample["y"].item()) == 2
    np.testing.assert_allclose(
        sample["snap_x"].numpy(),
        np.arange(MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32) + 2.0,
    )


def test_liveness_storage_indices_use_compact_array_coordinates() -> None:
    class CompactDataset:
        indices = np.asarray([3, 10_113, 20_000], dtype=np.int64)
        _compact_row_indices = indices.copy()
        _np_seq = np.empty((3, 0), dtype=np.float32)
        _np_snap = np.empty((3, 0), dtype=np.float32)
        _np_ctx_cont = np.empty((3, 0), dtype=np.float32)

    observed = trainer._deterministic_liveness_storage_indices(
        CompactDataset(),
        sample_rows=3,
    )

    np.testing.assert_array_equal(observed, np.asarray([0, 1, 2]))


def test_liveness_storage_indices_preserve_full_array_coordinates() -> None:
    class FullDataset:
        indices = np.asarray([3, 7, 11], dtype=np.int64)
        _compact_row_indices = None
        _np_seq = np.empty((12, 0), dtype=np.float32)
        _np_snap = np.empty((12, 0), dtype=np.float32)
        _np_ctx_cont = np.empty((12, 0), dtype=np.float32)

    observed = trainer._deterministic_liveness_storage_indices(
        FullDataset(),
        sample_rows=3,
    )

    np.testing.assert_array_equal(observed, np.asarray([3, 7, 11]))
