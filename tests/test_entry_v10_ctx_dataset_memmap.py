from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset


def _write_advanced_parquet(path: Path, *, times: list[str] | None = None) -> None:
    rows = 3
    seq_len = 2
    signal_dim = 4
    ctx_cont_dim = 43
    ctx_cat_dim = 5

    seq = [
        [[float(row + step + col) for col in range(signal_dim)] for step in range(seq_len)]
        for row in range(rows)
    ]
    snap = [[float(row + col) for col in range(signal_dim)] for row in range(rows)]
    ctx_cont = [[float(row + col) for col in range(ctx_cont_dim)] for row in range(rows)]
    ctx_cat = [[int((row + col) % 3) for col in range(ctx_cat_dim)] for row in range(rows)]

    table = pa.table(
        {
            "time": pa.array(
                times or [f"2026-01-0{row + 1}T00:00:00Z" for row in range(rows)],
                type=pa.string(),
            ),
            "seq": pa.array(seq, type=pa.list_(pa.list_(pa.float64()))),
            "snap": pa.array(snap, type=pa.list_(pa.float64())),
            "ctx_cont": pa.array(ctx_cont, type=pa.list_(pa.float64())),
            "ctx_cat": pa.array(ctx_cat, type=pa.list_(pa.int64())),
            "y_direction": pa.array([0, 1, 2], type=pa.int64()),
            "mae_first_n_bps": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "y_early_move": pa.array([0.0, 1.0, 0.0], type=pa.float64()),
            "y_quality_score": pa.array([0.2, 0.4, 0.6], type=pa.float64()),
            "y_tradable": pa.array([1.0, 0.0, 1.0], type=pa.float64()),
            "mfe_first_n_bps": pa.array([2.0, 3.0, 4.0], type=pa.float64()),
            "path_quality_bps": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
            "y_bad_path": pa.array([0.0, 1.0, 0.0], type=pa.float64()),
            "y_dead_negative_long": pa.array([0.0, 0.0, 1.0], type=pa.float64()),
            "y_teaser_negative_long": pa.array([0.0, 1.0, 0.0], type=pa.float64()),
            "y_hard_negative_long": pa.array([0.0, 0.0, 0.0], type=pa.float64()),
            "y_clean_edge_long": pa.array([1.0, 0.0, 1.0], type=pa.float64()),
            "y_survival_long": pa.array([1.0, 1.0, 0.0], type=pa.float64()),
            "y_teacher_bad_long": pa.array([0.0, 0.0, 1.0], type=pa.float64()),
            "y_teacher_winner_long": pa.array([1.0, 0.0, 0.0], type=pa.float64()),
            "y_selector_long_mask": pa.array([1.0, 1.0, 1.0], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_advanced_dataset_uses_memmap_when_nested_arrays_exceed_threshold(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path)

    monkeypatch.setenv("ENTRY_V10_CTX_MEMMAP_MIN_GB", "0")
    monkeypatch.setenv("ENTRY_V10_CTX_MEMMAP_ROOT", str(memmap_root))

    ds = EntryV10CtxDataset(parquet_path, seq_len=2, allow_constant_labels=False)

    assert isinstance(ds._np_seq, np.memmap)
    assert isinstance(ds._np_snap, np.memmap)
    assert isinstance(ds._np_ctx_cont, np.memmap)
    assert isinstance(ds._np_ctx_cat, np.memmap)
    assert ds._np_seq.shape == (3, 2, 4)
    assert ds._np_snap.shape == (3, 4)
    assert ds._np_ctx_cont.shape == (3, 43)
    assert ds._np_ctx_cat.shape == (3, 5)
    assert len(ds) == 3

    sample = ds[1]

    assert tuple(sample["seq_x"].shape) == (2, 4)
    assert tuple(sample["snap_x"].shape) == (4,)
    assert tuple(sample["ctx_cont"].shape) == (43,)
    assert tuple(sample["ctx_cat"].shape) == (5,)
    assert int(sample["y"].item()) == 1
    assert memmap_root.exists()


def test_advanced_dataset_rejects_unsorted_time_rows(tmp_path) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(
        parquet_path,
        times=[
            "2026-01-02T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
    )

    with pytest.raises(RuntimeError, match="ENTRY_V10_CTX_ADVANCED_TIME_ORDER_FAIL"):
        EntryV10CtxDataset(parquet_path, seq_len=2, allow_constant_labels=False)
