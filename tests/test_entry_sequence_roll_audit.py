from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts.audit_entry_sequence_roll_v1 import audit_sequence_roll


def _write_split(
    path: Path,
    *,
    break_roll: bool = False,
    time_offsets_m5: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    rows, seq_len, width = 4, 96, 238
    source = np.arange((rows + seq_len - 1) * width, dtype=np.float32).reshape(
        rows + seq_len - 1, width
    )
    sequence = np.stack([source[index : index + seq_len] for index in range(rows)])
    snapshot = sequence[:, -1, :].copy()
    if break_roll:
        sequence[2, 3, 7] += np.float32(1.0)
    pq.write_table(
        pa.table(
            {
                "time": pa.array(
                    pd.Timestamp("2025-01-06T00:00:00Z")
                    + pd.to_timedelta(np.asarray(time_offsets_m5, dtype=np.int64) * 5, unit="min")
                ),
                "seq": pa.array(sequence.tolist()),
                "snap": pa.array(snapshot.tolist()),
            }
        ),
        path,
    )


def test_roll_audit_proves_every_sequence_is_the_snapshot_chain(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet)
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    report = audit_sequence_roll(parquet_path=parquet, manifest_path=manifest)

    assert report["decision"] == "PASS"
    assert report["rows"] == 4
    assert report["checks"]["every_seq_last_equals_snap_bit_identical"] is True
    assert report["checks"]["every_adjacent_sequence_rolls_one_snapshot_bit_identical"] is True
    assert len(report["sequence_snapshot_chain_sha256"]) == 64
    assert report["authority"]["candidate"] is False


def test_roll_audit_rejects_one_nonrolling_sequence_row(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet, break_roll=True)
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ADJACENT_MISMATCH"):
        audit_sequence_roll(parquet_path=parquet, manifest_path=manifest)


def test_roll_audit_rejects_filtered_emitted_rows_before_nested_scan(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet, time_offsets_m5=(0, 1, 4, 5))
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="EMITTED_ROWS_NONCONTIGUOUS"):
        audit_sequence_roll(parquet_path=parquet, manifest_path=manifest)
