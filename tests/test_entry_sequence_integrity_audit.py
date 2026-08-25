from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_sequence_integrity_v1 import (
    AUTHORITY,
    REQUIRED_CHECKS,
    require_sequence_integrity_audit,
)
from gx1.scripts.audit_entry_sequence_integrity_v1 import audit_sequence_integrity


def _write_split(
    path: Path,
    *,
    event_positions: tuple[int, ...] = (0, 1, 3, 4),
    time_offsets_m5: tuple[int, ...] = (0, 1, 4, 5),
    break_chain: bool = False,
) -> None:
    seq_len, width = 96, 238
    source = np.arange((max(event_positions) + seq_len) * width, dtype=np.float32).reshape(
        max(event_positions) + seq_len, width
    )
    sequence = np.stack(
        [source[index : index + seq_len] for index in event_positions]
    )
    if break_chain:
        sequence[2, 7, 11] += np.float32(1.0)
    snapshot = sequence[:, -1, :].copy()
    times = pd.Timestamp("2025-01-06T00:00:00Z") + pd.to_timedelta(
        np.asarray(time_offsets_m5, dtype=np.int64) * 5, unit="min"
    )
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times),
                "seq": pa.array(sequence.tolist()),
                "snap": pa.array(snapshot.tolist()),
            }
        ),
        path,
    )


def test_sequence_integrity_proves_physical_event_chain_across_calendar_gap(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet)
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    report = audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)

    assert report["decision"] == "PASS"
    assert report["checks"] == REQUIRED_CHECKS
    assert report["authority"] == AUTHORITY
    assert report["transition_summary"] == {
        "pairs": 3,
        "calendar_one_bar_pairs": 2,
        "calendar_gap_pairs": 1,
        "physical_one_bar_pairs": 2,
        "physical_multi_bar_pairs": 1,
        "calendar_elapsed_bars_total": 5,
        "physical_event_bars_total": 4,
        "nontrading_calendar_bars_total": 1,
    }
    assert len(report["sequence_event_chain_sha256"]) == 64

    assert require_sequence_integrity_audit(
        report,
        expected_parquet_path=parquet,
        expected_manifest_path=manifest,
        expected_parquet_sha256=report["parquet_sha256"],
        expected_manifest_sha256=report["manifest_sha256"],
        expected_rows=4,
        expected_seq_len=96,
        expected_signal_dim=238,
    ) == report


def test_sequence_integrity_rejects_broken_event_chain(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet, break_chain=True)
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="EVENT_CHAIN_MISMATCH"):
        audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)


def test_sequence_integrity_rejects_non_m5_timestamp_gap(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    _write_split(parquet, time_offsets_m5=(0, 1, 4, 5))
    table = pq.read_table(parquet)
    times = table.column("time").to_pandas().copy()
    times.iloc[2] = times.iloc[1] + pd.Timedelta(minutes=7)
    table = table.set_column(0, "time", pa.array(times))
    pq.write_table(table, parquet)
    manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="TIME_NOT_M5_ALIGNED"):
        audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)
