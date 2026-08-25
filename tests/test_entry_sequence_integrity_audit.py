from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(
    path: Path,
    *,
    event_positions: tuple[int, ...] = (0, 1, 3, 4),
    time_offsets_m5: tuple[int, ...] = (0, 1, 4, 5),
    break_chain: bool = False,
) -> Path:
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
    end_positions = np.asarray(event_positions, dtype=np.int64) + (seq_len - 1)
    source_offsets = np.arange(len(source), dtype=np.int64)
    for previous, current, previous_offset, current_offset in zip(
        end_positions[:-1],
        end_positions[1:],
        time_offsets_m5[:-1],
        time_offsets_m5[1:],
    ):
        extra_bars = (current_offset - previous_offset) - (current - previous)
        if extra_bars < 0:
            raise ValueError("time offsets cannot be shorter than source distance")
        source_offsets[current:] += extra_bars
    source_times = pd.Timestamp("2025-01-05T16:05:00Z") + pd.to_timedelta(
        source_offsets * 5, unit="min"
    )
    times = source_times[end_positions]
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
    source_path = path.with_name("source.parquet")
    pq.write_table(pa.table({"time": pa.array(source_times)}), source_path)
    return source_path


def _write_manifest(path: Path, source_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "extra": {
                    "source_frame": {
                        "parquet_path": str(source_path.resolve()),
                        "parquet_sha256": _sha256(source_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_sequence_integrity_proves_physical_event_chain_across_calendar_gap(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    source = _write_split(parquet)
    _write_manifest(manifest, source)

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
        "source_overlap_eligible_pairs": 3,
        "source_nonoverlap_boundary_pairs": 0,
    }
    assert len(report["sequence_event_chain_sha256"]) == 64

    assert require_sequence_integrity_audit(
        report,
        expected_parquet_path=parquet,
        expected_manifest_path=manifest,
        expected_parquet_sha256=report["parquet_sha256"],
        expected_manifest_sha256=report["manifest_sha256"],
        expected_source_parquet_path=source.resolve(),
        expected_source_parquet_sha256=report["source_parquet_sha256"],
        expected_rows=4,
        expected_seq_len=96,
        expected_signal_dim=238,
    ) == report


def test_sequence_integrity_rejects_broken_event_chain(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    source = _write_split(parquet, break_chain=True)
    _write_manifest(manifest, source)

    with pytest.raises(RuntimeError, match="OVERLAP_MISMATCH"):
        audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)


def test_sequence_integrity_rejects_non_m5_timestamp_gap(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    source = _write_split(parquet, time_offsets_m5=(0, 1, 4, 5))
    table = pq.read_table(parquet)
    times = table.column("time").to_pandas().copy()
    times.iloc[2] = times.iloc[1] + pd.Timedelta(minutes=7)
    table = table.set_column(0, "time", pa.array(times))
    pq.write_table(table, parquet)
    _write_manifest(manifest, source)

    with pytest.raises(RuntimeError, match="TIME_NOT_M5_ALIGNED"):
        audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)


def test_sequence_integrity_accepts_source_bound_nonoverlap_boundary(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    source = _write_split(
        parquet,
        event_positions=(0, 1, 97, 98),
        time_offsets_m5=(0, 1, 97, 98),
    )
    _write_manifest(manifest, source)

    report = audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)

    assert report["decision"] == "PASS"
    assert report["transition_summary"]["source_overlap_eligible_pairs"] == 2
    assert report["transition_summary"]["source_nonoverlap_boundary_pairs"] == 1


def test_sequence_integrity_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    parquet = tmp_path / "split.parquet"
    manifest = tmp_path / "split.manifest.json"
    source = _write_split(parquet)
    _write_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["extra"]["source_frame"]["parquet_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SOURCE_FRAME_HASH_MISMATCH"):
        audit_sequence_integrity(parquet_path=parquet, manifest_path=manifest)
