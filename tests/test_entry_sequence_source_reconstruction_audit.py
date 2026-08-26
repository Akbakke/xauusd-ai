from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
    AUTHORITY,
    REQUIRED_CHECKS,
    require_sequence_source_reconstruction_audit,
)
from gx1.scripts.audit_entry_sequence_source_reconstruction_v1 import (
    audit_sequence_source_reconstruction,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, break_sequence: bool = False) -> tuple[Path, Path]:
    width, seq_len, rows = 238, 96, 4
    source = np.arange((seq_len + 8) * width, dtype=np.float32).reshape(seq_len + 8, width)
    source_times = pd.Timestamp("2025-01-05T16:05:00Z") + pd.to_timedelta(
        np.arange(len(source)) * 5, unit="min"
    )
    surface = tmp_path / "m5_feature_base.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(source_times),
                "signal": pa.array(source.tolist()),
                "ctx_cont": pa.array(np.zeros((len(source), 71), dtype=np.float32).tolist()),
                "ctx_cat": pa.array(np.zeros((len(source), 1), dtype=np.int64).tolist()),
            }
        ),
        surface,
    )
    surface_manifest = Path(f"{surface}.manifest.json")
    surface_manifest.write_text(
        json.dumps(
            {
                "schema_version": "gx1_entry_exit_m5_feature_surface_v8",
                "output_parquet": str(surface.resolve()),
                "output_parquet_sha256": _sha256(surface),
                "rows": len(source),
                "signal_dim": width,
                "ctx_cont_dim": 71,
                "ctx_cat_dim": 1,
            }
        ),
        encoding="utf-8",
    )
    positions = np.asarray((95, 97, 101, 103), dtype=np.int64)
    sequence = np.stack([source[position - 95 : position + 1] for position in positions])
    if break_sequence:
        sequence[2, 3, 9] += np.float32(1.0)
    split = tmp_path / "split.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(source_times[positions]),
                "seq": pa.array(sequence.tolist()),
                "snap": pa.array(source[positions].tolist()),
            }
        ),
        split,
    )
    manifest = tmp_path / "split.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "signal_bridge": {
                        "seq_structure_extension_v1": {
                            "feature_surface": {
                                "dataset_run_id": "FIXTURE_DATASET_20260826",
                                "inline_split_recomputation": False,
                                "manifest_path": str(surface_manifest.resolve()),
                                "manifest_sha256": _sha256(surface_manifest),
                                "pair_generation_id": "fixture_generation",
                                "path": str(surface.resolve()),
                                "rows": len(source),
                                "schema_version": "gx1_entry_exit_m5_feature_surface_v8",
                                "sha256": _sha256(surface),
                                "signal_manifest_sha256": "1" * 64,
                                "time_alignment": "exact_entry_m5_source_timeline",
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return split, manifest


def test_source_reconstruction_audit_proves_filtered_windows(tmp_path: Path) -> None:
    split, manifest = _write_fixture(tmp_path)

    report = audit_sequence_source_reconstruction(
        parquet_path=split.resolve(), manifest_path=manifest.resolve()
    )

    assert report["decision"] == "PASS"
    assert report["checks"] == REQUIRED_CHECKS
    assert report["authority"] == AUTHORITY
    assert len(report["sequence_source_chain_sha256"]) == 64
    assert require_sequence_source_reconstruction_audit(
        report,
        expected_parquet_path=split.resolve(),
        expected_manifest_path=manifest.resolve(),
        expected_parquet_sha256=report["parquet_sha256"],
        expected_manifest_sha256=report["manifest_sha256"],
        expected_feature_surface=json.loads(manifest.read_text(encoding="utf-8")),
        expected_rows=4,
        expected_seq_len=96,
        expected_signal_dim=238,
    ) == report


def test_source_reconstruction_audit_rejects_mutated_stored_window(tmp_path: Path) -> None:
    split, manifest = _write_fixture(tmp_path, break_sequence=True)

    with pytest.raises(RuntimeError, match="SEQUENCE_MISMATCH"):
        audit_sequence_source_reconstruction(
            parquet_path=split.resolve(), manifest_path=manifest.resolve()
        )
