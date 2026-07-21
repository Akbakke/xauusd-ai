from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.materialize_cv3_modelrange_v1 import (
    DEFAULT_END_UTC,
    DEFAULT_START_UTC,
    ENTRY_DEAD_CONSTANT_COLUMNS,
    EXPECTED_CV3_COLUMN_COUNT,
    EXPECTED_OUTPUT_COLUMN_COUNT,
    EXTRA_COLUMNS_FROM_CANONICAL_V2,
    SCHEMA_VERSION,
    run,
)


RUN_ID = "XAU_CV3_MODELRANGE_PYTEST_V1"


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    times = pd.date_range("2020-11-12T23:55:00Z", periods=4, freq="5min")
    columns = {
        "time": times,
        "close": np.arange(4, dtype=np.float64) + 1.0,
        **{
            f"cv3_{index:03d}": np.arange(4, dtype=np.float64) + index
            for index in range(
                EXPECTED_CV3_COLUMN_COUNT - 2 - len(ENTRY_DEAD_CONSTANT_COLUMNS)
            )
        },
        **{
            name: np.full(4, float(index), dtype=np.float64)
            for index, name in enumerate(ENTRY_DEAD_CONSTANT_COLUMNS)
        },
    }
    cv3 = pd.DataFrame(columns)
    canonical_v2 = cv3.copy()
    for index, name in enumerate(EXTRA_COLUMNS_FROM_CANONICAL_V2):
        canonical_v2[name] = np.arange(4, dtype=np.float64) + 1000.0 + index
    cv3_path = tmp_path / "cv3.parquet"
    canonical_v2_path = tmp_path / "canonical_v2.parquet"
    cv3.to_parquet(cv3_path, index=False)
    canonical_v2.to_parquet(canonical_v2_path, index=False)
    return cv3_path, canonical_v2_path


def _args(tmp_path: Path, cv3: Path, canonical_v2: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=RUN_ID,
        cv3=cv3,
        canonical_v2=canonical_v2,
        out=tmp_path / "cv3_modelrange.parquet",
        start="2020-11-13T00:00:00Z",
        end="2020-11-13T00:10:00Z",
    )


def test_materializes_exact_row_aligned_finite_modelrange(tmp_path: Path) -> None:
    cv3, canonical_v2 = _inputs(tmp_path)
    args = _args(tmp_path, cv3, canonical_v2)

    report = run(args)

    output = pd.read_parquet(args.out)
    assert output.shape == (3, EXPECTED_OUTPUT_COLUMN_COUNT)
    assert list(output.columns[-len(EXTRA_COLUMNS_FROM_CANONICAL_V2) :]) == list(
        EXTRA_COLUMNS_FROM_CANONICAL_V2
    )
    assert not set(ENTRY_DEAD_CONSTANT_COLUMNS) & set(output.columns)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["entry_run_id"] == RUN_ID
    assert report["output_sha256"] == hashlib.sha256(args.out.read_bytes()).hexdigest()
    sidecar = json.loads(args.out.with_suffix(".provenance.json").read_text())
    assert sidecar == report


def test_rejects_nonidentical_source_time_axis(tmp_path: Path) -> None:
    cv3, canonical_v2 = _inputs(tmp_path)
    frame = pd.read_parquet(canonical_v2)
    frame.loc[2, "time"] = pd.Timestamp("2020-11-13T00:11:00Z")
    frame = frame.sort_values("time")
    frame.to_parquet(canonical_v2, index=False)

    with pytest.raises(RuntimeError, match="SOURCE_TIME_ALIGNMENT_MISMATCH"):
        run(_args(tmp_path, cv3, canonical_v2))


def test_rejects_existing_output_or_sidecar(tmp_path: Path) -> None:
    cv3, canonical_v2 = _inputs(tmp_path)
    args = _args(tmp_path, cv3, canonical_v2)
    args.out.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="OUTPUT_NOT_FRESH"):
        run(args)


def test_active_window_defaults_are_pinned() -> None:
    assert DEFAULT_START_UTC == "2020-11-13T00:00:00Z"
    assert DEFAULT_END_UTC == "2026-06-14T23:59:59Z"
    assert EXTRA_COLUMNS_FROM_CANONICAL_V2 == ("atr",)
