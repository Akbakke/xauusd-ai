from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5
from gx1.scripts.materialize_current_m5_snapshot_v1 import (
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    run,
)


RUN_ID = "XAU_CURRENT_M5_SNAPSHOT_PYTEST_V1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _m1() -> pd.DataFrame:
    time = pd.date_range("2026-07-21T00:00:00Z", periods=15, freq="1min")
    close = 4000.0 + np.arange(len(time), dtype=np.float64) * 0.1
    frame = pd.DataFrame(
        {
            "time": time,
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": np.arange(len(time), dtype=np.float64) + 1.0,
            "bid_open": close - 0.12,
            "bid_high": close - 0.02,
            "bid_low": close - 0.18,
            "bid_close": close - 0.1,
            "ask_open": close + 0.08,
            "ask_high": close + 0.18,
            "ask_low": close + 0.02,
            "ask_close": close + 0.1,
        }
    )
    return frame.loc[:, ["time", *REQUIRED_COLUMNS]]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    part = base / "year=2026" / "part-000.parquet"
    part.parent.mkdir(parents=True)
    m1 = _m1()
    m5 = m1_to_m5(m1, tape_end=pd.Timestamp("2026-07-21T00:14:00Z"))
    m5.iloc[:2].loc[:, ["time", *REQUIRED_COLUMNS]].to_parquet(part, index=False)
    (base / "REPAIR_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "m5_tape_dec2024_repair_manifest_v1",
                "explicit_vedtak_id": RUN_ID,
                "geometry_bad_total_after": 0,
                "years": {"year=2026": {"output_sha256": _sha(part)}},
            }
        ),
        encoding="utf-8",
    )
    collector = tmp_path / "collector"
    collector.mkdir()
    m1.to_parquet(collector / "xauusd_m1_20260721.parquet", index=False)
    m1.iloc[:5].to_parquet(collector / "xauusd_m1_20260720.parquet", index=False)
    return base, collector


def _args(tmp_path: Path, base: Path, collector: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=RUN_ID,
        base_m5_root=base,
        collector_dir=collector,
        cutoff_utc="2026-07-21T00:14:00Z",
        out_root=tmp_path / "current",
        minimum_overlap_bars=2,
    )


def test_materializes_immutable_current_tape_with_exact_overlap(tmp_path: Path) -> None:
    base, collector = _fixture(tmp_path)
    args = _args(tmp_path, base, collector)

    report = run(args)

    output = pd.read_parquet(args.out_root / "year=2026" / "part-000.parquet")
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["entry_run_id"] == RUN_ID
    assert report["overlap_exact"] is True
    assert report["overlap_proof"]["rows"] == 2
    assert report["overlap_proof"]["new_tail_rows"] == 1
    assert report["collector_duplicate_timestamps_identical"] == 5
    assert len(output) == 3
    assert pd.Timestamp(output["time"].iloc[-1]) == pd.Timestamp(
        "2026-07-21T00:10:00Z"
    )
    manifest = json.loads((args.out_root / "REPAIR_MANIFEST.json").read_text())
    assert manifest == report
    assert len(list((args.out_root / "collector_snapshot").glob("*.parquet"))) == 2


def test_rejects_conflicting_collector_duplicate(tmp_path: Path) -> None:
    base, collector = _fixture(tmp_path)
    conflict_path = collector / "xauusd_m1_20260720.parquet"
    conflict = pd.read_parquet(conflict_path)
    conflict.loc[0, "volume"] += 1.0
    conflict.to_parquet(conflict_path, index=False)

    with pytest.raises(RuntimeError, match="DUPLICATE_CONFLICT"):
        run(_args(tmp_path, base, collector))

    assert not (tmp_path / "current").exists()
