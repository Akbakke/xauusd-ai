from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    BASE_REPAIR_METHOD,
    BASE_REPAIR_SCHEMA,
    XAU_INSTRUMENT,
    canonical_xau_source_descriptor_v1,
)
from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5
from gx1.scripts.materialize_current_m5_snapshot_v1 import (
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    _filter_supported_m5_buckets,
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
    canonical_sources = {}
    for key, timeframe in (("m5", "M5"), ("m1", "M1")):
        canonical_root = tmp_path / f"canonical_{key}"
        canonical_root.mkdir()
        (canonical_root / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "instrument": "XAUUSD",
                    "timeframe": timeframe,
                    "out_root": str(canonical_root.resolve()),
                }
            ),
            encoding="utf-8",
        )
        canonical_sources[key] = canonical_xau_source_descriptor_v1(
            canonical_root.resolve(),
            timeframe=timeframe,
        )
    base = tmp_path / "base"
    part = base / "year=2026" / "part-000.parquet"
    part.parent.mkdir(parents=True)
    m1 = _m1()
    m5 = m1_to_m5(m1, tape_end=pd.Timestamp("2026-07-21T00:14:00Z"))
    canonical_order = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "volume",
    ]
    m5.iloc[:2].loc[:, canonical_order].to_parquet(part, index=False)
    (base / "REPAIR_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": BASE_REPAIR_SCHEMA,
                "instrument": XAU_INSTRUMENT,
                "explicit_vedtak_id": RUN_ID,
                "method": BASE_REPAIR_METHOD,
                "geometry_bad_total_after": 0,
                "m5_tape_root": canonical_sources["m5"]["root"],
                "m1_tape_root": canonical_sources["m1"]["root"],
                "canonical_sources": canonical_sources,
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
    assert report["instrument"] == XAU_INSTRUMENT
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


def test_omits_unsupported_partial_bucket_without_filling() -> None:
    m1 = _m1().drop(index=6).reset_index(drop=True)
    aggregated = m1_to_m5(m1, tape_end=pd.Timestamp("2026-07-21T00:14:00Z"))

    filtered, proof = _filter_supported_m5_buckets(m1, aggregated)

    assert filtered["time"].tolist() == [
        pd.Timestamp("2026-07-21T00:00:00Z"),
        pd.Timestamp("2026-07-21T00:10:00Z"),
    ]
    assert proof["dropped_unsupported_partial_m5_rows"] == 1
    assert proof["dropped_unsupported_partial_m5_buckets"] == [
        {
            "time_utc": "2026-07-21T00:05:00+00:00",
            "m1_offsets_minutes": [0, 2, 3, 4],
            "reason": "unsupported_partial_m1_bucket",
        }
    ]


def test_rejects_dense_overlap_value_mismatch(tmp_path: Path) -> None:
    base, collector = _fixture(tmp_path)
    collector_path = collector / "xauusd_m1_20260721.parquet"
    changed = pd.read_parquet(collector_path)
    changed.loc[5, "volume"] += 1.0
    changed.to_parquet(collector_path, index=False)

    with pytest.raises(RuntimeError, match="OVERLAP_MISMATCH"):
        run(_args(tmp_path, base, collector))

    assert not (tmp_path / "current").exists()
