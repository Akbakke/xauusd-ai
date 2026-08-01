from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_builder():
    script = (
        Path(__file__).resolve().parents[1]
        / "gx1"
        / "scripts"
        / "build_entry_exit_m1_enriched_frame_v1.py"
    )
    spec = importlib.util.spec_from_file_location("m1_enriched_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
                utc=True,
            ),
        }
    )


def _write_pair_manifest(
    path: Path,
    *,
    pair_generation_id: str,
    native_m1: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_generation_id": pair_generation_id,
                "lineage": {"native_sources": {"m1": native_m1}},
            }
        ),
        encoding="utf-8",
    )


def test_m1_producer_requires_exact_pair_native_m1_binding(tmp_path: Path) -> None:
    builder = _load_builder()
    pair_id = "a" * 64
    frame = _source_frame()
    source = {
        "root": "/data/native/V3",
        "manifest_path": "/data/native/V3/MANIFEST.json",
        "manifest_sha256": "b" * 64,
    }
    expected = {
        **source,
        "row_count": len(frame),
        "time_min_utc": frame["time"].iloc[0].isoformat(),
        "time_max_utc": frame["time"].iloc[-1].isoformat(),
    }
    pair_manifest = tmp_path / "pair.json"
    _write_pair_manifest(
        pair_manifest,
        pair_generation_id=pair_id,
        native_m1=expected,
    )

    result = builder._require_pair_binding(
        pair_manifest_path=pair_manifest,
        pair_generation_id=pair_id,
        source_identity=source,
        native_m1=frame,
    )
    assert result["pair_generation_id"] == pair_id
    assert result["native_m1"] == expected

    stale = dict(expected)
    stale["manifest_sha256"] = "c" * 64
    _write_pair_manifest(
        pair_manifest,
        pair_generation_id=pair_id,
        native_m1=stale,
    )
    with pytest.raises(RuntimeError, match="M1_ENRICHED_PAIR_NATIVE_M1_BINDING_MISMATCH"):
        builder._require_pair_binding(
            pair_manifest_path=pair_manifest,
            pair_generation_id=pair_id,
            source_identity=source,
            native_m1=frame,
        )
