from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.features.htf_features import HTF_V2_CACHE_BUILDER_VERSION
from gx1.scripts import audit_seq513_source_cascade_v1 as audit
from gx1.scripts.materialize_cv3_modelrange_v1 import SCHEMA_VERSION as MODELRANGE_SCHEMA
from gx1.scripts.materialize_cv3_modelrange_v1 import (
    ENTRY_DEAD_CONSTANT_COLUMNS,
    EXTRA_COLUMNS_FROM_CANONICAL_V2,
)


RUN_ID = "XAU_SEQ513_SOURCE_AUDIT_PYTEST_V1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "event"
    root.mkdir()
    tape = root / "m5_tape_repaired_dec2024"
    years = {}
    for year in range(2020, 2027):
        part = tape / f"year={year}" / "part-000.parquet"
        part.parent.mkdir(parents=True)
        part.write_bytes(f"tape-{year}".encode())
        years[f"year={year}"] = {"output_sha256": _sha(part)}
    _write_json(
        tape / "REPAIR_MANIFEST.json",
        {
            "schema_version": "m5_tape_dec2024_repair_manifest_v1",
            "explicit_vedtak_id": RUN_ID,
            "geometry_bad_total_after": 0,
            "years": years,
        },
    )

    times = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="5min")
    cv2 = root / "canonical_features_v2.parquet"
    pd.DataFrame({"time": times}).to_parquet(cv2, index=False)
    _write_json(
        root / "canonical_features_v2_summary.json",
        {
            "out_path_v1": str(cv2.resolve()),
            "m5_tape_root_v1": str(tape.resolve()),
            "m5_bars_loaded_v1": 2,
            "total_columns_v1": 3,
            "htf_alignment_contract_v1": {"no_lookahead": True},
        },
    )
    cv3 = root / "cv3" / "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
    cv3.parent.mkdir()
    pd.DataFrame({"time": times}).to_parquet(cv3, index=False)
    _write_json(
        root / "cv3" / "CURRENT_MANIFEST.json",
        {
            "parquet_path": str(cv3.resolve()),
            "parquet_sha256": _sha(cv3),
            "source_v2_parquet": str(cv2.resolve()),
            "source_v2_parquet_sha256": _sha(cv2),
            "rows": 2,
            "cols_total": 3,
            "source_v2_no_lookahead": True,
        },
    )
    modelrange = root / "cv3_modelrange.parquet"
    pd.DataFrame({"time": times}).to_parquet(modelrange, index=False)
    _write_json(
        root / "cv3_modelrange.provenance.json",
        {
            "schema_version": MODELRANGE_SCHEMA,
            "entry_run_id": RUN_ID,
            "inputs": {
                "cv3": str(cv3.resolve()),
                "cv3_sha256": _sha(cv3),
                "canonical_v2": str(cv2.resolve()),
                "canonical_v2_sha256": _sha(cv2),
            },
            "output": str(modelrange.resolve()),
            "output_sha256": _sha(modelrange),
            "rows": 2,
            "columns": 3,
            "time_max_utc": times[-1].isoformat(),
            "extra_columns_from_canonical_v2": list(
                EXTRA_COLUMNS_FROM_CANONICAL_V2
            ),
            "entry_dead_constant_columns_removed": list(
                ENTRY_DEAD_CONSTANT_COLUMNS
            ),
        },
    )
    mtf_root = root / "MULTI_TF_V2_CACHE"
    mtf_root.mkdir()
    tfs = {}
    for tf in audit.EXPECTED_TFS:
        feats = mtf_root / f"{tf}_feats.npy"
        timestamps = mtf_root / f"{tf}_ts.npy"
        feats.write_bytes(f"{tf}-features".encode())
        timestamps.write_bytes(f"{tf}-timestamps".encode())
        tfs[tf] = {"feats_npy": feats.name, "ts_npy": timestamps.name}
    _write_json(
        mtf_root / "manifest.json",
        {
            "builder_version": HTF_V2_CACHE_BUILDER_VERSION,
            "m5_prebuilt_source": str(cv3.resolve()),
            "m5_prebuilt_source_sha256": _sha(cv3),
            "tfs": tfs,
        },
    )
    full = root / "FULL_PLUS_CTX_v3src.parquet"
    pd.DataFrame({"time": times, "one": [1.0, 2.0], "two": [3.0, 4.0]}).to_parquet(
        full, index=False
    )
    _write_json(
        root / "FULL_PLUS_CTX_v3src.manifest.json",
        {
            "kind": "entry_model_native_prebuilt_manifest_v2",
            "prebuilt_path": str(full.resolve()),
            "prebuilt_sha256": _sha(full),
            "no_fallback_enforced": True,
        },
    )
    raw_paths = [
        str((tape / f"year={year}" / "part-000.parquet").resolve())
        for year in range(2020, 2027)
    ]
    _write_json(
        root / "FULL_PLUS_CTX_v3src.ctx_diagnostics.json",
        {
            "prebuilt_path": str(modelrange.resolve()),
            "output_path": str(full.resolve()),
            "tape_root": str(tape.resolve()),
            "n_rows": 2,
            "raw_m5_paths": raw_paths,
        },
    )
    _write_json(
        root / "FULL_PLUS_CTX_v3src.schema_manifest.json",
        {"required_all_features": ["time", "one", "two"]},
    )
    monkeypatch.setattr(audit, "EXPECTED_CV2_COLUMNS", 3)
    monkeypatch.setattr(audit, "EXPECTED_CV3_MANIFEST_COLUMNS", 3)
    monkeypatch.setattr(audit, "EXPECTED_MODELRANGE_COLUMNS", 3)
    monkeypatch.setattr(audit, "EXPECTED_FULL_COLUMNS", 3)
    return root


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=RUN_ID,
        event_root=root,
        out=root / "SOURCE_CASCADE_PROOF.json",
        required_history_start="2026-01-01T00:00:00Z",
        expected_full_time_min="2026-01-01T00:00:00Z",
        expected_full_time_max="2026-01-01T00:05:00Z",
    )


def test_source_cascade_audit_binds_every_stage_and_emits_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)

    report = audit.run(_args(root))

    assert report["decision"] == "PASS"
    assert report["entry_run_id"] == RUN_ID
    assert report["contracts"]["no_stale_self_paths"] is True
    assert report["contracts"]["required_history_start_covered"] is True
    assert report["contracts"]["full_numeric_feature_liveness"]["decision"] == "PASS"
    assert json.loads((root / "SOURCE_CASCADE_PROOF.json").read_text()) == report


def test_source_cascade_audit_rejects_full_surface_after_required_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    args = _args(root)
    args.required_history_start = "2025-12-31T23:55:00Z"

    with pytest.raises(RuntimeError, match="REQUIRED_HISTORY_NOT_COVERED"):
        audit.run(args)


def test_source_cascade_audit_rejects_stale_cv3_self_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    manifest_path = root / "cv3" / "CURRENT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_v2_parquet"] = "/stale/V1/canonical_features_v2.parquet"
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="CV3_SOURCE_PATH_MISMATCH"):
        audit.run(_args(root))


@pytest.mark.parametrize("mutation,error", (("constant", "CONSTANT"), ("duplicate", "DUPLICATES")))
def test_source_cascade_audit_rejects_dead_or_duplicate_numeric_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    path = root / "FULL_PLUS_CTX_v3src.parquet"
    frame = pd.read_parquet(path)
    if mutation == "constant":
        frame["one"] = 1.0
    else:
        frame["two"] = frame["one"]
    frame.to_parquet(path, index=False)
    manifest_path = root / "FULL_PLUS_CTX_v3src.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prebuilt_sha256"] = _sha(path)
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match=error):
        audit.run(_args(root))
