from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    canonical_xau_source_descriptor_v1,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    build_multi_tf_v4_closed_timestamp_indices,
)
from gx1.scripts import audit_seq513_source_cascade_v1 as audit
from gx1.scripts import backfill_xauusd_m5_from_oanda as canonical_backfill
from gx1.scripts.prebuild_multi_tf_cache_v4 import publish_multi_tf_v4_cache
from gx1.scripts.materialize_cv3_modelrange_v1 import SCHEMA_VERSION as MODELRANGE_SCHEMA
from gx1.scripts.materialize_cv3_modelrange_v1 import (
    CTX_OWNED_SESSION_COLUMNS,
    ENTRY_DEAD_CONSTANT_COLUMNS,
    EXTRA_COLUMNS_FROM_CANONICAL_V2,
)
from tests.test_oanda_backfill_vedtak_gate import (
    _FakeOandaClient,
    materialize_native_xau_test_bundle,
)


RUN_ID = "XAU_SEQ513_SOURCE_AUDIT_PYTEST_V1"
SOURCE_TIMES = pd.date_range(
    "2026-01-01T00:00:00Z",
    "2026-01-03T00:00:00Z",
    freq="5min",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tape_kind: str = "native",
) -> Path:
    root = tmp_path / "event"
    root.mkdir()
    if tape_kind not in {"native", "native_v4"}:
        raise ValueError(f"unsupported tape_kind={tape_kind!r}")
    tape = root / "m5_tape_native_v3"
    if tape_kind == "native_v4":
        parent = tmp_path / "native_m5_parent"
        materialize_native_xau_test_bundle(
            parent,
            timeframe="M5",
        )
        parent_descriptor = canonical_xau_source_descriptor_v1(
            parent,
            timeframe="M5",
        )
        monkeypatch.setattr(
            canonical_backfill,
            "_require_clean_repository",
            lambda _root, *, timeframe: "a" * 40,
        )
        canonical_backfill.materialize_native_xau_successor(
            client=_FakeOandaClient(timeframe="M5"),
            timeframe="M5",
            vedtak_id="XAU_NATIVE_M5_FIXTURE_V3",
            end_utc=SOURCE_TIMES[-1] + pd.Timedelta(minutes=5),
            out_root=tape,
            parent_root=parent,
            expected_parent_manifest_sha256=parent_descriptor[
                "manifest_sha256"
            ],
        )
    else:
        materialize_native_xau_test_bundle(
            tape,
            timeframe="M5",
            end_utc=SOURCE_TIMES[-1] + pd.Timedelta(minutes=5),
        )

    times = SOURCE_TIMES
    row_count = len(times)
    cv2 = root / "canonical_features_v2.parquet"
    pd.DataFrame({"time": times}).to_parquet(cv2, index=False)
    _write_json(
        root / "canonical_features_v2_summary.json",
        {
            "out_path_v1": str(cv2.resolve()),
            "m5_tape_root_v1": str(tape.resolve()),
            "m5_bars_loaded_v1": row_count,
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
            "rows": row_count,
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
            "rows": row_count,
            "columns": 3,
            "time_max_utc": times[-1].isoformat(),
            "extra_columns_from_canonical_v2": list(
                EXTRA_COLUMNS_FROM_CANONICAL_V2
            ),
            "entry_dead_constant_columns_removed": list(
                ENTRY_DEAD_CONSTANT_COLUMNS
            ),
            "ctx_owned_session_columns_removed": list(
                CTX_OWNED_SESSION_COLUMNS
            ),
        },
    )
    mtf_root = root / "MULTI_TF_V4_CACHE"
    mtf_frames: dict[str, pd.DataFrame] = {}
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(
        pd.DatetimeIndex(times)
    )
    for tf_offset, tf in enumerate(MULTI_TF_RESAMPLE_RULES):
        timestamps = expected_indices[tf]
        row = np.arange(1, len(timestamps) + 1, dtype=np.float32)[:, None]
        column = np.arange(
            1,
            len(MULTI_TF_PER_BAR_FEATURES_V4) + 1,
            dtype=np.float32,
        )[None, :]
        values = np.ascontiguousarray(
            row * column + np.float32(tf_offset / 10.0)
        )
        timestamps_int64 = timestamps.as_unit("ns").asi8
        frame = pd.DataFrame(
            values,
            index=timestamps,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["feats_np"] = values
        frame.attrs["ts_int64"] = timestamps_int64
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        mtf_frames[tf] = frame
    publish_multi_tf_v4_cache(
        out_dir=mtf_root,
        m5_prebuilt=cv3.resolve(),
        expected_source_sha256=_sha(cv3),
        features=mtf_frames,
    )
    full = root / "FULL_PLUS_CTX_v3src.parquet"
    numeric_row = np.arange(1, row_count + 1, dtype=np.float64)
    pd.DataFrame(
        {
            "time": times,
            "one": numeric_row,
            "two": numeric_row**2 + 3.0,
        }
    ).to_parquet(full, index=False)
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
        str((year_dir / "part-000.parquet").resolve())
        for year_dir in sorted(tape.glob("year=*"))
    ]
    _write_json(
        root / "FULL_PLUS_CTX_v3src.ctx_diagnostics.json",
        {
            "prebuilt_path": str(modelrange.resolve()),
            "output_path": str(full.resolve()),
            "tape_root": str(tape.resolve()),
            "n_rows": row_count,
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
        expected_full_time_max=SOURCE_TIMES[-1].isoformat(),
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
    binding = audit.validate_seq513_source_cascade_proof(
        root / "SOURCE_CASCADE_PROOF.json",
        expected_run_id=RUN_ID,
        expected_source_parquet=root / "FULL_PLUS_CTX_v3src.parquet",
        expected_canonical_v2_parquet=root / "canonical_features_v2.parquet",
        expected_mtf_cache_dir=root / "MULTI_TF_V4_CACHE",
        expected_history_start_utc="2026-01-01T00:00:00Z",
        expected_time_max_utc=SOURCE_TIMES[-1].isoformat(),
    )
    assert binding["multi_tf_cache_identity_sha256"] == report["artifacts"][
        "multi_tf_cache_identity_sha256"
    ]
    tape_manifest = root / "m5_tape_native_v3" / "MANIFEST.json"
    assert report["artifacts"]["tape_manifest_sha256"] == _sha(tape_manifest)


def test_source_cascade_audit_accepts_native_v4_successor_tape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch, tape_kind="native_v4")

    report = audit.run(_args(root))

    assert report["decision"] == "PASS"
    assert (
        report["contracts"]["xau_tape_provenance"]["schema_version"]
        == "xau_canonical_native_source_v4"
    )


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
