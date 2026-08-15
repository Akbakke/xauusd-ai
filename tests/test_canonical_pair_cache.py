from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_canonical_incremental as incremental
from gx1.features.basic_v1 import BASIC_V1_FEATURES


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    index.name = "time"
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            **{
                name: np.arange(3, dtype=np.float64) + position
                for position, name in enumerate(BASIC_V1_FEATURES)
            },
            # V4 canonical must carry the owned m5h1 momentum column.
            "m5h1_momentum": [0.5, -0.25, 0.75],
        },
        index=index,
    )


def test_model_agnostic_canonical_cache_round_trips_exactly(tmp_path) -> None:
    checkpoint_key = "a" * 64
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    frame = _frame()

    incremental._write_model_agnostic_canonical_cache(
        canonical=frame,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )
    loaded = incremental._load_model_agnostic_canonical_cache(
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


def test_model_agnostic_canonical_cache_rejects_partial_pair(tmp_path) -> None:
    checkpoint_key = "b" * 64
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    parquet_path, _manifest_path = incremental._canonical_cache_paths(
        checkpoint_dir,
        checkpoint_key,
    )
    parquet_path.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="PAIR_CANONICAL_CACHE_INCOMPLETE"):
        incremental._load_model_agnostic_canonical_cache(
            checkpoint_dir=checkpoint_dir,
            checkpoint_key=checkpoint_key,
        )


def test_model_agnostic_cache_rejects_old_49_field_schema(tmp_path) -> None:
    checkpoint_key = "d" * 64
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    index.name = "time"
    old_surface = pd.DataFrame(
        {
            **{
                f"old_basic_{position:02d}": np.arange(3, dtype=np.float64)
                + position
                for position in range(48)
            },
            "m5h1_momentum": [0.5, -0.25, 0.75],
        },
        index=index,
    )
    incremental._write_model_agnostic_canonical_cache(
        canonical=old_surface,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )
    _parquet, manifest_path = incremental._canonical_cache_paths(
        checkpoint_dir,
        checkpoint_key,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "gx1_model_agnostic_canonical_cache_v3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PAIR_CANONICAL_CACHE_BINDING_INVALID"):
        incremental._load_model_agnostic_canonical_cache(
            checkpoint_dir=checkpoint_dir,
            checkpoint_key=checkpoint_key,
        )

    assert incremental.MODEL_AGNOSTIC_CACHE_SCHEMA == (
        "gx1_model_agnostic_canonical_cache_v6"
    )


def test_model_agnostic_cache_rejects_basic_formula_hash_mutation(
    tmp_path,
) -> None:
    checkpoint_key = "e" * 64
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    incremental._write_model_agnostic_canonical_cache(
        canonical=_frame(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )
    _parquet, manifest_path = incremental._canonical_cache_paths(
        checkpoint_dir,
        checkpoint_key,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["basic_v1_formula_sha256"] == (
        incremental.BASIC_V1_FORMULA_SHA256
    )
    manifest["basic_v1_formula_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PAIR_CANONICAL_CACHE_BINDING_INVALID"):
        incremental._load_model_agnostic_canonical_cache(
            checkpoint_dir=checkpoint_dir,
            checkpoint_key=checkpoint_key,
        )


def test_pair_frame_build_rejects_parallel_feature_workers(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="PAIR_FEATURE_WORKERS_MUST_EQUAL_ONE"):
        incremental._derive_pair_frames(
            native_m1=pd.DataFrame(),
            native_m5=pd.DataFrame(),
            checkpoint_dir=tmp_path,
            checkpoint_key="c" * 64,
            workers=2,
        )
