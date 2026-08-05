from __future__ import annotations

import pandas as pd
import pytest

from gx1.execution import v12_canonical_incremental as incremental


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    index.name = "time"
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
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


def test_pair_contract_locks_repaired_atr_regime() -> None:
    assert incremental.PREBUILT_CANONICAL_BUILDER_CONTRACT.endswith("_v2")
    assert incremental.PREBUILT_PAIR_FORMULA_CONTRACT[
        "canonical_m5_atr_regime"
    ] == (
        "atr14_rolling5760_min2880_q333_q667_shift1_"
        "integer_write_through_v2"
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
