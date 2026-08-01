from __future__ import annotations

import pandas as pd
import pytest

from gx1.execution import v12_canonical_incremental as incremental


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    index.name = "time"
    return pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0], "feature_b": [4.0, 5.0, 6.0]},
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
