from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import augment_forward_outcome_v2 as owner


def _frame() -> pd.DataFrame:
    rows = 6
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "high": np.linspace(101.0, 106.0, rows),
            "low": np.linspace(99.0, 104.0, rows),
            "close": np.linspace(100.0, 105.0, rows),
            "smc_swing_state": np.arange(rows) % 5,
        }
    )


def _multi_tf() -> dict[str, pd.DataFrame]:
    out = {}
    for index, name in enumerate(owner.TF_NAMES):
        frame = pd.DataFrame(index=pd.RangeIndex(2))
        frame.attrs["ts_int64"] = np.array([1, 2], dtype=np.int64) + index
        frame.attrs["feats_np"] = np.full((2, 2), index, dtype=np.float32)
        out[name] = frame
    return out


def _install_exact_fake_math(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def fake_context(df, **_kwargs):
        return object(), pd.DatetimeIndex(df["time"]), ["feature_a", "feature_b"]

    def compact(_ctx, _ts, lo, hi, *, extract):
        calls.append((lo, hi))
        rows = np.arange(lo, hi, dtype=np.float32)
        assert extract == ["feature_a", "feature_b"]
        return np.column_stack([rows, rows + 100.0]).astype(np.float32)

    def serial(_ctx, ts, lo, hi, *, extract):
        out = {name: np.full(len(ts), np.nan, dtype=np.float32) for name in extract}
        rows = np.arange(lo, hi, dtype=np.float32)
        out["feature_a"][lo:hi] = rows
        out["feature_b"][lo:hi] = rows + 100.0
        return out

    def finalize(df, cols, **_kwargs):
        result = df.copy()
        for name, values in cols.items():
            result[name] = values
        return result

    monkeypatch.setattr(owner, "build_attach_context", fake_context)
    monkeypatch.setattr(owner, "_compute_attach_rows_compact", compact)
    monkeypatch.setattr(owner, "compute_attach_rows", serial)
    monkeypatch.setattr(owner, "finalize_attach_columns", finalize)
    return calls


def test_group_a_checkpoint_resumes_exact_chunks_without_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_exact_fake_math(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoint"
    key = "a" * 64

    first = owner.attach_group_a_ctx_columns_parallel(
        _frame(),
        multi_tf=_multi_tf(),
        workers=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=key,
        checkpoint_chunk_rows=2,
    )

    assert calls == [(0, 2), (2, 4), (4, 6)]
    assert first["feature_a"].tolist() == list(np.arange(6, dtype=np.float32))
    complete_path = checkpoint_dir / "CHECKPOINT_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    assert complete["checkpoint_key"] == key
    assert complete["chunk_count"] == 3
    assert first.attrs["group_a_checkpoint_complete_path"] == str(complete_path)

    calls.clear()
    second = owner.attach_group_a_ctx_columns_parallel(
        _frame(),
        multi_tf=_multi_tf(),
        workers=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=key,
        checkpoint_chunk_rows=2,
    )
    assert calls == []
    assert np.array_equal(first["feature_b"], second["feature_b"])


def test_group_a_checkpoint_rejects_changed_identity_and_corrupt_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_exact_fake_math(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoint"
    kwargs = {
        "multi_tf": _multi_tf(),
        "workers": 1,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_key": "b" * 64,
        "checkpoint_chunk_rows": 2,
    }
    owner.attach_group_a_ctx_columns_parallel(_frame(), **kwargs)

    changed = dict(kwargs)
    changed["checkpoint_key"] = "c" * 64
    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        owner.attach_group_a_ctx_columns_parallel(_frame(), **changed)

    chunk = checkpoint_dir / "chunk_000000000_000000002.npz"
    chunk.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="invalid chunk"):
        owner.attach_group_a_ctx_columns_parallel(_frame(), **kwargs)
