from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import augment_forward_outcome_v2 as owner


def _canonical_json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _materialize_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    _install_exact_fake_math(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoint"
    result = owner.attach_group_a_ctx_columns_parallel(
        _frame(),
        multi_tf=_multi_tf(),
        workers=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key="d" * 64,
        checkpoint_chunk_rows=2,
    )
    return checkpoint_dir, Path(result.attrs["group_a_checkpoint_complete_path"])


def _read_complete(complete_path: Path) -> dict:
    return json.loads(complete_path.read_text(encoding="utf-8"))


def _rewrite_complete(complete_path: Path, payload: dict) -> None:
    complete_path.write_bytes(_canonical_json_bytes(payload))


def _rewrite_chunk(
    complete_path: Path,
    *,
    chunk_index: int,
    mutate,
) -> None:
    complete = _read_complete(complete_path)
    chunk_record = complete["chunks"][chunk_index]
    chunk_path = Path(chunk_record["path"])
    with np.load(chunk_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    mutate(arrays)
    with chunk_path.open("wb") as handle:
        np.savez(handle, **arrays)
    encoded = chunk_path.read_bytes()
    chunk_record["sha256"] = hashlib.sha256(encoded).hexdigest()
    chunk_record["size_bytes"] = len(encoded)
    _rewrite_complete(complete_path, complete)


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


def test_group_a_checkpoint_validator_is_read_only_and_accepts_expected_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    before = {
        path.name: path.read_bytes()
        for path in sorted(checkpoint_dir.iterdir())
    }
    expected_sha256 = hashlib.sha256(complete_path.read_bytes()).hexdigest()

    complete = owner.require_group_a_checkpoint_complete(
        complete_path,
        expected_sha256=expected_sha256,
    )

    assert complete == _read_complete(complete_path)
    assert before == {
        path.name: path.read_bytes()
        for path in sorted(checkpoint_dir.iterdir())
    }


def test_group_a_checkpoint_validator_rejects_hash_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    original_complete = complete_path.read_bytes()
    with pytest.raises(RuntimeError, match="completion SHA-256 mismatch"):
        owner.require_group_a_checkpoint_complete(
            complete_path,
            expected_sha256="0" * 64,
        )

    complete = _read_complete(complete_path)
    complete["checkpoint_manifest_sha256"] = "0" * 64
    _rewrite_complete(complete_path, complete)
    with pytest.raises(RuntimeError, match="canonical manifest SHA-256 mismatch"):
        owner.require_group_a_checkpoint_complete(complete_path)

    complete_path.write_bytes(original_complete)
    complete = _read_complete(complete_path)
    complete["chunks"][0]["sha256"] = "0" * 64
    _rewrite_complete(complete_path, complete)
    with pytest.raises(RuntimeError, match="chunk SHA-256 mismatch"):
        owner.require_group_a_checkpoint_complete(complete_path)


@pytest.mark.parametrize("entry_case", ["missing", "extra"])
def test_group_a_checkpoint_validator_rejects_missing_and_extra_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_case: str,
) -> None:
    checkpoint_dir, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    if entry_case == "missing":
        Path(_read_complete(complete_path)["chunks"][0]["path"]).unlink()
    else:
        (checkpoint_dir / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="directory entry set mismatch"):
        owner.require_group_a_checkpoint_complete(complete_path)


def test_group_a_checkpoint_validator_rejects_symlink_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    chunk_path = Path(_read_complete(complete_path)["chunks"][0]["path"])
    external_chunk = tmp_path / "external.npz"
    chunk_path.replace(external_chunk)
    chunk_path.symlink_to(external_chunk)

    with pytest.raises(RuntimeError, match="canonical and symlink-free"):
        owner.require_group_a_checkpoint_complete(complete_path)


def test_group_a_checkpoint_validator_rejects_rehashed_corrupt_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    complete = _read_complete(complete_path)
    chunk_record = complete["chunks"][0]
    chunk_path = Path(chunk_record["path"])
    chunk_path.write_bytes(b"not-an-npz")
    chunk_record["sha256"] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    chunk_record["size_bytes"] = chunk_path.stat().st_size
    _rewrite_complete(complete_path, complete)

    with pytest.raises(RuntimeError, match="corrupt chunk NPZ"):
        owner.require_group_a_checkpoint_complete(complete_path)


@pytest.mark.parametrize(
    ("metadata_case", "message"),
    [
        ("completion_schema", "completion schema mismatch"),
        ("manifest_schema", "manifest schema version mismatch"),
        ("chunk_order", "chunk path/order mismatch"),
        ("npz_identity", "chunk NPZ string metadata mismatch"),
        ("npz_time_dtype", "chunk NPZ time shape/dtype mismatch"),
        ("npz_value_dtype", "chunk NPZ value shape/dtype mismatch"),
    ],
)
def test_group_a_checkpoint_validator_rejects_wrong_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_case: str,
    message: str,
) -> None:
    checkpoint_dir, complete_path = _materialize_checkpoint(tmp_path, monkeypatch)
    complete = _read_complete(complete_path)
    if metadata_case == "completion_schema":
        complete["unexpected"] = True
        _rewrite_complete(complete_path, complete)
    elif metadata_case == "manifest_schema":
        manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "wrong"
        manifest_encoded = _canonical_json_bytes(manifest)
        manifest_path.write_bytes(manifest_encoded)
        complete["checkpoint_manifest_sha256"] = hashlib.sha256(
            manifest_encoded
        ).hexdigest()
        _rewrite_complete(complete_path, complete)
    elif metadata_case == "chunk_order":
        complete["chunks"][0], complete["chunks"][1] = (
            complete["chunks"][1],
            complete["chunks"][0],
        )
        _rewrite_complete(complete_path, complete)
    elif metadata_case == "npz_identity":
        _rewrite_chunk(
            complete_path,
            chunk_index=0,
            mutate=lambda arrays: arrays.__setitem__(
                "checkpoint_key", np.array("e" * 64)
            ),
        )
    elif metadata_case == "npz_time_dtype":
        _rewrite_chunk(
            complete_path,
            chunk_index=0,
            mutate=lambda arrays: arrays.__setitem__(
                "times_ns", arrays["times_ns"].astype(np.float64)
            ),
        )
    else:
        _rewrite_chunk(
            complete_path,
            chunk_index=0,
            mutate=lambda arrays: arrays.__setitem__(
                "values", arrays["values"].astype(np.float64)
            ),
        )

    with pytest.raises(RuntimeError, match=message):
        owner.require_group_a_checkpoint_complete(complete_path)
