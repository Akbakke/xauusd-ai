from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as technical_val
from gx1.scripts.validate_entry_model_native_technical_checkpoint_v1 import (
    TechnicalValidationError,
    _require_pretest_guard,
    _require_no_test_rows,
)


def test_technical_full_val_clock_guard_rejects_test_rows(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-06-30T23:55:00Z", "2026-07-01T00:00:00Z"], utc=True
            )
        }
    ).to_parquet(path, index=False)

    with pytest.raises(TechnicalValidationError, match="TEST_BOUNDARY_VIOLATION"):
        _require_no_test_rows(path.resolve())


def test_technical_full_val_clock_guard_accepts_strict_pretest_val(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-06-01T22:00:00Z", "2026-06-30T14:55:00Z"], utc=True
            )
        }
    ).to_parquet(path, index=False)

    observed = _require_no_test_rows(path.resolve())
    assert observed["rows"] == 2
    assert observed["timestamps_at_or_after_test_boundary"] == 0


def test_technical_full_val_binds_dataset_id_when_validating_test_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: the seal API rejects a call that omits dataset lineage."""

    captured: dict[str, object] = {}

    def fake_guard(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"access_proof": {"test_dataset_bytes_read": False}}

    monkeypatch.setattr(
        technical_val,
        "require_pretest_or_prefreeze_test_guard_lineage",
        fake_guard,
    )
    _require_pretest_guard(
        guard_json=tmp_path / "guard.json",
        guard_sha256="a" * 64,
        dataset_run_id="PRETEST_V3_20260829T173000Z",
        dataset_dir=tmp_path,
    )
    assert captured["kwargs"] == {
        "expected_dataset_run_id": "PRETEST_V3_20260829T173000Z",
        "expected_dataset_dir": tmp_path,
    }


def test_technical_full_val_session_is_hash_bound_and_monotonic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bounded VAL continuation cannot silently reuse or rewind evidence."""

    monkeypatch.setattr(
        technical_val,
        "_restore_candidate_validation_snapshot",
        lambda snapshot: dict(snapshot),
    )
    session = technical_val._TechnicalValidationSession(
        directory=(tmp_path / "session").resolve(),
        contract={"schema_version": "unit-test-v1", "input_sha256": "a" * 64},
    )
    assert session.load(expected_batches=10) is None

    session.save(
        expected_batches=10,
        next_batch_offset=3,
        validation_snapshot={"accumulator": torch.tensor([1, 2, 3])},
        elapsed_seconds_completed=12.5,
        complete=False,
    )
    restored = session.load(expected_batches=10)
    assert restored is not None
    assert restored["next_batch_offset"] == 3
    assert restored["elapsed_seconds_completed"] == 12.5

    with pytest.raises(TechnicalValidationError, match="NONMONOTONIC_PROGRESS"):
        session.save(
            expected_batches=10,
            next_batch_offset=3,
            validation_snapshot={"accumulator": torch.tensor([1, 2, 3])},
            elapsed_seconds_completed=13.0,
            complete=False,
        )
