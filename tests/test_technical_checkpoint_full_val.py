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


def test_direction_witness_cli_routes_before_legacy_val_parser(monkeypatch, tmp_path):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    observed = []
    monkeypatch.setattr(direction, "run", lambda **kwargs: observed.append(kwargs) or {"status": "source-only-test"})
    plan = tmp_path / "plan.json"
    out = (tmp_path / "out").resolve()
    result = technical_val.main(["--dynamic-m5-direction-witness-plan", str(plan),
        "--dynamic-m5-direction-witness-plan-sha256", "a" * 64,
        "--device", "cuda", "--out-dir", str(out)])
    assert result == 0
    assert observed == [{"plan_path": plan, "plan_sha256": "a" * 64, "device": "cuda", "out_dir": out}]


@pytest.mark.parametrize("extra", [
    ["--device", "cuda"],
    ["--dynamic-m5-direction-witness-plan", "another.json"],
    ["--bundle-dir", "legacy-bundle"],
    ["--dynamic-compact-input-influence-plan", "another-mode.json"],
])
def test_direction_witness_cli_rejects_duplicate_or_mixed_scope(monkeypatch, tmp_path, extra):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    calls = []
    monkeypatch.setattr(direction, "run", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit):
        technical_val.main(["--dynamic-m5-direction-witness-plan", str(tmp_path / "plan.json"),
            "--dynamic-m5-direction-witness-plan-sha256", "a" * 64,
            "--device", "cuda", "--out-dir", str((tmp_path / "out").resolve()), *extra])
    assert calls == []


@pytest.mark.parametrize("change", [
    {"training_enabled": True}, {"exit_evaluation": True}, {"test_data_used": True},
    {"max_forward_calls": 2}, {"max_reverse_calls": 23},
    {"headline_horizons": [1, 5, 15, 60, 120, 240, 481]},
])
def test_direction_witness_plan_rejects_scope_expansion(change):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    plan = {"schema_version": direction.PLAN_SCHEMA, "run_id": "fixed-train-witness",
        "policy_stage_key": "unit-scope", "input_scope": {"path": "/unused.json", "sha256": "a" * 64},
        "output_directory": "/unused-output", "one_shot_invocations": 1,
        "max_wall_seconds": 900, "max_forward_calls": 1, "max_reverse_calls": 22,
        "headline_horizons": [1, 5, 15, 60, 120, 240, 480],
        **{key: False for key in direction.FLAGS}}
    direction._require_plan(plan)
    with pytest.raises(RuntimeError, match="PLAN_BUDGET_OR_AUTHORITY"):
        direction._require_plan({**plan, **change})
