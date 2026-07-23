from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gx1.scripts import backfill_xauusd_m5_bidask_2020_2025 as bounded_backfill
from gx1.scripts import backfill_xauusd_m5_from_oanda as canonical_backfill
from gx1.execution import v12_m1_to_m5_downsample as m1_downsample
from gx1_guards.gates import GateError


@pytest.mark.parametrize("module", [bounded_backfill, canonical_backfill])
def test_backfill_cli_requires_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("module", [bounded_backfill, canonical_backfill])
def test_backfill_cli_rejects_invalid_vedtak_before_side_effect_setup(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__), "--vedtak", "short"])
    monkeypatch.setattr(
        module,
        "load_dotenv_if_present",
        lambda: pytest.fail("environment loading happened before the vedtak gate"),
    )

    with pytest.raises(GateError, match="blocked"):
        module.main()


def test_m1_downsample_cannot_write_canonical_m5_root() -> None:
    with pytest.raises(RuntimeError, match="CANONICAL_M5_SINGLE_OWNER_VIOLATION"):
        m1_downsample.main()


def test_bounded_backfill_cannot_target_canonical_m5_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "canonical_m5"
    monkeypatch.setattr(
        bounded_backfill,
        "PROTECTED_CANONICAL_M5_ROOT",
        protected,
    )

    with pytest.raises(RuntimeError, match="CANONICAL_M5_SINGLE_OWNER_VIOLATION"):
        bounded_backfill._reject_canonical_root_output(
            protected / "year=2026" / "part-000.parquet"
        )
