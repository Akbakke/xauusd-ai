from __future__ import annotations

import sys

import pytest

from gx1.scripts import backfill_xauusd_m5_bidask_2020_2025 as bounded_backfill
from gx1.scripts import backfill_xauusd_m5_from_oanda as canonical_backfill
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
