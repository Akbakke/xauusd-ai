from pathlib import Path
import subprocess

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    model_native_signal_contract_metadata,
    require_model_native_signal_contract,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


REPO = Path(__file__).resolve().parents[1]


def test_retired_seq215_cannot_satisfy_model_native_signal_contract() -> None:
    contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.retired_seq215_fixture"
        )
    )
    contract["contract_mode"] = "challenger_seq215"
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_SIGNAL_CONTRACT_INVALID"):
        require_model_native_signal_contract(contract, context="RETIRED_SEQ215_TEST")


def test_retired_direction_control_routes_are_physically_absent_and_fail_closed() -> None:
    control = (REPO / "scripts/entry_next_edge_control.sh").read_text(
        encoding="utf-8"
    )
    for command in (
        "materialize-smoke",
        "materialize-smoke-seq215",
        "smoke-manifest",
        "smoke-manifest-seq215",
        "candidate-readiness",
        "candidate-readiness-seq215",
        "replay-readiness",
        "replay-readiness-seq215",
        "candidate-train",
        "candidate-train-seq215",
        "smoke-train",
        "smoke-train-seq215",
    ):
        assert f"  {command})" not in control
        completed = subprocess.run(
            ["bash", str(REPO / "scripts/entry_next_edge_control.sh"), command],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 2
        assert f"unknown command: {command}" in completed.stderr
