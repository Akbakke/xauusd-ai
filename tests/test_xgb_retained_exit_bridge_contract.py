from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "gx1/xgb/multihead/xgb_multihead_model_v1.py"
MODULE_NAME = "gx1.xgb.multihead.xgb_multihead_model_v1"
CANONICAL_CLASS_ORDER = ["p_long", "p_short", "p_flat"]
CANONICAL_BRIDGE_ORDER = [
    *CANONICAL_CLASS_ORDER,
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]


def _import_with_fake_bridge_contract(setup: str) -> subprocess.CompletedProcess[str]:
    code = f"""
import sys
import types

contract = types.ModuleType("gx1.contracts.signal_bridge_v1")
{setup}
sys.modules["gx1.contracts.signal_bridge_v1"] = contract
sys.modules.pop("{MODULE_NAME}", None)
import {MODULE_NAME}
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )


def test_retained_exit_xgb_import_has_no_entry_contract_dependency() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "gx1.contracts.signal_bridge_v1" in imported_modules
    assert "gx1.contracts.signal_bridge_v3" not in imported_modules
    assert not any("entry_model_native" in module for module in imported_modules)


def test_retained_exit_xgb_bridge_values_follow_exact_contract_order() -> None:
    from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS
    from gx1.xgb.multihead.xgb_multihead_model_v1 import (
        proba_to_signal_bridge_v1,
    )

    probabilities = np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32)
    bridge = proba_to_signal_bridge_v1(probabilities)
    values_by_field = dict(zip(ORDERED_FIELDS, bridge[0], strict=True))

    assert tuple(ORDERED_FIELDS) == tuple(CANONICAL_BRIDGE_ORDER)
    assert values_by_field["p_long"] == pytest.approx(0.7)
    assert values_by_field["p_short"] == pytest.approx(0.2)
    assert values_by_field["p_flat"] == pytest.approx(0.1)
    assert values_by_field["p_hat"] == pytest.approx(0.7)
    assert values_by_field["uncertainty_score"] == pytest.approx(0.3)
    assert values_by_field["margin_top1_top2"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("setup", "expected_error"),
    (
        (
            f"contract.XGB_PROB_FIELDS_ORDERED = {CANONICAL_CLASS_ORDER!r}\n"
            "contract.ORDERED_FIELDS = "
            f"{[*CANONICAL_CLASS_ORDER, 'uncertainty_score', 'p_hat', 'margin_top1_top2', 'entropy']!r}",
            "[RETAINED_EXIT_XGB_BRIDGE_ORDER_MISMATCH]",
        ),
        (
            "contract.XGB_PROB_FIELDS_ORDERED = "
            f"{['p_short', 'p_long', 'p_flat']!r}\n"
            f"contract.ORDERED_FIELDS = {CANONICAL_BRIDGE_ORDER!r}",
            "[RETAINED_EXIT_XGB_CLASS_ORDER_MISMATCH]",
        ),
    ),
)
def test_retained_exit_xgb_import_rejects_order_drift(
    setup: str,
    expected_error: str,
) -> None:
    result = _import_with_fake_bridge_contract(setup)
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert expected_error in output


def test_retained_exit_xgb_import_rejects_missing_order_contract() -> None:
    result = _import_with_fake_bridge_contract(
        f"contract.XGB_PROB_FIELDS_ORDERED = {CANONICAL_CLASS_ORDER!r}"
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "ImportError" in output
    assert "ORDERED_FIELDS" in output
