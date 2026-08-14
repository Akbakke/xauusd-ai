from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    FIT_POPULATION,
    TRANSFORM,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
from gx1.features.entry_volatility_semantics_v1 import (
    BPS_PER_UNIT_FRACTION,
    LOCAL_VOLATILITY_DECISION_CLOCKS,
    RVOL_20_WINDOW_BARS,
    pk_sigma20_per_bar_bps,
    rvol_20_per_bar_bps,
)


@pytest.mark.parametrize("decision_clock", LOCAL_VOLATILITY_DECISION_CLOCKS)
def test_rvol_13_18_60_is_unit_correct_on_each_native_clock(
    decision_clock: str,
) -> None:
    # These are realistic raw rvol_20 magnitudes in its declared
    # bps*sqrt(20) unit: 18.44 was the measured TRAIN median and 60 represents
    # a high-volatility observation.  Conversion must neither saturate nor
    # reinterpret twenty M1 bars as twenty M5 bars.
    raw_rvol = np.asarray([13.0, 18.0, 60.0], dtype=np.float32)
    expected_per_bar_bps = raw_rvol / np.float32(np.sqrt(RVOL_20_WINDOW_BARS))
    np.testing.assert_allclose(
        rvol_20_per_bar_bps(raw_rvol, decision_clock=decision_clock),
        expected_per_bar_bps,
        rtol=1e-7,
        atol=0.0,
    )


@pytest.mark.parametrize("decision_clock", LOCAL_VOLATILITY_DECISION_CLOCKS)
def test_rvol_and_parkinson_estimators_share_per_bar_bps_unit(
    decision_clock: str,
) -> None:
    raw_rvol = np.asarray([13.0, 18.0, 60.0], dtype=np.float32)
    per_bar_bps = raw_rvol / np.float32(np.sqrt(RVOL_20_WINDOW_BARS))
    dimensionless_parkinson = per_bar_bps / np.float32(BPS_PER_UNIT_FRACTION)
    np.testing.assert_allclose(
        rvol_20_per_bar_bps(raw_rvol, decision_clock=decision_clock),
        pk_sigma20_per_bar_bps(
            dimensionless_parkinson,
            decision_clock=decision_clock,
        ),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("decision_clock", ["M1", "m15", "", None])
def test_local_volatility_conversion_requires_exact_m1_or_m5_clock(
    decision_clock: object,
) -> None:
    with pytest.raises(RuntimeError, match="LOCAL_CLOCK_INVALID"):
        rvol_20_per_bar_bps(
            np.asarray([18.0], dtype=np.float32),
            decision_clock=decision_clock,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("converter", "value", "error"),
    [
        (rvol_20_per_bar_bps, -1.0, "RVOL_NEGATIVE"),
        (pk_sigma20_per_bar_bps, -1.0e-4, "PARKINSON_NEGATIVE"),
    ],
)
def test_realized_volatility_magnitudes_fail_closed_on_negative_input(
    converter,
    value: float,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        converter(np.asarray([value], dtype=np.float32), decision_clock="m5")


def test_raw_rvol_is_train_normalized_and_no_active_feature_reintroduces_tanh() -> None:
    assert "rvol_20" in MODEL_NATIVE_BASE_FIELDS
    assert "train_only" in TRANSFORM
    assert FIT_POPULATION == "unique_physical_train_rows_entry_exit_union_v2"

    features_root = Path(__file__).resolve().parents[1] / "gx1" / "features"
    violations: list[str] = []
    for source_path in sorted(features_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if function_name not in {"tanh", "_tanh"}:
                continue
            if "rvol_20" in ast.unparse(node):
                violations.append(
                    f"{source_path.relative_to(features_root.parent.parent)}:"
                    f"{node.lineno}"
                )
    assert violations == []
