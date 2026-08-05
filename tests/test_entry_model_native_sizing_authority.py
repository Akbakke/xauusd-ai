from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    ModelNativeSizingContractError,
    calibrated_sizing_transform,
    recompute_sizing_oos_evidence,
)
from tests.model_native_sizing_support import (
    write_passing_sizing_calibration_and_proof,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_transform_is_monotone_exact_and_recomputable(tmp_path: Path) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    calibration = evidence["calibration"]
    constraints = evidence["offline_constraints"]
    applications = [
        calibrated_sizing_transform(
            calibration=calibration,
            position_size_logit=logit,
            model_direction_index=0,
            runtime_constraints=constraints,
            context="unit diagnostic transform",
        )
        for logit in (-4.0, 0.0, 4.0)
    ]

    fractions = [row["calibrated_size_fraction"] for row in applications]
    units = [row["units"] for row in applications]
    assert fractions[0] < fractions[1] < fractions[2]
    assert units == sorted(units)
    reference_fraction = calibration["parameters"]["reference_capacity_fraction"]
    assert applications[1]["reference_pre_round_units"] == pytest.approx(
        applications[1]["capacity_units"] * reference_fraction
    )
    assert applications[1]["pre_round_units"] == pytest.approx(
        applications[1]["capacity_units"]
        * applications[1]["calibrated_size_fraction"]
    )


def test_flat_is_exact_zero_without_changing_direction(tmp_path: Path) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    application = calibrated_sizing_transform(
        calibration=evidence["calibration"],
        position_size_logit=9.0,
        model_direction_index=2,
        runtime_constraints=evidence["offline_constraints"],
        context="unit flat sizing diagnostic",
    )

    assert application["model_direction_index"] == 2
    assert application["units"] == 0
    assert application["authorized_order"] is False
    assert application["no_order_reason"] == "MODEL_DIRECTION_FLAT"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        {"account_equity": float("nan")},
        {"account_equity": 0.0},
        {"instrument": "INVALID_INSTRUMENT"},
        {"margin_rate": 0.04},
        {"unit_step": 2},
        {"maximum_gross_xau_units": 999},
        {"margin_available": -1.0},
        {"fact_provenance_mode": "broker_live"},
    ],
)
def test_missing_or_mismatched_offline_constraints_never_fall_back(
    tmp_path: Path,
    mutation: dict[str, object] | None,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    observed = (
        None
        if mutation is None
        else {**evidence["offline_constraints"], **mutation}
    )

    with pytest.raises(ModelNativeSizingContractError):
        calibrated_sizing_transform(
            calibration=evidence["calibration"],
            position_size_logit=0.0,
            model_direction_index=1,
            runtime_constraints=observed,
            context="unit missing sizing constraints",
        )


def test_immutable_drawdown_and_margin_caps_return_no_order(tmp_path: Path) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    constraints = evidence["offline_constraints"]
    drawdown = {
        **constraints,
        "account_equity": 9_700.0,
        "account_balance": 10_000.0,
        "account_floating_drawdown_bps": 300.0,
    }
    no_margin = {**constraints, "margin_available": 0.0}

    dd_application = calibrated_sizing_transform(
        calibration=evidence["calibration"],
        position_size_logit=8.0,
        model_direction_index=0,
        runtime_constraints=drawdown,
        context="unit immutable drawdown",
    )
    margin_application = calibrated_sizing_transform(
        calibration=evidence["calibration"],
        position_size_logit=8.0,
        model_direction_index=1,
        runtime_constraints=no_margin,
        context="unit immutable margin",
    )

    assert dd_application["units"] == 0
    assert dd_application["no_order_reason"] == (
        "IMMUTABLE_ACCOUNT_FLOATING_DRAWDOWN_CAP"
    )
    assert margin_application["units"] == 0
    assert margin_application["no_order_reason"] == "INSUFFICIENT_ADMISSIBLE_CAPACITY"


def test_oos_prediction_direction_rejects_fraction_before_coercion(
    tmp_path: Path,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    bindings = copy.deepcopy(evidence["source_bindings"])
    source = Path(bindings["oos_rows"]["path"])
    path = tmp_path / (
        "entry_model_native_sizing_oos_rows_20260717T120000123456Z.parquet"
    )
    frame = pd.read_parquet(source)
    frame["model_direction_index"] = frame["model_direction_index"].astype(float)
    frame.loc[0, "model_direction_index"] = 0.5
    frame.to_parquet(path, index=False)
    bindings["oos_rows"] = {"path": str(path.resolve()), "sha256": _sha(path)}

    with pytest.raises(ModelNativeSizingContractError, match="exact integer table cell"):
        recompute_sizing_oos_evidence(
            calibration=evidence["calibration"],
            source_bindings=bindings,
            evaluation_bundle=evidence["oos_source"]["evaluation_bundle"],
            context="unit fractional direction",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("entry_bid", float("nan"), "non-finite"),
        ("exit_ask", -1.0, "finite and positive"),
        ("entry_bid", 2_501.0, "entry_bid must be <= entry_ask"),
    ],
)
def test_oos_market_and_account_numerics_fail_closed(
    tmp_path: Path,
    field: str,
    value: float,
    match: str,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    bindings = copy.deepcopy(evidence["source_bindings"])
    source = Path(bindings["oos_rows"]["path"])
    path = tmp_path / (
        "entry_model_native_sizing_oos_rows_20260717T120000123456Z.parquet"
    )
    frame = pd.read_parquet(source)
    frame.loc[0, field] = value
    frame.to_parquet(path, index=False)
    bindings["oos_rows"] = {"path": str(path.resolve()), "sha256": _sha(path)}

    with pytest.raises(ModelNativeSizingContractError, match=match):
        recompute_sizing_oos_evidence(
            calibration=evidence["calibration"],
            source_bindings=bindings,
            evaluation_bundle=evidence["oos_source"]["evaluation_bundle"],
            context="unit invalid market facts",
        )
