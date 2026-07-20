from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.contracts.entry_model_native_sizing_authority_v1 as sizing_module
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    ModelNativeSizingUnavailable,
    historical_fixed_1x_negative_control_metadata,
    learned_sizing_authority_contract_metadata,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    ModelNativeSizingContractError,
    calibrated_sizing_transform,
    recompute_sizing_oos_evidence,
)
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    SizingFinalizationError,
    adopt_learned_sizing,
)
from tests.model_native_sizing_support import (
    write_passing_sizing_calibration_and_proof,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_fixed_is_named_negative_control_with_no_execution_path() -> None:
    reference = historical_fixed_1x_negative_control_metadata()

    assert reference == {
        "name": "historical_fixed_1x",
        "role": "historical_negative_control_only",
        "executable_order_authority": False,
        "current_launch_authority": False,
        "fallback_allowed": False,
    }
    assert not hasattr(sizing_module, "require_fixed_size_application")
    with pytest.raises(ModelNativeSizingUnavailable):
        require_model_native_sizing_authority_contract(
            reference,
            context="current Entry sizing",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )


def test_capital_adoption_without_joint_exit_proof_publishes_terminal_fail(
    tmp_path: Path,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    assert evidence["proof"]["decision"] == "PASS"
    assert evidence["proof"]["evaluation_scope"] == (
        "FULL_TEST_LABEL_HORIZON_SIZING_HEAD_DIAGNOSTIC_ONLY"
    )

    with pytest.raises(SizingFinalizationError, match="joint active-Exit sizing proof"):
        adopt_learned_sizing(
            bundle_dir=evidence["bundle_dir"],
            calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
            proof_path=Path(evidence["oos_proof_artifact"]["json_path"]),
            joint_exit_proof_path=tmp_path / "missing_joint_exit_proof.json",
            authority_root=evidence["authority_root"],
            accepted_via_vedtak="UNIT_MUST_NOT_ADOPT",
        )

    adoption_path = max(
        (evidence["authority_root"] / "adoption").glob(
            "ENTRY_MODEL_NATIVE_SIZING_ADOPTION_*.json"
        )
    )
    terminal = json.loads(adoption_path.read_text(encoding="utf-8"))
    assert terminal["decision"] == "FAIL"
    assert terminal["attempted_stage"] == "adoption"
    assert "joint active-Exit sizing proof" in terminal["failures"][0]

    authority = learned_sizing_authority_contract_metadata(
        adoption_artifact={"json_path": str(adoption_path), "sha256": _sha(adoption_path)}
    )
    with pytest.raises(ModelNativeSizingUnavailable, match="exact keys mismatch"):
        prepare_model_native_sizing_authority(authority, context="unit blocked adoption")


def test_diagnostic_transform_is_monotone_exact_and_recomputable(tmp_path: Path) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    calibration = evidence["calibration"]
    constraints = evidence["runtime_constraints"]
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
        runtime_constraints=evidence["runtime_constraints"],
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
        {"instrument": "EUR_USD"},
        {"margin_rate": 0.04},
        {"unit_step": 2},
        {"maximum_gross_xau_units": 999},
        {"margin_available": -1.0},
        {"account_last_transaction_id": "different-snapshot"},
        {"account_observed_utc": "2026-07-17T12:58:00+00:00"},
    ],
)
def test_missing_or_mismatched_runtime_constraints_never_fall_back(
    tmp_path: Path,
    mutation: dict[str, object] | None,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    observed = (
        None
        if mutation is None
        else {**evidence["runtime_constraints"], **mutation}
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
    constraints = evidence["runtime_constraints"]
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
