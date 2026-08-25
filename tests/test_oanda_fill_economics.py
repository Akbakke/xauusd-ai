from __future__ import annotations

import pytest

from gx1.contracts.oanda_fill_economics_v1 import (
    OANDA_FILL_ECONOMICS_SCHEMA_VERSION,
    observed_oanda_order_fill_economics,
)


def _complete_fill() -> dict[str, str]:
    return {
        "type": "ORDER_FILL",
        "pl": "2.50",
        "financing": "-0.40",
        "commission": "0.25",
        "guaranteedExecutionFee": "0.00",
        "halfSpreadCost": "0.12",
    }


def test_observed_fill_economics_keeps_explicit_components_without_double_counting() -> None:
    observed = observed_oanda_order_fill_economics(
        _complete_fill(), context="UNIT_COMPLETE"
    )

    assert observed["schema_version"] == OANDA_FILL_ECONOMICS_SCHEMA_VERSION
    assert observed["economics_status"] == "COMPLETE_OBSERVED_FILL_CASHFLOW"
    assert observed["failures"] == []
    assert observed["net_cashflow_account_units"] == pytest.approx(1.85)
    assert observed["half_spread_cost_account_units"] == pytest.approx(0.12)
    assert "diagnostic_only" in observed["half_spread_cost_treatment"]
    assert observed["production_economics_claim_allowed"] is False


def test_missing_commission_never_becomes_zero_or_complete() -> None:
    fill = _complete_fill()
    del fill["commission"]

    observed = observed_oanda_order_fill_economics(fill, context="UNIT_MISSING")

    assert observed["economics_status"] == (
        "INCOMPLETE_OR_INVALID_OBSERVED_FILL_CASHFLOW"
    )
    assert observed["commission_account_units"] is None
    assert observed["net_cashflow_account_units"] is None
    assert observed["failures"] == ["commission_account_units_missing"]


def test_untyped_mapping_cannot_become_complete_cashflow_evidence() -> None:
    fill = _complete_fill()
    del fill["type"]

    observed = observed_oanda_order_fill_economics(fill, context="UNIT_TYPE")

    assert observed["economics_status"] == (
        "INCOMPLETE_OR_INVALID_OBSERVED_FILL_CASHFLOW"
    )
    assert observed["failures"] == ["transaction_type_not_order_fill"]
    assert observed["net_cashflow_account_units"] is None


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("commission", "-0.01", "commission_account_units_negative"),
        ("financing", "NaN", "financing_account_units_not_finite"),
        ("pl", True, "pl_account_units_not_numeric"),
    ],
)
def test_invalid_broker_cost_facts_are_preserved_as_invalid_not_repaired(
    field: str, value: object, failure: str
) -> None:
    fill = _complete_fill()
    fill[field] = value  # type: ignore[assignment]

    observed = observed_oanda_order_fill_economics(fill, context="UNIT_INVALID")

    assert observed["economics_status"] == (
        "INCOMPLETE_OR_INVALID_OBSERVED_FILL_CASHFLOW"
    )
    assert failure in observed["failures"]
    assert observed["net_cashflow_account_units"] is None


def test_non_mapping_fill_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="OANDA_ORDER_FILL_NOT_MAPPING"):
        observed_oanda_order_fill_economics(None, context="UNIT_SHAPE")
