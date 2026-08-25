"""Observed OANDA OrderFill economics, without synthetic cost assumptions.

This is an execution-journal contract, not a historical backtest cost model.
It preserves the broker's own transaction values when they are explicitly
present and makes every omitted or malformed cost component visible.  In
particular, an absent commission is *not* interpreted as zero commission.

The observed ``halfSpreadCost`` is diagnostic only: it is already reflected in
the broker's executed fill price and must not be subtracted again from realised
cash flow.  A complete fill still cannot prove production economics on its own;
that additionally requires a bound decision/fill population, financing over
the holding interval, and a shared-portfolio replay.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


OANDA_FILL_ECONOMICS_SCHEMA_VERSION = "gx1_oanda_order_fill_economics_v1"
_REQUIRED_CASHFLOW_FIELDS = (
    "pl_account_units",
    "financing_account_units",
    "commission_account_units",
    "guaranteed_execution_fee_account_units",
)


def _finite_number(
    value: object,
    *,
    field: str,
    nonnegative: bool,
    failures: list[str],
) -> float | None:
    """Parse one literal broker number without replacing absence by zero."""

    if value is None:
        failures.append(f"{field}_missing")
        return None
    if isinstance(value, bool):
        failures.append(f"{field}_not_numeric")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{field}_not_numeric")
        return None
    if not math.isfinite(parsed):
        failures.append(f"{field}_not_finite")
        return None
    if nonnegative and parsed < 0.0:
        failures.append(f"{field}_negative")
        return None
    return parsed


def observed_oanda_order_fill_economics(
    order_fill_transaction: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Return exact observed economics or an explicit incomplete observation.

    No exception is raised merely because a cost field is absent: the order may
    already have executed, so the caller must retain the transaction.  Instead
    the returned status is not complete and ``net_cashflow_account_units`` is
    ``None``. Structural absence of the transaction itself is an API misuse and
    raises.
    """

    if not isinstance(order_fill_transaction, Mapping):
        raise RuntimeError(f"[{context}_OANDA_ORDER_FILL_NOT_MAPPING]")
    fill = dict(order_fill_transaction)
    failures: list[str] = []
    if fill.get("type") != "ORDER_FILL":
        failures.append("transaction_type_not_order_fill")

    pl = _finite_number(
        fill.get("pl"),
        field="pl_account_units",
        nonnegative=False,
        failures=failures,
    )
    financing = _finite_number(
        fill.get("financing"),
        field="financing_account_units",
        nonnegative=False,
        failures=failures,
    )
    commission = _finite_number(
        fill.get("commission"),
        field="commission_account_units",
        nonnegative=True,
        failures=failures,
    )
    guaranteed_fee = _finite_number(
        fill.get("guaranteedExecutionFee"),
        field="guaranteed_execution_fee_account_units",
        nonnegative=True,
        failures=failures,
    )
    # OANDA documents this as an observed diagnostic and permits either sign.
    half_spread = _finite_number(
        fill.get("halfSpreadCost"),
        field="half_spread_cost_account_units",
        nonnegative=False,
        failures=failures,
    )

    values = {
        "pl_account_units": pl,
        "financing_account_units": financing,
        "commission_account_units": commission,
        "guaranteed_execution_fee_account_units": guaranteed_fee,
    }
    missing_cashflow = [
        field for field in _REQUIRED_CASHFLOW_FIELDS if values[field] is None
    ]
    complete = not failures and not missing_cashflow
    net_cashflow = (
        float(pl + financing - commission - guaranteed_fee)
        if complete
        and pl is not None
        and financing is not None
        and commission is not None
        and guaranteed_fee is not None
        else None
    )
    return {
        "schema_version": OANDA_FILL_ECONOMICS_SCHEMA_VERSION,
        "source": "literal_oanda_order_fill_transaction_v20",
        "economics_status": (
            "COMPLETE_OBSERVED_FILL_CASHFLOW"
            if complete
            else "INCOMPLETE_OR_INVALID_OBSERVED_FILL_CASHFLOW"
        ),
        "failures": sorted(set(failures)),
        "pl_account_units": pl,
        "financing_account_units": financing,
        "commission_account_units": commission,
        "guaranteed_execution_fee_account_units": guaranteed_fee,
        "half_spread_cost_account_units": half_spread,
        "half_spread_cost_treatment": (
            "diagnostic_only_embedded_in_observed_executed_fill_price"
        ),
        "net_cashflow_account_units": net_cashflow,
        "production_economics_claim_allowed": False,
    }


__all__ = [
    "OANDA_FILL_ECONOMICS_SCHEMA_VERSION",
    "observed_oanda_order_fill_economics",
]
