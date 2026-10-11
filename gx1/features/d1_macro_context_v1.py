"""D1 macro/positioning context at native M5 decision times (operator decision 11.10.2026).

The ninth context family for the native Entry model: the research surface's
positioning (CFTC COT), cross-asset panel (OANDA D1 session bars) and macro
driver fields, joined as-of to every M5 decision time with exactly the
availability rules the research owner declares:

* COT: a report is usable ``availability_lag_days`` after its as-of Tuesday.
* Panel: the last D1 session bar whose close (label + one session) is at or
  before the decision time; carried across the instrument's own holidays,
  never filled from a later bar.
* Drivers: an observation dated D (New York close) is usable strictly after
  the end of day D+1 America/New_York.

Source parsing stays with the research owner (``research_ta_campaign_v1``);
this module owns only the field order, the join at M5 times and the binding
that the dataset builder will consume. No field is admitted to the native
signal manifest by this module; that is a separate contract step.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.scripts.research_ta_campaign_v1 import (
    SWING_COT_FIELDS, SWING_DRIVER_FIELDS, SWING_PANEL_RETURN_DAYS, SWING_PANEL_RV_DAYS,
    TRADING_SESSION_DURATION, align_asof, build_driver_components, load_oanda_panel_d1,
    parse_cot_gold_archives, sealed_period_read_end,
)

D1_MACRO_CONTEXT_SCHEMA_VERSION = "d1_macro_context_v1"
D1_MACRO_CONTEXT_GROUPS = ("positioning_cot", "panel_oanda", "drivers")
D1_MACRO_PANEL_INSTRUMENTS = (
    "BCO_USD", "EUR_USD", "SPX500_USD", "USB10Y_USD", "USD_JPY", "WTICO_USD",
    "XAG_USD", "XAU_USD", "XCU_USD", "XPD_USD", "XPT_USD",
)
D1_MACRO_PANEL_FIELDS = tuple(
    column for instrument in D1_MACRO_PANEL_INSTRUMENTS
    for column in ([f"{instrument}_logret_{days}D" for days in SWING_PANEL_RETURN_DAYS]
                   + [f"{instrument}_rv_{SWING_PANEL_RV_DAYS}D"])
)
D1_MACRO_CONTEXT_FIELDS_BY_GROUP = {
    "positioning_cot": tuple(SWING_COT_FIELDS),
    "panel_oanda": D1_MACRO_PANEL_FIELDS,
    "drivers": tuple(SWING_DRIVER_FIELDS),
}
D1_MACRO_CONTEXT_FIELDS = tuple(
    field for group in D1_MACRO_CONTEXT_GROUPS for field in D1_MACRO_CONTEXT_FIELDS_BY_GROUP[group]
)
D1_MACRO_CONTEXT_DIM = len(D1_MACRO_CONTEXT_FIELDS)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def d1_macro_context_contract() -> dict[str, Any]:
    value = {
        "schema_version": D1_MACRO_CONTEXT_SCHEMA_VERSION,
        "groups": list(D1_MACRO_CONTEXT_GROUPS),
        "fields_by_group": {g: list(f) for g, f in D1_MACRO_CONTEXT_FIELDS_BY_GROUP.items()},
        "fields": list(D1_MACRO_CONTEXT_FIELDS),
        "dim": D1_MACRO_CONTEXT_DIM,
        "panel_instruments": list(D1_MACRO_PANEL_INSTRUMENTS),
        "availability": {
            "positioning_cot": "report as-of Tuesday usable strictly after as_of + availability_lag_days (declared in the source spec)",
            "panel_oanda": "last D1 session bar whose close (label + one session) is at or before the decision time; carried across the instrument's own holidays",
            "drivers": "observation dated D (New York close) usable strictly after the end of day D+1 America/New_York; each series carried across its own holidays",
        },
        "join_clock": "native M5 decision time (bar open + 5 minutes), UTC",
        "future_values_used": False,
        "admitted_to_native_signal_manifest": False,
    }
    return {**value, "contract_sha256": _sha(value)}


def require_d1_macro_context_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The source spec carries exactly the research owner's bindings; anything else fails closed."""
    required = {"test_seal_utc", "cot", "oanda_panel", "drivers"}
    if not isinstance(spec, Mapping) or not required.issubset(spec):
        raise RuntimeError("D1_MACRO_CONTEXT_SPEC_INVALID")
    if tuple(spec["oanda_panel"].get("instruments", ())) != D1_MACRO_PANEL_INSTRUMENTS:
        raise RuntimeError("D1_MACRO_CONTEXT_PANEL_INSTRUMENTS_INVALID")
    if spec["cot"].get("contract_market_code") != "088691" or not spec["drivers"].get("gld_item"):
        raise RuntimeError("D1_MACRO_CONTEXT_SOURCE_SCOPE_INVALID")
    return dict(spec)


def d1_macro_context_components(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Parse every source once on its own clock (cut at the spec's read end) with availability stamps."""
    checked = require_d1_macro_context_spec(spec)
    end = sealed_period_read_end(checked)
    cot = parse_cot_gold_archives(checked, checked["cot"]["contract_market_code"])
    daily = load_oanda_panel_d1(checked)
    if tuple(sorted(daily)) != D1_MACRO_PANEL_INSTRUMENTS:
        raise RuntimeError("D1_MACRO_CONTEXT_PANEL_INSTRUMENTS_MISSING")
    drivers = build_driver_components(checked)
    return {"read_end": end, "cot": cot, "panel": daily, "drivers": drivers}


def build_d1_macro_context(decision_times: pd.DatetimeIndex, components: Mapping[str, Any]) -> pd.DataFrame:
    """The 61 fields as-of each decision time, in contract order; NaN before a source's first usable row."""
    times = pd.DatetimeIndex(decision_times)
    if times.tz is None or str(times.tz) != "UTC" or not times.is_monotonic_increasing or times.has_duplicates:
        raise RuntimeError("D1_MACRO_CONTEXT_DECISION_CLOCK_INVALID")
    if bool((times > components["read_end"]).any()):
        raise RuntimeError("D1_MACRO_CONTEXT_BEYOND_READ_END")
    out = pd.DataFrame(index=range(len(times)))
    cot = components["cot"]
    joined = align_asof(times, cot["available_at"], cot[list(SWING_COT_FIELDS)], inclusive=False)
    for column in SWING_COT_FIELDS:
        out[column] = joined[column].to_numpy(float)
    for instrument in D1_MACRO_PANEL_INSTRUMENTS:
        features = components["panel"][instrument]
        closes = pd.Series(pd.DatetimeIndex(features.index) + TRADING_SESSION_DURATION)
        joined = align_asof(times, closes, features, inclusive=True)
        for column in features.columns:
            out[column] = joined[column].to_numpy(float)
    drivers = components["drivers"]
    joined = align_asof(times, drivers["available_at"], drivers[list(SWING_DRIVER_FIELDS)], inclusive=False)
    for column in SWING_DRIVER_FIELDS:
        out[column] = joined[column].to_numpy(float)
    if list(out.columns) != list(D1_MACRO_CONTEXT_FIELDS):
        raise RuntimeError("D1_MACRO_CONTEXT_FIELD_ORDER")
    values = out.to_numpy(float)
    if np.isinf(values).any():
        raise RuntimeError("D1_MACRO_CONTEXT_NONFINITE")
    out.insert(0, "decision_time", times)
    return out


def d1_macro_context_binding(spec: Mapping[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Immutable identity of one materialization: contract, sources, clock and value hashes."""
    values = frame[list(D1_MACRO_CONTEXT_FIELDS)].to_numpy(np.float64)
    complete = np.isfinite(values).all(axis=1)
    first_complete = str(frame["decision_time"].iloc[int(np.argmax(complete))]) if complete.any() else None
    value = {
        "schema_version": D1_MACRO_CONTEXT_SCHEMA_VERSION + "_binding",
        "contract": d1_macro_context_contract(),
        "sources": {key: spec[key] for key in ("cot", "oanda_panel", "drivers")},
        "test_seal_utc": spec["test_seal_utc"],
        "forward_journal": spec.get("forward_journal"),
        "rows": int(len(frame)),
        "complete_rows": int(complete.sum()),
        "first_complete_decision_time": first_complete,
        "decision_clock_sha256": hashlib.sha256(
            pd.DatetimeIndex(frame["decision_time"]).as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest(),
        "values_sha256": hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest(),
    }
    return {**value, "binding_sha256": _sha(value)}
