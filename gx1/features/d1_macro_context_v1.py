"""D1 macro/positioning context at native M5 decision times (operator decision 11.10.2026).

The ninth context family for the native Entry model: the research surface's
positioning (CFTC COT), cross-asset panel (OANDA D1 session bars) and macro
driver fields, joined as-of to every M5 decision time with exactly the
availability rules the research owner declares:

* COT: a report is usable after the later of the declared lag and the end of its
  official delayed-publication date; date metadata never supplies position values.
* Panel: the last D1 session bar whose close (label + one session) is at or
  before the decision time; carried across the instrument's own holidays,
  never filled from a later bar.
* Drivers: an observation dated D (New York close) is usable strictly after
  the end of day D+1 America/New_York.

This module owns the field names and order (the research owner imports them),
the join at M5 times and the materialization binding the dataset builder
consumes. Source parsing stays with ``research_ta_campaign_v1`` and is
imported lazily so the native contracts can import these names without the
research stack. The committed source spec is pinned by hash below: changing
the sources is a code change, never an ambient path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.time.session_detector import TRADING_SESSION_DURATION

D1_MACRO_CONTEXT_SCHEMA_VERSION = "d1_macro_context_v1"
D1_MACRO_CONTEXT_GROUPS = ("positioning_cot", "panel_oanda", "drivers")
D1_MACRO_COT_FIELDS = (
    "cot_noncommercial_net_over_oi", "cot_change4reports",
    "cot_open_interest_log_change4reports", "cot_commercial_net_over_oi",
)
D1_MACRO_PANEL_RETURN_DAYS = (1, 5, 20)
D1_MACRO_PANEL_RV_DAYS = 20
D1_MACRO_PANEL_INSTRUMENTS = (
    "BCO_USD", "EUR_USD", "SPX500_USD", "USB10Y_USD", "USD_JPY", "WTICO_USD",
    "XAG_USD", "XAU_USD", "XCU_USD", "XPD_USD", "XPT_USD",
)
D1_MACRO_PANEL_FIELDS = tuple(
    column for instrument in D1_MACRO_PANEL_INSTRUMENTS
    for column in ([f"{instrument}_logret_{days}D" for days in D1_MACRO_PANEL_RETURN_DAYS]
                   + [f"{instrument}_rv_{D1_MACRO_PANEL_RV_DAYS}D"])
)
D1_MACRO_DRIVER_FIELDS = (
    "real10y_level", "real10y_change1D", "real10y_change5D", "real10y_change21D",
    "nominal10y_level", "slope10y2y_level", "slope10y2y_change21D",
    "breakeven10y_level", "breakeven10y_change21D",
    "vix_log_level", "vix_log_change5D",
    "gld_log_level", "gld_log_change21D",
)
D1_MACRO_CONTEXT_FIELDS_BY_GROUP = {
    "positioning_cot": D1_MACRO_COT_FIELDS,
    "panel_oanda": D1_MACRO_PANEL_FIELDS,
    "drivers": D1_MACRO_DRIVER_FIELDS,
}
D1_MACRO_CONTEXT_FIELDS = tuple(
    field for group in D1_MACRO_CONTEXT_GROUPS for field in D1_MACRO_CONTEXT_FIELDS_BY_GROUP[group]
)
D1_MACRO_CONTEXT_DIM = len(D1_MACRO_CONTEXT_FIELDS)
# The committed source spec the native dataset builder reads; pinned by hash (rule 2a).
D1_MACRO_CONTEXT_SOURCE_SPEC = {
    "path": "configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json",
    "sha256": "b8e2c01ddc5c1941dd509f1c3507a4211b2d0f7e3820f64c7607e95a0916d6b3",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _research():
    from gx1.scripts import research_ta_campaign_v1 as research
    return research


def d1_macro_context_contract() -> dict[str, Any]:
    value = {
        "schema_version": D1_MACRO_CONTEXT_SCHEMA_VERSION,
        "groups": list(D1_MACRO_CONTEXT_GROUPS),
        "fields_by_group": {g: list(f) for g, f in D1_MACRO_CONTEXT_FIELDS_BY_GROUP.items()},
        "fields": list(D1_MACRO_CONTEXT_FIELDS),
        "dim": D1_MACRO_CONTEXT_DIM,
        "panel_instruments": list(D1_MACRO_PANEL_INSTRUMENTS),
        "availability": {
            "positioning_cot": "report usable strictly after max(declared as_of lag, end of bound official late-publication date)",
            "panel_oanda": "last D1 session bar whose close (label + one session) is at or before the decision time; carried across the instrument's own holidays",
            "drivers": "observation dated D (New York close) usable strictly after the end of day D+1 America/New_York; each series carried across its own holidays",
        },
        "join_clock": "native M5 decision time (bar open + 5 minutes), UTC",
        "future_values_used": False,
        "specialist_owner": "session_regime_encoder",
        "source_spec": dict(D1_MACRO_CONTEXT_SOURCE_SPEC),
    }
    return {**value, "contract_sha256": _sha(value)}


def load_d1_macro_context_source_spec(repo_root: Path) -> dict[str, Any]:
    """The pinned committed spec; any byte change is a code change."""
    path = Path(repo_root) / D1_MACRO_CONTEXT_SOURCE_SPEC["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != D1_MACRO_CONTEXT_SOURCE_SPEC["sha256"]:
        raise RuntimeError("D1_MACRO_CONTEXT_SOURCE_SPEC_HASH")
    return require_d1_macro_context_spec(json.loads(path.read_text()))


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
    research = _research()
    end = research.sealed_period_read_end(checked)
    cot = research.parse_cot_gold_archives(checked, checked["cot"]["contract_market_code"])
    daily = research.load_oanda_panel_d1(checked)
    if tuple(sorted(daily)) != D1_MACRO_PANEL_INSTRUMENTS:
        raise RuntimeError("D1_MACRO_CONTEXT_PANEL_INSTRUMENTS_MISSING")
    drivers = research.build_driver_components(checked)
    return {"read_end": end, "cot": cot, "panel": daily, "drivers": drivers}


def build_d1_macro_context(decision_times: pd.DatetimeIndex, components: Mapping[str, Any]) -> pd.DataFrame:
    """The 61 fields as-of each decision time, in contract order; NaN before a source's first usable row."""
    align_asof = _research().align_asof
    times = pd.DatetimeIndex(decision_times)
    if times.tz is None or str(times.tz) != "UTC" or not times.is_monotonic_increasing or times.has_duplicates:
        raise RuntimeError("D1_MACRO_CONTEXT_DECISION_CLOCK_INVALID")
    if bool((times > components["read_end"]).any()):
        raise RuntimeError("D1_MACRO_CONTEXT_BEYOND_READ_END")
    out = pd.DataFrame(index=range(len(times)))
    cot = components["cot"]
    joined = align_asof(times, cot["available_at"], cot[list(D1_MACRO_COT_FIELDS)], inclusive=False)
    for column in D1_MACRO_COT_FIELDS:
        out[column] = joined[column].to_numpy(float)
    for instrument in D1_MACRO_PANEL_INSTRUMENTS:
        features = components["panel"][instrument]
        closes = pd.Series(pd.DatetimeIndex(features.index) + TRADING_SESSION_DURATION)
        joined = align_asof(times, closes, features, inclusive=True)
        for column in features.columns:
            out[column] = joined[column].to_numpy(float)
    drivers = components["drivers"]
    joined = align_asof(times, drivers["available_at"], drivers[list(D1_MACRO_DRIVER_FIELDS)], inclusive=False)
    for column in D1_MACRO_DRIVER_FIELDS:
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
        "native_test_construction": spec.get("native_test_construction"),
        "rows": int(len(frame)),
        "complete_rows": int(complete.sum()),
        "first_complete_decision_time": first_complete,
        "decision_clock_sha256": hashlib.sha256(
            pd.DatetimeIndex(frame["decision_time"]).as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest(),
        "values_sha256": hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest(),
    }
    return {**value, "binding_sha256": _sha(value)}
