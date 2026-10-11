from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features import d1_macro_context_v1 as ctx
from gx1.scripts import research_ta_campaign_v1 as ta


def test_field_order_is_cot_panel_drivers_and_contract_hash_is_stable():
    assert ctx.D1_MACRO_CONTEXT_FIELDS[:4] == tuple(ta.SWING_COT_FIELDS)
    assert ctx.D1_MACRO_CONTEXT_FIELDS[-13:] == tuple(ta.SWING_DRIVER_FIELDS)
    assert ctx.D1_MACRO_CONTEXT_DIM == 4 + 44 + 13 == 61
    assert len(set(ctx.D1_MACRO_CONTEXT_FIELDS)) == 61
    a, b = ctx.d1_macro_context_contract(), ctx.d1_macro_context_contract()
    assert a == b and a["specialist_owner"] == "session_regime_encoder" and len(a["contract_sha256"]) == 64
    assert a["source_spec"]["path"].endswith("D1_MACRO_CONTEXT_SOURCES_20261011_001.json")


def _components():
    def utc(s):
        return pd.Timestamp(s, tz="UTC")
    cot = pd.DataFrame({f: [1.0, 2.0] for f in ta.SWING_COT_FIELDS})
    cot["available_at"] = pd.Series([utc("2024-01-09"), utc("2024-01-16")])
    panel = {}
    for inst in ctx.D1_MACRO_PANEL_INSTRUMENTS:
        labels = pd.DatetimeIndex([utc("2024-01-08 22:00"), utc("2024-01-09 22:00")])  # session opens
        frame = pd.DataFrame(index=labels)
        for d in ta.SWING_PANEL_RETURN_DAYS:
            frame[f"{inst}_logret_{d}D"] = [10.0, 20.0]
        frame[f"{inst}_rv_{ta.SWING_PANEL_RV_DAYS}D"] = [0.1, 0.2]
        panel[inst] = frame
    drivers = pd.DataFrame({f: [100.0, 200.0] for f in ta.SWING_DRIVER_FIELDS})
    drivers["available_at"] = pd.Series([utc("2024-01-10 05:00"), utc("2024-01-11 05:00")])
    return {"read_end": utc("2026-07-01"), "cot": cot, "panel": panel, "drivers": drivers}


def test_join_respects_each_group_availability_rule():
    comp = _components()
    times = pd.DatetimeIndex(["2024-01-09 00:00", "2024-01-09 22:00", "2024-01-10 05:00", "2024-01-10 22:00", "2024-01-16 00:00", "2024-01-16 00:05"], tz="UTC")
    out = ctx.build_d1_macro_context(times, comp)
    assert list(out.columns) == ["decision_time", *ctx.D1_MACRO_CONTEXT_FIELDS]
    # COT: strictly after available_at: not at 2024-01-09 00:00, yes at 22:00; the second report (available 2024-01-16 00:00)
    # is not usable at exactly 00:00 and is usable at 00:05
    assert np.isnan(out["cot_change4reports"].iloc[0]) and out["cot_change4reports"].iloc[1] == 1.0
    assert out["cot_change4reports"].iloc[4] == 1.0 and out["cot_change4reports"].iloc[5] == 2.0
    # Panel: inclusive at the bar close (label + one session): bar opened 2024-01-08 22:00 closes 2024-01-09 22:00
    assert np.isnan(out["XAU_USD_logret_1D"].iloc[0]) and out["XAU_USD_logret_1D"].iloc[1] == 10.0 and out["XAU_USD_logret_1D"].iloc[3] == 20.0
    # Drivers: exclusive at available_at 05:00 UTC
    assert np.isnan(out["vix_log_level"].iloc[2]) and out["vix_log_level"].iloc[3] == 100.0 and out["vix_log_level"].iloc[4] == 200.0


def test_clock_and_read_end_fail_closed():
    comp = _components()
    with pytest.raises(RuntimeError, match="DECISION_CLOCK"):
        ctx.build_d1_macro_context(pd.DatetimeIndex(["2024-01-09", "2024-01-08"], tz="UTC"), comp)
    with pytest.raises(RuntimeError, match="DECISION_CLOCK"):
        ctx.build_d1_macro_context(pd.DatetimeIndex(["2024-01-09"]), comp)
    with pytest.raises(RuntimeError, match="BEYOND_READ_END"):
        ctx.build_d1_macro_context(pd.DatetimeIndex(["2026-07-02"], tz="UTC"), comp)


def test_spec_requires_exact_panel_and_source_scope():
    good = {"test_seal_utc": "2026-07-01T00:00:00Z", "cot": {"contract_market_code": "088691"},
            "oanda_panel": {"instruments": list(ctx.D1_MACRO_PANEL_INSTRUMENTS)}, "drivers": {"gld_item": "X"}}
    assert ctx.require_d1_macro_context_spec(good)["cot"]["contract_market_code"] == "088691"
    with pytest.raises(RuntimeError, match="PANEL_INSTRUMENTS"):
        ctx.require_d1_macro_context_spec({**good, "oanda_panel": {"instruments": ["XAU_USD"]}})
    with pytest.raises(RuntimeError, match="SOURCE_SCOPE"):
        ctx.require_d1_macro_context_spec({**good, "cot": {"contract_market_code": "084691"}})


def test_binding_hashes_clock_and_values():
    comp = _components()
    times = pd.DatetimeIndex(["2024-01-10 22:00", "2024-01-16 00:05"], tz="UTC")
    out = ctx.build_d1_macro_context(times, comp)
    spec = {"test_seal_utc": "2026-07-01T00:00:00Z", "cot": {"a": 1}, "oanda_panel": {"b": 2}, "drivers": {"c": 3}}
    binding = ctx.d1_macro_context_binding(spec, out)
    assert binding["rows"] == 2 and binding["complete_rows"] == 2 and binding["first_complete_decision_time"] == str(times[0])
    assert binding["contract"] == ctx.d1_macro_context_contract() and len(binding["values_sha256"]) == 64
    assert ctx.d1_macro_context_binding(spec, out)["binding_sha256"] == binding["binding_sha256"]
