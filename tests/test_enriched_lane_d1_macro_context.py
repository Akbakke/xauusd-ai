from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features import d1_macro_context_v1 as ctx
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as lane
from tests.test_d1_macro_context_v1 import _components


def test_lane_attaches_the_ninth_family_in_chunks_as_float32(monkeypatch):
    comp = _components()
    monkeypatch.setattr(lane._d1ctx, "load_d1_macro_context_source_spec", lambda root: {"stub": True, "test_seal_utc": "2026-07-01T00:00:00Z"})
    monkeypatch.setattr(lane._d1ctx, "d1_macro_context_components", lambda spec: comp)
    monkeypatch.setattr(lane, "_D1_MACRO_ATTACH_CHUNK_ROWS", 4)
    # M5 bar opens; decision = open + 5 min. 2024-01-09 22:00 open -> decision 22:05 is after the
    # first panel close (22:00) and after the first COT availability (2024-01-09 00:00).
    opens = pd.date_range("2024-01-09T21:40:00Z", periods=10, freq="5min")
    enriched = pd.DataFrame({"close": np.arange(10, dtype=np.float32)}, index=opens)
    out = lane._attach_d1_macro_context(enriched, timeframe="M5")
    assert all(name in out.columns for name in ctx.D1_MACRO_CONTEXT_FIELDS)
    assert out["XAU_USD_logret_1D"].dtype == np.float32
    # the first panel bar (opened 2024-01-08 22:00) closes at 2024-01-09 22:00 and is usable inclusively at that
    # decision time, i.e. for the bar opened 21:55; earlier rows are NaN
    first_set = int(np.argmax(np.isfinite(out["XAU_USD_logret_1D"].to_numpy())))
    assert opens[first_set] == pd.Timestamp("2024-01-09T21:55:00Z") and out["XAU_USD_logret_1D"].iloc[first_set] == np.float32(10.0)
    assert np.isnan(out["XAU_USD_logret_1D"].iloc[first_set - 1])
    # chunking (4 rows) must not change values: recompute in one piece and compare bit for bit
    whole = ctx.build_d1_macro_context(pd.DatetimeIndex(opens + pd.Timedelta(minutes=5)), comp)
    for name in ctx.D1_MACRO_CONTEXT_FIELDS:
        a, b = out[name].to_numpy(np.float32), whole[name].to_numpy(np.float64).astype(np.float32)
        assert np.array_equal(a, b, equal_nan=True), name
