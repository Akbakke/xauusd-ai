from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    build_entry_m1_fill_surface,
    causal_m1_outcomes_at_horizon,
    causal_m1_terminal_outcomes_at_horizon,
    causal_m1_target_contract,
    prepare_causal_m1_quote_source,
)
from gx1.contracts.entry_execution_causality_v1 import (
    legacy_same_close_target_contract_failures,
)


def _m1_frame(*, periods: int = 24) -> pd.DataFrame:
    times = pd.date_range("2026-01-05T00:00:00Z", periods=periods, freq="min")
    bid_open = 100.0 + np.arange(periods, dtype=np.float64)
    bid_high = bid_open + 0.4
    bid_low = bid_open - 0.2
    ask_open = bid_open + 0.1
    return pd.DataFrame(
        {
            "time": times,
            "bid_open": bid_open,
            "bid_high": bid_high,
            "bid_low": bid_low,
            "ask_open": ask_open,
            "ask_high": bid_high + 0.1,
            "ask_low": bid_low + 0.1,
        }
    )


def test_causal_target_contract_has_no_same_close_price() -> None:
    assert legacy_same_close_target_contract_failures(causal_m1_target_contract()) == []


def test_m5_decision_binds_to_next_m1_open_and_path_excludes_exit_bar() -> None:
    m1 = _m1_frame()
    m5_times = pd.DatetimeIndex(
        pd.to_datetime(
            ["2026-01-05T00:00:00Z", "2026-01-05T00:05:00Z"], utc=True
        )
    )
    surface = build_entry_m1_fill_surface(
        m5_decision_times=m5_times,
        closed_m1=m1,
    )
    outcomes = causal_m1_outcomes_at_horizon(
        fill_surface=surface,
        closed_m1=m1,
        horizon_m5_bars=1,
    )

    first = outcomes.iloc[0]
    assert surface["entry_decision_at"].iloc[0] == pd.Timestamp(
        "2026-01-05T00:05:00Z"
    )
    assert surface["entry_m1_row"].iloc[0] == 5
    assert surface["entry_ask"].iloc[0] == pytest.approx(105.1)
    assert first["exit_decision_at"] == pd.Timestamp("2026-01-05T00:10:00Z")
    assert first["exit_bid"] == pytest.approx(110.0)
    assert first["long_executable_pnl_bps"] == pytest.approx(
        (110.0 / 105.1 - 1.0) * 1e4
    )
    # The exit's M1 bar begins only after the path ends. Its 110.4 high cannot
    # become an earlier favourable excursion.
    assert first["long_mfe_bps"] == pytest.approx(
        (109.4 / 105.1 - 1.0) * 1e4
    )
    assert first["long_mae_bps"] == pytest.approx(
        (1.0 - 104.8 / 105.1) * 1e4
    )
    assert bool(first["outcome_valid"]) is True


def test_missing_m1_path_never_substitutes_an_exit_quote() -> None:
    m1 = _m1_frame().drop(index=8).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    outcomes = causal_m1_outcomes_at_horizon(
        fill_surface=surface,
        closed_m1=m1,
        horizon_m5_bars=1,
    )

    row = outcomes.iloc[0]
    assert bool(row["outcome_valid"]) is False
    assert pd.isna(row["exit_decision_at"])
    assert np.isnan(row["exit_bid"])
    assert np.isnan(row["long_executable_pnl_bps"])
    assert np.isnan(row["long_mfe_bps"])


def test_missing_fill_minute_is_marked_unbound_not_replaced_by_m5_close() -> None:
    m1 = _m1_frame().drop(index=5).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(
                ["2026-01-05T00:00:00Z", "2026-01-05T00:05:00Z"], utc=True
            )
        ),
        closed_m1=m1,
    )

    assert surface["entry_fill_bound"].tolist() == [False, True]
    assert surface["entry_m1_row"].tolist()[0] == -1
    assert np.isnan(surface["entry_bid"].iloc[0])


def test_terminal_outcome_has_same_clock_and_pnl_without_path_windows() -> None:
    m1 = _m1_frame()
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    out = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=m1, horizon_m5_bars=1
    )
    assert bool(out.loc[0, "outcome_valid"])
    assert out.loc[0, "exit_decision_at"] == pd.Timestamp("2026-01-05T00:10:00Z")
    assert out.loc[0, "long_executable_pnl_bps"] == pytest.approx(
        (110.0 / 105.1 - 1.0) * 1e4
    )


def test_prepared_m1_source_reuses_the_exact_validated_quotes() -> None:
    m1 = _m1_frame()
    prepared = prepare_causal_m1_quote_source(m1)
    assert all(not values.flags.writeable for values in prepared.values.values())
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=prepared,
    )
    out = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared, horizon_m5_bars=1
    )
    assert bool(out.loc[0, "outcome_valid"])


def test_unbound_surface_row_cannot_hide_an_arbitrary_m1_pointer() -> None:
    m1 = _m1_frame().drop(index=5).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    surface.loc[0, "entry_m1_row"] = -2

    with pytest.raises(
        RuntimeError, match="ENTRY_CAUSAL_M1_FILL_SURFACE_QUOTE_INVALID"
    ):
        causal_m1_outcomes_at_horizon(
            fill_surface=surface,
            closed_m1=m1,
            horizon_m5_bars=1,
        )
