"""One-owner UTC trading-session clock and VWAP/resample parity contract."""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from gx1.execution.oanda_client import OandaClient
from gx1.execution import v12_ctx_augment_live as live_context
from gx1.features import htf_features as htf
from gx1.scripts import augment_forward_outcome_v2 as outcome_context
from gx1.time import session_detector as session


def test_trading_session_id_changes_only_at_named_utc_boundary() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2026-01-01T21:59:00Z",
            "2026-01-01T22:00:00Z",
            "2026-01-01T23:59:00Z",
            "2026-01-02T00:00:00Z",
        ]
    )
    observed = session.trading_session_id_vectorized(
        timestamps,
        context="TEST_TRADING_SESSION",
    )
    assert observed[1] == observed[0] + 1
    assert observed[1] == observed[2] == observed[3]
    assert session.get_session(timestamps[0]) == "US"
    assert session.get_session(timestamps[1]) == "ASIA"
    assert session.trading_session_label(
        timestamps[3],
        context="TEST_TRADING_SESSION_LABEL",
    ) == pd.Timestamp("2026-01-01T22:00:00Z")


@pytest.mark.parametrize(
    "date",
    [
        "2026-03-08",  # US DST starts; the source/owner clock remains UTC.
        "2026-11-01",  # US DST ends; the source/owner clock remains UTC.
    ],
)
def test_trading_session_boundary_does_not_move_on_dst_dates(date: str) -> None:
    timestamps = pd.DatetimeIndex(
        [f"{date}T21:59:00Z", f"{date}T22:00:00Z"]
    )
    observed = session.trading_session_id_vectorized(
        timestamps,
        context="TEST_TRADING_SESSION_DST",
    )
    assert observed[1] == observed[0] + 1


def test_weekend_source_absence_advances_id_without_synthetic_rows() -> None:
    observed_rows = pd.DatetimeIndex(
        ["2026-07-24T21:55:00Z", "2026-07-26T23:00:00Z"]
    )
    observed = session.trading_session_id_vectorized(
        observed_rows,
        context="TEST_TRADING_SESSION_WEEKEND",
    )
    assert len(observed) == len(observed_rows) == 2
    # Friday 21:55 still belongs to the session opened Thursday 22:00; the
    # Sunday row belongs to the session opened Sunday 22:00.
    assert observed[1] - observed[0] == 3


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (pd.DatetimeIndex(["2026-01-01T22:00:00"]), "CLOCK_IMPLICIT"),
        (
            pd.DatetimeIndex(["2026-01-01T22:00:00"], tz="Europe/Oslo"),
            "CLOCK_NOT_UTC",
        ),
        (
            pd.DatetimeIndex(
                ["2026-01-01T22:01:00Z", "2026-01-01T22:00:00Z"]
            ),
            "TIMESTAMP_ORDER_INVALID",
        ),
    ],
)
def test_trading_session_id_fails_closed_without_exact_utc_order(
    timestamps: pd.DatetimeIndex,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        session.trading_session_id_vectorized(
            timestamps,
            context="TEST_TRADING_SESSION_STRICT",
        )


def test_m1_m5_closed_bar_availability_uses_the_same_boundary() -> None:
    m1_available = session.m1_decision_availability(
        pd.DatetimeIndex(["2026-01-01T21:59:00Z"])
    )
    m5_available = session.m5_decision_availability(
        pd.DatetimeIndex(["2026-01-01T21:55:00Z"])
    )
    assert m1_available[0] == m5_available[0] == pd.Timestamp(
        "2026-01-01T22:00:00Z"
    )
    assert session.get_session(m1_available[0]) == "ASIA"
    assert session.get_session(m5_available[0]) == "ASIA"


def test_all_mtf_grids_derive_phase_from_one_session_boundary() -> None:
    assert htf.MULTI_TF_RESAMPLE_ORIGIN_OFFSET == {
        "M5": pd.Timedelta(0),
        "M15": pd.Timedelta(0),
        "H1": pd.Timedelta(0),
        "H4": pd.Timedelta(hours=2),
        "D1": pd.Timedelta(hours=22),
    }
    assert htf.multi_tf_bar_label(
        pd.Timestamp("2026-01-02T00:30:00Z"),
        "H4",
    ) == pd.Timestamp("2026-01-01T22:00:00Z")
    assert htf.multi_tf_last_closed_label(
        pd.Timestamp("2026-01-02T21:55:00Z"),
        "H4",
    ) == pd.Timestamp("2026-01-02T18:00:00Z")
    assert htf.multi_tf_last_closed_label(
        pd.Timestamp("2026-01-02T21:55:00Z"),
        "D1",
    ) == pd.Timestamp("2026-01-01T22:00:00Z")






def test_clock_source_guard_forbids_local_midnight_vwap_owner() -> None:
    htf_source = inspect.getsource(htf)
    oanda_source = inspect.getsource(OandaClient.get_candles)
    assert "SESSION_BOUNDARIES as _SESSION_BOUNDARIES" not in htf_source
    assert '"alignmentTimezone": "UTC"' in oanda_source
    assert "dailyAlignment" not in oanda_source

    forbidden: list[tuple[str, str, int]] = []
    for module in (htf, outcome_context, live_context):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue
            if node.func.attr == "normalize":
                forbidden.append((module.__name__, "normalize", node.lineno))
            if (
                node.func.attr in {"resample", "floor"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {"D", "1D"}
            ):
                forbidden.append(
                    (module.__name__, node.func.attr, node.lineno)
                )
    assert forbidden == []


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_session_ids_are_independent_of_explicit_utc_storage_unit(unit):
    stamps = pd.DatetimeIndex(["2026-01-01T21:59:00Z", "2026-01-01T22:00:00Z"])
    expected = session.trading_session_id_vectorized(stamps.as_unit("ns"), context="TEST_NS")
    observed = session.trading_session_id_vectorized(stamps.as_unit(unit), context="TEST_UNIT")
    assert observed.tolist() == expected.tolist()
    assert observed[1] == observed[0] + 1
    assert session.trading_session_label(stamps.as_unit(unit), context="TEST_LABEL").tolist() == [
        pd.Timestamp("2025-12-31T22:00:00Z"), pd.Timestamp("2026-01-01T22:00:00Z")]


@pytest.mark.parametrize("minutes", [1, 5])
def test_daily_pivots_roll_at_local_close_availability(monkeypatch, minutes):
    from types import SimpleNamespace
    import numpy as np
    duration = pd.Timedelta(minutes=minutes)
    boundary = pd.Timestamp("2026-01-02T22:00:00Z")
    older = {"R1": 103., "R2": 104., "S1": 97., "S2": 96.}
    latest = {"R1": 102., "R2": 103., "S1": 98., "S2": 97.}
    monkeypatch.setattr(outcome_context, "_closed_m5_index", lambda *_: 0)
    monkeypatch.setattr(outcome_context, "_per_tf_all", lambda *_: {
        name: 100. for name in outcome_context.PER_TF_FEATURE_NAMES})
    monkeypatch.setattr(outcome_context, "_liquidity_zones", lambda *_: {
        f"dist_to_{tf}_{side}_atr": 0. for tf in ("m5", "m15", "h1", "h4", "d1")
        for side in ("hi", "lo")})
    for bars_before, expected in ((2, older), (1, latest)):
        opening = boundary - bars_before*duration
        context = SimpleNamespace(
            decision_ts_ns=np.array([opening.value]), decision_close=np.array([100.]),
            m5_close=np.array([100.]), decision_bar_duration_ns=duration.value,
            daily_pivot_by_date={boundary-pd.Timedelta(days=2): older,
                                 boundary-pd.Timedelta(days=1): latest})
        result = outcome_context.augment_candidate(context, opening, include_portfolio=False)
        for name, value in expected.items():
            assert result[f"dist_to_{name}_atr"] == 100.-value
