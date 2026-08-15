"""One-owner UTC trading-session clock and VWAP/resample parity contract."""

from __future__ import annotations

import ast
import inspect

import numpy as np
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


def test_h4_session_vwap_resets_on_an_exact_bar_boundary() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-01T18:00:00Z",
            "2026-01-01T22:00:00Z",
            "2026-01-02T02:00:00Z",
        ]
    )
    close = pd.Series([10.0, 20.0, 30.0], index=index)
    volume = pd.Series(np.ones(len(index)), index=index)
    observed = htf._session_vwap(
        close,
        volume,
        bar_duration=pd.Timedelta(hours=4),
    )
    np.testing.assert_array_equal(observed.to_numpy(), [10.0, 20.0, 25.0])


@pytest.mark.parametrize("bar_duration", [pd.Timedelta(minutes=1), pd.Timedelta(minutes=5)])
def test_session_vwap_is_bit_exact_for_prefix_and_chunks(
    bar_duration: pd.Timedelta,
) -> None:
    index = pd.date_range(
        "2026-01-01T21:50:00Z",
        periods=20,
        freq=bar_duration,
    )
    close = pd.Series(np.linspace(2000.0, 2019.0, len(index)), index=index)
    volume = pd.Series(np.arange(1, len(index) + 1, dtype=np.float64), index=index)
    full = htf._session_vwap(
        close,
        volume,
        bar_duration=bar_duration,
    )
    split = 7 if bar_duration == pd.Timedelta(minutes=1) else 2
    prefix, state = htf._session_vwap(
        close.iloc[:split],
        volume.iloc[:split],
        bar_duration=bar_duration,
        return_state=True,
    )
    suffix, _ = htf._session_vwap(
        close.iloc[split:],
        volume.iloc[split:],
        bar_duration=bar_duration,
        state=state,
        return_state=True,
    )
    chunked = pd.concat([prefix, suffix])
    np.testing.assert_array_equal(prefix.to_numpy(), full.iloc[:split].to_numpy())
    np.testing.assert_array_equal(chunked.to_numpy(), full.to_numpy())

    stale = dict(state)
    stale["clock_schema_version"] = "retired_midnight_clock"
    with pytest.raises(RuntimeError, match="STATE_CONTRACT_MISMATCH"):
        htf._session_vwap(
            close.iloc[split:],
            volume.iloc[split:],
            bar_duration=bar_duration,
            state=stale,
        )


def test_clock_source_guard_forbids_local_midnight_vwap_owner() -> None:
    htf_source = inspect.getsource(htf)
    oanda_source = inspect.getsource(OandaClient.get_candles)
    assert "SESSION_BOUNDARIES as _SESSION_BOUNDARIES" not in htf_source
    assert "trading_session_id_vectorized(" in inspect.getsource(
        htf._session_vwap
    )
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
