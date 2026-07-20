from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_pipeline as pipeline_module
from gx1.execution.v12_pipeline import (
    EntryDecisionUnavailable,
    ExitDecisionUnavailable,
    V12Pipeline,
    _exact_closed_m5_row,
    _validated_v3_output,
)


def _collector_rows(*times: str) -> pd.DataFrame:
    rows = []
    for offset, time in enumerate(times):
        bid_close = 2400.0 + offset
        ask_close = bid_close + 0.2
        rows.append(
            {
                "time": pd.Timestamp(time),
                "bid_high": bid_close + 0.5,
                "bid_low": bid_close - 0.5,
                "ask_high": ask_close + 0.5,
                "ask_low": ask_close - 0.5,
                "bid_close": bid_close,
                "ask_close": ask_close,
            }
        )
    return pd.DataFrame(rows)


def _pipeline(loader: object | None = None, **kwargs: object) -> V12Pipeline:
    return V12Pipeline(
        prebuilt_loader=loader or SimpleNamespace(),
        exit_xgb=object(),
        **kwargs,
    )


class _CanonicalLoader:
    def __init__(self, cutoff: pd.Timestamp, window: pd.DataFrame) -> None:
        self.cutoff_ts = cutoff
        self.window = window
        self.requested: list[tuple[pd.Timestamp, int]] = []

    def refresh_if_changed(self) -> bool:
        return False

    def get_window(self, end_ts: pd.Timestamp, *, n_bars: int) -> pd.DataFrame:
        self.requested.append((end_ts, n_bars))
        return self.window.copy()


def _canonical_window(end: str) -> pd.DataFrame:
    index = pd.date_range(
        end=pd.Timestamp(end),
        periods=pipeline_module.ENTRY_SEQ_LEN,
        freq="5min",
    )
    return pd.DataFrame({"atr_bps": np.full(len(index), 12.0)}, index=index)


def test_entry_canonical_freshness_is_fixed_and_ignores_exit_staleness_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T11:55:00Z"),
        _canonical_window("2026-07-16T11:55:00Z"),
    )
    pipe = _pipeline(loader)
    monkeypatch.setenv("GX1_MAX_PREBUILT_STALENESS_MIN", "999999")

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:00Z"))

    assert raised.value.reason == "entry_canonical_stale"
    assert raised.value.evidence["canonical_cutoff_age_sec"] == 660.0
    assert raised.value.evidence["canonical_cutoff_age_cap_sec"] == 390.0
    assert loader.requested == []

    # The same variable remains an Exit-only operational control; splitting the
    # paths must not silently change the separately admitted Exit behavior.
    assert pipe._refresh_exit_canonical(pd.Timestamp("2026-07-16T12:06:00Z")) is True
    assert loader.requested == [
        (pd.Timestamp("2026-07-16T11:55:00Z"), pipeline_module.ENTRY_SEQ_LEN)
    ]


def test_entry_canonical_requires_the_exact_latest_closed_m5() -> None:
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T11:59:00Z"),
        _canonical_window("2026-07-16T11:55:00Z"),
    )
    pipe = _pipeline(loader)

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:05:00Z"))

    assert raised.value.reason == "entry_latest_closed_m5_unavailable"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"
    assert loader.requested == []


def test_entry_canonical_accepts_only_an_exact_96_bar_fresh_window() -> None:
    window = _canonical_window("2026-07-16T12:00:00Z")
    loader = _CanonicalLoader(pd.Timestamp("2026-07-16T12:00:00Z"), window)
    pipe = _pipeline(loader)

    pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:00Z"))

    assert loader.requested == [
        (pd.Timestamp("2026-07-16T12:00:00Z"), pipeline_module.ENTRY_SEQ_LEN)
    ]
    assert pipe._last_augmented is not None
    assert pipe._last_augmented.index[-1] == pd.Timestamp("2026-07-16T12:00:00Z")


def test_entry_canonical_age_does_not_floor_away_subminute_staleness() -> None:
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T12:00:00Z"),
        _canonical_window("2026-07-16T12:00:00Z"),
    )
    pipe = _pipeline(loader)

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:31Z"))

    assert raised.value.reason == "entry_canonical_stale"
    assert raised.value.evidence["canonical_cutoff_age_sec"] == 391.0
    assert loader.requested == []


def test_negative_retired_latency_env_cannot_enable_entry_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EntryMustNotRun:
        def predict_live_bar(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("stale Entry state must never reach model inference")

    pipe = _pipeline(smart_entry=_EntryMustNotRun())
    pipe._last_augmented = _canonical_window("2026-07-16T12:00:00Z")
    monkeypatch.setattr(pipe, "_refresh_entry_canonical", lambda _now: None)
    monkeypatch.setenv("GX1_MAX_ENTRY_DECISION_LATENCY_SEC", "-1")

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe.make_entry_decision(
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.0,
            ask=2400.2,
        )

    assert raised.value.reason == "entry_signal_stale"
    assert raised.value.evidence["entry_signal_latency_sec"] == 120.0
    assert raised.value.evidence["entry_signal_latency_cap_sec"] == 90.0


def test_closed_m1_does_not_substitute_an_older_cached_or_latest_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-07-16T12:10:00Z")
    expected_path = tmp_path / "xauusd_m1_20260716.parquet"
    expected_path.touch()
    pipe = _pipeline()
    pipe._last_m1_atr_minute = pd.Timestamp("2026-07-16T12:08:00Z")
    pipe._last_m1_bar = {"time": pipe._last_m1_atr_minute, "mid_close": 111.0}

    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _collector_rows("2026-07-16T12:08:00Z"),
    )

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe._refresh_m1_bar(now)

    assert raised.value.reason == "closed_m1_exact_bar_missing"
    assert raised.value.evidence["expected_m1"] == "2026-07-16 12:09:00+00:00"
    assert raised.value.evidence["latest_observed_m1"] == "2026-07-16 12:08:00+00:00"


def test_closed_m1_selects_the_unique_exact_bar_not_a_forming_later_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-07-16T12:10:00Z")
    expected_path = tmp_path / "xauusd_m1_20260716.parquet"
    expected_path.touch()
    frame = _collector_rows(
        "2026-07-16T12:09:00Z",
        "2026-07-16T12:10:00Z",
    )
    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame.copy(),
    )

    bar = _pipeline()._refresh_m1_bar(now)

    assert bar["time"] == pd.Timestamp("2026-07-16T12:09:00Z")
    assert bar["bid_close"] == 2400.0
    assert bar["mid_close"] == pytest.approx(2400.1)
    assert bar["atr_bps"] > 0.0


def test_closed_m1_midnight_uses_the_expected_bars_calendar_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = tmp_path / "xauusd_m1_20260715.parquet"
    expected_path.touch()
    observed_paths: list[Path] = []

    def fake_read(path: Path, **_kwargs: object) -> pd.DataFrame:
        observed_paths.append(Path(path))
        return _collector_rows("2026-07-15T23:59:00Z")

    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(pipeline_module.pd, "read_parquet", fake_read)

    bar = _pipeline()._refresh_m1_bar(pd.Timestamp("2026-07-16T00:00:00Z"))

    assert observed_paths == [expected_path]
    assert bar["time"] == pd.Timestamp("2026-07-15T23:59:00Z")


def test_exact_closed_m5_rejects_latest_row_substitution() -> None:
    augmented = pd.DataFrame(
        {"atr_bps": [12.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T11:55:00Z")]),
    )

    with pytest.raises(ExitDecisionUnavailable) as raised:
        _exact_closed_m5_row(augmented, pd.Timestamp("2026-07-16T12:07:00Z"))

    assert raised.value.reason == "canonical_exact_m5_missing"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"


@pytest.mark.parametrize(
    ("output", "q_head_required", "reason"),
    [
        (None, False, "v3_output_invalid"),
        ({}, False, "v3_output_missing"),
        (
            {
                "v3_v8_should_exit_prob": np.nan,
                "v3_v8_profit_protect_prob": 0.2,
                "v3_v8_family_argmax": 1,
                "v3_v8_family_logit_max": 2.0,
            },
            False,
            "v3_output_non_finite",
        ),
        (
            {
                "v3_v8_should_exit_prob": 0.3,
                "v3_v8_profit_protect_prob": 0.2,
                "v3_v8_family_argmax": 1,
                "v3_v8_family_logit_max": 2.0,
            },
            True,
            "v3_output_missing",
        ),
    ],
)
def test_v3_output_contract_fails_closed(
    output: object,
    q_head_required: bool,
    reason: str,
) -> None:
    with pytest.raises(ExitDecisionUnavailable) as raised:
        _validated_v3_output(output, q_head_required=q_head_required)

    assert raised.value.reason == reason


class _FakeTrade:
    def __init__(self, *, quote_pnl_bps: float = 0.0) -> None:
        self.side = "long"
        self.trade_id = "T-1"
        self.bars_in_trade = 0
        self.current_bid = 2400.0
        self.current_ask = 2400.2
        self.current_pnl_bps = 0.0
        self.cum_mfe_bps = 0.0
        self.cum_mae_bps = 0.0
        self.last_atr_bps = 0.0
        self._quote_pnl_bps = quote_pnl_bps
        self.updated_bar: dict[str, float] | None = None
        self.v3_updates: list[dict[str, object]] = []

    def _pnl_bps(self, _bid: float, _ask: float) -> float:
        return self._quote_pnl_bps

    def update_bar(self, **values: float) -> None:
        self.updated_bar = dict(values)
        self.bars_in_trade += 1
        self.current_bid = float(values["bid"])
        self.current_ask = float(values["ask"])

    def build_v3_overlay(self) -> dict[str, np.ndarray]:
        return {}

    def update_v3(self, output: dict[str, object]) -> None:
        self.v3_updates.append(output)


class _FailingV3:
    _enable_multi_tf = False
    _enable_q_head = False

    def predict(self, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("broken-v3")


class _ExitMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def decide_for_trade(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("Exit-IQL must not receive failed V3 state")


def _exact_m1_bar() -> dict[str, object]:
    return {
        "time": pd.Timestamp("2026-07-16T12:06:00Z"),
        "bid_high": 2401.0,
        "bid_low": 2399.0,
        "ask_high": 2401.2,
        "ask_low": 2399.2,
        "bid_close": 2400.0,
        "ask_close": 2400.2,
        "mid_close": 2400.1,
        "atr_bps": 9.16,
    }


def test_v3_failure_never_continues_into_exit_iql_zero_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_time = pd.Timestamp("2026-07-16T12:06:00Z")
    loader = SimpleNamespace(
        _base28=pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex([exact_time])),
        cutoff_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
    )
    exit_iql = _ExitMustNotRun()
    pipe = _pipeline(loader, v3=_FailingV3(), exit_iql=exit_iql)
    pipe._last_augmented = pd.DataFrame(
        {"atr_bps": [12.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T12:00:00Z")]),
    )
    monkeypatch.setattr(pipe, "_refresh_m1_bar", lambda _now: _exact_m1_bar())
    monkeypatch.setattr(pipe, "_refresh_exit_canonical", lambda _now: True)
    trade = _FakeTrade()

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe.make_exit_decision(
            trade,
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.3,
            ask=2400.5,
        )

    assert raised.value.reason == "v3_inference_failed"
    assert exit_iql.calls == 0
    assert trade.v3_updates == []
    assert trade.updated_bar is not None
    assert trade.updated_bar["bid"] == 2400.0
    assert trade.updated_bar["ask"] == 2400.2
    assert trade.updated_bar["m1_close"] == 2400.1


def test_missing_canonical_state_is_unavailable_not_synthetic_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = SimpleNamespace(cutoff_ts=pd.Timestamp("2026-07-16T11:55:00Z"))
    pipe = _pipeline(loader)
    trade = _FakeTrade()
    monkeypatch.setattr(pipe, "_refresh_m1_bar", lambda _now: _exact_m1_bar())
    monkeypatch.setattr(pipe, "_refresh_exit_canonical", lambda _now: False)

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe.make_exit_decision(
            trade,
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.3,
            ask=2400.5,
        )

    assert raised.value.reason == "canonical_data_unavailable"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"
    assert trade.bars_in_trade == 0
    assert trade.updated_bar is None


def test_fresh_quote_hard_stop_remains_available_before_model_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _pipeline()
    trade = _FakeTrade(quote_pnl_bps=-95.0)
    monkeypatch.setattr(pipeline_module, "_EXIT_HARD_STOP_BPS", 80.0)
    monkeypatch.setattr(
        pipe,
        "_refresh_m1_bar",
        lambda _now: (_ for _ in ()).throw(AssertionError("collector must not be read")),
    )

    decision = pipe.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:07:00Z"),
        bid=2300.0,
        ask=2300.2,
    )

    assert decision["action"] == "EXIT_NOW"
    assert decision["decision_source"] == "HARD_MAE_STOP"
    assert decision["decision_safety_scope"] == "fresh_quote_existing_position_close"
    assert trade.bars_in_trade == 0
    assert trade.current_pnl_bps == -95.0


def test_active_runtime_source_has_no_decision_state_substitution() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_source = (root / "gx1/execution/v12_pipeline.py").read_text(encoding="utf-8")
    runner_source = (root / "gx1/execution/v12_paper_runner.py").read_text(encoding="utf-8")
    exit_start = pipeline_source.index("    def make_exit_decision(")
    exit_end = pipeline_source.index("\n\n# Backwards-compat", exit_start)
    active_exit = pipeline_source[exit_start:exit_end]

    for forbidden in (
        "using zero fallback",
        "zero-fallback V3 state",
        "Use latest available bar as fallback",
        '"error": "no_canonical_data"',
        "current_m1_atr_bps_override=m1_atr_bps if",
        "v3_v8_out = None",
        "trade.update_bar(bid=bid",
    ):
        assert forbidden not in active_exit

    assert "m1_close = (bid + ask) / 2.0" not in runner_source
    assert ".read_parquet(_p" not in runner_source
    assert 'fill_price = float(order_result.get("fill_price") or 0.0)' not in runner_source
    assert 'float(t.get("currentUnits", 0) or 0)' not in runner_source
    assert 'get_open_trades().get("trades", [])' not in runner_source
    assert "fill_price - spread_abs" not in runner_source
    assert "fill_price + spread_abs" not in runner_source
    assert "except ExitDecisionUnavailable as exc:" in runner_source
    assert '"exit_decision": None' in runner_source
    assert "if exit_decision_unavailable:" in runner_source
    assert "FILLED_STATE_UNAVAILABLE_RECOVERY" in runner_source
    assert "EXIT_CLOSE_FAILED" in runner_source
    assert "EXIT_EXECUTION_UNRESOLVED" in runner_source
    assert "BROKER_RECONCILIATION_REQUIRED" in runner_source


def test_missing_trade_id_counter_order_is_retained_only_for_safe_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id=None, side="long", units=7)

    result = runner.attempt_close_trade(object(), trade)

    assert result == {"status": "filled"}
    assert calls == [("short", 7)]


def test_empty_trade_id_also_uses_counter_order_safe_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id="", side="short", units=3)

    result = runner.attempt_close_trade(object(), trade)

    assert result == {"status": "filled"}
    assert calls == [("long", 3)]


def test_filled_order_with_missing_price_is_explicitly_incomplete_not_zero() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-1",
                "orderID": "order-1",
                "units": "2",
                "tradeOpened": {"tradeID": "trade-1", "units": "2"},
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=2)

    assert result["status"] == "filled"
    assert result["trade_id"] == "trade-1"
    assert result["fill_price"] is None


def test_filled_order_units_must_exactly_match_requested_learned_units() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-2",
                "orderID": "order-2",
                "units": "1",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-2", "units": "1"},
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=2)

    assert result["status"] == "filled_units_mismatch"
    assert result["requested_signed_units"] == 2
    assert result["filled_signed_units"] == 1
    assert result["fill_units_exact"] is False


def test_mixed_netting_fill_is_never_accepted_as_new_trade_state() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-mixed",
                "orderID": "order-mixed",
                "units": "5",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-new", "units": "2"},
                "tradesClosed": [{"tradeID": "trade-old", "units": "-3"}],
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=5)

    assert result["status"] == "filled_structure_mismatch"
    assert result["fill_units_exact"] is True
    assert result["pure_trade_open"] is False


def _runtime_sizing_authority_for_broker_fact_tests():
    import json

    from gx1.contracts.entry_model_native_sizing_authority_v1 import (
        ValidatedLearnedSizingAuthority,
    )

    return ValidatedLearnedSizingAuthority(
        authority_json="{}",
        adoption_json="{}",
        calibration_json=json.dumps(
            {
                "instrument_constraints": {
                    "instrument": "XAU_USD",
                    "account_currency": "USD",
                    "quote_currency": "USD",
                    "unit_step": 1,
                    "minimum_order_units": 1,
                    "maximum_gross_xau_units": 1000,
                    "margin_rate": 0.05,
                }
            }
            ),
            proof_json="{}",
            joint_proof_json="{}",
            content_hash_key=(),
        file_stats=(),
    )


def _broker_fact_client(*, hedging_enabled: bool, transaction_ids: tuple[str, str, str]):
    account_tx, instrument_tx, exposure_tx = transaction_ids
    return SimpleNamespace(
        get_account_summary=lambda: {
            "account": {
                "currency": "USD",
                "hedgingEnabled": hedging_enabled,
                "NAV": "10000",
                "balance": "10000",
                "marginAvailable": "1000",
                "marginUsed": "0",
            },
            "lastTransactionID": account_tx,
        },
        get_account_instruments=lambda _instruments: {
            "instruments": [
                {
                    "name": "XAU_USD",
                    "tradeUnitsPrecision": 0,
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "100000",
                    "marginRate": "0.05",
                }
            ],
            "lastTransactionID": instrument_tx,
        },
        get_open_trades=lambda: {
            "trades": [],
            "lastTransactionID": exposure_tx,
        },
    )


def test_live_sizing_requires_one_coherent_hedging_broker_snapshot() -> None:
    from gx1.execution import v12_paper_runner as runner

    constraints = runner.learned_sizing_runtime_constraints(
        _broker_fact_client(
            hedging_enabled=True,
            transaction_ids=("9001", "9001", "9001"),
        ),
        bid=2400.0,
        ask=2400.2,
        validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
    )

    assert constraints["account_last_transaction_id"] == "9001"
    assert constraints["instrument_last_transaction_id"] == "9001"
    assert constraints["exposure_last_transaction_id"] == "9001"


@pytest.mark.parametrize(
    ("hedging_enabled", "transaction_ids", "match"),
    [
        (False, ("9001", "9001", "9001"), "hedgingEnabled=true"),
        (True, ("9001", "9002", "9001"), "different lastTransactionID"),
    ],
)
def test_live_sizing_rejects_netting_or_torn_broker_snapshot(
    hedging_enabled: bool,
    transaction_ids: tuple[str, str, str],
    match: str,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    with pytest.raises(RuntimeError, match=match):
        runner.learned_sizing_runtime_constraints(
            _broker_fact_client(
                hedging_enabled=hedging_enabled,
                transaction_ids=transaction_ids,
            ),
            bid=2400.0,
            ask=2400.2,
            validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
        )


@pytest.mark.parametrize(
    "quote",
    [
        {"bids": [{"price": "2400.0"}], "asks": [{"price": "2400.2"}]},
        {
            "time": "2026-07-16T12:00:00Z",
            "bids": [{"price": "2400.0"}],
            "asks": [{"price": "2399.9"}],
        },
    ],
)
def test_quote_missing_time_or_valid_bid_ask_contract_fails_closed(quote: dict) -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(get_pricing=lambda _instruments: {"prices": [quote]})

    with pytest.raises(ValueError):
        runner.get_current_spread_bps(
            client,
            now_utc=pd.Timestamp("2026-07-16T12:00:30Z").to_pydatetime(),
        )
