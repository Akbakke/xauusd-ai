import pandas as pd

from gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 import SourceTape


def _source_tape_for_prices() -> SourceTape:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="5min")
    frame = pd.DataFrame(
        {
            "time": times,
            "bid_open": [100.0, 100.05, 100.1, 100.1],
            "ask_open": [100.0, 100.05, 100.1, 100.1],
            "bid_close": [100.0, 100.2, 100.1, 100.1],
            "ask_close": [100.0, 100.2, 100.1, 100.1],
            "bid_high": [100.0, 100.3, 100.25, 100.15],
            "bid_low": [100.0, 99.8, 100.0, 100.05],
            "ask_high": [100.0, 100.2, 100.0, 100.05],
            "ask_low": [100.0, 99.7, 99.85, 99.95],
        }
    )
    return SourceTape(
        times=frame["time"].to_numpy(),
        index=pd.Index(frame["time"]),
        bid_open=frame["bid_open"].to_numpy(),
        ask_open=frame["ask_open"].to_numpy(),
        bid_close=frame["bid_close"].to_numpy(),
        ask_close=frame["ask_close"].to_numpy(),
        bid_high=frame["bid_high"].to_numpy(),
        bid_low=frame["bid_low"].to_numpy(),
        ask_high=frame["ask_high"].to_numpy(),
        ask_low=frame["ask_low"].to_numpy(),
    )


def test_mfe_protect_long_uses_prior_bar_activation() -> None:
    tape = _source_tape_for_prices()

    trade = tape.simulate_trade(
        start_idx=0,
        horizon_bars=3,
        side=0,
        exit_mode="stop_tp_mfe_protect",
        take_profit_bps=90.0,
        stop_loss_bps=45.0,
        same_bar_policy="stop_first",
        mfe_protect_activation_bps=20.0,
        mfe_protect_breakeven_offset_bps=0.0,
        mfe_protect_trailing_capture_ratio=0.0,
        mfe_protect_trailing_floor_bps=0.0,
    )

    assert trade is not None
    assert trade["exit_reason"] == "mfe_protect_stop"
    assert trade["held_bars"] == 2
    assert trade["gross_pnl_bps"] == 0.0
    assert trade["mfe_protect_activated"] is True
    assert trade["mfe_protect_activation_bar"] == 1
    assert trade["mfe_protect_peak_mfe_bps_at_exit"] >= 20.0


def test_replay_entry_uses_fill_bar_open_not_close() -> None:
    tape = _source_tape_for_prices()

    trade = tape.simulate_trade(
        start_idx=1,
        horizon_bars=2,
        side=0,
        exit_mode="horizon",
        take_profit_bps=90.0,
        stop_loss_bps=45.0,
        same_bar_policy="stop_first",
    )

    assert trade is not None
    assert trade["entry_price"] == 100.05


def test_mfe_protect_short_uses_prior_bar_activation() -> None:
    tape = _source_tape_for_prices()

    trade = tape.simulate_trade(
        start_idx=0,
        horizon_bars=3,
        side=1,
        exit_mode="stop_tp_mfe_protect",
        take_profit_bps=90.0,
        stop_loss_bps=45.0,
        same_bar_policy="stop_first",
        mfe_protect_activation_bps=20.0,
        mfe_protect_breakeven_offset_bps=0.0,
        mfe_protect_trailing_capture_ratio=0.0,
        mfe_protect_trailing_floor_bps=0.0,
    )

    assert trade is not None
    assert trade["exit_reason"] == "mfe_protect_stop"
    assert trade["held_bars"] == 2
    assert trade["gross_pnl_bps"] == 0.0
    assert trade["mfe_protect_activated"] is True
    assert trade["mfe_protect_activation_bar"] == 1
