import argparse
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import run


def test_candidate_replay_trade_log_materializes_iql_transition_columns(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    val_times = pd.date_range("2025-12-31T23:00:00Z", periods=4, freq="5min")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="5min")
    pd.DataFrame({"time": val_times, "label_horizon_bars": [1, 1, 1, 1]}).to_parquet(
        dataset_dir / "tiny_val.parquet", index=False
    )
    pd.DataFrame({"time": test_times, "label_horizon_bars": [1, 1, 1, 1, 1, 1]}).to_parquet(
        dataset_dir / "tiny_test.parquet", index=False
    )
    predictions = pd.DataFrame(
        {
            "split": ["val"] * 4 + ["test"] * 6,
            "model": ["candidate"] * 10,
            "time": list(val_times) + list(test_times),
            "y_direction": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "trade_side": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "session": ["EU"] * 10,
            "vol_regime": ["1"] * 10,
            "edge_score": [0.90, 0.80, 0.20, 0.10, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45],
            "p_long": [0.80, 0.10, 0.55, 0.20, 0.84, 0.82, 0.12, 0.13, 0.70, 0.20],
            "p_short": [0.10, 0.80, 0.20, 0.55, 0.10, 0.12, 0.82, 0.81, 0.20, 0.70],
            "p_flat": [0.10, 0.10, 0.25, 0.25, 0.06, 0.06, 0.06, 0.06, 0.10, 0.10],
            "path_quality_pred": [1.0] * 10,
            "bad_path_prob": [0.2] * 10,
        }
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    all_times = list(val_times) + list(test_times) + [test_times[-1] + pd.Timedelta(minutes=5)]
    source = pd.DataFrame(
        {
            "time": all_times,
            "bid_close": [100.00, 100.20, 100.10, 100.30, 100.00, 100.30, 100.60, 100.40, 100.10, 100.50, 100.20],
            "ask_close": [100.02, 100.22, 100.12, 100.32, 100.02, 100.32, 100.62, 100.42, 100.12, 100.52, 100.22],
            "bid_high": [100.05, 100.25, 100.15, 100.35, 100.05, 100.35, 100.65, 100.45, 100.15, 100.55, 100.25],
            "bid_low": [99.95, 100.15, 100.05, 100.25, 99.95, 100.25, 100.55, 100.35, 100.05, 100.45, 100.15],
            "ask_high": [100.07, 100.27, 100.17, 100.37, 100.07, 100.37, 100.67, 100.47, 100.17, 100.57, 100.27],
            "ask_low": [99.97, 100.17, 100.07, 100.27, 99.97, 100.27, 100.57, 100.37, 100.07, 100.47, 100.17],
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    out_dir = tmp_path / "out"

    report = run(
        argparse.Namespace(
            selective_edge_predictions=str(predictions_path),
            dataset_dir=str(dataset_dir),
            source_parquet=str(source_path),
            out_dir=str(out_dir),
            model_name="candidate",
            threshold_top_fracs="0.50",
            cost_stress_bps="0.0",
            policy_id="candidate_top50",
            exit_mode="horizon",
            take_profit_bps=60.0,
            stop_loss_bps=45.0,
            same_bar_policy="stop_first",
            cooldown_bars=0,
            max_trades_per_day=0,
            daily_loss_limit_bps=0.0,
            min_direction_prob=0.0,
            min_score_floor=0.0,
            slippage_bps=0.0,
            size_multiplier=1.0,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "PASS"
    trades = pd.read_csv(out_dir / "candidate_replay_trade_log.csv")
    required = {"entry_time", "policy_id", "session", "side", "score", "p_long", "p_short", "p_flat", "net_pnl_bps", "mfe_bps", "mae_bps", "held_bars"}
    assert required.issubset(trades.columns)
    assert set(pd.to_datetime(trades["entry_time"], utc=True).dt.year) == {2026}
    assert int(trades["held_bars"].min()) == 1
    assert report["trainer_started"] is False
    assert report["replay_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
