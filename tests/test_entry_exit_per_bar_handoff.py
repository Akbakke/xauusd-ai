import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_handoff_readiness_v1 import REQUIRED_EXIT_SUBSTRATE_FIELDS
from gx1.scripts.materialize_entry_exit_per_bar_handoff_v1 import run


def test_entry_exit_per_bar_handoff_materializes_required_fields(tmp_path: Path) -> None:
    trade_log = tmp_path / "iql_trades.csv"
    pd.DataFrame(
        [
            {
                "fold": "2026_TEST",
                "policy_id": "entry_iql_student",
                "session": "EU",
                "vol_regime": "4",
                "entry_time": "2026-01-01 00:00:00+00:00",
                "exit_time": "2026-01-01 00:10:00+00:00",
                "side": "LONG",
                "score": 0.9,
                "p_long": 0.8,
                "p_short": 0.1,
                "p_flat": 0.1,
                "path_quality_pred": 1.0,
                "bad_path_prob": 0.2,
                "entry_price": 101.0,
                "exit_price": 103.0,
                "gross_pnl_bps": 198.0198,
                "net_pnl_bps": 198.0198,
                "mfe_bps": 297.0297,
                "mae_bps": 99.0099,
                "held_bars": 2,
                "horizon_bars": 2,
                "exit_reason": "horizon",
            }
        ]
    ).to_csv(trade_log, index=False)
    price = tmp_path / "m5.parquet"
    pd.DataFrame(
        [
            {
                "time": "2026-01-01 00:00:00+00:00",
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 100.5,
                "bid_open": 100.0,
                "bid_high": 102.0,
                "bid_low": 100.0,
                "bid_close": 100.0,
                "ask_open": 101.0,
                "ask_high": 103.0,
                "ask_low": 101.0,
                "ask_close": 101.0,
                "atr_bps": 5.0,
                "spread_bps": 1.0,
            },
            {
                "time": "2026-01-01 00:05:00+00:00",
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "bid_open": 101.0,
                "bid_high": 103.0,
                "bid_low": 101.0,
                "bid_close": 102.0,
                "ask_open": 102.0,
                "ask_high": 104.0,
                "ask_low": 102.0,
                "ask_close": 103.0,
                "atr_bps": 6.0,
                "spread_bps": 1.0,
            },
            {
                "time": "2026-01-01 00:10:00+00:00",
                "open": 102.5,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "bid_open": 102.0,
                "bid_high": 104.0,
                "bid_low": 102.0,
                "bid_close": 103.0,
                "ask_open": 103.0,
                "ask_high": 105.0,
                "ask_low": 103.0,
                "ask_close": 104.0,
                "atr_bps": 7.0,
                "spread_bps": 1.0,
            },
        ]
    ).assign(time=lambda frame: pd.to_datetime(frame["time"], utc=True)).to_parquet(price, index=False)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_PROMOTION_REVIEW_VEDTAK",
                "evidence_identity": {"candidate_bundle_dir": "/candidate"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    slice_audit = tmp_path / "slice.json"
    slice_audit.write_text(json.dumps({"decision": "PASS"}) + "\n", encoding="utf-8")

    report = run(
        argparse.Namespace(
            iql_trade_log=str(trade_log),
            m5_price_parquet=str(price),
            supplemental_m1_glob=str(tmp_path / "none_*.parquet"),
            iql_comparison_json=str(comparison),
            iql_slice_audit_json=str(slice_audit),
            out_dir=str(tmp_path / "out"),
            min_covered_trade_ratio=0.95,
            min_covered_trades=1,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "PASS"
    assert report["dataset_rows"] == 3
    assert report["complete_trade_count"] == 1
    dataset = pd.read_csv(report["dataset_csv"])
    assert set(REQUIRED_EXIT_SUBSTRATE_FIELDS).issubset(dataset.columns)
    assert dataset.iloc[-1]["running_pnl_bps"] > 190.0
    assert dataset.iloc[-1]["running_mfe_bps"] > 290.0
    assert report["exit_training_allowed"] is False
    assert report["exit_iql_allowed"] is False


def test_entry_exit_per_bar_handoff_uses_hashed_m1_supplement_for_missing_m5_bars(tmp_path: Path) -> None:
    trade_log = tmp_path / "iql_trades.csv"
    pd.DataFrame(
        [
            {
                "fold": "2026_TEST",
                "policy_id": "entry_iql_student",
                "session": "EU",
                "vol_regime": "4",
                "entry_time": "2026-01-01 00:00:00+00:00",
                "exit_time": "2026-01-01 00:10:00+00:00",
                "side": "LONG",
                "score": 0.9,
                "p_long": 0.8,
                "p_short": 0.1,
                "p_flat": 0.1,
                "path_quality_pred": 1.0,
                "bad_path_prob": 0.2,
                "entry_price": 101.0,
                "exit_price": 103.0,
                "gross_pnl_bps": 198.0198,
                "net_pnl_bps": 198.0198,
                "mfe_bps": 297.0297,
                "mae_bps": 99.0099,
                "held_bars": 2,
                "horizon_bars": 2,
                "exit_reason": "horizon",
            }
        ]
    ).to_csv(trade_log, index=False)
    price = tmp_path / "m5.parquet"
    pd.DataFrame(
        [
            {
                "time": "2026-01-01 00:00:00+00:00",
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 100.5,
                "bid_open": 100.0,
                "bid_high": 102.0,
                "bid_low": 100.0,
                "bid_close": 100.0,
                "ask_open": 101.0,
                "ask_high": 103.0,
                "ask_low": 101.0,
                "ask_close": 101.0,
            }
        ]
    ).assign(time=lambda frame: pd.to_datetime(frame["time"], utc=True)).to_parquet(price, index=False)
    m1 = tmp_path / "xauusd_m1_20260101.parquet"
    pd.DataFrame(
        [
            {
                "time": f"2026-01-01 00:{minute:02d}:00+00:00",
                "open": 101.5 + minute,
                "high": 102.0 + minute,
                "low": 101.0 + minute,
                "close": 101.7 + minute,
                "volume": 1.0,
                "bid_open": 101.0 + minute,
                "bid_high": 102.0 + minute,
                "bid_low": 101.0 + minute,
                "bid_close": 101.5 + minute,
                "ask_open": 102.0 + minute,
                "ask_high": 103.0 + minute,
                "ask_low": 102.0 + minute,
                "ask_close": 102.5 + minute,
            }
            for minute in range(5, 11)
        ]
    ).assign(time=lambda frame: pd.to_datetime(frame["time"], utc=True)).to_parquet(m1, index=False)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_PROMOTION_REVIEW_VEDTAK",
                "evidence_identity": {"candidate_bundle_dir": "/candidate"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    slice_audit = tmp_path / "slice.json"
    slice_audit.write_text(json.dumps({"decision": "PASS"}) + "\n", encoding="utf-8")

    report = run(
        argparse.Namespace(
            iql_trade_log=str(trade_log),
            m5_price_parquet=str(price),
            supplemental_m1_glob=str(tmp_path / "xauusd_m1_*.parquet"),
            iql_comparison_json=str(comparison),
            iql_slice_audit_json=str(slice_audit),
            out_dir=str(tmp_path / "out"),
            min_covered_trade_ratio=0.95,
            min_covered_trades=1,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "PASS"
    assert report["dataset_rows"] == 3
    assert report["complete_trade_count"] == 1
    diagnostics = report["price_diagnostics"]
    assert diagnostics["supplemental_rows_used"] == 2
    assert diagnostics["supplemental_paths_used"] == [str(m1.resolve())]
    assert diagnostics["supplemental_input_sha256"][str(m1.resolve())]
    dataset = pd.read_csv(report["dataset_csv"])
    assert set(dataset["bar_price_source"]) == {"canonical_m5", "supplemental_live_m1_to_m5"}


def test_entry_exit_per_bar_handoff_excludes_non_contiguous_trade_bars(tmp_path: Path) -> None:
    trade_log = tmp_path / "iql_trades.csv"
    pd.DataFrame(
        [
            {
                "fold": "2026_TEST",
                "policy_id": "entry_iql_student",
                "session": "EU",
                "vol_regime": "4",
                "entry_time": "2026-01-01 00:00:00+00:00",
                "exit_time": "2026-01-01 00:20:00+00:00",
                "side": "LONG",
                "score": 0.9,
                "p_long": 0.8,
                "p_short": 0.1,
                "p_flat": 0.1,
                "path_quality_pred": 1.0,
                "bad_path_prob": 0.2,
                "entry_price": 101.0,
                "exit_price": 103.0,
                "gross_pnl_bps": 198.0198,
                "net_pnl_bps": 198.0198,
                "mfe_bps": 297.0297,
                "mae_bps": 99.0099,
                "held_bars": 2,
                "horizon_bars": 2,
                "exit_reason": "horizon",
            }
        ]
    ).to_csv(trade_log, index=False)
    price = tmp_path / "m5.parquet"
    pd.DataFrame(
        [
            {
                "time": "2026-01-01 00:00:00+00:00",
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 100.5,
                "bid_open": 100.0,
                "bid_high": 102.0,
                "bid_low": 100.0,
                "bid_close": 100.0,
                "ask_open": 101.0,
                "ask_high": 103.0,
                "ask_low": 101.0,
                "ask_close": 101.0,
            },
            {
                "time": "2026-01-01 00:05:00+00:00",
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "bid_open": 101.0,
                "bid_high": 103.0,
                "bid_low": 101.0,
                "bid_close": 102.0,
                "ask_open": 102.0,
                "ask_high": 104.0,
                "ask_low": 102.0,
                "ask_close": 103.0,
            },
            {
                "time": "2026-01-01 00:20:00+00:00",
                "open": 102.5,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "bid_open": 102.0,
                "bid_high": 104.0,
                "bid_low": 102.0,
                "bid_close": 103.0,
                "ask_open": 103.0,
                "ask_high": 105.0,
                "ask_low": 103.0,
                "ask_close": 104.0,
            },
        ]
    ).assign(time=lambda frame: pd.to_datetime(frame["time"], utc=True)).to_parquet(price, index=False)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_PROMOTION_REVIEW_VEDTAK",
                "evidence_identity": {"candidate_bundle_dir": "/candidate"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    slice_audit = tmp_path / "slice.json"
    slice_audit.write_text(json.dumps({"decision": "PASS"}) + "\n", encoding="utf-8")

    report = run(
        argparse.Namespace(
            iql_trade_log=str(trade_log),
            m5_price_parquet=str(price),
            supplemental_m1_glob=str(tmp_path / "none_*.parquet"),
            iql_comparison_json=str(comparison),
            iql_slice_audit_json=str(slice_audit),
            out_dir=str(tmp_path / "out"),
            min_covered_trade_ratio=0.0,
            min_covered_trades=0,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert report["included_trade_count"] == 0
    assert report["excluded_trade_count"] == 1
    exclusions = pd.read_csv(report["gap_exclusions_csv"])
    assert "non_contiguous_5min_per_bar_price_coverage" in exclusions.iloc[0]["reason"]
