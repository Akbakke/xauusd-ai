import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts.audit_entry_exit_per_bar_reconstruction_v1 import run


def _dataset(*, atr_nan: bool = False, terminal_middle: bool = False) -> pd.DataFrame:
    rows = []
    for trade_idx, side in enumerate(("LONG", "SHORT")):
        entry_time = pd.Timestamp(f"2026-01-01 0{trade_idx}:00:00+00:00")
        exit_time = entry_time + pd.Timedelta(minutes=10)
        trade_id = f"entry_iql_student:{trade_idx}:{entry_time.isoformat()}:{side}"
        for bar_index in range(3):
            bar_ts = entry_time + pd.Timedelta(minutes=5 * bar_index)
            rows.append(
                {
                    "entry_trade_id": trade_id,
                    "bar_ts": bar_ts.isoformat(),
                    "bar_index": bar_index,
                    "side": side,
                    "action_set": "HOLD,EXIT_NOW",
                    "running_pnl_bps": 10.0 + bar_index + trade_idx,
                    "running_mfe_bps": 20.0 + bar_index + trade_idx,
                    "running_mae_bps": 2.0 + bar_index,
                    "running_giveback_bps": 1.0,
                    "bars_held": bar_index,
                    "session": "EU" if trade_idx == 0 else "US",
                    "vol_regime": "4" if trade_idx == 0 else "5",
                    "spread_bps": 1.2,
                    "atr_bps": np.nan if atr_nan else 5.0 + bar_index + trade_idx,
                    "bar_price_source": "canonical_m5",
                    "bar_price_source_path": "/prices.parquet",
                    "entry_score": 0.9,
                    "entry_p_long": 0.8 if side == "LONG" else 0.1,
                    "entry_p_short": 0.1 if side == "LONG" else 0.8,
                    "entry_p_flat": 0.1,
                    "entry_path_quality_pred": 1.2,
                    "entry_bad_path_prob": 0.2,
                    "entry_candidate_bundle_dir": "/candidate",
                    "entry_iql_policy_id": "entry_iql_student",
                    "entry_replay_identity_hash": "hash123",
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "realized_net_pnl_bps": 25.0,
                    "realized_gross_pnl_bps": 26.0,
                    "realized_mfe_bps": 40.0,
                    "realized_mae_bps": 5.0,
                    "realized_exit_reason": "tp",
                    "is_realized_exit_bar": bar_index == (1 if terminal_middle and trade_idx == 0 else 2),
                }
            )
    return pd.DataFrame(rows)


def _write_handoff(tmp_path: Path, dataset: pd.DataFrame) -> Path:
    root = tmp_path / "handoff"
    root.mkdir(parents=True)
    dataset_csv = root / "entry_exit_per_bar_handoff.csv"
    gaps_csv = root / "entry_exit_per_bar_handoff_gap_exclusions.csv"
    dataset.to_csv(dataset_csv, index=False)
    pd.DataFrame(columns=["entry_trade_id", "reason"]).to_csv(gaps_csv, index=False)
    trade_count = int(dataset["entry_trade_id"].nunique())
    report = {
        "decision": "PASS",
        "dataset_csv": str(dataset_csv),
        "dataset_rows": int(len(dataset)),
        "source_trade_count": trade_count,
        "included_trade_count": trade_count,
        "complete_trade_count": trade_count,
        "excluded_trade_count": 0,
        "covered_trade_ratio": 1.0,
        "gap_exclusions_csv": str(gaps_csv),
        "failures": [],
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "promotion_shadow_live_allowed": False,
    }
    path = root / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, handoff_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        handoff_json=str(handoff_json),
        out_dir=str(tmp_path / "out"),
        min_covered_trade_ratio=0.95,
        min_included_trades=1,
        max_probability_sum_error=0.05,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_per_bar_reconstruction_audit_passes_live_reconstruction(tmp_path: Path) -> None:
    handoff_json = _write_handoff(tmp_path, _dataset())

    report = run(_args(tmp_path, handoff_json))

    assert report["decision"] == "READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW"
    assert report["exit_training_allowed"] is False
    assert report["exit_iql_allowed"] is False
    assert report["atr_liveness"]["ready"] is True
    assert report["per_trade_review"]["ready"] is True
    assert report["failures"] == []
    assert Path(report["json_path"]).exists()


def test_entry_exit_per_bar_reconstruction_audit_blocks_dead_atr(tmp_path: Path) -> None:
    handoff_json = _write_handoff(tmp_path, _dataset(atr_nan=True))

    report = run(_args(tmp_path, handoff_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_RECONSTRUCTION_AUDIT"
    failed = {row["check"] for row in report["failures"]}
    assert "per-bar numeric state fields are finite" in failed
    assert "atr_bps is positive and live" in failed
    assert report["exit_training_allowed"] is False


def test_entry_exit_per_bar_reconstruction_audit_blocks_terminal_not_last(tmp_path: Path) -> None:
    handoff_json = _write_handoff(tmp_path, _dataset(terminal_middle=True))

    report = run(_args(tmp_path, handoff_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_RECONSTRUCTION_AUDIT"
    failed = {row["check"] for row in report["failures"]}
    assert "per-trade timeline reconstruction is contiguous and terminal" in failed
    assert report["per_trade_review"]["failure_count"] == 1
