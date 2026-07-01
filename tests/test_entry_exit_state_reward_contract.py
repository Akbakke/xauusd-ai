import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_exit_state_reward_contract_v1 import (
    BASE_SPECIALIST_GATE_STATE_FEATURES,
    ENTRY_ALIGNMENT_STATE_FEATURES,
    run,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for trade_idx, side in enumerate(("LONG", "SHORT")):
        entry_time = pd.Timestamp(f"2026-01-01 0{trade_idx}:00:00+00:00")
        exit_time = entry_time + pd.Timedelta(minutes=10)
        trade_id = f"entry_iql_student:{trade_idx}:{entry_time.isoformat()}:{side}"
        for bar_index in range(3):
            bar_ts = entry_time + pd.Timedelta(minutes=5 * bar_index)
            running_pnl = 10.0 + bar_index
            running_mfe = 20.0 + bar_index
            running_mae = 2.0 + bar_index
            realized_net = 25.0
            realized_gross = 26.0
            realized_mfe = 40.0
            realized_mae = 5.0
            realized_exit_reason = "tp"
            if trade_idx == 1:
                running_pnl = [10.0, -10.0, -45.0][bar_index]
                running_mfe = 30.0
                running_mae = [2.0, 20.0, 45.0][bar_index]
                realized_net = -45.0
                realized_gross = -44.0
                realized_mfe = 30.0
                realized_mae = 45.0
                realized_exit_reason = "STOP_LOSS"
            row = {
                "entry_trade_id": trade_id,
                "bar_ts": bar_ts.isoformat(),
                "bar_index": bar_index,
                "side": side,
                "action_set": "HOLD,EXIT_NOW",
                "running_pnl_bps": running_pnl,
                "running_mfe_bps": running_mfe,
                "running_mae_bps": running_mae,
                "running_giveback_bps": 1.0,
                "bars_held": bar_index,
                "session": "EU" if trade_idx == 0 else "US",
                "vol_regime": "4" if trade_idx == 0 else "5",
                "spread_bps": 1.2,
                "atr_bps": 5.0 + bar_index + trade_idx,
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
                "realized_net_pnl_bps": realized_net,
                "realized_gross_pnl_bps": realized_gross,
                "realized_mfe_bps": realized_mfe,
                "realized_mae_bps": realized_mae,
                "realized_exit_reason": realized_exit_reason,
                "is_realized_exit_bar": bar_index == 2,
            }
            for pos, field in enumerate((*ENTRY_ALIGNMENT_STATE_FEATURES, *BASE_SPECIALIST_GATE_STATE_FEATURES)):
                row[field] = float(pos + bar_index + trade_idx + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_reconstruction(tmp_path: Path, dataset: pd.DataFrame, *, ready: bool = True) -> Path:
    root = tmp_path / "reconstruction"
    root.mkdir(parents=True)
    dataset_csv = root / "entry_exit_per_bar_handoff.csv"
    dataset.to_csv(dataset_csv, index=False)
    report = {
        "decision": "READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW" if ready else "BLOCKED_BY_EXIT_RECONSTRUCTION_AUDIT",
        "dataset_csv": str(dataset_csv),
        "dataset_rows": int(len(dataset)),
        "observed_trade_count": int(dataset["entry_trade_id"].nunique()),
        "failures": [] if ready else [{"check": "per-trade timeline reconstruction is contiguous and terminal"}],
    }
    path = root / "ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, reconstruction_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reconstruction_audit_json=str(reconstruction_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_state_reward_contract_materializes_actions_rewards_and_pointers(tmp_path: Path) -> None:
    reconstruction = _write_reconstruction(tmp_path, _dataset())

    report = run(_args(tmp_path, reconstruction))

    assert report["decision"] == "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY"
    assert report["dataset_rows"] == 6
    assert report["episode_count"] == 2
    assert report["action_counts"] == {"HOLD": 4, "EXIT_NOW": 2}
    assert report["pointer_review"]["ready"] is True
    assert report["exit_training_allowed"] is False
    dataset = pd.read_csv(report["state_reward_dataset_csv"])
    assert {"exit_now_label", "hold_reward_bps", "exit_now_reward_bps", "logged_reward_bps"}.issubset(dataset.columns)
    assert dataset.loc[dataset["logged_action"].eq("HOLD"), "logged_reward_bps"].eq(0.0).all()
    assert sorted(dataset.loc[dataset["logged_action"].eq("EXIT_NOW"), "logged_reward_bps"].tolist()) == [-45.0, 25.0]
    assert report["hazard_label_review"]["ready"] is True
    assert {
        "future_max_running_pnl_bps",
        "future_adverse_excursion_bps",
        "exit_hazard_adverse_15bps_label",
        "positive_mfe_stopout_episode_label",
        "oracle_exit_before_giveback_label",
    }.issubset(dataset.columns)
    assert dataset["positive_mfe_stopout_episode_label"].sum() == 3


def test_entry_exit_state_reward_contract_blocks_when_reconstruction_not_ready(tmp_path: Path) -> None:
    reconstruction = _write_reconstruction(tmp_path, _dataset(), ready=False)

    report = run(_args(tmp_path, reconstruction))

    assert report["decision"] == "BLOCKED_BY_EXIT_STATE_REWARD_CONTRACT"
    failed = {row["check"] for row in report["failures"]}
    assert "active reconstruction audit is ready for state/reward contract review" in failed
    assert report["exit_training_allowed"] is False
