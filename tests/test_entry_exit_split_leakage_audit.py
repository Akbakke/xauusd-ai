import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts.audit_entry_exit_split_leakage_v1 import run


SAFE_STATE_FEATURES = [
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "spread_bps",
    "atr_bps",
    "session",
    "vol_regime",
    "side",
    "entry_score",
]


def _state_reward_dataset(episodes: int = 6) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        episode_id = f"episode_{episode}"
        start = pd.Timestamp("2026-01-01 00:00:00+00:00") + pd.Timedelta(hours=episode)
        for step in range(3):
            terminal = step == 2
            rows.append(
                {
                    "exit_episode_id": episode_id,
                    "exit_timestep": step,
                    "bar_ts": (start + pd.Timedelta(minutes=5 * step)).isoformat(),
                    "logged_action": "EXIT_NOW" if terminal else "HOLD",
                    "logged_action_id": 1 if terminal else 0,
                    "is_terminal_transition": terminal,
                    "next_exit_episode_id": "" if terminal else episode_id,
                    "next_exit_timestep": np.nan if terminal else step + 1,
                    "hold_reward_bps": 0.0,
                    "forced_terminal_hold_reward_bps": 10.0 if terminal else 0.0,
                    "exit_now_reward_bps": 1.0 + step,
                    "logged_reward_bps": 10.0 if terminal else 0.0,
                    "terminal_reward_realized_net_pnl_bps": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _write_state_reward(tmp_path: Path, dataset: pd.DataFrame, *, ready: bool = True, state_features: list[str] | None = None) -> Path:
    root = tmp_path / "state_reward"
    root.mkdir(parents=True)
    dataset_csv = root / "entry_exit_state_reward_dataset.csv"
    dataset.to_csv(dataset_csv, index=False)
    report = {
        "decision": "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY" if ready else "BLOCKED_BY_EXIT_STATE_REWARD_CONTRACT",
        "state_reward_dataset_csv": str(dataset_csv),
        "dataset_rows": int(len(dataset)),
        "episode_count": int(dataset["exit_episode_id"].nunique()),
        "state_feature_names": state_features if state_features is not None else SAFE_STATE_FEATURES,
        "failures": [] if ready else [{"check": "state/reward dataset has rows"}],
    }
    path = root / "ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, state_reward_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        state_reward_json=str(state_reward_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_split_leakage_audit_passes_time_ordered_splits(tmp_path: Path) -> None:
    state_reward = _write_state_reward(tmp_path, _state_reward_dataset())

    report = run(_args(tmp_path, state_reward))

    assert report["decision"] == "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY"
    assert report["episode_count"] == 6
    assert report["temporal_review"]["ready"] is True
    assert report["next_pointer_split_review"]["ready"] is True
    assert report["reward_action_split_review"]["ready"] is True
    assert report["exit_training_allowed"] is False
    assert Path(report["split_dataset_csv"]).exists()


def test_entry_exit_split_leakage_audit_blocks_unready_state_reward(tmp_path: Path) -> None:
    state_reward = _write_state_reward(tmp_path, _state_reward_dataset(), ready=False)

    report = run(_args(tmp_path, state_reward))

    assert report["decision"] == "BLOCKED_BY_EXIT_SPLIT_LEAKAGE_AUDIT"
    failed = {row["check"] for row in report["failures"]}
    assert "active state/reward contract is ready" in failed


def test_entry_exit_split_leakage_audit_blocks_reward_shortcut_state_feature(tmp_path: Path) -> None:
    state_reward = _write_state_reward(
        tmp_path,
        _state_reward_dataset(),
        state_features=[*SAFE_STATE_FEATURES, "logged_reward_bps"],
    )

    report = run(_args(tmp_path, state_reward))

    assert report["decision"] == "BLOCKED_BY_EXIT_SPLIT_LEAKAGE_AUDIT"
    failed = {row["check"] for row in report["failures"]}
    assert "state features exclude reward/outcome shortcut fields" in failed
