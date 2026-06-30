import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts.materialize_entry_exit_model_dataset_readiness_v1 import run


NUMERIC = [
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "spread_bps",
    "atr_bps",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
]
CATEGORICAL = ["session", "vol_regime", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]


def _dataset() -> pd.DataFrame:
    rows = []
    split_by_episode = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test", 5: "test"}
    for episode in range(6):
        split = split_by_episode[episode]
        side = "LONG" if episode % 2 == 0 else "SHORT"
        session = "EU" if episode % 2 == 0 else "US"
        vol_regime = "2" if episode % 3 == 0 else "3"
        for step in range(3):
            terminal = step == 2
            rows.append(
                {
                    "entry_trade_id": f"trade_{episode}",
                    "bar_ts": f"2026-01-01T0{episode}:{step * 5:02d}:00+00:00",
                    "bar_index": step,
                    "exit_episode_id": f"episode_{episode}",
                    "exit_timestep": step,
                    "exit_split": split,
                    "entry_iql_policy_id": "entry_iql_student",
                    "entry_replay_identity_hash": "hash123",
                    "bar_price_source": "canonical_m5",
                    "bar_price_source_path": "/prices.parquet",
                    "running_pnl_bps": float(episode + step + 1),
                    "running_mfe_bps": float(episode + step + 3),
                    "running_mae_bps": float(step + 1),
                    "running_giveback_bps": float(step),
                    "bars_held": step,
                    "spread_bps": 1.0 + episode * 0.1 + step * 0.01,
                    "atr_bps": 5.0 + episode + step,
                    "session": session,
                    "vol_regime": vol_regime,
                    "side": side,
                    "entry_score": 0.7 + episode * 0.01,
                    "entry_p_long": 0.8 if side == "LONG" else 0.1,
                    "entry_p_short": 0.1 if side == "LONG" else 0.8,
                    "entry_p_flat": 0.08 + step * 0.01,
                    "entry_path_quality_pred": 1.0 + episode * 0.02,
                    "entry_bad_path_prob": 0.2 + step * 0.01,
                    "logged_action": "EXIT_NOW" if terminal else "HOLD",
                    "logged_action_id": 1 if terminal else 0,
                    "exit_now_label": terminal,
                    "hold_label": not terminal,
                    "is_terminal_transition": terminal,
                    "hold_reward_bps": 0.0,
                    "forced_terminal_hold_reward_bps": 10.0 if terminal else 0.0,
                    "exit_now_reward_bps": float(episode + step),
                    "logged_reward_bps": 10.0 if terminal else 0.0,
                    "terminal_reward_realized_net_pnl_bps": 10.0,
                    "next_exit_episode_id": "" if terminal else f"episode_{episode}",
                    "next_exit_timestep": np.nan if terminal else step + 1,
                    "next_row_available": not terminal,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(
    tmp_path: Path,
    dataset: pd.DataFrame,
    *,
    split_ready: bool = True,
    state_features: list[str] | None = None,
) -> Path:
    root = tmp_path / "inputs"
    root.mkdir(parents=True)
    state_reward_json = root / "ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.json"
    state_reward_json.write_text(
        json.dumps(
            {
                "decision": "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY",
                "state_feature_names": state_features or STATE_FEATURES,
                "state_feature_contract": {
                    "state_feature_names": state_features or STATE_FEATURES,
                    "numeric_state_features": NUMERIC,
                    "categorical_state_features": CATEGORICAL,
                    "state_timing": "AS_OF_CLOSED_M5_BAR_T_WITH_ENTRY_SNAPSHOT",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_csv = root / "entry_exit_state_reward_dataset_with_splits.csv"
    dataset.to_csv(dataset_csv, index=False)
    split_json = root / "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_latest.json"
    split_json.write_text(
        json.dumps(
            {
                "decision": "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY" if split_ready else "BLOCKED_BY_EXIT_SPLIT_LEAKAGE_AUDIT",
                "state_reward_json": str(state_reward_json),
                "split_dataset_csv": str(dataset_csv),
                "dataset_rows": int(len(dataset)),
                "episode_count": int(dataset["exit_episode_id"].nunique()),
                "state_feature_names": state_features or STATE_FEATURES,
                "failures": [] if split_ready else [{"check": "state features exclude reward/outcome shortcut fields"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return split_json


def _args(tmp_path: Path, split_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        split_leakage_json=str(split_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_model_dataset_readiness_writes_shards_schema_and_normalization(tmp_path: Path) -> None:
    split_json = _write_inputs(tmp_path, _dataset())

    report = run(_args(tmp_path, split_json))

    assert report["decision"] == "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
    assert report["dataset_rows"] == 18
    assert report["episode_count"] == 6
    assert report["split_review"]["ready"] is True
    assert report["numeric_review"]["ready"] is True
    assert report["categorical_review"]["ready"] is True
    assert report["exit_training_allowed"] is False
    assert set(report["model_dataset_shards"]) == {"train", "val", "test"}
    assert Path(report["feature_schema_json"]).exists()
    assert Path(report["normalization_json"]).exists()


def test_entry_exit_model_dataset_readiness_blocks_unready_split(tmp_path: Path) -> None:
    split_json = _write_inputs(tmp_path, _dataset(), split_ready=False)

    report = run(_args(tmp_path, split_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_MODEL_DATASET_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "active split/leakage audit is ready" in failed


def test_entry_exit_model_dataset_readiness_blocks_state_reward_shortcut(tmp_path: Path) -> None:
    split_json = _write_inputs(tmp_path, _dataset(), state_features=[*STATE_FEATURES, "logged_reward_bps"])

    report = run(_args(tmp_path, split_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_MODEL_DATASET_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "state features exclude reward/outcome/transition shortcuts" in failed
