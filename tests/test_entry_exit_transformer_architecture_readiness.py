import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_transformer_architecture_readiness_v1 import run


NUMERIC = ["running_pnl_bps", "running_mfe_bps", "bars_held", "atr_bps"]
CATEGORICAL = ["session", "vol_regime", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]


def _shard(split: str, episodes: int, offset: int = 0) -> pd.DataFrame:
    rows = []
    for episode in range(offset, offset + episodes):
        for step in range(3):
            rows.append(
                {
                    "exit_episode_id": f"{split}_{episode}",
                    "exit_timestep": step,
                    "is_terminal_transition": step == 2,
                }
            )
    return pd.DataFrame(rows)


def _write_model_dataset(
    tmp_path: Path,
    *,
    ready: bool = True,
    state_features: list[str] | None = None,
) -> Path:
    root = tmp_path / "model_dataset"
    root.mkdir(parents=True)
    shards = {
        "train": root / "train.csv",
        "val": root / "val.csv",
        "test": root / "test.csv",
    }
    _shard("train", 3, 0).to_csv(shards["train"], index=False)
    _shard("val", 1, 10).to_csv(shards["val"], index=False)
    _shard("test", 1, 20).to_csv(shards["test"], index=False)
    features = state_features or STATE_FEATURES
    report = {
        "decision": (
            "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
            if ready
            else "BLOCKED_BY_EXIT_MODEL_DATASET_READINESS"
        ),
        "dataset_rows": 15,
        "episode_count": 5,
        "model_dataset_shards": {key: str(path) for key, path in shards.items()},
        "feature_schema": {
            "state_feature_names": features,
            "numeric_state_features": NUMERIC,
            "categorical_state_features": CATEGORICAL,
        },
        "normalization": {
            "normalization_policy": "fit_numeric_mean_std_and_categorical_vocab_on_train_split_only",
            "numeric": {field: {"mean": 0.0, "std": 1.0} for field in NUMERIC},
            "categorical_vocab": {
                "session": ["EU", "US"],
                "vol_regime": ["2", "3"],
                "side": ["LONG", "SHORT"],
            },
        },
    }
    path = root / "ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, model_dataset_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_dataset_json=str(model_dataset_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_architecture_readiness_passes_contract(tmp_path: Path) -> None:
    model_dataset = _write_model_dataset(tmp_path)

    report = run(_args(tmp_path, model_dataset))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW"
    assert report["architecture_review"]["ready"] is True
    assert report["sequence_review"]["ready"] is True
    assert report["feature_contract_review"]["ready"] is True
    assert report["architecture_contract"]["sequence_encoder"]["d_model"] == 128
    assert report["architecture_contract"]["output_heads"] == [
        "exit_now_logit",
        "hold_value_bps",
        "exit_now_reward_bps",
        "giveback_risk_bps",
        "mfe_capture_ratio",
    ]
    assert report["exit_training_allowed"] is False


def test_entry_exit_transformer_architecture_readiness_blocks_unready_model_dataset(tmp_path: Path) -> None:
    model_dataset = _write_model_dataset(tmp_path, ready=False)

    report = run(_args(tmp_path, model_dataset))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "active model dataset readiness is ready" in failed


def test_entry_exit_transformer_architecture_readiness_blocks_shortcut_feature(tmp_path: Path) -> None:
    model_dataset = _write_model_dataset(tmp_path, state_features=[*STATE_FEATURES, "logged_reward_bps"])

    report = run(_args(tmp_path, model_dataset))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "feature contract is exact and train-normalized" in failed
