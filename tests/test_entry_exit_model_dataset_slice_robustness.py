import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_model_dataset_slice_robustness_v1 import run


NUMERIC = ["running_pnl_bps", "running_mfe_bps", "bars_held", "atr_bps"]
CATEGORICAL = ["session", "vol_regime", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]
REWARDS = [
    "hold_reward_bps",
    "forced_terminal_hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "terminal_reward_realized_net_pnl_bps",
]


def _rows(split: str, *, unsupported_long: bool = False) -> list[dict]:
    rows = []
    configs = [
        ("ASIA", "4", "SHORT"),
        ("US", "3", "LONG"),
    ]
    for idx, (session, regime, side) in enumerate(configs):
        for episode in range(2):
            for step in range(3):
                is_terminal = step == 2
                exit_now = is_terminal
                if unsupported_long and side == "LONG":
                    exit_now = False
                row = {
                    "exit_episode_id": f"{split}_{idx}_{episode}",
                    "exit_timestep": step,
                    "exit_split": split,
                    "session": session,
                    "vol_regime": regime,
                    "side": side,
                    "exit_now_label": exit_now,
                    "hold_label": not exit_now,
                    "is_terminal_transition": is_terminal,
                    "running_pnl_bps": float(step + idx),
                    "running_mfe_bps": float(step + 2),
                    "bars_held": float(step),
                    "atr_bps": float(10 + step + idx),
                }
                for reward in REWARDS:
                    row[reward] = float(step - idx)
                rows.append(row)
    return rows


def _write_inputs(tmp_path: Path, *, unsupported_long: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "data"
    root.mkdir()
    shards = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.csv"
        pd.DataFrame(_rows(split, unsupported_long=unsupported_long)).to_csv(path, index=False)
        shards[split] = path
    model_dataset = {
        "decision": "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW",
        "model_dataset_shards": {split: str(path) for split, path in shards.items()},
        "feature_schema": {
            "state_feature_names": STATE_FEATURES,
            "numeric_state_features": NUMERIC,
            "categorical_state_features": CATEGORICAL,
        },
    }
    model_dataset_json = root / "ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"
    model_dataset_json.write_text(json.dumps(model_dataset, indent=2) + "\n", encoding="utf-8")
    pretrain_manifest = {
        "decision": "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
    }
    pretrain_manifest_json = root / "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.json"
    pretrain_manifest_json.write_text(json.dumps(pretrain_manifest, indent=2) + "\n", encoding="utf-8")
    return model_dataset_json, pretrain_manifest_json


def _args(tmp_path: Path, model_dataset_json: Path, pretrain_manifest_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_dataset_json=str(model_dataset_json),
        pretrain_manifest_json=str(pretrain_manifest_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_model_dataset_slice_robustness_passes_with_weak_slice_disclosure(tmp_path: Path) -> None:
    model_dataset_json, pretrain_manifest_json = _write_inputs(tmp_path)

    report = run(_args(tmp_path, model_dataset_json, pretrain_manifest_json))

    assert report["decision"] == "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
    assert report["slice_review"]["unsupported_slice_count"] == 0
    assert report["slice_review"]["weak_slice_count"] > 0
    assert all(row["ready"] for row in report["split_reviews"].values())
    assert report["exit_training_allowed"] is False
    assert report["trainer_started"] is False


def test_entry_exit_model_dataset_slice_robustness_blocks_unsupported_slice(tmp_path: Path) -> None:
    model_dataset_json, pretrain_manifest_json = _write_inputs(tmp_path, unsupported_long=True)

    report = run(_args(tmp_path, model_dataset_json, pretrain_manifest_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS"
    failed = {row["check"] for row in report["failures"]}
    assert "session/regime/side slices are disclosed without unsupported slices" in failed
    assert report["slice_review"]["unsupported_slice_count"] > 0
