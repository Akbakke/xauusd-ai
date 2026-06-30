import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_exit_transformer_pretrain_manifest_v1 import run


HEADS = [
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
]
NUMERIC = ["running_pnl_bps", "bars_held"]
CATEGORICAL = ["session", "side"]


def _write_training_plan(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for episode in range(2):
        for step in range(3):
            rows.append(
                {
                    "exit_episode_id": f"ep_{episode}",
                    "exit_timestep": step,
                    "exit_split": "train",
                    "running_pnl_bps": float(step),
                    "bars_held": float(step),
                    "session": "ASIA",
                    "side": "LONG",
                }
            )
    shard = root / "train.csv"
    pd.DataFrame(rows).to_csv(shard, index=False)
    normalization_json = root / "normalization.json"
    normalization_json.write_text(
        json.dumps(
            {
                "normalization_policy": "fit_numeric_mean_std_and_categorical_vocab_on_train_split_only",
                "numeric": {
                    "running_pnl_bps": {"mean": 0.0, "std": 1.0},
                    "bars_held": {"mean": 1.0, "std": 1.0},
                },
                "categorical_vocab": {"session": ["ASIA"], "side": ["LONG"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "decision": "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW",
        "training_plan": {
            "model_family": "exit_sequence_transformer_v1",
            "architecture": {
                "encoder": {
                    "d_model": 16,
                    "n_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 32,
                    "dropout": 0.0,
                    "causal_mask_required": True,
                },
                "planned_max_sequence_length": 8,
                "numeric_state_features": NUMERIC,
                "categorical_state_features": CATEGORICAL,
                "output_heads": HEADS,
            },
            "dataset": {
                "shards": {"train": str(shard), "val": str(shard), "test": str(shard)},
                "normalization_json": str(normalization_json),
            },
            "resource_guardrails": {
                "num_workers": 0,
                "max_process_rss_gib": 8,
                "abort_if_mem_available_below_gib": 8,
            },
        },
    }
    path = root / "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _write_wrapper_readiness(tmp_path: Path, *, ready: bool = True) -> Path:
    path = tmp_path / "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json"
    path.write_text(
        json.dumps(
            {
                "decision": (
                    "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
                    if ready
                    else "BLOCKED_BY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS"
                ),
                "exit_training_allowed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, training_plan: Path, wrapper_readiness: Path) -> argparse.Namespace:
    return argparse.Namespace(
        training_plan_json=str(training_plan),
        wrapper_readiness_json=str(wrapper_readiness),
        out_dir=str(tmp_path / "out"),
        split="train",
        max_episodes=2,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_pretrain_manifest_passes_finite_forward(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)
    wrapper_readiness = _write_wrapper_readiness(tmp_path)

    report = run(_args(tmp_path, training_plan, wrapper_readiness))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
    assert report["preflight_manifest"]["decision"] == "PASS"
    assert report["preflight_manifest"]["output_heads"] == HEADS
    assert all(report["preflight_manifest"]["finite_by_head"].values())
    assert report["trainer_started"] is False
    assert report["exit_training_allowed_with_explicit_vedtak"] is False


def test_entry_exit_transformer_pretrain_manifest_blocks_unready_wrapper(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)
    wrapper_readiness = _write_wrapper_readiness(tmp_path, ready=False)

    report = run(_args(tmp_path, training_plan, wrapper_readiness))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer trainer wrapper readiness is ready" in failed
