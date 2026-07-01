import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.models.exit_sequence_transformer.train_v1 import TRAINING_COMPLETE_DECISION, run_preflight, run_training


HEADS = [
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
]
NUMERIC = ["running_pnl_bps", "bars_held"]
CATEGORICAL = ["session", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]


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
                    "running_pnl_bps": float(step - episode),
                    "bars_held": float(step),
                    "session": "ASIA" if episode == 0 else "US",
                    "side": "LONG" if episode == 0 else "SHORT",
                    "exit_now_label": step == 2,
                    "future_best_exit_lift_bps": float(10 - step),
                    "exit_now_reward_bps": float(step * 5 - episode),
                    "future_giveback_from_peak_bps": float(episode + step),
                    "exit_now_mfe_capture_ratio_reward": float(step / 3.0),
                }
            )
    shard = root / "train.csv"
    pd.DataFrame(rows).to_csv(shard, index=False)
    normalization = {
        "normalization_policy": "fit_numeric_mean_std_and_categorical_vocab_on_train_split_only",
        "numeric": {
            "running_pnl_bps": {"mean": 0.0, "std": 1.0},
            "bars_held": {"mean": 1.0, "std": 1.0},
        },
        "categorical_vocab": {
            "session": ["ASIA", "US"],
            "side": ["LONG", "SHORT"],
        },
    }
    normalization_json = root / "normalization.json"
    normalization_json.write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
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


def _write_ready_report(path: Path, decision: str) -> Path:
    path.write_text(json.dumps({"decision": decision, "promotion_shadow_live_allowed": False}, indent=2) + "\n", encoding="utf-8")
    return path


def test_exit_sequence_transformer_train_v1_preflight_passes(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)

    manifest = run_preflight(
        argparse.Namespace(
            training_plan_json=str(training_plan),
            preflight_only=True,
            out_manifest_json="",
            split="train",
            max_episodes=2,
            device="cpu",
        )
    )

    assert manifest["decision"] == "PASS"
    assert manifest["output_heads"] == HEADS
    assert all(manifest["finite_by_head"].values())
    assert manifest["trainer_started"] is False
    assert manifest["optimizer_steps"] == 0
    assert manifest["preflight_batch"]["valid_token_count"] == 6


def test_exit_sequence_transformer_train_v1_supervised_training_writes_bundle(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)
    train_review = _write_ready_report(
        tmp_path / "train_execution_review.json",
        "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE",
    )
    post_contract = _write_ready_report(
        tmp_path / "post_train_contract.json",
        "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY",
    )
    feature_alignment = _write_ready_report(
        tmp_path / "feature_alignment.json",
        "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW",
    )
    out_bundle = tmp_path / "bundle"

    report = run_training(
        argparse.Namespace(
            training_plan_json=str(training_plan),
            train_execution_review_json=str(train_review),
            post_train_contract_json=str(post_contract),
            feature_alignment_json=str(feature_alignment),
            vedtak="ENTRY_EXIT_TRANSFORMER_TRAIN_TEST_V1",
            enable_training=True,
            out_bundle_dir=str(out_bundle),
            device="cpu",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            clip_grad_norm=1.0,
            num_workers=0,
            seed=7,
            max_train_episodes=0,
            max_eval_episodes=0,
        )
    )

    assert report["decision"] == TRAINING_COMPLETE_DECISION
    assert report["trainer_started"] is True
    assert report["optimizer_steps"] > 0
    assert report["replay_started"] is False
    assert report["iql_distillation_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["final_metrics"]["val"]["finite_by_head"] == {head: True for head in HEADS}
    assert (out_bundle / "model_state_dict.pt").is_file()
    assert (out_bundle / "bundle_metadata.json").is_file()
