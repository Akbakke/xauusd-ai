import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts.materialize_entry_exit_transformer_training_plan_readiness_v1 import run


HEADS = [
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
]
NUMERIC = ["running_pnl_bps", "running_mfe_bps", "bars_held", "atr_bps"]
CATEGORICAL = ["session", "vol_regime", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_architecture_bundle(
    tmp_path: Path,
    *,
    architecture_ready: bool = True,
    shard_hash_mismatch: bool = False,
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    shards = {
        "train": root / "train.csv",
        "val": root / "val.csv",
        "test": root / "test.csv",
    }
    for split, path in shards.items():
        path.write_text(f"exit_episode_id,exit_timestep,exit_split\n{split}_1,0,{split}\n", encoding="utf-8")

    shard_hashes = {split: _sha256(path) for split, path in shards.items()}
    if shard_hash_mismatch:
        shard_hashes["train"] = "0" * 64

    model_dataset = {
        "decision": "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW",
        "dataset_rows": 3,
        "episode_count": 3,
        "model_dataset_shards": {split: str(path) for split, path in shards.items()},
        "model_dataset_shard_sha256": shard_hashes,
        "feature_schema": {
            "state_feature_names": STATE_FEATURES,
            "numeric_state_features": NUMERIC,
            "categorical_state_features": CATEGORICAL,
        },
        "normalization_json": str(root / "normalization.json"),
    }
    feature_schema_json = root / "entry_exit_model_dataset_feature_schema.json"
    feature_schema_json.write_text(json.dumps(model_dataset["feature_schema"], indent=2) + "\n", encoding="utf-8")
    model_dataset["feature_schema_json"] = str(feature_schema_json)
    model_dataset_json = root / "ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"
    model_dataset_json.write_text(json.dumps(model_dataset, indent=2) + "\n", encoding="utf-8")
    (root / "normalization.json").write_text('{"normalization_policy":"train_split_only"}\n', encoding="utf-8")

    architecture_contract = {
        "model_family": "exit_sequence_transformer_v1",
        "input_contract": {
            "planned_max_sequence_length": 8,
            "state_feature_names": STATE_FEATURES,
            "numeric_state_features": NUMERIC,
            "categorical_state_features": CATEGORICAL,
        },
        "sequence_encoder": {
            "architecture": "causal_masked_transformer_encoder",
            "d_model": 128,
            "n_heads": 4,
            "causal_mask_required": True,
        },
        "output_heads": HEADS,
    }
    architecture_contract_json = root / "entry_exit_transformer_architecture_contract.json"
    architecture_contract_json.write_text(json.dumps(architecture_contract, indent=2) + "\n", encoding="utf-8")

    architecture = {
        "decision": (
            "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW"
            if architecture_ready
            else "BLOCKED_BY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS"
        ),
        "dataset_rows": 3,
        "episode_count": 3,
        "model_dataset_json": str(model_dataset_json),
        "architecture_contract": architecture_contract,
        "architecture_contract_json": str(architecture_contract_json),
    }
    architecture_json = root / "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_latest.json"
    architecture_json.write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    return architecture_json


def _args(tmp_path: Path, architecture_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        architecture_json=str(architecture_json),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_training_plan_readiness_passes_contract(tmp_path: Path) -> None:
    architecture_json = _write_architecture_bundle(tmp_path)

    report = run(_args(tmp_path, architecture_json))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
    assert report["plan_review"]["ready"] is True
    assert report["hash_review"]["ready"] is True
    assert report["training_plan"]["future_training_command_contract"]["vedtak_prefix_required"] == "ENTRY_EXIT_TRANSFORMER_TRAIN_"
    assert report["training_plan"]["future_training_command_contract"]["requires_ram_guard"] is True
    assert report["training_plan"]["resource_guardrails"]["num_workers"] == 0
    assert report["training_plan"]["architecture"]["output_heads"] == HEADS
    assert report["training_plan"]["dataset"]["feature_schema_json"]
    assert len(report["training_plan"]["dataset"]["feature_schema_json_sha256"]) == 64
    assert report["exit_training_allowed"] is False
    assert report["exit_training_allowed_with_explicit_vedtak"] is False
    assert report["trainer_started"] is False


def test_entry_exit_transformer_training_plan_readiness_blocks_unready_architecture(tmp_path: Path) -> None:
    architecture_json = _write_architecture_bundle(tmp_path, architecture_ready=False)

    report = run(_args(tmp_path, architecture_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer architecture readiness is ready" in failed


def test_entry_exit_transformer_training_plan_readiness_blocks_shard_hash_mismatch(tmp_path: Path) -> None:
    architecture_json = _write_architecture_bundle(tmp_path, shard_hash_mismatch=True)

    report = run(_args(tmp_path, architecture_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "architecture, model dataset and shard hashes are pinned" in failed
    assert report["hash_review"]["model_dataset_shard_mismatches"]["train"]["expected"] == "0" * 64
