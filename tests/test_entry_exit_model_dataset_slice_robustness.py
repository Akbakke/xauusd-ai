import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_model_dataset_slice_robustness_v1 import run


REGIME_CONTEXT_FIELD = "entry_ctx_d1_regime_class_id_v2"
NUMERIC = ["running_pnl_bps", "running_mfe_bps", "bars_held", "atr_bps", REGIME_CONTEXT_FIELD]
CATEGORICAL = ["session", "vol_regime", "side"]
STATE_FEATURES = [*NUMERIC, *CATEGORICAL]
REWARDS = [
    "hold_reward_bps",
    "forced_terminal_hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "terminal_reward_realized_net_pnl_bps",
    "exit_now_mfe_capture_ratio_reward",
    "exit_now_mae_penalty_reward_bps",
    "exit_now_giveback_penalty_reward_bps",
    "exit_now_transparent_combined_reward_bps",
    "future_max_running_pnl_bps",
    "future_min_running_pnl_bps",
    "future_best_exit_lift_bps",
    "future_adverse_excursion_bps",
    "future_giveback_from_peak_bps",
    "exit_hazard_adverse_15bps_label",
    "exit_hazard_giveback_20bps_label",
    "positive_mfe_stopout_episode_label",
    "oracle_exit_before_giveback_label",
]


def _rows(
    split: str,
    *,
    unsupported_long: bool = False,
    constant_nontrain_regime_context: bool = False,
    constant_train_regime_context: bool = False,
) -> list[dict]:
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
                    REGIME_CONTEXT_FIELD: (
                        4.0
                        if (constant_nontrain_regime_context and split in {"val", "test"})
                        or (constant_train_regime_context and split == "train")
                        else float(step + idx + episode)
                    ),
                }
                for reward in REWARDS:
                    row[reward] = float(step - idx)
                row["exit_hazard_adverse_15bps_label"] = int(step == 1)
                row["exit_hazard_giveback_20bps_label"] = int(step == 2)
                row["positive_mfe_stopout_episode_label"] = int(idx == 1)
                row["oracle_exit_before_giveback_label"] = int(step == 0 and idx == 1)
                rows.append(row)
    return rows


def _write_inputs(
    tmp_path: Path,
    *,
    unsupported_long: bool = False,
    constant_nontrain_regime_context: bool = False,
    constant_train_regime_context: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "data"
    root.mkdir()
    shards = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.csv"
        pd.DataFrame(
            _rows(
                split,
                unsupported_long=unsupported_long,
                constant_nontrain_regime_context=constant_nontrain_regime_context,
                constant_train_regime_context=constant_train_regime_context,
            )
        ).to_csv(path, index=False)
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


def test_entry_exit_model_dataset_slice_robustness_discloses_finite_nontrain_constant_context(
    tmp_path: Path,
) -> None:
    model_dataset_json, pretrain_manifest_json = _write_inputs(
        tmp_path,
        constant_nontrain_regime_context=True,
    )

    report = run(_args(tmp_path, model_dataset_json, pretrain_manifest_json))

    assert report["decision"] == "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
    assert report["feature_liveness_review"]["weak_numeric_feature_count"] == 2
    weak = report["feature_liveness_review"]["weak_numeric_features"]
    assert {(row["split"], row["field"]) for row in weak} == {
        ("val", REGIME_CONTEXT_FIELD),
        ("test", REGIME_CONTEXT_FIELD),
    }
    assert report["split_reviews"]["val"]["feature_liveness"]["all_numeric_finite_and_live"] is False
    assert report["split_reviews"]["val"]["feature_liveness"]["all_numeric_ready"] is True
    assert all(row["ready"] for row in report["split_reviews"].values())


def test_entry_exit_model_dataset_slice_robustness_blocks_train_constant_context(tmp_path: Path) -> None:
    model_dataset_json, pretrain_manifest_json = _write_inputs(
        tmp_path,
        constant_train_regime_context=True,
    )

    report = run(_args(tmp_path, model_dataset_json, pretrain_manifest_json))

    assert report["decision"] == "BLOCKED_BY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS"
    assert report["feature_liveness_review"]["blocking_numeric_feature_count"] == 1
    blocking = report["feature_liveness_review"]["blocking_numeric_features"]
    assert blocking[0]["split"] == "train"
    assert blocking[0]["field"] == REGIME_CONTEXT_FIELD
    assert report["split_reviews"]["train"]["feature_liveness"]["all_numeric_ready"] is False
