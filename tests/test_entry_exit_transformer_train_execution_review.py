import argparse
import json
from pathlib import Path

from gx1.scripts.audit_entry_exit_transformer_train_execution_review_v1 import run


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path, *, unsupported_slices: bool = False, pretrain_ready: bool = True) -> dict[str, Path]:
    root = tmp_path / "reports"
    training_plan = _write_json(
        root / "training_plan.json",
        {
            "decision": "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW",
            "training_plan": {
                "future_training_command_contract": {
                    "requires_explicit_vedtak": True,
                    "vedtak_prefix_required": "ENTRY_EXIT_TRANSFORMER_TRAIN_",
                    "requires_clean_git": True,
                    "requires_ram_guard": True,
                },
                "resource_guardrails": {
                    "num_workers": 0,
                    "max_process_rss_gib": 8,
                    "abort_if_mem_available_below_gib": 8,
                },
            },
        },
    )
    wrapper = _write_json(
        root / "wrapper.json",
        {"decision": "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"},
    )
    pretrain = _write_json(
        root / "pretrain.json",
        {
            "decision": (
                "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
                if pretrain_ready
                else "BLOCKED_BY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST"
            )
        },
    )
    unsupported = [
        {
            "split": "val",
            "slice_type": "sessionxside",
            "keys": {"session": "US", "side": "LONG"},
            "episodes": 0,
            "unsupported_slice": True,
        }
    ] if unsupported_slices else []
    slice_robustness = _write_json(
        root / "slice.json",
        {
            "decision": "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE",
            "slice_review": {
                "weak_slice_count": 2,
                "unsupported_slice_count": len(unsupported),
                "weak_slices": [
                    {
                        "split": "test",
                        "slice_type": "sessionxside",
                        "keys": {"session": "US", "side": "LONG"},
                        "episodes": 2,
                        "weak_slice": True,
                    }
                ],
                "unsupported_slices": unsupported,
            },
        },
    )
    return {
        "training_plan": training_plan,
        "wrapper": wrapper,
        "pretrain": pretrain,
        "slice": slice_robustness,
    }


def _args(tmp_path: Path, paths: dict[str, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        training_plan_json=str(paths["training_plan"]),
        wrapper_readiness_json=str(paths["wrapper"]),
        pretrain_manifest_json=str(paths["pretrain"]),
        slice_robustness_json=str(paths["slice"]),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_train_execution_review_passes_but_keeps_training_closed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    report = run(_args(tmp_path, paths))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
    assert report["review_contract"]["ready"] is True
    assert report["review_contract"]["weak_slice_policy"]["weak_slice_count"] == 2
    assert report["review_contract"]["weak_slice_policy"]["unsupported_slice_count"] == 0
    assert report["exit_training_allowed"] is False
    assert report["exit_training_allowed_with_explicit_vedtak"] is False
    assert report["trainer_started"] is False


def test_entry_exit_transformer_train_execution_review_blocks_unsupported_slices(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, unsupported_slices=True)

    report = run(_args(tmp_path, paths))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW"
    failed = {row["check"] for row in report["failures"]}
    assert "train execution review accounts for weak slices and RAM guardrails" in failed


def test_entry_exit_transformer_train_execution_review_blocks_unready_pretrain(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, pretrain_ready=False)

    report = run(_args(tmp_path, paths))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer pretrain manifest is ready" in failed
