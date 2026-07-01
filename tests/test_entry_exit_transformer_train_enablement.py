import argparse
import json
from pathlib import Path

from gx1.scripts import materialize_entry_exit_transformer_train_enablement_package_v1 as gate


READY_TRAINING_PLAN = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_WRAPPER = "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
READY_REVIEW = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
READY_POST = "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
READY_ALIGNMENT = "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "reports"
    return {
        "training_plan": _write_json(root / "training_plan.json", {"decision": READY_TRAINING_PLAN}),
        "wrapper": _write_json(root / "wrapper.json", {"decision": READY_WRAPPER}),
        "train_review": _write_json(root / "train_review.json", {"decision": READY_REVIEW, "promotion_shadow_live_allowed": False}),
        "post_contract": _write_json(root / "post_contract.json", {"decision": READY_POST, "promotion_shadow_live_allowed": False}),
        "feature_alignment": _write_json(root / "feature_alignment.json", {"decision": READY_ALIGNMENT, "promotion_shadow_live_allowed": False}),
    }


def _args(tmp_path: Path, paths: dict[str, Path], *, vedtak: str = "ENTRY_EXIT_TRANSFORMER_TRAIN_TEST_V1") -> argparse.Namespace:
    return argparse.Namespace(
        vedtak=vedtak,
        training_plan_json=str(paths["training_plan"]),
        wrapper_readiness_json=str(paths["wrapper"]),
        train_execution_review_json=str(paths["train_review"]),
        post_train_contract_json=str(paths["post_contract"]),
        feature_alignment_json=str(paths["feature_alignment"]),
        out_bundle_dir=str(tmp_path / "bundle"),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        epochs=1,
        batch_size=8,
        mem_cap="8G",
        swap_cap="1G",
        fail_on_not_ready=False,
        quiet=True,
    )


def _dry_run_stub(**kwargs) -> dict:
    return {
        "argv": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-train", "--dry-run"],
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "future_capped_train_command": (
            "Future capped train command: scripts/gx1_capped_run.sh --mem 8G --swap 1G -- "
            "python -m gx1.models.exit_sequence_transformer.train_v1 --enable-training --num-workers 0"
        ),
        "has_capped_run": True,
        "has_mem_cap": True,
        "has_swap_cap": True,
        "has_enable_training": True,
        "has_num_workers_zero": True,
        "trainer_started": False,
    }


def test_entry_exit_transformer_train_enablement_passes_with_clean_package(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION"
    assert report["exit_training_allowed"] is False
    assert report["exit_training_allowed_with_this_package"] is True
    assert report["trainer_started"] is False
    assert report["replay_started"] is False
    assert report["iql_distillation_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["wrapper_dry_run"]["has_capped_run"] is True


def test_entry_exit_transformer_train_enablement_blocks_missing_vedtak(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths, vedtak=""))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_PACKAGE"
    failed = {row["check"] for row in report["failures"]}
    assert "explicit Exit Transformer train vedtak is present" in failed


def test_entry_exit_transformer_train_enablement_blocks_dirty_git(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [" M gx1/models/exit_sequence_transformer/train_v1.py"])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_PACKAGE"
    failed = {row["check"] for row in report["failures"]}
    assert "worktree is clean before train enablement package" in failed
