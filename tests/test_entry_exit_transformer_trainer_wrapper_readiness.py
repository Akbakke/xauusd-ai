import argparse
import json
import os
from pathlib import Path

from gx1.scripts.audit_entry_exit_transformer_trainer_wrapper_readiness_v1 import run


def _write_training_plan(tmp_path: Path, *, ready: bool = True) -> Path:
    path = tmp_path / "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"
    path.write_text(
        json.dumps(
            {
                "decision": (
                    "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
                    if ready
                    else "BLOCKED_BY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS"
                ),
                "exit_training_allowed": False,
                "exit_training_allowed_with_explicit_vedtak": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_wrapper(tmp_path: Path, *, complete: bool = True) -> Path:
    missing_disabled_flag = "" if complete else "# missing disabled flag in this fixture\n"
    disabled_flag = "TRAINER_IMPLEMENTATION_ENABLED=0\n" if complete else missing_disabled_flag
    path = tmp_path / "run_entry_exit_transformer_train.sh"
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
VEDTAK_PREFIX=ENTRY_EXIT_TRANSFORMER_TRAIN_
READY_DECISION=ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW
TRAIN_EXECUTION_REVIEW_JSON=/tmp/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json
TRAIN_EXECUTION_REVIEW_READY=ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE
POST_TRAIN_AUDIT_CONTRACT_JSON=/tmp/ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.json
POST_TRAIN_AUDIT_CONTRACT_READY=ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY
FEATURE_ALIGNMENT_JSON=/tmp/ENTRY_EXIT_FEATURE_ALIGNMENT_latest.json
FEATURE_ALIGNMENT_READY=ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW
NUM_WORKERS=0
{disabled_flag}
# scripts/gx1_capped_run.sh --num-workers
# This wrapper does not train, replay, distill, promote, shadow or touch live.
VEDTAK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vedtak) VEDTAK="$2"; shift 2 ;;
    --dry-run) shift ;;
    *) shift ;;
  esac
done
if [[ -z "$VEDTAK" ]]; then
  echo "FATAL: --vedtak is required for active Exit Transformer train." >&2
  exit 2
fi
if [[ "$VEDTAK" != "$VEDTAK_PREFIX"* ]]; then
  echo "FATAL: --vedtak must start with $VEDTAK_PREFIX" >&2
  exit 2
fi
echo "FATAL: active Exit Transformer trainer implementation is not enabled." >&2
exit 2
""",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
    return path


def _args(tmp_path: Path, training_plan: Path, wrapper: Path) -> argparse.Namespace:
    return argparse.Namespace(
        training_plan_json=str(training_plan),
        wrapper_path=str(wrapper),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_trainer_wrapper_readiness_passes_fail_closed_contract(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)
    wrapper = _write_wrapper(tmp_path)

    report = run(_args(tmp_path, training_plan, wrapper))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
    assert report["source_review"]["ready"] is True
    assert report["wrapper_rejection_cases"]["missing_vedtak"]["returncode"] == 2
    assert report["wrapper_rejection_cases"]["bad_vedtak_prefix"]["returncode"] == 2
    assert report["exit_training_allowed"] is False
    assert report["exit_training_allowed_with_explicit_vedtak"] is False
    assert report["trainer_started"] is False


def test_entry_exit_transformer_trainer_wrapper_readiness_blocks_unready_plan(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path, ready=False)
    wrapper = _write_wrapper(tmp_path)

    report = run(_args(tmp_path, training_plan, wrapper))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer training plan readiness is ready" in failed


def test_entry_exit_transformer_trainer_wrapper_readiness_blocks_incomplete_wrapper(tmp_path: Path) -> None:
    training_plan = _write_training_plan(tmp_path)
    wrapper = _write_wrapper(tmp_path, complete=False)

    report = run(_args(tmp_path, training_plan, wrapper))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer train wrapper is executable and fail-closed in source" in failed
    assert "trainer_disabled_flag" in report["source_review"]["missing_tokens"]
