from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


class _CompletedSession:
    def __init__(self, *, directory: Path, state: dict[str, object]) -> None:
        self.directory = directory
        self.contract_sha256 = "a" * 64
        self._state = state

    def load_checkpoint(self) -> dict[str, object]:
        return self._state


def _terminal_state() -> dict[str, object]:
    model_state = {"weight": torch.ones(1)}
    checkpoint = {
        "epoch": 1,
        "metric": 1.0,
        "path": "top_k/epoch_0001.pt",
        "sha256": "b" * 64,
    }
    return {
        "complete": True,
        "phase": "validation",
        "epoch_index": 0,
        "training_progress": {
            "checkpoint_selection": {
                "last_epoch": 1,
                "best_epoch": 1,
                "best_checkpoint": checkpoint,
                "top_k_checkpoints": [checkpoint],
                "best_state": model_state,
                "best_fitted_q_target_state": model_state,
            },
            "joint_task_supervision_observed": {"joint": True},
            "joint_task_gradient_observed": {"joint": True},
            "validation_snapshot": None,
        },
    }


def test_completed_candidate_epoch_seal_admits_terminal_val_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_bundle = tmp_path / "BUNDLE"
    session_dir = tmp_path / ".gx1-candidate-training-session.BUNDLE"
    session_dir.mkdir()
    (session_dir / "CANDIDATE_TRAINING_SESSION_CONTRACT.json").write_text(
        json.dumps(
            {
                "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                "out_bundle_dir": str(out_bundle),
                "profile": "candidate",
                "execution_tier": "canonical",
                "authority": {"candidate_training": True, "bundle": False},
            }
        )
    )
    state = _terminal_state()
    session = _CompletedSession(directory=session_dir, state=state)
    monkeypatch.setattr(trainer, "_CandidateTrainingSession", lambda **_kwargs: session)
    monkeypatch.setattr(
        trainer,
        "_require_candidate_training_progress",
        lambda progress: progress,
    )
    monkeypatch.setattr(
        trainer.torch,
        "load",
        lambda *_args, **_kwargs: {
            "session_contract_sha256": "a" * 64,
            "epoch": 1,
            "metric": 1.0,
            "model_state": {"weight": torch.ones(1)},
            "target_model_state": {"weight": torch.ones(1)},
        },
    )

    result = trainer.load_completed_candidate_epoch_for_seal(
        out_bundle_dir=out_bundle,
        completed_epoch=1,
    )

    assert result["best_epoch"] == 1
    assert result["session_directory"] == str(session_dir)


def test_completed_candidate_epoch_seal_rejects_nonterminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_bundle = tmp_path / "BUNDLE"
    session_dir = tmp_path / ".gx1-candidate-training-session.BUNDLE"
    session_dir.mkdir()
    (session_dir / "CANDIDATE_TRAINING_SESSION_CONTRACT.json").write_text(
        json.dumps(
            {
                "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                "out_bundle_dir": str(out_bundle),
                "profile": "candidate",
                "execution_tier": "canonical",
                "authority": {"candidate_training": True, "bundle": False},
            }
        )
    )
    state = _terminal_state()
    state["complete"] = False
    session = _CompletedSession(directory=session_dir, state=state)
    monkeypatch.setattr(trainer, "_CandidateTrainingSession", lambda **_kwargs: session)

    with pytest.raises(RuntimeError, match="CANDIDATE_EPOCH_SEAL_STATE_INVALID"):
        trainer.load_completed_candidate_epoch_for_seal(
            out_bundle_dir=out_bundle,
            completed_epoch=1,
        )
