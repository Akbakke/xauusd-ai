from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _contract(*, nonce: str = "a") -> dict[str, object]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "authority": {
            "candidate_training": True,
            "bundle": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "nonce": nonce,
    }


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.arange(12, dtype=torch.float32).reshape(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _state(
    session: trainer._CandidateTrainingSession,
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: trainer._WeightEma,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, object]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "session_contract_sha256": session.contract_sha256,
        "checkpoint_index": 1,
        "phase": "train",
        "epoch_index": 0,
        "next_batch_offset": 17,
        "epoch_order": torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.int64),
        "model_state": model.state_dict(),
        "target_model_state": target_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "weight_ema_state": ema.checkpoint_state(),
        "lr_scheduler_state": scheduler.state_dict(),
        "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
        "training_progress": {
            "joint_task_supervision": {"entry_action_q": True},
            "joint_task_gradients": {"entry_action_q": True},
            "checkpoint_selection": {"best_epoch": 0},
        },
        "complete": False,
    }


def test_candidate_session_round_trips_exact_torch_state(tmp_path: Path) -> None:
    out_bundle = tmp_path / "BUNDLE_20260828T140000Z"
    session = trainer._CandidateTrainingSession(
        out_bundle_dir=out_bundle,
        contract=_contract(),
    )
    model = torch.nn.Linear(3, 2)
    target_model = copy.deepcopy(model)
    target_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=0.0
    )
    ema = trainer._WeightEma(model, 0.5)
    _step(model, optimizer)
    ema.update(model)
    session.save_checkpoint(_state(session, model, target_model, optimizer, ema, scheduler))

    restored = session.load_checkpoint()
    assert restored is not None
    assert restored["phase"] == "train"
    assert restored["epoch_index"] == 0
    assert restored["next_batch_offset"] == 17
    assert torch.equal(
        restored["epoch_order"],
        torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.int64),
    )
    restored_model = torch.nn.Linear(3, 2)
    restored_target = copy.deepcopy(restored_model)
    restored_target.requires_grad_(False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=30, eta_min=0.0
    )
    restored_ema = trainer._WeightEma(restored_model, 0.5)
    progress = trainer._restore_candidate_training_checkpoint(
        restored,
        session=session,
        model=restored_model,
        target_model=restored_target,
        optimizer=restored_optimizer,
        weight_ema=restored_ema,
        lr_scheduler=restored_scheduler,
        device=torch.device("cpu"),
        dataset_rows=6,
    )
    assert progress["phase"] == "train"
    assert progress["next_batch_offset"] == 17
    assert progress["training_progress"]["checkpoint_selection"] == {
        "best_epoch": 0
    }
    for expected, observed in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, observed)
    for expected, observed in zip(target_model.parameters(), restored_target.parameters()):
        assert torch.equal(expected, observed)
    assert restored_ema.steps == ema.steps
    assert not out_bundle.exists()
    assert session.directory.is_dir()
    assert (
        session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    ).is_file()


def test_candidate_session_refuses_contract_and_state_tampering(tmp_path: Path) -> None:
    out_bundle = tmp_path / "BUNDLE_20260828T140000Z"
    session = trainer._CandidateTrainingSession(
        out_bundle_dir=out_bundle,
        contract=_contract(),
    )
    model = torch.nn.Linear(3, 2)
    target_model = copy.deepcopy(model)
    target_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=0.0
    )
    ema = trainer._WeightEma(model, 0.5)
    _step(model, optimizer)
    session.save_checkpoint(_state(session, model, target_model, optimizer, ema, scheduler))

    with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
        trainer._CandidateTrainingSession(
            out_bundle_dir=out_bundle,
            contract=_contract(nonce="changed"),
        )

    active = trainer._candidate_training_session_read_json(
        session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME,
        label="ACTIVE",
    )
    slot_path = session.directory / trainer._CANDIDATE_TRAINING_STATE_FILENAMES[
        int(active["slot"])
    ]
    slot_path.write_bytes(slot_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="STATE_SHA256_MISMATCH"):
        session.load_checkpoint()


def test_candidate_validation_snapshot_uses_only_weights_only_safe_values() -> None:
    snapshot = trainer._candidate_validation_snapshot(
        total=1.5,
        entry_q_loss_sum=2.5,
        cooperation_gate_epoch={"gate": {"rows": 1, "sum": torch.ones(2).numpy()}},
        feature_tf_gate_epoch={"rows": 1, "sum": torch.ones((2, 2)).numpy()},
        exit_cooperation_gate_epoch={"gate": {"rows": 1, "sum": torch.ones(2).numpy()}},
        exit_feature_tf_gate_epoch={"rows": 1, "sum": torch.ones((2, 2)).numpy()},
        rows=1,
        side_mae_loss_sum=0.1,
        trendline_event_loss_sum=0.2,
        trendline_event_rows_sum=1,
        trendline_support_rows_sum=1,
        trendline_resistance_rows_sum=0,
        unified_exit_loss_sum=0.3,
        unified_exit_population_rows=2,
        unified_exit_rows=2,
        unified_exit_tied_rows=0,
        unified_exit_eligible_entry_rows=1,
        unified_exit_hold_rows=1,
        unified_exit_now_rows=1,
        unified_exit_correct=1,
        active_head_epoch={"heads": {"entry_action_q": {"components": {}}}},
        entry_policy_realized_pnl_chunks=[torch.tensor([1.0]).numpy()],
        entry_unique_target_rows=1,
        entry_target_equivalent_rows=0,
        entry_unique_target_agreement_rows=1,
        full_trajectory_accumulator={
            "state_stream_chain_sha256": "0" * 64,
            "learned_realized": [1.0],
        },
    )
    assert isinstance(snapshot["cooperation_gate_epoch"]["gate"]["sum"], torch.Tensor)
    restored = trainer._restore_candidate_validation_snapshot(snapshot)
    assert restored["rows"] == 1
    assert restored["cooperation_gate_epoch"]["gate"]["sum"].shape == (2,)
    assert restored["full_trajectory_accumulator"]["state_stream_chain_sha256"] == "0" * 64
