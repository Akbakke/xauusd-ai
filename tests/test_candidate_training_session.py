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
            **trainer._new_candidate_training_progress(),
            "joint_task_supervision_observed": {
                name: True for name in trainer.JOINT_TASK_NAMES
            },
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
    assert progress["training_progress"]["checkpoint_selection"]["best_epoch"] == -1
    assert all(
        progress["training_progress"]["joint_task_supervision_observed"].values()
    )
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


def test_candidate_runner_resumes_completed_hash_bound_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate coordinator must not retrain after a completed resume."""

    class _Rows(torch.utils.data.Dataset):
        def __init__(self, rows: int) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return self.rows

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(index, dtype=torch.int64)

    calls = {"train": 0, "validation": 0}
    train_offsets: list[int] = []

    def _fake_train_epoch(*args, **kwargs):
        calls["train"] += 1
        loader = args[2]
        checkpoint = kwargs["session_checkpoint_hook"]
        train_offsets.append(int(kwargs["session_batch_offset"]))
        if calls["train"] == 1:
            checkpoint(
                next_batch_offset=kwargs["session_batch_offset"] + 1,
                complete_epoch=False,
            )
            raise RuntimeError("test-interrupt-after-checkpoint")
        checkpoint(
            next_batch_offset=kwargs["session_batch_offset"] + len(loader),
            complete_epoch=True,
        )
        return 0.0, {}, True

    def _fake_validate(*args, **kwargs):
        calls["validation"] += 1
        return (
            1.0,
            float("nan"),
            0.75,
            float("nan"),
            {
                "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean": 2.0,
                "active_head_health_ok": True,
                "cooperation_gate_health_ok": True,
                "exit_cooperation_gate_health_ok": True,
                "unified_exit_full_trajectory_validation": {
                    "schema_version": "test",
                    "decision": "PASS",
                },
            },
        )

    monkeypatch.setattr(trainer, "train_epoch", _fake_train_epoch)
    monkeypatch.setattr(trainer, "validate", _fake_validate)

    artifacts = {}
    for name in ("train", "val", "m5", "lifecycle"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = path
    out_bundle = tmp_path / "CANDIDATE_20260828T140000Z"
    lifecycle = {
        "splits": {"train": {"lifecycle_manifest_sha256": "a" * 64}},
        "root_manifest_sha256": "b" * 64,
    }

    def _run(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        return trainer._run_resumable_candidate_training(
            model=model,
            optimizer=optimizer,
            weight_ema=None,
            lr_scheduler=None,
            device=torch.device("cpu"),
            train_ds=_Rows(4),
            val_ds=_Rows(2),
            effective_train_rows=4,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            epochs=1,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            out_bundle_dir=out_bundle,
            gx1_data_override="",
            run_id="V46_20260825T170935Z_CANDIDATE",
            dataset_run_id="V46_20260825T170935Z",
            train_parquet=artifacts["train"],
            val_parquet=artifacts["val"],
            m5_prebuilt_path=artifacts["m5"],
            unified_exit_lifecycle_manifest_path=artifacts["lifecycle"],
            input_normalization={"contract_sha256": "c" * 64},
            seed=1337,
            grad_accum_steps=1,
            lr=0.001,
            dropout=0.0,
            seq_len=96,
            per_tf_seq_lens={
                "M5": 16,
                "M15": 64,
                "H1": 96,
                "H4": 96,
                "D1": 252,
            },
            multi_tf_num_layers=2,
            specialist_num_layers=1,
            multi_tf_scale=0.5,
            specialist_fusion_scale=0.25,
            cross_family_fusion_scale=0.25,
            unified_exit_lifecycle_evidence=lifecycle,
        )

    first_model = torch.nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="test-interrupt-after-checkpoint"):
        _run(first_model, torch.optim.AdamW(first_model.parameters(), lr=0.001))
    assert calls == {"train": 1, "validation": 0}
    assert train_offsets == [0]

    second_model = torch.nn.Linear(3, 2)
    second = _run(second_model, torch.optim.AdamW(second_model.parameters(), lr=0.001))
    assert calls == {"train": 2, "validation": 1}
    assert train_offsets == [0, 1]
    assert second["best_epoch"] == 1
    assert second["best_policy_pnl"] == 2.0
    assert not out_bundle.exists()

    third_model = torch.nn.Linear(3, 2)
    third = _run(third_model, torch.optim.AdamW(third_model.parameters(), lr=0.001))
    assert calls == {"train": 2, "validation": 1}
    assert third["best_epoch"] == 1
