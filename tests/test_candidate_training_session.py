from __future__ import annotations

import copy
import hashlib
import json
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


def _recipe_source_provenance(*, source_commit: str) -> dict[str, object]:
    bindings = {
        "trainer": {
            "path": "/repo/gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mtime_ns": 1,
            "device": 1,
            "inode": 1,
        }
    }
    encoded = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": "gx1_entry_training_recipe_source_provenance_v1",
        "recipe_audit_path": "/repo/recipe.json",
        "recipe_audit_sha256": "b" * 64,
        "source_commit": source_commit,
        "source_bindings": bindings,
        "source_bindings_sha256": hashlib.sha256(encoded).hexdigest(),
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
        "global_optimizer_steps": 17,
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
    assert restored["global_optimizer_steps"] == 17
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


def test_candidate_session_legacy_resume_requires_recipe_source_closure(
    tmp_path: Path,
) -> None:
    out_bundle = tmp_path / "BUNDLE_20260830T081406Z"
    source_commit = "1" * 40
    legacy_contract = _contract()
    legacy_contract["source_commit"] = source_commit
    legacy = trainer._CandidateTrainingSession(
        out_bundle_dir=out_bundle,
        contract=legacy_contract,
    )

    requested_contract = dict(legacy_contract)
    requested_contract["recipe_source_provenance"] = _recipe_source_provenance(
        source_commit=source_commit
    )
    resumed = trainer._CandidateTrainingSession(
        out_bundle_dir=out_bundle,
        contract=requested_contract,
    )
    assert resumed.contract_sha256 == legacy.contract_sha256

    wrong_source_contract = dict(requested_contract)
    wrong_source_contract["recipe_source_provenance"] = _recipe_source_provenance(
        source_commit="2" * 40
    )
    with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
        trainer._CandidateTrainingSession(
            out_bundle_dir=out_bundle,
            contract=wrong_source_contract,
        )


def test_candidate_session_refuses_state_without_active_pointer(tmp_path: Path) -> None:
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
    (session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="ACTIVE_POINTER_MISSING_WITH_STATE"):
        session.load_checkpoint()


def test_candidate_session_rejects_missing_optimizer_or_early_stop_state(
    tmp_path: Path,
) -> None:
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
    state = _state(session, model, target_model, optimizer, ema, scheduler)

    missing_optimizer = dict(state)
    missing_optimizer.pop("optimizer_state")
    with pytest.raises(RuntimeError, match="STATE_SCHEMA_INVALID"):
        session.save_checkpoint(missing_optimizer)

    missing_early_stop = dict(state)
    progress = dict(missing_early_stop["training_progress"])
    selection = dict(progress["checkpoint_selection"])
    selection.pop("epochs_since_improve")
    progress["checkpoint_selection"] = selection
    missing_early_stop["training_progress"] = progress
    with pytest.raises(RuntimeError, match="SELECTION_STATE_INVALID"):
        session.save_checkpoint(missing_early_stop)


def test_candidate_session_keeps_hash_bound_top_k_and_rejects_tampering(
    tmp_path: Path,
) -> None:
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
    state = _state(session, model, target_model, optimizer, ema, scheduler)
    selection = state["training_progress"]["checkpoint_selection"]
    records = []
    for epoch, metric in ((1, 1.0),):
        record = session.save_top_k_checkpoint(
            epoch=epoch,
            metric=metric,
            model_state=model.state_dict(),
            target_model_state=target_model.state_dict(),
        )
        records.append(record)
    selection["top_k_checkpoints"] = trainer.retain_top_k(records, top_k=1)
    selection["best_checkpoint"] = selection["top_k_checkpoints"][0]
    session.save_checkpoint(state)

    restored = session.load_checkpoint()
    assert restored is not None
    restored_selection = restored["training_progress"]["checkpoint_selection"]
    assert [row["epoch"] for row in restored_selection["top_k_checkpoints"]] == [1]
    assert restored_selection["best_checkpoint"]["epoch"] == 1

    selected_path = session.directory / restored_selection["best_checkpoint"]["path"]
    selected_path.write_bytes(selected_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="TOP_K_SHA256_MISMATCH"):
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
    # The candidate coordinator validates a resumed snapshot before passing it
    # to ``validate``, which performs the same restoration at its own entry.
    # Restoring an already-restored snapshot must therefore be safe and retain
    # the accumulated ndarray evidence.
    restored_again = trainer._restore_candidate_validation_snapshot(restored)
    assert restored_again["cooperation_gate_epoch"]["gate"]["sum"].shape == (2,)
    assert restored_again["entry_policy_realized_pnl_chunks"][0].shape == (1,)


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

    def _run(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip_norm: float = 1.0,
        weight_decay: float = 1e-5,
    ):
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
                # Candidate selection is intentionally terminal after its one
                # full TRAIN epoch and one complete VAL pass.
                epochs=1,
                early_stopping_patience=5,
                early_stopping_min_delta=0.0,
                minimum_epochs_before_stop=1,
                save_top_k=1,
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
            grad_clip_norm=grad_clip_norm,
            weight_decay=weight_decay,
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
            recipe_source_provenance=_recipe_source_provenance(
                source_commit="1" * 40
            ),
        )

    first_model = torch.nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="test-interrupt-after-checkpoint"):
        _run(first_model, torch.optim.AdamW(first_model.parameters(), lr=0.001))
    assert calls == {"train": 1, "validation": 0}
    assert train_offsets == [0]

    changed_hyperparameter_model = torch.nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="SESSION_CONTRACT_MISMATCH"):
        _run(
            changed_hyperparameter_model,
            torch.optim.AdamW(changed_hyperparameter_model.parameters(), lr=0.001),
            grad_clip_norm=0.5,
        )

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
