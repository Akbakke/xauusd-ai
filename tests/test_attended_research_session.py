from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _contract(*, nonce: str = "a") -> dict[str, object]:
    return {
        "schema_version": trainer._ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION,
        "authority": {
            "research_trainability_only": True,
            "candidate": False,
            "validation": False,
            "test": False,
            "bundle": False,
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


def test_attended_session_restores_exact_step_boundary_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_bundle = tmp_path / "BUNDLE_20260824T120000Z"
    session = trainer._AttendedResearchSession(
        out_bundle_dir=out_bundle,
        contract=_contract(),
    )
    model = torch.nn.Linear(3, 2)
    target_model = copy.deepcopy(model)
    target_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1, eta_min=0.0
    )
    ema = trainer._WeightEma(model, 0.5)
    _step(model, optimizer)
    ema.update(model)
    epoch_order = torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.int64)

    session.save_checkpoint(
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        weight_ema=ema,
        lr_scheduler=scheduler,
        device=torch.device("cpu"),
        checkpoint_index=1,
        complete_optimizer_steps=1,
        epoch_index=0,
        next_batch_offset=1,
        epoch_order=epoch_order,
        complete=False,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    original_load = trainer.torch.load
    observed_map_locations: list[object] = []

    def _cpu_staged_load(*args, **kwargs):
        observed_map_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(trainer.torch, "load", _cpu_staged_load)
    restored_state = session.load_checkpoint()
    assert restored_state is not None
    assert observed_map_locations == ["cpu"]
    restored_model = torch.nn.Linear(3, 2)
    restored_target = copy.deepcopy(restored_model)
    restored_target.requires_grad_(False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=1, eta_min=0.0
    )
    restored_ema = trainer._WeightEma(restored_model, 0.5)

    progress = trainer._restore_attended_research_checkpoint(
        restored_state,
        session=session,
        model=restored_model,
        target_model=restored_target,
        optimizer=restored_optimizer,
        weight_ema=restored_ema,
        lr_scheduler=restored_scheduler,
        device=torch.device("cpu"),
        dataset_rows=6,
    )

    assert progress["checkpoint_index"] == 1
    assert progress["complete_optimizer_steps"] == 1
    assert progress["next_batch_offset"] == 1
    assert torch.equal(progress["epoch_order"], epoch_order)
    for expected, observed in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, observed)
    for expected, observed in zip(target_model.parameters(), restored_target.parameters()):
        assert torch.equal(expected, observed)
    assert restored_ema.steps == ema.steps
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert not out_bundle.exists()
    assert session.directory.is_dir()
    assert (session.directory / trainer._ATTENDED_RESEARCH_ACTIVE_FILENAME).is_file()


def test_attended_session_refuses_contract_or_state_tampering(tmp_path: Path) -> None:
    out_bundle = tmp_path / "BUNDLE_20260824T120000Z"
    session = trainer._AttendedResearchSession(
        out_bundle_dir=out_bundle,
        contract=_contract(),
    )
    model = torch.nn.Linear(3, 2)
    target_model = copy.deepcopy(model)
    target_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    _step(model, optimizer)
    session.save_checkpoint(
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        weight_ema=None,
        lr_scheduler=None,
        device=torch.device("cpu"),
        checkpoint_index=1,
        complete_optimizer_steps=1,
        epoch_index=0,
        next_batch_offset=1,
        epoch_order=torch.arange(6, dtype=torch.int64),
        complete=False,
    )
    with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
        trainer._AttendedResearchSession(
            out_bundle_dir=out_bundle,
            contract=_contract(nonce="changed"),
        )

    active = trainer._attended_session_read_json(
        session.directory / trainer._ATTENDED_RESEARCH_ACTIVE_FILENAME,
        label="ACTIVE",
    )
    slot_path = session.directory / trainer._ATTENDED_RESEARCH_STATE_FILENAMES[
        int(active["slot"])
    ]
    slot_path.write_bytes(slot_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="STATE_SHA256_MISMATCH"):
        session.load_checkpoint()


def test_exact_index_sampler_starts_at_complete_batch_boundary() -> None:
    sampler = trainer._ExactIndexSampler(
        torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.int64),
        batch_offset=1,
        batch_size=2,
    )
    assert list(sampler) == [3, 2, 1, 0]
    assert len(sampler) == 4


def test_time_window_completion_report_requires_all_task_paths_and_movement(
    tmp_path: Path,
) -> None:
    out_bundle = tmp_path / "BUNDLE_20260824T120000Z"
    session = trainer._AttendedResearchSession(
        out_bundle_dir=out_bundle,
        contract=_contract(),
    )
    target_model = torch.nn.Linear(3, 2)
    model = copy.deepcopy(target_model)
    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    all_tasks = {name: True for name in trainer.JOINT_TASK_NAMES}
    report = trainer._write_pre_candidate_time_window_integration_report(
        session=session,
        train_time_window={
            "start_utc": "2024-12-01T00:00:00+00:00",
            "end_utc": "2025-06-01T00:00:00+00:00",
            "selected_rows": 32,
            "selection_sha256": "a" * 64,
            "first_selected_time_utc": "2024-12-01T00:05:00+00:00",
            "last_selected_time_utc": "2025-05-30T12:55:00+00:00",
        },
        model=model,
        target_model=target_model,
        task_supervision_observed=all_tasks,
        task_gradient_observed=all_tasks,
        train_stats={"unified_exit_q_valid_cells": 1},
        effective_train_rows=32,
    )
    payload = trainer._attended_session_read_json(report, label="REPORT")
    assert payload["decision"] == "PASS_TECHNICAL_INTEGRATION_NOT_EDGE"
    assert payload["authority"]["backtest"] is False
    assert payload["authority"]["candidate"] is False
    assert payload["model_state"]["different"] is True
    with pytest.raises(RuntimeError, match="REPORT_DESTINATION_INVALID"):
        trainer._write_pre_candidate_time_window_integration_report(
            session=session,
            train_time_window=payload["train_time_window"],
            model=model,
            target_model=target_model,
            task_supervision_observed=all_tasks,
            task_gradient_observed=all_tasks,
            train_stats={},
            effective_train_rows=32,
        )


def test_attended_session_source_keeps_speed_modes_forbidden() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")

    assert "_ATTENDED_RESEARCH_MAX_OPTIMIZER_STEPS = 60" in source
    assert "_ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS = 8" in source
    assert "_ATTENDED_RESEARCH_CUDA_MEMORY_FRACTION = 0.50" in source
    assert "_ATTENDED_RESEARCH_BATCH_SIZE = 8" in source
    assert "torch.cuda.set_per_process_memory_fraction(" in source
    assert "cuda_index = torch.cuda.current_device()" in source
    assert 'map_location="cpu", weights_only=True' in source
    assert '"precision": "deterministic_fp32"' in source
    assert '"tf32": False' in source
    assert '"autocast": False' in source
    assert "[TRAIN_PROFILE]" in source


def test_attended_time_window_uses_full_window_train_and_bounded_val_only() -> None:
    """Prevent a six-month attended run from silently becoming a 32-row TRAIN smoke.

    The attended lane deliberately uses ``subsample_rows`` to bound VAL model
    work, while the explicit causal time window owns the TRAIN population.
    Keep the assertion close to the trainer source because this precedence is
    a launch/runtime invariant rather than a new feature surface.
    """
    source = Path(trainer.__file__).read_text(encoding="utf-8")

    time_window_start = source.index(
        "if train_time_window_start is not None and train_time_window_end is not None:"
    )
    train_window_selection = source.index(
        "train_selected_indices = np.flatnonzero(", time_window_start
    )
    uniform_train_fallback = source.index(
        "train_selected_indices = deterministic_uniform_subsample_indices(",
        train_window_selection,
    )
    val_selection = source.index(
        "val_selected_indices = deterministic_uniform_subsample_indices(",
        uniform_train_fallback,
    )

    assert time_window_start < train_window_selection < uniform_train_fallback
    assert uniform_train_fallback < val_selection
    assert "train_times >= train_time_window_start" in source[
        time_window_start:uniform_train_fallback
    ]
    assert "requested_rows=int(subsample_rows)" in source[
        val_selection : val_selection + 500
    ]
