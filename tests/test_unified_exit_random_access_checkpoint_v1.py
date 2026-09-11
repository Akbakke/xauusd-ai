from __future__ import annotations
import copy
import random

import numpy as np
import torch
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    build_checkpoint,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import (
    bootstrap_random_access_v2_from_pretrained,
)
from tests.test_entry_v10_ctx_model_shapes import _make_model


def _old(model):
    return {
        n: v.clone()
        for n, v in model.state_dict().items()
        if not (
            n == "unified_exit_random_access_architecture_sha256"
            or n.startswith("exit_random_access_summary_proj.")
            or n.startswith("exit_random_access_fuse.")
        )
    }


def test_atomic_checkpoint_restores_only_strict_v2_state(tmp_path):
    model = _make_model(dropout=0.0)
    target = copy.deepcopy(model)
    online_receipt = bootstrap_random_access_v2_from_pretrained(model, _old(model))
    target_receipt = bootstrap_random_access_v2_from_pretrained(target, _old(target))
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    ema_state = {"decay": 0.9, "steps": 1, "shadow": {"x": torch.ones(1)}}
    state = build_checkpoint(
        model=model,
        target_model=target,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        weight_ema_state=ema_state,
        global_step=7,
        next_batch_offset=1,
        epoch_index=0,
        batch_size=16,
        epoch_schedule_sha256="6" * 64,
        launch_manifest_sha256="1" * 64,
        selected_sampler_artifact_sha256="2" * 64,
        bootstrap_source_receipt_sha256="3" * 64,
        bootstrap_model_receipts={"online": online_receipt, "target": target_receipt},
        base_normalization_sha256="4" * 64,
        summary_normalization_sha256="5" * 64,
    )
    pointer = save_checkpoint_atomic(state, directory=tmp_path)
    restored = _make_model(dropout=0.0)
    restored_target = copy.deepcopy(restored)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=30
    )
    progress = load_checkpoint_strict(
        pointer_path=tmp_path / "RESUME_POINTER.json",
        model=restored,
        target_model=restored_target,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        expected_launch_manifest_sha256="1" * 64,
        expected_selected_sampler_artifact_sha256="2" * 64,
        expected_bootstrap_source_receipt_sha256="3" * 64,
        expected_base_normalization_sha256="4" * 64,
        expected_summary_normalization_sha256="5" * 64,
        expected_batch_size=16,
        expected_epoch_schedule_sha256="6" * 64,
    )
    assert progress["global_step"] == 7
    assert pointer["next_batch_offset"] == 1
    assert progress["weight_ema_state"]["steps"] == 1


def test_checkpoint_fresh_process_continuation_is_bit_exact(tmp_path):
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    model = _make_model(dropout=0.0)
    target = copy.deepcopy(model)
    online_receipt = bootstrap_random_access_v2_from_pretrained(model, _old(model))
    target_receipt = bootstrap_random_access_v2_from_pretrained(target, _old(target))
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    parameter_name, parameter = next(iter(model.named_parameters()))

    def one_step(current_model, current_optimizer):
        py_value = random.random()
        np_value = float(np.random.random())
        current_parameter = dict(current_model.named_parameters())[parameter_name]
        noise = torch.rand_like(current_parameter)
        current_optimizer.zero_grad(set_to_none=True)
        (current_parameter * noise).sum().backward()
        current_optimizer.step()
        return py_value, np_value, noise.detach().clone()

    for _ in range(3):
        one_step(model, optimizer)
    state = build_checkpoint(
        model=model,
        target_model=target,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        weight_ema_state={"decay": 0.9, "steps": 3, "shadow": {"x": torch.ones(1)}},
        global_step=3,
        next_batch_offset=3,
        epoch_index=0,
        batch_size=16,
        epoch_schedule_sha256="6" * 64,
        launch_manifest_sha256="1" * 64,
        selected_sampler_artifact_sha256="2" * 64,
        bootstrap_source_receipt_sha256="3" * 64,
        bootstrap_model_receipts={"online": online_receipt, "target": target_receipt},
        base_normalization_sha256="4" * 64,
        summary_normalization_sha256="5" * 64,
    )
    save_checkpoint_atomic(state, directory=tmp_path)
    expected_random = one_step(model, optimizer)
    expected_digest = canonical_model_state_sha256(model.state_dict())

    restored = _make_model(dropout=0.0)
    restored_target = copy.deepcopy(restored)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=30
    )
    progress = load_checkpoint_strict(
        pointer_path=tmp_path / "RESUME_POINTER.json",
        model=restored,
        target_model=restored_target,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        expected_launch_manifest_sha256="1" * 64,
        expected_selected_sampler_artifact_sha256="2" * 64,
        expected_bootstrap_source_receipt_sha256="3" * 64,
        expected_base_normalization_sha256="4" * 64,
        expected_summary_normalization_sha256="5" * 64,
        expected_batch_size=16,
        expected_epoch_schedule_sha256="6" * 64,
    )
    observed_random = one_step(restored, restored_optimizer)
    assert progress["global_step"] == 3
    assert observed_random[0] == expected_random[0]
    assert observed_random[1] == expected_random[1]
    assert torch.equal(observed_random[2], expected_random[2])
    assert canonical_model_state_sha256(restored.state_dict()) == expected_digest
