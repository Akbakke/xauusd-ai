from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    build_checkpoint,
    file_sha256,
    save_checkpoint_atomic,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    bootstrap_random_access_v2_from_pretrained,
)
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    load_selected_weight_ema_checkpoint_readonly_v1,
)
from tests.test_entry_v10_ctx_model_shapes import _make_model


def _old(model):
    return {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not (
            name == "unified_exit_random_access_architecture_sha256"
            or name.startswith("exit_random_access_summary_proj.")
            or name.startswith("exit_random_access_fuse.")
        )
    }


def _fixture(tmp_path):
    model = _make_model(dropout=0.1)
    target = copy.deepcopy(model)
    online_receipt = bootstrap_random_access_v2_from_pretrained(model, _old(model))
    target_receipt = bootstrap_random_access_v2_from_pretrained(target, _old(target))
    target.requires_grad_(False)
    parameters = dict(model.named_parameters())
    shadow = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    changed_name = next(iter(parameters))
    shadow[changed_name] = shadow[changed_name] + 0.25
    decay = 1.0 - 1.0 / 1024.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    checkpoint = build_checkpoint(
        model=model,
        target_model=target,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        weight_ema_state={
            "decay": decay,
            "steps": 1,
            "parameter_names": sorted(parameters),
            "shadow": shadow,
        },
        global_step=1,
        next_batch_offset=1,
        epoch_index=0,
        batch_size=16,
        epoch_schedule_sha256="6" * 64,
        launch_manifest_sha256="1" * 64,
        selected_sampler_artifact_sha256="2" * 64,
        bootstrap_source_receipt_sha256="3" * 64,
        bootstrap_model_receipts={
            "online": online_receipt,
            "target": target_receipt,
        },
        base_normalization_sha256="4" * 64,
        summary_normalization_sha256="5" * 64,
    )
    save_checkpoint_atomic(checkpoint, directory=tmp_path)
    return model, changed_name, shadow, decay


def _load(tmp_path, model, decay):
    pointer = tmp_path / "RESUME_POINTER.json"
    return load_selected_weight_ema_checkpoint_readonly_v1(
        pointer_path=pointer,
        model=model,
        expected_checkpoint_pointer_file_sha256=file_sha256(pointer),
        expected_launch_manifest_sha256="1" * 64,
        expected_selected_sampler_artifact_sha256="2" * 64,
        expected_bootstrap_source_receipt_sha256="3" * 64,
        expected_base_normalization_sha256="4" * 64,
        expected_summary_normalization_sha256="5" * 64,
        expected_batch_size=16,
        expected_epoch_schedule_sha256="6" * 64,
        expected_weight_ema_decay=decay,
    )


def test_readonly_loader_selects_ema_parameters_and_online_buffers(tmp_path) -> None:
    source, changed_name, shadow, decay = _fixture(tmp_path)
    selected = _make_model(dropout=0.1)
    rng_before = torch.get_rng_state().clone()
    binding = _load(tmp_path, selected, decay)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert binding["model_variant"] == "weight_ema"
    assert binding["rng_mutated"] is False
    assert binding["optimizer_or_scheduler_loaded"] is False
    assert torch.equal(selected.state_dict()[changed_name], shadow[changed_name])
    parameter_names = set(dict(source.named_parameters()))
    for name, value in selected.state_dict().items():
        if name not in parameter_names:
            assert torch.equal(value, source.state_dict()[name])


def test_readonly_loader_rejects_wrong_decay_and_pointer_hash(tmp_path) -> None:
    _source, _changed_name, _shadow, decay = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="POINTER"):
        load_selected_weight_ema_checkpoint_readonly_v1(
            pointer_path=tmp_path / "RESUME_POINTER.json",
            model=_make_model(dropout=0.1),
            expected_checkpoint_pointer_file_sha256="f" * 64,
            expected_launch_manifest_sha256="1" * 64,
            expected_selected_sampler_artifact_sha256="2" * 64,
            expected_bootstrap_source_receipt_sha256="3" * 64,
            expected_base_normalization_sha256="4" * 64,
            expected_summary_normalization_sha256="5" * 64,
            expected_batch_size=16,
            expected_epoch_schedule_sha256="6" * 64,
            expected_weight_ema_decay=decay,
        )
    with pytest.raises(RuntimeError, match="EMA"):
        _load(tmp_path, _make_model(dropout=0.1), decay - 0.1)
