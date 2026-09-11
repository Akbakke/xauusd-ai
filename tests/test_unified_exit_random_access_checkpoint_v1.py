from __future__ import annotations
import copy
import torch
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    build_checkpoint,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
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
    state = build_checkpoint(
        model=model,
        target_model=target,
        optimizer=optimizer,
        global_step=1,
        next_batch_offset=1,
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
    progress = load_checkpoint_strict(
        pointer_path=tmp_path / "RESUME_POINTER.json",
        model=restored,
        target_model=restored_target,
        optimizer=restored_optimizer,
        expected_launch_manifest_sha256="1" * 64,
        expected_selected_sampler_artifact_sha256="2" * 64,
    )
    assert progress["global_step"] == 1
    assert pointer["next_batch_offset"] == 1
