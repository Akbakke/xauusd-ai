from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
    _load_checkpoint,
    canonical_sha256,
    file_sha256,
    require_final_train_checkpoint_authority,
)


def test_minimal_self_attested_final_train_pass_is_rejected() -> None:
    value = {
        "schema_version": "gx1_unified_exit_final_train_checkpoint_authority_v1",
        "decision": "PASS_FULL_VAL_ELIGIBLE",
        "model_variant_for_val": "weight_ema",
        "selected_batch_size": 16,
        "epoch_index": 0,
        "epoch_complete": True,
        "entry_pair_count": 16384,
        "transition_count": 65536,
        "total_batches": 1024,
        "global_optimizer_steps": 1024,
        "guard_decision": "PASS",
        "signed_guard_telemetry_owner": "gx1_guarded_trainer_exec.sh",
        "test_data_used": False,
    }
    value["authority_sha256"] = canonical_sha256(value)
    with pytest.raises(RuntimeError, match="AUTHORITY_INVALID"):
        require_final_train_checkpoint_authority(value, verify_files=False)


def _checkpoint_pointer(tmp_path: Path, *, valid_model_digest: bool) -> dict[str, str]:
    online = {"weight": torch.arange(3, dtype=torch.float32)}
    target = {"weight": torch.arange(3, dtype=torch.float32)}
    state = {
        "schema_version": "gx1_unified_exit_random_access_fixed_step_checkpoint_v1",
        "global_step": 4,
        "next_batch_offset": 4,
        "epoch_index": 0,
        "batch_size": 16,
        "epoch_schedule_sha256": "1" * 64,
        "launch_manifest_sha256": "2" * 64,
        "selected_sampler_artifact_sha256": "3" * 64,
        "online_model_state": online,
        "target_model_state": target,
        "online_model_state_sha256": (
            canonical_model_state_sha256(online) if valid_model_digest else "0" * 64
        ),
        "target_model_state_sha256": canonical_model_state_sha256(target),
        "weight_ema_state": {
            "decay": 0.99,
            "steps": 4,
            "parameter_names": ["weight"],
            "shadow": {"weight": online["weight"].clone()},
        },
        "test_data_used": False,
        "old_v1_progress_reused": False,
    }
    state_path = (tmp_path / "state.pt").resolve()
    torch.save(state, state_path)
    pointer = {
        "schema_version": "gx1_unified_exit_random_access_fixed_step_pointer_v1",
        "state_path": str(state_path),
        "state_file_sha256": file_sha256(state_path),
        "global_step": 4,
        "next_batch_offset": 4,
        "epoch_index": 0,
        "batch_size": 16,
        "epoch_schedule_sha256": "1" * 64,
        "launch_manifest_sha256": "2" * 64,
        "selected_sampler_artifact_sha256": "3" * 64,
    }
    from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
        canonical_sha256 as checkpoint_sha256,
    )

    pointer["pointer_sha256"] = checkpoint_sha256(pointer)
    pointer_path = (tmp_path / "pointer.json").resolve()
    pointer_path.write_text(json.dumps(pointer))
    return {"path": str(pointer_path), "sha256": file_sha256(pointer_path)}


def test_checkpoint_authority_recomputes_model_digests(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="STATE_INVALID"):
        _load_checkpoint(_checkpoint_pointer(tmp_path, valid_model_digest=False))
