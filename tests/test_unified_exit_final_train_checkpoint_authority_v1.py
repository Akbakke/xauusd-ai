from __future__ import annotations

import pytest

from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
    canonical_sha256,
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
