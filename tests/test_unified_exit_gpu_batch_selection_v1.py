from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    ARM_SCHEMA,
    build_selection,
    canonical_sha256,
    file_sha256,
    require_selection,
)


def _arm(tmp_path: Path, batch: int, seconds: float) -> dict:
    bindings = {}
    for name in ("progress", "checkpoint_pointer", "guard_log"):
        path = tmp_path / f"{batch}.{name}"
        path.write_text(name)
        bindings[name] = {"path": str(path), "sha256": file_sha256(path)}
    value = {
        "schema_version": ARM_SCHEMA,
        "decision": "PASS",
        "launch_manifest_sha256": "1" * 64,
        "batch_size": batch,
        "precision_policy": "deterministic_fp32",
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": 2,
        "measured_entry_rows": 2 * batch,
        "transitions_per_entry": 4,
        "measured_transition_count": 8 * batch,
        "measured_train_seconds": seconds,
        "measured_transitions_per_second": 8 * batch / seconds,
        "guard_decision": "PASS",
        "safety": {
            "physical_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
        },
        **bindings,
        "test_data_used": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    path = tmp_path / f"{batch}.arm_receipt.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": file_sha256(path)}


def test_selects_highest_transition_rate_then_lower_batch(tmp_path: Path):
    selection = build_selection(
        [_arm(tmp_path, 4, 1.0), _arm(tmp_path, 8, 2.0), _arm(tmp_path, 16, 2.0)]
    )
    assert require_selection(selection)["selected_batch_size"] == 16


def test_selection_rejects_tampered_winner(tmp_path: Path):
    selection = build_selection(
        [_arm(tmp_path, 4, 1.0), _arm(tmp_path, 8, 2.0), _arm(tmp_path, 16, 4.0)]
    )
    selection["selected_batch_size"] = 16
    selection["artifact_sha256"] = canonical_sha256(
        {k: v for k, v in selection.items() if k != "artifact_sha256"}
    )
    with pytest.raises(RuntimeError, match="WINNER_INVALID"):
        require_selection(selection)


@pytest.mark.parametrize(
    "winner,seconds,expected_batches",
    [
        (4, {4: 0.1, 8: 2.0, 16: 4.0}, 4096),
        (8, {4: 2.0, 8: 0.1, 16: 4.0}, 2048),
        (16, {4: 2.0, 8: 4.0, 16: 0.1}, 1024),
    ],
)
def test_epoch_budget_is_invariant_while_batch_count_is_derived(
    tmp_path: Path, winner: int, seconds: dict[int, float], expected_batches: int
):
    selection = build_selection(
        [_arm(tmp_path, batch, seconds[batch]) for batch in (4, 8, 16)]
    )
    checked = require_selection(selection)
    assert checked["selected_batch_size"] == winner
    assert checked["transition_budget_per_epoch"] == 65536
    assert checked["entry_pairs_per_epoch"] == 16384
    assert checked["total_batches_per_epoch"] == expected_batches
