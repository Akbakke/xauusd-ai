from __future__ import annotations

import json
from pathlib import Path

import pytest

import gx1.contracts.unified_exit_gpu_batch_selection_v1 as owner
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    ARM_SCHEMA,
    build_selection,
    canonical_sha256,
    file_sha256,
    require_arm_receipt,
    require_selection,
)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _arm(tmp_path: Path, batch: int, seconds: float) -> tuple[dict, dict]:
    files = {}
    for name in (
        "campaign_plan", "campaign_invocation", "campaign_receipt", "measurement",
        "progress", "checkpoint_pointer", "guard_log",
    ):
        path = (tmp_path / str(batch) / name).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = _binding(path)
    prelaunch = (tmp_path / "prelaunch_manifest").resolve()
    if not prelaunch.exists():
        prelaunch.write_text("prelaunch_manifest")
    files["prelaunch_manifest"] = _binding(prelaunch)
    value = {
        "schema_version": ARM_SCHEMA,
        "decision": "PASS",
        "campaign_plan": files["campaign_plan"],
        "campaign_plan_sha256": "a" * 64,
        "campaign_invocation": files["campaign_invocation"],
        "campaign_invocation_sha256": f"{batch:x}" * 64,
        "campaign_receipt": files["campaign_receipt"],
        "campaign_receipt_sha256": "b" * 64,
        "measurement": files["measurement"],
        "measurement_sha256": "c" * 64,
        "source_commit": "d" * 40,
        "prelaunch_manifest": files["prelaunch_manifest"],
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
        "boot_identity_sha256": "e" * 64,
        "safety": {
            "physical_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
        },
        "progress": files["progress"],
        "checkpoint_pointer": files["checkpoint_pointer"],
        "guard_log": files["guard_log"],
        "test_data_used": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    path = (tmp_path / str(batch) / "arm.json").resolve()
    path.write_text(json.dumps(value, sort_keys=True))
    return _binding(path), value


def _patch_loaded(monkeypatch: pytest.MonkeyPatch, pairs: list[tuple[dict, dict]]) -> None:
    monkeypatch.setattr(owner, "_checked_arm_set", lambda _bindings: pairs)


def test_selects_highest_transition_rate_then_lower_batch(tmp_path: Path, monkeypatch):
    pairs = [_arm(tmp_path, 4, 1.0), _arm(tmp_path, 8, 2.0), _arm(tmp_path, 16, 2.0)]
    _patch_loaded(monkeypatch, pairs)
    selection = build_selection([binding for binding, _ in pairs])
    assert require_selection(selection)["selected_batch_size"] == 16


def test_selection_rejects_tampered_winner(tmp_path: Path, monkeypatch):
    pairs = [_arm(tmp_path, 4, 1.0), _arm(tmp_path, 8, 2.0), _arm(tmp_path, 16, 4.0)]
    _patch_loaded(monkeypatch, pairs)
    selection = build_selection([binding for binding, _ in pairs])
    selection["selected_batch_size"] = 16
    selection["artifact_sha256"] = canonical_sha256(
        {k: v for k, v in selection.items() if k != "artifact_sha256"}
    )
    with pytest.raises(RuntimeError, match="REBUILD_MISMATCH"):
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
    tmp_path: Path, monkeypatch, winner: int, seconds: dict[int, float], expected_batches: int
):
    pairs = [_arm(tmp_path, batch, seconds[batch]) for batch in (4, 8, 16)]
    _patch_loaded(monkeypatch, pairs)
    selection = build_selection([binding for binding, _ in pairs])
    checked = require_selection(selection)
    assert checked["selected_batch_size"] == winner
    assert checked["transition_budget_per_epoch"] == 65536
    assert checked["entry_pairs_per_epoch"] == 16384
    assert checked["total_batches_per_epoch"] == expected_batches


def test_arm_receipt_rejects_legacy_self_attestation(tmp_path: Path) -> None:
    legacy = {
        "schema_version": ARM_SCHEMA,
        "decision": "PASS",
        "batch_size": 16,
        "test_data_used": False,
    }
    legacy["receipt_sha256"] = canonical_sha256(legacy)
    with pytest.raises(RuntimeError, match="ARM_RECEIPT_INVALID"):
        require_arm_receipt(legacy, verify_files=False)


def test_arm_receipt_requires_exact_prelaunch_file_binding(
    tmp_path: Path,
) -> None:
    _binding_value, arm = _arm(tmp_path, 16, 1.0)
    arm.pop("prelaunch_manifest")
    arm["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in arm.items() if key != "receipt_sha256"}
    )
    with pytest.raises(RuntimeError, match="ARM_RECEIPT_INVALID"):
        require_arm_receipt(arm, verify_files=False)


def test_selection_rejects_prelaunch_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = [_arm(tmp_path, 4, 1.0), _arm(tmp_path, 8, 1.0), _arm(tmp_path, 16, 1.0)]
    pairs[1][1]["prelaunch_manifest"] = _binding(
        (tmp_path / "8" / "campaign_plan").resolve()
    )
    pairs[1][1]["receipt_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in pairs[1][1].items()
            if key != "receipt_sha256"
        }
    )
    _patch_loaded(monkeypatch, pairs)
    with pytest.raises(RuntimeError, match="LAUNCH_MISMATCH"):
        build_selection([binding for binding, _ in pairs])
