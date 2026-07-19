from __future__ import annotations

from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_EXTRA_ACTIVE_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    artifact_fingerprint,
    artifact_fingerprint_checks,
    model_native_readiness_contract_metadata,
    require_model_native_readiness_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)


def test_readiness_contract_is_exact_seq513_head_and_specialist_authority() -> None:
    contract = model_native_readiness_contract_metadata()

    assert contract["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert contract["signal_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert tuple(contract["required_specialists"]) == MODEL_NATIVE_REQUIRED_SPECIALISTS
    assert tuple(contract["base_active_heads"]) == MODEL_NATIVE_BASE_ACTIVE_HEADS
    assert tuple(contract["extra_active_heads"]) == MODEL_NATIVE_EXTRA_ACTIVE_HEADS
    assert tuple(contract["active_heads"]) == MODEL_NATIVE_ACTIVE_HEADS
    assert tuple(contract["blocked_heads"]) == MODEL_NATIVE_BLOCKED_HEADS
    assert len(MODEL_NATIVE_REQUIRED_SPECIALISTS) == 8
    assert MODEL_NATIVE_ACTIVE_HEADS == (
        *MODEL_NATIVE_BASE_ACTIVE_HEADS,
        *MODEL_NATIVE_EXTRA_ACTIVE_HEADS,
    )
    assert contract["secondary_direction_authority_allowed"] is False
    assert contract["mutable_latest_evidence_allowed"] is False
    assert require_model_native_readiness_contract(
        contract,
        context="UNIT",
    ) == contract


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("signal_dim", 512),
        ("active_heads", ["direction"]),
        ("required_specialists", ["structure_swing_encoder"]),
        ("mutable_latest_evidence_allowed", True),
    ],
)
def test_readiness_contract_rejects_every_partial_or_mutated_surface(
    key: str,
    value: object,
) -> None:
    contract = model_native_readiness_contract_metadata()
    contract[key] = value

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_READINESS_CONTRACT_INVALID"):
        require_model_native_readiness_contract(contract, context="UNIT")


def test_content_fingerprint_rejects_symlink_and_never_uses_mtime(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "event.json"
    artifact.write_text('{"decision":"PASS"}\n', encoding="utf-8")
    link = tmp_path / "event-link.json"
    link.symlink_to(artifact)

    row = artifact_fingerprint(artifact)
    link_row = artifact_fingerprint(link)

    assert row["exists"] is True
    assert row["regular_file"] is True
    assert len(row["sha256"]) == 64
    assert "mtime_ns" not in row
    assert link_row["exists"] is True
    assert link_row["regular_file"] is False
    assert link_row["sha256"] is None
    checks = artifact_fingerprint_checks({"event": row, "link": link_row})
    assert checks == [
        {
            "name": "readiness records content-addressed regular-file evidence",
            "ok": False,
            "details": {"artifact_fingerprints": {"event": row, "link": link_row}},
        }
    ]
