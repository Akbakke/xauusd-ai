from __future__ import annotations

import pytest
import torch
from pathlib import Path

from gx1.contracts.unified_exit_incremental_carry_v1 import (
    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256,
    build_unified_exit_incremental_carry_envelope,
    decode_unified_exit_incremental_carry_tensors,
    require_unified_exit_incremental_carry_envelope,
)


def _build(*, step: int = 1, previous: str | None = None, mtf: str = "8"):
    return build_unified_exit_incremental_carry_envelope(
        tensor_state={
            "hidden/a": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            "hidden/b": torch.ones(1, 1, 3),
        },
        step_count=step,
        last_closed_m1_bar_ts=f"2026-01-01T00:{step - 1:02d}:00+00:00",
        trade_identity="trade-1",
        side="long",
        bundle_sha256="1" * 64,
        input_normalization_sha256="2" * 64,
        entry_token_snapshot_sha256="3" * 64,
        full_path_chain_sha256="4" * 64,
        input_envelope_sha256="5" * 64,
        previous_carry_envelope_sha256=(
            previous
            if previous is not None
            else UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
        ),
        mtf_last_row_sha256={
            "m5": mtf * 64,
            "m15": "6" * 64,
            "h1": "7" * 64,
            "h4": "a" * 64,
            "d1": "9" * 64,
        },
    )


def test_carry_round_trip_binds_exact_input_mtf_clock_and_identity():
    envelope = _build()
    validated = require_unified_exit_incremental_carry_envelope(
        envelope,
        expected_trade_identity="trade-1",
        expected_side="long",
        expected_bundle_sha256="1" * 64,
        expected_input_normalization_sha256="2" * 64,
        expected_entry_token_snapshot_sha256="3" * 64,
        expected_full_path_chain_sha256="4" * 64,
        expected_input_envelope_sha256="5" * 64,
        expected_mtf_last_row_sha256={
            "m5": "8" * 64,
            "m15": "6" * 64,
            "h1": "7" * 64,
            "h4": "a" * 64,
            "d1": "9" * 64,
        },
        expected_last_closed_m1_bar_ts="2026-01-01T00:00:00Z",
        expected_step_count=1,
        expected_previous_carry_envelope_sha256=(
            UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
        ),
    )
    restored = decode_unified_exit_incremental_carry_tensors(
        validated, device=torch.device("cpu")
    )
    assert torch.equal(
        restored["hidden/a"], torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"expected_input_envelope_sha256": "6" * 64}, "EXPECTED_BINDING"),
        (
            {
                "expected_mtf_last_row_sha256": {
                    "m5": "7" * 64,
                    "m15": "6" * 64,
                    "h1": "7" * 64,
                    "h4": "a" * 64,
                    "d1": "9" * 64,
                }
            },
            "EXPECTED_MTF_BINDING",
        ),
        (
            {"expected_last_closed_m1_bar_ts": "2026-01-01T00:02:00Z"},
            "EXPECTED_CLOCK",
        ),
    ),
)
def test_carry_rejects_cross_binding_and_clock_gap(kwargs, match):
    with pytest.raises(RuntimeError, match=match):
        require_unified_exit_incremental_carry_envelope(_build(), **kwargs)


def test_carry_chain_position_rejects_genesis_after_first_step():
    with pytest.raises(RuntimeError, match="CHAIN_POSITION"):
        _build(step=2, previous=UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256)
    second = _build(step=2, previous="a" * 64)
    assert second["previous_carry_envelope_sha256"] == "a" * 64


def test_carry_rejects_tensor_byte_mutation_even_when_json_shape_survives():
    envelope = _build()
    encoded = envelope["tensor_state"]["hidden/a"]["bytes_b64"]
    envelope["tensor_state"]["hidden/a"]["bytes_b64"] = (
        ("A" if encoded[0] != "A" else "B") + encoded[1:]
    )
    with pytest.raises(RuntimeError, match="TENSOR_BYTES|CONTENT_HASH"):
        require_unified_exit_incremental_carry_envelope(envelope)


def test_episode_native_exit_has_no_legacy_runtime_or_corpus_authority():
    root = Path(__file__).resolve().parents[1]
    production = tuple((root / "gx1").rglob("*.py"))
    allowed_audit_oracle = {
        root / "gx1/contracts/unified_exit_optimal_stopping_v1.py",
        root / "gx1/contracts/entry_exit_feature_usefulness_v1.py",
        root / "gx1/scripts/audit_entry_exit_feature_usefulness_v1.py",
    }
    for path in production:
        source = path.read_text(encoding="utf-8")
        assert "forward_exit_action(" not in source, path
        assert "iter_full_exit_trajectory_chunks(" not in source, path
        assert "iter_full_trajectory_chunks(" not in source, path
        if path not in allowed_audit_oracle:
            assert "unified_exit_optimal_stopping_targets(" not in source, path
            assert "_optimal_targets" not in source, path
