from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_COMPONENTS,
    ENTRY_DECISION_TOKEN_DIM,
    ENTRY_DECISION_TOKEN_SOURCE_DIM,
    build_entry_decision_token_snapshot,
    entry_decision_token_projection_metadata,
    entry_decision_token_tensor,
    require_entry_decision_token_bindings,
    require_entry_decision_token_snapshot,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
)


def _snapshot() -> dict[str, object]:
    token = np.linspace(-1.0, 1.0, ENTRY_DECISION_TOKEN_DIM, dtype=np.float32)
    return build_entry_decision_token_snapshot(
        token=token,
        decision_time="2026-08-14T10:00:00+00:00",
        fill_time="2026-08-14T10:05:02.125000+00:00",
        model_identity_kind="bundle_sha256",
        model_identity_sha256="a" * 64,
        input_normalization_sha256="b" * 64,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        model_direction_index=0,
        model_direction="LONG",
        side="long",
        entry_bid=3350.1,
        entry_ask=3350.3,
        trade_identity="trade-token-unit",
    )


def test_projection_contract_has_all_six_exact_ordered_blocks() -> None:
    metadata = entry_decision_token_projection_metadata()
    assert [
        (row["name"], row["width"])
        for row in metadata["components"]
    ] == list(ENTRY_DECISION_TOKEN_COMPONENTS)
    assert metadata["source_dim"] == ENTRY_DECISION_TOKEN_SOURCE_DIM == 643
    assert metadata["token_dim"] == ENTRY_DECISION_TOKEN_DIM == 128
    assert metadata["handwritten_component_weights"] is False


def test_fill_snapshot_roundtrips_exact_little_endian_float32_bytes() -> None:
    snapshot = _snapshot()
    assert require_entry_decision_token_snapshot(snapshot) == snapshot
    expected = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    assert np.array_equal(entry_decision_token_tensor(snapshot), expected)
    assert require_entry_decision_token_bindings(
        snapshot,
        raw_token_alias=expected.astype(np.float64),
        decision_time=snapshot["decision_time"],
        fill_time=snapshot["fill_time"],
        model_identity_kind="bundle_sha256",
        model_identity_sha256="a" * 64,
        input_normalization_sha256="b" * 64,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        model_direction_index=0,
        model_direction="LONG",
        side="long",
        entry_bid=3350.1,
        entry_ask=3350.3,
        trade_identity="trade-token-unit",
        context="UNIT",
    ) == snapshot


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("tensor_bytes_b64", "AAAA", "BYTES_INVALID"),
        ("tensor_sha256", "c" * 64, "TENSOR_HASH_MISMATCH"),
        (
            "decision_time",
            "2026-08-14T10:01:00+00:00",
            "SNAPSHOT_HASH_MISMATCH",
        ),
        ("model_identity_sha256", "d" * 64, "SNAPSHOT_HASH_MISMATCH"),
        ("input_normalization_sha256", "e" * 64, "SNAPSHOT_HASH_MISMATCH"),
    ),
)
def test_fill_snapshot_rejects_mutated_bytes_hash_time_bundle_or_normalization(
    field: str,
    replacement: object,
    match: str,
) -> None:
    tampered = deepcopy(_snapshot())
    tampered[field] = replacement
    with pytest.raises(RuntimeError, match=match):
        require_entry_decision_token_snapshot(tampered)


def test_rehashed_snapshot_still_rejects_wrong_fill_binding() -> None:
    snapshot = _snapshot()
    with pytest.raises(RuntimeError, match="BINDING_MISMATCH"):
        require_entry_decision_token_bindings(
            snapshot,
            raw_token_alias=entry_decision_token_tensor(snapshot),
            decision_time=snapshot["decision_time"],
            fill_time="2026-08-14T10:05:03+00:00",
            model_identity_kind="bundle_sha256",
            model_identity_sha256="a" * 64,
            input_normalization_sha256="b" * 64,
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
            model_direction_index=0,
            model_direction="LONG",
            side="long",
            entry_bid=3350.1,
            entry_ask=3350.3,
            trade_identity="trade-token-unit",
            context="UNIT",
        )
