from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
)
from gx1.models.entry_v10 import entry_v10_bundle as bundle
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    MODEL_ARCHITECTURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION,
)


def _current_state_dict() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for keys in bundle._ENTRY_HEAD_STATE_KEYS.values():
        for key in keys:
            state[key] = torch.ones(1)
    return state


def test_bundle_active_head_surface_is_exact_fitted_q_plus_genuine_aux() -> None:
    assert set(bundle._ENTRY_HEAD_STATE_KEYS) == {
        *MODEL_NATIVE_ACTIVE_HEADS,
        "unified_exit",
    }
    assert bundle._MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS == frozenset(
        {*MODEL_NATIVE_ACTIVE_HEADS, "unified_exit"}
    )
    assert "entry_action_q" in bundle._ENTRY_HEAD_STATE_KEYS
    assert "unified_exit" in bundle._ENTRY_HEAD_STATE_KEYS


def test_bundle_capability_inference_rejects_retired_or_missing_heads() -> None:
    state = _current_state_dict()
    metadata = {
        "supports_context_features": True,
        "train_recipe": {
            "active_heads": list(bundle._ENTRY_HEAD_STATE_KEYS),
        },
    }
    capabilities = bundle._infer_entry_bundle_capabilities(metadata, state)
    assert set(capabilities["supported_heads"]) == set(
        bundle._ENTRY_HEAD_STATE_KEYS
    )

    missing = dict(state)
    missing.pop("head_entry_action_q.weight")
    with pytest.raises(RuntimeError, match="CAPABILITY_MISMATCH"):
        bundle._infer_entry_bundle_capabilities(metadata, missing)


def test_entry_q_state_contract_is_shape_and_finiteness_bound() -> None:
    d_model = 128
    state = {
        "entry_q_joint_norm.weight": torch.ones(4 * d_model),
        "entry_q_joint_norm.bias": torch.zeros(4 * d_model),
        "entry_q_joint_in.weight": torch.ones(d_model, 4 * d_model),
        "entry_q_joint_in.bias": torch.zeros(d_model),
        "head_entry_action_q.weight": torch.stack(
            (
                torch.ones(d_model),
                torch.full((d_model,), 2.0),
                torch.full((d_model,), 3.0),
            )
        ),
        "head_entry_action_q.bias": torch.zeros(3),
    }
    bundle._require_entry_q_state_contract(state)
    state["head_entry_action_q.weight"][0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="ENTRY_FITTED_Q_STATE_INVALID"):
        bundle._require_entry_q_state_contract(state)


def test_bundle_source_has_no_probability_or_legacy_head_authority() -> None:
    source = Path(bundle.__file__).read_text(encoding="utf-8")
    active_region = source[: source.index("_RETIRED_DIRECTION_STATE_PREFIXES")]
    forbidden = (
        "direction_" + "logits",
        "direction_" + "probs",
        "exit_action_" + "logits",
        "exit_action_" + "probs",
        "evidence_" + "fusion",
        "head_" + "direction",
        "head_mtf_" + "direction",
        "head_action_" + "value",
        "head_expectile_" + "value",
    )
    assert [token for token in forbidden if token in active_region] == []


def test_current_model_and_decision_contract_versions_are_explicit() -> None:
    assert MODEL_ARCHITECTURE_SCHEMA_VERSION
    assert MODEL_OUTPUT_SCHEMA_VERSION
    assert DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION == (
        "gx1_model_direction_decision_v10"
    )
