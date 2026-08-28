"""The bundle loader must match the architecture that the trainer records."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import gx1.models.entry_v10.entry_v10_bundle as bundle


def test_active_architecture_marks_positional_encoding_and_retired_film() -> None:
    bundle._require_model_native_architecture_markers(
        {"enable_pos_enc": True, "enable_regime_film": False}
    )


@pytest.mark.parametrize(
    ("metadata", "error"),
    (
        ({"enable_pos_enc": False, "enable_regime_film": False}, "enable_pos_enc"),
        ({"enable_pos_enc": True, "enable_regime_film": True}, "RETIRED_REGIME_FILM"),
        ({"enable_pos_enc": True}, "RETIRED_REGIME_FILM"),
    ),
)
def test_architecture_markers_fail_closed_on_train_serve_split(
    metadata: dict[str, object], error: str
) -> None:
    with pytest.raises(RuntimeError, match=error):
        bundle._require_model_native_architecture_markers(metadata)


def test_active_event_head_contract_rejects_retired_head_metadata() -> None:
    active = {
        "trendline_event_head": {
            "enabled": True,
            "output_dim": 4,
            "hand_written_direction_pressure": False,
            "direction_mapping": "representation_only_no_entry_authority",
        }
    }
    bundle._require_model_native_retired_head_contract(active)

    with pytest.raises(RuntimeError, match="STALE_ENTRY_HEAD"):
        bundle._require_model_native_retired_head_contract(
            {**active, "hierarchical_entry_heads": {"enabled": True}}
        )


def test_feature_gate_row_liveness_is_candidate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rare event fields cannot be required to occur in a 32-row smoke."""

    routing = {"chart_geometry_encoder": (0, 1, 2)}
    monkeypatch.setattr(
        bundle,
        "require_multi_tf_specialist_routing_v4",
        lambda _features: routing,
    )
    monkeypatch.setattr(bundle, "_require_entry_q_state_contract", lambda _state: None)
    state = {
        key: torch.ones(1)
        for keys in bundle._MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS.values()
        for key in keys
    }
    for timeframe in bundle.TF_INPUT_SCALE_NAMES:
        state[f"mtf_feature_context_gate.{timeframe}__chart_geometry_encoder.weight"] = (
            torch.ones(3, 1)
        )
    state[
        "mtf_feature_context_gate.m5__chart_geometry_encoder.weight"
    ][2].zero_()

    bundle._require_model_native_learned_component_liveness(
        state,
        training_profile="smoke",
    )
    with pytest.raises(RuntimeError, match="zero_init_pass_through_rows=\\[2\\]"):
        bundle._require_model_native_learned_component_liveness(
            state,
            training_profile="candidate",
        )


def test_trainer_writes_every_direct_loader_metadata_requirement() -> None:
    """Catch producer/loader key splits before a bounded CUDA smoke runs."""

    root = Path(__file__).resolve().parents[1]
    trainer = ast.parse(
        (root / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text()
    )
    loader = ast.parse(
        (root / "gx1/models/entry_v10/entry_v10_bundle.py").read_text()
    )
    metadata_keys: set[str] = set()
    for node in ast.walk(trainer):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "meta" for target in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            candidate_keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "schema_version" in candidate_keys and "train_recipe" in candidate_keys:
                metadata_keys = candidate_keys
                break
    assert metadata_keys

    required: set[str] = set()
    for node in ast.walk(loader):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "meta"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            required.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_require_mapping_field"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "meta"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            required.add(node.args[1].value)
    assert required <= metadata_keys, sorted(required - metadata_keys)
