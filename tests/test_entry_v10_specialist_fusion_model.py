import json
from pathlib import Path

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract


def _specialist_indices() -> dict[str, list[int]]:
    return {
        name: [index]
        for index, name in enumerate(MODEL_NATIVE_TRAINING_SPECIALISTS)
    }


def _audit_payload() -> dict:
    return {
        "decision": "PASS",
        "created_utc": "2026-07-16T00:00:00+00:00",
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
        "selected_feature_count": 479,
        "specialist_model_contract": json.loads(
            json.dumps(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT)
        ),
        "specialist_model_contract_valid": True,
        "specialist_model_contract_failures": [],
        "architecture_contract": {
            "specialist_input_indices": _specialist_indices(),
            "recommended_fusion": {
                "active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
                "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
            },
        },
    }


def _write_audit(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "ENTRY_MODEL_NATIVE_SPECIALIST_AUDIT_20260716T000000Z.json"
    path.write_text(json.dumps(payload or _audit_payload()), encoding="utf-8")
    return path


def test_entry_v10_exact_model_always_has_specialist_state_and_output() -> None:
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=16,
        snap_input_dim=16,
        seq_len=16,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=3,
        m15_seq_dim=3,
        h1_seq_dim=3,
        h4_seq_dim=3,
        d1_seq_dim=3,
        m5_seq_len=16,
        m15_seq_len=16,
        h1_seq_len=16,
        h4_seq_len=16,
        d1_seq_len=16,
        specialist_input_indices=_specialist_indices(),
    ).eval()

    assert any("specialist" in key for key in model.state_dict())
    out = model(
        torch.randn(2, 16, 16),
        torch.rand(2, 16).clamp_min(0.01),
        ctx_cat=torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        ctx_cont=torch.randn(2, MODEL_NATIVE_CTX_CONT_DIM),
        **{f"seq_{tf}": torch.randn(2, 16, 3) for tf in ("m5", "m15", "h1", "h4", "d1")},
    )
    assert "specialist_gate" in out
    assert out["model_native_logits"].shape == (2, 3)
    assert not ({"anchor_logits", "delta_logits", "anchor_gate"} & set(out))


def test_entry_v10_specialist_fusion_forward_exact_model_native_contract() -> None:
    specialist_indices = _specialist_indices()
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=16,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=3,
        m15_seq_dim=3,
        h1_seq_dim=3,
        h4_seq_dim=3,
        d1_seq_dim=3,
        m5_seq_len=16,
        m15_seq_len=16,
        h1_seq_len=16,
        h4_seq_len=16,
        d1_seq_len=16,
        specialist_input_indices=specialist_indices,
        specialist_num_layers=1,
    ).eval()

    out = model(
        torch.randn(2, 16, MODEL_NATIVE_SIGNAL_DIM),
        torch.full((2, MODEL_NATIVE_SIGNAL_DIM), 0.05),
        ctx_cat=torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        ctx_cont=torch.randn(2, MODEL_NATIVE_CTX_CONT_DIM),
        **{f"seq_{tf}": torch.randn(2, 16, 3) for tf in ("m5", "m15", "h1", "h4", "d1")},
    )

    assert model._specialist_names == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert out["direction_logits"].shape == (2, 3)
    assert out["model_native_logits"].shape == (2, 3)
    assert not ({"anchor_logits", "delta_logits", "anchor_gate"} & set(out))
    assert out["specialist_gate"].shape == (2, len(MODEL_NATIVE_TRAINING_SPECIALISTS))
    assert torch.allclose(out["specialist_gate"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out["specialist_gate"]).all()


def test_entry_v10_specialist_loader_accepts_only_exact_model_native_audit(
    tmp_path: Path,
) -> None:
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        _write_audit(tmp_path),
        expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )

    assert tuple(specialist_indices) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert specialist_meta["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert specialist_meta["signal_field_count"] == MODEL_NATIVE_SIGNAL_DIM
    assert specialist_meta["selected_feature_count"] == 479
    assert specialist_meta["trainable_specialists"] == list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert specialist_meta["excluded_specialist_groups"] == {}
    assert specialist_meta["active_heads"] == list(SPECIALIST_FUSION_ACTIVE_HEADS)
    assert specialist_meta["blocked_heads"] == list(SPECIALIST_FUSION_BLOCKED_HEADS)
    assert specialist_meta["specialist_model_contract"] == json.loads(
        json.dumps(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT)
    )


def test_entry_v10_specialist_loader_rejects_tampered_model_contract(tmp_path: Path) -> None:
    payload = _audit_payload()
    payload["specialist_model_contract"]["structure_swing_encoder"]["owned_objectives"] = []

    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_CONTRACT_INVALID"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path, payload),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )


@pytest.mark.parametrize(
    "stale_mode",
    ["", "foundation_seq146", "challenger_seq215", "smart_seq520_candidate"],
)
def test_entry_v10_specialist_loader_rejects_stale_contract_modes(
    tmp_path: Path,
    stale_mode: str,
) -> None:
    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_NATIVE_CONTRACT_REQUIRED"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            contract_mode=stale_mode,
        )
