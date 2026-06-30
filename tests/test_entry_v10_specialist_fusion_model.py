import json
from pathlib import Path

import pytest
import torch

from gx1.features.entry_specialist_feature_groups_v1 import (
    CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT,
    CHALLENGER_SEQ215_TRAINING_SPECIALISTS,
    REQUIRED_TRAINING_SPECIALISTS,
    SPECIALIST_MODEL_CONTRACT,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract


def test_entry_v10_default_model_has_no_specialist_state_or_output() -> None:
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=7,
        snap_input_dim=7,
        seq_len=16,
        ctx_cont_dim=5,
        ctx_cat_dim=3,
    ).eval()

    assert not any("specialist" in key for key in model.state_dict())
    out = model(
        torch.randn(2, 16, 7),
        torch.rand(2, 7).clamp_min(0.01),
        ctx_cat=torch.zeros(2, 3, dtype=torch.long),
        ctx_cont=torch.randn(2, 5),
    )
    assert "specialist_gate" not in out


def test_entry_v10_specialist_fusion_forward_seq146_contract() -> None:
    specialist_indices = {
        "structure_swing_encoder": list(range(28, 32)) + list(range(89, 102)),
        "smc_liquidity_encoder": list(range(32, 37)) + list(range(102, 107)),
        "trend_ema_encoder": [15, 16, 18, 19, 22, 23],
        "vol_compression_encoder": [7, 8, 12, 17, 21, 24, 25, 37, 38, 39, 40] + list(range(107, 112)),
        "momentum_flow_encoder": [9, 10, 11, 20, 70, 112],
        "session_regime_encoder": list(range(118, 146)),
    }
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=146,
        snap_input_dim=146,
        seq_len=96,
        ctx_cont_dim=142,
        ctx_cat_dim=5,
        enable_specialist_fusion=True,
        specialist_input_indices=specialist_indices,
        specialist_num_layers=1,
    ).eval()

    seq = torch.randn(3, 96, 146)
    snap = torch.full((3, 146), 0.05)
    snap[:, 0:3] = torch.tensor([0.34, 0.33, 0.33])
    out = model(
        seq,
        snap,
        ctx_cat=torch.zeros(3, 5, dtype=torch.long),
        ctx_cont=torch.randn(3, 142),
    )

    assert out["direction_logits"].shape == (3, 3)
    assert out["specialist_gate"].shape == (3, len(specialist_indices))
    assert torch.allclose(out["specialist_gate"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.isfinite(out["specialist_gate"]).all()


def test_entry_v10_specialist_fusion_uses_current_audit_contract() -> None:
    audit_json = Path(
        "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
        "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
    )
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        audit_json,
        expected_signal_dim=146,
    )

    assert specialist_meta["enabled"] is True
    assert specialist_meta["signal_field_count"] == 146
    assert specialist_meta["selected_feature_count"] == 105
    assert set(specialist_meta["trainable_specialists"]) == set(REQUIRED_TRAINING_SPECIALISTS)
    assert set(specialist_meta["active_heads"]) == set(SPECIALIST_FUSION_ACTIVE_HEADS)
    assert set(specialist_meta["blocked_heads"]) == set(SPECIALIST_FUSION_BLOCKED_HEADS)
    assert specialist_meta["excluded_specialist_groups"]["neutral_bridge_anchor"] == 7
    assert specialist_meta["excluded_specialist_groups"]["price_action_candle_encoder"] == 3
    assert specialist_meta["specialist_model_contract_valid"] is True
    assert specialist_meta["specialist_model_contract_failures"] == []
    assert specialist_meta["specialist_model_contract"] == json.loads(json.dumps(SPECIALIST_MODEL_CONTRACT))
    assert specialist_meta["specialist_model_contract_set_exact"] is True
    assert specialist_meta["specialist_model_contract_owned_objectives_match"] is True
    assert specialist_meta["specialist_model_contract_signal_families_match"] is True
    assert specialist_meta["specialist_model_contract_support_heads_match"] is True
    assert specialist_meta["specialist_model_contract_model_roles_match"] is True
    assert "hold_horizon" not in set(specialist_meta["active_heads"])
    assert "hold_horizon" in set(specialist_meta["blocked_heads"])
    assert set(specialist_indices) == set(REQUIRED_TRAINING_SPECIALISTS)
    assert "neutral_bridge_anchor" not in specialist_indices
    assert "unmapped" not in specialist_indices
    assert "price_action_candle_encoder" not in specialist_indices
    for required in REQUIRED_TRAINING_SPECIALISTS:
        assert required in specialist_indices
        assert specialist_indices[required]

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=146,
        snap_input_dim=146,
        seq_len=96,
        ctx_cont_dim=142,
        ctx_cat_dim=5,
        enable_specialist_fusion=True,
        specialist_input_indices=specialist_indices,
        specialist_num_layers=1,
    ).eval()

    out = model(
        torch.randn(2, 96, 146),
        torch.full((2, 146), 0.05),
        ctx_cat=torch.zeros(2, 5, dtype=torch.long),
        ctx_cont=torch.randn(2, 142),
    )

    assert out["direction_logits"].shape == (2, 3)
    assert out["specialist_gate"].shape == (2, len(specialist_indices))
    assert torch.allclose(out["specialist_gate"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out["specialist_gate"]).all()


def test_entry_v10_specialist_fusion_rejects_tampered_model_contract(tmp_path: Path) -> None:
    audit_json = Path(
        "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
        "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
    )
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    payload["specialist_model_contract"]["structure_swing_encoder"]["owned_objectives"] = []
    tampered = tmp_path / "tampered_specialist_audit.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_CONTRACT_INVALID"):
        _load_specialist_fusion_contract(
            tampered,
            expected_signal_dim=146,
        )


def test_entry_v10_specialist_fusion_loads_challenger_seq215_contract() -> None:
    audit_json = Path(
        "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
        "challenger_seq215_20260630_contract8/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
    )
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        audit_json,
        expected_signal_dim=215,
        contract_mode="challenger_seq215",
    )

    assert specialist_meta["contract_mode"] == "challenger_seq215"
    assert specialist_meta["signal_field_count"] == 215
    assert specialist_meta["selected_feature_count"] == 174
    assert set(specialist_meta["trainable_specialists"]) == set(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert specialist_meta["specialist_model_contract"] == json.loads(
        json.dumps(CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT)
    )
    assert set(specialist_indices) == set(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert specialist_indices["chart_geometry_encoder"]
    assert specialist_indices["price_action_candle_encoder"]
    assert specialist_meta["excluded_specialist_groups"]["neutral_bridge_anchor"] == 7

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=215,
        snap_input_dim=215,
        seq_len=96,
        ctx_cont_dim=142,
        ctx_cat_dim=5,
        enable_specialist_fusion=True,
        specialist_input_indices=specialist_indices,
        specialist_num_layers=1,
    ).eval()

    out = model(
        torch.randn(2, 96, 215),
        torch.full((2, 215), 0.05),
        ctx_cat=torch.zeros(2, 5, dtype=torch.long),
        ctx_cont=torch.randn(2, 142),
    )

    assert out["direction_logits"].shape == (2, 3)
    assert out["specialist_gate"].shape == (2, len(CHALLENGER_SEQ215_TRAINING_SPECIALISTS))
    assert torch.allclose(out["specialist_gate"].sum(dim=1), torch.ones(2), atol=1e-6)
