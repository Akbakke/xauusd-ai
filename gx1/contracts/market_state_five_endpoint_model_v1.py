"""Five native market-state endpoints on the existing causal research backbone.

This is an output contract, not a fitted model, optimizer, serving route, or
permission to train. The caller must bind causal features and separate future
targets, normalization, numerical policy, and checkpoint state.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from gx1.contracts.entry_native_direction_research_v1 import _backbone_width
from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    CLASS_NAMES, LEAD_MINUTES, TIMEFRAMES, five_endpoint_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer


def five_endpoint_model_contract() -> dict:
    """Separate identity for five paired queries and freshly initialized heads."""
    value = {
        "schema_version": "gx1_market_state_five_endpoint_model_v1",
        "role": "research_only_no_entry_exit_or_trade_authority",
        "target_contract_sha256": five_endpoint_contract()["contract_sha256"],
        "timeframes": list(TIMEFRAMES),
        "lead_minutes": list(LEAD_MINUTES),
        "class_order": list(CLASS_NAMES),
        "backbone": "native_entry_q_joint_hidden_research_representation_only_without_detach",
        "query": "five_paired_timeframe_one_hot_with_fixed_native_leads",
        "head": "fresh_shared_Linear_hidden_plus_5_to_hidden_GELU_then_Linear_hidden_to_3",
        "output": "logits_batch_5_3",
        "current_state_or_future_target_as_forward_input": False,
        "old_checkpoint_compatible": False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _coordinates(device: torch.device) -> torch.Tensor:
    return torch.eye(len(TIMEFRAMES), device=device, dtype=torch.float32)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("FIVE_ENDPOINT_MODEL_" + reason)


class FiveEndpointMarketStateModel(nn.Module):
    """Five direct endpoint logits; no dense transition or action decoder."""

    def __init__(self, backbone: EntryV10CtxHybridTransformer):
        super().__init__()
        width = _backbone_width(backbone)
        _require(isinstance(backbone.entry_execution_profile, dict)
                 and backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
                 and backbone.gc_mtf_history is not None
                 and "relative_price" not in backbone.gc_mtf_history["normalization"], "FULL_GC_PROFILE")
        self.backbone = backbone
        self._hidden_width = width
        device = backbone.entry_q_joint_in.weight.device
        self.endpoint_query_projection = nn.Sequential(
            nn.Linear(width + len(TIMEFRAMES), width, device=device, dtype=torch.float32), nn.GELU())
        self.endpoint_head = nn.Linear(width, len(CLASS_NAMES), device=device, dtype=torch.float32)
        self.register_buffer("endpoint_query_coordinates", _coordinates(device))
        self.register_buffer("endpoint_lead_minutes", torch.tensor(LEAD_MINUTES, device=device, dtype=torch.int64))
        self.register_buffer("endpoint_objective_sha256", torch.tensor(
            list(bytes.fromhex(five_endpoint_model_contract()["contract_sha256"])),
            device=device, dtype=torch.uint8))

    @property
    def research_objective(self) -> dict:
        return five_endpoint_model_contract()

    def _require_binding(self) -> None:
        width = _backbone_width(self.backbone)
        _require(width == self._hidden_width, "HIDDEN_WIDTH")
        _require(isinstance(self.backbone.entry_execution_profile, dict)
                 and self.backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
                 and self.backbone.gc_mtf_history is not None
                 and "relative_price" not in self.backbone.gc_mtf_history["normalization"], "FULL_GC_PROFILE")
        device = self.backbone.entry_q_joint_in.weight.device
        expected = {
            "endpoint_query_coordinates": _coordinates(device),
            "endpoint_lead_minutes": torch.tensor(LEAD_MINUTES, device=device, dtype=torch.int64),
            "endpoint_objective_sha256": torch.tensor(
                list(bytes.fromhex(self.research_objective["contract_sha256"])), device=device, dtype=torch.uint8),
        }
        for name, reference in expected.items():
            value = getattr(self, name)
            _require(value.shape == reference.shape and value.dtype == reference.dtype
                     and value.device == device and not value.requires_grad
                     and torch.equal(value, reference), "BUFFER:" + name)
        _require(type(self.endpoint_query_projection) is nn.Sequential
                 and len(self.endpoint_query_projection) == 2
                 and type(self.endpoint_query_projection[1]) is nn.GELU, "QUERY_PROJECTION")
        for layer, inputs, outputs in (
            (self.endpoint_query_projection[0], width + len(TIMEFRAMES), width),
            (self.endpoint_head, width, len(CLASS_NAMES)),
        ):
            _require(type(layer) is nn.Linear and layer.in_features == inputs and layer.out_features == outputs
                     and layer.weight.shape == (outputs, inputs) and layer.bias is not None
                     and layer.bias.shape == (outputs,) and layer.weight.dtype == layer.bias.dtype == torch.float32
                     and layer.weight.device == layer.bias.device == device
                     and bool(torch.isfinite(layer.weight).all())
                     and bool(torch.isfinite(layer.bias).all()), "HEAD_BINDING")

    def forward(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor, gc_seq_m15: torch.Tensor,
        gc_seq_h1: torch.Tensor, gc_seq_h4: torch.Tensor, gc_seq_d1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._require_binding()
        _require(isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3
                 and seq_x.shape[0] > 0, "FEATURE_BATCH")
        output = self.backbone(
            seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
            gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15, gc_seq_h1=gc_seq_h1,
            gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1, research_representation_only=True)
        _require(isinstance(output, dict), "BACKBONE_OUTPUT")
        hidden = output.get("entry_q_joint_hidden")
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
                 and hidden.shape == (seq_x.shape[0], self._hidden_width)
                 and hidden.device == self.endpoint_query_coordinates.device
                 and bool(torch.isfinite(hidden).all()), "BACKBONE_HIDDEN")
        batch = hidden.shape[0]
        source = hidden[:, None, :].expand(-1, len(TIMEFRAMES), -1)
        coords = self.endpoint_query_coordinates[None].expand(batch, -1, -1)
        decoded = self.endpoint_query_projection(torch.cat((source, coords), dim=-1))
        logits = self.endpoint_head(decoded)
        _require(logits.shape == (batch, len(TIMEFRAMES), len(CLASS_NAMES))
                 and bool(torch.isfinite(logits).all()), "LOGITS")
        return {"logits": logits}
