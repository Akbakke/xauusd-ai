"""Shared causal state readout at preselected future M5 decision times.

One backbone encoding can decode multiple time queries. This is a research
output only: target binding, loss, scoring, fit and any trading are separate.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from gx1.contracts.entry_native_direction_research_v1 import _backbone_width
from gx1.contracts.market_state_five_endpoint_contract_v1 import CLASS_NAMES
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

MAX_QUERY_BLOCK = 32


def time_query_model_contract() -> dict:
    value = {
        "schema_version": "gx1_market_state_time_query_model_v1",
        "role": "research_only_no_entry_exit_or_trade_authority",
        "timeframes": list(TIMEFRAMES),
        "native_k": [int(k) for k in NATIVE_K],
        "future_validity": "closed_M5_issue_plus_5k_UTC_minutes",
        "class_order": list(CLASS_NAMES),
        "backbone": "native_entry_q_joint_hidden_research_representation_only_without_detach",
        "query": "five_timeframe_one_hot_plus_preselected_k_over_native_k",
        "head": "fresh_shared_Linear_hidden_plus_6_to_hidden_GELU_then_Linear_hidden_to_3",
        "output": "logits_batch_5_3_for_requested_future_validity",
        "backbone_reuse": "encode_features_once_decode_queries_or_bounded_grid_many_times",
        "dense_query_block": "decode_query_grid_Q_1_to_32_logits_B_Q_5_3",
        "current_state_or_future_target_as_forward_input": False,
        "old_checkpoint_compatible": False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("TIME_QUERY_MODEL_" + reason)


class TimeQueryMarketStateModel(nn.Module):
    """One shared decoder for five TF and a requested M5-tick offset per TF."""

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
        self.query_projection = nn.Sequential(
            nn.Linear(width + len(TIMEFRAMES) + 1, width, device=device, dtype=torch.float32),
            nn.GELU())
        self.query_head = nn.Linear(width, len(CLASS_NAMES), device=device, dtype=torch.float32)
        self.register_buffer("query_tf_coordinates", torch.eye(len(TIMEFRAMES), device=device,
                                                                dtype=torch.float32))
        self.register_buffer("query_native_k", torch.tensor([int(k) for k in NATIVE_K],
                                                      device=device, dtype=torch.int64))
        self.register_buffer("query_objective_sha256", torch.tensor(
            list(bytes.fromhex(time_query_model_contract()["contract_sha256"])),
            device=device, dtype=torch.uint8))

    @property
    def research_objective(self) -> dict:
        return time_query_model_contract()

    def _require_binding(self) -> None:
        width = _backbone_width(self.backbone)
        _require(width == self._hidden_width, "HIDDEN_WIDTH")
        _require(isinstance(self.backbone.entry_execution_profile, dict)
                 and self.backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
                 and self.backbone.gc_mtf_history is not None
                 and "relative_price" not in self.backbone.gc_mtf_history["normalization"], "FULL_GC_PROFILE")
        device = self.backbone.entry_q_joint_in.weight.device
        expected = {
            "query_tf_coordinates": torch.eye(len(TIMEFRAMES), device=device, dtype=torch.float32),
            "query_native_k": torch.tensor([int(k) for k in NATIVE_K], device=device, dtype=torch.int64),
            "query_objective_sha256": torch.tensor(
                list(bytes.fromhex(self.research_objective["contract_sha256"])),
                device=device, dtype=torch.uint8),
        }
        for name, reference in expected.items():
            value = getattr(self, name)
            _require(value.shape == reference.shape and value.dtype == reference.dtype
                     and value.device == device and not value.requires_grad
                     and torch.equal(value, reference), "BUFFER:" + name)
        _require(type(self.query_projection) is nn.Sequential
                 and len(self.query_projection) == 2
                 and type(self.query_projection[1]) is nn.GELU, "QUERY_PROJECTION")
        for layer, inputs, outputs in (
            (self.query_projection[0], width + len(TIMEFRAMES) + 1, width),
            (self.query_head, width, len(CLASS_NAMES)),
        ):
            _require(type(layer) is nn.Linear and layer.in_features == inputs
                     and layer.out_features == outputs and layer.weight.shape == (outputs, inputs)
                     and layer.bias is not None and layer.bias.shape == (outputs,)
                     and layer.weight.dtype == layer.bias.dtype == torch.float32
                     and layer.weight.device == layer.bias.device == device
                     and bool(torch.isfinite(layer.weight).all())
                     and bool(torch.isfinite(layer.bias).all()), "HEAD_BINDING")

    def encode_features(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor, gc_seq_m15: torch.Tensor, gc_seq_h1: torch.Tensor,
        gc_seq_h4: torch.Tensor, gc_seq_d1: torch.Tensor,
    ) -> torch.Tensor:
        self._require_binding()
        _require(isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3
                 and seq_x.shape[0] > 0, "FEATURE_BATCH")
        features = {"seq_x": seq_x, "snap_x": snap_x, "ctx_cat": ctx_cat,
                    "ctx_cont": ctx_cont, "seq_m5": seq_m5, "seq_m15": seq_m15,
                    "seq_h1": seq_h1, "seq_h4": seq_h4, "seq_d1": seq_d1,
                    "gc_seq_m5": gc_seq_m5, "gc_seq_m15": gc_seq_m15,
                    "gc_seq_h1": gc_seq_h1, "gc_seq_h4": gc_seq_h4,
                    "gc_seq_d1": gc_seq_d1}
        for name, value in features.items():
            integer = name == "ctx_cat"
            rank = 3 if name == "seq_x" or name.startswith(("seq_", "gc_seq_")) else 2
            _require(isinstance(value, torch.Tensor) and value.ndim == rank
                     and value.shape[0] == seq_x.shape[0]
                     and all(size > 0 for size in value.shape)
                     and value.device == self.query_tf_coordinates.device
                     and (value.dtype in (torch.int8, torch.int16, torch.int32,
                                          torch.int64, torch.uint8) if integer
                          else value.dtype == torch.float32 and bool(torch.isfinite(value).all())),
                     "FEATURE:" + name)
        output = self.backbone(
            seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
            gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15, gc_seq_h1=gc_seq_h1,
            gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1,
            research_representation_only=True)
        _require(isinstance(output, dict), "BACKBONE_OUTPUT")
        hidden = output.get("entry_q_joint_hidden")
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
                 and hidden.shape == (seq_x.shape[0], self._hidden_width)
                 and hidden.device == self.query_tf_coordinates.device
                 and bool(torch.isfinite(hidden).all()), "BACKBONE_HIDDEN")
        return hidden

    def decode_queries(self, hidden: torch.Tensor, query_k: torch.Tensor) -> torch.Tensor:
        """Decode one selected offset per TF; the original [B,5] API."""
        _require(isinstance(query_k, torch.Tensor) and query_k.ndim == 2, "QUERY_K")
        return self.decode_query_grid(hidden, query_k[:, None, :])[:, 0, :, :]

    def decode_query_grid(self, hidden: torch.Tensor,
                          query_k: torch.Tensor) -> torch.Tensor:
        """Decode one bounded preselected grid block with one head binding check."""
        self._require_binding()
        batch = hidden.shape[0] if isinstance(hidden, torch.Tensor) and hidden.ndim == 2 else -1
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
                 and hidden.shape == (batch, self._hidden_width) and batch > 0
                 and hidden.device == self.query_tf_coordinates.device
                 and bool(torch.isfinite(hidden).all()), "HIDDEN")
        _require(isinstance(query_k, torch.Tensor) and query_k.dtype == torch.int64
                 and query_k.ndim == 3 and query_k.shape[0] == batch
                 and 1 <= query_k.shape[1] <= MAX_QUERY_BLOCK
                 and query_k.shape[2] == len(TIMEFRAMES)
                 and query_k.device == hidden.device
                 and bool(((query_k >= 1)
                           & (query_k <= self.query_native_k[None, None, :])).all()),
                 "QUERY_K")
        blocks = query_k.shape[1]
        source = hidden[:, None, None, :].expand(-1, blocks, len(TIMEFRAMES), -1)
        tf = self.query_tf_coordinates[None, None].expand(batch, blocks, -1, -1)
        fraction = (query_k.to(torch.float32) /
                    self.query_native_k.to(torch.float32)[None, None, :]).unsqueeze(-1)
        logits = self.query_head(self.query_projection(torch.cat((source, tf, fraction), dim=-1)))
        _require(logits.shape == (batch, blocks, len(TIMEFRAMES), len(CLASS_NAMES))
                 and bool(torch.isfinite(logits).all()), "LOGITS")
        return logits

    def forward(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *, query_k: torch.Tensor,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor, gc_seq_m15: torch.Tensor, gc_seq_h1: torch.Tensor,
        gc_seq_h4: torch.Tensor, gc_seq_d1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_features(
            seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4,
            seq_d1=seq_d1, gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15,
            gc_seq_h1=gc_seq_h1, gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1)
        return {"logits": self.decode_queries(hidden, query_k)}
