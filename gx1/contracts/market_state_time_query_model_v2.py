"""Two-logit activity and conditional-side head for future state queries.

The v1 three-logit checkpoint and model contract remain untouched. This
research-only model emits primitive logits for the v2 masked loss and derived
three-class log probabilities for the existing proper-score interpretation.
"""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F
from torch import nn

from gx1.contracts.entry_native_direction_research_v1 import _backbone_width
from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    CLASS_NAMES, five_endpoint_contract,
)
from gx1.contracts.market_state_time_query_loss_v2 import (
    factorized_time_query_loss_contract,
)
from gx1.contracts.market_state_time_query_model_v1 import (
    MAX_QUERY_BLOCK, TimeQueryMarketStateModel, _require,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME, FULL_TA_ENTRY_EXECUTION_PROFILE_NAME,
    FULL_TA_ENTRY_FEATURE_KEYS, compile_full_ta_entry_execution_profile,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer


def factorized_time_query_model_contract(
    *, ta_only: bool = False, dedicated_m5_side: bool = False,
) -> dict:
    if type(ta_only) is not bool:
        raise ValueError("TIME_QUERY_TA_MODE")
    if type(dedicated_m5_side) is not bool or (dedicated_m5_side and not ta_only):
        raise ValueError("TIME_QUERY_DEDICATED_M5_SIDE_MODE")
    value = {
        "schema_version": "gx1_market_state_time_query_model_v2",
        "role": "research_only_no_entry_exit_or_trade_authority",
        "target_contract_sha256": five_endpoint_contract()["contract_sha256"],
        "loss_contract_sha256": factorized_time_query_loss_contract()["contract_sha256"],
        "timeframes": list(TIMEFRAMES),
        "native_k": [int(k) for k in NATIVE_K],
        "future_validity": "closed_M5_issue_plus_5k_UTC_minutes",
        "class_order": list(CLASS_NAMES),
        "backbone": "native_entry_q_joint_hidden_research_representation_only_without_detach",
        "query": "five_timeframe_one_hot_plus_preselected_k_over_native_k",
        "head": "fresh_shared_Linear_hidden_plus_6_to_hidden_GELU_then_Linear_hidden_to_2",
        "primitive_logit_order": ["ACTIVE", "UP_GIVEN_ACTIVE"],
        "output": "primitive_logits_batch_5_2_and_factorized_class_log_probabilities_batch_5_3",
        "class_probability_factorization": {"DOWN": "active*(1-up_given_active)", "FLAT": "1-active", "UP": "active*up_given_active"},
        "backbone_reuse": "encode_features_once_decode_queries_or_bounded_grid_many_times",
        "dense_query_block": "decode_query_grid_Q_1_to_32_class_log_probabilities_B_Q_5_3",
        "current_state_or_future_target_as_forward_input": False,
        "old_checkpoint_compatible": False,
    }
    if dedicated_m5_side:
        value["dedicated_m5_side"] = True
        value["head"] = (
            "shared_Linear_hidden_plus_6_to_hidden_GELU_then_Linear_hidden_to_2"
            "_with_dedicated_M5_conditional_side_Linear_hidden_to_1")
        value["M5_side_initialization"] = "clone_shared_side_row_preserving_CPU_and_active_CUDA_RNG"
    if ta_only:
        profile = compile_full_ta_entry_execution_profile()
        value["input_profile"] = {key: profile[key] for key in ("name", "contract_sha256")}
        value["feature_keys"] = sorted(FULL_TA_ENTRY_FEATURE_KEYS)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def factorized_log_probabilities(raw_logits: torch.Tensor) -> torch.Tensor:
    """Map [activity, conditional UP] to DOWN, FLAT, UP log probabilities."""
    if not (isinstance(raw_logits, torch.Tensor)
            and raw_logits.dtype == torch.float32
            and raw_logits.ndim in (3, 4)
            and raw_logits.shape[-2:] == (len(TIMEFRAMES), 2)
            and raw_logits.shape[0] > 0
            and bool(torch.isfinite(raw_logits).all())):
        raise ValueError("TIME_QUERY_FACTORIZED_LOGITS")
    activity = raw_logits[..., 0]
    up_given_active = raw_logits[..., 1]
    log_active = F.logsigmoid(activity)
    return torch.stack((
        log_active + F.logsigmoid(-up_given_active),
        F.logsigmoid(-activity),
        log_active + F.logsigmoid(up_given_active),
    ), dim=-1)


def _clone_side_head(head: nn.Linear) -> nn.Linear:
    """Clone only conditional side without shifting existing head initialization."""
    device = head.weight.device
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        result = nn.Linear(head.in_features, 1, device=device, dtype=torch.float32)
    with torch.no_grad():
        result.weight.copy_(head.weight[1:2])
        result.bias.copy_(head.bias[1:2])
    return result


class FactorizedTimeQueryMarketStateModel(TimeQueryMarketStateModel):
    """The v1 causal encoding and query domain with a distinct two-logit head."""

    def __init__(self, backbone: EntryV10CtxHybridTransformer, *,
                 dedicated_m5_side: bool = False):
        # Build the distinct v2 head directly; do not instantiate an unused v1
        # three-logit head or consume its initialization randomness.
        nn.Module.__init__(self)
        width = _backbone_width(backbone)
        self._ta_only = (isinstance(backbone.entry_execution_profile, dict)
                         and backbone.entry_execution_profile.get("name")
                         == FULL_TA_ENTRY_EXECUTION_PROFILE_NAME)
        _require(type(dedicated_m5_side) is bool
                 and (not dedicated_m5_side or self._ta_only),
                 "DEDICATED_M5_SIDE_MODE")
        self._dedicated_m5_side = dedicated_m5_side
        self._require_input_profile(backbone)
        self.backbone = backbone
        self._hidden_width = width
        device = backbone.entry_q_joint_in.weight.device
        self.query_projection = nn.Sequential(
            nn.Linear(width + len(TIMEFRAMES) + 1, width, device=device, dtype=torch.float32),
            nn.GELU())
        self.query_head = nn.Linear(width, 2, device=device, dtype=torch.float32)
        if self._dedicated_m5_side:
            self.m5_side_head = _clone_side_head(self.query_head)
        self.register_buffer("query_tf_coordinates", torch.eye(
            len(TIMEFRAMES), device=device, dtype=torch.float32))
        self.register_buffer("query_native_k", torch.tensor(
            [int(k) for k in NATIVE_K], device=device, dtype=torch.int64))
        self.register_buffer("query_objective_sha256", torch.tensor(
            list(bytes.fromhex(self.research_objective["contract_sha256"])),
            device=device, dtype=torch.uint8))

    @property
    def research_objective(self) -> dict:
        return factorized_time_query_model_contract(
            ta_only=self._ta_only, dedicated_m5_side=self._dedicated_m5_side)

    def _require_input_profile(self, backbone) -> None:
        if self._ta_only:
            profile = compile_full_ta_entry_execution_profile()
            _require(backbone.entry_execution_profile == {
                key: profile[key] for key in ("name", "contract_sha256")}
                and backbone.gc_mtf_history is None, "FULL_TA_PROFILE")
        else:
            _require(isinstance(backbone.entry_execution_profile, dict)
                     and backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
                     and backbone.gc_mtf_history is not None
                     and "relative_price" not in backbone.gc_mtf_history["normalization"], "FULL_GC_PROFILE")

    def _require_binding(self) -> None:
        _require(type(self._dedicated_m5_side) is bool
                 and (not self._dedicated_m5_side or self._ta_only),
                 "DEDICATED_M5_SIDE_MODE")
        _require(hasattr(self, "m5_side_head") == self._dedicated_m5_side,
                 "M5_SIDE_HEAD_BINDING")
        width = _backbone_width(self.backbone)
        _require(width == self._hidden_width, "HIDDEN_WIDTH")
        self._require_input_profile(self.backbone)
        device = self.backbone.entry_q_joint_in.weight.device
        expected = {
            "query_tf_coordinates": torch.eye(
                len(TIMEFRAMES), device=device, dtype=torch.float32),
            "query_native_k": torch.tensor(
                [int(k) for k in NATIVE_K], device=device, dtype=torch.int64),
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
        layers = [
            (self.query_projection[0], width + len(TIMEFRAMES) + 1, width),
            (self.query_head, width, 2),
        ]
        if self._dedicated_m5_side:
            layers.append((self.m5_side_head, width, 1))
        for layer, inputs, outputs in layers:
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
        seq_h4: torch.Tensor, seq_d1: torch.Tensor, **gc_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self._ta_only:
            return super().encode_features(
                seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
                seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1,
                seq_h4=seq_h4, seq_d1=seq_d1, **gc_features)
        self._require_binding()
        _require(not gc_features, "TA_GC_FEATURES_FORBIDDEN")
        features = dict(seq_x=seq_x, snap_x=snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
                        seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1,
                        seq_h4=seq_h4, seq_d1=seq_d1)
        _require(set(features) == FULL_TA_ENTRY_FEATURE_KEYS
                 and isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3
                 and seq_x.shape[0] > 0, "TA_FEATURE_BATCH")
        for name, value in features.items():
            rank = 3 if name == "seq_x" or name.startswith("seq_") else 2
            integer = name == "ctx_cat"
            _require(isinstance(value, torch.Tensor) and value.ndim == rank
                     and value.shape[0] == seq_x.shape[0]
                     and all(size > 0 for size in value.shape)
                     and value.device == self.query_tf_coordinates.device
                     and (value.dtype in (torch.int8, torch.int16, torch.int32,
                                          torch.int64, torch.uint8) if integer
                          else value.dtype == torch.float32 and bool(torch.isfinite(value).all())),
                     "FEATURE:" + name)
        output = self.backbone(**features, research_representation_only=True)
        _require(isinstance(output, dict), "BACKBONE_OUTPUT")
        hidden = output.get("entry_q_joint_hidden")
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
                 and hidden.shape == (seq_x.shape[0], self._hidden_width)
                 and hidden.device == self.query_tf_coordinates.device
                 and bool(torch.isfinite(hidden).all()), "BACKBONE_HIDDEN")
        return hidden

    def decode_raw_query_grid(
        self, hidden: torch.Tensor, query_k: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one bounded query block into the two primitive head logits."""
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
        representation = self.query_projection(torch.cat((source, tf, fraction), dim=-1))
        raw = self.query_head(representation)
        if self._dedicated_m5_side:
            raw = raw.clone()
            raw[..., 0, 1] = self.m5_side_head(representation[..., 0, :]).squeeze(-1)
        _require(raw.shape == (batch, blocks, len(TIMEFRAMES), 2)
                 and bool(torch.isfinite(raw).all()), "LOGITS")
        return raw

    def decode_query_grid(
        self, hidden: torch.Tensor, query_k: torch.Tensor,
    ) -> torch.Tensor:
        """Return derived class log probabilities for grid scoring."""
        return factorized_log_probabilities(self.decode_raw_query_grid(hidden, query_k))

    def forward(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *, query_k: torch.Tensor,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        **gc_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_features(
            seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4,
            seq_d1=seq_d1, **gc_features)
        _require(isinstance(query_k, torch.Tensor) and query_k.ndim == 2, "QUERY_K")
        raw = self.decode_raw_query_grid(hidden, query_k[:, None, :])[:, 0, :, :]
        return {"activity_side_logits": raw,
                "logits": factorized_log_probabilities(raw)}
