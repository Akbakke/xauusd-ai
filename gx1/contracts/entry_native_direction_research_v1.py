"""Pure-direction research readout over the unchanged native Entry forward.

This component is not a native training phase, serving bundle or action owner.
The caller must apply the existing FP32/MHA numerical policy to ``backbone``;
applying a phase-sensitive trainer context only to this wrapper is insufficient.
Fresh complete ONLINE/optimizer/RNG state, prefix normalization, real native/GC
consumption and exact resume require separate measured qualification. Metadata
and synthetic forward spies confer none of those qualifications.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)

SCHEMA_VERSION = "gx1_entry_native_direction_research_v1"
HORIZONS_MINUTES = (1, 5, 15)
_MINUTE_NS = 60 * 10**9
_MAX_NS = 2**63 - 1
_OBJECTIVE = {
    "schema_version": SCHEMA_VERSION,
    "research_objective": "pure_observed_nonzero_midpoint_direction",
    "research_role": "research_only_no_native_phase_or_action_authority",
    "backbone_class": "gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer.EntryV10CtxHybridTransformer",
    "backbone_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
    "representation": "entry_q_joint_hidden_without_detach",
    "head": "fresh_FP32_Linear_bound_native_hidden_width_to_three_horizon_logits",
    "head_initialization": "torch_nn_Linear_default_reset_parameters_caller_seeded",
    "horizons_minutes": list(HORIZONS_MINUTES),
    "binary_class_order": ["DOWN", "UP"],
    "probability_condition": "observed_nonzero_price_direction",
    "hard_prediction": "UP_iff_logit_gt_zero_else_DOWN",
    "loss": "unweighted_BCE_with_logits_sum_divided_by_eligible_cells",
    "eligible": "observed_and_sign_nonzero_and_terminal_time_strictly_before_cutoff",
    "empty_eligible": "loss_None_no_optimizer_EMA_or_scheduler_step",
    "future_targets_or_clocks_as_forward_inputs": False,
    "native_forwards_per_research_forward": 1,
    "inactive_backbone_outputs": [
        "dynamic_terminal_net_markout_bps", "dynamic_mfe_bps",
        "dynamic_pre_mfe_mae_upper_bps", "dynamic_closed_mfe_bps",
        "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred",
        "legacy_action_and_Exit_outputs",
    ],
    "backbone_path_forecast_and_task_variance_supervised": False,
    "numerical_policy": "caller_applies_existing_FP32_and_MHA_policy_to_bound_backbone",
    "fresh_initial_gradient_resume_and_prefix_data_qualification": "required_separately_before_genuine_fit",
    "native_export_authorized": False,
    "native_launch_authorized": False,
    "economic_edge_claimed": False,
}
_OBJECTIVE_JSON = json.dumps(_OBJECTIVE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
_OBJECTIVE_SHA256 = hashlib.sha256(_OBJECTIVE_JSON).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError("NATIVE_DIRECTION_RESEARCH_" + reason)


def native_direction_research_contract() -> dict[str, Any]:
    """Return a copy; mutating it cannot alter the canonical research identity."""
    return {**json.loads(_OBJECTIVE_JSON), "contract_sha256": _OBJECTIVE_SHA256}


def _backbone_width(backbone: nn.Module) -> int:
    _require(type(backbone) is EntryV10CtxHybridTransformer, "EXACT_NATIVE_TYPE")
    cfg = getattr(backbone, "cfg", None)
    _require(type(cfg) is CtxModelConfig
        and backbone.entry_training_phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        and cfg.entry_training_phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, "BACKBONE_PHASE")
    width = cfg.d_model
    joint = getattr(backbone, "entry_q_joint_in", None)
    _require(type(width) is int and width > 0 and type(joint) is nn.Linear
        and joint.in_features == 4 * width and joint.out_features == width
        and tuple(joint.weight.shape) == (width, 4 * width)
        and joint.bias is not None and tuple(joint.bias.shape) == (width,), "NATIVE_HIDDEN_WIDTH")
    _require(joint.weight.dtype == joint.bias.dtype == torch.float32
        and joint.weight.device == joint.bias.device and joint.weight.device.type != "meta", "BACKBONE_DTYPE_DEVICE")
    _require(bool(torch.isfinite(joint.weight).all()) and bool(torch.isfinite(joint.bias).all()), "BACKBONE_FINITE")
    return width


class NativeEntryDirectionResearch(nn.Module):
    """One native forward followed by a separate, undetached research readout.

    No ``entry_training_phase`` is advertised for this wrapper: the unchanged
    backbone's old phase is not this research objective. The caller applies
    the existing FP32/MHA context to ``self.backbone``. Unused native decoder,
    forecast and task-variance parameters receive no supervision here; their
    grad=None/no-update state must be checked by genuine qualification.
    """

    def __init__(self, backbone: EntryV10CtxHybridTransformer):
        super().__init__()
        width = _backbone_width(backbone)
        self.backbone = backbone
        self._bound_hidden_width = width
        self._research_contract_json = _OBJECTIVE_JSON
        self.direction_head = nn.Linear(width, len(HORIZONS_MINUTES),
            device=backbone.entry_q_joint_in.weight.device, dtype=torch.float32)
        self.register_buffer("direction_horizon_minutes", torch.tensor(HORIZONS_MINUTES,
            dtype=torch.int64, device=backbone.entry_q_joint_in.weight.device))

    @property
    def research_objective(self) -> dict[str, Any]:
        _require(self._research_contract_json == _OBJECTIVE_JSON, "RESEARCH_OBJECTIVE_IDENTITY")
        return native_direction_research_contract()

    def _require_binding(self) -> None:
        _require(self._research_contract_json == _OBJECTIVE_JSON, "RESEARCH_OBJECTIVE_IDENTITY")
        width = _backbone_width(self.backbone)
        _require(type(self._bound_hidden_width) is int and self._bound_hidden_width == width, "BOUND_HIDDEN_WIDTH")
        head = self.direction_head
        _require(type(head) is nn.Linear and head.in_features == width and head.out_features == 3
            and tuple(head.weight.shape) == (3, width) and head.bias is not None and tuple(head.bias.shape) == (3,), "HEAD_SHAPE")
        _require(head.weight.dtype == head.bias.dtype == torch.float32
            and head.weight.device == head.bias.device == self.backbone.entry_q_joint_in.weight.device, "HEAD_DTYPE_DEVICE")
        _require(bool(torch.isfinite(head.weight).all()) and bool(torch.isfinite(head.bias).all()), "HEAD_FINITE")
        horizons = self.direction_horizon_minutes
        _require(isinstance(horizons, torch.Tensor) and horizons.dtype == torch.int64
            and tuple(horizons.shape) == (3,) and horizons.device == head.weight.device
            and torch.equal(horizons, torch.tensor(HORIZONS_MINUTES, dtype=torch.int64, device=horizons.device)), "HORIZON_IDENTITY")

    def forward(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor | None = None, gc_seq_m15: torch.Tensor | None = None,
        gc_seq_h1: torch.Tensor | None = None, gc_seq_h4: torch.Tensor | None = None,
        gc_seq_d1: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._require_binding()
        _require(isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3 and seq_x.shape[0] > 0, "FEATURE_BATCH")
        output = self.backbone(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
            gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15, gc_seq_h1=gc_seq_h1,
            gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1)
        _require(isinstance(output, dict), "NATIVE_OUTPUT")
        hidden = output.get("entry_q_joint_hidden")
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
            and tuple(hidden.shape) == (seq_x.shape[0], self._bound_hidden_width)
            and hidden.device == self.direction_head.weight.device, "NATIVE_HIDDEN")
        _require(bool(torch.isfinite(hidden).all()), "NATIVE_HIDDEN_FINITE")
        logits = self.direction_head(hidden)
        _require(logits.dtype == torch.float32 and tuple(logits.shape) == (seq_x.shape[0], 3)
            and bool(torch.isfinite(logits).all()), "FINITE_LOGITS")
        # Legacy curve/forecast outputs leave scope; they are not research predictions.
        return {"logit_up": logits, "horizons_minutes": HORIZONS_MINUTES}


def native_direction_binary_loss(
    logit_up: torch.Tensor, *, sign: torch.Tensor, observed: torch.Tensor,
    decision_ns: torch.Tensor, terminal_time_ns: torch.Tensor, cutoff_ns: int,
) -> dict[str, Any]:
    """Train only own mature nonzero labels; targets never enter model forward.

    An empty mask returns loss=None/update_allowed=False. The caller may
    advance its input cursor, but must not step optimizer, EMA or scheduler.
    """
    _require(isinstance(logit_up, torch.Tensor) and logit_up.dtype == torch.float32
        and logit_up.ndim == 2 and logit_up.shape[0] > 0 and logit_up.shape[1] == 3, "LOGIT_SHAPE_DTYPE")
    size = logit_up.shape[0]
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= _MAX_NS, "CUTOFF")
    _require(isinstance(sign, torch.Tensor) and sign.dtype == torch.int8 and tuple(sign.shape) == (size, 3), "SIGN_SHAPE_DTYPE")
    _require(isinstance(observed, torch.Tensor) and observed.dtype == torch.bool and tuple(observed.shape) == (size, 3), "OBSERVED_SHAPE_DTYPE")
    _require(isinstance(decision_ns, torch.Tensor) and decision_ns.dtype == torch.int64 and tuple(decision_ns.shape) == (size,), "DECISION_SHAPE_DTYPE")
    _require(isinstance(terminal_time_ns, torch.Tensor) and terminal_time_ns.dtype == torch.int64
        and tuple(terminal_time_ns.shape) == (size, 3), "TERMINAL_SHAPE_DTYPE")
    _require(all(value.device == logit_up.device for value in (sign, observed, decision_ns, terminal_time_ns)), "DEVICE")
    _require(bool(torch.isfinite(logit_up).all()), "FINITE_LOGITS")
    _require(bool(torch.isin(sign, torch.tensor((-1, 0, 1), dtype=torch.int8, device=sign.device)).all())
        and bool((sign[~observed] == 0).all()), "SIGN_DOMAIN_OR_MISSING_STORAGE")
    _require(bool((decision_ns > 0).all()) and bool((decision_ns < cutoff_ns).all())
        and bool((decision_ns <= _MAX_NS - max(HORIZONS_MINUTES) * _MINUTE_NS).all()), "DECISION_CUTOFF_OR_OVERFLOW")
    expected = decision_ns[:, None] + torch.tensor(HORIZONS_MINUTES, dtype=torch.int64, device=decision_ns.device)[None, :] * _MINUTE_NS
    _require(torch.equal(terminal_time_ns, expected), "EXACT_TERMINAL_CLOCK")
    mature = terminal_time_ns < cutoff_ns
    nonzero = sign != 0
    eligible = observed & nonzero & mature
    masks = {
        "cells": torch.ones_like(observed), "observed_cells": observed, "missing_cells": ~observed,
        "observed_up_cells": observed & (sign == 1), "observed_down_cells": observed & (sign == -1),
        "observed_tie_cells": observed & ~nonzero, "mature_cells": mature, "immature_cells": ~mature,
        "eligible_cells": eligible, "eligible_up_cells": eligible & (sign == 1), "eligible_down_cells": eligible & (sign == -1),
        "immature_observed_nonzero_cells": observed & nonzero & ~mature,
        "mature_tie_cells": observed & ~nonzero & mature, "immature_tie_cells": observed & ~nonzero & ~mature,
        "mature_missing_cells": ~observed & mature, "immature_missing_cells": ~observed & ~mature,
    }
    # One small device-to-host transfer for all count surfaces, not one per cell.
    counted = torch.stack([value.sum(dim=0) for value in masks.values()]).cpu().tolist()
    per_horizon = [{"horizon_minutes": h, "parent_rows": size,
        **{name: int(counted[j][i]) for j, name in enumerate(masks)}} for i, h in enumerate(HORIZONS_MINUTES)]
    counts = {"parent_rows": size, **{name: sum(row[name] for row in per_horizon) for name in masks}}
    _require(counts["eligible_cells"] + counts["observed_tie_cells"] + counts["immature_observed_nonzero_cells"]
        + counts["missing_cells"] == counts["cells"] == size * 3, "COUNT_PARTITION")
    count = counts["eligible_cells"]
    loss = None
    if count:
        labels = (sign[eligible] == 1).to(dtype=logit_up.dtype)
        loss = F.binary_cross_entropy_with_logits(logit_up[eligible], labels, reduction="sum") / count
        _require(bool(torch.isfinite(loss)) and bool(loss >= 0), "FINITE_LOSS")
    return {"loss": loss, "update_allowed": count > 0, "eligible_mask": eligible,
        "counts": counts, "per_horizon_counts": per_horizon,
        "research_objective_sha256": _OBJECTIVE_SHA256}
