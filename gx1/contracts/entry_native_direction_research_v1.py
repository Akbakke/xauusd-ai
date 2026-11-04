"""Pure-direction research readout over the unchanged native Entry forward.

This component is not a native training phase, serving bundle or action owner.
The caller must apply the existing FP32/MHA numerical policy to ``backbone``;
applying a phase-sensitive trainer context only to this wrapper is insufficient.
Fresh complete ONLINE/optimizer/RNG state, prefix normalization, real native/GC
consumption and exact resume require separate measured qualification. Metadata
and synthetic forward spies confer none of those qualifications.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
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
_BPS_OBJECTIVE = {**_OBJECTIVE,
    "schema_version": "gx1_entry_native_direction_with_signed_bps_v1",
    "research_objective": "observed_nonzero_direction_plus_observed_signed_midpoint_bps",
    "head": "original_direction_Linear_plus_zero_initialized_normalized_mean_Linear",
    "bps_head_initialization": "zero_weight_and_bias_preserving_CPU_and_active_CUDA_RNG",
    "bps_prediction": "unbounded_normalized_mean_times_frozen_own_prefix_RMS_bps",
    "loss": "original_direction_BCE_plus_observed_mature_signed_bps_normalized_MSE",
    "bps_relative_weight": 1.0,
    "bps_eligible": "observed_and_terminal_time_strictly_before_cutoff_including_zero",
    "bps_scale": "own_prefix_float32_target_RMS_in_float64_no_center_or_fallback",
    "bps_uncertainty_estimated": False,
    "empty_eligible": "no_observed_mature_cells_loss_None_no_optimizer_step"}
_BPS_OBJECTIVE_JSON = json.dumps(_BPS_OBJECTIVE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
_BPS_OBJECTIVE_SHA256 = hashlib.sha256(_BPS_OBJECTIVE_JSON).hexdigest()
_MAG_WEIGHT_OBJECTIVE = {**_OBJECTIVE,
    "schema_version": "gx1_entry_native_direction_magnitude_weighted_v1",
    "research_objective": "own_train_prefix_bounded_magnitude_weighted_direction",
    "probability_condition": "magnitude_tilted_sign_score_not_unweighted_probability",
    "loss": "bounded_own_prefix_normalized_magnitude_weighted_BCE_sum_divided_by_eligible_cells",
    "weight": "abs_midpoint_bps_over_own_prefix_RMS_plus_abs_bps_divided_by_eligible_prefix_mean",
    "future_targets_or_clocks_as_forward_inputs": False}
_MAG_WEIGHT_OBJECTIVE_JSON = json.dumps(_MAG_WEIGHT_OBJECTIVE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError("NATIVE_DIRECTION_RESEARCH_" + reason)


def native_direction_research_contract(*, signed_bps: bool = False, magnitude_weighted: bool = False) -> dict[str, Any]:
    """Return a copy; mutating it cannot alter the canonical research identity."""
    _require(type(signed_bps) is bool and type(magnitude_weighted) is bool
        and not (signed_bps and magnitude_weighted), "OBJECTIVE_MODE")
    raw = _BPS_OBJECTIVE_JSON if signed_bps else _MAG_WEIGHT_OBJECTIVE_JSON if magnitude_weighted else _OBJECTIVE_JSON
    return {**json.loads(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()}


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

    def __init__(self, backbone: EntryV10CtxHybridTransformer, *, bps_scale=None, magnitude_weight_contract=None):
        super().__init__()
        _require(not (bps_scale is not None and magnitude_weight_contract is not None), "OBJECTIVE_MODE")
        width = _backbone_width(backbone)
        self.backbone = backbone
        self._bound_hidden_width = width
        self._research_contract_json = (_BPS_OBJECTIVE_JSON if bps_scale is not None else
            _MAG_WEIGHT_OBJECTIVE_JSON if magnitude_weight_contract is not None else _OBJECTIVE_JSON)
        self.direction_head = nn.Linear(width, len(HORIZONS_MINUTES),
            device=backbone.entry_q_joint_in.weight.device, dtype=torch.float32)
        self.register_buffer("direction_horizon_minutes", torch.tensor(HORIZONS_MINUTES,
            dtype=torch.int64, device=backbone.entry_q_joint_in.weight.device))
        if bps_scale is not None:
            from gx1.contracts.entry_native_direction_bps_v1 import require_direction_bps_scale
            scale = require_direction_bps_scale(bps_scale)
            self._bps_scale_json = json.dumps(scale, sort_keys=True, separators=(",", ":"), allow_nan=False)
            device = self.direction_head.weight.device
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                self.bps_mean_head = nn.Linear(width, 3, device=device, dtype=torch.float32)
                nn.init.zeros_(self.bps_mean_head.weight)
                nn.init.zeros_(self.bps_mean_head.bias)
            self.register_buffer("bps_rms", torch.tensor(scale["rms_bps"], dtype=torch.float32, device=device))
            self.register_buffer("bps_scale_sha256", torch.tensor(list(bytes.fromhex(scale["contract_sha256"])),
                dtype=torch.uint8, device=device))
        if magnitude_weight_contract is not None:
            from gx1.contracts.entry_native_direction_bps_v1 import require_direction_magnitude_weight_contract
            weight = require_direction_magnitude_weight_contract(magnitude_weight_contract)
            self._magnitude_weight_json = json.dumps(weight, sort_keys=True, separators=(",", ":"), allow_nan=False)
            device = self.direction_head.weight.device
            self.register_buffer("magnitude_rms_bps", torch.tensor(weight["base_scale"]["rms_bps"], dtype=torch.float32, device=device))
            self.register_buffer("magnitude_raw_weight_mean", torch.tensor(weight["raw_weight_means"], dtype=torch.float32, device=device))
            self.register_buffer("magnitude_weight_sha256", torch.tensor(list(bytes.fromhex(weight["contract_sha256"])),
                dtype=torch.uint8, device=device))

    @property
    def signed_bps(self) -> bool:
        return self._research_contract_json == _BPS_OBJECTIVE_JSON

    @property
    def magnitude_weighted(self) -> bool:
        return self._research_contract_json == _MAG_WEIGHT_OBJECTIVE_JSON

    @property
    def magnitude_weight_contract(self):
        _require(self.magnitude_weighted, "MAG_WEIGHT_MODE_REQUIRED")
        return json.loads(self._magnitude_weight_json)

    @property
    def bps_scale(self):
        _require(self.signed_bps, "BPS_MODE_REQUIRED")
        return json.loads(self._bps_scale_json)

    @property
    def research_objective(self) -> dict[str, Any]:
        expected = _BPS_OBJECTIVE_JSON if self.signed_bps else _MAG_WEIGHT_OBJECTIVE_JSON if self.magnitude_weighted else _OBJECTIVE_JSON
        _require(self._research_contract_json == expected, "RESEARCH_OBJECTIVE_IDENTITY")
        return native_direction_research_contract(signed_bps=self.signed_bps, magnitude_weighted=self.magnitude_weighted)

    def _require_binding(self) -> None:
        expected = _BPS_OBJECTIVE_JSON if self.signed_bps else _MAG_WEIGHT_OBJECTIVE_JSON if self.magnitude_weighted else _OBJECTIVE_JSON
        _require(self._research_contract_json == expected, "RESEARCH_OBJECTIVE_IDENTITY")
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
        if self.signed_bps:
            from gx1.contracts.entry_native_direction_bps_v1 import require_direction_bps_scale
            scale = require_direction_bps_scale(self.bps_scale)
            mean = self.bps_mean_head
            _require(type(mean) is nn.Linear and mean.in_features == width and mean.out_features == 3
                and mean.bias is not None and mean.weight.shape == head.weight.shape
                and mean.bias.shape == head.bias.shape and mean.weight.dtype == mean.bias.dtype == torch.float32
                and mean.weight.device == mean.bias.device == head.weight.device
                and bool(torch.isfinite(mean.weight).all()) and bool(torch.isfinite(mean.bias).all()), "BPS_HEAD")
            _require(self.bps_rms.dtype == torch.float32 and self.bps_rms.device == head.weight.device
                and not self.bps_rms.requires_grad
                and torch.equal(self.bps_rms, torch.tensor(scale["rms_bps"], dtype=torch.float32, device=head.weight.device))
                and self.bps_scale_sha256.dtype == torch.uint8 and self.bps_scale_sha256.device == head.weight.device
                and torch.equal(self.bps_scale_sha256, torch.tensor(list(bytes.fromhex(scale["contract_sha256"])),
                    dtype=torch.uint8, device=head.weight.device)), "BPS_SCALE_BINDING")
        else:
            _require(not any(hasattr(self, name) for name in ("bps_mean_head", "bps_rms", "bps_scale_sha256", "_bps_scale_json")),
                "BASELINE_HAS_BPS_STATE")
        if self.magnitude_weighted:
            from gx1.contracts.entry_native_direction_bps_v1 import require_direction_magnitude_weight_contract
            weight = require_direction_magnitude_weight_contract(self.magnitude_weight_contract)
            _require(self.magnitude_rms_bps.dtype == torch.float32 and self.magnitude_rms_bps.device == head.weight.device
                and not self.magnitude_rms_bps.requires_grad
                and torch.equal(self.magnitude_rms_bps, torch.tensor(weight["base_scale"]["rms_bps"], dtype=torch.float32, device=head.weight.device))
                and self.magnitude_raw_weight_mean.dtype == torch.float32 and self.magnitude_raw_weight_mean.device == head.weight.device
                and not self.magnitude_raw_weight_mean.requires_grad
                and torch.equal(self.magnitude_raw_weight_mean, torch.tensor(weight["raw_weight_means"], dtype=torch.float32, device=head.weight.device))
                and self.magnitude_weight_sha256.dtype == torch.uint8 and self.magnitude_weight_sha256.device == head.weight.device
                and torch.equal(self.magnitude_weight_sha256, torch.tensor(list(bytes.fromhex(weight["contract_sha256"])),
                    dtype=torch.uint8, device=head.weight.device)), "MAG_WEIGHT_BINDING")
        else:
            _require(not any(hasattr(self, name) for name in ("magnitude_rms_bps", "magnitude_raw_weight_mean",
                "magnitude_weight_sha256", "_magnitude_weight_json")), "UNWEIGHTED_HAS_MAG_STATE")

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
        result = {"logit_up": logits, "horizons_minutes": HORIZONS_MINUTES}
        if self.signed_bps:
            mean_bps = self.bps_mean_head(hidden) * self.bps_rms
            _require(mean_bps.dtype == torch.float32 and mean_bps.shape == logits.shape
                and bool(torch.isfinite(mean_bps).all()), "FINITE_BPS_MEAN")
            result["mean_return_bps"] = mean_bps
        return result


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


@lru_cache(maxsize=4, typed=True)
def _dynamic_mtf_state_objective_bytes(*, current_state_input=False, first_state_loss_weight=None,
        extended_forecast_minutes=None, first_flat_activity_weight=None, native_endpoint_loss=False):
    _require(type(current_state_input) is bool, "STATE_CURRENT_INPUT_MODE")
    _require((not current_state_input and first_state_loss_weight is None)
        or (current_state_input and type(first_state_loss_weight) is float
            and math.isfinite(first_state_loss_weight) and first_state_loss_weight > 0), "STATE_FIRST_LOSS_WEIGHT")
    _require(first_flat_activity_weight is None or (current_state_input and type(first_flat_activity_weight) is float
        and math.isfinite(first_flat_activity_weight) and first_flat_activity_weight > 0), "STATE_FIRST_ACTIVITY_WEIGHT")
    _require(type(native_endpoint_loss) is bool and (not native_endpoint_loss or
        (first_flat_activity_weight is not None and extended_forecast_minutes is None)),
        "STATE_NATIVE_ENDPOINT_MODE")
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
    target = targets.dynamic_mtf_state_target_contract()
    _require(extended_forecast_minutes is None or current_state_input, "STATE_EXTENDED_REQUIRES_CURRENT_INPUT")
    leads = (tuple(targets.default_trial_lead_minutes()) if extended_forecast_minutes is None
        else targets.extended_dense_lead_minutes(extended_forecast_minutes))
    _require(len(leads) > 1 and leads[0] == 1 and leads[-1] == len(leads), "STATE_QUERY_GRID")
    value = {
        "schema_version": "gx1_native_dynamic_mtf_state_research_v1",
        "research_objective": "joint_ordered_causal_local_mtf_state_forecast",
        "research_role": "research_only_no_native_phase_or_action_authority",
        "target_contract": target,
        "timeframes": list(targets.TIMEFRAMES),
        "class_names": list(targets.STATE_NAMES),
        "lead_minutes": list(leads),
        "forecast_domain": "all_declared_minutes_not_a_state_duration_or_holding_limit",
        "representation": "native_entry_q_joint_hidden_without_legacy_head_execution_or_detach",
        "query_coordinates": {
            "timeframe_encoding": "canonical_timeframe_one_hot",
            "lead_center_minutes": (leads[0] + leads[-1]) / 2.0,
            "lead_scale_minutes": math.sqrt((len(leads) ** 2 - 1) / 12.0),
            "statistics_source": "analytic_moments_of_declared_query_grid_no_data_fit"},
        "decoder": "shared_Linear_hidden_plus_query_to_hidden_GELU",
        "initial_head": "three_class_softmax_at_first_future_query",
        "transition_head": "three_by_three_row_softmax_at_each_later_future_query",
        "joint_distribution": "p_S1_times_product_P_Sj_given_Sjminus1_and_issue_features",
        "dependence_assumption": "first_order_between_declared_queries_conditional_on_issue_representation",
        "head_initialization": "torch_nn_Linear_default_reset_parameters_caller_seeded",
        "loss": "negative_log_probability_of_observed_state_sequence_marginalizing_unknown_states",
        "loss_normalization": "sum_row_timeframe_sequence_NLL_divided_by_all_eligible_observed_state_cells",
        "eligible": "observed_and_maturity_strictly_before_cutoff_including_FLAT",
        "empty_eligible": "loss_None_no_optimizer_EMA_or_scheduler_step",
        "state_probabilities": "forward_recursion_marginals_not_teacher_forced",
        "adjacent_pair_probabilities": "previous_marginal_times_conditional_transition",
        "future_targets_or_clocks_as_forward_inputs": False,
        "backbone_path_forecast_and_task_variance_supervised": False,
        "query_sampling": False,
        "query_evaluation": "all_timeframes_all_declared_leads",
        "numerical_policy": "caller_applies_existing_FP32_and_MHA_policy_to_bound_backbone",
        "native_export_authorized": False,
        "native_launch_authorized": False,
        "economic_edge_claimed": False,
    }
    if current_state_input:
        value.update({
            "schema_version": "gx1_native_dynamic_mtf_current_state_research_v2",
            "current_state_input": "per_tf_current_one_hot_plus_independent_state_and_path_observation_masks",
            "current_input_clock": "historical_event_time_only_online_delivery_not_qualified",
            "current_state_projection": "bias_free_FP32_Linear_5_to_hidden_added_per_timeframe",
            "decoder": "shared_Linear_hidden_plus_current_projection_plus_query_to_hidden_GELU",
            "loss": "normalized_observed_sequence_NLL_plus_weighted_first_observed_state_NLL",
            "first_state_loss_weight": first_state_loss_weight,
            "first_state_loss_normalization": "mean_over_observed_mature_first_query_cells",
            "current_input_available_online": False,
        })
    if extended_forecast_minutes is not None:
        value.update({
            "schema_version": "gx1_native_dynamic_mtf_extended_state_research_v3",
            "extended_forecast_minutes": extended_forecast_minutes,
            "target_trial_grid_role": "target_owner_original_grid_is_historical_default_not_query_limit",
            "extended_grid": "all_wall_minutes_from_one_through_declared_end_no_skipped_queries",
            "extended_runtime_and_storage_qualified": False,
        })
    if first_flat_activity_weight is not None:
        value.update({
            "schema_version": "gx1_native_dynamic_mtf_first_flat_activity_research_v4",
            "loss": "normalized_observed_sequence_NLL_plus_weighted_first_state_NLL_plus_weighted_first_flat_activity_NLL",
            "first_flat_activity_weight": first_flat_activity_weight,
            "first_flat_activity_target": "complete_local_range_lt_threshold_FLAT_else_nonFLAT_direction_may_be_unknown",
            "first_flat_activity_normalization": "mean_over_all_eligible_complete_first_query_local_windows",
            "first_flat_activity_is_future_label_only": True,
            "first_flat_activity_native_admission": False,
        })
    if native_endpoint_loss:
        value.update({
            "schema_version": "gx1_native_dynamic_mtf_endpoint_research_v5",
            "loss": value["loss"] + "_plus_native_endpoint_marginal_NLL",
            "native_endpoint_lead_minutes": [target["tf_span_ns"][tf] // targets.MINUTE_NS for tf in targets.TIMEFRAMES],
            "native_endpoint_weight": 1.0,
            "native_endpoint_normalization": "mean_over_observed_mature_native_cells_in_minibatch_omit_when_empty",
        })
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dynamic_mtf_state_research_contract(*, current_state_input=False, first_state_loss_weight=None,
                                        extended_forecast_minutes=None, first_flat_activity_weight=None,
                                        native_endpoint_loss=False) -> dict[str, Any]:
    """Separate target/output/loss identity; legacy objective bytes are untouched."""
    raw = _dynamic_mtf_state_objective_bytes(current_state_input=current_state_input,
        first_state_loss_weight=first_state_loss_weight, extended_forecast_minutes=extended_forecast_minutes,
        first_flat_activity_weight=first_flat_activity_weight, native_endpoint_loss=native_endpoint_loss)
    return {**json.loads(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()}


def encode_historical_current_state(current_state, current_observed, current_path_observed, *, batch, tfs, device):
    """Pure tensor encoding of already bound historical current-state evidence."""
    _require(type(batch) is int and batch > 0 and type(tfs) is int and tfs > 0, "CURRENT_SHAPE")
    _require(isinstance(current_state, torch.Tensor) and current_state.dtype == torch.int8
        and current_state.shape == (batch, tfs) and current_state.device == device, "CURRENT_STATE")
    for value in (current_observed, current_path_observed):
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.bool
            and value.shape == (batch, tfs) and value.device == device, "CURRENT_MASK")
    _require(bool((((current_state >= 0) & (current_state < 3) & current_observed)
        | ((current_state == -1) & ~current_observed)).all()), "CURRENT_STATE_DOMAIN")
    classes = torch.arange(3, device=device, dtype=current_state.dtype)
    one_hot = (current_state[..., None] == classes).to(dtype=torch.float32)
    return torch.cat((one_hot, current_observed[..., None].to(dtype=torch.float32),
        current_path_observed[..., None].to(dtype=torch.float32)), dim=-1)


def _state_query_coordinates(contract, *, device):
    timeframes, leads = contract["timeframes"], contract["lead_minutes"]
    coordinate = contract["query_coordinates"]
    lead = torch.tensor(leads, dtype=torch.float32, device=device)
    lead = (lead - coordinate["lead_center_minutes"]) / coordinate["lead_scale_minutes"]
    tf = torch.eye(len(timeframes), dtype=torch.float32, device=device)
    return torch.cat((tf[:, None, :].expand(-1, len(leads), -1),
        lead[None, :, None].expand(len(timeframes), -1, -1)), dim=-1)


class NativeDynamicMTFStateResearch(nn.Module):
    """One shared native representation and a joint future-state chain.

    True previous states select likelihood factors only. They never enter the
    decoder; scoring marginals integrate over the model's previous distribution.
    """

    def __init__(self, backbone: EntryV10CtxHybridTransformer, *, current_state_input=False,
                 first_state_loss_weight=None, extended_forecast_minutes=None, first_flat_activity_weight=None,
                 native_endpoint_loss=False):
        super().__init__()
        from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
        width = _backbone_width(backbone)
        contract = dynamic_mtf_state_research_contract(current_state_input=current_state_input,
            first_state_loss_weight=first_state_loss_weight, extended_forecast_minutes=extended_forecast_minutes,
            first_flat_activity_weight=first_flat_activity_weight, native_endpoint_loss=native_endpoint_loss)
        _require(isinstance(backbone.entry_execution_profile, dict)
            and backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
            and backbone.gc_mtf_history is not None
            and "relative_price" not in backbone.gc_mtf_history["normalization"], "STATE_FULL_RAW_GC_PROFILE")
        self.backbone = backbone
        self._bound_hidden_width = width
        self._current_state_input = current_state_input
        self._first_state_loss_weight = first_state_loss_weight
        self._extended_forecast_minutes = extended_forecast_minutes
        self._first_flat_activity_weight = first_flat_activity_weight
        self._native_endpoint_loss = native_endpoint_loss
        self._research_contract_json = _dynamic_mtf_state_objective_bytes(current_state_input=current_state_input,
            first_state_loss_weight=first_state_loss_weight, extended_forecast_minutes=extended_forecast_minutes,
            first_flat_activity_weight=first_flat_activity_weight, native_endpoint_loss=native_endpoint_loss)
        device = backbone.entry_q_joint_in.weight.device
        query_width = len(contract["timeframes"]) + 1
        classes = len(contract["class_names"])
        self.state_query_projection = nn.Sequential(
            nn.Linear(width + query_width, width, device=device, dtype=torch.float32), nn.GELU())
        self.state_initial_head = nn.Linear(width, classes, device=device, dtype=torch.float32)
        self.state_transition_head = nn.Linear(width, classes * classes, device=device, dtype=torch.float32)
        if current_state_input:
            self.state_current_projection = nn.Linear(5, width, bias=False, device=device, dtype=torch.float32)
        self.register_buffer("state_timeframe_codes", torch.arange(len(contract["timeframes"]), dtype=torch.int64, device=device))
        self.register_buffer("state_lead_minutes", torch.tensor(contract["lead_minutes"], dtype=torch.int64, device=device))
        self.register_buffer("state_class_codes", torch.arange(classes, dtype=torch.int64, device=device))
        self.register_buffer("state_query_coordinates", _state_query_coordinates(contract, device=device))
        self.register_buffer("state_objective_sha256", torch.tensor(
            list(bytes.fromhex(contract["contract_sha256"])), dtype=torch.uint8, device=device))

    @property
    def research_objective(self):
        kwargs = {"current_state_input": self._current_state_input, "first_state_loss_weight": self._first_state_loss_weight,
                  "extended_forecast_minutes": self._extended_forecast_minutes,
                  "first_flat_activity_weight": self._first_flat_activity_weight,
                  "native_endpoint_loss": self.native_endpoint_loss}
        _require(self._research_contract_json == _dynamic_mtf_state_objective_bytes(**kwargs), "STATE_OBJECTIVE_IDENTITY")
        return dynamic_mtf_state_research_contract(**kwargs)

    @property
    def native_endpoint_loss(self):
        return getattr(self, "_native_endpoint_loss", False)

    @property
    def current_state_input(self):
        return self._current_state_input

    @property
    def first_state_loss_weight(self):
        return self._first_state_loss_weight

    @property
    def extended_forecast_minutes(self):
        return self._extended_forecast_minutes

    @property
    def first_flat_activity_weight(self):
        return self._first_flat_activity_weight

    @property
    def signed_bps(self):
        return False

    @property
    def magnitude_weighted(self):
        return False

    def _require_binding(self):
        contract = self.research_objective
        width = _backbone_width(self.backbone)
        _require(width == self._bound_hidden_width, "STATE_HIDDEN_WIDTH")
        from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
        _require(self.backbone.entry_execution_profile.get("name") == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
            and self.backbone.gc_mtf_history is not None
            and "relative_price" not in self.backbone.gc_mtf_history["normalization"], "STATE_FULL_RAW_GC_PROFILE")
        classes, tfs, leads = len(contract["class_names"]), len(contract["timeframes"]), len(contract["lead_minutes"])
        device = self.backbone.entry_q_joint_in.weight.device
        expected = {
            "state_timeframe_codes": torch.arange(tfs, dtype=torch.int64, device=device),
            "state_lead_minutes": torch.tensor(contract["lead_minutes"], dtype=torch.int64, device=device),
            "state_class_codes": torch.arange(classes, dtype=torch.int64, device=device),
            "state_query_coordinates": _state_query_coordinates(contract, device=device),
            "state_objective_sha256": torch.tensor(list(bytes.fromhex(contract["contract_sha256"])), dtype=torch.uint8, device=device)}
        for name, reference in expected.items():
            value = getattr(self, name)
            _require(value.dtype == reference.dtype and value.device == device
                and value.shape == reference.shape and not value.requires_grad
                and torch.equal(value, reference), "STATE_BUFFER_BINDING:" + name)
        _require(type(self.state_query_projection) is nn.Sequential and len(self.state_query_projection) == 2
            and type(self.state_query_projection[1]) is nn.GELU, "STATE_QUERY_PROJECTION")
        for layer, inputs, outputs in (
            (self.state_query_projection[0], width + tfs + 1, width),
            (self.state_initial_head, width, classes),
            (self.state_transition_head, width, classes * classes)):
            _require(type(layer) is nn.Linear and layer.in_features == inputs and layer.out_features == outputs
                and layer.weight.shape == (outputs, inputs) and layer.bias is not None and layer.bias.shape == (outputs,)
                and layer.weight.dtype == layer.bias.dtype == torch.float32
                and layer.weight.device == layer.bias.device == device
                and bool(torch.isfinite(layer.weight).all()) and bool(torch.isfinite(layer.bias).all()), "STATE_HEAD_BINDING")

        if self.current_state_input:
            layer = self.state_current_projection
            _require(type(layer) is nn.Linear and layer.in_features == 5 and layer.out_features == width
                and layer.bias is None and layer.weight.shape == (width, 5)
                and layer.weight.dtype == torch.float32 and layer.weight.device == device
                and bool(torch.isfinite(layer.weight).all()), "CURRENT_PROJECTION_BINDING")
        else:
            _require(not hasattr(self, "state_current_projection"), "CURRENT_PROJECTION_UNBOUND")

    def forward(
        self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor, gc_seq_m15: torch.Tensor,
        gc_seq_h1: torch.Tensor, gc_seq_h4: torch.Tensor, gc_seq_d1: torch.Tensor,
        current_state=None, current_observed=None, current_path_observed=None,
    ) -> dict[str, torch.Tensor]:
        self._require_binding()
        _require(isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3 and seq_x.shape[0] > 0, "STATE_FEATURE_BATCH")
        if self.current_state_input:
            current_features = encode_historical_current_state(current_state, current_observed, current_path_observed,
                batch=seq_x.shape[0], tfs=self.state_timeframe_codes.numel(), device=self.state_query_coordinates.device)
        else:
            _require(current_state is current_observed is current_path_observed is None, "CURRENT_INPUT_NOT_BOUND")
            current_features = None
        output = self.backbone(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
            gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15, gc_seq_h1=gc_seq_h1,
            gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1, research_representation_only=True)
        _require(isinstance(output, dict), "STATE_NATIVE_OUTPUT")
        hidden = output.get("entry_q_joint_hidden")
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
            and hidden.shape == (seq_x.shape[0], self._bound_hidden_width)
            and hidden.device == self.state_query_coordinates.device
            and bool(torch.isfinite(hidden).all()), "STATE_NATIVE_HIDDEN")
        batch, (tfs, leads, _) = hidden.shape[0], self.state_query_coordinates.shape
        coordinates = self.state_query_coordinates[None].expand(batch, -1, -1, -1)
        source = hidden[:, None, None, :].expand(-1, tfs, leads, -1)
        if current_features is not None:
            source = source + self.state_current_projection(current_features)[:, :, None, :]
        decoded = self.state_query_projection(torch.cat((source, coordinates), dim=-1))
        classes = self.state_class_codes.numel()
        initial = self.state_initial_head(decoded[:, :, 0])
        transitions = self.state_transition_head(decoded[:, :, 1:]).reshape(batch, tfs, leads - 1, classes, classes)
        _validate_state_chain_logits(initial, transitions)
        return {"initial_state_logits": initial, "state_transition_logits": transitions}


def _validate_state_chain_logits(initial, transitions):
    _require(isinstance(initial, torch.Tensor) and initial.dtype == torch.float32
        and initial.ndim == 3 and initial.shape[0] > 0 and initial.shape[1] > 0 and initial.shape[-1] == 3,
        "STATE_INITIAL_LOGITS")
    _require(isinstance(transitions, torch.Tensor) and transitions.dtype == torch.float32
        and transitions.ndim == 5 and transitions.shape[:2] == initial.shape[:2]
        and transitions.shape[-2:] == (3, 3) and transitions.device == initial.device,
        "STATE_TRANSITION_LOGITS")
    _require(bool(torch.isfinite(initial).all()) and bool(torch.isfinite(transitions).all()), "STATE_FINITE_LOGITS")
    return tuple(initial.shape[:2]) + (transitions.shape[2] + 1,)


def _reachable_logsumexp(value, dim):
    """Exact log-sum for reachable paths; impossible paths have zero derivative.

    Feeding all -inf directly to logsumexp creates NaN derivatives. The finite
    temporary branch below is discarded exactly, not a probability floor.
    """
    reachable = torch.isfinite(value).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, value, torch.zeros_like(value))
    total = torch.logsumexp(safe, dim=dim)
    return torch.where(reachable.squeeze(dim), total, torch.full_like(total, -torch.inf))


def _state_log_matrix_product(left, right):
    return _reachable_logsumexp(left.unsqueeze(-1) + right.unsqueeze(-3), dim=-2)


def dynamic_mtf_state_marginal_probabilities(initial_state_logits, state_transition_logits):
    """All future marginal probabilities by an associative log-forward scan.

    No target, prior true class, future observation mask, device sync per query,
    sampling, or state-duration cap enters this calculation.
    """
    _validate_state_chain_logits(initial_state_logits, state_transition_logits)
    first = F.log_softmax(initial_state_logits, dim=-1)
    matrices = F.log_softmax(state_transition_logits, dim=-1)
    offset = 1
    while offset < matrices.shape[2]:
        matrices = torch.cat((matrices[:, :, :offset],
            _state_log_matrix_product(matrices[:, :, :-offset], matrices[:, :, offset:])), dim=2)
        offset *= 2
    if matrices.shape[2]:
        later = _reachable_logsumexp(first[:, :, None, :, None] + matrices, dim=-2)
        log_marginals = torch.cat((first[:, :, None], later), dim=2)
    else:
        log_marginals = first[:, :, None]
    # Algebraically the marginals already sum to one; softmax expresses the
    # normalized distribution without a clipping floor or fabricated mass.
    return F.softmax(log_marginals, dim=-1)


def _last_state_log_mass(initial, matrices):
    """Balanced ordered transition product. Inputs may already be restricted by observed states."""
    while matrices.shape[1] > 1:
        pairs = matrices.shape[1] // 2
        reduced = _state_log_matrix_product(matrices[:, :2 * pairs:2], matrices[:, 1:2 * pairs:2])
        matrices = torch.cat((reduced, matrices[:, -1:]), dim=1) if matrices.shape[1] % 2 else reduced
    if matrices.shape[1]:
        initial = _reachable_logsumexp(initial[:, :, None] + matrices[:, 0], dim=-2)
    return initial


def _partial_state_chain_nll(log_initial, log_transitions, state, observed):
    classes = torch.arange(3, device=state.device)
    allowed = (~observed[..., None]) | (state[..., None] == classes)
    initial = log_initial.masked_fill(~allowed[:, 0], -torch.inf)
    matrices = log_transitions.masked_fill(~allowed[:, 1:, None, :], -torch.inf)
    return -_reachable_logsumexp(_last_state_log_mass(initial, matrices), dim=-1)


def _native_endpoint_log_probabilities(initial_logits, transition_logits, lead_minutes):
    """Forecast one distribution per TF from issue time, without conditioning on future labels."""
    first = F.log_softmax(initial_logits, dim=-1)
    endpoints = []
    for tf, lead in enumerate(lead_minutes):
        transitions = F.log_softmax(transition_logits[:, tf, :lead - 1], dim=-1)
        endpoints.append(_last_state_log_mass(first[:, tf], transitions))
    # Normalize in log space; taking log after a float32 softmax loses rare classes.
    return F.log_softmax(torch.stack(endpoints, dim=1), dim=-1)


def state_chain_negative_log_likelihood(initial_state_logits, state_transition_logits, *, state, observed):
    """Per-row/timeframe NLL; arbitrary small lead grids support pure math tests."""
    shape = _validate_state_chain_logits(initial_state_logits, state_transition_logits)
    _require(isinstance(state, torch.Tensor) and state.dtype in (torch.int8, torch.int64)
        and state.shape == shape and state.device == initial_state_logits.device, "STATE_LABEL_SHAPE")
    _require(isinstance(observed, torch.Tensor) and observed.dtype == torch.bool
        and observed.shape == shape and observed.device == state.device, "STATE_OBSERVED_SHAPE")
    _require(bool(((state >= -1) & (state < 3)).all())
        and bool((state[observed] >= 0).all()), "STATE_LABEL_DOMAIN")
    first = F.log_softmax(initial_state_logits, dim=-1).reshape(-1, 3)
    transitions = F.log_softmax(state_transition_logits, dim=-1).reshape(shape[0] * shape[1], shape[-1] - 1, 3, 3)
    labels, mask = state.reshape(-1, shape[-1]), observed.reshape(-1, shape[-1])
    # Select once per batch, never synchronize inside the query calculation.
    complete, partial = mask.all(dim=1), mask.any(dim=1) & ~mask.all(dim=1)
    terms = first.sum(dim=-1) * 0 + transitions.sum(dim=(1, 2, 3)) * 0
    full_ids = torch.nonzero(complete, as_tuple=False).flatten()
    partial_ids = torch.nonzero(partial, as_tuple=False).flatten()
    if full_ids.numel():
        y = labels.index_select(0, full_ids).long()
        nll = -first.index_select(0, full_ids).gather(1, y[:, :1]).squeeze(1)
        if shape[-1] > 1:
            transition_rows = transitions.index_select(0, full_ids).flatten(-2)
            pair_codes = y[:, :-1] * 3 + y[:, 1:]
            nll = nll - transition_rows.gather(-1, pair_codes[..., None]).squeeze(-1).sum(dim=1)
        terms = terms.index_copy(0, full_ids, nll)
    if partial_ids.numel():
        nll = _partial_state_chain_nll(first.index_select(0, partial_ids),
            transitions.index_select(0, partial_ids), labels.index_select(0, partial_ids), mask.index_select(0, partial_ids))
        terms = terms.index_copy(0, partial_ids, nll)
    return terms.reshape(shape[:2])


def native_dynamic_mtf_state_loss(initial_state_logits, state_transition_logits, *,
        state, observed, issue_ns, validity_ns, maturity_ns, cutoff_ns, first_state_loss_weight=None,
        extended_forecast_minutes=None, first_flat_activity_weight=None,
        first_flat_activity_observed=None, first_flat_activity_is_flat=None, native_endpoint_loss=False):
    """Strict declared-grid loss; labels and future clocks stay outside forward."""
    _require(first_state_loss_weight is not None or extended_forecast_minutes is None, "STATE_EXTENDED_REQUIRES_CURRENT_INPUT")
    _require(type(native_endpoint_loss) is bool and (not native_endpoint_loss or first_state_loss_weight is not None),
        "STATE_NATIVE_ENDPOINT_MODE")
    contract = (dynamic_mtf_state_research_contract() if first_state_loss_weight is None
        else dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=first_state_loss_weight,
            extended_forecast_minutes=extended_forecast_minutes, first_flat_activity_weight=first_flat_activity_weight,
            native_endpoint_loss=native_endpoint_loss))
    _require(first_state_loss_weight is not None or first_flat_activity_weight is None, "STATE_FIRST_ACTIVITY_REQUIRES_CURRENT")
    shape = _validate_state_chain_logits(initial_state_logits, state_transition_logits)
    batch, tfs, leads = shape
    _require((tfs, leads) == (len(contract["timeframes"]), len(contract["lead_minutes"])), "STATE_NATIVE_GRID")
    _require(isinstance(state, torch.Tensor) and state.dtype == torch.int8 and tuple(state.shape) == shape, "STATE_NATIVE_LABELS")
    _require(isinstance(observed, torch.Tensor) and observed.dtype == torch.bool and tuple(observed.shape) == shape, "STATE_NATIVE_OBSERVED")
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= _MAX_NS, "STATE_CUTOFF")
    _require(isinstance(issue_ns, torch.Tensor) and issue_ns.dtype == torch.int64 and issue_ns.shape == (batch,),
        "STATE_ISSUE_CLOCK")
    _require(isinstance(validity_ns, torch.Tensor) and validity_ns.dtype == torch.int64
        and validity_ns.shape == (batch, leads), "STATE_VALIDITY_CLOCK")
    _require(isinstance(maturity_ns, torch.Tensor) and maturity_ns.dtype == torch.int64
        and tuple(maturity_ns.shape) == shape, "STATE_MATURITY_CLOCK")
    _require(all(value.device == initial_state_logits.device for value in (state, observed, issue_ns, validity_ns, maturity_ns)),
        "STATE_TARGET_DEVICE")
    _require(bool((issue_ns > 0).all()) and bool((issue_ns < cutoff_ns).all())
        and bool((issue_ns <= _MAX_NS - contract["lead_minutes"][-1] * _MINUTE_NS).all()), "STATE_ISSUE_CUTOFF_OR_OVERFLOW")
    expected = issue_ns[:, None] + torch.tensor(contract["lead_minutes"], dtype=torch.int64, device=issue_ns.device)[None] * _MINUTE_NS
    _require(torch.equal(validity_ns, expected), "STATE_EXACT_VALIDITY")
    _require(torch.equal(maturity_ns, validity_ns[:, None, :].expand(shape)), "STATE_CAUSAL_MATURITY")
    _require(bool(((state >= -1) & (state < len(contract["class_names"]))).all())
        and bool((state[observed] >= 0).all()) and bool((state[~observed] == -1).all()), "STATE_NATIVE_LABEL_DOMAIN")
    mature = maturity_ns < cutoff_ns
    eligible = observed & mature
    sequence_nll = state_chain_negative_log_likelihood(
        initial_state_logits, state_transition_logits, state=state, observed=eligible)
    masks = {"cells": torch.ones_like(observed), "observed_cells": observed, "missing_cells": ~observed,
        "mature_cells": mature, "immature_cells": ~mature, "eligible_cells": eligible,
        "eligible_down_cells": eligible & (state == contract["class_names"].index("DOWN")),
        "eligible_flat_cells": eligible & (state == contract["class_names"].index("FLAT")),
        "eligible_up_cells": eligible & (state == contract["class_names"].index("UP")),
        "immature_observed_cells": observed & ~mature}
    counted = torch.stack([value.sum(dim=(0, 2)) for value in masks.values()]).cpu().tolist()
    per_tf = [{"timeframe": tf, **{key: int(counted[j][i]) for j, key in enumerate(masks)}}
        for i, tf in enumerate(contract["timeframes"])]
    counts = {key: sum(row[key] for row in per_tf) for key in masks}
    counts["parent_rows"] = batch
    _require(counts["eligible_cells"] + counts["immature_observed_cells"] + counts["missing_cells"] == counts["cells"]
        and counts["eligible_down_cells"] + counts["eligible_flat_cells"] + counts["eligible_up_cells"] == counts["eligible_cells"],
        "STATE_COUNT_PARTITION")
    count = counts["eligible_cells"]
    sequence_loss = sequence_nll.sum() / count if count else None
    loss = sequence_loss
    first_loss, first_count = None, 0
    if first_state_loss_weight is not None:
        first_eligible = eligible[:, :, 0]
        first_count = int(first_eligible.sum().item())
        if first_count:
            first_state = torch.where(first_eligible, state[:, :, 0], torch.zeros_like(state[:, :, 0])).long()
            first_nll = -F.log_softmax(initial_state_logits, dim=-1).gather(-1, first_state[..., None]).squeeze(-1)
            first_loss = torch.where(first_eligible, first_nll, torch.zeros_like(first_nll)).sum() / first_count
            loss = sequence_loss + first_state_loss_weight * first_loss
    activity_loss, activity_count = None, 0
    if first_flat_activity_weight is None:
        _require(first_flat_activity_observed is None and first_flat_activity_is_flat is None, "STATE_FIRST_ACTIVITY_UNBOUND")
    else:
        for value in (first_flat_activity_observed, first_flat_activity_is_flat):
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.bool
                and value.shape == (batch, tfs) and value.device == initial_state_logits.device, "STATE_FIRST_ACTIVITY_LABEL")
        activity_mask = first_flat_activity_observed
        _require(bool((~first_flat_activity_is_flat | activity_mask).all())
            and bool((~activity_mask | mature[:, :, 0]).all()), "STATE_FIRST_ACTIVITY_MASK")
        known_first = activity_mask & observed[:, :, 0]
        flat_index = contract["class_names"].index("FLAT")
        _require(bool((~first_flat_activity_is_flat | (observed[:, :, 0] & (state[:, :, 0] == flat_index))).all())
            and bool((~known_first | ((state[:, :, 0] == flat_index) == first_flat_activity_is_flat)).all()),
            "STATE_FIRST_ACTIVITY_CONSISTENCY")
        activity_count = int(activity_mask.sum().item())
        if activity_count:
            logp = F.log_softmax(initial_state_logits, dim=-1)
            other_indices = [i for i in range(len(contract["class_names"])) if i != flat_index]
            flat_logp = logp[:, :, flat_index]
            nonflat_logp = torch.logsumexp(logp[:, :, other_indices], dim=-1)
            nll = -torch.where(first_flat_activity_is_flat, flat_logp, nonflat_logp)
            activity_loss = torch.where(activity_mask, nll, torch.zeros_like(nll)).sum() / activity_count
            term = first_flat_activity_weight * activity_loss
            loss = term if loss is None else loss + term
    if native_endpoint_loss:
        native_leads = contract["native_endpoint_lead_minutes"]
        tf_cols = torch.arange(tfs, device=state.device)
        endpoints = torch.tensor(native_leads, device=state.device) - 1  # Dense lead 1 is at column 0.
        native_mask = eligible[:, tf_cols, endpoints]
        native_counts = native_mask.sum(dim=0).tolist()
        native_count = sum(native_counts)
        native_nll = None
        if native_count:
            logp = _native_endpoint_log_probabilities(initial_state_logits, state_transition_logits, native_leads)
            labels = state[:, tf_cols, endpoints][native_mask].long()
            native_nll = -logp[native_mask].gather(-1, labels[:, None]).mean()
            # A native cell is also eligible for the base sequence loss.
            loss = loss + contract["native_endpoint_weight"] * native_nll
    _require(bool(torch.isfinite(sequence_nll).all()) and (loss is None or bool(torch.isfinite(loss))), "STATE_FINITE_NLL")
    result = {"loss": loss, "update_allowed": count > 0 or activity_count > 0, "eligible_mask": eligible,
        "sequence_nll": sequence_nll, "counts": counts, "per_timeframe_counts": per_tf,
        "research_objective_sha256": contract["contract_sha256"]}
    if first_state_loss_weight is not None:
        result.update(sequence_loss=sequence_loss, first_state_loss=first_loss,
            first_state_eligible_cells=first_count, first_state_loss_weight=first_state_loss_weight)
    if first_flat_activity_weight is not None:
        result.update(first_flat_activity_loss=activity_loss, first_flat_activity_eligible_cells=activity_count,
            first_flat_activity_weight=first_flat_activity_weight)
    if native_endpoint_loss:
        result.update(native_endpoint_loss=native_nll, native_endpoint_eligible_cells=native_count,
            native_endpoint_eligible_per_tf=native_counts)
    return result
