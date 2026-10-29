"""Separate 1..60-minute direction logits over the existing native backbone.

Research-only component: this code grants no data, optimizer or trading access.
Actual training additionally requires fresh source-bound qualification and
admission. The former 1/5/15 heads remain identically initialized; adding the
other 57 heads preserves the caller's RNG state. The new 60-cell objective is
not equivalent to the original three-cell objective after the first update.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gx1.contracts.entry_native_direction_research_v1 import (
    NativeEntryDirectionResearch, HORIZONS_MINUTES as ANCHORS,
    _MAX_NS, _MINUTE_NS,
)

HORIZONS_MINUTES = tuple(range(1, 61))
AUX_HORIZONS = tuple(h for h in HORIZONS_MINUTES if h not in ANCHORS)
_REORDER = tuple(ANCHORS.index(h) if h in ANCHORS else len(ANCHORS) + AUX_HORIZONS.index(h)
                 for h in HORIZONS_MINUTES)
_OBJECTIVE = {
    'schema_version': 'gx1_entry_native_direction_dense_v1',
    'research_objective': 'sixty_separate_observed_nonzero_midpoint_directions',
    'horizons_minutes': list(HORIZONS_MINUTES),
    'anchor_minutes': list(ANCHORS),
    'head': 'original_1_5_15_logits_plus_independent_57_logit_linear_head',
    'initialization': 'original_anchor_then_aux_linear_in_forked_rng',
    'loss': 'unweighted_BCE_with_logits_sum_divided_by_all_eligible_minute_cells',
    'eligible': 'observed_and_sign_nonzero_and_terminal_strictly_before_own_cutoff',
    'empty_eligible': 'loss_None_no_optimizer_EMA_or_scheduler_step',
    'future_targets_or_clocks_as_forward_inputs': False,
    'native_forwards_per_research_forward': 1,
    'native_export_authorized': False, 'native_launch_authorized': False,
    'economic_edge_claimed': False,
}
_OBJECTIVE_JSON = json.dumps(_OBJECTIVE, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
_OBJECTIVE_SHA = hashlib.sha256(_OBJECTIVE_JSON).hexdigest()

_BPS_OBJECTIVE = {**_OBJECTIVE,
    'schema_version': 'gx1_entry_native_direction_dense_signed_bps_v1',
    'research_objective': 'sixty_midpoint_directions_and_sixty_signed_bps_means',
    'bps_head': 'zero_initialized_independent_60_output_linear_head_on_shared_hidden',
    'bps_scale': 'own_train_prefix_per_horizon_rms_float32_target_no_centering',
    'loss': 'unweighted_60_direction_BCE_plus_unweighted_60_normalized_signed_bps_MSE',
    'bps_eligible': 'observed_including_ties_and_terminal_strictly_before_own_cutoff',
}
_BPS_OBJECTIVE_JSON = json.dumps(_BPS_OBJECTIVE, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
_BPS_OBJECTIVE_SHA = hashlib.sha256(_BPS_OBJECTIVE_JSON).hexdigest()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('NATIVE_DIRECTION_DENSE_' + reason)


def dense_direction_research_contract() -> dict[str, Any]:
    return {**json.loads(_OBJECTIVE_JSON), 'contract_sha256': _OBJECTIVE_SHA}


def dense_direction_bps_research_contract() -> dict[str, Any]:
    return {**json.loads(_BPS_OBJECTIVE_JSON), 'contract_sha256': _BPS_OBJECTIVE_SHA}


class DenseNativeEntryDirectionResearch(NativeEntryDirectionResearch):
    """One native forward, three original heads and 57 distinct new heads."""

    def __init__(self, backbone, *, m1_ofi_normalization=None, m1_ofi_coarsened=False, five_minute_priority=False, dense_bps_scale=None):
        _require(type(five_minute_priority) is bool and (not five_minute_priority or m1_ofi_normalization is None), 'FIVE_PRIORITY_SCOPE')
        _require(type(m1_ofi_coarsened) is bool and (not m1_ofi_coarsened or m1_ofi_normalization is not None), 'M1_REPRESENTATION')
        _require(dense_bps_scale is None or (not five_minute_priority and m1_ofi_normalization is None), 'BPS_SCOPE')
        super().__init__(backbone)
        device = self.direction_head.weight.device
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
            self.dense_aux_head = nn.Linear(self._bound_hidden_width, len(AUX_HORIZONS),
                device=device, dtype=torch.float32)
        self.register_buffer('dense_horizon_minutes', torch.tensor(HORIZONS_MINUTES, dtype=torch.int64, device=device))
        self.register_buffer('dense_reorder', torch.tensor(_REORDER, dtype=torch.int64, device=device))
        self._five_minute_priority = five_minute_priority
        self._dense_bps_scale_json = None
        if dense_bps_scale is not None:
            from gx1.contracts.entry_native_direction_bps_v1 import require_dense_direction_bps_scale
            fitted_bps = require_dense_direction_bps_scale(dense_bps_scale)
            self._dense_bps_scale_json = json.dumps(fitted_bps, sort_keys=True, separators=(',', ':'), allow_nan=False)
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                self.dense_bps_head = nn.Linear(self._bound_hidden_width, 60, device=device, dtype=torch.float32)
                nn.init.zeros_(self.dense_bps_head.weight)
                nn.init.zeros_(self.dense_bps_head.bias)
            self.register_buffer('dense_bps_rms', torch.tensor(fitted_bps['rms_bps'], dtype=torch.float32, device=device))
            self.register_buffer('dense_bps_scale_sha256', torch.tensor(list(bytes.fromhex(fitted_bps['contract_sha256'])), dtype=torch.uint8, device=device))
        if self.dense_signed_bps:
            self._dense_objective_json = _BPS_OBJECTIVE_JSON
        elif five_minute_priority:
            from gx1.contracts.entry_native_direction_five_priority_v1 import _PRIORITY_OBJECTIVE_JSON
            self._dense_objective_json = _PRIORITY_OBJECTIVE_JSON
        else:
            self._dense_objective_json = _OBJECTIVE_JSON
        self._m1_normalization_json = None
        self._m1_ofi_coarsened = m1_ofi_coarsened
        if m1_ofi_normalization is not None:
            from gx1.contracts.entry_native_direction_m1_ofi_normalization_v1 import require
            fitted = require(m1_ofi_normalization)
            self._m1_normalization_json = json.dumps(fitted, sort_keys=True, separators=(',', ':'), allow_nan=False)
            width = self._bound_hidden_width
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                self.m1_ofi_in = nn.Linear(width + 10, width, device=device, dtype=torch.float32)
                self.m1_ofi_out = nn.Linear(width, width, device=device, dtype=torch.float32)
                nn.init.zeros_(self.m1_ofi_out.weight)
                nn.init.zeros_(self.m1_ofi_out.bias)
            self.register_buffer('m1_ofi_center', torch.tensor(fitted['surface']['center'][0], dtype=torch.float64, device=device))
            self.register_buffer('m1_ofi_scale', torch.tensor(fitted['surface']['scale'][0], dtype=torch.float64, device=device))
            self.register_buffer('m1_ofi_normalization_sha256', torch.tensor(
                list(bytes.fromhex(fitted['contract_sha256'])), dtype=torch.uint8, device=device))
            self.register_buffer('m1_ofi_coarsened', torch.tensor(int(m1_ofi_coarsened), dtype=torch.uint8, device=device))

    @property
    def five_minute_priority(self) -> bool:
        _require(type(self._five_minute_priority) is bool, 'FIVE_PRIORITY_IDENTITY')
        return self._five_minute_priority

    @property
    def dense_signed_bps(self) -> bool:
        return self._dense_bps_scale_json is not None

    @property
    def dense_bps_scale(self):
        _require(self.dense_signed_bps, 'BPS_MODE_REQUIRED')
        return json.loads(self._dense_bps_scale_json)

    @property
    def research_objective(self) -> dict[str, Any]:
        if self.dense_signed_bps:
            _require(self._dense_objective_json == _BPS_OBJECTIVE_JSON, 'OBJECTIVE_IDENTITY')
            return dense_direction_bps_research_contract()
        if self.five_minute_priority:
            from gx1.contracts.entry_native_direction_five_priority_v1 import _PRIORITY_OBJECTIVE_JSON, five_priority_direction_contract
            _require(self._dense_objective_json == _PRIORITY_OBJECTIVE_JSON, 'OBJECTIVE_IDENTITY')
            return five_priority_direction_contract()
        _require(self._dense_objective_json == _OBJECTIVE_JSON, 'OBJECTIVE_IDENTITY')
        return dense_direction_research_contract()

    def _require_binding(self) -> None:
        super()._require_binding()
        if not self.dense_signed_bps:
            _require(not any(hasattr(self, name) for name in ('dense_bps_head', 'dense_bps_rms', 'dense_bps_scale_sha256')), 'UNBOUND_BPS_STATE')
        if self.dense_signed_bps:
            _require(self._dense_objective_json == _BPS_OBJECTIVE_JSON and not self.five_minute_priority
                and self._m1_normalization_json is None, 'BPS_OBJECTIVE_IDENTITY')
            from gx1.contracts.entry_native_direction_bps_v1 import require_dense_direction_bps_scale
            fitted = require_dense_direction_bps_scale(self.dense_bps_scale)
            head = self.dense_bps_head
            device = self.direction_head.weight.device
            _require(type(head) is nn.Linear and head.in_features == self._bound_hidden_width and head.out_features == 60
                and head.bias is not None and head.weight.dtype == head.bias.dtype == torch.float32
                and head.weight.device == head.bias.device == device
                and bool(torch.isfinite(head.weight).all()) and bool(torch.isfinite(head.bias).all()), 'BPS_HEAD')
            _require(self.dense_bps_rms.dtype == torch.float32 and self.dense_bps_rms.device == device
                and not self.dense_bps_rms.requires_grad and torch.equal(self.dense_bps_rms,
                    torch.tensor(fitted['rms_bps'], dtype=torch.float32, device=device))
                and self.dense_bps_scale_sha256.dtype == torch.uint8 and self.dense_bps_scale_sha256.device == device
                and torch.equal(self.dense_bps_scale_sha256,
                    torch.tensor(list(bytes.fromhex(fitted['contract_sha256'])), dtype=torch.uint8, device=device)), 'BPS_SCALE')
        elif self.five_minute_priority:
            from gx1.contracts.entry_native_direction_five_priority_v1 import _PRIORITY_OBJECTIVE_JSON
            _require(self._dense_objective_json == _PRIORITY_OBJECTIVE_JSON, 'OBJECTIVE_IDENTITY')
        else:
            _require(self._dense_objective_json == _OBJECTIVE_JSON, 'OBJECTIVE_IDENTITY')
        head, device, width = self.dense_aux_head, self.direction_head.weight.device, self._bound_hidden_width
        _require(type(head) is nn.Linear and head.in_features == width and head.out_features == 57
            and tuple(head.weight.shape) == (57, width) and head.bias is not None and tuple(head.bias.shape) == (57,)
            and head.weight.dtype == head.bias.dtype == torch.float32
            and head.weight.device == head.bias.device == device
            and bool(torch.isfinite(head.weight).all()) and bool(torch.isfinite(head.bias).all()), 'AUX_HEAD')
        for name, expected in (('dense_horizon_minutes', HORIZONS_MINUTES), ('dense_reorder', _REORDER)):
            value = getattr(self, name)
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and value.device == device
                and not value.requires_grad and tuple(value.shape) == (60,)
                and torch.equal(value, torch.tensor(expected, dtype=torch.int64, device=device)), name.upper())
        if self._m1_normalization_json is not None:
            from gx1.contracts.entry_native_direction_m1_ofi_normalization_v1 import require
            fitted = require(json.loads(self._m1_normalization_json))
            for name, inputs, outputs in (('m1_ofi_in', width + 10, width), ('m1_ofi_out', width, width)):
                layer = getattr(self, name)
                _require(type(layer) is nn.Linear and layer.in_features == inputs and layer.out_features == outputs
                    and layer.weight.dtype == layer.bias.dtype == torch.float32
                    and layer.weight.device == layer.bias.device == device
                    and bool(torch.isfinite(layer.weight).all()) and bool(torch.isfinite(layer.bias).all()), 'M1_LAYER:' + name)
            for name, expected in (('m1_ofi_center', fitted['surface']['center'][0]), ('m1_ofi_scale', fitted['surface']['scale'][0])):
                value = getattr(self, name)
                _require(value.dtype == torch.float64 and value.device == device and not value.requires_grad
                    and tuple(value.shape) == () and value.item() == float(expected), 'M1_STAT:' + name)
            value = self.m1_ofi_normalization_sha256
            _require(value.dtype == torch.uint8 and value.device == device and not value.requires_grad
                and torch.equal(value, torch.tensor(list(bytes.fromhex(fitted['contract_sha256'])), dtype=torch.uint8, device=device)), 'M1_IDENTITY')
            representation=self.m1_ofi_coarsened
            _require(type(self._m1_ofi_coarsened) is bool and representation.dtype==torch.uint8
                and representation.device==device and not representation.requires_grad and tuple(representation.shape)==()
                and representation.item()==int(self._m1_ofi_coarsened), 'M1_REPRESENTATION')

    def forward(self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
        ctx_cat: torch.Tensor, ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor, seq_m15: torch.Tensor, seq_h1: torch.Tensor,
        seq_h4: torch.Tensor, seq_d1: torch.Tensor,
        gc_seq_m5: torch.Tensor | None = None, gc_seq_m15: torch.Tensor | None = None,
        gc_seq_h1: torch.Tensor | None = None, gc_seq_h4: torch.Tensor | None = None,
        gc_seq_d1: torch.Tensor | None = None,
        m1_ofi_contracts: torch.Tensor | None = None, m1_ofi_observed: torch.Tensor | None = None) -> dict[str, Any]:
        self._require_binding()
        _require(isinstance(seq_x, torch.Tensor) and seq_x.ndim == 3 and seq_x.shape[0] > 0, 'FEATURE_BATCH')
        if self._m1_normalization_json is None:
            _require(m1_ofi_contracts is None and m1_ofi_observed is None, 'UNBOUND_M1_INPUT')
        else:
            _require(isinstance(m1_ofi_contracts, torch.Tensor) and m1_ofi_contracts.dtype == torch.float64
                and isinstance(m1_ofi_observed, torch.Tensor) and m1_ofi_observed.dtype == torch.bool
                and tuple(m1_ofi_contracts.shape) == tuple(m1_ofi_observed.shape) == (seq_x.shape[0], 5)
                and m1_ofi_contracts.device == m1_ofi_observed.device == self.direction_head.weight.device
                and bool(torch.isfinite(m1_ofi_contracts).all())
                and bool((m1_ofi_contracts[~m1_ofi_observed] == 0).all()), 'M1_INPUT')
        output = self.backbone(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont,
            seq_m5=seq_m5, seq_m15=seq_m15, seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
            gc_seq_m5=gc_seq_m5, gc_seq_m15=gc_seq_m15, gc_seq_h1=gc_seq_h1,
            gc_seq_h4=gc_seq_h4, gc_seq_d1=gc_seq_d1)
        _require(isinstance(output, dict), 'NATIVE_OUTPUT')
        hidden = output.get('entry_q_joint_hidden')
        _require(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
            and tuple(hidden.shape) == (seq_x.shape[0], self._bound_hidden_width)
            and hidden.device == self.direction_head.weight.device
            and bool(torch.isfinite(hidden).all()), 'NATIVE_HIDDEN')
        if self._m1_normalization_json is not None:
            value = m1_ofi_contracts
            if self._m1_ofi_coarsened:
                # Same raw inputs and masks; remove within-bucket temporal detail.
                # Multiplying the mean by observed count reconstructs the sum.
                count=m1_ofi_observed.sum(dim=1,keepdim=True)
                mean=value.sum(dim=1,keepdim=True)/count.clamp_min(1)
                value=torch.where(m1_ofi_observed,mean,torch.zeros_like(value))
            normalized = torch.asinh((value - self.m1_ofi_center) / self.m1_ofi_scale)
            normalized = torch.where(m1_ofi_observed, normalized, torch.zeros_like(normalized)).to(torch.float32)
            _require(bool(torch.isfinite(normalized).all()), 'M1_NORMALIZED_FINITE')
            candidate = self.m1_ofi_out(F.gelu(self.m1_ofi_in(torch.cat(
                (hidden, normalized, m1_ofi_observed.to(torch.float32)), dim=1))))
            _require(bool(torch.isfinite(candidate).all()), 'M1_RESIDUAL_FINITE')
            hidden = hidden + torch.where(m1_ofi_observed.any(dim=1, keepdim=True), candidate, torch.zeros_like(candidate))
        combined = torch.cat((self.direction_head(hidden), self.dense_aux_head(hidden)), dim=1)
        logits = combined.index_select(1, self.dense_reorder)
        _require(logits.dtype == torch.float32 and tuple(logits.shape) == (seq_x.shape[0], 60)
            and bool(torch.isfinite(logits).all()), 'FINITE_LOGITS')
        result = {'logit_up': logits, 'horizons_minutes': HORIZONS_MINUTES}
        if self.dense_signed_bps:
            mean = self.dense_bps_head(hidden) * self.dense_bps_rms
            _require(mean.dtype == torch.float32 and tuple(mean.shape) == tuple(logits.shape)
                and bool(torch.isfinite(mean).all()), 'FINITE_BPS_MEAN')
            result['mean_return_bps'] = mean
        return result


def dense_direction_binary_loss(logit_up: torch.Tensor, *, sign: torch.Tensor, observed: torch.Tensor,
    decision_ns: torch.Tensor, terminal_time_ns: torch.Tensor, cutoff_ns: int) -> dict[str, Any]:
    """BCE on each matured, observed, nonzero endpoint; no future model input."""
    _require(isinstance(logit_up, torch.Tensor) and logit_up.dtype == torch.float32
        and logit_up.ndim == 2 and logit_up.shape[0] > 0 and logit_up.shape[1] == 60, 'LOGIT_SHAPE_DTYPE')
    size = logit_up.shape[0]
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= _MAX_NS, 'CUTOFF')
    _require(isinstance(sign, torch.Tensor) and sign.dtype == torch.int8 and tuple(sign.shape) == (size, 60), 'SIGN_SHAPE_DTYPE')
    _require(isinstance(observed, torch.Tensor) and observed.dtype == torch.bool and tuple(observed.shape) == (size, 60), 'OBSERVED_SHAPE_DTYPE')
    _require(isinstance(decision_ns, torch.Tensor) and decision_ns.dtype == torch.int64 and tuple(decision_ns.shape) == (size,), 'DECISION_SHAPE_DTYPE')
    _require(isinstance(terminal_time_ns, torch.Tensor) and terminal_time_ns.dtype == torch.int64
        and tuple(terminal_time_ns.shape) == (size, 60), 'TERMINAL_SHAPE_DTYPE')
    _require(all(v.device == logit_up.device for v in (sign, observed, decision_ns, terminal_time_ns)), 'DEVICE')
    _require(bool(torch.isfinite(logit_up).all()), 'FINITE_LOGITS')
    _require(bool(torch.isin(sign, torch.tensor((-1, 0, 1), dtype=torch.int8, device=sign.device)).all())
        and bool((sign[~observed] == 0).all()), 'SIGN_DOMAIN_OR_MISSING_STORAGE')
    _require(bool((decision_ns > 0).all()) and bool((decision_ns < cutoff_ns).all())
        and bool((decision_ns <= _MAX_NS - 60 * _MINUTE_NS).all()), 'DECISION_CUTOFF_OR_OVERFLOW')
    expected = decision_ns[:, None] + torch.tensor(HORIZONS_MINUTES, dtype=torch.int64, device=decision_ns.device)[None, :] * _MINUTE_NS
    _require(torch.equal(terminal_time_ns, expected), 'EXACT_TERMINAL_CLOCK')
    mature = terminal_time_ns < cutoff_ns
    nonzero = sign != 0
    eligible = observed & nonzero & mature
    masks = {'cells': torch.ones_like(observed), 'observed_cells': observed, 'missing_cells': ~observed,
        'observed_up_cells': observed & (sign == 1), 'observed_down_cells': observed & (sign == -1),
        'observed_tie_cells': observed & ~nonzero, 'mature_cells': mature, 'immature_cells': ~mature,
        'eligible_cells': eligible, 'eligible_up_cells': eligible & (sign == 1), 'eligible_down_cells': eligible & (sign == -1),
        'immature_observed_nonzero_cells': observed & nonzero & ~mature,
        'mature_tie_cells': observed & ~nonzero & mature, 'immature_tie_cells': observed & ~nonzero & ~mature,
        'mature_missing_cells': ~observed & mature, 'immature_missing_cells': ~observed & ~mature}
    counted = torch.stack([value.sum(dim=0) for value in masks.values()]).cpu().tolist()
    per_horizon = [{'horizon_minutes': h, 'parent_rows': size,
        **{name: int(counted[j][i]) for j, name in enumerate(masks)}} for i, h in enumerate(HORIZONS_MINUTES)]
    counts = {'parent_rows': size, **{name: sum(row[name] for row in per_horizon) for name in masks}}
    _require(counts['eligible_cells'] + counts['observed_tie_cells'] + counts['immature_observed_nonzero_cells']
        + counts['missing_cells'] == counts['cells'] == size * 60, 'COUNT_PARTITION')
    count = counts['eligible_cells']
    loss = None
    if count:
        labels = (sign[eligible] == 1).to(dtype=logit_up.dtype)
        loss = F.binary_cross_entropy_with_logits(logit_up[eligible], labels, reduction='sum') / count
        _require(bool(torch.isfinite(loss)) and bool(loss >= 0), 'FINITE_LOSS')
    return {'loss': loss, 'update_allowed': count > 0, 'eligible_mask': eligible,
        'counts': counts, 'per_horizon_counts': per_horizon, 'research_objective_sha256': _OBJECTIVE_SHA}
