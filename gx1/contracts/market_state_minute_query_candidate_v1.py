"""Source-only minute-query candidate for the existing dynamic S state tape.

It reuses the sealed M1 allowed bits and the existing factorized head weights.
It does not select query times, change ONLINE forward, admit fit, or authorize
TEST, Entry, Exit or trade.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from gx1.contracts import market_state_five_endpoint_contract_v1 as five
from gx1.contracts import market_state_time_query_scorer_v2 as scorer
from gx1.contracts.market_state_time_query_model_v1 import MAX_QUERY_BLOCK
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_log_probabilities,
)
from gx1.contracts.market_state_time_query_target_v1 import (
    MINUTE_NS, TIMEFRAMES, TimeQueryTargetTape,
)

NATIVE_MINUTES = (5, 15, 60, 240, 1440)
VALID_BITS = (1, 2, 4, 5, 7)


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise ValueError('MARKET_STATE_MINUTE_QUERY_' + code)


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_minute_query_candidate_v1',
        'role': 'source_candidate_no_online_fit_or_trade_authority',
        'timeframes': list(TIMEFRAMES),
        'query_unit': 'observed_wall_clock_minute_after_closed_M5_issue',
        'native_minutes': list(NATIVE_MINUTES),
        'source_tape': 'existing_hash_bound_full_observed_M1_allowed_set_tape',
        'future_label_as_forward_input': False,
        'unknown_bits': {'active_side_unknown': 5, 'all_unknown': 7},
        'output': 'existing_factorized_two_logit_head_with_minute_over_native_minutes_fraction',
        'loss': 'equal_TF_masked_activity_plus_certified_side_BCE',
        'score': 'same_mask_activity_conditional_side_and_three_class_proper_scores',
        'query_schedule': 'caller_owned_and_predeclared_without_future_label_selection',
        'old_online_forward_unchanged': True,
        'fresh_initial_baseline_required_before_any_minute_online_fit': True,
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _minute_domain(issue_ns: np.ndarray, query_minutes: np.ndarray,
                   fold_upper_ns: int) -> None:
    need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
         and issue_ns.ndim == 1 and len(issue_ns) > 0
         and isinstance(query_minutes, np.ndarray)
         and query_minutes.dtype == np.int64
         and query_minutes.shape == (len(issue_ns), len(TIMEFRAMES))
         and type(fold_upper_ns) is int and fold_upper_ns > 0,
         'QUERY_SHAPE')
    need(bool(np.all(issue_ns >= 0))
         and bool(np.all(issue_ns % (5 * MINUTE_NS) == 0))
         and bool(np.all(issue_ns < fold_upper_ns))
         and bool(np.all(query_minutes >= 1))
         and bool(np.all(query_minutes <= np.asarray(NATIVE_MINUTES)[None, :]))
         and bool(np.all(issue_ns <= np.iinfo(np.int64).max
                         - NATIVE_MINUTES[-1] * MINUTE_NS)), 'QUERY_DOMAIN')


class MinuteQueryTargetTape(TimeQueryTargetTape):
    """Exact mature minute lookup; diagnostics cannot become forward inputs."""

    def __init__(self, validity_ns: np.ndarray, source_row: np.ndarray,
                 allowed_bits: np.ndarray, path_observed: np.ndarray,
                 gap_kind: np.ndarray):
        super().__init__(validity_ns, source_row, allowed_bits)
        n = len(self.validity_ns)
        need(isinstance(path_observed, np.ndarray)
             and path_observed.dtype == np.bool_
             and path_observed.shape == (n, len(TIMEFRAMES))
             and isinstance(gap_kind, np.ndarray)
             and gap_kind.dtype == np.uint8 and gap_kind.shape == (n,)
             and np.isin(gap_kind, (0, 1, 2)).all(), 'TAPE_DIAGNOSTICS')
        self.path_observed = np.frombuffer(
            np.ascontiguousarray(path_observed).tobytes(), dtype=np.bool_).reshape(
                n, len(TIMEFRAMES))
        self.gap_kind = np.frombuffer(
            np.ascontiguousarray(gap_kind).tobytes(), dtype=np.uint8)

    def gather_minutes(self, issue_ns: np.ndarray, query_minutes: np.ndarray,
                       *, fold_upper_ns: int) -> dict:
        _minute_domain(issue_ns, query_minutes, fold_upper_ns)
        validity = issue_ns[:, None] + query_minutes * MINUTE_NS
        mature = validity < fold_upper_ns
        allowed = np.full(query_minutes.shape, 7, np.uint8)
        source = np.full(query_minutes.shape, -1, np.int64)
        path = np.zeros(query_minutes.shape, np.bool_)
        gap = np.zeros(query_minutes.shape, np.uint8)
        wanted = validity[mature]
        at = np.searchsorted(self.validity_ns, wanted)
        present = at < len(self.validity_ns)
        present[present] &= self.validity_ns[at[present]] == wanted[present]
        selected = np.flatnonzero(mature)[present]
        source_at = at[present]
        tf_index = selected % len(TIMEFRAMES)
        allowed.flat[selected] = self.allowed_bits[source_at, tf_index]
        source.flat[selected] = self.source_row[source_at]
        path.flat[selected] = self.path_observed[source_at, tf_index]
        gap.flat[selected] = self.gap_kind[source_at]
        eligible = mature & (allowed != 7) & (source >= 0)
        return {
            'issue_ns': issue_ns.copy(), 'query_minutes': query_minutes.copy(),
            'validity_ns': validity, 'allowed': allowed,
            'm1_source_row': source, 'mature': mature, 'eligible': eligible,
            'path_observed': path, 'source_gap_kind': gap,
        }


def decode_raw_minute_grid(model: FactorizedTimeQueryMarketStateModel,
                           hidden: torch.Tensor,
                           query_minutes: torch.Tensor) -> torch.Tensor:
    """Use the existing head at caller-selected minute fractions, Q<=32."""
    need(isinstance(model, FactorizedTimeQueryMarketStateModel), 'MODEL')
    model._require_binding()
    batch = hidden.shape[0] if isinstance(hidden, torch.Tensor) and hidden.ndim == 2 else -1
    need(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
         and hidden.shape == (batch, model._hidden_width) and batch > 0
         and hidden.device == model.query_tf_coordinates.device
         and bool(torch.isfinite(hidden).all()), 'HIDDEN')
    limit = torch.tensor(NATIVE_MINUTES, device=hidden.device, dtype=torch.int64)
    need(isinstance(query_minutes, torch.Tensor)
         and query_minutes.dtype == torch.int64
         and query_minutes.ndim == 3 and query_minutes.shape[0] == batch
         and 1 <= query_minutes.shape[1] <= MAX_QUERY_BLOCK
         and query_minutes.shape[2] == len(TIMEFRAMES)
         and query_minutes.device == hidden.device
         and bool(((query_minutes >= 1)
                   & (query_minutes <= limit[None, None, :])).all()),
         'QUERY_MINUTES')
    blocks = query_minutes.shape[1]
    source = hidden[:, None, None, :].expand(-1, blocks, len(TIMEFRAMES), -1)
    tf = model.query_tf_coordinates[None, None].expand(batch, blocks, -1, -1)
    fraction = (query_minutes.to(torch.float32)
                / limit.to(torch.float32)[None, None, :]).unsqueeze(-1)
    representation = model.query_projection(torch.cat((source, tf, fraction), dim=-1))
    raw = model.query_head(representation)
    if model._dedicated_m5_side:
        raw = raw.clone()
        raw[..., 0, 1] = model.m5_side_head(representation[..., 0, :]).squeeze(-1)
    if model._dedicated_non_m5_side:
        for index, timeframe in enumerate(TIMEFRAMES[1:], start=1):
            raw[..., index, 1] = model.non_m5_side_heads[timeframe](
                representation[..., index, :]).squeeze(-1)
    if model._dedicated_tf_activity:
        for index, timeframe in enumerate(TIMEFRAMES):
            raw[..., index, 0] = model.tf_activity_heads[timeframe](
                representation[..., index, :]).squeeze(-1)
    need(raw.shape == (batch, blocks, len(TIMEFRAMES), 2)
         and bool(torch.isfinite(raw).all()), 'LOGITS')
    return raw


def minute_allowed_set_loss(raw: torch.Tensor, allowed: torch.Tensor,
                            eligible: torch.Tensor, mature: torch.Tensor,
                            query_minutes: torch.Tensor) -> dict:
    """Equal-TF loss with exact minute-domain and partial-label masks."""
    need(isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
         and raw.ndim == 3 and raw.shape[1:] == (len(TIMEFRAMES), 2)
         and raw.shape[0] > 0 and bool(torch.isfinite(raw).all()), 'LOSS_LOGITS')
    shape = raw.shape[:2]
    limit = torch.tensor(NATIVE_MINUTES, dtype=torch.int64, device=raw.device)
    need(isinstance(query_minutes, torch.Tensor)
         and query_minutes.dtype == torch.int64 and query_minutes.shape == shape
         and query_minutes.device == raw.device
         and bool(((query_minutes >= 1)
                   & (query_minutes <= limit[None, :])).all()), 'LOSS_QUERY')
    need(isinstance(allowed, torch.Tensor) and allowed.dtype == torch.uint8
         and allowed.shape == shape and allowed.device == raw.device
         and bool(torch.isin(allowed, torch.tensor(
             VALID_BITS, dtype=torch.uint8, device=raw.device)).all())
         and all(isinstance(x, torch.Tensor) and x.dtype == torch.bool
                 and x.shape == shape and x.device == raw.device
                 for x in (eligible, mature)), 'LOSS_LABELS')
    informative = eligible & mature & (allowed != 7)
    side_known = informative & ((allowed == 1) | (allowed == 4))
    counts = {tf: int(informative[:, j].sum().item())
              for j, tf in enumerate(TIMEFRAMES)}
    zero = raw.mul(0).sum()
    if sum(counts.values()):
        activity_nll = F.binary_cross_entropy_with_logits(
            raw[..., 0], (allowed != 2).to(raw.dtype), reduction='none')
        side_nll = F.binary_cross_entropy_with_logits(
            raw[..., 1], (allowed == 4).to(raw.dtype), reduction='none')
        per_cell = (torch.where(informative, activity_nll, torch.zeros_like(activity_nll))
                    + torch.where(side_known, side_nll, torch.zeros_like(side_nll)))
        terms = [(per_cell[:, j].sum() / counts[tf] if counts[tf] else zero)
                 for j, tf in enumerate(TIMEFRAMES)]
    else:
        terms = [zero for _ in TIMEFRAMES]
    loss = sum(terms, zero) / len(TIMEFRAMES)
    need(bool(torch.isfinite(loss)), 'LOSS_FINITE')
    return {'loss': loss, 'count': sum(counts.values()),
            'per_tf_counts': counts, 'per_tf_mean_loss': dict(zip(TIMEFRAMES, terms)),
            'informative_mask': informative, 'side_known_mask': side_known,
            'skip_update': not bool(sum(counts.values())),
            'contract_sha256': contract()['contract_sha256']}


def minute_proper_score_sums(raw: np.ndarray, allowed: np.ndarray,
                             eligible: np.ndarray, mature: np.ndarray,
                             query_minutes: np.ndarray) -> dict:
    """Existing proper scores with the actual minute query identity retained."""
    shape = raw.shape[:2] if isinstance(raw, np.ndarray) and raw.ndim == 3 else ()
    need(isinstance(query_minutes, np.ndarray) and query_minutes.dtype == np.int64
         and query_minutes.shape == shape and shape[1:] == (len(TIMEFRAMES),)
         and np.all((query_minutes >= 1)
                    & (query_minutes <= np.asarray(NATIVE_MINUTES)[None, :])),
         'SCORE_QUERY')
    classes, _, _ = scorer.factorized_probabilities(raw)
    result = five.proper_score_sums(classes, allowed, eligible, mature)
    return {**result, 'contract_sha256': contract()['contract_sha256'],
            'query_minutes_sha256': hashlib.sha256(query_minutes.tobytes()).hexdigest()}
