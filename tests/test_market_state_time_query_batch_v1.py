"""Synthetic batch bridge checks; no physical fit, TEST or optimizer."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import market_state_time_query_batch_v1 as batch
from gx1.contracts.market_state_time_query_binding_v1 import (
    BoundTimeQueryTargets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_model_v1 import TimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_target_v1 import (
    NATIVE_K, TICK_NS, TimeQueryTargetTape,
)
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def _features(rows: int) -> dict[str, torch.Tensor]:
    x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 1, 4) + 1
    result = {'seq_x': x, 'snap_x': x[:, 0],
              'ctx_cat': torch.zeros(rows, 1, dtype=torch.int64),
              'ctx_cont': torch.zeros(rows, 1, dtype=torch.float32)}
    for tf in ('m5', 'm15', 'h1', 'h4', 'd1'):
        result['seq_' + tf] = torch.ones(rows, 1, 1)
        result['gc_seq_' + tf] = torch.ones(rows, 1, 1)
    return result


def _model(monkeypatch, forward_calls: list[int]) -> TimeQueryMarketStateModel:
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        forward_calls.append(1)
        assert kwargs.pop('research_representation_only') is True
        assert set(kwargs) == batch.FEATURE_KEYS - {'seq_x', 'snap_x'}
        return {'entry_q_joint_hidden': self.entry_q_joint_in(
            torch.cat((seq_x[:, 0, :],) * 4, dim=1))}

    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', synthetic_forward)
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, 'd_model', 4)
    object.__setattr__(cfg, 'entry_training_phase', ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.entry_execution_profile = {'name': FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {'normalization': {}}
    return TimeQueryMarketStateModel(native)


def _bound() -> tuple[BoundTimeQueryTargets, np.ndarray, np.ndarray]:
    boundary = FOLD_BOUNDARIES_NS[0]
    issue = np.array([boundary - 600 * TICK_NS,
                      boundary - 300 * TICK_NS], dtype=np.int64)
    parents = np.array([11, 22], dtype=np.int64)
    query = sampled_query_k(issue, pass_index=0, seed=19)
    native = np.asarray(NATIVE_K, dtype=np.int64)
    validity = np.array(sorted({int(issue[i] + k * TICK_NS)
                                for i in range(2)
                                for k in (*query[i], *native)}), dtype=np.int64)
    source = np.arange(100, 100 + len(validity), dtype=np.int64)
    bits = np.full((len(validity), 5), 7, dtype=np.uint8)
    for j, bit in enumerate((1, 2, 4, 5, 5)):
        at = int(np.searchsorted(validity, issue[0] + query[0, j] * TICK_NS))
        bits[at, j] = bit
    tape = TimeQueryTargetTape(validity, source, bits)
    endpoint = np.broadcast_to(native, (2, 5)).copy()
    old = tape.gather(issue, endpoint, fold_upper_ns=FOLD_BOUNDARIES_NS[-1])
    return (BoundTimeQueryTargets(parents, issue, tape, issue,
                                  old['allowed'], old['m1_source_row'],
                                  train_end_ns=FOLD_BOUNDARIES_NS[-1]),
            parents, issue)


def test_fit_bridge_query_before_target_and_backbone_gradient(monkeypatch):
    forward_calls = []
    model = _model(monkeypatch, forward_calls)
    bound, parents, issue = _bound()
    original_gather = TimeQueryTargetTape.gather

    def gather_after_forward(self, *args, **kwargs):
        assert forward_calls == [1]
        return original_gather(self, *args, **kwargs)

    monkeypatch.setattr(TimeQueryTargetTape, 'gather', gather_after_forward)
    result = batch.evaluate_fit_batch(model, _features(2), bound, parents,
                                      issue, fold_id=1, pass_index=0, seed=19)
    assert forward_calls == [1]
    assert result['count'] == 5 and not result['skip_update']
    assert result['per_tf_counts'] == {tf: 1 for tf in ('M5', 'M15', 'H1', 'H4', 'D1')}
    assert np.array_equal(result['query_k'], sampled_query_k(issue, pass_index=0, seed=19))
    assert result['allowed'][0].tolist() == [1, 2, 4, 5, 5]
    assert result['allowed'][1].tolist() == [7] * 5
    assert np.isfinite(result['validity_ns']).all()
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for weight in (model.backbone.entry_q_joint_in.weight,
                   model.query_projection[0].weight, model.query_head.weight):
        assert weight.grad is not None and torch.count_nonzero(weight.grad) > 0


def test_empty_support_requires_skip_and_no_optimizer(monkeypatch):
    model = _model(monkeypatch, [])
    bound, parents, issue = _bound()
    result = batch.evaluate_fit_batch(model, _features(1), bound,
                                      parents[1:], issue[1:],
                                      fold_id=1, pass_index=0, seed=19)
    assert result['count'] == 0 and result['skip_update']
    assert result['loss'].item() == 0
    assert all(p.grad is None for p in model.parameters())


def test_feature_and_axis_guards_run_before_forward(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    bound, parents, issue = _bound()
    features = _features(1)
    features['allowed'] = torch.ones(1, 5)
    with pytest.raises(ValueError, match='FEATURE_KEYS'):
        batch.evaluate_fit_batch(model, features, bound, parents[:1], issue[:1],
                                 fold_id=1, pass_index=0, seed=19)
    with pytest.raises(ValueError, match='AXES'):
        batch.evaluate_fit_batch(model, _features(1), bound, parents,
                                 issue, fold_id=1, pass_index=0, seed=19)
    assert calls == []


def test_fit_embargo_still_rejected(monkeypatch):
    model = _model(monkeypatch, [])
    bound, parents, issue = _bound()
    near = np.array([FOLD_BOUNDARIES_NS[0] - 288 * TICK_NS], dtype=np.int64)
    with pytest.raises(ValueError, match='EXACT_PARENT_ISSUE|ISSUE_PARTITION'):
        batch.evaluate_fit_batch(model, _features(1), bound, parents[1:],
                                 near, fold_id=1, pass_index=0, seed=19)
