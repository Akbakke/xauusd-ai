"""Minute source candidate: exact clock, old-endpoint parity and loss math."""
import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    NATIVE_MINUTES, MinuteQueryTargetTape, decode_raw_minute_grid,
    minute_allowed_set_loss, minute_proper_score_sums,
)
from gx1.contracts.market_state_time_query_loss_v2 import allowed_set_loss
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_scorer_v2 import proper_score_sums
from gx1.contracts.market_state_time_query_target_v1 import (
    MINUTE_NS, NATIVE_K, TimeQueryTargetTape,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def _model():
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, 'd_model', 4)
    object.__setattr__(cfg, 'entry_training_phase',
                       ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.entry_execution_profile = {'name': FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {'normalization': {}}
    return FactorizedTimeQueryMarketStateModel(native)


def test_minute_one_is_exact_and_old_native_endpoints_match():
    minute = np.array([1, 5, 15, 60, 240, 1440], np.int64)
    axis = minute * MINUTE_NS
    bits = np.full((len(axis), 5), 7, np.uint8)
    bits[0] = [4, 5, 2, 1, 7]
    bits[1, 0] = 1
    bits[2, 1] = 4
    bits[3, 2] = 2
    bits[4, 3] = 1
    bits[5, 4] = 5
    source = np.arange(10, 16, dtype=np.int64)
    path = np.ones_like(bits, dtype=np.bool_)
    gap = np.zeros(len(axis), np.uint8)
    tape = MinuteQueryTargetTape(axis, source, bits, path, gap)
    issue = np.array([0], np.int64)
    early = tape.gather_minutes(issue, np.ones((1, 5), np.int64),
                                fold_upper_ns=1441 * MINUTE_NS)
    assert early['allowed'][0, 0] == 4 and early['m1_source_row'][0, 0] == 10
    assert early['allowed'][0, 1] == 5 and early['eligible'][0, 1]
    assert not early['eligible'][0, 4]
    old = TimeQueryTargetTape(axis, source, bits).gather(
        issue, np.asarray(NATIVE_K, np.int64)[None, :],
        fold_upper_ns=1441 * MINUTE_NS)
    new = tape.gather_minutes(
        issue, np.asarray(NATIVE_MINUTES, np.int64)[None, :],
        fold_upper_ns=1441 * MINUTE_NS)
    for key in ('allowed', 'm1_source_row', 'validity_ns', 'eligible', 'mature'):
        np.testing.assert_array_equal(new[key], old[key])
    axis[0] = 0; bits[0, 0] = 1; source[0] = 100
    assert tape.gather_minutes(issue, np.ones((1, 5), np.int64),
                               fold_upper_ns=1441 * MINUTE_NS)['allowed'][0, 0] == 4


def test_missing_and_immature_minute_never_read_future():
    tape = MinuteQueryTargetTape(
        np.array([1, 5], np.int64) * MINUTE_NS,
        np.array([10, 11], np.int64),
        np.full((2, 5), 4, np.uint8),
        np.ones((2, 5), np.bool_), np.zeros(2, np.uint8))
    issue = np.array([0], np.int64)
    query = np.array([[2, 2, 5, 5, 5]], np.int64)
    got = tape.gather_minutes(issue, query, fold_upper_ns=4 * MINUTE_NS)
    assert got['mature'][0, :2].all()
    assert not got['eligible'][0, :2].any()
    assert (got['allowed'][0, :2] == 7).all()
    assert not got['mature'][0, 2:].any()
    assert (got['m1_source_row'][0] == -1).all()
    query[0, 0] = 6
    with pytest.raises(ValueError, match='QUERY_DOMAIN'):
        tape.gather_minutes(issue, query, fold_upper_ns=10 * MINUTE_NS)


def test_existing_head_endpoint_output_and_loss_score_math_match():
    torch.manual_seed(17)
    model = _model()
    hidden = torch.randn(2, 4, dtype=torch.float32)
    old_query = torch.tensor(NATIVE_K, dtype=torch.int64)[None, None, :].expand(2, 1, -1)
    minute_query = torch.tensor(NATIVE_MINUTES, dtype=torch.int64)[None, None, :].expand(2, 1, -1)
    old_raw = model.decode_raw_query_grid(hidden, old_query)
    new_raw = decode_raw_minute_grid(model, hidden, minute_query)
    torch.testing.assert_close(new_raw, old_raw, rtol=0, atol=0)
    raw = new_raw[:, 0].detach().clone().requires_grad_(True)
    allowed = torch.tensor([[1, 2, 4, 5, 7], [4, 5, 1, 2, 7]], dtype=torch.uint8)
    ready = torch.ones((2, 5), dtype=torch.bool)
    minute = minute_query[:, 0]
    actual = minute_allowed_set_loss(raw, allowed, ready, ready, minute)
    old = allowed_set_loss(raw, allowed, ready, ready, old_query[:, 0])
    torch.testing.assert_close(actual['loss'], old['loss'], rtol=0, atol=0)
    actual['loss'].backward()
    assert torch.count_nonzero(raw.grad) > 0
    a = minute_proper_score_sums(raw.detach().numpy(), allowed.numpy(),
                                  ready.numpy(), ready.numpy(), minute.numpy())
    b = proper_score_sums(raw.detach().numpy(), allowed.numpy(),
                          ready.numpy(), ready.numpy(), old_query[:, 0].numpy())
    for tf in a['per_tf']:
        for metric in a['per_tf'][tf]:
            for name, value in a['per_tf'][tf][metric].items():
                assert value == pytest.approx(b['per_tf'][tf][metric][name], abs=3e-7)


def test_minute_one_query_backpropagates_into_existing_head():
    torch.manual_seed(19)
    model = _model()
    hidden = torch.randn(1, 4, dtype=torch.float32)
    minute = torch.ones((1, 1, 5), dtype=torch.int64)
    raw = decode_raw_minute_grid(model, hidden, minute)[:, 0]
    allowed = torch.full((1, 5), 7, dtype=torch.uint8)
    allowed[0, 0] = 4
    ready = torch.ones((1, 5), dtype=torch.bool)
    result = minute_allowed_set_loss(raw, allowed, ready, ready, minute[:, 0])
    result['loss'].backward()
    assert result['count'] == 1 and not result['skip_update']
    assert torch.count_nonzero(model.query_head.weight.grad) > 0
