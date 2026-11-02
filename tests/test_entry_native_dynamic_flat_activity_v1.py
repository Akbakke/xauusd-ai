"""First-query flat-activity targets, tensor loss and streaming score; no model calls."""
import json
import math

import numpy as np
import pytest
import torch

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.contracts import entry_native_direction_research_v1 as research
from gx1.contracts.entry_dynamic_mtf_state_report_v1 import FirstFlatActivityAccumulator
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
from tests.test_entry_dynamic_mtf_state_targets import START, atr_fixture, build, frame_from_closes, SOURCE, ATR_SOURCE
from tests.test_entry_native_dynamic_current_math_v1 import _small_loss
from tests.test_entry_native_dynamic_current_routing_v1 import _arguments
from tests.test_entry_native_direction_feature_view_v1 import bound_case


def _query(series, issue, *, cutoff=None, enabled=True):
    parent = np.array([0], np.int64)
    issue = np.array([issue], np.int64)
    bound = targets.bind_dynamic_mtf_state_parents(series, parent_rows=parent, issue_ns=issue)
    return targets.query_dynamic_mtf_states(bound, parent_rows=parent, issue_ns=issue,
        cutoff_ns=int(START + 100 * targets.MINUTE_NS) if cutoff is None else cutoff,
        lead_minutes=np.array([1, 2], np.int64), **({'include_first_flat_activity': True} if enabled else {}))


def test_complete_ambiguous_range_gives_nonflat_without_inventing_direction():
    frame = frame_from_closes([100.] * 12)
    frame.loc[5, ['high', 'low']] = [101., 99.]
    q = _query(build(frame), START + 5 * targets.MINUTE_NS)
    assert not q['observed'][0, 0, 0] and q['state'][0, 0, 0] == targets.UNKNOWN
    assert q['first_flat_activity_observed'][0, 0] and not q['first_flat_activity_is_flat'][0, 0]
    assert not q['first_flat_activity_observed'][0, 1:].any()  # not enough local history
    flat = _query(build(frame_from_closes([100.] * 12)), START + 5 * targets.MINUTE_NS)
    assert flat['observed'][0, 0, 0] and flat['state'][0, 0, 0] == targets.FLAT
    assert flat['first_flat_activity_is_flat'][0, 0]
    old = _query(build(frame), START + 5 * targets.MINUTE_NS, enabled=False)
    assert not any(key.startswith('first_flat_activity') for key in old)


def test_activity_masks_unavailable_atr_gap_missing_and_immature():
    minute = targets.MINUTE_NS
    cutoff = int(START + 120 * minute)
    frame = frame_from_closes([100.] * 12)
    no_atr = targets.build_dynamic_mtf_state_series(frame, tf_atr=atr_fixture(cutoff, values=[0.]),
        source_ref=SOURCE, atr_source_ref=ATR_SOURCE, cutoff_ns=cutoff)
    q = _query(no_atr, START + 5 * minute)
    assert not q['first_flat_activity_observed'].any()
    gap_frame = frame_from_closes([100.] * 12, minutes=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    gap = build(gap_frame)
    missing_frame = frame_from_closes([100.] * 12, minutes=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])
    q_missing = _query(build(missing_frame), START + 5 * minute)
    assert not q_missing['first_flat_activity_observed'].any()
    q_gap = _query(gap, START + 10 * minute)
    assert not q_gap['first_flat_activity_observed'][0, 0]
    q_immature = _query(build(frame), START + 5 * minute, cutoff=int(START + 6 * minute))
    assert not q_immature['first_flat_activity_observed'].any()


def test_objective_recipe_and_view_are_explicit_and_old_identity_stable(bound_case):
    old = research.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.)
    assert old['contract_sha256'] == '7b62e039ac38d616106180a4b11ee614f0f5dbf857381e4cbdb3dffdbacfee33'
    new = research.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.,
                                                        first_flat_activity_weight=0.25)
    assert new['contract_sha256'] != old['contract_sha256']
    assert new['first_flat_activity_is_future_label_only'] and not new['first_flat_activity_native_admission']
    recipe = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
        current_state_first_loss_weight=1., first_flat_activity_weight=0.25)
    view = views.bind_native_direction_feature_view(**_arguments(bound_case), current_state_first_loss_weight=1.,
                                                     first_flat_activity_weight=0.25)
    assert recipes.require_native_direction_research_recipe(recipe) == recipe
    assert view['research_objective']['contract_sha256'] == recipe['research_objective_sha256']
    assert recipe['first_flat_activity_weight'] == view['first_flat_activity_weight'] == 0.25
    with pytest.raises(RuntimeError):
        recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
                                                  first_flat_activity_weight=0.25)


@pytest.mark.parametrize('weight', [0., -1., float('inf'), float('nan'), 1, True])
def test_invalid_activity_weight_rejected(weight):
    with pytest.raises(RuntimeError, match='STATE_FIRST_ACTIVITY_WEIGHT'):
        research.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.,
                                                     first_flat_activity_weight=weight)


def test_batch_keeps_future_activity_labels_out_of_model_features(monkeypatch):
    from tests.test_entry_native_dynamic_current_routing_v1 import _batch_fixture
    prepared, target, parent = _batch_fixture()
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch',
        lambda *args, **kwargs: {'features': {'seq_x': np.zeros((1, 1, 1), np.float32)}, 'witness': {}})
    raw, outcome = prefix._batch(prepared, target,
        {'current_state_first_loss_weight': 1., 'first_flat_activity_weight': 0.25}, parent, np)
    for key in ('first_flat_activity_observed', 'first_flat_activity_is_flat'):
        assert key in outcome and key not in raw['features']
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch',
        lambda *args, **kwargs: {'features': {'first_flat_activity_is_flat': np.zeros((1, 5), bool)}, 'witness': {}})
    with pytest.raises(RuntimeError, match='FIRST_ACTIVITY_FEATURE_COLLISION'):
        prefix._batch(prepared, target,
            {'current_state_first_loss_weight': 1., 'first_flat_activity_weight': 0.25}, parent, np)
    with pytest.raises(RuntimeError, match='FIRST_ACTIVITY_TARGET_SERIES_REQUIRED'):
        prefix._batch(prepared, {},
            {'current_state_first_loss_weight': 1., 'first_flat_activity_weight': 0.25}, parent, np)


def _activity_args(monkeypatch):
    args = _small_loss(monkeypatch)
    args.update(first_state_loss_weight=2., first_flat_activity_weight=0.25,
        first_flat_activity_observed=torch.tensor([[True, True, True, True, False]]),
        first_flat_activity_is_flat=torch.tensor([[True, False, False, False, False]]))
    return args


def test_binary_loss_uses_known_and_unknown_direction_with_own_denominator(monkeypatch):
    args = _activity_args(monkeypatch)
    result = research.native_dynamic_mtf_state_loss(**args)
    logp = torch.log_softmax(args['initial_state_logits'], -1)[0]
    nll = torch.stack((-logp[0, 1], -torch.logsumexp(logp[1, [0, 2]], -1),
                       -torch.logsumexp(logp[2, [0, 2]], -1),
                       -torch.logsumexp(logp[3, [0, 2]], -1))).mean()
    assert result['first_flat_activity_eligible_cells'] == 4
    torch.testing.assert_close(result['first_flat_activity_loss'], nll)
    torch.testing.assert_close(result['loss'], result['sequence_loss'] + 2. * result['first_state_loss'] + 0.25 * nll)


def test_activity_only_and_fully_empty_batches(monkeypatch):
    args = _activity_args(monkeypatch)
    args['state'].fill_(-1); args['observed'].fill_(False); args['first_flat_activity_is_flat'].fill_(False)
    result = research.native_dynamic_mtf_state_loss(**args)
    assert result['update_allowed'] and result['counts']['eligible_cells'] == 0
    torch.testing.assert_close(result['loss'], .25 * result['first_flat_activity_loss'])
    args['first_flat_activity_observed'].fill_(False)
    result = research.native_dynamic_mtf_state_loss(**args)
    assert not result['update_allowed'] and result['loss'] is None


@pytest.mark.parametrize('fault', ['flat_unknown', 'flat_outside', 'known_nonflat_as_flat', 'bad_dtype', 'immature'])
def test_loss_rejects_inconsistent_activity(fault, monkeypatch):
    args = _activity_args(monkeypatch)
    if fault == 'flat_unknown': args['first_flat_activity_is_flat'][0, 1] = True
    if fault == 'flat_outside': args['first_flat_activity_is_flat'][0, 4] = True
    if fault == 'known_nonflat_as_flat': args['first_flat_activity_is_flat'][0, 2] = True
    if fault == 'bad_dtype': args['first_flat_activity_is_flat'] = args['first_flat_activity_is_flat'].to(torch.int8)
    if fault == 'immature': args['cutoff_ns'] = int(args['validity_ns'][0, 0])
    with pytest.raises(RuntimeError, match='STATE_FIRST_ACTIVITY'):
        research.native_dynamic_mtf_state_loss(**args)


def test_binary_report_scores_unknown_direction_and_same_current_markov():
    grid = np.empty((5, 4, 3), np.float64)
    grid[:] = [0.25, 0.5, 0.25]
    grid[:, 3] = [0.45, 0.1, 0.45]
    score = FirstFlatActivityAccumulator(grid)
    p = np.repeat(np.array([[.5], [.9], [0.], [1.]]), 5, axis=1)
    mask = np.repeat(np.array([[1], [1], [1], [0]], bool), 5, axis=1)
    flat = np.repeat(np.array([[1], [0], [0], [0]], bool), 5, axis=1)
    state = np.repeat(np.array([[1], [-1], [2], [-1]], np.int8), 5, axis=1)
    cur = np.repeat(np.array([[1], [-1], [2], [1]], np.int8), 5, axis=1)
    score.update(p, activity_observed=mask, is_flat=flat, state=state, state_observed=state >= 0,
                 current_state=cur, current_observed=cur >= 0)
    r = score.finalize()
    assert r['counts']['observed'] == [3] * 5 and r['counts']['direction_unknown'] == [1] * 5
    assert r['counts']['flat'] == [1] * 5 and r['counts']['nonflat'] == [2] * 5
    assert r['model']['logloss'] == pytest.approx([-(math.log(.5) + math.log(.1)) / 3] * 5)
    assert r['markov']['logloss'] == pytest.approx([-(math.log(.5) + math.log(.9) + math.log(.5)) / 3] * 5)
    assert r['model']['tied'] == [1] * 5
    json.dumps(r, allow_nan=False)
    score = FirstFlatActivityAccumulator(grid)
    p[0] = 0.
    score.update(p, activity_observed=mask, is_flat=flat, state=state, state_observed=state >= 0,
                 current_state=cur, current_observed=cur >= 0)
    r = score.finalize()
    assert r['model']['logloss'] == [None] * 5 and r['model']['logloss_infinite'] == [1] * 5


def test_extreme_logits_are_stable_and_inactive_labels_are_rejected(monkeypatch):
    args = _activity_args(monkeypatch)
    args['initial_state_logits'][0, :, :] = torch.tensor([10000., -10000., 10000.])
    result = research.native_dynamic_mtf_state_loss(**args)
    assert torch.isfinite(result['first_flat_activity_loss']) and torch.isfinite(result['loss'])
    args['first_flat_activity_weight'] = None
    with pytest.raises(RuntimeError, match='STATE_FIRST_ACTIVITY_UNBOUND'):
        research.native_dynamic_mtf_state_loss(**args)
