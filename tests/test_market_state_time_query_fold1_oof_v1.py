from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from gx1.audit.score_market_state_time_query_fold1_oof_v1 import (
    FIT_CHECKPOINT_SCHEMA, VIEW_SHA256, _load_checkpoint, _score_batch,
)
from gx1.contracts.entry_native_direction_research_state_v1 import _tree_sha
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_scorer_v1 import TimeQueryMatchedScorer
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256


class _FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.feature_view_identity = {'contract_sha256': VIEW_SHA256}
        self.encodes = 0
        self.decodes = 0

    def encode_features(self, **_features):
        self.encodes += 1
        return torch.zeros((2, 4))

    def decode_query_grid(self, hidden, query_k):
        self.decodes += 1
        assert hidden.shape == (2, 4)
        assert query_k.shape[0] == 2 and query_k.shape[2] == 5
        assert query_k.shape[1] <= 32
        assert bool((query_k <= torch.tensor(NATIVE_K)[None, None, :]).all())
        probabilities = torch.tensor([0.4, 0.2, 0.4]).log()
        return probabilities.expand(2, query_k.shape[1], 5, 3)


class _BoundTargets:
    def __init__(self, parent_rows, issues):
        self.parents = parent_rows
        self.issues = issues
        self.calls = 0

    def gather(self, parent_rows, issue_ns, query_k, *, fold_id, partition):
        self.calls += 1
        assert fold_id == 1 and partition == 'score'
        assert len(parent_rows) <= 2 * 32
        assert np.array_equal(parent_rows[::len(parent_rows) // 2], self.parents)
        assert np.array_equal(issue_ns[::len(issue_ns) // 2], self.issues)
        validity = issue_ns[:, None] + query_k * TICK_NS
        mature = validity < FOLD_BOUNDARIES_NS[1]
        allowed = np.full(query_k.shape, 5, dtype=np.uint8)
        select = query_k[:, 0] == 1
        allowed[select, 0] = np.where(parent_rows[select] == self.parents[0], 1, 4)
        eligible = mature & (allowed != 7)
        return {'issue_ns': issue_ns.copy(), 'query_k': query_k.copy(),
                'validity_ns': validity, 'allowed': allowed,
                'eligible': eligible, 'mature': mature}


def test_one_encode_nine_bounded_blocks_and_matched_m5_side():
    parents = np.array([10, 11], dtype=np.int64)
    issues = np.array([FOLD_BOUNDARIES_NS[0], FOLD_BOUNDARIES_NS[0] + TICK_NS],
                      dtype=np.int64)
    model = _FrozenModel()
    bound = _BoundTargets(parents, issues)
    scorer = TimeQueryMatchedScorer()
    prior = (np.full((288, 5), 0.9), np.full((288, 5), 0.5),
             np.broadcast_to([0.4, 0.2, 0.4], (288, 5, 3)).copy())
    _score_batch(model, {}, parents, issues, bound=bound, scorer=scorer,
                 priors=prior, device=torch.device('cpu'), check_deadline=lambda: None)
    result = scorer.result()
    assert model.encodes == 1 and model.decodes == bound.calls == 9
    assert result['issue_axis']['1']['count'] == 2
    assert result['primary_M5_k1']['certified_side']['count'] == 2
    assert result['primary_M5_k1']['certified_side']['prior_brier_sum'] == pytest.approx(0.5)
    assert result['per_tf_k']['M5']['1']['active_side_unknown'] == 0
    assert result['per_tf_k']['D1']['288']['issues'] == 2


def test_checkpoint_requires_full_seal_exact_model_digest_and_fit_schema(tmp_path):
    source_commit = 'a' * 40
    model = _FrozenModel()
    model_state = copy.deepcopy(model.state_dict())
    model_state['weight'] = torch.tensor([2.0, 3.0])
    digest = canonical_model_state_sha256(model_state)
    state = {'schema_version': FIT_CHECKPOINT_SCHEMA, 'source_commit': source_commit,
             'fold_id': 1, 'feature_view_sha256': VIEW_SHA256,
             'model_state': model_state, 'model_state_sha256': digest,
             'binding': {'feature_view_identity': model.feature_view_identity},
             'optimizer_state': {'step': 1}, 'rng_state': {'seed': 1},
             'cursor': 29574, 'optimizer_updates': 1849, 'skipped_batches': 0}
    state['contract_sha256'] = _tree_sha(state)
    path = tmp_path / 'FINAL_FOLD1_FIT_STATE.pt'
    torch.save(state, path)
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert _load_checkpoint(model, fit_checkpoint_ref=ref,
                            expected_model_state_sha256=digest,
                            source_commit=source_commit,
                            device=torch.device('cpu')) == ref
    assert torch.equal(model.weight.detach(), torch.tensor([2.0, 3.0]))
    other = _FrozenModel()
    with pytest.raises(RuntimeError, match='EXPECTED_MODEL_DIGEST'):
        _load_checkpoint(other, fit_checkpoint_ref=ref,
                         expected_model_state_sha256='bad',
                         source_commit=source_commit,
                         device=torch.device('cpu'))
    state['cursor'] = 0
    state['contract_sha256'] = _tree_sha({key: value for key, value in state.items()
                                          if key != 'contract_sha256'})
    torch.save(state, path)
    partial_ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match='CHECKPOINT_SCOPE'):
        _load_checkpoint(other, fit_checkpoint_ref=partial_ref,
                         expected_model_state_sha256=digest,
                         source_commit=source_commit,
                         device=torch.device('cpu'))
    state['cursor'] = 29574
    state['binding']['feature_view_identity'] = {'contract_sha256': 'f' * 64}
    state['contract_sha256'] = _tree_sha({key: value for key, value in state.items()
                                          if key != 'contract_sha256'})
    torch.save(state, path)
    wrong_view_ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match='CHECKPOINT_SCOPE'):
        _load_checkpoint(other, fit_checkpoint_ref=wrong_view_ref,
                         expected_model_state_sha256=digest,
                         source_commit=source_commit,
                         device=torch.device('cpu'))
    state['binding']['feature_view_identity'] = model.feature_view_identity
    state['schema_version'] = 'gx1_market_state_time_query_private_state_v1'
    state['contract_sha256'] = _tree_sha({key: value for key, value in state.items()
                                          if key != 'contract_sha256'})
    torch.save(state, path)
    bad_ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match='CHECKPOINT_SCOPE'):
        _load_checkpoint(other, fit_checkpoint_ref=bad_ref,
                         expected_model_state_sha256=digest,
                         source_commit=source_commit,
                         device=torch.device('cpu'))
