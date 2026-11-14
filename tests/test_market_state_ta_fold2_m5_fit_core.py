"""Synthetic controls for the fixed fold 2 fit core; no native fit is run."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS
from gx1.scripts import run_market_state_ta_fold2_m5_fit_window_v1 as worker


def _case(monkeypatch, *, lr=1e-4, wrong_model_contract=False):
    parents = np.arange(47338, dtype=np.int64)
    issues = parents + 1
    bound = SimpleNamespace(parent_rows=parents, decision_ns=issues)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    monkeypatch.setattr(worker, 'canonical_model_state_sha256',
                        lambda _: worker.INITIAL_MODEL_SHA256)
    monkeypatch.setattr(worker, 'require_m5_equal_observation_axes',
                        lambda *args, **kwargs: {})
    seen = []

    def evaluate(model, features, state, future, rows, clocks, **kwargs):
        index = kwargs['fit_batch_start'] // 16
        seen.append(index)
        skip = index in (0, 2958)
        return {'loss': model.weight.sum() * (0 if skip else .01),
                'count': 0 if skip else 1, 'skip_update': skip,
                'loss_contract_sha256': worker.LOSS_SHA256,
                'model_contract_sha256': ('wrong' if wrong_model_contract else
                                          worker.MODEL_CONTRACT_SHA256)}

    monkeypatch.setattr(worker, 'evaluate_fit_batch', evaluate)
    return seen, dict(model=model, optimizer=optimizer, parent_rows=parents,
                      issue_ns=issues, state_bound=bound, future_bound=bound,
                      gather_features=lambda *args: {},
                      deadline_monotonic=float(time.monotonic() + 30))


def test_complete_one_pass_visits_tail_and_skips_empty_support(monkeypatch):
    seen, kwargs = _case(monkeypatch)
    result = worker.fit_one_pass(**kwargs)
    assert kwargs['optimizer'].defaults == ADAMW_DEFAULTS
    assert seen == list(range(2959))
    assert (result['optimizer_updates'], result['skipped_batches']) == (2957, 2)
    assert (result['training_records'][-1]['start'],
            result['training_records'][-1]['stop']) == (47328, 47338)
    assert result['checkpoint_writes'] == 0 and result['test_data_used'] is False


@pytest.mark.parametrize('fault,reason', [
    ('wrong_lr', 'EMPTY_OPTIMIZER_AND_TRAIN_MODE'),
    ('wrong_model_contract', 'LOSS_AND_SUPPORT'),
])
def test_rejects_wrong_recipe_or_model_before_update(monkeypatch, fault, reason):
    seen, kwargs = _case(monkeypatch, lr=1e-3 if fault == 'wrong_lr' else 1e-4,
                         wrong_model_contract=fault == 'wrong_model_contract')
    before = {key: value.clone() for key, value in kwargs['model'].state_dict().items()}
    with pytest.raises(RuntimeError, match='FOLD2_M5_FIT_' + reason):
        worker.fit_one_pass(**kwargs)
    assert seen == ([] if fault == 'wrong_lr' else [0])
    assert kwargs['optimizer'].state == {}
    assert all(torch.equal(value, before[key]) for key, value in
               kwargs['model'].state_dict().items())
