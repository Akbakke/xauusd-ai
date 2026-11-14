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


def test_fixed_final_readout_uses_both_bound_panels_without_state_change(monkeypatch):
    from gx1.audit import score_market_state_ta_fold2_m5_fit_v1 as scorer
    from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective

    parents = np.arange(15000, dtype=np.int64)
    issues = parents + 1
    selected = {'TRAIN': np.arange(4096, dtype=np.int64),
                'LATER': np.arange(10000, 14096, dtype=np.int64)}
    calls = []

    class Bound:
        def gather(self, rows, clocks, *, fold_id, partition):
            assert fold_id == 2 and partition in ('fit', 'score')
            return {'allowed': np.ones((len(rows), 5), dtype=np.int64),
                    'eligible': np.ones((len(rows), 5), dtype=bool),
                    'mature': np.ones((len(rows), 5), dtype=bool)}

    def endpoint(model, features, state, future, rows, clocks, *, partition, **kwargs):
        calls.append((partition, int(rows[0]), len(rows)))
        raw = np.zeros((len(rows), 5, 2), dtype=np.float32)
        return {'measurement': {'rows': len(rows), 'batches': 1},
                'state_activity_side_logits': raw,
                'future_activity_side_logits': raw}

    monkeypatch.setattr(batch, 'evaluate_endpoint_batch', endpoint)
    monkeypatch.setattr(objective, 'merge_endpoint_measurements',
                        lambda prior, current: {'rows': (prior or {'rows': 0})['rows']
                                                  + current['rows'],
                                                'batches': (prior or {'batches': 0})['batches']
                                                   + current['batches']})
    monkeypatch.setattr(objective, 'finalize_endpoint_measurement', lambda x: x)
    monkeypatch.setattr(scorer, 'score_capture',
                        lambda data: {'rows': len(data['parents'])})
    monkeypatch.setattr(worker, 'canonical_model_state_sha256', lambda _: 'stable')
    model = torch.nn.Linear(1, 1)
    bound = Bound()
    context = {'axes': {'parent_rows': parents, 'decision_ns': issues},
               'train_positions': selected['TRAIN'],
               'later_positions': selected['LATER'],
               'state_bound': bound, 'future_bound': bound, 'priors': {},
               'features': lambda label, rows, clocks: {}}
    result = worker.evaluate_fixed_panels(
        model=model, context=context,
        deadline_monotonic=float(time.monotonic() + 30))
    assert result['evaluation_forwards'] == 512
    assert result['readouts'] == {'TRAIN': {'rows': 4096},
                                  'LATER': {'rows': 4096}}
    assert result['final_model_state_sha256'] == 'stable'
    assert calls[0] == ('fit', 0, 16)
    assert calls[255] == ('fit', 4080, 16)
    assert calls[256] == ('score', 10000, 16)
    assert calls[-1] == ('score', 14080, 16)
