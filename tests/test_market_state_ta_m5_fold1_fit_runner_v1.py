"""Fit warmstart orchestration; existing core migration and resume tests are reused."""
import random

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner


def _case(monkeypatch, *, fill_optimizer=False):
    model = torch.nn.Linear(2, 1)
    legacy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    calls = []

    def copy(source, candidate, *, expected_legacy_model_sha256):
        calls.append((source, candidate, expected_legacy_model_sha256))
        with torch.no_grad():
            candidate.weight.copy_(source.weight)
            candidate.bias.copy_(source.bias)
        # Constructor/migration consumption must not change the original checkpoint RNG.
        random.random()
        np.random.random()
        torch.rand(3)
        if fill_optimizer:
            optimizer.state[model.weight] = {'step': torch.tensor(1.)}
        return {'model_state_sha256': 'candidate'}

    monkeypatch.setattr(state, 'copy_legacy_weights_to_dedicated_m5', copy)
    random.seed(719)
    np.random.seed(719)
    torch.manual_seed(719)
    device = torch.device('cpu')
    rng = trainer._attended_session_rng_state(device=device)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    return model, legacy, optimizer, rng, expected, calls, device


def test_fit_warmstart_restores_original_rng_after_copy_and_keeps_optimizer_empty(monkeypatch):
    model, legacy, opt, rng, expected, calls, device = _case(monkeypatch)
    report = runner._ta_m5_apply_warmstart(
        model=model, optimizer=opt, legacy_model=legacy, legacy_rng=rng,
        origin={'model_state_sha256': 'bound-original'}, device=device)
    assert calls == [(legacy, model, 'bound-original')]
    assert report == {'model_state_sha256': 'candidate'}
    assert torch.equal(model.weight, legacy.weight)
    assert torch.equal(model.bias, legacy.bias)
    assert not opt.state
    assert (random.random(), float(np.random.random())) == expected[:2]
    assert torch.equal(torch.rand(3), expected[2])


def test_fit_warmstart_rejects_old_optimizer_history_before_copy(monkeypatch):
    model, legacy, opt, rng, _, calls, device = _case(monkeypatch)
    opt.state[model.weight] = {'step': torch.tensor(1.)}
    with pytest.raises(RuntimeError, match='M5_WARMSTART_REQUIRES_EMPTY_ADAMW'):
        runner._ta_m5_apply_warmstart(
            model=model, optimizer=opt, legacy_model=legacy, legacy_rng=rng,
            origin={'model_state_sha256': 'bound-original'}, device=device)
    assert calls == []


def test_fit_warmstart_rejects_history_introduced_during_migration(monkeypatch):
    model, legacy, opt, rng, _, _, device = _case(monkeypatch, fill_optimizer=True)
    with pytest.raises(RuntimeError, match='M5_WARMSTART_CHANGED_ADAMW'):
        runner._ta_m5_apply_warmstart(
            model=model, optimizer=opt, legacy_model=legacy, legacy_rng=rng,
            origin={'model_state_sha256': 'bound-original'}, device=device)
