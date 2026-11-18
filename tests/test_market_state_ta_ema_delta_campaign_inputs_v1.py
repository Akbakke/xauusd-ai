"""Retired EMA direction dependencies fail before input or metadata reads."""
from pathlib import Path

import pytest
import torch

from gx1.audit import market_state_ta_ema_delta_campaign_inputs_v1 as adapter


@pytest.mark.parametrize('entry', ('successor_input_refs', '_receipts', 'prepare'))
def test_retired_ema_entry_points_refuse_before_any_read(monkeypatch, entry):
    def forbidden(*args, **kwargs):
        raise AssertionError('retired direction provider reached input IO')
    monkeypatch.setattr(adapter.legacy, 'read', forbidden)
    monkeypatch.setattr(adapter.legacy, 'ref', forbidden)
    monkeypatch.setattr(adapter.legacy, '_tape', forbidden)
    monkeypatch.setattr(adapter.legacy.np, 'load', forbidden)
    monkeypatch.setattr(adapter.gathered, 'input_refs', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(RuntimeError,
                       match='^DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5$'):
        if entry == 'successor_input_refs':
            adapter.successor_input_refs()
        elif entry == '_receipts':
            adapter._receipts({})
        else:
            adapter.prepare(device=torch.device('cpu'), inputs={})
