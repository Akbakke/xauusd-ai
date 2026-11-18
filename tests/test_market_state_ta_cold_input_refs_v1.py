"""Retired warm and cold direction providers fail before M1-derived reads."""
from pathlib import Path

import pytest
import torch

from gx1.audit import market_state_ta_apr_jun_campaign_inputs_v1 as adapter


@pytest.mark.parametrize('entry', (
    'campaign_input_refs', 'cold_campaign_input_refs', '_tape',
    'prepare_warm', 'prepare_cold',
))
def test_retired_ta_entry_points_refuse_before_any_read(monkeypatch, entry):
    def forbidden(*args, **kwargs):
        raise AssertionError('retired direction provider reached input IO')
    monkeypatch.setattr(adapter, 'read', forbidden)
    monkeypatch.setattr(adapter, 'ref', forbidden)
    monkeypatch.setattr(adapter.np, 'load', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(RuntimeError,
                       match='^DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5$'):
        if entry == 'campaign_input_refs':
            adapter.campaign_input_refs()
        elif entry == 'cold_campaign_input_refs':
            adapter.cold_campaign_input_refs()
        elif entry == '_tape':
            adapter._tape({}, None, None)
        elif entry == 'prepare_warm':
            adapter.prepare(device=torch.device('cpu'))
        else:
            adapter.prepare(device=torch.device('cpu'), cold_inputs={})
