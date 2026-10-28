"""Synthetic scope checks: no model construction, data load or optimizer step."""
from pathlib import Path

import pytest

from gx1.contracts import gx1_capped_execution_v1 as capped
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_research_window_v1 as dispatcher
from gx1.scripts import run_entry_native_direction_order_qualification_v1 as qualifier


def test_order_private_worker_requires_gpu_guard_before_campaign_or_data(monkeypatch,tmp_path):
    calls=[]
    def rejected():
        calls.append('guard')
        raise RuntimeError('synthetic guard rejection')
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',rejected)
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:calls.append('context'))
    with pytest.raises(RuntimeError,match='guard rejection'):
        qualifier.run_order_qualification(policy_path=tmp_path/'missing',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert calls==['guard'] and list(tmp_path.iterdir())==[]


def test_existing_guarded_dispatcher_selects_only_explicit_order_scope(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:
        (None,None,None,None,{'run_scope':authority.ORDER_PRIVATE_SCOPE}))
    def run(**kwargs):
        calls.append('order_worker')
        return {'scope':'synthetic'}
    monkeypatch.setattr(qualifier,'run_order_qualification',run)
    result=dispatcher.run_native_direction_research_window(policy_path=tmp_path/'window',
        policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert result=={'scope':'synthetic'} and calls==['guard','order_worker']
    assert list(tmp_path.iterdir())==[]
