"""Focused cold initial preflight guards; never construct a real model here."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import qualify_market_state_ta_cold_initial_preflight_v1 as cold


def test_raw_receipt_accepts_only_exact_completed_four_batch_chain(monkeypatch):
    bindings = cold.inputs()
    result = cold.read(bindings['raw_result'])
    terminal = cold.read(bindings['raw_terminal'])
    batches = cold.verify_raw(result, terminal, bindings)
    assert [b['name'] for b in batches] == ['FIT_B16','FIT_TAIL6','SCORE_B16','SCORE_TAIL1']
    stale = deepcopy(result)
    stale['batches'][0]['parent_rows_sha256'] = '0'*64
    monkeypatch.setattr(cold.raw_audit, 'summarize_batch', lambda *args: batches[0])
    cold.verify_actual_batch('FIT_B16', {}, None, None, batches[0])
    with pytest.raises(RuntimeError, match='RAW_BATCH_FIT_B16'):
        cold.verify_actual_batch('FIT_B16', {}, None, None, stale['batches'][0])
    wrong = deepcopy(terminal)
    wrong['result']['sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='RAW_RECEIPTS'):
        cold.verify_raw(result, wrong, bindings)
    wrong = deepcopy(result)
    wrong['batches'][2]['name'] = 'SCORE_OLD'
    with pytest.raises(RuntimeError, match='RAW_RECEIPTS'):
        cold.verify_raw(wrong, terminal, bindings)


def test_normalized_eight_numeric_surfaces_and_nonfinite_guard():
    class Backbone:
        def _normalize_input_surface(self, value, *, surface):
            assert surface in ('signal','ctx_cont','mtf_m5','mtf_m15','mtf_h1','mtf_h4','mtf_d1')
            return value * 2
    model = SimpleNamespace(core=SimpleNamespace(dual=SimpleNamespace(
        state_model=SimpleNamespace(backbone=Backbone()))))
    keys = ('seq_x','snap_x','ctx_cont','seq_m5','seq_m15','seq_h1','seq_h4','seq_d1')
    features = {key:torch.ones((16,2),dtype=torch.float32) for key in keys}
    features['ctx_cat'] = torch.zeros((16,1),dtype=torch.int64)
    details = cold.normalized_surfaces(model, features)
    assert set(details) == set(keys)
    assert all(item['nonzero_cells'] == 32 for item in details.values())
    features['seq_h4'][0,0] = float('nan')
    with pytest.raises(RuntimeError, match='NORMALIZED_seq_h4'):
        cold.normalized_surfaces(model, features)


def test_score_probe_uses_only_one_fold1_score_batch(monkeypatch):
    rows = np.arange(16,dtype=np.int64)
    issues = 1727740800000000000 + rows * 300_000_000_000
    features = {'sentinel':object()}
    context = {'axes':{'parent_rows':rows,'decision_ns':issues},
        'selected':rows, 'score_tape':{'structure_hit':np.zeros((16,2),dtype=bool)},
        'features':lambda label,r,i:features}
    monkeypatch.setattr(cold,'verify_actual_batch',lambda *args:None)
    def scored(model, shallow, *, deadline, fold_id, partition, collect_decision):
        assert fold_id == 1 and partition == 'score' and collect_decision is True
        assert len(shallow['selected']) == 16 and len(shallow['score_tape']['structure_hit']) == 16
        assert shallow['features']('SCORE',rows,issues) is features
        return {'rows':16,'model_forwards':1}, {'m5_cost_path_bps':np.ones((16,6,2,3),np.float32)}
    monkeypatch.setattr(cold,'score',scored)
    digest = cold.score_probe(object(),context,{},deadline=1000.)
    assert len(digest) == 64


def test_plan_binds_cold_raw_receipt_two_forwards_and_no_fit():
    plan = cold.build_plan()
    assert plan['inputs']['raw_result'] == cold.raw_refs()['raw_result']
    assert plan['inputs']['fit_targets'] == cold.ref(cold.COLD_TARGETS)
    assert plan['scope']['fit_rows'] == 29574
    assert plan['scope']['model_forwards'] == 2
    assert plan['scope']['optimizer_steps'] == 0
    assert plan['scope']['test_data_used'] is False
