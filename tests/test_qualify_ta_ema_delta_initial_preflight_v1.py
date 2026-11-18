"""Numeric normalization guards and retired M1 initial-plan refusal."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import qualify_ta_ema_delta_initial_preflight_v1 as cold


def test_numeric_model_reference_parity_guard(monkeypatch):
    features={name:torch.zeros(shape,dtype=torch.int64 if name=='ctx_cat' else torch.float32)
              for name,shape in cold.EXPECTED_SHAPES.items()}
    class Backbone:
        def _normalize_input_surface(self,value,*,surface):return value
    model=SimpleNamespace(core=SimpleNamespace(dual=SimpleNamespace(
        state_model=SimpleNamespace(backbone=Backbone()))))
    ta={'surfaces':{surface:{} for surface in cold.SURFACES.values()}}
    monkeypatch.setattr(cold.normal,'apply_surface_normalization',lambda x,stats:x)
    details=cold.normalize_probe(model,features,ta,'FIT')
    assert set(details)==set(cold.SURFACES)
    assert all(item['max_abs_reference_error']==0 for item in details.values())
    monkeypatch.setattr(cold.normal,'apply_surface_normalization',lambda x,stats:x+1)
    with pytest.raises(RuntimeError,match='MODEL_NORMALIZATION_seq_x'):
        cold.normalize_probe(model,features,ta,'FIT')
    features['seq_x']=torch.zeros((16,96,317),dtype=torch.float32)
    with pytest.raises(RuntimeError,match='B16_SHAPES'):
        cold.normalize_probe(model,features,ta,'FIT')


def test_retired_initial_plan_refuses_before_reading_receipts(monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail('retired initial plan opened a receipt or target')

    monkeypatch.setattr(cold, 'checked', unexpected_read)
    monkeypatch.setattr(cold, 'load_targets', unexpected_read)
    with pytest.raises(RuntimeError, match='DIRECTION_M1_PROVENANCE_RETIRED'):
        cold.build_plan()


def test_wrong_fit_scope_stops_before_model_or_target_work():
    context={'fit_rows':29573}
    with pytest.raises(RuntimeError,match='COLD_AXES'):
        cold.probes(None,context,{},None,deadline=0.)
