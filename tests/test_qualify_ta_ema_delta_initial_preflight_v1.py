"""Focused 318/256 receipt, normalization and Fold1 FIT-statistic guards."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import qualify_ta_ema_delta_initial_preflight_v1 as cold


def gather_case():
    plan = cold.checked(cold.GATHER_REFS['plan'])
    refs = {'campaign': {'view':plan['inputs']['view'],
                         'normal_ta':plan['inputs']['normal_ta'],
                         'material_result':plan['inputs']['material_result']},
            'gather':cold.GATHER_REFS}
    result,coverage,terminal = (cold.checked(cold.GATHER_REFS[key]) for key in
        ('result','coverage','terminal'))
    return {'inputs':refs},result,coverage,terminal


def test_full_318_256_gather_chain_is_exact_and_old_qualification_rejected():
    plan,result,coverage,terminal=gather_case()
    fit,later=cold.verify_gather(result,coverage,terminal,plan)
    assert (fit['rows'],fit['batches'],fit['tail_rows'])==(29574,1849,6)
    assert (later['rows'],later['batches'],later['tail_rows'])==(17489,1094,1)
    assert fit['surfaces']['seq_x']['shape_per_row']==[96,318]
    assert later['surfaces']['seq_d1']['shape_per_row']==[252,256]
    bad=deepcopy(coverage)
    bad['qualification']=cold.GATHER_REFS['result']
    with pytest.raises(RuntimeError,match='GATHER_CHAIN'):
        cold.verify_gather(result,bad,terminal,plan)
    bad=deepcopy(result)
    bad['later']['surfaces']['seq_d1']['shape_per_row']=[252,255]
    with pytest.raises(RuntimeError,match='GATHER_SURFACES'):
        cold.verify_gather(bad,coverage,terminal,plan)


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


def test_current_v2_fold1_cost_statistics_are_not_old_v1():
    binding=cold.ref(cold.COLD_TARGETS)
    assert binding['sha256']==cold.FIT_TARGET_SHA
    full=cold.load_targets(binding,47338)
    fit=cold.target_slice(full,0,29574)
    assert cold.axis_hash(fit['parent_rows'])=='1c5a8c36a94b8fa39d613f0bc122cf869dc9fa3c2f87dc64c0506acd2f55a90f'
    assert cold.axis_hash(fit['issue_ns'])=='ebb3489c45b01b6fd8e77bf27529d650beb6883ba84b57ee936dc549ec74e427'
    stats=cold.fit_statistics(fit)
    assert stats['sha256']==cold.STATS_SHA
    assert stats['sha256']!='284b244f244ee0e08297d89b1d5edf9865926fa944946d32662e06396a64fee2'
    assert stats['paired_quality_std_bps'].shape==(6,)


def test_wrong_fit_scope_stops_before_model_or_target_work():
    context={'fit_rows':29573}
    with pytest.raises(RuntimeError,match='COLD_AXES'):
        cold.probes(None,context,{},None,deadline=0.)
