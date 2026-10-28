"""Synthetic272 routing/input/checkpoint boundaries; zero real native work.

Canonical state/optimizer mathematics is already qualified separately in271.
These tests use tiny arrays/fake contexts and a toy tensor checkpoint. No real
native constructor, model forward/backward, optimizer step or stats fit runs.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.contracts import gx1_capped_execution_v1 as capped
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_research_window_v1 as runner


def test_direct_public_call_checks_cuda_guard_before_context(monkeypatch,tmp_path):
    calls=[]
    def rejected():calls.append('guard');raise RuntimeError('synthetic CUDA guard rejection')
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',rejected)
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:calls.append('context'))
    with pytest.raises(RuntimeError,match='CUDA guard rejection'):
        runner.run_native_direction_research_window(policy_path=tmp_path/'missing.json',policy_file_sha256='0'*64,progress_path=tmp_path/'PROGRESS.json')
    assert calls==['guard'] and not (tmp_path/'PROGRESS.json').exists()


def test_context_rejection_precedes_output_or_data_preparation(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    def rejected(*args):calls.append('context');raise RuntimeError('synthetic context rejection')
    monkeypatch.setattr(authority,'require_native_direction_research_context',rejected)
    monkeypatch.setattr(runner,'_validate_metadata',lambda *args:calls.append('metadata'))
    with pytest.raises(RuntimeError,match='context rejection'):
        runner.run_native_direction_research_window(policy_path=tmp_path/'missing.json',policy_file_sha256='0'*64,progress_path=tmp_path/'PROGRESS.json')
    assert calls==['guard','context'] and list(tmp_path.iterdir())==[]


def _targets():
    rows=47602;step=300_000_000_000
    D=1735682100000000000-(rows-1-np.arange(rows,dtype=np.int64))*step
    arrays={'parent_rows':np.arange(rows,dtype=np.int64),'decision_ns':D}
    target={'terminal_time_ns':D[:,None]+np.asarray([1,5,15],np.int64)[None,:]*60_000_000_000,
        'observed':np.ones((rows,3),np.bool_),'sign':np.ones((rows,3),np.int8)}
    return arrays,target


def test_fixed_four_input_geometry_and_all_horizons_precede_model():
    arrays,target=_targets();batches,report=runner._qualify_fixed_targets(arrays,target,cutoff_ns=1735689600000000000,np=np)
    assert [item['parent_rows'] for item in report]==[[47595,47596],[47597,47598]]
    assert all(item['eligible_cells_per_horizon']==[2,2,2] for item in report)
    assert len(batches)==2 and all(value['sign'].shape==(2,3) for value in batches)
    assert max(batches[-1]['terminal_time_ns'][:,2])<=arrays['decision_ns'][-1]


@pytest.mark.parametrize('fault',['missing15','tie1','terminal','parents','margin'])
def test_fixed_target_guard_rejects_without_alternative_selection(fault):
    arrays,target=_targets();at=list(runner.PARENTS[:2])
    if fault=='missing15':target['observed'][at,2]=False;target['sign'][at,2]=0
    elif fault=='tie1':target['sign'][at,0]=0
    elif fault=='terminal':target['terminal_time_ns'][at[0],2]+=1
    elif fault=='parents':arrays['parent_rows'][runner.PARENTS[0]]+=1
    elif fault=='margin':
        arrays['decision_ns'][runner.PARENTS[-1]]=arrays['decision_ns'][-1]-10*60_000_000_000
        target['terminal_time_ns']=arrays['decision_ns'][:,None]+np.asarray([1,5,15],np.int64)[None,:]*60_000_000_000
    with pytest.raises(RuntimeError):runner._qualify_fixed_targets(arrays,target,cutoff_ns=1735689600000000000,np=np)


def _gc():
    timeframes=['M5','M15','H1','H4','D1']
    families=['momentum_flow_encoder','price_action_candle_encoder','session_regime_encoder','smc_liquidity_encoder','vol_compression_encoder']
    layout={'timeframes':timeframes,'family_indices':{tf:{name:[2*i,2*i+1] for i,name in enumerate(families)} for tf in timeframes},
        'value_mask_pairs':{tf:[[2*i,2*i+1] for i in range(5)] for tf in timeframes}}
    raw={'features':{'gc_seq_'+tf.lower():np.ones((2,1,10),np.float32) for tf in timeframes}}
    return raw,layout


def test_gc_support_preserves_observed_zero_values():
    raw,layout=_gc()
    for value in raw['features'].values():value[:,:,::2]=0
    report=runner._gc_support(raw,layout,np)
    assert len(report)==5 and all(len(value)==5 for value in report.values())
    assert all(item['observed_cells_per_parent']==[1,1] and item['raw_nonzero_values_per_parent']==[0,0]
        for families in report.values() for item in families.values())


@pytest.mark.parametrize('fault',['nonbinary','storage','unsupported'])
def test_gc_support_rejects_real_mask_domain_and_storage_errors(fault):
    raw,layout=_gc();value=raw['features']['gc_seq_d1']
    if fault=='nonbinary':value[0,0,1]=.5
    elif fault=='storage':value[0,0,1]=0
    elif fault=='unsupported':value[:,:,0:2]=0
    with pytest.raises(RuntimeError):runner._gc_support(raw,layout,np)


def test_checkpoint_atomic_roundtrip_no_replace_hash_and_size_guards(tmp_path,monkeypatch):
    path=tmp_path/'STATE.pt';value={'synthetic_tensor':torch.arange(3,dtype=torch.float32)}
    counters={key:0 for key in ('staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns')}
    ref=runner._save_state(path,value,torch=torch,maximum=1024*1024,counters=counters)
    loaded=runner._load_state(ref,torch=torch,maximum=1024*1024,counters=counters)
    assert torch.equal(loaded['synthetic_tensor'],value['synthetic_tensor']) and path.stat().st_mode&0o222==0
    with pytest.raises(RuntimeError,match='CHECKPOINT_EXISTS'):runner._save_state(path,value,torch=torch,maximum=1024*1024,counters=counters)
    assert counters=={'staged_state_load_calls':1,'staged_state_load_returns':1,'resume_state_load_calls':1,'resume_state_load_returns':1}
    original_load,original_save=torch.load,torch.save
    calls=[];monkeypatch.setattr(torch,'load',lambda *args,**kwargs:calls.append('load'))
    with pytest.raises(RuntimeError,match='LOAD_SHA_SIZE'):runner._load_state({**ref,'sha256':'0'*64},torch=torch,maximum=1024*1024,counters=counters)
    with pytest.raises(RuntimeError,match='LOAD_SHA_SIZE'):runner._load_state(ref,torch=torch,maximum=1,counters=counters)
    assert calls==[]
    with pytest.raises(RuntimeError,match='CHECKPOINT_BYTES'):runner._save_state(tmp_path/'TOO_BIG.pt',value,torch=torch,maximum=1,counters=counters)
    assert not (tmp_path/'TOO_BIG.pt').exists()
    monkeypatch.setattr(torch,'load',original_load)
    monkeypatch.setattr(torch,'save',lambda state,handle:original_save({'synthetic_tensor':state['synthetic_tensor']+1},handle))
    with pytest.raises(RuntimeError,match='CHECKPOINT_STAGED_ROUNDTRIP'):
        runner._save_state(tmp_path/'BAD_SERIALIZATION.pt',value,torch=torch,maximum=1024*1024,counters=counters)
    assert not (tmp_path/'BAD_SERIALIZATION.pt').exists() and not list(tmp_path.glob('.BAD_SERIALIZATION.pt.*'))


def test_gradient_evidence_names_zero_group_without_weakening_the_gate():
    model=torch.nn.Module();model.direction_head=torch.nn.Linear(2,3);model.backbone=torch.nn.Module()
    model.backbone.entry_q_joint_in=torch.nn.Linear(2,2);model.backbone.entry_q_joint_norm=torch.nn.LayerNorm(2)
    model.backbone.gc_mtf_family_proj=torch.nn.ModuleDict();model.backbone.gc_mtf_feature_context_gate=torch.nn.ModuleDict()
    for tf in ('m5','m15','h1','h4','d1'):
        for i in range(5):
            key=tf+'__'+str(i)
            model.backbone.gc_mtf_family_proj[key]=torch.nn.Linear(1,1)
            model.backbone.gc_mtf_feature_context_gate[key]=torch.nn.Linear(1,1)
    model.backbone.other=torch.nn.Linear(1,1)
    for p in model.parameters():p.grad=torch.ones_like(p)
    evidence=runner._gradient_evidence(model,torch)
    assert all(value>0 for key,value in evidence.items() if isinstance(value,int))
    for key,layer in model.backbone.gc_mtf_feature_context_gate.items():
        if key.startswith('h4__'):
            for p in layer.parameters():p.grad.zero_()
    with pytest.raises(RuntimeError,match='REQUIRED_GRADIENT_GROUP:gc_gate_h4'):
        runner._gradient_evidence(model,torch)


def test_gc_support_reports_empty_family_without_hiding_storage_errors():
    raw,layout=_gc();value=raw['features']['gc_seq_d1'];value[:,:,0:2]=0
    report=runner._gc_support(raw,layout,np,allow_empty=True)
    first=next(iter(report['D1'].values()))
    assert first['observed_cells_per_parent']==[0,0] and first['raw_nonzero_values_per_parent']==[0,0]
    value[0,0,0]=1
    with pytest.raises(RuntimeError,match='GC_MASK_STORAGE'):
        runner._gc_support(raw,layout,np,allow_empty=True)
