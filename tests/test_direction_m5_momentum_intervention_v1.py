import numpy as np
import pytest
import torch
from gx1.audit import direction_m5_momentum_intervention_v1 as audit
from gx1.audit import direction_gc_mtf_intervention_v1 as gc


def test_single_and_combined_neutralization_preserve_other_fields_and_input():
    layer=torch.nn.Linear(4,1,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1.,10.,100.,1000.]]))
    values=torch.tensor([[[1.,2.,3.,4.],[2.,3.,4.,5.]]]);before=values.clone();baseline=layer(values).detach()
    handle,calls=audit.install_neutralization(layer,(1,))
    try:
        changed=layer(values).detach();assert torch.equal(changed,baseline-10*values[...,1:2]);assert len(calls)==1
        assert torch.equal(values,before)
    finally:handle.remove()
    assert torch.equal(layer(values).detach(),baseline)
    handle,calls=audit.install_neutralization(layer,(0,1,2,3))
    try:assert torch.count_nonzero(layer(values))==0 and len(calls)==1
    finally:handle.remove()


def test_observer_records_five_routes_without_changing_projection():
    layer=torch.nn.Linear(4,2);x=torch.tensor([[[1.,0.,-2.,3.]]]);baseline=layer(x).detach().clone()
    handle,calls,observed=audit.install_input_observer(layer,('rsi','macd','hist','ret'))
    try:
        for _ in audit.TIMEFRAMES:assert torch.equal(layer(x).detach(),baseline)
    finally:handle.remove()
    assert tuple(calls)==audit.TIMEFRAMES
    totals={};audit.accumulate_observations(totals,observed,('rsi','macd','hist','ret'))
    assert all(totals[tf]['rsi']['nonzero']==1 and totals[tf]['macd']['nonzero']==0 and totals[tf]['hist']['absolute_sum']==2 for tf in audit.TIMEFRAMES)
    assert torch.equal(layer(x).detach(),baseline)


def test_neutralization_rejects_bad_scope_and_batch():
    layer=torch.nn.Linear(4,1)
    for columns in ((),(0,0),(4,),(-1,),('0',)):
        with pytest.raises(RuntimeError,match='NEUTRALIZATION_BINDING'):audit.install_neutralization(layer,columns)
    handle,_=audit.install_neutralization(layer,(0,))
    try:
        with pytest.raises(RuntimeError,match='NEUTRALIZATION_INPUT'):layer(torch.ones(1,4))
    finally:handle.remove()


def test_influence_uses_only_mature_observed_nonties_for_all_60():
    n=4;sign=np.ones((n,60),np.int8);sign[1,:]=-1;sign[2,:]=0
    observed=np.ones((n,60),np.bool_);observed[3,:]=False;terminal=np.zeros((n,60),np.int64)
    baseline=np.ones((n,60),np.float32);changed=baseline.copy();changed[0,:]=-1;changed[3,:]=-9
    result=audit.influence_summary(sign,observed,terminal,baseline,changed,1)
    assert len(result)==60 and all(x['eligible']==2 and x['direction_decisions_changed']==1 for x in result)
    assert all(x['original_correct_ablated_wrong']==1 and x['ablated_correct_original_wrong']==0 for x in result)


def test_fixed_sampling_and_multiple_comparison_contract():
    assert audit.BATCH==gc.BATCH==16 and audit.STRIDE==gc.BATCH_STRIDE==10 and audit.HEADLINES==gc.HEADLINE==(5,60)
    assert audit.REPLICATES==gc.REPLICATES==10000 and audit.ALPHA==pytest.approx(.05/(5*2*2*3))
    chosen=gc.selected_batches(17288);assert len(chosen)==109 and sum(len(x) for x in chosen)==1736
    assert chosen[-1].tolist()==list(range(17280,17288))


def test_cuda_validator_dispatch_and_rejected_duplicate_flags(monkeypatch,tmp_path):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls=[];monkeypatch.setattr(audit,'run',lambda **kwargs:calls.append(kwargs) or {'status':'SYNTHETIC'})
    args=['--direction-m5-momentum-intervention-plan',str(tmp_path/'plan.json'),
        '--direction-m5-momentum-intervention-plan-sha256','a'*64,'--device','cuda','--out-dir',str(tmp_path)]
    assert validator.main(args)==0 and len(calls)==1
    with pytest.raises(SystemExit):validator.main(args+['--device','cuda'])
    with pytest.raises(SystemExit):validator.main(args[:4]+['--device','cpu','--out-dir',str(tmp_path)])


def _upstream_arrays():
    n=300
    axes={'parent_rows':np.arange(n,dtype=np.int64),
        'bar_start_ns':np.arange(n,dtype=np.int64)*100,
        'decision_ns':np.arange(n,dtype=np.int64)*100+5}
    targets={'terminal_time_ns':np.ones((n,60),np.int64)*10,
        'observed':np.ones((n,60),np.bool_),'sign':np.ones((n,60),np.int8)}
    terminal={**axes,**targets,'logit_up':np.arange(n*60,dtype=np.float32).reshape(n,60)}
    rows=np.random.default_rng(1).permutation(n)[:256]
    initial={k:v[rows].copy() for k,v in terminal.items() if k!='logit_up'}
    return initial,terminal,axes,targets,n


def test_upstream_joins_fixed_shuffled_axes_to_terminal_logits():
    initial,terminal,axes,targets,n=_upstream_arrays()
    selected=audit.upstream_selected_predictions(initial,terminal,axes,targets,n)
    assert np.array_equal(selected['parent_rows'],initial['parent_rows'])
    assert np.array_equal(selected['logit_up'],terminal['logit_up'][initial['parent_rows']])


@pytest.mark.parametrize('corruption',['duplicate','target','decision','population'])
def test_upstream_rejects_population_or_target_mismatch(corruption):
    initial,terminal,axes,targets,n=_upstream_arrays()
    if corruption=='duplicate':initial['parent_rows'][0]=initial['parent_rows'][1]
    elif corruption=='target':initial['sign'][0,4]=-1
    elif corruption=='decision':initial['decision_ns'][0]+=1
    else:terminal['parent_rows'][0]=1
    with pytest.raises(RuntimeError,match='UPSTREAM_'):
        audit.upstream_selected_predictions(initial,terminal,axes,targets,n)


def test_upstream_source_transition_keeps_original_and_rejects_extra_change():
    import copy,hashlib
    names=sorted(audit.UPSTREAM_CHANGED_OWNERS)+[f'synthetic_{i}.py' for i in range(60)]
    historical={name:name.encode() for name in names}
    old={name:{'path':str(audit.REPO/name),'sha256':hashlib.sha256(historical[name]).hexdigest()} for name in names}
    current=copy.deepcopy(old)
    for name in audit.UPSTREAM_CHANGED_OWNERS:current[name]['sha256']='a'*64
    sources={**current,**{name:{'path':str(audit.REPO/name),'sha256':'b'*64}
        for name in audit.UPSTREAM_EXTRA_SOURCES if name not in current}}
    proof={'schema_version':'gx1_family_upstream_source_transition_v1','status':'SOURCE_ONLY_REVIEWED',
        'original_sources':old,'current_sources':current,'changed_paths':sorted(audit.UPSTREAM_CHANGED_OWNERS),
        'numerical_model_sources_unchanged':True,'test_data_used':False}
    launch={'source_owners':old}
    result=audit.upstream_source_transition(launch,sources,proof,historical)
    assert launch['source_owners']==old and result['source_owners']==current
    bad=copy.deepcopy(sources);bad['synthetic_0.py']['sha256']='c'*64
    proofbad=copy.deepcopy(proof);proofbad['current_sources']['synthetic_0.py']['sha256']='c'*64
    with pytest.raises(RuntimeError,match='UPSTREAM_SOURCE_DELTA'):
        audit.upstream_source_transition(launch,bad,proofbad,historical)
    broken=dict(historical);broken[names[0]]=b'wrong historical bytes'
    with pytest.raises(RuntimeError,match='UPSTREAM_HISTORICAL_SOURCE'):
        audit.upstream_source_transition(launch,sources,proof,broken)


def test_upstream_profiler_counts_reverse_and_blocks_wrong_phase_or_excess():
    from types import SimpleNamespace
    def reverse():pass
    code=reverse.__code__;frame=SimpleNamespace(f_code=code)
    phase={'name':'REVERSE'};counts={}
    profile=audit.upstream_profile_budget({code:('reverse','REVERSE',2)},phase,counts)
    for _ in range(2):profile(frame,'call',None);profile(frame,'return',None)
    assert counts=={'reverse_calls':2,'reverse_returns':2}
    with pytest.raises(RuntimeError,match='UPSTREAM_CALL_BUDGET'):profile(frame,'call',None)
    phase['name']='FORWARD'
    with pytest.raises(RuntimeError,match='UPSTREAM_CALL_PHASE'):profile(frame,'call',None)


@pytest.mark.parametrize('path,name',[
    ('/site-packages/torch/optim/adamw.py','__init__'),
    ('/site-packages/torch/optim/adamw.py','step'),
    ('/site-packages/torch/autograd/__init__.py','backward'),
    ('/site-packages/torch/serialization.py','save'),
])
def test_upstream_profiler_forbids_updates_saves_and_optimizer(path,name):
    from types import SimpleNamespace
    scope={};exec(compile('def '+name+'(): pass',path,'exec'),scope)
    frame=SimpleNamespace(f_code=scope[name].__code__)
    profile=audit.upstream_profile_budget({}, {'name':'REVERSE'}, {})
    with pytest.raises(RuntimeError,match='UPSTREAM_FORBIDDEN_OPERATION'):profile(frame,'call',None)


def test_upstream_exact_dispatch_and_power_scope(monkeypatch,tmp_path):
    from pathlib import Path
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls=[];monkeypatch.setattr(audit,'run_upstream',lambda **kwargs:calls.append(kwargs) or {'status':'SYNTHETIC'})
    args=['--direction-m5-momentum-intervention-plan',str(audit.UPSTREAM_PLAN),
        '--direction-m5-momentum-intervention-plan-sha256','a'*64,'--device','cuda','--out-dir',str(audit.UPSTREAM_OUTPUT)]
    assert validator.main(args)==0 and len(calls)==1
    source=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    assert str(audit.UPSTREAM_PLAN)+')' in source and 'm5_output='+str(audit.UPSTREAM_OUTPUT) in source
