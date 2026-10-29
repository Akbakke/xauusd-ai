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
