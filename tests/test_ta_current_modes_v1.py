"""Synthetic current-mode controls only; no GX1 model, checkpoints or market inputs."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from gx1.audit import observe_market_state_ta_current_modes_v1 as observer


class TinyModes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(.05)
        self.attention = torch.nn.MultiheadAttention(4,2,dropout=.13,batch_first=True)
        self.norm = torch.nn.LayerNorm(4)
        self.readout = torch.nn.Linear(4,1)

    def forward(self,x):
        x = self.drop(x)
        return self.readout(self.norm(self.attention(x,x,x,need_weights=False)[0]))


@pytest.mark.parametrize('fail',[False,True])
def test_temporary_mode_restores_mixed_modes_both_dropout_probabilities_and_state(fail):
    model=TinyModes().eval(); model.readout.train(); model.readout.bias.requires_grad_(False)
    modes={n:m.training for n,m in model.named_modules()}
    probabilities=observer._dropout_probabilities(model,torch=torch)
    values={n:p.detach().clone() for n,p in model.named_parameters()}
    identities=[(n,id(p),p.requires_grad) for n,p in model.named_parameters()]
    rng=torch.get_rng_state().clone()
    def observe():
        with observer.temporary_mode(model,training=True,zero_dropout=True,torch=torch):
            assert all(m.training for m in model.modules())
            assert model.drop.p == model.attention.dropout == 0.
            with torch.no_grad(): assert torch.isfinite(model(torch.ones(2,3,4))).all()
            if fail: raise ValueError('synthetic forward failure')
    if fail:
        with pytest.raises(ValueError,match='synthetic forward failure'): observe()
    else: observe()
    assert {n:m.training for n,m in model.named_modules()} == modes
    assert observer._dropout_probabilities(model,torch=torch) == probabilities
    assert probabilities['attention']['probability'] == .13
    assert identities == [(n,id(p),p.requires_grad) for n,p in model.named_parameters()]
    assert all(torch.equal(values[n],p) and p.grad is None for n,p in model.named_parameters())
    assert torch.equal(torch.get_rng_state(),rng)


def test_paired_fresh_rng_stream_is_checked_per_route_and_batch_without_historical_claim():
    original=torch.get_rng_state().clone()
    def digest(): return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    def phase():
        out=[]
        for slot,batch in zip(observer.PANEL_SLOTS,observer.BATCHES):
            record=dict(panel_slot=slot,original_batch_index=batch,before=digest(),routes=[])
            for index,route in enumerate(observer.ROUTES):
                before=digest()
                if index >= 3: torch.rand(5)
                record['routes'].append(dict(route=route,before=before,after=digest()))
            record['after']=digest(); out.append(record)
        return out
    try:
        torch.manual_seed(observer.OBSERVER_SEED); start=torch.get_rng_state().clone()
        initial=phase(); first_end=torch.get_rng_state().clone()
        torch.set_rng_state(start); final=phase()
        observer.require_paired_rng(initial,final)
        assert torch.equal(first_end,torch.get_rng_state()) and not torch.equal(start,first_end)
        bad=copy.deepcopy(final); bad[3]['routes'][4]['after']='f'*64
        with pytest.raises(RuntimeError,match='PAIRED_RNG_BATCH'): observer.require_paired_rng(initial,bad)
        with pytest.raises(RuntimeError,match='RNG_BATCH_COUNT'): observer.require_paired_rng(initial,final[:-1])
        # Equal malformed copies must still fail the independent route-chain check.
        bad=copy.deepcopy(initial); bad[1]['routes'][1]['before']='e'*64
        with pytest.raises(RuntimeError,match='RNG_ROUTE_CHAIN'): observer.require_paired_rng(bad,copy.deepcopy(bad))
    finally: torch.set_rng_state(original)


def piece():
    out=dict(parents=np.arange(16,dtype=np.int64),issues=np.arange(16,dtype=np.int64),
        state_query=np.ones((16,5),dtype=np.int64),future_query=np.ones((16,5),dtype=np.int64))
    for task in ('state','future'):
        out[task+'_raw']=np.zeros((16,5,2),dtype=np.float32)
        out[task+'_allowed']=np.full((16,5),4,dtype=np.uint8)
        for key in ('eligible','mature'): out[task+'_'+key]=np.ones((16,5),dtype=np.bool_)
    assert set(out)==set(observer.previous.BASE_KEYS)
    return out


def test_mode_effects_remain_measurements_while_native_reference_is_required():
    baseline=piece(); shifted={k:v.copy() for k,v in baseline.items()}
    shifted['future_raw'][:,1:,1]=np.float32(.3)
    native=observer.describe_route('NATIVE_NO_GRAD',baseline,baseline,1.,1.,np=np)
    bad_native=observer.describe_route('NATIVE_NO_GRAD',shifted,baseline,1.,1.,np=np)
    eval_result=observer.describe_route('EVAL_DYNAMIC_GRAD',shifted,baseline,1.1,1.,np=np)
    train=observer.describe_route('TRAIN_DROPOUT_0',shifted,baseline,1.1,1.,np=np,eval_piece=baseline,eval_loss=1.)
    zero=observer.describe_route('TRAIN_ZERO_DROPOUT_GRAD',shifted,baseline,1.1,1.,np=np,eval_piece=baseline,eval_loss=1.)
    assert native['required_reference_qualified'] is True and bad_native['required_reference_qualified'] is False
    for record in (eval_result,train,zero):
        assert record['required_reference_qualified'] is None
        assert record['retained_comparison']['within_existing_tolerance'] is False
    assert train['dynamic_eval_comparison']['within_existing_tolerance'] is False
    z=float(np.float32(.3)); expected=1/(1+np.exp(-z))
    assert train['outputs']['future']['mean_raw_by_tf_channel'][1][1] == z
    assert train['outputs']['future']['mean_probability_by_tf_channel'][1][1] == pytest.approx(expected,abs=1e-15)
    json.dumps([native,bad_native,eval_result,train,zero],allow_nan=False)
    shifted['state_allowed'][0,0]=1
    with pytest.raises(RuntimeError,match='AXES_TARGETS'):
        observer.describe_route('EVAL_DYNAMIC_GRAD',shifted,baseline,1.,1.,np=np)


def test_probability_summary_is_mean_sigmoid_and_loss_is_original_batch_mean():
    baseline=piece(); actual={k:v.copy() for k,v in baseline.items()}
    actual['future_raw'][:8,2,1]=np.float32(2.)
    actual['future_raw'][8:,2,1]=np.float32(-1.)
    a=observer.describe_route('EVAL_DYNAMIC_GRAD',actual,baseline,1.25,1.,np=np)
    b=observer.describe_route('EVAL_DYNAMIC_GRAD',baseline,baseline,.75,1.,np=np)
    summary=observer._summary([a,b],np=np)['EVAL_DYNAMIC_GRAD']
    expected=((1/(1+np.exp(-2.))+1/(1+np.exp(1.)))/2+.5)/2
    assert summary['rows']==32 and summary['batches']==2 and summary['mean_original_loss']==1.
    assert summary['outputs']['future']['mean_probability_by_tf_channel'][2][1] == pytest.approx(expected,abs=1e-15)
    assert summary['outputs']['future']['mean_probability_by_tf_channel'][2][1] != pytest.approx(1/(1+np.exp(-.25)),abs=1e-5)


def test_partial_phase_capture_is_retained_and_byte_cap_rejects_before_write(tmp_path,monkeypatch):
    monkeypatch.setattr(observer,'ROOT',tmp_path)
    p=dict(piece(),route_index=np.zeros(16,dtype=np.int8),panel_slot=np.zeros(16,dtype=np.int16),
        original_batch_index=np.zeros(16,dtype=np.int32),batch_loss=np.full(16,1.25,dtype=np.float64))
    counts=dict(phase_capture_writes=0,published_capture_bytes=0)
    saved=observer._save_phase('INITIAL',[p],counts,np=np)
    assert saved['rows']==16 and saved['route_records']==1 and saved['complete_phase'] is False
    assert counts['phase_capture_writes']==1 and 0 < counts['published_capture_bytes'] < observer.BOUNDS['max_capture_bytes']
    with np.load(saved['ref']['path'],allow_pickle=False) as archive:
        assert set(archive.files)==set(p)
        assert all(archive[k].dtype==v.dtype and archive[k].tobytes()==v.tobytes() for k,v in p.items())
    monkeypatch.setattr(observer,'ROOT',tmp_path/'rejected')
    monkeypatch.setitem(observer.BOUNDS,'max_capture_bytes',1)
    rejected=dict(phase_capture_writes=0,published_capture_bytes=0)
    with pytest.raises(RuntimeError,match='CAPTURE_BOUND'): observer._save_phase('FINAL',[p],rejected,np=np)
    assert rejected==dict(phase_capture_writes=0,published_capture_bytes=0)
    assert not (tmp_path/'rejected'/'FINAL_MODE_ROUTES.npz').exists()


def test_consumed_plan_and_wrong_path_fail_before_origin_or_model(tmp_path,monkeypatch):
    monkeypatch.setattr(observer,'ROOT',tmp_path)
    plan=tmp_path/'PLAN.json'; plan.write_text('{}\n')
    sha=hashlib.sha256(plan.read_bytes()).hexdigest()
    monkeypatch.setattr(observer,'build_plan',lambda *a:pytest.fail('consumed origin must not execute'))
    (tmp_path/'CLAIM.json').write_text('{}\n')
    with pytest.raises(RuntimeError,match='CONSUMED'): observer._inspect(plan,sha,tmp_path,execute=False)
    with pytest.raises(RuntimeError,match='EXACT_PATHS'): observer._inspect(plan,sha,tmp_path/'other',execute=False)


def test_runtime_cap_failure_precedes_claim_and_payload_load(monkeypatch):
    monkeypatch.setattr(observer,'_inspect',lambda *a,**kw:dict())
    def reject(): raise RuntimeError('synthetic unguarded runtime')
    monkeypatch.setattr(observer.previous,'_runtime_cap',reject)
    monkeypatch.setattr(observer.cache,'_publish',lambda *a,**kw:pytest.fail('no claim before cap'))
    with pytest.raises(RuntimeError,match='synthetic unguarded runtime'):
        observer.run(observer.ROOT/'PLAN.json','0'*64,observer.ROOT)


def test_cli_preflight_is_last_and_duplicate_or_other_device_is_rejected(monkeypatch,capsys):
    called=[]
    monkeypatch.setattr(observer,'preflight',lambda **kw:called.append(kw) or {'status':'synthetic metadata only'})
    args=['--market-state-ta-current-modes-plan',str(observer.ROOT/'PLAN.json'),
        '--market-state-ta-current-modes-plan-sha256','0'*64,'--device','cuda','--out-dir',str(observer.ROOT)]
    assert observer.main(args+['--preflight'])==0 and len(called)==1
    assert json.loads(capsys.readouterr().out)['status']=='synthetic metadata only'
    for bad in (args+['--preflight','--preflight'], ['--preflight']+args, args+['--device','cuda'],
                [*args[:5],'cpu',*args[6:]]):
        with pytest.raises(SystemExit): observer.main(bad)
    assert len(called)==1
