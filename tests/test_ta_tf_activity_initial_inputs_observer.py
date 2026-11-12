"""Focused synthetic observer/cap gates; no native model, checkpoint or market data."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from gx1.audit import observe_market_state_ta_tf_activity_initial_inputs_v1 as worker


class Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_model = torch.nn.Module()
        self.state_model._dedicated_tf_activity = True
        self.state_model._dedicated_m5_side = True
        self.state_model._dedicated_non_m5_side = False
        self.state_model.tf_activity_heads = torch.nn.ModuleDict(
            {tf: torch.nn.Linear(128,1) for tf in worker.TIMEFRAMES})
        self.future_tf_activity_heads = torch.nn.ModuleDict(
            {tf: torch.nn.Linear(128,1) for tf in worker.TIMEFRAMES})

    def forward(self, value):
        state = torch.stack([self.state_model.tf_activity_heads[tf](value[:,None,i,:])
                             for i,tf in enumerate(worker.TIMEFRAMES)], dim=2)
        future = torch.stack([self.future_tf_activity_heads[tf](value[:,i,:]+.125)
                              for i,tf in enumerate(worker.TIMEFRAMES)], dim=1)
        return state, future


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(93)
        return Heads()


def values():
    return torch.arange(16*5*128,dtype=torch.float32).reshape(16,5,128)/1024


def hooks(m):
    return {n:(tuple(x._forward_pre_hooks.items()),tuple(x._forward_hooks.items()))
            for n,x in m.named_modules()}


def test_all_ten_inputs_are_owned_ordered_and_passive():
    m=model().eval();x=values()
    first=next(m.parameters());first.requires_grad_(False);first.grad=torch.ones_like(first)
    existing=m.state_model.tf_activity_heads['M5'].register_forward_pre_hook(lambda *args:None)
    state={k:v.clone() for k,v in m.state_dict().items()}
    flags=[(id(p),p.requires_grad,None if p.grad is None else p.grad.clone()) for p in m.parameters()]
    modes=[v.training for v in m.modules()];before_hooks=hooks(m);rng=torch.get_rng_state().clone()
    with torch.no_grad():
        expected=m(x)
        counts={}
        with worker.capture_activity_inputs(m,counts=counts,torch=torch,batches=2) as captured:
            actual=m(x);m(x+1)
    assert counts=={'head_hook_registrations':10,'head_input_observations':20}
    assert len(captured)==2 and all(tuple(p)==worker.HEAD_KEYS for p in captured)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    for batch,delta in enumerate((0,1)):
        for i,tf in enumerate(worker.TIMEFRAMES):
            for task,offset in (('state',0),('future',.125)):
                got=captured[batch][task+'_'+tf]
                assert got.shape==(16,128) and got.dtype==np.float32 and got.flags.owndata
                assert got.tobytes()==(x[:,i,:]+delta+offset).numpy().tobytes()
    copied=deepcopy(captured);x.add_(10)
    assert all(np.array_equal(captured[i][k],v) for i,p in enumerate(copied) for k,v in p.items())
    captured[0]['state_M5'].fill(999)
    assert torch.equal(torch.get_rng_state(),rng) and hooks(m)==before_hooks
    assert [v.training for v in m.modules()]==modes
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    for p,(identity,flag,grad) in zip(m.parameters(),flags):
        assert id(p)==identity and p.requires_grad==flag
        assert p.grad is None if grad is None else torch.equal(p.grad,grad)
    existing.remove()


@pytest.mark.parametrize('fault,error,observations,partial',[
    ('shape','HEAD_INPUT',1,()),('dtype','HEAD_INPUT',1,()),('nonfinite','HEAD_NONFINITE',1,()),
    ('order','HEAD_CALL_ORDER',1,()),('duplicate','HEAD_CALL_ORDER',2,('state_M5',)),
    ('missing','INCOMPLETE_HEAD_CAPTURE',1,('state_M5',)),
    ('future_shape','HEAD_INPUT:future_M5',6,tuple('state_'+tf for tf in worker.TIMEFRAMES)),
])
def test_hook_failures_count_actual_calls_and_remove_all_hooks(fault,error,observations,partial):
    m=model();x=values();before=hooks(m);counts={};captured=[]
    with pytest.raises(RuntimeError,match=error):
        with worker.capture_activity_inputs(m,counts=counts,torch=torch,batches=1) as captured:
            h=m.state_model.tf_activity_heads;v=x[:,None,0,:]
            if fault=='shape':h['M5'](v[:,0])
            elif fault=='dtype':h['M5'](v.double())
            elif fault=='nonfinite':v=v.clone();v[0,0,0]=float('nan');h['M5'](v)
            elif fault=='order':h['M15'](v)
            elif fault=='duplicate':h['M5'](v);h['M5'](v)
            elif fault=='missing':h['M5'](v)
            else:
                for i,tf in enumerate(worker.TIMEFRAMES):h[tf](x[:,None,i,:])
                m.future_tf_activity_heads['M5'](v)
    assert hooks(m)==before and counts['head_input_observations']==observations
    assert tuple(captured[0])==partial if captured else partial==()


def test_swallowed_hook_error_cannot_complete_and_registration_failure_cleans_up(monkeypatch):
    m=model();before=hooks(m);counts={}
    with pytest.raises(RuntimeError,match='INCOMPLETE_HEAD_CAPTURE'):
        with worker.capture_activity_inputs(m,counts=counts,torch=torch,batches=1):
            try:m.state_model.tf_activity_heads['M15'](values()[:,None,0,:])
            except RuntimeError:pass
    assert hooks(m)==before
    def fail(*args,**kwargs):raise RuntimeError('registration failure')
    monkeypatch.setattr(m.state_model.tf_activity_heads['H1'],'register_forward_pre_hook',fail)
    counts={}
    with pytest.raises(RuntimeError,match='registration failure'):
        with worker.capture_activity_inputs(m,counts=counts,torch=torch,batches=1):pass
    assert hooks(m)==before and counts=={'head_input_observations':0,'head_hook_registrations':2}


@pytest.mark.parametrize('fault',('flag','keys','shape','alias'))
def test_malformed_head_layout_is_rejected_before_hook_registration(fault):
    m=model()
    if fault=='flag':m.state_model._dedicated_non_m5_side=True
    elif fault=='keys':del m.future_tf_activity_heads['D1']
    elif fault=='shape':m.future_tf_activity_heads['D1']=torch.nn.Linear(127,1)
    else:m.future_tf_activity_heads['D1'].weight=m.state_model.tf_activity_heads['D1'].weight
    before=hooks(m);counts={}
    with pytest.raises(RuntimeError,match='HEAD_'):
        with worker.capture_activity_inputs(m,counts=counts,torch=torch,batches=1):pass
    assert hooks(m)==before and counts=={}


def arrays(rows=16,representations=False):
    layout=worker.previous._layout(rows,np=np,prefix=True);layout.pop('phase')
    data={k:np.zeros(shape,dtype) for k,(shape,dtype) in layout.items()}
    data['parents']=np.arange(rows,dtype=np.int64)
    data['issues']=np.arange(rows,dtype=np.int64)*300_000_000_000
    for task in ('state','future'):
        data[task+'_query'][:]=1;data[task+'_allowed'][:]=4
        data[task+'_eligible'][:]=True;data[task+'_mature'][:]=True
        if representations:data[task+'_representation']=np.zeros((rows,5,128),np.float32)
    return data


@pytest.mark.parametrize('field',worker.BASE_KEYS)
def test_every_original_field_and_loss_requires_bitexact_parity(field):
    expected=arrays();actual=deepcopy(expected)
    if actual[field].dtype==np.bool_:actual[field].flat[0]=False
    elif actual[field].dtype==np.float32:actual[field].flat[0]=-0.0
    else:actual[field].flat[0]+=1
    with pytest.raises(RuntimeError,match='INITIAL_ARRAY_PARITY'):
        worker.previous._require_parity(actual,expected,1.,1.,np=np)
    with pytest.raises(RuntimeError,match='INITIAL_LOSS_PARITY'):
        worker.previous._require_parity(expected,expected,1.+1e-12,1.,np=np)


def test_partial_capture_keeps_ordered_inputs_and_enforces_own_byte_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    data=arrays(representations=True);counts={'published_npz_bytes':0}
    ref=worker._save([data],counts,partial=True,np=np)
    assert ref['path']==str(tmp_path/'TRAIN_FUSION_PARTIAL.npz')
    with np.load(ref['path'],allow_pickle=False) as z:
        assert set(z.files)==set(data) and all(z[k].tobytes()==v.tobytes() for k,v in data.items())
    assert counts['published_npz_bytes']==Path(ref['path']).stat().st_size
    with pytest.raises(RuntimeError,match='FULL_CAPTURE'):worker._save([data],counts,partial=False,np=np)
    monkeypatch.setitem(worker.BOUNDS,'max_cache_bytes',1)
    with pytest.raises(RuntimeError,match='OUTPUT_BYTES'):
        worker._write_arrays('FAILED_BATCH_INPUTS.npz',{'state_M5':data['state_representation'][:,0]},counts,np=np)
    assert not (tmp_path/'FAILED_BATCH_INPUTS.npz').exists()


def plan():
    return dict(schema_version=worker.SCHEMA,stage=worker.STAGE,run_scope=worker.SCOPE,
        source_repo=str(worker.REPO),source_base_commit='a'*40,worker={},cap_runner={},source_owners={},inputs={},
        feature_view_sha256='a'*64,numerical_recipe_sha256='b'*64,parent_order_sha256='c'*64,**deepcopy(worker.BOUNDS))


def test_exact_plan_has_no_extra_models_updates_or_scope(monkeypatch,tmp_path):
    p=plan();assert worker._require_plan(p,worker.ROOT/'PLAN.json',worker.ROOT)==p
    for key,value in (('max_backbone_encodings',257),('max_checkpoint_loads',2),('max_model_constructions',True),
                      ('max_optimizer_updates',1),('max_head_input_observations',2561),('max_rng_restores',1),
                      ('sampled_rows',4095),('batch_indices',[False]+list(range(1,256))),('training_enabled',True)):
        bad=deepcopy(p);bad[key]=value
        with pytest.raises(RuntimeError,match='EXACT_PLAN'):worker._require_plan(bad,worker.ROOT/'PLAN.json',worker.ROOT)
    with pytest.raises(RuntimeError,match='EXACT_PLAN'):worker._require_plan(p,worker.ACTUAL_ROOT/'PLAN.json',worker.ACTUAL_ROOT)
    monkeypatch.setattr(worker,'ROOT',tmp_path);(tmp_path/'CLAIM.json').write_text('{}')
    monkeypatch.setattr(worker.cache,'_read',lambda ref:p)
    monkeypatch.setattr(worker,'build_plan',lambda *a:pytest.fail('consumed plan reached metadata construction'))
    with pytest.raises(RuntimeError,match='CONSUMED'):worker._inspect(tmp_path/'PLAN.json','a'*64,tmp_path,execute=False)


def test_guard_failure_precedes_claim_checkpoint_and_constructor(monkeypatch):
    monkeypatch.setattr(worker,'_inspect',lambda *a,**k:{})
    def fail():raise RuntimeError('guard required')
    monkeypatch.setattr(worker.previous,'_runtime_cap',fail)
    monkeypatch.setattr(worker.cache,'_publish',lambda *a,**k:pytest.fail('claim before guard'))
    with pytest.raises(RuntimeError,match='guard required'):worker.run(worker.ROOT/'PLAN.json','a'*64,worker.ROOT)
    with pytest.raises(RuntimeError,match='CUDA_ONLY'):worker.run(worker.ROOT/'PLAN.json','a'*64,worker.ROOT,device='cpu')


def test_only_cap_owner_may_differ_from_actual_completed_source(monkeypatch):
    names=[worker.cache.CAP_RUNNER]+[f'fixed{i}.py' for i in range(91)]
    old={n:dict(path=str(worker.REPO/n),sha256=hashlib.sha256(n.encode()).hexdigest()) for n in names}
    owners=deepcopy(old);owners[worker.cache.CAP_RUNNER]['sha256']='b'*64
    monkeypatch.setattr(worker.campaign,'require_binding',lambda *a,**k:None)
    monkeypatch.setattr(worker.subprocess,'check_output',lambda args,**kw:args[2].partition(':')[2].encode())
    worker._require_sources(owners,{'source_owners':old})
    owners['fixed0.py']['sha256']='c'*64
    with pytest.raises(RuntimeError,match='ACTUAL_NUMERICAL_SOURCE_CHANGED:fixed0.py'):
        worker._require_sources(owners,{'source_owners':old})


def test_metadata_preflight_and_exact_cli_do_not_run(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(worker,'_inspect',lambda *a,**kw:calls.append(kw) or {'plan_ref':{},'view':{'contract_sha256':'a'*64}})
    result=worker.preflight(tmp_path/'PLAN.json','a'*64,tmp_path)
    assert calls==[{'execute':False}] and all(result[k]==0 for k in
        ('model_constructions','array_deserializations','checkpoint_loads','optimizer_updates','claims'))
    monkeypatch.setattr(worker,'preflight',lambda **kw:calls.append(('preflight',kw)) or {})
    monkeypatch.setattr(worker,'run',lambda **kw:calls.append(('run',kw)) or {})
    flag='--market-state-ta-tf-activity-initial-inputs-plan'
    argv=[flag,str(tmp_path/'PLAN.json'),flag+'-sha256','a'*64,'--device','cuda','--out-dir',str(tmp_path)]
    worker.main(argv+['--preflight']);worker.main(argv)
    assert calls[-2][0]=='preflight' and calls[-1][0]=='run' and calls[-1][1]['plan_sha']=='a'*64
    for extra in (['--device','cuda'],['--preflight','--preflight'],['--train']):
        with pytest.raises(SystemExit):worker.main(argv+extra)
    argv[-1]='relative'
    with pytest.raises(SystemExit):worker.main(argv)


def cap_gate(tmp_path,changes=None,extra=()):
    source=(worker.REPO/'scripts/gx1_capped_run.sh').read_text()
    name='matches_bound_market_state_ta_tf_activity_initial_inputs'
    helper=name+'() {\n'+source.split(name+'() {\n',1)[1].split('\n}\n',1)[0]+'\n}\n'
    lines=source.splitlines();var='TA_TF_ACTIVITY_INITIAL_INPUTS_PRODUCER_MODULE'
    start=next(i for i,line in enumerate(lines) if line.strip()=='if [[ "${3:-}" == "$'+var+'" ]]; then')
    end=next(i for i in range(start,len(lines)) if lines[i].strip()=='TRAINER_MODEL_MAX_WALL_SECONDS=1200')+2
    block='\n'.join(lines[start:end]);assert block.count(str(worker.ROOT))==1
    root=tmp_path/'capture';root.mkdir(exist_ok=True);p=root/'PLAN.json';p.write_text('{"synthetic":true}')
    module='gx1.audit.observe_market_state_ta_tf_activity_initial_inputs_v1'
    flag='--market-state-ta-tf-activity-initial-inputs-plan'
    argv=['/canonical/python','-m',module,flag,str(p),flag+'-sha256',hashlib.sha256(p.read_bytes()).hexdigest(),
          '--device','cuda','--out-dir',str(root)]
    for index,value in (changes or {}).items():argv[index]=value
    script=helper+'''invoke() {
NUMERICAL_THREAD_COUNT=1
TRAINER_GPU_MAX_POWER_LIMIT_W=160
TRAINER_MAX_WALL_SECONDS=3600
TRAINER_MODEL_MAX_WALL_SECONDS=3600
TRAINER_GPU_MAX_POWER_DRAW_W=170
TRAINER_GPU_MAX_CORE_TEMP_C=65
TRAINER_GPU_MAX_MEMORY_TEMP_C=80
'''+block.replace(str(worker.ROOT),str(root))+'''
printf '%s %s %s %s %s %s %s\\n' "$TRAINER_GPU_MAX_POWER_LIMIT_W" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_TEMP_C" "$NUMERICAL_THREAD_COUNT"
}
invoke "$@"
'''
    return subprocess.run(['bash','-c',script,'inputs-gate',*argv,*extra],capture_output=True,text=True,timeout=10,
        env={**os.environ,'CANONICAL_TRAINER_PYTHON':'/canonical/python',var:module})


def test_actual_cap_selector_keeps_exact_hardware_and_thread_limits(tmp_path):
    result=cap_gate(tmp_path)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='250 1200 1200 170 65 80 8'


@pytest.mark.parametrize('changes,extra',[
    ({4:str(worker.ACTUAL_ROOT/'PLAN.json')},()),({6:'0'*64},()),({8:'cpu'},()),
    ({},('--preflight',)),({},('--device','cuda')),
])
def test_actual_cap_selector_rejects_wrong_scope_hash_and_extra_flags(tmp_path,changes,extra):
    result=cap_gate(tmp_path,changes,extra)
    assert result.returncode==75 and result.stdout=='' and 'not exactly bound' in result.stderr
