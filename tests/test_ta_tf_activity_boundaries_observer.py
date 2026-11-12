"""Synthetic paired boundary, endpoint-copy and exact cap gates only."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import observe_market_state_ta_tf_activity_boundaries_v1 as worker
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
from tests.test_ta_tf_activity_initial_inputs_observer import Heads, arrays, hooks


class Boundaries(Heads):
    def __init__(self):
        super().__init__()
        self.state_model.backbone=torch.nn.Module()
        self.state_model.backbone.entry_q_joint_norm=torch.nn.LayerNorm(512)
        self.state_model.backbone.entry_q_joint_in=torch.nn.Linear(512,128)
        self.state_model.query_projection=torch.nn.Sequential(torch.nn.Linear(134,128),torch.nn.GELU())
        self.state_model.register_buffer('query_tf_coordinates',torch.eye(5))

    def forward(self,source,*,different_future_hidden=False):
        s=self.state_model
        hidden=torch.nn.functional.gelu(s.backbone.entry_q_joint_in(s.backbone.entry_q_joint_norm(source)))
        tf=s.query_tf_coordinates[None].expand(16,5,5)
        fraction=hidden.new_tensor([1.,.5,.25,.125,.0625])[None,:,None].expand(16,5,1)
        common=hidden[:,None,:].expand(16,5,128)
        sr=s.query_projection(torch.cat((common,tf,fraction),dim=-1)[:,None])
        state=torch.stack([s.tf_activity_heads[t](sr[:,:,i,:]) for i,t in enumerate(worker.initial.TIMEFRAMES)],dim=2)
        fr=s.query_projection(torch.cat((common+(1 if different_future_hidden else 0),tf,torch.ones_like(fraction)),dim=-1))
        future=torch.stack([self.future_tf_activity_heads[t](fr[:,i,:]) for i,t in enumerate(worker.initial.TIMEFRAMES)],dim=1)
        return state,future,hidden


def test_reused_joint_and_activity_hooks_are_passive_together_and_cleanup_on_error():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7);m=Boundaries().eval()
    x=torch.arange(16*512,dtype=torch.float32).reshape(16,512)/2048
    before_hooks=hooks(m);before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.no_grad():
        expected=m(x);counts={}
        with worker.initial.capture_activity_inputs(m,counts=counts,torch=torch,batches=1) as heads:
            with worker.capture_joint_boundaries(m,counts=counts,torch=torch) as joint:actual=m(x)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert joint['joint_source'].tobytes()==x.numpy().tobytes()
    assert joint['joint_hidden'].tobytes()==expected[2].numpy().tobytes()
    assert len(heads)==1 and tuple(heads[0])==worker.initial.HEAD_KEYS
    assert counts==dict(head_hook_registrations=10,head_input_observations=10,
        joint_source_captures=1,joint_hidden_captures=1,query_input_observations=2)
    assert hooks(m)==before_hooks and torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    with pytest.raises(RuntimeError,match='HIDDEN_STATE_FUTURE'):
        with worker.initial.capture_activity_inputs(m,counts={},torch=torch,batches=1) as partial:
            with worker.capture_joint_boundaries(m,counts={},torch=torch) as joint:
                m(x,different_future_hidden=True)
    assert hooks(m)==before_hooks and tuple(partial[0])==tuple('state_'+tf for tf in worker.initial.TIMEFRAMES)
    assert set(joint)=={'joint_source','joint_hidden'}


def tiny_endpoint_fixture(monkeypatch):
    m=torch.nn.Module();m.state_model=torch.nn.Module();m.state_model.feature_view_identity={'contract_sha256':'a'*64}
    m.state_model.head=torch.nn.Linear(2,1);m.register_buffer('fixed',torch.tensor([1.]));m.eval()
    def parameters(model):
        pairs=list(model.named_parameters());return pairs,pairs,[]
    fake_state=SimpleNamespace(TA_STATE_SCHEMA='synthetic',_parameters=parameters,_numerical=lambda recipe,model:recipe)
    recipe=dict(expected_view_sha256='a'*64,numerical_recipe={'profile':'same'})
    order=torch.arange(32,dtype=torch.int64)
    def saved(final=False):
        values=deepcopy(m.state_dict());buffers=deepcopy(dict(m.named_buffers()))
        if final:
            for key in values:values[key].add_(.125)
            for value in buffers.values():value.add_(.125)
        return dict(schema_version='synthetic',source_commit=worker.initial.ACTUAL_SOURCE,
            model_state=values,model_state_sha256=model_sha(values),buffer_state=buffers,buffer_state_sha256=old._tree_sha(buffers),
            feature_view_sha256='a'*64,cursor=4096 if final else 0,optimizer_updates=256 if final else 0,skipped_batches=0,
            parent_order=order.clone(),binding=dict(numerical_recipe=recipe['numerical_recipe'],
                feature_view_identity=m.state_model.feature_view_identity,active_parameter_names=[n for n,_ in m.named_parameters()],
                inactive_parameter_names=[]),module_training={n:True for n,_ in m.named_modules()})
    first,last=saved(),saved(True)
    monkeypatch.setattr(worker.short,'MODEL_SHA256',first['model_state_sha256'])
    monkeypatch.setattr(worker,'FINAL_MODEL',last['model_state_sha256'])
    kwargs=dict(recipe=recipe,order=order,torch=torch,old=old,state=fake_state,model_sha=model_sha)
    return m,first,last,kwargs


def signature(m):
    return dict(model=model_sha(m.state_dict()),buffers=old._tree_sha(dict(m.named_buffers())),
        rng=old._tree_sha(torch.get_rng_state()),modes={n:v.training for n,v in m.named_modules()},
        flags={n:p.requires_grad for n,p in m.named_parameters()},parameters={n:id(p) for n,p in m.named_parameters()},
        buffer_ids={n:id(p) for n,p in m.named_buffers()},modules={n:id(v) for n,v in m.named_modules()},
        hooks=hooks(m),view=deepcopy(m.state_model.feature_view_identity))


def test_strict_endpoint_copy_preserves_rng_modes_flags_storage_identities_and_no_grad(monkeypatch):
    m,first,last,kwargs=tiny_endpoint_fixture(monkeypatch)
    worker._load_endpoint_weights(m,first,phase='INITIAL_TRAIN',**kwargs)
    before=signature(m)
    identity=worker._load_endpoint_weights(m,last,phase='FINAL_TRAIN',**kwargs)
    receipt=worker._require_transition(before,signature(m),identity)
    assert receipt['non_weight_state_unchanged'] is True and receipt['optimizer_or_rng_restored'] is False
    assert all(torch.equal(v,last['model_state'][k]) for k,v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters()) and not m.training
    for key in ('rng','modes','flags','parameters','buffer_ids','modules','hooks','view'):
        bad=signature(m);bad[key]='changed'
        with pytest.raises(RuntimeError,match='WEIGHTS_ONLY_TRANSITION'):worker._require_transition(before,bad,identity)


@pytest.mark.parametrize('fault',('progress','source','model_digest','active_names'))
def test_endpoint_metadata_rejects_before_weight_mutation(monkeypatch,fault):
    m,first,last,kwargs=tiny_endpoint_fixture(monkeypatch);before=signature(m)
    if fault=='progress':last['cursor']=0
    elif fault=='source':last['source_commit']='0'*40
    elif fault=='model_digest':last['model_state_sha256']='0'*64
    else:last['binding']['active_parameter_names']=[]
    with pytest.raises(RuntimeError,match='ENDPOINT_'):worker._load_endpoint_weights(m,last,phase='FINAL_TRAIN',**kwargs)
    assert signature(m)==before


def test_initial_requires_all_tf_head_vector_parity_and_final_keeps_raw_loss_exact():
    baseline=arrays();piece=dict(deepcopy(baseline),state_representation=np.zeros((16,5,128),np.float32),
        future_representation=np.zeros((16,5,128),np.float32))
    heads=deepcopy(piece)
    worker._require_parity(piece,baseline,1.,1.,initial_heads=heads,np=np)
    piece['future_representation'][0,4,127]=np.float32(1e-8)
    with pytest.raises(RuntimeError,match='INITIAL_ALL_TF_HEAD_INPUT_PARITY'):
        worker._require_parity(piece,baseline,1.,1.,initial_heads=heads,np=np)
    worker._require_parity(piece,baseline,1.,1.,initial_heads=None,np=np)
    piece['state_raw'][0,4,0]=-0.0
    with pytest.raises(RuntimeError,match='INITIAL_ARRAY_PARITY'):
        worker._require_parity(piece,baseline,1.,1.,initial_heads=None,np=np)
    with pytest.raises(RuntimeError,match='INITIAL_LOSS_PARITY'):
        worker._require_parity(baseline,baseline,1.00000001,1.,initial_heads=None,np=np)


def paired_arrays():
    one=arrays(4096,representations=True)
    result={k:np.concatenate((v,v)) for k,v in one.items()}
    result.update(joint_source=np.zeros((8192,512),np.float32),joint_hidden=np.zeros((8192,128),np.float32),
                  phase=np.r_[np.zeros(4096,np.uint8),np.ones(4096,np.uint8)])
    return result


def test_exact_paired_layout_fits64mib_and_partial_keeps_actual_phase(tmp_path,monkeypatch):
    data=paired_arrays();worker._require_arrays(data,np=np,complete=True)
    assert sum(a.nbytes for a in data.values())==64_610_304 < worker.BOUNDS['max_cache_bytes']
    data['phase'][4096]=0
    with pytest.raises(RuntimeError,match='PHASE_ORDER'):worker._require_arrays(data,np=np,complete=True)
    data['phase'][4096]=1
    piece={k:v[:16].copy() for k,v in data.items()};del data
    monkeypatch.setattr(worker,'ROOT',tmp_path);counts={'published_npz_bytes':0}
    ref=worker._save([piece],counts,partial=True,np=np)
    with np.load(ref['path'],allow_pickle=False) as z:assert all(z[k].tobytes()==v.tobytes() for k,v in piece.items())
    with pytest.raises(RuntimeError,match='ARRAY_ROWS'):worker._save([piece],counts,partial=False,np=np)
    monkeypatch.setitem(worker.BOUNDS,'max_cache_bytes',counts['published_npz_bytes'])
    with pytest.raises(RuntimeError,match='OUTPUT_BYTES'):
        worker._write_arrays('FAILED_BATCH_BOUNDARIES.npz',{'joint_hidden':piece['joint_hidden']},counts,np=np)


def plan():
    return dict(schema_version=worker.SCHEMA,stage=worker.STAGE,run_scope=worker.SCOPE,source_repo=str(worker.REPO),
        source_base_commit='a'*40,worker={},cap_runner={},source_owners={},helpers=deepcopy(worker.HELPERS),inputs={},
        feature_view_sha256='a'*64,numerical_recipe_sha256='b'*64,parent_order_sha256='c'*64,**deepcopy(worker.BOUNDS))


def test_exact_new_scope_counts_consumed_and_guard_first(tmp_path,monkeypatch):
    p=plan();worker._require_plan(p,worker.ROOT/'PLAN.json',worker.ROOT)
    for key,value in (('max_backbone_encodings',513),('max_checkpoint_loads',3),('max_model_constructions',True),
        ('max_head_input_observations',5121),('max_joint_source_captures',513),('max_rng_restores',1),('sampled_rows',4096)):
        bad=deepcopy(p);bad[key]=value
        with pytest.raises(RuntimeError,match='EXACT_PLAN'):worker._require_plan(bad,worker.ROOT/'PLAN.json',worker.ROOT)
    with pytest.raises(RuntimeError,match='EXACT_PLAN'):worker._require_plan(p,worker.initial.ROOT/'PLAN.json',worker.initial.ROOT)
    with pytest.raises(RuntimeError,match='COUNTS'):worker._require_counts({'model_constructions':1})
    monkeypatch.setattr(worker,'ROOT',tmp_path);(tmp_path/'CLAIM.json').write_text('{}')
    monkeypatch.setattr(worker.cache,'_read',lambda ref:p)
    with pytest.raises(RuntimeError,match='CONSUMED'):worker._inspect(tmp_path/'PLAN.json','a'*64,tmp_path,execute=False)
    monkeypatch.setattr(worker,'_inspect',lambda *a,**k:{})
    monkeypatch.setattr(worker.previous,'_runtime_cap',lambda:(_ for _ in ()).throw(RuntimeError('guard first')))
    monkeypatch.setattr(worker.cache,'_publish',lambda *a,**k:pytest.fail('claim before guard'))
    with pytest.raises(RuntimeError,match='guard first'):worker.run(tmp_path/'PLAN.json','a'*64,tmp_path)


def test_exact_cli_preflight_never_runs_and_rejects_duplicate_flags(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(worker,'preflight',lambda **kw:calls.append(('preflight',kw)) or {})
    monkeypatch.setattr(worker,'run',lambda **kw:calls.append(('run',kw)) or {})
    flag='--market-state-ta-tf-activity-boundaries-plan'
    argv=[flag,str(tmp_path/'PLAN.json'),flag+'-sha256','a'*64,'--device','cuda','--out-dir',str(tmp_path)]
    worker.main(argv+['--preflight']);assert calls[0][0]=='preflight' and calls[0][1]['plan_sha']=='a'*64
    for extra in (['--device','cuda'],['--preflight','--preflight'],['--train']):
        with pytest.raises(SystemExit):worker.main(argv+extra)
    argv[-1]='relative'
    with pytest.raises(SystemExit):worker.main(argv)


def cap_gate(tmp_path,changes=None,extra=()):
    source=(worker.REPO/'scripts/gx1_capped_run.sh').read_text();name='matches_bound_market_state_ta_tf_activity_boundaries'
    helper=name+'() {\n'+source.split(name+'() {\n',1)[1].split('\n}\n',1)[0]+'\n}\n'
    lines=source.splitlines();var='TA_TF_ACTIVITY_BOUNDARIES_PRODUCER_MODULE'
    start=next(i for i,line in enumerate(lines) if line.strip()=='if [[ "${3:-}" == "$'+var+'" ]]; then')
    end=next(i for i in range(start,len(lines)) if lines[i].strip()=='TRAINER_MODEL_MAX_WALL_SECONDS=1200')+2
    block='\n'.join(lines[start:end]);assert block.count(str(worker.ROOT))==1
    root=tmp_path/'capture';root.mkdir(exist_ok=True);p=root/'PLAN.json';p.write_text('{"synthetic":true}')
    module='gx1.audit.observe_market_state_ta_tf_activity_boundaries_v1';flag='--market-state-ta-tf-activity-boundaries-plan'
    argv=['/canonical/python','-m',module,flag,str(p),flag+'-sha256',hashlib.sha256(p.read_bytes()).hexdigest(),
          '--device','cuda','--out-dir',str(root)]
    for i,value in (changes or {}).items():argv[i]=value
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
    return subprocess.run(['bash','-c',script,'boundary-gate',*argv,*extra],capture_output=True,text=True,timeout=10,
        env={**os.environ,'CANONICAL_TRAINER_PYTHON':'/canonical/python',var:module})


def test_actual_exact_cap_selector_preserves_hardware_stops(tmp_path):
    result=cap_gate(tmp_path);assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='250 1200 1200 170 65 80 8'


@pytest.mark.parametrize('changes,extra',[
    ({4:str(worker.initial.ROOT/'PLAN.json')},()),({6:'0'*64},()),({8:'cpu'},()),({},('--preflight',)),
])
def test_actual_cap_selector_rejects_consumed_old_root_bad_hash_and_extra_flags(tmp_path,changes,extra):
    result=cap_gate(tmp_path,changes,extra)
    assert result.returncode==75 and result.stdout=='' and 'not exactly bound' in result.stderr
