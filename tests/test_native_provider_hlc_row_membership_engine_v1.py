"""Exercise real engine prefixes before any seed/model/optimizer construction."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import gx1.audit as audit_package

REPO=Path('/home/andre2/src/GX1_CURRENT')
ENGINE=REPO/'gx1/audit/native_provider_hlc_learning_v1.py'
OLD='gx1.audit.native_provider_hlc_block_order_v1'
NEW='gx1.audit.native_provider_hlc_row_membership_fit_v1'


def engine_prefix(phase,namespace):
    name='physical_all_tf'if phase=='main'else'physical_all_tf_resume'
    tree=ast.parse(ENGINE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    stop=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
              and ast.unparse(n.value.func)=='core.future_cuda_seed_and_runtime')
    node.body=node.body[:stop]+ast.parse("return {'counts':counts,'batches':batches,'canonical':canonical_schedule,'root':"+('order_root'if phase=='main'else'output')+'}').body
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(module,str(ENGINE),'exec'),namespace)
    return namespace[name]


def fixture_engine(monkeypatch,phase,module_name):
    source=np.arange(100,106,dtype=np.int64);issues=np.arange(600,606,dtype=np.int64)
    batches=[np.arange(i,i+2,dtype=np.int64)for i in(0,2,4)]
    digest=lambda a:tuple(np.asarray(a).tolist())
    canonical={'logical_batches':3,'batch_source_rows_sha256':[digest(source[b])for b in batches],
               'batch_issue_ns_sha256':[digest(issues[b])for b in batches]}
    schedule=canonical if module_name is None else {'canonical_schedule':canonical,'logical_batches':3}
    limits={'feature_cache_payload_loads':1,'target_archive_loads':1}
    if module_name==NEW:limits['order_array_loads']=1
    original={'schedule':schedule,'recipe':{'train_batch_rows':2},'metadata':{}}
    plan={**original,'limits':limits,'schema_version':'synthetic_prefix',
          'metadata':{'original':original,'receipt':{},'trajectory':{}},'starting_cursor':0}
    binding={'path':'/synthetic/plan','sha256':'a'*64};scope=None if module_name is None else {'module':module_name,'mode':phase,'plan':binding}
    calls=[];context={'source_rows':source,'truth':{'issue_ns':issues}}
    def need(ok,why):
        if not ok:raise ValueError(why)
    def load(meta):calls.append(('load',meta));return context
    for module in(OLD,NEW):
        obj=ModuleType(module)
        def require(value,p,b,mode,expected=module):
            calls.append(('require',expected,mode));need(value=={'module':expected,'mode':phase,'plan':binding}and mode==phase,'scope')
            return Path('/synthetic')/expected.rsplit('.',1)[-1]/mode
        def resolve(ctx,sched,*extra,expected=module):
            calls.append(('resolve',expected,len(extra)));need(ctx is context and sched is schedule,'context')
            if expected==NEW:
                need(len(extra)==1 and extra[0]['order_array_loads']==0,'counted_order');extra[0]['order_array_loads']+=1
            else:need(not extra,'old_resolver_signature')
            return list(reversed(batches))
        obj.require_engine_scope=require;obj.resolve_batches=resolve
        monkeypatch.setitem(sys.modules,module,obj);monkeypatch.setattr(audit_package,module.rsplit('.',1)[-1],obj,raising=False)
    namespace={'time':SimpleNamespace(monotonic=lambda:0.0),'np':np,'need':need,'ALL_TF_ROOT':Path('/synthetic/default/main'),
               'all_tf_recovery_root':lambda sha:Path('/synthetic/default/resume'),
               'adapter':SimpleNamespace(load_context=load),
               'train':SimpleNamespace(position=lambda s,c:{'optimizer_updates':0}),
               'core':SimpleNamespace(all_score_batches=lambda rows,size:batches,digest=digest)}
    return engine_prefix(phase,namespace),plan,binding,scope,calls,batches,canonical


@pytest.mark.parametrize('phase',['main','resume'])
@pytest.mark.parametrize('module_name',[None,OLD,NEW])
def test_real_engine_prefix_preserves_old_defaults_and_counts_new_order_once(monkeypatch,phase,module_name):
    fn,plan,binding,scope,calls,batches,canonical=fixture_engine(monkeypatch,phase,module_name)
    got=fn(plan,binding,{}, {},order_scope=scope)
    assert got['canonical']is canonical
    expected=batches if module_name is None else list(reversed(batches))
    assert all(np.array_equal(a,b)for a,b in zip(got['batches'],expected))
    assert got['counts']['feature_cache_payload_loads']==got['counts']['target_archive_loads']==1
    if module_name is None:
        assert [c[0]for c in calls]==['load']and got['root']==Path('/synthetic/default')/phase
    else:
        assert [c[0]for c in calls]==['require','load','resolve']
        assert got['root']==Path('/synthetic')/module_name.rsplit('.',1)[-1]/phase
        assert calls[-1]==('resolve',module_name,int(module_name==NEW))
    if module_name==NEW:assert got['counts']['order_array_loads']==1
    else:assert 'order_array_loads'not in got['counts']


@pytest.mark.parametrize('phase',['main','resume'])
@pytest.mark.parametrize('bad',['mode','plan','module','extra'])
def test_bad_scope_stops_before_context_decode(monkeypatch,phase,bad):
    fn,plan,binding,scope,calls,_,_=fixture_engine(monkeypatch,phase,NEW)
    if bad=='mode':scope['mode']='resume'if phase=='main'else'main'
    elif bad=='plan':scope['plan']={'path':'/synthetic/other','sha256':'b'*64}
    elif bad=='module':scope['module']='gx1.audit.unknown'
    else:scope['extra']=True
    with pytest.raises(ValueError,match='scope'):fn(plan,binding,{}, {},order_scope=scope)
    assert [c[0]for c in calls]==['require']


@pytest.mark.parametrize('phase',['main','resume'])
def test_canonical_clock_mismatch_rejects_before_new_order_load(monkeypatch,phase):
    fn,plan,binding,scope,calls,_,canonical=fixture_engine(monkeypatch,phase,NEW)
    canonical['batch_issue_ns_sha256'][0]=('wrong',)
    with pytest.raises(ValueError,match='CLOCK_BINDING'):fn(plan,binding,{}, {},order_scope=scope)
    assert [c[0]for c in calls]==['require','load']
