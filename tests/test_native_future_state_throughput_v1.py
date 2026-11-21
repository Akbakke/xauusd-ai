"""Synthetic metadata, arithmetic, tiny CPU parity and consumed-failure tests only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from gx1.audit import measure_native_future_state_throughput_v1 as o


def binding(name='plan.json'):
    return {'path':'/synthetic/'+name,'sha256':'a'*64}


def test_frozen_schedule_counts_are_actual_forward_and_backward_geometry():
    parity=sum(384//b for b in (64,96))
    train=2+sum(384//b for b in o.TRAIN_ORDER)
    evaluate=2+sum((384+b-1)//b for b in o.EVAL_ORDER)
    assert (parity,train,evaluate)==(10,22,18)
    assert o.LIMITS['model_forwards']==o.LIMITS['backbone_forwards']==50==parity+train+evaluate
    assert o.LIMITS['backward_calls']==32==parity+train and o.LIMITS['gradient_clip_calls']==22
    assert o.LIMITS['measurement_writes']==11 and o.LIMITS['allocator_boundary_calls']==14
    assert all(o.LIMITS[k]==0 for k in ('optimizer_steps','normalization_fits','prior_fits','autograd_calls','CONTROL_rows_scored','TEST_rows'))
    assert o.LIMITS['checkpoint_loads']==o.LIMITS['model_state_restores']==1


def test_support_weighted_numerator_is_not_an_average_of_microbatch_means():
    x=torch.tensor(2.,requires_grad=True)
    first={'loss_sums':{'activity':torch.stack([x,x*2]),'side':torch.stack([x*3])}}
    second={'loss_sums':{'activity':torch.stack([x*4]),'side':torch.stack([x*5])}}
    total=o.weighted_numerator(first,9)+o.weighted_numerator(second,9)
    total.backward()
    assert float(total)==pytest.approx(30/9) and float(x.grad)==pytest.approx(15/9)
    assert float(total)!=pytest.approx(((12/2)+(18/7))/2)
    with pytest.raises(ValueError,match='GLOBAL_KNOWN'):o.weighted_numerator(first,0)


@pytest.mark.parametrize('bad',['/sealed/TEST.py','../escape.py','gx1/unexpected.npz'])
def test_exact_source_inventory_rejects_before_any_hash(monkeypatch,bad):
    owners=dict.fromkeys(o.SOURCE_PATHS,'a'*64);owners[bad]='b'*64
    monkeypatch.setattr(o.c,'sha',lambda *a:pytest.fail('unvalidated source was hashed'))
    with pytest.raises(ValueError,match='SOURCE_SET'):o.source_inventory(owners)


def test_parent_inventory_is_checked_before_numeric_hash_and_guard_bridge(monkeypatch):
    monkeypatch.setattr(o.c,'future_cuda_source_inventory',lambda v:(_ for _ in()).throw(ValueError('old inventory')))
    monkeypatch.setattr(o.c,'sha',lambda *a:pytest.fail('hash before old inventory'))
    with pytest.raises(ValueError,match='old inventory'):o.source_bridge({'/sealed/weights.pt':'a'*64})


def test_unexpected_input_roles_cannot_read_any_parent_or_payload(monkeypatch):
    monkeypatch.setattr(o,'read',lambda *a:pytest.fail('parent opened before exact roles'))
    with pytest.raises(ValueError,match='INPUT_ROLES'):o.metadata({'checkpoint':binding('weights.pt')})


@pytest.mark.parametrize('suffix',['.pt','.npz','.parquet'])
def test_json_reader_never_hashes_payload(monkeypatch,suffix):
    monkeypatch.setattr(o.c,'read',lambda *a:pytest.fail('payload hashed or read'))
    with pytest.raises(ValueError,match='JSON_REFERENCE'):o.read(binding('forbidden'+suffix))


def test_first384_is_input_chronology_only():
    rows=np.arange(1000,5000,dtype=np.int64)
    actual=o.selected_rows({'rows':rows,'truth':{'activity_known':np.zeros((4000,5),bool)}})
    np.testing.assert_array_equal(actual,rows[:384])
    with pytest.raises(ValueError,match='FIRST384'):o.selected_rows({'rows':rows[:383]})


def test_selection_uses_only_two_completed_fixed_passes_and_throughput():
    records=[]
    for b,seconds in ((64,2.),(96,1.)):
        records.extend([{'mode':'train','batch_rows':b,'rows':384,'wall_seconds':seconds,
            'completed_under_limits':True,'fake_accuracy':0. if b==96 else 1.}for _ in range(2)])
    records.append({'mode':'train','batch_rows':256,'rows':384,'wall_seconds':.001,'completed_under_limits':False})
    assert o.select_fastest(records,'train')['batch_rows']==96
    assert o.select_fastest(records,'train')['observed_rows_per_second']==384
    with pytest.raises(ValueError,match='NO_COMPLETED'):o.select_fastest(records[:1],'train')


def test_gradient_parity_rejects_vacuous_body_and_unused_parameter_drift():
    zero={'backbone.weight':np.zeros((2,2),np.float32)}
    with pytest.raises(ValueError,match='NONVACUOUS'):o.compare_gradients(zero,zero)
    real={'backbone.weight':np.ones((2,2),np.float32),'readout.unused':None}
    assert o.compare_gradients(real,real)['parameters_compared']==1
    with pytest.raises(ValueError,match='UNUSED_PARAMETER'):o.compare_gradients(real,{**real,'readout.unused':np.zeros(1,np.float32)})


def _inspect_fixture(monkeypatch,tmp_path):
    pr=binding();policy_ref=binding('policy.json')
    plan={'source_owners':dict.fromkeys(o.SOURCE_PATHS,'a'*64),'inputs':{},'source_base_commit':'a'*40,
        'metadata':{'runtime':{'synthetic':'runtime'}}}
    monkeypatch.setattr(o,'ROOT',tmp_path/'fresh')
    gate=o.gate_expected(pr,admitted=False)
    policy={k:False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')}
    policy[o.PARENT]={'goal_status':'ACTIVE','large_training_qualified':False,o.GATE:gate}
    monkeypatch.setattr(o,'PLAN',Path(pr['path']));monkeypatch.setattr(o,'read',lambda r:plan if r==pr else policy)
    monkeypatch.setattr(o,'build_plan',lambda **kw:plan);monkeypatch.setattr(o.c,'ref',lambda path:policy_ref)
    monkeypatch.setattr(o.subprocess,'check_output',lambda args,**kw:'work/gx1-current\n'if 'branch'in args else 'a'*40+'\n')
    monkeypatch.setattr(o.c,'collect_future_cuda_runtime_metadata',lambda:{'synthetic':'runtime'})
    monkeypatch.setattr(o.c,'future_cuda_runtime_matches',lambda a,b:None)
    return plan,policy,pr


def test_preflight_is_closed_exact_source_and_one_use_before_payload(monkeypatch,tmp_path):
    plan,policy,pr=_inspect_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('preflight opened checkpoint'))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('preflight opened arrays'))
    assert o.inspect('a'*64,admitted=False)[0]is plan
    policy['training_enabled']=True
    with pytest.raises(ValueError,match='TRAIN_FLAGS'):o.inspect('a'*64,admitted=False)
    policy['training_enabled']=False;policy[o.PARENT][o.GATE]['remaining_invocations']=False
    with pytest.raises(ValueError,match='ONEUSE'):o.inspect('a'*64,admitted=False)
    plan['source_owners']['/sealed/weights.pt']='b'*64
    with pytest.raises(ValueError,match='SOURCE_SET'):o.inspect('a'*64,admitted=False)


def test_packet_has_exact_bound256_and_never_passes_truth_forward():
    calls=[]
    class Provider:
        source_row=np.arange(400,dtype=np.int64)
        def gather(self,rows,*,max_rows):
            calls.append((len(rows),max_rows));return {'seq_x':np.ones((len(rows),5,2),np.float32)}
    context={'provider':Provider(),'rows':np.arange(400,dtype=np.int64),'truth':{'issue_ns':np.arange(400)}}
    p=o.packet(context,np.arange(256,dtype=np.int64));assert calls==[(256,256)]
    assert set(p['features'])=={'seq_x'}and set(p['truth'])=={'issue_ns'}
    with pytest.raises(ValueError,match='MAX256'):o.packet(context,np.arange(257,dtype=np.int64))


def _tiny_context(monkeypatch):
    from tests.test_native_future_state_cpu_mechanics_v1 import packet
    plan,prior,packets,old,opt=packet();target=plan['metadata']['target_contract'];rows=np.arange(384,dtype=np.int64)
    tick=300_000_000_000;issues=(rows+1)*tick;dur=np.array([target['tf_duration_ns'][tf]for tf in target['timeframes']],np.int64)
    origin=np.array([target['tf_origin_ns'][tf]for tf in target['timeframes']],np.int64)
    starts=((issues[:,None]-origin+dur-1)//dur)*dur+origin;known=np.ones((384,5),bool)
    active=np.broadcast_to((rows%3!=0)[:,None],known.shape).copy();side=active&np.broadcast_to((rows%5!=0)[:,None],known.shape)
    truth={'issue_ns':issues,'target_start_ns':starts,'validity_ns':starts+dur,'target_present':known.copy(),
        'mature':known.copy(),'nominal_complete':known.copy(),'state_window_complete':known.copy(),
        'partial_range_active':np.zeros_like(known),'activity_known':known.copy(),'activity':active,
        'side_known':side,'side_up':side&np.broadcast_to((rows%2==0)[:,None],known.shape)}
    values=np.linspace(-1,1,384*5*2,dtype=np.float32).reshape(384,5,2)
    class Provider:
        source_row=rows
        def gather(self,selected,*,max_rows):return {'seq_x':values[selected]}
    class Body(nn.Module):
        def __init__(self):super().__init__();self.linear=nn.Linear(2,2);self.dropout=nn.Dropout(.05)
        def forward(self,**features):return {'entry_q_timeframe_hidden':self.dropout(self.linear(features['seq_x']))}
    class Model(nn.Module):
        def __init__(self):super().__init__();self.backbone=Body();self.readout=old.readout
        def forward(self,**features):return self.readout(self.backbone(**features)['entry_q_timeframe_hidden'])
    model=Model()
    with torch.no_grad():
        for head in [*model.readout.activity_heads.values(),*model.readout.side_heads.values()]:head.weight.fill_(.03)
    monkeypatch.setattr(o.c,'restore_training_rng',o.c.future_cpu_restore_rng)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:1024)
    def forward(m,features,*,equal_tf=False):
        return m(**features),None,{'batch_eval_active':equal_tf,'scope_enabled':equal_tf,
            'grad_enabled':torch.is_grad_enabled(),'model_training':m.training}
    monkeypatch.setattr(o.c,'future_cuda_forward',forward)
    return model,{'provider':Provider(),'rows':rows,'truth':truth,'target_contract':target},o.c.future_cpu_rng()


def test_actual_tiny_eval_grad_cross_batch_parity_is_nonvacuous_and_read_only(monkeypatch):
    model,context,rng=_tiny_context(monkeypatch);before=o.c.state_sha(model.state_dict());counts=dict.fromkeys(o.LIMITS,0)
    reference=None;support=None
    for batch in (64,96):
        record,arrays,grads=o._cohort(model,context,context['rows'],batch,'parity',counts,rng)
        if reference is None:reference=(arrays,grads);support=record['support']
        assert o.compare_arrays(reference[0],arrays)
        assert o.compare_gradients(reference[1],grads)['all_qualified']
        assert record['support']==support and record['global_known_activity_cells']==1920
    assert counts['backward_calls']==10 and counts['gradient_clip_calls']==0
    assert o.c.state_sha(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    assert o.c.state_sha(o.c.future_cpu_rng())==o.c.state_sha(rng)


def test_actual_tiny_train_timing_and_eval_modes_never_step_or_mutate_weights(monkeypatch):
    model,context,rng=_tiny_context(monkeypatch);counts=dict.fromkeys(o.LIMITS,0);before=o.c.state_sha(model.state_dict())
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**kw:pytest.fail('benchmark optimizer update'))
    train,_,_=o._cohort(model,context,context['rows'],96,'train',counts,rng)
    evaluate,arrays,_=o._cohort(model,context,context['rows'],256,'eval',counts,rng)
    assert train['rows']==evaluate['rows']==384 and counts['backward_calls']==counts['gradient_clip_calls']==4
    assert evaluate['activation']['batch_eval_active']and not evaluate['activation']['grad_enabled']
    assert arrays['timeframe_hidden'].shape==(384,5,2)and o.c.state_sha(model.state_dict())==before


def _run_fixture(monkeypatch,tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    root=tmp_path/'out';monkeypatch.setattr(o,'ROOT',root);pr=binding();policy=binding('policy.json')
    live={'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'physical_windows_boot_id':701}
    refs={k:binding(k+'.json')for k in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest')}
    plan={'inputs':{},'metadata':{'function':refs,'donor_checkpoint':binding('cursor3.pt')}}
    proof={'plan':pr,'output_root':str(root),'outer_seconds':1800,'internal_seconds':1770}
    monkeypatch.setattr(guard,'require_native_future_state_throughput_execution',lambda **kw:proof)
    monkeypatch.setattr(o,'inspect',lambda *a,**kw:(plan,pr,policy,live));monkeypatch.setattr(o,'bookend',lambda *a:True)
    monkeypatch.setattr(o.c,'physical_windows_boot_id',lambda:701)
    return root,plan,pr


def test_preclaim_rejection_leaves_no_claim_or_output(monkeypatch,tmp_path):
    root,plan,pr=_run_fixture(monkeypatch,tmp_path)
    def blocked(*a,**kw):raise ValueError('preclaim gate')
    monkeypatch.setattr(o,'inspect',blocked)
    with pytest.raises(ValueError,match='preclaim gate'):o.run('a'*64)
    assert not root.exists()


def test_OOM_failure_consumes_once_preserves_completed_measurement_and_resets_timer(monkeypatch,tmp_path):
    root,plan,pr=_run_fixture(monkeypatch,tmp_path);calls=[]
    def fail(plan,pr,counts,artifacts,progress):
        calls.append(True);progress['stage']='parity_B96';counts['model_forwards']=1
        artifacts.append(o.publish('EVAL_B64_PASS001.json',{'completed':'synthetic'}))
        raise torch.cuda.OutOfMemoryError('synthetic allocation rejection')
    monkeypatch.setattr(o,'physical',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):o.run('a'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text())
    assert calls==[True]and terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0
    assert terminal['relaunch_allowed']is False and terminal['training_authorized']is False
    assert terminal['failed_stage']=='parity_B96'and len(terminal['completed_measurements'])==1
    assert (root/'CLAIM.json').is_file()and (root/'EVAL_B64_PASS001.json').is_file()and not(root/'RESULT.json').exists()
    assert o.signal.getitimer(o.signal.ITIMER_REAL)==(0.,0.)


def test_cli_metadata_preflight_is_fixed_and_never_decodes(monkeypatch,capsys):
    import sys
    monkeypatch.setattr(o,'inspect',lambda *a,**kw:({},binding(),binding('policy.json'),{'observed_utc':'synthetic'}))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('preflight checkpoint'))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('preflight array'))
    monkeypatch.setattr(sys,'argv',['leaf','--plan-sha256','a'*64,'--preflight']);o.main()
    assert json.loads(capsys.readouterr().out)['claim_created']is False
    monkeypatch.setattr(sys,'argv',['leaf','--preflight','--plan-sha256','a'*64])
    with pytest.raises(ValueError,match='EXACT_CLI'):o.main()


def _consumer_fixture(monkeypatch):
    report={'counts':dict(o.LIMITS),'optimizer_steps':0,'whole_TRAIN_INITIAL_replaced':False,
        'production_recipe_selected':False,'selection_criterion':'max_observed_rows_per_second_qualified_under_limits_only',
        'allocator_policy':'allocator_cold_at_measurement_phase_boundary','allocator_boundaries':[{}for _ in range(14)],
        'timing_excludes_allocator_cleanup':True,'unqualified_training_batch_rows':[128],'B128_production_candidate':False}
    table={};plan={'stage':o.STAGE,'limits':dict(o.LIMITS),'metadata':{'function':{'same':'current'},'source_bridge':{},'runtime':{}}}
    for mode,batches in (('train',(64,96)),('eval',(64,256))):
        report[mode]={}
        for batch in batches:
            seconds=2. if batch==64 else 1.;refs=[]
            for i in (1,2):
                r=binding(mode+str(batch)+'_'+str(i)+'.json');refs.append(r)
                table[r['path']]={'mode':mode,'batch_rows':batch,'rows':384,'completed_under_limits':True,'wall_seconds':seconds}
            proof=binding(mode+str(batch)+'_proof.json')
            table[proof['path']]={'batch_rows':batch,'rows':384,'gradients':{'all_qualified':True,'nonzero_backbone_coordinates':1},
                'arrays':{'hidden':{'qualified':True}},'numerical_parity':{'hidden':{'qualified':True}}}
            report[mode][str(batch)]={'qualified':True,'measured_rows':768,'elapsed_seconds':2*seconds,
                'rows_per_second':384/seconds,'peak_reserved_bytes':1024,'measurement_refs':refs,'parity_ref':proof}
        report['selected_'+mode+'_batch_rows']=min(batches,key=lambda b:(-report[mode][str(b)]['rows_per_second'],b))
    result={'stage':o.STAGE,'plan':binding('plan.json'),'report':report};terminal={'stage':o.STAGE}
    table['/synthetic/plan.json']=plan
    monkeypatch.setattr(o.c,'future_cuda_complete',lambda *args:(result,terminal))
    monkeypatch.setattr(o,'read',lambda r:table[r['path']])
    return report,table


def test_JSON_only_consumer_validates_zero_updates_all_configs_roles_and_parity(monkeypatch):
    report,table=_consumer_fixture(monkeypatch)
    result=o.require_throughput_result(binding('result.json'),binding('terminal.json'))
    assert result['train_batch_rows']==96 and result['eval_batch_rows']==256
    report['optimizer_steps']=1
    with pytest.raises(ValueError,match='ZERO_UPDATE'):o.require_throughput_result(binding(),binding())
    report['optimizer_steps']=0;table['/synthetic/train96_proof.json']['gradients']['all_qualified']=False
    with pytest.raises(ValueError,match='TRAIN_PARITY'):o.require_throughput_result(binding(),binding())


def test_fixed003_allocator_policy_is_postclaim_and_outside_measurement_timer():
    import ast
    source=Path(o.__file__).read_text();tree=ast.parse(source)
    assert o.STAGE.endswith('_003')and o.GATE=='throughput_003'
    assert o.PLAN.name=='NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003.json'
    assert o.ROOT.name=='NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003'
    assert len(o.SOURCE_PATHS)==64
    assert 'gx1/audit/native_future_state_training_v1.py'in o.SOURCE_PATHS
    physical=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='physical')
    measure=next(n for n in ast.walk(physical)if isinstance(n,ast.FunctionDef)and n.name=='measure')
    calls=[n for n in ast.walk(measure)if isinstance(n,ast.Call)]
    release=next(n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='release_phase_allocator_cache')
    cohort=next(n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='_cohort')
    diagnostic=next(n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='print')
    assert release.lineno<diagnostic.lineno<cohort.lineno
    assert any(k.arg=='flush'and isinstance(k.value,ast.Constant)and k.value.value is True for k in diagnostic.keywords)
    assert not any(isinstance(n,ast.ImportFrom)and n.module=='gx1.audit.native_future_state_training_v1'for n in tree.body)
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('empty_cache','reset_peak_memory_stats')for n in ast.walk(tree))


def test_003_rejects_old_B128_candidate_scope_before_forward(monkeypatch):
    monkeypatch.setattr(o.c,'restore_training_rng',lambda *a:pytest.fail('old scope reached CUDA/RNG phase'))
    with pytest.raises(ValueError,match='MEASUREMENT_GEOMETRY'):
        o._cohort(None,None,None,128,'train',{},None)
    records=[{'mode':'train','batch_rows':128,'rows':384,'wall_seconds':1.,'completed_under_limits':True}]*2
    with pytest.raises(ValueError,match='CANDIDATE_SCOPE'):o.select_fastest(records,'train')
    monkeypatch.setattr(o.c,'publish',lambda *a:pytest.fail('old scope output was published'))
    for name in ('TRAIN_B128_PASS001.json','BATCH_PARITY_B128.json'):
        with pytest.raises(ValueError,match='OUTPUT_NAME'):o.publish(name,{})


def test_003_consumer_rejects_old_three_training_candidates(monkeypatch):
    report,table=_consumer_fixture(monkeypatch)
    report['train']['128']=copy.deepcopy(report['train']['96'])
    with pytest.raises(ValueError,match='COMPLETE_CANDIDATES'):
        o.require_throughput_result(binding('result.json'),binding('terminal.json'))
