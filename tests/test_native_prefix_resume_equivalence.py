"""Resume equivalence is an operational replay, never additional learning authority."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_native_prefix_recipe import native, _write, _bind
from tests.test_native_prefix_coordinator import PrefixHarness, _prepared, equal_tree, digest, trainer


@pytest.fixture
def replay_scope(tmp_path):
    def rows(name, values):
        path=tmp_path/(name+'.npy')
        np.save(path,np.asarray(values,dtype='int64'))
        return _bind(path)
    order=rows('order',range(4500));parents=rows('parents',range(4500))
    # IDs intentionally overlap: TRAIN and CONTROL use different physical parquets.
    control=rows('control',range(256))
    physical={
        'train':{'parquet':{'path':'physical-train.parquet','sha256':'1'*64},
                 'physical_rows':4500,'first_entry_utc':'2024-01-01T00:00:00+00:00',
                 'last_entry_utc':'2025-05-30T12:55:00+00:00'},
        'val':{'parquet':{'path':'physical-val.parquet','sha256':'2'*64},
               'physical_rows':300,'first_entry_utc':'2025-06-01T22:00:00+00:00',
               'last_entry_utc':'2025-07-01T00:00:00+00:00'}}
    coordinates=_write(tmp_path/'coordinates.json',
                       {'train_source':physical['train']['parquet'],'control_source':physical['val']['parquet']})
    before={'run_id':'ORIGINAL','out_bundle_dir':str(tmp_path/'ORIGINAL'),
            'chronological_prefix':{'synthetic':'fixed'},
            'chronological_learning_measurement':{'synthetic':'initial'},
            'entry_observed_market':{'synthetic':'observed'},
            'source_bindings':{'model':'unchanged'},'source_commit':'a'*40,
            'trainer_cli':{'batch_size':16}}
    prefix={'artifacts':before['chronological_prefix'],'epoch0_parent_order':order,
            'train_parent_rows':parents,'control_parent_rows':control,'physical_sources':physical,
            'native_coordinates':coordinates}
    contract={'out_bundle_dir':before['out_bundle_dir'],'chronological_prefix':prefix}
    cb=_write(tmp_path/'contract.json',contract)
    state=tmp_path/'candidate_training_state_slot_0.pt';state.write_bytes(b'unit-reference')
    origin=tmp_path/'candidate_training_state_slot_1.pt';origin.write_bytes(b'unit-origin')
    rb,ob=_bind(state),_bind(origin)
    pb=_write(tmp_path/'pointer.json',{'global_optimizer_steps':256,'next_batch_offset':256,
        'epoch_index':0,'phase':'train','complete':False,'slot':0,
        'session_contract_sha256':cb['sha256'],'state_sha256':rb['sha256']})
    recipe_binding=_write(tmp_path/'recipe.json',before)
    review={'schema_version':'gx1_native_entry_smoke_completion_review_v1',
        'technical_native_cycle_passed':True,'source_unchanged':True,'optimizer_steps':256,
        'test_data_used':False,'training_expansion_authorized':False,
        'entry_exit_target_independence_verified':True,'target_model_exactly_preserved':True,
        'training_pointer':pb,'training_state':rb,'recipe':recipe_binding,
        'target_model_state_sha256':'3'*64,'model_state_sha256':'4'*64}
    review_binding=_write(tmp_path/'review.json',review)
    audit={'schema_version':'gx1_prefix_resume_equivalence_preflight_v1',
        'origin_contract':cb,'origin_state':ob,'reference_state':rb,'reference_pointer':pb,
        'origin_recipe':recipe_binding,'origin_review':review_binding,'from_optimizer_steps':192,
        'stop_after_optimizer_steps':256,'replayed_optimizer_steps':64,
        'previously_trained_entry_rows':4096,'new_unique_entry_rows':0,'replayed_entry_rows':1024,
        'parent_order':order,'train_rows':parents,'control_rows':control,'entry_order_exact':True,
        'control_overlap':0,'target_model_state_sha256':'3'*64,'original_reference_model_state_sha256':'4'*64,
        'physical_sources':physical,'native_coordinates':coordinates,'torch_cuda_rng_preserved':True,
        'new_optimizer_steps':0,'new_model_forwards':0,'test_data_used':False,
        'replayed_parent_rows_sha256':hashlib.sha256(np.arange(3072,4096,dtype='<i8').tobytes()).hexdigest()}
    recipe=copy.deepcopy(before);recipe.update(run_id='REPLAY',out_bundle_dir=str(tmp_path/'REPLAY'))
    plan={'schema_version':native.PREFIX_RESUME_EQUIVALENCE_SCHEMA,'from_optimizer_steps':192,
        'stop_after_optimizer_steps':256,'replayed_optimizer_steps':64,'maximum_trained_entry_rows':4096,
        'maximum_additional_entry_rows':0,'maximum_replayed_entry_rows':1024,'max_invocations':1,
        'teacher_refresh_allowed':False,'control_forwards':0,'full_epoch_allowed':False,'full_val_allowed':False,
        'test_data_used':False,'automatic_extension_allowed':False,'physical_reboot_required':True,
        'learning_admission':False,'require_bitwise_state_equivalence':True,'run_authority_created':False,
        'out_bundle_dir':recipe['out_bundle_dir'],'origin_review':review_binding,'origin_contract':cb,
        'origin_recipe':recipe_binding,'reference_pointer':pb,'reference_state':rb,'origin_state':ob,
        'comparison_state_fields':['model_state','target_model_state','optimizer_state','weight_ema_state',
            'lr_scheduler_state','rng_state','epoch_order','training_progress'],
        'comparison_cursor_fields':['phase','epoch_index','next_batch_offset','global_optimizer_steps','complete'],
        'only_allowed_state_metadata_difference':['session_contract_sha256']}
    def seal():
        plan['preflight']=_write(tmp_path/'preflight.json',audit)
        recipe['chronological_learning_continuation']=_write(tmp_path/'plan.json',plan)
    seal()
    return recipe,plan,audit,seal


def test_replay_accepts_source_qualified_rows_and_exact_existing_suffix(replay_scope):
    recipe,_,_,_=replay_scope
    scope=native.require_chronological_continuation(recipe)
    assert scope['resume_equivalence'] is True
    assert scope['plan']['from_optimizer_steps']==192
    assert scope['plan']['stop_after_optimizer_steps']==256
    assert scope['plan']['maximum_additional_entry_rows']==0


@pytest.mark.parametrize('fault',['extra_step','extra_row','teacher','control_forward','old_output',
                                  'optimizer_reset','model_source','changed_target','reference_hash',
                                  'row_order','unbound_preflight','no_cuda_rng'])
def test_replay_rejects_expansion_or_changed_provenance(replay_scope,fault):
    r,p,a,seal=replay_scope
    if fault=='extra_step':p['stop_after_optimizer_steps']=257
    if fault=='extra_row':p['maximum_additional_entry_rows']=16
    if fault=='teacher':p['teacher_refresh_allowed']=True
    if fault=='control_forward':p['control_forwards']=1
    if fault=='old_output':r['out_bundle_dir']=r['out_bundle_dir'].replace('REPLAY','ORIGINAL');p['out_bundle_dir']=r['out_bundle_dir']
    if fault=='optimizer_reset':p['from_optimizer_steps']=0
    if fault=='model_source':r['source_bindings']['model']='changed'
    if fault=='changed_target':r['entry_observed_market']={'synthetic':'different'}
    if fault=='reference_hash':p['reference_state']={**p['reference_state'],'sha256':'f'*64}
    if fault=='row_order':a['replayed_parent_rows_sha256']='f'*64
    if fault=='unbound_preflight':a['origin_state']={**a['origin_state'],'sha256':'f'*64}
    if fault=='no_cuda_rng':a['torch_cuda_rng_preserved']=False
    seal()
    with pytest.raises(RuntimeError):native.require_chronological_continuation(r)


def test_serialized_replay_reproduces_last64_steps_and_preserves_original(tmp_path):
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    h=PrefixHarness(tmp_path/'run',_prepared(artifacts))
    original=h.root/'ORIGINAL'
    h.run_prefix(original,256)
    pointer=h.pointer(original);pointer_bytes=pointer.read_bytes();meta=json.loads(pointer_bytes)
    origin_path=pointer.parent/f"candidate_training_state_slot_{1-meta['slot']}.pt"
    reference_path=pointer.parent/f"candidate_training_state_slot_{meta['slot']}.pt"
    expected=h.state(original)
    saved_origin=trainer.torch.load(origin_path,map_location='cpu',weights_only=True)
    assert saved_origin['global_optimizer_steps']==192
    contract=json.loads((pointer.parent/trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text())
    origin_bytes=origin_path.read_bytes();reference_bytes=reference_path.read_bytes()
    continuation={'resume_equivalence':True,'plan_binding':{'path':'synthetic-replay.json','sha256':'a'*64},
        'plan':{'from_optimizer_steps':192,'stop_after_optimizer_steps':256,
                'origin_state':_bind(origin_path),'reference_state':_bind(reference_path)},
        'origin_contract':contract,'origin_pointer':_bind(pointer),
        'origin_review':{'training_state':_bind(reference_path)}}
    h.batches.clear();h.run_prefix(h.root/'REPLAY',256,chronological_continuation=continuation)
    actual=h.state(h.root/'REPLAY')
    for key in expected:
        if key=='session_contract_sha256':assert expected[key]!=actual[key]
        else:equal_tree(expected[key],actual[key])
    assert [row for batch in h.batches for row in batch]==list(range(4500))[::-1][3072:4096]
    assert h.validation_batches==0
    assert pointer.read_bytes()==pointer_bytes and origin_path.read_bytes()==origin_bytes
    assert reference_path.read_bytes()==reference_bytes
    with pytest.raises(RuntimeError,match='PREFIX_FIXED_BUDGET_REQUIRED'):
        h.run_prefix(h.root/'INVALID_EXTENSION',257,chronological_continuation=continuation)


def test_native_dispatch_skips_all_final_measurements_for_equivalence(monkeypatch,tmp_path):
    from tests.test_native_prefix_recipe import runner
    import torch
    recipe={'out_bundle_dir':str(tmp_path/'REPLAY'),'run_id':'REPLAY','dataset_run_id':'DATA',
        'gx1_data_root':str(tmp_path),'chronological_prefix':{'bound':'prefix'},
        'chronological_learning_measurement':{'bound':'initial'},'chronological_learning_continuation':{'bound':'replay'},
        'trainer_cli':{'batch_size':16,'epochs':30,'seed':20260911,'learning_rate':.0001,
                       'weight_decay':.0001,'grad_clip_norm':1.},'val_limits':{'max_wall_seconds':10800}}
    continuation={'resume_equivalence':True,'plan':{'from_optimizer_steps':192,'stop_after_optimizer_steps':256}}
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:None)
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cpu'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *a:(recipe,{},{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    monkeypatch.setattr(native,'require_native_run_scope',lambda *a,**kw:256)
    monkeypatch.setattr(native,'require_chronological_learning_measurement',lambda *a:{})
    monkeypatch.setattr(native,'require_chronological_continuation',lambda *a:continuation)
    monkeypatch.setattr(runner,'_build_bound_full_train_components',lambda **kw:{})
    monkeypatch.setattr(runner,'_restore_prefix_initial_measurement_state',lambda **kw:None)
    def train(**kw):
        assert kw['components']['chronological_continuation']==continuation
        raise runner.trainer._CandidateExecutionPaused({'reason':'optimizer_step_ceiling','global_optimizer_steps':256})
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',train)
    def forbidden(**kw):pytest.fail('Replay must not create a new TRAIN/CONTROL measurement')
    monkeypatch.setattr(runner,'_run_prefix_initial_measurement',forbidden)
    receipt=tmp_path/'pause.json';receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer,'_write_candidate_execution_pause_receipt',lambda *a,**kw:receipt)
    monkeypatch.setattr(runner,'_native_resume_state',lambda **kw:{'global_optimizer_steps':256})
    result=runner.run_guarded_native_candidate_invocation(recipe_path=tmp_path/'recipe.json',recipe_file_sha256='a'*64,
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert result['resume_state']['global_optimizer_steps']==256
    assert 'chronological_final_measurement' not in result['pause']


@pytest.mark.parametrize('mode',['replay','ordinary','changed_baseline','changed_current'])
def test_original_baseline_binding_exception_is_confined_to_exact_replay(tmp_path,monkeypatch,mode):
    from gx1.contracts.entry_model_native_train_launch_v1 import artifact_binding,canonical_json_sha256
    path=tmp_path/'control.py';path.write_text('original')
    old={'wrapper':artifact_binding(path)}
    origin={'source_bindings':old,'source_bindings_sha256':canonical_json_sha256(old)}
    measurement={'native_recipe_source_bindings':old,
                 'native_recipe_source_bindings_sha256':origin['source_bindings_sha256']}
    path.write_text('current control path')
    current={'wrapper':artifact_binding(path)}
    recipe={'chronological_prefix':{'native_coordinates':{'bound':'physical'}},
            'source_bindings':current,'source_bindings_sha256':canonical_json_sha256(current)}
    scope=None if mode=='ordinary' else {'resume_equivalence':True,'origin_recipe':origin}
    monkeypatch.setattr(native,'require_chronological_continuation',lambda r:scope)
    if mode=='changed_baseline':measurement['native_recipe_source_bindings_sha256']='f'*64
    if mode=='changed_current':path.write_text('unbound change')
    if mode=='replay':
        result=native.require_prefix_measurement_source_binding(recipe,measurement=measurement)
        assert result['native_recipe_source_bindings']==current
    else:
        with pytest.raises(RuntimeError,match='MEASUREMENT_SOURCE'):
            native.require_prefix_measurement_source_binding(recipe,measurement=measurement)
