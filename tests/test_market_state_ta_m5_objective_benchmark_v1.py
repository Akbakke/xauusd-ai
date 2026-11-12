"""Synthetic metadata/routing checks; no model, market arrays or execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_m5_objective_benchmark_v1 as scope


def ref(path):
    return dict(path=str(path), sha256='a' * 64)


def seal(value, key):
    value[key] = campaign.canonical_sha256({k: v for k, v in value.items() if k != key})
    return value


def cursor(*, visits=1536, updates=1468, outcome='COMPLETE'):
    return scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH), outcome=outcome,
        optimizer_updates=updates, parent_cursor=16 * visits, parent_order_sha256=None,
        full_state=None, result=ref(scope.ROOT / 'RESULT.json'))


def test_fixed_panels_are_distinct_intact_batches_and_not_old_prefix():
    assert len(scope.TRAIN_BATCH_INDICES) == len(set(scope.TRAIN_BATCH_INDICES)) == 256
    assert len(scope.LATER_BATCH_INDICES) == len(set(scope.LATER_BATCH_INDICES)) == 256
    assert scope.TRAIN_BATCH_INDICES == [i * 1847 // 255 for i in range(256)]
    assert scope.LATER_BATCH_INDICES == [i * 1108 // 255 for i in range(256)]
    assert scope.TRAIN_BATCH_INDICES != scope.anchor.PANEL_BATCH_INDICES
    assert scope.PASSES == (1, 2, 3) and scope.EVALUATION_PASSES == (0, 1, 3)
    assert scope.BUDGETS['max_forwards'] == 4096
    assert scope.BUDGETS['max_candidate_evidence_bytes'] == 1024**3
    assert not authority._is_ta_short_learning_recipe_ref(ref(scope.RECIPE_PATH))
    assert authority._is_ta_short_learning_recipe_ref(ref(scope.anchor.RECIPE_PATH))


@pytest.mark.parametrize('outcome,visits,updates', [('COMPLETE',1536,1536), ('COMPLETE',1536,0),
    ('COMPLETE',1536,1468), ('FAILED',0,0), ('FAILED',19,7)])
def test_cursor_and_progress_count_visits_separately_from_skipped_updates(outcome, visits, updates):
    value = cursor(visits=visits, updates=updates, outcome=outcome)
    assert authority.require_research_cursor(value, expected_recipe=ref(scope.RECIPE_PATH), verify_files=False, batch_size=16) == value
    path = scope.RUNTIME_ROOT / 'research-cursor/RESUME_CURSOR.json'
    invocation = dict(kind=authority.KIND, invocation_sha256='b'*64, direction_research_recipe=ref(scope.RECIPE_PATH),
        batch_size=16, optimizer_step_budget=1536, checkpoint=dict(pointer_path=str(path)))
    progress = authority.build_research_progress(window=dict(recipe=ref(scope.RECIPE_PATH)),
        plan=dict(plan_sha256='c'*64), invocation=invocation, checkpoint_pointer=ref(path),
        optimizer_updates=updates, parent_cursor=16*visits, parent_order_sha256=None, outcome=outcome)
    assert progress['completed_units'] == visits and progress['optimizer_updates'] == updates
    assert progress['total_units'] == 1536
    assert value['canonical_resume_authority'] is False and value['full_state'] is None


@pytest.mark.parametrize('change', [dict(parent_cursor=24560), dict(parent_cursor=24577),
    dict(optimizer_updates=1537), dict(optimizer_updates=True), dict(parent_order_sha256='b'*64),
    dict(full_state=ref(scope.ROOT/'state.pt')), dict(canonical_resume_authority=True),
    dict(test_data_used=True), dict(recipe=ref(scope.ROOT/'OTHER.json'))])
def test_resealed_cursor_cannot_widen_scope(change):
    value = cursor(); value.update(change); seal(value,'cursor_sha256')
    with pytest.raises(RuntimeError):
        scope.require_cursor(value, expected_recipe=ref(scope.RECIPE_PATH), verify_files=False)


def test_exact_recipe_dispatch_does_not_open_old_or_wrong_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, 'require_binding', lambda v, **kw: v)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *a, **kw: dict(run_scope=scope.SCOPE))
    monkeypatch.setattr(scope, 'require_recipe', lambda v, **kw: calls.append(kw) or v)
    assert authority.require_research_launch_recipe(ref(scope.RECIPE_PATH), source_repo=scope.REPO,
        source_commit=scope.BASE_COMMIT)['run_scope'] == scope.SCOPE
    with pytest.raises(RuntimeError):
        authority.require_research_launch_recipe(ref(scope.ROOT/'OTHER.json'), source_repo=scope.REPO,
            source_commit=scope.BASE_COMMIT)
    assert len(calls) == 1


@pytest.mark.parametrize('key,value', [('optimizer_steps_authorized',1537), ('model_forwards_authorized',4097),
    ('selected_score_rows',4097), ('training_enabled',False), ('candidate_selection',True),
    ('model_constructions_authorized',True), ('canonical_resume_authority',True)])
def test_exact_gate_rejects_widening(key, value):
    gate = scope.gate_fields()
    scope.require_gate(gate, dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE))
    gate[key] = value
    with pytest.raises(RuntimeError):
        scope.require_gate(gate, dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE))


def test_full_window_and_power_profile_remain_existing_native_limits(monkeypatch):
    monkeypatch.setattr(campaign, 'require_binding', lambda v, **kw: v)
    value = dict(schema_version=authority.WINDOW_SCHEMA, recipe=ref(scope.RECIPE_PATH), invocation_number=1,
        max_invocation_seconds=12000, progress_path=str(scope.RUNTIME_ROOT/'progress/one.json'),
        campaign_cursor_path=str(scope.RUNTIME_ROOT/'research-cursor/one.json'),
        output_directory=str(scope.RUNTIME_ROOT/'private-research'), test_data_used=False)
    seal(value,'policy_sha256')
    assert authority.require_research_window_policy(value,verify_files=False) == value
    for limit in (3600, 12001):
        changed=deepcopy(value); changed['max_invocation_seconds']=limit; seal(changed,'policy_sha256')
        with pytest.raises(RuntimeError): authority.require_research_window_policy(changed,verify_files=False)
    assert authority.guarded_research_power_limit(dict(run_scope=scope.SCOPE,fold_id=1,
        training_enabled=True,test_data_used=False,numerical_recipe=dict(batch_size=16))) == 250


def test_metadata_recipe_rejects_resealed_budget_or_panel_changes(monkeypatch):
    expected = dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE, numerical_recipe={'batch_size':16},
        input_refs={}, panel_batch_indices=scope.TRAIN_BATCH_INDICES, **scope.BUDGETS)
    monkeypatch.setattr(scope,'_expected',lambda: (deepcopy(expected),dict(source_owners={})))
    monkeypatch.setattr(scope,'read_origin',lambda v: {})
    monkeypatch.setattr(campaign,'require_binding',lambda v,**kw: v)
    monkeypatch.setattr(scope.subprocess,'run',lambda *a,**kw: SimpleNamespace(returncode=0))
    owners={name:ref(scope.REPO/name) for name in scope.EXTRA_OWNERS}
    value=dict(expected,source_owners=owners,trace_worker=owners[scope.WORKER],mechanical_source_commit=scope.BASE_COMMIT)
    seal(value,'recipe_sha256')
    assert scope.require_recipe(value,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)==value
    for key,other in [('max_forwards',4097),('max_checkpoint_loads',2),('panel_batch_indices',list(range(256)))]:
        changed=deepcopy(value);changed[key]=other;seal(changed,'recipe_sha256')
        with pytest.raises(RuntimeError): scope.require_recipe(changed,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)


def completed_documents():
    value=cursor()
    counts=dict(model_constructions=1,optimizer_constructions=1,checkpoint_loads=1,
        strict_full_state_restore_calls=2,strict_full_state_restore_returns=2,initial_rng_imports=1,
        training_forwards=1536,evaluation_forwards=2560,forwards=4096,forward_returns=4096,
        candidate_evidence_writes=2,checkpoint_writes=0,
        **{k:1468 for k in ('backwards','backward_returns','optimizer_step_calls','optimizer_step_returns','optimizer_updates')})
    arms={arm:dict(batch_visits=768,optimizer_updates=updates,skipped_batches=768-updates,passes_completed=3,
        initial_model_state_sha256=scope.MODEL_SHA256,initial_rng_state_sha256=scope.INITIAL_RNG_SHA256,
        fixed_pass0_query_policy=True,final_model_state_sha256=digest,
        evidence_capsule=ref(scope.ROOT/(arm+'_EVIDENCE.pt')))
        for arm,updates,digest in [('FULL_OBJECTIVE',768,'e'*64),('M5_DIRECTION_ONLY',700,'f'*64)]}
    refs={label:ref(scope.ROOT/(label+'.json')) for label in scope.MEASUREMENT_LABELS}
    captures={label:ref(scope.ROOT/(label+'_READOUT.npz')) for label in scope.MEASUREMENT_LABELS}
    docs={}
    for label in scope.MEASUREMENT_LABELS:
        initial=label.startswith('INITIAL_');panel=label.rsplit('_',1)[1]
        arm=None if initial else next(a for a in scope.ARMS if label.startswith(a+'_'))
        updates=0 if initial else arms[arm]['optimizer_updates'] if '_PASS3_' in label else 256
        docs[refs[label]['path']]=dict(measurement=dict(rows=4096,batches=256,observation_chain_sha256=('1' if panel=='TRAIN' else '2')*64),
            optimizer_updates=updates,evaluated_model_state_sha256=scope.MODEL_SHA256 if initial else arms[arm]['final_model_state_sha256'],
            panel_batch_indices=scope.TRAIN_BATCH_INDICES if panel=='TRAIN' else scope.LATER_BATCH_INDICES,
            evaluation_state_unchanged=True,test_data_used=False)
    report=dict(schema_version=scope.RESULT_SCHEMA,outcome='COMPLETE',recipe=value['recipe'],parent_cursor=24576,
        optimizer_updates=1468,batch_visits=1536,skipped_batches=68,canonical_resume_authority=False,test_data_used=False,
        counts=counts,arms=arms,measurement_refs=refs,panel_captures=captures,initial_restore_exact=True,
        evaluation_state_unchanged=True,source_unchanged=True,same_training_queries=True,
        optimizer_initialization='fresh_adamw_empty_state',initial_model_state_sha256=scope.MODEL_SHA256)
    docs[value['result']['path']]=report
    return value,report,docs


@pytest.mark.parametrize('fault', ['none','wrong_visits','wrong_skips','wrong_initial','wrong_rng',
    'wrong_queries','missing_evidence','missing_panel','wrong_chain','wrong_panel','wrong_final_model',
    'extra_forward','wrong_restore_count','checkpoint_written'])
def test_complete_cursor_binds_all_arms_and_actual_measurements(monkeypatch,fault):
    value,report,docs=completed_documents(); arm=report['arms']['M5_DIRECTION_ONLY']
    later=docs[report['measurement_refs']['M5_DIRECTION_ONLY_PASS3_LATER']['path']]
    if fault=='wrong_visits':arm['batch_visits']=767
    if fault=='wrong_skips':report['skipped_batches']=0
    if fault=='wrong_initial':arm['initial_model_state_sha256']='0'*64
    if fault=='wrong_rng':arm['initial_rng_state_sha256']='0'*64
    if fault=='wrong_queries':arm['fixed_pass0_query_policy']=False
    if fault=='missing_evidence':arm['evidence_capsule']=ref(scope.ROOT/'OTHER.pt')
    if fault=='missing_panel':report['panel_captures'].pop('INITIAL_LATER')
    if fault=='wrong_chain':later['measurement']['observation_chain_sha256']='0'*64
    if fault=='wrong_panel':later['panel_batch_indices']=scope.TRAIN_BATCH_INDICES
    if fault=='wrong_final_model':later['evaluated_model_state_sha256']=scope.MODEL_SHA256
    if fault=='extra_forward':report['counts']['forwards']+=1
    if fault=='wrong_restore_count':report['counts']['strict_full_state_restore_returns']=1
    if fault=='checkpoint_written':report['counts']['checkpoint_writes']=1
    monkeypatch.setattr(campaign,'require_binding',lambda v,**kw: v)
    monkeypatch.setattr(scope,'_read',lambda r: docs[r['path']])
    if fault=='none':assert scope.require_cursor(value,expected_recipe=value['recipe'])==value
    else:
        with pytest.raises(RuntimeError):scope.require_cursor(value,expected_recipe=value['recipe'])


def test_native_dispatch_checks_guard_before_exact_worker(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.audit import run_market_state_ta_m5_objective_benchmark_v1 as worker
    calls=[]
    monkeypatch.setattr(cap,'require_guarded_cuda_trainer_execution',lambda: calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',
        lambda *a: ({},{},{},{},dict(run_scope=scope.SCOPE)))
    monkeypatch.setattr(worker,'run_objective_benchmark',lambda **kw:calls.append('worker') or kw)
    kwargs=dict(policy_path=scope.ROOT/'WINDOW.json',policy_file_sha256='a'*64,progress_path=scope.ROOT/'PROGRESS.json')
    assert runner.run_native_direction_research_window(**kwargs)==kwargs
    assert calls==['guard','worker']
    def reject(): raise RuntimeError('guard blocked')
    monkeypatch.setattr(cap,'require_guarded_cuda_trainer_execution',reject)
    with pytest.raises(RuntimeError,match='guard blocked'):runner.run_native_direction_research_window(**kwargs)
    assert calls==['guard','worker']


def test_materializer_exact_path_reaches_existing_cap_before_preparation(monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    binding_path=scope.ROOT/'PREPARATION_EXECUTION_BINDING.json'
    policy_path=scope.REPO/'NEXT_RUN_POLICY.json'
    binding=dict(materializer=ref(scope.REPO/'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'),
        recipe=ref(scope.RECIPE_PATH),policy=ref(policy_path),prepared_boot=ref(scope.ROOT/'PREPARED_WINDOWS_BOOT.json'),
        guard_template=ref(scope.ROOT/'GUARD_TEMPLATE.json'),source_head=scope.BASE_COMMIT)
    gate=dict(campaign_preparation_authorized=True,recipe=binding['recipe'],prepared_boot=binding['prepared_boot'],
        guard_template=binding['guard_template'],materializer=binding['materializer'],run_root=str(scope.RUNTIME_ROOT),
        claimed_invocations=0,completed_invocations=0,max_invocations=1,relaunch_allowed=False,test_data_used=False)
    monkeypatch.setattr(materializer,'_ref',ref)
    monkeypatch.setattr(campaign,'file_sha256',lambda p:'a'*64)
    monkeypatch.setattr(campaign,'read_bound_json',lambda p,h:binding if p==binding_path else {scope.STAGE:gate})
    monkeypatch.setattr(materializer.subprocess,'check_output',
        lambda argv,**kw:scope.BASE_COMMIT+'\n' if argv[1]=='rev-parse' else '')
    def stop():raise RuntimeError('existing cap before claim')
    monkeypatch.setattr(materializer,'require_capped_cpu_audit_execution',stop)
    with pytest.raises(RuntimeError,match='existing cap before claim'):materializer.materialize(binding_path)
    with pytest.raises(RuntimeError):materializer.materialize(scope.ROOT/'WRONG_BINDING.json')
