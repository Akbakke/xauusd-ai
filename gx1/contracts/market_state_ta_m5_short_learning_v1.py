"""One exact c171 native prefix check, using the existing 256-update owner."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import subprocess

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as legacy

REPO = legacy.CANONICAL_REPO
STAGE = 'market_state_ta_m5_c171_short_learning_20261111_002'
SCOPE = 'market_state_ta_m5_c171_short_learning_v1'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_market_state_ta_m5_c171_short_learning_recipe_v1'
RESULT_SCHEMA = 'gx1_market_state_ta_m5_c171_short_learning_result_v1'
CURSOR_SCHEMA = 'gx1_market_state_ta_m5_c171_short_learning_cursor_v1'
BASE_COMMIT = '2c7a6e789e2d532e67b75af19dd3d2e79d58eed4'
MODEL_SHA256 = 'c171736d2de79396dd86a79ad895f8bf2226e5ad8d996abf1dd7f52055e4a467'
WORKER = legacy.TA_M5_ORIGINAL_CUDA_PREFIX_WORKER
PANEL_BATCH_INDICES = [(i * 1847) // 15 for i in range(16)]
ORCHESTRATION = frozenset({
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'scripts/gx1_capped_run.sh',
})
EXTRA_OWNERS = frozenset({
    WORKER, 'gx1/contracts/market_state_ta_m5_short_learning_v1.py',
    'gx1/audit/cache_market_state_ta_direction_v1.py',
    'gx1/audit/run_market_state_ta_m5_paired_batch_order_v1.py',
})


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_M5_SHORT_LEARNING_' + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _ref(name):
    return {'path': str(REPO/name), 'sha256': campaign.file_sha256(REPO/name)}


def current_source_bridge(original_owners, current_owners):
    """Only named launch plumbing may change; every numerical owner stays exact."""
    from gx1.audit import cache_market_state_ta_direction_v1 as cache
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    _need(set(original_owners) == fit.required_source_paths(ta_sf=True)
          and len(original_owners) == 85
          and set(current_owners) == set(original_owners) | EXTRA_OWNERS, 'OWNER_SET')
    current = {}
    for name, old in original_owners.items():
        _need(old['path'] == str(REPO/name), 'HISTORICAL_PATH')
        if not name.startswith('.venv/'):
            raw = subprocess.check_output(['git','show',cache.REVERSED_SOURCE+':'+name],cwd=REPO)
            _need(hashlib.sha256(raw).hexdigest() == old['sha256'], 'HISTORICAL_SOURCE:'+name)
        new = current_owners[name]
        _need(new['path'] == str(REPO/name), 'CURRENT_PATH')
        campaign.require_binding(new,label='short native owner '+name,verify_file=True)
        _need(name in ORCHESTRATION or new == old, 'NUMERICAL_SOURCE_CHANGED:'+name)
        current[name] = new['sha256']
    for name in EXTRA_OWNERS:
        _need(current_owners[name]['path'] == str(REPO/name), 'EXTRA_OWNER_PATH')
        campaign.require_binding(current_owners[name],label='short native helper '+name,verify_file=True)
    return current


def require_historical_trace(ref):
    from gx1.audit import cache_market_state_ta_direction_v1 as cache
    _need(ref['path'] == str(REPO/WORKER), 'HISTORICAL_TRACE_PATH')
    raw = subprocess.check_output(['git','show',cache.REVERSED_SOURCE+':'+WORKER],cwd=REPO)
    _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'HISTORICAL_TRACE_BYTES')


def read_reversed_upstream(recipe):
    """Consumed R1 is already sealed; verify its old helper through Git, not current bytes."""
    refs = recipe['input_refs']
    original = _read(refs['original_recipe'])
    result = _read(refs['original_result'])
    reference = _read(refs['source_reference_recipe'])
    _need(refs['original_recipe'] == legacy.TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_recipe']
          and refs['original_result'] == legacy.TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_result']
          and result['source_commit'] == legacy.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE
          and result['outcome'] == result['phase_state'] == 'COMPLETE'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['optimizer_updates'] == 1849 and result['parent_cursor'] == 29574
          and result['skipped_batches'] == 0 and result['recipe'] == refs['original_recipe'], 'UPSTREAM_ORIGIN')
    require_historical_trace(refs['original_trace_helper'])
    return dict(original_recipe=original,original_result=result,source_reference_recipe=reference)


def read_origin(recipe):
    from gx1.audit import cache_market_state_ta_direction_v1 as cache
    refs = recipe['candidate_inputs']
    _need(refs == cache.FUSION_REFS, 'EXACT_CANDIDATE_REFS')
    for key, ref in refs.items():
        campaign.require_binding(ref,label='short candidate '+key,verify_file=True)
    plan = dict(inputs=refs, native_learning_scope=SCOPE,
                native_learning_source_owners=recipe['source_owners'])
    bound_refs = list(refs.values())
    q = cache._qualify_reversed_origin(plan,bound_refs)
    candidate_plan, candidate, review, terminal = (_read(refs[k]) for k in (
        'candidate_plan','candidate_result','candidate_review','candidate_terminal'))
    _need(candidate_plan['inputs']['reverse_recipe'] == refs['reverse_recipe']
          and candidate_plan['inputs']['reversed_evidence'] == refs['reversed_evidence']
          and candidate['status'] == 'COMPLETE' and candidate['source_unchanged'] is True
          and candidate['evaluation_state_unchanged'] is True and candidate['plan'] == refs['candidate_plan']
          and candidate['candidate_model_state_sha256'] == MODEL_SHA256
          and candidate['artifacts']['folded_coefficients'] == refs['folded_coefficients']
          and candidate['artifacts']['fresh_baseline'] == refs['candidate_baseline']
          and candidate['counts']['head_tensor_imports'] == 4 and candidate['counts']['scalar_fits'] == 0
          and review['status'] == 'PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC'
          and review['result'] == refs['candidate_result'] and review['plan'] == refs['candidate_plan']
          and review['actual_row0_outputs_bitexact_cached_heads'] is True
          and review['folded_coefficients_exact_from_original_FP32_and_locked_slopes'] is True
          and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['result'] == refs['candidate_result'] and terminal['plan'] == refs['candidate_plan']
          and terminal['no_relaunch_under_this_plan'] is True
          and all(x['test_data_used'] is False for x in (candidate,review,terminal)), 'CANDIDATE_RECEIPTS')
    return dict(q,plan=plan,bound_refs=bound_refs,
        original_recipe=q['origin']['original_recipe'],original_result=q['origin']['original_result'])


def _expected():
    from gx1.audit import cache_market_state_ta_direction_v1 as cache
    original = _read(legacy.TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_recipe'])
    candidate_refs = dict(cache.FUSION_REFS)
    refs = legacy._ta_m5_original_cuda_prefix_input_refs(original)
    refs.update({'candidate_'+key:dict(ref) for key,ref in candidate_refs.items()})
    value = dict(schema_version=RECIPE_SCHEMA,source_repo=str(REPO),runtime_root=str(RUNTIME_ROOT),
        admission_stage=STAGE,run_scope=SCOPE,execution_binding_path=str(RUNTIME_ROOT/'EXECUTION_BINDING.json'),
        original_source_commit=legacy.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE,
        input_refs=refs,candidate_inputs=candidate_refs,input_manifest=original['input_manifest'],
        numerical_recipe=original['numerical_recipe'],
        expected_view_sha256=original['expected_view_sha256'],
        expected_model_contract_sha256=original['expected_model_contract_sha256'],
        expected_loss_contract_sha256=original['expected_loss_contract_sha256'],
        expected_initial_model_state_sha256=MODEL_SHA256,parent_order_sha256=legacy.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        fold_id=1,pass_index=0,parent_order_rows=29574,selected_fit_rows=4096,
        parent_selection='original_chronological_fit_prefix_0_4096',panel_batch_indices=list(PANEL_BATCH_INDICES),
        optimizer_initialization='fresh_adamw_empty_state',
        device='cuda',max_private_updates=256,max_forwards=288,max_training_forwards=256,
        max_evaluation_forwards=32,max_backwards=256,max_model_constructions=1,max_optimizer_constructions=1,
        max_checkpoint_loads=2,max_full_state_restores=1,max_checkpoint_writes=2,
        max_model_seconds=1800,max_trace_bytes=67108864,
        max_checkpoint_bytes=legacy.MAX_CHECKPOINT_BYTES,max_published_json_bytes=legacy.MAX_PUBLISHED_JSON_BYTES,
        first_step_exact_required=False,final_full_state_exact_required=False,
        fresh_initial_restore_exact_required=True,initial_evaluation_forwards=16,later_evaluation_forwards=0,
        training_enabled=True,full_epoch_training_allowed=False,canonical_resume_authority=False,
        candidate_selection=False,test_data_used=False,predictive_value_claimed=False)
    return value,original


def build_recipe(*,source_commit):
    from gx1.audit import cache_market_state_ta_direction_v1 as cache
    value,original = _expected()
    reverse = _read(cache.FUSION_REFS['reverse_recipe'])
    value.update(source_owners={name:_ref(name) for name in set(reverse['source_owners']) | EXTRA_OWNERS},
        trace_worker=_ref(WORKER),mechanical_source_commit=source_commit)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value,source_repo=REPO,source_commit=source_commit)


def require_recipe(value,*,source_repo,source_commit,verify_files=True):
    expected,_ = _expected()
    _need(verify_files and source_repo == REPO and isinstance(value,Mapping)
          and set(value) == set(expected) | {'source_owners','trace_worker','mechanical_source_commit','recipe_sha256'},'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k:value[k] for k in expected}) == campaign.canonical_sha256(expected)
          and value['recipe_sha256'] == campaign.canonical_sha256({k:v for k,v in value.items() if k!='recipe_sha256'}), 'RECIPE_SCOPE')
    for revision in (value['mechanical_source_commit'],source_commit):
        _need(isinstance(revision,str) and campaign._COMMIT.fullmatch(revision)
              and subprocess.run(['git','merge-base','--is-ancestor',BASE_COMMIT,revision],cwd=REPO,capture_output=True).returncode == 0,'SOURCE_ANCESTRY')
    _need(value['trace_worker'] == value['source_owners'][WORKER], 'WORKER_BINDING')
    read_origin(value)
    for key,ref in value['input_refs'].items():
        campaign.require_binding(ref,label='short input '+key,verify_file=True)
    return dict(value)


def gate_fields():
    return dict(decision='ADMITTED',status='ADMITTED',ta_m5_c171_short_learning_authorized=True,
        fold_id=1,selected_fit_rows=4096,model_constructions_authorized=1,optimizer_constructions_authorized=1,
        model_forwards_authorized=288,backward_calls_authorized=256,optimizer_steps_authorized=256,
        training_enabled=True,native_training_enabled=True,genuine_updates_authorized=True,
        genuine_backward_authorized=True,genuine_predictor_fit_authorized=False,canonical_resume_authority=False,
        candidate_selection=False,test_data_used=False)


def require_gate(gate,recipe):
    _need(isinstance(gate,Mapping) and all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in gate_fields().items())
          and recipe['run_scope'] == SCOPE and recipe['admission_stage'] == STAGE, 'POLICY_GATE')


def build_cursor(*,recipe_ref,outcome,optimizer_updates,parent_cursor,parent_order_sha256,full_state,result,batch_size=16):
    value = dict(schema_version=CURSOR_SCHEMA,recipe=recipe_ref,outcome=outcome,optimizer_updates=optimizer_updates,
        parent_cursor=parent_cursor,parent_order_sha256=parent_order_sha256,full_state=full_state,result=result,
        test_data_used=False,canonical_resume_authority=False)
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_cursor(value,expected_recipe=recipe_ref,verify_files=False,batch_size=batch_size)


def require_cursor(value,*,expected_recipe,verify_files=True,batch_size=16):
    fields = {'schema_version','recipe','outcome','optimizer_updates','parent_cursor','parent_order_sha256',
              'full_state','result','test_data_used','canonical_resume_authority','cursor_sha256'}
    _need(isinstance(value,Mapping) and set(value) == fields and value['schema_version'] == CURSOR_SCHEMA
          and value['cursor_sha256'] == campaign.canonical_sha256({k:v for k,v in value.items() if k!='cursor_sha256'}),'CURSOR_FIELDS')
    _need(batch_size == 16 and expected_recipe.get('path') == str(RECIPE_PATH) and value['recipe'] == expected_recipe
          and value['outcome'] in ('COMPLETE','FAILED') and type(value['optimizer_updates']) is int
          and 0 <= value['optimizer_updates'] <= 256 and type(value['parent_cursor']) is int
          and value['parent_cursor'] == 16*value['optimizer_updates']
          and value['parent_order_sha256'] in (legacy.TA_M5_EQUAL_LOSS_ORDER_SHA256,'0'*64)
          and value['test_data_used'] is value['canonical_resume_authority'] is False,'CURSOR_SCOPE')
    complete = value['outcome'] == 'COMPLETE'
    _need(value['result'].get('path') == str(ROOT/'RESULT.json'),'RESULT_PATH')
    campaign.require_binding(value['result'],label='short learning result',verify_file=verify_files)
    if complete:
        _need(value['optimizer_updates'] == 256 and value['parent_order_sha256'] == legacy.TA_M5_EQUAL_LOSS_ORDER_SHA256
              and value['full_state'].get('path') == str(RUNTIME_ROOT/'private-research/FINAL_C171_SHORT_STATE.pt'),'COMPLETE_PROGRESS')
        campaign.require_binding(value['full_state'],label='short final state',verify_file=verify_files)
    else:
        _need(value['full_state'] is None,'FAILED_NO_FINAL_AUTHORITY')
    if verify_files:
        report = _read(value['result'])
        _need(report['schema_version'] == RESULT_SCHEMA and report['outcome'] == value['outcome']
              and report['recipe'] == expected_recipe and report['parent_cursor'] == value['parent_cursor']
              and report['physical_optimizer_updates'] == value['optimizer_updates']
              and report['parent_order_sha256'] == value['parent_order_sha256'] and report['final_state'] == value['full_state']
              and report['canonical_resume_authority'] is report['test_data_used'] is False,'CURSOR_RESULT')
        if complete:
            expected_counts = dict(model_constructions=1,optimizer_constructions=1,forwards=288,backwards=256,
                optimizer_step_calls=256,optimizer_updates=256,checkpoint_loads=2,strict_full_state_restore_calls=1,
                checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2)
            _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k,v in expected_counts.items())
                  and report['initial_restore_exact'] is report['evaluation_state_unchanged'] is report['source_unchanged'] is True
                  and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
                  and report['initial_model_state_sha256'] == MODEL_SHA256 and report['panel_batch_indices'] == PANEL_BATCH_INDICES
                  and report['initial_state']['path'] == str(RUNTIME_ROOT/'private-research/INITIAL_C171_STATE.pt'),'COMPLETE_COUNTS_AND_STATE')
            measurements = {}
            for label,updates in (('INITIAL_TRAIN',0),('FINAL_TRAIN',256)):
                campaign.require_binding(report['measurement_refs'][label],label=label,verify_file=True)
                measured = _read(report['measurement_refs'][label])
                _need(measured['measurement'] == report['measurements'][label]
                      and measured['optimizer_updates'] == updates
                      and measured['panel_batch_indices'] == PANEL_BATCH_INDICES
                      and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
                      and measured['measurement']['rows'] == 256 and measured['measurement']['batches'] == 16,
                      'COMPLETE_MEASUREMENT_BINDING')
                measurements[label] = measured
            _need(measurements['INITIAL_TRAIN']['evaluated_model_state_sha256'] == MODEL_SHA256
                  and measurements['FINAL_TRAIN']['evaluated_model_state_sha256'] == report['final_model_state_sha256']
                  and report['final_model_state_sha256'] != MODEL_SHA256
                  and measurements['INITIAL_TRAIN']['measurement']['observation_chain_sha256']
                      == measurements['FINAL_TRAIN']['measurement']['observation_chain_sha256'], 'MATCHED_PANEL_AND_MODELS')
            campaign.require_binding(report['initial_state'],label='fresh short initial',verify_file=True)
    return dict(value)


def preflight_read_only(recipe,*,source_commit=None):
    source_commit = source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    checked = require_recipe(recipe,source_repo=REPO,source_commit=source_commit)
    origin = read_origin(checked)
    _need(origin['view']['contract_sha256'] == checked['expected_view_sha256']
          and origin['recipe']['numerical_recipe'] == checked['numerical_recipe'], 'PREFLIGHT_VIEW_NUMERICS')
    return dict(status='PASS_READ_ONLY_NATIVE_SHORT_PREFLIGHT',run_scope=SCOPE,
        model_constructions=0,data_payload_reads=0,checkpoint_loads=0,optimizer_updates=0,
        actual_feature_normalization_identity_checked=True,test_data_used=False)
