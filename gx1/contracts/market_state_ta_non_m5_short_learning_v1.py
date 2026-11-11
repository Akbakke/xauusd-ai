"""One matched-budget independent-horizon fit, with full used-prefix measurements."""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import subprocess

from gx1.audit import cache_market_state_ta_direction_v1 as cache
from gx1.audit import observe_market_state_ta_non_m5_initial_v1 as initial
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_m5_short_learning_v1 as original_short

REPO=original_short.REPO
legacy=original_short.legacy
STAGE='market_state_ta_non_m5_short_learning_20261112_001'
SCOPE='market_state_ta_non_m5_short_learning_v1'
ROOT=Path('/home/andre2/GX1_RUNS')/STAGE.upper()
RUNTIME_ROOT=ROOT/'RUNTIME'
RECIPE_PATH=ROOT/'RECIPE.json'
RECIPE_SCHEMA='gx1_market_state_ta_non_m5_short_learning_recipe_v1'
RESULT_SCHEMA='gx1_market_state_ta_non_m5_short_learning_result_v1'
CURSOR_SCHEMA='gx1_market_state_ta_non_m5_short_learning_cursor_v1'
BASE_COMMIT='5e2f17a749ea5eb6dd3d1b4bafe4e184dd561b45'
WORKER=original_short.WORKER
MODEL_SHA256='2a3847d0adf85d8aade4a2ceed31cab39188ba99f14cb09773a6a79e873e0e1a'
INITIAL_RNG_SHA256=initial.previous.INITIAL_RNG_SHA256
FINAL_RNG_SHA256=initial.previous.FINAL_RNG_SHA256
PANEL_BATCH_INDICES=list(range(256))
EXTRA_OWNERS=initial.EXTRA_OWNERS|{'gx1/contracts/market_state_ta_non_m5_short_learning_v1.py'}
NUMERICAL_CHANGES=initial.NUMERICAL_CHANGES|{'gx1/contracts/market_state_time_query_fold1_fit_v1.py'}
INPUTS={**initial.INPUTS,
    'initial_parity_plan':{'path':str(initial.ROOT/'PLAN.json'),'sha256':'e828f7dda9c24c4d230c56059589f164d9b6d31caf920380e83fa2856005b0ef'},
    'initial_parity_result':{'path':str(initial.ROOT/'RESULT.json'),'sha256':'a4a248d5228839af18ffb73c09871041edddda6af8aa07671448f397922748ed'},
    'initial_parity_terminal':{'path':str(initial.ROOT/'TERMINAL.json'),'sha256':'d5c1223df034a2014f4e2e5116214a41d969ffa5b472a8b3ad4e128e6ac36020'},
    'initial_parity_independent':{'path':str(initial.ROOT/'INDEPENDENT_INITIAL_PARITY_REVIEW.json'),'sha256':'a6b5363586f87fa2c74664ec5e206abfb7a8caadd5575e308e13bdf3db3d9aa6'},
    'full_prefix_result':{'path':'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001/RESULT.json','sha256':'23afd4e6807661854b938634275994a433448a0df0fc3e7131dcd517061dcfe2'},
    'full_prefix_capture':{'path':'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001/TRAIN_FUSION.npz','sha256':'9fbfbc662a4a3e730afcdc0ecf29df6d5ae455380b9958637708743bc899e5fe'},
}


def _need(ok,reason):
    if not ok:raise RuntimeError('TA_NON_M5_SHORT_'+reason)


def _read(ref):return cache._read(ref)
def _ref(name):return cache._ref(REPO/name)


def current_source_bridge(original,current):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    _need(set(original)==fit.required_source_paths(ta_sf=True) and len(original)==85
          and set(current)==set(original)|EXTRA_OWNERS,'OWNER_SET')
    changed=set();result={}
    for name,old in original.items():
        _need(old['path']==str(REPO/name) and current[name]['path']==str(REPO/name),'OWNER_PATH')
        if not name.startswith('.venv/'):
            raw=subprocess.check_output(['git','show',cache.REVERSED_SOURCE+':'+name],cwd=REPO)
            _need(hashlib.sha256(raw).hexdigest()==old['sha256'],'HISTORICAL_SOURCE:'+name)
        campaign.require_binding(current[name],label='nonM5 learning source '+name,verify_file=True)
        if current[name]!=old:changed.add(name)
        _need(name in original_short.ORCHESTRATION|NUMERICAL_CHANGES or current[name]==old,
              'UNREVIEWED_SOURCE:'+name)
        result[name]=current[name]['sha256']
    _need(changed-original_short.ORCHESTRATION==NUMERICAL_CHANGES,'EXACT_NUMERICAL_CHANGE_SET')
    for name in EXTRA_OWNERS:
        _need(current[name]['path']==str(REPO/name),'EXTRA_OWNER_PATH')
        campaign.require_binding(current[name],label='nonM5 learning helper '+name,verify_file=True)
    return result


def require_view_source_transition(historical,current,plan):
    initial.require_view_source_transition(historical,current,
        {'non_m5_initial_source_owners':plan['non_m5_learning_source_owners']})


def require_unchanged_numerics(trial,original):
    identities={'research_objective_sha256','native_F_loss_contract_sha256','contract_sha256'}
    _need(set(trial)==set(original) and all(trial[k]==v for k,v in original.items() if k not in identities)
          and all(trial[k]!=original[k] for k in identities),'UNCHANGED_NUMERICAL_SETTINGS')


def read_origin(recipe):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    _need(recipe['candidate_inputs']==INPUTS,'EXACT_INPUTS')
    for key,ref in INPUTS.items():campaign.require_binding(ref,label='nonM5 input '+key,verify_file=True)
    context=dict(inputs=INPUTS,non_m5_learning_scope=SCOPE,non_m5_learning_source_owners=recipe['source_owners'])
    q=cache._qualify_reversed_origin(context,[])
    source_view=q['view'];view=zero._bind_ta_view(q['receipts'],dedicated_m5_side=True,dedicated_non_m5_side=True)
    initial._require_trial_view(source_view,view)
    original,result,measurement=(_read(INPUTS[k]) for k in ('short_recipe','short_result','short_initial_measurement'))
    initial.previous._qualify_short(original,result,measurement,original)
    plan,proof,terminal,peer=(_read(INPUTS[k]) for k in ('initial_parity_plan','initial_parity_result','initial_parity_terminal','initial_parity_independent'))
    _need(proof['status']==terminal['status']=='COMPLETE' and terminal['exit_code']==0
        and terminal['result']==INPUTS['initial_parity_result'] and terminal['plan']==INPUTS['initial_parity_plan']
        and proof['source_unchanged'] is True and all(v is True for v in proof['passivity'].values())
        and proof['counts']['backbone_encodings']==32 and proof['counts']['optimizer_updates']==0
        and proof['phase_states']['CURRENT_SHARED_INITIAL']['model']==original_short.MODEL_SHA256
        and proof['phase_states']['NON_M5_SIDE_INITIAL']['model']==MODEL_SHA256
        and plan['source_view_sha256']==source_view['contract_sha256']==proof['source_view_sha256']
        and plan['trial_view_sha256']==view['contract_sha256']==proof['trial_view_sha256']
        and peer['status']=='PASS_INDEPENDENT_RETAINED_INITIAL_PARITY'
        and peer['inputs']['result']==INPUTS['initial_parity_result'] and peer['max_loss_difference']==0.0,
        'MEASURED_INITIAL_ORIGIN')
    full=_read(INPUTS['full_prefix_result'])
    _need(full['status']=='COMPLETE' and full['source_unchanged'] is True
        and full['artifacts']['paired_used_prefix']==INPUTS['full_prefix_capture']
        and full['candidate_model_state_sha256']==original_short.MODEL_SHA256
        and full['batch_indices']==PANEL_BATCH_INDICES and full['evaluation_state_unchanged'] is True,
        'FULL_PREFIX_REFERENCE')
    require_unchanged_numerics(recipe['numerical_recipe'],original['numerical_recipe'])
    _need(recipe['expected_view_sha256']==view['contract_sha256'],'TRIAL_VIEW')
    return dict(q,source_view=source_view,view=view,original_recipe=q['origin']['original_recipe'],
        original_result=q['origin']['original_result'],short_recipe=original,initial_proof=proof,
        full_prefix_result=full)


def _expected():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    value,original=original_short._expected()
    numerical=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True,dedicated_non_m5_side=True)
    require_unchanged_numerics(numerical,value['numerical_recipe'])
    refs=dict(value['input_refs']);refs.update({k:v for k,v in INPUTS.items() if k not in cache.FUSION_REFS})
    proof=_read(INPUTS['initial_parity_plan'])
    value.update(schema_version=RECIPE_SCHEMA,runtime_root=str(RUNTIME_ROOT),admission_stage=STAGE,run_scope=SCOPE,
        execution_binding_path=str(RUNTIME_ROOT/'EXECUTION_BINDING.json'),input_refs=refs,candidate_inputs=INPUTS,
        numerical_recipe=numerical,expected_view_sha256=proof['trial_view_sha256'],
        expected_model_contract_sha256=numerical['research_objective_sha256'],
        expected_loss_contract_sha256=numerical['native_F_loss_contract_sha256'],expected_initial_model_state_sha256=MODEL_SHA256,
        panel_batch_indices=PANEL_BATCH_INDICES,max_forwards=768,max_evaluation_forwards=512,max_model_constructions=2,
        initial_evaluation_forwards=256,initial_rng_state_sha256=INITIAL_RNG_SHA256,
        terminal_rng_state_sha256=FINAL_RNG_SHA256,max_initial_rng_imports=1,dedicated_non_m5_side=True)
    return value,original


def build_recipe(*,source_commit):
    value,_=_expected();reverse=_read(cache.FUSION_REFS['reverse_recipe'])
    value.update(source_owners={name:_ref(name) for name in set(reverse['source_owners'])|EXTRA_OWNERS},
                 trace_worker=_ref(WORKER),mechanical_source_commit=source_commit)
    value['recipe_sha256']=campaign.canonical_sha256(value)
    return require_recipe(value,source_repo=REPO,source_commit=source_commit)


def require_recipe(value,*,source_repo,source_commit,verify_files=True):
    expected,_=_expected()
    _need(verify_files and source_repo==REPO and isinstance(value,Mapping)
        and set(value)==set(expected)|{'source_owners','trace_worker','mechanical_source_commit','recipe_sha256'},'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k:value[k] for k in expected})==campaign.canonical_sha256(expected)
        and value['recipe_sha256']==campaign.canonical_sha256({k:v for k,v in value.items() if k!='recipe_sha256'}),'RECIPE_SCOPE')
    for revision in (value['mechanical_source_commit'],source_commit):
        _need(isinstance(revision,str) and campaign._COMMIT.fullmatch(revision)
            and subprocess.run(['git','merge-base','--is-ancestor',BASE_COMMIT,revision],cwd=REPO,capture_output=True).returncode==0,
            'SOURCE_ANCESTRY')
    _need(value['trace_worker']==value['source_owners'][WORKER],'WORKER_BINDING')
    read_origin(value)
    for key,ref in value['input_refs'].items():campaign.require_binding(ref,label='nonM5 recipe '+key,verify_file=True)
    return dict(value)


def gate_fields():
    value=original_short.gate_fields();value.pop('ta_m5_c171_short_learning_authorized')
    value.update(ta_non_m5_short_learning_authorized=True,model_constructions_authorized=2,
                 model_forwards_authorized=768,initial_rng_imports_authorized=1)
    return value


def require_gate(gate,recipe):
    _need(isinstance(gate,Mapping) and all(type(gate.get(k)) is type(v) and gate[k]==v for k,v in gate_fields().items())
        and recipe['run_scope']==SCOPE and recipe['admission_stage']==STAGE,'POLICY_GATE')


def require_full_initial_reference(arrays,losses,counters,*,np):
    with np.load(INPUTS['full_prefix_capture']['path'],allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files}
    counters['archive_decodes']+=1
    initial.previous._require_arrays(reference,8192,np=np,prefix=True)
    expected=[v for v in _read(INPUTS['full_prefix_result'])['batch_records'] if v['phase']=='INITIAL_TRAIN']
    _need([v['batch_index'] for v in expected]==PANEL_BATCH_INDICES and len(losses)==256,'INITIAL_REFERENCE_BATCHES')
    maxima={'state':0.,'future':0.};loss_max=0.
    for i in PANEL_BATCH_INDICES:
        selected=slice(i*16,(i+1)*16)
        parity=initial._require_trial_parity({k:v[selected] for k,v in arrays.items()},
            {k:v[selected] for k,v in reference.items() if k!='phase'},losses[i],expected[i]['loss'],np=np)
        for task in maxima:maxima[task]=max(maxima[task],parity['tasks'][task]['max_abs_non_m5_side'])
        loss_max=max(loss_max,abs(parity['loss_delta']))
    return dict(reference=INPUTS['full_prefix_capture'],rows=4096,batches=256,axes_targets_m5_activity_bitexact=True,
                max_abs_non_m5_side=maxima,max_abs_loss=loss_max,atol=2e-6,rtol=1e-6,loss_atol=5e-6,
                cloned_sides_within_declared_tolerance=True)


def preflight_read_only(recipe,*,source_commit=None):
    source_commit=source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    checked=require_recipe(recipe,source_repo=REPO,source_commit=source_commit);origin=read_origin(checked)
    _need(origin['view']['contract_sha256']==checked['expected_view_sha256'],'PREFLIGHT_VIEW')
    return dict(status='PASS_READ_ONLY_NON_M5_SHORT_PREFLIGHT',run_scope=SCOPE,model_constructions=0,
        data_payload_reads=0,checkpoint_loads=0,optimizer_updates=0,actual_feature_normalization_identity_checked=True,test_data_used=False)

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
            expected_counts = dict(model_constructions=2,optimizer_constructions=1,forwards=768,backwards=256,
                optimizer_step_calls=256,optimizer_updates=256,checkpoint_loads=2,strict_full_state_restore_calls=1,
                checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2,
                initial_rng_imports=1,head_tensor_imports=16,training_forwards=256,evaluation_forwards=512,
                loss_component_owner_calls=512,archive_decodes=1,extra_target_gathers=1536)
            _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k,v in expected_counts.items())
                  and report['initial_restore_exact'] is report['evaluation_state_unchanged'] is report['source_unchanged'] is True
                  and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
                  and report['initial_model_state_sha256'] == MODEL_SHA256 and report['panel_batch_indices'] == PANEL_BATCH_INDICES
                  and report['initial_state']['path'] == str(RUNTIME_ROOT/'private-research/INITIAL_C171_STATE.pt'),'COMPLETE_COUNTS_AND_STATE')
            proof = report.get('full_initial_reference', {})
            _need(report.get('original_training_rng_endpoints_matched') is True
                and isinstance(proof, Mapping) and set(proof) == {'reference','rows','batches',
                    'axes_targets_m5_activity_bitexact','max_abs_non_m5_side','max_abs_loss',
                    'atol','rtol','loss_atol','cloned_sides_within_declared_tolerance'}
                and proof['reference'] == INPUTS['full_prefix_capture']
                and type(proof['rows']) is int and proof['rows'] == 4096
                and type(proof['batches']) is int and proof['batches'] == 256
                and proof['axes_targets_m5_activity_bitexact'] is True
                and proof['cloned_sides_within_declared_tolerance'] is True
                and proof['atol'] == 2e-6 and proof['rtol'] == 1e-6 and proof['loss_atol'] == 5e-6
                and type(proof['max_abs_loss']) is float and 0 <= proof['max_abs_loss'] <= 5e-6
                and isinstance(proof['max_abs_non_m5_side'], Mapping)
                and set(proof['max_abs_non_m5_side']) == {'state','future'}
                and all(type(x) is float and math.isfinite(x) and x >= 0
                        for x in proof['max_abs_non_m5_side'].values()), 'COMPLETE_MATCHED_INITIAL_AND_RNG')
            measurements = {}
            for label,updates in (('INITIAL_TRAIN',0),('FINAL_TRAIN',256)):
                campaign.require_binding(report['measurement_refs'][label],label=label,verify_file=True)
                measured = _read(report['measurement_refs'][label])
                _need(measured['measurement'] == report['measurements'][label]
                      and measured['optimizer_updates'] == updates
                      and measured['panel_batch_indices'] == PANEL_BATCH_INDICES
                      and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
                      and measured['measurement']['rows'] == 4096 and measured['measurement']['batches'] == 256,
                      'COMPLETE_MEASUREMENT_BINDING')
                measurements[label] = measured
            _need(measurements['INITIAL_TRAIN']['evaluated_model_state_sha256'] == MODEL_SHA256
                  and measurements['FINAL_TRAIN']['evaluated_model_state_sha256'] == report['final_model_state_sha256']
                  and report['final_model_state_sha256'] != MODEL_SHA256
                  and measurements['INITIAL_TRAIN']['measurement']['observation_chain_sha256']
                      == measurements['FINAL_TRAIN']['measurement']['observation_chain_sha256'], 'MATCHED_PANEL_AND_MODELS')
            campaign.require_binding(report['initial_state'],label='fresh short initial',verify_file=True)
    return dict(value)


