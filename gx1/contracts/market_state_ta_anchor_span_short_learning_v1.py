"""One fixed additive-input comparison using the existing guarded short runner.

Metadata preflight performs no model/array decoding. Physical composition and
INITIAL parity run inside the same admitted comparison, before its256updates.
"""
from __future__ import annotations
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO=Path('/home/andre2/src/GX1_CURRENT')
STAGE='market_state_ta_anchor_span_short_learning_20261112_002'
SCOPE='market_state_ta_anchor_span_short_learning_v1'
ROOT=Path('/home/andre2/GX1_RUNS')/STAGE.upper()
RUNTIME_ROOT=ROOT/'RUNTIME'
RECIPE_PATH=ROOT/'RECIPE.json'
RECIPE_SCHEMA='gx1_ta_anchor_span_short_learning_recipe_v1'
RESULT_SCHEMA='gx1_ta_anchor_span_short_learning_result_v1'
CURSOR_SCHEMA='gx1_ta_anchor_span_short_learning_cursor_v1'
WORKER='gx1/audit/trace_market_state_ta_m5_original_cuda_prefix_v1.py'
PARENT_ROOT=ROOT.parent/'MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001'
PARENT_RECIPE=dict(path=str(PARENT_ROOT/'RECIPE.json'),sha256='4ae0082feeb4fb36d943ae1b29efbaa53d2f8876d7f1e91e486b82bb2e232f8d')
PARENT_RESULT=dict(path=str(PARENT_ROOT/'RESULT.json'),sha256='aca65dc41d267a47528c38c036f777a084ed7255b87bf0a6305fa70db18d9901')
PARENT_MODEL_SHA='730409e615aa7aca449015ab33f443b06d3529e1d17408829df8f1d1ae5cf911'
PHYSICAL_RESULT=dict(path=str(ROOT.parent/'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json'),sha256='8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb')
PRIORS=dict(path=str(ROOT.parent/'MARKET_STATE_TA_DEDICATED_M5_SOURCE_20261109_001/ENDPOINT_PRIORS.json'),sha256='9f2cc89f5abc886fa4a89a1c3b0d0f9c39f877d8ab1374af651aee678c7a425c')
EXTRA_OWNERS={'gx1/contracts/entry_anchor_span_feature_view_v1.py','gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py','gx1/contracts/market_state_ta_anchor_span_short_learning_v1.py'}
PANEL_BATCH_INDICES=list(range(256))


def _need(ok,reason):
    if not bool(ok):raise RuntimeError('ANCHOR_SPAN_SHORT_'+reason)


def _read(ref):return campaign.read_bound_json(Path(ref['path']),ref['sha256'])
def _ref(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def prepare_metadata(*, output_root=ROOT):
    """Prepare this single experiment's JSON before claim; never overwrite runs."""
    from gx1.contracts import entry_anchor_span_feature_view_v1 as add
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    output_root=Path(output_root)
    _need(output_root.parent==ROOT.parent and output_root.is_dir()
        and not output_root.is_symlink() and output_root.resolve()==output_root,'METADATA_OUTPUT_ROOT')
    _need(not (output_root/'RESULT.json').exists()
        and not (output_root/'RUNTIME/EXECUTION_BINDING.json').exists(),'UNCLAIMED_PREPARATION')
    old=_read(PARENT_RECIPE)
    parent=_read(add.extension.PARENT_ARTIFACT)
    profile=_read(add.NORMALIZATION_EVIDENCE['units_profile'])
    packet=add.build_anchor_span_prefix_packet(parent,_read(add.NORMALIZATION_EVIDENCE['expanded_base']),unobserved_units_profile=profile)
    constructor=add.build_anchor_span_constructor_metadata(_read(add.PARENT_CONSTRUCTOR_REF))
    def publish(name,value):
        path=output_root/name;path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');return _ref(path)
    packet_ref=publish('PREFIX_PACKET.json',packet);constructor_ref=publish('CONSTRUCTOR_METADATA.json',constructor)
    source=_read(old['input_refs']['original_input_ta_candidate_view'])['source_bindings']
    sources={k:copy.deepcopy(source[k]) for k in views.SOURCE_REF_KEYS|views.TIME_QUERY_V2_SOURCE_REF_KEYS}
    provenance=parent['prefix_qualification']['proof']['source_bindings']
    sources.update(input_qualification=add.PARENT_QUALIFICATION_REF,
        raw_input_provider_plan=provenance['provider_plan'],raw_gc_layout=provenance['boolean_raw_gc_layout'],
        raw_gc_transition=provenance['boolean_raw_domain_transition'],time_query_mature_prefix_plan=provenance['current_plan'],
        anchor_span_inputs_source=_ref(add.OVERLAY_OWNER_PATH))
    # Repository owners identify current code, never relabel the old physical data.
    for key,value in sources.items():
        if value['path'].startswith(str(REPO)+'/gx1/'):
            sources[key]=_ref(value['path'])
    view=views.bind_native_direction_feature_view(prefix_normalization=packet,prefix_normalization_ref=packet_ref,
        constructor_metadata=constructor,constructor_metadata_ref=constructor_ref,source_bindings=sources,
        max_batch_rows=16,unobserved_units_profile=profile,full_feature_profile=True,full_ta_feature_profile=True,
        market_state_time_query_v2=True,dedicated_m5_side=True,dedicated_tf_activity=True)
    return publish('FEATURE_VIEW.json',view)


def _expected():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    old=_read(PARENT_RECIPE);result=_read(PARENT_RESULT)
    value=copy.deepcopy(old);view=_read(_ref(ROOT/'FEATURE_VIEW.json'))
    numerical=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True,dedicated_tf_activity=True)
    identities={'research_objective_sha256','native_F_loss_contract_sha256','contract_sha256'}
    _need(set(numerical)==set(old['numerical_recipe']) and all(numerical[k]==v for k,v in old['numerical_recipe'].items() if k not in identities),'UNCHANGED_NUMERICS')
    physical=_read(PHYSICAL_RESULT)
    inputs=dict(parent_recipe=PARENT_RECIPE,parent_result=PARENT_RESULT,physical_result=PHYSICAL_RESULT,
        physical_plan=physical['plan'],feature_view=_ref(ROOT/'FEATURE_VIEW.json'),
        prefix_packet=view['prefix_normalization']['binding'],constructor=view['constructor_metadata']['binding'],
        short_initial_checkpoint=result['initial_state'],short_initial_capture=result['panel_captures']['INITIAL_TRAIN'],
        short_initial_measurement=result['measurement_refs']['INITIAL_TRAIN'],endpoint_priors=PRIORS,
        source_view=old['input_refs']['original_input_ta_candidate_view'])
    value.update(schema_version=RECIPE_SCHEMA,runtime_root=str(RUNTIME_ROOT),admission_stage=STAGE,run_scope=SCOPE,
        execution_binding_path=str(RUNTIME_ROOT/'EXECUTION_BINDING.json'),input_refs=inputs,candidate_inputs=inputs,
        numerical_recipe=numerical,expected_view_sha256=view['contract_sha256'],
        expected_model_contract_sha256=numerical['research_objective_sha256'],expected_loss_contract_sha256=numerical['native_F_loss_contract_sha256'],
        expected_initial_model_state_sha256=None,max_model_constructions=1,anchor_span_inputs=True,
        original_source_commit=result['source_commit'])
    for key in ('source_owners','trace_worker','mechanical_source_commit','recipe_sha256'):value.pop(key,None)
    return value,old


def build_recipe(*,source_commit):
    value,old=_expected()
    value.update(source_owners={name:_ref(REPO/name) for name in set(old['source_owners'])|EXTRA_OWNERS},
        trace_worker=_ref(REPO/WORKER),mechanical_source_commit=source_commit)
    value['recipe_sha256']=campaign.canonical_sha256(value)
    return require_recipe(value,source_repo=REPO,source_commit=source_commit)


def require_recipe(value,*,source_repo,source_commit,verify_files=True):
    expected,old=_expected()
    _need(verify_files and source_repo==REPO and isinstance(value,Mapping)
        and set(value)==set(expected)|{'source_owners','trace_worker','mechanical_source_commit','recipe_sha256'},'RECIPE_FIELDS')
    _need(all(value[k]==v for k,v in expected.items())
        and value['recipe_sha256']==campaign.canonical_sha256({k:v for k,v in value.items() if k!='recipe_sha256'}),'FIXED_RECIPE')
    _need(set(value['source_owners'])==set(old['source_owners'])|EXTRA_OWNERS,'SOURCE_INVENTORY')
    for name,ref in value['source_owners'].items():
        _need(ref['path']==str(REPO/name),'SOURCE_PATH');campaign.require_binding(ref,label=name,verify_file=True)
    _need(value['trace_worker']==value['source_owners'][WORKER],'WORKER')
    for revision in (source_commit,value['mechanical_source_commit']):
        _need(isinstance(revision,str) and campaign._COMMIT.fullmatch(revision)
            and subprocess.run(['git','merge-base','--is-ancestor','d10249230be448356f284f3de89d08bd2011a305',revision],cwd=REPO,capture_output=True).returncode==0,'SOURCE_ANCESTRY')
    read_origin(value)
    return dict(value)


def read_origin(recipe):
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_anchor_span_feature_view_v1 as add
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    refs=recipe['input_refs'];old=_read(PARENT_RECIPE);result=_read(PARENT_RESULT)
    packet=_read(refs['prefix_packet']);constructor=_read(refs['constructor']);view=_read(refs['feature_view'])
    add.require_anchor_span_prefix_packet(packet)
    views.bind_native_direction_feature_view(prefix_normalization=packet,prefix_normalization_ref=refs['prefix_packet'],
        constructor_metadata=constructor,constructor_metadata_ref=refs['constructor'],source_bindings=view['source_bindings'],
        max_batch_rows=16,unobserved_units_profile=packet['unobserved_units_profile'],full_feature_profile=True,
        full_ta_feature_profile=True,market_state_time_query_v2=True,dedicated_m5_side=True,dedicated_tf_activity=True,expected=view)
    _need(result['outcome']=='COMPLETE' and result['source_unchanged'] is True
        and result['initial_model_state_sha256']==PARENT_MODEL_SHA and result['optimizer_updates']==256,'PARENT_COMPLETED')
    source_view=_read(refs['source_view'])
    fields=views._canonical(ta_only=True)[0]
    spans=set(add.extension.SPAN_FIELDS)
    old_fields={k:[n for n in names if n.removeprefix('chart.') not in spans] for k,names in fields.items()}
    _need({k:views._sha(v) for k,v in old_fields.items()}==source_view['physical_field_names_sha256'],'EXACT_OLD_TA_NAMES')
    # Counts and logits were fitted before this experiment and remain exact.
    # Only their input-profile-dependent metadata identity is explicitly rebound.
    original_priors=_read(PRIORS)
    _need(original_priors['fit_rows']==view['prefix']['rows']
        and original_priors['fit_parent_rows_sha256']==view['prefix']['parent_rows_sha256']
        and original_priors['fit_decision_ns_sha256']==view['prefix']['decision_ns_sha256'],'PRIOR_TRAIN_AXES')
    priors=copy.deepcopy(original_priors);priors.pop('contract_sha256')
    priors.update(loss_contract_sha256=objective.contract(ta_only=True,dedicated_m5_side=True)['contract_sha256'],
        source_frozen_prior=dict(binding=PRIORS,contract_sha256=original_priors['contract_sha256'],statistics_refitted=False))
    priors=objective._seal(priors)
    objective.require_endpoint_priors(priors,fold_id=1,ta_only=True,dedicated_m5_side=True)
    _need(priors['tasks']==original_priors['tasks'],'UNCHANGED_PRIOR_VALUES')
    return dict(view=view,source_view=source_view,source_field_names=old_fields,
        receipts=dict(fold1=dict(normal=packet,profile=packet['unobserved_units_profile']),constructor=constructor),
        original_recipe=old,original_result=result,priors=priors)


def require_unchanged_numerics(trial,original):
    ignored={'research_objective_sha256','native_F_loss_contract_sha256','contract_sha256'}
    _need(set(trial)==set(original) and all(trial[k]==v for k,v in original.items() if k not in ignored),'UNCHANGED_NUMERICS')


def gate_fields():
    from gx1.contracts.market_state_ta_tf_activity_short_learning_v1 import gate_fields as parent
    fields=parent();fields.pop('ta_tf_activity_short_learning_authorized')
    return dict(fields,ta_anchor_span_short_learning_authorized=True,model_constructions_authorized=1)


def require_gate(gate,recipe):
    _need(isinstance(gate,Mapping) and all(type(gate.get(k)) is type(v) and gate[k]==v for k,v in gate_fields().items())
        and recipe['run_scope']==SCOPE and recipe['admission_stage']==STAGE,'POLICY_GATE')


def preflight_read_only(recipe,*,source_commit=None):
    source_commit=source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require_recipe(recipe,source_repo=REPO,source_commit=source_commit)
    return dict(status='PASS_METADATA_ONLY_ANCHOR_SPAN_SHORT_PREFLIGHT',model_constructions=0,
        data_payload_reads=0,checkpoint_loads=0,optimizer_updates=0,test_data_used=False,
        actual_initial_parity_still_required_before_updates=True)


def prepare_fold1(receipts,view,*,np,counters):
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs,TIMEFRAMES
    physical=_read(PHYSICAL_RESULT);plan=_read(physical['plan'])
    proof=receipts['fold1']['normal']['parent_prefix_normalization']['prefix_qualification']['proof']
    provider=base._provider_module();provider_plan=_read(proof['source_bindings']['provider_plan'])
    info=provider.validate_metadata(provider_plan);axes,expected=provider.read_axes(provider_plan,info,np)
    groups=provider.map_reuse(info['reuse'],np);source,current=groups['source050'],groups['current051']
    _need(provider.regular(plan['local_clock_path'])==plan['local_clock_stat'],'LOCAL_CLOCK_STAT')
    times=load_m1_feature_surface_times(Path(plan['local_clock_path']),context='ANCHOR_SPAN_TRAIN_CLOCK_ONLY',expected_bar_seconds=300).asi8
    positions=np.searchsorted(times,axes['bar_start_ns']).astype(np.int64)
    _need(np.all(positions<len(times)) and np.array_equal(times[positions],axes['bar_start_ns']),'LOCAL_POSITIONS')
    for array in (*axes.values(),times,positions):array.setflags(write=False)
    seq=dict(signal=source['signal'],times_ns=times,sample_positions=positions,audit=info['audit'])
    payloads={};values={};clocks={}
    for lane in ('local',*TIMEFRAMES):
        ref=physical['surfaces'][lane]['artifact'];campaign.require_binding(ref,label=lane+' span sidecar',verify_file=True)
        with np.load(ref['path'],allow_pickle=False) as z:
            counters['input_sidecar_archive_decodes']+=1
            payloads[lane]={k:z[k] for k in z.files}
        if lane!='local':
            for ref in plan['caches'][lane].values():campaign.require_binding(ref,label=lane+' old cache',verify_file=True)
            values[lane]=np.load(plan['caches'][lane]['feats']['path'],mmap_mode='r',allow_pickle=False)
            clocks[lane]=np.load(plan['caches'][lane]['ts']['path'],mmap_mode='r',allow_pickle=False)
    old=receipts['fold1']['normal']['parent_prefix_normalization']['base_normalization']
    bound=build_anchor_span_inputs(view,entry_sequence_source=seq,snap_x=current['snap'],ctx_cont=current['ctx_cont'],ctx_cat=current['ctx_cat'],
        parent_signal_names=old['surfaces']['signal']['field_names'],parent_mtf_names=old['surfaces']['mtf_m5']['field_names'],
        parent_mtf_values=values,parent_mtf_times=clocks,span_payloads=payloads,physical_populations=plan['populations'],
        unobserved_units_profile=receipts['fold1']['profile'])
    for key in ('parent_rows','bar_start_ns','decision_ns'):
        _need(hashlib.sha256(axes[key].astype('<i8').tobytes()).hexdigest()==view['full_axes'][key+'_sha256'],'FULL_AXES:'+key)
    n=view['prefix']['rows'];rows=axes['parent_rows'][:n].copy();issues=axes['decision_ns'][:n].copy()
    _need(n==29574 and np.all(issues<view['prefix']['cutoff_ns']),'TRAIN_PREFIX')
    prepared=dict(axes=axes,expected_axes=expected,source=source,current=current,cache=bound.mtf_sources,
        gc=dict(surfaces={}),entry_sequence_source=seq,anchor_span_inputs=bound,anchor_span_parent_cache=[*values.values(),*clocks.values()])
    return provider,provider_plan,prepared,rows,issues


def release_inputs(provider,prepared):
    for values in (*prepared['source'].values(),*prepared['current'].values(),*prepared['anchor_span_parent_cache']):provider.flush_pages(values)


def require_full_initial_reference(arrays,losses,counters,*,np):
    parent=_read(PARENT_RESULT);ref=parent['panel_captures']['INITIAL_TRAIN']
    campaign.require_binding(ref,label='actual parent INITIAL',verify_file=True)
    with np.load(ref['path'],allow_pickle=False) as z:reference={k:z[k] for k in z.files}
    counters['archive_decodes']+=1
    _need(set(arrays)==set(reference) and len(losses)==256,'INITIAL_REFERENCE_KEYS')
    maxima={}
    for key,actual in arrays.items():
        expected=reference[key]
        _need(actual.shape==expected.shape and actual.dtype==expected.dtype,'INITIAL_HEADER:'+key)
        if key in ('state_raw','future_raw'):
            _need(np.allclose(actual,expected,atol=2e-6,rtol=1e-6),'INITIAL_LOGITS:'+key)
            maxima[key]=float(np.max(np.abs(actual.astype(np.float64)-expected)))
        else:_need(np.array_equal(actual,expected),'INITIAL_AXES_TARGETS:'+key)
    expected=_read(parent['measurement_refs']['INITIAL_TRAIN'])['measurement']['canonical_batch_losses']
    delta=float(np.max(np.abs(np.asarray(losses)-np.asarray(expected))))
    _need(delta<=5e-6,'INITIAL_LOSS')
    return dict(reference=ref,rows=4096,batches=256,axes_targets_bitexact=True,
        max_abs_logits=maxima,max_abs_loss=delta,atol=2e-6,rtol=1e-6,loss_atol=5e-6)


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
          and value['parent_order_sha256'] in ('db8e77282a7318873530888a2a636a728b1db109405da3f59c16b74aa9f4c3fb','0'*64)
          and value['test_data_used'] is value['canonical_resume_authority'] is False,'CURSOR_SCOPE')
    complete = value['outcome'] == 'COMPLETE'
    _need(value['result'].get('path') == str(ROOT/'RESULT.json'),'RESULT_PATH')
    campaign.require_binding(value['result'],label='short learning result',verify_file=verify_files)
    if complete:
        _need(value['optimizer_updates'] == 256 and value['parent_order_sha256'] == 'db8e77282a7318873530888a2a636a728b1db109405da3f59c16b74aa9f4c3fb'
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
            expected_counts = dict(model_constructions=1,optimizer_constructions=1,forwards=768,backwards=256,
                optimizer_step_calls=256,optimizer_updates=256,checkpoint_loads=2,strict_full_state_restore_calls=1,
                checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2,
                initial_rng_imports=1,head_tensor_imports=0,training_forwards=256,evaluation_forwards=512,
                loss_component_owner_calls=512,archive_decodes=1,input_sidecar_archive_decodes=6,extra_target_gathers=1536)
            _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k,v in expected_counts.items())
                  and report['initial_restore_exact'] is report['evaluation_state_unchanged'] is report['source_unchanged'] is True
                  and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
                  and report['initial_model_state_sha256'] == report['import_proof']['migration']['model_state_sha256'] and report['panel_batch_indices'] == PANEL_BATCH_INDICES
                  and report['initial_state']['path'] == str(RUNTIME_ROOT/'private-research/INITIAL_C171_STATE.pt'),'COMPLETE_COUNTS_AND_STATE')
            proof = report.get('full_initial_reference', {})
            parent = _read(PARENT_RESULT)
            _need(report.get('original_training_rng_endpoints_matched') is True
                and proof.get('reference') == parent['panel_captures']['INITIAL_TRAIN']
                and proof.get('rows') == 4096 and proof.get('batches') == 256
                and proof.get('axes_targets_bitexact') is True
                and proof.get('atol') == 2e-6 and proof.get('rtol') == 1e-6
                and proof.get('loss_atol') == 5e-6 and 0 <= proof.get('max_abs_loss',-1) <= 5e-6
                and report['import_proof']['source_checkpoint'] == parent['initial_state'],
                'COMPLETE_MATCHED_INITIAL_AND_RNG')
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
            _need(measurements['INITIAL_TRAIN']['evaluated_model_state_sha256'] == report['initial_model_state_sha256']
                  and measurements['FINAL_TRAIN']['evaluated_model_state_sha256'] == report['final_model_state_sha256']
                  and report['final_model_state_sha256'] != report['initial_model_state_sha256']
                  and measurements['INITIAL_TRAIN']['measurement']['observation_chain_sha256']
                      == measurements['FINAL_TRAIN']['measurement']['observation_chain_sha256'], 'MATCHED_PANEL_AND_MODELS')
            campaign.require_binding(report['initial_state'],label='fresh short initial',verify_file=True)
    return dict(value)
