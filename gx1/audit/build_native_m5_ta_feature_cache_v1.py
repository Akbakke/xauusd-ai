"""One native M5/closed-MTF six-year physical TA cache; no learning or targets."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import os
import resource
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def _reject_torch(event, args):
    if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
        raise RuntimeError('NATIVE_TA_CACHE_TORCH_FORBIDDEN')


if __name__ == '__main__':
    sys.addaudithook(_reject_torch)
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()

import numpy as np
import pandas as pd
from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as owner
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
from gx1.features.volatility_squeeze_state_v1 import load_volatility_squeeze_artifact_manifest

REPO=owner.REPO;POLICY=REPO/'NEXT_RUN_POLICY.json'
STAGE='native_m5_ta_history_expansion_20261120_003'
PLAN=REPO/'configs/research/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003.json'
ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003')
START_NS=1538352000000000000  # 2018-10-01T00:00:00Z
END_NS=1727740800000000000    # 2024-10-01T00:00:00Z
INPUT_REFS={
    'ta_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/RESULT.json',
        'sha256':'5c015b3af8a17c8a16459be0eeba5ca017b7625db91e193aa84b101e664e7bca'},
    'ta_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/TERMINAL.json',
        'sha256':'849bcda1483e088ca5fa2ec8671e8ed6b003654331a6556c70af53dcfc11c874'},
    'previous_exit_observation':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_001/OPERATOR_EXIT_OBSERVATION.json',
        'sha256':'3a77d7ae4c9169891fcd8d85a6c9874a2ad27d7c3623b50a2153970b3fae1060'},
    'previous_import_failure':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_002/OPERATOR_FAILURE_BOOKEND.json',
        'sha256':'25b3d3b99bef9ad5f6234f0976bf802a14e05565f2acfeb24841c43beda16957'}}
EXECUTION={'class':'producer','device':'cpu','memory':'12G','swap':'512M','tasks_max':64,
    'numerical_threads':1,'internal_seconds':3570,'external_wall_timeout_enforced':False}
COUNTS={'model_constructions':0,'model_forwards':0,'autograd_calls':0,'optimizer_steps':0,
    'normalization_fits':0,'parameter_fits':0,'target_values_decoded':0}
SOURCE_PATHS=tuple(dict.fromkeys((*owner.SOURCE_PATHS,
    'gx1/audit/build_native_m5_ta_feature_cache_v1.py','tests/test_build_native_m5_ta_feature_cache_v1.py')))
sha,ref,read=owner.sha,owner.ref,owner.read


def need(ok, reason):
    if not bool(ok):raise ValueError('NATIVE_TA_CACHE_'+reason)


def no_torch():
    need(not any(name=='torch' or name.startswith('torch.')for name in sys.modules),'TORCH_IMPORTED')


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def publish_artifact(name,value):
    raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    path=ROOT/name;_publish_bytes(path,raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


_ALIAS_HELPER_SOURCE = """def _native_m5_alias_vector(m5,per_bar):
    return m5.attrs['feats_np'][:,m5.columns.get_loc(per_bar)].astype(np.float32,copy=False)
"""
_ALIAS_OLD_EXPRESSION='m5[per_bar].to_numpy(np.float32,copy=False)'
_ALIAS_NEW_EXPRESSION='_native_m5_alias_vector(m5,per_bar)'
_ALIAS_EXCLUDED=('native_m5_feature_provider_contract','fit_native_m5_feature_normalization')


def alias_layout_numeric_identity(source):
    """Normalize only the proven Series-to-array alias access replacement."""
    tree=ast.parse(source);helper=ast.parse(_ALIAS_HELPER_SOURCE).body[0]
    helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name==helper.name]
    need(len(helpers)==1 and ast.dump(helpers[0])==ast.dump(helper),'EXACT_ALIAS_HELPER_AST')
    calls=[node for node in ast.walk(tree)if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name)and node.func.id==helper.name]
    expected=ast.dump(ast.parse(_ALIAS_NEW_EXPRESSION,mode='eval').body)
    need(len(calls)==1 and ast.dump(calls[0])==expected,'EXACT_SINGLE_ALIAS_CALL')
    builders=[node for node in tree.body if isinstance(node,ast.FunctionDef)
        and node.name=='build_native_m5_feature_surfaces']
    assignments=[node for node in ast.walk(builders[0])if isinstance(node,ast.Assign)
        and len(node.targets)==1 and ast.dump(node.targets[0])==ast.dump(ast.parse('values[local]=None').body[0].targets[0])
        and ast.dump(node.value)==expected]if len(builders)==1 else []
    need(len(assignments)==1,'EXACT_ALIAS_ASSIGNMENT')
    assignments[0].value=ast.parse(_ALIAS_OLD_EXPRESSION,mode='eval').body
    tree.body.remove(helpers[0])
    return owner.numeric_ast_identity(ast.unparse(tree),excluded_functions=_ALIAS_EXCLUDED)


def numeric_owner_identities(old_plan):
    identities={};bridge=None
    relative='gx1/contracts/market_state_native_m5_feature_inputs_v1.py'
    for name in owner.NUMERIC_OWNER_PATHS:
        original=subprocess.check_output(['git','show',old_plan['source_base_commit']+':'+name],cwd=REPO)
        if name in old_plan['source_owners']:
            need(hashlib.sha256(original).hexdigest()==old_plan['source_owners'][name],'ORIGINAL_NUMERIC_SOURCE')
        current=(REPO/name).read_text()
        excluded=_ALIAS_EXCLUDED if name==relative else ()
        before=owner.numeric_ast_identity(original.decode(),excluded_functions=excluded)
        after=alias_layout_numeric_identity(current)if name==relative else owner.numeric_ast_identity(current)
        need(before==after,'NUMERIC_OWNER_CHANGED:'+name)
        identities[name]=after
        if name==relative:
            bridge={'schema_version':'gx1_native_m5_alias_array_access_bridge_v1','source_owner':name,
                'original_sha256':hashlib.sha256(original).hexdigest(),'current_sha256':sha(REPO/name),
                'helper_ast_sha256':hashlib.sha256(ast.dump(ast.parse(_ALIAS_HELPER_SOURCE).body[0]).encode()).hexdigest(),
                'normalized_numeric_ast_sha256':after,'exact_replaced_alias_calls':1,
                'scope':'identical_ordered_float32_column_bytes_without_pandas_Series_attrs_copy'}
    return identities,bridge


RESEARCH_SOURCE_PATH='gx1/scripts/research_ta_campaign_v1.py'
RESEARCH_LAZY_IMPORTS=('gx1.scripts.research_entry_direction_walkforward_v1',
    'gx1.scripts.research_model_free_baselines_v1')


def research_import_numeric_identity(source):
    class ImportOnly(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if node.module in RESEARCH_LAZY_IMPORTS else node
    return hashlib.sha256(ast.dump(ImportOnly().visit(ast.parse(source)),include_attributes=False).encode()).hexdigest()


def research_import_bridge(old_plan):
    original=subprocess.check_output(['git','show',old_plan['source_base_commit']+':'+RESEARCH_SOURCE_PATH],cwd=REPO)
    original_sha=hashlib.sha256(original).hexdigest()
    need(original_sha==old_plan['source_owners'][RESEARCH_SOURCE_PATH],'ORIGINAL_RESEARCH_SOURCE')
    before=research_import_numeric_identity(original.decode())
    after=research_import_numeric_identity((REPO/RESEARCH_SOURCE_PATH).read_text())
    need(before==after,'RESEARCH_NON_IMPORT_AST_CHANGED')
    return {'schema_version':'gx1_native_m5_macro_research_lazy_import_bridge_v1',
        'source_owner':RESEARCH_SOURCE_PATH,'original_sha256':original_sha,'current_sha256':sha(REPO/RESEARCH_SOURCE_PATH),
        'import_only_normalized_ast_sha256':after,'relocated_import_sources':list(RESEARCH_LAZY_IMPORTS),
        'all_other_module_statements_and_macro_function_bodies_unchanged':True}


def previous_import_failure_metadata(inputs):
    observation=read(inputs['previous_import_failure'])
    terminal_ref={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_002/TERMINAL.json',
        'sha256':'949c05791e322f5a7300d8396dc46becf25b91abe68346f0359d54b3ad298a12'}
    need(observation['schema_version']=='gx1_native_ta_cache_import_failure_bookend_v1'
        and observation['stage']=='native_m5_ta_history_expansion_20261120_002'
        and observation['terminal']==terminal_ref and observation['outer_exit_code']==observation['native_exit_code']==1
        and observation['native_status']=='FAILED_CONSUMED'and observation['native_RESULT_present']is False
        and observation['native_terminal_source_unchanged_field_present']is False
        and observation['source_policy_plan_unchanged_at_external_bookend']is True
        and observation['alias_loop_was_passed_before_this_failure']is True
        and observation['cache_artifacts_present']is False and observation['training_started']is False
        and observation['test_data_used']is False and observation['relaunch_allowed']is False
        and observation['model_constructions']==observation['optimizer_steps']==0,'CONSUMED_002_IMPORT_FAILURE')
    terminal=read(terminal_ref);claim=read(observation['claim'])
    need(terminal['stage']==claim['stage']==observation['stage']and terminal['status']=='FAILED_CONSUMED'
        and terminal['exit_code']==1 and terminal['error']=='RuntimeError:NATIVE_TA_CACHE_TORCH_FORBIDDEN'
        and terminal['result']is None and terminal['plan']==claim['plan']==observation['plan']
        and terminal['claim']==observation['claim']and terminal['remaining_invocations']==0
        and terminal['relaunch_allowed']is False and terminal['test_data_used']is False
        and 'source_unchanged'not in terminal,'ORIGINAL_002_NATIVE_FAILURE')
    return {'observation':inputs['previous_import_failure'],'terminal':terminal_ref,'claim':observation['claim'],
        'plan':observation['plan'],'native_status':'FAILED_CONSUMED','native_exit_code':1,'native_RESULT_present':False,
        'native_terminal_source_unchanged_field_present':False,'source_policy_plan_unchanged_at_external_bookend':True,
        'error':terminal['error']}


def previous_exit_metadata(inputs):
    observation=read(inputs['previous_exit_observation'])
    need(observation['schema_version']=='gx1_native_ta_cache_external_exit_observation_v1'
        and observation['stage']=='native_m5_ta_history_expansion_20261120_001'
        and observation['outer_exit_code']==137 and observation['native_RESULT_present']is False
        and observation['native_TERMINAL_present']is False and observation['cache_artifacts_present']is False
        and observation['durable_artifacts_before_observation']==['CLAIM.json']
        and observation['source_policy_plan_unchanged_at_external_bookend']is True
        and observation['scope_directly_recorded_in_claim']is False
        and observation['independent_cache_qualification_passed']is False
        and observation['training_started']is False and observation['test_data_used']is False
        and observation['relaunch_allowed']is False,'CONSUMED_001_EXIT_OBSERVATION')
    return {'observation':inputs['previous_exit_observation'],'plan':observation['plan'],'claim':observation['claim'],
        'outer_exit_code':137,'native_RESULT_present':False,'native_TERMINAL_present':False,
        'source_policy_plan_unchanged_at_external_bookend':True,
        'failure_phase_unknown':observation['failure_phase_unknown'],'attribution_limit':observation['attribution_limit']}


def phase(name):
    return publish_artifact('PHASE_'+name+'.json',{'schema_version':'gx1_native_ta_cache_phase_v1',
        'stage':STAGE,'phase':name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'ru_maxrss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})


def metadata(inputs):
    need(type(inputs)is dict and inputs==INPUT_REFS,'EXACT_QUALIFIED_PARENT')
    previous_exit=previous_exit_metadata(inputs)
    previous_import_failure=previous_import_failure_metadata(inputs)
    result,terminal=(read(inputs[key])for key in ('ta_result','ta_terminal'))
    plan,claim=read(result['plan']),read(result['claim'])
    need(result['status']=='COMPLETE_NATIVE_M5_PIPELINE_TECHNICAL_QUALIFICATION'
        and result['stage']==terminal['stage']==plan['stage']==claim['stage']=='native_m5_pipeline_20261119_006'
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==inputs['ta_result']
        and terminal['plan']==result['plan']==claim['plan']and terminal['claim']==result['claim']
        and result['source_unchanged']is terminal['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False
        and result['optimizer_steps']==0 and result['clean_process_io_firewall']['denied_attempts']==0,'COMPLETE_PARENT')
    snapshot_ref=plan['inputs']['candidate_snapshot'];archive_ref=plan['inputs']['candidate_archive']
    saved=read(snapshot_ref);admission_ref=result['report']['feature_inputs_requalification'];admission=read(admission_ref)
    need(saved['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'and saved['archive']==archive_ref
        and len(saved['arrays'])==27 and saved['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256']is True
        and admission['candidate_snapshot']==snapshot_ref and admission['candidate_archive']==archive_ref
        and admission['all_array_bytes_dtype_shape_exact']is True
        and admission['ordered_fields_aliases_and_formula_lineage_equal']is True,'QUALIFIED_REFERENCE_CACHE')
    spec=plan['feature_source_spec'];old_source=saved['surfaces_metadata']['source_binding']
    contract=features.native_m5_feature_provider_contract()
    need(spec['cutoff_ns']==END_NS and spec==old_source['source_spec']and spec['manifest']==old_source['native_m5_manifest']
        and plan['feature_contract']==contract and old_source['rows']==1091836
        and spec['parts'][0]['path'].endswith('/year=2009/part-000.parquet')and len(spec['parts'])==16,'SAME_FULL_SOURCE_PREFIX')
    parameters,_=owner.parameter_artifact_metadata(plan['inputs'],plan['geometry'])
    need(parameters==plan['parameter_reuse']and saved['surfaces_metadata']['parameter_available_ns']<=START_NS,
        'FROZEN_NATIVE_PARAMETERS')
    macro_dependencies=owner.macro_metadata(plan['inputs']['macro_source_plan'])
    dependencies={**macro_dependencies,**parameters['parameter_dependencies'],
        'qualified_plan':result['plan'],'qualified_claim':result['claim'],'qualified_snapshot':snapshot_ref,
        'qualified_archive':archive_ref,'qualified_admission':admission_ref,
        'registry':parameters['registry_artifact'],'squeeze_manifest':parameters['squeeze_manifest'],
        'native_manifest':spec['manifest'],'macro_source_plan':plan['inputs']['macro_source_plan'],
        'squeeze_fit_plan':plan['inputs']['squeeze_fit_plan'],'registry_fit_plan':plan['inputs']['registry_fit_plan']}
    dependencies.update({'native_part_'+str(i):{'path':p['path'],'sha256':p['sha256']}for i,p in enumerate(spec['parts'])})
    dependencies.update({'previous_import_failure_'+key:previous_import_failure[key]for key in ('plan','claim','terminal')})
    source_metadata={part['path']:owner.data.part_metadata(part)for part in spec['parts']}
    no_torch()
    numeric_owners,alias_bridge=numeric_owner_identities(plan)
    return {'dependencies':dependencies,'feature_source_spec':spec,'feature_source_metadata':source_metadata,
        'parameter_inputs':{key:plan['inputs'][key]for key in ('prefitted_squeeze_manifest','prefitted_registry_constants',
            'squeeze_fit_plan','registry_fit_plan','native_source_plan')},
        'geometry':plan['geometry'],'parameter_reuse':parameters,'feature_contract':contract,
        'expected_full_source_binding':old_source,'reference_snapshot':snapshot_ref,'reference_archive':archive_ref,
        'numeric_owner_identities':numeric_owners,'alias_layout_bridge':alias_bridge,
        'previous_consumed_invocation':previous_exit,'previous_import_failure':previous_import_failure,
        'research_import_bridge':research_import_bridge(plan)}


def build_plan(*,inputs,source_base_commit=None):
    value=metadata(inputs)
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_ta_history_expansion_plan_v1','stage':STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(ROOT),'source_base_commit':head,
        'source_owners':{name:sha(REPO/name)for name in SOURCE_PATHS},'inputs':inputs,'metadata':value,
        'issue_start_ns':START_NS,'issue_end_ns':END_NS,'execution':EXECUTION,'limits':COUNTS,
        'training_authorized':False,'test_data_used':False,'GC_used':False,'M1_used':False,
        'normalization_or_model_state_emitted':False}


def source_clean(plan):
    need(set(plan['source_owners'])==set(SOURCE_PATHS),'SOURCE_SET')
    adapted={**plan,'source_owners':{name:plan['source_owners'][name]for name in owner.SOURCE_PATHS}}
    return owner.source_clean(adapted)and all(sha(REPO/name)==value for name,value in plan['source_owners'].items())


def checked_inputs(plan):
    seen={}
    for binding in [*plan['inputs'].values(),*plan['metadata']['dependencies'].values()]:
        path=binding['path'];need(path not in seen or seen[path]==binding,'SPLIT_INPUT_IDENTITY')
        if path not in seen:need(ref(path)==binding,'INPUT_BOOKEND');seen[path]=binding
    for part in plan['metadata']['feature_source_spec']['parts']:
        need(owner.data.part_metadata(part)==plan['metadata']['feature_source_metadata'][part['path']],'SOURCE_METADATA_BOOKEND')
    no_torch()


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not ROOT.exists()and not ROOT.is_symlink(),'CWD_OUTPUT')
    plan_ref={'path':str(PLAN),'sha256':plan_sha};plan=read(plan_ref)
    need(plan==build_plan(inputs=plan['inputs'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(STAGE,{})
    need(all(policy.get(key)is False for key in owner.data.CLOSED_FLAGS)
        and gate.get('plan')==plan_ref and gate.get('output_root')==str(ROOT)
        and gate.get('execution')==EXECUTION and gate.get('max_invocations')==1
        and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed')is False and gate.get('training_authorized')is False
        and gate.get('test_authorized')is False and all(gate.get(k)==v for k,v in COUNTS.items()),'CLOSED_POLICY')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'ONE_USE')
    checked_inputs(plan);return plan,plan_ref,policy_ref


def load_snapshot(snapshot_ref):
    saved=read(snapshot_ref);archive=saved['archive'];need(ref(archive['path'])==archive,'ARCHIVE_IDENTITY')
    top={'bar_start_ns','retained_source_rows','retained_issue_rows','signal','ctx_cont','ctx_cat','observed'}
    attrs={'ts_int64','feats_np','owner_row_indices','model_native_mtf_scalars_np_v4'}
    expected=top|{'tf_'+tf+'_'+key for tf in features.TIMEFRAMES for key in attrs}
    arrays={}
    with np.load(archive['path'],allow_pickle=False)as value:
        need(set(value.files)==set(saved['arrays'])==expected and len(expected)==27,'ARRAY_INVENTORY')
        for key in sorted(expected):
            array=value[key];header=saved['arrays'][key]
            dtype=np.bool_ if key=='observed'else np.float32 if key in('signal','ctx_cont')or key.endswith(('_feats_np','_scalars_np_v4'))else np.int64
            need(array.dtype==dtype and array.dtype.str==header['dtype']and list(array.shape)==header['shape']
                and digest(array)==header['sha256'],'ARRAY_BYTES:'+key)
            array.flags.writeable=False;arrays[key]=array
    surfaces={**saved['surfaces_metadata'],**{key:arrays[key]for key in top}};frames={}
    need(set(saved['timeframes'])==set(features.TIMEFRAMES),'CLOCK_INVENTORY')
    for tf,descriptor in saved['timeframes'].items():
        need(descriptor['array_attributes']=={key:'tf_'+tf+'_'+key for key in attrs},'TF_ATTRIBUTE_KEYS:'+tf)
        values=arrays['tf_'+tf+'_feats_np'];times=arrays['tf_'+tf+'_ts_int64']
        frame=pd.DataFrame(values,index=pd.to_datetime(times,utc=True),columns=descriptor['columns'],copy=False)
        frame.attrs={**descriptor['scalar_attributes'],**{key:arrays[name]for key,name in descriptor['array_attributes'].items()}}
        frames[tf]=frame
    surfaces['multi_tf_sources']=frames
    need(ref(archive['path'])==archive,'ARCHIVE_BOOKEND')
    return features.NativeM5FeatureProvider(surfaces)


def yearly_readiness(provider):
    rows=provider.surfaces['retained_issue_rows'];times=provider.issue_ns[rows]
    years=pd.to_datetime(times,utc=True).year.to_numpy();reports={}
    fields=provider.profile['physical_field_names']['signal'];macro=provider.profile['macro_context_fields']
    columns=[fields.index('ctx_cont.'+name)for name in macro]
    need(len(macro)==61 and len(provider.profile['local_specialists'])==8,'GENUINE_FAMILIES_MACRO')
    for year in sorted(set(years.tolist())):
        selected=rows[years==year];positions=provider._feature_positions(selected)
        finite=np.isfinite(provider.surfaces['signal'][np.ix_(positions,columns)])
        ready=provider.input_ready[selected];complete=finite.all(1)
        reports[str(year)]={'rows':len(selected),'readiness':provider.readiness_report(selected),
            'first_all_inputs_ready_ns':int(provider.issue_ns[selected[ready][0]])if ready.any()else None,
            'macro_current_all61_observed_rows':int(complete.sum()),
            'first_current_all61_macro_observed_ns':int(provider.issue_ns[selected[complete][0]])if complete.any()else None,
            'macro_missing_current_rows':{name:int((~finite[:,j]).sum())for j,name in enumerate(macro)}}
    return reports


def overlap_parity(provider,snapshot_ref,archive_ref):
    saved=read(snapshot_ref);need(saved['archive']==archive_ref and ref(archive_ref['path'])==archive_ref,'REFERENCE_ARCHIVE')
    need(provider.surfaces['field_lineage']==saved['surfaces_metadata']['field_lineage'],'OVERLAP_FIELD_LINEAGE')
    compared={}
    with np.load(archive_ref['path'],allow_pickle=False)as old:
        def get(name):
            value=old[name];header=saved['arrays'][name]
            need(value.dtype.str==header['dtype']and list(value.shape)==header['shape']and digest(value)==header['sha256'],'REFERENCE_BYTES:'+name)
            return value
        def equal(name,current,previous):
            need(current.dtype==previous.dtype and current.shape==previous.shape and digest(current)==digest(previous),'OVERLAP_EXACT:'+name)
            compared[name]={'shape':list(current.shape),'sha256':digest(current)}
        local=get('retained_source_rows');at=np.searchsorted(provider.retained_source_rows,local)
        need(np.all(at<len(provider.retained_source_rows))and np.array_equal(provider.retained_source_rows[at],local),'OVERLAP_GLOBAL_M5_IDS')
        for key in ('signal','ctx_cont','ctx_cat','observed'):equal(key,provider.surfaces[key][at],get(key))
        equal('retained_source_rows',provider.retained_source_rows[at],local)
        issues=get('retained_issue_rows');present=np.searchsorted(provider.surfaces['retained_issue_rows'],issues)
        equal('retained_issue_rows',provider.surfaces['retained_issue_rows'][present],issues)
        equal('bar_start_ns',provider.bar_start_ns,get('bar_start_ns'))
        need(provider.input_ready[issues].all(),'OVERLAP_PREVIOUSLY_READY')
        for tf in features.TIMEFRAMES:
            frame=provider.surfaces['multi_tf_sources'][tf];ids=get('tf_'+tf+'_owner_row_indices')
            positions=np.searchsorted(frame.attrs['owner_row_indices'],ids)
            need(np.all(positions<len(frame))and np.array_equal(frame.attrs['owner_row_indices'][positions],ids),'OVERLAP_GLOBAL_TF_IDS:'+tf)
            for key in ('owner_row_indices','ts_int64','feats_np','model_native_mtf_scalars_np_v4'):
                equal('tf_'+tf+'_'+key,frame.attrs[key][positions],get('tf_'+tf+'_'+key))
    need(len(compared)==27 and ref(archive_ref['path'])==archive_ref,'OVERLAP_BOOKEND')
    return {'array_count':27,'exact_dtype_shape_bytes_and_global_owner_clocks':True,'arrays':compared,
        'reference_snapshot':snapshot_ref,'reference_archive':archive_ref,'reference_issues':len(issues)}


def produce(plan):
    meta=plan['metadata'];no_torch()
    parameters,constants=owner.parameter_artifact_metadata(meta['parameter_inputs'],meta['geometry'])
    need(parameters==meta['parameter_reuse'],'PARAMETER_BOOKEND')
    squeeze=load_volatility_squeeze_artifact_manifest(Path(parameters['squeeze_manifest']['path']),
        expected_sha256=parameters['squeeze_manifest']['sha256'],clocks=features.TIMEFRAMES)
    phases=[phase('01_BEFORE_RAW_READ')]
    native=owner.read_feature_source(meta)
    phases.append(phase('02_AFTER_RAW_READ'))
    binding=features.bind_native_m5_feature_source(native,native_m5_manifest=meta['feature_source_spec']['manifest'],
        source_spec=meta['feature_source_spec'])
    need(binding==meta['expected_full_source_binding'],'FULL_PREFIX_BYTE_IDENTITY')
    phases.append(phase('03_BEFORE_FEATURE_BUILD'))
    surfaces=features.build_native_m5_feature_surfaces(native,source_binding=binding,v29_registry_constants=constants,
        volatility_squeeze_artifacts=squeeze,issue_start_ns=START_NS,issue_end_ns=END_NS)
    phases.append(phase('04_AFTER_FEATURE_BUILD'))
    phases.append(phase('05_BEFORE_PROVIDER_INIT'))
    provider=features.NativeM5FeatureProvider(surfaces)
    phases.append(phase('06_AFTER_PROVIDER_INIT'))
    del native,surfaces,constants,squeeze
    phases.append(phase('07_BEFORE_SNAPSHOT_PUBLISH'))
    snapshot=owner.publish_feature_snapshot(provider,publish_artifact=publish_artifact)
    phases.append(phase('08_AFTER_SNAPSHOT_PUBLISH'))
    del provider
    import gc
    gc.collect()
    phases.append(phase('09_BEFORE_SNAPSHOT_RELOAD'))
    provider=load_snapshot(snapshot)
    phases.append(phase('10_AFTER_SNAPSHOT_RELOAD'))
    need(provider.surfaces['axes']['issue_start_ns']==START_NS and provider.surfaces['axes']['issue_end_ns']==END_NS
        and provider.surfaces['provider_contract']==meta['feature_contract'],'RELOADED_SCOPE_CONTRACT')
    annual=yearly_readiness(provider);parity=overlap_parity(provider,meta['reference_snapshot'],meta['reference_archive'])
    no_torch()
    return {'snapshot':snapshot,'archive':read(snapshot)['archive'],'source_binding':binding,'phases':phases,
        'source_axes':provider.surfaces['axes'],'yearly_readiness':annual,'overlap_parity':parity,
        'families':list(provider.profile['local_specialists']),'macro_fields':list(provider.profile['macro_context_fields']),
        'rows':len(provider.surfaces['retained_issue_rows']),'retained_local_rows':len(provider.retained_source_rows),
        'primary_matrix_bytes':sum(provider.surfaces[key].nbytes for key in('signal','ctx_cont','ctx_cat','observed'))+
            sum(frame.attrs['feats_np'].nbytes for frame in provider.surfaces['multi_tf_sources'].values()),
        'actual_counts':COUNTS,'Torch_imported':False,'GC_used':False,'M1_used':False,'test_data_used':False}


def run(plan_sha):
    began=time.monotonic();no_torch();firewall=install_native_m5_direction_io_firewall()
    cap=owner.require_capped_cpu_research_producer_execution();lock=owner.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2 and cap['pids']==64
        and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]
        and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'EXACT_CAP')
    plan,plan_ref,policy_ref=inspect(plan_sha,admitted=True);ROOT.mkdir(exist_ok=False)
    claim=publish_artifact('CLAIM.json',{'stage':STAGE,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'capped_execution':cap,'lock_ancestry':lock,'cgroup_path':cap['cgroup_path']})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_TA_CACHE_INTERNAL_3570S')
    try:
        remaining=3570-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_DECODE')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=produce(plan);checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==policy_ref and ref(PLAN)==plan_ref
            and firewall['denied_attempts']==0,'SOURCE_POLICY_FIREWALL_BOOKEND')
        result=publish_artifact('RESULT.json',{'stage':STAGE,'status':'COMPLETE_NATIVE_M5_TA_FEATURE_CACHE',
            'plan':plan_ref,'claim':claim,'report':report,'source_unchanged':True,'actual_counts':COUNTS,
            'clean_process_io_firewall':firewall,'training_authorized':False,'test_data_used':False,
            'elapsed_seconds':time.monotonic()-began})
        terminal=publish_artifact('TERMINAL.json',{'stage':STAGE,'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':True,
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish_artifact('TERMINAL.json',{'stage':STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out-dir',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();no_torch();install_native_m5_direction_io_firewall()
    need(args.plan==str(PLAN)and args.out_dir==str(ROOT),'CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_TA_CACHE_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'cached_values_decoded':0,'Torch_imported':False,**COUNTS}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
